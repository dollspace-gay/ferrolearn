"""Divergence pin: ferrolearn.ARDRegression in the n_samples < n_features regime.

scikit-learn's ``ARDRegression.fit`` selects its posterior-covariance update by
the sample/feature ratio (``sklearn/linear_model/_bayes.py:670-674``)::

    update_sigma = (
        self._update_sigma
        if n_samples >= n_features
        else self._update_sigma_woodbury
    )

When ``n_samples < n_features`` sklearn uses ``_update_sigma_woodbury``
(``_bayes.py:732-748``), inverting the well-conditioned ``(n_samples,
n_samples)`` matrix. ferrolearn's Rust core ALWAYS uses the direct
``_update_sigma`` branch (``ferrolearn-linear/src/ard.rs:502``, ``:609``),
inverting the rank-deficient ``(n_features, n_features)`` Gram block. The two
formulas are equal in exact arithmetic but the direct branch accumulates
floating-point error that diverges the EM trajectory in the ``n < p`` regime.

Result: ferrolearn's ``coef_`` (which features survive pruning), ``n_iter_``,
``sigma_`` shape, and ``predict(return_std=True)`` all diverge from sklearn.

Expected values come from the LIVE sklearn 1.5.2 oracle in-test (R-CHAR-3),
never copied from ferrolearn.

Tracking: #2164.
"""

import numpy as np
import pytest

from sklearn.linear_model import ARDRegression as SkARD

try:
    from ferrolearn._extras import ARDRegression as FlARD
except Exception:  # pragma: no cover - extension not built
    FlARD = None


@pytest.mark.skipif(FlARD is None, reason="ferrolearn extension not built")
def test_ard_n_lt_p_matches_sklearn_woodbury():
    # n_samples = 6 < n_features = 25 (the Woodbury regime).
    rng = np.random.RandomState(23)
    X = rng.randn(6, 25)
    y = 2.0 * X[:, 0] + X[:, 5] - X[:, 10] + 0.02 * rng.randn(6)

    sk = SkARD().fit(X, y)  # LIVE oracle (R-CHAR-3)
    fl = FlARD().fit(X, y)

    # The pruned-feature set (and therefore coef_) must match sklearn.
    sk_kept = np.flatnonzero(sk.coef_ != 0.0)
    fl_kept = np.flatnonzero(fl.coef_ != 0.0)
    np.testing.assert_array_equal(
        fl_kept,
        sk_kept,
        err_msg=(
            f"pruned-feature set diverges in n<p regime: "
            f"ferrolearn kept {fl_kept.tolist()}, sklearn kept {sk_kept.tolist()} "
            f"(missing Woodbury branch, #2164)"
        ),
    )

    # n_iter_ must match sklearn's Woodbury trajectory.
    assert fl.n_iter_ == sk.n_iter_, (
        f"n_iter_ diverges in n<p regime: ferrolearn={fl.n_iter_}, "
        f"sklearn={sk.n_iter_} (#2164)"
    )

    # coef_ must match element-wise.
    np.testing.assert_allclose(
        fl.coef_,
        sk.coef_,
        rtol=1e-6,
        atol=1e-7,
        err_msg="coef_ diverges in n<p regime (#2164)",
    )

    # predict(return_std=True) must match (variance over kept columns).
    Xt = rng.randn(4, 25)
    sk_mean, sk_std = sk.predict(Xt, return_std=True)
    fl_mean, fl_std = fl.predict(Xt, return_std=True)
    np.testing.assert_allclose(
        fl_std, sk_std, rtol=1e-5, atol=1e-7,
        err_msg="predict std diverges in n<p regime (#2164)",
    )


@pytest.mark.skipif(FlARD is None, reason="ferrolearn extension not built")
def test_ard_n_lt_p_constant_y_fits_like_sklearn():
    """n<p with constant y: sklearn prunes ALL features and fits cleanly
    (intercept = mean(y)); ferrolearn's direct-inverse branch raises a singular
    -matrix ValueError because the rank-deficient Gram block is not invertible
    (the Woodbury branch inverts ``eye/alpha + ...``, which is never singular).
    sklearn does NOT raise. Tracking: #2164."""
    rng = np.random.RandomState(0)
    X = rng.randn(3, 10)  # n_samples=3 < n_features=10
    y = np.array([5.0, 5.0, 5.0])  # constant target

    sk = SkARD().fit(X, y)  # LIVE oracle: fits without raising
    assert int(np.sum(sk.coef_ != 0)) == 0  # all features pruned
    assert sk.intercept_ == pytest.approx(5.0)

    # ferrolearn must also fit without raising and match sklearn's intercept.
    fl = FlARD().fit(X, y)
    assert fl.intercept_ == pytest.approx(sk.intercept_, rel=1e-7, abs=1e-9)
