"""Live-oracle parity guards for `ferrolearn.QuadraticDiscriminantAnalysis`
`decision_function` (REQ-7 #581).

Verification model B (goal.md R-CHAR-3): every expected value comes from the
LIVE installed `sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis`
1.5.2 oracle in the SAME process, never copied from ferrolearn. We fit
`ferrolearn` and `sklearn` QDA on the SAME fixture and assert:

- binary `decision_function` shape is `(n_samples,)` (sklearn collapses the
  two-class case to the log-likelihood ratio of the positive class,
  `discriminant_analysis.py:1000-1001` `dec_func[:, 1] - dec_func[:, 0]`),
  and the values match within numerical tolerance,
- the binary SIGN convention: a positive score ⇒ predicted label `classes_[1]`,
- multiclass `decision_function` shape is `(n_samples, n_classes)` and the
  values match,
- `predict` still matches sklearn labels (the new method ADDED nothing to the
  predict path),
- `decision_function`'s argmax/sign agrees with `predict`.

QDA is deterministic (per-class thin SVD, no RNG), so ferrolearn VALUE-matches
sklearn on the well-conditioned path; these guards pin the binding-ABI binary
`(n,)` collapse + the raw multiclass passthrough through the Python layer.
"""

import numpy as np
import pytest
from sklearn.discriminant_analysis import (
    QuadraticDiscriminantAnalysis as SkQDA,
)

import ferrolearn as fl

ATOL = 1e-6


def _binary_fixture():
    rng = np.random.RandomState(0)
    X0 = rng.randn(40, 3) + np.array([0.0, 0.0, 0.0])
    X1 = rng.randn(40, 3) + np.array([3.0, 3.0, 3.0])
    X = np.vstack([X0, X1]).astype(np.float64)
    y = np.concatenate([np.zeros(40), np.ones(40)]).astype(np.int64)
    return X, y


def _multiclass_fixture():
    rng = np.random.RandomState(1)
    X0 = rng.randn(30, 4) + np.array([0.0, 0.0, 0.0, 0.0])
    X1 = rng.randn(30, 4) + np.array([4.0, 0.0, 0.0, 0.0])
    X2 = rng.randn(30, 4) + np.array([0.0, 4.0, 4.0, 0.0])
    X = np.vstack([X0, X1, X2]).astype(np.float64)
    y = np.concatenate([np.zeros(30), np.ones(30), np.full(30, 2)]).astype(np.int64)
    return X, y


def test_binary_decision_function_shape_is_1d():
    X, y = _binary_fixture()
    f = fl.QuadraticDiscriminantAnalysis().fit(X, y)
    s = SkQDA().fit(X, y)
    fd = np.asarray(f.decision_function(X))
    sd = np.asarray(s.decision_function(X))
    assert sd.shape == (X.shape[0],)
    assert fd.shape == sd.shape == (X.shape[0],)


def test_binary_decision_function_values_match_sklearn():
    X, y = _binary_fixture()
    f = fl.QuadraticDiscriminantAnalysis().fit(X, y)
    s = SkQDA().fit(X, y)
    fd = np.asarray(f.decision_function(X))
    sd = np.asarray(s.decision_function(X))
    np.testing.assert_allclose(fd, sd, atol=ATOL)


def test_binary_decision_function_sign_selects_positive_class():
    """A positive decision score ⇒ predicted label classes_[1] (sklearn
    `discriminant_analysis.py:984` log p(y=1|x) - log p(y=0|x))."""
    X, y = _binary_fixture()
    f = fl.QuadraticDiscriminantAnalysis().fit(X, y)
    fd = np.asarray(f.decision_function(X))
    preds = f.predict(X)
    # classes_ is [0, 1]; sign>0 picks classes_[1], sign<0 picks classes_[0].
    expected = np.where(fd > 0, f.classes_[1], f.classes_[0])
    np.testing.assert_array_equal(preds, expected)
    # and cross-check against the live sklearn oracle's sign convention
    s = SkQDA().fit(X, y)
    sd = np.asarray(s.decision_function(X))
    np.testing.assert_array_equal(np.sign(fd), np.sign(sd))


def test_multiclass_decision_function_shape_is_2d():
    X, y = _multiclass_fixture()
    f = fl.QuadraticDiscriminantAnalysis().fit(X, y)
    s = SkQDA().fit(X, y)
    fd = np.asarray(f.decision_function(X))
    sd = np.asarray(s.decision_function(X))
    assert sd.shape == (X.shape[0], 3)
    assert fd.shape == sd.shape == (X.shape[0], 3)


def test_multiclass_decision_function_values_match_sklearn():
    X, y = _multiclass_fixture()
    f = fl.QuadraticDiscriminantAnalysis().fit(X, y)
    s = SkQDA().fit(X, y)
    fd = np.asarray(f.decision_function(X))
    sd = np.asarray(s.decision_function(X))
    np.testing.assert_allclose(fd, sd, atol=ATOL)


def test_multiclass_decision_function_argmax_agrees_with_predict():
    X, y = _multiclass_fixture()
    f = fl.QuadraticDiscriminantAnalysis().fit(X, y)
    fd = np.asarray(f.decision_function(X))
    preds = f.predict(X)
    np.testing.assert_array_equal(f.classes_[np.argmax(fd, axis=1)], preds)


def test_binary_predict_matches_sklearn():
    X, y = _binary_fixture()
    f = fl.QuadraticDiscriminantAnalysis().fit(X, y)
    s = SkQDA().fit(X, y)
    np.testing.assert_array_equal(f.predict(X), s.predict(X))


def test_multiclass_predict_matches_sklearn():
    X, y = _multiclass_fixture()
    f = fl.QuadraticDiscriminantAnalysis().fit(X, y)
    s = SkQDA().fit(X, y)
    np.testing.assert_array_equal(f.predict(X), s.predict(X))


def test_decision_function_before_fit_raises():
    from sklearn.exceptions import NotFittedError

    est = fl.QuadraticDiscriminantAnalysis()
    with pytest.raises(NotFittedError):
        est.decision_function(np.zeros((3, 3)))


# ---------------------------------------------------------------------------
# Pinned divergences (FAILING against the current binding; live sklearn 1.5.2
# oracle is the source of truth). sklearn's QDA `decision_function` routes
# through `_decision_function` -> `self._validate_data(X, reset=False)`
# (discriminant_analysis.py:967), which defaults to `force_all_finite=True`
# and `ensure_min_samples=1`, so non-finite or zero-row X is rejected with a
# ValueError BEFORE any score is computed. ferrolearn's binding marshals X
# straight into the Rust core with no finite/min-sample check, so it returns
# (NaN-laden / empty) scores instead of raising.
# ---------------------------------------------------------------------------


@pytest.mark.skip(reason="divergence: QDA.decision_function accepts NaN/Inf X "
                         "(returns scores) where sklearn raises ValueError; "
                         "tracking #2254")
def test_decision_function_nan_input_raises_like_sklearn():
    """sklearn rejects non-finite X in `decision_function`
    (`discriminant_analysis.py:967` `_validate_data(reset=False)` ->
    `force_all_finite=True`): a NaN cell raises
    ``ValueError("Input X contains NaN.")``. ferrolearn returns a (n,) array
    of NaN instead of raising. Oracle: SkQDA().fit(...).decision_function(Xn)
    raises ValueError; fl returns shape (5,) with np.isnan(...).any() True.
    """
    X, y = _binary_fixture()
    f = fl.QuadraticDiscriminantAnalysis().fit(X, y)
    s = SkQDA().fit(X, y)
    Xn = X[:5].copy()
    Xn[0, 0] = np.nan
    # Live oracle: sklearn raises.
    with pytest.raises(ValueError):
        s.decision_function(Xn)
    # ferrolearn MUST mirror the oracle and raise too.
    with pytest.raises(ValueError):
        f.decision_function(Xn)


@pytest.mark.skip(reason="divergence: QDA.decision_function accepts Inf X "
                         "(returns scores) where sklearn raises ValueError; "
                         "tracking #2254")
def test_decision_function_inf_input_raises_like_sklearn():
    """sklearn rejects +/-Inf X in `decision_function`
    (`discriminant_analysis.py:967` `_validate_data(reset=False)`):
    ``ValueError("Input X contains infinity ...")``. ferrolearn returns a
    non-finite score array instead. Multiclass path verified identically.
    """
    X, y = _multiclass_fixture()
    f = fl.QuadraticDiscriminantAnalysis().fit(X, y)
    s = SkQDA().fit(X, y)
    Xi = X[:5].copy()
    Xi[0, 0] = np.inf
    with pytest.raises(ValueError):
        s.decision_function(Xi)
    with pytest.raises(ValueError):
        f.decision_function(Xi)


@pytest.mark.skip(reason="divergence: QDA.decision_function returns (0,) on "
                         "0-row X where sklearn raises ValueError (min 1 "
                         "sample); tracking #2255")
def test_decision_function_empty_X_raises_like_sklearn():
    """sklearn's `_validate_data` enforces ``ensure_min_samples=1``, so a
    0-row X raises ``ValueError("Found array with 0 sample(s) ... a minimum
    of 1 is required ...")``. ferrolearn returns an empty (0,) array. Oracle:
    SkQDA().fit(...).decision_function(np.zeros((0, n_features))) raises;
    fl returns shape (0,).
    """
    X, y = _binary_fixture()
    f = fl.QuadraticDiscriminantAnalysis().fit(X, y)
    s = SkQDA().fit(X, y)
    Xe = np.zeros((0, X.shape[1]))
    with pytest.raises(ValueError):
        s.decision_function(Xe)
    with pytest.raises(ValueError):
        f.decision_function(Xe)
