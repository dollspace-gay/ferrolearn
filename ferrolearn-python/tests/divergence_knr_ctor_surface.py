"""Divergence pins for the #2147 KNeighborsRegressor full-constructor surface.

Verification model B: compare `import ferrolearn` against `import sklearn` 1.5.2.
Every expected value is regenerated from the live sklearn 1.5.2 oracle inside the
test (R-CHAR-3 — never copied from the ferrolearn side).

Two pinned divergences:

* #2148 — `metric='euclidean'` with `p != 2`: sklearn ignores `p` when the metric
  is euclidean (p is only consumed in the `if self.metric == "minkowski"` branch,
  `sklearn/neighbors/_base.py:526-538`), so the config is VALID and returns a
  euclidean prediction. ferrolearn's `RsKNeighborsRegressor::fit` checks
  `self.p != 2.0` UNCONDITIONALLY and BEFORE the metric check
  (`ferrolearn-python/src/extras.rs:606-611`), so it rejects a config sklearn
  accepts and answers.

* #2149 — an INVALID `weights`/`algorithm` string: sklearn validates the choice
  and raises `InvalidParameterError`, a subclass of `ValueError`
  (`sklearn/utils/_param_validation.py`). ferrolearn raises `NotImplementedError`
  for `weights` (`extras.rs:579-584`), which is NOT a `ValueError` subclass, so a
  caller's `except ValueError:` that catches sklearn silently misses ferrolearn.
"""

import numpy as np
import pytest
import sklearn.neighbors as skn

import ferrolearn._extras as fe


def _train():
    X = np.array(
        [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [10.0, 10.0], [5.0, 5.0]],
        dtype=np.float64,
    )
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    return X, y


# ---------------------------------------------------------------------------
# Divergence #2148: metric='euclidean' with p!=2 must WORK (p ignored), not raise.
# ---------------------------------------------------------------------------
def test_divergence_euclidean_metric_ignores_p():
    """sklearn `neighbors/_base.py:526-538`: p is read only in the minkowski
    branch, so `metric='euclidean', p=3` is valid and returns a euclidean
    prediction. ferrolearn `extras.rs:606-611` rejects p!=2 unconditionally.
    Tracking: #2148.
    """
    X, y = _train()
    q = np.array([[0.5, 0.5], [3.0, 3.0]], dtype=np.float64)

    # Live sklearn 1.5.2 oracle (source of truth).
    sk_pred = skn.KNeighborsRegressor(
        n_neighbors=3, metric="euclidean", p=3
    ).fit(X, y).predict(q)

    fr = fe.KNeighborsRegressor(n_neighbors=3, metric="euclidean", p=3).fit(X, y)
    fr_pred = fr.predict(q)

    np.testing.assert_allclose(fr_pred, sk_pred, atol=1e-12)


# ---------------------------------------------------------------------------
# Divergence #2149: invalid weights string -> ValueError subclass, like sklearn.
# ---------------------------------------------------------------------------
def test_divergence_invalid_weights_string_is_valueerror():
    """sklearn raises `InvalidParameterError` (a `ValueError` subclass) for an
    invalid `weights` choice. ferrolearn raises `NotImplementedError`
    (`extras.rs:579-584`), which is NOT a `ValueError`, so `except ValueError`
    that catches sklearn misses ferrolearn. Tracking: #2149.
    """
    X, y = _train()

    # Confirm the sklearn behavior live (oracle): an invalid choice is a ValueError.
    with pytest.raises(ValueError):
        skn.KNeighborsRegressor(weights="foo").fit(X, y)

    # ferrolearn must mirror the exception *type contract*: a caller catching
    # ValueError (as sklearn users do) must also catch ferrolearn's error.
    with pytest.raises(ValueError):
        fe.KNeighborsRegressor(weights="foo").fit(X, y)
