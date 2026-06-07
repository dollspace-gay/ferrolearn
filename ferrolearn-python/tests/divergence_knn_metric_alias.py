"""Divergence pins for KNN metric/`p` validation gaps remaining AFTER the
#2148/#2149 KNN-constructor fixes (classifiers.rs + extras.rs).

Verification model B: compare `import ferrolearn` against `import sklearn` 1.5.2.
Every expected value is regenerated from the live sklearn 1.5.2 oracle inside the
test (R-CHAR-3 — never copied from the ferrolearn side).

Two pinned divergences (both apply IDENTICALLY to KNeighborsClassifier and
KNeighborsRegressor — the #2148/#2149 fixes were applied to both, and so are
these remaining gaps):

* #2150 — `metric='l2'`: sklearn lists `'l2'` in
  `VALID_METRICS['brute']`/`['kd_tree']` (the `metric` StrOptions constraint at
  `sklearn/neighbors/_base.py:401`); `'l2'` IS the Euclidean (p=2) distance, so
  sklearn fits and predicts the SAME numbers as `metric='euclidean'` under both
  uniform and distance weights. ferrolearn's `fit` only special-cases the
  literal strings `"euclidean"`/`"minkowski"` (`classifiers.rs:465-482`,
  `extras.rs:612-629`) and rejects every other string with
  `NotImplementedError`. `'l2'` is a config sklearn AND ferrolearn would agree on
  numerically (Euclidean), so rejecting it is a real divergence, not an honest
  unsupported-metric reject (unlike `'manhattan'`/`'cosine'`, which compute a
  genuinely different distance the Rust core does not implement).

* #2151 — `p <= 0` with `metric='euclidean'`: sklearn's `p` parameter constraint
  is `Interval(Real, 0, None, closed="right")` i.e. `(0.0, inf]`
  (`sklearn/neighbors/_base.py:400`), validated UNCONDITIONALLY by
  `_validate_params()` at the top of `fit` — BEFORE the metric is consulted. So
  `p=-1` / `p=0` raise `InvalidParameterError` (a `ValueError` subclass)
  regardless of metric. ferrolearn validates `p` only inside the
  `metric=='minkowski'` branch (`classifiers.rs:467-475`, `extras.rs:614-622`);
  for `metric='euclidean'` it never inspects `p`, so it silently ACCEPTS
  `p=-1`/`p=0` and fits, where sklearn raises.
"""

import numpy as np
import pytest

import ferrolearn as fl
import sklearn.neighbors as skn


def _train_clf():
    X = np.array(
        [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [0.5, 0.5], [2.5, 2.5]],
        dtype=np.float64,
    )
    y = np.array([0, 1, 1, 0, 0, 1])
    Xte = np.array([[1.2, 0.9], [2.6, 2.4]], dtype=np.float64)
    return X, y, Xte


def _train_reg():
    X = np.array(
        [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [0.5, 0.5], [2.5, 2.5]],
        dtype=np.float64,
    )
    y = np.array([10.0, 20.0, 30.0, 40.0, 15.0, 35.0])
    Xte = np.array([[1.2, 0.9], [2.6, 2.4]], dtype=np.float64)
    return X, y, Xte


# ---------------------------------------------------------------------------
# #2150 — metric='l2' (Euclidean alias) accepted by sklearn, rejected by ferrolearn
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("weights", ["uniform", "distance"])
def test_knn_clf_metric_l2_alias_matches_euclidean(weights):
    """sklearn accepts metric='l2' and computes the Euclidean result; ferrolearn
    rejects 'l2' with NotImplementedError (divergence #2150)."""
    X, y, Xte = _train_clf()
    # Oracle: sklearn 'l2' == sklearn 'euclidean' (regenerated live, R-CHAR-3).
    sk_l2 = skn.KNeighborsClassifier(3, weights=weights, metric="l2").fit(X, y)
    sk_proba = sk_l2.predict_proba(Xte)
    sk_pred = sk_l2.predict(Xte)

    fl_clf = fl.KNeighborsClassifier(3, weights=weights, metric="l2").fit(X, y)
    fl_proba = np.asarray(fl_clf.predict_proba(Xte))
    assert np.allclose(fl_proba, sk_proba)
    assert np.array_equal(np.asarray(fl_clf.predict(Xte)), sk_pred)


@pytest.mark.parametrize("weights", ["uniform", "distance"])
def test_knn_reg_metric_l2_alias_matches_euclidean(weights):
    """sklearn accepts metric='l2' and computes the Euclidean result; ferrolearn
    rejects 'l2' with NotImplementedError (divergence #2150)."""
    X, y, Xte = _train_reg()
    sk_pred = skn.KNeighborsRegressor(3, weights=weights, metric="l2").fit(X, y).predict(Xte)
    fl_pred = np.asarray(
        fl.KNeighborsRegressor(3, weights=weights, metric="l2").fit(X, y).predict(Xte)
    )
    assert np.allclose(fl_pred, sk_pred)


# ---------------------------------------------------------------------------
# #2151 — p <= 0 with metric='euclidean': sklearn rejects (ValueError),
#         ferrolearn silently accepts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("p", [-1.0, 0.0])
def test_knn_clf_euclidean_nonpositive_p_is_valueerror(p):
    """sklearn's p constraint (0, inf] (neighbors/_base.py:400) is validated
    regardless of metric, so p<=0 raises a ValueError; ferrolearn accepts it for
    metric='euclidean' (divergence #2151)."""
    X, y, _ = _train_clf()
    # Oracle: confirm sklearn rejects with a ValueError subclass.
    with pytest.raises(ValueError):
        skn.KNeighborsClassifier(2, metric="euclidean", p=p).fit(X, y)
    # ferrolearn must mirror sklearn: a p outside (0, inf] is a ValueError.
    with pytest.raises(ValueError):
        fl.KNeighborsClassifier(2, metric="euclidean", p=p).fit(X, y)


@pytest.mark.parametrize("p", [-1.0, 0.0])
def test_knn_reg_euclidean_nonpositive_p_is_valueerror(p):
    """sklearn's p constraint (0, inf] (neighbors/_base.py:400) is validated
    regardless of metric, so p<=0 raises a ValueError; ferrolearn accepts it for
    metric='euclidean' (divergence #2151)."""
    X, y, _ = _train_reg()
    with pytest.raises(ValueError):
        skn.KNeighborsRegressor(2, metric="euclidean", p=p).fit(X, y)
    with pytest.raises(ValueError):
        fl.KNeighborsRegressor(2, metric="euclidean", p=p).fit(X, y)
