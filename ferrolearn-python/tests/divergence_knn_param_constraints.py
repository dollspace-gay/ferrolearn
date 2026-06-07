"""Divergence pins for KNN _validate_params constraints remaining AFTER the
#2148-#2152 KNN-constructor fixes (classifiers.rs + extras.rs + the
_classifiers.py/_extras.py shims).

Verification model B: compare `import ferrolearn` against `import sklearn` 1.5.2.
Every expected/oracle behavior is regenerated from the live sklearn 1.5.2 oracle
inside the test (R-CHAR-3 — never copied from the ferrolearn side).

sklearn's KNN parameter constraints (sklearn/neighbors/_base.py:395-405, plus the
weights override at neighbors/_classification.py:189-191 and
neighbors/_regression.py:171-174):

    "n_neighbors":   [Interval(Integral, 1, None, closed="left"), None],
    "algorithm":     [StrOptions({"auto","ball_tree","kd_tree","brute"})],
    "leaf_size":     [Interval(Integral, 1, None, closed="left")],
    "p":             [Interval(Real, 0, None, closed="right"), None],
    "metric":        [StrOptions(...), callable],
    "metric_params": [dict, None],
    "n_jobs":        [Integral, None],
    "weights":       [StrOptions({"uniform","distance"}), callable, None],

A constraint violation raises InvalidParameterError, which IS a subclass of
ValueError (sklearn/utils/_param_validation.py). A VALID value is accepted and
fit proceeds. All four divergences below apply IDENTICALLY to
KNeighborsClassifier and KNeighborsRegressor (the validation logic in
classifiers.rs and extras.rs is line-for-line parallel).

Pinned divergences:

* #2152 — n_jobs non-int (float/str): sklearn constraint is [Integral, None]
  (neighbors/_base.py:404). A non-int n_jobs (1.5, 'x') is an
  InvalidParameterError (⊂ ValueError). ferrolearn binds n_jobs as PyO3
  Option<i64> (classifiers.rs:387, extras.rs), so a float/str raises a PyO3
  TypeError — NOT catchable as ValueError. (int/-1/0/None and np.int64 are
  accepted by both — no divergence there.)

* #2153 — metric_params={} (EMPTY dict): sklearn constraint is [dict, None]
  (neighbors/_base.py:402); an empty dict is a VALID dict, so sklearn accepts it
  and fits/predicts normally (an empty dict adds no custom metric kwargs).
  ferrolearn rejects ANY non-None metric_params with NotImplementedError
  (_classifiers.py:391, _extras.py:324) — so it raises where sklearn succeeds.

* #2154 — metric_params=[] (list): sklearn rejects a non-dict, non-None
  metric_params with InvalidParameterError (⊂ ValueError). ferrolearn raises
  NotImplementedError, which is NOT catchable as ValueError — an
  exception-TYPE divergence at the validation surface.

* #2155 — weights=None: sklearn's weights constraint EXPLICITLY includes None
  (neighbors/_classification.py:190, neighbors/_regression.py:173); None is
  valid and behaves IDENTICALLY to 'uniform' (_base.py::_get_weights returns
  None -> uniform averaging). ferrolearn binds weights as a PyO3 String
  (classifiers.rs:405, extras.rs), so weights=None raises a PyO3 TypeError —
  NOT catchable as ValueError, and the valid config is rejected outright.
"""

import numpy as np
import pytest

import ferrolearn as fl
import sklearn.neighbors as skn


def _data_clf():
    X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
    y = np.array([0, 0, 1, 1, 1])
    return X, y


def _data_reg():
    X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
    y = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    return X, y


# ---------------------------------------------------------------------------
# #2152 — n_jobs non-int (float/str): sklearn ValueError; ferrolearn TypeError
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_jobs", [1.5, "x"])
def test_knn_clf_n_jobs_noninteger_is_valueerror(n_jobs):
    X, y = _data_clf()
    # Oracle: sklearn rejects a non-Integral n_jobs as a ValueError subclass.
    with pytest.raises(ValueError):
        skn.KNeighborsClassifier(n_jobs=n_jobs).fit(X, y)
    # ferrolearn must mirror: catchable as ValueError, not a PyO3 TypeError.
    with pytest.raises(ValueError):
        fl.KNeighborsClassifier(n_jobs=n_jobs).fit(X, y)


@pytest.mark.parametrize("n_jobs", [1.5, "x"])
def test_knn_reg_n_jobs_noninteger_is_valueerror(n_jobs):
    X, y = _data_reg()
    with pytest.raises(ValueError):
        skn.KNeighborsRegressor(n_jobs=n_jobs).fit(X, y)
    with pytest.raises(ValueError):
        fl.KNeighborsRegressor(n_jobs=n_jobs).fit(X, y)


# ---------------------------------------------------------------------------
# #2153 — metric_params={} (empty dict): sklearn ACCEPTS; ferrolearn raises
# ---------------------------------------------------------------------------


def test_knn_clf_metric_params_empty_dict_is_accepted():
    X, y = _data_clf()
    Xq = np.array([[1.5]])
    # Oracle: sklearn accepts an empty dict and predicts exactly as None would.
    sk = skn.KNeighborsClassifier(n_neighbors=3, metric_params={}).fit(X, y)
    expected = sk.predict(Xq)
    # ferrolearn must also accept {} and produce the same prediction.
    got = fl.KNeighborsClassifier(n_neighbors=3, metric_params={}).fit(X, y).predict(Xq)
    np.testing.assert_array_equal(got, expected)


def test_knn_reg_metric_params_empty_dict_is_accepted():
    X, y = _data_reg()
    Xq = np.array([[1.5]])
    sk = skn.KNeighborsRegressor(n_neighbors=3, metric_params={}).fit(X, y)
    expected = sk.predict(Xq)
    got = fl.KNeighborsRegressor(n_neighbors=3, metric_params={}).fit(X, y).predict(Xq)
    np.testing.assert_allclose(got, expected, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# #2154 — metric_params=[] (list): sklearn ValueError; ferrolearn NotImplemented
# ---------------------------------------------------------------------------


def test_knn_clf_metric_params_list_is_valueerror():
    X, y = _data_clf()
    # Oracle: a non-dict, non-None metric_params is a ValueError subclass.
    with pytest.raises(ValueError):
        skn.KNeighborsClassifier(metric_params=[]).fit(X, y)
    # ferrolearn raises NotImplementedError (not catchable as ValueError).
    with pytest.raises(ValueError):
        fl.KNeighborsClassifier(metric_params=[]).fit(X, y)


def test_knn_reg_metric_params_list_is_valueerror():
    X, y = _data_reg()
    with pytest.raises(ValueError):
        skn.KNeighborsRegressor(metric_params=[]).fit(X, y)
    with pytest.raises(ValueError):
        fl.KNeighborsRegressor(metric_params=[]).fit(X, y)


# ---------------------------------------------------------------------------
# #2155 — weights=None: sklearn ACCEPTS (== 'uniform'); ferrolearn TypeError
# ---------------------------------------------------------------------------


def test_knn_clf_weights_none_is_accepted_as_uniform():
    X, y = _data_clf()
    Xq = np.array([[1.5]])
    # Oracle: sklearn treats weights=None identically to weights='uniform'.
    sk_none = skn.KNeighborsClassifier(n_neighbors=3, weights=None).fit(X, y)
    sk_unif = skn.KNeighborsClassifier(n_neighbors=3, weights="uniform").fit(X, y)
    expected = sk_none.predict(Xq)
    np.testing.assert_array_equal(expected, sk_unif.predict(Xq))
    np.testing.assert_allclose(
        sk_none.predict_proba(Xq), sk_unif.predict_proba(Xq), rtol=0, atol=0
    )
    # ferrolearn must accept weights=None and match the uniform prediction.
    got = fl.KNeighborsClassifier(n_neighbors=3, weights=None).fit(X, y).predict(Xq)
    np.testing.assert_array_equal(got, expected)


def test_knn_reg_weights_none_is_accepted_as_uniform():
    X, y = _data_reg()
    Xq = np.array([[1.5]])
    sk_none = skn.KNeighborsRegressor(n_neighbors=3, weights=None).fit(X, y)
    sk_unif = skn.KNeighborsRegressor(n_neighbors=3, weights="uniform").fit(X, y)
    expected = sk_none.predict(Xq)
    np.testing.assert_allclose(expected, sk_unif.predict(Xq), rtol=0, atol=0)
    got = fl.KNeighborsRegressor(n_neighbors=3, weights=None).fit(X, y).predict(Xq)
    np.testing.assert_allclose(got, expected, rtol=0, atol=0)
