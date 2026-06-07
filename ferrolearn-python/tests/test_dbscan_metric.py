"""Live-oracle parity guards for the `ferrolearn.DBSCAN` metric/p binding
surface (#2193): `metric` / `p` threaded into `sklearn.cluster.DBSCAN(metric=...,
p=...)`.

Verification model B (goal.md R-CHAR-3): every expected value comes from the
LIVE installed `sklearn.cluster.DBSCAN` 1.5.2 oracle in the SAME process, never
copied from ferrolearn. We run `ferrolearn.DBSCAN` and `sklearn.cluster.DBSCAN`
on the SAME fixture with the SAME `metric`/`p` and assert element-wise array
equality of `labels_` AND `core_sample_indices_`.

The Rust core's `DbscanMetric` enum + `with_metric`/`with_p` ship in
ferrolearn-cluster (commit 485c06fcd); this suite pins the PURE binding-surface
plumbing (`_RsDBSCAN(metric=..., p=...)` -> `resolve_dbscan_metric` ->
`.with_metric(...)`, `ferrolearn-python/src/extras.rs`) through the Python layer
(`ferrolearn/_extras.py::DBSCAN.__init__`/`_make_rs`).

DBSCAN is deterministic (no RNG, no iterative optimizer), so on the in-range-`eps`
path ferrolearn VALUE-matches sklearn EXACTLY for each metric.

Note #2192: `p` is the Minkowski order, used ONLY by `metric='minkowski'` and
IGNORED for every other metric — so `DBSCAN(metric='euclidean', p=3)` is plain
Euclidean, matching sklearn.
"""

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.cluster import DBSCAN as SkDBSCAN

import ferrolearn as fl


def _assert_metric_parity(X, eps, min_samples, metric, p=None):
    """Fit both estimators with the SAME metric/p and assert labels_ AND
    core_sample_indices_ match the live sklearn oracle EXACTLY."""
    fkw = dict(min_samples=min_samples, metric=metric)
    skw = dict(eps=eps, min_samples=min_samples, metric=metric)
    if p is not None:
        fkw["p"] = p
        skw["p"] = p
    f = fl.DBSCAN(eps, **fkw).fit(X)
    s = SkDBSCAN(**skw).fit(X)
    np.testing.assert_array_equal(f.labels_, s.labels_)
    np.testing.assert_array_equal(f.core_sample_indices_, s.core_sample_indices_)
    return f, s


# ---------------------------------------------------------------------------
# Metric-discriminating fixtures (the metric CHANGES the clustering, verified
# against the live oracle below — see the per-test docstrings).
# ---------------------------------------------------------------------------

def _diagonal_chain():
    # Points on the main diagonal; consecutive gap is sqrt(2) Euclidean but 2.0
    # Manhattan and 1.0 Chebyshev. Choosing eps in (1, 1.41) makes Chebyshev
    # link them all while Euclidean/Manhattan leave them isolated.
    return np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])


def test_euclidean_default_unchanged():
    """`metric='euclidean'` (explicit) AND the no-metric default must match
    sklearn's euclidean DBSCAN on a fixture with clear clusters."""
    X = np.array(
        [[0.0, 0.0], [0.3, 0.0], [0.0, 0.3], [5.0, 5.0], [5.3, 5.0], [9.0, 0.0]]
    )
    # explicit euclidean
    _assert_metric_parity(X, eps=0.5, min_samples=2, metric="euclidean")
    # default (no metric kwarg) == sklearn default 'euclidean'
    f = fl.DBSCAN(0.5, min_samples=2).fit(X)
    s = SkDBSCAN(eps=0.5, min_samples=2).fit(X)
    np.testing.assert_array_equal(f.labels_, s.labels_)
    np.testing.assert_array_equal(f.core_sample_indices_, s.core_sample_indices_)


def test_l2_alias_equals_euclidean():
    X = _diagonal_chain()
    _assert_metric_parity(X, eps=1.5, min_samples=2, metric="l2")
    fe = fl.DBSCAN(1.5, min_samples=2, metric="euclidean").fit(X)
    fl2 = fl.DBSCAN(1.5, min_samples=2, metric="l2").fit(X)
    np.testing.assert_array_equal(fe.labels_, fl2.labels_)


def test_manhattan_changes_clustering():
    """[0,0]-[1,1] is sqrt2~1.41 Euclidean but 2.0 Manhattan. With eps=1.5 the
    Euclidean fit links the diagonal (one cluster, all core) but Manhattan
    leaves every point isolated (all noise) — verified vs the live oracle."""
    X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    fe = fl.DBSCAN(1.5, min_samples=2, metric="euclidean").fit(X)
    fm = fl.DBSCAN(1.5, min_samples=2, metric="manhattan").fit(X)
    # The metric MUST change the partition for this to be a real test.
    assert not np.array_equal(fe.labels_, fm.labels_)
    _assert_metric_parity(X, eps=1.5, min_samples=2, metric="manhattan")


@pytest.mark.parametrize("alias", ["manhattan", "l1", "cityblock"])
def test_manhattan_aliases(alias):
    X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [1.0, 0.0], [0.0, 1.0]])
    _assert_metric_parity(X, eps=1.5, min_samples=2, metric=alias)


def test_chebyshev_changes_clustering():
    """[0,0]-[1,1] is 1.0 Chebyshev but sqrt2~1.41 Euclidean. With eps=1.2 the
    Chebyshev fit links the whole diagonal chain (one cluster, all core) while
    Euclidean leaves every point isolated (all noise) — verified vs the oracle."""
    X = _diagonal_chain()
    fe = fl.DBSCAN(1.2, min_samples=2, metric="euclidean").fit(X)
    fc = fl.DBSCAN(1.2, min_samples=2, metric="chebyshev").fit(X)
    assert not np.array_equal(fe.labels_, fc.labels_)
    _assert_metric_parity(X, eps=1.2, min_samples=2, metric="chebyshev")


@pytest.mark.parametrize("p", [1, 2, 3])
def test_minkowski_p(p):
    """`metric='minkowski'` with p in {1,2,3} matches sklearn's Minkowski-p
    labels AND core indices."""
    X = np.array(
        [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [1.0, 0.0], [0.0, 1.0], [4.0, 4.0]]
    )
    _assert_metric_parity(X, eps=1.5, min_samples=2, metric="minkowski", p=p)


def test_minkowski_p1_equals_manhattan():
    """minkowski p=1 == manhattan (and both == sklearn)."""
    X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [1.0, 0.0], [0.0, 1.0]])
    fp1 = fl.DBSCAN(1.5, min_samples=2, metric="minkowski", p=1).fit(X)
    fman = fl.DBSCAN(1.5, min_samples=2, metric="manhattan").fit(X)
    np.testing.assert_array_equal(fp1.labels_, fman.labels_)
    np.testing.assert_array_equal(
        fp1.core_sample_indices_, fman.core_sample_indices_
    )
    s = SkDBSCAN(eps=1.5, min_samples=2, metric="minkowski", p=1).fit(X)
    np.testing.assert_array_equal(fp1.labels_, s.labels_)


def test_minkowski_p2_equals_euclidean():
    """minkowski p=2 == euclidean (and both == sklearn)."""
    X = _diagonal_chain()
    fp2 = fl.DBSCAN(1.5, min_samples=2, metric="minkowski", p=2).fit(X)
    feu = fl.DBSCAN(1.5, min_samples=2, metric="euclidean").fit(X)
    np.testing.assert_array_equal(fp2.labels_, feu.labels_)
    np.testing.assert_array_equal(
        fp2.core_sample_indices_, feu.core_sample_indices_
    )
    s = SkDBSCAN(eps=1.5, min_samples=2, metric="minkowski", p=2).fit(X)
    np.testing.assert_array_equal(fp2.labels_, s.labels_)


def test_minkowski_p_none_rejected_like_sklearn():
    """#2193: `metric='minkowski'` with `p=None` is REJECTED (not silently
    resolved to p=2), matching the LIVE sklearn 1.5.2 oracle. sklearn forwards
    `p=None` straight to `NearestNeighbors`, which raises `TypeError`
    (`None < 1`); R-CHAR-3 makes that live behavior the contract, so ferrolearn
    also raises (a ValueError, its param-error convention) rather than fitting."""
    X = _diagonal_chain()
    # sklearn 1.5.2 errors on metric='minkowski', p=None.
    with pytest.raises(Exception):
        SkDBSCAN(eps=1.5, min_samples=2, metric="minkowski", p=None).fit(X)
    # ferrolearn mirrors: this combination raises (does NOT silently fit at p=2).
    with pytest.raises(ValueError):
        fl.DBSCAN(1.5, min_samples=2, metric="minkowski").fit(X)  # p=None


def test_euclidean_ignores_p():
    """#2192: `p` is IGNORED for non-Minkowski metrics. `metric='euclidean',
    p=3` is plain Euclidean (NOT Minkowski-3) — matching sklearn. At eps=1.3 the
    diagonal gap is 1.41 Euclidean (no link, all noise) but 1.26 Minkowski-3
    (links all), so the fixture discriminates."""
    X = _diagonal_chain()
    f_p3 = fl.DBSCAN(1.3, min_samples=2, metric="euclidean", p=3).fit(X)
    f_plain = fl.DBSCAN(1.3, min_samples=2, metric="euclidean").fit(X)
    np.testing.assert_array_equal(f_p3.labels_, f_plain.labels_)
    # sklearn ALSO ignores p for euclidean.
    s = SkDBSCAN(eps=1.3, min_samples=2, metric="euclidean", p=3).fit(X)
    np.testing.assert_array_equal(f_p3.labels_, s.labels_)
    np.testing.assert_array_equal(f_p3.core_sample_indices_, s.core_sample_indices_)
    # And it must NOT equal Minkowski-3 on this fixture (p actually matters there).
    f_mink3 = fl.DBSCAN(1.3, min_samples=2, metric="minkowski", p=3).fit(X)
    assert not np.array_equal(f_p3.labels_, f_mink3.labels_)


def test_unknown_metric_raises_valueerror():
    """An unknown metric raises ValueError (sklearn `InvalidParameterError` is a
    ValueError subclass)."""
    X = _diagonal_chain()
    with pytest.raises(ValueError):
        fl.DBSCAN(1.0, min_samples=2, metric="nope").fit(X)
    # sklearn also raises a ValueError(subclass) for the same bad metric.
    with pytest.raises(ValueError):
        SkDBSCAN(eps=1.0, min_samples=2, metric="nope").fit(X)


def test_minkowski_nonpositive_p_raises():
    """A non-positive Minkowski p raises ValueError (the Rust core's
    InvalidParameter -> PyValueError, matching `NearestNeighbors`' p in
    (0, inf])."""
    X = _diagonal_chain()
    for bad_p in (0.0, -1.0):
        with pytest.raises(ValueError):
            fl.DBSCAN(1.5, min_samples=2, metric="minkowski", p=bad_p).fit(X)
        with pytest.raises(ValueError):
            SkDBSCAN(eps=1.5, min_samples=2, metric="minkowski", p=bad_p).fit(X)


def test_metric_with_sample_weight():
    """metric + sample_weight together (manhattan + weights) matches sklearn:
    the weighted core determination rides the chosen metric's neighborhoods."""
    X = np.array(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [10.0, 10.0]]
    )
    w = np.array([3.0, 1.0, 1.0, 5.0])
    f = fl.DBSCAN(1.5, min_samples=3, metric="manhattan").fit(X, sample_weight=w)
    s = SkDBSCAN(eps=1.5, min_samples=3, metric="manhattan").fit(X, sample_weight=w)
    np.testing.assert_array_equal(f.labels_, s.labels_)
    np.testing.assert_array_equal(f.core_sample_indices_, s.core_sample_indices_)


# ---------------------------------------------------------------------------
# sklearn-compat surface: clone / get_params / set_params
# ---------------------------------------------------------------------------

def test_get_params_includes_metric_p():
    est = fl.DBSCAN(0.7, min_samples=4, metric="manhattan", p=None)
    params = est.get_params()
    assert params["eps"] == 0.7
    assert params["min_samples"] == 4
    assert params["metric"] == "manhattan"
    assert params["p"] is None


def test_clone_roundtrips_metric_p():
    est = fl.DBSCAN(0.9, min_samples=2, metric="minkowski", p=3)
    c = clone(est)
    assert c.metric == "minkowski"
    assert c.p == 3
    assert c.eps == 0.9
    assert c.min_samples == 2


def test_set_params_metric_fit():
    """`set_params(metric='manhattan').fit(X)` rebuilds the Rust handle with the
    new metric and matches sklearn."""
    X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    est = fl.DBSCAN(1.5, min_samples=2).set_params(metric="manhattan")
    est.fit(X)
    s = SkDBSCAN(eps=1.5, min_samples=2, metric="manhattan").fit(X)
    np.testing.assert_array_equal(est.labels_, s.labels_)
    np.testing.assert_array_equal(est.core_sample_indices_, s.core_sample_indices_)
