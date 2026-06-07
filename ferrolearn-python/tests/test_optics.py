"""Live-oracle parity tests for ferrolearn.OPTICS vs sklearn.cluster.OPTICS (#1090).

Verification model B (goal.md): compare ``import ferrolearn`` against
``import sklearn`` 1.5.2 (the installed package IS the oracle). Every expected
value is produced by a LIVE ``sklearn.cluster.OPTICS`` call here (R-CHAR-3 — never
copied from the ferrolearn side). OPTICS is DETERMINISTIC (no RNG), so the
reachability ordering, reachability/core distances, predecessor array, labels, and
cluster hierarchy are bit/value-comparable.

The Rust OPTICS core (ferrolearn_cluster optics.rs) is pre-existing/audited and
bit-exact vs sklearn on f64; this binding surfaces it through ``ferrolearn.OPTICS``.
"""

import numpy as np
import pytest
from sklearn.cluster import OPTICS as SkOPTICS
from sklearn.datasets import make_blobs

import ferrolearn as fl


# The `_optics.py:891-896` doctest fixture: a small, exactly-reproducible set.
DOCTEST_X = np.array(
    [[1, 2], [2, 5], [3, 6], [8, 7], [8, 8], [7, 3]], dtype=np.float64
)


def _blobs():
    X, _ = make_blobs(n_samples=60, centers=3, cluster_std=0.4, random_state=7)
    return X.astype(np.float64)


def _assert_optics_parity(X, atol=1e-9, **kwargs):
    """Fit both, return (fl_model, sk_model) after asserting full attr parity."""
    sk = SkOPTICS(**kwargs).fit(X)
    f = fl.OPTICS(**kwargs).fit(X)

    # ordering_ — integer-exact reachability traversal order.
    np.testing.assert_array_equal(f.ordering_, sk.ordering_)
    # labels_ — integer-exact cluster assignment (incl. -1 noise).
    np.testing.assert_array_equal(f.labels_, sk.labels_)
    # predecessor_ — integer-exact -1-sentinel array.
    np.testing.assert_array_equal(f.predecessor_, sk.predecessor_)
    # core_distances_ — float, ~atol (inf must match position-wise).
    np.testing.assert_array_equal(
        np.isinf(f.core_distances_), np.isinf(sk.core_distances_)
    )
    finite = np.isfinite(sk.core_distances_)
    np.testing.assert_allclose(
        f.core_distances_[finite], sk.core_distances_[finite], atol=atol
    )
    # reachability_ — float with inf handling.
    np.testing.assert_array_equal(
        np.isinf(f.reachability_), np.isinf(sk.reachability_)
    )
    rfinite = np.isfinite(sk.reachability_)
    np.testing.assert_allclose(
        f.reachability_[rfinite], sk.reachability_[rfinite], atol=atol
    )
    return f, sk


# ---------------------------------------------------------------------------
# Xi method (default) — full attribute parity
# ---------------------------------------------------------------------------

def test_xi_doctest_fixture_full_parity():
    f, sk = _assert_optics_parity(DOCTEST_X, min_samples=2)
    # The doctest's documented values (sklearn/cluster/_optics.py:891-896).
    np.testing.assert_array_equal(f.labels_, [0, 0, 0, 1, 1, 1])
    np.testing.assert_array_equal(
        f.cluster_hierarchy_, sk.cluster_hierarchy_
    )
    np.testing.assert_array_equal(
        f.cluster_hierarchy_, [[0, 2], [3, 5], [0, 5]]
    )


def test_xi_blobs_full_parity():
    f, sk = _assert_optics_parity(_blobs(), min_samples=5)
    np.testing.assert_array_equal(f.cluster_hierarchy_, sk.cluster_hierarchy_)


def test_xi_min_cluster_size_int_parity():
    X = _blobs()
    f, sk = _assert_optics_parity(X, min_samples=5, min_cluster_size=10)
    np.testing.assert_array_equal(f.cluster_hierarchy_, sk.cluster_hierarchy_)
    # min_cluster_size=10 drops a small cluster vs the default — distinct labels.
    f_def = fl.OPTICS(min_samples=5).fit(X)
    assert set(f.labels_.tolist()) != set(f_def.labels_.tolist())


@pytest.mark.parametrize("xi", [0.01, 0.05, 0.1, 0.2])
def test_xi_threshold_variants_parity(xi):
    f, sk = _assert_optics_parity(_blobs(), min_samples=5, xi=xi)
    np.testing.assert_array_equal(f.cluster_hierarchy_, sk.cluster_hierarchy_)


def test_xi_predecessor_correction_toggle_parity():
    _assert_optics_parity(
        _blobs(), min_samples=5, predecessor_correction=False
    )


# ---------------------------------------------------------------------------
# DBSCAN method — labels parity (+ eps resolution)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("eps", [1.0, 2.0, 5.0])
def test_dbscan_method_blobs_parity(eps):
    f, sk = _assert_optics_parity(
        _blobs(), min_samples=5, cluster_method="dbscan", eps=eps
    )
    np.testing.assert_array_equal(f.labels_, sk.labels_)


def test_dbscan_method_doctest_parity():
    f, sk = _assert_optics_parity(
        DOCTEST_X, min_samples=2, cluster_method="dbscan", eps=2.0
    )
    np.testing.assert_array_equal(f.labels_, sk.labels_)


def test_dbscan_method_has_no_cluster_hierarchy():
    # sklearn does NOT set cluster_hierarchy_ on the dbscan branch
    # (`_optics.py:373` is Xi-only); the wrapper must omit the attribute too.
    sk = SkOPTICS(min_samples=2, cluster_method="dbscan", eps=2.0).fit(DOCTEST_X)
    f = fl.OPTICS(min_samples=2, cluster_method="dbscan", eps=2.0).fit(DOCTEST_X)
    assert not hasattr(sk, "cluster_hierarchy_")
    assert not hasattr(f, "cluster_hierarchy_")


def test_xi_method_has_cluster_hierarchy():
    f = fl.OPTICS(min_samples=2).fit(DOCTEST_X)
    assert hasattr(f, "cluster_hierarchy_")


# ---------------------------------------------------------------------------
# Validation — raises matching sklearn (InvalidParameterError ⊂ ValueError)
# ---------------------------------------------------------------------------

def test_min_samples_one_raises_valueerror():
    # sklearn: min_samples >= 2 (Interval(Integral, 2, ...), _optics.py:243).
    with pytest.raises(ValueError):
        SkOPTICS(min_samples=1).fit(DOCTEST_X)
    with pytest.raises(ValueError):
        fl.OPTICS(min_samples=1).fit(DOCTEST_X)


def test_min_samples_exceeds_n_samples_raises_valueerror():
    # sklearn: _validate_size requires min_samples <= n_samples (_optics.py:393).
    with pytest.raises(ValueError):
        SkOPTICS(min_samples=7).fit(DOCTEST_X)
    with pytest.raises(ValueError):
        fl.OPTICS(min_samples=7).fit(DOCTEST_X)


def test_unknown_cluster_method_raises_valueerror():
    # sklearn: StrOptions({"xi","dbscan"}) (_optics.py:251).
    with pytest.raises(ValueError):
        SkOPTICS(cluster_method="zzz").fit(DOCTEST_X)
    with pytest.raises(ValueError):
        fl.OPTICS(cluster_method="zzz").fit(DOCTEST_X)


def test_float_min_samples_raises_notimplemented():
    # sklearn ACCEPTS a (0, 1] fraction (RealNotInt, _optics.py:245); the Rust
    # core takes a usize, so ferrolearn honestly raises NotImplementedError
    # (REQ-10 NOT-STARTED #1088) rather than silently truncating.
    # Confirm sklearn does accept it (so this IS a real gap, not a shared error).
    SkOPTICS(min_samples=0.5).fit(DOCTEST_X)
    with pytest.raises(NotImplementedError):
        fl.OPTICS(min_samples=0.5).fit(DOCTEST_X)


def test_non_minkowski_metric_raises_notimplemented():
    # sklearn accepts metric='manhattan'; the Rust OPTICS core is Euclidean-only.
    SkOPTICS(min_samples=2, metric="manhattan").fit(DOCTEST_X)
    with pytest.raises(NotImplementedError):
        fl.OPTICS(min_samples=2, metric="manhattan").fit(DOCTEST_X)


def test_non_default_p_raises_notimplemented():
    with pytest.raises(NotImplementedError):
        fl.OPTICS(min_samples=2, p=3).fit(DOCTEST_X)


# ---------------------------------------------------------------------------
# sklearn-compat: clone / get_params / defaults / fit_predict
# ---------------------------------------------------------------------------

def test_default_min_samples_is_five():
    # sklearn default min_samples=5 (_optics.py:269).
    assert fl.OPTICS().min_samples == 5
    assert SkOPTICS().min_samples == 5


def test_get_params_includes_sklearn_params():
    params = fl.OPTICS().get_params()
    for key in (
        "min_samples", "max_eps", "metric", "p", "metric_params",
        "cluster_method", "eps", "xi", "predecessor_correction",
        "min_cluster_size", "algorithm", "leaf_size", "memory", "n_jobs",
    ):
        assert key in params, key
    # defaults mirror sklearn (_optics.py:266-297).
    assert params["cluster_method"] == "xi"
    assert params["xi"] == 0.05
    assert params["predecessor_correction"] is True
    assert params["metric"] == "minkowski"
    assert params["p"] == 2


def test_clone_roundtrips():
    from sklearn.base import clone

    est = fl.OPTICS(min_samples=4, xi=0.07, cluster_method="dbscan", eps=1.5)
    cl = clone(est)
    assert cl.min_samples == 4
    assert cl.xi == 0.07
    assert cl.cluster_method == "dbscan"
    assert cl.eps == 1.5
    # cloned estimator fits and matches the original.
    X = _blobs()
    np.testing.assert_array_equal(cl.fit(X).labels_, est.fit(X).labels_)


def test_fit_predict_matches_fit_labels():
    X = _blobs()
    f = fl.OPTICS(min_samples=5)
    fp = fl.OPTICS(min_samples=5).fit_predict(X)
    np.testing.assert_array_equal(fp, f.fit(X).labels_)
    # and matches sklearn's fit_predict.
    sk_fp = SkOPTICS(min_samples=5).fit_predict(X)
    np.testing.assert_array_equal(fp, sk_fp)


def test_n_features_in_set():
    f = fl.OPTICS(min_samples=2).fit(DOCTEST_X)
    assert f.n_features_in_ == 2
