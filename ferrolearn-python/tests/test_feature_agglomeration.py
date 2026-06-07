"""Live-sklearn parity/divergence tests for ferrolearn.FeatureAgglomeration (#943).

Verification model B (goal.md): compare `import ferrolearn` against the installed
`import sklearn` 1.5.2 oracle. ALL expected values come from the live
`sklearn.cluster.FeatureAgglomeration` (R-CHAR-3) — NEVER copied from ferrolearn.

The Rust core (ferrolearn_cluster::FeatureAgglomeration / FittedFeatureAgglomeration)
is pre-existing/audited (feature_agglomeration.rs REQ-1..9 SHIPPED); this file
verifies the PyO3 binding surface exposes it with sklearn-matching values.
"""

import numpy as np
import pytest

import ferrolearn as fl
from sklearn.cluster import FeatureAgglomeration as Sk

LINKAGES = ["ward", "complete", "average", "single"]
POOLINGS = {"mean": np.mean, "max": np.max}


def _fixture():
    # 6 features in 3 correlated pairs: (0,1), (2,3), (4,5). 8 samples so the
    # ward/complete/average/single dendrograms are well-defined and stable.
    rng = np.random.RandomState(0)
    base = rng.randn(8, 3) * 10.0
    X = np.empty((8, 6))
    X[:, 0] = base[:, 0]
    X[:, 1] = base[:, 0] + rng.randn(8) * 0.01
    X[:, 2] = base[:, 1]
    X[:, 3] = base[:, 1] + rng.randn(8) * 0.01
    X[:, 4] = base[:, 2]
    X[:, 5] = base[:, 2] + rng.randn(8) * 0.01
    return X


@pytest.mark.parametrize("linkage", LINKAGES)
@pytest.mark.parametrize("pooling_name", list(POOLINGS))
def test_labels_transform_inverse_match_sklearn(linkage, pooling_name):
    X = _fixture()
    pooling = POOLINGS[pooling_name]
    fa = fl.FeatureAgglomeration(3, linkage=linkage, pooling_func=pooling).fit(X)
    sk = Sk(n_clusters=3, linkage=linkage, pooling_func=pooling).fit(X)

    # labels_ integer-exact (the inner _hc_cut contiguous numbering).
    np.testing.assert_array_equal(fa.labels_, sk.labels_)

    # transform value-exact and column-ordered (sklearn bincount mean / unique max).
    np.testing.assert_allclose(fa.transform(X), sk.transform(X), rtol=0, atol=1e-9)

    # inverse_transform value-exact.
    fa_inv = fa.inverse_transform(fa.transform(X))
    sk_inv = sk.inverse_transform(sk.transform(X))
    np.testing.assert_allclose(fa_inv, sk_inv, rtol=0, atol=1e-9)


@pytest.mark.parametrize("linkage", LINKAGES)
def test_fitted_attrs_match_sklearn(linkage):
    X = _fixture()
    fa = fl.FeatureAgglomeration(3, linkage=linkage, compute_distances=True).fit(X)
    sk = Sk(n_clusters=3, linkage=linkage, compute_distances=True).fit(X)

    assert fa.n_clusters_ == sk.n_clusters_
    assert fa.n_leaves_ == sk.n_leaves_
    assert fa.n_connected_components_ == sk.n_connected_components_
    np.testing.assert_array_equal(fa.children_, sk.children_)
    np.testing.assert_allclose(fa.distances_, sk.distances_, rtol=0, atol=1e-9)


def test_distances_none_when_not_computed():
    # sklearn does not SET `distances_` when compute_distances=False (the attribute
    # is absent, _agglomerative.py:1093-1095); ferrolearn surfaces None for the
    # same "not computed" state. Both express "no distances".
    X = _fixture()
    fa = fl.FeatureAgglomeration(3).fit(X)
    sk = Sk(n_clusters=3).fit(X)
    assert fa.distances_ is None
    assert not hasattr(sk, "distances_")


def test_fit_transform_matches_fit_then_transform():
    X = _fixture()
    fa = fl.FeatureAgglomeration(3)
    direct = fa.fit_transform(X)
    sep = fl.FeatureAgglomeration(3).fit(X).transform(X)
    np.testing.assert_allclose(direct, sep, rtol=0, atol=1e-12)
    # And matches sklearn's fit_transform.
    sk = Sk(n_clusters=3).fit_transform(X)
    np.testing.assert_allclose(direct, sk, rtol=0, atol=1e-9)


def test_pooling_func_string_and_numpy_equivalent():
    X = _fixture()
    str_max = fl.FeatureAgglomeration(3, pooling_func="max").fit(X).transform(X)
    np_max = fl.FeatureAgglomeration(3, pooling_func=np.max).fit(X).transform(X)
    np.testing.assert_array_equal(str_max, np_max)
    str_mean = fl.FeatureAgglomeration(3, pooling_func="mean").fit(X).transform(X)
    np_mean = fl.FeatureAgglomeration(3, pooling_func=np.mean).fit(X).transform(X)
    np.testing.assert_array_equal(str_mean, np_mean)


def test_get_set_params_roundtrip_and_default_n_clusters():
    fa = fl.FeatureAgglomeration()
    params = fa.get_params()
    # sklearn default n_clusters == 2 (_agglomerative.py:1298).
    assert params["n_clusters"] == 2
    assert Sk().get_params()["n_clusters"] == 2
    fa.set_params(n_clusters=4, linkage="complete")
    assert fa.get_params()["n_clusters"] == 4
    assert fa.get_params()["linkage"] == "complete"


def test_n_clusters_positional_like_sklearn():
    # sklearn FeatureAgglomeration.n_clusters is POSITIONAL_OR_KEYWORD.
    assert fl.FeatureAgglomeration(5).n_clusters == 5


def test_arbitrary_callable_pooling_not_implemented():
    # sklearn permits any callable; the Rust core has only Mean/Max.
    with pytest.raises(NotImplementedError):
        fl.FeatureAgglomeration(3, pooling_func=np.median).fit(_fixture())


def test_unknown_pooling_string_raises_value_error():
    with pytest.raises(ValueError):
        fl.FeatureAgglomeration(3, pooling_func="median").fit(_fixture())


def test_distance_threshold_not_implemented():
    # sklearn supports a distance-threshold cut; the Rust core cuts by n_clusters.
    with pytest.raises(NotImplementedError):
        fl.FeatureAgglomeration(None, distance_threshold=1.0).fit(_fixture())


def test_min_features_two_rejected_like_sklearn():
    # sklearn _validate_data(ensure_min_features=2) raises ValueError on 1 feature.
    X1 = _fixture()[:, :1]
    with pytest.raises(ValueError):
        fl.FeatureAgglomeration(1).fit(X1)
    with pytest.raises(ValueError):
        Sk(n_clusters=1).fit(X1)
