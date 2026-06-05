"""Divergence guards for the ferrolearn-python "extras" binding shim (unit #2054).

Targets: `ferrolearn-python/src/extras.rs` (the ~40 `_Rs*` pyclasses) + the
Python wrappers `ferrolearn-python/python/ferrolearn/_extras.py` (the
`_RegressorWrapper`/`_ClassifierWrapper`/`_ClusterWrapper`/`_TransformerWrapper`
subclasses), mirroring the corresponding sklearn estimators across
`sklearn.ensemble`, `sklearn.linear_model`, `sklearn.kernel_ridge`,
`sklearn.neighbors`, `sklearn.cluster`, `sklearn.mixture`,
`sklearn.decomposition`, `sklearn.preprocessing`, `sklearn.naive_bayes`,
`sklearn.discriminant_analysis`, `sklearn.svm`, `sklearn.tree`.

Verification model B (goal.md): every expected value is computed by the LIVE
sklearn 1.5.2 oracle in the SAME test and compared against `import ferrolearn`.
No expected value is literal-copied from the ferrolearn side (R-CHAR-3).

Contents:
  - ONE parametrized FAILING constructor-ABI pin
    (`test_red_extras_primary_param_positional`) over the 16 estimators whose
    sklearn primary hyperparameter is POSITIONAL_OR_KEYWORD (R-DEV-2,
    REQ-CTOR-ABI-POSITIONAL). For each, the live sklearn oracle accepts the
    param positionally (asserted in-test); ferrolearn raises TypeError because
    every `_extras.py` `__init__` makes the primary param keyword-only
    (`def __init__(self, *, <param>=...)`). All 16 cases currently FAIL. The
    single-file fix: move each primary param before the `*` in `_extras.py`.

  - PASSING per-category API-conformance GREEN guards (structure + oracle-anchored
    accuracy/match, NOT exact RNG values). Regressors / classifiers / clusterers /
    transformers. The deterministic classifiers additionally assert
    ferrolearn's prediction accuracy EQUALS the live sklearn oracle's accuracy
    on identical separable data (R-CHAR-3 — the oracle, not a hardcoded 1.0),
    and that the accuracy is 1.0 on cleanly-separable data.

The CTOR-ABI-POSITIONAL anti-pattern (16 estimators, one `_extras.py` edit) and
the module-root `#![allow(non_snake_case)]` at `extras.rs:5` are filed as
-l blocker crosslink issues. Only the single-file-fixable CTOR-ABI pin is RED
here, per R-DEFER-3.
"""

import inspect

import numpy as np
import pytest
from sklearn.datasets import make_blobs

import ferrolearn as fl

# Live sklearn 1.5.2 oracle imports (R-CHAR-3 — expected values come from here).
from sklearn.cluster import (
    DBSCAN as SkDBSCAN,
    AgglomerativeClustering as SkAgglomerativeClustering,
    MiniBatchKMeans as SkMiniBatchKMeans,
)
from sklearn.decomposition import (
    FactorAnalysis as SkFactorAnalysis,
    FastICA as SkFastICA,
    IncrementalPCA as SkIncrementalPCA,
    KernelPCA as SkKernelPCA,
    NMF as SkNMF,
    SparsePCA as SkSparsePCA,
    TruncatedSVD as SkTruncatedSVD,
)
from sklearn.discriminant_analysis import (
    QuadraticDiscriminantAnalysis as SkQDA,
)
from sklearn.ensemble import (
    ExtraTreesRegressor as SkExtraTreesRegressor,
    RandomForestRegressor as SkRandomForestRegressor,
)
from sklearn.kernel_ridge import KernelRidge as SkKernelRidge
from sklearn.linear_model import RidgeClassifier as SkRidgeClassifier
from sklearn.mixture import GaussianMixture as SkGaussianMixture
from sklearn.naive_bayes import (
    BernoulliNB as SkBernoulliNB,
    ComplementNB as SkComplementNB,
    MultinomialNB as SkMultinomialNB,
)
from sklearn.neighbors import (
    KNeighborsRegressor as SkKNeighborsRegressor,
    NearestCentroid as SkNearestCentroid,
)


# ---------------------------------------------------------------------------
# RED pin: constructor-ABI positional primary hyperparameter (REQ-CTOR-ABI-POSITIONAL)
#
# The 16 estimators whose sklearn primary hyperparameter is
# POSITIONAL_OR_KEYWORD. Each ferrolearn wrapper `__init__` makes it keyword-only
# (leading `*`), so `ferrolearn.X(val)` raises TypeError while `sklearn.X(val)`
# accepts it. The (sklearn class, ferrolearn name, primary param, sample value).
# The sklearn class is carried so the live oracle is asserted IN THE SAME TEST.
# ---------------------------------------------------------------------------

# (sklearn_cls, ferrolearn_name, primary_param, sample_value)
_POSITIONAL_CASES = [
    (SkRandomForestRegressor, "RandomForestRegressor", "n_estimators", 10),
    (SkExtraTreesRegressor, "ExtraTreesRegressor", "n_estimators", 10),
    (SkKNeighborsRegressor, "KNeighborsRegressor", "n_neighbors", 3),
    (SkRidgeClassifier, "RidgeClassifier", "alpha", 0.5),
    (SkKernelRidge, "KernelRidge", "alpha", 0.5),
    (SkMiniBatchKMeans, "MiniBatchKMeans", "n_clusters", 3),
    (SkAgglomerativeClustering, "AgglomerativeClustering", "n_clusters", 3),
    (SkDBSCAN, "DBSCAN", "eps", 0.3),
    (SkGaussianMixture, "GaussianMixture", "n_components", 2),
    (SkTruncatedSVD, "TruncatedSVD", "n_components", 3),
    (SkFastICA, "FastICA", "n_components", 3),
    (SkNMF, "NMF", "n_components", 3),
    (SkIncrementalPCA, "IncrementalPCA", "n_components", 3),
    (SkKernelPCA, "KernelPCA", "n_components", 3),
    (SkSparsePCA, "SparsePCA", "n_components", 3),
    (SkFactorAnalysis, "FactorAnalysis", "n_components", 3),
]


@pytest.mark.parametrize(
    "sk_cls,name,param,value",
    _POSITIONAL_CASES,
    ids=[c[1] for c in _POSITIONAL_CASES],
)
def test_red_extras_primary_param_positional(sk_cls, name, param, value):
    """RED (REQ-CTOR-ABI-POSITIONAL, R-DEV-2): the primary hyperparameter is
    POSITIONAL_OR_KEYWORD in sklearn 1.5.2, so `Estimator(value)` sets the param.
    ferrolearn makes it keyword-only and raises TypeError on the positional call.

    Live oracle (asserted here, R-CHAR-3): sklearn accepts the param positionally
    and round-trips the value. The same call against ferrolearn must succeed and
    round-trip the value. FAILS until the param moves before the `*` in
    `_extras.py`.
    """
    # --- Live sklearn oracle: the param IS positional-or-keyword and round-trips.
    kind = inspect.signature(sk_cls.__init__).parameters[param].kind
    assert kind == inspect.Parameter.POSITIONAL_OR_KEYWORD, (
        f"oracle invariant broken: sklearn {name}.{param} kind={kind!r}"
    )
    sk_est = sk_cls(value)  # sklearn accepts positionally
    assert getattr(sk_est, param) == value

    # --- ferrolearn must mirror: positional construction sets the same param.
    fl_est = getattr(fl, name)(value)
    assert getattr(fl_est, param) == value


# ---------------------------------------------------------------------------
# GREEN guards: per-category API conformance (must PASS).
# Structure (shapes/types/method surface), not exact RNG values. Deterministic
# classifiers additionally compare ferrolearn accuracy to the live sklearn
# oracle on identical data (R-CHAR-3).
# ---------------------------------------------------------------------------

# Deterministic regression fixture (well-conditioned).
_XR = np.random.RandomState(0).randn(20, 3)
_YR = np.random.RandomState(1).randn(20)

_REGRESSORS = [
    "BayesianRidge",
    "ARDRegression",
    "HuberRegressor",
    "DecisionTreeRegressor",
    "KNeighborsRegressor",
    "KernelRidge",
]


@pytest.mark.parametrize("name", _REGRESSORS)
def test_green_regressor_api_conform(name):
    """GREEN (REQ-REGRESSOR-API-CONFORM): fit(X,y).predict(X) returns finite
    float64 of length n_samples; `score` (RegressorMixin) present."""
    m = getattr(fl, name)().fit(_XR, _YR)
    p = m.predict(_XR)
    assert p.shape == (_XR.shape[0],)
    assert np.issubdtype(np.asarray(p).dtype, np.floating)
    assert np.all(np.isfinite(p))
    assert hasattr(m, "score")  # from RegressorMixin
    assert m.n_features_in_ == _XR.shape[1]


# Classifier fixtures.
# Continuous, linearly-separable (one-vs-rest) 3-class corner blobs for the
# linear / centroid / QDA classifiers.
_XC_CONT, _YC_CONT = make_blobs(
    n_samples=60,
    centers=[[0.0, 0.0], [10.0, 0.0], [5.0, 10.0]],
    cluster_std=0.4,
    random_state=1,
)


def _nb_count_data():
    """Cleanly-separable non-negative COUNT data for MultinomialNB/ComplementNB:
    each class loads a distinct 2-feature block."""
    rng = np.random.RandomState(0)
    rows, ys = [], []
    for c in range(3):
        block = np.zeros((20, 6))
        block[:, c * 2 : (c + 1) * 2] = rng.randint(5, 15, size=(20, 2))
        block += rng.randint(0, 2, size=(20, 6))
        rows.append(block)
        ys.append(np.full(20, c))
    return np.vstack(rows).astype(float), np.concatenate(ys)


def _bernoulli_binary_data():
    """Cleanly-separable BINARY-feature data for BernoulliNB: each class has a
    distinct on/off pattern."""
    patt = [[1, 1, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0], [0, 0, 0, 0, 1, 1]]
    rows, ys = [], []
    for c in range(3):
        for _ in range(20):
            rows.append(patt[c])
            ys.append(c)
    return np.array(rows, dtype=float), np.array(ys)


# (ferrolearn name, sklearn oracle class, X, y) for the DETERMINISTIC classifiers.
_XC_NB, _YC_NB = _nb_count_data()
_XC_BERN, _YC_BERN = _bernoulli_binary_data()

_DETERMINISTIC_CLASSIFIERS = [
    ("RidgeClassifier", SkRidgeClassifier, _XC_CONT, _YC_CONT),
    ("QuadraticDiscriminantAnalysis", SkQDA, _XC_CONT, _YC_CONT),
    ("NearestCentroid", SkNearestCentroid, _XC_CONT, _YC_CONT),
    ("MultinomialNB", SkMultinomialNB, _XC_NB, _YC_NB),
    ("ComplementNB", SkComplementNB, _XC_NB, _YC_NB),
    ("BernoulliNB", SkBernoulliNB, _XC_BERN, _YC_BERN),
]


@pytest.mark.parametrize(
    "name,sk_cls,X,y",
    _DETERMINISTIC_CLASSIFIERS,
    ids=[c[0] for c in _DETERMINISTIC_CLASSIFIERS],
)
def test_green_deterministic_classifier_api_conform(name, sk_cls, X, y):
    """GREEN (REQ-CLASSIFIER-API-CONFORM): fit(X,y).predict(X) returns labels
    in classes_, length n_samples. On cleanly-separable data ferrolearn's
    accuracy EQUALS the live sklearn oracle's accuracy AND is 1.0 (R-CHAR-3 —
    the oracle is the expected value, not a hardcoded constant)."""
    m = getattr(fl, name)().fit(X, y)
    p = np.asarray(m.predict(X))
    assert p.shape == (X.shape[0],)
    assert set(np.unique(p)) <= set(m.classes_)

    sk_acc = (np.asarray(sk_cls().fit(X, y).predict(X)) == y).mean()
    fl_acc = (p == y).mean()
    # ferrolearn matches the sklearn oracle on identical data ...
    assert fl_acc == pytest.approx(sk_acc)
    # ... and the data is cleanly separable for both.
    assert fl_acc == pytest.approx(1.0)


# Non-deterministic / margin classifiers: structural-only guard (no value parity).
_STRUCTURAL_CLASSIFIERS = ["LinearSVC", "ExtraTreeClassifier"]


@pytest.mark.parametrize("name", _STRUCTURAL_CLASSIFIERS)
def test_green_classifier_structural(name):
    """GREEN (REQ-CLASSIFIER-API-CONFORM): fit/predict produce valid labels of
    the right length (structure only — no value parity for the margin/tree
    classifiers)."""
    m = getattr(fl, name)().fit(_XC_CONT, _YC_CONT)
    p = np.asarray(m.predict(_XC_CONT))
    assert p.shape == (_XC_CONT.shape[0],)
    assert set(np.unique(p)) <= set(m.classes_)
    assert m.n_features_in_ == _XC_CONT.shape[1]


def test_green_classifier_string_label_roundtrip():
    """GREEN (REQ-CLASSIFIER-API-CONFORM): the LabelEncoder round-trip maps
    arbitrary (string) label dtypes through the usize-label Rust core and decodes
    back; predict returns the original dtype, classes_ are sorted unique."""
    ystr = np.array(["a", "b", "c"])[_YC_CONT]
    m = fl.NearestCentroid().fit(_XC_CONT, ystr)
    p = np.asarray(m.predict(_XC_CONT))
    assert list(m.classes_) == ["a", "b", "c"]
    assert set(p) <= {"a", "b", "c"}
    assert p.dtype.kind in ("U", "S", "O")  # original (string) dtype preserved


# Clusterer fixtures.
_XCL, _ = make_blobs(n_samples=30, centers=3, cluster_std=0.5, random_state=0)

_CLUSTERERS = ["MiniBatchKMeans", "DBSCAN", "AgglomerativeClustering", "Birch"]


@pytest.mark.parametrize("name", _CLUSTERERS)
def test_green_clusterer_api_conform(name):
    """GREEN (REQ-CLUSTERER-API-CONFORM): fit(X).labels_ has length n_samples
    with >=1 cluster; fit_predict works."""
    est = getattr(fl, name)()
    m = est.fit(_XCL)
    labels = np.asarray(m.labels_)
    assert labels.shape == (_XCL.shape[0],)
    n_clusters = len({int(v) for v in labels if v >= 0})
    assert n_clusters >= 1
    fp = np.asarray(est.fit_predict(_XCL))
    assert fp.shape == (_XCL.shape[0],)


def test_green_gaussian_mixture_api_conform():
    """GREEN (REQ-CLUSTERER-API-CONFORM): GaussianMixture fit(X).predict(X) is
    valid, length n_samples."""
    m = fl.GaussianMixture().fit(_XCL)
    p = np.asarray(m.predict(_XCL))
    assert p.shape == (_XCL.shape[0],)
    assert np.all(p >= 0)


# Transformer fixture (non-negative so NMF-style transformers are valid).
_XT = np.abs(np.random.RandomState(0).randn(20, 4))

_TRANSFORMERS = [
    "MinMaxScaler",
    "MaxAbsScaler",
    "RobustScaler",
    "TruncatedSVD",
    "IncrementalPCA",
    "FactorAnalysis",
]


@pytest.mark.parametrize("name", _TRANSFORMERS)
def test_green_transformer_api_conform(name):
    """GREEN (REQ-TRANSFORMER-API-CONFORM): fit(X).transform(X) returns a 2-D
    array with n_samples rows; fit_transform works."""
    est = getattr(fl, name)()
    m = est.fit(_XT)
    xt = np.asarray(m.transform(_XT))
    assert xt.ndim == 2
    assert xt.shape[0] == _XT.shape[0]
    xft = np.asarray(getattr(fl, name)().fit_transform(_XT))
    assert xft.ndim == 2
    assert xft.shape[0] == _XT.shape[0]
