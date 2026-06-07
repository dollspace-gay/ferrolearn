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


def test_red_hist_gradient_boosting_uses_max_iter_not_n_estimators():
    """REQ #2089: sklearn HistGradientBoostingRegressor uses `max_iter` (the
    number of boosting iterations), NOT `n_estimators`. ferrolearn must mirror
    the sklearn constructor name.

    Live oracle (R-CHAR-3): sklearn HGB has `max_iter` (default 100) and no
    `n_estimators` parameter.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor as SkHGB

    sk_params = inspect.signature(SkHGB.__init__).parameters
    assert "max_iter" in sk_params and "n_estimators" not in sk_params
    assert sk_params["max_iter"].default == 100

    fl_params = inspect.signature(
        fl.HistGradientBoostingRegressor.__init__
    ).parameters
    assert "max_iter" in fl_params and "n_estimators" not in fl_params
    assert fl_params["max_iter"].default == 100

    est = fl.HistGradientBoostingRegressor(max_iter=20)
    assert est.max_iter == 20
    # `n_estimators` is not a sklearn HGB param -> must be rejected.
    with pytest.raises(TypeError):
        fl.HistGradientBoostingRegressor(n_estimators=10)


# ---------------------------------------------------------------------------
# Discrete-NB fitted attributes (#2103): the four `_BaseDiscreteNB` attrs
# (`feature_log_prob_`/`class_log_prior_`/`feature_count_`/`class_count_`,
# sklearn/naive_bayes.py) surfaced on MultinomialNB/BernoulliNB/ComplementNB.
# Expected values come from the LIVE sklearn 1.5.2 oracle in-test (R-CHAR-3),
# never copied from ferrolearn.
# ---------------------------------------------------------------------------

def _assert_discrete_nb_attrs_match(fl_est, sk_est, X, y):
    fl_est.fit(X, y)
    sk_est.fit(X, y)

    for attr in (
        "feature_log_prob_",
        "class_log_prior_",
        "feature_count_",
        "class_count_",
    ):
        assert hasattr(fl_est, attr), f"ferrolearn missing {attr}"
        assert hasattr(sk_est, attr), f"sklearn missing {attr}"
        np.testing.assert_allclose(
            np.asarray(getattr(fl_est, attr)),
            np.asarray(getattr(sk_est, attr)),
            atol=1e-7,
            err_msg=f"{attr} diverges from sklearn",
        )


def test_multinomial_discrete_nb_fitted_attrs_match_sklearn():
    """MultinomialNB exposes feature_log_prob_/class_log_prior_/feature_count_/
    class_count_ matching the live sklearn 1.5.2 oracle (sklearn/naive_bayes.py
    _BaseDiscreteNB attrs)."""
    X = np.array(
        [[2, 1, 0], [0, 3, 1], [1, 1, 2], [0, 0, 4], [3, 0, 1], [1, 2, 2]],
        dtype=np.float64,
    )
    y = np.array([0, 1, 0, 1, 0, 1])
    _assert_discrete_nb_attrs_match(fl.MultinomialNB(), SkMultinomialNB(), X, y)


def test_bernoulli_discrete_nb_fitted_attrs_match_sklearn():
    """BernoulliNB exposes the four `_BaseDiscreteNB` fitted attrs matching the
    live sklearn 1.5.2 oracle."""
    X = np.array(
        [[1, 1, 0], [1, 0, 0], [1, 1, 0], [0, 0, 1], [0, 1, 1], [0, 0, 1]],
        dtype=np.float64,
    )
    y = np.array([0, 0, 0, 1, 1, 1])
    _assert_discrete_nb_attrs_match(fl.BernoulliNB(), SkBernoulliNB(), X, y)


def test_complement_discrete_nb_fitted_attrs_match_sklearn():
    """ComplementNB exposes the four `_BaseDiscreteNB` fitted attrs matching the
    live sklearn 1.5.2 oracle. `feature_log_prob_` is the `-logged` complement
    weight (positive) — sklearn's exposed value (naive_bayes.py:1032-1042)."""
    X = np.array(
        [[5, 1, 0], [4, 2, 0], [6, 0, 1], [0, 1, 5], [1, 0, 4], [0, 2, 6]],
        dtype=np.float64,
    )
    y = np.array([0, 0, 0, 1, 1, 1])
    _assert_discrete_nb_attrs_match(fl.ComplementNB(), SkComplementNB(), X, y)


# ---------------------------------------------------------------------------
# RED pin (#2104): BernoulliNB(binarize=None) divergence.
#
# sklearn documents `binarize : float or None, default=0.0`
# (sklearn/naive_bayes.py:1076); the param constraint admits `None`
# (`"binarize": [None, Interval(Real, 0, None, closed="left")]`,
# naive_bayes.py:1156); and when `binarize is None` sklearn SKIPS binarization
# (`if self.binarize is not None: X = binarize(X, ...)`, naive_bayes.py:1179,
# 1185). Consequently `feature_count_` (one of the four #2103 fitted attrs) is
# accumulated from the RAW, non-binarized X.
#
# ferrolearn's `_RsBernoulliNB` binds `binarize: f64` (extras.rs:613), so the
# `_extras.py::BernoulliNB(binarize=None)` -> `_RsBernoulliNB(binarize=None)`
# call raises `TypeError: argument 'binarize': must be real number, not
# NoneType` at fit time. The sklearn-correct `feature_count_` is unreachable.
# Expected value is computed from the live sklearn oracle in-test (R-CHAR-3).
# ---------------------------------------------------------------------------

def test_red_bernoulli_binarize_none_feature_count_matches_sklearn():
    """RED (#2104): `BernoulliNB(binarize=None)` is valid in sklearn 1.5.2 and
    skips binarization, so `feature_count_` accumulates RAW X
    (sklearn/naive_bayes.py:1076,1156,1179,1185). ferrolearn raises TypeError
    because `_RsBernoulliNB` types `binarize` as f64 (extras.rs:613), so the
    surfaced `feature_count_` for this documented param value is unreachable.

    Live oracle (R-CHAR-3): the expected `feature_count_` is computed from
    sklearn in-test, never copied from ferrolearn.
    """
    # Non-binary (fractional) feature values so binarize=None vs the default
    # binarize=0.0 are OBSERVABLY different in feature_count_.
    X = np.array(
        [
            [0.2, 0.8, 0.0],
            [0.9, 0.1, 0.0],
            [0.7, 0.6, 0.0],
            [0.0, 0.0, 0.9],
            [0.1, 0.7, 0.8],
            [0.0, 0.3, 0.6],
        ],
        dtype=np.float64,
    )
    y = np.array([0, 0, 0, 1, 1, 1])

    # --- Live sklearn oracle: binarize=None is accepted and skips binarization.
    sk = SkBernoulliNB(binarize=None).fit(X, y)
    expected_feature_count = np.asarray(sk.feature_count_)
    # Sanity: with binarize=None this is the raw per-class column sum, which is
    # NOT integral here (distinguishes it from the binarized default path).
    assert not np.array_equal(
        expected_feature_count, np.round(expected_feature_count)
    ), "oracle invariant broken: binarize=None should sum raw (fractional) X"

    # --- ferrolearn must mirror: binarize=None skips binarization and surfaces
    # the same feature_count_. Currently raises TypeError at fit (extras.rs:613).
    fl_est = fl.BernoulliNB(binarize=None).fit(X, y)
    np.testing.assert_allclose(
        np.asarray(fl_est.feature_count_),
        expected_feature_count,
        atol=1e-9,
        err_msg=(
            "BernoulliNB(binarize=None).feature_count_ diverges from sklearn "
            "naive_bayes.py:1179 (None skips binarization)"
        ),
    )


# RED pin (#2104 follow-up): BernoulliNB(binarize=<negative>) parameter validation.
#
# sklearn's `BernoulliNB._parameter_constraints["binarize"]` is
# `[None, Interval(Real, 0, None, closed="left")]` (sklearn/naive_bayes.py:1156),
# enforced by `_validate_params()` at the top of `fit`. A negative threshold is
# OUT of the half-open interval [0, inf), so sklearn raises
# `InvalidParameterError` (a subclass of `ValueError`):
#   "The 'binarize' parameter of BernoulliNB must be None or a float in the
#    range [0.0, inf). Got -1.0 instead."
# The #2104 fix retyped `binarize` to `Option<f64>` (admitting None) but added no
# lower-bound check, so ferrolearn SILENTLY ACCEPTS a negative threshold and fits
# (every value v > -1.0 binarizes to 1), diverging from sklearn's hard reject.
#
# Live oracle (R-CHAR-3): the expectation (sklearn raises ValueError) is observed
# from the live sklearn 1.5.2 oracle in-test, never copied from ferrolearn.
def test_red_bernoulli_negative_binarize_rejected_like_sklearn():
    """RED: `BernoulliNB(binarize=-1.0).fit(X, y)` raises a ValueError in
    sklearn 1.5.2 (binarize constraint Interval[0, inf) or None,
    naive_bayes.py:1156) but ferrolearn accepts it silently and fits."""
    X = np.array(
        [[2.0, 0.0], [0.0, 3.0], [1.0, 1.0], [0.0, 4.0]],
        dtype=np.float64,
    )
    y = np.array([0, 0, 1, 1])

    # --- Live sklearn oracle: a negative binarize threshold is rejected at fit.
    with pytest.raises(ValueError):
        SkBernoulliNB(binarize=-1.0).fit(X, y)

    # --- ferrolearn must mirror the reject. Currently it does NOT raise:
    # `_RsBernoulliNB` (extras.rs:614) and `BernoulliNB::with_binarize_option`
    # (bernoulli.rs:165) / `fit` (bernoulli.rs:266) apply ANY threshold without a
    # lower-bound check, so this fit succeeds -> divergence.
    with pytest.raises(ValueError):
        fl.BernoulliNB(binarize=-1.0).fit(X, y)


# RED pin (#2106): BernoulliNB(binarize=NaN / +inf) parameter validation.
#
# Follow-up to the #2105 fix, which added a `binarize < 0` reject in
# `BernoulliNB::fit` (bernoulli.rs:257-264). That guard mirrors only the LOWER
# bound of sklearn's constraint
#   `binarize: [None, Interval(Real, 0, None, closed="left")]`
#   (sklearn/naive_bayes.py:1156)
# via `b < F::zero()`. But IEEE comparisons make that guard INCOMPLETE:
#   * `NaN < 0.0` is `false`  -> ferrolearn ACCEPTS NaN and fits.
#   * `+inf < 0.0` is `false` -> ferrolearn ACCEPTS +inf and fits.
# sklearn's `Interval(Real, 0, None, closed="left")` is the half-open interval
# [0.0, +inf): NaN is not a member (every comparison is false) and +inf is the
# OPEN right end (`closed="left"` excludes the upper bound), so sklearn's
# `_validate_params()` raises `InvalidParameterError` (a `ValueError`) for BOTH.
# (-inf is correctly rejected by both, since `-inf < 0.0` is true.)
#
# Live oracle (R-CHAR-3): the expectation (sklearn raises ValueError) is observed
# from the live sklearn 1.5.2 oracle in-test, never copied from ferrolearn.
def test_red_bernoulli_nan_binarize_rejected_like_sklearn():
    """RED: `BernoulliNB(binarize=NaN).fit(X, y)` raises a ValueError in sklearn
    1.5.2 (binarize constraint Interval[0, inf) or None, naive_bayes.py:1156;
    NaN is outside the interval) but ferrolearn's `b < 0` guard
    (bernoulli.rs:257) lets `NaN < 0 == false` through and fits. Tracking #2106."""
    X = np.array(
        [[2.0, 0.0], [0.0, 3.0], [1.0, 1.0], [0.0, 4.0]],
        dtype=np.float64,
    )
    y = np.array([0, 0, 1, 1])

    nan = float("nan")
    # --- Live sklearn oracle: NaN binarize is rejected at fit.
    with pytest.raises(ValueError):
        SkBernoulliNB(binarize=nan).fit(X, y)

    # --- ferrolearn must mirror the reject. Currently it does NOT raise.
    with pytest.raises(ValueError):
        fl.BernoulliNB(binarize=nan).fit(X, y)


def test_red_bernoulli_inf_binarize_rejected_like_sklearn():
    """RED: `BernoulliNB(binarize=+inf).fit(X, y)` raises a ValueError in sklearn
    1.5.2 (binarize constraint Interval[0, inf) with closed="left", so the +inf
    upper bound is OPEN/excluded, naive_bayes.py:1156) but ferrolearn's `b < 0`
    guard (bernoulli.rs:257) lets `+inf < 0 == false` through and fits.
    Tracking #2106."""
    X = np.array(
        [[2.0, 0.0], [0.0, 3.0], [1.0, 1.0], [0.0, 4.0]],
        dtype=np.float64,
    )
    y = np.array([0, 0, 1, 1])

    inf = float("inf")
    # --- Live sklearn oracle: +inf binarize is rejected at fit.
    with pytest.raises(ValueError):
        SkBernoulliNB(binarize=inf).fit(X, y)

    # --- ferrolearn must mirror the reject. Currently it does NOT raise.
    with pytest.raises(ValueError):
        fl.BernoulliNB(binarize=inf).fit(X, y)


# ---------------------------------------------------------------------------
# KNeighborsRegressor full constructor surface (#2147, REQ-KNR-CTOR-SURFACE)
#
# `ferrolearn.KNeighborsRegressor` now mirrors sklearn's full
# `(n_neighbors=5, *, weights='uniform', algorithm='auto', leaf_size=30, p=2,
#  metric='minkowski', metric_params=None, n_jobs=None)` surface
# (sklearn/neighbors/_regression.py:178-189). The headline: `weights='distance'`
# CHANGES the prediction (inverse-distance-weighted average) and must match the
# live sklearn 1.5.2 oracle. Every expected value comes from the SAME-test
# sklearn oracle (R-CHAR-3) — none is literal-copied from ferrolearn.
# ---------------------------------------------------------------------------

# Deterministic continuous-target regression fixture (well-conditioned, no ties).
_XKNR = np.random.RandomState(7).randn(50, 4)
_YKNR = np.random.RandomState(8).randn(50)
_XKNR_TEST = np.random.RandomState(9).randn(15, 4)


@pytest.mark.parametrize("weights", ["uniform", "distance"])
def test_knr_weights_predict_parity_vs_sklearn(weights):
    """GREEN (#2147): `KNeighborsRegressor(weights=...)` predict matches the live
    sklearn 1.5.2 oracle for BOTH 'uniform' and 'distance'. The 'distance' case
    is the headline — inverse-distance-weighted neighbor-target average
    (sklearn/neighbors/_regression.py:43-45)."""
    fl_reg = fl.KNeighborsRegressor(n_neighbors=4, weights=weights).fit(_XKNR, _YKNR)
    sk_reg = SkKNeighborsRegressor(n_neighbors=4, weights=weights).fit(_XKNR, _YKNR)

    fl_pred = np.asarray(fl_reg.predict(_XKNR_TEST))
    sk_pred = sk_reg.predict(_XKNR_TEST)  # live oracle (R-CHAR-3)

    np.testing.assert_allclose(fl_pred, sk_pred, rtol=1e-9, atol=1e-9)


def test_knr_distance_weighting_changes_prediction():
    """GREEN (#2147): `weights='distance'` produces a DIFFERENT prediction than
    `weights='uniform'` (the whole point of the param). Oracle-anchored: sklearn
    itself diverges between the two on this fixture, and ferrolearn mirrors that
    divergence."""
    fl_uni = np.asarray(
        fl.KNeighborsRegressor(n_neighbors=4, weights="uniform")
        .fit(_XKNR, _YKNR)
        .predict(_XKNR_TEST)
    )
    fl_dist = np.asarray(
        fl.KNeighborsRegressor(n_neighbors=4, weights="distance")
        .fit(_XKNR, _YKNR)
        .predict(_XKNR_TEST)
    )
    sk_uni = (
        SkKNeighborsRegressor(n_neighbors=4, weights="uniform")
        .fit(_XKNR, _YKNR)
        .predict(_XKNR_TEST)
    )
    sk_dist = (
        SkKNeighborsRegressor(n_neighbors=4, weights="distance")
        .fit(_XKNR, _YKNR)
        .predict(_XKNR_TEST)
    )
    # Live oracle: the two weighting schemes diverge.
    assert not np.allclose(sk_uni, sk_dist)
    # ferrolearn mirrors the same divergence.
    assert not np.allclose(fl_uni, fl_dist)


@pytest.mark.parametrize("algorithm", ["auto", "brute", "kd_tree", "ball_tree"])
def test_knr_algorithm_is_search_strategy_only(algorithm):
    """GREEN (#2147): `algorithm` is a search-strategy knob only — every value
    gives the SAME predict, identical to the live sklearn oracle
    (sklearn/neighbors/_regression.py:56-66)."""
    fl_pred = np.asarray(
        fl.KNeighborsRegressor(n_neighbors=5, algorithm=algorithm)
        .fit(_XKNR, _YKNR)
        .predict(_XKNR_TEST)
    )
    sk_pred = (
        SkKNeighborsRegressor(n_neighbors=5, algorithm=algorithm)
        .fit(_XKNR, _YKNR)
        .predict(_XKNR_TEST)
    )
    np.testing.assert_allclose(fl_pred, sk_pred, rtol=1e-9, atol=1e-9)


def test_knr_get_params_exposes_full_surface_like_sklearn():
    """GREEN (#2147): `get_params()` exposes all 8 sklearn constructor params
    (sklearn `clone()` compatibility). Oracle: the param NAME SET equals
    sklearn's."""
    sk_params = set(SkKNeighborsRegressor().get_params())  # live oracle
    fl_params = set(fl.KNeighborsRegressor().get_params())
    assert fl_params == sk_params
    assert len(fl_params) == 8


def test_knr_set_params_clone_roundtrip():
    """GREEN (#2147): `clone()`/`set_params()` round-trip preserves all params,
    including the result-affecting `weights='distance'` (sklearn clone protocol
    re-reads `get_params()`)."""
    from sklearn.base import clone

    est = fl.KNeighborsRegressor(
        3, weights="distance", algorithm="kd_tree", leaf_size=10, p=2, n_jobs=2
    )
    cloned = clone(est)
    assert cloned.weights == "distance"
    assert cloned.n_neighbors == 3
    assert cloned.algorithm == "kd_tree"
    assert cloned.leaf_size == 10
    assert cloned.n_jobs == 2

    est.set_params(weights="uniform", n_neighbors=7)
    assert est.weights == "uniform"
    assert est.n_neighbors == 7


@pytest.mark.parametrize(
    "kwargs",
    [
        {"p": 3},
        {"metric": "manhattan"},
        {"weights": (lambda d: np.ones_like(d))},
        {"metric_params": {"w": [1.0, 1.0, 1.0, 1.0]}},
    ],
    ids=["p3", "metric_manhattan", "weights_callable", "metric_params"],
)
def test_knr_unsupported_params_raise_notimplemented(kwargs):
    """GREEN (#2147): params the Euclidean-only Rust core cannot honor raise
    NotImplementedError at fit (mirrors `KNeighborsClassifier`; NOT-STARTED
    #876). The live sklearn oracle ACCEPTS these (asserted) — ferrolearn's
    reject is the explicit, honest divergence, not a silent wrong answer."""
    # --- Live oracle: sklearn fits with these params (no error).
    SkKNeighborsRegressor(n_neighbors=4, **kwargs).fit(_XKNR, _YKNR)
    # --- ferrolearn raises NotImplementedError (clear, not a wrong result).
    with pytest.raises(NotImplementedError):
        fl.KNeighborsRegressor(n_neighbors=4, **kwargs).fit(_XKNR, _YKNR)


def test_knr_pickle_roundtrip_preserves_distance_predictions():
    """GREEN (#2147): a fitted `weights='distance'` regressor survives pickling
    (the Rust `_rs` is dropped and rebuilt by re-fit on unpickle, mirroring the
    classifier pickle mixin) and reproduces identical predictions."""
    import pickle

    reg = fl.KNeighborsRegressor(n_neighbors=4, weights="distance").fit(_XKNR, _YKNR)
    before = np.asarray(reg.predict(_XKNR_TEST))
    restored = pickle.loads(pickle.dumps(reg))
    after = np.asarray(restored.predict(_XKNR_TEST))
    np.testing.assert_allclose(after, before, rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# HuberRegressor: sample_weight (#501) + warm_start (#500)
# ---------------------------------------------------------------------------

from sklearn.linear_model import HuberRegressor as SkHuberRegressor


def _huber_outlier_data():
    rng = np.random.RandomState(0)
    X = rng.randn(50, 3)
    y = X @ np.array([1.0, 2.0, -1.0]) + 0.1 * rng.randn(50)
    y[:5] += 20
    return X, y


def test_huber_sample_weight_matches_sklearn():
    """REQ-11 (#501): `fl.HuberRegressor().fit(X, y, sample_weight=w)` reproduces
    sklearn's weighted `coef_`/`intercept_`/`scale_`. Oracle computed live in the
    SAME test (R-CHAR-3)."""
    X, y = _huber_outlier_data()
    rng = np.random.RandomState(42)
    w = np.abs(rng.randn(len(y))) + 0.5

    sk = SkHuberRegressor().fit(X, y, sample_weight=w)
    fr = fl.HuberRegressor().fit(X, y, sample_weight=w)

    np.testing.assert_allclose(np.asarray(fr.coef_), sk.coef_, atol=1e-3)
    assert abs(fr.intercept_ - sk.intercept_) < 1e-3
    assert abs(fr.scale_ - sk.scale_) < 1e-3


def test_huber_sample_weight_none_equals_default():
    """REQ-11: `sample_weight=None` is byte-identical to the unweighted fit."""
    X, y = _huber_outlier_data()
    a = fl.HuberRegressor().fit(X, y)
    b = fl.HuberRegressor().fit(X, y, sample_weight=None)
    np.testing.assert_array_equal(np.asarray(a.coef_), np.asarray(b.coef_))
    assert a.intercept_ == b.intercept_
    assert a.scale_ == b.scale_


def test_huber_warm_start_same_optimum_matches_sklearn():
    """REQ-10 (#500): a `warm_start=True` refit reaches the SAME optimum as the
    cold fit (Huber is convex), matching the live sklearn oracle. sklearn's
    warm refit drops `n_iter_` from 15 to 1; we assert the fitted attributes are
    unchanged and still match sklearn (the value contract, R-DEV-7)."""
    X, y = _huber_outlier_data()

    sk = SkHuberRegressor(warm_start=True)
    sk.fit(X, y)
    sk.fit(X, y)  # warm refit

    fr = fl.HuberRegressor(warm_start=True)
    fr.fit(X, y)
    cold_coef = np.asarray(fr.coef_).copy()
    fr.fit(X, y)  # warm refit reuses the prior fit's coef_/intercept_/scale_

    np.testing.assert_allclose(np.asarray(fr.coef_), sk.coef_, atol=1e-3)
    assert abs(fr.intercept_ - sk.intercept_) < 1e-3
    assert abs(fr.scale_ - sk.scale_) < 1e-3
    # Convex unique minimum: warm refit unchanged from cold.
    np.testing.assert_allclose(np.asarray(fr.coef_), cold_coef, atol=1e-3)


def test_huber_warm_start_keyword_only_like_sklearn():
    """REQ-10 (R-DEV-2): `warm_start` is keyword-only, matching sklearn's
    `def __init__(self, *, ..., warm_start=False, ...)` (`_huber.py:259-268`)."""
    sig = inspect.signature(fl.HuberRegressor.__init__)
    assert sig.parameters["warm_start"].kind == inspect.Parameter.KEYWORD_ONLY
    assert sig.parameters["warm_start"].default is False


def test_huber_n_iter_attribute_positive_bounded():
    """REQ-9 (#499): `fl.HuberRegressor().fit(...).n_iter_` is exposed (sklearn
    `n_iter_`, `_huber.py:342` `self.n_iter_ = opt_res.nit`) as a positive int
    `<= max_iter`. R-DEV-7: the Rust core is not scipy's L-BFGS-B, so the raw
    count need not equal sklearn's exactly — we assert the same positivity/bound
    contract sklearn satisfies, not equality."""
    X, y = _huber_outlier_data()
    max_iter = 200
    fr = fl.HuberRegressor(max_iter=max_iter).fit(X, y)
    sk = SkHuberRegressor(max_iter=max_iter).fit(X, y)

    assert isinstance(fr.n_iter_, int)
    assert fr.n_iter_ >= 1
    assert fr.n_iter_ <= max_iter
    # sklearn's own n_iter_ is likewise a positive int <= max_iter (the contract
    # both share; the exact counts differ across the two L-BFGS implementations).
    assert int(sk.n_iter_) >= 1


def test_huber_n_iter_warm_start_fewer_than_cold_like_sklearn():
    """REQ-9 (#499): a warm-start refit reports FEWER `n_iter_` than the cold fit
    — the oracle-comparable property sklearn also exhibits (live sklearn 1.5.2:
    cold `n_iter_`=15, warm=1; `_huber.py:308-309`/`:342`). R-CHAR-3: the sklearn
    inequality is computed live in this test, not copied."""
    X, y = _huber_outlier_data()

    # sklearn oracle: warm refit n_iter_ < cold n_iter_.
    sk = SkHuberRegressor(warm_start=True)
    sk.fit(X, y)
    sk_cold_iters = int(sk.n_iter_)
    sk.fit(X, y)
    sk_warm_iters = int(sk.n_iter_)
    assert sk_warm_iters < sk_cold_iters, (
        f"sklearn oracle expected warm < cold, got warm={sk_warm_iters} "
        f"cold={sk_cold_iters}"
    )

    # ferrolearn must exhibit the same warm < cold property.
    fr = fl.HuberRegressor(warm_start=True)
    fr.fit(X, y)
    fr_cold_iters = fr.n_iter_
    fr.fit(X, y)  # warm refit reuses the prior fit's coef_/intercept_/scale_
    fr_warm_iters = fr.n_iter_

    assert fr_warm_iters < fr_cold_iters, (
        f"ferrolearn warm n_iter_ ({fr_warm_iters}) must be < cold "
        f"({fr_cold_iters}) (sklearn: cold={sk_cold_iters}, warm={sk_warm_iters})"
    )
    assert fr_warm_iters >= 1


# ===========================================================================
# BayesianRidge REQ-6/7/8/9 (#2161): compute_score/scores_, n_iter_,
# predict(return_std=True)/sigma_, sample_weight. Oracle = live sklearn 1.5.2
# `sklearn.linear_model.BayesianRidge` (imported here), compared head-to-head.
# ===========================================================================

from sklearn.linear_model import BayesianRidge as SkBayesianRidge


def _bayes_dataset(seed=0, n=30, p=5):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, p)
    y = X @ np.array([3.0, -1.8, 0.0, 0.5, 1.2][:p]) + 0.5 * rng.randn(n)
    return X, y


def test_bayesian_ridge_n_iter_matches_sklearn():
    """REQ-7: `n_iter_` matches the live sklearn oracle (positive int <= max_iter,
    sklearn `_bayes.py:316`)."""
    X, y = _bayes_dataset()
    sk = SkBayesianRidge().fit(X, y)
    fr = fl.BayesianRidge().fit(X, y)
    assert fr.n_iter_ == sk.n_iter_, (
        f"n_iter_: ferrolearn={fr.n_iter_}, sklearn={sk.n_iter_}"
    )
    assert 1 <= fr.n_iter_ <= 300


def test_bayesian_ridge_scores_matches_sklearn():
    """REQ-6: `compute_score=True` populates `scores_` (length n_iter_+1) with the
    log marginal likelihood; the converged (final) value matches the sklearn
    oracle tightly and the full sequence to a documented tolerance
    (sklearn `_bayes.py:302/330`)."""
    X, y = _bayes_dataset()
    sk = SkBayesianRidge(compute_score=True).fit(X, y)
    fr = fl.BayesianRidge(compute_score=True).fit(X, y)

    assert len(fr.scores_) == len(sk.scores_)
    assert len(fr.scores_) == fr.n_iter_ + 1
    # Final converged LML: tight.
    assert abs(fr.scores_[-1] - sk.scores_[-1]) < 1e-6 * abs(sk.scores_[-1]), (
        f"final scores_: ferrolearn={fr.scores_[-1]}, sklearn={sk.scores_[-1]}"
    )
    # Per-iteration LML: looser (EM trajectory is path-sensitive).
    np.testing.assert_allclose(fr.scores_, sk.scores_, rtol=1e-4, atol=1e-4)


def test_bayesian_ridge_scores_empty_without_compute_score():
    """REQ-6: with compute_score=False (default), `scores_` is empty — matching
    sklearn which only populates it under the flag (`_bayes.py:198`)."""
    X, y = _bayes_dataset()
    fr = fl.BayesianRidge().fit(X, y)
    assert fr.scores_.size == 0


def test_bayesian_ridge_return_std_matches_sklearn():
    """REQ-8: `predict(X, return_std=True)` returns `(mean, std)` matching the
    sklearn oracle; the mean equals the plain predict (`_bayes.py:367-371`)."""
    X, y = _bayes_dataset()
    sk = SkBayesianRidge().fit(X, y)
    fr = fl.BayesianRidge().fit(X, y)

    sk_mean, sk_std = sk.predict(X, return_std=True)
    fr_mean, fr_std = fr.predict(X, return_std=True)

    np.testing.assert_allclose(fr_mean, sk_mean, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(fr_std, sk_std, rtol=1e-5, atol=1e-8)

    # return_std=False returns a bare mean array equal to the (mean, std) mean.
    fr_mean_only = fr.predict(X)
    np.testing.assert_allclose(fr_mean_only, fr_mean, rtol=0, atol=1e-12)


def test_bayesian_ridge_sigma_full_matches_sklearn():
    """REQ-8: the fitted `sigma_` is the full (n_features, n_features) posterior
    covariance matrix matching the sklearn oracle (`_bayes.py:333-337`)."""
    X, y = _bayes_dataset()
    sk = SkBayesianRidge().fit(X, y)
    fr = fl.BayesianRidge().fit(X, y)
    assert fr.sigma_.shape == sk.sigma_.shape == (5, 5)
    np.testing.assert_allclose(fr.sigma_, sk.sigma_, rtol=1e-5, atol=1e-9)


def test_bayesian_ridge_sample_weight_matches_sklearn():
    """REQ-9: a weighted fit (`fit(X, y, sample_weight=...)`) matches the sklearn
    oracle's coef_/intercept_/alpha_/lambda_ (`_bayes.py:254-256`)."""
    X, y = _bayes_dataset()
    sw = np.abs(np.sin(np.arange(X.shape[0]) * 0.7)) + 0.1

    sk = SkBayesianRidge().fit(X, y, sample_weight=sw)
    fr = fl.BayesianRidge().fit(X, y, sample_weight=sw)

    np.testing.assert_allclose(fr.coef_, sk.coef_, rtol=1e-3, atol=1e-6)
    assert abs(fr.intercept_ - sk.intercept_) < 1e-3
    assert abs(fr.alpha_ - sk.alpha_) / abs(sk.alpha_) < 1e-2
    assert abs(fr.lambda_ - sk.lambda_) / abs(sk.lambda_) < 1e-2


def test_bayesian_ridge_sample_weight_none_matches_unweighted():
    """REQ-9: `sample_weight=None` is byte-identical to the unweighted fit."""
    X, y = _bayes_dataset()
    a = fl.BayesianRidge().fit(X, y, sample_weight=None)
    b = fl.BayesianRidge().fit(X, y)
    np.testing.assert_array_equal(a.coef_, b.coef_)
    assert a.intercept_ == b.intercept_


# ===========================================================================
# ARDRegression REQ-6/7/8 (#2163): compute_score/scores_, n_iter_,
# predict(return_std=True)/sigma_ (kept-feature covariance). Oracle = live
# sklearn 1.5.2 `sklearn.linear_model.ARDRegression`, compared head-to-head.
# ===========================================================================

from sklearn.linear_model import ARDRegression as SkARDRegression


def _ard_dataset():
    # 4-feature mixed-relevance design (y = 2*x0 + 0.5*x2; features 1, 3 are
    # irrelevant and get pruned), so sigma_ is the kept (2,2) block.
    X = np.array(
        [
            [1.0, 50.0, 2.0, -3.0],
            [2.0, 10.0, 4.0, 1.0],
            [3.0, 90.0, 6.0, -7.0],
            [4.0, 20.0, 8.0, 5.0],
            [5.0, 70.0, 10.0, -2.0],
            [6.0, 40.0, 12.0, 9.0],
            [7.0, 60.0, 14.0, -1.0],
            [8.0, 30.0, 16.0, 4.0],
        ]
    )
    y = 2.0 * X[:, 0] + 0.5 * X[:, 2]
    return X, y


def test_ard_n_iter_matches_sklearn():
    """REQ-7: `n_iter_` matches the live sklearn oracle (`_bayes.py:716`,
    `self.n_iter_ = iter_ + 1`)."""
    X, y = _ard_dataset()
    sk = SkARDRegression(max_iter=1000).fit(X, y)
    fr = fl.ARDRegression(max_iter=1000).fit(X, y)
    assert fr.n_iter_ == sk.n_iter_, (
        f"n_iter_: ferrolearn={fr.n_iter_}, sklearn={sk.n_iter_}"
    )
    assert 1 <= fr.n_iter_ <= 1000


def test_ard_scores_matches_sklearn():
    """REQ-6: `compute_score=True` populates `scores_` (length n_iter_, NO
    post-loop append — ARD appends inside the loop before the convergence break)
    with the ARD objective per iteration, matching the sklearn oracle tightly
    when the EM iterates align (`_bayes.py:695-704`)."""
    X, y = _ard_dataset()
    sk = SkARDRegression(max_iter=1000, compute_score=True).fit(X, y)
    fr = fl.ARDRegression(max_iter=1000, compute_score=True).fit(X, y)

    assert len(fr.scores_) == len(sk.scores_)
    # ARD: len(scores_) == n_iter_ (unlike BayesianRidge's n_iter_+1).
    assert len(fr.scores_) == fr.n_iter_
    np.testing.assert_allclose(fr.scores_, sk.scores_, rtol=1e-7, atol=1e-7)


def test_ard_scores_empty_without_compute_score():
    """REQ-6: with compute_score=False (default), `scores_` is empty — matching
    sklearn which only populates it under the flag (`_bayes.py:587`)."""
    X, y = _ard_dataset()
    fr = fl.ARDRegression(max_iter=1000).fit(X, y)
    assert fr.scores_.size == 0


def test_ard_return_std_matches_sklearn():
    """REQ-8: `predict(X, return_std=True)` returns `(mean, std)` matching the
    sklearn oracle (variance over the KEPT columns only, `_bayes.py:787-790`),
    including an out-of-range query point. The mean equals the plain predict."""
    X, y = _ard_dataset()
    sk = SkARDRegression(max_iter=1000).fit(X, y)
    fr = fl.ARDRegression(max_iter=1000).fit(X, y)

    # Training rows + an out-of-range query (3x the last row).
    Xq = np.vstack([X[:3], X[-1] * 3.0])
    sk_mean, sk_std = sk.predict(Xq, return_std=True)
    fr_mean, fr_std = fr.predict(Xq, return_std=True)

    np.testing.assert_allclose(fr_mean, sk_mean, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(fr_std, sk_std, rtol=1e-5, atol=1e-8)

    # return_std=False returns a bare mean array equal to the (mean, std) mean.
    fr_mean_only = fr.predict(Xq)
    np.testing.assert_allclose(fr_mean_only, fr_mean, rtol=0, atol=1e-12)


def test_ard_sigma_matches_sklearn():
    """REQ-8: the fitted `sigma_` is the KEPT-feature (n_kept, n_kept) posterior
    covariance matching the sklearn oracle (`_bayes.py:727`). On this design
    features 1, 3 are pruned, so sigma_ is (2, 2), not (4, 4)."""
    X, y = _ard_dataset()
    sk = SkARDRegression(max_iter=1000).fit(X, y)
    fr = fl.ARDRegression(max_iter=1000).fit(X, y)
    assert fr.sigma_.shape == sk.sigma_.shape == (2, 2)
    np.testing.assert_allclose(fr.sigma_, sk.sigma_, rtol=1e-5, atol=1e-9)


# ===========================================================================
# QuantileRegressor REQ-6 (n_iter_) + REQ-7 (solver / solver_options) (#507/#508)
# ---------------------------------------------------------------------------
# Live sklearn 1.5.2 oracle: `sklearn.linear_model.QuantileRegressor`
# (`_quantile.py:128-141`/`:300`). The HiGHS family + "revised simplex" all
# reach the SAME unique LP vertex; "interior-point" raises ValueError
# (removed in scipy>=1.11); an invalid solver string raises a ValueError
# (InvalidParameterError ⊂ ValueError). `solver_options` is `[dict, None]`.
# R-DEV-7: ferrolearn's `n_iter_` is the honest two-phase-simplex pivot count,
# NOT scipy's `result.nit`, so it is a positive int <= max_iter and
# deterministic — NOT asserted == sklearn.
# ===========================================================================

from sklearn.linear_model import QuantileRegressor as SkQuantileRegressor


def _quantile_dataset():
    rng = np.random.RandomState(0)
    X = rng.randn(30, 3)
    y = X @ np.array([1.0, 2.0, -1.0]) + 0.5 * rng.randn(30)
    return X, y


def test_quantile_default_solver_predict_matches_sklearn():
    """REQ-1 regression guard + REQ-7 default: with the default solver="highs"
    ferrolearn's predictions match the live sklearn HiGHS LP optimum
    (`_quantile.py:300-304`) to 1e-6 — the new ctor params did not perturb the
    fit. (The `_extras.py` wrapper exposes the fit via `predict`; `coef_` is the
    separately-tracked REQ-MISSING-METHODS surface.) `predict = X@coef_ +
    intercept_`, so prediction parity pins coef_/intercept_ parity."""
    X, y = _quantile_dataset()
    sk = SkQuantileRegressor(quantile=0.8, alpha=0.0).fit(X, y)
    fr = fl.QuantileRegressor(quantile=0.8, alpha=0.0).fit(X, y)
    np.testing.assert_allclose(fr.predict(X), sk.predict(X), rtol=0, atol=1e-6)


def test_quantile_highs_family_solvers_give_identical_predict():
    """REQ-7: solver in {"highs","highs-ds","highs-ipm","revised simplex"} all
    reach the SAME LP vertex (verified live: sklearn returns identical
    predictions for all of them), so ferrolearn's predictions are unchanged
    across them and equal the default fit (`_quantile.py:114-124`/`:250`)."""
    X, y = _quantile_dataset()
    # Live oracle: all four solvers give the same sklearn prediction.
    sk_default = SkQuantileRegressor(quantile=0.5, alpha=0.0).fit(X, y)
    fr_default = fl.QuantileRegressor(quantile=0.5, alpha=0.0).fit(X, y)
    np.testing.assert_allclose(fr_default.predict(X), sk_default.predict(X),
                               rtol=0, atol=1e-6)

    for solver in ("highs", "highs-ds", "highs-ipm", "revised simplex"):
        sk = SkQuantileRegressor(quantile=0.5, alpha=0.0, solver=solver).fit(X, y)
        # sklearn itself returns the same vertex across these solvers.
        np.testing.assert_allclose(sk.predict(X), sk_default.predict(X),
                                   rtol=0, atol=1e-6)
        fr = fl.QuantileRegressor(quantile=0.5, alpha=0.0, solver=solver).fit(X, y)
        np.testing.assert_allclose(fr.predict(X), fr_default.predict(X),
                                   rtol=0, atol=1e-9)


def test_quantile_invalid_solver_raises_valueerror():
    """REQ-7: an invalid solver string raises ValueError in BOTH sklearn
    (InvalidParameterError ⊂ ValueError, `_quantile.py:114-124`) and
    ferrolearn."""
    X, y = _quantile_dataset()
    with pytest.raises(ValueError):
        SkQuantileRegressor(solver="not-a-solver").fit(X, y)
    with pytest.raises(ValueError):
        fl.QuantileRegressor(solver="not-a-solver").fit(X, y)


def test_quantile_interior_point_raises_valueerror():
    """REQ-7: "interior-point" was removed in scipy>=1.11; both sklearn
    (`_quantile.py:196-199`) and ferrolearn raise ValueError."""
    X, y = _quantile_dataset()
    with pytest.raises(ValueError):
        SkQuantileRegressor(solver="interior-point").fit(X, y)
    with pytest.raises(ValueError):
        fl.QuantileRegressor(solver="interior-point").fit(X, y)


def test_quantile_solver_options_none_and_empty_accepted():
    """REQ-7: `solver_options=None` and an empty dict are accepted (HiGHS tuning
    knobs that do not change the optimum, `_quantile.py:206`); the fit (probed
    via `predict`) is unchanged and still matches the sklearn oracle."""
    X, y = _quantile_dataset()
    sk = SkQuantileRegressor(quantile=0.5, alpha=0.0, solver_options=None).fit(X, y)
    fr_none = fl.QuantileRegressor(quantile=0.5, alpha=0.0,
                                   solver_options=None).fit(X, y)
    fr_empty = fl.QuantileRegressor(quantile=0.5, alpha=0.0,
                                    solver_options={}).fit(X, y)
    np.testing.assert_allclose(fr_none.predict(X), sk.predict(X), rtol=0, atol=1e-6)
    np.testing.assert_allclose(fr_empty.predict(X), fr_none.predict(X),
                               rtol=0, atol=1e-12)


def test_quantile_nonempty_solver_options_accepted_matches_sklearn():
    """REQ-7 (#2168): sklearn's `solver_options` is `[dict, None]`
    (`_quantile.py:125`), so a non-empty dict is a valid construction and sklearn
    fits successfully — the options (`presolve`, `disp`, ...) are HiGHS tuning
    knobs that do NOT change the LP optimum. ferrolearn must mirror that: it
    ACCEPTS the dict, ignores it, and returns the SAME fit it does with
    `solver_options=None`, matching the live sklearn oracle (R-CHAR-3)."""
    X, y = _quantile_dataset()
    sk = SkQuantileRegressor(
        quantile=0.5, alpha=0.0, solver_options={"presolve": True}
    ).fit(X, y)
    fr = fl.QuantileRegressor(
        quantile=0.5, alpha=0.0, solver_options={"presolve": True}
    ).fit(X, y)
    np.testing.assert_allclose(fr.predict(X), sk.predict(X), rtol=0, atol=1e-6)
    # ... and identical to the solver_options=None fit (options are ignored).
    fr_none = fl.QuantileRegressor(quantile=0.5, alpha=0.0).fit(X, y)
    np.testing.assert_allclose(fr.predict(X), fr_none.predict(X), rtol=0, atol=1e-12)


def test_quantile_n_iter_contract():
    """REQ-6: `n_iter_` is a positive int <= max_iter and deterministic. R-DEV-7:
    ferrolearn's two-phase-simplex pivot count is NOT scipy's HiGHS `result.nit`
    (`_quantile.py:300`), so we assert the CONTRACT, not equality with sklearn.
    sklearn's own n_iter_ is read only to confirm both are positive ints."""
    X, y = _quantile_dataset()
    max_iter = 20000
    fr = fl.QuantileRegressor(quantile=0.5, alpha=0.0, max_iter=max_iter).fit(X, y)
    assert isinstance(fr.n_iter_, int)
    assert 0 < fr.n_iter_ <= max_iter
    # Deterministic across refits.
    fr2 = fl.QuantileRegressor(quantile=0.5, alpha=0.0, max_iter=max_iter).fit(X, y)
    assert fr.n_iter_ == fr2.n_iter_
    # sklearn also reports a positive int (different solver, different value).
    sk = SkQuantileRegressor(quantile=0.5, alpha=0.0).fit(X, y)
    assert int(sk.n_iter_) > 0


# ===========================================================================
# OrthogonalMatchingPursuit (#2172): the binding-surface unit. Oracle = live
# sklearn 1.5.2 `sklearn.linear_model.OrthogonalMatchingPursuit` (imported here),
# compared head-to-head. coef_/intercept_/predict per several `n_nonzero_coefs`,
# the `tol` path, the default (None/None -> max(int(0.1*n_features),1)),
# fit_intercept True/False, the 4-param keyword-only ctor ABI / clone, and
# precompute='auto'/False/True acceptance (Gram knob, no result change). All
# expected values are computed by the live sklearn oracle in the SAME test
# (R-CHAR-3) — never copied from ferrolearn.
# ===========================================================================

from sklearn.linear_model import (
    OrthogonalMatchingPursuit as SkOrthogonalMatchingPursuit,
)


def _omp_dataset(seed=0, n=80, p=12, informative=5, noise=4.0):
    from sklearn.datasets import make_regression

    X, y = make_regression(
        n_samples=n, n_features=p, n_informative=informative, noise=noise,
        random_state=seed,
    )
    return X, y


def test_omp_coef_intercept_predict_match_sklearn_per_n_nonzero():
    """REQ-OMP: for several `n_nonzero_coefs`, `coef_`/`intercept_`/`predict`
    reproduce the live sklearn oracle to ~1e-12 (omp.rs REQ-1/2/5 SHIPPED)."""
    X, y = _omp_dataset()
    for k in (1, 2, 3, 5, 8):
        sk = SkOrthogonalMatchingPursuit(n_nonzero_coefs=k).fit(X, y)
        fr = fl.OrthogonalMatchingPursuit(n_nonzero_coefs=k).fit(X, y)
        np.testing.assert_allclose(
            np.asarray(fr.coef_), sk.coef_, rtol=0, atol=1e-10,
            err_msg=f"coef_ diverges at n_nonzero_coefs={k}",
        )
        assert abs(fr.intercept_ - sk.intercept_) < 1e-10, (
            f"intercept_ diverges at n_nonzero_coefs={k}"
        )
        np.testing.assert_allclose(
            fr.predict(X), sk.predict(X), rtol=0, atol=1e-9,
            err_msg=f"predict diverges at n_nonzero_coefs={k}",
        )


def test_omp_default_n_nonzero_matches_sklearn():
    """REQ-OMP: with `n_nonzero_coefs=None` AND `tol=None`, sklearn sets
    `n_nonzero_coefs_ = max(int(0.1 * n_features), 1)` (`_omp.py:785`) and fits;
    ferrolearn reproduces the SAME `coef_`/`intercept_`/`predict` (omp.rs REQ-2)."""
    X, y = _omp_dataset(p=12)
    sk = SkOrthogonalMatchingPursuit().fit(X, y)
    fr = fl.OrthogonalMatchingPursuit().fit(X, y)
    # sklearn's effective support size is the documented default (oracle, R-CHAR-3).
    assert sk.n_nonzero_coefs_ == max(int(0.1 * X.shape[1]), 1)
    np.testing.assert_allclose(np.asarray(fr.coef_), sk.coef_, rtol=0, atol=1e-10)
    assert abs(fr.intercept_ - sk.intercept_) < 1e-10
    np.testing.assert_allclose(fr.predict(X), sk.predict(X), rtol=0, atol=1e-9)
    # Exactly one non-zero coefficient (12 features -> int(1.2) -> 1).
    assert int(np.count_nonzero(np.abs(np.asarray(fr.coef_)) > 1e-10)) == 1


def test_omp_tol_path_matches_sklearn():
    """REQ-OMP: the `tol` stopping path (sklearn `tol` overrides
    `n_nonzero_coefs`, `_omp.py:786-789`) reproduces the sklearn oracle's
    `coef_`/`predict`."""
    X, y = _omp_dataset()
    tol = float(np.sum((y - y.mean()) ** 2) * 0.05)
    sk = SkOrthogonalMatchingPursuit(tol=tol).fit(X, y)
    fr = fl.OrthogonalMatchingPursuit(tol=tol).fit(X, y)
    np.testing.assert_allclose(np.asarray(fr.coef_), sk.coef_, rtol=0, atol=1e-9)
    assert abs(fr.intercept_ - sk.intercept_) < 1e-9
    np.testing.assert_allclose(fr.predict(X), sk.predict(X), rtol=0, atol=1e-8)


def test_omp_fit_intercept_true_false_match_sklearn():
    """REQ-OMP: both `fit_intercept=True` and `fit_intercept=False` reproduce the
    sklearn oracle (`_omp.py:660-663`/`:815`)."""
    X, y = _omp_dataset()
    for fit_intercept in (True, False):
        sk = SkOrthogonalMatchingPursuit(
            n_nonzero_coefs=4, fit_intercept=fit_intercept).fit(X, y)
        fr = fl.OrthogonalMatchingPursuit(
            n_nonzero_coefs=4, fit_intercept=fit_intercept).fit(X, y)
        np.testing.assert_allclose(
            np.asarray(fr.coef_), sk.coef_, rtol=0, atol=1e-10,
            err_msg=f"coef_ diverges at fit_intercept={fit_intercept}",
        )
        assert abs(fr.intercept_ - sk.intercept_) < 1e-10
        if not fit_intercept:
            assert fr.intercept_ == 0.0


def test_omp_precompute_accepted_no_result_change():
    """REQ-OMP: sklearn's `precompute='auto'`/`True`/`False` (`_omp.py:665-669`,
    `_parameter_constraints` `:739`) is a Gram-matrix speed knob that does NOT
    change the OMP solution (`_omp.py:791-813`). ferrolearn accepts every value
    and yields the IDENTICAL fit as the `precompute='auto'` default (and matches
    the sklearn oracle, which is itself precompute-invariant)."""
    X, y = _omp_dataset()
    sk = SkOrthogonalMatchingPursuit(n_nonzero_coefs=4).fit(X, y)
    base = fl.OrthogonalMatchingPursuit(n_nonzero_coefs=4, precompute='auto').fit(X, y)
    for pc in ('auto', False, True):
        fr = fl.OrthogonalMatchingPursuit(n_nonzero_coefs=4, precompute=pc).fit(X, y)
        # Identical to the 'auto' fit (precompute is ignored / result-invariant).
        np.testing.assert_array_equal(
            np.asarray(fr.coef_), np.asarray(base.coef_),
        )
        assert fr.intercept_ == base.intercept_
        # ... and still matches the sklearn oracle.
        np.testing.assert_allclose(
            np.asarray(fr.coef_), sk.coef_, rtol=0, atol=1e-10,
            err_msg=f"coef_ diverges at precompute={pc!r}",
        )


def test_omp_ctor_abi_keyword_only_four_params_like_sklearn():
    """REQ-OMP (R-DEV-2): the constructor exposes exactly sklearn's four
    keyword-only params with sklearn's defaults/order
    (`_omp.py:742-753`: `(*, n_nonzero_coefs=None, tol=None, fit_intercept=True,
    precompute='auto')`)."""
    sig = inspect.signature(fl.OrthogonalMatchingPursuit.__init__)
    params = [p for p in sig.parameters if p != "self"]
    assert params == ["n_nonzero_coefs", "tol", "fit_intercept", "precompute"]
    for name in params:
        assert sig.parameters[name].kind == inspect.Parameter.KEYWORD_ONLY, (
            f"{name} must be keyword-only (sklearn leading `*`)"
        )
    assert sig.parameters["n_nonzero_coefs"].default is None
    assert sig.parameters["tol"].default is None
    assert sig.parameters["fit_intercept"].default is True
    assert sig.parameters["precompute"].default == "auto"
    # Live sklearn oracle exposes the same four keyword-only params.
    sk_sig = inspect.signature(SkOrthogonalMatchingPursuit.__init__)
    sk_params = [p for p in sk_sig.parameters if p != "self"]
    assert sk_params == params


def test_omp_get_params_clone_roundtrip():
    """REQ-OMP (R-DEV-2): `get_params`/`clone` round-trip the four params, matching
    the live sklearn oracle's `get_params` key set."""
    from sklearn.base import clone

    est = fl.OrthogonalMatchingPursuit(
        n_nonzero_coefs=4, tol=None, fit_intercept=False, precompute=False)
    params = est.get_params()
    assert params == {
        "n_nonzero_coefs": 4, "tol": None,
        "fit_intercept": False, "precompute": False,
    }
    c = clone(est)
    assert c.get_params() == params
    # Same param key set as the sklearn oracle.
    sk_keys = set(SkOrthogonalMatchingPursuit().get_params().keys())
    assert set(params.keys()) == sk_keys


def test_omp_score_and_n_features_in_match_sklearn():
    """REQ-OMP: inherits `RegressorMixin.score` (R^2) and sets `n_features_in_`;
    the score equals the live sklearn oracle's score on the same data."""
    X, y = _omp_dataset()
    sk = SkOrthogonalMatchingPursuit(n_nonzero_coefs=5).fit(X, y)
    fr = fl.OrthogonalMatchingPursuit(n_nonzero_coefs=5).fit(X, y)
    assert fr.n_features_in_ == X.shape[1] == sk.n_features_in_
    assert abs(fr.score(X, y) - sk.score(X, y)) < 1e-9


def test_omp_pickle_roundtrip_preserves_predictions():
    """REQ-OMP: pickle round-trips (Rust fitted object not picklable ->
    refit-on-unpickle via `_RegressorPickleMixin`); predictions are preserved."""
    import pickle

    X, y = _omp_dataset()
    fr = fl.OrthogonalMatchingPursuit(n_nonzero_coefs=4).fit(X, y)
    pred_before = fr.predict(X)
    fr2 = pickle.loads(pickle.dumps(fr))
    np.testing.assert_allclose(fr2.predict(X), pred_before, rtol=0, atol=1e-12)


# ===========================================================================
# Lars / LassoLars (#2174): the binding-surface unit. Oracle = live sklearn
# 1.5.2 `sklearn.linear_model.Lars`/`LassoLars` (imported here), compared
# head-to-head. coef_/intercept_/predict per several `n_nonzero_coefs` and the
# default (n_nonzero_coefs=500 -> capped at n_features), fit_intercept True/False,
# `LassoLars(alpha=a)` per several alpha, the ctor ABI / get_params / clone, and
# the validated/NotImplemented param surface (jitter / positive / bad precompute /
# n_nonzero_coefs<1 / negative alpha). All expected values are computed by the
# live sklearn oracle in the SAME test (R-CHAR-3) — never copied from ferrolearn.
# ===========================================================================

from sklearn.linear_model import Lars as SkLars
from sklearn.linear_model import LassoLars as SkLassoLars


def _lars_dataset(seed=0, n=60, p=8, informative=5, noise=4.0):
    from sklearn.datasets import make_regression

    X, y = make_regression(
        n_samples=n, n_features=p, n_informative=informative, noise=noise,
        random_state=seed,
    )
    return X, y


def test_lars_coef_intercept_predict_match_sklearn_per_n_nonzero():
    """REQ-LARS: for several `n_nonzero_coefs`, `coef_`/`intercept_`/`predict`
    reproduce the live sklearn oracle to ~1e-6 (lars.rs REQ-1/3/4 SHIPPED)."""
    X, y = _lars_dataset()
    for k in (1, 2, 3, 5):
        sk = SkLars(n_nonzero_coefs=k).fit(X, y)
        fr = fl.Lars(n_nonzero_coefs=k).fit(X, y)
        np.testing.assert_allclose(
            np.asarray(fr.coef_), sk.coef_, rtol=0, atol=1e-6,
            err_msg=f"coef_ diverges at n_nonzero_coefs={k}",
        )
        assert abs(fr.intercept_ - sk.intercept_) < 1e-6, (
            f"intercept_ diverges at n_nonzero_coefs={k}"
        )
        np.testing.assert_allclose(
            fr.predict(X), sk.predict(X), rtol=0, atol=1e-6,
            err_msg=f"predict diverges at n_nonzero_coefs={k}",
        )
        # k features in the active set -> exactly k non-zero coefficients.
        assert int(np.count_nonzero(np.abs(np.asarray(fr.coef_)) > 1e-9)) == k


def test_lars_default_n_nonzero_capped_at_n_features_matches_sklearn():
    """REQ-LARS: sklearn's default `n_nonzero_coefs=500` is an UPPER bound on the
    active set, capped at `n_features` (`_least_angle.py` lars_path
    `max_features`). With p=8 features the cap is 8, so the default uses all
    features; ferrolearn reproduces the SAME `coef_`/`intercept_`/`predict`."""
    X, y = _lars_dataset(p=8)
    sk = SkLars().fit(X, y)
    fr = fl.Lars().fit(X, y)
    # The wrapper exposes sklearn's default 500 (capped internally to n_features).
    assert fr.n_nonzero_coefs == 500
    np.testing.assert_allclose(np.asarray(fr.coef_), sk.coef_, rtol=0, atol=1e-6)
    assert abs(fr.intercept_ - sk.intercept_) < 1e-6
    np.testing.assert_allclose(fr.predict(X), sk.predict(X), rtol=0, atol=1e-6)


def test_lars_fit_intercept_true_false_match_sklearn():
    """REQ-LARS: both `fit_intercept=True`/`False` reproduce the sklearn oracle;
    with `fit_intercept=False` the intercept is exactly 0.0."""
    X, y = _lars_dataset()
    for fit_intercept in (True, False):
        sk = SkLars(n_nonzero_coefs=4, fit_intercept=fit_intercept).fit(X, y)
        fr = fl.Lars(n_nonzero_coefs=4, fit_intercept=fit_intercept).fit(X, y)
        np.testing.assert_allclose(
            np.asarray(fr.coef_), sk.coef_, rtol=0, atol=1e-6,
            err_msg=f"coef_ diverges at fit_intercept={fit_intercept}",
        )
        assert abs(fr.intercept_ - sk.intercept_) < 1e-6
        if not fit_intercept:
            assert fr.intercept_ == 0.0


def test_lars_precompute_accepted_no_result_change():
    """REQ-LARS: `precompute='auto'`/`True`/`False`/`None` (`_least_angle.py:1035`)
    is a Gram-matrix speed knob that does NOT change the LARS solution. ferrolearn
    accepts every value, yielding the IDENTICAL fit as the `'auto'` default (and
    matching the precompute-invariant sklearn oracle)."""
    X, y = _lars_dataset()
    sk = SkLars(n_nonzero_coefs=4).fit(X, y)
    base = fl.Lars(n_nonzero_coefs=4, precompute='auto').fit(X, y)
    for pc in ('auto', False, True, None):
        fr = fl.Lars(n_nonzero_coefs=4, precompute=pc).fit(X, y)
        np.testing.assert_array_equal(
            np.asarray(fr.coef_), np.asarray(base.coef_),
        )
        np.testing.assert_allclose(
            np.asarray(fr.coef_), sk.coef_, rtol=0, atol=1e-6,
            err_msg=f"coef_ diverges at precompute={pc!r}",
        )


def test_lars_ctor_abi_keyword_only_like_sklearn():
    """REQ-LARS (R-DEV-2): the constructor exposes exactly sklearn's nine
    keyword-only params with sklearn's defaults/order
    (`_least_angle.py:1047-1058`)."""
    sig = inspect.signature(fl.Lars.__init__)
    params = [p for p in sig.parameters if p != "self"]
    sk_sig = inspect.signature(SkLars.__init__)
    sk_params = [p for p in sk_sig.parameters if p != "self"]
    assert params == sk_params
    for name in params:
        assert sig.parameters[name].kind == inspect.Parameter.KEYWORD_ONLY, (
            f"{name} must be keyword-only (sklearn leading `*`)"
        )
        assert (
            sig.parameters[name].kind == sk_sig.parameters[name].kind
        ), f"{name} kind diverges from sklearn"
    assert sig.parameters["n_nonzero_coefs"].default == 500
    assert sig.parameters["fit_intercept"].default is True
    assert sig.parameters["precompute"].default == "auto"
    assert sig.parameters["jitter"].default is None


def test_lars_get_params_clone_roundtrip_matches_sklearn_keys():
    """REQ-LARS (R-DEV-2): `get_params`/`clone` round-trip the params, matching
    the live sklearn oracle's `get_params` key set."""
    from sklearn.base import clone

    est = fl.Lars(n_nonzero_coefs=3, fit_intercept=False, precompute=False)
    params = est.get_params()
    assert params["n_nonzero_coefs"] == 3
    assert params["fit_intercept"] is False
    assert params["precompute"] is False
    c = clone(est)
    assert c.get_params() == params
    sk_keys = set(SkLars().get_params().keys())
    assert set(params.keys()) == sk_keys


def test_lars_invalid_n_nonzero_coefs_rejected_like_sklearn():
    """REQ-LARS (R-DEV-2): `n_nonzero_coefs < 1` is an invalid parameter; sklearn
    raises a ValueError-subclass (`InvalidParameterError`) at fit, ferrolearn
    raises `ValueError`."""
    X, y = _lars_dataset()
    # sklearn oracle: fitting with n_nonzero_coefs=0 raises a ValueError subclass.
    with pytest.raises(ValueError):
        SkLars(n_nonzero_coefs=0).fit(X, y)
    with pytest.raises(ValueError):
        fl.Lars(n_nonzero_coefs=0).fit(X, y)
    with pytest.raises(ValueError):
        fl.Lars(n_nonzero_coefs=-1).fit(X, y)


def test_lars_bad_precompute_rejected_like_sklearn():
    """REQ-LARS (R-DEV-2): an invalid `precompute` string is an invalid parameter
    (ValueError-subclass in sklearn; ValueError in ferrolearn)."""
    X, y = _lars_dataset()
    with pytest.raises(ValueError):
        SkLars(precompute="nonsense").fit(X, y)
    with pytest.raises(ValueError):
        fl.Lars(precompute="nonsense").fit(X, y)


def test_lars_jitter_non_none_raises_notimplemented():
    """REQ-LARS: a non-None `jitter` adds SEEDED gaussian noise to y
    (`_least_angle.py:1170-1175`) which changes coef_; the Rust core cannot
    reproduce numpy's RNG stream (R-SUBSTRATE-5), so ferrolearn raises an honest
    `NotImplementedError` while sklearn fits successfully (#2174)."""
    X, y = _lars_dataset()
    # sklearn oracle: jitter fits fine (no exception).
    SkLars(n_nonzero_coefs=3, jitter=0.1, random_state=0).fit(X, y)
    with pytest.raises(NotImplementedError):
        fl.Lars(n_nonzero_coefs=3, jitter=0.1, random_state=0).fit(X, y)


def test_lasso_lars_coef_intercept_predict_match_sklearn_per_alpha():
    """REQ-LASSOLARS: for several `alpha`, `coef_`/`intercept_`/`predict`
    reproduce the live sklearn oracle to ~1e-6 (lars.rs REQ-2/3/4 SHIPPED)."""
    X, y = _lars_dataset()
    for a in (0.1, 0.5, 1.0, 2.0):
        sk = SkLassoLars(alpha=a).fit(X, y)
        fr = fl.LassoLars(alpha=a).fit(X, y)
        np.testing.assert_allclose(
            np.asarray(fr.coef_), sk.coef_, rtol=0, atol=1e-6,
            err_msg=f"coef_ diverges at alpha={a}",
        )
        assert abs(fr.intercept_ - sk.intercept_) < 1e-6, (
            f"intercept_ diverges at alpha={a}"
        )
        np.testing.assert_allclose(
            fr.predict(X), sk.predict(X), rtol=0, atol=1e-6,
            err_msg=f"predict diverges at alpha={a}",
        )


def test_lasso_lars_alpha_positional_first_like_sklearn():
    """REQ-LASSOLARS (R-DEV-2): `alpha` is positional-or-keyword FIRST, the rest
    keyword-only (`_least_angle.py:1363-1376`) — matching the live oracle."""
    sig = inspect.signature(fl.LassoLars.__init__)
    params = [p for p in sig.parameters if p != "self"]
    sk_sig = inspect.signature(SkLassoLars.__init__)
    sk_params = [p for p in sk_sig.parameters if p != "self"]
    assert params == sk_params
    for name in params:
        assert (
            sig.parameters[name].kind == sk_sig.parameters[name].kind
        ), f"{name} kind diverges from sklearn"
    assert (
        sig.parameters["alpha"].kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
    )
    assert sig.parameters["alpha"].default == 1.0
    # alpha accepted positionally, like sklearn.
    assert fl.LassoLars(0.3).alpha == 0.3
    assert SkLassoLars(0.3).alpha == 0.3


def test_lasso_lars_fit_intercept_true_false_match_sklearn():
    """REQ-LASSOLARS: both `fit_intercept=True`/`False` reproduce the oracle."""
    X, y = _lars_dataset()
    for fit_intercept in (True, False):
        sk = SkLassoLars(alpha=0.5, fit_intercept=fit_intercept).fit(X, y)
        fr = fl.LassoLars(alpha=0.5, fit_intercept=fit_intercept).fit(X, y)
        np.testing.assert_allclose(
            np.asarray(fr.coef_), sk.coef_, rtol=0, atol=1e-6,
            err_msg=f"coef_ diverges at fit_intercept={fit_intercept}",
        )
        assert abs(fr.intercept_ - sk.intercept_) < 1e-6
        if not fit_intercept:
            assert fr.intercept_ == 0.0


def test_lasso_lars_get_params_clone_roundtrip_matches_sklearn_keys():
    """REQ-LASSOLARS (R-DEV-2): `get_params`/`clone` round-trip, matching the
    sklearn oracle's `get_params` key set."""
    from sklearn.base import clone

    est = fl.LassoLars(alpha=0.7, fit_intercept=False, max_iter=200)
    params = est.get_params()
    assert params["alpha"] == 0.7
    assert params["fit_intercept"] is False
    assert params["max_iter"] == 200
    c = clone(est)
    assert c.get_params() == params
    sk_keys = set(SkLassoLars().get_params().keys())
    assert set(params.keys()) == sk_keys


def test_lasso_lars_negative_alpha_rejected_like_sklearn():
    """REQ-LASSOLARS (R-DEV-2): negative `alpha` is an invalid parameter
    (ValueError-subclass in sklearn; ValueError in ferrolearn)."""
    X, y = _lars_dataset()
    with pytest.raises(ValueError):
        SkLassoLars(alpha=-1.0).fit(X, y)
    with pytest.raises(ValueError):
        fl.LassoLars(alpha=-1.0).fit(X, y)


def test_lasso_lars_positive_true_raises_notimplemented():
    """REQ-LASSOLARS: `positive=True` (`_least_angle.py:1357`) constrains
    coefficients to be non-negative — a different optimization the Rust core does
    NOT implement, so ferrolearn raises an honest `NotImplementedError` while
    sklearn fits successfully (NOT-STARTED #2174)."""
    X, y = _lars_dataset()
    # sklearn oracle: positive=True fits fine.
    SkLassoLars(alpha=0.5, positive=True).fit(X, y)
    with pytest.raises(NotImplementedError):
        fl.LassoLars(alpha=0.5, positive=True).fit(X, y)


def test_lasso_lars_jitter_non_none_raises_notimplemented():
    """REQ-LASSOLARS: a non-None `jitter` raises `NotImplementedError`
    (seeded gaussian noise, R-SUBSTRATE-5; #2174) while sklearn fits."""
    X, y = _lars_dataset()
    SkLassoLars(alpha=0.5, jitter=0.1, random_state=0).fit(X, y)
    with pytest.raises(NotImplementedError):
        fl.LassoLars(alpha=0.5, jitter=0.1, random_state=0).fit(X, y)


def test_lars_score_and_n_features_in_match_sklearn():
    """REQ-LARS/LASSOLARS: inherit `RegressorMixin.score` (R^2) and set
    `n_features_in_`; the score equals the live sklearn oracle's score."""
    X, y = _lars_dataset()
    sk = SkLars(n_nonzero_coefs=5).fit(X, y)
    fr = fl.Lars(n_nonzero_coefs=5).fit(X, y)
    assert fr.n_features_in_ == X.shape[1] == sk.n_features_in_
    assert abs(fr.score(X, y) - sk.score(X, y)) < 1e-6

    skl = SkLassoLars(alpha=0.3).fit(X, y)
    frl = fl.LassoLars(alpha=0.3).fit(X, y)
    assert frl.n_features_in_ == X.shape[1] == skl.n_features_in_
    assert abs(frl.score(X, y) - skl.score(X, y)) < 1e-6


def test_lars_pickle_roundtrip_preserves_predictions():
    """REQ-LARS/LASSOLARS: pickle round-trips (Rust fitted object not picklable ->
    refit-on-unpickle via `_RegressorPickleMixin`); predictions are preserved."""
    import pickle

    X, y = _lars_dataset()
    for fr in (fl.Lars(n_nonzero_coefs=4).fit(X, y),
               fl.LassoLars(alpha=0.5).fit(X, y)):
        pred_before = fr.predict(X)
        fr2 = pickle.loads(pickle.dumps(fr))
        np.testing.assert_allclose(
            fr2.predict(X), pred_before, rtol=0, atol=1e-12)


# ===========================================================================
# RANSACRegressor (#2178) — the binding over the default LinearRegression base.
# Every expected value is computed by the LIVE sklearn 1.5.2 oracle in the SAME
# test (R-CHAR-3) — never copied from ferrolearn. Well-separated outlier data is
# used so the best inlier set is UNIQUE -> the refit coef_/intercept_/predict
# converge to sklearn's REGARDLESS of the RNG substrate (#2118).
# ===========================================================================

from sklearn.linear_model import LinearRegression as SkLinearRegression
from sklearn.linear_model import RANSACRegressor as SkRANSACRegressor


def _ransac_well_separated(seed=0, n_inliers=80, n_outliers=12):
    """1-feature data y = 3x + 2 with a handful of FAR outliers. The inlier set
    is cleanly separable, so RANSAC's best consensus set is unique."""
    rng = np.random.RandomState(seed)
    x = np.linspace(-5, 5, n_inliers)
    y = 3.0 * x + 2.0 + rng.normal(scale=0.05, size=n_inliers)
    # Outliers: far off the line.
    xo = rng.uniform(-5, 5, n_outliers)
    yo = 3.0 * xo + 2.0 + rng.choice([-1, 1], n_outliers) * rng.uniform(40, 60, n_outliers)
    X = np.concatenate([x, xo]).reshape(-1, 1)
    Y = np.concatenate([y, yo])
    return X, Y


def test_ransac_default_coef_intercept_predict_match_sklearn():
    """REQ-RANSAC: default (min_samples=None -> n_features+1, residual_threshold
    None -> MAD) on well-separated data reproduces sklearn coef_/intercept_/predict
    to a reasonable tolerance (the unique best inlier set's refit is deterministic).
    """
    X, y = _ransac_well_separated()
    sk = SkRANSACRegressor(random_state=0, max_trials=1000).fit(X, y)
    fr = fl.RANSACRegressor(random_state=0, max_trials=1000).fit(X, y)
    np.testing.assert_allclose(np.asarray(fr.coef_), sk.estimator_.coef_,
                               rtol=0, atol=1e-2)
    assert abs(fr.intercept_ - sk.estimator_.intercept_) < 1e-2
    Xt = np.linspace(-5, 5, 11).reshape(-1, 1)
    np.testing.assert_allclose(fr.predict(Xt), sk.predict(Xt), rtol=0, atol=5e-2)


def test_ransac_inlier_mask_matches_sklearn():
    """REQ-RANSAC: the inlier_mask_ excludes the (well-separated) outliers exactly
    as sklearn does — the last 12 points are outliers in both."""
    X, y = _ransac_well_separated()
    sk = SkRANSACRegressor(random_state=0, max_trials=1000).fit(X, y)
    fr = fl.RANSACRegressor(random_state=0, max_trials=1000).fit(X, y)
    assert fr.inlier_mask_.dtype == bool
    # The 12 trailing outliers are NOT inliers (structural contract).
    assert not fr.inlier_mask_[-12:].any()
    # Same set of excluded outliers as sklearn.
    np.testing.assert_array_equal(fr.inlier_mask_, sk.inlier_mask_)


def test_ransac_explicit_int_min_samples_matches_sklearn():
    """REQ-RANSAC: an explicit int min_samples threads through; coef_ matches."""
    X, y = _ransac_well_separated()
    sk = SkRANSACRegressor(min_samples=10, random_state=0, max_trials=1000).fit(X, y)
    fr = fl.RANSACRegressor(min_samples=10, random_state=0, max_trials=1000).fit(X, y)
    np.testing.assert_allclose(np.asarray(fr.coef_), sk.estimator_.coef_,
                               rtol=0, atol=1e-2)
    np.testing.assert_array_equal(fr.inlier_mask_, sk.inlier_mask_)


def test_ransac_float_min_samples_resolves_to_ceil_like_sklearn():
    """REQ-RANSAC: a float min_samples in [0,1] resolves to
    ceil(min_samples * n_samples), matching sklearn (`_ransac.py:389-392`)."""
    X, y = _ransac_well_separated()
    sk = SkRANSACRegressor(min_samples=0.2, random_state=0, max_trials=1000).fit(X, y)
    fr = fl.RANSACRegressor(min_samples=0.2, random_state=0, max_trials=1000).fit(X, y)
    np.testing.assert_allclose(np.asarray(fr.coef_), sk.estimator_.coef_,
                               rtol=0, atol=1e-2)
    np.testing.assert_array_equal(fr.inlier_mask_, sk.inlier_mask_)


def test_ransac_residual_threshold_set_vs_none_match_sklearn():
    """REQ-RANSAC: an explicit residual_threshold and the None(MAD) default both
    reproduce sklearn's coef_ on well-separated data."""
    X, y = _ransac_well_separated()
    for rt in (None, 5.0):
        sk = SkRANSACRegressor(residual_threshold=rt, random_state=0,
                               max_trials=1000).fit(X, y)
        fr = fl.RANSACRegressor(residual_threshold=rt, random_state=0,
                                max_trials=1000).fit(X, y)
        np.testing.assert_allclose(np.asarray(fr.coef_), sk.estimator_.coef_,
                                   rtol=0, atol=1e-2,
                                   err_msg=f"coef_ diverges at residual_threshold={rt}")
        np.testing.assert_array_equal(fr.inlier_mask_, sk.inlier_mask_)


def test_ransac_get_params_clone_exposes_sklearn_key_set():
    """REQ-RANSAC: get_params/clone expose sklearn's full key set (so RANSAC plugs
    into Pipeline/GridSearch). The keys are exactly sklearn's."""
    from sklearn.base import clone
    sk_keys = set(SkRANSACRegressor().get_params(deep=False).keys())
    fr = fl.RANSACRegressor()
    assert set(fr.get_params(deep=False).keys()) == sk_keys
    fr2 = clone(fr)
    assert set(fr2.get_params(deep=False).keys()) == sk_keys


def test_ransac_custom_estimator_raises_notimplemented():
    """REQ-RANSAC: a non-default base estimator has no Rust binding ->
    NotImplementedError (only the default LinearRegression base is supported)."""
    X, y = _ransac_well_separated()
    from sklearn.tree import DecisionTreeRegressor as SkDTR
    with pytest.raises(NotImplementedError):
        fl.RANSACRegressor(estimator=SkDTR()).fit(X, y)
    # A LinearRegression with NON-default params is also rejected.
    with pytest.raises(NotImplementedError):
        fl.RANSACRegressor(estimator=SkLinearRegression(fit_intercept=False)).fit(X, y)


def test_ransac_default_estimator_none_accepted():
    """REQ-RANSAC: estimator=None (default) and an explicit DEFAULT-param
    LinearRegression instance are both accepted (the default base)."""
    X, y = _ransac_well_separated()
    fl.RANSACRegressor(estimator=None, random_state=0).fit(X, y)
    fl.RANSACRegressor(estimator=SkLinearRegression(), random_state=0).fit(X, y)


def test_ransac_squared_error_loss_raises_notimplemented():
    """REQ-RANSAC: loss='squared_error' selects a different inlier set; the Rust
    core hardcodes absolute-error residuals -> NotImplementedError (#516)."""
    X, y = _ransac_well_separated()
    with pytest.raises(NotImplementedError):
        fl.RANSACRegressor(loss='squared_error').fit(X, y)
    with pytest.raises(NotImplementedError):
        fl.RANSACRegressor(loss=lambda a, b: np.abs(a - b)).fit(X, y)


def test_ransac_nondefault_stop_params_raise_notimplemented():
    """REQ-RANSAC: non-default max_skips/stop_*/inf max_trials change the trial
    count / early-stop; the Rust core has no such tracking -> NotImplementedError
    (#515)."""
    X, y = _ransac_well_separated()
    for kw in (dict(max_skips=5), dict(stop_n_inliers=50),
               dict(stop_score=0.9), dict(stop_probability=0.5),
               dict(max_trials=np.inf)):
        with pytest.raises(NotImplementedError):
            fl.RANSACRegressor(**kw).fit(X, y)


def test_ransac_invalid_params_raise_valueerror():
    """REQ-RANSAC: invalid min_samples / max_trials / residual_threshold mirror
    sklearn's InvalidParameterError (a ValueError subclass)."""
    X, y = _ransac_well_separated()
    # min_samples out of [0,1] float range and < 1 int.
    with pytest.raises(ValueError):
        fl.RANSACRegressor(min_samples=1.5).fit(X, y)
    with pytest.raises(ValueError):
        fl.RANSACRegressor(min_samples=0).fit(X, y)
    # min_samples > n_samples.
    with pytest.raises(ValueError):
        fl.RANSACRegressor(min_samples=10_000).fit(X, y)
    # negative max_trials / residual_threshold.
    with pytest.raises(ValueError):
        fl.RANSACRegressor(max_trials=-1).fit(X, y)
    with pytest.raises(ValueError):
        fl.RANSACRegressor(residual_threshold=-1.0).fit(X, y)


def test_ransac_not_fitted_predict_raises():
    """REQ-RANSAC: predict before fit raises (NotFittedError via check_is_fitted)."""
    from sklearn.exceptions import NotFittedError
    with pytest.raises(NotFittedError):
        fl.RANSACRegressor().predict(np.zeros((3, 1)))


def test_ransac_score_and_n_features_in_match_sklearn():
    """REQ-RANSAC: inherit RegressorMixin.score (R^2) + set n_features_in_; the
    score equals the live sklearn oracle's on well-separated data."""
    X, y = _ransac_well_separated()
    sk = SkRANSACRegressor(random_state=0, max_trials=1000).fit(X, y)
    fr = fl.RANSACRegressor(random_state=0, max_trials=1000).fit(X, y)
    assert fr.n_features_in_ == X.shape[1] == sk.n_features_in_
    assert abs(fr.score(X, y) - sk.score(X, y)) < 1e-2


def test_ransac_pickle_roundtrip_preserves_predictions():
    """REQ-RANSAC: pickle round-trips (Rust fitted object not picklable ->
    refit-on-unpickle via `_RegressorPickleMixin`); predictions are preserved."""
    import pickle

    X, y = _ransac_well_separated()
    fr = fl.RANSACRegressor(random_state=0, max_trials=1000).fit(X, y)
    pred_before = fr.predict(X)
    fr2 = pickle.loads(pickle.dumps(fr))
    np.testing.assert_allclose(fr2.predict(X), pred_before, rtol=0, atol=1e-9)
