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
