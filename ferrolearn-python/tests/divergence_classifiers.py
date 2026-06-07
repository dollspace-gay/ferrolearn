"""Divergence guards for ferrolearn-python classifier bindings (unit #2044).

Targets: `ferrolearn-python/src/classifiers.rs`
(RsLogisticRegression / RsDecisionTreeClassifier / RsRandomForestClassifier /
RsKNeighborsClassifier / RsGaussianNB) + the Python wrappers
`ferrolearn-python/python/ferrolearn/_classifiers.py`
(class LogisticRegression / DecisionTreeClassifier / RandomForestClassifier /
KNeighborsClassifier / GaussianNB), mirroring
`sklearn.linear_model.LogisticRegression` (`sklearn/linear_model/_logistic.py:1129`),
`sklearn.tree.DecisionTreeClassifier` (`sklearn/tree/_classes.py:945`),
`sklearn.ensemble.RandomForestClassifier` (`sklearn/ensemble/_forest.py:1494`),
`sklearn.neighbors.KNeighborsClassifier`
(`sklearn/neighbors/_classification.py:193`), and
`sklearn.naive_bayes.GaussianNB` (`sklearn/naive_bayes.py:234`).

Verification model B (goal.md): every expected value is computed by the LIVE
sklearn 1.5.2 oracle in the same test and compared against `import ferrolearn`.
No expected value is literal-copied from the ferrolearn side (R-CHAR-3).

Contents:
  - FIVE FAILING single-wrapper-fixable pins (R-DEFER-3 — each closable THIS
    iteration in `_classifiers.py` alone):
      * RandomForestClassifier(10).n_estimators -> sklearn 10; ferrolearn raises
        TypeError (n_estimators keyword-only). Fix: move before the `*`.
      * KNeighborsClassifier(3).n_neighbors     -> sklearn 3; ferrolearn raises
        TypeError (n_neighbors keyword-only). Fix: move before the `*`.
      * DecisionTreeClassifier().fit(X,y).feature_importances_ -> sklearn sums to
        1.0; ferrolearn raises AttributeError (wrapper never reads the existing
        RsDecisionTreeClassifier.feature_importances_ getter).
      * RandomForestClassifier(...).fit(X,y).feature_importances_ -> sklearn sums
        to 1.0; ferrolearn raises AttributeError (wrapper never reads the
        existing RsRandomForestClassifier.feature_importances_ getter).
      * LogisticRegression().max_iter -> sklearn 100; ferrolearn default 1000.
        EMPIRICALLY CHECKED (this iteration): ferrolearn's LBFGS at max_iter=100
        still fits the separable dataset to accuracy 1.0, so the ABI default fix
        is single-wrapper-fixable -> pinned RED (PATH A).
  - PASSING API-conformance green guards for ALL FIVE estimators (method/
    attribute surface + value parity against the live oracle on a deterministic
    separable dataset). RF is RNG-dependent -> structure + accuracy only, NOT
    exact trees. KNN asserts valid-label structure + correctness on a cleanly
    separated dataset (per-row tie/boundary parity owned downstream #876).

The structural NOT-STARTED gaps (the 5 single-wrapper pins; RF/KNN missing
predict_proba; LogReg faked n_iter_; KNN tie/value divergence; missing ctor
params) are filed as -l blocker crosslink issues (referencing, not duplicating,
downstream library issues). Only the single-wrapper-fixable pins are RED here.
"""

import inspect

import numpy as np
import pytest

import ferrolearn

# ---------------------------------------------------------------------------
# Deterministic, cleanly separable two-cluster dataset.
# Cluster 0 near the origin, cluster 1 near (5, 5). Linearly separable, so
# LogReg / DT / RF / KNN / GNB all achieve accuracy 1.0 on it.
# ---------------------------------------------------------------------------
X = np.array(
    [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [5.0, 5.0], [6.0, 5.0], [5.0, 6.0]]
)
y = np.array([0, 0, 0, 1, 1, 1])


# ===========================================================================
# (1) FAILING single-wrapper-fixable pins (R-DEFER-3)
# ===========================================================================


def test_red_rf_n_estimators_positional():
    """RED: RandomForestClassifier(10) accepts n_estimators positionally.

    sklearn `_forest.py:1494`: `__init__(self, n_estimators=100, *, ...)` —
    n_estimators PRECEDES the `*`, so it is positional-or-keyword.
    ferrolearn makes it keyword-only -> `RandomForestClassifier(10)` raises
    TypeError. Fix: move `n_estimators` before the `*` in `_classifiers.py`.
    """
    from sklearn.ensemble import RandomForestClassifier as SkRF

    expected = SkRF(10).n_estimators  # live oracle -> 10
    assert ferrolearn.RandomForestClassifier(10).n_estimators == expected


def test_red_knn_n_neighbors_positional():
    """RED: KNeighborsClassifier(3) accepts n_neighbors positionally.

    sklearn `_classification.py:193`: `__init__(self, n_neighbors=5, *, ...)` —
    n_neighbors PRECEDES the `*`. ferrolearn makes it keyword-only ->
    `KNeighborsClassifier(3)` raises TypeError. Fix: move `n_neighbors` before
    the `*` in `_classifiers.py`.
    """
    from sklearn.neighbors import KNeighborsClassifier as SkKNN

    expected = SkKNN(3).n_neighbors  # live oracle -> 3
    assert ferrolearn.KNeighborsClassifier(3).n_neighbors == expected


def test_red_dt_feature_importances_exposed():
    """RED: DecisionTreeClassifier exposes feature_importances_ (sums to 1.0).

    sklearn exposes `feature_importances_` `(n_features,)` summing to 1.0
    (`sklearn/tree/_classes.py`). The Rust `RsDecisionTreeClassifier` ALREADY
    has a `feature_importances_` getter, but `_classifiers.py::
    DecisionTreeClassifier.fit` never reads it -> AttributeError. Fix: the
    wrapper `fit` should set `self.feature_importances_ = np.array(
    self._rs.feature_importances_)`.
    """
    from sklearn.tree import DecisionTreeClassifier as SkDT

    sk_fi = SkDT(random_state=0).fit(X, y).feature_importances_
    fl_fi = ferrolearn.DecisionTreeClassifier().fit(X, y).feature_importances_
    fl_fi = np.asarray(fl_fi)
    assert fl_fi.shape == (X.shape[1],)
    # sklearn's feature_importances_ sum to 1.0 by construction.
    assert sk_fi.sum() == pytest.approx(1.0)
    assert fl_fi.sum() == pytest.approx(1.0)


def test_red_rf_feature_importances_exposed():
    """RED: RandomForestClassifier exposes feature_importances_ (sums to 1.0).

    sklearn exposes `feature_importances_` summing to 1.0
    (`sklearn/ensemble/_forest.py`). The Rust `RsRandomForestClassifier`
    ALREADY has the getter; `_classifiers.py::RandomForestClassifier.fit` never
    reads it -> AttributeError. Fix: the wrapper reads the getter.
    """
    from sklearn.ensemble import RandomForestClassifier as SkRF

    sk_fi = SkRF(n_estimators=10, random_state=0).fit(X, y).feature_importances_
    fitted = ferrolearn.RandomForestClassifier(
        n_estimators=10, random_state=0
    ).fit(X, y)
    fl_fi = np.asarray(fitted.feature_importances_)
    assert fl_fi.shape == (X.shape[1],)
    assert sk_fi.sum() == pytest.approx(1.0)
    assert fl_fi.sum() == pytest.approx(1.0)


def test_red_rf_predict_proba_surfaced():
    """RED: RandomForestClassifier exposes predict_proba (rows sum to 1.0).

    sklearn `_forest.py:922`: `predict_proba` returns `(n_samples, n_classes)`,
    the mean of per-tree class probabilities, rows summing to 1.0; and
    `_forest.py:883` `predict = classes_[argmax(predict_proba)]`. The Rust
    `FittedRandomForestClassifier::predict_proba` ALREADY exists and is
    value-correct, but neither `RsRandomForestClassifier` nor the
    `_classifiers.py::RandomForestClassifier` wrapper surfaced it -> the wrapper
    raised AttributeError. Fix: bind + surface predict_proba.

    Exact sklearn value parity is the documented numpy-MT bootstrap RNG boundary
    (#673), so we pin the STRUCTURAL contract, not exact values vs sklearn.
    """
    from sklearn.ensemble import RandomForestClassifier as SkRF

    # sklearn's RandomForestClassifier exposes predict_proba (oracle invariant).
    assert hasattr(SkRF, "predict_proba")

    X_train = np.array(
        [[0.0, 0.0], [0.1, 0.1], [0.2, 0.0], [5.0, 5.0], [5.1, 4.9], [4.9, 5.1]]
    )
    y_train = np.array([0, 0, 0, 1, 1, 1])
    Xq = np.array([[0.05, 0.05], [5.0, 5.0], [0.2, 0.1], [4.95, 5.05]])

    m = ferrolearn.RandomForestClassifier(n_estimators=10, random_state=0).fit(
        X_train, y_train
    )
    proba = np.asarray(m.predict_proba(Xq))

    # (a) shape == (n_query, n_classes)
    assert proba.shape == (Xq.shape[0], 2)
    # (b) every row sums to 1.0
    assert np.allclose(proba.sum(axis=1), 1.0)
    # (c) predict == classes_[argmax(predict_proba)] (sklearn `_forest.py:883`)
    assert np.array_equal(
        np.asarray(m.predict(Xq)),
        np.asarray(m.classes_)[np.argmax(proba, axis=1)],
    )

    # Well-separated training data -> predict_proba argmax recovers y_train.
    proba_train = np.asarray(m.predict_proba(X_train))
    assert np.array_equal(
        np.asarray(m.classes_)[np.argmax(proba_train, axis=1)], y_train
    )


def test_red_knn_predict_proba_surfaced():
    """RED: KNeighborsClassifier exposes predict_proba (rows sum to 1.0).

    sklearn `_classification.py:307`: `predict_proba` returns
    `(n_queries, n_classes)` per-neighbor weighted class shares, rows summing to
    1.0, with `predict = classes_[argmax(predict_proba)]`. The Rust
    `FittedKNeighborsClassifier::predict_proba` (`knn.rs:487`) ALREADY exists and
    is value-correct, but neither `RsKNeighborsClassifier` nor the
    `_classifiers.py::KNeighborsClassifier` wrapper surfaced it -> the wrapper
    raised AttributeError. Fix: bind + surface predict_proba.

    Per-row tie/boundary value parity vs sklearn is owned downstream (#876/#877),
    so we pin the STRUCTURAL contract, not exact per-row values.
    """
    from sklearn.neighbors import KNeighborsClassifier as SkKNN

    # sklearn's KNeighborsClassifier exposes predict_proba (oracle invariant).
    assert hasattr(SkKNN, "predict_proba")

    X_train = np.array(
        [[0.0, 0.0], [0.1, 0.1], [5.0, 5.0], [5.1, 4.9], [9.0, 9.0], [9.1, 9.1]]
    )
    y_train = np.array([0, 0, 1, 1, 2, 2])
    Xq = np.array([[0.05, 0.05], [5.0, 5.0], [9.05, 9.05], [0.2, 0.1]])

    m = ferrolearn.KNeighborsClassifier(n_neighbors=1).fit(X_train, y_train)
    proba = np.asarray(m.predict_proba(Xq))

    # (a) shape == (n_query, n_classes)
    assert proba.shape == (Xq.shape[0], 3)
    # (b) every row sums to 1.0
    assert np.allclose(proba.sum(axis=1), 1.0)
    # (c) predict == classes_[argmax(predict_proba)] (sklearn `_classification.py:307`)
    assert np.array_equal(
        np.asarray(m.predict(Xq)),
        np.asarray(m.classes_)[np.argmax(proba, axis=1)],
    )


def test_red_logreg_max_iter_default():
    """RED: LogisticRegression default max_iter is 100 (matching sklearn).

    sklearn `_logistic.py:1129`: `__init__(self, ..., max_iter=100, ...)`.
    ferrolearn defaults `max_iter=1000` (R-DEV-2 ABI divergence).

    PATH A (pinned RED): empirically (this iteration) ferrolearn's LBFGS at
    max_iter=100 still fits the separable dataset to accuracy 1.0 — see the
    convergence sanity-check below — so changing the wrapper default to 100 is
    single-wrapper-fixable and does NOT break value parity. Fix: change the
    default in `_classifiers.py::LogisticRegression.__init__` to 100.
    """
    from sklearn.linear_model import LogisticRegression as SkLR

    expected = inspect.signature(SkLR.__init__).parameters["max_iter"].default
    assert expected == 100  # guards the oracle itself

    # Convergence sanity-check that makes PATH A legitimate: a fit at the
    # sklearn default still classifies the separable training set perfectly.
    fitted = ferrolearn.LogisticRegression(max_iter=100).fit(X, y)
    assert (fitted.predict(X) == y).mean() == 1.0

    actual = inspect.signature(
        ferrolearn.LogisticRegression.__init__
    ).parameters["max_iter"].default
    assert actual == expected


# ===========================================================================
# (2) GREEN guards — API conformance (must PASS) for all five estimators
# ===========================================================================


def test_green_logreg_api_conformance():
    """GREEN: LogisticRegression structure + value parity vs sklearn."""
    from sklearn.linear_model import LogisticRegression as SkLR

    sk = SkLR().fit(X, y)
    fl = ferrolearn.LogisticRegression().fit(X, y)

    n_features = X.shape[1]
    assert np.asarray(fl.coef_).shape == (1, n_features)
    assert np.asarray(fl.intercept_).shape == (1,)
    assert np.asarray(fl.classes_).tolist() == sk.classes_.tolist()

    preds = np.asarray(fl.predict(X))
    assert preds.shape == (X.shape[0],)
    assert set(np.unique(preds)).issubset(set(sk.classes_.tolist()))
    # Separable dataset -> sklearn achieves accuracy 1.0; ferrolearn must too.
    assert (sk.predict(X) == y).mean() == 1.0
    assert (preds == y).mean() == 1.0

    proba = np.asarray(fl.predict_proba(X))
    assert proba.shape == (X.shape[0], 2)
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(X.shape[0]), atol=1e-9)


def test_green_logreg_label_encoder_roundtrip():
    """GREEN: LogisticRegression decodes string labels like sklearn."""
    from sklearn.linear_model import LogisticRegression as SkLR

    Xs = np.array([[0.0, 0.0], [1.0, 0.0], [5.0, 5.0], [6.0, 5.0]])
    ys = np.array(["a", "a", "b", "b"])

    sk = SkLR().fit(Xs, ys)
    fl = ferrolearn.LogisticRegression().fit(Xs, ys)

    assert np.asarray(fl.classes_).tolist() == sk.classes_.tolist() == ["a", "b"]
    assert np.asarray(fl.predict(Xs)).tolist() == sk.predict(Xs).tolist()


def test_green_decision_tree_api_conformance():
    """GREEN: DecisionTreeClassifier predict matches sklearn (deterministic)."""
    from sklearn.tree import DecisionTreeClassifier as SkDT

    sk = SkDT(random_state=0).fit(X, y)
    fl = ferrolearn.DecisionTreeClassifier().fit(X, y)

    assert np.asarray(fl.classes_).tolist() == sk.classes_.tolist()
    preds = np.asarray(fl.predict(X))
    assert preds.tolist() == sk.predict(X).tolist()
    assert (preds == y).mean() == 1.0

    proba = np.asarray(fl.predict_proba(X))
    assert proba.shape == (X.shape[0], len(sk.classes_))


def test_green_random_forest_api_conformance():
    """GREEN: RandomForestClassifier structure + accuracy (RNG -> not exact)."""
    from sklearn.ensemble import RandomForestClassifier as SkRF

    sk = SkRF(n_estimators=10, random_state=0).fit(X, y)
    fl = ferrolearn.RandomForestClassifier(n_estimators=10, random_state=0).fit(
        X, y
    )

    assert np.asarray(fl.classes_).tolist() == sk.classes_.tolist()
    preds = np.asarray(fl.predict(X))
    assert preds.shape == (X.shape[0],)
    assert set(np.unique(preds)).issubset(set(sk.classes_.tolist()))
    # Separable -> both achieve accuracy 1.0 (structure, not exact trees).
    assert (sk.predict(X) == y).mean() == 1.0
    assert (preds == y).mean() == 1.0


def test_green_kneighbors_api_conformance():
    """GREEN: KNeighborsClassifier valid-label structure + clean-separation.

    Per-row tie/boundary parity vs sklearn is owned downstream (#876); here we
    assert valid-label structure + correctness on a CLEANLY separated dataset.
    """
    from sklearn.neighbors import KNeighborsClassifier as SkKNN

    sk = SkKNN(n_neighbors=3).fit(X, y)
    fl = ferrolearn.KNeighborsClassifier(n_neighbors=3).fit(X, y)

    assert np.asarray(fl.classes_).tolist() == sk.classes_.tolist()
    preds = np.asarray(fl.predict(X))
    assert preds.shape == (X.shape[0],)
    assert set(np.unique(preds)).issubset(set(sk.classes_.tolist()))
    # Cleanly separated -> sklearn predicts the training labels exactly.
    assert sk.predict(X).tolist() == y.tolist()
    assert preds.tolist() == y.tolist()


def test_green_gaussian_nb_api_conformance():
    """GREEN: GaussianNB predict + predict_proba match sklearn (deterministic)."""
    from sklearn.naive_bayes import GaussianNB as SkGNB

    sk = SkGNB().fit(X, y)
    fl = ferrolearn.GaussianNB().fit(X, y)

    assert np.asarray(fl.classes_).tolist() == sk.classes_.tolist()
    preds = np.asarray(fl.predict(X))
    assert preds.tolist() == sk.predict(X).tolist()
    assert (preds == y).mean() == 1.0

    proba = np.asarray(fl.predict_proba(X))
    assert proba.shape == (X.shape[0], len(sk.classes_))
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(X.shape[0]), atol=1e-9)


def test_classifiers_predict_log_proba():
    """REQ #2093: GaussianNB/LogisticRegression/DecisionTreeClassifier expose
    `predict_log_proba` = log(predict_proba).

    sklearn DecisionTree/LogisticRegression return `np.log(predict_proba)`
    verbatim; GaussianNB computes `jll - logsumexp(jll)` which equals
    log(predict_proba) where finite. For the deterministic classifiers
    (GaussianNB, DecisionTree) the value matches sklearn's `predict_log_proba`
    element-wise; for all three the `predict_log_proba == log(predict_proba)`
    contract holds (LogisticRegression's value vs sklearn differs only by the
    pre-existing solver-precision gap in `predict_proba`).
    """
    import warnings

    from sklearn.naive_bayes import GaussianNB as SkGNB
    from sklearn.tree import DecisionTreeClassifier as SkDTC

    Xl = np.array(
        [[1.0, 2.0], [1.2, 1.9], [2.0, 2.5], [5.0, 5.0], [5.1, 4.8], [4.8, 5.2]]
    )
    yl = np.array([0, 0, 0, 1, 1, 1])
    q = np.array([[1.5, 2.0], [5.0, 5.0]])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # log(0) -> -inf, sklearn does the same
        # Contract: predict_log_proba == log(predict_proba) for all three.
        for name in ("GaussianNB", "LogisticRegression", "DecisionTreeClassifier"):
            m = getattr(ferrolearn, name)().fit(Xl, yl)
            assert np.allclose(
                m.predict_log_proba(q), np.log(m.predict_proba(q)), equal_nan=True
            )

        # Value parity vs sklearn for the deterministic classifiers.
        fg = ferrolearn.GaussianNB().fit(Xl, yl)
        sg = SkGNB().fit(Xl, yl)
        assert np.allclose(fg.predict_log_proba(q), sg.predict_log_proba(q), atol=1e-6)

        fd = ferrolearn.DecisionTreeClassifier().fit(Xl, yl)
        sd = SkDTC(random_state=0).fit(Xl, yl)
        assert np.allclose(
            fd.predict_log_proba(q), sd.predict_log_proba(q), atol=1e-6, equal_nan=True
        )


def test_gaussiannb_fitted_attrs_match_sklearn():
    """GREEN (#2102): GaussianNB surfaces theta_/var_/class_prior_/class_count_/
    epsilon_ matching the live sklearn oracle.

    sklearn `GaussianNB` exposes these five fitted attributes
    (`sklearn/naive_bayes.py`: `theta_` per-class feature means, `var_`
    epsilon-smoothed per-class variances, `class_prior_`, `class_count_`,
    `epsilon_` = `var_smoothing * max(var(X))`). The ferrolearn wrapper
    `_classifiers.py::GaussianNB.fit` reads them from the pre-existing
    `FittedGaussianNB` accessors via the new `_RsGaussianNB` getters. Expected
    values come from the live sklearn 1.5.2 oracle fit in this test (R-CHAR-3),
    never copied from ferrolearn.
    """
    from sklearn.naive_bayes import GaussianNB as SkGNB

    X = np.array(
        [[1.0, 2.0], [2.0, 1.0], [3.0, 4.0], [4.0, 3.0], [5.0, 5.0], [0.0, 1.0]]
    )
    y = np.array([0, 0, 1, 1, 1, 0])

    fr = ferrolearn.GaussianNB().fit(X, y)
    sk = SkGNB().fit(X, y)

    for attr in ("theta_", "var_", "class_prior_", "class_count_", "epsilon_"):
        assert hasattr(fr, attr), f"GaussianNB missing fitted attribute {attr!r}"

    np.testing.assert_allclose(fr.theta_, sk.theta_, atol=1e-7)
    np.testing.assert_allclose(fr.var_, sk.var_, atol=1e-7)
    np.testing.assert_allclose(fr.class_prior_, sk.class_prior_, atol=1e-9)
    np.testing.assert_allclose(fr.class_count_, sk.class_count_, atol=1e-9)
    assert abs(fr.epsilon_ - sk.epsilon_) < 1e-15


# ===========================================================================
# (3) KNeighborsClassifier constructor-param surface (#2138)
#
# sklearn signature (`neighbors/_classification.py:193`):
#   KNeighborsClassifier(n_neighbors=5, *, weights='uniform', algorithm='auto',
#       leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None)
# ferrolearn surfaces weights ('uniform'/'distance') + algorithm
# ('auto'/'brute'/'kd_tree'/'ball_tree'); leaf_size/n_jobs are ABI no-ops; the
# Euclidean-only restriction (p=2, metric in {minkowski, euclidean},
# metric_params=None, no callable weights) is enforced (NOT-STARTED #876).
#
# Shared oracle dataset from the dispatch (live sklearn 1.5.2; R-CHAR-3).
# ===========================================================================

_KNN_X = np.array(
    [
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [3.0, 3.0],
        [4.0, 3.0],
        [3.0, 4.0],
        [1.0, 1.0],
        [3.5, 3.5],
    ]
)
_KNN_Y = np.array([0, 0, 0, 1, 1, 1, 0, 1])
_KNN_XQ = np.array([[0.5, 0.5], [3.2, 3.3], [2.0, 2.0]])


def test_knn_weights_distance_predict_proba_matches_sklearn():
    """GREEN (#2138): weights='distance' predict_proba matches the live oracle.

    sklearn `_classification.py:307` + `_get_weights`: 'distance' weights each
    neighbor by 1/distance. On the dispatch dataset (k=3) the oracle gives
    row 2 (query [3.2,3.3]) = [0.375, 0.625], which DIFFERS from the
    'uniform' row 2 = [0.333..., 0.666...]. Expected values come from the live
    sklearn fit in this test (R-CHAR-3), never copied from ferrolearn.
    """
    from sklearn.neighbors import KNeighborsClassifier as SkKNN

    sk = SkKNN(n_neighbors=3, weights="distance").fit(_KNN_X, _KNN_Y)
    fl = ferrolearn.KNeighborsClassifier(n_neighbors=3, weights="distance").fit(
        _KNN_X, _KNN_Y
    )

    sk_proba = sk.predict_proba(_KNN_XQ)
    fl_proba = np.asarray(fl.predict_proba(_KNN_XQ))

    np.testing.assert_allclose(fl_proba, sk_proba, atol=1e-9)
    # Guard the documented oracle value for the third query row [2, 2], where
    # inverse-distance weighting DIFFERS from uniform (live sklearn 1.5.2).
    np.testing.assert_allclose(sk_proba[2], [0.375, 0.625], atol=1e-9)
    np.testing.assert_allclose(fl_proba[2], [0.375, 0.625], atol=1e-9)


def test_knn_weights_uniform_default_matches_sklearn():
    """GREEN (#2138): default weights='uniform' predict_proba matches sklearn.

    The 'uniform' row 2 differs from 'distance' (oracle: [0.333..., 0.666...]),
    confirming the weighting branch actually changes the result.
    """
    from sklearn.neighbors import KNeighborsClassifier as SkKNN

    sk = SkKNN(n_neighbors=3).fit(_KNN_X, _KNN_Y)
    fl = ferrolearn.KNeighborsClassifier(n_neighbors=3).fit(_KNN_X, _KNN_Y)

    sk_proba = sk.predict_proba(_KNN_XQ)
    fl_proba = np.asarray(fl.predict_proba(_KNN_XQ))

    np.testing.assert_allclose(fl_proba, sk_proba, atol=1e-9)
    np.testing.assert_allclose(sk_proba[2], [1 / 3, 2 / 3], atol=1e-9)
    # uniform != distance on the third query row (proves weights is honored).
    assert not np.allclose(sk_proba[2], [0.375, 0.625])


def test_knn_algorithm_gives_same_result_as_sklearn():
    """GREEN (#2138): 'brute'/'kd_tree'/'auto'/'ball_tree' all give the SAME
    predict/predict_proba (algorithm is a search strategy only).

    sklearn returns identical results across algorithms; ferrolearn must match
    the oracle for each, and the four ferrolearn results must agree.
    """
    from sklearn.neighbors import KNeighborsClassifier as SkKNN

    sk = SkKNN(n_neighbors=3).fit(_KNN_X, _KNN_Y)
    sk_pred = sk.predict(_KNN_XQ)
    sk_proba = sk.predict_proba(_KNN_XQ)

    results_pred = []
    for algo in ("auto", "brute", "kd_tree", "ball_tree"):
        fl = ferrolearn.KNeighborsClassifier(
            n_neighbors=3, algorithm=algo
        ).fit(_KNN_X, _KNN_Y)
        fl_pred = np.asarray(fl.predict(_KNN_XQ))
        fl_proba = np.asarray(fl.predict_proba(_KNN_XQ))
        # Each algorithm matches the sklearn oracle.
        assert fl_pred.tolist() == sk_pred.tolist()
        np.testing.assert_allclose(fl_proba, sk_proba, atol=1e-9)
        results_pred.append(fl_pred.tolist())

    # All four ferrolearn algorithms agree (search strategy only).
    assert all(r == results_pred[0] for r in results_pred)


def test_knn_get_params_clone_roundtrip():
    """GREEN (#2138): get_params returns all 9 sklearn params with matching
    defaults, and clone() round-trips (check_estimator prerequisite).
    """
    from sklearn.base import clone
    from sklearn.neighbors import KNeighborsClassifier as SkKNN

    fl = ferrolearn.KNeighborsClassifier()
    sk = SkKNN()

    fl_params = fl.get_params()
    sk_params = sk.get_params()

    # Same 9 param names as sklearn.
    assert set(fl_params) == set(sk_params)
    # Matching defaults for every param.
    for name, value in sk_params.items():
        assert fl_params[name] == value, name

    # Signature: n_neighbors positional, rest keyword-only (sklearn parity).
    fl_sig = inspect.signature(ferrolearn.KNeighborsClassifier.__init__)
    sk_sig = inspect.signature(SkKNN.__init__)
    for name in sk_params:
        assert fl_sig.parameters[name].kind == sk_sig.parameters[name].kind, name

    # clone() round-trips (deep copy of the unfitted estimator).
    fl2 = ferrolearn.KNeighborsClassifier(
        n_neighbors=3, weights="distance", algorithm="kd_tree"
    )
    cloned = clone(fl2)
    assert cloned.get_params() == fl2.get_params()
    # A cloned estimator still fits and predicts.
    cloned.fit(_KNN_X, _KNN_Y)
    assert np.asarray(cloned.predict(_KNN_XQ)).shape == (_KNN_XQ.shape[0],)


def test_knn_unsupported_p_raises():
    """GREEN (#2138): p != 2 raises NotImplementedError (Euclidean-only, #876)."""
    fl = ferrolearn.KNeighborsClassifier(n_neighbors=3, p=3)
    with pytest.raises(NotImplementedError):
        fl.fit(_KNN_X, _KNN_Y)


def test_knn_unsupported_metric_raises():
    """GREEN (#2138): a non-minkowski/euclidean metric raises (NOT-STARTED #876)."""
    fl = ferrolearn.KNeighborsClassifier(n_neighbors=3, metric="manhattan")
    with pytest.raises(NotImplementedError):
        fl.fit(_KNN_X, _KNN_Y)


def test_knn_euclidean_metric_supported():
    """GREEN (#2138): metric='euclidean' is accepted (== minkowski p=2)."""
    from sklearn.neighbors import KNeighborsClassifier as SkKNN

    sk = SkKNN(n_neighbors=3, metric="euclidean").fit(_KNN_X, _KNN_Y)
    fl = ferrolearn.KNeighborsClassifier(n_neighbors=3, metric="euclidean").fit(
        _KNN_X, _KNN_Y
    )
    assert np.asarray(fl.predict(_KNN_XQ)).tolist() == sk.predict(_KNN_XQ).tolist()


def test_knn_euclidean_metric_ignores_p():
    """GREEN (#2148): metric='euclidean' ignores p (sklearn _base.py:526-538) —
    KNeighborsClassifier(metric='euclidean', p=3) is VALID and matches sklearn,
    NOT a NotImplementedError. p is only consumed for metric='minkowski'."""
    from sklearn.neighbors import KNeighborsClassifier as SkKNN

    sk = SkKNN(n_neighbors=3, metric="euclidean", p=3).fit(_KNN_X, _KNN_Y)
    fl = ferrolearn.KNeighborsClassifier(
        n_neighbors=3, metric="euclidean", p=3
    ).fit(_KNN_X, _KNN_Y)
    assert np.asarray(fl.predict(_KNN_XQ)).tolist() == sk.predict(_KNN_XQ).tolist()


def test_knn_invalid_weights_string_is_valueerror():
    """GREEN (#2149): an invalid weights STRING raises a ValueError (sklearn's
    InvalidParameterError is a ValueError subclass), NOT NotImplementedError —
    so the sklearn-idiomatic `except ValueError` catches it. (Callable weights
    remain NotImplementedError #876, tested separately.)"""
    fl = ferrolearn.KNeighborsClassifier(n_neighbors=3, weights="foo")
    with pytest.raises(ValueError):
        fl.fit(_KNN_X, _KNN_Y)


def test_knn_unsupported_callable_weights_raises():
    """GREEN (#2138): callable weights raise NotImplementedError (#876)."""
    fl = ferrolearn.KNeighborsClassifier(
        n_neighbors=3, weights=lambda d: 1.0 / (d + 1e-9)
    )
    with pytest.raises(NotImplementedError):
        fl.fit(_KNN_X, _KNN_Y)


def test_knn_unsupported_metric_params_raises():
    """GREEN (#2138): metric_params != None raises NotImplementedError (#876)."""
    fl = ferrolearn.KNeighborsClassifier(
        n_neighbors=3, metric_params={"w": 2.0}
    )
    with pytest.raises(NotImplementedError):
        fl.fit(_KNN_X, _KNN_Y)


def test_knn_check_estimator_green():
    """GREEN (#2138): KNeighborsClassifier passes sklearn's check_estimator.

    All 9 ctor params are plain stored attributes (get_params/set_params/clone
    round-trip), the prerequisite for `parametrize_with_checks`.
    """
    from sklearn.utils.estimator_checks import check_estimator

    # check_estimator raises on the first failed check; returning normally =
    # green. The default-param estimator uses the SHIPPED uniform/minkowski(p=2)
    # path, so no NotImplementedError is triggered.
    check_estimator(ferrolearn.KNeighborsClassifier(n_neighbors=3))




# ===========================================================================
# (4) KNeighborsClassifier DIVERGENCES found by adversarial re-audit of #2138
#     (live sklearn 1.5.2 oracle; R-CHAR-3 — no expected value copied from
#     the ferrolearn side; every expected value is the sklearn oracle result).
# ===========================================================================


def test_red_knn_distance_tie_break_lowest_index_diverges():
    """RED divergence (tracking #2139): ferrolearn's tree algorithms break a
    k-th-boundary DISTANCE TIE by a different rule than sklearn.

    sklearn selects the k nearest by `np.argpartition(dist, n_neighbors-1)`
    then a stable `np.argsort(..., kind="mergesort")`
    (`sklearn/neighbors/_base.py:738` and `:740-741`; cf. `:277`), so when
    several training points sit at exactly the k-th distance, the LOWEST
    training INDEX wins the tie.

    Dataset: query [0,0] exactly coincides with idx0 (class 5). idx1 (class 1),
    idx2 (class 0), idx3 (class 0) are all at distance 1 -> a 3-way tie for the
    single remaining (k=2) slot. sklearn keeps idx1 (lowest index) ->
    proba [0, 0.5, 0.5], predict 1.

    ferrolearn 'auto'/'kd_tree'/'ball_tree' instead keep idx2 -> proba
    [0.5, 0, 0.5], predict 0. (ferrolearn 'brute' happens to match sklearn,
    which is itself a separate divergence: see test below.)
    """
    from sklearn.neighbors import KNeighborsClassifier as SkKNN

    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    y = np.array([5, 1, 0, 0])
    Xq = np.array([[0.0, 0.0]])

    sk = SkKNN(n_neighbors=2, algorithm="auto").fit(X, y)
    fl = ferrolearn.KNeighborsClassifier(n_neighbors=2, algorithm="auto").fit(X, y)

    sk_pred = sk.predict(Xq)
    sk_proba = sk.predict_proba(Xq)
    fl_pred = np.asarray(fl.predict(Xq))
    fl_proba = np.asarray(fl.predict_proba(Xq))

    # Oracle: lowest-index tie-break keeps idx1 (class 1).
    assert sk_pred.tolist() == [1]
    np.testing.assert_allclose(sk_proba, [[0.0, 0.5, 0.5]], atol=1e-12)

    # FAILS today: ferrolearn 'auto' keeps idx2 (class 0) -> predict 0.
    assert fl_pred.tolist() == sk_pred.tolist()
    np.testing.assert_allclose(fl_proba, sk_proba, atol=1e-9)


def test_red_knn_algorithm_not_result_identical_on_ties():
    """RED divergence (tracking #2139): the #2138 claim that all `algorithm`
    values give the SAME predict/predict_proba is FALSE on tie-heavy data.

    On the same k-th-boundary tie dataset, ferrolearn 'brute' returns a
    DIFFERENT result than ferrolearn 'auto'/'kd_tree'/'ball_tree':
      brute      -> predict 1, proba [0, 0.5, 0.5]   (matches sklearn here)
      auto/trees -> predict 0, proba [0.5, 0, 0.5]
    sklearn returns the SAME result for every algorithm string (search
    strategy only); ferrolearn does not. This pins the internal inconsistency
    independently of which one matches the oracle.
    """
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    y = np.array([5, 1, 0, 0])
    Xq = np.array([[0.0, 0.0]])

    preds = {}
    for algo in ("brute", "auto", "kd_tree", "ball_tree"):
        fl = ferrolearn.KNeighborsClassifier(
            n_neighbors=2, algorithm=algo
        ).fit(X, y)
        preds[algo] = np.asarray(fl.predict(Xq)).tolist()

    # FAILS today: 'brute' -> [1] but 'auto'/'kd_tree'/'ball_tree' -> [0].
    assert preds["brute"] == preds["auto"] == preds["kd_tree"] == preds["ball_tree"], (
        f"algorithm not result-identical: {preds}"
    )


def test_red_knn_n_neighbors_gt_n_samples_must_raise():
    """RED divergence (tracking #2140): n_neighbors > n_samples must raise.

    sklearn raises ValueError at predict/kneighbors time
    (`sklearn/neighbors/_base.py:828-838`):
      "Expected n_neighbors <= n_samples_fit, but n_neighbors = 10,
       n_samples_fit = 5, n_samples = 1".
    ferrolearn silently returns a prediction (it clamps to the available
    neighbors), so an over-large k goes unflagged.
    """
    from sklearn.neighbors import KNeighborsClassifier as SkKNN

    X = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]])
    y = np.array([0, 1, 0, 1, 0])
    Xq = X[:1]

    # Oracle: sklearn raises at predict time.
    sk = SkKNN(n_neighbors=10).fit(X, y)
    with pytest.raises(ValueError, match="n_neighbors"):
        sk.predict(Xq)

    # FAILS today: ferrolearn returns a prediction instead of raising.
    fl = ferrolearn.KNeighborsClassifier(n_neighbors=10).fit(X, y)
    with pytest.raises(ValueError):
        fl.predict(Xq)


# ---------------------------------------------------------------------------
# LogisticRegression: sample_weight / class_weight / n_iter_ / random_state /
# n_jobs (downstream #450/#451/#445/#452).
# Every expected value is the LIVE sklearn 1.5.2 oracle (R-CHAR-3).
# ---------------------------------------------------------------------------


def _lr_binary_imbalanced():
    X = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [2.5, 3.5], [5.0, 6.0], [7.0, 8.0]])
    y = np.array([0, 0, 0, 0, 1, 1])
    return X, y


def test_logreg_sample_weight_matches_sklearn_binary():
    """Weighted coef_/intercept_/predict_proba vs sklearn (binary)."""
    from sklearn.linear_model import LogisticRegression as SkLR

    X = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [5.0, 6.0], [6.0, 7.0], [7.0, 8.0]])
    y = np.array([0, 0, 0, 1, 1, 1])
    w = np.array([1.0, 2.0, 1.0, 3.0, 1.0, 2.0])

    sk = SkLR(C=1.0, solver="lbfgs", max_iter=3000, tol=1e-9).fit(X, y, sample_weight=w)
    fl = ferrolearn.LogisticRegression(C=1.0, max_iter=3000, tol=1e-9).fit(
        X, y, sample_weight=w
    )
    assert np.abs(fl.coef_ - sk.coef_).max() < 1e-6, (fl.coef_, sk.coef_)
    assert np.abs(fl.intercept_ - sk.intercept_).max() < 1e-5
    assert np.abs(fl.predict_proba(X) - sk.predict_proba(X)).max() < 1e-5


def test_logreg_sample_weight_matches_sklearn_3class():
    """Weighted 3-class multinomial coef_ vs sklearn."""
    from sklearn.linear_model import LogisticRegression as SkLR

    X = np.array(
        [
            [0.0, 0.0], [0.5, 0.0], [0.0, 0.5],
            [5.0, 0.0], [5.5, 0.0], [5.0, 0.5],
            [0.0, 5.0], [0.5, 5.0], [0.0, 5.5],
        ]
    )
    y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    w = np.array([1.0, 2.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 1.0])

    sk = SkLR(C=10.0, solver="lbfgs", max_iter=5000, tol=1e-8).fit(X, y, sample_weight=w)
    fl = ferrolearn.LogisticRegression(C=10.0, max_iter=5000, tol=1e-8).fit(
        X, y, sample_weight=w
    )
    # The multinomial coef_ matrix is non-identifiable up to a per-row constant,
    # but predict_proba IS identifiable: assert weighted-fit parity there (the
    # binding's multiclass coef_ exposure collapse is the separate #2170 item).
    assert np.abs(fl.predict_proba(X) - sk.predict_proba(X)).max() < 5e-3, (
        fl.predict_proba(X),
        sk.predict_proba(X),
    )


def test_logreg_sample_weight_none_is_unweighted():
    """sample_weight=None == not passing sample_weight (unweighted fit)."""
    X = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [5.0, 6.0], [6.0, 7.0], [7.0, 8.0]])
    y = np.array([0, 0, 0, 1, 1, 1])
    a = ferrolearn.LogisticRegression(C=1.0, max_iter=2000, tol=1e-9).fit(X, y)
    b = ferrolearn.LogisticRegression(C=1.0, max_iter=2000, tol=1e-9).fit(
        X, y, sample_weight=None
    )
    assert np.array_equal(a.coef_, b.coef_)
    assert np.array_equal(a.intercept_, b.intercept_)


def test_logreg_integer_sample_weight_equals_row_dup():
    """Integer sample_weight == duplicating those rows (sklearn invariant)."""
    X = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [5.0, 6.0], [6.0, 7.0], [7.0, 8.0]])
    y = np.array([0, 0, 0, 1, 1, 1])
    w = np.array([2.0, 1.0, 1.0, 1.0, 1.0, 2.0])
    weighted = ferrolearn.LogisticRegression(C=1.0, max_iter=5000, tol=1e-10).fit(
        X, y, sample_weight=w
    )
    Xd = np.vstack([X[0:1], X, X[5:6]])
    yd = np.concatenate([y[0:1], y, y[5:6]])
    dup = ferrolearn.LogisticRegression(C=1.0, max_iter=5000, tol=1e-10).fit(Xd, yd)
    assert np.abs(weighted.coef_ - dup.coef_).max() < 1e-2, (weighted.coef_, dup.coef_)


def test_logreg_class_weight_balanced_matches_sklearn():
    """class_weight='balanced' vs sklearn (imbalanced 2-class)."""
    from sklearn.linear_model import LogisticRegression as SkLR

    X, y = _lr_binary_imbalanced()
    sk = SkLR(C=1.0, solver="lbfgs", max_iter=5000, tol=1e-10, class_weight="balanced").fit(X, y)
    fl = ferrolearn.LogisticRegression(
        C=1.0, max_iter=5000, tol=1e-10, class_weight="balanced"
    ).fit(X, y)
    assert np.abs(fl.coef_ - sk.coef_).max() < 1e-5, (fl.coef_, sk.coef_)
    assert np.abs(fl.intercept_ - sk.intercept_).max() < 1e-4


def test_logreg_class_weight_dict_matches_sklearn():
    """class_weight={0:1,1:3} vs sklearn."""
    from sklearn.linear_model import LogisticRegression as SkLR

    X, y = _lr_binary_imbalanced()
    cw = {0: 1.0, 1: 3.0}
    sk = SkLR(C=1.0, solver="lbfgs", max_iter=5000, tol=1e-10, class_weight=cw).fit(X, y)
    fl = ferrolearn.LogisticRegression(
        C=1.0, max_iter=5000, tol=1e-10, class_weight=cw
    ).fit(X, y)
    assert np.abs(fl.coef_ - sk.coef_).max() < 1e-5, (fl.coef_, sk.coef_)


def test_logreg_class_weight_balanced_3class_matches_sklearn():
    """class_weight='balanced' on an imbalanced 3-class problem vs sklearn."""
    from sklearn.linear_model import LogisticRegression as SkLR

    X = np.array(
        [
            [0.0, 0.0], [0.5, 0.0], [0.0, 0.5],
            [5.0, 0.0], [5.5, 0.0],
            [0.0, 5.0], [0.5, 5.0], [0.0, 5.5], [6.0, 6.0],
        ]
    )
    y = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2])
    sk = SkLR(C=10.0, solver="lbfgs", max_iter=5000, tol=1e-8, class_weight="balanced").fit(X, y)
    fl = ferrolearn.LogisticRegression(
        C=10.0, max_iter=5000, tol=1e-8, class_weight="balanced"
    ).fit(X, y)
    # Identifiable comparison via predict_proba (see #2170 for the multiclass
    # coef_ exposure collapse, which is orthogonal to the class_weight fit).
    assert np.abs(fl.predict_proba(X) - sk.predict_proba(X)).max() < 5e-3, (
        fl.predict_proba(X),
        sk.predict_proba(X),
    )


def test_logreg_class_weight_none_unchanged():
    """class_weight=None == the plain unweighted fit."""
    X, y = _lr_binary_imbalanced()
    plain = ferrolearn.LogisticRegression(C=1.0, max_iter=3000, tol=1e-9).fit(X, y)
    none_cw = ferrolearn.LogisticRegression(
        C=1.0, max_iter=3000, tol=1e-9, class_weight=None
    ).fit(X, y)
    assert np.array_equal(plain.coef_, none_cw.coef_)


def test_logreg_class_weight_composes_with_sample_weight():
    """class_weight + sample_weight together == sklearn (the product folds in)."""
    from sklearn.linear_model import LogisticRegression as SkLR

    X, y = _lr_binary_imbalanced()
    cw = {0: 1.0, 1: 3.0}
    usw = np.array([1.0, 2.0, 1.0, 1.0, 1.0, 2.0])
    sk = SkLR(C=1.0, solver="lbfgs", max_iter=5000, tol=1e-10, class_weight=cw).fit(
        X, y, sample_weight=usw
    )
    fl = ferrolearn.LogisticRegression(
        C=1.0, max_iter=5000, tol=1e-10, class_weight=cw
    ).fit(X, y, sample_weight=usw)
    assert np.abs(fl.coef_ - sk.coef_).max() < 1e-5, (fl.coef_, sk.coef_)


def test_logreg_n_iter_contract():
    """n_iter_ is a positive int <= max_iter, shape (1,), deterministic.

    R-DEV-7: ferrolearn's L-BFGS != scipy's, so we assert the CONTRACT, not
    equality with sklearn's count (which is also shape (1,) here).
    """
    from sklearn.linear_model import LogisticRegression as SkLR

    X = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [5.0, 6.0], [6.0, 7.0], [7.0, 8.0]])
    y = np.array([0, 0, 0, 1, 1, 1])
    sk = SkLR(C=1.0, solver="lbfgs", max_iter=200, tol=1e-4).fit(X, y)
    fl = ferrolearn.LogisticRegression(C=1.0, max_iter=200, tol=1e-4).fit(X, y)
    # Shape parity with sklearn's n_iter_.
    assert fl.n_iter_.shape == sk.n_iter_.shape == (1,)
    assert fl.n_iter_.dtype == np.int32
    assert 1 <= int(fl.n_iter_[0]) <= 200
    # Determinism.
    fl2 = ferrolearn.LogisticRegression(C=1.0, max_iter=200, tol=1e-4).fit(X, y)
    assert int(fl.n_iter_[0]) == int(fl2.n_iter_[0])


def test_logreg_random_state_n_jobs_noop_and_get_params():
    """random_state/n_jobs are accepted, no-op on the result, and exposed via
    get_params + survive clone (sklearn API parity)."""
    from sklearn.base import clone

    X = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [5.0, 6.0], [6.0, 7.0], [7.0, 8.0]])
    y = np.array([0, 0, 0, 1, 1, 1])
    base = ferrolearn.LogisticRegression(C=1.0, max_iter=2000, tol=1e-9).fit(X, y)
    rs = ferrolearn.LogisticRegression(
        C=1.0, max_iter=2000, tol=1e-9, random_state=42, n_jobs=4
    ).fit(X, y)
    assert np.array_equal(base.coef_, rs.coef_)
    assert np.array_equal(base.intercept_, rs.intercept_)

    m = ferrolearn.LogisticRegression(
        C=2.0, class_weight="balanced", random_state=7, n_jobs=3
    )
    params = m.get_params()
    assert params["random_state"] == 7
    assert params["n_jobs"] == 3
    assert params["class_weight"] == "balanced"
    assert clone(m).get_params() == params
