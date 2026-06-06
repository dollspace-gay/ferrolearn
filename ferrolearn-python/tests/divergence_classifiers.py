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


