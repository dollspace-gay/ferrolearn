//! Adversarial divergence pins / value-contract guards for
//! `ferrolearn-neighbors/src/knn.rs` (`KNeighborsClassifier` /
//! `FittedKNeighborsClassifier` + `KNeighborsRegressor` /
//! `FittedKNeighborsRegressor`: `fit` / `predict` / `predict_proba` / `score` /
//! `kneighbors`) against the live scikit-learn 1.5.2 oracle
//! (`from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor`,
//! mirroring `sklearn/neighbors/_classification.py` + `_regression.py` +
//! `_base.py`).
//!
//! Every expected value below is captured from a live `python3 -c "..."` run of
//! sklearn 1.5.2, executed from `/tmp` (R-CHAR-3 — never literal-copied from the
//! ferrolearn side). The exact oracle call and its output are quoted above each
//! assertion block.
//!
//! Design doc: `.design/neighbors/knn.md` (commit ec39edda).
//! Upstream: `sklearn/neighbors/_classification.py` (`KNeighborsClassifier`),
//! `sklearn/neighbors/_regression.py` (`KNeighborsRegressor`),
//! `sklearn/neighbors/_base.py` (`KNeighborsMixin.kneighbors`, `_get_weights`,
//! the fit-vs-query timing of the `n_neighbors <= n_samples_fit` check).
//!
//! ferrolearn classifier labels are `Array1<usize>` — fixtures use 0/1 integer
//! labels.
//!
//! GREEN guards (must PASS — SHIPPED value contracts, R-DEV-1/3):
//!   * `green_classifier_predict_value_uniform_and_distance` — clf `predict`.
//!   * `green_classifier_predict_proba_value`                — clf `predict_proba`.
//!   * `green_classifier_tiebreak_smallest_label`            — argmax first-max.
//!   * `green_regressor_predict_value_uniform_and_distance`  — reg `predict`.
//!   * `green_classifier_score_accuracy`                     — clf `score`.
//!   * `green_kneighbors_value`                              — shared k-NN search.
//!   * `green_kneighbors_k_too_large_errors`                 — guard EXISTS.
//!
//! RED pin (must FAIL now — the one deterministic, single-file-fixable divergence,
//! present in BOTH `fit` methods):
//!   * `divergence_fit_does_not_error_when_n_neighbors_gt_n_samples` — REQ-8 / #874.
//!
//! ==========================================================================
//! NOT-STARTED / MISSING-SURFACE divergences — DOCUMENTED ONLY, no forced test.
//! ==========================================================================
//! Following the `divergence_nearest_neighbors.rs` / `divergence_balltree.rs`
//! sibling precedent, the divergences below are NOT pinned with a committed RED
//! test because they require the generator to add API surface (a discriminator may
//! not author production code), or are inexpressible through the current public
//! API:
//!
//! **regressor multi-output 2-D `y`** (REQ-9 / #875): `FittedKNeighborsRegressor`
//! stores `y_train: Array1<F>` and `impl Fit<Array2<F>, Array1<F>>` — 1-D `y`
//! only. sklearn `KNeighborsRegressor.predict` reshapes `_y` to 2-D and returns
//! `(n_queries, n_outputs)` when `_y.ndim > 1` (`_regression.py:253-270`). Live
//! oracle: `KNeighborsRegressor(n_neighbors=3).fit(X5x1, Y5x2).predict([[1.0]])`
//! -> shape `(1,2)` value `[[1.0, 200.0]]`. MISSING SURFACE — there is no 2-D `y`
//! `Fit` impl to call, so it is not runtime-pinnable without inventing API.
//!
//! **missing constructor params + callable weights** (REQ-10 / #876): ferrolearn
//! `KNeighbors{Classifier,Regressor}` expose `n_neighbors` / `algorithm` /
//! `weights ∈ {Uniform, Distance}` but NO `leaf_size=30` (`_classification.py:199`),
//! `p=2` (`:200`), `metric='minkowski'` (`:201`), `metric_params=None` (`:202`),
//! `n_jobs=None` (`:203`); and `Weights` has no callable variant (sklearn `weights`
//! accepts a callable, `:190`). Euclidean-only. MISSING FIELDS — not runtime-
//! pinnable without inventing surface.
//!
//! **PyO3 binding under-exposes the surface** (REQ-11 / #877): `_RsKNeighbors
//! Classifier` (`ferrolearn-python/src/classifiers.rs:305-353`) exposes only
//! `new(n_neighbors)` / `fit` / `predict` / `classes_` — NO `predict_proba`, NO
//! `score`, NO `weights`/`algorithm` knob. `_RsKNeighborsRegressor`
//! (`extras.rs:460`, via the `py_regressor!` macro at `extras.rs:17-60`) exposes
//! only `new(n_neighbors)` / `fit` / `predict` — NO `weights`, and (contrary to the
//! design doc) NO `score` (the macro generates no `score` method). This is a
//! ferrolearn-python (pytest) concern, not a library `#[test]`; documented only.
//!
//! **ferray substrate** (REQ-12 / #878): `knn.rs` imports `ndarray::{Array1,
//! Array2}` + `num_traits::Float`, not `ferray-core` (R-SUBSTRATE). Not behavior-
//! pinnable.
//!
//! **k > n at predict/kneighbors — type/message** (#872-message portion): both
//! ferrolearn and sklearn ERROR when `k > n_samples_fit` at `kneighbors`/`predict`
//! time, but the type (`FerroError::InsufficientSamples` vs `ValueError`) and the
//! message ("Expected n_neighbors <= n_samples_fit, but n_neighbors = {k},
//! n_samples_fit = {n}, n_samples = {nq}", `_base.py:828-832`) differ. The guard-
//! EXISTS contract is SHIPPED (pinned green by `green_kneighbors_k_too_large_errors`
//! below); the EXACT-MESSAGE/type match is NOT-STARTED. Documented, no RED.

use ferrolearn_core::traits::{Fit, Predict};
use ferrolearn_neighbors::{KNeighborsClassifier, KNeighborsRegressor, Weights};
use ndarray::{Array1, array};

// ===========================================================================
// GREEN 1 — clf `predict` value parity (uniform + distance), tie-free fixture.
// ===========================================================================
//
// Oracle (run from /tmp):
//   python3 -c "import numpy as np; from sklearn.neighbors import \
//   KNeighborsClassifier as C; \
//   X=np.array([[0.,0.],[1.,0.],[0.,1.],[5.,5.],[6.,5.],[5.,6.]]); \
//   y=np.array([0,0,0,1,1,1]); Xq=np.array([[0.2,0.1],[5.2,5.1]]); \
//   print(C(n_neighbors=3).fit(X,y).predict(Xq).tolist(), \
//   C(n_neighbors=3,weights='distance').fit(X,y).predict(Xq).tolist())"
//   -> predict uniform : [0, 1]
//      predict distance: [0, 1]
#[test]
fn green_classifier_predict_value_uniform_and_distance() {
    let x = array![
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [5.0, 5.0],
        [6.0, 5.0],
        [5.0, 6.0]
    ];
    let y: Array1<usize> = array![0, 0, 0, 1, 1, 1];
    let xq = array![[0.2, 0.1], [5.2, 5.1]];

    // uniform (default weights)
    let fitted_u = KNeighborsClassifier::<f64>::new()
        .with_n_neighbors(3)
        .fit(&x, &y)
        .expect("fit should succeed");
    let preds_u = fitted_u.predict(&xq).expect("predict should succeed");
    assert_eq!(
        preds_u.to_vec(),
        vec![0, 1],
        "uniform predict must match sklearn [0, 1]"
    );

    // distance weights
    let fitted_d = KNeighborsClassifier::<f64>::new()
        .with_n_neighbors(3)
        .with_weights(Weights::Distance)
        .fit(&x, &y)
        .expect("fit should succeed");
    let preds_d = fitted_d.predict(&xq).expect("predict should succeed");
    assert_eq!(
        preds_d.to_vec(),
        vec![0, 1],
        "distance predict must match sklearn [0, 1]"
    );
}

// ===========================================================================
// GREEN 2 — clf `predict_proba` value (distance + uniform + zero-distance).
// ===========================================================================
//
// Oracle (run from /tmp):
//   python3 -c "import numpy as np; from sklearn.neighbors import \
//   KNeighborsClassifier as C; X=np.array([[0.],[1.],[2.],[10.],[11.]]); \
//   y=np.array([0,1,0,1,1]); q=np.array([[1.5]]); \
//   print(C(n_neighbors=3,weights='distance').fit(X,y).predict_proba(q).tolist(), \
//   C(n_neighbors=3).fit(X,y).predict_proba(q).tolist())"
//   -> proba distance: [[0.5714285714285715, 0.4285714285714286]]
//      proba uniform : [[0.6666666666666666, 0.3333333333333333]]
//
// Zero-distance row (query coincides with a training point) oracle (/tmp):
//   python3 -c "import numpy as np; from sklearn.neighbors import \
//   KNeighborsClassifier as C; X=np.array([[0.],[1.],[2.]]); \
//   y=np.array([0,1,0]); q=np.array([[1.0]]); \
//   print(C(n_neighbors=3,weights='distance').fit(X,y).predict_proba(q).tolist())"
//   -> proba zerodist: [[0.0, 1.0]]
//
// NOTE (consumer-backing, R-DEFER-2): `FittedKNeighborsClassifier::predict_proba`
// has NO non-test production consumer — `_RsKNeighborsClassifier`
// (`classifiers.rs:305-353`) exposes only fit/predict/classes_; the only
// `predict_proba` binding in `classifiers.rs` (line 404) belongs to `RsGaussianNB`.
// This guard is value-correct but test-only-consumed; reported in the critic note.
#[test]
fn green_classifier_predict_proba_value() {
    let x = array![[0.0], [1.0], [2.0], [10.0], [11.0]];
    let y: Array1<usize> = array![0, 1, 0, 1, 1];
    let q = array![[1.5]];

    // distance
    let fitted_d = KNeighborsClassifier::<f64>::new()
        .with_n_neighbors(3)
        .with_weights(Weights::Distance)
        .fit(&x, &y)
        .expect("fit should succeed");
    let proba_d = fitted_d.predict_proba(&q).expect("predict_proba");
    let exp_d = [0.571_428_571_428_571_5, 0.428_571_428_571_428_6];
    for (j, e) in exp_d.iter().enumerate() {
        assert!(
            (proba_d[[0, j]] - e).abs() < 1e-12,
            "distance proba[{j}]: got {} expected {}",
            proba_d[[0, j]],
            e
        );
    }

    // uniform
    let fitted_u = KNeighborsClassifier::<f64>::new()
        .with_n_neighbors(3)
        .fit(&x, &y)
        .expect("fit should succeed");
    let proba_u = fitted_u.predict_proba(&q).expect("predict_proba");
    let exp_u = [0.666_666_666_666_666_6, 0.333_333_333_333_333_3];
    for (j, e) in exp_u.iter().enumerate() {
        assert!(
            (proba_u[[0, j]] - e).abs() < 1e-12,
            "uniform proba[{j}]: got {} expected {}",
            proba_u[[0, j]],
            e
        );
    }

    // zero-distance row -> coincident point takes all weight.
    let xz = array![[0.0], [1.0], [2.0]];
    let yz: Array1<usize> = array![0, 1, 0];
    let qz = array![[1.0]];
    let fitted_z = KNeighborsClassifier::<f64>::new()
        .with_n_neighbors(3)
        .with_weights(Weights::Distance)
        .fit(&xz, &yz)
        .expect("fit should succeed");
    let proba_z = fitted_z.predict_proba(&qz).expect("predict_proba");
    let exp_z = [0.0, 1.0];
    for (j, e) in exp_z.iter().enumerate() {
        assert!(
            (proba_z[[0, j]] - e).abs() < 1e-12,
            "zero-dist proba[{j}]: got {} expected {}",
            proba_z[[0, j]],
            e
        );
    }
}

// ===========================================================================
// GREEN 3 — clf vote-tie tie-break: smallest class label wins (np.argmax first).
// ===========================================================================
//
// Oracle (/tmp):
//   python3 -c "import numpy as np; from sklearn.neighbors import \
//   KNeighborsClassifier as C; \
//   print(C(n_neighbors=2).fit(np.array([[0.],[1.]]),np.array([0,1])).predict( \
//   np.array([[0.5]])).tolist())"
//   -> [0]
// k=2, query 0.5 equidistant from both -> 1-1 vote tie -> smallest label 0.
#[test]
fn green_classifier_tiebreak_smallest_label() {
    let x = array![[0.0], [1.0]];
    let y: Array1<usize> = array![0, 1];
    let q = array![[0.5]];

    let fitted = KNeighborsClassifier::<f64>::new()
        .with_n_neighbors(2)
        .fit(&x, &y)
        .expect("fit should succeed");
    let preds = fitted.predict(&q).expect("predict should succeed");
    assert_eq!(
        preds.to_vec(),
        vec![0],
        "even-k vote tie must break to the smallest label (sklearn np.argmax first-max)"
    );
}

// ===========================================================================
// GREEN 4 — reg `predict` value (uniform + distance + zero-distance).
// ===========================================================================
//
// Oracle (/tmp):
//   python3 -c "import numpy as np; from sklearn.neighbors import \
//   KNeighborsRegressor as R; \
//   X=np.array([[0.],[1.],[2.],[3.],[100.]]); y=np.array([0.,10.,20.,30.,1000.]); \
//   q=np.array([[1.0]]); \
//   print(R(n_neighbors=3).fit(X,y).predict(q).tolist(), \
//   R(n_neighbors=3,weights='distance').fit(X,y).predict(q).tolist(), \
//   R(n_neighbors=3,weights='distance').fit(X,y).predict(np.array([[2.0]])).tolist())"
//   -> reg uniform : [10.0]
//      reg distance: [10.0]
//      reg zerodist: [20.0]   (query 2.0 coincides with training row -> all weight)
#[test]
fn green_regressor_predict_value_uniform_and_distance() {
    let x = array![[0.0], [1.0], [2.0], [3.0], [100.0]];
    let y: Array1<f64> = array![0.0, 10.0, 20.0, 30.0, 1000.0];
    let q = array![[1.0]];

    // uniform
    let fitted_u = KNeighborsRegressor::<f64>::new()
        .with_n_neighbors(3)
        .fit(&x, &y)
        .expect("fit should succeed");
    let preds_u = fitted_u.predict(&q).expect("predict should succeed");
    assert!(
        (preds_u[0] - 10.0).abs() < 1e-12,
        "uniform predict must be 10.0, got {}",
        preds_u[0]
    );

    // distance
    let fitted_d = KNeighborsRegressor::<f64>::new()
        .with_n_neighbors(3)
        .with_weights(Weights::Distance)
        .fit(&x, &y)
        .expect("fit should succeed");
    let preds_d = fitted_d.predict(&q).expect("predict should succeed");
    assert!(
        (preds_d[0] - 10.0).abs() < 1e-12,
        "distance predict must be 10.0, got {}",
        preds_d[0]
    );

    // zero-distance query (coincides with training row x=2.0, y=20.0).
    let qz = array![[2.0]];
    let preds_z = fitted_d.predict(&qz).expect("predict should succeed");
    assert!(
        (preds_z[0] - 20.0).abs() < 1e-12,
        "zero-dist distance predict must be 20.0 (coincident point takes all weight), got {}",
        preds_z[0]
    );
}

// ===========================================================================
// GREEN 5 — clf `score` mean accuracy parity.
// ===========================================================================
//
// Oracle (/tmp):
//   python3 -c "import numpy as np; from sklearn.neighbors import \
//   KNeighborsClassifier as C; X=np.array([[0.],[1.],[5.],[6.]]); \
//   y=np.array([0,0,1,1]); print(C(n_neighbors=1).fit(X,y).score(X,y))"
//   -> 1.0
//
// NOTE (consumer-backing, R-DEFER-2): `FittedKNeighborsClassifier::score` has NO
// non-test production consumer — `_RsKNeighborsClassifier` exposes no `score`, and
// `grep score ferrolearn-python/src` is EMPTY. Value-correct but test-only-
// consumed; reported in the critic note.
#[test]
fn green_classifier_score_accuracy() {
    let x = array![[0.0], [1.0], [5.0], [6.0]];
    let y: Array1<usize> = array![0, 0, 1, 1];

    let fitted = KNeighborsClassifier::<f64>::new()
        .with_n_neighbors(1)
        .fit(&x, &y)
        .expect("fit should succeed");
    let score = fitted.score(&x, &y).expect("score should succeed");
    assert!(
        (score - 1.0).abs() < 1e-12,
        "k=1 self-score must be 1.0 (perfect accuracy), got {score}"
    );
}

// ===========================================================================
// GREEN 6 — shared k-NN search value (`kneighbors`): nearest-first (d, i).
// ===========================================================================
//
// Oracle (/tmp), tie-free query (distances all distinct):
//   python3 -c "import numpy as np; from sklearn.neighbors import \
//   KNeighborsRegressor as R; \
//   X=np.array([[0.,0.],[1.,0.],[0.,1.],[5.,5.],[6.,5.],[5.,6.]]); \
//   q=np.array([[0.2,0.1]]); \
//   d,i=R(n_neighbors=3).fit(X,np.zeros(6)).kneighbors(q); \
//   print(d.tolist(), i.tolist())"
//   -> kneighbors d: [[0.223606797749979, 0.8062257748298549, 0.9219544457292888]]
//      kneighbors i: [[0, 1, 2]]
#[test]
fn green_kneighbors_value() {
    let x = array![
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [5.0, 5.0],
        [6.0, 5.0],
        [5.0, 6.0]
    ];
    let y: Array1<f64> = Array1::zeros(6);
    let q = array![[0.2, 0.1]];

    let fitted = KNeighborsRegressor::<f64>::new()
        .with_n_neighbors(3)
        .fit(&x, &y)
        .expect("fit should succeed");
    let (d, i) = fitted.kneighbors(&q, Some(3)).expect("kneighbors");

    assert_eq!(d.dim(), (1, 3), "kneighbors shape");
    assert_eq!(i[[0, 0]], 0, "kneighbors idx0");
    assert_eq!(i[[0, 1]], 1, "kneighbors idx1");
    assert_eq!(i[[0, 2]], 2, "kneighbors idx2");
    let exp = [
        0.223_606_797_749_979,
        0.806_225_774_829_854_9,
        0.921_954_445_729_288_8,
    ];
    for (j, e) in exp.iter().enumerate() {
        assert!(
            (d[[0, j]] - e).abs() < 1e-12,
            "kneighbors dist[{j}]: got {} expected {}",
            d[[0, j]],
            e
        );
    }
}

// ===========================================================================
// GREEN 7 — k > n_samples error guard EXISTS at kneighbors/predict time.
// ===========================================================================
//
// sklearn raises a ValueError at query time; ferrolearn returns
// Err(InsufficientSamples). Here we pin only that the guard EXISTS (the SHIPPED
// contract); the exact type/message divergence (#872-message portion) is
// documented in the header, not pinned. Oracle (/tmp):
//   python3 -c "import numpy as np; from sklearn.neighbors import \
//   KNeighborsRegressor as R; \
//   R(n_neighbors=3).fit(np.array([[0.],[1.],[2.]]),np.zeros(3)).kneighbors( \
//   np.array([[0.5]]), n_neighbors=100)"
//   -> ValueError: Expected n_neighbors <= n_samples_fit, but n_neighbors = 100,
//      n_samples_fit = 3, n_samples = 1
#[test]
fn green_kneighbors_k_too_large_errors() {
    let x = array![[0.0], [1.0], [2.0]];
    let y: Array1<f64> = Array1::zeros(3);
    let q = array![[0.5]];

    // n_neighbors=3 lets fit succeed (3 rows == k), then kneighbors(k=100) errors.
    let fitted = KNeighborsRegressor::<f64>::new()
        .with_n_neighbors(3)
        .fit(&x, &y)
        .expect("fit should succeed");
    assert!(
        fitted.kneighbors(&q, Some(100)).is_err(),
        "k=100 > n_samples_fit=3 must error at kneighbors time (sklearn raises \
         ValueError; ferrolearn raises FerroError — the type/message divergence is \
         documented, not pinned)"
    );
}

// ===========================================================================
// RED 1 — REQ-8 / #874 (DIVERGENCE, must FAIL now): fit timing in BOTH fits.
// ===========================================================================
//
// sklearn does NOT validate n_neighbors vs n_samples at FIT time — the fit
// succeeds (in BOTH KNeighborsClassifier and KNeighborsRegressor) and the
// `n_neighbors <= n_samples_fit` check is deferred to query (predict/kneighbors)
// time (`_base.py:828-832`). Oracle (/tmp):
//   python3 -c "from sklearn.neighbors import KNeighborsClassifier as C, \
//   KNeighborsRegressor as R; import numpy as np; \
//   C(n_neighbors=5).fit(np.array([[0.],[1.],[2.]]), np.array([0,1,0])); \
//   R(n_neighbors=5).fit(np.array([[0.],[1.],[2.]]), np.array([0.,1.,0.])); \
//   print('both fit ok')"
//   -> both fit ok
//
// ferrolearn `fn fit` (clf knn.rs:292-301, reg knn.rs:696-705) returns
// Err(InsufficientSamples) at FIT time in BOTH — a fit-time guard with NO sklearn
// analog. This assertion (BOTH fits are Ok) FAILS against current ferrolearn,
// pinning the single coherent divergence. Minimally fixable: remove the fit-time
// `n_samples < n_neighbors` guard in both `fit` methods; the `kneighbors_impl`
// `k > n_train` guard (knn.rs:896-902) already fires at predict/kneighbors time
// (green_kneighbors_k_too_large_errors above).
#[test]
fn divergence_fit_does_not_error_when_n_neighbors_gt_n_samples() {
    // X has 3 rows; n_neighbors = 5 > 3.
    let x_3rows = array![[0.0], [1.0], [2.0]];
    let y_clf: Array1<usize> = array![0, 1, 0];
    let y_reg: Array1<f64> = array![0.0, 1.0, 0.0];

    let clf_ok = KNeighborsClassifier::<f64>::new()
        .with_n_neighbors(5)
        .fit(&x_3rows, &y_clf)
        .is_ok();
    let reg_ok = KNeighborsRegressor::<f64>::new()
        .with_n_neighbors(5)
        .fit(&x_3rows, &y_reg)
        .is_ok();

    assert!(
        clf_ok && reg_ok,
        "sklearn KNeighborsClassifier(n_neighbors=5).fit(X_3rows, y) AND \
         KNeighborsRegressor(n_neighbors=5).fit(X_3rows, y) both SUCCEED (no \
         fit-time n_neighbors-vs-n_samples validation; deferred to predict/ \
         kneighbors). ferrolearn fit returns Err(InsufficientSamples) in BOTH — a \
         fit-time guard with no sklearn analog. (clf_ok={clf_ok}, reg_ok={reg_ok}; #874)"
    );
}
