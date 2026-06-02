//! Divergence pins for `ferrolearn-tree/src/adaboost.rs`
//! (`AdaBoostClassifier` — SAMME, discrete) against scikit-learn 1.5.2.
//!
//! Reference: scikit-learn 1.5.2 (commit 156ef14),
//! `sklearn/ensemble/_weight_boosting.py`:
//!   - `_boost_discrete` :660-708 — WEIGHTED base fit
//!     `estimator.fit(X, y, sample_weight=sample_weight)` :664; perfect-fit guard
//!     `if estimator_error <= 0: return sample_weight, 1.0, 0.0` :679-680; SAMME
//!     `estimator_weight = lr*(log((1-err)/err) + log(K-1))` :696-698.
//!   - `BaseWeightBoosting.fit` :111 — stop on `estimator_error == 0` :180.
//!   - `__init__` :501-517 — `n_estimators=50`, `learning_rate=1.0`,
//!     `algorithm='SAMME.R'` :507, `random_state=None`.
//!   - `decision_function` (SAMME) :799-813 — each stump contributes its
//!     `estimator_weight` to the class it predicts.
//!
//! Structural divergence (`.design/tree/adaboost.md`, REQ-6): sklearn fits each
//! round's stump on the WEIGHTED data (`estimator.fit(X, y, sample_weight)`,
//! :664), deterministic, NO bootstrap/RNG. ferrolearn fits an UNWEIGHTED stump on
//! a deterministic systematic RESAMPLE (`resample_weighted` in `fit_samme`).
//! Round 1 has UNIFORM weights, so weighted-fit == resample there; the divergence
//! first becomes observable in round 2, when the weights are non-uniform.
//!
//! `FittedAdaBoostClassifier` exposes no `estimator_weights()` accessor, so the
//! per-round weight is observed through `decision_function`: the SAMME
//! `decision_function` for a single perfect stump places exactly that stump's
//! `estimator_weight` in the predicted-class column (ferrolearn:
//! `out[[i, class_idx]] += estimator_weights[t]`, adaboost.rs `decision_function`
//! SAMME arm). This isolates the *weight magnitude* independent of the separate
//! REQ-5 `decision_function` form divergence (the `-w/(K-1)` / `/sum(w)` / binary
//! collapse), which these pins deliberately do NOT assert.
//!
//! `tests/*.rs` is anti-pattern-gate-exempt: `.unwrap()`/`assert!` are used
//! deliberately (no `panic!`/`unreachable!`).

use ferrolearn_core::introspection::HasFeatureImportances;
use ferrolearn_core::traits::{Fit, Predict};
use ferrolearn_tree::{AdaBoostAlgorithm, AdaBoostClassifier};
use ndarray::{Array1, Array2, array};

/// Binary dataset (8 samples, 2 features; feature 1 is noise). Round-1 depth-1
/// stump splits on `feat0 <= 2.5` → predicts `[0,0,1,1,1,1,1,1]`, misclassifying
/// the two `feat0 in {5,6}` (class 0) samples → round-1 `err = 2/8 = 0.25`,
/// `K = 2`. This is the shared fixture for PIN 1 and GREEN-2.
fn dataset_b() -> (Array2<f64>, Array1<usize>) {
    let x = Array2::from_shape_vec(
        (8, 2),
        vec![
            1.0, 0.0, 2.0, 1.0, 3.0, 0.0, 4.0, 1.0, 5.0, 0.0, 6.0, 1.0, 7.0, 0.0, 8.0, 1.0,
        ],
    )
    .unwrap();
    let y = array![0, 0, 1, 1, 0, 0, 1, 1];
    (x, y)
}

// ===========================================================================
// PIN 1 (RED — headline, REQ-6, blocker #709) — weighted base fit vs resample.
// ===========================================================================

/// PIN 1 (RED) — at `n_estimators = 2` the SECOND stump is fit on NON-uniform
/// round-2 weights. sklearn fits it WEIGHTED on the full data
/// (`_boost_discrete:664`); ferrolearn fits an unweighted stump on a systematic
/// resample (`resample_weighted` in `fit_samme`). The resulting round-2 stumps
/// differ → different ensemble → different end-to-end `predict`.
///
/// Live oracle (sklearn 1.5.2; deterministic — no RNG in the SAMME classifier
/// path):
/// ```text
/// python3 -c "
/// import numpy as np
/// from sklearn.ensemble import AdaBoostClassifier
/// from sklearn.tree import DecisionTreeClassifier
/// X=np.array([[1.,0.],[2.,1.],[3.,0.],[4.,1.],[5.,0.],[6.,1.],[7.,0.],[8.,1.]])
/// y=np.array([0,0,1,1,0,0,1,1])
/// clf=AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1,random_state=0),
///                        algorithm='SAMME',n_estimators=2,random_state=0).fit(X,y)
/// print(clf.predict(X).tolist())     # -> [0, 0, 0, 0, 0, 0, 1, 1]
/// print(clf.estimator_weights_.tolist())  # -> [1.0986122886681098, 1.6094379124341005]
/// "
/// ```
/// Round 1 alone (uniform weights) agrees with ferrolearn — verified separately —
/// so the divergence is exactly the round-2 weighted-fit step. sklearn predicts
/// `[0,0,0,0,0,0,1,1]`; ferrolearn predicts `[0,0,0,0,0,0,0,1]` (differs at
/// index 6). This MUST currently FAIL.
///
/// Tracking: #709
#[test]
fn divergence_weighted_fit_round2_predict() {
    let (x, y) = dataset_b();

    let model = AdaBoostClassifier::<f64>::new()
        .with_n_estimators(2)
        .with_algorithm(AdaBoostAlgorithm::Samme)
        .with_random_state(0);
    let fitted = model.fit(&x, &y).unwrap();
    let preds = fitted.predict(&x).unwrap();

    // sklearn 1.5.2 live oracle (see docstring); NOT copied from ferrolearn.
    let sklearn_predict: Vec<usize> = vec![0, 0, 0, 0, 0, 0, 1, 1];

    assert_eq!(
        preds.to_vec(),
        sklearn_predict,
        "n_estimators=2 SAMME predict diverges from sklearn: the round-2 stump must \
         be fit WEIGHTED on the full data (estimator.fit(X,y,sample_weight), \
         _weight_boosting.py:664), not on a systematic resample (resample_weighted \
         in fit_samme). sklearn predicts {sklearn_predict:?}; ferrolearn predicts \
         {:?} (differs at index 6).",
        preds.to_vec()
    );
}

// ===========================================================================
// PIN 2 (RED — REQ-7, blocker #710) — perfect-fit estimator_weight == 1.0 guard.
// ===========================================================================

/// PIN 2 (RED) — a binary dataset separable by a single feature threshold: a
/// depth-1 stump achieves `err == 0`. sklearn's `_boost_discrete:679-680` returns
/// `(sample_weight, 1.0, 0.0)` the instant `estimator_error <= 0`, so the single
/// estimator's weight is exactly `1.0` (and `fit:180` then stops boosting).
/// ferrolearn's `fit_samme` has NO such guard: `err == 0` flows into the alpha
/// formula as `lr*(ln((1-0).max(1e-10)/(0).max(1e-10)) + ln(K-1)) = ln(1/1e-10)
/// = 23.0258...` for `lr=1, K=2`.
///
/// `estimator_weights()` is not exposed, so the weight is read off
/// `decision_function`: with a single perfect stump every sample's predicted
/// class is correct, and ferrolearn's SAMME `decision_function` places exactly
/// `estimator_weights[0]` in that class's column
/// (`out[[i, class_idx]] += estimator_weights[t]`). The max-magnitude entry of
/// each row therefore EQUALS the single estimator weight — which sklearn pins to
/// `1.0`.
///
/// Live oracle (sklearn 1.5.2):
/// ```text
/// python3 -c "
/// import numpy as np
/// from sklearn.ensemble import AdaBoostClassifier
/// from sklearn.tree import DecisionTreeClassifier
/// X=np.array([[1.],[2.],[3.],[4.],[5.],[6.]]); y=np.array([0,0,0,1,1,1])
/// clf=AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1,random_state=0),
///                        algorithm='SAMME',n_estimators=1,random_state=0).fit(X,y)
/// print(clf.estimator_weights_.tolist())  # -> [1.0]
/// print(len(clf.estimators_))             # -> 1
/// "
/// ```
/// sklearn `estimator_weight == 1.0`; ferrolearn `~23.0258`. MUST currently FAIL.
///
/// Tracking: #710
#[test]
fn divergence_perfect_fit_estimator_weight_one() {
    // feat0 <= 3.5 perfectly separates class 0 from class 1.
    let x = Array2::from_shape_vec((6, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let y = array![0, 0, 0, 1, 1, 1];

    let model = AdaBoostClassifier::<f64>::new()
        .with_n_estimators(1)
        .with_algorithm(AdaBoostAlgorithm::Samme)
        .with_random_state(0);
    let fitted = model.fit(&x, &y).unwrap();

    // ferrolearn SAMME decision_function: predicted-class column == estimator
    // weight of the single (perfect) stump. Read its magnitude.
    let df = fitted.decision_function(&x).unwrap();
    let weight = (0..df.nrows())
        .map(|i| {
            (0..df.ncols())
                .map(|k| df[[i, k]].abs())
                .fold(0.0, f64::max)
        })
        .fold(0.0, f64::max);

    // sklearn 1.5.2 live oracle: estimator_weights_ == [1.0] under the perfect-fit
    // guard (_boost_discrete:679-680). NOT copied from ferrolearn.
    let sklearn_perfect_fit_weight = 1.0;

    assert!(
        (weight - sklearn_perfect_fit_weight).abs() < 1e-9,
        "perfect-fit (err==0) single-stump weight = {weight} != sklearn 1.0 \
         (perfect-fit guard `if estimator_error <= 0: return sample_weight, 1.0, \
         0.0`, _weight_boosting.py:679-680). ferrolearn lacks the guard and emits \
         alpha = ln(1/1e-10) ~= 23.0258 instead."
    );
}

// ===========================================================================
// GREEN pins (deterministic, oracle-grounded — must pass now).
// ===========================================================================

/// GREEN-1 — constructor numeric defaults match `AdaBoostClassifier().get_params()`
/// (sklearn 1.5.2) for the params ferrolearn exposes.
///
/// Live oracle: `AdaBoostClassifier().get_params()` ->
///   `{'algorithm': 'SAMME.R', 'estimator': None, 'learning_rate': 1.0,
///     'n_estimators': 50, 'random_state': None}` (`__init__`, :501-517).
///
/// NOTE on `algorithm`: ferrolearn `Default` is `Samme`, NOT sklearn 1.5.2's
/// `'SAMME.R'` — a DELIBERATE R-DEV-6 deviation (sklearn deprecated `'SAMME.R'`
/// in 1.4 with a `FutureWarning`, `_validate_estimator:526-534`, and removed it in
/// 1.6, making `SAMME` the surviving default). This pin asserts the deviation
/// (`Samme`) and the matching numeric defaults; it deliberately does NOT pin a
/// `SAMME.R`-default match as failing.
#[test]
fn defaults_match_sklearn_get_params() {
    let m = AdaBoostClassifier::<f64>::default();
    assert_eq!(m.n_estimators, 50, "sklearn default n_estimators=50 (:505)");
    assert!(
        (m.learning_rate - 1.0).abs() < 1e-15,
        "sklearn default learning_rate=1.0 (:506)"
    );
    assert!(
        m.random_state.is_none(),
        "sklearn default random_state=None (:508)"
    );
    // DELIBERATE R-DEV-6 deviation: ferrolearn ships sklearn's surviving (1.6)
    // default `SAMME`, not 1.5.2's deprecated `'SAMME.R'` (:507).
    assert_eq!(
        m.algorithm,
        AdaBoostAlgorithm::Samme,
        "ferrolearn default algorithm = Samme (R-DEV-6: sklearn deprecated SAMME.R \
         in 1.4, removed in 1.6, _validate_estimator:526-534)"
    );
}

/// GREEN-2 — SAMME `estimator_weight` formula on a controlled first round.
///
/// With `n_estimators = 1` the single estimator weight IS the round-1 alpha, and
/// ferrolearn's SAMME `decision_function` places it in the predicted-class column.
/// On `dataset_b`, the round-1 depth-1 stump (uniform weights → weighted ==
/// unweighted) splits `feat0 <= 2.5`, predicting `[0,0,1,1,1,1,1,1]` and
/// misclassifying the two class-0 samples at `feat0 in {5,6}` →
/// `err = 2/8 = 0.25`, `K = 2`.
///
/// Expected (sklearn `_boost_discrete:696-698`, NOT copied from ferrolearn):
///   `alpha = lr*(log((1-err)/err) + log(K-1)) = log(0.75/0.25) + log(1) = log(3)`.
/// Live cross-check:
/// ```text
/// python3 -c "import numpy as np; err=0.25; K=2; print(np.log((1-err)/err)+np.log(K-1))"
/// # -> 1.0986122886681098
/// ```
#[test]
fn samme_estimator_weight_formula_first_round() {
    let (x, y) = dataset_b();

    let model = AdaBoostClassifier::<f64>::new()
        .with_n_estimators(1)
        .with_algorithm(AdaBoostAlgorithm::Samme)
        .with_random_state(0);
    let fitted = model.fit(&x, &y).unwrap();

    let df = fitted.decision_function(&x).unwrap();
    let weight = (0..df.nrows())
        .map(|i| {
            (0..df.ncols())
                .map(|k| df[[i, k]].abs())
                .fold(0.0, f64::max)
        })
        .fold(0.0, f64::max);

    // sklearn closed form `:696-698` with err=0.25 (2/8 misclassified), K=2.
    let err = 0.25_f64;
    let k = 2.0_f64;
    let sklearn_alpha = ((1.0 - err) / err).ln() + (k - 1.0).ln(); // = ln(3)

    assert!(
        (weight - sklearn_alpha).abs() < 1e-12,
        "round-1 SAMME estimator_weight = {weight} != sklearn closed form \
         {sklearn_alpha} = log((1-0.25)/0.25)+log(1) (_weight_boosting.py:696-698)"
    );
}

/// GREEN-3 — deterministic reproducibility (AC-10): ferrolearn's SAMME `fit` uses
/// NO RNG, so two fits with identical params produce identical `predict`,
/// `feature_importances`, and `decision_function`.
#[test]
fn fit_is_deterministic_and_reproducible() {
    let (x, y) = dataset_b();

    let model = AdaBoostClassifier::<f64>::new()
        .with_n_estimators(10)
        .with_algorithm(AdaBoostAlgorithm::Samme)
        .with_random_state(7);

    let f1 = model.fit(&x, &y).unwrap();
    let f2 = model.fit(&x, &y).unwrap();

    assert_eq!(
        f1.predict(&x).unwrap(),
        f2.predict(&x).unwrap(),
        "predict must be identical across fits (no RNG)"
    );
    assert_eq!(
        f1.feature_importances(),
        f2.feature_importances(),
        "feature_importances must be identical across fits (no RNG)"
    );
    assert_eq!(
        f1.decision_function(&x).unwrap(),
        f2.decision_function(&x).unwrap(),
        "decision_function must be identical across fits (no RNG)"
    );
}
