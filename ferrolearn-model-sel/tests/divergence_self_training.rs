//! Divergence audit: `ferrolearn_model_sel::self_training::SelfTrainingClassifier`
//! vs scikit-learn 1.5.2 `SelfTrainingClassifier`
//! (`sklearn/semi_supervised/_self_training.py:39`).
//!
//! Two classes of test live here:
//!
//! 1. GREEN GUARDS for the narrow SHIPPED surface (REQ-SELFTRAIN-LOOP, the binary
//!    `criterion='threshold'` self-training ITERATION SHAPE). These must PASS; if
//!    they ever fail the shipped mechanic has regressed.
//!
//! 2. FAILING `#[ignore]`d PINS for the deterministic-fixable divergences
//!    (#1841 strict `>` vs `>=`, #1842 threshold interval `[0,1)` vs `(0,1]`) and
//!    one unclaimed all-labeled `n_iter` divergence (noted under #1846). Each
//!    asserts the LIVE sklearn 1.5.2 oracle value (R-CHAR-3 — expected values come
//!    from `cd /tmp && python3 -c "import sklearn; ..."`, never copied from the
//!    ferrolearn side). They are `#[ignore]`d so the suite stays green; remove the
//!    ignore to watch them FAIL against current ferrolearn.

use ferrolearn_core::FerroError;
use ferrolearn_model_sel::calibration::{FitFn, PredictFn};
use ferrolearn_model_sel::self_training::{SelfTrainingClassifier, UNLABELED};
use ndarray::{Array1, Array2};

// ---------------------------------------------------------------------------
// Shared base estimators (FitFn closures — the sanctioned R-DEV-7 idiom).
// ---------------------------------------------------------------------------

/// Base whose predicted score equals feature column 0 (interpreted as
/// `P(class 1)`). Separable when feature ≈ class.
fn feature0_fit_fn() -> FitFn {
    Box::new(|_x: &Array2<f64>, _y: &Array1<usize>| {
        Ok(Box::new(|x: &Array2<f64>| Ok(x.column(0).to_owned())) as PredictFn)
    })
}

/// Base that returns a CONSTANT score for every row.
fn constant_fit_fn(value: f64) -> FitFn {
    Box::new(move |_x: &Array2<f64>, _y: &Array1<usize>| {
        let v = value;
        Ok(Box::new(move |x: &Array2<f64>| Ok(Array1::from_elem(x.nrows(), v))) as PredictFn)
    })
}

// ---------------------------------------------------------------------------
// GREEN GUARD — REQ-SELFTRAIN-LOOP (SHIPPED binary threshold-criterion SHAPE).
// ---------------------------------------------------------------------------

/// Green guard for the SHIPPED self-training iteration SHAPE
/// (`self_training.rs` `fit`, mirroring sklearn's refit->predict->pseudo-label
/// ->iterate loop, `sklearn/semi_supervised/_self_training.py:247-282`).
///
/// LIVE ORACLE (sklearn 1.5.2, `cd /tmp`, R-CHAR-3): a separable column-0 base
/// `ColZero` with `threshold=0.7` on
/// `X=[[0],[0.1],[0.9],[1.0],[0.05],[0.95]], y=[0,0,1,1,-1,-1]` produces
/// `transduction_ == [0, 0, 1, 1, 0, 1]` — the two confident unlabeled rows
/// (idx 4 score 0.05 -> class 0, idx 5 score 0.95 -> class 1) are pseudo-labeled.
///
/// This asserts the SHAPE only: confident unlabeled rows get filled with the
/// score-derived binary label, non-confident rows keep the sentinel. It does NOT
/// claim end-to-end transduction equality (ferrolearn is binary/score-based).
#[test]
fn guard_selftrain_loop_pseudo_labels_confident_rows() {
    let x = Array2::from_shape_vec((6, 1), vec![0.0, 0.1, 0.9, 1.0, 0.05, 0.95]).unwrap();
    let y = Array1::from_vec(vec![0, 0, 1, 1, UNLABELED, UNLABELED]);

    let st = SelfTrainingClassifier::new(feature0_fit_fn()).threshold(0.7);
    let fitted = st.fit(&x, &y).unwrap();
    let labels = fitted.transduced_labels();

    // Original labeled rows are untouched.
    assert_eq!(labels[0], 0);
    assert_eq!(labels[1], 0);
    assert_eq!(labels[2], 1);
    assert_eq!(labels[3], 1);
    // Confident unlabeled rows pseudo-labeled to the oracle classes (0, 1).
    assert_eq!(
        labels[4], 0,
        "score 0.05 -> class 0 (oracle transduction_[4]=0)"
    );
    assert_eq!(
        labels[5], 1,
        "score 0.95 -> class 1 (oracle transduction_[5]=1)"
    );
}

/// Green guard: a NON-confident unlabeled row (max_prob below threshold) retains
/// the `UNLABELED` sentinel — the "sentinel retained for non-confident" half of
/// the SHAPE. With a constant score 0.5 and `threshold=0.75`,
/// `max(0.5, 0.5) = 0.5 < 0.75` so nothing is selected. sklearn's analogue
/// (`max_proba > threshold`, `:262`) likewise selects nothing
/// (`termination_='no_change'`). The two agree here because 0.5 is strictly
/// below 0.75 — the `>=`/`>` boundary (#1841) is not exercised.
#[test]
fn guard_selftrain_loop_non_confident_keeps_sentinel() {
    let x = Array2::from_elem((4, 1), 0.5);
    let y = Array1::from_vec(vec![0, 1, UNLABELED, UNLABELED]);

    let st = SelfTrainingClassifier::new(constant_fit_fn(0.5))
        .threshold(0.75)
        .max_iter(5);
    let fitted = st.fit(&x, &y).unwrap();
    let labels = fitted.transduced_labels();

    assert_eq!(
        labels[2], UNLABELED,
        "max_prob 0.5 < 0.75 -> stays unlabeled"
    );
    assert_eq!(
        labels[3], UNLABELED,
        "max_prob 0.5 < 0.75 -> stays unlabeled"
    );
}

// ---------------------------------------------------------------------------
// PIN — #1841 REQ-THRESHOLD-STRICT: selection `>` (sklearn) vs `>=` (ferrolearn).
// ---------------------------------------------------------------------------

/// Divergence (#1841): ferrolearn selects with `if max_prob >= self.threshold`
/// (`ferrolearn-model-sel/src/self_training.rs:200`) where sklearn uses
/// `selected = max_proba > self.threshold`
/// (`sklearn/semi_supervised/_self_training.py:262`, STRICT `>`).
///
/// Input: a base returning a CONSTANT score == threshold (0.75) for the unlabeled
/// row, so `prob = 0.75`, `max_prob = max(0.75, 1-0.75) = 0.75 == threshold`.
///
/// LIVE ORACLE (sklearn 1.5.2, `cd /tmp`, R-CHAR-3): the analogous
/// `SelfTrainingClassifier(ExactProba(), threshold=0.75)` on `y=[0,0,-1]` where
/// the unlabeled row gets `max_proba == 0.75` yields
/// `transduction_ == [0, 0, -1]`, `termination_condition_ == 'no_change'` — the
/// `==threshold` row is NOT selected (strict `>`), it KEEPS the -1 sentinel.
///
/// EXPECTED (sklearn): the unlabeled row stays `UNLABELED`.
/// ACTUAL (ferrolearn `>=`): it is pseudo-labeled (`max_prob >= threshold` true).
/// This test asserts the sklearn behavior and therefore FAILS un-ignored.
/// Tracking: #1841. Single-spot fixable (`>=` -> `>`).
#[test]
fn divergence_threshold_strict_boundary_1841() {
    // Two labeled rows (forces >=1 labeled), one unlabeled row.
    let x = Array2::from_shape_vec((3, 1), vec![0.0, 1.0, 5.0]).unwrap();
    let y = Array1::from_vec(vec![0, 0, UNLABELED]);

    // Constant score 0.75 == threshold for the unlabeled row -> max_prob == 0.75.
    let st = SelfTrainingClassifier::new(constant_fit_fn(0.75)).threshold(0.75);
    let fitted = st.fit(&x, &y).unwrap();

    // sklearn oracle: transduction_[2] stays -1 (NOT selected, strict >).
    assert_eq!(
        fitted.transduced_labels()[2],
        UNLABELED,
        "max_prob == threshold (0.75) must NOT be selected (sklearn strict `>`, \
         _self_training.py:262); oracle transduction_==[0,0,-1] termination='no_change'"
    );
}

// ---------------------------------------------------------------------------
// PIN — #1842 REQ-THRESHOLD-BOUND: interval [0,1) (sklearn) vs (0,1] (ferrolearn).
// ---------------------------------------------------------------------------

/// Divergence (#1842) facet (a): ferrolearn ACCEPTS `threshold=1.0`
/// (`self_training.rs:126` rejects only `threshold > 1.0`) but sklearn's
/// constraint `Interval(Real, 0.0, 1.0, closed="left")`
/// (`sklearn/semi_supervised/_self_training.py:164`) is `[0, 1)` — 1.0 is OUT.
///
/// LIVE ORACLE (sklearn 1.5.2, `cd /tmp`, R-CHAR-3):
/// `SelfTrainingClassifier(LogisticRegression(), threshold=1.0).fit(X,y)` raises
/// `InvalidParameterError`. EXPECTED: ferrolearn `fit` returns `Err`.
/// ACTUAL: ferrolearn returns `Ok` (1.0 is inside its `(0, 1]`). FAILS un-ignored.
/// Tracking: #1842.
#[test]
#[ignore = "divergence: ferrolearn accepts threshold=1.0 but sklearn rejects ([0,1)); tracking #1842"]
fn divergence_threshold_bound_one_rejected_1842() {
    let x = Array2::from_shape_vec((4, 1), vec![0.0, 1.0, 2.0, 3.0]).unwrap();
    let y = Array1::from_vec(vec![0, 1, UNLABELED, UNLABELED]);

    let st = SelfTrainingClassifier::new(feature0_fit_fn()).threshold(1.0);
    let result = st.fit(&x, &y);

    // sklearn oracle: threshold=1.0 -> InvalidParameterError.
    assert!(
        matches!(result, Err(FerroError::InvalidParameter { .. })),
        "threshold=1.0 must be REJECTED (sklearn Interval closed=left [0,1), \
         _self_training.py:164); oracle: threshold=1.0 -> InvalidParameterError"
    );
}

/// Divergence (#1842) facet (b): ferrolearn REJECTS `threshold=0.0`
/// (`self_training.rs:126` rejects `threshold <= 0.0`) but sklearn's `[0, 1)`
/// constraint (`_self_training.py:164`) ACCEPTS 0.0 (closed on the left).
///
/// LIVE ORACLE (sklearn 1.5.2, `cd /tmp`, R-CHAR-3):
/// `SelfTrainingClassifier(LogisticRegression(), threshold=0.0).fit(X,y)` ->
/// ACCEPTED. EXPECTED: ferrolearn `fit` returns `Ok`.
/// ACTUAL: ferrolearn returns `Err` ("must be in (0, 1]"). FAILS un-ignored.
/// Tracking: #1842.
#[test]
#[ignore = "divergence: ferrolearn rejects threshold=0.0 but sklearn accepts ([0,1)); tracking #1842"]
fn divergence_threshold_bound_zero_accepted_1842() {
    let x = Array2::from_shape_vec((4, 1), vec![0.0, 1.0, 2.0, 3.0]).unwrap();
    let y = Array1::from_vec(vec![0, 1, UNLABELED, UNLABELED]);

    let st = SelfTrainingClassifier::new(feature0_fit_fn()).threshold(0.0);
    let result = st.fit(&x, &y);

    // sklearn oracle: threshold=0.0 -> ACCEPTED.
    assert!(
        result.is_ok(),
        "threshold=0.0 must be ACCEPTED (sklearn Interval closed=left [0,1), \
         _self_training.py:164); oracle: threshold=0.0 -> ACCEPTED, got {:?}",
        result.err()
    );
}

// ---------------------------------------------------------------------------
// PIN — unclaimed all-labeled `n_iter` divergence (noted under #1846 REQ-ATTRS).
// ---------------------------------------------------------------------------

/// Divergence (part of #1846 REQ-ATTRS): all-labeled-initially input.
/// ferrolearn increments `n_iter` at the TOP of the `for`-loop
/// (`self_training.rs:158`), so the single iteration that hits the
/// empty-unlabeled break leaves `n_iter == 1`. sklearn evaluates
/// `while not np.all(has_label)` BEFORE the body and `n_iter_ += 1` is INSIDE
/// the loop (`_self_training.py:245-250`), so all-labeled input never enters the
/// loop and `n_iter_` stays 0 (with `termination_condition_ == 'all_labeled'`).
///
/// LIVE ORACLE (sklearn 1.5.2, `cd /tmp`, R-CHAR-3): a fully-labeled
/// `SelfTrainingClassifier(ColZero()).fit(X, y)` (no -1 rows) yields
/// `n_iter_ == 0`, `termination_condition_ == 'all_labeled'`.
///
/// EXPECTED (sklearn): `n_iter() == 0`. ACTUAL (ferrolearn): `1`. FAILS un-ignored.
/// (ferrolearn's own `test_self_training_all_labeled` asserts the divergent `1`.)
/// Tracking: #1846.
#[test]
#[ignore = "divergence: ferrolearn n_iter==1 for all-labeled input vs sklearn n_iter_==0; tracking #1846"]
fn divergence_all_labeled_n_iter_1846() {
    let x = Array2::from_shape_vec((4, 1), vec![0.0, 0.1, 0.9, 1.0]).unwrap();
    let y = Array1::from_vec(vec![0, 0, 1, 1]); // fully labeled, no UNLABELED

    let st = SelfTrainingClassifier::new(feature0_fit_fn());
    let fitted = st.fit(&x, &y).unwrap();

    // sklearn oracle: all-labeled -> n_iter_ == 0 (loop never entered).
    assert_eq!(
        fitted.n_iter(),
        0,
        "all-labeled input must give n_iter_==0 (sklearn while-guard before body, \
         _self_training.py:247-250); oracle n_iter_==0 termination='all_labeled'"
    );
}
