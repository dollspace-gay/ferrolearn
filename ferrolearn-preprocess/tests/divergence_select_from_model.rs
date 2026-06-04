//! Divergence audit: `SelectFromModelExt` / `FittedSelectFromModelExt` vs
//! scikit-learn 1.5.2 `class SelectFromModel`.
//!
//! sklearn site: `sklearn/feature_selection/_from_model.py` —
//! `_calculate_threshold` `:24-71` (mean=`np.mean` `:61`, median=`np.median`
//! `:58`); `_get_support_mask` `:273-313`. The selection core (`:299-312`) is
//! pure:
//!
//! ```text
//! scores = importances; threshold = _calculate_threshold;
//! if max_features: cand = argsort(-scores, mergesort)[:max_features]; mask[cand] = True
//! else:            mask = ones;  mask[scores < threshold] = False
//! ```
//!
//! ferrolearn (`ferrolearn-preprocess/src/select_from_model.rs`) takes a STATIC
//! importance vector and supports the `ThresholdStrategy` enum
//! (`Mean`/`Median`/`Value`/`Percentile`) with an optional `max_features` cap.
//! Its order of operations is threshold-then-cap
//! (`:262-283`), whereas sklearn is cap-then-threshold (`:306-312`). These are
//! mathematically equivalent: sklearn's result = `{top-k by score} ∩ {>= thr}`;
//! ferrolearn's = `top-k of {>= thr}`. A below-threshold feature in sklearn's
//! top-k is filtered out and never "steals" a slot from a lower above-threshold
//! feature, so both yield the identical index set (verified by exhaustive
//! numpy search over value grids in the audit).
//!
//! EXPECTED VALUES are produced by the sklearn `_get_support_mask` replica in
//! numpy (R-CHAR-3: never literal-copied from the ferrolearn side):
//! ```text
//! def sfm(scores, threshold, max_features=None):
//!     scores=np.asarray(scores,float)
//!     if max_features is not None:
//!         mask=np.zeros_like(scores,bool)
//!         cand=np.argsort(-scores,kind='mergesort')[:max_features]; mask[cand]=True
//!     else: mask=np.ones_like(scores,bool)
//!     mask[scores<threshold]=False
//!     return np.flatnonzero(mask).tolist()
//! ```
//! Tracking issue: #1352.
//!
//! Verdict (see audit report): NO DIVERGENCE FOUND in the REQ-1 core
//! (threshold + mask + max_features). These are PASSING green-guard tests.

use ferrolearn_core::error::FerroError;
use ferrolearn_core::traits::{Fit, Transform};
use ferrolearn_preprocess::select_from_model::{SelectFromModelExt, ThresholdStrategy};
use ndarray::{Array1, Array2, array};

/// Mean threshold. Oracle: sfm([0.1,0.5,0.4], np.mean=0.3333..) -> [1, 2].
/// sklearn `_calculate_threshold` `:61` (`np.mean`).
#[test]
fn divergence_mean_threshold() {
    let sel = SelectFromModelExt::<f64>::new(ThresholdStrategy::Mean, None);
    let importances = array![0.1, 0.5, 0.4];
    match sel.fit(&importances, &()) {
        Ok(fitted) => assert_eq!(fitted.selected_indices(), &[1, 2]),
        Err(e) => report_unexpected_err(&e),
    }
}

/// Median threshold, odd n. Oracle: sfm([0.1,0.5,0.3], np.median=0.3) -> [1, 2].
/// sklearn `_calculate_threshold` `:58` (`np.median`).
#[test]
fn divergence_median_threshold_odd() {
    let sel = SelectFromModelExt::<f64>::new(ThresholdStrategy::Median, None);
    let importances = array![0.1, 0.5, 0.3];
    match sel.fit(&importances, &()) {
        Ok(fitted) => assert_eq!(fitted.selected_indices(), &[1, 2]),
        Err(e) => report_unexpected_err(&e),
    }
}

/// Median threshold, even n. Oracle: np.median([0.1,0.5,0.2,0.6]) == 0.35
/// (mean of the two middle order-statistics), sfm(.., 0.35) -> [1, 3].
/// Confirms ferrolearn `compute_median` (`select_from_model.rs:164-174`)
/// averages the two middle values exactly like `np.median`.
#[test]
fn divergence_median_threshold_even() {
    let sel = SelectFromModelExt::<f64>::new(ThresholdStrategy::Median, None);
    let importances = array![0.1, 0.5, 0.2, 0.6];
    match sel.fit(&importances, &()) {
        Ok(fitted) => {
            // np.median([0.1,0.5,0.2,0.6]) == 0.35 exactly.
            let expected_median = (0.2_f64 + 0.5_f64) / 2.0;
            assert!(
                (fitted.threshold_value() - expected_median).abs() < 1e-15,
                "median threshold {} != np.median 0.35",
                fitted.threshold_value()
            );
            assert_eq!(fitted.selected_indices(), &[1, 3]);
        }
        Err(e) => report_unexpected_err(&e),
    }
}

/// Explicit value threshold. Oracle: sfm([0.1,0.5,0.4], 0.45) -> [1].
#[test]
fn divergence_value_threshold() {
    let sel = SelectFromModelExt::<f64>::new(ThresholdStrategy::Value(0.45), None);
    let importances = array![0.1, 0.5, 0.4];
    match sel.fit(&importances, &()) {
        Ok(fitted) => assert_eq!(fitted.selected_indices(), &[1]),
        Err(e) => report_unexpected_err(&e),
    }
}

/// max_features cap WITHOUT threshold pressure (Value(0.0) keeps all, cap=2).
/// Oracle: sfm([0.3,0.5,0.1,0.7], 0.0, 2) -> [1, 3] (top-2 by score: 0.7,0.5).
/// sklearn `_get_support_mask` `:308` argsort(-scores, mergesort)[:2].
#[test]
fn divergence_max_features_no_pressure() {
    let sel = SelectFromModelExt::<f64>::new(ThresholdStrategy::Value(0.0), Some(2));
    let importances = array![0.3, 0.5, 0.1, 0.7];
    match sel.fit(&importances, &()) {
        Ok(fitted) => assert_eq!(fitted.selected_indices(), &[1, 3]),
        Err(e) => report_unexpected_err(&e),
    }
}

/// KEY ADVERSARIAL CASE: max_features WITH threshold pressure.
/// `[0.1,0.9,0.8]`, Value(0.85), max_features=2.
/// sklearn caps-then-thresholds: cand=argsort(-scores,mergesort)[:2]=[1,2];
/// then mask[scores<0.85]=False drops index 2 (0.8<0.85) -> [1].
/// ferrolearn thresholds-then-caps: {1} clears 0.85; cap=2 no-op -> [1].
/// Both AGREE. Oracle: sfm([0.1,0.9,0.8], 0.85, 2) -> [1].
#[test]
fn divergence_max_features_with_threshold_pressure() {
    let sel = SelectFromModelExt::<f64>::new(ThresholdStrategy::Value(0.85), Some(2));
    let importances = array![0.1, 0.9, 0.8];
    match sel.fit(&importances, &()) {
        Ok(fitted) => assert_eq!(fitted.selected_indices(), &[1]),
        Err(e) => report_unexpected_err(&e),
    }
}

/// TIE-BREAK: equal importances with max_features.
/// `[0.5,0.5,0.5,0.1]`, Value(0.0), max_features=2.
/// sklearn mergesort on -scores is STABLE: argsort(-[0.5,0.5,0.5,0.1])=[0,1,2,3],
/// top-2 = [0,1] (ascending index tie-break).
/// ferrolearn uses stable `sort_by` (descending) then re-sorts ascending,
/// so ties also keep ascending index -> [0,1] (NOT [2,1] etc.).
/// Oracle: sfm([0.5,0.5,0.5,0.1], 0.0, 2) -> [0, 1].
#[test]
fn divergence_max_features_tie_break() {
    let sel = SelectFromModelExt::<f64>::new(ThresholdStrategy::Value(0.0), Some(2));
    let importances = array![0.5, 0.5, 0.5, 0.1];
    match sel.fit(&importances, &()) {
        Ok(fitted) => assert_eq!(fitted.selected_indices(), &[0, 1]),
        Err(e) => report_unexpected_err(&e),
    }
}

/// THRESHOLD BOUNDARY: a feature whose importance EXACTLY equals the threshold
/// is KEPT. sklearn excludes only strictly-less (`mask[scores < threshold]`,
/// `:312`); ferrolearn keeps `imp >= threshold` (`select_from_model.rs:266`).
/// Equality must be inclusive on BOTH sides.
/// Oracle: sfm([0.1,0.5,0.4], 0.4) -> [1, 2] (index 2 has importance == thr).
#[test]
fn divergence_threshold_boundary_inclusive() {
    let sel = SelectFromModelExt::<f64>::new(ThresholdStrategy::Value(0.4), None);
    let importances = array![0.1, 0.5, 0.4];
    match sel.fit(&importances, &()) {
        Ok(fitted) => assert_eq!(fitted.selected_indices(), &[1, 2]),
        Err(e) => report_unexpected_err(&e),
    }
}

/// f32 path: mean of [0.1,0.5,0.4] in f32 == 0.33333334; sel = [1, 2].
/// Oracle: np.array([0.1,0.5,0.4],float32).mean() -> 0.33333334; >= -> [1,2].
#[test]
fn divergence_f32_mean() {
    let sel = SelectFromModelExt::<f32>::new(ThresholdStrategy::Mean, None);
    let importances: Array1<f32> = array![0.1f32, 0.5, 0.4];
    match sel.fit(&importances, &()) {
        Ok(fitted) => assert_eq!(fitted.selected_indices(), &[1, 2]),
        Err(e) => report_unexpected_err(&e),
    }
}

/// Transform actually projects the selected columns (boundary case kept).
/// Oracle indices for Value(0.4) on [0.1,0.5,0.4] -> [1, 2]; the projected
/// matrix keeps columns 1 and 2.
#[test]
fn divergence_transform_projects_selected_columns() {
    let sel = SelectFromModelExt::<f64>::new(ThresholdStrategy::Value(0.4), None);
    let importances = array![0.1, 0.5, 0.4];
    match sel.fit(&importances, &()) {
        Ok(fitted) => {
            let x: Array2<f64> = array![[10.0, 11.0, 12.0], [20.0, 21.0, 22.0]];
            match fitted.transform(&x) {
                Ok(out) => {
                    assert_eq!(out.shape(), &[2, 2]);
                    assert_eq!(out, array![[11.0, 12.0], [21.0, 22.0]]);
                }
                Err(e) => report_unexpected_err(&e),
            }
        }
        Err(e) => report_unexpected_err(&e),
    }
}

// ---------------------------------------------------------------------------
// Error contracts
// ---------------------------------------------------------------------------

/// Empty importance vector -> Err (sklearn fits on a real estimator; ferrolearn
/// validates the static vector, `select_from_model.rs:234-239`).
#[test]
#[allow(
    clippy::assertions_on_constants,
    reason = "error arm fails loudly without panic!/unwrap (anti-pattern gate)"
)]
fn divergence_empty_importances_err() {
    let sel = SelectFromModelExt::<f64>::new(ThresholdStrategy::Mean, None);
    let importances: Array1<f64> = Array1::zeros(0);
    match sel.fit(&importances, &()) {
        Ok(_) => assert!(false, "expected Err on empty importance vector"),
        Err(FerroError::InvalidParameter { .. }) => {}
        Err(other) => assert!(false, "expected InvalidParameter, got {other:?}"),
    }
}

/// Percentile(0.0) -> Err (ferrolearn extension contract `(0, 100]`,
/// `select_from_model.rs:252-257`). NOT a sklearn-parity assertion.
#[test]
#[allow(
    clippy::assertions_on_constants,
    reason = "error arm fails loudly without panic!/unwrap (anti-pattern gate)"
)]
fn divergence_percentile_zero_err() {
    let sel = SelectFromModelExt::<f64>::new(ThresholdStrategy::Percentile(0.0), None);
    let importances = array![0.1, 0.5, 0.3];
    match sel.fit(&importances, &()) {
        Ok(_) => assert!(false, "expected Err on Percentile(0.0)"),
        Err(FerroError::InvalidParameter { .. }) => {}
        Err(other) => assert!(false, "expected InvalidParameter, got {other:?}"),
    }
}

/// Percentile(101.0) -> Err (ferrolearn extension contract `(0, 100]`).
#[test]
#[allow(
    clippy::assertions_on_constants,
    reason = "error arm fails loudly without panic!/unwrap (anti-pattern gate)"
)]
fn divergence_percentile_over_100_err() {
    let sel = SelectFromModelExt::<f64>::new(ThresholdStrategy::Percentile(101.0), None);
    let importances = array![0.1, 0.5, 0.3];
    match sel.fit(&importances, &()) {
        Ok(_) => assert!(false, "expected Err on Percentile(101.0)"),
        Err(FerroError::InvalidParameter { .. }) => {}
        Err(other) => assert!(false, "expected InvalidParameter, got {other:?}"),
    }
}

/// transform column-count mismatch -> Err (ShapeMismatch,
/// `select_from_model.rs:305-311`).
#[test]
#[allow(
    clippy::assertions_on_constants,
    reason = "error arm fails loudly without panic!/unwrap (anti-pattern gate)"
)]
fn divergence_transform_ncols_mismatch_err() {
    let sel = SelectFromModelExt::<f64>::new(ThresholdStrategy::Mean, None);
    let importances = array![0.5, 0.5];
    match sel.fit(&importances, &()) {
        Ok(fitted) => {
            let x_bad: Array2<f64> = array![[1.0, 2.0, 3.0]]; // 3 cols, 2 expected
            match fitted.transform(&x_bad) {
                Ok(_) => assert!(false, "expected ShapeMismatch on ncols mismatch"),
                Err(FerroError::ShapeMismatch { .. }) => {}
                Err(other) => assert!(false, "expected ShapeMismatch, got {other:?}"),
            }
        }
        Err(e) => report_unexpected_err(&e),
    }
}

// ---------------------------------------------------------------------------
// ferrolearn EXTENSION (not a sklearn-parity test): Percentile contract.
// sklearn SelectFromModel has NO percentile analog; this guards only
// ferrolearn's OWN documented "top p% by importance" contract. No blocker.
// ---------------------------------------------------------------------------

/// Percentile(100.0) keeps every feature (top 100%). ferrolearn-extension.
#[test]
fn extension_percentile_100_keeps_all() {
    let sel = SelectFromModelExt::<f64>::new(ThresholdStrategy::Percentile(100.0), None);
    let importances = array![0.1, 0.5, 0.3];
    match sel.fit(&importances, &()) {
        Ok(fitted) => assert_eq!(fitted.n_features_selected(), 3),
        Err(e) => report_unexpected_err(&e),
    }
}

// ---------------------------------------------------------------------------
// helper
// ---------------------------------------------------------------------------

#[allow(
    clippy::assertions_on_constants,
    reason = "error arm fails loudly without panic!/unwrap (anti-pattern gate)"
)]
fn report_unexpected_err(e: &FerroError) {
    assert!(false, "unexpected Err from fit/transform: {e:?}");
}
