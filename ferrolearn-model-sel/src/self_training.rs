//! Semi-supervised self-training classifier.
//!
//! [`SelfTrainingClassifier`] is a meta-estimator that wraps a supervised
//! classifier and iteratively assigns pseudo-labels to unlabeled samples
//! whose predicted probabilities exceed a confidence `threshold`.
//!
//! # Algorithm
//!
//! 1. Split data into labeled (where `y != UNLABELED`) and unlabeled subsets.
//! 2. Fit the base model on the labeled data only.
//! 3. Predict probabilities on the unlabeled data.
//! 4. For each unlabeled sample whose maximum predicted probability exceeds
//!    the threshold, add it to the labeled set with the predicted label.
//! 5. Repeat steps 2--4 until convergence (no new pseudo-labels) or
//!    `max_iter` is reached.
//!
//! # Sentinel Value
//!
//! Unlabeled samples are identified by `y[i] == usize::MAX`.
//!
//! ## REQ status
//!
//! Mirrors `sklearn.semi_supervised.SelfTrainingClassifier`
//! (`sklearn/semi_supervised/_self_training.py:39`) at v1.5.2. ferrolearn is a
//! SIMPLIFIED BINARY version: a `FitFn` closure base (R-DEV-7), a single score
//! interpreted as P(class 1), threshold criterion only. Every REQ is BINARY
//! (R-DEFER-2): SHIPPED or NOT-STARTED (with a concrete blocker).
//!
//! | REQ | Status | Notes |
//! |---|---|---|
//! | REQ-SELFTRAIN-LOOP (binary threshold self-training iteration + n_iter) | SHIPPED | `while not-all-labeled && n_iter < max_iter { n_iter+=1; refit on labeled; predict unlabeled; pseudo-label confident; if 0 selected break }` then ONE final fit on all labeled — mirrors `_self_training.py:247-297`; `n_iter` matches sklearn `n_iter_` (all-labeled ⇒ 0, no_change/converge/max_iter cases oracle-verified). Guards `guard_selftrain_loop_pseudo_labels_confident_rows`, `guard_selftrain_loop_non_confident_keeps_sentinel`, `divergence_all_labeled_n_iter_1846`. |
//! | REQ-THRESHOLD-STRICT (selection `>` not `>=`) | SHIPPED | `max_prob > threshold` (strict), matching `selected = max_proba > self.threshold` (`:262`) — a sample at exactly the threshold is NOT selected (was `>=`, fixed #1841). Test `divergence_threshold_strict_boundary_1841`. |
//! | REQ-THRESHOLD-BOUND (threshold ∈ [0,1)) | SHIPPED | rejects `< 0.0`, `>= 1.0`, non-finite — matching `Interval(Real, 0.0, 1.0, closed="left")` (`:164`); accepts 0.0, rejects 1.0 (was `(0,1]`, fixed #1842). Tests `divergence_threshold_bound_one_rejected_1842`, `divergence_threshold_bound_zero_accepted_1842`. |
//! | REQ-MULTICLASS (predict_proba argmax + classes_) | NOT-STARTED | binary-only labels {0,1} from a single score; sklearn `classes_[argmax(predict_proba)]` (`:256-257`). Architectural. Blocker #1843. |
//! | REQ-CRITERION-KBEST (criterion='k_best' + k_best) | NOT-STARTED | no `criterion`/`k_best` params (`:264-269`). Blocker #1844. |
//! | REQ-PREDICT-LABELS (predict returns class labels) | NOT-STARTED | `Predict` returns raw `f64` scores; sklearn `predict` returns labels via `base_estimator_.predict` (`:322`). Blocker #1845. |
//! | REQ-ATTRS (termination_condition_ / labeled_iter_) | NOT-STARTED | `n_iter` count is correct (REQ-SELFTRAIN-LOOP) but the `termination_condition_`/`labeled_iter_` attributes are absent (`:121`,`:242`). Blocker #1846. |
//! | REQ-MAX-ITER (max_iter=0 / max_iter=None) | NOT-STARTED | `max_iter: usize` rejects 0, no unlimited `None` (`:167`). Blocker #1847. |
//! | REQ-PROBA-DELEGATION (predict_proba/decision_function/score) | NOT-STARTED | no proba/decision/score delegation to the base (`:324+`). Blocker #1848. |
//! | REQ-X-1 (R-SUBSTRATE) | NOT-STARTED | `ndarray::{Array1, Array2}`; destination `ferray-core` (R-SUBSTRATE-1). Blocker #1849. |
//! | REQ-X-2 (non-test production consumer) | SHIPPED | re-exported `pub use self_training::{SelfTrainingClassifier, FittedSelfTrainingClassifier, UNLABELED}` in `lib.rs` (boundary API, S5/R-DEFER-1). No internal caller, no Python binding (honest underclaim). |

use ferrolearn_core::{FerroError, Predict};
use ndarray::{Array1, Array2};

use crate::calibration::{FitFn, PredictFn};

/// Sentinel value indicating that a sample is unlabeled.
///
/// When constructing the target array for [`SelfTrainingClassifier`], set
/// `y[i] = UNLABELED` for any sample without a known label.
pub const UNLABELED: usize = usize::MAX;

// ---------------------------------------------------------------------------
// SelfTrainingClassifier
// ---------------------------------------------------------------------------

/// A semi-supervised meta-estimator that iteratively pseudo-labels
/// unlabeled data.
///
/// # Example
///
/// ```rust,no_run
/// use ferrolearn_model_sel::self_training::{SelfTrainingClassifier, UNLABELED};
/// use ndarray::{Array1, Array2};
///
/// let fit_fn = Box::new(|_x: &Array2<f64>, _y: &Array1<usize>| {
///     Ok(Box::new(|x: &Array2<f64>| {
///         Ok(x.column(0).to_owned())
///     }) as Box<dyn Fn(&Array2<f64>) -> Result<Array1<f64>, ferrolearn_core::FerroError>>)
/// });
///
/// let st = SelfTrainingClassifier::new(fit_fn)
///     .threshold(0.8)
///     .max_iter(15);
/// ```
pub struct SelfTrainingClassifier {
    /// Closure that fits the base model and returns a predict function.
    fit_fn: FitFn,
    /// Confidence threshold for pseudo-labeling (default: 0.75).
    threshold: f64,
    /// Maximum number of self-training iterations (default: 10).
    max_iter: usize,
}

impl SelfTrainingClassifier {
    /// Create a new [`SelfTrainingClassifier`] with default parameters.
    ///
    /// - `threshold` defaults to `0.75`.
    /// - `max_iter` defaults to `10`.
    pub fn new(fit_fn: FitFn) -> Self {
        Self {
            fit_fn,
            threshold: 0.75,
            max_iter: 10,
        }
    }

    /// Set the confidence threshold for pseudo-labeling.
    ///
    /// Only unlabeled samples whose maximum predicted probability meets or
    /// exceeds this value will be added to the labeled set.
    #[must_use]
    pub fn threshold(mut self, threshold: f64) -> Self {
        self.threshold = threshold;
        self
    }

    /// Set the maximum number of self-training iterations.
    #[must_use]
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Fit the self-training classifier.
    ///
    /// Iteratively pseudo-labels unlabeled samples (marked with
    /// [`UNLABELED`]) until convergence or `max_iter` is reached.
    ///
    /// # Parameters
    ///
    /// - `x` — feature matrix with shape `(n_samples, n_features)`.
    /// - `y` — target array of length `n_samples`. Unlabeled samples must
    ///   have `y[i] == UNLABELED`.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError`] if:
    /// - `x` and `y` have mismatched lengths.
    /// - No labeled samples are present.
    /// - The threshold is outside `[0, 1)` (or non-finite).
    /// - The base model fails to fit or predict.
    pub fn fit(
        &self,
        x: &Array2<f64>,
        y: &Array1<usize>,
    ) -> Result<FittedSelfTrainingClassifier, FerroError> {
        let n_samples = x.nrows();

        if y.len() != n_samples {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_samples],
                actual: vec![y.len()],
                context: "SelfTrainingClassifier::fit: y length must equal x rows".into(),
            });
        }
        if !self.threshold.is_finite() || self.threshold < 0.0 || self.threshold >= 1.0 {
            return Err(FerroError::InvalidParameter {
                name: "threshold".into(),
                reason: format!("must be in [0, 1), got {}", self.threshold),
            });
        }
        if self.max_iter == 0 {
            return Err(FerroError::InvalidParameter {
                name: "max_iter".into(),
                reason: "must be >= 1".into(),
            });
        }

        // Separate labeled and unlabeled indices.
        let mut labeled_mask: Vec<bool> = y.iter().map(|&l| l != UNLABELED).collect();
        let initial_labeled_count = labeled_mask.iter().filter(|&&m| m).count();

        if initial_labeled_count == 0 {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "SelfTrainingClassifier: no labeled samples".into(),
            });
        }

        // Working copy of labels — we'll fill in pseudo-labels.
        let mut labels: Vec<usize> = y.to_vec();

        let mut n_iter = 0;

        // Self-training loop, mirroring sklearn's while-semantics
        // (`sklearn/semi_supervised/_self_training.py:247-282`): iterate only
        // while unlabeled samples remain AND the iteration cap is not reached.
        // `n_iter` is incremented INSIDE the loop, so all-labeled input never
        // increments it (`n_iter == 0`).
        while labeled_mask.iter().any(|&m| !m) && n_iter < self.max_iter {
            n_iter += 1;

            // Gather labeled indices and refit on the currently-labeled set.
            let labeled_idx: Vec<usize> = labeled_mask
                .iter()
                .enumerate()
                .filter_map(|(i, &m)| if m { Some(i) } else { None })
                .collect();
            let x_labeled = select_rows(x, &labeled_idx);
            let y_labeled = Array1::from_vec(labeled_idx.iter().map(|&i| labels[i]).collect());
            let pred_fn = (self.fit_fn)(&x_labeled, &y_labeled)?;

            // Predict on the current unlabeled set.
            let unlabeled_idx: Vec<usize> = labeled_mask
                .iter()
                .enumerate()
                .filter_map(|(i, &m)| if m { None } else { Some(i) })
                .collect();
            let x_unlabeled = select_rows(x, &unlabeled_idx);
            let scores = pred_fn(&x_unlabeled)?;

            // Pseudo-label high-confidence samples.
            // Scores are treated as probabilities; samples with
            // max(score, 1-score) > threshold are pseudo-labeled (strict `>`,
            // mirroring sklearn `_self_training.py:262`).
            let mut new_labels_count = 0;
            for (local_i, &global_i) in unlabeled_idx.iter().enumerate() {
                let score = scores[local_i];
                // Interpret score as probability of class 1.
                let prob = score.clamp(0.0, 1.0);
                let max_prob = prob.max(1.0 - prob);

                if max_prob > self.threshold {
                    let predicted_label = if prob >= 0.5 { 1 } else { 0 };
                    labels[global_i] = predicted_label;
                    labeled_mask[global_i] = true;
                    new_labels_count += 1;
                }
            }

            if new_labels_count == 0 {
                // Converged — no new pseudo-labels (sklearn 'no_change').
                break;
            }
        }

        // One final fit on ALL currently-labeled samples, matching sklearn's
        // post-loop `base_estimator_.fit(...)` (`_self_training.py:295`).
        let final_labeled_idx: Vec<usize> = labeled_mask
            .iter()
            .enumerate()
            .filter_map(|(i, &m)| if m { Some(i) } else { None })
            .collect();
        let x_final = select_rows(x, &final_labeled_idx);
        let y_final = Array1::from_vec(final_labeled_idx.iter().map(|&i| labels[i]).collect());
        let final_predict = (self.fit_fn)(&x_final, &y_final)?;

        let final_labels = Array1::from_vec(labels);

        Ok(FittedSelfTrainingClassifier {
            predict_fn: final_predict,
            transduced_labels: final_labels,
            n_iter,
        })
    }
}

// ---------------------------------------------------------------------------
// FittedSelfTrainingClassifier
// ---------------------------------------------------------------------------

/// A fitted self-training classifier.
///
/// Obtained by calling [`SelfTrainingClassifier::fit`]. Implements
/// [`Predict`] to produce raw scores on new data.
pub struct FittedSelfTrainingClassifier {
    /// Predict function from the final base model fit.
    predict_fn: PredictFn,
    /// The transduced labels: the original labels plus any pseudo-labels
    /// assigned during self-training. Samples that were never pseudo-labeled
    /// retain the [`UNLABELED`] sentinel.
    transduced_labels: Array1<usize>,
    /// Number of productive self-training iterations performed (0 if all
    /// samples were labeled to begin with), mirroring sklearn's `n_iter_`.
    n_iter: usize,
}

impl FittedSelfTrainingClassifier {
    /// Return the transduced labels.
    ///
    /// This includes both the original labeled samples and any pseudo-labels
    /// assigned during self-training. Samples that were never assigned a
    /// pseudo-label retain the [`UNLABELED`] sentinel value.
    pub fn transduced_labels(&self) -> &Array1<usize> {
        &self.transduced_labels
    }

    /// Return the number of self-training iterations that were performed.
    pub fn n_iter(&self) -> usize {
        self.n_iter
    }
}

impl Predict<Array2<f64>> for FittedSelfTrainingClassifier {
    type Output = Array1<f64>;
    type Error = FerroError;

    /// Predict raw scores for the given feature matrix.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError`] if the base model prediction fails.
    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>, FerroError> {
        (self.predict_fn)(x)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Select rows from a 2D array by index.
fn select_rows(x: &Array2<f64>, indices: &[usize]) -> Array2<f64> {
    let n_cols = x.ncols();
    let n_rows = indices.len();
    let mut data = Vec::with_capacity(n_rows * n_cols);
    for &i in indices {
        data.extend(x.row(i).iter().copied());
    }
    Array2::from_shape_vec((n_rows, n_cols), data)
        .expect("select_rows: shape should always be valid")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a `FitFn` that uses the first feature as a score.
    fn feature0_fit_fn() -> FitFn {
        Box::new(|_x: &Array2<f64>, _y: &Array1<usize>| {
            Ok(Box::new(|x: &Array2<f64>| Ok(x.column(0).to_owned())) as PredictFn)
        })
    }

    /// Build a `FitFn` that returns a constant score.
    fn constant_fit_fn(value: f64) -> FitFn {
        Box::new(move |_x: &Array2<f64>, _y: &Array1<usize>| {
            let v = value;
            Ok(Box::new(move |x: &Array2<f64>| Ok(Array1::from_elem(x.nrows(), v))) as PredictFn)
        })
    }

    #[test]
    fn test_self_training_all_labeled() {
        // When all samples are labeled, self-training should just fit once.
        let x = Array2::from_shape_vec((6, 1), vec![0.0, 0.1, 0.2, 0.8, 0.9, 1.0]).unwrap();
        let y = Array1::from_vec(vec![0, 0, 0, 1, 1, 1]);

        let st = SelfTrainingClassifier::new(feature0_fit_fn());
        let fitted = st.fit(&x, &y).unwrap();

        // All labels should remain unchanged.
        for (i, &l) in fitted.transduced_labels().iter().enumerate() {
            assert_eq!(l, y[i]);
        }
        // All-labeled input: the while-body never runs (sklearn n_iter_==0).
        assert_eq!(fitted.n_iter(), 0);
    }

    #[test]
    fn test_self_training_pseudo_labels_assigned() {
        // Labeled: first 4 samples, unlabeled: last 2.
        // Class 0 has features near 0, class 1 near 1.
        let x = Array2::from_shape_vec((6, 1), vec![0.0, 0.1, 0.9, 1.0, 0.05, 0.95]).unwrap();
        let y = Array1::from_vec(vec![0, 0, 1, 1, UNLABELED, UNLABELED]);

        let st = SelfTrainingClassifier::new(feature0_fit_fn()).threshold(0.5);
        let fitted = st.fit(&x, &y).unwrap();

        // The unlabeled samples should have been pseudo-labeled.
        let labels = fitted.transduced_labels();
        assert_eq!(labels[4], 0); // feature 0.05 -> class 0
        assert_eq!(labels[5], 1); // feature 0.95 -> class 1
    }

    #[test]
    fn test_self_training_high_threshold_no_labels() {
        // With threshold = 0.99, scores whose max_prob <= 0.99 never qualify.
        // (threshold must be in [0, 1); 1.0 is rejected — see #1842.)
        let x = Array2::from_shape_vec((4, 1), vec![0.1, 0.9, 0.4, 0.6]).unwrap();
        let y = Array1::from_vec(vec![0, 1, UNLABELED, UNLABELED]);

        let st = SelfTrainingClassifier::new(feature0_fit_fn()).threshold(0.99);
        let fitted = st.fit(&x, &y).unwrap();

        // Unlabeled samples should remain UNLABELED (max_prob 0.9 < 0.99).
        assert_eq!(fitted.transduced_labels()[2], UNLABELED);
        assert_eq!(fitted.transduced_labels()[3], UNLABELED);
    }

    #[test]
    fn test_self_training_no_labeled_samples() {
        let x = Array2::from_elem((4, 1), 0.5);
        let y = Array1::from_elem(4, UNLABELED);

        let st = SelfTrainingClassifier::new(feature0_fit_fn());
        assert!(st.fit(&x, &y).is_err());
    }

    #[test]
    fn test_self_training_shape_mismatch() {
        let x = Array2::from_elem((4, 1), 0.5);
        let y = Array1::from_vec(vec![0, 1]); // wrong length

        let st = SelfTrainingClassifier::new(feature0_fit_fn());
        assert!(st.fit(&x, &y).is_err());
    }

    #[test]
    fn test_self_training_invalid_threshold() {
        let x = Array2::from_elem((4, 1), 0.5);
        let y = Array1::from_vec(vec![0, 1, UNLABELED, UNLABELED]);

        // Negative thresholds are out of the [0, 1) interval (#1842).
        let st = SelfTrainingClassifier::new(feature0_fit_fn()).threshold(-0.1);
        assert!(st.fit(&x, &y).is_err());

        let st2 = SelfTrainingClassifier::new(feature0_fit_fn()).threshold(1.5);
        assert!(st2.fit(&x, &y).is_err());
    }

    #[test]
    fn test_self_training_invalid_max_iter() {
        let x = Array2::from_elem((4, 1), 0.5);
        let y = Array1::from_vec(vec![0, 1, UNLABELED, UNLABELED]);

        let st = SelfTrainingClassifier::new(feature0_fit_fn()).max_iter(0);
        assert!(st.fit(&x, &y).is_err());
    }

    #[test]
    fn test_self_training_predict() {
        let x = Array2::from_shape_vec((4, 1), vec![0.0, 0.1, 0.9, 1.0]).unwrap();
        let y = Array1::from_vec(vec![0, 0, 1, 1]);

        let st = SelfTrainingClassifier::new(feature0_fit_fn());
        let fitted = st.fit(&x, &y).unwrap();

        let new_x = Array2::from_shape_vec((2, 1), vec![0.2, 0.8]).unwrap();
        let scores = fitted.predict(&new_x).unwrap();
        assert_eq!(scores.len(), 2);
        assert!((scores[0] - 0.2).abs() < 1e-10);
        assert!((scores[1] - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_self_training_max_iter_respected() {
        // Constant score of 0.5 means max_prob = 0.5, which won't meet
        // threshold = 0.75 so we should run all max_iter iterations.
        let x = Array2::from_elem((4, 1), 0.5);
        let y = Array1::from_vec(vec![0, 1, UNLABELED, UNLABELED]);

        let st = SelfTrainingClassifier::new(constant_fit_fn(0.5))
            .threshold(0.75)
            .max_iter(3);
        let fitted = st.fit(&x, &y).unwrap();

        // Should converge after iteration 1 since no labels are added.
        assert!(fitted.n_iter() <= 3);
    }

    #[test]
    fn test_self_training_iterative_labeling() {
        // Design a scenario where pseudo-labeling happens across multiple
        // iterations. With threshold 0.7:
        // - Iteration 1: samples with score >= 0.7 get labeled
        // - Iteration 2: the model may label more.
        let x =
            Array2::from_shape_vec((8, 1), vec![0.0, 0.1, 0.9, 1.0, 0.05, 0.95, 0.3, 0.7]).unwrap();
        let y = Array1::from_vec(vec![0, 0, 1, 1, UNLABELED, UNLABELED, UNLABELED, UNLABELED]);

        let st = SelfTrainingClassifier::new(feature0_fit_fn())
            .threshold(0.7)
            .max_iter(10);
        let fitted = st.fit(&x, &y).unwrap();

        // Samples with feature 0.05 and 0.95 should be pseudo-labeled
        // (their scores 0.05 and 0.95 have max_prob 0.95 and 0.95 >= 0.7).
        assert_ne!(fitted.transduced_labels()[4], UNLABELED);
        assert_ne!(fitted.transduced_labels()[5], UNLABELED);
    }
}
