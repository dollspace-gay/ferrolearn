//! Gaussian Naive Bayes classifier.
//!
//! This module provides [`GaussianNB`], a Naive Bayes classifier that assumes
//! features are normally (Gaussian) distributed within each class. Each
//! feature's likelihood is modelled by the Gaussian density
//! `N(mu_ci, sigma_ci^2)` estimated from the training data.
//!
//! # Examples
//!
//! ```
//! use ferrolearn_bayes::GaussianNB;
//! use ferrolearn_core::{Fit, Predict};
//! use ndarray::{array, Array2};
//!
//! let x = Array2::from_shape_vec(
//!     (6, 2),
//!     vec![1.0, 2.0, 1.5, 1.8, 2.0, 2.5,
//!          6.0, 7.0, 6.5, 6.8, 7.0, 7.5],
//! ).unwrap();
//! let y = array![0usize, 0, 0, 1, 1, 1];
//!
//! let model = GaussianNB::<f64>::new();
//! let fitted = model.fit(&x, &y).unwrap();
//! let preds = fitted.predict(&x).unwrap();
//! assert_eq!(preds.len(), 6);
//! ```
//!
//! # `## REQ status`
//!
//! Binary classification (R-DEFER-2): two states only — SHIPPED needs impl + a
//! non-test production consumer + green verification; NOT-STARTED carries the
//! open prereq blocker. The non-test production consumer is `_RsGaussianNB`
//! (`ferrolearn-python/src/classifiers.rs`), which exercises `fit` /
//! `predict` / `predict_proba` / `classes_` against the library `FittedGaussianNB`
//! and is surfaced as `ferrolearn.GaussianNB`; plus the in-crate
//! `impl PipelineEstimator for GaussianNB` (`fit_pipeline` / `predict_pipeline`).
//! Green verification = the in-tree `gaussian` lib tests + the live-sklearn
//! divergence pins (`ferrolearn-bayes/tests/divergence_gaussian.rs`:
//! `divergence_gaussian_epsilon_global_var_no_floor` (#891),
//! `divergence_gaussian_priors_sum_not_one_rejected` (#893),
//! `green_gaussian_predict_labels`, `green_gaussian_predict_proba_sums_to_one`,
//! `green_gaussian_score_accuracy` — all passing). Cites use symbol anchors
//! (ferrolearn) / `file:line` (sklearn 1.5.2, commit 156ef14). Live oracle =
//! installed sklearn 1.5.2.
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (`epsilon_` — global per-feature variance `var_smoothing * np.var(X, axis=0).max()`, no floor) | SHIPPED | `fn fit` for `GaussianNB` computes `global_max_var` as `np.var(X, axis=0).max()` (population variance, ddof=0, over ALL rows irrespective of class) and `let epsilon = self.var_smoothing * global_max_var` — no `.max(1.0)` floor — mirroring `epsilon_ = self.var_smoothing * np.var(X, axis=0).max()` (`sklearn/naive_bayes.py:431`). Non-test consumer: `_RsGaussianNB::fit` (`classifiers.rs`) → `FittedGaussianNB`. Verified: green pin `divergence_gaussian_epsilon_global_var_no_floor` (#891): on `X=[[1,2],[1.5,1.8],[2,2.5],[6,7],[6.5,6.8],[7,7.5]]`, `y=[0,0,0,1,1,1]`, sklearn `epsilon_=6.416666666666667e-9`; `predict_joint_log_proba([[1.2,2.1],[6.6,7.1]])[0][0] = -0.6823015899121332`, ferrolearn matches to ≤1e-9. |
//! | REQ-2 (`priors` validation — length + sum≈1 + non-negative) | SHIPPED | `fn fit` validates `priors.len() != n_classes`, then `(prior_sum - 1).abs() > 1e-9` → "The sum of the priors should be 1." and `priors.iter().any(|&p| p < 0)` → "Priors must be non-negative.", mirroring `naive_bayes.py:448-455`. Non-test consumer: `_RsGaussianNB::fit` (maps `FerroError` → `PyValueError`). Verified: green pin `divergence_gaussian_priors_sum_not_one_rejected` (#893): `with_class_prior([0.5,0.3]).fit(X,y)` returns `Err` (sklearn raises `ValueError("The sum of the priors should be 1.")`). |
//! | REQ-3 (`_joint_log_likelihood` + `predict` / `predict_proba` / `predict_log_proba` / `predict_joint_log_proba` VALUE) | SHIPPED | `impl BaseNB::joint_log_likelihood` for `FittedGaussianNB` computes `log_prior[ci] - 0.5*(log(2*pi*var) + (x-mu)^2/var)` summed over features (`naive_bayes.py:506-515`); the four `predict_*` delegate to the `BaseNB` provided methods. After the REQ-1 epsilon fix the smoothed `var` matches sklearn, so the VALUES match to ~1e-9. Non-test consumer: `_RsGaussianNB::predict`/`predict_proba` (`classifiers.rs`). Verified: green pins `divergence_gaussian_epsilon_global_var_no_floor` (`predict_joint_log_proba` value ≤1e-9), `green_gaussian_predict_labels` (`[0,1]`), `green_gaussian_predict_proba_sums_to_one` (rows sum to 1.0). |
//! | REQ-4 (`theta_` per-class mean + data-derived `log_prior` / `class_prior_` + `score`) | SHIPPED | `fn fit` computes the per-class per-feature mean into `theta` (`np.mean(X_class, axis=0)`, `naive_bayes.py:324`) and `log_prior[ci] = ln(count_c / n_total)` (empirical `class_prior_ = class_count_ / class_count_.sum()`, `naive_bayes.py:502`); `pub fn score` is mean accuracy (`ClassifierMixin.score` analog). Non-test consumer: `_RsGaussianNB::predict`/`classes_` → `fitted.predict`/`fitted.classes()`. Verified: `green_gaussian_score_accuracy` (`score(X,y)=1.0`), `green_gaussian_predict_labels` (`[0,1]`), in-tree `test_gaussian_nb_has_classes`/`test_gaussian_nb_three_classes`. |
//! | REQ-5 (`sample_weight` in `fit`) | NOT-STARTED | open prereq blocker **#894**. sklearn `fit(X, y, sample_weight=None)` (`naive_bayes.py:239`) supports weighted `theta_`/`var_`/`class_count_` via `_update_mean_variance` (`np.average(..., weights=sw)`, `naive_bayes.py:319-320`). ferrolearn's `impl Fit<Array2<F>, Array1<usize>>` has signature `fn fit(&self, x, y)` — no `sample_weight` parameter on `fit` or `partial_fit`. |
//! | REQ-6 (`partial_fit` epsilon-once semantics) | NOT-STARTED | open prereq blocker **#895**. sklearn fixes `epsilon_` at the first fit and does the subtract-before / re-add-after dance (`var_ -= epsilon_` `naive_bayes.py:465`, `var_ += epsilon_` `naive_bayes.py:497`). `FittedGaussianNB::partial_fit` RECOMPUTES `epsilon = var_smoothing * max_var.max(F::one())` from the current `sigma` each call (the `max_var`/`epsilon` block) — different smoothing per call (also still per-class-max with a `1.0` floor, unlike the now-fixed `fit`). |
//! | REQ-7 (fitted accessors `theta_` / `var_` / `epsilon_` / `class_count_` / `class_prior_`) | NOT-STARTED | open prereq blocker **#896**. sklearn exposes these (`naive_bayes.py:171-202`). `FittedGaussianNB` stores `theta` / `sigma` / `raw_sigma` / `class_counts` / `log_prior` / `var_smoothing` as PRIVATE fields with no accessor; only `classes()` (via `HasClasses`) is public — no `theta_` / `var_` / `epsilon_` / `class_count_` / `class_prior_` getter. |
//! | REQ-8 (PyO3 surface — `var_smoothing` / `priors` / `sample_weight` + getters) | NOT-STARTED | open prereq blocker **#897**. `_RsGaussianNB` (`ferrolearn-python/src/classifiers.rs`) exposes `new(var_smoothing)` / `fit` / `predict` / `predict_proba` / `classes_` only — no `priors` kwarg, no `sample_weight` on `fit`, no `theta_` / `var_` / `epsilon_` / `class_count_` / `class_prior_` getters, no `predict_log_proba` / `score` / `partial_fit`. The fix belongs in `ferrolearn-python` (multi-file). |
//! | REQ-9 (ferray substrate) | NOT-STARTED | open prereq blocker **#898**. `gaussian.rs` imports `ndarray::{Array1, Array2}` + `num_traits::Float` (the wrong substrate, R-SUBSTRATE-1); not migrated to `ferray-core`. |

use crate::base::BaseNB;
use ferrolearn_core::error::FerroError;
use ferrolearn_core::introspection::HasClasses;
use ferrolearn_core::pipeline::{FittedPipelineEstimator, PipelineEstimator};
use ferrolearn_core::traits::{Fit, Predict};
use ndarray::{Array1, Array2};
use num_traits::{Float, FromPrimitive, ToPrimitive};

/// Gaussian Naive Bayes classifier.
///
/// Assumes features are Gaussian-distributed within each class.
/// Variance smoothing is applied to avoid numerical issues with zero variance.
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
#[derive(Debug, Clone)]
pub struct GaussianNB<F> {
    /// Variance smoothing parameter added to all variances.
    /// Prevents division by zero when a feature has near-zero variance.
    /// Default: `1e-9`.
    pub var_smoothing: F,
    /// Optional user-supplied class priors. If set, these are used
    /// instead of computing priors from the data.
    pub class_prior: Option<Vec<F>>,
}

impl<F: Float> GaussianNB<F> {
    /// Create a new `GaussianNB` with default settings.
    ///
    /// Default: `var_smoothing = 1e-9`, `class_prior = None`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            var_smoothing: F::from(1e-9).unwrap(),
            class_prior: None,
        }
    }

    /// Set the variance smoothing parameter.
    #[must_use]
    pub fn with_var_smoothing(mut self, var_smoothing: F) -> Self {
        self.var_smoothing = var_smoothing;
        self
    }

    /// Set user-supplied class priors.
    ///
    /// The priors must sum to 1.0 and have length equal to the number
    /// of classes discovered during fitting.
    #[must_use]
    pub fn with_class_prior(mut self, priors: Vec<F>) -> Self {
        self.class_prior = Some(priors);
        self
    }
}

impl<F: Float> Default for GaussianNB<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted Gaussian Naive Bayes classifier.
///
/// Stores the per-class prior, mean, and variance computed during fitting.
/// Also stores sufficient statistics for incremental (partial) fitting.
#[derive(Debug, Clone)]
pub struct FittedGaussianNB<F> {
    /// Sorted unique class labels.
    classes: Vec<usize>,
    /// Log prior probability for each class, shape `(n_classes,)`.
    log_prior: Array1<F>,
    /// Per-class per-feature mean, shape `(n_classes, n_features)`.
    theta: Array2<F>,
    /// Per-class per-feature variance (smoothed), shape `(n_classes, n_features)`.
    sigma: Array2<F>,
    /// Per-class sample counts (for partial_fit updates).
    class_counts: Vec<usize>,
    /// Per-class per-feature unsmoothed variance, shape `(n_classes, n_features)`.
    /// Stored for Welford update during partial_fit.
    raw_sigma: Array2<F>,
    /// Variance smoothing parameter carried forward for partial_fit.
    var_smoothing: F,
    /// Optional user-supplied class priors.
    class_prior: Option<Vec<F>>,
}

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, Array1<usize>> for GaussianNB<F> {
    type Fitted = FittedGaussianNB<F>;
    type Error = FerroError;

    /// Fit the Gaussian NB model.
    ///
    /// # Errors
    ///
    /// - [`FerroError::ShapeMismatch`] if `x` and `y` have different numbers of rows.
    /// - [`FerroError::InsufficientSamples`] if there are no samples.
    fn fit(&self, x: &Array2<F>, y: &Array1<usize>) -> Result<FittedGaussianNB<F>, FerroError> {
        let (n_samples, n_features) = x.dim();

        if n_samples == 0 {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "GaussianNB requires at least one sample".into(),
            });
        }

        if n_samples != y.len() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_samples],
                actual: vec![y.len()],
                context: "y length must match number of samples in X".into(),
            });
        }

        // Collect sorted unique classes.
        let mut classes: Vec<usize> = y.to_vec();
        classes.sort_unstable();
        classes.dedup();
        let n_classes = classes.len();

        let mut theta = Array2::<F>::zeros((n_classes, n_features));
        let mut sigma = Array2::<F>::zeros((n_classes, n_features));
        let mut log_prior = Array1::<F>::zeros(n_classes);

        let n_f = F::from(n_samples).unwrap();

        for (ci, &class_label) in classes.iter().enumerate() {
            // Collect indices of samples belonging to this class.
            let class_mask: Vec<usize> = y
                .iter()
                .enumerate()
                .filter_map(|(i, &label)| if label == class_label { Some(i) } else { None })
                .collect();

            let n_c = class_mask.len();
            let n_c_f = F::from(n_c).unwrap();

            // Log prior: log(count_c / n_total).
            log_prior[ci] = (n_c_f / n_f).ln();

            // Compute per-feature mean.
            for j in 0..n_features {
                let mean = class_mask.iter().fold(F::zero(), |acc, &i| acc + x[[i, j]]) / n_c_f;
                theta[[ci, j]] = mean;

                // Compute variance (population variance).
                let var = if n_c > 1 {
                    class_mask.iter().fold(F::zero(), |acc, &i| {
                        let diff = x[[i, j]] - mean;
                        acc + diff * diff
                    }) / n_c_f
                } else {
                    F::zero()
                };

                sigma[[ci, j]] = var;
            }
        }

        // Store raw (unsmoothed) variance for partial_fit.
        let raw_sigma = sigma.clone();

        // Apply variance smoothing, mirroring scikit-learn
        // (`naive_bayes.py:431`): epsilon_ = var_smoothing * np.var(X, axis=0).max()
        // — the GLOBAL per-feature population variance (ddof=0) over ALL samples
        // (every row, irrespective of class), reduced by .max() over features,
        // times var_smoothing. No floor.
        let mut global_max_var = F::zero();
        for j in 0..n_features {
            let mean_j = (0..n_samples).fold(F::zero(), |acc, i| acc + x[[i, j]]) / n_f;
            let var_j = (0..n_samples).fold(F::zero(), |acc, i| {
                let diff = x[[i, j]] - mean_j;
                acc + diff * diff
            }) / n_f;
            if var_j > global_max_var {
                global_max_var = var_j;
            }
        }
        let epsilon = self.var_smoothing * global_max_var;
        sigma.mapv_inplace(|v| v + epsilon);

        // Use user-supplied class priors if provided.
        if let Some(ref priors) = self.class_prior {
            if priors.len() != n_classes {
                return Err(FerroError::InvalidParameter {
                    name: "class_prior".into(),
                    reason: format!(
                        "length {} does not match number of classes {}",
                        priors.len(),
                        n_classes
                    ),
                });
            }
            // sklearn `GaussianNB._partial_fit` (naive_bayes.py:451-452):
            //   if not np.isclose(priors.sum(), 1.0):
            //       raise ValueError("The sum of the priors should be 1.")
            let prior_sum = priors.iter().fold(F::zero(), |acc, &p| acc + p);
            let tol = F::from(1e-9).unwrap_or_else(F::epsilon);
            if (prior_sum - F::one()).abs() > tol {
                return Err(FerroError::InvalidParameter {
                    name: "class_prior".into(),
                    reason: "The sum of the priors should be 1.".into(),
                });
            }
            // sklearn (naive_bayes.py:454-455):
            //   if (priors < 0).any():
            //       raise ValueError("Priors must be non-negative.")
            if priors.iter().any(|&p| p < F::zero()) {
                return Err(FerroError::InvalidParameter {
                    name: "class_prior".into(),
                    reason: "Priors must be non-negative.".into(),
                });
            }
            for (ci, &p) in priors.iter().enumerate() {
                log_prior[ci] = p.ln();
            }
        }

        // Collect class counts for partial_fit.
        let class_counts: Vec<usize> = classes
            .iter()
            .map(|&label| y.iter().filter(|&&l| l == label).count())
            .collect();

        Ok(FittedGaussianNB {
            classes,
            log_prior,
            theta,
            sigma,
            class_counts,
            raw_sigma,
            var_smoothing: self.var_smoothing,
            class_prior: self.class_prior.clone(),
        })
    }
}

impl<F: Float + Send + Sync + 'static> FittedGaussianNB<F> {
    /// Incrementally update the model with new data using Welford's algorithm.
    ///
    /// This method updates the running mean and variance for each class using
    /// a numerically stable online algorithm.
    ///
    /// # Errors
    ///
    /// - [`FerroError::ShapeMismatch`] if `x` and `y` have different row counts
    ///   or the number of features does not match the fitted model.
    pub fn partial_fit(&mut self, x: &Array2<F>, y: &Array1<usize>) -> Result<(), FerroError> {
        let (n_samples, n_features) = x.dim();

        if n_samples == 0 {
            return Ok(());
        }

        if n_samples != y.len() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_samples],
                actual: vec![y.len()],
                context: "y length must match number of samples in X".into(),
            });
        }

        if n_features != self.theta.ncols() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![self.theta.ncols()],
                actual: vec![n_features],
                context: "number of features must match fitted GaussianNB".into(),
            });
        }

        for &class_label in y {
            let ci = if let Some(idx) = self.classes.iter().position(|&c| c == class_label) {
                idx
            } else {
                // New class discovered: extend arrays.
                self.classes.push(class_label);
                let ci = self.classes.len() - 1;

                // Sort classes and find the new index.
                let mut sorted_classes = self.classes.clone();
                sorted_classes.sort_unstable();

                // Rebuild with insertion.
                let insert_pos = sorted_classes
                    .iter()
                    .position(|&c| c == class_label)
                    .unwrap();

                self.classes = sorted_classes;
                let n_classes = self.classes.len();

                // Expand theta, raw_sigma, sigma, log_prior, class_counts.
                let mut new_theta = Array2::<F>::zeros((n_classes, n_features));
                let mut new_sigma = Array2::<F>::zeros((n_classes, n_features));
                let mut new_raw_sigma = Array2::<F>::zeros((n_classes, n_features));
                let mut new_log_prior = Array1::<F>::zeros(n_classes);
                let mut new_counts = vec![0usize; n_classes];

                let mut old_idx = 0;
                for new_idx in 0..n_classes {
                    if new_idx == insert_pos {
                        // New class: zero-initialized.
                        continue;
                    }
                    if old_idx < self.theta.nrows() {
                        for j in 0..n_features {
                            new_theta[[new_idx, j]] = self.theta[[old_idx, j]];
                            new_sigma[[new_idx, j]] = self.sigma[[old_idx, j]];
                            new_raw_sigma[[new_idx, j]] = self.raw_sigma[[old_idx, j]];
                        }
                        new_log_prior[new_idx] = self.log_prior[old_idx];
                        new_counts[new_idx] = self.class_counts[old_idx];
                        old_idx += 1;
                    }
                }

                self.theta = new_theta;
                self.sigma = new_sigma;
                self.raw_sigma = new_raw_sigma;
                self.log_prior = new_log_prior;
                self.class_counts = new_counts;

                let _ = ci; // suppress unused
                insert_pos
            };

            // We already have the class index. Now gather samples for this class.
            let _ = ci; // Will be used below.
        }

        // Now update per class.
        for (ci, &class_label) in self.classes.iter().enumerate() {
            let new_indices: Vec<usize> = y
                .iter()
                .enumerate()
                .filter_map(|(i, &label)| if label == class_label { Some(i) } else { None })
                .collect();

            if new_indices.is_empty() {
                continue;
            }

            let old_count = self.class_counts[ci];
            let new_count = new_indices.len();
            let total_count = old_count + new_count;
            let total_f = F::from(total_count).unwrap();

            for j in 0..n_features {
                let old_mean = self.theta[[ci, j]];
                let old_var = self.raw_sigma[[ci, j]];
                let old_count_f = F::from(old_count).unwrap();
                let new_count_f = F::from(new_count).unwrap();

                // Compute new batch mean and variance.
                let new_mean_batch = new_indices
                    .iter()
                    .fold(F::zero(), |acc, &i| acc + x[[i, j]])
                    / new_count_f;

                let new_var_batch = if new_count > 1 {
                    new_indices.iter().fold(F::zero(), |acc, &i| {
                        let d = x[[i, j]] - new_mean_batch;
                        acc + d * d
                    }) / new_count_f
                } else {
                    F::zero()
                };

                // Combined mean (Welford's parallel algorithm).
                let combined_mean =
                    (old_count_f * old_mean + new_count_f * new_mean_batch) / total_f;

                // Combined variance.
                let delta = new_mean_batch - old_mean;
                let combined_var = (old_count_f * old_var
                    + new_count_f * new_var_batch
                    + old_count_f * new_count_f * delta * delta / total_f)
                    / total_f;

                self.theta[[ci, j]] = combined_mean;
                self.raw_sigma[[ci, j]] = combined_var;
            }

            self.class_counts[ci] = total_count;
        }

        // Recompute smoothed sigma.
        self.sigma = self.raw_sigma.clone();
        let max_var = self
            .sigma
            .iter()
            .fold(F::zero(), |acc, &v| if v > acc { v } else { acc });
        let epsilon = self.var_smoothing * max_var.max(F::one());
        self.sigma.mapv_inplace(|v| v + epsilon);

        // Recompute log priors.
        if self.class_prior.is_none() {
            let total_samples: usize = self.class_counts.iter().sum();
            let total_f = F::from(total_samples).unwrap();
            for (ci, &count) in self.class_counts.iter().enumerate() {
                self.log_prior[ci] = (F::from(count).unwrap() / total_f).ln();
            }
        }

        Ok(())
    }

    /// Predict class probabilities for the given feature matrix.
    ///
    /// Returns an array of shape `(n_samples, n_classes)` where each row
    /// sums to 1. Delegates to [`BaseNB::nb_predict_proba`] (the shared
    /// `_BaseNB.predict_proba` pipeline).
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of features does
    /// not match the fitted model.
    pub fn predict_proba(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        BaseNB::nb_predict_proba(self, x)
    }

    /// Compute the unnormalized joint log-likelihood `log P(c) + log P(x|c)`.
    ///
    /// Returns shape `(n_samples, n_classes)`. Matches sklearn
    /// `GaussianNB._joint_log_likelihood`. Delegates to
    /// [`BaseNB::nb_predict_joint_log_proba`].
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of features does
    /// not match the fitted model.
    pub fn predict_joint_log_proba(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        BaseNB::nb_predict_joint_log_proba(self, x)
    }

    /// Compute log of class probabilities (numerically stable).
    ///
    /// Returns shape `(n_samples, n_classes)` where each row's exponential
    /// sums to 1. Delegates to [`BaseNB::nb_predict_log_proba`] (the shared
    /// `_BaseNB.predict_log_proba` = `jll - logsumexp(jll)` pipeline).
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of features does
    /// not match the fitted model.
    pub fn predict_log_proba(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        BaseNB::nb_predict_log_proba(self, x)
    }

    /// Mean accuracy on the given test data and labels.
    ///
    /// Equivalent to sklearn's `ClassifierMixin.score`. Returns
    /// `(predict(x) == y).mean()`.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if `x.nrows() != y.len()` or
    /// the feature count does not match the fitted model.
    pub fn score(&self, x: &Array2<F>, y: &Array1<usize>) -> Result<F, FerroError> {
        if x.nrows() != y.len() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![x.nrows()],
                actual: vec![y.len()],
                context: "y length must match number of samples in X".into(),
            });
        }
        let preds = self.predict(x)?;
        let n = y.len();
        if n == 0 {
            return Ok(F::zero());
        }
        let correct = preds.iter().zip(y.iter()).filter(|(p, t)| p == t).count();
        Ok(F::from(correct).unwrap() / F::from(n).unwrap())
    }
}

impl<F: Float + Send + Sync + 'static> BaseNB<F> for FittedGaussianNB<F> {
    /// Compute the joint log-likelihood for each class — sklearn
    /// `GaussianNB._joint_log_likelihood`.
    ///
    /// Returns an array of shape `(n_samples, n_classes)` containing the
    /// unnormalized log-posterior scores.
    fn joint_log_likelihood(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let n_features_fitted = self.theta.ncols();
        if x.ncols() != n_features_fitted {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_features_fitted],
                actual: vec![x.ncols()],
                context: "number of features must match fitted GaussianNB".into(),
            });
        }

        let n_samples = x.nrows();
        let n_classes = self.classes.len();
        let n_features = x.ncols();

        let two = F::one() + F::one();
        // PI converts exactly for f32/f64; the fallback is never reached.
        let pi = F::from(std::f64::consts::PI).unwrap_or_else(F::nan);
        let log_two_pi = (two * pi).ln();

        let mut scores = Array2::<F>::zeros((n_samples, n_classes));

        for ci in 0..n_classes {
            for i in 0..n_samples {
                let mut log_likelihood = self.log_prior[ci];
                for j in 0..n_features {
                    let mu = self.theta[[ci, j]];
                    let var = self.sigma[[ci, j]];
                    let diff = x[[i, j]] - mu;
                    // log N(x; mu, var) = -0.5 * (log(2*pi*var) + (x-mu)^2/var)
                    log_likelihood =
                        log_likelihood - (log_two_pi + var.ln()) / two - diff * diff / (two * var);
                }
                scores[[i, ci]] = log_likelihood;
            }
        }

        Ok(scores)
    }

    fn nb_classes(&self) -> &[usize] {
        &self.classes
    }
}

impl<F: Float + Send + Sync + 'static> Predict<Array2<F>> for FittedGaussianNB<F> {
    type Output = Array1<usize>;
    type Error = FerroError;

    /// Predict class labels for the given feature matrix.
    ///
    /// Delegates to [`BaseNB::nb_predict`] (the shared `_BaseNB.predict` =
    /// `classes_[argmax(jll)]` pipeline, first-max tie-break).
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of features does
    /// not match the fitted model.
    fn predict(&self, x: &Array2<F>) -> Result<Array1<usize>, FerroError> {
        BaseNB::nb_predict(self, x)
    }
}

impl<F: Float + Send + Sync + 'static> HasClasses for FittedGaussianNB<F> {
    fn classes(&self) -> &[usize] {
        &self.classes
    }

    fn n_classes(&self) -> usize {
        self.classes.len()
    }
}

// Pipeline integration.
impl<F: Float + ToPrimitive + FromPrimitive + Send + Sync + 'static> PipelineEstimator<F>
    for GaussianNB<F>
{
    fn fit_pipeline(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
    ) -> Result<Box<dyn FittedPipelineEstimator<F>>, FerroError> {
        let y_usize: Array1<usize> = y.mapv(|v| v.to_usize().unwrap_or(0));
        let fitted = self.fit(x, &y_usize)?;
        Ok(Box::new(FittedGaussianNBPipeline(fitted)))
    }
}

struct FittedGaussianNBPipeline<F: Float + Send + Sync + 'static>(FittedGaussianNB<F>);

unsafe impl<F: Float + Send + Sync + 'static> Send for FittedGaussianNBPipeline<F> {}
unsafe impl<F: Float + Send + Sync + 'static> Sync for FittedGaussianNBPipeline<F> {}

impl<F: Float + ToPrimitive + FromPrimitive + Send + Sync + 'static> FittedPipelineEstimator<F>
    for FittedGaussianNBPipeline<F>
{
    fn predict_pipeline(&self, x: &Array2<F>) -> Result<Array1<F>, FerroError> {
        let preds = self.0.predict(x)?;
        Ok(preds.mapv(|v| F::from_usize(v).unwrap_or_else(F::nan)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    fn make_2class_data() -> (Array2<f64>, Array1<usize>) {
        // Two well-separated Gaussian clusters.
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                1.0, 1.0, 1.2, 0.8, 0.9, 1.1, 1.1, 0.9, // class 0
                5.0, 5.0, 5.1, 4.9, 4.8, 5.2, 5.0, 4.8, // class 1
            ],
        )
        .unwrap();
        let y = array![0usize, 0, 0, 0, 1, 1, 1, 1];
        (x, y)
    }

    #[test]
    fn test_gaussian_nb_fit_predict_2class() {
        let (x, y) = make_2class_data();
        let model = GaussianNB::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        // Should classify the training data correctly.
        let correct = preds.iter().zip(y.iter()).filter(|(p, a)| p == a).count();
        assert_eq!(correct, 8);
    }

    #[test]
    fn test_gaussian_nb_predict_proba_sums_to_one() {
        let (x, y) = make_2class_data();
        let model = GaussianNB::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();
        let proba = fitted.predict_proba(&x).unwrap();
        assert_eq!(proba.nrows(), 8);
        assert_eq!(proba.ncols(), 2);
        for i in 0..proba.nrows() {
            assert_relative_eq!(proba.row(i).sum(), 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_gaussian_nb_predict_proba_ordering() {
        let (x, y) = make_2class_data();
        let model = GaussianNB::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();
        let proba = fitted.predict_proba(&x).unwrap();
        // First 4 samples should have higher probability of class 0.
        for i in 0..4 {
            assert!(proba[[i, 0]] > proba[[i, 1]]);
        }
        // Last 4 samples should have higher probability of class 1.
        for i in 4..8 {
            assert!(proba[[i, 1]] > proba[[i, 0]]);
        }
    }

    #[test]
    fn test_gaussian_nb_has_classes() {
        let (x, y) = make_2class_data();
        let model = GaussianNB::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();
        assert_eq!(fitted.classes(), &[0, 1]);
        assert_eq!(fitted.n_classes(), 2);
    }

    #[test]
    fn test_gaussian_nb_three_classes() {
        let x = Array2::from_shape_vec(
            (9, 2),
            vec![
                0.0, 0.0, 0.1, 0.0, 0.0, 0.1, // class 0
                5.0, 0.0, 5.1, 0.0, 5.0, 0.1, // class 1
                0.0, 5.0, 0.1, 5.0, 0.0, 5.1, // class 2
            ],
        )
        .unwrap();
        let y = array![0usize, 0, 0, 1, 1, 1, 2, 2, 2];

        let model = GaussianNB::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();
        assert_eq!(fitted.n_classes(), 3);
        assert_eq!(fitted.classes(), &[0, 1, 2]);

        let preds = fitted.predict(&x).unwrap();
        let correct = preds.iter().zip(y.iter()).filter(|(p, a)| p == a).count();
        assert_eq!(correct, 9);
    }

    #[test]
    fn test_gaussian_nb_var_smoothing_effect() {
        // When all samples in a class have identical features (zero variance),
        // var_smoothing prevents division by zero.
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 1.0, 5.0, 5.0]).unwrap();
        let y = array![0usize, 0, 1, 1];

        let model_default = GaussianNB::<f64>::new();
        let fitted = model_default.fit(&x, &y).unwrap();
        // Should not panic — var_smoothing handles zero variance.
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 4);

        // With higher smoothing the model still predicts.
        let model_high = GaussianNB::<f64>::new().with_var_smoothing(0.1);
        let fitted_high = model_high.fit(&x, &y).unwrap();
        let preds_high = fitted_high.predict(&x).unwrap();
        assert_eq!(preds_high.len(), 4);
    }

    #[test]
    fn test_gaussian_nb_shape_mismatch_fit() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let y = array![0usize, 1, 0]; // Wrong length
        let model = GaussianNB::<f64>::new();
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_gaussian_nb_shape_mismatch_predict() {
        let (x, y) = make_2class_data();
        let model = GaussianNB::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();

        // Wrong number of features.
        let x_bad = Array2::from_shape_vec((4, 3), vec![1.0; 12]).unwrap();
        assert!(fitted.predict(&x_bad).is_err());
        assert!(fitted.predict_proba(&x_bad).is_err());
    }

    #[test]
    fn test_gaussian_nb_single_class() {
        // Single class — still fits, always predicts that class.
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 1.5, 2.5, 0.8, 1.8]).unwrap();
        let y = array![0usize, 0, 0];

        let model = GaussianNB::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();
        assert_eq!(fitted.classes(), &[0]);
        let preds = fitted.predict(&x).unwrap();
        assert!(preds.iter().all(|&p| p == 0));
    }

    #[test]
    fn test_gaussian_nb_empty_data() {
        let x = Array2::<f64>::zeros((0, 2));
        let y = Array1::<usize>::zeros(0);
        let model = GaussianNB::<f64>::new();
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_gaussian_nb_default() {
        let model = GaussianNB::<f64>::default();
        assert_relative_eq!(model.var_smoothing, 1e-9, epsilon = 1e-15);
    }

    #[test]
    fn test_gaussian_nb_pipeline() {
        let x = Array2::from_shape_vec((6, 1), vec![-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]).unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
        let model = GaussianNB::<f64>::new();
        let fitted = model.fit_pipeline(&x, &y).unwrap();
        let preds = fitted.predict_pipeline(&x).unwrap();
        assert_eq!(preds.len(), 6);
    }

    #[test]
    fn test_gaussian_nb_unordered_classes() {
        // Classes are not 0..n, and not in order in y.
        let x = Array2::from_shape_vec((6, 1), vec![1.0, 1.2, 1.1, 5.0, 4.9, 5.1]).unwrap();
        let y = array![3usize, 3, 3, 7, 7, 7];
        let model = GaussianNB::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();
        assert_eq!(fitted.classes(), &[3, 7]);
        let preds = fitted.predict(&x).unwrap();
        assert!(preds[0] == 3 || preds[0] == 7);
        assert_eq!(preds[0], 3);
        assert_eq!(preds[3], 7);
    }

    #[test]
    fn test_gaussian_nb_partial_fit() {
        // Fit on first batch, then partial_fit on second batch.
        let x1 =
            Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 1.2, 0.8, 5.0, 5.0, 5.1, 4.9]).unwrap();
        let y1 = array![0usize, 0, 1, 1];

        let model = GaussianNB::<f64>::new();
        let mut fitted = model.fit(&x1, &y1).unwrap();

        let x2 =
            Array2::from_shape_vec((4, 2), vec![0.9, 1.1, 1.1, 0.9, 4.8, 5.2, 5.0, 4.8]).unwrap();
        let y2 = array![0usize, 0, 1, 1];

        fitted.partial_fit(&x2, &y2).unwrap();

        // Should still classify correctly after partial_fit.
        let x_test = Array2::from_shape_vec((2, 2), vec![1.0, 1.0, 5.0, 5.0]).unwrap();
        let preds = fitted.predict(&x_test).unwrap();
        assert_eq!(preds[0], 0);
        assert_eq!(preds[1], 1);
    }

    #[test]
    fn test_gaussian_nb_partial_fit_shape_mismatch() {
        let (x, y) = make_2class_data();
        let model = GaussianNB::<f64>::new();
        let mut fitted = model.fit(&x, &y).unwrap();

        // Wrong number of features.
        let x_bad = Array2::from_shape_vec((2, 3), vec![1.0; 6]).unwrap();
        let y_bad = array![0usize, 1];
        assert!(fitted.partial_fit(&x_bad, &y_bad).is_err());

        // Wrong y length.
        let x_ok = Array2::from_shape_vec((2, 2), vec![1.0; 4]).unwrap();
        let y_wrong = array![0usize];
        assert!(fitted.partial_fit(&x_ok, &y_wrong).is_err());
    }

    #[test]
    fn test_gaussian_nb_partial_fit_empty() {
        let (x, y) = make_2class_data();
        let model = GaussianNB::<f64>::new();
        let mut fitted = model.fit(&x, &y).unwrap();

        let x_empty = Array2::<f64>::zeros((0, 2));
        let y_empty = Array1::<usize>::zeros(0);
        // Should succeed without changes.
        assert!(fitted.partial_fit(&x_empty, &y_empty).is_ok());
    }

    #[test]
    fn test_gaussian_nb_class_prior() {
        let (x, y) = make_2class_data();
        let model = GaussianNB::<f64>::new().with_class_prior(vec![0.9, 0.1]);
        let fitted = model.fit(&x, &y).unwrap();

        // With strongly biased prior toward class 0, predictions should favor class 0.
        // Even class 1 samples might be classified as class 0.
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 8);
    }

    #[test]
    fn test_gaussian_nb_class_prior_wrong_length() {
        let (x, y) = make_2class_data();
        let model = GaussianNB::<f64>::new().with_class_prior(vec![0.5, 0.3, 0.2]);
        assert!(model.fit(&x, &y).is_err());
    }
}
