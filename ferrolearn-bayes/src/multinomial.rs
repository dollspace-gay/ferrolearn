//! Multinomial Naive Bayes classifier.
//!
//! This module provides [`MultinomialNB`], suitable for discrete count data
//! (e.g., word counts in text classification). Features must be non-negative.
//!
//! The log-likelihood for feature `j` in class `c` is:
//!
//! ```text
//! log theta_cj = log( (N_cj + alpha) / (N_c + alpha * n_features) )
//! ```
//!
//! where `N_cj` is the total count of feature `j` in class `c`, `N_c` is the
//! total count of all features in class `c`, and `alpha` is the Laplace (add-1)
//! smoothing parameter.
//!
//! # Examples
//!
//! ```
//! use ferrolearn_bayes::MultinomialNB;
//! use ferrolearn_core::{Fit, Predict};
//! use ndarray::{array, Array2};
//!
//! let x = Array2::from_shape_vec(
//!     (6, 3),
//!     vec![
//!         3.0, 1.0, 0.0,
//!         2.0, 0.0, 1.0,
//!         4.0, 2.0, 0.0,
//!         0.0, 1.0, 4.0,
//!         1.0, 0.0, 3.0,
//!         0.0, 2.0, 5.0,
//!     ],
//! ).unwrap();
//! let y = array![0usize, 0, 0, 1, 1, 1];
//!
//! let model = MultinomialNB::<f64>::new();
//! let fitted = model.fit(&x, &y).unwrap();
//! let preds = fitted.predict(&x).unwrap();
//! assert_eq!(preds.len(), 6);
//! ```
//!
//! # `## REQ status`
//!
//! Binary classification (R-DEFER-2): two states only — SHIPPED needs impl + a
//! non-test production consumer + green verification; NOT-STARTED carries the
//! open prereq blocker. The non-test production consumer is `_RsMultinomialNB`
//! / `RsMultinomialNB` (`ferrolearn-python/src/extras.rs`, built via the
//! `py_classifier!` macro), which exercises `new(alpha, fit_prior)` / `fit` /
//! `predict` against the library `FittedMultinomialNB` and is surfaced as
//! `ferrolearn.MultinomialNB` (`_extras.py`, `class MultinomialNB`); plus the
//! in-crate `impl PipelineEstimator for MultinomialNB` (`fit_pipeline` /
//! `predict_pipeline`). Green verification = the in-tree `multinomial` lib tests
//! plus the live-sklearn pins (`ferrolearn-bayes/tests/divergence_multinomial.rs`):
//! `divergence_multinomial_negative_alpha_rejected` (#904, now PASSING after the
//! `alpha < 0` reject landed), then the green guards
//! `green_multinomial_predict_proba_log_proba_value`,
//! `green_multinomial_class_prior_length_only_accepts_non_unit_sum`,
//! `green_multinomial_score_accuracy`,
//! `green_multinomial_negative_features_rejected`,
//! `green_multinomial_partial_fit_equals_fit` — all passing. Cites use symbol
//! anchors (ferrolearn) / `file:line` (sklearn 1.5.2, commit 156ef14). Live
//! oracle = installed sklearn 1.5.2.
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (`feature_log_prob_` smoothing + `_joint_log_likelihood` / `predict` / `predict_proba` / `predict_log_proba` / `predict_joint_log_proba` VALUE) | SHIPPED | `fn fit` for `MultinomialNB` sets `log_theta[[ci,j]] = ((count + alpha) / (total_count + alpha * n_features)).ln()`, the algebraic identity of `_update_feature_log_prob` (`naive_bayes.py:885-892`, `log(fc+alpha) - log((fc+alpha).sum(axis=1))`); `impl BaseNB::joint_log_likelihood` for `FittedMultinomialNB` computes `log_prior[ci] + sum_j x*log_theta[ci,j]`, mirroring `X @ feature_log_prob_.T + class_log_prior_` (`naive_bayes.py:894-896`); the four `predict_*` delegate to the `BaseNB` provided methods. Non-test consumer: `RsMultinomialNB::fit`/`predict` (`ferrolearn-python/src/extras.rs`, `py_classifier!`) → `FittedMultinomialNB`, surfaced as `ferrolearn.MultinomialNB`; plus `impl PipelineEstimator`. Verified: green guard `green_multinomial_predict_proba_log_proba_value` — on `X=[[3,1,0],[2,0,1],[4,2,0],[0,1,4],[1,0,3],[0,2,5]]`, `y=[0,0,0,1,1,1]`, `q=[[2,1,1],[0,1,3]]`, sklearn `predict_proba(q) = [[0.8843694464372913, 0.11563055356270838], [0.007188876743869827, 0.9928111232561301]]`, `predict_log_proba(q) = [[-0.12288037781713079, -2.1573550534903774], [-4.935220344228254, -0.007214841230117397]]`, `predict(q) = [0, 1]`; ferrolearn matches to ≤1e-12. In-tree `test_multinomial_nb_fit_predict` / `test_multinomial_nb_predict_proba_sums_to_one`. |
//! | REQ-2 (`alpha >= 0` validation) | SHIPPED | `fn fit` rejects `self.alpha < F::zero()` with `FerroError::InvalidParameter { name: "alpha", reason: "alpha must be >= 0 (sklearn Interval[0, inf))" }`, mirroring `MultinomialNB._parameter_constraints` `alpha: [Interval(Real, 0, None, closed="left"), "array-like"]` (`naive_bayes.py:530`) — the HARD `>= 0` reject `_validate_params` enforces at `fit` (distinct from `_check_alpha`'s `1e-10` floor, `naive_bayes.py:604-626`, which only fires under `force_alpha=false`; `alpha=0` stays allowed). Non-test consumer: `RsMultinomialNB::fit` maps the `FerroError` → `PyValueError`. Verified: green pin `divergence_multinomial_negative_alpha_rejected` (#904, now PASSING): `with_alpha(-0.5).fit(X,y)` returns `Err` (sklearn raises `InvalidParameterError`, "The 'alpha' parameter of MultinomialNB must be a float in the range [0.0, inf) … Got -0.5 instead."). |
//! | REQ-3 (`class_log_prior_` empirical / uniform VALUE) | SHIPPED | `fn fit` sets the empirical `log_prior[ci] = (n_c / n).ln()` (default) and the uniform `(1 / n_classes).ln()` (`fit_prior == false`), mirroring `class_log_prior_ = log(class_count_) - log(class_count_.sum())` (`naive_bayes.py:600`) and `np.full(n_classes, -np.log(n_classes))` (`naive_bayes.py:602`). Non-test consumer: `RsMultinomialNB::predict` → `fitted.predict` (the `class_log_prior_` term enters the jll additively). Verified: on the balanced fixture the empirical prior is `log(0.5)` for both classes; `predict(q) = [0, 1]` (green `green_multinomial_predict_proba_log_proba_value`). In-tree `test_multinomial_nb_fit_predict`. |
//! | REQ-4 (`class_prior` explicit + LENGTH-only validation — MATCH) | SHIPPED | `fn fit` validates ONLY `priors.len() != n_classes` then sets `log_prior[ci] = p.ln()`, mirroring `_update_class_log_prior` (`naive_bayes.py:589-591`, `if len != n_classes: ValueError; class_log_prior_ = np.log(class_prior)`) — discrete NB has NO sum-to-1 / non-negativity check (UNLIKE GaussianNB, which rejects a non-unit-sum prior). A deliberate MATCH. Non-test consumer: `RsMultinomialNB` builds `MultinomialNB`; the `with_class_prior` path is exercised in-crate + pipeline. Verified: green guard `green_multinomial_class_prior_length_only_accepts_non_unit_sum` — `with_class_prior([0.5, 0.3]).fit(X,y)` SUCCEEDS (sklearn `class_log_prior_ = log([0.5,0.3])`, inter-class gap `0.5108256237659908`), `with_class_prior([0.5]).fit` errors. In-tree `test_multinomial_nb_class_prior` / `test_multinomial_nb_class_prior_wrong_length`. (Wrong-length error TYPE differs — `InvalidParameter` vs `ValueError` — folded into REQ-9's surface gap.) |
//! | REQ-5 (negative-feature guard — both reject) | SHIPPED | `fn fit` rejects any `x[i,j] < F::zero()` with `FerroError::InvalidParameter { name: "X", reason: "MultinomialNB requires non-negative feature values" }`, mirroring `check_non_negative(X, "MultinomialNB (input X)")` → `ValueError` (`naive_bayes.py:881`). Both REJECT. Non-test consumer: `RsMultinomialNB::fit` maps the `FerroError` → `PyValueError`. Verified: green guard `green_multinomial_negative_features_rejected` — `fit(X_neg, y)` returns `Err` (sklearn raises `ValueError("Negative values in data passed to MultinomialNB (input X)")`). In-tree `test_multinomial_nb_negative_features_error`. The exact sklearn MESSAGE text is NOT matched — that message-parity sub-item is captured under REQ-9 (the binding maps to `PyValueError` so the Python-facing TYPE coincides; the text differs). |
//! | REQ-6 (`force_alpha` floor + `fit_prior` toggle) | SHIPPED | `fn fit` calls `crate::clamp_alpha(self.alpha, self.force_alpha)` (`base::check_alpha`, the `_check_alpha` floor `1e-10` unless `force_alpha`, `naive_bayes.py:604-626`) and selects empirical/uniform prior on `fit_prior`. Non-test consumer: `RsMultinomialNB` passes `fit_prior` through `with_fit_prior` and `alpha` through `with_alpha`. Verified: `alpha=0` (in `[0,inf)`, `force_alpha=true` default) is accepted — ferrolearn `clamp_alpha(0, true) = 0` matches; `score(X,y) = 1.0` (green `green_multinomial_score_accuracy`). In-tree `test_multinomial_nb_alpha_smoothing_effect` / `test_multinomial_nb_default`; `base.rs` `test_check_alpha_*`. |
//! | REQ-7 (`partial_fit` VALUE — same-classes path) | SHIPPED | `FittedMultinomialNB::partial_fit` accumulates `class_counts` / `feature_counts` for each existing class then recomputes `log_theta` / `log_prior` (same smoothing), mirroring the shared `_BaseDiscreteNB.partial_fit` accumulate-then-resmooth (`naive_bayes.py:628-709`, `_count` → `_update_feature_log_prob` → `_update_class_log_prior`). Non-test consumer: in-crate (the PyO3 `partial_fit` gap is REQ-9). Verified: green guard `green_multinomial_partial_fit_equals_fit` — two-chunk `partial_fit(X[:4],y[:4])` + `partial_fit(X[4:],y[4:])` reproduces the whole-`fit` `predict_proba` to ≤1e-12 (sklearn `np.allclose(feature_log_prob_) == True`). In-tree `test_multinomial_nb_partial_fit` / `test_multinomial_nb_partial_fit_shape_mismatch`. KNOWN GAP: `partial_fit` has NO `classes=` argument — `FittedMultinomialNB::partial_fit(&mut self, x, y)` loops only over the already-fitted `self.classes`, so a brand-new label is silently dropped (sklearn's `_BaseDiscreteNB.partial_fit` requires `classes=` at the first call, `naive_bayes.py:628-709`); this `classes=`/unseen-label path is NOT-STARTED (needs the multi-file surface change under #902) and is documented-not-pinned in the divergence header. |
//! | REQ-8 (`sample_weight`) | NOT-STARTED | open prereq blocker **#901**. sklearn `fit(X, y, sample_weight=None)` (`naive_bayes.py:712`) weights the binarized `Y` (`Y *= sample_weight.T`, `naive_bayes.py:751`) so `feature_count_ = Y.T @ X` / `class_count_ = Y.sum(axis=0)` become weighted (e.g. `fit(X,y,sample_weight=[1,2,1,1,1,3])` → `feature_count_ = [[11,3,2],[1,7,22]]`, `class_count_ = [4,5]`). ferrolearn's `impl Fit<Array2<F>, Array1<usize>>` has signature `fn fit(&self, x, y)` — NO `sample_weight` parameter on `fit` or `partial_fit`. |
//! | REQ-9 (fitted-attribute + PyO3 surface) | NOT-STARTED | open prereq blocker **#902**. sklearn exposes `feature_log_prob_` / `class_log_prior_` / `feature_count_` / `class_count_` / `classes_` / `n_features_in_` + the `_BaseDiscreteNB` `coef_ = feature_log_prob_[1:]` / `intercept_ = class_log_prior_[1:]` properties. `FittedMultinomialNB` stores `log_theta` / `log_prior` / `feature_counts` / `class_counts` as PRIVATE fields with no accessor — only `classes()` (via `HasClasses`) is public; no `coef_` / `intercept_`. `_RsMultinomialNB` (`ferrolearn-python/src/extras.rs`, the `py_classifier!` macro) exposes ONLY `new(alpha, fit_prior)` + `fit` + `predict` — no `class_prior` / `force_alpha` kwargs, no `predict_proba` / `predict_log_proba` / `predict_joint_log_proba` / `score` / `partial_fit` (which the library HAS), no fitted-attr getters. Also subsumes the negative-feature MESSAGE-parity sub-item (REQ-5) and the `partial_fit` `classes=` surface (REQ-7 gap). The fix belongs in `ferrolearn-python` (multi-file). |
//! | REQ-10 (ferray substrate) | NOT-STARTED | open prereq blocker **#903**. `multinomial.rs` imports `ndarray::{Array1, Array2}` + `num_traits::{Float, FromPrimitive, ToPrimitive}` (the wrong substrate, R-SUBSTRATE-1); not migrated to `ferray-core`. |

use crate::base::BaseNB;
use ferrolearn_core::error::FerroError;
use ferrolearn_core::introspection::HasClasses;
use ferrolearn_core::pipeline::{FittedPipelineEstimator, PipelineEstimator};
use ferrolearn_core::traits::{Fit, Predict};
use ndarray::{Array1, Array2};
use num_traits::{Float, FromPrimitive, ToPrimitive};

/// Multinomial Naive Bayes classifier.
///
/// Suitable for discrete count data. Features must be non-negative.
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
#[derive(Debug, Clone)]
pub struct MultinomialNB<F> {
    /// Additive (Laplace) smoothing parameter. Default: `1.0`.
    pub alpha: F,
    /// Optional user-supplied class priors. If set, these are used
    /// instead of computing priors from the data.
    pub class_prior: Option<Vec<F>>,
    /// Whether to learn class priors from the data. When `false` and
    /// `class_prior` is `None`, uniform priors `1 / n_classes` are used.
    /// Default: `true`.
    pub fit_prior: bool,
    /// When `false`, `alpha` values below `1e-10` are silently raised to
    /// `1e-10` (legacy behavior). When `true` (default), the user-supplied
    /// `alpha` is used as-is.
    pub force_alpha: bool,
}

impl<F: Float> MultinomialNB<F> {
    /// Create a new `MultinomialNB` with Laplace smoothing (`alpha = 1.0`).
    #[must_use]
    pub fn new() -> Self {
        Self {
            alpha: F::one(),
            class_prior: None,
            fit_prior: true,
            force_alpha: true,
        }
    }

    /// Set the Laplace smoothing parameter.
    #[must_use]
    pub fn with_alpha(mut self, alpha: F) -> Self {
        self.alpha = alpha;
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

    /// Toggle whether to learn class priors from data. Mirrors sklearn's
    /// `fit_prior`. When `false` and no `class_prior` is set, uniform priors
    /// are used.
    #[must_use]
    pub fn with_fit_prior(mut self, fit_prior: bool) -> Self {
        self.fit_prior = fit_prior;
        self
    }

    /// Toggle the `force_alpha` policy. See struct field doc.
    #[must_use]
    pub fn with_force_alpha(mut self, force_alpha: bool) -> Self {
        self.force_alpha = force_alpha;
        self
    }
}

impl<F: Float> Default for MultinomialNB<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted Multinomial Naive Bayes classifier.
#[derive(Debug, Clone)]
pub struct FittedMultinomialNB<F> {
    /// Sorted unique class labels.
    classes: Vec<usize>,
    /// Log prior probability for each class, shape `(n_classes,)`.
    log_prior: Array1<F>,
    /// Log feature probabilities per class, shape `(n_classes, n_features)`.
    log_theta: Array2<F>,
    /// Raw per-class feature count sums, shape `(n_classes, n_features)`.
    feature_counts: Array2<F>,
    /// Per-class sample counts.
    class_counts: Vec<usize>,
    /// Smoothing parameter carried forward for partial_fit (already
    /// post-clamp under `force_alpha=false`).
    alpha: F,
    /// Optional user-supplied class priors.
    class_prior: Option<Vec<F>>,
    /// Whether priors were fit from data (carried forward for partial_fit).
    fit_prior: bool,
}

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, Array1<usize>> for MultinomialNB<F> {
    type Fitted = FittedMultinomialNB<F>;
    type Error = FerroError;

    /// Fit the Multinomial NB model.
    ///
    /// # Errors
    ///
    /// - [`FerroError::ShapeMismatch`] if `x` and `y` have different numbers of rows.
    /// - [`FerroError::InsufficientSamples`] if there are no samples.
    /// - [`FerroError::InvalidParameter`] if any feature value is negative.
    fn fit(&self, x: &Array2<F>, y: &Array1<usize>) -> Result<FittedMultinomialNB<F>, FerroError> {
        let (n_samples, n_features) = x.dim();

        if n_samples == 0 {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "MultinomialNB requires at least one sample".into(),
            });
        }

        if n_samples != y.len() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_samples],
                actual: vec![y.len()],
                context: "y length must match number of samples in X".into(),
            });
        }

        // Validate non-negative features.
        if x.iter().any(|&v| v < F::zero()) {
            return Err(FerroError::InvalidParameter {
                name: "X".into(),
                reason: "MultinomialNB requires non-negative feature values".into(),
            });
        }

        // Reject alpha < 0 — sklearn `MultinomialNB._parameter_constraints`
        // declares `alpha: Interval(Real, 0, None, closed="left")`
        // (naive_bayes.py:530), a HARD `>= 0` reject enforced at `fit` by
        // `_validate_params` (distinct from `_check_alpha`'s 1e-10 floor, which
        // only fires under `force_alpha=false`). alpha=0 stays allowed.
        if self.alpha < F::zero() {
            return Err(FerroError::InvalidParameter {
                name: "alpha".into(),
                reason: "alpha must be >= 0 (sklearn Interval[0, inf))".into(),
            });
        }

        // Collect sorted unique classes.
        let mut classes: Vec<usize> = y.to_vec();
        classes.sort_unstable();
        classes.dedup();
        let n_classes = classes.len();

        let n_f = F::from(n_samples).unwrap();
        let n_feat_f = F::from(n_features).unwrap();
        let alpha = crate::clamp_alpha(self.alpha, self.force_alpha);

        let mut log_prior = Array1::<F>::zeros(n_classes);
        let mut log_theta = Array2::<F>::zeros((n_classes, n_features));

        let mut all_feature_counts = Array2::<F>::zeros((n_classes, n_features));
        let mut class_counts_vec = vec![0usize; n_classes];

        for (ci, &class_label) in classes.iter().enumerate() {
            let class_indices: Vec<usize> = y
                .iter()
                .enumerate()
                .filter_map(|(i, &label)| if label == class_label { Some(i) } else { None })
                .collect();

            let n_c = class_indices.len();
            let n_c_f = F::from(n_c).unwrap();
            // Empirical prior; overwritten below if fit_prior=false or
            // class_prior is set.
            log_prior[ci] = (n_c_f / n_f).ln();
            class_counts_vec[ci] = n_c;

            // Sum of feature counts for this class.
            for &i in &class_indices {
                for j in 0..n_features {
                    all_feature_counts[[ci, j]] = all_feature_counts[[ci, j]] + x[[i, j]];
                }
            }

            // Total count across all features for this class.
            let total_count = all_feature_counts.row(ci).sum();

            // Smoothed log probabilities.
            let denom = total_count + alpha * n_feat_f;
            for j in 0..n_features {
                log_theta[[ci, j]] = ((all_feature_counts[[ci, j]] + alpha) / denom).ln();
            }
        }

        // Resolve priors: explicit class_prior wins; else fit_prior chooses
        // between empirical (already filled) and uniform.
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
            for (ci, &p) in priors.iter().enumerate() {
                log_prior[ci] = p.ln();
            }
        } else if !self.fit_prior {
            let uniform = (F::one() / F::from(n_classes).unwrap()).ln();
            for ci in 0..n_classes {
                log_prior[ci] = uniform;
            }
        }

        Ok(FittedMultinomialNB {
            classes,
            log_prior,
            log_theta,
            feature_counts: all_feature_counts,
            class_counts: class_counts_vec,
            alpha,
            class_prior: self.class_prior.clone(),
            fit_prior: self.fit_prior,
        })
    }
}

impl<F: Float + Send + Sync + 'static> FittedMultinomialNB<F> {
    /// Incrementally update the model with new data.
    ///
    /// Accumulates feature counts and class counts, then recomputes
    /// the log probabilities.
    ///
    /// # Errors
    ///
    /// - [`FerroError::ShapeMismatch`] if `x` and `y` have different row counts
    ///   or the number of features does not match the fitted model.
    /// - [`FerroError::InvalidParameter`] if any feature value is negative.
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

        if n_features != self.log_theta.ncols() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![self.log_theta.ncols()],
                actual: vec![n_features],
                context: "number of features must match fitted MultinomialNB".into(),
            });
        }

        if x.iter().any(|&v| v < F::zero()) {
            return Err(FerroError::InvalidParameter {
                name: "X".into(),
                reason: "MultinomialNB requires non-negative feature values".into(),
            });
        }

        let n_feat_f = F::from(n_features).unwrap();

        // Accumulate counts for each existing class.
        for (ci, &class_label) in self.classes.clone().iter().enumerate() {
            let new_indices: Vec<usize> = y
                .iter()
                .enumerate()
                .filter_map(|(i, &label)| if label == class_label { Some(i) } else { None })
                .collect();

            if new_indices.is_empty() {
                continue;
            }

            self.class_counts[ci] += new_indices.len();

            for &i in &new_indices {
                for j in 0..n_features {
                    self.feature_counts[[ci, j]] = self.feature_counts[[ci, j]] + x[[i, j]];
                }
            }
        }

        // Recompute log_theta from accumulated feature_counts.
        let n_classes = self.classes.len();
        for ci in 0..n_classes {
            let total_count = self.feature_counts.row(ci).sum();
            let denom = total_count + self.alpha * n_feat_f;
            for j in 0..n_features {
                self.log_theta[[ci, j]] =
                    ((self.feature_counts[[ci, j]] + self.alpha) / denom).ln();
            }
        }

        // Recompute log priors. Explicit class_prior is sticky; otherwise
        // honor fit_prior.
        if self.class_prior.is_none() {
            if self.fit_prior {
                let total: usize = self.class_counts.iter().sum();
                let total_f = F::from(total).unwrap();
                for (ci, &count) in self.class_counts.iter().enumerate() {
                    self.log_prior[ci] = (F::from(count).unwrap() / total_f).ln();
                }
            } else {
                let uniform = (F::one() / F::from(n_classes).unwrap()).ln();
                for ci in 0..n_classes {
                    self.log_prior[ci] = uniform;
                }
            }
        }

        Ok(())
    }

    /// Predict class probabilities for the given feature matrix.
    ///
    /// Returns shape `(n_samples, n_classes)` where each row sums to 1.
    /// Delegates to [`BaseNB::nb_predict_proba`].
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
    /// `MultinomialNB._joint_log_likelihood`. Delegates to
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
    /// Returns shape `(n_samples, n_classes)`. Delegates to
    /// [`BaseNB::nb_predict_log_proba`].
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
    /// Equivalent to sklearn's `ClassifierMixin.score`.
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

impl<F: Float + Send + Sync + 'static> BaseNB<F> for FittedMultinomialNB<F> {
    /// Compute joint log-likelihood for each class — sklearn
    /// `MultinomialNB._joint_log_likelihood`.
    ///
    /// Returns shape `(n_samples, n_classes)`.
    fn joint_log_likelihood(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let n_features_fitted = self.log_theta.ncols();
        if x.ncols() != n_features_fitted {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_features_fitted],
                actual: vec![x.ncols()],
                context: "number of features must match fitted MultinomialNB".into(),
            });
        }

        let n_samples = x.nrows();
        let n_classes = self.classes.len();
        let n_features = x.ncols();

        let mut scores = Array2::<F>::zeros((n_samples, n_classes));

        for i in 0..n_samples {
            for ci in 0..n_classes {
                let mut score = self.log_prior[ci];
                for j in 0..n_features {
                    score = score + x[[i, j]] * self.log_theta[[ci, j]];
                }
                scores[[i, ci]] = score;
            }
        }

        Ok(scores)
    }

    fn nb_classes(&self) -> &[usize] {
        &self.classes
    }
}

impl<F: Float + Send + Sync + 'static> Predict<Array2<F>> for FittedMultinomialNB<F> {
    type Output = Array1<usize>;
    type Error = FerroError;

    /// Predict class labels for the given feature matrix.
    ///
    /// Delegates to [`BaseNB::nb_predict`].
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of features does
    /// not match the fitted model.
    fn predict(&self, x: &Array2<F>) -> Result<Array1<usize>, FerroError> {
        BaseNB::nb_predict(self, x)
    }
}

impl<F: Float + Send + Sync + 'static> HasClasses for FittedMultinomialNB<F> {
    fn classes(&self) -> &[usize] {
        &self.classes
    }

    fn n_classes(&self) -> usize {
        self.classes.len()
    }
}

// Pipeline integration.
impl<F: Float + ToPrimitive + FromPrimitive + Send + Sync + 'static> PipelineEstimator<F>
    for MultinomialNB<F>
{
    fn fit_pipeline(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
    ) -> Result<Box<dyn FittedPipelineEstimator<F>>, FerroError> {
        let y_usize: Array1<usize> = y.mapv(|v| v.to_usize().unwrap_or(0));
        let fitted = self.fit(x, &y_usize)?;
        Ok(Box::new(FittedMultinomialNBPipeline(fitted)))
    }
}

struct FittedMultinomialNBPipeline<F: Float + Send + Sync + 'static>(FittedMultinomialNB<F>);

unsafe impl<F: Float + Send + Sync + 'static> Send for FittedMultinomialNBPipeline<F> {}
unsafe impl<F: Float + Send + Sync + 'static> Sync for FittedMultinomialNBPipeline<F> {}

impl<F: Float + ToPrimitive + FromPrimitive + Send + Sync + 'static> FittedPipelineEstimator<F>
    for FittedMultinomialNBPipeline<F>
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

    fn make_count_data() -> (Array2<f64>, Array1<usize>) {
        // Simple word-count like data: two classes
        // class 0 has many feature 0, class 1 has many feature 2.
        let x = Array2::from_shape_vec(
            (6, 3),
            vec![
                5.0, 1.0, 0.0, 4.0, 2.0, 0.0, 6.0, 0.0, 1.0, 0.0, 1.0, 5.0, 1.0, 0.0, 4.0, 0.0,
                2.0, 6.0,
            ],
        )
        .unwrap();
        let y = array![0usize, 0, 0, 1, 1, 1];
        (x, y)
    }

    #[test]
    fn test_multinomial_nb_fit_predict() {
        let (x, y) = make_count_data();
        let model = MultinomialNB::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        let correct = preds.iter().zip(y.iter()).filter(|(p, a)| p == a).count();
        assert_eq!(correct, 6);
    }

    #[test]
    fn test_multinomial_nb_predict_proba_sums_to_one() {
        let (x, y) = make_count_data();
        let model = MultinomialNB::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();
        let proba = fitted.predict_proba(&x).unwrap();
        for i in 0..proba.nrows() {
            assert_relative_eq!(proba.row(i).sum(), 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_multinomial_nb_has_classes() {
        let (x, y) = make_count_data();
        let model = MultinomialNB::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();
        assert_eq!(fitted.classes(), &[0, 1]);
        assert_eq!(fitted.n_classes(), 2);
    }

    #[test]
    fn test_multinomial_nb_alpha_smoothing_effect() {
        let (x, y) = make_count_data();

        // With alpha = 0, very sharp probabilities.
        let model_sharp = MultinomialNB::<f64>::new().with_alpha(0.0);
        let fitted_sharp = model_sharp.fit(&x, &y).unwrap();
        let proba_sharp = fitted_sharp.predict_proba(&x).unwrap();

        // With alpha = 100, very smoothed probabilities (closer to uniform).
        let model_smooth = MultinomialNB::<f64>::new().with_alpha(100.0);
        let fitted_smooth = model_smooth.fit(&x, &y).unwrap();
        let proba_smooth = fitted_smooth.predict_proba(&x).unwrap();

        // Smoothed probabilities for class 0 on class-0 samples should be less extreme.
        // i.e., max probability should be lower with high alpha.
        assert!(proba_smooth[[0, 0]] < proba_sharp[[0, 0]]);
    }

    #[test]
    fn test_multinomial_nb_negative_features_error() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, -1.0, 3.0, 2.0, 1.0, 0.0, 4.0]).unwrap();
        let y = array![0usize, 0, 1, 1];
        let model = MultinomialNB::<f64>::new();
        let result = model.fit(&x, &y);
        assert!(result.is_err());
        match result.unwrap_err() {
            FerroError::InvalidParameter { name, .. } => assert_eq!(name, "X"),
            e => panic!("expected InvalidParameter, got {e:?}"),
        }
    }

    #[test]
    fn test_multinomial_nb_shape_mismatch_fit() {
        let x = Array2::from_shape_vec((4, 3), vec![1.0; 12]).unwrap();
        let y = array![0usize, 1]; // Wrong length
        let model = MultinomialNB::<f64>::new();
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_multinomial_nb_shape_mismatch_predict() {
        let (x, y) = make_count_data();
        let model = MultinomialNB::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();
        let x_bad = Array2::from_shape_vec((3, 5), vec![1.0; 15]).unwrap();
        assert!(fitted.predict(&x_bad).is_err());
        assert!(fitted.predict_proba(&x_bad).is_err());
    }

    #[test]
    fn test_multinomial_nb_single_class() {
        let x = Array2::from_shape_vec((3, 3), vec![1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0, 0.0, 1.0])
            .unwrap();
        let y = array![2usize, 2, 2];
        let model = MultinomialNB::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();
        assert_eq!(fitted.classes(), &[2]);
        let preds = fitted.predict(&x).unwrap();
        assert!(preds.iter().all(|&p| p == 2));
    }

    #[test]
    fn test_multinomial_nb_empty_data() {
        let x = Array2::<f64>::zeros((0, 3));
        let y = Array1::<usize>::zeros(0);
        let model = MultinomialNB::<f64>::new();
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_multinomial_nb_default() {
        let model = MultinomialNB::<f64>::default();
        assert_relative_eq!(model.alpha, 1.0, epsilon = 1e-15);
    }

    #[test]
    fn test_multinomial_nb_partial_fit() {
        let x1 = Array2::from_shape_vec(
            (4, 3),
            vec![5.0, 1.0, 0.0, 4.0, 2.0, 0.0, 0.0, 1.0, 5.0, 1.0, 0.0, 4.0],
        )
        .unwrap();
        let y1 = array![0usize, 0, 1, 1];

        let model = MultinomialNB::<f64>::new();
        let mut fitted = model.fit(&x1, &y1).unwrap();

        let x2 = Array2::from_shape_vec((2, 3), vec![6.0, 0.0, 1.0, 0.0, 2.0, 6.0]).unwrap();
        let y2 = array![0usize, 1];

        fitted.partial_fit(&x2, &y2).unwrap();

        // Should still predict correctly on training data.
        let preds = fitted.predict(&x1).unwrap();
        let correct = preds.iter().zip(y1.iter()).filter(|(p, a)| p == a).count();
        assert!(correct >= 3);
    }

    #[test]
    fn test_multinomial_nb_partial_fit_shape_mismatch() {
        let (x, y) = make_count_data();
        let model = MultinomialNB::<f64>::new();
        let mut fitted = model.fit(&x, &y).unwrap();

        let x_bad = Array2::from_shape_vec((2, 5), vec![1.0; 10]).unwrap();
        let y_bad = array![0usize, 1];
        assert!(fitted.partial_fit(&x_bad, &y_bad).is_err());
    }

    #[test]
    fn test_multinomial_nb_class_prior() {
        let (x, y) = make_count_data();
        let model = MultinomialNB::<f64>::new().with_class_prior(vec![0.9, 0.1]);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 6);
    }

    #[test]
    fn test_multinomial_nb_class_prior_wrong_length() {
        let (x, y) = make_count_data();
        let model = MultinomialNB::<f64>::new().with_class_prior(vec![0.5]);
        assert!(model.fit(&x, &y).is_err());
    }
}
