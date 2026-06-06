//! Bernoulli Naive Bayes classifier.
//!
//! This module provides [`BernoulliNB`], suitable for binary/boolean feature
//! data. An optional binarization threshold can be used to convert continuous
//! features to binary values before fitting and prediction.
//!
//! The log-likelihood for feature `j` in class `c` is:
//!
//! ```text
//! log P(x_j | c) = x_j * log(p_cj) + (1 - x_j) * log(1 - p_cj)
//! ```
//!
//! where `p_cj = (N_cj + alpha) / (N_c + 2 * alpha)` is the smoothed
//! probability that feature `j` is present in class `c`.
//!
//! # Examples
//!
//! ```
//! use ferrolearn_bayes::BernoulliNB;
//! use ferrolearn_core::{Fit, Predict};
//! use ndarray::{array, Array2};
//!
//! let x = Array2::from_shape_vec(
//!     (6, 3),
//!     vec![
//!         1.0, 1.0, 0.0,
//!         1.0, 0.0, 0.0,
//!         1.0, 1.0, 0.0,
//!         0.0, 0.0, 1.0,
//!         0.0, 1.0, 1.0,
//!         0.0, 0.0, 1.0,
//!     ],
//! ).unwrap();
//! let y = array![0usize, 0, 0, 1, 1, 1];
//!
//! let model = BernoulliNB::<f64>::new();
//! let fitted = model.fit(&x, &y).unwrap();
//! let preds = fitted.predict(&x).unwrap();
//! assert_eq!(preds.len(), 6);
//! ```
//!
//! # `## REQ status`
//!
//! Binary classification (R-DEFER-2): two states only — SHIPPED needs impl + a
//! non-test production consumer + green verification; NOT-STARTED carries the
//! open prereq blocker. The non-test production consumer is `_RsBernoulliNB` /
//! `RsBernoulliNB` (`ferrolearn-python/src/extras.rs`, built via the
//! `py_classifier!` macro), which exercises `new(alpha, fit_prior, binarize)` /
//! `fit` / `predict` against the library `FittedBernoulliNB` and is surfaced as
//! `ferrolearn.BernoulliNB`; plus the in-crate `impl PipelineEstimator for
//! BernoulliNB` (`fit_pipeline` / `predict_pipeline`). The pipeline adapter
//! preserves the ORIGINAL labels: `fit_pipeline` sets `classes_ = np.unique(y)`
//! (sorted unique original float labels, via `label_binarize`) and
//! `predict_pipeline` returns those original labels (`classes_[argmax(jll)]`,
//! `naive_bayes.py:103`), not `0..n_classes` indices. Green verification = the
//! in-tree `bernoulli` lib tests plus the live-sklearn pins / guards
//! (`ferrolearn-bayes/tests/divergence_bernoulli.rs`):
//! `divergence_bernoulli_binarize_default_is_zero` (#911, now PASSING after
//! `new()` defaults `binarize = Some(0.0)`) and
//! `divergence_bernoulli_negative_alpha_rejected` (#912, now PASSING after the
//! `alpha < 0` reject landed), then the green guards
//! `green_bernoulli_value_on_binary_data`,
//! `green_bernoulli_with_binarize_threshold_value`,
//! `green_bernoulli_class_prior_length_only`,
//! `green_bernoulli_score_accuracy` — all passing. Cites use symbol anchors
//! (ferrolearn) / `file:line` (sklearn 1.5.2, commit 156ef14). Live oracle =
//! installed sklearn 1.5.2. (Note: this table follows the in-tree pin numbering
//! and REQ grouping; the design doc `.design/bayes/bernoulli.md` uses a wider
//! REQ split — REQ-3/#906 binarize-default and REQ-6/#907 alpha-reject there
//! correspond to the now-green #911 / #912 pins here.)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (`feature_log_prob_` smoothing + Bernoulli `_joint_log_likelihood` / `predict` / `predict_proba` / `predict_log_proba` / `predict_joint_log_proba` VALUE) | SHIPPED | `fn fit` for `BernoulliNB` sets `p = (fc + alpha) / (n_c + 2*alpha)`, `log_prob[[ci,j]] = p.ln()`, `log_neg_prob[[ci,j]] = (1-p).ln()` — the algebraic identity of `_update_feature_log_prob` (`naive_bayes.py:1194-1201`, `log(fc+alpha) - log(cc+alpha*2)`); `impl BaseNB::joint_log_likelihood` for `FittedBernoulliNB` computes `log_prior[ci] + sum_j [x*log_prob + (1-x)*log_neg_prob]`, the row-wise form of `X @ (flp-neg).T + class_log_prior_ + neg.sum(axis=1)` (`naive_bayes.py:1203-1219`); the four `predict_*` delegate to the `BaseNB` provided methods. Non-test consumer: `RsBernoulliNB::fit`/`predict` (`ferrolearn-python/src/extras.rs`, `py_classifier!`) → `FittedBernoulliNB`, surfaced as `ferrolearn.BernoulliNB`; plus `impl PipelineEstimator`. Verified: green guard `green_bernoulli_value_on_binary_data` — on `Xbin=[[1,1,0],[1,0,0],[1,1,0],[0,0,1],[0,1,1],[0,0,1]]`, `y=[0,0,0,1,1,1]`, `q=[[1,0,0],[0,1,1]]` (where sklearn `binarize=0.0` ≡ ferro path, `np.allclose==True`), sklearn `predict_proba(q) = [[0.9142857142857143, 0.08571428571428572], [0.08571428571428567, 0.9142857142857145]]`, `predict_log_proba(q) = [[-0.0896121586896872, -2.456735772821304], [-2.4567357728213044, -0.08961215868968697]]`, `predict(q) = [0, 1]`; ferrolearn matches to ≤1e-12. In-tree `test_bernoulli_nb_fit_predict` / `test_bernoulli_nb_predict_proba_sums_to_one` / `test_bernoulli_nb_predict_proba_ordering`. |
//! | REQ-2 (`binarize` DEFAULT `0.0` + strictly-greater threshold-application VALUE) | SHIPPED | `pub fn new` sets `binarize: Some(F::zero())`, mirroring `BernoulliNB.__init__(..., binarize=0.0, ...)` (`naive_bayes.py:1164`), and `fn binarize_array` is `x.mapv(\|v\| if v > threshold { 1 } else { 0 })` — strictly-greater — applied in `fit`/`partial_fit`/`joint_log_likelihood` when `binarize` is `Some`, mirroring `binarize(X, threshold=self.binarize)` (strictly `X > threshold`) invoked by `_check_X_y`/`_check_X` only when `binarize is not None` (`naive_bayes.py:1176-1187`). Non-test consumer: `RsBernoulliNB` builds `with_binarize(binarize)` (default `0.0`); the threshold path feeds `fit`/`predict`. Verified: green pin `divergence_bernoulli_binarize_default_is_zero` (#911, now PASSING) — on NON-binary `Xc=[[2,0,1],[0,3,0],[1,1,2],[0,0,4]]`, `yc=[0,0,1,1]`, `BernoulliNB::new().predict(Xc) = [1,0,1,1]` and `predict_proba(Xc)[1] = [0.6666666666666669, 0.3333333333333332]` (sklearn default-binarize values); plus green guard `green_bernoulli_with_binarize_threshold_value` — `with_binarize(0.5)` on continuous data matches sklearn `BernoulliNB(binarize=0.5)` `predict_proba`/`predict` to ≤1e-12 (`0.5→0`, strictly-greater). In-tree `test_bernoulli_nb_default` / `test_bernoulli_nb_binarize_threshold` / `test_bernoulli_nb_binarize_zero_threshold`. |
//! | REQ-3 (`alpha >= 0` validation) | SHIPPED | `fn fit` rejects `self.alpha < F::zero()` with `FerroError::InvalidParameter { name: "alpha", reason: "alpha must be >= 0 (sklearn Interval[0, inf))" }`, mirroring the shared `_BaseDiscreteNB._parameter_constraints` `alpha: [Interval(Real, 0, None, closed="left"), "array-like"]` (`naive_bayes.py:530`) merged into `BernoulliNB._parameter_constraints` (`naive_bayes.py:1154-1157`) — the HARD `>= 0` reject `_validate_params` enforces at `fit` (distinct from `_check_alpha`'s `1e-10` floor, `naive_bayes.py:604-626`, which only fires under `force_alpha=false`; `alpha=0` stays allowed). Non-test consumer: `RsBernoulliNB::fit` maps the `FerroError` → `PyValueError`. Verified: green pin `divergence_bernoulli_negative_alpha_rejected` (#912, now PASSING): `with_alpha(-0.5).fit(Xc,yc)` returns `Err` (sklearn raises `InvalidParameterError`, "The 'alpha' parameter of BernoulliNB must be a float in the range [0.0, inf) or an array-like. Got -0.5 instead."). |
//! | REQ-4 (`class_log_prior_` empirical/uniform/explicit + LENGTH-only validation — MATCH) | SHIPPED | `fn fit` sets the empirical `log_prior[ci] = (n_c / n).ln()` (default), the uniform `(1 / n_classes).ln()` (`fit_prior == false`), and the explicit `log_prior[ci] = p.ln()` after validating ONLY `priors.len() != n_classes`, mirroring `_update_class_log_prior` (`naive_bayes.py:580-602`: `log(class_count_)-log(class_count_.sum())` `:600`, `-log(n_classes)` `:602`, `log(class_prior)` after length-only check `:589-591`) — discrete NB has NO sum-to-1 / non-negativity check (UNLIKE GaussianNB). A deliberate MATCH. Non-test consumer: `RsBernoulliNB::predict` → `fitted.predict` (the `class_log_prior_` term enters the jll additively); `with_fit_prior` passes `fit_prior` through. Verified: green guard `green_bernoulli_class_prior_length_only` — `with_class_prior([0.5,0.3]).fit(Xbin,y)` SUCCEEDS (sum 0.8; sklearn `class_log_prior_ = log([0.5,0.3])`, inter-class gap `0.5108256237659908`), `with_class_prior([0.5]).fit` errors. In-tree `test_bernoulli_nb_class_prior` / `test_bernoulli_nb_class_prior_wrong_length`. (Wrong-length error TYPE differs — `InvalidParameter` vs `ValueError` — folded into REQ-9's surface gap.) |
//! | REQ-5 (`force_alpha` floor + `fit_prior` toggle) | SHIPPED | `fn fit` calls `crate::clamp_alpha(self.alpha, self.force_alpha)` (`base::check_alpha`, the `_check_alpha` floor `1e-10` unless `force_alpha`, `naive_bayes.py:604-626`) and selects empirical/uniform prior on `fit_prior`. Non-test consumer: `RsBernoulliNB` passes `fit_prior` through `with_fit_prior`; `alpha` through `with_alpha`. Verified: with `force_alpha=true` default and `alpha=1`, `score(Xbin,y) = 1.0` (green `green_bernoulli_score_accuracy`); `clamp_alpha(1, true) = 1`. In-tree `test_bernoulli_nb_default`; `base.rs` `test_check_alpha_*`. |
//! | REQ-6 (`score` mean accuracy) | SHIPPED | `FittedBernoulliNB::score` computes `correct / n` over `predict`, the `ClassifierMixin.score` analog. Non-test consumer: in-crate + the library surface (the PyO3 `score` exposure gap is REQ-9). Verified: green guard `green_bernoulli_score_accuracy` — `BernoulliNB::new().fit(Xbin,y).score(Xbin,y) = 1.0` (sklearn `BernoulliNB().fit(Xbin,y).score(Xbin,y) == 1.0`). |
//! | REQ-7 (`partial_fit` VALUE — same-classes path) | SHIPPED | `FittedBernoulliNB::partial_fit` binarizes when `binarize` is `Some`, accumulates `class_counts`/`feature_counts` for each EXISTING class, then recomputes `log_prob`/`log_neg_prob` (same `p=(fc+alpha)/(n_c+2*alpha)` smoothing) and `log_prior`, mirroring the shared `_BaseDiscreteNB.partial_fit` accumulate-then-resmooth (`naive_bayes.py:629-708`, `_count` → `_update_feature_log_prob` → `_update_class_log_prior`). Non-test consumer: in-crate (the PyO3 `partial_fit` gap is REQ-9). Verified: in-tree `test_bernoulli_nb_partial_fit` / `test_bernoulli_nb_partial_fit_shape_mismatch` — chunked `partial_fit` over already-fitted classes reproduces the accumulate-then-resmooth path (sklearn `partial_fit` over chunks == `fit` on the whole). KNOWN GAP: `partial_fit` has NO `classes=` argument — `FittedBernoulliNB::partial_fit(&mut self, x, y)` loops only over the already-fitted `self.classes`, so a brand-new label is silently dropped (sklearn's `_BaseDiscreteNB.partial_fit` binarizes against the full `classes=` list from the first call, `naive_bayes.py:629-708`); this `classes=`/unseen-label path is NOT-STARTED (folded into #908) and is documented-not-pinned in the divergence header. |
//! | REQ-8 (`sample_weight` + `partial_fit` `classes=`) | NOT-STARTED | open prereq blocker **#908**. sklearn `fit(X, y, sample_weight=None)` (`naive_bayes.py:712`) weights the binarized `Y` so `feature_count_ = Y.T @ X` / `class_count_ = Y.sum(axis=0)` become weighted (`naive_bayes.py:1189-1192`) — e.g. `BernoulliNB().fit(Xbin,y,sample_weight=[1,2,1,1,1,3]).feature_count_ = [[4,2,0],[0,1,5]]`, `class_count_ = [4,5]`. ferrolearn's `impl Fit<Array2<F>, Array1<usize>>` has signature `fn fit(&self, x, y)` — NO `sample_weight` parameter on `fit` or `partial_fit`; also no `classes=` argument on `partial_fit` (the unseen-label sub-gap of REQ-7). |
//! | REQ-9a (Rust fitted-attribute accessors) | SHIPPED | `FittedBernoulliNB` exposes `feature_log_prob(&self) -> &Array2<F>` (`&self.log_prob`, sklearn `feature_log_prob_`, `naive_bayes.py:1201`), `class_log_prior(&self) -> &Array1<F>` (`&self.log_prior`, sklearn `class_log_prior_`, `naive_bayes.py:600`), `feature_count(&self) -> &Array2<F>` (`&self.feature_counts`, sklearn `feature_count_`, `naive_bayes.py:1189`), and `class_count(&self) -> Array1<F>` (the integer `class_counts` cast to `F`, sklearn `class_count_`, `naive_bayes.py:1190`). `coef_` / `intercept_` are DEPRECATED and REMOVED in sklearn 1.5.2 (`hasattr(BernoulliNB().fit(...), 'coef_') == False`), so no `coef_` / `intercept_` getter is added. Live oracle (`X=[[1,1,0],[1,0,0],[1,1,0],[0,0,1],[0,1,1],[0,0,1]]`, `y=[0,0,0,1,1,1]`): `feature_log_prob_ = [[-0.2231435513,-0.5108256238,-1.6094379124],[-1.6094379124,-0.9162907319,-0.2231435513]]`, `class_log_prior_ = [-0.6931471806,-0.6931471806]`, `feature_count_ = [[3,2,0],[0,1,3]]`, `class_count_ = [3,3]`. In-tree `bernoulli_feature_log_prob_and_class_log_prior_match_sklearn` / `bernoulli_feature_count_and_class_count_match_sklearn`. |
//! | REQ-9b (PyO3 surface + `sample_weight`) | NOT-STARTED | open prereq blocker **#909**. `_RsBernoulliNB` (`ferrolearn-python/src/extras.rs`, the `py_classifier!` macro) exposes ONLY `new(alpha, fit_prior, binarize)` + `fit` + `predict` — NO `class_prior`/`force_alpha` kwargs, NO `predict_proba`/`predict_log_proba`/`predict_joint_log_proba`/`score`/`partial_fit` (which the library HAS), NO fitted-attr getters bridged to Python (`feature_log_prob_` / `class_log_prior_` / `feature_count_` / `class_count_` / `classes_` / `n_features_in_`). `coef_` / `intercept_` are deprecated/removed in sklearn 1.5.2 (`hasattr == False`) and stay absent. Also subsumes the `class_prior` wrong-length MESSAGE/TYPE-parity sub-item (REQ-4: `InvalidParameter` vs `ValueError`) and the `partial_fit` `classes=` surface (REQ-7 gap). The fix belongs in `ferrolearn-python` (multi-file). |
//! | REQ-10 (ferray substrate) | NOT-STARTED | open prereq blocker **#910**. `bernoulli.rs` imports `ndarray::{Array1, Array2}` + `num_traits::{Float, FromPrimitive, ToPrimitive}` (the wrong substrate, R-SUBSTRATE-1); not migrated to `ferray-core`. |

use crate::base::BaseNB;
use ferrolearn_core::error::FerroError;
use ferrolearn_core::introspection::HasClasses;
use ferrolearn_core::pipeline::{FittedPipelineEstimator, PipelineEstimator};
use ferrolearn_core::traits::{Fit, Predict};
use ndarray::{Array1, Array2};
use num_traits::{Float, FromPrimitive, ToPrimitive};

/// Bernoulli Naive Bayes classifier.
///
/// Suitable for binary feature data. Features can be binarized automatically
/// by setting `binarize` to a threshold value.
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
#[derive(Debug, Clone)]
pub struct BernoulliNB<F> {
    /// Additive (Laplace) smoothing parameter. Default: `1.0`.
    pub alpha: F,
    /// Optional threshold for binarizing features. Values strictly above this
    /// threshold are set to 1; others to 0. If `None`, features are used as-is.
    /// Default: `Some(0.0)` (binarize at 0), mirroring scikit-learn's
    /// `BernoulliNB(binarize=0.0)` (`sklearn/naive_bayes.py:1164`).
    pub binarize: Option<F>,
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

impl<F: Float> BernoulliNB<F> {
    /// Create a new `BernoulliNB` with default settings.
    ///
    /// Default: `alpha = 1.0`, `binarize = Some(0.0)`, `class_prior = None`,
    /// `fit_prior = true`, `force_alpha = true`. The `binarize = Some(0.0)`
    /// default mirrors scikit-learn's `BernoulliNB(binarize=0.0)`
    /// (`sklearn/naive_bayes.py:1164`): by default `X` is binarized at threshold
    /// `0.0` (values `> 0` become `1`) on every fit/predict. Set the field to
    /// `None` (e.g. via direct assignment) to disable binarization, matching
    /// sklearn's explicit `binarize=None`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            alpha: F::one(),
            binarize: Some(F::zero()),
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

    /// Set the binarization threshold.
    ///
    /// Values strictly above this threshold become 1; all others become 0.
    #[must_use]
    pub fn with_binarize(mut self, threshold: F) -> Self {
        self.binarize = Some(threshold);
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

impl<F: Float> Default for BernoulliNB<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Binarize an array using the given threshold.
///
/// Values strictly above `threshold` become `F::one()`; others become `F::zero()`.
fn binarize_array<F: Float>(x: &Array2<F>, threshold: F) -> Array2<F> {
    x.mapv(|v| if v > threshold { F::one() } else { F::zero() })
}

/// Fitted Bernoulli Naive Bayes classifier.
#[derive(Debug, Clone)]
pub struct FittedBernoulliNB<F> {
    /// Sorted unique class labels.
    classes: Vec<usize>,
    /// Log prior probability for each class, shape `(n_classes,)`.
    log_prior: Array1<F>,
    /// Log feature-present probability per class, shape `(n_classes, n_features)`.
    log_prob: Array2<F>,
    /// Log complement (1 - p) per class, shape `(n_classes, n_features)`.
    log_neg_prob: Array2<F>,
    /// Binarization threshold (carried forward for prediction).
    binarize: Option<F>,
    /// Raw per-class feature occurrence counts, shape `(n_classes, n_features)`.
    feature_counts: Array2<F>,
    /// Per-class sample counts.
    class_counts: Vec<usize>,
    /// Smoothing parameter carried forward for partial_fit (post-clamp
    /// when `force_alpha=false`).
    alpha: F,
    /// Optional user-supplied class priors.
    class_prior: Option<Vec<F>>,
    /// Whether priors were fit from data (carried forward for partial_fit).
    fit_prior: bool,
}

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, Array1<usize>> for BernoulliNB<F> {
    type Fitted = FittedBernoulliNB<F>;
    type Error = FerroError;

    /// Fit the Bernoulli NB model.
    ///
    /// # Errors
    ///
    /// - [`FerroError::ShapeMismatch`] if `x` and `y` have different numbers of rows.
    /// - [`FerroError::InsufficientSamples`] if there are no samples.
    fn fit(&self, x: &Array2<F>, y: &Array1<usize>) -> Result<FittedBernoulliNB<F>, FerroError> {
        let (n_samples, n_features) = x.dim();

        if n_samples == 0 {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "BernoulliNB requires at least one sample".into(),
            });
        }

        if n_samples != y.len() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_samples],
                actual: vec![y.len()],
                context: "y length must match number of samples in X".into(),
            });
        }

        // Optionally binarize.
        let x_bin = if let Some(threshold) = self.binarize {
            binarize_array(x, threshold)
        } else {
            x.clone()
        };

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
        let two = F::from(2.0).unwrap();
        let alpha = crate::clamp_alpha(self.alpha, self.force_alpha);

        let mut log_prior = Array1::<F>::zeros(n_classes);
        let mut log_prob = Array2::<F>::zeros((n_classes, n_features));
        let mut log_neg_prob = Array2::<F>::zeros((n_classes, n_features));

        let mut feature_counts = Array2::<F>::zeros((n_classes, n_features));
        let mut class_counts_vec = vec![0usize; n_classes];

        for (ci, &class_label) in classes.iter().enumerate() {
            let class_indices: Vec<usize> = y
                .iter()
                .enumerate()
                .filter_map(|(i, &label)| if label == class_label { Some(i) } else { None })
                .collect();

            let n_c = class_indices.len();
            let n_c_f = F::from(n_c).unwrap();
            log_prior[ci] = (n_c_f / n_f).ln();
            class_counts_vec[ci] = n_c;

            // Count occurrences of each feature in this class.
            for j in 0..n_features {
                let fc = class_indices
                    .iter()
                    .fold(F::zero(), |acc, &i| acc + x_bin[[i, j]]);

                feature_counts[[ci, j]] = fc;

                // Smoothed probability: (N_cj + alpha) / (N_c + 2*alpha).
                let p = (fc + alpha) / (n_c_f + two * alpha);
                log_prob[[ci, j]] = p.ln();
                log_neg_prob[[ci, j]] = (F::one() - p).ln();
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

        Ok(FittedBernoulliNB {
            classes,
            log_prior,
            log_prob,
            log_neg_prob,
            binarize: self.binarize,
            feature_counts,
            class_counts: class_counts_vec,
            alpha,
            class_prior: self.class_prior.clone(),
            fit_prior: self.fit_prior,
        })
    }
}

impl<F: Float + Send + Sync + 'static> FittedBernoulliNB<F> {
    /// Incrementally update the model with new data.
    ///
    /// Accumulates feature counts and class counts, then recomputes
    /// the log probabilities.
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

        if n_features != self.log_prob.ncols() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![self.log_prob.ncols()],
                actual: vec![n_features],
                context: "number of features must match fitted BernoulliNB".into(),
            });
        }

        let x_bin = if let Some(threshold) = self.binarize {
            binarize_array(x, threshold)
        } else {
            x.clone()
        };

        let two = F::from(2.0).unwrap();

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
                    self.feature_counts[[ci, j]] = self.feature_counts[[ci, j]] + x_bin[[i, j]];
                }
            }
        }

        // Recompute log probabilities from accumulated feature_counts.
        let n_classes = self.classes.len();
        for ci in 0..n_classes {
            let n_c_f = F::from(self.class_counts[ci]).unwrap();
            for j in 0..n_features {
                let p = (self.feature_counts[[ci, j]] + self.alpha) / (n_c_f + two * self.alpha);
                self.log_prob[[ci, j]] = p.ln();
                self.log_neg_prob[[ci, j]] = (F::one() - p).ln();
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
    /// If binarize was set during fitting, features are binarized before
    /// prediction. Delegates to [`BaseNB::nb_predict_proba`].
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
    /// If binarize was set during fitting, features are binarized first.
    /// Returns shape `(n_samples, n_classes)`. Matches sklearn
    /// `BernoulliNB._joint_log_likelihood`. Delegates to
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

impl<F: Float + Send + Sync + 'static> FittedBernoulliNB<F> {
    /// Empirical log probability of features given a class,
    /// shape `(n_classes, n_features)`.
    ///
    /// Mirrors sklearn `BernoulliNB.feature_log_prob_`
    /// (`_update_feature_log_prob`, `naive_bayes.py:1201`).
    #[must_use]
    pub fn feature_log_prob(&self) -> &Array2<F> {
        &self.log_prob
    }

    /// Smoothed empirical log probability for each class,
    /// shape `(n_classes,)`.
    ///
    /// Mirrors sklearn `BernoulliNB.class_log_prior_`
    /// (`_update_class_log_prior`, `naive_bayes.py:600`).
    #[must_use]
    pub fn class_log_prior(&self) -> &Array1<F> {
        &self.log_prior
    }

    /// Number of samples encountered for each (class, feature) during fitting,
    /// shape `(n_classes, n_features)`.
    ///
    /// Mirrors sklearn `BernoulliNB.feature_count_`
    /// (`_count`, `naive_bayes.py:1189`).
    #[must_use]
    pub fn feature_count(&self) -> &Array2<F> {
        &self.feature_counts
    }

    /// Number of samples encountered for each class during fitting,
    /// shape `(n_classes,)`.
    ///
    /// Mirrors sklearn `BernoulliNB.class_count_`
    /// (`_count`, `naive_bayes.py:1190`). `class_counts` is stored as
    /// integer counts; this casts each to `F`.
    #[must_use]
    pub fn class_count(&self) -> Array1<F> {
        Array1::from_iter(
            self.class_counts
                .iter()
                .map(|&c| F::from(c).unwrap_or_else(F::zero)),
        )
    }
}

impl<F: Float + Send + Sync + 'static> BaseNB<F> for FittedBernoulliNB<F> {
    /// Compute joint log-likelihood for each class — sklearn
    /// `BernoulliNB._joint_log_likelihood`.
    ///
    /// If binarize was set during fitting, features are binarized first.
    /// Returns shape `(n_samples, n_classes)`.
    fn joint_log_likelihood(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let n_features_fitted = self.log_prob.ncols();
        if x.ncols() != n_features_fitted {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_features_fitted],
                actual: vec![x.ncols()],
                context: "number of features must match fitted BernoulliNB".into(),
            });
        }

        let x_bin = if let Some(threshold) = self.binarize {
            binarize_array(x, threshold)
        } else {
            x.clone()
        };

        let n_samples = x_bin.nrows();
        let n_classes = self.classes.len();
        let n_features = x_bin.ncols();

        let mut scores = Array2::<F>::zeros((n_samples, n_classes));

        for i in 0..n_samples {
            for ci in 0..n_classes {
                let mut score = self.log_prior[ci];
                for j in 0..n_features {
                    let xij = x_bin[[i, j]];
                    // log P(x_j | c) = x_j * log(p_cj) + (1-x_j) * log(1-p_cj)
                    score = score
                        + xij * self.log_prob[[ci, j]]
                        + (F::one() - xij) * self.log_neg_prob[[ci, j]];
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

impl<F: Float + Send + Sync + 'static> Predict<Array2<F>> for FittedBernoulliNB<F> {
    type Output = Array1<usize>;
    type Error = FerroError;

    /// Predict class labels for the given feature matrix.
    ///
    /// If binarize was set during fitting, features are binarized before
    /// prediction. Delegates to [`BaseNB::nb_predict`].
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of features does
    /// not match the fitted model.
    fn predict(&self, x: &Array2<F>) -> Result<Array1<usize>, FerroError> {
        BaseNB::nb_predict(self, x)
    }
}

impl<F: Float + Send + Sync + 'static> HasClasses for FittedBernoulliNB<F> {
    fn classes(&self) -> &[usize] {
        &self.classes
    }

    fn n_classes(&self) -> usize {
        self.classes.len()
    }
}

// Pipeline integration.
impl<F: Float + ToPrimitive + FromPrimitive + Send + Sync + 'static> PipelineEstimator<F>
    for BernoulliNB<F>
{
    fn fit_pipeline(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
    ) -> Result<Box<dyn FittedPipelineEstimator<F>>, FerroError> {
        // sklearn `BernoulliNB.fit` sets `classes_ = np.unique(y)` — the sorted
        // unique ORIGINAL labels (via `label_binarize`); `predict` returns
        // `self.classes_[np.argmax(jll, axis=1)]` — the original labels, NOT
        // class indices (`naive_bayes.py:103`). Preserve the original float
        // labels here instead of collapsing them to usize indices.
        let mut classes_orig: Vec<F> = y.to_vec();
        classes_orig.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
        classes_orig.dedup();
        // Map each label to its index into `classes_orig` (0..n_classes).
        let y_idx: Array1<usize> =
            y.mapv(|v| classes_orig.iter().position(|&c| c == v).unwrap_or(0));
        let fitted = self.fit(x, &y_idx)?;
        Ok(Box::new(FittedBernoulliNBPipeline {
            fitted,
            classes_orig,
        }))
    }
}

struct FittedBernoulliNBPipeline<F: Float + Send + Sync + 'static> {
    fitted: FittedBernoulliNB<F>,
    classes_orig: Vec<F>,
}

// SAFETY: `FittedBernoulliNB<F>` and `Vec<F>` are both Send when `F: Send`; this
// mirrors the existing inner-type bound and adds no interior mutability.
unsafe impl<F: Float + Send + Sync + 'static> Send for FittedBernoulliNBPipeline<F> {}
// SAFETY: `FittedBernoulliNB<F>` and `Vec<F>` are both Sync when `F: Sync`; no
// shared interior mutability is introduced.
unsafe impl<F: Float + Send + Sync + 'static> Sync for FittedBernoulliNBPipeline<F> {}

impl<F: Float + ToPrimitive + FromPrimitive + Send + Sync + 'static> FittedPipelineEstimator<F>
    for FittedBernoulliNBPipeline<F>
{
    fn predict_pipeline(&self, x: &Array2<F>) -> Result<Array1<F>, FerroError> {
        // `self.fitted.predict` returns the class indices (`0..n_classes`) the
        // model was trained on; map each back to the original label, mirroring
        // sklearn `classes_[argmax(jll)]` (`naive_bayes.py:103`).
        let preds = self.fitted.predict(x)?;
        Ok(preds.mapv(|i| self.classes_orig.get(i).copied().unwrap_or_else(F::nan)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    fn make_binary_data() -> (Array2<f64>, Array1<usize>) {
        let x = Array2::from_shape_vec(
            (6, 3),
            vec![
                1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0,
                0.0, 1.0,
            ],
        )
        .unwrap();
        let y = array![0usize, 0, 0, 1, 1, 1];
        (x, y)
    }

    #[test]
    fn test_bernoulli_nb_fit_predict() {
        let (x, y) = make_binary_data();
        let model = BernoulliNB::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        let correct = preds.iter().zip(y.iter()).filter(|(p, a)| p == a).count();
        assert_eq!(correct, 6);
    }

    #[test]
    fn test_bernoulli_nb_predict_proba_sums_to_one() {
        let (x, y) = make_binary_data();
        let model = BernoulliNB::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();
        let proba = fitted.predict_proba(&x).unwrap();
        for i in 0..proba.nrows() {
            assert_relative_eq!(proba.row(i).sum(), 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_bernoulli_nb_has_classes() {
        let (x, y) = make_binary_data();
        let model = BernoulliNB::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();
        assert_eq!(fitted.classes(), &[0, 1]);
        assert_eq!(fitted.n_classes(), 2);
    }

    #[test]
    fn test_bernoulli_nb_binarize_threshold() {
        // Continuous data binarized at 0.5.
        let x = Array2::from_shape_vec(
            (6, 3),
            vec![
                0.9, 0.8, 0.1, 0.7, 0.2, 0.3, 0.8, 0.9, 0.1, 0.2, 0.1, 0.9, 0.1, 0.8, 0.7, 0.3,
                0.2, 0.8,
            ],
        )
        .unwrap();
        let y = array![0usize, 0, 0, 1, 1, 1];

        let model = BernoulliNB::<f64>::new().with_binarize(0.5);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        let correct = preds.iter().zip(y.iter()).filter(|(p, a)| p == a).count();
        assert_eq!(correct, 6);
    }

    #[test]
    fn test_bernoulli_nb_binarize_zero_threshold() {
        // With threshold=0.0, all positive values become 1.
        let x =
            Array2::from_shape_vec((4, 2), vec![2.0, 0.0, 3.0, 0.0, 0.0, 2.0, 0.0, 3.0]).unwrap();
        let y = array![0usize, 0, 1, 1];

        let model = BernoulliNB::<f64>::new().with_binarize(0.0);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds[0], 0);
        assert_eq!(preds[3], 1);
    }

    #[test]
    fn test_bernoulli_nb_shape_mismatch_fit() {
        let x = Array2::from_shape_vec((4, 3), vec![1.0; 12]).unwrap();
        let y = array![0usize, 1]; // Wrong length
        let model = BernoulliNB::<f64>::new();
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_bernoulli_nb_shape_mismatch_predict() {
        let (x, y) = make_binary_data();
        let model = BernoulliNB::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();
        let x_bad = Array2::from_shape_vec((3, 5), vec![0.0; 15]).unwrap();
        assert!(fitted.predict(&x_bad).is_err());
        assert!(fitted.predict_proba(&x_bad).is_err());
    }

    #[test]
    fn test_bernoulli_nb_single_class() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();
        let y = array![5usize, 5, 5];
        let model = BernoulliNB::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();
        assert_eq!(fitted.classes(), &[5]);
        let preds = fitted.predict(&x).unwrap();
        assert!(preds.iter().all(|&p| p == 5));
    }

    #[test]
    fn test_bernoulli_nb_empty_data() {
        let x = Array2::<f64>::zeros((0, 3));
        let y = Array1::<usize>::zeros(0);
        let model = BernoulliNB::<f64>::new();
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_bernoulli_nb_default() {
        let model = BernoulliNB::<f64>::default();
        assert_relative_eq!(model.alpha, 1.0, epsilon = 1e-15);
        // sklearn default: binarize=0.0 (naive_bayes.py:1164).
        assert_eq!(model.binarize, Some(0.0));
    }

    #[test]
    fn test_bernoulli_nb_predict_proba_ordering() {
        let (x, y) = make_binary_data();
        let model = BernoulliNB::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();
        let proba = fitted.predict_proba(&x).unwrap();
        // First 3 samples should prefer class 0.
        for i in 0..3 {
            assert!(proba[[i, 0]] > proba[[i, 1]]);
        }
        // Last 3 samples should prefer class 1.
        for i in 3..6 {
            assert!(proba[[i, 1]] > proba[[i, 0]]);
        }
    }

    #[test]
    fn test_bernoulli_nb_partial_fit() {
        let x1 = Array2::from_shape_vec(
            (4, 3),
            vec![1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0],
        )
        .unwrap();
        let y1 = array![0usize, 0, 1, 1];

        let model = BernoulliNB::<f64>::new();
        let mut fitted = model.fit(&x1, &y1).unwrap();

        let x2 = Array2::from_shape_vec((2, 3), vec![1.0, 1.0, 0.0, 0.0, 0.0, 1.0]).unwrap();
        let y2 = array![0usize, 1];

        fitted.partial_fit(&x2, &y2).unwrap();

        let preds = fitted.predict(&x1).unwrap();
        assert_eq!(preds.len(), 4);
    }

    #[test]
    fn test_bernoulli_nb_partial_fit_shape_mismatch() {
        let (x, y) = make_binary_data();
        let model = BernoulliNB::<f64>::new();
        let mut fitted = model.fit(&x, &y).unwrap();

        let x_bad = Array2::from_shape_vec((2, 5), vec![1.0; 10]).unwrap();
        let y_bad = array![0usize, 1];
        assert!(fitted.partial_fit(&x_bad, &y_bad).is_err());
    }

    #[test]
    fn test_bernoulli_nb_class_prior() {
        let (x, y) = make_binary_data();
        let model = BernoulliNB::<f64>::new().with_class_prior(vec![0.8, 0.2]);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 6);
    }

    #[test]
    fn test_bernoulli_nb_class_prior_wrong_length() {
        let (x, y) = make_binary_data();
        let model = BernoulliNB::<f64>::new().with_class_prior(vec![0.5, 0.3, 0.2]);
        assert!(model.fit(&x, &y).is_err());
    }

    // sklearn 1.5.2 oracle fixture (R-CHAR-3) for the REQ-9a fitted accessors.
    // X = [[1,1,0],[1,0,0],[1,1,0],[0,0,1],[0,1,1],[0,0,1]], y = [0,0,0,1,1,1].
    fn oracle_xy() -> (Array2<f64>, Array1<usize>) {
        let x = array![
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0],
        ];
        let y = array![0usize, 0, 0, 1, 1, 1];
        (x, y)
    }

    #[test]
    #[allow(
        clippy::approx_constant,
        reason = "literal -0.6931471806 is the sklearn class_log_prior_ oracle value log(0.5), not a use of the LN_2 constant"
    )]
    fn bernoulli_feature_log_prob_and_class_log_prior_match_sklearn() -> Result<(), FerroError> {
        // sklearn BernoulliNB().fit(X, y):
        //   feature_log_prob_ = [[-0.2231435513, -0.5108256238, -1.6094379124],
        //                        [-1.6094379124, -0.9162907319, -0.2231435513]]
        //   class_log_prior_  = [-0.6931471806, -0.6931471806]
        let (x, y) = oracle_xy();
        let fitted = BernoulliNB::<f64>::new().fit(&x, &y)?;

        let expected_flp = array![
            [-0.2231435513, -0.5108256238, -1.6094379124],
            [-1.6094379124, -0.9162907319, -0.2231435513],
        ];
        let flp = fitted.feature_log_prob();
        assert_eq!(flp.dim(), (2, 3));
        for ((i, j), &e) in expected_flp.indexed_iter() {
            assert_relative_eq!(flp[[i, j]], e, epsilon = 1e-9);
        }

        let expected_clp = array![-0.6931471806, -0.6931471806];
        let clp = fitted.class_log_prior();
        assert_eq!(clp.len(), 2);
        for (i, &e) in expected_clp.iter().enumerate() {
            assert_relative_eq!(clp[i], e, epsilon = 1e-9);
        }
        Ok(())
    }

    #[test]
    fn bernoulli_feature_count_and_class_count_match_sklearn() -> Result<(), FerroError> {
        // sklearn BernoulliNB().fit(X, y):
        //   feature_count_ = [[3.0, 2.0, 0.0], [0.0, 1.0, 3.0]]
        //   class_count_   = [3.0, 3.0]
        let (x, y) = oracle_xy();
        let fitted = BernoulliNB::<f64>::new().fit(&x, &y)?;

        let expected_fc = array![[3.0, 2.0, 0.0], [0.0, 1.0, 3.0]];
        let fc = fitted.feature_count();
        assert_eq!(fc.dim(), (2, 3));
        for ((i, j), &e) in expected_fc.indexed_iter() {
            assert_relative_eq!(fc[[i, j]], e, epsilon = 1e-9);
        }

        let expected_cc = array![3.0, 3.0];
        let cc = fitted.class_count();
        assert_eq!(cc.len(), 2);
        for (i, &e) in expected_cc.iter().enumerate() {
            assert_relative_eq!(cc[i], e, epsilon = 1e-9);
        }
        Ok(())
    }

    // The `PipelineEstimator` adapter must preserve the ORIGINAL float labels:
    // sklearn `BernoulliNB.fit` sets `classes_ = np.unique(y)` and `predict`
    // returns `classes_[argmax(jll)]` — the original labels, NOT class indices
    // (`naive_bayes.py:103`). Live sklearn 1.5.2 oracle (run from /tmp):
    //   X=[[3,0],[4,0],[0,3],[0,4]], y=[-1,-1,1,1], q=[[5,0],[0,5]]
    //   BernoulliNB().fit(X,y).classes_  -> [-1, 1]
    //   BernoulliNB().fit(X,y).predict(q) -> [-1, 1]   (NOT [0, 1])
    #[test]
    fn bernoulli_pipeline_preserves_original_float_labels() -> Result<(), FerroError> {
        let x = array![[3.0, 0.0], [4.0, 0.0], [0.0, 3.0], [0.0, 4.0]];
        let y = array![-1.0, -1.0, 1.0, 1.0];
        let f = BernoulliNB::<f64>::new().fit_pipeline(&x, &y)?;
        let p = f.predict_pipeline(&array![[5.0, 0.0], [0.0, 5.0]])?;
        // Original labels [-1.0, 1.0], not the collapsed indices [0.0, 1.0].
        assert_eq!(p, array![-1.0, 1.0]);
        Ok(())
    }
}
