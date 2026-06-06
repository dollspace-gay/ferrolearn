//! Complement Naive Bayes classifier.
//!
//! This module provides [`ComplementNB`], a variant of Multinomial Naive Bayes
//! that is particularly well-suited for imbalanced datasets. Instead of estimating
//! the likelihood of a feature given a class, it estimates the likelihood of the
//! feature given all *other* (complement) classes and inverts the weights.
//!
//! The weight for feature `j` in class `c` is:
//!
//! ```text
//! w_cj = log( (N_~cj + alpha) / (N_~c + alpha * n_features) )
//! ```
//!
//! where `N_~cj` is the total count of feature `j` in all classes except `c`,
//! and `N_~c` is the total count of all features in all classes except `c`.
//!
//! Stores weights with sklearn's sign convention (positive
//! `-log(complement_prob)`), and prediction uses
//! `argmax_c sum_j x_j * w_cj` — matching sklearn's
//! `argmax(X @ feature_log_prob.T)` exactly.
//!
//! # Examples
//!
//! ```
//! use ferrolearn_bayes::ComplementNB;
//! use ferrolearn_core::{Fit, Predict};
//! use ndarray::{array, Array2};
//!
//! let x = Array2::from_shape_vec(
//!     (6, 3),
//!     vec![
//!         5.0, 1.0, 0.0,
//!         4.0, 2.0, 0.0,
//!         6.0, 0.0, 1.0,
//!         0.0, 1.0, 5.0,
//!         1.0, 0.0, 4.0,
//!         0.0, 2.0, 6.0,
//!     ],
//! ).unwrap();
//! let y = array![0usize, 0, 0, 1, 1, 1];
//!
//! let model = ComplementNB::<f64>::new();
//! let fitted = model.fit(&x, &y).unwrap();
//! let preds = fitted.predict(&x).unwrap();
//! assert_eq!(preds.len(), 6);
//! ```
//!
//! # `## REQ status`
//!
//! Binary classification (R-DEFER-2): two states only — SHIPPED needs impl + a
//! non-test production consumer + green verification; NOT-STARTED carries the
//! open prereq blocker. The non-test production consumer is `_RsComplementNB` /
//! `RsComplementNB` (`ferrolearn-python/src/extras.rs`, built via the
//! `py_classifier!` macro), which exercises `new(alpha, fit_prior, norm)` / `fit`
//! / `predict` against the library `FittedComplementNB` and is surfaced as
//! `ferrolearn.ComplementNB`; plus the in-crate `impl PipelineEstimator for
//! ComplementNB` (`fit_pipeline` / `predict_pipeline`). The pipeline adapter
//! preserves the ORIGINAL labels: `fit_pipeline` sets `classes_ = np.unique(y)`
//! (sorted unique original float labels, via `label_binarize`) and
//! `predict_pipeline` returns those original labels (`classes_[argmax(jll)]`,
//! `naive_bayes.py:103`), not `0..n_classes` indices. Green verification = the
//! in-tree `complement` lib tests plus the live-sklearn pin / guards
//! (`ferrolearn-bayes/tests/divergence_complement.rs`):
//! `divergence_complement_negative_alpha_rejected` (#914, now PASSING after the
//! `alpha < 0` reject landed in `fn fit`), then the green guards
//! `green_complement_predict_value_norm_false`,
//! `green_complement_predict_value_norm_true`,
//! `green_complement_class_prior_length_only`,
//! `green_complement_score_accuracy`,
//! `green_complement_negative_features_rejected` — all passing. Cites use symbol
//! anchors (ferrolearn) / `file:line` (sklearn 1.5.2, commit 156ef14). Live
//! oracle = installed sklearn 1.5.2. (REQ numbering follows the design doc
//! `.design/bayes/complement.md`; suggested blocker numbers continue the bayes
//! layer past bernoulli #905-910.)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (`feature_log_prob_` complement-weight + `_joint_log_likelihood` / `predict` / `predict_proba` / `predict_log_proba` / `predict_joint_log_proba` VALUE, norm=False) | SHIPPED | `fn fit` for `ComplementNB` sets `weights[[ci,j]] = -((total_feature_counts[j] - class_feature_counts[ci,j] + alpha) / (complement_total + alpha*n_features)).ln()` — the algebraic identity of `_update_feature_log_prob`'s `-logged` (`naive_bayes.py:1032-1042`: `comp_count = feature_all_ + alpha - feature_count_`; `logged = log(comp_count / comp_count.sum(axis=1, keepdims=True))`; `feature_log_prob_ = -logged`); `impl BaseNB::joint_log_likelihood` for `FittedComplementNB` computes `X @ weights.T` (`scores[[i,ci]] = sum_j x[i,j] * weights[ci,j]`), mirroring `jll = safe_sparse_dot(X, feature_log_prob_.T)` (`naive_bayes.py:1046`); the four `predict_*` delegate to the `BaseNB` provided methods. The single-class `+ class_log_prior_` add (`naive_bayes.py:1047-1048`) is omitted and BENIGN — single-class `class_log_prior_ = [0.0]` and one-column softmax is always `[[1.0]]`. Non-test consumer: `RsComplementNB::fit`/`predict` (`ferrolearn-python/src/extras.rs`, `py_classifier!`) → `FittedComplementNB`, surfaced as `ferrolearn.ComplementNB`; plus `impl PipelineEstimator`. Verified: green guard `green_complement_predict_value_norm_false` — on `X=[[5,1,0],[4,2,0],[6,0,1],[0,1,5],[1,0,4],[0,2,6]]`, `y=[0,0,0,1,1,1]`, `q=[[3,1,1],[0,1,4]]`, sklearn `predict_proba(q) = [[0.9846153846153846, 0.015384615384615375], [0.0002440810349035878, 0.9997559189650967]]`, `predict_joint_log_proba(q) = [[9.216887641752072, 5.058004558392399], [2.9785630167125636, 11.296329183431908]]`, `predict(q) = [0, 1]`; ferrolearn matches to ≤1e-12. In-tree `test_complement_nb_fit_predict` / `test_complement_nb_predict_proba_sums_to_one` / `test_complement_nb_imbalanced_data` / `test_complement_nb_three_classes` / `test_complement_nb_single_class`. |
//! | REQ-2 (`alpha >= 0` validation) | SHIPPED | `fn fit` rejects `self.alpha < F::zero()` with `FerroError::InvalidParameter { name: "alpha", reason: "alpha must be >= 0 (sklearn Interval[0, inf))" }`, mirroring the shared `_BaseDiscreteNB._parameter_constraints` `alpha: [Interval(Real, 0, None, closed="left"), "array-like"]` (`naive_bayes.py:530`) inherited by `ComplementNB._parameter_constraints` (`naive_bayes.py:1000-1003`) — the HARD `>= 0` reject `_validate_params` enforces at `fit`, DISTINCT from `_check_alpha`'s `1e-10` floor (`naive_bayes.py:604-626`, `force_alpha`-only; `alpha=0` stays allowed). Non-test consumer: `RsComplementNB::fit` (`extras.rs`) maps the `FerroError` → `PyErr`. Verified: green pin `divergence_complement_negative_alpha_rejected` (#914, now PASSING): `with_alpha(-0.5).fit(X,y)` returns `Err` (sklearn raises `InvalidParameterError`, "The 'alpha' parameter of ComplementNB must be a float in the range [0.0, inf) or an array-like. Got -0.5 instead."). |
//! | REQ-3 (`norm=True` VALUE) | SHIPPED | `fn fit` / `partial_fit` call `fn apply_norm_inplace`, which divides each `weights` row (= `-logged`) by its row sum — the algebraic identity of sklearn's `feature_log_prob_ = logged / logged.sum(axis=1, keepdims=True)` (`naive_bayes.py:1037-1039`); the two minus signs in `(-logged)/sum(-logged)` cancel. Non-test consumer: `RsComplementNB` threads `norm` through `with_norm(norm)` (`extras.rs`); surfaced as `ferrolearn.ComplementNB(norm=...)`. Verified: green guard `green_complement_predict_value_norm_true` — `ComplementNB(norm=True).fit(X,y)` sklearn `predict_proba(q) = [[0.7192390704948571, 0.2807609295051429], [0.13223037910101987, 0.8677696208989801]]`, `predict(q) = [0, 1]`; ferrolearn `with_norm(true)` produces the IDENTICAL proba/labels to ≤1e-12. |
//! | REQ-4 (`class_prior` LENGTH-only validation — MATCH) | SHIPPED | `fn fit` validates ONLY `priors.len() != n_classes` (then carries the priors), mirroring `_update_class_log_prior` (`naive_bayes.py:589-591`: `if len(class_prior) != n_classes: ValueError; class_log_prior_ = log(class_prior)`) — discrete NB has NO sum-to-1 / non-negativity check. A deliberate MATCH. Non-test consumer: `RsComplementNB` builds `ComplementNB` (the `with_class_prior` path is exercised in-crate + pipeline). Verified: green guard `green_complement_class_prior_length_only` — `with_class_prior([0.5,0.3]).fit(X,y)` SUCCEEDS (sum 0.8; sklearn `class_log_prior_ = log([0.5,0.3])`, NO error), `with_class_prior([0.5]).fit` errors. In-tree `test_complement_nb_class_prior` / `test_complement_nb_class_prior_wrong_length`. (Wrong-length error TYPE differs — `InvalidParameter` vs `ValueError` — folded into REQ-9's surface gap. For ComplementNB `class_prior` is "Not used" in multi-class predict, `naive_bayes.py:929` — only the length decision is observable.) |
//! | REQ-5 (`force_alpha` floor + `fit_prior` carry) | SHIPPED | `fn fit` / `partial_fit` call `crate::clamp_alpha(self.alpha, self.force_alpha)` (`base::check_alpha`, the `_check_alpha` floor `1e-10` unless `force_alpha`, `naive_bayes.py:604-626`); `fit_prior` is stored (matching sklearn, only the single-class edge case consults the prior — benign here). Non-test consumer: `RsComplementNB` passes `fit_prior` through `with_fit_prior`, `alpha` through `with_alpha`. Verified: with `force_alpha=true` default and `alpha=1`, `score(X,y) = 1.0` (green `green_complement_score_accuracy`); `clamp_alpha(1, true) = 1`. In-tree `test_complement_nb_default`; `base.rs` `test_check_alpha_*`. |
//! | REQ-6 (`partial_fit` VALUE — same-classes path) | SHIPPED | `FittedComplementNB::partial_fit` accumulates `class_counts` / `feature_counts` for each EXISTING class, then re-derives `total_feature_counts` (the `feature_all_` analog) / `total_all` and recomputes `weights` (same `-log` complement smoothing), re-applying `apply_norm_inplace` when `norm`, mirroring the shared `_BaseDiscreteNB.partial_fit` accumulate-then-recompute (`naive_bayes.py:628-709`, `_count` re-deriving `feature_all_` → `_update_feature_log_prob`). Non-test consumer: in-crate (the PyO3 `partial_fit` gap is REQ-9). Verified: in-tree `test_complement_nb_partial_fit` / `test_complement_nb_partial_fit_shape_mismatch` — chunked `partial_fit` over already-fitted classes reproduces the accumulate-then-recompute path (sklearn two-chunk `partial_fit` == `fit` on the whole, `np.allclose == True`). KNOWN GAP: `partial_fit` has NO `classes=` argument — it loops only over the already-fitted `self.classes`, so a brand-new later-chunk label is silently dropped (sklearn binarizes against the full `classes=` list from the first call, `naive_bayes.py:628-709`); this `classes=`/unseen-label path is NOT-STARTED (folded into #915). |
//! | REQ-7 (negative-feature guard — both reject) | SHIPPED | `fn fit` (and `partial_fit`) reject any `x[i,j] < 0` with `FerroError::InvalidParameter { name: "X", reason: "ComplementNB requires non-negative feature values" }`, mirroring `check_non_negative(X, "ComplementNB (input X)")` → `ValueError` (`naive_bayes.py:1027`; ComplementNB DOES guard non-negativity, unlike BernoulliNB). Both REJECT. Non-test consumer: `RsComplementNB::fit` (`extras.rs`) maps the `FerroError` to a `PyErr`. Verified: green guard `green_complement_negative_features_rejected` — `ComplementNB().fit(X_neg, y)` returns `Err` (sklearn `ValueError("Negative values in data passed to ComplementNB (input X)")`). In-tree `test_complement_nb_negative_features_error`. The exact sklearn MESSAGE/TYPE is NOT matched — that sub-item is captured under REQ-9. |
//! | REQ-8 (`sample_weight` + `partial_fit` `classes=`) | NOT-STARTED | open prereq blocker **#915**. sklearn `fit(X, y, sample_weight=None)` (`naive_bayes.py:712`) weights the binarized `Y` so `feature_count_ = Y.T @ X` / `class_count_ = Y.sum(axis=0)` / `feature_all_ = feature_count_.sum(axis=0)` become weighted (`naive_bayes.py:1025-1030`). ferrolearn's `impl Fit<Array2<F>, Array1<usize>>` has signature `fn fit(&self, x, y)` — NO `sample_weight` parameter on `fit` or `partial_fit`; also no `classes=` argument on `partial_fit` (the unseen-label sub-gap of REQ-6). |
//! | REQ-9a (Rust fitted-attribute accessors) | SHIPPED | `FittedComplementNB` exposes `feature_log_prob(&self) -> &Array2<F>` (`&self.weights`, sklearn `feature_log_prob_`, `naive_bayes.py:1042`), `feature_count(&self) -> &Array2<F>` (`&self.feature_counts`, sklearn `feature_count_`, `naive_bayes.py:961`), `class_count(&self) -> Array1<F>` (the integer `class_counts` cast to `F`, sklearn `class_count_`, `naive_bayes.py:951`), `feature_all(&self) -> Array1<F>` (DERIVED `feature_counts.sum_axis(Axis(0))`, sklearn `feature_all_ = feature_count_.sum(axis=0)`, `naive_bayes.py:1029`), and `class_log_prior(&self) -> Array1<F>` (DERIVED empirical `log(class_count_) - log(class_count_.sum())`, sklearn `class_log_prior_`, `naive_bayes.py:600`). `coef_`/`intercept_` are DEPRECATED and REMOVED in sklearn 1.5.2 (`hasattr(ComplementNB().fit(...), 'coef_') == False`), so no `coef_`/`intercept_` getter is added. Live oracle (`X=[[5,1,0],[4,2,0],[6,0,1],[0,1,5],[1,0,4],[0,2,6]]`, `y=[0,0,0,1,1,1]`): `feature_log_prob_ = [[2.3978952728,1.7047480922,0.3184537311],[0.3184537311,1.7047480922,2.3978952728]]`, `feature_count_ = [[15,3,1],[1,3,15]]`, `class_count_ = [3,3]`, `feature_all_ = [16,6,16]`, `class_log_prior_ = [-0.6931471806,-0.6931471806]`. In-tree `complement_feature_log_prob_and_count_match_sklearn` / `complement_feature_all_class_count_prior_match_sklearn`. |
//! | REQ-9b (PyO3 surface + `sample_weight`) | NOT-STARTED | open prereq blocker **#916**. `_RsComplementNB` (`ferrolearn-python/src/extras.rs`, the `py_classifier!` macro) exposes ONLY `new(alpha, fit_prior, norm)` + `fit` + `predict` — NO `class_prior`/`force_alpha` kwargs, NO `predict_proba`/`predict_log_proba`/`predict_joint_log_proba`/`score`/`partial_fit` (which the library HAS), NO fitted-attr getters bridged to Python (`feature_log_prob_` / `feature_all_` / `feature_count_` / `class_count_` / `class_log_prior_` / `classes_` / `n_features_in_`). `coef_`/`intercept_` are deprecated/removed in sklearn 1.5.2 (`hasattr == False`) and stay absent. Also subsumes the negative-feature MESSAGE/TYPE-parity sub-item (REQ-7: `InvalidParameter` vs `ValueError`) and the `class_prior` wrong-length TYPE sub-item (REQ-4). The fix belongs in `ferrolearn-python` (multi-file). |
//! | REQ-10 (ferray substrate) | NOT-STARTED | open prereq blocker **#917**. `complement.rs` imports `ndarray::{Array1, Array2}` + `num_traits::{Float, FromPrimitive, ToPrimitive}` (the wrong substrate, R-SUBSTRATE-1); not migrated to `ferray-core`. |

use crate::base::BaseNB;
use ferrolearn_core::error::FerroError;
use ferrolearn_core::introspection::HasClasses;
use ferrolearn_core::pipeline::{FittedPipelineEstimator, PipelineEstimator};
use ferrolearn_core::traits::{Fit, Predict};
use ndarray::{Array1, Array2};
use num_traits::{Float, FromPrimitive, ToPrimitive};

/// Complement Naive Bayes classifier.
///
/// A variant of Multinomial NB that uses complement-class statistics.
/// More robust for imbalanced datasets.
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
#[derive(Debug, Clone)]
pub struct ComplementNB<F> {
    /// Additive (Laplace) smoothing parameter. Default: `1.0`.
    pub alpha: F,
    /// Optional user-supplied class priors. Note: ComplementNB does not
    /// use priors in the standard way (it uses complement weights), but
    /// this field is provided for API consistency with other NB variants.
    pub class_prior: Option<Vec<F>>,
    /// Whether to learn class priors from the data. Stored for API
    /// consistency; ComplementNB's predict does not consult priors in the
    /// multi-class case. Default: `true`.
    pub fit_prior: bool,
    /// When `false`, `alpha` values below `1e-10` are silently raised to
    /// `1e-10` (legacy behavior). Default: `true`.
    pub force_alpha: bool,
    /// When `true`, performs a second L1 normalization of the weights
    /// (Rennie et al. 2003 §4.4 "normalized weights" variant). Default:
    /// `false`.
    pub norm: bool,
}

impl<F: Float> ComplementNB<F> {
    /// Create a new `ComplementNB` with Laplace smoothing (`alpha = 1.0`).
    #[must_use]
    pub fn new() -> Self {
        Self {
            alpha: F::one(),
            class_prior: None,
            fit_prior: true,
            force_alpha: true,
            norm: false,
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
    /// The priors must have length equal to the number of classes discovered
    /// during fitting. Note: ComplementNB uses complement-class weights rather
    /// than direct class priors, but the priors are stored for API consistency.
    #[must_use]
    pub fn with_class_prior(mut self, priors: Vec<F>) -> Self {
        self.class_prior = Some(priors);
        self
    }

    /// Toggle `fit_prior`. Stored for API consistency with other discrete NBs.
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

    /// Toggle the second L1 normalization on weights (sklearn's `norm`
    /// parameter; Rennie et al. 2003 §4.4).
    #[must_use]
    pub fn with_norm(mut self, norm: bool) -> Self {
        self.norm = norm;
        self
    }
}

impl<F: Float> Default for ComplementNB<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted Complement Naive Bayes classifier.
#[derive(Debug, Clone)]
pub struct FittedComplementNB<F> {
    /// Sorted unique class labels.
    classes: Vec<usize>,
    /// Complement weights per class, shape `(n_classes, n_features)`.
    /// Each entry is `log( (N_~cj + alpha) / (N_~c + alpha * n_features) )`.
    weights: Array2<F>,
    /// Raw per-class feature count sums, shape `(n_classes, n_features)`.
    feature_counts: Array2<F>,
    /// Per-class sample counts.
    class_counts: Vec<usize>,
    /// Smoothing parameter carried forward for partial_fit (post-clamp
    /// when `force_alpha=false`).
    alpha: F,
    /// Whether to apply the second L1 normalization on weights (carried
    /// forward for partial_fit).
    norm: bool,
}

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, Array1<usize>> for ComplementNB<F> {
    type Fitted = FittedComplementNB<F>;
    type Error = FerroError;

    /// Fit the Complement NB model.
    ///
    /// # Errors
    ///
    /// - [`FerroError::ShapeMismatch`] if `x` and `y` have different numbers of rows.
    /// - [`FerroError::InsufficientSamples`] if there are no samples.
    /// - [`FerroError::InvalidParameter`] if any feature value is negative.
    fn fit(&self, x: &Array2<F>, y: &Array1<usize>) -> Result<FittedComplementNB<F>, FerroError> {
        let (n_samples, n_features) = x.dim();

        if n_samples == 0 {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "ComplementNB requires at least one sample".into(),
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
                reason: "ComplementNB requires non-negative feature values".into(),
            });
        }

        // Collect sorted unique classes.
        let mut classes: Vec<usize> = y.to_vec();
        classes.sort_unstable();
        classes.dedup();
        let n_classes = classes.len();

        let n_feat_f = F::from(n_features).unwrap();
        // sklearn rejects alpha < 0 at fit via _parameter_constraints
        // `alpha: Interval(Real, 0, None, closed="left")` (naive_bayes.py:530,
        // inherited by ComplementNB at :1000-1003) — a HARD reject distinct
        // from `_check_alpha`'s 1e-10 floor (:619, force_alpha-only).
        if self.alpha < F::zero() {
            return Err(FerroError::InvalidParameter {
                name: "alpha".into(),
                reason: "alpha must be >= 0 (sklearn Interval[0, inf))".into(),
            });
        }

        let alpha = crate::clamp_alpha(self.alpha, self.force_alpha);

        // Compute per-class feature count sums, shape (n_classes, n_features).
        let mut class_feature_counts = Array2::<F>::zeros((n_classes, n_features));
        let mut class_counts = vec![0usize; n_classes];

        for (sample_idx, &label) in y.iter().enumerate() {
            let ci = classes.iter().position(|&c| c == label).unwrap();
            class_counts[ci] += 1;
            for j in 0..n_features {
                class_feature_counts[[ci, j]] = class_feature_counts[[ci, j]] + x[[sample_idx, j]];
            }
        }

        // Total feature counts across all classes.
        let total_feature_counts: Array1<F> = class_feature_counts.rows().into_iter().fold(
            Array1::<F>::zeros(n_features),
            |acc, row| {
                let mut result = acc;
                for j in 0..n_features {
                    result[j] = result[j] + row[j];
                }
                result
            },
        );

        let total_all: F = total_feature_counts.sum();

        // Compute complement-log weights for each class. sklearn stores
        // `feature_log_prob_ = -log((complement_count + alpha) / (total + alpha*n_features))`
        // (positive values — see #346). ferrolearn previously stored the
        // pre-negation value; we now match sklearn's convention so
        // introspection is parity-correct and predict uses argmax.
        let mut weights = Array2::<F>::zeros((n_classes, n_features));

        for ci in 0..n_classes {
            // Complement counts: sum over all other classes.
            let complement_total = total_all - class_feature_counts.row(ci).sum();

            let denom = complement_total + alpha * n_feat_f;

            for j in 0..n_features {
                let complement_count_j = total_feature_counts[j] - class_feature_counts[[ci, j]];
                // Negate so the stored value matches sklearn's
                // `feature_log_prob_` exactly: positive values whose
                // *smaller* indicates higher complement probability.
                weights[[ci, j]] = -((complement_count_j + alpha) / denom).ln();
            }
        }

        if self.norm {
            apply_norm_inplace(&mut weights);
        }

        // Validate class_prior length if provided.
        if let Some(ref priors) = self.class_prior
            && priors.len() != n_classes
        {
            return Err(FerroError::InvalidParameter {
                name: "class_prior".into(),
                reason: format!(
                    "length {} does not match number of classes {}",
                    priors.len(),
                    n_classes
                ),
            });
        }

        Ok(FittedComplementNB {
            classes,
            weights,
            feature_counts: class_feature_counts,
            class_counts,
            alpha,
            norm: self.norm,
        })
    }
}

/// Apply sklearn's `norm=True` second L1 normalization to complement weights.
///
/// `weights` is already stored as sklearn's positive `-log(complement_prob)`.
/// sklearn divides each row by its sum so rows sum to 1 (still positive,
/// since the unnormalised values are positive).
fn apply_norm_inplace<F: Float>(weights: &mut Array2<F>) {
    let n_classes = weights.nrows();
    let n_features = weights.ncols();
    for ci in 0..n_classes {
        let row_sum = (0..n_features).fold(F::zero(), |acc, j| acc + weights[[ci, j]]);
        if row_sum == F::zero() {
            continue;
        }
        for j in 0..n_features {
            weights[[ci, j]] = weights[[ci, j]] / row_sum;
        }
    }
}

impl<F: Float + Send + Sync + 'static> FittedComplementNB<F> {
    /// Incrementally update the model with new data.
    ///
    /// Accumulates feature counts and class counts, then recomputes
    /// the complement weights.
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

        if n_features != self.weights.ncols() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![self.weights.ncols()],
                actual: vec![n_features],
                context: "number of features must match fitted ComplementNB".into(),
            });
        }

        if x.iter().any(|&v| v < F::zero()) {
            return Err(FerroError::InvalidParameter {
                name: "X".into(),
                reason: "ComplementNB requires non-negative feature values".into(),
            });
        }

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

        // Recompute complement weights from accumulated feature_counts.
        let n_classes = self.classes.len();
        let n_feat_f = F::from(n_features).unwrap();

        let total_feature_counts: Array1<F> = self.feature_counts.rows().into_iter().fold(
            Array1::<F>::zeros(n_features),
            |acc, row| {
                let mut result = acc;
                for j in 0..n_features {
                    result[j] = result[j] + row[j];
                }
                result
            },
        );

        let total_all: F = total_feature_counts.sum();

        for ci in 0..n_classes {
            let complement_total = total_all - self.feature_counts.row(ci).sum();
            let denom = complement_total + self.alpha * n_feat_f;
            for j in 0..n_features {
                let complement_count_j = total_feature_counts[j] - self.feature_counts[[ci, j]];
                // sklearn-parity sign: positive -log(complement_prob).
                self.weights[[ci, j]] = -((complement_count_j + self.alpha) / denom).ln();
            }
        }

        if self.norm {
            apply_norm_inplace(&mut self.weights);
        }

        Ok(())
    }

    /// Predict class probabilities for the given feature matrix.
    ///
    /// Returns shape `(n_samples, n_classes)` where each row sums to 1.
    /// Delegates to [`BaseNB::nb_predict_proba`] — with ComplementNB's
    /// sklearn-parity sign, the joint log-likelihood is `X @ weights.T`
    /// directly, so `exp(jll - logsumexp(jll))` is the softmax of the
    /// complement scores.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of features does
    /// not match the fitted model.
    pub fn predict_proba(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        BaseNB::nb_predict_proba(self, x)
    }

    /// Compute the joint log-likelihood scores using sklearn's sign
    /// convention: argmax(jll) gives the predicted class.
    ///
    /// Returns shape `(n_samples, n_classes)`. With the sklearn-parity sign,
    /// `X @ weights.T` IS the joint log-likelihood. Matches sklearn
    /// `ComplementNB._joint_log_likelihood`. Delegates to
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

impl<F: Float + Send + Sync + 'static> FittedComplementNB<F> {
    /// Empirical complement weights (the negated smoothed complement-class
    /// log-probabilities), shape `(n_classes, n_features)`.
    ///
    /// Mirrors sklearn `ComplementNB.feature_log_prob_`
    /// (`_update_feature_log_prob`, `naive_bayes.py:1042`).
    #[must_use]
    pub fn feature_log_prob(&self) -> &Array2<F> {
        &self.weights
    }

    /// Number of samples encountered for each (class, feature) during fitting,
    /// shape `(n_classes, n_features)`.
    ///
    /// Mirrors sklearn `ComplementNB.feature_count_`
    /// (`_count`, `naive_bayes.py:961`).
    #[must_use]
    pub fn feature_count(&self) -> &Array2<F> {
        &self.feature_counts
    }

    /// Number of samples encountered for each class during fitting,
    /// shape `(n_classes,)`.
    ///
    /// Mirrors sklearn `ComplementNB.class_count_`
    /// (`_count`, `naive_bayes.py:951`). `class_counts` is stored as integer
    /// counts; this casts each to `F`.
    #[must_use]
    pub fn class_count(&self) -> Array1<F> {
        Array1::from_iter(
            self.class_counts
                .iter()
                .map(|&c| F::from(c).unwrap_or_else(F::zero)),
        )
    }

    /// Number of samples encountered for each feature during fitting (the
    /// per-feature total across all classes), shape `(n_features,)`.
    ///
    /// Derived (not stored) as `feature_count_.sum(axis=0)`, mirroring sklearn
    /// `ComplementNB.feature_all_` (`_count`, `feature_all_ =
    /// feature_count_.sum(axis=0)`, `naive_bayes.py:1029`).
    #[must_use]
    pub fn feature_all(&self) -> Array1<F> {
        self.feature_counts.sum_axis(ndarray::Axis(0))
    }

    /// Smoothed empirical log probability for each class, shape `(n_classes,)`.
    ///
    /// Derived (not stored) as `log(class_count_) - log(class_count_.sum())`,
    /// mirroring sklearn's EMPIRICAL `class_log_prior_` under the default
    /// `fit_prior=True` (`_update_class_log_prior`, `naive_bayes.py:600`).
    /// ComplementNB stores the empirical class-prior derivation; this returns
    /// the EMPIRICAL prior (matching sklearn's `class_log_prior_` value on any
    /// fit). Note: ComplementNB only consults `class_log_prior_` in the
    /// single-class edge case (`naive_bayes.py:1047-1048`); it does not affect
    /// multi-class predictions.
    #[must_use]
    pub fn class_log_prior(&self) -> Array1<F> {
        let total = self.class_counts.iter().fold(F::zero(), |acc, &c| {
            acc + F::from(c).unwrap_or_else(F::zero)
        });
        Array1::from_iter(
            self.class_counts
                .iter()
                .map(|&c| (F::from(c).unwrap_or_else(F::zero) / total).ln()),
        )
    }
}

impl<F: Float + Send + Sync + 'static> BaseNB<F> for FittedComplementNB<F> {
    /// Compute the joint log-likelihood scores for each class — sklearn
    /// `ComplementNB._joint_log_likelihood`.
    ///
    /// Returns `X @ feature_log_prob_.T` (shape `(n_samples, n_classes)`).
    /// With ferrolearn's sklearn-parity sign for `feature_log_prob_`,
    /// **higher is better** and `argmax(scores, axis=1)` predicts the class.
    fn joint_log_likelihood(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let n_features_fitted = self.weights.ncols();
        if x.ncols() != n_features_fitted {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_features_fitted],
                actual: vec![x.ncols()],
                context: "number of features must match fitted ComplementNB".into(),
            });
        }

        let n_samples = x.nrows();
        let n_classes = self.classes.len();
        let n_features = x.ncols();

        let mut scores = Array2::<F>::zeros((n_samples, n_classes));

        for i in 0..n_samples {
            for ci in 0..n_classes {
                let mut score = F::zero();
                for j in 0..n_features {
                    score = score + x[[i, j]] * self.weights[[ci, j]];
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

impl<F: Float + Send + Sync + 'static> Predict<Array2<F>> for FittedComplementNB<F> {
    type Output = Array1<usize>;
    type Error = FerroError;

    /// Predict class labels for the given feature matrix.
    ///
    /// With ComplementNB's sklearn-parity sign, the highest joint
    /// log-likelihood wins. Delegates to [`BaseNB::nb_predict`].
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of features does
    /// not match the fitted model.
    fn predict(&self, x: &Array2<F>) -> Result<Array1<usize>, FerroError> {
        BaseNB::nb_predict(self, x)
    }
}

impl<F: Float + Send + Sync + 'static> HasClasses for FittedComplementNB<F> {
    fn classes(&self) -> &[usize] {
        &self.classes
    }

    fn n_classes(&self) -> usize {
        self.classes.len()
    }
}

// Pipeline integration.
impl<F: Float + ToPrimitive + FromPrimitive + Send + Sync + 'static> PipelineEstimator<F>
    for ComplementNB<F>
{
    fn fit_pipeline(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
    ) -> Result<Box<dyn FittedPipelineEstimator<F>>, FerroError> {
        // sklearn `ComplementNB.fit` sets `classes_ = np.unique(y)` — the sorted
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
        Ok(Box::new(FittedComplementNBPipeline {
            fitted,
            classes_orig,
        }))
    }
}

struct FittedComplementNBPipeline<F: Float + Send + Sync + 'static> {
    fitted: FittedComplementNB<F>,
    classes_orig: Vec<F>,
}

// SAFETY: `FittedComplementNB<F>` and `Vec<F>` are both Send when `F: Send`; this
// mirrors the existing inner-type bound and adds no interior mutability.
unsafe impl<F: Float + Send + Sync + 'static> Send for FittedComplementNBPipeline<F> {}
// SAFETY: `FittedComplementNB<F>` and `Vec<F>` are both Sync when `F: Sync`; no
// shared interior mutability is introduced.
unsafe impl<F: Float + Send + Sync + 'static> Sync for FittedComplementNBPipeline<F> {}

impl<F: Float + ToPrimitive + FromPrimitive + Send + Sync + 'static> FittedPipelineEstimator<F>
    for FittedComplementNBPipeline<F>
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

    fn make_count_data() -> (Array2<f64>, Array1<usize>) {
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
    fn test_complement_nb_fit_predict() {
        let (x, y) = make_count_data();
        let model = ComplementNB::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        let correct = preds.iter().zip(y.iter()).filter(|(p, a)| p == a).count();
        assert_eq!(correct, 6);
    }

    #[test]
    fn test_complement_nb_predict_proba_sums_to_one() {
        let (x, y) = make_count_data();
        let model = ComplementNB::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();
        let proba = fitted.predict_proba(&x).unwrap();
        for i in 0..proba.nrows() {
            assert_relative_eq!(proba.row(i).sum(), 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_complement_nb_has_classes() {
        let (x, y) = make_count_data();
        let model = ComplementNB::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();
        assert_eq!(fitted.classes(), &[0, 1]);
        assert_eq!(fitted.n_classes(), 2);
    }

    #[test]
    fn test_complement_nb_shape_mismatch_fit() {
        let x = Array2::from_shape_vec((4, 3), vec![1.0; 12]).unwrap();
        let y = array![0usize, 1]; // Wrong length
        let model = ComplementNB::<f64>::new();
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_complement_nb_shape_mismatch_predict() {
        let (x, y) = make_count_data();
        let model = ComplementNB::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();
        let x_bad = Array2::from_shape_vec((3, 5), vec![1.0; 15]).unwrap();
        assert!(fitted.predict(&x_bad).is_err());
        assert!(fitted.predict_proba(&x_bad).is_err());
    }

    #[test]
    fn test_complement_nb_negative_features_error() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, -0.5, 3.0, 2.0, 1.0, 0.0, 4.0]).unwrap();
        let y = array![0usize, 0, 1, 1];
        let model = ComplementNB::<f64>::new();
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_complement_nb_single_class() {
        let x = Array2::from_shape_vec((3, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
            .unwrap();
        let y = array![0usize, 0, 0];
        let model = ComplementNB::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();
        assert_eq!(fitted.classes(), &[0]);
        let preds = fitted.predict(&x).unwrap();
        assert!(preds.iter().all(|&p| p == 0));
    }

    #[test]
    fn test_complement_nb_empty_data() {
        let x = Array2::<f64>::zeros((0, 3));
        let y = Array1::<usize>::zeros(0);
        let model = ComplementNB::<f64>::new();
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_complement_nb_default() {
        let model = ComplementNB::<f64>::default();
        assert_relative_eq!(model.alpha, 1.0, epsilon = 1e-15);
    }

    #[test]
    fn test_complement_nb_imbalanced_data() {
        // ComplementNB is designed for imbalanced data.
        // 10 samples of class 0, 2 samples of class 1.
        let x = Array2::from_shape_vec(
            (12, 3),
            vec![
                5.0, 1.0, 0.0, 4.0, 2.0, 0.0, 6.0, 0.0, 1.0, 5.0, 1.0, 0.0, 4.0, 2.0, 0.0, 6.0,
                0.0, 1.0, 5.0, 1.0, 0.0, 4.0, 2.0, 0.0, 6.0, 0.0, 1.0, 5.0, 1.0, 0.0, 0.0, 1.0,
                5.0, // class 1
                0.0, 2.0, 6.0, // class 1
            ],
        )
        .unwrap();
        let y = array![0usize, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1];

        let model = ComplementNB::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        // Class 1 samples should be predicted as class 1.
        assert_eq!(preds[10], 1);
        assert_eq!(preds[11], 1);
    }

    #[test]
    fn test_complement_nb_partial_fit() {
        let x1 = Array2::from_shape_vec(
            (4, 3),
            vec![5.0, 1.0, 0.0, 4.0, 2.0, 0.0, 0.0, 1.0, 5.0, 1.0, 0.0, 4.0],
        )
        .unwrap();
        let y1 = array![0usize, 0, 1, 1];

        let model = ComplementNB::<f64>::new();
        let mut fitted = model.fit(&x1, &y1).unwrap();

        let x2 = Array2::from_shape_vec((2, 3), vec![6.0, 0.0, 1.0, 0.0, 2.0, 6.0]).unwrap();
        let y2 = array![0usize, 1];

        fitted.partial_fit(&x2, &y2).unwrap();

        let preds = fitted.predict(&x1).unwrap();
        assert_eq!(preds.len(), 4);
    }

    #[test]
    fn test_complement_nb_partial_fit_shape_mismatch() {
        let (x, y) = make_count_data();
        let model = ComplementNB::<f64>::new();
        let mut fitted = model.fit(&x, &y).unwrap();

        let x_bad = Array2::from_shape_vec((2, 5), vec![1.0; 10]).unwrap();
        let y_bad = array![0usize, 1];
        assert!(fitted.partial_fit(&x_bad, &y_bad).is_err());
    }

    #[test]
    fn test_complement_nb_class_prior() {
        let (x, y) = make_count_data();
        let model = ComplementNB::<f64>::new().with_class_prior(vec![0.5, 0.5]);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 6);
    }

    #[test]
    fn test_complement_nb_class_prior_wrong_length() {
        let (x, y) = make_count_data();
        let model = ComplementNB::<f64>::new().with_class_prior(vec![1.0]);
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_complement_nb_three_classes() {
        let x = Array2::from_shape_vec(
            (9, 3),
            vec![
                5.0, 0.0, 0.0, 6.0, 0.0, 0.0, 4.0, 1.0, 0.0, 0.0, 5.0, 0.0, 0.0, 6.0, 0.0, 1.0,
                4.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 6.0, 0.0, 1.0, 4.0,
            ],
        )
        .unwrap();
        let y = array![0usize, 0, 0, 1, 1, 1, 2, 2, 2];

        let model = ComplementNB::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();
        assert_eq!(fitted.n_classes(), 3);
        let preds = fitted.predict(&x).unwrap();
        let correct = preds.iter().zip(y.iter()).filter(|(p, a)| p == a).count();
        assert!(correct >= 7);
    }

    // sklearn 1.5.2 oracle fixture (R-CHAR-3) for the REQ-9a fitted accessors.
    // X = [[5,1,0],[4,2,0],[6,0,1],[0,1,5],[1,0,4],[0,2,6]], y = [0,0,0,1,1,1].
    fn oracle_xy() -> (Array2<f64>, Array1<usize>) {
        let x = array![
            [5.0, 1.0, 0.0],
            [4.0, 2.0, 0.0],
            [6.0, 0.0, 1.0],
            [0.0, 1.0, 5.0],
            [1.0, 0.0, 4.0],
            [0.0, 2.0, 6.0],
        ];
        let y = array![0usize, 0, 0, 1, 1, 1];
        (x, y)
    }

    #[test]
    fn complement_feature_log_prob_and_count_match_sklearn() -> Result<(), FerroError> {
        // sklearn ComplementNB().fit(X, y):
        //   feature_log_prob_ = [[2.3978952728, 1.7047480922, 0.3184537311],
        //                        [0.3184537311, 1.7047480922, 2.3978952728]]
        //   feature_count_    = [[15, 3, 1], [1, 3, 15]]
        let (x, y) = oracle_xy();
        let fitted = ComplementNB::<f64>::new().fit(&x, &y)?;

        let expected_flp = array![
            [2.3978952728, 1.7047480922, 0.3184537311],
            [0.3184537311, 1.7047480922, 2.3978952728],
        ];
        let flp = fitted.feature_log_prob();
        assert_eq!(flp.dim(), (2, 3));
        for ((i, j), &e) in expected_flp.indexed_iter() {
            assert_relative_eq!(flp[[i, j]], e, epsilon = 1e-9);
        }

        let expected_fc = array![[15.0, 3.0, 1.0], [1.0, 3.0, 15.0]];
        let fc = fitted.feature_count();
        assert_eq!(fc.dim(), (2, 3));
        for ((i, j), &e) in expected_fc.indexed_iter() {
            assert_relative_eq!(fc[[i, j]], e, epsilon = 1e-9);
        }
        Ok(())
    }

    #[test]
    #[allow(
        clippy::approx_constant,
        reason = "literal -0.6931471806 is the sklearn class_log_prior_ oracle value ln(0.5), not a use of the LN_2 constant"
    )]
    fn complement_feature_all_class_count_prior_match_sklearn() -> Result<(), FerroError> {
        // sklearn ComplementNB().fit(X, y):
        //   feature_all_     = [16, 6, 16]
        //   class_count_     = [3, 3]
        //   class_log_prior_ = [-0.6931471806, -0.6931471806]
        let (x, y) = oracle_xy();
        let fitted = ComplementNB::<f64>::new().fit(&x, &y)?;

        let expected_fa = array![16.0, 6.0, 16.0];
        let fa = fitted.feature_all();
        assert_eq!(fa.len(), 3);
        for (i, &e) in expected_fa.iter().enumerate() {
            assert_relative_eq!(fa[i], e, epsilon = 1e-9);
        }

        let expected_cc = array![3.0, 3.0];
        let cc = fitted.class_count();
        assert_eq!(cc.len(), 2);
        for (i, &e) in expected_cc.iter().enumerate() {
            assert_relative_eq!(cc[i], e, epsilon = 1e-9);
        }

        let expected_clp = array![-0.6931471806, -0.6931471806];
        let clp = fitted.class_log_prior();
        assert_eq!(clp.len(), 2);
        for (i, &e) in expected_clp.iter().enumerate() {
            assert_relative_eq!(clp[i], e, epsilon = 1e-9);
        }
        Ok(())
    }

    // The `PipelineEstimator` adapter must preserve the ORIGINAL float labels:
    // sklearn `ComplementNB.fit` sets `classes_ = np.unique(y)` and `predict`
    // returns `classes_[argmax(jll)]` — the original labels, NOT class indices
    // (`naive_bayes.py:103`). Live sklearn 1.5.2 oracle (run from /tmp):
    //   X=[[3,0],[4,0],[0,3],[0,4]], y=[-1,-1,1,1], q=[[5,0],[0,5]]
    //   ComplementNB().fit(X,y).classes_  -> [-1, 1]
    //   ComplementNB().fit(X,y).predict(q) -> [-1, 1]   (NOT [0, 1])
    #[test]
    fn complement_pipeline_preserves_original_float_labels() -> Result<(), FerroError> {
        let x = array![[3.0, 0.0], [4.0, 0.0], [0.0, 3.0], [0.0, 4.0]];
        let y = array![-1.0, -1.0, 1.0, 1.0];
        let f = ComplementNB::<f64>::new().fit_pipeline(&x, &y)?;
        let p = f.predict_pipeline(&array![[5.0, 0.0], [0.0, 5.0]])?;
        // Original labels [-1.0, 1.0], not the collapsed indices [0.0, 1.0].
        assert_eq!(p, array![-1.0, 1.0]);
        Ok(())
    }
}
