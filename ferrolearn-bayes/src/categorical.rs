//! Categorical Naive Bayes classifier.
//!
//! This module provides [`CategoricalNB`], suitable for features that take on
//! one of K discrete categorical values (encoded as non-negative integers cast
//! to floats). Each feature column may have a different number of categories.
//!
//! The log-likelihood for feature `j` in class `c` taking value `k` is:
//!
//! ```text
//! log P(x_j = k | c) = log( (N_cjk + alpha) / (N_c + alpha * K_j) )
//! ```
//!
//! where `N_cjk` is the count of feature `j` equal to `k` in class `c`,
//! `N_c` is the total number of samples in class `c`, `K_j` is the number
//! of distinct categories for feature `j`, and `alpha` is the Laplace
//! smoothing parameter.
//!
//! # Examples
//!
//! ```
//! use ferrolearn_bayes::CategoricalNB;
//! use ferrolearn_core::{Fit, Predict};
//! use ndarray::{array, Array2};
//!
//! let x = Array2::from_shape_vec(
//!     (6, 3),
//!     vec![
//!         0.0, 1.0, 2.0,
//!         1.0, 0.0, 2.0,
//!         0.0, 1.0, 1.0,
//!         2.0, 0.0, 0.0,
//!         2.0, 1.0, 0.0,
//!         1.0, 0.0, 1.0,
//!     ],
//! ).unwrap();
//! let y = array![0usize, 0, 0, 1, 1, 1];
//!
//! let model = CategoricalNB::<f64>::new();
//! let fitted = model.fit(&x, &y).unwrap();
//! let preds = fitted.predict(&x).unwrap();
//! assert_eq!(preds.len(), 6);
//! ```
//!
//! # `## REQ status`
//!
//! Binary classification (R-DEFER-2): two states only — SHIPPED needs impl + a
//! non-test production consumer + green verification; NOT-STARTED carries the
//! open prereq blocker. **CategoricalNB has NO PyO3 binding** — there is no
//! `_RsCategoricalNB` in `ferrolearn-python/src/extras.rs` and no
//! `ferrolearn.CategoricalNB` (confirmed by grep: zero hits for `CategoricalNB`
//! under `ferrolearn-python/`, unlike its discrete-NB siblings
//! `_RsMultinomialNB` / `_RsBernoulliNB` / `_RsComplementNB`). The non-test
//! production consumer is therefore the in-crate `impl PipelineEstimator<F> for
//! CategoricalNB<F>` (`fn fit_pipeline`) plus `FittedCategoricalNBPipeline` (`fn
//! predict_pipeline`) — the `Box<dyn FittedPipelineEstimator<F>>`-producing
//! surface the same as `pipeline.rs` cites for GaussianNB/BernoulliNB. The
//! missing PyO3 binding is the REQ-9 gap (#923). Green verification = the
//! in-tree `categorical` lib tests plus the live-sklearn pins / guards
//! (`ferrolearn-bayes/tests/divergence_categorical.rs`): the two RED pins
//! `divergence_categorical_alpha_zero_allowed` (#921) and
//! `divergence_categorical_negative_features_rejected` (#922), now PASSING after
//! the fixes landed; plus the green guards `green_categorical_predict_value`,
//! `green_categorical_min_categories`,
//! `green_categorical_class_prior_length_only`,
//! `green_categorical_score_accuracy`. Cites use symbol anchors (ferrolearn) /
//! `file:line` (sklearn 1.5.2, commit 156ef14). Live oracle = installed sklearn
//! 1.5.2. (REQ numbering follows `.design/bayes/categorical.md`; blocker numbers
//! continue the bayes layer past complement #914-917.)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (`feature_log_prob_` smoothing + `_joint_log_likelihood` / `predict` / `predict_proba` / `predict_log_proba` / `predict_joint_log_proba` VALUE) | SHIPPED | `fn recompute_feature_log_prob` computes `((N_cjk + alpha) / (N_c + alpha*K_j)).ln()` per (feature, class, category) — the algebraic identity of `_update_feature_log_prob` (`naive_bayes.py:1498-1506`: `smoothed_cat_count = category_count_[i] + alpha`; `feature_log_prob_[i] = log(smoothed_cat_count) - log(smoothed_cat_count.sum(axis=1).reshape(-1,1))`, since `category_count_[i].sum(axis=1) == N_c`); `impl BaseNB::joint_log_likelihood` for `FittedCategoricalNB` computes `scores[[i,ci]] = class_log_prior[ci] + sum_j log_prob_for(j, ci, x[i,j])`, mirroring `jll += feature_log_prob_[i][:, X[:,i]].T; jll += class_log_prior_` (`naive_bayes.py:1508-1515`); the four `predict_*` delegate to the `BaseNB` provided methods. Non-test consumer: `impl PipelineEstimator for CategoricalNB` / `FittedCategoricalNBPipeline` (`fn fit_pipeline` / `fn predict_pipeline`). Verified: green guard `green_categorical_predict_value` — on `X=[[0,1],[1,0],[0,0],[2,1],[2,0],[1,1]]`, `y=[0,0,0,1,1,1]`, `q=[[0,0],[2,1]]`, sklearn `predict_proba(q) = [[0.8181818181818182, 0.18181818181818182], [0.18181818181818182, 0.8181818181818182]]`, `predict_joint_log_proba(q) = [[-1.8971199848858809, -3.401197381662155], [-3.401197381662155, -1.8971199848858809]]`, `predict(q) = [0, 1]`; ferrolearn matches to ≤1e-12. |
//! | REQ-2 (`alpha = 0` accepted + `alpha < 0` rejected) | SHIPPED | `fn fit` for `CategoricalNB` rejects ONLY `self.alpha < F::zero()` with `FerroError::InvalidParameter { name: "alpha", reason: "alpha must be >= 0 (sklearn Interval[0, inf))" }`, mirroring `CategoricalNB._parameter_constraints` which OVERRIDES `alpha` to `Interval(Real, 0, None, closed="left")` (`naive_bayes.py:1333`) — `alpha = 0` is INSIDE the closed-left interval so it is ACCEPTED (used as-is under the default `force_alpha=true`; only a divide-by-zero RuntimeWarning where a count is zero, NOT an error). Non-test consumer: `fit_pipeline` → `fit` propagates the `FerroError`. Verified: green pin `divergence_categorical_alpha_zero_allowed` (#921, now PASSING after the over-rejection fix): `with_alpha(0.0).fit(X,y)` returns `Ok` (sklearn `CategoricalNB(alpha=0.0).fit(X,y)` → "fit ok"); `test_categorical_nb_invalid_alpha_negative` confirms `with_alpha(-1.0).fit` still errors. |
//! | REQ-3 (negative-feature reject AT FIT) | SHIPPED (fit path) | `fn fit` rejects `x.iter().any(\|&v\| v < F::zero())` with `FerroError::InvalidParameter { name: "X", reason: "Negative values in data passed to CategoricalNB (input X)" }`, mirroring `_check_X_y` → `check_non_negative(X, "CategoricalNB (input X)")` (`naive_bayes.py:1435-1440`) → `ValueError`. Non-test consumer: `fit_pipeline` → `fit`. Verified: green pin `divergence_categorical_negative_features_rejected` (#922, now PASSING): `CategoricalNB().fit(X_with_neg, y)` returns `Err` (sklearn raises `ValueError("Negative values in data passed to CategoricalNB (input X)")`). GAP (NOT-STARTED, folded into #920): the reject landed in `fn fit` ONLY — `partial_fit` and the predict path still silently map a negative value to category 0 via `x[[i,j]].to_usize().unwrap_or(0)`. sklearn's predict-path `_check_X` runs `check_non_negative` too (`naive_bayes.py:1432`); the predict-path non-negative validation is part of #920 (the partial_fit negative-validation sub-gap is folded into #924). |
//! | REQ-4 (`min_categories` / `n_categories_` semantics) | SHIPPED | `fn fit` ensures `categories[j]` covers `0..min_cats[j]` (`MinCategories::Scalar` broadcast / `MinCategories::PerFeature` length-validated against `n_features`), mirroring `_validate_n_categories` (`naive_bayes.py:1446-1466`, `n_categories_ = max(X.max(0)+1, min_categories)`) + the `_count` `np.pad` padding (`naive_bayes.py:1491-1493`) so an allocated-but-unobserved category gets the smoothed `alpha/(N_c+alpha*K_j)` weight. Non-test consumer: `fit_pipeline` → `fit`. Verified: green guard `green_categorical_min_categories` — `CategoricalNB(min_categories=4).fit(X,y)` sklearn `n_categories_ = [4,4]`, `predict_joint_log_proba([[3,0]]) = [[-3.4863551900024623, -3.891820298110627]]`, `predict_proba([[3,0]]) = [[0.6000000000000001, 0.39999999999999997]]`, `predict([[3,0]]) = [0]`; ferrolearn `with_min_categories(4)` matches to ≤1e-12. (Scalar path only; a category `>= n_categories_` at predict is the REQ-? unseen-category divergence #920.) |
//! | REQ-5 (`class_log_prior_` empirical / uniform + `class_prior` LENGTH-only — MATCH) | SHIPPED | `fn resolve_class_log_prior` validates ONLY `priors.len() != n_classes` (then `p.ln()`), else empirical `ln(count_c / total)` (`fit_prior`), else uniform `(1/n_classes).ln()` — mirroring `_update_class_log_prior` (`naive_bayes.py:580-602`: LENGTH-only check `:589-591`, empirical `log(class_count_) - log(class_count_.sum())` `:600`, uniform `-log(n_classes)` `:602`). Discrete NB has NO sum-to-1 / non-negativity check — a deliberate MATCH. Non-test consumer: `fit_pipeline` → `fit` (the `fit_prior=true` empirical path; `with_class_prior` exercised in-crate). Verified: green guard `green_categorical_class_prior_length_only` — `with_class_prior([0.5,0.3]).fit(X,y)` SUCCEEDS (sum 0.8; sklearn `class_log_prior_ = log([0.5,0.3]) = [-0.6931471805599453, -1.2039728043259361]`, NO error); `test_categorical_nb_default` covers the empirical path. (Wrong-length error TYPE differs — `InvalidParameter` vs sklearn `ValueError("Number of priors must match number of classes.")` — folded into REQ-9's surface gap.) |
//! | REQ-6 (`force_alpha` floor + `fit_prior` toggle + `score`) | SHIPPED | `fn fit` calls `crate::clamp_alpha(self.alpha, self.force_alpha)` (`base::check_alpha`, the `_check_alpha` `1e-10` floor unless `force_alpha`, `naive_bayes.py:604-626`); `fn resolve_class_log_prior` selects empirical vs uniform on `fit_prior`; `pub fn score` returns mean accuracy (`ClassifierMixin.score` analog). Non-test consumer: `fit_pipeline` passes `force_alpha`/`fit_prior` through the builder. Verified: green guard `green_categorical_score_accuracy` — `CategoricalNB().fit(X,y).score(X,y) = 1.0` (sklearn oracle) on the separable fixture; `clamp_alpha(1, true) = 1`. In-tree `test_categorical_nb_alpha_smoothing_effect`. |
//! | REQ-7 (`partial_fit` VALUE — same-categories/classes path) | SHIPPED (same-categories path) | `FittedCategoricalNB::partial_fit` increments the per-(feature, class, category) `category_counts` / `class_counts` for existing categories/classes, then recomputes `feature_log_prob` (same `recompute_feature_log_prob` smoothing) and `class_log_prior` (`resolve_class_log_prior`), mirroring the shared `_BaseDiscreteNB.partial_fit` accumulate-then-recompute (`naive_bayes.py:628-709` → `_count` → `_update_feature_log_prob` / `_update_class_log_prior`). Non-test consumer: in-crate (the PyO3 `partial_fit` surface is REQ-9). Verified: in-tree fit-then-predict coverage on within-fitted data. EXTENSION + GAPS (NOT-STARTED, folded into #924): `partial_fit` APPENDS never-seen category values to `categories[j]` and INSERTS never-seen class labels into `classes` (a non-sklearn flexibility — documented in the method doc-comment; sklearn keeps `n_categories_` fixed and `IndexError`s a category `>= n_categories_`, and binarizes against the full `classes=` list); `partial_fit` also has NO `sample_weight` and NO negative-feature validation. |
//! | REQ-8 (unseen-category at predict + predict-path negative validation) | NOT-STARTED | open prereq blocker **#920**. sklearn requires category indices `< n_categories_[i]`; the `_joint_log_likelihood` fancy-index `feature_log_prob_[i][:, X[:,i]]` (`naive_bayes.py:1513`) raises `IndexError("index 5 is out of bounds for axis 1 with size 2")` for an index `>= n_categories_`, and `_check_X` runs `check_non_negative` on the predict path (`naive_bayes.py:1432`). `fn log_prob_for` returns a uniform `(1/(n_known_cats+1)).ln()` fallback for any unknown category — NO error (`predict([[5,0]])` → `[0]`, `predict_proba` → `[[0.5,0.5]]`); the predict path also maps a negative value to 0 via `to_usize().unwrap_or(0)` with no guard. Matching sklearn's `IndexError` / predict-path reject requires threading a `Result` through `log_prob_for` / `joint_log_likelihood` — not a one-line fix. |
//! | REQ-9 (fitted-attribute accessors + PyO3 surface) | NOT-STARTED | open prereq blocker **#923**. sklearn exposes `category_count_` / `feature_log_prob_` / `class_count_` / `class_log_prior_` / `n_categories_` / `classes_` / `n_features_in_` (`naive_bayes.py:1266-1303`). `FittedCategoricalNB` exposes ONLY `classes()` (via `HasClasses`); `feature_log_prob` / `category_counts` / `categories` / `class_log_prior` / `class_counts` are PRIVATE fields with no accessor. **CategoricalNB has NO PyO3 binding** — no `_RsCategoricalNB` in `ferrolearn-python/src/extras.rs`, no `ferrolearn.CategoricalNB` (grep confirms zero `CategoricalNB` hits under `ferrolearn-python/`), so `import ferrolearn` cannot reach it. Also subsumes the wrong-length `class_prior` TYPE sub-item (REQ-5) and the negative-feature MESSAGE/TYPE-parity sub-item (REQ-3). The fix belongs in `ferrolearn-python` (multi-file). |
//! | REQ-10 (`sample_weight` + `partial_fit` new-category/new-class extension + negative validation) | NOT-STARTED | open prereq blocker **#924**. sklearn `fit(X, y, sample_weight=None)` (`naive_bayes.py:1353`, `:712`) weights the binarized `Y` so `class_count_ = Y.sum(axis=0)` and the `np.bincount(X_feature[mask], weights=...)` per-category counts become weighted (`naive_bayes.py:1468-1496`). ferrolearn's `impl Fit<Array2<F>, Array1<usize>>` is `fn fit(&self, x, y)` — NO `sample_weight` on `fit` or `partial_fit`; also the `partial_fit` new-category/new-class EXTENSION (R-DEV-7 deviation) and `partial_fit` negative-validation gap carved out of REQ-7 live here. |
//! | REQ-11 (ferray substrate) | NOT-STARTED | open prereq blocker **#925**. `categorical.rs` imports `ndarray::{Array1, Array2}` + `num_traits::{Float, FromPrimitive, ToPrimitive}` (the wrong substrate, R-SUBSTRATE-1); not migrated to `ferray-core`. |

use crate::base::BaseNB;
use ferrolearn_core::error::FerroError;
use ferrolearn_core::introspection::HasClasses;
use ferrolearn_core::pipeline::{FittedPipelineEstimator, PipelineEstimator};
use ferrolearn_core::traits::{Fit, Predict};
use ndarray::{Array1, Array2};
use num_traits::{Float, FromPrimitive, ToPrimitive};
use std::marker::PhantomData;

/// Specification of the minimum number of categories per feature.
///
/// Used by [`CategoricalNB::with_min_categories`] /
/// [`CategoricalNB::with_min_categories_per_feature`] to ensure the count
/// tables have enough slots for categories that may not appear in the
/// training data but could appear at predict / partial_fit time.
#[derive(Debug, Clone)]
pub enum MinCategories {
    /// Same minimum count broadcast across every feature.
    Scalar(usize),
    /// Explicit minimum count per feature; length must equal `n_features`
    /// at fit time.
    PerFeature(Vec<usize>),
}

/// Categorical Naive Bayes classifier.
///
/// Suitable for features where each column takes on one of several discrete
/// categorical values (encoded as non-negative integers cast to floats).
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
#[derive(Debug, Clone)]
pub struct CategoricalNB<F: Float + Send + Sync + 'static> {
    /// Additive (Laplace) smoothing parameter. Default: `1.0`.
    alpha: F,
    /// Optional user-supplied class priors. If set, these are used
    /// instead of computing priors from the data.
    class_prior: Option<Vec<F>>,
    /// Whether to learn class priors from the data. When `false` and
    /// `class_prior` is `None`, uniform priors `1 / n_classes` are used.
    /// Default: `true`.
    fit_prior: bool,
    /// When `false`, `alpha` values below `1e-10` are silently raised to
    /// `1e-10` (legacy behavior). Default: `true`.
    force_alpha: bool,
    /// Optional minimum number of categories per feature. When set, the
    /// fitted count tables are sized to `max(observed_max + 1, min_cats[j])`,
    /// pre-populating slots for category values that might appear at
    /// predict / `partial_fit` time but were not in the training data.
    min_categories: Option<MinCategories>,
    _marker: PhantomData<F>,
}

impl<F: Float + Send + Sync + 'static> CategoricalNB<F> {
    /// Create a new `CategoricalNB` with default settings.
    ///
    /// Defaults: `alpha = 1.0`, `class_prior = None`, `fit_prior = true`,
    /// `force_alpha = true`, `min_categories = None`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            alpha: F::one(),
            class_prior: None,
            fit_prior: true,
            force_alpha: true,
            min_categories: None,
            _marker: PhantomData,
        }
    }

    /// Set the Laplace smoothing parameter.
    ///
    /// Invalid alpha values are caught at `fit()` time.
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

    /// Set the same minimum category count for every feature.
    #[must_use]
    pub fn with_min_categories(mut self, min: usize) -> Self {
        self.min_categories = Some(MinCategories::Scalar(min));
        self
    }

    /// Set per-feature minimum category counts. Length must match
    /// `n_features` at fit time.
    #[must_use]
    pub fn with_min_categories_per_feature(mut self, mins: Vec<usize>) -> Self {
        self.min_categories = Some(MinCategories::PerFeature(mins));
        self
    }
}

impl<F: Float + Send + Sync + 'static> Default for CategoricalNB<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted Categorical Naive Bayes classifier.
///
/// Stores the per-class log prior, the raw per-(feature, class, category)
/// integer counts, and the cached log probabilities derived from them.
/// `partial_fit` increments the counts and recomputes log probabilities.
#[derive(Debug, Clone)]
pub struct FittedCategoricalNB<F: Float + Send + Sync + 'static> {
    /// Sorted unique class labels.
    classes: Vec<usize>,
    /// Log prior probability for each class, length `n_classes`.
    class_log_prior: Vec<F>,
    /// For each feature i and class c, store log P(x_i = k | c) for each
    /// category k. Indexed as `feature_log_prob[feature_idx][class_idx][category_idx]`.
    /// Categories are mapped from their integer value to a contiguous index via
    /// `categories[feature_idx]`. Cached; recomputed by `recompute_log_prob`.
    feature_log_prob: Vec<Vec<Vec<F>>>,
    /// Raw per-(feature, class, category) integer counts. Same shape as
    /// `feature_log_prob`. Used by `partial_fit` to incrementally update.
    category_counts: Vec<Vec<Vec<usize>>>,
    /// For each feature, the sorted list of known category values.
    /// `categories[feature_idx]` is a `Vec<usize>` of category integer values.
    /// May contain min_categories padding values that were not in the
    /// training data.
    categories: Vec<Vec<usize>>,
    /// Per-class sample counts. Index aligns with `classes`.
    class_counts: Vec<usize>,
    /// Number of features the model was fitted on.
    n_features: usize,
    /// Smoothing parameter (post-clamp), carried for `partial_fit`.
    alpha: F,
    /// Optional explicit class priors carried for prior recomputation.
    class_prior: Option<Vec<F>>,
    /// Whether to refit empirical priors during `partial_fit`.
    fit_prior: bool,
}

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, Array1<usize>> for CategoricalNB<F> {
    type Fitted = FittedCategoricalNB<F>;
    type Error = FerroError;

    /// Fit the Categorical NB model.
    ///
    /// # Errors
    ///
    /// - [`FerroError::ShapeMismatch`] if `x` and `y` have different numbers of rows.
    /// - [`FerroError::InsufficientSamples`] if there are no samples.
    /// - [`FerroError::InvalidParameter`] if `alpha < 0`.
    fn fit(&self, x: &Array2<F>, y: &Array1<usize>) -> Result<FittedCategoricalNB<F>, FerroError> {
        let (n_samples, n_features) = x.dim();

        if n_samples == 0 {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "CategoricalNB requires at least one sample".into(),
            });
        }

        if n_samples != y.len() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_samples],
                actual: vec![y.len()],
                context: "y length must match number of samples in X".into(),
            });
        }

        if self.alpha < F::zero() {
            return Err(FerroError::InvalidParameter {
                name: "alpha".into(),
                reason: "alpha must be >= 0 (sklearn Interval[0, inf))".into(),
            });
        }
        let alpha = crate::clamp_alpha(self.alpha, self.force_alpha);

        // Reject negative feature values (sklearn `CategoricalNB._check_X_y`
        // calls `check_non_negative(X, "CategoricalNB (input X)")`,
        // naive_bayes.py:1435-1440). Without this guard, a negative value
        // would silently map to category 0 via `to_usize().unwrap_or(0)`.
        if x.iter().any(|&v| v < F::zero()) {
            return Err(FerroError::InvalidParameter {
                name: "X".into(),
                reason: "Negative values in data passed to CategoricalNB (input X)".into(),
            });
        }

        // Validate min_categories shape against n_features (only relevant
        // for PerFeature; Scalar broadcasts to all).
        if let Some(MinCategories::PerFeature(ref mins)) = self.min_categories
            && mins.len() != n_features
        {
            return Err(FerroError::InvalidParameter {
                name: "min_categories".into(),
                reason: format!(
                    "PerFeature length {} does not match n_features {}",
                    mins.len(),
                    n_features
                ),
            });
        }

        // Collect sorted unique classes.
        let mut classes: Vec<usize> = y.to_vec();
        classes.sort_unstable();
        classes.dedup();
        let n_classes = classes.len();

        // Build per-class sample counts and indices.
        let mut class_counts = vec![0usize; n_classes];
        let mut class_indices: Vec<Vec<usize>> = vec![Vec::new(); n_classes];
        for (sample_idx, &label) in y.iter().enumerate() {
            let ci = classes.iter().position(|&c| c == label).unwrap();
            class_counts[ci] += 1;
            class_indices[ci].push(sample_idx);
        }

        // For each feature, discover categories and build raw count tables.
        let mut category_counts: Vec<Vec<Vec<usize>>> = Vec::with_capacity(n_features);
        let mut categories_per_feature: Vec<Vec<usize>> = Vec::with_capacity(n_features);

        for j in 0..n_features {
            // Discover observed unique category values for this feature.
            let mut cats: Vec<usize> = Vec::new();
            for i in 0..n_samples {
                let val = x[[i, j]].to_usize().unwrap_or(0);
                cats.push(val);
            }
            cats.sort_unstable();
            cats.dedup();

            // Apply min_categories: ensure cats covers 0..min_cats[j] so the
            // count tables have slots even for unobserved category indices.
            let min_cats_j = match self.min_categories {
                Some(MinCategories::Scalar(m)) => m,
                Some(MinCategories::PerFeature(ref v)) => v[j],
                None => 0,
            };
            if min_cats_j > 0 {
                for cv in 0..min_cats_j {
                    if cats.binary_search(&cv).is_err() {
                        let pos = cats.partition_point(|&c| c < cv);
                        cats.insert(pos, cv);
                    }
                }
            }

            // Per-class, per-category count table.
            let mut counts_for_feature: Vec<Vec<usize>> = vec![vec![0usize; cats.len()]; n_classes];
            for (ci, indices) in class_indices.iter().enumerate() {
                for &sample_idx in indices {
                    let val = x[[sample_idx, j]].to_usize().unwrap_or(0);
                    if let Ok(cat_idx) = cats.binary_search(&val) {
                        counts_for_feature[ci][cat_idx] += 1;
                    }
                }
            }

            category_counts.push(counts_for_feature);
            categories_per_feature.push(cats);
        }

        // Cached log probabilities derived from counts.
        let feature_log_prob = recompute_feature_log_prob(&category_counts, &class_counts, alpha);

        // Resolve priors.
        let class_log_prior =
            resolve_class_log_prior(&class_counts, n_classes, &self.class_prior, self.fit_prior)?;

        Ok(FittedCategoricalNB {
            classes,
            class_log_prior,
            feature_log_prob,
            category_counts,
            categories: categories_per_feature,
            class_counts,
            n_features,
            alpha,
            class_prior: self.class_prior.clone(),
            fit_prior: self.fit_prior,
        })
    }
}

/// Recompute `feature_log_prob[j][c][k]` from raw `category_counts` and
/// `class_counts`, using Laplace smoothing.
#[allow(clippy::needless_range_loop)] // matrix-style triple indexing reads cleaner than nested .iter().enumerate()
fn recompute_feature_log_prob<F: Float>(
    category_counts: &[Vec<Vec<usize>>],
    class_counts: &[usize],
    alpha: F,
) -> Vec<Vec<Vec<F>>> {
    let n_features = category_counts.len();
    let n_classes = class_counts.len();
    let mut out: Vec<Vec<Vec<F>>> = Vec::with_capacity(n_features);
    for j in 0..n_features {
        let n_cats = category_counts[j].first().map_or(0, Vec::len);
        let n_cats_f = F::from(n_cats).unwrap();
        let mut per_class: Vec<Vec<F>> = Vec::with_capacity(n_classes);
        for ci in 0..n_classes {
            let n_c_f = F::from(class_counts[ci]).unwrap();
            let denom = n_c_f + alpha * n_cats_f;
            let mut row: Vec<F> = Vec::with_capacity(n_cats);
            for k in 0..n_cats {
                let count_f = F::from(category_counts[j][ci][k]).unwrap();
                row.push(((count_f + alpha) / denom).ln());
            }
            per_class.push(row);
        }
        out.push(per_class);
    }
    out
}

/// Resolve `class_log_prior` from `class_counts`, honoring an optional
/// explicit `class_prior` and the `fit_prior` flag.
#[allow(clippy::needless_range_loop)] // index-by-class is the natural loop here
fn resolve_class_log_prior<F: Float>(
    class_counts: &[usize],
    n_classes: usize,
    class_prior: &Option<Vec<F>>,
    fit_prior: bool,
) -> Result<Vec<F>, FerroError> {
    let mut out = vec![F::zero(); n_classes];
    if let Some(priors) = class_prior {
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
            out[ci] = p.ln();
        }
    } else if fit_prior {
        let total: usize = class_counts.iter().sum();
        let total_f = F::from(total).unwrap();
        for (ci, &c) in class_counts.iter().enumerate() {
            out[ci] = (F::from(c).unwrap() / total_f).ln();
        }
    } else {
        let uniform = (F::one() / F::from(n_classes).unwrap()).ln();
        for ci in 0..n_classes {
            out[ci] = uniform;
        }
    }
    Ok(out)
}

impl<F: Float + Send + Sync + 'static> FittedCategoricalNB<F> {
    /// Incrementally update the model with new data.
    ///
    /// Increments the per-(feature, class, category) counts and recomputes
    /// the cached log probabilities. Unlike sklearn's strict integer-index
    /// model, ferrolearn's CategoricalNB allows new category values to
    /// appear at `partial_fit` time — they are appended to `categories[j]`
    /// and a new count slot is allocated for every class.
    ///
    /// New class labels not seen at `fit` time are likewise added.
    ///
    /// # Errors
    ///
    /// - [`FerroError::ShapeMismatch`] if `x` and `y` have different row
    ///   counts or the feature count does not match the fitted model.
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

        if n_features != self.n_features {
            return Err(FerroError::ShapeMismatch {
                expected: vec![self.n_features],
                actual: vec![n_features],
                context: "number of features must match fitted CategoricalNB".into(),
            });
        }

        // Discover and integrate any new class labels (preserving sorted
        // order so `classes` stays a sorted unique list).
        for &label in y {
            if self.classes.binary_search(&label).is_err() {
                let pos = self.classes.partition_point(|&c| c < label);
                self.classes.insert(pos, label);
                self.class_counts.insert(pos, 0);
                self.class_log_prior.insert(pos, F::neg_infinity());
                for j in 0..self.n_features {
                    let n_cats = self.categories[j].len();
                    self.category_counts[j].insert(pos, vec![0usize; n_cats]);
                    self.feature_log_prob[j].insert(pos, vec![F::zero(); n_cats]);
                }
            }
        }

        // Walk every sample, extending `categories[j]` for unseen values.
        for sample_idx in 0..n_samples {
            let label = y[sample_idx];
            let ci = self.classes.binary_search(&label).unwrap();
            self.class_counts[ci] += 1;
            for j in 0..self.n_features {
                let val = x[[sample_idx, j]].to_usize().unwrap_or(0);
                let cat_idx = match self.categories[j].binary_search(&val) {
                    Ok(idx) => idx,
                    Err(insert_pos) => {
                        // New category: insert into categories[j] and add a
                        // count slot for every class.
                        self.categories[j].insert(insert_pos, val);
                        for c in 0..self.classes.len() {
                            self.category_counts[j][c].insert(insert_pos, 0);
                        }
                        insert_pos
                    }
                };
                self.category_counts[j][ci][cat_idx] += 1;
            }
        }

        // Recompute cached log probabilities and class priors.
        self.feature_log_prob =
            recompute_feature_log_prob(&self.category_counts, &self.class_counts, self.alpha);
        self.class_log_prior = resolve_class_log_prior(
            &self.class_counts,
            self.classes.len(),
            &self.class_prior,
            self.fit_prior,
        )?;

        Ok(())
    }

    /// Look up the log probability for a given feature, class, and category value.
    ///
    /// If the category value was not seen during training, returns a uniform
    /// log probability based on the number of known categories for that feature.
    fn log_prob_for(&self, feature_idx: usize, class_idx: usize, cat_value: usize) -> F {
        let cats = &self.categories[feature_idx];
        if let Ok(cat_idx) = cats.binary_search(&cat_value) {
            self.feature_log_prob[feature_idx][class_idx][cat_idx]
        } else {
            // Unseen category: use uniform probability 1 / (n_known_cats + 1).
            // This gracefully degrades for unseen categories.
            let n_cats_plus_one = F::from(cats.len() + 1).unwrap();
            (F::one() / n_cats_plus_one).ln()
        }
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
    /// `CategoricalNB._joint_log_likelihood`. Delegates to
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

impl<F: Float + Send + Sync + 'static> BaseNB<F> for FittedCategoricalNB<F> {
    /// Compute joint log-likelihood for each class — sklearn
    /// `CategoricalNB._joint_log_likelihood`.
    ///
    /// Returns shape `(n_samples, n_classes)`.
    fn joint_log_likelihood(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        if x.ncols() != self.n_features {
            return Err(FerroError::ShapeMismatch {
                expected: vec![self.n_features],
                actual: vec![x.ncols()],
                context: "number of features must match fitted CategoricalNB".into(),
            });
        }

        let n_samples = x.nrows();
        let n_classes = self.classes.len();

        let mut scores = Array2::<F>::zeros((n_samples, n_classes));

        for i in 0..n_samples {
            for ci in 0..n_classes {
                let mut score = self.class_log_prior[ci];
                for j in 0..self.n_features {
                    let cat_value = x[[i, j]].to_usize().unwrap_or(0);
                    score = score + self.log_prob_for(j, ci, cat_value);
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

impl<F: Float + Send + Sync + 'static> Predict<Array2<F>> for FittedCategoricalNB<F> {
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

impl<F: Float + Send + Sync + 'static> HasClasses for FittedCategoricalNB<F> {
    fn classes(&self) -> &[usize] {
        &self.classes
    }

    fn n_classes(&self) -> usize {
        self.classes.len()
    }
}

// Pipeline integration.
impl<F: Float + ToPrimitive + FromPrimitive + Send + Sync + 'static> PipelineEstimator<F>
    for CategoricalNB<F>
{
    fn fit_pipeline(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
    ) -> Result<Box<dyn FittedPipelineEstimator<F>>, FerroError> {
        let y_usize: Array1<usize> = y.mapv(|v| v.to_usize().unwrap_or(0));
        let fitted = self.fit(x, &y_usize)?;
        Ok(Box::new(FittedCategoricalNBPipeline(fitted)))
    }
}

struct FittedCategoricalNBPipeline<F: Float + Send + Sync + 'static>(FittedCategoricalNB<F>);

unsafe impl<F: Float + Send + Sync + 'static> Send for FittedCategoricalNBPipeline<F> {}
unsafe impl<F: Float + Send + Sync + 'static> Sync for FittedCategoricalNBPipeline<F> {}

impl<F: Float + ToPrimitive + FromPrimitive + Send + Sync + 'static> FittedPipelineEstimator<F>
    for FittedCategoricalNBPipeline<F>
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

    fn make_categorical_data() -> (Array2<f64>, Array1<usize>) {
        // Categorical features: 3 features, each taking values in {0, 1, 2}.
        // Class 0 tends to have low values, class 1 tends to have high values.
        let x = Array2::from_shape_vec(
            (8, 3),
            vec![
                0.0, 0.0, 1.0, // class 0
                1.0, 0.0, 0.0, // class 0
                0.0, 1.0, 0.0, // class 0
                0.0, 0.0, 0.0, // class 0
                2.0, 2.0, 1.0, // class 1
                2.0, 1.0, 2.0, // class 1
                1.0, 2.0, 2.0, // class 1
                2.0, 2.0, 2.0, // class 1
            ],
        )
        .unwrap();
        let y = array![0usize, 0, 0, 0, 1, 1, 1, 1];
        (x, y)
    }

    #[test]
    fn test_categorical_nb_fit_predict() {
        let (x, y) = make_categorical_data();
        let model = CategoricalNB::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        let correct = preds.iter().zip(y.iter()).filter(|(p, a)| p == a).count();
        // Should get most or all correct on training data.
        assert!(correct >= 6, "expected at least 6 correct, got {correct}");
    }

    #[test]
    fn test_categorical_nb_predict_proba_sums_to_one() {
        let (x, y) = make_categorical_data();
        let model = CategoricalNB::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();
        let proba = fitted.predict_proba(&x).unwrap();
        assert_eq!(proba.nrows(), 8);
        assert_eq!(proba.ncols(), 2);
        for i in 0..proba.nrows() {
            assert_relative_eq!(proba.row(i).sum(), 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_categorical_nb_has_classes() {
        let (x, y) = make_categorical_data();
        let model = CategoricalNB::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();
        assert_eq!(fitted.classes(), &[0, 1]);
        assert_eq!(fitted.n_classes(), 2);
    }

    #[test]
    fn test_categorical_nb_alpha_smoothing_effect() {
        let (x, y) = make_categorical_data();

        // With small alpha, sharper probabilities.
        let model_sharp = CategoricalNB::<f64>::new().with_alpha(0.01);
        let fitted_sharp = model_sharp.fit(&x, &y).unwrap();
        let proba_sharp = fitted_sharp.predict_proba(&x).unwrap();

        // With large alpha, smoother probabilities (closer to uniform).
        let model_smooth = CategoricalNB::<f64>::new().with_alpha(100.0);
        let fitted_smooth = model_smooth.fit(&x, &y).unwrap();
        let proba_smooth = fitted_smooth.predict_proba(&x).unwrap();

        // Smoothed probabilities for the dominant class on a class-0 sample
        // should be less extreme.
        let sharp_max = proba_sharp[[0, 0]].max(proba_sharp[[0, 1]]);
        let smooth_max = proba_smooth[[0, 0]].max(proba_smooth[[0, 1]]);
        assert!(smooth_max < sharp_max);
    }

    #[test]
    fn test_categorical_nb_alpha_zero_allowed() {
        // sklearn `CategoricalNB._parameter_constraints` overrides `alpha` to
        // `Interval(Real, 0, None, closed="left")` (naive_bayes.py:1333) — 0 is
        // INSIDE the interval, so `alpha = 0` is ACCEPTED at fit (used as-is
        // under the default `force_alpha=True`; only a divide-by-zero
        // RuntimeWarning where a count is zero, NOT an error). Only `alpha < 0`
        // is rejected.
        let (x, y) = make_categorical_data();
        let model = CategoricalNB::<f64>::new().with_alpha(0.0);
        let result = model.fit(&x, &y);
        assert!(
            result.is_ok(),
            "alpha=0 should be allowed (sklearn accepts)"
        );
        // predict still runs on the fitted model.
        if let Ok(fitted) = result {
            let preds = fitted.predict(&x);
            assert!(preds.is_ok());
        }
    }

    #[test]
    fn test_categorical_nb_invalid_alpha_negative() {
        let (x, y) = make_categorical_data();
        let model = CategoricalNB::<f64>::new().with_alpha(-1.0);
        let result = model.fit(&x, &y);
        assert!(result.is_err());
        match result.unwrap_err() {
            FerroError::InvalidParameter { name, .. } => assert_eq!(name, "alpha"),
            e => panic!("expected InvalidParameter, got {e:?}"),
        }
    }

    #[test]
    fn test_categorical_nb_shape_mismatch_fit() {
        let x = Array2::from_shape_vec((4, 3), vec![0.0; 12]).unwrap();
        let y = array![0usize, 1]; // Wrong length
        let model = CategoricalNB::<f64>::new();
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_categorical_nb_shape_mismatch_predict() {
        let (x, y) = make_categorical_data();
        let model = CategoricalNB::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();
        let x_bad = Array2::from_shape_vec((3, 5), vec![0.0; 15]).unwrap();
        assert!(fitted.predict(&x_bad).is_err());
        assert!(fitted.predict_proba(&x_bad).is_err());
    }

    #[test]
    fn test_categorical_nb_empty_data() {
        let x = Array2::<f64>::zeros((0, 3));
        let y = Array1::<usize>::zeros(0);
        let model = CategoricalNB::<f64>::new();
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_categorical_nb_single_class() {
        let x = Array2::from_shape_vec((3, 2), vec![0.0, 1.0, 1.0, 0.0, 0.0, 1.0]).unwrap();
        let y = array![2usize, 2, 2];
        let model = CategoricalNB::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();
        assert_eq!(fitted.classes(), &[2]);
        let preds = fitted.predict(&x).unwrap();
        assert!(preds.iter().all(|&p| p == 2));
    }

    #[test]
    fn test_categorical_nb_default() {
        let model = CategoricalNB::<f64>::default();
        assert_relative_eq!(model.alpha, 1.0, epsilon = 1e-15);
    }

    #[test]
    fn test_categorical_nb_unseen_category() {
        // Fit on categories {0, 1}, then predict with a sample containing
        // category 5 (unseen). Should not panic, should return valid probabilities.
        let x = Array2::from_shape_vec(
            (4, 2),
            vec![
                0.0, 0.0, // class 0
                0.0, 1.0, // class 0
                1.0, 0.0, // class 1
                1.0, 1.0, // class 1
            ],
        )
        .unwrap();
        let y = array![0usize, 0, 1, 1];

        let model = CategoricalNB::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();

        // Predict with unseen category 5 in feature 0.
        let x_new = Array2::from_shape_vec((1, 2), vec![5.0, 0.0]).unwrap();
        let preds = fitted.predict(&x_new).unwrap();
        assert_eq!(preds.len(), 1);

        let proba = fitted.predict_proba(&x_new).unwrap();
        assert_relative_eq!(proba.row(0).sum(), 1.0, epsilon = 1e-10);
        // Both probabilities should be between 0 and 1.
        assert!(proba[[0, 0]] > 0.0 && proba[[0, 0]] < 1.0);
        assert!(proba[[0, 1]] > 0.0 && proba[[0, 1]] < 1.0);
    }

    #[test]
    fn test_categorical_nb_three_classes() {
        let x = Array2::from_shape_vec(
            (9, 2),
            vec![
                0.0, 0.0, // class 0
                0.0, 0.0, // class 0
                0.0, 0.0, // class 0
                1.0, 1.0, // class 1
                1.0, 1.0, // class 1
                1.0, 1.0, // class 1
                2.0, 2.0, // class 2
                2.0, 2.0, // class 2
                2.0, 2.0, // class 2
            ],
        )
        .unwrap();
        let y = array![0usize, 0, 0, 1, 1, 1, 2, 2, 2];

        let model = CategoricalNB::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();
        assert_eq!(fitted.n_classes(), 3);
        assert_eq!(fitted.classes(), &[0, 1, 2]);

        let preds = fitted.predict(&x).unwrap();
        let correct = preds.iter().zip(y.iter()).filter(|(p, a)| p == a).count();
        assert_eq!(correct, 9);
    }

    #[test]
    fn test_categorical_nb_pipeline() {
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 2.0, 1.0, 2.0, 0.0],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
        let model = CategoricalNB::<f64>::new();
        let fitted = model.fit_pipeline(&x, &y).unwrap();
        let preds = fitted.predict_pipeline(&x).unwrap();
        assert_eq!(preds.len(), 6);
    }

    #[test]
    fn test_categorical_nb_predict_proba_ordering() {
        let (x, y) = make_categorical_data();
        let model = CategoricalNB::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();
        let proba = fitted.predict_proba(&x).unwrap();

        // First 4 samples are class 0 — class 0 probability should be higher.
        for i in 0..4 {
            assert!(
                proba[[i, 0]] > proba[[i, 1]],
                "sample {i}: P(c=0)={} should be > P(c=1)={}",
                proba[[i, 0]],
                proba[[i, 1]]
            );
        }
        // Last 4 samples are class 1 — class 1 probability should be higher.
        for i in 4..8 {
            assert!(
                proba[[i, 1]] > proba[[i, 0]],
                "sample {i}: P(c=1)={} should be > P(c=0)={}",
                proba[[i, 1]],
                proba[[i, 0]]
            );
        }
    }

    #[test]
    fn test_categorical_nb_f32() {
        let x = Array2::from_shape_vec((4, 2), vec![0.0f32, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0])
            .unwrap();
        let y = array![0usize, 0, 1, 1];
        let model = CategoricalNB::<f32>::new();
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 4);

        let proba = fitted.predict_proba(&x).unwrap();
        for i in 0..proba.nrows() {
            let sum: f32 = proba.row(i).sum();
            assert!((sum - 1.0f32).abs() < 1e-5);
        }
    }

    #[test]
    fn test_categorical_nb_unordered_classes() {
        // Classes are not 0..n, and not contiguous.
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 2.0, 1.0, 2.0, 0.0],
        )
        .unwrap();
        let y = array![5usize, 5, 5, 10, 10, 10];
        let model = CategoricalNB::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();
        assert_eq!(fitted.classes(), &[5, 10]);

        let preds = fitted.predict(&x).unwrap();
        // First 3 should predict class 5.
        for i in 0..3 {
            assert_eq!(preds[i], 5);
        }
        // Last 3 should predict class 10.
        for i in 3..6 {
            assert_eq!(preds[i], 10);
        }
    }
}
