//! Ridge classifier with built-in cross-validation for alpha selection.
//!
//! This module provides [`RidgeClassifierCV`], the cross-validated variant of
//! [`crate::RidgeClassifier`]. It mirrors scikit-learn's
//! `class RidgeClassifierCV` (`sklearn/linear_model/_ridge.py:2676`): the target
//! is binarized with a `LabelBinarizer(pos_label=1, neg_label=-1)` (binary →
//! single `{-1, +1}` column, multiclass → one-hot `{-1, +1}` columns), and a
//! SHARED regularization strength `alpha` is selected by efficient leave-one-out
//! Generalized Cross-Validation over the binarized multi-target problem
//! (`_RidgeGCV`, `_ridge.py:1688`). For `scoring=None` (the default), sklearn
//! scores each candidate by `-squared_errors.mean()` where
//! `squared_errors = (c / G_inverse_diag) ** 2` is shape `(n_samples, n_y)`
//! (`_ridge.py:2148-2150` + `_score_without_scorer`, `_ridge.py:2211-2218`):
//! the closed-form LOO errors are summed over BOTH samples and indicator
//! columns, sharing a single matrix decomposition across all `alphas`. The
//! single chosen `alpha_` then drives a final multi-output Ridge refit on the
//! indicator matrix, recovering `coef_`/`intercept_` exactly as
//! [`crate::RidgeClassifier`] does; prediction is the binary sign / multiclass
//! argmax of the decision function.
//!
//! The GCV closed form (centering for `fit_intercept`, the per-alpha
//! `G_inverse_diag`/`c` computation, intercept-dimension cancellation) is
//! REPLICATED from [`crate::ridge_cv`]'s verified 1-D `_RidgeGCV` path, extended
//! to accumulate the squared LOO errors over every indicator-target column
//! against the SAME shared hat matrix (the hat-matrix diagonal depends only on
//! `X + alpha`, not on the target).
//!
//! ## REQ status
//!
//! Two states only (SHIPPED / NOT-STARTED), per goal.md R-DEFER-2.
//!
//! See `.design/linear/ridge_classifier.md` for the full requirements table.
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-10 (`RidgeClassifierCV`) | SHIPPED | this module: shared-alpha LOO-GCV over the binarized indicator targets (`_ridge.py:2676` + `_RidgeGCV`, `_ridge.py:1688`), final multi-output Ridge refit. |
//! | REQ-11a (store_cv_results/cv_results_) | SHIPPED | #2248. `RidgeClassifierCV<F>` adds `pub store_cv_results: bool` (default `false`, mirroring sklearn's documented default, `_ridge.py:2547`/`:2727`; the ctor `store_cv_results=None` sentinel resolves to `False`, `_ridge.py:2349-2350`) + `with_store_cv_results` builder + getter. `fn gcv_scores_svd`/`fn gcv_scores_eigen in ridge_classifier_cv.rs` retain the un-summed per-sample-per-target squared LOO errors `(c / G_inverse_diag)²` (the SAME terms the alpha-selection sums) into an `Array3<F>` shaped `(n_samples, n_targets, n_alphas)` — alpha axis = input `alphas` order, target axis = indicator columns — mirroring sklearn `cv_results_` (`squared_errors.ravel()`, `_ridge.py:2152`; reshape to `(n_samples, n_y, n_alphas)`, `_ridge.py:2199-2204`). `FittedRidgeClassifierCV<F>` stores `cv_results_: Option<Array3<F>>` + `pub fn cv_results(&self) -> Option<&Array3<F>>` (`None` when `store_cv_results=false`). The selection sum is byte-identical regardless of the flag, so `alpha_`/`coef_`/`predict` are unchanged. Non-test consumer: crate-root re-export `pub use ridge_classifier_cv::{RidgeClassifierCV, FittedRidgeClassifierCV}` in `lib.rs`. Verification (live sklearn 1.5.2, R-CHAR-3): binary `store_cv_results=True` on `oracle_x`, `y=[0,0,0,0,0,1,1,1]`, `alphas=[0.1,1,10]` → `cv_results_` shape `(8,1,3)` values `[[0.01905,0.014503,0.000155],…,[0.595118,0.52245,0.111111]]`; 3-class `y=[0,0,1,1,2,2,1,0]` → `(8,3,3)` per-target; non-monotone `alphas=[10,0.1,1]` → axis-0 ↔ alpha 10. Tests `ridge_classifier_cv_store_cv_results_binary_matches_sklearn`, `…_multiclass_…`, `…_none_default`, `…_alpha_axis_order` PASS. |
//! | REQ-11b (scoring/cv/class_weight/store_cv_values ctor params) | NOT-STARTED | #2248. sklearn's `RidgeClassifierCV` ctor also exposes `scoring` (when set, `cv_results_` stores predictions/scores not squared errors, `_ridge.py:2153-2156`/`:2211-2218`), `cv` (actual k-fold instead of the GCV path), `class_weight` (`_ridge.py:2812`/`:2836`), and the DEPRECATED `store_cv_values` alias (`_ridge.py:2826`/`:2333-2348`). ferrolearn carries only `alphas`/`fit_intercept`/`store_cv_results`; these remain unimplemented. |
//!
//! # Examples
//!
//! ```
//! use ferrolearn_linear::RidgeClassifierCV;
//! use ferrolearn_core::{Fit, Predict};
//! use ndarray::{array, Array2};
//!
//! let x = Array2::from_shape_vec((6, 2), vec![
//!     1.0, 1.0, 1.0, 2.0, 2.0, 1.0,
//!     5.0, 5.0, 5.0, 6.0, 6.0, 5.0,
//! ]).unwrap();
//! let y = array![0usize, 0, 0, 1, 1, 1];
//!
//! let model = RidgeClassifierCV::<f64>::new();
//! let fitted = model.fit(&x, &y).unwrap();
//! let preds = fitted.predict(&x).unwrap();
//! assert_eq!(preds.len(), 6);
//! ```

use ferray::linalg::LinalgFloat;
use ferray::{Array as FerrayArray, Ix2};
use ferrolearn_core::error::FerroError;
use ferrolearn_core::introspection::{HasClasses, HasCoefficients};
use ferrolearn_core::traits::{Fit, Predict};
use ndarray::{Array1, Array2, Array3, Axis, ScalarOperand};
use num_traits::{Float, FromPrimitive};

use crate::Ridge;

/// Ridge classifier with built-in cross-validated alpha selection.
///
/// Selects a single shared regularization strength `alpha` from a candidate
/// grid by leave-one-out Generalized Cross-Validation over the binarized
/// indicator targets, then refits a multi-output Ridge at the chosen alpha.
/// Mirrors scikit-learn's `RidgeClassifierCV` (`sklearn/linear_model/_ridge.py:2676`).
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
#[derive(Debug, Clone)]
pub struct RidgeClassifierCV<F> {
    /// Candidate regularization strengths to evaluate (sklearn `alphas`,
    /// default `(0.1, 1.0, 10.0)`, `_ridge.py:2688`).
    pub alphas: Vec<F>,
    /// Whether to fit an intercept (bias) term (sklearn `fit_intercept`,
    /// default `True`, `_ridge.py:2698`).
    pub fit_intercept: bool,
    /// Whether to retain the per-sample cross-validation results
    /// (`cv_results_`). Mirrors sklearn `store_cv_results` (documented default
    /// `False`, `_ridge.py:2547`/`:2727`; the ctor sentinel `None` resolves to
    /// `False`, `_ridge.py:2349-2350`). When `true`, the fitted model exposes the
    /// per-sample-per-target-per-alpha squared leave-one-out errors via
    /// [`FittedRidgeClassifierCV::cv_results`].
    pub store_cv_results: bool,
}

impl<F: Float + FromPrimitive> RidgeClassifierCV<F> {
    /// Create a new `RidgeClassifierCV` with default settings.
    ///
    /// Defaults: `alphas = [0.1, 1.0, 10.0]` and `fit_intercept = true`,
    /// mirroring sklearn's ctor defaults (`sklearn/linear_model/_ridge.py:2688`,
    /// `:2698`).
    #[must_use]
    pub fn new() -> Self {
        // `F::from(_)` returns `Option`; fall back to `one` (never hit for
        // f32/f64 literals) rather than unwrap in library code (R-CODE-2).
        let one = <F as num_traits::One>::one();
        let p1 = F::from(0.1).unwrap_or(one);
        let ten = F::from(10.0).unwrap_or(one);
        Self {
            alphas: vec![p1, one, ten],
            fit_intercept: true,
            store_cv_results: false,
        }
    }

    /// Set the candidate regularization strengths (sklearn `alphas`,
    /// `_ridge.py:2688`).
    ///
    /// Each value must be non-negative.
    #[must_use]
    pub fn with_alphas(mut self, alphas: Vec<F>) -> Self {
        self.alphas = alphas;
        self
    }

    /// Set whether to fit an intercept term (sklearn `fit_intercept`,
    /// `_ridge.py:2698`).
    #[must_use]
    pub fn with_fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = fit_intercept;
        self
    }

    /// Set whether to retain the cross-validation results (sklearn
    /// `store_cv_results`, default `False`, `_ridge.py:2547`/`:2727`).
    ///
    /// When `true`, the fitted model's [`FittedRidgeClassifierCV::cv_results`]
    /// returns the per-sample-per-target-per-alpha squared leave-one-out errors;
    /// when `false` (the default) it returns `None`.
    #[must_use]
    pub fn with_store_cv_results(mut self, store_cv_results: bool) -> Self {
        self.store_cv_results = store_cv_results;
        self
    }
}

impl<F: Float + FromPrimitive> Default for RidgeClassifierCV<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted Ridge classifier with cross-validated alpha.
///
/// Stores the selected shared alpha, the per-class coefficient matrix, the
/// per-class intercepts, and the sorted class labels. Implements [`Predict`],
/// [`HasCoefficients`], and [`HasClasses`] for introspection.
#[derive(Debug, Clone)]
pub struct FittedRidgeClassifierCV<F> {
    /// The shared alpha that achieved the lowest mean squared LOO error.
    alpha_: F,
    /// Coefficient matrix, shape `(n_classes_or_1, n_features)` matching
    /// sklearn `coef_`. Binary problems store a single `(1, n_features)` row.
    coefficients: Array2<F>,
    /// First coefficient row materialized as a 1-D vector for the
    /// [`HasCoefficients`] (1-D) contract used across the crate.
    coefficients_row0: Array1<F>,
    /// Per-class intercept vector, length `n_classes_or_1`.
    intercepts: Array1<F>,
    /// Sorted unique class labels.
    classes: Vec<usize>,
    /// Whether this is a binary problem (single decision column).
    is_binary: bool,
    /// Number of features (for the predict-time shape check).
    n_features: usize,
    /// Per-sample cross-validation results, shape `(n_samples, n_targets,
    /// n_alphas)`, populated only when `store_cv_results` was set. `Some` holds
    /// the squared leave-one-out errors `(c / G_inverse_diag)²` (sklearn
    /// `cv_results_`, `_ridge.py:2149`/`:2199-2204`); `None` when
    /// `store_cv_results` is `false`.
    cv_results_: Option<Array3<F>>,
}

impl<F: Float> FittedRidgeClassifierCV<F> {
    /// Returns the shared alpha selected by cross-validation (sklearn
    /// `alpha_`, `_ridge.py:2766`).
    #[must_use]
    pub fn alpha_(&self) -> F {
        self.alpha_
    }

    /// Alias for [`alpha_`](Self::alpha_) — the selected shared alpha.
    #[must_use]
    pub fn best_alpha(&self) -> F {
        self.alpha_
    }

    /// Returns the per-class coefficient matrix, shape
    /// `(n_classes_or_1, n_features)` (sklearn `coef_`, `_ridge.py:2757`).
    #[must_use]
    pub fn coefficients(&self) -> &Array2<F> {
        &self.coefficients
    }

    /// Returns the per-class intercept vector, length `n_classes_or_1`
    /// (sklearn `intercept_`, `_ridge.py:2762`).
    #[must_use]
    pub fn intercepts(&self) -> &Array1<F> {
        &self.intercepts
    }

    /// Returns the sorted unique class labels (sklearn `classes_`,
    /// `_ridge.py:2774`).
    #[must_use]
    pub fn classes(&self) -> &[usize] {
        &self.classes
    }

    /// Returns the per-sample cross-validation results, shape
    /// `(n_samples, n_targets, n_alphas)`, or `None` when `store_cv_results`
    /// was not set.
    ///
    /// The values are the per-sample-per-target-per-alpha SQUARED leave-one-out
    /// errors `(c / G_inverse_diag)²` that the GCV minimizes — the same
    /// per-sample terms the alpha selection sums (sklearn `cv_results_`,
    /// `_ridge.py:2149` `squared_errors`, reshaped to `(n_samples, n_y,
    /// n_alphas)` at `_ridge.py:2199-2204`). The alpha axis follows the input
    /// `alphas` order; the target axis follows the indicator columns (the sorted
    /// `classes_`; a single column for binary problems).
    #[must_use]
    pub fn cv_results(&self) -> Option<&Array3<F>> {
        self.cv_results_.as_ref()
    }
}

impl<F: Float + Send + Sync + ScalarOperand + FromPrimitive + LinalgFloat + 'static>
    Fit<Array2<F>, Array1<usize>> for RidgeClassifierCV<F>
{
    type Fitted = FittedRidgeClassifierCV<F>;
    type Error = FerroError;

    /// Fit the `RidgeClassifierCV` model.
    ///
    /// Binarizes `y` into a `{-1, +1}` indicator matrix (binary → single
    /// column, multiclass → one-hot), selects a single shared `alpha` by
    /// leave-one-out Generalized Cross-Validation over that multi-target
    /// problem (mirroring sklearn's `_RidgeGCV` path on the binarized `Y`,
    /// `_ridge.py:2876-2881`; `scoring=None` → `-squared_errors.mean()`,
    /// `_ridge.py:2148-2150`/`:2211-2218`), then refits a multi-output Ridge at
    /// the chosen alpha.
    ///
    /// # Errors
    ///
    /// - [`FerroError::ShapeMismatch`] if `x` and `y` have different numbers of
    ///   samples.
    /// - [`FerroError::InvalidParameter`] if `x` contains a non-finite value
    ///   (NaN/±Inf), or if `alphas` is empty or contains a non-positive value
    ///   (`alpha <= 0`; the GCV path is undefined at `alpha = 0`).
    /// - [`FerroError::InsufficientSamples`] if there are no samples or fewer
    ///   than two distinct classes.
    fn fit(
        &self,
        x: &Array2<F>,
        y: &Array1<usize>,
    ) -> Result<FittedRidgeClassifierCV<F>, FerroError> {
        let (n_samples, n_features) = x.dim();

        if n_samples != y.len() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_samples],
                actual: vec![y.len()],
                context: "y length must match number of samples in X".into(),
            });
        }

        // sklearn `RidgeClassifierCV.fit` -> `_BaseRidge._prepare_data` calls
        // `self._validate_data(X, y, ..., force_all_finite=True[default])`
        // (`_ridge.py:1291`), so any NaN/+/-inf in X raises a `ValueError`
        // BEFORE any decomposition. Fire this up front so BOTH the wide/eigen
        // (`n_samples <= n_features`, today `Ok(NaN)`) and the SVD
        // (`n_samples > n_features`, today an incidental linalg-convergence
        // error) paths get the same clean validation rejection. `y` is class
        // labels (`usize`, always finite by type), so only X is checked —
        // mirroring the sibling pattern in `multi_task_lasso.rs`. (#2246)
        if x.iter().any(|v| !v.is_finite()) {
            return Err(FerroError::InvalidParameter {
                name: "X".into(),
                reason: "Input X contains NaN or infinity.".into(),
            });
        }

        if self.alphas.is_empty() {
            return Err(FerroError::InvalidParameter {
                name: "alphas".into(),
                reason: "must contain at least one candidate".into(),
            });
        }

        for &a in &self.alphas {
            // `<F as num_traits::Zero>::zero()`: the `LinalgFloat` bound pulls
            // `ferray::Element` (also defining `zero`) into scope, making a bare
            // `F::zero()` ambiguous. Disambiguate to `num_traits::Zero`.
            //
            // Strictly positive (`a <= 0` rejected, not just `a < 0`): on the
            // GCV path (ferrolearn's only path, `cv is None`) sklearn validates
            // each alpha with `Interval(Real, 0, None, closed="neither")`
            // (`_ridge.py:2259`) / `include_boundaries="neither"`
            // (`_ridge.py:2354-2360`), raising `ValueError("alphas[i] == 0.0,
            // must be > 0.0")` because "_RidgeGCV does not work for alpha = 0"
            // (`_ridge.py:2354`). (#2247)
            if a <= <F as num_traits::Zero>::zero() {
                return Err(FerroError::InvalidParameter {
                    name: "alphas".into(),
                    reason: "alphas must be > 0.0".into(),
                });
            }
        }

        if n_samples == 0 {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "RidgeClassifierCV requires at least one sample".into(),
            });
        }

        // Sorted unique class labels (mirrors sklearn's `LabelBinarizer.classes_`).
        let mut classes: Vec<usize> = y.to_vec();
        classes.sort_unstable();
        classes.dedup();

        if classes.len() < 2 {
            return Err(FerroError::InsufficientSamples {
                required: 2,
                actual: classes.len(),
                context: "RidgeClassifierCV requires at least 2 distinct classes".into(),
            });
        }

        let is_binary = classes.len() == 2;

        // Build the `{-1, +1}` indicator matrix `Y` (mirrors
        // `LabelBinarizer(pos_label=1, neg_label=-1)`, `_ridge.py:1300-1301`).
        // Binary → single column (+1 for class index 1, -1 for class 0);
        // multiclass → one-hot `{-1, +1}` (+1 on the active class, -1 elsewhere).
        let n_targets = if is_binary { 1 } else { classes.len() };
        let one = <F as num_traits::One>::one();
        let neg_one = -one;
        let mut y_indicator = Array2::<F>::from_elem((n_samples, n_targets), neg_one);

        if is_binary {
            for i in 0..n_samples {
                if y[i] == classes[1] {
                    y_indicator[[i, 0]] = one;
                }
            }
        } else {
            for i in 0..n_samples {
                // `classes` is the sorted-deduped image of `y`, so `y[i]` is
                // always present; fall back to a typed error rather than panic.
                let ci = classes.iter().position(|&c| c == y[i]).ok_or_else(|| {
                    FerroError::NumericalInstability {
                        message: "class label missing from class set".into(),
                    }
                })?;
                y_indicator[[i, ci]] = one;
            }
        }

        // SHARED-ALPHA leave-one-out GCV over the binarized multi-target
        // problem. The hat-matrix diagonal depends only on `X + alpha`, so a
        // single decomposition is reused across both alphas AND target columns;
        // the per-alpha score sums the squared LOO errors over every column and
        // sample (sklearn `-squared_errors.mean()` with `squared_errors` shape
        // `(n_samples, n_y)`, `_ridge.py:2148-2150`/`:2211-2218`).
        let (alpha_, cv_results_) = self.select_alpha_gcv(x, &y_indicator)?;

        // Final refit: multi-output Ridge at the selected alpha on the indicator
        // matrix (sklearn refits `coef_ = dual_coef_.T @ X` + `_set_intercept`,
        // `_ridge.py:2191-2197`; an equivalent direct Ridge refit reproduces the
        // same centering/intercept handling RidgeClassifier uses).
        let final_model = Ridge::<F>::new()
            .with_alpha(alpha_)
            .with_fit_intercept(self.fit_intercept);
        let fitted_multi = final_model.fit(x, &y_indicator)?;

        // `FittedRidgeMulti` stores `(n_features, n_targets)`; transpose to the
        // sklearn `coef_` orientation `(n_classes_or_1, n_features)`.
        let coef_ft = fitted_multi.coefficients();
        let coefficients = coef_ft.t().to_owned();
        let coefficients_row0 = coefficients.row(0).to_owned();
        let intercepts = fitted_multi.intercepts().clone();

        Ok(FittedRidgeClassifierCV {
            alpha_,
            coefficients,
            coefficients_row0,
            intercepts,
            classes,
            is_binary,
            n_features,
            cv_results_,
        })
    }
}

impl<F: Float + Send + Sync + ScalarOperand + FromPrimitive + LinalgFloat + 'static>
    RidgeClassifierCV<F>
{
    /// Shared-alpha leave-one-out Generalized Cross-Validation over the
    /// binarized multi-target indicator `Y`, mirroring sklearn `_RidgeGCV.fit`
    /// (`_ridge.py:2059`) on the binarized targets.
    ///
    /// Centers `X`/`Y` when `fit_intercept` (uniform weights → `sqrt_sw = 1`;
    /// sklearn `_preprocess_data`, `_ridge.py:2106`), decomposes once via the
    /// shape-appropriate mode (sklearn `_check_gcv_mode`, `_ridge.py:1569`: SVD
    /// of the design when `n_samples > n_features`, else eigendecomposition of
    /// the Gram `X·Xᵀ`), then for each alpha sums the squared closed-form LOO
    /// errors `(c / G_inverse_diag)²` over EVERY indicator column and sample and
    /// picks the alpha minimising that total (equivalently the mean — `n_y` is
    /// constant across alphas; sklearn `-squared_errors.mean()`,
    /// `_ridge.py:2148-2150`/`:2211-2218`). Ties → the first (smallest-index)
    /// alpha, matching sklearn's strict `alpha_score > best_score` update
    /// (`_ridge.py:2185`).
    ///
    /// When `self.store_cv_results`, ALSO returns the per-sample-per-target
    /// squared LOO errors as an `Array3<F>` shaped `(n_samples, n_targets,
    /// n_alphas)` (alpha axis = input `alphas` order, target axis = indicator
    /// columns) — the un-summed terms the alpha selection sums (sklearn
    /// `cv_results_`, `_ridge.py:2149`/`:2199-2204`). The selection math itself
    /// is unchanged, so `alpha_` stays bit-exact regardless of the flag.
    #[allow(
        clippy::type_complexity,
        reason = "GCV returns the selected alpha plus the optional retained cv_results_ buffer in one pass to avoid recomputing the decomposition"
    )]
    fn select_alpha_gcv(
        &self,
        x: &Array2<F>,
        y: &Array2<F>,
    ) -> Result<(F, Option<Array3<F>>), FerroError> {
        let (n_samples, n_features) = x.dim();

        // Center X and Y per column (sklearn `_preprocess_data`,
        // `_ridge.py:2106`); with uniform weights the centered design has zero
        // column means and the square-root sample weights are all 1.
        let (x_c, y_c) = if self.fit_intercept {
            let x_mean = x
                .mean_axis(Axis(0))
                .ok_or_else(|| FerroError::NumericalInstability {
                    message: "RidgeClassifierCV GCV: failed to compute X column means".into(),
                })?;
            let y_mean = y
                .mean_axis(Axis(0))
                .ok_or_else(|| FerroError::NumericalInstability {
                    message: "RidgeClassifierCV GCV: failed to compute Y column means".into(),
                })?;
            (x - &x_mean, y - &y_mean)
        } else {
            (x.to_owned(), y.to_owned())
        };

        // Per-alpha total squared LOO error (summed over samples AND columns).
        // When `store_cv_results`, the score functions ALSO return the
        // un-summed per-sample-per-target squared LOO errors as
        // `(n_samples, n_targets, n_alphas)`.
        let (scores, cv_results) = if n_samples > n_features {
            self.gcv_scores_svd(&x_c, &y_c)?
        } else {
            self.gcv_scores_eigen(&x_c, &y_c)?
        };

        let mut best_alpha = self.alphas[0];
        let mut best_err = F::infinity();
        for (&alpha, &total_sq_err) in self.alphas.iter().zip(scores.iter()) {
            if total_sq_err < best_err {
                best_err = total_sq_err;
                best_alpha = alpha;
            }
        }

        Ok((best_alpha, cv_results))
    }

    /// SVD-mode shared-alpha GCV totals, used when `n_samples > n_features`
    /// (sklearn `_svd_decompose_design_matrix` `_ridge.py:2025` +
    /// `_solve_svd_design_matrix` `_ridge.py:2039`). Returns the total squared
    /// LOO error (summed over every indicator column and sample) for each
    /// candidate alpha (lower is better).
    ///
    /// REPLICATES `crate::ridge_cv`'s verified 1-D SVD path, extended to
    /// accumulate over the columns of `Y` against the SAME `U`/singular values.
    ///
    /// When `self.store_cv_results`, the returned `Option<Array3<F>>` holds the
    /// un-summed per-sample-per-target squared LOO errors shaped `(n_samples,
    /// n_targets, n_alphas)` (sklearn `cv_results_`, `_ridge.py:2149`/
    /// `:2199-2204`). The summation that selects `alpha_` is byte-identical
    /// whether or not retention is enabled.
    #[allow(
        clippy::type_complexity,
        reason = "returns the per-alpha totals plus the optional retained cv_results_ buffer computed in the same loop"
    )]
    fn gcv_scores_svd(
        &self,
        x_c: &Array2<F>,
        y_c: &Array2<F>,
    ) -> Result<(Vec<F>, Option<Array3<F>>), FerroError> {
        let n_samples = x_c.nrows();
        let n_targets = y_c.ncols();
        let one = <F as num_traits::One>::one();

        // Build the (possibly intercept-augmented) design matrix. With uniform
        // weights `sqrt_sw = 1`, so the appended intercept column is all ones
        // (sklearn `_svd_decompose_design_matrix`, `_ridge.py:2032`).
        let n_cols = if self.fit_intercept {
            x_c.ncols() + 1
        } else {
            x_c.ncols()
        };
        let mut design = Array2::<F>::zeros((n_samples, n_cols));
        design.slice_mut(ndarray::s![.., ..x_c.ncols()]).assign(x_c);
        if self.fit_intercept {
            design.column_mut(x_c.ncols()).fill(one);
        }

        // Thin SVD: `U` is `(n_samples, k)`, singular values length `k`
        // (sklearn `linalg.svd(X, full_matrices=0)`, `_ridge.py:2034`).
        let (u, singvals) = svd_u_s(&design)?;
        let k = singvals.len();

        let singvals_sq: Vec<F> = (0..k).map(|j| singvals[j] * singvals[j]).collect();

        // UT_Y[j, t] = Σ_i U[i,j] · Y[i,t] (sklearn `_ridge.py:2036`, per column).
        let mut ut_y = Array2::<F>::zeros((k, n_targets));
        for j in 0..k {
            for t in 0..n_targets {
                let mut acc = <F as num_traits::Zero>::zero();
                for i in 0..n_samples {
                    acc += u[(i, j)] * y_c[[i, t]];
                }
                ut_y[[j, t]] = acc;
            }
        }

        // Intercept dimension: the column of U most aligned with the normalized
        // sqrt_sw (uniform → ones/√n) (sklearn `_find_smallest_angle`,
        // `_ridge.py:1579`).
        let intercept_dim = if self.fit_intercept {
            Some(find_intercept_dim(&u, n_samples, k))
        } else {
            None
        };

        // Optional cv_results_ retention buffer, shaped (n_samples, n_targets,
        // n_alphas); the alpha axis follows the input `alphas` enumeration order
        // (sklearn `cv_results_`, `_ridge.py:2199-2204`).
        let mut cv_results = if self.store_cv_results {
            Some(Array3::<F>::zeros((
                n_samples,
                n_targets,
                self.alphas.len(),
            )))
        } else {
            None
        };

        let mut out = Vec::with_capacity(self.alphas.len());
        for (a_idx, &alpha) in self.alphas.iter().enumerate() {
            let inv_alpha = one / alpha;
            // w_j = (singvals_sq_j + alpha)^-1 - alpha^-1 (sklearn :2045).
            let mut w: Vec<F> = singvals_sq
                .iter()
                .map(|&s2| one / (s2 + alpha) - inv_alpha)
                .collect();
            if let Some(d) = intercept_dim {
                // Cancel regularization for the intercept (sklearn :2051).
                w[d] = -inv_alpha;
            }

            // For each sample i: G_inverse_diag_i = Σ_j w_j U[i,j]² + alpha^-1
            // (sklearn :2053, shared across columns). Per column t:
            // c[i,t] = Σ_j U[i,j]·w_j·UT_Y[j,t] + alpha^-1·Y[i,t] (sklearn :2052).
            // squared_error[i,t] = (c[i,t] / G_inverse_diag_i)² (sklearn :2149).
            let mut total = <F as num_traits::Zero>::zero();
            for i in 0..n_samples {
                let mut g_i = <F as num_traits::Zero>::zero();
                for j in 0..k {
                    let uij = u[(i, j)];
                    g_i += w[j] * uij * uij;
                }
                g_i += inv_alpha;
                for t in 0..n_targets {
                    let mut c_it = <F as num_traits::Zero>::zero();
                    for j in 0..k {
                        c_it += u[(i, j)] * (w[j] * ut_y[[j, t]]);
                    }
                    c_it += inv_alpha * y_c[[i, t]];
                    let looe = c_it / g_i;
                    let sq_err = looe * looe;
                    total += sq_err;
                    // Retain the un-summed per-sample-per-target squared LOO
                    // error (`cv_results_[:, t, a]`, sklearn `_ridge.py:2152`
                    // ravel + `:2199-2204` reshape).
                    if let Some(buf) = cv_results.as_mut() {
                        buf[[i, t, a_idx]] = sq_err;
                    }
                }
            }
            out.push(total);
        }
        Ok((out, cv_results))
    }

    /// Eigen-mode shared-alpha GCV totals, used when
    /// `n_samples <= n_features` (sklearn `_eigen_decompose_gram`
    /// `_ridge.py:1900` then `_solve_eigen_gram` `_ridge.py:1914`). Returns the
    /// total squared LOO error (summed over every indicator column and sample)
    /// for each candidate alpha (lower is better).
    ///
    /// REPLICATES `crate::ridge_cv`'s verified 1-D eigen path, extended to
    /// accumulate over the columns of `Y` against the SAME `Q`/eigenvalues.
    ///
    /// When `self.store_cv_results`, the returned `Option<Array3<F>>` holds the
    /// un-summed per-sample-per-target squared LOO errors shaped `(n_samples,
    /// n_targets, n_alphas)` (sklearn `cv_results_`, `_ridge.py:2149`/
    /// `:2199-2204`). The summation that selects `alpha_` is byte-identical
    /// whether or not retention is enabled.
    #[allow(
        clippy::type_complexity,
        reason = "returns the per-alpha totals plus the optional retained cv_results_ buffer computed in the same loop"
    )]
    fn gcv_scores_eigen(
        &self,
        x_c: &Array2<F>,
        y_c: &Array2<F>,
    ) -> Result<(Vec<F>, Option<Array3<F>>), FerroError> {
        let n_samples = x_c.nrows();
        let n_targets = y_c.ncols();
        let one = <F as num_traits::One>::one();

        // Gram matrix K = X·Xᵀ on the centered design (sklearn dense
        // `_compute_gram` → `X X^T`, `_ridge.py:1799`).
        let mut k_mat = x_c.dot(&x_c.t());
        if self.fit_intercept {
            // Add outer(sqrt_sw, sqrt_sw): uniform weights → the all-ones rank-1
            // matrix, emulating centering with the intercept eigenvector
            // (sklearn `_eigen_decompose_gram`, `_ridge.py:1909`).
            for i in 0..n_samples {
                for j in 0..n_samples {
                    k_mat[(i, j)] += one;
                }
            }
        }

        // Eigendecomposition K = Q diag(eigvals) Qᵀ (sklearn `linalg.eigh`,
        // `_ridge.py:1910`).
        let (eigvals, q) = eigh_sym(&k_mat)?;
        let m = eigvals.len();

        // QT_Y[j, t] = Σ_i Q[i,j] · Y[i,t] (sklearn :1911, per column).
        let mut qt_y = Array2::<F>::zeros((m, n_targets));
        for j in 0..m {
            for t in 0..n_targets {
                let mut acc = <F as num_traits::Zero>::zero();
                for i in 0..n_samples {
                    acc += q[(i, j)] * y_c[[i, t]];
                }
                qt_y[[j, t]] = acc;
            }
        }

        // Intercept eigenvector: the column of Q most aligned with the
        // normalized sqrt_sw (uniform → ones/√n) (sklearn :1926-1927).
        let intercept_dim = if self.fit_intercept {
            Some(find_intercept_dim(&q, n_samples, m))
        } else {
            None
        };

        // Optional cv_results_ retention buffer, shaped (n_samples, n_targets,
        // n_alphas); the alpha axis follows the input `alphas` enumeration order
        // (sklearn `cv_results_`, `_ridge.py:2199-2204`).
        let mut cv_results = if self.store_cv_results {
            Some(Array3::<F>::zeros((
                n_samples,
                n_targets,
                self.alphas.len(),
            )))
        } else {
            None
        };

        let mut out = Vec::with_capacity(self.alphas.len());
        for (a_idx, &alpha) in self.alphas.iter().enumerate() {
            // w_j = 1 / (eigvals_j + alpha) (sklearn :1919).
            let mut w: Vec<F> = eigvals.iter().map(|&ev| one / (ev + alpha)).collect();
            if let Some(d) = intercept_dim {
                // Cancel regularization for the intercept (sklearn :1928).
                w[d] = <F as num_traits::Zero>::zero();
            }

            // G_inverse_diag_i = Σ_j w_j Q[i,j]² (shared across columns,
            // sklearn :1931). c[i,t] = Σ_j Q[i,j]·w_j·QT_Y[j,t] (sklearn :1930).
            let mut total = <F as num_traits::Zero>::zero();
            for i in 0..n_samples {
                let mut g_i = <F as num_traits::Zero>::zero();
                for j in 0..m {
                    let qij = q[(i, j)];
                    g_i += w[j] * qij * qij;
                }
                for t in 0..n_targets {
                    let mut c_it = <F as num_traits::Zero>::zero();
                    for j in 0..m {
                        c_it += q[(i, j)] * (w[j] * qt_y[[j, t]]);
                    }
                    let looe = c_it / g_i;
                    let sq_err = looe * looe;
                    total += sq_err;
                    // Retain the un-summed per-sample-per-target squared LOO
                    // error (`cv_results_[:, t, a]`, sklearn `_ridge.py:2152`
                    // ravel + `:2199-2204` reshape).
                    if let Some(buf) = cv_results.as_mut() {
                        buf[[i, t, a_idx]] = sq_err;
                    }
                }
            }
            out.push(total);
        }
        Ok((out, cv_results))
    }
}

/// Find the column index of an orthonormal factor (`U` or `Q`, both
/// `(n_samples, k)`) most aligned with the normalized uniform-weight vector
/// `ones/√n`. Mirrors sklearn `_find_smallest_angle` (`_ridge.py:1579`): the
/// query and columns are unit vectors, so the most-aligned column maximises
/// `|query · column|`; with `query = ones/√n` the per-column dot product is
/// proportional to the column sum, so `|column-sum|` is the discriminant.
fn find_intercept_dim<F: Float + std::ops::AddAssign + 'static>(
    u: &Array2<F>,
    n_samples: usize,
    k: usize,
) -> usize {
    let mut best_idx = 0usize;
    let mut best_abs = F::neg_infinity();
    for j in 0..k {
        let mut col_sum = <F as num_traits::Zero>::zero();
        for i in 0..n_samples {
            col_sum += u[(i, j)];
        }
        let a = col_sum.abs();
        if a > best_abs {
            best_abs = a;
            best_idx = j;
        }
    }
    best_idx
}

/// Thin SVD via the ferray substrate, returning `(U, S)` as ndarray types.
///
/// Bridges `ndarray → ferray` for the decomposition and back (R-SUBSTRATE-4),
/// routing through [`ferray::linalg::svd`] (`ferray-linalg/src/decomp/svd.rs`).
fn svd_u_s<F: LinalgFloat>(a: &Array2<F>) -> Result<(Array2<F>, Array1<F>), FerroError> {
    let (rows, cols) = a.dim();
    let flat: Vec<F> = a.iter().copied().collect();
    let fa = FerrayArray::<F, Ix2>::from_vec(Ix2::new([rows, cols]), flat).map_err(|e| {
        FerroError::NumericalInstability {
            message: format!("RidgeClassifierCV GCV: failed to build design matrix for SVD: {e}"),
        }
    })?;
    let (u, s, _vt) =
        ferray::linalg::svd(&fa, false).map_err(|e| FerroError::NumericalInstability {
            message: format!("RidgeClassifierCV GCV: SVD failed: {e}"),
        })?;
    let u_nd = ferray_to_ndarray2(&u)?;
    let s_nd = ferray_to_ndarray1(&s)?;
    Ok((u_nd, s_nd))
}

/// Symmetric eigendecomposition via the ferray substrate, returning
/// `(eigvals, Q)` as ndarray types (ascending eigenvalues).
///
/// Bridges `ndarray → ferray` and back (R-SUBSTRATE-4), routing through
/// [`ferray::linalg::eigh`] (`ferray-linalg/src/decomp/eigen.rs`).
fn eigh_sym<F: LinalgFloat>(a: &Array2<F>) -> Result<(Array1<F>, Array2<F>), FerroError> {
    let (rows, cols) = a.dim();
    let flat: Vec<F> = a.iter().copied().collect();
    let fa = FerrayArray::<F, Ix2>::from_vec(Ix2::new([rows, cols]), flat).map_err(|e| {
        FerroError::NumericalInstability {
            message: format!("RidgeClassifierCV GCV: failed to build Gram matrix for eigh: {e}"),
        }
    })?;
    let (vals, q) = ferray::linalg::eigh(&fa).map_err(|e| FerroError::NumericalInstability {
        message: format!("RidgeClassifierCV GCV: eigendecomposition failed: {e}"),
    })?;
    let vals_nd = ferray_to_ndarray1(&vals)?;
    let q_nd = ferray_to_ndarray2(&q)?;
    Ok((vals_nd, q_nd))
}

/// Bridge a ferray 2-D array back to `ndarray::Array2` (R-SUBSTRATE-4).
fn ferray_to_ndarray2<F: LinalgFloat>(a: &FerrayArray<F, Ix2>) -> Result<Array2<F>, FerroError> {
    let shape = a.shape();
    let (rows, cols) = (shape[0], shape[1]);
    let nd = a.clone().into_ndarray();
    let flat: Vec<F> = nd.iter().copied().collect();
    Array2::from_shape_vec((rows, cols), flat).map_err(|e| FerroError::NumericalInstability {
        message: format!("RidgeClassifierCV GCV: ferray→ndarray (2-D) bridge failed: {e}"),
    })
}

/// Bridge a ferray 1-D array back to `ndarray::Array1` (R-SUBSTRATE-4).
fn ferray_to_ndarray1<F: LinalgFloat>(
    a: &FerrayArray<F, ferray::Ix1>,
) -> Result<Array1<F>, FerroError> {
    let nd = a.clone().into_ndarray();
    let flat: Vec<F> = nd.iter().copied().collect();
    Ok(Array1::from_vec(flat))
}

impl<F: Float + Send + Sync + ScalarOperand + 'static> Predict<Array2<F>>
    for FittedRidgeClassifierCV<F>
{
    type Output = Array1<usize>;
    type Error = FerroError;

    /// Predict class labels for the given feature matrix.
    ///
    /// Computes the decision `X · coefficientsᵀ + intercepts`: binary takes the
    /// strict-sign rule (`classes[1]` if `decision > 0` else `classes[0]`,
    /// mirroring `LinearClassifierMixin.predict`, `_base.py:384` `scores > 0`);
    /// multiclass takes the argmax over class columns → `classes[idx]`.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of features does not
    /// match the fitted model.
    fn predict(&self, x: &Array2<F>) -> Result<Array1<usize>, FerroError> {
        let n_features = x.ncols();
        if n_features != self.n_features {
            return Err(FerroError::ShapeMismatch {
                expected: vec![self.n_features],
                actual: vec![n_features],
                context: "number of features must match fitted model".into(),
            });
        }

        let n_samples = x.nrows();
        let mut predictions = Array1::<usize>::zeros(n_samples);

        // Decision scores: X · coefficientsᵀ + intercepts, shape
        // `(n_samples, n_classes_or_1)`.
        let scores = x.dot(&self.coefficients.t()) + &self.intercepts;

        if self.is_binary {
            for i in 0..n_samples {
                // Strict `> 0` (sklearn `_base.py:384`): an exact-0 decision maps
                // to index 0 → `classes[0]`.
                predictions[i] = if scores[[i, 0]] > <F as num_traits::Zero>::zero() {
                    self.classes[1]
                } else {
                    self.classes[0]
                };
            }
        } else {
            for i in 0..n_samples {
                let mut best_class = 0;
                let mut best_score = scores[[i, 0]];
                for c in 1..self.classes.len() {
                    if scores[[i, c]] > best_score {
                        best_score = scores[[i, c]];
                        best_class = c;
                    }
                }
                predictions[i] = self.classes[best_class];
            }
        }

        Ok(predictions)
    }
}

impl<F: Float + Send + Sync + ScalarOperand + 'static> HasCoefficients<F>
    for FittedRidgeClassifierCV<F>
{
    /// Returns the first coefficient row as a flat vector (the binary decision
    /// vector / first class for multiclass), matching the `HasCoefficients`
    /// 1-D contract used across the crate.
    fn coefficients(&self) -> &Array1<F> {
        &self.coefficients_row0
    }

    fn intercept(&self) -> F {
        self.intercepts[0]
    }
}

impl<F: Float + Send + Sync + ScalarOperand + 'static> HasClasses for FittedRidgeClassifierCV<F> {
    fn classes(&self) -> &[usize] {
        &self.classes
    }

    fn n_classes(&self) -> usize {
        self.classes.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// Shared 8×2 design used by the oracle tests.
    fn oracle_x() -> Array2<f64> {
        Array2::from_shape_vec(
            (8, 2),
            vec![
                1.0, 2.0, 2.0, 1.0, 3.0, 1.0, 1.0, 3.0, 2.0, 2.0, 6.0, 5.0, 5.0, 6.0, 7.0, 7.0,
            ],
        )
        .unwrap()
    }

    #[test]
    fn ridge_classifier_cv_binary_matches_sklearn() -> Result<(), FerroError> {
        // Live sklearn 1.5.2 oracle (R-CHAR-3):
        //   python3 -c "import numpy as np; \
        //     from sklearn.linear_model import RidgeClassifierCV; \
        //     X=np.array([[1,2],[2,1],[3,1],[1,3],[2,2],[6,5],[5,6],[7,7]],float); \
        //     y=np.array([0,0,0,0,0,1,1,1]); \
        //     m=RidgeClassifierCV(alphas=[0.1,1.0,10.0]).fit(X,y); \
        //     print(m.alpha_, m.coef_.tolist(), m.intercept_.tolist(), m.predict(X).tolist())"
        //   -> alpha_ 10.0
        //      coef_ [[0.1974921630094044, 0.1974921630094044]]
        //      intercept_ [-1.5830721003134798]
        //      predict [0, 0, 0, 0, 0, 1, 1, 1]
        let x = oracle_x();
        let y = array![0usize, 0, 0, 0, 0, 1, 1, 1];

        let model = RidgeClassifierCV::<f64>::new().with_alphas(vec![0.1, 1.0, 10.0]);
        let fitted = model.fit(&x, &y)?;

        assert!(
            (fitted.alpha_() - 10.0).abs() < 1e-12,
            "alpha_={} expected 10.0",
            fitted.alpha_()
        );

        let coef = fitted.coefficients();
        assert_eq!(coef.shape(), &[1, 2], "binary coef_ must be (1, 2)");
        assert!(
            (coef[[0, 0]] - 0.197_492_163_0).abs() < 1e-6,
            "coef[0,0]={} expected 0.197492163",
            coef[[0, 0]]
        );
        assert!(
            (coef[[0, 1]] - 0.197_492_163_0).abs() < 1e-6,
            "coef[0,1]={} expected 0.197492163",
            coef[[0, 1]]
        );

        assert!(
            (fitted.intercepts()[0] - (-1.583_072_100_3)).abs() < 1e-6,
            "intercept={} expected -1.5830721003",
            fitted.intercepts()[0]
        );

        let preds = fitted.predict(&x)?;
        assert_eq!(preds.to_vec(), vec![0, 0, 0, 0, 0, 1, 1, 1]);
        Ok(())
    }

    #[test]
    fn ridge_classifier_cv_multiclass_matches_sklearn() -> Result<(), FerroError> {
        // Live sklearn 1.5.2 oracle (R-CHAR-3):
        //   python3 -c "import numpy as np; \
        //     from sklearn.linear_model import RidgeClassifierCV; \
        //     X=np.array([[1,2],[2,1],[3,1],[1,3],[2,2],[6,5],[5,6],[7,7]],float); \
        //     y=np.array([0,0,1,1,2,2,1,0]); \
        //     m=RidgeClassifierCV(alphas=[0.1,1.0,10.0]).fit(X,y); \
        //     print(m.alpha_, m.coef_.tolist(), m.predict(X).tolist())"
        //   -> alpha_ 10.0
        //      coef_ [[-0.0031348, -0.0031348],
        //             [-0.07817398, 0.04682602],
        //             [0.08130878, -0.04369122]]
        //      predict [1, 0, 0, 1, 1, 0, 1, 0]
        let x = oracle_x();
        let y = array![0usize, 0, 1, 1, 2, 2, 1, 0];

        let model = RidgeClassifierCV::<f64>::new().with_alphas(vec![0.1, 1.0, 10.0]);
        let fitted = model.fit(&x, &y)?;

        assert!(
            (fitted.alpha_() - 10.0).abs() < 1e-12,
            "alpha_={} expected 10.0",
            fitted.alpha_()
        );

        let coef = fitted.coefficients();
        assert_eq!(coef.shape(), &[3, 2], "multiclass coef_ must be (3, 2)");

        let expected = [
            [-0.003_134_8, -0.003_134_8],
            [-0.078_173_98, 0.046_826_02],
            [0.081_308_78, -0.043_691_22],
        ];
        for r in 0..3 {
            for c in 0..2 {
                assert!(
                    (coef[[r, c]] - expected[r][c]).abs() < 1e-6,
                    "coef[{r},{c}]={} expected {}",
                    coef[[r, c]],
                    expected[r][c]
                );
            }
        }

        let preds = fitted.predict(&x)?;
        assert_eq!(preds.to_vec(), vec![1, 0, 0, 1, 1, 0, 1, 0]);
        Ok(())
    }

    #[test]
    fn ridge_classifier_cv_single_class_errors() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 2.0, 1.0, 3.0, 1.0]).unwrap();
        let y = array![0usize, 0, 0];

        let model = RidgeClassifierCV::<f64>::new();
        assert!(
            model.fit(&x, &y).is_err(),
            "single-class input must error (>= 2-class guard)"
        );
    }

    #[test]
    fn ridge_classifier_cv_selects_from_alphas() -> Result<(), FerroError> {
        // The selected alpha_ must always be one of the provided alphas, and
        // the prediction must be sensible (recovers the well-separated labels).
        let x = oracle_x();
        let y = array![0usize, 0, 0, 0, 0, 1, 1, 1];
        let alphas = vec![0.01, 0.1, 1.0, 10.0, 100.0];

        let model = RidgeClassifierCV::<f64>::new().with_alphas(alphas.clone());
        let fitted = model.fit(&x, &y)?;

        assert!(
            alphas.iter().any(|&a| (a - fitted.alpha_()).abs() < 1e-12),
            "selected alpha_={} is not one of the candidates",
            fitted.alpha_()
        );

        let preds = fitted.predict(&x)?;
        let correct = preds.iter().zip(y.iter()).filter(|(p, t)| p == t).count();
        assert!(correct >= 7, "expected >= 7 correct, got {correct}");
        Ok(())
    }

    #[test]
    fn ridge_classifier_cv_default_builder() {
        let m = RidgeClassifierCV::<f64>::new();
        assert_eq!(m.alphas.len(), 3);
        assert!(m.fit_intercept);
    }

    #[test]
    fn ridge_classifier_cv_shape_mismatch() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 2.0, 1.0, 3.0, 1.0]).unwrap();
        let y = array![0usize, 1];

        let model = RidgeClassifierCV::<f64>::new();
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn ridge_classifier_cv_empty_alphas_error() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 2.0, 1.0, 6.0, 5.0, 5.0, 6.0]).unwrap();
        let y = array![0usize, 0, 1, 1];

        let model = RidgeClassifierCV::<f64>::new().with_alphas(vec![]);
        assert!(model.fit(&x, &y).is_err());
    }
}
