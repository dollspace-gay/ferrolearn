//! Ordinary Least Squares linear regression.
//!
//! This module provides [`LinearRegression`], which fits a linear model by
//! solving the least squares problem via a single SVD (the LAPACK-`gelsd`
//! minimum-norm path, through `ferray::linalg::lstsq`):
//!
//! ```text
//! minimize ||X @ w - y||^2
//! ```
//!
//! ## REQ status (per `.design/linear/linear_regression.md`, mirrors `sklearn/linear_model/_base.py` @ 1.5.2)
//!
//! Mirrors `sklearn.linear_model.LinearRegression` (`_base.py:465`). Full-rank,
//! rank-deficient, and underdetermined OLS all match the live sklearn oracle to
//! 1e-8: the solve routes through `crate::linalg::solve_lstsq` →
//! `ferray::linalg::lstsq` (single-SVD, LAPACK-`gelsd`-equivalent min-norm),
//! mirroring sklearn's `linalg.lstsq(X, y)` (`_base.py:687`).
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (full-rank OLS coef_/intercept_) | SHIPPED | `Fit for LinearRegression` (centering + `linalg::solve_lstsq` via `ferray::linalg::lstsq`); full-rank coef/intercept match oracle to 1e-8. Consumer: `RsLinearRegression` in `ferrolearn-python/src/regressors.rs`. Mirrors `_base.py:582`, intercept `_base.py:308`. |
//! | REQ-2 (predict = X·coef + intercept) | SHIPPED | `Predict for FittedLinearRegression`. Mirrors `_base.py:282`. |
//! | REQ-3 (fit_intercept incl. false) | SHIPPED | `with_fit_intercept`; `fit_intercept=false` forces intercept 0. Mirrors `_base.py:571`. |
//! | REQ-4 (HasCoefficients introspection) | SHIPPED | `HasCoefficients for FittedLinearRegression`. Mirrors fitted attrs `_base.py:499/511`. |
//! | REQ-5 (min-norm for rank-deficient / underdetermined X) | SHIPPED | `Fit for LinearRegression` calls `crate::linalg::solve_lstsq` → `ferray::linalg::lstsq` (`ferray-linalg/src/solve.rs:208`), the single-SVD gelsd-equivalent min-norm solver mirroring `_base.py:687`. Closes #376 (rank-deficient min-norm) + #377 (underdetermined accepted). Tests now passing (`#[ignore]` removed): `divergence_rank_deficient_no_intercept_min_norm`, `divergence_rank_deficient_with_intercept_min_norm`, `divergence_underdetermined_accepted_min_norm` in `tests/divergence_linreg_minnorm.rs`. |
//! | REQ-6 (positive=True / NNLS) | NOT-STARTED | blocker #371 (`_base.py:574/645`). |
//! | REQ-7 (multi-output 2-D Y → 2-D coef_) | SHIPPED | Additive `Fit<Array2<F>, Array2<F>>` arm (does NOT touch the 1-D `Fit`/`FittedLinearRegression`/`Predict`) producing `FittedMultiOutputLinearRegression<F>` — `coefficients` shape `(n_targets, n_features)` (sklearn `coef_` orientation, `coef_.T` of the lstsq solution, `_base.py:688`), `intercepts` `(n_targets,)`, `rank_`/`singular_`. Solves all targets in one SVD via `linalg::solve_lstsq_multi` → `ferray::linalg::lstsq` with a 2-D `b` (mirrors `linalg.lstsq(X, Y)`, `_base.py:687`); shared X-centering + per-target y-offset, `intercepts = y_off − coefficients · x_off` (`_set_intercept`, `_base.py:322`); `fit_intercept=false` → raw solve, intercepts 0. `Predict<Array2<F>, Output=Array2<F>>` returns `X · coef_.T + intercepts` shape `(n_samples, n_targets)` (`_base.py:290`). Oracle tests `linreg_multioutput_coef_intercept_match_sklearn` (coef `[[2.06666667,-0.06666667],[0.86666667,0.23333333]]`, intercept `[-0.06666667,0.13333333]`), `linreg_multioutput_predict_shape_and_values` (`predict(X[:2]) = [[2.0,1.0],[4.0,2.1]]`), `linreg_multioutput_no_intercept` (coef `[[2.0195121951,-0.0097560976],[0.9609756098,0.1195121951]]`, intercepts 0), `linreg_single_output_unchanged` (1-D path byte-identical). Closes #372. |
//! | REQ-8 (sample_weight in fit) | SHIPPED | `LinearRegression::fit_with_sample_weight` solves WEIGHTED least squares `min Σᵢ wᵢ(yᵢ−xᵢ·w)²`: weighted offsets `x_off[j]=Σwᵢx[i,j]/Σwᵢ`, `y_off=Σwᵢyᵢ/Σwᵢ` (mirrors `_average(...,weights=sample_weight)`, `_base.py:193`/`:198`), centering, then `√wᵢ` row-rescaling (`_rescale_data`, `_base.py:641`), `linalg::solve_lstsq` on the rescaled design, `intercept = y_off − x_off·coef` (`_set_intercept`, `_base.py:320`); `fit_intercept=false` skips centering, intercept 0. `Fit::fit` delegates `fit_with_sample_weight(x, y, None)` (None path byte-identical to the historic OLS body). Oracle tests `linreg_fit_sample_weight_with_intercept_matches_sklearn` (coef 2.0935828877, intercept −0.2326203209), `linreg_fit_sample_weight_no_intercept_matches_sklearn` (coef 2.0350877193, intercept 0), `linreg_fit_none_sample_weight_equals_unweighted`. Mirrors `fit(..., sample_weight=None)` (`_base.py:582`). Closes #373. |
//! | REQ-9 (rank_/singular_/copy_X/n_jobs) | SHIPPED | `FittedLinearRegression` stores `rank_`/`singular_` (captured from `linalg::solve_lstsq` on the matrix actually solved — centered `X` when `fit_intercept`, raw `X` otherwise, matching sklearn `_base.py:687`), exposed via `rank()`/`singular_values()`; `LinearRegression` adds `copy_x` (default `true`) + `n_jobs` (default `None`) fields with `with_copy_x`/`with_n_jobs` builders, mirroring `_parameter_constraints` (`_base.py:561`) and the ctor (`_base.py:572-573`). `copy_x` is ABI-only (fit never mutates `x`); `n_jobs` stored-but-ignored (single-threaded). Oracle tests `linreg_rank_singular_match_sklearn_with_intercept` (rank 2, singular `[1.61803399, 0.61803399]` on centered X), `linreg_singular_no_intercept_matches_raw_x` (singular `[5.25371017, 0.63129192]` on raw X), `linreg_copy_x_default_and_builder`. Closes #374. |
//! | REQ-10 (ferray substrate) | NOT-STARTED | blocker #375 — OLS solve now on `ferray::linalg::lstsq`, but `LinearRegression`'s coef storage is still `ndarray` (coef return type tied to #359); fully on-substrate when the boundary `ndarray` types migrate. |
//!
//! Two states only per goal.md R-DEFER-2. The OLS min-norm contract (#376/#377)
//! is fixed in `linalg.rs` via the ferray substrate.
//!
//! # Examples
//!
//! ```
//! use ferrolearn_linear::LinearRegression;
//! use ferrolearn_core::{Fit, Predict};
//! use ndarray::{array, Array1, Array2};
//!
//! let model = LinearRegression::<f64>::new();
//! let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
//! let y = array![2.0, 4.0, 6.0, 8.0];
//!
//! let fitted = model.fit(&x, &y).unwrap();
//! let preds = fitted.predict(&x).unwrap();
//! ```

use ferrolearn_core::error::FerroError;
use ferrolearn_core::introspection::HasCoefficients;
use ferrolearn_core::pipeline::{FittedPipelineEstimator, PipelineEstimator};
use ferrolearn_core::traits::{Fit, Predict};
use ndarray::{Array1, Array2, Axis, ScalarOperand};
use num_traits::{Float, FromPrimitive};

use crate::linalg;

/// Ordinary least squares linear regression.
///
/// Solves the least-squares problem via a single SVD (minimum-norm,
/// LAPACK-`gelsd`-equivalent, through `ferray::linalg::lstsq`). The
/// `fit_intercept` option controls whether a bias (intercept) term is
/// included.
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
#[derive(Debug, Clone)]
pub struct LinearRegression<F> {
    /// Whether to fit an intercept (bias) term.
    pub fit_intercept: bool,
    /// Whether `X` may be overwritten during fit (sklearn `copy_X`,
    /// `_base.py:480`). ferrolearn's `fit` never mutates `x` (it reads via
    /// `.iter()`/`.mean_axis()`), so the observable non-mutation contract
    /// holds for either value; the field is exposed for ABI parity. Default
    /// `true`, matching sklearn (`_base.py:572`).
    pub copy_x: bool,
    /// Number of jobs for the computation (sklearn `n_jobs`, `_base.py:483`).
    /// ferrolearn's dense OLS solve is single-threaded, so this is stored but
    /// ignored — parallelism is a no-op here and behaviour matches sklearn's
    /// `n_jobs=None` single-job default. Default `None` (`_base.py:573`).
    pub n_jobs: Option<usize>,
    _marker: std::marker::PhantomData<F>,
}

impl<
    F: Float
        + Send
        + Sync
        + ScalarOperand
        + num_traits::FromPrimitive
        + ferray::linalg::LinalgFloat
        + 'static,
> LinearRegression<F>
{
    /// Fit the linear regression model with optional per-sample weights.
    ///
    /// Mirrors scikit-learn's `LinearRegression.fit(X, y, sample_weight=None)`
    /// (`sklearn/linear_model/_base.py:582`). When `sample_weight` is `Some(w)`,
    /// this solves the WEIGHTED least-squares problem `min Σᵢ wᵢ (yᵢ − xᵢ·w)²`:
    ///
    /// - `fit_intercept=true`: offsets are the WEIGHTED means
    ///   `x_off[j] = Σᵢ wᵢ·x[i,j] / Σwᵢ`, `y_off = Σᵢ wᵢ·yᵢ / Σwᵢ`
    ///   (sklearn `_preprocess_data` → `_average(..., weights=sample_weight)`,
    ///   `_base.py:193`/`:198`). `X` and `y` are centered by those offsets, each
    ///   row is then rescaled by `√wᵢ` (sklearn `_rescale_data`, `_base.py:641`),
    ///   `linalg.lstsq` solves for `coef`, and
    ///   `intercept = y_off − x_off · coef` (`_set_intercept`, `_base.py:320`).
    /// - `fit_intercept=false`: no centering; each row is rescaled by `√wᵢ`, the
    ///   solve runs on the rescaled `X`, and `intercept = 0`.
    ///
    /// `sample_weight=None` is BYTE-IDENTICAL to [`Fit::fit`] (the unweighted
    /// centering + `solve_lstsq` path), which delegates here.
    ///
    /// `rank_`/`singular_` are captured from `solve_lstsq` on the matrix actually
    /// solved (centered-and-rescaled `X` when `fit_intercept`, rescaled `X`
    /// otherwise), matching sklearn's `linalg.lstsq` operands (`_base.py:687`).
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of samples in `x` and
    /// `y` (or, when provided, `sample_weight`) differ.
    /// Returns [`FerroError::InsufficientSamples`] if there are no samples.
    /// Returns [`FerroError::NumericalInstability`] if the system is singular or
    /// the weighted-offset denominator (`Σwᵢ`) cannot be formed.
    pub fn fit_with_sample_weight(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
        sample_weight: Option<&Array1<F>>,
    ) -> Result<FittedLinearRegression<F>, FerroError> {
        let (n_samples, n_features) = x.dim();

        // Validate input shapes.
        if n_samples != y.len() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_samples],
                actual: vec![y.len()],
                context: "y length must match number of samples in X".into(),
            });
        }

        if n_samples == 0 {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "LinearRegression requires at least one sample".into(),
            });
        }

        if let Some(w) = sample_weight
            && w.len() != n_samples
        {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_samples],
                actual: vec![w.len()],
                context: "sample_weight length must match number of samples in X".into(),
            });
        }

        match sample_weight {
            None => {
                // Unweighted path — identical to the original `Fit::fit` body.
                if self.fit_intercept {
                    // Centering trick: center X and y, solve the (uncentered)
                    // OLS problem on the centered design, then recover the
                    // intercept as y_mean - x_mean . w. sklearn centers
                    // identically before its `linalg.lstsq` call (`_base.py`
                    // `_preprocess_data` + `:687`).
                    let n = <F as num_traits::NumCast>::from(n_samples).ok_or_else(|| {
                        FerroError::NumericalInstability {
                            message: "could not represent n_samples as the float type".into(),
                        }
                    })?;
                    let x_mean =
                        x.mean_axis(Axis(0))
                            .ok_or_else(|| FerroError::InsufficientSamples {
                                required: 1,
                                actual: 0,
                                context: "cannot compute feature means of an empty design".into(),
                            })?;
                    let y_mean = y.sum() / n;

                    let x_centered = x - &x_mean;
                    let y_centered = y - y_mean;

                    let (w, rank, singular) = linalg::solve_lstsq(&x_centered, &y_centered)?;

                    let intercept = y_mean - x_mean.dot(&w);

                    Ok(FittedLinearRegression {
                        coefficients: w,
                        intercept,
                        rank_: rank,
                        singular_: singular,
                    })
                } else {
                    let (w, rank, singular) = linalg::solve_lstsq(x, y)?;

                    Ok(FittedLinearRegression {
                        coefficients: w,
                        intercept: <F as num_traits::Zero>::zero(),
                        rank_: rank,
                        singular_: singular,
                    })
                }
            }
            Some(w) => {
                // Per-row √w factor (sklearn `_rescale_data`, `_base.py:641`).
                let w_sqrt = w.mapv(<F as Float>::sqrt);

                if self.fit_intercept {
                    // WEIGHTED centering: offsets are the weighted means
                    // x_off[j] = Σ wᵢ x[i,j] / Σ wᵢ, y_off = Σ wᵢ yᵢ / Σ wᵢ
                    // (sklearn `_average(..., weights=sample_weight)`,
                    // `_base.py:193`/`:198`).
                    let w_sum = w.sum();
                    if w_sum <= <F as num_traits::Zero>::zero() {
                        return Err(FerroError::NumericalInstability {
                            message: "sum of sample_weight must be positive to center".into(),
                        });
                    }

                    let mut x_off = Array1::<F>::zeros(n_features);
                    for (i, row) in x.outer_iter().enumerate() {
                        let wi = w[i];
                        x_off = &x_off + &row.mapv(|v| v * wi);
                    }
                    x_off.mapv_inplace(|v| v / w_sum);

                    let y_off = y
                        .iter()
                        .zip(w.iter())
                        .fold(<F as num_traits::Zero>::zero(), |acc, (&yi, &wi)| {
                            acc + wi * yi
                        })
                        / w_sum;

                    // Center, then row-rescale by √w.
                    let x_centered = x - &x_off;
                    let y_centered = y - y_off;
                    let x_scaled = &x_centered * &w_sqrt.view().insert_axis(Axis(1));
                    let y_scaled = &y_centered * &w_sqrt;

                    let (coef, rank, singular) = linalg::solve_lstsq(&x_scaled, &y_scaled)?;

                    let intercept = y_off - x_off.dot(&coef);

                    Ok(FittedLinearRegression {
                        coefficients: coef,
                        intercept,
                        rank_: rank,
                        singular_: singular,
                    })
                } else {
                    // No centering; just √w row-rescaling, intercept 0.
                    let x_scaled = x * &w_sqrt.view().insert_axis(Axis(1));
                    let y_scaled = y * &w_sqrt;

                    let (coef, rank, singular) = linalg::solve_lstsq(&x_scaled, &y_scaled)?;

                    Ok(FittedLinearRegression {
                        coefficients: coef,
                        intercept: <F as num_traits::Zero>::zero(),
                        rank_: rank,
                        singular_: singular,
                    })
                }
            }
        }
    }
}

impl<F: Float> LinearRegression<F> {
    /// Create a new `LinearRegression` with default settings.
    ///
    /// Defaults: `fit_intercept = true`, `copy_x = true`, `n_jobs = None`
    /// (mirroring sklearn's ctor defaults, `_base.py:571-573`).
    #[must_use]
    pub fn new() -> Self {
        Self {
            fit_intercept: true,
            copy_x: true,
            n_jobs: None,
            _marker: std::marker::PhantomData,
        }
    }

    /// Set whether to fit an intercept term.
    #[must_use]
    pub fn with_fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = fit_intercept;
        self
    }

    /// Set the `copy_X` flag (sklearn `copy_X`, `_base.py:480`).
    ///
    /// ferrolearn's fit never mutates `x`, so this is exposed for ABI parity
    /// with sklearn and does not change the result.
    #[must_use]
    pub fn with_copy_x(mut self, copy_x: bool) -> Self {
        self.copy_x = copy_x;
        self
    }

    /// Set the `n_jobs` parameter (sklearn `n_jobs`, `_base.py:483`).
    ///
    /// The dense OLS solve is single-threaded; this is stored but ignored.
    #[must_use]
    pub fn with_n_jobs(mut self, n_jobs: Option<usize>) -> Self {
        self.n_jobs = n_jobs;
        self
    }
}

impl<F: Float> Default for LinearRegression<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted ordinary least squares linear regression model.
///
/// Stores the learned coefficients and intercept. Implements [`Predict`]
/// to generate predictions and [`HasCoefficients`] for introspection.
#[derive(Debug, Clone)]
pub struct FittedLinearRegression<F> {
    /// Learned coefficient vector (one per feature).
    coefficients: Array1<F>,
    /// Learned intercept (bias) term.
    intercept: F,
    /// Effective rank of the design matrix actually solved (sklearn `rank_`,
    /// `_base.py:505`/`:687`) — the centered `X` when `fit_intercept`, the
    /// raw `X` otherwise.
    rank_: usize,
    /// Singular values of the design matrix actually solved (sklearn
    /// `singular_`, `_base.py:508`/`:687`).
    singular_: Array1<F>,
}

impl<
    F: Float
        + Send
        + Sync
        + ScalarOperand
        + num_traits::FromPrimitive
        + ferray::linalg::LinalgFloat
        + 'static,
> Fit<Array2<F>, Array1<F>> for LinearRegression<F>
{
    type Fitted = FittedLinearRegression<F>;
    type Error = FerroError;

    /// Fit the linear regression model.
    ///
    /// Solves the OLS least-squares problem via the SVD-based
    /// minimum-norm solver [`crate::linalg::solve_lstsq`] (routed through
    /// [`ferray::linalg::lstsq`], LAPACK-`gelsd`-equivalent), matching
    /// scikit-learn's dense path `linalg.lstsq(X, y)`
    /// (`sklearn/linear_model/_base.py:687`). When `fit_intercept` is true,
    /// `X` and `y` are centered first and the intercept is recovered as
    /// `y_mean - x_mean . w`.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of samples in `x`
    /// and `y` differ.
    /// Returns [`FerroError::InsufficientSamples`] if there are fewer samples
    /// than features.
    /// Returns [`FerroError::NumericalInstability`] if the system is singular.
    fn fit(&self, x: &Array2<F>, y: &Array1<F>) -> Result<FittedLinearRegression<F>, FerroError> {
        // Unweighted OLS is the `sample_weight=None` arm of the weighted fit;
        // delegating keeps the None path byte-identical to the historic body
        // (centering + `solve_lstsq`), mirroring sklearn's single `fit` entry
        // (`_base.py:582`, `sample_weight=None` default).
        self.fit_with_sample_weight(x, y, None)
    }
}

/// Fitted multi-output ordinary least squares linear regression model.
///
/// The 2-D-target companion to [`FittedLinearRegression`]: produced by
/// `Fit<Array2<F>, Array2<F>>` when fitting a 2-D `Y` of shape
/// `(n_samples, n_targets)`. Mirrors scikit-learn's multi-output
/// `LinearRegression` (`MultiOutputMixin`, `_base.py:465`), whose `coef_` is a
/// 2-D array of shape `(n_targets, n_features)` and `intercept_` an array of
/// shape `(n_targets,)` (`_base.py:499`/`:511`). Stored in sklearn's `coef_`
/// orientation (target rows), so `coefficients()` maps directly onto
/// `sklearn.coef_`.
#[derive(Debug, Clone)]
pub struct FittedMultiOutputLinearRegression<F> {
    /// Learned coefficient matrix in sklearn `coef_` orientation: shape
    /// `(n_targets, n_features)`, row `t` the coefficients for target `t`
    /// (`_base.py:499`).
    coefficients: Array2<F>,
    /// Learned per-target intercepts, shape `(n_targets,)` (sklearn
    /// `intercept_`, `_base.py:511`).
    intercepts: Array1<F>,
    /// Effective rank of the design matrix actually solved (sklearn `rank_`,
    /// `_base.py:505`/`:687`) — the centered `X` when `fit_intercept`, the raw
    /// `X` otherwise.
    rank_: usize,
    /// Singular values of the design matrix actually solved (sklearn
    /// `singular_`, `_base.py:508`/`:687`).
    singular_: Array1<F>,
}

impl<F: Float> FittedMultiOutputLinearRegression<F> {
    /// Learned coefficient matrix, shape `(n_targets, n_features)` (sklearn
    /// `coef_`, `_base.py:499`).
    #[must_use]
    pub fn coefficients(&self) -> &Array2<F> {
        &self.coefficients
    }

    /// Learned per-target intercepts, shape `(n_targets,)` (sklearn
    /// `intercept_`, `_base.py:511`).
    #[must_use]
    pub fn intercepts(&self) -> &Array1<F> {
        &self.intercepts
    }

    /// Effective rank of the design matrix (sklearn `rank_`, `_base.py:505`).
    #[must_use]
    pub fn rank(&self) -> usize {
        self.rank_
    }

    /// Singular values of the design matrix (sklearn `singular_`,
    /// `_base.py:508`).
    #[must_use]
    pub fn singular_values(&self) -> &Array1<F> {
        &self.singular_
    }
}

impl<
    F: Float
        + Send
        + Sync
        + ScalarOperand
        + num_traits::FromPrimitive
        + ferray::linalg::LinalgFloat
        + 'static,
> Fit<Array2<F>, Array2<F>> for LinearRegression<F>
{
    type Fitted = FittedMultiOutputLinearRegression<F>;
    type Error = FerroError;

    /// Fit the multi-output linear regression model on a 2-D target `Y`.
    ///
    /// Mirrors scikit-learn's multi-output dense path: `linalg.lstsq(X, Y)`
    /// with `Y` of shape `(n_samples, n_targets)` solves all targets in one
    /// SVD, yielding `coef_` of shape `(n_targets, n_features)` and a per-target
    /// `intercept_` of shape `(n_targets,)` (`sklearn/linear_model/_base.py:687`,
    /// `coef_.T`; intercept `_set_intercept`, `_base.py:308`/`:322`). When
    /// `fit_intercept` is true, `X` and each column of `Y` are centered by their
    /// column means and the intercept is recovered as
    /// `y_off − coefficients · x_off` per target; when false, the solve runs on
    /// raw `X`/`Y` and the intercepts are all `0`.
    ///
    /// The 1-D `Fit<Array2<F>, Array1<F>>` impl is unaffected — this is an
    /// additive 2-D arm.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of samples in `x`
    /// and `y` differ.
    /// Returns [`FerroError::InsufficientSamples`] if there are no samples.
    /// Returns [`FerroError::NumericalInstability`] if the system is singular.
    fn fit(
        &self,
        x: &Array2<F>,
        y: &Array2<F>,
    ) -> Result<FittedMultiOutputLinearRegression<F>, FerroError> {
        let n_samples = x.nrows();
        let n_targets = y.ncols();

        if n_samples != y.nrows() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_samples],
                actual: vec![y.nrows()],
                context: "Y rows must match number of samples in X".into(),
            });
        }

        if n_samples == 0 {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "LinearRegression requires at least one sample".into(),
            });
        }

        if self.fit_intercept {
            // Same column-centering as the 1-D fit, generalized to Y's columns
            // (sklearn `_preprocess_data` centers X and every column of Y by
            // their per-column means, `_base.py:193`/`:198`).
            let x_off = x
                .mean_axis(Axis(0))
                .ok_or_else(|| FerroError::InsufficientSamples {
                    required: 1,
                    actual: 0,
                    context: "cannot compute feature means of an empty design".into(),
                })?;
            let y_off = y
                .mean_axis(Axis(0))
                .ok_or_else(|| FerroError::InsufficientSamples {
                    required: 1,
                    actual: 0,
                    context: "cannot compute target means of an empty Y".into(),
                })?;

            let x_centered = x - &x_off;
            let y_centered = y - &y_off;

            // coef_ft is (n_features, n_targets); store in sklearn `coef_`
            // orientation (n_targets, n_features) via transpose.
            let (coef_ft, rank, singular) = linalg::solve_lstsq_multi(&x_centered, &y_centered)?;
            let coefficients = coef_ft.t().to_owned();

            // intercept_[t] = y_off[t] − coefficients[t] · x_off
            // (sklearn `_set_intercept`: `y_offset − X_offset @ coef_.T`,
            // `_base.py:322`).
            let intercepts = &y_off - &coefficients.dot(&x_off);

            Ok(FittedMultiOutputLinearRegression {
                coefficients,
                intercepts,
                rank_: rank,
                singular_: singular,
            })
        } else {
            let (coef_ft, rank, singular) = linalg::solve_lstsq_multi(x, y)?;
            let coefficients = coef_ft.t().to_owned();
            let intercepts = Array1::<F>::zeros(n_targets);

            Ok(FittedMultiOutputLinearRegression {
                coefficients,
                intercepts,
                rank_: rank,
                singular_: singular,
            })
        }
    }
}

impl<F: Float + Send + Sync + ScalarOperand + 'static> Predict<Array2<F>>
    for FittedMultiOutputLinearRegression<F>
{
    type Output = Array2<F>;
    type Error = FerroError;

    /// Predict 2-D target values for the given feature matrix.
    ///
    /// Computes `X @ coefficients.T + intercepts` (broadcasting the per-target
    /// intercepts over rows), shape `(n_samples, n_targets)`, mirroring sklearn's
    /// 2-D `_decision_function` arm `X @ coef_.T + self.intercept_`
    /// (`_base.py:290`).
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of features does not
    /// match the fitted model.
    fn predict(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let n_features = x.ncols();
        if n_features != self.coefficients.ncols() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![self.coefficients.ncols()],
                actual: vec![n_features],
                context: "number of features must match fitted model".into(),
            });
        }

        // X (n_samples, n_features) @ coef_.T (n_features, n_targets) -> (n_samples, n_targets)
        let preds = x.dot(&self.coefficients.t());
        Ok(preds + &self.intercepts)
    }
}

impl<F: Float> FittedLinearRegression<F> {
    /// Effective rank of the design matrix (sklearn `rank_`, `_base.py:505`).
    ///
    /// The rank of the matrix actually solved by `linalg.lstsq` — the
    /// centered `X` when `fit_intercept` is true, the raw `X` otherwise
    /// (`_base.py:687`).
    #[must_use]
    pub fn rank(&self) -> usize {
        self.rank_
    }

    /// Singular values of the design matrix (sklearn `singular_`,
    /// `_base.py:508`).
    ///
    /// The singular values of the matrix actually solved by `linalg.lstsq`
    /// — the centered `X` when `fit_intercept` is true, the raw `X`
    /// otherwise (`_base.py:687`).
    #[must_use]
    pub fn singular_values(&self) -> &Array1<F> {
        &self.singular_
    }
}

impl<F: Float + Send + Sync + ScalarOperand + 'static> Predict<Array2<F>>
    for FittedLinearRegression<F>
{
    type Output = Array1<F>;
    type Error = FerroError;

    /// Predict target values for the given feature matrix.
    ///
    /// Computes `X @ coefficients + intercept`.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of features
    /// does not match the fitted model.
    fn predict(&self, x: &Array2<F>) -> Result<Array1<F>, FerroError> {
        let n_features = x.ncols();
        if n_features != self.coefficients.len() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![self.coefficients.len()],
                actual: vec![n_features],
                context: "number of features must match fitted model".into(),
            });
        }

        let preds = x.dot(&self.coefficients) + self.intercept;
        Ok(preds)
    }
}

impl<F: Float + Send + Sync + ScalarOperand + 'static> HasCoefficients<F>
    for FittedLinearRegression<F>
{
    fn coefficients(&self) -> &Array1<F> {
        &self.coefficients
    }

    fn intercept(&self) -> F {
        self.intercept
    }
}

// Pipeline integration.
impl<F> PipelineEstimator<F> for LinearRegression<F>
where
    F: Float + FromPrimitive + ScalarOperand + ferray::linalg::LinalgFloat + Send + Sync + 'static,
{
    fn fit_pipeline(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
    ) -> Result<Box<dyn FittedPipelineEstimator<F>>, FerroError> {
        let fitted = self.fit(x, y)?;
        Ok(Box::new(fitted))
    }
}

impl<F> FittedPipelineEstimator<F> for FittedLinearRegression<F>
where
    F: Float + ScalarOperand + Send + Sync + 'static,
{
    fn predict_pipeline(&self, x: &Array2<F>) -> Result<Array1<F>, FerroError> {
        self.predict(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn test_simple_linear_regression() {
        // y = 2*x + 1
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = array![3.0, 5.0, 7.0, 9.0, 11.0];

        let model = LinearRegression::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();

        assert_relative_eq!(fitted.coefficients()[0], 2.0, epsilon = 1e-10);
        assert_relative_eq!(fitted.intercept(), 1.0, epsilon = 1e-10);

        let preds = fitted.predict(&x).unwrap();
        for (p, &actual) in preds.iter().zip(y.iter()) {
            assert_relative_eq!(*p, actual, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_multiple_linear_regression() {
        // y = 1*x1 + 2*x2 + 3
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 2.0, 1.0, 3.0, 2.0, 4.0, 2.0]).unwrap();
        let y = array![6.0, 7.0, 10.0, 11.0];

        let model = LinearRegression::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();

        assert_relative_eq!(fitted.coefficients()[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(fitted.coefficients()[1], 2.0, epsilon = 1e-10);
        assert_relative_eq!(fitted.intercept(), 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_no_intercept() {
        // y = 2*x (through origin)
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![2.0, 4.0, 6.0, 8.0];

        let model = LinearRegression::<f64>::new().with_fit_intercept(false);
        let fitted = model.fit(&x, &y).unwrap();

        assert_relative_eq!(fitted.coefficients()[0], 2.0, epsilon = 1e-10);
        assert_relative_eq!(fitted.intercept(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_shape_mismatch_fit() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![1.0, 2.0]; // Wrong length

        let model = LinearRegression::<f64>::new();
        let result = model.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_shape_mismatch_predict() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = array![1.0, 2.0, 3.0];

        let model = LinearRegression::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();

        // Wrong number of features
        let x_bad = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let result = fitted.predict(&x_bad);
        assert!(result.is_err());
    }

    #[test]
    fn test_has_coefficients() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![2.0, 4.0, 6.0];

        let model = LinearRegression::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();

        assert_eq!(fitted.coefficients().len(), 1);
    }

    #[test]
    fn test_pipeline_integration() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![3.0, 5.0, 7.0, 9.0];

        let model = LinearRegression::<f64>::new();
        let fitted = model.fit_pipeline(&x, &y).unwrap();
        let preds = fitted.predict_pipeline(&x).unwrap();
        assert_eq!(preds.len(), 4);
    }

    #[test]
    fn linreg_rank_singular_match_sklearn_with_intercept() {
        // Live sklearn 1.5.2 oracle (fit_intercept=True centers X before
        // linalg.lstsq, so singular_ are the singular values of CENTERED X):
        //   cd /tmp && python3 -c "import numpy as np; \
        //     from sklearn.linear_model import LinearRegression; \
        //     X=np.array([[1.,1.],[1.,2.],[2.,2.],[2.,3.]]); \
        //     y=np.array([6.,8.,9.,11.]); m=LinearRegression().fit(X,y); \
        //     print(m.rank_, [round(s,8) for s in m.singular_])"
        //   -> 2 [1.61803399, 0.61803399]
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0]).unwrap();
        let y = array![6.0, 8.0, 9.0, 11.0];

        let model = LinearRegression::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();

        assert_eq!(fitted.rank(), 2);
        let sv = fitted.singular_values();
        assert_eq!(sv.len(), 2);
        assert_relative_eq!(sv[0], 1.618_033_99, epsilon = 1e-6);
        assert_relative_eq!(sv[1], 0.618_033_99, epsilon = 1e-6);
    }

    #[test]
    fn linreg_singular_no_intercept_matches_raw_x() {
        // Live sklearn 1.5.2 oracle (fit_intercept=False → singular_ are the
        // singular values of the RAW X):
        //   cd /tmp && python3 -c "import numpy as np; \
        //     from sklearn.linear_model import LinearRegression; \
        //     X=np.array([[1.,1.],[1.,2.],[2.,2.],[2.,3.]]); \
        //     y=np.array([6.,8.,9.,11.]); \
        //     m=LinearRegression(fit_intercept=False).fit(X,y); \
        //     print(m.rank_, [round(s,8) for s in m.singular_])"
        //   -> 2 [5.25371017, 0.63129192]
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0]).unwrap();
        let y = array![6.0, 8.0, 9.0, 11.0];

        let model = LinearRegression::<f64>::new().with_fit_intercept(false);
        let fitted = model.fit(&x, &y).unwrap();

        assert_eq!(fitted.rank(), 2);
        let sv = fitted.singular_values();
        assert_eq!(sv.len(), 2);
        assert_relative_eq!(sv[0], 5.253_710_17, epsilon = 1e-6);
        assert_relative_eq!(sv[1], 0.631_291_92, epsilon = 1e-6);
    }

    #[test]
    fn linreg_copy_x_default_and_builder() {
        // copy_X default is true (sklearn `_base.py:572`); the builder flips
        // it; n_jobs builder stores Some(4); and fit produces identical coef_
        // regardless of copy_x (no behaviour change — fit never mutates X).
        assert!(LinearRegression::<f64>::new().copy_x);
        assert!(!LinearRegression::<f64>::new().with_copy_x(false).copy_x);
        assert_eq!(
            LinearRegression::<f64>::new().with_n_jobs(Some(4)).n_jobs,
            Some(4)
        );

        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![3.0, 5.0, 7.0, 9.0];

        let fitted_copy = LinearRegression::<f64>::new()
            .with_copy_x(true)
            .fit(&x, &y)
            .unwrap();
        let fitted_nocopy = LinearRegression::<f64>::new()
            .with_copy_x(false)
            .fit(&x, &y)
            .unwrap();

        assert_relative_eq!(
            fitted_copy.coefficients()[0],
            fitted_nocopy.coefficients()[0],
            epsilon = 1e-12
        );
        assert_relative_eq!(
            fitted_copy.intercept(),
            fitted_nocopy.intercept(),
            epsilon = 1e-12
        );
    }

    #[test]
    fn linreg_fit_sample_weight_with_intercept_matches_sklearn() {
        // Live sklearn 1.5.2 oracle (WEIGHTED OLS, fit_intercept=True):
        //   cd /tmp && python3 -c "import numpy as np; \
        //     from sklearn.linear_model import LinearRegression; \
        //     X=np.array([[1.],[2.],[3.],[4.],[5.]]); \
        //     y=np.array([2.1,3.9,6.2,7.7,10.3]); w=np.array([1.,5.,1.,1.,5.]); \
        //     m=LinearRegression().fit(X,y,sample_weight=w); \
        //     print(round(m.coef_[0],10), round(m.intercept_,10))"
        //   -> 2.0935828877 -0.2326203209
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = array![2.1, 3.9, 6.2, 7.7, 10.3];
        let w = array![1.0, 5.0, 1.0, 1.0, 5.0];

        let model = LinearRegression::<f64>::new();
        let fitted = model.fit_with_sample_weight(&x, &y, Some(&w)).unwrap();

        assert_relative_eq!(fitted.coefficients()[0], 2.093_582_887_7, epsilon = 1e-7);
        assert_relative_eq!(fitted.intercept(), -0.232_620_320_9, epsilon = 1e-7);

        // Non-tautological: the weighted result MUST differ from the unweighted
        // fit (oracle unweighted coef_ 2.02, intercept_ -0.02).
        let unweighted = model.fit(&x, &y).unwrap();
        assert_relative_eq!(unweighted.coefficients()[0], 2.02, epsilon = 1e-7);
        assert!((fitted.coefficients()[0] - unweighted.coefficients()[0]).abs() > 1e-3);
        assert!((fitted.intercept() - unweighted.intercept()).abs() > 1e-3);
    }

    #[test]
    fn linreg_fit_sample_weight_no_intercept_matches_sklearn() {
        // Live sklearn 1.5.2 oracle (WEIGHTED OLS, fit_intercept=False):
        //   cd /tmp && python3 -c "import numpy as np; \
        //     from sklearn.linear_model import LinearRegression; \
        //     X=np.array([[1.],[2.],[3.],[4.],[5.]]); \
        //     y=np.array([2.1,3.9,6.2,7.7,10.3]); w=np.array([1.,5.,1.,1.,5.]); \
        //     m=LinearRegression(fit_intercept=False).fit(X,y,sample_weight=w); \
        //     print(round(m.coef_[0],10))"
        //   -> 2.0350877193
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = array![2.1, 3.9, 6.2, 7.7, 10.3];
        let w = array![1.0, 5.0, 1.0, 1.0, 5.0];

        let model = LinearRegression::<f64>::new().with_fit_intercept(false);
        let fitted = model.fit_with_sample_weight(&x, &y, Some(&w)).unwrap();

        assert_relative_eq!(fitted.coefficients()[0], 2.035_087_719_3, epsilon = 1e-7);
        assert_eq!(fitted.intercept(), 0.0);
    }

    #[test]
    fn linreg_fit_none_sample_weight_equals_unweighted() {
        // Regression guard: the `None` path is BYTE-IDENTICAL to `fit`.
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = array![2.1, 3.9, 6.2, 7.7, 10.3];

        let model = LinearRegression::<f64>::new();
        let via_fit = model.fit(&x, &y).unwrap();
        let via_none = model.fit_with_sample_weight(&x, &y, None).unwrap();

        assert_eq!(
            via_fit.coefficients()[0].to_bits(),
            via_none.coefficients()[0].to_bits()
        );
        assert_eq!(
            via_fit.intercept().to_bits(),
            via_none.intercept().to_bits()
        );

        // Same for fit_intercept=false.
        let model_ni = LinearRegression::<f64>::new().with_fit_intercept(false);
        let via_fit_ni = model_ni.fit(&x, &y).unwrap();
        let via_none_ni = model_ni.fit_with_sample_weight(&x, &y, None).unwrap();
        assert_eq!(
            via_fit_ni.coefficients()[0].to_bits(),
            via_none_ni.coefficients()[0].to_bits()
        );
        assert_eq!(
            via_fit_ni.intercept().to_bits(),
            via_none_ni.intercept().to_bits()
        );
    }

    #[test]
    fn linreg_multioutput_coef_intercept_match_sklearn() {
        // Live sklearn 1.5.2 oracle (multi-output, fit_intercept=True):
        //   cd /tmp && python3 -c "import numpy as np; \
        //     from sklearn.linear_model import LinearRegression; \
        //     X=np.array([[1.,0.],[2.,1.],[3.,1.],[4.,2.],[5.,3.]]); \
        //     Y=np.array([[2.1,1.0],[3.9,2.1],[6.2,2.9],[7.7,4.2],[10.3,5.1]]); \
        //     m=LinearRegression().fit(X,Y); print(m.coef_.shape); \
        //     print([[round(v,8) for v in r] for r in m.coef_]); \
        //     print([round(v,8) for v in m.intercept_])"
        //   -> (2, 2)
        //      [[2.06666667, -0.06666667], [0.86666667, 0.23333333]]
        //      [-0.06666667, 0.13333333]
        let x = Array2::from_shape_vec(
            (5, 2),
            vec![1.0, 0.0, 2.0, 1.0, 3.0, 1.0, 4.0, 2.0, 5.0, 3.0],
        )
        .unwrap();
        let y = Array2::from_shape_vec(
            (5, 2),
            vec![2.1, 1.0, 3.9, 2.1, 6.2, 2.9, 7.7, 4.2, 10.3, 5.1],
        )
        .unwrap();

        let model = LinearRegression::<f64>::new();
        let fitted = Fit::<Array2<f64>, Array2<f64>>::fit(&model, &x, &y).unwrap();

        assert_eq!(fitted.coefficients().dim(), (2, 2));
        let c = fitted.coefficients();
        assert_relative_eq!(c[[0, 0]], 2.066_666_67, epsilon = 1e-7);
        assert_relative_eq!(c[[0, 1]], -0.066_666_67, epsilon = 1e-7);
        assert_relative_eq!(c[[1, 0]], 0.866_666_67, epsilon = 1e-7);
        assert_relative_eq!(c[[1, 1]], 0.233_333_33, epsilon = 1e-7);

        let b = fitted.intercepts();
        assert_eq!(b.len(), 2);
        assert_relative_eq!(b[0], -0.066_666_67, epsilon = 1e-7);
        assert_relative_eq!(b[1], 0.133_333_33, epsilon = 1e-7);
    }

    #[test]
    fn linreg_multioutput_predict_shape_and_values() {
        // Oracle (same model as above): predict(X).shape == (5, 2);
        //   m.predict(X[:2]) -> [[2.0, 1.0], [4.0, 2.1]]
        let x = Array2::from_shape_vec(
            (5, 2),
            vec![1.0, 0.0, 2.0, 1.0, 3.0, 1.0, 4.0, 2.0, 5.0, 3.0],
        )
        .unwrap();
        let y = Array2::from_shape_vec(
            (5, 2),
            vec![2.1, 1.0, 3.9, 2.1, 6.2, 2.9, 7.7, 4.2, 10.3, 5.1],
        )
        .unwrap();

        let model = LinearRegression::<f64>::new();
        let fitted = Fit::<Array2<f64>, Array2<f64>>::fit(&model, &x, &y).unwrap();

        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.dim(), (5, 2));

        let x2 = x.slice(ndarray::s![0..2, ..]).to_owned();
        let preds2 = fitted.predict(&x2).unwrap();
        assert_eq!(preds2.dim(), (2, 2));
        assert_relative_eq!(preds2[[0, 0]], 2.0, epsilon = 1e-6);
        assert_relative_eq!(preds2[[0, 1]], 1.0, epsilon = 1e-6);
        assert_relative_eq!(preds2[[1, 0]], 4.0, epsilon = 1e-6);
        assert_relative_eq!(preds2[[1, 1]], 2.1, epsilon = 1e-6);
    }

    #[test]
    fn linreg_multioutput_no_intercept() {
        // Live sklearn 1.5.2 oracle (multi-output, fit_intercept=False):
        //   cd /tmp && python3 -c "import numpy as np; \
        //     from sklearn.linear_model import LinearRegression; \
        //     X=np.array([[1.,0.],[2.,1.],[3.,1.],[4.,2.],[5.,3.]]); \
        //     Y=np.array([[2.1,1.0],[3.9,2.1],[6.2,2.9],[7.7,4.2],[10.3,5.1]]); \
        //     m=LinearRegression(fit_intercept=False).fit(X,Y); \
        //     print([[round(v,10) for v in r] for r in m.coef_]); print(m.intercept_)"
        //   -> [[2.0195121951, -0.0097560976], [0.9609756098, 0.1195121951]]
        //      0.0
        let x = Array2::from_shape_vec(
            (5, 2),
            vec![1.0, 0.0, 2.0, 1.0, 3.0, 1.0, 4.0, 2.0, 5.0, 3.0],
        )
        .unwrap();
        let y = Array2::from_shape_vec(
            (5, 2),
            vec![2.1, 1.0, 3.9, 2.1, 6.2, 2.9, 7.7, 4.2, 10.3, 5.1],
        )
        .unwrap();

        let model = LinearRegression::<f64>::new().with_fit_intercept(false);
        let fitted = Fit::<Array2<f64>, Array2<f64>>::fit(&model, &x, &y).unwrap();

        let c = fitted.coefficients();
        assert_eq!(c.dim(), (2, 2));
        assert_relative_eq!(c[[0, 0]], 2.019_512_195_1, epsilon = 1e-7);
        assert_relative_eq!(c[[0, 1]], -0.009_756_097_6, epsilon = 1e-7);
        assert_relative_eq!(c[[1, 0]], 0.960_975_609_8, epsilon = 1e-7);
        assert_relative_eq!(c[[1, 1]], 0.119_512_195_1, epsilon = 1e-7);

        let b = fitted.intercepts();
        assert_eq!(b.len(), 2);
        assert_eq!(b[0], 0.0);
        assert_eq!(b[1], 0.0);
    }

    #[test]
    fn linreg_single_output_unchanged() {
        // Regression guard: the additive 2-D arm must not disturb the 1-D path.
        // y = 2*x + 1 (same fixture as `test_simple_linear_regression`).
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y1 = array![3.0, 5.0, 7.0, 9.0, 11.0];

        let model = LinearRegression::<f64>::new();
        let fitted = model.fit(&x, &y1).unwrap();

        assert_relative_eq!(fitted.coefficients()[0], 2.0, epsilon = 1e-10);
        assert_relative_eq!(fitted.intercept(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_f32_support() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let y = Array1::from_vec(vec![2.0f32, 4.0, 6.0, 8.0]);

        let model = LinearRegression::<f32>::new();
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 4);
    }
}
