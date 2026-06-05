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
//! | REQ-7 (multi-output 2-D Y → 2-D coef_) | NOT-STARTED | blocker #372 (fit takes `Array1` only). |
//! | REQ-8 (sample_weight in fit) | NOT-STARTED | blocker #373 (`_base.py` `fit(..., sample_weight=None)`). |
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
        let (n_samples, _n_features) = x.dim();

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

        if self.fit_intercept {
            // Centering trick: center X and y, solve the (uncentered) OLS
            // problem on the centered design, then recover the intercept as
            // y_mean - x_mean . w. sklearn centers identically before its
            // `linalg.lstsq` call (`_base.py` `_preprocess_data` + `:687`).
            let n = <F as num_traits::NumCast>::from(n_samples).ok_or_else(|| {
                FerroError::NumericalInstability {
                    message: "could not represent n_samples as the float type".into(),
                }
            })?;
            let x_mean = x
                .mean_axis(Axis(0))
                .ok_or_else(|| FerroError::InsufficientSamples {
                    required: 1,
                    actual: 0,
                    context: "cannot compute feature means of an empty design".into(),
                })?;
            let y_mean = y.sum() / n;

            let x_centered = x - &x_mean;
            let y_centered = y - y_mean;

            // SVD-based minimum-norm least squares (gelsd-equivalent), so
            // rank-deficient designs yield the min-norm split, not an
            // arbitrary basic solution (#376). `rank`/`singular` are captured
            // on the CENTERED design — exactly what sklearn does, since it
            // calls `linalg.lstsq` on the centered `X` when `fit_intercept`
            // (`_base.py` `_preprocess_data` centers, then `:687`).
            let (w, rank, singular) = linalg::solve_lstsq(&x_centered, &y_centered)?;

            let intercept = y_mean - x_mean.dot(&w);

            Ok(FittedLinearRegression {
                coefficients: w,
                intercept,
                rank_: rank,
                singular_: singular,
            })
        } else {
            // SVD-based minimum-norm least squares; accepts underdetermined
            // (n_samples < n_features) input as sklearn does (#377). With no
            // intercept, sklearn passes the RAW `X` to `linalg.lstsq`, so
            // `rank`/`singular` are captured on `x` directly (`_base.py:687`).
            let (w, rank, singular) = linalg::solve_lstsq(x, y)?;

            Ok(FittedLinearRegression {
                coefficients: w,
                intercept: <F as num_traits::Zero>::zero(),
                rank_: rank,
                singular_: singular,
            })
        }
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
    fn test_f32_support() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let y = Array1::from_vec(vec![2.0f32, 4.0, 6.0, 8.0]);

        let model = LinearRegression::<f32>::new();
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 4);
    }
}
