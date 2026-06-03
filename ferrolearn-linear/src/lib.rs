//! # ferrolearn-linear
//!
//! Linear models for the ferrolearn machine learning framework.
//!
//! This crate provides implementations of the most common linear models
//! for both regression and classification tasks:
//!
//! - **[`LinearRegression`]** — Ordinary Least Squares via QR decomposition
//! - **[`Ridge`]** — L2-regularized regression via Cholesky decomposition
//! - **[`RidgeCV`]** — Ridge with built-in cross-validated alpha selection
//! - **[`Lasso`]** — L1-regularized regression via coordinate descent
//! - **[`LassoCV`]** — Lasso with built-in cross-validated alpha selection
//! - **[`ElasticNet`]** — Combined L1/L2 regularization via coordinate descent
//! - **[`ElasticNetCV`]** — ElasticNet with cross-validated (alpha, l1_ratio) selection
//! - **[`BayesianRidge`]** — Bayesian Ridge with automatic regularization tuning
//! - **[`HuberRegressor`]** — Robust regression via IRLS with Huber loss
//! - **[`LogisticRegression`]** — Binary and multiclass classification via L-BFGS
//!
//! All models implement the [`ferrolearn_core::Fit`] and [`ferrolearn_core::Predict`]
//! traits, and produce fitted types that implement [`ferrolearn_core::introspection::HasCoefficients`].
//!
//! # Design
//!
//! Each model follows the compile-time safety pattern:
//!
//! - The unfitted struct (e.g., `LinearRegression<F>`) holds hyperparameters
//!   and implements [`Fit`](ferrolearn_core::Fit).
//! - Calling `fit()` produces a new fitted type (e.g., `FittedLinearRegression<F>`)
//!   that implements [`ferrolearn_core::Predict`].
//! - Calling `predict()` on an unfitted model is a compile-time error.
//!
//! # Pipeline Integration
//!
//! All models implement [`PipelineEstimator`](ferrolearn_core::pipeline::PipelineEstimator)
//! for `f64`, allowing them to be used as the final step in a
//! [`Pipeline`](ferrolearn_core::pipeline::Pipeline).
//!
//! # Float Generics
//!
//! All models are generic over `F: num_traits::Float + Send + Sync + 'static`,
//! supporting both `f32` and `f64`.
//!
//! # `## REQ status`
//!
//! Binary (R-DEFER-2) for the crate-root RE-EXPORT BOUNDARY (this file is the public-API
//! surface, not an estimator). Mirrors `sklearn/linear_model/__init__.py` `__all__`
//! (`:48-98`) + the `score` mixins `sklearn/base.py` `ClassifierMixin.score` (`:738-764`,
//! accuracy) / `RegressorMixin.score` (`:805-849`, R²). Design doc: `.design/linear/lib.md`.
//! Per-estimator REQs live in the sibling modules' own routed docs. Score traits
//! (`ClassifierScore`/`RegressorScore`) are pre-existing crate-root `pub trait`s re-exported
//! via the meta-crate (`ferrolearn::linear`) — grandfathered public API (goal.md S5); honest
//! underclaim (R-HONEST-3): no production `.score()` caller yet, and `sample_weight` is
//! unsupported (REQ-6).
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (re-export boundary) | SHIPPED | the `pub use` block re-exports every implemented linear/svm/discriminant_analysis/isotonic estimator at the crate root, mirroring sklearn `linear_model.__all__` (`__init__.py:48-98`), broadened per goal.md scope §2. Consumers: meta-crate `pub use ferrolearn_linear as linear` + PyO3 shim `ferrolearn-python/src/{regressors,classifiers,extras}.rs`. |
//! | REQ-2 (`ClassifierScore::score` == mean accuracy) | SHIPPED | `ClassifierScore` blanket impl body `mean_accuracy` (`correct / n`) mirrors `ClassifierMixin.score` → `accuracy_score` (`base.py:738-764`); critic-verified vs live oracle (`accuracy_score([0,1,2,1],[0,1,1,1])=0.75`). Consumer: grandfathered crate/meta re-export (S5). Underclaim: no production `.score()` caller; single-label (`Output=Array1<usize>`). |
//! | REQ-3 (`RegressorScore::score` == in-regime R²) | SHIPPED | `RegressorScore` blanket impl body `r2_score` = `1 − ss_res/ss_tot` mirrors `RegressorMixin.score` → `metrics.r2_score` (`base.py:805-849`); matches live oracle `r2_score([3.,5.,2.,7.],[2.5,5.,2.,8.])=0.9152542372881356` (`r2_in_regime_matches_oracle`). Consumer: grandfathered re-export (S5). |
//! | REQ-4 (constant-y R² edge parity) | SHIPPED | FIXED #1104. `r2_score` now returns `0.0` (was `neg_infinity()`) when `ss_tot==0 ∧ ss_res!=0`, matching `metrics.r2_score` (`_regression.py:891`); zero-residual stays `1.0`. Green: `divergence_r2_constant_ytrue_nonzero_residual_returns_zero` + `r2_constant_ytrue_zero_residual_returns_one`. |
//! | REQ-5 (`log_proba` behind predict_log_proba) | SHIPPED | FIXED #1105. `log_proba` is now the unclamped `p.ln()`, matching sklearn `predict_log_proba = np.log(predict_proba)` (`discriminant_analysis.py:1059`); `0.0`→`-inf`. Consumers: `logistic_regression.rs`/`logistic_regression_cv.rs`/`qda.rs` `predict_log_proba`. Green: `divergence_log_proba_zero_clamps_instead_of_neg_inf`. |
//! | REQ-6 (sample_weight on score) | NOT-STARTED | open prereq blocker #1106. The score traits take only `(&self, x, y)`; sklearn `score(self, X, y, sample_weight=None)` (`base.py:738`,`:805`) forwards `sample_weight` into `accuracy_score`/`r2_score`. |
//! | REQ-substrate (ferray) | NOT-STARTED | open prereq blocker #1107. Helpers + score traits run on `ndarray::{Array1,Array2}` + `num_traits::Float`, not `ferray-core` arrays (R-SUBSTRATE-1). |

pub mod ard;
pub mod bayesian_ridge;
pub mod elastic_net;
pub mod elastic_net_cv;
pub mod glm;
pub mod huber_regressor;
pub mod isotonic;
pub mod lars;
pub mod lasso;
pub mod lasso_cv;
pub mod lda;
mod linalg;
pub mod linear_regression;
pub mod linear_svc;
pub mod linear_svr;
pub mod logistic_regression;
pub mod logistic_regression_cv;
pub mod nu_svm;
pub mod omp;
pub mod one_class_svm;
mod optim;
pub mod qda;
pub mod quantile_regressor;
pub mod ransac;
pub mod ridge;
pub mod ridge_classifier;
pub mod ridge_cv;
pub mod sgd;
pub mod svm;

// Re-export the main types at the crate root.
pub use ard::{ARDRegression, FittedARDRegression};
pub use bayesian_ridge::{BayesianRidge, FittedBayesianRidge};
pub use elastic_net::{ElasticNet, FittedElasticNet};
pub use elastic_net_cv::{ElasticNetCV, FittedElasticNetCV};
pub use glm::{
    FittedGLMRegressor, GLMFamily, GLMRegressor, GammaRegressor, PoissonRegressor, TweedieRegressor,
};
pub use huber_regressor::{FittedHuberRegressor, HuberRegressor};
pub use isotonic::{FittedIsotonicRegression, IsotonicRegression};
pub use lars::{FittedLars, FittedLassoLars, Lars, LassoLars};
pub use lasso::{FittedLasso, Lasso};
pub use lasso_cv::{FittedLassoCV, LassoCV};
pub use lda::{FittedLDA, LDA};
pub use linear_regression::{FittedLinearRegression, LinearRegression};
pub use linear_svc::{FittedLinearSVC, LinearSVC, LinearSVCLoss};
pub use linear_svr::{FittedLinearSVR, LinearSVR, LinearSVRLoss};
pub use logistic_regression::{FittedLogisticRegression, LogisticRegression};
pub use logistic_regression_cv::{FittedLogisticRegressionCV, LogisticRegressionCV};
pub use nu_svm::{FittedNuSVC, FittedNuSVR, NuSVC, NuSVR};
pub use omp::{FittedOMP, OrthogonalMatchingPursuit};
pub use one_class_svm::{FittedOneClassSVM, OneClassSVM};
pub use qda::{FittedQDA, QDA};
pub use quantile_regressor::{FittedQuantileRegressor, QuantileRegressor};
pub use ransac::{FittedRANSACRegressor, RANSACRegressor};
pub use ridge::{FittedRidge, FittedRidgeMulti, Ridge};
pub use ridge_classifier::{FittedRidgeClassifier, RidgeClassifier};
pub use ridge_cv::{FittedRidgeCV, RidgeCV};
pub use sgd::{
    FittedSGDClassifier, FittedSGDOneClassSVM, FittedSGDRegressor, SGDClassifier, SGDOneClassSVM,
    SGDRegressor,
};
pub use svm::{
    FittedSVC, FittedSVR, Kernel, LinearKernel, PolynomialKernel, RbfKernel, SVC, SVR,
    SigmoidKernel,
};

use ferrolearn_core::error::FerroError;
use ferrolearn_core::traits::Predict;
use ndarray::{Array1, Array2};
use num_traits::Float;

/// Mean-accuracy `score(x, y)` exposed on every fitted classifier in this
/// crate via a blanket impl over [`Predict<Array2<F>, Output=Array1<usize>>`].
///
/// Users just `use ferrolearn_linear::ClassifierScore;` to call
/// `fitted.score(&x, &y)` and get the same result as sklearn's
/// `ClassifierMixin.score`.
pub trait ClassifierScore<F: Float> {
    /// Mean accuracy on the given test data and labels.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if `x.nrows() != y.len()`,
    /// or any error forwarded from the inner `predict`.
    fn score(&self, x: &Array2<F>, y: &Array1<usize>) -> Result<F, FerroError>;
}

impl<T, F> ClassifierScore<F> for T
where
    T: Predict<Array2<F>, Output = Array1<usize>, Error = FerroError>,
    F: Float,
{
    fn score(&self, x: &Array2<F>, y: &Array1<usize>) -> Result<F, FerroError> {
        if x.nrows() != y.len() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![x.nrows()],
                actual: vec![y.len()],
                context: "y length must match number of samples in X".into(),
            });
        }
        let preds = self.predict(x)?;
        Ok(mean_accuracy(&preds, y))
    }
}

/// R² `score(x, y)` exposed on every fitted regressor in this crate via
/// a blanket impl over [`Predict<Array2<F>, Output=Array1<F>>`].
///
/// Users just `use ferrolearn_linear::RegressorScore;` to call
/// `fitted.score(&x, &y)`.
pub trait RegressorScore<F: Float> {
    /// R² coefficient of determination on the given test data and targets.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if `x.nrows() != y.len()`,
    /// or any error forwarded from the inner `predict`.
    fn score(&self, x: &Array2<F>, y: &Array1<F>) -> Result<F, FerroError>;
}

impl<T, F> RegressorScore<F> for T
where
    T: Predict<Array2<F>, Output = Array1<F>, Error = FerroError>,
    F: Float,
{
    fn score(&self, x: &Array2<F>, y: &Array1<F>) -> Result<F, FerroError> {
        if x.nrows() != y.len() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![x.nrows()],
                actual: vec![y.len()],
                context: "y length must match number of samples in X".into(),
            });
        }
        let preds = self.predict(x)?;
        Ok(r2_score(&preds, y))
    }
}

/// Mean accuracy: `(sum(predictions == targets)) / n`.
///
/// Used as the body of every classifier `score(&self, x, y)` method in
/// this crate to mirror sklearn's `ClassifierMixin.score`.
pub(crate) fn mean_accuracy<F: Float>(predictions: &Array1<usize>, targets: &Array1<usize>) -> F {
    let n = targets.len();
    if n == 0 {
        return F::zero();
    }
    let correct = predictions
        .iter()
        .zip(targets.iter())
        .filter(|(p, t)| p == t)
        .count();
    F::from(correct).unwrap() / F::from(n).unwrap()
}

/// R² coefficient of determination: `1 - SSres / SStot`. Used as the
/// body of every regressor `score(&self, x, y)` method to mirror
/// sklearn's `RegressorMixin.score`. Constant-y returns `1.0` if
/// predictions are also constant-perfect (zero residual), else `0.0`
/// when the residual is non-zero — matching `sklearn.metrics.r2_score`
/// (`_regression.py:891`: `output_scores[nonzero_numerator &
/// ~nonzero_denominator] = 0.0`).
pub(crate) fn r2_score<F: Float>(y_pred: &Array1<F>, y_true: &Array1<F>) -> F {
    let n = y_true.len();
    if n == 0 {
        return F::zero();
    }
    let mean = y_true.iter().copied().fold(F::zero(), |a, b| a + b) / F::from(n).unwrap();
    let mut ss_res = F::zero();
    let mut ss_tot = F::zero();
    for i in 0..n {
        let r = y_true[i] - y_pred[i];
        let t = y_true[i] - mean;
        ss_res = ss_res + r * r;
        ss_tot = ss_tot + t * t;
    }
    if ss_tot == F::zero() {
        if ss_res == F::zero() {
            F::one()
        } else {
            F::zero()
        }
    } else {
        F::one() - ss_res / ss_tot
    }
}

/// Element-wise natural log of a probability matrix, used as the body of every
/// classifier `predict_log_proba` method in this crate. Unclamped, mirroring
/// scikit-learn `predict_log_proba = np.log(predict_proba)`
/// (`sklearn/discriminant_analysis.py:1059`: `return np.log(probas_)`): a `0.0`
/// probability maps to `-inf`. Inputs are always in `[0, 1]`, so the result is
/// either finite or `-inf` (never `NaN`).
pub(crate) fn log_proba<F: Float>(proba: &Array2<F>) -> Array2<F> {
    proba.mapv(|p| p.ln())
}
