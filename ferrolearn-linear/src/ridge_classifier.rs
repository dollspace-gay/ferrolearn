//! Ridge Classifier.
//!
//! This module provides [`RidgeClassifier`], which applies Ridge regression
//! to classification tasks by converting class labels into a binary indicator
//! matrix and fitting a multivariate Ridge regression.
//!
//! For binary classification, the indicator matrix has a single column
//! (`{-1, +1}`). For multiclass, it has one column per class (one-hot
//! encoding). The predicted class is the one with the highest decision
//! value (`argmax(X @ coef + intercept)`).
//!
//! This approach is significantly faster than logistic regression for
//! large datasets while often achieving competitive accuracy.
//!
//! ## REQ status (per `.design/linear/ridge_classifier.md`, mirrors `sklearn/linear_model/_ridge.py` @ 1.5.2)
//!
//! Mirrors `sklearn.linear_model.RidgeClassifier` (`_ridge.py:1344`): `LabelBinarizer(pos_label=1,
//! neg_label=-1)` encoding (`_ridge.py:1300`) + per-class Ridge fit + sign/argmax predict.
//! coef_/intercept_/decision_function match the live sklearn oracle to 1e-9.
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (±1/one-hot encoding + per-class Ridge fit) | SHIPPED | `Fit for RidgeClassifier` (binary {-1,+1}, multiclass one-hot, per-column `linalg::solve_ridge`, centering). Consumer: `RsRidgeClassifier` in `ferrolearn-python`. Mirrors `_ridge.py:1300`. |
//! | REQ-2 (predict: sign/argmax → original labels) | SHIPPED | binary uses strict `> 0` (mirrors `_base.py:384` `scores > 0`); multiclass argmax; returns `classes[idx]` (original label values). Closed #405 (boundary `>=`→`>`); test `divergence_binary_decision_boundary_strict_gt`. |
//! | REQ-3 (fit_intercept incl. false) | SHIPPED | centering; matches oracle. |
//! | REQ-4 (coef_/intercept_/classes_ introspection) | SHIPPED | `HasCoefficients`/`HasClasses`; values match oracle. NOTE: `coef_matrix` is `(n_features, n_targets)`, transposed vs sklearn `coef_` `(n_classes, n_features)` — orientation contract owned by the `ferrolearn-python` binding layer. |
//! | REQ-5 (alpha≥0 validation; ≥2-class guard) | SHIPPED | negative-alpha → `InvalidParameter`; <2 classes → error. |
//! | REQ-6 (class_weight / solver variants / solver_ / positive) | NOT-STARTED | blocker #393. |
//! | REQ-7 (max_iter/tol + n_iter_) | NOT-STARTED | blocker #394. |
//! | REQ-8 (sample_weight) | SHIPPED | `RidgeClassifier::fit_with_sample_weight(x, y, sample_weight: Option<&Array1<F>>)` forwards weights into the underlying weighted ridge on the indicator matrix `Y`: weighted offsets `x_off[j]=Σwᵢx[i,j]/Σwᵢ`, `y_off[t]=Σwᵢ·Y[i,t]/Σwᵢ` (fit_intercept), centering, then `√wᵢ` row-rescale of `X`/`Y` (sklearn `_rescale_data`, `_ridge.py:682-688`), per-target `linalg::solve_ridge` with `alpha` UNSCALED, `intercept[t]=y_off[t]−Σⱼ x_off[j]·coef[j,t]`; `fit_intercept=false` skips centering (raw `√w`-rescale, intercept 0). `Fit::fit` delegates `fit_with_sample_weight(x, y, None)` (None byte-identical to the historic centering + `solve_ridge` body). Mirrors `RidgeClassifier.fit(X, y, sample_weight=None)` (`_ridge.py:1220`) forwarding through `_prepare_data` (`_ridge.py:1305`) into `_BaseRidge.fit`. Oracle tests `ridge_classifier_sample_weight_matches_sklearn` (alpha=1 binary coef `[0.25333333, 0.36]`, intercept `-1.70666667`, differs from unweighted `[0.31840796, 0.31840796]`), `ridge_classifier_none_sample_weight_equals_unweighted` (byte-identical guard). Closes #395. |
//! | REQ-9 (RidgeClassifierCV) | NOT-STARTED | blocker #396. |
//! | REQ-10 (ferray substrate) | NOT-STARTED | solve_ridge already on ferray::linalg fallback; coef storage ndarray (tied to #359). |
//!
//! acto-critic: binary + multiclass coef_/intercept_/decision_function match the live oracle to
//! 1e-9; classes_ returns original label values (no #368-style collapse); one divergence (#405,
//! binary boundary operator) found and fixed. Two states only per goal.md R-DEFER-2.
//!
//! # Examples
//!
//! ```
//! use ferrolearn_linear::ridge_classifier::RidgeClassifier;
//! use ferrolearn_core::{Fit, Predict};
//! use ndarray::{array, Array2};
//!
//! let x = Array2::from_shape_vec((6, 2), vec![
//!     1.0, 1.0, 1.0, 2.0, 2.0, 1.0,
//!     5.0, 5.0, 5.0, 6.0, 6.0, 5.0,
//! ]).unwrap();
//! let y = array![0usize, 0, 0, 1, 1, 1];
//!
//! let model = RidgeClassifier::<f64>::new();
//! let fitted = model.fit(&x, &y).unwrap();
//! let preds = fitted.predict(&x).unwrap();
//! assert_eq!(preds.len(), 6);
//! ```

use ferray::linalg::LinalgFloat;
use ferrolearn_core::error::FerroError;
use ferrolearn_core::introspection::{HasClasses, HasCoefficients};
use ferrolearn_core::traits::{Fit, Predict};
use ndarray::{Array1, Array2, Axis, ScalarOperand};
use num_traits::{Float, FromPrimitive};

use crate::linalg;

/// Ridge Classifier.
///
/// Applies Ridge regression (L2-regularized least squares) to classification
/// by converting labels to a binary indicator matrix.
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
#[derive(Debug, Clone)]
pub struct RidgeClassifier<F> {
    /// Regularization strength. Larger values specify stronger regularization.
    pub alpha: F,
    /// Whether to fit an intercept (bias) term.
    pub fit_intercept: bool,
}

impl<F: Float> RidgeClassifier<F> {
    /// Create a new `RidgeClassifier` with default settings.
    ///
    /// Defaults: `alpha = 1.0`, `fit_intercept = true`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            alpha: F::one(),
            fit_intercept: true,
        }
    }

    /// Set the regularization strength.
    #[must_use]
    pub fn with_alpha(mut self, alpha: F) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set whether to fit an intercept term.
    #[must_use]
    pub fn with_fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = fit_intercept;
        self
    }
}

impl<F: Float> Default for RidgeClassifier<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted Ridge Classifier.
///
/// Stores the learned coefficient matrix, intercept vector, and class labels.
#[derive(Debug, Clone)]
pub struct FittedRidgeClassifier<F> {
    /// Coefficient matrix, shape `(n_features, n_targets)`.
    /// For binary, `n_targets = 1`.
    coef_matrix: Array2<F>,
    /// Intercept vector, one per target.
    intercept_vec: Array1<F>,
    /// For HasCoefficients: first column of coef_matrix.
    coefficients: Array1<F>,
    /// For HasCoefficients: first element of intercept_vec.
    intercept: F,
    /// Sorted unique class labels.
    classes: Vec<usize>,
    /// Whether this is a binary problem.
    is_binary: bool,
    /// Number of features.
    n_features: usize,
}

impl<F: Float> FittedRidgeClassifier<F> {
    /// Returns the full coefficient matrix, shape `(n_features, n_targets)`.
    #[must_use]
    pub fn coef_matrix(&self) -> &Array2<F> {
        &self.coef_matrix
    }

    /// Returns the intercept vector.
    #[must_use]
    pub fn intercept_vec(&self) -> &Array1<F> {
        &self.intercept_vec
    }
}

impl<F: Float + ndarray::ScalarOperand + Send + Sync + 'static> FittedRidgeClassifier<F> {
    /// Raw `X @ coef + intercept` per class. Mirrors sklearn
    /// `RidgeClassifier.decision_function`.
    ///
    /// Returns shape `(n_samples, n_classes)`. argmax of each row agrees
    /// with [`Predict`].
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of features
    /// does not match the fitted model.
    pub fn decision_function(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let n_features = x.ncols();
        if n_features != self.n_features {
            return Err(FerroError::ShapeMismatch {
                expected: vec![self.n_features],
                actual: vec![n_features],
                context: "number of features must match fitted model".into(),
            });
        }
        Ok(x.dot(&self.coef_matrix) + &self.intercept_vec)
    }
}

impl<F: Float + Send + Sync + ScalarOperand + FromPrimitive + LinalgFloat + 'static>
    Fit<Array2<F>, Array1<usize>> for RidgeClassifier<F>
{
    type Fitted = FittedRidgeClassifier<F>;
    type Error = FerroError;

    /// Fit the Ridge Classifier by converting labels to a binary indicator
    /// matrix and solving multivariate Ridge regression.
    ///
    /// Equivalent to [`RidgeClassifier::fit_with_sample_weight`] with
    /// `sample_weight = None`.
    ///
    /// # Errors
    ///
    /// - [`FerroError::ShapeMismatch`] — sample count mismatch.
    /// - [`FerroError::InvalidParameter`] — negative alpha.
    /// - [`FerroError::InsufficientSamples`] — fewer than 2 classes.
    fn fit(
        &self,
        x: &Array2<F>,
        y: &Array1<usize>,
    ) -> Result<FittedRidgeClassifier<F>, FerroError> {
        // Unweighted fit is the `sample_weight=None` arm of the weighted fit,
        // mirroring sklearn `RidgeClassifier.fit(X, y, sample_weight=None)`
        // (`_ridge.py:1220`, default `sample_weight=None`).
        self.fit_with_sample_weight(x, y, None)
    }
}

impl<F: Float + Send + Sync + ScalarOperand + FromPrimitive + LinalgFloat + 'static>
    RidgeClassifier<F>
{
    /// Fit the Ridge Classifier with optional per-sample weights.
    ///
    /// Mirrors scikit-learn's `RidgeClassifier.fit(X, y, sample_weight=None)`
    /// (`sklearn/linear_model/_ridge.py:1220`). The target is encoded as an
    /// indicator matrix `Y` (binary `{-1, +1}` single column, multiclass
    /// one-hot) via `LabelBinarizer(pos_label=1, neg_label=-1)`
    /// (`_ridge.py:1300-1301`), then the underlying weighted ridge is solved on
    /// `(X, Y)`: `sample_weight` is forwarded into `_BaseRidge.fit` which does
    /// weighted `_preprocess_data` (weighted offsets) + `_rescale_data` (`√w`
    /// row-rescale, `_ridge.py:682-688`).
    ///
    /// When `sample_weight` is `Some(w)` and `fit_intercept` is `true`, the
    /// weighted offsets `x_off[j] = Σᵢ wᵢ·x[i,j] / Σwᵢ`,
    /// `y_off[t] = Σᵢ wᵢ·Y[i,t] / Σwᵢ` center `X`/`Y`, then each row is rescaled
    /// by `√wᵢ` before the per-target ridge solve, and
    /// `intercept[t] = y_off[t] − Σⱼ x_off[j]·coef[j,t]`. With `fit_intercept`
    /// `false` only the `√w` row-rescale is applied and the intercept is `0`.
    ///
    /// `sample_weight = None` is BYTE-IDENTICAL to [`Fit::fit`] (the unweighted
    /// centering + per-target `linalg::solve_ridge` path).
    ///
    /// # Errors
    ///
    /// - [`FerroError::ShapeMismatch`] — sample count mismatch, or
    ///   `sample_weight.len() != n_samples`.
    /// - [`FerroError::InvalidParameter`] — negative alpha.
    /// - [`FerroError::InsufficientSamples`] — fewer than 2 classes / no samples.
    pub fn fit_with_sample_weight(
        &self,
        x: &Array2<F>,
        y: &Array1<usize>,
        sample_weight: Option<&Array1<F>>,
    ) -> Result<FittedRidgeClassifier<F>, FerroError> {
        let (n_samples, n_features) = x.dim();

        if n_samples != y.len() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_samples],
                actual: vec![y.len()],
                context: "y length must match number of samples in X".into(),
            });
        }

        // `<F as num_traits::Zero>::zero()`: the `LinalgFloat` bound pulls
        // `ferray::Element` (which also defines `zero`/`one`) into scope, so
        // bare `F::zero()`/`F::one()` are ambiguous between `Element` and
        // `num_traits`. Disambiguate to the `num_traits` items used elsewhere.
        if self.alpha < <F as num_traits::Zero>::zero() {
            return Err(FerroError::InvalidParameter {
                name: "alpha".into(),
                reason: "must be non-negative".into(),
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

        let mut classes: Vec<usize> = y.to_vec();
        classes.sort_unstable();
        classes.dedup();

        if classes.len() < 2 {
            return Err(FerroError::InsufficientSamples {
                required: 2,
                actual: classes.len(),
                context: "RidgeClassifier requires at least 2 distinct classes".into(),
            });
        }

        if n_samples == 0 {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "RidgeClassifier requires at least one sample".into(),
            });
        }

        let is_binary = classes.len() == 2;

        // Build indicator matrix Y.
        let n_targets = if is_binary { 1 } else { classes.len() };
        let mut y_indicator = Array2::<F>::zeros((n_samples, n_targets));

        if is_binary {
            // Binary: encode as {-1, +1}.
            for i in 0..n_samples {
                y_indicator[[i, 0]] = if y[i] == classes[1] {
                    <F as num_traits::One>::one()
                } else {
                    -<F as num_traits::One>::one()
                };
            }
        } else {
            // Multiclass: one-hot.
            for i in 0..n_samples {
                // `classes` is the sorted-deduped image of `y`, so `y[i]` is
                // always present; fall back to a typed error rather than panic.
                let ci = classes.iter().position(|&c| c == y[i]).ok_or_else(|| {
                    FerroError::NumericalInstability {
                        message: "class label missing from class set".into(),
                    }
                })?;
                y_indicator[[i, ci]] = <F as num_traits::One>::one();
            }
        }

        // Center data if fit_intercept. With sample_weight the offsets are the
        // weighted means and the rows are √w-rescaled before the solve
        // (sklearn `_preprocess_data` weighted + `_rescale_data`,
        // `_ridge.py:682-688`); the penalty `alpha` stays UNSCALED since
        // (√w·Xc)ᵀ(√w·Xc) == Xcᵀ·W·Xc.
        let (x_work, y_work, x_off, y_off) = match sample_weight {
            None => {
                if self.fit_intercept {
                    let x_mean =
                        x.mean_axis(Axis(0))
                            .ok_or_else(|| FerroError::NumericalInstability {
                                message: "failed to compute column means".into(),
                            })?;
                    let y_mean = y_indicator.mean_axis(Axis(0)).ok_or_else(|| {
                        FerroError::NumericalInstability {
                            message: "failed to compute target means".into(),
                        }
                    })?;
                    let x_c = x - &x_mean;
                    let y_c = &y_indicator - &y_mean;
                    (x_c, y_c, Some(x_mean), Some(y_mean))
                } else {
                    (x.clone(), y_indicator.clone(), None, None)
                }
            }
            Some(w) => {
                // Per-row √w factor (sklearn `_rescale_data`, `_ridge.py:682-688`).
                let w_sqrt = w.mapv(<F as Float>::sqrt);

                if self.fit_intercept {
                    // WEIGHTED offsets: x_off[j] = Σ wᵢ x[i,j] / Σ wᵢ,
                    // y_off[t] = Σ wᵢ Y[i,t] / Σ wᵢ.
                    let w_sum = w.sum();
                    if w_sum <= <F as num_traits::Zero>::zero() {
                        return Err(FerroError::NumericalInstability {
                            message: "sum of sample_weight must be positive to center".into(),
                        });
                    }

                    let mut x_mean = Array1::<F>::zeros(n_features);
                    for (i, row) in x.outer_iter().enumerate() {
                        let wi = w[i];
                        x_mean = &x_mean + &row.mapv(|v| v * wi);
                    }
                    x_mean.mapv_inplace(|v| v / w_sum);

                    let mut y_mean = Array1::<F>::zeros(n_targets);
                    for (i, row) in y_indicator.outer_iter().enumerate() {
                        let wi = w[i];
                        y_mean = &y_mean + &row.mapv(|v| v * wi);
                    }
                    y_mean.mapv_inplace(|v| v / w_sum);

                    let x_centered = x - &x_mean;
                    let y_centered = &y_indicator - &y_mean;
                    let x_scaled = &x_centered * &w_sqrt.view().insert_axis(Axis(1));
                    let y_scaled = &y_centered * &w_sqrt.view().insert_axis(Axis(1));
                    (x_scaled, y_scaled, Some(x_mean), Some(y_mean))
                } else {
                    // No centering; just √w row-rescaling, intercept 0.
                    let x_scaled = x * &w_sqrt.view().insert_axis(Axis(1));
                    let y_scaled = &y_indicator * &w_sqrt.view().insert_axis(Axis(1));
                    (x_scaled, y_scaled, None, None)
                }
            }
        };

        // Solve Ridge for each target column.
        let mut coef_matrix = Array2::<F>::zeros((n_features, n_targets));
        for t in 0..n_targets {
            let y_col = y_work.column(t).to_owned();
            let w = linalg::solve_ridge(&x_work, &y_col, self.alpha)?;
            for j in 0..n_features {
                coef_matrix[[j, t]] = w[j];
            }
        }

        // Compute intercepts.
        let intercept_vec = if let (Some(xm), Some(ym)) = (&x_off, &y_off) {
            let xm_dot = xm.dot(&coef_matrix);
            ym - &xm_dot
        } else {
            Array1::<F>::zeros(n_targets)
        };

        let coefficients = coef_matrix.column(0).to_owned();
        let intercept = intercept_vec[0];

        Ok(FittedRidgeClassifier {
            coef_matrix,
            intercept_vec,
            coefficients,
            intercept,
            classes,
            is_binary,
            n_features,
        })
    }
}

impl<F: Float + Send + Sync + ScalarOperand + 'static> Predict<Array2<F>>
    for FittedRidgeClassifier<F>
{
    type Output = Array1<usize>;
    type Error = FerroError;

    /// Predict class labels for the given feature matrix.
    ///
    /// Computes `X @ coef_matrix + intercept_vec` and takes `argmax` per row.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of features
    /// does not match the fitted model.
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

        // Compute decision values: X @ coef_matrix + intercept_vec.
        let scores = x.dot(&self.coef_matrix) + &self.intercept_vec;

        if self.is_binary {
            for i in 0..n_samples {
                // sklearn `LinearClassifierMixin.predict` uses STRICT `scores > 0`
                // (`sklearn/linear_model/_base.py:384`:
                // `indices = xp.astype(scores > 0, ...)`), so a decision of
                // exactly 0 maps to index 0 -> `classes_[0]`.
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
    for FittedRidgeClassifier<F>
{
    fn coefficients(&self) -> &Array1<F> {
        &self.coefficients
    }

    fn intercept(&self) -> F {
        self.intercept
    }
}

impl<F: Float + Send + Sync + ScalarOperand + 'static> HasClasses for FittedRidgeClassifier<F> {
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

    #[test]
    fn ridge_classifier_sample_weight_matches_sklearn() {
        // Live sklearn 1.5.2 oracle (R-CHAR-3):
        //   cd /tmp && python3 -c "import numpy as np; \
        //     from sklearn.linear_model import RidgeClassifier; \
        //     X=np.array([[1.,2.],[2.,1.],[3.,4.],[4.,3.],[5.,5.],[1.,1.],[4.,4.]]); \
        //     y=np.array([0,0,1,1,1,0,1]); w=np.array([1.,3.,1.,1.,2.,1.,1.]); \
        //     m=RidgeClassifier(alpha=1.0).fit(X,y,sample_weight=w); \
        //     print([round(c,8) for c in m.coef_[0]], round(m.intercept_[0],8))"
        //   -> weighted   coef_ [0.25333333, 0.36],       intercept_ -1.70666667
        //   -> unweighted coef_ [0.31840796, 0.31840796], intercept_ -1.67661692
        let x = Array2::from_shape_vec(
            (7, 2),
            vec![
                1.0, 2.0, 2.0, 1.0, 3.0, 4.0, 4.0, 3.0, 5.0, 5.0, 1.0, 1.0, 4.0, 4.0,
            ],
        )
        .unwrap();
        let y = array![0usize, 0, 1, 1, 1, 0, 1];
        let w = array![1.0, 3.0, 1.0, 1.0, 2.0, 1.0, 1.0];

        let model = RidgeClassifier::<f64>::new().with_alpha(1.0);
        let fitted = model.fit_with_sample_weight(&x, &y, Some(&w)).unwrap();

        // Binary => single target column 0; coef_matrix is (n_features, n_targets).
        let coef = fitted.coef_matrix();
        assert!(
            (coef[[0, 0]] - 0.253_333_33).abs() < 1e-6,
            "coef[0]={} expected 0.25333333",
            coef[[0, 0]]
        );
        assert!(
            (coef[[1, 0]] - 0.36).abs() < 1e-6,
            "coef[1]={} expected 0.36",
            coef[[1, 0]]
        );
        assert!(
            (fitted.intercept_vec()[0] - (-1.706_666_67)).abs() < 1e-6,
            "intercept={} expected -1.70666667",
            fitted.intercept_vec()[0]
        );

        // Non-tautological: must differ from the unweighted fit.
        let unweighted = model.fit(&x, &y).unwrap();
        let uw = unweighted.coef_matrix();
        assert!(
            (uw[[0, 0]] - 0.318_407_96).abs() < 1e-6 && (uw[[1, 0]] - 0.318_407_96).abs() < 1e-6,
            "unweighted oracle mismatch: [{}, {}]",
            uw[[0, 0]],
            uw[[1, 0]]
        );
        assert!(
            (coef[[0, 0]] - uw[[0, 0]]).abs() > 1e-3,
            "weighted fit must differ from unweighted fit"
        );
    }

    #[test]
    fn ridge_classifier_none_sample_weight_equals_unweighted() {
        // `fit_with_sample_weight(.., None)` must be byte-identical to `fit`.
        let x = Array2::from_shape_vec(
            (7, 2),
            vec![
                1.0, 2.0, 2.0, 1.0, 3.0, 4.0, 4.0, 3.0, 5.0, 5.0, 1.0, 1.0, 4.0, 4.0,
            ],
        )
        .unwrap();
        let y = array![0usize, 0, 1, 1, 1, 0, 1];

        let model = RidgeClassifier::<f64>::new().with_alpha(1.0);
        let via_fit = model.fit(&x, &y).unwrap();
        let via_none = model.fit_with_sample_weight(&x, &y, None).unwrap();

        assert_eq!(via_fit.coef_matrix(), via_none.coef_matrix());
        assert_eq!(via_fit.intercept_vec(), via_none.intercept_vec());

        // Same for fit_intercept=false.
        let model_ni = RidgeClassifier::<f64>::new()
            .with_alpha(1.0)
            .with_fit_intercept(false);
        let via_fit_ni = model_ni.fit(&x, &y).unwrap();
        let via_none_ni = model_ni.fit_with_sample_weight(&x, &y, None).unwrap();
        assert_eq!(via_fit_ni.coef_matrix(), via_none_ni.coef_matrix());
        assert_eq!(via_fit_ni.intercept_vec(), via_none_ni.intercept_vec());
    }

    #[test]
    fn test_default_constructor() {
        let m = RidgeClassifier::<f64>::new();
        assert!(m.alpha == 1.0);
        assert!(m.fit_intercept);
    }

    #[test]
    fn test_builder() {
        let m = RidgeClassifier::<f64>::new()
            .with_alpha(0.5)
            .with_fit_intercept(false);
        assert!(m.alpha == 0.5);
        assert!(!m.fit_intercept);
    }

    #[test]
    fn test_binary_classification() {
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 2.0, 2.0, 8.0, 8.0, 8.0, 9.0, 9.0, 8.0, 9.0, 9.0,
            ],
        )
        .unwrap();
        let y = array![0, 0, 0, 0, 1, 1, 1, 1];

        let model = RidgeClassifier::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        let correct: usize = preds.iter().zip(y.iter()).filter(|(p, a)| p == a).count();
        assert!(correct >= 6, "expected at least 6 correct, got {correct}");
    }

    #[test]
    fn test_multiclass_classification() {
        let x = Array2::from_shape_vec(
            (9, 2),
            vec![
                0.0, 0.0, 0.5, 0.0, 0.0, 0.5, 10.0, 0.0, 10.5, 0.0, 10.0, 0.5, 0.0, 10.0, 0.5,
                10.0, 0.0, 10.5,
            ],
        )
        .unwrap();
        let y = array![0, 0, 0, 1, 1, 1, 2, 2, 2];

        let model = RidgeClassifier::<f64>::new().with_alpha(0.1);
        let fitted = model.fit(&x, &y).unwrap();

        assert_eq!(fitted.n_classes(), 3);
        assert_eq!(fitted.classes(), &[0, 1, 2]);

        let preds = fitted.predict(&x).unwrap();
        let correct: usize = preds.iter().zip(y.iter()).filter(|(p, a)| p == a).count();
        assert!(correct >= 7, "expected at least 7 correct, got {correct}");
    }

    #[test]
    fn test_shape_mismatch() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![0, 1]; // Wrong length

        let model = RidgeClassifier::<f64>::new();
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_negative_alpha() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![0, 0, 1, 1];

        let model = RidgeClassifier::<f64>::new().with_alpha(-1.0);
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_single_class_error() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![0, 0, 0];

        let model = RidgeClassifier::<f64>::new();
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_has_coefficients() {
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 8.0, 8.0, 8.0, 9.0, 9.0, 8.0],
        )
        .unwrap();
        let y = array![0, 0, 0, 1, 1, 1];

        let fitted = RidgeClassifier::<f64>::new().fit(&x, &y).unwrap();
        assert_eq!(fitted.coefficients().len(), 2);
    }

    #[test]
    fn test_has_classes() {
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 8.0, 8.0, 8.0, 9.0, 9.0, 8.0],
        )
        .unwrap();
        let y = array![0, 0, 0, 1, 1, 1];

        let fitted = RidgeClassifier::<f64>::new().fit(&x, &y).unwrap();
        assert_eq!(fitted.classes(), &[0, 1]);
        assert_eq!(fitted.n_classes(), 2);
    }

    #[test]
    fn test_predict_feature_mismatch() {
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 8.0, 8.0, 8.0, 9.0, 9.0, 8.0],
        )
        .unwrap();
        let y = array![0, 0, 0, 1, 1, 1];

        let fitted = RidgeClassifier::<f64>::new().fit(&x, &y).unwrap();

        let x_bad = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        assert!(fitted.predict(&x_bad).is_err());
    }

    #[test]
    fn test_alpha_zero() {
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 8.0, 8.0, 8.0, 9.0, 9.0, 8.0],
        )
        .unwrap();
        let y = array![0, 0, 0, 1, 1, 1];

        let model = RidgeClassifier::<f64>::new().with_alpha(0.0);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 6);
    }
}
