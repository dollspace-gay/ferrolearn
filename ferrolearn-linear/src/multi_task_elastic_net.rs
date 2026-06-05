//! Multi-task ElasticNet regression (joint multi-output L1/L2,1 mixed-norm
//! block coordinate descent).
//!
//! This module provides [`MultiTaskElasticNet`], the multi-output linear model
//! that fits all target columns jointly under a blended L2,1 (group-Lasso) +
//! squared-Frobenius (L2) penalty, minimizing
//!
//! ```text
//! (1 / (2 * n_samples)) * ||Y - X W||_F^2
//!     + alpha * l1_ratio * ||W||_21
//!     + 0.5 * alpha * (1 - l1_ratio) * ||W||_F^2
//! ```
//!
//! where `||W||_21 = sum_j sqrt(sum_k W[j,k]^2)` is the sum over features of the
//! L2 norm of each feature's coefficient ROW across tasks, and `||W||_F^2` is
//! the squared Frobenius norm. The mixed L2,1 term couples a feature's
//! coefficients across all tasks (joint feature selection across outputs), while
//! the L2 term shrinks all coefficients smoothly.
//!
//! Mirrors `sklearn.linear_model.MultiTaskElasticNet`
//! (`sklearn/linear_model/_coordinate_descent.py:402`, objective documented at
//! `:426-430`); the production solver is the Cython
//! `enet_coordinate_descent_multi_task` in `_cd_fast.pyx:740` (objective at
//! `:756`, `0.5 * norm(Y - X W.T)^2 + l1_reg ||W.T||_21 + 0.5 * l2_reg
//! norm(W.T)^2`), with `l1_reg = alpha * l1_ratio * n_samples`, `l2_reg =
//! alpha * (1 - l1_ratio) * n_samples` (`_coordinate_descent.py:655-656`).
//! `MultiTaskLasso` is `MultiTaskElasticNet(l1_ratio=1.0)`, i.e. `l2_reg = 0`.
//! ferrolearn implements the dense block-coordinate-descent core directly.
//!
//! ## REQ status (per `.design/linear/elastic_net.md`, mirrors `sklearn/linear_model/_coordinate_descent.py` @ 1.5.2)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-14 (MultiTaskElasticNet, multi-output L1/L21 mixed-norm block CD) | SHIPPED | `MultiTaskElasticNet<F>` / `FittedMultiTaskElasticNet<F>` in this module: `impl Fit<Array2<F>, Array2<F>>` runs block coordinate descent porting `_cd_fast.pyx::enet_coordinate_descent_multi_task` (`:740-960`, full `l2_reg != 0` version): `l1_reg = alpha*l1_ratio*n`, `l2_reg = alpha*(1-l1_ratio)*n`, per-feature block update `W[j,:] = tmp * max(1 - l1_reg/||tmp||, 0) / (norm_cols_X[j] + l2_reg)` (`:872-874`), residual rank-1 maintenance, and the two-level relative-change + dual-gap stop (`:903-950`, `XtA = Xc^T R - l2_reg*W`, `gap += l1_reg*||W||_21 - const*R_dot_Y + 0.5*l2_reg*(1+const^2)*||W||_F^2`, `tol_scaled = tol*||Y||_F^2`). `coef_` is stored `(n_tasks, n_features)` matching sklearn `coef_`. Verified against the live sklearn 1.5.2 oracle (R-CHAR-3): on `X=[[1,2],[2,1],[3,4],[4,3],[5,5]]`, `Y=[[3,1],[2.5,2],[7.1,3.5],[6,4.2],[11.2,6]]`, `MultiTaskElasticNet(alpha=0.3, l1_ratio=0.5)` -> `coef_=[[0.8163842473,1.3248847997],[0.8318338149,0.3403462995]]`, `intercept_=[-0.4638071409,-0.1765403432]`, `n_iter_=16`. Tests `multi_task_elastic_net_matches_sklearn`, `multi_task_elastic_net_no_intercept_matches_sklearn`, `multi_task_elastic_net_predict_matches_sklearn`, `multi_task_elastic_net_l1_ratio_one_equals_multitask_lasso`, `multi_task_elastic_net_validation_errors`. Non-test consumer: `MultiTaskElasticNet` is the public estimator boundary API re-exported at the crate root (`ferrolearn_linear::MultiTaskElasticNet`, grandfathered boundary per goal.md S5). |
//!
//! # Examples
//!
//! ```
//! use ferrolearn_linear::MultiTaskElasticNet;
//! use ferrolearn_core::{Fit, Predict};
//! use ndarray::{array, Array2};
//!
//! let model = MultiTaskElasticNet::<f64>::new().with_alpha(0.3).with_l1_ratio(0.5);
//! let x: Array2<f64> = array![[1.0, 2.0], [2.0, 1.0], [3.0, 4.0], [4.0, 3.0], [5.0, 5.0]];
//! let y: Array2<f64> = array![[3.0, 1.0], [2.5, 2.0], [7.1, 3.5], [6.0, 4.2], [11.2, 6.0]];
//!
//! let fitted = model.fit(&x, &y).unwrap();
//! let preds = fitted.predict(&x).unwrap();
//! ```

use ferrolearn_core::error::FerroError;
use ferrolearn_core::traits::{Fit, Predict};
use ndarray::{Array1, Array2, Axis, ScalarOperand};
use num_traits::{Float, FromPrimitive};

/// Multi-task ElasticNet regression (joint multi-output L1/L2,1-regularized
/// least squares).
///
/// Fits all target columns jointly under a blended group-Lasso (L2,1) +
/// squared-Frobenius (L2) penalty. Mirrors
/// `sklearn.linear_model.MultiTaskElasticNet`
/// (`_coordinate_descent.py:402`). `MultiTaskLasso` is
/// `MultiTaskElasticNet(l1_ratio=1.0)`.
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
#[derive(Debug, Clone)]
pub struct MultiTaskElasticNet<F> {
    /// Regularization strength scaling both penalty terms. Larger values
    /// specify stronger regularization.
    pub alpha: F,
    /// ElasticNet mixing parameter in `[0, 1]`. `l1_ratio = 1` is the pure
    /// L2,1 (group-Lasso) `MultiTaskLasso`; `l1_ratio = 0` is a pure L2
    /// (squared-Frobenius) penalty; intermediate values blend both.
    pub l1_ratio: F,
    /// Whether to fit a per-task intercept (bias) term.
    pub fit_intercept: bool,
    /// Maximum number of block-coordinate-descent iterations.
    pub max_iter: usize,
    /// Convergence tolerance on the relative coefficient change / dual gap.
    pub tol: F,
}

impl<F: Float> MultiTaskElasticNet<F> {
    /// Create a new `MultiTaskElasticNet` with default settings.
    ///
    /// Defaults: `alpha = 1.0`, `l1_ratio = 0.5`, `fit_intercept = true`,
    /// `max_iter = 1000`, `tol = 1e-4` — mirroring sklearn's ctor defaults
    /// `MultiTaskElasticNet(alpha=1.0, l1_ratio=0.5, fit_intercept=True,
    /// max_iter=1000, tol=1e-4)` (`_coordinate_descent.py:402`).
    #[must_use]
    pub fn new() -> Self {
        Self {
            alpha: F::one(),
            l1_ratio: F::from(0.5).unwrap_or_else(F::epsilon),
            fit_intercept: true,
            max_iter: 1000,
            tol: F::from(1e-4).unwrap_or_else(F::epsilon),
        }
    }

    /// Set the regularization strength.
    #[must_use]
    pub fn with_alpha(mut self, alpha: F) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set the ElasticNet mixing parameter (`l1_ratio` in `[0, 1]`).
    #[must_use]
    pub fn with_l1_ratio(mut self, l1_ratio: F) -> Self {
        self.l1_ratio = l1_ratio;
        self
    }

    /// Set whether to fit per-task intercept terms.
    #[must_use]
    pub fn with_fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = fit_intercept;
        self
    }

    /// Set the maximum number of iterations.
    #[must_use]
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the convergence tolerance.
    #[must_use]
    pub fn with_tol(mut self, tol: F) -> Self {
        self.tol = tol;
        self
    }
}

impl<F: Float> Default for MultiTaskElasticNet<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted multi-task ElasticNet regression model.
///
/// Stores the learned coefficient matrix (shape `(n_tasks, n_features)`, the
/// sklearn `coef_` layout), the per-task intercept vector, and the number of
/// block-coordinate-descent sweeps run. Implements [`Predict`].
#[derive(Debug, Clone)]
pub struct FittedMultiTaskElasticNet<F> {
    /// Learned coefficients, shape `(n_tasks, n_features)` — matches sklearn's
    /// `MultiTaskElasticNet.coef_` layout exactly. Whole feature columns
    /// (across the task rows) are jointly zero or jointly non-zero.
    coefficients: Array2<F>,
    /// Per-task intercept vector, length `n_tasks`. Filled with zeros when
    /// `fit_intercept = false`.
    intercepts: Array1<F>,
    /// Number of block-coordinate-descent sweeps run by the solver (1-based;
    /// mirrors sklearn `MultiTaskElasticNet.n_iter_`).
    n_iter: usize,
}

impl<F: Float> FittedMultiTaskElasticNet<F> {
    /// Borrow the learned coefficient matrix, shape `(n_tasks, n_features)`
    /// (sklearn `coef_` layout).
    #[must_use]
    pub fn coefficients(&self) -> &Array2<F> {
        &self.coefficients
    }

    /// Borrow the per-task intercept vector, length `n_tasks` (sklearn
    /// `intercept_`).
    #[must_use]
    pub fn intercepts(&self) -> &Array1<F> {
        &self.intercepts
    }

    /// Number of block-coordinate-descent sweeps run by the solver (sklearn
    /// `n_iter_`).
    #[must_use]
    pub fn n_iter(&self) -> usize {
        self.n_iter
    }
}

impl<F: Float + Send + Sync + ScalarOperand + FromPrimitive + 'static> Fit<Array2<F>, Array2<F>>
    for MultiTaskElasticNet<F>
{
    type Fitted = FittedMultiTaskElasticNet<F>;
    type Error = FerroError;

    /// Fit the multi-task ElasticNet model using block coordinate descent.
    ///
    /// Ports sklearn's `enet_coordinate_descent_multi_task`
    /// (`_cd_fast.pyx:740`) with `l1_reg = alpha * l1_ratio * n_samples` and
    /// `l2_reg = alpha * (1 - l1_ratio) * n_samples`
    /// (`_coordinate_descent.py:655-656`).
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if `Y.nrows()` differs from the
    /// number of samples in `X`.
    /// Returns [`FerroError::InvalidParameter`] if `alpha` is negative or
    /// `l1_ratio` is outside `[0, 1]`.
    /// Returns [`FerroError::InsufficientSamples`] if there are no samples.
    /// Returns [`FerroError::NumericalInstability`] if a required float constant
    /// or column mean cannot be formed.
    fn fit(
        &self,
        x: &Array2<F>,
        y: &Array2<F>,
    ) -> Result<FittedMultiTaskElasticNet<F>, FerroError> {
        let (n_samples, n_features) = x.dim();

        if n_samples != y.nrows() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_samples],
                actual: vec![y.nrows()],
                context: "Y rows must match number of samples in X".into(),
            });
        }

        if self.alpha < F::zero() {
            return Err(FerroError::InvalidParameter {
                name: "alpha".into(),
                reason: "must be non-negative".into(),
            });
        }

        if self.l1_ratio < F::zero() || self.l1_ratio > F::one() {
            return Err(FerroError::InvalidParameter {
                name: "l1_ratio".into(),
                reason: "must be in [0, 1]".into(),
            });
        }

        if n_samples == 0 {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "MultiTaskElasticNet requires at least one sample".into(),
            });
        }

        let n = n_samples;
        let p = n_features;
        let t = y.ncols();

        let n_f = F::from(n).ok_or_else(|| FerroError::NumericalInstability {
            message: "failed to convert n_samples to float".into(),
        })?;
        let half = F::from(0.5).ok_or_else(|| FerroError::NumericalInstability {
            message: "failed to convert 0.5 to float".into(),
        })?;

        // Center the data when fitting per-task intercepts (sklearn
        // `_preprocess_data`): `x_mean` is the column mean of X (len p),
        // `y_mean` the column mean of Y (len t). When `!fit_intercept`, work on
        // the raw design with zero means.
        let (xc, yc, x_mean, y_mean) = if self.fit_intercept {
            let x_mean = x
                .mean_axis(Axis(0))
                .ok_or_else(|| FerroError::NumericalInstability {
                    message: "failed to compute column means of X".into(),
                })?;
            let y_mean = y
                .mean_axis(Axis(0))
                .ok_or_else(|| FerroError::NumericalInstability {
                    message: "failed to compute column means of Y".into(),
                })?;
            let xc = x - &x_mean;
            let yc = y - &y_mean;
            (xc, yc, x_mean, y_mean)
        } else {
            (
                x.clone(),
                y.clone(),
                Array1::<F>::zeros(p),
                Array1::<F>::zeros(t),
            )
        };

        // Internal working coefficient matrix is (n_features, n_tasks); the
        // stored `coefficients` is its transpose to match sklearn's
        // `(n_tasks, n_features)` `coef_`. Initialized to zeros.
        let mut w_mat = Array2::<F>::zeros((p, t));

        // Residual R = Yc - Xc.dot(W) = Yc initially (W == 0).
        let mut r = yc.clone();

        // norm_cols_x[j] = sum_i Xc[i,j]^2.
        let norm_cols_x: Vec<F> = (0..p)
            .map(|j| {
                let col = xc.column(j);
                col.dot(&col)
            })
            .collect();

        // Penalty split (`_coordinate_descent.py:655-656`):
        //   l1_reg = alpha * l1_ratio * n; l2_reg = alpha * (1 - l1_ratio) * n.
        let l1_reg = self.alpha * self.l1_ratio * n_f;
        let l2_reg = self.alpha * (F::one() - self.l1_ratio) * n_f;

        // tol_scaled = tol * ||Yc||_F^2 (sklearn `:832`, `tol *= norm(Y)^2`).
        let y_norm2 = yc.iter().fold(F::zero(), |s, &v| s + v * v);
        let tol_scaled = self.tol * y_norm2;
        let d_w_tol = self.tol;

        let mut n_iter_done = 0usize;

        for it in 0..self.max_iter {
            n_iter_done = it + 1; // 1-based, mirroring sklearn `n_iter_`.
            let mut w_max = F::zero();
            let mut d_w_max = F::zero();

            for j in 0..p {
                if norm_cols_x[j] == F::zero() {
                    continue;
                }

                // Store previous block W[j, :] (length t).
                let w_j_old = w_mat.row(j).to_owned();

                // tmp = norm_cols_x[j] * w_j_old + Xc[:, j]^T R.
                let mut tmp = xc.column(j).dot(&r);
                tmp = &tmp + &(&w_j_old * norm_cols_x[j]);

                // nn = ||tmp||_2.
                let nn = tmp.dot(&tmp).sqrt();

                // scaling = max(1 - l1_reg/nn, 0) / (norm_cols_x[j] + l2_reg)
                // (`_cd_fast.pyx:872-874`; 0 if nn == 0). The l2_reg term in the
                // denominator is the ONLY per-update difference vs MultiTaskLasso.
                let scaling = if nn == F::zero() {
                    F::zero()
                } else {
                    (F::one() - l1_reg / nn).max(F::zero()) / (norm_cols_x[j] + l2_reg)
                };

                let w_j_new = &tmp * scaling;

                // R -= Xc[:, j] outer (w_j_new - w_j_old).
                let delta = &w_j_new - &w_j_old;
                for i in 0..n {
                    let xij = xc[[i, j]];
                    for k in 0..t {
                        r[[i, k]] = r[[i, k]] - xij * delta[k];
                    }
                }

                w_mat.row_mut(j).assign(&w_j_new);

                // Track the largest coordinate update and coefficient magnitude
                // this sweep (`:894-901`).
                let d_w_j = delta.iter().fold(F::zero(), |m, &v| m.max(v.abs()));
                if d_w_j > d_w_max {
                    d_w_max = d_w_j;
                }
                let w_j_abs = w_j_new.iter().fold(F::zero(), |m, &v| m.max(v.abs()));
                if w_j_abs > w_max {
                    w_max = w_j_abs;
                }
            }

            // sklearn's two-level stop (`:903-952`): the relative-change gate
            // opens the (expensive) dual-gap check; break only when the gap
            // clears `tol * ||Y||_F^2`.
            let last_iter = it == self.max_iter - 1;
            if w_max == F::zero() || d_w_max / w_max < d_w_tol || last_iter {
                // XtA = Xc^T R - l2_reg * W, shape (p, t) (`:907-912`).
                let mut xta = xc.t().dot(&r);
                xta = &xta - &(&w_mat * l2_reg);

                // dual_norm_XtA = max_j ||XtA[j, :]||_2.
                let mut dual_norm = F::zero();
                for j in 0..p {
                    let row = xta.row(j);
                    let rn = row.dot(&row).sqrt();
                    if rn > dual_norm {
                        dual_norm = rn;
                    }
                }

                let r_norm2 = r.iter().fold(F::zero(), |s, &v| s + v * v);
                // w_norm2 = ||W||_F^2 = sum of squares of all entries of W.
                let w_norm2 = w_mat.iter().fold(F::zero(), |s, &v| s + v * v);

                let const_factor;
                let mut gap;
                if dual_norm > l1_reg {
                    let c = l1_reg / dual_norm;
                    let a_norm2 = r_norm2 * c * c;
                    gap = half * (r_norm2 + a_norm2);
                    const_factor = c;
                } else {
                    const_factor = F::one();
                    gap = r_norm2;
                }

                // l21 norm of W = sum_j ||W[j, :]||_2.
                let mut w21 = F::zero();
                for j in 0..p {
                    let row = w_mat.row(j);
                    w21 = w21 + row.dot(&row).sqrt();
                }

                // ry = sum elementwise R * Yc over all entries.
                let ry = (&r * &yc).sum();

                // gap += l1_reg*||W||_21 - const*R·Y
                //        + 0.5*l2_reg*(1 + const^2)*||W||_F^2   (`:944-948`).
                gap = gap + l1_reg * w21 - const_factor * ry
                    + half * l2_reg * (F::one() + const_factor * const_factor) * w_norm2;

                if gap < tol_scaled {
                    break;
                }
            }
        }

        // Store coef_ as (n_tasks, n_features) = W^T.
        let coefficients = w_mat.t().to_owned();

        // intercept[k] = y_mean[k] - x_mean · W[:, k] (zeros when !fit_intercept,
        // since both means are zero).
        let mut intercepts = Array1::<F>::zeros(t);
        if self.fit_intercept {
            for k in 0..t {
                intercepts[k] = y_mean[k] - x_mean.dot(&w_mat.column(k));
            }
        }

        Ok(FittedMultiTaskElasticNet {
            coefficients,
            intercepts,
            n_iter: n_iter_done,
        })
    }
}

impl<F: Float + Send + Sync + ScalarOperand + 'static> Predict<Array2<F>>
    for FittedMultiTaskElasticNet<F>
{
    type Output = Array2<F>;
    type Error = FerroError;

    /// Predict target values for the given feature matrix.
    ///
    /// Computes `pred[i, k] = sum_j X[i, j] * coef[k, j] + intercept[k]`, i.e.
    /// `X.dot(coef^T) + intercepts` (broadcast per task column), returning an
    /// `(n_samples, n_tasks)` array.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of features does not
    /// match the fitted model.
    fn predict(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let n_features = x.ncols();
        // coefficients is (n_tasks, n_features); columns == features.
        if n_features != self.coefficients.ncols() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![self.coefficients.ncols()],
                actual: vec![n_features],
                context: "number of features must match fitted model".into(),
            });
        }

        // pred = X · coef^T, shape (n_samples, n_tasks).
        let mut preds = x.dot(&self.coefficients.t());
        // Broadcast-add per-task intercepts.
        for (k, &b) in self.intercepts.iter().enumerate() {
            let mut col = preds.column_mut(k);
            col.mapv_inplace(|v| v + b);
        }
        Ok(preds)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::multi_task_lasso::MultiTaskLasso;
    use approx::assert_relative_eq;
    use ndarray::{array, s};

    /// Shared oracle fixture (R-CHAR-3).
    fn fixture() -> (Array2<f64>, Array2<f64>) {
        let x: Array2<f64> = array![[1.0, 2.0], [2.0, 1.0], [3.0, 4.0], [4.0, 3.0], [5.0, 5.0],];
        let y: Array2<f64> = array![[3.0, 1.0], [2.5, 2.0], [7.1, 3.5], [6.0, 4.2], [11.2, 6.0],];
        (x, y)
    }

    #[test]
    fn multi_task_elastic_net_matches_sklearn() -> Result<(), FerroError> {
        // Live sklearn 1.5.2 oracle (R-CHAR-3):
        //   from sklearn.linear_model import MultiTaskElasticNet; import numpy as np
        //   X=np.array([[1,2],[2,1],[3,4],[4,3],[5,5]],float)
        //   Y=np.array([[3,1],[2.5,2],[7.1,3.5],[6,4.2],[11.2,6]])
        //   m=MultiTaskElasticNet(alpha=0.3, l1_ratio=0.5).fit(X,Y)
        //   m.coef_  -> [[0.8163842473,1.3248847997],[0.8318338149,0.3403462995]]
        //   m.intercept_ -> [-0.4638071409,-0.1765403432]
        //   m.n_iter_ -> 16
        let (x, y) = fixture();
        let fitted = MultiTaskElasticNet::<f64>::new()
            .with_alpha(0.3)
            .with_l1_ratio(0.5)
            .fit(&x, &y)?;

        let coef = fitted.coefficients();
        assert_eq!(coef.dim(), (2, 2));
        assert_relative_eq!(coef[[0, 0]], 0.816_384_247_3, epsilon = 1e-6);
        assert_relative_eq!(coef[[0, 1]], 1.324_884_799_7, epsilon = 1e-6);
        assert_relative_eq!(coef[[1, 0]], 0.831_833_814_9, epsilon = 1e-6);
        assert_relative_eq!(coef[[1, 1]], 0.340_346_299_5, epsilon = 1e-6);

        let intercepts = fitted.intercepts();
        assert_relative_eq!(intercepts[0], -0.463_807_140_9, epsilon = 1e-6);
        assert_relative_eq!(intercepts[1], -0.176_540_343_2, epsilon = 1e-6);

        assert_eq!(fitted.n_iter(), 16);
        Ok(())
    }

    #[test]
    fn multi_task_elastic_net_no_intercept_matches_sklearn() -> Result<(), FerroError> {
        // Live sklearn 1.5.2 oracle (R-CHAR-3):
        //   m=MultiTaskElasticNet(alpha=0.3, l1_ratio=0.5,
        //                         fit_intercept=False).fit(X,Y)
        //   m.coef_ -> [[0.7558077559,1.2576697088],[0.8053172672,0.318103642]]
        //   m.intercept_ -> [0.,0.]
        //   m.n_iter_ -> 68
        let (x, y) = fixture();
        let fitted = MultiTaskElasticNet::<f64>::new()
            .with_alpha(0.3)
            .with_l1_ratio(0.5)
            .with_fit_intercept(false)
            .fit(&x, &y)?;

        let coef = fitted.coefficients();
        assert_relative_eq!(coef[[0, 0]], 0.755_807_755_9, epsilon = 1e-6);
        assert_relative_eq!(coef[[0, 1]], 1.257_669_708_8, epsilon = 1e-6);
        assert_relative_eq!(coef[[1, 0]], 0.805_317_267_2, epsilon = 1e-6);
        assert_relative_eq!(coef[[1, 1]], 0.318_103_642, epsilon = 1e-6);

        let intercepts = fitted.intercepts();
        assert_eq!(intercepts[0], 0.0);
        assert_eq!(intercepts[1], 0.0);

        assert_eq!(fitted.n_iter(), 68);
        Ok(())
    }

    #[test]
    fn multi_task_elastic_net_predict_matches_sklearn() -> Result<(), FerroError> {
        // Live sklearn 1.5.2 oracle (R-CHAR-3):
        //   m=MultiTaskElasticNet(alpha=0.3, l1_ratio=0.5).fit(X,Y)
        //   m.predict(X[:2])
        //   -> [[3.00234671,1.33598607],[2.49384615,1.82747359]]
        let (x, y) = fixture();
        let fitted = MultiTaskElasticNet::<f64>::new()
            .with_alpha(0.3)
            .with_l1_ratio(0.5)
            .fit(&x, &y)?;

        let x_head = x.slice(s![0..2, ..]).to_owned();
        let preds = fitted.predict(&x_head)?;

        assert_eq!(preds.dim(), (2, 2));
        assert_relative_eq!(preds[[0, 0]], 3.002_346_71, epsilon = 1e-6);
        assert_relative_eq!(preds[[0, 1]], 1.335_986_07, epsilon = 1e-6);
        assert_relative_eq!(preds[[1, 0]], 2.493_846_15, epsilon = 1e-6);
        assert_relative_eq!(preds[[1, 1]], 1.827_473_59, epsilon = 1e-6);
        Ok(())
    }

    #[test]
    fn multi_task_elastic_net_l1_ratio_one_equals_multitask_lasso() -> Result<(), FerroError> {
        // `MultiTaskElasticNet(l1_ratio=1)` == `MultiTaskLasso` (l2_reg = 0).
        // Live sklearn 1.5.2 oracle (R-CHAR-3): MultiTaskLasso(alpha=0.3).coef_
        //   -> [[0.7874471321,1.3745821226],[0.8341004367,0.3460953631]]
        let (x, y) = fixture();
        let enet = MultiTaskElasticNet::<f64>::new()
            .with_alpha(0.3)
            .with_l1_ratio(1.0)
            .fit(&x, &y)?;

        let coef = enet.coefficients();
        assert_relative_eq!(coef[[0, 0]], 0.787_447_132_1, epsilon = 1e-6);
        assert_relative_eq!(coef[[0, 1]], 1.374_582_122_6, epsilon = 1e-6);
        assert_relative_eq!(coef[[1, 0]], 0.834_100_436_7, epsilon = 1e-6);
        assert_relative_eq!(coef[[1, 1]], 0.346_095_363_1, epsilon = 1e-6);

        // And it should agree with the actual MultiTaskLasso estimator.
        let lasso = MultiTaskLasso::<f64>::new().with_alpha(0.3).fit(&x, &y)?;
        let lcoef = lasso.coefficients();
        for (a, b) in coef.iter().zip(lcoef.iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-6);
        }
        Ok(())
    }

    #[test]
    fn multi_task_elastic_net_validation_errors() {
        let (x, y) = fixture();

        // Negative alpha -> InvalidParameter.
        let res = MultiTaskElasticNet::<f64>::new()
            .with_alpha(-1.0)
            .fit(&x, &y);
        assert!(matches!(res, Err(FerroError::InvalidParameter { .. })));

        // l1_ratio = 1.5 (out of [0, 1]) -> InvalidParameter.
        let res2 = MultiTaskElasticNet::<f64>::new()
            .with_l1_ratio(1.5)
            .fit(&x, &y);
        assert!(matches!(res2, Err(FerroError::InvalidParameter { .. })));

        // Y rows != X rows -> ShapeMismatch.
        let x_bad: Array2<f64> = array![[1.0, 2.0], [2.0, 1.0], [3.0, 4.0]];
        let y_bad: Array2<f64> = array![[3.0, 1.0], [2.5, 2.0]];
        let res3 = MultiTaskElasticNet::<f64>::new().fit(&x_bad, &y_bad);
        assert!(matches!(res3, Err(FerroError::ShapeMismatch { .. })));
    }
}
