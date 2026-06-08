//! Multi-task Lasso regression (joint multi-output L21 block coordinate descent).
//!
//! This module provides [`MultiTaskLasso`], the multi-output linear model that
//! fits all target columns jointly under an L2,1 (group-Lasso) penalty,
//! minimizing
//!
//! ```text
//! (1 / (2 * n_samples)) * ||Y - X W||_F^2 + alpha * ||W||_21
//! ```
//!
//! where `||W||_21 = sum_j sqrt(sum_k W[j,k]^2)` is the sum over features of the
//! L2 norm of each feature's coefficient ROW across tasks. The mixed L2,1 norm
//! couples a feature's coefficients across all tasks: a feature is either active
//! for ALL tasks (a non-zero row) or inactive for all of them (an all-zero row),
//! so `MultiTaskLasso` performs joint feature selection across outputs.
//!
//! Mirrors `sklearn.linear_model.MultiTaskLasso`
//! (`sklearn/linear_model/_coordinate_descent.py:2663`, `class
//! MultiTaskLasso(MultiTaskElasticNet)`); the production solver is the Cython
//! `enet_coordinate_descent_multi_task` in `_cd_fast.pyx:740` (objective at
//! `:756`, `0.5 * norm(Y - X W.T)^2 + l1_reg ||W.T||_21 + 0.5 * l2_reg
//! norm(W.T)^2`). `MultiTaskLasso` is `MultiTaskElasticNet(l1_ratio=1.0)`, i.e.
//! `l2_reg = 0`, `l1_reg = alpha * n_samples`. ferrolearn implements the dense
//! block-coordinate-descent core directly.
//!
//! ## REQ status (per `.design/linear/lasso.md`, mirrors `sklearn/linear_model/_coordinate_descent.py` @ 1.5.2)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-13 (MultiTaskLasso, multi-output L21 block CD) | SHIPPED | `MultiTaskLasso<F>` / `FittedMultiTaskLasso<F>` in this module: `impl Fit<Array2<F>, Array2<F>>` runs block coordinate descent porting `_cd_fast.pyx::enet_coordinate_descent_multi_task` (`:740-959`, `l2_reg=0`): `l1_reg = alpha*n`, per-feature block update `W[j,:] = tmp * max(1 - l1_reg/||tmp||, 0) / norm_cols_X[j]`, residual rank-1 maintenance, and the two-level relative-change + dual-gap stop (`:903-950`, `tol_scaled = tol*||Y||_F^2`). `coef_` is stored `(n_tasks, n_features)` matching sklearn `coef_`. The `dual_gap_` fitted attribute is now exposed via the `dual_gap: F` field + `#[must_use] pub fn dual_gap()` getter: `Fit::fit` captures the final block-CD duality gap (the value deciding convergence) into `final_gap` and stores it scaled `final_gap / n_samples`, mirroring `self.dual_gap_ /= n_samples` (`_coordinate_descent.py:2652`, unpacked at `:2636`). Verified against the live sklearn 1.5.2 oracle (R-CHAR-3): `MultiTaskLasso(alpha=0.3)` -> `dual_gap_=0.00021539018133829302`, `alpha=0.1` -> `0.00016093048471601534`, `alpha=1.0` -> `0.0001449879028545098` (`tests/divergence_multi_task_lasso.rs::mtl_dual_gap_matches_sklearn`, tol 1e-9). Verified against the live sklearn 1.5.2 oracle (R-CHAR-3): on `X=[[1,2],[2,1],[3,4],[4,3],[5,5]]`, `Y=[[3,1],[2.5,2],[7.1,3.5],[6,4.2],[11.2,6]]`, `MultiTaskLasso(alpha=0.3)` -> `coef_=[[0.7874471321,1.3745821226],[0.8341004367,0.3460953631]]`, `intercept_=[-0.5260877641,-0.2005873993]`, `n_iter_=19`. Input validation matches sklearn's `_validate_data(force_all_finite=True)` (`_coordinate_descent.py:2602`): any NaN/+/-inf in X or Y is rejected with `FerroError::InvalidParameter` BEFORE the solver (#2, `tests/divergence_multi_task_lasso_nonfinite.rs::mtl_rejects_non_finite_input_like_sklearn`). Tests `multi_task_lasso_matches_sklearn`, `multi_task_lasso_no_intercept_matches_sklearn`, `multi_task_lasso_group_sparsity`, `multi_task_lasso_predict_matches_sklearn`, `multi_task_lasso_shape_mismatch_errors`. Non-test consumer: `MultiTaskLasso` is the public estimator boundary API re-exported at the crate root (`ferrolearn_linear::MultiTaskLasso`, grandfathered boundary per goal.md S5). |
//!
//! # Examples
//!
//! ```
//! use ferrolearn_linear::MultiTaskLasso;
//! use ferrolearn_core::{Fit, Predict};
//! use ndarray::{array, Array2};
//!
//! let model = MultiTaskLasso::<f64>::new().with_alpha(0.3);
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

/// Multi-task Lasso regression (joint multi-output L2,1-regularized least
/// squares).
///
/// Fits all target columns jointly under a group-Lasso (L2,1) penalty, so each
/// feature is either active for all tasks or zero for all of them. Mirrors
/// `sklearn.linear_model.MultiTaskLasso`
/// (`_coordinate_descent.py:2663`).
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
#[derive(Debug, Clone)]
pub struct MultiTaskLasso<F> {
    /// Regularization strength on the L2,1 penalty. Larger values specify
    /// stronger regularization and zero out more whole feature rows.
    pub alpha: F,
    /// Whether to fit a per-task intercept (bias) term.
    pub fit_intercept: bool,
    /// Maximum number of block-coordinate-descent iterations.
    pub max_iter: usize,
    /// Convergence tolerance on the relative coefficient change / dual gap.
    pub tol: F,
}

impl<F: Float> MultiTaskLasso<F> {
    /// Create a new `MultiTaskLasso` with default settings.
    ///
    /// Defaults: `alpha = 1.0`, `fit_intercept = true`, `max_iter = 1000`,
    /// `tol = 1e-4` — mirroring sklearn's ctor defaults
    /// `MultiTaskLasso(alpha=1.0, fit_intercept=True, max_iter=1000, tol=1e-4)`
    /// (`_coordinate_descent.py:2663`).
    #[must_use]
    pub fn new() -> Self {
        Self {
            alpha: F::one(),
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

impl<F: Float> Default for MultiTaskLasso<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted multi-task Lasso regression model.
///
/// Stores the learned coefficient matrix (shape `(n_tasks, n_features)`, the
/// sklearn `coef_` layout), the per-task intercept vector, and the number of
/// block-coordinate-descent sweeps run. Implements [`Predict`].
#[derive(Debug, Clone)]
pub struct FittedMultiTaskLasso<F> {
    /// Learned coefficients, shape `(n_tasks, n_features)` — matches sklearn's
    /// `MultiTaskLasso.coef_` layout exactly. Whole feature columns (across the
    /// task rows) are jointly zero or jointly non-zero.
    coefficients: Array2<F>,
    /// Per-task intercept vector, length `n_tasks`. Filled with zeros when
    /// `fit_intercept = false`.
    intercepts: Array1<F>,
    /// Number of block-coordinate-descent sweeps run by the solver (1-based;
    /// mirrors sklearn `MultiTaskLasso.n_iter_`).
    n_iter: usize,
    /// Duality gap at the returned solution, on the `(1 / (2 * n_samples))`-scaled
    /// objective (mirrors sklearn `MultiTaskLasso.dual_gap_`).
    dual_gap: F,
}

impl<F: Float> FittedMultiTaskLasso<F> {
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

    /// Duality gap at the returned solution, on the `(1 / (2 * n_samples))`-scaled
    /// objective.
    ///
    /// Mirrors sklearn's `MultiTaskLasso.dual_gap_` attribute
    /// (`_coordinate_descent.py:2636` — unpacked from
    /// `enet_coordinate_descent_multi_task` — then `:2652`
    /// `self.dual_gap_ /= n_samples`, the final objective-scaling). This is the
    /// final block-CD duality gap (the value that decided convergence), scaled by
    /// `1 / n_samples` so the exposed value matches sklearn's `dual_gap_`.
    #[must_use]
    pub fn dual_gap(&self) -> F {
        self.dual_gap
    }
}

impl<F: Float + Send + Sync + ScalarOperand + FromPrimitive + 'static> Fit<Array2<F>, Array2<F>>
    for MultiTaskLasso<F>
{
    type Fitted = FittedMultiTaskLasso<F>;
    type Error = FerroError;

    /// Fit the multi-task Lasso model using block coordinate descent.
    ///
    /// Ports sklearn's `enet_coordinate_descent_multi_task`
    /// (`_cd_fast.pyx:740`) with `l2_reg = 0` and `l1_reg = alpha * n_samples`.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if `Y.nrows()` differs from the
    /// number of samples in `X`.
    /// Returns [`FerroError::InvalidParameter`] if `alpha` is negative.
    /// Returns [`FerroError::InsufficientSamples`] if there are no samples.
    /// Returns [`FerroError::NumericalInstability`] if a required float constant
    /// or column mean cannot be formed.
    fn fit(&self, x: &Array2<F>, y: &Array2<F>) -> Result<FittedMultiTaskLasso<F>, FerroError> {
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

        if n_samples == 0 {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "MultiTaskLasso requires at least one sample".into(),
            });
        }

        // sklearn `MultiTaskElasticNet.fit` -> `self._validate_data(X, y,
        // validate_separately=(check_X_params, check_y_params))`
        // (`_coordinate_descent.py:2602`); both param dicts (`:2595`,`:2601`)
        // inherit the default `force_all_finite=True`, so `check_array` rejects
        // any NaN or +/-inf in X OR Y with a `ValueError` BEFORE the solver runs.
        // `.iter().any(|v| !v.is_finite())` rejects both NaN and Inf, matching
        // the crate idiom (e.g. `one_hot_encoder.rs`). (#2)
        if x.iter().any(|v| !v.is_finite()) {
            return Err(FerroError::InvalidParameter {
                name: "X".into(),
                reason: "Input X contains NaN or infinity.".into(),
            });
        }
        if y.iter().any(|v| !v.is_finite()) {
            return Err(FerroError::InvalidParameter {
                name: "y".into(),
                reason: "Input y contains NaN or infinity.".into(),
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

        // l1_reg = alpha * n (MultiTaskLasso has l2_reg = 0).
        let l1_reg = self.alpha * n_f;

        // tol_scaled = tol * ||Yc||_F^2 (sklearn `:832`, `tol *= norm(Y)^2`).
        let y_norm2 = yc.iter().fold(F::zero(), |s, &v| s + v * v);
        let tol_scaled = self.tol * y_norm2;
        let d_w_tol = self.tol;

        let mut n_iter_done = 0usize;
        // Final duality gap (un-normalized objective, like sklearn's
        // `enet_coordinate_descent_multi_task` return); scaled by `1/n` on store
        // to mirror `dual_gap_ /= n_samples` (`_coordinate_descent.py:2652`).
        let mut final_gap = F::zero();

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

                // scaling = max(1 - l1_reg/nn, 0) / norm_cols_x[j] (0 if nn==0).
                let scaling = if nn == F::zero() {
                    F::zero()
                } else {
                    (F::one() - l1_reg / nn).max(F::zero()) / norm_cols_x[j]
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
                // XtA = Xc^T R (l2_reg = 0), shape (p, t).
                let xta = xc.t().dot(&r);

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

                gap = gap + l1_reg * w21 - const_factor * ry;

                // Record the most recent gap; whichever iteration last runs the
                // dual-gap check (the convergence break or the final sweep) is the
                // value sklearn returns as `dual_gap_` (pre `/n_samples`).
                final_gap = gap;

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

        Ok(FittedMultiTaskLasso {
            coefficients,
            intercepts,
            n_iter: n_iter_done,
            // sklearn `_coordinate_descent.py:2652`: `self.dual_gap_ /= n_samples`
            // maps the solver's un-normalized gap to the `(1/2n)`-scaled objective.
            dual_gap: final_gap / n_f,
        })
    }
}

impl<F: Float + Send + Sync + ScalarOperand + 'static> Predict<Array2<F>>
    for FittedMultiTaskLasso<F>
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
    use approx::assert_relative_eq;
    use ndarray::{array, s};

    /// Shared oracle fixture (R-CHAR-3).
    fn fixture() -> (Array2<f64>, Array2<f64>) {
        let x: Array2<f64> = array![[1.0, 2.0], [2.0, 1.0], [3.0, 4.0], [4.0, 3.0], [5.0, 5.0],];
        let y: Array2<f64> = array![[3.0, 1.0], [2.5, 2.0], [7.1, 3.5], [6.0, 4.2], [11.2, 6.0],];
        (x, y)
    }

    #[test]
    fn multi_task_lasso_matches_sklearn() -> Result<(), FerroError> {
        // Live sklearn 1.5.2 oracle (R-CHAR-3):
        //   from sklearn.linear_model import MultiTaskLasso; import numpy as np
        //   X=np.array([[1,2],[2,1],[3,4],[4,3],[5,5]],float)
        //   Y=np.array([[3,1],[2.5,2],[7.1,3.5],[6,4.2],[11.2,6]])
        //   m=MultiTaskLasso(alpha=0.3).fit(X,Y)
        //   m.coef_  -> [[0.7874471321,1.3745821226],[0.8341004367,0.3460953631]]
        //   m.intercept_ -> [-0.5260877641,-0.2005873993]
        //   m.n_iter_ -> 19
        let (x, y) = fixture();
        let fitted = MultiTaskLasso::<f64>::new().with_alpha(0.3).fit(&x, &y)?;

        let coef = fitted.coefficients();
        assert_eq!(coef.dim(), (2, 2));
        assert_relative_eq!(coef[[0, 0]], 0.787_447_132_1, epsilon = 1e-6);
        assert_relative_eq!(coef[[0, 1]], 1.374_582_122_6, epsilon = 1e-6);
        assert_relative_eq!(coef[[1, 0]], 0.834_100_436_7, epsilon = 1e-6);
        assert_relative_eq!(coef[[1, 1]], 0.346_095_363_1, epsilon = 1e-6);

        let intercepts = fitted.intercepts();
        assert_relative_eq!(intercepts[0], -0.526_087_764_1, epsilon = 1e-6);
        assert_relative_eq!(intercepts[1], -0.200_587_399_3, epsilon = 1e-6);

        assert_eq!(fitted.n_iter(), 19);
        Ok(())
    }

    #[test]
    fn multi_task_lasso_no_intercept_matches_sklearn() -> Result<(), FerroError> {
        // Live sklearn 1.5.2 oracle (R-CHAR-3):
        //   m=MultiTaskLasso(alpha=0.3, fit_intercept=False).fit(X,Y)
        //   m.coef_ -> [[0.7223086317,1.2938631723],[0.8006773177,0.3236384717]]
        //   m.intercept_ -> [0.,0.]
        //   m.n_iter_ -> 85
        let (x, y) = fixture();
        let fitted = MultiTaskLasso::<f64>::new()
            .with_alpha(0.3)
            .with_fit_intercept(false)
            .fit(&x, &y)?;

        let coef = fitted.coefficients();
        assert_relative_eq!(coef[[0, 0]], 0.722_308_631_7, epsilon = 1e-6);
        assert_relative_eq!(coef[[0, 1]], 1.293_863_172_3, epsilon = 1e-6);
        assert_relative_eq!(coef[[1, 0]], 0.800_677_317_7, epsilon = 1e-6);
        assert_relative_eq!(coef[[1, 1]], 0.323_638_471_7, epsilon = 1e-6);

        let intercepts = fitted.intercepts();
        assert_eq!(intercepts[0], 0.0);
        assert_eq!(intercepts[1], 0.0);

        assert_eq!(fitted.n_iter(), 85);
        Ok(())
    }

    #[test]
    fn multi_task_lasso_group_sparsity() -> Result<(), FerroError> {
        // With a large alpha the L21 penalty zeros WHOLE feature rows jointly:
        // every coefficient is driven to (bit-near) zero.
        let (x, y) = fixture();
        let fitted = MultiTaskLasso::<f64>::new().with_alpha(5.0).fit(&x, &y)?;

        let coef = fitted.coefficients();
        for &c in coef.iter() {
            assert_relative_eq!(c, 0.0, epsilon = 1e-9);
        }
        Ok(())
    }

    #[test]
    fn multi_task_lasso_predict_matches_sklearn() -> Result<(), FerroError> {
        // Live sklearn 1.5.2 oracle (R-CHAR-3):
        //   m=MultiTaskLasso(alpha=0.3).fit(X,Y); m.predict(X[:2])
        //   -> [[3.01052361,1.32570376],[2.42338862,1.81370884]]
        let (x, y) = fixture();
        let fitted = MultiTaskLasso::<f64>::new().with_alpha(0.3).fit(&x, &y)?;

        let x_head = x.slice(s![0..2, ..]).to_owned();
        let preds = fitted.predict(&x_head)?;

        assert_eq!(preds.dim(), (2, 2));
        assert_relative_eq!(preds[[0, 0]], 3.010_523_61, epsilon = 1e-6);
        assert_relative_eq!(preds[[0, 1]], 1.325_703_76, epsilon = 1e-6);
        assert_relative_eq!(preds[[1, 0]], 2.423_388_62, epsilon = 1e-6);
        assert_relative_eq!(preds[[1, 1]], 1.813_708_84, epsilon = 1e-6);
        Ok(())
    }

    #[test]
    fn multi_task_lasso_shape_mismatch_errors() {
        // Y rows != X rows -> ShapeMismatch.
        let x: Array2<f64> = array![[1.0, 2.0], [2.0, 1.0], [3.0, 4.0]];
        let y_bad: Array2<f64> = array![[3.0, 1.0], [2.5, 2.0]];
        let res = MultiTaskLasso::<f64>::new().fit(&x, &y_bad);
        assert!(matches!(res, Err(FerroError::ShapeMismatch { .. })));

        // Negative alpha -> InvalidParameter.
        let y_ok: Array2<f64> = array![[3.0, 1.0], [2.5, 2.0], [7.1, 3.5]];
        let res2 = MultiTaskLasso::<f64>::new().with_alpha(-1.0).fit(&x, &y_ok);
        assert!(matches!(res2, Err(FerroError::InvalidParameter { .. })));
    }
}
