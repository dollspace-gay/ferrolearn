//! Ridge regression (L2-regularized linear regression).
//!
//! This module provides [`Ridge`], which fits a linear model with L2
//! regularization using the closed-form solution:
//!
//! ```text
//! w = (X^T X + alpha * I)^{-1} X^T y
//! ```
//!
//! The regularization parameter `alpha` controls the strength of the
//! L2 penalty, shrinking coefficients toward zero.
//!
//! ## REQ status (per `.design/linear/ridge.md`, mirrors `sklearn/linear_model/_ridge.py` @ 1.5.2)
//!
//! Mirrors `sklearn.linear_model.Ridge` (`_ridge.py:1016`), default dense path
//! `solver='auto'`→`'cholesky'` with `fit_intercept` via centering (intercept unpenalized).
//! coef_/intercept_ match the live sklearn oracle to 1e-8 across alpha∈{0.1,1,10,100}.
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (L2 cholesky fit, intercept unpenalized) | SHIPPED | `Fit for Ridge` (centering + `linalg::solve_ridge`). Consumer: `RsRidge` in `ferrolearn-python/src/regressors.rs`. |
//! | REQ-2 (predict = X·coef + intercept) | SHIPPED | `Predict for FittedRidge`. |
//! | REQ-3 (fit_intercept incl. false) | SHIPPED | `with_fit_intercept`. |
//! | REQ-4 (HasCoefficients introspection) | SHIPPED | `HasCoefficients for FittedRidge`. |
//! | REQ-5 (alpha≥0 validation; alpha=0 → OLS incl. rank-deficient min-norm) | SHIPPED | negative-alpha → `InvalidParameter`; alpha=0 singular falls back `solve_ridge` → `solve_lstsq` (ferray min-norm), mirroring sklearn cholesky→SVD (`_ridge.py:752-756`). Closed #392; test `divergence_ridge_alpha_zero_rank_deficient_min_norm`. |
//! | REQ-6 (multi-output 2-D Y → 2-D coef_) | NOT-STARTED | `FittedRidgeMulti` exists, no production consumer (#384). |
//! | REQ-7 (per-target alpha array) | NOT-STARTED | #385. |
//! | REQ-8 (solver variants + solver_) | NOT-STARTED | #386. |
//! | REQ-9 (positive=True) | SHIPPED | `Ridge<F>` adds `pub positive: bool` (default `false`, `_ridge.py:902`/`:911`) + `with_positive(bool)` builder. When `self.positive`, `fit_with_sample_weight` routes the coefficient solve through `solve_nonneg_ridge` — projected coordinate descent minimizing `0.5·‖A·w−b‖² + 0.5·alpha·‖w‖²` s.t. `w ≥ 0` (`new = max(0, (A[:,j]ᵀr + col_sq[j]·old)/(col_sq[j] + alpha))`, incremental residual update, `max_iter=self.max_iter.unwrap_or(1000)`/`self.tol`) on the SAME centered (and `√w`-rescaled) design `solve_ridge` uses, then recovers `intercept = y_off − x_off·coef` (fit_intercept) / `0` identically — the same unique optimum sklearn's L-BFGS-B (`_solve_lbfgs`, `_ridge.py:300`, objective `0.5·‖Xw−y‖²+0.5·alpha·‖w‖²`, bounds `[(0,inf)]`) reaches, dispatched at `_ridge.py:923-928`. `n_iter_ = Some(iters)` on the positive (iterative CD) path; `None` for the direct Cholesky path. `positive=false` (default) is byte-identical to the unconstrained Cholesky path. Oracle tests: `ridge_positive_matches_sklearn` (alpha=1, fit_intercept coef `[1.19891304, 0.0]`, intercept `-6.17744565`, all ≥ 0, differs from unconstrained `[0.95708502, -1.85401484]`), `ridge_positive_false_unchanged` (byte-identical guard), `ridge_positive_all_nonneg_equals_unconstrained` (inactive-constraint sanity). Closes #387. |
//! | REQ-10 (max_iter/tol + n_iter_) | SHIPPED | `Ridge<F>` adds `pub max_iter: Option<usize>` (default `None`) and `pub tol: F` (default `1e-4`) with `with_max_iter`/`with_tol` builders. `FittedRidge<F>` adds `n_iter_: Option<usize>` (always `None` for the direct Cholesky solver) with `pub fn n_iter(&self) -> Option<usize>`. Mirrors sklearn ctor `max_iter=None, tol=1e-4` (`_ridge.py:899-900`) and `n_iter_` set at `_ridge.py:994`; `max_iter`/`tol` are no-ops for the direct solver (closed-form normal equations, no iteration) — matching sklearn's direct `cholesky`/`svd` paths which also yield `n_iter_=None`. Test: `ridge_max_iter_tol_niter_defaults_and_builders`. Closes #388. |
//! | REQ-11 (sample_weight) | SHIPPED | `Ridge::fit_with_sample_weight(x, y, sample_weight: Option<&Array1<F>>)` solves WEIGHTED ridge `min Σᵢ wᵢ(yᵢ−xᵢ·coef)² + alpha·‖coef‖²`: weighted offsets `x_off[j]=Σwᵢx[i,j]/Σwᵢ`, `y_off=Σwᵢyᵢ/Σwᵢ` (fit_intercept), centering, then `√wᵢ` row-rescaling (`_rescale_data`, `_ridge.py:682-688`), `linalg::solve_ridge(&Xs, &ys, alpha)` with the penalty `alpha` UNSCALED (since `Xsᵀ·Xs == Xᵀ·W·X`), `intercept = y_off − x_off·coef`; `fit_intercept=false` skips centering (raw `√w`-rescale, intercept 0). `Fit::fit` delegates `fit_with_sample_weight(x, y, None)` (None byte-identical to the historic centering + `solve_ridge` body; alpha=0 OLS min-norm fallback preserved). Oracle tests `ridge_fit_sample_weight_with_intercept_matches_sklearn` (alpha=1 coef `[0.9233502538, 1.39678511]`, intercept `-0.8033840948`, differs from unweighted `[0.8228070175, 1.3561403509]`), `ridge_fit_sample_weight_no_intercept_matches_sklearn` (alpha=2 coef `[0.7273779983, 1.3737799835]`, intercept 0), `ridge_fit_none_sample_weight_equals_unweighted` (byte-identical guard). Closes #389. |
//! | REQ-12 (copy_X/random_state) | SHIPPED | `Ridge<F>` adds `pub copy_x: bool` (default `true`) and `pub random_state: Option<u64>` (default `None`) fields with `with_copy_x`/`with_random_state` builders. `copy_x` ABI-only (fit never mutates `x`); `random_state` stored-but-no-op for the deterministic Cholesky solver (only `sag`/`saga` use it, `_ridge.py:898`/`:903`). Test: `ridge_copy_x_random_state_defaults_and_builders`. Closes #390. |
//! | REQ-13 (ferray substrate) | NOT-STARTED | #391 (alpha=0 fallback already on ferray::linalg::lstsq; coef return tied to #359). |
//!
//! acto-critic: core L2 numerics (coef/intercept, alpha scaling, fit_intercept, f32) match the
//! live oracle; one divergence (#392, alpha=0 rank-deficient min-norm) found and fixed.
//! Two states only per goal.md R-DEFER-2.
//!
//! # Examples
//!
//! ```
//! use ferrolearn_linear::Ridge;
//! use ferrolearn_core::{Fit, Predict};
//! use ndarray::{array, Array1, Array2};
//!
//! let model = Ridge::<f64>::new();
//! let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
//! let y = array![2.0, 4.0, 6.0, 8.0];
//!
//! let fitted = model.fit(&x, &y).unwrap();
//! let preds = fitted.predict(&x).unwrap();
//! ```

use ferray::linalg::LinalgFloat;
use ferrolearn_core::error::FerroError;
use ferrolearn_core::introspection::HasCoefficients;
use ferrolearn_core::pipeline::{FittedPipelineEstimator, PipelineEstimator};
use ferrolearn_core::traits::{Fit, Predict};
use ndarray::{Array1, Array2, Axis, ScalarOperand};
use num_traits::{Float, FromPrimitive};

use crate::linalg;

/// Ridge regression (L2-regularized least squares).
///
/// Adds an L2 penalty to the ordinary least squares objective, which
/// shrinks coefficients toward zero and can help with multicollinearity.
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
#[derive(Debug, Clone)]
pub struct Ridge<F> {
    /// Regularization strength. Larger values specify stronger
    /// regularization.
    pub alpha: F,
    /// Whether to fit an intercept (bias) term.
    pub fit_intercept: bool,
    /// Whether `X` may be overwritten during fit (sklearn `copy_X`,
    /// `_ridge.py:898`). ferrolearn's `fit` never mutates `x` (it reads
    /// via `.mean_axis()`/`.outer_iter()`), so the observable
    /// non-mutation contract holds regardless; the field is exposed for
    /// ABI parity with sklearn. Default `true`, matching sklearn's
    /// `copy_X=True` default (`_ridge.py:898`).
    pub copy_x: bool,
    /// Random seed for the `sag`/`saga` solvers (sklearn `random_state`,
    /// `_ridge.py:903`). ferrolearn's default solver is deterministic
    /// Cholesky (`solver='auto'`→`'cholesky'`), so this field is stored
    /// for ABI parity and has no effect on the computed coefficients.
    /// Default `None`, matching sklearn's `random_state=None` (`_ridge.py:903`).
    pub random_state: Option<u64>,
    /// Maximum number of iterations for iterative solvers (sklearn `max_iter`,
    /// `_ridge.py:899`). Exposed for sklearn ABI parity; the implemented
    /// direct Cholesky solver solves the normal equations in closed form with
    /// no iteration, so this field is stored but has no effect on the computed
    /// result. When an iterative solver is added (future REQ-8 #386), this
    /// will control convergence. Default `None`, matching sklearn's default
    /// (`_ridge.py:899`).
    pub max_iter: Option<usize>,
    /// Tolerance for iterative solvers (sklearn `tol`, `_ridge.py:900`).
    /// Exposed for sklearn ABI parity; the implemented direct Cholesky solver
    /// solves the normal equations in closed form with no iteration, so this
    /// field is stored but has no effect on the computed result. When an
    /// iterative solver is added (future REQ-8 #386), this will control
    /// convergence. Default `1e-4`, matching sklearn's default (`_ridge.py:900`).
    pub tol: F,
    /// When `true`, constrain the fitted coefficients to be non-negative
    /// (sklearn `positive`, `_ridge.py:902`/`:911`). sklearn solves the
    /// non-negative ridge QP `min 0.5·‖X·w − y‖² + 0.5·alpha·‖w‖²` subject to
    /// `w ≥ 0` via the L-BFGS-B solver (`_solve_lbfgs`, `_ridge.py:300`,
    /// dispatched at `:923-928`); ferrolearn solves the same unique optimum
    /// with projected coordinate descent. Default `false`, matching sklearn's
    /// `positive=False` (`_ridge.py:902`); `positive=false` is byte-identical
    /// to the unconstrained Cholesky path.
    pub positive: bool,
}

impl<F: Float> Ridge<F> {
    /// Create a new `Ridge` with default settings.
    ///
    /// Defaults: `alpha = 1.0`, `fit_intercept = true`, `copy_x = true`,
    /// `random_state = None`, `max_iter = None`, `tol = 1e-4`,
    /// `positive = false` — mirroring sklearn's ctor defaults
    /// (`sklearn/linear_model/_ridge.py:895-903`).
    #[must_use]
    pub fn new() -> Self {
        Self {
            alpha: F::one(),
            fit_intercept: true,
            copy_x: true,
            random_state: None,
            max_iter: None,
            tol: F::from(1e-4).unwrap_or_else(F::epsilon),
            positive: false,
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

    /// Set the `copy_X` flag (sklearn `copy_X`, `_ridge.py:898`).
    ///
    /// ferrolearn's fit never mutates `x`, so this is exposed for ABI
    /// parity with sklearn and does not change the computed result.
    #[must_use]
    pub fn with_copy_x(mut self, copy_x: bool) -> Self {
        self.copy_x = copy_x;
        self
    }

    /// Set the `random_state` seed (sklearn `random_state`, `_ridge.py:903`).
    ///
    /// Only used by sklearn's `sag`/`saga` solvers. ferrolearn's default
    /// solver is deterministic Cholesky, so this is stored for ABI parity
    /// and has no effect on the computed coefficients.
    #[must_use]
    pub fn with_random_state(mut self, random_state: Option<u64>) -> Self {
        self.random_state = random_state;
        self
    }

    /// Set the maximum number of iterations for iterative solvers (sklearn
    /// `max_iter`, `_ridge.py:899`).
    ///
    /// ferrolearn's direct Cholesky solver solves the normal equations in
    /// closed form with no iteration, so this is stored for sklearn ABI
    /// parity and does not affect the computed result. When an iterative
    /// solver is added (future REQ-8 #386), this will take effect.
    #[must_use]
    pub fn with_max_iter(mut self, max_iter: Option<usize>) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the convergence tolerance for iterative solvers (sklearn `tol`,
    /// `_ridge.py:900`).
    ///
    /// ferrolearn's direct Cholesky solver solves the normal equations in
    /// closed form with no iteration, so this is stored for sklearn ABI
    /// parity and does not affect the computed result. When an iterative
    /// solver is added (future REQ-8 #386), this will take effect.
    #[must_use]
    pub fn with_tol(mut self, tol: F) -> Self {
        self.tol = tol;
        self
    }

    /// Set whether to constrain the fitted coefficients to be non-negative
    /// (sklearn `positive`, `_ridge.py:902`/`:911`).
    ///
    /// When `true`, ferrolearn solves the non-negative ridge QP
    /// `min 0.5·‖X·w − y‖² + 0.5·alpha·‖w‖²` subject to `w ≥ 0` via projected
    /// coordinate descent — the same unique optimum sklearn reaches with its
    /// L-BFGS-B solver (`_solve_lbfgs`, `_ridge.py:300`). `false` (default)
    /// uses the unconstrained Cholesky path, byte-identical to today.
    #[must_use]
    pub fn with_positive(mut self, positive: bool) -> Self {
        self.positive = positive;
        self
    }
}

impl<F: Float + Send + Sync + ScalarOperand + FromPrimitive + LinalgFloat + 'static> Ridge<F> {
    /// Fit the Ridge regression model with optional per-sample weights.
    ///
    /// Mirrors scikit-learn's `Ridge.fit(X, y, sample_weight=None)`
    /// (`sklearn/linear_model/_ridge.py`). When `sample_weight` is `Some(w)`,
    /// this solves the WEIGHTED ridge problem
    /// `min Σᵢ wᵢ (yᵢ − xᵢ·coef)² + alpha·‖coef‖²`:
    ///
    /// - `fit_intercept=true`: offsets are the WEIGHTED means
    ///   `x_off[j] = Σᵢ wᵢ·x[i,j] / Σwᵢ`, `y_off = Σᵢ wᵢ·yᵢ / Σwᵢ`. `X` and `y`
    ///   are centered by those offsets, each row is then rescaled by `√wᵢ`
    ///   (sklearn `_rescale_data`, `_ridge.py:682-688`), the cholesky ridge solve
    ///   runs on the rescaled design with the penalty `alpha` UNSCALED (because
    ///   `Xsᵀ·Xs == Xᵀ·W·X`), and `intercept = y_off − x_off·coef`.
    /// - `fit_intercept=false`: no centering; each row is rescaled by `√wᵢ`, the
    ///   ridge solve runs on the rescaled `X`/`y`, and `intercept = 0`.
    ///
    /// `sample_weight=None` is BYTE-IDENTICAL to [`Fit::fit`] (the unweighted
    /// centering + `linalg::solve_ridge` path), which delegates here. The
    /// `alpha=0` → OLS min-norm fallback is preserved (handled inside
    /// `linalg::solve_ridge`).
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of samples in `x` and
    /// `y` (or, when provided, `sample_weight`) differ.
    /// Returns [`FerroError::InvalidParameter`] if `alpha` is negative.
    /// Returns [`FerroError::InsufficientSamples`] if there are no samples.
    /// Returns [`FerroError::NumericalInstability`] if the weighted-offset
    /// denominator (`Σwᵢ`) cannot be formed or a mean cannot be computed.
    /// Solve the non-negative ridge problem
    /// `min 0.5·‖A·w − b‖² + 0.5·alpha·‖w‖²` subject to `w ≥ 0` via projected
    /// coordinate descent.
    ///
    /// Mirrors the unique optimum sklearn reaches with its L-BFGS-B solver
    /// (`_solve_lbfgs`, `sklearn/linear_model/_ridge.py:300`, objective
    /// `0.5·‖Xw−y‖² + 0.5·alpha·‖w‖²` with bounds `[(0, inf)]`). For a smooth
    /// strongly-convex QP with box constraints, coordinate descent with
    /// per-coordinate projection converges to that exact optimum.
    ///
    /// Returns `(coef, n_iter)`. A column with `‖A[:,j]‖² + alpha == 0` keeps
    /// its coordinate at zero (no division by zero).
    fn solve_nonneg_ridge(&self, a: &Array2<F>, b: &Array1<F>) -> (Array1<F>, usize) {
        let n_features = a.ncols();
        let alpha = self.alpha;
        let zero = <F as num_traits::Zero>::zero();

        // col_sq[j] = ‖A[:,j]‖²
        let mut col_sq = Array1::<F>::zeros(n_features);
        for j in 0..n_features {
            let col = a.column(j);
            col_sq[j] = col.dot(&col);
        }

        let mut w = Array1::<F>::zeros(n_features);
        // r = b − A·w  (= b initially since w = 0)
        let mut r = b.clone();

        let max_iter = self.max_iter.unwrap_or(1000);
        let tol = self.tol;

        let mut iters = 0;
        for _ in 0..max_iter {
            iters += 1;
            let mut max_change = zero;
            for j in 0..n_features {
                let col = a.column(j);
                let old = w[j];
                let denom = col_sq[j] + alpha;
                if denom <= zero {
                    // All-zero column with alpha == 0: coordinate stays 0.
                    continue;
                }
                // rho = A[:,j]ᵀ·r + col_sq[j]·old = A[:,j]ᵀ(b − A·w + A[:,j]·w[j])
                let rho = col.dot(&r) + col_sq[j] * old;
                let candidate = rho / denom;
                let new = if candidate > zero { candidate } else { zero };
                if new != old {
                    let delta = new - old;
                    // Incremental residual update: r -= A[:,j]·delta
                    r.scaled_add(-delta, &col);
                    w[j] = new;
                    let abs_change = <F as Float>::abs(delta);
                    if abs_change > max_change {
                        max_change = abs_change;
                    }
                }
            }
            if max_change < tol {
                break;
            }
        }

        (w, iters)
    }

    pub fn fit_with_sample_weight(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
        sample_weight: Option<&Array1<F>>,
    ) -> Result<FittedRidge<F>, FerroError> {
        let (n_samples, n_features) = x.dim();

        if n_samples != y.len() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_samples],
                actual: vec![y.len()],
                context: "y length must match number of samples in X".into(),
            });
        }

        // `<F as num_traits::Zero>::zero()`: the `LinalgFloat` bound pulls
        // `ferray::Element` (which also defines a `zero`) into scope, so a bare
        // `F::zero()` is ambiguous between `Element` and `num_traits::Zero`.
        if self.alpha < <F as num_traits::Zero>::zero() {
            return Err(FerroError::InvalidParameter {
                name: "alpha".into(),
                reason: "must be non-negative".into(),
            });
        }

        if n_samples == 0 {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "Ridge requires at least one sample".into(),
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
                    // Center the data to handle the intercept.
                    let x_mean =
                        x.mean_axis(Axis(0))
                            .ok_or_else(|| FerroError::NumericalInstability {
                                message: "failed to compute column means".into(),
                            })?;
                    let y_mean = y.mean().ok_or_else(|| FerroError::NumericalInstability {
                        message: "failed to compute target mean".into(),
                    })?;

                    let x_centered = x - &x_mean;
                    let y_centered = y - y_mean;

                    let (w, n_iter_) = if self.positive {
                        let (coef, iters) = self.solve_nonneg_ridge(&x_centered, &y_centered);
                        (coef, Some(iters))
                    } else {
                        (
                            linalg::solve_ridge(&x_centered, &y_centered, self.alpha)?,
                            None,
                        )
                    };
                    let intercept = y_mean - x_mean.dot(&w);

                    Ok(FittedRidge {
                        coefficients: w,
                        intercept,
                        n_iter_,
                    })
                } else {
                    let (w, n_iter_) = if self.positive {
                        let (coef, iters) = self.solve_nonneg_ridge(x, y);
                        (coef, Some(iters))
                    } else {
                        (linalg::solve_ridge(x, y, self.alpha)?, None)
                    };

                    Ok(FittedRidge {
                        coefficients: w,
                        // Disambiguate `Element::zero` vs `num_traits::Zero::zero`
                        // (both in scope under the `LinalgFloat` bound).
                        intercept: <F as num_traits::Zero>::zero(),
                        n_iter_,
                    })
                }
            }
            Some(w) => {
                // Per-row √w factor (sklearn `_rescale_data`, `_ridge.py:682-688`).
                let w_sqrt = w.mapv(<F as Float>::sqrt);

                if self.fit_intercept {
                    // WEIGHTED centering: offsets are the weighted means
                    // x_off[j] = Σ wᵢ x[i,j] / Σ wᵢ, y_off = Σ wᵢ yᵢ / Σ wᵢ
                    // (sklearn `_preprocess_data` weighted `_average`).
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

                    // Center, then row-rescale by √w. Penalty `alpha` is UNSCALED:
                    // (√w·Xc)ᵀ(√w·Xc) == Xcᵀ·W·Xc, so the closed form
                    // (Xsᵀ·Xs + alpha·I)·coef = Xsᵀ·ys IS the weighted ridge solve.
                    let x_centered = x - &x_off;
                    let y_centered = y - y_off;
                    let x_scaled = &x_centered * &w_sqrt.view().insert_axis(Axis(1));
                    let y_scaled = &y_centered * &w_sqrt;

                    let (coef, n_iter_) = if self.positive {
                        let (c, iters) = self.solve_nonneg_ridge(&x_scaled, &y_scaled);
                        (c, Some(iters))
                    } else {
                        (linalg::solve_ridge(&x_scaled, &y_scaled, self.alpha)?, None)
                    };
                    let intercept = y_off - x_off.dot(&coef);

                    Ok(FittedRidge {
                        coefficients: coef,
                        intercept,
                        n_iter_,
                    })
                } else {
                    // No centering; just √w row-rescaling, intercept 0.
                    let x_scaled = x * &w_sqrt.view().insert_axis(Axis(1));
                    let y_scaled = y * &w_sqrt;

                    let (coef, n_iter_) = if self.positive {
                        let (c, iters) = self.solve_nonneg_ridge(&x_scaled, &y_scaled);
                        (c, Some(iters))
                    } else {
                        (linalg::solve_ridge(&x_scaled, &y_scaled, self.alpha)?, None)
                    };

                    Ok(FittedRidge {
                        coefficients: coef,
                        intercept: <F as num_traits::Zero>::zero(),
                        n_iter_,
                    })
                }
            }
        }
    }
}

impl<F: Float> Default for Ridge<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted Ridge regression model.
///
/// Stores the learned coefficients and intercept. Implements [`Predict`]
/// to generate predictions and [`HasCoefficients`] for introspection.
#[derive(Debug, Clone)]
pub struct FittedRidge<F> {
    /// Learned coefficient vector (one per feature).
    coefficients: Array1<F>,
    /// Learned intercept (bias) term.
    intercept: F,
    /// Number of iterations run by an iterative solver, or `None` for direct
    /// solvers (sklearn `n_iter_`, `_ridge.py:994`). ferrolearn's Cholesky
    /// solver is direct (no iteration), so this is always `None` — matching
    /// sklearn's behaviour when `solver='cholesky'` or `solver='svd'` resolve
    /// the normal equations in closed form.
    n_iter_: Option<usize>,
}

impl<F> FittedRidge<F> {
    /// Return the number of iterations run by an iterative solver, or `None`
    /// for the direct Cholesky solver (sklearn `n_iter_`, `_ridge.py:994`).
    ///
    /// ferrolearn implements only the direct Cholesky path, so this is always
    /// `None`. When an iterative solver is added (future REQ-8 #386), it will
    /// return `Some(n)`.
    #[must_use]
    pub fn n_iter(&self) -> Option<usize> {
        self.n_iter_
    }
}

impl<F: Float + Send + Sync + ScalarOperand + FromPrimitive + LinalgFloat + 'static>
    Fit<Array2<F>, Array1<F>> for Ridge<F>
{
    type Fitted = FittedRidge<F>;
    type Error = FerroError;

    /// Fit the Ridge regression model using Cholesky decomposition.
    ///
    /// Solves `(X^T X + alpha * I)^{-1} X^T y`.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of samples in
    /// `x` and `y` differ.
    /// Returns [`FerroError::InvalidParameter`] if `alpha` is negative.
    fn fit(&self, x: &Array2<F>, y: &Array1<F>) -> Result<FittedRidge<F>, FerroError> {
        // Unweighted ridge is the `sample_weight=None` arm of the weighted fit;
        // delegating keeps the None path byte-identical to the historic body
        // (centering + `solve_ridge`), mirroring sklearn's single `fit` entry
        // (`_ridge.py`, `sample_weight=None` default).
        self.fit_with_sample_weight(x, y, None)
    }
}

impl<F: Float + Send + Sync + ScalarOperand + 'static> Predict<Array2<F>> for FittedRidge<F> {
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

impl<F: Float + Send + Sync + ScalarOperand + 'static> HasCoefficients<F> for FittedRidge<F> {
    fn coefficients(&self) -> &Array1<F> {
        &self.coefficients
    }

    fn intercept(&self) -> F {
        self.intercept
    }
}

/// Fitted multi-output Ridge regression model.
///
/// Companion to [`FittedRidge`] for the case where `Y` has multiple
/// target columns. Stores a `(n_features, n_targets)` coefficient matrix
/// and a per-target intercept vector. The Cholesky factor of
/// `X^T X + alpha * I` is computed once during [`Ridge::fit`] and shared
/// across all targets, so multi-output fitting costs the same `O(p^3)`
/// factorization as the single-output path.
#[derive(Debug, Clone)]
pub struct FittedRidgeMulti<F> {
    /// Learned coefficients, shape `(n_features, n_targets)`.
    coefficients: Array2<F>,
    /// Per-target intercept vector, length `n_targets`. Filled with
    /// zeros when `fit_intercept = false`.
    intercepts: Array1<F>,
}

impl<F: Float> FittedRidgeMulti<F> {
    /// Borrow the learned coefficient matrix `(n_features, n_targets)`.
    #[must_use]
    pub fn coefficients(&self) -> &Array2<F> {
        &self.coefficients
    }

    /// Borrow the per-target intercept vector `(n_targets,)`.
    #[must_use]
    pub fn intercepts(&self) -> &Array1<F> {
        &self.intercepts
    }
}

impl<F: Float + Send + Sync + ScalarOperand + FromPrimitive + 'static> Fit<Array2<F>, Array2<F>>
    for Ridge<F>
{
    type Fitted = FittedRidgeMulti<F>;
    type Error = FerroError;

    /// Fit the multi-output Ridge regression model using a single
    /// shared Cholesky factorization across all `Y` columns.
    ///
    /// Solves `(X^T X + alpha * I)^{-1} X^T Y` where `Y` is
    /// `(n_samples, n_targets)`.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of samples in
    /// `x` and `y` differ.
    /// Returns [`FerroError::InvalidParameter`] if `alpha` is negative.
    fn fit(&self, x: &Array2<F>, y: &Array2<F>) -> Result<FittedRidgeMulti<F>, FerroError> {
        let (n_samples, _n_features) = x.dim();

        if n_samples != y.nrows() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_samples],
                actual: vec![y.nrows()],
                context: "y rows must match number of samples in X".into(),
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
                context: "Ridge requires at least one sample".into(),
            });
        }

        let n_targets = y.ncols();

        if self.fit_intercept {
            // Center the data to handle the intercept (per-target).
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

            let x_centered = x - &x_mean;
            let y_centered = y - &y_mean;

            let w = linalg::solve_ridge_multi(&x_centered, &y_centered, self.alpha)?;
            // intercept[k] = y_mean[k] - x_mean · w[:, k]
            let mut intercepts = Array1::<F>::zeros(n_targets);
            for k in 0..n_targets {
                let col = w.column(k);
                let dot = x_mean.dot(&col);
                intercepts[k] = y_mean[k] - dot;
            }

            Ok(FittedRidgeMulti {
                coefficients: w,
                intercepts,
            })
        } else {
            let w = linalg::solve_ridge_multi(x, y, self.alpha)?;
            Ok(FittedRidgeMulti {
                coefficients: w,
                intercepts: Array1::<F>::zeros(n_targets),
            })
        }
    }
}

impl<F: Float + Send + Sync + ScalarOperand + 'static> Predict<Array2<F>> for FittedRidgeMulti<F> {
    type Output = Array2<F>;
    type Error = FerroError;

    /// Predict target values for the given feature matrix.
    ///
    /// Computes `X @ coefficients + intercepts` and returns an
    /// `(n_samples, n_targets)` array.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of features
    /// does not match the fitted model.
    fn predict(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let n_features = x.ncols();
        if n_features != self.coefficients.nrows() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![self.coefficients.nrows()],
                actual: vec![n_features],
                context: "number of features must match fitted model".into(),
            });
        }

        let mut preds = x.dot(&self.coefficients);
        // Broadcast-add per-target intercepts.
        for (k, &b) in self.intercepts.iter().enumerate() {
            let mut col = preds.column_mut(k);
            col.mapv_inplace(|v| v + b);
        }
        Ok(preds)
    }
}

// Pipeline integration.
impl<F> PipelineEstimator<F> for Ridge<F>
where
    F: Float + FromPrimitive + ScalarOperand + LinalgFloat + Send + Sync + 'static,
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

impl<F> FittedPipelineEstimator<F> for FittedRidge<F>
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
    fn test_ridge_no_regularization() {
        // With alpha=0, Ridge should behave like OLS.
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = array![3.0, 5.0, 7.0, 9.0, 11.0];

        let model = Ridge::<f64>::new().with_alpha(0.0);
        let fitted = model.fit(&x, &y).unwrap();

        assert_relative_eq!(fitted.coefficients()[0], 2.0, epsilon = 1e-8);
        assert_relative_eq!(fitted.intercept(), 1.0, epsilon = 1e-8);
    }

    #[test]
    fn test_ridge_shrinks_coefficients() {
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = array![3.0, 5.0, 7.0, 9.0, 11.0];

        let model_low = Ridge::<f64>::new().with_alpha(0.01);
        let model_high = Ridge::<f64>::new().with_alpha(100.0);

        let fitted_low = model_low.fit(&x, &y).unwrap();
        let fitted_high = model_high.fit(&x, &y).unwrap();

        // Higher alpha should shrink coefficients more.
        assert!(fitted_high.coefficients()[0].abs() < fitted_low.coefficients()[0].abs());
    }

    #[test]
    fn test_ridge_no_intercept() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![2.0, 4.0, 6.0, 8.0];

        let model = Ridge::<f64>::new()
            .with_alpha(0.0)
            .with_fit_intercept(false);
        let fitted = model.fit(&x, &y).unwrap();

        assert_relative_eq!(fitted.coefficients()[0], 2.0, epsilon = 1e-10);
        assert_relative_eq!(fitted.intercept(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_ridge_negative_alpha() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![1.0, 2.0, 3.0];

        let model = Ridge::<f64>::new().with_alpha(-1.0);
        let result = model.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_ridge_shape_mismatch() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![1.0, 2.0];

        let model = Ridge::<f64>::new();
        let result = model.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_ridge_predict() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0]).unwrap();
        let y = array![1.0, 2.0, 3.0, 6.0];

        let model = Ridge::<f64>::new().with_alpha(0.01);
        let fitted = model.fit(&x, &y).unwrap();

        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 4);
    }

    #[test]
    fn test_ridge_pipeline_integration() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![3.0, 5.0, 7.0, 9.0];

        let model = Ridge::<f64>::new();
        let fitted = model.fit_pipeline(&x, &y).unwrap();
        let preds = fitted.predict_pipeline(&x).unwrap();
        assert_eq!(preds.len(), 4);
    }

    #[test]
    fn test_ridge_has_coefficients() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = array![1.0, 2.0, 3.0];

        let model = Ridge::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();

        assert_eq!(fitted.coefficients().len(), 2);
    }

    #[test]
    fn ridge_fit_sample_weight_with_intercept_matches_sklearn() -> Result<(), FerroError> {
        // Live sklearn 1.5.2 oracle (WEIGHTED ridge, alpha=1, fit_intercept=True):
        //   cd /tmp && python3 -c "import numpy as np; \
        //     from sklearn.linear_model import Ridge; \
        //     X=np.array([[1.,2.],[2.,1.],[3.,4.],[4.,3.],[5.,5.]]); \
        //     y=np.array([3.0,2.5,7.1,6.0,11.2]); w=np.array([1.,4.,1.,1.,3.]); \
        //     m=Ridge(alpha=1.0).fit(X,y,sample_weight=w); \
        //     print([round(c,10) for c in m.coef_], round(m.intercept_,10))"
        //   -> [0.9233502538, 1.39678511] -0.8033840948
        let x = Array2::from_shape_vec((5, 2), vec![1., 2., 2., 1., 3., 4., 4., 3., 5., 5.])
            .map_err(|e| FerroError::ShapeMismatch {
                expected: vec![5, 2],
                actual: vec![],
                context: e.to_string(),
            })?;
        let y = array![3.0, 2.5, 7.1, 6.0, 11.2];
        let w = array![1.0, 4.0, 1.0, 1.0, 3.0];

        let model = Ridge::<f64>::new().with_alpha(1.0);
        let fitted = model.fit_with_sample_weight(&x, &y, Some(&w))?;

        assert_relative_eq!(fitted.coefficients()[0], 0.923_350_253_8, epsilon = 1e-7);
        assert_relative_eq!(fitted.coefficients()[1], 1.396_785_11, epsilon = 1e-7);
        assert_relative_eq!(fitted.intercept(), -0.803_384_094_8, epsilon = 1e-7);

        // Non-tautological: the weighted result MUST differ from the unweighted
        // fit (oracle unweighted coef_ [0.8228070175, 1.3561403509]).
        let unweighted = model.fit(&x, &y)?;
        assert_relative_eq!(
            unweighted.coefficients()[0],
            0.822_807_017_5,
            epsilon = 1e-7
        );
        assert_relative_eq!(
            unweighted.coefficients()[1],
            1.356_140_350_9,
            epsilon = 1e-7
        );
        assert!((fitted.coefficients()[0] - unweighted.coefficients()[0]).abs() > 1e-3);
        assert!((fitted.intercept() - unweighted.intercept()).abs() > 1e-3);
        Ok(())
    }

    #[test]
    fn ridge_fit_sample_weight_no_intercept_matches_sklearn() -> Result<(), FerroError> {
        // Live sklearn 1.5.2 oracle (WEIGHTED ridge, alpha=2, fit_intercept=False):
        //   cd /tmp && python3 -c "import numpy as np; \
        //     from sklearn.linear_model import Ridge; \
        //     X=np.array([[1.,2.],[2.,1.],[3.,4.],[4.,3.],[5.,5.]]); \
        //     y=np.array([3.0,2.5,7.1,6.0,11.2]); w=np.array([1.,4.,1.,1.,3.]); \
        //     m=Ridge(alpha=2.0,fit_intercept=False).fit(X,y,sample_weight=w); \
        //     print([round(c,10) for c in m.coef_])"
        //   -> [0.7273779983, 1.3737799835]
        let x = Array2::from_shape_vec((5, 2), vec![1., 2., 2., 1., 3., 4., 4., 3., 5., 5.])
            .map_err(|e| FerroError::ShapeMismatch {
                expected: vec![5, 2],
                actual: vec![],
                context: e.to_string(),
            })?;
        let y = array![3.0, 2.5, 7.1, 6.0, 11.2];
        let w = array![1.0, 4.0, 1.0, 1.0, 3.0];

        let model = Ridge::<f64>::new()
            .with_alpha(2.0)
            .with_fit_intercept(false);
        let fitted = model.fit_with_sample_weight(&x, &y, Some(&w))?;

        assert_relative_eq!(fitted.coefficients()[0], 0.727_377_998_3, epsilon = 1e-7);
        assert_relative_eq!(fitted.coefficients()[1], 1.373_779_983_5, epsilon = 1e-7);
        assert_eq!(fitted.intercept(), 0.0);
        Ok(())
    }

    #[test]
    fn ridge_fit_none_sample_weight_equals_unweighted() -> Result<(), FerroError> {
        // Regression guard: the `None` path is BYTE-IDENTICAL to `fit`.
        let x = Array2::from_shape_vec((5, 2), vec![1., 2., 2., 1., 3., 4., 4., 3., 5., 5.])
            .map_err(|e| FerroError::ShapeMismatch {
                expected: vec![5, 2],
                actual: vec![],
                context: e.to_string(),
            })?;
        let y = array![3.0, 2.5, 7.1, 6.0, 11.2];

        let model = Ridge::<f64>::new().with_alpha(1.0);
        let via_fit = model.fit(&x, &y)?;
        let via_none = model.fit_with_sample_weight(&x, &y, None)?;

        assert_eq!(
            via_fit.coefficients()[0].to_bits(),
            via_none.coefficients()[0].to_bits()
        );
        assert_eq!(
            via_fit.coefficients()[1].to_bits(),
            via_none.coefficients()[1].to_bits()
        );
        assert_eq!(
            via_fit.intercept().to_bits(),
            via_none.intercept().to_bits()
        );

        // Same for fit_intercept=false.
        let model_ni = Ridge::<f64>::new()
            .with_alpha(1.0)
            .with_fit_intercept(false);
        let via_fit_ni = model_ni.fit(&x, &y)?;
        let via_none_ni = model_ni.fit_with_sample_weight(&x, &y, None)?;
        assert_eq!(
            via_fit_ni.coefficients()[0].to_bits(),
            via_none_ni.coefficients()[0].to_bits()
        );
        assert_eq!(
            via_fit_ni.intercept().to_bits(),
            via_none_ni.intercept().to_bits()
        );
        Ok(())
    }

    // -- copy_x / random_state ABI-parity ----------------------------------

    #[test]
    fn ridge_copy_x_random_state_defaults_and_builders() -> Result<(), FerroError> {
        // Defaults mirror sklearn Ridge(copy_X=True, random_state=None)
        // (`sklearn/linear_model/_ridge.py:898`/`:903`).
        assert!(Ridge::<f64>::new().copy_x, "copy_x default must be true");
        assert_eq!(
            Ridge::<f64>::new().random_state,
            None,
            "random_state default must be None"
        );

        // Builders store the supplied value.
        assert!(!Ridge::<f64>::new().with_copy_x(false).copy_x);
        assert!(Ridge::<f64>::new().with_copy_x(true).copy_x);
        assert_eq!(
            Ridge::<f64>::new().with_random_state(Some(42)).random_state,
            Some(42)
        );
        assert_eq!(
            Ridge::<f64>::new().with_random_state(None).random_state,
            None
        );

        // No behavior change: fit produces byte-identical coef_/intercept_
        // regardless of copy_x or random_state (deterministic Cholesky
        // solver is unaffected by either param).
        //
        // Live sklearn oracle (alpha=1.0, fit_intercept=true):
        //   python3 -c "import numpy as np; from sklearn.linear_model import Ridge;
        //     X=np.array([[1.,2.],[3.,4.],[5.,6.],[7.,8.],[9.,10.]]);
        //     y=np.array([1.,2.,3.,4.,5.]);
        //     m=Ridge(alpha=1.0).fit(X,y); print(m.coef_.tolist(), m.intercept_)"
        //   -> [0.07692307692307693, 0.4230769230769231] 0.0
        let x = Array2::from_shape_vec((5, 2), vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10.])
            .map_err(|e| FerroError::ShapeMismatch {
                expected: vec![5, 2],
                actual: vec![],
                context: e.to_string(),
            })?;
        let y = array![1., 2., 3., 4., 5.];

        let ref_fitted = Ridge::<f64>::new().with_alpha(1.0).fit(&x, &y)?;
        let copy_false = Ridge::<f64>::new()
            .with_alpha(1.0)
            .with_copy_x(false)
            .fit(&x, &y)?;
        let rs_some = Ridge::<f64>::new()
            .with_alpha(1.0)
            .with_random_state(Some(42))
            .fit(&x, &y)?;

        // Byte-identical (deterministic Cholesky, no mutation).
        assert_eq!(
            ref_fitted.coefficients()[0].to_bits(),
            copy_false.coefficients()[0].to_bits()
        );
        assert_eq!(
            ref_fitted.coefficients()[1].to_bits(),
            copy_false.coefficients()[1].to_bits()
        );
        assert_eq!(
            ref_fitted.intercept().to_bits(),
            copy_false.intercept().to_bits()
        );
        assert_eq!(
            ref_fitted.coefficients()[0].to_bits(),
            rs_some.coefficients()[0].to_bits()
        );
        assert_eq!(
            ref_fitted.coefficients()[1].to_bits(),
            rs_some.coefficients()[1].to_bits()
        );
        assert_eq!(
            ref_fitted.intercept().to_bits(),
            rs_some.intercept().to_bits()
        );
        Ok(())
    }

    // -- Multi-output Ridge -------------------------------------------------

    #[test]
    fn test_ridge_multi_recovers_two_targets_with_zero_alpha() {
        // Two synthetic targets sharing the same features:
        //   y1 = 2*x1 - 3*x2 + 5
        //   y2 = 3*x1 +   x2 - 1
        let n = 50;
        let x_data: Vec<f64> = (0..n)
            .flat_map(|i| {
                let i = i as f64;
                [i / 10.0, (i / 7.0).sin()]
            })
            .collect();
        let x = Array2::from_shape_vec((n, 2), x_data).unwrap();
        let y_data: Vec<f64> = (0..n)
            .flat_map(|i| {
                let x1 = i as f64 / 10.0;
                let x2 = (i as f64 / 7.0).sin();
                [2.0 * x1 - 3.0 * x2 + 5.0, 3.0 * x1 + x2 - 1.0]
            })
            .collect();
        let y = Array2::from_shape_vec((n, 2), y_data).unwrap();

        let model = Ridge::<f64>::new().with_alpha(1e-8);
        let fitted: FittedRidgeMulti<f64> = model.fit(&x, &y).unwrap();

        let coef = fitted.coefficients();
        assert_eq!(coef.shape(), &[2, 2]);
        // Target 0
        assert_relative_eq!(coef[[0, 0]], 2.0, epsilon = 1e-4);
        assert_relative_eq!(coef[[1, 0]], -3.0, epsilon = 1e-4);
        // Target 1
        assert_relative_eq!(coef[[0, 1]], 3.0, epsilon = 1e-4);
        assert_relative_eq!(coef[[1, 1]], 1.0, epsilon = 1e-4);
        // Intercepts
        assert_relative_eq!(fitted.intercepts()[0], 5.0, epsilon = 1e-4);
        assert_relative_eq!(fitted.intercepts()[1], -1.0, epsilon = 1e-4);
    }

    #[test]
    fn test_ridge_multi_no_intercept() {
        // y = X @ B with no bias; verify intercepts come out zero and
        // coefficients match the OLS solve.
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y =
            Array2::from_shape_vec((4, 2), vec![2.0, 4.0, 4.0, 8.0, 6.0, 12.0, 8.0, 16.0]).unwrap();

        let model = Ridge::<f64>::new()
            .with_alpha(0.0)
            .with_fit_intercept(false);
        let fitted = model.fit(&x, &y).unwrap();

        // y[:, 0] = 2*x, y[:, 1] = 4*x
        assert_relative_eq!(fitted.coefficients()[[0, 0]], 2.0, epsilon = 1e-8);
        assert_relative_eq!(fitted.coefficients()[[0, 1]], 4.0, epsilon = 1e-8);
        assert_relative_eq!(fitted.intercepts()[0], 0.0, epsilon = 1e-12);
        assert_relative_eq!(fitted.intercepts()[1], 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_ridge_multi_shrinks_with_large_alpha() {
        // Heavy regularization should pull all targets toward zero
        // coefficients (intercepts may still recover the means).
        let n = 20;
        let x = Array2::from_shape_vec((n, 1), (0..n).map(|i| i as f64).collect()).unwrap();
        let y_data: Vec<f64> = (0..n)
            .flat_map(|i| [(i as f64) * 10.0, (i as f64) * 5.0])
            .collect();
        let y = Array2::from_shape_vec((n, 2), y_data).unwrap();

        let model = Ridge::<f64>::new()
            .with_alpha(1e6)
            .with_fit_intercept(false);
        let fitted = model.fit(&x, &y).unwrap();

        assert!(fitted.coefficients()[[0, 0]].abs() < 1.0);
        assert!(fitted.coefficients()[[0, 1]].abs() < 1.0);
    }

    #[test]
    fn test_ridge_multi_predict_round_trips_training_data() {
        // Verify Fit→Predict round-trip on the training data: the model
        // should reproduce y up to the regularization-induced bias.
        let n = 40;
        let x_data: Vec<f64> = (0..n)
            .flat_map(|i| [(i as f64) / 10.0, ((i as f64) / 3.0).cos()])
            .collect();
        let x = Array2::from_shape_vec((n, 2), x_data).unwrap();
        let y_data: Vec<f64> = (0..n)
            .flat_map(|i| {
                let x1 = (i as f64) / 10.0;
                let x2 = ((i as f64) / 3.0).cos();
                [1.5 * x1 + 0.7 * x2 + 0.2, -0.5 * x1 + 2.0 * x2 - 1.0]
            })
            .collect();
        let y = Array2::from_shape_vec((n, 2), y_data).unwrap();

        let model = Ridge::<f64>::new().with_alpha(1e-6);
        let fitted = model.fit(&x, &y).unwrap();
        let y_hat = fitted.predict(&x).unwrap();

        assert_eq!(y_hat.shape(), y.shape());
        // Maximum element-wise error across both targets.
        let max_err = y_hat
            .iter()
            .zip(y.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(max_err < 1e-3, "max_err = {max_err}");
    }

    #[test]
    fn test_ridge_multi_shape_mismatch() {
        // 5-sample X but 3-sample Y — should error.
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let model = Ridge::<f64>::new();
        let err = <Ridge<f64> as Fit<Array2<f64>, Array2<f64>>>::fit(&model, &x, &y).unwrap_err();
        matches!(err, FerroError::ShapeMismatch { .. });
    }

    #[test]
    fn test_ridge_multi_single_target_matches_single_output_path() {
        // The same data routed through the single-output and the
        // multi-output Fit impls should produce coefficients that agree
        // to within numerical noise. This pins the parallel paths from
        // drifting apart over time.
        let n = 30;
        let x = Array2::from_shape_vec(
            (n, 3),
            (0..n)
                .flat_map(|i| {
                    let i = i as f64;
                    [i / 10.0, (i / 5.0).sin(), (i / 11.0).cos()]
                })
                .collect(),
        )
        .unwrap();
        let y_1d: Array1<f64> = x
            .rows()
            .into_iter()
            .map(|r| 2.0 * r[0] - r[1] + 0.5 * r[2] + 3.0)
            .collect();
        let y_2d = Array2::from_shape_vec((n, 1), y_1d.to_vec()).unwrap();

        let model = Ridge::<f64>::new().with_alpha(0.1);
        let single: FittedRidge<f64> = model.fit(&x, &y_1d).unwrap();
        let multi: FittedRidgeMulti<f64> = model.fit(&x, &y_2d).unwrap();

        // Coefficients agree element-wise.
        for j in 0..3 {
            assert_relative_eq!(
                single.coefficients()[j],
                multi.coefficients()[[j, 0]],
                epsilon = 1e-10
            );
        }
        // Intercepts agree.
        assert_relative_eq!(single.intercept(), multi.intercepts()[0], epsilon = 1e-10);
    }

    // -- max_iter / tol / n_iter_ ABI-parity (REQ-10) ----------------------

    #[test]
    fn ridge_max_iter_tol_niter_defaults_and_builders() -> Result<(), FerroError> {
        // Defaults mirror sklearn Ridge(max_iter=None, tol=1e-4)
        // (`sklearn/linear_model/_ridge.py:899-900`).
        assert_eq!(
            Ridge::<f64>::new().max_iter,
            None,
            "max_iter default must be None"
        );
        assert!(
            (Ridge::<f64>::new().tol - 1e-4_f64).abs() < 1e-12,
            "tol default must be 1e-4"
        );

        // Builders store the supplied value.
        assert_eq!(
            Ridge::<f64>::new().with_max_iter(Some(500)).max_iter,
            Some(500)
        );
        assert_eq!(Ridge::<f64>::new().with_max_iter(None).max_iter, None);
        assert!(
            (Ridge::<f64>::new().with_tol(1e-6_f64).tol - 1e-6_f64).abs() < 1e-18,
            "with_tol must store the supplied value"
        );

        // n_iter_ is always None for the direct Cholesky solver, matching
        // sklearn: Ridge(alpha=1.0).fit(X,y).n_iter_ is None when
        // solver='auto' resolves to 'cholesky' (`_ridge.py:994`).
        let x = Array2::from_shape_vec((4, 2), vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0])
            .map_err(|e| FerroError::ShapeMismatch {
                expected: vec![4, 2],
                actual: vec![],
                context: e.to_string(),
            })?;
        let y = array![1.0, 2.0, 3.0, 6.0];

        let fitted = Ridge::<f64>::new().fit(&x, &y)?;
        assert_eq!(
            fitted.n_iter(),
            None,
            "n_iter_ must be None for direct Cholesky"
        );

        // No behavior change: fit produces byte-identical coef_/intercept_
        // regardless of max_iter or tol (direct Cholesky is unaffected).
        //
        // Live sklearn oracle (alpha=1.0, fit_intercept=true):
        //   python3 -c "import numpy as np; from sklearn.linear_model import Ridge;
        //     X=np.array([[1.,0.],[0.,1.],[1.,1.],[2.,2.]]);
        //     y=np.array([1.,2.,3.,6.]);
        //     m=Ridge(alpha=1.0).fit(X,y); print(m.coef_.tolist(), m.intercept_)"
        //   -> [0.875, 1.375] 0.75
        let ref_fitted = Ridge::<f64>::new().with_alpha(1.0).fit(&x, &y)?;
        let with_max_iter = Ridge::<f64>::new()
            .with_alpha(1.0)
            .with_max_iter(Some(500))
            .fit(&x, &y)?;
        let with_tol = Ridge::<f64>::new()
            .with_alpha(1.0)
            .with_tol(1e-6_f64)
            .fit(&x, &y)?;

        // Byte-identical (deterministic Cholesky, max_iter/tol are no-ops).
        for j in 0..2 {
            assert_eq!(
                ref_fitted.coefficients()[j].to_bits(),
                with_max_iter.coefficients()[j].to_bits(),
                "coef_[{j}] must be byte-identical regardless of max_iter"
            );
            assert_eq!(
                ref_fitted.coefficients()[j].to_bits(),
                with_tol.coefficients()[j].to_bits(),
                "coef_[{j}] must be byte-identical regardless of tol"
            );
        }
        assert_eq!(
            ref_fitted.intercept().to_bits(),
            with_max_iter.intercept().to_bits()
        );
        assert_eq!(
            ref_fitted.intercept().to_bits(),
            with_tol.intercept().to_bits()
        );
        assert_eq!(with_max_iter.n_iter(), None);
        assert_eq!(with_tol.n_iter(), None);

        Ok(())
    }

    // -- positive=True non-negative ridge (REQ-9) --------------------------

    #[test]
    fn ridge_positive_matches_sklearn() -> Result<(), FerroError> {
        // Live sklearn 1.5.2 oracle (non-negative ridge, alpha=1, positive=True):
        //   cd /tmp && python3 -c "import numpy as np; \
        //     from sklearn.linear_model import Ridge; \
        //     X=np.array([[1.,3.],[2.,1.],[3.,4.],[4.,2.],[5.,5.],[6.,1.],[2.,4.],[5.,2.]]); \
        //     y=1.0*X[:,0]-2.0*X[:,1]+np.array([0.1,-0.2,0.15,0.0,-0.1,0.05,0.2,-0.05]); \
        //     m=Ridge(alpha=1.0,positive=True).fit(X,y); \
        //     print([round(c,8) for c in m.coef_], round(m.intercept_,8))"
        //   -> [1.19891304, 0.0] -6.17744565
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                1., 3., 2., 1., 3., 4., 4., 2., 5., 5., 6., 1., 2., 4., 5., 2.,
            ],
        )
        .map_err(|e| FerroError::ShapeMismatch {
            expected: vec![8, 2],
            actual: vec![],
            context: e.to_string(),
        })?;
        let raw = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 2.0, 5.0];
        let f1 = [3.0, 1.0, 4.0, 2.0, 5.0, 1.0, 4.0, 2.0];
        let noise = [0.1, -0.2, 0.15, 0.0, -0.1, 0.05, 0.2, -0.05];
        let y: Array1<f64> = (0..8).map(|i| raw[i] - 2.0 * f1[i] + noise[i]).collect();

        let model = Ridge::<f64>::new().with_alpha(1.0).with_positive(true);
        let fitted = model.fit(&x, &y)?;

        assert_relative_eq!(fitted.coefficients()[0], 1.198_913_04, epsilon = 1e-5);
        assert_relative_eq!(fitted.coefficients()[1], 0.0, epsilon = 1e-5);
        assert_relative_eq!(fitted.intercept(), -6.177_445_65, epsilon = 1e-4);

        // All coefficients are non-negative.
        for &c in fitted.coefficients().iter() {
            assert!(
                c >= 0.0,
                "coef {c} must be non-negative under positive=True"
            );
        }

        // Non-tautological: the constrained fit MUST differ from the
        // unconstrained ridge (oracle unconstrained coef_
        // [0.95708502, -1.85401484], intercept -0.23250675 — feature 1
        // is strongly negative and clamps to 0).
        let unconstrained = Ridge::<f64>::new().with_alpha(1.0).fit(&x, &y)?;
        assert_relative_eq!(
            unconstrained.coefficients()[0],
            0.957_085_02,
            epsilon = 1e-5
        );
        assert_relative_eq!(
            unconstrained.coefficients()[1],
            -1.854_014_84,
            epsilon = 1e-5
        );
        assert!(
            (fitted.coefficients()[1] - unconstrained.coefficients()[1]).abs() > 1e-2,
            "positive fit must differ from unconstrained on the clamped coordinate"
        );
        Ok(())
    }

    #[test]
    fn ridge_positive_false_unchanged() -> Result<(), FerroError> {
        // Regression guard: positive=false (default) is BYTE-IDENTICAL to the
        // current unconstrained Cholesky `fit`.
        let x = Array2::from_shape_vec((5, 2), vec![1., 2., 2., 1., 3., 4., 4., 3., 5., 5.])
            .map_err(|e| FerroError::ShapeMismatch {
                expected: vec![5, 2],
                actual: vec![],
                context: e.to_string(),
            })?;
        let y = array![3.0, 2.5, 7.1, 6.0, 11.2];

        let model = Ridge::<f64>::new().with_alpha(1.0);
        let baseline = model.fit(&x, &y)?;
        let explicit_false = Ridge::<f64>::new()
            .with_alpha(1.0)
            .with_positive(false)
            .fit(&x, &y)?;

        for j in 0..2 {
            assert_eq!(
                baseline.coefficients()[j].to_bits(),
                explicit_false.coefficients()[j].to_bits(),
                "coef_[{j}] must be byte-identical with positive=false"
            );
        }
        assert_eq!(
            baseline.intercept().to_bits(),
            explicit_false.intercept().to_bits()
        );
        // positive=false keeps the direct-solver n_iter_ = None contract.
        assert_eq!(explicit_false.n_iter(), None);
        Ok(())
    }

    #[test]
    fn ridge_positive_all_nonneg_equals_unconstrained() -> Result<(), FerroError> {
        // Sanity: data whose unconstrained ridge is already non-negative —
        // the box constraint is inactive, so positive=True must reproduce the
        // unconstrained coefficients.
        //
        // Live sklearn 1.5.2 oracle:
        //   cd /tmp && python3 -c "import numpy as np; \
        //     from sklearn.linear_model import Ridge; \
        //     X=np.array([[1.,2.],[2.,1.],[3.,4.],[4.,3.],[5.,5.]]); \
        //     y=np.array([3.0,2.5,7.1,6.0,11.2]); \
        //     u=Ridge(alpha=1.0).fit(X,y); p=Ridge(alpha=1.0,positive=True).fit(X,y); \
        //     print([round(c,8) for c in u.coef_], [round(c,8) for c in p.coef_])"
        //   -> [0.82280702, 1.35614035] [0.82280702, 1.35614035]
        let x = Array2::from_shape_vec((5, 2), vec![1., 2., 2., 1., 3., 4., 4., 3., 5., 5.])
            .map_err(|e| FerroError::ShapeMismatch {
                expected: vec![5, 2],
                actual: vec![],
                context: e.to_string(),
            })?;
        let y = array![3.0, 2.5, 7.1, 6.0, 11.2];

        let unconstrained = Ridge::<f64>::new().with_alpha(1.0).fit(&x, &y)?;
        let positive = Ridge::<f64>::new()
            .with_alpha(1.0)
            .with_positive(true)
            .fit(&x, &y)?;

        for j in 0..2 {
            assert!(unconstrained.coefficients()[j] >= 0.0);
            // CD converges to the unconstrained optimum at the default
            // tol=1e-4; sklearn's own L-BFGS-B agrees only to ~1e-6 on this
            // fixture (oracle: 0.82280707 vs 0.82280702).
            assert_relative_eq!(
                positive.coefficients()[j],
                unconstrained.coefficients()[j],
                epsilon = 1e-4
            );
        }
        assert_relative_eq!(
            positive.intercept(),
            unconstrained.intercept(),
            epsilon = 1e-4
        );
        Ok(())
    }
}
