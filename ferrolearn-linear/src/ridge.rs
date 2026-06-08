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
//! | REQ-6 (multi-output 2-D Y → 2-D coef_) | SHIPPED | `FittedRidgeMulti<F>` + `Fit<Array2<F>, Array2<F>> for Ridge` (per-target `solve_ridge`, `coefficients()` `(n_features, n_targets)` / `intercepts()` `(n_targets,)`), mirroring sklearn `Ridge` 2-D `coef_` `(n_targets, n_features)` / `intercept_` `(n_targets,)` (`_ridge.py:543`/`:550`). Non-test consumer (R-HONEST-4 — prior NOT-STARTED note predates the multi-output Python binding #29): `RsRidgeMultiOutput in ferrolearn-python/src/regressors.rs` (field `Option<FittedRidgeMulti<f64>>`, transposes `coefficients()` to `(n_targets, n_features)`); `_regressors.py::Ridge.fit` routes the `y.ndim == 2 && y.shape[1] > 1` path to `_RsRidgeMultiOutput`. Verified: `tests/divergence_regressors.py::test_ridge_multioutput_matches_sklearn` (predict/coef ≤ 1e-8), `ferrolearn.Ridge(alpha=a).fit(X, Y_2d)` matches sklearn ≤ 1e-16. |
//! | REQ-7 (per-target alpha array) | SHIPPED | `Ridge<F>` adds `pub alpha_per_target: Option<Array1<F>>` (default `None`) + `with_alpha_per_target(Array1<F>)` builder. On the multi-output `Fit<Array2, Array2>` path, when `Some(alphas)` it validates `alphas.len() == n_targets` (else `ShapeMismatch`) and each `alphas[k] >= 0` (else `InvalidParameter`), then solves each target column `k` independently with its own penalty via `linalg::solve_ridge` on the SAME centered (fit_intercept) / raw design the scalar path uses — mathematically identical to an independent scalar-`alpha` Ridge per column, mirroring sklearn's array-valued `alpha` (`_ridge.py:701-712`). `None` is byte-identical to the historic scalar `solve_ridge_multi` path. Oracle tests: `ridge_per_target_alpha_matches_sklearn` (alpha `[0.5, 2.0]`, coef col0 `[0.79891892, 1.43891892]`/col1 `[0.78, 0.355]`, intercepts `[-0.75351351, -0.065]`), `ridge_per_target_alpha_equals_independent_scalar_fits`, `ridge_per_target_alpha_length_mismatch_errors`, `ridge_multi_scalar_alpha_unchanged` (regression guard). Closes #385. |
//! | REQ-8a (dense solver variants auto/cholesky/svd + solver_) | SHIPPED | `pub enum RidgeSolver { #[default] Auto, Cholesky, Svd }` + `Ridge<F>` gains `pub solver: RidgeSolver` (default `Auto`) + `with_solver`; `FittedRidge<F>` gains `solver_` + `pub fn solver()`. The single-output `fit_with_sample_weight` resolves `Auto`→`Cholesky` (`resolve_solver`, `_ridge.py:830`) and dispatches the unconstrained dense solve via `solve_unconstrained`: `Svd` → `linalg::solve_ridge_svd` (`coef = V·diag(sᵢ/(sᵢ²+alpha))·Uᵀy` from the thin SVD via `ferray::linalg::svd`, the analog of sklearn `_solve_svd` `_ridge.py:200-216`); `Cholesky`/`Auto` → the unchanged `linalg::solve_ridge` (byte-identical). All dense solvers yield the IDENTICAL unique solution (strictly convex). `solver_` stores the resolved value. Governs only the single-output unconstrained dense path. Consumer: `Fit<Array2, Array1>::fit for Ridge` (production: `RsRidge::fit` in `ferrolearn-python/src/regressors.rs`; `ridge_cv.rs`). Oracle tests `ridge_solver_svd_matches_sklearn_and_cholesky` (coef `[0.8228070175, 1.3561403509]`, intercept `-0.5768421053`), `ridge_solver_resolution`, `ridge_solver_default_cholesky_unchanged`, `ridge_solver_svd_no_intercept`. Closes #386 (8a). |
//! | REQ-8b (iterative solver variants lsqr/sparse_cg/sag/saga/lbfgs) | NOT-STARTED | #386 — needs iterative/SGD substrate + RNG. Not represented as `RidgeSolver` variants until the substrate lands. |
//! | REQ-9 (positive=True) | SHIPPED | `Ridge<F>` adds `pub positive: bool` (default `false`, `_ridge.py:902`/`:911`) + `with_positive(bool)` builder. When `self.positive`, `fit_with_sample_weight` routes the coefficient solve through `solve_nonneg_ridge`, which first computes the UNCONSTRAINED Cholesky optimum (`crate::linalg::solve_ridge`); if that solve succeeds AND all components are `≥ 0` the positivity constraint is INACTIVE and (by strict convexity + KKT) that feasible unconstrained minimizer IS the constrained minimizer, so it is returned EXACTLY as `(w_unc, 0)` (the exact closed-form Cholesky optimum, `n_iter_ = Some(0)`). Otherwise (solve errored, or any component `< 0` → constraint active) it falls through to projected coordinate descent minimizing `0.5·‖A·w−b‖² + 0.5·alpha·‖w‖²` s.t. `w ≥ 0` (`new = max(0, (A[:,j]ᵀr + col_sq[j]·old)/(col_sq[j] + alpha))`, incremental residual update, `max_iter=self.max_iter.unwrap_or(1000)`/`self.tol`). Either way it runs on the SAME centered (and `√w`-rescaled) design `solve_ridge` uses, then recovers `intercept = y_off − x_off·coef` (fit_intercept) / `0` identically. R-DEV-6: sklearn's default positive-Ridge L-BFGS-B (`_solve_lbfgs`, `_ridge.py:300`, objective `0.5·‖Xw−y‖²+0.5·alpha·‖w‖²`, bounds `[(0,inf)]`, dispatched at `_ridge.py:923-928`) runs at `tol=1e-4` and UNDER-CONVERGES on interior optima by ~1e-4; at `tol=1e-12` it reaches the true optimum = ferrolearn's exact Cholesky optimum (to ~1e-11). ferrolearn therefore ships the TRUE non-negative-ridge optimum, matching a CONVERGED sklearn exactly and being at least as precise as the default. `positive=false` (default) is byte-identical to the unconstrained Cholesky path. Oracle tests: `ridge_positive_matches_sklearn` (active constraint, alpha=1, fit_intercept coef `[1.19891304, 0.0]`, intercept `-6.17744565`, all ≥ 0, differs from unconstrained `[0.95708502, -1.85401484]`), `ridge_positive_false_unchanged` (byte-identical guard), `ridge_positive_all_nonneg_equals_unconstrained` (inactive-constraint sanity); divergence suite `divergence_ridge_positive_interior` (interior optimum == converged sklearn tol=1e-12 == exact unconstrained Cholesky). Closes #387, #2131. |
//! | REQ-10 (max_iter/tol + n_iter_) | SHIPPED | `Ridge<F>` adds `pub max_iter: Option<usize>` (default `None`) and `pub tol: F` (default `1e-4`) with `with_max_iter`/`with_tol` builders. `FittedRidge<F>` adds `n_iter_: Option<usize>` (always `None` for the direct Cholesky solver) with `pub fn n_iter(&self) -> Option<usize>`. Mirrors sklearn ctor `max_iter=None, tol=1e-4` (`_ridge.py:899-900`) and `n_iter_` set at `_ridge.py:994`; `max_iter`/`tol` are no-ops for the direct solver (closed-form normal equations, no iteration) — matching sklearn's direct `cholesky`/`svd` paths which also yield `n_iter_=None`. Test: `ridge_max_iter_tol_niter_defaults_and_builders`. Closes #388. |
//! | REQ-11 (sample_weight) | SHIPPED | `Ridge::fit_with_sample_weight(x, y, sample_weight: Option<&Array1<F>>)` solves WEIGHTED ridge `min Σᵢ wᵢ(yᵢ−xᵢ·coef)² + alpha·‖coef‖²`: weighted offsets `x_off[j]=Σwᵢx[i,j]/Σwᵢ`, `y_off=Σwᵢyᵢ/Σwᵢ` (fit_intercept), centering, then `√wᵢ` row-rescaling (`_rescale_data`, `_ridge.py:682-688`), `linalg::solve_ridge(&Xs, &ys, alpha)` with the penalty `alpha` UNSCALED (since `Xsᵀ·Xs == Xᵀ·W·X`), `intercept = y_off − x_off·coef`; `fit_intercept=false` skips centering (raw `√w`-rescale, intercept 0). `Fit::fit` delegates `fit_with_sample_weight(x, y, None)` (None byte-identical to the historic centering + `solve_ridge` body; alpha=0 OLS min-norm fallback preserved). Oracle tests `ridge_fit_sample_weight_with_intercept_matches_sklearn` (alpha=1 coef `[0.9233502538, 1.39678511]`, intercept `-0.8033840948`, differs from unweighted `[0.8228070175, 1.3561403509]`), `ridge_fit_sample_weight_no_intercept_matches_sklearn` (alpha=2 coef `[0.7273779983, 1.3737799835]`, intercept 0), `ridge_fit_none_sample_weight_equals_unweighted` (byte-identical guard). Closes #389. |
//! | REQ-12 (copy_X/random_state) | SHIPPED | `Ridge<F>` adds `pub copy_x: bool` (default `true`) and `pub random_state: Option<u64>` (default `None`) fields with `with_copy_x`/`with_random_state` builders. `copy_x` ABI-only (fit never mutates `x`); `random_state` stored-but-no-op for the deterministic Cholesky solver (only `sag`/`saga` use it, `_ridge.py:898`/`:903`). Test: `ridge_copy_x_random_state_defaults_and_builders`. Closes #390. |
//! | REQ-13 (ferray substrate) | NOT-STARTED | #391 (alpha=0 fallback already on ferray::linalg::lstsq; coef return tied to #359). |
//! | REQ-14 (non-finite input rejected) | SHIPPED | Both fit entries reject any NaN/+/-inf in X, y, or `sample_weight` BEFORE centering/solve with `FerroError::InvalidParameter`, mirroring sklearn's `_validate_data(force_all_finite=True)` (`_ridge.py:1242`) + `_check_sample_weight` (default `force_all_finite=True`) → `ValueError("Input X contains NaN.")` / `"... contains infinity ..."`. The single-output `fit_with_sample_weight` (shared entry `Fit::fit` delegates to) checks X/y/sample_weight; the SEPARATE multi-output arm `Fit<Array2, Array2>::fit` checks X/y independently. `.iter().any(|v| !v.is_finite())` catches both NaN and Inf; the finite path is byte-identical (the guard never fires on finite input). Verified vs the live sklearn 1.5.2 oracle (R-CHAR-3): `Ridge().fit` raises `ValueError` for NaN/+inf/-inf in X, NaN/inf in y, and NaN/inf in sample_weight (`tests/divergence_linear_nonfinite_batch2.rs::ridge_*`). Non-test consumer: the existing `Fit::fit` / `RsRidge` consumers. (#2259) |
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

/// Solver used for the dense single-output Ridge coefficient solve.
///
/// Mirrors a (dense) subset of scikit-learn's `solver` parameter
/// (`sklearn/linear_model/_ridge.py:830` `resolve_solver`). Only the
/// closed-form dense solvers are implemented here; the iterative solvers
/// (`lsqr`/`sparse_cg`/`sag`/`saga`/`lbfgs`) are tracked separately as
/// REQ-8b (blocker #386) and are intentionally NOT represented as variants
/// until their iterative/SGD substrate lands.
///
/// Every dense solver returns the IDENTICAL ridge solution — the problem is
/// strictly convex (`alpha > 0`, or `alpha = 0` with full-rank `X`), so the
/// minimizer is unique. The variant therefore only governs *how* the unique
/// solution is computed, not *what* it is.
///
/// This governs only the single-output (`Fit<Array2, Array1>`) dense
/// unconstrained solve. The multi-output (`Fit<Array2, Array2>`) path and the
/// `positive=true` constrained path are unaffected by this setting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RidgeSolver {
    /// `'auto'` — resolve automatically. For the dense path this resolves to
    /// [`Cholesky`](RidgeSolver::Cholesky) (sklearn `resolve_solver`,
    /// `_ridge.py:830`: dense, non-positive data → `'cholesky'`).
    #[default]
    Auto,
    /// `'cholesky'` — solve the normal equations `(XᵀX + alpha·I)·w = Xᵀy`
    /// via a Cholesky factorization (`_solve_cholesky`, `_ridge.py:741`).
    Cholesky,
    /// `'svd'` — solve via the singular value decomposition of `X`
    /// (`_solve_svd`, `_ridge.py:200`). Numerically identical to
    /// [`Cholesky`](RidgeSolver::Cholesky) on the strictly-convex ridge
    /// problem (unique minimizer).
    Svd,
}

impl RidgeSolver {
    /// Resolve `Auto` to the concrete dense solver scikit-learn picks for the
    /// dense, non-positive path: `'cholesky'` (sklearn `resolve_solver`,
    /// `_ridge.py:830`). `Cholesky`/`Svd` resolve to themselves.
    #[must_use]
    fn resolve(self) -> RidgeSolver {
        match self {
            RidgeSolver::Auto => RidgeSolver::Cholesky,
            other => other,
        }
    }
}

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
    /// Optional per-target L2 penalties for the multi-output
    /// `Fit<Array2, Array2>` path (sklearn accepts `alpha` as an array of
    /// shape `(n_targets,)`, validated at
    /// `sklearn/linear_model/_ridge.py:701-712`). When `Some`, each target
    /// column `k` is fitted with its own penalty `alpha[k]`, overriding the
    /// scalar [`alpha`](Self::alpha); this is mathematically identical to
    /// fitting each target column with an independent scalar-`alpha` Ridge.
    /// Default `None` (the scalar `alpha` applies to every target).
    pub alpha_per_target: Option<Array1<F>>,
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
    /// Closed-form dense solver for the single-output unconstrained solve
    /// (sklearn `solver`, `_ridge.py:830` `resolve_solver`). One of
    /// [`RidgeSolver::Auto`] (default, resolves to `Cholesky` for the dense
    /// path), [`RidgeSolver::Cholesky`], or [`RidgeSolver::Svd`]. All dense
    /// solvers yield the IDENTICAL (unique) ridge solution; this only selects
    /// the factorization used. Governs ONLY the single-output
    /// (`Fit<Array2, Array1>`) unconstrained path — the multi-output and
    /// `positive=true` paths ignore it. The iterative solvers
    /// (`lsqr`/`sparse_cg`/`sag`/`saga`/`lbfgs`) are tracked as REQ-8b (#386).
    /// Default [`RidgeSolver::Auto`], matching sklearn's `solver='auto'`
    /// (`_ridge.py:903`).
    pub solver: RidgeSolver,
}

impl<F: Float> Ridge<F> {
    /// Create a new `Ridge` with default settings.
    ///
    /// Defaults: `alpha = 1.0`, `fit_intercept = true`, `copy_x = true`,
    /// `random_state = None`, `max_iter = None`, `tol = 1e-4`,
    /// `positive = false`, `solver = Auto` — mirroring sklearn's ctor defaults
    /// (`sklearn/linear_model/_ridge.py:895-903`).
    #[must_use]
    pub fn new() -> Self {
        Self {
            alpha: F::one(),
            alpha_per_target: None,
            fit_intercept: true,
            copy_x: true,
            random_state: None,
            max_iter: None,
            tol: F::from(1e-4).unwrap_or_else(F::epsilon),
            positive: false,
            solver: RidgeSolver::Auto,
        }
    }

    /// Set the regularization strength.
    #[must_use]
    pub fn with_alpha(mut self, alpha: F) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set per-target L2 penalties for the multi-output
    /// `Fit<Array2, Array2>` path (sklearn array-valued `alpha`,
    /// `sklearn/linear_model/_ridge.py:701-712`).
    ///
    /// When set, each target column `k` of `Y` is fitted with its own penalty
    /// `alphas[k]`, overriding the scalar [`alpha`](Self::alpha) on the
    /// multi-output fit. This is mathematically identical to fitting each
    /// target column with an independent scalar-`alpha` Ridge. The array
    /// length must equal the number of target columns and every entry must be
    /// non-negative (validated at fit time).
    #[must_use]
    pub fn with_alpha_per_target(mut self, alphas: Array1<F>) -> Self {
        self.alpha_per_target = Some(alphas);
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

    /// Set the dense closed-form solver for the single-output unconstrained
    /// solve (sklearn `solver`, `_ridge.py:830`).
    ///
    /// [`RidgeSolver::Auto`] (default) resolves to [`RidgeSolver::Cholesky`]
    /// for the dense path; [`RidgeSolver::Svd`] solves the same unique ridge
    /// problem via the SVD of `X`. All dense solvers yield the identical
    /// coefficients (the minimizer is unique); only the factorization differs.
    /// This governs ONLY the single-output (`Fit<Array2, Array1>`)
    /// unconstrained path.
    #[must_use]
    pub fn with_solver(mut self, solver: RidgeSolver) -> Self {
        self.solver = solver;
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
    ///
    /// Inactive-constraint short-circuit (R-DEV-6): we first compute the
    /// UNCONSTRAINED ridge optimum via the exact Cholesky solve
    /// (`crate::linalg::solve_ridge`). If that succeeds AND every component is
    /// `>= 0`, the positivity constraint is inactive, and by strict convexity +
    /// the KKT conditions a feasible unconstrained minimizer IS the constrained
    /// minimizer — so we return it as `(w_unc, 0)` (zero iterations, exact). This
    /// matches a CONVERGED sklearn exactly: sklearn's default positive-Ridge
    /// L-BFGS-B (`_solve_lbfgs`, `_ridge.py:300`) runs at `tol=1e-4` and
    /// under-converges on interior optima by ~1e-4, whereas at `tol=1e-12` it
    /// reaches the same true optimum ferrolearn computes here in closed form.
    /// Otherwise (the Cholesky solve errored, or any component is `< 0` so the
    /// constraint is active) we fall through to projected coordinate descent.
    fn solve_nonneg_ridge(&self, a: &Array2<F>, b: &Array1<F>) -> (Array1<F>, usize) {
        // Inactive-constraint short-circuit (R-DEV-6): the exact unconstrained
        // Cholesky optimum, when feasible (all components >= 0), IS the
        // constrained optimum (strict convexity + KKT). Return it exactly.
        if let Ok(w_unc) = crate::linalg::solve_ridge(a, b, self.alpha)
            && w_unc.iter().all(|&v| v >= <F as num_traits::Zero>::zero())
        {
            return (w_unc, 0);
        }

        // Active constraint (or the Cholesky solve failed) — delegate to the
        // shared projected-coordinate-descent kernel
        // (`crate::linalg::nonneg_ridge_cd`) so `Ridge` and `RidgeClassifier`
        // solve the non-negative ridge problem identically.
        crate::linalg::nonneg_ridge_cd(a, b, self.alpha, self.max_iter.unwrap_or(1000), self.tol)
    }

    /// Solve the unconstrained dense ridge system on the given (already
    /// centered / `√w`-rescaled) design, dispatching on the RESOLVED solver.
    ///
    /// - [`RidgeSolver::Svd`] → `linalg::solve_ridge_svd` (`_solve_svd`,
    ///   `_ridge.py:200`).
    /// - [`RidgeSolver::Cholesky`] / [`RidgeSolver::Auto`] →
    ///   `linalg::solve_ridge` (Cholesky normal equations, `_ridge.py:741`) —
    ///   byte-identical to the historic path.
    ///
    /// `resolved` is `self.solver.resolve()` (precomputed by the caller). Every
    /// dense solver returns the same unique minimizer; the choice only affects
    /// the factorization.
    fn solve_unconstrained(
        &self,
        resolved: RidgeSolver,
        x: &Array2<F>,
        y: &Array1<F>,
    ) -> Result<Array1<F>, FerroError> {
        match resolved {
            RidgeSolver::Svd => linalg::solve_ridge_svd(x, y, self.alpha),
            // `Auto` is resolved to `Cholesky` before this point; matched here
            // for exhaustiveness with byte-identical behaviour.
            RidgeSolver::Cholesky | RidgeSolver::Auto => linalg::solve_ridge(x, y, self.alpha),
        }
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

        // Non-finite input validation (#2259). sklearn `Ridge.fit` ->
        // `self._validate_data(X, y, ...)` (`_ridge.py:1242`) keeps the default
        // `force_all_finite=True`, so `check_array` rejects any NaN or +/-inf in
        // X OR y with a `ValueError` BEFORE the solve. sklearn also validates
        // `sample_weight` via `_check_sample_weight` (default `force_all_finite=
        // True`), raising on a non-finite weight. `.iter().any(|v| !v.is_finite())`
        // rejects both NaN and Inf (bounds-safe, no panic, R-CODE-2), matching
        // the crate idiom (`linear_regression.rs`/`lasso.rs`). The finite path is
        // byte-identical (the guard never fires on finite input). This is the
        // shared single-output entry; `Fit::fit` delegates here with `None`.
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
        if let Some(w) = sample_weight
            && w.iter().any(|v| !v.is_finite())
        {
            return Err(FerroError::InvalidParameter {
                name: "sample_weight".into(),
                reason: "Input sample_weight contains NaN or infinity.".into(),
            });
        }

        // Resolve the dense solver once (sklearn `resolve_solver`,
        // `_ridge.py:830`): `Auto` → `Cholesky` for the dense path. This is the
        // value stored as the fitted `solver_`. It governs only the
        // unconstrained dense solve; the `positive` path reports the same
        // resolved dense value (the `solver` field does not select the
        // constrained CD solver — see the `solver` field doc).
        let resolved = self.solver.resolve();

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
                            self.solve_unconstrained(resolved, &x_centered, &y_centered)?,
                            None,
                        )
                    };
                    let intercept = y_mean - x_mean.dot(&w);

                    Ok(FittedRidge {
                        coefficients: w,
                        intercept,
                        n_iter_,
                        solver_: resolved,
                    })
                } else {
                    let (w, n_iter_) = if self.positive {
                        let (coef, iters) = self.solve_nonneg_ridge(x, y);
                        (coef, Some(iters))
                    } else {
                        (self.solve_unconstrained(resolved, x, y)?, None)
                    };

                    Ok(FittedRidge {
                        coefficients: w,
                        // Disambiguate `Element::zero` vs `num_traits::Zero::zero`
                        // (both in scope under the `LinalgFloat` bound).
                        intercept: <F as num_traits::Zero>::zero(),
                        n_iter_,
                        solver_: resolved,
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
                        (
                            self.solve_unconstrained(resolved, &x_scaled, &y_scaled)?,
                            None,
                        )
                    };
                    let intercept = y_off - x_off.dot(&coef);

                    Ok(FittedRidge {
                        coefficients: coef,
                        intercept,
                        n_iter_,
                        solver_: resolved,
                    })
                } else {
                    // No centering; just √w row-rescaling, intercept 0.
                    let x_scaled = x * &w_sqrt.view().insert_axis(Axis(1));
                    let y_scaled = y * &w_sqrt;

                    let (coef, n_iter_) = if self.positive {
                        let (c, iters) = self.solve_nonneg_ridge(&x_scaled, &y_scaled);
                        (c, Some(iters))
                    } else {
                        (
                            self.solve_unconstrained(resolved, &x_scaled, &y_scaled)?,
                            None,
                        )
                    };

                    Ok(FittedRidge {
                        coefficients: coef,
                        intercept: <F as num_traits::Zero>::zero(),
                        n_iter_,
                        solver_: resolved,
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
    /// The RESOLVED dense solver actually used for the coefficient solve
    /// (sklearn `solver_`, the resolution of `solver='auto'`,
    /// `_ridge.py:830`/`:994`). `Auto` resolves to [`RidgeSolver::Cholesky`]
    /// for the dense path; an explicit `Cholesky`/`Svd` resolves to itself.
    solver_: RidgeSolver,
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

    /// Return the RESOLVED dense solver used for the coefficient solve
    /// (sklearn `solver_`, `_ridge.py:830`/`:994`).
    ///
    /// [`RidgeSolver::Auto`] resolves to [`RidgeSolver::Cholesky`] for the
    /// dense path, so a default-`solver` fit reports `Cholesky`; an explicit
    /// `Svd` reports `Svd`.
    #[must_use]
    pub fn solver(&self) -> RidgeSolver {
        self.solver_
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

impl<F: Float + Send + Sync + ScalarOperand + FromPrimitive + LinalgFloat + 'static>
    Fit<Array2<F>, Array2<F>> for Ridge<F>
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

        if n_samples == 0 {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "Ridge requires at least one sample".into(),
            });
        }

        // Non-finite input validation (#2259) — SEPARATE multi-output arm.
        // `Fit<Array2, Array2>::fit` does NOT delegate to
        // `fit_with_sample_weight`, so the same `_validate_data(force_all_finite=
        // True)` reject-at-fit contract (`_ridge.py:1242`, `multi_output=True`)
        // is enforced here independently before centering/solve.
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

        let n_targets = y.ncols();

        // Per-target alpha array (sklearn array-valued `alpha`,
        // `sklearn/linear_model/_ridge.py:701-712`) — validate the array and
        // solve each target column independently with its own penalty. When
        // `None`, fall through to the byte-identical scalar-alpha path below.
        if let Some(alphas) = &self.alpha_per_target {
            if alphas.len() != n_targets {
                return Err(FerroError::ShapeMismatch {
                    expected: vec![n_targets],
                    actual: vec![alphas.len()],
                    context: "alpha array length must equal number of targets".into(),
                });
            }
            for &a in alphas.iter() {
                if a < <F as num_traits::Zero>::zero() {
                    return Err(FerroError::InvalidParameter {
                        name: "alpha".into(),
                        reason: "must be non-negative".into(),
                    });
                }
            }

            let n_features = x.ncols();
            let mut coefficients = Array2::<F>::zeros((n_features, n_targets));
            let mut intercepts = Array1::<F>::zeros(n_targets);

            if self.fit_intercept {
                // Center once (identical to the scalar path), then solve each
                // centered target column with its own alpha.
                let x_mean =
                    x.mean_axis(Axis(0))
                        .ok_or_else(|| FerroError::NumericalInstability {
                            message: "failed to compute column means of X".into(),
                        })?;
                let y_mean =
                    y.mean_axis(Axis(0))
                        .ok_or_else(|| FerroError::NumericalInstability {
                            message: "failed to compute column means of Y".into(),
                        })?;

                let x_centered = x - &x_mean;
                let y_centered = y - &y_mean;

                for k in 0..n_targets {
                    let y_col = y_centered.column(k).to_owned();
                    let w_col = linalg::solve_ridge(&x_centered, &y_col, alphas[k])?;
                    intercepts[k] = y_mean[k] - x_mean.dot(&w_col);
                    coefficients.column_mut(k).assign(&w_col);
                }
            } else {
                for k in 0..n_targets {
                    let y_col = y.column(k).to_owned();
                    let w_col = linalg::solve_ridge(x, &y_col, alphas[k])?;
                    coefficients.column_mut(k).assign(&w_col);
                }
            }

            return Ok(FittedRidgeMulti {
                coefficients,
                intercepts,
            });
        }

        if self.alpha < <F as num_traits::Zero>::zero() {
            return Err(FerroError::InvalidParameter {
                name: "alpha".into(),
                reason: "must be non-negative".into(),
            });
        }

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

    // -- per-target alpha array (REQ-7) ------------------------------------

    /// Build the shared 5x2 fixture `X` and 5x2 `Y` used by the per-target
    /// alpha tests.
    fn per_target_fixture() -> Result<(Array2<f64>, Array2<f64>), FerroError> {
        let shape_err = |e: ndarray::ShapeError| FerroError::ShapeMismatch {
            expected: vec![5, 2],
            actual: vec![],
            context: e.to_string(),
        };
        let x = Array2::from_shape_vec((5, 2), vec![1., 2., 2., 1., 3., 4., 4., 3., 5., 5.])
            .map_err(shape_err)?;
        let y = Array2::from_shape_vec(
            (5, 2),
            vec![3.0, 1.0, 2.5, 2.0, 7.1, 3.5, 6.0, 4.2, 11.2, 6.0],
        )
        .map_err(shape_err)?;
        Ok((x, y))
    }

    #[test]
    fn ridge_per_target_alpha_matches_sklearn() -> Result<(), FerroError> {
        // Live sklearn 1.5.2 oracle (per-target alpha [0.5, 2.0]):
        //   cd /tmp && python3 -c "import numpy as np; \
        //     from sklearn.linear_model import Ridge; \
        //     X=np.array([[1.,2.],[2.,1.],[3.,4.],[4.,3.],[5.,5.]]); \
        //     Y=np.array([[3.,1.],[2.5,2.],[7.1,3.5],[6.,4.2],[11.2,6.]]); \
        //     m=Ridge(alpha=np.array([0.5,2.0])).fit(X,Y); \
        //     print(m.coef_.tolist(), m.intercept_.tolist())"
        //   -> coef_ (n_targets, n_features) = [[0.79891892, 1.43891892],
        //                                       [0.78, 0.355]]
        //      intercept_ = [-0.75351351, -0.065]
        // ferrolearn stores coefficients as (n_features, n_targets) = the
        // TRANSPOSE: column 0 = target 0, column 1 = target 1.
        let (x, y) = per_target_fixture()?;

        let model = Ridge::<f64>::new().with_alpha_per_target(array![0.5, 2.0]);
        let fitted = model.fit(&x, &y)?;

        let c0 = fitted.coefficients().column(0).to_owned();
        let c1 = fitted.coefficients().column(1).to_owned();
        assert_relative_eq!(c0[0], 0.798_918_92, epsilon = 1e-6);
        assert_relative_eq!(c0[1], 1.438_918_92, epsilon = 1e-6);
        assert_relative_eq!(c1[0], 0.78, epsilon = 1e-6);
        assert_relative_eq!(c1[1], 0.355, epsilon = 1e-6);

        assert_relative_eq!(fitted.intercepts()[0], -0.753_513_51, epsilon = 1e-6);
        assert_relative_eq!(fitted.intercepts()[1], -0.065, epsilon = 1e-6);
        Ok(())
    }

    #[test]
    fn ridge_per_target_alpha_equals_independent_scalar_fits() -> Result<(), FerroError> {
        // The per-target array path must reproduce, column by column, what an
        // independent scalar-alpha Ridge produces on each target column via the
        // single-output (1-D) `fit` path.
        let (x, y) = per_target_fixture()?;
        let alphas = array![0.5, 2.0];

        let multi = Ridge::<f64>::new()
            .with_alpha_per_target(alphas.clone())
            .fit(&x, &y)?;

        for k in 0..2 {
            let y_col = y.column(k).to_owned();
            let scalar = Ridge::<f64>::new().with_alpha(alphas[k]).fit(&x, &y_col)?;
            for j in 0..2 {
                assert_relative_eq!(
                    multi.coefficients()[[j, k]],
                    scalar.coefficients()[j],
                    epsilon = 1e-9
                );
            }
            assert_relative_eq!(multi.intercepts()[k], scalar.intercept(), epsilon = 1e-9);
        }
        Ok(())
    }

    #[test]
    fn ridge_per_target_alpha_length_mismatch_errors() -> Result<(), FerroError> {
        // alpha array of length 1 with 2 target columns must error.
        let (x, y) = per_target_fixture()?;

        let model = Ridge::<f64>::new().with_alpha_per_target(array![0.5]);
        let result = <Ridge<f64> as Fit<Array2<f64>, Array2<f64>>>::fit(&model, &x, &y);
        assert!(matches!(result, Err(FerroError::ShapeMismatch { .. })));
        Ok(())
    }

    #[test]
    fn ridge_multi_scalar_alpha_unchanged() -> Result<(), FerroError> {
        // Regression guard: with NO per-target array, the multi-output scalar
        // path is byte-identical to the historic `solve_ridge_multi` behavior.
        //
        // Live sklearn 1.5.2 oracle (scalar alpha=1.0):
        //   cd /tmp && python3 -c "import numpy as np; \
        //     from sklearn.linear_model import Ridge; \
        //     X=np.array([[1.,2.],[2.,1.],[3.,4.],[4.,3.],[5.,5.]]); \
        //     Y=np.array([[3.,1.],[2.5,2.],[7.1,3.5],[6.,4.2],[11.2,6.]]); \
        //     m=Ridge(alpha=1.0).fit(X,Y); \
        //     print(m.coef_.tolist(), m.intercept_.tolist())"
        let (x, y) = per_target_fixture()?;

        let baseline = Ridge::<f64>::new().with_alpha(1.0).fit(&x, &y)?;
        let explicit_none = Ridge::<f64> {
            alpha_per_target: None,
            ..Ridge::<f64>::new().with_alpha(1.0)
        }
        .fit(&x, &y)?;

        for j in 0..2 {
            for k in 0..2 {
                assert_eq!(
                    baseline.coefficients()[[j, k]].to_bits(),
                    explicit_none.coefficients()[[j, k]].to_bits(),
                    "coef_[{j},{k}] must be byte-identical without per-target alpha"
                );
            }
        }
        for k in 0..2 {
            assert_eq!(
                baseline.intercepts()[k].to_bits(),
                explicit_none.intercepts()[k].to_bits(),
                "intercept_[{k}] must be byte-identical without per-target alpha"
            );
        }
        Ok(())
    }

    // -- solver variants (auto/cholesky/svd) + solver_ (REQ-8a) ------------

    /// Build the shared 5x2 fixture `X` and 1-D `y` used by the solver tests.
    fn solver_fixture() -> Result<(Array2<f64>, Array1<f64>), FerroError> {
        let x = Array2::from_shape_vec((5, 2), vec![1., 2., 2., 1., 3., 4., 4., 3., 5., 5.])
            .map_err(|e| FerroError::ShapeMismatch {
                expected: vec![5, 2],
                actual: vec![],
                context: e.to_string(),
            })?;
        let y = array![3.0, 2.5, 7.1, 6.0, 11.2];
        Ok((x, y))
    }

    #[test]
    fn ridge_solver_svd_matches_sklearn_and_cholesky() -> Result<(), FerroError> {
        // Live sklearn 1.5.2 oracle (alpha=1, fit_intercept=True, solver='svd'):
        //   cd /tmp && python3 -c "import numpy as np; \
        //     from sklearn.linear_model import Ridge; \
        //     X=np.array([[1.,2.],[2.,1.],[3.,4.],[4.,3.],[5.,5.]]); \
        //     y=np.array([3.0,2.5,7.1,6.0,11.2]); \
        //     m=Ridge(alpha=1.0,solver='svd').fit(X,y); \
        //     print([round(c,10) for c in m.coef_], round(m.intercept_,10))"
        //   -> [0.8228070175, 1.3561403509] -0.5768421053
        // Every dense solver returns this same unique ridge solution.
        let (x, y) = solver_fixture()?;

        let svd_fit = Ridge::<f64>::new()
            .with_alpha(1.0)
            .with_solver(RidgeSolver::Svd)
            .fit(&x, &y)?;

        assert_relative_eq!(svd_fit.coefficients()[0], 0.822_807_017_5, epsilon = 1e-7);
        assert_relative_eq!(svd_fit.coefficients()[1], 1.356_140_350_9, epsilon = 1e-7);
        assert_relative_eq!(svd_fit.intercept(), -0.576_842_105_3, epsilon = 1e-7);

        // SVD and Cholesky solve the same strictly-convex (unique) problem, so
        // they must agree to ~1e-9.
        let chol_fit = Ridge::<f64>::new()
            .with_alpha(1.0)
            .with_solver(RidgeSolver::Cholesky)
            .fit(&x, &y)?;
        for j in 0..2 {
            assert_relative_eq!(
                svd_fit.coefficients()[j],
                chol_fit.coefficients()[j],
                epsilon = 1e-9
            );
        }
        assert_relative_eq!(svd_fit.intercept(), chol_fit.intercept(), epsilon = 1e-9);
        Ok(())
    }

    #[test]
    fn ridge_solver_resolution() -> Result<(), FerroError> {
        // Default `solver` is Auto; the fitted `solver_` resolves Auto→Cholesky
        // for the dense path (sklearn `resolve_solver`, `_ridge.py:830`).
        assert_eq!(Ridge::<f64>::new().solver, RidgeSolver::Auto);

        let (x, y) = solver_fixture()?;

        let auto_fit = Ridge::<f64>::new().with_alpha(1.0).fit(&x, &y)?;
        assert_eq!(
            auto_fit.solver(),
            RidgeSolver::Cholesky,
            "Auto must resolve to Cholesky for the dense path"
        );

        let svd_fit = Ridge::<f64>::new()
            .with_alpha(1.0)
            .with_solver(RidgeSolver::Svd)
            .fit(&x, &y)?;
        assert_eq!(svd_fit.solver(), RidgeSolver::Svd);

        let chol_fit = Ridge::<f64>::new()
            .with_alpha(1.0)
            .with_solver(RidgeSolver::Cholesky)
            .fit(&x, &y)?;
        assert_eq!(chol_fit.solver(), RidgeSolver::Cholesky);
        Ok(())
    }

    #[test]
    fn ridge_solver_default_cholesky_unchanged() -> Result<(), FerroError> {
        // Regression guard: the default (Auto→Cholesky) coef_/intercept_ are
        // BYTE-IDENTICAL to a pre-existing direct `fit` (no `with_solver`).
        let (x, y) = solver_fixture()?;

        let direct = Ridge::<f64>::new().with_alpha(1.0).fit(&x, &y)?;
        let explicit_chol = Ridge::<f64>::new()
            .with_alpha(1.0)
            .with_solver(RidgeSolver::Cholesky)
            .fit(&x, &y)?;
        let explicit_auto = Ridge::<f64>::new()
            .with_alpha(1.0)
            .with_solver(RidgeSolver::Auto)
            .fit(&x, &y)?;

        for j in 0..2 {
            assert_eq!(
                direct.coefficients()[j].to_bits(),
                explicit_chol.coefficients()[j].to_bits(),
                "coef_[{j}] must be byte-identical for explicit Cholesky"
            );
            assert_eq!(
                direct.coefficients()[j].to_bits(),
                explicit_auto.coefficients()[j].to_bits(),
                "coef_[{j}] must be byte-identical for explicit Auto"
            );
        }
        assert_eq!(
            direct.intercept().to_bits(),
            explicit_chol.intercept().to_bits()
        );
        assert_eq!(
            direct.intercept().to_bits(),
            explicit_auto.intercept().to_bits()
        );
        Ok(())
    }

    #[test]
    fn ridge_solver_svd_no_intercept() -> Result<(), FerroError> {
        // With fit_intercept=false, the SVD solver must match the Cholesky
        // no-intercept fit (same unique solution) to ~1e-9. sklearn:
        //   Ridge(alpha=1.0, fit_intercept=False, solver='svd') ==
        //   Ridge(alpha=1.0, fit_intercept=False, solver='cholesky').
        let (x, y) = solver_fixture()?;

        let svd_fit = Ridge::<f64>::new()
            .with_alpha(1.0)
            .with_fit_intercept(false)
            .with_solver(RidgeSolver::Svd)
            .fit(&x, &y)?;
        let chol_fit = Ridge::<f64>::new()
            .with_alpha(1.0)
            .with_fit_intercept(false)
            .with_solver(RidgeSolver::Cholesky)
            .fit(&x, &y)?;

        for j in 0..2 {
            assert_relative_eq!(
                svd_fit.coefficients()[j],
                chol_fit.coefficients()[j],
                epsilon = 1e-9
            );
        }
        assert_eq!(svd_fit.intercept(), 0.0);
        assert_eq!(chol_fit.intercept(), 0.0);
        assert_eq!(svd_fit.solver(), RidgeSolver::Svd);
        Ok(())
    }
}
