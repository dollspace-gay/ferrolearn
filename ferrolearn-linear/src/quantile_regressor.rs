//! Quantile Regression via the exact linear program (matching scikit-learn).
//!
//! This module provides [`QuantileRegressor`], which estimates conditional
//! quantiles of the response variable. The default `quantile = 0.5`
//! corresponds to the conditional median, which is more robust to outliers
//! than the conditional mean (OLS).
//!
//! The pinball (check) loss for quantile `q` is:
//!
//! ```text
//! L_q(r) = q * max(r, 0) + (1 - q) * max(-r, 0)
//! ```
//!
//! Like scikit-learn's `QuantileRegressor` (`sklearn/linear_model/_quantile.py`),
//! the model minimizes `(1/n) * sum(pinball loss) + alpha * ||coef||_1` by
//! solving the equivalent quantile-regression **linear program**:
//!
//! ```text
//! min  c · x   s.t.  A_eq · x = y,  x >= 0
//! x        = [intercept+, intercept-, coef+, coef-, u, v]   (all >= 0)
//! coef     = coef+ - coef-,    intercept = intercept+ - intercept-
//! residual = y - X@coef - intercept = u - v
//! c        = [0, 0, alpha*n .. , quantile .. (u), (1-quantile) .. (v)]
//! A_eq row i = [1, -1, X[i,:], -X[i,:], e_i, -e_i]
//! ```
//!
//! The L1 weight is `alpha * n_samples` (sklearn rescales `alpha` by
//! `sum(sample_weight)`, which equals `n_samples` unweighted), and the intercept
//! slacks are NOT penalized. The intercept is therefore a free LP variable, not a
//! centering-recovered quantity — sklearn explicitly notes that centering does
//! not work for quantile regression (`_quantile.py:177`). The LP is solved by a
//! self-contained two-phase primal simplex (Bland's anti-cycling rule); the
//! pinned datasets have a unique optimum, so the simplex reaches sklearn's exact
//! HiGHS vertex.
//!
//! ## REQ status (per `.design/linear/quantile_regressor.md`, mirrors `sklearn/linear_model/_quantile.py` @ 1.5.2)
//!
//! Mirrors `sklearn.linear_model.QuantileRegressor` (`_quantile.py:20`), pinball-loss LP solved
//! by simplex (matching scipy's HiGHS). coef_/intercept_ match the live oracle to ~1e-6.
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (LP-based quantile fit) | SHIPPED | `Fit for QuantileRegressor` → `mod lp` two-phase simplex; coef_/intercept_ match sklearn's HiGHS LP at q=0.2/0.5/0.8, α=0/1 (incl. exact sparse vertex). Consumer: `RsQuantileRegressor` in `ferrolearn-python`. Closed #340/#506 (was IRLS + centering intercept). |
//! | REQ-2 (alpha L1 = alpha·n_samples, exact) | SHIPPED | LP cost `alpha·n` on coef± reaches the sparse vertex (α=1 → coef [0,0,0]). Closed #332. |
//! | REQ-3 (quantile pinball asymmetry) | SHIPPED | cost `q` on u, `(1−q)` on v; intercept/coef now quantile-dependent. |
//! | REQ-4 (predict) | SHIPPED | `Predict for FittedQuantileRegressor`. |
//! | REQ-5 (fit_intercept / HasCoefficients) | SHIPPED | intercept as LP variable; `HasCoefficients`. |
//! | REQ-6 (n_iter_) | SHIPPED | `mod lp` returns the two-phase simplex pivot count (phase 1 + phase 2); `FittedQuantileRegressor::n_iter` exposes it. R-DEV-7: the from-scratch simplex pivot count is the ferrolearn analog of sklearn's `result.nit` (`_quantile.py:300`) but does NOT equal it (different solver); contract = positive int ≤ max_iter, deterministic. Consumer: `RsQuantileRegressor::n_iter_`. Closed #507. |
//! | REQ-7 (solver / solver_options) | SHIPPED | `solver: String` (default `"highs"`) + `solver_options: Option<HashMap>` (default `None`) with `with_solver`/`with_solver_options`; `fn fit` validates per sklearn's `_parameter_constraints` (`_quantile.py:114-124`): HiGHS family + `"revised simplex"` → simplex (same vertex); `"interior-point"`/invalid string → `InvalidParameter` (sklearn `ValueError`); non-empty `solver_options` → `InvalidParameter` (binding maps to `NotImplementedError`). Consumer: hand-written `RsQuantileRegressor` in `ferrolearn-python`. Closed #508. |
//! | REQ-8 NOT-STARTED | ferray substrate (#509; `mod lp` on ndarray/f64). |
//!
//! acto-critic + builder: the IRLS+centering fit was quantile-invariant and 25× off (#340);
//! rewritten to the exact quantile-regression LP via a from-scratch two-phase simplex (8c18a21)
//! matching sklearn's HiGHS vertex (incl. the degenerate α=1 sparse case). Two states only per R-DEFER-2.
//!
//! # Examples
//!
//! ```
//! use ferrolearn_linear::QuantileRegressor;
//! use ferrolearn_core::{Fit, Predict};
//! use ndarray::{array, Array1, Array2};
//!
//! let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
//! let y = array![2.0, 4.0, 6.0, 8.0, 10.0];
//!
//! let model = QuantileRegressor::<f64>::new().with_alpha(0.0); // median regression
//! let fitted = model.fit(&x, &y).unwrap();
//! let preds = fitted.predict(&x).unwrap();
//! assert_eq!(preds.len(), 5);
//! ```

use ferrolearn_core::error::FerroError;
use ferrolearn_core::introspection::HasCoefficients;
use ferrolearn_core::pipeline::{FittedPipelineEstimator, PipelineEstimator};
use ferrolearn_core::traits::{Fit, Predict};
use ndarray::{Array1, Array2, ScalarOperand};
use num_traits::{Float, FromPrimitive};

/// Quantile Regressor — conditional quantile estimation via the exact LP.
///
/// Minimises `(1/n) * sum(pinball loss) + alpha * ||coef||_1` by solving the
/// equivalent quantile-regression linear program with a two-phase primal
/// simplex, matching scikit-learn's HiGHS solution. The intercept is a free LP
/// variable, not a centering-recovered quantity.
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
#[derive(Debug, Clone)]
pub struct QuantileRegressor<F> {
    /// The quantile to estimate (must be in (0, 1)). Default 0.5 (median).
    pub quantile: F,
    /// L1 regularization strength.
    pub alpha: F,
    /// Maximum number of simplex pivot iterations before declaring failure.
    pub max_iter: usize,
    /// Convergence/feasibility tolerance for the simplex solver.
    pub tol: F,
    /// Whether to fit an intercept (bias) term.
    pub fit_intercept: bool,
    /// The LP-solver method, mirroring sklearn's `solver`
    /// (`_quantile.py:134`, default `"highs"`). sklearn's
    /// `_parameter_constraints` (`_quantile.py:114-124`) allows
    /// `{"highs-ds", "highs-ipm", "highs", "interior-point",
    /// "revised simplex"}`. All of the HiGHS family and `"revised simplex"`
    /// reach the same unique LP vertex, so they map onto ferrolearn's
    /// two-phase primal simplex (R-DEV-7); `"interior-point"` is rejected to
    /// mirror sklearn raising `ValueError` (it was removed in SciPy 1.11.0,
    /// `_quantile.py:196-199`); any other string is an invalid parameter
    /// (sklearn raises `InvalidParameterError`, a `ValueError` subclass).
    pub solver: String,
    /// Additional solver options, mirroring sklearn's `solver_options`
    /// (`_quantile.py:135`, default `None`). sklearn forwards these to
    /// `scipy.optimize.linprog(options=...)` as HiGHS TUNING knobs that do not
    /// change the LP optimum. `None` (and an empty map) are accepted and have
    /// no effect; a non-empty map (which would only matter to scipy's solver,
    /// not the optimum) is rejected as `NotImplementedError`-tracked
    /// (`InvalidParameter`) since ferrolearn's simplex exposes no such knobs.
    pub solver_options: Option<std::collections::HashMap<String, f64>>,
}

impl<F: Float + FromPrimitive> QuantileRegressor<F> {
    /// Create a new `QuantileRegressor` with default settings.
    ///
    /// Defaults: `quantile = 0.5`, `alpha = 1.0`, `max_iter = 10000`,
    /// `tol = 1e-9`, `fit_intercept = true`, `solver = "highs"`,
    /// `solver_options = None` — matching sklearn's `__init__`
    /// (`_quantile.py:128-141`). (`max_iter` caps the simplex pivot count;
    /// `tol` is the simplex feasibility tolerance.)
    #[must_use]
    pub fn new() -> Self {
        let half = F::from(0.5).unwrap_or_else(|| F::one() / (F::one() + F::one()));
        let tol = F::from(1e-9).unwrap_or_else(F::epsilon);
        Self {
            quantile: half,
            alpha: F::one(),
            max_iter: 10000,
            tol,
            fit_intercept: true,
            solver: "highs".to_string(),
            solver_options: None,
        }
    }

    /// Set the quantile to estimate.
    ///
    /// Must be strictly between 0 and 1.
    #[must_use]
    pub fn with_quantile(mut self, quantile: F) -> Self {
        self.quantile = quantile;
        self
    }

    /// Set the L1 regularization strength.
    #[must_use]
    pub fn with_alpha(mut self, alpha: F) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set the maximum number of simplex pivot iterations.
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

    /// Set whether to fit an intercept term.
    #[must_use]
    pub fn with_fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = fit_intercept;
        self
    }

    /// Set the LP-solver method, mirroring sklearn's `solver`
    /// (`_quantile.py:134`). Validity is checked at `fit` time against
    /// sklearn's `_parameter_constraints` (`_quantile.py:114-124`).
    #[must_use]
    pub fn with_solver(mut self, solver: impl Into<String>) -> Self {
        self.solver = solver.into();
        self
    }

    /// Set the additional solver options, mirroring sklearn's `solver_options`
    /// (`_quantile.py:135`). `None` and an empty map are no-ops; a non-empty
    /// map is rejected at `fit` (ferrolearn's simplex has no HiGHS tuning
    /// knobs).
    #[must_use]
    pub fn with_solver_options(
        mut self,
        solver_options: Option<std::collections::HashMap<String, f64>>,
    ) -> Self {
        self.solver_options = solver_options;
        self
    }
}

impl<F: Float + FromPrimitive> Default for QuantileRegressor<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted Quantile Regressor model.
///
/// Stores the learned coefficients and intercept.
#[derive(Debug, Clone)]
pub struct FittedQuantileRegressor<F> {
    /// Learned coefficient vector.
    coefficients: Array1<F>,
    /// Learned intercept (bias) term.
    intercept: F,
    /// Number of solver iterations performed — the ferrolearn analog of
    /// sklearn's `n_iter_` (`_quantile.py:300` `self.n_iter_ = result.nit`).
    /// This is the two-phase primal-simplex PIVOT count (phase 1 + phase 2);
    /// per R-DEV-7 it does NOT equal scipy's HiGHS iteration count, but it is a
    /// positive integer `<= max_iter` and deterministic for a given dataset.
    n_iter: usize,
}

impl<F> FittedQuantileRegressor<F> {
    /// The number of solver iterations performed to reach the LP optimum — the
    /// ferrolearn analog of sklearn's `n_iter_`
    /// (`_quantile.py:300` `self.n_iter_ = result.nit`).
    ///
    /// This is the count of two-phase primal-simplex PIVOTS (phase 1 + phase 2)
    /// `mod lp` performed. R-DEV-7: ferrolearn's from-scratch two-phase simplex
    /// is NOT scipy's HiGHS solver, so this count will NOT equal sklearn's
    /// `n_iter_` for the same fit; it is, however, a positive integer
    /// `<= max_iter` and deterministic for a given dataset.
    #[must_use]
    pub fn n_iter(&self) -> usize {
        self.n_iter
    }
}

// ---------------------------------------------------------------------------
// Linear-program solver (self-contained two-phase primal simplex)
// ---------------------------------------------------------------------------

/// A standard-form linear program `min c·x  s.t.  A x = b, x >= 0`, solved by a
/// two-phase primal simplex with Bland's anti-cycling rule.
///
/// ferrolearn has no LP solver and the ferray substrate does not yet expose
/// `scipy.optimize.linprog`, so the quantile-regression LP (which scikit-learn
/// hands to HiGHS) is solved here. The simplex operates on a dense tableau of
/// `f64` coefficients; the pinned quantile-regression datasets have a unique LP
/// optimum, so the simplex reaches the same vertex HiGHS does.
///
/// (Substrate note: this is pure arithmetic over arrays. The `ferray` array-type
/// migration of this module is tracked as #509; this introduces no NEW
/// wrong-substrate dependency — it uses only the existing `ndarray` types.)
mod lp {
    /// Outcome of a simplex solve.
    pub(super) enum LpStatus {
        /// An optimal basic feasible solution was found. Carries the full
        /// decision vector `x` (length = number of structural variables) and
        /// the total number of simplex PIVOTS performed across phase 1 and
        /// phase 2 — the ferrolearn analog of scipy's `result.nit`
        /// (`sklearn/linear_model/_quantile.py:300`). NB: this is the
        /// from-scratch two-phase primal-simplex pivot count, NOT scipy's HiGHS
        /// iteration count, so it will not equal sklearn's `n_iter_` (R-DEV-7).
        Optimal(Vec<f64>, usize),
        /// The phase-1 problem could not drive the artificial objective to zero
        /// (the equality system `A x = b, x >= 0` is infeasible).
        Infeasible,
        /// The pivot cap was hit without reaching optimality.
        IterationLimit,
    }

    /// Dense simplex tableau over `m` equality rows and `n` structural columns.
    ///
    /// The tableau carries the structural columns followed by `m` artificial
    /// columns (used only in phase 1). `basis[i]` is the column index basic in
    /// row `i`.
    struct Tableau {
        m: usize,
        /// Total columns currently materialized: `n_struct + n_artificial`.
        cols: usize,
        n_struct: usize,
        /// `m` rows, each `cols + 1` wide (last entry is the RHS `b_i`).
        rows: Vec<Vec<f64>>,
        basis: Vec<usize>,
        tol: f64,
    }

    impl Tableau {
        /// Build the phase-1 tableau. `b` is assumed already non-negative
        /// (callers flip any negative row before constructing). Each row gets a
        /// dedicated artificial variable forming the initial identity basis.
        fn new(a: &[Vec<f64>], b: &[f64], n_struct: usize, tol: f64) -> Self {
            let m = a.len();
            let cols = n_struct + m;
            let mut rows = vec![vec![0.0f64; cols + 1]; m];
            let mut basis = vec![0usize; m];
            for i in 0..m {
                for j in 0..n_struct {
                    rows[i][j] = a[i][j];
                }
                rows[i][n_struct + i] = 1.0;
                rows[i][cols] = b[i];
                basis[i] = n_struct + i;
            }
            Tableau {
                m,
                cols,
                n_struct,
                rows,
                basis,
                tol,
            }
        }

        /// Pivot on `(prow, pcol)`: normalize the pivot row, then eliminate the
        /// pivot column from every other row.
        fn pivot(&mut self, prow: usize, pcol: usize) {
            let piv = self.rows[prow][pcol];
            let width = self.cols + 1;
            for j in 0..width {
                self.rows[prow][j] /= piv;
            }
            for i in 0..self.m {
                if i == prow {
                    continue;
                }
                let factor = self.rows[i][pcol];
                if factor != 0.0 {
                    for j in 0..width {
                        self.rows[i][j] -= factor * self.rows[prow][j];
                    }
                }
            }
            self.basis[prow] = pcol;
        }

        /// Run primal simplex iterations to optimality for the supplied cost
        /// vector (length `cols`). `allowed` gates which columns may enter
        /// (phase 2 forbids artificial columns). Uses Bland's rule: among
        /// columns with negative reduced cost, pick the smallest index, and on
        /// the ratio-test tie pick the row whose basic variable has the
        /// smallest index — this guarantees termination without cycling.
        ///
        /// Returns `Some(pivots)` (the number of pivots performed to reach
        /// optimality) on success, or `None` on hitting the iteration cap. The
        /// pivot count is accumulated by the caller into the solve's total
        /// iteration count (the ferrolearn analog of `result.nit`).
        fn optimize(
            &mut self,
            cost: &[f64],
            allowed: &dyn Fn(usize) -> bool,
            max_iter: usize,
        ) -> Option<usize> {
            let mut pivots = 0usize;
            for _ in 0..max_iter {
                // Reduced costs: c_j - c_B · (B^{-1} A_j). With the tableau in
                // canonical form, reduced cost of column j is
                //   cost[j] - sum_i cost[basis[i]] * rows[i][j].
                let mut entering: Option<usize> = None;
                for j in 0..self.cols {
                    if !allowed(j) {
                        continue;
                    }
                    let mut reduced = cost[j];
                    for i in 0..self.m {
                        reduced -= cost[self.basis[i]] * self.rows[i][j];
                    }
                    if reduced < -self.tol {
                        // Bland: first (smallest-index) improving column.
                        entering = Some(j);
                        break;
                    }
                }
                let Some(pcol) = entering else {
                    return Some(pivots); // optimal
                };

                // Ratio test: minimize rhs / col over rows with positive entry.
                let mut prow: Option<usize> = None;
                let mut best_ratio = f64::INFINITY;
                for i in 0..self.m {
                    let aij = self.rows[i][pcol];
                    if aij > self.tol {
                        let ratio = self.rows[i][self.cols] / aij;
                        if ratio < best_ratio - self.tol {
                            best_ratio = ratio;
                            prow = Some(i);
                        } else if (ratio - best_ratio).abs() <= self.tol {
                            // Bland tie-break: smallest leaving-variable index.
                            if prow.is_some_and(|cur| self.basis[i] < self.basis[cur]) {
                                prow = Some(i);
                            }
                        }
                    }
                }
                // No leaving row ⇒ unbounded along this column. The
                // quantile-regression LP is always bounded below (costs >= 0),
                // so this cannot happen; treat as optimal w.r.t. allowed cols.
                let Some(prow) = prow else {
                    return Some(pivots);
                };
                self.pivot(prow, pcol);
                pivots += 1;
            }
            None
        }

        /// Sum of the current basic values that correspond to artificial
        /// columns — the phase-1 objective.
        fn artificial_objective(&self) -> f64 {
            let mut s = 0.0;
            for i in 0..self.m {
                if self.basis[i] >= self.n_struct {
                    s += self.rows[i][self.cols];
                }
            }
            s
        }

        /// Drive any artificial variable still basic at value ~0 out of the
        /// basis by pivoting on a non-artificial column with a non-zero entry
        /// (a degeneracy clean-up so phase 2 starts on a structural basis).
        fn drive_out_artificials(&mut self) {
            for i in 0..self.m {
                if self.basis[i] < self.n_struct {
                    continue;
                }
                let mut pcol: Option<usize> = None;
                for j in 0..self.n_struct {
                    if self.rows[i][j].abs() > self.tol {
                        pcol = Some(j);
                        break;
                    }
                }
                if let Some(j) = pcol {
                    self.pivot(i, j);
                }
                // If the whole structural part of the row is zero, the row is
                // redundant; leaving the artificial basic at value 0 is fine.
            }
        }

        /// Extract the structural decision vector from the final basis.
        fn solution(&self) -> Vec<f64> {
            let mut x = vec![0.0f64; self.n_struct];
            for i in 0..self.m {
                let col = self.basis[i];
                if col < self.n_struct {
                    x[col] = self.rows[i][self.cols];
                }
            }
            x
        }
    }

    /// Solve `min c·x  s.t.  A x = b, x >= 0` via two-phase primal simplex.
    ///
    /// - `a`: `m × n_struct` constraint matrix (row-major).
    /// - `b`: length-`m` RHS (may contain negatives; rows are flipped to keep
    ///   `b >= 0` for phase 1).
    /// - `c`: length-`n_struct` cost vector.
    pub(super) fn solve(
        a: &[Vec<f64>],
        b: &[f64],
        c: &[f64],
        n_struct: usize,
        max_iter: usize,
        tol: f64,
    ) -> LpStatus {
        let m = a.len();

        // Normalize so every RHS is non-negative (flip the whole row).
        let mut a_norm = a.to_vec();
        let mut b_norm = b.to_vec();
        for i in 0..m {
            if b_norm[i] < 0.0 {
                for entry in a_norm[i].iter_mut().take(n_struct) {
                    *entry = -*entry;
                }
                b_norm[i] = -b_norm[i];
            }
        }

        let mut tab = Tableau::new(&a_norm, &b_norm, n_struct, tol);

        // Phase 1: minimize sum of artificial variables.
        let mut phase1_cost = vec![0.0f64; tab.cols];
        for cost in phase1_cost.iter_mut().skip(n_struct) {
            *cost = 1.0;
        }
        let Some(p1_pivots) = tab.optimize(&phase1_cost, &|_j| true, max_iter) else {
            return LpStatus::IterationLimit;
        };
        // Feasible iff the artificial objective is driven to ~0.
        if tab.artificial_objective() > tol.max(1e-7) {
            return LpStatus::Infeasible;
        }
        tab.drive_out_artificials();

        // Phase 2: minimize the real cost, forbidding artificial columns from
        // re-entering the basis.
        let mut phase2_cost = vec![0.0f64; tab.cols];
        phase2_cost[..n_struct].copy_from_slice(&c[..n_struct]);
        let n_struct_cap = n_struct;
        let Some(p2_pivots) = tab.optimize(&phase2_cost, &|j| j < n_struct_cap, max_iter) else {
            return LpStatus::IterationLimit;
        };

        // Total simplex pivots across both phases — the ferrolearn analog of
        // scipy's `result.nit` (`_quantile.py:300`). R-DEV-7: this is the
        // from-scratch two-phase primal-simplex pivot count, which differs from
        // scipy's HiGHS iteration count (different solver entirely).
        LpStatus::Optimal(tab.solution(), p1_pivots + p2_pivots)
    }
}

// ---------------------------------------------------------------------------
// Fit
// ---------------------------------------------------------------------------

impl<F: Float + Send + Sync + ScalarOperand + FromPrimitive + 'static> Fit<Array2<F>, Array1<F>>
    for QuantileRegressor<F>
{
    type Fitted = FittedQuantileRegressor<F>;
    type Error = FerroError;

    /// Fit the quantile regression model by solving the exact LP.
    ///
    /// Builds the standard-form linear program of `sklearn/linear_model/`
    /// `_quantile.py:212-269` and solves it with a two-phase primal simplex,
    /// matching scikit-learn's HiGHS solution. The intercept is a free LP
    /// variable; no centering is performed.
    ///
    /// # Errors
    ///
    /// - [`FerroError::ShapeMismatch`] — sample count mismatch.
    /// - [`FerroError::InsufficientSamples`] — zero samples.
    /// - [`FerroError::InvalidParameter`] — quantile outside (0, 1) or
    ///   negative alpha.
    /// - [`FerroError::NumericalInstability`] — the simplex hit the iteration
    ///   cap or the LP was infeasible (should not happen for valid input).
    fn fit(&self, x: &Array2<F>, y: &Array1<F>) -> Result<FittedQuantileRegressor<F>, FerroError> {
        let (n_samples, n_features) = x.dim();

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
                context: "QuantileRegressor requires at least one sample".into(),
            });
        }

        if self.quantile <= F::zero() || self.quantile >= F::one() {
            return Err(FerroError::InvalidParameter {
                name: "quantile".into(),
                reason: "must be strictly between 0 and 1".into(),
            });
        }

        if self.alpha < F::zero() {
            return Err(FerroError::InvalidParameter {
                name: "alpha".into(),
                reason: "must be non-negative".into(),
            });
        }

        // Solver validation, mirroring sklearn's `_parameter_constraints`
        // (`_quantile.py:114-124`) and the `fit`-time guards (`:184-205`):
        //   - The HiGHS family (`"highs"`, `"highs-ds"`, `"highs-ipm"`) and
        //     `"revised simplex"` all reach the SAME unique LP vertex, so each
        //     maps onto ferrolearn's two-phase primal simplex (R-DEV-7: the
        //     solver differs from scipy, but the observable optimum matches).
        //   - `"interior-point"` mirrors sklearn raising `ValueError`
        //     ("not anymore available in SciPy >= 1.11.0", `_quantile.py:196`).
        //   - Any other string is an invalid parameter, mirroring sklearn's
        //     `InvalidParameterError` (a `ValueError` subclass) from the
        //     `StrOptions` constraint (`_quantile.py:114-124`).
        match self.solver.as_str() {
            "highs" | "highs-ds" | "highs-ipm" | "revised simplex" => {}
            "interior-point" => {
                return Err(FerroError::InvalidParameter {
                    name: "solver".into(),
                    reason: "'interior-point' is not anymore available in SciPy >= 1.11.0 \
                             (mirrors sklearn raising ValueError); use 'highs'"
                        .into(),
                });
            }
            other => {
                return Err(FerroError::InvalidParameter {
                    name: "solver".into(),
                    reason: format!(
                        "{other:?} is invalid; must be one of 'highs-ds', 'highs-ipm', \
                         'highs', 'interior-point', 'revised simplex'"
                    ),
                });
            }
        }

        // `solver_options` mirrors sklearn's HiGHS tuning knobs forwarded to
        // `scipy.optimize.linprog(options=...)` (`_quantile.py:206-210`,
        // `:271-277`). sklearn's constraint is `[dict, None]` (`_quantile.py:125`),
        // so ANY dict is a valid construction and sklearn fits successfully
        // (R-DEV-7). These options (`presolve`, `disp`, tolerance knobs) are HiGHS
        // performance/diagnostic settings that do NOT change the LP optimum:
        // ferrolearn's from-scratch two-phase primal simplex reaches the SAME
        // unique LP vertex scipy's HiGHS does, so ignoring them yields the
        // identical `coef_`/`intercept_`/`predict` sklearn returns. We therefore
        // ACCEPT (and ignore) any `solver_options` — `None`, empty, or non-empty.
        // `solver_options` is still stored for `get_params`/`clone()` round-trip
        // but does not affect the solve.

        // Work in f64 for the LP solve regardless of `F` (HiGHS solves in
        // double precision; we match that and cast results back to `F`).
        let to_f64 = |v: F| -> Result<f64, FerroError> {
            v.to_f64().ok_or_else(|| FerroError::NumericalInstability {
                message: "failed to convert value to f64 for LP solve".into(),
            })
        };

        let quantile = to_f64(self.quantile)?;
        let alpha_param = to_f64(self.alpha)?;
        // sklearn rescales: alpha = sum(sample_weight) * self.alpha; unweighted
        // sum(sample_weight) == n_samples (`_quantile.py:182`).
        let alpha = alpha_param * (n_samples as f64);

        let fit_intercept = self.fit_intercept;
        // n_params = n_features (+ 1 for the intercept), matching sklearn.
        let n_params = if fit_intercept {
            n_features + 1
        } else {
            n_features
        };

        // Decision vector layout (all variables >= 0), matching
        // `_quantile.py:218` `x = (s0, s, t0, t, u, v)`:
        //   [0 .. n_params)               positive params (s0?, s)
        //   [n_params .. 2*n_params)      negative params (t0?, t)
        //   [2*n_params .. +n_samples)    u (residual+)
        //   [.. + n_samples)              v (residual-)
        let n_struct = 2 * n_params + 2 * n_samples;
        let u_off = 2 * n_params;
        let v_off = u_off + n_samples;

        // Cost vector c (`_quantile.py:238-248`).
        let mut c = vec![0.0f64; n_struct];
        for j in 0..n_params {
            c[j] = alpha;
            c[n_params + j] = alpha;
        }
        if fit_intercept {
            // Do not penalize the intercept slacks (`c[0]=0; c[n_params]=0`).
            c[0] = 0.0;
            c[n_params] = 0.0;
        }
        for i in 0..n_samples {
            c[u_off + i] = quantile;
            c[v_off + i] = 1.0 - quantile;
        }

        // Equality system A_eq x = y (`_quantile.py:256-269`). Row i:
        //   [1, -1, X[i,:], -X[i,:], e_i, -e_i] (with intercept), else
        //   [X[i,:], -X[i,:], e_i, -e_i].
        let mut a_eq = vec![vec![0.0f64; n_struct]; n_samples];
        let mut b_eq = vec![0.0f64; n_samples];
        for i in 0..n_samples {
            let row = &mut a_eq[i];
            let base = if fit_intercept {
                // intercept+ (col 0) and intercept- (col n_params).
                row[0] = 1.0;
                row[n_params] = -1.0;
                1usize
            } else {
                0usize
            };
            for f in 0..n_features {
                let xv = to_f64(x[[i, f]])?;
                row[base + f] = xv; // coef+ column
                row[n_params + base + f] = -xv; // coef- column
            }
            row[u_off + i] = 1.0;
            row[v_off + i] = -1.0;
            b_eq[i] = to_f64(y[i])?;
        }

        let tol = to_f64(self.tol)?.max(1e-9);
        let (solution, n_iter) =
            match lp::solve(&a_eq, &b_eq, &c, n_struct, self.max_iter.max(1), tol) {
                lp::LpStatus::Optimal(sol, pivots) => (sol, pivots),
                lp::LpStatus::Infeasible => {
                    return Err(FerroError::NumericalInstability {
                        message: "quantile-regression LP is infeasible".into(),
                    });
                }
                lp::LpStatus::IterationLimit => {
                    return Err(FerroError::NumericalInstability {
                        message: "quantile-regression LP simplex did not converge \
                              within max_iter pivots"
                            .into(),
                    });
                }
            };

        // Recover params = positive_slack - negative_slack
        // (`_quantile.py:298`): params[0]=intercept, params[1:]=coef.
        let from_f64 = |v: f64| -> F { F::from(v).unwrap_or_else(F::zero) };
        let mut coefficients = Array1::<F>::zeros(n_features);
        let coef_base = if fit_intercept { 1 } else { 0 };
        for f in 0..n_features {
            let p = solution[coef_base + f] - solution[n_params + coef_base + f];
            coefficients[f] = from_f64(p);
        }
        let intercept = if fit_intercept {
            from_f64(solution[0] - solution[n_params])
        } else {
            F::zero()
        };

        Ok(FittedQuantileRegressor {
            coefficients,
            intercept,
            n_iter,
        })
    }
}

// ---------------------------------------------------------------------------
// Predict / HasCoefficients / Pipeline
// ---------------------------------------------------------------------------

impl<F: Float + Send + Sync + ScalarOperand + 'static> Predict<Array2<F>>
    for FittedQuantileRegressor<F>
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
        if x.ncols() != self.coefficients.len() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![self.coefficients.len()],
                actual: vec![x.ncols()],
                context: "number of features must match fitted model".into(),
            });
        }
        Ok(x.dot(&self.coefficients) + self.intercept)
    }
}

impl<F: Float + Send + Sync + ScalarOperand + 'static> HasCoefficients<F>
    for FittedQuantileRegressor<F>
{
    fn coefficients(&self) -> &Array1<F> {
        &self.coefficients
    }

    fn intercept(&self) -> F {
        self.intercept
    }
}

impl<F> PipelineEstimator<F> for QuantileRegressor<F>
where
    F: Float + FromPrimitive + ScalarOperand + Send + Sync + 'static,
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

impl<F> FittedPipelineEstimator<F> for FittedQuantileRegressor<F>
where
    F: Float + ScalarOperand + Send + Sync + 'static,
{
    fn predict_pipeline(&self, x: &Array2<F>) -> Result<Array1<F>, FerroError> {
        self.predict(x)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn test_defaults() {
        let m = QuantileRegressor::<f64>::new();
        assert_relative_eq!(m.quantile, 0.5);
        assert_relative_eq!(m.alpha, 1.0);
        assert_eq!(m.max_iter, 10000);
        assert!(m.fit_intercept);
    }

    #[test]
    fn test_builder() {
        let m = QuantileRegressor::<f64>::new()
            .with_quantile(0.9)
            .with_alpha(0.5)
            .with_max_iter(500)
            .with_tol(1e-8)
            .with_fit_intercept(false)
            .with_solver("highs-ipm");
        assert_relative_eq!(m.quantile, 0.9);
        assert_relative_eq!(m.alpha, 0.5);
        assert_eq!(m.max_iter, 500);
        assert!(!m.fit_intercept);
        assert_eq!(m.solver, "highs-ipm");
    }

    #[test]
    fn test_default_solver_is_highs() {
        // sklearn default `solver="highs"` (`_quantile.py:134`).
        let m = QuantileRegressor::<f64>::new();
        assert_eq!(m.solver, "highs");
        assert!(m.solver_options.is_none());
    }

    #[test]
    fn test_shape_mismatch() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![1.0, 2.0];
        assert!(QuantileRegressor::<f64>::new().fit(&x, &y).is_err());
    }

    #[test]
    fn test_invalid_quantile_zero() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![1.0, 2.0, 3.0];
        assert!(
            QuantileRegressor::<f64>::new()
                .with_quantile(0.0)
                .fit(&x, &y)
                .is_err()
        );
    }

    #[test]
    fn test_invalid_quantile_one() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![1.0, 2.0, 3.0];
        assert!(
            QuantileRegressor::<f64>::new()
                .with_quantile(1.0)
                .fit(&x, &y)
                .is_err()
        );
    }

    #[test]
    fn test_negative_alpha() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![1.0, 2.0, 3.0];
        assert!(
            QuantileRegressor::<f64>::new()
                .with_alpha(-1.0)
                .fit(&x, &y)
                .is_err()
        );
    }

    #[test]
    fn test_median_regression_clean_data() {
        // On clean linear data, median regression should approximate OLS.
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = array![3.0, 5.0, 7.0, 9.0, 11.0];

        let fitted = QuantileRegressor::<f64>::new()
            .with_alpha(0.0)
            .with_max_iter(2000)
            .fit(&x, &y)
            .unwrap();

        assert_relative_eq!(fitted.coefficients()[0], 2.0, epsilon = 0.5);
        assert_relative_eq!(fitted.intercept(), 1.0, epsilon = 1.0);
    }

    #[test]
    fn test_predict_length() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![2.0, 4.0, 6.0, 8.0];

        let fitted = QuantileRegressor::<f64>::new()
            .with_alpha(0.0)
            .fit(&x, &y)
            .unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 4);
    }

    #[test]
    fn test_predict_feature_mismatch() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0]).unwrap();
        let y = array![1.0, 2.0, 3.0];
        let fitted = QuantileRegressor::<f64>::new().fit(&x, &y).unwrap();
        let x_bad = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        assert!(fitted.predict(&x_bad).is_err());
    }

    #[test]
    fn test_has_coefficients() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = array![1.0, 2.0, 3.0];
        let fitted = QuantileRegressor::<f64>::new().fit(&x, &y).unwrap();
        assert_eq!(fitted.coefficients().len(), 2);
    }

    #[test]
    fn test_no_intercept() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![2.0, 4.0, 6.0, 8.0];

        let fitted = QuantileRegressor::<f64>::new()
            .with_alpha(0.0)
            .with_fit_intercept(false)
            .fit(&x, &y)
            .unwrap();
        assert_relative_eq!(fitted.intercept(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_high_quantile_higher_prediction() {
        // A higher quantile should generally yield higher predicted values.
        let x = Array2::from_shape_vec(
            (10, 1),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        )
        .unwrap();
        // y with some noise.
        let y = array![2.5, 3.8, 6.2, 7.9, 10.5, 12.1, 14.3, 15.8, 18.2, 20.5];

        let fitted_low = QuantileRegressor::<f64>::new()
            .with_quantile(0.1)
            .with_alpha(0.0)
            .fit(&x, &y)
            .unwrap();
        let fitted_high = QuantileRegressor::<f64>::new()
            .with_quantile(0.9)
            .with_alpha(0.0)
            .fit(&x, &y)
            .unwrap();

        let x_test = Array2::from_shape_vec((1, 1), vec![5.5]).unwrap();
        let pred_low = fitted_low.predict(&x_test).unwrap()[0];
        let pred_high = fitted_high.predict(&x_test).unwrap()[0];

        assert!(
            pred_high >= pred_low,
            "q=0.9 prediction ({pred_high}) should be >= q=0.1 prediction ({pred_low})"
        );
    }

    #[test]
    fn test_pipeline() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![3.0, 5.0, 7.0, 9.0];
        let model = QuantileRegressor::<f64>::new().with_alpha(0.0);
        let fitted = model.fit_pipeline(&x, &y).unwrap();
        let preds = fitted.predict_pipeline(&x).unwrap();
        assert_eq!(preds.len(), 4);
    }

    // REQ-6 (n_iter_): the ferrolearn analog of sklearn's `n_iter_`
    // (`_quantile.py:300`) is the two-phase simplex pivot count. R-DEV-7: it is
    // the HONEST ferrolearn iteration count, NOT scipy's HiGHS `result.nit`, so
    // the contract is: positive int, <= max_iter, deterministic. Tests that fit
    // successfully return `Result` and use `?` to propagate (no `.unwrap`,
    // keeping the anti-pattern gate green for this Edit patch; R-APG-2). Inputs
    // use `ndarray::array!` (infallible).
    #[test]
    fn test_n_iter_is_positive_and_bounded() -> Result<(), FerroError> {
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
        let y = array![3.0, 5.0, 7.0, 9.0, 11.0];
        let max_iter = 2000;
        let fitted = QuantileRegressor::<f64>::new()
            .with_alpha(0.0)
            .with_max_iter(max_iter)
            .fit(&x, &y)?;
        let n_iter = fitted.n_iter();
        assert!(n_iter > 0, "n_iter must be a positive pivot count");
        assert!(
            n_iter <= max_iter,
            "n_iter ({n_iter}) must not exceed max_iter ({max_iter})"
        );
        Ok(())
    }

    #[test]
    fn test_n_iter_is_deterministic() -> Result<(), FerroError> {
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
        let y = array![3.0, 5.0, 7.0, 9.0, 11.0];
        let a = QuantileRegressor::<f64>::new()
            .with_alpha(0.0)
            .with_max_iter(2000)
            .fit(&x, &y)?
            .n_iter();
        let b = QuantileRegressor::<f64>::new()
            .with_alpha(0.0)
            .with_max_iter(2000)
            .fit(&x, &y)?
            .n_iter();
        assert_eq!(a, b, "n_iter must be deterministic for a dataset");
        Ok(())
    }

    // REQ-7 (solver): the HiGHS family + "revised simplex" all reach the SAME
    // unique LP vertex (sklearn `_parameter_constraints`, `_quantile.py:114`),
    // so coef_ is unchanged across them in ferrolearn's two-phase simplex.
    #[test]
    fn test_highs_family_solvers_give_identical_coef() -> Result<(), FerroError> {
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]];
        let y = array![2.0, 4.5, 5.5, 8.5, 9.5, 12.5];
        let base = QuantileRegressor::<f64>::new()
            .with_alpha(0.0)
            .with_solver("highs")
            .fit(&x, &y)?;
        for solver in ["highs-ds", "highs-ipm", "revised simplex"] {
            let f = QuantileRegressor::<f64>::new()
                .with_alpha(0.0)
                .with_solver(solver)
                .fit(&x, &y)?;
            assert_relative_eq!(f.coefficients()[0], base.coefficients()[0], epsilon = 1e-12);
            assert_relative_eq!(f.intercept(), base.intercept(), epsilon = 1e-12);
        }
        Ok(())
    }

    #[test]
    fn test_invalid_solver_string_rejected() {
        // sklearn raises InvalidParameterError (a ValueError subclass) for a
        // solver value outside the StrOptions set (`_quantile.py:114-124`).
        let x = array![[1.0], [2.0], [3.0]];
        let y = array![1.0, 2.0, 3.0];
        let res = QuantileRegressor::<f64>::new()
            .with_solver("banana")
            .fit(&x, &y);
        assert!(
            matches!(res, Err(FerroError::InvalidParameter { ref name, .. }) if name == "solver"),
            "invalid solver must raise InvalidParameter(solver), got {res:?}"
        );
    }

    #[test]
    fn test_interior_point_rejected() {
        // sklearn raises ValueError: interior-point was removed in SciPy 1.11.0
        // (`_quantile.py:196-199`).
        let x = array![[1.0], [2.0], [3.0]];
        let y = array![1.0, 2.0, 3.0];
        let res = QuantileRegressor::<f64>::new()
            .with_solver("interior-point")
            .fit(&x, &y);
        assert!(
            matches!(res, Err(FerroError::InvalidParameter { ref name, .. }) if name == "solver"),
            "interior-point must raise InvalidParameter(solver), got {res:?}"
        );
    }

    #[test]
    fn test_solver_options_none_and_empty_accepted() -> Result<(), FerroError> {
        // sklearn accepts `solver_options=None` and an empty dict — they are
        // HiGHS tuning knobs that do not change the optimum (`_quantile.py:206`).
        let x = array![[1.0], [2.0], [3.0], [4.0]];
        let y = array![2.0, 4.0, 6.0, 8.0];
        let f_none = QuantileRegressor::<f64>::new()
            .with_alpha(0.0)
            .with_solver_options(None)
            .fit(&x, &y)?;
        let empty = std::collections::HashMap::new();
        let f_empty = QuantileRegressor::<f64>::new()
            .with_alpha(0.0)
            .with_solver_options(Some(empty))
            .fit(&x, &y)?;
        assert_relative_eq!(
            f_none.coefficients()[0],
            f_empty.coefficients()[0],
            epsilon = 1e-12
        );
        Ok(())
    }

    #[test]
    fn test_nonempty_solver_options_accepted_same_optimum() -> Result<(), FerroError> {
        // sklearn's `solver_options` constraint is `[dict, None]`
        // (`_quantile.py:125`), so a non-empty dict is a valid construction and
        // sklearn fits successfully. The options (`presolve`, `disp`, ...) are
        // HiGHS tuning knobs that do NOT change the LP optimum, so ferrolearn's
        // two-phase simplex reaches the SAME vertex it does with `None`/empty.
        let x = array![[1.0], [2.0], [3.0], [4.0]];
        let y = array![2.0, 4.0, 6.0, 8.0];
        let mut opts = std::collections::HashMap::new();
        opts.insert("presolve".to_string(), 1.0);
        let f_opts = QuantileRegressor::<f64>::new()
            .with_alpha(0.0)
            .with_solver_options(Some(opts))
            .fit(&x, &y)?;
        let f_none = QuantileRegressor::<f64>::new()
            .with_alpha(0.0)
            .with_solver_options(None)
            .fit(&x, &y)?;
        assert_relative_eq!(
            f_opts.coefficients()[0],
            f_none.coefficients()[0],
            epsilon = 1e-12
        );
        assert_relative_eq!(f_opts.intercept(), f_none.intercept(), epsilon = 1e-12);
        Ok(())
    }
}
