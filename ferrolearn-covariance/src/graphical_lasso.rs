//! Sparse inverse-covariance estimation via the graphical lasso.
//!
//! [`GraphicalLasso`] estimates a sparse precision matrix by maximising the
//! L1-penalised Gaussian log-likelihood
//!
//! ```text
//! max_{P ≻ 0}   log|P| - tr(S P) - alpha * ||P||_1
//! ```
//!
//! where `S` is the empirical covariance and `alpha` is the L1 penalty.
//! [`GraphicalLassoCV`] picks `alpha` by k-fold cross-validation over a grid.
//!
//! The solver (`solve_glasso`) is a faithful port of scikit-learn 1.5.2's
//! `_graphical_lasso` (`mode="cd"`): initialise with the `0.95`-shrunk empirical
//! covariance (diagonal restored) and a `pinvh` precision seed; per feature, run
//! a warm-started Gram coordinate-descent lasso, then update the precision and
//! covariance blocks; stop on the dual-gap criterion. See `solve_glasso` for
//! the line-by-line correspondence.
//!
//! ## REQ status
//!
//! Mirrors `sklearn.covariance` `GraphicalLasso`/`GraphicalLassoCV`/`graphical_lasso`
//! (`sklearn/covariance/_graph_lasso.py`) at v1.5.2. Every REQ is BINARY
//! (R-DEFER-2): SHIPPED or NOT-STARTED (with a concrete blocker).
//!
//! | REQ | Status | Notes |
//! |---|---|---|
//! | REQ-GLASSO-VALUE (covariance_/precision_ value parity) | SHIPPED | `solve_glasso` ports `_graphical_lasso` mode='cd' (`:70-200`): `0.95`-shrunk init + restored diagonal + `pinvh` seed, per-feature warm-started Gram-CD, dual-gap convergence. Matches the live oracle to ≤1e-12 across alpha∈{0.01..0.5}, p=4 and p=6, well-conditioned AND ill-conditioned (was 1.7e-2/2.7e-1, fixed #1880). Tests `divergence_graphical_lasso_value_alpha_0_1`, `_precision_diag_alpha_0_2`, sparsity guard. |
//! | REQ-GLASSO-FN (graphical_lasso function) | SHIPPED | the free `graphical_lasso(emp_cov, alpha, max_iter, tol)` shares `solve_glasso`; matches live `sklearn.covariance.graphical_lasso`. Test `divergence_graphical_lasso_function_alpha_0_1`. |
//! | REQ-CONVERGENCE (outer dual-gap + inner enet duality-gap) | SHIPPED | outer breaks on `|_dual_gap| < tol` (`:57-66`,`:184`); inner Gram-CD uses sklearn's duality-gap stop `gap < enet_tol*y_norm2` (max-change gate, `_cd_fast.pyx:685-727`) — was Frobenius-W (outer) + max-change-only (inner), fixed #1881/#1889. n_iter matches sklearn `n_iter_`. Test `divergence_graphical_lasso_ill_conditioned_inner_cd_alpha_0_01`. |
//! | REQ-ENET-TOL (separate public enet_tol param) | NOT-STARTED | inner uses `enet_tol=1e-4` (sklearn default — value-correct), but it is not exposed as a public parameter (`:425`). Blocker #1883. |
//! | REQ-GLASSOCV-CV (CV fold scheme + scoring) | NOT-STARTED | sequential consecutive-row folds (n_folds=3); sklearn `cv=None` ⇒ KFold(5); per-fold log-likelihood scoring now consumes the correct precision. Downstream of #1884 defaults. |
//! | REQ-GLASSOCV-REFINEMENT (adaptive alpha grid + n_refinements) | NOT-STARTED | sklearn refines the alpha grid around the best `n_refinements=4` times from an `alpha_max`-generated grid; ferrolearn takes an explicit `alphas` Vec, no refinement. Architectural. Blocker #1882. |
//! | REQ-DEFAULTS (alpha=0.01, max_iter=100, n_folds 3-vs-5, n_refinements=4) | NOT-STARTED | `new(alpha)` requires alpha; GraphicalLassoCV requires explicit alphas + n_folds=3. Blocker #1884. |
//! | REQ-MODE-LARS (mode='lars') | NOT-STARTED | CD-only; sklearn supports `mode='lars'` (`:151-161`). Blocker #1885. |
//! | REQ-REFIT-ESTIMATOR (GraphicalLassoCV cv_results_/alpha_ attrs) | NOT-STARTED | has `best_alpha`/`cv_scores`; sklearn exposes `alpha_`/`cv_results_`. Blocker #1886. |
//! | REQ-X-1 (R-SUBSTRATE) | NOT-STARTED | `ndarray` + hand-rolled Cholesky/pinvh CD; destination `ferray-core`/`ferray::linalg`. Blocker #1887. |
//! | REQ-X-2 (non-test production consumer) | SHIPPED | re-exported `pub use graphical_lasso::{GraphicalLasso, GraphicalLassoCV, graphical_lasso}` in `lib.rs` (boundary API, S5/R-DEFER-1). |

use ferrolearn_core::FerroError;
use ferrolearn_core::traits::Fit;
use ndarray::{Array1, Array2};
use num_traits::Float;

use crate::helpers::empirical_covariance;

/// Sparse inverse-covariance estimator (graphical lasso).
#[derive(Debug, Clone)]
pub struct GraphicalLasso<F> {
    /// L1 regularisation strength.
    alpha: F,
    /// Outer-loop iteration cap.
    max_iter: usize,
    /// Coordinate-descent inner iteration cap.
    max_inner_iter: usize,
    /// Convergence tolerance on the dual gap between outer iterations.
    tol: F,
    /// If `true`, skip mean centering during the empirical covariance step.
    assume_centered: bool,
}

impl<F: Float + Send + Sync + 'static> GraphicalLasso<F> {
    /// Construct a new [`GraphicalLasso`] with the given L1 penalty.
    #[must_use]
    pub fn new(alpha: F) -> Self {
        Self {
            alpha,
            max_iter: 100,
            max_inner_iter: 100,
            tol: F::from(1e-4).unwrap_or(F::epsilon()),
            assume_centered: false,
        }
    }

    /// Set the maximum number of outer iterations (default `100`).
    #[must_use]
    pub fn max_iter(mut self, n: usize) -> Self {
        self.max_iter = n;
        self
    }

    /// Set the maximum number of inner coordinate-descent iterations
    /// per column (default `100`).
    #[must_use]
    pub fn max_inner_iter(mut self, n: usize) -> Self {
        self.max_inner_iter = n;
        self
    }

    /// Set the outer convergence tolerance (default `1e-4`).
    #[must_use]
    pub fn tol(mut self, tol: F) -> Self {
        self.tol = tol;
        self
    }

    /// If `true`, skip mean centering during the empirical covariance step.
    #[must_use]
    pub fn assume_centered(mut self, value: bool) -> Self {
        self.assume_centered = value;
        self
    }
}

/// A fitted [`GraphicalLasso`] model.
#[derive(Debug, Clone)]
pub struct FittedGraphicalLasso<F> {
    covariance: Array2<F>,
    precision: Array2<F>,
    location: Array1<F>,
    n_iter: usize,
}

impl<F: Float + Send + Sync + 'static> FittedGraphicalLasso<F> {
    /// The estimated covariance matrix.
    pub fn covariance(&self) -> &Array2<F> {
        &self.covariance
    }
    /// The estimated precision matrix (inverse covariance).
    pub fn precision(&self) -> &Array2<F> {
        &self.precision
    }
    /// The per-feature mean used during centering.
    pub fn location(&self) -> &Array1<F> {
        &self.location
    }
    /// Number of outer iterations run.
    pub fn n_iter(&self) -> usize {
        self.n_iter
    }
}

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, ()> for GraphicalLasso<F> {
    type Fitted = FittedGraphicalLasso<F>;
    type Error = FerroError;

    fn fit(&self, x: &Array2<F>, _y: &()) -> Result<FittedGraphicalLasso<F>, FerroError> {
        let n = x.nrows();
        let p = x.ncols();
        if n < 2 || p < 2 {
            return Err(FerroError::InsufficientSamples {
                required: 2,
                actual: n.min(p),
                context: "GraphicalLasso requires n >= 2 and p >= 2".into(),
            });
        }

        // Per-column means (used both for location and centering when needed).
        let n_f = F::from(n).ok_or_else(|| FerroError::InvalidParameter {
            name: "n".into(),
            reason: "could not convert".into(),
        })?;
        let mut mean = Array1::<F>::zeros(p);
        if !self.assume_centered {
            for j in 0..p {
                let s = x.column(j).iter().copied().fold(F::zero(), |a, b| a + b);
                mean[j] = s / n_f;
            }
        }

        let emp_cov = empirical_covariance(x, self.assume_centered)?;
        let (cov, prec, n_iter) = solve_glasso(
            &emp_cov,
            self.alpha,
            self.max_iter,
            self.max_inner_iter,
            self.tol,
        );

        Ok(FittedGraphicalLasso {
            covariance: cov,
            precision: prec,
            location: mean,
            n_iter,
        })
    }
}

/// Cross-validated [`GraphicalLasso`].
#[derive(Debug, Clone)]
pub struct GraphicalLassoCV<F> {
    alphas: Vec<F>,
    n_folds: usize,
    max_iter: usize,
    tol: F,
    assume_centered: bool,
}

impl<F: Float + Send + Sync + 'static> GraphicalLassoCV<F> {
    /// Construct a new cross-validated graphical lasso over the given alpha
    /// grid.
    #[must_use]
    pub fn new(alphas: Vec<F>) -> Self {
        Self {
            alphas,
            n_folds: 3,
            max_iter: 100,
            tol: F::from(1e-4).unwrap_or(F::epsilon()),
            assume_centered: false,
        }
    }

    /// Set the number of CV folds (default `3`).
    #[must_use]
    pub fn n_folds(mut self, n: usize) -> Self {
        self.n_folds = n;
        self
    }

    /// Set the per-fit max iterations (default `100`).
    #[must_use]
    pub fn max_iter(mut self, n: usize) -> Self {
        self.max_iter = n;
        self
    }

    /// Set the per-fit tolerance (default `1e-4`).
    #[must_use]
    pub fn tol(mut self, tol: F) -> Self {
        self.tol = tol;
        self
    }

    /// If `true`, skip mean centering during the empirical covariance step.
    #[must_use]
    pub fn assume_centered(mut self, value: bool) -> Self {
        self.assume_centered = value;
        self
    }
}

/// A fitted [`GraphicalLassoCV`] model.
#[derive(Debug, Clone)]
pub struct FittedGraphicalLassoCV<F> {
    inner: FittedGraphicalLasso<F>,
    best_alpha: F,
    cv_scores: Vec<F>,
}

impl<F: Float + Send + Sync + 'static> FittedGraphicalLassoCV<F> {
    /// The chosen alpha that maximised the CV log-likelihood.
    pub fn best_alpha(&self) -> F {
        self.best_alpha
    }
    /// The per-fold per-alpha mean log-likelihoods (one entry per `alpha`).
    pub fn cv_scores(&self) -> &[F] {
        &self.cv_scores
    }
    /// The estimated covariance matrix at the chosen alpha.
    pub fn covariance(&self) -> &Array2<F> {
        self.inner.covariance()
    }
    /// The estimated precision matrix at the chosen alpha.
    pub fn precision(&self) -> &Array2<F> {
        self.inner.precision()
    }
    /// The per-feature mean used during centering.
    pub fn location(&self) -> &Array1<F> {
        self.inner.location()
    }
    /// Number of outer iterations run on the chosen alpha refit.
    pub fn n_iter(&self) -> usize {
        self.inner.n_iter()
    }
}

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, ()> for GraphicalLassoCV<F> {
    type Fitted = FittedGraphicalLassoCV<F>;
    type Error = FerroError;

    fn fit(&self, x: &Array2<F>, _y: &()) -> Result<FittedGraphicalLassoCV<F>, FerroError> {
        if self.alphas.is_empty() {
            return Err(FerroError::InvalidParameter {
                name: "alphas".into(),
                reason: "GraphicalLassoCV: alpha grid must be non-empty".into(),
            });
        }
        let n = x.nrows();
        if self.n_folds < 2 || n < self.n_folds {
            return Err(FerroError::InvalidParameter {
                name: "n_folds".into(),
                reason: format!(
                    "GraphicalLassoCV: need n_folds in [2, n_samples]; got {} folds and {n} samples",
                    self.n_folds
                ),
            });
        }
        let p = x.ncols();
        let fold_size = n / self.n_folds;

        let mut scores: Vec<F> = Vec::with_capacity(self.alphas.len());
        for &alpha in &self.alphas {
            let mut acc = F::zero();
            let mut count = 0usize;
            for fold in 0..self.n_folds {
                let lo = fold * fold_size;
                let hi = if fold + 1 == self.n_folds {
                    n
                } else {
                    lo + fold_size
                };

                let mut train = Vec::with_capacity((n - (hi - lo)) * p);
                let mut test = Vec::with_capacity((hi - lo) * p);
                for i in 0..n {
                    let row = x.row(i);
                    if (lo..hi).contains(&i) {
                        for v in row.iter() {
                            test.push(*v);
                        }
                    } else {
                        for v in row.iter() {
                            train.push(*v);
                        }
                    }
                }
                let train_arr = Array2::from_shape_vec((n - (hi - lo), p), train).map_err(|e| {
                    FerroError::InvalidParameter {
                        name: "fold".into(),
                        reason: format!("could not reshape train fold: {e}"),
                    }
                })?;
                let test_arr = Array2::from_shape_vec((hi - lo, p), test).map_err(|e| {
                    FerroError::InvalidParameter {
                        name: "fold".into(),
                        reason: format!("could not reshape test fold: {e}"),
                    }
                })?;
                let model = GraphicalLasso::<F>::new(alpha)
                    .max_iter(self.max_iter)
                    .tol(self.tol)
                    .assume_centered(self.assume_centered);
                let fitted = model.fit(&train_arr, &())?;
                let test_emp = empirical_covariance(&test_arr, self.assume_centered)?;
                let ll = crate::helpers::log_likelihood(&test_emp, fitted.precision())?;
                acc = acc + ll;
                count += 1;
            }
            let mean = acc
                / F::from(count).ok_or_else(|| FerroError::InvalidParameter {
                    name: "n_folds".into(),
                    reason: "could not convert".into(),
                })?;
            scores.push(mean);
        }

        // Pick the alpha that maximises CV log-likelihood.
        let mut best_idx = 0usize;
        for i in 1..scores.len() {
            if scores[i] > scores[best_idx] {
                best_idx = i;
            }
        }
        let best_alpha = self.alphas[best_idx];

        let model = GraphicalLasso::<F>::new(best_alpha)
            .max_iter(self.max_iter)
            .tol(self.tol)
            .assume_centered(self.assume_centered);
        let inner = model.fit(x, &())?;

        Ok(FittedGraphicalLassoCV {
            inner,
            best_alpha,
            cv_scores: scores,
        })
    }
}

/// Function-style equivalent of [`GraphicalLasso::fit`] returning
/// `(covariance, precision)`.
pub fn graphical_lasso<F>(
    emp_cov: &Array2<F>,
    alpha: F,
    max_iter: usize,
    tol: F,
) -> Result<(Array2<F>, Array2<F>), FerroError>
where
    F: Float + Send + Sync + 'static,
{
    let n = emp_cov.nrows();
    if n != emp_cov.ncols() {
        return Err(FerroError::ShapeMismatch {
            expected: vec![n, n],
            actual: vec![emp_cov.nrows(), emp_cov.ncols()],
            context: "graphical_lasso: emp_cov must be square".into(),
        });
    }
    let (cov, prec, _n_iter) = solve_glasso(emp_cov, alpha, max_iter, 100, tol);
    Ok((cov, prec))
}

// ---------------------------------------------------------------------------
// Graphical-lasso solver — faithful port of scikit-learn 1.5.2
// `sklearn/covariance/_graph_lasso.py:70-200` (`_graphical_lasso`, mode="cd").
// ---------------------------------------------------------------------------

/// Solve the graphical-lasso problem for an empirical covariance `emp_cov`.
///
/// This is a faithful port of scikit-learn's `_graphical_lasso` (`mode="cd"`).
/// The algorithm:
///
/// * **Init** (`_graph_lasso.py:91-104`): `covariance_ = emp_cov * 0.95` with the
///   empirical diagonal restored, then `precision_ = pinvh(covariance_)`. The
///   `covariance_` diagonal stays equal to the empirical diagonal for the whole
///   run (no `W = S + alpha*I` shift — only off-diagonals move).
/// * **Outer loop** (`:120-185`): for each feature `idx`, warm-start the inner
///   lasso from `-precision_[!=idx, idx] / (precision_[idx,idx] + 1000*eps)`
///   (`:135-138`), run a Gram coordinate-descent lasso
///   ([`enet_coordinate_descent_gram`], `:139-150`) over `sub_covariance` (the
///   `idx`-deleted `covariance_`) and `row = emp_cov[idx, !=idx]`, then update
///   `precision_` (`:163-168`) and `covariance_` off-diagonals (`:169-171`).
/// * **Convergence** (`:176-185`): break when `|dual_gap| < tol`, with
///   `dual_gap = sum(emp_cov*precision_) - p + alpha*(|precision_|.sum()
///   - |diag(precision_)|.sum())` (`_dual_gap`, `:57-66`).
///
/// `max_inner_iter` caps the inner Gram coordinate descent (sklearn passes the
/// outer `max_iter`; either works as long as it is large enough to converge).
/// Returns `(covariance_, precision_, n_iter)`.
fn solve_glasso<F: Float>(
    emp_cov: &Array2<F>,
    alpha: F,
    max_iter: usize,
    max_inner_iter: usize,
    tol: F,
) -> (Array2<F>, Array2<F>, usize) {
    let p = emp_cov.nrows();
    let eps = F::epsilon();

    // alpha == 0: early return without regularisation (`:83-89`).
    if alpha == F::zero() {
        let precision = match spd_inverse(emp_cov) {
            Ok(inv) => inv,
            // pinvh-style fallback: a symmetric pseudo-inverse via Jacobi eigen.
            Err(()) => pinvh(emp_cov),
        };
        return (emp_cov.clone(), precision, 0);
    }

    // Init: shrink off-diagonals to 0.95, restore the empirical diagonal,
    // seed precision_ = pinvh(covariance_) (`:91-104`).
    let mut covariance = emp_cov.clone();
    let scale = F::from(0.95).unwrap_or_else(F::one);
    for a in 0..p {
        for b in 0..p {
            covariance[[a, b]] = covariance[[a, b]] * scale;
        }
    }
    for i in 0..p {
        covariance[[i, i]] = emp_cov[[i, i]];
    }
    // covariance_ is SPD after the 0.95-shrink + diagonal restore; use a
    // Cholesky-based inverse, falling back to the symmetric pseudo-inverse.
    let mut precision = match spd_inverse(&covariance) {
        Ok(inv) => inv,
        Err(()) => pinvh(&covariance),
    };

    let thousand_eps = F::from(1000.0).unwrap_or_else(F::one) * eps;
    let enet_tol = F::from(1e-4).unwrap_or(eps);

    let mut iter = 0usize;
    for it in 0..max_iter {
        iter = it + 1;
        for idx in 0..p {
            // sub_covariance = covariance_ with row/column `idx` removed.
            let mut sub_cov = Array2::<F>::zeros((p - 1, p - 1));
            let mut ia = 0usize;
            for a in 0..p {
                if a == idx {
                    continue;
                }
                let mut ib = 0usize;
                for b in 0..p {
                    if b == idx {
                        continue;
                    }
                    sub_cov[[ia, ib]] = covariance[[a, b]];
                    ib += 1;
                }
                ia += 1;
            }
            // row = emp_cov[idx, indices != idx].
            let mut row = Array1::<F>::zeros(p - 1);
            let mut k = 0usize;
            for a in 0..p {
                if a == idx {
                    continue;
                }
                row[k] = emp_cov[[a, idx]];
                k += 1;
            }

            // Warm start: coefs = -precision_[!=idx, idx] / (precision_[idx,idx]
            // + 1000*eps)  (`:135-138`).
            let denom_ws = precision[[idx, idx]] + thousand_eps;
            let mut coefs = Array1::<F>::zeros(p - 1);
            let mut k = 0usize;
            for a in 0..p {
                if a == idx {
                    continue;
                }
                coefs[k] = -(precision[[a, idx]] / denom_ws);
                k += 1;
            }

            // Inner Gram lasso (`:139-150`).
            enet_coordinate_descent_gram(
                &mut coefs,
                alpha,
                &sub_cov,
                &row,
                max_inner_iter,
                enet_tol,
            );

            // Update precision_ (`:163-168`). Use the CURRENT covariance_ column
            // (before the covariance update below).
            let mut dot = F::zero();
            let mut k = 0usize;
            for a in 0..p {
                if a == idx {
                    continue;
                }
                dot = dot + covariance[[a, idx]] * coefs[k];
                k += 1;
            }
            let prec_ii = F::one() / (covariance[[idx, idx]] - dot);
            precision[[idx, idx]] = prec_ii;
            let mut k = 0usize;
            for a in 0..p {
                if a == idx {
                    continue;
                }
                let v = -prec_ii * coefs[k];
                precision[[a, idx]] = v;
                precision[[idx, a]] = v;
                k += 1;
            }

            // Update covariance_ off-diagonals: coefs_full = sub_cov @ coefs
            // (`:169-171`).
            let mut k = 0usize;
            for a in 0..p {
                if a == idx {
                    continue;
                }
                let mut acc = F::zero();
                for b in 0..(p - 1) {
                    acc = acc + sub_cov[[k, b]] * coefs[b];
                }
                covariance[[a, idx]] = acc;
                covariance[[idx, a]] = acc;
                k += 1;
            }
        }

        // Dual-gap convergence (`_dual_gap` :57-66, break :184).
        let mut gap = F::zero();
        let mut abs_sum = F::zero();
        let mut abs_diag = F::zero();
        for a in 0..p {
            for b in 0..p {
                gap = gap + emp_cov[[a, b]] * precision[[a, b]];
                abs_sum = abs_sum + precision[[a, b]].abs();
            }
            abs_diag = abs_diag + precision[[a, a]].abs();
        }
        let pf = F::from(p).unwrap_or_else(F::zero);
        let d_gap = gap - pf + alpha * (abs_sum - abs_diag);
        if d_gap.abs() < tol {
            break;
        }
    }

    (covariance, precision, iter)
}

/// Gram coordinate-descent lasso — port of scikit-learn's
/// `enet_coordinate_descent_gram` (`linear_model/_cd_fast.pyx`) with `l2 = 0`.
///
/// Minimises `0.5 w^T Q w - q^T w + alpha*||w||_1` over `w` in place, starting
/// from the warm-started `w`. `Q` is the Gram matrix (`sub_covariance`), `q` is
/// the right-hand side (`row`); the graphical-lasso caller passes `row` as both
/// `q` AND the target `y`.
///
/// Convergence follows sklearn's duality-gap criterion (`_cd_fast.pyx:685-727`):
/// the max-coefficient-change rule `w_max == 0 || d_w_max / w_max < enet_tol`
/// (or the last iteration) is only a GATE; when gated, the full enet duality
/// gap is computed and the loop breaks only when `gap < enet_tol * y_norm2`
/// (`:635` rescales `tol` by `y_norm2 = dot(y, y)`, `:725` breaks on
/// `gap < tol`). On near-singular Gram matrices this runs longer than (or stops
/// differently from) the max-change rule alone.
fn enet_coordinate_descent_gram<F: Float>(
    w: &mut Array1<F>,
    alpha: F,
    q: &Array2<F>,
    qvec: &Array1<F>,
    max_iter: usize,
    enet_tol: F,
) {
    let n = w.len();
    let half = F::from(0.5).unwrap_or_else(|| F::one() / (F::one() + F::one()));
    let two = F::one() + F::one();

    // H = Q @ w.
    let mut h = Array1::<F>::zeros(n);
    for a in 0..n {
        let mut acc = F::zero();
        for b in 0..n {
            acc = acc + q[[a, b]] * w[b];
        }
        h[a] = acc;
    }

    // y_norm2 = dot(y, y) with y = qvec; tol_scaled = enet_tol * y_norm2
    // (`_cd_fast.pyx:629,635`).
    let mut y_norm2 = F::zero();
    for i in 0..n {
        y_norm2 = y_norm2 + qvec[i] * qvec[i];
    }
    let tol_scaled = enet_tol * y_norm2;

    for it in 0..max_iter {
        let mut w_max = F::zero();
        let mut d_w_max = F::zero();
        for j in 0..n {
            let q_jj = q[[j, j]];
            if q_jj == F::zero() {
                continue;
            }
            let w_j = w[j];
            // Gradient excluding coordinate j: H[j] = (Q@w)[j] includes
            // Q[j,j]*w[j], so add it back to remove j's self-term.
            let tmp = qvec[j] - h[j] + q_jj * w_j;
            let new_wj = soft_threshold(tmp, alpha) / q_jj;
            w[j] = new_wj;
            if new_wj != w_j {
                let delta = new_wj - w_j;
                for a in 0..n {
                    h[a] = h[a] + q[[a, j]] * delta;
                }
            }
            let change = (new_wj - w_j).abs();
            if change > d_w_max {
                d_w_max = change;
            }
            let aw = new_wj.abs();
            if aw > w_max {
                w_max = aw;
            }
        }

        // GATE: max-change rule or last iteration (`_cd_fast.pyx:685`).
        if w_max == F::zero() || d_w_max / w_max < enet_tol || it == max_iter - 1 {
            // Duality gap (`_cd_fast.pyx:690-723`), beta = 0.
            // q_dot_w = dot(w, q)
            let mut q_dot_w = F::zero();
            for i in 0..n {
                q_dot_w = q_dot_w + w[i] * qvec[i];
            }
            // XtA[i] = q[i] - H[i] (beta term = 0); dual_norm_XtA = max|XtA|.
            let mut dual_norm_xta = F::zero();
            for i in 0..n {
                let a = (qvec[i] - h[i]).abs();
                if a > dual_norm_xta {
                    dual_norm_xta = a;
                }
            }
            // tmp = sum(w * H) = w^T Q w; R_norm2 = y_norm2 + w^T Q w - 2*q_dot_w.
            let mut wh = F::zero();
            for i in 0..n {
                wh = wh + w[i] * h[i];
            }
            let r_norm2 = y_norm2 + wh - two * q_dot_w;
            // l1 = ||w||_1.
            let mut l1 = F::zero();
            for i in 0..n {
                l1 = l1 + w[i].abs();
            }

            let (const_, mut gap) = if dual_norm_xta > alpha {
                let c = alpha / dual_norm_xta;
                (c, half * (r_norm2 + r_norm2 * c * c))
            } else {
                (F::one(), r_norm2)
            };
            // gap += alpha*||w||_1 - const*y_norm2 + const*q_dot_w (beta term = 0).
            gap = gap + alpha * l1 - const_ * y_norm2 + const_ * q_dot_w;

            if gap < tol_scaled {
                break;
            }
        }
    }
}

/// Soft-thresholding operator `sign(x) * max(|x| - gamma, 0)`.
#[inline]
fn soft_threshold<F: Float>(x: F, gamma: F) -> F {
    if x > gamma {
        x - gamma
    } else if x < -gamma {
        x + gamma
    } else {
        F::zero()
    }
}

/// Invert a symmetric positive-definite matrix via its Cholesky factor
/// `A = L L^T`, solving `A X = I` column by column. Returns `Err(())` if `A`
/// is not strictly positive definite (a non-positive pivot is encountered).
fn spd_inverse<F: Float>(a: &Array2<F>) -> Result<Array2<F>, ()> {
    let n = a.nrows();
    // Lower Cholesky factor L.
    let mut l = Array2::<F>::zeros((n, n));
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[[i, j]];
            for k in 0..j {
                sum = sum - l[[i, k]] * l[[j, k]];
            }
            if i == j {
                if sum <= F::zero() {
                    return Err(());
                }
                l[[i, j]] = sum.sqrt();
            } else {
                l[[i, j]] = sum / l[[j, j]];
            }
        }
    }

    // Solve A x = e_c for each column c via forward/back substitution.
    let mut inv = Array2::<F>::zeros((n, n));
    for c in 0..n {
        // Forward solve L y = e_c.
        let mut y = Array1::<F>::zeros(n);
        for i in 0..n {
            let mut sum = if i == c { F::one() } else { F::zero() };
            for k in 0..i {
                sum = sum - l[[i, k]] * y[k];
            }
            y[i] = sum / l[[i, i]];
        }
        // Back solve L^T x = y.
        for i in (0..n).rev() {
            let mut sum = y[i];
            for k in (i + 1)..n {
                sum = sum - l[[k, i]] * inv[[k, c]];
            }
            inv[[i, c]] = sum / l[[i, i]];
        }
    }
    Ok(inv)
}

/// Symmetric pseudo-inverse (`scipy.linalg.pinvh` analogue) via a Jacobi
/// eigendecomposition: `A = V diag(λ) V^T`, then `A^+ = V diag(1/λ) V^T` with
/// eigenvalues below `eps * max|λ|` zeroed. Used as a fallback when the
/// Cholesky inverse fails (matrix not strictly positive definite).
fn pinvh<F: Float>(a: &Array2<F>) -> Array2<F> {
    let n = a.nrows();
    // Jacobi eigenvalue algorithm on a copy of the symmetric matrix.
    let mut m = a.clone();
    let mut v = Array2::<F>::zeros((n, n));
    for i in 0..n {
        v[[i, i]] = F::one();
    }
    let max_sweeps = 100usize;
    for _ in 0..max_sweeps {
        // Largest off-diagonal magnitude.
        let mut off = F::zero();
        for p in 0..n {
            for q in (p + 1)..n {
                off = off + m[[p, q]] * m[[p, q]];
            }
        }
        if off.sqrt() <= F::epsilon() {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                if m[[p, q]] == F::zero() {
                    continue;
                }
                let app = m[[p, p]];
                let aqq = m[[q, q]];
                let apq = m[[p, q]];
                let two = F::one() + F::one();
                let theta = (aqq - app) / (two * apq);
                let t = {
                    let sign = if theta >= F::zero() {
                        F::one()
                    } else {
                        -F::one()
                    };
                    sign / (theta.abs() + (theta * theta + F::one()).sqrt())
                };
                let c = F::one() / (t * t + F::one()).sqrt();
                let s = t * c;
                // Rotate rows/columns p, q.
                for k in 0..n {
                    let mkp = m[[k, p]];
                    let mkq = m[[k, q]];
                    m[[k, p]] = c * mkp - s * mkq;
                    m[[k, q]] = s * mkp + c * mkq;
                }
                for k in 0..n {
                    let mpk = m[[p, k]];
                    let mqk = m[[q, k]];
                    m[[p, k]] = c * mpk - s * mqk;
                    m[[q, k]] = s * mpk + c * mqk;
                }
                for k in 0..n {
                    let vkp = v[[k, p]];
                    let vkq = v[[k, q]];
                    v[[k, p]] = c * vkp - s * vkq;
                    v[[k, q]] = s * vkp + c * vkq;
                }
            }
        }
    }
    // Eigenvalues are the diagonal of m; eigenvectors are columns of v.
    let mut max_eig = F::zero();
    for i in 0..n {
        let e = m[[i, i]].abs();
        if e > max_eig {
            max_eig = e;
        }
    }
    let cutoff = F::epsilon() * max_eig;
    let mut inv = Array2::<F>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            let mut acc = F::zero();
            for k in 0..n {
                let lam = m[[k, k]];
                if lam.abs() > cutoff {
                    acc = acc + v[[i, k]] * (F::one() / lam) * v[[j, k]];
                }
            }
            inv[[i, j]] = acc;
        }
    }
    inv
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn data() -> Array2<f64> {
        // 10 samples × 3 features with mild correlation.
        array![
            [1.0, 2.0, 1.5],
            [3.0, 4.0, 3.5],
            [5.0, 6.0, 5.5],
            [7.0, 8.0, 7.5],
            [2.0, 3.0, 2.5],
            [4.0, 5.0, 4.5],
            [6.0, 7.0, 6.5],
            [8.0, 9.0, 8.5],
            [1.5, 2.5, 2.0],
            [9.0, 10.0, 9.5],
        ]
    }

    #[test]
    fn test_graphical_lasso_basic() {
        let est = GraphicalLasso::<f64>::new(0.1).max_iter(50).tol(1e-3);
        let fitted = est.fit(&data(), &()).unwrap();
        assert_eq!(fitted.covariance().dim(), (3, 3));
        assert_eq!(fitted.precision().dim(), (3, 3));
        assert!(fitted.n_iter() > 0);
    }

    #[test]
    fn test_graphical_lasso_function() {
        let emp = empirical_covariance(&data(), false).unwrap();
        let (cov, prec) = graphical_lasso(&emp, 0.1, 50, 1e-3).unwrap();
        assert_eq!(cov.dim(), (3, 3));
        assert_eq!(prec.dim(), (3, 3));
    }

    #[test]
    fn test_graphical_lasso_cv() {
        let est = GraphicalLassoCV::<f64>::new(vec![0.05, 0.1, 0.2])
            .n_folds(2)
            .max_iter(20)
            .tol(1e-2);
        let fitted = est.fit(&data(), &()).unwrap();
        assert_eq!(fitted.covariance().dim(), (3, 3));
        assert!(fitted.cv_scores().len() == 3);
        let alpha = fitted.best_alpha();
        assert!([0.05, 0.1, 0.2].contains(&alpha));
    }

    #[test]
    fn test_graphical_lasso_too_small() {
        let x: Array2<f64> = array![[1.0]];
        assert!(GraphicalLasso::<f64>::new(0.1).fit(&x, &()).is_err());
    }
}
