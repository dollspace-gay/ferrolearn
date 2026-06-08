//! Mini-Batch Non-negative Matrix Factorization (MiniBatchNMF).
//!
//! [`MiniBatchNMF`] decomposes a non-negative matrix `X` into two non-negative
//! factors `W` and `H` such that `X ~ W * H`, processing the data in
//! mini-batches for scalability to large datasets.
//!
//! # Algorithm
//!
//! 1. Initialise `W` and `H` (random or NNDSVD).
//! 2. For each mini-batch `X_batch`:
//!    a. Fix `H`, update `W_batch` via coordinate descent on
//!    `||X_batch - W_batch @ H||^2`.
//!    b. Fix `W`, update `H` via multiplicative update:
//!    `H *= (W^T X_batch) / (W^T W H + eps)`.
//!    c. Online averaging of `W` across batches.
//!
//! # Examples
//!
//! ```
//! use ferrolearn_decomp::MiniBatchNMF;
//! use ferrolearn_core::traits::{Fit, Transform};
//! use ndarray::array;
//!
//! let nmf = MiniBatchNMF::<f64>::new(2);
//! let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
//! let fitted = nmf.fit(&x, &()).unwrap();
//! let projected = fitted.transform(&x).unwrap();
//! assert_eq!(projected.ncols(), 2);
//! ```
//!
//! ## REQ status
//!
//! Design: `.design/decomp/minibatch_nmf.md`. Tracking: #1485. Each REQ is BINARY —
//! SHIPPED (impl + non-test consumer + tests + green verification) or NOT-STARTED
//! (concrete open blocker). Non-test consumer: crate re-export (`lib.rs:96`); there
//! is NO PyO3 binding. Oracle = live sklearn 1.5.2 (`_nmf.py`, `class MiniBatchNMF`),
//! run from `/tmp` (R-CHAR-3). ferrolearn is a SIMPLIFIED reimplementation (5-iter
//! coordinate-descent W + plain MU H, deterministic batching, no forget_factor/EWA),
//! so component VALUES are a carve-out (different algorithm + RNG).
//!
//! | REQ | Scope | Status | Evidence / Blocker |
//! |---|---|---|---|
//! | REQ-1 | Structural: components shape `(n_components,n_features)`, finite `reconstruction_err_`, `n_iter_` in `[1, max_iter]`, seed-determinism | SHIPPED (scoped) | `fit` (`minibatch_nmf.rs:365`) stores `components_` shape, finite Frobenius `reconstruction_err_`, positive `n_iter_`; seeded `StdRng` ⇒ deterministic. Green-guards in `tests/divergence_minibatch_nmf.rs` + in-module tests. STRUCTURAL only, NOT component values (REQ-4) |
//! | REQ-2 | Non-negativity of `components_` (H) and `transform` (W) — NMF invariant | SHIPPED | `fit` clamps H ≥ 0 (`:468-470`); `update_w_batch` clamps W ≥ 0 (`:340-344`). Green-guard `guard_nonnegative` + `test_minibatch_nmf_components_nonnegative` |
//! | REQ-3 | Error / parameter contracts (n_components 0/>n_features, negative input, 0 samples, transform col mismatch, NON-FINITE rejection) | SHIPPED (scoped) | `fit` guards; `transform` `ShapeMismatch`. NON-FINITE: `fit`+`transform` call `reject_non_finite` (`minibatch_nmf.rs` symbol `reject_non_finite`) BEFORE the non-negativity check and the factorization, returning the CLEAN finiteness `InvalidParameter{name:"X", reason:"Input X contains NaN or infinity."}` = sklearn `_validate_data(force_all_finite=True)` finiteness-before-non-negativity (`_nmf.py:2236`,`:2407`,`utils/validation.py:147-154`) — the real input gate (the `is_finite` at `:664` is TEST-only). `tests/divergence_nonfinite_spillover.rs::divergence_minibatch_nmf_fit_nan`/`_transform_nan` match the live sklearn 1.5.2 oracle (#2290). FLAG: sklearn raises `InvalidParameterError`, accepts `n_components=None`, does not pre-reject `n_components>n_features` |
//! | REQ-4 | EXACT `components_` value parity with sklearn online MU | NOT-STARTED | CARVE-OUT (R-DEFER-3): NNDSVDa + EWA aggregates A/B + `forget_factor` rho + numpy RNG (`_nmf.py:2254-2349`) vs ferrolearn CD-W + plain-MU-H + deterministic batching — blocker #1486 |
//! | REQ-5 | `transform` = `_solve_W` MU formula | NOT-STARTED | CARVE-OUT, folds into REQ-4: critic confirmed (live oracle) ferrolearn's 5-iter CD reaches the SAME convex NNLS optimum as sklearn `_solve_W` (residual match relative ~1.4e-5 on its own H); not observable — blocker #1487 |
//! | REQ-6 | `beta_loss` (kullback-leibler/itakura-saito) + `_gamma` | NOT-STARTED | sklearn `_nmf.py:2011,:2057-2062,:89`; ferrolearn is Frobenius-only — blocker #1488 |
//! | REQ-7 | `solver` / multiplicative-update for W | NOT-STARTED | sklearn MU `_multiplicative_update_w` (`_nmf.py:530,:2118`); ferrolearn uses 5-iter coordinate descent — blocker #1489 |
//! | REQ-8 | `forget_factor`/`_rho` + EWA online aggregates A/B for H | NOT-STARTED | sklearn `_nmf.py:2054,:2130-2141,:2312-2313`; ferrolearn H update is plain per-batch MU (no A/B, no rho) — blocker #1490 |
//! | REQ-9 | `_minibatch_convergence` EWA cost + `max_no_improvement` early stop | NOT-STARTED | sklearn `_nmf.py:2149-2208`; ferrolearn uses whole-X relative-reconstruction-error stop — blocker #1491 |
//! | REQ-10 | Real NNDSVDa SVD init + `random_state` batch shuffle | NOT-STARTED | sklearn `_nmf.py:225,:2308,:2319-2320`; ferrolearn `init_nndsvd_simple` is power-iteration pseudo-NNDSVD, deterministic `rotate_left` batching — blocker #1492 |
//! | REQ-11 | Regularization `alpha_W`/`alpha_H`/`l1_ratio` | NOT-STARTED | sklearn `_nmf.py:2013-2016,:1275`; ferrolearn unpenalised — blocker #1493 |
//! | REQ-12 | `fresh_restarts`/`fresh_restarts_max_iter` | NOT-STARTED | sklearn `_nmf.py:2019-2020,:2117`; ferrolearn warm-continues W per batch — blocker #1494 |
//! | REQ-13 | `partial_fit` online out-of-core fit | NOT-STARTED | sklearn `_nmf.py:2373+`; ferrolearn has only batch `Fit::fit` — blocker #1495 |
//! | REQ-14 | Fitted attrs `n_components_`/`n_features_in_`/`n_steps_` | NOT-STARTED | `FittedMiniBatchNMF` exposes only `components()`/`reconstruction_err()`/`n_iter()` — blocker #1496 |
//! | REQ-15 | PyO3 binding | NOT-STARTED | no `_RsMiniBatchNMF`; only consumer is the re-export `lib.rs:96` — blocker #1497 |
//! | REQ-16 | ferray substrate | NOT-STARTED | dense `ndarray::Array2` + `rand` `StdRng` — blocker #1498 |
//!
//! Count: **3 SHIPPED (REQ-1,2,3) / 13 NOT-STARTED (REQ-4..16)**.

use ferrolearn_core::error::FerroError;
use ferrolearn_core::traits::{Fit, Transform};
use ndarray::Array2;
use num_traits::Float;
use rand::SeedableRng;
use rand_distr::{Distribution, Uniform};

/// Reject non-finite input the way sklearn's `_validate_data` does.
///
/// sklearn runs `check_array` with the default `force_all_finite=True` at the
/// top of `MiniBatchNMF.fit`/`fit_transform`/`transform`
/// (`sklearn/decomposition/_nmf.py:2236`,`:2407`), raising
/// `ValueError("Input X contains NaN.")` / `"... contains infinity ..."`
/// (`sklearn/utils/validation.py:147-154`) BEFORE the non-negativity check and
/// any factorization math. NaN AND infinity are both rejected, and finiteness
/// is checked BEFORE non-negativity (sklearn's `_validate_data` → `check_array`
/// finiteness runs before `_check_X`'s non-negative `check_non_negative`). The
/// message names "NaN" and "infinity" to mirror sklearn's `ValueError`. Never
/// panics (R-CODE-2). (The `is_finite` check in `#[cfg(test)]` below is
/// test-only and does NOT gate input — this guard is the real input gate.)
fn reject_non_finite<F: Float>(x: &Array2<F>) -> Result<(), FerroError> {
    if x.iter().any(|v| !v.is_finite()) {
        return Err(FerroError::InvalidParameter {
            name: "X".into(),
            reason: "Input X contains NaN or infinity.".into(),
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Configuration enums
// ---------------------------------------------------------------------------

/// Initialisation strategy for `MiniBatchNMF`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MiniBatchNMFInit {
    /// Random non-negative initialisation.
    Random,
    /// Non-Negative Double SVD initialisation (simplified).
    Nndsvd,
}

// ---------------------------------------------------------------------------
// MiniBatchNMF (unfitted)
// ---------------------------------------------------------------------------

/// Mini-Batch NMF configuration.
///
/// Holds hyperparameters for the mini-batch NMF decomposition. Calling
/// [`Fit::fit`] performs the iterative procedure and returns a
/// [`FittedMiniBatchNMF`] that can project new data.
#[derive(Debug, Clone)]
pub struct MiniBatchNMF<F> {
    /// Number of components to extract.
    n_components: usize,
    /// Maximum number of iterations over the full dataset.
    max_iter: usize,
    /// Mini-batch size.
    batch_size: usize,
    /// Convergence tolerance.
    tol: f64,
    /// Optional random seed for reproducibility.
    random_state: Option<u64>,
    /// Initialisation strategy.
    init: MiniBatchNMFInit,
    _marker: std::marker::PhantomData<F>,
}

impl<F: Float + Send + Sync + 'static> MiniBatchNMF<F> {
    /// Create a new `MiniBatchNMF` that extracts `n_components` components.
    ///
    /// Defaults: `max_iter = 200`, `batch_size = 1024`, `tol = 1e-4`,
    /// `init = Random`, `random_state = None`.
    #[must_use]
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            max_iter: 200,
            batch_size: 1024,
            tol: 1e-4,
            random_state: None,
            init: MiniBatchNMFInit::Random,
            _marker: std::marker::PhantomData,
        }
    }

    /// Set the maximum number of iterations.
    #[must_use]
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the mini-batch size.
    #[must_use]
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Set the convergence tolerance.
    #[must_use]
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Set the random seed for reproducible results.
    #[must_use]
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Set the initialisation strategy.
    #[must_use]
    pub fn with_init(mut self, init: MiniBatchNMFInit) -> Self {
        self.init = init;
        self
    }

    /// Return the configured number of components.
    #[must_use]
    pub fn n_components(&self) -> usize {
        self.n_components
    }

    /// Return the configured maximum iterations.
    #[must_use]
    pub fn max_iter(&self) -> usize {
        self.max_iter
    }

    /// Return the configured batch size.
    #[must_use]
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Return the configured tolerance.
    #[must_use]
    pub fn tol(&self) -> f64 {
        self.tol
    }

    /// Return the configured initialisation strategy.
    #[must_use]
    pub fn init(&self) -> MiniBatchNMFInit {
        self.init
    }
}

// ---------------------------------------------------------------------------
// FittedMiniBatchNMF
// ---------------------------------------------------------------------------

/// A fitted Mini-Batch NMF model holding the learned components.
///
/// Created by calling [`Fit::fit`] on a [`MiniBatchNMF`]. Implements
/// [`Transform<Array2<F>>`] to project new data onto the learned components.
#[derive(Debug, Clone)]
pub struct FittedMiniBatchNMF<F> {
    /// Learned component matrix H, shape `(n_components, n_features)`.
    components_: Array2<F>,
    /// Frobenius norm of the reconstruction error at convergence.
    reconstruction_err_: F,
    /// Number of iterations performed.
    n_iter_: usize,
}

impl<F: Float + Send + Sync + 'static> FittedMiniBatchNMF<F> {
    /// Learned components (H matrix), shape `(n_components, n_features)`.
    #[must_use]
    pub fn components(&self) -> &Array2<F> {
        &self.components_
    }

    /// Frobenius norm of the reconstruction error `||X - W*H||_F`.
    #[must_use]
    pub fn reconstruction_err(&self) -> F {
        self.reconstruction_err_
    }

    /// Number of iterations performed during fitting.
    #[must_use]
    pub fn n_iter(&self) -> usize {
        self.n_iter_
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Small epsilon to prevent division by zero.
#[inline]
fn eps<F: Float>() -> F {
    F::from(1e-12).unwrap_or_else(F::epsilon)
}

/// Initialise W and H with random non-negative values.
fn init_random<F: Float>(
    n_samples: usize,
    n_features: usize,
    n_components: usize,
    seed: u64,
) -> (Array2<F>, Array2<F>) {
    let mut rng: rand::rngs::StdRng = SeedableRng::seed_from_u64(seed);
    let uniform = Uniform::new(0.0f64, 1.0f64).unwrap();

    let mut w = Array2::<F>::zeros((n_samples, n_components));
    for elem in w.iter_mut() {
        *elem = F::from(uniform.sample(&mut rng)).unwrap_or_else(F::zero) + eps::<F>();
    }

    let mut h = Array2::<F>::zeros((n_components, n_features));
    for elem in h.iter_mut() {
        *elem = F::from(uniform.sample(&mut rng)).unwrap_or_else(F::zero) + eps::<F>();
    }

    (w, h)
}

/// Simplified NNDSVD initialisation: compute `X^T X`, use the top eigenvectors.
fn init_nndsvd_simple<F: Float + Send + Sync + 'static>(
    x: &Array2<F>,
    n_components: usize,
    seed: u64,
) -> (Array2<F>, Array2<F>) {
    let (n_samples, n_features) = x.dim();

    // Compute column means for scale.
    let mut avg = F::zero();
    for &v in x.iter() {
        avg = avg + v.abs();
    }
    avg = avg / F::from(n_samples * n_features).unwrap();
    if avg < eps::<F>() {
        avg = F::one();
    }
    let scale = avg.sqrt();

    // Compute X^T X.
    let xtx = x.t().dot(x);

    // Simple power iteration to find dominant eigenvectors.
    let mut rng: rand::rngs::StdRng = SeedableRng::seed_from_u64(seed);
    let uniform = Uniform::new(0.0f64, 1.0f64).unwrap();

    let mut h = Array2::<F>::zeros((n_components, n_features));

    for k in 0..n_components {
        // Random initial vector.
        let mut v = Array2::<F>::zeros((n_features, 1));
        for elem in v.iter_mut() {
            *elem = F::from(uniform.sample(&mut rng)).unwrap_or_else(F::one);
        }

        // Power iteration (20 steps).
        for _ in 0..20 {
            let v_new = xtx.dot(&v);
            let norm: F = v_new.iter().fold(F::zero(), |a, &b| a + b * b).sqrt();
            if norm > eps::<F>() {
                for (dst, &src) in v.iter_mut().zip(v_new.iter()) {
                    *dst = src / norm;
                }
            }
        }

        // Clamp negatives to zero and store as row of H.
        for j in 0..n_features {
            let val = v[[j, 0]];
            h[[k, j]] = if val > F::zero() {
                val
            } else {
                eps::<F>() * scale
            };
        }
    }

    // W = X * H^T, clamped non-negative.
    let w_raw = x.dot(&h.t());
    let mut w = Array2::<F>::zeros((n_samples, n_components));
    for i in 0..n_samples {
        for k in 0..n_components {
            let val = w_raw[[i, k]];
            w[[i, k]] = if val > F::zero() { val } else { eps::<F>() };
        }
    }

    (w, h)
}

/// Compute the Frobenius norm of `X - W * H`.
fn reconstruction_error<F: Float + 'static>(x: &Array2<F>, w: &Array2<F>, h: &Array2<F>) -> F {
    let wh = w.dot(h);
    let mut err = F::zero();
    for (a, b) in x.iter().zip(wh.iter()) {
        let diff = *a - *b;
        err = err + diff * diff;
    }
    err.sqrt()
}

/// Solve for W_batch via coordinate descent on `||X_batch - W_batch @ H||^2`,
/// keeping H fixed. All values in W_batch are clamped non-negative.
fn update_w_batch<F: Float + 'static>(x_batch: &Array2<F>, w_batch: &mut Array2<F>, h: &Array2<F>) {
    let n_batch = x_batch.nrows();
    let n_components = h.nrows();
    let n_features = h.ncols();
    let epsilon = eps::<F>();

    // Pre-compute H * H^T for efficiency.
    let hht = h.dot(&h.t());

    for _cd_iter in 0..5 {
        for i in 0..n_batch {
            for k in 0..n_components {
                // Compute numerator: sum_j x[i,j] * h[k,j] - sum_{l!=k} w[i,l] * hht[l,k]
                let mut num = F::zero();
                for j in 0..n_features {
                    num = num + x_batch[[i, j]] * h[[k, j]];
                }
                for l in 0..n_components {
                    if l != k {
                        num = num - w_batch[[i, l]] * hht[[l, k]];
                    }
                }

                let den = hht[[k, k]] + epsilon;
                let new_val = num / den;
                w_batch[[i, k]] = if new_val > F::zero() {
                    new_val
                } else {
                    F::zero()
                };
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Trait implementations
// ---------------------------------------------------------------------------

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, ()> for MiniBatchNMF<F> {
    type Fitted = FittedMiniBatchNMF<F>;
    type Error = FerroError;

    /// Fit Mini-Batch NMF by iterating over mini-batches.
    ///
    /// # Errors
    ///
    /// - [`FerroError::InvalidParameter`] if `n_components` is zero or exceeds
    ///   the number of features, or if the data contains negative values.
    /// - [`FerroError::InsufficientSamples`] if there are no samples.
    fn fit(&self, x: &Array2<F>, _y: &()) -> Result<FittedMiniBatchNMF<F>, FerroError> {
        let (n_samples, n_features) = x.dim();

        if self.n_components == 0 {
            return Err(FerroError::InvalidParameter {
                name: "n_components".into(),
                reason: "must be at least 1".into(),
            });
        }
        if self.n_components > n_features {
            return Err(FerroError::InvalidParameter {
                name: "n_components".into(),
                reason: format!(
                    "n_components ({}) exceeds n_features ({})",
                    self.n_components, n_features
                ),
            });
        }
        if n_samples == 0 {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "MiniBatchNMF::fit requires at least 1 sample".into(),
            });
        }

        // Reject NaN/Inf BEFORE the non-negativity check and the factorization
        // (sklearn's `_validate_data(force_all_finite=True)` at `_nmf.py:2236`
        // runs `check_array` finiteness BEFORE `_check_X`'s non-negativity,
        // `utils/validation.py:147-154`).
        reject_non_finite(x)?;

        // Check non-negativity.
        for &v in x.iter() {
            if v < F::zero() {
                return Err(FerroError::InvalidParameter {
                    name: "X".into(),
                    reason: "MiniBatchNMF requires non-negative input data".into(),
                });
            }
        }

        let n_comp = self.n_components;
        let seed = self.random_state.unwrap_or(42);
        let epsilon = eps::<F>();

        // Initialise W and H.
        let (mut w, mut h) = match self.init {
            MiniBatchNMFInit::Random => init_random(n_samples, n_features, n_comp, seed),
            MiniBatchNMFInit::Nndsvd => init_nndsvd_simple(x, n_comp, seed),
        };

        let batch_size = self.batch_size.min(n_samples);
        let tol_f = F::from(self.tol).unwrap_or_else(F::epsilon);
        let mut prev_err = reconstruction_error(x, &w, &h);
        let mut actual_iter = 0;

        // Use a simple permutation for batching.
        let mut indices: Vec<usize> = (0..n_samples).collect();

        for iteration in 0..self.max_iter {
            actual_iter = iteration + 1;

            // Simple rotation of indices (deterministic).
            indices.rotate_left(batch_size % n_samples.max(1));

            // Process batches.
            let mut batch_start = 0;
            while batch_start < n_samples {
                let batch_end = (batch_start + batch_size).min(n_samples);
                let batch_indices: Vec<usize> = indices[batch_start..batch_end].to_vec();
                let actual_batch = batch_indices.len();

                // Extract X_batch.
                let mut x_batch = Array2::<F>::zeros((actual_batch, n_features));
                for (bi, &idx) in batch_indices.iter().enumerate() {
                    for j in 0..n_features {
                        x_batch[[bi, j]] = x[[idx, j]];
                    }
                }

                // Extract W_batch.
                let mut w_batch = Array2::<F>::zeros((actual_batch, n_comp));
                for (bi, &idx) in batch_indices.iter().enumerate() {
                    for k in 0..n_comp {
                        w_batch[[bi, k]] = w[[idx, k]];
                    }
                }

                // Update W_batch (fix H, solve for W_batch).
                update_w_batch(&x_batch, &mut w_batch, &h);

                // Write back W_batch.
                for (bi, &idx) in batch_indices.iter().enumerate() {
                    for k in 0..n_comp {
                        w[[idx, k]] = w_batch[[bi, k]];
                    }
                }

                // Update H via multiplicative update: H *= (W^T X_batch) / (W^T W H + eps).
                let wt = w_batch.t();
                let numerator_h = wt.dot(&x_batch);
                let denominator_h = wt.dot(&w_batch).dot(&h);

                for k in 0..n_comp {
                    for j in 0..n_features {
                        let num = numerator_h[[k, j]];
                        let den = denominator_h[[k, j]] + epsilon;
                        h[[k, j]] = h[[k, j]] * (num / den);
                        if h[[k, j]] < F::zero() {
                            h[[k, j]] = epsilon;
                        }
                    }
                }

                batch_start = batch_end;
            }

            // Check convergence.
            let err = reconstruction_error(x, &w, &h);
            if prev_err > epsilon && (prev_err - err).abs() / prev_err < tol_f {
                break;
            }
            prev_err = err;
        }

        let final_err = reconstruction_error(x, &w, &h);

        Ok(FittedMiniBatchNMF {
            components_: h,
            reconstruction_err_: final_err,
            n_iter_: actual_iter,
        })
    }
}

impl<F: Float + Send + Sync + 'static> Transform<Array2<F>> for FittedMiniBatchNMF<F> {
    type Output = Array2<F>;
    type Error = FerroError;

    /// Project new data onto the learned NMF components.
    ///
    /// Solves `min_{W >= 0} ||X - W H||_F^2` for W using coordinate descent.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of columns does not
    /// match the number of features seen during fitting.
    fn transform(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let n_features = self.components_.ncols();
        if x.ncols() != n_features {
            return Err(FerroError::ShapeMismatch {
                expected: vec![x.nrows(), n_features],
                actual: vec![x.nrows(), x.ncols()],
                context: "FittedMiniBatchNMF::transform".into(),
            });
        }

        // Reject NaN/Inf BEFORE solving for W (sklearn re-validates with
        // `_validate_data(force_all_finite=True)`, `utils/validation.py:147-154`).
        reject_non_finite(x)?;

        let n_samples = x.nrows();
        let n_comp = self.components_.nrows();
        let mut w = Array2::<F>::zeros((n_samples, n_comp));
        // Initialise W with uniform values.
        let init_val = F::from(0.1).unwrap_or_else(F::one);
        w.fill(init_val);

        update_w_batch(x, &mut w, &self.components_);
        Ok(w)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_minibatch_nmf_basic() {
        let nmf = MiniBatchNMF::<f64>::new(2).with_random_state(42);
        let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let fitted = nmf.fit(&x, &()).unwrap();
        let projected = fitted.transform(&x).unwrap();
        assert_eq!(projected.dim(), (3, 2));
    }

    #[test]
    fn test_minibatch_nmf_components_shape() {
        let nmf = MiniBatchNMF::<f64>::new(3).with_random_state(0);
        let x = array![
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ];
        let fitted = nmf.fit(&x, &()).unwrap();
        assert_eq!(fitted.components().dim(), (3, 4));
    }

    #[test]
    fn test_minibatch_nmf_nndsvd_init() {
        let nmf = MiniBatchNMF::<f64>::new(2)
            .with_init(MiniBatchNMFInit::Nndsvd)
            .with_random_state(42);
        let x = array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
        ];
        let fitted = nmf.fit(&x, &()).unwrap();
        assert_eq!(fitted.components().nrows(), 2);
    }

    #[test]
    fn test_minibatch_nmf_components_nonnegative() {
        let nmf = MiniBatchNMF::<f64>::new(2).with_random_state(7);
        let x = array![
            [1.0, 2.0, 0.0],
            [0.0, 5.0, 6.0],
            [7.0, 0.0, 9.0],
            [0.0, 0.0, 1.0],
        ];
        let fitted = nmf.fit(&x, &()).unwrap();
        for &v in fitted.components().iter() {
            assert!(v >= 0.0, "negative component value: {v}");
        }
    }

    #[test]
    fn test_minibatch_nmf_negative_input_error() {
        let nmf = MiniBatchNMF::<f64>::new(1);
        let x = array![[1.0, -2.0], [3.0, 4.0]];
        assert!(nmf.fit(&x, &()).is_err());
    }

    #[test]
    fn test_minibatch_nmf_zero_components_error() {
        let nmf = MiniBatchNMF::<f64>::new(0);
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        assert!(nmf.fit(&x, &()).is_err());
    }

    #[test]
    fn test_minibatch_nmf_too_many_components_error() {
        let nmf = MiniBatchNMF::<f64>::new(5);
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        assert!(nmf.fit(&x, &()).is_err());
    }

    #[test]
    fn test_minibatch_nmf_empty_data() {
        let nmf = MiniBatchNMF::<f64>::new(1);
        let x = Array2::<f64>::zeros((0, 3));
        assert!(nmf.fit(&x, &()).is_err());
    }

    #[test]
    fn test_minibatch_nmf_transform_shape_mismatch() {
        let nmf = MiniBatchNMF::<f64>::new(1).with_random_state(0);
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let fitted = nmf.fit(&x, &()).unwrap();
        let x_bad = array![[1.0, 2.0, 3.0]];
        assert!(fitted.transform(&x_bad).is_err());
    }

    #[test]
    fn test_minibatch_nmf_reconstruction_err_positive() {
        let nmf = MiniBatchNMF::<f64>::new(1).with_random_state(42);
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let fitted = nmf.fit(&x, &()).unwrap();
        assert!(fitted.reconstruction_err() >= 0.0);
        assert!(fitted.reconstruction_err().is_finite());
    }

    #[test]
    fn test_minibatch_nmf_n_iter_positive() {
        let nmf = MiniBatchNMF::<f64>::new(1)
            .with_max_iter(5)
            .with_random_state(0);
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let fitted = nmf.fit(&x, &()).unwrap();
        assert!(fitted.n_iter() > 0);
    }

    #[test]
    fn test_minibatch_nmf_small_batch() {
        let nmf = MiniBatchNMF::<f64>::new(1)
            .with_batch_size(2)
            .with_random_state(42);
        let x = array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0],
        ];
        let fitted = nmf.fit(&x, &()).unwrap();
        assert_eq!(fitted.components().nrows(), 1);
    }

    #[test]
    fn test_minibatch_nmf_f32() {
        let nmf = MiniBatchNMF::<f32>::new(1).with_random_state(0);
        let x: Array2<f32> = array![[1.0f32, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let fitted = nmf.fit(&x, &()).unwrap();
        let projected = fitted.transform(&x).unwrap();
        assert_eq!(projected.ncols(), 1);
    }

    #[test]
    fn test_minibatch_nmf_builder_methods() {
        let nmf = MiniBatchNMF::<f64>::new(3)
            .with_max_iter(100)
            .with_batch_size(512)
            .with_tol(1e-5)
            .with_init(MiniBatchNMFInit::Nndsvd)
            .with_random_state(99);
        assert_eq!(nmf.n_components(), 3);
        assert_eq!(nmf.max_iter(), 100);
        assert_eq!(nmf.batch_size(), 512);
        assert!((nmf.tol() - 1e-5).abs() < 1e-15);
        assert_eq!(nmf.init(), MiniBatchNMFInit::Nndsvd);
    }
}
