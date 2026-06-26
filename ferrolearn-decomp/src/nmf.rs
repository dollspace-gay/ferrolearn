//! Non-negative Matrix Factorization (NMF).
//!
//! [`NMF`] decomposes a non-negative matrix `X` into two non-negative
//! factors `W` and `H` such that `X ~ W * H`, where:
//! - `X` has shape `(n_samples, n_features)`
//! - `W` has shape `(n_samples, n_components)`
//! - `H` has shape `(n_components, n_features)`
//!
//! # Algorithm
//!
//! Two solvers are supported:
//!
//! - **Multiplicative Update** (Lee & Seung, 2001): iteratively update `W` and
//!   `H` using multiplicative rules that guarantee non-negativity.
//! - **Coordinate Descent**: iteratively solve for each element of `W` and `H`
//!   using closed-form coordinate-wise updates.
//!
//! # Initialization
//!
//! - **Random**: initialize `W` and `H` with random non-negative values.
//! - **NNDSVD**: Non-Negative Double SVD, initializes `W` and `H` from a
//!   truncated SVD of `X`, setting negative entries to zero.
//!
//! # Examples
//!
//! ```
//! use ferrolearn_decomp::NMF;
//! use ferrolearn_core::traits::{Fit, Transform};
//! use ndarray::array;
//!
//! let nmf = NMF::<f64>::new(2);
//! let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
//! let fitted = nmf.fit(&x, &()).unwrap();
//! let projected = fitted.transform(&x).unwrap();
//! assert_eq!(projected.ncols(), 2);
//! ```
//!
//! ## REQ status
//!
//! Design: `.design/decomp/nmf.md`. Tracking: #1608. Each REQ is BINARY — SHIPPED
//! (impl + non-test consumer + tests + green verification) or NOT-STARTED (concrete
//! open blocker). Includes the standalone `non_negative_factorization` helper.
//! Non-test consumers: crate re-export (`lib.rs:97`), the PyO3
//! `_RsNMF` binding (`ferrolearn-python/src/extras.rs:1116`, registered `lib.rs:75`),
//! `PipelineTransformer`. Oracle = live sklearn 1.5.2 (`_nmf.py`, `class NMF`), run
//! from `/tmp` (R-CHAR-3). ferrolearn's ctor still DEFAULTS to MU + Random init;
//! exact component VALUES on the RANDOM/MU path are a carve-out (numpy RNG vs Rust
//! RNG; NMF identifiable only up to permutation/scaling). BUT the DETERMINISTIC
//! `init='nndsvd', solver='cd'` path is now BIT-EXACT to sklearn (#2398/#2394/#2395/
//! #2396/#2397): the real SVD-based NNDSVD init (`init_nndsvd` via
//! `ferray::linalg::svd_lapack` + `svd_flip_u_based`), the violation-ratio CD
//! convergence (`solve_coordinate_descent`/`update_cd_sweep`), the CD transform
//! W-solve, and the standalone `non_negative_factorization` W/H/n_iter all
//! reproduce sklearn to ~1e-9 (`tests/divergence_nmf_cd_nndsvd_2393.rs`,
//! `tests/divergence_non_negative_factorization.rs`).
//!
//! | REQ | Scope | Status | Evidence / Blocker |
//! |---|---|---|---|
//! | REQ-1 | Structural: `components_` shape `(n_components,n_features)`, transform W shape, finite + decreasing `reconstruction_err_`, `n_iter_`, seed-determinism | SHIPPED (scoped) | `fit` (`nmf.rs:617`); green-guards + in-module tests. STRUCTURAL, NOT values (REQ-5) |
//! | REQ-2 | Non-negativity of `components_` (H) + transform (W) | SHIPPED | MU multiplicative + CD clamp; `test_nmf_components_non_negative`/`_transform_non_negative` |
//! | REQ-3 | Both solvers (MU/CD) × both inits (Random/NNDSVD) run | SHIPPED | `fit` dispatch (`:657-670`); 4-combo tests |
//! | REQ-4 | Reconstruction QUALITY (`‖X−WH‖` small/decreasing — "did NMF work") | SHIPPED | `reconstruction_error` (`:247`) monotone-decreasing + small residual; `test_nmf_*` |
//! | REQ-5 | EXACT `components_` value parity | SHIPPED (deterministic cd+nndsvd path) / NOT-STARTED (random/MU) | The DETERMINISTIC `init='nndsvd', solver='cd'` `components_` is bit-exact to sklearn — `divergence_components_cd_nndsvd` (`tests/divergence_nmf_cd_nndsvd_2393.rs`) matches `components_[0]` to 1e-6 (was #2394). The random-init/MU path stays a CARVE-OUT (numpy vs Rust RNG, perm/scale) — blocker #1609 |
//! | REQ-6 | real SVD `init='nndsvd'` (+ `nndsvda`/`nndsvdar`/`custom`, `nndsvda` default) | SHIPPED (`nndsvd`) / NOT-STARTED (nndsvda default + nndsvdar/custom) | `init_nndsvd` (`nmf.rs` symbol `init_nndsvd`) now does the REAL SVD-based NNDSVD = sklearn `_initialize_nmf` (`_nmf.py:320-358`): `svd_lapack` (LAPACK gesdd, the SAME driver `randomized_svd`/scipy use) → `svd_flip_u_based` (`extmath.py` `svd_flip(u_based_decision=True)`) → leading-triplet `sqrt(S[0])·\|U\|`/`sqrt(S[0])·\|Vt\|` + per-component pos/neg-part split `lbd=sqrt(S[j]·sigma)` → `<eps`-zero. `test_nmf_nndsvd_init_matches_sklearn` matches sklearn `_initialize_nmf(X,3,'nndsvd')` W/H to 1e-9. STILL NOT-STARTED: the `nndsvda`/`nndsvdar` zero-fill + `custom` + `init=None→nndsvda` default — blocker #1610 |
//! | REQ-7 | `solver='cd'` matching `_fit_coordinate_descent` (+ cd DEFAULT) | SHIPPED (cd algorithm) / NOT-STARTED (cd ctor DEFAULT) | `solve_coordinate_descent`/`update_cd_sweep` now match sklearn's Cython `_update_cdnmf_fast` (Gram-based per-coordinate `W[i,t]=max(0,W[i,t]−grad/hess)`, in-place sweep) + the VIOLATION-RATIO stop `violation/violation_init<=tol` (`_nmf.py:500-525`), reproducing sklearn `n_iter_` (`divergence_n_iter_cd_nndsvd` → 151) + `reconstruction_err_` (`divergence_reconstruction_err_cd_nndsvd` → 5.5136, 1e-6). STILL NOT-STARTED: `NMF::new` still defaults to `MultiplicativeUpdate` not `cd` — blocker #1611 |
//! | REQ-8 | `beta_loss` (kullback-leibler/itakura-saito) + `_gamma` | NOT-STARTED | sklearn `_nmf.py:89,:919`; ferrolearn Frobenius-only — blocker #1612 |
//! | REQ-9 | `transform` NNLS-W VALUE | SHIPPED | `transform` (`impl Transform for FittedNMF`) now solves W via the CD with H fixed = `_fit_transform(X, H=components_, update_H=False)` (`_nmf.py:1213`): W=zeros (`_nmf.py:1254`), violation-ratio CD. `divergence_transform_w_cd_nndsvd` matches sklearn `m.transform(X)[0]` to 1e-6 (was #2397) |
//! | REQ-10 | `inverse_transform` = `W·H` | SHIPPED | `nmf.rs:229` (`= _nmf.py:1238`); exact algebra + col-mismatch `ShapeMismatch` |
//! | REQ-11 | Error/parameter contracts (incl. NON-FINITE rejection, finiteness-before-nonnegative) | SHIPPED (scoped) | `fit`/`transform` guards. FLAG: sklearn raises `InvalidParameterError`, accepts `n_components=None`, doesn't pre-reject `>min(n,p)`. NON-FINITE: `fit`+`transform` call `reject_non_finite` (`nmf.rs` symbol `reject_non_finite`) BEFORE the non-negativity check and factorization, returning `InvalidParameter{name:"X", reason:"Input X contains NaN or infinity."}` = sklearn `_validate_data(force_all_finite=True)` (`_nmf.py:1652`) which runs BEFORE `check_non_negative` (`_nmf.py:1706`), so a NaN+negative input rejects for finiteness first (`utils/validation.py:147-154`). `tests/divergence_nonfinite.rs::divergence_nmf_fit_nan_`/`_fit_nan_and_negative_finiteness_fires_first` match the live sklearn 1.5.2 oracle. Was #2288/#2289, fixed. Consumer: re-export `lib.rs` + NMF `fit`/`transform` |
//! | REQ-12 | PyO3 `_RsNMF` binding (thin n_components ctor + fit + transform) | SHIPPED (scoped) | `extras.rs:1116`, registered `lib.rs:75`; NO params/getters/inverse_transform |
//! | REQ-13 | `n_components=None` default | NOT-STARTED | sklearn `_nmf.py:914` → min(n,p); ferrolearn requires explicit usize — blocker #1614 |
//! | REQ-14 | `alpha_W`/`alpha_H`/`l1_ratio` regularization | NOT-STARTED | sklearn `_nmf.py:921-923,:1275` — blocker #1615 |
//! | REQ-15 | `shuffle` (CD) + fitted attrs `n_components_`/`n_features_in_` | NOT-STARTED | sklearn `_nmf.py:924` — blocker #1616 |
//! | REQ-16 | ferray substrate | NOT-STARTED | `ndarray` + `rand` + hand-rolled Jacobi — blocker #1617 |
//!
//! Count: **9 SHIPPED (REQ-1,2,3,4,9,10,11,12 + REQ-6/7 cd+nndsvd algorithm) / 7
//! NOT-STARTED (REQ-8,13,14,15,16 + REQ-6 nndsvda-default/nndsvdar/custom + REQ-7
//! cd-ctor-default)**. REQ-5 is SHIPPED on the deterministic cd+nndsvd path,
//! NOT-STARTED on random/MU (carve-out #1609).

use ferrolearn_core::error::FerroError;
use ferrolearn_core::pipeline::{FittedPipelineTransformer, PipelineTransformer};
use ferrolearn_core::traits::{Fit, Transform};
use ndarray::{Array1, Array2};
use num_traits::Float;
use rand::SeedableRng;
use rand_distr::{Distribution, Uniform};
use std::any::TypeId;

/// Reject non-finite input the way sklearn's `_validate_data` does.
///
/// sklearn runs `check_array` with the default `force_all_finite=True` at the
/// top of `NMF.fit`/`transform` (`_nmf.py:1652`), raising
/// `ValueError("Input X contains NaN.")` / `"... contains infinity ..."`
/// (`sklearn/utils/validation.py:147-154`) BEFORE `check_non_negative`
/// (`_nmf.py:1706`) and the factorization. The finiteness check therefore wins
/// even when a negative value is also present. NaN AND infinity both rejected.
/// The message names "NaN" and "infinity" to mirror sklearn. Never panics
/// (R-CODE-2).
fn reject_non_finite<F: Float>(x: &Array2<F>) -> Result<(), FerroError> {
    if x.iter().any(|v| !v.is_finite()) {
        return Err(FerroError::InvalidParameter {
            name: "X".into(),
            reason: "Input X contains NaN or infinity.".into(),
        });
    }
    Ok(())
}

/// Compute non-negative matrix factorization, returning `(W, H, n_iter)`.
///
/// This is the no-regularization Frobenius-loss path corresponding to
/// scikit-learn's `non_negative_factorization(..., update_H=true,
/// beta_loss="frobenius", alpha_W=0, alpha_H=0, l1_ratio=0, shuffle=false)`.
/// Pass [`NMFInit::Nndsvd`] with [`NMFSolver::CoordinateDescent`] for the
/// deterministic sklearn-parity path pinned by the divergence tests.
///
/// # Errors
///
/// Returns [`FerroError`] when input validation fails or NNDSVD initialization
/// cannot be computed.
pub fn non_negative_factorization<F: Float + Send + Sync + 'static>(
    x: &Array2<F>,
    n_components: usize,
    init: NMFInit,
    solver: NMFSolver,
    max_iter: usize,
    tol: f64,
    random_state: Option<u64>,
) -> Result<(Array2<F>, Array2<F>, usize), FerroError> {
    let (w, h, n_iter, _) =
        factorize_nmf(x, n_components, init, solver, max_iter, tol, random_state)?;
    Ok((w, h, n_iter))
}

// ---------------------------------------------------------------------------
// Configuration enums
// ---------------------------------------------------------------------------

/// The solver algorithm for NMF.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NMFSolver {
    /// Multiplicative update rules (Lee & Seung, 2001).
    MultiplicativeUpdate,
    /// Coordinate descent.
    CoordinateDescent,
}

/// The initialization strategy for NMF.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NMFInit {
    /// Random non-negative initialization.
    Random,
    /// Non-Negative Double SVD initialization.
    Nndsvd,
}

// ---------------------------------------------------------------------------
// NMF (unfitted)
// ---------------------------------------------------------------------------

/// Non-negative Matrix Factorization configuration.
///
/// Holds hyperparameters for the NMF decomposition. Calling [`Fit::fit`]
/// computes the factorization and returns a [`FittedNMF`] that can
/// project new data via [`Transform::transform`].
#[derive(Debug, Clone)]
pub struct NMF<F> {
    /// Number of components to extract.
    n_components: usize,
    /// Maximum number of iterations for the solver.
    max_iter: usize,
    /// Convergence tolerance for the solver.
    tol: f64,
    /// The solver algorithm to use.
    solver: NMFSolver,
    /// The initialization strategy.
    init: NMFInit,
    /// Optional random seed for reproducibility.
    random_state: Option<u64>,
    _marker: std::marker::PhantomData<F>,
}

impl<F: Float + Send + Sync + 'static> NMF<F> {
    /// Create a new `NMF` that extracts `n_components` components.
    ///
    /// Defaults: `max_iter=200`, `tol=1e-4`, solver=`MultiplicativeUpdate`,
    /// init=`Random`, no random seed.
    #[must_use]
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            max_iter: 200,
            tol: 1e-4,
            solver: NMFSolver::MultiplicativeUpdate,
            init: NMFInit::Random,
            random_state: None,
            _marker: std::marker::PhantomData,
        }
    }

    /// Set the maximum number of iterations.
    #[must_use]
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the convergence tolerance.
    #[must_use]
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Set the solver algorithm.
    #[must_use]
    pub fn with_solver(mut self, solver: NMFSolver) -> Self {
        self.solver = solver;
        self
    }

    /// Set the initialization strategy.
    #[must_use]
    pub fn with_init(mut self, init: NMFInit) -> Self {
        self.init = init;
        self
    }

    /// Set the random seed for reproducible results.
    #[must_use]
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
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

    /// Return the configured tolerance.
    #[must_use]
    pub fn tol(&self) -> f64 {
        self.tol
    }

    /// Return the configured solver.
    #[must_use]
    pub fn solver(&self) -> NMFSolver {
        self.solver
    }

    /// Return the configured initialization strategy.
    #[must_use]
    pub fn init(&self) -> NMFInit {
        self.init
    }

    /// Return the configured random state, if any.
    #[must_use]
    pub fn random_state(&self) -> Option<u64> {
        self.random_state
    }
}

// ---------------------------------------------------------------------------
// FittedNMF
// ---------------------------------------------------------------------------

/// A fitted NMF model holding the learned components and reconstruction error.
///
/// Created by calling [`Fit::fit`] on an [`NMF`]. Implements
/// [`Transform<Array2<F>>`] to project new data onto the learned components.
#[derive(Debug, Clone)]
pub struct FittedNMF<F> {
    /// Learned component matrix H, shape `(n_components, n_features)`.
    components_: Array2<F>,
    /// The Frobenius norm of the reconstruction error at convergence.
    reconstruction_err_: F,
    /// Number of iterations performed.
    n_iter_: usize,
}

impl<F: Float + Send + Sync + 'static> FittedNMF<F> {
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

    /// Reconstruct the original feature space from the latent representation.
    /// Mirrors sklearn `NMF.inverse_transform`. Returns `W @ H` where `W`
    /// is the input transformed matrix and `H = self.components_`.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if `w.ncols()` does not equal
    /// the number of components.
    pub fn inverse_transform(&self, w: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let n_components = self.components_.nrows();
        if w.ncols() != n_components {
            return Err(FerroError::ShapeMismatch {
                expected: vec![w.nrows(), n_components],
                actual: vec![w.nrows(), w.ncols()],
                context: "FittedNMF::inverse_transform".into(),
            });
        }
        Ok(w.dot(&self.components_))
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

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

fn validate_nmf_input<F: Float>(x: &Array2<F>, n_components: usize) -> Result<(), FerroError> {
    let (n_samples, n_features) = x.dim();

    if n_components == 0 {
        return Err(FerroError::InvalidParameter {
            name: "n_components".into(),
            reason: "must be at least 1".into(),
        });
    }
    if n_samples == 0 {
        return Err(FerroError::InsufficientSamples {
            required: 1,
            actual: 0,
            context: "NMF::fit".into(),
        });
    }
    if n_components > n_samples.min(n_features) {
        return Err(FerroError::InvalidParameter {
            name: "n_components".into(),
            reason: format!(
                "n_components ({n_components}) exceeds min(n_samples, n_features) = {}",
                n_samples.min(n_features)
            ),
        });
    }

    // Finiteness FIRST: sklearn `NMF.fit_transform` and
    // `non_negative_factorization` run `check_array` before
    // `check_non_negative`, so NaN/Inf wins even when negatives are present.
    reject_non_finite(x)?;

    for &val in x {
        if val < F::zero() {
            return Err(FerroError::InvalidParameter {
                name: "X".into(),
                reason: "NMF requires all entries in X to be non-negative".into(),
            });
        }
    }

    Ok(())
}

/// Small epsilon to prevent division by zero.
fn eps<F: Float>() -> F {
    F::from(1e-12).unwrap_or_else(F::epsilon)
}

/// Initialize W and H with random non-negative values.
fn init_random<F: Float>(
    n_samples: usize,
    n_features: usize,
    n_components: usize,
    seed: u64,
) -> (Array2<F>, Array2<F>) {
    let mut rng: rand::rngs::StdRng = SeedableRng::seed_from_u64(seed);
    let uniform = Uniform::new(0.0f64, 1.0f64).unwrap();

    let mut w = Array2::<F>::zeros((n_samples, n_components));
    for elem in &mut w {
        *elem = F::from(uniform.sample(&mut rng)).unwrap_or_else(F::zero) + eps::<F>();
    }

    let mut h = Array2::<F>::zeros((n_components, n_features));
    for elem in &mut h {
        *elem = F::from(uniform.sample(&mut rng)).unwrap_or_else(F::zero) + eps::<F>();
    }

    (w, h)
}

/// Bridge an `ndarray::Array2<f64>` into a `ferray` 2-D array.
fn ndarray_to_ferray_f64(a: &Array2<f64>) -> Result<ferray::Array<f64, ferray::Ix2>, FerroError> {
    let (m, n) = a.dim();
    let data: Vec<f64> = a.iter().copied().collect();
    ferray::Array::<f64, ferray::Ix2>::from_vec(ferray::Ix2::new([m, n]), data).map_err(|e| {
        FerroError::NumericalInstability {
            message: format!("ferray array construction failed: {e}"),
        }
    })
}

/// Bridge an `ndarray::Array2<f32>` into a `ferray` 2-D array.
fn ndarray_to_ferray_f32(a: &Array2<f32>) -> Result<ferray::Array<f32, ferray::Ix2>, FerroError> {
    let (m, n) = a.dim();
    let data: Vec<f32> = a.iter().copied().collect();
    ferray::Array::<f32, ferray::Ix2>::from_vec(ferray::Ix2::new([m, n]), data).map_err(|e| {
        FerroError::NumericalInstability {
            message: format!("ferray array construction failed: {e}"),
        }
    })
}

/// Full thin SVD of `a` (`m × n`) returning `(U, s, Vt)` via
/// `ferray::linalg::svd_lapack` (LAPACK `gesdd`). `U` is `(m, k)`, `s` length
/// `k`, `Vt` is `(k, n)` with `k = min(m, n)`. This is the SAME driver
/// `scipy.linalg.svd` calls and the SAME helper `pca.rs` uses, so the spectrum
/// and singular vectors are bit-identical to scipy/sklearn (up to per-vector
/// sign, fixed by [`svd_flip_u_based`]).
#[allow(
    clippy::type_complexity,
    reason = "(U, s, Vt) is the standard thin-SVD triple, not worth a named struct"
)]
fn svd_full_f64(a: &Array2<f64>) -> Result<(Array2<f64>, Array1<f64>, Array2<f64>), FerroError> {
    let fa = ndarray_to_ferray_f64(a)?;
    let (u, s, vt) =
        ferray::linalg::svd_lapack(&fa, false).map_err(|e| FerroError::NumericalInstability {
            message: format!("ferray svd_lapack (gesdd) failed: {e}"),
        })?;
    Ok((u.into_ndarray(), s.into_ndarray(), vt.into_ndarray()))
}

/// f32 analogue of [`svd_full_f64`].
#[allow(
    clippy::type_complexity,
    reason = "(U, s, Vt) is the standard thin-SVD triple, not worth a named struct"
)]
fn svd_full_f32(a: &Array2<f32>) -> Result<(Array2<f32>, Array1<f32>, Array2<f32>), FerroError> {
    let fa = ndarray_to_ferray_f32(a)?;
    let (u, s, vt) =
        ferray::linalg::svd_lapack(&fa, false).map_err(|e| FerroError::NumericalInstability {
            message: format!("ferray svd_lapack (gesdd) failed: {e}"),
        })?;
    Ok((u.into_ndarray(), s.into_ndarray(), vt.into_ndarray()))
}

/// sklearn's `svd_flip(U, Vt, u_based_decision=True)`
/// (`sklearn/utils/extmath.py`): for each column `k` of `U`, find the entry with
/// the largest absolute value (numpy `argmax` — first index on ties via strict
/// `>`); if that entry is negative, negate column `k` of `U` and row `k` of
/// `Vt`. This is the sign convention `randomized_svd` applies, so the
/// deterministic top-`k` full SVD reproduces sklearn's `_initialize_nmf`
/// `U`/`Vt` exactly.
fn svd_flip_u_based<F: Float>(u: &mut Array2<F>, vt: &mut Array2<F>) {
    let (m, k) = u.dim();
    let n = vt.ncols();
    for col in 0..k {
        let mut max_abs = F::neg_infinity();
        let mut max_row = 0usize;
        for row in 0..m {
            let av = u[[row, col]].abs();
            if av > max_abs {
                max_abs = av;
                max_row = row;
            }
        }
        if u[[max_row, col]] < F::zero() {
            for row in 0..m {
                u[[row, col]] = -u[[row, col]];
            }
            for j in 0..n {
                vt[[col, j]] = -vt[[col, j]];
            }
        }
    }
}

/// Compute the top-`n_components` `(U, S, Vt)` of `X` (sign-corrected by
/// [`svd_flip_u_based`]) for the NNDSVD init. f64/f32 route through
/// `ferray::linalg::svd_lapack` (LAPACK `gesdd`, sklearn's `gesdd` driver);
/// exotic `F` falls back to a Jacobi eigendecomposition of `XᵀX` reconstructing
/// `U = X·V/s`.
#[allow(
    clippy::type_complexity,
    reason = "(U, s, Vt) is the standard thin-SVD triple, not worth a named struct"
)]
fn nndsvd_svd<F: Float + Send + Sync + 'static>(
    x: &Array2<F>,
    n_components: usize,
) -> Result<(Array2<F>, Array1<F>, Array2<F>), FerroError> {
    let (n_samples, n_features) = x.dim();
    let k = n_components;

    // f64/f32 fast path through ferray's LAPACK gesdd SVD (sklearn's driver).
    let full = if TypeId::of::<F>() == TypeId::of::<f64>() {
        // SAFETY: TypeId proves F == f64; the cast/transmute is between identical
        // types, the same pattern PCA's `svd_dispatch` uses.
        let x_f64: &Array2<f64> = unsafe { &*(std::ptr::from_ref(x).cast::<Array2<f64>>()) };
        let (u, s, vt) = svd_full_f64(x_f64)?;
        let u_f: Array2<F> = unsafe { std::mem::transmute_copy::<Array2<f64>, Array2<F>>(&u) };
        let s_f: Array1<F> = unsafe { std::mem::transmute_copy::<Array1<f64>, Array1<F>>(&s) };
        let vt_f: Array2<F> = unsafe { std::mem::transmute_copy::<Array2<f64>, Array2<F>>(&vt) };
        std::mem::forget(u);
        std::mem::forget(s);
        std::mem::forget(vt);
        Some((u_f, s_f, vt_f))
    } else if TypeId::of::<F>() == TypeId::of::<f32>() {
        // SAFETY: TypeId proves F == f32; the cast/transmute is between identical
        // types, the same pattern PCA's `svd_dispatch` uses.
        let x_f32: &Array2<f32> = unsafe { &*(std::ptr::from_ref(x).cast::<Array2<f32>>()) };
        let (u, s, vt) = svd_full_f32(x_f32)?;
        let u_f: Array2<F> = unsafe { std::mem::transmute_copy::<Array2<f32>, Array2<F>>(&u) };
        let s_f: Array1<F> = unsafe { std::mem::transmute_copy::<Array1<f32>, Array1<F>>(&s) };
        let vt_f: Array2<F> = unsafe { std::mem::transmute_copy::<Array2<f32>, Array2<F>>(&vt) };
        std::mem::forget(u);
        std::mem::forget(s);
        std::mem::forget(vt);
        Some((u_f, s_f, vt_f))
    } else {
        None
    };

    let (mut u, s, mut vt) = match full {
        Some((u_full, s_full, vt_full)) => {
            // Truncate the thin SVD to the leading k triplets.
            let u_t = u_full.slice(ndarray::s![.., ..k]).to_owned();
            let s_t = s_full.slice(ndarray::s![..k]).to_owned();
            let vt_t = vt_full.slice(ndarray::s![..k, ..]).to_owned();
            (u_t, s_t, vt_t)
        }
        None => {
            // Exotic-F fallback: V/S from Jacobi(XᵀX), U = X·V/s.
            let max_iter = n_features * n_features * 100 + 1000;
            let xtx = x.t().dot(x);
            let (eigenvalues, eigenvectors) = jacobi_eigen_symmetric(&xtx, max_iter)?;
            let mut indices: Vec<usize> = (0..n_features).collect();
            indices.sort_by(|&a, &b| {
                eigenvalues[b]
                    .partial_cmp(&eigenvalues[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let mut s_t = Array1::<F>::zeros(k);
            let mut vt_t = Array2::<F>::zeros((k, n_features));
            for (row, &idx) in indices.iter().take(k).enumerate() {
                let sv = eigenvalues[idx].max(F::zero()).sqrt();
                s_t[row] = sv;
                for j in 0..n_features {
                    vt_t[[row, j]] = eigenvectors[[j, idx]];
                }
            }
            // U = X · Vᵀᵀ / s = X · V / s; columns where s ~ 0 stay zero.
            let mut u_t = Array2::<F>::zeros((n_samples, k));
            for row in 0..k {
                let sv = s_t[row];
                if sv <= eps::<F>() {
                    continue;
                }
                for i in 0..n_samples {
                    let mut acc = F::zero();
                    for j in 0..n_features {
                        acc = acc + x[[i, j]] * vt_t[[row, j]];
                    }
                    u_t[[i, row]] = acc / sv;
                }
            }
            (u_t, s_t, vt_t)
        }
    };

    svd_flip_u_based(&mut u, &mut vt);
    Ok((u, s, vt))
}

/// NNDSVD initialization (Boutsidis & Gallopoulos, 2008), matching sklearn's
/// `_initialize_nmf(X, n_components, init="nndsvd")`
/// (`sklearn/decomposition/_nmf.py:320-358`).
///
/// `U, S, Vt = svd(X)` (top `n_components`, sign-corrected like
/// `randomized_svd`). The leading triplet seeds `W[:,0] = sqrt(S[0])·|U[:,0]|`,
/// `H[0,:] = sqrt(S[0])·|Vt[0,:]|`. For each later component the positive and
/// negative parts of `U[:,j]`/`Vt[j,:]` are split, the larger-norm pair chosen,
/// and scaled by `lbd = sqrt(S[j]·sigma)`. Entries below machine `eps` are
/// zeroed (plain `nndsvd`: zeros stay). Deterministic — no RNG.
fn init_nndsvd<F: Float + Send + Sync + 'static>(
    x: &Array2<F>,
    n_components: usize,
) -> Result<(Array2<F>, Array2<F>), FerroError> {
    let (n_samples, n_features) = x.dim();
    let (u, s, vt) = nndsvd_svd(x, n_components)?;

    // sklearn uses `eps = np.finfo(X.dtype).eps`. Match per concrete F.
    let machine_eps = if TypeId::of::<F>() == TypeId::of::<f32>() {
        F::from(f32::EPSILON).unwrap_or_else(F::epsilon)
    } else {
        F::from(f64::EPSILON).unwrap_or_else(F::epsilon)
    };

    let mut w = Array2::<F>::zeros((n_samples, n_components));
    let mut h = Array2::<F>::zeros((n_components, n_features));

    // Leading singular triplet is non-negative — use as-is.
    let sqrt_s0 = s[0].max(F::zero()).sqrt();
    for i in 0..n_samples {
        w[[i, 0]] = sqrt_s0 * u[[i, 0]].abs();
    }
    for j in 0..n_features {
        h[[0, j]] = sqrt_s0 * vt[[0, j]].abs();
    }

    for comp in 1..n_components {
        // Positive / negative parts of U[:,comp] and Vt[comp,:].
        let mut xp_nrm_sq = F::zero();
        let mut xn_nrm_sq = F::zero();
        for i in 0..n_samples {
            let val = u[[i, comp]];
            if val > F::zero() {
                xp_nrm_sq = xp_nrm_sq + val * val;
            } else {
                xn_nrm_sq = xn_nrm_sq + val * val;
            }
        }
        let mut yp_nrm_sq = F::zero();
        let mut yn_nrm_sq = F::zero();
        for j in 0..n_features {
            let val = vt[[comp, j]];
            if val > F::zero() {
                yp_nrm_sq = yp_nrm_sq + val * val;
            } else {
                yn_nrm_sq = yn_nrm_sq + val * val;
            }
        }
        let x_p_nrm = xp_nrm_sq.sqrt();
        let y_p_nrm = yp_nrm_sq.sqrt();
        let x_n_nrm = xn_nrm_sq.sqrt();
        let y_n_nrm = yn_nrm_sq.sqrt();

        let m_p = x_p_nrm * y_p_nrm;
        let m_n = x_n_nrm * y_n_nrm;

        // `use_positive`, `nrm_u`, `nrm_v`, `sigma`.
        let (use_positive, nrm_u, nrm_v, sigma) = if m_p > m_n {
            (true, x_p_nrm, y_p_nrm, m_p)
        } else {
            (false, x_n_nrm, y_n_nrm, m_n)
        };

        let lbd = (s[comp] * sigma).max(F::zero()).sqrt();

        // W[:,comp] = lbd * (selected part of U[:,comp]) / nrm_u.
        for i in 0..n_samples {
            let val = u[[i, comp]];
            let part = if use_positive {
                if val > F::zero() { val } else { F::zero() }
            } else if val < F::zero() {
                -val
            } else {
                F::zero()
            };
            w[[i, comp]] = if nrm_u > F::zero() {
                lbd * (part / nrm_u)
            } else {
                F::zero()
            };
        }
        // H[comp,:] = lbd * (selected part of Vt[comp,:]) / nrm_v.
        for j in 0..n_features {
            let val = vt[[comp, j]];
            let part = if use_positive {
                if val > F::zero() { val } else { F::zero() }
            } else if val < F::zero() {
                -val
            } else {
                F::zero()
            };
            h[[comp, j]] = if nrm_v > F::zero() {
                lbd * (part / nrm_v)
            } else {
                F::zero()
            };
        }
    }

    // sklearn: `W[W < eps] = 0; H[H < eps] = 0` (plain nndsvd: zeros stay).
    for val in &mut w {
        if *val < machine_eps {
            *val = F::zero();
        }
    }
    for val in &mut h {
        if *val < machine_eps {
            *val = F::zero();
        }
    }

    Ok((w, h))
}

/// Jacobi eigendecomposition for symmetric matrices.
///
/// Returns `(eigenvalues, eigenvectors)` where column `i` of `eigenvectors`
/// is the eigenvector for `eigenvalues[i]`. Eigenvalues are NOT sorted.
fn jacobi_eigen_symmetric<F: Float + Send + Sync + 'static>(
    a: &Array2<F>,
    max_iter: usize,
) -> Result<(Array1<F>, Array2<F>), FerroError> {
    let n = a.nrows();
    if n == 0 {
        return Ok((Array1::zeros(0), Array2::zeros((0, 0))));
    }
    if n == 1 {
        let eigenvalues = Array1::from_vec(vec![a[[0, 0]]]);
        let eigenvectors = Array2::from_shape_vec((1, 1), vec![F::one()]).unwrap();
        return Ok((eigenvalues, eigenvectors));
    }

    let mut mat = a.to_owned();
    let mut v = Array2::<F>::zeros((n, n));
    for i in 0..n {
        v[[i, i]] = F::one();
    }

    let tol = F::from(1e-12).unwrap_or_else(F::epsilon);

    for _iteration in 0..max_iter {
        let mut max_off = F::zero();
        let mut p = 0;
        let mut q = 1;
        for i in 0..n {
            for j in (i + 1)..n {
                let val = mat[[i, j]].abs();
                if val > max_off {
                    max_off = val;
                    p = i;
                    q = j;
                }
            }
        }

        if max_off < tol {
            let eigenvalues = Array1::from_shape_fn(n, |i| mat[[i, i]]);
            return Ok((eigenvalues, v));
        }

        let app = mat[[p, p]];
        let aqq = mat[[q, q]];
        let apq = mat[[p, q]];

        let theta = if (app - aqq).abs() < tol {
            F::from(std::f64::consts::FRAC_PI_4).unwrap_or_else(F::one)
        } else {
            let tau = (aqq - app) / (F::from(2.0).unwrap() * apq);
            let t = if tau >= F::zero() {
                F::one() / (tau.abs() + (F::one() + tau * tau).sqrt())
            } else {
                -F::one() / (tau.abs() + (F::one() + tau * tau).sqrt())
            };
            t.atan()
        };

        let c = theta.cos();
        let s = theta.sin();

        let mut new_mat = mat.clone();
        for i in 0..n {
            if i != p && i != q {
                let mip = mat[[i, p]];
                let miq = mat[[i, q]];
                new_mat[[i, p]] = c * mip - s * miq;
                new_mat[[p, i]] = new_mat[[i, p]];
                new_mat[[i, q]] = s * mip + c * miq;
                new_mat[[q, i]] = new_mat[[i, q]];
            }
        }

        new_mat[[p, p]] = c * c * app - F::from(2.0).unwrap() * s * c * apq + s * s * aqq;
        new_mat[[q, q]] = s * s * app + F::from(2.0).unwrap() * s * c * apq + c * c * aqq;
        new_mat[[p, q]] = F::zero();
        new_mat[[q, p]] = F::zero();

        mat = new_mat;

        for i in 0..n {
            let vip = v[[i, p]];
            let viq = v[[i, q]];
            v[[i, p]] = c * vip - s * viq;
            v[[i, q]] = s * vip + c * viq;
        }
    }

    Err(FerroError::ConvergenceFailure {
        iterations: max_iter,
        message: "Jacobi eigendecomposition did not converge in NMF NNDSVD init".into(),
    })
}

/// Multiplicative update solver (Lee & Seung, 2001).
///
/// Update rules:
///   W <- W * (X H^T) / (W H H^T + eps)
///   H <- H * (W^T X) / (W^T W H + eps)
fn solve_multiplicative_update<F: Float + 'static>(
    x: &Array2<F>,
    w: &mut Array2<F>,
    h: &mut Array2<F>,
    max_iter: usize,
    tol: f64,
) -> usize {
    let tol_f = F::from(tol).unwrap_or_else(F::epsilon);
    let epsilon = eps::<F>();
    let mut prev_err = reconstruction_error(x, w, h);

    for iteration in 0..max_iter {
        // Update H: H <- H * (W^T X) / (W^T W H + eps)
        let wt = w.t();
        let numerator_h = wt.dot(x);
        let denominator_h = wt.dot(&*w).dot(&*h);

        for (h_val, (num, den)) in h
            .iter_mut()
            .zip(numerator_h.iter().zip(denominator_h.iter()))
        {
            *h_val = *h_val * (*num / (*den + epsilon));
        }

        // Update W: W <- W * (X H^T) / (W H H^T + eps)
        let ht = h.t();
        let numerator_w = x.dot(&ht);
        let denominator_w = w.dot(&*h).dot(&ht);

        for (w_val, (num, den)) in w
            .iter_mut()
            .zip(numerator_w.iter().zip(denominator_w.iter()))
        {
            *w_val = *w_val * (*num / (*den + epsilon));
        }

        // Check convergence.
        let err = reconstruction_error(x, w, h);
        if (prev_err - err).abs() < tol_f {
            return iteration + 1;
        }
        prev_err = err;
    }

    max_iter
}

/// One cyclic coordinate-descent sweep over the columns of `w`, with `ht`
/// playing the role of `Ht` (`= H.T`). Mirrors sklearn's Cython
/// `_update_cdnmf_fast` (`sklearn/decomposition/_cdnmf_fast.pyx`):
///
/// ```text
/// HHt = Ht.T @ Ht ; XHt = X @ Ht
/// for t in 0..n_components:
///     for i in 0..n_samples:
///         grad = -XHt[i,t] + sum_r HHt[t,r] * W[i,r]    # W updated in place
///         pg   = grad if W[i,t] != 0 else min(0, grad)  # projected gradient
///         violation += |pg|
///         if HHt[t,t] != 0: W[i,t] = max(W[i,t] - grad / HHt[t,t], 0)
/// ```
///
/// Returns the accumulated projected-gradient `violation`. `w` is mutated in
/// place so later coordinates in the sweep see the already-updated entries.
/// To update `H`, call with `X.T`, `Ht`, `W` (the symmetry sklearn exploits in
/// `_fit_coordinate_descent`).
fn update_cd_sweep<F: Float + 'static>(x: &Array2<F>, w: &mut Array2<F>, ht: &Array2<F>) -> F {
    let n_components = ht.ncols();
    let n_samples = w.nrows();

    // HHt = Ht.T @ Ht  (n_components × n_components)
    let hht = ht.t().dot(ht);
    // XHt = X @ Ht      (n_samples × n_components)
    let xht = x.dot(ht);

    let mut violation = F::zero();
    for t in 0..n_components {
        let hess = hht[[t, t]];
        for i in 0..n_samples {
            // grad = GW[i,t] where GW = W @ HHt - XHt
            let mut grad = -xht[[i, t]];
            for r in 0..n_components {
                grad = grad + hht[[t, r]] * w[[i, r]];
            }
            // projected gradient
            let pg = if w[[i, t]] == F::zero() {
                grad.min(F::zero())
            } else {
                grad
            };
            violation = violation + pg.abs();
            if hess != F::zero() {
                w[[i, t]] = (w[[i, t]] - grad / hess).max(F::zero());
            }
        }
    }
    violation
}

/// Coordinate-descent solver, matching sklearn's `_fit_coordinate_descent`
/// (`sklearn/decomposition/_nmf.py:410-527`).
///
/// Each iteration updates `W` (via [`update_cd_sweep`] over `Ht = H.T`) and, if
/// `update_h`, `H` (via the `X.T`/`Ht`/`W` symmetry), accumulating the total
/// projected-gradient `violation`. The stopping rule is the VIOLATION RATIO
/// `violation / violation_init <= tol` (`_nmf.py:513-525`), NOT a
/// reconstruction-error delta. `H` is carried as its transpose `ht` (`= H.T`)
/// throughout — the `C`-order layout sklearn keeps (`_nmf.py:495`). Returns the
/// 1-based iteration count `n_iter` sklearn stores as `n_iter_`.
fn solve_coordinate_descent<F: Float + 'static>(
    x: &Array2<F>,
    w: &mut Array2<F>,
    h: &mut Array2<F>,
    max_iter: usize,
    tol: f64,
    update_h: bool,
) -> usize {
    let tol_f = F::from(tol).unwrap_or_else(F::epsilon);

    // Work on Ht = H.T (n_features × n_components), as sklearn does.
    let mut ht = h.t().to_owned();
    let xt = x.t().to_owned();

    let mut violation_init = F::zero();
    let mut n_iter = if max_iter == 0 { 0 } else { 1 };

    for iteration in 1..=max_iter {
        n_iter = iteration;
        let mut violation = F::zero();

        // Update W (Ht plays the role of Ht).
        violation = violation + update_cd_sweep(x, w, &ht);

        // Update H (symmetry: X.T, Ht, W).
        if update_h {
            violation = violation + update_cd_sweep(&xt, &mut ht, w);
        }

        if iteration == 1 {
            violation_init = violation;
        }
        if violation_init == F::zero() {
            break;
        }
        if violation / violation_init <= tol_f {
            break;
        }
    }

    // Write Ht.T back into H.
    if update_h {
        for k in 0..h.nrows() {
            for j in 0..h.ncols() {
                h[[k, j]] = ht[[j, k]];
            }
        }
    }

    n_iter
}

fn factorize_nmf<F: Float + Send + Sync + 'static>(
    x: &Array2<F>,
    n_components: usize,
    init: NMFInit,
    solver: NMFSolver,
    max_iter: usize,
    tol: f64,
    random_state: Option<u64>,
) -> Result<(Array2<F>, Array2<F>, usize, F), FerroError> {
    let (n_samples, n_features) = x.dim();
    validate_nmf_input(x, n_components)?;

    let seed = random_state.unwrap_or(0);

    let (mut w, mut h) = match init {
        NMFInit::Random => init_random(n_samples, n_features, n_components, seed),
        NMFInit::Nndsvd => init_nndsvd(x, n_components)?,
    };

    let n_iter = match solver {
        NMFSolver::MultiplicativeUpdate => {
            solve_multiplicative_update(x, &mut w, &mut h, max_iter, tol)
        }
        NMFSolver::CoordinateDescent => {
            solve_coordinate_descent(x, &mut w, &mut h, max_iter, tol, true)
        }
    };

    let reconstruction_err = reconstruction_error(x, &w, &h);

    Ok((w, h, n_iter, reconstruction_err))
}

// ---------------------------------------------------------------------------
// Trait implementations
// ---------------------------------------------------------------------------

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, ()> for NMF<F> {
    type Fitted = FittedNMF<F>;
    type Error = FerroError;

    /// Fit the NMF model by decomposing `X ~ W * H`.
    ///
    /// # Errors
    ///
    /// - [`FerroError::InvalidParameter`] if `n_components` is zero or exceeds
    ///   the minimum of `n_samples` and `n_features`.
    /// - [`FerroError::InvalidParameter`] if any entry of `X` is negative.
    /// - [`FerroError::InsufficientSamples`] if there are zero samples.
    /// - [`FerroError::ConvergenceFailure`] if NNDSVD initialization fails.
    fn fit(&self, x: &Array2<F>, _y: &()) -> Result<FittedNMF<F>, FerroError> {
        let (_, h, n_iter, reconstruction_err) = factorize_nmf(
            x,
            self.n_components,
            self.init,
            self.solver,
            self.max_iter,
            self.tol,
            self.random_state,
        )?;

        Ok(FittedNMF {
            components_: h,
            reconstruction_err_: reconstruction_err,
            n_iter_: n_iter,
        })
    }
}

impl<F: Float + Send + Sync + 'static> Transform<Array2<F>> for FittedNMF<F> {
    type Output = Array2<F>;
    type Error = FerroError;

    /// Project data onto the learned NMF components.
    ///
    /// Mirrors sklearn `NMF.transform` (`sklearn/decomposition/_nmf.py:1213`):
    /// `W = _fit_transform(X, H=self.components_, update_H=False)[0]` — solve the
    /// non-negative least squares `min_{W>=0} ||X - W·H||²` for the FIXED fitted
    /// `H` via the same coordinate descent. With `solver="cd"`, `W` is
    /// initialised to ZEROS (`_nmf.py:1254`) and only `W` is updated
    /// (`update_H=False`), so `transform` re-solves `W` rather than returning the
    /// fitted `W` — both reach the convex NNLS optimum but, like sklearn itself,
    /// the re-solve has its own (looser) violation-ratio stop, so `transform(X)`
    /// differs slightly (~2e-5 here) from `fit_transform`'s co-optimised `W`.
    ///
    /// Uses `max_iter = 200`, `tol = 1e-4` (sklearn's `NMF` defaults).
    ///
    /// # Errors
    ///
    /// - [`FerroError::ShapeMismatch`] if the number of columns does not
    ///   match the number of features seen during fitting.
    /// - [`FerroError::InvalidParameter`] if any entry of `X` is negative.
    fn transform(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let n_features = self.components_.ncols();
        if x.ncols() != n_features {
            return Err(FerroError::ShapeMismatch {
                expected: vec![x.nrows(), n_features],
                actual: vec![x.nrows(), x.ncols()],
                context: "FittedNMF::transform".into(),
            });
        }

        // Finiteness FIRST (before the non-negativity check): sklearn
        // `NMF.transform` likewise runs `_validate_data(force_all_finite=True)`
        // before `check_non_negative`, so a NaN/Inf raises the finiteness
        // `ValueError` (`utils/validation.py:147-154`) BEFORE the projection.
        // NaN AND infinity both rejected (#2289).
        reject_non_finite(x)?;

        // Check non-negativity.
        for &val in x {
            if val < F::zero() {
                return Err(FerroError::InvalidParameter {
                    name: "X".into(),
                    reason: "NMF requires all entries in X to be non-negative".into(),
                });
            }
        }

        let n_samples = x.nrows();
        let n_components = self.components_.nrows();

        // sklearn `_check_w_h` inits W to zeros for the `cd` solver when
        // `update_H=False` (`_nmf.py:1253-1254`); H stays the fitted components_.
        let mut w = Array2::<F>::zeros((n_samples, n_components));
        let mut h = self.components_.clone();

        // Solve W with H fixed (update_H=false) via the violation-ratio CD.
        solve_coordinate_descent(x, &mut w, &mut h, 200, 1e-4, false);

        Ok(w)
    }
}

// ---------------------------------------------------------------------------
// Pipeline integration (generic)
// ---------------------------------------------------------------------------

impl<F: Float + Send + Sync + 'static> PipelineTransformer<F> for NMF<F> {
    /// Fit NMF using the pipeline interface.
    ///
    /// The `y` argument is ignored; NMF is unsupervised.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`Fit::fit`].
    fn fit_pipeline(
        &self,
        x: &Array2<F>,
        _y: &Array1<F>,
    ) -> Result<Box<dyn FittedPipelineTransformer<F>>, FerroError> {
        let fitted = self.fit(x, &())?;
        Ok(Box::new(fitted))
    }
}

impl<F: Float + Send + Sync + 'static> FittedPipelineTransformer<F> for FittedNMF<F> {
    /// Transform data using the pipeline interface.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`Transform::transform`].
    fn transform_pipeline(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        self.transform(x)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    /// Helper: create a small non-negative dataset.
    fn small_dataset() -> Array2<f64> {
        array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
        ]
    }

    /// Helper: create a larger non-negative dataset.
    fn medium_dataset() -> Array2<f64> {
        array![
            [5.0, 3.0, 0.0, 1.0],
            [4.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 5.0],
            [1.0, 0.0, 0.0, 4.0],
            [0.0, 1.0, 5.0, 4.0],
            [0.0, 0.0, 4.0, 3.0],
        ]
    }

    #[test]
    fn test_nmf_basic_fit() {
        let nmf = NMF::<f64>::new(2).with_random_state(42);
        let x = small_dataset();
        let fitted = nmf.fit(&x, &()).unwrap();
        assert_eq!(fitted.components().dim(), (2, 3));
    }

    #[test]
    fn test_nmf_components_non_negative() {
        let nmf = NMF::<f64>::new(2).with_random_state(42);
        let x = small_dataset();
        let fitted = nmf.fit(&x, &()).unwrap();
        for &val in fitted.components() {
            assert!(
                val >= 0.0,
                "component value should be non-negative, got {val}"
            );
        }
    }

    #[test]
    fn test_nmf_transform_dimensions() {
        let nmf = NMF::<f64>::new(2).with_random_state(42);
        let x = small_dataset();
        let fitted = nmf.fit(&x, &()).unwrap();
        let projected = fitted.transform(&x).unwrap();
        assert_eq!(projected.dim(), (4, 2));
    }

    #[test]
    fn test_nmf_transform_non_negative() {
        let nmf = NMF::<f64>::new(2).with_random_state(42);
        let x = small_dataset();
        let fitted = nmf.fit(&x, &()).unwrap();
        let projected = fitted.transform(&x).unwrap();
        for &val in &projected {
            assert!(val >= 0.0, "W value should be non-negative, got {val}");
        }
    }

    #[test]
    fn test_nmf_reconstruction_error_decreases() {
        let nmf_few = NMF::<f64>::new(2).with_random_state(42).with_max_iter(10);
        let nmf_many = NMF::<f64>::new(2).with_random_state(42).with_max_iter(200);
        let x = small_dataset();
        let fitted_few = nmf_few.fit(&x, &()).unwrap();
        let fitted_many = nmf_many.fit(&x, &()).unwrap();
        assert!(
            fitted_many.reconstruction_err() <= fitted_few.reconstruction_err() + 1e-6,
            "more iterations should reduce error: few={}, many={}",
            fitted_few.reconstruction_err(),
            fitted_many.reconstruction_err()
        );
    }

    #[test]
    fn test_nmf_reconstruction_error_positive() {
        let nmf = NMF::<f64>::new(2).with_random_state(42);
        let x = small_dataset();
        let fitted = nmf.fit(&x, &()).unwrap();
        assert!(fitted.reconstruction_err() >= 0.0);
    }

    #[test]
    fn test_nmf_coordinate_descent_solver() {
        let nmf = NMF::<f64>::new(2)
            .with_solver(NMFSolver::CoordinateDescent)
            .with_random_state(42);
        let x = medium_dataset();
        let fitted = nmf.fit(&x, &()).unwrap();
        assert_eq!(fitted.components().dim(), (2, 4));
        for &val in fitted.components() {
            assert!(val >= 0.0, "CD component should be non-negative, got {val}");
        }
    }

    #[test]
    fn test_nmf_nndsvd_init() {
        let nmf = NMF::<f64>::new(2)
            .with_init(NMFInit::Nndsvd)
            .with_random_state(42);
        let x = medium_dataset();
        let fitted = nmf.fit(&x, &()).unwrap();
        assert_eq!(fitted.components().dim(), (2, 4));
        for &val in fitted.components() {
            assert!(
                val >= 0.0,
                "NNDSVD component should be non-negative, got {val}"
            );
        }
    }

    #[test]
    fn test_nmf_cd_with_nndsvd() {
        let nmf = NMF::<f64>::new(2)
            .with_solver(NMFSolver::CoordinateDescent)
            .with_init(NMFInit::Nndsvd)
            .with_random_state(42);
        let x = medium_dataset();
        let fitted = nmf.fit(&x, &()).unwrap();
        assert_eq!(fitted.components().dim(), (2, 4));
    }

    #[test]
    fn test_nmf_invalid_n_components_zero() {
        let nmf = NMF::<f64>::new(0);
        let x = small_dataset();
        assert!(nmf.fit(&x, &()).is_err());
    }

    #[test]
    fn test_nmf_invalid_n_components_too_large() {
        let nmf = NMF::<f64>::new(10);
        let x = small_dataset(); // 4x3
        assert!(nmf.fit(&x, &()).is_err());
    }

    #[test]
    fn test_nmf_negative_input_rejected() {
        let nmf = NMF::<f64>::new(1);
        let x = array![[1.0, -2.0], [3.0, 4.0]];
        assert!(nmf.fit(&x, &()).is_err());
    }

    #[test]
    fn test_nmf_transform_shape_mismatch() {
        let nmf = NMF::<f64>::new(2).with_random_state(42);
        let x = small_dataset();
        let fitted = nmf.fit(&x, &()).unwrap();
        let x_bad = array![[1.0, 2.0]]; // wrong number of features
        assert!(fitted.transform(&x_bad).is_err());
    }

    #[test]
    fn test_nmf_transform_negative_rejected() {
        let nmf = NMF::<f64>::new(2).with_random_state(42);
        let x = small_dataset();
        let fitted = nmf.fit(&x, &()).unwrap();
        let x_neg = array![[1.0, -2.0, 3.0]];
        assert!(fitted.transform(&x_neg).is_err());
    }

    #[test]
    fn test_nmf_reproducibility() {
        let nmf1 = NMF::<f64>::new(2).with_random_state(42);
        let nmf2 = NMF::<f64>::new(2).with_random_state(42);
        let x = small_dataset();
        let fitted1 = nmf1.fit(&x, &()).unwrap();
        let fitted2 = nmf2.fit(&x, &()).unwrap();
        for (a, b) in fitted1.components().iter().zip(fitted2.components().iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_nmf_single_component() {
        let nmf = NMF::<f64>::new(1).with_random_state(42);
        let x = small_dataset();
        let fitted = nmf.fit(&x, &()).unwrap();
        assert_eq!(fitted.components().nrows(), 1);
        let projected = fitted.transform(&x).unwrap();
        assert_eq!(projected.ncols(), 1);
    }

    #[test]
    fn test_nmf_n_iter_positive() {
        let nmf = NMF::<f64>::new(2).with_random_state(42);
        let x = small_dataset();
        let fitted = nmf.fit(&x, &()).unwrap();
        assert!(fitted.n_iter() > 0);
    }

    #[test]
    fn test_nmf_getters() {
        let nmf = NMF::<f64>::new(3)
            .with_max_iter(100)
            .with_tol(1e-5)
            .with_solver(NMFSolver::CoordinateDescent)
            .with_init(NMFInit::Nndsvd)
            .with_random_state(99);
        assert_eq!(nmf.n_components(), 3);
        assert_eq!(nmf.max_iter(), 100);
        assert_abs_diff_eq!(nmf.tol(), 1e-5);
        assert_eq!(nmf.solver(), NMFSolver::CoordinateDescent);
        assert_eq!(nmf.init(), NMFInit::Nndsvd);
        assert_eq!(nmf.random_state(), Some(99));
    }

    #[test]
    fn test_nmf_f32() {
        let nmf = NMF::<f32>::new(1).with_random_state(42);
        let x: Array2<f32> = array![[1.0f32, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let fitted = nmf.fit(&x, &()).unwrap();
        let projected = fitted.transform(&x).unwrap();
        assert_eq!(projected.ncols(), 1);
    }

    #[test]
    fn test_nmf_zero_entries() {
        let nmf = NMF::<f64>::new(2).with_random_state(42);
        let x = array![[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]];
        let fitted = nmf.fit(&x, &()).unwrap();
        assert_eq!(fitted.components().dim(), (2, 3));
    }

    #[test]
    fn test_nmf_pipeline_integration() {
        use ferrolearn_core::pipeline::{FittedPipelineEstimator, Pipeline, PipelineEstimator};
        use ferrolearn_core::traits::Predict;

        struct SumEstimator;

        impl PipelineEstimator<f64> for SumEstimator {
            fn fit_pipeline(
                &self,
                _x: &Array2<f64>,
                _y: &Array1<f64>,
            ) -> Result<Box<dyn FittedPipelineEstimator<f64>>, FerroError> {
                Ok(Box::new(FittedSumEstimator))
            }
        }

        struct FittedSumEstimator;

        impl FittedPipelineEstimator<f64> for FittedSumEstimator {
            fn predict_pipeline(&self, x: &Array2<f64>) -> Result<Array1<f64>, FerroError> {
                let sums: Vec<f64> = x.rows().into_iter().map(|r| r.sum()).collect();
                Ok(Array1::from_vec(sums))
            }
        }

        let pipeline = Pipeline::new()
            .transform_step("nmf", Box::new(NMF::<f64>::new(2).with_random_state(42)))
            .estimator_step("sum", Box::new(SumEstimator));

        let x = small_dataset();
        let y = Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0]);

        let fitted = pipeline.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 4);
    }

    #[test]
    fn test_nmf_medium_dataset_mu() {
        let nmf = NMF::<f64>::new(3)
            .with_solver(NMFSolver::MultiplicativeUpdate)
            .with_random_state(42)
            .with_max_iter(500);
        let x = medium_dataset();
        let fitted = nmf.fit(&x, &()).unwrap();
        assert_eq!(fitted.components().dim(), (3, 4));
        // Reconstruction error should be reasonable.
        assert!(
            fitted.reconstruction_err() < 10.0,
            "reconstruction error too large: {}",
            fitted.reconstruction_err()
        );
    }

    #[test]
    fn test_nmf_insufficient_samples() {
        let nmf = NMF::<f64>::new(1);
        let x = Array2::<f64>::zeros((0, 3));
        assert!(nmf.fit(&x, &()).is_err());
    }

    /// NNDSVD init (isolated) matches sklearn's `_initialize_nmf(X, 3,
    /// init='nndsvd')` element-wise. Expected `H`/`W` are the LIVE sklearn 1.5.2
    /// oracle (`_initialize_nmf`, run from /tmp, R-CHAR-3) on the deterministic
    /// `(RandomState(0).rand(12,6)*5).round(3)` fixture — the SVD pos/neg split
    /// + `lbd = sqrt(S[j]*sigma)` scaling, NOT copied from ferrolearn.
    #[test]
    fn test_nmf_nndsvd_init_matches_sklearn() {
        let x: Array2<f64> = array![
            [2.744, 3.576, 3.014, 2.724, 2.118, 3.229],
            [2.188, 4.459, 4.818, 1.917, 3.959, 2.644],
            [2.84, 4.628, 0.355, 0.436, 0.101, 4.163],
            [3.891, 4.35, 4.893, 3.996, 2.307, 3.903],
            [0.591, 3.2, 0.717, 4.723, 2.609, 2.073],
            [1.323, 3.871, 2.281, 2.842, 0.094, 3.088],
            [3.06, 3.085, 4.719, 3.409, 1.798, 2.185],
            [3.488, 0.301, 3.334, 3.353, 1.052, 0.645],
            [1.577, 1.819, 2.851, 2.193, 4.942, 0.51],
            [1.044, 0.807, 3.266, 1.266, 2.332, 1.222],
            [0.795, 0.552, 3.282, 0.691, 0.983, 1.844],
            [4.105, 0.486, 4.19, 0.48, 4.882, 2.343],
        ];
        let result = init_nndsvd(&x, 3);
        assert!(
            result.is_ok(),
            "init_nndsvd should succeed on the 12x6 fixture"
        );
        let (w, h) = match result {
            Ok(pair) => pair,
            Err(_) => return,
        };
        assert_eq!(h.dim(), (3, 6));
        assert_eq!(w.dim(), (12, 3));
        // sklearn 1.5.2 oracle: _initialize_nmf(X, 3, init='nndsvd').
        let sk_h: [[f64; 6]; 3] = [
            [
                1.7741112747292909,
                2.0242809154746326,
                2.3973352370908354,
                1.7673562411021053,
                1.7161976904877514,
                1.750048077389621,
            ],
            [
                0.0,
                1.5897115406013913,
                0.0,
                0.41349076122979694,
                0.0,
                1.0036389906737029,
            ],
            [
                0.0,
                0.00561171974406682,
                0.0,
                1.6761257358883248,
                0.3407178035232749,
                0.0,
            ],
        ];
        for (k, row) in sk_h.iter().enumerate() {
            for (j, &expected) in row.iter().enumerate() {
                assert_abs_diff_eq!(h[[k, j]], expected, epsilon = 1e-9);
            }
        }
        // W column 0 (leading triplet: sqrt(S[0])*|U[:,0]|).
        let sk_w_col0_head = [1.5111518021294372, 1.7749072337282812, 1.0616211988216013];
        for (i, &expected) in sk_w_col0_head.iter().enumerate() {
            assert_abs_diff_eq!(w[[i, 0]], expected, epsilon = 1e-9);
        }
    }

    #[test]
    fn test_nmf_more_components_lower_error() {
        let nmf1 = NMF::<f64>::new(1).with_random_state(42).with_max_iter(300);
        let nmf2 = NMF::<f64>::new(2).with_random_state(42).with_max_iter(300);
        let x = medium_dataset();
        let fitted1 = nmf1.fit(&x, &()).unwrap();
        let fitted2 = nmf2.fit(&x, &()).unwrap();
        assert!(
            fitted2.reconstruction_err() <= fitted1.reconstruction_err() + 1e-6,
            "more components should reduce error: 1comp={}, 2comp={}",
            fitted1.reconstruction_err(),
            fitted2.reconstruction_err()
        );
    }
}
