//! Principal Component Analysis (PCA).
//!
//! PCA performs linear dimensionality reduction by projecting data onto
//! the directions of maximum variance (principal components). The input
//! data is first centred (mean-subtracted), then the covariance matrix
//! is eigendecomposed to find the top `n_components` directions.
//!
//! # Algorithm
//!
//! Mirrors sklearn's `'auto'` solver selection + `_fit_full`
//! (`sklearn/decomposition/_pca.py:531-543`,`:551-698`):
//!
//! 1. Compute the per-feature mean and centre the data.
//! 2. Select the solver (`'auto'`, dense): `covariance_eigh` when
//!    `n_features <= 1000 && n_samples >= 10*n_features` (`_pca.py:534-535`),
//!    otherwise `full` (the `max(shape) <= 500` and the `randomized`-fallback
//!    cases all route to `full` — ferrolearn has no `randomized`/`arpack`,
//!    REQ-12).
//! 3. `full`: `U, S, Vt = svd(X_centered, full_matrices=False)` via
//!    `ferray::linalg::svd_lapack` (LAPACK `gesdd`, f64/f32) — the SAME driver
//!    `scipy.linalg.svd` calls, so the reconstructed spectrum (incl. the
//!    rank-deficient noise-floor tail) is bit-identical to sklearn. `S`
//!    descending, `explained_variance_ = S²/(n-1)` for all `min(n,p)` directions
//!    (`_pca.py:588-591`). `covariance_eigh`: eigendecompose
//!    `C = X_centered^T X_centered / (n-1)` via faer's self-adjoint
//!    eigensolver, clipping tiny negatives to 0 (`_pca.py:611-643`).
//! 4. `svd_flip(U, Vt, u_based_decision=False)` (`_pca.py:647`): each component
//!    row's max-abs entry is made positive.
//! 5. Resolve `n_components` (count / variance-ratio cumsum / auto) against the
//!    FULL spectrum, compute `noise_variance_` from the discarded tail
//!    (`_pca.py:657-688`), then truncate to the top `n_components_`.
//!
//! # Examples
//!
//! ```
//! use ferrolearn_decomp::PCA;
//! use ferrolearn_core::traits::{Fit, Transform};
//! use ndarray::array;
//!
//! let pca = PCA::<f64>::new(1);
//! let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
//! let fitted = pca.fit(&x, &()).unwrap();
//! let projected = fitted.transform(&x).unwrap();
//! assert_eq!(projected.ncols(), 1);
//! ```
//!
//! ## REQ status
//!
//! Design: `.design/decomp/pca.md`. Tracking: #1499. Each REQ is BINARY — SHIPPED
//! (impl + non-test consumer + tests + green verification) or NOT-STARTED (concrete
//! open blocker). Non-test consumers: crate re-export (`lib.rs:98`), the PyO3
//! `_RsPCA` binding (`ferrolearn-python/src/transformers.rs:89`, registered
//! `lib.rs:23`), and `PipelineTransformer` (`pca.rs:509-536`). Oracle = live sklearn
//! 1.5.2 (`_pca.py`), run from `/tmp` (R-CHAR-3). ferrolearn's `fit` exactly mirrors
//! sklearn's `covariance_eigh` solver (`_pca.py:593-644`) including the `svd_flip`
//! sign step.
//!
//! | REQ | Scope | Status | Evidence / Blocker |
//! |---|---|---|---|
//! | REQ-1 | `components_` sign via `svd_flip(u_based_decision=False)` + EXACT value parity (`components_`/`transform`/`explained_variance_`/`ratio`/`singular_values_`) | SHIPPED | `fit` (`pca.rs:461-481`) applies the per-row max-abs-positive flip (numpy `argmax` first-on-ties via strict `>`, whole-row negate) = sklearn `svd_flip` (`_pca.py:647`, `extmath.py:897-905`). `tests/divergence_pca.rs` matches the live sklearn `PCA` oracle element-wise incl. sign to 1e-6 (3 DIV tests, fixtures 6×3/7×4/near-tie). Was #1500, fixed |
//! | REQ-2 | Degenerate/repeated-eigenvalue + rank-deficient component VALUE parity | NOT-STARTED | CARVE-OUT (R-DEFER-3): repeated eigenvalues → faer/LAPACK pick different orthonormal bases (same class as spectral_embedding) — blocker #1501 |
//! | REQ-3 | Components orthonormality (unit rows + mutual orthogonality) | SHIPPED | `test_pca_components_orthonormal` + green-guard; eigenvectors of symmetric covariance |
//! | REQ-4 | `explained_variance_` ordering/non-negativity + `explained_variance_ratio_` (÷ sum of ALL eigenvalues = sklearn `total_var`) | SHIPPED | matches sklearn element-wise to 1e-6 (critic green-guards); `test_pca_explained_variance_*`, `test_pca_n_components_equals_n_features` |
//! | REQ-5 | `singular_values_` = `sqrt(eigval·(n−1))` ≥ 0 | SHIPPED | matches sklearn to 1e-6 (green-guard); `test_pca_singular_values_positive` |
//! | REQ-6 | `inverse_transform` round-trip exact when `n_components == n_features` | SHIPPED | `test_pca_inverse_transform_roundtrip`/`_approx` + green-guard (sign-invariant) |
//! | REQ-7 | Error/parameter contracts (n_components 0/>n_features, n_samples<2, transform/inverse_transform col mismatch, NON-FINITE rejection) | SHIPPED (scoped) | `fit`/`transform`/`inverse_transform` guards; FLAG: sklearn raises `InvalidParameterError`, accepts `n_components=None`. NON-FINITE: `fit`+`transform` call `reject_non_finite` (`pca.rs` symbol `reject_non_finite`) BEFORE the SVD/projection, returning the CLEAN finiteness `InvalidParameter{name:"X", reason:"Input X contains NaN or infinity."}` = sklearn `_validate_data(force_all_finite=True)` (`_pca.py:511`,`utils/validation.py:147-154`) — replaces the incidental gesdd `NumericalInstability` (R-DEV-2). `tests/divergence_nonfinite.rs::divergence_pca_fit_nan_rejects_for_finiteness`/`_fit_inf_`/`_transform_nan_` match the live sklearn 1.5.2 oracle. Was #2288/#2289, fixed. Consumer: `_RsPCA` `fit`/`transform` (`transformers.rs`) |
//! | REQ-8 | f32 generic support | SHIPPED | `test_pca_f32`; faer f32 eigensolver path |
//! | REQ-9 | `PipelineTransformer` integration | SHIPPED | `pca.rs:509-536`; `test_pca_pipeline_integration` |
//! | REQ-10 | PyO3 `_RsPCA` binding (fit/transform/inverse_transform + components_/explained_variance_/explained_variance_ratio_/mean_/singular_values_ getters) | SHIPPED | `transformers.rs:89`, registered `lib.rs:23`; inherits REQ-1's deterministic signs |
//! | REQ-11 | `whiten` (transform /sqrt(explained_variance_), inverse un-scale) | SHIPPED | `PCA::with_whiten` sets `whiten` (threaded into `FittedPCA`); `Transform::transform` divides each column j by `sqrt(explained_variance_[j])` (eps-clipped) = sklearn `_base.py:157-165`, and `inverse_transform` multiplies back = `_base.py:192-196`. `whiten=false` byte-identical to default. In-module `pca_whiten_transform_matches_sklearn`/`pca_whiten_false_unchanged`/`pca_whiten_inverse_matches_sklearn` match the live sklearn 1.5.2 oracle to 1e-6. Was #1502, fixed |
//! | REQ-12 | `svd_solver` param + full-SVD/randomized/arpack paths | NOT-STARTED | PARTIAL: `fit` now implements sklearn's `'auto'` dense solver-selection (`svd_dispatch`/`use_covariance_eigh`, = `_pca.py:531-543`) and BOTH the `full` SVD-of-centered-X path (`svd_dispatch` → `ferray::linalg::svd_lapack`, LAPACK `gesdd`, = `_pca.py:575-591`) AND the `covariance_eigh` path (`eigen_dispatch`, = `_pca.py:593-643`); `full` is the default for small/wide data (`max(shape)<=500`), restoring sklearn's rank-deficient `noise_variance_ = S[-1]²/(n−1)` tail (greens #2111). The `full` SVD engine is now `ferray::linalg::svd_lapack` (`ferray_svd_lapack_f64`/`_f32`) — the SAME LAPACK `gesdd` driver `scipy.linalg.svd` calls, proven bit-identical to scipy on all singular values incl. the noise-floor value (ferray #2116), so for f64/f32 ferrolearn's `noise_variance_` is byte-for-byte sklearn's (`pca_full_solver_noise_variance_rank_deficient_matches_sklearn` now asserts bit-identity to `rtol=1e-12`). The `randomized` truncated solver is now SHIPPED (#2115): `fit` routes the `'auto'`-selected `randomized` shape (NOT covariance_eigh, `max(shape)>500`, `1<=n_components<0.8*min(shape)`, = `_pca.py:539-540`) to `fit_randomized` → `randomized_svd_dispatch` (`randomized_svd_f64`/`_f32`), a faithful translation of sklearn `randomized_svd`+`randomized_range_finder` (`extmath.py:217-557`): Gaussian range-finder draw via `ferray::random::RandomState::new(seed).standard_normal_2d` (bit-identical to `numpy.random.RandomState(seed).normal`, ferray #2118), `n_iter='auto'`→`7 if n_components<0.1*min else 4`, `transpose='auto'`→`n_samples<n_features` (transpose path implemented), `n_oversamples=10`, power-iteration normaliser = faer economic-QR via `ferray::linalg::qr` (sklearn's `'auto'` resolves to LU at n_iter≥3; faer-QR matches LU to ~9e-15 abs on the #2115 fixture, far inside rtol-1e-6 — NO LAPACK geqrf/orgqr follow-up needed), matmuls via `ferray::linalg::gemm`, inner SVD via `ferray::linalg::svd_lapack` (gesdd); then `svd_flip(u_based_decision=False)` row-wise (`_pca.py:773`) and the TRUNCATED `noise_variance_ = (total_var − Σexp_var)/(min(n,p)−n_components)` (`_pca.py:797-799`, total_var from `Σ X_centered²/(n−1)`, `:789-792`). In-module `pca_randomized_solver_matches_sklearn` (600×100 fixture, seed 42) matches the live sklearn 1.5.2 `'auto'` (randomized) oracle: explained_variance_ to 1e-6, singular_values_ to 1e-5, noise_variance_ to 1e-6. Non-test consumer: `_RsPCA` (`transformers.rs`) threads `random_state` → `PCA::with_random_state` → the randomized branch; pin `divergence_transformers.py::test_red_pca_auto_solver_randomized_branch_matches_sklearn` green. STILL NOT-STARTED: no `svd_solver` ctor param to OVERRIDE `'auto'` (randomized is reachable only via `'auto'` selection), and the `arpack` truncated solver (`_pca.py:753-760`) is absent — blocker #1503. |
//! | REQ-13a | `n_components` as float (variance ratio) + auto/`None` | SHIPPED | `NComponents<F>` enum (`Count`/`Ratio`/`Auto`) replaces the `usize` field; `PCA::with_variance_ratio`/`PCA::auto` constructors. `fit` resolves the spec AFTER the full eigendecomposition: `Ratio(r)` → `1 + count(ratio_cumsum[i] <= r)` clamped to `min_dim` = sklearn `searchsorted(ratio_cumsum, r, "right")+1` (`_pca.py:680-681`); `Auto` → `min(n_samples, n_features)` (`_pca.py:685`). In-module `pca_n_components_ratio_095_selects_2`/`_05_selects_1`/`_0999_selects_3`/`pca_n_components_auto_selects_all`/`pca_n_components_ratio_validation_rejects` match the live sklearn 1.5.2 oracle (cumsum `[0.898229, 0.987108, 1.0]`). Consumers: re-export `lib.rs:98`, `_RsPCA` `transformers.rs:143` (`PCA::new` backward-compatible → `Count`). Was #1504 |
//! | REQ-13b | `n_components = "mle"` (Minka automatic dimensionality) | NOT-STARTED | sklearn `_infer_dimension` (`_pca.py:657-658`); `NComponents` has no `Mle` variant — blocker #1504 (carved from REQ-13) |
//! | REQ-14 | `get_covariance` / `get_precision` | SHIPPED | `FittedPCA::get_covariance` (`pca.rs` symbol `get_covariance`) builds `components_ᵀ·diag(exp_var_diff)·components_ + noise_variance_·I` = sklearn `_base.py:30-56`; `get_precision` (symbol `get_precision`/`precision_via_lemma`) uses sklearn's matrix-inversion (Woodbury) lemma `_base.py:58-101` — corner cases `n_components_==0 → eye/noise_variance_` (`:74-75`) and `noise_variance_==0 → inv(get_covariance())` (`:82-83`); general branch forms `P = comps·compsᵀ/noise_variance_ + diag(1/exp_var_diff)` (comps rescaled by `sqrt(exp_var)` when `whiten`, `:88-89`) and returns `compsᵀ·P⁻¹·comps / −noise_variance_² + diag(1/noise_variance_)` (`:96-100`). The lemma only inverts a small `n_comp × n_comp` matrix, so it stays FINITE for rank-deficient (singular-covariance) fits where the old eigendecompose-invert aborted on zero eigenvalues. IEEE inf-diagonal (`exp_var_diff[k]==0 → 1/0=+inf → P⁻¹` row/col 0) reproduced by inverting only the `exp_var_diff>0` submatrix. In-module `pca_get_covariance_matches_sklearn`/`pca_get_precision_matches_sklearn`/`pca_get_precision_general_branch_matches_sklearn` (whiten True+False) match the live sklearn 1.5.2 oracle to 1e-6, and `pca_get_precision_rank_deficient_does_not_raise` locks the no-ValueError/finiteness contract the lemma delivers for `n_samples < n_features`. Consumer: `score_samples`/`score` call `precision_and_logdet`; re-export `lib.rs:98`. Was #1505; the wide-rank-deficient VALUE pin #2110 now runs through the `full` SVD (REQ-12): `noise_variance_` is the tiny tail `S[-1]²/(n−1)` (not the covariance_eigh clip-to-0), so `get_precision`/`score` are FINITE. With the `full` SVD now on `ferray::linalg::svd_lapack` (LAPACK `gesdd`, bit-identical to scipy on the near-zero rank-deficient direction, `~1.34e-31`), `noise_variance_` is byte-for-byte sklearn's and the reconstructed Woodbury inputs `comps`/`P` are BIT-IDENTICAL to sklearn's, so `get_covariance` (atol 1e-6) and `get_precision` (rtol 1e-6) MATCH the live oracle for the #2110 fixture. `precision_via_lemma` now routes the inner `P`-inverse through `ferray::linalg::inv_lapack` (LAPACK `getrf`+`getri` = scipy `linalg.inv`, via `inv_dispatch`→`ferray_inv_lapack_f64`/`_f32`, #2117) and the lemma matmuls `P = comps@compsᵀ` and `compsᵀ@inv_p@comps` through `ferray::linalg::gemm` (OpenBLAS, via `gemm_dispatch`→`ferray_gemm_f64`/`_f32`) evaluated LEFT-TO-RIGHT; `score_samples`'s quadratic `Xr@precision` likewise goes through `gemm_dispatch`. The `fast_logdet(precision)` SIGN (`score`/`score_samples`) is now numpy-faithful: `fast_logdet_dispatch`→`fast_logdet_getrf_f64`/`_f32` factorize the precision with LAPACK `getrf` (the SAME factorization `numpy.linalg.slogdet` uses), so the rank-deficient whiten=True precision yields the FINITE log-magnitude (det>0, sign +1) sklearn gives — the prior hand-rolled / faer LU flipped the sign → spurious `−inf`. `pca_score_rank_deficient_whiten_finite_matches_sklearn` locks this FINITENESS/SIGN contract. PARTIAL on #2110 (the `score_samples` exact-VALUE pin `pytest divergence_transformers.py::test_red_pca_whiten_precision_rank_deficient_matches_sklearn` STAYS OPEN): the score quadratic is catastrophically ill-conditioned (cond `~6.7e15`), so the per-sample value hinges on the precision being BIT-identical to numpy's. The lemma's inner `P` off-diagonal is rounding NOISE on (numerically) orthogonal component rows (`~2e-16`) ÷ `noise_variance_ ~1e-31`; `ferray::linalg::gemm` requires both operands row-major and calls OpenBLAS no-transpose, whereas numpy `comps @ comps.T` passes a transpose-FLAGGED view → a different OpenBLAS kernel/summation order → a different residual on that product → the precision differs by 1–2 ULPs → O(1) relative error in the cancelled score. EXACT score parity needs a numpy-`@`-identical TRANSPOSE-AWARE `gemm` in ferray (and a numpy-`slogdet`-identical `slogdet` so the knife-edge whiten=False singular case yields `−inf`) — a ferray substrate gap (R-SUBSTRATE-5), escalated. The `get_precision` rtol-1e-6 line of the pin DOES pass; only `score_samples` does not |
//! | REQ-15 | `score` / `score_samples` (Gaussian log-likelihood) + `noise_variance_` | SHIPPED | `fit` captures the FULL eigenvalue spectrum and sets `noise_variance_ = mean(sorted_eigenvalues[n_comp..min_dim])` = sklearn `_pca.py:685-688` (getter symbol `noise_variance`). `FittedPCA::score_samples` computes `ll_i = −0.5·Σ_j Xr[i,j]·(Xr@precision)[i,j] − 0.5·(p·ln(2π) − logdet(precision))` (`Xr@precision` via `gemm_dispatch` = numpy `@`; `logdet = fast_logdet(precision)` via `fast_logdet_dispatch`→LAPACK `getrf`, numpy-`slogdet`-faithful, `−inf` when `det ≤ 0` = sklearn `extmath.py:93-130`) = sklearn `_pca.py:805-830`; `score = mean(score_samples)` = `_pca.py:832-853`. In-module `pca_noise_variance_matches_sklearn`/`pca_score_samples_matches_sklearn`/`pca_score_matches_sklearn` match the live sklearn 1.5.2 oracle to 1e-6. `whiten=true` is handled: `get_covariance` folds the `components_ * sqrt(exp_var)` rescale (`_base.py:46-47`) into the per-component weight (`exp_var_diff[k] * explained_variance_[k]`), so `precision`/`score`/`score_samples` follow — `pca_whiten_get_covariance_and_score_match_sklearn` matches the live oracle (#2107/#2108). Consumer: `score`/`score_samples` consume `noise_variance_`+`get_precision`; re-export `lib.rs:98`. Was #1507, fixed |
//! | REQ-16 | Fitted attrs `n_components_` / `n_features_in_` | SHIPPED | `FittedPCA::n_components_()` (= `components_.nrows()`) + `n_features_in_()` (= `mean_.len()`); `pca_n_components_and_n_features_in_match_sklearn` |
//! | REQ-17 | ctor params `tol`/`iterated_power`/`n_oversamples`/`power_iteration_normalizer`/`random_state`/`copy` | NOT-STARTED | PARTIAL: `random_state` is now SHIPPED (#2115) — `PCA<F>` carries a `random_state: Option<u64>` field + `PCA::with_random_state` builder (`_pca.py:418`), threaded into the `randomized` solver's range-finder seed (`fit_randomized`); `None` maps to a fixed reproducible draw (the Rust analog of numpy's non-reproducible GLOBAL RNG default, R-DEV-4). Non-test consumer: `_RsPCA` `random_state` ctor param (`transformers.rs`) → `_transformers.py::PCA.__init__(.., random_state=None)` → `_resolve_random_state`. `n_oversamples`/`iterated_power`/`power_iteration_normalizer` are pinned to sklearn defaults (`PCA::n_oversamples`=10, `PCA::iterated_power_spec`=`'auto'`/`None`) but NOT yet user-settable; `tol`/`copy` and the param surface for the rest remain absent — blocker #1509. |
//! | REQ-18 | ferray substrate | NOT-STARTED | PARTIAL: the `full`-SVD engine is migrated to the ferray substrate — `svd_dispatch` (f64/f32) routes through `ferray::linalg::svd_lapack` (LAPACK `gesdd`) via `ferray_svd_lapack_f64`/`_f32` (bridge: centred-X `ndarray` → `ferray::Array<_, Ix2>` `from_vec` → `svd_lapack` → `into_ndarray()`, R-SUBSTRATE-4). The Woodbury `get_precision`/`score` path is also on the ferray substrate: `inv_dispatch`→`ferray::linalg::inv_lapack` (LAPACK `getrf`+`getri`) and `gemm_dispatch`→`ferray::linalg::gemm` (OpenBLAS) replace the hand-rolled Gauss-Jordan inverse + nested-loop matmuls for f64/f32 (same `from_vec`/`into_ndarray` bridge, R-SUBSTRATE-4). Non-test consumers: `Fit::fit`'s `full` path calls `svd_dispatch`; `score_samples`/`get_precision` call `precision_via_lemma` (`inv_dispatch`/`gemm_dispatch`) + `fast_logdet_dispatch` (`pca.rs` symbols), re-exported `lib.rs:98` + `_RsPCA` `transformers.rs`. STILL NOT-STARTED: the array type is still `ndarray` (not `ferray-core`); the `covariance_eigh` eigensolver is still direct `faer` (`faer_eigen_f64`/`_f32`); the exotic-`F` fallbacks are Jacobi/Gauss-Jordan; and `fast_logdet` uses a LOCAL LAPACK `getrf` FFI shim (`dgetrf_`/`sgetrf_`) because `ferray::linalg::slogdet` is faer-LU and diverges from numpy on ill-conditioned matrices — file the ferray numpy-`slogdet` + transpose-aware `gemm` gaps (R-SUBSTRATE-5). Blocker #1510 |
//!
//! Count: **13 SHIPPED (REQ-1,3,4,5,6,7,8,9,10,11,13a,14,15) / 6 NOT-STARTED
//! (REQ-2,12,13b,16,17,18)**.

use ferrolearn_core::error::FerroError;
use ferrolearn_core::pipeline::{FittedPipelineTransformer, PipelineTransformer};
use ferrolearn_core::traits::{Fit, Transform};
use ndarray::{Array1, Array2};
use num_traits::Float;
use std::any::TypeId;

/// Reject non-finite input the way sklearn's `_validate_data` does.
///
/// sklearn runs `check_array` with the default `force_all_finite=True` at the
/// top of every decomposition `fit`/`transform`, raising
/// `ValueError("Input X contains NaN.")` / `"... contains infinity ..."`
/// (`sklearn/utils/validation.py:147-154`) BEFORE any decomposition math. PCA
/// has no missing-value support, so NaN AND infinity are both rejected. The
/// message names "NaN" and "infinity" to mirror sklearn's `ValueError`. Never
/// panics (R-CODE-2).
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
// PCA (unfitted)
// ---------------------------------------------------------------------------

/// Specification of how many principal components to retain.
///
/// Mirrors the polymorphic `n_components` argument of sklearn's `PCA`
/// (`sklearn/decomposition/_pca.py:407`,`:657-688`): an explicit integer, a
/// variance-ratio float, or `None`. The `"mle"` (Minka automatic dimensionality)
/// case is NOT yet supported (REQ-13b NOT-STARTED).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NComponents<F> {
    /// Retain exactly this many components (an explicit integer count).
    ///
    /// Validated `1 <= count <= min(n_samples, n_features)` at [`Fit::fit`]
    /// time. Equivalent to passing an `int` to sklearn `PCA(n_components=int)`.
    Count(usize),
    /// Retain the smallest number of components whose CUMULATIVE
    /// explained-variance ratio is `>=` this value.
    ///
    /// The value must lie in `(0, 1]`. Mirrors sklearn's float `n_components`
    /// branch (`sklearn/decomposition/_pca.py:659-681`): it computes the cumsum
    /// of the FULL `explained_variance_ratio_` and selects
    /// `n_components_ = searchsorted(ratio_cumsum, ratio, side="right") + 1`
    /// (clamped to `min(n_samples, n_features)`).
    Ratio(F),
    /// Auto: retain `min(n_samples, n_features)` components (sklearn
    /// `n_components=None`, `sklearn/decomposition/_pca.py:523-527`,`:685`).
    Auto,
}

/// Principal Component Analysis configuration.
///
/// Holds the `n_components` hyperparameter. Calling [`Fit::fit`] centres
/// the data, computes the eigendecomposition of the covariance matrix,
/// and returns a [`FittedPCA`] that can project new data.
#[derive(Debug, Clone)]
pub struct PCA<F> {
    /// How many principal components to retain (explicit count, variance ratio,
    /// or auto). See [`NComponents`].
    n_components: NComponents<F>,
    /// When `true`, [`Transform::transform`] divides each projected component
    /// by `sqrt(explained_variance_)` so the transformed output has unit
    /// component-wise variance, and [`FittedPCA::inverse_transform`]
    /// re-multiplies before reconstructing. Mirrors sklearn `PCA(whiten=...)`
    /// (`sklearn/decomposition/_base.py:157-165` for the transform scale,
    /// `:192-196` for the inverse un-scale). Defaults to `false`.
    whiten: bool,
    /// Seed for the `randomized` solver's Gaussian range-finder draw (sklearn
    /// `PCA(random_state=...)`, `sklearn/decomposition/_pca.py:418`). `None`
    /// is sklearn's default; ferrolearn requires an explicit seed for the
    /// randomized branch to be deterministic, so `None` is treated as seed `0`
    /// (sklearn's `check_random_state(None)` uses the global numpy RNG, which we
    /// cannot reproduce — `None` here pins to a fixed reproducible draw rather
    /// than a global one). Only consumed when the `'auto'` solver selects the
    /// `randomized` path (`_pca.py:539-540`); ignored by `full`/`covariance_eigh`.
    random_state: Option<u64>,
}

impl<F: Float + Send + Sync + 'static> PCA<F> {
    /// Create a new `PCA` that retains exactly `n_components` principal
    /// components (an explicit integer count, [`NComponents::Count`]).
    ///
    /// Backward-compatible with the historical `usize`-only constructor. For a
    /// variance-ratio target use [`PCA::with_variance_ratio`]; for the
    /// `min(n_samples, n_features)` auto default use [`PCA::auto`].
    ///
    /// Whitening is disabled by default (sklearn `whiten=False`); enable it
    /// with [`PCA::with_whiten`].
    ///
    /// # Panics
    ///
    /// Does not panic. Validation of `n_components` against the data
    /// dimensions happens during [`Fit::fit`].
    #[must_use]
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components: NComponents::Count(n_components),
            whiten: false,
            random_state: None,
        }
    }

    /// Create a `PCA` that retains the smallest number of components whose
    /// cumulative explained-variance ratio is `>= ratio` ([`NComponents::Ratio`]).
    ///
    /// `ratio` must lie in `(0, 1]`; this is validated at [`Fit::fit`] time
    /// (an out-of-range value yields [`FerroError::InvalidParameter`]). Mirrors
    /// sklearn's float `n_components` branch
    /// (`sklearn/decomposition/_pca.py:659-681`).
    #[must_use]
    pub fn with_variance_ratio(ratio: F) -> Self {
        Self {
            n_components: NComponents::Ratio(ratio),
            whiten: false,
            random_state: None,
        }
    }

    /// Create a `PCA` that retains `min(n_samples, n_features)` components
    /// ([`NComponents::Auto`], sklearn `n_components=None`,
    /// `sklearn/decomposition/_pca.py:523-527`).
    #[must_use]
    pub fn auto() -> Self {
        Self {
            n_components: NComponents::Auto,
            whiten: false,
            random_state: None,
        }
    }

    /// Set the random seed used by the `randomized` SVD solver's Gaussian
    /// range-finder draw (sklearn `PCA(random_state=...)`,
    /// `sklearn/decomposition/_pca.py:418`).
    ///
    /// Only the `'auto'`-selected `randomized` path (`_pca.py:539-540`) consumes
    /// this seed; the `full`/`covariance_eigh` paths are deterministic and
    /// ignore it. The seed feeds `ferray::random::RandomState::new(seed)`, whose
    /// `standard_normal` is bit-identical to `numpy.random.RandomState(seed)`,
    /// so for a fixed seed the randomized spectrum reproduces sklearn's.
    #[must_use]
    pub fn with_random_state(mut self, random_state: Option<u64>) -> Self {
        self.random_state = random_state;
        self
    }

    /// Enable or disable whitening (sklearn `PCA(whiten=...)`).
    ///
    /// When `whiten` is `true`, [`Transform::transform`] divides each projected
    /// component by `sqrt(explained_variance_)`
    /// (`sklearn/decomposition/_base.py:157-165`), and
    /// [`FittedPCA::inverse_transform`] re-multiplies it back (`:192-196`).
    #[must_use]
    pub fn with_whiten(mut self, whiten: bool) -> Self {
        self.whiten = whiten;
        self
    }

    /// Return the [`NComponents`] specification this PCA is configured with.
    ///
    /// This is the requested spec (count / variance-ratio / auto), NOT the
    /// resolved integer count — that is available on the fitted model via
    /// [`FittedPCA::n_components_`].
    #[must_use]
    pub fn n_components(&self) -> &NComponents<F> {
        &self.n_components
    }

    /// Return whether whitening is enabled.
    #[must_use]
    pub fn whiten(&self) -> bool {
        self.whiten
    }

    /// Return the random seed used by the `randomized` solver (sklearn
    /// `random_state`), or `None` if unset.
    #[must_use]
    pub fn random_state(&self) -> Option<u64> {
        self.random_state
    }

    /// `n_oversamples` for the `randomized` solver. Fixed to sklearn's default
    /// `10` (`sklearn/decomposition/_pca.py:416`,`:767`); ferrolearn does not yet
    /// expose it as a ctor param (REQ-17 partial).
    #[must_use]
    fn n_oversamples(&self) -> usize {
        10
    }

    /// `iterated_power` for the `randomized` solver as an explicit count, or
    /// `None` for sklearn's `'auto'` (`sklearn/decomposition/_pca.py:415`,`:768`;
    /// resolved to `7 if n_components < 0.1*min(shape) else 4` inside
    /// `randomized_svd`, `extmath.py:512`). Fixed to `'auto'` (`None`) —
    /// ferrolearn does not yet expose `iterated_power` as a ctor param.
    #[must_use]
    fn iterated_power_spec(&self) -> Option<usize> {
        None
    }
}

// ---------------------------------------------------------------------------
// FittedPCA
// ---------------------------------------------------------------------------

/// A fitted PCA model holding the learned principal components and statistics.
///
/// Created by calling [`Fit::fit`] on a [`PCA`]. Implements
/// [`Transform<Array2<F>>`] to project new data, and provides
/// [`inverse_transform`](FittedPCA::inverse_transform) to reconstruct
/// approximate original data.
#[derive(Debug, Clone)]
pub struct FittedPCA<F> {
    /// Principal component directions, shape `(n_components, n_features)`.
    /// Each row is a unit eigenvector of the covariance matrix.
    components_: Array2<F>,

    /// Variance explained by each component (eigenvalues of the covariance
    /// matrix, sorted descending).
    explained_variance_: Array1<F>,

    /// Ratio of variance explained by each component to total variance.
    explained_variance_ratio_: Array1<F>,

    /// Per-feature mean computed during fitting, used for centring.
    mean_: Array1<F>,

    /// Singular values corresponding to each component.
    singular_values_: Array1<F>,

    /// Estimated noise covariance of the probabilistic-PCA model: the mean of
    /// the DISCARDED (tail) eigenvalues of the sample covariance. Mirrors
    /// sklearn `noise_variance_` (`sklearn/decomposition/_pca.py:686`):
    /// `mean(explained_variance_[n_components:min(n_samples, n_features)])`, or
    /// `0.0` when `n_components >= min(n_samples, n_features)`. Used by
    /// [`FittedPCA::get_covariance`] / [`FittedPCA::get_precision`] /
    /// [`FittedPCA::score_samples`].
    noise_variance_: F,

    /// Whether whitening is applied in `transform`/`inverse_transform`.
    /// Propagated from the unfitted [`PCA::whiten`] setting. Mirrors sklearn
    /// `PCA(whiten=...)` (`sklearn/decomposition/_base.py:157-165,:192-196`).
    whiten: bool,
}

impl<F: Float + Send + Sync + 'static> FittedPCA<F> {
    /// Principal components, shape `(n_components, n_features)`.
    #[must_use]
    pub fn components(&self) -> &Array2<F> {
        &self.components_
    }

    /// Explained variance per component (eigenvalues).
    #[must_use]
    pub fn explained_variance(&self) -> &Array1<F> {
        &self.explained_variance_
    }

    /// Explained variance ratio per component.
    #[must_use]
    pub fn explained_variance_ratio(&self) -> &Array1<F> {
        &self.explained_variance_ratio_
    }

    /// Per-feature mean learned during fitting.
    #[must_use]
    pub fn mean(&self) -> &Array1<F> {
        &self.mean_
    }

    /// Singular values corresponding to each component.
    #[must_use]
    pub fn singular_values(&self) -> &Array1<F> {
        &self.singular_values_
    }

    /// The resolved number of retained components, `n_components_`.
    ///
    /// Equals the number of rows of `components_`. Mirrors sklearn
    /// `PCA.n_components_` (`sklearn/decomposition/_pca.py:691`), the count
    /// after any `n_components` float/`"mle"`/`None` post-processing.
    #[must_use]
    pub fn n_components_(&self) -> usize {
        self.components_.nrows()
    }

    /// The number of features seen during fitting, `n_features_in_`.
    ///
    /// Equals the length of `mean_`. Mirrors sklearn's `n_features_in_`
    /// fitted attribute.
    #[must_use]
    pub fn n_features_in_(&self) -> usize {
        self.mean_.len()
    }

    /// Estimated noise covariance of the probabilistic-PCA model.
    ///
    /// Equals `mean(explained_variance_[n_components:min(n_samples, n_features)])`
    /// — the average of the discarded (tail) eigenvalues of the sample
    /// covariance — or `0` when `n_components >= min(n_samples, n_features)`.
    /// Mirrors sklearn `noise_variance_`
    /// (`sklearn/decomposition/_pca.py:686`).
    #[must_use]
    pub fn noise_variance(&self) -> F {
        self.noise_variance_
    }

    /// Compute the data covariance with the generative (probabilistic-PCA)
    /// model: `cov = components_ᵀ · diag(exp_var_diff) · components_`, then add
    /// `noise_variance_` to the diagonal, where
    /// `exp_var_diff[k] = max(explained_variance_[k] − noise_variance_, 0)`.
    ///
    /// Mirrors sklearn `_BasePCA.get_covariance`
    /// (`sklearn/decomposition/_base.py:30-56`:
    /// `exp_var_diff = where(exp_var > noise_variance_, exp_var − noise_variance_, 0)`;
    /// `cov = (components_.T * exp_var_diff) @ components_`;
    /// `_add_to_diagonal(cov, noise_variance_)`). Returns a
    /// `(n_features, n_features)` matrix.
    ///
    /// # Whitening
    ///
    /// When `whiten == true`, sklearn rescales the components by
    /// `sqrt(explained_variance_)` BEFORE the outer product
    /// (`components_ = components_ * sqrt(exp_var)`, sklearn `_base.py:46-47`),
    /// so each of the two component factors carries a `sqrt(exp_var[k])`. This
    /// implementation folds that into the per-component weight as
    /// `exp_var_diff[k] * explained_variance_[k]`; `whiten == false` uses the
    /// bare `exp_var_diff[k]`. Both paths feed [`FittedPCA::get_precision`] and
    /// the score methods identically.
    #[must_use]
    pub fn get_covariance(&self) -> Array2<F> {
        let n_features = self.mean_.len();
        let n_comp = self.components_.nrows();

        // exp_var_diff[k] = max(explained_variance_[k] - noise_variance_, 0)
        // (sklearn `_base.py:48-53` with the `where(... , 0)` clamp).
        let mut exp_var_diff = Array1::<F>::zeros(n_comp);
        for k in 0..n_comp {
            let diff = self.explained_variance_[k] - self.noise_variance_;
            exp_var_diff[k] = if self.explained_variance_[k] > self.noise_variance_ {
                diff
            } else {
                F::zero()
            };
        }

        // cov = components_ᵀ · diag(exp_var_diff) · components_
        //     = Σ_k exp_var_diff[k] · (component_k ⊗ component_k)
        // (sklearn `cov = (components_.T * exp_var_diff) @ components_`,
        // `_base.py:54`).
        let mut cov = Array2::<F>::zeros((n_features, n_features));
        for k in 0..n_comp {
            // When `whiten` is enabled, sklearn rescales the components by
            // `sqrt(exp_var)` BEFORE forming the outer product
            // (`_base.py:46-47`: `components_ = components_ * sqrt(exp_var)`), so
            // EACH component factor carries a `sqrt(exp_var[k])` — the two
            // factors multiply to `exp_var[k]`, folded into the per-component
            // weight here. `whiten == false` keeps the bare `exp_var_diff[k]`.
            let w = if self.whiten {
                exp_var_diff[k] * self.explained_variance_[k]
            } else {
                exp_var_diff[k]
            };
            for i in 0..n_features {
                let ci = self.components_[[k, i]];
                for j in 0..n_features {
                    cov[[i, j]] = cov[[i, j]] + w * ci * self.components_[[k, j]];
                }
            }
        }

        // cov[i,i] += noise_variance_ (sklearn `_add_to_diagonal`, `_base.py:55`).
        for i in 0..n_features {
            cov[[i, i]] = cov[[i, i]] + self.noise_variance_;
        }

        cov
    }

    /// Compute the data precision matrix (inverse of [`get_covariance`]) of the
    /// generative (probabilistic-PCA) model, via sklearn's matrix-inversion
    /// (Woodbury) lemma (`sklearn/decomposition/_base.py:58-101`).
    ///
    /// The lemma only ever inverts a small `n_components × n_components` matrix
    /// `P`, so it stays FINITE even when [`get_covariance`] is singular
    /// (rank-deficient fits with `n_samples < n_features` have zero-variance
    /// directions, hence a non-invertible covariance). The previous approach
    /// (eigendecompose the full covariance, invert each eigenvalue) aborted on
    /// the resulting zero eigenvalues; the lemma does not.
    ///
    /// Corner cases mirror sklearn exactly: `n_components_ == 0` returns
    /// `eye(n_features) / noise_variance_` (`_base.py:74-75`); a zero
    /// `noise_variance_` (full-rank covariance) inverts [`get_covariance`]
    /// directly (`_base.py:82-83`). The general branch (`noise_variance_ > 0`)
    /// forms `P = comps·compsᵀ / noise_variance_ + diag(1 / exp_var_diff)` and
    /// returns `precision = compsᵀ · P⁻¹ · comps / −noise_variance_² +
    /// diag(1 / noise_variance_)` (`_base.py:85-101`), where `comps` are the
    /// components rescaled by `sqrt(exp_var)` when `whiten` (`_base.py:88-89`).
    ///
    /// [`get_covariance`]: FittedPCA::get_covariance
    ///
    /// IEEE inf-diagonal: when a (degenerate) component has
    /// `exp_var_diff[k] == 0`, sklearn adds `1.0 / 0.0 == +inf` to `P[k,k]` and
    /// numpy/LAPACK yield a zero row/col in `P⁻¹`. This implementation reproduces
    /// that by inverting only the submatrix of `P` over indices with
    /// `exp_var_diff[k] > 0` and placing zeros in the dropped positions
    /// (the exact IEEE limit of the inf-diagonal inverse).
    ///
    /// Both `whiten == true` and `whiten == false` are handled.
    ///
    /// # Errors
    ///
    /// - [`FerroError::NumericalInstability`] if a required float constant cannot
    ///   be represented in `F`, or the small `P`-matrix inversion fails.
    pub fn get_precision(&self) -> Result<Array2<F>, FerroError> {
        let (precision, _logdet) = self.precision_and_logdet()?;
        Ok(precision)
    }

    /// Compute both the precision matrix (via the Woodbury lemma, see
    /// [`get_precision`](FittedPCA::get_precision)) and `logdet(precision)`.
    ///
    /// Returns `(precision, logdet_precision)`. `logdet_precision` is the
    /// sign-aware `slogdet` of the LEMMA precision — sklearn's
    /// `fast_logdet(precision)` (`utils/extmath.py:93-130`): `slogdet` returns
    /// `(sign, ld)` and `fast_logdet` yields `ld` when `sign > 0` else `−inf`.
    /// This is implemented with an LU decomposition with partial pivoting
    /// (mirroring numpy/LAPACK `slogdet`), NOT a symmetric eigendecomposition:
    /// for a rank-deficient (singular-covariance) fit the lemma precision is
    /// near-singular and the symmetric eigensolver's spurious tiny-negative
    /// eigenvalues give the WRONG `slogdet` sign, whereas the LU pivot signs
    /// match sklearn's `numpy.linalg.slogdet` (verified against the live oracle).
    ///
    /// Shared by [`get_precision`](FittedPCA::get_precision) and
    /// [`score_samples`](FittedPCA::score_samples) so the score path computes the
    /// precision and its `slogdet` once. Cites sklearn `get_precision`
    /// (`_base.py:58-101`) and the `fast_logdet(precision)` term of
    /// `score_samples` (`_pca.py:829`).
    fn precision_and_logdet(&self) -> Result<(Array2<F>, F), FerroError> {
        let precision = self.precision_via_lemma()?;
        // logdet = fast_logdet(precision): sign-aware slogdet, −inf if det ≤ 0
        // (sklearn `extmath.py:126-130`). Dispatched to LAPACK `getrf` for
        // f64/f32 — the SAME factorization `numpy.linalg.slogdet` uses — so the
        // near-singular rank-deficient precision yields the SAME finite-or-`−inf`
        // sign as sklearn (#2110): a hand-rolled partial-pivot LU / the faer
        // self-adjoint inertia disagree with numpy on the determinant sign of
        // these cond-`~1e16` matrices.
        let log_det = fast_logdet_dispatch(&precision)?;
        Ok((precision, log_det))
    }

    /// Compute the precision matrix via sklearn's matrix-inversion (Woodbury)
    /// lemma (`sklearn/decomposition/_base.py:58-101`). See
    /// [`get_precision`](FittedPCA::get_precision) for the algorithm and the
    /// IEEE inf-diagonal handling.
    fn precision_via_lemma(&self) -> Result<Array2<F>, FerroError> {
        let n_features = self.mean_.len();
        let n_comp = self.components_.nrows();
        let nv = self.noise_variance_;

        // Corner case: n_components_ == 0 → eye(n_features) / noise_variance_
        // (sklearn `_base.py:74-75`).
        if n_comp == 0 {
            let mut precision = Array2::<F>::zeros((n_features, n_features));
            let inv_nv = F::one() / nv;
            for i in 0..n_features {
                precision[[i, i]] = inv_nv;
            }
            return Ok(precision);
        }

        // Corner case: noise_variance_ == 0 → inv(get_covariance())
        // (sklearn `_base.py:82-83`). With zero noise the covariance is
        // full-rank, so a direct general inverse is well-defined. sklearn calls
        // `scipy.linalg.inv` (LAPACK getri) here; route f64/f32 through
        // `ferray::linalg::inv_lapack` (the SAME driver) so the result is
        // scipy-identical (#2110), with the Gauss-Jordan fallback for exotic F.
        if nv == F::zero() {
            return inv_dispatch(&self.get_covariance());
        }

        // General branch (noise_variance_ > 0): the matrix inversion lemma.
        //
        // comps (n_comp × n_features). When `whiten`, sklearn rescales each row
        // k by sqrt(exp_var[k]) (`_base.py:88-89`).
        let mut comps = Array2::<F>::zeros((n_comp, n_features));
        for k in 0..n_comp {
            let scale = if self.whiten {
                self.explained_variance_[k].sqrt()
            } else {
                F::one()
            };
            for j in 0..n_features {
                comps[[k, j]] = self.components_[[k, j]] * scale;
            }
        }

        // exp_var_diff[k] = where(exp_var > nv, exp_var - nv, 0)
        // (sklearn `_base.py:90-95`).
        let mut exp_var_diff = Array1::<F>::zeros(n_comp);
        for k in 0..n_comp {
            exp_var_diff[k] = if self.explained_variance_[k] > nv {
                self.explained_variance_[k] - nv
            } else {
                F::zero()
            };
        }

        // P = comps · compsᵀ / nv  (n_comp × n_comp)  (sklearn `_base.py:96`:
        // `components_ @ components_.T / self.noise_variance_`). numpy's `@`
        // matmul then an elementwise `/ nv`; route f64/f32 through ferray `gemm`
        // (openblas, numpy-@-identical) so P is bit-identical to numpy's — the
        // near-singular precision (#2110) amplifies any ULP difference in P
        // through inv_lapack into an O(1) error in the cancelled score quadratic.
        let comps_t_full = comps.t().to_owned(); // (n_features × n_comp)
        let mut p = gemm_dispatch(&comps, &comps_t_full)?; // (n_comp × n_comp)
        for v in p.iter_mut() {
            *v = *v / nv;
        }

        // P[k,k] += 1 / exp_var_diff[k]  (sklearn `_base.py:97`). When
        // exp_var_diff[k] == 0 the IEEE value 1/0 = +inf drives the k-th row/col
        // of P⁻¹ to zero; we reproduce that by inverting only the submatrix over
        // indices with exp_var_diff[k] > 0 and zeroing the dropped positions.
        let mut keep = Vec::with_capacity(n_comp);
        for k in 0..n_comp {
            if exp_var_diff[k] > F::zero() {
                p[[k, k]] = p[[k, k]] + F::one() / exp_var_diff[k];
                keep.push(k);
            }
        }

        // inv_p = P⁻¹ with zero rows/cols for the dropped (inf-diagonal) indices.
        // sklearn inverts P with `scipy.linalg.inv` (LAPACK getri); route f64/f32
        // through `ferray::linalg::inv_lapack` (the SAME driver, scipy-identical,
        // ferray #2117) so the near-singular precision's slogdet sign matches
        // scipy's +1 (#2110); the Gauss-Jordan fallback covers exotic F.
        let mut inv_p = Array2::<F>::zeros((n_comp, n_comp));
        if keep.len() == n_comp {
            inv_p = inv_dispatch(&p)?;
        } else if !keep.is_empty() {
            // Invert only the kept submatrix, scatter back into inv_p.
            let m = keep.len();
            let mut sub = Array2::<F>::zeros((m, m));
            for (ia, &a) in keep.iter().enumerate() {
                for (ib, &b) in keep.iter().enumerate() {
                    sub[[ia, ib]] = p[[a, b]];
                }
            }
            let sub_inv = inv_dispatch(&sub)?;
            for (ia, &a) in keep.iter().enumerate() {
                for (ib, &b) in keep.iter().enumerate() {
                    inv_p[[a, b]] = sub_inv[[ia, ib]];
                }
            }
        }
        // (keep empty ⇒ inv_p stays all-zero — every direction is degenerate.)

        // precision = compsᵀ · inv_p · comps  (n_features × n_features)
        // (sklearn `_base.py:98`: `components_.T @ linalg_inv(precision) @
        // components_`). numpy's `@` chains LEFT-TO-RIGHT, so this is
        // `(compsᵀ @ inv_p) @ comps`. The association order is load-bearing: on
        // the near-singular rank-deficient precision it is what yields scipy's
        // +1 `slogdet` sign (→ finite score) instead of −inf (#2110). Route
        // f64/f32 through `ferray::linalg::gemm` (numpy-@-identical openblas,
        // ferray #2117) evaluated left-to-right; exotic F use ndarray `.dot()`.
        let comps_t = comps.t().to_owned(); // (n_features × n_comp)
        let left = gemm_dispatch(&comps_t, &inv_p)?; // (n_features × n_comp)
        let mut precision = gemm_dispatch(&left, &comps)?; // (n_features × n_features)

        // precision /= -(noise_variance_²)  (sklearn `_base.py:99`).
        let neg_nv2 = -(nv * nv);
        for v in precision.iter_mut() {
            *v = *v / neg_nv2;
        }
        // precision[i,i] += 1 / noise_variance_  (sklearn `_base.py:100`).
        let inv_nv = F::one() / nv;
        for i in 0..n_features {
            precision[[i, i]] = precision[[i, i]] + inv_nv;
        }

        Ok(precision)
    }

    /// Convert a finite `f64` constant into the target float type, returning a
    /// typed error instead of panicking when the conversion is impossible.
    ///
    /// Used by the score path for `0.5`, `2π`, and the sample/feature counts —
    /// all of which always convert cleanly for `f32`/`f64` but could in
    /// principle fail for an exotic `Float` impl (R-CODE-2: no `.unwrap`).
    fn const_f(v: f64) -> Result<F, FerroError> {
        F::from(v).ok_or_else(|| FerroError::NumericalInstability {
            message: "failed to convert a constant into the target float type".into(),
        })
    }

    /// Return the per-sample Gaussian log-likelihood under the probabilistic-PCA
    /// model.
    ///
    /// For centered rows `Xr = X − mean_` and `precision = get_precision()`,
    /// each `ll_i = −0.5 · (Xr_i · precision · Xr_iᵀ)
    ///            − 0.5 · (p · ln(2π) − logdet(precision))`, where
    /// `p = n_features`, `precision` is the Woodbury-lemma precision, and
    /// `logdet(precision) = fast_logdet(precision)` (the sign-aware `slogdet`,
    /// `−inf` when the precision is singular). Mirrors sklearn `PCA.score_samples`
    /// (`sklearn/decomposition/_pca.py:805-830`:
    /// `log_like = -0.5 * sum(Xr * (Xr @ precision), axis=1)`;
    /// `log_like -= 0.5 * (n_features * log(2π) - fast_logdet(precision))`).
    /// For a rank-deficient fit (`n_samples < n_features`) the precision is
    /// finite (the lemma never inverts the singular covariance directly) and the
    /// score follows sklearn's `fast_logdet` exactly, including `−inf` when the
    /// lemma precision is singular.
    ///
    /// # Errors
    ///
    /// - [`FerroError::ShapeMismatch`] if `x.ncols()` differs from the number of
    ///   features seen during fitting.
    /// - [`FerroError::NumericalInstability`] propagated from
    ///   [`get_precision`](FittedPCA::get_precision) (a float-constant conversion
    ///   or small `P`-matrix inversion failure). `whiten` is handled (the
    ///   `sqrt(exp_var)` rescale lives in the lemma, sklearn `_base.py:88-89`).
    pub fn score_samples(&self, x: &Array2<F>) -> Result<Array1<F>, FerroError> {
        let n_features = self.mean_.len();
        if x.ncols() != n_features {
            return Err(FerroError::ShapeMismatch {
                expected: vec![x.nrows(), n_features],
                actual: vec![x.nrows(), x.ncols()],
                context: "FittedPCA::score_samples".into(),
            });
        }

        let (precision, log_det) = self.precision_and_logdet()?;

        // Constant term: 0.5 * (p·ln(2π) − logdet(precision)).
        let two_pi = Self::const_f(2.0 * std::f64::consts::PI)?;
        let half = Self::const_f(0.5)?;
        let p = Self::const_f(n_features as f64)?;
        let const_term = half * (p * two_pi.ln() - log_det);

        // Xr = X − mean_  (n_samples × n_features).
        let n_samples = x.nrows();
        let mut xr = Array2::<F>::zeros((n_samples, n_features));
        for i in 0..n_samples {
            for j in 0..n_features {
                xr[[i, j]] = x[[i, j]] - self.mean_[j];
            }
        }

        // log_like_i = −0.5 · Σ_j Xr[i,j] · (Xr @ precision)[i,j]
        // (sklearn `_pca.py:828`: `-0.5 * sum(Xr * (Xr @ precision), axis=1)`).
        // The `Xr @ precision` is numpy's `@`; route f64/f32 through ferray
        // `gemm` (openblas, numpy-@-identical) so the per-sample quadratic
        // matches sklearn bit-for-bit. With the precision entries ~1e30 and
        // catastrophic cancellation, a nested-loop matmul accumulates in a
        // different order and diverges by ~100% from sklearn (#2110); gemm
        // reproduces openblas's accumulation.
        let xr_prec = gemm_dispatch(&xr, &precision)?; // (n_samples × n_features)

        let mut ll = Array1::<F>::zeros(n_samples);
        for i in 0..n_samples {
            let mut quad = F::zero();
            for j in 0..n_features {
                quad = quad + xr[[i, j]] * xr_prec[[i, j]];
            }
            ll[i] = -half * quad - const_term;
        }

        Ok(ll)
    }

    /// Return the average Gaussian log-likelihood of all samples under the
    /// probabilistic-PCA model: `mean(score_samples(X))`.
    ///
    /// Mirrors sklearn `PCA.score` (`sklearn/decomposition/_pca.py:832-853`:
    /// `float(mean(self.score_samples(X)))`).
    ///
    /// # Errors
    ///
    /// Propagates the errors of [`score_samples`](FittedPCA::score_samples).
    pub fn score(&self, x: &Array2<F>) -> Result<F, FerroError> {
        let ll = self.score_samples(x)?;
        let n = Self::const_f(ll.len() as f64)?;
        let sum = ll.iter().copied().fold(F::zero(), |a, b| a + b);
        Ok(sum / n)
    }

    /// Reconstruct approximate original data from the reduced representation.
    ///
    /// Computes `X_approx = X_reduced @ components + mean`. When whitening is
    /// enabled, each input column j is first multiplied by
    /// `sqrt(explained_variance_[j])` to reverse the `transform` scaling,
    /// mirroring sklearn `inverse_transform`
    /// (`sklearn/decomposition/_base.py:192-196`:
    /// `scaled_components = sqrt(explained_variance_)[:, newaxis] * components_;
    /// X @ scaled_components + mean_`).
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of columns in
    /// `x_reduced` does not equal `n_components`.
    pub fn inverse_transform(&self, x_reduced: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let n_components = self.components_.nrows();
        if x_reduced.ncols() != n_components {
            return Err(FerroError::ShapeMismatch {
                expected: vec![x_reduced.nrows(), n_components],
                actual: vec![x_reduced.nrows(), x_reduced.ncols()],
                context: "FittedPCA::inverse_transform".into(),
            });
        }

        // Whitening: multiply each column j by sqrt(explained_variance_[j])
        // BEFORE projecting back, reversing the `transform` divide. This is
        // algebraically identical to sklearn folding the scale into the
        // components (`_base.py:192-196`). When `whiten` is false the input is
        // used unchanged.
        let x_scaled = if self.whiten {
            let mut x_scaled = x_reduced.to_owned();
            for j in 0..n_components {
                let scale = self.explained_variance_[j].sqrt();
                for v in x_scaled.column_mut(j) {
                    *v = *v * scale;
                }
            }
            std::borrow::Cow::Owned(x_scaled)
        } else {
            std::borrow::Cow::Borrowed(x_reduced)
        };

        // X_approx = X_scaled @ components + mean
        let mut result = x_scaled.dot(&self.components_);
        for mut row in result.rows_mut() {
            for (v, &m) in row.iter_mut().zip(self.mean_.iter()) {
                *v = *v + m;
            }
        }
        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// Jacobi eigendecomposition for symmetric matrices
// ---------------------------------------------------------------------------

/// Perform eigendecomposition of a symmetric matrix using the Jacobi method.
///
/// Returns `(eigenvalues, eigenvectors)` where `eigenvectors` is column-major
/// (column `i` is the eigenvector for `eigenvalues[i]`).
///
/// The eigenvalues are NOT sorted; the caller is responsible for sorting.
fn jacobi_eigen<F: Float + Send + Sync + 'static>(
    a: &Array2<F>,
    max_iter: usize,
) -> Result<(Array1<F>, Array2<F>), FerroError> {
    let n = a.nrows();
    let mut mat = a.to_owned();
    let mut v = Array2::<F>::zeros((n, n));
    // Initialise V to identity.
    for i in 0..n {
        v[[i, i]] = F::one();
    }

    let tol = F::from(1e-12).unwrap_or_else(F::epsilon);

    for iteration in 0..max_iter {
        // Find the largest off-diagonal element.
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
            // Converged.
            let eigenvalues = Array1::from_shape_fn(n, |i| mat[[i, i]]);
            return Ok((eigenvalues, v));
        }

        // Compute the Jacobi rotation.
        let app = mat[[p, p]];
        let aqq = mat[[q, q]];
        let apq = mat[[p, q]];

        let theta = if (app - aqq).abs() < tol {
            F::from(std::f64::consts::FRAC_PI_4).unwrap_or_else(F::one)
        } else {
            let tau = (aqq - app) / (F::from(2.0).unwrap() * apq);
            // t = sign(tau) / (|tau| + sqrt(1 + tau^2))
            let t = if tau >= F::zero() {
                F::one() / (tau.abs() + (F::one() + tau * tau).sqrt())
            } else {
                -F::one() / (tau.abs() + (F::one() + tau * tau).sqrt())
            };
            t.atan()
        };

        let c = theta.cos();
        let s = theta.sin();

        // Apply rotation to mat: mat' = G^T mat G
        // Update rows/columns p and q.
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

        // Update eigenvector matrix.
        for i in 0..n {
            let vip = v[[i, p]];
            let viq = v[[i, q]];
            v[[i, p]] = c * vip - s * viq;
            v[[i, q]] = s * vip + c * viq;
        }

        let _ = iteration; // suppress unused warning
    }

    Err(FerroError::ConvergenceFailure {
        iterations: max_iter,
        message: "Jacobi eigendecomposition did not converge".into(),
    })
}

// ---------------------------------------------------------------------------
// faer-accelerated eigendecomposition for f64 and f32
// ---------------------------------------------------------------------------

/// Perform eigendecomposition of a symmetric matrix using faer's optimised
/// self-adjoint eigensolver. Returns `(eigenvalues, eigenvectors)` where
/// `eigenvectors` is column-major (column `i` is the eigenvector for
/// `eigenvalues[i]`). Eigenvalues are returned in ascending order.
fn faer_eigen_f64(a: &Array2<f64>) -> Result<(Array1<f64>, Array2<f64>), FerroError> {
    let n = a.nrows();
    let mat = faer::Mat::from_fn(n, n, |i, j| a[[i, j]]);
    let decomp = mat.self_adjoint_eigen(faer::Side::Lower).map_err(|e| {
        FerroError::NumericalInstability {
            message: format!("faer symmetric eigendecomposition failed: {e:?}"),
        }
    })?;

    let eigenvalues = Array1::from_shape_fn(n, |i| decomp.S().column_vector()[i]);
    let eigenvectors = Array2::from_shape_fn((n, n), |(i, j)| decomp.U()[(i, j)]);

    Ok((eigenvalues, eigenvectors))
}

/// Perform eigendecomposition of a symmetric f32 matrix using faer's
/// optimised self-adjoint eigensolver.
fn faer_eigen_f32(a: &Array2<f32>) -> Result<(Array1<f32>, Array2<f32>), FerroError> {
    let n = a.nrows();
    let mat = faer::Mat::from_fn(n, n, |i, j| a[[i, j]]);
    let decomp = mat.self_adjoint_eigen(faer::Side::Lower).map_err(|e| {
        FerroError::NumericalInstability {
            message: format!("faer symmetric eigendecomposition failed: {e:?}"),
        }
    })?;

    let eigenvalues = Array1::from_shape_fn(n, |i| decomp.S().column_vector()[i]);
    let eigenvectors = Array2::from_shape_fn((n, n), |(i, j)| decomp.U()[(i, j)]);

    Ok((eigenvalues, eigenvectors))
}

/// Thin SVD of `a` (shape `m × n`) via `ferray::linalg::svd_lapack` (LAPACK
/// `gesdd`), mirroring scipy's `linalg.svd(X_centered, full_matrices=False)` of
/// sklearn's `full` solver (`sklearn/decomposition/_pca.py:588`).
///
/// sklearn's `_fit_full` computes the SVD via `scipy.linalg.svd`, which calls
/// LAPACK `gesdd`. Routing ferrolearn through `ferray::linalg::svd_lapack` —
/// the same LAPACK `gesdd` driver, proven bit-identical to scipy on all
/// singular values incl. the rank-deficient noise-floor value (ferray #2116) —
/// makes the reconstructed `noise_variance_` (the discarded near-zero tail) and
/// hence `get_precision`/`score` byte-for-byte match sklearn, where faer's
/// thin-SVD floor differed (#2110).
///
/// Returns `(s, vt)` where `s` are the `min(m, n)` singular values in
/// NON-INCREASING order (LAPACK `gesdd` contract) and `vt` is `Vᵀ`, shape
/// `(min(m, n), n)` — i.e. row `k` is the `k`-th right singular vector, exactly
/// sklearn's `Vt` whose rows become `components_`. `U` is not returned:
/// sklearn's `svd_flip(U, Vt, u_based_decision=False)` (`_pca.py:647`) only
/// needs `Vt`.
///
/// # Bridge
///
/// The centred-X `ndarray::Array2<f64>` is copied into a `ferray-core`
/// `Array<f64, Ix2>` via the public `Array::from_vec` constructor (row-major,
/// the layout `svd_lapack` requires), `svd_lapack(.., full_matrices=false)` is
/// called, and the returned ferray `S`/`Vt` arrays are converted back to
/// `ndarray` via `into_ndarray()` (R-SUBSTRATE-4 boundary bridge).
fn ferray_svd_lapack_f64(a: &Array2<f64>) -> Result<(Array1<f64>, Array2<f64>), FerroError> {
    let (m, n) = a.dim();
    // ferray-core wants a contiguous row-major buffer; collect X_centered in
    // C order (ndarray `iter()` yields logical row-major order).
    let data: Vec<f64> = a.iter().copied().collect();
    let fa = ferray::Array::<f64, ferray::Ix2>::from_vec(ferray::Ix2::new([m, n]), data).map_err(
        |e| FerroError::NumericalInstability {
            message: format!("ferray array construction failed: {e}"),
        },
    )?;
    let (_u, s, vt) =
        ferray::linalg::svd_lapack(&fa, false).map_err(|e| FerroError::NumericalInstability {
            message: format!("ferray svd_lapack (gesdd) failed: {e}"),
        })?;
    // svd_lapack returns ferray-core arrays: S length min(m, n) descending,
    // Vt shape (min(m, n), n). Bridge back to ndarray (into_ndarray = move).
    let s_nd: Array1<f64> = s.into_ndarray();
    let vt_nd: Array2<f64> = vt.into_ndarray();
    Ok((s_nd, vt_nd))
}

/// Thin SVD of an f32 matrix via `ferray::linalg::svd_lapack` (LAPACK `gesdd`).
/// See [`ferray_svd_lapack_f64`].
fn ferray_svd_lapack_f32(a: &Array2<f32>) -> Result<(Array1<f32>, Array2<f32>), FerroError> {
    let (m, n) = a.dim();
    let data: Vec<f32> = a.iter().copied().collect();
    let fa = ferray::Array::<f32, ferray::Ix2>::from_vec(ferray::Ix2::new([m, n]), data).map_err(
        |e| FerroError::NumericalInstability {
            message: format!("ferray array construction failed: {e}"),
        },
    )?;
    let (_u, s, vt) =
        ferray::linalg::svd_lapack(&fa, false).map_err(|e| FerroError::NumericalInstability {
            message: format!("ferray svd_lapack (gesdd) failed: {e}"),
        })?;
    let s_nd: Array1<f32> = s.into_ndarray();
    let vt_nd: Array2<f32> = vt.into_ndarray();
    Ok((s_nd, vt_nd))
}

/// Invert a square `f64` matrix via `ferray::linalg::inv_lapack` (LAPACK
/// `getrf` + `getri`) — the SAME path `scipy.linalg.inv` calls, which sklearn's
/// `get_precision` uses for the inner `n_components × n_components` inverse of
/// the matrix-inversion lemma (`sklearn/decomposition/_base.py:80`,`:98`).
///
/// The hand-rolled Gauss-Jordan `invert_general` produces a numerically
/// different inverse from LAPACK `getri`; on a near-singular lemma matrix `P`
/// (rank-deficient fits, cond `~1e16`) that difference flips the `slogdet` SIGN
/// of the reconstructed precision, so `fast_logdet` returns `−inf` and the score
/// goes to `−inf` while sklearn stays finite (#2110). Routing through
/// `ferray::linalg::inv_lapack` (proven bit-identical to scipy, ferray #2117)
/// reproduces sklearn's `+1` sign and the finite score.
///
/// # Bridge
///
/// `a` (`ndarray::Array2<f64>`, row-major logical order) is copied into a
/// `ferray-core` `Array<f64, Ix2>` via `Array::from_vec`, inverted, and bridged
/// back via `into_ndarray()` — the SAME pattern as [`ferray_svd_lapack_f64`]
/// (R-SUBSTRATE-4 boundary bridge).
fn ferray_inv_lapack_f64(a: &Array2<f64>) -> Result<Array2<f64>, FerroError> {
    let (m, n) = a.dim();
    let data: Vec<f64> = a.iter().copied().collect();
    let fa = ferray::Array::<f64, ferray::Ix2>::from_vec(ferray::Ix2::new([m, n]), data).map_err(
        |e| FerroError::NumericalInstability {
            message: format!("ferray array construction failed: {e}"),
        },
    )?;
    let inv = ferray::linalg::inv_lapack(&fa).map_err(|e| FerroError::NumericalInstability {
        message: format!("ferray inv_lapack (getrf+getri) failed: {e}"),
    })?;
    Ok(inv.into_ndarray())
}

/// Invert a square `f32` matrix via `ferray::linalg::inv_lapack` (LAPACK
/// `getrf` + `getri`). See [`ferray_inv_lapack_f64`].
fn ferray_inv_lapack_f32(a: &Array2<f32>) -> Result<Array2<f32>, FerroError> {
    let (m, n) = a.dim();
    let data: Vec<f32> = a.iter().copied().collect();
    let fa = ferray::Array::<f32, ferray::Ix2>::from_vec(ferray::Ix2::new([m, n]), data).map_err(
        |e| FerroError::NumericalInstability {
            message: format!("ferray array construction failed: {e}"),
        },
    )?;
    let inv = ferray::linalg::inv_lapack(&fa).map_err(|e| FerroError::NumericalInstability {
        message: format!("ferray inv_lapack (getrf+getri) failed: {e}"),
    })?;
    Ok(inv.into_ndarray())
}

/// Matrix product `a @ b` of `f64` matrices via `ferray::linalg::gemm`
/// (openblas `gemm`) — numpy-`@`-identical (ferray #2117), mirroring numpy's
/// `@` in sklearn's `get_precision` lemma (`sklearn/decomposition/_base.py:98`:
/// `components_.T @ linalg_inv(precision) @ components_`).
///
/// The association order of the two products is LEFT-TO-RIGHT and MATTERS: the
/// near-singular precision's `slogdet` sign is only `+1` (matching scipy) when
/// the products are evaluated `(compsᵀ @ inv_p) @ comps`, exactly numpy's
/// left-to-right `@` chaining (#2110). A naive nested-loop matmul or a different
/// association flips the sign.
///
/// # Bridge
///
/// Both operands are copied into `ferray-core` arrays via `Array::from_vec`,
/// multiplied, and bridged back via `into_ndarray()` (R-SUBSTRATE-4). The inner
/// dimension must agree (`a.ncols() == b.nrows()`).
fn ferray_gemm_f64(a: &Array2<f64>, b: &Array2<f64>) -> Result<Array2<f64>, FerroError> {
    let (am, an) = a.dim();
    let (bm, bn) = b.dim();
    if an != bm {
        return Err(FerroError::ShapeMismatch {
            expected: vec![am, an],
            actual: vec![bm, bn],
            context: "ferray_gemm_f64: inner dimensions must agree".into(),
        });
    }
    let a_data: Vec<f64> = a.iter().copied().collect();
    let b_data: Vec<f64> = b.iter().copied().collect();
    let fa = ferray::Array::<f64, ferray::Ix2>::from_vec(ferray::Ix2::new([am, an]), a_data)
        .map_err(|e| FerroError::NumericalInstability {
            message: format!("ferray array construction failed: {e}"),
        })?;
    let fb = ferray::Array::<f64, ferray::Ix2>::from_vec(ferray::Ix2::new([bm, bn]), b_data)
        .map_err(|e| FerroError::NumericalInstability {
            message: format!("ferray array construction failed: {e}"),
        })?;
    let prod = ferray::linalg::gemm(&fa, &fb).map_err(|e| FerroError::NumericalInstability {
        message: format!("ferray gemm (openblas) failed: {e}"),
    })?;
    Ok(prod.into_ndarray())
}

/// Matrix product `a @ b` of `f32` matrices via `ferray::linalg::gemm`
/// (openblas `gemm`). See [`ferray_gemm_f64`].
fn ferray_gemm_f32(a: &Array2<f32>, b: &Array2<f32>) -> Result<Array2<f32>, FerroError> {
    let (am, an) = a.dim();
    let (bm, bn) = b.dim();
    if an != bm {
        return Err(FerroError::ShapeMismatch {
            expected: vec![am, an],
            actual: vec![bm, bn],
            context: "ferray_gemm_f32: inner dimensions must agree".into(),
        });
    }
    let a_data: Vec<f32> = a.iter().copied().collect();
    let b_data: Vec<f32> = b.iter().copied().collect();
    let fa = ferray::Array::<f32, ferray::Ix2>::from_vec(ferray::Ix2::new([am, an]), a_data)
        .map_err(|e| FerroError::NumericalInstability {
            message: format!("ferray array construction failed: {e}"),
        })?;
    let fb = ferray::Array::<f32, ferray::Ix2>::from_vec(ferray::Ix2::new([bm, bn]), b_data)
        .map_err(|e| FerroError::NumericalInstability {
            message: format!("ferray array construction failed: {e}"),
        })?;
    let prod = ferray::linalg::gemm(&fa, &fb).map_err(|e| FerroError::NumericalInstability {
        message: format!("ferray gemm (openblas) failed: {e}"),
    })?;
    Ok(prod.into_ndarray())
}

/// Dispatch a square-matrix inverse to LAPACK `getrf`+`getri` (via
/// `ferray::linalg::inv_lapack`) for f64/f32, falling back to the hand-rolled
/// Gauss-Jordan [`invert_general`] for exotic float types.
///
/// sklearn's `get_precision` (`sklearn/decomposition/_base.py:80`,`:98`) inverts
/// the inner lemma matrix with `scipy.linalg.inv` (LAPACK `getri`). For f64/f32
/// — the tested paths — `ferray::linalg::inv_lapack` is the SAME driver, proven
/// bit-identical to scipy (ferray #2117); routing through it makes the
/// near-singular precision's `slogdet` sign match scipy's `+1` (#2110). Other
/// `F` keep the Gauss-Jordan fallback (no LAPACK binding for exotic floats).
fn inv_dispatch<F: Float + Send + Sync + 'static>(a: &Array2<F>) -> Result<Array2<F>, FerroError> {
    // SAFETY: TypeId is checked at runtime; the transmutes only reinterpret
    // between identical types (Array<f64> <-> Array<F> when F == f64, etc.),
    // exactly as in `svd_dispatch`/`eigen_dispatch`.
    if TypeId::of::<F>() == TypeId::of::<f64>() {
        let a_f64: &Array2<f64> = unsafe { &*(std::ptr::from_ref(a).cast::<Array2<f64>>()) };
        let inv = ferray_inv_lapack_f64(a_f64)?;
        let inv_f: Array2<F> = unsafe { std::mem::transmute_copy::<Array2<f64>, Array2<F>>(&inv) };
        std::mem::forget(inv);
        Ok(inv_f)
    } else if TypeId::of::<F>() == TypeId::of::<f32>() {
        let a_f32: &Array2<f32> = unsafe { &*(std::ptr::from_ref(a).cast::<Array2<f32>>()) };
        let inv = ferray_inv_lapack_f32(a_f32)?;
        let inv_f: Array2<F> = unsafe { std::mem::transmute_copy::<Array2<f32>, Array2<F>>(&inv) };
        std::mem::forget(inv);
        Ok(inv_f)
    } else {
        invert_general(a)
    }
}

/// Dispatch the matrix product `a @ b` to openblas `gemm` (via
/// `ferray::linalg::gemm`) for f64/f32, falling back to ndarray's `.dot()` for
/// exotic float types.
///
/// Mirrors numpy's `@` in sklearn's `get_precision` lemma
/// (`sklearn/decomposition/_base.py:98`). For f64/f32 `ferray::linalg::gemm` is
/// numpy-`@`-identical (openblas, ferray #2117); the LEFT-TO-RIGHT association
/// `gemm(gemm(compsᵀ, inv_p), comps)` is what reproduces scipy's `+1` precision
/// `slogdet` sign (#2110). Exotic `F` use `a.dot(b)`.
fn gemm_dispatch<F: Float + Send + Sync + 'static>(
    a: &Array2<F>,
    b: &Array2<F>,
) -> Result<Array2<F>, FerroError> {
    // SAFETY: TypeId is checked at runtime; the transmutes only reinterpret
    // between identical types (Array<f64> <-> Array<F> when F == f64, etc.).
    if TypeId::of::<F>() == TypeId::of::<f64>() {
        let a_f64: &Array2<f64> = unsafe { &*(std::ptr::from_ref(a).cast::<Array2<f64>>()) };
        let b_f64: &Array2<f64> = unsafe { &*(std::ptr::from_ref(b).cast::<Array2<f64>>()) };
        let prod = ferray_gemm_f64(a_f64, b_f64)?;
        let prod_f: Array2<F> =
            unsafe { std::mem::transmute_copy::<Array2<f64>, Array2<F>>(&prod) };
        std::mem::forget(prod);
        Ok(prod_f)
    } else if TypeId::of::<F>() == TypeId::of::<f32>() {
        let a_f32: &Array2<f32> = unsafe { &*(std::ptr::from_ref(a).cast::<Array2<f32>>()) };
        let b_f32: &Array2<f32> = unsafe { &*(std::ptr::from_ref(b).cast::<Array2<f32>>()) };
        let prod = ferray_gemm_f32(a_f32, b_f32)?;
        let prod_f: Array2<F> =
            unsafe { std::mem::transmute_copy::<Array2<f32>, Array2<F>>(&prod) };
        std::mem::forget(prod);
        Ok(prod_f)
    } else {
        Ok(a.dot(b))
    }
}

/// Dispatch the thin SVD of the centred data matrix to LAPACK `gesdd` (via
/// `ferray::linalg::svd_lapack`) for f64/f32, falling back to an
/// eigendecomposition of the Gram matrix for other float types.
///
/// Returns `(s, vt)`: `s` are the `min(m, n)` singular values in non-increasing
/// order and `vt` is `Vᵀ` (`(min(m, n), n)`, row `k` = `k`-th right singular
/// vector). This is the `U, S, Vt = svd(X_centered, full_matrices=False)` of
/// sklearn's `full` solver (`sklearn/decomposition/_pca.py:588`), whose
/// `scipy.linalg.svd` calls the SAME LAPACK `gesdd` driver — so for f64/f32 the
/// reconstructed spectrum (incl. the rank-deficient tail) is bit-identical to
/// sklearn.
///
/// The exotic-`F` fallback reconstructs the right singular vectors / singular
/// values from the Jacobi eigendecomposition of `Cᵀ = X_cᵀ X_c` (eigenvectors
/// are the right singular vectors `V`; `S = sqrt(max(eigval, 0))`). f64/f32 —
/// the tested paths — use LAPACK directly.
fn svd_dispatch<F: Float + Send + Sync + 'static>(
    x_centered: &Array2<F>,
    max_jacobi_iter: usize,
) -> Result<(Array1<F>, Array2<F>), FerroError> {
    // SAFETY: we check TypeId at runtime and only reinterpret when the types
    // match. The transmutes are between identical types (Array<f64> -> Array<F>
    // when F == f64, etc.), exactly as in `eigen_dispatch`.
    if TypeId::of::<F>() == TypeId::of::<f64>() {
        let x_f64: &Array2<f64> =
            unsafe { &*(std::ptr::from_ref(x_centered).cast::<Array2<f64>>()) };
        let (s, vt) = ferray_svd_lapack_f64(x_f64)?;
        let s_f: Array1<F> = unsafe { std::mem::transmute_copy::<Array1<f64>, Array1<F>>(&s) };
        let vt_f: Array2<F> = unsafe { std::mem::transmute_copy::<Array2<f64>, Array2<F>>(&vt) };
        std::mem::forget(s);
        std::mem::forget(vt);
        Ok((s_f, vt_f))
    } else if TypeId::of::<F>() == TypeId::of::<f32>() {
        let x_f32: &Array2<f32> =
            unsafe { &*(std::ptr::from_ref(x_centered).cast::<Array2<f32>>()) };
        let (s, vt) = ferray_svd_lapack_f32(x_f32)?;
        let s_f: Array1<F> = unsafe { std::mem::transmute_copy::<Array1<f32>, Array1<F>>(&s) };
        let vt_f: Array2<F> = unsafe { std::mem::transmute_copy::<Array2<f32>, Array2<F>>(&vt) };
        std::mem::forget(s);
        std::mem::forget(vt);
        Ok((s_f, vt_f))
    } else {
        // Exotic-F fallback: eigendecompose the Gram matrix C = X_cᵀ X_c
        // (n_features × n_features). Its eigenvectors are the right singular
        // vectors V; singular values are sqrt(max(eigval, 0)). Eigenvalues are
        // sorted descending so the SVD ordering matches faer's.
        let (n_samples, n_features) = x_centered.dim();
        let size = n_samples.min(n_features);
        let xt = x_centered.t();
        let gram = xt.dot(x_centered);
        let (eigenvalues, eigenvectors) = jacobi_eigen(&gram, max_jacobi_iter)?;
        let mut indices: Vec<usize> = (0..n_features).collect();
        indices.sort_by(|&a, &b| {
            eigenvalues[b]
                .partial_cmp(&eigenvalues[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut s = Array1::<F>::zeros(size);
        let mut vt = Array2::<F>::zeros((size, n_features));
        for (k, &idx) in indices.iter().take(size).enumerate() {
            let ev = eigenvalues[idx];
            let ev_clamped = if ev < F::zero() { F::zero() } else { ev };
            s[k] = ev_clamped.sqrt();
            for j in 0..n_features {
                vt[[k, j]] = eigenvectors[[j, idx]];
            }
        }
        Ok((s, vt))
    }
}

/// Dispatch eigendecomposition to faer for f64/f32, falling back to
/// Jacobi for other float types.
///
/// Returns `(eigenvalues, eigenvectors)` where `eigenvectors` is column-major.
/// Eigenvalues are NOT guaranteed to be sorted in any particular order.
fn eigen_dispatch<F: Float + Send + Sync + 'static>(
    a: &Array2<F>,
    max_jacobi_iter: usize,
) -> Result<(Array1<F>, Array2<F>), FerroError> {
    // SAFETY: We check TypeId at runtime and only reinterpret when the types
    // match. The transmutes are between identical types (Array<f64> -> Array<F>
    // when F == f64, etc.).
    if TypeId::of::<F>() == TypeId::of::<f64>() {
        // F is f64 — cast through raw pointers to call the f64 specialisation.
        let a_f64: &Array2<f64> = unsafe { &*(std::ptr::from_ref(a).cast::<Array2<f64>>()) };
        let (eigenvalues, eigenvectors) = faer_eigen_f64(a_f64)?;
        // Cast back from f64 to F (which is f64).
        let eigenvalues_f: Array1<F> =
            unsafe { std::mem::transmute_copy::<Array1<f64>, Array1<F>>(&eigenvalues) };
        let eigenvectors_f: Array2<F> =
            unsafe { std::mem::transmute_copy::<Array2<f64>, Array2<F>>(&eigenvectors) };
        // Prevent double-free of the originals.
        std::mem::forget(eigenvalues);
        std::mem::forget(eigenvectors);
        Ok((eigenvalues_f, eigenvectors_f))
    } else if TypeId::of::<F>() == TypeId::of::<f32>() {
        let a_f32: &Array2<f32> = unsafe { &*(std::ptr::from_ref(a).cast::<Array2<f32>>()) };
        let (eigenvalues, eigenvectors) = faer_eigen_f32(a_f32)?;
        let eigenvalues_f: Array1<F> =
            unsafe { std::mem::transmute_copy::<Array1<f32>, Array1<F>>(&eigenvalues) };
        let eigenvectors_f: Array2<F> =
            unsafe { std::mem::transmute_copy::<Array2<f32>, Array2<F>>(&eigenvectors) };
        std::mem::forget(eigenvalues);
        std::mem::forget(eigenvectors);
        Ok((eigenvalues_f, eigenvectors_f))
    } else {
        // Fallback to Jacobi for exotic float types.
        jacobi_eigen(a, max_jacobi_iter)
    }
}

/// Invert a square matrix by Gauss-Jordan elimination with partial pivoting.
///
/// Used by the PCA precision lemma for the small `n_components × n_components`
/// matrix `P` and, in the `noise_variance_ == 0` corner case, for the
/// (full-rank) covariance. Mirrors scipy's `linalg.inv` (sklearn `_base.py:80`)
/// as a dense LU-based solve. No panics — returns
/// [`FerroError::NumericalInstability`] when a pivot is exactly zero (singular).
fn invert_general<F: Float + Send + Sync + 'static>(
    a: &Array2<F>,
) -> Result<Array2<F>, FerroError> {
    let n = a.nrows();
    if n != a.ncols() {
        return Err(FerroError::ShapeMismatch {
            expected: vec![n, n],
            actual: vec![a.nrows(), a.ncols()],
            context: "invert_general: matrix must be square".into(),
        });
    }

    // Augmented matrix [A | I].
    let mut aug = Array2::<F>::zeros((n, 2 * n));
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = a[[i, j]];
        }
        aug[[i, n + i]] = F::one();
    }

    // Forward elimination with partial pivoting.
    for col in 0..n {
        let mut max_val = aug[[col, col]].abs();
        let mut max_row = col;
        for row in (col + 1)..n {
            let val = aug[[row, col]].abs();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }
        if max_val == F::zero() {
            return Err(FerroError::NumericalInstability {
                message: "invert_general: matrix is singular (zero pivot)".into(),
            });
        }
        if max_row != col {
            for j in 0..(2 * n) {
                let tmp = aug[[col, j]];
                aug[[col, j]] = aug[[max_row, j]];
                aug[[max_row, j]] = tmp;
            }
        }
        let pivot = aug[[col, col]];
        for row in (col + 1)..n {
            let factor = aug[[row, col]] / pivot;
            for j in col..(2 * n) {
                let above = aug[[col, j]];
                aug[[row, j]] = aug[[row, j]] - factor * above;
            }
        }
    }

    // Back substitution.
    for col in (0..n).rev() {
        let pivot = aug[[col, col]];
        for j in 0..(2 * n) {
            aug[[col, j]] = aug[[col, j]] / pivot;
        }
        for row in 0..col {
            let factor = aug[[row, col]];
            for j in 0..(2 * n) {
                let below = aug[[col, j]];
                aug[[row, j]] = aug[[row, j]] - factor * below;
            }
        }
    }

    // Extract inverse.
    let mut inv = Array2::<F>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            inv[[i, j]] = aug[[i, n + j]];
        }
    }
    Ok(inv)
}

/// Sign and natural-log-magnitude of `det(a)` via LU with partial pivoting.
///
/// Returns `(sign, ld)` where `sign ∈ {−1, 0, +1}` and `ld = Σ ln|U_ii|` (the
/// log-magnitude of the determinant). `sign == 0` (an exactly-zero pivot →
/// singular `a`) carries `ld = −∞`. This mirrors `numpy.linalg.slogdet`'s
/// LU (LAPACK `getrf`) accumulation: `sign = (permutation parity) · Π
/// sign(U_ii)`, `ld = Σ ln|U_ii|`.
///
/// CAVEAT: on a near-singular INDEFINITE matrix (the rank-deficient lemma
/// precision, cond `~1e16`) the partial-pivot SIGN can disagree with LAPACK
/// `getrf` (different pivot ties / cancellation). The MAGNITUDE `ld` is robust;
/// callers that need a numpy-faithful sign take it from
/// [`fast_logdet_dispatch`] (LAPACK `getrf`) instead. This routine is the
/// exotic-`F` fallback only (no LAPACK binding for non-f32/f64 floats).
fn slogdet_lu_parts<F: Float + Send + Sync + 'static>(a: &Array2<F>) -> Result<(F, F), FerroError> {
    let n = a.nrows();
    if n != a.ncols() {
        return Err(FerroError::ShapeMismatch {
            expected: vec![n, n],
            actual: vec![a.nrows(), a.ncols()],
            context: "slogdet_lu: matrix must be square".into(),
        });
    }
    let neg_inf = F::neg_infinity();
    if n == 0 {
        // log|det| of the 0×0 matrix is ln(1) = 0 (empty product), sign +1.
        return Ok((F::one(), F::zero()));
    }

    let mut m = a.clone();
    let mut sign = F::one();
    let neg_one = -F::one();
    let mut ld = F::zero();

    for col in 0..n {
        // Partial pivot: largest-magnitude entry in/below the diagonal.
        let mut max_val = m[[col, col]].abs();
        let mut max_row = col;
        for row in (col + 1)..n {
            let val = m[[row, col]].abs();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }
        if max_row != col {
            for j in 0..n {
                let tmp = m[[col, j]];
                m[[col, j]] = m[[max_row, j]];
                m[[max_row, j]] = tmp;
            }
            sign = sign * neg_one;
        }
        let pivot = m[[col, col]];
        if pivot == F::zero() {
            // Singular: det == 0 → sign 0, ld −∞.
            return Ok((F::zero(), neg_inf));
        }
        if pivot < F::zero() {
            sign = sign * neg_one;
        }
        ld = ld + pivot.abs().ln();
        // Eliminate below using the rank-1 update over the trailing submatrix.
        for row in (col + 1)..n {
            let factor = m[[row, col]] / pivot;
            for j in (col + 1)..n {
                let above = m[[col, j]];
                m[[row, j]] = m[[row, j]] - factor * above;
            }
        }
    }

    Ok((sign, ld))
}

// LAPACK Fortran LU-factorization symbols (`dgetrf_`/`sgetrf_`) — a leaf
// BLAS/FFI shim (R-CODE-1). numpy's `linalg.slogdet` and scipy's `lu_factor`
// both call LAPACK `getrf`; the determinant sign sklearn's `fast_logdet` reads
// (`utils/extmath.py:127`) is reproducible ONLY through `getrf`, because on a
// singular fit `getrf` produces an EXACTLY-zero diagonal pivot (sign 0 → −inf)
// while a partial-pivot LU or the faer self-adjoint inertia find tiny NONZERO
// pivots and disagree with numpy on whether the determinant is zero (#2110).
//
// The OpenBLAS shared object (which also backs `ferray::linalg::svd_lapack`/
// `inv_lapack`/`gemm`) exports these Fortran symbols (`dgetrf_`/`sgetrf_`); we
// link them transitively via the `ferray` dependency's `openblas` feature
// (`openblas-src`, system mode). ferray exposes no `getrf`-backed `slogdet`
// (its `ferray::linalg::slogdet` is faer-LU and diverges from numpy on these
// ill-conditioned matrices) — a substrate gap (R-SUBSTRATE-5); until ferray
// ships a numpy-identical `slogdet`, this leaf shim is the sanctioned bridge.
unsafe extern "C" {
    // SUBROUTINE DGETRF( M, N, A, LDA, IPIV, INFO ) (netlib `dgetrf.f`).
    fn dgetrf_(
        m: *const i32,
        n: *const i32,
        a: *mut f64,
        lda: *const i32,
        ipiv: *mut i32,
        info: *mut i32,
    );
    // SUBROUTINE SGETRF( M, N, A, LDA, IPIV, INFO ) (netlib `sgetrf.f`).
    fn sgetrf_(
        m: *const i32,
        n: *const i32,
        a: *mut f32,
        lda: *const i32,
        ipiv: *mut i32,
        info: *mut i32,
    );
}

/// `fast_logdet` of a square `f64` matrix via LAPACK `getrf` — bit-faithful to
/// `numpy.linalg.slogdet` (sklearn `fast_logdet`, `utils/extmath.py:93-130`).
///
/// Returns `ld = Σ ln|U_ii|` when `det > 0` (`sign > 0`), else `−inf` (sklearn
/// `extmath.py:128-130`). The SIGN is `(permutation parity) · Π sign(U_ii)`, and
/// an EXACTLY-zero `U_ii` (singular, as LAPACK `getrf` reports it via `info > 0`)
/// makes the sign 0 → `−inf`. This is the ONLY path that reproduces numpy on the
/// rank-deficient PCA precision: whiten=True → finite (sign +1), whiten=False →
/// `−inf` (getrf hits an exact-zero pivot), exactly as sklearn (#2110).
///
/// `getrf` factorizes COLUMN-major; the row-major `a` is transposed into a
/// column-major scratch buffer first (`det(Aᵀ) == det(A)`, so the transpose does
/// not change the determinant).
fn fast_logdet_getrf_f64(a: &Array2<f64>) -> Result<f64, FerroError> {
    let n = a.nrows();
    if n != a.ncols() {
        return Err(FerroError::ShapeMismatch {
            expected: vec![n, n],
            actual: vec![a.nrows(), a.ncols()],
            context: "fast_logdet_getrf_f64: matrix must be square".into(),
        });
    }
    if n == 0 {
        return Ok(0.0);
    }
    // Column-major buffer: col_major[i + j*n] = a[i, j].
    let mut buf = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..n {
            buf[i + j * n] = a[[i, j]];
        }
    }
    let n_i = i32::try_from(n).map_err(|_| FerroError::NumericalInstability {
        message: "fast_logdet_getrf_f64: matrix dimension exceeds i32".into(),
    })?;
    let mut ipiv = vec![0_i32; n];
    let mut info = 0_i32;
    // SAFETY: `buf` is n*n, `ipiv` is n, all pointers valid for the call; `lda`
    // == n == leading dimension of the column-major n×n matrix. LAPACK writes
    // the LU factors into `buf` and the pivots into `ipiv`; `info` receives the
    // status. The symbol resolves from the OpenBLAS `.so` (linked via ferray).
    unsafe {
        dgetrf_(
            &n_i,
            &n_i,
            buf.as_mut_ptr(),
            &n_i,
            ipiv.as_mut_ptr(),
            &mut info,
        );
    }
    if info < 0 {
        return Err(FerroError::NumericalInstability {
            message: format!("LAPACK dgetrf reported illegal argument {info}"),
        });
    }
    // info > 0 ⇒ U[info-1, info-1] is exactly zero ⇒ singular ⇒ det 0 ⇒ −inf.
    if info > 0 {
        return Ok(f64::NEG_INFINITY);
    }
    // sign = (permutation parity from ipiv) · Π sign(U_ii); ld = Σ ln|U_ii|.
    let mut sign = 1.0_f64;
    let mut ld = 0.0_f64;
    for k in 0..n {
        let u = buf[k + k * n];
        if u == 0.0 {
            return Ok(f64::NEG_INFINITY);
        }
        if u < 0.0 {
            sign = -sign;
        }
        ld += u.abs().ln();
        // LAPACK ipiv is 1-based; a swap of row k with ipiv[k]-1 (≠ k) flips sign.
        if ipiv[k] - 1 != i32::try_from(k).unwrap_or(i32::MAX) {
            sign = -sign;
        }
    }
    if sign > 0.0 {
        Ok(ld)
    } else {
        Ok(f64::NEG_INFINITY)
    }
}

/// `fast_logdet` of a square `f32` matrix via LAPACK `getrf`. See
/// [`fast_logdet_getrf_f64`].
fn fast_logdet_getrf_f32(a: &Array2<f32>) -> Result<f32, FerroError> {
    let n = a.nrows();
    if n != a.ncols() {
        return Err(FerroError::ShapeMismatch {
            expected: vec![n, n],
            actual: vec![a.nrows(), a.ncols()],
            context: "fast_logdet_getrf_f32: matrix must be square".into(),
        });
    }
    if n == 0 {
        return Ok(0.0);
    }
    let mut buf = vec![0.0_f32; n * n];
    for i in 0..n {
        for j in 0..n {
            buf[i + j * n] = a[[i, j]];
        }
    }
    let n_i = i32::try_from(n).map_err(|_| FerroError::NumericalInstability {
        message: "fast_logdet_getrf_f32: matrix dimension exceeds i32".into(),
    })?;
    let mut ipiv = vec![0_i32; n];
    let mut info = 0_i32;
    // SAFETY: identical contract to the f64 path; `sgetrf_` is the f32 LAPACK
    // symbol from the same OpenBLAS `.so`.
    unsafe {
        sgetrf_(
            &n_i,
            &n_i,
            buf.as_mut_ptr(),
            &n_i,
            ipiv.as_mut_ptr(),
            &mut info,
        );
    }
    if info < 0 {
        return Err(FerroError::NumericalInstability {
            message: format!("LAPACK sgetrf reported illegal argument {info}"),
        });
    }
    if info > 0 {
        return Ok(f32::NEG_INFINITY);
    }
    let mut sign = 1.0_f32;
    let mut ld = 0.0_f32;
    for k in 0..n {
        let u = buf[k + k * n];
        if u == 0.0 {
            return Ok(f32::NEG_INFINITY);
        }
        if u < 0.0 {
            sign = -sign;
        }
        ld += u.abs().ln();
        if ipiv[k] - 1 != i32::try_from(k).unwrap_or(i32::MAX) {
            sign = -sign;
        }
    }
    if sign > 0.0 {
        Ok(ld)
    } else {
        Ok(f32::NEG_INFINITY)
    }
}

/// `fast_logdet(a)` (sklearn `utils/extmath.py:93-130`) dispatched to LAPACK
/// `getrf` for f64/f32 (numpy-`slogdet`-identical), with the hand-rolled
/// partial-pivot LU [`slogdet_lu_parts`] as the exotic-`F` fallback.
///
/// For f64/f32 the sign comes from LAPACK `getrf` — the same factorization
/// numpy's `slogdet` uses — so a rank-deficient PCA precision yields the SAME
/// finite-or-`−inf` outcome as sklearn (#2110): the partial-pivot LU and the
/// symmetric-eigensolver inertia both disagree with numpy on the determinant
/// sign of these cond-`~1e16` matrices. Exotic `F` (no LAPACK binding) keep the
/// partial-pivot LU, which is correct for well-conditioned determinants.
fn fast_logdet_dispatch<F: Float + Send + Sync + 'static>(a: &Array2<F>) -> Result<F, FerroError> {
    // SAFETY: TypeId is checked at runtime; the transmutes only reinterpret
    // identical types (f64↔F when F == f64, etc.), as in `svd_dispatch`.
    if TypeId::of::<F>() == TypeId::of::<f64>() {
        let a_f64: &Array2<f64> = unsafe { &*(std::ptr::from_ref(a).cast::<Array2<f64>>()) };
        let ld = fast_logdet_getrf_f64(a_f64)?;
        Ok(F::from(ld).ok_or_else(|| FerroError::NumericalInstability {
            message: "fast_logdet: f64 result not representable in F".into(),
        })?)
    } else if TypeId::of::<F>() == TypeId::of::<f32>() {
        let a_f32: &Array2<f32> = unsafe { &*(std::ptr::from_ref(a).cast::<Array2<f32>>()) };
        let ld = fast_logdet_getrf_f32(a_f32)?;
        Ok(F::from(ld).ok_or_else(|| FerroError::NumericalInstability {
            message: "fast_logdet: f32 result not representable in F".into(),
        })?)
    } else {
        let (sign, ld) = slogdet_lu_parts(a)?;
        if sign > F::zero() {
            Ok(ld)
        } else {
            Ok(F::neg_infinity())
        }
    }
}

// ---------------------------------------------------------------------------
// Randomized SVD (sklearn `randomized_svd`, `utils/extmath.py:361-557`)
// ---------------------------------------------------------------------------

/// Bridge an `ndarray::Array2<f64>` (row-major logical order) into a
/// `ferray-core` `Array<f64, Ix2>` for the linalg/random helpers.
fn ndarray_to_ferray_f64(a: &Array2<f64>) -> Result<ferray::Array<f64, ferray::Ix2>, FerroError> {
    let (m, n) = a.dim();
    let data: Vec<f64> = a.iter().copied().collect();
    ferray::Array::<f64, ferray::Ix2>::from_vec(ferray::Ix2::new([m, n]), data).map_err(|e| {
        FerroError::NumericalInstability {
            message: format!("ferray array construction failed: {e}"),
        }
    })
}

/// Economic ("reduced") QR factorisation of `a` via `ferray::linalg::qr`
/// (faer), returning only `Q` (shape `(m, min(m, n))`).
///
/// This is sklearn's `power_iteration_normalizer` and final orthonormaliser in
/// `randomized_range_finder` (`sklearn/utils/extmath.py:325`,`:337-342`).
/// sklearn's non-Array-API path uses `scipy.linalg.qr(mode='economic')`; faer's
/// thin QR is the same factorisation up to column-sign gauge. The randomised
/// SVD's `n_iter` power iterations make the recovered subspace insensitive to
/// the normaliser choice / QR sign gauge: verified that this faer-QR path
/// reproduces sklearn's `'auto'` (LU) singular values to ~5e-15 on the #2115
/// fixture (well inside the rtol-1e-6 pin), so no LAPACK `geqrf`/`orgqr` QR is
/// required.
fn qr_economic_q_f64(a: &Array2<f64>) -> Result<Array2<f64>, FerroError> {
    let fa = ndarray_to_ferray_f64(a)?;
    let (q, _r) = ferray::linalg::qr(&fa, ferray::linalg::QrMode::Reduced).map_err(|e| {
        FerroError::NumericalInstability {
            message: format!("ferray qr (faer, economic) failed: {e}"),
        }
    })?;
    Ok(q.into_ndarray())
}

/// Compute an orthonormal matrix whose range approximates the range of `a`,
/// mirroring sklearn `randomized_range_finder`
/// (`sklearn/utils/extmath.py:217-344`).
///
/// `a` is `(m, n)`; the returned `Q` is `(m, size)` with orthonormal columns.
///
/// 1. Draw `Q0 = random_state.normal(size=(n, size))` (`extmath.py:284`). The
///    seed feeds `ferray::random::RandomState::new(seed)`, whose
///    `standard_normal_2d` is bit-identical to
///    `numpy.random.RandomState(seed).standard_normal(size=(n, size))` (C-order),
///    and numpy's `RandomState.normal(0, 1, size)` equals `standard_normal(size)`
///    value-wise — so the projection drawn here is exactly sklearn's.
/// 2. `n_iter` power iterations, each `Q = qr(a @ Q)` then `Q = qr(aᵀ @ Q)`
///    (`extmath.py:336-338`). sklearn's `'auto'` normaliser at `n_iter >= 3`
///    (non-Array-API) is LU; faer-QR is used here as the (sklearn-supported)
///    `'QR'` normaliser and matches LU to ~2e-15 on the #2115 fixture.
/// 3. A final `Q = qr(a @ Q)` orthonormalises (`extmath.py:342`).
///
/// The matmuls use `ferray::linalg::gemm` (OpenBLAS, numpy-`@`-identical).
fn randomized_range_finder_f64(
    a: &Array2<f64>,
    size: usize,
    n_iter: usize,
    seed: u64,
) -> Result<Array2<f64>, FerroError> {
    let (_m, n) = a.dim();

    // Q0 = random_state.normal(size=(A.shape[1], size))  (`extmath.py:284`).
    let mut rng = ferray::random::RandomState::new(seed);
    let q0 = rng
        .standard_normal_2d((n, size))
        .map_err(|e| FerroError::NumericalInstability {
            message: format!("ferray standard_normal_2d draw failed: {e}"),
        })?;
    let mut q: Array2<f64> = q0.into_ndarray();

    let at = a.t().to_owned(); // (n, m)

    // Power iterations: Q = QR(A @ Q); Q = QR(A.T @ Q)  (`extmath.py:336-338`).
    for _ in 0..n_iter {
        let aq = ferray_gemm_f64(a, &q)?; // (m, size)
        q = qr_economic_q_f64(&aq)?; // (m, size)
        let atq = ferray_gemm_f64(&at, &q)?; // (n, size)
        q = qr_economic_q_f64(&atq)?; // (n, size)
    }

    // Final: Q = QR(A @ Q)  (`extmath.py:342`), shape (m, size).
    let aq = ferray_gemm_f64(a, &q)?;
    qr_economic_q_f64(&aq)
}

/// Compute a truncated randomized SVD of `m` (centred data), mirroring sklearn
/// `randomized_svd(M, n_components, n_oversamples=10, n_iter='auto',
/// power_iteration_normalizer='auto', flip_sign=False, random_state)` as called
/// by `_fit_truncated` (`sklearn/decomposition/_pca.py:762-772`,
/// `sklearn/utils/extmath.py:361-557`).
///
/// Returns `(s, vt)`: `s` are the top-`n_components` singular values
/// (non-increasing) and `vt` the corresponding right singular vectors as rows
/// (`(n_components, n_features)`). `U` is not returned — `_fit_truncated` only
/// needs `Vt` for `components_` and `S` for `explained_variance_`, and the
/// `svd_flip(U, Vt, u_based_decision=False)` sign step is applied row-wise by
/// the caller exactly as for the `full` path.
///
/// Algorithm (`extmath.py:505-557`):
/// - `n_random = n_components + n_oversamples`.
/// - `n_iter`: `'auto'` → `7 if n_components < 0.1*min(M.shape) else 4`
///   (`extmath.py:512`).
/// - `transpose='auto'` → `transpose = n_samples < n_features` (`:514-515`); if
///   transposing, operate on `Mᵀ` and swap/transpose `U`/`Vt` back (`:516-518`,
///   `:553-555`).
/// - `Q = randomized_range_finder(M, n_random, n_iter, seed)` (`:520-526`).
/// - `B = Qᵀ @ M` (`:529`); `Uhat, s, Vt = svd(B, full_matrices=False)` via
///   `ferray::linalg::svd_lapack` (LAPACK `gesdd`, sklearn's `gesdd` driver,
///   `:539-541`); `U = Q @ Uhat` (`:543`). `flip_sign=False` here (`_pca.py:770`)
///   so no flip inside.
/// - Return `s[:n_components]`, `Vt[:n_components, :]` (transpose-back when
///   transposed, `:553-555`).
fn randomized_svd_f64(
    m: &Array2<f64>,
    n_components: usize,
    n_oversamples: usize,
    n_iter_spec: Option<usize>,
    seed: u64,
) -> Result<(Array1<f64>, Array2<f64>), FerroError> {
    let (n_samples, n_features) = m.dim();
    let n_random = n_components + n_oversamples;

    // n_iter (`'auto'` → 7 if n_components < 0.1*min(shape) else 4, `:512`).
    let min_shape = n_samples.min(n_features);
    let n_iter = n_iter_spec.unwrap_or_else(|| {
        // 7 if n_components < 0.1 * min(M.shape) else 4. Compare without floats:
        // n_components < 0.1*min  <=>  10*n_components < min.
        if 10 * n_components < min_shape { 7 } else { 4 }
    });

    // transpose='auto' → n_samples < n_features (`:514-515`).
    let transpose = n_samples < n_features;
    let work = if transpose {
        m.t().to_owned()
    } else {
        m.to_owned()
    };

    // Q = randomized_range_finder(work, n_random, n_iter, seed)  (`:520-526`).
    let q = randomized_range_finder_f64(&work, n_random, n_iter, seed)?;

    // B = Q.T @ work  (`:529`), shape (n_random, work.ncols()).
    let qt = q.t().to_owned();
    let b = ferray_gemm_f64(&qt, &work)?;

    // Uhat, s, Vt = svd(B, full_matrices=False)  (`:539-541`, gesdd). The
    // ferray_svd_lapack_f64 helper returns (s, Vt); recover Uhat = (B Vt^T) / s
    // is NOT needed because U = Q @ Uhat is only used for the non-transposed Vt
    // sign / the transposed return — but we DO need Uhat to form U. Compute the
    // thin SVD with U via a direct ferray svd_lapack call returning U too.
    let (uhat, s_full, vt_full) = svd_lapack_full_f64(&b)?;

    // U = Q @ Uhat  (`:543`), shape (work.nrows(), n_random).
    let u_full = ferray_gemm_f64(&q, &uhat)?;

    // flip_sign=False (`_pca.py:770`) — no internal svd_flip.
    if transpose {
        // Return Vt[:n_components,:].T, s[:n_components], U[:,:n_components].T
        // (`:553-555`). The PCA caller wants (s, Vt_pca) where Vt_pca rows are
        // right singular vectors of the ORIGINAL M: that is U[:,:n_components].T.
        let mut s = Array1::<f64>::zeros(n_components);
        let mut vt = Array2::<f64>::zeros((n_components, n_features));
        for k in 0..n_components {
            s[k] = s_full[k];
            // Vt_pca[k, :] = U[:, k]  (U has work.nrows() == n_features rows).
            for j in 0..n_features {
                vt[[k, j]] = u_full[[j, k]];
            }
        }
        Ok((s, vt))
    } else {
        // Return U[:,:n_components], s[:n_components], Vt[:n_components,:]
        // (`:556-557`). PCA wants (s, Vt[:n_components,:]).
        let mut s = Array1::<f64>::zeros(n_components);
        let mut vt = Array2::<f64>::zeros((n_components, n_features));
        for k in 0..n_components {
            s[k] = s_full[k];
            for j in 0..n_features {
                vt[[k, j]] = vt_full[[k, j]];
            }
        }
        Ok((s, vt))
    }
}

/// Full thin SVD of `a` (`m × n`) returning `(U, s, Vt)` via
/// `ferray::linalg::svd_lapack` (LAPACK `gesdd`). `U` is `(m, k)`, `s` length
/// `k`, `Vt` is `(k, n)` with `k = min(m, n)`. Used by [`randomized_svd_f64`]
/// for the `B = Qᵀ M` decomposition (sklearn `extmath.py:539-541`), where `U`
/// is needed to form `U = Q @ Uhat` (`:543`).
#[allow(
    clippy::type_complexity,
    reason = "(U, s, Vt) is the standard thin-SVD triple, not worth a named struct"
)]
fn svd_lapack_full_f64(
    a: &Array2<f64>,
) -> Result<(Array2<f64>, Array1<f64>, Array2<f64>), FerroError> {
    let fa = ndarray_to_ferray_f64(a)?;
    let (u, s, vt) =
        ferray::linalg::svd_lapack(&fa, false).map_err(|e| FerroError::NumericalInstability {
            message: format!("ferray svd_lapack (gesdd) failed: {e}"),
        })?;
    Ok((u.into_ndarray(), s.into_ndarray(), vt.into_ndarray()))
}

/// f32 analogue of [`ndarray_to_ferray_f64`].
fn ndarray_to_ferray_f32(a: &Array2<f32>) -> Result<ferray::Array<f32, ferray::Ix2>, FerroError> {
    let (m, n) = a.dim();
    let data: Vec<f32> = a.iter().copied().collect();
    ferray::Array::<f32, ferray::Ix2>::from_vec(ferray::Ix2::new([m, n]), data).map_err(|e| {
        FerroError::NumericalInstability {
            message: format!("ferray array construction failed: {e}"),
        }
    })
}

/// f32 analogue of [`qr_economic_q_f64`].
fn qr_economic_q_f32(a: &Array2<f32>) -> Result<Array2<f32>, FerroError> {
    let fa = ndarray_to_ferray_f32(a)?;
    let (q, _r) = ferray::linalg::qr(&fa, ferray::linalg::QrMode::Reduced).map_err(|e| {
        FerroError::NumericalInstability {
            message: format!("ferray qr (faer, economic) failed: {e}"),
        }
    })?;
    Ok(q.into_ndarray())
}

/// f32 analogue of [`svd_lapack_full_f64`].
#[allow(
    clippy::type_complexity,
    reason = "(U, s, Vt) is the standard thin-SVD triple, not worth a named struct"
)]
fn svd_lapack_full_f32(
    a: &Array2<f32>,
) -> Result<(Array2<f32>, Array1<f32>, Array2<f32>), FerroError> {
    let fa = ndarray_to_ferray_f32(a)?;
    let (u, s, vt) =
        ferray::linalg::svd_lapack(&fa, false).map_err(|e| FerroError::NumericalInstability {
            message: format!("ferray svd_lapack (gesdd) failed: {e}"),
        })?;
    Ok((u.into_ndarray(), s.into_ndarray(), vt.into_ndarray()))
}

/// f32 analogue of [`randomized_range_finder_f64`]. The Gaussian draw uses the
/// f64 `RandomState` (numpy-bit-identical) then narrows to f32, mirroring
/// sklearn's `Q = astype(random_state.normal(...), float32)`
/// (`sklearn/utils/extmath.py:284-287`).
fn randomized_range_finder_f32(
    a: &Array2<f32>,
    size: usize,
    n_iter: usize,
    seed: u64,
) -> Result<Array2<f32>, FerroError> {
    let (_m, n) = a.dim();
    let mut rng = ferray::random::RandomState::new(seed);
    let q0 = rng
        .standard_normal_2d((n, size))
        .map_err(|e| FerroError::NumericalInstability {
            message: format!("ferray standard_normal_2d draw failed: {e}"),
        })?;
    let q0_nd: Array2<f64> = q0.into_ndarray();
    let mut q: Array2<f32> = q0_nd.mapv(|v| v as f32);

    let at = a.t().to_owned();
    for _ in 0..n_iter {
        let aq = ferray_gemm_f32(a, &q)?;
        q = qr_economic_q_f32(&aq)?;
        let atq = ferray_gemm_f32(&at, &q)?;
        q = qr_economic_q_f32(&atq)?;
    }
    let aq = ferray_gemm_f32(a, &q)?;
    qr_economic_q_f32(&aq)
}

/// f32 analogue of [`randomized_svd_f64`].
fn randomized_svd_f32(
    m: &Array2<f32>,
    n_components: usize,
    n_oversamples: usize,
    n_iter_spec: Option<usize>,
    seed: u64,
) -> Result<(Array1<f32>, Array2<f32>), FerroError> {
    let (n_samples, n_features) = m.dim();
    let n_random = n_components + n_oversamples;
    let min_shape = n_samples.min(n_features);
    let n_iter = n_iter_spec.unwrap_or_else(|| if 10 * n_components < min_shape { 7 } else { 4 });
    let transpose = n_samples < n_features;
    let work = if transpose {
        m.t().to_owned()
    } else {
        m.to_owned()
    };
    let q = randomized_range_finder_f32(&work, n_random, n_iter, seed)?;
    let qt = q.t().to_owned();
    let b = ferray_gemm_f32(&qt, &work)?;
    let (uhat, s_full, vt_full) = svd_lapack_full_f32(&b)?;
    let u_full = ferray_gemm_f32(&q, &uhat)?;
    if transpose {
        let mut s = Array1::<f32>::zeros(n_components);
        let mut vt = Array2::<f32>::zeros((n_components, n_features));
        for k in 0..n_components {
            s[k] = s_full[k];
            for j in 0..n_features {
                vt[[k, j]] = u_full[[j, k]];
            }
        }
        Ok((s, vt))
    } else {
        let mut s = Array1::<f32>::zeros(n_components);
        let mut vt = Array2::<f32>::zeros((n_components, n_features));
        for k in 0..n_components {
            s[k] = s_full[k];
            for j in 0..n_features {
                vt[[k, j]] = vt_full[[k, j]];
            }
        }
        Ok((s, vt))
    }
}

/// Dispatch the randomized truncated SVD to the f64/f32 ferray-backed
/// implementation, returning `(s, vt)` (top-`n_components` singular values and
/// right singular vectors as rows). Mirrors sklearn `randomized_svd`
/// (`sklearn/utils/extmath.py:361-557`) as invoked by `_fit_truncated`
/// (`sklearn/decomposition/_pca.py:764-772`).
///
/// Only f64/f32 are supported (the RNG/QR/SVD/gemm primitives are LAPACK/BLAS-
/// backed for those types); an exotic `F` yields
/// [`FerroError::NumericalInstability`] (the randomized branch is only reached
/// from `'auto'` solver-selection, and ferrolearn fits f64/f32 in practice).
fn randomized_svd_dispatch<F: Float + Send + Sync + 'static>(
    m: &Array2<F>,
    n_components: usize,
    n_oversamples: usize,
    n_iter_spec: Option<usize>,
    seed: u64,
) -> Result<(Array1<F>, Array2<F>), FerroError> {
    // SAFETY: TypeId is checked at runtime; the transmutes only reinterpret
    // between identical types (Array<f64> <-> Array<F> when F == f64, etc.),
    // exactly as in `svd_dispatch`/`gemm_dispatch`.
    if TypeId::of::<F>() == TypeId::of::<f64>() {
        let m_f64: &Array2<f64> = unsafe { &*(std::ptr::from_ref(m).cast::<Array2<f64>>()) };
        let (s, vt) = randomized_svd_f64(m_f64, n_components, n_oversamples, n_iter_spec, seed)?;
        let s_f: Array1<F> = unsafe { std::mem::transmute_copy::<Array1<f64>, Array1<F>>(&s) };
        let vt_f: Array2<F> = unsafe { std::mem::transmute_copy::<Array2<f64>, Array2<F>>(&vt) };
        std::mem::forget(s);
        std::mem::forget(vt);
        Ok((s_f, vt_f))
    } else if TypeId::of::<F>() == TypeId::of::<f32>() {
        let m_f32: &Array2<f32> = unsafe { &*(std::ptr::from_ref(m).cast::<Array2<f32>>()) };
        let (s, vt) = randomized_svd_f32(m_f32, n_components, n_oversamples, n_iter_spec, seed)?;
        let s_f: Array1<F> = unsafe { std::mem::transmute_copy::<Array1<f32>, Array1<F>>(&s) };
        let vt_f: Array2<F> = unsafe { std::mem::transmute_copy::<Array2<f32>, Array2<F>>(&vt) };
        std::mem::forget(s);
        std::mem::forget(vt);
        Ok((s_f, vt_f))
    } else {
        Err(FerroError::NumericalInstability {
            message: "randomized PCA solver supports only f32/f64".into(),
        })
    }
}

// ---------------------------------------------------------------------------
// Trait implementations
// ---------------------------------------------------------------------------

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, ()> for PCA<F> {
    type Fitted = FittedPCA<F>;
    type Error = FerroError;

    /// Fit PCA by centring the data and eigendecomposing the covariance matrix.
    ///
    /// # Errors
    ///
    /// - [`FerroError::InvalidParameter`] if `n_components` is an explicit count
    ///   that is zero or exceeds the number of features, or a variance ratio
    ///   outside `(0, 1]`.
    /// - [`FerroError::InsufficientSamples`] if there are fewer than 2 samples.
    /// - [`FerroError::ConvergenceFailure`] if the Jacobi eigendecomposition
    ///   does not converge.
    fn fit(&self, x: &Array2<F>, _y: &()) -> Result<FittedPCA<F>, FerroError> {
        let (n_samples, n_features) = x.dim();

        // Validate the `n_components` SPEC eagerly. The explicit-count checks are
        // unchanged from the historical `usize` path; the ratio range check is
        // new (`Ratio`/`Auto` resolution to an integer count happens AFTER the
        // eigendecomposition, mirroring sklearn `_pca.py:657-688`).
        match self.n_components {
            NComponents::Count(0) => {
                return Err(FerroError::InvalidParameter {
                    name: "n_components".into(),
                    reason: "must be at least 1".into(),
                });
            }
            NComponents::Count(k) if k > n_features => {
                return Err(FerroError::InvalidParameter {
                    name: "n_components".into(),
                    reason: format!("n_components ({k}) exceeds n_features ({n_features})"),
                });
            }
            NComponents::Ratio(r) if !(r > F::zero() && r <= F::one()) => {
                return Err(FerroError::InvalidParameter {
                    name: "n_components".into(),
                    reason: "variance ratio must be in (0, 1]".into(),
                });
            }
            _ => {}
        }

        if n_samples < 2 {
            return Err(FerroError::InsufficientSamples {
                required: 2,
                actual: n_samples,
                context: "PCA::fit requires at least 2 samples".into(),
            });
        }

        // Finiteness: sklearn `PCA.fit` runs `_validate_data` with the default
        // `force_all_finite=True` (`_pca.py:511`), raising
        // `ValueError("Input X contains NaN."/"...infinity...")`
        // (`utils/validation.py:147-154`) BEFORE any SVD/eigendecomposition.
        // This fires before the `gesdd`/faer call, so the clean finiteness error
        // replaces the incidental `NumericalInstability` the SVD would otherwise
        // raise on non-finite input (R-DEV-2). Decomposition has no
        // missing-value support → NaN AND infinity both rejected (#2288).
        reject_non_finite(x)?;

        let n_f = F::from(n_samples).unwrap_or_else(F::one);

        // Step 1: compute mean and centre data.
        let mut mean = Array1::<F>::zeros(n_features);
        for j in 0..n_features {
            let col = x.column(j);
            let sum = col.iter().copied().fold(F::zero(), |a, b| a + b);
            mean[j] = sum / n_f;
        }

        let mut x_centered = x.to_owned();
        for mut row in x_centered.rows_mut() {
            for (v, &m) in row.iter_mut().zip(mean.iter()) {
                *v = *v - m;
            }
        }

        let n_minus_1 = FittedPCA::<F>::const_f((n_samples - 1) as f64)?;
        let min_dim = n_samples.min(n_features);
        let max_jacobi_iter = n_features * n_features * 100 + 1000;

        // Resolve `n_components` (None → min(X.shape)) BEFORE solver selection,
        // mirroring sklearn `_fit` (`_pca.py:523-529`): `Auto`/`Ratio` resolve to
        // `min_dim` for the purpose of the `'auto'` solver-selection comparison.
        // (`n_components_for_selection` matters only for the `randomized` branch,
        // which ferrolearn folds into `full`; kept for fidelity to sklearn.)
        let _n_components_for_selection: usize = match self.n_components {
            NComponents::Count(k) => k,
            NComponents::Ratio(_) | NComponents::Auto => min_dim,
        };

        // `'auto'` solver selection for DENSE X (ferrolearn has no sparse PCA),
        // mirroring sklearn `_fit` (`_pca.py:531-543`):
        //   - X.shape[1] <= 1000 and X.shape[0] >= 10*X.shape[1] → covariance_eigh
        //   - max(X.shape) <= 500                                → full
        //   - 1 <= n_components < 0.8*min(X.shape)               → randomized
        //   - else                                               → full
        // The first matching branch wins. `randomized` is now implemented
        // (REQ-12) via `randomized_svd_dispatch`; only `arpack` remains unmapped.
        let use_covariance_eigh = n_features <= 1000 && n_samples >= 10 * n_features;
        let max_shape = n_samples.max(n_features);
        // sklearn's `1 <= n_components < 0.8 * min(X.shape)` (`_pca.py:539`). The
        // comparison uses the resolved-for-selection count (`Auto`/`Ratio` →
        // `min_dim`, never `< 0.8*min_dim`, so those select `full`); avoid float
        // arithmetic: `n_c < 0.8*min_dim  <=>  5*n_c < 4*min_dim`.
        let use_randomized = !use_covariance_eigh
            && max_shape > 500
            && _n_components_for_selection >= 1
            && 5 * _n_components_for_selection < 4 * min_dim;

        // For the `randomized` solver `n_components_` is fixed (no post-SVD
        // float/`None` resolution — `_fit_truncated` requires an explicit int,
        // `_pca.py:718-722`), so it equals the selection count.
        let randomized_n_components = _n_components_for_selection;

        // The full SVD path (sklearn `_fit_full` `full` branch, `_pca.py:575-591`)
        // and the covariance_eigh path (`_pca.py:593-643`) both reconstruct the
        // SAME `(S, Vt)`: `S` descending singular values of `X_centered`, `Vt`
        // the right singular vectors as component rows. The `randomized` path
        // (`_fit_truncated`, `_pca.py:762-801`) reconstructs only the TOP
        // `n_components` directions — handled in its own block below.
        if use_randomized {
            return Self::fit_randomized(
                x,
                &x_centered,
                mean,
                randomized_n_components,
                n_samples,
                n_features,
                min_dim,
                n_minus_1,
                self.n_oversamples(),
                self.iterated_power_spec(),
                self.random_state.unwrap_or(0),
                self.whiten,
            );
        }

        let (s_all, vt_all): (Array1<F>, Array2<F>) = if use_covariance_eigh {
            // covariance_eigh: eigendecompose C = X_cᵀX_c/(n−1) (faer for
            // f64/f32, Jacobi otherwise). sklearn clips tiny negative
            // eigenvalues to 0 (`_pca.py:637`), so the tall-skinny
            // `noise_variance_` is 0 — matching sklearn's covariance_eigh.
            let xt = x_centered.t();
            let mut cov = xt.dot(&x_centered);
            cov.mapv_inplace(|v| v / n_minus_1);
            let (eigenvalues, eigenvectors) = eigen_dispatch(&cov, max_jacobi_iter)?;
            let mut indices: Vec<usize> = (0..n_features).collect();
            indices.sort_by(|&a, &b| {
                eigenvalues[b]
                    .partial_cmp(&eigenvalues[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let mut s = Array1::<F>::zeros(min_dim);
            let mut vt = Array2::<F>::zeros((min_dim, n_features));
            for (k, &idx) in indices.iter().take(min_dim).enumerate() {
                let ev = eigenvalues[idx];
                let ev_clamped = if ev < F::zero() { F::zero() } else { ev };
                // S = sqrt(eigval·(n−1)) (sklearn `_pca.py:642`).
                s[k] = (ev_clamped * n_minus_1).sqrt();
                for j in 0..n_features {
                    vt[[k, j]] = eigenvectors[[j, idx]];
                }
            }
            (s, vt)
        } else {
            // full: U, S, Vt = svd(X_centered, full_matrices=False)
            // (sklearn `_pca.py:588`). faer for f64/f32, Gram-eigen fallback
            // otherwise. `S` is descending; `vt` rows are right singular vectors.
            svd_dispatch(&x_centered, max_jacobi_iter)?
        };

        // explained_variance_ = S²/(n−1) for ALL min_dim directions
        // (sklearn `_pca.py:591`). With the FULL SVD this captures the
        // rank-deficient tail (S[-1]²/(n−1)) that covariance_eigh clips to 0 —
        // THIS is the #2114/#2110 fix.
        let mut explained_variance_all = Array1::<F>::zeros(min_dim);
        for k in 0..min_dim {
            explained_variance_all[k] = (s_all[k] * s_all[k]) / n_minus_1;
        }
        // total_var = sum(explained_variance_) (sklearn `_pca.py:652`).
        let total_variance = explained_variance_all
            .iter()
            .copied()
            .fold(F::zero(), |a, b| a + b);

        // Resolve the requested `n_components` SPEC into an integer count
        // (`n_components_`), mirroring sklearn's post-SVD postprocess
        // (`_pca.py:657-681`): the full SVD is computed FIRST, then `None`/float
        // `n_components` is resolved against the FULL `explained_variance_ratio_`.
        let n_comp = match self.n_components {
            NComponents::Count(k) => k,
            // Variance ratio: `n_components_ = searchsorted(ratio_cumsum, r,
            // side="right") + 1` (`_pca.py:680-681`). `searchsorted(..,r,"right")`
            // counts entries `<= r` (insertion point after equal entries), so
            // `n_components_ = 1 + count(ratio_cumsum <= r)`. The cumsum is over
            // the FULL descending `explained_variance_ratio_`.
            NComponents::Ratio(r) => {
                let mut cum = F::zero();
                let mut count_le = 0usize;
                for k in 0..min_dim {
                    let ratio = if total_variance > F::zero() {
                        explained_variance_all[k] / total_variance
                    } else {
                        F::zero()
                    };
                    cum = cum + ratio;
                    if cum <= r {
                        count_le += 1;
                    }
                }
                (count_le + 1).min(min_dim)
            }
            // Auto (sklearn `n_components=None`): keep all `min(n_samples,
            // n_features)` components (`_pca.py:523-527`,`:685`).
            NComponents::Auto => min_dim,
        };

        // noise_variance_ from the DISCARDED tail (probabilistic PCA, sklearn
        // `_pca.py:685-688`): if `n_comp < min_dim`, the MEAN of
        // `explained_variance_[n_comp:]` over the FULL spectrum; else 0. This is
        // computed from the FULL `explained_variance_all` BEFORE truncation, so
        // the full-SVD tail (rank-deficient S[-1]²/(n−1)) is included.
        let noise_variance = if n_comp < min_dim {
            let mut tail_sum = F::zero();
            for ev in explained_variance_all.iter().skip(n_comp) {
                tail_sum = tail_sum + *ev;
            }
            let count = FittedPCA::<F>::const_f((min_dim - n_comp) as f64)?;
            tail_sum / count
        } else {
            F::zero()
        };

        // Truncate components_/explained_variance_/ratio/singular_values_ to the
        // top n_comp (sklearn `_pca.py:698`), applying svd_flip per row.
        let mut components = Array2::<F>::zeros((n_comp, n_features));
        let mut explained_variance = Array1::<F>::zeros(n_comp);
        let mut explained_variance_ratio = Array1::<F>::zeros(n_comp);
        let mut singular_values = Array1::<F>::zeros(n_comp);

        for k in 0..n_comp {
            explained_variance[k] = explained_variance_all[k];
            explained_variance_ratio[k] = if total_variance > F::zero() {
                explained_variance_all[k] / total_variance
            } else {
                F::zero()
            };
            singular_values[k] = s_all[k];

            for j in 0..n_features {
                components[[k, j]] = vt_all[[k, j]];
            }

            // Sign convention: mirror sklearn `svd_flip(U, Vt,
            // u_based_decision=False)` (`_pca.py:647`, `extmath.py:897-905`).
            // For each component row, find the column with the maximum absolute
            // value (numpy `argmax` → FIRST on ties: iterate from 0 and update
            // only on STRICT `>`). If that entry is negative, negate the whole
            // row so its max-abs entry is positive. This pins the otherwise
            // arbitrary faer SVD / eigenvector signs deterministically.
            let mut j_max = 0usize;
            let mut max_abs = components[[k, 0]].abs();
            for j in 1..n_features {
                let abs_j = components[[k, j]].abs();
                if abs_j > max_abs {
                    max_abs = abs_j;
                    j_max = j;
                }
            }
            if components[[k, j_max]] < F::zero() {
                for j in 0..n_features {
                    components[[k, j]] = -components[[k, j]];
                }
            }
        }

        Ok(FittedPCA {
            components_: components,
            explained_variance_: explained_variance,
            explained_variance_ratio_: explained_variance_ratio,
            mean_: mean,
            singular_values_: singular_values,
            noise_variance_: noise_variance,
            whiten: self.whiten,
        })
    }
}

impl<F: Float + Send + Sync + 'static> PCA<F> {
    /// Fit PCA via the `randomized` truncated SVD solver, mirroring sklearn
    /// `_fit_truncated`'s `svd_solver == "randomized"` branch
    /// (`sklearn/decomposition/_pca.py:762-803`).
    ///
    /// Unlike the `full`/`covariance_eigh` paths, the randomized solver computes
    /// only the TOP `n_components` directions and derives `total_var` from the
    /// CENTRED data variance directly (`var(X, axis=0, ddof=1).sum()`,
    /// `_pca.py:789-792`) rather than from the full eigenvalue spectrum.
    ///
    /// Steps (`_pca.py:775-801`):
    /// - `U, S, Vt = randomized_svd(X_centered, n_components, n_oversamples=10,
    ///   n_iter='auto', power_iteration_normalizer='auto', flip_sign=False,
    ///   random_state)` (`:764-772`); then `svd_flip(U, Vt,
    ///   u_based_decision=False)` (`:773`) — here applied row-wise to `Vt`.
    /// - `explained_variance_ = S² / (n_samples − 1)` (`:780`).
    /// - `total_var = Σ var(X_centered, axis=0, ddof=1) = Σ X_centered² / (n−1)`
    ///   (`:789-792`); `explained_variance_ratio_ = explained_variance_ /
    ///   total_var` (`:794`); `singular_values_ = S` (`:795`).
    /// - `noise_variance_`: if `n_components < min(n_features, n_samples)`,
    ///   `(total_var − Σ explained_variance_) / (min(n_features, n_samples) −
    ///   n_components)` (`:797-799`); else `0.0` (`:800-801`). This is the
    ///   TRUNCATED noise-variance formula — it differs from the `full`-path mean
    ///   of discarded eigenvalues because the randomized solver never computes
    ///   the tail spectrum.
    #[allow(
        clippy::too_many_arguments,
        reason = "randomized fit threads sklearn's full state"
    )]
    fn fit_randomized(
        x: &Array2<F>,
        x_centered: &Array2<F>,
        mean: Array1<F>,
        n_components: usize,
        n_samples: usize,
        n_features: usize,
        min_dim: usize,
        n_minus_1: F,
        n_oversamples: usize,
        iterated_power: Option<usize>,
        seed: u64,
        whiten: bool,
    ) -> Result<FittedPCA<F>, FerroError> {
        let _ = x; // X is centred already; kept for signature parity with sklearn.

        // randomized_svd returns (S, Vt) for the top `n_components` directions
        // (`_pca.py:764-772`, flip_sign=False).
        let (s, mut components) = randomized_svd_dispatch(
            x_centered,
            n_components,
            n_oversamples,
            iterated_power,
            seed,
        )?;

        // svd_flip(U, Vt, u_based_decision=False) (`_pca.py:773`,
        // `extmath.py:897-905`): per component row, make the max-abs entry
        // positive. numpy `argmax` → FIRST on ties (strict `>`).
        for k in 0..n_components {
            let mut j_max = 0usize;
            let mut max_abs = components[[k, 0]].abs();
            for j in 1..n_features {
                let abs_j = components[[k, j]].abs();
                if abs_j > max_abs {
                    max_abs = abs_j;
                    j_max = j;
                }
            }
            if components[[k, j_max]] < F::zero() {
                for j in 0..n_features {
                    components[[k, j]] = -components[[k, j]];
                }
            }
        }

        // explained_variance_ = S² / (n_samples − 1) (`_pca.py:780`).
        let mut explained_variance = Array1::<F>::zeros(n_components);
        for k in 0..n_components {
            explained_variance[k] = (s[k] * s[k]) / n_minus_1;
        }

        // total_var = Σ var(X_centered, axis=0, ddof=1) = Σ X_centered² / (n−1)
        // (`_pca.py:789-792`: `X_centered **= 2; total_var = sum(X_centered)/N`
        // with `N = X.shape[0] - 1`). Sum the squared centred entries.
        let mut sum_sq = F::zero();
        for &v in x_centered.iter() {
            sum_sq = sum_sq + v * v;
        }
        let total_var = sum_sq / n_minus_1;

        // explained_variance_ratio_ = explained_variance_ / total_var (`:794`).
        let mut explained_variance_ratio = Array1::<F>::zeros(n_components);
        for k in 0..n_components {
            explained_variance_ratio[k] = if total_var > F::zero() {
                explained_variance[k] / total_var
            } else {
                F::zero()
            };
        }

        // singular_values_ = S (`:795`).
        let singular_values = s.clone();

        // noise_variance_ (`_pca.py:797-801`): the TRUNCATED formula.
        let noise_variance = if n_components < n_features.min(n_samples) {
            let exp_var_sum = explained_variance
                .iter()
                .copied()
                .fold(F::zero(), |a, b| a + b);
            let denom = FittedPCA::<F>::const_f((n_features.min(n_samples) - n_components) as f64)?;
            (total_var - exp_var_sum) / denom
        } else {
            F::zero()
        };

        let _ = min_dim; // kept for signature parity; randomized truncates to n_components.

        Ok(FittedPCA {
            components_: components,
            explained_variance_: explained_variance,
            explained_variance_ratio_: explained_variance_ratio,
            mean_: mean,
            singular_values_: singular_values,
            noise_variance_: noise_variance,
            whiten,
        })
    }
}

impl<F: Float + Send + Sync + 'static> Transform<Array2<F>> for FittedPCA<F> {
    type Output = Array2<F>;
    type Error = FerroError;

    /// Project data onto the principal components: `(X - mean) @ components^T`.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of columns does not
    /// match the number of features seen during fitting.
    fn transform(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let n_features = self.mean_.len();
        if x.ncols() != n_features {
            return Err(FerroError::ShapeMismatch {
                expected: vec![x.nrows(), n_features],
                actual: vec![x.nrows(), x.ncols()],
                context: "FittedPCA::transform".into(),
            });
        }

        // Finiteness on the query X: sklearn `PCA.transform` runs
        // `_validate_data(..., reset=False)` (`_pca.py:824`), the default
        // `force_all_finite=True` raising a `ValueError` BEFORE the projection
        // (`utils/validation.py:147-154`). NaN AND infinity both rejected (#2289).
        reject_non_finite(x)?;

        // Centre the data.
        let mut x_centered = x.to_owned();
        for mut row in x_centered.rows_mut() {
            for (v, &m) in row.iter_mut().zip(self.mean_.iter()) {
                *v = *v - m;
            }
        }

        // Project: X_centered @ components^T
        let mut result = x_centered.dot(&self.components_.t());

        // Whitening: divide each column j by sqrt(explained_variance_[j]) so
        // the transformed output has unit component-wise variance. Mirrors
        // sklearn `_transform` (`sklearn/decomposition/_base.py:157-165`):
        // `scale = sqrt(explained_variance_); scale[scale < eps] = eps;
        // X_transformed /= scale`. The eps clip guards against components with
        // a variance arbitrarily close to zero on rank-deficient data
        // (`_base.py:158-164`). When `whiten` is false this block is skipped,
        // leaving the result byte-identical to the plain projection.
        if self.whiten {
            let min_scale = F::epsilon();
            for j in 0..result.ncols() {
                let mut scale = self.explained_variance_[j].sqrt();
                if scale < min_scale {
                    scale = min_scale;
                }
                for v in result.column_mut(j) {
                    *v = *v / scale;
                }
            }
        }

        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// Pipeline integration (generic)
// ---------------------------------------------------------------------------

impl<F: Float + Send + Sync + 'static> PipelineTransformer<F> for PCA<F> {
    /// Fit PCA using the pipeline interface.
    ///
    /// The `y` argument is ignored; PCA is unsupervised.
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

impl<F: Float + Send + Sync + 'static> FittedPipelineTransformer<F> for FittedPCA<F> {
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

    #[test]
    fn test_pca_dimensionality_reduction() {
        let pca = PCA::<f64>::new(1);
        let x = array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
        ];
        let fitted = pca.fit(&x, &()).unwrap();
        let projected = fitted.transform(&x).unwrap();
        assert_eq!(projected.dim(), (4, 1));
    }

    #[test]
    fn test_pca_explained_variance_ratio_sums_le_1() {
        let pca = PCA::<f64>::new(2);
        let x = array![
            [2.5, 2.4],
            [0.5, 0.7],
            [2.2, 2.9],
            [1.9, 2.2],
            [3.1, 3.0],
            [2.3, 2.7],
            [2.0, 1.6],
            [1.0, 1.1],
            [1.5, 1.6],
            [1.1, 0.9],
        ];
        let fitted = pca.fit(&x, &()).unwrap();
        let ratio_sum: f64 = fitted.explained_variance_ratio().iter().sum();
        // When n_components == n_features, ratio should sum to ~1.0.
        assert!(ratio_sum <= 1.0 + 1e-10, "ratio sum = {ratio_sum}");
    }

    #[test]
    fn test_pca_explained_variance_ratio_partial() {
        let pca = PCA::<f64>::new(1);
        let x = array![
            [2.5, 2.4],
            [0.5, 0.7],
            [2.2, 2.9],
            [1.9, 2.2],
            [3.1, 3.0],
            [2.3, 2.7],
            [2.0, 1.6],
            [1.0, 1.1],
            [1.5, 1.6],
            [1.1, 0.9],
        ];
        let fitted = pca.fit(&x, &()).unwrap();
        let ratio_sum: f64 = fitted.explained_variance_ratio().iter().sum();
        // With 1 component out of 2, ratio should be strictly less than 1.
        assert!(ratio_sum <= 1.0 + 1e-10);
        assert!(ratio_sum > 0.0);
    }

    #[test]
    fn test_pca_components_orthonormal() {
        let pca = PCA::<f64>::new(2);
        let x = array![
            [2.5, 2.4],
            [0.5, 0.7],
            [2.2, 2.9],
            [1.9, 2.2],
            [3.1, 3.0],
            [2.3, 2.7],
            [2.0, 1.6],
            [1.0, 1.1],
            [1.5, 1.6],
            [1.1, 0.9],
        ];
        let fitted = pca.fit(&x, &()).unwrap();
        let c = fitted.components();

        // Check that each component is unit length.
        for i in 0..c.nrows() {
            let norm: f64 = c.row(i).iter().map(|v| v * v).sum::<f64>().sqrt();
            assert_abs_diff_eq!(norm, 1.0, epsilon = 1e-8);
        }

        // Check mutual orthogonality.
        for i in 0..c.nrows() {
            for j in (i + 1)..c.nrows() {
                let dot: f64 = c
                    .row(i)
                    .iter()
                    .zip(c.row(j).iter())
                    .map(|(a, b)| a * b)
                    .sum();
                assert_abs_diff_eq!(dot, 0.0, epsilon = 1e-8);
            }
        }
    }

    #[test]
    fn test_pca_inverse_transform_roundtrip() {
        let pca = PCA::<f64>::new(2);
        let x = array![[2.5, 2.4], [0.5, 0.7], [2.2, 2.9], [1.9, 2.2], [3.1, 3.0],];
        let fitted = pca.fit(&x, &()).unwrap();
        let projected = fitted.transform(&x).unwrap();
        let recovered = fitted.inverse_transform(&projected).unwrap();

        // With n_components == n_features, reconstruction should be exact.
        for (a, b) in x.iter().zip(recovered.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-8);
        }
    }

    #[test]
    fn test_pca_inverse_transform_approx() {
        // With fewer components, reconstruction is lossy but the error
        // should be bounded.
        let pca = PCA::<f64>::new(1);
        let x = array![[2.5, 2.4], [0.5, 0.7], [2.2, 2.9], [1.9, 2.2], [3.1, 3.0],];
        let fitted = pca.fit(&x, &()).unwrap();
        let projected = fitted.transform(&x).unwrap();
        let recovered = fitted.inverse_transform(&projected).unwrap();

        // Reconstruction should not be wildly off.
        let total_error: f64 = x
            .iter()
            .zip(recovered.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum();
        let total_var: f64 = {
            let mean_x: f64 = x.iter().sum::<f64>() / x.len() as f64;
            x.iter().map(|&v| (v - mean_x).powi(2)).sum()
        };
        // Relative reconstruction error should be reasonable.
        assert!(
            total_error < total_var,
            "error={total_error}, var={total_var}"
        );
    }

    #[test]
    fn test_pca_n_components_equals_n_features() {
        let pca = PCA::<f64>::new(3);
        let x = array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
        ];
        let fitted = pca.fit(&x, &()).unwrap();
        let ratio_sum: f64 = fitted.explained_variance_ratio().iter().sum();
        assert_abs_diff_eq!(ratio_sum, 1.0, epsilon = 1e-8);
    }

    #[test]
    fn test_pca_single_component() {
        let pca = PCA::<f64>::new(1);
        let x = array![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0],];
        let fitted = pca.fit(&x, &()).unwrap();
        assert_eq!(fitted.components().nrows(), 1);
        assert_eq!(fitted.explained_variance().len(), 1);
    }

    #[test]
    fn test_pca_shape_mismatch_transform() {
        let pca = PCA::<f64>::new(1);
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let fitted = pca.fit(&x, &()).unwrap();
        let x_bad = array![[1.0, 2.0, 3.0]];
        assert!(fitted.transform(&x_bad).is_err());
    }

    #[test]
    fn test_pca_shape_mismatch_inverse_transform() {
        let pca = PCA::<f64>::new(1);
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let fitted = pca.fit(&x, &()).unwrap();
        // inverse_transform expects 1 column (n_components), not 3
        let x_bad = array![[1.0, 2.0, 3.0]];
        assert!(fitted.inverse_transform(&x_bad).is_err());
    }

    #[test]
    fn test_pca_invalid_n_components_zero() {
        let pca = PCA::<f64>::new(0);
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        assert!(pca.fit(&x, &()).is_err());
    }

    #[test]
    fn test_pca_invalid_n_components_too_large() {
        let pca = PCA::<f64>::new(5);
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        assert!(pca.fit(&x, &()).is_err());
    }

    #[test]
    fn test_pca_insufficient_samples() {
        let pca = PCA::<f64>::new(1);
        let x = array![[1.0, 2.0]]; // only 1 sample
        assert!(pca.fit(&x, &()).is_err());
    }

    #[test]
    fn test_pca_explained_variance_positive() {
        let pca = PCA::<f64>::new(2);
        let x = array![[2.5, 2.4], [0.5, 0.7], [2.2, 2.9], [1.9, 2.2], [3.1, 3.0],];
        let fitted = pca.fit(&x, &()).unwrap();
        for &v in fitted.explained_variance() {
            assert!(v >= 0.0, "negative variance: {v}");
        }
    }

    #[test]
    fn test_pca_singular_values_positive() {
        let pca = PCA::<f64>::new(2);
        let x = array![[2.5, 2.4], [0.5, 0.7], [2.2, 2.9], [1.9, 2.2], [3.1, 3.0],];
        let fitted = pca.fit(&x, &()).unwrap();
        for &s in fitted.singular_values() {
            assert!(s >= 0.0, "negative singular value: {s}");
        }
    }

    #[test]
    fn test_pca_f32() {
        let pca = PCA::<f32>::new(1);
        let x: Array2<f32> = array![[1.0f32, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0],];
        let fitted = pca.fit(&x, &()).unwrap();
        let projected = fitted.transform(&x).unwrap();
        assert_eq!(projected.ncols(), 1);
    }

    #[test]
    fn test_pca_n_components_getter() {
        let pca = PCA::<f64>::new(3);
        assert_eq!(pca.n_components(), &NComponents::Count(3));
    }

    #[test]
    fn test_pca_pipeline_integration() {
        use ferrolearn_core::pipeline::{FittedPipelineEstimator, Pipeline, PipelineEstimator};
        use ferrolearn_core::traits::Predict;

        // Trivial estimator that sums each row.
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
            .transform_step("pca", Box::new(PCA::<f64>::new(1)))
            .estimator_step("sum", Box::new(SumEstimator));

        let x = array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
        ];
        let y = Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0]);

        let fitted = pipeline.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 4);
    }

    // ---- REQ-11: whiten (sklearn _base.py:157-165,:192-196) --------------
    //
    // Oracle = live scikit-learn 1.5.2 (R-CHAR-3), run from /tmp:
    //   X = [[1,2,3],[4,5,6],[7,8,10],[2,1,0],[5,3,2]]
    //   m = PCA(n_components=2, whiten=<...>).fit(X)
    //   m.transform(X) / m.inverse_transform(m.transform(X))
    // PCA component signs are deterministic per REQ-1's svd_flip, so the
    // element-wise comparison (including sign) is valid.

    fn whiten_fixture() -> Array2<f64> {
        array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 10.0],
            [2.0, 1.0, 0.0],
            [5.0, 3.0, 2.0],
        ]
    }

    #[test]
    fn pca_whiten_transform_matches_sklearn() -> Result<(), FerroError> {
        let x = whiten_fixture();
        let fitted = PCA::<f64>::new(2).with_whiten(true).fit(&x, &())?;

        // sklearn explained_variance_ sanity (oracle [26.42340146, 2.16534729]).
        let ev = fitted.explained_variance();
        assert_abs_diff_eq!(ev[0], 26.423_401_46, epsilon = 1e-6);
        assert_abs_diff_eq!(ev[1], 2.165_347_29, epsilon = 1e-6);

        let got = fitted.transform(&x)?;
        let expected = array![
            [-0.576_684_77, -1.310_790_08],
            [0.402_024_00, -0.444_643_84],
            [1.525_646_44, 0.083_497_24],
            [-1.039_932_95, 0.253_103_31],
            [-0.311_052_72, 1.418_833_38],
        ];
        for (a, b) in got.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-6);
        }
        Ok(())
    }

    #[test]
    fn pca_whiten_false_unchanged() -> Result<(), FerroError> {
        let x = whiten_fixture();

        // whiten=false must match the no-whiten sklearn oracle.
        let fitted = PCA::<f64>::new(2).with_whiten(false).fit(&x, &())?;
        let got = fitted.transform(&x)?;
        let expected = array![
            [-2.964_372_97, -1.928_843_21],
            [2.066_552_01, -0.654_298_71],
            [7.842_386_84, 0.122_867_18],
            [-5.345_639_88, 0.372_444_53],
            [-1.598_926_00, 2.087_830_21],
        ];
        for (a, b) in got.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-6);
        }

        // Regression guard: whiten=false MUST be byte-identical to the default
        // (no-whiten) transform — the whiten block is purely additive.
        let default_fitted = PCA::<f64>::new(2).fit(&x, &())?;
        let default_got = default_fitted.transform(&x)?;
        for (a, b) in got.iter().zip(default_got.iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
        Ok(())
    }

    #[test]
    fn pca_whiten_inverse_matches_sklearn() -> Result<(), FerroError> {
        let x = whiten_fixture();
        let fitted = PCA::<f64>::new(2).with_whiten(true).fit(&x, &())?;

        let transformed = fitted.transform(&x)?;
        let got = fitted.inverse_transform(&transformed)?;
        let expected = array![
            [0.965_922_29, 2.092_258_20, 2.951_174_68],
            [4.045_247_61, 4.877_501_66, 6.064_829_15],
            [6.986_571_28, 8.036_355_42, 9.980_759_81],
            [2.022_846_87, 0.938_146_92, 0.032_734_18],
            [4.979_411_95, 3.055_737_80, 1.970_502_18],
        ];
        for (a, b) in got.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-6);
        }
        Ok(())
    }

    // ---- REQ-14/15: noise_variance_ / get_covariance / get_precision /
    //      score_samples / score (probabilistic-PCA chain) -----------------
    //
    // Oracle = live scikit-learn 1.5.2 (R-CHAR-3), run from /tmp:
    //   X = [[1,2,3],[4,5,6],[7,8,10],[2,1,0],[5,3,2]]
    //   m = PCA(n_components=2).fit(X)   # whiten=False (default)
    //   m.noise_variance_ / m.get_covariance() / m.get_precision()
    //   m.score_samples(X) / m.score(X)
    // n_components=2 < min(n_samples=5, n_features=3)=3, so noise_variance_ is
    // the mean of the single discarded eigenvalue tail.

    fn prob_fixture() -> Array2<f64> {
        array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 10.0],
            [2.0, 1.0, 0.0],
            [5.0, 3.0, 2.0],
        ]
    }

    #[test]
    fn pca_noise_variance_matches_sklearn() -> Result<(), FerroError> {
        let x = prob_fixture();
        let fitted = PCA::<f64>::new(2).fit(&x, &())?;
        // sklearn m.noise_variance_ = 0.011251254758681639
        assert_abs_diff_eq!(
            fitted.noise_variance(),
            0.011_251_254_758_681_639,
            epsilon = 1e-6
        );
        Ok(())
    }

    #[test]
    fn pca_n_components_and_n_features_in_match_sklearn() -> Result<(), FerroError> {
        // sklearn PCA(n_components=2).fit(X 5x3): n_components_ == 2, n_features_in_ == 3.
        let x = prob_fixture();
        let fitted = PCA::<f64>::new(2).fit(&x, &())?;
        assert_eq!(fitted.n_components_(), 2);
        assert_eq!(fitted.n_features_in_(), 3);
        Ok(())
    }

    #[test]
    fn pca_get_covariance_matches_sklearn() -> Result<(), FerroError> {
        let x = prob_fixture();
        let fitted = PCA::<f64>::new(2).fit(&x, &())?;
        let cov = fitted.get_covariance();
        // sklearn m.get_covariance()
        let expected = array![[5.7, 5.7, 6.8], [5.7, 7.7, 10.55], [6.8, 10.55, 15.2],];
        assert_eq!(cov.dim(), (3, 3));
        for (a, b) in cov.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-6);
        }
        Ok(())
    }

    #[test]
    fn pca_get_precision_matches_sklearn() -> Result<(), FerroError> {
        let x = prob_fixture();
        let fitted = PCA::<f64>::new(2).fit(&x, &())?;
        let precision = fitted.get_precision()?;
        // sklearn m.get_precision()
        let expected = array![
            [8.912_621_36, -23.145_631_07, 12.077_669_9],
            [-23.145_631_07, 62.757_281_55, -33.203_883_5],
            [12.077_669_9, -33.203_883_5, 17.708_737_86],
        ];
        assert_eq!(precision.dim(), (3, 3));
        for (a, b) in precision.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-6);
        }
        Ok(())
    }

    #[test]
    fn pca_score_samples_matches_sklearn() -> Result<(), FerroError> {
        let x = prob_fixture();
        let fitted = PCA::<f64>::new(2).fit(&x, &())?;
        let ll = fitted.score_samples(&x)?;
        // sklearn m.score_samples(X)
        let expected = array![
            -4.097_758_23,
            -3.660_865_03,
            -3.787_078_62,
            -3.350_185_42,
            -3.787_078_62,
        ];
        assert_eq!(ll.len(), 5);
        for (a, b) in ll.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-6);
        }
        Ok(())
    }

    #[test]
    fn pca_score_matches_sklearn() -> Result<(), FerroError> {
        let x = prob_fixture();
        let fitted = PCA::<f64>::new(2).fit(&x, &())?;
        // sklearn m.score(X) = -3.736593186111911
        assert_abs_diff_eq!(fitted.score(&x)?, -3.736_593_186_111_911, epsilon = 1e-6);
        Ok(())
    }

    // 6x3 fixture for the whiten=true covariance/score path. Oracle = live
    // scikit-learn 1.5.2 (R-CHAR-3), run from /tmp:
    //   import numpy as np; from sklearn.decomposition import PCA
    //   X = np.array([[1,2,3],[4,5,7],[2,0,1],[8,6,5],[3,3,2],[0,1,4]],float)
    //   m = PCA(n_components=2, whiten=True).fit(X)
    //   np.diag(m.get_covariance())  -> [99.92727433, 72.73104573, 41.32698497]
    //   float(m.score(X))            -> -6.577981760662929
    fn whiten_cov_fixture() -> Array2<f64> {
        array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 7.0],
            [2.0, 0.0, 1.0],
            [8.0, 6.0, 5.0],
            [3.0, 3.0, 2.0],
            [0.0, 1.0, 4.0],
        ]
    }

    #[test]
    fn pca_whiten_get_covariance_and_score_match_sklearn() -> Result<(), FerroError> {
        let x = whiten_cov_fixture();
        let fitted = PCA::<f64>::new(2).with_whiten(true).fit(&x, &())?;

        // sklearn np.diag(m.get_covariance()) under whiten=True. The whitening
        // rescale (`components_ * sqrt(exp_var)`, sklearn `_base.py:46-47`)
        // multiplies each component outer product by exp_var[k].
        let cov = fitted.get_covariance();
        let expected_diag = [99.927_274_33, 72.731_045_73, 41.326_984_97];
        assert_eq!(cov.dim(), (3, 3));
        for (i, e) in expected_diag.iter().enumerate() {
            assert_abs_diff_eq!(cov[[i, i]], e, epsilon = 1e-6);
        }
        // Covariance is symmetric.
        for i in 0..3 {
            for j in 0..3 {
                assert_abs_diff_eq!(cov[[i, j]], cov[[j, i]], epsilon = 1e-9);
            }
        }

        // sklearn m.score(X) under whiten=True (precision/score now work).
        assert_abs_diff_eq!(fitted.score(&x)?, -6.577_981_760_662_929, epsilon = 1e-6);
        Ok(())
    }

    // REQ-14: the Woodbury-lemma precision matches sklearn's `get_precision`
    // (which sklearn ALSO computes via the lemma, `_base.py:85-101`) on a fixture
    // where `noise_variance_ > 0` exercises the FULL general branch (the small
    // `P`-matrix inverse + the `1/noise_variance_` diagonal terms), not just the
    // `noise_variance_ == 0` corner case. Oracle = live scikit-learn 1.5.2
    // (R-CHAR-3), run from /tmp:
    //   import numpy as np; from sklearn.decomposition import PCA
    //   X = np.array([[1,2,3],[3,5,4],[5,4,6],[7,8,9],[4,3,2]], float)  # prob_fixture
    //   m = PCA(n_components=2, whiten=W).fit(X)   # n_comp 2 < min_dim 3 → nv>0
    //   m.noise_variance_ == 0.011251254758681639
    //   whiten=False: m.get_precision() ==  (below)
    //   whiten=True : m.get_precision() ==  (below)
    //   m.score(X)  matches the existing pca_score_matches_sklearn oracle.
    // This is the SAME fixture as the full-rank precision/score tests, here run
    // for BOTH whiten settings to lock the lemma's whiten `sqrt(exp_var)` rescale
    // (`_base.py:88-89`) against the oracle. (The wide n_samples<n_features pin
    // #2110 additionally needs the SVD-based `noise_variance_` of sklearn's
    // `full` solver — REQ-12 / blocker #1503 — which covariance_eigh cannot
    // reproduce; see the report on #2113.)
    //
    // sklearn m.get_precision() for prob_fixture, whiten=False (n_comp=2):
    fn prob_precision_whiten_false() -> Array2<f64> {
        array![
            [8.912_621_359_223_3, -23.145_631_067_961, 12.077_669_902_912],
            [-23.145_631_067_961, 62.757_281_553_398, -33.203_883_495_146],
            [12.077_669_902_912, -33.203_883_495_146, 17.708_737_864_078],
        ]
    }

    #[test]
    fn pca_get_precision_general_branch_matches_sklearn() -> Result<(), FerroError> {
        // whiten=false: the lemma (n_comp=2 < n_features=3, noise_variance_>0)
        // must match sklearn's lemma precision element-wise (≤1e-6).
        let x = prob_fixture();
        let fitted = PCA::<f64>::new(2).fit(&x, &())?;
        let precision = fitted.get_precision()?;
        let expected = prob_precision_whiten_false();
        assert_eq!(precision.dim(), (3, 3));
        for (a, b) in precision.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-6);
        }

        // Precision is symmetric and finite (the lemma never inverts the full,
        // possibly ill-conditioned covariance — only the small P-matrix).
        assert!(precision.iter().all(|v| v.is_finite()));
        for i in 0..3 {
            for j in 0..3 {
                assert_abs_diff_eq!(precision[[i, j]], precision[[j, i]], epsilon = 1e-9);
            }
        }

        // whiten=true: the lemma's `sqrt(exp_var)` rescale path
        // (sklearn `_base.py:88-89`). Oracle:
        //   PCA(n_components=2, whiten=True).fit(prob_fixture).get_precision()
        //   == [[ 8.721190415086, -23.165479211667,  12.173775248485],
        //       [-23.165479211667, 62.745898613098, -33.211539158895],
        //       [ 12.173775248485, -33.211539158895, 17.627195177324]]
        let fitted_w = PCA::<f64>::new(2).with_whiten(true).fit(&x, &())?;
        let prec_w = fitted_w.get_precision()?;
        assert!(prec_w.iter().all(|v| v.is_finite()));
        for i in 0..3 {
            for j in 0..3 {
                assert_abs_diff_eq!(prec_w[[i, j]], prec_w[[j, i]], epsilon = 1e-9);
            }
        }
        let expected_w = array![
            [8.721_190_415_086, -23.165_479_211_667, 12.173_775_248_485],
            [-23.165_479_211_667, 62.745_898_613_098, -33.211_539_158_895],
            [12.173_775_248_485, -33.211_539_158_895, 17.627_195_177_324],
        ];
        for (a, b) in prec_w.iter().zip(expected_w.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-6);
        }
        Ok(())
    }

    // REQ-14/15: a rank-deficient (n_samples < n_features) fit must NOT raise the
    // old `ValueError("eigenvalue <= 0")` from get_precision/score — the Woodbury
    // lemma never inverts the singular covariance directly, so it returns FINITE
    // values. (The exact value-parity for this fixture additionally needs the
    // SVD-based noise_variance_ of sklearn's `full` solver — REQ-12, blocker
    // #1503 — covariance_eigh sets noise_variance_ == 0 here whereas sklearn's
    // full SVD yields ~1.34e-31; see #2113. This test asserts only the
    // no-raise / finiteness contract that the lemma rewrite DOES deliver.)
    #[test]
    fn pca_get_precision_rank_deficient_does_not_raise() -> Result<(), FerroError> {
        let x = array![
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [2.0, 1.0, 0.0, 3.0, 2.0],
            [5.0, 4.0, 2.0, 1.0, 0.0],
            [0.0, 3.0, 1.0, 2.0, 4.0],
        ];
        for whiten in [true, false] {
            let fitted = PCA::<f64>::new(3).with_whiten(whiten).fit(&x, &())?;
            // The previous eigendecompose-invert hit the `eigenvalue <= 0` guard
            // and returned Err; the lemma returns Ok with no NaN.
            let precision = fitted.get_precision()?;
            assert_eq!(precision.dim(), (5, 5));
            assert!(
                precision.iter().all(|v| !v.is_nan()),
                "whiten={whiten}: lemma precision must not contain NaN"
            );
            // score_samples/score also no longer raise (they may be ±inf when the
            // lemma precision is singular, mirroring sklearn's fast_logdet).
            let _ = fitted.score_samples(&x)?;
            let _ = fitted.score(&x)?;
        }
        Ok(())
    }

    // #2110: rank-deficient (n_samples < n_features) score FINITENESS / SIGN
    // parity. `get_precision` now routes through `ferray::linalg::inv_lapack`
    // (LAPACK getri = scipy.linalg.inv) + `ferray::linalg::gemm` (openblas) and
    // the score's `fast_logdet(precision)` SIGN is taken from the SYMMETRIC
    // precision's inertia (negative-eigenvalue parity via the faer eigensolver),
    // which is numpy-faithful where the LU partial-pivot sign flips. The result:
    // for the whiten=True fit the precision determinant is POSITIVE (an even
    // number of tiny negative eigenvalues), so `fast_logdet` returns the finite
    // log-magnitude and `score`/`score_samples` are FINITE — matching sklearn's
    // finite behavior, where the previous Gauss-Jordan + nested-loop matmul +
    // LU-sign path yielded `−inf`.
    //
    // Oracle = live scikit-learn 1.5.2 (R-CHAR-3), run from /tmp:
    //   import numpy as np; from sklearn.decomposition import PCA
    //   X = np.array([[1,2,3,4,5],[2,1,0,3,2],[5,4,2,1,0],[0,3,1,2,4]], float)
    //   PCA(n_components=3, whiten=True).fit(X):
    //     score(X) == -3522835255394185.5  (FINITE)
    //     slogdet(get_precision()) == (sign=+1, ld=246.67217674983294)
    //
    // NOTE on EXACT VALUE parity (the pin's element-wise `rtol=1e-6`): the score
    // quadratic `Xr·precision·Xrᵀ` is catastrophically ILL-CONDITIONED here
    // (cond ~6.7e15). The off-diagonal of the inner lemma matrix
    // `P = comps @ comps.T / noise_variance_` is pure rounding NOISE on a product
    // of (numerically) orthogonal component rows (~2e-16) divided by
    // noise_variance_ ~1e-31, amplifying ULP-level accumulation differences to
    // ~1e15. `ferray::linalg::gemm` requires both operands row-major and always
    // calls OpenBLAS no-transpose, whereas numpy `comps @ comps.T` passes a
    // transpose-FLAGGED view → OpenBLAS uses a different kernel with a different
    // summation order → a different rounding residual on that orthogonal-row
    // product. So the precision differs from numpy by 1–2 ULPs, which (given the
    // cond ~1e16 quadratic) is O(1) relative error in the per-sample score. EXACT
    // score VALUE parity therefore needs a numpy-`@`-identical transpose-aware
    // gemm in ferray — a real ferray substrate gap (R-SUBSTRATE-5), escalated.
    // This test pins the FINITENESS / SIGN contract the wiring DOES deliver.
    #[test]
    fn pca_score_rank_deficient_whiten_finite_matches_sklearn() -> Result<(), FerroError> {
        let x = array![
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [2.0, 1.0, 0.0, 3.0, 2.0],
            [5.0, 4.0, 2.0, 1.0, 0.0],
            [0.0, 3.0, 1.0, 2.0, 4.0],
        ];

        // whiten=True: sklearn yields a FINITE score (precision slogdet sign +1).
        // ferrolearn must now also be FINITE (was −inf before the getrf slogdet).
        let fitted = PCA::<f64>::new(3).with_whiten(true).fit(&x, &())?;
        let ss = fitted.score_samples(&x)?;
        assert_eq!(ss.len(), 4);
        for (i, v) in ss.iter().enumerate() {
            assert!(
                v.is_finite(),
                "whiten=True score_samples[{i}] must be finite like sklearn (was {v})"
            );
        }
        let score = fitted.score(&x)?;
        assert!(
            score.is_finite(),
            "whiten=True score must be finite like sklearn's -3.52e15 (was {score})"
        );

        // The precision's fast_logdet must be the finite log-magnitude (sign +1),
        // NOT −inf. sklearn slogdet(precision) == (+1, 246.672...).
        let (_prec, log_det) = fitted.precision_and_logdet()?;
        assert!(
            log_det.is_finite() && log_det > 0.0,
            "whiten=True fast_logdet(precision) must be finite +ve (sklearn 246.67), was {log_det}"
        );

        Ok(())
    }

    // ---- REQ-13a: n_components as variance ratio / auto (None) -----------
    //
    // Oracle = live scikit-learn 1.5.2 (R-CHAR-3), run from /tmp:
    //   X = [[1,2,3],[4,5,6],[7,8,10],[2,1,0],[5,3,2],[6,7,5]]  (6x3)
    //   PCA(n_components=3).fit(X).explained_variance_ratio_
    //     = [0.898229, 0.088879, 0.012892]
    //   cumsum                = [0.898229, 0.987108, 1.0]
    //   PCA(n_components=0.95 ).fit(X).n_components_ == 2
    //   PCA(n_components=0.5  ).fit(X).n_components_ == 1
    //   PCA(n_components=0.999).fit(X).n_components_ == 3
    //   PCA(n_components=None ).fit(X).n_components_ == 3  (= min(6,3))
    // sklearn `_pca.py:659-681` (float cumsum) / `:685` (None=auto).

    fn ratio_fixture() -> Array2<f64> {
        array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 10.0],
            [2.0, 1.0, 0.0],
            [5.0, 3.0, 2.0],
            [6.0, 7.0, 5.0],
        ]
    }

    #[test]
    fn pca_n_components_ratio_095_selects_2() -> Result<(), FerroError> {
        // cumsum [0.898229, 0.987108, 1.0]: searchsorted(.,0.95,"right")+1 = 1+1 = 2.
        let x = ratio_fixture();
        let fitted = PCA::<f64>::with_variance_ratio(0.95).fit(&x, &())?;
        assert_eq!(fitted.n_components_(), 2);
        Ok(())
    }

    #[test]
    fn pca_n_components_ratio_05_selects_1() -> Result<(), FerroError> {
        // 0.5 < 0.898229 (first cumsum entry): selects 1 component.
        let x = ratio_fixture();
        let fitted = PCA::<f64>::with_variance_ratio(0.5).fit(&x, &())?;
        assert_eq!(fitted.n_components_(), 1);
        Ok(())
    }

    #[test]
    fn pca_n_components_ratio_0999_selects_3() -> Result<(), FerroError> {
        // 0.999 > 0.987108 (second cumsum entry): selects all 3 components.
        let x = ratio_fixture();
        let fitted = PCA::<f64>::with_variance_ratio(0.999).fit(&x, &())?;
        assert_eq!(fitted.n_components_(), 3);
        Ok(())
    }

    #[test]
    fn pca_n_components_auto_selects_all() -> Result<(), FerroError> {
        // sklearn n_components=None -> min(n_samples=6, n_features=3) = 3.
        let x = ratio_fixture();
        let fitted = PCA::<f64>::auto().fit(&x, &())?;
        assert_eq!(fitted.n_components_(), 3);
        Ok(())
    }

    // ---- REQ-12: 'full' SVD solver — noise_variance_ on a rank-deficient
    //      (n_samples < n_features) fit + small-fixture value parity ----------
    //
    // Oracle = live scikit-learn 1.5.2 (R-CHAR-3), run from /tmp:
    //   import numpy as np; from sklearn.decomposition import PCA
    //   XW = np.array([[1,2,3,4,5],[2,1,0,3,2],[5,4,2,1,0],[0,3,1,2,4]], float)
    //   m = PCA(n_components=3).fit(XW)        # 'auto' → 'full' (max(shape)=5<=500)
    //   m._fit_svd_solver == 'full'
    //   m.noise_variance_ == 1.3441289858383478e-31
    //   # full spectrum (n_components=4): the discarded tail S[-1]²/(n-1)
    //   PCA(n_components=4).fit(XW).explained_variance_
    //     == [10.518684809961808, 2.4996912680006216, 1.564957255370908,
    //         1.3441289858383478e-31]
    // sklearn's 'full' SVD captures the tiny rank-deficient tail singular value
    // (the covariance_eigh path would clip it to 0). This is the #2114/#2110 fix.

    #[test]
    fn pca_full_solver_noise_variance_rank_deficient_matches_sklearn() -> Result<(), FerroError> {
        // 4×5 wide fixture: n_samples=4 < n_features=5, so the fit is
        // rank-deficient (rank <= 3). 'auto' selects 'full' (max(4,5)=5 <= 500),
        // NOT covariance_eigh (4 >= 10*5 is false). n_components=3 < min_dim=4.
        let x = array![
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [2.0, 1.0, 0.0, 3.0, 2.0],
            [5.0, 4.0, 2.0, 1.0, 0.0],
            [0.0, 3.0, 1.0, 2.0, 4.0],
        ];
        let fitted = PCA::<f64>::new(3).fit(&x, &())?;

        // The leading explained_variance_ is large (O(1..10)); sklearn:
        // [10.518684809961808, 2.4996912680006216, 1.564957255370908].
        let ev = fitted.explained_variance();
        assert_abs_diff_eq!(ev[0], 10.518_684_809_961_808, epsilon = 1e-6);
        assert_abs_diff_eq!(ev[1], 2.499_691_268_000_621_6, epsilon = 1e-6);
        assert_abs_diff_eq!(ev[2], 1.564_957_255_370_908, epsilon = 1e-6);

        // noise_variance_ = mean(explained_variance_[3:4]) = S[-1]²/(n-1).
        // sklearn's full SVD (scipy.linalg.svd → LAPACK gesdd) yields the tiny
        // rank-deficient tail value 1.3441289858383478e-31 — which the
        // covariance_eigh path clips to exactly 0. ferrolearn now routes the
        // full SVD through `ferray::linalg::svd_lapack` (the SAME LAPACK gesdd
        // driver, proven bit-identical to scipy on this noise-floor value,
        // ferray #2116), so `noise_variance_` is bit-identical to sklearn. We
        // pin the EXACT 1e-31 value to a tight relative tolerance (rtol 1e-12);
        // the prior faer thin-SVD floor (~3.75e-32) would fail this. This is the
        // crux of #2110: the precision (~1/noise_variance_ ~1e30) amplifies any
        // delta in this value.
        let nv = fitted.noise_variance();
        let sk_nv = 1.344_128_985_838_347_8e-31;
        assert!(
            (nv - sk_nv).abs() <= 1e-12 * sk_nv,
            "noise_variance_ ({nv:e}) must be bit-identical to sklearn's LAPACK \
             gesdd value ({sk_nv:e}); the old faer thin-SVD floor (~3.75e-32) diverged"
        );
        Ok(())
    }

    #[test]
    fn pca_full_solver_matches_sklearn_small() -> Result<(), FerroError> {
        // 6×3 small fixture: 'auto' → 'full' (max(6,3)=6 <= 500, and
        // 6 >= 10*3 is false so NOT covariance_eigh). This routes the existing
        // ratio_fixture through the NEW full-SVD path; it must STILL match the
        // live sklearn 1.5.2 oracle to 1e-6.
        //   m = PCA(n_components=3).fit(ratio_fixture)
        //   m.components_ / m.explained_variance_ratio_ / m.singular_values_
        let x = ratio_fixture();
        let fitted = PCA::<f64>::new(3).fit(&x, &())?;

        // sklearn m.components_ (rows already svd_flip'd to max-abs-positive).
        let sk_components = array![
            [
                0.417_874_090_823_892_04,
                0.574_352_566_831_245_1,
                0.703_917_873_897_563
            ],
            [
                0.751_983_293_234_584_8,
                0.216_118_802_350_880_02,
                -0.622_746_971_061_674
            ],
            [
                -0.509_806_209_175_774_9,
                0.789_564_305_325_906_5,
                -0.341_593_086_641_174_5
            ],
        ];
        let c = fitted.components();
        for (a, b) in c.iter().zip(sk_components.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-6);
        }

        // sklearn m.explained_variance_ratio_.
        let sk_ratio = [
            0.898_229_234_558_290_1,
            0.088_879_156_459_763_48,
            0.012_891_608_981_946_438,
        ];
        let evr = fitted.explained_variance_ratio();
        for (k, e) in sk_ratio.iter().enumerate() {
            assert_abs_diff_eq!(evr[k], e, epsilon = 1e-6);
        }

        // sklearn m.singular_values_.
        let sk_sv = [
            10.701_599_291_983_511,
            3.366_317_342_233_177_4,
            1.282_060_897_616_868_6,
        ];
        let sv = fitted.singular_values();
        for (k, e) in sk_sv.iter().enumerate() {
            assert_abs_diff_eq!(sv[k], e, epsilon = 1e-6);
        }
        Ok(())
    }

    #[test]
    fn pca_n_components_ratio_validation_rejects() {
        // Variance ratio must be in (0, 1]: 0.0 and 1.5 are out of range.
        let x = ratio_fixture();
        let too_low = PCA::<f64>::with_variance_ratio(0.0).fit(&x, &());
        assert!(matches!(too_low, Err(FerroError::InvalidParameter { .. })));
        let too_high = PCA::<f64>::with_variance_ratio(1.5).fit(&x, &());
        assert!(matches!(too_high, Err(FerroError::InvalidParameter { .. })));
    }

    /// REQ-12 (randomized solver). For a 600×100 fixture with `n_components=10`,
    /// sklearn's `'auto'` policy (`_pca.py:539-540`) selects the `randomized`
    /// truncated SVD (max(shape)=600>500, 600<10·100, 1≤10<0.8·100). ferrolearn's
    /// `fit` now routes the same shape to `fit_randomized` (`randomized_svd_dispatch`),
    /// so its `explained_variance_`/`singular_values_`/`noise_variance_` reproduce
    /// sklearn's randomized output.
    ///
    /// Fixture: `np.random.RandomState(0).randn(600, 100)`, reproduced
    /// bit-identically by `ferray::random::RandomState::new(0).standard_normal_2d`
    /// (numpy `randn`/`standard_normal` are the same draw). Seed `random_state=42`.
    ///
    /// R-CHAR-3: the expected spectrum is the LIVE sklearn 1.5.2 `'auto'`
    /// (randomized) oracle, computed out of band and pasted as symbolic
    /// constants — NOT copied from ferrolearn:
    /// ```text
    /// import numpy as np; from sklearn.decomposition import PCA
    /// X = np.random.RandomState(0).randn(600, 100)
    /// m = PCA(n_components=10, svd_solver="auto", random_state=42).fit(X)
    /// assert m._fit_svd_solver == "randomized"
    /// m.explained_variance_, m.singular_values_, m.noise_variance_
    /// ```
    /// Tracking: #2115.
    ///
    /// # Errors
    /// Propagates fixture-construction / fit errors (none in practice).
    #[test]
    fn pca_randomized_solver_matches_sklearn() -> Result<(), FerroError> {
        // Build the 600×100 fixture via the numpy-bit-identical RNG (the SAME
        // draw as `RandomState(0).randn(600, 100)`).
        let mut rng = ferray::random::RandomState::new(0);
        let x_ferray =
            rng.standard_normal_2d((600, 100))
                .map_err(|e| FerroError::NumericalInstability {
                    message: format!("fixture draw failed: {e}"),
                })?;
        let x: Array2<f64> = x_ferray.into_ndarray();

        // Confirm the 'auto' policy selects randomized for this shape, exactly
        // as sklearn (`_pca.py:531-543`): NOT covariance_eigh (600 < 10·100),
        // max(shape)=600 > 500, 1 ≤ 10 < 0.8·100 = 80.
        let (n_samples, n_features) = x.dim();
        assert!(!(n_features <= 1000 && n_samples >= 10 * n_features));
        assert!(n_samples.max(n_features) > 500);
        assert!(5 * 10 < 4 * n_samples.min(n_features));

        let fitted = PCA::<f64>::new(10)
            .with_random_state(Some(42))
            .fit(&x, &())?;

        // Live sklearn 1.5.2 'auto' (randomized) oracle, random_state=42.
        let ev_oracle = [
            1.886714085271514,
            1.8329606542457184,
            1.7869096438307568,
            1.737047891621905,
            1.711124787035177,
            1.6727067608796928,
            1.6165848631431592,
            1.5820038314712825,
            1.5730661617712987,
            1.5272202268451576,
        ];
        let sv_oracle = [
            33.61758077372072,
            33.1352294679422,
            32.71633959743393,
            32.25665337696273,
            32.015055012198104,
            31.653615113710725,
            31.118070843526795,
            30.783441897411315,
            30.696361851219567,
            30.245742111580753,
        ];
        let nv_oracle = 0.9166336967101316_f64;

        let ev = fitted.explained_variance();
        let sv = fitted.singular_values();
        assert_eq!(ev.len(), 10);
        for k in 0..10 {
            assert_abs_diff_eq!(ev[k], ev_oracle[k], epsilon = 1e-6);
            assert_abs_diff_eq!(sv[k], sv_oracle[k], epsilon = 1e-5);
        }
        assert_abs_diff_eq!(fitted.noise_variance(), nv_oracle, epsilon = 1e-6);

        // components_ rows are unit-norm and follow svd_flip (max-abs entry
        // positive), matching the full path's deterministic sign convention.
        let comps = fitted.components();
        assert_eq!(comps.dim(), (10, 100));
        for k in 0..10 {
            let norm: f64 = comps.row(k).iter().map(|&v| v * v).sum::<f64>().sqrt();
            assert_abs_diff_eq!(norm, 1.0, epsilon = 1e-6);
        }
        Ok(())
    }
}
