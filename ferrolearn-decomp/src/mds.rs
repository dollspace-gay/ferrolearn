//! Multidimensional Scaling (MDS) — SMACOF.
//!
//! Embeds data into a low-dimensional space such that pairwise distances are
//! preserved as well as possible, using scikit-learn's algorithm: **SMACOF**
//! (Scaling by MAjorizing a COmplicated Function), an iterative stress
//! majorization via the Guttman transform (`sklearn/manifold/_mds.py`).
//!
//! # Algorithm
//!
//! 1. Start from an initial embedding `X` — either a fixed [`MDS::with_init`]
//!    configuration (deterministic, parity with sklearn `smacof(init=X0,
//!    n_init=1)`) or a random uniform draw (sklearn `random_state.uniform`; the
//!    Rust RNG is NOT numpy-parity — documented carve-out, REQ-1).
//! 2. Each iteration: compute the current pairwise distances `dis =
//!    euclidean_distances(X)`, the raw stress `((dis - D)²).sum()/2`
//!    (`_mds.py:147`), and the Guttman transform `X ← (1/n)·B·X` with
//!    `B[i,j] = -D[i,j]/dis[i,j]`, `B[i,i] += Σ_j B-ratio` (`_mds.py:151-155`).
//! 3. Stop when the relative stress change `< eps` (`_mds.py:157-165`) or after
//!    `max_iter` iterations.
//! 4. With the default random init, run `n_init` independent restarts and keep
//!    the lowest-stress run (`smacof`, `_mds.py:348-387`). A fixed `init` forces
//!    `n_init = 1` (`_mds.py:339-346`).
//!
//! The classical (PCoA) eigendecomposition is retained as an internal helper
//! ([`classical_mds`]) consumed by `isomap.rs` (Isomap is classical MDS on
//! geodesic distances) — it is NOT sklearn's `MDS` algorithm.
//!
//! # Examples
//!
//! ```
//! use ferrolearn_decomp::{MDS, Dissimilarity};
//! use ferrolearn_core::traits::Fit;
//! use ndarray::array;
//!
//! let init = array![[0.1, 0.2], [0.3, -0.1], [-0.2, 0.4], [0.5, 0.05]];
//! let mds = MDS::new(2).with_init(init);
//! let x = array![
//!     [0.0, 0.0],
//!     [1.0, 0.0],
//!     [0.0, 1.0],
//!     [1.0, 1.0],
//! ];
//! let fitted = mds.fit(&x, &()).unwrap();
//! assert_eq!(fitted.embedding().ncols(), 2);
//! ```
//!
//! The standalone [`smacof`] helper exposes the sklearn function shape directly
//! for callers that already have a precomputed dissimilarity matrix:
//!
//! ```
//! use ferrolearn_decomp::smacof;
//! use ndarray::array;
//!
//! let d = array![
//!     [0.0, 1.0, 1.0],
//!     [1.0, 0.0, 2.0],
//!     [1.0, 2.0, 0.0],
//! ];
//! let init = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
//! let (embedding, _stress, n_iter) =
//!     smacof(&d, 2, Some(&init), 1, 300, 1e-3, false, None).unwrap();
//! assert_eq!(embedding.dim(), (3, 2));
//! assert!(n_iter > 0);
//! ```
//!
//! ## REQ status
//!
//! Translation target: scikit-learn 1.5.2 `class MDS` + `smacof`
//! (`sklearn/manifold/_mds.py`). Tracking: #1451 / #2406. Each REQ is BINARY —
//! SHIPPED (impl + non-test consumer + tests + green verification) or
//! NOT-STARTED (with a concrete open blocker). The fit algorithm is now SMACOF
//! (matching sklearn). With a FIXED `init` the Guttman trajectory is
//! deterministic and matches sklearn element-wise (~1e-6). The DEFAULT random
//! init (`n_init` restarts) is a documented non-parity carve-out (Xoshiro ≠
//! numpy RandomState, REQ-1, same class as KMeans #1388).
//!
//! | REQ | Scope | Status | Evidence / Blocker |
//! |-----|-------|--------|--------------------|
//! | REQ-1 | Exact coordinate parity for the DEFAULT random init | NOT-STARTED | RNG CARVE-OUT (Xoshiro ≠ numpy RandomState `uniform`); parity tested via the fixed-init path instead — blocker #1452 |
//! | REQ-2 | Distance preservation (SMACOF descends the stress objective; full-rank fixed-init fit reconstructs the input distances) | SHIPPED | `smacof_single` Guttman loop drives stress→0; fixed-init parity tests reconstruct distances. Classical helper `classical_mds` consumed by `isomap.rs:38,339` |
//! | REQ-3 | Structural (embedding shape `(n_samples, n_components)`, deterministic for a fixed init) | SHIPPED | `fit`; shape + fixed-init determinism guards |
//! | REQ-4 | Kruskal stress-1 (`normalized_stress=True`) | SHIPPED | `smacof_single` `normalized_stress` arm `√(raw / (Σ disparities²/2))` (`_mds.py:148-149`); `kruskal_stress` helper retained |
//! | REQ-5 | Error/parameter contracts (n_components 0 / > n_samples, NON-FINITE rejection) | SHIPPED (scoped) | `fit` guards. NON-FINITE: `reject_non_finite` BEFORE the SMACOF math = sklearn `_validate_data(force_all_finite=True)` (`_mds.py:627`,`utils/validation.py:147-154`) |
//! | REQ-6 | SMACOF algorithm (Guttman majorization + `n_init` restarts + `random_state` + `init`) | SHIPPED | `smacof_single` (`_mds.py:22-167`) + public `smacof` (`_mds.py:187-392`, `n_init`/min-stress, `init` forces `n_init=1`); consumer: `MDS::fit` |
//! | REQ-7 | `metric=False` non-metric MDS (IsotonicRegression on disparities) | NOT-STARTED | sklearn `_mds.py:130-144` — blocker #1454 |
//! | REQ-8 | `normalized_stress` + sklearn `stress_` (raw SSR) definition + `max_iter`/`eps` | SHIPPED | `stress_` raw SSR/2 default (`_mds.py:147`), `with_normalized_stress` Kruskal-1 (`:148-149`), `with_max_iter`/`with_eps` (`:157-165`); consumer `MDS::fit` |
//! | REQ-9 | `dissimilarity_matrix_`/`n_iter_`/`embedding_` fitted attrs + `init` | SHIPPED | `FittedMDS::{dissimilarity_matrix, n_iter, embedding, stress}` accessors; `fit` sets all four (`_mds.py:636-654`) |
//! | REQ-10 | PyO3 binding | NOT-STARTED | no `ferrolearn-python` registration — blocker #1457 |
//! | REQ-11 | ferray substrate | NOT-STARTED | dense `Array2` only — blocker #1458 |

use ferrolearn_core::error::FerroError;
use ferrolearn_core::traits::Fit;
use ndarray::{Array1, Array2};
use rand::{Rng, SeedableRng};

/// Reject non-finite input the way sklearn's `_validate_data` does.
///
/// sklearn runs `check_array` with the default `force_all_finite=True` at the
/// top of `MDS.fit`/`fit_transform` (`sklearn/manifold/_mds.py:627`), raising
/// `ValueError("Input X contains NaN.")` / `"... contains infinity ..."`
/// (`sklearn/utils/validation.py:147-154`) BEFORE the dissimilarity / SMACOF
/// math. NaN AND infinity are both rejected. The message names "NaN" and
/// "infinity" to mirror sklearn's `ValueError`. Never panics (R-CODE-2).
fn reject_non_finite(x: &Array2<f64>) -> Result<(), FerroError> {
    if x.iter().any(|v| !v.is_finite()) {
        return Err(FerroError::InvalidParameter {
            name: "X".into(),
            reason: "Input X contains NaN or infinity.".into(),
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Dissimilarity type
// ---------------------------------------------------------------------------

/// How the input matrix should be interpreted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dissimilarity {
    /// The input is a feature matrix; pairwise Euclidean distances will be
    /// computed internally.
    Euclidean,
    /// The input is already a square pairwise-distance matrix.
    Precomputed,
}

// ---------------------------------------------------------------------------
// ClassicalMDS (unfitted)
// ---------------------------------------------------------------------------

/// Classical multidimensional scaling configuration.
///
/// This is the Rust public surface corresponding to
/// `sklearn.manifold.ClassicalMDS`: it computes pairwise dissimilarities
/// (or consumes a precomputed dissimilarity matrix), double-centres the squared
/// dissimilarities, eigendecomposes the centred matrix, applies sklearn's
/// deterministic `svd_flip` sign convention, and returns the closed-form
/// embedding.
#[derive(Debug, Clone)]
pub struct ClassicalMDS {
    /// Number of embedding dimensions.
    n_components: usize,
    /// Whether input is a feature matrix or a precomputed distance matrix.
    metric: Dissimilarity,
}

impl ClassicalMDS {
    /// Create a new `ClassicalMDS` that embeds into `n_components` dimensions.
    ///
    /// The default metric is Euclidean, matching sklearn's
    /// `ClassicalMDS(metric="euclidean")` default.
    #[must_use]
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            metric: Dissimilarity::Euclidean,
        }
    }

    /// Set the metric mode.
    ///
    /// `Dissimilarity::Euclidean` computes pairwise Euclidean distances from a
    /// feature matrix. `Dissimilarity::Precomputed` treats the input as an
    /// already-computed symmetric dissimilarity matrix.
    #[must_use]
    pub fn with_metric(mut self, metric: Dissimilarity) -> Self {
        self.metric = metric;
        self
    }

    /// Alias for [`ClassicalMDS::with_metric`] retained for consistency with
    /// [`MDS::with_dissimilarity`].
    #[must_use]
    pub fn with_dissimilarity(self, dissimilarity: Dissimilarity) -> Self {
        self.with_metric(dissimilarity)
    }

    /// Return the configured number of components.
    #[must_use]
    pub fn n_components(&self) -> usize {
        self.n_components
    }

    /// Return the configured metric mode.
    #[must_use]
    pub fn metric(&self) -> Dissimilarity {
        self.metric
    }

    /// Fit and return the embedding coordinates.
    ///
    /// This mirrors sklearn's `ClassicalMDS.fit_transform` convenience while the
    /// primary ferrolearn stateful API remains [`Fit::fit`].
    ///
    /// # Errors
    ///
    /// Returns the same validation errors as [`Fit::fit`].
    pub fn fit_transform(&self, x: &Array2<f64>) -> Result<Array2<f64>, FerroError> {
        Ok(self.fit(x, &())?.embedding().clone())
    }
}

/// A fitted ClassicalMDS model holding the learned embedding.
#[derive(Debug, Clone)]
pub struct FittedClassicalMDS {
    /// The embedding, shape `(n_samples, min(n_components, n_samples))`.
    embedding_: Array2<f64>,
    /// The pairwise dissimilarity matrix actually used.
    dissimilarity_matrix_: Array2<f64>,
    /// Eigenvalues of the double-centred dissimilarity matrix selected for the
    /// embedding, sorted descending.
    eigenvalues_: Array1<f64>,
}

impl FittedClassicalMDS {
    /// The embedding coordinates.
    #[must_use]
    pub fn embedding(&self) -> &Array2<f64> {
        &self.embedding_
    }

    /// The pairwise dissimilarity matrix used for the fit.
    #[must_use]
    pub fn dissimilarity_matrix(&self) -> &Array2<f64> {
        &self.dissimilarity_matrix_
    }

    /// Selected eigenvalues of the double-centred dissimilarity matrix.
    #[must_use]
    pub fn eigenvalues(&self) -> &Array1<f64> {
        &self.eigenvalues_
    }
}

// ---------------------------------------------------------------------------
// MDS (unfitted)
// ---------------------------------------------------------------------------

/// SMACOF Multidimensional Scaling configuration.
///
/// Holds hyperparameters for the SMACOF MDS algorithm (matching
/// `sklearn.manifold.MDS`). Call [`Fit::fit`] to compute the embedding and
/// obtain a [`FittedMDS`].
///
/// Constructor defaults mirror sklearn `MDS.__init__` (`_mds.py:493-509`):
/// `n_init = 4`, `max_iter = 300`, `eps = 1e-3`, `normalized_stress = false`
/// (the `"auto"` → `not metric` resolution is `false` for the metric default,
/// `_mds.py:331-332`).
#[derive(Debug, Clone)]
pub struct MDS {
    /// Number of embedding dimensions.
    n_components: usize,
    /// Whether input is a feature matrix or a precomputed distance matrix.
    dissimilarity: Dissimilarity,
    /// Fixed starting configuration. `Some` forces a single deterministic
    /// SMACOF run (`n_init = 1`), matching sklearn `smacof(init=X0, n_init=1)`.
    init: Option<Array2<f64>>,
    /// Number of independent random-init restarts (ignored when `init` is set).
    n_init: usize,
    /// Maximum SMACOF iterations per run.
    max_iter: usize,
    /// Relative-stress convergence tolerance.
    eps: f64,
    /// Report Kruskal Stress-1 (normalized) instead of the raw SSR/2 stress.
    normalized_stress: bool,
    /// Seed for the random-init RNG (default-init path only).
    random_state: Option<u64>,
}

impl MDS {
    /// Create a new `MDS` that embeds into `n_components` dimensions.
    ///
    /// By default the input is treated as a feature matrix
    /// ([`Dissimilarity::Euclidean`]) and the SMACOF defaults mirror sklearn's
    /// `MDS` (`n_init = 4`, `max_iter = 300`, `eps = 1e-3`, raw stress).
    #[must_use]
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            dissimilarity: Dissimilarity::Euclidean,
            init: None,
            n_init: 4,
            max_iter: 300,
            eps: 1e-3,
            normalized_stress: false,
            random_state: None,
        }
    }

    /// Set the dissimilarity mode.
    #[must_use]
    pub fn with_dissimilarity(mut self, d: Dissimilarity) -> Self {
        self.dissimilarity = d;
        self
    }

    /// Set a FIXED starting configuration `init` of shape
    /// `(n_samples, n_components)`.
    ///
    /// With a fixed init the SMACOF Guttman trajectory is deterministic and a
    /// single run is performed (sklearn forces `n_init = 1` when an explicit
    /// init is passed, `_mds.py:339-346`). This is the parity path against
    /// sklearn `smacof(init=X0, n_init=1)`. The init's column count overrides
    /// `n_components` (mirroring `_mds.py:117`).
    #[must_use]
    pub fn with_init(mut self, init: Array2<f64>) -> Self {
        self.init = Some(init);
        self
    }

    /// Set the number of random-init restarts (`n_init`, default 4). Ignored
    /// when a fixed [`MDS::with_init`] is set.
    #[must_use]
    pub fn with_n_init(mut self, n_init: usize) -> Self {
        self.n_init = n_init;
        self
    }

    /// Set the maximum number of SMACOF iterations per run (default 300).
    #[must_use]
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the relative-stress convergence tolerance `eps` (default `1e-3`).
    #[must_use]
    pub fn with_eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }

    /// Report Kruskal Stress-1 (normalized) instead of the raw SSR/2 stress.
    ///
    /// Mirrors sklearn `normalized_stress=True` (`_mds.py:148-149`). sklearn
    /// only supports this for non-metric MDS; ferrolearn currently exposes only
    /// the metric path, so this toggles the reported `stress_` definition.
    #[must_use]
    pub fn with_normalized_stress(mut self, normalized: bool) -> Self {
        self.normalized_stress = normalized;
        self
    }

    /// Seed the random-init RNG (default-init path only). NOTE: the Rust RNG is
    /// NOT numpy-`RandomState`-parity — the default random-init embedding is a
    /// documented non-parity carve-out (REQ-1). Use [`MDS::with_init`] for
    /// element-wise parity with sklearn.
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

    /// Return the configured dissimilarity mode.
    #[must_use]
    pub fn dissimilarity(&self) -> Dissimilarity {
        self.dissimilarity
    }
}

// ---------------------------------------------------------------------------
// FittedMDS
// ---------------------------------------------------------------------------

/// A fitted MDS model holding the learned embedding.
///
/// Created by calling [`Fit::fit`] on an [`MDS`].
#[derive(Debug, Clone)]
pub struct FittedMDS {
    /// The embedding, shape `(n_samples, n_components)`.
    embedding_: Array2<f64>,
    /// The final stress of the best run (raw SSR/2 by default, Kruskal-1 when
    /// `normalized_stress` is set).
    stress_: f64,
    /// The pairwise dissimilarity matrix actually used (`euclidean_distances(X)`
    /// for the Euclidean mode, or the precomputed input).
    dissimilarity_matrix_: Array2<f64>,
    /// The number of SMACOF iterations of the best run.
    n_iter_: usize,
}

impl FittedMDS {
    /// The embedding coordinates, shape `(n_samples, n_components)`.
    #[must_use]
    pub fn embedding(&self) -> &Array2<f64> {
        &self.embedding_
    }

    /// The final SMACOF stress of the best run.
    ///
    /// By default this is the RAW sum-of-squared-residuals `((dis - D)²).sum()/2`
    /// (sklearn `_mds.py:147`). When [`MDS::with_normalized_stress`] is set it is
    /// Kruskal's Stress-1 (`_mds.py:148-149`).
    #[must_use]
    pub fn stress(&self) -> f64 {
        self.stress_
    }

    /// The pairwise dissimilarity matrix used for the fit (sklearn
    /// `dissimilarity_matrix_`, `_mds.py:637-639`).
    #[must_use]
    pub fn dissimilarity_matrix(&self) -> &Array2<f64> {
        &self.dissimilarity_matrix_
    }

    /// The number of SMACOF iterations of the best run (sklearn `n_iter_`).
    #[must_use]
    pub fn n_iter(&self) -> usize {
        self.n_iter_
    }
}

// ---------------------------------------------------------------------------
// Helper: pairwise squared-Euclidean distance matrix
// ---------------------------------------------------------------------------

/// Compute the pairwise squared-Euclidean distance matrix.
pub(crate) fn pairwise_sq_distances(x: &Array2<f64>) -> Array2<f64> {
    let n = x.nrows();
    let mut d = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in (i + 1)..n {
            let mut sq = 0.0;
            for k in 0..x.ncols() {
                let diff = x[[i, k]] - x[[j, k]];
                sq += diff * diff;
            }
            d[[i, j]] = sq;
            d[[j, i]] = sq;
        }
    }
    d
}

/// Compute Kruskal's stress-1.
fn kruskal_stress(dist_orig: &Array2<f64>, embedding: &Array2<f64>) -> f64 {
    let n = embedding.nrows();
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    for i in 0..n {
        for j in (i + 1)..n {
            let d_orig = dist_orig[[i, j]].sqrt();
            let mut sq = 0.0;
            for k in 0..embedding.ncols() {
                let diff = embedding[[i, k]] - embedding[[j, k]];
                sq += diff * diff;
            }
            let d_embed = sq.sqrt();
            let diff = d_orig - d_embed;
            numerator += diff * diff;
            denominator += d_orig * d_orig;
        }
    }
    if denominator > 0.0 {
        (numerator / denominator).sqrt()
    } else {
        0.0
    }
}

/// Pairwise Euclidean distance matrix of an embedding (NOT squared).
///
/// Mirrors `sklearn.metrics.euclidean_distances(X)` used inside the SMACOF loop
/// (`_mds.py:128`). The diagonal is exactly zero.
fn euclidean_distances(x: &Array2<f64>) -> Array2<f64> {
    let n = x.nrows();
    let mut d = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in (i + 1)..n {
            let mut sq = 0.0;
            for k in 0..x.ncols() {
                let diff = x[[i, k]] - x[[j, k]];
                sq += diff * diff;
            }
            let dist = sq.sqrt();
            d[[i, j]] = dist;
            d[[j, i]] = dist;
        }
    }
    d
}

/// Validate that a precomputed dissimilarity matrix is symmetric.
fn ensure_symmetric(x: &Array2<f64>, context: &str) -> Result<(), FerroError> {
    let n = x.nrows();
    for i in 0..n {
        for j in (i + 1)..n {
            let a = x[[i, j]];
            let b = x[[j, i]];
            let scale = a.abs().max(b.abs()).max(1.0);
            if (a - b).abs() > 1e-12 * scale {
                return Err(FerroError::InvalidParameter {
                    name: "X".into(),
                    reason: format!("{context} must be symmetric"),
                });
            }
        }
    }
    Ok(())
}

/// One SMACOF run (metric MDS) from a fixed starting configuration `init`.
///
/// Faithful transcription of `_smacof_single` (`sklearn/manifold/_mds.py:104-167`)
/// for the metric (`metric=True`) case:
///
/// ```text
/// X = init
/// old_stress = None
/// for it in range(max_iter):
///     dis = euclidean_distances(X)                       # :128
///     disparities = dissimilarities                      # metric -> :131
///     stress = ((dis - disparities)**2).sum() / 2        # :147
///     if normalized_stress:
///         stress = sqrt(stress / ((disparities**2).sum()/2))   # :148-149
///     dis[dis == 0] = 1e-5                               # :151
///     ratio = disparities / dis                          # :152
///     B = -ratio; B[i,i] += ratio.sum(axis=1)            # :153-154
///     X = (1/n) * B @ X                                   # :155
///     norm = sqrt((X**2).sum(axis=1)).sum()              # :157
///     if old_stress is not None and (old_stress - stress/norm) < eps: break  # :160-164
///     old_stress = stress / norm                         # :165
/// return X, stress, it+1                                 # :167
/// ```
///
/// `disparities` is the symmetric dissimilarity matrix (zero diagonal). Returns
/// `(embedding, stress, n_iter)`. The returned `stress` is the value computed at
/// the TOP of the LAST executed iteration (before that iteration's Guttman
/// update) — exactly what sklearn returns (`stress` is not recomputed after the
/// final `X` update). Never panics (R-CODE-2).
fn smacof_single(
    disparities: &Array2<f64>,
    init: &Array2<f64>,
    max_iter: usize,
    eps: f64,
    normalized_stress: bool,
) -> (Array2<f64>, f64, usize) {
    let n = disparities.nrows();
    let n_f = n as f64;
    let mut x = init.clone();
    let mut old_stress: Option<f64> = None;
    let mut stress = 0.0_f64;
    let mut iters = 0_usize;

    for it in 0..max_iter {
        iters = it + 1;

        // Current pairwise distances (`_mds.py:128`).
        let dis = euclidean_distances(&x);

        // Raw stress = ((dis - disparities)**2).sum() / 2 (`_mds.py:147`).
        let mut raw = 0.0_f64;
        for i in 0..n {
            for j in 0..n {
                let diff = dis[[i, j]] - disparities[[i, j]];
                raw += diff * diff;
            }
        }
        raw /= 2.0;

        stress = if normalized_stress {
            // sqrt(stress / ((disparities**2).sum()/2)) (`_mds.py:148-149`).
            let mut denom = 0.0_f64;
            for i in 0..n {
                for j in 0..n {
                    denom += disparities[[i, j]] * disparities[[i, j]];
                }
            }
            denom /= 2.0;
            if denom > 0.0 {
                (raw / denom).sqrt()
            } else {
                0.0
            }
        } else {
            raw
        };

        // Guttman transform. dis[dis == 0] = 1e-5 (`_mds.py:151`).
        // ratio = disparities / dis_safe; B = -ratio; B[i,i] += ratio.sum(row).
        // X = (1/n) * B @ X (`_mds.py:152-155`).
        let mut b = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            let mut row_sum = 0.0_f64;
            for j in 0..n {
                let dis_safe = if dis[[i, j]] == 0.0 {
                    1e-5
                } else {
                    dis[[i, j]]
                };
                let ratio = disparities[[i, j]] / dis_safe;
                b[[i, j]] = -ratio;
                row_sum += ratio;
            }
            b[[i, i]] += row_sum;
        }
        let bx = b.dot(&x);
        x = bx.mapv(|v| v / n_f);

        // Convergence on the embedding-norm-normalized stress (`_mds.py:157-165`).
        let mut norm = 0.0_f64;
        for i in 0..n {
            let mut sq = 0.0_f64;
            for k in 0..x.ncols() {
                sq += x[[i, k]] * x[[i, k]];
            }
            norm += sq.sqrt();
        }
        // Match sklearn's `stress / dis` exactly (`_mds.py:161,165`): when the
        // embedding collapses to the origin `norm == 0`, `stress / 0.0` is `+inf`
        // (f64 div-by-zero is inf/nan, NOT a panic — R-CODE-2); the subsequent
        // `prev - normed` yields `nan`/`-inf` and `nan < eps` is False, so SMACOF
        // never breaks early and runs the full `max_iter`, mirroring numpy.
        let normed = stress / norm;
        if let Some(prev) = old_stress
            && (prev - normed) < eps
        {
            break;
        }
        old_stress = Some(normed);
    }

    (x, stress, iters)
}

/// SMACOF over `n_init` independent runs, keeping the lowest-stress result.
///
/// Mirrors `smacof` (`sklearn/manifold/_mds.py:328-392`): when a fixed `init` is
/// supplied a single deterministic run is performed (`n_init` forced to 1,
/// `_mds.py:339-346`); otherwise `n_init` runs start from independent random
/// uniform configurations (`X = random_state.uniform(size=n*n_components)`,
/// `_mds.py:113`). The Rust RNG is Xoshiro256++ (NOT numpy-`RandomState`
/// parity — REQ-1 carve-out). Returns `(embedding, stress, n_iter)` of the
/// best (min-stress) run. Never panics (R-CODE-2).
#[allow(
    clippy::too_many_arguments,
    reason = "mirrors sklearn smacof's parameter set (dissimilarities, n_components, init, n_init, max_iter, eps, normalized_stress, random_state)"
)]
fn smacof_impl(
    disparities: &Array2<f64>,
    n_components: usize,
    init: Option<&Array2<f64>>,
    n_init: usize,
    max_iter: usize,
    eps: f64,
    normalized_stress: bool,
    random_state: Option<u64>,
) -> (Array2<f64>, f64, usize) {
    let n = disparities.nrows();

    if let Some(x0) = init {
        // Fixed init forces a single run (`_mds.py:339-346`). The init's column
        // count determines the embedding dimension (`_mds.py:117`).
        return smacof_single(disparities, x0, max_iter, eps, normalized_stress);
    }

    let seed = random_state.unwrap_or(0);
    let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(seed);
    let runs = n_init.max(1);

    let mut best: Option<(Array2<f64>, f64, usize)> = None;
    for _ in 0..runs {
        // Random uniform init in [0, 1) (mirrors numpy `random_state.uniform`,
        // `_mds.py:113`; NOT numpy-parity — REQ-1 carve-out).
        let x0 = Array2::<f64>::from_shape_fn((n, n_components), |_| rng.random::<f64>());
        let (pos, stress, n_iter) =
            smacof_single(disparities, &x0, max_iter, eps, normalized_stress);
        match &best {
            Some((_, best_stress, _)) if stress >= *best_stress => {}
            _ => best = Some((pos, stress, n_iter)),
        }
    }

    // `runs >= 1`, so `best` is always populated.
    best.unwrap_or_else(|| {
        let x0 = Array2::<f64>::zeros((n, n_components));
        smacof_single(disparities, &x0, max_iter, eps, normalized_stress)
    })
}

/// Run metric SMACOF on a precomputed dissimilarity matrix.
///
/// This is the public helper corresponding to `sklearn.manifold.smacof` for the
/// metric (`metric=True`) path. It returns `(embedding, stress, n_iter)`, where
/// `stress` is raw SSR/2 by default or Kruskal Stress-1 when
/// `normalized_stress` is `true`. A fixed `init` forces a single deterministic
/// run, matching sklearn's `smacof(init=X0, n_init=1)` path.
///
/// The default random-initialization path uses ferrolearn's Rust RNG, not
/// numpy's `RandomState`, so fixed `init` is the value-parity path.
#[allow(
    clippy::too_many_arguments,
    reason = "mirrors sklearn smacof's parameter set (dissimilarities, n_components, init, n_init, max_iter, eps, normalized_stress, random_state)"
)]
pub fn smacof(
    dissimilarities: &Array2<f64>,
    n_components: usize,
    init: Option<&Array2<f64>>,
    n_init: usize,
    max_iter: usize,
    eps: f64,
    normalized_stress: bool,
    random_state: Option<u64>,
) -> Result<(Array2<f64>, f64, usize), FerroError> {
    if n_components == 0 {
        return Err(FerroError::InvalidParameter {
            name: "n_components".into(),
            reason: "must be at least 1".into(),
        });
    }

    reject_non_finite(dissimilarities)?;

    let n_samples = dissimilarities.nrows();
    if n_samples != dissimilarities.ncols() {
        return Err(FerroError::ShapeMismatch {
            expected: vec![n_samples, n_samples],
            actual: vec![dissimilarities.nrows(), dissimilarities.ncols()],
            context: "smacof requires a square dissimilarity matrix".into(),
        });
    }
    if n_samples < 2 {
        return Err(FerroError::InsufficientSamples {
            required: 2,
            actual: n_samples,
            context: "smacof requires at least 2 samples".into(),
        });
    }

    if let Some(x0) = init {
        reject_non_finite(x0)?;
        if x0.nrows() != n_samples {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_samples, n_components],
                actual: vec![x0.nrows(), x0.ncols()],
                context: "smacof init must have shape (n_samples, n_components)".into(),
            });
        }
        if x0.ncols() == 0 {
            return Err(FerroError::InvalidParameter {
                name: "init".into(),
                reason: "must have at least one component column".into(),
            });
        }
    }

    Ok(smacof_impl(
        dissimilarities,
        n_components,
        init,
        n_init,
        max_iter,
        eps,
        normalized_stress,
        random_state,
    ))
}

/// Eigendecompose a symmetric matrix using faer's self-adjoint eigen.
pub(crate) fn eigh_faer(a: &Array2<f64>) -> Result<(Vec<f64>, Array2<f64>), FerroError> {
    let n = a.nrows();
    let mat = faer::Mat::from_fn(n, n, |i, j| a[[i, j]]);
    let decomp = mat.self_adjoint_eigen(faer::Side::Lower).map_err(|e| {
        FerroError::NumericalInstability {
            message: format!("Symmetric eigendecomposition failed: {e:?}"),
        }
    })?;

    let eigenvalues: Vec<f64> = decomp.S().column_vector().iter().copied().collect();
    let eigenvectors = Array2::from_shape_fn((n, n), |(i, j)| decomp.U()[(i, j)]);

    Ok((eigenvalues, eigenvectors))
}

/// Core classical MDS on a squared-distance matrix.
///
/// Returns `(embedding, stress)`.
pub(crate) fn classical_mds(
    sq_dist: &Array2<f64>,
    n_components: usize,
) -> Result<(Array2<f64>, f64), FerroError> {
    let n = sq_dist.nrows();

    // Double-centre: B = -0.5 * J * D^2 * J, where J = I - (1/n) * 11^T
    let n_f = n as f64;
    let mut row_means = vec![0.0; n];
    let mut col_means = vec![0.0; n];
    let mut grand_mean = 0.0;

    for i in 0..n {
        for j in 0..n {
            row_means[i] += sq_dist[[i, j]];
            col_means[j] += sq_dist[[i, j]];
            grand_mean += sq_dist[[i, j]];
        }
    }
    for i in 0..n {
        row_means[i] /= n_f;
        col_means[i] /= n_f;
    }
    grand_mean /= n_f * n_f;

    let mut b = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            b[[i, j]] = -0.5 * (sq_dist[[i, j]] - row_means[i] - col_means[j] + grand_mean);
        }
    }

    // Eigendecompose B
    let (eigenvalues, eigenvectors) = eigh_faer(&b)?;

    // Sort eigenvalues descending
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b_idx| {
        eigenvalues[b_idx]
            .partial_cmp(&eigenvalues[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Build embedding: X_k = v_k * sqrt(lambda_k)
    let n_comp = n_components.min(n);
    let mut embedding = Array2::<f64>::zeros((n, n_comp));
    for (k, &idx) in indices.iter().take(n_comp).enumerate() {
        let eigval = eigenvalues[idx].max(0.0);
        let scale = eigval.sqrt();
        for i in 0..n {
            embedding[[i, k]] = eigenvectors[[i, idx]] * scale;
        }
    }

    // Compute stress
    let stress = kruskal_stress(sq_dist, &embedding);

    Ok((embedding, stress))
}

/// sklearn-style ClassicalMDS on a dissimilarity matrix.
///
/// Unlike the internal `classical_mds` helper used by Isomap, this keeps the
/// selected eigenvalues as sklearn reports them and applies sklearn's
/// `svd_flip(U, None)` sign convention before scaling eigenvectors.
fn classical_mds_sklearn(
    dissimilarities: &Array2<f64>,
    n_components: usize,
) -> Result<(Array2<f64>, Array1<f64>), FerroError> {
    let n = dissimilarities.nrows();
    let n_f = n as f64;

    let sq_dist = dissimilarities.mapv(|v| v * v);
    let mut row_means = vec![0.0; n];
    let mut col_means = vec![0.0; n];
    let mut grand_mean = 0.0;

    for i in 0..n {
        for j in 0..n {
            row_means[i] += sq_dist[[i, j]];
            col_means[j] += sq_dist[[i, j]];
            grand_mean += sq_dist[[i, j]];
        }
    }
    for i in 0..n {
        row_means[i] /= n_f;
        col_means[i] /= n_f;
    }
    grand_mean /= n_f * n_f;

    let mut b = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            b[[i, j]] = -0.5 * (sq_dist[[i, j]] - row_means[i] - col_means[j] + grand_mean);
        }
    }

    let (eigenvalues, eigenvectors) = eigh_faer(&b)?;
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b_idx| {
        eigenvalues[b_idx]
            .partial_cmp(&eigenvalues[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let n_comp = n_components.min(n);
    let mut embedding = Array2::<f64>::zeros((n, n_comp));
    let mut selected_eigenvalues = Array1::<f64>::zeros(n_comp);

    for (k, &idx) in indices.iter().take(n_comp).enumerate() {
        let eigval = eigenvalues[idx];
        selected_eigenvalues[k] = eigval;

        // sklearn `svd_flip(U, None)` with u_based_decision=True: find the
        // maximum-absolute entry in each eigenvector column and make it positive.
        let mut max_abs = 0.0_f64;
        let mut sign = 1.0_f64;
        for i in 0..n {
            let v = eigenvectors[[i, idx]];
            let a = v.abs();
            if a > max_abs {
                max_abs = a;
                sign = if v < 0.0 { -1.0 } else { 1.0 };
            }
        }

        let scale = eigval.sqrt();
        for i in 0..n {
            embedding[[i, k]] = sign * eigenvectors[[i, idx]] * scale;
        }
    }

    Ok((embedding, selected_eigenvalues))
}

// ---------------------------------------------------------------------------
// Trait implementations
// ---------------------------------------------------------------------------

impl Fit<Array2<f64>, ()> for ClassicalMDS {
    type Fitted = FittedClassicalMDS;
    type Error = FerroError;

    /// Fit ClassicalMDS.
    ///
    /// Computes pairwise Euclidean distances for feature input or validates a
    /// precomputed symmetric dissimilarity matrix, then performs classical MDS.
    fn fit(&self, x: &Array2<f64>, _y: &()) -> Result<FittedClassicalMDS, FerroError> {
        if self.n_components == 0 {
            return Err(FerroError::InvalidParameter {
                name: "n_components".into(),
                reason: "must be at least 1".into(),
            });
        }

        reject_non_finite(x)?;

        let dissimilarity_matrix = match self.metric {
            Dissimilarity::Euclidean => {
                if x.nrows() == 0 {
                    return Err(FerroError::InsufficientSamples {
                        required: 1,
                        actual: 0,
                        context: "ClassicalMDS::fit requires at least 1 sample".into(),
                    });
                }
                if x.ncols() == 0 {
                    return Err(FerroError::ShapeMismatch {
                        expected: vec![x.nrows(), 1],
                        actual: vec![x.nrows(), 0],
                        context: "ClassicalMDS::fit requires at least 1 feature".into(),
                    });
                }
                euclidean_distances(x)
            }
            Dissimilarity::Precomputed => {
                if x.nrows() != x.ncols() {
                    return Err(FerroError::ShapeMismatch {
                        expected: vec![x.nrows(), x.nrows()],
                        actual: vec![x.nrows(), x.ncols()],
                        context: "ClassicalMDS with Precomputed metric requires a square matrix"
                            .into(),
                    });
                }
                if x.nrows() == 0 {
                    return Err(FerroError::InsufficientSamples {
                        required: 1,
                        actual: 0,
                        context: "ClassicalMDS::fit requires at least 1 sample".into(),
                    });
                }
                ensure_symmetric(x, "ClassicalMDS precomputed dissimilarity")?;
                x.clone()
            }
        };

        let (embedding, eigenvalues) =
            classical_mds_sklearn(&dissimilarity_matrix, self.n_components)?;

        Ok(FittedClassicalMDS {
            embedding_: embedding,
            dissimilarity_matrix_: dissimilarity_matrix,
            eigenvalues_: eigenvalues,
        })
    }
}

impl Fit<Array2<f64>, ()> for MDS {
    type Fitted = FittedMDS;
    type Error = FerroError;

    /// Fit SMACOF MDS.
    ///
    /// Computes the dissimilarity matrix (`euclidean_distances(X)` for
    /// [`Dissimilarity::Euclidean`], or `X` for [`Dissimilarity::Precomputed`],
    /// mirroring sklearn `_mds.py:636-639`), then runs SMACOF. With a fixed
    /// [`MDS::with_init`] a single deterministic run is performed (parity with
    /// sklearn `smacof(init=X0, n_init=1)`); otherwise `n_init` random restarts
    /// keep the lowest-stress run.
    ///
    /// # Errors
    ///
    /// - [`FerroError::InvalidParameter`] if `n_components` is zero or too large.
    /// - [`FerroError::InsufficientSamples`] if there are fewer than 2 samples.
    /// - [`FerroError::ShapeMismatch`] if `Precomputed` is set but the matrix
    ///   is not square, or if a supplied `init` has the wrong shape.
    fn fit(&self, x: &Array2<f64>, _y: &()) -> Result<FittedMDS, FerroError> {
        if self.n_components == 0 {
            return Err(FerroError::InvalidParameter {
                name: "n_components".into(),
                reason: "must be at least 1".into(),
            });
        }

        // Reject NaN/Inf BEFORE the SMACOF math (sklearn's
        // `_validate_data(force_all_finite=True)` at `_mds.py:627`,
        // `utils/validation.py:147-154`).
        reject_non_finite(x)?;

        // Build the dissimilarity matrix `D` (pairwise distances, NOT squared —
        // SMACOF operates on distances directly, `_mds.py:636-639`).
        let dissimilarity_matrix = match self.dissimilarity {
            Dissimilarity::Euclidean => {
                let n_samples = x.nrows();
                if n_samples < 2 {
                    return Err(FerroError::InsufficientSamples {
                        required: 2,
                        actual: n_samples,
                        context: "MDS::fit requires at least 2 samples".into(),
                    });
                }
                if self.n_components > n_samples {
                    return Err(FerroError::InvalidParameter {
                        name: "n_components".into(),
                        reason: format!(
                            "n_components ({}) exceeds n_samples ({})",
                            self.n_components, n_samples
                        ),
                    });
                }
                euclidean_distances(x)
            }
            Dissimilarity::Precomputed => {
                if x.nrows() != x.ncols() {
                    return Err(FerroError::ShapeMismatch {
                        expected: vec![x.nrows(), x.nrows()],
                        actual: vec![x.nrows(), x.ncols()],
                        context: "MDS with Precomputed dissimilarity requires a square matrix"
                            .into(),
                    });
                }
                let n = x.nrows();
                if n < 2 {
                    return Err(FerroError::InsufficientSamples {
                        required: 2,
                        actual: n,
                        context: "MDS::fit requires at least 2 samples".into(),
                    });
                }
                if self.n_components > n {
                    return Err(FerroError::InvalidParameter {
                        name: "n_components".into(),
                        reason: format!(
                            "n_components ({}) exceeds n_samples ({})",
                            self.n_components, n
                        ),
                    });
                }
                // Input is already a distance matrix; use it directly.
                x.clone()
            }
        };

        let n_samples = dissimilarity_matrix.nrows();

        // Validate a supplied fixed init (`_mds.py:118-121`): it must have
        // `n_samples` rows; its column count overrides `n_components`.
        if let Some(init) = &self.init
            && init.nrows() != n_samples
        {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_samples, self.n_components],
                actual: vec![init.nrows(), init.ncols()],
                context: "MDS init must have shape (n_samples, n_components)".into(),
            });
        }
        // The init's column count determines the embedding dimension
        // (`_mds.py:117`); otherwise use `n_components`.
        let n_components = self
            .init
            .as_ref()
            .map_or(self.n_components, ndarray::Array2::ncols);

        let (embedding, stress, n_iter) = smacof_impl(
            &dissimilarity_matrix,
            n_components,
            self.init.as_ref(),
            self.n_init,
            self.max_iter,
            self.eps,
            self.normalized_stress,
            self.random_state,
        );

        Ok(FittedMDS {
            embedding_: embedding,
            stress_: stress,
            dissimilarity_matrix_: dissimilarity_matrix,
            n_iter_: n_iter,
        })
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

    /// Helper: simple 2D dataset.
    fn square_data() -> Array2<f64> {
        array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0],]
    }

    #[test]
    fn test_mds_basic_embedding_shape() {
        let mds = MDS::new(2);
        let x = square_data();
        let fitted = mds.fit(&x, &()).unwrap();
        assert_eq!(fitted.embedding().dim(), (4, 2));
    }

    #[test]
    fn test_mds_1d_embedding() {
        let mds = MDS::new(1);
        let x = array![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0],];
        let fitted = mds.fit(&x, &()).unwrap();
        assert_eq!(fitted.embedding().ncols(), 1);
    }

    #[test]
    fn test_mds_stress_non_negative() {
        let mds = MDS::new(2);
        let x = square_data();
        let fitted = mds.fit(&x, &()).unwrap();
        assert!(fitted.stress() >= 0.0);
    }

    /// A small deterministic init for the unit-square fixture.
    fn square_init() -> Array2<f64> {
        array![[0.1, 0.2], [0.3, -0.1], [-0.2, 0.4], [0.5, 0.05]]
    }

    #[test]
    #[allow(
        clippy::assertions_on_constants,
        reason = "assert!(false, ...) reports the unexpected fit Err with diagnostics"
    )]
    fn test_mds_perfect_embedding_low_stress() {
        // 2D points embedded into 2D: SMACOF from a fixed init descends the raw
        // stress to ~0 on this exactly-solvable fixture.
        let mds = MDS::new(2).with_init(square_init());
        let Ok(fitted) = mds.fit(&square_data(), &()) else {
            assert!(false, "fit failed");
            return;
        };
        assert!(fitted.stress() < 0.1, "stress = {}", fitted.stress());
    }

    #[test]
    #[allow(
        clippy::assertions_on_constants,
        reason = "assert!(false, ...) reports the unexpected fit Err with diagnostics"
    )]
    fn test_mds_preserves_distances() {
        // SMACOF from a fixed init recovers the input distances on a fixture
        // that is exactly embeddable in 2D.
        let mds = MDS::new(2).with_init(square_init());
        let x = square_data();
        let Ok(fitted) = mds.fit(&x, &()) else {
            assert!(false, "fit failed");
            return;
        };
        let emb = fitted.embedding();

        // Check that pairwise distances in the embedding approximately
        // match the original pairwise distances.
        let orig = pairwise_sq_distances(&x);
        for i in 0..4 {
            for j in (i + 1)..4 {
                let d_orig = orig[[i, j]].sqrt();
                let mut sq = 0.0;
                for k in 0..emb.ncols() {
                    let diff = emb[[i, k]] - emb[[j, k]];
                    sq += diff * diff;
                }
                let d_emb = sq.sqrt();
                assert_abs_diff_eq!(d_orig, d_emb, epsilon = 0.05);
            }
        }
    }

    #[test]
    fn test_mds_precomputed() {
        // Build a precomputed distance matrix.
        let x = square_data();
        let sq = pairwise_sq_distances(&x);
        let dist = sq.mapv(f64::sqrt);

        let mds = MDS::new(2).with_dissimilarity(Dissimilarity::Precomputed);
        let fitted = mds.fit(&dist, &()).unwrap();
        assert_eq!(fitted.embedding().dim(), (4, 2));
    }

    #[test]
    fn test_mds_invalid_n_components_zero() {
        let mds = MDS::new(0);
        let x = square_data();
        assert!(mds.fit(&x, &()).is_err());
    }

    #[test]
    fn test_mds_invalid_n_components_too_large() {
        let mds = MDS::new(10);
        let x = square_data(); // 4 samples
        assert!(mds.fit(&x, &()).is_err());
    }

    #[test]
    fn test_mds_insufficient_samples() {
        let mds = MDS::new(1);
        let x = array![[1.0, 2.0]]; // 1 sample
        assert!(mds.fit(&x, &()).is_err());
    }

    #[test]
    fn test_mds_precomputed_not_square() {
        let mds = MDS::new(1).with_dissimilarity(Dissimilarity::Precomputed);
        let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        assert!(mds.fit(&x, &()).is_err());
    }

    #[test]
    fn test_mds_collinear_data() {
        // Points on a line should embed well into 1D.
        let mds = MDS::new(1);
        let x = array![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0],];
        let fitted = mds.fit(&x, &()).unwrap();
        assert_eq!(fitted.embedding().ncols(), 1);
        // Differences between consecutive embeddings should be roughly equal.
        let emb = fitted.embedding();
        let mut vals: Vec<f64> = (0..5).map(|i| emb[[i, 0]]).collect();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let diffs: Vec<f64> = vals.windows(2).map(|w| (w[1] - w[0]).abs()).collect();
        for d in &diffs {
            assert_abs_diff_eq!(d, &diffs[0], epsilon = 0.1);
        }
    }

    #[test]
    fn test_mds_getters() {
        let mds = MDS::new(3).with_dissimilarity(Dissimilarity::Precomputed);
        assert_eq!(mds.n_components(), 3);
        assert_eq!(mds.dissimilarity(), Dissimilarity::Precomputed);
    }

    #[test]
    fn test_mds_larger_dataset() {
        let n = 20;
        let d = 5;
        let mut data = Array2::<f64>::zeros((n, d));
        for i in 0..n {
            for j in 0..d {
                data[[i, j]] = (i * d + j) as f64 / (n * d) as f64;
            }
        }
        let mds = MDS::new(2);
        let fitted = mds.fit(&data, &()).unwrap();
        assert_eq!(fitted.embedding().dim(), (20, 2));
        assert!(fitted.stress() >= 0.0);
    }

    // -----------------------------------------------------------------------
    // SMACOF fixed-init parity vs live sklearn 1.5.2 (R-CHAR-3).
    // -----------------------------------------------------------------------

    /// The 2404 non-Euclidean precomputed dissimilarity fixture.
    fn fixture_d() -> Array2<f64> {
        array![
            [0.0, 2.0, 5.0, 9.0],
            [2.0, 0.0, 3.0, 4.0],
            [5.0, 3.0, 0.0, 6.0],
            [9.0, 4.0, 6.0, 0.0],
        ]
    }

    /// A fixed init shared by the parity oracles.
    fn fixed_init() -> Array2<f64> {
        array![[0.1, 0.2], [0.3, -0.1], [-0.2, 0.4], [0.5, 0.05]]
    }

    /// Fixed-init SMACOF (precomputed) matches sklearn `smacof(D, metric=True,
    /// init=X0, n_init=1, normalized_stress=False)` element-wise.
    ///
    /// Live sklearn 1.5.2 oracle (`/tmp`, R-CHAR-3):
    /// ```text
    /// smacof(D, metric=True, init=X0, n_init=1, normalized_stress=False,
    ///        max_iter=300, eps=1e-3, return_n_iter=True)
    ///   -> stress_ = 3.148219331054871, n_iter_ = 13,
    ///      embedding = [[-3.333717200034, -1.658330631573],
    ///                   [-0.431085112947, -0.700165295708],
    ///                   [-0.78675047678,   2.465105803376],
    ///                   [ 4.551552789761, -0.106609876095]]
    /// ```
    #[test]
    #[allow(
        clippy::assertions_on_constants,
        reason = "assert!(false, ...) reports the unexpected fit Err with diagnostics"
    )]
    fn smacof_fixed_init_precomputed_parity() {
        const SK_STRESS: f64 = 3.148_219_331_054_871;
        const SK_N_ITER: usize = 13;
        let sk_emb: Array2<f64> = array![
            [-3.333_717_200_034, -1.658_330_631_573],
            [-0.431_085_112_947, -0.700_165_295_708],
            [-0.786_750_476_78, 2.465_105_803_376],
            [4.551_552_789_761, -0.106_609_876_095],
        ];

        let mds = MDS::new(2)
            .with_dissimilarity(Dissimilarity::Precomputed)
            .with_init(fixed_init());
        let Ok(fitted) = mds.fit(&fixture_d(), &()) else {
            assert!(false, "fit failed");
            return;
        };

        assert!(
            (fitted.stress() - SK_STRESS).abs() <= 1e-6,
            "stress {} vs sklearn {SK_STRESS}",
            fitted.stress()
        );
        assert_eq!(fitted.n_iter(), SK_N_ITER, "n_iter mismatch");
        let emb = fitted.embedding();
        assert_eq!(emb.dim(), (4, 2));
        for i in 0..4 {
            for k in 0..2 {
                assert_abs_diff_eq!(emb[[i, k]], sk_emb[[i, k]], epsilon = 1e-6);
            }
        }
    }

    /// Fixed-init SMACOF (Euclidean) matches sklearn `MDS(dissimilarity=
    /// 'euclidean').fit_transform(X, init=X0)` element-wise.
    ///
    /// Live sklearn 1.5.2 oracle (`/tmp`, R-CHAR-3) on the 3-4-5 rectangle:
    /// ```text
    /// stress_ = 0.0013111846996572488, n_iter_ = 13,
    /// embedding = [[-2.164424557023, -1.234049962647],
    ///              [ 0.57663887645,  -2.435876213413],
    ///              [-0.587315085045,  2.433308813391],
    ///              [ 2.175100765618,  1.236617362669]]
    /// ```
    #[test]
    #[allow(
        clippy::assertions_on_constants,
        reason = "assert!(false, ...) reports the unexpected fit Err with diagnostics"
    )]
    fn smacof_fixed_init_euclidean_parity() {
        const SK_STRESS: f64 = 0.001_311_184_699_657_248_8;
        const SK_N_ITER: usize = 13;
        let sk_emb: Array2<f64> = array![
            [-2.164_424_557_023, -1.234_049_962_647],
            [0.576_638_876_45, -2.435_876_213_413],
            [-0.587_315_085_045, 2.433_308_813_391],
            [2.175_100_765_618, 1.236_617_362_669],
        ];

        let x = array![[0.0, 0.0], [3.0, 0.0], [0.0, 4.0], [3.0, 4.0]];
        let mds = MDS::new(2).with_init(fixed_init());
        let Ok(fitted) = mds.fit(&x, &()) else {
            assert!(false, "fit failed");
            return;
        };

        assert!(
            (fitted.stress() - SK_STRESS).abs() <= 1e-6,
            "stress {} vs sklearn {SK_STRESS}",
            fitted.stress()
        );
        assert_eq!(fitted.n_iter(), SK_N_ITER, "n_iter mismatch");
        let emb = fitted.embedding();
        for i in 0..4 {
            for k in 0..2 {
                assert_abs_diff_eq!(emb[[i, k]], sk_emb[[i, k]], epsilon = 1e-6);
            }
        }
    }

    /// `dissimilarity_matrix_` is `euclidean_distances(X)` for the Euclidean
    /// mode (sklearn `_mds.py:639`).
    #[test]
    #[allow(
        clippy::assertions_on_constants,
        reason = "assert!(false, ...) reports the unexpected fit Err with diagnostics"
    )]
    fn smacof_dissimilarity_matrix_is_euclidean() {
        let x = array![[0.0, 0.0], [3.0, 0.0], [0.0, 4.0], [3.0, 4.0]];
        let mds = MDS::new(2).with_init(fixed_init());
        let Ok(fitted) = mds.fit(&x, &()) else {
            assert!(false, "fit failed");
            return;
        };
        let expected: Array2<f64> = array![
            [0.0, 3.0, 4.0, 5.0],
            [3.0, 0.0, 5.0, 4.0],
            [4.0, 5.0, 0.0, 3.0],
            [5.0, 4.0, 3.0, 0.0],
        ];
        let dm = fitted.dissimilarity_matrix();
        for i in 0..4 {
            for j in 0..4 {
                assert_abs_diff_eq!(dm[[i, j]], expected[[i, j]], epsilon = 1e-12);
            }
        }
    }

    /// `with_normalized_stress(true)` reports Kruskal Stress-1
    /// (`sqrt(raw / (Σ disparities²/2))`, sklearn `_mds.py:148-149`) instead of
    /// the raw SSR/2 — a different magnitude class.
    ///
    /// Live sklearn 1.5.2 oracle (`/tmp`, R-CHAR-3): on `fixture_d` the raw
    /// stress is ~3.148 (`> 1`) and the normalized Stress-1 is in `(0, 1)`
    /// (`0.1356...` at the final embedding). Pins that the toggle changes the
    /// reported definition, not the trajectory.
    #[test]
    #[allow(
        clippy::assertions_on_constants,
        reason = "assert!(false, ...) reports the unexpected fit Err with diagnostics"
    )]
    fn smacof_normalized_stress_is_kruskal1() {
        let mds_raw = MDS::new(2)
            .with_dissimilarity(Dissimilarity::Precomputed)
            .with_init(fixed_init());
        let mds_norm = MDS::new(2)
            .with_dissimilarity(Dissimilarity::Precomputed)
            .with_init(fixed_init())
            .with_normalized_stress(true);

        let Ok(raw) = mds_raw.fit(&fixture_d(), &()) else {
            assert!(false, "raw fit failed");
            return;
        };
        let Ok(norm) = mds_norm.fit(&fixture_d(), &()) else {
            assert!(false, "norm fit failed");
            return;
        };
        // Raw stress is the SSR/2 (~3.148); normalized Stress-1 is in [0,1].
        assert!(raw.stress() > 1.0, "raw stress = {}", raw.stress());
        assert!(
            norm.stress() > 0.0 && norm.stress() < 1.0,
            "normalized stress = {}",
            norm.stress()
        );
    }
}
