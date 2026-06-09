//! Random projection transformers for dimensionality reduction.
//!
//! Random projections preserve pairwise distances in expectation (Johnson-Lindenstrauss lemma).
//!
//! - [`GaussianRandomProjection`] — dense Gaussian random matrix
//! - [`SparseRandomProjection`] — sparse random matrix with `{-1, 0, +1}` entries
//!
//! ## REQ status
//!
//! Translation target: scikit-learn 1.5.2 `GaussianRandomProjection` +
//! `SparseRandomProjection` (`sklearn/random_projection.py`). Tracking: #1387.
//! Each REQ is BINARY — SHIPPED (impl + non-test consumer + tests + green
//! verification) or NOT-STARTED (with a concrete open blocker). RNG-COUPLED
//! unit: ferrolearn uses Rust `SmallRng`, sklearn numpy `RandomState`, so exact
//! projection-matrix VALUE parity is impossible (carve-out); the SHIPPED claims
//! are the projection DISTRIBUTION/scale (vs the sklearn formula) + transform
//! contract + determinism, not bit-exact matrix values.
//!
//! | REQ | Scope | Status | Evidence / Blocker |
//! |-----|-------|--------|--------------------|
//! | REQ-1 | Gaussian projection DISTRIBUTION `N(0, 1/n_components)` + transform `X@R` + determinism-given-seed | SHIPPED | [`GaussianRandomProjection`] `fit` scale `1/sqrt(n_components)` matches sklearn `_gaussian_random_matrix` `random_projection.py:200`; empirical variance ≈ 1/k guard + determinism + transform tests in `tests/divergence_random_projection.rs`. Consumer: re-export `lib.rs:167` + `PipelineTransformer` |
//! | REQ-2 | Sparse projection DISTRIBUTION (support `±sqrt(1/(d·n_components))`, probs `{d/2, 1-d, d/2}`, default density `1/sqrt(n_features)`, `density=1` case) + transform | SHIPPED | [`SparseRandomProjection`] `fit` scale/support/probs match sklearn `_sparse_random_matrix` `:301` + `_check_density` `:148-149`; nonzero-entry == ±scale (exact, tol 1e-12) + default-density + density=1 tests |
//! | REQ-3 | Error/parameter contracts (n_components==0, density∉(0,1], zero rows, transform ncols, unfitted) | SHIPPED (scoped) | both `fit`/`Fitted*` `transform`; in-module + divergence error tests |
//! | REQ-4 | Exact projection-matrix VALUE parity (numpy `RandomState` stream) | NOT-STARTED | RNG-coupled CARVE-OUT (Rust `SmallRng` ≠ numpy MT19937), no committed failing test — blocker #1388 |
//! | REQ-5 | Sparse sampling METHOD (per-row `binomial` + `sample_without_replacement` vs per-entry Bernoulli) | NOT-STARTED | sklearn `random_projection.py:281-301` — blocker #1389 |
//! | REQ-6 | `n_components='auto'` + `eps` + `johnson_lindenstrauss_min_dim` | SHIPPED | [`johnson_lindenstrauss_min_dim`] = `floor(4*ln(n_samples)/(eps^2/2-eps^3/3))` matches sklearn `random_projection.py:142-143` (int64 truncation), validates `eps∈(0,1)`/`n_samples>0` (`:133,136`); [`NComponents::Auto`] default resolves via JL at `fit` (sklearn `:388-391`) with the `n_components_ <= 0` reject (`:393-397`). Consumer: re-export `lib.rs:167` + both `fit`. Tests `tests/divergence_random_projection_jl_2347.rs` |
//! | REQ-7 | `components_` `(n_components, n_features)` orientation (#2346) | SHIPPED (orientation) — CSR sparse storage + `dense_output` NOT-STARTED | `components()` / `projection()` now store `(n_components, n_features)` matching sklearn `:419`, transform `X @ components_.T` (`:604,810`); CSR sparse storage + `dense_output` still dense — blocker #1391. Tests `tests/divergence_random_projection_components_orientation_2345.rs` |
//! | REQ-8 | `inverse_transform` + `compute_inverse_components` | NOT-STARTED | sklearn `random_projection.py:356,431-458` — blocker #1392 |
//! | REQ-9 | `n_components > n_features` warning + `components_`/`n_components_`/`n_features_in_`/`density_` attrs + `get_feature_names_out` | SHIPPED (attrs) — `DataDimensionalityWarning` emission NOT-STARTED | `n_components_()`/`n_features_in_()`/`density_()` getters + `(n_components,n_features)` `components_` (`:416,419,788`); the `n_components>n_features` `DataDimensionalityWarning` (`:407-414`) is NOT emitted (no warning facade in ferrolearn — fit still proceeds) + `get_feature_names_out` absent — blocker #1393 |
//! | REQ-10 | PyO3 binding | NOT-STARTED | no `ferrolearn-python` registration — blocker #1394 |
//! | REQ-11 | ferray substrate | NOT-STARTED | dense `Array2` + `num_traits::Float` only — blocker #1395 |

use ferrolearn_core::error::FerroError;
use ferrolearn_core::pipeline::{FittedPipelineTransformer, PipelineTransformer};
use ferrolearn_core::traits::{Fit, FitTransform, Transform};
use ndarray::{Array1, Array2};
use num_traits::Float;
use rand::SeedableRng;
use rand::rngs::SmallRng;
use rand_distr::{Distribution, StandardNormal};

// ---------------------------------------------------------------------------
// johnson_lindenstrauss_min_dim + NComponents
// ---------------------------------------------------------------------------

/// Find a "safe" number of components to randomly project to.
///
/// Mirrors scikit-learn 1.5.2 `johnson_lindenstrauss_min_dim`
/// (`sklearn/random_projection.py:64-143`). The minimum number of components
/// that guarantees an `eps`-embedding for `n_samples` points is
///
/// ```text
/// n_components >= 4 * ln(n_samples) / (eps^2 / 2 - eps^3 / 3)
/// ```
///
/// sklearn casts the result with `.astype(np.int64)`
/// (`random_projection.py:142-143`), which TRUNCATES toward zero (a FLOOR for
/// the positive values this expression produces). This function matches that
/// truncation, e.g. `johnson_lindenstrauss_min_dim(1000, 0.1) == 5920`.
///
/// # Errors
///
/// Returns [`FerroError::InvalidParameter`] if `eps` is not in the open
/// interval `(0, 1)` (sklearn raises `ValueError`, `random_projection.py:133`),
/// or if `n_samples == 0` (sklearn `random_projection.py:136-140`).
#[must_use = "the returned target dimension should be used"]
pub fn johnson_lindenstrauss_min_dim<F: Float>(
    n_samples: usize,
    eps: F,
) -> Result<usize, FerroError> {
    // sklearn random_projection.py:133 — reject eps outside ]0, 1[.
    let zero = F::zero();
    let one = F::one();
    if !(eps > zero && eps < one) {
        return Err(FerroError::InvalidParameter {
            name: "eps".into(),
            reason: "the JL bound is defined for eps in (0, 1)".into(),
        });
    }
    // sklearn random_projection.py:136 — reject n_samples <= 0.
    if n_samples == 0 {
        return Err(FerroError::InvalidParameter {
            name: "n_samples".into(),
            reason: "the JL bound is defined for n_samples greater than zero".into(),
        });
    }

    // denominator = eps^2 / 2 - eps^3 / 3 (random_projection.py:142).
    let two = one + one;
    let three = two + one;
    let denominator = (eps * eps / two) - (eps * eps * eps / three);

    let n = F::from(n_samples).unwrap_or_else(F::infinity);
    let value = F::from(4.0).unwrap_or_else(F::one) * n.ln() / denominator;

    // .astype(np.int64) truncates toward zero (random_projection.py:143).
    // `value` is positive for n_samples >= 1 and eps in (0, 1), so truncation
    // equals floor. Guard against a non-finite/negative result defensively.
    let truncated = value.trunc();
    if !truncated.is_finite() || truncated < zero {
        return Err(FerroError::InvalidParameter {
            name: "n_samples".into(),
            reason: "the JL bound produced a non-finite target dimension".into(),
        });
    }
    Ok(truncated.to_usize().unwrap_or(0))
}

/// Target projection dimensionality, mirroring sklearn's `n_components`
/// parameter which accepts an integer or the string `'auto'`
/// (`random_projection.py:314-317`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NComponents {
    /// Resolve `n_components` from the Johnson-Lindenstrauss bound at fit time
    /// using the configured `eps` (sklearn `n_components='auto'`,
    /// `random_projection.py:388-391`). This is the sklearn default.
    Auto,
    /// An explicit, user-provided number of output dimensions.
    Fixed(usize),
}

impl Default for NComponents {
    /// sklearn defaults `n_components='auto'` (`random_projection.py:326,477`).
    fn default() -> Self {
        Self::Auto
    }
}

impl From<usize> for NComponents {
    fn from(value: usize) -> Self {
        Self::Fixed(value)
    }
}

/// Resolve the configured `n_components` to a concrete dimension at fit time.
///
/// For [`NComponents::Auto`] this calls [`johnson_lindenstrauss_min_dim`] with
/// `n_samples` and `eps` (sklearn `random_projection.py:388-391`). For both
/// branches a resolved value of `0` is rejected — sklearn rejects the `auto`
/// bound resolving to `<= 0` (`random_projection.py:393-397`) and requires
/// explicit `n_components >= 1` (`_parameter_constraints`, `:314-315`).
///
/// For the [`NComponents::Auto`] path ONLY, a resolved dimension GREATER than
/// `n_features` is also rejected — sklearn raises `ValueError` when the
/// JL-resolved `n_components_` exceeds the original space
/// (`random_projection.py:399-405`). The [`NComponents::Fixed`] path does NOT
/// error here: sklearn only emits a `DataDimensionalityWarning` for an explicit
/// `n_components > n_features` (`random_projection.py:407-414`, NOT-STARTED
/// warning REQ-9 / blocker #1393), so the fixed path proceeds unchanged.
///
/// # Errors
///
/// Returns [`FerroError::InvalidParameter`] if the resolved dimension is `0`,
/// if the `Auto` resolved dimension exceeds `n_features`, or if the JL bound's
/// `eps`/`n_samples` validation fails.
fn resolve_n_components(
    n_components: NComponents,
    eps: f64,
    n_samples: usize,
    n_features: usize,
) -> Result<usize, FerroError> {
    let resolved = match n_components {
        NComponents::Fixed(k) => k,
        NComponents::Auto => johnson_lindenstrauss_min_dim(n_samples, eps)?,
    };
    if resolved == 0 {
        return Err(FerroError::InvalidParameter {
            name: "n_components".into(),
            reason: match n_components {
                NComponents::Auto => format!(
                    "eps={eps} and n_samples={n_samples} lead to a target dimension of 0 which is invalid"
                ),
                NComponents::Fixed(_) => "must be >= 1".into(),
            },
        });
    }
    // AUTO PATH ONLY (sklearn `random_projection.py:399-405`): a JL-resolved
    // target dimension larger than the original space is a ValueError. The
    // Fixed path is intentionally excluded — sklearn only WARNS there (`:407-414`).
    if matches!(n_components, NComponents::Auto) && resolved > n_features {
        return Err(FerroError::InvalidParameter {
            name: "n_components".into(),
            reason: format!(
                "eps={eps} and n_samples={n_samples} lead to a target dimension of {resolved} which is larger than the original space with n_features={n_features}"
            ),
        });
    }
    Ok(resolved)
}

// ---------------------------------------------------------------------------
// GaussianRandomProjection
// ---------------------------------------------------------------------------

/// Gaussian random projection transformer.
///
/// Projects data into a lower-dimensional space using a random matrix drawn
/// from `N(0, 1/n_components)`.
///
/// # Examples
///
/// ```
/// use ferrolearn_preprocess::random_projection::GaussianRandomProjection;
/// use ferrolearn_core::traits::{Fit, Transform};
/// use ndarray::Array2;
///
/// let x = Array2::<f64>::ones((10, 50));
/// let proj = GaussianRandomProjection::<f64>::new(5);
/// let fitted = proj.fit(&x, &()).unwrap();
/// let out = fitted.transform(&x).unwrap();
/// assert_eq!(out.shape(), &[10, 5]);
/// ```
#[derive(Debug, Clone)]
pub struct GaussianRandomProjection<F> {
    /// Number of output dimensions, or [`NComponents::Auto`] to resolve via the
    /// Johnson-Lindenstrauss bound at fit time. Defaults to `Auto`.
    n_components: NComponents,
    /// Distortion-rate parameter for the `Auto` JL bound. Defaults to `0.1`
    /// (sklearn `eps=0.1`, `random_projection.py:328`).
    eps: f64,
    /// Optional RNG seed for reproducibility.
    random_state: Option<u64>,
    _marker: std::marker::PhantomData<F>,
}

impl<F: Float + Send + Sync + 'static> GaussianRandomProjection<F> {
    /// Create a new Gaussian random projection with an explicit `n_components`
    /// number of output dimensions (`NComponents::Fixed`).
    #[must_use]
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components: NComponents::Fixed(n_components),
            eps: 0.1,
            random_state: None,
            _marker: std::marker::PhantomData,
        }
    }

    /// Create a new Gaussian random projection with `n_components='auto'`
    /// (sklearn default): the target dimension is resolved at fit time from the
    /// Johnson-Lindenstrauss bound for the configured `eps`
    /// (`random_projection.py:388-391`).
    #[must_use]
    pub fn new_auto() -> Self {
        Self {
            n_components: NComponents::Auto,
            eps: 0.1,
            random_state: None,
            _marker: std::marker::PhantomData,
        }
    }

    /// Set the distortion-rate `eps` used by the `Auto` JL bound
    /// (sklearn `eps`, `random_projection.py:328`).
    #[must_use]
    pub fn eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }

    /// Set the random seed for reproducibility.
    #[must_use]
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }
}

/// Fitted Gaussian random projection holding the projection matrix.
#[derive(Debug, Clone)]
pub struct FittedGaussianRandomProjection<F> {
    /// Projection matrix of shape `(n_components, n_features)` — the sklearn
    /// `components_` orientation (`random_projection.py:419`).
    components: Array2<F>,
    /// Resolved number of output dimensions (`n_components_`).
    n_components: usize,
    /// Number of features seen during fit (`n_features_in_`).
    n_features_in: usize,
}

impl<F: Float + Send + Sync + 'static> FittedGaussianRandomProjection<F> {
    /// Return a reference to the projection matrix `components_` of shape
    /// `(n_components, n_features)` (sklearn `components_`,
    /// `random_projection.py:419`).
    #[must_use]
    pub fn components(&self) -> &Array2<F> {
        &self.components
    }

    /// Alias for [`Self::components`] kept for backward compatibility. Returns
    /// the projection matrix in the sklearn `(n_components, n_features)`
    /// orientation.
    #[must_use]
    pub fn projection(&self) -> &Array2<F> {
        &self.components
    }

    /// Resolved number of output dimensions (`n_components_`).
    #[must_use]
    pub fn n_components_(&self) -> usize {
        self.n_components
    }

    /// Number of features seen during fit (`n_features_in_`).
    #[must_use]
    pub fn n_features_in_(&self) -> usize {
        self.n_features_in
    }
}

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, ()> for GaussianRandomProjection<F> {
    type Fitted = FittedGaussianRandomProjection<F>;
    type Error = FerroError;

    /// Fit the projection by generating a random matrix `R ~ N(0, 1/n_components)`.
    ///
    /// When `n_components` is [`NComponents::Auto`] the target dimension is
    /// resolved from the Johnson-Lindenstrauss bound for `eps`
    /// (`random_projection.py:388-391`).
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::InvalidParameter`] if the resolved `n_components`
    /// is `0` (explicit, or the JL bound resolving to `<= 0`,
    /// `random_projection.py:393-397`), or if `eps` is outside `(0, 1)` for the
    /// `Auto` path. Returns [`FerroError::InsufficientSamples`] if `x` has zero
    /// rows.
    fn fit(&self, x: &Array2<F>, _y: &()) -> Result<FittedGaussianRandomProjection<F>, FerroError> {
        if x.nrows() == 0 {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "GaussianRandomProjection::fit".into(),
            });
        }

        let n_features = x.ncols();
        let n_components =
            resolve_n_components(self.n_components, self.eps, x.nrows(), n_features)?;

        let mut rng: SmallRng = match self.random_state {
            Some(seed) => SmallRng::seed_from_u64(seed),
            None => SmallRng::from_os_rng(),
        };

        let scale = F::one() / F::from(n_components).unwrap_or_else(F::one).sqrt();
        let normal = StandardNormal;
        // components_ orientation (n_components, n_features) — sklearn
        // `random_projection.py:419`.
        let mut components = Array2::zeros((n_components, n_features));
        for v in &mut components {
            let sample: f64 = normal.sample(&mut rng);
            *v = F::from(sample).unwrap_or_else(F::zero) * scale;
        }

        Ok(FittedGaussianRandomProjection {
            components,
            n_components,
            n_features_in: n_features,
        })
    }
}

impl<F: Float + Send + Sync + 'static> Transform<Array2<F>> for FittedGaussianRandomProjection<F> {
    type Output = Array2<F>;
    type Error = FerroError;

    /// Transform data by computing `X @ components_.T`
    /// (sklearn `random_projection.py:604`).
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if `x.ncols() != n_features`.
    fn transform(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        if x.ncols() != self.n_features_in {
            return Err(FerroError::ShapeMismatch {
                expected: vec![x.nrows(), self.n_features_in],
                actual: vec![x.nrows(), x.ncols()],
                context: "FittedGaussianRandomProjection::transform".into(),
            });
        }
        // components_ is (n_components, n_features); X @ components_.T gives
        // (n_samples, n_components) — sklearn `random_projection.py:604`.
        Ok(x.dot(&self.components.t()))
    }
}

impl<F: Float + Send + Sync + 'static> Transform<Array2<F>> for GaussianRandomProjection<F> {
    type Output = Array2<F>;
    type Error = FerroError;

    /// Always returns an error — the projection must be fitted first.
    fn transform(&self, _x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        Err(FerroError::InvalidParameter {
            name: "GaussianRandomProjection".into(),
            reason: "projection must be fitted before calling transform; use fit() first".into(),
        })
    }
}

impl<F: Float + Send + Sync + 'static> FitTransform<Array2<F>> for GaussianRandomProjection<F> {
    type FitError = FerroError;

    /// Fit and transform in one step.
    fn fit_transform(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let fitted = self.fit(x, &())?;
        fitted.transform(x)
    }
}

impl<F: Float + Send + Sync + 'static> PipelineTransformer<F> for GaussianRandomProjection<F> {
    fn fit_pipeline(
        &self,
        x: &Array2<F>,
        _y: &Array1<F>,
    ) -> Result<Box<dyn FittedPipelineTransformer<F>>, FerroError> {
        let fitted = self.fit(x, &())?;
        Ok(Box::new(fitted))
    }
}

impl<F: Float + Send + Sync + 'static> FittedPipelineTransformer<F>
    for FittedGaussianRandomProjection<F>
{
    fn transform_pipeline(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        self.transform(x)
    }
}

// ---------------------------------------------------------------------------
// SparseRandomProjection
// ---------------------------------------------------------------------------

/// Sparse random projection transformer.
///
/// Projects data into a lower-dimensional space using a sparse random matrix
/// with entries `{-1, 0, +1}` drawn with probabilities
/// `{d/2, 1 - d, d/2}`, scaled by `sqrt(1 / (d * n_components))`.
///
/// The default density `d = 1 / sqrt(n_features)` is used when not specified.
///
/// # Examples
///
/// ```
/// use ferrolearn_preprocess::random_projection::SparseRandomProjection;
/// use ferrolearn_core::traits::{Fit, Transform};
/// use ndarray::Array2;
///
/// let x = Array2::<f64>::ones((10, 100));
/// let proj = SparseRandomProjection::<f64>::new(5).random_state(42);
/// let fitted = proj.fit(&x, &()).unwrap();
/// let out = fitted.transform(&x).unwrap();
/// assert_eq!(out.shape(), &[10, 5]);
/// ```
#[derive(Debug, Clone)]
pub struct SparseRandomProjection<F> {
    /// Number of output dimensions, or [`NComponents::Auto`] to resolve via the
    /// Johnson-Lindenstrauss bound at fit time. Defaults to `Auto`.
    n_components: NComponents,
    /// Density of non-zero entries. `None` means `'auto' = 1/sqrt(n_features)`.
    density: Option<f64>,
    /// Distortion-rate parameter for the `Auto` JL bound. Defaults to `0.1`.
    eps: f64,
    /// Optional RNG seed for reproducibility.
    random_state: Option<u64>,
    _marker: std::marker::PhantomData<F>,
}

impl<F: Float + Send + Sync + 'static> SparseRandomProjection<F> {
    /// Create a new sparse random projection with an explicit `n_components`
    /// number of output dimensions (`NComponents::Fixed`).
    #[must_use]
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components: NComponents::Fixed(n_components),
            density: None,
            eps: 0.1,
            random_state: None,
            _marker: std::marker::PhantomData,
        }
    }

    /// Create a new sparse random projection with `n_components='auto'`
    /// (sklearn default): resolved at fit time from the Johnson-Lindenstrauss
    /// bound for the configured `eps` (`random_projection.py:388-391`).
    #[must_use]
    pub fn new_auto() -> Self {
        Self {
            n_components: NComponents::Auto,
            density: None,
            eps: 0.1,
            random_state: None,
            _marker: std::marker::PhantomData,
        }
    }

    /// Set the density of non-zero entries.
    #[must_use]
    pub fn density(mut self, density: f64) -> Self {
        self.density = Some(density);
        self
    }

    /// Set the distortion-rate `eps` used by the `Auto` JL bound.
    #[must_use]
    pub fn eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }

    /// Set the random seed for reproducibility.
    #[must_use]
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }
}

/// Fitted sparse random projection holding the projection matrix.
#[derive(Debug, Clone)]
pub struct FittedSparseRandomProjection<F> {
    /// Projection matrix of shape `(n_components, n_features)` — the sklearn
    /// `components_` orientation (`random_projection.py:419`).
    components: Array2<F>,
    /// Resolved number of output dimensions (`n_components_`).
    n_components: usize,
    /// Number of features seen during fit (`n_features_in_`).
    n_features_in: usize,
    /// Resolved density (`density_`), i.e. `1/sqrt(n_features)` when `'auto'`
    /// (sklearn `density_`, `random_projection.py:788`).
    density: f64,
}

impl<F: Float + Send + Sync + 'static> FittedSparseRandomProjection<F> {
    /// Return a reference to the projection matrix `components_` of shape
    /// `(n_components, n_features)` (sklearn `components_`,
    /// `random_projection.py:419`).
    #[must_use]
    pub fn components(&self) -> &Array2<F> {
        &self.components
    }

    /// Alias for [`Self::components`] kept for backward compatibility. Returns
    /// the projection matrix in the sklearn `(n_components, n_features)`
    /// orientation.
    #[must_use]
    pub fn projection(&self) -> &Array2<F> {
        &self.components
    }

    /// Resolved number of output dimensions (`n_components_`).
    #[must_use]
    pub fn n_components_(&self) -> usize {
        self.n_components
    }

    /// Number of features seen during fit (`n_features_in_`).
    #[must_use]
    pub fn n_features_in_(&self) -> usize {
        self.n_features_in
    }

    /// Resolved density (`density_`), i.e. `1/sqrt(n_features)` when the
    /// constructor density was left at the `'auto'` default
    /// (sklearn `density_`, `random_projection.py:788`).
    #[must_use]
    pub fn density_(&self) -> f64 {
        self.density
    }
}

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, ()> for SparseRandomProjection<F> {
    type Fitted = FittedSparseRandomProjection<F>;
    type Error = FerroError;

    /// Fit the projection by generating a sparse random matrix.
    ///
    /// When `n_components` is [`NComponents::Auto`] the target dimension is
    /// resolved from the Johnson-Lindenstrauss bound for `eps`
    /// (`random_projection.py:388-391`).
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::InvalidParameter`] if the resolved `n_components`
    /// is `0` or `density` is not in `(0, 1]`.
    /// Returns [`FerroError::InsufficientSamples`] if `x` has zero rows.
    fn fit(&self, x: &Array2<F>, _y: &()) -> Result<FittedSparseRandomProjection<F>, FerroError> {
        if x.nrows() == 0 {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "SparseRandomProjection::fit".into(),
            });
        }

        let n_features = x.ncols();
        let n_components =
            resolve_n_components(self.n_components, self.eps, x.nrows(), n_features)?;

        // density='auto' => 1/sqrt(n_features) (sklearn `_check_density:148-149`).
        let d = self
            .density
            .unwrap_or_else(|| 1.0 / (n_features as f64).sqrt());

        if d <= 0.0 || d > 1.0 {
            return Err(FerroError::InvalidParameter {
                name: "density".into(),
                reason: format!("must be in (0, 1], got {d}"),
            });
        }

        let mut rng: SmallRng = match self.random_state {
            Some(seed) => SmallRng::seed_from_u64(seed),
            None => SmallRng::from_os_rng(),
        };

        let scale = F::from(1.0 / (d * n_components as f64).sqrt()).unwrap_or_else(F::one);
        let uniform =
            rand::distr::Uniform::new(0.0_f64, 1.0).map_err(|e| FerroError::InvalidParameter {
                name: "density".into(),
                reason: format!("failed to build uniform sampler: {e}"),
            })?;

        // components_ orientation (n_components, n_features) — sklearn
        // `random_projection.py:419`.
        let mut components = Array2::zeros((n_components, n_features));
        for v in &mut components {
            let u: f64 = uniform.sample(&mut rng);
            if u < d / 2.0 {
                *v = scale.neg();
            } else if u < d {
                *v = scale;
            }
            // else: remains 0
        }

        Ok(FittedSparseRandomProjection {
            components,
            n_components,
            n_features_in: n_features,
            density: d,
        })
    }
}

impl<F: Float + Send + Sync + 'static> Transform<Array2<F>> for FittedSparseRandomProjection<F> {
    type Output = Array2<F>;
    type Error = FerroError;

    /// Transform data by computing `X @ components_.T`
    /// (sklearn `random_projection.py:810`).
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if `x.ncols() != n_features`.
    fn transform(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        if x.ncols() != self.n_features_in {
            return Err(FerroError::ShapeMismatch {
                expected: vec![x.nrows(), self.n_features_in],
                actual: vec![x.nrows(), x.ncols()],
                context: "FittedSparseRandomProjection::transform".into(),
            });
        }
        // components_ is (n_components, n_features); X @ components_.T gives
        // (n_samples, n_components) — sklearn `random_projection.py:810`.
        Ok(x.dot(&self.components.t()))
    }
}

impl<F: Float + Send + Sync + 'static> Transform<Array2<F>> for SparseRandomProjection<F> {
    type Output = Array2<F>;
    type Error = FerroError;

    /// Always returns an error — the projection must be fitted first.
    fn transform(&self, _x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        Err(FerroError::InvalidParameter {
            name: "SparseRandomProjection".into(),
            reason: "projection must be fitted before calling transform; use fit() first".into(),
        })
    }
}

impl<F: Float + Send + Sync + 'static> FitTransform<Array2<F>> for SparseRandomProjection<F> {
    type FitError = FerroError;

    /// Fit and transform in one step.
    fn fit_transform(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let fitted = self.fit(x, &())?;
        fitted.transform(x)
    }
}

impl<F: Float + Send + Sync + 'static> PipelineTransformer<F> for SparseRandomProjection<F> {
    fn fit_pipeline(
        &self,
        x: &Array2<F>,
        _y: &Array1<F>,
    ) -> Result<Box<dyn FittedPipelineTransformer<F>>, FerroError> {
        let fitted = self.fit(x, &())?;
        Ok(Box::new(fitted))
    }
}

impl<F: Float + Send + Sync + 'static> FittedPipelineTransformer<F>
    for FittedSparseRandomProjection<F>
{
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
    use ndarray::Array2;

    // -- GaussianRandomProjection --

    #[test]
    fn test_gaussian_rp_output_shape() {
        let x = Array2::<f64>::ones((10, 50));
        let proj = GaussianRandomProjection::<f64>::new(5).random_state(42);
        let fitted = proj.fit(&x, &()).unwrap();
        let out = fitted.transform(&x).unwrap();
        assert_eq!(out.shape(), &[10, 5]);
    }

    #[test]
    fn test_gaussian_rp_deterministic() {
        let x = Array2::<f64>::ones((10, 20));
        let proj = GaussianRandomProjection::<f64>::new(3).random_state(42);
        let fitted1 = proj.fit(&x, &()).unwrap();
        let out1 = fitted1.transform(&x).unwrap();
        let fitted2 = proj.fit(&x, &()).unwrap();
        let out2 = fitted2.transform(&x).unwrap();
        for (a, b) in out1.iter().zip(out2.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn test_gaussian_rp_zero_components() {
        let x = Array2::<f64>::ones((5, 10));
        let proj = GaussianRandomProjection::<f64>::new(0);
        assert!(proj.fit(&x, &()).is_err());
    }

    #[test]
    fn test_gaussian_rp_empty_input() {
        let x = Array2::<f64>::zeros((0, 10));
        let proj = GaussianRandomProjection::<f64>::new(5);
        assert!(proj.fit(&x, &()).is_err());
    }

    #[test]
    fn test_gaussian_rp_shape_mismatch() {
        let x_train = Array2::<f64>::ones((10, 20));
        let proj = GaussianRandomProjection::<f64>::new(5).random_state(42);
        let fitted = proj.fit(&x_train, &()).unwrap();
        let x_bad = Array2::<f64>::ones((5, 15));
        assert!(fitted.transform(&x_bad).is_err());
    }

    #[test]
    fn test_gaussian_rp_fit_transform() {
        let x = Array2::<f64>::ones((10, 20));
        let proj = GaussianRandomProjection::<f64>::new(5).random_state(42);
        let out = proj.fit_transform(&x).unwrap();
        assert_eq!(out.shape(), &[10, 5]);
    }

    #[test]
    fn test_gaussian_rp_f32() {
        let x = Array2::<f32>::ones((5, 10));
        let proj = GaussianRandomProjection::<f32>::new(3).random_state(42);
        let fitted = proj.fit(&x, &()).unwrap();
        let out = fitted.transform(&x).unwrap();
        assert_eq!(out.shape(), &[5, 3]);
    }

    // -- SparseRandomProjection --

    #[test]
    fn test_sparse_rp_output_shape() {
        let x = Array2::<f64>::ones((10, 100));
        let proj = SparseRandomProjection::<f64>::new(5).random_state(42);
        let fitted = proj.fit(&x, &()).unwrap();
        let out = fitted.transform(&x).unwrap();
        assert_eq!(out.shape(), &[10, 5]);
    }

    #[test]
    fn test_sparse_rp_deterministic() {
        let x = Array2::<f64>::ones((10, 50));
        let proj = SparseRandomProjection::<f64>::new(3).random_state(42);
        let fitted1 = proj.fit(&x, &()).unwrap();
        let out1 = fitted1.transform(&x).unwrap();
        let fitted2 = proj.fit(&x, &()).unwrap();
        let out2 = fitted2.transform(&x).unwrap();
        for (a, b) in out1.iter().zip(out2.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn test_sparse_rp_sparsity() {
        let x = Array2::<f64>::ones((5, 100));
        let proj = SparseRandomProjection::<f64>::new(10).random_state(42);
        let fitted = proj.fit(&x, &()).unwrap();
        let r = fitted.projection();
        // With density = 1/sqrt(100) = 0.1, about 90% should be zero
        let total = r.len();
        let zeros = r.iter().filter(|&&v| v == 0.0).count();
        let sparsity = zeros as f64 / total as f64;
        assert!(
            sparsity > 0.5,
            "expected sparse matrix, got sparsity={sparsity}"
        );
    }

    #[test]
    fn test_sparse_rp_custom_density() {
        let x = Array2::<f64>::ones((5, 20));
        let proj = SparseRandomProjection::<f64>::new(5)
            .density(0.5)
            .random_state(42);
        let fitted = proj.fit(&x, &()).unwrap();
        let out = fitted.transform(&x).unwrap();
        assert_eq!(out.shape(), &[5, 5]);
    }

    #[test]
    fn test_sparse_rp_zero_components() {
        let x = Array2::<f64>::ones((5, 10));
        let proj = SparseRandomProjection::<f64>::new(0);
        assert!(proj.fit(&x, &()).is_err());
    }

    #[test]
    fn test_sparse_rp_invalid_density() {
        let x = Array2::<f64>::ones((5, 10));
        let proj = SparseRandomProjection::<f64>::new(5).density(0.0);
        assert!(proj.fit(&x, &()).is_err());
    }

    #[test]
    fn test_sparse_rp_empty_input() {
        let x = Array2::<f64>::zeros((0, 10));
        let proj = SparseRandomProjection::<f64>::new(5);
        assert!(proj.fit(&x, &()).is_err());
    }

    #[test]
    fn test_sparse_rp_shape_mismatch() {
        let x_train = Array2::<f64>::ones((10, 20));
        let proj = SparseRandomProjection::<f64>::new(5).random_state(42);
        let fitted = proj.fit(&x_train, &()).unwrap();
        let x_bad = Array2::<f64>::ones((5, 15));
        assert!(fitted.transform(&x_bad).is_err());
    }

    #[test]
    fn test_sparse_rp_fit_transform() {
        let x = Array2::<f64>::ones((10, 20));
        let proj = SparseRandomProjection::<f64>::new(5).random_state(42);
        let out = proj.fit_transform(&x).unwrap();
        assert_eq!(out.shape(), &[10, 5]);
    }

    #[test]
    fn test_sparse_rp_f32() {
        let x = Array2::<f32>::ones((5, 10));
        let proj = SparseRandomProjection::<f32>::new(3).random_state(42);
        let fitted = proj.fit(&x, &()).unwrap();
        let out = fitted.transform(&x).unwrap();
        assert_eq!(out.shape(), &[5, 3]);
    }
}
