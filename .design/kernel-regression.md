# Feature: `ferrolearn-kernel` — Kernel Regression

## Summary

Add a new `ferrolearn-kernel` crate implementing multivariate kernel regression with automatic bandwidth selection, heteroscedasticity diagnostics, and robust confidence intervals. This is a Rust-native reimplementation of the [dollspace-gay/package](https://github.com/dollspace-gay/package) Python library, leveraging ferrolearn's existing spatial indexing (`BallTree`, `KdTree`), cross-validation infrastructure (`CrossValidator`, `KFold`), and metrics (`r2_score`, `mean_squared_error`), with `ferray-stats` for SIMD-accelerated statistical reductions and `statrs` for distribution CDFs.

The Python package passes 244 tests and 4 adversarial benchmarks. The Rust port must match or exceed that coverage.

---

## Motivation

Kernel regression is a foundational nonparametric method used in:
- Biotech dose-response curve estimation (this project's regulated domain)
- Causal inference (regression discontinuity designs)
- Exploratory data analysis where parametric assumptions are suspect
- Variance function estimation for heteroscedasticity-robust inference

No Rust crate provides production-quality kernel regression with bandwidth selection, diagnostics, and confidence intervals. ferrolearn's existing infrastructure (spatial trees, CV, metrics) covers ~60% of the dependency surface; `ferray-stats`'s SIMD variance (20x faster than NumPy at 1K elements) makes the hot paths faster than the Python original.

---

## Requirements

### Estimators
- REQ-1: `NadarayaWatson<F>` — local constant (order 0) kernel regression with configurable kernel, bandwidth, and boundary correction (reflection, local linear)
- REQ-2: `LocalPolynomialRegression<F>` — local polynomial regression (orders 0–3+) with Tikhonov regularization and automatic order selection via CV
- REQ-3: Both estimators implement `Fit<Array2<F>, Array1<F>>` producing fitted types that implement `Predict<Array2<F>>`

### Kernel Functions
- REQ-4: Built-in kernels: Gaussian, Epanechnikov, Tricube, Biweight, Triweight, Uniform, Cosine
- REQ-5: Custom kernel support via `Fn(&Array1<F>) -> Array1<F>` closures
- REQ-6: Product kernel for multivariate data: `K(x) = prod_j K((x_j - X_ij) / h_j)`

### Bandwidth Selection
- REQ-7: `CrossValidatedBandwidth<F>` — LOO and k-fold CV bandwidth selection
- REQ-8: O(n) LOOCV via hat matrix diagonal shortcut for Nadaraya-Watson
- REQ-9: Per-dimension bandwidth optimization (anisotropic smoothing) via coordinate descent
- REQ-10: Rule-of-thumb initialization: Silverman (`1.06 * sigma_robust * n^(-1/5)`) and Scott (`sigma * n^(-1/(d+4))`)
- REQ-11: Bandwidth search via log-spaced grid or `argmin`-based optimization

### Diagnostics
- REQ-12: `GoodnessOfFit<F>` — R², adjusted R², AIC, BIC, effective degrees of freedom, leverage values (hat matrix diagonal)
- REQ-13: Heteroscedasticity tests: White, Breusch-Pagan, Goldfeld-Quandt, Dette-Munk-Wagner (bootstrap p-values)
- REQ-14: Residual diagnostics: normality (Jarque-Bera), skewness, kurtosis, standardized residuals

### Confidence Intervals
- REQ-15: Wild bootstrap confidence intervals with Rademacher and Mammen distributions
- REQ-16: Bias correction methods: none, undersmooth, RBC (robust bias-corrected), big brother, RBC studentized (CCF 2018/2022)
- REQ-17: Armstrong-Kolesár honest critical values for bias-adjusted inference
- REQ-18: Fan-Yao nonparametric variance function estimation: `sigma²(x) = E[e² | X=x]`
- REQ-19: Conformal calibration for finite-sample coverage guarantees

### Spatial Acceleration
- REQ-20: Use ferrolearn-neighbors' `BallTree::within_radius` and `KdTree::query` for kernel weight computation, avoiding O(n²) pairwise distances for compact-support kernels

### Testing
- REQ-21: Oracle fixture tests — generate Python reference outputs for standard cases + 3 non-default hyperparameter configurations per estimator
- REQ-22: Property-based tests with minimum 8 invariants (see Section "Invariants")
- REQ-23: Adversarial tests matching the Python package's 4 adversarial benchmarks
- REQ-24: Statistical equivalence tests against Python outputs with ULP tolerance

---

## Acceptance Criteria

- [ ] AC-1: `NadarayaWatson` with `bandwidth=0.5, kernel=gaussian` on a `sin(x)` fixture matches Python predictions within 4 ULPs
- [ ] AC-2: `LocalPolynomialRegression` with `order=1` eliminates boundary bias — prediction at x=1.0 for y=x on [0,1] has bias < 1e-6
- [ ] AC-3: LOOCV hat matrix shortcut produces identical CV error to naive O(n²) LOOCV within 4 ULPs
- [ ] AC-4: Per-dimension bandwidth on (signal, noise) data produces bandwidth ratio > 10x (noise dimension smoothed out)
- [ ] AC-5: White, Breusch-Pagan, and Dette-Munk-Wagner tests detect known heteroscedastic data at alpha=0.05
- [ ] AC-6: `GoodnessOfFit` R² matches `ferrolearn_metrics::r2_score` within 4 ULPs
- [ ] AC-7: Wild bootstrap CI with `rbc_studentized` achieves >= 93% coverage on 1000 simulations with nominal 95%
- [ ] AC-8: Conformal calibration achieves >= 94% coverage on held-out calibration set with nominal 95%
- [ ] AC-9: Collinear input (X1 = X2) does not panic; produces finite predictions with R² >= 0.99 on noiseless linear data
- [ ] AC-10: `BallTree`-accelerated kernel weight computation matches brute-force within 4 ULPs and is faster for n >= 5000
- [ ] AC-11: All fuzz targets run 1 hour without panics
- [ ] AC-12: `cargo clippy -p ferrolearn-kernel -- -D warnings` passes clean

---

## Architecture

### Crate: `ferrolearn-kernel`

```
ferrolearn-kernel/
  Cargo.toml
  README.md
  src/
    lib.rs              # Public API, re-exports
    kernels.rs          # Kernel function trait + built-in kernels
    nadaraya_watson.rs  # NadarayaWatson<F>, FittedNadarayaWatson<F>
    local_polynomial.rs # LocalPolynomialRegression<F>, FittedLocalPolynomialRegression<F>
    bandwidth.rs        # CrossValidatedBandwidth, silverman_bandwidth, scott_bandwidth
    diagnostics.rs      # GoodnessOfFit, heteroscedasticity tests, residual diagnostics
    confidence.rs       # Wild bootstrap CI, conformal calibration, Fan-Yao variance
    hat_matrix.rs       # Hat matrix computation, leverage values, LOOCV shortcut
    weights.rs          # Kernel weight computation (brute-force + spatial-tree dispatch)
  tests/
    oracle_tests.rs
    proptest_invariants.rs
    adversarial_tests.rs
  fixtures/
    nw_gaussian_sin.json
    lpr_order1_boundary.json
    bandwidth_cv_results.json
    heteroscedasticity_known.json
```

### Dependencies

```toml
[dependencies]
ferrolearn-core = { path = "../ferrolearn-core" }
ferrolearn-neighbors = { path = "../ferrolearn-neighbors" }
ndarray = { workspace = true }
num-traits = { workspace = true }
rayon = { workspace = true }
serde = { workspace = true }
thiserror = { workspace = true }
rand = { workspace = true }
statrs = "0.18"

# SIMD-accelerated reductions (default-on, see Decision D1)
ferray-stats = { version = "0.2", optional = true }
ferray-core = { version = "0.2", optional = true }

[dev-dependencies]
proptest = { workspace = true }
approx = { workspace = true }
float-cmp = { workspace = true }
ferrolearn-metrics = { path = "../ferrolearn-metrics" }

[features]
default = ["simd-stats"]
simd-stats = ["dep:ferray-stats", "dep:ferray-core"]
```

### New External Dependencies

| Crate | Version | Purpose | Accuracy |
|-------|---------|---------|----------|
| `statrs` | 0.18 | Chi-squared, F, Normal CDFs for hypothesis tests | 1e-12 to 1e-15 (Lanczos + continued fraction, NIST-validated) |
| `ferray-stats` | 0.2 | SIMD-accelerated mean, variance, std reductions | < 3 ULP (pairwise summation, fused FMA variance) |
| `ferray-core` | 0.2 | Array type for ferray-stats interop (`From`/`Into` ndarray) | N/A |

**Why `statrs`:** Needed for p-value computation in heteroscedasticity tests (chi-squared CDF), normality tests (chi-squared CDF), and F-tests (F-distribution CDF). Same algorithmic family as scipy's Cephes library. Precision gap vs scipy is negligible for hypothesis testing (binary decisions at alpha thresholds). 20M+ downloads, actively maintained.

**Why `ferray-stats`:** Variance and standard deviation are computed on every prediction (kernel weight normalization), every CV iteration (bandwidth evaluation), and every diagnostic (residual analysis). Benchmarks show 2.1–20.8x speedup over NumPy for var/std across all array sizes, with < 3 ULP accuracy via fused SIMD sum-of-squared-differences with FMA.

---

## Type Design

### Kernel Trait (Decision D3: Generic with DynKernel wrapper)

```rust
/// A univariate kernel function K: R -> R+.
///
/// Generic over `F` to enable monomorphization — the compiler inlines
/// kernel evaluation into the weight computation inner loop, enabling
/// auto-vectorization and FMA fusion.
pub trait Kernel<F: Float>: Send + Sync {
    /// Evaluate the kernel at scaled distances u = (x - x_i) / h
    fn evaluate(&self, u: &ArrayView1<F>) -> Array1<F>;

    /// Whether this kernel has compact support (|u| <= 1).
    /// Enables BallTree acceleration for weight computation.
    fn has_compact_support(&self) -> bool;
}
```

Built-in kernels are unit structs implementing `Kernel<F>`:

```rust
pub struct GaussianKernel;      // K(u) = (1/sqrt(2*pi)) * exp(-0.5 * u^2)
pub struct EpanechnikovKernel;  // K(u) = 0.75 * (1 - u^2)  for |u| <= 1
pub struct TricubeKernel;       // K(u) = (70/81) * (1 - |u|^3)^3  for |u| <= 1
pub struct BiweightKernel;      // K(u) = (15/16) * (1 - u^2)^2  for |u| <= 1
pub struct TriweightKernel;     // K(u) = (35/32) * (1 - u^2)^3  for |u| <= 1
pub struct UniformKernel;       // K(u) = 0.5  for |u| <= 1
pub struct CosineKernel;        // K(u) = (pi/4) * cos(pi*u/2)  for |u| <= 1
```

**Runtime selection wrapper** for the string-based API (e.g., `kernel = "gaussian"`):

```rust
/// Type-erased kernel for runtime selection. Wraps `Box<dyn Kernel<F>>`.
pub struct DynKernel<F: Float>(Box<dyn Kernel<F>>);

impl<F: Float> DynKernel<F> {
    /// Construct from a kernel name string.
    pub fn from_name(name: &str) -> Result<Self, FerroError> {
        match name {
            "gaussian" => Ok(Self(Box::new(GaussianKernel))),
            "epanechnikov" => Ok(Self(Box::new(EpanechnikovKernel))),
            "tricube" => Ok(Self(Box::new(TricubeKernel))),
            "biweight" => Ok(Self(Box::new(BiweightKernel))),
            "triweight" => Ok(Self(Box::new(TriweightKernel))),
            "uniform" => Ok(Self(Box::new(UniformKernel))),
            "cosine" => Ok(Self(Box::new(CosineKernel))),
            _ => Err(FerroError::invalid_value(format!("unknown kernel: {name}"))),
        }
    }
}

impl<F: Float> Kernel<F> for DynKernel<F> {
    fn evaluate(&self, u: &ArrayView1<F>) -> Array1<F> { self.0.evaluate(u) }
    fn has_compact_support(&self) -> bool { self.0.has_compact_support() }
}
```

Users who know the kernel at compile time use the concrete type directly for maximum performance. Users who select kernels at runtime (config files, CLI args) use `DynKernel::from_name`.

### Estimator Types

```rust
/// Nadaraya-Watson kernel regression (local constant).
///
/// `K` is the kernel type — use a concrete kernel (e.g., `GaussianKernel`)
/// for maximum performance, or `DynKernel<F>` for runtime selection.
pub struct NadarayaWatson<F: Float, K: Kernel<F> = DynKernel<F>> {
    kernel: K,
    bandwidth: BandwidthStrategy<F>,
    boundary_correction: Option<BoundaryCorrection>,
}

pub struct FittedNadarayaWatson<F: Float, K: Kernel<F> = DynKernel<F>> {
    x_train: Array2<F>,
    y_train: Array1<F>,
    bandwidth: Array1<F>,          // Per-dimension fitted bandwidth
    kernel: K,
    boundary_correction: Option<BoundaryCorrection>,
    x_min: Array1<F>,              // Data bounds for boundary detection
    x_max: Array1<F>,
    spatial_index: SpatialIndex,   // BallTree or BruteForce, built at fit time
}

/// Local polynomial kernel regression.
pub struct LocalPolynomialRegression<F: Float, K: Kernel<F> = DynKernel<F>> {
    kernel: K,
    bandwidth: BandwidthStrategy<F>,
    order: OrderStrategy,
    regularization: F,
}

pub struct FittedLocalPolynomialRegression<F: Float, K: Kernel<F> = DynKernel<F>> {
    x_train: Array2<F>,
    y_train: Array1<F>,
    bandwidth: Array1<F>,
    kernel: K,
    order: usize,                   // Resolved polynomial order
    regularization: F,
    spatial_index: SpatialIndex,
}
```

### Trait Implementations

```rust
impl<F, K> Fit<Array2<F>, Array1<F>> for NadarayaWatson<F, K>
where
    F: Float + Send + Sync + 'static,
    K: Kernel<F> + Clone,
{
    type Fitted = FittedNadarayaWatson<F, K>;
    type Error = FerroError;
    fn fit(&self, x: &Array2<F>, y: &Array1<F>) -> Result<FittedNadarayaWatson<F, K>, FerroError>;
}

impl<F, K> Predict<Array2<F>> for FittedNadarayaWatson<F, K>
where
    F: Float + Send + Sync + 'static,
    K: Kernel<F>,
{
    type Output = Array1<F>;
    type Error = FerroError;
    fn predict(&self, x: &Array2<F>) -> Result<Array1<F>, FerroError>;
}

// Identical pattern for LocalPolynomialRegression<F, K> / FittedLocalPolynomialRegression<F, K>
```

### Configuration Enums

```rust
/// How bandwidth is determined.
pub enum BandwidthStrategy<F: Float> {
    /// Fixed scalar bandwidth (applied to all dimensions).
    Fixed(F),
    /// Fixed per-dimension bandwidth.
    PerDimension(Array1<F>),
    /// Cross-validated selection.
    CrossValidated { cv: CvStrategy, per_dimension: bool },
    /// Silverman's rule of thumb.
    Silverman,
    /// Scott's rule of thumb.
    Scott,
}

/// Cross-validation strategy for bandwidth selection.
pub enum CvStrategy {
    /// Leave-one-out (uses O(n) hat matrix shortcut for NW).
    Loo,
    /// K-fold cross-validation.
    KFold(usize),
}

/// Polynomial order strategy.
pub enum OrderStrategy {
    Fixed(usize),
    CrossValidated { max_order: usize },
}

/// Boundary correction method.
pub enum BoundaryCorrection {
    /// Reflect data near boundaries.
    Reflection,
    /// Use local linear regression near boundaries.
    LocalLinear,
}

/// Bias correction method for wild bootstrap CI.
pub enum BiasCorrection {
    None,
    Undersmooth,
    Rbc,
    BigBrother,
    RbcStudentized,
}
```

---

## Key Algorithms

### 1. Kernel Weight Computation (weights.rs)

The hot inner loop. For each prediction point x, compute weights for all training points:

```
w_i = prod_j K((x_j - X_ij) / h_j) / prod_j h_j
```

**Dispatch strategy:**
- If `kernel.has_compact_support()` and `n_train >= 500`: use `BallTree::within_radius(x, max_bandwidth)` to find candidate neighbors, compute weights only for those. Reduces O(n) to O(k) per query where k << n.
- Otherwise: brute-force vectorized computation over all training points.

**ferray-stats integration:** The weight normalization `sum(w_i)` and weighted mean `sum(w_i * y_i) / sum(w_i)` use `ferray_stats::sum` on the weight slice after conversion to `ferray_core::Array<F, Ix1>` via `From`. This is a zero-copy move for contiguous data.

### 2. LOOCV Hat Matrix Shortcut (hat_matrix.rs)

For Nadaraya-Watson, the LOOCV prediction is:

```
y_hat_{-i} = (y_hat_i - H_ii * y_i) / (1 - H_ii)
```

where `H_ii` is the diagonal of the smoothing matrix `H` with `H_ij = w_ij / sum_k w_ik`.

This gives O(n) LOOCV instead of O(n²) refitting. The hat matrix `H` is an `Array2<F>` computed once; `H_ii` extraction is a diagonal view.

For local polynomial regression (order >= 1), this shortcut does not apply. Fall back to O(n²) naive LOOCV.

### 3. Local Polynomial Fitting (local_polynomial.rs)

At each prediction point x, solve the weighted least squares problem:

```
min_beta  sum_i  w_i * (y_i - beta^T * p(X_i - x))^2 + lambda * ||beta||^2
```

where `p(d)` is the polynomial basis up to the specified order (including cross-terms via `combinations_with_replacement`).

**Solver:** Use `faer` for the weighted least squares solve via QR decomposition. This replaces scipy's `lstsq(lapack_driver='gelsd')`. For numerical stability, augment the design matrix with `sqrt(lambda) * I` (Tikhonov regularization) and solve the augmented system.

**Fallback:** If the solve produces non-finite coefficients, fall back to Nadaraya-Watson (weighted average) for that point.

### 4. Heteroscedasticity Tests (diagnostics.rs)

All tests return a `TestResult { statistic: F, p_value: F, is_significant: bool }`.

**White test:** Regress squared residuals on X, X², and cross-products. Test statistic `n * R²` follows chi-squared(p) under H0. Use `statrs::distribution::ChiSquared::new(p).cdf(n * r2)` for the p-value.

**Breusch-Pagan:** Regress squared residuals on X. Test statistic `n * R²` follows chi-squared(k). Same CDF machinery.

**Goldfeld-Quandt:** Sort by suspected heteroscedastic variable, split into thirds, compare variance of first and last third via F-test. Use `statrs::distribution::FisherSnedecor` for the p-value.

**Dette-Munk-Wagner:** Nonparametric test using kernel smoothing of squared residuals with bootstrap p-values. No distribution CDF needed — p-value comes from bootstrap rank.

### 5. Wild Bootstrap Confidence Intervals (confidence.rs)

```
For b = 1..B:
    1. Generate wild bootstrap weights w_b ~ Rademacher or Mammen
    2. Construct bootstrap response: y*_i = y_hat_i + w_b_i * residual_i
    3. Refit model on (X, y*)
    4. Compute bootstrap prediction at X_pred
    5. Store t*_b = (pred*_b - pred) / SE
Quantile the t* distribution for CI bounds.
```

**Parallelization:** Bootstrap iterations are embarrassingly parallel. Use `rayon::par_iter` over `0..n_bootstrap`, each with an independent RNG seeded from a master seed.

**Bias correction variants:**
- `BigBrother`: Use order p+1 polynomial for cleaner residuals + undersmooth bandwidth by 0.75x
- `RbcStudentized` (CCF 2018/2022): Coverage-error optimal bandwidth (0.6x MSE-optimal), explicit bias estimate from higher-order polynomial, variance inflation for bias estimation uncertainty

### 6. Fan-Yao Variance Estimation (confidence.rs)

Estimate conditional variance `sigma²(x) = E[e² | X=x]`:
1. Compute residuals `e_i = y_i - y_hat_i`
2. Fit a separate kernel regression of `e_i²` on `X` (using a larger bandwidth, typically 1.5x the original)
3. Clip negative estimates to a small positive floor

---

## ferray-stats Integration Points

The following ferrolearn-kernel operations benefit from ferray-stats' SIMD reductions:

| Operation | Frequency | ferray-stats function | Expected speedup |
|-----------|-----------|----------------------|------------------|
| Kernel weight sum | Every prediction point | `ferray_stats::sum` | 1.1–1.9x |
| Weighted mean | Every prediction point | `ferray_stats::mean` (on weighted values) | 1.1–8.7x |
| Bandwidth CV: variance of residuals | Every CV iteration | `ferray_stats::var` | 2.1–20.8x |
| Silverman bandwidth: robust std | Once at fit time | `ferray_stats::std_` | 2.1–15.8x |
| Residual diagnostics: skewness/kurtosis | Once per diagnostic call | Manual (no ferray-stats equivalent yet) | N/A |

**Conversion pattern (feature-gated, see Decision D1):**

```rust
/// Compute variance with SIMD acceleration when available, ndarray fallback otherwise.
fn compute_variance<F: Float + Send + Sync + 'static>(
    data: &ndarray::Array1<F>,
    ddof: usize,
) -> Result<F, FerroError> {
    #[cfg(feature = "simd-stats")]
    {
        use ferray_core::{Array as FeArray, Ix1};
        // Zero-copy move: ndarray::Array1 -> ferray_core::Array<F, Ix1>
        let fe_arr: FeArray<F, Ix1> = data.clone().into();
        let result = ferray_stats::var(&fe_arr, None, ddof)
            .map_err(|e| FerroError::computation(format!("variance: {e}")))?;
        Ok(*result.iter().next().unwrap())
    }
    #[cfg(not(feature = "simd-stats"))]
    {
        let n = F::from(data.len()).unwrap();
        let ddof_f = F::from(ddof).unwrap();
        let mean = data.sum() / n;
        let sum_sq = data.iter().fold(F::zero(), |acc, &x| {
            let d = x - mean;
            acc + d * d
        });
        Ok(sum_sq / (n - ddof_f))
    }
}
```

The `clone()` in the `simd-stats` path is required because ferray-stats takes `&Array` (borrowed) but the `From` impl moves ownership. For hot paths where the source array is not needed afterward, use the owned array directly. For paths where the source is borrowed, the clone cost is negligible compared to the SIMD variance speedup (e.g., clone 10K f64s = 80KB memcpy ≈ 2µs vs variance computation ≈ 310ns with SIMD vs 6.4µs without).

**Note:** The `#[cfg(not(feature = "simd-stats"))]` fallback uses naive two-pass variance. This is intentionally less accurate than ferray-stats' pairwise FMA approach — users who disable `simd-stats` accept this tradeoff. The default-on feature ensures most users get the accurate path.

---

## Spatial Index Dispatch (Decision D2: BallTree only for radius queries)

```rust
enum SpatialIndex<F: Float> {
    /// No spatial index; compute weights over all training points.
    BruteForce,
    /// BallTree for radius queries (compact-support kernels) and k-NN.
    BallTree(ferrolearn_neighbors::BallTree),
}

impl<F: Float + Send + Sync + 'static> SpatialIndex<F> {
    /// Select spatial index strategy based on kernel type and data size.
    fn auto(x: &Array2<F>, compact_support: bool) -> Self {
        let n_samples = x.nrows();
        if compact_support && n_samples >= 500 {
            SpatialIndex::BallTree(BallTree::build(x))
        } else {
            SpatialIndex::BruteForce
        }
    }

    /// Find all training points within `radius` of `query`.
    /// Returns (indices, distances).
    fn within_radius(&self, query: &[f64], radius: f64) -> Vec<(usize, f64)> {
        match self {
            SpatialIndex::BruteForce => {
                unreachable!("within_radius called on BruteForce — use brute-force weight path")
            }
            SpatialIndex::BallTree(tree) => tree.within_radius(query, radius),
        }
    }
}
```

`BallTree::within_radius` already exists in ferrolearn-neighbors. `KdTree` is not used here — it only supports k-NN queries and is not extended (see Decision D2). For Gaussian kernels (infinite support), `BruteForce` is used since no finite radius cutoff is exact. A configurable `gaussian_cutoff_sigma` (default 3.0) can be added later to enable BallTree acceleration for Gaussian kernels with negligible weight truncation.

---

## Property-Based Test Invariants

1. **Symmetry:** `predict(X)` is invariant to row permutation of training data
2. **Interpolation:** With sufficiently small bandwidth, `predict(X_train) ≈ y_train`
3. **Smoothness:** With large bandwidth, predictions converge to `mean(y_train)`
4. **Non-negativity of weights:** All kernel weights are >= 0
5. **Normalization:** For Nadaraya-Watson, prediction is always in `[min(y_train), max(y_train)]`
6. **Dimensionality reduction:** Adding a pure-noise feature should not improve CV score
7. **Bandwidth monotonicity:** For fixed data, increasing bandwidth always increases effective DF (smoother fit)
8. **Variance positivity:** Fan-Yao variance estimates are non-negative at all points
9. **Coverage monotonicity:** Increasing confidence level always widens bootstrap CI
10. **Hat matrix trace:** `trace(H) = sum(H_ii)` equals effective degrees of freedom, and `1 <= trace(H) <= n`

---

## Adversarial Tests (from Python package)

### Test 1: Boundary Bias Trap
Fit `y = x` on `[0, 1]`, predict at `x = 1.0`. NW should show bias > 0.1; LocalPolynomial(order=1) should have bias < 1e-6.

### Test 2: Heteroscedasticity Ghost
Generate data with variance proportional to x. All heteroscedasticity tests must detect it at alpha=0.05.

### Test 3: Curse of Irrelevance
Two features: X1 = signal, X2 = noise. Per-dimension bandwidth must assign bandwidth(X2) > 10 * bandwidth(X1).

### Test 4: Matrix Kill (Collinearity)
Perfectly collinear features (X1 = X2). Model must fit without panic and produce finite predictions.

---

## Implementation Plan

### Agent Breakdown

| Agent | Scope | Depends On | Estimated Tests |
|-------|-------|------------|-----------------|
| A1 | Crate skeleton + kernels + kernel weights (brute-force) | — | 20 |
| A2 | NadarayaWatson + hat matrix LOOCV shortcut | A1 | 30 |
| A3 | Bandwidth selection (CV, Silverman, Scott, per-dimension) | A2 | 25 |
| A4 | LocalPolynomialRegression + order selection | A1, A3 | 25 |
| A5 | Spatial index integration (BallTree dispatch) | A2, A4 | 10 |
| A6 | GoodnessOfFit + heteroscedasticity tests | A2, statrs | 30 |
| A7 | Wild bootstrap CI + bias corrections + Fan-Yao variance | A2, A6 | 35 |
| A8 | Conformal calibration + honest critical values | A7 | 15 |
| A9 | Adversarial tests + benchmarks + README | all | 15 |
| **Total** | | | **~205** |

### Oracle Test Generation

Before Rust implementation, run the Python package to generate JSON fixture files:

```bash
cd /tmp/dollspace-package
python3 -c "
import json, numpy as np
from kernel_regression import NadarayaWatson, LocalPolynomialRegression

np.random.seed(42)
X = np.linspace(0, 2*np.pi, 100).reshape(-1, 1)
y = np.sin(X.ravel()) + 0.1 * np.random.randn(100)

# NW fixture
nw = NadarayaWatson(bandwidth=0.5, kernel='gaussian').fit(X, y)
pred_nw = nw.predict(X)
json.dump({
    'X': X.tolist(), 'y': y.tolist(),
    'predictions': pred_nw.tolist(),
    'bandwidth': nw.bandwidth_.tolist(),
}, open('nw_gaussian_sin.json', 'w'))

# LPR fixture
lpr = LocalPolynomialRegression(order=1, bandwidth=0.5).fit(X, y)
pred_lpr = lpr.predict(X)
json.dump({
    'X': X.tolist(), 'y': y.tolist(),
    'predictions': pred_lpr.tolist(),
    'bandwidth': lpr.bandwidth_.tolist(),
    'order': lpr.order_,
}, open('lpr_order1_sin.json', 'w'))
"
```

---

## Performance Expectations

| Operation | Python (NumPy) | Rust (expected) | Source of speedup |
|-----------|---------------|-----------------|-------------------|
| NW predict (n=10K, d=2) | ~50ms | ~5ms | SIMD weights + BallTree |
| Bandwidth LOOCV (n=1K) | ~3ms | ~0.3ms | Hat matrix shortcut + ferray-stats var |
| Wild bootstrap (B=1000, n=1K) | ~8s | ~0.5s | Rayon parallel + SIMD reductions |
| Per-dim bandwidth CV (n=1K, d=5) | ~2s | ~0.2s | Rayon parallel CV iterations |
| Fan-Yao variance (n=1K) | ~5ms | ~0.5ms | SIMD variance + BallTree |

---

## Resolved Design Decisions

### D1: ferray-stats is default-on (feature `simd-stats`, enabled by default)

**Decision:** ferray-stats is a required dependency, gated behind a default-on `simd-stats` feature flag. Disabling the feature falls back to ndarray's `.mean()` / `.var()`.

**Rationale:** This is an **accuracy** decision, not just performance. ferray-stats uses pairwise summation with FMA for variance computation, which is provably more numerically stable than naive iterative accumulation. ndarray's `.var()` uses standard two-pass computation that suffers from catastrophic cancellation on large arrays with values clustered near the mean. The affected hot paths are:
- **Bandwidth CV** — evaluates variance of residuals hundreds of times during grid search. Accumulated rounding error in variance leads to wrong bandwidth selection.
- **Heteroscedasticity tests** — compute variance of squared residuals (large positive numbers close together, worst case for cancellation).
- **RBC studentized CI** — divides by an inflated standard error estimate. Precision noise in SE degrades coverage.

ferray-stats achieves < 3 ULP accuracy; the speedup (2–20x) is a bonus.

### D2: BallTree only for radius queries; KdTree for k-NN only

**Decision:** The `SpatialIndex` enum uses `BallTree` for all `within_radius` queries (compact-support kernels). `KdTree` is not extended with radius queries.

**Rationale:** Both trees return exact neighbor sets — no approximation, no accuracy difference. `BallTree::within_radius` already exists in ferrolearn-neighbors. Adding `within_radius` to `KdTree` is straightforward but increases maintenance surface for zero accuracy gain. The `SpatialIndex::auto` heuristic prefers `BallTree` when the kernel has compact support, regardless of dimensionality.

### D3: Generic `Kernel<F>` trait with `DynKernel<F>` wrapper for runtime selection

**Decision:** The `Kernel` trait is generic over `F`. A `DynKernel<F>` wrapper provides runtime kernel selection via `Box<dyn Kernel<F>>` for the string-based API (`kernel = "gaussian"`).

**Rationale:** Generic dispatch enables monomorphization, which lets the compiler inline kernel evaluation into the weight computation inner loop. This enables auto-vectorization and FMA fusion that marginally reduces intermediate rounding (fraction of a ULP). The `DynKernel<F>` wrapper preserves the ergonomic runtime-selection API from the Python original without sacrificing performance for users who specify the kernel type statically.

### D4: Pin `statrs = "0.18"`

**Decision:** Use statrs 0.18, not 0.19.

**Rationale:** statrs 0.19 requires Rust 1.87; our MSRV is 1.85. Both versions use identical algorithms (Lanczos + continued fraction) with identical precision (1e-12 to 1e-15). The 0.19 changes are API cleanup (private `prec` module), not accuracy improvements. Revisit when MSRV is bumped.

---

## References

- Nadaraya, E. A. (1964). "On Estimating Regression." Theory of Probability & Its Applications.
- Watson, G. S. (1964). "Smooth Regression Analysis." Sankhya.
- Fan, J., & Gijbels, I. (1996). "Local Polynomial Modelling and Its Applications."
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). "The Elements of Statistical Learning."
- Dette, H., Munk, A., & Wagner, T. (1998). "Estimating the variance in nonparametric regression."
- Armstrong, T. B., & Kolesár, M. (2020). "Simple and Honest Confidence Intervals in Nonparametric Regression."
- Calonico, S., Cattaneo, M. D., & Farrell, M. H. (2018, 2022). Coverage error optimal CI methodology.
- Fan, J., & Yao, Q. (1998). "Efficient Estimation of Conditional Variance Functions in Stochastic Regression."
- Lei, J., et al. (2018). "Distribution-Free Predictive Inference for Regression."
