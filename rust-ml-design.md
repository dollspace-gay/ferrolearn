# Design Document: `ferrolearn` — A Scikit-Learn Equivalent for Rust

**Status:** Proposal  
**Version:** 0.1.0-draft  
**Target Audience:** Rust library authors, ML engineers, systems programmers

---

## 1. Motivation

Python's scikit-learn is arguably the most successful machine learning library ever built. Its success comes not from raw performance, but from a carefully designed API surface: consistent interfaces, composable pipelines, and a principled separation between data, transformers, and estimators. The Rust ecosystem has fragments — `linfa` covers some classical ML, `ndarray` handles N-dimensional arrays, `polars` handles dataframes — but nothing unifies them under a coherent, ergonomic, production-ready framework.

`ferrolearn` (working name) should not be a port of scikit-learn. It should be a Rust-native design that achieves the same *outcomes*: a toolkit where a practitioner can go from raw data to a trained, validated, serializable model without leaving the library ecosystem or fighting the borrow checker.

---

## 2. Goals

- **Complete parity** with scikit-learn's classical ML algorithm coverage
- **Zero mandatory unsafe** in the public API
- **Composable pipelines** with full type safety at compile time
- **Native sparse matrix support** — a first-class citizen, not an afterthought
- **Serialization/deserialization** of trained models via `serde`
- **Optional GPU acceleration** via a pluggable backend trait
- **`no_std` compatibility** for core algorithms where feasible
- **Interoperability** with `ndarray`, `polars`, and `arrow`
- **Formally verified accuracy parity** with scikit-learn via a mandatory six-layer correctness stack (see Section 20)

---

## 3. Non-Goals

- A deep learning framework (that is a separate domain)
- Python bindings (a separate crate should handle this via `pyo3`)
- A data visualization layer (delegate to `plotters` or similar)
- Replacing `ndarray` or `faer` as the linear algebra primitive

---

## 4. Core Architecture

### 4.1 The Estimator Trait

Everything in the library implements at least one of three core traits. This mirrors scikit-learn's design but encodes it in Rust's type system.

```rust
/// A model that can be fit to data.
pub trait Fit<X, Y> {
    type FitResult: Predict<X>;
    type Error: std::error::Error;

    fn fit(&self, x: &X, y: &Y) -> Result<Self::FitResult, Self::Error>;
}

/// A fitted model that can make predictions.
pub trait Predict<X> {
    type Output;
    type Error: std::error::Error;

    fn predict(&self, x: &X) -> Result<Self::Output, Self::Error>;
}

/// A stateful data transformer (e.g. StandardScaler).
pub trait Transform<X> {
    type Output;
    type Error: std::error::Error;

    fn transform(&self, x: &X) -> Result<Self::Output, Self::Error>;
}

/// Fit and transform in one step.
pub trait FitTransform<X>: Sized {
    type Output;
    type Fitted: Transform<X>;
    type Error: std::error::Error;

    fn fit_transform(self, x: &X) -> Result<(Self::Fitted, Self::Output), Self::Error>;
}
```

The key design decision here: `fit()` consumes parameters and returns a *new, fitted type*. This enforces at compile time that you cannot call `predict()` on an unfitted model — there is no unfitted model that implements `Predict`. This is a strict improvement over scikit-learn, where calling `predict()` before `fit()` is a runtime error.

### 4.2 Data Representations

The library must be agnostic over input representation. A blanket-impl strategy over a `Dataset` trait allows flexibility.

```rust
pub trait Dataset {
    fn n_samples(&self) -> usize;
    fn n_features(&self) -> usize;
    fn is_sparse(&self) -> bool { false }
}

// Implementations for:
// - ndarray::Array2<f32>
// - ndarray::Array2<f64>
// - SparseMatrix<f32> (CSR and CSC formats, see Section 8)
// - polars::DataFrame (via feature flag)
// - arrow::RecordBatch (via feature flag)
```

Numeric precision should be parametric. Algorithms should be generic over `Float: num_traits::Float + Send + Sync`.

### 4.3 Pipeline

The pipeline is the most important ergonomic feature in scikit-learn. In Rust, this requires careful design to avoid losing type information.

```rust
// Desired usage:
let pipeline = Pipeline::new()
    .step("scaler", StandardScaler::new())
    .step("pca", PCA::new().n_components(10))
    .step("clf", LogisticRegression::new().max_iter(1000));

let fitted = pipeline.fit(&x_train, &y_train)?;
let predictions = fitted.predict(&x_test)?;
```

Internally, pipeline steps are stored as boxed trait objects to allow heterogeneous step types while keeping the API ergonomic. A compile-time pipeline (using const generics or type-level lists) should be available as an opt-in for zero-cost abstraction.

---

## 5. Algorithm Coverage

The following table maps scikit-learn's algorithm categories to required implementations. All algorithms must support both `f32` and `f64`.

### 5.1 Supervised Learning — Classification

| Algorithm | Priority | Notes |
|---|---|---|
| Logistic Regression | P0 | L1, L2, ElasticNet penalties; multi-class via OvR and softmax |
| Linear SVM (SVC) | P0 | SMO solver; kernel trick via pluggable `Kernel` trait |
| Kernel SVM | P1 | RBF, polynomial, sigmoid kernels |
| k-Nearest Neighbors | P0 | Ball tree and KD-tree backends |
| Decision Tree | P0 | Gini and entropy criteria; max depth, min samples controls |
| Random Forest | P0 | Parallelized via Rayon |
| Gradient Boosting | P0 | Must include histogram-based variant (HistGB) |
| AdaBoost | P1 | |
| Naive Bayes | P0 | Gaussian, Multinomial, Bernoulli, Complement variants |
| Linear Discriminant Analysis | P1 | |
| Quadratic Discriminant Analysis | P2 | |
| Perceptron | P1 | |
| Ridge Classifier | P1 | |

### 5.2 Supervised Learning — Regression

| Algorithm | Priority | Notes |
|---|---|---|
| Linear Regression | P0 | OLS via QR decomposition; closed-form and iterative solvers |
| Ridge | P0 | L2 regularization |
| Lasso | P0 | Coordinate descent solver |
| ElasticNet | P0 | |
| Bayesian Ridge | P1 | |
| SGD Regressor | P0 | Mini-batch support; multiple loss functions |
| SVR | P1 | |
| k-Nearest Neighbors Regression | P0 | |
| Decision Tree Regressor | P0 | |
| Random Forest Regressor | P0 | |
| Gradient Boosting Regressor | P0 | |
| Huber Regressor | P1 | |
| Isotonic Regression | P2 | |
| RANSAC | P2 | |

### 5.3 Unsupervised Learning — Clustering

| Algorithm | Priority | Notes |
|---|---|---|
| k-Means | P0 | k-Means++ initialization; parallelized via Rayon |
| Mini-Batch k-Means | P1 | |
| DBSCAN | P0 | |
| HDBSCAN | P1 | |
| Agglomerative Clustering | P1 | Ward, complete, average, single linkage |
| Gaussian Mixture Models | P1 | EM algorithm; full, tied, diag, spherical covariance |
| Mean Shift | P2 | |
| Spectral Clustering | P2 | |
| OPTICS | P2 | |
| Birch | P2 | |

### 5.4 Dimensionality Reduction

| Algorithm | Priority | Notes |
|---|---|---|
| PCA | P0 | Full, truncated (randomized SVD), incremental variants |
| Truncated SVD | P0 | Works on sparse matrices — critical for NLP |
| Kernel PCA | P1 | |
| t-SNE | P1 | Barnes-Hut approximation for large datasets |
| UMAP | P1 | |
| LDA (topic model) | P1 | |
| Factor Analysis | P2 | |
| ICA | P2 | |
| NMF | P1 | Multiplicative update and coordinate descent solvers |
| Dictionary Learning | P2 | |

### 5.5 Manifold Learning

| Algorithm | Priority | Notes |
|---|---|---|
| Isomap | P2 | |
| Locally Linear Embedding | P2 | |
| MDS | P2 | |
| Spectral Embedding | P2 | |

---

## 6. Preprocessing

Preprocessing transformers must implement `FitTransform` and be pipeline-composable.

### 6.1 Scaling

| Transformer | Notes |
|---|---|
| `StandardScaler` | Zero mean, unit variance; handles `f32`/`f64` |
| `MinMaxScaler` | Configurable feature range |
| `RobustScaler` | Median and IQR based; resistant to outliers |
| `MaxAbsScaler` | Scales each feature by its max absolute value |
| `Normalizer` | Per-sample L1, L2, or max normalization |
| `PowerTransformer` | Box-Cox and Yeo-Johnson |
| `QuantileTransformer` | Uniform and Gaussian output distributions |

### 6.2 Encoding

| Transformer | Notes |
|---|---|
| `OneHotEncoder` | Handles unknown categories at transform time |
| `OrdinalEncoder` | |
| `LabelEncoder` | |
| `TargetEncoder` | With cross-fitting to prevent leakage |
| `BinaryEncoder` | |

### 6.3 Feature Engineering

| Transformer | Notes |
|---|---|
| `PolynomialFeatures` | Interaction terms; configurable degree |
| `SplineTransformer` | B-spline basis expansion |
| `KBinsDiscretizer` | Uniform, quantile, k-means binning strategies |
| `Binarizer` | |
| `FunctionTransformer` | Wrap any `Fn` as a transformer |

### 6.4 Imputation

| Transformer | Notes |
|---|---|
| `SimpleImputer` | Mean, median, most frequent, constant |
| `KNNImputer` | |
| `IterativeImputer` | MICE-style multivariate imputation |

### 6.5 Feature Selection

| Transformer | Notes |
|---|---|
| `VarianceThreshold` | |
| `SelectKBest` | Pluggable scoring functions |
| `SelectPercentile` | |
| `RFE` | Recursive Feature Elimination |
| `RFECV` | RFE with cross-validation |
| `SelectFromModel` | Threshold on `feature_importances_` or coefficients |
| `SequentialFeatureSelector` | Forward and backward |

---

## 7. Model Selection and Evaluation

### 7.1 Cross-Validation

```rust
let cv = KFold::new(5).shuffle(true).random_state(42);
let scores = cross_val_score(&estimator, &x, &y, &cv, accuracy_score)?;
```

Required splitters:

- `KFold` / `StratifiedKFold` / `GroupKFold`
- `ShuffleSplit` / `StratifiedShuffleSplit`
- `TimeSeriesSplit`
- `LeaveOneOut` / `LeavePOut`
- `LeaveOneGroupOut` / `LeavePGroupsOut`

### 7.2 Hyperparameter Search

```rust
let param_grid = param_grid! {
    "max_depth" => [3, 5, 10, None],
    "min_samples_split" => [2, 5, 10],
};

let search = GridSearchCV::new(estimator, param_grid)
    .cv(StratifiedKFold::new(5))
    .scoring(f1_score)
    .n_jobs(-1); // Use all available threads via Rayon

let fitted_search = search.fit(&x_train, &y_train)?;
println!("Best params: {:?}", fitted_search.best_params());
```

| Search Method | Notes |
|---|---|
| `GridSearchCV` | Exhaustive grid; parallel via Rayon |
| `RandomizedSearchCV` | Samples from distributions |
| `HalvingGridSearchCV` | Successive halving for large grids |

Hyperparameter distributions must be expressible:

```rust
pub trait Distribution<T>: Send + Sync {
    fn sample(&self, rng: &mut impl Rng) -> T;
}

// Provided: Uniform, LogUniform, Normal, LogNormal, IntUniform, Choice
```

### 7.3 Metrics

All metrics must work with both owned and borrowed arrays.

**Classification:**
- `accuracy_score`, `balanced_accuracy_score`
- `precision_score`, `recall_score`, `f1_score`, `fbeta_score`
- `roc_auc_score`, `average_precision_score`
- `confusion_matrix`, `classification_report`
- `matthews_corrcoef`, `cohen_kappa_score`
- `log_loss`, `brier_score_loss`
- `roc_curve`, `precision_recall_curve`

**Regression:**
- `mean_absolute_error`, `mean_squared_error`, `root_mean_squared_error`
- `mean_absolute_percentage_error`
- `r2_score`, `explained_variance_score`
- `median_absolute_error`
- `mean_squared_log_error`
- `max_error`
- `d2_tweedie_score`

**Clustering:**
- `adjusted_rand_score`, `rand_score`
- `adjusted_mutual_info_score`, `normalized_mutual_info_score`
- `homogeneity_score`, `completeness_score`, `v_measure_score`
- `silhouette_score`, `silhouette_samples`
- `calinski_harabasz_score`
- `davies_bouldin_score`

---

## 8. Sparse Matrix Support

This is the most commonly cited gap in the Rust ML ecosystem. Sparse support must be a **first-class primitive**, not an optional extension.

### 8.1 Formats

```rust
pub enum SparseFormat {
    CSR,  // Compressed Sparse Row — fast row slicing, matrix-vector products
    CSC,  // Compressed Sparse Column — fast column slicing
    COO,  // Coordinate format — good for construction
    LIL,  // List of Lists — good for incremental construction
    DOK,  // Dictionary of Keys — fast element access
}
```

All algorithms that make statistical sense on sparse data (TruncatedSVD, NMF, Naive Bayes, linear models) must accept sparse inputs natively without converting to dense.

### 8.2 Sparse Operations

Required operations:

- Matrix-vector and matrix-matrix multiply (via BLAS-compatible routines)
- Elementwise arithmetic preserving sparsity
- Slicing (rows, columns, arbitrary indexing)
- Conversion between formats
- Vertical and horizontal stacking (`vstack`, `hstack`)
- `toarray()` / `todense()` conversions
- Arithmetic with dense arrays

### 8.3 Memory Layout

```rust
pub struct CsrMatrix<T> {
    data: Vec<T>,        // Non-zero values
    indices: Vec<usize>, // Column indices for each value
    indptr: Vec<usize>,  // Row pointer array
    shape: (usize, usize),
}
```

The library should integrate with or wrap `sprs` where it is sufficient, but not depend on it for correctness of algorithm implementations.

---

## 9. Parallelism Strategy

Parallelism should be opt-in at the algorithm level and transparent to the caller.

```rust
let rf = RandomForest::new()
    .n_estimators(100)
    .n_jobs(-1); // -1 = use all cores, matching scikit-learn convention
```

Internally, `-1` maps to `rayon::current_num_threads()`. Individual algorithms document their parallelism strategy.

Rules:
- **Rayon** for CPU parallelism across samples and estimators
- All parallel code must be deterministic when a `random_state` seed is provided
- Reproducibility is a hard requirement, not a best-effort

---

## 10. Serialization and Model Persistence

Trained models must be serializable. This is a known gap in the current Rust ML ecosystem.

```rust
use ferrolearn::io::{save_model, load_model};

let model = RandomForest::new().n_estimators(100).fit(&x, &y)?;

// Save
save_model(&model, "model.fl")?;

// Load — type must be specified
let loaded: FittedRandomForest<f64> = load_model("model.fl")?;
```

### 10.1 Format Requirements

- **Native format:** `MessagePack` or `bincode` + metadata envelope. Compact, fast, version-tagged.
- **JSON export:** Human-readable via `serde_json` for debugging and interoperability
- **ONNX export (P1):** For deployment interop with runtimes like `tract` or `onnxruntime`
- **PMML export (P2):** For enterprise/legacy system interoperability

### 10.2 Versioning

Every serialized model must include a schema version. Deserialization must fail fast with a clear error when the version is incompatible, rather than silently producing wrong results.

---

## 11. Numeric Backends

The library must support a pluggable compute backend to avoid locking into a single linear algebra stack.

```rust
pub trait Backend: Send + Sync {
    fn gemm<F: Float>(&self, a: &Array2<F>, b: &Array2<F>) -> Array2<F>;
    fn svd<F: Float>(&self, a: &Array2<F>) -> (Array2<F>, Array1<F>, Array2<F>);
    // ... etc
}

pub struct NdarrayBackend;   // Default; pure Rust via ndarray + faer
pub struct BlasBackend;      // Links system BLAS/LAPACK (OpenBLAS, MKL)
pub struct CudaBackend;      // Optional GPU via cuBLAS (feature flag)
```

This allows the library to be used in environments without BLAS (e.g. embedded, WASM) while still achieving near-LAPACK performance in production deployments where BLAS is available.

---

## 12. Error Handling

All public functions return `Result<T, FerroError>`. Panics are forbidden in library code except for internal invariant violations (which should be unreachable in correct usage).

```rust
#[non_exhaustive]
pub enum FerroError {
    ShapeMismatch { expected: Shape, got: Shape },
    InsufficientSamples { needed: usize, got: usize },
    ConvergenceFailure { iterations: usize, tolerance: f64 },
    InvalidParameter { name: &'static str, reason: String },
    NumericalInstability { context: String },
    IoError(std::io::Error),
    SerdeError(String),
}
```

Every error variant must carry enough context to identify the root cause without a debugger.

---

## 13. Introspection and Interpretability

Fitted models that support it must expose their learned parameters.

```rust
pub trait HasCoefficients {
    fn coef(&self) -> ArrayView2<f64>;
    fn intercept(&self) -> ArrayView1<f64>;
}

pub trait HasFeatureImportances {
    fn feature_importances(&self) -> ArrayView1<f64>;
}

pub trait HasClasses<L> {
    fn classes(&self) -> &[L];
}
```

This mirrors scikit-learn's `coef_`, `feature_importances_`, `classes_` attributes. The postfix underscore convention from Python becomes a method in Rust (no underscore needed since methods are not confused with parameters).

---

## 14. Dataset Utilities

The library should ship utilities for working with datasets, both real and synthetic.

### 14.1 Toy Datasets

```rust
use ferrolearn::datasets;

let iris = datasets::load_iris();
let (x, y) = (iris.data, iris.target);
```

Required toy datasets: Iris, Digits, Wine, Breast Cancer, Diabetes, Linnerud, Olivetti Faces.

### 14.2 Generators

```rust
let (x, y) = make_classification()
    .n_samples(1000)
    .n_features(20)
    .n_informative(10)
    .random_state(0)
    .generate()?;
```

Required generators: `make_classification`, `make_regression`, `make_blobs`, `make_moons`, `make_circles`, `make_swiss_roll`, `make_s_curve`, `make_sparse_uncorrelated`.

### 14.3 Train/Test Split

```rust
let (x_train, x_test, y_train, y_test) =
    train_test_split(&x, &y, 0.2, Some(42))?;
```

---

## 15. Crate Structure

The repository should be organized as a Cargo workspace:

```
ferrolearn/
├── ferrolearn/              # Main crate — re-exports everything
├── ferrolearn-core/         # Traits, errors, Dataset abstractions
├── ferrolearn-linear/       # Linear models
├── ferrolearn-tree/         # Decision trees, random forests, boosting
├── ferrolearn-cluster/      # Clustering algorithms
├── ferrolearn-decomp/       # PCA, SVD, NMF, manifold methods
├── ferrolearn-preprocess/   # All preprocessing transformers
├── ferrolearn-metrics/      # All metrics
├── ferrolearn-model-sel/    # Cross-validation, hyperparameter search
├── ferrolearn-sparse/       # Sparse matrix types and operations
├── ferrolearn-datasets/     # Toy datasets and generators
└── ferrolearn-io/           # Model serialization and ONNX/PMML export
```

Feature flags on the main crate:

| Feature | Default | Description |
|---|---|---|
| `full` | No | Enables everything |
| `blas` | No | Link system BLAS/LAPACK |
| `cuda` | No | GPU backend via cuBLAS |
| `polars` | No | `polars::DataFrame` as `Dataset` |
| `arrow` | No | `arrow::RecordBatch` as `Dataset` |
| `onnx` | No | ONNX model export |
| `rayon` | Yes | Parallel execution |
| `serde` | Yes | Model serialization |

---

## 16. Testing Requirements

> **Note:** All correctness, oracle, property-based, statistical equivalence, fuzz, and formal verification requirements are defined authoritatively in **Section 20** and are hard release gates. The summaries below are superseded by Section 20.

### 16.1 Correctness Testing

See **Section 20** for the complete mandatory correctness stack. Every algorithm requires oracle fixture tests (20.1), property-based invariant tests (20.2), statistical equivalence benchmarking (20.3), algorithm equivalence documentation (20.4), fuzz targets (20.5), and formal verification of metrics and data structures (20.6).

### 16.2 Performance Benchmarks

Use `criterion` for all performance benchmarks. Benchmarks must run against:
- Small (100 u00d7 10), medium (10k u00d7 100), and large (100k u00d7 1000) datasets
- Both `f32` and `f64`
- Dense and sparse inputs where applicable

Benchmark targets: match or exceed scikit-learn+NumPy throughput on CPU for all classical algorithms. Performance benchmarks do not gate releases independently — correctness always takes precedence.
---

## 17. Documentation Standards

Every public item must have:
- A one-line summary
- A description of the algorithm and its time/space complexity
- Parameter documentation with valid ranges and defaults
- At least one complete, runnable `# Examples` block
- A `# References` section with the authoritative paper or textbook citation

---

## 18. Minimum Supported Rust Version (MSRV)

The crate targets **Rust 1.75** (stable, December 2023) as the MSRV to maximize compatibility while leveraging `impl Trait` in return position, `async fn` in traits (where needed), and const generics.

---

## 19. Roadmap

### Phase 1 — Foundation (Months 1–3)
- Core traits (`Fit`, `Predict`, `Transform`, `FitTransform`)
- Dense matrix integration with `ndarray` and `faer`
- Sparse matrix types (CSR, CSC, COO)
- `StandardScaler`, `MinMaxScaler`, `RobustScaler`
- `OneHotEncoder`, `LabelEncoder`
- `train_test_split`, `KFold`, `StratifiedKFold`
- All regression metrics, all classification metrics
- Logistic Regression, Linear Regression, Ridge, Lasso
- `Pipeline` (dynamic dispatch variant)

### Phase 2 — Classical ML Core (Months 4–6)
- Decision Tree (classification + regression)
- Random Forest (classification + regression)
- k-Nearest Neighbors
- Naive Bayes (all variants)
- SVM (linear; kernel as stretch goal)
- k-Means, DBSCAN
- PCA, TruncatedSVD
- `GridSearchCV`, `RandomizedSearchCV`
- Model serialization (native format)
- Toy datasets + generators

### Phase 3 — Completeness (Months 7–12)
- Gradient Boosting (including histogram variant)
- Full unsupervised suite (GMM, HDBSCAN, agglomerative)
- Full dimensionality reduction suite (t-SNE, UMAP, NMF)
- All imputers
- Full feature selection suite
- `TimeSeriesSplit`
- ONNX export
- BLAS backend
- Polars + Arrow integration

### Phase 4 — Beyond Scikit-Learn (Month 13+)
- GPU backend (CUDA)
- Online/streaming learning API for estimators that support it
- Calibration (`CalibratedClassifierCV`)
- Semi-supervised learning
- `ColumnTransformer` equivalent with column selection from `polars`

---

## 20. Correctness Verification Requirements

This section defines the mandatory correctness infrastructure that must ship alongside every algorithm. Passing all layers described here is a hard release gate — no algorithm may be published to crates.io without satisfying all applicable tiers. The goal is to make provable accuracy parity with scikit-learn a first-class, auditable property of the library, not an informal claim.

There are three distinct kinds of correctness that must each be addressed, and no single technique covers all three:

- **Numerical correctness** — does the implementation produce the same floating-point results as scikit-learn on identical inputs?
- **Algorithmic correctness** — does the procedure satisfy its mathematical specification as a logical invariant, independent of any reference implementation?
- **Statistical correctness** — does the fitted model achieve equivalent predictive quality on real-world data?

The six verification layers below address these in combination.

---

### 20.1 Layer 1 — Oracle Testing with Golden Fixtures (REQUIRED for all algorithms)

Every algorithm must have a fixture suite generated from scikit-learn's output and committed to the repository. Fixtures are the authoritative ground truth for numerical correctness.

**Generation:** A Python script (`scripts/generate_fixtures.py`) runs scikit-learn with fixed random seeds on a curated set of inputs and writes the results to JSON files under `fixtures/`. This script must be re-run and fixtures re-committed whenever scikit-learn releases a new version that changes numerical behavior.

```python
# Example fixture generation for LogisticRegression
import numpy as np, json
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

rng = np.random.RandomState(42)
X, y = make_classification(n_samples=200, n_features=10, random_state=42)
model = LogisticRegression(C=1.0, max_iter=1000, random_state=42).fit(X, y)

fixture = {
    "sklearn_version": sklearn.__version__,
    "X": X.tolist(),
    "y": y.tolist(),
    "coef": model.coef_.tolist(),
    "intercept": model.intercept_.tolist(),
    "predictions": model.predict(X).tolist(),
    "probabilities": model.predict_proba(X).tolist(),
}
with open("fixtures/logistic_regression_basic.json", "w") as f:
    json.dump(fixture, f)
```

```rust
// Corresponding Rust oracle test
#[test]
fn logistic_regression_matches_sklearn_basic() {
    let fixture: Fixture = load_fixture("logistic_regression_basic.json");

    let model = LogisticRegression::new()
        .C(1.0)
        .max_iter(1000)
        .random_state(42)
        .fit(&fixture.X, &fixture.y)
        .unwrap();

    assert_ulps_eq!(model.coef(), fixture.coef, max_ulps = 4);
    assert_ulps_eq!(model.intercept(), fixture.intercept, max_ulps = 4);
    assert_eq!(model.predict(&fixture.X).unwrap(), fixture.predictions);
    assert_ulps_eq!(model.predict_proba(&fixture.X).unwrap(), fixture.probabilities, max_ulps = 4);
}
```

**Tolerance standard:** All floating-point comparisons must use ULP-based tolerances, not absolute epsilon. The budget is:

| Output type | Max ULP tolerance |
|---|---|
| Fitted coefficients / weights | 4 ULPs |
| Predictions (class labels) | Exact match |
| Probabilities / scores | 4 ULPs |
| Metrics (accuracy, R², etc.) | 4 ULPs |
| Iterative solver outputs (>1000 iterations) | 10 ULPs |

The rationale for ULP-based comparison: pure absolute epsilon tolerances do not scale correctly across the floating-point range and will produce false passes for large values and false failures for small ones. ULP differences measure how many representable floating-point numbers separate the two values, which is the correct measure of floating-point rounding divergence.

**Required fixture scenarios per algorithm:**
- Standard case (well-conditioned data, default hyperparameters)
- Non-default hyperparameters (at least three distinct configurations)
- Edge case: minimum viable input (e.g. `n_samples = n_features + 1`)
- Edge case: single-class or near-degenerate data (where applicable)
- Edge case: features with zero variance
- Edge case: very large values (`X` scaled to `[1e6, 1e7]`)
- Edge case: very small values (`X` scaled to `[1e-7, 1e-6]`)
- Sparse input equivalent (for all algorithms that accept sparse input)

**CI enforcement:** Fixture tests run on every pull request. A PR that changes algorithm output without a corresponding fixture update must be rejected by CI.

---

### 20.2 Layer 2 — Property-Based Testing for Algorithmic Invariants (REQUIRED for all algorithms)

Property-based tests verify mathematical invariants that must hold for *all valid inputs*, not just the fixture inputs. They use `proptest` to generate randomized inputs and assert logical properties derived from the algorithm's mathematical definition.

Unlike oracle tests, property tests do not require access to scikit-learn — they test correctness against the mathematical specification directly. They catch classes of bugs that oracle tests cannot, including failures on inputs that the fixture author did not anticipate.

**Tooling:** `proptest` crate. Strategies must be defined for each algorithm's valid input domain, with appropriate shrinking so that counterexamples are minimized on failure.

**Required invariants by category:**

*Scalers and preprocessors:*
```rust
// StandardScaler: transformed training data must have zero mean and unit variance
proptest! {
    fn scaler_zero_mean(matrix in valid_matrix_f64()) {
        let (_, t) = StandardScaler::new().fit_transform(&matrix).unwrap();
        let means = t.mean_axis(Axis(0)).unwrap();
        prop_assert!(means.iter().all(|m| m.abs() < 1e-10));
    }

    fn scaler_unit_variance(matrix in valid_matrix_f64()) {
        let (_, t) = StandardScaler::new().fit_transform(&matrix).unwrap();
        let stds = t.std_axis(Axis(0), 0.0);
        prop_assert!(stds.iter().all(|s| (s - 1.0).abs() < 1e-10));
    }

    // fit_transform(X) == fit(X).transform(X) for all valid X
    fn scaler_fit_transform_equivalence(matrix in valid_matrix_f64()) {
        let (fitted, t1) = StandardScaler::new().fit_transform(&matrix).unwrap();
        let t2 = fitted.transform(&matrix).unwrap();
        prop_assert_ulps_eq!(t1, t2, max_ulps = 1);
    }
}
```

*Classifiers:*
```rust
// predict_proba rows must sum to 1.0
fn classifier_proba_sums_to_one(X in valid_matrix(), y in valid_labels()) { ... }

// classes() must be sorted and contain exactly the unique labels seen during fit
fn classifier_classes_sorted_and_complete(X in valid_matrix(), y in valid_labels()) { ... }

// predict(X) == argmax(predict_proba(X)) for all probabilistic classifiers
fn predict_consistent_with_proba(X in valid_matrix(), y in valid_labels()) { ... }

// Re-fitting with identical data and seed must produce identical output
fn classifier_deterministic_with_seed(X in valid_matrix(), y in valid_labels()) { ... }
```

*Regressors:*
```rust
// R² on training data for a sufficiently expressive model must be ≥ 0.0
// predict() output shape must match input n_samples
// Predicting a constant target must produce zero residuals for LinearRegression
```

*Clustering:*
```rust
// Every sample must be assigned to its nearest centroid (k-Means hard invariant)
fn kmeans_nearest_centroid_assignment(X in valid_matrix(), k in 2usize..=10) {
    let model = KMeans::new(k).fit(&X).unwrap();
    let labels = model.labels();
    let centroids = model.cluster_centers();
    for (i, &label) in labels.iter().enumerate() {
        let sample = X.row(i);
        let assigned_dist = dist(&sample, &centroids.row(label));
        for (j, centroid) in centroids.rows().into_iter().enumerate() {
            prop_assert!(assigned_dist <= dist(&sample, &centroid) + 1e-10);
        }
    }
}

// Inertia must be non-negative
// n_iter_ must be ≥ 1
```

*Dimensionality reduction:*
```rust
// PCA components must be orthonormal: components @ components.T ≈ I
fn pca_components_orthonormal(X in valid_matrix(), n_comp in 1usize..=5) { ... }

// Explained variance ratios must be non-negative and sum to ≤ 1.0
fn pca_explained_variance_valid(X in valid_matrix(), n_comp in 1usize..=5) { ... }

// Reconstruction error must decrease monotonically as n_components increases
fn pca_reconstruction_monotone(X in valid_matrix()) { ... }
```

*Pipelines:*
```rust
// Pipeline.fit(X, y).predict(X) must equal Pipeline.fit_predict(X, y)
// A pipeline with a no-op transformer inserted must produce identical predictions
// fit_transform on a pipeline must equal fit then transform
```

*Metrics:*
```rust
// accuracy_score must be in [0.0, 1.0] for all inputs
// confusion_matrix rows must sum to the count of each true label
// r2_score on training data of LinearRegression must be ≥ r2_score of a constant predictor
```

**Coverage requirement:** Every public algorithm must have a minimum of 8 distinct property tests covering its core mathematical invariants. PRs introducing a new algorithm without the full property test suite must be rejected.

---

### 20.3 Layer 3 — Statistical Equivalence Benchmarking (REQUIRED before any stable release)

Oracle tests prove numerical closeness on fixture data. Statistical benchmarking proves that predictive quality is equivalent on real, diverse datasets. This is a separate concern — an implementation could match all fixture outputs to within 1 ULP and still underperform scikit-learn on novel data due to a subtle algorithmic difference in solver convergence or initialization.

**Infrastructure:** A Python harness (`benchmarks/statistical_equivalence.py`) runs both implementations on the same datasets and applies Welch's t-test to cross-validated scores. A Rust binary (`benchmarks/ferrolearn_bench`) outputs scores as JSON for the harness to consume.

```python
import scipy.stats, json, subprocess
from sklearn.model_selection import cross_val_score

DATASETS = [
    # scikit-learn built-ins
    "iris", "digits", "wine", "breast_cancer", "diabetes",
    # OpenML benchmarks — diverse domains and sizes
    "credit-g",       # credit scoring, 1000 samples, 20 features
    "adult",          # income classification, 48k samples
    "covertype",      # multiclass, 500k samples
    "bank-marketing", # imbalanced binary classification
    "california_housing",  # regression
    "ames_housing",        # regression, high-dimensional
]

ALGORITHMS = [
    "logistic_regression", "random_forest", "decision_tree",
    "knn", "naive_bayes_gaussian", "sgd_classifier",
    "linear_regression", "ridge", "lasso", "svr",
    "kmeans",  # evaluated by silhouette score
    "pca",     # evaluated by reconstruction error on held-out data
]

def run_equivalence_test(algo, dataset, cv=10):
    sklearn_scores = cross_val_score(get_sklearn(algo), X, y, cv=cv, scoring=metric)
    
    result = subprocess.run(
        ["./target/release/ferrolearn_bench", algo, dataset, str(cv)],
        capture_output=True
    )
    ferrolearn_scores = json.loads(result.stdout)["scores"]

    t_stat, p_value = scipy.stats.ttest_ind(sklearn_scores, ferrolearn_scores)

    # FAIL: ferrolearn is statistically significantly worse (one-sided, α=0.05)
    assert not (p_value < 0.05 and mean(ferrolearn_scores) < mean(sklearn_scores)), (
        f"FAIL: {algo} on {dataset}: ferrolearn significantly worse "
        f"(ferrolearn={mean(ferrolearn_scores):.4f}, sklearn={mean(sklearn_scores):.4f}, p={p_value:.4f})"
    )
    
    # WARN: ferrolearn is meaningfully worse but not statistically significant
    if mean(ferrolearn_scores) < mean(sklearn_scores) - 0.005:
        print(f"WARN: {algo} on {dataset}: ferrolearn mean {mean(ferrolearn_scores):.4f} "
              f"vs sklearn {mean(sklearn_scores):.4f} — investigate")
```

**Hard requirements:**
- The benchmark suite must cover a minimum of 10 datasets and all P0 algorithms before a 1.0 release
- No P0 algorithm may produce a statistically significantly worse result (Welch's t-test, α = 0.05, one-sided) than scikit-learn on any benchmark dataset
- Results of each benchmark run must be committed as a machine-readable artifact (`benchmarks/results/YYYY-MM-DD.json`) so regressions can be detected over time
- Benchmarks must run on every release candidate and on any PR that touches algorithm implementation code

---

### 20.4 Layer 4 — Algorithm Equivalence Documentation (REQUIRED for all algorithms)

For each algorithm, a structured documentation block must be maintained alongside the implementation that proves the Rust implementation uses the same algorithm variant as scikit-learn, not merely a similar one. This is the paper trail that converts "it seems to match" into "it implements the same procedure."

The documentation must be kept in `docs/algorithm_equivalence/` as one Markdown file per algorithm. Each file must contain:

```markdown
# Algorithm Equivalence: Logistic Regression (L-BFGS solver)

## scikit-learn Reference
- File: `sklearn/linear_model/_logistic.py`
- Commit: `abc1234` (scikit-learn 1.5.0)
- Relevant functions: `_logistic_loss_and_grad`, `_fit_liblinear`

## Mathematical Specification
Loss function: L(w) = -Σ [y_i log(p_i) + (1-y_i) log(1-p_i)] + (C⁻¹/2) ||w||²
Gradient: ∇L(w) = Xᵀ(p - y) + C⁻¹ w
where p_i = σ(X_i · w + b)

## Equivalence Claims

| Component         | sklearn behavior                         | ferrolearn behavior        | Status |
|-------------------|------------------------------------------|----------------------------|--------|
| Loss function     | Log-loss with L2 penalty                 | Identical formulation      | ✓      |
| Gradient          | Analytical gradient of cross-entropy+L2  | Identical                  | ✓      |
| Solver            | L-BFGS, m=10 history                    | L-BFGS, m=10 history       | ✓      |
| Line search       | Wolfe conditions                         | Wolfe conditions           | ✓      |
| Convergence check | |grad|_inf < tol                         | |grad|_inf < tol           | ✓      |
| Multiclass        | OvR by default, multinomial w/ lbfgs    | Identical dispatch         | ✓      |

## Known Numerical Differences
- BLAS implementation differences: max observed 3 ULPs in coef_ on benchmark suite
- FMA instruction availability: CPU-dependent, max 1 ULP difference
- These are unavoidable consequences of floating-point non-associativity, not algorithmic differences

## Fixture Coverage
- `fixtures/logistic_regression_basic.json` — standard case
- `fixtures/logistic_regression_c0.01.json` — strong regularization
- `fixtures/logistic_regression_multinomial.json` — multiclass softmax
- `fixtures/logistic_regression_sparse.json` — CSR input
- `fixtures/logistic_regression_illconditioned.json` — near-singular X
```

**Enforcement:** No algorithm PR may be merged without the corresponding equivalence document. The document must be reviewed by a maintainer with numerical analysis familiarity before merge.

---

### 20.5 Layer 5 — Fuzz Testing for Numerical Robustness (REQUIRED for all public APIs)

Fuzzing verifies that no combination of inputs — however malformed, adversarial, or degenerate — causes a panic, undefined behavior, silent NaN propagation, or infinite loop. This is distinct from correctness: the fuzz target's contract is that the library either returns a valid `Result::Ok` or a well-typed `Result::Err`, never panics or hangs.

**Tooling:** `cargo-fuzz` with `libFuzzer` backend. Fuzz targets live in `fuzz/fuzz_targets/`.

```rust
// fuzz/fuzz_targets/standard_scaler.rs
#![no_main]
use libfuzzer_sys::fuzz_target;
use ferrolearn::preprocess::StandardScaler;

fuzz_target!(|data: &[u8]| {
    if let Some(matrix) = Matrix::from_fuzz_bytes(data) {
        // Contract: must never panic. May return Err.
        let result = StandardScaler::new().fit(&matrix);
        if let Ok(scaler) = result {
            let _ = scaler.transform(&matrix);
        }
    }
});
```

**Hard requirements:**
- Every public-facing `fit`, `transform`, `predict`, and `fit_transform` function must have a fuzz target
- The fuzz corpus must be run for a minimum of 24 CPU-hours before any release
- Any panic discovered by fuzzing is a P0 bug that blocks release regardless of input validity
- NaN propagation through any public function on non-NaN input is a P0 bug
- Infinite loops or hangs (detectable via timeout) are P0 bugs
- The fuzz corpus (seed inputs) must be committed to `fuzz/corpus/` and grown over time
- Crashes found by fuzzing must produce a minimized reproducer and a regression test before the fix is merged

**Specific adversarial inputs that must be covered by the fuzz seed corpus:**
- All-zero matrix
- All-NaN matrix (must return `Err`, not panic)
- Matrix with a single row
- Matrix with a single column
- Matrix where `n_samples < n_features`
- Extremely large values (`f64::MAX`)
- Extremely small values (`f64::MIN_POSITIVE`)
- Values near overflow boundaries
- Matrices with duplicate rows
- Perfectly collinear feature matrices

---

### 20.6 Layer 6 — Formal Verification of Core Primitives (REQUIRED for metrics and data structures; RECOMMENDED for algorithm kernels)

Full formal verification of all ML algorithms is not yet practical with current tooling. However, specific components are tractable for formal verification today and must be verified before the 1.0 release. Formal verification here means machine-checked proofs, not just testing.

**Mandatory formally verified components:**

*Metric functions* — All metric functions in `ferrolearn-metrics` are pure functions with unambiguous mathematical definitions. They must be verified using `Prusti` (the Rust formal verifier) with pre/postcondition annotations:

```rust
#[requires(y_true.len() == y_pred.len())]
#[requires(y_true.len() > 0)]
#[ensures(0.0 <= result && result <= 1.0)]
pub fn accuracy_score(y_true: &[usize], y_pred: &[usize]) -> f64 {
    let correct = y_true.iter().zip(y_pred).filter(|(a, b)| a == b).count();
    correct as f64 / y_true.len() as f64
}
```

*Sparse matrix structural invariants* — The `CsrMatrix<T>` type must carry Prusti-verified invariants proving that after any public constructor or mutation:
- `indptr.len() == n_rows + 1`
- `indptr` is monotonically non-decreasing
- `indices[indptr[i]..indptr[i+1]]` are all in `0..n_cols` for every row `i`
- `data.len() == indices.len()`
- No index appears twice within the same row's slice

*Type-system proofs (already enforced by the compiler):*
The compile-time guarantee that `predict()` cannot be called on an unfitted model — enforced by the `Fit`/`Predict` trait split described in Section 4.1 — is a formal proof carried by Rust's type checker. This must be documented explicitly as a correctness guarantee in the API documentation, not treated as an implementation detail.

**Recommended formally verified components (stretch goals):**

- SGD update step: prove that a single gradient step reduces the objective by at least the theoretically guaranteed amount for a given learning rate and Lipschitz constant
- Convergence criterion: prove that the L-BFGS convergence check `|grad|_inf < tol` is equivalent to the stated stopping condition
- `train_test_split`: prove that the returned index sets are disjoint and their union equals `0..n_samples`

**Tooling:** Prusti for Rust-native proofs. For algorithms where Prusti's current capabilities are insufficient, the mathematical core may be specified in Lean 4 as a reference specification, with the Rust implementation proven equivalent via fixture tests against the Lean-evaluated reference.

---

### 20.7 Correctness Verification Summary

The following table defines what must be in place before an algorithm can ship at each stability level:

| Requirement | Alpha | Beta | Stable 1.0 |
|---|---|---|---|
| Oracle fixtures (standard case) | ✓ | ✓ | ✓ |
| Oracle fixtures (all edge cases) | | ✓ | ✓ |
| Property-based invariant tests (≥ 8) | | ✓ | ✓ |
| Algorithm equivalence document | | ✓ | ✓ |
| Statistical benchmark suite | | | ✓ |
| Fuzz target exists | ✓ | ✓ | ✓ |
| Fuzz corpus run for ≥ 24 CPU-hours | | | ✓ |
| Formal verification (metrics / data structures) | | | ✓ |
| No open P0 correctness bugs | | ✓ | ✓ |

**Definition of a P0 correctness bug:** any condition where the library produces a result that differs from the algorithm's mathematical specification by more than the allowed ULP budget, panics on valid input, propagates NaN on non-NaN input, hangs, or produces output that fails a required property invariant.

No release may proceed with an open P0 correctness bug under any circumstances.

---

## 21. Prior Art and Relationships

| Crate | Relationship |
|---|---|
| `linfa` | Closest existing attempt; provides some algorithms but lacks preprocessing, sparse support, cross-validation, and a unified pipeline. `ferrolearn` should consider `linfa` for algorithm contributions or offer migration paths. |
| `ndarray` | Primary dense array type; `ferrolearn` depends on it |
| `faer` | High-performance linear algebra; used for SVD, solvers |
| `sprs` | Existing sparse matrix crate; `ferrolearn-sparse` may wrap or replace it |
| `polars` | DataFrame integration via feature flag |
| `smartcore` | Another incomplete scikit-learn attempt; referenced for API decisions |
| `tract` | ONNX runtime; target for exported models |
| `rayon` | All CPU parallelism |
| `serde` | Serialization of all model types |

---

*This document is a living specification. Open questions and tradeoffs should be resolved in tracked issues before implementation begins.*
