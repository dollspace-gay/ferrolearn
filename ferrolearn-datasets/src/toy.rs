//! Classic toy datasets embedded as compiled-in CSV data.
//!
//! Each loader parses a CSV at load time and returns arrays suitable for
//! use with ferrolearn estimators. Classification datasets return
//! `(Array2<F>, Array1<usize>)`, regression datasets return
//! `(Array2<F>, Array1<F>)`.

use ferrolearn_core::FerroError;
use ndarray::{Array1, Array2};
use num_traits::Float;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

// ---------------------------------------------------------------------------
// Embedded CSV data
// ---------------------------------------------------------------------------

/// The classic Iris dataset (150 samples × 4 features, 3 classes).
const IRIS_CSV: &str = include_str!("../data/iris.csv");

/// The Wine recognition dataset (178 samples × 13 features, 3 classes).
///
/// Copied verbatim from scikit-learn 1.5.2
/// (`sklearn/datasets/data/wine_data.csv`). Row 0 is metadata
/// `178,13,class_0,class_1,class_2`; rows 1..179 are 13 features + integer label.
const WINE_CSV: &str = include_str!("../data/wine_data.csv");

/// The Breast Cancer Wisconsin (Diagnostic) dataset (569 samples × 30 features).
///
/// Copied verbatim from scikit-learn 1.5.2
/// (`sklearn/datasets/data/breast_cancer.csv`). Row 0 is metadata
/// `569,30,malignant,benign`; rows 1..570 are 30 features + integer label
/// (0 = malignant, 1 = benign).
const BREAST_CANCER_CSV: &str = include_str!("../data/breast_cancer.csv");

/// The raw (unscaled) Diabetes feature matrix (442 samples × 10 features).
///
/// Decompressed from scikit-learn 1.5.2
/// (`sklearn/datasets/data/diabetes_data_raw.csv.gz`). Whitespace-delimited,
/// no header, 10 columns per row.
const DIABETES_DATA_CSV: &str = include_str!("../data/diabetes_data_raw.csv");

/// The Diabetes regression target (442 samples).
///
/// Decompressed from scikit-learn 1.5.2
/// (`sklearn/datasets/data/diabetes_target.csv.gz`). One float per line.
const DIABETES_TARGET_CSV: &str = include_str!("../data/diabetes_target.csv");

/// The Optical Recognition of Handwritten Digits dataset
/// (1797 samples × 64 features, 10 classes).
///
/// Decompressed from scikit-learn 1.5.2 (`sklearn/datasets/data/digits.csv.gz`).
/// Comma-delimited, no header, 65 columns per row (64 pixel features + integer
/// digit label 0..=9).
const DIGITS_CSV: &str = include_str!("../data/digits.csv");

// ---------------------------------------------------------------------------
// Internal parsing helpers
// ---------------------------------------------------------------------------

/// Parse a CSV string where the last column is an integer class label.
///
/// The first row is a header and is skipped.
fn parse_classification_csv<F>(
    csv: &str,
    expected_samples: usize,
    expected_features: usize,
) -> Result<(Array2<F>, Array1<usize>), FerroError>
where
    F: Float,
{
    let mut x_data: Vec<F> = Vec::with_capacity(expected_samples * expected_features);
    let mut y_data: Vec<usize> = Vec::with_capacity(expected_samples);

    for (line_idx, line) in csv.lines().enumerate() {
        // Skip header row.
        if line_idx == 0 {
            continue;
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() < 2 {
            return Err(FerroError::SerdeError {
                message: format!(
                    "line {line_idx}: expected at least 2 fields, got {}",
                    fields.len()
                ),
            });
        }

        let n_feat = fields.len() - 1;
        // Parse feature columns.
        for f in &fields[..n_feat] {
            let val: f64 = f.trim().parse().map_err(|_| FerroError::SerdeError {
                message: format!("line {line_idx}: cannot parse feature '{}'", f.trim()),
            })?;
            x_data.push(F::from(val).ok_or_else(|| FerroError::SerdeError {
                message: format!("line {line_idx}: float conversion failed for '{val}'"),
            })?);
        }
        // Parse label column.
        let label: usize = fields[n_feat]
            .trim()
            .parse()
            .map_err(|_| FerroError::SerdeError {
                message: format!(
                    "line {line_idx}: cannot parse class label '{}'",
                    fields[n_feat].trim()
                ),
            })?;
        y_data.push(label);
    }

    let n_samples = y_data.len();
    let n_features = x_data.len().checked_div(n_samples).unwrap_or(0);

    let x = Array2::from_shape_vec((n_samples, n_features), x_data).map_err(|e| {
        FerroError::SerdeError {
            message: format!("cannot reshape X: {e}"),
        }
    })?;
    let y = Array1::from_vec(y_data);

    // Validate shape expectations.
    if n_samples != expected_samples {
        return Err(FerroError::ShapeMismatch {
            expected: vec![expected_samples],
            actual: vec![n_samples],
            context: "toy dataset sample count".into(),
        });
    }
    if n_features != expected_features {
        return Err(FerroError::ShapeMismatch {
            expected: vec![expected_features],
            actual: vec![n_features],
            context: "toy dataset feature count".into(),
        });
    }

    Ok((x, y))
}

// ---------------------------------------------------------------------------
// Internal parsing helpers (headerless / gzip-sourced datasets)
// ---------------------------------------------------------------------------

/// Parse a headerless, comma-delimited CSV where the last column is an integer
/// class label and every preceding column is a `float64` feature.
///
/// Mirrors the `digits.csv` layout decompressed from
/// `sklearn/datasets/data/digits.csv.gz` (64 pixel features + label per row).
fn parse_headerless_classification_csv<F>(
    csv: &str,
    expected_samples: usize,
    expected_features: usize,
) -> Result<(Array2<F>, Array1<usize>), FerroError>
where
    F: Float,
{
    let mut x_data: Vec<F> = Vec::with_capacity(expected_samples * expected_features);
    let mut y_data: Vec<usize> = Vec::with_capacity(expected_samples);

    for (line_idx, line) in csv.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() != expected_features + 1 {
            return Err(FerroError::SerdeError {
                message: format!(
                    "line {line_idx}: expected {} fields, got {}",
                    expected_features + 1,
                    fields.len()
                ),
            });
        }

        for f in &fields[..expected_features] {
            let val: f64 = f.trim().parse().map_err(|_| FerroError::SerdeError {
                message: format!("line {line_idx}: cannot parse feature '{}'", f.trim()),
            })?;
            x_data.push(F::from(val).ok_or_else(|| FerroError::SerdeError {
                message: format!("line {line_idx}: float conversion failed for '{val}'"),
            })?);
        }
        let label: usize =
            fields[expected_features]
                .trim()
                .parse()
                .map_err(|_| FerroError::SerdeError {
                    message: format!(
                        "line {line_idx}: cannot parse class label '{}'",
                        fields[expected_features].trim()
                    ),
                })?;
        y_data.push(label);
    }

    let n_samples = y_data.len();
    let x = Array2::from_shape_vec((n_samples, expected_features), x_data).map_err(|e| {
        FerroError::SerdeError {
            message: format!("cannot reshape X: {e}"),
        }
    })?;
    let y = Array1::from_vec(y_data);

    if n_samples != expected_samples {
        return Err(FerroError::ShapeMismatch {
            expected: vec![expected_samples],
            actual: vec![n_samples],
            context: "toy dataset sample count".into(),
        });
    }

    Ok((x, y))
}

/// Parse a whitespace-delimited matrix of `float64` values into an `Array2<F>`.
///
/// Mirrors `np.loadtxt` over the decompressed `diabetes_data_raw.csv`
/// (space-separated, no header, `expected_features` columns per row).
fn parse_whitespace_matrix<F>(
    csv: &str,
    expected_samples: usize,
    expected_features: usize,
) -> Result<Array2<F>, FerroError>
where
    F: Float,
{
    let mut data: Vec<F> = Vec::with_capacity(expected_samples * expected_features);
    let mut n_rows = 0usize;

    for (line_idx, line) in csv.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() != expected_features {
            return Err(FerroError::SerdeError {
                message: format!(
                    "line {line_idx}: expected {expected_features} fields, got {}",
                    fields.len()
                ),
            });
        }
        for f in &fields {
            let val: f64 = f.parse().map_err(|_| FerroError::SerdeError {
                message: format!("line {line_idx}: cannot parse value '{f}'"),
            })?;
            data.push(F::from(val).ok_or_else(|| FerroError::SerdeError {
                message: format!("line {line_idx}: float conversion failed for '{val}'"),
            })?);
        }
        n_rows += 1;
    }

    let x = Array2::from_shape_vec((n_rows, expected_features), data).map_err(|e| {
        FerroError::SerdeError {
            message: format!("cannot reshape matrix: {e}"),
        }
    })?;

    if n_rows != expected_samples {
        return Err(FerroError::ShapeMismatch {
            expected: vec![expected_samples],
            actual: vec![n_rows],
            context: "diabetes data sample count".into(),
        });
    }

    Ok(x)
}

/// Parse a single-column CSV of `float64` values into an `Array1<F>`.
///
/// Mirrors `np.loadtxt` over the decompressed `diabetes_target.csv`.
fn parse_float_column<F>(csv: &str, expected_samples: usize) -> Result<Array1<F>, FerroError>
where
    F: Float,
{
    let mut data: Vec<F> = Vec::with_capacity(expected_samples);
    for (line_idx, line) in csv.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let val: f64 = line.parse().map_err(|_| FerroError::SerdeError {
            message: format!("line {line_idx}: cannot parse value '{line}'"),
        })?;
        data.push(F::from(val).ok_or_else(|| FerroError::SerdeError {
            message: format!("line {line_idx}: float conversion failed for '{val}'"),
        })?);
    }

    if data.len() != expected_samples {
        return Err(FerroError::ShapeMismatch {
            expected: vec![expected_samples],
            actual: vec![data.len()],
            context: "diabetes target sample count".into(),
        });
    }

    Ok(Array1::from_vec(data))
}

/// Apply scikit-learn's default diabetes feature scaling, in place.
///
/// Mirrors `sklearn/datasets/_base.py:1132-1134`:
/// ```text
/// data = scale(data, copy=False)      # per-column (x - mean) / population_std
/// data /= data.shape[0] ** 0.5
/// ```
/// i.e. for each column `j`, `scaled[i,j] = (raw[i,j] - mean_j) / std_j / sqrt(n)`
/// where `std_j` is the population standard deviation (ddof=0). Statistics are
/// accumulated in `f64` for numerical fidelity, then converted back to `F`.
fn scale_diabetes_columns<F>(x: &mut Array2<F>) -> Result<(), FerroError>
where
    F: Float,
{
    let n_samples = x.nrows();
    let n_features = x.ncols();
    if n_samples == 0 {
        return Ok(());
    }
    let n = n_samples as f64;
    let sqrt_n = n.sqrt();

    for j in 0..n_features {
        // Mean of column j (computed in f64).
        let mut sum = 0.0_f64;
        for i in 0..n_samples {
            sum += x[[i, j]].to_f64().ok_or_else(|| FerroError::SerdeError {
                message: format!("diabetes scaling: f64 conversion failed at [{i},{j}]"),
            })?;
        }
        let mean = sum / n;

        // Population variance (ddof=0).
        let mut var_sum = 0.0_f64;
        for i in 0..n_samples {
            let v = x[[i, j]].to_f64().ok_or_else(|| FerroError::SerdeError {
                message: format!("diabetes scaling: f64 conversion failed at [{i},{j}]"),
            })?;
            let d = v - mean;
            var_sum += d * d;
        }
        let std = (var_sum / n).sqrt();
        // sklearn's `scale` leaves constant columns (std == 0) unscaled.
        let denom = if std == 0.0 { 1.0 } else { std };

        for i in 0..n_samples {
            let v = x[[i, j]].to_f64().ok_or_else(|| FerroError::SerdeError {
                message: format!("diabetes scaling: f64 conversion failed at [{i},{j}]"),
            })?;
            let scaled = (v - mean) / denom / sqrt_n;
            x[[i, j]] = F::from(scaled).ok_or_else(|| FerroError::SerdeError {
                message: format!("diabetes scaling: float conversion failed for '{scaled}'"),
            })?;
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Public loaders
// ---------------------------------------------------------------------------

/// Load the Iris dataset.
///
/// Returns `(X, y)` where `X` has shape `(150, 4)` and `y` has 3 unique
/// class labels (0 = setosa, 1 = versicolor, 2 = virginica).
///
/// The data is the canonical UCI Iris dataset, embedded at compile time.
///
/// # Errors
///
/// Returns [`FerroError::SerdeError`] if the embedded CSV is malformed.
///
/// # Examples
///
/// ```rust
/// let (x, y) = ferrolearn_datasets::load_iris::<f64>().unwrap();
/// assert_eq!(x.shape(), &[150, 4]);
/// assert_eq!(y.len(), 150);
/// ```
pub fn load_iris<F>() -> Result<(Array2<F>, Array1<usize>), FerroError>
where
    F: Float,
{
    parse_classification_csv(IRIS_CSV, 150, 4)
}

/// Load the Wine recognition dataset.
///
/// Returns `(X, y)` where `X` has shape `(178, 13)` and `y` has 3 unique
/// class labels (0/1/2, the three cultivars). The data is the real
/// UCI Wine dataset embedded from scikit-learn's
/// `sklearn/datasets/data/wine_data.csv`, matching
/// `sklearn.datasets.load_wine(return_X_y=True)` element-wise.
///
/// # Errors
///
/// Returns [`FerroError::SerdeError`] or [`FerroError::ShapeMismatch`] if the
/// embedded CSV is malformed.
///
/// # Examples
///
/// ```rust
/// let (x, y) = ferrolearn_datasets::load_wine::<f64>().unwrap();
/// assert_eq!(x.shape(), &[178, 13]);
/// assert_eq!(y.len(), 178);
/// // First sample (sklearn oracle): real measurements, class 0.
/// assert!((x[[0, 0]] - 14.23_f64).abs() < 1e-6);
/// assert_eq!(y[0], 0);
/// ```
pub fn load_wine<F>() -> Result<(Array2<F>, Array1<usize>), FerroError>
where
    F: Float,
{
    parse_classification_csv(WINE_CSV, 178, 13)
}

/// Load the Breast Cancer Wisconsin (Diagnostic) dataset.
///
/// Returns `(X, y)` where `X` has shape `(569, 30)` and `y` has 2 unique
/// class labels (0 = malignant, 1 = benign). The data is the real WDBC
/// dataset embedded from scikit-learn's
/// `sklearn/datasets/data/breast_cancer.csv`, matching
/// `sklearn.datasets.load_breast_cancer(return_X_y=True)` element-wise.
///
/// # Errors
///
/// Returns [`FerroError::SerdeError`] or [`FerroError::ShapeMismatch`] if the
/// embedded CSV is malformed.
///
/// # Examples
///
/// ```rust
/// let (x, y) = ferrolearn_datasets::load_breast_cancer::<f64>().unwrap();
/// assert_eq!(x.shape(), &[569, 30]);
/// assert_eq!(y.len(), 569);
/// // First sample (sklearn oracle): real measurements, malignant (class 0).
/// assert!((x[[0, 0]] - 17.99_f64).abs() < 1e-6);
/// assert_eq!(y[0], 0);
/// ```
pub fn load_breast_cancer<F>() -> Result<(Array2<F>, Array1<usize>), FerroError>
where
    F: Float,
{
    parse_classification_csv(BREAST_CANCER_CSV, 569, 30)
}

/// Load the Diabetes dataset (regression), with sklearn's default scaling.
///
/// Returns `(X, y)` where `X` has shape `(442, 10)` and `y` is the continuous
/// disease-progression target (integers 25–346, returned as `F`). The features
/// are the real raw measurements (embedded from
/// `sklearn/datasets/data/diabetes_data_raw.csv.gz`) transformed by sklearn's
/// default `scaled=True` path (`_base.py:1132-1134`): each column is
/// mean-centered, divided by its population standard deviation, then divided by
/// `sqrt(n_samples)`. The target is the raw value (not scaled). Matches
/// `sklearn.datasets.load_diabetes(return_X_y=True)` element-wise.
///
/// # Errors
///
/// Returns [`FerroError::SerdeError`] or [`FerroError::ShapeMismatch`] if the
/// embedded data is malformed.
///
/// # Examples
///
/// ```rust
/// let (x, y) = ferrolearn_datasets::load_diabetes::<f64>().unwrap();
/// assert_eq!(x.shape(), &[442, 10]);
/// assert_eq!(y.len(), 442);
/// // sklearn oracle (default scaled=True): X[0,0] ≈ 0.03807591, y[0] == 151.
/// assert!((x[[0, 0]] - 0.038075906433423026_f64).abs() < 1e-9);
/// assert!((y[0] - 151.0_f64).abs() < 1e-9);
/// ```
pub fn load_diabetes<F>() -> Result<(Array2<F>, Array1<F>), FerroError>
where
    F: Float,
{
    let mut x = parse_whitespace_matrix::<F>(DIABETES_DATA_CSV, 442, 10)?;
    scale_diabetes_columns(&mut x)?;
    let y = parse_float_column::<F>(DIABETES_TARGET_CSV, 442)?;
    Ok((x, y))
}

/// Load the Optical Recognition of Handwritten Digits dataset.
///
/// Returns `(X, y)` where `X` has shape `(1797, 64)` (8×8 pixel images flattened
/// to 64 features) and `y` has 10 unique class labels (digits 0–9). The data is
/// the real dataset embedded from `sklearn/datasets/data/digits.csv.gz`,
/// matching `sklearn.datasets.load_digits(return_X_y=True)` element-wise (the
/// `n_class=10` default — all 1797 samples).
///
/// The upstream `n_class` parameter (which trims to the first `n_class` digit
/// classes) is not yet exposed; this no-argument loader returns the full default
/// (`n_class=10`).
///
/// # Errors
///
/// Returns [`FerroError::SerdeError`] or [`FerroError::ShapeMismatch`] if the
/// embedded CSV is malformed.
///
/// # Examples
///
/// ```rust
/// let (x, y) = ferrolearn_datasets::load_digits::<f64>().unwrap();
/// assert_eq!(x.shape(), &[1797, 64]);
/// assert_eq!(y.len(), 1797);
/// // First sample (sklearn oracle): X[0,:3] == [0, 0, 5].
/// assert!((x[[0, 2]] - 5.0_f64).abs() < 1e-9);
/// ```
pub fn load_digits<F>() -> Result<(Array2<F>, Array1<usize>), FerroError>
where
    F: Float,
{
    parse_headerless_classification_csv(DIGITS_CSV, 1797, 64)
}

// ---------------------------------------------------------------------------
// Linnerud dataset (embedded)
// ---------------------------------------------------------------------------

/// Linnerud exercise features: Chins, Situps, Jumps (20 samples x 3 features).
const LINNERUD_FEATURES: [[f64; 3]; 20] = [
    [5.0, 162.0, 60.0],
    [2.0, 110.0, 60.0],
    [12.0, 101.0, 101.0],
    [12.0, 105.0, 37.0],
    [13.0, 155.0, 58.0],
    [4.0, 101.0, 42.0],
    [8.0, 101.0, 38.0],
    [6.0, 125.0, 40.0],
    [15.0, 200.0, 40.0],
    [17.0, 251.0, 250.0],
    [17.0, 120.0, 38.0],
    [13.0, 210.0, 115.0],
    [14.0, 215.0, 105.0],
    [1.0, 50.0, 50.0],
    [6.0, 70.0, 31.0],
    [12.0, 210.0, 120.0],
    [4.0, 60.0, 25.0],
    [11.0, 230.0, 80.0],
    [15.0, 225.0, 73.0],
    [2.0, 110.0, 43.0],
];

/// Linnerud physiological targets: Weight, Waist, Pulse (20 samples x 3 targets).
const LINNERUD_TARGETS: [[f64; 3]; 20] = [
    [191.0, 36.0, 50.0],
    [189.0, 37.0, 52.0],
    [193.0, 38.0, 58.0],
    [162.0, 35.0, 62.0],
    [189.0, 35.0, 46.0],
    [182.0, 36.0, 56.0],
    [211.0, 38.0, 56.0],
    [167.0, 34.0, 60.0],
    [176.0, 31.0, 74.0],
    [154.0, 33.0, 56.0],
    [169.0, 34.0, 50.0],
    [166.0, 33.0, 52.0],
    [154.0, 34.0, 64.0],
    [247.0, 46.0, 50.0],
    [193.0, 36.0, 46.0],
    [202.0, 37.0, 62.0],
    [176.0, 37.0, 54.0],
    [157.0, 32.0, 52.0],
    [156.0, 33.0, 54.0],
    [138.0, 33.0, 68.0],
];

/// Load the Linnerud physical exercise dataset.
///
/// A small multi-output regression dataset with 20 samples.
///
/// **Features** (exercise): Chins (pull-ups), Situps, Jumps
/// **Targets** (physiological): Weight, Waist, Pulse
///
/// Both `X` and `y` are returned as 2-D arrays because this is a
/// multi-output regression task.
///
/// # Errors
///
/// Returns [`FerroError::SerdeError`] if float conversion fails.
///
/// # Examples
///
/// ```rust
/// let (x, y) = ferrolearn_datasets::load_linnerud::<f64>().unwrap();
/// assert_eq!(x.shape(), &[20, 3]);
/// assert_eq!(y.shape(), &[20, 3]);
/// ```
pub fn load_linnerud<F>() -> Result<(Array2<F>, Array2<F>), FerroError>
where
    F: Float + Send + Sync + 'static,
{
    let n_samples = LINNERUD_FEATURES.len();
    let n_feat = 3;
    let n_targets = 3;

    let mut x_data: Vec<F> = Vec::with_capacity(n_samples * n_feat);
    for row in &LINNERUD_FEATURES {
        for &v in row {
            x_data.push(F::from(v).ok_or_else(|| FerroError::SerdeError {
                message: format!("load_linnerud: cannot convert feature value {v}"),
            })?);
        }
    }

    let mut y_data: Vec<F> = Vec::with_capacity(n_samples * n_targets);
    for row in &LINNERUD_TARGETS {
        for &v in row {
            y_data.push(F::from(v).ok_or_else(|| FerroError::SerdeError {
                message: format!("load_linnerud: cannot convert target value {v}"),
            })?);
        }
    }

    let x = Array2::from_shape_vec((n_samples, n_feat), x_data).map_err(|e| {
        FerroError::SerdeError {
            message: format!("load_linnerud: cannot reshape X: {e}"),
        }
    })?;
    let y = Array2::from_shape_vec((n_samples, n_targets), y_data).map_err(|e| {
        FerroError::SerdeError {
            message: format!("load_linnerud: cannot reshape y: {e}"),
        }
    })?;

    Ok((x, y))
}

// ---------------------------------------------------------------------------
// Olivetti faces (synthetic)
// ---------------------------------------------------------------------------

/// Load a synthetic version of the Olivetti Faces dataset.
///
/// Generates 400 synthetic 64x64 "face-like" images (40 classes, 10 per class)
/// using a seeded random generator. This is **not** the real Olivetti dataset
/// but has the same shape and label structure for API compatibility and
/// testing.
///
/// For the real dataset, download from sklearn or the AT&T website.
///
/// # Errors
///
/// Returns [`FerroError::SerdeError`] if float conversion fails.
///
/// # Examples
///
/// ```rust
/// let (x, y) = ferrolearn_datasets::load_olivetti_faces::<f64>().unwrap();
/// assert_eq!(x.shape(), &[400, 4096]);
/// assert_eq!(y.len(), 400);
/// ```
pub fn load_olivetti_faces<F>() -> Result<(Array2<F>, Array1<usize>), FerroError>
where
    F: Float + Send + Sync + 'static,
{
    use rand::Rng;

    const N_CLASSES: usize = 40;
    const SAMPLES_PER_CLASS: usize = 10;
    const N_SAMPLES: usize = N_CLASSES * SAMPLES_PER_CLASS;
    const N_FEATURES: usize = 64 * 64; // 4096

    // Fixed seed for reproducibility.
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xFACE_DA7A);

    // Generate a base pattern per class (40 patterns of 4096 values in [0, 1]).
    let mut base_patterns: Vec<Vec<f64>> = Vec::with_capacity(N_CLASSES);
    for _ in 0..N_CLASSES {
        let pattern: Vec<f64> = (0..N_FEATURES).map(|_| rng.random::<f64>()).collect();
        base_patterns.push(pattern);
    }

    let mut x_data: Vec<F> = Vec::with_capacity(N_SAMPLES * N_FEATURES);
    let mut y_data: Vec<usize> = Vec::with_capacity(N_SAMPLES);

    // Generate samples: each class gets 10 samples = base + small noise.
    for (class, pattern) in base_patterns.iter().enumerate() {
        for _ in 0..SAMPLES_PER_CLASS {
            for &base_val in pattern {
                // Add noise in [-0.05, 0.05] and clamp to [0, 1].
                let noise = (rng.random::<f64>() - 0.5) * 0.1;
                let val = (base_val + noise).clamp(0.0, 1.0);
                x_data.push(F::from(val).ok_or_else(|| FerroError::SerdeError {
                    message: format!("load_olivetti_faces: cannot convert pixel value {val}"),
                })?);
            }
            y_data.push(class);
        }
    }

    let x = Array2::from_shape_vec((N_SAMPLES, N_FEATURES), x_data).map_err(|e| {
        FerroError::SerdeError {
            message: format!("load_olivetti_faces: cannot reshape X: {e}"),
        }
    })?;
    let y = Array1::from_vec(y_data);

    Ok((x, y))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    // --- Iris ---

    #[test]
    fn test_load_iris_shape() {
        let (x, y) = load_iris::<f64>().unwrap();
        assert_eq!(x.shape(), &[150, 4]);
        assert_eq!(y.len(), 150);
    }

    #[test]
    fn test_load_iris_classes() {
        let (_, y) = load_iris::<f64>().unwrap();
        let unique: HashSet<usize> = y.iter().copied().collect();
        assert_eq!(unique.len(), 3, "Iris should have 3 unique classes");
        assert!(unique.contains(&0));
        assert!(unique.contains(&1));
        assert!(unique.contains(&2));
    }

    #[test]
    fn test_load_iris_first_row() {
        let (x, y) = load_iris::<f64>().unwrap();
        // First sample: 5.1, 3.5, 1.4, 0.2, class 0 (setosa)
        let row = x.row(0);
        assert!(
            (row[0] - 5.1_f64).abs() < 1e-9,
            "sepal_length mismatch: {}",
            row[0]
        );
        assert!(
            (row[1] - 3.5_f64).abs() < 1e-9,
            "sepal_width mismatch: {}",
            row[1]
        );
        assert!(
            (row[2] - 1.4_f64).abs() < 1e-9,
            "petal_length mismatch: {}",
            row[2]
        );
        assert!(
            (row[3] - 0.2_f64).abs() < 1e-9,
            "petal_width mismatch: {}",
            row[3]
        );
        assert_eq!(y[0], 0);
    }

    #[test]
    fn test_load_iris_last_row() {
        let (x, y) = load_iris::<f64>().unwrap();
        // Last sample (index 149): 5.9, 3.0, 5.1, 1.8, class 2 (virginica)
        let row = x.row(149);
        assert!(
            (row[0] - 5.9_f64).abs() < 1e-9,
            "sepal_length mismatch: {}",
            row[0]
        );
        assert_eq!(y[149], 2);
    }

    #[test]
    fn test_load_iris_f32() {
        let (x, y) = load_iris::<f32>().unwrap();
        assert_eq!(x.shape(), &[150, 4]);
        assert_eq!(y.len(), 150);
        let row = x.row(0);
        assert!(
            (row[0] - 5.1_f32).abs() < 1e-5,
            "f32 sepal_length mismatch: {}",
            row[0]
        );
    }

    #[test]
    fn test_load_iris_class_balance() {
        let (_, y) = load_iris::<f64>().unwrap();
        // Each class has exactly 50 samples.
        let count_0 = y.iter().filter(|&&c| c == 0).count();
        let count_1 = y.iter().filter(|&&c| c == 1).count();
        let count_2 = y.iter().filter(|&&c| c == 2).count();
        assert_eq!(count_0, 50);
        assert_eq!(count_1, 50);
        assert_eq!(count_2, 50);
    }

    // --- Wine ---

    #[test]
    fn test_load_wine_shape() {
        let (x, y) = load_wine::<f64>().unwrap();
        assert_eq!(x.shape(), &[178, 13]);
        assert_eq!(y.len(), 178);
    }

    #[test]
    fn test_load_wine_classes() {
        let (_, y) = load_wine::<f64>().unwrap();
        let unique: HashSet<usize> = y.iter().copied().collect();
        assert_eq!(unique.len(), 3);
    }

    // --- Breast Cancer ---

    #[test]
    fn test_load_breast_cancer_shape() {
        let (x, y) = load_breast_cancer::<f64>().unwrap();
        assert_eq!(x.shape(), &[569, 30]);
        assert_eq!(y.len(), 569);
    }

    #[test]
    fn test_load_breast_cancer_classes() {
        let (_, y) = load_breast_cancer::<f64>().unwrap();
        let unique: HashSet<usize> = y.iter().copied().collect();
        assert_eq!(unique.len(), 2);
    }

    // --- Diabetes ---

    #[test]
    fn test_load_diabetes_shape() {
        let (x, y) = load_diabetes::<f64>().unwrap();
        assert_eq!(x.shape(), &[442, 10]);
        assert_eq!(y.len(), 442);
    }

    // --- Digits ---

    #[test]
    fn test_load_digits_shape() {
        let (x, y) = load_digits::<f64>().unwrap();
        assert_eq!(x.shape(), &[1797, 64]);
        assert_eq!(y.len(), 1797);
    }

    #[test]
    fn test_load_digits_classes() {
        let (_, y) = load_digits::<f64>().unwrap();
        let unique: HashSet<usize> = y.iter().copied().collect();
        assert_eq!(unique.len(), 10, "Digits should have 10 unique classes");
    }

    // --- Linnerud ---

    #[test]
    fn test_load_linnerud_shape() {
        let (x, y) = load_linnerud::<f64>().unwrap();
        assert_eq!(x.shape(), &[20, 3]);
        assert_eq!(y.shape(), &[20, 3]);
    }

    #[test]
    fn test_load_linnerud_first_row() {
        let (x, y) = load_linnerud::<f64>().unwrap();
        // First sample: features (5, 162, 60), targets (191, 36, 50)
        let feat = x.row(0);
        assert!(
            (feat[0] - 5.0_f64).abs() < 1e-9,
            "chins mismatch: {}",
            feat[0]
        );
        assert!(
            (feat[1] - 162.0_f64).abs() < 1e-9,
            "situps mismatch: {}",
            feat[1]
        );
        assert!(
            (feat[2] - 60.0_f64).abs() < 1e-9,
            "jumps mismatch: {}",
            feat[2]
        );
        let tgt = y.row(0);
        assert!(
            (tgt[0] - 191.0_f64).abs() < 1e-9,
            "weight mismatch: {}",
            tgt[0]
        );
        assert!(
            (tgt[1] - 36.0_f64).abs() < 1e-9,
            "waist mismatch: {}",
            tgt[1]
        );
        assert!(
            (tgt[2] - 50.0_f64).abs() < 1e-9,
            "pulse mismatch: {}",
            tgt[2]
        );
    }

    #[test]
    fn test_load_linnerud_last_row() {
        let (x, y) = load_linnerud::<f64>().unwrap();
        // Last sample (index 19): features (2, 110, 43), targets (138, 33, 68)
        let feat = x.row(19);
        assert!((feat[0] - 2.0_f64).abs() < 1e-9);
        assert!((feat[1] - 110.0_f64).abs() < 1e-9);
        assert!((feat[2] - 43.0_f64).abs() < 1e-9);
        let tgt = y.row(19);
        assert!((tgt[0] - 138.0_f64).abs() < 1e-9);
        assert!((tgt[1] - 33.0_f64).abs() < 1e-9);
        assert!((tgt[2] - 68.0_f64).abs() < 1e-9);
    }

    #[test]
    fn test_load_linnerud_f32() {
        let (x, y) = load_linnerud::<f32>().unwrap();
        assert_eq!(x.shape(), &[20, 3]);
        assert_eq!(y.shape(), &[20, 3]);
        assert!((x[[0, 0]] - 5.0_f32).abs() < 1e-5);
    }

    // --- Olivetti Faces ---

    #[test]
    fn test_load_olivetti_faces_shape() {
        let (x, y) = load_olivetti_faces::<f64>().unwrap();
        assert_eq!(x.shape(), &[400, 4096]);
        assert_eq!(y.len(), 400);
    }

    #[test]
    fn test_load_olivetti_faces_labels() {
        let (_, y) = load_olivetti_faces::<f64>().unwrap();
        let unique: HashSet<usize> = y.iter().copied().collect();
        assert_eq!(unique.len(), 40, "Olivetti should have 40 unique classes");
        // Each class should have exactly 10 samples.
        for class in 0..40 {
            let count = y.iter().filter(|&&c| c == class).count();
            assert_eq!(
                count, 10,
                "class {class} should have 10 samples, got {count}"
            );
        }
    }

    #[test]
    fn test_load_olivetti_faces_range() {
        let (x, _) = load_olivetti_faces::<f64>().unwrap();
        // All pixel values should be in [0, 1].
        for &val in &x {
            assert!(
                (0.0..=1.0).contains(&val),
                "pixel value out of range: {val}"
            );
        }
    }

    #[test]
    fn test_load_olivetti_faces_reproducible() {
        let (x1, y1) = load_olivetti_faces::<f64>().unwrap();
        let (x2, y2) = load_olivetti_faces::<f64>().unwrap();
        assert_eq!(x1, x2, "Olivetti X should be reproducible");
        assert_eq!(y1, y2, "Olivetti y should be reproducible");
    }
}
