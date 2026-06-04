//! Divergence tests for `ferrolearn-datasets/src/toy.rs` vs scikit-learn 1.5.2.
//!
//! Expected values are taken from the live sklearn 1.5.2 oracle (run from /tmp):
//! `from sklearn.datasets import load_* ; load_*(return_X_y=True)` (R-CHAR-3 —
//! never copied from the ferrolearn side). The synthetic-stub loaders
//! (`load_wine`, `load_breast_cancer`, `load_diabetes`, `load_digits`) return
//! deterministic FAKE data with the correct shape but wrong values, so the
//! value-parity tests below FAIL today. `load_iris` and `load_linnerud` embed
//! the real data, so their element-wise guards PASS (confirmed-SHIPPED green
//! guards, not blockers).

use ferrolearn_datasets::{
    load_breast_cancer, load_diabetes, load_digits, load_iris, load_linnerud, load_wine,
};

// ---------------------------------------------------------------------------
// REQ-3 — load_wine value parity (FAILS today: synthetic_classification stub)
// ---------------------------------------------------------------------------

/// Divergence: ferrolearn's `load_wine` (`toy.rs` → `synthetic_classification(178,13,3)`)
/// diverges from `sklearn/datasets/_base.py:496` (`def load_wine`) for X[0].
/// sklearn returns the real wine measurements; ferrolearn returns `class*5 + idx*0.001`.
/// Oracle (live `load_wine(return_X_y=True)[0][0]`):
///   [14.23,1.71,2.43,15.6,127.0,2.8,3.06,0.28,2.29,5.64,1.04,3.92,1065.0]
/// Tracking: #1652
#[test]
fn divergence_wine_x0() {
    let (x, y) = load_wine::<f64>().unwrap();
    let expected = [
        14.23, 1.71, 2.43, 15.6, 127.0, 2.8, 3.06, 0.28, 2.29, 5.64, 1.04, 3.92, 1065.0,
    ];
    let row = x.row(0);
    for (j, &want) in expected.iter().enumerate() {
        assert!(
            (row[j] - want).abs() < 1e-6,
            "wine X[0,{j}] = {} (sklearn = {want})",
            row[j]
        );
    }
    // sklearn class 0 is the first of three cultivars; first sample is class 0.
    assert_eq!(y[0], 0, "wine y[0] = {} (sklearn = 0)", y[0]);
}

// ---------------------------------------------------------------------------
// REQ-4 — load_breast_cancer value parity (FAILS today: synthetic stub)
// ---------------------------------------------------------------------------

/// Divergence: ferrolearn's `load_breast_cancer` (`toy.rs` →
/// `synthetic_classification(569,30,2)`) diverges from
/// `sklearn/datasets/_base.py:750` (`def load_breast_cancer`) for X[0] and y[0].
/// sklearn returns the real WDBC measurements (target 0=malignant); ferrolearn
/// returns synthetic offset-blob values.
/// Oracle (live `load_breast_cancer(return_X_y=True)[0][0,:3]`): [17.99,10.38,122.8];
/// y[0] = 0 (malignant).
/// Tracking: #1652
#[test]
fn divergence_breast_cancer_x0() {
    let (x, y) = load_breast_cancer::<f64>().unwrap();
    let expected_first3 = [17.99, 10.38, 122.8];
    let row = x.row(0);
    for (j, &want) in expected_first3.iter().enumerate() {
        assert!(
            (row[j] - want).abs() < 1e-6,
            "breast_cancer X[0,{j}] = {} (sklearn = {want})",
            row[j]
        );
    }
    assert_eq!(y[0], 0, "breast_cancer y[0] = {} (sklearn = 0)", y[0]);
}

// ---------------------------------------------------------------------------
// REQ-5 — load_diabetes value parity (FAILS today: synthetic, no scaling)
// ---------------------------------------------------------------------------

/// Divergence: ferrolearn's `load_diabetes` (`toy.rs` →
/// `synthetic_regression(442,10)`) diverges from
/// `sklearn/datasets/_base.py:1044` (`def load_diabetes`, default `scaled=True`)
/// for X[0,:3] and y[:3]. sklearn's default scaling (`_base.py:1133-1134`:
/// `data = scale(data, copy=False); data /= data.shape[0] ** 0.5`) yields the
/// oracle below; ferrolearn applies no scaling transform and emits synthetic
/// `idx*0.01` values.
/// Oracle (live `load_diabetes(return_X_y=True)`):
///   X[0,:3] = [0.038075906433423026,0.05068011873981862,0.061696206518683294];
///   y[:3]   = [151.0,75.0,141.0]
/// Tracking: #1652
#[test]
fn divergence_diabetes_x0_y() {
    let (x, y) = load_diabetes::<f64>().unwrap();
    let expected_x = [
        0.038075906433423026,
        0.05068011873981862,
        0.061696206518683294,
    ];
    let row = x.row(0);
    for (j, &want) in expected_x.iter().enumerate() {
        assert!(
            (row[j] - want).abs() < 1e-9,
            "diabetes X[0,{j}] = {} (sklearn = {want})",
            row[j]
        );
    }
    let expected_y = [151.0, 75.0, 141.0];
    for (i, &want) in expected_y.iter().enumerate() {
        assert!(
            (y[i] - want).abs() < 1e-9,
            "diabetes y[{i}] = {} (sklearn = {want})",
            y[i]
        );
    }
}

// ---------------------------------------------------------------------------
// REQ-6 — load_digits full shape + value parity (FAILS today: 200 rows, random)
// ---------------------------------------------------------------------------

/// Divergence: ferrolearn's `load_digits` (`toy.rs` →
/// `synthetic_classification(200,64,10)`) diverges from
/// `sklearn/datasets/_base.py:907` (`def load_digits(*, n_class=10, ...)`) in
/// BOTH shape and values. sklearn returns all 1797 samples; ferrolearn truncates
/// to 200. sklearn returns the real 8x8 digit pixels; ferrolearn returns synthetic.
/// Oracle (live `load_digits(return_X_y=True)`): X.shape == (1797,64),
///   X[0,:8] == [0,0,5,13,9,1,0,0].
/// Tracking: #1652
#[test]
fn divergence_digits_shape_and_x0() {
    let (x, _y) = load_digits::<f64>().unwrap();
    assert_eq!(
        x.shape(),
        &[1797, 64],
        "digits shape = {:?} (sklearn = [1797, 64])",
        x.shape()
    );
    let expected_first8 = [0.0, 0.0, 5.0, 13.0, 9.0, 1.0, 0.0, 0.0];
    let row = x.row(0);
    for (j, &want) in expected_first8.iter().enumerate() {
        assert!(
            (row[j] - want).abs() < 1e-9,
            "digits X[0,{j}] = {} (sklearn = {want})",
            row[j]
        );
    }
}

// ---------------------------------------------------------------------------
// REQ-1 — load_iris GREEN GUARD (embedded real data; element-wise vs oracle)
// ---------------------------------------------------------------------------

/// Guard (SHIPPED): ferrolearn's `load_iris` (`toy.rs` → `parse_classification_csv`)
/// matches `sklearn/datasets/_base.py:620` (`def load_iris`) element-wise on the
/// oracle anchor rows + class balance.
/// Oracle (live `load_iris(return_X_y=True)`): X[0]==[5.1,3.5,1.4,0.2],
///   X[149]==[5.9,3.0,5.1,1.8], y[0]==0, y[149]==2, counts 50/50/50.
/// PASSES today; pins iris as confirmed-SHIPPED (any future regression fails).
#[test]
fn guard_iris_oracle_parity() {
    let (x, y) = load_iris::<f64>().unwrap();
    assert_eq!(x.shape(), &[150, 4]);
    let x0 = [5.1, 3.5, 1.4, 0.2];
    let x149 = [5.9, 3.0, 5.1, 1.8];
    for j in 0..4 {
        assert!((x[[0, j]] - x0[j]).abs() < 1e-9, "iris X[0,{j}]");
        assert!((x[[149, j]] - x149[j]).abs() < 1e-9, "iris X[149,{j}]");
    }
    assert_eq!(y[0], 0);
    assert_eq!(y[149], 2);
    for c in 0..3 {
        let n = y.iter().filter(|&&v| v == c).count();
        assert_eq!(n, 50, "iris class {c} count");
    }
}

// ---------------------------------------------------------------------------
// REQ-2 — load_linnerud GREEN GUARD (embedded real data; ALL 20x3 of X and y)
// ---------------------------------------------------------------------------

/// Guard (SHIPPED): ferrolearn's `load_linnerud` (`toy.rs` → embedded
/// `LINNERUD_FEATURES`/`LINNERUD_TARGETS`) matches
/// `sklearn/datasets/_base.py:1171` (`def load_linnerud`) element-wise over ALL
/// 20x3 cells of both X (exercise) and y (physiological).
/// Oracle: full `load_linnerud(return_X_y=True)` matrices (the doc-author flagged
/// linnerud as least-verified — this diffs every cell).
/// PASSES today; pins linnerud as confirmed-SHIPPED.
#[test]
fn guard_linnerud_full_oracle_parity() {
    let (x, y) = load_linnerud::<f64>().unwrap();
    assert_eq!(x.shape(), &[20, 3]);
    assert_eq!(y.shape(), &[20, 3]);

    // Live oracle `load_linnerud(return_X_y=True)[0]` (exercise features).
    let x_oracle: [[f64; 3]; 20] = [
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
    // Live oracle `load_linnerud(return_X_y=True)[1]` (physiological targets).
    let y_oracle: [[f64; 3]; 20] = [
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

    for i in 0..20 {
        for j in 0..3 {
            assert!(
                (x[[i, j]] - x_oracle[i][j]).abs() < 1e-9,
                "linnerud X[{i},{j}] = {} (sklearn = {})",
                x[[i, j]],
                x_oracle[i][j]
            );
            assert!(
                (y[[i, j]] - y_oracle[i][j]).abs() < 1e-9,
                "linnerud y[{i},{j}] = {} (sklearn = {})",
                y[[i, j]],
                y_oracle[i][j]
            );
        }
    }
}
