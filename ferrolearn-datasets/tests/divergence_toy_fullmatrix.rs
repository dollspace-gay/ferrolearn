//! FULL-matrix adversarial divergence tests for `ferrolearn-datasets/src/toy.rs`
//! vs the live scikit-learn 1.5.2 oracle.
//!
//! The existing `divergence_toy.rs` only pins X[0] / a few anchor rows. A
//! row-order, parse, encoding, or scaling bug can pass an X[0] check while
//! corrupting the rest of the matrix. These tests compare ferrolearn's loaded
//! arrays element-wise against the live oracle over the ENTIRE matrix plus the
//! full `y` vector.
//!
//! Oracle source (R-CHAR-3 — expected values are the live sklearn 1.5.2 oracle,
//! never copied from the ferrolearn side):
//! ```text
//! python3 -c "from sklearn.datasets import load_wine, load_breast_cancer, \
//!     load_diabetes, load_digits; ... np.savetxt(...)"   # run from /tmp
//! ```
//! The dumps live at `/tmp/oracle/<name>_{X,y}.txt` (full %.17g precision). If
//! the dumps are absent the test fails loudly so the comparison can never be
//! silently skipped.

use ferrolearn_datasets::{load_breast_cancer, load_diabetes, load_digits, load_wine};
use ndarray::{Array1, Array2};

const ORACLE_DIR: &str = "/tmp/oracle";

/// Read a whitespace-delimited `%.17g` oracle matrix dump into an `Array2<f64>`.
fn read_oracle_matrix(name: &str) -> Array2<f64> {
    let path = format!("{ORACLE_DIR}/{name}_X.txt");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!("oracle dump {path} missing ({e}); regenerate from live sklearn 1.5.2")
    });
    let rows: Vec<Vec<f64>> = text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            l.split_whitespace()
                .map(|t| t.parse::<f64>().unwrap())
                .collect()
        })
        .collect();
    let nrows = rows.len();
    let ncols = rows[0].len();
    let flat: Vec<f64> = rows.into_iter().flatten().collect();
    Array2::from_shape_vec((nrows, ncols), flat).unwrap()
}

/// Read a one-value-per-line `%.17g` oracle vector dump into an `Array1<f64>`.
fn read_oracle_vector(name: &str) -> Array1<f64> {
    let path = format!("{ORACLE_DIR}/{name}_y.txt");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!("oracle dump {path} missing ({e}); regenerate from live sklearn 1.5.2")
    });
    let v: Vec<f64> = text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.trim().parse::<f64>().unwrap())
        .collect();
    Array1::from_vec(v)
}

/// Assert two matrices match element-wise; report the worst cell on failure.
fn assert_matrix_eq(ferro: &Array2<f64>, oracle: &Array2<f64>, tol: f64, label: &str) {
    assert_eq!(
        ferro.shape(),
        oracle.shape(),
        "{label}: shape ferro={:?} oracle={:?}",
        ferro.shape(),
        oracle.shape()
    );
    let mut worst = 0.0_f64;
    let mut worst_at = (0usize, 0usize);
    for ((i, j), &fv) in ferro.indexed_iter() {
        let ov = oracle[[i, j]];
        let d = (fv - ov).abs();
        if d > worst {
            worst = d;
            worst_at = (i, j);
        }
    }
    let (i, j) = worst_at;
    assert!(
        worst <= tol,
        "{label}: max|ferro-sklearn| = {worst:e} at [{i},{j}] (ferro={} sklearn={}, tol={tol:e})",
        ferro[[i, j]],
        oracle[[i, j]]
    );
}

// ---------------------------------------------------------------------------
// wine — full 178x13 X + full y (counts 59/71/48), encoding 0/1/2
// ---------------------------------------------------------------------------

/// Guard: ferrolearn `load_wine` (`toy.rs:424`) full-matrix parity with
/// `sklearn/datasets/_base.py:496` (`def load_wine`).
/// Oracle: live `load_wine(return_X_y=True)`, dumped to `/tmp/oracle/wine_*.txt`.
#[test]
fn guard_wine_full_matrix() {
    let (x, y) = load_wine::<f64>().unwrap();
    let ox = read_oracle_matrix("wine");
    let oy = read_oracle_vector("wine");
    assert_matrix_eq(&x, &ox, 1e-9, "wine X");
    assert_eq!(y.len(), oy.len(), "wine y length");
    for (i, (&yi, &oyi)) in y.iter().zip(oy.iter()).enumerate() {
        assert_eq!(yi as f64, oyi, "wine y[{i}] = {yi} (sklearn = {oyi})");
    }
    // class counts 59/71/48
    let c0 = y.iter().filter(|&&c| c == 0).count();
    let c1 = y.iter().filter(|&&c| c == 1).count();
    let c2 = y.iter().filter(|&&c| c == 2).count();
    assert_eq!((c0, c1, c2), (59, 71, 48), "wine class counts");
}

// ---------------------------------------------------------------------------
// breast_cancer — full 569x30 X + full y; encoding 0=malignant(212)/1=benign(357)
// ---------------------------------------------------------------------------

/// Guard: ferrolearn `load_breast_cancer` (`toy.rs:454`) full-matrix parity with
/// `sklearn/datasets/_base.py:750` (`def load_breast_cancer`), including the
/// target encoding (0=malignant, 1=benign — NOT inverted). Oracle: live
/// `load_breast_cancer(return_X_y=True)`.
#[test]
fn guard_breast_cancer_full_matrix() {
    let (x, y) = load_breast_cancer::<f64>().unwrap();
    let ox = read_oracle_matrix("breast_cancer");
    let oy = read_oracle_vector("breast_cancer");
    assert_matrix_eq(&x, &ox, 1e-9, "breast_cancer X");
    assert_eq!(y.len(), oy.len(), "bc y length");
    for (i, (&yi, &oyi)) in y.iter().zip(oy.iter()).enumerate() {
        assert_eq!(yi as f64, oyi, "bc y[{i}] = {yi} (sklearn = {oyi})");
    }
    let malignant = y.iter().filter(|&&c| c == 0).count();
    let benign = y.iter().filter(|&&c| c == 1).count();
    assert_eq!(
        (malignant, benign),
        (212, 357),
        "bc encoding: 0=malignant should be 212, 1=benign should be 357"
    );
}

// ---------------------------------------------------------------------------
// diabetes — full 442x10 scaled X (all columns incl. col 9) + full raw y
// ---------------------------------------------------------------------------

/// Guard: ferrolearn `load_diabetes` (`toy.rs:487`) full-matrix parity with
/// `sklearn/datasets/_base.py:1044` (default `scaled=True`,
/// `_base.py:1132-1134`). Verifies the population-std (ddof=0) `/sqrt(442)`
/// scaling over EVERY cell (not just X[0,:3]) and the raw integer target over
/// all 442 samples. Oracle: live `load_diabetes(return_X_y=True)`.
#[test]
fn guard_diabetes_full_matrix() {
    let (x, y) = load_diabetes::<f64>().unwrap();
    let ox = read_oracle_matrix("diabetes");
    let oy = read_oracle_vector("diabetes");
    assert_matrix_eq(&x, &ox, 1e-7, "diabetes X");
    assert_eq!(y.len(), oy.len(), "diabetes y length");
    for (i, (&yi, &oyi)) in y.iter().zip(oy.iter()).enumerate() {
        assert!(
            (yi - oyi).abs() < 1e-9,
            "diabetes y[{i}] = {yi} (sklearn = {oyi})"
        );
    }
}

// ---------------------------------------------------------------------------
// digits — full 1797x64 X (integers 0..16) + full y (counts ~180 each)
// ---------------------------------------------------------------------------

/// Guard: ferrolearn `load_digits` (`toy.rs:523`) full-matrix parity with
/// `sklearn/datasets/_base.py:907` (`def load_digits`, default n_class=10 → all
/// 1797 samples). Verifies all 1797x64 pixels are integers in 0..=16 and the
/// full `y`. Oracle: live `load_digits(return_X_y=True)`.
#[test]
fn guard_digits_full_matrix() {
    let (x, y) = load_digits::<f64>().unwrap();
    let ox = read_oracle_matrix("digits");
    let oy = read_oracle_vector("digits");
    assert_eq!(x.shape(), &[1797, 64], "digits shape");
    assert_matrix_eq(&x, &ox, 1e-9, "digits X");
    // pixel values are integers in 0..=16
    for ((i, j), &v) in x.indexed_iter() {
        assert!(
            v.fract() == 0.0 && (0.0..=16.0).contains(&v),
            "digits X[{i},{j}] = {v} not an integer in 0..=16"
        );
    }
    assert_eq!(y.len(), oy.len(), "digits y length");
    for (i, (&yi, &oyi)) in y.iter().zip(oy.iter()).enumerate() {
        assert_eq!(yi as f64, oyi, "digits y[{i}] = {yi} (sklearn = {oyi})");
    }
    let mut counts = [0usize; 10];
    for &c in &y {
        counts[c] += 1;
    }
    assert_eq!(
        counts,
        [178, 182, 177, 183, 181, 182, 181, 179, 174, 180],
        "digits class counts"
    );
}
