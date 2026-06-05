//! Divergence tests for `ferrolearn-datasets/src/generators.rs` vs scikit-learn 1.5.2.
//!
//! Crosslink unit: #1890. Design doc: `.design/datasets/generators.md`.
//!
//! These pin the DETERMINISTIC (RNG-independent) divergences against the live
//! sklearn 1.5.2 oracle (run from /tmp; R-CHAR-3 — expected values NEVER copied
//! from the ferrolearn side). The stochastic value-parity divergences are
//! RNG-substrate-blocked (numpy Mersenne-Twister vs `SmallRng`+`rand_distr`) and
//! are filed as `-l blocker` issues WITHOUT a doomed value test (R-SUBSTRATE-5,
//! R-DEFER-3, prior RNG carve-out precedent): #1893 (umbrella RNG), #1894
//! (make_classification), #1895 (make_regression/make_blobs), #1896
//! (make_sparse_uncorrelated weights), #1897 (make_friedman3 +1e-6 / draw order),
//! #1898 (low_rank/spd/sparse_spd), #1899 (make_gaussian_quantiles reorder),
//! #1900 (make_multilabel_classification), #1901 (ferray substrate).
//!
//! Oracle commands (sklearn 1.5.2):
//!   make_moons(n_samples=10, shuffle=False, noise=None) -> X below
//!   make_circles(n_samples=10, shuffle=False, noise=None, factor=0.5) -> X below
//!   make_hastie_10_2(n_samples=20, random_state=0) -> np.unique(y)==[-1.,1.], dtype float64

use ferrolearn_datasets::{make_circles, make_hastie_10_2, make_moons};

// ---------------------------------------------------------------------------
// REQ-4 — make_moons geometry (FAILS today: endpoint off-by-one)
// ---------------------------------------------------------------------------

/// Divergence: ferrolearn's `make_moons` (`generators.rs` →
/// `theta = PI * i / n_upper`, endpoint=False spacing `i/n`) diverges from
/// `sklearn/datasets/_samples_generator.py:901-904`
/// (`np.cos(np.linspace(0, np.pi, n_samples_out))` — endpoint=True, spacing
/// `i/(n-1)`) in the deterministic geometry.
///
/// Oracle (live `make_moons(n_samples=10, shuffle=False, noise=None)`):
///   row 1 = [0.7071067811865476, 0.7071067811865475]  (theta = pi/4)
/// ferrolearn emits theta = pi/5 for row 1 -> [0.809..., 0.587...]. FAILS.
/// Tracking: #1891
#[test]
#[allow(
    clippy::approx_constant,
    reason = "exact sklearn make_moons oracle literals (1/sqrt(2)), must stay bit-for-bit, not the math constant"
)]
fn divergence_moons_geometry() {
    // n=10 -> n_upper = 5, n_lower = 5. noise=0.0 -> deterministic.
    let (x, y) = make_moons::<f64>(10, 0.0, Some(0)).unwrap();
    assert_eq!(x.shape(), &[10, 2]);

    // Live sklearn oracle (make_moons(10, shuffle=False, noise=None)).
    let oracle: [[f64; 2]; 10] = [
        [1.0, 0.0],
        [0.7071067811865476, 0.7071067811865475],
        [6.123233995736766e-17, 1.0],
        [-0.7071067811865475, 0.7071067811865476],
        [-1.0, 1.2246467991473532e-16],
        [0.0, 0.5],
        [0.2928932188134524, -0.20710678118654746],
        [0.999_999_999_999_999_9, -0.5],
        [1.7071067811865475, -0.20710678118654757],
        [2.0, 0.499_999_999_999_999_9],
    ];

    for i in 0..10 {
        for j in 0..2 {
            assert!(
                (x[[i, j]] - oracle[i][j]).abs() < 1e-12,
                "moons X[{i},{j}] = {} (sklearn = {})",
                x[[i, j]],
                oracle[i][j]
            );
        }
    }
    // Labels: contiguous 0-block then 1-block (shuffle=False).
    let expected_y = [0usize, 0, 0, 0, 0, 1, 1, 1, 1, 1];
    for i in 0..10 {
        assert_eq!(y[i], expected_y[i], "moons y[{i}]");
    }
}

// ---------------------------------------------------------------------------
// REQ-18 — make_hastie_10_2 label encoding (FAILS today: {0,1} usize)
// ---------------------------------------------------------------------------

/// Divergence: ferrolearn's `make_hastie_10_2` (`generators.rs` →
/// `Array1<usize>` with `y[i] = if s > 9.34 { 1 } else { 0 }`) diverges from
/// `sklearn/datasets/_samples_generator.py:567-568`
/// (`y = ((X**2.0).sum(axis=1) > 9.34).astype(np.float64); y[y == 0.0] = -1.0`).
/// sklearn's label set is `{-1.0, +1.0}` as float64; ferrolearn returns `{0, 1}`
/// as usize.
///
/// Oracle (live `make_hastie_10_2(n_samples=20, random_state=0)`):
///   np.unique(y) == [-1.0, 1.0], y.dtype == float64.
///
/// This is an R-DEV-3 output-contract divergence (wrong dtype AND wrong label
/// set). The label values are RNG-independent: the negative class is sklearn's
/// `-1.0`, not `0`. We pin the label SET regardless of which rows fall in which
/// class. The current usize return cannot represent -1, so the assertion that
/// the negative label equals sklearn's -1.0 fails. FAILS.
/// Tracking: #1892
#[test]
fn divergence_hastie_label_encoding() {
    // Generate enough samples that both classes appear under any RNG.
    // y is now Array1<f64> (sklearn's np.float64 dtype, _samples_generator.py:567).
    let (_x, y) = make_hastie_10_2::<f64>(200, Some(0)).unwrap();

    // Live sklearn oracle label set: {-1.0, +1.0} as f64.
    // sklearn maps the "below threshold" class to -1.0, not 0
    // (_samples_generator.py:568 `y[y == 0.0] = -1.0`).
    let sk_negative_label: f64 = -1.0;
    let sk_positive_label: f64 = 1.0;

    // Float-safe assertion of the distinct label set: every value is exactly
    // -1.0 or +1.0 (the negative class is -1.0, NOT 0.0), and BOTH appear.
    let mut saw_negative = false;
    let mut saw_positive = false;
    for &v in y.iter() {
        assert!(
            v == sk_negative_label || v == sk_positive_label,
            "hastie label {v} is not in sklearn's {{-1.0, +1.0}} set \
             (_samples_generator.py:567-568); the negative class must be -1.0, not 0.0"
        );
        if v == sk_negative_label {
            saw_negative = true;
        }
        if v == sk_positive_label {
            saw_positive = true;
        }
    }
    assert!(
        saw_negative,
        "hastie never emitted the -1.0 negative class (sklearn maps below-threshold \
         to -1.0, _samples_generator.py:568)"
    );
    assert!(
        saw_positive,
        "hastie never emitted the +1.0 positive class \
         (.astype(float64) of `sum(x^2) > 9.34`, _samples_generator.py:567)"
    );
}

// ---------------------------------------------------------------------------
// REQ-6 — make_circles geometry GREEN GUARD (SHIPPED slice; must PASS)
// ---------------------------------------------------------------------------

/// Guard (SHIPPED — REQ-6 geometry slice): ferrolearn's `make_circles`
/// (`generators.rs` → `theta = 2*PI*i / n_outer`, endpoint=False) matches
/// `sklearn/datasets/_samples_generator.py:813-818`
/// (`np.linspace(0, 2*np.pi, n, endpoint=False)`, inner scaled by `factor`)
/// element-wise at `noise=None`.
///
/// Oracle (live `make_circles(n_samples=10, shuffle=False, noise=None,
/// factor=0.5)`): X below. PASSES today; pins the one SHIPPED slice (any future
/// regression of the endpoint-False geometry or the factor scaling fails here).
#[test]
fn guard_circles_geometry_oracle_parity() {
    // ferrolearn make_circles(n_samples, noise, factor, seed); noise=0.0 -> deterministic.
    let (x, y) = make_circles::<f64>(10, 0.0, 0.5, Some(0)).unwrap();
    assert_eq!(x.shape(), &[10, 2]);

    // Live sklearn oracle (make_circles(10, shuffle=False, noise=None, factor=0.5)).
    let oracle: [[f64; 2]; 10] = [
        [1.0, 0.0],
        [0.30901699437494745, 0.9510565162951535],
        [-0.8090169943749473, 0.5877852522924732],
        [-0.8090169943749476, -0.587785252292473],
        [0.30901699437494723, -0.9510565162951536],
        [0.5, 0.0],
        [0.15450849718747373, 0.47552825814757677],
        [-0.40450849718747367, 0.2938926261462366],
        [-0.4045084971874738, -0.2938926261462365],
        [0.15450849718747361, -0.4755282581475768],
    ];

    for i in 0..10 {
        for j in 0..2 {
            assert!(
                (x[[i, j]] - oracle[i][j]).abs() < 1e-12,
                "circles X[{i},{j}] = {} (sklearn = {})",
                x[[i, j]],
                oracle[i][j]
            );
        }
    }
    let expected_y = [0usize, 0, 0, 0, 0, 1, 1, 1, 1, 1];
    for i in 0..10 {
        assert_eq!(y[i], expected_y[i], "circles y[{i}]");
    }
}
