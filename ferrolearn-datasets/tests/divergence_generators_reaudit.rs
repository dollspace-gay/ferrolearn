//! Re-audit guards for the #1891 (make_moons endpoint) and #1892
//! (make_hastie_10_2 label encoding) fixes vs scikit-learn 1.5.2.
//!
//! Crosslink unit: #1890. Design doc: `.design/datasets/generators.md`.
//!
//! The first audit round (`divergence_generators.rs`) covered make_moons ONLY at
//! n=10 (even split, n_upper==n_lower==5, equal denominators). These guards close
//! the adversarial gaps the first round did not cover, all against the LIVE
//! sklearn 1.5.2 oracle (run from /tmp; R-CHAR-3 — expected values NEVER copied
//! from the ferrolearn side):
//!
//!   * ODD n (n=11 -> n_upper=5, n_lower=6: DIFFERENT counts, DIFFERENT
//!     denominators i/(n_upper-1) vs i/(n_lower-1)).
//!   * SMALL-n EDGE (n=2 -> n_upper=1,n_lower=1; n=3 -> n_upper=1,n_lower=2):
//!     numpy `np.linspace(0, pi, 1) == [0.0]`, so the single-point moon's angle
//!     must be 0.0 (NOT a divide-by-zero, NOT pi). Confirms the fix's
//!     `n_upper.saturating_sub(1).max(1)` denominator gives angle 0.0 when the
//!     moon has exactly 1 point.
//!   * make_hastie_10_2 THRESHOLD SEMANTICS: strict `> 9.34` -> +1.0, `<= 9.34`
//!     -> -1.0; output shapes (n,10) and (n,). The RNG-bound which-row-is-which
//!     assignment is NOT value-matched here (blocker #1893/#1901 territory).
//!
//! Oracle commands (sklearn 1.5.2, from /tmp):
//!   make_moons(n_samples=2,  shuffle=False, noise=None)
//!     -> X=[[1.0,0.0],[0.0,0.5]], y=[0,1]
//!   make_moons(n_samples=3,  shuffle=False, noise=None)
//!     -> X=[[1.0,0.0],[0.0,0.5],[2.0,0.4999999999999999]], y=[0,1,1]
//!   make_moons(n_samples=11, shuffle=False, noise=None)
//!     -> X below, y=[0,0,0,0,0,1,1,1,1,1,1]
//!   np.linspace(0, pi, 1) == [0.0]
//!   make_hastie_10_2(n_samples=20, random_state=0)
//!     -> y == np.where((X**2).sum(1)>9.34, 1.0, -1.0); dtype float64; shapes (20,10),(20,)

use ferrolearn_datasets::{make_hastie_10_2, make_moons};

// ---------------------------------------------------------------------------
// #1891 — make_moons endpoint fix, SMALL-n EDGE (n=2)
// ---------------------------------------------------------------------------

/// Guard (#1891): single-point moons. For n=2, n_upper=1 and n_lower=1; numpy
/// `np.linspace(0, pi, 1) == [0.0]` (`_samples_generator.py:901-904`), so each
/// moon's lone angle is 0.0. The fix's `n_upper.saturating_sub(1).max(1)`
/// denominator must give angle `pi*0/1 == 0.0` (NOT divide-by-zero, NOT pi).
///
/// Oracle: `make_moons(n_samples=2, shuffle=False, noise=None)`
///   -> X=[[1.0,0.0],[0.0,0.5]], y=[0,1].
#[test]
fn guard_moons_n2_single_point_edge() {
    let (x, y) = make_moons::<f64>(2, 0.0, Some(0)).unwrap();
    assert_eq!(x.shape(), &[2, 2]);

    // Live sklearn oracle.
    let oracle: [[f64; 2]; 2] = [[1.0, 0.0], [0.0, 0.5]];
    for i in 0..2 {
        for j in 0..2 {
            assert!(
                (x[[i, j]] - oracle[i][j]).abs() < 1e-12,
                "moons(n=2) X[{i},{j}] = {} (sklearn = {})",
                x[[i, j]],
                oracle[i][j]
            );
        }
    }
    let expected_y = [0usize, 1];
    for i in 0..2 {
        assert_eq!(y[i], expected_y[i], "moons(n=2) y[{i}]");
    }
}

// ---------------------------------------------------------------------------
// #1891 — make_moons endpoint fix, SMALL-n EDGE (n=3, odd, n_lower=2)
// ---------------------------------------------------------------------------

/// Guard (#1891): n=3 -> n_upper=1 (single-point upper moon, angle 0.0),
/// n_lower=2 (`np.linspace(0, pi, 2) == [0.0, pi]`). The lower moon's second
/// point uses theta=pi: `1-cos(pi)=2.0`, `1-sin(pi)-0.5=0.5`.
///
/// Oracle: `make_moons(n_samples=3, shuffle=False, noise=None)`
///   -> X=[[1.0,0.0],[0.0,0.5],[2.0,0.4999999999999999]], y=[0,1,1].
#[test]
fn guard_moons_n3_odd_split() {
    let (x, y) = make_moons::<f64>(3, 0.0, Some(0)).unwrap();
    assert_eq!(x.shape(), &[3, 2]);

    // Live sklearn oracle.
    let oracle: [[f64; 2]; 3] = [[1.0, 0.0], [0.0, 0.5], [2.0, 0.499_999_999_999_999_9]];
    for i in 0..3 {
        for j in 0..2 {
            assert!(
                (x[[i, j]] - oracle[i][j]).abs() < 1e-12,
                "moons(n=3) X[{i},{j}] = {} (sklearn = {})",
                x[[i, j]],
                oracle[i][j]
            );
        }
    }
    let expected_y = [0usize, 1, 1];
    for i in 0..3 {
        assert_eq!(y[i], expected_y[i], "moons(n=3) y[{i}]");
    }
}

// ---------------------------------------------------------------------------
// #1891 — make_moons endpoint fix, ODD n=11 (different denominators)
// ---------------------------------------------------------------------------

/// Guard (#1891): n=11 -> n_upper=5, n_lower=6. The two moons have DIFFERENT
/// counts, hence DIFFERENT linspace denominators: upper uses `i/(5-1)`, lower
/// `i/(6-1)` (`_samples_generator.py:901-904`, endpoint=True). This is the gap
/// the n=10 even-split test (equal denominators) could not detect.
///
/// Oracle: `make_moons(n_samples=11, shuffle=False, noise=None)` -> X below,
/// labels = contiguous 5 zeros then 6 ones.
#[test]
#[allow(
    clippy::approx_constant,
    reason = "exact sklearn make_moons oracle literals (1/sqrt(2)), must stay bit-for-bit, not the math constant"
)]
fn guard_moons_n11_odd_different_denominators() {
    let (x, y) = make_moons::<f64>(11, 0.0, Some(0)).unwrap();
    assert_eq!(x.shape(), &[11, 2]);

    // Live sklearn oracle (make_moons(11, shuffle=False, noise=None)).
    let oracle: [[f64; 2]; 11] = [
        [1.0, 0.0],
        [0.7071067811865476, 0.7071067811865475],
        [6.123233995736766e-17, 1.0],
        [-0.7071067811865475, 0.7071067811865476],
        [-1.0, 1.2246467991473532e-16],
        [0.0, 0.5],
        [0.19098300562505255, -0.08778525229247314],
        [0.6909830056250525, -0.45105651629515353],
        [1.3090169943749475, -0.45105651629515364],
        [1.8090169943749475, -0.08778525229247325],
        [2.0, 0.499_999_999_999_999_9],
    ];
    for i in 0..11 {
        for j in 0..2 {
            assert!(
                (x[[i, j]] - oracle[i][j]).abs() < 1e-12,
                "moons(n=11) X[{i},{j}] = {} (sklearn = {})",
                x[[i, j]],
                oracle[i][j]
            );
        }
    }
    // Contiguous label blocks under shuffle=False: 5 zeros then 6 ones.
    let expected_y = [0usize, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1];
    for i in 0..11 {
        assert_eq!(y[i], expected_y[i], "moons(n=11) y[{i}]");
    }
}

// ---------------------------------------------------------------------------
// #1892 — make_hastie_10_2 threshold semantics + output shapes
// ---------------------------------------------------------------------------

/// Guard (#1892): make_hastie_10_2 threshold + shapes. sklearn computes
/// `y = ((X**2).sum(axis=1) > 9.34).astype(float64); y[y==0.0] = -1.0`
/// (`_samples_generator.py:567-568`): STRICT `> 9.34` maps a row to +1.0, and
/// `<= 9.34` maps it to -1.0. The positive class is the above-threshold one.
///
/// We do NOT value-match which rows are +1/-1 (RNG-bound: SmallRng != numpy
/// Mersenne-Twister; blocker #1893/#1901). We DO confirm, against the per-row
/// sum(x^2) computed from ferrolearn's OWN X, that ferrolearn applies the SAME
/// strict `> 9.34 -> +1.0 / <= 9.34 -> -1.0` rule the oracle does, and that the
/// shapes are (n,10) and (n,). The threshold constant 9.34 and the strict-`>`
/// direction are sklearn-source constants, not copied from the ferrolearn side.
#[test]
fn guard_hastie_threshold_semantics_and_shapes() {
    let (x, y) = make_hastie_10_2::<f64>(200, Some(0)).unwrap();

    // Output-contract shapes: X (n,10), y (n,).
    assert_eq!(x.shape(), &[200, 10], "hastie X shape must be (n, 10)");
    assert_eq!(y.len(), 200, "hastie y len must be n");

    // sklearn source constants (_samples_generator.py:567-568).
    const SK_THRESHOLD: f64 = 9.34;
    const SK_POSITIVE: f64 = 1.0;
    const SK_NEGATIVE: f64 = -1.0;

    for i in 0..200 {
        // Recompute sum(x_i^2) from ferrolearn's own emitted X row, then apply
        // the SKLEARN rule and require ferrolearn's label to agree. This pins the
        // threshold direction without depending on the RNG bit-stream.
        let mut ssq = 0.0_f64;
        for j in 0..10 {
            let v = x[[i, j]];
            ssq += v * v;
        }
        let expected = if ssq > SK_THRESHOLD {
            SK_POSITIVE
        } else {
            SK_NEGATIVE
        };
        assert_eq!(
            y[i], expected,
            "hastie row {i}: sum(x^2)={ssq}, sklearn rule (>{SK_THRESHOLD} -> {SK_POSITIVE}, \
             else {SK_NEGATIVE}, _samples_generator.py:567-568) expects {expected}, got {}",
            y[i]
        );
    }
}
