//! Divergence audit for `ferrolearn-numerical/src/interpolate.rs` against the
//! LIVE scipy.interpolate oracle (scipy 1.17.1). See
//! `.design/numerical/interpolate.md`.
//!
//! All expected values are computed by a live scipy call run from `/tmp`
//! (R-CHAR-3) and are NEVER copied from the ferrolearn side. The single
//! oracle command that produced every constant below:
//!
//! ```text
//! cd /tmp && python3 -c "
//! import numpy as np
//! from scipy.interpolate import CubicSpline
//! # 3-point not-a-knot (the headline bug):
//! print('3pt nak', CubicSpline(np.array([0.,1.,2.]), np.array([0.,1.,4.]), bc_type='not-a-knot')([0.5,1.0,1.5]).tolist())
//! x=np.array([0.,1.,2.,3.,4.]); y=np.array([0.,1.,4.,9.,16.])
//! print('nat eval', CubicSpline(x,y,bc_type='natural')([0.5,1.5,2.5,3.5]).tolist())
//! print('nak eval', CubicSpline(x,y,bc_type='not-a-knot')([0.5,1.5,2.5,3.5]).tolist())
//! print('nat d1@1.5', float(CubicSpline(x,y,bc_type='natural')(1.5,1)), 'd2@2.5', float(CubicSpline(x,y,bc_type='natural')(2.5,2)))
//! print('nak d1@1.5', float(CubicSpline(x,y,bc_type='not-a-knot')(1.5,1)), 'd2@2.5', float(CubicSpline(x,y,bc_type='not-a-knot')(2.5,2)))
//! print('nat integ', float(CubicSpline(x,y,bc_type='natural').integrate(0,4)))
//! print('nak integ', float(CubicSpline(x,y,bc_type='not-a-knot').integrate(0,4)))
//! xs=np.linspace(0,6,7); ys=np.sin(xs)
//! print('sin nat', CubicSpline(xs,ys,bc_type='natural')([0.3,2.7,5.1]).tolist())
//! print('sin nak', CubicSpline(xs,ys)([0.3,2.7,5.1]).tolist())
//! print('extrap nat', CubicSpline(x,y,bc_type='natural')([-0.5,4.5]).tolist())
//! print('extrap nak', CubicSpline(x,y,bc_type='not-a-knot')([-0.5,4.5]).tolist())
//! "
//! ```
//!
//! scipy upstream: `scipy/interpolate/_cubic.py`, `class CubicSpline.__init__`
//! lines 820-842 — the `n == 3 and bc[0] == 'not-a-knot' and bc[1] ==
//! 'not-a-knot'` special case that constructs the parabola through the 3 points
//! ("In this case 'not-a-knot' can't be handled regularly as the both
//! conditions are identical."). ferrolearn's `solve_not_a_knot_general` has no
//! analog of this special case.

use ferrolearn_core::error::FerroError;
use ferrolearn_numerical::interpolate::{BoundaryCondition, CubicSpline};

// ===========================================================================
// (1) RED PIN — REQ-FEWPOINT: 3-point not-a-knot divergence.
//
// Single-file-fixable in `interpolate.rs` (`solve_not_a_knot_general` / `new`
// must special-case the 3-point / single-interior-knot not-a-knot system as
// the unique parabola, exactly as scipy/_cubic.py:824 does).
//
// Oracle (live scipy 1.17.1, from /tmp):
//   CubicSpline([0,1,2],[0,1,4],bc_type='not-a-knot')([0.5,1.0,1.5])
//     = [0.2499999999999999, 1.0, 2.25]
// The data is the parabola y = x^2; 3-point not-a-knot reproduces it exactly,
// so eval(0.5)=0.25 and eval(1.5)=2.25.
//
// ferrolearn `NotAKnot` yields [0.125, 1.0, 2.375] (abs error ~0.125 at the
// interior; the knot eval(1.0)=1.0 is exact, confirming this is an
// interior-shape bug from the degenerate n=2 not-a-knot elimination, NOT a
// knot/value-placement bug).
// ===========================================================================

/// Divergence: `CubicSpline::new(_, _, NotAKnot)` over 3 points in
/// `interpolate.rs` (`solve_not_a_knot_general` at single interior knot)
/// diverges from `scipy/interpolate/_cubic.py:824` (the n==3 not-a-knot
/// parabola special case).
///
/// Input: `x=[0,1,2]`, `y=[0,1,4]` (the parabola y=x^2), `NotAKnot`.
/// scipy returns `eval([0.5,1.5]) = [0.25, 2.25]` (exact parabola).
/// ferrolearn returns `[0.125, 2.375]` (abs error ~0.125).
///
/// Tracking: see filed blocker (REQ-FEWPOINT).
#[test]
fn divergence_three_point_not_a_knot_parabola() {
    // Oracle constants (live scipy 1.17.1, run from /tmp — NEVER from ferrolearn):
    //   CubicSpline([0,1,2],[0,1,4],bc_type='not-a-knot')([0.5,1.0,1.5])
    //     = [0.2499999999999999, 1.0, 2.25]
    const SCIPY_AT_0_5: f64 = 0.2499999999999999;
    const SCIPY_AT_1_0: f64 = 1.0; // the interior knot — must be exact
    const SCIPY_AT_1_5: f64 = 2.25;

    let spline = CubicSpline::new(
        &[0.0, 1.0, 2.0],
        &[0.0, 1.0, 4.0],
        BoundaryCondition::NotAKnot,
    )
    .unwrap();

    // Knot value is exact (confirms the divergence is interior shape, not a
    // value/placement bug).
    assert!(
        (spline.eval(1.0) - SCIPY_AT_1_0).abs() <= 1e-12,
        "3-point not-a-knot knot eval(1.0): ferrolearn {} vs scipy {}",
        spline.eval(1.0),
        SCIPY_AT_1_0
    );

    // Interior points — these FAIL against the current implementation.
    let got_0_5 = spline.eval(0.5);
    let got_1_5 = spline.eval(1.5);
    assert!(
        (got_0_5 - SCIPY_AT_0_5).abs() <= 1e-10,
        "3-point not-a-knot eval(0.5): ferrolearn {got_0_5} vs scipy {SCIPY_AT_0_5} (parabola)"
    );
    assert!(
        (got_1_5 - SCIPY_AT_1_5).abs() <= 1e-10,
        "3-point not-a-knot eval(1.5): ferrolearn {got_1_5} vs scipy {SCIPY_AT_1_5} (parabola)"
    );
}

// ===========================================================================
// (1b) GREEN PIN — REQ-ERR-TYPE: error-type parity (`FerroError` vs scipy
// `ValueError`).
//
// scipy's `CubicSpline` validates its inputs in `prepare_input`
// (`scipy/interpolate/_cubic.py:48-65`) and raises `ValueError` on each of the
// three bad inputs below. Confirmed LIVE (R-CHAR-3, scipy 1.17.1, run from /tmp):
//
//   cd /tmp && python3 -c "
//   import numpy as np
//   from scipy.interpolate import CubicSpline
//   for x, y in [([0.,1.],[0.,1.,2.]), ([0.],[0.]), ([0.,0.,1.],[0.,1.,2.])]:
//       try:
//           CubicSpline(np.array(x), np.array(y))
//       except ValueError as e:
//           print('ValueError:', str(e).strip())
//   "
//   -> length mismatch  -> ValueError: The length of `y` along `axis`=0 doesn't
//                          match the length of `x`           (_cubic.py:51-53)
//   -> too few points   -> ValueError: `x` must contain at least 2 elements.
//                                                            (_cubic.py:50)
//   -> non-increasing x -> ValueError: `x` must be strictly increasing sequence.
//                                                            (_cubic.py:65)
//
// `FerroError::InvalidParameter` is the ferrolearn analog of scipy's
// input-validation `ValueError` (cf. `ferrolearn-core/src/error.rs`
// REQ-4: "Mirrors sklearn `ValueError` from `_parameter_constraints`"). This
// pin asserts the error TYPE (not the human-readable message) is the
// `FerroError` contract per CLAUDE.md / R-CODE-2.
// ===========================================================================

/// REQ-ERR-TYPE: `CubicSpline::new` returns `Err(FerroError::InvalidParameter)`
/// — NOT `Err(String)` — on each input scipy rejects with `ValueError`
/// (`scipy/interpolate/_cubic.py:48-65`, `prepare_input`).
#[test]
fn cubic_spline_new_invalid_returns_ferroerror() {
    // (a) x/y length mismatch — scipy `ValueError` (_cubic.py:51-53).
    let length_mismatch =
        CubicSpline::new(&[0.0, 1.0], &[0.0, 1.0, 2.0], BoundaryCondition::Natural);
    assert!(
        matches!(length_mismatch, Err(FerroError::InvalidParameter { .. })),
        "x/y length mismatch must return FerroError::InvalidParameter (scipy ValueError)"
    );

    // (b) fewer than 2 data points — scipy `ValueError` (_cubic.py:50).
    let too_few = CubicSpline::new(&[0.0], &[0.0], BoundaryCondition::Natural);
    assert!(
        matches!(too_few, Err(FerroError::InvalidParameter { .. })),
        "fewer than 2 points must return FerroError::InvalidParameter (scipy ValueError)"
    );

    // (c) non-increasing x — scipy `ValueError` (_cubic.py:65).
    let non_increasing = CubicSpline::new(
        &[0.0, 0.0, 1.0],
        &[0.0, 1.0, 2.0],
        BoundaryCondition::Natural,
    );
    assert!(
        matches!(non_increasing, Err(FerroError::InvalidParameter { .. })),
        "non-increasing x must return FerroError::InvalidParameter (scipy ValueError)"
    );
}

// ===========================================================================
// (2) GREEN GUARDS — the SHIPPED value-parity slices (n >= 4). These MUST PASS
// against the current implementation; they pin the parity contract so a future
// change can't silently regress it.
// ===========================================================================

/// Element-wise closeness helper for the green guards.
fn all_close(actual: &[f64], expected: &[f64], tol: f64) -> bool {
    actual.len() == expected.len()
        && actual
            .iter()
            .zip(expected.iter())
            .all(|(a, e)| (a - e).abs() <= tol)
}

/// REQ-NAT-EVAL: natural-BC eval parity, n>=4 (y = x^2).
/// scipy: `CubicSpline(x,y,bc_type='natural')([0.5,1.5,2.5,3.5])`.
#[test]
fn green_natural_eval_squares() {
    // Live scipy 1.17.1:
    const SCIPY: [f64; 4] = [
        0.33928571428571425,
        2.232142857142857,
        6.232142857142858,
        12.339285714285714,
    ];
    let s = CubicSpline::new(
        &[0.0, 1.0, 2.0, 3.0, 4.0],
        &[0.0, 1.0, 4.0, 9.0, 16.0],
        BoundaryCondition::Natural,
    )
    .unwrap();
    let got: Vec<f64> = [0.5, 1.5, 2.5, 3.5].iter().map(|&q| s.eval(q)).collect();
    assert!(
        all_close(&got, &SCIPY, 1e-10),
        "natural eval: {got:?} vs {SCIPY:?}"
    );
}

/// REQ-NAK-EVAL: not-a-knot-BC eval parity, n>=4 (y = x^2 -> exact squares).
/// scipy: `CubicSpline(x,y,bc_type='not-a-knot')([0.5,1.5,2.5,3.5]) = [0.25,2.25,6.25,12.25]`.
#[test]
fn green_not_a_knot_eval_squares() {
    const SCIPY: [f64; 4] = [0.25, 2.25, 6.25, 12.25];
    let s = CubicSpline::new(
        &[0.0, 1.0, 2.0, 3.0, 4.0],
        &[0.0, 1.0, 4.0, 9.0, 16.0],
        BoundaryCondition::NotAKnot,
    )
    .unwrap();
    let got: Vec<f64> = [0.5, 1.5, 2.5, 3.5].iter().map(|&q| s.eval(q)).collect();
    assert!(
        all_close(&got, &SCIPY, 1e-10),
        "not-a-knot eval: {got:?} vs {SCIPY:?}"
    );
}

/// REQ-NAT-DERIV / REQ-NAK-DERIV: first/second derivative parity, n>=4.
/// scipy: natural `cs(1.5,1)=3.0357142857142856`, `cs(2.5,2)=2.1428571428571423`;
///        not-a-knot `cs(1.5,1)=3.0`, `cs(2.5,2)=2.0`.
#[test]
fn green_derivatives() {
    const SCIPY_NAT_D1_AT_1_5: f64 = 3.0357142857142856;
    const SCIPY_NAT_D2_AT_2_5: f64 = 2.1428571428571423;
    const SCIPY_NAK_D1_AT_1_5: f64 = 3.0;
    const SCIPY_NAK_D2_AT_2_5: f64 = 2.0;

    let x = [0.0, 1.0, 2.0, 3.0, 4.0];
    let y = [0.0, 1.0, 4.0, 9.0, 16.0];

    let nat = CubicSpline::new(&x, &y, BoundaryCondition::Natural).unwrap();
    assert!((nat.derivative(1.5) - SCIPY_NAT_D1_AT_1_5).abs() <= 1e-10);
    assert!((nat.second_derivative(2.5) - SCIPY_NAT_D2_AT_2_5).abs() <= 1e-10);

    let nak = CubicSpline::new(&x, &y, BoundaryCondition::NotAKnot).unwrap();
    assert!((nak.derivative(1.5) - SCIPY_NAK_D1_AT_1_5).abs() <= 1e-10);
    assert!((nak.second_derivative(2.5) - SCIPY_NAK_D2_AT_2_5).abs() <= 1e-10);
}

/// REQ-NAT-INTEG / REQ-NAK-INTEG: definite-integral parity, n>=4.
/// scipy: natural `integrate(0,4)=21.428571428571427`,
///        not-a-knot `integrate(0,4)=21.333333333333336`.
#[test]
fn green_integrate() {
    const SCIPY_NAT_INTEG_0_4: f64 = 21.428571428571427;
    const SCIPY_NAK_INTEG_0_4: f64 = 21.333333333333336;

    let x = [0.0, 1.0, 2.0, 3.0, 4.0];
    let y = [0.0, 1.0, 4.0, 9.0, 16.0];

    let nat = CubicSpline::new(&x, &y, BoundaryCondition::Natural).unwrap();
    assert!((nat.integrate(0.0, 4.0) - SCIPY_NAT_INTEG_0_4).abs() <= 1e-10);

    let nak = CubicSpline::new(&x, &y, BoundaryCondition::NotAKnot).unwrap();
    assert!((nak.integrate(0.0, 4.0) - SCIPY_NAK_INTEG_0_4).abs() <= 1e-10);
}

/// REQ-NAT-EVAL / REQ-NAK-EVAL on a genuinely non-cubic dataset (sin, 7 pts),
/// where the two boundary conditions DIFFER — proves both BCs track scipy and
/// the test isn't accidentally cubic-degenerate.
/// scipy: natural `[0.2939944300893482, 0.425335344554224, -0.9229635816106156]`,
///        not-a-knot `[0.31890041025200067, 0.4272713683778276, -0.9309715276490006]`.
#[test]
fn green_sin_both_boundary_conditions() {
    const SCIPY_NAT: [f64; 3] = [0.2939944300893482, 0.425335344554224, -0.9229635816106156];
    const SCIPY_NAK: [f64; 3] = [0.31890041025200067, 0.4272713683778276, -0.9309715276490006];

    // xs = linspace(0,6,7), ys = sin(xs).
    let xs: Vec<f64> = (0..7).map(|i| i as f64).collect();
    let ys: Vec<f64> = xs.iter().map(|&v| v.sin()).collect();
    let q = [0.3, 2.7, 5.1];

    let nat = CubicSpline::new(&xs, &ys, BoundaryCondition::Natural).unwrap();
    let got_nat: Vec<f64> = q.iter().map(|&v| nat.eval(v)).collect();
    assert!(
        all_close(&got_nat, &SCIPY_NAT, 1e-10),
        "sin natural: {got_nat:?} vs {SCIPY_NAT:?}"
    );

    let nak = CubicSpline::new(&xs, &ys, BoundaryCondition::NotAKnot).unwrap();
    let got_nak: Vec<f64> = q.iter().map(|&v| nak.eval(v)).collect();
    assert!(
        all_close(&got_nak, &SCIPY_NAK, 1e-10),
        "sin not-a-knot: {got_nak:?} vs {SCIPY_NAK:?}"
    );

    // Guard against accidental degeneracy: the two BCs must actually differ.
    assert!(
        got_nat
            .iter()
            .zip(got_nak.iter())
            .any(|(a, b)| (a - b).abs() > 1e-6),
        "natural and not-a-knot must differ on a non-cubic dataset"
    );
}

/// REQ-EXTRAP: extrapolation parity (scipy `extrapolate=True` default — first/
/// last cubic piece is used outside the data range), n>=4.
/// scipy: natural `([-0.5,4.5]) = [-0.33928571428571436, 19.66071428571429]`,
///        not-a-knot `([-0.5,4.5]) = [0.25, 20.25]`.
#[test]
fn green_extrapolation() {
    const SCIPY_NAT: [f64; 2] = [-0.33928571428571436, 19.66071428571429];
    const SCIPY_NAK: [f64; 2] = [0.25, 20.25];

    let x = [0.0, 1.0, 2.0, 3.0, 4.0];
    let y = [0.0, 1.0, 4.0, 9.0, 16.0];
    let q = [-0.5, 4.5];

    let nat = CubicSpline::new(&x, &y, BoundaryCondition::Natural).unwrap();
    let got_nat: Vec<f64> = q.iter().map(|&v| nat.eval(v)).collect();
    assert!(
        all_close(&got_nat, &SCIPY_NAT, 1e-12),
        "natural extrap: {got_nat:?} vs {SCIPY_NAT:?}"
    );

    let nak = CubicSpline::new(&x, &y, BoundaryCondition::NotAKnot).unwrap();
    let got_nak: Vec<f64> = q.iter().map(|&v| nak.eval(v)).collect();
    assert!(
        all_close(&got_nak, &SCIPY_NAK, 1e-12),
        "not-a-knot extrap: {got_nak:?} vs {SCIPY_NAK:?}"
    );
}
