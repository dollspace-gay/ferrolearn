//! Converged-x* GREEN GUARDS for `ferrolearn-numerical/src/optimize.rs`
//! (the `scipy.optimize.minimize` / `minimize_scalar` analog) against the LIVE
//! scipy 1.17.1 oracle. See `.design/numerical/optimize.md` (crosslink unit
//! #1986).
//!
//! ## R-CHAR-3 provenance (expected values are NEVER copied from ferrolearn)
//!
//! Every optimizer here minimizes a CONVEX/smooth objective whose minimizer
//! `x*` is a strict ground truth — the unique minimum of the function, which IS
//! the construction (not a value copied from anywhere). The four ground-truth
//! minima are cross-checked by the live scipy oracle, run from `/tmp`:
//!
//! ```text
//! cd /tmp && python3 -c "
//! import numpy as np
//! from scipy.optimize import minimize, minimize_scalar
//! f=lambda x:0.5*(x[0]**2+2*x[1]**2); g=lambda x:np.array([x[0],2*x[1]])
//! hp=lambda x,p:np.array([p[0],2*p[1]])
//! print('newton-cg', minimize(f,[5.,3.],method='newton-cg',jac=g,hessp=hp).x.tolist())
//! print('trust-ncg', minimize(f,[5.,3.],method='trust-ncg',jac=g,hessp=hp).x.tolist())
//! ros=lambda x:sum(100*(x[1:]-x[:-1]**2)**2+(1-x[:-1])**2)
//! print('trust-ncg rosen', minimize(ros,[-1.2,1.0],method='trust-ncg',
//!       jac=lambda x:np.array([-2*(1-x[0])-400*x[0]*(x[1]-x[0]**2),
//!       200*(x[1]-x[0]**2)]),
//!       hessp=lambda x,p:np.array([
//!         (2-400*(x[1]-x[0]**2)+800*x[0]**2)*p[0]-400*x[0]*p[1],
//!         -400*x[0]*p[0]+200*p[1]])).x.tolist())
//! print('powell quad', minimize(lambda x:(x[0]-1)**2+(x[1]-2)**2,[0.,0.],
//!       method='powell').x.tolist())
//! r=minimize(ros,[-1.2,1.0],method='powell')
//! print('powell rosen', r.x.tolist(), 'f', r.fun)
//! print('bounded quad', minimize_scalar(lambda x:(x-2)**2,bounds=(0,5),
//!       method='bounded').x)
//! print('bounded sin', minimize_scalar(np.sin,bounds=(0,2*np.pi),
//!       method='bounded').x)
//! "
//! ```
//! → newton-cg `[0.0, 0.0]`; trust-ncg `[0.0, 0.0]`; trust-ncg rosen
//!   `[1.0, 1.0]`; powell quad `[1.0, 1.9999999999999998]`; powell rosen
//!   `[1.0000000000000913, 1.0000000000001923]` (`f = 1.79e-26`); bounded quad
//!   `2.0`; bounded sin `4.7123876779707` (≈ `3π/2 = 4.71238898038469`, within
//!   `1.3e-6`).
//!
//! These minimizers are deterministic and version-stable for the converged
//! `x*`, so the installed scipy 1.17.1 is a valid live oracle (the
//! sklearn-1.5.2 / scipy-1.17.1 split is irrelevant — the minimum of
//! `0.5(x0²+2x1²)` is `[0,0]` in every release). The iteration PATH differs
//! from scipy (ferrolearn's optimizers are hand-rolled / textbook NR), so the
//! contract is converged-`x*` parity to a tolerance matching each optimizer's
//! stopping rule, NOT iteration-count or path bit-exactness.

use ferrolearn_numerical::optimize::{NewtonCG, Powell, TrustRegionNCG, brent_bounded};
use ndarray::{Array1, array};

// Oracle ground-truth minimizers (the unique minimum = the construction,
// cross-checked by the live scipy call above; NEVER copied from ferrolearn).
const QUAD_MIN: [f64; 2] = [0.0, 0.0]; // argmin 0.5(x0²+2x1²)
const SHIFTED_QUAD_MIN: [f64; 2] = [1.0, 2.0]; // argmin (x0−1)²+(x1−2)²
const ROSEN_MIN: [f64; 2] = [1.0, 1.0]; // argmin Σ100(x[1:]−x[:-1]²)²+(1−x[:-1])²
const BOUNDED_QUAD_MIN: f64 = 2.0; // argmin (x−2)² on [0,5]
// argmin sin on [0,2π] = 3π/2; scipy's bounded value 4.7123876779707 is within
// 1.3e-6 of this (a 1e-5 guard holds against the TRUE minimum for both).
#[allow(
    clippy::approx_constant,
    reason = "exact true-minimum oracle literal: 3π/2 = argmin sin on [0,2π]"
)]
const BOUNDED_SIN_MIN: f64 = 4.712_388_980_384_69;

// --- objectives (built in Rust, the SAME functions scipy minimized) ---------

/// f(x) = 0.5(x0² + 2 x1²); grad = [x0, 2 x1]; H p = [p0, 2 p1].
fn quad_fun_grad(x: &Array1<f64>) -> (f64, Array1<f64>) {
    let f = 0.5 * (x[0] * x[0] + 2.0 * x[1] * x[1]);
    let g = array![x[0], 2.0 * x[1]];
    (f, g)
}
fn quad_hessp(_x: &Array1<f64>, p: &Array1<f64>) -> Array1<f64> {
    array![p[0], 2.0 * p[1]]
}

/// Rosenbrock f(x) = 100(x1 − x0²)² + (1 − x0)² with analytic grad + Hessian-
/// vector product (for the gradient-based TrustRegionNCG).
fn rosen_fun_grad(x: &Array1<f64>) -> (f64, Array1<f64>) {
    let a = 1.0 - x[0];
    let b = x[1] - x[0] * x[0];
    let f = a * a + 100.0 * b * b;
    let g = array![-2.0 * a - 400.0 * x[0] * b, 200.0 * b];
    (f, g)
}
fn rosen_hessp(x: &Array1<f64>, p: &Array1<f64>) -> Array1<f64> {
    let h00 = 2.0 - 400.0 * (x[1] - x[0] * x[0]) + 800.0 * x[0] * x[0];
    let h01 = -400.0 * x[0];
    let h11 = 200.0;
    array![h00 * p[0] + h01 * p[1], h01 * p[0] + h11 * p[1]]
}

// ---------------------------------------------------------------------------
// REQ-1: NewtonCG converged-x* parity — quadratic from [5,3] → [0,0].
// ---------------------------------------------------------------------------

/// Oracle: `minimize(0.5(x0²+2x1²),[5,3],method='newton-cg',jac=,hessp=).x`
/// = `[0.0, 0.0]` (scipy 1.17.1). ferrolearn must reach the SAME unique minimum.
#[test]
fn newton_cg_quadratic_reaches_scipy_minimum() {
    let r = NewtonCG::new()
        .minimize(quad_fun_grad, quad_hessp, array![5.0, 3.0])
        .expect("NewtonCG should succeed on a convex quadratic");
    assert!(
        r.converged,
        "NewtonCG should converge (n_iter={})",
        r.n_iter
    );
    assert!(
        (r.x[0] - QUAD_MIN[0]).abs() <= 1e-8 && (r.x[1] - QUAD_MIN[1]).abs() <= 1e-8,
        "NewtonCG x={:?} != scipy newton-cg x*={:?} (abs tol 1e-8)",
        r.x.to_vec(),
        QUAD_MIN
    );
    assert!(r.fun <= 1e-12, "fun={} should be ~0 at the minimum", r.fun);
}

// ---------------------------------------------------------------------------
// REQ-2: TrustRegionNCG converged-x* parity — quadratic → [0,0]; Rosenbrock
// → [1,1] (with analytic gradient + hessp).
// ---------------------------------------------------------------------------

/// Oracle: `minimize(...,method='trust-ncg',jac=,hessp=).x = [0.0, 0.0]`.
#[test]
fn trust_ncg_quadratic_reaches_scipy_minimum() {
    let r = TrustRegionNCG::new()
        .minimize(quad_fun_grad, quad_hessp, array![5.0, 3.0])
        .expect("TrustRegionNCG should succeed on a convex quadratic");
    assert!(
        r.converged,
        "TrustRegionNCG should converge (n_iter={})",
        r.n_iter
    );
    assert!(
        (r.x[0] - QUAD_MIN[0]).abs() <= 1e-8 && (r.x[1] - QUAD_MIN[1]).abs() <= 1e-8,
        "TrustRegionNCG x={:?} != scipy trust-ncg x*={:?} (abs tol 1e-8)",
        r.x.to_vec(),
        QUAD_MIN
    );
    assert!(r.fun <= 1e-12, "fun={} should be ~0 at the minimum", r.fun);
}

/// Oracle: `minimize(rosenbrock,[-1.2,1.0],method='trust-ncg',jac=,hessp=).x`
/// = `[1.0, 1.0]`. TrustRegionNCG is gradient-based, so provide the analytic
/// Rosenbrock gradient + Hessian-vector product.
#[test]
fn trust_ncg_rosenbrock_reaches_scipy_minimum() {
    let r = TrustRegionNCG::new()
        .with_max_iter(2000)
        .with_tol(1e-10)
        .minimize(rosen_fun_grad, rosen_hessp, array![-1.2, 1.0])
        .expect("TrustRegionNCG should succeed on Rosenbrock");
    assert!(
        (r.x[0] - ROSEN_MIN[0]).abs() <= 1e-4 && (r.x[1] - ROSEN_MIN[1]).abs() <= 1e-4,
        "TrustRegionNCG x={:?} != scipy trust-ncg Rosenbrock x*={:?} (abs tol 1e-4)",
        r.x.to_vec(),
        ROSEN_MIN
    );
}

// ---------------------------------------------------------------------------
// REQ-3: Powell converged-x* parity (DERIVATIVE-FREE — no gradient/hessp) —
// shifted quadratic → [1,2]; Rosenbrock → [1,1]. This is the FRAGILE case.
// ---------------------------------------------------------------------------

/// Oracle: `minimize((x0−1)²+(x1−2)²,[0,0],method='powell').x`
/// = `[1.0, 1.9999999999999998]` → unique minimum `[1,2]`.
#[test]
fn powell_shifted_quadratic_reaches_scipy_minimum() {
    let f = |x: &Array1<f64>| (x[0] - 1.0).powi(2) + (x[1] - 2.0).powi(2);
    let r = Powell::new().minimize(f, array![0.0, 0.0]);
    assert!(r.converged, "Powell should converge (n_iter={})", r.n_iter);
    assert!(
        (r.x[0] - SHIFTED_QUAD_MIN[0]).abs() <= 1e-5
            && (r.x[1] - SHIFTED_QUAD_MIN[1]).abs() <= 1e-5,
        "Powell x={:?} != scipy powell x*={:?} (abs tol 1e-5)",
        r.x.to_vec(),
        SHIFTED_QUAD_MIN
    );
}

/// Oracle: `minimize(rosenbrock,[-1.2,1.0],method='powell').x`
/// = `[1.0000000000000913, 1.0000000000001923]` with `fun = 1.79e-26` → unique
/// minimum `[1,1]`. The FRAGILE case (REQ-3, least-confident): verify Powell
/// actually REACHES `[1,1]` and drives `fun` to ~0 without a gradient.
#[test]
fn powell_rosenbrock_reaches_scipy_minimum() {
    let ros = |x: &Array1<f64>| {
        let b = x[1] - x[0] * x[0];
        (1.0 - x[0]).powi(2) + 100.0 * b * b
    };
    let r = Powell::new().minimize(ros, array![-1.2, 1.0]);
    assert!(
        r.converged,
        "Powell should converge on Rosenbrock (n_iter={})",
        r.n_iter
    );
    assert!(
        (r.x[0] - ROSEN_MIN[0]).abs() <= 1e-4 && (r.x[1] - ROSEN_MIN[1]).abs() <= 1e-4,
        "Powell x={:?} != scipy powell Rosenbrock x*={:?} (abs tol 1e-4)",
        r.x.to_vec(),
        ROSEN_MIN
    );
    assert!(
        r.fun < 1e-8,
        "Powell Rosenbrock fun={} should be < 1e-8 (scipy reaches 1.79e-26)",
        r.fun
    );
}

// ---------------------------------------------------------------------------
// REQ-4: brent_bounded converged-x* parity — (x−2)² on [0,5] → 2.0;
// sin on [0,2π] → 3π/2.
// ---------------------------------------------------------------------------

/// Oracle: `minimize_scalar((x−2)²,bounds=(0,5),method='bounded').x = 2.0`.
#[test]
fn brent_bounded_quadratic_reaches_scipy_minimum() {
    let r = brent_bounded(|x| (x - 2.0) * (x - 2.0), 0.0, 5.0, 1e-8, 500);
    assert!(r.success, "brent_bounded should converge");
    assert!(
        (r.x - BOUNDED_QUAD_MIN).abs() <= 1e-6,
        "brent_bounded x={} != scipy bounded x*={} (abs tol 1e-6)",
        r.x,
        BOUNDED_QUAD_MIN
    );
}

/// Oracle: `minimize_scalar(sin,bounds=(0,2π),method='bounded').x`
/// = `4.7123876779707` ≈ `3π/2 = 4.71238898038469` (the TRUE unique minimum on
/// the interval). Guard against the closed-form minimum within 1e-5 (scipy's
/// own value is within 1.3e-6 of 3π/2).
#[test]
fn brent_bounded_sin_reaches_scipy_minimum() {
    let two_pi = 2.0 * std::f64::consts::PI;
    let r = brent_bounded(f64::sin, 0.0, two_pi, 1e-8, 500);
    assert!(r.success, "brent_bounded should converge");
    assert!(
        (r.x - BOUNDED_SIN_MIN).abs() <= 1e-5,
        "brent_bounded x={} != true min 3π/2={} (scipy bounded 4.7123876779707; abs tol 1e-5)",
        r.x,
        BOUNDED_SIN_MIN
    );
    assert!(
        (r.fun - (-1.0)).abs() <= 1e-8,
        "sin at the minimum should be -1, got {}",
        r.fun
    );
}
