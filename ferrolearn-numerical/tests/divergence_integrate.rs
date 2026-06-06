//! Divergence audit + GREEN GUARDS for `ferrolearn-numerical/src/integrate.rs`
//! (the `scipy.integrate` quadrature analog) against the LIVE scipy 1.17.1 /
//! numpy 2.4.5 oracle. See `.design/numerical/integrate.md` (crosslink unit
//! #1971).
//!
//! All expected values are computed by a live scipy/numpy call run from `/tmp`
//! (R-CHAR-3) and are NEVER copied from the ferrolearn side. The `n`-point
//! Gauss-Legendre rule is UNIQUE, so `numpy.polynomial.legendre.leggauss`
//! (= scipy's `fixed_quad` rule) is the exact ground truth for the GL guards;
//! `scipy.integrate.quad` is the value-to-tolerance ground truth for `quad`.
//!
//! The single oracle command that produced the per-order `GL_EXP` / `GL_COS`
//! constants below:
//!
//! ```text
//! cd /tmp && python3 -c "
//! import numpy as np
//! def gl(f,a,b,n):
//!     x,w=np.polynomial.legendre.leggauss(n); xm=0.5*(b-a)*x+0.5*(b+a)
//!     return 0.5*(b-a)*np.sum(w*f(xm))
//! print([repr(float(gl(np.exp,0,1,n))) for n in range(1,21)])
//! print([repr(float(gl(np.cos,0,2,n))) for n in range(1,21)])
//! "
//! ```
//!
//! And for the `quad` value-convergence guards:
//!
//! ```text
//! cd /tmp && python3 -c "
//! import numpy as np, scipy.integrate as si
//! print(si.quad(np.exp,0,1)[0])             # 1.7182818284590453  (= e - 1)
//! print(si.quad(np.sin,0,np.pi)[0])         # 2.0
//! print(si.quad(lambda x:1/(1+x*x),0,1)[0]) # 0.7853981633974484  (= pi/4)
//! "
//! ```
//!
//! scipy upstream surface: `scipy/integrate/__init__.py` (`quad` :14,
//! `fixed_quad` :21). The numerical kernels are QUADPACK (Fortran), so the
//! oracle is the documented function NAMES + live values, never Fortran lines.
//!
//! VERDICT (from this audit): all 20 GL orders (hardcoded tables 1-10 AND
//! Golub-Welsch 11-20), the composite rule, and `quad` value-convergence MATCH
//! the live oracle. NO numerical divergence found — these are GREEN guards, not
//! red pins. The structural gaps (REQ-4..REQ-9: API/return contract, missing
//! functions, infinite bounds, `FerroError`, no consumer, ferray substrate) are
//! filed as `-l blocker` issues, not pinned as failing tests (they are
//! structural facts, not numerical assertions).

use ferrolearn_core::error::FerroError;
use ferrolearn_numerical::integrate::{gauss_legendre, gauss_legendre_composite, quad};

/// Exact `n`-point Gauss-Legendre rule for `∫₀¹ eˣ dx`, n = 1..=20, computed
/// live by `numpy.polynomial.legendre.leggauss` (R-CHAR-3, NEVER from
/// ferrolearn). `GL_EXP[k]` is the `(k+1)`-point rule.
const GL_EXP: [f64; 20] = [
    1.6487212707001282,
    1.717896378007504,
    1.718281004372522,
    1.7182818275260776,
    1.7182818284583914,
    1.718281828459045,
    1.7182818284590449,
    1.7182818284590453,
    1.7182818284590453,
    1.718281828459045,
    1.718281828459045,
    1.718281828459045,
    1.7182818284590453,
    1.7182818284590455,
    1.718281828459045,
    1.718281828459045,
    1.7182818284590453,
    1.7182818284590446,
    1.7182818284590449,
    1.7182818284590449,
];

/// Exact `n`-point Gauss-Legendre rule for `∫₀² cos x dx`, n = 1..=20, computed
/// live by `numpy.polynomial.legendre.leggauss` (R-CHAR-3). `GL_COS[k]` is the
/// `(k+1)`-point rule.
const GL_COS: [f64; 20] = [
    1.0806046117362795,
    0.9054513852355843,
    0.9093306976211131,
    0.9092972750440557,
    0.9092974272532763,
    0.9092974268248637,
    0.9092974268256828,
    0.9092974268256815,
    0.9092974268256817,
    0.9092974268256815,
    0.9092974268256817,
    0.9092974268256816,
    0.9092974268256819,
    0.9092974268256822,
    0.9092974268256817,
    0.9092974268256818,
    0.9092974268256816,
    0.9092974268256823,
    0.9092974268256815,
    0.9092974268256825,
];

// ===========================================================================
// (1) GREEN GUARD — REQ-1: gauss_legendre value parity across ALL 20 orders.
//
// The unique n-point GL rule has a strict ground truth (numpy `leggauss`).
// This guard exercises BOTH the hardcoded order-1..10 tables (`gl_table`) AND
// the order-11..20 Golub-Welsch path (`golub_welsch` -> `trid_qr_eigen`). A
// wrong table constant or a wrong Golub-Welsch eigenvalue would be a
// single-file-fixable divergence and would trip this guard (it would then be
// the RED pin). Observed: every order matches the leggauss rule to < 1e-12.
// ===========================================================================

/// Green guard: `gauss_legendre(exp, 0, 1, n)` equals the exact `n`-point
/// leggauss rule `GL_EXP[n-1]` for every `n ∈ 1..=20` (tables + Golub-Welsch).
/// Mirrors `scipy.integrate.fixed_quad` / numpy `leggauss`.
#[test]
fn gauss_legendre_exp_all_orders_match_leggauss() {
    for n in 1..=20usize {
        let got = gauss_legendre(|x: f64| x.exp(), 0.0, 1.0, n)
            .expect("orders 1..=20 are valid")
            .value;
        let oracle = GL_EXP[n - 1];
        let diff = (got - oracle).abs();
        assert!(
            diff <= 1e-12,
            "GL exp order n={n}: ferrolearn={got:.17e} leggauss-oracle={oracle:.17e} diff={diff:.3e} > 1e-12"
        );
    }
}

/// Green guard: `gauss_legendre(cos, 0, 2, n)` equals the exact `n`-point
/// leggauss rule `GL_COS[n-1]` for every `n ∈ 1..=20` (a second integrand so
/// the parity is not artifact of a single function). Mirrors
/// `scipy.integrate.fixed_quad` / numpy `leggauss`.
#[test]
fn gauss_legendre_cos_all_orders_match_leggauss() {
    for n in 1..=20usize {
        let got = gauss_legendre(|x: f64| x.cos(), 0.0, 2.0, n)
            .expect("orders 1..=20 are valid")
            .value;
        let oracle = GL_COS[n - 1];
        let diff = (got - oracle).abs();
        assert!(
            diff <= 1e-12,
            "GL cos order n={n}: ferrolearn={got:.17e} leggauss-oracle={oracle:.17e} diff={diff:.3e} > 1e-12"
        );
    }
}

// ===========================================================================
// (2) GREEN GUARD — REQ-2: gauss_legendre_composite.
//
// 10 panels of the exact 5-point GL rule over [0, pi] for sin x. True value
// ∫₀^π sin x dx = 2.0 (oracle `python3 -c "import numpy as np; print(2.0)"`,
// also si.quad(np.sin,0,np.pi)[0] = 2.0). Composite high-order GL is exact for
// smooth integrands to machine precision.
// ===========================================================================

/// Green guard: `gauss_legendre_composite(sin, 0, π, 5, 10)` equals the true
/// `∫₀^π sin x dx = 2.0` to ~1e-12. Mirrors a composite `fixed_quad`.
#[test]
fn gauss_legendre_composite_sin_matches_true_two() {
    // True value: integral of sin over [0, pi] = 2.0 (= si.quad(np.sin,0,np.pi)[0]).
    const TRUE_SIN_INTEGRAL: f64 = 2.0;
    let got = gauss_legendre_composite(|x: f64| x.sin(), 0.0, std::f64::consts::PI, 5, 10)
        .expect("5 points, 10 panels are valid")
        .value;
    let diff = (got - TRUE_SIN_INTEGRAL).abs();
    assert!(
        diff <= 1e-12,
        "composite GL sin: ferrolearn={got:.17e} oracle=2.0 diff={diff:.3e} > 1e-12"
    );
}

// ===========================================================================
// (3) GREEN GUARD — REQ-3: quad value-convergence.
//
// Adaptive Simpson `quad` uses a DIFFERENT algorithm than scipy's QUADPACK
// Gauss-Kronrod, so the contract is value-to-tolerance: it converges to the
// SAME true integral that `scipy.integrate.quad` returns, within the requested
// tol. NOT bit-exact. Oracle values are the live `si.quad(...)[0]`:
//   si.quad(np.exp,0,1)[0]             = 1.7182818284590453  (= e - 1)
//   si.quad(np.sin,0,np.pi)[0]         = 2.0
//   si.quad(lambda x:1/(1+x*x),0,1)[0] = 0.7853981633974484  (= pi/4)
// ===========================================================================

/// Green guard: `quad(exp, 0, 1, 1e-10)` converges to `si.quad(np.exp,0,1)[0]`
/// (= e - 1) within ~1e-9.
#[test]
fn quad_exp_converges_to_scipy_value() {
    // Live oracle: si.quad(np.exp,0,1)[0] = 1.7182818284590453 (= e - 1).
    const SCIPY_QUAD_EXP: f64 = 1.7182818284590453;
    let got = quad(|x: f64| x.exp(), 0.0, 1.0, 1e-10).value;
    let diff = (got - SCIPY_QUAD_EXP).abs();
    assert!(
        diff <= 1e-9,
        "quad exp: ferrolearn={got:.17e} scipy-oracle={SCIPY_QUAD_EXP:.17e} diff={diff:.3e} > 1e-9"
    );
}

/// Green guard: `quad(sin, 0, π, 1e-10)` converges to `si.quad(np.sin,0,π)[0]`
/// (= 2.0) within ~1e-9.
#[test]
fn quad_sin_converges_to_scipy_value() {
    // Live oracle: si.quad(np.sin,0,np.pi)[0] = 2.0.
    const SCIPY_QUAD_SIN: f64 = 2.0;
    let got = quad(|x: f64| x.sin(), 0.0, std::f64::consts::PI, 1e-10).value;
    let diff = (got - SCIPY_QUAD_SIN).abs();
    assert!(
        diff <= 1e-9,
        "quad sin: ferrolearn={got:.17e} scipy-oracle={SCIPY_QUAD_SIN:.17e} diff={diff:.3e} > 1e-9"
    );
}

/// Green guard: `quad(1/(1+x²), 0, 1, 1e-10)` converges to
/// `si.quad(lambda x:1/(1+x*x),0,1)[0]` (= π/4) within ~1e-9.
#[test]
#[allow(
    clippy::approx_constant,
    reason = "exact scipy/true-integral oracle literal pi/4 (R-CHAR-3)"
)]
fn quad_arctan_converges_to_scipy_value() {
    // Live oracle: si.quad(lambda x:1/(1+x*x),0,1)[0] = 0.7853981633974484 (= pi/4).
    const SCIPY_QUAD_ARCTAN: f64 = 0.7853981633974484;
    let got = quad(|x: f64| 1.0 / (1.0 + x * x), 0.0, 1.0, 1e-10).value;
    let diff = (got - SCIPY_QUAD_ARCTAN).abs();
    assert!(
        diff <= 1e-9,
        "quad arctan: ferrolearn={got:.17e} scipy-oracle={SCIPY_QUAD_ARCTAN:.17e} diff={diff:.3e} > 1e-9"
    );
}

// ===========================================================================
// (4) GREEN GUARD — REQ-7: error type is `FerroError`.
//
// scipy raises `ValueError` on an invalid quadrature parameter (e.g. an
// invalid Gauss order or panel count). The ferrolearn library error contract
// (CLAUDE.md / R-CODE-2) expresses that as `FerroError::InvalidParameter`, not
// the previous `Result<_, String>`. These guards pin the error TYPE.
// ===========================================================================

/// Green guard: an out-of-range Gauss-Legendre order (`0` and `21`, both
/// outside the supported `1..=20`) returns `FerroError::InvalidParameter`,
/// mirroring scipy's `ValueError` on an invalid quadrature order.
#[test]
fn gauss_legendre_invalid_order_returns_ferroerror() {
    let too_small = gauss_legendre(|x: f64| x, 0.0, 1.0, 0);
    assert!(
        matches!(too_small, Err(FerroError::InvalidParameter { .. })),
        "gauss_legendre order 0 should be Err(FerroError::InvalidParameter), got {too_small:?}"
    );

    let too_large = gauss_legendre(|x: f64| x, 0.0, 1.0, 21);
    assert!(
        matches!(too_large, Err(FerroError::InvalidParameter { .. })),
        "gauss_legendre order 21 should be Err(FerroError::InvalidParameter), got {too_large:?}"
    );
}

/// Green guard: `gauss_legendre_composite` with `n_panels = 0` returns
/// `FerroError::InvalidParameter`, mirroring scipy's `ValueError` on an
/// invalid quadrature parameter.
#[test]
fn gauss_legendre_composite_invalid_npanels_returns_ferroerror() {
    // Signature: gauss_legendre_composite(f, a, b, n_points, n_panels).
    let got = gauss_legendre_composite(|x: f64| x, 0.0, 1.0, 5, 0);
    assert!(
        matches!(got, Err(FerroError::InvalidParameter { .. })),
        "gauss_legendre_composite n_panels=0 should be Err(FerroError::InvalidParameter), got {got:?}"
    );
}
