//! Divergence audit for `ferrolearn-numerical/src/special.rs` against the LIVE
//! scipy.special oracle (scipy 1.17.1). See `.design/numerical/special.md`.
//!
//! All expected values are computed by a live scipy call run from `/tmp`
//! (R-CHAR-3) and are NEVER copied from the ferrolearn side:
//!
//! ```text
//! cd /tmp && python3 -c "import scipy.special as sp; import numpy as np; \
//!   print(repr(sp.erf(0.5)), repr(sp.erf(1.0)), repr(sp.erf(2.0))); \
//!   print(repr(sp.erfc(1.0)), repr(sp.erfc(2.0))); \
//!   print(repr(sp.erf(np.inf)), repr(sp.erfc(np.inf))); \
//!   print(repr(sp.gamma(5.5)), repr(sp.gamma(20.0)), repr(sp.gamma(0.5)), \
//!         repr(sp.gamma(-0.5)), repr(sp.gamma(0.1)), repr(sp.gamma(10.0))); \
//!   print(repr(sp.gammaln(0.5)), repr(sp.gammaln(1.5)), repr(sp.gammaln(12.0)), \
//!         repr(sp.gammaln(20.0)), repr(sp.gammaln(-0.5))); \
//!   print(repr(sp.digamma(0.5)), repr(sp.digamma(1.0)), repr(sp.digamma(2.0)), \
//!         repr(sp.digamma(10.0)), repr(sp.digamma(-1.5))); \
//!   print(repr(sp.beta(2,3)), repr(sp.beta(0.5,0.5)), repr(sp.beta(2,5)), \
//!         repr(sp.betaln(2,3))); \
//!   print(repr(sp.gamma(0.0)), repr(sp.gamma(-1.0)), repr(sp.gammaln(0.0)), \
//!         repr(sp.gammaln(-1.0)))"
//! ```

use ferrolearn_numerical::special::{beta, digamma, erf, erfc, gamma, lbeta, lgamma};

/// Relative-or-absolute closeness used by the green guards.
fn rel_close(actual: f64, expected: f64, tol: f64) -> bool {
    if actual.is_nan() || expected.is_nan() {
        return false;
    }
    let abs = (actual - expected).abs();
    if expected == 0.0 {
        return abs <= tol;
    }
    (abs / expected.abs()) <= tol
}

// ===========================================================================
// (1) RED PIN — REQ-5: erf/erfc accuracy divergence (single-file-fixable).
//
// ferrolearn's `erf` uses the Abramowitz & Stegun 7.1.26 5-coefficient
// polynomial (max error ~1.4e-7). scipy.special.erf is machine-precision.
// Fix: delegate to `statrs::function::erf::{erf, erfc}` (statrs 0.18 is
// already a ferrolearn-numerical dependency; used in gp_classifier.rs).
//
// Live scipy oracle (scipy 1.17.1, from /tmp):
//   sp.erf(0.5) = 0.5204998778130465
//   sp.erf(1.0) = 0.8427007929497148
//   sp.erf(2.0) = 0.9953222650189527
// ferrolearn erf(1.0) ~ 0.8427006897475899 (abs diff ~1.03e-7) -> FAILS 1e-12.
// ===========================================================================

/// Divergence: `special::erf` diverges from `scipy.special.erf`
/// (A&S 7.1.26 ~1.4e-7 error vs machine precision).
/// scipy erf(1.0)=0.8427007929497148; ferrolearn ~0.8427006897475899.
/// Tracking: #1942
#[test]
fn divergence_erf_accuracy() {
    // Live scipy oracle values (R-CHAR-3) — computed via the command above.
    const SP_ERF_05: f64 = 0.520_499_877_813_046_5;
    const SP_ERF_10: f64 = 0.842_700_792_949_714_8;
    const SP_ERF_20: f64 = 0.995_322_265_018_952_7;
    const TOL: f64 = 1e-12;

    assert!(
        rel_close(erf(0.5), SP_ERF_05, TOL),
        "erf(0.5)={} expected scipy {SP_ERF_05} (abs diff {:e})",
        erf(0.5),
        (erf(0.5) - SP_ERF_05).abs()
    );
    assert!(
        rel_close(erf(1.0), SP_ERF_10, TOL),
        "erf(1.0)={} expected scipy {SP_ERF_10} (abs diff {:e})",
        erf(1.0),
        (erf(1.0) - SP_ERF_10).abs()
    );
    assert!(
        rel_close(erf(2.0), SP_ERF_20, TOL),
        "erf(2.0)={} expected scipy {SP_ERF_20} (abs diff {:e})",
        erf(2.0),
        (erf(2.0) - SP_ERF_20).abs()
    );
}

/// Divergence: `special::erfc` diverges from `scipy.special.erfc`
/// (same A&S 7.1.26 ~1.4e-7 error). scipy erfc(2.0)=0.004677734981047266.
/// Tracking: #1942
#[test]
fn divergence_erfc_accuracy() {
    // Live scipy oracle values (R-CHAR-3).
    const SP_ERFC_10: f64 = 0.157_299_207_050_285_16;
    const SP_ERFC_20: f64 = 0.004_677_734_981_047_266;
    const TOL: f64 = 1e-12;

    assert!(
        rel_close(erfc(1.0), SP_ERFC_10, TOL),
        "erfc(1.0)={} expected scipy {SP_ERFC_10} (abs diff {:e})",
        erfc(1.0),
        (erfc(1.0) - SP_ERFC_10).abs()
    );
    assert!(
        rel_close(erfc(2.0), SP_ERFC_20, TOL),
        "erfc(2.0)={} expected scipy {SP_ERFC_20} (abs diff {:e})",
        erfc(2.0),
        (erfc(2.0) - SP_ERFC_20).abs()
    );
}

// ===========================================================================
// (2) RED PIN — REQ-6: pole / edge handling divergence (single-file-fixable).
//
// Live scipy oracle (scipy 1.17.1, from /tmp):
//   sp.gamma(0.0)    = inf
//   sp.gamma(-1.0)   = nan
//   sp.gammaln(0.0)  = inf
//   sp.gammaln(-1.0) = inf
//
// ferrolearn:
//   gamma(0.0)    = inf  (MATCH — PI/0.0)
//   gamma(-1.0)   ~ -2.565e16  (DIVERGES: sin(-pi) is the tiny nonzero float
//                               -1.2246e-16, so the reflection produces a huge
//                               FINITE garbage value instead of nan)
//   lgamma(0.0)   = nan  (DIVERGES from +inf)
//   lgamma(-1.0)  = nan  (DIVERGES from +inf)
//
// Fix: a clean single-file non-positive-integer pole check — `gamma` returns
// nan at negative integers; `lgamma` returns +inf at non-positive integers.
// ===========================================================================

/// Divergence: `special::gamma(-1.0)` returns a huge finite value where
/// `scipy.special.gamma(-1.0)` is `nan` (negative-integer pole).
/// Tracking: #1943
#[test]
fn divergence_gamma_negative_integer_pole_is_nan() {
    // scipy.special.gamma(-1.0) == nan (live oracle).
    let g = gamma(-1.0);
    assert!(
        g.is_nan(),
        "gamma(-1.0)={g:e} expected scipy nan (negative-integer pole)"
    );
    let g2 = gamma(-2.0);
    assert!(
        g2.is_nan(),
        "gamma(-2.0)={g2:e} expected scipy nan (negative-integer pole)"
    );
}

/// Divergence: `special::lgamma` returns `nan` at non-positive integers where
/// `scipy.special.gammaln` returns `+inf`.
/// scipy gammaln(0.0)=inf, gammaln(-1.0)=inf, gammaln(-5.0)=inf (live oracle).
/// Tracking: #1943
#[test]
fn divergence_lgamma_nonpositive_integer_pole_is_pos_inf() {
    for x in [0.0_f64, -1.0, -5.0] {
        let lg = lgamma(x);
        assert!(
            lg.is_infinite() && lg > 0.0,
            "lgamma({x})={lg} expected scipy +inf (non-positive-integer pole)"
        );
    }
}

// ===========================================================================
// (3) GREEN GUARDS — REQ-1..4: SHIPPED functions must match the live oracle.
// All expected values from the live scipy oracle (R-CHAR-3).
// ===========================================================================

/// REQ-1: `special::gamma` matches `scipy.special.gamma` to rel <= 1e-12.
#[test]
fn green_gamma_matches_scipy() {
    // Live scipy oracle (sp.gamma(...)).
    let cases = [
        (1.0_f64, 1.0_f64),
        (2.0, 1.0),
        (3.0, 2.0),
        (4.0, 6.0),
        (5.0, 24.0),
        (5.5, 52.342_777_784_553_52),
        (10.0, 362_880.0),
        (20.0, 1.216_451_004_088_32e17),
        (0.5, 1.772_453_850_905_515_9),
        (1.5, 0.886_226_925_452_758),
        (-0.5, -3.544_907_701_811_031_8),
        (0.1, 9.513_507_698_668_732),
    ];
    for (x, expected) in cases {
        assert!(
            rel_close(gamma(x), expected, 1e-12),
            "gamma({x})={} expected scipy {expected} (rel {:e})",
            gamma(x),
            ((gamma(x) - expected) / expected).abs()
        );
    }
}

/// REQ-2: `special::lgamma` matches `scipy.special.gammaln` to rel <= 1e-12.
#[test]
fn green_lgamma_matches_scipy() {
    // Live scipy oracle (sp.gammaln(...)).
    let cases = [
        (0.5_f64, 0.572_364_942_924_7_f64),
        (1.5, -0.120_782_237_635_245_26),
        (12.0, 17.502_307_845_873_887),
        (20.0, 39.339_884_187_199_495),
        (-0.5, 1.265_512_123_484_645_4),
    ];
    for (x, expected) in cases {
        assert!(
            rel_close(lgamma(x), expected, 1e-12),
            "lgamma({x})={} expected scipy {expected} (rel {:e})",
            lgamma(x),
            ((lgamma(x) - expected) / expected).abs()
        );
    }
}

/// REQ-3: `special::digamma` matches `scipy.special.digamma` (= psi).
/// The design doc notes a ~2.1e-11 worst-case floor from the 5-term Bernoulli
/// truncation; we use a 1e-10 contract (NOTE: looser than the others by design).
#[test]
fn green_digamma_matches_scipy() {
    // Live scipy oracle (sp.digamma(...)).
    let cases = [
        (0.5_f64, -1.963_510_026_021_423_5_f64),
        (1.0, -0.577_215_664_901_532_9),
        (2.0, 0.422_784_335_098_467_13),
        (10.0, 2.251_752_589_066_721),
        (-1.5, 0.703_156_640_645_243_4),
    ];
    for (x, expected) in cases {
        assert!(
            rel_close(digamma(x), expected, 1e-10),
            "digamma({x})={} expected scipy {expected} (rel {:e})",
            digamma(x),
            ((digamma(x) - expected) / expected).abs()
        );
    }
}

/// REQ-4: `special::beta`/`lbeta` match `scipy.special.beta`/`betaln`.
#[test]
fn green_beta_matches_scipy() {
    // Live scipy oracle (sp.beta / sp.betaln).
    let beta_cases = [
        (2.0_f64, 3.0_f64, 0.083_333_333_333_333_33_f64),
        (0.5, 0.5, 3.141_592_653_589_792_7),
        (2.0, 5.0, 0.033_333_333_333_333_33),
    ];
    for (a, b, expected) in beta_cases {
        assert!(
            rel_close(beta(a, b), expected, 1e-12),
            "beta({a},{b})={} expected scipy {expected} (rel {:e})",
            beta(a, b),
            ((beta(a, b) - expected) / expected).abs()
        );
    }
    // scipy.special.betaln(2,3) = -2.4849066497880004 (live oracle).
    const SP_BETALN_23: f64 = -2.484_906_649_788_000_4;
    assert!(
        rel_close(lbeta(2.0, 3.0), SP_BETALN_23, 1e-12),
        "lbeta(2,3)={} expected scipy {SP_BETALN_23}",
        lbeta(2.0, 3.0)
    );
}

/// REQ-6 (partial GREEN): the edge values ferrolearn already matches —
/// erf(inf)=1, erfc(inf)=0, gamma(0.0)=+inf (live scipy oracle).
#[test]
fn green_erf_inf_and_gamma_zero_pole() {
    // sp.erf(inf)=1.0, sp.erfc(inf)=0.0, sp.gamma(0.0)=inf (live oracle).
    assert!(
        rel_close(erf(f64::INFINITY), 1.0, 1e-12),
        "erf(inf)={}",
        erf(f64::INFINITY)
    );
    assert!(
        rel_close(erf(10.0), 1.0, 1e-9),
        "erf(10.0)={} expected ~1.0",
        erf(10.0)
    );
    assert!(
        erfc(f64::INFINITY).abs() <= 1e-12,
        "erfc(inf)={}",
        erfc(f64::INFINITY)
    );
    let g0 = gamma(0.0);
    assert!(
        g0.is_infinite() && g0 > 0.0,
        "gamma(0.0)={g0} expected scipy +inf"
    );
}

// ===========================================================================
// (4) RE-AUDIT of #1942 (libm erf/erfc) + #1943 (gamma/lgamma poles) over a
//     DENSER range, against the LIVE scipy 1.17 oracle (R-CHAR-3). Run from
//     /tmp:
//
//   python3 -c "import scipy.special as sp; \
//     print([repr(float(sp.erf(x))) for x in [-3.0,-1.5,-0.3,0.3,1.5,3.0,6.0]]); \
//     print([repr(float(sp.erfc(x))) for x in [6.0,10.0]]); \
//     print([repr(float(sp.gamma(x))) for x in [-2.0,-3.0,-10.0,-2.5,-3.5,171.0]]); \
//     print([repr(float(sp.gammaln(x))) for x in [-2.0,-3.0,-10.0,-2.5,100.0]])"
// ===========================================================================

/// #1942 re-audit: `special::erf` matches `scipy.special.erf` across negative,
/// small, moderate, and large x to rel/abs <= 1e-12 (odd symmetry included).
#[test]
fn green_erf_range_matches_scipy() {
    // Live scipy oracle (sp.erf(...)) — NEVER copied from ferrolearn.
    let cases = [
        (-3.0_f64, -0.999_977_909_503_001_4_f64),
        (-1.5, -0.966_105_146_475_310_8),
        (-0.3, -0.328_626_759_459_127_4),
        (0.3, 0.328_626_759_459_127_4),
        (1.5, 0.966_105_146_475_310_8),
        (3.0, 0.999_977_909_503_001_4),
        (6.0, 1.0),
    ];
    for (x, expected) in cases {
        assert!(
            rel_close(erf(x), expected, 1e-12),
            "erf({x})={} expected scipy {expected} (diff {:e})",
            erf(x),
            (erf(x) - expected).abs()
        );
    }
    // Odd symmetry: erf(-x) == -erf(x) at machine precision.
    for x in [0.3_f64, 1.5, 3.0] {
        assert!(
            (erf(-x) + erf(x)).abs() <= 1e-15,
            "erf odd symmetry broken at x={x}: erf(-x)={}, erf(x)={}",
            erf(-x),
            erf(x)
        );
    }
}

/// #1942 re-audit: `special::erfc` matches `scipy.special.erfc` in the TAIL,
/// where values are ~2e-17 / ~2e-45. Uses RELATIVE tolerance to confirm libm
/// does not flush the tail to zero prematurely.
#[test]
fn green_erfc_tail_matches_scipy() {
    // Live scipy oracle (sp.erfc(...)).
    const SP_ERFC_6: f64 = 2.151_973_671_249_891_3e-17;
    const SP_ERFC_10: f64 = 2.088_487_583_762_544_6e-45;
    assert!(
        erfc(6.0) > 0.0 && rel_close(erfc(6.0), SP_ERFC_6, 1e-12),
        "erfc(6.0)={:e} expected scipy {SP_ERFC_6:e} (rel {:e})",
        erfc(6.0),
        ((erfc(6.0) - SP_ERFC_6) / SP_ERFC_6).abs()
    );
    assert!(
        erfc(10.0) > 0.0 && rel_close(erfc(10.0), SP_ERFC_10, 1e-12),
        "erfc(10.0)={:e} expected scipy {SP_ERFC_10:e} (rel {:e})",
        erfc(10.0),
        ((erfc(10.0) - SP_ERFC_10) / SP_ERFC_10).abs()
    );
}

/// #1943 re-audit: `special::gamma` is `nan` at MULTIPLE negative-integer poles
/// (scipy returns nan), and matches scipy's FINITE reflection values at
/// negative NON-integers near those poles.
#[test]
fn green_gamma_poles_and_reflection_matches_scipy() {
    // scipy.special.gamma(-2.0)=-3.0=-10.0 all == nan (live oracle).
    for x in [-2.0_f64, -3.0, -10.0] {
        assert!(
            gamma(x).is_nan(),
            "gamma({x})={:e} expected scipy nan (negative-integer pole)",
            gamma(x)
        );
    }
    // Negative non-integers go through the reflection path (NOT the nan guard):
    // live scipy oracle values.
    let refl = [
        (-2.5_f64, -0.945_308_720_482_941_9_f64),
        (-3.5, 0.270_088_205_852_269_06),
    ];
    for (x, expected) in refl {
        assert!(
            rel_close(gamma(x), expected, 1e-12),
            "gamma({x})={} expected scipy {expected} (rel {:e})",
            gamma(x),
            ((gamma(x) - expected) / expected).abs()
        );
    }
}

/// #1943 re-audit: `special::lgamma` is `+inf` at MULTIPLE non-positive-integer
/// poles (scipy gammaln=inf), matches scipy at negative non-integers, and
/// matches scipy at large x.
#[test]
fn green_lgamma_poles_reflection_and_large_matches_scipy() {
    // scipy.special.gammaln(-2.0)=-3.0=-10.0 all == inf (live oracle).
    for x in [-2.0_f64, -3.0, -10.0] {
        let lg = lgamma(x);
        assert!(
            lg.is_infinite() && lg > 0.0,
            "lgamma({x})={lg} expected scipy +inf (non-positive-integer pole)"
        );
    }
    // Negative non-integer reflection + large x: live scipy oracle.
    let cases = [
        (-2.5_f64, -0.056_243_716_497_674_01_f64),
        (100.0, 359.134_205_369_575_4),
    ];
    for (x, expected) in cases {
        assert!(
            rel_close(lgamma(x), expected, 1e-12),
            "lgamma({x})={} expected scipy {expected} (rel {:e})",
            lgamma(x),
            ((lgamma(x) - expected) / expected).abs()
        );
    }
}

// ===========================================================================
// (5) NEW RED PIN — `special::gamma` overflows to +inf for large finite x where
//     scipy stays FINITE. The Lanczos branch computes `t.powf(z+0.5)` (with
//     t ~= z+7.5 ~= 178, exponent ~= 170.5) which overflows f64 to +inf before
//     the multiplication by the tiny `(-t).exp()` factor; scipy evaluates the
//     same expression in log-space so it never overflows until the true
//     gamma overflow point (gamma(172) = inf).
//
//     Single-file-fixable in special.rs (log-space evaluation of the Lanczos
//     tail). Until fixed this is RED-ignored so default `cargo test` stays
//     green (no stranded reds).
//
//     Live scipy oracle (from /tmp):
//       sp.gamma(171.0) = 7.257415615308e+306   (finite)
//     ferrolearn gamma(171.0) = inf  (overflow ~x>=168).
// ===========================================================================

/// Divergence: `special::gamma` overflows to `+inf` for large finite x
/// (`special.rs:61`, `t.powf(z+0.5)` intermediate overflow) where
/// `scipy.special.gamma(171.0)` is the finite `7.257415615308e+306`.
/// Tracking: #1946
#[test]
#[ignore = "divergence: gamma(171) overflows to inf vs scipy finite 7.26e306; tracking #1946"]
fn divergence_gamma_large_x_overflows() {
    // Live scipy oracle: sp.gamma(171.0) == 7.257415615308e+306 (finite).
    const SP_GAMMA_171: f64 = 7.257_415_615_308e306;
    let g = gamma(171.0);
    assert!(
        g.is_finite() && rel_close(g, SP_GAMMA_171, 1e-12),
        "gamma(171.0)={g:e} expected scipy finite {SP_GAMMA_171:e}"
    );
}
