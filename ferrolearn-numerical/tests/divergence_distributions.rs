//! Divergence audit for `ferrolearn-numerical/src/distributions.rs` against the
//! LIVE scipy.stats oracle (scipy 1.17.1). See `.design/numerical/distributions.md`.
//!
//! All expected values are computed by a live scipy call run from `/tmp`
//! (R-CHAR-3) and are NEVER copied from the ferrolearn side:
//!
//! ```text
//! cd /tmp && python3 -c "
//! import scipy.stats as st
//! print(repr(st.norm.cdf(1.5)), repr(st.norm.sf(2.0)), repr(st.norm.sf(4.0)),
//!       repr(st.norm.cdf(-0.5)), repr(st.norm.pdf(0.0)), repr(st.norm.ppf(0.975)))
//! print(repr(st.chi2.cdf(5.991,2)), repr(st.chi2.sf(5.991,2)),
//!       repr(st.chi2.ppf(0.95,2)), repr(st.chi2.pdf(3.0,2)))
//! print(repr(st.f.sf(3.0,2,10)), repr(st.f.ppf(0.95,2,10)),
//!       repr(st.f.pdf(2.0,2,10)), repr(st.f.cdf(2.0,2,10)))
//! print(repr(st.t.sf(2.0,5)), repr(st.t.ppf(0.975,5)), repr(st.t.cdf(1.0,5)))
//! print(repr(st.beta.cdf(0.5,2,3)), repr(st.beta.pdf(0.5,2,3)))
//! print(repr(st.gamma.cdf(2.0,a=2,scale=1.0)), repr(st.gamma.cdf(4.0,a=3,scale=2.0)))
//! print(repr(2*st.t.sf(2.0,5)))
//! print(repr(st.dirichlet.logpdf([0.2,0.3,0.5],[2.0,3.0,4.0])))
//! print(repr(st.chi2.sf(5.0,-1.0)))
//! "
//! ```
//!
//! scipy 1.17.1 output (the source of truth for every expected value below):
//!
//! ```text
//! norm: cdf(1.5)=0.9331927987311419 sf(2)=0.022750131948179195
//!       sf(4)=3.167124183311986e-05 cdf(-0.5)=0.3085375387259869
//!       pdf(0)=0.3989422804014327 ppf(0.975)=1.959963984540054
//! chi2: cdf(5.991,2)=0.9499883849734209 sf(5.991,2)=0.05001161502657909
//!       ppf(0.95,2)=5.991464547107979 pdf(3,2)=0.11156508007421491
//! f:    sf(3,2,10)=0.09536743164062497 ppf(0.95,2,10)=4.1028210151304
//!       pdf(2,2,10)=0.13281030862990806 cdf(2,2,10)=0.8140655679181293
//! t:    sf(2,5)=0.050969739414929174 ppf(0.975,5)=2.5705818356363146
//!       cdf(1,5)=0.8183912661754386
//! beta: cdf(0.5,2,3)=0.6875 pdf(0.5,2,3)=1.5000000000000004
//! gamma:cdf(2,a=2,scale=1)=0.5939941502901616 cdf(4,a=3,scale=2)=0.32332358381693654
//! 2*t.sf(2,5)=0.10193947882985835
//! dirichlet.logpdf([0.2,0.3,0.5],[2,3,4])=2.0228711901914433
//! chi2.sf(5,-1)=nan
//! ```

use ferrolearn_numerical::distributions::{
    Beta, ChiSquared, ContinuousDistribution, Dirichlet, FDist, Gamma, Normal, StudentsT, chi2_sf,
    f_sf, norm_sf, t_test_two_tailed,
};

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
// (1) RED PIN — REQ-1: Normal cdf/sf accuracy divergence (single-file-fixable).
//
// statrs's `Normal::cdf`/`sf` use an erf approximation accurate to only ~1e-11.
// scipy.stats.norm is machine-precision (Cephes ndtr). Pinned at a TIGHT 1e-13
// tolerance — the test FAILS by ~1e-11 until the Gaussian cdf/sf is routed
// through `libm::erf`/`erfc` (cdf(x)=0.5*(1+erf(x/sqrt2)), sf(x)=0.5*erfc(x/sqrt2)).
// Normal pdf/ppf are NOT pinned: they are already machine-precision (see green
// guard `green_normal_pdf_ppf`).
//
// Tracking: #1965.
// ===========================================================================

/// Divergence: `Normal::cdf`/`sf` in `distributions.rs` (statrs erf approximation)
/// diverges from `scipy.stats.norm` (Cephes ndtr, machine precision).
/// scipy: cdf(1.5)=0.9331927987311419, sf(2)=0.022750131948179195,
/// sf(4)=3.167124183311986e-05, cdf(-0.5)=0.3085375387259869.
/// ferrolearn returns these to only ~1e-11 relative — FAILS the 1e-13 contract.
#[test]
fn red_normal_cdf_sf_machine_precision() {
    let n = Normal::new(0.0, 1.0).unwrap();
    const TOL: f64 = 1e-13;

    // scipy.stats.norm.cdf(1.5)
    assert!(
        rel_close(n.cdf(1.5), 0.9331927987311419, TOL),
        "Normal::cdf(1.5) = {:.17e}, scipy = 0.9331927987311419 (rel > {TOL})",
        n.cdf(1.5)
    );
    // scipy.stats.norm.sf(2.0)
    assert!(
        rel_close(n.sf(2.0), 0.022750131948179195, TOL),
        "Normal::sf(2.0) = {:.17e}, scipy = 0.022750131948179195 (rel > {TOL})",
        n.sf(2.0)
    );
    // scipy.stats.norm.sf(4.0) — tail
    assert!(
        rel_close(n.sf(4.0), 3.167124183311986e-05, TOL),
        "Normal::sf(4.0) = {:.17e}, scipy = 3.167124183311986e-05 (rel > {TOL})",
        n.sf(4.0)
    );
    // scipy.stats.norm.cdf(-0.5)
    assert!(
        rel_close(n.cdf(-0.5), 0.3085375387259869, TOL),
        "Normal::cdf(-0.5) = {:.17e}, scipy = 0.3085375387259869 (rel > {TOL})",
        n.cdf(-0.5)
    );
}

/// Divergence: `norm_sf` inherits the `Normal::sf` erf-approximation gap.
/// scipy: norm.sf(2.0)=0.022750131948179195. FAILS the 1e-13 contract by ~1e-11.
/// Tracking: #1965.
#[test]
fn red_norm_sf_machine_precision() {
    const TOL: f64 = 1e-13;
    assert!(
        rel_close(norm_sf(2.0), 0.022750131948179195, TOL),
        "norm_sf(2.0) = {:.17e}, scipy norm.sf(2.0) = 0.022750131948179195 (rel > {TOL})",
        norm_sf(2.0)
    );
}

// ===========================================================================
// (2) RED PIN — REQ-10: panic-in-production on invalid params (single-file-fixable
// for the convenience fns). scipy.stats returns `nan` for invalid params
// (chi2.sf(5,-1)=nan); ferrolearn's `chi2_sf` calls `.expect(...)` and PANICS.
//
// This is a DIRECT assertion `assert!(chi2_sf(5.0,-1.0).is_nan())`: the test
// FAILS by PANICKING inside `chi2_sf` (the `.expect("chi2_sf: invalid df")`)
// before the assertion is even reached. The fix makes the convenience fns
// return `nan` for invalid params instead of `.expect`.
//
// Scope note: this pin covers ONLY the convenience-fn nan behavior
// (chi2_sf/f_sf/t_test_two_tailed). The `unwrap_stat` panic in mean/variance
// and the `Dirichlet::ln_pdf`/`sample` asserts/expects are a SEPARATE structural
// concern filed as blocker #1970 (not single-convenience-fn-fixable).
//
// Tracking: #1966.
// ===========================================================================

/// Divergence: `chi2_sf(5.0, -1.0)` in `distributions.rs` PANICS via
/// `.expect("chi2_sf: invalid df")`, whereas `scipy.stats.chi2.sf(5.0,-1.0)=nan`.
/// FAILS by panicking until the convenience fn returns `nan` for invalid `df`.
/// Tracking: #1966.
#[test]
fn red_chi2_sf_invalid_df_returns_nan() {
    // scipy.stats.chi2.sf(5.0, -1.0) == nan (does NOT raise from the sf call).
    assert!(
        chi2_sf(5.0, -1.0).is_nan(),
        "chi2_sf(5.0,-1.0) should be nan (scipy), got a value or panicked"
    );
}

/// Divergence: `f_sf(3.0, -1.0, 10.0)` PANICS via `.expect`, whereas
/// `scipy.stats.f.sf(3.0,-1.0,10.0)=nan`. Tracking: #1966.
#[test]
fn red_f_sf_invalid_df_returns_nan() {
    assert!(
        f_sf(3.0, -1.0, 10.0).is_nan(),
        "f_sf(3.0,-1.0,10.0) should be nan (scipy), got a value or panicked"
    );
}

/// Divergence: `t_test_two_tailed(2.0, -1.0)` PANICS via `.expect`, whereas
/// `2*scipy.stats.t.sf(2.0,-1.0)=nan`. Tracking: #1966.
#[test]
fn red_t_test_invalid_df_returns_nan() {
    assert!(
        t_test_two_tailed(2.0, -1.0).is_nan(),
        "t_test_two_tailed(2.0,-1.0) should be nan (scipy), got a value or panicked"
    );
}

// ===========================================================================
// (3) GREEN GUARDS — the SHIPPED distributions match the live scipy oracle at a
// TIGHT ~1e-12 tolerance. These MUST PASS now (they are not #[ignore]d).
// ===========================================================================

const GREEN_TOL: f64 = 1e-12;

/// Green: Normal pdf/ppf are machine-precision (only cdf/sf diverge — see RED pin).
#[test]
fn green_normal_pdf_ppf() {
    let n = Normal::new(0.0, 1.0).unwrap();
    // scipy norm.pdf(0.0)=0.3989422804014327, norm.ppf(0.975)=1.959963984540054
    assert!(rel_close(n.pdf(0.0), 0.3989422804014327, GREEN_TOL));
    assert!(rel_close(n.ppf(0.975), 1.959963984540054, GREEN_TOL));
}

/// Green: ChiSquared pdf/cdf/sf/ppf vs scipy.stats.chi2 (df=2).
#[test]
fn green_chi2() {
    let c = ChiSquared::new(2.0).unwrap();
    // scipy chi2.cdf(5.991,2)=0.9499883849734209, sf=0.05001161502657909
    assert!(rel_close(c.cdf(5.991), 0.9499883849734209, GREEN_TOL));
    assert!(rel_close(c.sf(5.991), 0.05001161502657909, GREEN_TOL));
    // scipy chi2.pdf(3,2)=0.11156508007421491
    assert!(rel_close(c.pdf(3.0), 0.11156508007421491, GREEN_TOL));
    // scipy chi2.ppf(0.95,2)=5.991464547107979
    assert!(rel_close(c.ppf(0.95), 5.991464547107979, GREEN_TOL));
}

/// Green: FDist pdf/cdf/sf/ppf vs scipy.stats.f (df1=2, df2=10), incl. a tail.
#[test]
fn green_f() {
    let f = FDist::new(2.0, 10.0).unwrap();
    // scipy f.sf(3,2,10)=0.09536743164062497 (a tail-ish value)
    assert!(rel_close(f.sf(3.0), 0.09536743164062497, GREEN_TOL));
    // scipy f.cdf(2,2,10)=0.8140655679181293, pdf(2,2,10)=0.13281030862990806
    assert!(rel_close(f.cdf(2.0), 0.8140655679181293, GREEN_TOL));
    assert!(rel_close(f.pdf(2.0), 0.13281030862990806, GREEN_TOL));
    // scipy f.ppf(0.95,2,10)=4.1028210151304
    assert!(rel_close(f.ppf(0.95), 4.1028210151304, GREEN_TOL));
}

/// Green: StudentsT cdf/sf/ppf vs scipy.stats.t (df=5).
#[test]
fn green_t() {
    let t = StudentsT::new(5.0).unwrap();
    // scipy t.sf(2,5)=0.050969739414929174, cdf(1,5)=0.8183912661754386
    assert!(rel_close(t.sf(2.0), 0.050969739414929174, GREEN_TOL));
    assert!(rel_close(t.cdf(1.0), 0.8183912661754386, GREEN_TOL));
    // scipy t.ppf(0.975,5)=2.5705818356363146
    assert!(rel_close(t.ppf(0.975), 2.5705818356363146, GREEN_TOL));
}

/// Green: Beta cdf/pdf vs scipy.stats.beta (a=2, b=3).
#[test]
fn green_beta() {
    let b = Beta::new(2.0, 3.0).unwrap();
    // scipy beta.cdf(0.5,2,3)=0.6875, pdf(0.5,2,3)=1.5000000000000004
    assert!(rel_close(b.cdf(0.5), 0.6875, GREEN_TOL));
    assert!(rel_close(b.pdf(0.5), 1.5000000000000004, GREEN_TOL));
}

/// Green: Gamma cdf vs scipy.stats.gamma — CONFIRMS the rate=1/scale convention.
/// ferrolearn `Gamma::new(shape, rate)`; scipy `gamma(a=shape, scale=1/rate)`.
#[test]
fn green_gamma_rate_convention() {
    // scipy gamma.cdf(2.0,a=2,scale=1.0)=0.5939941502901616; rate=1=1/scale.
    let g1 = Gamma::new(2.0, 1.0).unwrap();
    assert!(rel_close(g1.cdf(2.0), 0.5939941502901616, GREEN_TOL));

    // scipy gamma.cdf(4.0,a=3,scale=2.0)=0.32332358381693654; rate=0.5=1/2.
    let g2 = Gamma::new(3.0, 0.5).unwrap();
    assert!(rel_close(g2.cdf(4.0), 0.32332358381693654, GREEN_TOL));
}

/// Green: convenience p-value fns at VALID params vs scipy.
#[test]
fn green_convenience_valid_params() {
    // scipy chi2.sf(5.991,2)=0.05001161502657909
    assert!(rel_close(
        chi2_sf(5.991, 2.0),
        0.05001161502657909,
        GREEN_TOL
    ));
    // scipy f.sf(3,2,10)=0.09536743164062497
    assert!(rel_close(
        f_sf(3.0, 2.0, 10.0),
        0.09536743164062497,
        GREEN_TOL
    ));
    // scipy 2*t.sf(2,5)=0.10193947882985835
    assert!(rel_close(
        t_test_two_tailed(2.0, 5.0),
        0.10193947882985835,
        GREEN_TOL
    ));
}

/// Green: Dirichlet ln_pdf vs scipy.stats.dirichlet.logpdf.
#[test]
fn green_dirichlet_logpdf() {
    let d = Dirichlet::new(&[2.0, 3.0, 4.0]).unwrap();
    // scipy dirichlet.logpdf([0.2,0.3,0.5],[2,3,4])=2.0228711901914433
    assert!(rel_close(
        d.ln_pdf(&[0.2, 0.3, 0.5]),
        2.0228711901914433,
        GREEN_TOL
    ));
}
