//! scipy.special parity: high-value scalar special functions.
//!
//! All functions take and return `f64`. Implementations are concise series /
//! Lanczos-style approximations chosen for adequate accuracy across the
//! domain typically used by ML estimators (e.g. priors, log-likelihoods,
//! Gaussian CDFs).
//!
//! Provided:
//!
//! - [`gamma`] — Γ(x) via Lanczos.
//! - [`lgamma`] — log|Γ(x)| (signed for negative non-integer x).
//! - [`digamma`] — ψ(x) = d/dx ln Γ(x).
//! - [`beta`] — B(a, b).
//! - [`lbeta`] — log B(a, b).
//! - [`erf`] — error function, delegating to `libm::erf` (machine precision).
//! - [`erfc`] — complementary error function, delegating to `libm::erfc`.
//!
//! ## REQ status
//!
//! Mirrors `scipy.special` scalar functions (the scipy substrate sklearn is
//! built on; live oracle: installed scipy 1.17, version-stable math functions).
//! Design doc: `.design/numerical/special.md` (8 REQs). Every REQ is BINARY
//! (R-DEFER-2): SHIPPED or NOT-STARTED (with a concrete blocker). All values are
//! oracle-verified element-wise against the live scipy (R-CHAR-3) — see
//! `tests/divergence_special.rs`.
//!
//! **6 SHIPPED / 2 NOT-STARTED.**
//!
//! | REQ | Status | Notes |
//! |---|---|---|
//! | REQ-1 (`gamma`) | SHIPPED | Lanczos g=7 n=9 (scipy's coeffs) + reflection; evaluated in LOG-space for x≥0.5 to avoid premature overflow (FIXED #1946: `gamma(171)=7.257e306` finite, `+inf` at ≥172, matching scipy); negative-integer poles → `NaN`, `gamma(0)=+inf` (FIXED #1943). Oracle-verified across integers/half-integers/large(20,171)/negative-non-integer within 1e-12. |
//! | REQ-2 (`lgamma`) | SHIPPED | log\|Γ\| via Lanczos + reflection; non-positive-integer poles → `+inf` matching `scipy.special.gammaln` (FIXED #1943). Oracle-verified ≤1e-12. |
//! | REQ-3 (`digamma`) | SHIPPED | recurrence to z≥6 + 5-term Bernoulli asymptotic; matches `scipy.special.digamma` ≤~2e-11 (the truncation floor). |
//! | REQ-4 (`beta` / `lbeta`) | SHIPPED | via `lgamma`; matches `scipy.special.beta`/`betaln` ≤1e-12. |
//! | REQ-5 (`erf` / `erfc`) | SHIPPED | FIXED #1942: delegate to `libm::erf`/`libm::erfc` (Cephes/musl, machine precision) — was Abramowitz & Stegun 7.1.26 (~1.5e-7). Matches `scipy.special.erf`/`erfc` ≤1e-12 across the range incl. negatives and the tail (erfc(10)≈2.09e-45). |
//! | REQ-6 (pole / inf / nan handling) | SHIPPED | gamma negative-integer `NaN`, gamma(0)/erf(∞)/erfc(∞) edge values, lgamma non-positive-integer `+inf` — all match scipy (FIXED #1943; guards `green_erf_inf_and_gamma_zero_pole`, `green_gamma_poles_*`). |
//! | REQ-7 (production consumer) | NOT-STARTED | no non-test workspace caller — crates needing erf use `statrs`/`libm` directly; special.rs is currently a standalone public-API module. Future consumer: gp_kernels Matern-general-nu (#1914) needs `gamma`. Blocker #1944. |
//! | REQ-8 (ferray substrate) | NOT-STARTED | hand-rolled scipy.special reimplementation (+ `libm` transitional) vs the `ferray::stats`/ferray special-functions destination (R-SUBSTRATE-1/5). Blocker #1945. |

use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Lanczos coefficients (g = 7, n = 9 — same set scipy uses)
// ---------------------------------------------------------------------------
const LANCZOS_G: f64 = 7.0;
const LANCZOS_COEFFS: [f64; 9] = [
    0.999_999_999_999_809_9,
    676.520_368_121_885_1,
    -1_259.139_216_722_402_8,
    771.323_428_777_653_2,
    -176.615_029_162_140_6,
    12.507_343_278_686_905,
    -0.138_571_095_265_720_12,
    9.984_369_578_019_572e-6,
    1.505_632_735_149_311_6e-7,
];

/// Compute Γ(x) for any real `x`.
///
/// Uses Lanczos's approximation; reflection formula handles negative `x`.
///
/// For `x >= 0.5` the Lanczos product is evaluated in **log-space** and
/// exponentiated at the end (`exp(0.5·ln(2π) + (z+0.5)·ln t − t + ln a)`) to
/// avoid the premature overflow of the `t^(z+0.5)` intermediate; this keeps the
/// result finite up to `x ≈ 171` and overflows to `+inf` only at `x >= 172`,
/// matching `scipy.special.gamma` (`gamma(171.0) ≈ 7.257e306`, `gamma(172.0) = inf`).
///
/// At the poles of Γ this matches `scipy.special.gamma`: `gamma(0.0)` is `+inf`
/// (via the reflection `π / (sin(0) · …)` where `sin(0) == 0` exactly), and
/// every negative integer is a pole returning `NaN`.
#[must_use]
pub fn gamma(x: f64) -> f64 {
    if x < 0.0 && x == x.floor() {
        // Negative-integer poles: scipy.special.gamma returns nan. The
        // reflection formula would otherwise yield huge finite garbage because
        // sin(π·n) is a tiny nonzero float rather than exactly 0.
        return f64::NAN;
    }
    if x < 0.5 {
        // reflection: Γ(x) Γ(1-x) = π / sin(π x)
        PI / ((PI * x).sin() * gamma(1.0 - x))
    } else {
        let z = x - 1.0;
        let mut a = LANCZOS_COEFFS[0];
        for (i, &c) in LANCZOS_COEFFS.iter().enumerate().skip(1) {
            a += c / (z + i as f64);
        }
        let t = z + LANCZOS_G + 0.5;
        // Evaluate in log-space and exponentiate at the end. The direct product
        // `√(2π)·t^(z+0.5)·e^(−t)·a` forms the intermediate `t^(z+0.5)` which
        // overflows f64 to `+inf` for large x (e.g. x≈170 ⇒ t^(z+0.5)≈exp(864))
        // BEFORE the tiny `e^(−t)` factor brings it back down — even though the
        // true product is finite. `exp(0.5·ln(2π) + (z+0.5)·ln t − t + ln a)` is
        // algebraically identical (it is exactly what `lgamma` computes for this
        // branch) but never forms the overflowing intermediate, so it stays
        // finite up to the TRUE overflow point (≈171), matching scipy. For
        // x ≥ 0.5 the Lanczos sum `a` is positive, so `a.ln()` is finite.
        (0.5 * (2.0 * PI).ln() + (z + 0.5) * t.ln() - t + a.ln()).exp()
    }
}

/// Compute ln|Γ(x)|.
///
/// At the poles of Γ (non-positive integers) this matches
/// `scipy.special.gammaln`, returning `+inf`. A `NaN` input is passed through.
#[must_use]
pub fn lgamma(x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    if x <= 0.0 && x == x.floor() {
        return f64::INFINITY;
    }
    if x < 0.5 {
        let s = (PI * x).sin().abs();
        if s == 0.0 {
            return f64::NAN;
        }
        // log Γ(x) = log π − log|sin πx| − log Γ(1 − x)
        return PI.ln() - s.ln() - lgamma(1.0 - x);
    }
    let z = x - 1.0;
    let mut a = LANCZOS_COEFFS[0];
    for (i, &c) in LANCZOS_COEFFS.iter().enumerate().skip(1) {
        a += c / (z + i as f64);
    }
    let t = z + LANCZOS_G + 0.5;
    0.5 * (2.0 * PI).ln() + (z + 0.5) * t.ln() - t + a.ln()
}

/// Compute the digamma function ψ(x) = d/dx ln Γ(x).
///
/// Uses the recurrence ψ(x) = ψ(x + 1) − 1/x to push x ≥ 6, then an
/// asymptotic expansion.
#[must_use]
pub fn digamma(x: f64) -> f64 {
    let mut result = 0.0_f64;
    let mut z = x;
    if z < 0.5 {
        // reflection
        return digamma(1.0 - z) - PI / (PI * z).tan();
    }
    while z < 6.0 {
        result -= 1.0 / z;
        z += 1.0;
    }
    let inv = 1.0 / z;
    let inv2 = inv * inv;
    // Asymptotic series: ψ(z) ≈ ln z − 1/(2z) − Σ B_{2n}/(2n z^{2n})
    result += z.ln() - 0.5 * inv;
    let mut acc = inv2;
    let coeffs = [
        1.0 / 12.0,
        -1.0 / 120.0,
        1.0 / 252.0,
        -1.0 / 240.0,
        5.0 / 660.0,
    ];
    for &c in &coeffs {
        result -= c * acc;
        acc *= inv2;
    }
    result
}

/// Compute the Beta function B(a, b) = Γ(a) Γ(b) / Γ(a + b).
#[must_use]
pub fn beta(a: f64, b: f64) -> f64 {
    (lbeta(a, b)).exp()
}

/// Compute log B(a, b).
#[must_use]
pub fn lbeta(a: f64, b: f64) -> f64 {
    lgamma(a) + lgamma(b) - lgamma(a + b)
}

/// Compute the error function erf(x).
///
/// Delegates to [`libm::erf`] (the FreeBSD/Sun msun Cephes-family
/// implementation, machine precision ~1e-16) — the `scipy.special.erf`
/// analog. Matches `scipy.special.erf` to ~machine precision.
#[must_use]
pub fn erf(x: f64) -> f64 {
    libm::erf(x)
}

/// Compute the complementary error function erfc(x) = 1 - erf(x).
///
/// Delegates to [`libm::erfc`] (the same Cephes-family implementation as
/// [`erf`], machine precision) — the `scipy.special.erfc` analog, with good
/// relative accuracy in the tails.
#[must_use]
pub fn erfc(x: f64) -> f64 {
    libm::erfc(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        if a.is_nan() || b.is_nan() {
            return false;
        }
        (a - b).abs() < tol || (b != 0.0 && ((a - b) / b).abs() < tol)
    }

    #[test]
    fn gamma_integers() {
        assert!(approx(gamma(1.0), 1.0, 1e-10));
        assert!(approx(gamma(2.0), 1.0, 1e-10));
        assert!(approx(gamma(3.0), 2.0, 1e-10));
        assert!(approx(gamma(4.0), 6.0, 1e-10));
        assert!(approx(gamma(5.0), 24.0, 1e-10));
        assert!(approx(gamma(10.0), 362_880.0, 1e-7));
    }

    #[test]
    fn gamma_half() {
        // Γ(1/2) = sqrt(π)
        assert!(approx(gamma(0.5), PI.sqrt(), 1e-10));
        // Γ(3/2) = sqrt(π) / 2
        assert!(approx(gamma(1.5), PI.sqrt() / 2.0, 1e-10));
    }

    #[test]
    fn gamma_reflection_negative() {
        // Γ(-0.5) = -2 sqrt(π)
        let expected = -2.0 * PI.sqrt();
        assert!(approx(gamma(-0.5), expected, 1e-10));
    }

    #[test]
    fn lgamma_matches_gamma() {
        for x in [0.5, 1.5, 2.5, 7.5, 12.0] {
            let lg = lgamma(x);
            let g = gamma(x);
            assert!(
                approx(lg, g.ln(), 1e-9),
                "lgamma({x})={lg} vs ln gamma={}",
                g.ln()
            );
        }
    }

    #[test]
    fn lgamma_pole_at_nonpositive_int() {
        // scipy.special.gammaln returns +inf at non-positive integer poles.
        for x in [0.0_f64, -1.0, -5.0] {
            let lg = lgamma(x);
            assert!(
                lg.is_infinite() && lg > 0.0,
                "lgamma({x})={lg} expected +inf"
            );
        }
    }

    #[test]
    fn digamma_known_values() {
        // ψ(1) = -γ ≈ -0.577_215_664_901_532_86
        assert!(approx(digamma(1.0), -0.577_215_664_901_532_9, 1e-9));
        // ψ(2) = 1 - γ
        assert!(approx(digamma(2.0), 1.0 - 0.577_215_664_901_532_9, 1e-9));
        // ψ(0.5) = -γ - 2 ln 2
        let expected = -0.577_215_664_901_532_9 - 2.0 * 2.0_f64.ln();
        assert!(approx(digamma(0.5), expected, 1e-9));
    }

    #[test]
    fn beta_symmetry() {
        // B(a, b) = B(b, a)
        assert!(approx(beta(2.0, 5.0), beta(5.0, 2.0), 1e-12));
    }

    #[test]
    fn beta_known_value() {
        // B(2, 3) = 1/12
        assert!(approx(beta(2.0, 3.0), 1.0 / 12.0, 1e-9));
        // B(0.5, 0.5) = π
        assert!(approx(beta(0.5, 0.5), PI, 1e-9));
    }

    #[test]
    fn erf_known_values() {
        assert!(approx(erf(0.0), 0.0, 1e-9));
        assert!(approx(erf(1.0), 0.842_700_792_949_715, 2e-7));
        assert!(approx(erf(-1.0), -0.842_700_792_949_715, 2e-7));
        // erf(infinity) -> 1
        assert!(approx(erf(10.0), 1.0, 1e-9));
        assert!(approx(erf(-10.0), -1.0, 1e-9));
    }

    #[test]
    fn erfc_consistent_with_erf() {
        for &x in &[-2.0_f64, -1.0, -0.1, 0.0, 0.1, 1.0, 2.0, 5.0] {
            let lhs = erf(x) + erfc(x);
            assert!(
                (lhs - 1.0).abs() < 5e-7,
                "erf+erfc != 1 at x={x} (got {lhs})"
            );
        }
    }
}
