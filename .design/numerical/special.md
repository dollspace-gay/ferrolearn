# scipy.special scalar functions

<!--
tier: 3-component
status: draft
baseline-commit: 46c522e6e
upstream-paths:
  - scipy/special/__init__.py        # documented scalar-function surface
  - scipy/special/_basic.py          # Python-level wrappers (impls are Cephes/boost C)
-->

## Summary

`ferrolearn-numerical/src/special.rs` is the scipy-analog substrate that mirrors
a slice of **scipy.special** scalar functions: `gamma` (`scipy.special.gamma`),
`lgamma` (`scipy.special.gammaln`), `digamma` (`scipy.special.digamma` = `psi`),
`beta`/`lbeta` (`scipy.special.beta`/`betaln`), and `erf`/`erfc`
(`scipy.special.erf`/`erfc`). All functions are `f64 -> f64`, hand-rolled from
classical series: a Lanczos (`g=7`, `n=9`) gamma core with reflection, a 5-term
Bernoulli asymptotic digamma, and an **Abramowitz & Stegun 7.1.26** polynomial
`erf`. scipy.special's implementations are Cephes/boost (C) accurate to ~machine
precision; these mathematical functions are version-stable, so the installed
**scipy 1.17.1** is a valid live oracle for the true mathematical values (the
sklearn-1.5.2 / scipy-1.17.1 version split does not matter — `gamma(5.5)` is
`gamma(5.5)` in every scipy release).

Divergence classes: (1) **erf/erfc accuracy** — A&S 7.1.26 caps the error at
~1.4e-7, where scipy is ~1e-15 (the headline, deterministic, FIXABLE via
`statrs`); (2) **gamma/lgamma/digamma/beta accuracy** — all match scipy to a
tight tolerance (gamma/lgamma/beta ~1e-15, digamma ~2e-11) and are numerically
SHIPPED; (3) **pole/edge handling** — `gamma`/`lgamma` return NaN or garbage at
negative-integer poles where scipy returns signed `inf`/`nan`; (4) **no
consumer** — nothing in the workspace calls `special::`; (5) **ferray
substrate** — the destination is a `ferray::stats` / ferray special-functions
module (the scipy.special analog), not a hand-rolled module in `ferrolearn-numerical`.

## Upstream reference (scipy.special, live oracle scipy 1.17.1)

The Python wrappers live in `scipy/special/__init__.py` (the documented function
list — `gamma`, `gammaln`, `psi`/`digamma`, `beta`, `betaln`, `erf`, `erfc`) and
`scipy/special/_basic.py`; the numerical kernels are Cephes/boost C, so cite the
scipy.special **function names** and the **live-oracle values**, never C line
numbers. Live oracle (`cd /tmp && python3 -c "import scipy.special as sp; ..."`,
scipy 1.17.1):

- `sp.gamma`: `gamma(5.5)=52.34277778455352`, `gamma(20.0)=1.21645100408832e+17`,
  `gamma(0.5)=1.7724538509055159`, `gamma(-0.5)=-3.5449077018110318`,
  `gamma(0.1)=9.513507698668732`. Poles: `gamma(0.0)=inf`, `gamma(-1.0)=nan`.
- `sp.gammaln`: `gammaln(0.5)=0.5723649429247`, `gammaln(12.0)=17.502307845873887`,
  `gammaln(20.0)=39.339884187199495`, `gammaln(-0.5)=1.2655121234846454`.
  Poles: `gammaln(0.0)=inf`, `gammaln(-1.0)=inf`.
- `sp.digamma` (= `sp.psi`): `digamma(0.5)=-1.9635100260214235`,
  `digamma(1.0)=-0.5772156649015329`, `digamma(2.0)=0.42278433509846713`,
  `digamma(10.0)=2.251752589066721`, `digamma(-1.5)=0.7031566406452434`.
- `sp.beta`: `beta(2,3)=0.08333333333333333` (= 1/12),
  `beta(0.5,0.5)=3.1415926535897927` (= π), `beta(2,5)=0.03333333333333333`.
  `sp.betaln`: `betaln(2,3)=-2.4849066497880004`.
- `sp.erf`: `erf(0.5)=0.5204998778130465`, `erf(1.0)=0.8427007929497148`,
  `erf(2.0)=0.9953222650189527`, `erf(0.1)=0.1124629160182849`, `erf(inf)=1.0`.
  `sp.erfc`: `erfc(1.0)=0.15729920705028516`, `erfc(2.0)=0.004677734981047266`,
  `erfc(inf)=0.0`.

## Requirements

- REQ-1: `gamma` numerical parity. `gamma(x)` matches `scipy.special.gamma(x)` to
  a tight relative tolerance over integers, half-integers, large `x`, and
  negative non-integers (the Lanczos `g=7,n=9` core + reflection).
- REQ-2: `lgamma` numerical parity. `lgamma(x)` matches `scipy.special.gammaln(x)`
  to a tight relative tolerance for positive and negative non-integer `x`.
- REQ-3: `digamma` numerical parity. `digamma(x)` matches `scipy.special.digamma`
  (= `psi`) to a tight tolerance via recurrence-to-`z>=6` + the 5-term Bernoulli
  asymptotic series, including the negative-`x` reflection.
- REQ-4: `beta`/`lbeta` numerical parity. `beta(a,b)`/`lbeta(a,b)` match
  `scipy.special.beta`/`betaln` (computed through `lgamma`).
- REQ-5: `erf`/`erfc` numerical parity. `erf(x)`/`erfc(x)` match
  `scipy.special.erf`/`erfc` to ~machine precision. They do NOT: A&S 7.1.26 caps
  the error at ~1.4e-7. The deterministic fix is `statrs::function::erf::{erf,erfc}`
  (already a `ferrolearn-numerical` dependency), which is machine-precision.
- REQ-6: Pole / `inf` / `nan` edge parity. At non-positive-integer poles
  `scipy.special.gamma` returns signed `inf`/`nan` and `gammaln` returns `inf`;
  `erf(inf)=1`, `erfc(inf)=0`. ferrolearn's `gamma`/`lgamma` diverge at
  negative-integer poles.
- REQ-7: Non-test production consumer. A non-test caller in the workspace (an
  estimator, or the `ferrolearn-python` binding) consumes `special::*` so it is
  part of the live translation surface.
- REQ-8: ferray substrate (R-SUBSTRATE-1). The scipy.special analog lives in
  `ferray::stats` / a ferray special-functions module — `ferrolearn-numerical`
  consumes ferray's special functions rather than hand-rolling them.

## Acceptance criteria

All expected values come from the live scipy oracle (R-CHAR-3), never from
ferrolearn. Run from `/tmp`.

- AC-1 (REQ-1): `python3 -c "import scipy.special as sp; print(sp.gamma(5.5))"` →
  `52.34277778455352`; `gamma(5.5)` agrees to rel ≤ 1e-12 (observed 1.2e-15).
  Spot grid `{1,2,3,4,5,5.5,10,20,0.5,1.5,-0.5,0.1}` agrees to rel ≤ 1e-12
  (observed max ~4.2e-15 at `gamma(10)`).
- AC-2 (REQ-2): `python3 -c "import scipy.special as sp; print(sp.gammaln(0.5))"`
  → `0.5723649429247`; `lgamma(0.5)` agrees to rel ≤ 1e-12 (observed 7.8e-16).
  Grid `{0.5,1.5,2.5,7.5,12,20,-0.5}` agrees to rel ≤ 1e-12.
- AC-3 (REQ-3): `python3 -c "import scipy.special as sp; print(sp.digamma(0.5))"`
  → `-1.9635100260214235`; `digamma(0.5)` agrees to rel ≤ 1e-10 (observed
  1.7e-12). Grid `{0.5,1,2,10,-1.5}` agrees to rel ≤ 1e-10 (observed worst-case
  ~2.1e-11 at `digamma(2.0)` — the 5-term Bernoulli truncation floor).
- AC-4 (REQ-4): `python3 -c "import scipy.special as sp; print(sp.beta(2,3), sp.betaln(2,3))"`
  → `0.08333333333333333 -2.4849066497880004`; `beta(2,3)`/`lbeta(2,3)` agree to
  rel ≤ 1e-12 (observed 2.7e-15 / 1.1e-15). `beta(0.5,0.5)` agrees with `π`.
- AC-5 (REQ-5): `python3 -c "import scipy.special as sp; print(sp.erf(1.0))"` →
  `0.8427007929497148`; ferrolearn `erf(1.0) ≈ 0.8427006897475899`, abs diff
  ~1.03e-7 — FAILS a `1e-12` tolerance. Max abs error over `x∈[0,4]` is ~1.4e-7
  (observed `1.385e-7` near `x=0.5`). The fix target `statrs::function::erf::erf(1.0)`
  equals the scipy value to ~1e-15. erfc mirrors the same gap.
- AC-6 (REQ-6): `python3 -c "import scipy.special as sp; print(sp.gamma(0.0), sp.gamma(-1.0), sp.gammaln(0.0), sp.gammaln(-1.0))"`
  → `inf nan inf inf`. ferrolearn: `gamma(0.0)=inf` (MATCH), `gamma(-1.0)=-2.565e16`
  (a huge finite garbage value — DIVERGES from `nan`), `lgamma(0.0)=nan`,
  `lgamma(-1.0)=nan` (DIVERGE from `inf`). `erf(10.0)→1`, `erfc(10.0)→~0` match.
- AC-7 (REQ-7): `grep -rn "special::" --include=*.rs ferrolearn-*/src | grep -v 'mod special'`
  returns nothing outside `special.rs` — there is no non-test production consumer.
- AC-8 (REQ-8): the owned special-function computation routes through ferray's
  scipy.special analog (`ferray::stats` / ferray special-functions), with no
  hand-rolled Lanczos/A&S series in `ferrolearn-numerical`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (gamma parity) | SHIPPED | impl `pub fn gamma in special.rs` (Lanczos `g=7,n=9` `LANCZOS_COEFFS` + reflection `PI / ((PI * x).sin() * gamma(1.0 - x))`) mirrors `scipy.special.gamma`. Live oracle (R-CHAR-3): `sp.gamma(5.5)=52.34277778455352`, `sp.gamma(20.0)=1.21645100408832e+17`; ferrolearn agrees to rel ≤ 1.2e-15 / 4.2e-15 (worst-case `gamma(10)` 4.2e-15) over `{1..5,5.5,10,20,0.5,1.5,-0.5,0.1}`. In-crate `gamma_integers`/`gamma_half`/`gamma_reflection_negative` (green). Verification: `cargo test -p ferrolearn-numerical --lib special` → 10 passed, 0 failed. NOTE on the consumer: see REQ-7 — `gamma` has NO non-test workspace consumer; this REQ is the numerical-parity slice that ships on impl+oracle, but the module as a whole is gated by REQ-7/REQ-8. |
| REQ-2 (lgamma parity) | SHIPPED | impl `pub fn lgamma in special.rs` (same Lanczos core, log form `0.5*ln(2π) + (z+0.5)*ln t − t + ln a`, reflection `ln π − ln|sin πx| − lgamma(1−x)`) mirrors `scipy.special.gammaln`. Live oracle: `sp.gammaln(0.5)=0.5723649429247`, `sp.gammaln(20.0)=39.339884187199495`; ferrolearn agrees to rel ≤ 7.8e-16 (exact at `7.5/12/20`). In-crate `lgamma_matches_gamma` (green). Same consumer caveat as REQ-1 (REQ-7). |
| REQ-3 (digamma parity) | SHIPPED | impl `pub fn digamma in special.rs` (recurrence `ψ(x)=ψ(x+1)−1/x` to `z>=6` + 5-term Bernoulli asymptotic + reflection `digamma(1−z) − π/tan(πz)`) mirrors `scipy.special.digamma`/`psi`. Live oracle: `sp.digamma(0.5)=-1.9635100260214235`, `sp.digamma(1.0)=-0.5772156649015329`, `sp.digamma(10.0)=2.251752589066721`; ferrolearn agrees to rel ≤ 1.7e-12 at `0.5`, worst-case ~2.1e-11 at `digamma(2.0)` (the 5-term Bernoulli truncation floor — well inside a 1e-10 contract, below scipy's ~1e-15 but a tight, deterministic match). In-crate `digamma_known_values` (green). Same consumer caveat (REQ-7). |
| REQ-4 (beta/lbeta parity) | SHIPPED | impl `pub fn beta in special.rs` (`lbeta(a,b).exp()`) and `pub fn lbeta in special.rs` (`lgamma(a)+lgamma(b)-lgamma(a+b)`) mirror `scipy.special.beta`/`betaln`. Live oracle: `sp.beta(2,3)=0.08333333333333333`, `sp.beta(0.5,0.5)=3.1415926535897927`, `sp.betaln(2,3)=-2.4849066497880004`; ferrolearn agrees to rel ≤ 2.7e-15 / 1.1e-15. In-crate `beta_symmetry`/`beta_known_value` (green). Inherits lgamma's pole gap (REQ-6) and consumer gap (REQ-7). |
| REQ-5 (erf/erfc parity) | SHIPPED | FIXED #1942: `pub fn erf`/`erfc in special.rs` now delegate to `libm::erf`/`libm::erfc` (Cephes/musl, machine precision) — replacing Abramowitz & Stegun 7.1.26 (~1.5e-7). NOTE: `statrs` 0.18's erf is only ~1e-11 (fails a 1e-12 pin), so `libm` (added as a direct dep) was used instead. Live oracle: ferrolearn `erf(1.0)=0.8427007929497148` matches `scipy.special.erf` ≤1e-12 across the range incl. negatives (`erf(-3)`, odd symmetry) and the tail (`erfc(10)≈2.088e-45`, rel ~2e-16). Pinned by `divergence_erf_accuracy`/`divergence_erfc_accuracy` + `green_erf_range_matches_scipy`/`green_erfc_tail_matches_scipy`. (R-DEV-1 numerical contract.) |
| REQ-6 (pole/inf/nan edge parity) | SHIPPED | FIXED #1943: `gamma` returns `NaN` at negative integers (guard `x < 0.0 && x == x.floor()`) and `+inf` at `x=0` (matching `scipy.special.gamma`); `lgamma` returns `+inf` at every non-positive integer (matching `scipy.special.gammaln`, was `NaN`). Negative-NON-integer reflection (`gamma(-2.5)`, `gammaln(-2.5)`) unaffected (≤1e-12). Plus large-x: `gamma` now evaluates in log-space for x≥0.5 (FIXED #1946) so `gamma(171)=7.257e306` is finite (`+inf` at ≥172), matching scipy instead of overflowing prematurely. Pinned by `divergence_gamma_negative_integer_pole_is_nan`/`divergence_lgamma_nonpositive_integer_pole_is_pos_inf`/`divergence_gamma_large_x_overflows` + `green_gamma_poles_*`/`green_erf_inf_and_gamma_zero_pole`. (R-DEV-1 NaN/Inf handling.) |
| REQ-7 (non-test production consumer) | NOT-STARTED | open prereq blocker #1944. `grep -rn "special::" --include=*.rs ferrolearn-*/src` returns NOTHING outside `special.rs`; `lib.rs` exposes only `pub mod special` (no re-export). No estimator, no `ferrolearn-python` registration consumes `gamma`/`lgamma`/`digamma`/`beta`/`erf`. Crucially, the crates that DO need these functions bypass this module: `ferrolearn-kernel/src/gp_classifier.rs` calls `statrs::function::erf::erf` directly, and `ferrolearn-decomp`/`ferrolearn-kernel` hand-roll their own kernel gamma. S5 grandfathering does NOT rescue this REQ: `special::gamma` is an internal substrate helper, not a boundary estimator type (`LinearRegression`/`StandardScaler`) — there are no external users and no Python binding for it. With zero in-workspace consumers AND a live alternative (`statrs`) already in use elsewhere, the honest call (R-HONEST-3) is NOT-STARTED: this module is currently dead code. The fix is to make the estimators that need special functions route through `special::*` (or to fold the module into ferray per REQ-8). |
| REQ-8 (ferray substrate) | NOT-STARTED | open prereq blocker #1945. `special.rs` is a hand-rolled scipy.special reimplementation in `ferrolearn-numerical` using only `std::f64`. Per R-SUBSTRATE-1 the scipy.special analog is a **ferray** concern (`ferray::stats` / a ferray special-functions module — the scipy.special-on-ferray layer), and `ferrolearn-numerical` should consume it rather than hand-roll Lanczos/A&S series. ferray does not yet expose this surface (R-SUBSTRATE-5: a ferray gap is real work, filed upstream to ferray; the ferrolearn unit is NOT-STARTED on this REQ until ferray ships the special-functions module). Linked to REQ-5 — once on ferray, the erf accuracy comes from ferray's (machine-precision) implementation, not A&S 7.1.26. |

## Architecture

`special.rs` is a flat module of free `f64 -> f64` functions; there is no
unfitted/Fitted split (these are pure mathematical functions, not estimators).
It deviates from the project's `F: Float + Send + Sync + 'static` generic bound
(CLAUDE.md) — every function is `f64`-only. This is a substrate detail rather
than a divergence from scipy: `scipy.special.{gamma,erf,...}` operate on float64,
and ferrolearn's downstream consumers (had they existed) compute in `f64`. The
generic-`F` gap is noted but not a scipy-parity divergence; the destination
(ferray special functions, REQ-8) would set the dtype contract.

The numerical core is the shared **Lanczos** approximation
(`LANCZOS_G = 7.0`, the 9-element `LANCZOS_COEFFS` — the same coefficient set
scipy/Cephes-family implementations use): `gamma` returns
`√(2π)·t^(z+0.5)·e^(−t)·a` with `t = z + g + 0.5` for `x ≥ 0.5`, and the
reflection `Γ(x)Γ(1−x) = π/sin(πx)` for `x < 0.5`; `lgamma` is the log of the
same expression with the log-domain reflection `ln π − ln|sin πx| − lgamma(1−x)`.
`beta`/`lbeta` are pure compositions of `lgamma`, so they inherit its accuracy
(REQ-4 SHIPPED) AND its pole behavior (REQ-6 NOT-STARTED). `digamma` uses the
recurrence `ψ(x) = ψ(x+1) − 1/x` to push the argument to `z ≥ 6`, then the
asymptotic `ψ(z) ≈ ln z − 1/(2z) − Σ B_{2n}/(2n z^{2n})` with five Bernoulli
terms `[1/12, −1/120, 1/252, −1/240, 5/660]`; the 5-term truncation is the
source of digamma's ~2e-11 floor (vs scipy ~1e-15) — still a tight match, hence
SHIPPED, but not ULP-exact.

`erf` is the **Abramowitz & Stegun 7.1.26** rational-polynomial approximation:
`t = 1/(1 + 0.327_591_1·|x|)`, a degree-5 Horner polynomial in `t`, times
`e^(−x²)`, with a sign carry. Its documented max error ~1.5e-7 is the headline
divergence (REQ-5): scipy.special.erf is Cephes/boost to ~1e-15. `erfc` avoids
the `1 − erf` cancellation by returning `poly·t·e^(−x²)` directly (good tail
relative accuracy) but carries the SAME ~1.4e-7 absolute error. The concrete fix
is `statrs::function::erf::{erf,erfc}` — machine-precision and already a crate
dependency (so REQ-5 is fixable today, independent of the ferray migration REQ-8).

The module's defining structural fact is REQ-7: it has **no consumer**. It is
re-export-less (`lib.rs`: `pub mod special` only) and grep-clean of callers; the
estimators that need `erf` (e.g. `gp_classifier.rs`) reach for `statrs` directly.
The module is, at baseline `46c522e6e`, dead translation surface — which is why
the cross-cutting consumer/substrate REQs (7, 8) are NOT-STARTED even though five
of the six numerical REQs match scipy.

## Verification

Commands establishing the claims (run at baseline `46c522e6e`):

- `cargo test -p ferrolearn-numerical --lib special` → 10 passed, 0 failed
  (`gamma_integers`, `gamma_half`, `gamma_reflection_negative`,
  `lgamma_matches_gamma`, `lgamma_pole_at_nonpositive_int`, `digamma_known_values`,
  `beta_symmetry`, `beta_known_value`, `erf_known_values`, `erfc_consistent_with_erf`).
  NOTE: `erf_known_values` asserts only a `2e-7` tolerance, so it passes despite
  the ~1.4e-7 divergence — it characterizes the A&S approximation, it does NOT
  pin scipy parity (REQ-5 is NOT-STARTED accordingly).
- gamma/lgamma/digamma/beta oracle (REQ-1..4, R-CHAR-3 — expected from scipy,
  never from ferrolearn):
  `python3 -c "import scipy.special as sp; print(sp.gamma(5.5), sp.gammaln(0.5), sp.digamma(0.5), sp.beta(2,3), sp.betaln(2,3))"`
  → `52.34277778455352 0.5723649429247 -1.9635100260214235 0.08333333333333333 -2.4849066497880004`.
  ferrolearn matches to rel ≤ 1.2e-15 (gamma), 7.8e-16 (lgamma), 1.7e-12 (digamma
  at 0.5; ~2.1e-11 worst-case), 2.7e-15 (beta).
- erf/erfc divergence oracle (REQ-5):
  `python3 -c "import scipy.special as sp; print(sp.erf(1.0), sp.erf(0.5), sp.erfc(2.0))"`
  → `0.8427007929497148 0.5204998778130465 0.004677734981047266`. ferrolearn
  `erf(1.0)=0.8427006897475899` (abs diff 1.03e-7), `erf(0.5)` abs diff 1.385e-7.
  A critic pins this as a FAILING `#[test]` asserting `erf(1.0)` equals
  `0.8427007929497148` to ≤1e-12 (fails until `statrs` replaces A&S 7.1.26).
- pole/edge oracle (REQ-6):
  `python3 -c "import scipy.special as sp; print(sp.gamma(0.0), sp.gamma(-1.0), sp.gammaln(0.0), sp.gammaln(-1.0))"`
  → `inf nan inf inf`. ferrolearn `gamma(0.0)=inf` (match), `gamma(-1.0)=-2.565e16`
  (diverges), `lgamma(0.0)=nan`/`lgamma(-1.0)=nan` (diverge). Pinned as a FAILING
  `#[test]` asserting `gamma(-1.0).is_nan()` and `lgamma(0.0).is_infinite() &&
  lgamma(0.0) > 0.0`.
- consumer check (REQ-7): `grep -rn "special::" --include=*.rs ferrolearn-*/src`
  → empty (no caller). Documented as the blocker, no failing `#[test]` (a
  missing-consumer fact is structural, not a numerical assertion).

Per R-DEFER-2 the table is binary SHIPPED/NOT-STARTED. SHIPPED (impl + scipy
oracle to a tight tolerance): REQ-1 (gamma, incl. log-space large-x #1946),
REQ-2 (lgamma), REQ-3 (digamma), REQ-4 (beta/lbeta), REQ-5 (erf/erfc via `libm`,
FIXED #1942), REQ-6 (pole/inf/nan + large-x, FIXED #1943/#1946). NOT-STARTED
(open `-l blocker` issues): REQ-7 (no non-test consumer — dead module #1944),
REQ-8 (ferray scipy.special substrate #1945).
