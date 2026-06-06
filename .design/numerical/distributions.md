# scipy.stats continuous distributions

<!--
tier: 3-component
status: draft
baseline-commit: 91999d036
upstream-paths:
  - scipy/stats/_continuous_distns.py   # norm/chi2/f/t/beta/gamma generators
  - scipy/stats/_multivariate.py        # dirichlet generator
  - scipy/stats/__init__.py             # documented distribution surface
-->

## Summary

`ferrolearn-numerical/src/distributions.rs` is the scipy-analog substrate that
mirrors a slice of **scipy.stats** continuous distributions: a
[`ContinuousDistribution`] trait (`pdf`/`cdf`/`sf`/`ppf`/`mean`/`variance`) with
six wrappers over the `statrs` crate — [`Normal`] (`scipy.stats.norm`),
[`ChiSquared`] (`scipy.stats.chi2`), [`FDist`] (`scipy.stats.f`), [`StudentsT`]
(`scipy.stats.t`), [`Beta`] (`scipy.stats.beta`), [`Gamma`]
(`scipy.stats.gamma`) — plus the multivariate [`Dirichlet`]
(`scipy.stats.dirichlet`, `sample` + `ln_pdf`), plus four p-value convenience
functions ([`chi2_sf`], [`f_sf`], [`t_test_two_tailed`], [`norm_sf`]). scipy.stats
continuous distributions are Cephes/boost C accurate to ~machine precision; the
distribution math is version-stable, so the installed **scipy 1.17.1** is a valid
live oracle for the true values (the sklearn-1.5.2 / scipy-1.17.1 version split
does not matter — `chi2.ppf(0.95, 2)` is `5.991464547107979` in every scipy
release).

Divergence classes measured against the live scipy 1.17.1 oracle at a TIGHT
(~1e-10) tolerance — NOT the LOOSE 1e-2/1e-3 in-crate test tolerances, which mask
real gaps:

1. **pdf/cdf/sf parity** — chi2, f, t, beta, gamma match scipy to ~1e-15
   (machine precision) across body and tails; **`Normal`'s cdf/sf is the one
   numerical divergence** (statrs's `Normal` cdf is an erf approximation accurate
   to only ~1e-11, worst observed rel ~8e-11 in the tail; `Normal` pdf/ppf are
   machine-precision).
2. **ppf / inverse_cdf accuracy** — contrary to the expected concern, statrs's
   `inverse_cdf` is machine-precision for chi2/f/t/beta/gamma (rel ≤ ~3e-15),
   because statrs root-finds against its own accurate cdf; `Normal::ppf` uses the
   closed form (rel ~1e-16). ppf is SHIPPED.
3. **gamma parameter convention** (R-DEV-2 porting hazard) — ferrolearn
   `Gamma::new(shape, rate)` (statrs rate convention) vs scipy.stats
   `gamma(a=shape, scale=1/rate)`. Documented; oracle-confirmed equivalent.
4. **panic-in-production** (R-CODE-2 / R-APG-1) — `unwrap_stat` calls `panic!`;
   the four convenience functions `.expect(...)` on invalid params;
   `Dirichlet::ln_pdf` `assert!`s on bad input; `Dirichlet::sample` `.expect(...)`.
   scipy returns `nan` for invalid params (e.g. `chi2.sf(5, -1) → nan`) rather
   than raising from the sf call. NOT-STARTED.
5. **convenience-function parity** — `chi2_sf`/`f_sf`/`t_test_two_tailed`/
   `norm_sf` match the corresponding scipy call (modulo the Normal-cdf gap that
   `norm_sf` inherits).
6. **Dirichlet** — `ln_pdf` matches `scipy.stats.dirichlet.logpdf` to ~1e-14;
   `sample` is numpy-RNG-substrate-blocked for exact value parity.
7. **error type / substrate / consumer** — `Result<_, String>` instead of
   `FerroError` (R-DEV-2); `statrs`/`rand_distr` instead of the `ferray::stats` /
   `ferray::random` substrate (R-SUBSTRATE-1); and **no non-test production
   consumer** in the workspace.

## Upstream reference (scipy.stats, live oracle scipy 1.17.1)

Distribution generators live in `scipy/stats/_continuous_distns.py`
(`norm_gen` :394 / `norm = norm_gen(...)` :506; `chi2_gen` :1608 / `chi2 = ...`
:1696; `f_gen` :2524 / `f = ...` :2634; `t_gen` :7977 / `t = ...` :8091;
`beta_gen` :708 / `beta = ...` :970; `gamma_gen` :3558 / `gamma = ...` :3768) and
`scipy/stats/_multivariate.py` (`dirichlet_gen` :2260 / `dirichlet = ...` :2542);
the documented surface is `scipy/stats/__init__.py`. The numerical kernels are
Cephes/boost C, so cite the scipy.stats **distribution names** and **live-oracle
values**, never C line numbers.

The gamma parameterization is fixed at `_continuous_distns.py:3573` — pdf
`x^{a-1} e^{-x} / Γ(a)` with `a` the shape — and `:3591-3592` documents the
two-parameter form with `scale = 1/beta` (so scipy `scale` is statrs's
`1/rate`).

Live oracle (`cd /tmp && python3 -c "import scipy.stats as st; ..."`, scipy 1.17.1):

- `norm`: `pdf(0)=0.3989422804014327`, `cdf(1.5)=0.9331927987311419`,
  `sf(2)=0.022750131948179195`, `sf(8)=6.22096057427174e-16`,
  `ppf(0.975)=1.959963984540054`.
- `chi2`: `cdf(5.991,2)=0.9499883849734209`, `sf(5.991,2)=0.05001161502657909`,
  `ppf(0.95,2)=5.991464547107979`, `ppf(0.99,5)=15.08627246938899`.
- `f`: `sf(3,2,10)=0.09536743164062497`, `sf(100,2,10)=2.448519270213934e-07`,
  `ppf(0.95,2,10)=4.1028210151304`.
- `t`: `sf(2,5)=0.050969739414929174`, `sf(30,5)=3.8593243102480265e-07`,
  `ppf(0.975,5)=2.5705818356363146`.
- `beta`: `cdf(0.5,2,3)=0.6875`, `pdf(0.5,2,3)=1.5000000000000004`,
  `ppf(0.95,2,3)=0.7513953742698181`.
- `gamma`: `cdf(2,a=2,scale=1)=0.5939941502901616`,
  `pdf(2,a=2,scale=1)=0.2706705664732254`,
  `cdf(4,a=3,scale=2)=0.32332358381693654`,
  `ppf(0.95,a=2,scale=1)=4.743864518390577`.
- `dirichlet`: `logpdf([0.2,0.3,0.5],[2,3,4])=2.0228711901914433`.
- invalid params: `chi2.sf(5,-1)=nan`, `chi2.sf(5,0)=nan`,
  `norm(0,-1).pdf(0)=nan` (scipy returns `nan`, does NOT raise from the call).

## Requirements

- REQ-1: `Normal` pdf/cdf/sf/ppf parity with `scipy.stats.norm`. `pdf`/`ppf` to
  ~machine precision; `cdf`/`sf` to ~machine precision over the whole support.
- REQ-2: `ChiSquared` pdf/cdf/sf parity with `scipy.stats.chi2`.
- REQ-3: `FDist` pdf/cdf/sf parity with `scipy.stats.f`, including the tail.
- REQ-4: `StudentsT` pdf/cdf/sf parity with `scipy.stats.t`, including the tail.
- REQ-5: `Beta` pdf/cdf/sf parity with `scipy.stats.beta`.
- REQ-6: `Gamma` pdf/cdf/sf parity with `scipy.stats.gamma`, under the
  `(shape, rate) = (a, 1/scale)` convention (R-DEV-2).
- REQ-7: `ppf` (inverse CDF) parity. `ChiSquared`/`FDist`/`StudentsT`/`Beta`/
  `Gamma`/`Normal` `ppf(p)` match `scipy.stats.<name>.ppf` to a tight tolerance.
- REQ-8: Convenience p-value functions. `chi2_sf`/`f_sf`/`t_test_two_tailed`/
  `norm_sf` match `scipy.stats.chi2.sf` / `f.sf` / `2*t.sf(|t|,df)` / `norm.sf`.
- REQ-9: `Dirichlet` log-PDF parity with `scipy.stats.dirichlet.logpdf`.
- REQ-10: Invalid-parameter / domain handling — no panic in production
  (R-CODE-2). scipy returns `nan` for invalid params; ferrolearn must not
  `panic!`/`.expect`/`assert!` on user-reachable input.
- REQ-11: Error-type contract (R-DEV-2). Constructors return
  `Result<_, FerroError>` (the workspace error type), not `Result<_, String>`.
- REQ-12: ferray substrate (R-SUBSTRATE-1). Distributions route through
  `ferray::stats` (the scipy.stats analog) and sampling through `ferray::random`
  (the numpy.random analog), not `statrs`/`rand_distr`.
- REQ-13: Non-test production consumer (R-DEFER-1). A non-test caller in the
  workspace (an estimator, or the `ferrolearn-python` binding) consumes
  `distributions::*`.

## Acceptance criteria

All expected values come from the live scipy 1.17.1 oracle (R-CHAR-3), run from
`/tmp`, NEVER copied from ferrolearn. Tolerances are the TIGHT measured ones, not
the LOOSE in-crate test tolerances.

- AC-1 (REQ-1): `python3 -c "import scipy.stats as st; print(st.norm.pdf(0), st.norm.cdf(1.5), st.norm.sf(2.0), st.norm.ppf(0.975))"`
  → `0.3989422804014327 0.9331927987311419 0.022750131948179195 1.959963984540054`.
  ferrolearn `Normal::new(0,1)`: `pdf(0)` rel 1.4e-16, `ppf(0.975)` rel 1.1e-16
  (PASS ≤1e-12) — but `cdf(1.5)` rel **3.1e-12**, `sf(2)` rel **4.5e-11**,
  `sf(8)` rel **5.3e-11**, `cdf(1.0)` rel **1.6e-11** (FAIL ≤1e-12). The
  statrs `Normal` cdf is the headline divergence.
- AC-2 (REQ-2): `python3 -c "import scipy.stats as st; print(st.chi2.cdf(5.991,2), st.chi2.sf(5.991,2))"`
  → `0.9499883849734209 0.05001161502657909`; `ChiSquared::new(2)` matches to rel
  ≤ 4.2e-16 (cdf exact, sf rel 4.2e-16).
- AC-3 (REQ-3): `python3 -c "import scipy.stats as st; print(st.f.sf(3.0,2,10), st.f.sf(100.0,2,10))"`
  → `0.09536743164062497 2.448519270213934e-07`; `FDist::new(2,10).sf` matches to
  rel ≤ 7.4e-15 (body) / 1.9e-15 (tail) — no statrs F-tail divergence.
- AC-4 (REQ-4): `python3 -c "import scipy.stats as st; print(st.t.sf(2.0,5), st.t.sf(30.0,5))"`
  → `0.050969739414929174 3.8593243102480265e-07`; `StudentsT::new(5).sf` matches
  to rel ≤ 1.5e-15 (body) / 1.4e-15 (tail) — no statrs t-tail divergence.
- AC-5 (REQ-5): `python3 -c "import scipy.stats as st; print(st.beta.cdf(0.5,2,3), st.beta.pdf(0.5,2,3))"`
  → `0.6875 1.5000000000000004`; `Beta::new(2,3)` matches to rel ≤ 1.0e-15.
- AC-6 (REQ-6): `python3 -c "import scipy.stats as st; print(st.gamma.cdf(2.0,a=2,scale=1.0), st.gamma.cdf(4.0,a=3,scale=2.0))"`
  → `0.5939941502901616 0.32332358381693654`. ferrolearn `Gamma::new(2.0, 1.0)`
  (rate=1=1/scale) `.cdf(2.0)` rel 1.9e-16; `Gamma::new(3.0, 0.5)` (rate=0.5=1/2)
  `.cdf(4.0)` rel 3.6e-15 — confirming `rate = 1/scale`.
- AC-7 (REQ-7): `python3 -c "import scipy.stats as st; print(st.chi2.ppf(0.95,2), st.f.ppf(0.95,2,10), st.t.ppf(0.975,5), st.beta.ppf(0.95,2,3), st.gamma.ppf(0.95,a=2,scale=1.0))"`
  → `5.991464547107979 4.1028210151304 2.5705818356363146 0.7513953742698181
  4.743864518390577`. ferrolearn `ppf`: chi2 rel 3.0e-16, f rel 3.2e-15, t rel
  1.7e-16, beta rel 5.9e-16, gamma rel 1.9e-16; `Normal::ppf(0.975)` rel 1.1e-16
  (all PASS ≤1e-12 — statrs `inverse_cdf` is machine-precision here).
- AC-8 (REQ-8): `python3 -c "import scipy.stats as st; print(st.chi2.sf(5.991,2), st.f.sf(3.0,2,10), 2*st.t.sf(2.0,5), st.norm.sf(2.0))"`
  → `0.05001161502657909 0.09536743164062497 0.10193947882985835
  0.022750131948179195`. ferrolearn `chi2_sf(5.991,2)` rel 4.2e-16,
  `f_sf(3,2,10)` rel 7.4e-15, `t_test_two_tailed(2,5)` rel 1.5e-15 — PASS;
  `norm_sf(2.0)` rel 4.5e-11 (inherits the AC-1 Normal-cdf gap — FAIL ≤1e-12).
- AC-9 (REQ-9): `python3 -c "import scipy.stats as st; print(st.dirichlet.logpdf([0.2,0.3,0.5],[2.0,3.0,4.0]))"`
  → `2.0228711901914433`; `Dirichlet::new(&[2,3,4]).ln_pdf(&[0.2,0.3,0.5])` rel
  7.0e-15.
- AC-10 (REQ-10): `python3 -c "import scipy.stats as st; print(st.chi2.sf(5,-1))"`
  → `nan` (scipy returns `nan`, does NOT raise from the sf call). ferrolearn
  `chi2_sf(5.0, -1.0)` **panics** (`ChiSquared::new(-1.0)` is `Err`, then
  `.expect("chi2_sf: invalid df")`); `unwrap_stat`, `Dirichlet::ln_pdf` `assert!`,
  and `Dirichlet::sample` `.expect` are the other production panics. A critic pins
  this with a `#[test]` asserting `chi2_sf` returns/propagates a non-panicking
  result for invalid `df` (fails until the `.expect` becomes a `Result`/`nan`).
- AC-11 (REQ-11): `Normal::new`/`ChiSquared::new`/… return `Result<_, String>`
  (grep the signatures), not `Result<_, FerroError>`; a critic pins a `#[test]`
  asserting the error type is `FerroError` (fails until the substrate migration).
- AC-12 (REQ-12): the owned distribution math routes through `ferray::stats` /
  `ferray::random`; `grep -n "statrs\|rand_distr" distributions.rs` is empty
  (fails until the ferray substrate ships REQ-12).
- AC-13 (REQ-13): `grep -rn "distributions::\|chi2_sf\|f_sf\|norm_sf\|t_test_two_tailed\|ContinuousDistribution" --include=*.rs ferrolearn-*/src`
  returns nothing referencing **this** module outside `distributions.rs` (the
  model-sel `crate::distributions` hits are a DIFFERENT module — model-sel's own
  hyperparameter `Distribution`/`Uniform`/`IntUniform`/`Choice`/`LogUniform`); the
  numerical distributions module has no non-test consumer.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (Normal parity) | SHIPPED | FIXED #1965: `Normal::cdf`/`sf` now compute via `libm::erf`/`erfc` (machine precision: `cdf = 0.5(1+erf((x-μ)/(σ√2)))`, `sf = 0.5·erfc((x-μ)/(σ√2))`), replacing statrs's ~1e-11 erf approximation (the single numerical divergence in the file). Now matches `scipy.stats.norm` ≤1e-13 (was `norm.sf(2)` rel 4.5e-11, `cdf(1.5)` rel 3.1e-12). `Normal` gained `mean`/`std_dev` fields; `pdf`/`ppf`/`mean`/`variance` unchanged (already exact). `norm_sf` inherits the fix. Pinned by `red_normal_cdf_sf_machine_precision`/`red_norm_sf_machine_precision`. |
| REQ-2 (ChiSquared parity) | SHIPPED | impl `impl ContinuousDistribution for ChiSquared in distributions.rs` (`statrs::distribution::ChiSquared`) mirrors `scipy.stats.chi2`. Live oracle (R-CHAR-3): `chi2.cdf(5.991,2)=0.9499883849734209` (exact match), `chi2.sf(5.991,2)=0.05001161502657909` (rel 4.2e-16). In-crate `chi2_cdf` (green). Verification: `cargo test -p ferrolearn-numerical --lib distributions` → all passed. NOTE: no non-test consumer (see REQ-13); this is the numerical-parity slice. |
| REQ-3 (FDist parity) | SHIPPED | impl `impl ContinuousDistribution for FDist in distributions.rs` (`statrs::distribution::FisherSnedecor`) mirrors `scipy.stats.f`. Live oracle: `f.sf(3,2,10)=0.09536743164062497` (rel 7.4e-15), `f.sf(100,2,10)=2.448519270213934e-07` (tail, rel 1.9e-15) — no F-tail divergence. In-crate `f_dist_sf` (green, but `epsilon=1e-2` — characterization only). Consumer caveat: REQ-13. |
| REQ-4 (StudentsT parity) | SHIPPED | impl `impl ContinuousDistribution for StudentsT in distributions.rs` (`statrs::distribution::StudentsT` with location 0, scale 1, freedom `df`) mirrors `scipy.stats.t`. Live oracle: `t.sf(2,5)=0.050969739414929174` (rel 1.5e-15), `t.sf(30,5)=3.8593243102480265e-07` (tail, rel 1.4e-15) — no t-tail divergence. In-crate `t_dist_symmetry` (green). Consumer caveat: REQ-13. |
| REQ-5 (Beta parity) | SHIPPED | impl `impl ContinuousDistribution for Beta in distributions.rs` (`statrs::distribution::Beta`) mirrors `scipy.stats.beta`. Live oracle: `beta.cdf(0.5,2,3)=0.6875` (rel 3.2e-16), `beta.pdf(0.5,2,3)=1.5000000000000004` (rel 1.0e-15). In-crate `beta_mean_variance` (green). Consumer caveat: REQ-13. |
| REQ-6 (Gamma parity + convention) | SHIPPED | impl `impl ContinuousDistribution for Gamma in distributions.rs` (`statrs::distribution::Gamma`, `pub fn new(shape, rate)`) mirrors `scipy.stats.gamma` (a=shape, scale=1/rate; cited `_continuous_distns.py:3591-3592` `scale = 1/beta`). Live oracle: `gamma.cdf(2,a=2,scale=1)=0.5939941502901616` vs `Gamma::new(2.0,1.0).cdf(2.0)` rel 1.9e-16; `gamma.cdf(4,a=3,scale=2)=0.32332358381693654` vs `Gamma::new(3.0,0.5).cdf(4.0)` (rate=1/2) rel 3.6e-15 — confirming `rate=1/scale`. The convention difference is a porting hazard (R-DEV-2), documented; numerically SHIPPED. Consumer caveat: REQ-13. |
| REQ-7 (ppf / inverse_cdf parity) | SHIPPED | impl `fn ppf` on each wrapper delegates to `statrs ... ContinuousCDF::inverse_cdf` (Normal closed-form; chi2/f/t/beta/gamma root-found against statrs's own cdf). Live oracle (R-CHAR-3): `chi2.ppf(0.95,2)=5.991464547107979` rel 3.0e-16; `chi2.ppf(0.99,5)=15.08627246938899` rel 9.4e-16; `f.ppf(0.95,2,10)=4.1028210151304` rel 3.2e-15; `t.ppf(0.975,5)=2.5705818356363146` rel 1.7e-16; `beta.ppf(0.95,2,3)=0.7513953742698181` rel 5.9e-16; `gamma.ppf(0.95,a=2,scale=1)=4.743864518390577` rel 1.9e-16; `norm.ppf(0.975)` rel 1.1e-16. statrs's `inverse_cdf` is machine-precision — the anticipated ppf-imprecision divergence did NOT materialize. In-crate `gamma_ppf_round_trip` (green). Consumer caveat: REQ-13. |
| REQ-8 (convenience p-value fns) | SHIPPED (parity) — but see REQ-1/REQ-10 | impl `pub fn chi2_sf`/`pub fn f_sf`/`pub fn t_test_two_tailed`/`pub fn norm_sf in distributions.rs`. Live oracle: `chi2_sf(5.991,2)` rel 4.2e-16, `f_sf(3,2,10)` rel 7.4e-15, `t_test_two_tailed(2,5)=2*t.sf(2,5)` rel 1.5e-15 (all match scipy). `norm_sf(2.0)` rel 4.5e-11 — INHERITS the Normal-cdf gap (REQ-1) so `norm_sf` specifically is below the 1e-12 contract. All four also panic on invalid params (REQ-10). In-crate `chi2_sf_matches_cdf`/`convenience_p_values` (green at loose tolerance). The chi2/f/t convenience values are SHIPPED; `norm_sf`'s precision is gated by REQ-1 and all four are gated by the panic REQ-10. |
| REQ-9 (Dirichlet ln_pdf parity) | SHIPPED | impl `pub fn ln_pdf in distributions.rs` (`ln_gamma(Σα) − Σ ln_gamma(α_i) + Σ(α_i−1)ln x_i`) mirrors `scipy.stats.dirichlet.logpdf` (`_multivariate.py:2260` dirichlet_gen). Live oracle: `dirichlet.logpdf([0.2,0.3,0.5],[2,3,4])=2.0228711901914433` vs `Dirichlet::new(&[2,3,4]).ln_pdf(&[0.2,0.3,0.5])` rel 7.0e-15. NOTE: `ln_pdf` itself `assert!`s on bad input (folded into REQ-10); `Dirichlet::sample` value-parity vs `numpy.random` is substrate-blocked (REQ-12), but the deterministic `ln_pdf` value is SHIPPED. Consumer caveat: REQ-13. |
| REQ-10 (no-panic on invalid params) | NOT-STARTED | PARTIALLY addressed: `chi2_sf`/`f_sf`/`t_test_two_tailed` now return `f64::NAN` on invalid params matching scipy (FIXED #1966, was `.expect` panic; `norm_sf` never panicked). RESIDUAL (the open REQ): `unwrap_stat` (`mean`/`variance`) still `panic!`s, and `Dirichlet::ln_pdf` `assert!`s on length/domain/sum + `Dirichlet::sample` `.expect`s — all R-CODE-2/R-APG-1 on user-reachable paths where scipy returns `nan`. Fix: return `Result`/`nan` from those too. Blocker #1970. |
| REQ-11 (FerroError error type) | SHIPPED (#1967) | FIXED — all 7 constructors (`Normal`/`ChiSquared`/`FDist`/`StudentsT`/`Beta`/`Gamma`/`Dirichlet` `::new`) now return `Result<Self, FerroError>`; the 8 `Err` sites map to `FerroError::InvalidParameter { name, reason }` (messages preserved; `name` = the validated param), mirroring scipy.stats' `ValueError`/nan on invalid distribution parameters (CLAUDE.md / R-CODE-2). `ferrolearn-core` is a workspace dep (added #1961). The type-agnostic `Err(_) => f64::NAN` arms in the pdf/cdf/ppf helpers are unchanged. Guard `distributions::tests::distribution_constructors_invalid_params_return_ferroerror` (`matches!` on `Err(FerroError::InvalidParameter)` for all 7). Note: this resolves the error-type independently of REQ-12 (ferray substrate, #1968) — the swap did NOT require waiting for ferray::stats. |
| REQ-12 (ferray substrate) | NOT-STARTED | open prereq blocker (to be filed by critic). `distributions.rs` is built on the WRONG substrate (R-SUBSTRATE-1/3): `use statrs::distribution::...` for all six wrappers + `ln_gamma`, and `rand_distr::Gamma` + `rand` for `Dirichlet::sample`. The destination is `ferray::stats` (the scipy.stats analog) for the distributions and `ferray::random` (the numpy.random analog) for sampling. ferray does not yet expose this surface (R-SUBSTRATE-5: a ferray gap is real work, filed upstream to ferray; the ferrolearn unit is NOT-STARTED on this REQ until ferray ships the stats/random modules). Linked to REQ-1 (Normal cdf accuracy) and REQ-11 (error type) — both resolve once on ferray. |
| REQ-13 (non-test production consumer) | NOT-STARTED | open prereq blocker (to be filed by critic). `grep -rn "distributions::\|chi2_sf\|f_sf\|norm_sf\|t_test_two_tailed\|ContinuousDistribution" --include=*.rs ferrolearn-*/src` returns NOTHING referencing this module outside `distributions.rs`. The only hits — `ferrolearn-model-sel/src/{random_search,halving_random_search}.rs` `use crate::distributions::{Distribution, Uniform, IntUniform, Choice, LogUniform}` — are a DIFFERENT module (model-sel's own hyperparameter-sampling distributions, NOT `ferrolearn_numerical::distributions`). `lib.rs` exposes only `pub mod distributions` (no re-export, no `ferrolearn-python` registration). S5 grandfathering does NOT rescue this: `ContinuousDistribution`/`chi2_sf`/etc. are internal substrate helpers, not boundary estimator types — no external users, no Python binding. With zero in-workspace consumers, the honest call (R-HONEST-3) is NOT-STARTED: this module is currently dead translation surface. The fix is to make the estimators that need p-values (e.g. feature-selection F-tests, chi2 tests) route through `distributions::*`. |

## Architecture

`distributions.rs` is a flat module: a `ContinuousDistribution` trait
(`pdf`/`cdf`/`sf`/`ppf`/`mean`/`variance`, with `sf` defaulting to `1 - cdf`) and
six newtype wrappers (`Normal`/`ChiSquared`/`FDist`/`StudentsT`/`Beta`/`Gamma`),
each holding a `statrs` `inner` and forwarding through statrs's `Continuous`
(pdf), `ContinuousCDF` (cdf/sf/inverse_cdf) and `statistics::Distribution`
(mean/variance) traits. There is no unfitted/Fitted split — distributions are
parameterized values, not estimators. The module deviates from the project's
`F: Float + Send + Sync + 'static` generic bound (CLAUDE.md): every method is
`f64`-only, matching scipy.stats's float64 contract; the destination dtype
contract is set by the ferray substrate (REQ-12).

Numerically, statrs is an excellent scipy.stats mirror for five of the six
univariate distributions: chi2/f/t/beta/gamma cdf/sf/pdf AND `inverse_cdf` (ppf)
all match scipy to ~1e-15 across body and tail (REQ-2..7), because statrs computes
these through the regularized incomplete gamma/beta functions and root-finds ppf
against its own cdf. The **one numerical divergence is `Normal::cdf`** (REQ-1):
statrs evaluates the Gaussian cdf via an erf approximation accurate only to
~1e-11 (worst observed rel ~8e-11 at `sf(4)`), where scipy's `ndtr` is
machine-precision. `Normal::pdf` (closed-form `exp`) and `Normal::ppf`
(closed-form `ndtri`-style inverse) are unaffected (~1e-16). Because `norm_sf`
wraps `Normal::sf`, it inherits the gap (REQ-8). The fix mirrors `special.rs`
REQ-5: route the Gaussian cdf/sf through `libm::erf`/`erfc` (machine-precision),
or — preferably — through the ferray `stats` substrate (REQ-12).

`Dirichlet` (REQ-9) is hand-rolled rather than statrs-backed: `ln_pdf` evaluates
the closed-form `ln Γ(Σα) − Σ ln Γ(α_i) + Σ(α_i−1) ln x_i` via
`statrs::function::gamma::ln_gamma` (matching `scipy.stats.dirichlet.logpdf` to
~1e-14), and `sample` uses the standard Gamma-normalize algorithm over
`rand_distr::Gamma` (value-parity with `numpy.random` is substrate-blocked,
REQ-12).

The module's three structural facts are the cross-cutting NOT-STARTED REQs.
**Panic-in-production (REQ-10):** `unwrap_stat` `panic!`s; the four convenience
functions `.expect` on invalid params (`chi2_sf(5.0,-1.0)` panics where scipy
returns `nan`); `Dirichlet::ln_pdf` `assert!`s and `Dirichlet::sample` `.expect`s
— all R-CODE-2 / R-APG-1 violations on user-reachable input. **Error type
(REQ-11):** constructors return `Result<_, String>`, not `FerroError`.
**No consumer (REQ-13):** nothing in the workspace calls this module; the
model-sel `distributions` hits are an unrelated module. At baseline `91999d036`
this is dead translation surface — which is why the consumer/substrate/error-type
REQs are NOT-STARTED even though seven of the nine numerical REQs match scipy to
machine precision.

## Verification

Commands establishing the claims (run at baseline `91999d036`; expected values
from the live scipy 1.17.1 oracle per R-CHAR-3, never from ferrolearn):

- `cargo test -p ferrolearn-numerical --lib distributions` → all passing
  (`normal_pdf_cdf_ppf`, `chi2_cdf`, `f_dist_sf`, `t_dist_symmetry`,
  `beta_mean_variance`, `gamma_ppf_round_trip`, `dirichlet_sample_sums_to_one`,
  `chi2_sf_matches_cdf`, `convenience_p_values`). NOTE: these use LOOSE
  tolerances (`epsilon = 1e-2`/`1e-3`/`1e-6`), so they pass DESPITE the
  `Normal::cdf` ~1e-11 divergence (REQ-1) — they characterize statrs, they do NOT
  pin scipy parity at machine precision.
- chi2/f/t/beta/gamma cdf-sf-pdf oracle (REQ-2..6):
  `python3 -c "import scipy.stats as st; print(st.chi2.sf(5.991,2), st.f.sf(3.0,2,10), st.t.sf(2.0,5), st.beta.cdf(0.5,2,3), st.gamma.cdf(2.0,a=2,scale=1.0))"`
  → `0.05001161502657909 0.09536743164062497 0.050969739414929174 0.6875
  0.5939941502901616`. ferrolearn matches to rel ≤ 4.2e-16 (chi2), 7.4e-15 (f),
  1.5e-15 (t), 3.2e-16 (beta), 1.9e-16 (gamma).
- ppf oracle (REQ-7):
  `python3 -c "import scipy.stats as st; print(st.chi2.ppf(0.95,2), st.f.ppf(0.95,2,10), st.t.ppf(0.975,5), st.beta.ppf(0.95,2,3), st.gamma.ppf(0.95,a=2,scale=1.0))"`
  → `5.991464547107979 4.1028210151304 2.5705818356363146 0.7513953742698181
  4.743864518390577`. ferrolearn rel ≤ 3.2e-15 across all five + `Normal::ppf`
  rel 1.1e-16. A critic green-pins `chi2.ppf(0.95,2)` to ≤1e-12.
- Normal divergence oracle (REQ-1, the headline):
  `python3 -c "import scipy.stats as st; print(st.norm.cdf(1.5), st.norm.sf(2.0), st.norm.sf(8.0))"`
  → `0.9331927987311419 0.022750131948179195 6.22096057427174e-16`. ferrolearn
  `Normal::new(0,1)`: `cdf(1.5)` rel 3.1e-12, `sf(2)` rel 4.5e-11, `sf(8)` rel
  5.3e-11. A critic pins this as a FAILING `#[test]` asserting `cdf(1.5)` equals
  `0.9331927987311419` to ≤1e-12 (fails until the Gaussian cdf is routed through
  `libm`/ferray).
- dirichlet oracle (REQ-9):
  `python3 -c "import scipy.stats as st; print(st.dirichlet.logpdf([0.2,0.3,0.5],[2.0,3.0,4.0]))"`
  → `2.0228711901914433`; ferrolearn `ln_pdf` rel 7.0e-15.
- panic oracle (REQ-10):
  `python3 -c "import scipy.stats as st; print(st.chi2.sf(5,-1))"` → `nan`
  (no raise). ferrolearn `chi2_sf(5.0, -1.0)` panics. A critic pins a `#[test]`
  asserting invalid `df` does not panic and yields `nan`/`Err` (fails until the
  `.expect` becomes a `Result`).
- error-type check (REQ-11): grep the constructor signatures —
  `Result<Self, String>` everywhere; a critic pins `FerroError` (fails until the
  substrate migration).
- substrate check (REQ-12): `grep -n "statrs\|rand_distr" distributions.rs` →
  non-empty (statrs distributions + `rand_distr::Gamma`); the destination is
  `ferray::stats`/`ferray::random`. ferray gap → filed upstream (R-SUBSTRATE-5).
- consumer check (REQ-13): `grep -rn "distributions::\|chi2_sf\|f_sf\|norm_sf\|t_test_two_tailed\|ContinuousDistribution" --include=*.rs ferrolearn-*/src`
  → only `distributions.rs` itself + model-sel's UNRELATED `crate::distributions`
  module. No non-test consumer of this module. Structural blocker, no failing
  `#[test]`.

Per R-DEFER-2 the table is binary SHIPPED/NOT-STARTED. SHIPPED (impl + scipy
oracle to a tight ~1e-15 tolerance): REQ-1 (Normal cdf/sf via `libm` FIXED #1965),
REQ-2 (chi2), REQ-3 (f), REQ-4 (t), REQ-5 (beta), REQ-6 (gamma + convention),
REQ-7 (ppf), REQ-8 (convenience fns — now incl. `nan`-on-invalid #1966 and the
machine-precision `norm_sf`), REQ-9 (Dirichlet ln_pdf). NOT-STARTED (open
blockers): REQ-10 (residual panic-in-production: `unwrap_stat`/Dirichlet #1970),
REQ-11 (FerroError type #1967), REQ-12 (ferray stats/random substrate #1968),
REQ-13 (no non-test consumer — dead module #1969).
