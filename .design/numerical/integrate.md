# scipy.integrate numerical quadrature

<!--
tier: 3-component
status: draft
baseline-commit: ee8fccfea
upstream-paths:
  - scipy/integrate/__init__.py     # documented quadrature + ODE function surface
-->

## Summary

`ferrolearn-numerical/src/integrate.rs` is the scipy-analog substrate that
mirrors a slice of **scipy.integrate**'s numerical quadrature surface. It exposes
two families:

- **Adaptive Simpson's rule** — `quad` / `quad_with_limit` (the
  `scipy.integrate.quad` analog): recursive interval bisection with a
  Richardson-extrapolation error estimate `(s_left + s_right − s)/15`, a
  depth/tolerance-halving stopping rule, and a default maximum recursion depth of
  50.
- **Fixed-order Gauss-Legendre** — `gauss_legendre` /
  `gauss_legendre_composite` (the `scipy.integrate.fixed_quad` analog): an
  `n`-point rule for `n ∈ 1..=20`. Orders 1–10 use hardcoded high-precision
  node/weight tables (`gl_table`); orders 11–20 are computed on the fly via the
  **Golub-Welsch** algorithm (`golub_welsch`) — the Jacobi tridiagonal
  eigenproblem solved by a hand-rolled implicit-shift QR iteration
  (`trid_qr_eigen` / `implicit_qr_step` / `wilkinson_shift`).

The `n`-point Gauss-Legendre rule is **unique**, so it has a strict ground truth:
numpy's `numpy.polynomial.legendre.leggauss` (= scipy's `fixed_quad` rule). The
adaptive Simpson `quad` uses a *different* algorithm than scipy's QUADPACK
Gauss-Kronrod, so its contract is **value-to-tolerance** (both approximate the
same true integral), not bit-exactness. Quadrature of fixed integrands is
deterministic and version-stable, so the installed **scipy 1.17.1** (+
`numpy.polynomial.legendre.leggauss`) is a valid live oracle for the true values
(the sklearn-1.5.2 / scipy-1.17.1 version split does not matter — the
Gauss-Legendre rule and `∫₀¹ eˣ dx` are the same in every release).

Divergence classes: (1) **GL-value-parity** — `gauss_legendre(f,a,b,n)` matches
the unique exact GL rule (numpy `leggauss`) to ~machine precision for ALL orders
1–20, exercising both the hardcoded tables AND the Golub-Welsch path (SHIPPED);
(2) **quad-value-convergence** — adaptive Simpson converges to
`scipy.integrate.quad(...)[0]` within the requested `tol` (SHIPPED, value-to-tol,
NOT bit-exact with QUADPACK); (3) **API-contract** — `quad(f,a,b,tol)` /
`gauss_legendre(...) -> Result<_, String>` diverge from scipy's
`quad(...) -> (value, abserr)` tuple and `fixed_quad`'s vectorized-func / `epsabs`
/ `epsrel` / `limit` / `points` signature; (4) **missing-functions** —
`trapezoid`, `simpson`, `romberg`, `cumulative_trapezoid`, `cumulative_simpson`,
`newton_cotes`, `dblquad`/`tplquad`/`nquad`, `quad_vec`, `qmc_quad`, and the whole
ODE-solver family (`solve_ivp`/`odeint`/`RK45`/…) are absent (NOT-STARTED);
(5) **infinite-bounds/singularities/weights** — `quad` handles only finite
`[a,b]` smooth integrands, vs scipy's `±inf` bounds, `points=` singularities, and
`weight=` factors (NOT-STARTED); (6) **error-type** — `Result<_, String>` vs
`FerroError`; (7) **no-consumer** — nothing in the workspace calls `integrate::`;
(8) **ferray-substrate** — the destination is a `ferray`/`scipy.integrate`-analog
module, and the Golub-Welsch eigenproblem should route through `ferray::linalg`,
not a hand-rolled tridiagonal QR in `ferrolearn-numerical`.

## Upstream reference (scipy.integrate, live oracle scipy 1.17.1)

The documented function surface lives in `scipy/integrate/__init__.py`. The
numerical kernels are QUADPACK (Fortran, `quad`) and compiled C/Fortran, so cite
the scipy.integrate **function names** and the **live-oracle values**, never the
Fortran line numbers. Documented surface (`scipy/integrate/__init__.py`):

- General quadrature: `quad` (`:14`), `quad_vec` (`:15`), `dblquad` (`:17`),
  `tplquad` (`:18`), `nquad` (`:19`), `fixed_quad` (`:21`), `newton_cotes`
  (`:22`), `qmc_quad` (`:24`).
- Sample-based: `trapezoid` (`:34`), `cumulative_trapezoid` (`:35`), `simpson`
  (`:36`), `cumulative_simpson` (`:37`).
- ODE solvers: `solve_ivp` (`:63`), `RK23` (`:64`), `RK45` (`:65`), `DOP853`
  (`:66`), `Radau` (`:67`), `BDF` (`:68`), `LSODA` (`:69`), `odeint` (`:87`).

Live oracle (`cd /tmp && python3 -c "import numpy as np, scipy.integrate as si; ..."`,
scipy 1.17.1, the exact GL rule via `numpy.polynomial.legendre.leggauss`):

- GL exact rule, `∫₀¹ eˣ dx` (true value `e − 1 = 1.718281828459045`), all
  orders converge: `gl(exp,0,1,1)=1.6487212707001282`,
  `…,2=1.717896378007504`, `…,3=1.718281004372522`,
  `…,5=1.7182818284583914`, `…,8=1.7182818284590453`,
  `…,20=1.7182818284590449`.
- GL exact rule, `∫₀² cos x dx` (true value `sin 2 = 0.9092974268256817`):
  `gl(cos,0,2,1)=1.0806046117362795`, `…,3=0.9093306976211131`,
  `…,5=0.9092974272532763`, `…,8=0.9092974268256815`,
  `…,20=0.9092974268256825`.
- `si.quad(np.exp,0,1)[0] = 1.7182818284590453` (= `e − 1`).
- `si.quad(np.sin,0,np.pi)[0] = 2.0`.
- `si.quad(lambda x:1/(1+x*x),0,1)[0] = 0.7853981633974484` (= `π/4 =
  0.7853981633974483`).
- `si.fixed_quad(np.exp,0,1,n=5)[0] = 1.7182818284583916` (the 5-point rule —
  *not* the true value; the GL rule's own approximation).

## Requirements

- REQ-1: **gauss_legendre value parity (all orders 1–20).** `gauss_legendre(f,a,b,n)`
  equals the unique exact `n`-point Gauss-Legendre rule (numpy
  `leggauss` / `scipy.integrate.fixed_quad`) to ~machine precision for EVERY
  `n ∈ 1..=20`. This is the strict check — it exercises both the hardcoded
  `gl_table` (1–10) AND the `golub_welsch` Golub-Welsch path (11–20).
- REQ-2: **gauss_legendre_composite.** `gauss_legendre_composite(f,a,b,n_points,n_panels)`
  partitions `[a,b]` into `n_panels` equal sub-intervals and sums the
  `n_points`-point GL rule on each — equal to the panel-wise sum of the exact GL
  rule, improving accuracy for smooth functions.
- REQ-3: **quad value convergence.** Adaptive Simpson `quad(f,a,b,tol)` converges
  to the true integral = `scipy.integrate.quad(f,a,b)[0]` within the requested
  `tol` (value-to-tolerance — the ALGORITHM differs from scipy's QUADPACK
  Gauss-Kronrod, so it is NOT bit-exact, but the VALUE agrees to tol).
- REQ-4: **quad API / return contract.** scipy's `quad` returns a `(value,
  abserr)` TUPLE and takes `epsabs`/`epsrel`/`limit`/`points`/`weight`/infinite
  bounds; ferrolearn `quad(f,a,b,tol)` takes a single absolute `tol` and returns
  a `QuadratureResult{value, error_estimate, n_evals}`. `fixed_quad` returns
  `(value, None)` and requires a VECTORIZED func; `gauss_legendre` takes a scalar
  `Fn(f64)->f64` and returns `Result<QuadratureResult, String>`. (R-DEV-2/3:
  match scipy's user-API ABI / output object contract.)
- REQ-5: **missing scipy.integrate functions.** `trapezoid`, `simpson`,
  `romberg`, `cumulative_trapezoid`, `cumulative_simpson`, `newton_cotes`,
  `dblquad`/`tplquad`/`nquad`, `quad_vec`, `qmc_quad`, and the ODE-solver family
  (`solve_ivp`/`odeint`/`RK45`/`RK23`/`DOP853`/`Radau`/`BDF`/`LSODA`) have no
  ferrolearn analog.
- REQ-6: **infinite bounds / singularities / weight functions.** scipy's `quad`
  handles `±inf` bounds, singular `points=`, and `weight=` factors; ferrolearn's
  adaptive Simpson handles only finite `[a,b]` smooth integrands.
- REQ-7: **error type.** Public errors are `Result<_, String>`
  (`gauss_legendre`/`gauss_legendre_composite`/`gl_nodes_weights`), not
  `ferrolearn_core::error::FerroError` (CLAUDE.md / R-CODE-2: the library error
  contract is `FerroError`). `quad` cannot signal error at all (returns a bare
  `QuadratureResult`).
- REQ-8: **non-test production consumer.** A non-test caller in the workspace (an
  estimator, or the `ferrolearn-python` binding) consumes `integrate::*` so it is
  part of the live translation surface.
- REQ-9: **ferray substrate (R-SUBSTRATE-1).** The Golub-Welsch tridiagonal
  eigenproblem (orders 11–20) should route through `ferray::linalg`'s symmetric
  eigensolver rather than the hand-rolled `trid_qr_eigen`/`implicit_qr_step`, and
  the scipy.integrate analog ultimately belongs on the ferray substrate.

## Acceptance criteria

All expected values come from the live scipy/numpy oracle (R-CHAR-3), never from
ferrolearn. Run from `/tmp`.

- AC-1 (REQ-1): `python3 -c "import numpy as np; x,w=np.polynomial.legendre.leggauss(5); print(0.5*np.sum(w*np.exp(0.5*x+0.5)))"`
  → `1.7182818284583914`. For EVERY `n ∈ 1..=20`, `gauss_legendre(exp,0,1,n)`
  equals the numpy `leggauss` `gl(exp,0,1,n)` and `gauss_legendre(cos,0,2,n)`
  equals `gl(cos,0,2,n)` to abs ≤ 1e-13. Observed max abs diff over all 20 orders:
  **1.11e-15** (`exp`) / **2.00e-15** (`cos`) — both well inside the bound, the
  Golub-Welsch orders 11–20 included (worst GL order: `n=15` exp 1.11e-15,
  `n=20` cos 2.00e-15). `gauss_legendre(exp,0,1,5)` matches
  `scipy.integrate.fixed_quad(np.exp,0,1,n=5)[0] = 1.7182818284583916` to
  2.22e-16.
- AC-2 (REQ-2): `python3 -c "import numpy as np; print(2.0)"` (`∫₀^π sin = 2`);
  `gauss_legendre_composite(sin,0,π,5,10)` = `2.0` exactly (observed diff 0.0,
  10 panels of the exact 5-point rule).
- AC-3 (REQ-3): `python3 -c "import numpy as np, scipy.integrate as si; print(si.quad(np.exp,0,1)[0], si.quad(np.sin,0,np.pi)[0], si.quad(lambda x:1/(1+x*x),0,1)[0])"`
  → `1.7182818284590453 2.0 0.7853981633974484`. `quad(exp,0,1,1e-10)` agrees to
  abs **6.66e-16** (129 evals); `quad(sin,0,π,1e-10)` to **8.88e-16** (473 evals);
  `quad(1/(1+x²),0,1,1e-10)` to **1.68e-14** (229 evals) — each ≤ its `1e-10`
  tol. NOT bit-exact with QUADPACK (different algorithm); value-to-tol holds.
- AC-4 (REQ-4): `python3 -c "import numpy as np, scipy.integrate as si; r=si.quad(np.exp,0,1); print(type(r), len(r))"`
  → `<class 'tuple'> 2` (returns `(value, abserr)`). ferrolearn `quad` returns a
  3-field `QuadratureResult{value, error_estimate, n_evals}` and takes a single
  scalar `tol` (no `epsabs`/`epsrel`/`limit`/`points`). `si.fixed_quad`'s `func`
  must accept an array; `gauss_legendre`'s `f` is `Fn(f64)->f64`. Structural API
  divergence, documented (R-DEV-2/3).
- AC-5 (REQ-5): `python3 -c "import scipy.integrate as si; print([s for s in ('trapezoid','simpson','romberg','cumulative_trapezoid','cumulative_simpson','newton_cotes','dblquad','tplquad','nquad','quad_vec','solve_ivp','odeint') if hasattr(si,s)])"`
  lists every one as present in scipy; `grep -n "pub fn" ferrolearn-numerical/src/integrate.rs`
  shows ferrolearn exposes only `quad`, `quad_with_limit`, `gauss_legendre`,
  `gauss_legendre_composite` — none of the listed functions exist.
- AC-6 (REQ-6): `python3 -c "import numpy as np, scipy.integrate as si; print(si.quad(lambda x:np.exp(-x*x),-np.inf,np.inf)[0])"`
  → `1.7724538509055159` (= `√π`, infinite bounds). ferrolearn's `quad` cannot
  represent `±inf` bounds (`(b-a)` becomes `inf`/`nan` and the recursion produces
  `nan`); there is no `points=`/`weight=` argument.
- AC-7 (REQ-7): `gauss_legendre(|x| x, 0.0, 1.0, 0)` and `…, 21` return
  `Err(String)` (`"Gauss-Legendre order must be in 1..=20, got 0"`), and
  `gauss_legendre_composite(…, 0)` returns `Err("n_panels must be at least 1")`.
  No `FerroError` variant is used; `quad` has no error channel. The library error
  contract (CLAUDE.md / `ferrolearn_core::error::FerroError`) is not satisfied.
- AC-8 (REQ-8): `grep -rn "integrate::\|gauss_legendre\|quad_with_limit\|QuadratureResult" --include=*.rs ferrolearn-*/src | grep -v 'src/integrate.rs'`
  returns NOTHING — there is no non-test production consumer. `lib.rs` exposes
  only `pub mod integrate` (no re-export, no estimator caller).
- AC-9 (REQ-9): `grep -n "ferray\|ferray::linalg" ferrolearn-numerical/src/integrate.rs`
  returns nothing — the Golub-Welsch eigenproblem is solved by a hand-rolled
  `trid_qr_eigen`/`implicit_qr_step` instead of routing through `ferray::linalg`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (gauss_legendre value parity, all orders 1–20) | SHIPPED | impl `pub fn gauss_legendre in integrate.rs` (→ `gl_nodes_weights` → `gl_table` for 1–10, `golub_welsch` for 11–20 → `gl_evaluate`) mirrors `scipy.integrate.fixed_quad` / numpy `leggauss`. Live oracle (R-CHAR-3, numpy `leggauss`, NEVER copied from ferrolearn): `gl(exp,0,1,5)=1.7182818284583914`, `gl(exp,0,1,8)=1.7182818284590453`, `gl(cos,0,2,8)=0.9092974268256815`. Verified for EVERY `n ∈ 1..=20`: observed max abs diff **1.11e-15** (`exp`, worst `n=15`) / **2.00e-15** (`cos`, worst `n=20`), both ≪ 1e-13 — the hardcoded tables AND the Golub-Welsch orders 11–20 are correct. `gauss_legendre(exp,0,1,5)` vs `scipy.integrate.fixed_quad(np.exp,0,1,n=5)[0]=1.7182818284583916` → diff 2.22e-16. In-crate `gauss_legendre_exact_polynomial`/`gauss_legendre_order_15`/`gauss_legendre_order_20`/`golub_welsch_weights_sum_to_two` (green). Consumer caveat: see REQ-8 — no non-test workspace consumer; this REQ ships on impl + the unique-rule oracle, but the module as a whole is gated by REQ-8/REQ-9. |
| REQ-2 (gauss_legendre_composite) | SHIPPED | impl `pub fn gauss_legendre_composite in integrate.rs` (panel loop summing `gl_evaluate` per equal sub-interval) mirrors a composite `fixed_quad`. Live oracle: `∫₀^π sin x dx = 2.0`; `gauss_legendre_composite(sin,0,π,5,10)` = `2.0` (observed abs diff **0.0** — 10 panels of the exact 5-point rule). In-crate `composite_gl_accuracy` (`epsilon = 1e-14`, green), `composite_gl_zero_panels_error` (green). Same consumer caveat (REQ-8). |
| REQ-3 (quad value convergence) | SHIPPED | impl `pub fn quad`/`pub fn quad_with_limit in integrate.rs` (adaptive Simpson via `SimpsonContext::adaptive_recurse`, Richardson error `(s_left+s_right−s)/15`, depth/tol halving) converges to the same true integral as `scipy.integrate.quad`. Live oracle: `si.quad(np.exp,0,1)[0]=1.7182818284590453`, `si.quad(np.sin,0,np.pi)[0]=2.0`, `si.quad(lambda x:1/(1+x*x),0,1)[0]=0.7853981633974484`. ferrolearn `quad(…,1e-10)`: exp diff **6.66e-16** (129 evals), sin diff **8.88e-16** (473 evals), arctan diff **1.68e-14** (229 evals) — each ≤ tol. NOT bit-exact with QUADPACK Gauss-Kronrod (different algorithm); the contract is value-to-tolerance and it holds. In-crate `quad_constant`/`quad_linear`/`quad_polynomial`/`quad_sin`/`quad_gaussian`/`quad_tight_tolerance`/`quad_counts_evals` (green). Same consumer caveat (REQ-8). |
| REQ-4 (quad API / return contract) | NOT-STARTED | open prereq blocker (to be filed by critic). `scipy.integrate.quad` returns a `(value, abserr)` TUPLE and takes `epsabs`/`epsrel`/`limit`/`points`/`weight` + infinite bounds (`scipy/integrate/__init__.py:14`); ferrolearn `quad(f,a,b,tol)` takes a single absolute `tol` and returns a 3-field `QuadratureResult{value, error_estimate, n_evals}` — no `epsrel`, no `limit`, no `points`/`weight`. `scipy.integrate.fixed_quad` (`:21`) requires a VECTORIZED `func` and returns `(value, None)`; ferrolearn `gauss_legendre` takes a scalar `Fn(f64)->f64` and returns `Result<_, String>`. The user-API ABI (R-DEV-2) and output-object contract (R-DEV-3) diverge from scipy. |
| REQ-5 (missing scipy.integrate functions) | NOT-STARTED | open prereq blocker (to be filed by critic). `scipy/integrate/__init__.py` documents `trapezoid` (`:34`), `cumulative_trapezoid` (`:35`), `simpson` (`:36`), `cumulative_simpson` (`:37`), `newton_cotes` (`:22`), `romberg` (historically; deprecated/removed in newer scipy — superseded by `quad`), `dblquad`/`tplquad`/`nquad` (`:17`–`:19`), `quad_vec` (`:15`), `qmc_quad` (`:24`), and the ODE family `solve_ivp`/`RK23`/`RK45`/`DOP853`/`Radau`/`BDF`/`LSODA`/`odeint` (`:63`–`:87`). ferrolearn exposes only `quad`/`quad_with_limit`/`gauss_legendre`/`gauss_legendre_composite` — NONE of those exist. The closest sklearn-relevant analogs are the sample-based `trapezoid`/`simpson` (`metrics.auc` uses the trapezoidal rule); the ODE solvers are a large separate surface. |
| REQ-6 (infinite bounds / singularities / weight functions) | NOT-STARTED | open prereq blocker (to be filed by critic). Live oracle: `si.quad(lambda x:np.exp(-x*x),-np.inf,np.inf)[0]=1.7724538509055159` (= `√π`, infinite bounds via QUADPACK's `qagi`). ferrolearn `quad`/`gl_evaluate` compute `(b−a)` / `f64::midpoint(a,b)` literally, so `±inf` bounds produce `nan`; there is no `points=` singularity handling and no `weight=` factor argument. Only finite `[a,b]` smooth integrands are supported. |
| REQ-7 (error type — FerroError) | NOT-STARTED | open prereq blocker (to be filed by critic). `gauss_legendre`/`gauss_legendre_composite`/`gl_nodes_weights` return `Result<_, String>` (e.g. `"Gauss-Legendre order must be in 1..=20, got {n}"`), not `ferrolearn_core::error::FerroError` (CLAUDE.md: "All public functions return `Result<T, FerroError>`"; R-CODE-2). `quad`/`quad_with_limit` return a bare `QuadratureResult` with NO error channel — a non-convergent integrand (depth exhausted) is reported as a value + `error_estimate`, never an error. The crate-wide error contract is not satisfied. |
| REQ-8 (non-test production consumer) | NOT-STARTED | open prereq blocker (to be filed by critic). `grep -rn "integrate::\|gauss_legendre\|quad_with_limit\|QuadratureResult" --include=*.rs ferrolearn-*/src` returns NOTHING outside `src/integrate.rs`; `lib.rs` exposes only `pub mod integrate` (no re-export). No estimator and no `ferrolearn-python` registration consumes `quad`/`gauss_legendre`. S5 grandfathering does NOT rescue this REQ: these are internal substrate helpers, not a boundary estimator type (`LinearRegression`/`StandardScaler`) — no external users and no Python binding for them. With zero in-workspace consumers, the honest call (R-HONEST-3) is NOT-STARTED: the module is currently dead code. The fix is to route an estimator that needs quadrature (or a `metrics`-style AUC trapezoid) through `integrate::*`, or to fold the module into ferray per REQ-9. |
| REQ-9 (ferray substrate) | NOT-STARTED | open prereq blocker (to be filed by critic). `integrate.rs` uses only `std::f64` — no array substrate. The Golub-Welsch path (orders 11–20) is a HAND-ROLLED symmetric-tridiagonal eigensolver (`golub_welsch` → `trid_qr_eigen` → `implicit_qr_step` + `wilkinson_shift`). Per R-SUBSTRATE-1 the eigenproblem is a `ferray::linalg` concern (the `numpy.linalg`/`scipy.linalg` analog), and the scipy.integrate analog ultimately belongs on the ferray substrate. ferray does not yet expose a routed symmetric-eigensolver entry point for this use (R-SUBSTRATE-5: a ferray gap is real work, filed upstream to ferray; the ferrolearn unit is NOT-STARTED on this REQ until ferray ships it). Do NOT silently keep the hand-rolled QR as the destination. |

## Architecture

`integrate.rs` is a flat module of free functions over `Fn(f64) -> f64`
integrands; there is no unfitted/Fitted split (these are pure numerical routines,
not estimators) and no generic `F: Float` bound — everything is `f64`-only, a
substrate detail rather than a scipy divergence (`scipy.integrate.quad` operates
on float64). The only public type is `QuadratureResult{value, error_estimate,
n_evals}`.

**Adaptive Simpson (`quad`/`quad_with_limit`).** `quad_with_limit` pre-evaluates
`f(a)`, `f(b)`, `f(m)` (via `SimpsonContext::eval`, which also increments
`n_evals`), forms the whole-interval Simpson estimate `s = (b−a)/6·(fa+4fm+fb)`
(`simpson_from_values`), and hands a `SimpsonInterval` to
`SimpsonContext::adaptive_recurse`. The recursion bisects, computes the two
half-interval Simpson estimates `s_left`/`s_right`, and forms the Richardson error
`error = (s_left + s_right − s)/15`. It stops when `depth == 0` or
`|error| < tol`, returning the corrected value `s_left + s_right + error`;
otherwise it recurses on each half with `tol/2` and `depth−1`, summing values and
error estimates. The default depth is 50 (`DEFAULT_MAX_DEPTH`). This is a textbook
adaptive-Simpson method — distinct from scipy's QUADPACK adaptive Gauss-Kronrod
(`qagse`), hence REQ-3 is value-to-tolerance, not bit-exact.

**Gauss-Legendre (`gauss_legendre`/`gauss_legendre_composite`).** `gl_nodes_weights`
dispatches on order: 1–10 read the hardcoded `gl_table` (Abramowitz & Stegun /
DLMF node/weight constants at full f64 precision), 11–20 call `golub_welsch`.
`gl_evaluate` maps the reference `[-1,1]` nodes onto `[a,b]` via
`x = (b−a)/2·t + midpoint(a,b)` and accumulates `(b−a)/2·Σ wᵢ f(xᵢ)`. The
composite variant partitions `[a,b]` into `n_panels` equal panels and sums the
per-panel rule. Because the `n`-point GL rule is unique, the live numpy `leggauss`
rule is an exact oracle: all 20 orders match to ≤ 2e-15 (REQ-1/REQ-2 SHIPPED).

**Golub-Welsch (`golub_welsch`).** For orders 11–20 the nodes are the eigenvalues
of the symmetric tridiagonal Jacobi matrix for Legendre polynomials (zero
diagonal, off-diagonal `βᵢ = i/√(4i²−1)`), and the weights are `2·v_i[0]²` from
the eigenvectors. The eigenproblem is solved by a HAND-ROLLED implicit-shift QR
iteration (`trid_qr_eigen` with deflation + Wilkinson shift, `implicit_qr_step`
bulge-chase). This is the locus of REQ-9: per R-SUBSTRATE-1 a symmetric-tridiagonal
eigensolve is a `ferray::linalg` concern, not a hand-rolled QR in
`ferrolearn-numerical`. It works (weights sum to 2 to 1e-12; values match the
oracle to 2e-15) but it is the wrong substrate.

The module's defining structural fact is REQ-8: it has **no consumer**. It is
re-export-less (`lib.rs`: `pub mod integrate` only) and grep-clean of callers. At
baseline `ee8fccfea` it is dead translation surface — which is why the
cross-cutting REQs (4 API, 5 missing-functions, 6 infinite-bounds, 7 error-type,
8 consumer, 9 substrate) are NOT-STARTED even though the three numerical REQs
(1 GL parity, 2 composite, 3 quad convergence) match the live oracle.

## Verification

Commands establishing the claims (run at baseline `ee8fccfea`):

- `cargo test -p ferrolearn-numerical --lib integrate` → all pass
  (`quad_constant`, `quad_linear`, `quad_polynomial`, `quad_sin`,
  `quad_gaussian`, `gauss_legendre_exact_polynomial`, `gauss_legendre_sin`,
  `composite_gl_accuracy`, `quad_tight_tolerance`, `gauss_legendre_order_15`,
  `gauss_legendre_order_20`, `gauss_legendre_invalid_order`,
  `composite_gl_zero_panels_error`, `quad_counts_evals`,
  `golub_welsch_weights_sum_to_two`). NOTE: these in-crate tests use
  closed-form expected values (`2/9`, `1/6`, `2/29`, weights-sum-to-2); they
  do NOT compare against the numpy `leggauss` oracle per-order — that comparison
  is the critic's REQ-1 pin below.
- GL value-parity oracle (REQ-1, R-CHAR-3 — expected from numpy `leggauss`,
  NEVER from ferrolearn):
  `python3 -c "import numpy as np; gl=lambda f,a,b,n:(lambda x,w:0.5*(b-a)*np.sum(w*f(0.5*(b-a)*x+0.5*(b+a))))(*np.polynomial.legendre.leggauss(n)); print([repr(gl(np.exp,0,1,n)) for n in range(1,21)])"`
  → e.g. `n=5 → 1.7182818284583914`, `n=8 → 1.7182818284590453`,
  `n=20 → 1.7182818284590449`. A critic pins, for every `n ∈ 1..=20`, a
  `#[test]` asserting `gauss_legendre(exp,0,1,n).value` (and `cos[0,2]`) equals
  the numpy value to abs ≤ 1e-13. Observed (this iteration): max diff
  **1.11e-15** (exp) / **2.00e-15** (cos) — PASSES.
- quad value-convergence oracle (REQ-3):
  `python3 -c "import numpy as np, scipy.integrate as si; print(si.quad(np.exp,0,1)[0], si.quad(np.sin,0,np.pi)[0], si.quad(lambda x:1/(1+x*x),0,1)[0])"`
  → `1.7182818284590453 2.0 0.7853981633974484`. `quad(…,1e-10)` matches to abs
  6.66e-16 / 8.88e-16 / 1.68e-14 — each ≤ tol (NOT bit-exact with QUADPACK;
  value-to-tol).
- `fixed_quad` cross-check (REQ-1):
  `python3 -c "import numpy as np, scipy.integrate as si; print(si.fixed_quad(np.exp,0,1,n=5)[0])"`
  → `1.7182818284583916`; `gauss_legendre(exp,0,1,5).value` matches to 2.22e-16.
- infinite-bounds oracle (REQ-6):
  `python3 -c "import numpy as np, scipy.integrate as si; print(si.quad(lambda x:np.exp(-x*x),-np.inf,np.inf)[0])"`
  → `1.7724538509055159` (= `√π`). ferrolearn `quad(…,-inf,inf,…)` cannot
  represent infinite bounds → `nan`. Documented as the blocker; the
  finite-domain `quad(exp(-x²),-5,5)` (`quad_gaussian` test) approximates `√π`
  to 1e-8 only because the tails are truncated.
- consumer check (REQ-8): `grep -rn "integrate::\|gauss_legendre\|quad_with_limit\|QuadratureResult" --include=*.rs ferrolearn-*/src | grep -v 'src/integrate.rs'`
  → empty (no caller). Documented as the blocker, no failing `#[test]` (a
  missing-consumer fact is structural, not a numerical assertion).
- substrate check (REQ-9): `grep -n "ferray" ferrolearn-numerical/src/integrate.rs`
  → empty; the Golub-Welsch eigensolve is the hand-rolled `trid_qr_eigen`.

Per R-DEFER-2 the table is binary SHIPPED/NOT-STARTED. SHIPPED (impl + live
numpy/scipy oracle to a tight tolerance): REQ-1 (gauss_legendre value parity, all
20 orders incl. Golub-Welsch), REQ-2 (gauss_legendre_composite), REQ-3 (quad
value convergence). NOT-STARTED (open `-l blocker` issues to be filed by the
critic): REQ-4 (quad API/return tuple vs `QuadratureResult`), REQ-5 (missing
`trapezoid`/`simpson`/`newton_cotes`/`quad_vec`/`dblquad`/ODE family), REQ-6
(infinite bounds / singularities / weight functions), REQ-7 (`String` error vs
`FerroError`; `quad` has no error channel), REQ-8 (no non-test consumer — dead
module), REQ-9 (ferray `linalg` substrate for the Golub-Welsch eigenproblem).
