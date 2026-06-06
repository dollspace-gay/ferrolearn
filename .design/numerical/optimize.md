# scipy.optimize.minimize / minimize_scalar — smooth-objective optimizers

<!--
tier: 3-component
status: draft
baseline-commit: bc82e38c8
upstream-paths:
  - scipy/optimize/__init__.py     # documented minimize/minimize_scalar method + function surface
-->

## Summary

`ferrolearn-numerical/src/optimize.rs` is the scipy-analog substrate that mirrors
a slice of **scipy.optimize.minimize** (multivariate) and
**scipy.optimize.minimize_scalar** (univariate, bounded). It exposes four
optimizers:

- **`NewtonCG`** — truncated Newton with a conjugate-gradient inner solve
  (`cg_solve`, Eisenstat-Walker forcing term, negative-curvature early exit) and
  an Armijo `backtracking_line_search`. Mirrors
  `scipy.optimize.minimize(method='newton-cg')`.
- **`TrustRegionNCG`** — trust-region Newton-CG with the Steihaug-Toint CG
  subproblem (`steihaug_cg` + `boundary_step`) and an actual/predicted-reduction
  radius update. Mirrors `scipy.optimize.minimize(method='trust-ncg')`.
- **`Powell`** — derivative-free direction-set method (Numerical Recipes §10.5):
  per-sweep line minimisation (`line_minimise_powell` → `bracket_min_powell` +
  `golden_section_powell`) with the NR extrapolation test and direction
  replacement. Mirrors `scipy.optimize.minimize(method='powell')`.
- **`brent_bounded`** — Brent's method for 1-D bounded minimisation on `[a,b]`
  (golden-section + parabolic interpolation). Mirrors
  `scipy.optimize.minimize_scalar(method='bounded')`. Returns `Minimize1DResult{
  x, fun, nfev, success}`.

`NewtonCG`/`TrustRegionNCG`/`Powell` return `Result<OptimizeResult, String>`
where `OptimizeResult{x, fun, grad, n_iter, converged}`.

Each optimizer converges to the **unique minimum** of a convex/smooth objective,
so the converged minimiser `x*` has a strict ground truth: a live
`scipy.optimize.minimize(...).x` / `minimize_scalar(...).x` call. The *iteration
path* differs from scipy (ferrolearn's `Powell` is the textbook NR direction-set
method, scipy's is "modified Powell"; ferrolearn's Newton-CG/trust-NCG are
hand-rolled), so the contract is **converged-x\* parity** to a tolerance matching
the optimizer's stopping criterion, NOT iteration-count or path bit-exactness.
These minimisers are deterministic and version-stable for the converged `x*`, so
the installed **scipy 1.17.1** is a valid live oracle (the sklearn-1.5.2 /
scipy-1.17.1 split is irrelevant — the minimum of `0.5(x0²+2x1²)` is `[0,0]` in
every release).

Divergence classes: (1) **converged-x\*-parity** — each of the four optimizers'
returned `x` matches scipy's converged `x*` to a tolerance matching the
optimizer's stopping rule (SHIPPED, all four verified); (2)
**OptimizeResult-attribute contract** — scipy's `OptimizeResult` carries
`x/fun/jac/hess_inv/nfev/njev/nit/status/success/message`; ferrolearn has only
`x/fun/grad/n_iter/converged` (NOT-STARTED); (3) **missing `minimize` methods** —
scipy supports `BFGS/L-BFGS-B/CG/Nelder-Mead/SLSQP/TNC/COBYLA/dogleg/
trust-krylov/trust-exact/trust-constr`; ferrolearn has only newton-cg/trust-ncg/
powell (NOT-STARTED; `L-BFGS-B` is the GPR/GPC hyperparameter optimiser
prerequisite, `lbfgs`/`newton-cg` are LogisticRegression solvers); (4) **missing
scipy.optimize functions** — `root`/`least_squares`/`curve_fit`/`brentq`/`brenth`/
`bisect`/`fsolve`/`linprog`/`nnls`/`root_scalar` and `minimize_scalar`'s
`brent`/`golden` methods are absent (NOT-STARTED); (5) **bounds / constraints** —
scipy `minimize` takes `bounds=`/`constraints=`; ferrolearn's three multivariate
optimizers are UNCONSTRAINED (only `brent_bounded` has an interval) (NOT-STARTED);
(6) **API** — scipy `minimize(fun, x0, jac=, hessp=, method=, tol=, options=)`
keyword interface vs ferrolearn's builder `NewtonCG::new().with_*().minimize(
fun_grad, hessp, x0)` (NOT-STARTED); (7) **error type / convergence signalling** —
`Result<_, String>` + a `converged: bool` flag vs scipy's `success`/`status`/
`message`; no `FerroError` (NOT-STARTED); (8) **consumer** — `brent_bounded` HAS
a real non-test production consumer (`PowerTransformer` Box-Cox/Yeo-Johnson lambda
search), so its consumer REQ is SHIPPED; `NewtonCG`/`TrustRegionNCG`/`Powell` have
NONE (NOT-STARTED); (9) **ferray-substrate** — the vector algebra (`norm`/`dot`,
the CG recurrences, line searches) is hand-rolled on `ndarray::Array1`, not routed
through `ferray::linalg` (NOT-STARTED).

## Upstream reference (scipy.optimize, live oracle scipy 1.17.1)

The documented surface lives in `scipy/optimize/__init__.py`. The numerical
kernels are compiled C/Fortran (MINPACK, L-BFGS-B, etc.), so cite the
scipy.optimize **function/method names** and the **live-oracle values**, never
the Fortran line numbers.

- `minimize` (`scipy/optimize/__init__.py:54`) supports the methods (`:60`–`:74`):
  `neldermead`, `powell`, `cg`, `bfgs`, `newtoncg`, `lbfgsb`, `tnc`, `cobyla`,
  `cobyqa`, `slsqp`, `trustconstr`, `dogleg`, `trustncg`, `trustkrylov`,
  `trustexact`. Signature `minimize(fun, x0, args, method, jac, hess, hessp,
  bounds, constraints, tol, callback, options)`.
- `minimize_scalar` (`:38`) supports `brent`, `bounded`, `golden` (`:44`–`:46`).
  Signature `minimize_scalar(fun, bracket, bounds, args, method, tol, options)`.
- Other documented functions: `root` (`:227`), `least_squares` (`:129`),
  `nnls` (`:137`), `curve_fit` (`:147`), `root_scalar` (`:157`), `brentq`
  (`:158`), `brenth` (`:159`), `bisect` (`:161`), `fsolve` (`:394`),
  `linprog` (`:259`), and `OptimizeResult` (`:25`).

Live oracle (`cd /tmp && python3 -c "import numpy as np; from scipy.optimize
import minimize, minimize_scalar; ..."`, scipy 1.17.1):

- newton-cg / trust-ncg on `f(x)=0.5(x0²+2x1²)` from `[5,3]`
  (`jac=[x0,2x1]`, `hessp(p)=[p0,2p1]`): both → `x = [0.0, 0.0]`, `fun = 0.0`,
  `nit = 7`.
- powell on `(x0−1)²+(x1−2)²` from `[0,0]` → `x = [1.0, 1.9999999999999998]`,
  `fun = 4.93e-32`.
- powell on Rosenbrock `Σ 100(x[1:]−x[:-1]²)²+(1−x[:-1])²` from `[-1.2,1.0]` →
  `x = [1.0000000000000913, 1.0000000000001923]`, `fun = 1.79e-26`.
- `minimize_scalar((x−2)², bounds=(0,5), method='bounded')` → `x = 2.0`,
  `fun = 0.0`.
- `minimize_scalar(sin, bounds=(0,2π), method='bounded')` → `x = 4.7123876779707`
  (≈ `3π/2 = 4.71238898038469`).
- `OptimizeResult` attribute set (from a `BFGS` run):
  `['fun','hess_inv','jac','message','nfev','nit','njev','status','success','x']`.
- `minimize` accepts every method `['newton-cg','trust-ncg','powell','BFGS',
  'L-BFGS-B','CG','Nelder-Mead','SLSQP','TNC','COBYLA','dogleg','trust-krylov',
  'trust-exact','trust-constr']`; `scipy.optimize` exposes
  `['root','least_squares','curve_fit','brentq','brenth','bisect','fsolve',
  'linprog','nnls','minimize','minimize_scalar','root_scalar']`.

## Requirements

- REQ-1: **NewtonCG converged-x\* parity.** `NewtonCG::new().minimize(fun_grad,
  hessp, x0).x` equals `scipy.optimize.minimize(method='newton-cg', jac=, hessp=).x`
  for a convex quadratic — `0.5(x0²+2x1²)` from `[5,3]` → `[0,0]` — to the
  optimizer's `tol` (`1e-8` gradient norm). The minimum is unique → strict check.
- REQ-2: **TrustRegionNCG converged-x\* parity.** `TrustRegionNCG::new().minimize(
  ...).x` equals `scipy.optimize.minimize(method='trust-ncg', jac=, hessp=).x` on
  the same quadratic → `[0,0]`, and on Rosenbrock → `[1,1]`, to the optimizer's
  `tol`. Unique minimum → strict check.
- REQ-3: **Powell converged-x\* parity.** `Powell::new().minimize(f, x0).x`
  equals `scipy.optimize.minimize(method='powell').x` on a shifted quadratic
  `(x0−1)²+(x1−2)²` → `[1,2]` AND on Rosenbrock → `[1,1]` to ~1e-4 (Powell's
  `ftol=1e-8` relative-improvement criterion). NOTE the algorithm differs (NR
  direction-set vs scipy "modified Powell"), so this is converged-x\* parity, not
  iteration parity.
- REQ-4: **brent_bounded converged-x\* parity.** `brent_bounded(f,a,b,tol,
  max_iter).x` equals `scipy.optimize.minimize_scalar(method='bounded',
  bounds=(a,b)).x` on `(x−2)²` over `[0,5]` → `2.0` AND on `sin` over `[0,2π]` →
  `≈4.712` (`3π/2`), to the optimizer's `tol`. Unique minimum on the interval →
  strict check.
- REQ-5: **OptimizeResult attribute contract (R-DEV-3).** scipy's `OptimizeResult`
  carries `x/fun/jac/hess_inv/nfev/njev/nit/status/success/message`; ferrolearn's
  `OptimizeResult` carries only `x/fun/grad/n_iter/converged` (and the scalar
  `Minimize1DResult{x,fun,nfev,success}`). Missing: `nfev`/`njev` (function/jacobian
  eval counts — `Minimize1DResult` has `nfev` but `OptimizeResult` does not),
  `hess_inv`, `status` (integer code), `message` (string); and the names diverge
  (`n_iter` vs `nit`, `converged` vs `success`, `grad` vs `jac`). The output-object
  contract is not mirrored.
- REQ-6: **missing `minimize` methods (R-DEV-2).** scipy.optimize.minimize supports
  `BFGS`, `L-BFGS-B`, `CG`, `Nelder-Mead`, `SLSQP`, `TNC`, `COBYLA`, `dogleg`,
  `trust-krylov`, `trust-exact`, `trust-constr` (`scipy/optimize/__init__.py:60`–
  `:74`). ferrolearn has only `newton-cg`/`trust-ncg`/`powell`. `L-BFGS-B` is the
  hyperparameter optimiser GPR/GPC need (`#1922`/`#1934`); `lbfgs`/`newton-cg` are
  LogisticRegression solvers; `bfgs` backs several sklearn estimators. None of the
  missing methods exist.
- REQ-7: **missing scipy.optimize functions.** `root` (`:227`), `least_squares`
  (`:129`), `curve_fit` (`:147`), `root_scalar` (`:157`) and the scalar
  root-finders `brentq` (`:158`)/`brenth` (`:159`)/`bisect` (`:161`), `fsolve`
  (`:394`), `linprog` (`:259`), `nnls` (`:137`), and `minimize_scalar`'s `brent`
  and `golden` methods (`:44`/`:46`) have no ferrolearn analog. (`nnls` is the
  one sklearn leans on directly — `LinearRegression(positive=True)` /
  `NMF` use a non-negative least squares.)
- REQ-8: **bounds / constraints (R-DEV-2).** scipy.optimize.minimize accepts
  `bounds=` (for `L-BFGS-B`/`TNC`/`SLSQP`/`trust-constr`) and `constraints=`
  (for `SLSQP`/`COBYLA`/`trust-constr`). ferrolearn's `NewtonCG`/`TrustRegionNCG`/
  `Powell` are strictly UNCONSTRAINED — `minimize` takes only `x0`, no bound/
  constraint argument; only `brent_bounded` carries an interval `[a,b]`. Box
  bounds and general constraints are not supported on the multivariate path.
- REQ-9: **API contract (R-DEV-2).** scipy uses a keyword interface
  `minimize(fun, x0, jac=, hessp=, method=, bounds=, constraints=, tol=,
  options=)` / `minimize_scalar(fun, bounds=, method=, tol=, options=)`.
  ferrolearn uses a builder: `NewtonCG::new().with_max_iter().with_tol().
  with_max_cg_iter().minimize(fun_grad, hessp, x0)`, where `fun_grad` returns
  `(f, grad)` as a tuple (vs scipy's separate `fun`/`jac` callables) and
  `brent_bounded(f, a, b, tol, max_iter)` is a free function with positional
  args. The user-API ABI diverges from scipy.
- REQ-10: **error type / convergence signalling (R-DEV-2, R-CODE-2).** The three
  multivariate optimizers return `Result<OptimizeResult, String>` — the only
  `Err` is `"initial guess x0 must have at least one element"`; non-convergence is
  signalled by `converged: false` in an `Ok`, never an `Err`. `brent_bounded`
  returns a bare `Minimize1DResult` (NO error channel; non-convergence is
  `success: false`). scipy signals via `OptimizeResult.success`/`status`/`message`
  (and never raises on non-convergence). Neither uses
  `ferrolearn_core::error::FerroError` (CLAUDE.md: "All public functions return
  `Result<T, FerroError>`"); the `String` error variant does not map to a Python
  exception for the binding boundary. The crate-wide error contract is not
  satisfied.
- REQ-11: **brent_bounded non-test production consumer.** `brent_bounded` is
  consumed by a real non-test estimator so it is part of the live translation
  surface.
- REQ-12: **NewtonCG/TrustRegionNCG/Powell non-test production consumer.** Each
  multivariate optimizer is consumed by a non-test caller (an estimator that
  needs gradient-based or derivative-free unconstrained minimisation, or the
  `ferrolearn-python` binding) so it is part of the live translation surface.
- REQ-13: **ferray substrate (R-SUBSTRATE-1).** The vector algebra (`norm`/`dot`
  over `ndarray::Array1`, the CG/Steihaug recurrences, the Powell line searches,
  and the Brent updates) should route through `ferray::linalg` / `ferray-core`
  rather than the hand-rolled `ndarray` implementation, and the scipy.optimize
  analog ultimately belongs on the ferray substrate.

## Acceptance criteria

All expected values come from the live scipy oracle (R-CHAR-3), never from
ferrolearn. Run from `/tmp`. The converged-x\* tolerance matches each optimizer's
stopping rule (Newton-CG/trust-NCG `tol=1e-8` gradient norm; Powell `ftol=1e-8`
→ x to ~1e-4; Brent `tol`-driven bracket).

- AC-1 (REQ-1): `python3 -c "import numpy as np; from scipy.optimize import
  minimize; g=lambda x:np.array([x[0],2*x[1]]); hp=lambda x,p:np.array([p[0],
  2*p[1]]); f=lambda x:0.5*(x[0]**2+2*x[1]**2); print(minimize(f,[5.,3.],
  method='newton-cg',jac=g,hessp=hp).x.tolist())"` → `[0.0, 0.0]`.
  `NewtonCG::new().minimize(fg,hp,[5,3]).x` = `[0.0, 0.0]` (observed exactly,
  `fun=0`, `n_iter=5`) → abs ≤ 1e-8.
- AC-2 (REQ-2): same harness with `method='trust-ncg'` → `[0.0, 0.0]`; and on
  Rosenbrock `minimize(ros,[-1.2,1.0],method='trust-ncg',jac=,hessp=).x` →
  `[1.0, 1.0]`. `TrustRegionNCG::new().minimize(...).x` = `[0.0, 0.0]` (`n_iter=7`)
  and the in-crate `trust_region_rosenbrock` reaches `[1,1]` to ε=1e-4 → abs ≤ 1e-8
  / 1e-4.
- AC-3 (REQ-3): `python3 -c "import numpy as np; from scipy.optimize import
  minimize; print(minimize(lambda x:(x[0]-1)**2+(x[1]-2)**2,[0.,0.],
  method='powell').x.tolist()); ros=lambda x:sum(100*(x[1:]-x[:-1]**2)**2+
  (1-x[:-1])**2); print(minimize(ros,[-1.2,1.0],method='powell').x.tolist())"`
  → `[1.0, 1.9999999999999998]` and `[1.0000000000000913, 1.0000000000001923]`.
  `Powell::new().minimize(...).x` = `[1.0000000000000002, 2.0]` (quad) and
  `[1.0000000001018314, 1.0000000002080396]` (Rosenbrock, `fun=1.2e-20`,
  `converged`) → abs ≤ 1e-4. (Algorithm differs from scipy's modified Powell —
  converged-x\* parity, not iteration parity.)
- AC-4 (REQ-4): `python3 -c "import numpy as np; from scipy.optimize import
  minimize_scalar; print(minimize_scalar(lambda x:(x-2)**2,bounds=(0,5),
  method='bounded').x); print(minimize_scalar(np.sin,bounds=(0,2*np.pi),
  method='bounded').x)"` → `2.0` and `4.7123876779707`.
  `brent_bounded((x-2)²,0,5,1e-8,500).x` = `2.0` (`fun=0`) and
  `brent_bounded(sin,0,2π,1e-8,500).x` = `4.712388981532386` (`fun=-1`) →
  abs ≤ 1e-6. In-crate `brent_minimize_x_squared`/`brent_minimize_shifted_quadratic`/
  `brent_minimize_sin` pin closed-form minima (`0`, `1.5`, `3π/2`).
- AC-5 (REQ-5): `python3 -c "from scipy.optimize import minimize; print(sorted(
  minimize(lambda x:(x[0]-1)**2,[0.],method='BFGS').keys()))"` →
  `['fun','hess_inv','jac','message','nfev','nit','njev','status','success','x']`.
  `grep -n "pub struct OptimizeResult" -A8 ferrolearn-numerical/src/optimize.rs`
  shows fields `x/fun/grad/n_iter/converged` only — no
  `nfev/njev/hess_inv/status/message`; `nit`→`n_iter`, `success`→`converged`,
  `jac`→`grad`.
- AC-6 (REQ-6): `python3 -c "from scipy.optimize import minimize; import numpy as
  np; [minimize(lambda x:(x[0]-1)**2,[0.],method=m,jac=lambda x:np.array([2*(
  x[0]-1)])) for m in ('BFGS','L-BFGS-B','CG','TNC')]; print('accepted')"`
  → `accepted` (scipy supports all). `grep -nE "newton-cg|trust-ncg|powell|BFGS|
  L-BFGS-B|Nelder" ferrolearn-numerical/src/optimize.rs` shows only the three
  implemented; no `BFGS`/`L-BFGS-B`/`CG`/`Nelder-Mead`/`SLSQP`/`TNC`/`COBYLA`/
  `dogleg`/`trust-krylov`/`trust-exact`/`trust-constr` exist.
- AC-7 (REQ-7): `python3 -c "import scipy.optimize as so; print([s for s in
  ('root','least_squares','curve_fit','brentq','brenth','bisect','fsolve',
  'linprog','nnls','root_scalar') if hasattr(so,s)])"` → all ten present.
  `grep -n "pub fn" ferrolearn-numerical/src/optimize.rs` shows only `new`/
  `with_*`/`minimize`/`brent_bounded` — none of the listed functions exist.
- AC-8 (REQ-8): `python3 -c "import inspect; from scipy.optimize import minimize;
  print('bounds' in inspect.signature(minimize).parameters, 'constraints' in
  inspect.signature(minimize).parameters)"` → `True True`. ferrolearn
  `NewtonCG::minimize`/`TrustRegionNCG::minimize`/`Powell::minimize` take only
  `(fun_grad, hessp, x0)` / `(f, x0)` — no `bounds`/`constraints` parameter;
  only `brent_bounded` carries an `[a,b]` interval.
- AC-9 (REQ-9): `python3 -c "import inspect; from scipy.optimize import minimize,
  minimize_scalar; print(list(inspect.signature(minimize).parameters));
  print(list(inspect.signature(minimize_scalar).parameters))"` →
  `['fun','x0','args','method','jac','hess','hessp','bounds','constraints','tol',
  'callback','options']` and `['fun','bracket','bounds','args','method','tol',
  'options']`. ferrolearn uses `NewtonCG::new().with_*().minimize(fun_grad, hessp,
  x0)` (builder, `(f,grad)`-tuple closure) and `brent_bounded(f,a,b,tol,max_iter)`
  (free fn). Structural API divergence, documented (R-DEV-2).
- AC-10 (REQ-10): `grep -n "Result<OptimizeResult, String>\|converged: false\|
  success: false\|FerroError" ferrolearn-numerical/src/optimize.rs` shows the
  `Result<_, String>` signature and the `converged`/`success` flags, and NO
  `FerroError`. scipy never raises on non-convergence — it sets
  `OptimizeResult.success=False`; ferrolearn matches the no-raise behaviour but
  with a `String` error type and a bool flag, not a `FerroError` or a
  scipy-equivalent `status`/`message`.
- AC-11 (REQ-11): `grep -rn "brent_bounded" --include=*.rs ferrolearn-*/src |
  grep -v 'src/optimize.rs'` → `ferrolearn-preprocess/src/power_transformer.rs:270`
  (`ferrolearn_numerical::optimize::brent_bounded(...)`), the PowerTransformer
  Box-Cox/Yeo-Johnson lambda search minimising the negative log-likelihood on
  `[-3,3]`. A REAL non-test production consumer.
- AC-12 (REQ-12): `grep -rn "NewtonCG\|TrustRegionNCG\|Powell\|optimize::minimize"
  --include=*.rs ferrolearn-*/src | grep -v 'src/optimize.rs'` → returns NOTHING.
  No estimator and no `ferrolearn-python` registration consumes `NewtonCG`/
  `TrustRegionNCG`/`Powell`.
- AC-13 (REQ-13): `grep -n "ferray" ferrolearn-numerical/src/optimize.rs` →
  empty; the vector algebra (`norm`/`dot`, CG/Steihaug recurrences, Powell line
  searches, Brent updates) is hand-rolled on `ndarray::Array1`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (NewtonCG converged-x\* parity) | SHIPPED | impl `pub fn minimize in optimize.rs` (`NewtonCG::minimize` → `cg_solve` inner loop + `backtracking_line_search`, Eisenstat-Walker `cg_tol = min(0.5, ‖g‖^½)·‖g‖`, `tol`=1e-8 gradient norm) mirrors `scipy.optimize.minimize(method='newton-cg')`. Live oracle (R-CHAR-3, from `scipy.optimize.minimize`, NEVER copied from ferrolearn): `minimize(0.5(x0²+2x1²),[5,3],method='newton-cg',jac=,hessp=).x = [0.0, 0.0]` (`fun=0`, `nit=7`). ferrolearn `NewtonCG::new().minimize(fg,hp,[5,3]).x = [0.0, 0.0]` exactly (`fun=0`, `n_iter=5`, `converged`) → abs ≤ 1e-8. Unique minimum → strict check; the algorithm differs from scipy (hand-rolled truncated Newton vs scipy's), so this is converged-x\* parity, not iteration parity. In-crate `newton_cg_quadratic` (diag(2,4,6), x\*=[0.5,0.5,0.5]), `newton_cg_rosenbrock` (→[1,1] ε=1e-4) green. Consumer caveat: REQ-12 — no non-test consumer for `NewtonCG`; this REQ ships on impl + the unique-minimum oracle, but the optimizer as a whole is gated by REQ-12. |
| REQ-2 (TrustRegionNCG converged-x\* parity) | SHIPPED | impl `pub fn minimize in optimize.rs` (`TrustRegionNCG::minimize` → `steihaug_cg` Steihaug-Toint subproblem + `boundary_step`, actual/predicted-reduction radius update) mirrors `scipy.optimize.minimize(method='trust-ncg')`. Live oracle: `minimize(0.5(x0²+2x1²),[5,3],method='trust-ncg',jac=,hessp=).x = [0.0, 0.0]` (`fun=0`, `nit=7`); Rosenbrock → `[1.0, 1.0]`. ferrolearn `TrustRegionNCG::new().minimize(...).x = [0.0, 0.0]` exactly (`n_iter=7`, `converged`) → abs ≤ 1e-8; in-crate `trust_region_rosenbrock` reaches `[1,1]` (ε=1e-4) and `trust_region_high_dimensional` (10-D `diag(1..10)`, x\*ᵢ=1/(i+1)) green. Unique minimum → strict check, converged-x\* parity (Steihaug-Toint differs from scipy's exact trust-region step). Same consumer caveat (REQ-12). |
| REQ-3 (Powell converged-x\* parity) | SHIPPED | impl `pub fn minimize in optimize.rs` (`Powell::minimize` → `line_minimise_powell` per-direction + NR extrapolation/direction-replacement, `ftol`=1e-8 relative improvement) mirrors `scipy.optimize.minimize(method='powell')`. Live oracle: `minimize((x0−1)²+(x1−2)²,[0,0],'powell').x = [1.0, 1.9999999999999998]`; `minimize(rosenbrock,[-1.2,1.0],'powell').x = [1.0000000000000913, 1.0000000000001923]`. ferrolearn `Powell::new().minimize(...).x = [1.0000000000000002, 2.0]` (quad, `fun=4.9e-32`) and `[1.0000000001018314, 1.0000000002080396]` (Rosenbrock, `fun=1.2e-20`, `converged`) → abs ≤ 1e-4. NOTE (R-HONEST-3): ferrolearn's `Powell` is the textbook NR direction-set method, scipy's is "modified Powell" — DIFFERENT algorithms that converge to the SAME unique minimum, so the contract is converged-x\* parity (~1e-4 for Powell's `ftol`), NOT iteration count; both reach Rosenbrock's `[1,1]`. In-crate `powell_finds_2d_quadratic_minimum`/`powell_finds_skewed_quadratic`/`powell_3d_anisotropic_quadratic`/`powell_one_dim_reduces_to_line_search` green. Same consumer caveat (REQ-12). |
| REQ-4 (brent_bounded converged-x\* parity) | SHIPPED | impl `pub fn brent_bounded in optimize.rs` (golden-section `(3−√5)/2` + parabolic interpolation, bracket-narrowing convergence) mirrors `scipy.optimize.minimize_scalar(method='bounded')`. Live oracle: `minimize_scalar((x−2)²,bounds=(0,5),'bounded').x = 2.0` (`fun=0`); `minimize_scalar(sin,bounds=(0,2π),'bounded').x = 4.7123876779707` (≈ `3π/2 = 4.71238898`). ferrolearn `brent_bounded((x−2)²,0,5,1e-8,500).x = 2.0` (`fun=0`, `success`) and `brent_bounded(sin,0,2π,1e-8,500).x = 4.712388981532386` (`fun=-1`, `success`) → abs ≤ 1e-6. Unique minimum on the interval → strict check. In-crate `brent_minimize_x_squared`/`brent_minimize_shifted_quadratic`/`brent_minimize_sin` pin closed-form minima. This optimizer HAS a real consumer (REQ-11 SHIPPED). |
| REQ-5 (OptimizeResult attribute contract) | NOT-STARTED | open prereq blocker (to be filed by critic). Live oracle: scipy `OptimizeResult` keys = `['fun','hess_inv','jac','message','nfev','nit','njev','status','success','x']`. ferrolearn `OptimizeResult` carries only `x/fun/grad/n_iter/converged` (`Minimize1DResult` carries `x/fun/nfev/success`). MISSING on `OptimizeResult`: `nfev`/`njev` (eval counts), `hess_inv`, `status` (integer code), `message` (string); and the names diverge — `n_iter`↔`nit`, `converged`↔`success`, `grad`↔`jac`. For derivative-free `Powell` the reported `grad` is filled with zeros (the algorithm never computes a gradient), unlike scipy's `jac` for gradient methods. The output-object contract (R-DEV-3) is not mirrored. |
| REQ-6 (missing `minimize` methods) | NOT-STARTED | open prereq blocker (`L-BFGS-B` relates to GPR/GPC `#1922`/`#1934`; remaining methods to be filed by critic). `scipy/optimize/__init__.py:60`–`:74` documents `neldermead`/`powell`/`cg`/`bfgs`/`newtoncg`/`lbfgsb`/`tnc`/`cobyla`/`cobyqa`/`slsqp`/`trustconstr`/`dogleg`/`trustncg`/`trustkrylov`/`trustexact`; `minimize` accepts all (live-verified). ferrolearn has only `newton-cg`/`trust-ncg`/`powell`. NO `BFGS`/`L-BFGS-B`/`CG`/`Nelder-Mead`/`SLSQP`/`TNC`/`COBYLA`/`dogleg`/`trust-krylov`/`trust-exact`/`trust-constr`. `L-BFGS-B` is the bound-constrained quasi-Newton GPR/GPC hyperparameter optimisation needs (`#1922`/`#1934`); `lbfgs`/`newton-cg` are LogisticRegression solvers; `bfgs` backs several sklearn estimators. These are real downstream prerequisites, none shipped. |
| REQ-7 (missing scipy.optimize functions) | NOT-STARTED | open prereq blocker (to be filed by critic). `scipy/optimize/__init__.py` documents `root` (`:227`), `least_squares` (`:129`), `nnls` (`:137`), `curve_fit` (`:147`), `root_scalar` (`:157`), `brentq` (`:158`)/`brenth` (`:159`)/`bisect` (`:161`), `fsolve` (`:394`), `linprog` (`:259`), and `minimize_scalar`'s `brent`/`golden` methods (`:44`/`:46`); `scipy.optimize` exposes all (live-verified). `grep -n "pub fn" optimize.rs` shows only `new`/`with_*`/`minimize`/`brent_bounded` — NONE of `root`/`least_squares`/`curve_fit`/`brentq`/`brenth`/`bisect`/`fsolve`/`linprog`/`nnls`/`root_scalar` exist. `nnls` is the sklearn-relevant one (`LinearRegression(positive=True)` / `NMF` non-negative least squares). |
| REQ-8 (bounds / constraints) | NOT-STARTED | open prereq blocker (to be filed by critic). `inspect.signature(minimize)` contains `bounds` and `constraints` (live-verified `True True`); scipy `bounds=` backs `L-BFGS-B`/`TNC`/`SLSQP`/`trust-constr` and `constraints=` backs `SLSQP`/`COBYLA`/`trust-constr`. ferrolearn `NewtonCG::minimize`/`TrustRegionNCG::minimize` take `(fun_grad, hessp, x0)` and `Powell::minimize` takes `(f, x0)` — NO `bounds`/`constraints` argument; the three multivariate optimizers are strictly UNCONSTRAINED. Only `brent_bounded` carries an `[a,b]` interval. Box bounds and general constraints on the multivariate path are not shipped (and block bound-constrained estimators that need `L-BFGS-B`). |
| REQ-9 (API contract) | NOT-STARTED | open prereq blocker (to be filed by critic). `inspect.signature(minimize)` = `[fun,x0,args,method,jac,hess,hessp,bounds,constraints,tol,callback,options]`; `minimize_scalar` = `[fun,bracket,bounds,args,method,tol,options]`. ferrolearn uses a BUILDER: `NewtonCG::new().with_max_iter().with_tol().with_max_cg_iter().minimize(fun_grad, hessp, x0)` where `fun_grad` returns a `(f, grad)` TUPLE (vs scipy's separate `fun`/`jac` callables) and `hessp` is a separate closure; `brent_bounded(f, a, b, tol, max_iter)` is a free function with positional args (vs `minimize_scalar(fun, bounds=, method=, tol=, options=)`). There is no single `minimize(method=)` dispatch entry point and no `OptimizeResult`-returning keyword ABI. The user-API ABI (R-DEV-2) diverges from scipy. |
| REQ-10 (error type / convergence signalling) | PARTIAL (#1992) | error-TYPE SHIPPED: `NewtonCG::minimize`/`TrustRegionNCG::minimize` now return `Result<OptimizeResult, FerroError>`; the empty-`x0` `Err` maps to `FerroError::InvalidParameter { name: "x0", reason: "initial guess x0 must have at least one element" }` (message preserved), mirroring scipy `minimize`'s `ValueError` on invalid `x0` (`_minimize.py`). `ferrolearn-core` is a workspace dep (#1961). Guard `optimize::tests::optimize_empty_x0_returns_ferroerror` (`matches!` on `Err(FerroError::InvalidParameter)` for both). STILL NOT-STARTED (remains #1992): the convergence-signalling remodel — `converged: bool` → scipy `OptimizeResult.success`/`status`/`message` for the `ferrolearn-python` boundary; and `Powell::minimize` + `brent_bounded` still return BARE results (no `Result`/`FerroError`), correct as no-raise paths but not yet carrying the richer scipy status surface. |
| REQ-11 (brent_bounded non-test consumer) | SHIPPED | impl `pub fn brent_bounded in optimize.rs` is consumed by a REAL non-test production caller: `ferrolearn-preprocess/src/power_transformer.rs` (`PowerTransformer::fit`) calls `ferrolearn_numerical::optimize::brent_bounded(closure, -3.0, 3.0, 1e-8, 500)` to find the Box-Cox/Yeo-Johnson `lambda` per feature by minimising the negative log-likelihood (`log_likelihood_yj`). `grep -rn "brent_bounded" --include=*.rs ferrolearn-*/src \| grep -v 'src/optimize.rs'` → `power_transformer.rs:270`. This is a production estimator path (sklearn `PowerTransformer`), not a test caller — `brent_bounded`'s consumer REQ is satisfied. |
| REQ-12 (NewtonCG/TrustRegionNCG/Powell non-test consumer) | NOT-STARTED | open prereq blocker (to be filed by critic). `grep -rn "NewtonCG\|TrustRegionNCG\|Powell\|optimize::minimize" --include=*.rs ferrolearn-*/src \| grep -v 'src/optimize.rs'` returns NOTHING; `lib.rs` exposes only `pub mod optimize` (no re-export). The three multivariate optimizers have ZERO in-workspace callers — no estimator and no `ferrolearn-python` registration. S5 grandfathering does NOT rescue this REQ: `NewtonCG`/`TrustRegionNCG`/`Powell` are internal substrate helpers, not boundary estimator types (`LinearRegression`/`StandardScaler`), with no external users and no Python binding. With zero consumers the honest call (R-HONEST-3) is NOT-STARTED — these three are dead code at baseline `bc82e38c8`. The fix is to route an estimator that needs unconstrained minimisation (e.g. a `LogisticRegression`/`GPR` solver) through them, or to ship the `L-BFGS-B`/`newton-cg` solver those estimators actually call (REQ-6). |
| REQ-13 (ferray substrate) | NOT-STARTED | open prereq blocker (to be filed by critic). `grep -n "ferray" optimize.rs` is empty: the module uses `ndarray::Array1` for all state and hand-rolls the vector algebra (`norm`/`dot`, the CG/Steihaug recurrences, the Powell `bracket_min_powell`/`golden_section_powell` line searches, the Brent updates). Per R-SUBSTRATE-1 the dense vector algebra is a `ferray::linalg`/`ferray-core` concern, and the scipy.optimize analog ultimately belongs on the ferray substrate. ferray does not yet expose a routed optimisation/BLAS-1 entry point for this use (R-SUBSTRATE-5: a ferray gap is real work, filed upstream to ferray; the ferrolearn unit is NOT-STARTED on this REQ until ferray ships it). Do NOT silently keep the hand-rolled `ndarray` path as the destination. |

## Architecture

`optimize.rs` is a flat module of optimizers over closures; there is no
unfitted/Fitted split (these are pure numerical routines, not estimators) and no
generic `F: Float` bound — everything is `f64`-only (a substrate detail, not a
scipy divergence; `scipy.optimize.minimize` operates on float64). The public
types are the config structs `NewtonCG`/`TrustRegionNCG`/`Powell` (builder
`new`/`with_*`), the result `OptimizeResult{x, fun, grad, n_iter, converged}`,
and the scalar `Minimize1DResult{x, fun, nfev, success}` plus the free
`brent_bounded`.

**NewtonCG (`minimize` → `cg_solve` → `backtracking_line_search`).** Outer
truncated-Newton loop: at each step compute `‖g‖`, exit if `< tol`, set the
Eisenstat-Walker forcing term `cg_tol = min(0.5, ‖g‖^½)·‖g‖`, approximately solve
`H d = −g` with CG (`cg_solve`, early-exit on negative curvature `pᵀHp ≤ 1e-30`,
falling back to `−g` if no progress), then an Armijo `backtracking_line_search`
(`c=1e-4`, `ρ=0.5`, 40 halvings); non-descent CG directions fall back to steepest
descent. Mirrors `minimize(method='newton-cg')` to converged-x\* parity (REQ-1) —
distinct from scipy's CG-Steihaug Newton-CG, so it is value-of-`x*`, not path.

**TrustRegionNCG (`minimize` → `steihaug_cg` → `boundary_step`).** Outer
trust-region loop: solve the Steihaug-Toint CG subproblem `min mₖ(d) s.t.
‖d‖ ≤ δ` (CG that, on negative curvature OR a step leaving the region, truncates
to the trust boundary via `boundary_step`'s quadratic root), compute the
actual/predicted-reduction ratio `ρ`, shrink `δ` by ¼ if `ρ<0.25`, grow it (×2,
capped at `max_radius`) if `ρ>0.75` at the boundary, accept the step iff `ρ>η`
(η=1e-4). Mirrors `minimize(method='trust-ncg')` to converged-x\* parity (REQ-2).

**Powell (`minimize` → `line_minimise_powell` → `bracket_min_powell` +
`golden_section_powell`).** Direction-set method seeded with the canonical basis:
each sweep line-minimises along every direction (golden-section after a
golden-ratio tripling bracket), tracks the direction of largest decrease, applies
the NR §10.5 extrapolation test on the net displacement, and — if it passes —
replaces the largest-decrease direction with the displacement (rotating the last
basis vector into the freed slot to keep the set independent). Converges when the
relative function improvement `2|f_start−f| ≤ ftol·(|f_start|+|f|)`. Derivative-
free → `OptimizeResult::grad` is zeros (REQ-5 detail). Mirrors
`minimize(method='powell')` to converged-x\* parity (REQ-3) — ferrolearn's NR
direction-set differs from scipy's "modified Powell", same unique minimum.

**brent_bounded.** Brent's 1-D bounded minimisation: maintains the three best
points `x`/`w`/`v`, tries a parabolic step through them (accepted only if inside
`[lo,hi]` and smaller than half the prior step), else a golden-section step
(`(3−√5)/2`), with a `tol·|x| + 1e-10` floor on step size, returning when the
bracket half-width `≤ 2tol1 − ½(hi−lo)`. Mirrors `minimize_scalar(
method='bounded')` to converged-x\* parity (REQ-4). This is the ONLY optimizer
with an interval, and the ONLY one with a real consumer (REQ-11).

The module's defining structural facts are REQ-11 vs REQ-12: `brent_bounded` has
a real production consumer (`PowerTransformer`), but `NewtonCG`/`TrustRegionNCG`/
`Powell` have NONE — at baseline `bc82e38c8` the three multivariate optimizers
are dead translation surface. This is why the cross-cutting REQs (5
OptimizeResult-attrs, 6 missing-methods, 7 missing-functions, 8 bounds/
constraints, 9 API, 10 error-type, 12 multivariate-consumer, 13 substrate) are
NOT-STARTED even though the four converged-x\* parity REQs (1/2/3/4) match the
live scipy oracle.

## Verification

Commands establishing the claims (run at baseline `bc82e38c8`):

- `cargo test -p ferrolearn-numerical --lib optimize` → all pass
  (`newton_cg_quadratic`, `newton_cg_rosenbrock`, `newton_cg_convergence_flag`,
  `trust_region_quadratic`, `trust_region_rosenbrock`,
  `trust_region_high_dimensional`, `brent_minimize_x_squared`,
  `brent_minimize_shifted_quadratic`, `brent_minimize_sin`,
  `powell_finds_2d_quadratic_minimum`, `powell_finds_skewed_quadratic`,
  `powell_3d_anisotropic_quadratic`, `powell_iteration_limit_marks_unconverged`,
  `powell_one_dim_reduces_to_line_search`,
  `powell_handles_fnmut_closure_with_eval_counter`). NOTE: these in-crate tests
  pin closed-form analytical minima (`[0.5,0.5,0.5]`, `[1,1]`, `1/(i+1)`,
  `3π/2`); they do NOT compare against a live `scipy.optimize.minimize` call —
  that comparison is the critic's REQ-1/2/3/4 pin below.
- converged-x\* parity oracle (REQ-1/2/3/4, R-CHAR-3 — expected from
  `scipy.optimize.minimize`/`minimize_scalar`, NEVER from ferrolearn):
  ```
  cd /tmp && python3 -c "
  import numpy as np
  from scipy.optimize import minimize, minimize_scalar
  g=lambda x:np.array([x[0],2*x[1]]); hp=lambda x,p:np.array([p[0],2*p[1]])
  f=lambda x:0.5*(x[0]**2+2*x[1]**2)
  print('newton-cg', minimize(f,[5.,3.],method='newton-cg',jac=g,hessp=hp).x.tolist())
  print('trust-ncg', minimize(f,[5.,3.],method='trust-ncg',jac=g,hessp=hp).x.tolist())
  print('powell quad', minimize(lambda x:(x[0]-1)**2+(x[1]-2)**2,[0.,0.],method='powell').x.tolist())
  ros=lambda x:sum(100*(x[1:]-x[:-1]**2)**2+(1-x[:-1])**2)
  print('powell rosen', minimize(ros,[-1.2,1.0],method='powell').x.tolist())
  print('bounded', minimize_scalar(lambda x:(x-2)**2,bounds=(0,5),method='bounded').x)
  print('bounded sin', minimize_scalar(np.sin,bounds=(0,2*np.pi),method='bounded').x)"
  ```
  → newton-cg `[0.0, 0.0]`, trust-ncg `[0.0, 0.0]`, powell quad
  `[1.0, 1.9999999999999998]`, powell rosen
  `[1.0000000000000913, 1.0000000000001923]`, bounded `2.0`, bounded sin
  `4.7123876779707`. Observed (this iteration, ferrolearn): newton-cg `[0,0]`
  (n_iter=5), trust-ncg `[0,0]` (n_iter=7), powell quad `[1.0000000000000002, 2.0]`,
  powell rosen `[1.0000000001018314, 1.0000000002080396]`, bounded `2.0`, bounded
  sin `4.712388981532386` — each matches the oracle `x*` to abs ≤ 1e-6 (≤ 1e-8 for
  newton-cg/trust-ncg/bounded). A critic pins these as `#[test]`s asserting the
  converged `x` (not iteration count) equals the live-scipy `x*`.
- OptimizeResult-attr surface (REQ-5): `python3 -c "from scipy.optimize import
  minimize; print(sorted(minimize(lambda x:(x[0]-1)**2,[0.],method='BFGS').keys()))"`
  → `['fun','hess_inv','jac','message','nfev','nit','njev','status','success','x']`;
  ferrolearn `OptimizeResult` has `x/fun/grad/n_iter/converged` only.
- missing-methods / functions surface (REQ-6/REQ-7): `python3 -c "import
  scipy.optimize as so; from scipy.optimize import minimize; import numpy as np;
  print([s for s in ('root','least_squares','curve_fit','brentq','brenth',
  'bisect','fsolve','linprog','nnls','root_scalar') if hasattr(so,s)])"` → all
  ten present; `grep -nE 'BFGS|L-BFGS-B|Nelder|SLSQP|root|least_squares|nnls'
  ferrolearn-numerical/src/optimize.rs` → empty.
- bounds/constraints + API surface (REQ-8/REQ-9): `python3 -c "import inspect;
  from scipy.optimize import minimize, minimize_scalar; print(list(
  inspect.signature(minimize).parameters)); print(list(inspect.signature(
  minimize_scalar).parameters))"` → minimize has `bounds`/`constraints`;
  ferrolearn `minimize` takes only `(fun_grad, hessp, x0)` / `(f, x0)`.
- error-type / substrate checks (REQ-10/REQ-13): `grep -n "Result<OptimizeResult,
  String>\|Minimize1DResult\|FerroError\|ferray" ferrolearn-numerical/src/
  optimize.rs` → `Result<_, String>` + bare `Minimize1DResult`, no `FerroError`,
  no `ferray`.
- consumer checks (REQ-11/REQ-12): `grep -rn "brent_bounded" --include=*.rs
  ferrolearn-*/src | grep -v 'src/optimize.rs'` → `power_transformer.rs:270`
  (REQ-11 SHIPPED); `grep -rn "NewtonCG\|TrustRegionNCG\|Powell\|optimize::minimize"
  --include=*.rs ferrolearn-*/src | grep -v 'src/optimize.rs'` → empty (REQ-12
  blocker; a missing-consumer fact is structural, not a numerical assertion).

Per R-DEFER-2 the table is binary SHIPPED/NOT-STARTED. SHIPPED (impl + live scipy
oracle to the optimizer's tolerance): REQ-1 (NewtonCG converged-x\* `[0,0]`),
REQ-2 (TrustRegionNCG converged-x\* `[0,0]` + Rosenbrock `[1,1]`), REQ-3 (Powell
converged-x\* `[1,2]` + Rosenbrock `[1,1]`), REQ-4 (brent_bounded `2.0` + `3π/2`),
REQ-11 (brent_bounded consumer — `PowerTransformer`). NOT-STARTED (open `-l
blocker` issues to be filed by the critic): REQ-5 (OptimizeResult attrs
`nfev`/`njev`/`hess_inv`/`status`/`message`, name drift), REQ-6 (missing
`BFGS`/`L-BFGS-B`/`CG`/`Nelder-Mead`/`SLSQP`/`TNC`/… methods; `L-BFGS-B` →
`#1922`/`#1934`), REQ-7 (missing `root`/`least_squares`/`curve_fit`/`brentq`/
`fsolve`/`linprog`/`nnls`/`root_scalar`/`brent`/`golden`), REQ-8 (no
bounds/constraints on the multivariate path), REQ-9 (builder API vs scipy
`minimize(method=)` ABI), REQ-10 (`String`/bool signalling vs `FerroError` /
scipy `success`/`status`/`message`), REQ-12 (no non-test consumer for
`NewtonCG`/`TrustRegionNCG`/`Powell` — dead code), REQ-13 (ferray `linalg`/core
substrate for the vector algebra).
