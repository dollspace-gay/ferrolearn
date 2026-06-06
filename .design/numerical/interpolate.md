# scipy.interpolate.CubicSpline

<!--
tier: 3-component
status: draft
baseline-commit: 58c6b818c
upstream-paths:
  - scipy/interpolate/_cubic.py        # CubicSpline class (PPoly subclass)
-->

## Summary

`ferrolearn-numerical/src/interpolate.rs` is the scipy-analog substrate that
mirrors **scipy.interpolate.CubicSpline** — a piecewise-cubic `C²` interpolant.
It exposes one type, [`CubicSpline`] (`new`, `eval`, `eval_array`, `derivative`,
`second_derivative`, `integrate`), built from per-interval coefficients
`(a_i, b_i, c_i, d_i)` derived by solving a tridiagonal system for the knot
second-derivatives. Two boundary conditions are offered via
[`BoundaryCondition`]: `Natural` (`S''=0` at the ends, scipy `bc_type='natural'`)
and `NotAKnot` (third derivative continuous at the second / second-to-last knots,
scipy `bc_type='not-a-knot'`). Cubic-spline construction is deterministic, so the
installed **scipy 1.17.1** is a valid live oracle for the true interpolant values
(the scipy-1.17.1 / sklearn-1.5.2 version split is irrelevant — a `C²` cubic
spline through fixed knots is the same in every scipy release).

The numbers ferrolearn computes for the **standard (n ≥ 4-point) case match scipy
to ~1e-14** across eval, first/second derivative, integration, extrapolation, and
both boundary conditions — verified element-wise against the live oracle below.
The divergences are concentrated in (a) a genuine **3-point not-a-knot numerical
bug**, (b) **API / surface** gaps, and (c) **substrate / consumer** gaps.

Divergence classes:
1. **value-parity slices** (the bulk — SHIPPED) — Natural and NotAKnot `eval`,
   `derivative`, `second_derivative`, `integrate`, and `eval`-based extrapolation
   all match the scipy oracle to ~1e-14 on n ≥ 4 datasets (squares, sin).
2. **few-point edge bug (headline divergence, NOT-STARTED)** — for a 3-point
   not-a-knot spline `solve_not_a_knot_general(n=2, …)` produces the WRONG
   coefficients: ferrolearn yields `[0.125, 2.375]` where scipy reproduces the
   parabola exactly `[0.25, 2.25]`. The degenerate `n=2` not-a-knot elimination is
   incorrect.
3. **default-bc (R-DEV-2)** — scipy's `bc_type` defaults to `'not-a-knot'`;
   ferrolearn `new` REQUIRES an explicit `BoundaryCondition` argument (no default).
4. **missing boundary conditions** — scipy supports `'clamped'`, `'periodic'`,
   and custom `((order, value), (order, value))` derivative tuples; ferrolearn has
   ONLY Natural + NotAKnot.
5. **missing PPoly methods / attributes** — scipy `CubicSpline` is a `PPoly`
   exposing `__call__(x, nu)`, `derivative()`/`antiderivative()` (returning a NEW
   spline), `roots()`, `solve()`, `.c`/`.x`; ferrolearn has only scalar
   `eval`/`derivative`/`second_derivative`/`integrate` + `eval_array`.
6. **error-type** — `Result<_, String>` rather than `Result<_, FerroError>`;
   scipy raises `ValueError`.
7. **no-consumer** — `pub mod interpolate` with no non-test caller anywhere in the
   workspace.
8. **ferray-substrate** — `ndarray::Array1` + hand-rolled `Vec<f64>` Thomas
   solvers rather than ferray's interpolate / linalg analog (R-SUBSTRATE-1).

## Upstream reference (scipy.interpolate.CubicSpline, live oracle scipy 1.17.1)

The class is defined in `scipy/interpolate/_cubic.py` (`class CubicSpline(CubicHermiteSpline)`,
itself a `PPoly` subclass). Its `__init__` signature is
`CubicSpline(x, y, axis=0, bc_type='not-a-knot', extrapolate=None)`. `bc_type`
accepts the strings `'not-a-knot'`, `'clamped'`, `'natural'`, `'periodic'`, or a
2-tuple of `(order, value)` derivative conditions; `extrapolate` defaults to
`True` (and to `'periodic'` when `bc_type='periodic'`), so out-of-range queries
use the first/last polynomial piece. not-a-knot requires ≥ 3 points; the 2-point
case is handled specially. Because the kernel is partly Cython/Fortran-adjacent,
cite the **scipy.interpolate symbol names** and the **live-oracle values**, never
internal line numbers of the solve.

Live oracle (`cd /tmp && python3 -c "..."`, scipy 1.17.1).
Dataset A: `x=[0,1,2,3,4]`, `y=[0,1,4,9,16]` (= `x²`):

- `CubicSpline(x,y,bc_type='natural')([0.5,1.5,2.5,3.5])` →
  `[0.33928571428571425, 2.232142857142857, 6.232142857142858, 12.339285714285714]`;
  `d1@1.5 = 3.0357142857142856`, `d2@2.5 = 2.1428571428571423`,
  `integrate(0,4) = 21.428571428571427`.
- `CubicSpline(x,y,bc_type='not-a-knot')([0.5,1.5,2.5,3.5])` →
  `[0.25, 2.25, 6.25, 12.25]` (EXACT squares — not-a-knot reproduces cubics, and
  here the data is a parabola); `d1@1.5 = 3.0`, `d2@2.5 = 2.0`,
  `integrate(0,4) = 21.333333333333336`.
- extrapolation (`extrapolate=True` default):
  `natural([-0.5,4.5]) = [-0.33928571428571436, 19.66071428571429]`,
  `not-a-knot([-0.5,4.5]) = [0.25, 20.25]`.

Dataset B: `xs=linspace(0,6,7)`, `ys=sin(xs)` (a non-cubic — the two BCs genuinely
differ):

- `CubicSpline(xs,ys,bc_type='natural')([0.3,2.7,5.1])` →
  `[0.2939944300893482, 0.425335344554224, -0.9229635816106156]`.
- `CubicSpline(xs,ys)([0.3,2.7,5.1])` (default not-a-knot) →
  `[0.31890041025200067, 0.4272713683778276, -0.9309715276490006]`.

Few-point oracle (`x3=[0,1,2]`, `y3=[0,1,4]` = `x²`):

- `CubicSpline(x3,y3,bc_type='natural')([0.5,1.5]) = [0.3125, 2.3125]`.
- `CubicSpline(x3,y3,bc_type='not-a-knot')([0.5,1.5]) = [0.25, 2.25]` (exact —
  3-point not-a-knot collapses to the single parabola through the points).
- 2-point: both `natural` and `not-a-knot` interpolate linearly
  (`CubicSpline([0,1],[0,2],bc_type='not-a-knot')([0.5]) = [1.0]`).

Other surface: default `bc_type='not-a-knot'` (via
`inspect.signature(CubicSpline.__init__)`); `clamped` exists
(`CubicSpline(x,y,bc_type='clamped')([0.5,2.5]) = [0.23214285714285715, 5.910714285714286]`);
unsorted/duplicate `x` raises `ValueError`.

## Requirements

- REQ-NAT-EVAL: natural-BC `eval` parity. `CubicSpline::new(x,y,Natural).eval(xq)`
  matches `scipy.interpolate.CubicSpline(x,y,bc_type='natural')(xq)` element-wise
  on n ≥ 4 datasets (squares and sin).
- REQ-NAT-DERIV: natural-BC `derivative` / `second_derivative` parity — match
  `cs(xq, 1)` and `cs(xq, 2)`.
- REQ-NAT-INTEG: natural-BC `integrate(a,b)` parity — match `cs.integrate(a,b)`.
- REQ-NAK-EVAL: not-a-knot-BC `eval` parity on n ≥ 4 datasets (exact cubic
  reproduction on `x²`; sin agreement to ~1e-14).
- REQ-NAK-DERIV: not-a-knot-BC `derivative` / `second_derivative` parity.
- REQ-NAK-INTEG: not-a-knot-BC `integrate(a,b)` parity.
- REQ-EXTRAP: extrapolation parity — for `xq` outside `[x_0, x_n]`, `eval` uses
  the first/last cubic piece, matching scipy's `extrapolate=True` default.
- REQ-FEWPOINT: few-point edge parity — 2-point (line) AND 3-point splines match
  scipy for both BCs, including 3-point not-a-knot reproducing the parabola.
- REQ-DEFAULT-BC: constructor default parity (R-DEV-2) — the not-a-knot default
  scipy applies when `bc_type` is omitted is reflected in ferrolearn's API.
- REQ-MISSING-BC: the remaining scipy boundary conditions exist —
  `'clamped'` (first derivative fixed at the ends), `'periodic'`, and custom
  `((order, value), (order, value))` tuples.
- REQ-PPOLY-API: the `PPoly` method/attribute surface exists — `__call__(x, nu)`
  (arbitrary derivative order in one call), `derivative()`/`antiderivative()`
  returning a NEW spline, `roots()`, `solve(y)`, and the `.c` / `.x` accessors.
- REQ-ERR-TYPE: `new` returns `Result<_, FerroError>` (CLAUDE.md / R-CODE-2) with
  scipy-matching `ValueError` semantics for invalid input.
- REQ-CONSUMER: a non-test workspace caller (an estimator or the
  `ferrolearn-python` binding) consumes `interpolate::CubicSpline` so it is part
  of the live translation surface.
- REQ-FERRAY: the spline operates on ferray's array / linalg analog rather than
  `ndarray::Array1` + hand-rolled `Vec<f64>` Thomas solvers (R-SUBSTRATE-1).

## Acceptance criteria

All expected values come from the live scipy oracle (R-CHAR-3), never from
ferrolearn. Run from `/tmp`. `x=[0,1,2,3,4]`, `y=[0,1,4,9,16]` unless noted.

- AC-NAT-EVAL (REQ-NAT-EVAL):
  `python3 -c "import numpy as np; from scipy.interpolate import CubicSpline; print(CubicSpline(np.array([0.,1,2,3,4]),np.array([0.,1,4,9,16]),bc_type='natural')([0.5,1.5,2.5,3.5]).tolist())"`
  → `[0.33928571428571425, 2.232142857142857, 6.232142857142858, 12.339285714285714]`.
  ferrolearn `Natural` eval gives
  `[0.33928571428571425, 2.232142857142857, 6.232142857142858, 12.339285714285715]`
  — matches to ≤ 1e-14 (worst element abs diff ~1e-15). The sin dataset (B) gives
  `[0.2939944300893482, 0.4253353445542241, -0.9229635816106155]` vs scipy
  `[0.2939944300893482, 0.425335344554224, -0.9229635816106156]` (≤ 1e-15).
- AC-NAT-DERIV (REQ-NAT-DERIV): `cs=CubicSpline(...,bc_type='natural')`;
  `cs(1.5,1)=3.0357142857142856`, `cs(2.5,2)=2.1428571428571423`. ferrolearn
  `derivative(1.5)=3.0357142857142856`, `second_derivative(2.5)=2.142857142857143`
  (≤ 1e-15).
- AC-NAT-INTEG (REQ-NAT-INTEG): `cs.integrate(0,4)=21.428571428571427`; ferrolearn
  `integrate(0.,4.)=21.428571428571427` (exact match).
- AC-NAK-EVAL (REQ-NAK-EVAL):
  `CubicSpline(x,y,bc_type='not-a-knot')([0.5,1.5,2.5,3.5]) = [0.25, 2.25, 6.25, 12.25]`
  (exact squares); ferrolearn `NotAKnot` eval = `[0.25, 2.25, 6.25, 12.25]`. The
  sin dataset: scipy `[0.31890041025200067, 0.4272713683778276, -0.9309715276490006]`
  vs ferrolearn `[0.31890041025200055, 0.4272713683778276, -0.9309715276490006]`
  (≤ 1e-15).
- AC-NAK-DERIV (REQ-NAK-DERIV): `cs(1.5,1)=3.0`, `cs(2.5,2)=2.0`; ferrolearn
  `derivative(1.5)=3.0`, `second_derivative(2.5)=2.0` (exact).
- AC-NAK-INTEG (REQ-NAK-INTEG): `cs.integrate(0,4)=21.333333333333336`; ferrolearn
  `integrate(0.,4.)=21.333333333333336` (exact match).
- AC-EXTRAP (REQ-EXTRAP):
  `CubicSpline(x,y,bc_type='natural')([-0.5,4.5]) = [-0.33928571428571436, 19.66071428571429]`
  and `CubicSpline(x,y)([-0.5,4.5]) = [0.25, 20.25]`. ferrolearn `Natural`:
  `[-0.33928571428571425, 19.660714285714285]`; `NotAKnot`: `[0.25, 20.25]`
  (≤ 1e-13). `find_interval` clamps to the first/last interval, matching scipy's
  `extrapolate=True` default.
- AC-FEWPOINT (REQ-FEWPOINT — headline divergence): `x3=[0,1,2]`, `y3=[0,1,4]`.
  `CubicSpline(x3,y3,bc_type='natural')([0.5,1.5]) = [0.3125, 2.3125]` — ferrolearn
  `Natural` gives `[0.3125, 2.3125]` (MATCH). BUT
  `CubicSpline(x3,y3,bc_type='not-a-knot')([0.5,1.5]) = [0.25, 2.25]` (exact
  parabola) — ferrolearn `NotAKnot` gives `[0.12499999999999994, 2.375]`
  (DIVERGES — abs error ~0.125). A critic pins a FAILING `#[test]` asserting the
  3-point not-a-knot spline equals `[0.25, 2.25]` to ≤ 1e-12. 2-point matches:
  `CubicSpline([0,1],[0,2],bc_type='not-a-knot')([0.5]) = [1.0]`; ferrolearn
  `NotAKnot` eval at 0.5 = `1.0`.
- AC-DEFAULT-BC (REQ-DEFAULT-BC):
  `python3 -c "import inspect; from scipy.interpolate import CubicSpline; print(inspect.signature(CubicSpline.__init__).parameters['bc_type'].default)"`
  → `not-a-knot`. ferrolearn `CubicSpline::new(x, y, bc)` has NO default — `bc`
  is a required third argument; omitting a BC is a compile error, so a caller
  porting `CubicSpline(x, y)` must know to pass `BoundaryCondition::NotAKnot`.
- AC-MISSING-BC (REQ-MISSING-BC):
  `CubicSpline(x,y,bc_type='clamped')([0.5,2.5]) = [0.23214285714285715, 5.910714285714286]`,
  and `'periodic'` / custom `((1, d0), (1, dn))` tuples are accepted by scipy.
  `BoundaryCondition` has only `Natural`/`NotAKnot` variants — no `Clamped`,
  `Periodic`, or custom-derivative form. FAILS until added.
- AC-PPOLY-API (REQ-PPOLY-API): `cs=CubicSpline(x,y)`; `cs(1.5, nu=2)` (2nd
  derivative in one call), `cs.derivative()` / `cs.antiderivative()` (new
  `PPoly`), `cs.roots()`, `cs.solve(0.5)`, `cs.c` (4×n coefficient matrix),
  `cs.x` (breakpoints) all exist. `grep -n "pub fn" interpolate.rs` lists only
  `new`, `eval`, `eval_array`, `derivative`, `second_derivative`, `integrate` —
  no general `nu`, no antiderivative-as-spline, no `roots`/`solve`, no `.c`/`.x`
  accessors. FAILS.
- AC-ERR-TYPE (REQ-ERR-TYPE): `new` returns `Result<Self, String>` (e.g.
  `Err(format!("x must be strictly increasing, …"))`); scipy raises `ValueError`
  on unsorted/duplicate `x`
  (`python3 -c "import numpy as np; from scipy.interpolate import CubicSpline; CubicSpline(np.array([0.,2,1]), np.array([0.,1,2]))"`
  → `ValueError`). The `String` error type fails the CLAUDE.md/R-CODE-2
  `FerroError` contract.
- AC-CONSUMER (REQ-CONSUMER):
  `grep -rn "interpolate::\|CubicSpline\|BoundaryCondition" --include=*.rs ferrolearn-*/src | grep -v 'interpolate.rs'`
  returns nothing — no estimator, no `ferrolearn-python` registration consumes
  `CubicSpline`. `lib.rs` exposes only `pub mod interpolate` (no re-export).
- AC-FERRAY (REQ-FERRAY): the owned spline computation routes through ferray's
  array / linalg analog, not `ndarray::Array1` (used in `eval_array`) plus the
  hand-rolled `Vec<f64>` `thomas_solve` / `solve_not_a_knot_general` tridiagonal
  solvers.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-NAT-EVAL (natural eval) | SHIPPED | impl `pub fn eval in interpolate.rs` (Horner `a[i] + t*(b[i] + t*(c[i] + t*d[i]))` over interval from `find_interval`; coefficients from `solve_natural` — Thomas-solved tridiagonal with `c_0=c_n=0`) mirrors `scipy.interpolate.CubicSpline(…,bc_type='natural')`. Live oracle (R-CHAR-3): `[0.5,1.5,2.5,3.5]` → `[0.33928571428571425, 2.232142857142857, 6.232142857142858, 12.339285714285714]`; ferrolearn matches to ≤1e-14 (worst elem ~1e-15). sin dataset matches to ≤1e-15. Verification: in-crate `interpolates_data_points`/`linear_data_exact`/`quadratic_natural` green + the live-oracle example. NOTE: ships on numerical parity of the eval values; the module-as-surface is gated by REQ-CONSUMER/REQ-FERRAY. |
| REQ-NAT-DERIV (natural deriv) | SHIPPED | impl `pub fn derivative in interpolate.rs` (`b[i] + t*(2c[i] + t*3d[i])`) and `pub fn second_derivative in interpolate.rs` (`2c[i] + 6d[i]*t`) mirror `scipy`'s `cs(x,1)`/`cs(x,2)`. Live oracle: natural `cs(1.5,1)=3.0357142857142856`, `cs(2.5,2)=2.1428571428571423`; ferrolearn `derivative(1.5)=3.0357142857142856`, `second_derivative(2.5)=2.142857142857143` (≤1e-15). Verification: `derivative_of_linear` green. |
| REQ-NAT-INTEG (natural integrate) | SHIPPED | impl `pub fn integrate in interpolate.rs` (analytic per-piece antiderivative `F_i(t)=a t + b t²/2 + c t³/3 + d t⁴/4` via `antiderivative_at`, summed over covered intervals, sign-flip for `a>b`) mirrors `scipy`'s `cs.integrate`. Live oracle: natural `integrate(0,4)=21.428571428571427`; ferrolearn exact match. Verification: `integration_linear`/`integration_quadratic` green. |
| REQ-NAK-EVAL (not-a-knot eval) | SHIPPED | impl `pub fn eval in interpolate.rs` with `solve_not_a_knot`→`solve_not_a_knot_general` (modified Gaussian elimination enforcing `d_0=d_1`, `d_{n-2}=d_{n-1}`) mirrors `scipy`'s default `bc_type='not-a-knot'`. Live oracle: `[0.5,1.5,2.5,3.5]` → `[0.25, 2.25, 6.25, 12.25]` (exact squares — cubic reproduction); ferrolearn matches exactly. sin: scipy `[0.31890041025200067, …, -0.9309715276490006]` vs ferrolearn `[0.31890041025200055, …, -0.9309715276490006]` (≤1e-15). Verification: `not_a_knot_cubic_exact` (x³ reproduced to 1e-10) green. Holds for n ≥ 4; the 3-point case is a separate divergence (REQ-FEWPOINT). |
| REQ-NAK-DERIV (not-a-knot deriv) | SHIPPED | same `derivative`/`second_derivative` impls over not-a-knot coefficients. Live oracle: not-a-knot `cs(1.5,1)=3.0`, `cs(2.5,2)=2.0`; ferrolearn `derivative(1.5)=3.0`, `second_derivative(2.5)=2.0` (exact, n ≥ 4). |
| REQ-NAK-INTEG (not-a-knot integrate) | SHIPPED | same `integrate` impl over not-a-knot coefficients. Live oracle: `integrate(0,4)=21.333333333333336`; ferrolearn exact match. |
| REQ-EXTRAP (extrapolation) | SHIPPED | impl `fn find_interval in interpolate.rs` clamps `x ≤ knots[0]` to interval 0 and `x ≥ knots[n]` to interval `n-1`, so `eval` evaluates the first/last cubic outside the range — matching scipy's `extrapolate=True` default (first/last polynomial piece). Live oracle: natural `([-0.5,4.5])=[-0.33928571428571436, 19.66071428571429]`, not-a-knot `=[0.25, 20.25]`; ferrolearn `[-0.33928571428571425, 19.660714285714285]` / `[0.25, 20.25]` (≤1e-13). NOTE: ferrolearn has no `extrapolate=False`/NaN-out-of-range mode — it always extrapolates (a sub-facet folded into REQ-PPOLY-API, since scipy gates this via the `extrapolate` ctor arg). |
| REQ-FEWPOINT (few-point edge) | SHIPPED | FIXED #1957: `solve_not_a_knot` now special-cases `n==2` (one interior knot) — the two not-a-knot end-conditions are degenerate/identical, so the spline is the unique PARABOLA through the 3 points (`c_knots = [dd,dd,dd]` with `dd` the second divided difference → `d_i=0`), matching scipy `_cubic.py:824`. Oracle `CubicSpline([0,1,2],[0,1,4],bc_type='not-a-knot')([0.5,1.5]) = [0.25, 2.25]`; ferrolearn now matches (was `[0.125, 2.375]`). 2-point and 3-point-natural already matched. Pinned by `divergence_three_point_not_a_knot_parabola` (green); n≥4 not-a-knot unaffected (`solve_not_a_knot_general` path unchanged). |
| REQ-DEFAULT-BC (default bc_type) | SHIPPED (#1958) | FIXED — added `impl Default for BoundaryCondition` returning `NotAKnot`, mirroring scipy `CubicSpline(bc_type='not-a-knot')` (`scipy/interpolate/_cubic.py:790`; `inspect.signature(CubicSpline.__init__).parameters['bc_type'].default == 'not-a-knot'`). So `CubicSpline::new(&x, &y, BoundaryCondition::default())` reproduces scipy's `CubicSpline(x, y)`. Guard `tests/divergence_interpolate.rs::cubic_spline_default_bc_is_not_a_knot_matches_scipy`: `BoundaryCondition::default() == NotAKnot` AND the default-bc spline eval at `[0.5,1.5,2.5,3.5]` on `x=[0..4],y=[0,1,8,27,64]` matches the live scipy default-bc oracle `[0.125, 3.375, 15.625, 42.875]` (<1e-9), distinct from `'natural'` (`[0.0982…, 3.4554…, …]`). NOTE: `new` keeps `bc` an explicit arg (Rust has no default args); the `Default` impl is the faithful encoding of scipy's default value. |
| REQ-MISSING-BC (clamped/periodic/custom) | NOT-STARTED | open prereq blocker (to be filed by critic). `BoundaryCondition` has only `Natural` and `NotAKnot`. scipy's `bc_type` also accepts `'clamped'` (first derivative 0 at the ends — `CubicSpline(x,y,bc_type='clamped')([0.5,2.5]) = [0.23214285714285715, 5.910714285714286]`), `'periodic'` (requires `y[0]==y[-1]`; `extrapolate` defaults to `'periodic'`), and a custom `((order, value), (order, value))` 2-tuple of derivative conditions at each end. None exist in ferrolearn. R-DEV-2 (constructor parameter domain). |
| REQ-PPOLY-API (PPoly methods/attrs) | NOT-STARTED | open prereq blocker (to be filed by critic). scipy `CubicSpline` is a `PPoly` subclass exposing: `__call__(x, nu)` (nu-th derivative in one call — ferrolearn hardcodes nu∈{0,1,2} as three named methods, no general nu); `derivative()`/`antiderivative()` returning a NEW spline object (ferrolearn's `derivative` returns a scalar value, and there is no antiderivative-as-spline); `roots()` and `solve(y)` (root-finding — absent); and the `.c` (4×n coefficient matrix) / `.x` (breakpoints) attribute accessors (ferrolearn's `knots`/`a`/`b`/`c`/`d` fields are PRIVATE — no public accessor). Also the `extrapolate=False` mode (NaN outside range) has no analog. R-DEV-2/R-DEV-3 (method + output-object contract). |
| REQ-ERR-TYPE (FerroError + ValueError) | SHIPPED (#1961) | FIXED — `pub fn new` now returns `Result<Self, FerroError>`; all three validation failures (x/y length mismatch, `<2` points, non-strictly-increasing `x`) return `FerroError::InvalidParameter { name: "x", reason }` (messages preserved), mirroring scipy `CubicSpline.__init__`'s `ValueError` (`scipy/interpolate/_cubic.py:48-65`). `ferrolearn-core` added as a workspace dependency (`ferrolearn-core.workspace = true`). Live scipy oracle (R-CHAR-3): `CubicSpline([0,1],[0,1,2])`, `CubicSpline([0],[0])`, `CubicSpline([0,0,1],[0,1,2])` each raise `ValueError`. Guard `tests/divergence_interpolate.rs::cubic_spline_new_invalid_returns_ferroerror` (`matches!` on `Err(FerroError::InvalidParameter)`). NOTE: the other `Result<_, String>` numerical APIs (distributions/integrate/optimize) remain separate blockers #1967/#1975/#1992. |
| REQ-CONSUMER (non-test production caller) | NOT-STARTED | open prereq blocker (to be filed by critic). `lib.rs` exposes only `pub mod interpolate` (no re-export). `grep -rn "interpolate::\|CubicSpline\|BoundaryCondition" --include=*.rs ferrolearn-*/src \| grep -v 'interpolate.rs'` returns NOTHING — no estimator, no `ferrolearn-python` registration consumes `CubicSpline`. S5 grandfathering does NOT rescue this: `interpolate::CubicSpline` is an internal scipy.interpolate substrate helper, not a boundary estimator type (`LinearRegression`/`StandardScaler`) — there is no external user and no Python binding for it. Dead translation surface (R-HONEST-3). Fix: route an estimator that needs spline interpolation (e.g. a calibration/monotone path, or expose it through `ferrolearn-python`) through `interpolate::CubicSpline`, or fold the module into ferray per REQ-FERRAY. |
| REQ-FERRAY (ferray substrate) | NOT-STARTED | open prereq blocker (to be filed by critic). `interpolate.rs` uses `ndarray::Array1` (in `eval_array`) and hand-rolls the tridiagonal linear solve (`thomas_solve`, `solve_not_a_knot_general`) over `Vec<f64>`. Per R-SUBSTRATE-1 the scipy.interpolate analog is a **ferray** concern (a ferray interpolate module + `ferray::linalg` for the tridiagonal solve); `ferrolearn-numerical` should consume it rather than hand-roll. ferray does not yet expose this surface (R-SUBSTRATE-5: a ferray gap is real work, filed upstream to ferray; the ferrolearn unit is NOT-STARTED on this REQ until ferray ships the interpolate / banded-solve layer). |

## Architecture

`interpolate.rs` is a flat module exposing one struct, `CubicSpline`, plus the
`BoundaryCondition` enum and four private tridiagonal-solver free functions
(`solve_natural`, `solve_not_a_knot`, `solve_not_a_knot_general`, `thomas_solve`).
There is no unfitted/Fitted split — `CubicSpline::new` does the whole construction
(validate → solve for knot second-derivatives → derive per-interval
coefficients), which is appropriate because `scipy.interpolate.CubicSpline` is
likewise a single constructor that builds the `PPoly` representation eagerly. The
struct stores the standard per-interval cubic coefficients
`S_i(x) = a_i + b_i·t + c_i·t² + d_i·t³` with `t = x − x_i` and `a_i = y_i`; scipy
stores the same information transposed in its `.c` 4×n matrix (highest power
first), which is why the `.c` accessor (REQ-PPOLY-API) would be a representation
re-pack, not a recompute.

The construction solves for the knot second-derivatives `c_i` (scipy's `S''`):
for **natural** BC, `solve_natural` builds the `(n−1)×(n−1)` interior tridiagonal
system with `c_0 = c_n = 0` and runs the textbook `thomas_solve` (Thomas
algorithm, `O(n)`); for **not-a-knot**, `solve_not_a_knot_general` builds an
`(n+1)×(n+1)` *almost*-tridiagonal system whose first and last rows encode
`d_0 = d_1` and `d_{n-2} = d_{n-1}` (the third-derivative-continuity conditions),
each touching three consecutive unknowns that extend one position beyond the
tridiagonal band, then runs a modified Gaussian elimination handling the two
extra entries. For n ≥ 3 intervals this produces coefficients that match scipy to
~1e-14 (REQ-NAK-EVAL SHIPPED, including exact cubic reproduction on `x³`).

The defining numeric divergence (REQ-FEWPOINT, headline) lives in
`solve_not_a_knot_general` at **n = 2** (3 data points, 1 interior knot). With a
single interior knot the left and right not-a-knot conditions become degenerate —
they describe the SAME single cubic, and the correct answer is the unique parabola
through the three points (scipy: `[0.25, 2.25]` on `x²` data). The modified
elimination, written for the general band-plus-corner structure, does not collapse
correctly in this degenerate `n=2` case and returns wrong `c` coefficients, so
ferrolearn yields `[0.125, 2.375]` — abs error ~0.125. The natural path at n = 2
is correct (`[0.3125, 2.3125]`) because `solve_natural`'s interior system is a
single equation with no corner rows. The 2-point case (n = 1) is short-circuited
in `new` to a straight line (`c = d = 0`) for both BCs, matching scipy.

`find_interval` is a clamped binary search: it returns interval `0` for any
`x ≤ knots[0]` and `n−1` for any `x ≥ knots[n]`, so `eval`/`derivative`/`integrate`
ALWAYS extrapolate using the boundary cubic — matching scipy's `extrapolate=True`
default (REQ-EXTRAP SHIPPED), but with no `extrapolate=False` (NaN-outside) mode
(folded into REQ-PPOLY-API). `integrate` is exact per-piece analytic integration
(`antiderivative_at` evaluates `F_i(t) = a·t + b·t²/2 + c·t³/3 + d·t⁴/4` at the
local interval limits), summing the covered intervals and negating for `a > b`.

The cross-cutting structural facts are REQ-CONSUMER and REQ-FERRAY. The module has
**no caller**: `lib.rs` exposes `pub mod interpolate` with no re-export and a
workspace-wide grep finds no production consumer of `CubicSpline` — at baseline
`58c6b818c`, `interpolate` is dead translation surface. And it is on the wrong
substrate (`ndarray` + hand-rolled `Vec<f64>` Thomas solvers rather than ferray's
interpolate / `ferray::linalg` banded-solve analog, R-SUBSTRATE-1). The honest
call (R-HONEST-3): the seven n ≥ 4 value-parity REQs ship on impl + oracle, while
the 3-point not-a-knot bug, the API/default/missing-BC/missing-method gaps, the
error type, and the integration/substrate REQs do not.

## Verification

Commands establishing the claims (run at baseline `58c6b818c`):

- `cargo test -p ferrolearn-numerical --lib interpolate` → 10 passed, 0 failed
  (`interpolates_data_points`, `linear_data_exact`, `quadratic_natural`,
  `not_a_knot_cubic_exact`, `derivative_of_linear`, `integration_linear`,
  `integration_quadratic`, `monotone_data`, `two_points_minimum`,
  `unsorted_x_error`). NOTE: none of the existing tests exercises 3-point
  not-a-knot, which is why the REQ-FEWPOINT bug survives the in-crate suite — the
  critic must add a failing pin.
- value-parity oracle (REQ-NAT-*/REQ-NAK-*/REQ-EXTRAP, R-CHAR-3 — expected from
  scipy, never from ferrolearn):
  `python3 -c "import numpy as np; from scipy.interpolate import CubicSpline; x=np.array([0.,1,2,3,4]); y=np.array([0.,1,4,9,16]); cn=CubicSpline(x,y,bc_type='natural'); ck=CubicSpline(x,y,bc_type='not-a-knot'); print(cn([0.5,1.5,2.5,3.5]).tolist(), float(cn(1.5,1)), float(cn(2.5,2)), float(cn.integrate(0,4))); print(ck([0.5,1.5,2.5,3.5]).tolist(), float(ck(1.5,1)), float(ck(2.5,2)), float(ck.integrate(0,4))); print(cn([-0.5,4.5]).tolist(), ck([-0.5,4.5]).tolist())"`
  → natural `[0.33928571428571425, 2.232142857142857, 6.232142857142858, 12.339285714285714] 3.0357142857142856 2.1428571428571423 21.428571428571427`;
  not-a-knot `[0.25, 2.25, 6.25, 12.25] 3.0 2.0 21.333333333333336`;
  extrap `[-0.33928571428571436, 19.66071428571429] [0.25, 20.25]`. ferrolearn
  matches each to ≤1e-13 (eval/deriv/integrate ≤1e-15, extrap ≤1e-13). A non-cubic
  cross-check (sin, dataset B) confirms the two BCs genuinely differ and both
  match scipy to ≤1e-15.
- few-point divergence oracle (REQ-FEWPOINT, headline):
  `python3 -c "import numpy as np; from scipy.interpolate import CubicSpline; x3=np.array([0.,1,2]); y3=np.array([0.,1,4]); print(CubicSpline(x3,y3,bc_type='natural')([0.5,1.5]).tolist()); print(CubicSpline(x3,y3,bc_type='not-a-knot')([0.5,1.5]).tolist())"`
  → natural `[0.3125, 2.3125]` (ferrolearn MATCH), not-a-knot `[0.2499999999999999, 2.25]`
  (ferrolearn `[0.12499999999999994, 2.375]` — DIVERGES). A critic pins this as a
  FAILING `#[test]` asserting the 3-point not-a-knot spline equals `[0.25, 2.25]`
  to ≤1e-12 (fails until `solve_not_a_knot_general` handles `n=2`).
- default-bc oracle (REQ-DEFAULT-BC):
  `python3 -c "import inspect; from scipy.interpolate import CubicSpline; print(inspect.signature(CubicSpline.__init__).parameters['bc_type'].default)"`
  → `not-a-knot`. ferrolearn `new` requires an explicit `BoundaryCondition`.
  Documented as a structural API divergence (no failing numeric `#[test]`; a
  compile-level contract).
- missing-BC / PPoly-API oracle (REQ-MISSING-BC, REQ-PPOLY-API):
  `python3 -c "import numpy as np; from scipy.interpolate import CubicSpline; x=np.array([0.,1,2,3,4]); y=np.array([0.,1,4,9,16]); cs=CubicSpline(x,y); print(CubicSpline(x,y,bc_type='clamped')([0.5,2.5]).tolist(), float(cs(1.5,2)), cs.derivative().__class__.__name__, cs.x.tolist(), cs.c.shape)"`
  → clamped `[0.23214285714285715, 5.910714285714286]`, `cs(1.5,2)`,
  `cs.derivative()` is a `CubicSpline`/`PPoly`, `cs.x` and `cs.c` (shape `(4, n)`)
  exist. ferrolearn has none of clamped/periodic/custom-BC, general `nu`,
  spline-returning derivative/antiderivative, roots/solve, or `.c`/`.x`.
  Documented as absent surface (no failing numeric `#[test]`; missing-API facts).
- error-type oracle (REQ-ERR-TYPE):
  `python3 -c "import numpy as np; from scipy.interpolate import CubicSpline; CubicSpline(np.array([0.,2,1]), np.array([0.,1,2]))"`
  → `ValueError`. ferrolearn returns `Err(String)`. Convention divergence.
- consumer check (REQ-CONSUMER):
  `grep -rn "interpolate::\|CubicSpline\|BoundaryCondition" --include=*.rs ferrolearn-*/src | grep -v 'interpolate.rs'`
  → empty (no caller). Documented as the blocker, no failing `#[test]` (a
  missing-consumer fact is structural, not a numerical assertion).

Per R-DEFER-2 the table is binary SHIPPED/NOT-STARTED. SHIPPED (impl + scipy
oracle to ~1e-14): REQ-NAT-EVAL, REQ-NAT-DERIV, REQ-NAT-INTEG, REQ-NAK-EVAL,
REQ-NAK-DERIV, REQ-NAK-INTEG, REQ-EXTRAP, REQ-FEWPOINT (3-point not-a-knot FIXED
#1957). NOT-STARTED (open blockers): REQ-DEFAULT-BC (no not-a-knot default #1958),
REQ-MISSING-BC
(no clamped/periodic/custom), REQ-PPOLY-API (no general-nu/spline-returning
derivative/roots/solve/`.c`/`.x`), REQ-ERR-TYPE (`String` not `FerroError`),
REQ-CONSUMER (dead module — no caller), REQ-FERRAY (ndarray + hand-rolled solvers,
not ferray).
