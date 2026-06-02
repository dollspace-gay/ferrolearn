# Isotonic Regression

<!--
tier: 3-component
status: draft
baseline-commit: e9d6069
upstream: scikit-learn 1.5.2 (commit 156ef14)
upstream-paths:
  - sklearn/isotonic.py
ferrolearn-module: ferrolearn-linear/src/isotonic.rs
parity-op: IsotonicRegression (+ free isotonic_regression, check_increasing)
crosslink-issue: 562
-->

## Summary

`ferrolearn-linear/src/isotonic.rs` mirrors scikit-learn's
`sklearn.isotonic.IsotonicRegression` (`sklearn/isotonic.py`): a non-parametric
regressor that fits a monotonic (increasing or decreasing) step function to 1-D
data via the **Pool Adjacent Violators (PAVA)** algorithm, then predicts by
**piecewise-linear interpolation** over the fitted unique-X / pooled-y
breakpoints. ferrolearn provides `pav_increasing` (the PAVA pool), the
unfitted/fitted builder API (`IsotonicRegression::new` /
`FittedIsotonicRegression`), the increasing/decreasing flag, the three
out-of-bounds strategies as an `OutOfBounds` enum, and linear interpolation in
`predict`. The fit produces the correct pooled values on distinct, equal-weight
X (verified against the live oracle below), and the predict-side interpolation
matches sklearn's `interp1d(kind="linear")` contract.

The default `out_of_bounds` (now `'nan'`, REQ-4) and the `_make_unique` weighted
duplicate-X collapse (REQ-8) are SHIPPED (the weighted PAVA + weighted
`make_unique` internal machinery also stage REQ-7's `sample_weight`). The
remaining divergences are: the absence of `increasing='auto'`
(Spearman `check_increasing`), `y_min`/`y_max` clipping, the public
`sample_weight` API (the weighted pool already exists internally),
public fitted-attribute accessors
(`X_min_`/`X_max_`/`X_thresholds_`/`y_thresholds_`/`increasing_`), the free
module functions `isotonic_regression` and `check_increasing`, and the ferray
substrate. Because no current test pins the PAVA-pooled `y_thresholds_` against
the sklearn oracle on a dataset where pooling actually occurs, the fit-parity
REQs (REQ-1/REQ-2) are classified NOT-STARTED pending the critic's pinned
characterization test (R-CHAR-3, R-HONEST-3).

## Algorithm (sklearn — the contract)

`IsotonicRegression.fit(X, y, sample_weight)` (`isotonic.py:349-397`) runs a
pipeline through `_build_y` (`isotonic.py:300-347`) → `isotonic_regression`
(`isotonic.py:111-170`) → `_build_f` (`isotonic.py:288-298`):

1. **Shape check** — `X` must be 1-D `(n_samples,)` or 2-D `(n_samples, 1)`;
   otherwise `ValueError` (`_check_input_data_shape`, `isotonic.py:280-286`).
2. **Resolve direction** (`isotonic.py:306-309`): if `increasing == 'auto'`,
   `self.increasing_ = check_increasing(X, y)` = `sign(spearman_rho(X, y)) >= 0`
   (`check_increasing`, `isotonic.py:76-77`), with a Fisher-transform 95% CI
   warning when the CI spans zero (`isotonic.py:80-96`). Otherwise
   `self.increasing_ = self.increasing`.
3. **Drop zero-weight samples** (`isotonic.py:313-315`): `_check_sample_weight`,
   then `mask = sample_weight > 0`.
4. **Sort** by `np.lexsort((y, X))` — primary key `X`, secondary key `y`
   (`isotonic.py:317-318`).
5. **`_make_unique`** (`isotonic.py:319`, C kernel `sklearn/_isotonic.pyx`):
   collapse runs of equal `X` into a single point whose `y` is the
   **sample-weight-weighted mean** of the run, with summed weight. (Live oracle:
   `X=[1,1,2,3], y=[1,3,2,4]` collapses to `X_thr=[1,2,3]`, `y_thr` pooled from
   the weighted mean of the duplicate `X=1`.)
6. **Weighted PAVA** — `isotonic_regression(unique_y,
   sample_weight=unique_sample_weight, y_min, y_max, increasing)`
   (`isotonic.py:322-328`). For decreasing it sets `order = [::-1]`
   (`isotonic.py:156`), runs the in-place weighted PAVA
   `_inplace_contiguous_isotonic_regression(y, sample_weight)`
   (`isotonic.py:162`, C kernel), then reverses back. The pool merges adjacent
   blocks that violate monotonicity using the **weighted** block mean
   `sum(w_i y_i) / sum(w_i)`.
7. **`y_min`/`y_max` clip** (`isotonic.py:163-169`): if either bound is set,
   `np.clip(y, y_min, y_max)` on the pooled values (default `None` → `±inf`).
8. **Bounds** (`isotonic.py:331`): `X_min_, X_max_ = X.min(), X.max()`.
9. **Trim duplicates** (`isotonic.py:333-341`): aside from the first/last point,
   drop interior points whose pooled `y` equals BOTH neighbors (a speed
   optimization that does not change the linear interpolant).
10. **Store thresholds** (`isotonic.py:393`): `X_thresholds_, y_thresholds_`.
11. **Build interpolant** (`_build_f`, `isotonic.py:288-298`):
    `f_ = scipy.interpolate.interp1d(X, y, kind="linear",
    bounds_error=(out_of_bounds=='raise'))` — so **predict at an interior X
    linearly interpolates** between the two surrounding thresholds. A single
    pooled point yields a constant prediction (`isotonic.py:292-294`).

`predict(T)` / `transform(T)` (`_transform`, `isotonic.py:399-427`): reshape to
1-D, if `out_of_bounds == 'clip'` clamp `T` to `[X_min_, X_max_]`, then evaluate
`f_(T)`; `'nan'` lets `interp1d` return `NaN` outside the domain, `'raise'`
raises `ValueError`.

The free `isotonic_regression(y, *, sample_weight=None, y_min=None, y_max=None,
increasing=True)` (`isotonic.py:111-170`) is the pooling primitive on a `y`
vector (no X); `check_increasing(x, y)` (`isotonic.py:32-98`) returns the
Spearman-rho-sign boolean used by `increasing='auto'`.

## ferrolearn (what exists)

`fn pav_increasing in isotonic.rs` sorts `(x, y)` by `x` only (NOT a
`lexsort((y, x))`), then runs an **unweighted** PAVA: each new point starts a
block `{sum, count, first_idx, last_idx}`, merging with the previous block while
`prev_mean > curr_mean` (`prev.sum/prev.count`). It then emits breakpoints per
block — `(first_x, mean)` and, when `first_x != last_x`, `(last_x, mean)`. The
`Fit` impl (`impl Fit<Array2<F>, Array1<F>> for IsotonicRegression in
isotonic.rs`) extracts the single feature column, dispatches to `pav_increasing`
for `increasing == true` or **negates `y` → increasing PAVA → negates back** for
decreasing (`isotonic.rs:331-335`), and stores
`FittedIsotonicRegression { x_thresholds, y_thresholds, out_of_bounds,
increasing }`. `fn predict_single in isotonic.rs` binary-searches the bracketing
interval and **linearly interpolates** `y0 + t*(y1 - y0)`
(`isotonic.rs:197-199`), with the `OutOfBounds::{Clip,Nan,Raise}` branch applied
when `x < x_min` / `x > x_max`.

`IsotonicRegression` / `FittedIsotonicRegression` are boundary estimator types
re-exported at the crate root (`pub use isotonic::{FittedIsotonicRegression,
IsotonicRegression} in lib.rs`). There is currently **no `ferrolearn-python`
binding** for isotonic regression, and no weighted-block, no `_make_unique`,
no `y_min`/`y_max`, no `'auto'`, and no public fitted-attribute accessors beyond
`is_increasing`.

## Requirements

- REQ-1: Increasing PAVA fit — `pav_increasing` pools adjacent violators to the
  monotone-increasing solution; the fitted breakpoints/`y_thresholds_` match the
  live `IsotonicRegression().fit` oracle on data where pooling occurs.
- REQ-2: Decreasing fit — `increasing=False` produces the monotone-decreasing
  solution matching `IsotonicRegression(increasing=False)` oracle.
- REQ-3: Predict by **piecewise-linear interpolation** over the fitted
  thresholds — `predict` at an interior X linearly interpolates between the two
  surrounding breakpoints, matching sklearn's `interp1d(kind="linear")`.
- REQ-4: `out_of_bounds` ∈ {`nan`, `clip`, `raise`} with **default `'nan'`** —
  predictions outside `[X_min_, X_max_]` are NaN (default), clipped, or raise.
- REQ-5: `y_min` / `y_max` clipping — the pooled `y` is clipped to
  `[y_min, y_max]` (default `None` → unbounded).
- REQ-6: `increasing='auto'` — resolve direction via `check_increasing` (sign of
  the Spearman rho), exposed as `increasing_`.
- REQ-7: `sample_weight` — weighted PAVA (block mean `Σw_i y_i / Σw_i`) plus
  zero-weight-sample removal, matching `fit(X, y, sample_weight=...)`.
- REQ-8: `_make_unique` duplicate-X collapse — runs of equal X collapse to one
  point with the (weighted) mean `y` BEFORE PAVA, matching the oracle on
  duplicate-X inputs.
- REQ-9: Fitted-attribute introspection — `X_min_`, `X_max_`, `X_thresholds_`,
  `y_thresholds_`, `increasing_` exposed on the fitted object.
- REQ-10: Free module functions — `isotonic_regression(y, *, sample_weight,
  y_min, y_max, increasing)` and `check_increasing(x, y)`.
- REQ-11: ferray substrate migration (array type → `ferray-core`; sorting/stats
  → `ferray` analogs) per R-SUBSTRATE.

## Acceptance criteria

- AC-1 (REQ-1): on `X=[1,2,3,4,5,6]`, `y=[1,4,2,5,3,7]`,
  `IsotonicRegression().fit` yields `y_thresholds_ == [1,3,3,4,4,7]` (live
  oracle); ferrolearn's fitted pooled values match within `1e-10`.
- AC-2 (REQ-2): on `X=[1,2,3,4,5]`, `y=[5,3,4,2,1]`,
  `IsotonicRegression(increasing=False).fit` matches the live oracle's pooled
  values (and is monotone non-increasing).
- AC-3 (REQ-3): on `X=[1,3,5]`, `y=[1,3,5]`, predict at `[2,3,4]` equals
  `[2,3,4]` — linear interpolation, distinguishing from a step/nearest function
  (which would give `1`/`3`/`5`). Matches the live oracle `predict([2,3,4]) ==
  [2,3,4]`.
- AC-4 (REQ-4): `IsotonicRegression().out_of_bounds == 'nan'` (live oracle);
  default-fit predict at `[0,4]` outside `[1,3]` returns `[NaN, NaN]`. `'clip'`
  returns the boundary y; `'raise'` errors.
- AC-5 (REQ-5): `IsotonicRegression(y_min=2, y_max=6).fit` on `X=[1,2,3,4]`,
  `y=[1,4,2,7]` clips the pooled `y_thresholds_` to `[2,3,3,6]` (live oracle).
- AC-6 (REQ-6): `IsotonicRegression(increasing='auto').fit` on a decreasing
  relationship resolves `increasing_ == False` (live oracle:
  `check_increasing([1..5],[10,8,6,4,2]) == False`).
- AC-7 (REQ-7): on `X=[1,2,3,4]`, `y=[1,4,2,5]`, `sample_weight=[1,1,5,1]`, the
  weighted fit yields `y_thresholds_ == [1, 2.333…, 2.333…, 5]` (live oracle),
  distinct from the unweighted `[1,3,3,5]`.
- AC-8 (REQ-8): on `X=[1,1,2,3]`, `y=[1,3,2,4]`, the duplicate `X=1` collapses to
  one threshold; `X_thresholds_ == [1,2,3]`, `y_thresholds_ == [2,2,4]` (live
  oracle) — NOT ferrolearn's current `X=[1,2,3]`, `y≈[1,2.5,4]`.
- AC-9 (REQ-9): the fitted object exposes `X_min_`, `X_max_`, `X_thresholds_`,
  `y_thresholds_`, `increasing_` matching the oracle attributes.
- AC-10 (REQ-10): `isotonic_regression([5,3,1,2,8,10,7,9,6,4])` equals the oracle
  (`[2.75×4, 7.33…×6]`); `check_increasing([1..5],[2,4,6,8,10]) == True`.
- AC-11 (REQ-11): `isotonic.rs` owns its computation on `ferray-core` arrays, not
  `ndarray`.

## REQ status

Binary classification (R-DEFER-2): SHIPPED = impl + non-test production consumer
+ tests + green oracle verification; NOT-STARTED = concrete open blocker
referenced by `#`-number. `IsotonicRegression`/`FittedIsotonicRegression` are
boundary estimator types re-exported at the crate root (`pub use
isotonic::{FittedIsotonicRegression, IsotonicRegression} in lib.rs`); under
S5/R-DEFER-1 the public estimator type IS the consumer surface, grandfathered
(there is no `ferrolearn-python` binding for isotonic yet). The existing tests
(`test_monotonicity_increasing`, `test_monotonicity_decreasing`,
`test_already_monotonic`, `test_interpolation`, `test_out_of_bounds_*`,
`test_pav_all_equal`, `test_unsorted_x in isotonic.rs`) assert monotonicity,
identity-ramp interpolation, and the enum branches — they do NOT pin a
PAVA-pooled `y_thresholds_` value against the sklearn oracle on a dataset where
pooling occurs, so the fit-parity REQs cannot be SHIPPED (R-CHAR-3,
R-HONEST-3).

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (increasing PAVA fit) | NOT-STARTED | open prereq blocker #563. Impl `fn pav_increasing in isotonic.rs` pools adjacent violators by the unweighted block mean (`isotonic.rs:240-253`: merge while `prev_mean > curr_mean`), and on `X=[1..6], y=[1,4,2,5,3,7]` it does produce the correct pooled `[1,3,3,4,4,7]` (manually traced; matches the live oracle `y_thresholds_ == [1,3,3,4,4,7]`). BUT no test pins the pooled output against the oracle — `test_monotonicity_increasing in isotonic.rs` asserts only `preds[i] >= preds[i-1]` (monotonicity, not value), and `test_interpolation`/`test_already_monotonic` use already-monotone ramps where PAV is a no-op. Per R-CHAR-3 a live-oracle pooled-value parity test is required and absent; parity is also entangled with the missing weighted-block / `_make_unique` path (REQ-7/REQ-8) on tied/duplicate X. |
| REQ-2 (decreasing fit) | NOT-STARTED | open prereq blocker #564. Impl negates `y`, runs `pav_increasing`, negates back (`fn fit in isotonic.rs`, `isotonic.rs:331-335`: `let neg_ys: Vec<F> = ys.iter().map(\|&v\| -v).collect(); let (rx, ry) = pav_increasing(&xs, &neg_ys); let ry_neg = ...`), structurally mirroring sklearn's `order = [::-1]` (`isotonic.py:156`). `test_monotonicity_decreasing in isotonic.rs` asserts only `preds[i] <= preds[i-1]` (monotonicity, not value). No live-oracle pooled-value parity test (`IsotonicRegression(increasing=False)`), so parity is unverified (R-CHAR-3). |
| REQ-3 (predict piecewise-linear interpolation) | SHIPPED | impl `fn predict_single in isotonic.rs` binary-searches the bracketing interval and **linearly interpolates** `let t = (x - x0) / (x1 - x0); Ok(y0 + t * (y1 - y0))` (`isotonic.rs:197-199`), mirroring sklearn's `interp1d(kind="linear")` (`_build_f`, `isotonic.py:296-298`). Consumer: `FittedIsotonicRegression::predict` (crate-root export). Tests: `test_interpolation in isotonic.rs` fits `X=[1,3,5], y=[1,3,5]` and asserts `predict([2,3,4]) == [2,3,4]` (`isotonic.rs:467-469`) — this distinguishes linear from step/nearest (a step function would give `1`/`3`/`5` at the interior points) and matches the live oracle `predict([2,3,4]) == [2,3,4]`. Verification: `cargo test -p ferrolearn-linear isotonic`. |
| REQ-4 (out_of_bounds {nan,clip,raise} + default nan) | SHIPPED | impl: the three strategies are `pub enum OutOfBounds { Clip, Nan, Raise }`, applied in `fn predict_single in isotonic.rs`; `fn IsotonicRegression::new` now defaults `out_of_bounds: OutOfBounds::Nan`, matching sklearn `out_of_bounds="nan"` (`isotonic.py:274`; `_parameter_constraints` `StrOptions({"nan","clip","raise"})`, `isotonic.py:271`). Consumer: `FittedIsotonicRegression::predict` (crate-root export `pub use isotonic::{...} in lib.rs`). Tests: `isotonic_default_out_of_bounds_nan in divergence_isotonic_fit.rs` (default-fit `predict([0.,4.])` outside `[1,3]` → `[NaN, NaN]`, live oracle `IsotonicRegression().out_of_bounds == 'nan'`); `test_out_of_bounds_clip`/`test_out_of_bounds_nan`/`test_out_of_bounds_raise in isotonic.rs` pin the explicit strategies. R-DEV-2 default-ABI divergence resolved (#565). |
| REQ-5 (y_min / y_max clipping) | NOT-STARTED | open prereq blocker #566. `IsotonicRegression` struct (`isotonic.rs:64-72`) has no `y_min`/`y_max` fields; `fn fit` never clips the pooled `y`. sklearn clips `np.clip(y, y_min, y_max)` after pooling (`isotonic.py:163-169`; constructor `y_min=None, y_max=None`, `isotonic.py:274`; live oracle `y_min=2,y_max=6` → `y_thresholds_ == [2,3,3,6]`). Missing parameter and clip step. |
| REQ-6 (increasing='auto' / check_increasing) | NOT-STARTED | open prereq blocker #567. `IsotonicRegression.increasing: bool` (`isotonic.rs:68`) admits only `true`/`false`; there is no `'auto'` option and no Spearman-rho `check_increasing`. sklearn's `increasing` is `bool or 'auto'` (default `True`; `_parameter_constraints` `["boolean", StrOptions({"auto"})]`, `isotonic.py:270`), `'auto'` → `check_increasing(X, y)` = `sign(spearman_rho) >= 0` (`isotonic.py:306-307, :76-77`), exposed as `increasing_`. Requires a Spearman correlation primitive (ferray::stats analog). |
| REQ-7 (sample_weight / weighted PAVA) | NOT-STARTED | open prereq blocker #568. `fn fit in isotonic.rs` takes only `(x, y)` and `pav_increasing` uses an **unweighted** block mean `block.sum / block.count` (`isotonic.rs:262`); there is no `sample_weight` argument or weighted pool. sklearn's `fit(X, y, sample_weight=None)` removes zero-weight samples and runs the **weighted** `_inplace_contiguous_isotonic_regression(y, sample_weight)` (`isotonic.py:159-162, :313-315`; live oracle: `sample_weight=[1,1,5,1]` → `y_thresholds_ == [1, 2.333…, 2.333…, 5]`, distinct from the unweighted `[1,3,3,5]`). Missing weighted pooling. |
| REQ-8 (_make_unique duplicate-X collapse) | SHIPPED | impl: `fn make_unique in isotonic.rs` orders by `X` (ties broken by `y`, mirroring `np.lexsort((y, X))`, `isotonic.py:317`, via `total_cmp`-safe `partial_cmp`) and collapses each run of equal `X` to one point `(x, Σwᵢyᵢ/Σwᵢ, Σwᵢ)` — the sample-weighted mean and summed weight of `_make_unique` (`sklearn/_isotonic.pyx`); `fn fit in isotonic.rs` then runs `fn pav_increasing_unique_weighted` (the **weighted** PAVA, pooling adjacent violators by `(w₁v₁+w₂v₂)/(w₁+w₂)`) on the unique points. The unweighted `fn pav_increasing` delegates to this pipeline with unit weights (distinct-X byte-identical). Consumer: `IsotonicRegression::fit` → `FittedIsotonicRegression` (crate-root export). Tests: `isotonic_make_unique_duplicate_x in divergence_isotonic_fit.rs` (`X=[1,1,2,3], y=[1,3,2,4]` → `predict([1,2,3]) == [2,2,4]`, live oracle); `test_make_unique_weighted_collapse in isotonic.rs` exercises the weighted collapse (`sample_weight=[3,1,1,1]` → unique `y=[1.5,2,4]`, weights `[4,1,1]`, live oracle). The weighted PAVA + weighted `make_unique` are the internal machinery for `sample_weight` (#568, REQ-7) — only the public-API surface remains there. #569 resolved. |
| REQ-9 (fitted attributes) | NOT-STARTED | open prereq blocker #570. `FittedIsotonicRegression` stores `x_thresholds`/`y_thresholds` privately and exposes only `fn is_increasing` (`isotonic.rs:128-133`). sklearn exposes `X_min_`, `X_max_`, `X_thresholds_`, `y_thresholds_`, `increasing_` as public fitted attributes (`isotonic.py:204-228, :331, :393`); ferrolearn has no `X_min_`/`X_max_`/`X_thresholds_`/`y_thresholds_`/`increasing_` accessors (only the internal binary-search uses `x_thresholds[0]`/`.last()`). Missing introspection surface. |
| REQ-10 (free isotonic_regression + check_increasing) | NOT-STARTED | open prereq blocker #571. The module exposes only the `IsotonicRegression` estimator; there is no free `pub fn isotonic_regression` nor `pub fn check_increasing`. sklearn's `__all__ = ["check_increasing", "isotonic_regression", "IsotonicRegression"]` (`isotonic.py:22`) — `isotonic_regression(y, *, sample_weight, y_min, y_max, increasing)` (`isotonic.py:111-170`, live oracle `isotonic_regression([5,3,1,2,8,10,7,9,6,4]) == [2.75×4, 7.33…×6]`) and `check_increasing(x, y)` (`isotonic.py:32-98`) are public API. Missing both functions. |
| REQ-11 (ferray substrate) | NOT-STARTED | open prereq blocker #572. `isotonic.rs` imports `ndarray::{Array1, Array2}` (`isotonic.rs:34`) and operates on `Array2<F>`/`Array1<F>`, not `ferray-core` arrays (R-SUBSTRATE-1/2). Consistent with the crate-wide deferral (`glm.md`, `ransac.md` keep substrate NOT-STARTED). |

## Architecture

### sklearn (the contract)

`IsotonicRegression(RegressorMixin, TransformerMixin, BaseEstimator)`
(`isotonic.py:173`) — constructor `(*, y_min=None, y_max=None, increasing=True,
out_of_bounds="nan")` (`isotonic.py:274`), all keyword-only. `fit` validates and
runs `_build_y` (sort → `_make_unique` → weighted PAVA → `y_min`/`y_max` clip →
trim duplicates), stores `X_thresholds_`/`y_thresholds_`/`X_min_`/`X_max_`/
`increasing_`, and builds the `interp1d(kind="linear")` interpolant `f_`.
`predict`/`transform` are both `_transform` (clip/`raise`/`nan` handling +
`f_`). The pooling is the weighted PAVA; the prediction is piecewise-linear over
the breakpoints. The free `isotonic_regression`/`check_increasing` functions are
the pooling primitive and the `'auto'` direction oracle.

### ferrolearn (what exists)

`IsotonicRegression<F> { increasing: bool, out_of_bounds: OutOfBounds, _marker }`
(`isotonic.rs:64-72`) with a builder API (`new`, `with_increasing`,
`with_out_of_bounds`). `fn fit` (`impl Fit<Array2<F>, Array1<F>>`) requires
exactly one feature column (`FerroError::ShapeMismatch` otherwise), at least 2
samples (`FerroError::InsufficientSamples`), extracts the column, and runs
`pav_increasing` (or negate→PAVA→negate for decreasing). `pav_increasing` sorts
by X (no `lexsort`/`_make_unique`), pools by the **unweighted** block mean, and
emits `(first_x, last_x)` breakpoints. `FittedIsotonicRegression<F> {
x_thresholds, y_thresholds, out_of_bounds, increasing }` predicts via
`predict_single` (binary-search bracket + linear interpolation + out-of-bounds
branch). Public surface: `is_increasing` only.

### Why the fit-parity REQs are NOT-STARTED rather than SHIPPED

The increasing/decreasing pool is *structurally* PAVA and produces the right
answer on distinct, equal-weight X (REQ-1 manual trace matches the oracle), but
(a) no test pins a pooled `y_thresholds_` value against the live sklearn oracle —
the existing tests check only monotonicity or already-monotone ramps (R-CHAR-3
forbids treating that as parity evidence), and (b) the pool is unweighted and
lacks `_make_unique`, so it provably diverges on duplicate-X (REQ-8) and weighted
(REQ-7) inputs. Under R-HONEST-3 (honest underclaim over unverified overclaim),
REQ-1/REQ-2 are NOT-STARTED until the critic pins the pooled-value oracle test.
REQ-3 (linear interpolation) IS distinguished from a step function by
`test_interpolation` and matches the oracle, so it is SHIPPED.

## Verification

Commands that establish the SHIPPED claim (baseline `e9d6069`):

- `cargo test -p ferrolearn-linear isotonic` — the module unit tests; REQ-3 is
  pinned by `test_interpolation in isotonic.rs` (`predict([2,3,4]) == [2,3,4]`
  on `X=[1,3,5], y=[1,3,5]`, distinguishing linear from step).
- `cargo clippy -p ferrolearn-linear --all-targets -- -D warnings`,
  `cargo fmt --all --check`.

Live sklearn oracle (establishes the NOT-STARTED gaps; expected values per
R-CHAR-3 come from sklearn, never copied from ferrolearn):

```bash
python3 -c "import numpy as np; from sklearn.isotonic import IsotonicRegression, isotonic_regression, check_increasing
# REQ-1: pooled y_thresholds on a pooling dataset
m=IsotonicRegression().fit(np.array([1.,2.,3.,4.,5.,6.]).reshape(-1,1), np.array([1.,4.,2.,5.,3.,7.]))
print('REQ-1 y_thr', m.y_thresholds_.tolist())            # [1,3,3,4,4,7]
# REQ-3: linear interpolation (SHIPPED)
m3=IsotonicRegression(out_of_bounds='clip').fit(np.array([1.,3.,5.]).reshape(-1,1), np.array([1.,3.,5.]))
print('REQ-3 interp', m3.predict(np.array([2.,3.,4.]).reshape(-1,1)).tolist())  # [2,3,4]
# REQ-4: default out_of_bounds
print('REQ-4 default oob', IsotonicRegression().out_of_bounds)                  # nan
print('REQ-4 default predict oob', IsotonicRegression().fit(np.array([1.,2.,3.]).reshape(-1,1), np.array([1.,2.,3.])).predict(np.array([0.,4.]).reshape(-1,1)).tolist())  # [nan,nan]
# REQ-5: y_min/y_max
print('REQ-5 ymm', IsotonicRegression(y_min=2.,y_max=6.,out_of_bounds='clip').fit(np.array([1.,2.,3.,4.]).reshape(-1,1), np.array([1.,4.,2.,7.])).y_thresholds_.tolist())  # [2,3,3,6]
# REQ-6: auto
print('REQ-6 auto', check_increasing([1,2,3,4,5],[10,8,6,4,2]))                 # False
# REQ-7: sample_weight
print('REQ-7 sw', IsotonicRegression(out_of_bounds='clip').fit(np.array([1.,2.,3.,4.]).reshape(-1,1), np.array([1.,4.,2.,5.]), sample_weight=np.array([1.,1.,5.,1.])).y_thresholds_.tolist())  # [1,2.333..,2.333..,5]
# REQ-8: duplicate X
md=IsotonicRegression(out_of_bounds='clip').fit(np.array([1.,1.,2.,3.]).reshape(-1,1), np.array([1.,3.,2.,4.]))
print('REQ-8 dup', md.X_thresholds_.tolist(), md.y_thresholds_.tolist())        # [1,2,3] [2,2,4]
# REQ-10: free fn
print('REQ-10 iso', isotonic_regression([5.,3.,1.,2.,8.,10.,7.,9.,6.,4.]).tolist())"
```

A NOT-STARTED REQ closes only when its fix lands AND a divergence test (expected
values from the live oracle / a sklearn `file:line` constant per R-CHAR-3) goes
green; see the REQ-status table for the SHIPPED/NOT-STARTED split (REQ-3 SHIPPED,
REQ-1/2/4/5/6/7/8/9/10/11 NOT-STARTED).

## Blockers to open

- **#563** — REQ-1 of isotonic: pin the increasing-PAVA pooled `y_thresholds_`
  against the live `IsotonicRegression().fit` oracle on a pooling dataset
  (`X=[1..6], y=[1,4,2,5,3,7]` → `[1,3,3,4,4,7]`); resolve the
  unweighted/no-`_make_unique` entanglement.
- **#564** — REQ-2 of isotonic: pin the decreasing-PAVA pooled values against the
  live `IsotonicRegression(increasing=False)` oracle (not just monotonicity).
- **#565** — REQ-4 of isotonic: change the default `out_of_bounds` from `Clip` to
  `Nan` to match sklearn's `out_of_bounds="nan"` default (R-DEV-2).
- **#566** — REQ-5 of isotonic: add `y_min`/`y_max` parameters and clip the
  pooled `y` to `[y_min, y_max]` after PAVA (`isotonic.py:163-169`).
- **#567** — REQ-6 of isotonic: add `increasing='auto'` resolving the direction
  via a Spearman-rho `check_increasing`, exposed as `increasing_`.
- **#568** — REQ-7 of isotonic: add `sample_weight` with weighted-block PAVA
  (`Σw_i y_i / Σw_i`) and zero-weight-sample removal.
- **#569** — REQ-8 of isotonic: add the `_make_unique` weighted duplicate-X
  collapse (`lexsort((y, X))` then collapse equal-X runs) BEFORE pooling.
- **#570** — REQ-9 of isotonic: expose `X_min_`, `X_max_`, `X_thresholds_`,
  `y_thresholds_`, `increasing_` on the fitted object.
- **#571** — REQ-10 of isotonic: add the free `isotonic_regression(y, *,
  sample_weight, y_min, y_max, increasing)` and `check_increasing(x, y)`
  functions to mirror `sklearn.isotonic.__all__`.
- **#572** — REQ-11 of isotonic: migrate `isotonic.rs` off `ndarray` onto the
  ferray substrate (`ferray-core` arrays, ferray sort/stats analogs) per
  R-SUBSTRATE.
