# SequentialFeatureSelector

<!--
tier: 3-component
status: draft
baseline-commit: 1953dbb5
upstream: scikit-learn 1.5.2 (commit 156ef14)
upstream-paths:
  - sklearn/feature_selection/_sequential.py  # class SequentialFeatureSelector(_RoutingNotSupportedMixin, SelectorMixin, MetaEstimatorMixin, BaseEstimator) (:19-21); __init__(estimator, *, n_features_to_select="auto", tol=None, direction="forward", scoring=None, cv=5, n_jobs=None) (:169-186); _parameter_constraints {estimator:[HasMethods(["fit"])], n_features_to_select:[StrOptions({"auto"}), Interval(RealNotInt,0,1,closed="right"), Interval(Integral,0,None,closed="neither")], tol:[None, Interval(Real,None,None,closed="neither")], direction:[StrOptions({"forward","backward"})], scoring:[None, StrOptions(get_scorer_names()), callable], cv:["cv_object"], n_jobs:[None, Integral]} (:155-167); fit(X, y=None) (:192-270): _validate_data(accept_sparse="csc", ensure_min_features=2, force_all_finite=...) (:211-216) REQUIRES >=2 features; n_features_to_select=="auto" -> n_features-1 (tol set) / n_features//2 (tol None) (:219-225); Integral -> must be < n_features else ValueError "n_features_to_select must be < n_features." (:226-229); Real (0,1] -> int(n_features * frac) (:230-231); tol<0 with forward -> ValueError "tol must be strictly positive when doing forward selection" (:233-236); cv=check_cv(self.cv, y, classifier=is_classifier(estimator)) (:238); cloned_estimator=clone(estimator) (:240); greedy loop (:254-262) over n_iterations = n_features_to_select_ (auto OR forward) else n_features - n_features_to_select_ (backward) (:246-250); is_auto_select tol early-stop (new_score - old_score) < tol -> break (:253,:258-259); backward support_ = ~current_mask (:264-265); n_features_to_select_ = support_.sum() (:268). _get_best_new_feature_score(estimator, X, y, cv, current_mask) (:272-295): candidate_feature_indices = np.flatnonzero(~current_mask) (:278); for each, candidate_mask[feature_idx]=True (:281-282), BACKWARD candidate_mask = ~candidate_mask (complement) (:283-284), X_new=X[:, candidate_mask] (:285), scores[idx]=cross_val_score(estimator, X_new, y, cv=cv, scoring=self.scoring, n_jobs=self.n_jobs).mean() (:286-293); new_feature_idx = max(scores, key=...) (:294) -> Python max over ascending-index dict picks LOWEST index on ties. _get_support_mask -> self.support_ (:297-299). _more_tags allow_nan from estimator (:301-304). get_support/transform/inverse_transform/get_feature_names_out from SelectorMixin (sklearn/feature_selection/_base.py).
ferrolearn-module: ferrolearn-preprocess/src/sequential_feature_selector.rs
parity-ops: SequentialFeatureSelector
crosslink-issue: 1283
-->

## Summary

scikit-learn's `SequentialFeatureSelector` (`_sequential.py:19`) is a greedy
wrapper feature selector: it adds (forward) or removes (backward) one feature at
a time, choosing at each stage the candidate that maximizes the **cross-validation
score of a wrapped, unfitted estimator** — `cross_val_score(estimator, X_subset,
y, cv=cv, scoring=scoring).mean()` (`_get_best_new_feature_score`, `:286-293`).
The estimator, the CV splitter (`cv=5`), the `scoring` metric, the `"auto"`
target-count default, and `tol`-based early stopping are all load-bearing parts of
its contract (`:169-186`,`:219-262`). It exposes the `SelectorMixin` surface
(`support_` bool mask, `get_support`, `transform`, `inverse_transform`,
`get_feature_names_out`) and the `n_features_to_select_` fitted attribute.

`ferrolearn-preprocess/src/sequential_feature_selector.rs` ships the **greedy
search shape** with the unfitted/fitted split, but with a **fundamentally
different scoring interface**: `SequentialFeatureSelector { n_features_to_select:
usize, direction: Direction }` (`new(n, dir)`, accessors `n_features_to_select()`
/ `direction()`; `Direction::{Forward, Backward}`) and `fit<F>(x, y, score_fn)`
takes a **user-supplied scoring CALLBACK** `impl Fn(&Array2<F>, &Array1<F>) ->
Result<F, FerroError>` — there is NO wrapped estimator, NO cross-validation, NO
`scoring` metric, NO `tol`. `forward_search` (start empty, add the candidate
maximizing `score_fn` with strict `score > best_score` so the lowest index wins
ties, until `n_features_to_select` selected) and `backward_search` (start with
all, remove the candidate whose removal maximizes `score_fn`, until
`n_features_to_select` remain) reproduce sklearn's greedy add-best / remove-best
loop and `max`-tie-break. `FittedSequentialFeatureSelector<F> { n_features_in,
selected_indices }` exposes `selected_indices()` (sorted) and
`n_features_selected()`; `Transform` returns the selected columns. Non-test
consumer: the crate re-export `pub use sequential_feature_selector::{Direction,
FittedSequentialFeatureSelector, SequentialFeatureSelector};`
(`ferrolearn-preprocess/src/lib.rs:156-157`), the boundary public API. There is
**no PyO3 binding**.

**Headline finding (the structural gap, REQ-3).** The two APIs are not 1:1. The
GREEDY SEARCH SHAPE matches (REQ-1 SHIPPED), and the scoped error contracts ship
(REQ-2 SHIPPED). But the **score SOURCE** — an unfitted estimator scored by
`cross_val_score(...).mean()` over a `cv` splitter — is **entirely absent** in
ferrolearn, which delegates scoring to an opaque user callback. Everything
sklearn layers on top of that scoring contract (`"auto"` default REQ-4, `tol`
early-stop REQ-5, float-fraction `n_features_to_select` REQ-6, `cv`/`scoring`/
`n_jobs` REQ-7, the `SelectorMixin` bool-mask surface REQ-9) is NOT-STARTED. (REQ-8
validation boundaries are SHIPPED this iteration — #1284/#1285.) This is a
**mostly-NOT-STARTED** unit: 3 SHIPPED (REQ-1/2/8) / 8 NOT-STARTED.

**Two boundary divergences FIXED this iteration (DIV-A #1284 / DIV-B #1285, REQ-8
SHIPPED).** (a) sklearn requires `n_features_to_select < n_features` and raises
`ValueError("n_features_to_select must be < n_features.")` (`:227-228`); ferrolearn
previously allowed `== n_features` — its count guard now uses `>= n_features`.
(b) sklearn's `_validate_data(ensure_min_features=2)` (`:214`) REQUIRES at least 2
features; ferrolearn now rejects a 1-feature `X` via a dedicated guard placed
before the count check (faithful precedence). Two-round critic-verified CLEAN.

## Probes (live sklearn oracle, 1.5.2)

```bash
# REQ-3 — the SCORE SOURCE is an unfitted estimator scored by cross_val_score(...).mean()
# over a cv splitter (NOT a user callback). Forward, n_features_to_select=2, cv=2.
python3 -c "import numpy as np; \
from sklearn.feature_selection import SequentialFeatureSelector; \
from sklearn.linear_model import LinearRegression; \
X=np.array([[1.,10.,0.1],[2.,20.,0.2],[3.,30.,0.3],[4.,40.,0.4],[5.,50.,0.5],[6.,5.,0.6]]); \
y=np.array([1.,2.,3.,4.,5.,6.]); \
s=SequentialFeatureSelector(LinearRegression(), n_features_to_select=2, direction='forward', cv=2).fit(X,y); \
print('get_support', s.get_support().tolist(), 'indices', s.get_support(indices=True).tolist(), 'n_sel', s.n_features_to_select_)"
# -> get_support [True, False, True] indices [0, 2] n_sel 2
#    Each candidate subset is scored by cross_val_score(LinearRegression(), X_subset, y, cv=2).mean()
#    (_get_best_new_feature_score :286-293). ferrolearn has NO estimator / NO cv / NO scoring:
#    fit(x, y, score_fn) takes an opaque user callback Fn(&Array2<F>,&Array1<F>)->Result<F>.

# REQ-8a (DIV-A) — sklearn requires n_features_to_select < n_features (:227-228):
python3 -c "import numpy as np; \
from sklearn.feature_selection import SequentialFeatureSelector; \
from sklearn.linear_model import LinearRegression; \
X=np.array([[1.,10.,0.1],[2.,20.,0.2],[3.,30.,0.3],[4.,40.,0.4]]); y=np.array([1.,2.,3.,4.]); \
SequentialFeatureSelector(LinearRegression(), n_features_to_select=3, cv=2).fit(X,y)"
# -> ValueError: n_features_to_select must be < n_features.   (3 == n_features=3 REJECTED)
#    ferrolearn ALLOWS == n_features: its test_select_all_features selects 3/3. DIVERGENCE.

# REQ-8b (DIV-B) — sklearn ensure_min_features=2 rejects a 1-feature X (:214):
python3 -c "import numpy as np; \
from sklearn.feature_selection import SequentialFeatureSelector; \
from sklearn.linear_model import LinearRegression; \
X1=np.array([[1.],[2.],[3.],[4.]]); y=np.array([1.,2.,3.,4.]); \
SequentialFeatureSelector(LinearRegression(), n_features_to_select=1, cv=2).fit(X1,y)"
# -> ValueError: Found array with 1 feature(s) (shape=(4, 1)) while a minimum of 2 is required ...
#    ferrolearn accepts a 1-feature X (no ensure_min_features guard). DIVERGENCE.

# REQ-4 — n_features_to_select="auto" default -> n_features // 2 when tol is None (:225):
python3 -c "import numpy as np; \
from sklearn.feature_selection import SequentialFeatureSelector; \
from sklearn.linear_model import LinearRegression; \
X=np.array([[1.,10.,0.1],[2.,20.,0.2],[3.,30.,0.3],[4.,40.,0.4]]); y=np.array([1.,2.,3.,4.]); \
s=SequentialFeatureSelector(LinearRegression(), cv=2).fit(X,y); \
print('auto n_sel (n//2)', s.n_features_to_select_, 'support', s.get_support().tolist())"
# -> auto n_sel (n//2) 1 support [True, False, False]
#    ferrolearn has no "auto": n_features_to_select is a required usize ctor arg.
```

## Requirements

- REQ-1: **Greedy forward/backward search shape + lowest-index tie-break** — at
  each step evaluate every not-yet-decided candidate, pick the one maximizing the
  per-subset score, with strict `>` so the lowest index wins ties; forward starts
  empty and adds until `n_features_to_select` selected, backward starts full and
  removes until `n_features_to_select` remain. Mirrors sklearn's
  `_get_best_new_feature_score` loop (`:280-293`) + `max(scores, key=...)`
  lowest-index tie-break (`:294`) and the `fit` greedy loop (`:254-262`). **Scope:
  the SCORE is the user callback, not estimator+CV (that is REQ-3).** Supports
  `f32` and `f64`.
- REQ-2: **Error contracts (scoped)** — `InvalidParameter` on
  `n_features_to_select == 0` and on `n_features_to_select > n_features`,
  `InsufficientSamples` on zero rows, `ShapeMismatch` on `y.len() != n_samples` in
  `fit` and on a column-count mismatch in `transform`, plus propagation of
  `score_fn` errors. (sklearn's analogous validation lives in `_validate_data`,
  `:211-216`, and the `n_features_to_select` constraints `:226-231`.)
- REQ-3: **Wrapped estimator + `cross_val_score` scoring** (the load-bearing
  structural gap, HEADLINE) — score each candidate subset with
  `cross_val_score(clone(estimator), X_subset, y, cv=cv, scoring=scoring).mean()`
  (`:286-293`) over an unfitted estimator (`HasMethods(["fit"])`, `:156`) and a
  `cv` splitter (`check_cv`, `:238`), with the backward direction scoring the
  COMPLEMENT mask (`candidate_mask = ~candidate_mask`, `:283-284`). ferrolearn's
  `fit(x, y, score_fn)` instead takes an opaque user callback `Fn(&Array2<F>,
  &Array1<F>) -> Result<F>` — no estimator, no cross-validation, no `scoring`
  (Probe REQ-3).
- REQ-4: **`n_features_to_select="auto"` default** (`:219-225`) — when `"auto"`,
  set `n_features_to_select_ = n_features // 2` (tol None) or `n_features - 1`
  (tol set). ferrolearn's `n_features_to_select` is a required `usize`; there is
  no `"auto"` mode (Probe REQ-4: sklearn `auto -> n//2 = 1`).
- REQ-5: **`tol` early-stopping + forward `tol > 0` validation** (`:233-236`,
  `:253`,`:258-259`) — when `is_auto_select` (`tol is not None` and
  `n_features_to_select == "auto"`), stop adding/removing once `(new_score -
  old_score) < tol`; reject `tol < 0` with `direction="forward"`
  (`ValueError("tol must be strictly positive ...")`). ferrolearn has no `tol`
  parameter and no early stopping.
- REQ-6: **Float `n_features_to_select` fraction** (`Interval(RealNotInt, 0, 1,
  closed="right")`, `:159`,`:230-231`) — a float in `(0, 1]` selects
  `int(n_features * frac)` features. ferrolearn's `n_features_to_select` is `usize`
  (absolute count only).
- REQ-7: **`cv` / `scoring` / `n_jobs` parameters** (`:164-166`,`:177-178`,
  `:286-293`) — a CV splitter (`cv=5` default via `check_cv`), a `scoring` metric
  str/callable, and `n_jobs` fold parallelism. All absent in ferrolearn.
- REQ-8: **`n_features_to_select < n_features` + `ensure_min_features=2`
  validation contract** (`:214`,`:227-228`) — reject `n_features_to_select >=
  n_features` (DIV-A; ferrolearn allows `== n_features`, Probe REQ-8a) and reject
  fewer than 2 features (DIV-B; ferrolearn accepts a 1-feature `X`, Probe REQ-8b).
  Candidate fixable divergences the critic may pin (DIV-A especially).
- REQ-9: **`SelectorMixin` surface + `n_features_to_select_` attr** (`:297-299`,
  `:267-268`; `SelectorMixin` in `sklearn/feature_selection/_base.py`) — boolean
  `support_` mask, `get_support()` / `get_support(indices=True)`,
  `inverse_transform` (zero-pad dropped columns), `get_feature_names_out`, and the
  `n_features_to_select_` fitted attribute. ferrolearn exposes
  `selected_indices()` (the `get_support(indices=True)` analog) and `transform`,
  but not the boolean mask / inverse / feature-name surface.
- REQ-10: **PyO3 binding** — `import ferrolearn` exposing a registered
  `SequentialFeatureSelector` marshalling `fit`/`transform`, the project boundary
  CPython consumer. Absent.
- REQ-11: **ferray substrate** — compute over `ferray-core` arrays / `ferray-ufunc`
  rather than `ndarray::{Array1, Array2}` + `num_traits::Float` (R-SUBSTRATE).

## Acceptance criteria

- AC-1 (REQ-1): `SequentialFeatureSelector::new(1, Direction::Forward).fit(&x, &y,
  score_fn)` for `x=[[1,10,0.1],[2,20,0.2],[3,30,0.3]]`, a sum-of-column-means
  `score_fn`, yields `selected_indices() == [1]` (the highest-mean column); a
  two-column tie resolves to the lowest index via the strict `>` rule. Backward
  with the same fixture yields `[1]`. Pinned by `test_forward_selects_best`,
  `test_backward_selects_best`, `test_forward_select_two`,
  `test_backward_select_two`, `test_indices_sorted`.
- AC-2 (REQ-2): `fit` with `n_features_to_select == 0` returns
  `Err(InvalidParameter)`; `> n_features` returns `Err(InvalidParameter)`; `(0, n)`
  rows returns `Err(InsufficientSamples)`; `y.len() != n_rows` returns
  `Err(ShapeMismatch)`; a column-count mismatch on `transform` returns
  `Err(ShapeMismatch)`; a failing `score_fn` propagates `Err`. Pinned by
  `test_zero_features_error`, `test_too_many_features_error`,
  `test_zero_rows_error`, `test_y_length_mismatch`,
  `test_shape_mismatch_on_transform`, `test_score_fn_error_propagated`.
- AC-3 (REQ-3): a fitted handle scores candidate subsets with
  `cross_val_score(estimator, X_subset, y, cv, scoring).mean()` and reproduces the
  Probe REQ-3 forward result `get_support [True, False, True]` for
  `LinearRegression()`, `n_features_to_select=2`, `cv=2` on the 6-row fixture;
  backward scores the complement mask (`:283-284`).
- AC-4 (REQ-4): `n_features_to_select="auto"` with `tol=None` selects `n // 2`
  features (Probe REQ-4: `n//2 = 1` on the 4-feature fixture); with `tol` set it
  starts from `n - 1`.
- AC-5 (REQ-5): with `"auto"` + `tol`, the loop stops once `(new_score -
  old_score) < tol`; `tol < 0` with forward selection returns
  `Err(InvalidParameter)`.
- AC-6 (REQ-6): a float `n_features_to_select=0.5` on a 4-feature `X` selects
  `int(4 * 0.5) = 2` features.
- AC-7 (REQ-7): `cv`, `scoring`, and `n_jobs` are accepted and routed into the
  per-candidate `cross_val_score` call (`:286-293`).
- AC-8 (REQ-8): `fit` with `n_features_to_select == n_features` returns
  `Err(InvalidParameter)` (sklearn `must be < n_features`, Probe REQ-8a) — NOT the
  current success; a 1-feature `X` returns `Err` (`ensure_min_features=2`, Probe
  REQ-8b).
- AC-9 (REQ-9): a fitted handle exposes `get_support()` (`[true,false,true]`),
  `inverse_transform` (zero-pad dropped columns), `get_feature_names_out`, and the
  `n_features_to_select_` count alongside `selected_indices()`.
- AC-10 (REQ-10): `python3 -c "import ferrolearn; ..."` resolves a registered
  `SequentialFeatureSelector`; `.fit(X, y).transform(X)` matches the Probe REQ-3
  selection.
- AC-11 (REQ-11): the search/score path computes on `ferray-core` arrays.

`## REQ status`

## REQ status

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (greedy forward/backward search shape + lowest-index tie-break) | SHIPPED | impl `pub fn fit in sequential_feature_selector.rs` dispatches to `fn forward_search` / `fn backward_search`. `forward_search` starts `selected` empty, and per step over `0..n_features_to_select` iterates every `candidate` in `remaining`, sets `best_score = F::neg_infinity()`, and keeps the candidate with strict `if score > best_score` — mirroring sklearn `_get_best_new_feature_score` (`_sequential.py:280-293`, candidate loop) + `new_feature_idx = max(scores, key=...)` (`:294`), whose Python `max` over an ascending-index dict picks the LOWEST index on ties (ferrolearn's strict `>` gives the identical lowest-index tie-break). `backward_search` starts `remaining = 0..n_features` and removes the candidate whose removal maximizes the score (same strict-`>` rule) until `remaining.len() == n_features_to_select` — mirroring the backward greedy loop (`:264-265` complement is folded into the by-removal scoring here). `selected_indices` returned sorted ascending. Generic `F: Float + Send + Sync + 'static` covers `f32`/`f64`. **Scope (R-HONEST-4):** the per-subset SCORE is the user `score_fn` callback, NOT `cross_val_score(estimator, ...)` — the search *shape* matches; the score *source* is REQ-3 (NOT-STARTED). Non-test consumer: crate re-export `pub use sequential_feature_selector::{Direction, FittedSequentialFeatureSelector, SequentialFeatureSelector};` (`ferrolearn-preprocess/src/lib.rs:156-157`), the boundary public API (grandfathered S5/R-DEFER-1). Verification: `cargo test -p ferrolearn-preprocess` (`test_forward_selects_best`, `test_backward_selects_best`, `test_forward_select_two`, `test_backward_select_two`, `test_indices_sorted`). |
| REQ-2 (error contracts, scoped) | SHIPPED | `fit` returns `Err(FerroError::InvalidParameter { name:"n_features_to_select", reason:"must be at least 1" })` when `self.n_features_to_select == 0`; `Err(FerroError::InvalidParameter { name:"n_features_to_select", .. })` (reason `"n_features_to_select (N) exceeds number of features (M)"`) when `self.n_features_to_select > n_features`; `Err(FerroError::InsufficientSamples { required:1, actual:0, context:"SequentialFeatureSelector::fit" })` when `n_samples == 0`; `Err(FerroError::ShapeMismatch { context:"SequentialFeatureSelector::fit — y must match x rows" })` when `y.len() != n_samples`; `forward_search`/`backward_search` propagate `score_fn` errors via `?`. `Transform::transform` returns `Err(FerroError::ShapeMismatch { context:"FittedSequentialFeatureSelector::transform" })` when `x.ncols() != self.n_features_in`. Non-test consumer: the error path guards every fitted instance reached through the crate re-export (`lib.rs:156-157`). Verification: `cargo test -p ferrolearn-preprocess` (`test_zero_features_error`, `test_too_many_features_error`, `test_zero_rows_error`, `test_y_length_mismatch`, `test_shape_mismatch_on_transform`, `test_score_fn_error_propagated`). **Scoped + DIV flag:** sklearn's analogous validation is `_validate_data(ensure_min_features=2)` (`:214`) and the `n_features_to_select >= n_features -> ValueError` constraint (`:227-228`); ferrolearn's `> n_features` guard (`fit`) allows `== n_features` and accepts a 1-feature `X` — those two boundary divergences are REQ-8 (DIV-A/DIV-B), not this row. |
| REQ-3 (wrapped estimator + `cross_val_score` scoring) | NOT-STARTED | open prereq blocker #1286. **The load-bearing structural gap.** `fit<F>(x, y, score_fn)` takes a user callback `impl Fn(&Array2<F>, &Array1<F>) -> Result<F, FerroError>` and `forward_search`/`backward_search` call `score_fn(&x_sub, y)?` directly. sklearn scores each candidate subset with `cross_val_score(estimator, X_new, y, cv=cv, scoring=self.scoring, n_jobs=self.n_jobs).mean()` (`_sequential.py:286-293`) over an unfitted `estimator` (`HasMethods(["fit"])`, `:156`) cloned via `clone(self.estimator)` (`:240`) and a `cv` splitter from `check_cv(self.cv, y, classifier=is_classifier(estimator))` (`:238`); the backward direction scores the COMPLEMENT mask (`candidate_mask = ~candidate_mask`, `:283-284`). ferrolearn has NO estimator, NO cross-validation, NO `scoring` — the score source is entirely absent (Probe REQ-3: sklearn `LinearRegression()`, `cv=2`, forward -> `[0, 2]`). |
| REQ-4 (`n_features_to_select="auto"` default) | NOT-STARTED | open prereq blocker #1287. `SequentialFeatureSelector { n_features_to_select: usize, .. }` is a required absolute count; `new(n, dir)` has no `"auto"` mode. sklearn's `n_features_to_select="auto"` default (`:173`) sets `n_features_to_select_ = n_features // 2` (tol None) or `n_features - 1` (tol set) (`:219-225`). Probe REQ-4 (`auto -> n//2 = 1`) unavailable. |
| REQ-5 (`tol` early-stop + forward `tol > 0` validation) | NOT-STARTED | open prereq blocker #1288. There is no `tol` parameter and no early stopping: the greedy loops run a fixed `0..n_features_to_select` / `while remaining.len() > n_features_to_select`. sklearn breaks once `(new_score - old_score) < tol` when `is_auto_select` (`:253`,`:258-259`) and rejects `tol < 0` with `direction="forward"` (`ValueError`, `:233-236`). |
| REQ-6 (float `n_features_to_select` fraction) | NOT-STARTED | open prereq blocker #1289. `n_features_to_select: usize` is an absolute count; `new(n: usize, ..)` cannot express a fraction. sklearn accepts `Interval(RealNotInt, 0, 1, closed="right")` (`:159`) -> `int(n_features * frac)` (`:230-231`). |
| REQ-7 (`cv` / `scoring` / `n_jobs` parameters) | NOT-STARTED | open prereq blocker #1290. The ctor has only `n_features_to_select` and `direction`; there is no `cv` splitter (sklearn `cv=5` default via `check_cv`, `:177`,`:238`), no `scoring` metric (`:176`), and no `n_jobs` fold parallelism (`:178`,`:292`). All three feed the per-candidate `cross_val_score` of REQ-3 and are absent. |
| REQ-8 (`< n_features` + `ensure_min_features=2` validation; DIV-A/DIV-B) | SHIPPED (closed #1284, #1285) | Both boundary divergences fixed. (a) DIV-A (#1284): `fit`'s count guard changed from `> n_features` to `>= n_features` with message mirroring sklearn "must be < n_features" (`:227-228`) — `n_features_to_select == n_features` now errors; in-module `test_select_all_features` rewritten to `test_select_all_features_rejected` (R-HONEST-4). (b) DIV-B (#1285): added `if n_features < 2 { Err(InvalidParameter "Found array with N feature(s) while a minimum of 2 is required...") }` placed BEFORE the count guard, mirroring sklearn `_validate_data(ensure_min_features=2)` precedence (`:214`,`:211-216` runs before `:227`). Live oracle (R-CHAR-3): `sel==n_features` → Err; 1-feature X → Err (min-features msg); 1-feature X + `sel=5` → min-features error FIRST (not count). Guards `divergence_n_features_to_select_equals_n_features` + `divergence_ensure_min_features_two` + 7 re-audit precedence guards PASS. Consumer: re-export lib.rs:156-157. Two-round critic-verified CLEAN (incl. valid `sel < n_features` not over-rejected). |
| REQ-9 (SelectorMixin surface + `n_features_to_select_` attr) | NOT-STARTED | open prereq blocker #1291. `FittedSequentialFeatureSelector<F>` exposes `selected_indices()` (the `get_support(indices=True)` analog) and `n_features_selected()` and a `Transform` impl, but NOT the boolean `support_` / `get_support()` mask, `inverse_transform` (zero-pad dropped columns), `get_feature_names_out`, or the `n_features_to_select_ = support_.sum()` fitted attribute that sklearn inherits from `SelectorMixin` (`_get_support_mask`, `:297-299`; `:268`). |
| REQ-10 (PyO3 binding) | NOT-STARTED | open prereq blocker #1292. No `ferrolearn-python` registration of `SequentialFeatureSelector` (grep `SequentialFeatureSelector`/`sequential_feature` across `ferrolearn-python/` returns nothing); the only non-test consumer is the crate re-export (`lib.rs:156-157`). The boundary CPython `import ferrolearn` selector surface is absent. |
| REQ-11 (ferray substrate) | NOT-STARTED | open prereq blocker #1293. The search/score path uses `ndarray::{Array1, Array2}` (`x.ncols()`, `Array2::zeros`, the `select_columns` gather) + `num_traits::Float` and `Vec` bookkeeping — not `ferray-core` / `ferray-ufunc` (R-SUBSTRATE-1/2). |

## Architecture

**ferrolearn (existing).** `sequential_feature_selector.rs` exposes the
unfitted/fitted pair plus a `Direction` enum. `SequentialFeatureSelector {
n_features_to_select: usize, direction: Direction }` is constructed by
`new(n_features_to_select, direction)` with accessors `n_features_to_select()` and
`direction()`; `Direction` has `Forward` and `Backward`. `fit<F: Float + Send +
Sync + 'static>(&self, x: &Array2<F>, y: &Array1<F>, score_fn: impl Fn(&Array2<F>,
&Array1<F>) -> Result<F, FerroError>)` validates `n_features_to_select == 0`
(`InvalidParameter`), `n_features_to_select > n_features` (`InvalidParameter`),
`n_samples == 0` (`InsufficientSamples`), and `y.len() != n_samples`
(`ShapeMismatch`), then dispatches to `fn forward_search` or `fn backward_search`.
`forward_search` maintains `selected` and `remaining`, and per step picks the
candidate with strict `score > best_score` (lowest-index tie-break), building
`x_sub` via the free `fn select_columns<F>(x, indices) -> Array2<F>` gather.
`backward_search` maintains `remaining = 0..n_features` and removes the candidate
whose removal maximizes the score until `remaining.len() == n_features_to_select`.
`FittedSequentialFeatureSelector<F> { n_features_in: usize, selected_indices:
Vec<usize>, _marker: PhantomData<F> }` exposes `selected_indices()` and
`n_features_selected()`; `impl Transform<Array2<F>>` returns `ShapeMismatch` on a
column-count mismatch else `select_columns(x, &self.selected_indices)`. The crate
re-exports all three public items (`lib.rs:156-157`); there is no PyO3 binding.

**sklearn (target contract).** `SequentialFeatureSelector(_RoutingNotSupportedMixin,
SelectorMixin, MetaEstimatorMixin, BaseEstimator)` (`:19-21`) takes
`__init__(estimator, *, n_features_to_select="auto", tol=None, direction="forward",
scoring=None, cv=5, n_jobs=None)` (`:169-186`) under `_parameter_constraints`
(`:155-167`) requiring `estimator` to have `fit` (`HasMethods(["fit"])`). `fit(X,
y=None)` (`:192`) calls `_validate_data(accept_sparse="csc", ensure_min_features=2,
...)` (`:211-216`) — at least 2 features required; resolves
`n_features_to_select_` from `"auto"` (`n_features // 2` or `n_features - 1`,
`:219-225`), an `Integral` (`< n_features` else `ValueError`, `:226-229`), or a
`Real` in `(0, 1]` (`int(n_features * frac)`, `:230-231`); rejects forward `tol <
0` (`:233-236`); builds `cv = check_cv(self.cv, y, classifier=...)` (`:238`) and
`cloned_estimator = clone(self.estimator)` (`:240`). The greedy loop (`:254-262`)
runs `n_iterations = n_features_to_select_` (auto or forward) else `n_features -
n_features_to_select_` (backward, `:246-250`); each iteration calls
`_get_best_new_feature_score` (`:272-295`), which for every candidate not in
`current_mask` builds `candidate_mask` (complemented for backward, `:283-284`),
scores `cross_val_score(estimator, X[:, candidate_mask], y, cv=cv,
scoring=self.scoring, n_jobs=self.n_jobs).mean()` (`:286-293`), and returns the
`max`-scoring (lowest-index-on-ties) feature (`:294`). `is_auto_select` triggers
the `tol` early break (`:258-259`). Backward sets `support_ = ~current_mask`
(`:264-265`); finally `n_features_to_select_ = support_.sum()` (`:268`).
`_get_support_mask` returns `support_` (`:297-299`); `SelectorMixin` supplies
`get_support`, `transform`, `inverse_transform`, `get_feature_names_out`.

**The structural gap.** ferrolearn matches sklearn on the *greedy search shape*
(REQ-1: the strict-`>` add-best / remove-best loop reproduces
`_get_best_new_feature_score` + `max` lowest-index tie-break) and on the scoped
error contracts (REQ-2). But the *score source* (REQ-3) is fundamentally
different: ferrolearn delegates scoring to an opaque user callback
`Fn(&Array2<F>, &Array1<F>) -> Result<F>`, whereas sklearn scores each candidate
subset by **cross-validating a wrapped, cloned estimator** (`cross_val_score(...)
.mean()`, `:286-293`). Every contract that sklearn layers on the estimator+CV
scoring is therefore NOT-STARTED: the `"auto"` target-count default (REQ-4), the
`tol` early-stop (REQ-5), the float-fraction `n_features_to_select` (REQ-6), the
`cv`/`scoring`/`n_jobs` parameters (REQ-7), the `< n_features` + `ensure_min_features=2`
validation (REQ-8, DIV-A/DIV-B), the `SelectorMixin` bool-mask surface (REQ-9),
the PyO3 binding (REQ-10), and the ferray substrate (REQ-11). This is a
**mostly-NOT-STARTED** unit (2 SHIPPED / 9 NOT-STARTED).

## Verification

Commands establishing the SHIPPED claims (REQ-1, REQ-2):

```bash
# Crate gauntlet — REQ-1 (search shape + tie-break) and REQ-2 (error contracts):
cargo test -p ferrolearn-preprocess   # incl. test_forward_selects_best,
                                      #       test_backward_selects_best,
                                      #       test_forward_select_two,
                                      #       test_backward_select_two,
                                      #       test_indices_sorted,
                                      #       test_transform,
                                      #       test_zero_features_error,
                                      #       test_too_many_features_error,
                                      #       test_zero_rows_error,
                                      #       test_y_length_mismatch,
                                      #       test_shape_mismatch_on_transform,
                                      #       test_score_fn_error_propagated,
                                      #       test_accessors
cargo clippy -p ferrolearn-preprocess --all-targets -- -D warnings
cargo fmt --all --check

# Oracle (Probe REQ-3) — the estimator+CV score SOURCE ferrolearn does NOT model:
python3 -c "import numpy as np; \
from sklearn.feature_selection import SequentialFeatureSelector; \
from sklearn.linear_model import LinearRegression; \
X=np.array([[1.,10.,0.1],[2.,20.,0.2],[3.,30.,0.3],[4.,40.,0.4],[5.,50.,0.5],[6.,5.,0.6]]); \
y=np.array([1.,2.,3.,4.,5.,6.]); \
print(SequentialFeatureSelector(LinearRegression(), n_features_to_select=2, cv=2).fit(X,y).get_support(indices=True).tolist())"
#   -> [0, 2]   (cross_val_score-driven; ferrolearn's score_fn callback is a different interface)
```

The existing `#[test]`s exercise REQ-1 (search shape, tie-break, sorted indices,
transform) and REQ-2 (every error path) with a synthetic `mean_sum_score`
callback; they are **callback-grounded, not estimator+CV oracle-grounded** — by
construction, since the scoring interface diverges (REQ-3). No currently-green
command establishes REQ-3..REQ-11. Note `test_select_all_features` selects 3/3,
which sklearn REJECTS (`n_features_to_select must be < n_features`, REQ-8 DIV-A) —
the critic may pin that boundary.

## Blockers

REQ-1, REQ-2 are SHIPPED (search shape + scoped error contracts); REQ-8 was
SHIPPED this iteration (DIV-A #1284 + DIV-B #1285 fixed). The remaining
NOT-STARTED REQs are open `-l blocker` issues referenced by the REQ status table:

- #1286 — REQ-3 (HEADLINE): `fit(x, y, score_fn)` takes a user scoring callback;
  sklearn scores each candidate subset with `cross_val_score(clone(estimator),
  X_subset, y, cv, scoring, n_jobs).mean()` over an unfitted estimator + `cv`
  splitter, backward scoring the complement mask (`_sequential.py:240`,`:283-293`).
  The score SOURCE (estimator + cross-validation) is entirely absent.
- #1287 — REQ-4: no `n_features_to_select="auto"` mode (`n_features // 2` /
  `n_features - 1`, `:219-225`); the count is a required `usize`.
- #1288 — REQ-5: no `tol` parameter / early stopping (`:253`,`:258-259`) and no
  forward `tol > 0` validation (`:233-236`).
- #1289 — REQ-6: `n_features_to_select` is `usize`, not `Interval(RealNotInt, 0,
  1)` — cannot express a fractional `int(n_features * frac)` (`:159`,`:230-231`).
- #1290 — REQ-7: no `cv` / `scoring` / `n_jobs` parameters (`:164-166`,`:286-293`).
- #1284 — REQ-8 DIV-A (CLOSED/fixed): `fit` count guard `> n_features` →
  `>= n_features`, mirroring sklearn `must be < n_features` (`:227-228`).
- #1285 — REQ-8 DIV-B (CLOSED/fixed): added `n_features < 2` guard before the
  count check, mirroring sklearn `ensure_min_features=2` precedence (`:214`).
- #1291 — REQ-9: no boolean `support_`/`get_support` / `inverse_transform` /
  `get_feature_names_out` / `n_features_to_select_` (SelectorMixin surface,
  `:297-299`,`:268`).
- #1292 — REQ-10: no `ferrolearn-python` `SequentialFeatureSelector` binding
  (boundary CPython consumer absent).
- #1293 — REQ-11: search/score path on `ndarray`/`num_traits`/`Vec`, not ferray
  (R-SUBSTRATE-1/2).
```
