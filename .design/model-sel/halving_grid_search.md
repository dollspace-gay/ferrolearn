# halving_grid_search

<!--
tier: 3-component
status: draft
baseline-commit: 5ce2c995d12609b0fe0215585b395b6beae6f8a6
upstream-paths:
  - sklearn/model_selection/_search_successive_halving.py   # BaseSuccessiveHalving (:63), __init__ (:90-124), _check_input_parameters/min_resources_ (:126-189), _select_best_index (:191-212), _run_search (:258-365), _top_k (:48-60), HalvingGridSearchCV (:384), _generate_candidate_params (:713)
-->

## Summary

`ferrolearn-model-sel/src/halving_grid_search.rs` mirrors scikit-learn's
`HalvingGridSearchCV` (`sklearn/model_selection/_search_successive_halving.py:384`,
subclass of `BaseSuccessiveHalving` `:63`, itself a `BaseSearchCV` subclass) —
successive-halving hyperparameter search: evaluate all candidates on a small
resource budget, keep the top fraction by score, grow the budget, and repeat.

ferrolearn exposes `pub struct HalvingGridSearchCV<'a>` with the same closure-base
shape as `GridSearchCV` (R-DEV-7): a `pipeline_factory: Box<dyn Fn(&ParamSet) ->
Pipeline + 'a>`, a pre-expanded `param_grid: Vec<ParamSet>`, a `cv: Box<dyn
CrossValidator>`, a `scoring: fn(&Array1<f64>, &Array1<f64>) -> Result<f64,
FerroError>`, and builder fields `factor`/`min_resources`/`max_resources`/
`aggressive_elimination`. It REUSES the shared `CvResults` type from
`grid_search.rs` (`best_index`/`best_params`/`best_score` are the SAME fixed
first-on-tie/NaN-worst reduction).

This unit is RIGOROUSLY honest about being an APPROXIMATE, heuristic
successive-halving: the LOOP SHAPE matches sklearn (evaluate on a budget, eliminate
the worst by `factor`, grow the budget, repeat), and — verified against the live
oracle — the per-round KEEP-COUNT arithmetic `ceil(n_candidates / factor)` MATCHES
sklearn's `_top_k` (`:359`, `n_candidates_to_keep = ceil(n_candidates / factor)`;
ferrolearn `n.div_ceil(self.factor)`). Those facets are SHIPPED. But nearly every
NUMERIC facet of the SCHEDULE diverges: `min_resources_` uses a `ceil(max_res /
factor^n_rounds)` heuristic instead of sklearn's `n_splits * 2 * [n_classes]` +
`exhaust` adjustment; the `n_resources`/`n_iterations` schedule is a `budget *=
factor`-until-`max_res` loop, not sklearn's `int(factor^power * min_res)` /
`min(n_possible, n_required)`; `cv_results` records ONLY the final round's survivors
(no `iter`/`n_resources` columns, no per-iteration history); resource subsampling
takes the FIRST `budget` rows rather than sklearn's RNG-coupled
`_SubsampleMetaSplitter` random fraction; there is no refit/`best_estimator_`, none
of the eight halving attributes (`n_resources_`/`n_candidates_`/…), and no
string-default / classifier-aware constructor surface. Each of those is NOT-STARTED
with a filed blocker.

**Honest note on a NON-divergence.** A naive reading expects the elimination
keep-count to differ (sklearn floor vs ferrolearn ceil). It does NOT: sklearn
`_top_k` keeps `ceil(n_candidates / factor)` (`:359`), and ferrolearn's `div_ceil`
keeps the same count. There is therefore NO deterministic single-spot fixable
divergence in this unit — every divergence below is architectural/missing-feature.
The overall schedule still diverges (via `min_resources_`/`n_resources`), so even
the matching keep-count does NOT make the unit match sklearn end-to-end.

## Upstream reference (scikit-learn 1.5.2, commit 156ef14)

### `BaseSuccessiveHalving.__init__` (`sklearn/model_selection/_search_successive_halving.py:90-124`)
- `:94-106` — keyword-only defaults: `scoring=None`, `n_jobs=None`, `refit=True`,
  `cv=5`, `verbose=0`, `random_state=None`, `error_score=np.nan`,
  `return_train_score=True`, `max_resources="auto"`, `min_resources="exhaust"`,
  `resource="n_samples"`, `factor=3`, `aggressive_elimination=False`.
- `:71-86` — `_parameter_constraints`: `factor` is `Interval(Real, 0, None,
  closed="neither")` (>0); `max_resources` ∈ `{Interval(Integral,0,..), "auto"}`;
  `min_resources` ∈ `{Interval(Integral,0,..), "exhaust", "smallest"}`.

### `_check_input_parameters` / `min_resources_` (`:126-189`)
- `:154-167` — if `min_resources_ in ("smallest","exhaust")` and
  `resource == "n_samples"`: `n_splits = cv.get_n_splits(...)`; `magic_factor = 2`;
  `min_resources_ = n_splits * magic_factor`; if `is_classifier(estimator)`:
  `min_resources_ *= n_classes` (`n_classes = np.unique(y).shape[0]`). For
  `resource != "n_samples"`: `min_resources_ = 1`.
- `:171-177` — `max_resources_ = max_resources`; if `"auto"`: `= _num_samples(X)`
  (requires `resource == "n_samples"`).
- `:179-189` — raise `ValueError` if `min_resources_ > max_resources_` or
  `min_resources_ == 0`.

### `_select_best_index` (`:191-212`)
- `:201-202` — `last_iter = np.max(results["iter"])`;
  `last_iter_indices = np.flatnonzero(results["iter"] == last_iter)`.
- `:204-210` — among the LAST iteration's candidates, `best_idx =
  np.nanargmax(test_scores)` (or `0` if all NaN); `return
  last_iter_indices[best_idx]`. The best candidate comes from the LAST iteration
  ONLY (BaseSearchCV's default would search all iterations).

### `_run_search` (`:258-365`)
- `:259` — `candidate_params = self._generate_candidate_params()` — for
  HalvingGridSearchCV this is the full `ParameterGrid(self.param_grid)` (`:713`).
- `:272` — `n_required_iterations = 1 + floor(log(len(candidate_params), factor))`.
- `:274-282` — if `min_resources == "exhaust"`: `min_resources_ = max(
  min_resources_, max_resources_ // factor**(n_required_iterations-1))`.
- `:289-291` — `n_possible_iterations = 1 + floor(log(max_resources_ //
  min_resources_, factor))`.
- `:293-296` — `n_iterations = n_required_iterations` if `aggressive_elimination`
  else `min(n_possible_iterations, n_required_iterations)`.
- `:310-322` — loop `for itr in range(n_iterations)`: `power = itr` (aggressive
  padding `max(0, itr - n_required + n_possible)`, `:312-317`); `n_resources =
  min(int(factor**power * min_resources_), max_resources_)`; append to
  `n_resources_`; append `len(candidate_params)` to `n_candidates_`.
- `:333-340` — for `resource == "n_samples"`: `cv = _SubsampleMetaSplitter(
  base_cv=cv_orig, fraction=n_resources / n_samples_orig, subsample_test=True,
  random_state=self.random_state)` — a RANDOM fraction subsample (RNG-coupled).
- `:350-357` — `more_results = {"iter": [itr]*n_cand, "n_resources":
  [n_resources]*n_cand}`; `evaluate_candidates(candidate_params, cv,
  more_results=more_results)` — so `cv_results_` records EVERY (iter, candidate).
- `:359-360` — `n_candidates_to_keep = ceil(n_candidates / factor)`;
  `candidate_params = _top_k(results, n_candidates_to_keep, itr)`.
- `:362-365` — record `n_remaining_candidates_`, `n_required_iterations_`,
  `n_possible_iterations_`, `n_iterations_`.

### `_top_k` (`:48-60`)
- `:54-60` — among iteration `itr`'s candidates, `argsort` mean_test_score (NaNs
  rolled to the front), return the LAST `k` (highest scores). `k = ceil(n/factor)`.

## Requirements

R-DEV mental test applied per REQ ("numerical/structural contract" → MATCH;
"missing feature" → NOT-STARTED with a blocker).

- REQ-HALVING-MECHANIC (successive-halving loop SHAPE + keep-count + best_index):
  evaluate the active candidates on a budget, sort by mean CV score, keep the top
  `ceil(n / factor)`, grow the budget, repeat until the max budget or one candidate.
  Mirrors sklearn's `_run_search` loop (`:310-360`) and `_top_k` (`:48-60`).
  **MATCH** (R-DEV-1 structural contract) for the SHAPE and the KEEP-COUNT: sklearn
  `n_candidates_to_keep = ceil(n_candidates / factor)` (`:359`), ferrolearn
  `n.div_ceil(self.factor)` — the SAME ceil arithmetic (verified live, AC-1). The
  best candidate via the shared `CvResults::best_index` (fixed first-on-tie /
  NaN-worst) is the SAME reduction as `grid_search.md` REQ-BESTIDX (now landed).
  SHIPPED (for the shape + keep-count + best_index). The SCHEDULE facets
  (`min_resources_`/`n_resources`/`n_iterations`) are split out to their own REQs.
- REQ-MIN-RESOURCES (min_resources_ formula): sklearn `min_resources_ = n_splits *
  magic_factor(2) * [n_classes if classifier]` with the `exhaust` adjustment
  `max(min_resources_, max_resources_ // factor**(n_required-1))`
  (`:154-167`, `:274-282`). ferrolearn `compute_min_resources` uses
  `ceil(max_res / factor^n_rounds).max(1)` where `n_rounds = compute_n_rounds(
  n_candidates, factor)` — a HEURISTIC unaware of `n_splits`/`n_classes` and of the
  exhaust rule. **DIVERGENT** (architectural — needs the sklearn schedule +
  n_splits/classifier awareness). NOT-STARTED (blocker #1851).
- REQ-N-ITERATIONS (n_iterations / n_resources schedule): sklearn `n_required = 1 +
  floor(log(n_cand, factor))`, `n_possible = 1 + floor(log(max_res // min_res,
  factor))`, `n_iterations = min(n_possible, n_required)` (`:272-296`), and
  `n_resources = int(factor**power * min_res)` per iteration (`:319-321`).
  ferrolearn loops `budget = budget.saturating_mul(factor).min(max_res + 1)` until
  `effective_budget >= max_res` or one candidate (`fit` `:272-333`). Because
  `min_resources_` already diverges, the budget sequence diverges too (live: sklearn
  `[10,30,90]` vs ferrolearn `[4,12,36,90]` for 10 candidates / factor 3 / 90 rows,
  AC-2). **DIVERGENT** (architectural). NOT-STARTED (blocker #1852).
- REQ-CV-RESULTS (cv_results_ records ALL iterations with iter/n_resources):
  sklearn passes `more_results = {"iter": [...], "n_resources": [...]}` to
  `evaluate_candidates` for EVERY iteration (`:350-357`), so `cv_results_` carries
  one row per (iter, candidate) with `iter`/`n_resources` columns, plus
  `_select_best_index` over the last iteration (`:191-212`). ferrolearn records ONLY
  the FINAL round's survivors into `CvResults` (`fit` `:295-307`/`:317-328`) — no
  per-iteration history, no `iter`/`n_resources` columns, and `best_index` runs over
  those final-round rows (NOT a last-iteration filter over a full table). **DIVERGENT**
  (missing-feature). NOT-STARTED (blocker #1853).
- REQ-SUBSAMPLE (resource subsampling: random fraction vs first-rows): sklearn
  `_SubsampleMetaSplitter(fraction=n_resources/n_samples, random_state=...)`
  randomly subsamples a fraction of the data each iteration (`:333-340`,
  RNG-coupled, `subsample_test=True`). ferrolearn slices the FIRST `budget` ROWS
  (`x.slice(s![..budget, ..])`, `y.iter().take(budget)`) in `evaluate_candidates`
  (`:195-197`) and again in the final-round re-evaluation (`:300-301`/`:321-322`).
  **DIVERGENT** (structural: first-rows-vs-random, plus an RNG carve-out for the
  exact subsample). NOT-STARTED (blocker #1854).
- REQ-REFIT (refit + best_estimator_ + predict/score): sklearn `refit=True` DEFAULT
  refits the best params on the full data via BaseSearchCV.fit and delegates
  `predict`/`predict_proba`/`score`/`transform` to `best_estimator_`. ferrolearn
  `HalvingGridSearchCV` is search-only — no refit, no `best_estimator_`, no
  delegating estimator surface (same gap as `grid_search.md` REQ-REFIT).
  **DIVERGENT** (architectural). NOT-STARTED (blocker #1855).
- REQ-ATTRS (halving attributes): sklearn exposes `n_resources_`, `n_candidates_`,
  `n_remaining_candidates_`, `n_iterations_`, `n_possible_iterations_`,
  `n_required_iterations_`, `min_resources_`, `max_resources_` (`:307-308`,
  `:322`, `:325`, `:362-365`, `:154-177`). ferrolearn exposes NONE — only
  `cv_results`/`best_params`/`best_score`. **DIVERGENT** (missing-feature).
  NOT-STARTED (blocker #1856).
- REQ-DEFAULTS (string-defaults + classifier-aware semantics): sklearn defaults
  `factor=3`, `min_resources="exhaust"`, `max_resources="auto"`,
  `resource="n_samples"`, `cv=5`, `aggressive_elimination=False`, `random_state=None`
  (`:94-106`), and `min_resources_` is classifier-aware (`:161-165`). ferrolearn
  defaults `factor=3` (constructor) but has NO `"exhaust"`/`"auto"`/`"smallest"`
  string forms, NO `resource` parameter, NO `random_state`, mandatory explicit
  `cv`/`scoring`, and no classifier awareness. **DIVERGENT** (API surface).
  NOT-STARTED (blocker #1857).
- REQ-X-1 (R-SUBSTRATE ndarray→ferray-core): production code imports
  `use ndarray::{Array1, Array2}`; the destination substrate is `ferray-core`
  (R-SUBSTRATE-1). NOT-STARTED (blocker #1858).
- REQ-X-2 (non-test production consumer): the crate re-export boundary
  (`pub use halving_grid_search::HalvingGridSearchCV in lib.rs`). SHIPPED.

## Acceptance criteria

Each AC is pinnable against a LIVE sklearn 1.5.2 call (R-CHAR-3 — expected values
come from the oracle, never copied from the ferrolearn side). The HalvingGridSearchCV
oracle requires `from sklearn.experimental import enable_halving_search_cv`.

- AC-1 (REQ-HALVING-MECHANIC, keep-count ceil + best candidate): the per-round
  keep-count is `ceil(n_candidates / factor)`, and the search returns the best
  candidate of the last round. Oracle (live sklearn — `n_candidates_` shows the
  ceil keep schedule; ferrolearn `div_ceil` matches):
  ```
  python3 -c "import numpy as np
  from sklearn.experimental import enable_halving_search_cv  # noqa
  from sklearn.model_selection import HalvingGridSearchCV, KFold
  from sklearn.base import BaseEstimator, RegressorMixin
  class Const(BaseEstimator, RegressorMixin):
      def __init__(self,c=0.0): self.c=c
      def fit(self,X,y): self.c_=self.c; return self
      def predict(self,X): return np.full(X.shape[0], self.c_)
  rng=np.random.RandomState(0); X=rng.rand(90,2); y=rng.rand(90)
  gs=HalvingGridSearchCV(Const(), {'c':[float(i) for i in range(5)]}, factor=3,
      cv=KFold(3), random_state=0).fit(X,y)
  import math
  print('n_candidates_:', gs.n_candidates_, 'keep ceil(5/3)=', math.ceil(5/3))"
  # -> n_candidates_: [5, 2] keep ceil(5/3)= 2
  ```
  ferrolearn `n_survive = n.div_ceil(self.factor).max(1)` keeps the SAME count
  (5→2); `test_halving_finds_best_constant` asserts the predict-1.0 constant wins on
  constant `y=1.0`.
- AC-2 (REQ-N-ITERATIONS / REQ-MIN-RESOURCES, schedule DIVERGENCE): the
  `n_resources_` / `min_resources_` schedule differs. Oracle (live sklearn — exhaust
  schedule):
  ```
  python3 -c "import numpy as np
  from sklearn.experimental import enable_halving_search_cv  # noqa
  from sklearn.model_selection import HalvingGridSearchCV, KFold
  from sklearn.base import BaseEstimator, RegressorMixin
  class Const(BaseEstimator, RegressorMixin):
      def __init__(self,c=0.0): self.c=c
      def fit(self,X,y): self.c_=self.c; return self
      def predict(self,X): return np.full(X.shape[0], self.c_)
  rng=np.random.RandomState(0); X=rng.rand(90,2); y=rng.rand(90)
  gs=HalvingGridSearchCV(Const(), {'c':[float(i) for i in range(10)]}, factor=3,
      cv=KFold(3), random_state=0).fit(X,y)
  print('min_resources_:', gs.min_resources_, 'n_resources_:', gs.n_resources_,
        'n_iterations_:', gs.n_iterations_)"
  # -> min_resources_: 10 n_resources_: [10, 30, 90] n_iterations_: 3
  ```
  ferrolearn `compute_min_resources` → `ceil(90 / 3^3) = 4`, schedule `[4,12,36,90]`
  (4 rounds). DIVERGENCE (#1851, #1852).
- AC-3 (REQ-CV-RESULTS, history + columns DIVERGENCE): sklearn `cv_results_` records
  EVERY (iter, candidate) with `iter`/`n_resources` columns. Oracle (live sklearn):
  ```
  python3 -c "import numpy as np
  from sklearn.experimental import enable_halving_search_cv  # noqa
  from sklearn.model_selection import HalvingGridSearchCV, KFold
  from sklearn.base import BaseEstimator, RegressorMixin
  class Const(BaseEstimator, RegressorMixin):
      def __init__(self,c=0.0): self.c=c
      def fit(self,X,y): self.c_=self.c; return self
      def predict(self,X): return np.full(X.shape[0], self.c_)
  rng=np.random.RandomState(0); X=rng.rand(90,2); y=rng.rand(90)
  gs=HalvingGridSearchCV(Const(), {'c':[float(i) for i in range(5)]}, factor=3,
      cv=KFold(3), random_state=0).fit(X,y)
  print('iter col:', gs.cv_results_['iter'].tolist())
  print('n_resources col:', gs.cv_results_['n_resources'].tolist())
  print('n rows:', len(gs.cv_results_['params']))"
  # -> iter col: [0, 0, 0, 0, 0, 1, 1]
  # -> n_resources col: [30, 30, 30, 30, 30, 90, 90]
  # -> n rows: 7
  ```
  ferrolearn `CvResults` records only the 2 final-round survivors, no
  `iter`/`n_resources` columns. DIVERGENCE (#1853).
- AC-4 (REQ-SUBSAMPLE, random vs first-rows DIVERGENCE): sklearn subsamples a RANDOM
  fraction per iteration via `_SubsampleMetaSplitter` (RNG-coupled). Oracle (live
  sklearn — the subsample is a random fraction, not the first rows):
  ```
  python3 -c "import inspect
  from sklearn.model_selection import _search_successive_halving as m
  src=inspect.getsource(m._SubsampleMetaSplitter.split)
  print('uses _approximate_mode/check_random_state:',
        'check_random_state' in inspect.getsource(m._SubsampleMetaSplitter))"
  # -> uses _approximate_mode/check_random_state: True
  ```
  ferrolearn `evaluate_candidates` slices `x.slice(s![..budget, ..])` — the FIRST
  `budget` rows, deterministic, no RNG. DIVERGENCE (#1854).
- AC-5 (REQ-REFIT, refit/predict DIVERGENCE): sklearn `refit=True` (default) exposes
  `best_estimator_` and a delegating `predict`. Oracle (live sklearn):
  ```
  python3 -c "import numpy as np
  from sklearn.experimental import enable_halving_search_cv  # noqa
  from sklearn.model_selection import HalvingGridSearchCV, KFold
  from sklearn.base import BaseEstimator, RegressorMixin
  class Const(BaseEstimator, RegressorMixin):
      def __init__(self,c=0.0): self.c=c
      def fit(self,X,y): self.c_=self.c; return self
      def predict(self,X): return np.full(X.shape[0], self.c_)
  rng=np.random.RandomState(0); X=rng.rand(90,2); y=np.full(90,4.0)
  gs=HalvingGridSearchCV(Const(), {'c':[4.0,6.0]}, factor=2, cv=KFold(3),
      random_state=0).fit(X,y)
  print('has best_estimator_:', hasattr(gs,'best_estimator_'))"
  # -> has best_estimator_: True
  ```
  ferrolearn has no `best_estimator_`/`predict`/`score`. DIVERGENCE (#1855).
- AC-6 (REQ-ATTRS, missing attributes DIVERGENCE): sklearn exposes the eight halving
  attributes. Oracle (live sklearn — attribute presence):
  ```
  python3 -c "import numpy as np
  from sklearn.experimental import enable_halving_search_cv  # noqa
  from sklearn.model_selection import HalvingGridSearchCV, KFold
  from sklearn.base import BaseEstimator, RegressorMixin
  class Const(BaseEstimator, RegressorMixin):
      def __init__(self,c=0.0): self.c=c
      def fit(self,X,y): self.c_=self.c; return self
      def predict(self,X): return np.full(X.shape[0], self.c_)
  rng=np.random.RandomState(0); X=rng.rand(90,2); y=rng.rand(90)
  gs=HalvingGridSearchCV(Const(), {'c':[float(i) for i in range(5)]}, factor=3,
      cv=KFold(3), random_state=0).fit(X,y)
  print([a for a in ('n_resources_','n_candidates_','n_remaining_candidates_',
        'n_iterations_','n_possible_iterations_','n_required_iterations_',
        'min_resources_','max_resources_') if hasattr(gs,a)])"
  # -> all eight present
  ```
  ferrolearn exposes none. DIVERGENCE (#1856).
- AC-7 (REQ-DEFAULTS, string-defaults DIVERGENCE): sklearn defaults
  `min_resources="exhaust"`, `max_resources="auto"`, `resource="n_samples"`, `cv=5`,
  and is classifier-aware. Oracle (live sklearn — the defaults resolve):
  ```
  python3 -c "from sklearn.experimental import enable_halving_search_cv  # noqa
  from sklearn.model_selection import HalvingGridSearchCV
  from sklearn.base import BaseEstimator
  gs=HalvingGridSearchCV(BaseEstimator(), {'x':[1]})
  print(gs.min_resources, gs.max_resources, gs.resource, gs.cv, gs.factor)"
  # -> exhaust auto n_samples 5 3
  ```
  ferrolearn `new` requires explicit `cv`/`scoring`, has no `resource`/`random_state`
  and no string-default forms (only `factor=3` default). DIVERGENCE (#1857).
- AC-X-1 (REQ-X-1, substrate): `grep -n "ndarray" halving_grid_search.rs` shows
  production `use ndarray::{Array1, Array2}` — wrong substrate, migration to
  `ferray-core` owed (#1858).

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-HALVING-MECHANIC (loop shape + ceil keep-count + best_index) | SHIPPED | impl `pub fn fit in halving_grid_search.rs`: `loop { let scored = self.evaluate_candidates(&active, effective_budget, x, y)?; let n_survive = ... n.div_ceil(self.factor).max(1); ... active = scored.into_iter().take(n_survive)...; budget = budget.saturating_mul(self.factor).min(max_res + 1); }` — evaluate active candidates on a budget, keep the top `ceil(n/factor)`, grow the budget, repeat. The SHAPE mirrors sklearn `_run_search` loop (`sklearn/model_selection/_search_successive_halving.py:310-360`); the KEEP-COUNT `div_ceil` MATCHES `_top_k`'s `n_candidates_to_keep = ceil(n_candidates / factor)` (`:359`, verified live AC-1: `n_candidates_ = [5,2]`, `ceil(5/3)=2`). Best candidate via shared `pub fn best_index in grid_search.rs` (fixed first-on-tie / NaN-worst), the SAME landed reduction as `grid_search.md` REQ-BESTIDX. Tests: `test_halving_finds_best_constant`, `test_halving_basic_runs`, `test_compute_n_rounds_basic`. Non-test consumer: REQ-X-2 (re-export). SCHEDULE facets split to REQ-MIN-RESOURCES/REQ-N-ITERATIONS. |
| REQ-MIN-RESOURCES (min_resources_ formula) | NOT-STARTED | open prereq blocker #1851. sklearn `min_resources_ = n_splits * 2 * [n_classes if classifier]` + exhaust adjustment `max(min_resources_, max_resources_ // factor**(n_required-1))` (`sklearn/model_selection/_search_successive_halving.py:154-167`, `:274-282`). impl `fn compute_min_resources in halving_grid_search.rs`: `let min_res = (max_res as f64 / factor_pow).ceil() as usize; min_res.max(1)` with `factor_pow = factor^compute_n_rounds(n_candidates, factor)` — a heuristic unaware of `n_splits`/`n_classes`/exhaust. Oracle (AC-2): sklearn `min_resources_ = 10` (n_splits=3·magic=2·exhaust→max(6, 90//9=10)); ferrolearn `ceil(90/3^3)=4`. Architectural divergence. |
| REQ-N-ITERATIONS (n_iterations / n_resources schedule) | NOT-STARTED | open prereq blocker #1852. sklearn `n_iterations = min(n_possible, n_required)` and per-iter `n_resources = int(factor**power * min_resources_)` (`sklearn/.../_search_successive_halving.py:272-321`). impl `pub fn fit in halving_grid_search.rs`: `budget = budget.saturating_mul(self.factor).min(max_res + 1)` looped until `effective_budget >= max_res || scored.len() <= 1` — no `n_required`/`n_possible`, and the budget base diverges via REQ-MIN-RESOURCES. Oracle (AC-2): sklearn `n_resources_ = [10,30,90]` (3 iters); ferrolearn `[4,12,36,90]` (4 rounds). Architectural divergence. |
| REQ-CV-RESULTS (cv_results_ records ALL iterations with iter/n_resources) | NOT-STARTED | open prereq blocker #1853. sklearn passes `more_results = {"iter": [...], "n_resources": [...]}` to every `evaluate_candidates` call (`sklearn/.../_search_successive_halving.py:350-357`), recording one row per (iter, candidate); `_select_best_index` then filters to the last iteration (`:191-212`). impl `pub fn fit in halving_grid_search.rs` pushes ONLY the final round's survivors into `CvResults` (`results.push(params, fold_scores)` inside the `is_final`/single-candidate branches) — no per-iteration history, no `iter`/`n_resources` columns; `best_index` runs over those final rows directly. Oracle (AC-3): sklearn `iter col = [0,0,0,0,0,1,1]`, 7 rows; ferrolearn keeps 2. Missing-feature divergence. |
| REQ-SUBSAMPLE (random fraction vs first-rows) | NOT-STARTED | open prereq blocker #1854. sklearn `_SubsampleMetaSplitter(fraction=n_resources/n_samples_orig, random_state=...)` RANDOM-subsamples a fraction each iteration (`sklearn/.../_search_successive_halving.py:333-340`, RNG-coupled). impl `fn evaluate_candidates in halving_grid_search.rs`: `let x_sub = x.slice(ndarray::s![..budget, ..]).to_owned(); let y_sub = y.iter().take(budget).copied().collect();` — the FIRST `budget` rows, deterministic, no RNG (and again in `fit`'s final-round re-eval). Oracle (AC-4): `_SubsampleMetaSplitter` uses `check_random_state` (random subsample). Structural + RNG-carve-out divergence. |
| REQ-REFIT (refit + best_estimator_ + predict/score) | NOT-STARTED | open prereq blocker #1855. sklearn `refit=True` DEFAULT (`sklearn/.../_search_successive_halving.py:96`) refits the best params on the full data (via `BaseSearchCV.fit`) and delegates `predict`/`predict_proba`/`score`/`transform` to `best_estimator_`. impl `pub struct HalvingGridSearchCV in halving_grid_search.rs` is search-only — fields `pipeline_factory`/`param_grid`/`cv`/`scoring`/`factor`/`min_resources`/`max_resources`/`aggressive_elimination`/`results`, accessors `cv_results`/`best_params`/`best_score` only; NO refit, NO `best_estimator_`, NO `predict`/`score`. Oracle (AC-5): `hasattr(gs,'best_estimator_') == True`. Architectural (same gap as `grid_search.md` REQ-REFIT). |
| REQ-ATTRS (n_resources_/n_candidates_/n_remaining_candidates_/n_iterations_/n_possible_iterations_/n_required_iterations_/min_resources_/max_resources_) | NOT-STARTED | open prereq blocker #1856. sklearn records all eight (`sklearn/.../_search_successive_halving.py:307-308`, `:322`, `:325`, `:362-365`, `:154-177`). impl `pub struct HalvingGridSearchCV in halving_grid_search.rs` exposes NONE — only `cv_results`/`best_params`/`best_score`; `compute_min_resources`/budget are local `fit` variables, never surfaced. Oracle (AC-6): all eight present on the fitted sklearn estimator. Missing-feature divergence. |
| REQ-DEFAULTS (string-defaults + classifier-aware + resource/random_state) | NOT-STARTED | open prereq blocker #1857. sklearn `min_resources="exhaust"`, `max_resources="auto"`, `resource="n_samples"`, `cv=5`, `random_state=None`, `factor=3` (`sklearn/.../_search_successive_halving.py:94-106`), with classifier-aware `min_resources_` (`:161-165`). impl `pub fn new in halving_grid_search.rs` sets `factor: 3` but requires explicit `cv: Box<dyn CrossValidator>` + `scoring: fn(...)`, has `min_resources: Option<usize>` / `max_resources: Option<usize>` (no `"exhaust"`/`"auto"`/`"smallest"` strings), no `resource`, no `random_state`, no classifier awareness. Oracle (AC-7): sklearn defaults resolve to `exhaust auto n_samples 5 3`. API-surface divergence. |
| REQ-X-1 (R-SUBSTRATE) | NOT-STARTED | open prereq blocker #1858. Production code in `halving_grid_search.rs` imports `use ndarray::{Array1, Array2}`; `fit`/`evaluate_candidates` take `x: &Array2<f64>`/`y: &Array1<f64>` and use `ndarray::s!` slicing. Per R-SUBSTRATE-1 the destination is `ferray-core`; `ndarray` is the wrong substrate (R-SUBSTRATE-2). |
| REQ-X-2 (non-test production consumer) | SHIPPED | `pub struct HalvingGridSearchCV` is re-exported at `pub use halving_grid_search::HalvingGridSearchCV in lib.rs` — the boundary public API per S5/R-DEFER-1 grandfathering (boundary estimator types ARE the public API). It additionally consumes the shared `CvResults` from `grid_search.rs` in non-test production (`use crate::grid_search::CvResults`; `CvResults::new`/`push`/`best_index`). No internal caller and no `ferrolearn-python` binding yet — the re-export IS its production consumer. |

## Architecture

ferrolearn implements `HalvingGridSearchCV<'a>` as a search-only struct that REUSES
the shared `CvResults` from `grid_search.rs` (`halving_grid_search.rs`). The
closure base (R-DEV-7) matches `GridSearchCV`: `pipeline_factory: Box<dyn
Fn(&ParamSet) -> Pipeline + 'a>`, a pre-expanded `param_grid: Vec<ParamSet>`, a
`cv: Box<dyn CrossValidator>`, a `scoring: fn(&Array1<f64>, &Array1<f64>) ->
Result<f64, FerroError>`, plus builder fields `factor` (default 3),
`min_resources`/`max_resources: Option<usize>`, `aggressive_elimination: bool`, and
`results: Option<CvResults>`.

The SHIPPED surface is narrow and structural (REQ-HALVING-MECHANIC). `fit` validates
the empty grid, `factor < 2`, and the `x`/`y` shape, then computes `max_res =
max_resources.unwrap_or(n_samples).min(n_samples)` and `min_res =
compute_min_resources(...)`, and runs the successive-halving loop: evaluate the
active candidates on `effective_budget` via `evaluate_candidates` (which calls
`cross_val_score` per candidate and sorts descending by mean), keep the top
`n.div_ceil(self.factor).max(1)`, then `budget *= factor`. This LOOP SHAPE mirrors
sklearn's `_run_search` loop (`:310-360`), and the keep-count `div_ceil` MATCHES
`_top_k`'s `ceil(n_candidates / factor)` (`:359`) — verified live (AC-1). The best
candidate uses the shared `CvResults::best_index` (the fixed first-on-tie /
NaN-worst reduction landed for `grid_search.md` REQ-BESTIDX).

A NON-divergence worth recording, because it is easy to assume otherwise: the
elimination KEEP-COUNT arithmetic does NOT diverge. sklearn keeps `ceil(n / factor)`
(`_top_k`, `:359`), and ferrolearn's `div_ceil` keeps the same count (5→2, 10→4
live). There is therefore NO deterministic single-spot fixable divergence in this
unit. The divergences below are ALL architectural/missing-feature; even the matching
keep-count does NOT make the unit match sklearn end-to-end, because the SCHEDULE
(`min_resources_` and the `n_resources` sequence) diverges upstream of it.

Where ferrolearn DIVERGES (all NOT-STARTED, each with a filed blocker):

- **min_resources_ schedule** (REQ-MIN-RESOURCES, #1851): `compute_min_resources`
  uses the heuristic `ceil(max_res / factor^n_rounds).max(1)` where `n_rounds =
  compute_n_rounds(n_candidates, factor)`; sklearn uses `n_splits * 2 * [n_classes]`
  with the exhaust adjustment (`:154-167`, `:274-282`). Live: sklearn 10 vs
  ferrolearn 4 (90 rows, 10 candidates, factor 3).
- **n_iterations / n_resources schedule** (REQ-N-ITERATIONS, #1852): ferrolearn loops
  `budget *= factor` until `>= max_res` or one candidate; sklearn computes
  `min(n_possible, n_required)` iterations with `n_resources = int(factor**power *
  min_resources_)` (`:272-321`). Live: `[10,30,90]` vs `[4,12,36,90]`.
- **cv_results_ richness / history** (REQ-CV-RESULTS, #1853): ferrolearn records ONLY
  the final round's survivors; sklearn records every (iter, candidate) with
  `iter`/`n_resources` columns and selects the best from the LAST iteration
  (`:191-212`, `:350-357`). Live: 7 rows / `iter=[0,0,0,0,0,1,1]` vs 2 rows.
- **resource subsampling** (REQ-SUBSAMPLE, #1854): ferrolearn slices the FIRST
  `budget` rows (`x.slice(s![..budget,..])`); sklearn random-subsamples a fraction
  via `_SubsampleMetaSplitter` (`:333-340`, RNG-coupled).
- **refit / best_estimator_ / search-as-estimator** (REQ-REFIT, #1855): absent
  (same gap as `grid_search.md`).
- **halving attributes** (REQ-ATTRS, #1856): the eight `n_resources_`/… attributes
  are absent.
- **string-defaults / classifier-aware / resource / random_state** (REQ-DEFAULTS,
  #1857): no `"exhaust"`/`"auto"`/`"smallest"`, no `resource`, no `random_state`, no
  classifier awareness; `cv`/`scoring` are mandatory.
- **substrate** (REQ-X-1, #1858): `ndarray::{Array1, Array2}` must migrate to
  `ferray-core` (R-SUBSTRATE-1/2).

`HalvingGridSearchCV` reaches production via the `lib.rs` re-export (REQ-X-2) and
consumes the shared `CvResults` from `grid_search.rs` in non-test production.

## Verification

Commands establishing the SHIPPED claims (baseline
`5ce2c995d12609b0fe0215585b395b6beae6f8a6`):

- `cargo test -p ferrolearn-model-sel --lib halving_grid_search` → 15 passed, 0
  failed (`halving_grid_search::tests::{test_compute_n_rounds_basic,
  test_halving_basic_runs, test_halving_finds_best_constant,
  test_halving_best_score_near_zero_for_perfect, test_halving_empty_grid_returns_error,
  test_halving_factor_less_than_2_returns_error, test_halving_returns_none_before_fit,
  test_halving_shape_mismatch_returns_error, test_halving_single_candidate,
  test_halving_cv_results_non_empty_after_fit, test_halving_custom_max_resources,
  test_halving_factor_2, test_halving_aggressive_elimination,
  test_halving_best_score_is_finite, test_halving_cv_results_all_scores_have_fold_scores}`).
- REQ-HALVING-MECHANIC keep-count oracle (live sklearn 1.5.2 — ceil keep schedule):
  ```
  python3 -c "import numpy as np, math
  from sklearn.experimental import enable_halving_search_cv  # noqa
  from sklearn.model_selection import HalvingGridSearchCV, KFold
  from sklearn.base import BaseEstimator, RegressorMixin
  class Const(BaseEstimator, RegressorMixin):
      def __init__(self,c=0.0): self.c=c
      def fit(self,X,y): self.c_=self.c; return self
      def predict(self,X): return np.full(X.shape[0], self.c_)
  rng=np.random.RandomState(0); X=rng.rand(90,2); y=rng.rand(90)
  gs=HalvingGridSearchCV(Const(), {'c':[float(i) for i in range(5)]}, factor=3,
      cv=KFold(3), random_state=0).fit(X,y)
  print('n_candidates_:', gs.n_candidates_, 'ceil(5/3)=', math.ceil(5/3))"
  # -> n_candidates_: [5, 2] ceil(5/3)= 2   (ferrolearn div_ceil keeps 2)
  ```
- REQ-MIN-RESOURCES / REQ-N-ITERATIONS schedule DIVERGENCE oracle (live sklearn):
  ```
  python3 -c "import numpy as np
  from sklearn.experimental import enable_halving_search_cv  # noqa
  from sklearn.model_selection import HalvingGridSearchCV, KFold
  from sklearn.base import BaseEstimator, RegressorMixin
  class Const(BaseEstimator, RegressorMixin):
      def __init__(self,c=0.0): self.c=c
      def fit(self,X,y): self.c_=self.c; return self
      def predict(self,X): return np.full(X.shape[0], self.c_)
  rng=np.random.RandomState(0); X=rng.rand(90,2); y=rng.rand(90)
  gs=HalvingGridSearchCV(Const(), {'c':[float(i) for i in range(10)]}, factor=3,
      cv=KFold(3), random_state=0).fit(X,y)
  print('min_resources_:', gs.min_resources_, 'n_resources_:', gs.n_resources_)"
  # -> min_resources_: 10 n_resources_: [10, 30, 90]
  #    (ferrolearn ceil(90/3^3)=4, schedule [4,12,36,90])  (#1851, #1852)
  ```
- REQ-CV-RESULTS DIVERGENCE oracle: `gs.cv_results_['iter'] == [0,0,0,0,0,1,1]`,
  `gs.cv_results_['n_resources'] == [30,30,30,30,30,90,90]`, 7 rows for 5 candidates
  (factor 3) — ferrolearn keeps only the 2 final-round survivors, no `iter`/
  `n_resources` columns (#1853).
- REQ-SUBSAMPLE DIVERGENCE oracle: `_SubsampleMetaSplitter` in
  `sklearn/model_selection/_search_successive_halving.py` uses `check_random_state`
  (random fraction subsample) — ferrolearn slices the first `budget` rows (#1854).
- REQ-REFIT DIVERGENCE oracle: `hasattr(gs,'best_estimator_') == True` (refit=True
  default) — ferrolearn has no `best_estimator_`/`predict`/`score` (#1855).
- REQ-ATTRS DIVERGENCE oracle: the fitted sklearn estimator has
  `n_resources_`/`n_candidates_`/`n_remaining_candidates_`/`n_iterations_`/
  `n_possible_iterations_`/`n_required_iterations_`/`min_resources_`/`max_resources_`
  — ferrolearn exposes none (#1856).
- REQ-DEFAULTS DIVERGENCE oracle: `HalvingGridSearchCV(est, grid)` defaults are
  `min_resources='exhaust'`, `max_resources='auto'`, `resource='n_samples'`, `cv=5`,
  `factor=3` — ferrolearn requires explicit `cv`/`scoring`, no string defaults,
  no `resource`/`random_state` (#1857).
- REQ-X-1 substrate: `grep -n "ndarray" halving_grid_search.rs` shows production
  `use ndarray::{Array1, Array2}` — wrong substrate, migration owed (#1858).
- REQ-X-2 consumer: `grep -rn "HalvingGridSearchCV" ferrolearn-model-sel/src/lib.rs`
  shows `pub use halving_grid_search::HalvingGridSearchCV`; `grep -n "CvResults"
  halving_grid_search.rs` shows the shared-type consumption.

SHIPPED: REQ-HALVING-MECHANIC (loop shape + ceil keep-count + best_index),
REQ-X-2 (consumer — re-export + shared `CvResults`).
NOT-STARTED: REQ-MIN-RESOURCES (#1851), REQ-N-ITERATIONS (#1852),
REQ-CV-RESULTS (#1853), REQ-SUBSAMPLE (#1854), REQ-REFIT (#1855), REQ-ATTRS (#1856),
REQ-DEFAULTS (#1857), REQ-X-1 (ferray substrate, #1858). Per R-DEFER-2 every REQ is
binary SHIPPED/NOT-STARTED.

There is NO deterministic single-spot fixable divergence in this unit: the
elimination keep-count `ceil(n/factor)` already MATCHES sklearn (`_top_k`, `:359`).
Every NOT-STARTED REQ is architectural/missing-feature — the critic pins each as a
FAILING `#[test]` against the live oracle, but no single-line fix closes any of them
(REQ-MIN-RESOURCES/REQ-N-ITERATIONS require the full sklearn schedule;
REQ-CV-RESULTS the per-iteration history; REQ-SUBSAMPLE the RNG splitter; the rest
are absent surface).

Least-confident SHIPPED claim: REQ-HALVING-MECHANIC. The keep-count and loop SHAPE
match, and `best_index` is the fixed first-on-tie reduction — but the claim is
deliberately narrow: it asserts ONLY the structural mechanic (eliminate by ceil,
grow the budget, best-of-final), NOT numeric parity. Because `min_resources_` and
the `n_resources` sequence diverge (REQ-MIN-RESOURCES/REQ-N-ITERATIONS), ferrolearn
evaluates candidates on DIFFERENT budgets than sklearn at every iteration, so the
selected best candidate can differ from sklearn's on non-degenerate data even though
the mechanic itself is faithful. The SHIPPED surface is the SHAPE + keep-count +
best-of-recorded, not end-to-end sklearn equivalence.
