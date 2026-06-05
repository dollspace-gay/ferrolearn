# halving_random_search

<!--
tier: 3-component
status: draft
baseline-commit: fedc1ed468cac887cd10660eceaa0c7bc92f2be9
upstream-paths:
  - sklearn/model_selection/_search_successive_halving.py   # BaseSuccessiveHalving (:63), __init__ (:90-124), _check_input_parameters/min_resources_ (:126-189), _select_best_index (:191-212), _run_search (:258-365), _top_k (:48-60), HalvingRandomSearchCV (:717), __init__ (:1030-1067), _generate_candidate_params (:1069-1079)
-->

## Summary

`ferrolearn-model-sel/src/halving_random_search.rs` mirrors scikit-learn's
`HalvingRandomSearchCV` (`sklearn/model_selection/_search_successive_halving.py:717`,
subclass of `BaseSuccessiveHalving` `:63`, itself a `BaseSearchCV` subclass) —
successive-halving hyperparameter search where the candidates are sampled at
random from user-supplied parameter distributions, then run through the same
elimination tournament as `HalvingGridSearchCV`.

This is the SIBLING of `halving_grid_search.rs` (see
`.design/model-sel/halving_grid_search.md`). The halving LOOP is IDENTICAL between
the two — same `BaseSuccessiveHalving._run_search` schedule (`:258-365`), same
`_top_k` keep-count `ceil(n_candidates / factor)` (`:359`), same first-rows
subsampling, same final-round-only `cv_results`, same shared `CvResults::best_index`
reduction. The ONLY structural difference is candidate generation: sklearn's
`_generate_candidate_params` (`:1069-1079`) uses `ParameterSampler` to RANDOMLY
sample, whereas `HalvingGridSearchCV` enumerates a `ParameterGrid`. Accordingly,
all SHARED facets are cross-referenced to `halving_grid_search.md` rather than
re-analysed, and only the random-sampling facet is analysed fresh here.

ferrolearn exposes `pub struct HalvingRandomSearchCV<'a>` with the closure-base
shape (R-DEV-7): a `pipeline_factory: Box<dyn Fn(&ParamSet) -> Pipeline + 'a>`, a
`param_distributions: Vec<(String, Box<dyn Distribution>)>`, an `n_candidates:
usize`, a `cv: Box<dyn CrossValidator>`, a `scoring: fn(&Array1<f64>, &Array1<f64>)
-> Result<f64, FerroError>`, a `random_state: Option<u64>`, and builder fields
`factor`/`min_resources`/`max_resources`. It REUSES the shared `CvResults` type from
`grid_search.rs`.

This unit is RIGOROUSLY honest about being an APPROXIMATE, heuristic
successive-halving with an RNG carve-out. SHIPPED is narrow: the LOOP SHAPE +
keep-count (`fn fit`/`evaluate_candidates`, IDENTICAL to the sibling — `div_ceil`
MATCHES `_top_k`'s ceil), the random-sampling COUNT + seed-determinism-across-runs
(`fn sample_params` + `SmallRng`), and the `lib.rs` re-export consumer. Everything
else diverges: `min_resources_`/`n_resources`/`n_iterations` schedule,
`cv_results_` richness, random-fraction subsampling, refit/`best_estimator_`, the
eight halving attributes, the string-default constructor surface (`n_candidates=
"exhaust"` / `min_resources="smallest"` defaults), the EXACT sampled candidates
(SmallRng vs numpy `RandomState` `ParameterSampler` draw order), and the ferray
substrate. Each is NOT-STARTED with a filed blocker.

**Honest note on the two NON-divergences.** (1) The keep-count does NOT differ:
sklearn `_top_k` keeps `ceil(n_candidates / factor)` (`:359`), ferrolearn's
`div_ceil` keeps the same count — verified live, AC-1 (`n_candidates_ = [9,3,1]`,
`ceil(9/3)=3`). (2) The number of candidates SAMPLED (when `n_candidates` is an
explicit int) is just that int on BOTH sides. There is therefore NO deterministic
single-spot fixable divergence in this unit (same audit conclusion as
`halving_grid_search.md`); every divergence below is architectural / missing-feature
/ RNG-coupled. Even the matching keep-count does NOT make the unit match sklearn
end-to-end, because the SCHEDULE (`min_resources_` / `n_resources`) and the EXACT
sampled candidates diverge.

## Upstream reference (scikit-learn 1.5.2, commit 156ef14)

### Shared `BaseSuccessiveHalving` (cross-reference `halving_grid_search.md`)
The `__init__` defaults (`:90-124`), `_check_input_parameters` / `min_resources_`
formula (`:126-189`: `n_splits * magic_factor(2) * [n_classes if classifier]`,
`max_resources_="auto"→_num_samples`), `_select_best_index` over the last iteration
(`:191-212`), the `_run_search` schedule (`:258-365`: `n_required_iterations = 1 +
floor(log(n_cand, factor))`, `n_possible_iterations`, `n_iterations = min(...)`,
`n_resources = int(factor**power * min_resources_)`, `_SubsampleMetaSplitter`
random-fraction subsample `:333-340`, per-iteration `more_results={"iter",
"n_resources"}` `:350-357`, keep `n_candidates_to_keep = ceil(n_candidates /
factor)` `:359`), and `_top_k` (`:48-60`) are IDENTICAL to what
`halving_grid_search.md` documents. They are cross-referenced, not re-quoted.

### `HalvingRandomSearchCV.__init__` (`sklearn/model_selection/_search_successive_halving.py:1030-1067`)
- `:1035-1048` — keyword-only defaults: `n_candidates="exhaust"`, `factor=3`,
  `resource="n_samples"`, `max_resources="auto"`, `min_resources="smallest"`
  (NOTE: HalvingRandomSearchCV defaults `min_resources="smallest"`, NOT `"exhaust"`
  — verified at source; HalvingGridSearchCV defaults `"exhaust"`),
  `aggressive_elimination=False`, `cv=5`, `scoring=None`, `refit=True`,
  `random_state=None`, `n_jobs=None`, `verbose=0`.
- `:1066-1067` — stores `self.param_distributions` and `self.n_candidates`.
- `:1021-1028` — `_parameter_constraints`: `n_candidates` ∈
  `{Interval(Integral, 0, None, closed="neither"), StrOptions({"exhaust"})}`.

### `_check_input_parameters` — `n_candidates`/`min_resources` "exhaust" mutual-exclusion (`:145-152`)
- `:145-152` — if `isinstance(self, HalvingRandomSearchCV)` and
  `self.min_resources == self.n_candidates == "exhaust"`: raise `ValueError`
  ("n_candidates and min_resources cannot be both set to 'exhaust'.").

### `_generate_candidate_params` (`:1069-1079`)
- `:1070-1074` — `n_candidates_first_iter = self.n_candidates`; if it equals
  `"exhaust"`: `n_candidates_first_iter = self.max_resources_ // self.min_resources_`
  (the number of candidates that exhausts the resources).
- `:1075-1079` — `ParameterSampler(self.param_distributions,
  n_candidates_first_iter, random_state=self.random_state)` — RANDOMLY samples that
  many parameter sets (each via the distributions' `rvs`, with list values sampled
  uniformly), draw order over the SORTED parameter keys, seeded by
  `self.random_state` (numpy `RandomState`).

## Requirements

R-DEV mental test applied per REQ ("numerical/structural contract" → MATCH;
"missing feature" / "RNG-coupled exact value" → NOT-STARTED with a blocker).

- REQ-HALVING-MECHANIC (successive-halving loop SHAPE + keep-count + best_index):
  evaluate the active candidates on a budget, sort by mean CV score, keep the top
  `ceil(n / factor)`, grow the budget, repeat until the max budget or one candidate;
  best candidate via the shared `CvResults::best_index`. IDENTICAL to
  `halving_grid_search.md` REQ-HALVING-MECHANIC. Mirrors sklearn's `_run_search` loop
  (`:310-360`) and `_top_k` (`:48-60`). **MATCH** (R-DEV-1 structural contract) for
  the SHAPE and the KEEP-COUNT: sklearn `n_candidates_to_keep = ceil(n_candidates /
  factor)` (`:359`), ferrolearn `n.div_ceil(self.factor)` — the SAME ceil arithmetic
  (verified live, AC-1: `n_candidates_ = [9,3,1]`, `ceil(9/3)=3`). SHIPPED (for the
  shape + keep-count + best_index). The SCHEDULE facets are split to their own REQs.
- REQ-RANDOM-SAMPLING (sample `n_candidates` ParamSets from the distributions,
  up-front, seed-reproducible across runs): ferrolearn `fn sample_params` draws one
  value per distribution (insertion order) into a `ParamSet`, and `fn fit` builds
  `all_params = (0..n_candidates).map(sample_params)` up-front from a seeded
  `SmallRng`. Mirrors sklearn `_generate_candidate_params` →
  `ParameterSampler(self.param_distributions, n_candidates_first_iter,
  random_state=...)` (`:1075-1079`). **MATCH** (R-DEV-1) for the COUNT (`n_candidates`
  sets sampled up-front) and seed-determinism-across-runs (same seed ⇒ same candidates
  ⇒ same best score, AC-2). SHIPPED for the count + reproducibility. The EXACT sampled
  candidates (SmallRng vs numpy `RandomState`, sorted-key draw order) are carved out
  to REQ-RNG-EXACT. The `"exhaust"` resolution of the count is carved out to
  REQ-N-CANDIDATES-EXHAUST.
- REQ-N-CANDIDATES-EXHAUST (`n_candidates="exhaust"` default ⇒
  `max_resources_ // min_resources_`): sklearn defaults `n_candidates="exhaust"`
  (`:1035`) and `_generate_candidate_params` resolves it to `max_resources_ //
  min_resources_` (`:1074`), with the `n_candidates==min_resources=="exhaust"`
  mutual-exclusion check (`:145-152`). ferrolearn `n_candidates: usize` is a mandatory
  positive integer (constructor arg, validated `== 0`); there is NO `"exhaust"` string
  form and NO auto-resolution. **DIVERGENT** (API surface + missing auto-count).
  NOT-STARTED (blocker #1860).
- REQ-RNG-EXACT (exact sampled candidates / draw order): sklearn `ParameterSampler`
  draws via numpy `RandomState(self.random_state)` over the SORTED parameter keys
  (`:1075-1079`). ferrolearn draws via `SmallRng` (`seed_from_u64`/`from_os_rng`) in
  `param_distributions` INSERTION order (`fn sample_params`). Different PRNG + different
  draw order ⇒ the exact sampled candidates differ even for the same integer seed.
  Per R-DEFER-3 this is an explicit RNG carve-out (no failing test pins exact-candidate
  parity). **DIVERGENT** (RNG-coupled). NOT-STARTED (blocker #1861).
- REQ-MIN-RESOURCES (min_resources_ formula): sklearn `min_resources_ = n_splits *
  magic_factor(2) * [n_classes if classifier]` (`:154-167`; HalvingRandomSearchCV
  default `min_resources="smallest"`, `:1039`), so `"smallest"` resolves directly to
  that heuristic (no exhaust adjustment by default). ferrolearn
  `fn compute_min_resources` uses `ceil(max_res / factor^n_rounds).max(1)` where
  `n_rounds = compute_n_rounds(n_candidates, factor)` — unaware of `n_splits`/
  `n_classes`. **DIVERGENT** (architectural). NOT-STARTED (blocker #1862). Live (AC-3,
  9 candidates / factor 3 / 90 rows / KFold(3)): sklearn `min_resources_ = 6`
  (n_splits=3·magic=2); ferrolearn `ceil(90 / 3^2) = 10`.
- REQ-N-ITERATIONS (n_iterations / n_resources schedule): sklearn `n_required = 1 +
  floor(log(n_cand, factor))`, `n_possible = 1 + floor(log(max_res // min_res,
  factor))`, `n_iterations = min(n_possible, n_required)` (`:272-296`), and
  `n_resources = int(factor**power * min_res)` per iteration (`:319-321`). ferrolearn
  loops `budget = budget.saturating_mul(factor).min(max_res + 1)` until
  `effective_budget >= max_res` or one candidate (`fn fit`). Because `min_resources_`
  already diverges, the budget sequence diverges too (live AC-3: sklearn `[6,18,54]`
  vs ferrolearn `[10,30,90]` for 9 candidates / factor 3 / 90 rows). **DIVERGENT**
  (architectural). NOT-STARTED (blocker #1863).
- REQ-CV-RESULTS (cv_results_ records ALL iterations with iter/n_resources): sklearn
  passes `more_results = {"iter": [...], "n_resources": [...]}` to every
  `evaluate_candidates` call (`:350-357`), so `cv_results_` carries one row per (iter,
  candidate) with `iter`/`n_resources` columns, plus `_select_best_index` over the last
  iteration (`:191-212`). ferrolearn records ONLY the FINAL round's survivors into
  `CvResults` (`fn fit`) — no per-iteration history, no `iter`/`n_resources` columns;
  `best_index` runs over those final rows directly. **DIVERGENT** (missing-feature).
  NOT-STARTED (blocker #1864). Live (AC-4): sklearn 13 rows, `iter=[0×9,1×3,2×1]`;
  ferrolearn keeps only the final-round survivors.
- REQ-SUBSAMPLE (resource subsampling: random fraction vs first-rows): sklearn
  `_SubsampleMetaSplitter(fraction=n_resources/n_samples_orig, random_state=...)`
  RANDOM-subsamples a fraction each iteration (`:333-340`, RNG-coupled,
  `subsample_test=True`). ferrolearn slices the FIRST `budget` ROWS
  (`x.slice(s![..budget, ..])`, `y.iter().take(budget)`) in `fn evaluate_candidates`
  and again in the final-round re-evaluation. **DIVERGENT** (structural +
  RNG-carve-out). NOT-STARTED (blocker #1865).
- REQ-REFIT (refit + best_estimator_ + predict/score): sklearn `refit=True` DEFAULT
  (`:1043`) refits the best params on the full data via `BaseSearchCV.fit` and
  delegates `predict`/`predict_proba`/`score`/`transform` to `best_estimator_`.
  ferrolearn `HalvingRandomSearchCV` is search-only — no refit, no `best_estimator_`,
  no delegating estimator surface. **DIVERGENT** (architectural). NOT-STARTED
  (blocker #1866). Live (AC-5): `hasattr(gs,'best_estimator_') == True`.
- REQ-ATTRS (halving attributes): sklearn exposes `n_resources_`, `n_candidates_`,
  `n_remaining_candidates_`, `n_iterations_`, `n_possible_iterations_`,
  `n_required_iterations_`, `min_resources_`, `max_resources_` (`:307-308`, `:322`,
  `:325`, `:362-365`, `:154-177`). ferrolearn exposes NONE — only
  `cv_results`/`best_params`/`best_score`. **DIVERGENT** (missing-feature).
  NOT-STARTED (blocker #1867).
- REQ-DEFAULTS (string-defaults + classifier-aware semantics + resource/random_state):
  sklearn defaults `n_candidates="exhaust"`, `factor=3`, `min_resources="smallest"`,
  `max_resources="auto"`, `resource="n_samples"`, `cv=5`,
  `aggressive_elimination=False`, `random_state=None` (`:1035-1048`), with
  classifier-aware `min_resources_` (`:161-165`). ferrolearn `fn new` sets `factor=3`
  but requires explicit `cv`/`scoring`, has `n_candidates: usize` (no `"exhaust"`),
  `min_resources`/`max_resources: Option<usize>` (no `"smallest"`/`"auto"` strings),
  no `resource`, no `aggressive_elimination` builder, and no classifier awareness.
  **DIVERGENT** (API surface). NOT-STARTED (blocker #1868). Live (AC-6): defaults
  resolve to `n_candidates='exhaust' min_resources='smallest' max_resources='auto'
  resource='n_samples' cv=5 factor=3`.
- REQ-X-1 (R-SUBSTRATE ndarray→ferray-core): production code imports
  `use ndarray::{Array1, Array2}`; the destination substrate is `ferray-core`
  (R-SUBSTRATE-1). NOT-STARTED (blocker #1869).
- REQ-X-2 (non-test production consumer): the crate re-export boundary
  (`pub use halving_random_search::HalvingRandomSearchCV in lib.rs`). SHIPPED.

## Acceptance criteria

Each AC is pinnable against a LIVE sklearn 1.5.2 call (R-CHAR-3 — expected values
come from the oracle, never copied from the ferrolearn side). The
HalvingRandomSearchCV oracle requires `from sklearn.experimental import
enable_halving_search_cv`. The shared `Const` regressor fixture (used below):

```
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
class Const(BaseEstimator, RegressorMixin):
    def __init__(self,c=0.0): self.c=c
    def fit(self,X,y): self.c_=self.c; return self
    def predict(self,X): return np.full(X.shape[0], self.c_)
```

- AC-1 (REQ-HALVING-MECHANIC, keep-count ceil + best candidate): the per-round
  keep-count is `ceil(n_candidates / factor)`, and the search returns the best
  candidate of the last round. Oracle (live sklearn — `n_candidates_` shows the ceil
  keep schedule; ferrolearn `div_ceil` matches):
  ```
  python3 -c "import numpy as np, math
  from sklearn.experimental import enable_halving_search_cv  # noqa
  from sklearn.model_selection import HalvingRandomSearchCV, KFold
  from scipy.stats import uniform
  from sklearn.base import BaseEstimator, RegressorMixin
  class Const(BaseEstimator, RegressorMixin):
      def __init__(self,c=0.0): self.c=c
      def fit(self,X,y): self.c_=self.c; return self
      def predict(self,X): return np.full(X.shape[0], self.c_)
  rng=np.random.RandomState(0); X=rng.rand(90,2); y=rng.rand(90)
  gs=HalvingRandomSearchCV(Const(), {'c':uniform(0,5)}, n_candidates=9, factor=3,
      cv=KFold(3), min_resources='smallest', random_state=0).fit(X,y)
  print('n_candidates_:', gs.n_candidates_, 'ceil(9/3)=', math.ceil(9/3))"
  # -> n_candidates_: [9, 3, 1] ceil(9/3)= 3   (ferrolearn div_ceil keeps 3)
  ```
  ferrolearn `n_survive = scored.len().div_ceil(self.factor).max(1)` keeps the SAME
  count; `test_halving_random_best_score_near_zero_for_perfect` asserts the mean
  estimator wins on constant `y` (best of final round).
- AC-2 (REQ-RANDOM-SAMPLING, count + seed-determinism): the same integer seed
  produces the same result across two independent runs. Oracle (live sklearn — the
  two fits give the identical best score; ferrolearn asserts the same property via
  `SmallRng` seeding):
  ```
  python3 -c "import numpy as np
  from sklearn.experimental import enable_halving_search_cv  # noqa
  from sklearn.model_selection import HalvingRandomSearchCV, KFold
  from scipy.stats import uniform
  from sklearn.base import BaseEstimator, RegressorMixin
  class Const(BaseEstimator, RegressorMixin):
      def __init__(self,c=0.0): self.c=c
      def fit(self,X,y): self.c_=self.c; return self
      def predict(self,X): return np.full(X.shape[0], self.c_)
  rng=np.random.RandomState(0); X=rng.rand(90,2); y=rng.rand(90)
  def run(): return HalvingRandomSearchCV(Const(), {'c':uniform(0,5)},
      n_candidates=6, factor=3, cv=KFold(3), random_state=99).fit(X,y).best_score_
  print('seed-stable:', run()==run())"
  # -> seed-stable: True
  ```
  ferrolearn `test_halving_random_deterministic_with_seed` builds two
  `HalvingRandomSearchCV` with `random_state=Some(99)`, fits both, and asserts
  `|s1 - s2| < 1e-10`. (Count: `n_candidates=6` ParamSets are sampled up-front on
  BOTH sides; the EXACT candidates diverge — REQ-RNG-EXACT, #1861.)
- AC-3 (REQ-MIN-RESOURCES / REQ-N-ITERATIONS, schedule DIVERGENCE): the
  `min_resources_` / `n_resources_` schedule differs. Oracle (live sklearn —
  `"smallest"` default heuristic):
  ```
  python3 -c "import numpy as np
  from sklearn.experimental import enable_halving_search_cv  # noqa
  from sklearn.model_selection import HalvingRandomSearchCV, KFold
  from scipy.stats import uniform
  from sklearn.base import BaseEstimator, RegressorMixin
  class Const(BaseEstimator, RegressorMixin):
      def __init__(self,c=0.0): self.c=c
      def fit(self,X,y): self.c_=self.c; return self
      def predict(self,X): return np.full(X.shape[0], self.c_)
  rng=np.random.RandomState(0); X=rng.rand(90,2); y=rng.rand(90)
  gs=HalvingRandomSearchCV(Const(), {'c':uniform(0,5)}, n_candidates=9, factor=3,
      cv=KFold(3), min_resources='smallest', random_state=0).fit(X,y)
  print('min_resources_:', gs.min_resources_, 'n_resources_:', gs.n_resources_,
        'n_iterations_:', gs.n_iterations_)"
  # -> min_resources_: 6 n_resources_: [6, 18, 54] n_iterations_: 3
  ```
  ferrolearn `compute_min_resources` → `ceil(90 / 3^2) = 10`, schedule `[10,30,90]`.
  DIVERGENCE (#1862, #1863).
- AC-4 (REQ-CV-RESULTS, history + columns DIVERGENCE): sklearn `cv_results_` records
  EVERY (iter, candidate) with `iter`/`n_resources` columns. Oracle (live sklearn):
  ```
  python3 -c "import numpy as np
  from sklearn.experimental import enable_halving_search_cv  # noqa
  from sklearn.model_selection import HalvingRandomSearchCV, KFold
  from scipy.stats import uniform
  from sklearn.base import BaseEstimator, RegressorMixin
  class Const(BaseEstimator, RegressorMixin):
      def __init__(self,c=0.0): self.c=c
      def fit(self,X,y): self.c_=self.c; return self
      def predict(self,X): return np.full(X.shape[0], self.c_)
  rng=np.random.RandomState(0); X=rng.rand(90,2); y=rng.rand(90)
  gs=HalvingRandomSearchCV(Const(), {'c':uniform(0,5)}, n_candidates=9, factor=3,
      cv=KFold(3), min_resources='smallest', random_state=0).fit(X,y)
  print('iter col:', gs.cv_results_['iter'].tolist())
  print('n rows:', len(gs.cv_results_['params']))"
  # -> iter col: [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2]
  # -> n rows: 13
  ```
  ferrolearn `CvResults` records only the final-round survivors, no
  `iter`/`n_resources` columns. DIVERGENCE (#1864).
- AC-5 (REQ-REFIT, refit/predict DIVERGENCE): sklearn `refit=True` (default) exposes
  `best_estimator_` and a delegating `predict`. Oracle (live sklearn):
  ```
  python3 -c "import numpy as np
  from sklearn.experimental import enable_halving_search_cv  # noqa
  from sklearn.model_selection import HalvingRandomSearchCV, KFold
  from scipy.stats import uniform
  from sklearn.base import BaseEstimator, RegressorMixin
  class Const(BaseEstimator, RegressorMixin):
      def __init__(self,c=0.0): self.c=c
      def fit(self,X,y): self.c_=self.c; return self
      def predict(self,X): return np.full(X.shape[0], self.c_)
  rng=np.random.RandomState(0); X=rng.rand(90,2); y=rng.rand(90)
  gs=HalvingRandomSearchCV(Const(), {'c':uniform(0,5)}, n_candidates=4, factor=2,
      cv=KFold(3), random_state=0).fit(X,y)
  print('has best_estimator_:', hasattr(gs,'best_estimator_'))"
  # -> has best_estimator_: True
  ```
  ferrolearn has no `best_estimator_`/`predict`/`score`. DIVERGENCE (#1866).
- AC-6 (REQ-DEFAULTS / REQ-N-CANDIDATES-EXHAUST, string-defaults DIVERGENCE): sklearn
  defaults `n_candidates="exhaust"`, `min_resources="smallest"`,
  `max_resources="auto"`, `resource="n_samples"`, `cv=5`, and resolves `"exhaust"` to
  `max_resources_ // min_resources_`. Oracle (live sklearn — the defaults resolve and
  exhaust expands):
  ```
  python3 -c "import numpy as np
  from sklearn.experimental import enable_halving_search_cv  # noqa
  from sklearn.model_selection import HalvingRandomSearchCV, KFold
  from scipy.stats import uniform
  from sklearn.base import BaseEstimator, RegressorMixin
  class Const(BaseEstimator, RegressorMixin):
      def __init__(self,c=0.0): self.c=c
      def fit(self,X,y): self.c_=self.c; return self
      def predict(self,X): return np.full(X.shape[0], self.c_)
  g=HalvingRandomSearchCV(Const(), {'c':uniform(0,5)})
  print(g.n_candidates, g.min_resources, g.max_resources, g.resource, g.cv, g.factor)
  rng=np.random.RandomState(0); X=rng.rand(90,2); y=rng.rand(90)
  gs=HalvingRandomSearchCV(Const(), {'c':uniform(0,5)}, factor=3, cv=KFold(3),
      random_state=0).fit(X,y)
  print('exhaust n_candidates_:', gs.n_candidates_)"
  # -> exhaust smallest auto n_samples 5 3
  # -> exhaust n_candidates_: [15, 5, 2]   (90//6 = 15 candidates)
  ```
  ferrolearn `new` requires explicit `cv`/`scoring`, mandatory integer `n_candidates`
  (no `"exhaust"`), no `"smallest"`/`"auto"` strings, no `resource`. DIVERGENCE
  (#1860, #1868).
- AC-X-1 (REQ-X-1, substrate): `grep -n "ndarray" halving_random_search.rs` shows
  production `use ndarray::{Array1, Array2}` — wrong substrate, migration to
  `ferray-core` owed (#1869).

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-HALVING-MECHANIC (loop shape + ceil keep-count + best_index) | SHIPPED | impl `pub fn fit in halving_random_search.rs`: `loop { let scored = self.evaluate_candidates(&all_params, &active, effective_budget, x, y)?; ... let n_survive = scored.len().div_ceil(self.factor).max(1); active = scored.into_iter().take(n_survive)...; budget = budget.saturating_mul(self.factor).min(max_res + 1); }` — evaluate active candidates on a budget, keep the top `ceil(n/factor)`, grow the budget, repeat. The SHAPE mirrors sklearn `_run_search` loop (`sklearn/model_selection/_search_successive_halving.py:310-360`); the KEEP-COUNT `div_ceil` MATCHES `_top_k`'s `n_candidates_to_keep = ceil(n_candidates / factor)` (`:359`, verified live AC-1: `n_candidates_ = [9,3,1]`, `ceil(9/3)=3`). Best candidate via shared `pub fn best_index in grid_search.rs` (fixed first-on-tie / NaN-worst), the SAME reduction as the sibling. Tests: `test_halving_random_best_score_near_zero_for_perfect`, `test_halving_random_basic_runs`, `test_halving_random_single_candidate`. Non-test consumer: REQ-X-2 (re-export). SCHEDULE facets split to REQ-MIN-RESOURCES/REQ-N-ITERATIONS. IDENTICAL to `halving_grid_search.md` REQ-HALVING-MECHANIC. |
| REQ-RANDOM-SAMPLING (sample n_candidates ParamSets up-front + seed-determinism) | SHIPPED | impl `fn sample_params in halving_random_search.rs`: `self.param_distributions.iter().map(|(name, dist)| (name.clone(), dist.sample(rng))).collect()` draws one value per distribution into a `ParamSet`; `fn fit` builds `let all_params: Vec<ParamSet> = (0..self.n_candidates).map(|_| self.sample_params(&mut rng)).collect()` from a seeded `let mut rng: SmallRng = match self.random_state { Some(seed) => SmallRng::seed_from_u64(seed), None => SmallRng::from_os_rng() }`. Mirrors sklearn `_generate_candidate_params` → `ParameterSampler(self.param_distributions, n_candidates_first_iter, random_state=self.random_state)` (`sklearn/model_selection/_search_successive_halving.py:1075-1079`). SHIPPED for the COUNT (`n_candidates` sets sampled up-front, same on both sides for an explicit int) + seed-determinism-across-runs (verified live AC-2: same seed ⇒ same best score; ferrolearn `test_halving_random_deterministic_with_seed` asserts `|s1 - s2| < 1e-10`). The EXACT candidates → REQ-RNG-EXACT (#1861); the `"exhaust"` count → REQ-N-CANDIDATES-EXHAUST (#1860). Non-test consumer: REQ-X-2 (re-export). |
| REQ-N-CANDIDATES-EXHAUST (n_candidates="exhaust" ⇒ max_resources_//min_resources_) | NOT-STARTED | open prereq blocker #1860. sklearn defaults `n_candidates="exhaust"` (`sklearn/.../_search_successive_halving.py:1035`); `_generate_candidate_params` resolves it to `n_candidates_first_iter = self.max_resources_ // self.min_resources_` (`:1074`), with the `n_candidates==min_resources=="exhaust"` mutual-exclusion check (`:145-152`). impl `pub struct HalvingRandomSearchCV in halving_random_search.rs` has `n_candidates: usize` (mandatory `new` arg, validated `== 0` in `fn fit`) — NO `"exhaust"` string, NO auto-resolution. Oracle (AC-6): default `n_candidates='exhaust'` expands to `n_candidates_ = [15,5,2]` (90//6=15). API + missing-auto-count divergence. |
| REQ-RNG-EXACT (exact sampled candidates / draw order) | NOT-STARTED | open prereq blocker #1861. sklearn `ParameterSampler` draws via numpy `RandomState(self.random_state)` over the SORTED parameter keys (`sklearn/.../_search_successive_halving.py:1075-1079`). impl `fn sample_params in halving_random_search.rs` draws via `SmallRng` (`seed_from_u64`/`from_os_rng`) in `param_distributions` INSERTION order. Different PRNG + draw order ⇒ the exact sampled candidates differ for the same integer seed. Per R-DEFER-3 an explicit RNG carve-out — NO failing test pins exact-candidate parity (seed-determinism-across-runs IS pinned, REQ-RANDOM-SAMPLING). |
| REQ-MIN-RESOURCES (min_resources_ formula) | NOT-STARTED | open prereq blocker #1862. sklearn `min_resources_ = n_splits * magic_factor(2) * [n_classes if classifier]` with default `min_resources="smallest"` for HalvingRandomSearchCV (`sklearn/.../_search_successive_halving.py:154-167`, `:1039`). impl `fn compute_min_resources in halving_random_search.rs`: `let min_res = (max_res as f64 / factor_pow).ceil() as usize; min_res.max(1)` with `factor_pow = factor^compute_n_rounds(n_candidates, factor)` — a heuristic unaware of `n_splits`/`n_classes`. Oracle (AC-3): sklearn `min_resources_ = 6` (n_splits=3·magic=2); ferrolearn `ceil(90/3^2)=10`. Architectural divergence. |
| REQ-N-ITERATIONS (n_iterations / n_resources schedule) | NOT-STARTED | open prereq blocker #1863. sklearn `n_iterations = min(n_possible, n_required)` and per-iter `n_resources = int(factor**power * min_resources_)` (`sklearn/.../_search_successive_halving.py:272-321`). impl `pub fn fit in halving_random_search.rs`: `budget = budget.saturating_mul(self.factor).min(max_res + 1)` looped until `effective_budget >= max_res || scored.len() <= 1` — no `n_required`/`n_possible`, and the budget base diverges via REQ-MIN-RESOURCES. Oracle (AC-3): sklearn `n_resources_ = [6,18,54]` (3 iters); ferrolearn `[10,30,90]`. Architectural divergence. |
| REQ-CV-RESULTS (cv_results_ records ALL iterations with iter/n_resources) | NOT-STARTED | open prereq blocker #1864. sklearn passes `more_results = {"iter": [...], "n_resources": [...]}` to every `evaluate_candidates` call (`sklearn/.../_search_successive_halving.py:350-357`), recording one row per (iter, candidate); `_select_best_index` then filters to the last iteration (`:191-212`). impl `pub fn fit in halving_random_search.rs` pushes ONLY the final round's survivors into `CvResults` (`results.push(params, fold_scores)` inside the `is_final`/single-candidate branches) — no per-iteration history, no `iter`/`n_resources` columns; `best_index` runs over those final rows directly. Oracle (AC-4): sklearn 13 rows, `iter=[0×9,1×3,2×1]`. Missing-feature divergence. |
| REQ-SUBSAMPLE (random fraction vs first-rows) | NOT-STARTED | open prereq blocker #1865. sklearn `_SubsampleMetaSplitter(fraction=n_resources/n_samples_orig, random_state=...)` RANDOM-subsamples a fraction each iteration (`sklearn/.../_search_successive_halving.py:333-340`, RNG-coupled, `subsample_test=True`). impl `fn evaluate_candidates in halving_random_search.rs`: `let x_sub = x.slice(ndarray::s![..budget, ..]).to_owned(); let y_sub: Array1<f64> = y.iter().take(budget).copied().collect();` — the FIRST `budget` rows, deterministic, no RNG (and again in `fit`'s final-round re-eval). Oracle (AC-references the sibling AC-4): `_SubsampleMetaSplitter` uses `check_random_state`. Structural + RNG-carve-out divergence. |
| REQ-REFIT (refit + best_estimator_ + predict/score) | NOT-STARTED | open prereq blocker #1866. sklearn `refit=True` DEFAULT (`sklearn/.../_search_successive_halving.py:1043`) refits the best params on the full data (via `BaseSearchCV.fit`) and delegates `predict`/`predict_proba`/`score`/`transform` to `best_estimator_`. impl `pub struct HalvingRandomSearchCV in halving_random_search.rs` is search-only — fields `pipeline_factory`/`param_distributions`/`n_candidates`/`cv`/`scoring`/`random_state`/`factor`/`min_resources`/`max_resources`/`results`, accessors `cv_results`/`best_params`/`best_score` only; NO refit, NO `best_estimator_`, NO `predict`/`score`. Oracle (AC-5): `hasattr(gs,'best_estimator_') == True`. Architectural. |
| REQ-ATTRS (n_resources_/n_candidates_/n_remaining_candidates_/n_iterations_/n_possible_iterations_/n_required_iterations_/min_resources_/max_resources_) | NOT-STARTED | open prereq blocker #1867. sklearn records all eight (`sklearn/.../_search_successive_halving.py:307-308`, `:322`, `:325`, `:362-365`, `:154-177`). impl `pub struct HalvingRandomSearchCV in halving_random_search.rs` exposes NONE — only `cv_results`/`best_params`/`best_score`; `compute_min_resources`/budget are local `fit` variables, never surfaced. Oracle (AC-1: `n_candidates_` present on the fitted sklearn estimator). Missing-feature divergence. |
| REQ-DEFAULTS (string-defaults + classifier-aware + resource/random_state) | NOT-STARTED | open prereq blocker #1868. sklearn `n_candidates="exhaust"`, `min_resources="smallest"`, `max_resources="auto"`, `resource="n_samples"`, `cv=5`, `random_state=None`, `factor=3` (`sklearn/.../_search_successive_halving.py:1035-1048`), with classifier-aware `min_resources_` (`:161-165`). impl `pub fn new in halving_random_search.rs` sets `factor: 3` but requires explicit `cv: Box<dyn CrossValidator>` + `scoring: fn(...)`, has mandatory `n_candidates: usize` (no `"exhaust"`), `min_resources`/`max_resources: Option<usize>` (no `"smallest"`/`"auto"`), no `resource`, no `aggressive_elimination` builder, no classifier awareness. Oracle (AC-6): defaults resolve to `exhaust smallest auto n_samples 5 3`. API-surface divergence. |
| REQ-X-1 (R-SUBSTRATE) | NOT-STARTED | open prereq blocker #1869. Production code in `halving_random_search.rs` imports `use ndarray::{Array1, Array2}`; `fit`/`evaluate_candidates` take `x: &Array2<f64>`/`y: &Array1<f64>` and use `ndarray::s!` slicing. Per R-SUBSTRATE-1 the destination is `ferray-core`; `ndarray` is the wrong substrate (R-SUBSTRATE-2). |
| REQ-X-2 (non-test production consumer) | SHIPPED | `pub struct HalvingRandomSearchCV` is re-exported at `pub use halving_random_search::HalvingRandomSearchCV in lib.rs` — the boundary public API per S5/R-DEFER-1 grandfathering (boundary estimator types ARE the public API). It additionally consumes the shared `CvResults` from `grid_search.rs` in non-test production (`use crate::grid_search::CvResults`; `CvResults::new`/`push`/`best_index`) and `cross_val_score` from `cross_validation.rs`. No internal caller and no `ferrolearn-python` binding yet — the re-export IS its production consumer. |

## Architecture

ferrolearn implements `HalvingRandomSearchCV<'a>` as a search-only struct that
REUSES the shared `CvResults` from `grid_search.rs` and `cross_val_score` from
`cross_validation.rs` (`halving_random_search.rs`). The closure base (R-DEV-7) is the
sibling shape of `HalvingGridSearchCV`, with `param_grid: Vec<ParamSet>` replaced by
`param_distributions: Vec<(String, Box<dyn Distribution>)>`, plus an `n_candidates:
usize` and a `random_state: Option<u64>`: `pipeline_factory: Box<dyn Fn(&ParamSet) ->
Pipeline + 'a>`, `cv: Box<dyn CrossValidator>`, `scoring: fn(&Array1<f64>,
&Array1<f64>) -> Result<f64, FerroError>`, builder fields `factor` (default 3),
`min_resources`/`max_resources: Option<usize>`, and `results: Option<CvResults>`.

The SHIPPED surface is narrow: two structural mechanics plus the consumer.

1. **REQ-HALVING-MECHANIC** — `fit` validates `n_candidates == 0`, empty
   `param_distributions`, `factor < 2`, and the `x`/`y` shape, computes `max_res =
   max_resources.unwrap_or(n_samples).min(n_samples)` and `min_res =
   compute_min_resources(...)`, then runs the successive-halving loop: evaluate the
   active candidates on `effective_budget` via `evaluate_candidates` (one
   `cross_val_score` per candidate, sorted descending by mean), keep the top
   `n.div_ceil(self.factor).max(1)`, then `budget *= factor`. This LOOP SHAPE is
   IDENTICAL to the sibling and mirrors sklearn's `_run_search` loop (`:310-360`); the
   keep-count `div_ceil` MATCHES `_top_k`'s `ceil(n_candidates / factor)` (`:359`) —
   verified live (AC-1). The best candidate uses the shared `CvResults::best_index`.

2. **REQ-RANDOM-SAMPLING** — this is the ONLY facet that differs from the sibling.
   `fit` seeds a `SmallRng` (`seed_from_u64(seed)` when `random_state` is `Some`, else
   `from_os_rng()`) and samples ALL candidates up-front: `all_params = (0..
   n_candidates).map(|_| self.sample_params(&mut rng))`. `sample_params` draws one
   value per distribution in `param_distributions` insertion order. This mirrors
   sklearn's `_generate_candidate_params` → `ParameterSampler(...,
   random_state=self.random_state)` (`:1075-1079`) for the COUNT and for
   determinism-across-runs (same seed ⇒ same candidates ⇒ same result, AC-2). The
   EXACT candidates diverge (SmallRng vs numpy `RandomState`, insertion vs sorted-key
   draw order — REQ-RNG-EXACT, #1861), and `"exhaust"` auto-resolution is absent
   (REQ-N-CANDIDATES-EXHAUST, #1860).

A NON-divergence worth recording, because it is easy to assume otherwise: like the
sibling, the elimination KEEP-COUNT does NOT diverge — sklearn keeps `ceil(n /
factor)` (`_top_k`, `:359`), ferrolearn's `div_ceil` keeps the same count (9→3
live). And the COUNT of candidates sampled (for an explicit integer `n_candidates`)
is identical on both sides. There is therefore NO deterministic single-spot fixable
divergence in this unit. Every NOT-STARTED REQ below is architectural /
missing-feature / RNG-coupled; even the matching keep-count + sample-count does NOT
make the unit match sklearn end-to-end, because the SCHEDULE (`min_resources_` and the
`n_resources` sequence) and the EXACT sampled candidates diverge.

Where ferrolearn DIVERGES (all NOT-STARTED, each with a filed blocker):

- **n_candidates="exhaust"** (REQ-N-CANDIDATES-EXHAUST, #1860): sklearn defaults
  `n_candidates="exhaust"` and resolves it to `max_resources_ // min_resources_`
  (`:1074`); ferrolearn requires a mandatory integer. Live: default exhaust expands to
  `n_candidates_ = [15,5,2]` (90//6=15).
- **exact sampled candidates** (REQ-RNG-EXACT, #1861): SmallRng (insertion order) vs
  numpy `RandomState` `ParameterSampler` (sorted-key order) — same-seed candidates
  differ. RNG carve-out.
- **min_resources_ schedule** (REQ-MIN-RESOURCES, #1862): `compute_min_resources`
  uses `ceil(max_res / factor^n_rounds).max(1)`; sklearn `"smallest"` default →
  `n_splits * 2 * [n_classes]` (`:154-167`). Live: sklearn 6 vs ferrolearn 10.
- **n_iterations / n_resources schedule** (REQ-N-ITERATIONS, #1863): `budget *=
  factor` loop vs `min(n_possible, n_required)` iterations with `n_resources =
  int(factor**power * min_resources_)` (`:272-321`). Live: `[6,18,54]` vs `[10,30,90]`.
- **cv_results_ richness / history** (REQ-CV-RESULTS, #1864): final-round survivors
  only vs every (iter, candidate) with `iter`/`n_resources` columns + last-iteration
  selection (`:191-212`, `:350-357`). Live: 13 rows / `iter=[0×9,1×3,2×1]`.
- **resource subsampling** (REQ-SUBSAMPLE, #1865): first `budget` rows
  (`x.slice(s![..budget,..])`) vs random-fraction `_SubsampleMetaSplitter` (`:333-340`).
- **refit / best_estimator_ / search-as-estimator** (REQ-REFIT, #1866): absent
  (same gap as the sibling).
- **halving attributes** (REQ-ATTRS, #1867): the eight `n_resources_`/… attributes
  are absent.
- **string-defaults / classifier-aware / resource / random_state semantics**
  (REQ-DEFAULTS, #1868): no `"exhaust"`/`"smallest"`/`"auto"`, no `resource`, no
  `aggressive_elimination` builder, no classifier awareness; `cv`/`scoring` mandatory.
- **substrate** (REQ-X-1, #1869): `ndarray::{Array1, Array2}` must migrate to
  `ferray-core` (R-SUBSTRATE-1/2).

`HalvingRandomSearchCV` reaches production via the `lib.rs` re-export (REQ-X-2) and
consumes the shared `CvResults` from `grid_search.rs` and `cross_val_score` from
`cross_validation.rs` in non-test production.

## Verification

Commands establishing the SHIPPED claims (baseline
`fedc1ed468cac887cd10660eceaa0c7bc92f2be9`):

- `cargo test -p ferrolearn-model-sel --lib halving_random_search` → 12 passed, 0
  failed (`halving_random_search::tests::{test_halving_random_basic_runs,
  test_halving_random_deterministic_with_seed,
  test_halving_random_best_score_near_zero_for_perfect,
  test_halving_random_n_candidates_zero_error,
  test_halving_random_empty_distributions_error,
  test_halving_random_factor_less_than_2_error,
  test_halving_random_returns_none_before_fit, test_halving_random_shape_mismatch_error,
  test_halving_random_single_candidate, test_halving_random_custom_factor,
  test_halving_random_best_score_is_finite,
  test_halving_random_cv_results_non_empty_after_fit}`).
- REQ-HALVING-MECHANIC keep-count oracle (live sklearn 1.5.2 — ceil keep schedule):
  `n_candidates_ = [9,3,1]`, `ceil(9/3)=3` (ferrolearn `div_ceil` keeps 3) — AC-1.
- REQ-RANDOM-SAMPLING determinism oracle (live sklearn — two fits with the same seed
  give the identical best score; ferrolearn
  `test_halving_random_deterministic_with_seed` asserts `|s1 - s2| < 1e-10`) — AC-2.
- REQ-MIN-RESOURCES / REQ-N-ITERATIONS schedule DIVERGENCE oracle (live sklearn):
  `min_resources_ = 6`, `n_resources_ = [6,18,54]`, `n_iterations_ = 3` (ferrolearn
  `ceil(90/3^2)=10`, schedule `[10,30,90]`) — AC-3 (#1862, #1863).
- REQ-CV-RESULTS DIVERGENCE oracle: `gs.cv_results_['iter'] == [0×9,1×3,2×1]`, 13
  rows for 9 candidates (factor 3) — ferrolearn keeps only the final-round survivors,
  no `iter`/`n_resources` columns (#1864) — AC-4.
- REQ-SUBSAMPLE DIVERGENCE oracle: `_SubsampleMetaSplitter` in
  `sklearn/model_selection/_search_successive_halving.py` uses `check_random_state` /
  `resample` (random fraction subsample) — ferrolearn slices the first `budget` rows
  (#1865).
- REQ-REFIT DIVERGENCE oracle: `hasattr(gs,'best_estimator_') == True` (refit=True
  default) — ferrolearn has no `best_estimator_`/`predict`/`score` (#1866) — AC-5.
- REQ-ATTRS DIVERGENCE oracle: the fitted sklearn estimator has
  `n_resources_`/`n_candidates_`/`n_remaining_candidates_`/`n_iterations_`/
  `n_possible_iterations_`/`n_required_iterations_`/`min_resources_`/`max_resources_`
  — ferrolearn exposes none (#1867).
- REQ-DEFAULTS / REQ-N-CANDIDATES-EXHAUST DIVERGENCE oracle: `HalvingRandomSearchCV(
  est, dists)` defaults are `n_candidates='exhaust'`, `min_resources='smallest'`,
  `max_resources='auto'`, `resource='n_samples'`, `cv=5`, `factor=3`; the default
  exhaust expands to `n_candidates_ = [15,5,2]` (90//6=15) — ferrolearn requires
  explicit `cv`/`scoring` and a mandatory integer `n_candidates`, no string defaults
  (#1860, #1868) — AC-6.
- REQ-X-1 substrate: `grep -n "ndarray" halving_random_search.rs` shows production
  `use ndarray::{Array1, Array2}` — wrong substrate, migration owed (#1869).
- REQ-X-2 consumer: `grep -rn "HalvingRandomSearchCV"
  ferrolearn-model-sel/src/lib.rs` shows `pub use
  halving_random_search::HalvingRandomSearchCV`; `grep -n "CvResults\|cross_val_score"
  halving_random_search.rs` shows the shared-type consumption.

SHIPPED: REQ-HALVING-MECHANIC (loop shape + ceil keep-count + best_index),
REQ-RANDOM-SAMPLING (count + seed-determinism-across-runs), REQ-X-2 (consumer —
re-export + shared `CvResults`/`cross_val_score`).
NOT-STARTED: REQ-N-CANDIDATES-EXHAUST (#1860), REQ-RNG-EXACT (#1861),
REQ-MIN-RESOURCES (#1862), REQ-N-ITERATIONS (#1863), REQ-CV-RESULTS (#1864),
REQ-SUBSAMPLE (#1865), REQ-REFIT (#1866), REQ-ATTRS (#1867), REQ-DEFAULTS (#1868),
REQ-X-1 (ferray substrate, #1869). Per R-DEFER-2 every REQ is binary
SHIPPED/NOT-STARTED.

There is NO deterministic single-spot fixable divergence in this unit (same audit
conclusion as `halving_grid_search.md`): the elimination keep-count `ceil(n/factor)`
already MATCHES sklearn (`_top_k`, `:359`), and the sample COUNT for an explicit
integer `n_candidates` is identical on both sides. Every NOT-STARTED REQ is
architectural / missing-feature (REQ-N-CANDIDATES-EXHAUST/REQ-MIN-RESOURCES/
REQ-N-ITERATIONS/REQ-CV-RESULTS/REQ-REFIT/REQ-ATTRS/REQ-DEFAULTS require new schedule
or surface) or RNG-coupled (REQ-RNG-EXACT/REQ-SUBSAMPLE — SmallRng vs numpy
`RandomState` / `_SubsampleMetaSplitter`); no single-line fix closes any of them.

Least-confident SHIPPED claim: REQ-RANDOM-SAMPLING. It asserts ONLY the sampled COUNT
and seed-determinism-ACROSS-RUNS (same `random_state` ⇒ same result, twice), NOT
parity with sklearn's actual sampled candidates. Because ferrolearn uses `SmallRng`
in insertion order while sklearn uses numpy `RandomState` `ParameterSampler` in
sorted-key order (REQ-RNG-EXACT, #1861), the EXACT candidates — and therefore the
selected best params — differ from sklearn for the same integer seed on
non-degenerate distributions. The SHIPPED surface is the up-front sampling mechanic +
reproducibility, not sklearn-equivalent candidate draws.
