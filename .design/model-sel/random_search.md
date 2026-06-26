# random_search

<!--
tier: 3-component
status: draft
baseline-commit: 585562c3f311b45f9ad3f6ac3d96d74dd791468c
upstream-paths:
  - sklearn/model_selection/_search.py   # RandomizedSearchCV (:1576), _run_search (:1958), ParameterSampler (:216), _is_all_lists (:302), __iter__ (:308), __len__ (:343), BaseSearchCV.fit (:890), _select_best_index (:829), _format_results._store (:1087)
-->

## Summary

`ferrolearn-model-sel/src/random_search.rs` mirrors scikit-learn's
`RandomizedSearchCV` (`sklearn/model_selection/_search.py:1576`, subclass of
`BaseSearchCV` `:436`) and the sampling engine `ParameterSampler` (`:216`) that
its `_run_search` (`:1958`) drives — randomized hyperparameter search that draws
`n_iter` parameter combinations from user-supplied distributions, cross-validates
each, records per-fold scores, and reports the best combination. It is
`GridSearchCV`'s sibling and SHARES the `CvResults` result type defined in
`grid_search.rs`.

ferrolearn exposes `pub struct ParameterSampler` and
`pub struct RandomizedSearchCV<'a>`. `ParameterSampler` owns a single
`param_distributions: Vec<(String, Box<dyn Distribution>)>`, `n_iter: usize`, and
an optional `random_state: Option<u64>`; `sample()` rejects `n_iter == 0`, seeds a
`SmallRng` (`seed_from_u64(seed)` or `from_os_rng()` when `None`), then samples
ONE value from EACH distribution per candidate. An empty distribution list yields
one empty `ParamSet`, matching sklearn's empty-grid cap. `RandomizedSearchCV`
uses that sampler in `fit`, builds a pipeline via the factory, calls
`cross_val_score`, and pushes the per-fold scores into the shared `CvResults`.
Accessors `cv_results()`/`best_params()`/`best_score()` read the search back via
`CvResults::best_index`.

The CORE RANDOM-SAMPLING MECHANIC is faithful to sklearn's `ParameterSampler`
CONTINUOUS branch (`:330-341`, the WITH-replacement case for distributions that
have `rvs`): `n_iter` draws, one value per distribution per draw, each
cross-validated; `len(params) == n_iter`. `mean_test_score` is the UNWEIGHTED fold
mean and `best_index` is the shared FIXED first-on-tie / NaN-worst reduction
(REQ-BESTIDX of `grid_search.md`). Seed determinism ACROSS RUNS holds
structurally.

Several behaviors DIVERGE. NONE is a single-spot DETERMINISTIC fixable divergence
in `random_search.rs` (the only such spot, `CvResults::best_index`, is owned and
already-pinned by `grid_search.md` REQ-BESTIDX #1776 — `random_search.rs` merely
consumes it). The divergences split three ways: (a) an R-DEFER-3 RNG carve-out —
SmallRng vs numpy `RandomState` means the EXACT sampled values and draw order
differ, with NO failing test (#1786); (b) DETERMINISTIC-but-ARCHITECTURAL gaps —
the all-discrete WITHOUT-replacement + cap-at-grid_size + `UserWarning` branch
(#1784, needs a cardinality/discreteness extension to the `Distribution` trait),
and the list-of-dicts `rng.choice` surface (#1785); (c) the same missing-feature
gaps as `grid_search.md` — no refit / `best_estimator_` / predict (#1777), sparse
`CvResults` (#1778), no default cv / scoring / `n_iter` (#1779), no
parallel/verbose/return_train_score (#1780), `ndarray` substrate (#1781).

## Upstream reference (scikit-learn 1.5.2, commit 156ef14)

### `RandomizedSearchCV` (`sklearn/model_selection/_search.py:1576`)
- `:1576` — `class RandomizedSearchCV(BaseSearchCV)`. `__init__(self, estimator,
  param_distributions, *, n_iter=10, scoring=None, n_jobs=None, refit=True,
  cv=None, verbose=0, pre_dispatch="2*n_jobs", random_state=None,
  error_score=np.nan, return_train_score=False)`. `n_iter` DEFAULTS to 10; every
  param after `param_distributions` is keyword-only; `refit` DEFAULTS to `True`,
  `cv`/`scoring`/`random_state` DEFAULT to `None`.
- `:1958-1964` — `_run_search(self, evaluate_candidates)`: the whole
  RandomizedSearchCV body is `evaluate_candidates(ParameterSampler(
  self.param_distributions, self.n_iter, random_state=self.random_state))` — the
  candidate set is whatever `ParameterSampler` yields.

### `ParameterSampler` (`sklearn/model_selection/_search.py:216`)
- `:302-306` — `_is_all_lists(self)`: `all(all(not hasattr(v, "rvs") for v in
  dist.values()) for dist in self.param_distributions)` — True iff EVERY value in
  EVERY distribution-dict is a plain list (none has an `rvs` method).
- `:308-341` — `__iter__`: `rng = check_random_state(self.random_state)`. If
  `_is_all_lists()` (`:313-328`): build `param_grid = ParameterGrid(...)`,
  `grid_size = len(param_grid)`; if `grid_size < n_iter` WARN `UserWarning`
  ("The total space of parameters %d is smaller than n_iter=%d ...") and cap
  `n_iter = grid_size` (`:319-326`); then `for i in sample_without_replacement(
  grid_size, n_iter, random_state=rng): yield param_grid[i]` — DISTINCT candidates,
  WITHOUT replacement. ELSE (some value has `rvs`, `:330-341`):
  `for _ in range(self.n_iter): dist = rng.choice(self.param_distributions);
  items = sorted(dist.items()); for k, v in items: params[k] = v.rvs(
  random_state=rng) if hasattr(v, "rvs") else v[rng.randint(len(v))]; yield params`
  — WITH replacement, keys SORTED for reproducibility, ONE dict picked per iter.
- `:343-349` — `__len__`: `min(self.n_iter, grid_size)` when `_is_all_lists()`
  else `self.n_iter`.

### `BaseSearchCV.fit` / `_select_best_index` / `_store`
(shared with `GridSearchCV`; see `grid_search.md` for the full annotation)
- `:890-1075` — `fit`: `check_scoring` for `scoring=None` (`:857`), `check_cv` for
  `cv=None` → 5-fold classifier-aware (`:928`), per-(candidate, split)
  `_fit_and_score`, `_format_results`, then `_run_search(evaluate_candidates)`
  (`:1019`).
- `:1034-1061` — `best_index_ = self._select_best_index(...)`; `refit=True`
  (default) refits the best params on the full data and exposes `best_estimator_`
  / `refit_time_` and delegating `predict`/`predict_proba`/`score`/`transform`
  (`:577`, `:546-552`).
- `:829-841` — `_select_best_index`: `best_index = results[
  f"rank_test_{refit_metric}"].argmin()` — `np.argmin` returns the FIRST minimum →
  lowest-index candidate wins on a tie.
- `:1087-1139` — `_store`: `array_means = np.average(array, axis=1, weights=None)`
  (UNWEIGHTED fold mean, `:1097`); per-split `split{i}_test_score` (`:1095`);
  `std_test_score` (`:1117`); `rank_test_score` via `rankdata(-means,
  method="min")` over `nan_to_num(..., nan=nanmin-1)` (NaN treated tied-worst,
  `:1123-1132`); timing + `param_<name>` (`:1134-1139`).

## Requirements

R-DEV mental test applied per REQ ("numerical/structural contract" or "API
contract" → MATCH; "Cython/CPython footgun" → deviate; "missing feature" →
NOT-STARTED with a blocker; "non-deterministic RNG substrate" → R-DEFER-3 carve-out
blocker with NO failing test).

- REQ-1 (random-sampling mechanic — continuous / WITH-replacement branch):
  draw `n_iter` `ParamSet`s, one value per distribution per draw, build a fresh
  pipeline per draw, cross-validate each, collect per-fold scores;
  `len(params) == n_iter` when at least one distribution is continuous. Mirrors
  `ParameterSampler.__iter__` ELSE branch (`:330-341`) — the WITH-replacement case
  for distributions that have `rvs`, and `__len__` returns `n_iter` (`:349`).
  **MATCH** (R-DEV-1 structural contract): the observable "n_iter candidates,
  each one-value-per-distribution, each cross-validated, scores collected" is the
  contract. ferrolearn's single `Vec<(name, dist)>` matches sklearn's SINGLE-dict
  case (the list-of-dicts `rng.choice` per-iter pick `:332` is a separate REQ,
  below); the factory closure is the R-DEV-7 analog of
  `clone(base_estimator).set_params(**parameters)` (no reflection in Rust).
  SHIPPED.
- REQ-2 (seed determinism ACROSS RUNS — structural): the same `random_state`
  yields identical sampled `ParamSet`s across two independent `fit` runs. Mirrors
  sklearn's `random_state` determinism CONTRACT (R-DEV-1: `random_state`
  reproducibility is an explicit MATCH item). **MATCH** for the across-runs
  structural contract — ferrolearn seeds `SmallRng::seed_from_u64(seed)`, so two
  runs with the same seed draw the same sequence. The EXACT sampled VALUES and
  draw order differ from numpy (`SmallRng` vs `check_random_state`/`RandomState`)
  — that exact-match is an R-DEFER-3 carve-out (REQ-RNG-EXACT, below), NOT part of
  this REQ. SHIPPED.
- REQ-EMPTY-GRID (empty param_distributions ⇒ one empty candidate): sklearn treats
  an empty distribution dict as an all-list grid with `grid_size = 1`, so
  `ParameterSampler({}, n_iter=3)` yields `[{}]` and `RandomizedSearchCV` evaluates
  exactly one candidate. ferrolearn `ParameterSampler::sample` mirrors that cap
  and `RandomizedSearchCV::fit` evaluates the single empty `ParamSet`. SHIPPED.
- REQ-BESTIDX-MEAN (best_index + mean via shared `CvResults`): `best_score`/
  `best_params` select the candidate via `CvResults::best_index` (the shared FIXED
  first-on-tie / NaN-worst reduction — `grid_search.md` REQ-BESTIDX) and
  `mean_test_score` is the UNWEIGHTED fold mean (`CvResults::push`). Mirrors
  `_select_best_index` (`:840`) + `np.average(..., weights=None)` (`:1097`).
  **MATCH** (R-DEV-1 numerical + tie-break contract). `random_search.rs` CONSUMES
  the shared type and does not define its own reduction; the tie-break fixable
  divergence is owned and pinned by `grid_search.md` REQ-BESTIDX (#1776), not a
  `random_search.rs` blocker. SHIPPED (the consumption of the shared mean +
  best_index path).
- REQ-RNG-EXACT (exact sampled values / draw order — R-DEFER-3 CARVE-OUT): sklearn
  draws via numpy `check_random_state(self.random_state)` (`:309`) — for
  continuous, `v.rvs(random_state=rng)`; for discrete, `v[rng.randint(len(v))]`,
  with keys SORTED per iter (`:334`). ferrolearn draws via `SmallRng` and samples
  distributions in INSERTION order, not sorted-key order. The exact value sequence
  is unreproducible across the two PRNGs. **R-DEFER-3 carve-out** (the `random_state`
  CONTRACT is determinism-across-runs, MATCH/SHIPPED via REQ-2; bit-exact numpy
  agreement is not a determinism-class divergence with a single-spot fix — it is a
  substrate difference). Filed as a blocker WITH NO failing test (#1786) — there is
  no Rust `#[test]` whose green state a fixer could land, because matching numpy's
  exact stream requires the `ferray::random` substrate (REQ-X-1), not a code edit
  here. NOT-STARTED.
- REQ-WITHOUT-REPLACEMENT (all-discrete ⇒ WITHOUT replacement + cap n_iter at
  grid_size + UserWarning): sklearn `_is_all_lists()` branch (`:313-328`) samples
  DISTINCT candidates via `sample_without_replacement`, caps `n_iter = grid_size`
  when `grid_size < n_iter`, and emits a `UserWarning`; `__len__` returns
  `min(n_iter, grid_size)` (`:347`). ferrolearn ALWAYS samples WITH replacement
  (no cap, duplicates possible, no warning) — and the `Distribution` trait
  (`distributions.rs`) exposes ONLY `fn sample(&self, rng) -> ParamValue`, with NO
  discreteness or cardinality method, so `random_search.rs` cannot detect the
  all-lists case or compute `grid_size`. **DETERMINISTIC-but-ARCHITECTURAL**
  divergence: a faithful fix requires extending the `Distribution` trait
  (cardinality + discreteness) AND a without-replacement sampler — not a single-spot
  edit. NOT-STARTED (blocker #1784).
- REQ-LIST-OF-DICTS (list of distribution-dicts + per-iter `rng.choice`): sklearn
  accepts a LIST of distribution-dicts and, per iter, picks one via
  `rng.choice(self.param_distributions)` then samples from it (`:330-341`).
  ferrolearn takes a SINGLE `Vec<(name, Box<dyn Distribution>)>` and samples every
  distribution every iter. **MATCH-intent / missing-feature**: the single-dict case
  is covered by REQ-1; the list-of-dicts surface is absent. NOT-STARTED (blocker
  #1785).
- REQ-REFIT (refit + best_estimator_ + delegating predict/score): sklearn
  `refit=True` DEFAULT (`:1576` init) refits on the full data (`:1046-1061`) and
  delegates `predict`/`predict_proba`/`score`/`transform` (`:577`, `:546-552`).
  ferrolearn `RandomizedSearchCV` is search-only — no refit, no `best_estimator_`,
  no `predict`/`score`. **MATCH-intent / missing-feature** (architectural; same gap
  as `grid_search.md` REQ-REFIT). NOT-STARTED (shared blocker #1777).
- REQ-CVRESULTS (cv_results_ richness): the SHARED `CvResults` has only
  `params`/`mean_scores`/`all_scores` — no `std_test_score` (`:1117`),
  `rank_test_score` (`:1129`), per-split `split{i}_test_score` (`:1095`), timing
  (`:1134-1135`), or `param_<name>` (`:1137-1139`). **MATCH-intent /
  missing-feature** (same gap as `grid_search.md` REQ-CVRESULTS). NOT-STARTED
  (shared blocker #1778).
- REQ-DEFAULT-CV / REQ-DEFAULT-SCORING / REQ-N-ITER-DEFAULT: sklearn defaults
  `n_iter=10` (`:1576`), `cv=None` → 5-fold classifier-aware (`:928`),
  `scoring=None` → estimator default scorer (`:857`). ferrolearn requires
  `n_iter`, `cv`, and `scoring` all as MANDATORY constructor args. **MATCH-intent /
  gap**. NOT-STARTED (shared blocker #1779).
- REQ-PARALLEL (n_jobs/pre_dispatch/verbose/return_train_score/multimetric):
  sklearn exposes all of these (`:1576` init); ferrolearn's `new` signature is
  `(pipeline_factory, param_distributions, n_iter, cv, scoring, random_state)` —
  none exist. **MATCH-intent / missing-feature**. NOT-STARTED (shared blocker
  #1780).
- REQ-ERROR-SCORE (error_score=np.nan continue): sklearn fills a failing
  `(candidate, split)` cell with nan and continues (`:996`). ferrolearn's `fit`
  delegates each candidate to `cross_val_score(...)?`, which propagates a failure
  via `?` and aborts. The divergence is REAL but lives in `cross_validation.rs`
  (the iteration unit), NOT `random_search.rs` — per S8 it is a CROSS-UNIT
  observation owned by that unit, not a random_search blocker. Noted, classified to
  the owning unit.
- REQ-X-1 (R-SUBSTRATE ndarray + rand→ferray): production code imports
  `use ndarray::{Array1, Array2}` (array type) and `use rand::{SeedableRng,
  rngs::SmallRng}` (random); the destination substrate is `ferray-core` +
  `ferray::random` (R-SUBSTRATE-1). NOT-STARTED (shared blocker #1781).
- REQ-X-2 (non-test production consumer): the crate re-export boundary
  `pub use random_search::{ParameterSampler, RandomizedSearchCV} in lib.rs`
  (S5/R-DEFER-1 boundary grandfathering — these are public model-selection
  surfaces). SHIPPED.

## Acceptance criteria

Each AC is pinnable against a LIVE sklearn 1.5.2 call (R-CHAR-3 — expected values
come from the oracle, NEVER copied from the ferrolearn side).

- AC-1 (REQ-1, continuous WITH-replacement count): a continuous distribution with
  `n_iter=15` yields exactly 15 candidates. Oracle (live sklearn — continuous ⇒
  with replacement ⇒ `len == n_iter`):
  ```
  python3 -c "from scipy.stats import uniform
  from sklearn.model_selection import ParameterSampler
  print('continuous len:', len(list(ParameterSampler({'a': uniform(0,1)}, n_iter=15, random_state=0))))"
  # -> continuous len: 15
  ```
  ferrolearn `test_random_search_samples_correct_n_iter` asserts
  `results.params.len() == 7` for `n_iter=7` over a continuous `Uniform`;
  `test_random_search_with_int_uniform` (`n_iter=10`) and
  `test_random_search_with_choice` (`n_iter=15`) likewise return the full `n_iter`
  count (ferrolearn ALWAYS uses the with-replacement count — see AC-WITHOUT-REPL
  for the divergence on the all-discrete case).
- AC-2 (REQ-2, seed determinism across runs): two `ParameterSampler`s with the
  same int seed yield the same sequence. Oracle (live sklearn — determinism
  contract):
  ```
  python3 -c "from scipy.stats import uniform
  from sklearn.model_selection import ParameterSampler
  a=[round(d['a'],9) for d in ParameterSampler({'a': uniform(0,1)}, n_iter=5, random_state=7)]
  b=[round(d['a'],9) for d in ParameterSampler({'a': uniform(0,1)}, n_iter=5, random_state=7)]
  print('same-seed identical:', a==b)"
  # -> same-seed identical: True
  ```
  ferrolearn `test_random_search_deterministic_with_seed` runs two
  `RandomizedSearchCV` instances with `random_state=Some(99)` and asserts every
  sampled `alpha` is pairwise-equal across the two runs. (The EXACT VALUES differ
  from sklearn's — REQ-RNG-EXACT carve-out #1786 — so this AC checks
  determinism-across-runs, never numpy-value equality.)
- AC-EMPTY-GRID (REQ-EMPTY-GRID): an empty distribution dict yields exactly one
  empty candidate. Oracle (live sklearn):
  ```
  python3 -c "from sklearn.model_selection import ParameterSampler
  print(list(ParameterSampler({}, n_iter=3, random_state=0)))"
  # -> [{}]
  ```
  ferrolearn `ParameterSampler::new(Vec::new(), 3, Some(0)).sample()` returns a
  single empty `ParamSet`, and `RandomizedSearchCV::fit` evaluates exactly that
  one candidate.
- AC-BESTIDX-MEAN (REQ-BESTIDX-MEAN, unweighted mean + first-on-tie best_index):
  `mean_test_score` is the simple mean of per-split scores, and on a tie the FIRST
  candidate is best. Oracle (live sklearn — mean is unweighted; argmin first-wins):
  ```
  python3 -c "import numpy as np
  from sklearn.model_selection import RandomizedSearchCV, KFold
  from sklearn.base import BaseEstimator, RegressorMixin
  class Const(BaseEstimator, RegressorMixin):
      def __init__(self,c=0.0): self.c=c
      def fit(self,X,y): self.c_=self.c; return self
      def predict(self,X): return np.full(X.shape[0], self.c_)
  X=np.zeros((30,2)); y=np.full(30,5.0)
  rs=RandomizedSearchCV(Const(), {'c':[4.0,6.0]}, n_iter=2,
      scoring='neg_mean_squared_error', cv=KFold(3), refit=False, random_state=0).fit(X,y)
  splits=[rs.cv_results_['split%d_test_score'%i][0] for i in range(3)]
  print('mean==simple mean:', np.isclose(rs.cv_results_['mean_test_score'][0], np.mean(splits)),
        '| ranks:', rs.cv_results_['rank_test_score'].tolist())"
  # -> mean==simple mean: True | ranks: [1, 1]   (tie -> argmin picks index 0)
  ```
  ferrolearn `CvResults::push` sets `mean = scores.mean()` and
  `best_params`/`best_score` route through `CvResults::best_index` (the shared
  first-on-tie reduction). Exercised by `test_random_search_best_params_selected_correctly`
  (best `c=1.0` predictor on `y=1.0` → `best_score ≈ 0`).
- AC-WITHOUT-REPL (REQ-WITHOUT-REPLACEMENT, all-discrete cap DIVERGENCE): an
  all-list distribution with `n_iter > grid_size` is CAPPED to the (distinct) grid
  size in sklearn; ferrolearn returns `n_iter` with duplicates. Oracle (live
  sklearn — the dispatch-supplied check):
  ```
  python3 -c "from sklearn.model_selection import ParameterSampler
  print('all-lists capped distinct:', len(list(ParameterSampler({'c':[1,2,3]}, n_iter=15, random_state=0))))"
  # -> all-lists capped distinct: 3
  ```
  (and it emits `UserWarning: The total space of parameters 3 is smaller than
  n_iter=15 ...`). ferrolearn with a `Choice([1,2,3])` and `n_iter=15` returns 15
  candidates WITH duplicates and NO warning — `test_random_search_with_choice`
  (three-option Choice, `n_iter=15`) yields 15 rows, demonstrating the
  no-cap/with-replacement behavior. DETERMINISTIC-but-ARCHITECTURAL DIVERGENCE.
  NOT-STARTED (#1784).
- AC-LIST-OF-DICTS (REQ-LIST-OF-DICTS DIVERGENCE): sklearn accepts a LIST of
  distribution-dicts. Oracle (live sklearn — list input is valid, picks one dict
  per iter):
  ```
  python3 -c "from sklearn.model_selection import ParameterSampler
  out=list(ParameterSampler([{'a':[1]}, {'b':[2]}], n_iter=10, random_state=0))
  print('keys seen:', sorted({k for d in out for k in d}))"
  # -> keys seen: ['a', 'b']
  ```
  ferrolearn `RandomizedSearchCV::new` takes a single
  `Vec<(String, Box<dyn Distribution>)>` — no list-of-dicts form. DIVERGENCE.
  NOT-STARTED (#1785).
- AC-REFIT (REQ-REFIT DIVERGENCE): sklearn `refit=True` (default) exposes
  `best_estimator_` and a delegating `predict`. Oracle (live sklearn):
  ```
  python3 -c "import numpy as np
  from sklearn.model_selection import RandomizedSearchCV, KFold
  from sklearn.base import BaseEstimator, RegressorMixin
  class Const(BaseEstimator, RegressorMixin):
      def __init__(self,c=0.0): self.c=c
      def fit(self,X,y): self.c_=self.c; return self
      def predict(self,X): return np.full(X.shape[0], self.c_)
  X=np.zeros((30,2)); y=np.full(30,5.0)
  rs=RandomizedSearchCV(Const(), {'c':[5.0]}, n_iter=1,
      scoring='neg_mean_squared_error', cv=KFold(3), refit=True, random_state=0).fit(X,y)
  print('has best_estimator_:', hasattr(rs,'best_estimator_'),
        '| predict:', rs.predict(np.zeros((2,2))).tolist())"
  # -> has best_estimator_: True | predict: [5.0, 5.0]
  ```
  ferrolearn `RandomizedSearchCV` has no `best_estimator_`, no `predict`/`score`.
  DIVERGENCE. NOT-STARTED (#1777).
- AC-CVRESULTS (REQ-CVRESULTS DIVERGENCE): sklearn `cv_results_` exposes
  `std_test_score`/`rank_test_score`/`split{i}_test_score`/timing/`param_<name>`.
  Oracle (live sklearn — key set):
  ```
  python3 -c "import numpy as np
  from sklearn.model_selection import RandomizedSearchCV, KFold
  from sklearn.base import BaseEstimator, RegressorMixin
  class Const(BaseEstimator, RegressorMixin):
      def __init__(self,c=0.0): self.c=c
      def fit(self,X,y): self.c_=self.c; return self
      def predict(self,X): return np.full(X.shape[0], self.c_)
  X=np.zeros((30,2)); y=np.full(30,5.0)
  rs=RandomizedSearchCV(Const(), {'c':[4.0,6.0]}, n_iter=2,
      scoring='neg_mean_squared_error', cv=KFold(3), refit=False, random_state=0).fit(X,y)
  print(sorted(rs.cv_results_.keys()))"
  # -> ['mean_fit_time','mean_score_time','mean_test_score','param_c','params',
  #     'rank_test_score','split0_test_score','split1_test_score','split2_test_score',
  #     'std_fit_time','std_score_time','std_test_score']
  ```
  ferrolearn `CvResults` carries only `params`/`mean_scores`/`all_scores`.
  DIVERGENCE. NOT-STARTED (#1778).
- AC-DEFAULT (REQ-DEFAULT-CV / REQ-DEFAULT-SCORING / REQ-N-ITER-DEFAULT
  DIVERGENCE): sklearn `n_iter` defaults to 10, `cv=None` → 5-fold,
  `scoring=None` → estimator scorer. Oracle (live sklearn):
  ```
  python3 -c "import inspect
  from sklearn.model_selection import RandomizedSearchCV, check_cv
  sig=inspect.signature(RandomizedSearchCV.__init__)
  print('n_iter default:', sig.parameters['n_iter'].default,
        '| cv default:', sig.parameters['cv'].default,
        '| scoring default:', sig.parameters['scoring'].default)
  print('cv=None classifier:', type(check_cv(None, [0,1,0,1], classifier=True)).__name__,
        check_cv(None, [0,1], classifier=True).get_n_splits())"
  # -> n_iter default: 10 | cv default: None | scoring default: None
  # -> cv=None classifier: StratifiedKFold 5
  ```
  ferrolearn `RandomizedSearchCV::new` requires explicit `n_iter`/`cv`/`scoring`.
  DIVERGENCE. NOT-STARTED (#1779).
- AC-PARALLEL (REQ-PARALLEL DIVERGENCE): sklearn accepts
  `n_jobs`/`pre_dispatch`/`verbose`/`return_train_score`. ferrolearn's `new`
  signature is `(pipeline_factory, param_distributions, n_iter, cv, scoring,
  random_state)` — none of these channels exist. STRUCTURAL. NOT-STARTED (#1780).
- AC-X-1 (REQ-X-1 substrate): `grep -n "ndarray\|rand::" random_search.rs` shows
  production `use ndarray::{Array1, Array2}` and `use rand::{SeedableRng,
  rngs::SmallRng}` — wrong substrate, migration to `ferray-core` + `ferray::random`
  owed (#1781).

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (random-sampling mechanic — continuous WITH-replacement) | SHIPPED | `ParameterSampler::sample` rejects `n_iter==0`, seeds `SmallRng`, and draws one `ParamSet` per iteration by sampling ONE value per distribution; `RandomizedSearchCV::fit` delegates candidate creation to `ParameterSampler`, then builds a pipeline and calls `cross_val_score` for each sample. Mirrors `ParameterSampler.__iter__` ELSE branch (`sklearn/model_selection/_search.py:330-341`, WITH-replacement) driven by `_run_search` → `evaluate_candidates(ParameterSampler(...))` (`:1958-1964`); `__len__` returns `n_iter` for the non-all-lists case (`:349`). Oracle (AC-1): `len(list(ParameterSampler({'a': uniform(0,1)}, n_iter=15, random_state=0))) == 15`. Tests: `green_parameter_sampler_public_surface_count`, `test_random_search_samples_correct_n_iter`, `test_random_search_with_int_uniform`/`_with_choice`/`_with_log_uniform`. Non-test consumer: REQ-X-2. |
| REQ-2 (seed determinism across runs — structural) | SHIPPED | `ParameterSampler::sample` seeds `SmallRng` from `random_state`; same seed ⇒ same draw sequence across runs. Mirrors sklearn's `random_state` determinism CONTRACT (`:309` `check_random_state(self.random_state)`, R-DEV-1 reproducibility MATCH item). Oracle (AC-2): two `ParameterSampler`s with `random_state=7` yield identical sequences. Tests: `green_parameter_sampler_public_surface_count`, `test_random_search_deterministic_with_seed` (two instances, `random_state=Some(99)`, asserts pairwise-equal sampled `alpha`). The EXACT numpy values are NOT claimed (REQ-RNG-EXACT carve-out #1786). Non-test consumer: REQ-X-2. |
| REQ-EMPTY-GRID (empty param_distributions ⇒ 1 empty candidate) | SHIPPED | `ParameterSampler::sample` caps an empty `param_distributions` list to one empty `ParamSet`, and `RandomizedSearchCV::fit` evaluates that single candidate. Mirrors sklearn's empty-grid path: `_is_all_lists()` is vacuously true, `ParameterGrid({})` has `grid_size = 1`, and `__len__` is `min(n_iter, 1)` (`sklearn/model_selection/_search.py:313-328`, `:345-347`). Tests: `green_parameter_sampler_empty_grid_one_candidate`, `green_empty_param_distributions_one_candidate`. |
| REQ-BESTIDX-MEAN (best_index + unweighted mean via shared CvResults) | SHIPPED | impl `pub fn best_params`/`pub fn best_score in random_search.rs` route through `results.best_index()` (the SHARED `pub fn best_index in grid_search.rs` — the FIXED first-on-tie / NaN-worst reduction, `grid_search.md` REQ-BESTIDX #1776); `pub(crate) fn push in grid_search.rs` (`impl CvResults`) sets `mean = scores.mean()`. Mirrors `_select_best_index` `rank_test_score.argmin()` (`sklearn/model_selection/_search.py:840`) + `np.average(array, axis=1, weights=None)` (`:1097`, UNWEIGHTED). Oracle (AC-BESTIDX-MEAN): `np.isclose(mean_test_score[0], np.mean(splits)) == True`, tie `ranks==[1,1]` ⇒ argmin index 0. Test: `test_random_search_best_params_selected_correctly` (`best_score().abs() < 1e-10`). `random_search.rs` CONSUMES the shared reduction; the tie-break fixable divergence is OWNED by `grid_search.md` (#1776), not a random_search blocker. Non-test consumer: REQ-X-2. |
| REQ-RNG-EXACT (exact sampled values / draw order — R-DEFER-3 CARVE-OUT) | NOT-STARTED | open prereq blocker #1786 (NO failing test). sklearn draws via numpy `check_random_state` (`sklearn/model_selection/_search.py:309`); discrete `v[rng.randint(len(v))]`, keys SORTED per iter (`:334`). `ParameterSampler::sample_one` uses `SmallRng` and samples in insertion order — the exact value stream is unreproducible across the two PRNGs. R-DEFER-3 carve-out: the `random_state` CONTRACT (determinism-across-runs) is SHIPPED via REQ-2; bit-exact numpy agreement is a substrate difference (resolved by REQ-X-1 `ferray::random`), NOT a single-spot deterministic fix — hence a blocker with NO failing `#[test]`. |
| REQ-WITHOUT-REPLACEMENT (all-discrete ⇒ without-replacement + cap + UserWarning) | NOT-STARTED | open prereq blocker #1784. sklearn `_is_all_lists()` branch (`sklearn/model_selection/_search.py:313-328`) samples DISTINCT via `sample_without_replacement`, caps `n_iter = grid_size` when `grid_size < n_iter` with a `UserWarning`; `__len__` → `min(n_iter, grid_size)` (`:347`). `ParameterSampler` always uses the continuous/with-replacement branch for non-empty distributions (no cap, duplicates possible, no warning); `pub trait Distribution in distributions.rs` exposes only `fn sample(&self, rng) -> ParamValue` — NO discreteness/cardinality method, so the all-lists case and `grid_size` are undetectable. Oracle (AC-WITHOUT-REPL): `len(list(ParameterSampler({'c':[1,2,3]}, n_iter=15, random_state=0))) == 3` (capped, distinct, + UserWarning); ferrolearn `Choice([1,2,3])` + `n_iter=15` → 15 with dups (`test_random_search_with_choice`). DETERMINISTIC-but-ARCHITECTURAL: needs a `Distribution`-trait extension + without-replacement sampler. |
| REQ-LIST-OF-DICTS (list of distribution-dicts + per-iter rng.choice) | NOT-STARTED | open prereq blocker #1785. sklearn accepts a LIST of distribution-dicts and picks one per iter via `rng.choice(self.param_distributions)` (`sklearn/model_selection/_search.py:330-341`). impl `pub fn new in random_search.rs` takes a SINGLE `param_distributions: Vec<(String, Box<dyn Distribution>)>` and samples EVERY distribution every iter. Oracle (AC-LIST-OF-DICTS): `ParameterSampler([{'a':[1]}, {'b':[2]}], n_iter=10, random_state=0)` is valid; ferrolearn has no list-of-dicts form. Missing-feature (architectural). |
| REQ-REFIT (refit + best_estimator_ + delegating predict/score) | NOT-STARTED | open prereq blocker #1777 (shared with `grid_search.md` REQ-REFIT). sklearn `refit=True` DEFAULT (`sklearn/model_selection/_search.py:1576`) refits on full data (`:1046-1061`) and delegates `predict`/`predict_proba`/`score`/`transform` (`:577`, `:546-552`). impl `pub struct RandomizedSearchCV in random_search.rs` is search-only — fields `pipeline_factory`/`param_distributions`/`n_iter`/`cv`/`scoring`/`random_state`/`results`, accessors `cv_results`/`best_params`/`best_score` only; NO refit, NO `best_estimator_`, NO `predict`/`score`. Oracle (AC-REFIT): `hasattr(rs,'best_estimator_') == True`, `rs.predict(...) == [5.0,5.0]`. Architectural blocker (no single-fixer test). |
| REQ-CVRESULTS (cv_results_ richness) | NOT-STARTED | open prereq blocker #1778 (shared `CvResults`, same as `grid_search.md` REQ-CVRESULTS). sklearn `cv_results_` has `std_test_score` (`:1117`), `rank_test_score` (`:1129`), `split{i}_test_score` (`:1095`), timing (`:1134-1135`), `param_<name>`/`params` (`:1137-1139`). impl `pub struct CvResults in grid_search.rs` (consumed via `use crate::grid_search::CvResults in random_search.rs`) has ONLY `params`/`mean_scores`/`all_scores`. Oracle (AC-CVRESULTS): key set includes `std_test_score`/`rank_test_score`/`split0_test_score`/`mean_fit_time`. Missing-feature blocker (no single-fixer test). |
| REQ-DEFAULT-CV / REQ-DEFAULT-SCORING / REQ-N-ITER-DEFAULT | NOT-STARTED | open prereq blocker #1779 (shared with `grid_search.md` REQ-DEFAULT). sklearn defaults `n_iter=10` (`sklearn/model_selection/_search.py:1576`), `cv=None` → 5-fold classifier-aware (`:928`), `scoring=None` → estimator scorer (`:857`). impl `pub fn new in random_search.rs` takes `n_iter: usize`, `cv: Box<dyn CrossValidator>`, `scoring: fn(...)` all MANDATORY — no defaults, no `is_classifier` stratification dispatch. Oracle (AC-DEFAULT): `n_iter` default 10, `cv`/`scoring` default `None`; `check_cv(None, [0,1,..], classifier=True)` → StratifiedKFold n_splits 5. |
| REQ-PARALLEL (n_jobs/pre_dispatch/verbose/return_train_score/multimetric) | NOT-STARTED | open prereq blocker #1780 (shared with `grid_search.md` REQ-PARALLEL). sklearn exposes `n_jobs`/`pre_dispatch="2*n_jobs"`/`verbose`/`return_train_score`/multimetric (`sklearn/model_selection/_search.py:1576` init). impl `pub fn new in random_search.rs` signature `(pipeline_factory, param_distributions, n_iter, cv, scoring, random_state)` — none of these channels exist. Missing-feature blocker (no single-fixer test). |
| REQ-ERROR-SCORE (error_score=np.nan continue) | NOT-STARTED | CROSS-UNIT (owned by `cross_validation.rs`, NOT a random_search blocker — S8). sklearn `error_score=np.nan` fills a failing cell and CONTINUES (`sklearn/model_selection/_search.py:996`). impl `pub fn fit in random_search.rs` delegates each candidate to `cross_val_score(...)?`; `cross_val_score in cross_validation.rs` propagates a failure via `?` and aborts. Real divergence, but lives in `cross_validation.rs`'s iteration unit — classified there per S8/R-DEFER-5, not pinned as a random_search blocker. |
| REQ-X-1 (R-SUBSTRATE ndarray + rand → ferray) | NOT-STARTED | open prereq blocker #1781 (shared substrate blocker). Production code in `random_search.rs` imports `use ndarray::{Array1, Array2}` (array type) and `use rand::{SeedableRng, rngs::SmallRng}` (random); `fit` takes `x: &Array2<f64>`/`y: &Array1<f64>`. Per R-SUBSTRATE-1 the destination is `ferray-core` + `ferray::random`; `ndarray`/`rand` are the wrong substrate. Not on the ferray substrate (R-SUBSTRATE-2). |
| REQ-X-2 (non-test production consumer) | SHIPPED | `pub struct ParameterSampler` and `pub struct RandomizedSearchCV` are re-exported at `pub use random_search::{ParameterSampler, RandomizedSearchCV} in lib.rs` — the boundary public model-selection API per S5/R-DEFER-1 grandfathering. `RandomizedSearchCV` consumes `ParameterSampler` internally; the re-export is the production boundary for both surfaces. (Note: `halving_random_search.rs` builds its OWN search and does not call `RandomizedSearchCV`; only the SHARED `CvResults` is cross-consumed.) |

## Architecture

ferrolearn implements `RandomizedSearchCV<'a>` as a search-only struct over the
SHARED `CvResults` result type (defined in `grid_search.rs`, consumed via
`use crate::grid_search::CvResults`). The struct holds a
`pipeline_factory: Box<dyn Fn(&ParamSet) -> Pipeline + 'a>`, a single
`param_distributions: Vec<(String, Box<dyn Distribution>)>`, `n_iter: usize`, a
`cv: Box<dyn CrossValidator>`, a `scoring: fn(&Array1<f64>, &Array1<f64>) ->
Result<f64, FerroError>` (higher is better), `random_state: Option<u64>`, and
`results: Option<CvResults>` populated by `fit`.

The CORE RANDOM-SAMPLING MECHANIC (REQ-1) is the faithful translation of
`ParameterSampler`'s CONTINUOUS / WITH-replacement branch. sklearn's
RandomizedSearchCV body is `_run_search(evaluate_candidates)` →
`evaluate_candidates(ParameterSampler(self.param_distributions, self.n_iter,
random_state=self.random_state))` (`:1958-1964`); for distributions with `rvs`,
`ParameterSampler.__iter__` loops `for _ in range(self.n_iter)`, sorts the dict
keys, and draws `v.rvs(random_state=rng)` per key (`:330-341`), with
`__len__ == n_iter` (`:349`). ferrolearn exposes `ParameterSampler::sample`,
which loops for the effective sample count, draws one value per distribution in
insertion order, and returns materialized `ParamSet`s. `RandomizedSearchCV::fit`
builds a fresh `(self.pipeline_factory)(&params)` for each sample and
`cross_val_score`s it — the factory closure is the R-DEV-7 analog of
`clone(base_estimator).set_params(**parameters)` (Rust has no `set_params`
reflection). `n_iter == 0` is an eager `FerroError::InvalidParameter` guard; an
empty distribution list yields one empty candidate (REQ-EMPTY-GRID).

Seed determinism ACROSS RUNS (REQ-2) is the structural `random_state` contract:
`SmallRng::seed_from_u64(seed)` (or `from_os_rng()` when `None`) — two runs with
the same seed draw the same sequence (`test_random_search_deterministic_with_seed`).

`CvResults::push` records `(params, mean = scores.mean(), all_scores = scores)`;
`best_params`/`best_score` route through the SHARED `CvResults::best_index`
(REQ-BESTIDX-MEAN). The unweighted fold mean matches `np.average(..., weights=None)`
(`:1097`), and the first-on-tie / NaN-worst reduction is the SHARED fixed behavior
documented under `grid_search.md` REQ-BESTIDX (#1776) — `random_search.rs`
CONSUMES it and introduces NO new reduction logic.

There is NO single-spot DETERMINISTIC fixable divergence INSIDE `random_search.rs`
for a critic to pin to a one-edit fixer. The only deterministic single-spot tie
issue (`CvResults::best_index`) is owned and pinned by `grid_search.md` (#1776);
fixing it there fixes `random_search.rs` for free. The remaining divergences are
either RNG carve-outs (no failing test) or architectural:

- **RNG carve-out** (REQ-RNG-EXACT, #1786, R-DEFER-3): `SmallRng` vs numpy
  `check_random_state` (`:309`) — exact sampled values and draw order differ
  (ferrolearn also samples in INSERTION order, not sklearn's per-iter sorted-key
  order `:334`). Not a deterministic single-spot fix; resolved only by the
  `ferray::random` substrate (REQ-X-1). NO failing test.
- **all-discrete WITHOUT-replacement + cap + UserWarning** (REQ-WITHOUT-REPLACEMENT,
  #1784): sklearn `_is_all_lists()` (`:302-306`) routes all-list inputs through
  `sample_without_replacement` with an `n_iter = min(n_iter, grid_size)` cap and a
  `UserWarning` (`:313-328`, `:347`). ferrolearn ALWAYS samples with replacement
  (duplicates possible, no cap, no warning) because `pub trait Distribution in
  distributions.rs` has no discreteness/cardinality method to detect the all-lists
  case or compute `grid_size`. Faithful parity is ARCHITECTURAL — extend the trait
  + add a without-replacement sampler.
- **list-of-dicts** (REQ-LIST-OF-DICTS, #1785): sklearn accepts a LIST of
  distribution-dicts and `rng.choice`s one per iter (`:330-332`); ferrolearn takes
  a single `Vec<(name, dist)>`.
- **refit / best_estimator_ / search-as-estimator** (REQ-REFIT, #1777),
  **cv_results_ richness** (REQ-CVRESULTS, #1778),
  **default n_iter/cv/scoring** (REQ-DEFAULT-*, #1779),
  **n_jobs/pre_dispatch/verbose/return_train_score/multimetric** (REQ-PARALLEL,
  #1780), and **ndarray + rand substrate** (REQ-X-1, #1781) — the SAME missing-
  feature / substrate gaps as `grid_search.md` (the two estimators share
  `BaseSearchCV` upstream and `CvResults` downstream).

A CROSS-UNIT note (REQ-ERROR-SCORE): sklearn's `error_score=np.nan` fills a failing
cell with nan and continues (`:996`); ferrolearn's `cross_val_score` propagates a
fit/score failure via `?` and aborts. Real, but OWNED by `cross_validation.rs`
(per S8), not pinned here.

`ParameterSampler` and `RandomizedSearchCV` reach production via the re-export
`pub use random_search::{ParameterSampler, RandomizedSearchCV} in lib.rs`
(REQ-X-2). The SHARED `CvResults` (defined in `grid_search.rs`) is the
cross-consumed type; `halving_random_search.rs` builds its OWN search loop and
does not call `RandomizedSearchCV`.

## Verification

Commands establishing the SHIPPED claims (baseline
`585562c3f311b45f9ad3f6ac3d96d74dd791468c`):

- `cargo test -p ferrolearn-model-sel --lib random_search` → 8 passed, 0 failed
  (`random_search::tests::{test_random_search_samples_correct_n_iter,
  test_random_search_deterministic_with_seed, test_random_search_returns_none_before_fit,
  test_random_search_n_iter_zero_error, test_random_search_with_log_uniform,
  test_random_search_with_int_uniform, test_random_search_with_choice,
  test_random_search_best_params_selected_correctly}`).
- REQ-1 continuous-count oracle (live sklearn 1.5.2 — with replacement ⇒
  `len == n_iter`):
  ```
  python3 -c "from scipy.stats import uniform
  from sklearn.model_selection import ParameterSampler
  print('continuous len:', len(list(ParameterSampler({'a': uniform(0,1)}, n_iter=15, random_state=0))))"
  # -> continuous len: 15
  ```
- REQ-2 determinism-across-runs oracle (live sklearn 1.5.2):
  ```
  python3 -c "from scipy.stats import uniform
  from sklearn.model_selection import ParameterSampler
  a=[round(d['a'],9) for d in ParameterSampler({'a': uniform(0,1)}, n_iter=5, random_state=7)]
  b=[round(d['a'],9) for d in ParameterSampler({'a': uniform(0,1)}, n_iter=5, random_state=7)]
  print('same-seed identical:', a==b)"
  # -> same-seed identical: True
  ```
- REQ-EMPTY-GRID oracle (live sklearn 1.5.2):
  ```
  python3 -c "from sklearn.model_selection import ParameterSampler
  print(list(ParameterSampler({}, n_iter=3, random_state=0)))"
  # -> [{}]
  ```
  ferrolearn `green_parameter_sampler_empty_grid_one_candidate` and
  `green_empty_param_distributions_one_candidate` pin the same one-candidate
  behavior.
- REQ-BESTIDX-MEAN oracle (live sklearn 1.5.2 — unweighted mean + first-on-tie):
  ```
  python3 -c "import numpy as np
  from sklearn.model_selection import RandomizedSearchCV, KFold
  from sklearn.base import BaseEstimator, RegressorMixin
  class Const(BaseEstimator, RegressorMixin):
      def __init__(self,c=0.0): self.c=c
      def fit(self,X,y): self.c_=self.c; return self
      def predict(self,X): return np.full(X.shape[0], self.c_)
  X=np.zeros((30,2)); y=np.full(30,5.0)
  rs=RandomizedSearchCV(Const(), {'c':[4.0,6.0]}, n_iter=2,
      scoring='neg_mean_squared_error', cv=KFold(3), refit=False, random_state=0).fit(X,y)
  splits=[rs.cv_results_['split%d_test_score'%i][0] for i in range(3)]
  print('mean==simple mean:', np.isclose(rs.cv_results_['mean_test_score'][0], np.mean(splits)),
        '| ranks:', rs.cv_results_['rank_test_score'].tolist())"
  # -> mean==simple mean: True | ranks: [1, 1]
  ```
- REQ-WITHOUT-REPLACEMENT DIVERGENCE oracle (the dispatch-supplied check — live
  sklearn 1.5.2):
  ```
  python3 -c "from sklearn.model_selection import ParameterSampler
  print('all-lists capped distinct:', len(list(ParameterSampler({'c':[1,2,3]}, n_iter=15, random_state=0))))"
  # -> all-lists capped distinct: 3   (+ UserWarning; ferrolearn Choice+n_iter=15 -> 15 with dups)  (#1784)
  ```
- REQ-LIST-OF-DICTS DIVERGENCE oracle: `ParameterSampler([{'a':[1]}, {'b':[2]}],
  n_iter=10, random_state=0)` is valid (keys `['a','b']` seen) — ferrolearn has no
  list-of-dicts form (#1785).
- REQ-RNG-EXACT carve-out (#1786): NO failing test — `SmallRng` vs numpy
  `RandomState` means exact values/draw-order differ; resolved by `ferray::random`
  (REQ-X-1), not a code edit here.
- REQ-REFIT DIVERGENCE oracle: `hasattr(rs,'best_estimator_') == True`,
  `rs.predict(np.zeros((2,2))) == [5.0,5.0]` (refit=True default) — ferrolearn has
  no `best_estimator_`/`predict` (#1777).
- REQ-CVRESULTS DIVERGENCE oracle: `sorted(rs.cv_results_.keys())` includes
  `std_test_score`/`rank_test_score`/`split0_test_score`/`mean_fit_time` —
  ferrolearn `CvResults` has only `params`/`mean_scores`/`all_scores` (#1778).
- REQ-DEFAULT DIVERGENCE oracle: `RandomizedSearchCV.__init__` defaults
  `n_iter=10`, `cv=None`, `scoring=None`; `check_cv(None, [0,1,0,1],
  classifier=True)` → StratifiedKFold n_splits 5 — ferrolearn requires explicit
  `n_iter`/`cv`/`scoring` (#1779).
- REQ-X-1 substrate: `grep -n "ndarray\|rand::" random_search.rs` shows production
  `use ndarray::{Array1, Array2}` + `use rand::{SeedableRng, rngs::SmallRng}` —
  wrong substrate, migration owed (#1781).
- REQ-X-2 consumer: `grep -rn "ParameterSampler\|RandomizedSearchCV" ferrolearn-model-sel/src`
  shows `pub use random_search::{ParameterSampler, RandomizedSearchCV} in lib.rs`
  (the production re-export boundary) and `RandomizedSearchCV::fit` consuming
  `ParameterSampler`.

SHIPPED: REQ-1 (random-sampling mechanic — continuous WITH-replacement),
REQ-2 (seed determinism across runs), REQ-EMPTY-GRID (one empty candidate),
REQ-BESTIDX-MEAN (best_index + unweighted mean via shared `CvResults`), REQ-X-2
(consumer — re-export).
NOT-STARTED: REQ-RNG-EXACT (R-DEFER-3 RNG carve-out, NO failing test, #1786),
REQ-WITHOUT-REPLACEMENT (deterministic-but-architectural, #1784),
REQ-LIST-OF-DICTS (#1785), REQ-REFIT (#1777), REQ-CVRESULTS (#1778),
REQ-DEFAULT-CV/SCORING/N-ITER (#1779), REQ-PARALLEL (#1780),
REQ-ERROR-SCORE (cross-unit, owned by `cross_validation.rs`, S8 — not pinned here),
REQ-X-1 (ferray substrate, #1781). Per R-DEFER-2 every REQ is binary
SHIPPED/NOT-STARTED.

There is NO single-spot DETERMINISTIC divergence INSIDE `random_search.rs` for a
fixer to pin: the lone deterministic tie-break spot (`CvResults::best_index`) is
already-pinned and OWNED by `grid_search.md` REQ-BESTIDX (#1776) — fixing it there
fixes random_search transitively. REQ-RNG-EXACT is an R-DEFER-3 carve-out (NO
failing test); REQ-WITHOUT-REPLACEMENT and REQ-LIST-OF-DICTS are deterministic but
ARCHITECTURAL (trait extension / new input form); the rest are shared
missing-feature / substrate blockers. This is a VERIFY-AND-DOCUMENT unit, not a
single-fix-and-pin unit.

Least-confident SHIPPED claim: REQ-1 — ferrolearn's count is faithful to sklearn's
WITH-replacement branch (continuous distributions, `len == n_iter`), but ferrolearn
ALSO returns `n_iter` for ALL-DISCRETE distributions where sklearn would cap to the
distinct grid size (REQ-WITHOUT-REPLACEMENT #1784). The SHIPPED surface is therefore
the continuous / with-replacement mechanic specifically; the all-discrete count
diverges and is carved out as NOT-STARTED.
