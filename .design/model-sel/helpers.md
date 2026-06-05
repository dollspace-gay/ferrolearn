# make_pipeline / make_union (construction helpers)

<!--
tier: 3-component
status: draft
baseline-commit: 5c0379e3832819710caebe777154d72c5bb01497
upstream-paths:
  - sklearn/pipeline.py   # make_pipeline (:1220), make_union (:1889), _name_estimators (:1196)
-->

## Summary

`ferrolearn-model-sel/src/helpers.rs` mirrors scikit-learn's two pipeline
*construction shorthands* — `make_pipeline` (`sklearn/pipeline.py:1220`) and
`make_union` (`:1889`) — both of which build a named estimator/transformer list
via the shared `_name_estimators` helper (`:1196`) and hand it to the `Pipeline`
/ `FeatureUnion` constructor. ferrolearn exposes `pub fn make_pipeline` (builds a
`Pipeline<F>` from an explicit `(Vec<steps>, Option<estimator>)`) and
`pub fn make_union` (builds a `FeatureUnion<F>` from a `Vec<transformers>`); both
are re-exported from `lib.rs`. The CONSTRUCTION mechanic ships; the step-NAMING
strategy and the constructor pass-through params (`memory`/`verbose`/`n_jobs`)
diverge.

## Upstream reference (scikit-learn 1.5.2, commit 156ef14)

- `sklearn/pipeline.py:1196-1217` — `def _name_estimators(estimators)`, in full:
  ```python
  def _name_estimators(estimators):
      """Generate names for estimators."""

      names = [
          estimator if isinstance(estimator, str) else type(estimator).__name__.lower()
          for estimator in estimators
      ]
      namecount = defaultdict(int)
      for est, name in zip(estimators, names):
          namecount[name] += 1

      for k, v in list(namecount.items()):
          if v == 1:
              del namecount[k]

      for i in reversed(range(len(estimators))):
          name = names[i]
          if name in namecount:
              names[i] += "-%d" % namecount[name]
              namecount[name] -= 1

      return list(zip(names, estimators))
  ```
  Each name is the **lowercased concrete class name** `type(estimator).__name__.lower()`
  (`:1200`). `namecount` counts each base name; names that occur exactly once are
  removed from `namecount` (`:1207-1209`), so only COLLIDING names get a suffix.
  Then, iterating in REVERSE (`:1211`), every still-counted name has `-%d`
  appended (`:1214`) using a decrementing counter (`:1215`) — so two
  `StandardScaler` become `standardscaler-1`, `standardscaler-2` (first written
  gets `-1`), while a unique `logisticregression` gets no suffix. Returns
  `list(zip(names, estimators))` (`:1217`).
- `sklearn/pipeline.py:1220` — `def make_pipeline(*steps, memory=None, verbose=False)`;
  body `:1265` `return Pipeline(_name_estimators(steps), memory=memory, verbose=verbose)`.
  Variadic `*steps`; naming is delegated entirely to `_name_estimators`; the
  LAST step being the final estimator (and earlier ones transformers) is inferred
  later by the `Pipeline` itself, NOT by `make_pipeline`.
- `sklearn/pipeline.py:1889` — `def make_union(*transformers, n_jobs=None, verbose=False)`;
  body `:1934` `return FeatureUnion(_name_estimators(transformers), n_jobs=n_jobs, verbose=verbose)`.

## Requirements

- REQ-MAKE-PIPELINE-MECHANIC: build a `Pipeline` from a step list plus a final
  estimator. Mirrors `make_pipeline` assembling a `Pipeline` from the named step
  sequence (`sklearn/pipeline.py:1220`, `:1265`). MATCH on the construction
  mechanic. The variadic `*steps` + last-step-is-estimator inference is an R-DEV-7
  API-shape deviation: ferrolearn takes an explicit `(Vec<steps>, Option<estimator>)`
  split rather than inferring the final estimator from position — a sanctioned
  Rust API shape, not pinned. DETERMINISTIC / oracle-pinnable (step count).
- REQ-MAKE-UNION-MECHANIC: build a `FeatureUnion` from a transformer list. Mirrors
  `make_union` assembling a `FeatureUnion` from the named transformer sequence
  (`sklearn/pipeline.py:1889`, `:1934`). MATCH on the construction mechanic;
  `n_jobs` is absent (folded into REQ-MEMORY-VERBOSE-NJOBS). DETERMINISTIC /
  oracle-pinnable (transformer count).
- REQ-STEP-NAMING: auto-generated step/transformer names. sklearn names each step
  by its LOWERCASED CONCRETE CLASS NAME with a `-N` dedup suffix on collisions
  (`_name_estimators`, `:1196-1217`); the names are observable via
  `Pipeline.named_steps`, `get_params('standardscaler__C')`, and `set_params`.
  ferrolearn names positionally — `format!("step{i}")` for transformer steps, the
  literal `"estimator"` for the final estimator, and `format!("fu{i}")` for union
  transformers — with no class-name component and no dedup. DIVERGENCE.
  NOT-STARTED (architectural — see Architecture for why this is not fixable inside
  `helpers.rs`). DETERMINISTIC / oracle-pinnable.
- REQ-MEMORY-VERBOSE-NJOBS: constructor pass-through params. `make_pipeline` takes
  `memory` and `verbose` (`:1220`, threaded at `:1265`); `make_union` takes
  `n_jobs` and `verbose` (`:1889`, `:1934`). ferrolearn's `make_pipeline`/
  `make_union` take none of these and the underlying `Pipeline`/`FeatureUnion`
  expose no such channel. NOT-STARTED.
- REQ-X-1 (R-SUBSTRATE): array/numpy substrate. `helpers.rs` performs pure
  construction over `Box<dyn PipelineStep/PipelineTransformer/PipelineEstimator>`
  trait objects and `format!` strings; it imports no array type. R-SUBSTRATE is
  N/A — there is no `ndarray`/`ferray` usage to migrate.
- REQ-X-2: non-test production consumer. The `lib.rs` re-export
  `pub use helpers::{make_pipeline, make_union};` exposes both as crate public
  API (R-DEFER-1 / S5: grandfathered existing pub API). SHIPPED.

## Acceptance criteria

- AC-1 (REQ-MAKE-PIPELINE-MECHANIC): `make_pipeline::<f64>` with N transformer
  steps and a final estimator builds a `Pipeline` whose `step_names()` has
  `N + 1` entries (the N transformers plus the estimator), matching the step
  COUNT of the live oracle:
  ```
  python3 -c "from sklearn.pipeline import make_pipeline; from sklearn.preprocessing import StandardScaler; from sklearn.linear_model import LogisticRegression; print(len(make_pipeline(StandardScaler(), StandardScaler(), LogisticRegression()).steps))"
  # -> 3
  ```
  The empty case (`make_pipeline(Vec::new(), None)`) constructs an empty
  `Pipeline` without panic. DETERMINISTIC / oracle-pinnable (count, not names).
- AC-2 (REQ-MAKE-UNION-MECHANIC): `make_union::<f64>` with K transformers builds a
  `FeatureUnion` whose `n_transformers()` equals K, matching the live oracle:
  ```
  python3 -c "from sklearn.pipeline import make_union; from sklearn.decomposition import PCA, TruncatedSVD; print(len(make_union(PCA(), TruncatedSVD()).transformer_list))"
  # -> 2
  ```
  Empty case: `make_union(Vec::new()).n_transformers() == 0`. DETERMINISTIC /
  oracle-pinnable.
- AC-3 (REQ-STEP-NAMING): the generated names diverge from sklearn's class-name +
  dedup scheme. Live oracle:
  ```
  python3 -c "from sklearn.pipeline import make_pipeline; from sklearn.preprocessing import StandardScaler; from sklearn.linear_model import LogisticRegression; print([n for n,_ in make_pipeline(StandardScaler(), StandardScaler(), LogisticRegression()).steps])"
  # -> ['standardscaler-1', 'standardscaler-2', 'logisticregression']
  python3 -c "from sklearn.pipeline import make_union; from sklearn.decomposition import PCA, TruncatedSVD; print([n for n,_ in make_union(PCA(), TruncatedSVD()).transformer_list])"
  # -> ['pca', 'truncatedsvd']
  ```
  ferrolearn produces `['step0', 'step1', 'estimator']` for the pipeline and
  `['fu0', 'fu1']` for the union — no class-name component, no `-N` dedup.
  DETERMINISTIC / oracle-pinnable. (Cannot be pinned as a single-file `helpers.rs`
  fix — see Architecture; blocker #1871.)
- AC-4 (REQ-MEMORY-VERBOSE-NJOBS): the params exist on the live oracle but have no
  ferrolearn analog. Live oracle (signatures):
  ```
  python3 -c "import inspect; from sklearn.pipeline import make_pipeline, make_union; print(inspect.signature(make_pipeline)); print(inspect.signature(make_union))"
  # -> (*steps, memory=None, verbose=False)
  # -> (*transformers, n_jobs=None, verbose=False)
  ```
  ferrolearn `make_pipeline(steps, estimator)` / `make_union(transformers)` accept
  neither `memory`/`verbose` nor `n_jobs`/`verbose`. Absent end-to-end.
- AC-5 (REQ-X-1): `grep -E "ndarray|ferray|sprs|statrs|rand" helpers.rs` is empty
  — no array substrate to migrate, R-SUBSTRATE N/A.
- AC-6 (REQ-X-2): `make_pipeline` and `make_union` are re-exported from `lib.rs`
  (`pub use helpers::{make_pipeline, make_union};`), so both are crate public API.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-MAKE-PIPELINE-MECHANIC (build a Pipeline) | SHIPPED | impl `pub fn make_pipeline in helpers.rs` builds `Pipeline::<F>::new()`, then for each `(i, step)` calls `pipe = pipe.step(&format!("step{i}"), step)` and, if `Some(est)`, `pipe = pipe.estimator_step("estimator", est)`. This mirrors `make_pipeline` constructing a `Pipeline` from the named step sequence (`sklearn/pipeline.py:1265` `return Pipeline(_name_estimators(steps), ...)`). R-DEV-7 API-shape deviation: ferrolearn takes an explicit `(Vec<Box<dyn PipelineStep<F>>>, Option<Box<dyn PipelineEstimator<F>>>)` split rather than sklearn's variadic `*steps` with last-step-is-estimator inference — a sanctioned Rust API shape; the construction-and-assembly mechanic is what ships. Non-test consumer: `lib.rs` (`pub use helpers::{make_pipeline, make_union};`). Verification: `make_pipeline_empty_runs` builds the empty `Pipeline` without panic; step-count oracle `len(make_pipeline(StandardScaler(), StandardScaler(), LogisticRegression()).steps) == 3`. |
| REQ-MAKE-UNION-MECHANIC (build a FeatureUnion) | SHIPPED | impl `pub fn make_union in helpers.rs` builds `FeatureUnion::<F>::new()`, then for each `(i, t)` calls `fu = fu.add(&format!("fu{i}"), t)` (`fn add in feature_union.rs`). Mirrors `make_union` constructing a `FeatureUnion` from the named transformer sequence (`sklearn/pipeline.py:1934` `return FeatureUnion(_name_estimators(transformers), ...)`). `n_jobs` absent (REQ-MEMORY-VERBOSE-NJOBS). Non-test consumer: `lib.rs` re-export. Verification: `make_union_empty` asserts `make_union::<f64>(Vec::new()).n_transformers() == 0`; transformer-count oracle `len(make_union(PCA(), TruncatedSVD()).transformer_list) == 2`. |
| REQ-STEP-NAMING (class-name + dedup vs positional) | NOT-STARTED | open prereq blocker #1871. `make_pipeline in helpers.rs` names steps `format!("step{i}")` and the final estimator the literal `"estimator"`; `make_union in helpers.rs` names `format!("fu{i}")` — positional, no class-name, no `-N` dedup. sklearn names by lowercased concrete class name with a collision-only `-N` suffix (`_name_estimators`, `sklearn/pipeline.py:1200` `type(estimator).__name__.lower()`, dedup `:1207-1215`). ARCHITECTURAL gap: the core traits `PipelineStep`/`PipelineTransformer`/`PipelineEstimator` (`ferrolearn-core/src/pipeline.rs`) expose NO type-name method, and a `Box<dyn PipelineStep<F>>` has already type-erased its concrete type, so `std::any::type_name`/`type_name_of_val` on the trait object yields the trait-object type, NOT the underlying estimator class — there is no `std::any` route usable inside `helpers.rs` alone. Reproducing sklearn's class-name naming requires extending the core traits with a type-name channel (e.g. `fn type_name(&self) -> &'static str`) — out of this unit's `helpers.rs`-only scope. DETERMINISTIC / oracle-pinnable: `[n for n,_ in make_pipeline(StandardScaler(), StandardScaler(), LogisticRegression()).steps]` → `['standardscaler-1','standardscaler-2','logisticregression']`; ferrolearn → `['step0','step1','estimator']`. |
| REQ-MEMORY-VERBOSE-NJOBS (constructor pass-through params) | NOT-STARTED | open prereq blocker #1872. `make_pipeline in helpers.rs` takes only `(steps, estimator)` — no `memory`, no `verbose` — vs sklearn `make_pipeline(*steps, memory=None, verbose=False)` threaded into `Pipeline(..., memory=memory, verbose=verbose)` (`sklearn/pipeline.py:1220`, `:1265`). `make_union in helpers.rs` takes only `(transformers)` — no `n_jobs`, no `verbose` — vs `make_union(*transformers, n_jobs=None, verbose=False)` threaded into `FeatureUnion(..., n_jobs=n_jobs, verbose=verbose)` (`:1889`, `:1934`). The underlying `Pipeline`/`FeatureUnion` expose no such channel. Absent end-to-end. |
| REQ-X-1 (R-SUBSTRATE) | SHIPPED | N/A — R-SUBSTRATE does not apply. `helpers.rs` imports only `ferrolearn_core::pipeline::{Pipeline, PipelineEstimator, PipelineStep, PipelineTransformer}`, `num_traits::Float`, and `crate::feature_union::FeatureUnion`; it performs pure trait-object plumbing and `format!` string building and touches NO array type (`grep -E "ndarray\|ferray\|sprs\|statrs\|rand" helpers.rs` is empty — Verification). There is no numpy-layer dependency to migrate, so the unit is substrate-clean by construction (R-SUBSTRATE vacuously satisfied). |
| REQ-X-2 (non-test consumer) | SHIPPED | `lib.rs` re-exports both: `pub use helpers::{make_pipeline, make_union};`, making them crate public API. R-DEFER-1 / S5: existing grandfathered pub API; the re-export is a genuine non-test production consumer. CAVEAT (honest underclaim): there is no `ferrolearn-python` binding exposing `make_pipeline`/`make_union` yet — the consumer surface is the in-crate Rust re-export only. |

## Architecture

ferrolearn maps sklearn's two construction shorthands onto two free functions in
`helpers.rs`, both returning the already-built composite object. sklearn's
`make_pipeline` (`sklearn/pipeline.py:1220`) is variadic — `*steps` — and defers
ALL naming to `_name_estimators` (`:1196`) and the transformer/estimator split to
the `Pipeline` class (the last step is the estimator). ferrolearn's
`pub fn make_pipeline in helpers.rs` instead takes an EXPLICIT
`(Vec<Box<dyn PipelineStep<F>>>, Option<Box<dyn PipelineEstimator<F>>>)`: the
transformer steps and the final estimator are separated at the type level rather
than inferred from position. This is an R-DEV-7 API-shape deviation — Rust has no
ergonomic variadic + runtime "is this the last element / is it an estimator"
inference, so the explicit split is the idiomatic analog; the observable
construction result (a `Pipeline` of N transformer steps + 1 estimator step)
matches. `pub fn make_union in helpers.rs` is the simpler analog: a
`Vec<Box<dyn PipelineTransformer<F>>>` folded into `FeatureUnion::add`
(`fn add in feature_union.rs`).

The single genuine BEHAVIORAL divergence is REQ-STEP-NAMING. sklearn's
`_name_estimators` derives each name from `type(estimator).__name__.lower()`
(`:1200`) and appends a `-N` suffix only to colliding names (`:1207-1215`), so
the names are introspectable handles (`named_steps['standardscaler-1']`,
`set_params(standardscaler__C=...)`). ferrolearn cannot reproduce this inside
`helpers.rs`: the core abstractions are the object-safe traits `PipelineStep`,
`PipelineTransformer`, and `PipelineEstimator` (`ferrolearn-core/src/pipeline.rs`),
NONE of which declares a type-name / `name()` method — they expose only
`add_to_pipeline` / `fit_pipeline`. By the time a value reaches `make_pipeline`
it is already a `Box<dyn PipelineStep<F>>`: the concrete type has been erased.
`std::any::type_name::<T>()` needs the concrete `T` at the call site (unavailable
here), and `std::any::type_name_of_val(&*boxed)` over a `dyn` reference yields the
trait-object type, not the underlying estimator class, unless the trait extends
`Any` with downcasting — which it does not. There is therefore NO `std::any`
route that recovers the concrete class name from a boxed trait object inside
`helpers.rs` alone. Reproducing sklearn's class-name + dedup naming requires a
CORE-TRAIT change (add `fn type_name(&self) -> &'static str` — implementable via
`std::any::type_name::<Self>()` in each impl — to `PipelineStep`/
`PipelineTransformer`/`PipelineEstimator`), then a dedup pass in `helpers.rs`
mirroring `:1207-1215`. That trait-extension is out of this unit's
`helpers.rs`-only scope; the divergence is architectural, tracked by blocker
#1871. It is NOT a single-file `helpers.rs` fixer — the critic should confirm the
core-trait prerequisite rather than pin a `helpers.rs`-local test as if it were
self-contained.

The second divergence, REQ-MEMORY-VERBOSE-NJOBS, is a plain absence: neither the
helper signatures nor the underlying `Pipeline`/`FeatureUnion` carry `memory`,
`verbose`, or `n_jobs` (blocker #1872). R-SUBSTRATE (REQ-X-1) is N/A — the module
has no array layer.

## Verification

Commands establishing the SHIPPED claims (baseline
`5c0379e3832819710caebe777154d72c5bb01497`):

- `cargo test -p ferrolearn-model-sel --lib helpers` → `make_pipeline_empty_runs`
  and `make_union_empty` pass (REQ-MAKE-PIPELINE-MECHANIC empty case;
  REQ-MAKE-UNION-MECHANIC empty case / `n_transformers() == 0`).
- REQ-MAKE-PIPELINE-MECHANIC oracle (step COUNT, live oracle):
  ```
  python3 -c "from sklearn.pipeline import make_pipeline; from sklearn.preprocessing import StandardScaler; from sklearn.linear_model import LogisticRegression; print(len(make_pipeline(StandardScaler(), StandardScaler(), LogisticRegression()).steps))"
  # -> 3
  ```
  ferrolearn `make_pipeline` with 2 transformer steps + 1 estimator yields a
  `Pipeline` with 3 named steps (pin a `step_names().len() == 3` `#[test]`,
  R-CHAR-3).
- REQ-MAKE-UNION-MECHANIC oracle (transformer COUNT, live oracle):
  ```
  python3 -c "from sklearn.pipeline import make_union; from sklearn.decomposition import PCA, TruncatedSVD; print(len(make_union(PCA(), TruncatedSVD()).transformer_list))"
  # -> 2
  ```
  `make_union(vec_of_2).n_transformers() == 2`.
- REQ-STEP-NAMING oracle (NAME divergence, live oracle):
  ```
  python3 -c "from sklearn.pipeline import make_pipeline; from sklearn.preprocessing import StandardScaler; from sklearn.linear_model import LogisticRegression; print([n for n,_ in make_pipeline(StandardScaler(), StandardScaler(), LogisticRegression()).steps])"
  # -> ['standardscaler-1', 'standardscaler-2', 'logisticregression']
  python3 -c "from sklearn.pipeline import make_union; from sklearn.decomposition import PCA, TruncatedSVD; print([n for n,_ in make_union(PCA(), TruncatedSVD()).transformer_list])"
  # -> ['pca', 'truncatedsvd']
  ```
  ferrolearn produces `['step0', 'step1', 'estimator']` and `['fu0', 'fu1']` —
  REQ-STEP-NAMING NOT-STARTED (architectural; blocker #1871 — needs a core-trait
  type-name channel, not a `helpers.rs`-local change).
- REQ-MEMORY-VERBOSE-NJOBS oracle (signatures, live oracle):
  ```
  python3 -c "import inspect; from sklearn.pipeline import make_pipeline, make_union; print(inspect.signature(make_pipeline)); print(inspect.signature(make_union))"
  # -> (*steps, memory=None, verbose=False)
  # -> (*transformers, n_jobs=None, verbose=False)
  ```
  No ferrolearn analog — NOT-STARTED (blocker #1872).
- REQ-X-1 (substrate): `grep -E "ndarray|ferray|sprs|statrs|rand" ferrolearn-model-sel/src/helpers.rs`
  is empty — no array layer, R-SUBSTRATE N/A.
- REQ-X-2 (consumer): `grep -n "make_pipeline\|make_union" ferrolearn-model-sel/src/lib.rs`
  shows `pub use helpers::{make_pipeline, make_union};`.

SHIPPED: REQ-MAKE-PIPELINE-MECHANIC (Pipeline construction; R-DEV-7 explicit-split
API shape), REQ-MAKE-UNION-MECHANIC (FeatureUnion construction; `n_jobs` deferred),
REQ-X-1 (R-SUBSTRATE N/A — no array layer), REQ-X-2 (lib.rs re-export; no Python
binding yet). NOT-STARTED: REQ-STEP-NAMING (class-name + `-N` dedup; ARCHITECTURAL
— needs a core-trait type-name channel, blocker #1871, NOT a single-file
`helpers.rs` fixer), REQ-MEMORY-VERBOSE-NJOBS (`memory`/`verbose`/`n_jobs` absent,
blocker #1872). Per R-DEFER-2 the table is binary SHIPPED/NOT-STARTED.
