# ColumnTransformer / make_column_transformer — per-column-subset composition meta-transformer

<!--
tier: 3-component
status: shipped-partial
baseline-commit: dba9a076
upstream: scikit-learn 1.5.2
upstream-paths:
  - sklearn/compose/_column_transformer.py  # ColumnTransformer(TransformerMixin, _BaseComposition) (:59); __init__(transformers, *, remainder="drop", sparse_threshold=0.3, n_jobs=None, transformer_weights=None, verbose=False, verbose_feature_names_out=True, force_int_remainder_cols=True) (:290-309); _parameter_constraints remainder {'drop','passthrough'} or estimator (:277-281), sparse_threshold Interval(Real,0,1) (:282); fit_transform calls _call_func_on_transformers(_fit_transform_one) IN LIST ORDER then _hstack(list(Xs)) (:976-1006); transform (:1008-1089); _validate_remainder remaining = sorted(set(range(n_features_in_)) - cols) (:546-547) -> _remainder=("remainder", self.remainder, remainder_cols) (:550) appended LAST via chain in _iter (:460-462); _hstack np.hstack(Xs) dense / sparse.hstack when sparse_output_ (:1091-1200), sparse_output_ = density < sparse_threshold (:998); _get_feature_name_out_for_transformer (:584) + verbose prefix f"{name}__{i}" (:662); get_feature_names_out (:599); named_transformers_ Bunch (:574-582), transformers_ (:694-726); make_column_transformer(*transformers, remainder="drop", sparse_threshold=0.3, ...) (:1334) -> _get_transformer_list -> _name_estimators lowercased-class-name dedup (:1326,:1456-1465); make_column_selector (:1468).
ferrolearn-module: ferrolearn-preprocess/src/column_transformer.rs
parity-ops: ColumnTransformer, make_column_transformer
crosslink-issue: 1434
-->

## Summary

scikit-learn's `ColumnTransformer` (`_column_transformer.py:59`) is a composition
meta-transformer: it applies a list of `(name, transformer, columns)` triples each to
its own column subset and horizontally concatenates the per-transformer outputs **in
list order**, then appends the uncovered "remainder" columns LAST (`fit_transform`
→ `_call_func_on_transformers(...)` → `_hstack(list(Xs))`, `:976-1006`).
`make_column_transformer` (`:1334`) is the keyword-free builder that auto-names each
step by its lowercased class name.

`ferrolearn-preprocess/src/column_transformer.rs` ships `ColumnTransformer`
(`Fit<Array2<f64>,()>`, `column_transformer.rs:236`) holding
`Vec<(String, Box<dyn PipelineTransformer<f64>>, ColumnSelector)>` + a `Remainder`
policy; it resolves each `ColumnSelector::Indices`, fits each sub-transformer on its
`select_columns(x, indices)` sub-matrix via `fit_pipeline`, computes
`remainder_indices = (0..n).filter(!covered)` in ASCENDING order, and at transform
time (`column_transformer.rs:371`) hstacks the per-transformer outputs in
REGISTRATION ORDER then appends the remainder LAST iff `Remainder::Passthrough`.
`make_column_transformer` (`column_transformer.rs:465`) auto-names steps
`"transformer-0"`, `"transformer-1"`, … . It is re-exported at the crate boundary
(`lib.rs:128-129`).

**This is a DETERMINISTIC COMPOSITION unit.** What this translation unit OWNS and
verifies is the COMPOSITION: column routing, per-transformer output ordering,
remainder handling (drop / passthrough, ascending), and overlapping-column fan-out.
The sub-transformer VALUES (StandardScaler population-std z-scores, MinMaxScaler
[0,1] scaling) come from those units' own parity proofs
(`.design/preprocess/standard_scaler.md`, `.design/preprocess/min_max_scaler.md`) —
both are independently parity-verified. Given matching sub-transformers, the combined
ferrolearn output is bit-for-bit equal to sklearn's on the oracle fixtures (Probe A/B/C).

This is a **shipped-partial** unit: **3 SHIPPED** (REQ-1 column routing + output
ordering + drop/passthrough + overlapping fan-out + combined VALUES; REQ-2
`make_column_transformer` builds the CT; REQ-3 error/parameter contracts) /
**8 NOT-STARTED** (REQ-4 non-index `ColumnSelector`s + `make_column_selector`;
REQ-5 estimator-remainder + step-level `'drop'`/`'passthrough'`; REQ-6
`sparse_threshold`/sparse output + `transformer_weights` + `n_jobs`/`verbose`; REQ-7
`get_feature_names_out`/`verbose_feature_names_out` + sklearn class-name auto-naming;
REQ-8 `named_transformers_`/`transformers_`/`feature_names_in_` fitted-attr surface;
REQ-9 generic `F`; REQ-10 PyO3 binding; REQ-11 ferray substrate).

## Probes (live sklearn oracle, scikit-learn 1.5.2)

All values below are live oracle output captured from `/tmp` at baseline `dba9a076`;
ferrolearn values are the behavior of the shipped code on the identical inputs,
captured by constructing the equivalent `ColumnTransformer::new(...)` with ferrolearn
`StandardScaler`/`MinMaxScaler` sub-transformers (both ddof=0 / population-std, which
matches sklearn — see those units). Re-run with the commands shown (R-CHAR-3 —
expected values are the live sklearn call, never copied from the ferrolearn side).

```bash
# ---------------------------------------------------------------------------
# PROBE A (REQ-1) — output ordering + remainder='passthrough' + combined VALUES
#   X = [[1,10,100,5],[2,20,200,6],[3,30,300,7]]
#   transformers: ('std', StandardScaler, [0,1]), ('mm', MinMaxScaler, [2])
#   list order: std cols [0,1] -> 2 cols, mm col [2] -> 1 col, then remainder col 3 LAST
# ---------------------------------------------------------------------------
python3 -c "
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
X=np.array([[1,10,100,5],[2,20,200,6],[3,30,300,7]],float)
ct=ColumnTransformer([('std',StandardScaler(),[0,1]),('mm',MinMaxScaler(),[2])],remainder='passthrough')
print('shape', ct.fit_transform(X).shape); print(np.round(ct.fit_transform(X),8))"
#   sklearn shape (3, 4)
#   sklearn values:
#     [[-1.22474487 -1.22474487 0.0 5.0]
#      [ 0.          0.          0.5 6.0]
#      [ 1.22474487  1.22474487 1.0 7.0]]
#   (std z-scores cols 0,1 | mm col 2 in [0,1] | remainder col 3 = [5,6,7] appended LAST)
#
#   ferrolearn ColumnTransformer::new([("std",StandardScaler,[0,1]),("mm",MinMaxScaler,[2])], Passthrough)
#     -> shape (3,4); values
#        [[-1.224744871391589, -1.224744871391589, 0.0, 5.0], ... ]   (EXACT MATCH ~1e-15)

# ---------------------------------------------------------------------------
# PROBE B (REQ-1) — remainder='drop' drops the uncovered column
# ---------------------------------------------------------------------------
python3 -c "
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
X=np.array([[1,10,100,5],[2,20,200,6],[3,30,300,7]],float)
ct=ColumnTransformer([('std',StandardScaler(),[0,1]),('mm',MinMaxScaler(),[2])],remainder='drop')
print('shape', ct.fit_transform(X).shape); print(np.round(ct.fit_transform(X),8))"
#   sklearn shape (3, 3)   (col 3 dropped; no remainder block)
#   ferrolearn Remainder::Drop -> shape (3,3), identical values to Probe A first 3 cols. MATCH.

# ---------------------------------------------------------------------------
# PROBE C (REQ-1) — OVERLAPPING columns: col 0 feeds BOTH transformers
# ---------------------------------------------------------------------------
python3 -c "
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
X=np.array([[1,10,100,5],[2,20,200,6],[3,30,300,7]],float)
ct=ColumnTransformer([('a',StandardScaler(),[0]),('b',MinMaxScaler(),[0])],remainder='drop')
print('shape', ct.fit_transform(X).shape); print(np.round(ct.fit_transform(X),8))"
#   sklearn shape (3, 2):
#     [[-1.22474487 0.0],[0.0 0.5],[1.22474487 1.0]]   (col 0 emitted by BOTH a and b)
#   ferrolearn ColumnSelector::Indices(vec![0]) on both -> shape (3,2), identical values. MATCH.

# ---------------------------------------------------------------------------
# PROBE D (REQ-2 / REQ-7) — make_column_transformer auto-naming
# ---------------------------------------------------------------------------
python3 -c "
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
mct=make_column_transformer((StandardScaler(),[0,1]),(MinMaxScaler(),[2]))
print('sklearn names', [n for n,_,_ in mct.transformers])"
#   sklearn names ['standardscaler', 'minmaxscaler']   (lowercased class name, deduped)
#   ferrolearn make_column_transformer names ['transformer-0','transformer-1']  (NAMING GAP -> REQ-7)

# ---------------------------------------------------------------------------
# PROBE E (REQ-7) — get_feature_names_out + verbose prefix (NOT in ferrolearn)
# ---------------------------------------------------------------------------
python3 -c "
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
X=np.array([[1,10,100,5],[2,20,200,6],[3,30,300,7]],float)
ct=ColumnTransformer([('std',StandardScaler(),[0,1]),('mm',MinMaxScaler(),[2])],remainder='passthrough')
ct.fit(X); print(list(ct.get_feature_names_out()))"
#   sklearn ['std__x0','std__x1','mm__x2','remainder__x3']   (verbose_feature_names_out prefix name__feat)
#   ferrolearn has NO get_feature_names_out (REQ-7 NOT-STARTED).
```

**Composition parity summary (REQ-1 — combined output VALUES match the oracle).**

| probe | sklearn shape | sklearn values (rounded) | ferrolearn | parity |
|---|---|---|---|---|
| A passthrough | `(3,4)` | `[[-1.2247,-1.2247,0,5],…,[1.2247,1.2247,1,7]]` | `(3,4)`, identical | MATCH (~1e-15) |
| B drop | `(3,3)` | `[[-1.2247,-1.2247,0],…]` | `(3,3)`, identical | MATCH |
| C overlap | `(3,2)` | `[[-1.2247,0],[0,0.5],[1.2247,1]]` | `(3,2)`, identical | MATCH |
| D make names | — | `['standardscaler','minmaxscaler']` | `['transformer-0','transformer-1']` | NAMING GAP (REQ-7) |

The composition (routing/ordering/remainder/overlap) is verified bit-for-bit; the
sub-transformer VALUES (`-1.22474487…` = population z-score, `0/0.5/1` = min-max)
are owned by the StandardScaler / MinMaxScaler units, which are independently
parity-verified.

## Requirements

- REQ-1: **column routing + output ORDERING + drop/passthrough + overlapping
  fan-out + combined VALUES** (SHIPPED). Each transformer is applied to its column
  subset and the per-transformer outputs are hstacked **in registration/list order**,
  then the uncovered remainder columns are appended LAST in ASCENDING original-column
  order when `Remainder::Passthrough` (dropped when `Remainder::Drop`), and a column
  may feed multiple transformers (each emits its output). This mirrors sklearn:
  `fit_transform` calls `_call_func_on_transformers(X, y, _fit_transform_one, …)` over
  the transformer list in order then `return self._hstack(list(Xs), n_samples=…)`
  (`_column_transformer.py:976-1006`); the remainder block is computed as
  `remaining = sorted(set(range(self.n_features_in_)) - cols)` (`:546-547`),
  stored as `self._remainder = ("remainder", self.remainder, remainder_cols)`
  (`:550`) and chained LAST in `_iter` (`:460-462`); dense `_hstack` is
  `np.hstack(Xs)` (`:1200`). ferrolearn's `impl Fit<Array2<f64>,()> for
  ColumnTransformer` (`column_transformer.rs:236`) builds
  `covered = ⋃ resolved_selectors`, `remainder_indices = (0..n_features)
  .filter(|c| !covered.contains(c))` (ASCENDING), and fits each transformer on
  `select_columns(x, &indices)` via `fit_pipeline`; `impl Transform<Array2<f64>> for
  FittedColumnTransformer` (`column_transformer.rs:371`) pushes each transformer's
  `transform_pipeline` output in order, then `if matches!(self.remainder,
  Remainder::Passthrough) && !self.remainder_indices.is_empty()` appends
  `select_columns(x, &self.remainder_indices)` LAST, and returns `hstack(&parts)`.
  Overlapping selections are supported because each transformer gets its own
  `select_columns` copy. Combined output VALUES match the live oracle bit-for-bit
  (Probe A/B/C). The composition is verified; sub-transformer VALUES come from the
  StandardScaler / MinMaxScaler units.

- REQ-2: **`make_column_transformer` builds the CT (auto-named)** (SHIPPED, scoped).
  sklearn's `make_column_transformer(*transformers, remainder="drop", …)`
  (`_column_transformer.py:1334`) wraps the `(transformer, columns)` pairs into a
  `ColumnTransformer` via `_get_transformer_list` → `_name_estimators` (`:1326`,
  `:1456-1465`). ferrolearn's `make_column_transformer(Vec<(Box<dyn
  PipelineTransformer<f64>>, ColumnSelector)>, Remainder) -> ColumnTransformer`
  (`column_transformer.rs:465`) enumerates the pairs and names them
  `format!("transformer-{i}")`, then defers to `ColumnTransformer::new`. The
  BUILD/compose behavior is faithful (the resulting CT routes/orders/stacks
  identically — Probe A built either way produces the same matrix). SCOPED: the
  AUTO-NAMING differs — sklearn names by the lowercased transformer class name
  (deduped with numeric suffixes, `standardscaler`/`minmaxscaler`, Probe D) while
  ferrolearn uses positional `transformer-N`. That naming gap is folded into REQ-7
  (it only matters via `named_transformers_` / `get_feature_names_out`, both absent).

- REQ-3: **error / parameter contracts (out-of-range index, transform ncols
  mismatch)** (SHIPPED, scoped). `ColumnSelector::resolve` returns
  `Err(InvalidParameter { name: "ColumnSelector::Indices", … })` for any
  `idx >= n_features` (`column_transformer.rs:70-86`), and `ColumnTransformer::fit`
  re-wraps it enriched with the step name (`name: format!("ColumnTransformer step
  '{name}'")`, `column_transformer.rs:260-266`) — resolving all selectors eagerly up
  front before fitting any transformer (mirrors sklearn validating columns in
  `_validate_column_callables` / `fit_transform` before fitting). At transform time
  `FittedColumnTransformer::transform` returns `Err(ShapeMismatch { … context:
  "FittedColumnTransformer::transform" })` when `x.ncols() != self.n_features_in`
  (`column_transformer.rs:388-395`), and `hstack` returns `ShapeMismatch` on row-count
  mismatch (`column_transformer.rs:148-154`). SCOPED to the dense `Array2<f64>` API:
  sklearn additionally constrains `remainder ∈ {'drop','passthrough'} or estimator`
  and `sparse_threshold ∈ [0,1]` (`_parameter_constraints` `:277-282`) — parameters
  ferrolearn does not have (REQ-5 / REQ-6).

- REQ-4: **non-index `ColumnSelector`s (str / slice / bool-mask / callable /
  `make_column_selector`)** (NOT-STARTED). sklearn's `columns` may be an int, str,
  slice, boolean mask, or a callable applied to the fitted dataframe
  (`_validate_column_callables` `:520`, `_get_column_indices`), and
  `make_column_selector` (`:1468`) builds a callable selecting by dtype / name regex.
  ferrolearn's `enum ColumnSelector` (`column_transformer.rs:53-61`) has the single
  variant `Indices(Vec<usize>)` — no string/label selection (ferrolearn has no
  dataframe column-name concept), no slice, no boolean mask, no callable, and no
  `make_column_selector`. NOT-STARTED on prereq blocker #1435.

- REQ-5: **`remainder` as an ESTIMATOR + step-level `'drop'`/`'passthrough'`
  transformer** (NOT-STARTED). sklearn allows `remainder` to be a fitted estimator
  applied to the uncovered columns (`__init__` `remainder="drop"` `:294`, doc
  `:108-110`, `_validate_remainder` stores `("remainder", self.remainder, cols)`
  `:550` and fits it like any step), and an individual transformer entry may itself be
  the string `'drop'` or `'passthrough'`. ferrolearn's `enum Remainder`
  (`column_transformer.rs:97-104`) is only `Drop` / `Passthrough` — there is NO
  estimator-remainder (the remainder columns are passed through verbatim, never fit by
  a transformer), and a step's transformer is always a `Box<dyn
  PipelineTransformer<f64>>` (no `'drop'`/`'passthrough'` step sentinel). NOT-STARTED
  on prereq blocker #1436.

- REQ-6: **`sparse_threshold` + sparse output + `transformer_weights` + `n_jobs` /
  `verbose`** (NOT-STARTED). sklearn computes `density = nnz / total` and sets
  `self.sparse_output_ = density < self.sparse_threshold` (`fit_transform`
  `:991-1000`), and `_hstack` returns `sparse.hstack(converted_Xs).tocsr()` when
  `sparse_output_` (`:1105-1120`); `__init__` also carries `sparse_threshold=0.3`,
  `n_jobs`, `transformer_weights`, `verbose` (`:290-309`). ferrolearn's `hstack`
  (`column_transformer.rs:133`) is dense-`Array2<f64>` only (no `sprs`/sparse path,
  no `sparse_threshold`), `ColumnTransformer` has no `transformer_weights` scaling,
  no `n_jobs` parallelism, and no `verbose` logging. NOT-STARTED on prereq blocker
  #1437.

- REQ-7: **`get_feature_names_out` + `verbose_feature_names_out` (`name__feature`) +
  sklearn class-name auto-naming in `make_column_transformer`** (NOT-STARTED).
  sklearn builds output feature names per transformer
  (`_get_feature_name_out_for_transformer` `:584`), and when
  `verbose_feature_names_out=True` prefixes each with `f"{name}__{i}"` (`:662`) via
  `get_feature_names_out` (`:599`) — e.g. `['std__x0','std__x1','mm__x2',
  'remainder__x3']` (Probe E); `make_column_transformer` names steps by lowercased
  class name (`_name_estimators` `:1326`, Probe D `['standardscaler','minmaxscaler']`).
  ferrolearn surfaces `transformer_names()` (`column_transformer.rs:353`) only — there
  is NO `get_feature_names_out`, NO `verbose_feature_names_out` prefixing, and
  `make_column_transformer` names positionally (`transformer-N`,
  `column_transformer.rs:472`) rather than by class name. NOT-STARTED on prereq
  blocker #1438.

- REQ-8: **`named_transformers_` / `transformers_` / `n_features_in_` /
  `feature_names_in_` fitted-attr surface** (NOT-STARTED). sklearn exposes
  `named_transformers_` (a `Bunch` of name→fitted-transformer, `:574-582`),
  `transformers_` (the fitted `(name, trans, cols)` list incl. the remainder entry,
  `:694-726`), `n_features_in_`, and `feature_names_in_`. ferrolearn's
  `FittedColumnTransformer` exposes `n_features_in()` (`column_transformer.rs:347`),
  `transformer_names()` (`:353`), and `remainder_indices()` (`:362`) — but there is NO
  `named_transformers_`-style map giving access to each FITTED sub-transformer, NO
  `transformers_` list with the remainder entry, and NO `feature_names_in_` (no
  dataframe column-name concept). NOT-STARTED on prereq blocker #1439.

- REQ-9: **generic `F` (currently `f64`-only)** (NOT-STARTED). Every type and impl in
  `column_transformer.rs` is hard-wired to `f64`: `Box<dyn PipelineTransformer<f64>>`,
  `Array2<f64>`, `impl Fit<Array2<f64>,()>` (`:236`), `impl Transform<Array2<f64>>`
  (`:371`), and `fn select_columns(x: &Array2<f64>, …)` / `fn hstack(matrices:
  &[Array2<f64>])`. Unlike the `<F: Float>` selectors elsewhere in the crate, the
  ColumnTransformer is not generic over the float type and cannot operate on f32.
  NOT-STARTED on prereq blocker #1440.

- REQ-10: **PyO3 binding** (NOT-STARTED). There is no `ColumnTransformer` /
  `make_column_transformer` CPython binding in `ferrolearn-python`
  (`grep -rln "ColumnTransformer\|column_transformer" ferrolearn-python/src` finds
  none), so the meta-transformer is unreachable from Python. NOT-STARTED on prereq
  blocker #1441.

- REQ-11: **ferray substrate** (NOT-STARTED). The transformer composes over
  `ndarray::Array2<f64>` / `Array1<f64>` (`x.column(j)`, `Array2::zeros`,
  `slice_mut`) rather than `ferray-core` arrays (R-SUBSTRATE-1/2). NOT-STARTED on
  prereq blocker #1442.

## Acceptance criteria

- AC-1 (REQ-1): `ColumnTransformer::new([("std",StandardScaler,[0,1]),
  ("mm",MinMaxScaler,[2])], Passthrough)` on Probe A's `X` yields a `(3,4)` matrix
  equal to `ColumnTransformer([...],remainder='passthrough').fit_transform(X)` to
  `|Δ| < 1e-8` (transformer outputs in list order, remainder col 3 = `[5,6,7]`
  appended LAST); the `Drop` variant yields `(3,3)` (Probe B); two transformers both
  selecting col 0 yield `(3,2)` (Probe C, overlapping fan-out). Pinned by in-module
  `test_basic_two_transformers_drop_remainder`, `test_remainder_drop`,
  `test_remainder_passthrough`, `test_overlapping_column_selections`,
  `test_passthrough_values_are_exact`, `test_remainder_indices_accessor`, plus the
  Probe A/B/C oracle gates.

- AC-2 (REQ-2): `make_column_transformer([(StandardScaler,[0,1]),(MinMaxScaler,[2])],
  Drop)` builds a CT that routes/orders/stacks identically to the explicit
  `ColumnTransformer::new` form, and `transformer_names() == ["transformer-0",
  "transformer-1"]`. Pinned by `test_make_column_transformer_auto_names`,
  `test_make_column_transformer_single`. (sklearn names `['standardscaler',
  'minmaxscaler']` — REQ-7.)

- AC-3 (REQ-3): a step selecting an out-of-range index (`Indices(vec![0,99])` on a
  4-col `X`) → `Err(InvalidParameter)` enriched with the step name; transforming a
  matrix whose `ncols != n_features_in` → `Err(ShapeMismatch)`. Pinned by
  `test_invalid_column_index_out_of_range`, `test_shape_mismatch_on_transform`.

- AC-4 (REQ-4): `ColumnSelector` accepts string labels / slices / boolean masks /
  callables and `make_column_selector(dtype_include=…)`; ferrolearn has only
  `Indices(Vec<usize>)`. NOT-STARTED.

- AC-5 (REQ-5): `ColumnTransformer(..., remainder=StandardScaler())` fits the
  remainder columns with an estimator, and a step transformer of `'passthrough'`
  passes its selected columns through; ferrolearn's `Remainder` is only
  `Drop`/`Passthrough` and a step is always a real transformer. NOT-STARTED.

- AC-6 (REQ-6): a CT producing >70%-sparse output returns a CSR matrix
  (`sparse_output_ = density < sparse_threshold`); `transformer_weights` scales a
  transformer's output; `n_jobs=-1` parallelizes. ferrolearn is dense-only with no
  weights / parallelism. NOT-STARTED.

- AC-7 (REQ-7): `ct.get_feature_names_out()` returns `['std__x0','std__x1','mm__x2',
  'remainder__x3']` (verbose prefix), and `make_column_transformer` names steps
  `['standardscaler','minmaxscaler']`; ferrolearn has no `get_feature_names_out` and
  names steps `transformer-N`. NOT-STARTED.

- AC-8 (REQ-8): `ct.named_transformers_['std']` returns the fitted StandardScaler and
  `ct.transformers_` includes the `('remainder', …)` entry; ferrolearn exposes
  `transformer_names()` / `remainder_indices()` but no fitted-sub-transformer map.
  NOT-STARTED.

- AC-9 (REQ-9): the composition operates on an `Array2<f32>` input; it is
  `f64`-hard-wired today. NOT-STARTED.

- AC-10 (REQ-10): a CPython `ColumnTransformer` binding composes transformers from
  Python; no such binding exists. NOT-STARTED.

- AC-11 (REQ-11): the composition operates on `ferray-core` arrays rather than
  `ndarray::Array2<f64>`. NOT-STARTED.

`## REQ status`

## REQ status

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (routing + ordering + drop/passthrough + overlap + combined VALUES) | SHIPPED | impl `impl Fit<Array2<f64>,()> for ColumnTransformer` (`pub fn fit in column_transformer.rs`, `:236`) builds `covered = ⋃ resolved_selectors` and `remainder_indices = (0..n_features).filter(|c| !covered.contains(c))` (ASCENDING), fitting each transformer on `select_columns(x, &indices)` via `fit_pipeline`; `impl Transform<Array2<f64>> for FittedColumnTransformer` (`pub fn transform in column_transformer.rs`, `:371`) pushes each `transform_pipeline` output in REGISTRATION ORDER then appends `select_columns(x, &self.remainder_indices)` LAST iff `matches!(self.remainder, Remainder::Passthrough)` and returns `hstack(&parts)`. Mirrors sklearn `fit_transform` → `_call_func_on_transformers(... _fit_transform_one ...)` in list order then `self._hstack(list(Xs))` (`sklearn/compose/_column_transformer.py:976-1006`), remainder `remaining = sorted(set(range(n_features_in_)) - cols)` (`:546-547`) chained LAST (`:460-462`), dense `np.hstack(Xs)` (`:1200`); overlap supported because each step gets its own `select_columns` copy. Non-test consumer: boundary re-export `pub use column_transformer::{ColumnSelector, ColumnTransformer, FittedColumnTransformer, Remainder, make_column_transformer}` (`lib.rs:128-129`) + `impl PipelineTransformer<f64>/FittedPipelineTransformer<f64>` (used by `ferrolearn_core::pipeline::Pipeline`). Verification: Probe A passthrough `(3,4)` `[[-1.22474487,-1.22474487,0,5],…]`, Probe B drop `(3,3)`, Probe C overlap `(3,2)` — all bit-for-bit (`~1e-15`) vs sklearn; `cargo test -p ferrolearn-preprocess --lib column_transformer` → 22 tests green (`test_basic_two_transformers_drop_remainder`, `test_remainder_drop`, `test_remainder_passthrough`, `test_overlapping_column_selections`, `test_passthrough_values_are_exact`, `test_output_shape_partial_passthrough`, `test_remainder_indices_accessor`). Composition verified; sub-transformer VALUES owned by StandardScaler / MinMaxScaler units. |
| REQ-2 (`make_column_transformer` builds the CT) | SHIPPED | impl `pub fn make_column_transformer in column_transformer.rs` (`:465`) enumerates `(Box<dyn PipelineTransformer<f64>>, ColumnSelector)` pairs, names each `format!("transformer-{i}")`, and defers to `ColumnTransformer::new`, mirroring sklearn `make_column_transformer(*transformers, remainder="drop", …)` → `_get_transformer_list` → `ColumnTransformer(transformer_list, …)` (`sklearn/compose/_column_transformer.py:1334`, `:1456-1465`). Non-test consumer: boundary re-export (`lib.rs:128-129`). Verification: `cargo test -p ferrolearn-preprocess --lib column_transformer` → `test_make_column_transformer_auto_names` (`transformer_names() == ["transformer-0","transformer-1"]`), `test_make_column_transformer_single`; Probe A built via `make_column_transformer` yields the same matrix as the explicit form. SCOPED: AUTO-NAMING differs — sklearn `_name_estimators` lowercased-class-name (`:1326`, Probe D `['standardscaler','minmaxscaler']`) vs ferrolearn positional `transformer-N`; naming gap folded into REQ-7. |
| REQ-3 (error / parameter contracts) | SHIPPED | impl `ColumnSelector::resolve` (`column_transformer.rs:70-86`) returns `Err(InvalidParameter { name: "ColumnSelector::Indices", reason: "column index {idx} is out of range …" })` for `idx >= n_features`; `ColumnTransformer::fit` re-wraps it as `InvalidParameter { name: format!("ColumnTransformer step '{name}'"), … }` resolving ALL selectors eagerly before fitting (`column_transformer.rs:258-268`) — mirroring sklearn validating columns in `fit_transform` before fitting any step (`sklearn/compose/_column_transformer.py:962-969`). `FittedColumnTransformer::transform` returns `Err(ShapeMismatch { …, context: "FittedColumnTransformer::transform" })` on `x.ncols() != n_features_in` (`column_transformer.rs:388-395`), and `hstack` returns `ShapeMismatch` on row-count mismatch (`:148-154`). Non-test consumer: boundary re-export (`lib.rs:128-129`). Verification: `cargo test -p ferrolearn-preprocess --lib column_transformer` → `test_invalid_column_index_out_of_range`, `test_shape_mismatch_on_transform`. SCOPED to the dense `Array2<f64>` API: sklearn's `remainder ∈ {'drop','passthrough'} or estimator` / `sparse_threshold ∈ [0,1]` constraints (`_parameter_constraints` `:277-282`) are parameters ferrolearn lacks (REQ-5 / REQ-6). |
| REQ-4 (non-index `ColumnSelector`s + `make_column_selector`) | NOT-STARTED | open prereq blocker #1435. sklearn `columns` may be int/str/slice/bool-mask/callable (`_validate_column_callables` `sklearn/compose/_column_transformer.py:520`) and `make_column_selector` (`:1468`) selects by dtype / name regex. ferrolearn's `enum ColumnSelector` (`column_transformer.rs:53-61`) has only `Indices(Vec<usize>)` — no string/slice/bool/callable selection and no `make_column_selector`. |
| REQ-5 (estimator-remainder + step-level `'drop'`/`'passthrough'`) | NOT-STARTED | open prereq blocker #1436. sklearn `remainder` may be an estimator fit on the uncovered columns (`__init__` `:294`, doc `:108-110`, `_validate_remainder` `:550`) and a step transformer may be the string `'drop'`/`'passthrough'`. ferrolearn's `enum Remainder` (`column_transformer.rs:97-104`) is only `Drop`/`Passthrough` (remainder columns are passed through verbatim, never fit by an estimator) and a step is always a real `Box<dyn PipelineTransformer<f64>>`. |
| REQ-6 (`sparse_threshold` + sparse output + `transformer_weights` + `n_jobs`/`verbose`) | NOT-STARTED | open prereq blocker #1437. sklearn sets `sparse_output_ = density < sparse_threshold` (`fit_transform` `sklearn/compose/_column_transformer.py:991-1000`) and `_hstack` returns `sparse.hstack(...).tocsr()` when sparse (`:1105-1120`); `__init__` carries `sparse_threshold`, `n_jobs`, `transformer_weights`, `verbose` (`:290-309`). ferrolearn's `hstack` (`column_transformer.rs:133`) is dense-`Array2<f64>` only — no sparse path / `sparse_threshold`, no `transformer_weights`, no `n_jobs`, no `verbose`. |
| REQ-7 (`get_feature_names_out` + `verbose_feature_names_out` + class-name auto-naming) | NOT-STARTED | open prereq blocker #1438. sklearn builds per-transformer names (`_get_feature_name_out_for_transformer` `sklearn/compose/_column_transformer.py:584`), verbose-prefixes `f"{name}__{i}"` (`:662`) via `get_feature_names_out` (`:599`) → `['std__x0','std__x1','mm__x2','remainder__x3']` (Probe E), and `make_column_transformer` names by lowercased class name (`_name_estimators` `:1326`, Probe D). ferrolearn exposes `transformer_names()` (`column_transformer.rs:353`) only — no `get_feature_names_out`, no `name__feat` prefixing, positional `transformer-N` naming (`:472`). |
| REQ-8 (`named_transformers_` / `transformers_` / `feature_names_in_` surface) | NOT-STARTED | open prereq blocker #1439. sklearn exposes `named_transformers_` Bunch (`sklearn/compose/_column_transformer.py:574-582`), `transformers_` incl. the remainder entry (`:694-726`), `n_features_in_`, `feature_names_in_`. ferrolearn's `FittedColumnTransformer` exposes `n_features_in()` (`column_transformer.rs:347`), `transformer_names()` (`:353`), `remainder_indices()` (`:362`) — no fitted-sub-transformer map, no `transformers_` list with the remainder entry, no `feature_names_in_`. |
| REQ-9 (generic `F`) | NOT-STARTED | open prereq blocker #1440. The whole module is `f64`-hard-wired: `Box<dyn PipelineTransformer<f64>>`, `impl Fit<Array2<f64>,()>` (`column_transformer.rs:236`), `impl Transform<Array2<f64>>` (`:371`), `fn select_columns(x: &Array2<f64>, …)` (`:113`), `fn hstack(matrices: &[Array2<f64>])` (`:133`) — not generic over `F: Float`, cannot operate on f32. |
| REQ-10 (PyO3 binding) | NOT-STARTED | open prereq blocker #1441. No CPython binding for `ColumnTransformer`/`make_column_transformer` exists in `ferrolearn-python/src` (`grep -rln "ColumnTransformer\|column_transformer" ferrolearn-python/src` → none), so the meta-transformer is unreachable from Python. |
| REQ-11 (ferray substrate) | NOT-STARTED | open prereq blocker #1442. The composition + `select_columns`/`hstack` helpers compute over `ndarray::Array2<f64>`/`Array1<f64>` (`x.column(j)`, `Array2::zeros`, `slice_mut`), not `ferray-core` arrays (R-SUBSTRATE-1/2). |

## Architecture

**ferrolearn (existing, compiled at baseline `dba9a076`).** `column_transformer.rs`
exposes the composition meta-transformer plus two private helpers:

- `select_columns(x: &Array2<f64>, indices: &[usize]) -> Array2<f64>`
  (`column_transformer.rs:113`) — gathers the columns at `indices` into a new matrix
  in `indices` order (returns an `(nrows, 0)` matrix when `indices` is empty). Used by
  both `fit` (to build each sub-matrix) and `transform` (each step + the remainder
  block).
- `hstack(matrices: &[Array2<f64>]) -> Result<Array2<f64>, FerroError>`
  (`column_transformer.rs:133`) — dense horizontal concatenation; sums column counts,
  `ShapeMismatch`-checks row counts, and copies each block into a column slice. Mirrors
  the dense `np.hstack(Xs)` branch of sklearn `_hstack` (`:1200`); the sparse branch
  (`:1105-1120`) is absent (REQ-6).
- `enum ColumnSelector { Indices(Vec<usize>) }` (`:53-61`) with
  `fn resolve(&self, n_features) -> Result<Vec<usize>, FerroError>` (`:70-86`)
  validating every index `< n_features`. Single variant only (REQ-4).
- `enum Remainder { Drop, Passthrough }` (`:97-104`) — no estimator-remainder (REQ-5).
- `ColumnTransformer { transformers: Vec<(String, Box<dyn PipelineTransformer<f64>>,
  ColumnSelector)>, remainder: Remainder }` (`new`, `:213-230`). Its
  `Fit<Array2<f64>,()>` impl (`:236`) resolves all selectors eagerly (enriching
  out-of-range errors with the step name), computes
  `covered = HashSet<⋃ resolved>` then `remainder_indices = (0..n_features)
  .filter(!covered)` (ASCENDING), and fits each step on its `select_columns` sub-matrix
  via `fit_pipeline` (a `dummy_y` of zeros is supplied because the pipeline trait
  requires a `y`). It also impls `PipelineTransformer<f64>` (`:302`) delegating to
  `fit`.
- `FittedColumnTransformer { fitted_transformers: Vec<(String, Box<dyn
  FittedPipelineTransformer<f64>>, Vec<usize>)>, remainder, remainder_indices,
  n_features_in }` (`:333`). Accessors: `n_features_in()` (`:347`),
  `transformer_names()` (`:353`), `remainder_indices()` (`:362`). Its
  `Transform<Array2<f64>>` impl (`:371`) `ShapeMismatch`-checks `ncols`, pushes each
  step's `transform_pipeline` output in REGISTRATION ORDER, appends the remainder
  block LAST iff `Passthrough` and non-empty, and returns `hstack(&parts)`. It also
  impls `FittedPipelineTransformer<f64>` (`:419`).
- `make_column_transformer(transformers: Vec<(Box<dyn PipelineTransformer<f64>>,
  ColumnSelector)>, remainder) -> ColumnTransformer` (`:465`) — positional
  `transformer-{i}` naming, then `ColumnTransformer::new`.

All public types are re-exported `pub use column_transformer::{ColumnSelector,
ColumnTransformer, FittedColumnTransformer, Remainder, make_column_transformer}`
(`lib.rs:128-129`) — that boundary re-export + the `PipelineTransformer` /
`FittedPipelineTransformer` integration are the grandfathered consumers (S5 /
R-DEFER-1) pinning the SHIPPED rows. There is NO PyO3 binding (REQ-10).

**sklearn (target contract).** `ColumnTransformer(TransformerMixin,
_BaseComposition)` (`_column_transformer.py:59`) takes `transformers` (a list of
`(name, transformer, columns)`), `remainder="drop"`, `sparse_threshold=0.3`,
`n_jobs`, `transformer_weights`, `verbose`, `verbose_feature_names_out=True`,
`force_int_remainder_cols=True` (`__init__` `:290-309`). `fit_transform`
(`:929-1006`) validates columns / remainder, runs `_fit_transform_one` over the
transformer list IN ORDER, computes `sparse_output_` from density vs `sparse_threshold`
(`:991-1000`), and returns `self._hstack(list(Xs), n_samples=…)` (`:1006`).
`_validate_remainder` (`:541-550`) defines the remainder as the SORTED set difference
`set(range(n_features_in_)) - covered`, appended LAST in `_iter` (`:460-462`).
`_hstack` (`:1091-1200`) is `np.hstack` dense / `sparse.hstack(...).tocsr()` sparse.
`get_feature_names_out` (`:599`) verbose-prefixes `f"{name}__{i}"` (`:662`).
`make_column_transformer` (`:1334`) auto-names via `_name_estimators` (`:1326`,
lowercased class name, deduped).

**The gap.** ferrolearn matches sklearn on the COMPOSITION end-to-end: column routing
+ per-transformer output ORDERING (list/registration order) + remainder drop/passthrough
(ascending) + overlapping-column fan-out + the combined output VALUES (REQ-1,
oracle-verified bit-for-bit on Probe A/B/C given the parity-verified StandardScaler /
MinMaxScaler sub-transformers); `make_column_transformer` builds an equivalent CT
(REQ-2); and the dense error/parameter contracts hold (REQ-3). The remaining gaps are
input-flexibility and surface: only `Indices` selectors (REQ-4), only `Drop`/`Passthrough`
remainder — no estimator-remainder (REQ-5), dense-only — no `sparse_threshold` /
`transformer_weights` / `n_jobs` / `verbose` (REQ-6), no `get_feature_names_out` /
`verbose_feature_names_out` prefix / class-name auto-naming (REQ-7), no
`named_transformers_` / `transformers_` / `feature_names_in_` (REQ-8), `f64`-only
(REQ-9), no PyO3 (REQ-10), and the non-ferray substrate (REQ-11). This is a
**shipped-partial** unit (3 SHIPPED / 8 NOT-STARTED).

## Verification

Commands establishing the SHIPPED claims (REQ-1 routing + ordering + remainder +
overlap + combined VALUES, REQ-2 `make_column_transformer`, REQ-3 error contracts):

```bash
# Module is compiled + re-exported (the boundary consumer):
grep -n "mod column_transformer" ferrolearn-preprocess/src/lib.rs                 # lib.rs:94
grep -n "ColumnTransformer\|make_column_transformer" ferrolearn-preprocess/src/lib.rs  # lib.rs:128-129

# REQ-1/2/3 (in-module tests):
cargo test -p ferrolearn-preprocess --lib column_transformer            # 22 tests GREEN
#   REQ-1: test_basic_two_transformers_drop_remainder, test_remainder_drop,
#          test_remainder_passthrough, test_overlapping_column_selections,
#          test_passthrough_values_are_exact, test_output_shape_all_selected_drop,
#          test_output_shape_partial_passthrough, test_remainder_indices_accessor,
#          test_standard_scaler_zero_mean_in_output, test_min_max_values_in_range,
#          test_single_transformer, test_empty_transformer_list_{drop,passthrough},
#          test_all_remainder_passthrough_unchanged, test_pipeline_integration,
#          test_pipeline_transformer_interface, test_n_features_in
#   REQ-2: test_make_column_transformer_auto_names, test_make_column_transformer_single
#   REQ-3: test_invalid_column_index_out_of_range, test_shape_mismatch_on_transform,
#          test_transformer_names_explicit
cargo clippy -p ferrolearn-preprocess --all-targets -- -D warnings
cargo fmt --all --check

# REQ-1 oracle gate A — passthrough ordering + combined VALUES (3,4):
python3 -c "
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
X=np.array([[1,10,100,5],[2,20,200,6],[3,30,300,7]],float)
ct=ColumnTransformer([('std',StandardScaler(),[0,1]),('mm',MinMaxScaler(),[2])],remainder='passthrough')
print(ct.fit_transform(X).shape); print(np.round(ct.fit_transform(X),8))"
#   -> (3,4)  [[-1.22474487 -1.22474487 0. 5.] [0 0 0.5 6.] [1.22474487 1.22474487 1. 7.]]
#      (ferrolearn ColumnTransformer::new(...Passthrough) matches bit-for-bit ~1e-15)

# REQ-1 oracle gate B — drop drops the uncovered col (3,3):
python3 -c "
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
X=np.array([[1,10,100,5],[2,20,200,6],[3,30,300,7]],float)
ct=ColumnTransformer([('std',StandardScaler(),[0,1]),('mm',MinMaxScaler(),[2])],remainder='drop')
print(ct.fit_transform(X).shape)"
#   -> (3,3)   (ferrolearn Remainder::Drop matches)

# REQ-1 oracle gate C — overlapping col 0 feeds both (3,2):
python3 -c "
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
X=np.array([[1,10,100,5],[2,20,200,6],[3,30,300,7]],float)
ct=ColumnTransformer([('a',StandardScaler(),[0]),('b',MinMaxScaler(),[0])],remainder='drop')
print(ct.fit_transform(X).shape); print(np.round(ct.fit_transform(X),8))"
#   -> (3,2)  [[-1.22474487 0.][0 0.5][1.22474487 1.]]  (ferrolearn matches)

# REQ-2 oracle gate D — make_column_transformer naming (NAMING GAP -> REQ-7):
python3 -c "
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
mct=make_column_transformer((StandardScaler(),[0,1]),(MinMaxScaler(),[2]))
print([n for n,_,_ in mct.transformers])"
#   -> ['standardscaler','minmaxscaler']   (ferrolearn names ['transformer-0','transformer-1'])
```

The in-module `#[test]`s pin REQ-1 (routing + ordering + drop/passthrough + overlap +
the StandardScaler/MinMaxScaler-composed VALUES), REQ-2 (`make_column_transformer`
build + positional naming), and REQ-3 (out-of-range index + transform ncols
mismatch), all 22 GREEN; the Probe A/B/C oracle gates pin the combined output shapes +
VALUES bit-for-bit. No green command establishes REQ-4 (non-index selectors), REQ-5
(estimator-remainder), REQ-6 (`sparse_threshold` / `transformer_weights` / `n_jobs`),
REQ-7 (`get_feature_names_out` / class-name naming), REQ-8 (`named_transformers_` /
`transformers_`), REQ-9 (generic `F`), REQ-10 (PyO3), or REQ-11 (ferray).

## Blockers

REQ-1 (column routing + output ordering + drop/passthrough + overlapping fan-out +
combined VALUES), REQ-2 (`make_column_transformer` builds the CT), and REQ-3 (dense
error / parameter contracts) are SHIPPED — the module is compiled (`lib.rs:94`) and
re-exported (`lib.rs:128-129`, the grandfathered boundary consumer) plus
pipeline-integrated, the combined output matches the live oracle bit-for-bit on the
passthrough / drop / overlap fixtures (given the independently parity-verified
StandardScaler / MinMaxScaler sub-transformers), and the 22 in-module tests are green.

The remaining REQs (REQ-4..11) are NOT-STARTED. Each should be filed as a `-l blocker`
issue against tracking issue #1434 (placeholder `#1435` … `#1442` until filed):

- #1435 — REQ-4: only `ColumnSelector::Indices(Vec<usize>)`
  (`column_transformer.rs:53-61`); no str/slice/bool-mask/callable selection and no
  `make_column_selector` (`sklearn/compose/_column_transformer.py:520`, `:1468`).
- #1436 — REQ-5: `enum Remainder` is only `Drop`/`Passthrough`
  (`column_transformer.rs:97-104`); no estimator-remainder and no step-level
  `'drop'`/`'passthrough'` (`_column_transformer.py:294`, `:550`).
- #1437 — REQ-6: dense `hstack` only (`column_transformer.rs:133`); no
  `sparse_threshold` / sparse output (`_column_transformer.py:991-1000`, `:1105-1120`),
  no `transformer_weights`, no `n_jobs`, no `verbose`.
- #1438 — REQ-7: no `get_feature_names_out` / `verbose_feature_names_out` prefix
  (`_column_transformer.py:599`, `:662`) and positional `transformer-N` naming vs
  sklearn lowercased-class-name (`_name_estimators` `:1326`).
- #1439 — REQ-8: no `named_transformers_` / `transformers_` (incl. remainder entry) /
  `feature_names_in_` (`_column_transformer.py:574-582`, `:694-726`); only
  `n_features_in()` / `transformer_names()` / `remainder_indices()`.
- #1440 — REQ-9: `f64`-hard-wired throughout (`column_transformer.rs:236`, `:371`,
  `:113`, `:133`); not generic over `F: Float`.
- #1441 — REQ-10: no PyO3 `ColumnTransformer` / `make_column_transformer` binding in
  `ferrolearn-python`.
- #1442 — REQ-11: composes on `ndarray::Array2<f64>` / `num_traits`, not ferray
  (R-SUBSTRATE-1/2).
