# ParameterGrid (param_grid! / ParamValue / ParamSet)

<!--
tier: 3-component
status: draft
baseline-commit: 03154d8d1affa2d64c05fc6fad9c2f4d89bafde1
upstream-paths:
  - sklearn/model_selection/_search.py   # class ParameterGrid (:63-:213)
-->

## Summary

`ferrolearn-model-sel/src/param_grid.rs` mirrors the *enumeration* side of
scikit-learn's `ParameterGrid` (`sklearn/model_selection/_search.py:63`): the
exhaustive Cartesian product of a dict of `param -> list-of-values`. ferrolearn
expresses it as the `param_grid!` macro (builds a `Vec<ParamSet>`), the typed
`ParamValue` enum (the Python "arbitrary object" value, narrowed to a Rust enum),
and `ParamSet = HashMap<String, ParamValue>` (one combination). The macro is the
real grid producer consumed by `grid_search.rs` (`GridSearchCV`) and
`random_search.rs` (`RandomizedSearchCV`).

ferrolearn ships the **single-dict Cartesian-product CONTENTS** end-to-end. Two
behaviors diverge from sklearn and are NOT-STARTED: (1) the enumeration **ORDER**
— sklearn sorts the keys before `itertools.product` (`_search.py:157`) so the
sequence is sorted-key order; the macro products in *written* order, so when keys
are written out of sorted order the `Vec<ParamSet>` sequence differs from
`list(ParameterGrid(...))`, which matters because `GridSearchCV` breaks
equal-CV-score ties by position; and (2) the **list-of-dicts** grid
(`_search.py:114-117`). The `ParameterGrid` *class surface* (`__len__`,
O(1) `__getitem__` without materializing) is replaced by the native `Vec`
`.len()`/`[i]` (R-DEV-7). One edge case (empty value-list) diverges. R-SUBSTRATE
is **N/A** here: the module touches no arrays (strings and scalars only), so it
has no `ndarray`/`ferray` dependency to migrate.

## Upstream reference (scikit-learn 1.5.2, commit 156ef14)

- `sklearn/model_selection/_search.py:63` — `class ParameterGrid`. Docstring
  (`:68`): "The order of the generated parameter combinations is deterministic."
- `:107-144` — `__init__(self, param_grid)`: a `Mapping` is wrapped in a singleton
  list (`:114-117` `param_grid = [param_grid]`) — so a dict OR a list-of-dicts is
  accepted; each sub-grid is validated to be a dict of sequences (`:120-142`);
  an empty value-list raises `ValueError` (`:138-142`
  `"... need to be a non-empty sequence"`).
- `:146-164` — `__iter__`: `for p in self.param_grid:` then
  `items = sorted(p.items())` (`:157`) — **keys are sorted before product**;
  empty dict yields one empty `{}` (`:158-159`); otherwise
  `for v in product(*values): yield dict(zip(keys, v))` (`:162-164`). With
  `itertools.product`, the FIRST (sorted) key varies SLOWEST and the LAST varies
  fastest.
- `:166-172` — `__len__`: `sum(prod(len(v) for v in p.values()) ... else 1 ...)` —
  product of axis lengths per sub-grid, `1` for an empty dict, summed across
  sub-grids. O(1), no materialization.
- `:174-213` — `__getitem__(self, ind)`: O(1) mixed-radix decode of the `ind`-th
  combination using `divmod` over the sorted (reversed) axis sizes — no
  materialization. `IndexError` when out of range (`:213`).

## Requirements

- REQ-1: Single-dict Cartesian-product CONTENTS. `param_grid!` builds, for axes
  `name => [v...]`, every combination as a `ParamSet`; the SET of produced
  combinations equals `set(list(ParameterGrid({...})))`. Mirrors
  `product(*values)` (`sklearn/model_selection/_search.py:162`). DETERMINISTIC /
  oracle-pinnable (compare as a multiset of key->value maps).
- REQ-2: Enumeration ORDER parity. sklearn sorts keys before product
  (`_search.py:157`), so `list(ParameterGrid(...))` is in sorted-key order (first
  sorted key slowest). The macro products in WRITTEN order. When keys are written
  non-sorted the i-th combination differs. This matters: `GridSearchCV::fit`
  iterates the `Vec` in order and ties are broken by position. DETERMINISTIC /
  oracle-pinnable.
- REQ-3: List-of-dicts grid. sklearn `ParameterGrid` accepts a sequence of dicts
  (union of sub-grids, `_search.py:114-117`, `:155`). The macro is single-dict
  only. NOT-STARTED.
- REQ-4: `ParameterGrid` class surface (`__len__` / `__getitem__`). sklearn
  supports `len(grid)` and O(1) `grid[i]` without materializing
  (`_search.py:166-213`). ferrolearn materializes the full `Vec<ParamSet>` and
  uses native `.len()` / `vec[i]` — the Rust analog (R-DEV-7).
- REQ-5: Edge cases (empty dict; empty value-list). sklearn: empty dict `{}`
  yields ONE empty combo (`:158-159`, `__len__` = 1); an empty value-list raises
  `ValueError` (`:138-142`). DETERMINISTIC / oracle-pinnable.
- REQ-6: `ParamValue` type coverage. sklearn values are arbitrary Python objects;
  ferrolearn narrows to a typed enum (`Float`/`Int`/`Bool`/`String`) with `From`
  conversions for `f64`/`f32`/`i64`/`i32`/`usize`/`bool`/`String`/`&str`
  (R-DEV-7). Note the coverage and the gap.
- REQ-7: Non-test production consumer.
- REQ-8: R-SUBSTRATE. Array/numpy substrate migration to `ferray`.

## Acceptance criteria

- AC-1 (REQ-1): `param_grid!{"a"=>[1_i64,2],"b"=>[true,false]}` produces the same
  4-combination multiset as the live oracle
  `list(ParameterGrid({'a':[1,2],'b':[True,False]}))`
  (`{a:1,b:T},{a:1,b:F},{a:2,b:T},{a:2,b:F}`). DETERMINISTIC / oracle-pinnable.
- AC-2 (REQ-2): with keys written non-sorted —
  `param_grid!{"b"=>[1_i64,2],"a"=>[3_i64,4]}` — the SEQUENCE is
  `[{b:1,a:3},{b:1,a:4},{b:2,a:3},{b:2,a:4}]` (b slowest), whereas the live oracle
  `list(ParameterGrid({'b':[1,2],'a':[3,4]}))` is
  `[{a:3,b:1},{a:3,b:2},{a:4,b:1},{a:4,b:2}]` (a slowest); the i-th entries differ.
  DETERMINISTIC / oracle-pinnable.
- AC-3 (REQ-3): a list-of-dicts grid
  (`[{'kernel':['linear']},{'kernel':['rbf'],'gamma':[1,10]}]` → 3 combos via the
  live oracle) has no `param_grid!` form.
- AC-4 (REQ-4): `param_grid!{...}.len()` equals `len(ParameterGrid({...}))` and
  `grid[i]` indexes a materialized combination; sklearn's no-materialization
  `__getitem__` contract is not separately ported.
- AC-5 (REQ-5): `param_grid!{}` → exactly one empty `ParamSet`
  (matching live-oracle `[{}]`, `len`=1); `param_grid!{"x"=>[]}` → ZERO combos and
  NO error, whereas the live oracle raises `ValueError`.
- AC-6 (REQ-6): `ParamValue::from` for each of the eight source types yields the
  expected variant; non-listed types (e.g. arbitrary structs) have no conversion.
- AC-7 (REQ-7): `param_grid!`/`ParamSet`/`ParamValue` are consumed from non-test
  production code.
- AC-8 (REQ-8): N/A — no array substrate in this module.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (single-dict Cartesian-product contents) | SHIPPED | The `param_grid!` macro in `param_grid.rs` seeds `result = vec![ParamSet::new()]` and, for each `(name, values)` axis, replaces `result` with the product `for existing in &result { for val in &values { entry.insert(name, val) } }`. This is exactly `itertools.product(*values)` over the axis value-lists (`sklearn/model_selection/_search.py:162`), so the SET of combinations equals the oracle's. DETERMINISTIC / oracle-pinnable: `param_grid!{"a"=>[1_i64,2],"b"=>[true,false]}` (4 combos) equals the multiset of `list(ParameterGrid({'a':[1,2],'b':[True,False]}))` (Verification). Tests: `test_param_grid_two_axes_cartesian`, `test_param_grid_three_axes`, `test_param_grid_all_keys_present`, `test_param_grid_values_correct` assert the count and per-key presence. CAVEAT (R-CHAR-3): existing tests assert `.len()` and `contains_key`, not multiset equality against the live oracle — the critic should pin an oracle-grounded multiset-equality `#[test]`. Non-test consumer: REQ-7. |
| REQ-2 (enumeration ORDER parity) | NOT-STARTED | open prereq blocker (tracking #1697; critic to file specific). The macro builds `axes` in the order the `$key => [...]` arms are WRITTEN — `let axes: Vec<(String, Vec<ParamValue>)> = vec![ ($key.to_string(), ...) , ... ]` in `param_grid.rs` — and never sorts by key. sklearn sorts: `items = sorted(p.items())` BEFORE `product` (`sklearn/model_selection/_search.py:157`), so the LAST sorted key varies fastest. When keys are written non-sorted the SEQUENCE diverges (per-combo key order is irrelevant — `ParamSet` is a `HashMap` — only the Vec sequence). DETERMINISTIC / oracle-pinnable: `param_grid!{"b"=>[1_i64,2],"a"=>[3_i64,4]}` → sequence `[{b:1,a:3},{b:1,a:4},{b:2,a:3},{b:2,a:4}]`; live oracle `list(ParameterGrid({'b':[1,2],'a':[3,4]}))` → `[{a:3,b:1},{a:3,b:2},{a:4,b:1},{a:4,b:2}]` (Verification) — i-th entries differ. This MATTERS: `GridSearchCV::fit` in `grid_search.rs` iterates the `Vec` in order (`for params in &self.param_grid`) and `CvResults::best_index` (`grid_search.rs`) breaks equal-mean-score ties via `Iterator::max_by` (returns the LAST max), so a different Vec order yields a different best-param on a CV-score tie. (sklearn `GridSearchCV` selects `np.argmax` = the FIRST max in sorted-key order — a separate tie-break divergence the order fix is a prerequisite for.) Gap: the macro must sort `axes` by key. |
| REQ-3 (list-of-dicts grid) | NOT-STARTED | open prereq blocker (tracking #1697). `param_grid!` accepts a single `$( $key => [...] ),*` dict; there is no list-of-dicts arm. sklearn wraps a dict in a singleton list and unions sub-grids (`sklearn/model_selection/_search.py:114-117`, `:155`); live oracle `list(ParameterGrid([{'kernel':['linear']},{'kernel':['rbf'],'gamma':[1,10]}]))` → 3 combos with heterogeneous keys (Verification). Absent end-to-end. |
| REQ-4 (len / getitem class surface) | SHIPPED | `param_grid!` returns a fully-materialized `Vec<ParamSet>`; `len(grid)` and `grid[i]` are `Vec::len`/`Index` and equal the oracle's `__len__`/materialized `__getitem__` values. R-DEV-7: the Rust analog of `ParameterGrid.__len__`/`__getitem__` (`sklearn/model_selection/_search.py:166-213`) is the native `Vec` API — the observable values match (`param_grid!{"a"=>[1_i64,2],"b"=>[true,false]}.len() == len(ParameterGrid({...})) == 4`, Verification). DIVERGENCE (honest underclaim): sklearn's `__getitem__` is O(1) WITHOUT materializing the grid (mixed-radix `divmod` decode, `:198-211`); ferrolearn materializes the whole `Vec` up front, so the memory-efficiency contract `ParameterSampler` relies on is NOT ported — only the indexed-value equality is. Test: `test_param_grid_single_axis`/`_two_axes_cartesian` assert `.len()`. SHIPPED on observable `len`/indexed value; the lazy-indexing memory contract is N/A under the eager-`Vec` model. Non-test consumer: REQ-7. |
| REQ-5 (edge cases: empty dict / empty value-list) | NOT-STARTED | open prereq blocker (tracking #1697). EMPTY DICT matches: `param_grid!{}` leaves `result = vec![ParamSet::new()]` (the per-axis loop never runs) → exactly one empty `ParamSet`, mirroring sklearn's `if not items: yield {}` (`sklearn/model_selection/_search.py:158-159`; live oracle `list(ParameterGrid({}))==[{}]`, `len==1`). EMPTY VALUE-LIST DIVERGES: `param_grid!{"x"=>[]}` produces an axis with an empty `values` vec, so the product collapses `result` to ZERO combos and returns `vec![]` with NO error; sklearn raises `ValueError("... need to be a non-empty sequence")` (`:138-142`; live oracle confirms `ValueError`, Verification). NOT-STARTED on the strict-empty-list rejection — the macro cannot raise (it has no `Result`), so it silently yields an empty grid instead of erroring. (The empty-dict half is correct; the REQ is binary and the empty-value-list half is absent — the critic may split it.) |
| REQ-6 (ParamValue type coverage) | SHIPPED | `pub enum ParamValue { Float(f64), Int(i64), Bool(bool), String(String) }` in `param_grid.rs` with `From` impls for `f64`, `f32`(→`Float`), `i64`, `i32`(→`Int`), `usize`(→`Int`), `bool`, `String`, `&str`(→`String`), plus a `Display` impl. The macro's `@into` arm calls `ParamValue::from($val)`, so any of the eight types is a valid grid value. R-DEV-7: a typed enum replaces Python's arbitrary-object value; the four scalar kinds (float/int/bool/string) cover sklearn's common hyperparameter value space. Tests: `test_param_value_from_conversions`, `test_param_value_display`. CAVEAT (honest underclaim): this is NARROWER than sklearn — a hyperparameter whose value is itself an estimator/callable/tuple/`None` (e.g. `kernel=[chi2_kernel]`, `class_weight=[None,'balanced']` mixing `None` and `str`) has no `ParamValue` variant; `None`/`Option` is unrepresentable. SHIPPED on scalar coverage; the object-valued-param gap is a known narrowing. Non-test consumer: REQ-7. |
| REQ-7 (consumer) | SHIPPED | Real non-test production consumers: `grid_search.rs` stores `param_grid: Vec<ParamSet>` and iterates it in `GridSearchCV::fit` (`for params in &self.param_grid { (self.pipeline_factory)(params) ... }`), with `pipeline_factory: Box<dyn Fn(&ParamSet) -> Pipeline>` and `best_params(&self) -> Option<&ParamSet>`; `random_search.rs` builds `ParamSet` in `RandomizedSearchCV::sample_params(&self, rng) -> ParamSet` and reads `ParamValue` via `p.get(...)`. Crate re-export: `lib.rs` `pub use param_grid::{ParamSet, ParamValue};` and the `param_grid!` macro is `#[macro_export]`. R-DEFER-1 / S5: `param_grid!`/`ParamSet`/`ParamValue` are grandfathered existing pub API and have genuine in-crate production consumers (not test-only). CAVEAT (honest underclaim): there is no `ferrolearn-python` binding exposing `ParameterGrid` yet — the consumer surface is in-crate Rust only. |
| REQ-8 (ferray substrate) | SHIPPED | N/A — R-SUBSTRATE does not apply. `param_grid.rs` imports only `std::collections::HashMap` and `std::fmt`; it manipulates strings and scalar `ParamValue`s and touches NO array type (`grep` for `ndarray`/`ferray`/`sprs`/`statrs`/`rand` in `param_grid.rs` is empty, Verification). There is no numpy-layer dependency to migrate, so the unit is substrate-clean by construction (R-SUBSTRATE vacuously satisfied). |

## Architecture

ferrolearn maps sklearn's one `ParameterGrid` object onto three pieces. The
*value* — a Python "arbitrary object" — becomes the typed `pub enum ParamValue`
(`Float(f64)`/`Int(i64)`/`Bool(bool)`/`String(String)`) with `From` conversions
and `Display`. A single *combination* — sklearn's `dict(zip(keys, v))`
(`sklearn/model_selection/_search.py:163`) — becomes
`pub type ParamSet = HashMap<String, ParamValue>`; because it is a `HashMap`, the
per-combination KEY ORDER is irrelevant (only membership/values matter), which is
why the ORDER divergence (REQ-2) is purely about the SEQUENCE of `ParamSet`s, not
about within-`ParamSet` ordering. The *grid producer* — sklearn's
`__init__`+`__iter__` — becomes the `#[macro_export] macro_rules! param_grid!`,
which materializes the full Cartesian product as a `Vec<ParamSet>` eagerly (the
Rust analog of `ParameterGrid` + `list(...)`; R-DEV-7).

The product is built iteratively (`param_grid.rs`): `result` starts as
`vec![ParamSet::new()]`; for each axis `(name, values)` in WRITTEN order, every
existing partial `ParamSet` is cloned once per value and extended with
`name -> val`. This is the imperative equivalent of `itertools.product`
(`:162`) and yields the same CONTENTS (REQ-1). The two structural divergences
both live in this macro: (a) it iterates `axes` in source-written order, never
`sorted` like `:157` — REQ-2; and (b) it accepts only one dict, with no
list-of-dicts arm — REQ-3. The macro returns a plain `Vec`, so `len`/`[i]` are
native (REQ-4) and there is no `Result` channel to raise sklearn's empty-value
`ValueError` (REQ-5).

Consumers: `GridSearchCV` (`grid_search.rs`) takes a `param_grid: Vec<ParamSet>`,
iterates it in declaration order in `fit`, scores each via `cross_val_score`, and
selects `best_index` with `Iterator::max_by` (LAST max on ties — order-sensitive,
hence REQ-2's relevance). `RandomizedSearchCV` (`random_search.rs`) constructs
`ParamSet`s by sampling and reads them back through `ParamValue` pattern matches.

What is structurally absent vs sklearn: sorted-key enumeration order (REQ-2,
`:157`), list-of-dicts grids (REQ-3, `:114-117`), the lazy no-materialization
`__getitem__` memory contract (REQ-4 caveat, `:174-213`), strict empty-value-list
rejection (REQ-5, `:138-142`), and object/`None`-valued hyperparameters (REQ-6
caveat). R-SUBSTRATE is N/A — the module has no array layer (REQ-8).

## Verification

Commands establishing the SHIPPED claims (baseline
`03154d8d1affa2d64c05fc6fad9c2f4d89bafde1`):

- `cargo test -p ferrolearn-model-sel --lib param_grid` → the 7 `param_grid`/
  `param_value` tests pass (REQ-1 contents/count; REQ-4 `.len()`; REQ-6 `From`/
  `Display`).
- REQ-1 oracle (single-dict contents, live oracle):
  ```
  python3 -c "from sklearn.model_selection import ParameterGrid; \
  print(list(ParameterGrid({'a':[1,2],'b':[True,False]})))"
  # -> [{'a':1,'b':True},{'a':1,'b':False},{'a':2,'b':True},{'a':2,'b':False}]
  ```
  The macro's 4-combination multiset matches (pin an oracle-grounded
  multiset-equality `#[test]`, R-CHAR-3).
- REQ-2 oracle (sorted-key ORDER divergence, live oracle):
  ```
  python3 -c "from sklearn.model_selection import ParameterGrid; \
  print(list(ParameterGrid({'b':[1,2],'a':[3,4]})))"
  # -> [{'a':3,'b':1},{'a':3,'b':2},{'a':4,'b':1},{'a':4,'b':2}]  (a slowest)
  ```
  `param_grid!{"b"=>[1_i64,2],"a"=>[3_i64,4]}` →
  `[{b:1,a:3},{b:1,a:4},{b:2,a:3},{b:2,a:4}]` (b slowest); the i-th entries
  differ → REQ-2 NOT-STARTED (macro must `sort` axes by key).
- REQ-3 oracle (list-of-dicts, live oracle):
  ```
  python3 -c "from sklearn.model_selection import ParameterGrid; \
  print(list(ParameterGrid([{'kernel':['linear']},{'kernel':['rbf'],'gamma':[1,10]}])))"
  # -> [{'kernel':'linear'},{'gamma':1,'kernel':'rbf'},{'gamma':10,'kernel':'rbf'}]
  ```
  No `param_grid!` form — REQ-3 NOT-STARTED.
- REQ-4 oracle (len parity, live oracle):
  `len(ParameterGrid({'a':[1,2],'b':[True,False]}))` → `4`; macro `.len()` → `4`.
  sklearn's lazy `__getitem__` (`:198-211`) is not separately ported.
- REQ-5 oracle (edge cases, live oracle):
  ```
  python3 -c "from sklearn.model_selection import ParameterGrid; \
  print(list(ParameterGrid({})), len(ParameterGrid({}))); \
  ParameterGrid({'x':[]})"
  # -> [{}] 1   then ValueError: Parameter grid for parameter 'x' need to be a non-empty sequence
  ```
  `param_grid!{}` → one empty `ParamSet` (matches `[{}]`/len 1); `param_grid!{"x"=>[]}`
  → ZERO combos with NO error (sklearn raises `ValueError`) — REQ-5 NOT-STARTED on
  the empty-value-list rejection.
- REQ-6 (`ParamValue` coverage): `test_param_value_from_conversions` /
  `test_param_value_display` exercise the eight `From` impls and `Display`.
- REQ-7 (consumer): `grep -n "ParamSet\|param_grid\|ParamValue" grid_search.rs
  random_search.rs lib.rs` (filtered for non-test lines) shows
  `GridSearchCV { param_grid: Vec<ParamSet> }` iterated in `fit`,
  `RandomizedSearchCV::sample_params -> ParamSet`, and the `lib.rs` re-export.
- REQ-8 (substrate): `grep -E "ndarray|ferray|sprs|statrs|rand" param_grid.rs`
  is empty — no array layer, R-SUBSTRATE N/A.

SHIPPED: REQ-1 (single-dict Cartesian-product contents), REQ-4 (native `Vec`
`len`/indexed-value parity; lazy-`__getitem__` memory contract N/A),
REQ-6 (`ParamValue` scalar coverage; object/`None`-valued params a known
narrowing), REQ-7 (in-crate `GridSearchCV`/`RandomizedSearchCV` consumers; no
Python binding yet), REQ-8 (R-SUBSTRATE N/A — no array layer). NOT-STARTED
(tracking #1697; the critic files per-REQ blockers): REQ-2 (sorted-key
enumeration order — headline; the macro must `sort` axes by key, and a
GridSearchCV CV-tie depends on it), REQ-3 (list-of-dicts grid),
REQ-5 (empty-value-list `ValueError` rejection; the empty-dict half is correct).
Per R-DEFER-2 the table is binary SHIPPED/NOT-STARTED.
