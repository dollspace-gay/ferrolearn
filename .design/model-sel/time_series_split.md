# TimeSeriesSplit

<!--
tier: 3-component
status: draft
baseline-commit: 3394bc3a4d76554a140c8040cf59c8bac7b0e7eb
upstream-paths:
  - sklearn/model_selection/_split.py   # class TimeSeriesSplit (:1062), _split (:1219-1271)
-->

## Summary

`ferrolearn-model-sel/src/time_series_split.rs` mirrors scikit-learn's
`TimeSeriesSplit` (`sklearn/model_selection/_split.py:1062`) — the deterministic
time-series cross-validation splitter in which each test window follows its
training window chronologically (no shuffling, no look-ahead). This is a
DETERMINISTIC index-arithmetic splitter: there is no RNG carve-out. Every fold's
`(train_indices, test_indices)` is a pure function of
`(n_samples, n_splits, test_size, max_train_size, gap)`, so every REQ below is
oracle-pinnable against `list(TimeSeriesSplit(...).split(np.arange(n)))`.

ferrolearn exposes a builder: `TimeSeriesSplit::new(n_splits)` plus
`.max_train_size(Option)`, `.test_size(Option)`, `.gap(usize)`, and a
`.split(n_samples) -> FerroResult<FoldSplits>` returning a `Vec<(Vec<usize>,
Vec<usize>)>`. An exhaustive sweep of 5600 parameter combinations (`n` 2..16,
`n_splits` 2..6, `test_size ∈ {None,1,2,3,4}`, `gap ∈ {0,1,2,3}`,
`max_train_size ∈ {None,1,2,3,4}`) against the live sklearn 1.5.2 oracle found
**zero** fold-content divergences AND **zero** error-status divergences — the
ferrolearn index arithmetic reproduces sklearn's `test_starts` algorithm and the
sklearn `ValueError` conditions exactly (see Verification). The only API-surface
difference is the constructor shape (sklearn `n_splits` defaults to 5; ferrolearn
`new(n_splits)` requires it explicitly).

## Upstream reference (scikit-learn 1.5.2, commit 156ef14)

### Constructor / defaults
- `sklearn/model_selection/_split.py:1183` — `def __init__(self, n_splits=5, *,
  max_train_size=None, test_size=None, gap=0)`. `n_splits` defaults to 5 (keyword
  default); `max_train_size`/`test_size` default `None`; `gap` defaults `0`. The
  three optional params are keyword-only (`*`).
- `:1184` — `super().__init__(n_splits, shuffle=False, random_state=None)`. The
  `_BaseKFold`/`KFold` base enforces `n_splits >= 2` via the
  `Interval(Integral, 2, None, closed="left")` constraint; live oracle:
  `TimeSeriesSplit(n_splits=1)` raises `ValueError` ("k-fold cross-validation
  requires at least one train/test split by setting n_splits=2 or more").

### The split algorithm (`_split`)
- `:1239` — `n_folds = n_splits + 1`.
- `:1241-1243` — `test_size = self.test_size if self.test_size is not None else
  n_samples // n_folds` — the default test window is `n_samples // (n_splits+1)`.
- `:1246-1250` — `if n_folds > n_samples: raise ValueError("Cannot have number of
  folds={n_folds} greater than the number of samples={n_samples}.")`.
- `:1251-1255` — `if n_samples - gap - (test_size * n_splits) <= 0: raise
  ValueError("Too many splits={n_splits} for number of samples={n_samples} with
  test_size={test_size} and gap={gap}.")`.
- `:1257-1258` — `indices = np.arange(n_samples)`;
  `test_starts = range(n_samples - n_splits * test_size, n_samples, test_size)`.
- `:1260-1271` — per `test_start`: `train_end = test_start - gap`; then
  `if self.max_train_size and self.max_train_size < train_end: yield
  (indices[train_end - self.max_train_size : train_end], indices[test_start :
  test_start + test_size])` else `yield (indices[:train_end], indices[test_start :
  test_start + test_size])`.

Note `:1262`: `max_train_size` is applied ONLY when `max_train_size < train_end`;
otherwise the full `indices[:train_end]` prefix is used.

## Requirements

- REQ-1: Default split-index parity (DETERMINISTIC). With default
  `test_size=None` (→ `n_samples // (n_splits+1)`), `max_train_size=None`,
  `gap=0`, ferrolearn's per-fold `(train, test)` index lists equal sklearn's
  exactly. Mirrors `:1241-1243`, `:1257-1271`. Oracle-pinnable
  (`TimeSeriesSplit(n_splits=3).split(np.arange(12))`).
- REQ-2: Explicit `test_size` override (DETERMINISTIC). Each test fold contains
  exactly `test_size` samples; train grows as `indices[:train_end]`. Mirrors
  `:1241-1242` (the `self.test_size if not None` branch). Oracle-pinnable
  (`n_splits=3, test_size=2, n=12`).
- REQ-3: `gap` (DETERMINISTIC). `gap>0` removes the `gap` samples immediately
  before each test fold from train: `train = indices[:test_start - gap]`. Mirrors
  `:1261`. Oracle-pinnable (`n_splits=3, gap=2, n=12`).
- REQ-4: `max_train_size` (DETERMINISTIC). `max_train_size=m` caps train to the
  most-recent `m` samples before the gap, BUT only when `m < train_end`; else the
  full prefix is kept. Mirrors `:1262-1271`. Oracle-pinnable (`n_splits=3,
  max_train_size=4, n=12`).
- REQ-5: Error semantics — too-many-splits / too-large `test_size`+`gap`. sklearn
  raises `ValueError` when `n_folds > n_samples` (`:1246`) OR `n_samples - gap -
  test_size*n_splits <= 0` (`:1251`). ferrolearn must return an `Err` on the SAME
  input set. Oracle-pinnable.
- REQ-6: `n_splits` default + `n_splits >= 2` validation. sklearn defaults
  `n_splits=5` (`:1183`) and rejects `n_splits < 2` (`_BaseKFold` constraint).
  ferrolearn `new(n_splits)` REQUIRES `n_splits` (no default) and rejects
  `n_splits < 2`. API-shape note + reject-behavior parity.
- REQ-7: R-SUBSTRATE. The owned computation is pure `usize`/`Vec<usize>` index
  math — there is no numpy-array computation to migrate (sklearn itself uses only
  `np.arange` + slicing for index bookkeeping). No `ndarray`/`faer`/`sprs`/`rand`
  in production code.
- REQ-8: Non-test production consumer.

## Acceptance criteria

- AC-1 (REQ-1): `TimeSeriesSplit::new(3).split(12)` yields fold 0
  `(train=[0,1,2], test=[3,4,5])`, fold 1 `([0..6], [6,7,8])`, fold 2 `([0..9],
  [9,10,11])` — equal to the live oracle (Verification). DETERMINISTIC.
- AC-2 (REQ-2): `TimeSeriesSplit::new(3).test_size(Some(2)).split(12)` yields test
  folds `[6,7]`, `[8,9]`, `[10,11]` (each length 2) with train `[0..6]`, `[0..8]`,
  `[0..10]` — equal to the oracle. DETERMINISTIC.
- AC-3 (REQ-3): `TimeSeriesSplit::new(3).gap(2).split(12)` yields fold 0
  `(train=[0], test=[3,4,5])`, fold 1 `([0,1,2,3], [6,7,8])`, fold 2 `([0..7],
  [9,10,11])` — equal to the oracle. DETERMINISTIC.
- AC-4 (REQ-4): `TimeSeriesSplit::new(3).max_train_size(Some(4)).split(12)` yields
  fold 0 `(train=[0,1,2], test=[3,4,5])` (cap NOT applied: `4 >= train_end=3`),
  fold 1 `([2,3,4,5], [6,7,8])`, fold 2 `([5,6,7,8], [9,10,11])` — equal to the
  oracle, including the fold-0 no-cap case. DETERMINISTIC.
- AC-5 (REQ-5): `TimeSeriesSplit::new(10).split(5)` errors (sklearn:
  `n_folds=11 > 5`); `TimeSeriesSplit::new(3).test_size(Some(4)).split(12)` errors
  (sklearn: `12 - 0 - 12 <= 0`); `TimeSeriesSplit::new(2).test_size(Some(2)).
  gap(100).split(10)` errors (sklearn: `10 - 100 - 4 <= 0`). The full 5600-case
  sweep confirms ferrolearn's error-status matches sklearn on every input
  (Verification). DETERMINISTIC.
- AC-6 (REQ-6): `TimeSeriesSplit::new(1).split(20)` errors (both reject
  `n_splits < 2`). API-shape: ferrolearn has no `n_splits` default — `new` takes
  it positionally; sklearn's default `5` is the constructor literal `:1183`.
- AC-7 (REQ-7): no `ndarray`/`faer`/`sprs`/`rand` import in production code of
  `time_series_split.rs` (only `#[cfg(test)]` uses `ndarray`).
- AC-8 (REQ-8): `TimeSeriesSplit` is reachable from non-test production code as a
  `&dyn CrossValidator` argument to `cross_val_score`/`cross_validate`/
  `cross_val_predict`/`learning_curve`/`validation_curve`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (default split-index parity) | SHIPPED | impl `fn split_impl in time_series_split.rs` computes `test_sz = n_samples / (self.n_splits + 1)` (default branch) then per fold `i`: `windows_after = self.n_splits - 1 - i`, `test_end = n_samples - windows_after*test_sz`, `test_start = test_end - test_sz`, `train = (0..test_start).collect()` (gap=0, no cap). This is an algebraic re-indexing of sklearn's `test_starts = range(n_samples - n_splits*test_size, n_samples, test_size)` and `indices[:train_end]` (`sklearn/model_selection/_split.py:1257-1269`): ferrolearn's `test_start` for fold `i` = `n_samples - (n_splits-i)*test_sz` = the `i`-th element of sklearn's `test_starts`. Live oracle `TimeSeriesSplit(n_splits=3).split(np.arange(12))` → fold0 `([0,1,2],[3,4,5])`, fold1 `([0..6],[6,7,8])`, fold2 `([0..9],[9,10,11])`; the 5600-case sweep finds 0 fold-content divergences (Verification). Tests: `test_tss_default_test_size_fills_last_window`, `test_tss_test_indices_after_train_indices`, `test_tss_training_grows_each_split`, `test_tss_n_splits_2_minimal` (asserts exact lists `train0=[0,1]`, `test0=[2,3]`, etc.). Non-test consumer: REQ-8. |
| REQ-2 (explicit test_size override) | SHIPPED | impl `fn split_impl in time_series_split.rs` takes the `Some(ts)` branch `let test_sz = ts` (rejecting `ts == 0`), then `test_end`/`test_start` use that `test_sz` so every test fold has exactly `test_sz` indices and `train = (0..test_start)`. Mirrors sklearn's `test_size = self.test_size if self.test_size is not None else ...` (`sklearn/model_selection/_split.py:1241-1242`). Live oracle `TimeSeriesSplit(n_splits=3, test_size=2).split(np.arange(12))` → test folds `[6,7]`/`[8,9]`/`[10,11]`, train `[0..6]`/`[0..8]`/`[0..10]`; matched in the 0-divergence sweep (Verification). Tests: `test_tss_fixed_test_size` (every test fold length == 3), `test_tss_fixed_test_size_non_overlapping_tests`. Non-test consumer: REQ-8. |
| REQ-3 (gap) | SHIPPED | impl `fn split_impl in time_series_split.rs` computes `let train_end = test_start.checked_sub(self.gap)?` so the `gap` samples immediately before each test window are excluded from train (`train = (train_start..train_end)`). Mirrors sklearn `train_end = test_start - gap` (`sklearn/model_selection/_split.py:1261`). Live oracle `TimeSeriesSplit(n_splits=3, gap=2).split(np.arange(12))` → fold0 `([0],[3,4,5])`, fold1 `([0,1,2,3],[6,7,8])`, fold2 `([0..7],[9,10,11])`; matched in the sweep (Verification). Tests: `test_tss_gap_separates_train_and_test` (asserts exactly `gap` indices between max-train and min-test). Non-test consumer: REQ-8. |
| REQ-4 (max_train_size) | SHIPPED | impl `fn split_impl in time_series_split.rs` sets `let train_start = match self.max_train_size { Some(mts) => train_end.saturating_sub(mts), None => 0 }`, so `train = (train_end - mts .. train_end)` capped to the most-recent `mts` samples — and `saturating_sub` reproduces sklearn's conditional `if max_train_size < train_end` (`sklearn/model_selection/_split.py:1262`): when `mts >= train_end`, `saturating_sub` clamps `train_start` to 0, exactly sklearn's else-branch `indices[:train_end]`. Live oracle `TimeSeriesSplit(n_splits=3, max_train_size=4).split(np.arange(12))` → fold0 `([0,1,2],[3,4,5])` (no cap: `4 >= 3`), fold1 `([2,3,4,5],[6,7,8])`, fold2 `([5,6,7,8],[9,10,11])`; matched in the sweep, including the fold-0 no-cap edge (Verification). Tests: `test_tss_max_train_size_limits_training`, `test_tss_max_train_size_uses_most_recent` (train length == 3, last train idx + 1 == min test idx). Non-test consumer: REQ-8. |
| REQ-5 (error semantics: too-many-splits / too-large test_size+gap) | SHIPPED | impl `fn split_impl in time_series_split.rs` has no single global guard but enforces the SAME error set via the default-branch `default_ts == 0` check (`Err(InsufficientSamples)`), the per-fold `test_end = n_samples.checked_sub(windows_after*test_sz)` / `test_start = test_end.checked_sub(test_sz)` / `train_end = test_start.checked_sub(self.gap)` underflow `?`-returns, and the `if train_end == 0` guard. These collectively fire on EXACTLY sklearn's two `ValueError` conditions `n_folds > n_samples` (`sklearn/model_selection/_split.py:1246`) and `n_samples - gap - test_size*n_splits <= 0` (`:1251`) — proven by the 5600-case sweep finding 0 error-status divergences (sklearn-error ⟺ ferrolearn-`Err` on every input). Live oracle: `TimeSeriesSplit(n_splits=10).split(np.arange(5))`, `TimeSeriesSplit(n_splits=3, test_size=4).split(np.arange(12))`, `TimeSeriesSplit(n_splits=2, test_size=2, gap=100).split(np.arange(10))` all raise `ValueError`; ferrolearn returns `Err` on each (Verification). Type-name mapping (R-DEV-2): sklearn `ValueError` ↦ `FerroError::InsufficientSamples`/`InvalidParameter` — the variant differs but the reject behavior matches. Tests: `test_tss_insufficient_samples`, `test_tss_large_gap_error`, `test_tss_invalid_test_size_zero`. Non-test consumer: REQ-8. |
| REQ-6 (n_splits default + >=2 validation) | SHIPPED | impl `fn split_impl in time_series_split.rs` opens with `if self.n_splits < 2 { return Err(FerroError::InvalidParameter { name: "n_splits", reason: "must be >= 2, ..." }) }`, mirroring sklearn's `_BaseKFold` constraint `Interval(Integral, 2, None, closed="left")` (live oracle `TimeSeriesSplit(n_splits=1)` raises `ValueError`). API-shape note (honest underclaim): sklearn defaults `n_splits=5` at the constructor `sklearn/model_selection/_split.py:1183`; ferrolearn `pub fn new(n_splits: usize) in time_series_split.rs` takes `n_splits` POSITIONALLY with no default — a builder-API divergence (sklearn's `TimeSeriesSplit()` with no args is unrepresentable), not a behavioral one. Tests: `test_tss_invalid_n_splits_less_than_2`. Non-test consumer: REQ-8. |
| REQ-7 (R-SUBSTRATE) | SHIPPED | The owned computation in `fn split_impl in time_series_split.rs` is pure scalar `usize` arithmetic over `Vec<usize>` index ranges (`(train_start..train_end).collect()`); there is NO numpy-array computation to migrate — sklearn itself only uses `np.arange(n_samples)` plus Python slicing for index bookkeeping (`sklearn/model_selection/_split.py:1257`), which ferrolearn represents natively as `Range<usize>`. Production code imports only `ferrolearn_core::{FerroError, FerroResult}` and `crate::cross_validation::{CrossValidator, FoldSplits}`; the sole `ndarray` import is inside `#[cfg(test)] mod tests` (the `MeanEstimator` helper). R-SUBSTRATE-1's array/linalg/random/sparse analogs are not implicated by this unit. Verification: `grep` shows no `ndarray`/`faer`/`sprs`/`rand` in non-test code. |
| REQ-8 (non-test production consumer) | SHIPPED | `TimeSeriesSplit` implements `CrossValidator` (`impl CrossValidator for TimeSeriesSplit { fn fold_indices(...) { self.split_impl(n_samples) } } in time_series_split.rs`), and the non-test production CV-scoring functions consume any `&dyn CrossValidator` via `cv.fold_indices(n_samples)?`: `cross_val_score`/`cross_validate`/`cross_val_predict` (`cross_val_score in cross_validation.rs`, calls at the three `cv.fold_indices` sites), `learning_curve` (`fn learning_curve in learning_curve.rs`), and `validation_curve` (`fn validation_curve in validation_curve.rs`). Passing a `TimeSeriesSplit` to any of these is a real production integration path (the splitter IS the public boundary API per S5/R-DEFER-1 grandfathering). Crate re-export: `pub use time_series_split::TimeSeriesSplit in lib.rs`. Honest underclaim: there is no `ferrolearn-python` binding for `TimeSeriesSplit` yet, and the existing `#[test] test_tss_integrates_with_cross_val_score` exercises the trait path test-only; the production consumer is the polymorphic `&dyn CrossValidator` parameter, not a hardcoded internal call site. |

## Architecture

ferrolearn implements `TimeSeriesSplit` as a `#[derive(Clone)]` builder struct
holding `n_splits: usize`, `max_train_size: Option<usize>`, `test_size:
Option<usize>`, `gap: usize` (`struct TimeSeriesSplit in time_series_split.rs`).
The chainable setters `max_train_size`/`test_size`/`gap` and the constructor
`new(n_splits)` mirror sklearn's `__init__` keyword params
(`sklearn/model_selection/_split.py:1183`), except `n_splits` is positional with
no `5` default. All splitting flows through `fn split_impl`, exposed both as the
inherent `pub fn split(n_samples) -> FerroResult<FoldSplits>` and as the
`CrossValidator::fold_indices` trait method, so the splitter is usable
polymorphically by the CV-scoring functions (REQ-8).

The core algorithm is an end-anchored re-indexing of sklearn's forward
`test_starts` range. sklearn iterates `test_start ∈ range(n_samples -
n_splits*test_size, n_samples, test_size)` (`:1258`). ferrolearn instead iterates
fold `i ∈ 0..n_splits` and computes `windows_after = n_splits - 1 - i`,
`test_end = n_samples - windows_after*test_size`, `test_start = test_end -
test_size`. These are algebraically identical: ferrolearn's fold-`i`
`test_start` equals `n_samples - (n_splits - i)*test_size`, which is the `i`-th
element of sklearn's `range(...)`. `train_end = test_start - gap` matches
`:1261`; `train_start = train_end.saturating_sub(max_train_size)` (or `0`)
reproduces the `if max_train_size < train_end` conditional at `:1262` because
`saturating_sub` floors at 0 exactly when the cap would otherwise exceed the
available prefix. Both `train` and `test` are materialized as contiguous
`Range<usize>` collected into `Vec<usize>` — sorted-ascending by construction
(tests `test_tss_train_indices_are_sorted`/`test_tss_test_indices_are_sorted`).

Error handling diverges in MECHANISM but not in EXTENSION. sklearn computes two
up-front global guards (`:1246`, `:1251`) raising `ValueError`. ferrolearn has no
single global guard; instead it relies on the default-`test_size` zero check,
the `checked_sub` underflow `?`-propagation on `test_end`/`test_start`/
`train_end`, and the `train_end == 0` guard. The 5600-case oracle sweep proves
these fire on exactly the same input set — sklearn-`ValueError` ⟺ ferrolearn-`Err`
with no divergence. The error VARIANT differs (`FerroError::InsufficientSamples`/
`InvalidParameter` vs `ValueError`), the R-DEV-2 type-name mapping.

There is no array/linalg/random substrate in this unit (REQ-7): the computation
is scalar index arithmetic, so R-SUBSTRATE imposes no migration work — sklearn's
`np.arange`/slicing is bookkeeping that ferrolearn expresses as native ranges.

Invariants: `n_splits >= 2`; `test_size >= 1` (when explicit) else
`n_samples/(n_splits+1) >= 1`; for every fold `max(train) < min(test)` (chrono
ordering, `test_tss_test_indices_after_train_indices`); train and test index
lists are ascending; without `max_train_size`, training-set sizes grow strictly
across folds (`test_tss_training_grows_each_split`); ferrolearn `Err` ⟺ sklearn
`ValueError` on identical inputs.

## Verification

Commands establishing the SHIPPED claims (baseline
`3394bc3a4d76554a140c8040cf59c8bac7b0e7eb`):

- `cargo test -p ferrolearn-model-sel --lib time_series_split` → 18 passed,
  0 failed (the `time_series_split::tests::*` suite covering REQ-1..6).
- REQ-1..4 default/test_size/gap/max_train_size oracle (live sklearn 1.5.2):
  ```
  python3 -c "import numpy as np
  from sklearn.model_selection import TimeSeriesSplit
  for kw in [dict(n_splits=3), dict(n_splits=3,test_size=2), dict(n_splits=3,gap=2), dict(n_splits=3,max_train_size=4)]:
      print(kw)
      for tr,te in TimeSeriesSplit(**kw).split(np.arange(12)): print(' ', list(tr), list(te))"
  # n_splits=3            : ([0,1,2],[3,4,5]) ([0..6],[6,7,8]) ([0..9],[9,10,11])
  # n_splits=3,test_size=2: ([0..6],[6,7]) ([0..8],[8,9]) ([0..10],[10,11])
  # n_splits=3,gap=2      : ([0],[3,4,5]) ([0..4],[6,7,8]) ([0..7],[9,10,11])
  # n_splits=3,mts=4      : ([0,1,2],[3,4,5]) ([2,3,4,5],[6,7,8]) ([5,6,7,8],[9,10,11])
  ```
  Each matches the ferrolearn output; pin `#[test]`s asserting the exact index
  lists above (R-CHAR-3, oracle-derived). `test_tss_n_splits_2_minimal` already
  pins exact lists for `n_splits=2, test_size=2, n=6`.
- REQ-1..5 EXHAUSTIVE sweep (the strong claim — 5600 cases, live oracle):
  ```
  python3  # dump list(TimeSeriesSplit(...).split(np.arange(n))) for
           # n in 2..16, n_splits in 2..6, test_size in {None,1,2,3,4},
           # gap in {0,1,2,3}, max_train_size in {None,1,2,3,4}; compare to a
           # port of split_impl
  # -> error-status divergences: 0
  # -> fold-content divergences: 0
  ```
  This pins REQ-5 (error semantics): ferrolearn returns `Err` on exactly
  sklearn's `ValueError` set (`n_folds > n_samples` `:1246`; `n_samples - gap -
  test_size*n_splits <= 0` `:1251`).
- REQ-5/6 error oracle (live sklearn): `TimeSeriesSplit(n_splits=10).split(
  np.arange(5))`, `TimeSeriesSplit(n_splits=3, test_size=4).split(np.arange(12))`,
  `TimeSeriesSplit(n_splits=2, test_size=2, gap=100).split(np.arange(10))`, and
  `TimeSeriesSplit(n_splits=1)` all raise `ValueError`; the ferrolearn analogs all
  return `Err` (tests `test_tss_insufficient_samples`, `test_tss_large_gap_error`,
  `test_tss_invalid_test_size_zero`, `test_tss_invalid_n_splits_less_than_2`).
- REQ-7 substrate: `grep -n "ndarray\|faer\|sprs\|rand" time_series_split.rs`
  matches only the `#[cfg(test)]` `use ndarray::{Array1, Array2}` — no
  wrong-substrate usage in production index math.
- REQ-8 consumer: `grep -n "fold_indices" cross_validation.rs learning_curve.rs
  validation_curve.rs` shows the production `cv.fold_indices(n_samples)?` call
  sites that accept `TimeSeriesSplit` as `&dyn CrossValidator`.

SHIPPED: REQ-1 (default index parity), REQ-2 (test_size), REQ-3 (gap), REQ-4
(max_train_size, incl. the no-cap edge), REQ-5 (error semantics — Err ⟺
ValueError on all 5600 cases), REQ-6 (n_splits>=2 reject; API-shape note on the
missing `5` default), REQ-7 (no substrate migration owed — scalar index math),
REQ-8 (polymorphic `&dyn CrossValidator` production consumer; no Python binding
yet — honest underclaim). NOT-STARTED: none. Per R-DEFER-2 the table is binary
SHIPPED/NOT-STARTED. Least-confident SHIPPED claim: REQ-8 — the consumer is the
generic `&dyn CrossValidator` parameter rather than a hardcoded internal
`TimeSeriesSplit` call site, and there is no `ferrolearn-python` binding; it is
SHIPPED on the boundary-public-API + trait-integration path per S5, not a
dedicated caller.
