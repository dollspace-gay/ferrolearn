# Group-aware cross-validation splitters (GroupKFold / StratifiedGroupKFold / LeaveOneGroupOut / LeavePGroupsOut / GroupShuffleSplit)

<!--
tier: 3-component
status: draft
baseline-commit: d12a2ba22d20ca29ab3e662355a51e32c11e77bb
upstream-paths:
  - sklearn/model_selection/_split.py   # GroupKFold (:537), StratifiedGroupKFold (:856), LeaveOneGroupOut (:1274), LeavePGroupsOut (:1393), GroupShuffleSplit (:1994), _validate_shuffle_split (:2343)
-->

## Summary

`ferrolearn-model-sel/src/group_splitters.rs` mirrors five group-aware
cross-validation splitters from `sklearn/model_selection/_split.py`. Each takes an
`Array1<usize>` of per-sample group labels and produces folds in which samples
sharing a group never straddle the train/test boundary:

- `GroupKFold` (`:537`) — greedy balanced partition of groups into `n_splits` folds.
- `StratifiedGroupKFold` (`:856`) — group-aware k-fold that also preserves per-class
  proportions via a std-minimisation heuristic.
- `LeaveOneGroupOut` (`:1274`) — one fold per unique group.
- `LeavePGroupsOut` (`:1393`) — one fold per `p`-subset of groups.
- `GroupShuffleSplit` (`:1994`) — random group-wise train/test splits (inner
  ShuffleSplit on group ids).

They divide into three determinism classes:

- **DETERMINISTIC / oracle-pinnable end-to-end** — `LeaveOneGroupOut`,
  `LeavePGroupsOut`. Fold membership is a pure function of `(groups, params)`;
  every REQ is pinnable against `list(Splitter().split(X, y, groups))`.
- **DETERMINISTIC-in-sklearn but DIVERGENT here** — `GroupKFold` (tie-break /
  iteration-order divergence makes ferrolearn's greedy assignment
  non-deterministic for equal-size groups — REQ-GKF-2) and `StratifiedGroupKFold`
  (a completely different greedy objective — REQ-SGKF-1). sklearn is deterministic;
  ferrolearn does NOT match, so these carry NOT-STARTED REQs with concrete blockers.
- **RNG carve-out (membership) + DETERMINISTIC sizing** — `GroupShuffleSplit`. The
  selected groups depend on the random permutation stream (numpy `Mersenne` vs Rust
  `SmallRng`), so exact membership is an R-DEFER-3 carve-out (blocker, NO failing
  test). The NUMBER of test groups is deterministic and pinnable — and CURRENTLY
  DIVERGES (ferrolearn `round`, sklearn `ceil`; REQ-GSS-2).

All five expose an inherent `split(...)` returning `FoldSplits = Vec<(Vec<usize>,
Vec<usize>)>` (`crate::cross_validation::FoldSplits`). None implements
`CrossValidator` (whose `fold_indices(n_samples)` has no group/label channel), so
none is reachable from `cross_val_score`/`cross_validate`/`cross_val_predict`.
Their sole non-test production consumer is the crate re-export boundary API
(`pub use group_splitters::{…} in lib.rs` — the boundary public surface per S5).

## Upstream reference (scikit-learn 1.5.2, commit 156ef14)

### GroupKFold (`sklearn/model_selection/_split.py:537`)
- `:597-598` — `__init__(self, n_splits=5)` → `super().__init__(n_splits,
  shuffle=False, random_state=None)`. Default `n_splits=5`.
- `:608-612` — `if self.n_splits > n_groups: raise ValueError("Cannot have number
  of splits n_splits=%d greater than the number of groups: %d")`.
- `:614-636` — the greedy assignment:
  - `:615` `n_samples_per_group = np.bincount(groups)` (groups remapped to dense
    indices by `np.unique(..., return_inverse=True)` at `:605`).
  - `:618` `indices = np.argsort(n_samples_per_group)[::-1]` — process groups by
    DESCENDING size; **ties break by descending group-index** (`argsort` ascending,
    then `[::-1]`).
  - `:628-631` for each group in that order, `lightest_fold =
    np.argmin(n_samples_per_fold)` (first fold of minimum load), add the group's
    weight there, record `group_to_fold`.
  - `:635-636` `yield np.where(indices == f)[0]` for `f in range(n_splits)` — test
    fold `f` = samples whose group landed in fold `f`.
- Docstring example (`:566-585`): `groups=[0,0,2,2,3,3]`, `n_splits=2` → fold0 test
  groups `{0,3}`, fold1 test group `{2}`.

### StratifiedGroupKFold (`sklearn/model_selection/_split.py:856`)
- `:955-956` — `__init__(self, n_splits=5, shuffle=False, random_state=None)`.
  Default `shuffle=False` → DETERMINISTIC.
- `:986-1000` — class checks: `n_splits` cannot exceed members of each class
  (`:987-991`); `UserWarning` if least-populated class < `n_splits` (`:993-999`).
- `:1002-1007` — `y_counts_per_group[group_idx, class_idx]`: per-group class-count
  matrix.
- `:1015-1019` — **sort key**: `sorted_groups_idx = np.argsort(-np.std(
  y_counts_per_group, axis=1), kind="mergesort")` — descending per-group STD of its
  class counts, STABLE (mergesort).
- `:1021-1029` — for each group in that order, `best_fold = _find_best_fold(...)`,
  add the group's class-counts there.
- `:1039-1059` — `_find_best_fold`: tentatively add `group_y_counts` to each fold,
  compute `std_per_class = np.std(y_counts_per_fold / y_cnt, axis=0)`, score =
  `np.mean(std_per_class)`; pick the fold MINIMISING that mean-of-per-class-std,
  **tie-break (`np.isclose`) by fewest samples in fold** (`:1049-1054`).
- Docstring example (`:903-930`) and the live oracle on it (n_splits=3) →
  fold0 test groups `{3,6,7}`, fold1 `{1,2,8}`, fold2 `{4,5}`.

### LeaveOneGroupOut (`sklearn/model_selection/_split.py:1274`)
- `:1323-1337` — `_iter_test_masks`: `unique_groups = np.unique(groups)`; raise
  `ValueError` if `len(unique_groups) <= 1` (`:1331-1335`); then `for i in
  unique_groups: yield groups == i`. ORDER = sorted unique groups (ascending).
- `:1339-1364` — `get_n_splits` returns `len(np.unique(groups))`.
- Docstring example (`:1296-1316`): `groups=[1,1,2,2]` → fold0 test `[0,1]`, fold1
  test `[2,3]`.

### LeavePGroupsOut (`sklearn/model_selection/_split.py:1393`)
- `:1448-1449` — `__init__(self, n_groups)`; `n_groups` is `p`.
- `:1451-1470` — `_iter_test_masks`: `unique_groups = np.unique(groups)`; raise
  `ValueError` if `self.n_groups >= len(unique_groups)` (`:1458-1464`, needs
  `n_groups + 1` unique groups); then `combi = combinations(range(len(
  unique_groups)), self.n_groups)` and for each, mark all samples in those groups.
  ORDER = `itertools.combinations` LEXICOGRAPHIC over sorted-unique-group indices.
- Docstring example (`:1418-1441`): `groups=[1,2,3]`, `n_groups=2` → 3 folds, test
  groups `{1,2},{1,3},{2,3}`.

### GroupShuffleSplit (`sklearn/model_selection/_split.py:1994`) + `_validate_shuffle_split`
- `:2082-2091` — `__init__(self, n_splits=5, *, test_size=None, train_size=None,
  random_state=None)`; `self._default_test_size = 0.2`. Default `n_splits=5`,
  default effective `test_size=0.2`.
- `:2093-2105` — `_iter_indices`: `classes, group_indices = np.unique(groups,
  return_inverse=True)`; run the inner `BaseShuffleSplit._iter_indices(X=classes)`
  — i.e. a ShuffleSplit OVER THE UNIQUE GROUP IDS — then map the selected group
  partition back to sample indices via `np.isin`. `test_size`/`train_size` count
  GROUPS, not samples (`:2018-2019`).
- `_validate_shuffle_split` (`:2343`): for a float `test_size`, `n_test =
  ceil(test_size * n_samples)` (`:2389-2390`) where here `n_samples == n_groups`;
  raises `ValueError` if `test_size <= 0 or test_size >= 1` (`:2357-2358`) and if
  the resulting `n_train == 0` (`:2414-2419`).
- Docstring example (`:2059-2073`): `groups=[1,1,2,2,2,3,3,3]` (3 groups),
  `train_size=.7, random_state=42` → fold0 test group `{1}`, fold1 test group `{2}`.

## Requirements

### GroupKFold (DETERMINISTIC in sklearn; DIVERGENT here)
- REQ-GKF-1 (greedy mechanics — argmin load, no group split): for each group (in
  some descending-size order) assign it to the fold of currently-smallest load via
  first-min argmin; a group's samples never straddle folds; `n_splits` folds; reject
  `n_splits < 2` and `n_splits > n_groups`. Mirrors `:614-636`, `:608-612`. The
  per-step argmin tie-break (first fold of minimum load) matches sklearn's
  `np.argmin`. STRUCTURAL part SHIPPED.
- REQ-GKF-2 (greedy ORDERING / tie-break parity — THE KEY REQ, DETERMINISTIC,
  oracle-pinnable): sklearn processes groups in `argsort(n_samples_per_group)[::-1]`
  order — descending size, **ties broken by descending group-index** (`:618`).
  ferrolearn builds `ordered` from a `HashMap<usize,usize>` collected into a `Vec`
  then `sort_by_key(Reverse(count))` — Rust's sort is stable but the HashMap
  iteration order feeding it is NON-DETERMINISTIC, and even when stabilised it does
  not reproduce sklearn's descending-group-index tie-break. For equal-size groups
  the fold membership is therefore both non-deterministic AND mismatched. NOT-STARTED.
- REQ-GKF-3 (default n_splits): sklearn `n_splits=5` (`:597`); ferrolearn `new(
  n_splits)` requires it explicitly — API-shape gap. NOT-STARTED.

### StratifiedGroupKFold (DETERMINISTIC in sklearn; DIVERGENT here)
- REQ-SGKF-1 (greedy objective parity — DETERMINISTIC, oracle-pinnable): sklearn
  sorts groups by DESCENDING per-group STD-of-class-counts (mergesort stable,
  `:1017-1019`) and assigns each to the fold minimising the MEAN of per-class STD of
  `y_counts_per_fold / y_cnt`, tie-broken (`np.isclose`) by fewest samples
  (`_find_best_fold`, `:1039-1059`). ferrolearn sorts by descending group SIZE and
  minimises the SUM-OF-SQUARED deviation of each fold's class counts from a uniform
  `target_per_fold`. These are different sort keys AND different objectives; the
  result diverges from the live oracle on the sklearn docstring example.
  NOT-STARTED.
- REQ-SGKF-2 (structural + validation): `n_splits` folds; whole groups, no straddle;
  `y.len() == groups.len()` enforced; reject `n_splits < 2` and `n_splits >
  n_groups`. Mirrors the non-overlap contract + `:987-991`. STRUCTURAL part SHIPPED.
- REQ-SGKF-3 (per-class-count validation + UserWarning + default n_splits): sklearn
  rejects when `n_splits > members of each class` (`:987-991`) and emits a
  `UserWarning` when the least-populated class < `n_splits` (`:993-999`); default
  `n_splits=5`. ferrolearn validates only `n_splits <= n_groups`, has no
  per-class-count check, no warning, and requires explicit `n_splits`. NOT-STARTED.

### LeaveOneGroupOut (DETERMINISTIC)
- REQ-LOGO-1 (split-index parity, oracle-pinnable): one fold per unique group in
  ASCENDING group order; fold for group `g` has `test = {i : groups[i]==g}`, train =
  rest (ascending by construction). Mirrors `:1330-1337`. Oracle-pinnable
  (`LeaveOneGroupOut().split(X, groups=[0,0,1,1,2])`). SHIPPED.
- REQ-LOGO-2 (`< 2` unique-groups rejection): reject when `len(unique) < 2`. Mirrors
  `:1331-1335` (`fewer than 2 unique groups`). SHIPPED.

### LeavePGroupsOut (DETERMINISTIC)
- REQ-LPGO-1 (combination-order index parity, oracle-pinnable): `C(n_groups, p)`
  folds; test set = each `p`-combination of the sorted unique groups in
  `itertools.combinations` LEXICOGRAPHIC order; train = rest. Mirrors `:1465-1470`.
  Oracle-pinnable (`LeavePGroupsOut(2).split(X, groups=[0,1,2,3])` → 6 folds in
  combination order). SHIPPED.
- REQ-LPGO-2 (error semantics): `p == 0` rejected; `n_unique_groups <= p` rejected
  (sklearn needs `n_groups + 1` unique groups, i.e. `p >= len(unique)` raises).
  Mirrors `:1458-1464`. SHIPPED.

### GroupShuffleSplit (RNG carve-out + sizing)
- REQ-GSS-1 (structural contract): `n_splits` folds; each selects a subset of unique
  GROUPS at random, `test` = all samples in the selected groups, `train` = the rest
  (disjoint, covering at sample level for a single split); `random_state` makes the
  stream reproducible. Mirrors `:2093-2105`. STRUCTURAL part SHIPPED.
- REQ-GSS-2 (test-GROUP-count sizing parity — DETERMINISTIC, oracle-pinnable):
  sklearn `n_test = ceil(test_size * n_groups)` (`:2390`). ferrolearn computes
  `n_test = ((n_groups as f64) * self.test_size).round().max(1.0) as usize`. DIVERGES:
  `n_groups=7, test_size=0.3` → sklearn `ceil(2.1)=3`, ferrolearn `round(2.1)=2`
  (same `ceil`-vs-`round` bug class as `train_test_split` / `ShuffleSplit`).
  NOT-STARTED. Oracle-pinnable on the COUNT of unique test groups per split.
- REQ-GSS-3 (default test_size + n_splits + exact-membership carve-out): (a) sklearn
  `GroupShuffleSplit(n_splits=5, *, test_size=None → 0.2, …)` (`:2082-2091`,
  `_default_test_size = 0.2`); ferrolearn `pub fn new(n_splits, test_size: f64)`
  requires an explicit `f64` and an explicit `n_splits` — the no-arg sklearn form is
  unrepresentable. (b) exact GROUP SELECTION is an RNG carve-out (numpy permutation
  vs Rust `SmallRng::seed_from_u64(seed + split)` + `SliceRandom::shuffle`): per
  R-DEFER-3 a blocker, NO failing test. NOT-STARTED.

### Cross-cutting
- REQ-X-1 (R-SUBSTRATE): production code imports `ndarray::Array1` (array type),
  `rand::{SeedableRng, rngs::SmallRng, seq::SliceRandom}` (random substrate), and
  `std::collections::{BTreeMap, HashMap, HashSet}`. The destination substrate is
  `ferray-core` / `ferray::random` (R-SUBSTRATE-1). NOT-STARTED (migration owed).
- REQ-X-2 (non-test production consumer): all five splitters are re-exported at
  `pub use group_splitters::{GroupKFold, GroupShuffleSplit, LeaveOneGroupOut,
  LeavePGroupsOut, StratifiedGroupKFold} in lib.rs` — the boundary public API per
  S5/R-DEFER-1 grandfathering. None implements `CrossValidator` (its
  `fold_indices(n_samples)` has no group/label channel), so none is reachable from
  `cross_val_score`/`cross_validate`/`cross_val_predict`; the re-export is the sole
  production reach. SHIPPED (re-export boundary only — honest underclaim).

## Acceptance criteria

- AC-GKF-1 (REQ-GKF-1): `GroupKFold::new(3).split(&array![0,0,1,1,2,2,3,3])` → 3
  folds, each fold's test groups disjoint from its train groups, no group split.
  STRUCTURAL.
- AC-GKF-2 (REQ-GKF-2): live oracle `GroupKFold(n_splits=3).split(X,
  groups=[0,0,1,1,2,2,3,3])` → fold0 test groups `{0,3}`, fold1 `{2}`, fold2 `{1}`
  (descending-group-index tie-break). ferrolearn's HashMap-ordered greedy does NOT
  reproduce this membership deterministically. DIVERGENCE. NOT-STARTED.
- AC-SGKF-1 (REQ-SGKF-1): live oracle `StratifiedGroupKFold(n_splits=3).split(X, y,
  groups)` on the sklearn docstring example (`y`/`groups` at `:903-904`) → fold0 test
  groups `{3,6,7}`, fold1 `{1,2,8}`, fold2 `{4,5}`. ferrolearn's size-sort +
  squared-deviation heuristic does NOT reproduce this. DIVERGENCE. NOT-STARTED.
- AC-LOGO-1 (REQ-LOGO-1): `LeaveOneGroupOut::new().split(&array![0,0,1,1,2])` → fold0
  `(train=[2,3,4], test=[0,1])`, fold1 `([0,1,4],[2,3])`, fold2 `([0,1,2,3],[4])` —
  equal to the live oracle. DETERMINISTIC.
- AC-LOGO-2 (REQ-LOGO-2): `LeaveOneGroupOut::new().split(&array![0,0,0])` → `Err`
  (sklearn raises `ValueError` for `< 2` unique groups).
- AC-LPGO-1 (REQ-LPGO-1): `LeavePGroupsOut::new(2).split(&array![0,1,2,3])` → 6 =
  `C(4,2)` folds, test sets in order `{0,1},{0,2},{0,3},{1,2},{1,3},{2,3}` with
  complementary train sets — equal to the live oracle. DETERMINISTIC.
- AC-LPGO-2 (REQ-LPGO-2): `LeavePGroupsOut::new(0).split(g)` → `Err`;
  `LeavePGroupsOut::new(4).split(&array![0,1,2,3])` → `Err` (4 groups, p=4 needs ≥5).
- AC-GSS-1 (REQ-GSS-1): `GroupShuffleSplit::new(2, 0.5).random_state(7).split(g)` →
  2 folds, each test/train disjoint at sample level, reproducible across calls with
  the same seed. STRUCTURAL.
- AC-GSS-2 (REQ-GSS-2): on `n_groups=7, test_size=0.3` the number of unique test
  groups per split is CURRENTLY 2 but sklearn `GroupShuffleSplit(test_size=0.3)` has
  3 (`ceil(2.1)`). DIVERGENCE. NOT-STARTED.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-GKF-1 (greedy mechanics: argmin load, no group split, validation) | SHIPPED | impl `pub fn split in group_splitters.rs` (`GroupKFold`) computes per-group `sizes: HashMap`, then for each group picks the fold of smallest current load via a first-`<` argmin scan (`min_idx`/`min_val` loop), increments that fold's load, and finally emits `test`/`train` so a group's samples never straddle. This mirrors sklearn's `lightest_fold = np.argmin(n_samples_per_fold)` + per-group accumulation (`sklearn/model_selection/_split.py:628-636`); the first-minimum argmin matches `np.argmin`. Rejects `n_splits < 2` (`InvalidParameter`) and `n_splits > n_groups` (`InvalidParameter`, mirroring `:608-612`). Test: `group_kfold_partitions_groups` (disjoint train/test groups per fold). Non-test consumer: REQ-X-2 (re-export). STRUCTURAL ONLY — exact fold membership is REQ-GKF-2 (NOT-STARTED). |
| REQ-GKF-2 (greedy ORDERING / tie-break parity — KEY REQ, DETERMINISTIC) | NOT-STARTED | DIVERGENCE. sklearn processes groups in `np.argsort(n_samples_per_group)[::-1]` order — descending size, ties by DESCENDING group-index (`sklearn/model_selection/_split.py:618`). impl `pub fn split in group_splitters.rs` builds `ordered: Vec<(usize,usize)>` by collecting a `HashMap<usize,usize>` and `sort_by_key(Reverse(count))`: the HashMap iteration order feeding the (stable) sort is NON-DETERMINISTIC, and even stabilised it does not reproduce sklearn's descending-group-index tie-break. Live oracle `GroupKFold(n_splits=3).split(X, groups=[0,0,1,1,2,2,3,3])` → fold0 test groups `{0,3}`, fold1 `{2}`, fold2 `{1}`; ferrolearn's membership for equal-size groups is unstable/mismatched. Blocker (tracking #1749; critic files the failing-test spec): replace HashMap-collected `ordered` with a deterministic descending-(size, then descending-group-index) order over `bincount`-style dense group ids. Oracle-pinnable on exact fold membership once fixed. |
| REQ-GKF-3 (default n_splits) | NOT-STARTED | API-shape gap. sklearn `GroupKFold(n_splits=5)` (`sklearn/model_selection/_split.py:597`) defaults `n_splits=5`; ferrolearn `pub fn new(n_splits) in group_splitters.rs` requires it positionally — the no-arg sklearn form is unrepresentable. Blocker (#1749): constructor default surface. |
| REQ-SGKF-1 (greedy objective parity — DETERMINISTIC) | NOT-STARTED | DIVERGENCE. sklearn sorts groups by DESCENDING per-group STD of class counts (mergesort stable, `sklearn/model_selection/_split.py:1017-1019`) and assigns each to the fold minimising the MEAN of per-class STD of `y_counts_per_fold / y_cnt`, tie-broken by fewest samples (`_find_best_fold`, `:1039-1059`). impl `pub fn split in group_splitters.rs` (`StratifiedGroupKFold`) instead sorts by descending group SIZE (`ordered.sort_by(... sb.cmp(&sa))`) and minimises the SUM-OF-SQUARED deviation of fold class-counts from a uniform `target_per_fold = total_per_class / n_splits` — a different sort key AND a different objective. Live oracle on the sklearn docstring example (`:903-930`, n_splits=3) → fold0 test groups `{3,6,7}`, fold1 `{1,2,8}`, fold2 `{4,5}`; ferrolearn does not reproduce this. Blocker (#1749; critic files the failing-test spec): port sklearn's std-of-class-distribution sort + `_find_best_fold` mean-std objective with the min-samples tie-break. Oracle-pinnable on exact fold membership once ported. |
| REQ-SGKF-2 (structural + length/n_splits validation) | SHIPPED | impl `pub fn split in group_splitters.rs` (`StratifiedGroupKFold`) returns `Err(FerroError::ShapeMismatch{..})` when `y.len() != groups.len()`, rejects `n_splits < 2` and `n_splits > n_groups` (`InvalidParameter`), assigns whole groups (`group_to_fold`) so no group straddles, and emits `n_splits` folds. The non-overlap contract mirrors `:863-864`; the n_splits>n_groups guard mirrors the `_BaseKFold` group-count constraint. Test: `stratified_group_kfold_balances` (disjoint train/test groups per fold), `stratified_group_kfold_shape_mismatch` (`y`/`groups` length mismatch → `Err`). Non-test consumer: REQ-X-2 (re-export). STRUCTURAL ONLY — exact membership is REQ-SGKF-1 (NOT-STARTED). |
| REQ-SGKF-3 (per-class-count validation + UserWarning + default n_splits) | NOT-STARTED | sklearn rejects `n_splits > members of each class` (`sklearn/model_selection/_split.py:987-991`) and emits a `UserWarning` when the least-populated class < `n_splits` (`:993-999`); default `n_splits=5` (`:955`). impl `pub fn split in group_splitters.rs` validates only `n_splits <= n_groups`, has no per-class-member check, no warning channel, and requires explicit `n_splits`. Blocker (#1749): per-class-count `ValueError` + least-populated-class warning + default `n_splits`. |
| REQ-LOGO-1 (split-index parity, DETERMINISTIC) | SHIPPED | impl `pub fn split in group_splitters.rs` (`LeaveOneGroupOut`) iterates `unique_groups(groups)` (sorted ascending, deduped) and for each `target` emits `test = {i : groups[i]==target}`, `train` = the rest (ascending by construction over `0..n`) — exactly sklearn's `for i in unique_groups: yield groups == i` over `np.unique(groups)` (`sklearn/model_selection/_split.py:1330-1337`). Live oracle `LeaveOneGroupOut().split(X, groups=[0,0,1,1,2])` → fold0 `([2,3,4],[0,1])`, fold1 `([0,1,4],[2,3])`, fold2 `([0,1,2,3],[4])` matches. Test: `leave_one_group_out_one_fold_per_group` (3 folds for 3 groups). Oracle-pinnable: a `#[test]` asserting the exact 3 ordered folds above (R-CHAR-3). Non-test consumer: REQ-X-2 (re-export). |
| REQ-LOGO-2 (`< 2` unique-groups rejection) | SHIPPED | impl `pub fn split in group_splitters.rs` returns `Err(FerroError::InvalidParameter{ name: "n_groups", .. })` when `unique.len() < 2`, mirroring sklearn `:1331-1335` (`if len(unique_groups) <= 1: raise ValueError("fewer than 2 unique groups …")`). R-DEV-2 type mapping `ValueError` ↦ `FerroError::InvalidParameter`. Live oracle `LeaveOneGroupOut().split(X, groups=[0,0,0])` raises `ValueError`; ferrolearn → `Err`. Pinnable. Non-test consumer: REQ-X-2. |
| REQ-LPGO-1 (combination-order parity, DETERMINISTIC) | SHIPPED | impl `pub fn split in group_splitters.rs` (`LeavePGroupsOut`) initialises `combo = (0..p).collect()` over the sorted-unique-group INDICES and advances it via the standard lexicographic next-combination loop (`while i > 0 { if combo[i] < n_g - p + i { combo[i]+=1; reset suffix } }`), each iteration marking `test = {i : groups[i] ∈ unique[combo]}`. This reproduces `combinations(range(len(unique_groups)), n_groups)` order exactly (`sklearn/model_selection/_split.py:1465-1470`). Live oracle `LeavePGroupsOut(2).split(X, groups=[0,1,2,3])` → 6 folds, test groups in order `{0,1},{0,2},{0,3},{1,2},{1,3},{2,3}` matches. Test: `leave_p_groups_out_combinations` (6 = `C(4,2)`). Oracle-pinnable: a `#[test]` asserting the exact ordered test-group sequence (R-CHAR-3). Non-test consumer: REQ-X-2. |
| REQ-LPGO-2 (error semantics) | SHIPPED | impl `pub fn split in group_splitters.rs` returns `Err(FerroError::InvalidParameter{ name: "p", .. })` when `self.p == 0`, and when `unique.len() <= self.p` — mirroring sklearn `:1458-1464` (`if self.n_groups >= len(unique_groups): raise ValueError("… expects at least n_groups + 1 unique groups")`). R-DEV-2 type mapping. Live oracle: `LeavePGroupsOut(4).split(X, groups=[0,1,2,3])` raises `ValueError` (4 groups, p=4 needs ≥5). ferrolearn `LeavePGroupsOut::new(4).split(&array![0,1,2,3])` → `Err`; `new(0)` → `Err`. Pinnable. Non-test consumer: REQ-X-2. |
| REQ-GSS-1 (GroupShuffleSplit structural, STRUCTURAL) | SHIPPED | impl `pub fn split in group_splitters.rs` (`GroupShuffleSplit`) shuffles `unique` group ids with a per-split `SmallRng::seed_from_u64(seed + split)`, takes the first `n_test` as the test-group set, then emits `test = {i : groups[i] ∈ test_groups}`, `train` = rest — a sample-level disjoint covering partition per split, mirroring sklearn's inner ShuffleSplit on group ids + `np.isin` mapping (`sklearn/model_selection/_split.py:2097-2105`). `random_state(seed)` makes the stream reproducible (R-DEV-1 determinism; mechanism differs from numpy). Rejects `test_size ∉ (0,1)` (`InvalidParameter`, mirroring `_validate_shuffle_split` `:2357-2358`) and `< 2` groups. Structural facts (fold count, partition, reproducibility) hold; exact membership is the REQ-GSS-3 carve-out. Test: `group_shuffle_split_deterministic` (same seed → identical splits). Non-test consumer: REQ-X-2 (re-export). |
| REQ-GSS-2 (test-GROUP-count sizing parity, DETERMINISTIC) | NOT-STARTED | DIVERGENCE. impl `pub fn split in group_splitters.rs` computes `n_test = ((n_groups as f64) * self.test_size).round().max(1.0) as usize` — uses `.round()`. sklearn's `_validate_shuffle_split` computes `n_test = ceil(test_size * n_samples)` where `n_samples == n_groups` (`sklearn/model_selection/_split.py:2389-2390`). Live oracle: `GroupShuffleSplit(test_size=0.3).split(X, groups=[7 groups])` → 3 unique test groups (`ceil(2.1)`); ferrolearn → `round(2.1)=2`. Same `ceil`-vs-`round` bug class as `train_test_split`/`ShuffleSplit`. Blocker (#1749; critic files the failing-test spec): `round` → `ceil` in the `n_test` computation. Oracle-pinnable on the COUNT of unique test groups once fixed. |
| REQ-GSS-3 (default test_size + n_splits + exact-membership carve-out) | NOT-STARTED | Two gaps. (a) API-shape: sklearn `GroupShuffleSplit(n_splits=5, *, test_size=None → 0.2, …)` (`sklearn/model_selection/_split.py:2082-2091`, `_default_test_size=0.2`) has a default `test_size` AND a default `n_splits`; ferrolearn `pub fn new(n_splits, test_size: f64) in group_splitters.rs` requires both explicitly — the no-arg sklearn form is unrepresentable. (b) Exact GROUP SELECTION is an RNG carve-out (numpy `permutation` vs Rust `SmallRng` + `SliceRandom::shuffle`): per R-DEFER-3 a blocker but NO failing test. Blocker (#1749): default `test_size=0.2` / default `n_splits=5` constructor surface + RNG-stream membership carve-out documented. |
| REQ-X-1 (R-SUBSTRATE) | NOT-STARTED | Production code in `group_splitters.rs` imports `ndarray::Array1` (array type), `rand::{SeedableRng, rngs::SmallRng, seq::SliceRandom}` (random substrate), and `std::collections::{BTreeMap, HashMap, HashSet}`. Per R-SUBSTRATE-1 the destination is `ferray-core` (array) and `ferray::random` (sampling); these are the wrong substrate. Blocker (#1749): migrate `ndarray::Array1` → `ferray-core` and `rand`/`SmallRng` → `ferray::random`. Until then this unit is not on the ferray substrate (R-SUBSTRATE-2). |
| REQ-X-2 (non-test production consumer) | SHIPPED | All five splitters are re-exported at `pub use group_splitters::{GroupKFold, GroupShuffleSplit, LeaveOneGroupOut, LeavePGroupsOut, StratifiedGroupKFold} in lib.rs` — the boundary public API per S5/R-DEFER-1 grandfathering. Honest underclaim: NONE implements `CrossValidator` (whose `fold_indices(n_samples)` has no group/label channel), so none is reachable from `cross_val_score`/`cross_validate`/`cross_val_predict` as a `&dyn CrossValidator`; the sole production reach is the re-export boundary. No `ferrolearn-python` binding exists for any of the five. |

## Architecture

ferrolearn implements all five splitters as `#[derive(Debug, Clone)]` builder
structs in `group_splitters.rs`, each with an inherent `split(...)` returning
`FoldSplits = Vec<(Vec<usize>, Vec<usize>)>` (`crate::cross_validation::FoldSplits`).
`StratifiedGroupKFold::split` takes `(&y, &groups)`; the other four take `&groups`.
Shared helpers: `unique_groups` (sorted-deduped group list) and `check_non_empty`
(empty-input `InsufficientSamples`).

Unlike the index-only splitters in `splitters.rs`, NONE of the group splitters
implements `CrossValidator`: that trait's `fold_indices(n_samples)` carries no group
(or label) channel, so the group splitters cannot be passed to
`cross_val_score`/`cross_validate`/`cross_val_predict` as `&dyn CrossValidator`.
Their entire production reach is the crate re-export (`lib.rs`,
`pub use group_splitters::{…}` — REQ-X-2). This is the honest underclaim: the public
boundary surface, not a polymorphic CV call site.

The two unconditionally-DETERMINISTIC splitters are oracle-pinnable end to end.
`LeaveOneGroupOut` emits one fold per sorted-unique group (`:1330-1337`).
`LeavePGroupsOut` runs an explicit lexicographic next-combination loop reproducing
`combinations(range(n_groups), p)` ORDER (`:1465`) — the order, not just the
`C(n_groups, p)` count, is the contract; train sets are the complements.

`GroupKFold` and `StratifiedGroupKFold` are DETERMINISTIC in sklearn but DIVERGE
here, so each carries a NOT-STARTED parity REQ alongside a SHIPPED structural REQ:

- `GroupKFold` (REQ-GKF-2): sklearn's greedy processes groups in
  `argsort(bincount)[::-1]` order — descending size with ties broken by descending
  group-index — then assigns each to the lightest fold (`np.argmin`). ferrolearn's
  per-step argmin matches, but it derives the group ORDER from a `HashMap` collected
  into a `Vec` then stable-sorted by descending count: the HashMap iteration order is
  non-deterministic, so for equal-size groups the fold membership is both unstable
  AND mismatched against sklearn's descending-index tie-break. THE KEY REQ.

- `StratifiedGroupKFold` (REQ-SGKF-1): sklearn sorts by descending per-group STD of
  class counts (stable mergesort) and minimises, per candidate fold, the MEAN of the
  per-class STD of `y_counts_per_fold / y_cnt` with a fewest-samples tie-break
  (`_find_best_fold`, `:1039-1059`). ferrolearn sorts by descending group SIZE and
  minimises the SUM-OF-SQUARED deviation of fold class-counts from a uniform target.
  Different sort key, different objective — it does not reproduce the live oracle on
  the sklearn docstring example.

`GroupShuffleSplit` selects a random subset of unique GROUPS (an inner ShuffleSplit
on group ids, `:2093-2105`); the test set is all samples of the selected groups. Two
behaviors separate cleanly: (1) the test-GROUP COUNT is deterministic and DIVERGES —
ferrolearn `round((n_groups * test_size))`, sklearn `ceil(test_size * n_groups)`
(`:2390`); `n_groups=7, test_size=0.3` → 3 vs 2 (REQ-GSS-2, NOT-STARTED,
oracle-pinnable on the COUNT). (2) the exact group SELECTION is an RNG carve-out —
ferrolearn seeds a per-split `SmallRng::seed_from_u64(seed + split)` and
`SliceRandom::shuffle`, a stream that differs from numpy's `permutation`; per
R-DEFER-3 the membership carve-out gets a blocker but NO failing test (REQ-GSS-3).

Error handling maps sklearn `ValueError` to `FerroError` variants (R-DEV-2 type
mapping): empty groups → `InsufficientSamples`; `GroupKFold`/`StratifiedGroupKFold`
`n_splits < 2` or `n_splits > n_groups` → `InvalidParameter`; `LeaveOneGroupOut`
`< 2` groups → `InvalidParameter`; `LeavePGroupsOut` `p == 0` or `n <= p` →
`InvalidParameter`; `GroupShuffleSplit` `test_size ∉ (0,1)` or `< 2` groups →
`InvalidParameter`; `StratifiedGroupKFold` `y`/`groups` length mismatch →
`ShapeMismatch`.

API-shape gaps (all NOT-STARTED, all the same class): sklearn defaults `n_splits=5`
for `GroupKFold`/`StratifiedGroupKFold`/`GroupShuffleSplit` and `test_size=0.2` for
`GroupShuffleSplit`; ferrolearn's constructors require every parameter explicitly, so
the no-arg sklearn forms are unrepresentable (REQ-GKF-3, REQ-SGKF-3, REQ-GSS-3).

Substrate (REQ-X-1, NOT-STARTED): `group_splitters.rs` uses `ndarray::Array1` (the
array type) for `groups`/`y` and `rand`/`SmallRng` (the random substrate) for the
`GroupShuffleSplit` shuffle, which must migrate to `ferray-core` / `ferray::random`
per R-SUBSTRATE-1/2.

## Verification

Commands establishing the SHIPPED claims (baseline
`d12a2ba22d20ca29ab3e662355a51e32c11e77bb`):

- `cargo test -p ferrolearn-model-sel --lib group_splitters` → 6 passed, 0 failed
  (`group_splitters::tests::{group_kfold_partitions_groups,
  group_shuffle_split_deterministic, leave_one_group_out_one_fold_per_group,
  leave_p_groups_out_combinations, stratified_group_kfold_balances,
  stratified_group_kfold_shape_mismatch}`).
- REQ-LOGO-1 / REQ-LPGO-1 DETERMINISTIC oracle (live sklearn 1.5.2):
  ```
  python3 -c "import numpy as np
  from sklearn.model_selection import LeaveOneGroupOut, LeavePGroupsOut
  X=np.zeros((5,1)); g=np.array([0,0,1,1,2])
  print([(list(tr),list(te)) for tr,te in LeaveOneGroupOut().split(X,groups=g)])
  X4=np.zeros((4,1)); g4=np.array([0,1,2,3])
  print([(list(tr),list(te)) for tr,te in LeavePGroupsOut(2).split(X4,groups=g4)])"
  # LOGO  : ([2,3,4],[0,1]) ([0,1,4],[2,3]) ([0,1,2,3],[4])
  # LPGO2 : ([2,3],[0,1]) ([1,3],[0,2]) ([1,2],[0,3]) ([0,3],[1,2]) ([0,2],[1,3]) ([0,1],[2,3])
  ```
  Each matches the ferrolearn `split` output. These are the oracle-pinnable
  `#[test]`s the critic should add (R-CHAR-3, oracle-derived).
- REQ-GKF-2 DIVERGENCE oracle (live sklearn — the greedy tie-break pin):
  ```
  python3 -c "import numpy as np
  from sklearn.model_selection import GroupKFold
  g=np.array([0,0,1,1,2,2,3,3]); X=np.zeros((8,1))
  for i,(tr,te) in enumerate(GroupKFold(3).split(X,groups=g)):
      print(i, sorted(set(g[te])))   # 0 {0,3} | 1 {2} | 2 {1}"
  # ferrolearn HashMap-ordered greedy does NOT reproduce {0,3}/{2}/{1} deterministically
  ```
- REQ-SGKF-1 DIVERGENCE oracle (live sklearn — the std-objective pin):
  ```
  python3 -c "import numpy as np
  from sklearn.model_selection import StratifiedGroupKFold
  X=np.ones((17,2))
  y=np.array([0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0])
  g=np.array([1,1,2,2,3,3,3,4,5,5,5,5,6,6,7,8,8])
  for i,(tr,te) in enumerate(StratifiedGroupKFold(3).split(X,y,g)):
      print(i, sorted(set(g[te])))   # 0 {3,6,7} | 1 {1,2,8} | 2 {4,5}"
  # ferrolearn size-sort + squared-deviation heuristic does NOT reproduce this
  ```
- REQ-GSS-2 DIVERGENCE oracle (live sklearn — the `ceil` vs `round` pin):
  ```
  python3 -c "from math import ceil
  import numpy as np
  from sklearn.model_selection import GroupShuffleSplit
  g=np.array([0,0,1,1,2,2,3,3,4,4,5,5,6,6])   # 7 groups
  print('sklearn n_test', ceil(0.3*7))         # 3
  for tr,te in GroupShuffleSplit(1,test_size=0.3,random_state=0).split(np.zeros((14,1)),groups=g):
      print('sklearn test groups', len(set(g[te])))   # 3
  # ferrolearn round(0.3*7)=2 -> DIVERGENCE, REQ-GSS-2 NOT-STARTED"
  ```
- REQ-LOGO-2 / REQ-LPGO-2 error oracle (live sklearn):
  `LeaveOneGroupOut().split(X, groups=[0,0,0])` raises `ValueError` (`< 2` groups);
  `LeavePGroupsOut(4).split(X, groups=[0,1,2,3])` raises `ValueError` (needs ≥5
  groups). ferrolearn `LeaveOneGroupOut::new().split(&array![0,0,0])` /
  `LeavePGroupsOut::new(4).split(&array![0,1,2,3])` return `Err` — pinnable.
- REQ-X-1 substrate: `grep -n "ndarray\|rand\|SmallRng" group_splitters.rs` shows
  PRODUCTION `use ndarray::Array1` and `use rand::{SeedableRng, rngs::SmallRng,
  seq::SliceRandom}` — wrong substrate, migration owed.
- REQ-X-2 consumer: `grep -n "group_splitters" lib.rs` shows
  `pub use group_splitters::{GroupKFold, GroupShuffleSplit, LeaveOneGroupOut,
  LeavePGroupsOut, StratifiedGroupKFold}` (the boundary public API); no
  `impl CrossValidator` exists for any of the five, so there is no `&dyn
  CrossValidator` call site.

SHIPPED: REQ-GKF-1 (structural), REQ-SGKF-2 (structural), REQ-LOGO-1, REQ-LOGO-2,
REQ-LPGO-1, REQ-LPGO-2, REQ-GSS-1 (structural), REQ-X-2 (consumer, re-export only).
NOT-STARTED: REQ-GKF-2 (greedy ordering/tie-break divergence — KEY REQ),
REQ-GKF-3 (default n_splits), REQ-SGKF-1 (std-objective divergence),
REQ-SGKF-3 (per-class validation + warning + default n_splits),
REQ-GSS-2 (round→ceil group-count sizing divergence),
REQ-GSS-3 (default test_size/n_splits + RNG membership carve-out),
REQ-X-1 (ferray substrate migration). Per R-DEFER-2 every REQ is binary. Per
R-DEFER-3 the GroupShuffleSplit exact-group-SELECTION carve-out (REQ-GSS-3) gets a
blocker but NO failing test; the deterministic divergences (REQ-GKF-2, REQ-SGKF-1,
REQ-GSS-2) ARE oracle-pinnable and the critic should pin them as failing tests
under tracking #1749.

Least-confident SHIPPED claim: REQ-GKF-1 — `GroupKFold` is SHIPPED only on the
STRUCTURAL contract (argmin-load mechanics, no group straddle, n_splits validation).
Its exact fold membership already diverges (REQ-GKF-2 — the HashMap-ordered greedy is
non-deterministic for equal-size groups and does not match sklearn's
descending-index tie-break), so the "shipped" surface is the mechanics, not the
output; the dispatcher flagged this as the KEY REQ and it is correctly NOT-STARTED.
