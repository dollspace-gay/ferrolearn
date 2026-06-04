# Cross-validation splitters (LeaveOneOut / LeavePOut / ShuffleSplit / StratifiedShuffleSplit / RepeatedKFold / RepeatedStratifiedKFold / PredefinedSplit)

<!--
tier: 3-component
status: draft
baseline-commit: 1eae2f6f65137888feec13905ffe12eb97e54d0d
upstream-paths:
  - sklearn/model_selection/_split.py   # LeaveOneOut (:176), LeavePOut (:255), BaseShuffleSplit (:1772), ShuffleSplit (:1889), StratifiedShuffleSplit (:2140), RepeatedKFold (:1633), RepeatedStratifiedKFold (:1699), PredefinedSplit (:2424), _validate_shuffle_split (:2343)
-->

## Summary

`ferrolearn-model-sel/src/splitters.rs` mirrors seven scikit-learn
cross-validation splitters from `sklearn/model_selection/_split.py`:
`LeaveOneOut` (`:176`), `LeavePOut` (`:255`), `ShuffleSplit` (`:1889`),
`StratifiedShuffleSplit` (`:2140`), `RepeatedKFold` (`:1633`),
`RepeatedStratifiedKFold` (`:1699`), and `PredefinedSplit` (`:2424`). They split
into three classes by determinism:

- **DETERMINISTIC / oracle-pinnable** — `LeaveOneOut`, `LeavePOut`,
  `PredefinedSplit`. Each fold's `(train, test)` index list is a pure function of
  `(n_samples, params)`; every REQ is pinnable against
  `list(Splitter(...).split(np.arange(n)))`.
- **RNG carve-out (membership) + DETERMINISTIC sizing** — `ShuffleSplit`,
  `StratifiedShuffleSplit`. The exact split MEMBERSHIP depends on the random
  permutation stream (numpy `Mersenne` vs Rust `SmallRng`), so per R-DEFER-3 it
  is a carve-out (a blocker, NO failing test). The per-split test-fold SIZE is
  deterministic and pinnable — and it CURRENTLY DIVERGES (ferrolearn `round`,
  sklearn `ceil`; see REQ-SS-2).
- **RNG carve-out (membership) + structural** — `RepeatedKFold`,
  `RepeatedStratifiedKFold`. Fold COUNT (`n_splits * n_repeats`) and the
  per-repeat partition structure are deterministic; the shuffled membership is a
  carve-out.

ferrolearn exposes builder structs returning `FoldSplits = Vec<(Vec<usize>,
Vec<usize>)>` (`crate::cross_validation::{FoldSplit, FoldSplits}`). The
index-only splitters (`LeaveOneOut`, `LeavePOut`, `ShuffleSplit`, `RepeatedKFold`,
`PredefinedSplit`) implement `CrossValidator::fold_indices(n_samples)` and are
therefore consumed polymorphically by the production CV-scoring functions
(`cross_val_score`/`cross_validate`/`cross_val_predict`). The label-dependent
splitters (`StratifiedShuffleSplit`, `RepeatedStratifiedKFold`) expose an
inherent `split(&Array1<usize>)` instead and do NOT implement `CrossValidator`;
their only production consumer is the crate re-export boundary API
(`pub use splitters::… in lib.rs`).

## Upstream reference (scikit-learn 1.5.2, commit 156ef14)

### LeaveOneOut (`sklearn/model_selection/_split.py:176`)
- `:222-228` — `_iter_test_indices`: `if n_samples <= 1: raise ValueError(...)`;
  `return range(n_samples)`. Test fold `i` = `{i}`; train = the rest.
- `:230-252` — `get_n_splits(X)` returns `_num_samples(X)` (== `n_samples`).
- Docstring example (`:208-213`): `n=2` → fold0 `train=[1] test=[0]`, fold1
  `train=[0] test=[1]`.

### LeavePOut (`sklearn/model_selection/_split.py:255`)
- `:313-314` — `__init__(self, p)`; `p` is the test-set size.
- `:316-325` — `_iter_test_indices`: `if n_samples <= self.p: raise ValueError("p
  ={} must be strictly less than the number of samples={}")`; then `for
  combination in combinations(range(n_samples), self.p): yield
  np.array(combination)`. ORDER = `itertools.combinations` lexicographic.
- Docstring example (`:293-310`): `LeavePOut(2)` on `n=4` → 6 folds, test sets
  `[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]` in that order.

### ShuffleSplit (`sklearn/model_selection/_split.py:1889`) + BaseShuffleSplit
- `:1803-1810` — `BaseShuffleSplit.__init__(self, n_splits=10, *, test_size=None,
  train_size=None, random_state=None)`; `self._default_test_size = 0.1`.
- `:1846-1862` — `_iter_indices`: `n_train, n_test = _validate_shuffle_split(...,
  default_test_size=self._default_test_size)`; then per split `permutation =
  rng.permutation(n_samples); ind_test = permutation[:n_test]; ind_train =
  permutation[n_test:(n_test+n_train)]`.
- `:1864-1883` — `get_n_splits` returns `self.n_splits`.
- `_validate_shuffle_split` (`:2343`): when `test_size` is a float (`:2389`)
  `n_test = ceil(test_size * n_samples)`; `n_train = n_samples - n_test`
  (`:2399-2400`); raises `ValueError` if `test_size <= 0 or test_size >= 1`
  (`:2357-2358`) or if `n_train == 0` (`:2414`).

### StratifiedShuffleSplit (`sklearn/model_selection/_split.py:2140`)
- Same `n_test = ceil(test_size * n)` global sizing via `_validate_shuffle_split`,
  default `test_size=0.1`. Per split it allocates per-class test counts so the
  test set preserves class proportions, then draws a random permutation within
  each class. Class membership of the per-split test set is RNG-dependent.

### RepeatedKFold / RepeatedStratifiedKFold (`:1633` / `:1699`)
- `:1693-1696` — `RepeatedKFold.__init__(self, *, n_splits=5, n_repeats=10,
  random_state=None)` → `_RepeatedSplits(KFold, ...)`. Defaults `n_splits=5`,
  `n_repeats=10`.
- `:1699` `RepeatedStratifiedKFold` — same shape over `StratifiedKFold`, defaults
  `n_splits=5, n_repeats=10`.
- `_RepeatedSplits.split` runs the base splitter `n_repeats` times, each with a
  fresh `random_state` drawn from the seeded RNG → `n_splits * n_repeats` folds.
  Each repeat is a valid (shuffled) KFold / StratifiedKFold partition.

### PredefinedSplit (`sklearn/model_selection/_split.py:2424`)
- `:2466-2470` — `__init__`: `self.test_fold = np.array(test_fold, dtype=int)`;
  `self.unique_folds = np.unique(self.test_fold)`; then `self.unique_folds =
  self.unique_folds[self.unique_folds != -1]` — `-1` is EXCLUDED. `np.unique`
  returns the unique fold ids SORTED ASCENDING.
- `_split` (helper below the class): for each unique fold `f` (ascending), `test =
  {i : test_fold[i] == f}`, `train = rest`. Samples with `test_fold[i] == -1` are
  never in any test set.
- Docstring example (`:2448-2463`): `test_fold=[0,1,-1,1]` → fold0
  `train=[1,2,3] test=[0]`, fold1 `train=[0,2] test=[1,3]`.

## Requirements

### LeaveOneOut (DETERMINISTIC)
- REQ-LOO-1: split-index parity. `n_samples` folds; fold `i` has `test=[i]`,
  `train` = all other indices ascending. Mirrors `:222-228`. Oracle-pinnable
  (`LeaveOneOut().split(np.arange(n))`).
- REQ-LOO-2: `get_n_splits == n_samples` and `n_samples < 2` rejection. Mirrors
  `:230-252` and the `:224` `n_samples <= 1` `ValueError`.

### LeavePOut (DETERMINISTIC)
- REQ-LPO-1: combination-order index parity. `C(n, p)` folds; test set = each
  `p`-combination of `0..n` in `itertools.combinations` LEXICOGRAPHIC order;
  train = rest. Mirrors `:316-325`. Oracle-pinnable (`LeavePOut(2).split(
  np.arange(4))` → 6 folds in combination order).
- REQ-LPO-2: error semantics. `p == 0` rejected; `n_samples <= p` rejected.
  Mirrors `:318-323` (`p must be strictly less than the number of samples`).

### PredefinedSplit (DETERMINISTIC)
- REQ-PS-1: split-index parity with `-1` exclusion. Unique folds (ascending,
  EXCLUDING `-1`); per fold `test = {i : test_fold[i]==fold}`, `train` = rest;
  every `-1` sample stays in train for all folds. Mirrors `:2466-2470` + `_split`.
  Oracle-pinnable (`PredefinedSplit([0,1,-1,1,0]).split()`).
- REQ-PS-2: `n_splits == #unique non-`-1` folds`; `test_fold` length must equal
  `n_samples`.

### ShuffleSplit (RNG carve-out + sizing)
- REQ-SS-1: structural contract. `n_splits` folds; each fold's `train ∪ test` is a
  size-`n_samples` partition (disjoint, covering); `get_n_splits == n_splits`;
  `random_state` makes the stream reproducible. Mirrors `:1846-1862`. STRUCTURAL.
- REQ-SS-2: test-fold SIZING parity (DETERMINISTIC, oracle-pinnable). sklearn
  `n_test = ceil(test_size * n_samples)` (`:2390`). ferrolearn uses `.round()`
  (`((n_samples as f64) * self.test_size).round()`). DIVERGES: `n=7,
  test_size=0.3` → sklearn `ceil(2.1)=3`, ferrolearn `round(2.1)=2`. NOT-STARTED.
- REQ-SS-3: default `test_size` and exact-membership carve-out. sklearn defaults
  `test_size=None → 0.1` (`:1803-1810`); ferrolearn `new(n_splits, test_size:
  f64)` REQUIRES an explicit `f64` (no default) — API-shape gap. Exact split
  membership is an RNG carve-out (numpy permutation vs Rust `SmallRng`),
  R-DEFER-3: blocker, NO failing test.

### StratifiedShuffleSplit (RNG carve-out + structural)
- REQ-SSS-1: structural contract. `n_splits` folds; each preserves (approximately)
  the per-class proportion in the test set; train/test disjoint; reproducible
  under `random_state`. Mirrors `:2140`. STRUCTURAL.
- REQ-SSS-2: sizing + per-class allocation parity. sklearn allocates per-class
  test counts from a global `n_test = ceil(test_size*n)` (`:2390`). ferrolearn
  computes a PER-CLASS `round(class_size * test_size)` independently. NOT-STARTED
  (both the `round`-vs-`ceil` divergence AND the per-class-vs-global allocation
  divergence). Exact membership = RNG carve-out (blocker, no failing test).
- REQ-SSS-3: default `test_size=0.1` + `CrossValidator` integration. ferrolearn
  requires explicit `f64` AND exposes only `split(&Array1<usize>)` (no
  `CrossValidator` impl). NOT-STARTED.

### RepeatedKFold / RepeatedStratifiedKFold (RNG carve-out + structural)
- REQ-RKF-1: fold count + per-repeat partition structure. `n_splits * n_repeats`
  folds; each repeat is a valid KFold (`RepeatedKFold`) / StratifiedKFold
  (`RepeatedStratifiedKFold`) partition with a distinct shuffle. Mirrors
  `:1633`/`:1699` (`_RepeatedSplits`). STRUCTURAL (count + partition shape
  pinnable; exact membership carve-out).
- REQ-RKF-2: defaults. sklearn `n_splits=5, n_repeats=10` (`:1693`, `:1699`);
  ferrolearn `new(n_splits, n_repeats)` requires both — API-shape gap.
  NOT-STARTED.
- REQ-RKF-3: exact-membership carve-out (R-DEFER-3). The shuffle RNG (numpy vs
  Rust) makes exact per-fold membership a carve-out: blocker, NO failing test.

### Cross-cutting
- REQ-X-1: R-SUBSTRATE. Production code uses `ndarray::Array1` (the array type),
  `rand::{SeedableRng, rngs::SmallRng, seq::SliceRandom}` (the random substrate),
  and `std::collections::HashMap`. The destination substrate is `ferray-core` /
  `ferray::random` (R-SUBSTRATE-1). NOT-STARTED (migration owed).
- REQ-X-2: non-test production consumer. The five `CrossValidator`-implementing
  splitters are consumed polymorphically by `cross_val_score`/`cross_validate`/
  `cross_val_predict`; all seven are re-exported at `pub use splitters::… in
  lib.rs` (the boundary public API per S5). SHIPPED for the five; the two
  label-dependent splitters are SHIPPED on the re-export boundary only.

## Acceptance criteria

- AC-LOO-1 (REQ-LOO-1): `LeaveOneOut::new().fold_indices(3)` → fold0
  `(train=[1,2], test=[0])`, fold1 `([0,2],[1])`, fold2 `([0,1],[2])` — equal to
  the live oracle. DETERMINISTIC.
- AC-LOO-2 (REQ-LOO-2): `get_n_splits(n) == n`; `fold_indices(1)` returns `Err`
  (sklearn `LeaveOneOut().split(np.arange(1))` raises `ValueError`).
- AC-LPO-1 (REQ-LPO-1): `LeavePOut::new(2).fold_indices(4)` → test sets in order
  `[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]` with the complementary train sets — equal
  to the live oracle (combination order). 6 folds = `C(4,2)`. DETERMINISTIC.
- AC-LPO-2 (REQ-LPO-2): `LeavePOut::new(0).fold_indices(5)` → `Err`;
  `LeavePOut::new(4).fold_indices(4)` → `Err` (sklearn: `p` must be `< n`).
- AC-PS-1 (REQ-PS-1): `PredefinedSplit::new(array![0,1,-1,1,0]).fold_indices(5)` →
  fold0 `(train=[1,2,3], test=[0,4])`, fold1 `(train=[0,2,4], test=[1,3])` — equal
  to the live oracle; index 2 (`-1`) is in train for BOTH folds. DETERMINISTIC.
- AC-PS-2 (REQ-PS-2): 2 unique non-`-1` folds → 2 splits; a `test_fold` whose
  length `!= n_samples` returns `Err`.
- AC-SS-1 (REQ-SS-1): `ShuffleSplit::new(3, 0.25).random_state(42).
  fold_indices(8)` → 3 folds, each `train.len() + test.len() == 8`, train/test
  disjoint, reproducible across calls with the same seed. STRUCTURAL.
- AC-SS-2 (REQ-SS-2): `ShuffleSplit::new(1, 0.3).fold_indices(7)` test-fold size
  is CURRENTLY 2 but sklearn `ShuffleSplit(test_size=0.3).split(np.arange(7))` has
  test size 3 (`ceil(2.1)`). DIVERGENCE. NOT-STARTED.
- AC-SSS-1 (REQ-SSS-1): `StratifiedShuffleSplit::new(2, 0.25).random_state(7).
  split(&y)` (y has 3 classes ×... ) → 2 folds, test set contains ≥1 of each
  class, train/test disjoint. STRUCTURAL.
- AC-RKF-1 (REQ-RKF-1): `RepeatedKFold::new(3, 2).random_state(5).fold_indices(9)`
  → 6 folds; within each consecutive group of 3 the test sets partition `0..9`.
  STRUCTURAL.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-LOO-1 (split-index parity, DETERMINISTIC) | SHIPPED | impl `fn fold_indices in splitters.rs` (`impl CrossValidator for LeaveOneOut`) builds, for `i in 0..n_samples`, `test = vec![i]` and `train = (0..n_samples).filter(|&j| j != i).collect()` — exactly sklearn's `_iter_test_indices` returning `range(n_samples)` with each singleton as the test fold (`sklearn/model_selection/_split.py:222-228`). Train is ascending by construction (`filter` over `0..n`). Live oracle `LeaveOneOut().split(np.arange(3))` → `([1,2],[0])`, `([0,2],[1])`, `([0,1],[2])` matches. Test: `test_loo_basic` (5 folds, each `test.len()==1`, `train.len()==4`). Oracle-pinnable: a `#[test]` asserting the exact `n=3` lists above (R-CHAR-3). Non-test consumer: REQ-X-2 (`&dyn CrossValidator`). |
| REQ-LOO-2 (get_n_splits + n<2 reject) | SHIPPED | impl `fn get_n_splits in splitters.rs` returns `n_samples` (mirrors `get_n_splits(X) → _num_samples(X)`, `:230-252`); `fn fold_indices` returns `Err(FerroError::InsufficientSamples { required: 2, actual: n_samples, .. })` when `n_samples < 2`, mirroring sklearn's `:224` `if n_samples <= 1: raise ValueError`. R-DEV-2 type-name mapping: sklearn `ValueError` ↦ `FerroError::InsufficientSamples`. Live oracle `LeaveOneOut().split(np.arange(1))` raises `ValueError`. Test: `test_loo_basic` exercises the n≥2 path; the reject path is pinnable. Non-test consumer: REQ-X-2. |
| REQ-LPO-1 (combination-order parity, DETERMINISTIC) | SHIPPED | impl `fn fold_indices in splitters.rs` (`impl CrossValidator for LeavePOut`) initializes `combo = (0..p).collect()` and advances it via the standard lexicographic next-combination loop (`while i > 0 { … if combo[i] < n - p + i { combo[i] += 1; reset suffix } }`), pushing `test = combo.clone()` and `train = (0..n).filter(!test_set.contains)` each iteration. This reproduces Python `itertools.combinations(range(n), p)` order exactly (`sklearn/model_selection/_split.py:324`). Live oracle `LeavePOut(2).split(np.arange(4))` → test sets `[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]` (6 = `C(4,2)`), each matching ferrolearn's `combo` sequence. Test: `test_lpo_p2` (6 folds for `C(4,2)`). Oracle-pinnable: a `#[test]` asserting the exact 6-fold ordered list (R-CHAR-3). Non-test consumer: REQ-X-2. |
| REQ-LPO-2 (error semantics) | SHIPPED | impl `fn fold_indices in splitters.rs` returns `Err(FerroError::InvalidParameter { name: "p", .. })` when `self.p == 0`, and `Err(FerroError::InsufficientSamples { required: p+1, actual: n_samples, .. })` when `n_samples <= self.p`, mirroring sklearn `:318-323` (`if n_samples <= self.p: raise ValueError("p={} must be strictly less than the number of samples={}")`). R-DEV-2 type mapping ValueError ↦ FerroError. Live oracle: `LeavePOut(4).split(np.arange(4))` raises `ValueError`. Pinnable: `fold_indices(4)` on `LeavePOut::new(4)` → `Err`; `LeavePOut::new(0)` → `Err`. Non-test consumer: REQ-X-2. |
| REQ-PS-1 (split-index parity + `-1` exclusion, DETERMINISTIC) | SHIPPED | impl `fn fold_indices in splitters.rs` (`impl CrossValidator for PredefinedSplit`) collects `folds: HashMap<isize, Vec<usize>>` over `i where test_fold[i] >= 0` (so `-1` is never added to any test set), sorts the keys ascending (`keys.sort_unstable()`), and for each key emits `test = folds[k]`, `train = (0..n).filter(!test_set.contains)`. This mirrors sklearn `:2466-2470` (`unique_folds = unique(test_fold); unique_folds[unique_folds != -1]`, ascending) and `_split`. Live oracle `PredefinedSplit([0,1,-1,1,0]).split()` → fold0 `([1,2,3],[0,4])`, fold1 `([0,2,4],[1,3])`; index 2 (`-1`) in train for both. Test: `test_predefined_split` (2 folds; index 2 always in train). Oracle-pinnable: `#[test]` asserting the exact 2 ordered folds above (R-CHAR-3). Non-test consumer: REQ-X-2. |
| REQ-PS-2 (n_splits + length validation) | SHIPPED | impl `fn fold_indices in splitters.rs` emits exactly one fold per unique non-`-1` key (`out: Vec` of length `keys.len()`), so the split count equals `#unique non-`-1` folds` (sklearn `len(self.unique_folds)`, `:2470`). It returns `Err(FerroError::ShapeMismatch { expected: [n_samples], actual: [test_fold.len()], .. })` when `test_fold.len() != n_samples` — a ferrolearn-added guard (sklearn indexes by position and would misbehave on a length mismatch; matching contract is the test-fold length == n_samples invariant). Live oracle: `PredefinedSplit([0,1,-1,1,0]).get_n_splits() == 2`. Test: `test_predefined_split` (len == 2). Non-test consumer: REQ-X-2. |
| REQ-SS-1 (ShuffleSplit structural, STRUCTURAL) | SHIPPED | impl `fn fold_indices in splitters.rs` (`impl CrossValidator for ShuffleSplit`) produces `self.n_splits` folds; per split it shuffles `indices = (0..n).collect()` then slices `test = indices[..n_test]`, `train = indices[n_test..]` — a disjoint covering partition of `0..n`, mirroring sklearn's `permutation = rng.permutation(n); ind_test = permutation[:n_test]; ind_train = permutation[n_test:(n_test+n_train)]` (`sklearn/model_selection/_split.py:1857-1862`). With `random_state(seed)` the per-split `SmallRng::seed_from_u64(seed + split)` makes the stream reproducible (R-DEV-1 determinism, mechanism differs from numpy). Structural facts (fold count, partition, reproducibility) hold; exact membership is the REQ-SS-3 carve-out. Test: `test_shuffle_split_basic` (3 folds, `train+test==8`, test non-empty). Non-test consumer: REQ-X-2. |
| REQ-SS-2 (test-fold sizing parity) | NOT-STARTED | DIVERGENCE. impl `fn fold_indices in splitters.rs` computes `n_test = ((n_samples as f64) * self.test_size).round().max(1.0) as usize` — uses `.round()`. sklearn's `_validate_shuffle_split` computes `n_test = ceil(test_size * n_samples)` (`sklearn/model_selection/_split.py:2390`). Live oracle: `ShuffleSplit(test_size=0.3).split(np.arange(7))` → test size 3 (`ceil(2.1)`); ferrolearn → `round(2.1)=2`. This is the train_test_split `ceil`-vs-`round` bug class. Blocker (tracking #1744; critic files the specific failing-test spec): needs `round` → `ceil` in the `n_test` computation. Oracle-pinnable once fixed. |
| REQ-SS-3 (default test_size + exact-membership carve-out) | NOT-STARTED | Two gaps. (a) API-shape: sklearn `ShuffleSplit(n_splits=10, *, test_size=None → 0.1, …)` (`:1803-1810`) has a default `test_size`; ferrolearn `pub fn new(n_splits, test_size: f64) in splitters.rs` REQUIRES an explicit `f64` — `ShuffleSplit()` with sklearn defaults is unrepresentable. (b) Exact per-split MEMBERSHIP is an RNG carve-out (numpy `permutation` vs Rust `SmallRng`): per R-DEFER-3 it gets a blocker but NO failing test. Blocker (#1744): default-`test_size` constructor gap + RNG-stream membership carve-out documented. |
| REQ-SSS-1 (StratifiedShuffleSplit structural, STRUCTURAL) | SHIPPED | impl `fn split in splitters.rs` (inherent `StratifiedShuffleSplit::split(&Array1<usize>)`) groups indices `by_class: HashMap`, shuffles each class's indices, draws `n_class_test` per class into `test` and the rest into `train`, then sorts both — yielding `n_splits` folds that each contain test samples from every class (proportion preserved per class), train/test disjoint, reproducible under `random_state`. Structurally mirrors sklearn `StratifiedShuffleSplit` (`sklearn/model_selection/_split.py:2140`) class-proportion preservation. Exact membership + exact per-class counts are the REQ-SSS-2 carve-out/divergence. Test: `test_stratified_shuffle_split` (2 folds on a 3-class y). Non-test consumer: REQ-X-2 (re-export boundary only — no `CrossValidator` impl). |
| REQ-SSS-2 (sizing + per-class allocation parity) | NOT-STARTED | DIVERGENCE (two parts). (a) impl `fn split in splitters.rs` computes a PER-CLASS `n_class_test = ((idx.len() as f64) * self.test_size).round()`, whereas sklearn derives a GLOBAL `n_test = ceil(test_size*n)` (`:2390`) and allocates it across classes via `_approximate_mode` — so both the `round`-vs-`ceil` and the per-class-independent-vs-global-allocation behaviors diverge. Blocker (#1744): per-class round → global ceil + `_approximate_mode` allocation. Oracle-pinnable on TEST-SET SIZE per split (the total `n_test`), not membership. |
| REQ-SSS-3 (default test_size + CrossValidator integration) | NOT-STARTED | API-shape gap. sklearn `StratifiedShuffleSplit` defaults `test_size=None → 0.1`; ferrolearn `pub fn new(n_splits, test_size: f64) in splitters.rs` requires explicit `f64`. Additionally it exposes only `split(&Array1<usize>)` and does NOT implement `CrossValidator`, so it cannot be passed to `cross_val_score`/`cross_validate`/`cross_val_predict` as a `&dyn CrossValidator` (unlike the index-only splitters). Blocker (#1744): default-test_size constructor + a label-aware CV integration path. |
| REQ-RKF-1 (Repeated fold count + per-repeat partition, STRUCTURAL) | SHIPPED | impl `fn fold_indices in splitters.rs` (`impl CrossValidator for RepeatedKFold`) loops `repeat in 0..n_repeats`, builds `KFold::new(n_splits).shuffle(true)` with `random_state(seed + repeat)`, and `extend`s its `fold_indices` — yielding exactly `n_splits * n_repeats` folds where each consecutive group of `n_splits` is a valid shuffled KFold partition (mirrors `_RepeatedSplits` running the base splitter `n_repeats` times, `sklearn/model_selection/_split.py:1633`). `RepeatedStratifiedKFold::split(&Array1<usize>)` does the same over `StratifiedKFold` (`:1699`). Test: `test_repeated_kfold` (6 = 3×2 folds), `test_repeated_stratified` (6 folds). Structural (count + partition) holds; exact membership is REQ-RKF-3. Non-test consumer: REQ-X-2 (`RepeatedKFold` via `&dyn CrossValidator`; `RepeatedStratifiedKFold` via re-export only). |
| REQ-RKF-2 (defaults) | NOT-STARTED | API-shape gap. sklearn defaults `n_splits=5, n_repeats=10` (`sklearn/model_selection/_split.py:1693`, `:1699`); ferrolearn `pub fn new(n_splits, n_repeats) in splitters.rs` requires BOTH positionally — `RepeatedKFold()`/`RepeatedStratifiedKFold()` with sklearn defaults is unrepresentable. Blocker (#1744): constructor default-parameter surface. |
| REQ-RKF-3 (exact-membership carve-out, R-DEFER-3) | NOT-STARTED | Exact per-fold membership depends on the underlying `KFold`/`StratifiedKFold` shuffle RNG (numpy `Mersenne` vs ferrolearn `SmallRng`), so per R-DEFER-3 it is an RNG carve-out: blocker (#1744), NO failing test. Only the structural facts (REQ-RKF-1) are pinned. |
| REQ-X-1 (R-SUBSTRATE) | NOT-STARTED | Production code in `splitters.rs` imports `ndarray::Array1` (array type), `rand::{SeedableRng, rngs::SmallRng, seq::SliceRandom}` (random substrate), and `std::collections::HashMap`. Per R-SUBSTRATE-1 the destination is `ferray-core` (array) and `ferray::random` (sampling); these are the wrong substrate. Blocker (#1744): migrate `ndarray::Array1` → `ferray-core` and `rand`/`SmallRng` → `ferray::random`. Until then this unit is not on the ferray substrate (R-SUBSTRATE-2). |
| REQ-X-2 (non-test production consumer) | SHIPPED | The five index-only splitters (`LeaveOneOut`, `LeavePOut`, `ShuffleSplit`, `RepeatedKFold`, `PredefinedSplit`) implement `CrossValidator` (`impl CrossValidator for … in splitters.rs`, each with `fn fold_indices`) and are consumed polymorphically by the production CV-scoring functions via `cv.fold_indices(n_samples)?`: `cross_val_score`/`cross_validate`/`cross_val_predict` (`cross_validation.rs`, the three `cv: &dyn CrossValidator` parameters / `cv.fold_indices` call sites). All seven are re-exported at `pub use splitters::{LeaveOneOut, LeavePOut, PredefinedSplit, RepeatedKFold, RepeatedStratifiedKFold, ShuffleSplit, StratifiedShuffleSplit} in lib.rs` — the boundary public API per S5/R-DEFER-1 grandfathering. Honest underclaim: `StratifiedShuffleSplit`/`RepeatedStratifiedKFold` expose only `split(&Array1<usize>)` (no `CrossValidator` impl), so their sole production consumer is the re-export boundary, not a `&dyn CrossValidator` call site; no `ferrolearn-python` binding exists for any of the seven. |

## Architecture

ferrolearn implements all seven splitters as `#[derive(Debug, Clone)]` builder
structs in `splitters.rs` returning `FoldSplits = Vec<(Vec<usize>, Vec<usize>)>`
(`crate::cross_validation::{FoldSplit, FoldSplits}`). They divide along the
data-dependency axis:

- **Index-only** (`LeaveOneOut`, `LeavePOut`, `ShuffleSplit`, `RepeatedKFold`,
  `PredefinedSplit`) implement `CrossValidator::fold_indices(n_samples)`, so the
  production CV-scoring functions accept them as `&dyn CrossValidator` (REQ-X-2).
- **Label-dependent** (`StratifiedShuffleSplit`, `RepeatedStratifiedKFold`) expose
  an inherent `split(&Array1<usize>)` taking class labels and do NOT implement
  `CrossValidator` (the trait's `fold_indices(usize)` has no label channel). Their
  only production reach is the crate re-export — a structural integration gap
  captured in REQ-SSS-3/REQ-X-2 (honest underclaim).

The three DETERMINISTIC splitters are oracle-pinnable end to end. `LeaveOneOut`
emits the `n` singletons (`:222-228`). `LeavePOut` runs an explicit lexicographic
next-combination loop reproducing `itertools.combinations(range(n), p)` order
(`:324`) — the ORDER, not just the count, is the contract. `PredefinedSplit`
groups by `test_fold` value, EXCLUDES `-1` (`:2470`), and emits folds in
ascending fold-id order; `-1` samples stay in every train set.

The two ShuffleSplit-family splitters carry a REAL deterministic sizing
divergence: ferrolearn computes `n_test` with `.round()` (and
`StratifiedShuffleSplit` does so PER CLASS), whereas sklearn's
`_validate_shuffle_split` uses `ceil(test_size * n_samples)` globally
(`:2390`). `n=7, test_size=0.3` → sklearn 3, ferrolearn 2 (REQ-SS-2, REQ-SSS-2 —
NOT-STARTED, oracle-pinnable on test-set SIZE). The exact MEMBERSHIP of each
random split is, separately, an RNG carve-out: ferrolearn seeds a per-split
`SmallRng::seed_from_u64(seed + split)` and `SliceRandom::shuffle`, whose stream
differs from numpy's `RandomState.permutation`. Per R-DEFER-3 the membership
carve-out gets a blocker but NO failing test; only structural REQs are pinned.

The Repeated splitters delegate to the base `KFold`/`StratifiedKFold` (from
`crate::cross_validation`) `n_repeats` times with `random_state(seed + repeat)`,
giving `n_splits * n_repeats` folds. Fold count and per-repeat partition shape are
structural (REQ-RKF-1, SHIPPED); membership is the base-splitter RNG carve-out
(REQ-RKF-3).

Error handling maps sklearn `ValueError` to `FerroError` variants (R-DEV-2 type
mapping): `LeaveOneOut`/`ShuffleSplit` n<2 → `InsufficientSamples`; `LeavePOut`
`p==0` → `InvalidParameter`, `n<=p` → `InsufficientSamples`; `ShuffleSplit`/
`StratifiedShuffleSplit` `test_size ∉ (0,1)` → `InvalidParameter`;
`PredefinedSplit` length mismatch → `ShapeMismatch`.

API-shape gaps (all NOT-STARTED, all the same class): sklearn gives `ShuffleSplit`
/`StratifiedShuffleSplit` a default `test_size=0.1` and `RepeatedKFold`/
`RepeatedStratifiedKFold` defaults `n_splits=5, n_repeats=10`; ferrolearn's
constructors require every parameter explicitly, so the no-arg sklearn forms are
unrepresentable (REQ-SS-3, REQ-SSS-3, REQ-RKF-2).

Substrate (REQ-X-1, NOT-STARTED): unlike `time_series_split.rs` (pure scalar
index math), `splitters.rs` DOES use array/random substrate — `ndarray::Array1`
for `test_fold`/`y` and `rand`/`SmallRng` for the shuffles — which must migrate to
`ferray-core` / `ferray::random` per R-SUBSTRATE-1/2.

## Verification

Commands establishing the SHIPPED claims (baseline
`1eae2f6f65137888feec13905ffe12eb97e54d0d`):

- `cargo test -p ferrolearn-model-sel --lib splitters` → 7 passed, 0 failed
  (`splitters::tests::{test_loo_basic, test_lpo_p2, test_shuffle_split_basic,
  test_stratified_shuffle_split, test_repeated_kfold, test_repeated_stratified,
  test_predefined_split}`).
- REQ-LOO-1/LPO-1/PS-1 DETERMINISTIC oracle (live sklearn 1.5.2):
  ```
  python3 -c "import numpy as np
  from sklearn.model_selection import LeaveOneOut, LeavePOut, PredefinedSplit
  print([(list(tr),list(te)) for tr,te in LeaveOneOut().split(np.arange(3))])
  print([(list(tr),list(te)) for tr,te in LeavePOut(2).split(np.arange(4))])
  print([(list(tr),list(te)) for tr,te in PredefinedSplit([0,1,-1,1,0]).split()])"
  # LOO n=3 : ([1,2],[0]) ([0,2],[1]) ([0,1],[2])
  # LPO(2)n4: ([2,3],[0,1]) ([1,3],[0,2]) ([1,2],[0,3]) ([0,3],[1,2]) ([0,2],[1,3]) ([0,1],[2,3])
  # PS      : ([1,2,3],[0,4]) ([0,2,4],[1,3])
  ```
  Each matches the ferrolearn `fold_indices` output. These are the
  oracle-pinnable `#[test]`s the critic should add (R-CHAR-3, oracle-derived).
- REQ-SS-2 DIVERGENCE oracle (live sklearn — the `ceil` vs `round` pin):
  ```
  python3 -c "from math import ceil
  from sklearn.model_selection import ShuffleSplit; import numpy as np
  print('sklearn n_test', ceil(0.3*7))                 # 3
  for tr,te in ShuffleSplit(n_splits=2,test_size=0.3,random_state=0).split(np.arange(7)):
      print('sklearn test size', len(te))               # 3, 3
  # ferrolearn round(0.3*7)=2  -> DIVERGENCE, REQ-SS-2 NOT-STARTED"
  ```
- REQ-LOO-2/LPO-2 error oracle (live sklearn): `LeaveOneOut().split(np.arange(1))`
  raises `ValueError`; `LeavePOut(4).split(np.arange(4))` raises `ValueError`
  (`p` must be `< n`). ferrolearn `fold_indices(1)` / `LeavePOut::new(4).
  fold_indices(4)` return `Err` — pinnable.
- REQ-RKF-1 structural: `RepeatedKFold::new(3,2).random_state(5).fold_indices(9)`
  → 6 folds; assert each consecutive group of 3 test sets partitions `0..9`
  (`test_repeated_kfold` pins the count; the partition assertion is pinnable).
- REQ-X-1 substrate: `grep -n "ndarray\|rand\|SmallRng" splitters.rs` shows
  PRODUCTION `use ndarray::Array1` and `use rand::{SeedableRng, rngs::SmallRng,
  seq::SliceRandom}` — wrong substrate, migration owed.
- REQ-X-2 consumer: `grep -n "fold_indices\|dyn CrossValidator" cross_validation.rs`
  shows the `cv: &dyn CrossValidator` parameters and `cv.fold_indices(n_samples)?`
  call sites in `cross_val_score`/`cross_validate`/`cross_val_predict` that accept
  the five `CrossValidator`-implementing splitters.

SHIPPED: REQ-LOO-1, REQ-LOO-2, REQ-LPO-1, REQ-LPO-2, REQ-PS-1, REQ-PS-2, REQ-SS-1
(structural), REQ-SSS-1 (structural), REQ-RKF-1 (structural), REQ-X-2 (consumer).
NOT-STARTED: REQ-SS-2 (round→ceil sizing divergence), REQ-SS-3 (default test_size
+ RNG membership carve-out), REQ-SSS-2 (per-class round + allocation divergence),
REQ-SSS-3 (default test_size + no CrossValidator impl), REQ-RKF-2 (constructor
defaults), REQ-RKF-3 (RNG membership carve-out), REQ-X-1 (ferray substrate
migration). Per R-DEFER-2 every REQ is binary. Per R-DEFER-3 the RNG-membership
carve-outs (REQ-SS-3, REQ-RKF-3, and the membership half of REQ-SSS-2) get
blockers but NO failing test; the test-set-SIZE divergences (REQ-SS-2,
REQ-SSS-2) ARE oracle-pinnable and the critic should pin them as failing tests
under tracking #1744.

Least-confident SHIPPED claim: REQ-SSS-1 — `StratifiedShuffleSplit` is SHIPPED
only on the STRUCTURAL contract (class-proportion preservation, disjoint
train/test, reproducibility); its sizing already diverges (REQ-SSS-2) and it has
no `CrossValidator` consumer (REQ-SSS-3), so the "shipped" surface is narrow and
its production reach is the re-export boundary alone.
