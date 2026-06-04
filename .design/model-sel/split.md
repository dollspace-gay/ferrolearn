# train_test_split

<!--
tier: 3-component
status: draft
baseline-commit: 4ca941196bef12467cffceb865d996a0d6c263c9
upstream-paths:
  - sklearn/model_selection/_split.py   # train_test_split (:2686), _validate_shuffle_split (:2343)
-->

## Summary

`ferrolearn-model-sel/src/split.rs` mirrors scikit-learn's
`train_test_split` (`sklearn/model_selection/_split.py:2686`) — the one-liner
that shuffles a dataset and partitions it into train/test subsets. sklearn wraps
input validation, sizing (`_validate_shuffle_split`, `:2343`), and
`next(ShuffleSplit().split(...))` (`:2697-2699`, `:2804-2806`).

ferrolearn ships a narrow slice: a single `(x, y)` pair, a required float
`test_size`, an always-on shuffle seeded by `random_state`, and `(0,1)` float
validation. It DIVERGES from sklearn on the test-set sizing rule
(`round` vs `ceil`) and on empty-split handling (it clamps instead of raising).
The rest of sklearn's surface — `test_size=None` default of 0.25, `train_size`,
`shuffle=False`, `stratify`, integer `test_size`, and the variadic `*arrays`
signature — is absent. The R-SUBSTRATE migration to `ferray-core`/`ferray::random`
is also NOT-STARTED.

## Upstream reference (scikit-learn 1.5.2, commit 156ef14)

### train_test_split
- `sklearn/model_selection/_split.py:2686-2693` — signature
  `train_test_split(*arrays, test_size=None, train_size=None, random_state=None,
  shuffle=True, stratify=None)` — variadic arrays, keyword-only params.
- `:2778-2780` — `if n_arrays == 0: raise ValueError("At least one array required as input")`.
- `:2784-2787` — `n_train, n_test = _validate_shuffle_split(n_samples, test_size,
  train_size, default_test_size=0.25)` — the 0.25 default lives at the call site.
- `:2789-2796` — `shuffle is False` branch: `train = np.arange(n_train)`,
  `test = np.arange(n_train, n_train + n_test)` — deterministic sequential split;
  `stratify is not None` with `shuffle=False` raises `ValueError`.
- `:2798-2806` — shuffle branch: `CVClass = StratifiedShuffleSplit` if `stratify`
  else `ShuffleSplit`; `cv = CVClass(test_size=n_test, train_size=n_train,
  random_state=random_state)`; `train, test = next(cv.split(X=arrays[0], y=stratify))`.
- `:2810-2814` — returns `2 * len(arrays)` slices via `_safe_indexing(a, train/test)`.

### _validate_shuffle_split (the sizing/validation helper)
- `:2348-2349` — `if test_size is None and train_size is None: test_size = default_test_size`
  (= 0.25 from the call site).
- `:2354-2364` — float `test_size` validation: raises `ValueError` if
  `test_size <= 0 or test_size >= 1`; integer validation if `test_size >= n_samples
  or test_size <= 0`.
- `:2389-2390` — **`if test_size_type == "f": n_test = ceil(test_size * n_samples)`** —
  the CEIL sizing rule.
- `:2391-2392` — integer path: `n_test = float(test_size)` (absolute count).
- `:2399-2400` — `if train_size is None: n_train = n_samples - n_test`.
- `:2414-2419` — `if n_train == 0: raise ValueError("… the resulting train set
  will be empty …")` — empty-split RAISE.

## Requirements

- REQ-1: Float `test_size` validation `(0,1)`. Reject `test_size <= 0` or `>= 1`.
  Mirrors `_validate_shuffle_split` (`sklearn/model_selection/_split.py:2357-2359`).
  DETERMINISTIC / oracle-pinnable.
- REQ-2: Test-set sizing rule. sklearn computes
  `n_test = ceil(test_size * n_samples)` (`:2390`); ferrolearn computes
  `round(test_size * n_samples)` then clamps to `[1, n_samples-1]`. ROUND vs CEIL
  diverges for non-integer products. DETERMINISTIC / oracle-pinnable (the SPLIT
  SIZES diverge regardless of the shuffle).
- REQ-3: Exact split membership (which sample lands in train vs test). Depends on
  the shuffle PRNG (`SmallRng` vs numpy `RandomState.permutation`); cannot
  bit-match. RNG carve-out (R-DEFER-3) — blocker, NO failing test.
- REQ-4: Structural partition. train ∪ test = all indices, disjoint, sizes sum to
  `n_samples`. DETERMINISTIC / oracle-pinnable (holds for every seed).
- REQ-5: `test_size=None` default of 0.25. sklearn defaults `test_size` to the
  complement of `train_size`, or 0.25 when both are `None`
  (`sklearn/model_selection/_split.py:2348-2349`, `:2714`, `:2786`); ferrolearn
  REQUIRES a `test_size: f64`. API gap.
- REQ-6: `train_size` parameter. sklearn supports `train_size` (float proportion
  or int count) (`:2689`, `:2394-2402`); ferrolearn has none. API gap.
- REQ-7: `shuffle=False` (deterministic sequential split). sklearn's `shuffle`
  defaults to True; `shuffle=False` yields `np.arange(n_train)` / the contiguous
  tail (`:2789-2796`); ferrolearn ALWAYS shuffles. API gap.
- REQ-8: `stratify` (class-balanced split). sklearn routes through
  `StratifiedShuffleSplit` (`:2799-2806`); ferrolearn has none. API gap.
- REQ-9: Integer `test_size` + variadic `*arrays`. sklearn accepts an integer
  absolute count (`:2391-2392`) and splits any number of arrays (`:2687`,
  `:2810-2814`); ferrolearn takes a single `(x, y)` and a float `test_size`. API gap.
- REQ-10: Empty-split behavior. sklearn RAISES `ValueError` when the resulting
  train (or test) set is empty (`:2414-2419`); ferrolearn CLAMPS `n_test` to
  `[1, n_samples-1]` so an extreme `test_size` never errors. DIVERGENCE.
- REQ-11: R-SUBSTRATE — array type on `ferray-core` (not `ndarray`) and shuffle
  on `ferray::random` (not `rand`'s `SmallRng`). NOT migrated.
- REQ-12: Non-test production consumer.

## Acceptance criteria

- AC-1 (REQ-1): `test_size=0.0` and `test_size=1.0` both error; the live oracle
  `train_test_split(X, y, test_size=1.5)` raises (validation rejects out-of-range
  floats). DETERMINISTIC / oracle-pinnable.
- AC-2 (REQ-2): `n=7, test_size=0.3` → sklearn `len(x_test)==3` (ceil(2.1)) vs
  ferrolearn `2` (round(2.1)); `n=10, test_size=0.33` → sklearn `4` vs ferrolearn
  `3`. The split SIZES are asserted against the live oracle and currently DIVERGE.
  DETERMINISTIC / oracle-pinnable.
- AC-3 (REQ-3): `random_state=Some(seed)` is reproducible call-to-call, but the
  exact membership is NOT asserted equal to sklearn (different PRNG); a carve-out
  blocker is filed, NO failing test.
- AC-4 (REQ-4): for any seed, the union of `y_train` and `y_test` indices equals
  `0..n` with no duplicates, and `len(x_train)+len(x_test)==n`. DETERMINISTIC.
- AC-5 (REQ-5): calling with no `test_size` is a compile error in ferrolearn (it
  is a required positional argument); sklearn's `train_test_split(X, y)` returns a
  75/25 split. API gap.
- AC-6 (REQ-6): no `train_size` argument exists; sklearn's
  `train_test_split(X, y, train_size=0.6)` is unrepresentable. API gap.
- AC-7 (REQ-7): no `shuffle` argument exists; sklearn's
  `train_test_split(range(5), shuffle=False)` → `[[0,1,2],[3,4]]` is
  unrepresentable. API gap.
- AC-8 (REQ-8): no `stratify` argument; sklearn's stratified split is absent.
- AC-9 (REQ-9): `test_size` is `f64` only (no int count) and the signature takes
  exactly `(x, y)` (no `*arrays`); sklearn's `test_size=3` / three-array call is
  unrepresentable. API gap.
- AC-10 (REQ-10): `train_test_split(X, y, test_size=0.99)` errors in sklearn
  (empty train set) but SUCCEEDS in ferrolearn (clamped to `n_test=n-1`).
  DETERMINISTIC / oracle-pinnable divergence.
- AC-11 (REQ-11): owned computation runs on `ferray-core` + `ferray::random`, no
  `ndarray`/`rand` in owned computation.
- AC-12 (REQ-12): `train_test_split` is invoked from non-test production code.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (float test_size validation) | SHIPPED | impl `pub fn train_test_split in split.rs` returns `FerroError::InvalidParameter { name: "test_size", reason: "must be in (0, 1), got {test_size}" }` when `test_size <= 0.0 || test_size >= 1.0`. Mirrors sklearn's float guard `test_size <= 0 or test_size >= 1` (`sklearn/model_selection/_split.py:2357-2359`). Live oracle: `train_test_split(X, y, test_size=1.5)` raises `ValueError` (Verification). DIVERGENCE (honest underclaim): the error VARIANT is `FerroError::InvalidParameter`, not `ValueError`; under R-DEV-2 sklearn `ValueError` maps to ferrolearn's parameter/validation family, so SHIPPED on the rejection behavior with the type-name mapping noted. Tests: `test_invalid_test_size_zero`, `test_invalid_test_size_one`. Non-test consumer: REQ-12. |
| REQ-2 (test-set sizing: ceil vs round) | NOT-STARTED | open prereq blocker (tracking #1723; critic to file specific). impl `pub fn train_test_split in split.rs` computes `let n_test = ((n_samples as f64) * test_size).round() as usize;` then `let n_test = n_test.max(1).min(n_samples - 1);`. sklearn computes `n_test = ceil(test_size * n_samples)` with no clamp (`sklearn/model_selection/_split.py:2390`). ROUND vs CEIL diverges for non-integer products: `n=7, test_size=0.3` → ferrolearn `round(2.1)=2`, sklearn `ceil(2.1)=3`; `n=10, test_size=0.33` → `3` vs `4` (live oracle, Verification). DETERMINISTIC / oracle-pinnable: the split SIZES `len(x_test)`/`len(x_train)` diverge for EVERY seed (independent of the shuffle). Fix is to replace `.round()` with `.ceil()` (and remove the clamp per REQ-10). |
| REQ-3 (exact split membership — RNG carve-out) | NOT-STARTED | open prereq blocker (tracking #1723; R-DEFER-3 carve-out — NO failing test). impl `pub fn train_test_split in split.rs` shuffles `indices` via `indices.shuffle(&mut rng)` where `rng = SmallRng::seed_from_u64(seed)` (seeded) or `SmallRng::from_os_rng()` (`random_state=None`). sklearn shuffles via `ShuffleSplit(..., random_state=random_state)` → numpy `RandomState.permutation` (`sklearn/model_selection/_split.py:2804-2806`). Because the PRNG substrates differ (`SmallRng` vs numpy `RandomState`), the exact train/test MEMBERSHIP cannot bit-match even with equal seeds; `random_state=None` is non-deterministic in BOTH. Per R-DEFER-3 this is a documented carve-out — a blocker is filed but NO failing `#[test]` is pinned. Existing `test_split_is_deterministic_with_seed` asserts only call-to-call reproducibility (ferrolearn↔ferrolearn), not oracle equality. |
| REQ-4 (structural partition) | SHIPPED | impl `pub fn train_test_split in split.rs` builds `indices = (0..n_samples).collect()`, shuffles, then slices `train_idx = &indices[..n_train]` and `test_idx = &indices[n_train..]` — a partition of `0..n_samples` into two disjoint, exhaustive slices with `n_train + n_test == n_samples` (`n_train = n_samples - n_test`). y values are gathered by `train_idx.iter().map(|&i| y[i])`. Mirrors sklearn's `train`/`test` index sets covering all samples (`sklearn/model_selection/_split.py:2806`, `:2810-2814`). DETERMINISTIC / oracle-pinnable: holds for every seed regardless of the shuffle. Tests: `test_no_data_overlap` (all of `0..10` appears exactly once across train+test), `test_split_sizes` (`x_train.nrows()+x_test.nrows()==10`). Non-test consumer: REQ-12. |
| REQ-5 (test_size=None default 0.25) | NOT-STARTED | open prereq blocker (tracking #1723). impl `pub fn train_test_split in split.rs` takes `test_size: f64` as a REQUIRED positional argument — there is no `None`/default path. sklearn defaults `test_size` to 0.25 when both `test_size` and `train_size` are `None` (`sklearn/model_selection/_split.py:2348-2349` with `default_test_size=0.25` at `:2786`; live oracle `train_test_split(X,y)` on `n=10` → 7/3, Verification). Absent: ferrolearn cannot express the default-split call. |
| REQ-6 (train_size parameter) | NOT-STARTED | open prereq blocker (tracking #1723). impl `pub fn train_test_split in split.rs` has no `train_size` parameter; `n_train` is derived solely as `n_samples - n_test`. sklearn supports `train_size` as a float proportion or int count (`sklearn/model_selection/_split.py:2689`, `:2394-2402`), including the `train_size + test_size > 1` consistency check (`:2383-2387`). Absent end-to-end. |
| REQ-7 (shuffle=False sequential split) | NOT-STARTED | open prereq blocker (tracking #1723). impl `pub fn train_test_split in split.rs` ALWAYS calls `indices.shuffle(&mut rng)` — there is no `shuffle` parameter and no sequential path. sklearn's `shuffle=False` yields `train = np.arange(n_train)`, `test = np.arange(n_train, n_train + n_test)` (`sklearn/model_selection/_split.py:2789-2796`); live oracle `train_test_split(range(5), shuffle=False)` → `[[0,1,2],[3,4]]` (Verification). Absent. |
| REQ-8 (stratify) | NOT-STARTED | open prereq blocker (tracking #1723). impl `pub fn train_test_split in split.rs` has no `stratify` parameter and no class-balanced path. sklearn routes a non-`None` `stratify` through `StratifiedShuffleSplit` (`sklearn/model_selection/_split.py:2799-2806`) and forbids `stratify` with `shuffle=False` (`:2790-2793`). Absent. |
| REQ-9 (int test_size + variadic *arrays) | NOT-STARTED | open prereq blocker (tracking #1723). impl `pub fn train_test_split in split.rs` has signature `train_test_split<F>(x: &Array2<F>, y: &Array1<F>, test_size: f64, random_state: Option<u64>)` — a single `(x, y)` pair and a `f64`-only `test_size`. sklearn accepts an integer absolute count (`n_test = float(test_size)`, `sklearn/model_selection/_split.py:2391-2392`) and a variadic `*arrays` returning `2 * len(arrays)` slices (`:2687`, `:2810-2814`); live oracle `train_test_split(arange(10), test_size=3)` → 7/3 (Verification). Absent. |
| REQ-10 (empty-split: raise vs clamp) | NOT-STARTED | open prereq blocker (tracking #1723). impl `pub fn train_test_split in split.rs` CLAMPS the test count: `let n_test = n_test.max(1).min(n_samples - 1);` — so an extreme `test_size` never produces an empty split and never errors. sklearn RAISES `ValueError` when the resulting train set is empty: `if n_train == 0: raise ValueError("… the resulting train set will be empty …")` (`sklearn/model_selection/_split.py:2414-2419`). Live oracle: `train_test_split(X, y, test_size=0.99)` on `n=10` raises `ValueError`; ferrolearn clamps to `n_test=9` and SUCCEEDS (Verification). DIVERGENCE in observable behavior (success vs raise). |
| REQ-11 (ferray substrate) | NOT-STARTED | open prereq blocker (tracking #1723; R-SUBSTRATE-2/3). `split.rs` imports `ndarray::{Array1, Array2}` and `rand::{SeedableRng, rngs::SmallRng, seq::SliceRandom}`. Destination substrate is the `ferray-core` array type and `ferray::random` (R-SUBSTRATE-1). Not migrated; the `SmallRng` shuffle is also the proximate cause of the REQ-3 RNG carve-out. |
| REQ-12 (consumer) | SHIPPED | Crate re-export: `lib.rs` (`pub use split::train_test_split` and `pub mod split`); meta-crate re-export reachable as `ferrolearn::model_selection::train_test_split` (used at `ferrolearn/tests/integration_tests.rs`, test-only). R-DEFER-1 / S5: `train_test_split` is an existing boundary public function (grandfathered); the boundary utility IS the public API. CAVEAT (honest underclaim): grep finds NO non-test, non-re-export caller, NO internal cross-validation consumer (the CV splitters live in `cross_validation.rs`/`splitters.rs` and do not call `train_test_split`), and NO `ferrolearn-python` binding. SHIPPED on the boundary re-export per S5, not a dedicated production caller; the missing Python binding and internal consumer are noted. |

## Architecture

ferrolearn collapses sklearn's `*arrays`/`ShuffleSplit` wrapper into a single
free function `train_test_split<F>(x: &Array2<F>, y: &Array1<F>, test_size: f64,
random_state: Option<u64>) -> Result<TrainTestSplit<F>, FerroError>` where
`TrainTestSplit<F> = (Array2<F>, Array2<F>, Array1<F>, Array1<F>)`. sklearn
instead validates sizes via `_validate_shuffle_split`
(`sklearn/model_selection/_split.py:2343`), then dispatches to
`ShuffleSplit`/`StratifiedShuffleSplit` and re-indexes every input array
(`:2798-2814`).

The function body has three phases. (1) Validation: float `test_size ∈ (0,1)`
(REQ-1, mirroring `:2357-2359`), `y.len() == x.nrows()` (`FerroError::ShapeMismatch`),
and `n_samples >= 2` (`FerroError::InsufficientSamples`). sklearn's analogous
guards are spread across `indexable`/`_num_samples` and the `n_train == 0` raise
(`:2414`). (2) Shuffle: `indices = (0..n_samples).collect()` permuted by
`SmallRng::seed_from_u64(seed)` or `SmallRng::from_os_rng()` — REQ-3's RNG
substrate, which cannot reproduce numpy `RandomState.permutation`. (3) Sizing +
gather: `n_test = round(test_size * n_samples)` clamped to `[1, n_samples-1]`
(REQ-2 divergence: sklearn uses `ceil` with no clamp, `:2390`; REQ-10 divergence:
sklearn raises on empty instead of clamping, `:2414`), `n_train = n_samples -
n_test`, then `indices[..n_train]` / `indices[n_train..]` are gathered into the
four output arrays.

The structural partition (REQ-4) is correct and seed-independent: every index in
`0..n_samples` lands in exactly one of the two disjoint, exhaustive slices, and
the sizes sum to `n_samples`. This is the property that survives the RNG
substrate difference.

What is structurally absent vs sklearn: the `test_size=None` 0.25 default
(REQ-5, `:2348-2349`), `train_size` (REQ-6, `:2394-2402`), `shuffle=False`
(REQ-7, `:2789-2796`), `stratify` (REQ-8, `:2799-2806`), integer `test_size` and
variadic `*arrays` (REQ-9, `:2391-2392`, `:2810-2814`), the empty-split raise
(REQ-10, `:2414`), and the `ferray` substrate (REQ-11). The two sizing/clamp
divergences (REQ-2, REQ-10) are the FIXABLE, oracle-pinnable correctness bugs;
the rest are feature gaps; REQ-3 is the RNG carve-out.

Invariants: `test_size ∈ (0,1)`; `n_samples >= 2`; `y.len() == x.nrows()`;
`n_train + n_test == n_samples`; train and test index sets partition
`0..n_samples`; `random_state=Some(s)` is reproducible call-to-call.

## Verification

Commands establishing the SHIPPED claims and pinning the divergences (baseline
`4ca941196bef12467cffceb865d996a0d6c263c9`):

- `cargo test -p ferrolearn-model-sel --lib split::tests` → the 9 `split::*`
  tests pass (REQ-1 validation; REQ-4 partition via `test_no_data_overlap` /
  `test_split_sizes`; reproducibility via `test_split_is_deterministic_with_seed`).
- REQ-2 oracle (sizing: ceil vs round, live oracle — the SPLIT SIZES, seed-independent):
  ```
  python3 -c "import numpy as np, math; from sklearn.model_selection import train_test_split
  for n, ts in [(7,0.3),(10,0.33)]:
      X=np.arange(n*2).reshape((n,2)); y=np.arange(n)
      _,Xte,_,_=train_test_split(X,y,test_size=ts,random_state=0)
      print(n, ts, 'sklearn n_test', len(Xte), 'ceil', math.ceil(ts*n), 'ferro round', round(ts*n))"
  # -> 7 0.3 sklearn n_test 3 ceil 3 ferro round 2
  # -> 10 0.33 sklearn n_test 4 ceil 4 ferro round 3
  ```
  Pin a `#[test]` asserting `x_test.nrows()` for `(n=7, test_size=0.3)` and
  `(n=10, test_size=0.33)` equals the live-oracle `ceil` count above (R-CHAR-3);
  it fails today (ferrolearn returns `round`) — REQ-2 NOT-STARTED.
- REQ-5 oracle (default 0.25, live oracle):
  `train_test_split(np.arange(20).reshape(10,2), np.arange(10), random_state=0)`
  → `len(train)=7, len(test)=3`. ferrolearn requires an explicit `test_size` —
  REQ-5 NOT-STARTED.
- REQ-7 oracle (shuffle=False, live oracle):
  `train_test_split(list(range(5)), shuffle=False)` → `[[0,1,2],[3,4]]`.
  ferrolearn always shuffles — REQ-7 NOT-STARTED.
- REQ-9 oracle (int test_size, live oracle):
  `train_test_split(np.arange(10), test_size=3, random_state=0)` → `7/3`.
  ferrolearn's `test_size` is `f64`-only and takes a single `(x,y)` — REQ-9 NOT-STARTED.
- REQ-10 oracle (empty-split raise vs clamp, live oracle):
  `train_test_split(np.arange(20).reshape(10,2), np.arange(10), test_size=0.99)`
  → `ValueError("… the resulting train set will be empty …")`; ferrolearn clamps
  to `n_test=9` and returns `Ok` — REQ-10 NOT-STARTED. Pin a `#[test]` asserting
  the extreme-`test_size` call ERRORS (it currently succeeds).
- REQ-3 (RNG carve-out): `random_state=None` is non-deterministic in both;
  `SmallRng` ≠ numpy `RandomState.permutation`, so exact membership cannot match.
  Per R-DEFER-3, NO failing test is pinned — a carve-out blocker is filed.

SHIPPED: REQ-1 (float `test_size` validation; `FerroError` variant diverges from
`ValueError`), REQ-4 (structural partition — disjoint, exhaustive, sizes sum to
`n_samples`), REQ-12 (boundary re-export consumer; no dedicated non-test caller,
no internal CV consumer, no Python binding — honest underclaim). NOT-STARTED
(tracking #1723; the critic files per-REQ blockers): REQ-2 (ceil sizing —
FIXABLE, oracle-pinnable), REQ-3 (exact membership — RNG carve-out, NO failing
test), REQ-5 (`test_size=None` 0.25 default), REQ-6 (`train_size`), REQ-7
(`shuffle=False`), REQ-8 (`stratify`), REQ-9 (int `test_size` + variadic
`*arrays`), REQ-10 (empty-split raise vs clamp — FIXABLE, oracle-pinnable),
REQ-11 (ferray substrate). Per R-DEFER-2 the table is binary SHIPPED/NOT-STARTED.
