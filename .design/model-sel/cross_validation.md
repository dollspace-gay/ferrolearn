# cross_validation (KFold / StratifiedKFold / CrossValidator / cross_val_score / cross_validate / cross_val_predict / permutation_test_score)

<!--
tier: 3-component
status: draft
baseline-commit: 699b4835d444cfa3c0d2ea5631b7b840bdb60f89
upstream-paths:
  - sklearn/model_selection/_split.py        # KFold (:441), _iter_test_indices (:521), StratifiedKFold (:665), _make_test_folds (:746-806)
  - sklearn/model_selection/_validation.py   # cross_validate (:122), cross_val_score (:560), _fit_and_score (:729), cross_val_predict (:1054), permutation_test_score (:1502)
-->

## Summary

`ferrolearn-model-sel/src/cross_validation.rs` is the WIDEST-consumed unit in the
crate. It mirrors scikit-learn's cross-validation core from two upstream files:

- the **splitters** `KFold` (`sklearn/model_selection/_split.py:441`) and
  `StratifiedKFold` (`:665`), plus the `CrossValidator` trait that abstracts a
  fold-index generator (the ferrolearn analog of `BaseCrossValidator`);
- the **scoring drivers** `cross_val_score` (`sklearn/model_selection/_validation.py:560`),
  `cross_validate` (`:122`), `cross_val_predict` (`:1054`), and
  `permutation_test_score` (`:1502`).

The deterministic mechanics SHIP: non-shuffled `KFold` fold membership matches
sklearn exactly; `cross_val_score`/`cross_validate` per-fold fit/predict/score and
timing are faithful; `cross_val_predict` original-order placement is correct on a
partition; the `permutation_test_score` p-value formula
`(count(perm>=real)+1)/(n_permutations+1)` matches sklearn. These are consumed
in production by `GridSearchCV::fit` (`grid_search.rs`),
`RandomizedSearchCV::fit` (`random_search.rs`), and the halving searches
(`halving_grid_search.rs`/`halving_random_search.rs`) via
`cross_val_score(&pipeline, x, y, self.cv.as_ref(), self.scoring)?`, and
`KFold` flows through them as `&dyn CrossValidator`.

Several behaviors DIVERGE deterministically and are NOT-STARTED with filed
blockers — the critic should pin failing tests for each:

- **error_score=np.nan continue** (the KEY fixable divergence; this unit OWNS the
  cross-unit blocker that `grid_search`/`random_search`/`validation_curve`
  deferred here, S8): all three drivers `?`-abort on a fold fit/predict/score
  failure, where sklearn `_fit_and_score` (`:729`) NaN-fills the failing fold
  (both train+test, independently scored) and CONTINUES.
- **StratifiedKFold non-shuffled allocation**: ferrolearn sorts classes
  LEXICOGRAPHICALLY with a rotating-`fold_offset` round-robin; sklearn
  `_make_test_folds` encodes classes by ORDER OF APPEARANCE and allocates via
  `bincount(y_order[i::n_splits])` (`:746-806`).
- **StratifiedKFold error-vs-warn**: ferrolearn errors if ANY class count <
  `n_splits`; sklearn raises only if `n_splits > ALL` class counts (`:770-774`)
  and merely warns otherwise (`:775-781`).
- **cross_val_predict non-partition**: ferrolearn leaves unpredicted samples at
  `0.0`; sklearn raises `ValueError("cross_val_predict only works for
  partitions")` (`:1054`).

The remaining gaps are architectural NOT-STARTED (StratifiedKFold has no
`CrossValidator` impl — the trait `fold_indices(n_samples)` has no y/groups
channel; mandatory `cv`/`scoring`; no `return_estimator`/`return_indices`/
multimetric/groups/n_jobs) or an R-DEFER-3 RNG carve-out (shuffle/permutation
EXACT membership: `SmallRng` vs numpy — blocker, NO failing test).

## Upstream reference (scikit-learn 1.5.2, commit 156ef14)

### `KFold` (`sklearn/model_selection/_split.py:441`)
- `:441` — `class KFold(_UnsupportedGroupCVMixin, _BaseKFold)`;
  `__init__(self, n_splits=5, *, shuffle=False, random_state=None)` (on `_BaseKFold`).
- `:521-535` — `_iter_test_indices`: `indices = np.arange(n_samples)`; if
  `self.shuffle`, `check_random_state(self.random_state).shuffle(indices)` FIRST;
  then `fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=int);
  fold_sizes[: n_samples % n_splits] += 1` — the first `n_samples % n_splits`
  folds get `+1` — and folds are CONSECUTIVE slices `indices[start:stop]`.
  NON-shuffled is fully DETERMINISTIC: for `n=10, k=3` the test folds are
  `[0,1,2,3],[4,5,6],[7,8,9]`.

### `StratifiedKFold` (`sklearn/model_selection/_split.py:665`)
- `:665` — `class StratifiedKFold(_BaseKFold)`;
  `__init__(self, n_splits=5, *, shuffle=False, random_state=None)`.
- `:760-765` — `_, y_idx, y_inv = np.unique(y, return_index=True,
  return_inverse=True); _, class_perm = np.unique(y_idx, return_inverse=True);
  y_encoded = class_perm[y_inv]` — classes are encoded by ORDER OF APPEARANCE
  (NOT lexicographic).
- `:770-774` — `y_counts = np.bincount(y_encoded)`; `if np.all(self.n_splits >
  y_counts): raise ValueError("n_splits=%d cannot be greater than the number of
  members in each class.")` — raises only when EVERY class count < `n_splits`.
- `:775-781` — `if self.n_splits > min_groups: warnings.warn(... UserWarning)` —
  a class smaller than `n_splits` (but not all) only WARNS, it does NOT error.
- `:786-792` — `y_order = np.sort(y_encoded); allocation[i,k] =
  bincount(y_order[i::self.n_splits], minlength=n_classes)[k]` — round-robin over
  the SORTED encoded labels gives the per-fold per-class allocation.
- `:798-805` — for each class `k`, `folds_for_class =
  np.arange(self.n_splits).repeat(allocation[:, k])` (shuffled iff
  `self.shuffle`), then `test_folds[y_encoded == k] = folds_for_class` —
  class-`k` samples (in ORIGINAL order) are block-assigned to folds.
  NON-shuffled is fully DETERMINISTIC.

### `cross_val_score` (`sklearn/model_selection/_validation.py:560`)
- `:560-575` — `cross_val_score(estimator, X, y=None, *, groups=None,
  scoring=None, cv=None, n_jobs=None, verbose=0, fit_params=None, params=None,
  pre_dispatch="2*n_jobs", error_score=np.nan)` — a thin wrapper around
  `cross_validate` that returns `cv_results["test_score"]`.

### `cross_validate` (`sklearn/model_selection/_validation.py:122`)
- `:122-160` — `cross_validate(estimator, X, y=None, *, groups=None,
  scoring=None, cv=None, n_jobs=None, verbose=0, params=None,
  pre_dispatch="2*n_jobs", return_train_score=False, return_estimator=False,
  return_indices=False, error_score=np.nan)` — runs `_fit_and_score` per fold,
  returns a dict with `test_score`/`fit_time`/`score_time` and, optionally,
  `train_score`/`estimator`/`indices`. Supports MULTIMETRIC scoring.

### `_fit_and_score` (`sklearn/model_selection/_validation.py:729`)
- `:890-905` — on a fit FAILURE with numeric `error_score` (default `np.nan`),
  it NaN-fills BOTH `train_scores` and `test_scores` for that fold, emits a
  `FitFailedWarning`, and CONTINUES (does not raise).
- `:910` / `:915` — train and test scores are computed by INDEPENDENT `_score`
  calls. (This is the SAME contract pinned SHIPPED-as-divergence for
  `validation_curve` REQ-7, blockers #1758/#1762 — this unit OWNS the fix.)

### `cross_val_predict` (`sklearn/model_selection/_validation.py:1054`)
- `:1054-1070` — `cross_val_predict(estimator, X, y=None, *, groups=None,
  cv=None, n_jobs=None, verbose=0, params=None, pre_dispatch="2*n_jobs",
  method="predict")`. Each sample is predicted exactly once (in its test fold);
  predictions are returned in ORIGINAL sample order. It inverts the test indices
  and REQUIRES a partition — raises `ValueError("cross_val_predict only works for
  partitions")` if some sample is never in a test fold.

### `permutation_test_score` (`sklearn/model_selection/_validation.py:1502`)
- `:1502-1520` — `permutation_test_score(estimator, X, y, *, groups=None, cv=None,
  n_permutations=100, n_jobs=None, random_state=0, verbose=0, scoring=None,
  fit_params=None)`.
- `:1677` — `score = _permutation_test_score(...)` = the mean CV score on the real
  (unpermuted) `y`.
- `:1691` — `permutation_scores` = the mean CV score for each of `n_permutations`
  shuffles of `y` via `_shuffle`.
- `:1697` — `pvalue = (np.sum(permutation_scores >= score) + 1.0) /
  (n_permutations + 1)`.

## Requirements

- REQ-KFOLD (non-shuffled KFold fold membership — DETERMINISTIC): `n_splits`
  consecutive folds; the first `n_samples % n_splits` folds get `+1` sample;
  reject `n_splits < 2` and `n_samples < n_splits`. **MATCH** (R-DEV-1
  numerical/structural contract). Mirrors `:521-535`. SHIPPED.
- REQ-SKFOLD (non-shuffled StratifiedKFold fold membership — DETERMINISTIC,
  DIVERGENT): ferrolearn's LEXICOGRAPHIC class sort + rotating-`fold_offset`
  round-robin diverges from sklearn's appearance-order encoding (`:760-765`) +
  `bincount(y_order[i::n_splits])` allocation (`:786-792`) + block assignment
  (`:798-805`). **MATCH-intent / DEVIATE-actual** (R-DEV-1 — fold membership IS
  the contract; ferrolearn computes a different deterministic membership).
  NOT-STARTED (blocker #1791).
- REQ-SKFOLD-ERRWARN (StratifiedKFold error-vs-warn — DIVERGENT): ferrolearn
  errors (`InsufficientSamples`) if ANY class count < `n_splits`; sklearn raises
  `ValueError` only if `n_splits > ALL` class counts (`:770-774`) and WARNS
  otherwise (`:775-781`). **DEVIATE-actual** (R-DEV-2 — wrong error condition).
  NOT-STARTED (blocker #1792).
- REQ-CVS (cross_val_score mechanic): per-fold build train/test subsets, fit a
  fresh pipeline, predict on test, score; return the per-fold test scores.
  **MATCH** (R-DEV-1/3) for the NON-FAILING path. Mirrors `cross_validate ->
  test_score` (`:560`, `:122`). SHIPPED.
- REQ-CVALIDATE (cross_validate test/train scores + timing): like REQ-CVS plus
  `fit_time`/`score_time` per fold and optional train scores under
  `return_train_score`. **MATCH** (R-DEV-3) for those facets. Mirrors `:122`.
  SHIPPED. (`return_estimator`/`return_indices`/multimetric are absent — covered
  by REQ-DEFAULTS.)
- REQ-CVPREDICT (cross_val_predict original-order placement): out-of-fold
  predictions placed at `test_idx` positions in a length-`n_samples` array
  (ORIGINAL order). **MATCH** (R-DEV-3) for the PARTITION case. Mirrors
  `:1054`. SHIPPED. The NON-PARTITION case diverges — see REQ-CVPREDICT-PARTITION.
- REQ-CVPREDICT-PARTITION (non-partition behavior — DIVERGENT): ferrolearn leaves
  unpredicted samples at `0.0`; sklearn raises `ValueError("cross_val_predict only
  works for partitions")` (`:1054`). **DEVIATE-actual** (R-DEV-2). NOT-STARTED
  (blocker #1793).
- REQ-PERM (permutation_test_score p-value): `p_value =
  (count(perm >= real) + 1) / (n_permutations + 1)`; `real` = mean CV score.
  **MATCH** (R-DEV-1) for the p-value mechanic. Mirrors `:1697`. SHIPPED. The
  EXACT permuted scores are an RNG carve-out (REQ-SHUFFLE-RNG).
- REQ-ERROR-SCORE (error_score=np.nan continue — the KEY fixable divergence,
  cross-unit OWNER): `cross_val_score`/`cross_validate`/`cross_val_predict`
  `?`-abort on a fold fit/predict/score failure; sklearn `_fit_and_score`
  (default `error_score=np.nan`) NaN-fills the failing fold (both train+test,
  independently scored) and CONTINUES (`:729`, `:890-905`). **DEVIATE-actual**
  (R-DEV-1 — sklearn returns a partial NaN-bearing result, ferrolearn returns
  `Err`). NOT-STARTED (blocker #1790). This unit OWNS the blocker the
  curve/search units deferred here (S8).
- REQ-SKFOLD-CV (StratifiedKFold has no CrossValidator impl — architectural): the
  trait `fold_indices(n_samples)` has no y/groups channel, so `StratifiedKFold`
  cannot be passed to `cross_val_score`/`grid_search` as `&dyn CrossValidator`.
  **MATCH-intent / gap** (same channel-gap as `group_splitters`). NOT-STARTED
  (blocker #1794).
- REQ-SHUFFLE-RNG (KFold/StratifiedKFold shuffle + permutation EXACT membership —
  RNG carve-out): the shuffle/permutation RNG is `SmallRng` vs numpy's
  `Mersenne`, so EXACT membership differs. Per R-DEFER-3 this is a carve-out:
  blocker, NO failing test. The STRUCTURAL facts (fold sizes, partition,
  seed-determinism-across-runs) ARE SHIPPED. NOT-STARTED (blocker #1795).
- REQ-DEFAULTS (cv=None / scoring=None / groups / n_jobs / fit_params /
  return_estimator / return_indices / multimetric): ferrolearn requires a
  MANDATORY `cv` and `scoring` fn, has no `groups`/`n_jobs`/`fit_params` channel,
  and `cross_validate` lacks `return_estimator`/`return_indices`/multimetric.
  sklearn defaults `cv=None` (→5-fold classifier-aware via `check_cv`),
  `scoring=None` (→`check_scoring`), and supports all of those (`:122`, `:560`).
  **MATCH-intent / gap**. NOT-STARTED (blocker #1796).
- REQ-X-1 (R-SUBSTRATE): production code uses `ndarray::{Array1, Array2}` (array
  type), `rand::{SeedableRng, rngs::SmallRng, seq::SliceRandom}` (random
  substrate), and `std::collections::HashMap`; the destination is `ferray-core` /
  `ferray::random` (R-SUBSTRATE-1). NOT-STARTED (blocker #1797).
- REQ-X-2 (non-test production consumer): `cross_val_score` is called by
  `GridSearchCV::fit` (`grid_search.rs`), `RandomizedSearchCV::fit`
  (`random_search.rs`), and both halving searches
  (`halving_grid_search.rs`/`halving_random_search.rs`); `KFold` feeds all of
  them as `&dyn CrossValidator`. The whole surface is re-exported at
  `pub use cross_validation::{...} in lib.rs`. SHIPPED (the widest in the crate).

## Acceptance criteria

Each AC is pinnable against a LIVE sklearn 1.5.2 call (R-CHAR-3 — expected values
come from the oracle, never copied from the ferrolearn side).

- AC-KFOLD (REQ-KFOLD): live oracle
  ```
  python3 -c "import numpy as np
  from sklearn.model_selection import KFold
  print([list(te) for tr,te in KFold(3,shuffle=False).split(np.zeros((10,1)))])"
  # -> [[0,1,2,3],[4,5,6],[7,8,9]]
  ```
  ferrolearn `KFold::new(3).split(10)` → test folds `[0,1,2,3]`, `[4,5,6]`,
  `[7,8,9]`. DETERMINISTIC, pinnable. Also `KFold::new(1).fold_indices(10)` →
  `Err`; `KFold::new(5).fold_indices(3)` → `Err`.
- AC-SKFOLD (REQ-SKFOLD, DIVERGENCE — the allocation pin): a y whose appearance
  order differs from lexicographic order isolates the divergence:
  ```
  python3 -c "import numpy as np
  from sklearn.model_selection import StratifiedKFold
  y = np.array([2,2,2,2, 0,0,0,0, 1,1,1,1])   # classes appear 2,0,1
  print([sorted(int(i) for i in te) for tr,te in StratifiedKFold(3,shuffle=False).split(np.zeros((12,1)), y)])"
  # sklearn -> [[0,1,4,8],[2,5,6,9],[3,7,10,11]]
  ```
  ferrolearn `StratifiedKFold::new(3).split(&y)` → `[[0,4,5,8],[1,6,9,10],
  [2,3,7,11]]` (lexicographic class sort + rotating-fold_offset). DIVERGENCE.
  NOT-STARTED (blocker #1791). The critic should pin a failing test asserting the
  sklearn membership.
- AC-SKFOLD-ERRWARN (REQ-SKFOLD-ERRWARN, DIVERGENCE): one class smaller than
  `n_splits` but not all:
  ```
  python3 -c "import numpy as np, warnings
  from sklearn.model_selection import StratifiedKFold
  y = np.array([0,0,0,0,0, 1,1])   # class 1 has 2 < n_splits=3
  with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter('always')
      list(StratifiedKFold(3,shuffle=False).split(np.zeros((7,1)), y))
      print('SUCCEEDS, warning:', [str(x.message)[:40] for x in w])"
  # -> SUCCEEDS, warning: ['The least populated class in y has only ...']
  ```
  ferrolearn `StratifiedKFold::new(3).split(&y)` → `Err(InsufficientSamples)`.
  DIVERGENCE. NOT-STARTED (blocker #1792). (Contrast: `y=[0,0,1,1]`, `n_splits=3`
  — n_splits > ALL class counts — sklearn raises `ValueError`; ferrolearn also
  errors. The divergence is ONLY the "one-class-too-small" case.)
- AC-CVS (REQ-CVS): `cross_val_score` returns one score per fold on the
  non-failing path. ferrolearn tests
  `test_cross_val_score_returns_correct_number_of_scores` (5 scores for
  `KFold(5)`), `test_cross_val_score_perfect_constant_target` (constant y, mean
  estimator → MSE 0 each fold). The mechanic mirrors `cross_validate ->
  test_score` (`:560`).
- AC-CVALIDATE (REQ-CVALIDATE): `cross_validate(..., return_train_score=true)`
  yields `test_scores`, `train_scores: Some(..)`, `fit_times`, `score_times`,
  each length `n_folds`; with `false`, `train_scores` is `None`. ferrolearn tests
  `test_cross_validate_returns_correct_fold_count`,
  `test_cross_validate_with_train_scores`, `test_cross_validate_timing_non_negative`.
- AC-CVPREDICT (REQ-CVPREDICT): on a partition cv, each sample is predicted once,
  in original order, length `n_samples`. Oracle:
  ```
  python3 -c "import numpy as np
  from sklearn.model_selection import cross_val_predict, KFold
  from sklearn.linear_model import LinearRegression
  X=np.arange(20).reshape(10,2).astype(float); y=np.arange(10).astype(float)
  print(len(cross_val_predict(LinearRegression(), X, y, cv=KFold(5))))"   # 10
  ```
  ferrolearn `test_cross_val_predict_length` (length 20),
  `test_cross_val_predict_constant_target` (constant y → constant predictions).
- AC-CVPREDICT-PARTITION (REQ-CVPREDICT-PARTITION, DIVERGENCE): a non-partition
  cv:
  ```
  python3 -c "import numpy as np
  from sklearn.model_selection import cross_val_predict, ShuffleSplit
  from sklearn.linear_model import LinearRegression
  X=np.arange(20).reshape(10,2).astype(float); y=np.arange(10).astype(float)
  try:
      cross_val_predict(LinearRegression(), X, y, cv=ShuffleSplit(3,test_size=0.3,random_state=0))
  except Exception as e: print(type(e).__name__, str(e)[:50])"
  # -> ValueError cross_val_predict only works for partitions
  ```
  ferrolearn leaves unpredicted samples at `0.0` and returns `Ok`. DIVERGENCE.
  NOT-STARTED (blocker #1793).
- AC-PERM (REQ-PERM, p-value formula): live oracle confirming
  `(count(perm>=score)+1)/(n+1)`:
  ```
  python3 -c "import numpy as np
  from sklearn.model_selection import permutation_test_score, KFold
  from sklearn.linear_model import LinearRegression
  X=np.random.RandomState(0).randn(30,3); y=np.random.RandomState(1).randn(30)
  s,perm,p = permutation_test_score(LinearRegression(), X, y, cv=KFold(3),
      n_permutations=20, random_state=0, scoring='neg_mean_squared_error')
  manual=(np.sum(np.array(perm)>=s)+1.0)/(20+1.0)
  print(round(p,6), round(manual,6), abs(p-manual)<1e-12)"   # 0.809524 0.809524 True
  ```
  ferrolearn `permutation_test_score(..)` computes `p_value = (n_ge as f64 + 1.0)
  / (n_permutations as f64 + 1.0)` — the IDENTICAL formula. Tests
  `test_permutation_test_score_returns_correct_counts`,
  `test_permutation_test_score_deterministic`,
  `test_permutation_test_score_p_value_range`. The EXACT permuted scores are an
  RNG carve-out (REQ-SHUFFLE-RNG); only the FORMULA is pinned.
- AC-ERROR-SCORE (REQ-ERROR-SCORE, DIVERGENCE — the KEY pin): when one fold's fit
  fails, sklearn returns a NaN in that fold and CONTINUES:
  ```
  python3 -c "import numpy as np, warnings
  from sklearn.model_selection import cross_val_score, KFold
  from sklearn.base import BaseEstimator, RegressorMixin
  class G(BaseEstimator, RegressorMixin):
      def fit(self,X,y):
          if np.any(X[:,0]==99): raise ValueError('boom')
          self.m_=y.mean(); return self
      def predict(self,X): return np.full(X.shape[0], self.m_)
  X=np.arange(12).reshape(6,2).astype(float); X[0,0]=99.0
  y=np.arange(6).astype(float)
  with warnings.catch_warnings():
      warnings.simplefilter('ignore')
      s = cross_val_score(G(), X, y, cv=KFold(3), scoring='neg_mean_squared_error')
  print(s.tolist(), 'any nan:', bool(np.isnan(s).any()))"
  # -> [-9.25, nan, nan] any nan: True
  ```
  ferrolearn `cross_val_score` calls `pipeline.fit(...)?` / `predict(...)?` /
  `scoring(...)?` — a failing fold returns `Err`, not a NaN-bearing
  `Array1<f64>`. DIVERGENCE. NOT-STARTED (blocker #1790). The critic should pin a
  failing test: a pipeline failing on one fold ⇒ ferrolearn returns scores with
  NaN in that fold, not `Err`.
- AC-SKFOLD-CV (REQ-SKFOLD-CV): `StratifiedKFold` exposes `split(&Array1<usize>)`
  and does NOT `impl CrossValidator`, so `cross_val_score(&pipeline, x, y,
  &StratifiedKFold::new(3), scoring)` does not type-check (`&StratifiedKFold` is
  not `&dyn CrossValidator`). Architectural gap. NOT-STARTED (blocker #1794).

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-KFOLD (non-shuffled KFold membership, DETERMINISTIC) | SHIPPED | impl `fn split_result in cross_validation.rs` (and `impl CrossValidator for KFold`'s `fn fold_indices`) builds `indices = (0..n_samples).collect()`, computes `base_size = n/k`, `remainder = n%k`, and `pos += base_size + if fold < remainder { 1 } else { 0 }` so the first `remainder` folds get `+1`, then emits CONSECUTIVE slices `indices[test_start..test_end]` as the test fold — exactly sklearn's `fold_sizes = np.full(n_splits, n//n_splits); fold_sizes[:n % n_splits] += 1` and consecutive `indices[start:stop]` (`sklearn/model_selection/_split.py:521-535`). It rejects `n_splits < 2` (`FerroError::InvalidParameter`) and `n_samples < n_splits` (`FerroError::InsufficientSamples`). Live oracle (AC-KFOLD): `KFold(3,shuffle=False).split(np.zeros((10,1)))` → `[[0,1,2,3],[4,5,6],[7,8,9]]` matches ferrolearn. Tests: `test_kfold_fold_sizes_unequal` (10/3 → sizes differ by ≤1), `test_kfold_no_overlap_full_coverage`, `test_kfold_invalid_n_splits`, `test_kfold_insufficient_samples`. The exact `n=10,k=3` membership is the oracle-pinnable `#[test]` (R-CHAR-3). Non-test consumer: REQ-X-2 (`&dyn CrossValidator` in `grid_search.rs`/`random_search.rs`/halving). |
| REQ-SKFOLD (non-shuffled StratifiedKFold membership, DETERMINISTIC) | NOT-STARTED | open prereq blocker #1791. DIVERGENCE. impl `fn split in cross_validation.rs` sorts classes LEXICOGRAPHICALLY (`classes.sort_unstable()`) and assigns folds by a rotating `fold_offset` round-robin (`base = count/k`, `extra = count%k`, the `extra` folds starting at `fold_offset` get `+1`, `fold_offset = (fold_offset + extra) % k` across classes), block-assigning each class's samples in original order. sklearn `_make_test_folds` instead encodes classes by ORDER OF APPEARANCE (`np.unique(return_index)` + `class_perm`, `:760-765`), computes `allocation[i,k] = bincount(y_order[i::n_splits])[k]` over the SORTED encoded labels (`:786-792`), and block-assigns (`:798-805`). Oracle (AC-SKFOLD): `y=[2,2,2,2,0,0,0,0,1,1,1,1]`, k=3 → sklearn `[[0,1,4,8],[2,5,6,9],[3,7,10,11]]` vs ferrolearn `[[0,4,5,8],[1,6,9,10],[2,3,7,11]]`. (On y whose appearance order == lexicographic and counts divisible cases they coincide — `test_skfold_basic`/`test_skfold_class_balance` pass — but the appearance-order case diverges.) |
| REQ-SKFOLD-ERRWARN (StratifiedKFold error-vs-warn) | NOT-STARTED | open prereq blocker #1792. DIVERGENCE. impl `fn split in cross_validation.rs` loops over each class and returns `Err(FerroError::InsufficientSamples { .. })` as soon as ANY `count < self.n_splits`. sklearn raises `ValueError` ONLY when `np.all(self.n_splits > y_counts)` — i.e. EVERY class count < n_splits (`sklearn/model_selection/_split.py:770-774`) — and merely `warnings.warn(... UserWarning)` when `self.n_splits > min_groups` but not all (`:775-781`). Oracle (AC-SKFOLD-ERRWARN): `y=[0]*5+[1]*2`, n_splits=3 → sklearn SUCCEEDS with a UserWarning; ferrolearn errors. Test `test_skfold_class_too_small` currently ASSERTS the (divergent) error — the critic should pin a test that the one-class-too-small case SUCCEEDS. |
| REQ-CVS (cross_val_score mechanic, non-failing path) | SHIPPED | impl `pub fn cross_val_score in cross_validation.rs` shape-checks `y.len() == x.nrows()` (`FerroError::ShapeMismatch`), calls `cv.fold_indices(n_samples)?`, and per fold copies the train/test row subsets into fresh `Array2`/`Array1`, fits `pipeline.fit(&x_train, &y_train)?`, predicts `fitted.predict(&x_test)?`, and pushes `scoring(&y_test, &y_pred)?` — returning `Array1::from_vec(scores)` of length `n_folds`. This mirrors `cross_validate` → `cv_results["test_score"]` (`sklearn/model_selection/_validation.py:560`, `:122`) for the NON-FAILING path (the failing path is REQ-ERROR-SCORE). Live oracle (AC-CVS): `cross_val_score(LinearRegression(), X, y, cv=KFold(5))` returns 5 scores. Tests: `test_cross_val_score_returns_correct_number_of_scores`, `test_cross_val_score_perfect_constant_target`, `test_cross_val_score_with_transformer`, `test_cross_val_score_shape_mismatch`. Non-test consumer: REQ-X-2 — `cross_val_score(&pipeline, x, y, self.cv.as_ref(), self.scoring)?` in `GridSearchCV::fit` (`grid_search.rs`) and `RandomizedSearchCV::fit` (`random_search.rs`). |
| REQ-CVALIDATE (cross_validate test/train + timing) | SHIPPED | impl `pub fn cross_validate in cross_validation.rs` runs the REQ-CVS fold loop, wraps the fit in `Instant::now()`/`elapsed().as_secs_f64()` (`fit_times`) and the test predict+score in another (`score_times`), and when `return_train_score` is `true` additionally computes `fitted.predict(&x_train)?` → `scoring(&y_train, &y_train_pred)?` into `train_scores`. Returns `CrossValidateResult { test_scores, train_scores: Option<Vec<f64>>, fit_times, score_times }`, mirroring sklearn's dict `test_score`/`fit_time`/`score_time`/`train_score` (`sklearn/model_selection/_validation.py:122`). Tests: `test_cross_validate_returns_correct_fold_count` (test_scores/fit_times/score_times len 5, train_scores `None`), `test_cross_validate_with_train_scores` (train_scores `Some`, len 5), `test_cross_validate_timing_non_negative`, `test_cross_validate_perfect_constant_target`, `test_cross_validate_shape_mismatch`. Honest underclaim: `return_estimator`/`return_indices`/multimetric scoring are ABSENT (REQ-DEFAULTS, NOT-STARTED). Non-test consumer: REQ-X-2 (re-export `pub use cross_validation::CrossValidateResult, cross_validate in lib.rs`). |
| REQ-CVPREDICT (cross_val_predict original-order, partition case) | SHIPPED | impl `pub fn cross_val_predict in cross_validation.rs` shape-checks `y`, allocates `predictions = Array1::<f64>::zeros(n_samples)`, and per fold fits on train, predicts on test, and writes `predictions[idx] = y_pred[j]` for each `(j, idx)` in `test_idx` — placing each out-of-fold prediction at its ORIGINAL sample position. On a PARTITION cv (every sample in exactly one test fold) this matches sklearn's "each sample predicted once, returned in original order" (`sklearn/model_selection/_validation.py:1054`). Live oracle (AC-CVPREDICT): `cross_val_predict(LinearRegression(), X, y, cv=KFold(5))` has length 10. Tests: `test_cross_val_predict_length`, `test_cross_val_predict_constant_target` (constant y → constant preds at every position), `test_cross_val_predict_with_transformer`, `test_cross_val_predict_shape_mismatch`. The NON-PARTITION case diverges — REQ-CVPREDICT-PARTITION. Non-test consumer: REQ-X-2 (re-export). |
| REQ-CVPREDICT-PARTITION (non-partition behavior) | NOT-STARTED | open prereq blocker #1793. DIVERGENCE. impl `pub fn cross_val_predict in cross_validation.rs` initializes `predictions` to `Array1::zeros(n_samples)` and only OVERWRITES positions that appear in some `test_idx`; samples never in a test fold KEEP `0.0` and the call returns `Ok`. sklearn inverts the test indices and REQUIRES a partition, raising `ValueError("cross_val_predict only works for partitions")` (`sklearn/model_selection/_validation.py:1054`). Oracle (AC-CVPREDICT-PARTITION): `cross_val_predict(.., cv=ShuffleSplit(3,test_size=0.3))` raises in sklearn; ferrolearn silently returns `0.0`-filled gaps. The critic should pin a test that a non-partition cv yields `Err`. |
| REQ-PERM (permutation_test_score p-value formula) | SHIPPED | impl `pub fn permutation_test_score in cross_validation.rs` computes `real_score = cross_val_score(...)?.mean()`, then for `n_permutations` shuffles of `y` (via `SmallRng` + `indices.shuffle`) re-runs `cross_val_score` and pushes `fold_scores.mean()` into `perm_scores`, and finally `p_value = (n_ge as f64 + 1.0) / (n_permutations as f64 + 1.0)` where `n_ge = perm_scores.iter().filter(|&&s| s >= real_score).count()`. This is sklearn's `pvalue = (np.sum(permutation_scores >= score) + 1.0) / (n_permutations + 1)` (`sklearn/model_selection/_validation.py:1697`) and `score` = mean CV score (`:1677`) EXACTLY. Live oracle (AC-PERM): the formula `(count(perm>=score)+1)/(n+1)` reproduces sklearn's `pvalue` to `1e-12`. Tests: `test_permutation_test_score_returns_correct_counts`, `test_permutation_test_score_deterministic`, `test_permutation_test_score_p_value_range`. Honest underclaim: only the FORMULA + real-score mean is pinned; the EXACT `perm_scores` depend on the RNG (REQ-SHUFFLE-RNG carve-out). Non-test consumer: REQ-X-2 (re-export `pub use cross_validation::permutation_test_score in lib.rs`). |
| REQ-ERROR-SCORE (error_score=np.nan continue — KEY fixable, cross-unit OWNER) | NOT-STARTED | open prereq blocker #1790. DIVERGENCE. impl `pub fn cross_val_score`/`cross_validate`/`cross_val_predict in cross_validation.rs` each `?`-propagate `pipeline.fit(...)`, `fitted.predict(...)`, and `scoring(...)` — a single fold failure ABORTS the whole call with `Err`. sklearn `_fit_and_score` with the default `error_score=np.nan` NaN-fills BOTH train and test for the failing fold (independent `_score` calls), emits `FitFailedWarning`, and CONTINUES (`sklearn/model_selection/_validation.py:729`, `:890-905`, `:910`/`:915`). Oracle (AC-ERROR-SCORE): an estimator failing on the fold whose train rows include `X[:,0]==99` → `cross_val_score` returns `[-9.25, nan, nan]` (`np.isnan(...).any() == True`), NOT an error. DETERMINISTIC, fixable (same pattern as `validation_curve` #1758/#1762). This unit OWNS the blocker the curve/search units deferred here (S8 — `grid_search.rs`/`random_search.rs` REQ-ERROR-SCORE both cite this owner). The critic should pin a failing test: a one-fold-failing pipeline ⇒ `cross_val_score` returns scores with NaN in that fold, not `Err`. |
| REQ-SKFOLD-CV (StratifiedKFold has no CrossValidator impl) | NOT-STARTED | open prereq blocker #1794. ARCHITECTURAL. `StratifiedKFold` in `cross_validation.rs` exposes only the inherent `pub fn split(&self, y: &Array1<usize>) -> Result<FoldSplits, FerroError>` and has NO `impl CrossValidator for StratifiedKFold` — because the trait `pub trait CrossValidator { fn fold_indices(&self, n_samples: usize) -> ... }` takes ONLY `n_samples` (no y/groups channel). Consequently `&StratifiedKFold` is not `&dyn CrossValidator`, so it cannot be passed to `cross_val_score`/`cross_validate`/`cross_val_predict`/`GridSearchCV`/`RandomizedSearchCV` (all of which take `&dyn CrossValidator` / `Box<dyn CrossValidator>`). Oracle (AC-SKFOLD-CV): `cross_val_score(&pipeline, x, y, &StratifiedKFold::new(3), scoring)` does not type-check. Same channel-gap class as `group_splitters`/`StratifiedShuffleSplit` (REQ-SSS-3 in splitters.md). |
| REQ-SHUFFLE-RNG (shuffle + permutation EXACT membership — RNG carve-out) | NOT-STARTED | open prereq blocker #1795. R-DEFER-3 carve-out (blocker, NO failing test). The `shuffle` path of `KFold`/`StratifiedKFold` (`indices.shuffle(&mut rng)` / per-stratum `class_indices.get_mut(class).unwrap().shuffle(&mut rng)`) and `permutation_test_score`'s `indices.shuffle(&mut rng)` all use `rand::rngs::SmallRng`, whose stream differs from numpy's `RandomState`/`check_random_state`. So EXACT shuffled fold membership and the EXACT `perm_scores` sequence cannot match sklearn. Per R-DEFER-3 only the STRUCTURAL facts are pinned (and SHIPPED): shuffled folds remain a disjoint covering partition of `0..n` of the same sizes as the non-shuffled case, and `random_state(seed)` makes the stream reproducible ACROSS RUNS (`test_kfold_shuffle_deterministic`, `test_skfold_shuffle_deterministic`, `test_permutation_test_score_deterministic`). No failing test for the membership. |
| REQ-DEFAULTS (cv=None / scoring=None / groups / n_jobs / return_estimator / return_indices / multimetric) | NOT-STARTED | open prereq blocker #1796. API-shape + feature gaps. ferrolearn `cross_val_score`/`cross_validate`/`cross_val_predict`/`permutation_test_score in cross_validation.rs` take a MANDATORY `cv: &dyn CrossValidator` and (the scoring ones) a MANDATORY `scoring: fn(&Array1<f64>, &Array1<f64>) -> Result<f64, FerroError>`, and have NO `groups`/`n_jobs`/`fit_params`/`pre_dispatch` channel. sklearn defaults `cv=None` → 5-fold classifier-aware `check_cv`, `scoring=None` → `check_scoring` (`sklearn/model_selection/_validation.py:560`, `:122`), threads `groups` to `cv.split(X, y, groups)`, and `cross_validate` additionally supports `return_estimator`/`return_indices`/multimetric scoring (`:122`). `CrossValidateResult` carries none of those. The no-cv / no-scoring / multimetric forms are unrepresentable. |
| REQ-X-1 (R-SUBSTRATE) | NOT-STARTED | open prereq blocker #1797. Production code in `cross_validation.rs` imports `use ndarray::{Array1, Array2}` (array type), `use rand::{SeedableRng, rngs::SmallRng, seq::SliceRandom}` (random substrate), and `std::collections::HashMap`, and builds `Array2::from_shape_vec(...)` / `Array1` subsets per fold. Per R-SUBSTRATE-1 the destination is `ferray-core` (array) and `ferray::random` (sampling/shuffle); `ndarray`/`rand` are the wrong substrate. Until migrated this unit is not on the ferray substrate (R-SUBSTRATE-2). |
| REQ-X-2 (non-test production consumer — the WIDEST in the crate) | SHIPPED | `pub fn cross_val_score in cross_validation.rs` is called in production by FOUR internal consumers: `GridSearchCV::fit` (`grid_search.rs` — `let scores = cross_val_score(&pipeline, x, y, self.cv.as_ref(), self.scoring)?`), `RandomizedSearchCV::fit` (`random_search.rs` — same call), and both halving searches (`halving_grid_search.rs`, `halving_random_search.rs` — `cross_val_score(&pipeline, &x_sub, &y_sub, self.cv.as_ref(), self.scoring)`). `KFold` flows into all of them as `&dyn CrossValidator` (each takes `Box<dyn CrossValidator>`/`&dyn CrossValidator`; their tests/examples construct `Box::new(KFold::new(k))`). The whole surface — `CrossValidateResult, CrossValidator, KFold, StratifiedKFold, cross_val_predict, cross_val_score, cross_validate, permutation_test_score` — is re-exported at `pub use cross_validation::{...} in lib.rs`. This is the widest-consumed unit in `ferrolearn-model-sel`. Honest underclaim: `StratifiedKFold` reaches production only via the inherent `split(&Array1<usize>)` + re-export (no `&dyn CrossValidator` path — REQ-SKFOLD-CV); no `ferrolearn-python` binding exists. |

## Architecture

`cross_validation.rs` is two layers: the **splitters** that produce
`FoldSplits = Vec<(Vec<usize>, Vec<usize>)>` and the **drivers** that consume a
`&dyn CrossValidator` to fit/predict/score per fold.

**The `CrossValidator` trait** (`pub trait CrossValidator { fn
fold_indices(&self, n_samples: usize) -> Result<FoldSplits, FerroError>; }`) is
ferrolearn's `BaseCrossValidator` analog, but its single channel is
`n_samples` — there is NO y/groups channel. This is the architectural fault line:

- `KFold` implements it (`impl CrossValidator for KFold`), so it is consumed
  polymorphically by every driver and by `GridSearchCV`/`RandomizedSearchCV`/the
  halving searches (REQ-X-2).
- `StratifiedKFold` CANNOT implement it (it needs `y`), so it exposes an inherent
  `split(&Array1<usize>)` and is unreachable as `&dyn CrossValidator` (REQ-SKFOLD-CV,
  #1794) — the same channel-gap as `StratifiedShuffleSplit`/`group_splitters`.

**KFold** (`fn split_result`) is DETERMINISTIC and oracle-pinnable end to end:
`base_size = n/k`, `remainder = n%k`, the first `remainder` folds get `+1`, and
folds are consecutive slices of `indices` — element-for-element sklearn's
`fold_sizes` + `indices[start:stop]` (`:521-535`). The shuffle path (`SmallRng`)
is structurally a partition of the same sizes; its EXACT membership is the
REQ-SHUFFLE-RNG carve-out (#1795).

**StratifiedKFold** (`fn split`) DIVERGES deterministically in two ways. (1)
Allocation (REQ-SKFOLD, #1791): ferrolearn sorts classes LEXICOGRAPHICALLY and
uses a rotating-`fold_offset` round-robin (the `extra` folds starting at
`fold_offset` get `+1`, accumulating `extra % k` across classes), whereas sklearn
encodes classes by APPEARANCE order (`:760-765`) and allocates via
`bincount(y_order[i::n_splits])` (`:786-792`). They coincide when appearance
order equals lexicographic order, but on `y=[2,2,2,2,0,0,0,0,1,1,1,1]` (k=3) they
differ: sklearn `[[0,1,4,8],[2,5,6,9],[3,7,10,11]]` vs ferrolearn
`[[0,4,5,8],[1,6,9,10],[2,3,7,11]]`. (2) Error-vs-warn (REQ-SKFOLD-ERRWARN,
#1792): ferrolearn errors if ANY class < `n_splits`; sklearn errors only if ALL
classes are too small (`:770-774`) and WARNS otherwise (`:775-781`).

**The drivers** (`cross_val_score`, `cross_validate`, `cross_val_predict`,
`permutation_test_score`) all share a per-fold subset-gather (`x.row(i)` copied
into fresh `Array2`s, `y[i]` collected into `Array1`s) + `pipeline.fit` +
`predict`. The non-failing mechanics MATCH sklearn (REQ-CVS, REQ-CVALIDATE,
REQ-CVPREDICT, REQ-PERM). `cross_validate` adds `Instant`-based `fit_time`/
`score_time` and optional train scoring. `cross_val_predict` writes each
out-of-fold prediction at its original index. `permutation_test_score` reuses
`cross_val_score` for both the real and the `n_permutations` shuffled-`y` runs and
applies the sklearn p-value formula `(n_ge + 1)/(n_permutations + 1)` (`:1697`)
EXACTLY.

The shared driver flaw is **error_score** (REQ-ERROR-SCORE, #1790 — the KEY
fixable divergence): every `pipeline.fit(...)?` / `predict(...)?` / `scoring(...)?`
PROPAGATES, so one failing fold aborts the whole call, where sklearn's default
`error_score=np.nan` NaN-fills the fold (both train+test, independent scoring) and
continues (`_fit_and_score`, `:729`, `:890-905`). This is the SAME contract pinned
for `validation_curve` REQ-7 (#1758/#1762), and `grid_search.rs`/`random_search.rs`
already cite this unit as the OWNER (S8). The fix lands here.

Remaining gaps (all NOT-STARTED): the mandatory `cv`/`scoring`, the absent
`groups`/`n_jobs`/`fit_params`/`return_estimator`/`return_indices`/multimetric
surface (REQ-DEFAULTS, #1796), and the ferray substrate migration (REQ-X-1,
#1797).

## Verification

Commands establishing the SHIPPED claims (baseline
`699b4835d444cfa3c0d2ea5631b7b840bdb60f89`):

- `cargo test -p ferrolearn-model-sel --lib cross_validation` → 30 passed, 0
  failed (`cross_validation::tests::{test_kfold_*, test_skfold_*,
  test_cross_val_score_*, test_cross_validate_*, test_cross_val_predict_*,
  test_permutation_test_score_*}`).
- REQ-KFOLD DETERMINISTIC oracle (live sklearn 1.5.2):
  ```
  python3 -c "import numpy as np
  from sklearn.model_selection import KFold
  print([list(te) for tr,te in KFold(3,shuffle=False).split(np.zeros((10,1)))])"
  # -> [[0,1,2,3],[4,5,6],[7,8,9]]   matches ferrolearn KFold::new(3).split(10)
  ```
- REQ-SKFOLD DIVERGENCE oracle (the allocation pin, #1791):
  ```
  python3 -c "import numpy as np
  from sklearn.model_selection import StratifiedKFold
  y=np.array([2,2,2,2,0,0,0,0,1,1,1,1])
  print([sorted(int(i) for i in te) for tr,te in StratifiedKFold(3,shuffle=False).split(np.zeros((12,1)), y)])"
  # sklearn -> [[0,1,4,8],[2,5,6,9],[3,7,10,11]]
  # ferrolearn StratifiedKFold::new(3).split(&y) -> [[0,4,5,8],[1,6,9,10],[2,3,7,11]]  DIVERGENCE
  ```
- REQ-SKFOLD-ERRWARN DIVERGENCE oracle (#1792):
  ```
  python3 -c "import numpy as np, warnings
  from sklearn.model_selection import StratifiedKFold
  y=np.array([0,0,0,0,0,1,1])
  with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter('always')
      list(StratifiedKFold(3,shuffle=False).split(np.zeros((7,1)), y))
      print('SUCCEEDS', [str(x.message)[:30] for x in w])"
  # -> SUCCEEDS ['The least populated class in y']   ferrolearn errors
  ```
- REQ-CVS / REQ-CVALIDATE / REQ-CVPREDICT mechanics: the `MeanEstimator`
  in-module tests pin per-fold count, constant-target correctness, timing
  non-negativity, and original-order placement (listed above).
- REQ-CVPREDICT-PARTITION DIVERGENCE oracle (#1793):
  ```
  python3 -c "import numpy as np
  from sklearn.model_selection import cross_val_predict, ShuffleSplit
  from sklearn.linear_model import LinearRegression
  X=np.arange(20).reshape(10,2).astype(float); y=np.arange(10).astype(float)
  try: cross_val_predict(LinearRegression(),X,y,cv=ShuffleSplit(3,test_size=0.3,random_state=0))
  except Exception as e: print(type(e).__name__, str(e)[:50])"
  # -> ValueError cross_val_predict only works for partitions   ferrolearn returns 0.0-filled gaps
  ```
- REQ-PERM p-value oracle (live sklearn — the formula pin, R-CHAR-3):
  ```
  python3 -c "import numpy as np
  from sklearn.model_selection import permutation_test_score, KFold
  from sklearn.linear_model import LinearRegression
  X=np.random.RandomState(0).randn(30,3); y=np.random.RandomState(1).randn(30)
  s,perm,p=permutation_test_score(LinearRegression(),X,y,cv=KFold(3),
      n_permutations=20,random_state=0,scoring='neg_mean_squared_error')
  manual=(np.sum(np.array(perm)>=s)+1.0)/(20+1.0)
  print(round(p,6),round(manual,6),abs(p-manual)<1e-12)"   # 0.809524 0.809524 True
  ```
  ferrolearn `p_value = (n_ge + 1.0)/(n_permutations + 1.0)` is the identical
  formula.
- REQ-ERROR-SCORE DIVERGENCE oracle (the KEY pin, #1790):
  ```
  python3 -c "import numpy as np, warnings
  from sklearn.model_selection import cross_val_score, KFold
  from sklearn.base import BaseEstimator, RegressorMixin
  class G(BaseEstimator, RegressorMixin):
      def fit(self,X,y):
          if np.any(X[:,0]==99): raise ValueError('boom')
          self.m_=y.mean(); return self
      def predict(self,X): return np.full(X.shape[0], self.m_)
  X=np.arange(12).reshape(6,2).astype(float); X[0,0]=99.0
  y=np.arange(6).astype(float)
  with warnings.catch_warnings(): warnings.simplefilter('ignore')
  s=cross_val_score(G(),X,y,cv=KFold(3),scoring='neg_mean_squared_error')
  print(s.tolist(),'any nan:',bool(np.isnan(s).any()))"
  # -> [-9.25, nan, nan] any nan: True   ferrolearn returns Err (#1790)
  ```
- REQ-X-2 consumer: `grep -rn "cross_val_score(" ferrolearn-model-sel/src/*.rs |
  grep -v 'mod tests\|fn cross_val_score'` shows the production call sites in
  `grid_search.rs`, `random_search.rs`, `halving_grid_search.rs`,
  `halving_random_search.rs` (`cross_val_score(&pipeline, .., self.cv.as_ref(),
  self.scoring)`). `grep -n "pub use cross_validation" lib.rs` shows the full
  re-export.
- REQ-X-1 substrate: `grep -n "ndarray\|rand\|SmallRng" cross_validation.rs`
  shows production `use ndarray::{Array1, Array2}` and `use rand::{SeedableRng,
  rngs::SmallRng, seq::SliceRandom}` — wrong substrate, migration owed (#1797).

SHIPPED: REQ-KFOLD (DETERMINISTIC), REQ-CVS, REQ-CVALIDATE, REQ-CVPREDICT
(partition case), REQ-PERM (p-value formula), REQ-X-2 (the widest consumer).
NOT-STARTED: REQ-SKFOLD (allocation divergence, #1791), REQ-SKFOLD-ERRWARN
(error-vs-warn, #1792), REQ-CVPREDICT-PARTITION (non-partition, #1793),
REQ-ERROR-SCORE (error_score continue, #1790 — KEY fixable, cross-unit owner),
REQ-SKFOLD-CV (StratifiedKFold channel gap, #1794), REQ-SHUFFLE-RNG (RNG
carve-out, #1795), REQ-DEFAULTS (mandatory cv/scoring + missing
groups/n_jobs/return_estimator/return_indices/multimetric, #1796), REQ-X-1
(ferray substrate, #1797).

Per R-DEFER-2 every REQ is binary SHIPPED/NOT-STARTED. The DETERMINISTIC FIXABLE
divergences the critic should pin as FAILING tests: **REQ-ERROR-SCORE**
(error_score-continue, #1790), **REQ-SKFOLD** (StratifiedKFold allocation,
#1791), **REQ-SKFOLD-ERRWARN** (StratifiedKFold error-vs-warn, #1792), and
**REQ-CVPREDICT-PARTITION** (cross_val_predict non-partition, #1793). Per
R-DEFER-3 the RNG-membership carve-out (REQ-SHUFFLE-RNG, #1795) gets a blocker but
NO failing test. The architectural gaps (REQ-SKFOLD-CV channel, REQ-DEFAULTS,
REQ-X-1) are NOT-STARTED with blockers, not pins.

Least-confident SHIPPED claim: REQ-CVPREDICT — it is SHIPPED only for the
PARTITION case; the moment a non-partition cv is used (which the type system
permits, e.g. a future `ShuffleSplit` as `&dyn CrossValidator`) the `0.0`-fill
silently diverges from sklearn's hard `ValueError` (REQ-CVPREDICT-PARTITION,
#1793), so the "shipped" surface is the partition-cv subset, not the full
`cross_val_predict` contract.
