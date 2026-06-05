# learning_curve

<!--
tier: 3-component
status: draft
baseline-commit: b16c37a0ad3ac8d593207810314700423fe64173
upstream-paths:
  - sklearn/model_selection/_validation.py   # learning_curve (:1724), _translate_train_sizes (:1992), _fit_and_score error_score (:890)
-->

## Summary

`ferrolearn-model-sel/src/learning_curve.rs` mirrors scikit-learn's
`learning_curve` (`sklearn/model_selection/_validation.py:1724`) — the utility
that, for a sequence of increasing training-set sizes, runs cross-validation and
reports per-fold TRAIN and TEST scores, yielding two `(n_ticks, n_cv_folds)`
matrices plus the absolute training sizes used, for diagnosing whether more data
would improve the model (bias/variance).

ferrolearn exposes `pub fn learning_curve(pipeline: &Pipeline, x: &Array2<f64>,
y: &Array1<f64>, cv: &dyn CrossValidator, train_sizes: &[f64], scoring: fn(...) ->
Result<f64, FerroError>) -> Result<LearningCurveResult, FerroError>`. The result
struct `LearningCurveResult { train_sizes: Vec<usize>, train_scores:
Array2<f64> (n_sizes, n_folds), test_scores: Array2<f64> (n_sizes, n_folds) }`
matches sklearn's returned `(train_sizes_abs, train_scores, test_scores)` shape
and orientation.

The CORE MECHANIC is faithful: ferrolearn fills the `(n_sizes, n_folds)` matrices
in `(size OUTER, fold INNER)` row-major order, element-for-element IDENTICAL to
sklearn's `(fold OUTER, size INNER)` flat list `.reshape(-1, n_unique_ticks).T`
(verified below). The training subset at a tick is the FIRST `n_train` indices of
the fold's train indices in BOTH (`train[:n_train_samples]` sklearn,
`&train_idx[..effective_size]` ferrolearn). ferrolearn always returns train
scores, matching sklearn's hardcoded `return_train_score=True`.

The SIZE-TRANSLATION layer (sklearn's `_translate_train_sizes`, `:1992`) diverges
in five deterministic, oracle-pinnable ways — these are MATCH-class divergences
(fixable bugs, not deviations): fraction→absolute rounding (sklearn floors via
`.astype(int)`, ferrolearn `.ceil()`s — REQ-A); float-array-entry `> 1.0`
handling (sklearn `ValueError`, ferrolearn silently treats as absolute — REQ-B);
absolute-overshoot handling (sklearn `ValueError`, ferrolearn clamps — REQ-C);
sort+dedup via `np.unique` (sklearn sorts ascending and de-duplicates,
ferrolearn preserves input order and keeps duplicates — REQ-D); and the
`error_score=np.nan` continue-the-curve contract (sklearn nan-fills a failing
cell and continues, ferrolearn `?`-aborts — REQ-E, the same contract just SHIPPED
for `validation_curve` REQ-7). Several whole features are absent and NOT-STARTED:
default `train_sizes`/`cv`/`scoring`, `shuffle`+`random_state`, the `groups`
channel, and `return_times`/`exploit_incremental_learning`/`fit_params`.

## Upstream reference (scikit-learn 1.5.2, commit 156ef14)

### `learning_curve` (`sklearn/model_selection/_validation.py:1724`)
- `:1724-1742` — signature `learning_curve(estimator, X, y, *, groups=None,
  train_sizes=np.linspace(0.1, 1.0, 5), cv=None, scoring=None,
  exploit_incremental_learning=False, n_jobs=None, pre_dispatch="all", verbose=0,
  shuffle=False, random_state=None, error_score=np.nan, return_times=False,
  fit_params=None)`. All params after `y` are keyword-only.
- `:1730` — `train_sizes` default `np.linspace(0.1, 1.0, 5)` (= `[0.1, 0.325,
  0.55, 0.775, 1.0]`).
- `:1771-1774` — `groups` "Only used in conjunction with a 'Group' cv instance".
- `:1786-1798` — `cv` default `None` → 5-fold; classifier with binary/multiclass
  `y` → `StratifiedKFold`, else `KFold`, always `shuffle=False`.
- `:1806-1809` — `scoring` default `None`, resolved by `check_scoring` (`:1912`).
- `:1830-1837` — `shuffle` / `random_state`: when `shuffle=True`, train indices
  are permuted before prefixes are taken.
- `:1839-1842` — `error_score : 'raise' or numeric, default=np.nan`.
- `:1854-1873` — Returns `train_sizes_abs : (n_unique_ticks,)`, `train_scores :
  (n_ticks, n_cv_folds)`, `test_scores : (n_ticks, n_cv_folds)` (+ `fit_times`,
  `score_times` if `return_times`). "the number of ticks might be less than
  n_ticks because duplicate entries will be removed".
- `:1906` — `X, y, groups = indexable(X, y, groups)`.
- `:1908` — `cv = check_cv(cv, y, classifier=is_classifier(estimator))`.
- `:1910` — `cv_iter = list(cv.split(X, y, groups))`.
- `:1912` — `scorer = check_scoring(estimator, scoring=scoring)`.
- `:1914` — `n_max_training_samples = len(cv_iter[0][0])` — the FIRST fold's train
  length is the reference for fraction→absolute translation.
- `:1918` — `train_sizes_abs = _translate_train_sizes(train_sizes,
  n_max_training_samples)`.
- `:1919` — `n_unique_ticks = train_sizes_abs.shape[0]`.
- `:1925-1927` — if `shuffle`: `cv_iter = ((rng.permutation(train), test) for
  train, test in cv_iter)`.
- `:1949-1952` — the non-incremental path builds `train_test_proportions = for
  train, test in cv_iter: for n_train_samples in train_sizes_abs:
  (train[:n_train_samples], test)` — **fold OUTER, size INNER**; each train subset
  is the FIRST `n_train_samples` of the fold's train indices.
- `:1954-1972` — `_fit_and_score(clone(estimator), ..., parameters=None,
  return_train_score=True, error_score=error_score)` per `(train_subset, test)`.
- `:1973` — `_warn_or_raise_about_fit_failures(results, error_score)`.
- `:1975-1976` — `train_scores = results["train_scores"].reshape(-1,
  n_unique_ticks).T; test_scores = ... .T` → shape `(n_unique_ticks, n_cv_folds)`.
- `:1984` — `return train_sizes_abs, out[0], out[1]` (+ times if `return_times`).

### `_translate_train_sizes` (`sklearn/model_selection/_validation.py:1992`)
- `:2016` — `train_sizes_abs = np.asarray(train_sizes)`.
- `:2020` — **dtype-based mode**: `if np.issubdtype(train_sizes_abs.dtype,
  np.floating)` — the WHOLE array's dtype decides fraction-vs-absolute, not
  per-element.
- `:2021-2027` — float mode: raise `ValueError` if `min <= 0.0 or max > 1.0`.
- `:2028-2030` — `train_sizes_abs = (train_sizes_abs *
  n_max_training_samples).astype(int)` — TRUNCATION toward zero (floor for
  positives).
- `:2031` — `np.clip(train_sizes_abs, 1, n_max_training_samples)`.
- `:2033-2046` — integer mode: raise `ValueError` if `min <= 0 or max >
  n_max_training_samples` (NO clamp — overshoot is an error).
- `:2048` — `train_sizes_abs = np.unique(train_sizes_abs)` — SORTS ascending AND
  removes duplicates.
- `:2049-2055` — `RuntimeWarning` if ticks were removed.

### `_fit_and_score` error_score path (`sklearn/model_selection/_validation.py:890`)
- `:890-905` — on a fit failure with numeric `error_score` (default np.nan), BOTH
  `test_scores` and (when `return_train_score`) `train_scores` are set to
  `error_score`; the loop CONTINUES rather than aborting.
- `:910`/`:915` — independent `_score` of test then train on success. This is the
  SAME contract just SHIPPED for `validation_curve` REQ-7 (see
  `.design/model-sel/validation_curve.md`).

## Requirements

The R-DEV mental test is applied to each REQ ("why did sklearn choose this?
numerical/API contract → MATCH; Cython/CPython footgun → DEVIATE"). REQ-A..E are
MATCH-class divergences (sklearn's behavior IS the contract; ferrolearn computes
something observably different) and are deterministic single-spot fixes — they are
NOT deviations.

- REQ-1 (core per-(size,fold) fit+score mechanic + iteration-order/orientation
  equivalence — the KEY REQ): for each `(size, fold)` pair, take the FIRST
  `n_train` of the fold's train indices, fit a fresh-fit pipeline on that subset,
  and score on BOTH the train subset and the FULL test fold, producing
  `(n_sizes, n_folds)` train and test matrices. ferrolearn fills `(size OUTER,
  fold INNER)` row-major; sklearn builds a `(fold OUTER, size INNER)` flat list
  then `.reshape(-1, n_unique_ticks).T`. These yield element `[size i][fold j]`
  IDENTICALLY (proven below). The train subset is the FIRST `n_train` indices in
  BOTH (`train[:n_train_samples]` `:1952` vs `&train_idx[..effective_size]`).
  **MATCH** (R-DEV-1 numerical/structural contract): the observable
  `(n_ticks, n_cv_folds)` matrices and their element ordering are the contract;
  the loop nesting is an internal re-indexing producing the same array. SHIPPED.
- REQ-2 (train scores always on; shape/orientation `(n_ticks, n_cv_folds)`):
  ferrolearn unconditionally computes and returns `train_scores`, matching
  sklearn's hardcoded `return_train_score=True` (`:1967`); both matrices are
  `(n_sizes, n_folds)` — row=size tick, col=fold — matching sklearn's
  post-transpose `(n_unique_ticks, n_cv_folds)` (`:1861-1864`, `:1975-1976`).
  **MATCH** (R-DEV-3 output-object contract). SHIPPED.
- REQ-A (fraction→absolute rounding: FLOOR vs CEIL): sklearn `(train_sizes_abs *
  n_max).astype(int)` (`:2028-2030`) TRUNCATES toward zero (floor for positives);
  ferrolearn uses `(s * reference_train_len).ceil()` in `pub fn learning_curve`.
  DIVERGENCE — e.g. `s=0.33, n_max=20`: sklearn `int(6.6)=6`, ferrolearn
  `ceil(6.6)=7`. **MATCH** (R-DEV-1 — the absolute tick count is a numerical
  contract; users compare `train_sizes_abs` array-by-array). Deterministic
  single-line fixable divergence. NOT-STARTED — open prereq blocker #1764.
- REQ-B (float-array entry `> 1.0` → ValueError vs treat-as-absolute): sklearn
  decides fraction-vs-absolute by ARRAY DTYPE (`:2020`) — an all-float array is
  fraction mode and EVERY entry must be `<= 1.0` else `ValueError`
  (`:2021-2027`); ferrolearn decides PER-ELEMENT (`if s <= 1.0` in `pub fn
  learning_curve`), so a float `2.0` is silently treated as an absolute count of
  2. DIVERGENCE — sklearn `learning_curve(train_sizes=[0.5, 2.0])` raises
  `ValueError`; ferrolearn returns size 2. **MATCH** (R-DEV-2 — exception-type /
  input-validation contract). NOT-STARTED — open prereq blocker #1765.
- REQ-C (absolute `> n_max` → ValueError vs clamp): sklearn's integer-dtype path
  raises `ValueError` if `max > n_max_training_samples` (`:2033-2046`, NO clamp);
  ferrolearn clamps via `.min(reference_train_len)` in `pub fn learning_curve`.
  DIVERGENCE — sklearn raises, ferrolearn silently clamps. **MATCH** (R-DEV-2 —
  validation contract). NOT-STARTED — open prereq blocker #1766.
- REQ-D (dedup + sort via np.unique): sklearn `np.unique(train_sizes_abs)`
  (`:2048`) SORTS ascending AND removes duplicates (RuntimeWarning if removed),
  yielding `(n_unique_ticks,)` rows; ferrolearn preserves input order and keeps
  duplicates in `abs_sizes`. DIVERGENCE — `train_sizes=[1.0, 0.5]` → sklearn
  `[10, 20]` (sorted), ferrolearn `[20, 10]`; `train_sizes=[0.5, 0.5]` → sklearn
  `[10]` (1 tick), ferrolearn `[10, 10]` (2 ticks → an extra matrix row).
  **MATCH** (R-DEV-1/3 — both the `train_sizes_abs` vector and the matrix ROW
  COUNT are observable contract). NOT-STARTED — open prereq blocker #1767.
- REQ-E (error_score=np.nan continue-the-curve): on a per-`(size, fold)` fit/score
  failure sklearn assigns `error_score` (default np.nan) to that cell and
  CONTINUES the curve (`:1968`, `_fit_and_score` `:890-905`); ferrolearn
  PROPAGATES the failure via `pipeline.fit(...)?` / `scoring(...)?` in `pub fn
  learning_curve` and ABORTS the whole call. DIVERGENCE — identical contract to
  the just-SHIPPED `validation_curve` REQ-7. **MATCH** (R-DEV-1 — partial
  nan-bearing curve is the contract). NOT-STARTED — open prereq blocker #1768.
- REQ-DEFAULTS (default train_sizes / cv / scoring): sklearn defaults
  `train_sizes=np.linspace(0.1, 1.0, 5)` (`:1730`), `cv=None` → 5-fold
  classifier-aware StratifiedKFold/KFold via `check_cv` (`:1908`), `scoring=None`
  → r2/accuracy via `check_scoring` (`:1912`). ferrolearn requires explicit
  `train_sizes: &[f64]`, `cv: &dyn CrossValidator`, and `scoring: fn(...)` — no
  defaults, no `is_classifier` dispatch. **MATCH-intent / gap** (same as
  validation_curve REQ-5/6). NOT-STARTED — open prereq blocker #1769.
- REQ-SHUFFLE (shuffle + random_state prefix): sklearn `shuffle=True` permutes
  each fold's train indices before taking prefixes (`:1925-1927`); ferrolearn
  always takes the first `effective_size` of `train_idx` (no shuffle, no RNG).
  **MATCH-intent / gap** (RNG-coupled — needs `random_state` determinism).
  NOT-STARTED — open prereq blocker #1770.
- REQ-GROUPS (groups channel for Group cv): sklearn threads `groups` to
  `cv.split(X, y, groups)` (`:1910`); ferrolearn calls
  `cv.fold_indices(n_samples)` — `CrossValidator::fold_indices(n_samples) in
  cross_validation.rs` has NO group/label channel, so Group cv is unreachable
  (same channel-gap as validation_curve REQ-8). **MATCH-intent / gap**.
  NOT-STARTED — open prereq blocker #1771.
- REQ-EXTRAS (return_times, exploit_incremental_learning, fit_params): sklearn's
  `return_times` adds two `(n_ticks, n_cv_folds)` timing matrices (`:1979-1982`),
  `exploit_incremental_learning` uses `partial_fit` (`:1929-1947`), and
  `fit_params` are forwarded to `fit`. ferrolearn has none of these.
  **MATCH-intent / gap**. NOT-STARTED — open prereq blocker #1772.
- REQ-X-1 (R-SUBSTRATE): production code imports `ndarray::{Array1, Array2}` (the
  array type) and builds `Array2::from_shape_vec(...)`; the destination substrate
  is `ferray-core` (R-SUBSTRATE-1). NOT-STARTED — open prereq blocker #1773.
- REQ-X-2 (non-test production consumer): the crate re-export boundary
  (`pub use learning_curve::{LearningCurveResult, learning_curve} in lib.rs`,
  S5/R-DEFER-1 grandfathering). SHIPPED.

## Acceptance criteria

Each AC is pinnable against a LIVE sklearn 1.5.2 call (R-CHAR-3 — expected values
come from the oracle, never copied from the ferrolearn side). The
`_translate_train_sizes` ones call the helper directly; the end-to-end ones use
`learning_curve`.

- AC-1 (REQ-1, ordering/orientation equivalence — the KEY pin): the `(size OUTER,
  fold INNER)` row-major fill equals sklearn's `(fold OUTER, size INNER)`
  `.reshape(-1, n_unique_ticks).T`. Oracle (pure-numpy reproduction of sklearn's
  reshape algebra, synthetic per-`(fold, size)` value `f*10 + s` isolating the
  index math):
  ```
  python3 -c "import numpy as np
  n_folds=3; n_ticks=4
  flat = np.array([f*10+s for f in range(n_folds) for s in range(n_ticks)])  # sklearn fold-major
  M = flat.reshape(-1, n_ticks).T                                            # sklearn (n_ticks, n_folds)
  F = np.array([[f*10+s for f in range(n_folds)] for s in range(n_ticks)])   # ferrolearn size-outer fill
  print('IDENTICAL:', np.array_equal(M, F))"   # -> IDENTICAL: True
  ```
  STRUCTURAL: ferrolearn `result.train_scores[[i, j]]` (size `i`, fold `j`) equals
  sklearn `train_scores[i, j]`; the FIRST-`n_train`-indices subset matches
  (`train[:n] == &train_idx[..n]`).
- AC-2 (REQ-2, shape + orientation): live oracle
  ```
  python3 -c "import numpy as np
  from sklearn.model_selection import learning_curve, KFold
  from sklearn.tree import DecisionTreeRegressor
  X=np.arange(60).reshape(30,2).astype(float); y=np.arange(30).astype(float)
  ts,tr,te=learning_curve(DecisionTreeRegressor(random_state=0), X, y,
      train_sizes=[5,10,20], cv=KFold(3), scoring='neg_mean_squared_error')
  print(tr.shape, te.shape)"   # -> (3, 3) (3, 3)  == (n_ticks=3, n_folds=3)
  ```
  ferrolearn `learning_curve(..., &KFold::new(3), &[5.0,10.0,20.0], ..)` returns
  `train_scores.shape() == [3, 3]` and `test_scores.shape() == [3, 3]`
  (`test_learning_curve_basic`, `test_learning_curve_absolute_sizes`).
- AC-A (REQ-A, FLOOR-vs-CEIL DIVERGENCE): sklearn truncates toward zero:
  ```
  python3 -c "from sklearn.model_selection._validation import _translate_train_sizes
  import numpy as np
  print(_translate_train_sizes(np.array([0.33]), 20).tolist())"   # -> [6]
  ```
  ferrolearn `(0.33 * 20).ceil() = ceil(6.6) = 7`. DIVERGENCE. NOT-STARTED
  (blocker #1764).
- AC-B (REQ-B, float `> 1.0` → ValueError DIVERGENCE): sklearn raises on an
  all-float array whose max exceeds 1.0:
  ```
  python3 -c "from sklearn.model_selection._validation import _translate_train_sizes
  import numpy as np
  try: _translate_train_sizes(np.array([0.5, 2.0]), 20); print('no error')
  except ValueError as e: print('ValueError:', str(e).split(',')[0])"
  # -> ValueError: train_sizes has been interpreted as fractions ...
  ```
  ferrolearn returns a size-2 tick for the `2.0` entry. DIVERGENCE. NOT-STARTED
  (blocker #1765).
- AC-C (REQ-C, absolute overshoot → ValueError DIVERGENCE): sklearn's integer
  path raises if `max > n_max`:
  ```
  python3 -c "from sklearn.model_selection._validation import _translate_train_sizes
  import numpy as np
  try: _translate_train_sizes(np.array([5, 25]), 20); print('no error')
  except ValueError as e: print('ValueError:', str(e).split(',')[0])"
  # -> ValueError: train_sizes has been interpreted as absolute ...
  ```
  ferrolearn clamps `25 -> 20` via `.min(reference_train_len)`. DIVERGENCE.
  NOT-STARTED (blocker #1766).
- AC-D (REQ-D, sort + dedup DIVERGENCE): sklearn `np.unique` sorts and dedups:
  ```
  python3 -c "from sklearn.model_selection._validation import _translate_train_sizes
  import numpy as np, warnings; warnings.simplefilter('ignore')
  print('sort:', _translate_train_sizes(np.array([1.0, 0.5]), 20).tolist())   # [10, 20]
  print('dedup:', _translate_train_sizes(np.array([0.5, 0.5]), 20).tolist())" # [10]
  ```
  ferrolearn yields `[20, 10]` (input order, 2 ticks) and `[10, 10]` (2 ticks —
  an extra matrix row). DIVERGENCE on BOTH the `train_sizes` vector AND the row
  count. NOT-STARTED (blocker #1767).
- AC-E (REQ-E, error_score=np.nan DIVERGENCE): when fitting fails for ONE size,
  sklearn fills that cell with nan and CONTINUES (default `error_score=np.nan`):
  ```
  python3 -c "import numpy as np, warnings
  from sklearn.model_selection import learning_curve, KFold
  from sklearn.base import BaseEstimator, RegressorMixin
  class F(BaseEstimator, RegressorMixin):
      def fit(self, X, y):
          if X.shape[0] == 5: raise ValueError('boom')
          self.m_ = y.mean(); return self
      def predict(self, X): return np.full(X.shape[0], self.m_)
  X=np.arange(60).reshape(30,2).astype(float); y=np.arange(30).astype(float)
  with warnings.catch_warnings(): warnings.simplefilter('ignore')
  ts,tr,te=learning_curve(F(), X, y, train_sizes=[5,10,20], cv=KFold(3),
      scoring='neg_mean_squared_error')
  print('any nan:', np.isnan(te).any())"   # -> any nan: True
  ```
  ferrolearn `pipeline.fit(...)?` / `scoring(...)?` PROPAGATES the error → the
  whole `learning_curve` returns `Err`, no partial curve. DIVERGENCE. NOT-STARTED
  (blocker #1768).
- AC-DEFAULTS (REQ-DEFAULTS, default train_sizes/cv/scoring DIVERGENCE):
  ```
  python3 -c "import numpy as np
  print('default train_sizes:', np.linspace(0.1, 1.0, 5).tolist())
  from sklearn.model_selection import check_cv
  print('default cv:', type(check_cv(None, np.array([0,1,0,1]), classifier=True)).__name__,
        check_cv(None, np.array([0,1,0,1]), classifier=True).get_n_splits())
  from sklearn.metrics import check_scoring
  from sklearn.tree import DecisionTreeRegressor
  print('default scoring:', check_scoring(DecisionTreeRegressor(), scoring=None)._score_func.__name__)"
  # default train_sizes: [0.1, 0.325, 0.55, 0.775, 1.0]
  # default cv: StratifiedKFold 5
  # default scoring: r2_score
  ```
  ferrolearn has no no-arg default form — all three are mandatory. DIVERGENCE.
  NOT-STARTED (blocker #1769).
- AC-SHUFFLE (REQ-SHUFFLE, shuffle DIVERGENCE): sklearn `shuffle=True,
  random_state=0` permutes train indices before taking prefixes, so the size-5
  subset is NOT the first 5 fold indices:
  ```
  python3 -c "import numpy as np
  from sklearn.model_selection import learning_curve, KFold
  from sklearn.dummy import DummyRegressor
  X=np.arange(60).reshape(30,2).astype(float); y=np.arange(30).astype(float)
  a=learning_curve(DummyRegressor(), X, y, train_sizes=[5,10], cv=KFold(3),
      shuffle=True, random_state=0, scoring='neg_mean_squared_error')[1]
  b=learning_curve(DummyRegressor(), X, y, train_sizes=[5,10], cv=KFold(3),
      shuffle=False, scoring='neg_mean_squared_error')[1]
  print('shuffle changes train scores:', not np.allclose(a, b))"  # -> True
  ```
  ferrolearn always takes the first `effective_size` (no shuffle option).
  DIVERGENCE. NOT-STARTED (blocker #1770).
- AC-GROUPS (REQ-GROUPS, groups DIVERGENCE): sklearn threads `groups` to a Group
  cv: `learning_curve(est, X, y, train_sizes=.., cv=GroupKFold(3), groups=g)`
  produces group-respecting folds. ferrolearn's `cv.fold_indices(n_samples)` has
  no `groups` argument, so no Group cv is reachable. DIVERGENCE. NOT-STARTED
  (blocker #1771).
- AC-EXTRAS (REQ-EXTRAS, return_times DIVERGENCE): sklearn `return_times=True`
  returns 5 values with two extra `(n_ticks, n_folds)` timing matrices:
  ```
  python3 -c "import numpy as np
  from sklearn.model_selection import learning_curve, KFold
  from sklearn.dummy import DummyRegressor
  X=np.arange(60).reshape(30,2).astype(float); y=np.arange(30).astype(float)
  out=learning_curve(DummyRegressor(), X, y, train_sizes=[5,10], cv=KFold(3),
      return_times=True, scoring='neg_mean_squared_error')
  print('n returned:', len(out), 'fit_times shape:', out[3].shape)"  # -> 5 (2, 3)
  ```
  ferrolearn returns only `(train_sizes, train_scores, test_scores)`; no
  `return_times`/incremental/fit_params. DIVERGENCE. NOT-STARTED (blocker #1772).

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (core (size,fold) mechanic + iteration-order/orientation equivalence — KEY REQ) | SHIPPED | impl `pub fn learning_curve in learning_curve.rs`: outer `for &size in &abs_sizes`, inner `for (train_idx, test_idx) in &folds` takes `let effective_size = size.min(train_idx.len()); let sub_train_idx = &train_idx[..effective_size];` (FIRST `n_train` of the fold's train indices — mirrors sklearn `train[:n_train_samples]`, `sklearn/model_selection/_validation.py:1952`), `pipeline.fit(&x_train, &y_train)?` then `scoring(&y_train, &y_train_pred)?` / `scoring(&y_test, &y_test_pred)?` on the train subset + FULL test fold, pushing `train_scores_data` / `test_scores_data` in `(size OUTER, fold INNER)` order, finalized `Array2::from_shape_vec((n_sizes, n_folds), ...)` (row-major). Element-for-element IDENTICAL to sklearn's `(fold OUTER, size INNER)` flat list `.reshape(-1, n_unique_ticks).T` (`:1949-1976`): cell `(size i, fold j)` is `flat[j*n_ticks + i]` (sklearn) and `data[i*n_folds + j]` (ferrolearn) — the same `(i, j)`. The per-cell fit+score primitive mirrors `_fit_and_score` (`:1954-1972`). Verification (AC-1): `np.array_equal(M, F)` → `True`. Tests: `test_learning_curve_basic`/`test_learning_curve_absolute_sizes` (`(3,3)` shapes), `test_learning_curve_constant_target_scores_near_zero` (perfect-fit train+test scores ~0), `test_learning_curve_with_transformer`. Non-test consumer: REQ-X-2 (re-export). |
| REQ-2 (train scores always on + shape/orientation `(n_ticks, n_cv_folds)`) | SHIPPED | impl `pub fn learning_curve in learning_curve.rs` unconditionally computes `let y_train_pred = fitted.predict(&x_train)?; let train_score = scoring(&y_train, &y_train_pred)?;` and returns both matrices as `Array2::from_shape_vec((n_sizes, n_folds), ...)` — `struct LearningCurveResult` documents `train_scores`/`test_scores` as `(n_sizes, n_folds)`, row=size tick, col=fold. Mirrors sklearn's hardcoded `return_train_score=True` (`sklearn/model_selection/_validation.py:1967`) and post-transpose `(n_unique_ticks, n_cv_folds)` (`:1861-1864`, `:1975-1976`). Live oracle (AC-2): `learning_curve(DecisionTreeRegressor(), X, y, train_sizes=[5,10,20], cv=KFold(3))` → both `(3, 3)`; ferrolearn `test_learning_curve_basic` asserts `[2,3]`/`[2,3]`, `test_learning_curve_scores_are_finite` asserts every train+test score finite. Non-test consumer: REQ-X-2. |
| REQ-A (fraction→absolute: FLOOR vs CEIL) | NOT-STARTED | open prereq blocker #1764. sklearn `(train_sizes_abs * n_max).astype(int)` (`sklearn/model_selection/_validation.py:2028-2030`) truncates toward zero (floor); impl `pub fn learning_curve in learning_curve.rs` uses `((s * reference_train_len as f64).ceil() as usize)`. Oracle (AC-A): `_translate_train_sizes([0.33], 20)` → `[6]`; ferrolearn `ceil(6.6)=7`. Deterministic single-spot fixable divergence — critic pins a failing tick-count test. |
| REQ-B (float entry `> 1.0` → ValueError vs treat-as-absolute) | NOT-STARTED | open prereq blocker #1765. sklearn uses ARRAY DTYPE (`sklearn/model_selection/_validation.py:2020`): all-float ⇒ fraction mode, every entry `<= 1.0` else `ValueError` (`:2021-2027`); impl `pub fn learning_curve in learning_curve.rs` decides PER-ELEMENT (`if s <= 1.0`), so a float `2.0` becomes an absolute count of 2. Oracle (AC-B): `_translate_train_sizes([0.5, 2.0], 20)` raises `ValueError`; ferrolearn returns size 2. Fixable divergence — critic pins a failing should-error test. |
| REQ-C (absolute `> n_max` → ValueError vs clamp) | NOT-STARTED | open prereq blocker #1766. sklearn's integer path raises if `max > n_max_training_samples` (`sklearn/model_selection/_validation.py:2033-2046`, no clamp); impl `pub fn learning_curve in learning_curve.rs` clamps via `(s as usize).max(1).min(reference_train_len)`. Oracle (AC-C): `_translate_train_sizes([5, 25], 20)` raises `ValueError`; ferrolearn clamps `25 -> 20`. Fixable divergence — critic pins a failing should-error test. |
| REQ-D (dedup + sort via np.unique) | NOT-STARTED | open prereq blocker #1767. sklearn `np.unique(train_sizes_abs)` (`sklearn/model_selection/_validation.py:2048`) sorts ascending AND removes duplicates (RuntimeWarning, `:2049-2055`); impl `pub fn learning_curve in learning_curve.rs` builds `abs_sizes` preserving input order and keeping duplicates. Oracle (AC-D): `[1.0, 0.5]` → `[10, 20]` (sorted) vs ferrolearn `[20, 10]`; `[0.5, 0.5]` → `[10]` (1 tick) vs ferrolearn `[10, 10]` (2 ticks → extra matrix row). Fixable divergence — critic pins a failing sort/dedup test on both the `train_sizes` vector and the row count. |
| REQ-E (error_score=np.nan continue-the-curve) | NOT-STARTED | open prereq blocker #1768. sklearn assigns `error_score` (default np.nan) to a failing `(size, fold)` cell and CONTINUES (`sklearn/model_selection/_validation.py:1968`, `_fit_and_score` `:890-905`); impl `pub fn learning_curve in learning_curve.rs` propagates via `pipeline.fit(&x_train, &y_train)?` / `scoring(...)?` and aborts. Oracle (AC-E): an estimator failing at train size 5 yields `np.isnan(te).any() == True`; ferrolearn returns `Err`. Same contract just SHIPPED for `validation_curve` REQ-7. Fixable divergence — critic pins a failing nan-continue test. |
| REQ-DEFAULTS (default train_sizes / cv / scoring) | NOT-STARTED | open prereq blocker #1769. sklearn defaults `train_sizes=np.linspace(0.1, 1.0, 5)` (`sklearn/model_selection/_validation.py:1730`), `cv=None` → `check_cv` 5-fold classifier-aware StratifiedKFold/KFold (`:1908`), `scoring=None` → r2/accuracy via `check_scoring` (`:1912`). impl `pub fn learning_curve in learning_curve.rs` takes `train_sizes: &[f64]`, `cv: &dyn CrossValidator`, `scoring: fn(...)` as MANDATORY args — no defaults, no `is_classifier` dispatch. Oracle (AC-DEFAULTS): default cv `StratifiedKFold` n_splits 5, default scoring `r2_score`. Missing-feature blocker. |
| REQ-SHUFFLE (shuffle + random_state prefix) | NOT-STARTED | open prereq blocker #1770. sklearn `shuffle=True` permutes each fold's train indices before prefixes (`sklearn/model_selection/_validation.py:1925-1927`); impl `pub fn learning_curve in learning_curve.rs` always takes `&train_idx[..effective_size]` (no shuffle, no RNG). Oracle (AC-SHUFFLE): `shuffle=True, random_state=0` changes train scores vs `shuffle=False`. Missing-feature blocker (RNG-coupled — needs `random_state` determinism). |
| REQ-GROUPS (groups channel for Group cv) | NOT-STARTED | open prereq blocker #1771. sklearn threads `groups` to `cv.split(X, y, groups)` (`sklearn/model_selection/_validation.py:1910`); impl `pub fn learning_curve in learning_curve.rs` calls `cv.fold_indices(n_samples)?`; `CrossValidator::fold_indices(n_samples) in cross_validation.rs` has NO group/label channel, so Group cv is unreachable (same channel-gap as `validation_curve` REQ-8). Missing-feature blocker. |
| REQ-EXTRAS (return_times / exploit_incremental_learning / fit_params) | NOT-STARTED | open prereq blocker #1772. sklearn `return_times` adds two `(n_ticks, n_folds)` timing matrices (`sklearn/model_selection/_validation.py:1979-1982`), `exploit_incremental_learning` uses `partial_fit` (`:1929-1947`), `fit_params` forwards to `fit` (`:1964`). impl `pub fn learning_curve in learning_curve.rs` returns only `(train_sizes, train_scores, test_scores)` and has none of these knobs. Oracle (AC-EXTRAS): `return_times=True` returns 5 values, `fit_times.shape == (2, 3)`. Missing-feature blocker. |
| REQ-X-1 (R-SUBSTRATE) | NOT-STARTED | open prereq blocker #1773. Production code in `learning_curve.rs` imports `use ndarray::{Array1, Array2}` (array type) and builds `Array2::from_shape_vec(...)` / collects `Array1` subsets. Per R-SUBSTRATE-1 the destination is `ferray-core`; `ndarray` is the wrong substrate. Until migrated this unit is not on the ferray substrate (R-SUBSTRATE-2). |
| REQ-X-2 (non-test production consumer) | SHIPPED | `pub fn learning_curve` and `LearningCurveResult` are re-exported at `pub use learning_curve::{LearningCurveResult, learning_curve} in lib.rs` — the boundary public API per S5/R-DEFER-1 grandfathering. Honest underclaim: the SOLE non-test production reach is this re-export; `grep -rn "learning_curve" ferrolearn-model-sel/src ferrolearn/src ferrolearn-python/src` shows no internal caller (no `grid_search`/estimator invokes it) and no `ferrolearn-python` binding. The function IS the public boundary surface, consumed by external users. |

## Architecture

ferrolearn implements `learning_curve` as a single free function plus a plain
result struct (`struct LearningCurveResult in learning_curve.rs`). It takes an
unfitted `pipeline: &Pipeline`, the feature matrix `x: &Array2<f64>`, target
`y: &Array1<f64>`, a cross-validator `cv: &dyn CrossValidator`, the training-size
grid `train_sizes: &[f64]`, and a scoring function `scoring: fn(&Array1<f64>,
&Array1<f64>) -> Result<f64, FerroError>`.

The core fit+score loop is the faithful translation. sklearn builds a FLAT list
over `for train, test in cv_iter: for n_train_samples in train_sizes_abs:
(train[:n_train_samples], test)` (fold OUTER, size INNER, `:1949-1952`), runs
`_fit_and_score` per element, aggregates, and reshapes
`flat.reshape(-1, n_unique_ticks).T` into `(n_unique_ticks, n_cv_folds)`
(`:1975-1976`). ferrolearn instead nests `for &size in &abs_sizes` OUTER and `for
(train_idx, test_idx) in &folds` INNER, pushing into a row-major `(n_sizes,
n_folds)` buffer. These are NOT the same loop nesting, but they produce the SAME
array: cell `(size i, fold j)` is `flat[j*n_ticks + i]` in sklearn (the
`.reshape(-1, n_unique_ticks).T` algebra) and `data[i*n_folds + j]` in ferrolearn
— both the `(i, j)` element. AC-1's `np.array_equal(M, F)` proves the equivalence
on a synthetic `f*10 + s` value that isolates the index algebra from estimator
noise. This ordering-equivalence is the KEY REQ (REQ-1) and it agrees.

The training subset at a tick is the FIRST `effective_size = size.min(
train_idx.len())` indices of the fold's train indices (`&train_idx[..effective_size]`),
mirroring sklearn's `train[:n_train_samples]` (`:1952`). Per fold, ferrolearn
row-copies `x.row(i)` into fresh `Array2`s and collects `y[i]` into `Array1`s,
fits the pipeline, and scores on BOTH the train subset and the FULL test fold
(REQ-2 — unconditional, matching sklearn's hardcoded `return_train_score=True` at
`:1967`). `n_max_training_samples` in sklearn is the FIRST fold's train length
(`:1914`); ferrolearn uses `reference_train_len = folds[0].0.len()` identically.

Validation guards (eager, before the fold loop): `y.len() != n_samples` →
`FerroError::ShapeMismatch`; empty `train_sizes` → `FerroError::InvalidParameter`;
any `train_sizes` entry `<= 0.0` or non-finite → `FerroError::InvalidParameter`.

Where ferrolearn DIVERGES — the size-translation layer (sklearn's
`_translate_train_sizes`, `:1992-2057`) is collapsed into a per-element closure in
`pub fn learning_curve` that differs in five deterministic, oracle-pinnable ways
(all NOT-STARTED, each a filed fixable-divergence blocker the critic will pin as a
failing test):

- **FLOOR vs CEIL** (REQ-A, #1764): ferrolearn `((s * reference_train_len).ceil()
  as usize)` vs sklearn `.astype(int)` truncation (`:2028-2030`).
- **float `> 1.0`** (REQ-B, #1765): ferrolearn per-element `if s <= 1.0` treats a
  float `2.0` as absolute; sklearn's all-float dtype mode raises `ValueError`
  (`:2020-2027`).
- **absolute overshoot** (REQ-C, #1766): ferrolearn clamps via
  `.min(reference_train_len)`; sklearn raises `ValueError` (`:2033-2046`).
- **sort + dedup** (REQ-D, #1767): ferrolearn keeps input order and duplicates;
  sklearn `np.unique` sorts and dedups (`:2048`), changing both the
  `train_sizes_abs` vector and the matrix row count.
- **error_score=np.nan** (REQ-E, #1768): ferrolearn propagates a fit/score
  failure via `?` and aborts; sklearn nan-fills the cell and continues (`:1968`,
  `_fit_and_score` `:890-905`) — the same contract just SHIPPED for
  `validation_curve` REQ-7.

And several whole features are absent (missing-feature blockers): default
`train_sizes`/`cv`/`scoring` (REQ-DEFAULTS, #1769, mandatory args, no `check_cv`/
`check_scoring`/`is_classifier` dispatch), `shuffle`+`random_state` prefix
permutation (REQ-SHUFFLE, #1770), the `groups` channel
(REQ-GROUPS, #1771, `CrossValidator::fold_indices` has no group channel),
`return_times`/`exploit_incremental_learning`/`fit_params` (REQ-EXTRAS, #1772),
and the ferray substrate migration (REQ-X-1, #1773, `ndarray` → `ferray-core`).

The sole non-test production consumer is the crate re-export (REQ-X-2,
`pub use learning_curve::{LearningCurveResult, learning_curve} in lib.rs`). No
internal ferrolearn caller (`grid_search`, any estimator) and no `ferrolearn-python`
binding consume it; the function IS the public boundary surface.

## Verification

Commands establishing the SHIPPED claims (baseline
`b16c37a0ad3ac8d593207810314700423fe64173`):

- `cargo test -p ferrolearn-model-sel --lib learning_curve` → 10 passed, 0 failed
  (`learning_curve::tests::{test_learning_curve_absolute_sizes,
  test_learning_curve_basic, test_learning_curve_constant_target_scores_near_zero,
  test_learning_curve_empty_sizes_error, test_learning_curve_fraction_sizes,
  test_learning_curve_negative_size_error, test_learning_curve_scores_are_finite,
  test_learning_curve_shape_mismatch, test_learning_curve_with_transformer,
  test_learning_curve_zero_size_error}`).
- REQ-1 ordering/orientation-equivalence oracle (the KEY pin — pure-numpy
  reproduction of sklearn's reshape algebra, synthetic per-`(fold, size)` value
  isolates the index math; R-CHAR-3, oracle-derived):
  ```
  python3 -c "import numpy as np
  n_folds=3; n_ticks=4
  flat = np.array([f*10+s for f in range(n_folds) for s in range(n_ticks)])
  M = flat.reshape(-1, n_ticks).T
  F = np.array([[f*10+s for f in range(n_folds)] for s in range(n_ticks)])
  print('IDENTICAL:', np.array_equal(M, F))"   # -> IDENTICAL: True
  ```
- REQ-2 shape+orientation oracle (live sklearn 1.5.2):
  ```
  python3 -c "import numpy as np
  from sklearn.model_selection import learning_curve, KFold
  from sklearn.tree import DecisionTreeRegressor
  X=np.arange(60).reshape(30,2).astype(float); y=np.arange(30).astype(float)
  ts,tr,te=learning_curve(DecisionTreeRegressor(random_state=0), X, y,
      train_sizes=[5,10,20], cv=KFold(3), scoring='neg_mean_squared_error')
  print(tr.shape, te.shape)"   # -> (3, 3) (3, 3)
  ```
  matches ferrolearn `test_learning_curve_basic`/`_absolute_sizes` (`[N,3]`/`[N,3]`).
- REQ-A FLOOR DIVERGENCE oracle: `_translate_train_sizes([0.33], 20)` → `[6]`;
  ferrolearn `ceil(6.6)=7` (#1764).
- REQ-B float-`>1.0` DIVERGENCE oracle: `_translate_train_sizes([0.5, 2.0], 20)`
  raises `ValueError`; ferrolearn returns size 2 (#1765).
- REQ-C overshoot DIVERGENCE oracle: `_translate_train_sizes([5, 25], 20)` raises
  `ValueError`; ferrolearn clamps `25 -> 20` (#1766).
- REQ-D sort/dedup DIVERGENCE oracle: `_translate_train_sizes([1.0, 0.5], 20)` →
  `[10, 20]` (ferrolearn `[20, 10]`); `_translate_train_sizes([0.5, 0.5], 20)` →
  `[10]` (ferrolearn `[10, 10]`, extra row) (#1767).
- REQ-E error_score DIVERGENCE oracle (live sklearn — the nan-continue pin):
  ```
  python3 -c "import numpy as np, warnings
  from sklearn.model_selection import learning_curve, KFold
  from sklearn.base import BaseEstimator, RegressorMixin
  class F(BaseEstimator, RegressorMixin):
      def fit(self, X, y):
          if X.shape[0] == 5: raise ValueError('boom')
          self.m_ = y.mean(); return self
      def predict(self, X): return np.full(X.shape[0], self.m_)
  X=np.arange(60).reshape(30,2).astype(float); y=np.arange(30).astype(float)
  with warnings.catch_warnings(): warnings.simplefilter('ignore')
  ts,tr,te=learning_curve(F(), X, y, train_sizes=[5,10,20], cv=KFold(3),
      scoring='neg_mean_squared_error')
  print('any nan:', np.isnan(te).any())"   # -> any nan: True -> ferrolearn aborts with Err (#1768)
  ```
- REQ-DEFAULTS DIVERGENCE oracle: default `train_sizes` `np.linspace(0.1, 1.0, 5)`
  = `[0.1, 0.325, 0.55, 0.775, 1.0]`; `check_cv(None, y, classifier=True)` →
  `StratifiedKFold` n_splits 5; `check_scoring(DecisionTreeRegressor(),
  scoring=None)` → `r2_score` — ferrolearn requires all three explicitly (#1769).
- REQ-X-1 substrate: `grep -n "ndarray" learning_curve.rs` shows production
  `use ndarray::{Array1, Array2}` and `Array2::from_shape_vec` — wrong substrate,
  migration owed (#1773).
- REQ-X-2 consumer: `grep -rn "learning_curve" ferrolearn-model-sel/src
  ferrolearn/src ferrolearn-python/src` shows only the `lib.rs` re-export
  (`pub use learning_curve::{LearningCurveResult, learning_curve}`), the module
  declaration, and a doc-comment mention in `time_series_split.rs` — no internal
  caller, no Python binding.

SHIPPED: REQ-1 (core (size,fold) mechanic + iteration-order/orientation
equivalence — KEY REQ, agrees), REQ-2 (train scores always + shape/orientation),
REQ-X-2 (consumer — re-export boundary only). NOT-STARTED: REQ-A (floor-vs-ceil,
#1764), REQ-B (float `>1.0` ValueError, #1765), REQ-C (absolute overshoot
ValueError, #1766), REQ-D (sort+dedup, #1767), REQ-E (error_score=np.nan
continue, #1768), REQ-DEFAULTS (default train_sizes/cv/scoring, #1769),
REQ-SHUFFLE (shuffle+random_state, #1770), REQ-GROUPS (groups channel, #1771),
REQ-EXTRAS (return_times/incremental/fit_params, #1772), REQ-X-1 (ferray
substrate, #1773). Per R-DEFER-2 every REQ is binary SHIPPED/NOT-STARTED.
REQ-A/B/C/D/E are FIXABLE-DIVERGENCE blockers (deterministic single-spot bugs the
critic will pin as failing tests); REQ-DEFAULTS/SHUFFLE/GROUPS/EXTRAS/X-1 are
MISSING-FEATURE blockers.

Least-confident SHIPPED claim: REQ-1 — the core fit+score mechanic and the
`(size,fold)` index-orientation equivalence are genuinely SHIPPED and agree, but
the `train_sizes_abs` VALUES feeding the loop are computed by the divergent
translation layer (REQ-A/B/C/D), so for inputs that exercise floor/clamp/dedup,
the SHIPPED mechanic operates on the WRONG set of ticks — the mechanic is correct,
its inputs are not. The honest scope of REQ-1 is the per-cell fit+score+fill
algebra given a tick vector, not end-to-end parity with `sklearn.learning_curve`
on arbitrary `train_sizes`.
