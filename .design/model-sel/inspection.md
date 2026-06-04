# partial_dependence & permutation_importance

<!--
tier: 3-component
status: draft
baseline-commit: dc19b3fb2db61fbfdabba2b7b28ca0d1b28ef985
upstream-paths:
  - sklearn/inspection/_partial_dependence.py   # partial_dependence, _partial_dependence_brute, _grid_from_X
  - sklearn/inspection/_permutation_importance.py # permutation_importance, _calculate_permutation_scores, _create_importances_bunch
-->

## Summary

`ferrolearn-model-sel/src/inspection.rs` mirrors two free functions of
scikit-learn's `sklearn.inspection` module: `partial_dependence`
(`sklearn/inspection/_partial_dependence.py:367`) and `permutation_importance`
(`sklearn/inspection/_permutation_importance.py:135`).

ferrolearn ships the **deterministic, single-feature brute-force** core of
partial dependence: given an explicit grid, for each grid value it overwrites
one feature column with that value, calls a `predict` closure, and averages the
predictions across samples — exactly sklearn's `_partial_dependence_brute`
averaging step (`np.average(pred, axis=0)`). It ships the **structural** core of
permutation importance: baseline score minus permuted score over `n_repeats`
shuffles, with `importances_mean`/`importances_std`/`importances` matching
sklearn's `_create_importances_bunch` shapes and `np.std` (ddof=0) reduction.

What is absent: percentile grid generation (`_grid_from_X`), multi-feature
(2D) PD, ICE (`kind='individual'`/`'both'`), `sample_weight`, the tree
`method='recursion'`, the estimator+`response_method` dispatch, the
`scoring`/`max_samples`/multi-scorer surface of permutation importance, and
exact RNG parity of the column shuffle (`SmallRng` vs numpy `RandomState`). The
R-SUBSTRATE migration to `ferray-core` + `ferray::random` is also NOT-STARTED.

## Upstream reference (scikit-learn 1.5.2, commit 156ef14)

### partial_dependence (`sklearn/inspection/_partial_dependence.py`)
- `:367` — `def partial_dependence(estimator, X, features, *, sample_weight=None,
  categorical_features=None, feature_names=None, response_method="auto",
  percentiles=(0.05, 0.95), grid_resolution=100, method="auto", kind="average")`.
- `:191-344` — `_partial_dependence_brute(est, grid, features, X, response_method,
  sample_weight=None)`: the brute averaging core. For each `new_values in grid`
  (`:293`) it overwrites the target column(s) via `_safe_assign(X_eval,
  new_values[i], column_indexer=variable)` (`:294-295`), predicts
  (`pred = prediction_method(X_eval)`, `:304`), and averages over samples:
  `averaged_predictions.append(np.average(pred, axis=0, weights=sample_weight))`
  (`:308`). `X_eval = X.copy()` (`:292`) so the original `X` is untouched.
- `:267-290` — response dispatch: regressors use `est.predict` (`:268`);
  classifiers try `predict_proba` then `decision_function` per
  `response_method` (`:270-280`).
- `:41-130` — `_grid_from_X(X, percentiles, is_categorical, grid_resolution)`:
  builds the grid. For each feature it uses `np.unique` values when
  `uniques.shape[0] < grid_resolution` or categorical (`:106-110`), else
  `np.linspace` between the empirical `mquantiles(..., prob=percentiles)`
  (`:113-127`). Default `percentiles=(0.05, 0.95)` (`:376`),
  `grid_resolution=100` (`:377`). Returns `cartesian(values)` (`:130`).
- `:133-188` — `_partial_dependence_recursion`: tree-specific fast path
  (`est._compute_partial_dependence_recursion`, `:182`).
- `:684-719` — driver: `_grid_from_X(...)` (`:684`) then `brute`/`recursion`
  branch (`:691-704`), reshape to `(n_outputs, n_values_feature_0, ...)`
  (`:708-710`), `kind` selects `average`/`individual`/`both` (`:713-719`).

### permutation_importance (`sklearn/inspection/_permutation_importance.py`)
- `:135` — `def permutation_importance(estimator, X, y, *, scoring=None,
  n_repeats=5, n_jobs=None, random_state=None, sample_weight=None,
  max_samples=1.0)`. Default `n_repeats=5`.
- `:282` — `baseline_score = _weights_scorer(scorer, estimator, X, y,
  sample_weight)`.
- `:28-79` — `_calculate_permutation_scores`: copies `X` (`:60`), then for
  `_ in range(n_repeats)` (`:64`): `random_state.shuffle(shuffling_idx)`
  (`:65`) and permutes the column in place:
  `X_permuted[:, col_idx] = X_permuted[shuffling_idx, col_idx]` (`:71`), scoring
  each permuted matrix (`:72`).
- `:82-108` — `_create_importances_bunch(baseline_score, permuted_score)`:
  `importances = baseline_score - permuted_score` (`:103`);
  `importances_mean=np.mean(importances, axis=1)` (`:105`),
  `importances_std=np.std(importances, axis=1)` (`:106`, default ddof=0),
  `importances=importances` (`:107`), shapes `(n_features,)`/`(n_features,)`/
  `(n_features, n_repeats)`.
- `:273-274` — `random_state = check_random_state(random_state)` then a single
  `random_seed = random_state.randint(...)` is shared across columns.
- `:276-279` — `max_samples` resolution (float → `int(max_samples*n)`); `:281`
  `scorer = check_scoring(estimator, scoring=scoring)`; `:284-297`
  per-column parallel dispatch.

## Requirements

- REQ-1: **PD brute-averaging value parity.** For each grid value `v`, set the
  target column to `v`, call `predict`, and set the output to the sample mean of
  the predictions. Mirrors `_partial_dependence_brute` averaging
  (`sklearn/inspection/_partial_dependence.py:293-308`). DETERMINISTIC /
  oracle-pinnable with a known predictor and explicit grid.
- REQ-2: **PD closure-vs-estimator idiom (R-DEV-7).** ferrolearn takes a
  `predict` closure (`Fn(&Array2<f64>) -> Result<Array1<f64>, FerroError>`);
  sklearn takes a fitted estimator and dispatches to
  `predict`/`predict_proba`/`decision_function` via `response_method`
  (`:267-290`). The predict-based observable PD matches for regressors; this is
  a Rust-idiom deviation, not a value divergence.
- REQ-3: **PD grid generation.** sklearn derives the grid from `_grid_from_X`
  (`:41-130`) using `percentiles=(0.05,0.95)` and `grid_resolution=100` (or
  unique values for low-cardinality/categorical features); ferrolearn requires
  the caller to pass an EXPLICIT `grid`. Feature gap.
- REQ-4: **PD multi-feature (2D) & ICE.** sklearn accepts a tuple of features
  (cartesian-product grid, `:684`/`:130`) and `kind='individual'`/`'both'` for
  ICE curves (`:713-719`); ferrolearn is single `feature_idx`, `average` only.
  Feature gap.
- REQ-5: **PD sample_weight & method='recursion'.** sklearn supports weighted
  averaging (`np.average(..., weights=sample_weight)`, `:308`) and a tree-only
  recursion fast path (`:133-188`); ferrolearn is unweighted brute only. Feature
  gap.
- REQ-6: **permutation_importance structure.** baseline `score(x,y)`; per
  feature, shuffle that column `n_repeats` times and record
  `importance = baseline - permuted_score`; expose
  `importances_mean`/`importances_std`/`importances` with shapes
  `(n_features,)`/`(n_features,)`/`(n_features, n_repeats)` and `np.std` ddof=0.
  Mirrors `_calculate_permutation_scores` + `_create_importances_bunch`
  (`:28-108`). The SIGN convention, the shapes, and the std reduction are
  DETERMINISTIC / oracle-pinnable (structurally); the EXACT per-repeat values are
  RNG-coupled.
- REQ-7: **permutation_importance exact-value RNG carve-out.** The column shuffle
  runs on `rand`'s `SmallRng`; sklearn shuffles on numpy `RandomState`
  (`:65`). With equal seeds the two PRNG streams differ, so exact per-repeat
  `importances` cannot bit-match; `random_state=None` is non-deterministic in
  both. R-DEFER-3 carve-out — blocker, NO failing test.
- REQ-8: **permutation_importance API gaps.** ferrolearn takes a `score` closure;
  sklearn takes an estimator + `scoring` (`check_scoring`, `:281`). Also missing:
  `max_samples` subsampling (`:48-58`, `:276-279`), `sample_weight` (`:282`),
  multiple scorers / dict result (`:299-307`), `n_jobs` parallelism. `n_repeats`
  default 5 matches (`:141`). Feature/API gap.
- REQ-9: **R-SUBSTRATE.** Owned computation runs on `ndarray` + `rand`, not
  `ferray-core` + `ferray::random`. Migration NOT-STARTED.
- REQ-10: **Non-test production consumer.** Classify the consumer surface
  honestly per S5.

## Acceptance criteria

- AC-1 (REQ-1): with a linear predictor `predict(X)=X·[0.5,0.5]-1.0` (a live
  `LinearRegression().fit(X,y)` on `X=[[1,2],[3,4],[5,6],[7,8]]`,
  `y=X·[2,-1]+0.5`) and explicit `grid=[0,1,2,10]` on `feature_idx=0`,
  ferrolearn's `averaged_predictions` equals the live-oracle brute PD
  `[mean(m.predict(X with col0=v)) for v in grid]` = `[1.5, 2.0, 2.5, 6.5]`
  within 1e-9. DETERMINISTIC / oracle-pinnable.
- AC-2 (REQ-2): a regressor's predict-based PD from the closure matches sklearn's
  `partial_dependence(estimator, ..., kind='average')['average']` row; the
  closure encodes the estimator's `predict`. Idiom deviation, value parity.
- AC-3 (REQ-3): ferrolearn requires an explicit `grid`; calling without one is a
  compile error (no overload). No percentile/`grid_resolution` path exists.
  NOT-STARTED.
- AC-4 (REQ-4): a tuple of features / `kind='individual'` has no ferrolearn
  analog (single `usize` feature, `averaged_predictions` only). NOT-STARTED.
- AC-5 (REQ-5): a `sample_weight` argument and a `method='recursion'` path have
  no ferrolearn analog. NOT-STARTED.
- AC-6 (REQ-6): the result has `importances.dim() == (n_features, n_repeats)`;
  `importances_std` equals `np.std(importances, axis=1)` (ddof=0); shuffling a
  column whose value drives the score yields a non-zero mean of the correct sign
  (`baseline - permuted`), while a constant column yields 0. Structurally
  oracle-pinnable (e.g. live `_create_importances_bunch` reduces
  `[[1,2,3,4,5]]` to mean `3.0`, std `1.41421356…` ddof=0 — ferrolearn's mean/std
  reduction matches).
- AC-7 (REQ-7): exact per-repeat `importances` are NOT asserted equal to sklearn
  (different PRNG); a carve-out blocker is filed, no failing test.
- AC-8 (REQ-8): `scoring=` strings, `max_samples`, `sample_weight`, and
  multi-scorer dict results have no ferrolearn analog; `n_repeats` default 5
  matches sklearn. NOT-STARTED on the gaps.
- AC-9 (REQ-9): owned computation uses `ferray-core` + `ferray::random`, no
  `ndarray`/`rand` in the owned path. NOT-STARTED.
- AC-10 (REQ-10): the functions are reachable from non-test production code.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (PD brute-averaging value parity) | SHIPPED | impl `pub fn partial_dependence` in `inspection.rs`: for each grid value it clones `x`, sets `x_modified[[i, feature_idx]] = v` over all rows, calls `predict(&x_modified)?`, and writes `out[gi] = preds.iter().sum::<f64>() / n` where `n = x.nrows()`. This is exactly sklearn's per-grid-value column overwrite + sample mean: `_safe_assign(X_eval, new_values[i], column_indexer=variable)` (`sklearn/inspection/_partial_dependence.py:294-295`) then `np.average(pred, axis=0)` (`:308`), with `X_eval = X.copy()` (`:292`) mirrored by the per-step `x.clone()`. DETERMINISTIC / oracle-pinnable: with `predict` = a live `LinearRegression().fit(X,y)` (`X=[[1,2],[3,4],[5,6],[7,8]]`, `y=X·[2,-1]+0.5`; fitted `coef_≈[0.5,0.5]`, `intercept_≈-1.0`) on `feature_idx=0`, `grid=[0,1,2,10]`, the live brute PD is `[1.5, 2.0, 2.5, 6.5]` (Verification) and ferrolearn's `averaged_predictions` equals it within 1e-9 (R-CHAR-3: computed from the live oracle's `.predict`, never copied from ferrolearn). Tests: `partial_dependence_constant_predict`, `partial_dependence_uses_target_feature`. CAVEAT: the two existing `#[test]`s use hand-built closures (constant 5.0; identity-of-column-0) and hand-derived expectations, not a real sklearn estimator — the critic should pin an oracle-grounded `#[test]` against the live `LinearRegression` brute PD above. Non-test consumer: REQ-10. |
| REQ-2 (PD closure-vs-estimator idiom, R-DEV-7) | SHIPPED | `pub fn partial_dependence` takes `predict: P` where `P: Fn(&Array2<f64>) -> Result<Array1<f64>, FerroError>` — the Rust-idiom analog of sklearn's fitted-estimator `prediction_method` (`sklearn/inspection/_partial_dependence.py:267-290`). For a regressor the closure encodes `est.predict` (`:268`), and the averaged PD over the closure equals sklearn's `kind='average'` `['average']` row (REQ-1's oracle establishes the value parity). This is a sanctioned R-DEV-7 deviation: the OBSERVABLE contract (predict-based averaged PD for regressors) is preserved; the API shape (closure vs estimator object) differs because Rust has no duck-typed `getattr(est, "predict_proba")`. HONEST UNDERCLAIM: the closure form cannot reproduce sklearn's classifier `predict_proba`/`decision_function` auto-dispatch (`:270-280`) or its positive-class selection for binary classifiers (`:323-342`) — those live in REQ-4/the estimator surface, not here. Non-test consumer: REQ-10. |
| REQ-3 (PD grid generation) | NOT-STARTED | open prereq blocker (tracking #1729; critic to file specific). `partial_dependence` requires the caller to pass an explicit `grid: &Array1<f64>` and only validates it is non-empty (`if grid.is_empty()` → `FerroError::InvalidParameter`); there is NO percentile/`grid_resolution` derivation. sklearn builds the grid internally via `_grid_from_X` from `percentiles=(0.05,0.95)`, `grid_resolution=100`, or unique values for low-cardinality/categorical features (`sklearn/inspection/_partial_dependence.py:41-130`, `:684`). Absent end-to-end: ferrolearn exposes neither `percentiles`, `grid_resolution`, nor the `mquantiles`/`linspace` machinery. |
| REQ-4 (PD multi-feature 2D & ICE) | NOT-STARTED | open prereq blocker (tracking #1729). `partial_dependence`'s `feature_idx: usize` is a SINGLE feature, and `PartialDependenceResult` carries only `grid`/`averaged_predictions` — no per-sample `individual` array. sklearn accepts a tuple of `features` (cartesian-product grid, `sklearn/inspection/_partial_dependence.py:684`, `:130`) and `kind='individual'`/`'both'` returning the ICE `predictions` (`:715-719`). No 2D-grid path, no ICE output. Absent. |
| REQ-5 (PD sample_weight & method='recursion') | NOT-STARTED | open prereq blocker (tracking #1729). `partial_dependence` averages UNWEIGHTED (`preds.iter().sum::<f64>() / n`) with no `sample_weight` parameter and no recursion branch. sklearn supports `np.average(pred, axis=0, weights=sample_weight)` (`sklearn/inspection/_partial_dependence.py:308`) and a tree-only `_partial_dependence_recursion` fast path (`:133-188`, `:701-704`). Both absent; the recursion path additionally requires fitted tree internals ferrolearn does not expose through the closure API. |
| REQ-6 (permutation_importance structure) | SHIPPED | impl `pub fn permutation_importance` in `inspection.rs`: `baseline = score(x, y)?`; for each feature `j` and repeat `r`, shuffles a row-index vector and writes `shuffled[[i, j]] = x[[indices[i], j]]` (column-`j` permutation, all other columns untouched), then `importances[[j, r]] = baseline - s` where `s = score(&shuffled, y)?`. This mirrors `_calculate_permutation_scores`' in-place `X_permuted[:, col_idx] = X_permuted[shuffling_idx, col_idx]` (`sklearn/inspection/_permutation_importance.py:71`) and the `importances = baseline_score - permuted_score` sign of `_create_importances_bunch` (`:103`). The reductions match: `means[j] = sum/n_repeats` = `np.mean(..., axis=1)` (`:105`); `stds[j] = sqrt(sum((v-m)^2)/n_repeats)` = `np.std(..., axis=1)` with default ddof=0 (`:106`) — verified live: `_create_importances_bunch` reduces `[[1,2,3,4,5]]` to mean `3.0`, std `1.4142135623730951` (ddof=0; ddof=1 would be `1.5811…`), and ferrolearn divides the squared-deviation sum by `n_repeats` (ddof=0), matching (Verification, R-CHAR-3). Shapes match: `importances` is `Array2(n_features, n_repeats)`, `importances_mean`/`importances_std` are `Array1(n_features)` (sklearn `:96-100`). DETERMINISTIC / oracle-pinnable on shape + sign + std-ddof; the EXACT per-repeat values are RNG-coupled (REQ-7). Tests: `permutation_importance_zero_for_useless_feature` (constant score → 0 importance), `permutation_importance_detects_useful_feature` (shape `(3,5)`, constant columns → 0), `permutation_importance_zero_repeats_rejected`. CAVEAT: the critic should pin an oracle-grounded std-ddof `#[test]` (e.g. a deterministic score whose per-repeat importances are `[1,2,3,4,5]` → std `1.41421356…`, the ddof=0 value above). Non-test consumer: REQ-10. |
| REQ-7 (permutation exact-value RNG carve-out) | NOT-STARTED | open prereq blocker (tracking #1729; R-DEFER-3 carve-out — NO failing test). The shuffle runs on `rand::rngs::SmallRng` seeded `SmallRng::seed_from_u64(seed.wrapping_add((j*n_repeats+r) as u64))` with `indices.shuffle(&mut rng)` (and `SmallRng::from_os_rng()` when `random_state=None`). sklearn shuffles on numpy `RandomState` via `random_state.shuffle(shuffling_idx)` (`sklearn/inspection/_permutation_importance.py:65`), with a single shared `random_seed` across columns (`:273-274`). Because the PRNG substrates differ, the exact per-repeat `importances` CANNOT bit-match even with equal seeds; ferrolearn's per-`(j,r)` re-seeding also differs from sklearn's per-column single-stream `shuffle`. `random_state=None` is non-deterministic in both. Per R-DEFER-3 a distributional carve-out: blocker filed, NO failing `#[test]` pinned. (Also tied to REQ-9: the `SmallRng` usage is the proximate substrate gap.) |
| REQ-8 (permutation API gaps) | NOT-STARTED | open prereq blocker (tracking #1729). `permutation_importance` takes a `score: S` closure (`Fn(&Array2<f64>, &Array1<f64>) -> Result<f64, FerroError>`) and exposes only `n_repeats`/`random_state`. Missing vs sklearn: `scoring=` string/callable/list/dict via `check_scoring` (`sklearn/inspection/_permutation_importance.py:281`), `max_samples` subsampling (`:48-58`, `:276-279`), `sample_weight` (`:282`, `_weights_scorer`), multiple scorers returning a dict of Bunches (`:299-307`), and `n_jobs` parallelism (`:284`). `n_repeats` default 5 MATCHES sklearn (`:141`) — noted as already-aligned, but the function has no default-argument mechanism (the caller always passes `n_repeats` explicitly). Absent end-to-end for the listed gaps. |
| REQ-9 (R-SUBSTRATE) | NOT-STARTED | open prereq blocker (tracking #1729; R-SUBSTRATE-2/3). `inspection.rs` imports `ndarray::{Array1, Array2}` and `rand::{SeedableRng, rngs::SmallRng, seq::SliceRandom}`. The destination substrate is the `ferray-core` array type and `ferray::random` (R-SUBSTRATE-1). Not migrated; the `SmallRng` shuffle is also the proximate cause of the REQ-7 RNG carve-out, and the `ndarray` matrices/vectors are the owned compute type. |
| REQ-10 (non-test production consumer) | SHIPPED | Crate re-export: `lib.rs` (`pub use inspection::{PartialDependenceResult, PermutationImportanceResult, partial_dependence, permutation_importance}`) over `pub mod inspection`. R-DEFER-1 / S5: these are existing free functions in the `sklearn.inspection` public surface, grandfathered as the public API. HONEST UNDERCLAIM: a grep across the workspace (excluding `inspection.rs` and `#[cfg(test)]`) finds NO non-test, non-re-export caller and NO `ferrolearn-python` binding for either function — the only references are the `lib.rs` re-export and the in-module tests. SHIPPED on the boundary re-export per S5, not a dedicated production caller; the narrower-than-sklearn surface (no Python binding, no internal estimator consumer such as a `PartialDependenceDisplay`) is noted. |

## Architecture

`inspection.rs` exposes two free functions and two result structs, paralleling
sklearn's free-function design (the functions are NOT methods on an estimator).

`partial_dependence<P>(predict, x, feature_idx, grid) -> PartialDependenceResult`
implements ONLY sklearn's `_partial_dependence_brute` averaging loop. It
validates `feature_idx < x.ncols()` and a non-empty `grid` (both →
`FerroError::InvalidParameter`), then for each grid value `v` clones `x`,
overwrites column `feature_idx` with `v`, calls the `predict` closure, and stores
the sample mean. `PartialDependenceResult { grid, averaged_predictions }` is the
single-output, `kind='average'` slice of sklearn's `Bunch(grid_values, average,
individual)`. The closure `P: Fn(&Array2<f64>) -> Result<Array1<f64>,
FerroError>` is the R-DEV-7 analog of sklearn's estimator + `response_method`
dispatch (`sklearn/inspection/_partial_dependence.py:267-290`) — observably
equal for regressors, but unable to express the classifier
`predict_proba`/`decision_function`/positive-class paths. The grid is a
caller-supplied `Array1<f64>`; sklearn's internal `_grid_from_X` percentile
machinery (`:41-130`) is entirely absent (REQ-3), as are 2D/ICE (REQ-4) and
`sample_weight`/recursion (REQ-5).

`permutation_importance<S>(score, x, y, n_repeats, random_state) ->
PermutationImportanceResult` implements the structural core. It computes
`baseline = score(x, y)`, then for each feature `j` and repeat `r` builds a
fresh `SmallRng` (seeded from `random_state + (j*n_repeats+r)`, or OS entropy
when `None`), shuffles a `0..n` index vector, materializes a column-`j`-permuted
copy of `x`, scores it, and records `baseline - s`. The mean and std are reduced
per feature with ddof=0 (`var = sum((v-m)^2)/n_repeats`, `std = sqrt(var)`),
matching sklearn's `np.mean`/`np.std` in `_create_importances_bunch`
(`sklearn/inspection/_permutation_importance.py:103-107`). The
`PermutationImportanceResult { importances_mean, importances_std, importances }`
field names and shapes (`(n_features,)`, `(n_features,)`, `(n_features,
n_repeats)`) mirror the sklearn `Bunch` exactly. What diverges is the RNG: numpy
`RandomState.shuffle` with a single per-column seed (`:65`, `:273-274`) vs
per-`(j,r)` `SmallRng` reseeds — exact per-repeat values cannot match (REQ-7
carve-out). The `scoring`/`max_samples`/`sample_weight`/multi-scorer/`n_jobs`
surface (`:281`, `:48-58`, `:282`, `:299-307`, `:284`) is absent (REQ-8).

Invariants: PD `averaged_predictions` has the same length as `grid`;
`feature_idx` must be in range; `grid` must be non-empty. Permutation
`importances` is `(n_features, n_repeats)`; `n_repeats >= 1`; a column whose
permutation does not change the score yields exactly 0 importance (the sign is
`baseline - permuted`, positive when shuffling hurts the score).

## Verification

Commands establishing the SHIPPED claims (baseline
`dc19b3fb2db61fbfdabba2b7b28ca0d1b28ef985`):

- `cargo test -p ferrolearn-model-sel --lib inspection` → 6 `partial_dependence_*`
  / `permutation_importance_*` tests pass (REQ-1 brute averaging; REQ-6
  structure/shape/sign; `n_repeats==0` and `feature_idx` validation).
- REQ-1 oracle (brute PD value parity, live oracle):
  ```
  python3 -c "import numpy as np; from sklearn.linear_model import LinearRegression; \
  X=np.array([[1.,2.],[3.,4.],[5.,6.],[7.,8.]]); y=X@np.array([2.,-1.])+0.5; \
  m=LinearRegression().fit(X,y); grid=[0.,1.,2.,10.]; \
  print([round(float(m.predict(np.column_stack([np.full(4,v),X[:,1]])).mean()),10) for v in grid])"
  # -> [1.5, 2.0, 2.5, 6.5]   (== ferrolearn averaged_predictions on feature_idx=0)
  ```
  Pin an oracle-grounded `#[test]`: a linear `predict(X)=X·coef_+intercept_`
  closure (from the live fit above) with `grid=[0,1,2,10]` on `feature_idx=0`
  must yield `[1.5,2.0,2.5,6.5]` within 1e-9 (R-CHAR-3: from the live oracle).
- REQ-6 oracle (permutation std uses ddof=0, live oracle):
  ```
  python3 -c "import numpy as np; \
  from sklearn.inspection._permutation_importance import _create_importances_bunch; \
  b=_create_importances_bunch(np.array([0.0]), -np.array([[1.,2.,3.,4.,5.]])); \
  print(b.importances_mean.tolist(), b.importances_std.tolist())"
  # -> [3.0] [1.4142135623730951]   (np.std ddof=0; ddof=1 would be 1.5811388300841898)
  ```
  ferrolearn's per-feature reduction (`var = sum((v-m)^2)/n_repeats`) is ddof=0
  and matches. Pin an oracle-grounded `#[test]` driving a deterministic score
  whose per-repeat importances are `[1,2,3,4,5]`, expecting mean `3.0`, std
  `1.4142135623730951`.
- REQ-2 (PD closure idiom): the closure-driven averaged PD equals sklearn's
  `partial_dependence(m, X, [0], grid_resolution=..., kind='average')['average']`
  row for a regressor; the value parity rides on REQ-1's oracle.

Commands that establish the NOT-STARTED REQs are absent: no `percentiles`/
`grid_resolution`/`_grid_from_X` path (REQ-3), no tuple-feature/ICE output
(REQ-4), no `sample_weight`/`recursion` (REQ-5), no numpy-`RandomState` stream
parity for the shuffle (REQ-7, carve-out — no failing test), no
`scoring`/`max_samples`/multi-scorer surface (REQ-8), no `ferray-core`/
`ferray::random` usage (REQ-9), no non-test/non-re-export caller or Python
binding (REQ-10, honest underclaim). Per R-DEFER-2 the table is binary
SHIPPED/NOT-STARTED.

SHIPPED: REQ-1 (PD brute-averaging value parity, oracle-pinnable), REQ-2 (PD
closure-vs-estimator R-DEV-7 idiom, regressor value parity), REQ-6
(permutation_importance structure: sign, shapes, std-ddof=0), REQ-10 (boundary
re-export consumer; no dedicated non-test caller, no Python binding — honest
underclaim). NOT-STARTED (tracking #1729; the critic files per-REQ blockers):
REQ-3 (percentile grid generation), REQ-4 (multi-feature 2D PD + ICE), REQ-5
(`sample_weight` + `method='recursion'`), REQ-7 (exact permutation values — RNG
carve-out, no failing test), REQ-8 (permutation `scoring`/`max_samples`/
multi-scorer API gaps), REQ-9 (ferray substrate).
