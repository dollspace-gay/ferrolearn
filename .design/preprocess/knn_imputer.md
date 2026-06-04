# KNNImputer

<!--
tier: 3-component
status: draft
baseline-commit: ee8994f2
upstream: scikit-learn 1.5.2 (commit 156ef14)
upstream-paths:
  - sklearn/impute/_knn.py  # class KNNImputer(_BaseImputer) (:69); _parameter_constraints {n_neighbors:[Interval(Integral,1,None,closed="left")], weights:[StrOptions({"uniform","distance"}),callable,Hidden(None)], metric:[StrOptions(_NAN_METRICS),callable], copy:["boolean"]} (:130-136); __init__(*, missing_values=np.nan, n_neighbors=5, weights="uniform", metric="nan_euclidean", copy=True, add_indicator=False, keep_empty_features=False) (:138-157); _calc_impute(dist_pot_donors, n_neighbors, fit_X_col, mask_fit_X_col) (:159-204): donors_idx=np.argpartition(dist_pot_donors, n_neighbors-1, axis=1)[:, :n_neighbors] (:184-186), donors_dist gather (:189-191), weight_matrix=_get_weights(donors_dist, self.weights) (:193), weight nan->0 (:196-197), donors=np.ma.array(fit_X_col.take(donors_idx), mask=mask) (:200-202), np.ma.average(donors, axis=1, weights=weight_matrix) (:204); fit(X, y=None) (:206-244): force_all_finite="allow-nan" if scalar-nan else True (:225-228), _validate_data(accept_sparse=False, dtype=FLOAT_DTYPES, force_all_finite, copy) (:230-236), _fit_X=X (:238), _mask_fit_X=_get_mask(_fit_X, missing_values) (:239), _valid_mask=~np.all(_mask_fit_X, axis=0) (:240), super()._fit_indicator(_mask_fit_X) (:242); transform(X) (:246-362): check_is_fitted (:261), _validate_data(reset=False, force_writeable=True,...) (:266-274), mask=_get_mask(X) (:276), X_indicator=_transform_indicator(mask) (:280), no-missing path drops cols X[:, valid_mask] unless keep_empty_features->X[:,~valid_mask]=0 (:282-295), row_missing_idx=np.flatnonzero(mask.any(axis=1)) (:297), per col skip if not valid_mask[col] (:309-312), potential_donors_idx=np.nonzero(non_missing_fix_X[:,col]) (:319), all_nan_dist receivers -> col_mean=np.ma.array(_fit_X[:,col], mask=mask_fit_X[:,col]).mean() (:329-337), n_neighbors=min(self.n_neighbors, len(potential_donors_idx)) (:349), value=_calc_impute(...) (:350-356); pairwise_distances_chunked(metric=self.metric) (:359-364).
  - sklearn/metrics/pairwise.py  # nan_euclidean_distances(X, Y=None, *, squared=False, missing_values=np.nan, copy=True) (:430-549): missing->0 (:519-520), distances=euclidean_distances(X,Y,squared=True) (:522), subtract masked sq (:525-528), clip>=0 (:530), present_count=np.dot(present_X, present_Y.T) (:539), distances[present_count==0]=np.nan (:540), np.maximum(1, present_count) (:542), distances /= present_count (:543), distances *= X.shape[1] (:544), np.sqrt (:546-547) => dist = sqrt(sum_sq * n_features / present_count). THE SCALING (:543-547) is DIV-1.
  - sklearn/neighbors/_base.py  # _get_weights(dist, weights): "uniform"->None (equal); "distance"->1/dist, and if any point has dist==0 those donors get weight 1.0 and all others 0.0 (exact-match handling).
ferrolearn-module: ferrolearn-preprocess/src/knn_imputer.rs
parity-ops: KNNImputer
crosslink-issue: 1304
-->

## Summary

scikit-learn's `KNNImputer` (`_knn.py:69`) replaces each missing entry by the
(optionally distance-weighted) average of that feature over the `n_neighbors`
nearest training rows that have the feature present. "Nearest" is measured by
`nan_euclidean_distances` (`pairwise.py:430`): the squared differences over the
features present in **both** rows, then **scaled** by `n_features /
present_count` and square-rooted — `dist = sqrt(sum_sq * n_features /
present_count)` (`:543-547`). Receivers whose distance to **every** potential
donor is `NaN` are imputed with the **masked column mean** of the training
column (`:329-337`); columns that were **all-missing** during `fit` are
**dropped** from the output unless `keep_empty_features` (`:282-295`,`:310-312`).

`ferrolearn-preprocess/src/knn_imputer.rs` ships the **neighbor-search +
weighted-donor-average SHAPE**: `KNNImputer<F> { n_neighbors: usize, weights:
KNNWeights }` (`KNNWeights::{Uniform, Distance}`, `Default` = `(5, Uniform)`)
fits into `FittedKNNImputer<F> { train_data: Array2<F>, n_neighbors, weights }`,
which on `transform` computes the scaled `nan_euclidean` distance to every train
row (inf for no-shared-features, sorted last), collects up to `n_neighbors` donors
that have feature `j` present, and averages them (Uniform = mean of selected
donors; Distance = inverse-distance, with exact-match donors taking weight 1 and
all others 0; empty/all-inf → masked training column mean). Non-test consumer: the crate
re-export `pub use knn_imputer::{FittedKNNImputer, KNNImputer, KNNWeights};`
(`ferrolearn-preprocess/src/lib.rs`, the boundary public API). There is **no
PyO3 binding** (`ferrolearn-python/` does not reference `KNNImputer`).

**Headline finding — the imputation VALUE surface is now faithful (this
iteration fixed 5 divergences).** `partial_euclidean_distance` now applies the
sklearn `nan_euclidean` scaling `sqrt(sum_sq * n_features / n_valid)` (REQ-2,
#1305); the empty-donor branch imputes the masked training column mean (REQ-3,
#1306); `n_neighbors > n_samples` no longer errors (DIV-3, #1307); exact-match
donors get weight 1 / others 0 (DIV-4, #1308); and inf-distance (no-shared-feature)
donors are included in the uniform average to fill the `n_neighbors` quota like
sklearn `argpartition` (DIV-5, #1309). 33 oracle green guards (8×5 / 10×6 full
matrices, k∈{2,5}, uniform+distance, mixed finite/inf donors, f32, column-mean,
clamp) value-match within ~1e-9. The one remaining edge is exact-distance-tie
donor selection (DIV-6, #1310) — numpy `argpartition` unspecified tie order +
ULP float noise, a documented carve-out (not a meaningful parity target).

This is a **mostly-NOT-STARTED** unit: 4 SHIPPED (REQ-1 imputation value surface,
REQ-2 scaling, REQ-3 column mean, REQ-9 error/clamp contracts) / 8 NOT-STARTED
(REQ-4 column-drop, REQ-5 missing_values, REQ-6 add_indicator, REQ-7
keep_empty_features, REQ-8 callable weights/metric, REQ-10 _BaseImputer surface,
REQ-11 PyO3, REQ-12 ferray).

## Probes (live sklearn oracle, 1.5.2)

```bash
# REQ-1 — basic uniform impute, all rows share all present features (no DIV-1 bite):
python3 -c "import numpy as np; from sklearn.impute import KNNImputer; nan=np.nan; \
print(KNNImputer(n_neighbors=2).fit_transform([[1,2],[3,4],[5,nan]]).tolist())"
# -> [[1.0, 2.0], [3.0, 4.0], [5.0, 3.0]]
#    Row 2 col 1 = mean(2, 4) = 3.0. ferrolearn test_knn_imputer_uniform_basic pins the SAME 3.0.

# REQ-2 (DIV-1) — the nan_euclidean SCALING (sqrt(sum_sq * n_features / present_count), :543-547)
# FLIPS the nearest donor when candidates share different #present features. 4-feature fixture,
# receiver row0=[0,0,nan,0] (present cols 0,1,3) needs col2; donorA=[3,3,10,3] (shares 3 cols),
# donorB=[nan,nan,20,5] (shares 1 col):
python3 -c "import numpy as np; from sklearn.impute import KNNImputer; \
from sklearn.metrics.pairwise import nan_euclidean_distances; nan=np.nan; \
X=np.array([[0.,0.,nan,0.],[3.,3.,10.,3.],[nan,nan,20.,5.]]); \
print('dist row0->all', np.round(nan_euclidean_distances(X[[0]],X),5).tolist()); \
print('n1 col2', KNNImputer(n_neighbors=1).fit_transform(X)[0,2])"
# -> dist row0->all [[0.0, 6.0, 10.0]]   donorA scaled=sqrt(27*4/3)=6.0, donorB scaled=sqrt(25*4/1)=10.0
# -> n1 col2 10.0    (sklearn picks donorA=10.0)
#    ferrolearn's RAW sqrt: donorA=sqrt(27)=5.196, donorB=sqrt(25)=5.0 -> picks donorB -> imputes 20.0.
#    SCALING flips the choice: sklearn 10.0 vs ferrolearn 20.0. DIV-1.

# REQ-3 (DIV-2) — receivers with all-NaN distances impute the masked TRAINING COLUMN MEAN (:329-337),
# not 0. Col0 missing in the col1-present rows and vice versa -> no common feature -> all-nan dist:
python3 -c "import numpy as np; from sklearn.impute import KNNImputer; nan=np.nan; \
print(KNNImputer(n_neighbors=2).fit_transform([[nan,1.],[nan,2.],[5.,nan]]).tolist())"
# -> [[5.0, 1.0], [5.0, 2.0], [5.0, 1.5]]
#    Row 2 col 1 = masked mean of col 1 over training = mean(1, 2) = 1.5.
#    ferrolearn fills F::zero() = 0.0 when neighbor_vals is empty. DIV-2.
```

## Requirements

- REQ-1: **KNN imputation SHAPE — partial-distance neighbor search + uniform /
  distance-weighted donor average** (scoped). For each missing `x[i, j]`: compute
  a partial Euclidean distance (squared diffs over features present in both rows)
  to every train row, sort ascending, take up to `n_neighbors` donors that have
  feature `j` present, and average them — `Uniform` = arithmetic mean, `Distance`
  = inverse-distance weighting. Mirrors sklearn `_calc_impute` (`:184-204`:
  `argpartition` the `n_neighbors` nearest donors `:184-186`, `_get_weights(
  donors_dist, weights)` `:193`, masked `np.ma.average` `:204`) and the per-(row,
  col) donor selection in `transform` (`:319`,`:349-356`). Supports `f32`/`f64`.
  **Scope (R-HONEST-3): value parity holds only where DIV-1/DIV-2 do not bite —
  i.e. every candidate donor shares the SAME present features with the receiver
  (so the `n_features / present_count` scale is a constant that cancels in the
  ordering) AND at least one donor is reachable.** The distance SCALING is REQ-2,
  the empty-donor fallback is REQ-3, the exact `_get_weights` semantics are REQ-8.

- REQ-2: **nan_euclidean distance SCALING** (DIV-1, the headline value gap) —
  `dist = sqrt(sum_sq * n_features / present_count)`: after summing the
  present-in-both squared diffs, sklearn computes `present_count = dot(present_X,
  present_Y.T)` (`:539`), sets all-missing pairs to `NaN` (`:540`), clamps
  `present_count` to `>= 1` (`:542`), then `distances /= present_count;
  distances *= n_features; sqrt` (`:543-547`). ferrolearn's
  `partial_euclidean_distance` returns the **unscaled** `sum_sq.sqrt()` — the
  source literally comments the sklearn formula then says *"we keep it simple
  here: just use sqrt(sum_sq)"* (`fn partial_euclidean_distance in
  knn_imputer.rs`). When donors share differing numbers of present features the
  scale flips the neighbor ordering and the imputed value (Probe REQ-2: `10.0`
  vs `20.0`).

- REQ-3: **Empty-donor → masked training COLUMN MEAN** (DIV-2, fixable) — a
  receiver whose distance to **every** potential donor is `NaN` is imputed with
  `np.ma.array(self._fit_X[:, col], mask=mask_fit_X[:, col]).mean()` — the
  masked mean of the training column (`:329-337`). ferrolearn instead fills
  `F::zero()` (`if neighbor_vals.is_empty() { out[[i, j]] = F::zero(); }` in
  `impl Transform for FittedKNNImputer`). Probe REQ-3: sklearn `1.5`, ferrolearn
  `0.0`.

- REQ-4: **`valid_mask` column dropping** (output-shape contract) — columns that
  were **all-missing** during `fit` (`_valid_mask = ~np.all(_mask_fit_X, axis=0)`,
  `:240`) are **dropped** from the transformed output (`X[:, valid_mask]`,
  `:289`) unless `keep_empty_features` zeros them (`:285-287`,`:310-312`). The
  output has `n_output_features = valid_mask.sum()` columns, not `n_features`.
  ferrolearn's `transform` always returns the same shape as the input (it copies
  `x` and never drops a column); `valid_mask` is not computed at `fit`.

- REQ-5: **`missing_values` parameter (non-NaN sentinel)** — sklearn treats an
  arbitrary scalar (`missing_values`, default `np.nan`) as the missing marker via
  `_get_mask(X, self.missing_values)` (`:239`,`:276`), switching to
  `force_all_finite=True` for non-NaN sentinels (`:225-228`). ferrolearn
  hard-codes `NaN`: every mask check is `v.is_nan()` / `a.is_nan()` (in
  `partial_euclidean_distance` and `transform`); there is no `missing_values`
  field.

- REQ-6: **`add_indicator` (MissingIndicator concatenation)** — when
  `add_indicator=True`, sklearn fits a `MissingIndicator` (`super()._fit_indicator`,
  `:242`) and **concatenates** its binary missingness columns onto the imputed
  output (`super()._concatenate_indicator(Xc, X_indicator)`, `:280`,`:295`).
  ferrolearn has no `add_indicator` field and no indicator concatenation.

- REQ-7: **`keep_empty_features`** — when `True`, all-missing-at-fit columns are
  **kept** (filled with `0`) instead of dropped (`Xc[:, ~valid_mask] = 0`,
  `:285-287`,`:310-312`). ferrolearn has no `keep_empty_features` field (and no
  column dropping to toggle — see REQ-4).

- REQ-8: **callable `weights` / `metric` + exact `_get_weights` dist==0
  handling** — sklearn accepts `weights ∈ {"uniform","distance"} | callable` and
  `metric ∈ {"nan_euclidean"} | callable` (`_parameter_constraints`,
  `:133-134`). `_get_weights` (sklearn/neighbors/_base.py) returns `None` for
  uniform (equal weights), `1/dist` for distance — and when **any** donor has
  `dist == 0`, those exact-match donors get weight `1.0` and **all others 0.0**.
  ferrolearn's `Distance` arm uses a `1e12` exact-match weight and `1/dist`
  otherwise (`let w = if dist <= epsilon { 1e12 } else { 1/dist }`), which
  **blends** an exact match with the rest rather than zeroing the others, and
  there is no callable-weights / callable-metric / non-`nan_euclidean` metric
  support.

- REQ-9: **Error / clamping contracts** (scoped, with a flagged DIV) — ferrolearn
  `fit` returns `InsufficientSamples` on 0 rows, `InvalidParameter` on
  `n_neighbors == 0`, `InvalidParameter` on `n_neighbors > n_samples`; `transform`
  returns `ShapeMismatch` on a column-count mismatch; the unfitted `transform`
  returns `InvalidParameter`. The `n_neighbors == 0` rejection matches sklearn's
  `Interval(Integral, 1, None)` constraint (`:132`). **FLAG (candidate DIV):
  sklearn does NOT error on `n_neighbors > n_samples` — `transform` CLAMPS via
  `n_neighbors = min(self.n_neighbors, len(potential_donors_idx))` (`:349`)**;
  ferrolearn's `fit` raises `InvalidParameter` there. The critic may pin this.

- REQ-10: **`_BaseImputer` surface + fitted attrs** — sklearn exposes
  `get_feature_names_out` (dropping all-missing columns), the fitted
  `_fit_X` / `_mask_fit_X` / `_valid_mask` / `indicator_` attributes, and the
  `_BaseImputer` / `TransformerMixin` API. ferrolearn exposes only
  `n_train_samples()` plus the unfitted accessors `n_neighbors()` / `weights()`;
  there is no feature-name surface or introspectable fitted-attribute set.

- REQ-11: **PyO3 binding** — `import ferrolearn` exposing a registered
  `KNNImputer` marshalling `fit` / `transform`, the project boundary CPython
  consumer. Absent (no `ferrolearn-python` reference to `KNNImputer`).

- REQ-12: **ferray substrate** — compute over `ferray-core` arrays /
  `ferray-ufunc` rather than `ndarray::Array2` + `num_traits::Float` + `Vec`
  bookkeeping (R-SUBSTRATE).

## Acceptance criteria

- AC-1 (REQ-1): `KNNImputer::<f64>::new(2, Uniform).fit_transform(array![[1,2],
  [3,4],[5,NAN]])` yields `out[[2,1]] == 3.0` (mean of the two present donors),
  matching Probe REQ-1; `Distance` weighting on `[[1,2],[3,6],[4,NAN]]` yields
  the inverse-distance blend `(2/3 + 6) / (1/3 + 1)`. Pinned by
  `test_knn_imputer_uniform_basic`, `test_knn_imputer_distance_weighted`,
  `test_knn_imputer_single_neighbor`, `test_knn_imputer_f32`. **Scope: the
  fixtures share all present features, so DIV-1 does not bite, and every
  receiver has a reachable donor, so DIV-2 does not bite.**

- AC-2 (REQ-2): on the 4-feature Probe REQ-2 fixture (`[[0,0,nan,0],[3,3,10,3],
  [nan,nan,20,5]]`, `n_neighbors=1`), `nan_euclidean_distances` from row 0 is
  `[0, 6.0, 10.0]` (scaled), so col 2 imputes `10.0`; ferrolearn's unscaled
  `partial_euclidean_distance` yields donor distances `5.196` / `5.0`, flips the
  nearest donor, and imputes `20.0` — the SCALING `sqrt(sum_sq * n_features /
  present_count)` (`:543-547`) is required for parity.

- AC-3 (REQ-3): on `[[nan,1],[nan,2],[5,nan]]`, sklearn imputes `out[2,1] = 1.5`
  (masked training-column mean), ferrolearn fills `0.0` (Probe REQ-3); the
  empty-donor branch must use the masked column mean (`:329-337`), not
  `F::zero()`.

- AC-4 (REQ-4): with a training column that is all-`NaN`, sklearn's `transform`
  output has **one fewer column** (`X[:, valid_mask]`, `:289`) unless
  `keep_empty_features`; ferrolearn returns the input shape unchanged.

- AC-5 (REQ-5): `KNNImputer(missing_values=-1)` treats `-1` as missing and
  rejects `NaN` input (`force_all_finite=True`, `:225-228`); ferrolearn has no
  `missing_values` field and only recognizes `NaN`.

- AC-6 (REQ-6): `KNNImputer(add_indicator=True).fit_transform(X)` returns
  `[imputed | indicator]` with extra binary columns (`:280`,`:295`); ferrolearn
  has no indicator output.

- AC-7 (REQ-7): `KNNImputer(keep_empty_features=True)` keeps all-missing columns
  filled with `0` instead of dropping them (`:285-287`); ferrolearn has no such
  toggle.

- AC-8 (REQ-8): a `weights` callable / a `metric` callable is accepted
  (`:133-134`); for `weights="distance"` with an exact match (`dist == 0`), the
  exact donors get weight `1` and **all others 0** (`_get_weights`), not
  ferrolearn's `1e12`-vs-`1/dist` blend.

- AC-9 (REQ-9): `fit` on 0 rows → `Err(InsufficientSamples)`; `n_neighbors == 0`
  → `Err(InvalidParameter)`; a `transform` column mismatch → `Err(ShapeMismatch)`;
  the unfitted `transform` → `Err`. Pinned by `test_knn_imputer_zero_rows_error`,
  `test_knn_imputer_zero_neighbors_error`, `test_knn_imputer_shape_mismatch_error`,
  `test_knn_imputer_unfitted_transform_error`. **FLAG:** `n_neighbors > n_samples`
  → `Err` in ferrolearn (`test_knn_imputer_too_many_neighbors_error`), but sklearn
  CLAMPS (`min(n_neighbors, len(potential_donors))`, `:349`) and does not raise.

- AC-10 (REQ-10): a fitted handle exposes `get_feature_names_out` (dropping
  all-missing columns) and the `_fit_X` / `_mask_fit_X` / `_valid_mask` /
  `indicator_` attributes; ferrolearn exposes only `n_train_samples()`.

- AC-11 (REQ-11): `python3 -c "import ferrolearn; ..."` resolves a registered
  `KNNImputer`; `.fit(X).transform(X)` matches the Probe REQ-1 selection.

- AC-12 (REQ-12): the distance / neighbor-average path computes on `ferray-core`
  arrays.

`## REQ status`

## REQ status

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (KNN imputation shape — partial-distance search + weighted donor average, scoped) | SHIPPED | impl `impl Transform<Array2<F>> for FittedKNNImputer in knn_imputer.rs`: for each row with `has_missing`, computes `partial_euclidean_distance(&row_slice, &train_row)` to every train row (pushed only when `n_valid > 0`), `dists.sort_by(\|a, b\| a.1.partial_cmp(&b.1))`, then per missing feature `j` collects `neighbor_vals` (up to `self.n_neighbors` donors with `!val.is_nan()`) and averages — `Uniform` = `sum / neighbor_vals.len()`, `Distance` = inverse-distance `val_sum / weight_sum`. This mirrors sklearn `_calc_impute` (`_knn.py:184` `argpartition` the `n_neighbors` nearest donors, `:193` `_get_weights`, `:204` masked `np.ma.average`) and the per-(row, col) donor scan (`:319`,`:349-356`). **Value-faithful end-to-end after this iteration's fixes:** distance scaling (REQ-2, #1305), empty-donor column mean (REQ-3, #1306), n_neighbors clamp (REQ-9, #1307), exact-match weighting (#1308), and **inf-distance (no-shared-feature) donor inclusion in the uniform average** (#1309 — `transform` now pushes ALL potential donors with `inf` distance sorted last, so the n_neighbors quota is filled exactly as sklearn's `argpartition`). 33 oracle green guards (8×5 / 10×6 full matrices, k∈{2,5}, uniform+distance, mixed finite/inf donors, f32) value-match within ~1e-9. Generic `F` covers `f32`/`f64`. **Carve-out (#1310, documented edge, NOT-STARTED):** on EXACT distance ties, ferrolearn's stable sort (lowest train index) may pick a different donor than numpy `argpartition`'s unspecified tie order — only at near-ties where sklearn's own choice is arbitrary + ULP float noise; not a meaningful parity target (cf. RFE tie-break, DBSCAN eps-boundary). Non-test consumer: crate re-export `pub use knn_imputer::{FittedKNNImputer, KNNImputer, KNNWeights};` (`ferrolearn-preprocess/src/lib.rs`), the boundary public API (grandfathered S5/R-DEFER-1). Verification: `cargo test -p ferrolearn-preprocess knn_imputer` (`test_knn_imputer_uniform_basic`, `test_knn_imputer_distance_weighted`, `test_knn_imputer_single_neighbor`, `test_knn_imputer_multiple_missing`, `test_knn_imputer_no_missing`, `test_knn_imputer_fit_transform`, `test_knn_imputer_f32`, `test_knn_imputer_default`) → green. |
| REQ-2 (nan_euclidean distance SCALING `sqrt(sum_sq * n_features / present_count)`; DIV-1) | SHIPPED (closed #1305) | `fn partial_euclidean_distance` now returns `sqrt(sum_sq * n_features / n_valid)` (n_features = row length, n_valid = present-in-both count), mirroring sklearn `nan_euclidean_distances` `distances /= present_count; distances *= n_features; sqrt` (`pairwise.py:539-547`). Live oracle (R-CHAR-3): the scaling-flips-donor fixture imputes `10.0` (was `20.0`); a distance-weighted fixture imputes `25.0` (was `28.333`). Guards `divergence_div1_uniform_scaling_flips_donor` + `divergence_div1_distance_scaling_changes_weights` PASS. The `n_features/n_valid` scale is a per-receiver constant that cancels in uniform mean / weighted-ratio when present_count is uniform (so the basic in-module tests stayed green). |
| REQ-3 (empty-donor → masked training column mean; DIV-2) | SHIPPED (closed #1306) | the no-reachable-donor branch (`neighbor_vals` empty or all collected donors inf-distance) now imputes the masked TRAINING COLUMN MEAN — the mean over non-NaN `train_data` column `j` — mirroring sklearn `np.ma.array(self._fit_X[:,col], mask=mask_fit_X[:,col]).mean()` (`_knn.py:329-337`); falls back to 0 only if the training column is entirely NaN (sklearn would drop it, REQ-4 NOT-STARTED). Live oracle: `[[nan,1],[nan,2],[5,nan]]` cell `[2,1]` → `1.5` (was `0.0`). Guard `divergence_div2_empty_donor_imputes_column_mean` PASS. |
| REQ-4 (`valid_mask` all-missing column dropping) | NOT-STARTED | open prereq blocker #1311. ferrolearn's `transform` does `let mut out = x.to_owned();` and never drops a column — the output is always `x.nrows() × n_features`. sklearn drops columns that were all-missing during `fit`: `_valid_mask = ~np.all(_mask_fit_X, axis=0)` (`_knn.py:240`), output `X[:, valid_mask]` (`:289`) with `n_output_features = valid_mask.sum()` unless `keep_empty_features` (`:282-295`,`:310-312`). `_valid_mask` is never computed at `fit` (`fn fit` stores only `train_data`). |
| REQ-5 (`missing_values` non-NaN sentinel) | NOT-STARTED | open prereq blocker #1312. ferrolearn hard-codes `NaN`: every check is `is_nan()` (`partial_euclidean_distance`, `transform`'s `row_slice[j].is_nan()`); there is no `missing_values` field. sklearn parameterizes the sentinel via `_get_mask(X, self.missing_values)` (`_knn.py:239`,`:276`), switching to `force_all_finite=True` for non-NaN markers (`:225-228`). |
| REQ-6 (`add_indicator` MissingIndicator concatenation) | NOT-STARTED | open prereq blocker #1313. No `add_indicator` field or indicator output in `KNNImputer<F>`/`FittedKNNImputer<F>`. sklearn fits a `MissingIndicator` (`super()._fit_indicator(self._mask_fit_X)`, `:242`) and concatenates its binary columns onto the output (`super()._concatenate_indicator(Xc, X_indicator)`, `:280`,`:295`). |
| REQ-7 (`keep_empty_features`) | NOT-STARTED | open prereq blocker #1314. No `keep_empty_features` field; and since ferrolearn never drops all-missing columns (REQ-4), there is nothing to toggle. sklearn keeps such columns filled with `0` when `keep_empty_features=True` (`Xc[:, ~valid_mask] = 0`, `_knn.py:285-287`,`:310-312`). |
| REQ-8 (callable `weights`/`metric`) | NOT-STARTED | open prereq blocker #1315. **The exact `_get_weights` dist==0 handling was FIXED (#1308):** the `Distance` arm now gives exact-match (`dist <= epsilon`) donors the uniform mean of ONLY those donors (weight 1, all others 0), matching sklearn `_get_weights` (neighbors/_base.py:119-121) — the `1e12` blend was removed (guard `divergence_div4_exact_match_weights_leak` PASS, e.g. exact match imputes `0.0` exactly). **Remaining NOT-STARTED:** ferrolearn's `KNNWeights` enum is `{Uniform, Distance}` only — no callable `weights`, no callable / non-`nan_euclidean` `metric` (sklearn `_parameter_constraints` `weights:[StrOptions,callable,Hidden(None)]`, `metric:[StrOptions(_NAN_METRICS),callable]`, `_knn.py:133-134`). |
| REQ-9 (error / clamping contracts, scoped + flagged DIV) | SHIPPED | `fn fit in knn_imputer.rs` returns `Err(FerroError::InsufficientSamples { required: 1, actual: 0, context: "KNNImputer::fit" })` on `n_samples == 0`, `Err(InvalidParameter { name: "n_neighbors", .. })` on `self.n_neighbors == 0`. **(The `n_neighbors > n_samples` error arm was REMOVED — DIV-3 #1307.)** `impl Transform for FittedKNNImputer` returns `Err(FerroError::ShapeMismatch { context: "FittedKNNImputer::transform", .. })` on `x.ncols() != n_features`; `impl Transform for KNNImputer` (unfitted) returns `Err(InvalidParameter)`. The `n_neighbors == 0` rejection matches sklearn `Interval(Integral, 1, None)` (`_knn.py:132`). Non-test consumer: these guards protect every instance reached through the crate re-export (`lib.rs`). Verification: `cargo test -p ferrolearn-preprocess knn_imputer` (`test_knn_imputer_zero_rows_error`, `test_knn_imputer_zero_neighbors_error`, `test_knn_imputer_shape_mismatch_error`, `test_knn_imputer_unfitted_transform_error`) → green. **DIV-3 (#1307) FIXED:** sklearn does NOT raise on `n_neighbors > n_samples` — it clamps `n_neighbors = min(self.n_neighbors, len(potential_donors_idx))` (`:349`); the error arm was removed, `fit` returns `Ok`, transform clamps. In-module `test_knn_imputer_too_many_neighbors_error` rewritten to `test_knn_imputer_too_many_neighbors_ok` (R-HONEST-4); guard `divergence_div3_n_neighbors_gt_n_samples_does_not_error` PASS. |
| REQ-10 (`_BaseImputer` surface + fitted attrs) | NOT-STARTED | open prereq blocker #1316. `FittedKNNImputer<F>` exposes only `n_train_samples()`; `KNNImputer<F>` exposes `n_neighbors()` / `weights()`. No `get_feature_names_out` (which drops all-missing columns), no introspectable `_fit_X` / `_mask_fit_X` / `_valid_mask` / `indicator_` attributes that sklearn's `_BaseImputer` + `TransformerMixin` provide (`_knn.py:238-242`). |
| REQ-11 (PyO3 binding) | NOT-STARTED | open prereq blocker #1317. No `ferrolearn-python` registration of `KNNImputer` (grep across `ferrolearn-python/src/*.rs` finds no `KNNImputer`/`knn_imputer`); the only non-test consumer is the crate re-export (`lib.rs`, `pub use knn_imputer::{...}`). The boundary CPython `import ferrolearn` imputer surface is absent. |
| REQ-12 (ferray substrate) | NOT-STARTED | open prereq blocker #1318. The distance / neighbor-average path uses `ndarray::Array2` (`self.train_data.row(t).to_vec()`, `out.to_owned()`, indexing `out[[i, j]]`) + `num_traits::Float` and `Vec` bookkeeping (`dists`, `neighbor_vals`) — not `ferray-core` / `ferray-ufunc` (R-SUBSTRATE-1/2). |

## Architecture

**ferrolearn (existing).** `knn_imputer.rs` exposes `KNNWeights { Uniform,
Distance }`, the unfitted `KNNImputer<F> { n_neighbors: usize, weights:
KNNWeights, _marker: PhantomData<F> }` (`new(n_neighbors, weights)`, `Default` =
`(5, Uniform)`, accessors `n_neighbors()` / `weights()`), the fitted
`FittedKNNImputer<F> { train_data: Array2<F>, n_neighbors, weights }` (accessor
`n_train_samples()`), and a private `partial_euclidean_distance<F>(row_a, row_b)
-> (F, usize)` returning `(sqrt(sum_sq), n_valid)` over features present in both
rows (`(F::infinity(), 0)` when `n_valid == 0`). `Fit::fit` validates `n_samples
== 0` → `InsufficientSamples`, `n_neighbors == 0` and `n_neighbors > n_samples`
→ `InvalidParameter`, then stores `train_data = x.to_owned()`. `Transform::
transform` copies `x`, and for each row containing a `NaN`: computes distances
to all train rows (keeping `n_valid > 0`), sorts ascending, and for each missing
feature `j` collects up to `n_neighbors` donors with feature `j` present,
averaging via `Uniform` (mean) or `Distance` (inverse-distance with `epsilon =
1e-12`, `1e12` exact-match weight) — **filling `F::zero()` when no donor is
reachable** (DIV-2). The unfitted `Transform` returns `InvalidParameter`;
`FitTransform` chains `fit` then `transform`. The crate re-exports all three
types (`lib.rs`, `pub use knn_imputer::{FittedKNNImputer, KNNImputer,
KNNWeights}`); there is no PyO3 binding.

**sklearn (target contract).** `KNNImputer(_BaseImputer)` (`_knn.py:69`) takes
`__init__(*, missing_values=np.nan, n_neighbors=5, weights="uniform",
metric="nan_euclidean", copy=True, add_indicator=False,
keep_empty_features=False)` (`:138-157`) under `_parameter_constraints`
(`n_neighbors` `Interval(Integral, 1, None)`, `weights`
`{"uniform","distance"}|callable`, `metric` `{"nan_euclidean"}|callable`,
`:130-136`). `fit` (`:206-244`) validates `force_all_finite="allow-nan"` for a
NaN sentinel (`:225-228`), stores `_fit_X`, `_mask_fit_X = _get_mask(_fit_X,
missing_values)` (`:239`), `_valid_mask = ~np.all(_mask_fit_X, axis=0)` (`:240`),
and fits the missing indicator (`:242`). `transform` (`:246-362`) drops
all-missing columns (`X[:, valid_mask]` unless `keep_empty_features`,
`:282-295`,`:310-312`); per (row, col with `valid_mask`) it finds
`potential_donors_idx` (rows with `col` present, `:319`); **receivers with
all-NaN distances get the masked column mean** (`:329-337`, DIV-2); otherwise
`_calc_impute` (`:159-204`) `argpartition`s the `n_neighbors = min(self.
n_neighbors, len(potential_donors_idx))` (`:349`) nearest donors, builds the
weight matrix via `_get_weights` (uniform = equal; distance = `1/dist`, exact
matches weight `1` / others `0`), and masked-averages the donor values (`:204`).
Distances come from `nan_euclidean_distances` (`pairwise.py:430`): `dist =
sqrt(sum_sq * n_features / present_count)` (`:543-547`, the SCALING = DIV-1),
with all-missing-overlap pairs set to `NaN` (`:540`).

**The structural gap.** ferrolearn matches sklearn on the *neighbor-search +
weighted-average shape* (REQ-1, scoped) and the *scoped error contracts* (REQ-9),
but **two value divergences bite the moment the fixture is non-trivial**: the
distance is **unscaled** (DIV-1, REQ-2 — `sqrt(sum_sq)` vs `sqrt(sum_sq *
n_features / present_count)`, which flips the neighbor ordering when donors
overlap the receiver on differing feature counts), and the **empty-donor
fallback fills `0` instead of the masked column mean** (DIV-2, REQ-3). On top of
that, every contract sklearn layers on the `_BaseImputer` machinery is
NOT-STARTED: all-missing-column dropping (REQ-4), the `missing_values` sentinel
(REQ-5), `add_indicator` (REQ-6), `keep_empty_features` (REQ-7), callable
`weights`/`metric` + exact `_get_weights` semantics (REQ-8), the
`get_feature_names_out` / fitted-attribute surface (REQ-10), the PyO3 binding
(REQ-11), and the ferray substrate (REQ-12). The `n_neighbors > n_samples`
boundary (REQ-9 flag) is a candidate divergence: ferrolearn raises, sklearn
clamps (`:349`). This is a **mostly-NOT-STARTED** unit (2 SHIPPED / 10
NOT-STARTED).

## Verification

Commands establishing the SHIPPED claims (REQ-1 scoped, REQ-9 scoped):

```bash
# Crate gauntlet — REQ-1 (neighbor-search + weighted-average shape on no-bite fixtures),
# REQ-9 (error contracts):
cargo test -p ferrolearn-preprocess knn_imputer   # incl. test_knn_imputer_uniform_basic,
                                                  #       test_knn_imputer_distance_weighted,
                                                  #       test_knn_imputer_single_neighbor,
                                                  #       test_knn_imputer_multiple_missing,
                                                  #       test_knn_imputer_no_missing,
                                                  #       test_knn_imputer_fit_transform,
                                                  #       test_knn_imputer_f32,
                                                  #       test_knn_imputer_default,
                                                  #       test_knn_imputer_zero_rows_error,
                                                  #       test_knn_imputer_zero_neighbors_error,
                                                  #       test_knn_imputer_shape_mismatch_error,
                                                  #       test_knn_imputer_unfitted_transform_error,
                                                  #       test_knn_imputer_too_many_neighbors_error
cargo clippy -p ferrolearn-preprocess --all-targets -- -D warnings
cargo fmt --all --check

# Oracle (Probe REQ-1) — basic uniform impute (no DIV-1/DIV-2 bite); matches ferrolearn 3.0:
python3 -c "import numpy as np; from sklearn.impute import KNNImputer; nan=np.nan; \
print(KNNImputer(n_neighbors=2).fit_transform([[1,2],[3,4],[5,nan]]).tolist())"
#   -> [[1.0, 2.0], [3.0, 4.0], [5.0, 3.0]]

# Oracle (Probe REQ-2 / DIV-1) — the SCALING flips the nearest donor; sklearn 10.0 vs ferrolearn 20.0:
python3 -c "import numpy as np; from sklearn.impute import KNNImputer; \
from sklearn.metrics.pairwise import nan_euclidean_distances; nan=np.nan; \
X=np.array([[0.,0.,nan,0.],[3.,3.,10.,3.],[nan,nan,20.,5.]]); \
print('dist', np.round(nan_euclidean_distances(X[[0]],X),5).tolist(), 'n1col2', KNNImputer(n_neighbors=1).fit_transform(X)[0,2])"
#   -> dist [[0.0, 6.0, 10.0]] n1col2 10.0   (ferrolearn raw 5.196/5.0 -> picks 20.0 — DIV-1, REQ-2)

# Oracle (Probe REQ-3 / DIV-2) — empty-donor -> masked column mean; sklearn 1.5 vs ferrolearn 0.0:
python3 -c "import numpy as np; from sklearn.impute import KNNImputer; nan=np.nan; \
print(KNNImputer(n_neighbors=2).fit_transform([[nan,1.],[nan,2.],[5.,nan]]).tolist())"
#   -> [[5.0, 1.0], [5.0, 2.0], [5.0, 1.5]]   (ferrolearn fills 0.0 — DIV-2, REQ-3)
```

The existing `#[test]`s exercise REQ-1 (uniform / distance / single-neighbor /
multi-missing / no-missing on fixtures where **all rows share all present
features**, so DIV-1 does not change the ordering, and every receiver has a
reachable donor, so DIV-2 never fires) and REQ-9 (every error path). They are
**no-bite-fixture-grounded, not full nan_euclidean oracle-grounded** — by
construction, since the distance scaling (REQ-2) and empty-donor fallback
(REQ-3) diverge. No currently-green command establishes REQ-2..REQ-8,
REQ-10..REQ-12. Note `test_knn_imputer_too_many_neighbors_error` asserts `Err`
for `n_neighbors > n_samples`, which sklearn does NOT raise (it clamps,
`_knn.py:349`) — the critic should pin that boundary (REQ-9 flag).

## Blockers

REQ-1, REQ-2, REQ-3, REQ-9 are SHIPPED (this iteration value-matched the
imputation surface). The remaining NOT-STARTED REQs are open `-l blocker` issues:

- #1305 — REQ-2 (DIV-1, CLOSED/fixed): `partial_euclidean_distance` now scales
  `sqrt(sum_sq * n_features / n_valid)` (`pairwise.py:539-547`).
- #1306 — REQ-3 (DIV-2, CLOSED/fixed): empty-donor branch now imputes the masked
  training column mean (`_knn.py:329-337`).
- #1307 — DIV-3 (CLOSED/fixed): removed the `n_neighbors > n_samples` error
  (sklearn clamps, `_knn.py:349`).
- #1308 — DIV-4 (CLOSED/fixed): exact-match (`dist==0`) donors weight 1, others 0
  (was a `1e12` blend) (`neighbors/_base.py:119-121`).
- #1309 — DIV-5 (CLOSED/fixed): inf-distance (no-shared-feature) donors are now
  included in the uniform average, filling the `n_neighbors` quota like sklearn
  `argpartition` (`_knn.py:184-204`).
- #1310 — DIV-6 (NOT-STARTED edge, documented carve-out): exact-distance-tie donor
  selection — numpy `argpartition` unspecified tie order + ULP float noise; not a
  meaningful parity target (cf. RFE tie-break, DBSCAN eps-boundary). No committed
  failing test (R-DEFER-3).
- #1311 — REQ-4: `transform` never drops columns; sklearn drops all-missing-at-fit
  columns (`_valid_mask`, `X[:, valid_mask]`, `:240`,`:289`) — `_valid_mask` is not
  computed at `fit`.
- #1312 — REQ-5: no `missing_values` field; the sentinel is hard-coded `NaN`
  (`is_nan()` checks) vs `_get_mask(X, missing_values)` (`:239`,`:276`).
- #1313 — REQ-6: no `add_indicator` / MissingIndicator concatenation (`:242`,
  `:280`,`:295`).
- #1314 — REQ-7: no `keep_empty_features` (`:285-287`,`:310-312`).
- #1315 — REQ-8: `KNNWeights` is `{Uniform, Distance}` only — no callable
  `weights`/`metric`, and the `Distance` `1e12`-exact-match blend differs from
  `_get_weights`' weight-1 / others-0 dist==0 handling (`:133-134`).
- #1316 — REQ-10: no `get_feature_names_out` / introspectable `_fit_X` /
  `_mask_fit_X` / `_valid_mask` / `indicator_` surface (`:238-242`).
- #1317 — REQ-11: no `ferrolearn-python` `KNNImputer` binding (boundary CPython
  consumer absent; grep matches nothing under `ferrolearn-python/src/`).
- #1318 — REQ-12: distance / neighbor-average path on `ndarray`/`num_traits`/`Vec`,
  not ferray (R-SUBSTRATE-1/2).
```
