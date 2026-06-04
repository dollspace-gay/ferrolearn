# SimpleImputer

<!--
tier: 3-component
status: shipped-partial
baseline-commit: ca206d44
upstream: scikit-learn 1.5.2
upstream-paths:
  - sklearn/impute/_base.py  # _BaseImputer(TransformerMixin, BaseEstimator) (:73). _most_frequent(array, extra_value, n_repeat) (:36-71): if array.size>0 numeric branch mode=_mode(array); most_frequent_value=mode[0][0] (:53-56); object branch Counter, tie -> min (:42-52); empty -> value=0,count=0 (:57-59); count==0 & n_repeat==0 -> np.nan (:62-63); tie count==n_repeat -> min(most_frequent_value, extra_value) (:68-70). class SimpleImputer(_BaseImputer) (:147). __init__(*, missing_values=np.nan, strategy="mean", fill_value=None, copy=True, add_indicator=False, keep_empty_features=False) (:288-304); fill_value=None -> 0 for numeric (:425-427). _dense_fit(X, strategy, missing_values, fill_value) (:489-552): missing_mask=_get_mask(X, missing_values) (:491); masked_X=ma.masked_array (:492); mean=np.ma.mean(masked_X, axis=0), masked entries -> 0 if keep_empty_features else np.nan (:498,500-501); median=np.ma.median, masked -> 0 if keep_empty_features else np.nan (:507,510-512); most_frequent per col _most_frequent(row, np.nan, 0), empty & keep_empty_features -> 0 (:531-537); constant np.full(X.shape[1], fill_value) (:545). transform (:554-639): check_is_fitted (:568); X.shape[1] != statistics_.shape[0] -> ValueError (:573-577); strategy=="constant" or keep_empty_features -> keep all (:583-585) else invalid_mask=_get_mask(statistics, np.nan); valid -> X=X[:, valid_statistics_indexes] + "Skipping features without any observed values" warning (:586-603); dense impute (:625-635). inverse_transform (:641) requires add_indicator. MissingIndicator (separate _BaseImputer subclass): features="missing-only"|"all" binary mask of missing positions.
ferrolearn-module: ferrolearn-preprocess/src/imputer.rs
parity-ops: SimpleImputer, MissingIndicator
crosslink-issue: 1363
-->

## Summary

scikit-learn's `SimpleImputer` (`_base.py:147`) replaces missing values in each
feature column with a per-column statistic learned at fit time — the column mean,
median, most-frequent value, or a user-supplied constant
(`_dense_fit:489-552`) — storing the results in `statistics_` and substituting them
for missing entries at `transform` (`:554-639`). It mirrors `MissingIndicator`'s
sibling estimator surface (`add_indicator`, `inverse_transform`) from the shared
`_BaseImputer` (`:73`).

`ferrolearn-preprocess/src/imputer.rs` ships the **per-column fill-value core plus
the sklearn-default all-NaN-column drop** over dense `Array2<F>` with NaN as the
sole missing sentinel: `SimpleImputer<F> { strategy: ImputeStrategy<F> }` (`new`,
`strategy()`) with `ImputeStrategy::{Mean, Median, MostFrequent, Constant(F)}`
produces `FittedSimpleImputer<F> { fill_values: Array1<F>, kept_indices: Vec<usize> }`
(`fill_values()`, `kept_indices()`).
`Fit::fit` (`fit in imputer.rs`) collects the non-NaN values of each column,
computes the chosen statistic (`Mean` = `sum/n`, `Median` = `median_of`,
`MostFrequent` = `most_frequent_of` with tie → smallest, `Constant(c)` = `c`), and
`Transform::transform` (`transform in imputer.rs`) substitutes `fill_values[j]` for
every NaN in column `j`.

The former HEADLINE divergence DIV-1 (#1364) is now **RESOLVED**: for a column
that is entirely NaN at fit time under `Mean`/`Median`/`MostFrequent`, ferrolearn
now mirrors sklearn's DEFAULT (`keep_empty_features=False`) by setting
`fill_values[j] = F::nan()` (its `statistics_` analogue), excluding `j` from
`kept_indices`, and **DROPPING** that column at `transform`
(`:501,510-512,534-537,586-603`); under `Constant`, every column (including
all-NaN ones) is KEPT and filled with the user constant (`:545,583`). Beyond the
now-resolved DIV-1, ferrolearn still has **no `MissingIndicator`, no
`inverse_transform`, no `add_indicator`, no `missing_values` param (NaN-only), no
`keep_empty_features` param, no `statistics_` attribute name, no `fill_value=None`
numeric default / `copy` param, no string/object dtype, no sparse `_sparse_fit`, no
`get_feature_names_out` / `n_features_in_`, and no PyO3 binding**. This is a
**shipped-partial** unit: **3 SHIPPED** (REQ-1 fill values for columns with ≥1
observed value, REQ-2 the all-NaN-column default-DROP, REQ-3 scoped error /
parameter contracts) / **10 NOT-STARTED** (REQ-4 `keep_empty_features`, REQ-5
`missing_values`, REQ-6 `add_indicator` + `MissingIndicator`, REQ-7
`inverse_transform`, REQ-8 `fill_value=None` default + `statistics_` name + `copy`,
REQ-9 string/object dtype, REQ-10 sparse `_sparse_fit`, REQ-11
`get_feature_names_out` / feature-name introspection, REQ-12 PyO3, REQ-13 ferray
substrate).

## Probes (live sklearn oracle, 1.5.2)

All values below are live output from `python3` against scikit-learn 1.5.2, run
from `/tmp`. They pin the fill-value algebra (REQ-1) and the all-NaN-column
default-drop (REQ-2, formerly DIV-1, now RESOLVED — ferrolearn matches these).

```bash
# PROBE 1 (REQ-2, DIV-1 HEADLINE) — all-NaN column DEFAULT (keep_empty_features=False)
# is DROPPED for strategy='mean' (transform:586-603) and statistics_ for that col is np.nan:
python3 -c "import numpy as np
from sklearn.impute import SimpleImputer
X=np.array([[1.,np.nan],[3.,np.nan],[5.,np.nan]])
si=SimpleImputer(strategy='mean'); out=si.fit_transform(X)
print('out.shape=',out.shape,'statistics_=',si.statistics_.tolist(),'out=',out.tolist())"
#   -> UserWarning: Skipping features without any observed values: [1]. ...
#   -> out.shape= (3, 1) statistics_= [3.0, nan] out= [[1.0], [3.0], [5.0]]
#   ferrolearn SimpleImputer::new(ImputeStrategy::Mean).fit_transform(same X) NOW MATCHES:
#     fill_values()=[3.0, nan], kept_indices()=[0], out.shape=(3, 1)  (col1 DROPPED — DIV-1 resolved).

# PROBE 2 (REQ-4) — keep_empty_features=True fills the all-NaN col with 0 and KEEPS it
# (transform:583-585, _dense_fit:501) — the toggle ferrolearn does NOT expose:
python3 -c "import numpy as np
from sklearn.impute import SimpleImputer
X=np.array([[1.,np.nan],[3.,np.nan],[5.,np.nan]])
out=SimpleImputer(strategy='mean',keep_empty_features=True).fit_transform(X)
print('out.shape=',out.shape,'out=',out.tolist())"
#   -> out.shape= (3, 2) out= [[1.0, 0.0], [3.0, 0.0], [5.0, 0.0]]
#   ferrolearn now implements the sklearn DEFAULT (drop, Probe 1) and has no
#   `keep_empty_features=True` toggle to reproduce THIS keep+fill-0 behavior (REQ-4, #1365).

# PROBE 3 (REQ-2/REQ-4) — strategy='constant' keeps the all-NaN col regardless
# (transform:583, _dense_fit:545):
python3 -c "import numpy as np
from sklearn.impute import SimpleImputer
X=np.array([[1.,np.nan],[3.,np.nan],[5.,np.nan]])
out=SimpleImputer(strategy='constant',fill_value=-1).fit_transform(X)
print('out.shape=',out.shape,'out=',out.tolist())"
#   -> out.shape= (3, 2) out= [[1.0, -1.0], [3.0, -1.0], [5.0, -1.0]]
#   ferrolearn ImputeStrategy::Constant(-1.0) also keeps + fills -1.0  (AGREES for constant).

# PROBE 4 (REQ-1) — fill VALUES on a column with >=1 observed value
# mean = np.ma.mean (_dense_fit:498), median = np.ma.median (:507):
python3 -c "import numpy as np
from sklearn.impute import SimpleImputer
X=np.array([[1.,np.nan],[3.,4.],[5.,6.]])
print('mean=',SimpleImputer(strategy='mean').fit(X).statistics_.tolist())
print('median=',SimpleImputer(strategy='median').fit(X).statistics_.tolist())
print('median even [1,3,5,7]=',SimpleImputer(strategy='median').fit(np.array([[1.],[3.],[5.],[7.]])).statistics_.tolist())"
#   -> mean= [3.0, 5.0]   median= [3.0, 5.0]   median even [1,3,5,7]= [4.0]
#   ferrolearn Mean (sum/n) = [3.0, 5.0]; Median (median_of, avg of two middle for even n) matches:
#     col0 [1,3,5]->3, col1 [4,6]->5; [1,3,5,7]->(3+5)/2=4.0  (IDENTICAL).

# PROBE 5 (REQ-1) — most_frequent: scipy _mode, tie broken to the SMALLEST value
# (_most_frequent:53-56,68-70):
python3 -c "import numpy as np
from sklearn.impute import SimpleImputer
print('tie [1,1,3,3]=',SimpleImputer(strategy='most_frequent').fit(np.array([[1.],[1.],[3.],[3.]])).statistics_.tolist())
print('plain [1,2,2,3]=',SimpleImputer(strategy='most_frequent').fit(np.array([[1.],[2.],[2.],[3.]])).statistics_.tolist())"
#   -> tie [1,1,3,3]= [1.0]   plain [1,2,2,3]= [2.0]
#   ferrolearn most_frequent_of (count runs, tie -> smallest via strict >) = [1.0] / [2.0]  (IDENTICAL).

# PROBE 6 (REQ-2) — statistics_ for an all-NaN col is np.nan under every non-constant strategy
# (_dense_fit:501,510-512,534-537):
python3 -c "import numpy as np
from sklearn.impute import SimpleImputer
Xn=np.array([[np.nan],[np.nan]])
for s in ('mean','median','most_frequent'):
    print(s,'all-NaN statistics_=',SimpleImputer(strategy=s).fit(Xn).statistics_.tolist())"
#   -> mean all-NaN statistics_= [nan]   median ...= [nan]   most_frequent ...= [nan]
#   ferrolearn NOW sets fill_values[j]=F::nan() for the all-NaN col and excludes it from kept_indices -> DROPPED (matches).

# PROBE 7 (REQ-8) — fill_value=None for a numeric constant resolves to 0 (_base.py:425-427):
python3 -c "import numpy as np
from sklearn.impute import SimpleImputer
print('constant fill_value=None =',SimpleImputer(strategy='constant').fit(np.array([[1.,np.nan]])).statistics_.tolist())"
#   -> constant fill_value=None = [0.0, 0.0]
#   ferrolearn ImputeStrategy::Constant requires an explicit F (no None default).
```

## Requirements

- REQ-1: **Per-column fill values for columns with at least one observed value**
  (HEADLINE, SHIPPED). For `Mean`/`Median`/`MostFrequent`/`Constant`, compute the
  per-column statistic over the non-missing entries exactly as `_dense_fit`
  (`:489-552`): `Mean` = `np.ma.mean(masked_X, axis=0)` (`:498`), `Median` =
  `np.ma.median(masked_X, axis=0)` (`:507`), `MostFrequent` = `_most_frequent`
  (scipy `_mode`, tie → smallest, `:53-56,68-70`), `Constant(c)` = `np.full(...,
  fill_value)` (`:545`); then substitute that value for every missing entry at
  `transform`. ferrolearn's `fit in imputer.rs` collects `col.iter().filter(|v|
  !v.is_nan())`, computes `Mean` as `sum/n`, `Median` via `median_of` (avg of two
  middle for even `n`, matching `np.ma.median`), `MostFrequent` via
  `most_frequent_of` (sort ascending, count runs, tie → smallest via strict `>`),
  `Constant(c)` = `c`; `transform in imputer.rs` replaces each NaN with
  `fill_values[j]` for the surviving columns. The all-NaN sub-case is explicitly
  **NOT** part of REQ-1 — see REQ-2 (Probes 1, 6). Pinned by 16 in-module tests +
  the green-guard value probes in `tests/divergence_imputer.rs`.

- REQ-2: **All-NaN column handling — sklearn DEFAULT drops the column** (formerly
  DIV-1 HEADLINE, now SHIPPED). For a fully-missing column under
  `Mean`/`Median`/`MostFrequent` with the default `keep_empty_features=False`,
  sklearn sets `statistics_` to `np.nan` (`:501,510-512`; for `MostFrequent`
  `_most_frequent([], np.nan, 0) = np.nan`, `:534-537,62-63`) and at `transform`
  builds `invalid_mask = _get_mask(statistics, np.nan)` and **drops** those columns
  (`X = X[:, valid_statistics_indexes]`, emitting "Skipping features without any
  observed values", `:586-603`). ferrolearn now matches: `fit in imputer.rs` sets
  `fill_values[j] = F::nan()` (its `statistics_` analogue) for an all-NaN
  non-constant column and **excludes `j` from `kept_indices`**, and `transform in
  imputer.rs` projects the output onto `kept_indices` (`X = X[:,
  valid_statistics_indexes]`, `:586-603`), so the all-NaN column is DROPPED. Under
  `Constant`, every column is KEPT and filled with the user constant (`:545,583`),
  also matching sklearn (Probes 1, 3, 6). This was DIV-1 (#1364), now **RESOLVED**;
  verified across column-order / all-dropped / MostFrequent / Constant-keep /
  `statistics_`-NaN / separate-matrix / f32 fixtures in
  `tests/divergence_imputer.rs`.

- REQ-3: **Error / parameter contracts** (scoped, SHIPPED). `fit in imputer.rs`
  returns `InsufficientSamples { required: 1, actual: 0, .. }` when `n_samples ==
  0` (mirroring sklearn's "0 sample(s)" validation at fit); `transform in
  imputer.rs` returns `ShapeMismatch` when `x.ncols()` differs from the fitted
  feature count (mirroring `transform`'s `X.shape[1] != statistics_.shape[0]`
  `ValueError`, `:573-577`); the unfitted `Transform for SimpleImputer` returns
  `InvalidParameter` (the "must fit first" guard, mirroring `check_is_fitted`,
  `:568`). Scoped to the contracts ferrolearn actually enforces over the dense
  NaN-only API.

- REQ-4: **`keep_empty_features` parameter** (NOT-STARTED). sklearn's ctor accepts
  `keep_empty_features=False` (`:296`); when `True`, all-NaN columns are filled with
  `0` and KEPT (`_dense_fit:501,510-512,534-537`; `transform:583-585`). ferrolearn
  has **no such param**; ferrolearn unconditionally implements the
  `keep_empty_features=False` default drop (REQ-2) and cannot toggle to the
  keep-and-fill-0 behavior. Open prereq blocker #1365.

- REQ-5: **`missing_values` parameter (non-NaN sentinel / None / str)**
  (NOT-STARTED). sklearn's ctor accepts `missing_values=np.nan` (`:291`); the
  sentinel may be any scalar (e.g. `-1`, `0`, `None`, or a string), and
  `missing_mask = _get_mask(X, missing_values)` (`_dense_fit:491`, `transform:580`)
  matches against it. ferrolearn detects missingness **only** via `v.is_nan()`
  (`fit`/`transform in imputer.rs`) — no configurable sentinel. Open prereq blocker #1366.

- REQ-6: **`add_indicator` + `MissingIndicator` estimator** (NOT-STARTED, route
  parity_op). sklearn's `add_indicator=False` (`:295`) appends a binary
  missing-mask via `_BaseImputer._fit_indicator` / `_concatenate_indicator`
  (`:485,637-639`), backed by the separate `MissingIndicator` class
  (`features="missing-only"|"all"`). ferrolearn has **no `MissingIndicator` and no
  `add_indicator`** — this estimator is ABSENT and requires acto-builder in a
  separate iteration. Open prereq blocker #1367.

- REQ-7: **`inverse_transform`** (NOT-STARTED). sklearn's `SimpleImputer.inverse_transform`
  (`:641`) reverses `transform` for rows recoverable via the `add_indicator` mask.
  ferrolearn exposes **no `inverse_transform`** (and, lacking `add_indicator`, has
  no mask to invert against). Open prereq blocker #1368.

- REQ-8: **`fill_value=None` numeric default + `statistics_` attribute name +
  `copy` param** (NOT-STARTED). sklearn's `fill_value=None` resolves to `0` for
  numeric data (`_base.py:425-427`, Probe 7), the learned per-column values live in
  the `statistics_` attribute, and `copy=True` (`:294`) governs in-place vs copy
  transform. ferrolearn requires an explicit `F` in `ImputeStrategy::Constant`, names
  the accessor `fill_values()` (not `statistics_`), and always allocates a fresh
  output. Open prereq blocker #1369.

- REQ-9: **String / object dtype** (NOT-STARTED). sklearn supports
  `strategy='most_frequent'` / `'constant'` on object / string arrays (the object
  branch of `_most_frequent`, `:42-52`; `np.full(fill_value)` with a string
  `:545`). ferrolearn is generic over `F: Float` only — **no non-numeric dtype**.
  Open prereq blocker #1370.

- REQ-10: **Sparse `_sparse_fit`** (NOT-STARTED). sklearn dispatches sparse input
  to `_sparse_fit` (`:444`) and imputes over `X.data` (`transform:606-624`).
  ferrolearn operates only on dense `ndarray::Array2<F>` — **no sparse path**.
  Open prereq blocker #1371.

- REQ-11: **`get_feature_names_out` + `n_features_in_` / `feature_names_in_`**
  (NOT-STARTED). sklearn's `_BaseImputer` (via `TransformerMixin` /
  `BaseEstimator`) exposes `get_feature_names_out`, `n_features_in_`, and
  `feature_names_in_` (the latter used to name the dropped invalid features,
  `transform:596-597`). ferrolearn's `FittedSimpleImputer` exposes only
  `fill_values()` / `kept_indices()` — **no feature-name introspection**. Open prereq blocker #1372.

- REQ-12: **PyO3 binding** (NOT-STARTED). There is no `_RsSimpleImputer` CPython
  binding in `ferrolearn-python` — `grep -rn "SimpleImputer\|Imputer"
  ferrolearn-python/src` finds none — so the imputer is unreachable from Python.
  Open prereq blocker #1373.

- REQ-13: **ferray substrate** (NOT-STARTED). Compute the per-column statistics and
  the NaN substitution over `ferray-core` arrays rather than `ndarray::Array2` /
  `Array1`, `num_traits::Float`, and the per-column `Vec<F>` collect path
  (R-SUBSTRATE). Open prereq blocker #1374.

## Acceptance criteria

- AC-1 (REQ-1): `SimpleImputer::<f64>::new(ImputeStrategy::Mean).fit(array![[1.,
  NAN],[3.,4.],[5.,6.]])` yields `fill_values() == [3.0, 5.0]` (Probe 4);
  `ImputeStrategy::Median` on `[[1],[3],[5],[7]]` yields `[4.0]` (avg of two middle,
  Probe 4); `ImputeStrategy::MostFrequent` on `[[1],[1],[3],[3]]` yields `[1.0]`
  (tie → smallest, Probe 5); `ImputeStrategy::Constant(-99.0)` fills `-99.0`; the
  transformed output replaces NaN with the learned value and leaves observed values
  untouched. Pinned by `test_mean_basic`, `test_mean_no_nan`,
  `test_mean_multiple_nans_same_column`, `test_median_odd_count`,
  `test_median_even_count`, `test_median_with_nan`, `test_most_frequent_basic`,
  `test_most_frequent_tie_chooses_smallest`, `test_most_frequent_with_nan`,
  `test_constant_strategy`, `test_fit_transform_equivalence`, `test_f32_imputer`,
  `test_multi_column_mixed_nan` (in-module).

- AC-2 (REQ-2): `SimpleImputer(strategy='mean').fit_transform([[1,nan],
  [3,nan],[5,nan]])` → sklearn `out.shape == (3, 1)` (col1 DROPPED), `statistics_ ==
  [3.0, nan]` (Probe 1); ferrolearn now MATCHES — `fill_values() == [3.0, nan]`,
  `kept_indices() == [0]`, `out.shape == (3, 1)`, surviving col is `[1,3,5]`
  (`divergence_mean_all_nan_column_dropped`, `test_mean_all_nan_column_dropped`).
  All columns all-NaN → `out.shape == (n, 0)`
  (`reaudit_b_all_columns_all_nan_zero_output`); `Constant` KEEPS+fills the all-NaN
  col with the constant, not 0 (`reaudit_d_constant_all_nan_kept_filled_constant`).
  → REQ-2 SHIPPED.

- AC-3 (REQ-3): `SimpleImputer::<f64>::new(Mean).fit(Array2::zeros((0, 3)))` returns
  `Err(InsufficientSamples)`; a fitted handle's `transform` on a wrong column count
  returns `Err(ShapeMismatch)` (sklearn `:573-577`); calling `transform` on the
  unfitted `SimpleImputer` returns `Err(InvalidParameter)`. Pinned by
  `test_fit_zero_rows_error`, `test_transform_shape_mismatch_error`,
  `test_unfitted_transform_error`.

- AC-4 (REQ-4): `SimpleImputer(strategy='mean', keep_empty_features=True)` keeps the
  all-NaN col filled with `0` (Probe 2); ferrolearn has no `keep_empty_features`
  param to toggle.

- AC-5 (REQ-5): `SimpleImputer(missing_values=-1).fit_transform([[−1, 2],[3, −1]])`
  treats `-1` as missing (`_get_mask`, `:491`); ferrolearn detects only `NaN`.

- AC-6 (REQ-6): `SimpleImputer(add_indicator=True)` appends a binary
  `MissingIndicator` mask (`:637-639`); ferrolearn has neither the param nor the
  estimator.

- AC-7 (REQ-7): `imp.inverse_transform(imp.transform(X))` recovers missing positions
  via the indicator (`:641`); ferrolearn has no `inverse_transform`.

- AC-8 (REQ-8): `SimpleImputer(strategy='constant').fit(...).statistics_ == [0.0,
  ...]` (`fill_value=None` → 0, Probe 7); ferrolearn requires an explicit constant
  and names the accessor `fill_values()`, not `statistics_`.

- AC-9 (REQ-9): `SimpleImputer(strategy='most_frequent')` on a string column returns
  the most common string (`_most_frequent` object branch, `:42-52`); ferrolearn is
  `F: Float`-only.

- AC-10 (REQ-10): a `scipy.sparse` input is imputed via `_sparse_fit` (`:444`);
  ferrolearn handles only dense `Array2`.

- AC-11 (REQ-11): a fitted imputer exposes `get_feature_names_out`,
  `n_features_in_`, `feature_names_in_`; ferrolearn exposes only `fill_values()`.

- AC-12 (REQ-12): a CPython `SimpleImputer` binding fits and transforms from Python;
  no such binding exists in `ferrolearn-python`.

- AC-13 (REQ-13): the statistic computation + NaN substitution path computes on
  `ferray-core` arrays rather than `ndarray` + `num_traits::Float` + the per-column
  `Vec<F>` collect.

`## REQ status`

## REQ status

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (per-column fill values for columns with ≥1 observed value; HEADLINE) | SHIPPED | impl `fn fit in imputer.rs` collects `x.column(j).iter().copied().filter(\|v\| !v.is_nan())` then matches `self.strategy`: `Mean => col_vals.iter().fold(F::zero(), \|acc, v\| acc + v) / n` (biased `np.ma.mean`, mirroring `_dense_fit` `mean_masked = np.ma.mean(masked_X, axis=0)` `_base.py:498`), `Median => median_of(&mut vals)` (`if n % 2 == 1 { values[n/2] } else { (values[mid-1] + values[mid]) / 2 }`, matching `np.ma.median` `:507`), `MostFrequent => most_frequent_of(&col_vals)` (sort ascending, count runs, keep best via strict `current_count > best_count` so ties resolve to the SMALLEST value, matching scipy `_mode` + `min` tie-break `:53-56,68-70`), `Constant(c) => *c` (matching `np.full(X.shape[1], fill_value)` `:545`); `fn transform in imputer.rs` then sets `*v = fill[j]` for every `v.is_nan()`. Value parity confirmed live: Probe 4 `mean=[3.0,5.0]`, `median=[3.0,5.0]`, even `[1,3,5,7]→4.0`; Probe 5 tie `[1,1,3,3]→1.0`, `[1,2,2,3]→2.0` (all IDENTICAL). Non-test consumer: the boundary re-export `pub use imputer::{FittedSimpleImputer, ImputeStrategy, SimpleImputer};` at `lib.rs:136` (grandfathered S5 / R-DEFER-1 boundary estimator API — `SimpleImputer` IS the public surface), alongside the `PipelineTransformer` / `FittedPipelineTransformer` impls on `SimpleImputer` / `FittedSimpleImputer`. Verification: `cargo test -p ferrolearn-preprocess imputer` → `test_mean_basic` (`fill_values=[3.0,5.0]`), `test_median_even_count` (`4.0`), `test_most_frequent_tie_chooses_smallest` (`1.0`), `test_constant_strategy`, `test_multi_column_mixed_nan`, `test_f32_imputer` green (16 in-module tests). NOTE: the all-NaN sub-case is explicitly NOT in REQ-1 (Probe 1/6 → REQ-2). |
| REQ-2 (all-NaN column handling — sklearn DEFAULT drops; formerly DIV-1 HEADLINE) | SHIPPED | impl `fn fit in imputer.rs` sets `fill_values[j] = F::nan()` and excludes `j` from `kept_indices` for an all-NaN non-constant column (`if col_vals.is_empty() { fill_values[j] = F::nan(); continue; }` — its `statistics_=nan` analogue, mirroring `_dense_fit:501,510-512,534-537`); `fn transform in imputer.rs` projects the output onto `kept_indices` (`for (out_j, &in_j) in self.kept_indices.iter().enumerate()`), mirroring sklearn `invalid_mask=_get_mask(statistics,nan)` + `X = X[:, valid_statistics_indexes]` (`_base.py:586-603`). `Constant` keeps every column and fills the user constant (sklearn `np.full(...,fill_value)` `:545`, `transform:583`). Non-test consumer: the boundary re-export `pub use imputer::{FittedSimpleImputer, ImputeStrategy, SimpleImputer};` at `lib.rs:136` (grandfathered S5 / R-DEFER-1 boundary estimator API). Verification: `cargo test -p ferrolearn-preprocess --test divergence_imputer` (19 tests green) — `divergence_mean_all_nan_column_dropped` / `divergence_median_all_nan_column_dropped` / `reaudit_a_column_order_two_all_nan_dropped` (`kept_indices()==[0,2]`) / `reaudit_b_all_columns_all_nan_zero_output` (`out.shape=(n,0)`) / `reaudit_c_most_frequent_all_nan_dropped` / `reaudit_d_constant_all_nan_kept_filled_constant` (KEPT, filled constant not 0) / `reaudit_e_fill_values_mirror_statistics_nan` (`fill_values=[3,NaN]`) / `reaudit_f_transform_separate_matrix_projection` / `reaudit_g_f32_all_nan_column_dropped`; in-module `test_mean_all_nan_column_dropped` (1 input col → 0 output cols). Was DIV-1 / #1364 (now RESOLVED). |
| REQ-3 (error / parameter contracts, scoped) | SHIPPED (scoped) | impl `fn fit in imputer.rs` returns `Err(FerroError::InsufficientSamples { required: 1, actual: 0, context: "SimpleImputer::fit".into() })` when `n_samples == 0`; impl `fn transform in imputer.rs` returns `Err(FerroError::ShapeMismatch { expected: vec![x.nrows(), n_features], actual: vec![x.nrows(), x.ncols()], context: "FittedSimpleImputer::transform".into() })` when `x.ncols() != n_features` (mirroring `transform`'s `X.shape[1] != statistics_.shape[0]` ValueError `_base.py:573-577`); impl `Transform for SimpleImputer in imputer.rs` (the unfitted handle) returns `Err(FerroError::InvalidParameter { name: "SimpleImputer".into(), reason: "imputer must be fitted before calling transform; use fit() first".into() })` (mirroring `check_is_fitted` `:568`). Non-test consumer: the boundary re-export at `lib.rs:136` routes every fit/transform through these guards. Verification: `cargo test -p ferrolearn-preprocess imputer` → `test_fit_zero_rows_error`, `test_transform_shape_mismatch_error`, `test_unfitted_transform_error` green. |
| REQ-4 (`keep_empty_features` param) | NOT-STARTED | open prereq blocker #1365. `SimpleImputer<F> { strategy }` has NO `keep_empty_features` field. sklearn's ctor accepts `keep_empty_features=False` (`_base.py:296`); when `True`, all-NaN cols are filled with `0` and KEPT (`_dense_fit:501,510-512,534-537`; `transform:583-585`). ferrolearn hardwires the `keep_empty_features=False` default drop (REQ-2) and cannot toggle to the keep-and-fill-0 behavior (Probe 2). |
| REQ-5 (`missing_values` param: non-NaN sentinel / None / str) | NOT-STARTED | open prereq blocker #1366. Missingness is detected ONLY via `v.is_nan()` (`fn fit` / `fn transform in imputer.rs`). sklearn's ctor accepts `missing_values=np.nan` (`_base.py:291`) and matches any scalar sentinel via `missing_mask = _get_mask(X, missing_values)` (`_dense_fit:491`, `transform:580`) — ferrolearn has no configurable sentinel. |
| REQ-6 (`add_indicator` + `MissingIndicator` estimator) | NOT-STARTED | open prereq blocker #1367. There is NO `MissingIndicator` type and NO `add_indicator` param in `imputer.rs`. sklearn's `add_indicator=False` (`_base.py:295`) appends a binary missing-mask via `_BaseImputer._fit_indicator` / `_concatenate_indicator` (`:485,637-639`), backed by the separate `MissingIndicator` class (`features="missing-only"\|"all"`). This route parity_op is ABSENT and needs acto-builder in a separate iteration. |
| REQ-7 (`inverse_transform`) | NOT-STARTED | open prereq blocker #1368. `FittedSimpleImputer<F>` exposes only `fill_values()` and a `Transform` impl — NO `inverse_transform`. sklearn's `SimpleImputer.inverse_transform` (`_base.py:641`) reverses `transform` using the `add_indicator` mask; ferrolearn lacks both the method and (REQ-6) the indicator to invert against. |
| REQ-8 (`fill_value=None` numeric default + `statistics_` name + `copy`) | NOT-STARTED | open prereq blocker #1369. `ImputeStrategy::Constant(F)` requires an explicit `F` (no `None`-default-to-0), the fitted accessor is `fill_values()` (NOT `statistics_`), and `fn transform in imputer.rs` always copies (`let mut out = x.to_owned()`) with no `copy` toggle. sklearn's `fill_value=None` → `0` for numeric (`_base.py:425-427`, Probe 7), stores results in `statistics_`, and `copy=True` (`:294`) governs in-place transform. |
| REQ-9 (string / object dtype) | NOT-STARTED | open prereq blocker #1370. `SimpleImputer<F>` is bounded `F: Float + Send + Sync + 'static` — numeric only. sklearn supports `strategy='most_frequent'` / `'constant'` on object/string arrays (the object branch of `_most_frequent` Counter + `min` tie-break `_base.py:42-52`; `np.full(fill_value)` with a string `:545`). |
| REQ-10 (sparse `_sparse_fit`) | NOT-STARTED | open prereq blocker #1371. `Fit<Array2<F>, ()>` and `Transform<Array2<F>>` operate only on dense `ndarray::Array2<F>`. sklearn dispatches sparse input to `_sparse_fit` (`_base.py:444`) and imputes over `X.data` (`transform:606-624`); ferrolearn has no sparse path. |
| REQ-11 (`get_feature_names_out` + `n_features_in_` / `feature_names_in_`) | NOT-STARTED | open prereq blocker #1372. `FittedSimpleImputer<F>` exposes only `fill_values()` / `kept_indices()` — NO `get_feature_names_out`, `n_features_in_`, or `feature_names_in_`. sklearn's `_BaseImputer` (via `TransformerMixin` / `BaseEstimator`) provides all three; `feature_names_in_` is used to name the dropped invalid features (`transform:596-597`). |
| REQ-12 (PyO3 binding) | NOT-STARTED | open prereq blocker #1373. No `_RsSimpleImputer` CPython binding exists — `grep -rn "SimpleImputer\|Imputer" ferrolearn-python/src` finds none — so the imputer is unreachable from Python. |
| REQ-13 (ferray substrate) | NOT-STARTED | open prereq blocker #1374. The statistic computation + NaN substitution path uses `ndarray::Array2` / `Array1` (`x.column(j)`, `out.columns_mut()`, `Array1::zeros`), `num_traits::Float`, and a per-column `Vec<F>` collect — not `ferray-core` arrays (R-SUBSTRATE-1/2). |

## Architecture

**ferrolearn (existing).** `imputer.rs` exposes the unfitted `SimpleImputer<F> {
strategy: ImputeStrategy<F> }` (`new(strategy)`, `strategy()`) with the `Copy`-free
`ImputeStrategy<F>` enum `{ Mean, Median, MostFrequent, Constant(F) }`, and the
fitted `FittedSimpleImputer<F> { fill_values: Array1<F>, kept_indices: Vec<usize> }`
with two accessors: `fill_values()` (one entry per INPUT column, mirroring sklearn's
`statistics_` — it holds `F::nan()` at a dropped all-NaN position) and
`kept_indices()` (the ascending input-column indices that survive `transform`).
`Fit<Array2<F>, ()> for SimpleImputer` (`fn fit in imputer.rs`) rejects zero rows
(`InsufficientSamples`), then per column collects the non-NaN entries. Under
`Constant` every column is filled and pushed to `kept_indices`; otherwise an empty
(all-NaN) column sets `fill_values[j] = F::nan()` and is omitted from `kept_indices`
(the REQ-2 default drop, sklearn `keep_empty_features=False`), while a column with
≥1 observed value computes `Mean` = `sum/n`, `Median` = `median_of` (sort, avg of
two middle for even `n`), `MostFrequent` = `most_frequent_of` (sort ascending, count
consecutive runs, keep the best by strict `>` so ties resolve to the smallest value)
and is kept. `Transform<Array2<F>> for FittedSimpleImputer` (`fn transform in
imputer.rs`) checks the column count against `fill_values.len()` (the full input
width, `ShapeMismatch`), then builds an `(nrows, kept_indices.len())` output by
iterating `kept_indices`, copying each surviving column and replacing its NaNs with
`fill_values[in_j]` — **projecting away the dropped all-NaN columns** (sklearn `X =
X[:, valid_statistics_indexes]`). The unfitted `Transform for SimpleImputer` is an
error stub (`InvalidParameter`) satisfying the `FitTransform: Transform` supertrait;
`FitTransform`, `PipelineTransformer`, and `FittedPipelineTransformer` wrap the
fit/transform path. The grandfathered boundary re-export at `lib.rs:136` (`pub use
imputer::{FittedSimpleImputer, ImputeStrategy, SimpleImputer}`) is the non-test
production consumer that pins REQ-1 / REQ-2 / REQ-3 SHIPPED.

**sklearn (target contract).** `SimpleImputer(_BaseImputer)` (`_base.py:147`,
`_BaseImputer:73`) takes `__init__(*, missing_values=np.nan, strategy="mean",
fill_value=None, copy=True, add_indicator=False, keep_empty_features=False)`
(`:288-304`). `_dense_fit` (`:489-552`) builds `missing_mask = _get_mask(X,
missing_values)` (`:491`), a `ma.masked_array` (`:492`), then computes `mean =
np.ma.mean` (`:498`), `median = np.ma.median` (`:507`), `most_frequent` per column
via `_most_frequent(row, np.nan, 0)` (scipy `_mode`, tie → `min`, `:36-71`), or
`np.full(fill_value)` for constant (`:545`); for `Mean`/`Median`/`MostFrequent` a
fully-masked column yields `0 if keep_empty_features else np.nan`
(`:501,510-512,534-537`). `transform` (`:554-639`) `check_is_fitted`s (`:568`),
validates the column count (`:573-577`), and — unless `strategy == "constant"` or
`keep_empty_features` — drops columns whose `statistics_` is `np.nan` via
`invalid_mask = _get_mask(statistics, np.nan)` / `X = X[:,
valid_statistics_indexes]` with a warning (`:586-603`), then imputes the survivors
(dense `:625-635`, sparse `:606-624`) and optionally concatenates the indicator
(`:637-639`). `inverse_transform` (`:641`) and the `add_indicator` /
`MissingIndicator` surface come from `_BaseImputer`.

**The gap.** ferrolearn matches sklearn on the *fill-value algebra for columns with
at least one observed value* (mean / median / most-frequent-with-smallest-tie /
constant — REQ-1, Probes 4-5), on the *all-NaN-column default drop* (REQ-2, formerly
DIV-1, now RESOLVED via `kept_indices` projection + `fill_values[j]=NaN`, Probes
1-3-6), and on the scoped structural contracts (zero-rows, column-count, unfitted —
REQ-3). The remaining gaps are the surrounding `_BaseImputer` surface and
configuration: no `keep_empty_features` toggle (REQ-4), no configurable
`missing_values` sentinel (REQ-5, NaN-only), no `add_indicator` / `MissingIndicator`
(REQ-6, the second route parity_op, ABSENT), no `inverse_transform` (REQ-7), no
`fill_value=None` default / `statistics_` name / `copy` (REQ-8), no string/object
dtype (REQ-9), no sparse `_sparse_fit` (REQ-10), no feature-name introspection
(REQ-11), no PyO3 binding (REQ-12), and the non-ferray substrate (REQ-13). This is
a **shipped-partial** unit (3 SHIPPED / 10 NOT-STARTED).

## Verification

Commands establishing the SHIPPED claims (REQ-1 fill values for columns with ≥1
observed value, REQ-2 all-NaN-column default drop, REQ-3 scoped error contracts):

```bash
# Consumer / module wiring check:
grep -rn "pub mod imputer" ferrolearn-preprocess/src/lib.rs                 # :99
grep -rn "pub use imputer::" ferrolearn-preprocess/src/lib.rs               # :136 boundary re-export consumer

# REQ-1 value parity + REQ-3 error contracts (16 in-module tests):
cargo test -p ferrolearn-preprocess imputer
#   REQ-1: test_mean_basic (fill_values=[3.0,5.0]), test_mean_no_nan,
#          test_mean_multiple_nans_same_column, test_median_odd_count,
#          test_median_even_count (4.0), test_median_with_nan,
#          test_most_frequent_basic, test_most_frequent_tie_chooses_smallest (1.0),
#          test_most_frequent_with_nan, test_constant_strategy,
#          test_multi_column_mixed_nan, test_f32_imputer, test_fit_transform_equivalence,
#          test_strategy_accessor, test_pipeline_integration
#   REQ-3: test_fit_zero_rows_error, test_transform_shape_mismatch_error,
#          test_unfitted_transform_error
# REQ-2 default-drop parity + REQ-1/REQ-3 green-guards (19 tests in the divergence suite):
cargo test -p ferrolearn-preprocess --test divergence_imputer
#   REQ-2 (all-NaN DROP): divergence_mean_all_nan_column_dropped,
#         divergence_median_all_nan_column_dropped,
#         divergence_median_multi_feature_one_all_nan_dropped,
#         reaudit_a_column_order_two_all_nan_dropped (kept_indices==[0,2]),
#         reaudit_b_all_columns_all_nan_zero_output (out.shape=(n,0)),
#         reaudit_c_most_frequent_all_nan_dropped,
#         reaudit_d_constant_all_nan_kept_filled_constant (KEPT+constant, not 0),
#         reaudit_e_fill_values_mirror_statistics_nan (fill_values=[3,NaN]),
#         reaudit_f_transform_separate_matrix_projection,
#         reaudit_g_f32_all_nan_column_dropped, green_constant_all_nan_column_kept
cargo clippy -p ferrolearn-preprocess --all-targets -- -D warnings
cargo fmt --all --check

# REQ-1 oracle gate — sklearn fill VALUES on columns with >=1 observed value:
python3 -c "import numpy as np
from sklearn.impute import SimpleImputer
X=np.array([[1.,np.nan],[3.,4.],[5.,6.]])
print('mean=',SimpleImputer(strategy='mean').fit(X).statistics_.tolist())
print('median even=',SimpleImputer(strategy='median').fit(np.array([[1.],[3.],[5.],[7.]])).statistics_.tolist())
print('mf tie=',SimpleImputer(strategy='most_frequent').fit(np.array([[1.],[1.],[3.],[3.]])).statistics_.tolist())"
#   -> mean= [3.0, 5.0]   median even= [4.0]   mf tie= [1.0]
#   (ferrolearn test_mean_basic / test_median_even_count / test_most_frequent_tie_chooses_smallest)

# REQ-2 oracle gate (now MATCHED — sklearn DEFAULT drops the all-NaN column):
python3 -c "import numpy as np
from sklearn.impute import SimpleImputer
X=np.array([[1.,np.nan],[3.,np.nan],[5.,np.nan]])
si=SimpleImputer(strategy='mean'); out=si.fit_transform(X)
print('sklearn out.shape=',out.shape,'statistics_=',si.statistics_.tolist())"
#   -> sklearn out.shape= (3, 1) statistics_= [3.0, nan]   (col1 DROPPED)
#   ferrolearn NOW MATCHES: fill_values=[3.0, nan], kept_indices=[0], out.shape=(3, 1)
#   (divergence_mean_all_nan_column_dropped) => REQ-2 SHIPPED
```

The in-module `#[test]`s exercise REQ-1 (the mean / median / most-frequent / constant
fill-value algebra for columns with ≥1 observed value), REQ-2 (`test_mean_all_nan_
column_dropped` — 1 all-NaN input column → 0 output columns, `fill_values()` NaN),
and REQ-3 (every error path — `test_fit_zero_rows_error`,
`test_transform_shape_mismatch_error`, `test_unfitted_transform_error`). The 19-test
`tests/divergence_imputer.rs` suite pins the REQ-2 default drop end-to-end against
the live sklearn 1.5.2 oracle (column-order, all-dropped, MostFrequent, Constant
keep+fill, `statistics_`-NaN, separate-matrix projection, f32). No green ferrolearn
command establishes REQ-4..REQ-13 (`keep_empty_features`, `missing_values`,
`add_indicator` / `MissingIndicator`, `inverse_transform`, `fill_value=None` /
`statistics_` / `copy`, string dtype, sparse, feature-name introspection, PyO3,
ferray).

## Blockers

REQ-1 (per-column fill values for columns with ≥1 observed value, HEADLINE), REQ-2
(all-NaN-column default drop, formerly DIV-1), and REQ-3 (scoped error / parameter
contracts) are SHIPPED, with the boundary re-export at `lib.rs:136` as the
grandfathered (S5 / R-DEFER-1) non-test production consumer.

DIV-1 / #1364 (REQ-2 all-NaN default drop) is **RESOLVED**: `fit` sets
`fill_values[j]=NaN` + excludes `j` from `kept_indices`, `transform` projects onto
`kept_indices` (`_dense_fit:501,510-512,534-537`; `transform:586-603`), and
`Constant` keeps+fills (`:545,583`) — verified by the 19-test divergence suite.

The remaining REQs are NOT-STARTED, filed as `-l blocker` issues against tracking
issue #1363:

- #1365 — REQ-4: no `keep_empty_features` param (`_base.py:296`,
  `transform:583-585`); ferrolearn hardwires the default drop.
- #1366 — REQ-5: no configurable `missing_values` sentinel — NaN-only
  (`_base.py:291`, `_get_mask` `_dense_fit:491`, `transform:580`).
- #1367 — REQ-6: no `add_indicator` / `MissingIndicator` (route parity_op, ABSENT;
  `_base.py:295,485,637-639`, `MissingIndicator` class) — needs acto-builder.
- #1368 — REQ-7: no `inverse_transform` (`_base.py:641`).
- #1369 — REQ-8: no `fill_value=None`→0 default, `statistics_` accessor name, or
  `copy` param (`_base.py:294,425-427`).
- #1370 — REQ-9: no string/object dtype (`_most_frequent` object branch
  `_base.py:42-52`, `:545`).
- #1371 — REQ-10: no sparse `_sparse_fit` (`_base.py:444`, `transform:606-624`).
- #1372 — REQ-11: no `get_feature_names_out` / `n_features_in_` /
  `feature_names_in_` (`_base.py:596-597`).
- #1373 — REQ-12: no PyO3 `SimpleImputer` binding in `ferrolearn-python`.
- #1374 — REQ-13: fit/transform on `ndarray` / `num_traits` / per-column `Vec<F>`,
  not ferray (R-SUBSTRATE-1/2).
