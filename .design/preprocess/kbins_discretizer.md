# KBinsDiscretizer

<!--
tier: 3-component
status: shipped-partial
baseline-commit: 6aece7ad
upstream: scikit-learn 1.5.2
upstream-paths:
  - sklearn/preprocessing/_discretization.py  # class KBinsDiscretizer(TransformerMixin, BaseEstimator). __init__(n_bins=5, *, encode="onehot", strategy="quantile", dtype=None, subsample=200_000, random_state=None) (:184-199). fit(X, y=None, sample_weight=None) (:202-327): _validate_data numeric (:226); subsample resample if n_samples>subsample (:242-249); n_bins=self._validate_n_bins(n_features) per-feature int array (:252); per col col_min,col_max=column.min(),column.max() (:260); CONSTANT col_min==col_max -> warn, n_bins[jj]=1, bin_edges[jj]=[-inf,inf], continue (:262-268); uniform np.linspace(col_min,col_max,n_bins[jj]+1) (:270-271); quantile np.percentile(column, np.linspace(0,100,n_bins[jj]+1)) (:273-284) or _weighted_percentile when sample_weight (:277-283); kmeans init=midpoints(linspace) n_init=1, centers.sort(), bin_edges=r_[col_min,(centers[1:]+centers[:-1])*.5,col_max] (:285-300); small-bin removal for quantile/kmeans mask=ediff1d(edges,to_begin=inf)>1e-8, edges=edges[mask], n_bins[jj]=len(edges)-1 + warn (:302-312); store bin_edges_,n_bins_ (:314-315); onehot OneHotEncoder(categories=[arange(i) for i in n_bins_], sparse_output=(encode=='onehot')) (:317-325). _validate_n_bins(n_features) (:329-352): scalar Integral -> np.full(n_features,orig_bins) else check_array shape (n_features,), reject n_bins<2 (:329-352). transform(X) (:354-391): check_is_fitted (:369); per col Xt[:,jj]=np.searchsorted(bin_edges[jj][1:-1], Xt[:,jj], side="right") (:377); ordinal -> Xt (:379-380) else _encoder.transform (:383-391). inverse_transform (:393).
ferrolearn-module: ferrolearn-preprocess/src/kbins_discretizer.rs
parity-ops: KBinsDiscretizer
crosslink-issue: 1375
-->

## Summary

scikit-learn's `KBinsDiscretizer` (`_discretization.py:184`) bins each continuous
feature independently into `n_bins` discrete intervals, learning per-feature
`bin_edges_` at `fit` (`:202-327`) under one of three strategies — `"uniform"`
(equal-width, `np.linspace`, `:271`), `"quantile"` (equal-frequency,
`np.percentile`, `:276`), or `"kmeans"` (1D k-means cluster midpoints, `:285-300`)
— then at `transform` assigns each value to a bin via
`np.searchsorted(bin_edges[jj][1:-1], ..., side="right")` (`:377`) and emits either
ordinal indices or a (sparse by default) one-hot encoding (`:317-325,379-391`).

`ferrolearn-preprocess/src/kbins_discretizer.rs` ships the **uniform / quantile /
kmeans edge computation plus ordinal & dense one-hot binning for NON-degenerate
features** over dense `Array2<F>`: `KBinsDiscretizer<F> { n_bins: usize, encode:
BinEncoding, strategy: BinStrategy }` (`new`, `n_bins()`, `encode()`, `strategy()`)
with `BinStrategy::{Uniform, Quantile, KMeans}` and `BinEncoding::{Ordinal, OneHot}`
produces `FittedKBinsDiscretizer<F> { bin_edges: Vec<Vec<F>>, n_bins: usize, encode
}` (`bin_edges()`, `n_bins()`, `encode()`). `Fit::fit` (`fn fit in
kbins_discretizer.rs`) sorts each column and computes edges per strategy; `assign_bin`
(`fn assign_bin in kbins_discretizer.rs`) places a value in the first edge it is
strictly below (the `searchsorted(side="right")` equivalent); `Transform::transform`
(`fn transform in kbins_discretizer.rs`) emits the ordinal index per feature or a
`n_features * n_bins` dense one-hot block.

The HEADLINE divergences DIV-1 and DIV-2 are now **RESOLVED** by the per-feature
variable bin-count fix (`n_bins_per_feature: Vec<usize>` on `FittedKBinsDiscretizer`,
mirroring sklearn `n_bins_`). DIV-1 / REQ-4 (was #1376, RESOLVED): for a CONSTANT
feature (`col_min == col_max`) `fn fit in kbins_discretizer.rs` now sets
`bin_edges=[-inf, +inf]`, `n_bins_per_feature[j]=1`, and `assign_bin` maps every value
to **0** (mirrors `:262-268`, sklearn `[0,0,0]` for `[[5],[5],[5]]`/`n_bins=3`).
DIV-2 / REQ-5 (was #1377, RESOLVED): for `quantile`/`kmeans`, `fit` collapses edges
whose gap to the previously-kept edge is **not** `> 1e-8` (mirrors `ediff1d > 1e-8`,
`:302-312`), reducing the per-feature bin count; `transform` one-hot width is now
`sum(n_bins_per_feature)` with cumulative per-feature offsets. This is a
**shipped-partial** unit: **5 SHIPPED** (REQ-1 uniform/quantile edges +
ordinal/onehot values, REQ-2 kmeans on well-separated data [scoped], REQ-3 scoped
error/parameter contracts, REQ-4 constant→bin 0 + `n_bins_per_feature[j]=1`, REQ-5
small-bin removal + onehot variable width) / **9 NOT-STARTED** (REQ-6
`subsample`/`random_state`, REQ-7 per-feature `n_bins` array + `_validate_n_bins`,
REQ-8 `encode='onehot'` sparse default + sklearn ctor defaults, REQ-9 `dtype` +
`sample_weight`, REQ-10 `inverse_transform`, REQ-11 `get_feature_names_out` +
`bin_edges_`/`n_bins_` attr names + `PipelineTransformer`, REQ-12 PyO3 binding,
REQ-13 ferray substrate, plus the KMeans EXACT-parity carve-out #1378 on
degenerate/duplicate-heavy data — empty-cluster relocation, no committed failing
test per R-DEFER-3).

## Probes (live sklearn oracle, 1.5.2)

All values below are live output from `python3` against scikit-learn 1.5.2, run from
`/tmp`. They pin the uniform / quantile / kmeans edge algebra and ordinal transform
values (REQ-1, REQ-2) and the constant-feature divergence (REQ-4, DIV-1 HEADLINE).

```bash
# PROBE 1 (REQ-1) — uniform edges = np.linspace(col_min,col_max,n_bins+1) (:271);
# transform via searchsorted(edges[1:-1], side="right") (:377):
python3 -c "import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
X=np.array([[0.],[1.],[2.],[3.],[4.],[5.]])
kb=KBinsDiscretizer(n_bins=3,encode='ordinal',strategy='uniform').fit(X)
print('uniform edges=',[e.tolist() for e in kb.bin_edges_])
print('uniform ordinal=',kb.transform(X).ravel().tolist())"
#   -> uniform edges= [[0.0, 1.6666666666666667, 3.3333333333333335, 5.0]]
#   -> uniform ordinal= [0.0, 0.0, 1.0, 1.0, 2.0, 2.0]
#   ferrolearn Uniform (min+(max-min)*i/n_bins) MATCHES edges; assign_bin -> [0,0,1,1,2,2].

# PROBE 2 (REQ-1) — quantile edges = np.percentile(col, np.linspace(0,100,n_bins+1)) (:276):
python3 -c "import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
X=np.array([[float(i)] for i in range(8)])
kb=KBinsDiscretizer(n_bins=4,encode='ordinal',strategy='quantile').fit(X)
print('quantile edges=',[e.tolist() for e in kb.bin_edges_])
print('quantile ordinal=',kb.transform(X).ravel().tolist())"
#   -> quantile edges= [[0.0, 1.75, 3.5, 5.25, 7.0]]
#   -> quantile ordinal= [0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0]
#   ferrolearn Quantile (linear-interp percentile pos=frac*(n-1)) MATCHES edges -> [0,0,1,1,2,2,3,3].

# PROBE 3 (REQ-4, DIV-1 HEADLINE) — CONSTANT feature: edges=[-inf,inf], n_bins_=1, all -> bin 0 (:262-268):
python3 -c "import numpy as np, warnings; warnings.simplefilter('ignore')
from sklearn.preprocessing import KBinsDiscretizer
X=np.array([[5.],[5.],[5.]])
kb=KBinsDiscretizer(n_bins=3,encode='ordinal',strategy='uniform').fit(X)
print('const edges=',[e.tolist() for e in kb.bin_edges_],'n_bins_=',kb.n_bins_.tolist(),'transform=',kb.transform(X).ravel().tolist())"
#   -> const edges= [[-inf, inf]] n_bins_= [1] transform= [0.0, 0.0, 0.0]
#   ferrolearn DIVERGES: degenerate all-equal edges -> assign_bin falls through to last bin -> [2,2,2]. (DIV-1, OPEN.)

# PROBE 4 (REQ-2) — kmeans on well-separated data; deterministic uniform-midpoint init (:289-300):
python3 -c "import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
X=np.array([[0.],[0.1],[0.2],[5.],[5.1],[5.2],[10.],[10.1],[10.2]])
kb=KBinsDiscretizer(n_bins=3,encode='ordinal',strategy='kmeans').fit(X)
print('kmeans edges=',[np.round(e,4).tolist() for e in kb.bin_edges_])
print('kmeans transform=',kb.transform(X).ravel().tolist())"
#   -> kmeans edges= [[0.0, 2.6, 7.6, 10.2]]
#   -> kmeans transform= [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0]
#   ferrolearn kmeans_1d (same uniform-midpoint init, Lloyd, midpoint edges) MATCHES on well-separated data.
```

## Requirements

- REQ-1: **Uniform + quantile bin EDGES and ordinal/one-hot transform VALUES for
  NON-degenerate features** (HEADLINE, SHIPPED, scoped). For `"uniform"` compute
  `np.linspace(col_min, col_max, n_bins+1)` (`:271`); for `"quantile"` compute
  `np.percentile(column, np.linspace(0,100,n_bins+1))` (`:276`); then at `transform`
  assign each value via `np.searchsorted(bin_edges[jj][1:-1], ..., side="right")`
  (`:377`), emitting the ordinal index or the one-hot block (`:379-391`). ferrolearn's
  `fn fit in kbins_discretizer.rs` sorts each column and computes `Uniform` =
  `min_val + (max_val - min_val) * i / n_bins` (`(0..=self.n_bins).map(...)`,
  identical to `np.linspace`) and `Quantile` via linear-interpolation percentile
  (`pos = frac * (n-1)`, `col_vals[lo]*(1-f) + col_vals[hi]*f`, identical to
  `np.percentile`); `fn assign_bin in kbins_discretizer.rs` returns the first edge
  index `i>=1` with `value < edge` minus one, else the last bin — the
  `searchsorted(side="right")` equivalent; `fn transform in kbins_discretizer.rs`
  emits the ordinal index (`BinEncoding::Ordinal`) or the dense one-hot column
  (`BinEncoding::OneHot`). Value parity is live-confirmed (Probe 1 edges
  `[0,1.667,3.333,5]` → `[0,0,1,1,2,2]`; Probe 2 edges `[0,1.75,3.5,5.25,7]` →
  `[0,0,1,1,2,2,3,3]`). **Scoped to non-degenerate features**: no constant column
  (REQ-4 / DIV-1) and no near-duplicate-edge collapse (REQ-5 / DIV-2). Pinned by the
  in-module tests.

- REQ-2: **KMeans bin edges / transform on well-separated data** (SHIPPED, scoped).
  sklearn's `"kmeans"` initializes `KMeans(n_clusters=n_bins, init=midpoints(linspace),
  n_init=1)`, sorts the resulting centers, and sets `bin_edges = np.r_[col_min,
  (centers[1:]+centers[:-1])*0.5, col_max]` (`:285-300`). ferrolearn's `fn kmeans_1d in
  kbins_discretizer.rs` uses the **same deterministic uniform-midpoint init** (`min +
  (max-min)*(i+0.5)/n_bins`), runs Lloyd to `max_iter=100` / convergence at max center
  shift `< 1e-10`, sorts centers, and builds edges `[min, midpoints.., max]` — matching
  sklearn on well-separated data (Probe 4: edges `[0,2.6,7.6,10.2]` →
  `[0,0,0,1,1,1,2,2,2]`). **Underclaim**: exact parity OFF the well-separated regime is
  not asserted — it depends on sklearn `KMeans` tolerance and empty-cluster relocation,
  which `kmeans_1d` (no empty-cluster handling, fixed `1e-10` tol) does not reproduce.
  **Carve-out #1378 (NOT-STARTED, no committed failing test per R-DEFER-3)**:
  `kmeans_1d` KEEPS empty clusters whereas sklearn's Lloyd KMeans RELOCATES them, so
  on duplicate-heavy data sklearn's centers coincide → near-duplicate edges → small-bin
  collapse fires (`n_bins_` reduced) while ferrolearn's centers stay spread → no
  collapse. EXACT parity on the degenerate/duplicate-heavy regime needs a
  sklearn-faithful 1D KMeans (empty-cluster relocation) — the same class as the
  `cluster/kmeans.rs` numpy-RNG carve-outs (see the REQ-14 row). Well-separated data
  MATCHES (Probe 4); pinned by `test_kbins_kmeans_strategy` (in-module) and
  `green_kmeans_well_separated` (`tests/divergence_kbins_discretizer.rs`).

- REQ-3: **Error / parameter contracts** (scoped, SHIPPED). `fn fit in
  kbins_discretizer.rs` returns `InsufficientSamples { required: 2, .. }` when
  `n_samples < 2`, and `InvalidParameter { name: "n_bins" }` when `n_bins < 2`
  (mirroring sklearn's `_validate_n_bins` "must be at least 2" guard, `:340-351`, for
  the scalar case); `fn transform in kbins_discretizer.rs` returns `ShapeMismatch` when
  `x.ncols()` differs from the fitted feature count (mirroring `_validate_data(...,
  reset=False)` column check, `:373`); the unfitted `Transform for KBinsDiscretizer`
  returns `InvalidParameter` (the "must fit first" guard, mirroring `check_is_fitted`,
  `:369`). Scoped to the contracts ferrolearn enforces over the dense scalar-`n_bins`
  API.

- REQ-4: **Constant feature → bin 0 + `n_bins_per_feature[j]=1`** (DIV-1 HEADLINE,
  SHIPPED). For a column with `col_min == col_max` sklearn warns, sets `n_bins_[jj]=1`,
  `bin_edges_[jj]=np.array([-inf, inf])`, and bins every value to **0** (`:262-268`,
  Probe 3). `fn fit in kbins_discretizer.rs` now detects `min_val == max_val` and pushes
  `bin_edges=[-inf, +inf]` with `n_bins_per_feature[j]=1`, so `fn assign_bin in
  kbins_discretizer.rs` returns bin 0 for every value (`[5,5,5]`/`n_bins=3` → `[0,0,0]`,
  matching sklearn). The per-feature `n_bins_per_feature: Vec<usize>` field on
  `FittedKBinsDiscretizer` is the structural mechanism. Was DIV-1 / #1376 — now
  RESOLVED. Pinned by `divergence_div1_constant_feature_uniform`,
  `divergence_div1_constant_feature_quantile`, `divergence_div1_constant_feature_multi`,
  `reaudit_g_f32_constant_column` (`tests/divergence_kbins_discretizer.rs`).

- REQ-5: **Small-bin removal (quantile/kmeans edge collapse + per-feature
  `n_bins_per_feature`) + one-hot variable width** (DIV-2, SHIPPED). For
  `"quantile"`/`"kmeans"` sklearn drops edges within `1e-8` of their predecessor —
  `mask = np.ediff1d(bin_edges[jj], to_begin=inf) > 1e-8; bin_edges[jj] =
  bin_edges[jj][mask]; n_bins_[jj] = len(bin_edges[jj]) - 1` + warn (`:302-312`) —
  yielding a SMALLER per-feature bin count when input quantiles tie. `fn fit in
  kbins_discretizer.rs` now keeps the first edge and each subsequent edge only when its
  gap to the previously-kept edge is `> 1e-8` (the `ediff1d > 1e-8` mask), then sets
  `n_bins_per_feature[j] = kept.len() - 1` (Uniform is never collapsed). `fn transform
  in kbins_discretizer.rs` one-hot output width is now `sum(n_bins_per_feature)` with
  feature `j`'s columns starting at the cumulative offset of the preceding features'
  bin counts. Was DIV-2 / #1377 — now RESOLVED. Pinned by
  `divergence_div2_quantile_small_bin_removal`, `reaudit_b_onehot_quantile_collapse`,
  `reaudit_a_onehot_constant_among_normal`, `reaudit_d_multi_feature_mixed`,
  `reaudit_e1_threshold_below_collapses`, `reaudit_e2_threshold_above_kept`,
  `reaudit_f_quantile_no_spurious_collapse` (`tests/divergence_kbins_discretizer.rs`).

- REQ-6: **`subsample` + `random_state`** (NOT-STARTED). sklearn defaults
  `subsample=200_000`; when `n_samples > subsample` it `resample`s without replacement
  under `random_state` before computing quantile/kmeans edges (`:191-192,242-249`).
  ferrolearn's `KBinsDiscretizer<F>` has neither field and always fits on the full
  column. Open prereq blocker #1379.

- REQ-7: **`n_bins` as a per-feature array + `_validate_n_bins`** (NOT-STARTED).
  sklearn accepts `n_bins` as a scalar OR an array of shape `(n_features,)` and
  validates each entry via `_validate_n_bins` (`:329-352`), storing the resolved
  per-feature counts in `n_bins_` (`:315`). ferrolearn's `KBinsDiscretizer<F> { n_bins:
  usize }` and `FittedKBinsDiscretizer<F> { n_bins: usize }` carry a single scalar with
  no per-feature override and no `n_bins_` attribute — this is the structural blocker
  underlying REQ-4 and REQ-5. Open prereq blocker #1380.

- REQ-8: **`encode='onehot'` SPARSE default + sklearn ctor defaults** (NOT-STARTED).
  sklearn defaults `encode="onehot"` (SPARSE `OneHotEncoder(sparse_output=True)`,
  `:188,317-321`) and `strategy="quantile"` (`:189`); `encode="onehot-dense"` gives a
  dense one-hot, and `encode="ordinal"` the integer indices. ferrolearn's `OneHot`
  emits a **dense** `n_features*n_bins` block (sklearn's `"onehot-dense"`), has **no
  sparse output**, and its `Default` is `(5, Ordinal, Uniform)` — diverging from
  sklearn's `(5, "onehot", "quantile")` ctor defaults (`:184-189`). Open prereq blocker
  #1381.

- REQ-9: **`dtype` param + `sample_weight`** (NOT-STARTED). sklearn's `dtype`
  (`None`/`float64`/`float32`) selects the output dtype (`:228-231`), and
  `fit(sample_weight=...)` weights the quantile (`_weighted_percentile`, `:277-283`)
  and kmeans (`:295`) edges (rejected for `strategy="uniform"`, `:235-240`). ferrolearn
  is generic over `F: Float` with no output-dtype selector and no `sample_weight`
  argument to `Fit::fit` (its `_y: &()` is ignored). Open prereq blocker #1382.

- REQ-10: **`inverse_transform`** (NOT-STARTED). sklearn's `inverse_transform`
  (`:393`) maps each bin index back to the bin-center representative value.
  ferrolearn's `FittedKBinsDiscretizer<F>` exposes only `bin_edges()` / `n_bins()` /
  `encode()` and a `Transform` impl — no `inverse_transform`. Open prereq blocker
  #1383.

- REQ-11: **`get_feature_names_out` + `bin_edges_`/`n_bins_` attr names +
  `PipelineTransformer` impl** (NOT-STARTED). sklearn (via `TransformerMixin`) exposes
  `get_feature_names_out` and names the fitted attributes `bin_edges_` (`:314`) /
  `n_bins_` (`:315`). ferrolearn names the accessor `bin_edges()` (not `bin_edges_`),
  has no `get_feature_names_out`, and — unlike the sibling `SimpleImputer` — provides
  **NO `PipelineTransformer` / `FittedPipelineTransformer` impl** on
  `KBinsDiscretizer` / `FittedKBinsDiscretizer`, so the discretizer cannot be dropped
  into a ferrolearn pipeline. Open prereq blocker #1384.

- REQ-12: **PyO3 binding** (NOT-STARTED). There is no `_RsKBinsDiscretizer` CPython
  binding — `grep -rn "KBinsDiscretizer\|Discretizer" ferrolearn-python/src` finds none
  — so the discretizer is unreachable from Python. Open prereq blocker #1385.

- REQ-13: **ferray substrate** (NOT-STARTED). Compute the per-feature edges and the bin
  assignment over `ferray-core` arrays rather than `ndarray::Array2<F>` /
  `num_traits::Float` / the per-column `Vec<F>` sort path (R-SUBSTRATE). Open prereq
  blocker #1386.

## Acceptance criteria

- AC-1 (REQ-1): `KBinsDiscretizer::<f64>::new(3, Ordinal, Uniform).fit([[0],..,[5]])`
  yields `bin_edges()[0] == [0, 1.6667, 3.3333, 5]` and `transform` →
  `[0,0,1,1,2,2]` (Probe 1); `new(4, Ordinal, Quantile)` on `0..7` yields edges
  `[0,1.75,3.5,5.25,7]` and `[0,0,1,1,2,2,3,3]` (Probe 2); `OneHot` emits a
  `n_features*n_bins` block with exactly one `1.0` per row. Pinned by
  `test_kbins_ordinal_uniform`, `test_kbins_onehot_uniform`,
  `test_kbins_quantile_strategy`, `test_kbins_bin_edges`, `test_kbins_multi_feature`,
  `test_kbins_ordinal_values_in_range`, `test_kbins_fit_transform` (in-module).

- AC-2 (REQ-2): `new(3, Ordinal, KMeans)` on the well-separated
  `[[0],[0.1],[0.2],[5],[5.1],[5.2],[10],[10.1],[10.2]]` yields edges
  `[0,2.6,7.6,10.2]` and `transform` → `[0,0,0,1,1,1,2,2,2]` (Probe 4, MATCHES
  sklearn). Pinned by `test_kbins_kmeans_strategy` (in-module range check) and
  `green_kmeans_well_separated` (`tests/divergence_kbins_discretizer.rs`). EXACT parity
  on degenerate/duplicate-heavy data is the NOT-STARTED carve-out #1378 (AC-14, no
  committed failing test per R-DEFER-3).

- AC-3 (REQ-3): `new(3,..).fit(Array2::zeros((1,1)))` → `Err(InsufficientSamples)`;
  `new(1,..).fit([[0],[1]])` → `Err(InvalidParameter{name:"n_bins"})`; a fitted
  handle's `transform` on a wrong column count → `Err(ShapeMismatch)`; the unfitted
  `KBinsDiscretizer.transform` → `Err(InvalidParameter)`. Pinned by
  `test_kbins_insufficient_samples_error`, `test_kbins_too_few_bins_error`,
  `test_kbins_shape_mismatch_error`, `test_kbins_unfitted_error`.

- AC-4 (REQ-4, DIV-1, RESOLVED): `KBinsDiscretizer(n_bins=3,strategy='uniform')
  .fit_transform([[5],[5],[5]])` → `[0,0,0]` with `bin_edges()[0]==[-inf,+inf]`,
  `n_bins_per_feature()==[1]` (Probe 3), MATCHING sklearn. Pinned by
  `divergence_div1_constant_feature_uniform` /
  `divergence_div1_constant_feature_quantile` /
  `divergence_div1_constant_feature_multi` / `reaudit_g_f32_constant_column`.

- AC-5 (REQ-5, DIV-2, RESOLVED): a quantile column with tied quantiles drops sub-`1e-8`
  edges and reduces `n_bins_per_feature[j]` (`:302-312`) — `[[0]*4,[1],[2]]`/`n_bins=4`
  → `n_bins_per_feature()==[2]`, edges `[0,0.75,2]`, transform `[0,0,0,0,1,1]`; the
  one-hot width is `sum(n_bins_per_feature)` (e.g. `[3,1]`→4 columns). Pinned by
  `divergence_div2_quantile_small_bin_removal`, `reaudit_a_onehot_constant_among_normal`,
  `reaudit_b_onehot_quantile_collapse`, `reaudit_d_multi_feature_mixed`,
  `reaudit_e1_threshold_below_collapses`, `reaudit_e2_threshold_above_kept`,
  `reaudit_f_quantile_no_spurious_collapse`.

- AC-6 (REQ-6): `KBinsDiscretizer(subsample=100, random_state=0)` fits on a 100-row
  resample of a larger `X` (`:242-249`); ferrolearn fits on the full column.

- AC-7 (REQ-7): `KBinsDiscretizer(n_bins=[2,3])` validates and uses a per-feature bin
  count (`_validate_n_bins`, `:329-352`); ferrolearn accepts only a scalar `usize`.

- AC-8 (REQ-8): default `KBinsDiscretizer()` is `encode='onehot'` (SPARSE),
  `strategy='quantile'` (`:184-189`); ferrolearn's `Default` is
  `(5, Ordinal, Uniform)` and `OneHot` is dense-only.

- AC-9 (REQ-9): `KBinsDiscretizer(dtype=np.float32)` selects the output dtype
  (`:228-231`) and `fit(X, sample_weight=w)` weights quantile/kmeans edges
  (`:277-283,295`); ferrolearn has no `dtype` selector and ignores `_y: &()`.

- AC-10 (REQ-10): `kb.inverse_transform(kb.transform(X))` maps bin indices to bin
  centers (`:393`); ferrolearn has no `inverse_transform`.

- AC-11 (REQ-11): a fitted discretizer exposes `get_feature_names_out`, `bin_edges_`,
  `n_bins_`; ferrolearn exposes `bin_edges()` only and provides no
  `PipelineTransformer` impl.

- AC-12 (REQ-12): a CPython `KBinsDiscretizer` binding fits/transforms from Python; no
  such binding exists in `ferrolearn-python`.

- AC-13 (REQ-13): the edge computation + bin assignment runs on `ferray-core` arrays
  rather than `ndarray` + `num_traits::Float` + per-column `Vec<F>` sort.

- AC-14 (REQ-14, carve-out #1378): on duplicate-heavy/degenerate data sklearn's Lloyd
  KMeans relocates empty clusters so its centers coincide → small-bin collapse fires
  and `n_bins_` shrinks, whereas ferrolearn's `kmeans_1d` keeps empty clusters so its
  centers stay spread and no collapse occurs. NOT-STARTED; per R-DEFER-3 there is NO
  committed failing test pinning this (the removed degenerate pin was a placeholder, not
  retained). ferrolearn `kmeans_1d` MATCHES sklearn on well-separated data (AC-2).

`## REQ status`

## REQ status

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (uniform + quantile edges + ordinal/onehot values, non-degenerate; HEADLINE) | SHIPPED (scoped) | impl `fn fit in kbins_discretizer.rs` computes `BinStrategy::Uniform => (0..=self.n_bins).map(\|i\| min_val + (max_val - min_val) * F::from(i)/F::from(self.n_bins))` (identical to `np.linspace(col_min,col_max,n_bins+1)` `_discretization.py:271`) and `BinStrategy::Quantile` via linear-interp percentile (`pos = frac * (n-1)`, `col_vals[lo]*(1-f) + col_vals[hi]*f`, identical to `np.percentile(column, np.linspace(0,100,n_bins+1))` `:276`); `fn assign_bin in kbins_discretizer.rs` returns `i-1` for the first edge `i>=1` with `value < edge` else the last bin `n_bins-1` (the `np.searchsorted(bin_edges[jj][1:-1], ..., side="right")` equivalent `:377`); `fn transform in kbins_discretizer.rs` emits the ordinal index (`BinEncoding::Ordinal`) or a dense `n_features*self.n_bins` one-hot block (`BinEncoding::OneHot`). Value parity live-confirmed: Probe 1 edges `[0,1.6667,3.3333,5]` → `[0,0,1,1,2,2]`; Probe 2 edges `[0,1.75,3.5,5.25,7]` → `[0,0,1,1,2,2,3,3]`. Non-test consumer: boundary re-export `pub use kbins_discretizer::{BinEncoding, BinStrategy, FittedKBinsDiscretizer, KBinsDiscretizer};` at `lib.rs:151` (grandfathered S5 / R-DEFER-1 boundary estimator API). Verification: `cargo test -p ferrolearn-preprocess kbins` → `test_kbins_ordinal_uniform` (bin 0 / bin 2), `test_kbins_bin_edges` (edges `[0,2,4,6]`), `test_kbins_quantile_strategy`, `test_kbins_onehot_uniform` (one `1.0`/row), `test_kbins_ordinal_values_in_range`. SCOPED to non-degenerate features (constant col → REQ-4/DIV-1; near-duplicate-edge collapse → REQ-5/DIV-2). |
| REQ-2 (kmeans edges / transform on well-separated data) | SHIPPED (scoped) | impl `fn kmeans_1d in kbins_discretizer.rs` uses the same deterministic uniform-midpoint init `min_v + (max_v-min_v)*(i+0.5)/n_bins` as sklearn `init=(uniform_edges[1:]+uniform_edges[:-1])[:,None]*0.5` (`_discretization.py:289-290`), runs Lloyd to `max_iter=100` / convergence at max center shift `< 1e-10`, sorts centers, and builds `edges.push(min_v); midpoints; edges.push(max_v)` mirroring `np.r_[col_min, (centers[1:]+centers[:-1])*0.5, col_max]` (`:299-300`). Matches sklearn on well-separated data (Probe 4: edges `[0,2.6,7.6,10.2]` → `[0,0,0,1,1,1,2,2,2]`). UNDERCLAIM: exact parity OFF the well-separated regime is not asserted — sklearn `KMeans` tol + empty-cluster relocation (`:293`) are not reproduced by `kmeans_1d` (no empty-cluster handling). Non-test consumer: boundary re-export at `lib.rs:151`. Verification: `cargo test -p ferrolearn-preprocess kbins` → `test_kbins_kmeans_strategy` (well-separated 3-cluster input, valid-range check) and `green_kmeans_well_separated` (`tests/divergence_kbins_discretizer.rs`). CARVE-OUT: EXACT parity on degenerate/duplicate-heavy data (sklearn empty-cluster RELOCATION → coincident centers → small-bin collapse) is NOT-STARTED — carve-out #1378 (REQ-14), NO committed failing test per R-DEFER-3. |
| REQ-3 (error / parameter contracts, scoped) | SHIPPED (scoped) | impl `fn fit in kbins_discretizer.rs` returns `Err(FerroError::InsufficientSamples { required: 2, actual: n_samples, context: "KBinsDiscretizer::fit" })` when `n_samples < 2`, and `Err(FerroError::InvalidParameter { name: "n_bins", reason: "n_bins must be at least 2" })` when `self.n_bins < 2` (mirroring `_validate_n_bins` "must be at least 2" `_discretization.py:340-351`, scalar case); impl `fn transform in kbins_discretizer.rs` returns `Err(FerroError::ShapeMismatch { expected: vec![x.nrows(), n_features], actual: vec![x.nrows(), x.ncols()], context: "FittedKBinsDiscretizer::transform" })` when `x.ncols() != n_features` (mirroring `_validate_data(..., reset=False)` `:373`); impl `Transform for KBinsDiscretizer in kbins_discretizer.rs` (unfitted) returns `Err(FerroError::InvalidParameter { name: "KBinsDiscretizer", reason: "discretizer must be fitted before calling transform; use fit() first" })` (mirroring `check_is_fitted` `:369`). Non-test consumer: boundary re-export at `lib.rs:151`. Verification: `cargo test -p ferrolearn-preprocess kbins` → `test_kbins_insufficient_samples_error`, `test_kbins_too_few_bins_error`, `test_kbins_shape_mismatch_error`, `test_kbins_unfitted_error` green. |
| REQ-4 (constant feature → bin 0 + `n_bins_per_feature[j]=1`; DIV-1 HEADLINE) | SHIPPED | impl `fn fit in kbins_discretizer.rs` detects `min_val == max_val` and pushes `bin_edges=vec![F::neg_infinity(), F::infinity()]` with `n_bins_per_feature.push(1)` (mirrors `_discretization.py:262-268`); `fn assign_bin in kbins_discretizer.rs` then returns bin 0 for every value (`[[5],[5],[5]]`/`n_bins=3` → `[0,0,0]`, matching sklearn). The per-feature `n_bins_per_feature: Vec<usize>` field (sklearn `n_bins_`) on `FittedKBinsDiscretizer` is the structural mechanism. Was DIV-1 / #1376 — now RESOLVED. Non-test consumer: boundary re-export at `lib.rs:151`. Verification: `cargo test -p ferrolearn-preprocess --test divergence_kbins_discretizer` → `divergence_div1_constant_feature_uniform`, `divergence_div1_constant_feature_quantile`, `divergence_div1_constant_feature_multi`, `reaudit_g_f32_constant_column` green. |
| REQ-5 (small-bin removal / per-feature `n_bins_per_feature` + onehot variable width; DIV-2) | SHIPPED | impl `fn fit in kbins_discretizer.rs` (Quantile/KMeans arm) keeps the first edge then each subsequent edge only if `edge - last > tol` with `tol = 1e-8` (mirrors `mask = np.ediff1d(bin_edges[jj], to_begin=inf) > 1e-8` `_discretization.py:302-312`), and sets `n_bins_per_feature.push(kept.len() - 1)` (Uniform never collapses); impl `fn transform in kbins_discretizer.rs` `BinEncoding::OneHot` width is `acc = sum(n_bins_per_feature)` with feature `j` written at the cumulative `offsets[j]` (sklearn onehot over `n_bins_`). `[[0]*4,[1],[2]]`/`n_bins=4`,quantile → `n_bins_per_feature()==[2]`, edges `[0,0.75,2]`, transform `[0,0,0,0,1,1]`. Was DIV-2 / #1377 — now RESOLVED. Non-test consumer: boundary re-export at `lib.rs:151`. Verification: `cargo test -p ferrolearn-preprocess --test divergence_kbins_discretizer` → `divergence_div2_quantile_small_bin_removal`, `reaudit_a_onehot_constant_among_normal`, `reaudit_b_onehot_quantile_collapse`, `reaudit_d_multi_feature_mixed`, `reaudit_e1_threshold_below_collapses`, `reaudit_e2_threshold_above_kept`, `reaudit_f_quantile_no_spurious_collapse` green. |
| REQ-6 (`subsample` + `random_state`) | NOT-STARTED | open prereq blocker #1379. `KBinsDiscretizer<F> { n_bins, encode, strategy }` has NO `subsample` / `random_state` fields. sklearn defaults `subsample=200_000`, `random_state=None` (`_discretization.py:191-192`) and resamples without replacement when `n_samples > subsample` before computing edges (`:242-249`); ferrolearn always fits the full column. |
| REQ-7 (per-feature `n_bins` array + `_validate_n_bins`) | NOT-STARTED | open prereq blocker #1380. `KBinsDiscretizer<F> { n_bins: usize }` and `FittedKBinsDiscretizer<F> { n_bins: usize }` carry a single SCALAR; there is no per-feature override and no `n_bins_` attribute. sklearn accepts `n_bins` as a scalar or `(n_features,)` array validated per-feature by `_validate_n_bins` (`_discretization.py:329-352`) and stores the resolved counts in `n_bins_` (`:315`) — the structural blocker underlying REQ-4 and REQ-5. |
| REQ-8 (`encode='onehot'` SPARSE default + sklearn ctor defaults) | NOT-STARTED | open prereq blocker #1381. `fn transform in kbins_discretizer.rs` `BinEncoding::OneHot` emits a DENSE `n_features*self.n_bins` block (sklearn's `"onehot-dense"`); there is no sparse output, and `Default` is `(5, Ordinal, Uniform)`. sklearn defaults `encode="onehot"` (SPARSE `OneHotEncoder(sparse_output=True)` `_discretization.py:188,317-321`) and `strategy="quantile"` (`:189`). |
| REQ-9 (`dtype` param + `sample_weight`) | NOT-STARTED | open prereq blocker #1382. `Fit<Array2<F>, ()> for KBinsDiscretizer` ignores `_y: &()` — no `sample_weight` — and there is no `dtype` selector. sklearn's `dtype` selects output dtype (`_discretization.py:228-231`) and `fit(sample_weight=...)` weights quantile (`_weighted_percentile` `:277-283`) / kmeans (`:295`) edges, rejected for `strategy="uniform"` (`:235-240`). |
| REQ-10 (`inverse_transform`) | NOT-STARTED | open prereq blocker #1383. `FittedKBinsDiscretizer<F>` exposes only `bin_edges()` / `n_bins()` / `encode()` and a `Transform` impl — NO `inverse_transform`. sklearn's `inverse_transform` (`_discretization.py:393`) maps each bin index back to its bin-center representative. |
| REQ-11 (`get_feature_names_out` + `bin_edges_`/`n_bins_` names + `PipelineTransformer`) | NOT-STARTED | open prereq blocker #1384. `FittedKBinsDiscretizer<F>` names the accessor `bin_edges()` (not `bin_edges_`), has no `get_feature_names_out`, and — unlike the sibling `SimpleImputer` — there is NO `PipelineTransformer` / `FittedPipelineTransformer` impl on `KBinsDiscretizer` / `FittedKBinsDiscretizer` in `kbins_discretizer.rs`, so it cannot enter a ferrolearn pipeline. sklearn (via `TransformerMixin`) provides `get_feature_names_out` and the `bin_edges_` (`:314`) / `n_bins_` (`:315`) attributes. |
| REQ-12 (PyO3 binding) | NOT-STARTED | open prereq blocker #1385. No `_RsKBinsDiscretizer` CPython binding exists — `grep -rn "KBinsDiscretizer\|Discretizer" ferrolearn-python/src` finds none — so the discretizer is unreachable from Python. |
| REQ-13 (ferray substrate) | NOT-STARTED | open prereq blocker #1386. `fn fit in kbins_discretizer.rs` uses `x.column(j)` + per-column `Vec<F>` sort, `Array2<F>` output, and `num_traits::Float` — not `ferray-core` arrays (R-SUBSTRATE-1/2). |
| REQ-14 (KMeans EXACT parity on degenerate/duplicate-heavy data) | NOT-STARTED | carve-out blocker #1378 (NO committed failing test per R-DEFER-3). `fn kmeans_1d in kbins_discretizer.rs` KEEPS empty clusters (a centroid with `counts[c]==0` is left unmoved), whereas sklearn's Lloyd KMeans (`sklearn/cluster/_kmeans.py`) RELOCATES empty clusters; on duplicate-heavy data sklearn's centers coincide → near-duplicate edges → small-bin collapse fires and `n_bins_` shrinks, while ferrolearn's stay spread → no collapse. Well-separated data MATCHES (REQ-2). Same class as the `cluster/kmeans.rs` numpy-RNG carve-outs; needs a sklearn-faithful 1D KMeans with empty-cluster relocation. |

## Architecture

**ferrolearn (existing).** `kbins_discretizer.rs` exposes the unfitted
`KBinsDiscretizer<F> { n_bins: usize, encode: BinEncoding, strategy: BinStrategy }`
(`new(n_bins, encode, strategy)`, accessors `n_bins()` / `encode()` / `strategy()`,
`Default = (5, Ordinal, Uniform)`) with the `Copy` enums `BinStrategy::{Uniform,
Quantile, KMeans}` and `BinEncoding::{Ordinal, OneHot}`, and the fitted
`FittedKBinsDiscretizer<F> { bin_edges: Vec<Vec<F>>, n_bins_per_feature: Vec<usize>,
n_bins: usize, encode }` (accessors `bin_edges()` / `n_bins_per_feature()` /
`n_bins()` / `encode()`; `n_bins_per_feature` mirrors sklearn `n_bins_`). `Fit<Array2<F>, ()> for
KBinsDiscretizer` (`fn fit in kbins_discretizer.rs`) rejects `n_samples < 2`
(`InsufficientSamples`) and `n_bins < 2` (`InvalidParameter`), then per column sorts
the values and computes edges: `Uniform` = `min + (max-min)*i/n_bins` (== `np.linspace`),
`Quantile` = linear-interp percentile `pos = frac*(n-1)` (== `np.percentile`), `KMeans`
= `kmeans_1d` (uniform-midpoint init, Lloyd `max_iter=100`, converge max-shift `<1e-10`,
edges = `[min, sorted-center-midpoints.., max]`). A CONSTANT column (`min_val == max_val`) instead pushes `bin_edges=[-inf, +inf]` with `n_bins_per_feature[j]=1` (== sklearn `:262-268`); for `Quantile`/`KMeans` the edges are then collapsed by keeping the first edge and each later edge only when `edge - last > 1e-8` (== `ediff1d > 1e-8` `:302-312`), setting `n_bins_per_feature[j] = kept.len()-1`. The free helper `fn assign_bin in
kbins_discretizer.rs` returns the first edge index `i>=1` with `value < edge` minus one,
else the last bin — the `searchsorted(side="right")` equivalent. `Transform<Array2<F>>
for FittedKBinsDiscretizer` (`fn transform in kbins_discretizer.rs`) checks the column
count (`ShapeMismatch`), then for `Ordinal` writes the bin index per `(i,j)` and for
`OneHot` writes a `1.0` at `offsets[j] + bin` in a dense block whose width is
`sum(n_bins_per_feature)`, where `offsets[j]` is the cumulative sum of the preceding
features' bin counts — sklearn's variable `n_bins_` one-hot. The unfitted
`Transform for KBinsDiscretizer` is an error stub (`InvalidParameter`) satisfying the
`FitTransform: Transform` supertrait; `FitTransform for KBinsDiscretizer` wraps
fit→transform. There is **no `PipelineTransformer` impl** (unlike the sibling
`SimpleImputer`). The grandfathered boundary re-export at `lib.rs:151` (`pub use
kbins_discretizer::{BinEncoding, BinStrategy, FittedKBinsDiscretizer,
KBinsDiscretizer}`) is the non-test production consumer pinning REQ-1 / REQ-2 /
REQ-3 / REQ-4 / REQ-5.

**sklearn (target contract).** `KBinsDiscretizer(TransformerMixin, BaseEstimator)`
(`_discretization.py:184`) takes `__init__(n_bins=5, *, encode="onehot",
strategy="quantile", dtype=None, subsample=200_000, random_state=None)` (`:184-199`).
`fit` (`:202-327`) validates numeric data (`:226`), optionally subsamples
(`:242-249`), resolves a per-feature `n_bins` array via `_validate_n_bins`
(`:252,329-352`), then per column: if `col_min == col_max` warns, sets `n_bins_[jj]=1`,
`bin_edges_[jj]=[-inf, inf]`, and continues (`:262-268`); else computes `uniform`
(`np.linspace`, `:271`), `quantile` (`np.percentile` or `_weighted_percentile`,
`:273-284`), or `kmeans` (`KMeans(init=midpoints, n_init=1)`, sorted-center midpoints,
`:285-300`); for `quantile`/`kmeans` removes sub-`1e-8`-width bins and shrinks
`n_bins_[jj]` (`:302-312`); stores `bin_edges_` / `n_bins_` (`:314-315`) and fits a
`OneHotEncoder(categories=[arange(i) for i in n_bins_], sparse_output=encode=='onehot')`
(`:317-325`). `transform` (`:354-391`) `check_is_fitted`s (`:369`), then per column
`Xt[:,jj] = np.searchsorted(bin_edges[jj][1:-1], Xt[:,jj], side="right")` (`:377`) and
returns the ordinal indices (`:379-380`) or the encoder output (`:383-391`).
`inverse_transform` (`:393`) maps indices to bin centers.

**The gap.** ferrolearn matches sklearn on the *uniform / quantile edge algebra and
ordinal/dense-one-hot binning* (REQ-1, Probes 1-2), on *kmeans for well-separated data*
(REQ-2, Probe 4, same deterministic init), on the scoped structural contracts (REQ-3),
and — after the per-feature variable bin-count fix (`n_bins_per_feature: Vec<usize>`,
sklearn `n_bins_`) — on the previously-headline *constant feature → bin 0 +
`n_bins_per_feature[j]=1`* (REQ-4, was DIV-1 / #1376, RESOLVED) and the
*quantile/kmeans small-bin collapse `gap > 1e-8` + one-hot variable width
`sum(n_bins_per_feature)`* (REQ-5, was DIV-2 / #1377, RESOLVED). The remaining gaps are
configuration / surface: no `subsample`/`random_state` (REQ-6), no per-feature `n_bins`
ARGUMENT array + `_validate_n_bins` (REQ-7; note the resolved per-feature `n_bins_`
OUTPUT now exists), `encode='onehot'` is dense-only and the ctor defaults diverge
(REQ-8), no `dtype`/`sample_weight` (REQ-9), no `inverse_transform` (REQ-10), no
`get_feature_names_out` / sklearn attr names / `PipelineTransformer` (REQ-11), no PyO3
binding (REQ-12), and the non-ferray substrate (REQ-13). One scoped carve-out remains:
KMeans EXACT parity on degenerate/duplicate-heavy data (REQ-14 / #1378) — sklearn's
Lloyd relocates empty clusters, `kmeans_1d` does not; NOT-STARTED with NO committed
failing test per R-DEFER-3. This is a **shipped-partial** unit (5 SHIPPED /
9 NOT-STARTED).

## Verification

Commands establishing the SHIPPED claims (REQ-1 uniform/quantile edges + ordinal/onehot
values, REQ-2 kmeans on well-separated data, REQ-3 scoped error contracts, REQ-4
constant→bin 0 + `n_bins_per_feature[j]=1`, REQ-5 small-bin collapse + onehot variable
width):

```bash
# Consumer / module wiring check:
grep -rn "pub mod kbins_discretizer" ferrolearn-preprocess/src/lib.rs       # :101
grep -rn "pub use kbins_discretizer::" ferrolearn-preprocess/src/lib.rs     # :151 boundary re-export consumer

# REQ-1 / REQ-2 / REQ-3 (in-module tests):
cargo test -p ferrolearn-preprocess kbins
#   REQ-1: test_kbins_ordinal_uniform (bin 0 / bin 2), test_kbins_bin_edges (edges [0,2,4,6]),
#          test_kbins_quantile_strategy, test_kbins_onehot_uniform (one 1.0/row),
#          test_kbins_multi_feature, test_kbins_ordinal_values_in_range, test_kbins_fit_transform,
#          test_kbins_default
#   REQ-2: test_kbins_kmeans_strategy (well-separated 3-cluster input)
#   REQ-3: test_kbins_insufficient_samples_error, test_kbins_too_few_bins_error,
#          test_kbins_shape_mismatch_error, test_kbins_unfitted_error

# REQ-4 / REQ-5 (divergence + re-audit suite, 18 tests — full gauntlet GREEN post-fix):
cargo test -p ferrolearn-preprocess --test divergence_kbins_discretizer
#   REQ-4 (DIV-1 RESOLVED, was #1376): divergence_div1_constant_feature_uniform,
#          divergence_div1_constant_feature_quantile, divergence_div1_constant_feature_multi,
#          reaudit_g_f32_constant_column
#   REQ-5 (DIV-2 RESOLVED, was #1377): divergence_div2_quantile_small_bin_removal,
#          reaudit_a_onehot_constant_among_normal, reaudit_b_onehot_quantile_collapse,
#          reaudit_d_multi_feature_mixed, reaudit_e1_threshold_below_collapses,
#          reaudit_e2_threshold_above_kept, reaudit_f_quantile_no_spurious_collapse
#   REQ-1/REQ-2 green guards: green_uniform_edges_and_ordinal, green_quantile_edges_and_ordinal,
#          green_kmeans_well_separated, green_onehot_uniform, green_transform_edge_cases,
#          green_f32_uniform_ordinal, green_error_contracts
cargo clippy -p ferrolearn-preprocess --all-targets -- -D warnings
cargo fmt --all --check

# REQ-1 oracle gate — sklearn uniform + quantile edges / ordinal values:
python3 -c "import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
X=np.array([[0.],[1.],[2.],[3.],[4.],[5.]])
kb=KBinsDiscretizer(n_bins=3,encode='ordinal',strategy='uniform').fit(X)
print('uniform edges=',[e.tolist() for e in kb.bin_edges_],'ordinal=',kb.transform(X).ravel().tolist())
X2=np.array([[float(i)] for i in range(8)])
kb2=KBinsDiscretizer(n_bins=4,encode='ordinal',strategy='quantile').fit(X2)
print('quantile edges=',[e.tolist() for e in kb2.bin_edges_],'ordinal=',kb2.transform(X2).ravel().tolist())"
#   -> uniform edges= [[0.0, 1.6666666666666667, 3.3333333333333335, 5.0]] ordinal= [0.0, 0.0, 1.0, 1.0, 2.0, 2.0]
#   -> quantile edges= [[0.0, 1.75, 3.5, 5.25, 7.0]] ordinal= [0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0]
#   (ferrolearn test_kbins_ordinal_uniform / test_kbins_quantile_strategy)

# REQ-2 oracle gate — kmeans on well-separated data (MATCHES, same deterministic init):
python3 -c "import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
X=np.array([[0.],[0.1],[0.2],[5.],[5.1],[5.2],[10.],[10.1],[10.2]])
kb=KBinsDiscretizer(n_bins=3,encode='ordinal',strategy='kmeans').fit(X)
print('kmeans edges=',[np.round(e,4).tolist() for e in kb.bin_edges_],'transform=',kb.transform(X).ravel().tolist())"
#   -> kmeans edges= [[0.0, 2.6, 7.6, 10.2]] transform= [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0]
#   (ferrolearn test_kbins_kmeans_strategy)

# REQ-4 oracle gate (DIV-1 RESOLVED — sklearn AND ferrolearn now bin the constant feature to 0):
python3 -c "import numpy as np, warnings; warnings.simplefilter('ignore')
from sklearn.preprocessing import KBinsDiscretizer
X=np.array([[5.],[5.],[5.]])
kb=KBinsDiscretizer(n_bins=3,encode='ordinal',strategy='uniform').fit(X)
print('sklearn const edges=',[e.tolist() for e in kb.bin_edges_],'n_bins_=',kb.n_bins_.tolist(),'transform=',kb.transform(X).ravel().tolist())"
#   -> sklearn const edges= [[-inf, inf]] n_bins_= [1] transform= [0.0, 0.0, 0.0]
#   ferrolearn now MATCHES: bin_edges()[0]==[-inf,+inf], n_bins_per_feature()==[1], transform [0,0,0].
#   (DIV-1 was #1376, RESOLVED; pinned by divergence_div1_constant_feature_uniform.)

# REQ-5 oracle gate (DIV-2 RESOLVED — quantile small-bin collapse, n_bins_ shrinks):
python3 -c "import numpy as np, warnings; warnings.simplefilter('ignore')
from sklearn.preprocessing import KBinsDiscretizer
X=np.array([[0.],[0.],[0.],[0.],[1.],[2.]])
kb=KBinsDiscretizer(n_bins=4,encode='ordinal',strategy='quantile').fit(X)
print('sklearn edges=',[e.tolist() for e in kb.bin_edges_],'n_bins_=',kb.n_bins_.tolist(),'transform=',kb.transform(X).ravel().tolist())"
#   -> sklearn edges= [[0.0, 0.75, 2.0]] n_bins_= [2] transform= [0.0, 0.0, 0.0, 0.0, 1.0, 1.0]
#   ferrolearn now MATCHES: edges [0,0.75,2], n_bins_per_feature()==[2], transform [0,0,0,0,1,1].
#   (DIV-2 was #1377, RESOLVED; pinned by divergence_div2_quantile_small_bin_removal.)
```

The in-module `#[test]`s plus the 18-test `tests/divergence_kbins_discretizer.rs`
gauntlet (GREEN) exercise REQ-1 (uniform / quantile edges + ordinal & dense one-hot
binning), REQ-2 (kmeans on well-separated data), REQ-3 (every error path —
insufficient-samples, too-few-bins, shape-mismatch, unfitted), REQ-4 (constant→bin 0 +
`n_bins_per_feature[j]=1`, DIV-1 RESOLVED), and REQ-5 (quantile/kmeans small-bin
collapse + one-hot variable width, DIV-2 RESOLVED). No green ferrolearn command
establishes REQ-6..REQ-13 (`subsample`/`random_state`, per-feature `n_bins` argument,
sparse one-hot default, `dtype`/`sample_weight`, `inverse_transform`,
`get_feature_names_out` / `PipelineTransformer`, PyO3, ferray) or the REQ-14 carve-out
(#1378, KMeans EXACT parity on degenerate/duplicate-heavy data — NOT-STARTED with NO
committed failing test per R-DEFER-3).

## Blockers

REQ-1 (uniform/quantile edges + ordinal/onehot values, HEADLINE), REQ-2 (kmeans on
well-separated data, scoped), REQ-3 (scoped error / parameter contracts), REQ-4
(constant feature → bin 0 + `n_bins_per_feature[j]=1`, was DIV-1 / #1376, RESOLVED),
and REQ-5 (quantile/kmeans small-bin collapse + one-hot variable width, was DIV-2 /
#1377, RESOLVED) are SHIPPED, with the boundary re-export at `lib.rs:151` as the
grandfathered (S5 / R-DEFER-1) non-test production consumer.

DIV-1 (REQ-4 constant feature → bin 0, was blocker #1376) is **RESOLVED**: `fn fit in
kbins_discretizer.rs` now sets `bin_edges=[-inf,+inf]` + `n_bins_per_feature[j]=1`
(`:262-268`) so `assign_bin` returns bin 0, matching sklearn `[0,0,0]`. DIV-2 (REQ-5
quantile/kmeans small-bin removal, was blocker #1377) is **RESOLVED**: `fit` collapses
edges with `gap > 1e-8` (`ediff1d > 1e-8`, `:302-312`) reducing `n_bins_per_feature[j]`,
and `transform` one-hot width is `sum(n_bins_per_feature)` with cumulative offsets. Both
were enabled by the per-feature `n_bins_per_feature: Vec<usize>` field (sklearn
`n_bins_`). The one residual KMeans carve-out is REQ-14 / #1378 below.

The RESOLVED blockers (no longer open):

- #1376 — REQ-4: constant feature → bin 0 + `n_bins_per_feature[j]=1` + `[-inf,+inf]`
  edges (DIV-1 HEADLINE; `_discretization.py:262-268`). **RESOLVED.**
- #1377 — REQ-5: quantile/kmeans small-bin removal + per-feature `n_bins_per_feature`
  shrink + one-hot variable width (DIV-2; `:302-312`). **RESOLVED.**

The remaining REQs are NOT-STARTED, filed as `-l blocker` issues against tracking
issue #1375:

- #1379 — REQ-6: `subsample` (default 200000) + `random_state` resample (`:191-192,
  242-249`).
- #1380 — REQ-7: per-feature `n_bins` array + `_validate_n_bins` + `n_bins_` attribute
  (`:329-352,315`) — structural blocker under REQ-4 / REQ-5.
- #1381 — REQ-8: `encode='onehot'` SPARSE default + sklearn ctor defaults
  (`encode="onehot"`, `strategy="quantile"`; `:184-189,317-321`).
- #1382 — REQ-9: `dtype` output selector + `sample_weight` (weighted quantile/kmeans;
  `:228-231,277-283,295`).
- #1383 — REQ-10: `inverse_transform` (`:393`).
- #1384 — REQ-11: `get_feature_names_out` + `bin_edges_`/`n_bins_` attr names +
  `PipelineTransformer` impl (`:314-315`; absent unlike sibling `SimpleImputer`).
- #1385 — REQ-12: no PyO3 `KBinsDiscretizer` binding in `ferrolearn-python`.
- #1386 — REQ-13: fit/transform on `ndarray` / `num_traits` / per-column `Vec<F>`, not
  ferray (R-SUBSTRATE-1/2).
- #1378 — REQ-14 (carve-out): KMeans EXACT parity on degenerate/duplicate-heavy data.
  `kmeans_1d` keeps empty clusters; sklearn's Lloyd KMeans (`sklearn/cluster/_kmeans.py`)
  relocates them, so on duplicate-heavy data sklearn's centers coincide → near-duplicate
  edges → small-bin collapse fires (`n_bins_` reduced) while ferrolearn's stay spread →
  no collapse. Well-separated data MATCHES (REQ-2). Same class as the `cluster/kmeans.rs`
  numpy-RNG carve-outs; needs a sklearn-faithful 1D KMeans with empty-cluster relocation.
  NOT-STARTED with **NO committed failing test** (the degenerate pin was removed per
  R-DEFER-3 — the blocker is the open work item, not a red test).
