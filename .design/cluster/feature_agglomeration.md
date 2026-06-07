# Feature Agglomeration (sklearn.cluster.FeatureAgglomeration)

<!--
tier: 3-component
status: draft
baseline-commit: a40dff21
upstream: scikit-learn 1.5.2 (commit 156ef14)
upstream-paths:
  - sklearn/cluster/_agglomerative.py            # class FeatureAgglomeration(AgglomerationTransform, AgglomerativeClustering) (:1132); _parameter_constraints (:1280-1294); __init__ (:1296-1319, pooling_func=np.mean :1305); fit → _validate_data(ensure_min_features=2) + super()._fit(X.T) (:1338-1341); AgglomerativeClustering._fit (:992-1106); n_clusters_/n_leaves_/n_connected_components_/children_/distances_ (:1083-1095); fitted labels_ relabel _hc_cut / np.searchsorted (:1099/:1105)
  - sklearn/cluster/_feature_agglomeration.py    # class AgglomerationTransform(TransformerMixin) (:22); transform mean-bincount fast path (:51-57) + general pooling_func(X[:, labels_==l], axis=1) ordered by np.unique(labels_) (:58-63); inverse_transform broadcast via np.unique(return_inverse) (:66-92)
ferrolearn-module: ferrolearn-cluster/src/feature_agglomeration.rs
parity-ops: FeatureAgglomeration (.__init__, .fit, .transform, .inverse_transform, .labels_, .n_clusters_, .n_leaves_, .n_connected_components_, .children_, .distances_)
crosslink-issue: 937
-->

## Summary

`ferrolearn-cluster/src/feature_agglomeration.rs` mirrors scikit-learn's
`FeatureAgglomeration` (`sklearn/cluster/_agglomerative.py`, `class
FeatureAgglomeration(AgglomerationTransform, AgglomerativeClustering)` `:1132`) —
hierarchical clustering of **features** (columns), each cluster pooled to a single
column on `transform`. It exposes the unfitted `FeatureAgglomeration<F>`
(`n_clusters` required, `linkage: AgglomerativeLinkage = Ward`, `pooling_func:
PoolingFunc = Mean`), the fitted `FittedFeatureAgglomeration<F>` (stores
`feature_labels_` / `n_clusters_` / `n_features_`), and a `Transform` impl. It is
re-exported at the crate root (`pub use feature_agglomeration::{AgglomerativeLinkage,
FeatureAgglomeration, FittedFeatureAgglomeration, PoolingFunc}` in
`ferrolearn-cluster/src/lib.rs`). `fit` transposes `X`, delegates to
`AgglomerativeClustering::new(n_clusters).with_linkage(...)` on `X.T` (so
`feature_labels_` is the per-feature cluster assignment), then `transform` pools each
cluster (Mean = sum/count, Max) into an output of shape `(n_samples, n_clusters)`,
columns ordered by label index.

**Under honest underclaim (R-HONEST-3) the load-bearing parity claims — `labels_`
VALUE and `transform` VALUE — DIVERGE from the live sklearn 1.5.2 oracle, even
though the feature *partition* matches.** The probes below show: on the
`make_correlated_features` 5×6 test fixture, ferrolearn groups features into exactly
the same three sets `{0,1}`, `{2,3}`, `{4,5}` that sklearn does — but assigns
**different integer labels** to those sets. sklearn (via `_hc_cut`, `:1099`, which
numbers clusters in dendrogram-cut order, then `np.searchsorted(np.unique(...))`,
`:1105`) yields `labels_ = [0,0,2,2,1,1]`; ferrolearn (relabel by `active`-slot
position in `agglomerative.rs::agglomerate`) yields `[0,0,1,1,2,2]`. Because
sklearn's `transform` orders output columns by `np.unique(self.labels_)` =
`[0,1,2]` (`_feature_agglomeration.py:62`), the **column order diverges**: sklearn
MEAN row 0 = `[1.05, 9.05, 5.05]` vs ferrolearn `[1.05, 5.05, 9.05]`. As a *set* of
columns the two agree (a label permutation lines them up), but the
element-by-element `transform(X)[i,j]` contract — which is what a user compares —
does **not** value-match. The label-index divergence is owned DOWN in
`agglomerative.rs` (the `_hc_cut` re-labeling convention), so the value parity of
this unit is gated on the AgglomerativeClustering unit reaching label parity.

`FeatureAgglomeration` / `FittedFeatureAgglomeration` are existing pub APIs
(grandfathered per S5/R-DEFER-1); their **only** non-test consumer is the crate
re-export (`lib.rs`). There is **no `ferrolearn-python` binding**
(`grep -rln FeatureAgglomeration ferrolearn-python/` is empty) and no in-crate
pipeline consumer.

## Live oracle probes (sklearn 1.5.2, run from /tmp; ferrolearn via throwaway `cargo run --example` probe, since deleted)

Fixture = the `make_correlated_features` 5×6 matrix from `feature_agglomeration.rs`
tests (features paired (0,1)/(2,3)/(4,5)).

### Probe 1 — `labels_` feature-cluster VALUE parity (the crux)

```
python3 -c "import numpy as np; from sklearn.cluster import FeatureAgglomeration; \
X=np.array([[1.,1.1,5.,5.1,9.,9.1],[2.,2.1,6.,6.1,8.,8.1],[3.,3.1,7.,7.1,7.,7.1],[4.,4.1,8.,8.1,6.,6.1],[5.,5.1,9.,9.1,5.,5.1]]); \
[print(lk, FeatureAgglomeration(n_clusters=3, linkage=lk).fit(X).labels_.tolist()) for lk in ['ward','complete','average','single']]"
# ward     [0, 0, 2, 2, 1, 1]
# complete [0, 0, 2, 2, 1, 1]
# average  [0, 0, 2, 2, 1, 1]
# single   [0, 0, 2, 2, 1, 1]
```
ferrolearn `FeatureAgglomeration::<f64>::new(3).with_linkage(...).fit(&X).feature_labels()`:
```
# Ward     [0, 0, 1, 1, 2, 2]
# Complete [0, 0, 1, 1, 2, 2]
# Average  [0, 0, 1, 1, 2, 2]
# Single   [0, 0, 1, 1, 2, 2]
```
**Partition AGREES** (both: `{0,1}`, `{2,3}`, `{4,5}`); **label INDEX DIVERGES** —
sklearn maps `{2,3}→2, {4,5}→1`; ferrolearn maps `{2,3}→1, {4,5}→2`. This is **not**
a value-parity match: `labels_[2]` is `2` in sklearn, `1` in ferrolearn. Root cause:
sklearn cuts the dendrogram via `_hc_cut` (`_agglomerative.py:1099`, a max-heap pop
order over `children_`) then `np.searchsorted(np.unique(labels), labels)` (`:1105`);
ferrolearn re-labels by `active`-slot index in `agglomerative.rs::agglomerate`
(`for (new_id, &cluster_id) in active.iter().enumerate()`). Different conventions →
different integer labels for the same partition. Owned by the AgglomerativeClustering
unit (`agglomerative.rs`), not this file.

### Probe 2 — `transform` VALUE, mean pooling

```
python3 -c "import numpy as np; from sklearn.cluster import FeatureAgglomeration; \
X=np.array([[1.,1.1,5.,5.1,9.,9.1],[2.,2.1,6.,6.1,8.,8.1],[3.,3.1,7.,7.1,7.,7.1],[4.,4.1,8.,8.1,6.,6.1],[5.,5.1,9.,9.1,5.,5.1]]); \
print(FeatureAgglomeration(n_clusters=3, pooling_func=np.mean).fit(X).transform(X).tolist())"
# [[1.05, 9.05, 5.05], [2.05, 8.05, 6.05], [3.05, 7.05, 7.05], [4.05, 6.05, 8.05], [5.05, 5.05, 9.05]]
```
ferrolearn Mean transform:
```
# [[1.05, 5.05, 9.05], [2.05, 6.05, 8.05], [3.05, 7.05, 7.05], [4.05, 8.05, 6.05], [5.05, 9.05, 5.05]]
```
**Column ORDER DIVERGES.** Both pool the SAME three groups (the pooled VALUES are
identical as a set: `{1.05, 9.05, 5.05}` per row), but sklearn orders columns by
`np.unique(labels_) = [0,1,2]` so column 1 = cluster 1 = features `{4,5}` (= `9.05`),
column 2 = cluster 2 = features `{2,3}` (= `5.05`); ferrolearn writes cluster `1` =
`{2,3}` into column 1 (= `5.05`). Element-wise `transform(X)[0,1]` = `9.05` (sklearn)
vs `5.05` (ferrolearn) → the contract does NOT value-match. A permutation of columns
makes them equal as a SET, but `transform` is a column-ordered contract. The pooling
ARITHMETIC itself is correct (Mean = `np.bincount(labels, X[i]) / bincount(labels)`,
`_feature_agglomeration.py:51-57`); the divergence is entirely the label-index
ordering inherited from Probe 1.

### Probe 3 — `transform` VALUE, max pooling

```
python3 -c "import numpy as np; from sklearn.cluster import FeatureAgglomeration; \
X=np.array([[1.,1.1,5.,5.1,9.,9.1],[2.,2.1,6.,6.1,8.,8.1],[3.,3.1,7.,7.1,7.,7.1],[4.,4.1,8.,8.1,6.,6.1],[5.,5.1,9.,9.1,5.,5.1]]); \
print(FeatureAgglomeration(n_clusters=3, pooling_func=np.max).fit(X).transform(X).tolist())"
# [[1.1, 9.1, 5.1], [2.1, 8.1, 6.1], [3.1, 7.1, 7.1], [4.1, 6.1, 8.1], [5.1, 5.1, 9.1]]
```
ferrolearn Max transform: `[[1.1, 5.1, 9.1], [2.1, 6.1, 8.1], [3.1, 7.1, 7.1],
[4.1, 8.1, 6.1], [5.1, 9.1, 5.1]]`. Same story: per-cluster max is correct, column
order diverges (sklearn `transform(X)[0,1]=9.1` vs ferrolearn `5.1`). Note sklearn's
`np.max` callable takes the **general** pooling path (`_feature_agglomeration.py:58-63`,
only `np.mean` hits the fast `bincount` path), still ordered by `np.unique(labels_)`.

### Probe 4 — defaults / params

- **`n_clusters` default**: sklearn `n_clusters=2` (`_agglomerative.py:1298`);
  ferrolearn **requires** it (`fn new(n_clusters: usize)`) — no default.
- **`pooling_func` default**: sklearn `np.mean` (`:1305`); ferrolearn `PoolingFunc::Mean`
  (`fn new`) — matches.
- **`linkage` default**: sklearn `'ward'` (`:1304`); ferrolearn `AgglomerativeLinkage::Ward`
  (`fn new`) — matches.
- **missing params**: `metric='euclidean'`, `memory`, `connectivity`,
  `compute_full_tree='auto'`, `distance_threshold=None`, `compute_distances=False`
  (`:1300-1307`) are ALL absent from `FeatureAgglomeration<F>` (which has only
  `n_clusters`/`linkage`/`pooling_func`). `distance_threshold` is significant:
  sklearn allows `n_clusters=None` + `distance_threshold` to cut the tree by distance
  (`:1281` constraint `None` allowed; `n_clusters_` computed from `distances_`,
  `:1091-1092`).
- **`pooling_func` as arbitrary callable**: sklearn accepts any `callable`
  (`_parameter_constraints["pooling_func"] = [callable]`, `:1291`); ferrolearn offers
  only the `PoolingFunc::{Mean, Max}` enum.

### Probe 5 — validation guards

```
python3 -c "import numpy as np; from sklearn.cluster import FeatureAgglomeration; \
FeatureAgglomeration(n_clusters=1).fit(np.array([[1.],[2.],[3.]]))"
# ValueError: Found array with 1 feature(s) (shape=(3, 1)) while a minimum of 2 is required
python3 -c "import numpy as np; from sklearn.cluster import FeatureAgglomeration; \
FeatureAgglomeration(n_clusters=10).fit(np.random.rand(5,6))"
# ValueError: Cannot extract more clusters than samples: 10 clusters were given for a tree with 6 leaves
```
- sklearn requires `ensure_min_features=2` (`_agglomerative.py:1338`) — **rejects a
  single feature**. ferrolearn imposes no min-features guard (only `n_features >=
  n_clusters`, `fn fit`), so `n_clusters=1` on a 1-column `X` is **accepted** —
  DIVERGES.
- `n_clusters > n_features`: BOTH reject. sklearn → `ValueError` ("Cannot extract
  more clusters than samples", from `_hc_cut`); ferrolearn → `FerroError::InvalidParameter`
  ("n_clusters (N) exceeds n_features (M)", `fn fit`) — same intent, different error
  ABI/message (R-DEV-2 ValueError ABI not matched).
- `n_clusters == 0`: ferrolearn → `FerroError::InvalidParameter` (`fn fit`); sklearn's
  `_parameter_constraints` requires `n_clusters >= 1` (`Interval(Integral, 1, None)`,
  `:1281`) → `InvalidParameterError`. ferrolearn rejects (correct intent), wrong ABI.
- empty `X` (`n_samples == 0`): ferrolearn → `FerroError::InsufficientSamples`
  (`fn fit`); sklearn `_validate_data` rejects 0 samples too.

### Probe 6 — `inverse_transform` (missing surface)

```
python3 -c "import numpy as np; from sklearn.cluster import FeatureAgglomeration; \
X=np.array([[1.,1.1,5.,5.1,9.,9.1],[2.,2.1,6.,6.1,8.,8.1],[3.,3.1,7.,7.1,7.,7.1],[4.,4.1,8.,8.1,6.,6.1],[5.,5.1,9.,9.1,5.,5.1]]); \
fa=FeatureAgglomeration(n_clusters=3).fit(X); print(fa.inverse_transform(fa.transform(X)).tolist())"
# [[1.05,1.05,5.05,5.05,9.05,9.05], ...]  broadcasts each pooled column back to its features
```
sklearn `AgglomerationTransform.inverse_transform` (`_feature_agglomeration.py:66-92`)
broadcasts pooled values back to original feature positions via
`X[..., np.unique(labels_, return_inverse=True)[1]]` (`:91-92`). ferrolearn has **no
`inverse_transform`** — `FittedFeatureAgglomeration` impls only `Transform`. Missing
surface.

### Probe 7 — fitted attributes

sklearn (`_agglomerative.py:1083-1095`): `labels_` (feature labels),
`n_clusters_`, `n_leaves_` (= `n_features`), `n_connected_components_`,
`children_`, `distances_` (only if `compute_distances=True`). Probe:
`n_leaves_=6, n_clusters_=3, n_connected_components_=1, children_=[[2,3],[4,5],[0,1],[6,7],[8,9]],
hasattr distances_=False`. ferrolearn exposes `feature_labels_` (≈ `labels_` but
permuted, Probe 1), `n_clusters_`, `n_features_` — **missing** `n_leaves_`,
`n_connected_components_`, `children_`, `distances_`, and the sklearn attribute name
is `labels_` not `feature_labels_` (R-DEV-3 name mismatch).

### Probe 8 — non-test consumer

`grep -rln "FeatureAgglomeration" ferrolearn-python/` is **empty** — no PyO3 binding,
so `import ferrolearn` cannot reach `FeatureAgglomeration`. `grep -rn
FeatureAgglomeration ferrolearn-cluster/src/` outside `feature_agglomeration.rs`
finds only the crate re-export (`lib.rs:104-105`) and doc-comment references. There
is no pipeline / `PipelineEstimator` consumer in the crate. The sole non-test
consumer of `fit`/`transform`/`feature_labels()` is the crate re-export.

## Requirements

- REQ-1: **`transform` mean-pooling VALUE (R-DEV-1/3 — the core transform contract).**
  Mirror `AgglomerationTransform.transform` mean fast path
  (`_feature_agglomeration.py:51-57`): output `(n_samples, n_clusters)`, column `l` =
  mean of `X[:, labels_==l]`, columns ordered by `np.unique(labels_)`. ferrolearn
  computes the right per-cluster mean ARITHMETIC, but column ordering inherits the
  divergent label index (REQ-3) → `transform(X)[i,j]` diverges value-wise (Probe 2).
- REQ-2: **`transform` max-pooling VALUE (R-DEV-1/3).** Mirror the general pooling
  path with `pooling_func=np.max` (`_feature_agglomeration.py:58-63`). Per-cluster max
  is correct; column order diverges identically to REQ-1 (Probe 3).
- REQ-3: **`labels_` feature-cluster VALUE parity (R-DEV-1/3 — the load-bearing,
  upstream-owned requirement).** Mirror `FeatureAgglomeration.labels_` = the
  per-feature cluster assignment from `AgglomerativeClustering._fit(X.T)` +
  `_hc_cut`/`np.searchsorted` re-labeling (`_agglomerative.py:1099/1105`). ferrolearn's
  partition matches but its integer labels are PERMUTED (Probe 1) because
  `agglomerative.rs::agglomerate` re-labels by `active`-slot order, not by sklearn's
  dendrogram-cut order. The fix is owned by the **AgglomerativeClustering unit**
  (`agglomerative.rs`); this REQ is NOT-STARTED pending that unit.
- REQ-4: **validation guards — `n_clusters >= 1`, `n_features >= n_clusters`,
  `ensure_min_features=2`, empty-X (R-DEV-2).** sklearn requires
  `n_clusters >= 1` (`:1281`), `ensure_min_features=2` (`:1338`), and rejects
  `n_clusters > n_features` (`_hc_cut`). ferrolearn rejects `n_clusters==0`,
  `n_features < n_clusters`, and empty X — but has **no `ensure_min_features=2`
  guard** (accepts a single feature, Probe 5) and uses `FerroError` not the sklearn
  `ValueError`/`InvalidParameterError` ABI.
- REQ-5: **linkage variants `ward`/`complete`/`average`/`single` (R-DEV-2).** sklearn
  `_TREE_BUILDERS` (`:1290`). ferrolearn maps `AgglomerativeLinkage` → `Linkage` and
  delegates to `AgglomerativeClustering`. The partition matches on the fixture (Probe
  1) but `labels_`/`transform` VALUE is gated on REQ-3.
- REQ-6: **`n_clusters=2` default + missing params `metric`/`memory`/`connectivity`/
  `compute_full_tree`/`distance_threshold`/`compute_distances` (R-DEV-2).** sklearn
  `__init__` (`:1296-1319`). ferrolearn requires `n_clusters` (no default) and has
  only `linkage`/`pooling_func`. `distance_threshold` (cut by distance with
  `n_clusters=None`) is materially absent.
- REQ-7: **`pooling_func` as arbitrary callable (R-DEV-2).** sklearn accepts any
  `callable` (`:1291`); ferrolearn offers only the `PoolingFunc::{Mean, Max}` enum.
- REQ-8: **`inverse_transform` (R-DEV-3 — missing surface).** sklearn
  `AgglomerationTransform.inverse_transform` broadcasts pooled values back to original
  feature positions (`_feature_agglomeration.py:66-92`). ferrolearn has none.
- REQ-9: **fitted attributes `labels_`/`n_leaves_`/`n_connected_components_`/
  `children_`/`distances_` (R-DEV-3).** sklearn `:1083-1095`. ferrolearn exposes
  `feature_labels_`/`n_clusters_`/`n_features_` — wrong primary name (`feature_labels_`
  vs `labels_`), and missing `n_leaves_`/`n_connected_components_`/`children_`/`distances_`.
- REQ-10: **PyO3 binding (R-DEFER-1/3).** No `_RsFeatureAgglomeration` in
  `ferrolearn-python` — `import ferrolearn` cannot reach `FeatureAgglomeration`.
- REQ-11: **ferray substrate (R-SUBSTRATE).** `feature_agglomeration.rs` imports
  `ndarray::{Array1, Array2}` + `num_traits::Float`, not `ferray-core`. The delegated
  `agglomerative.rs` is likewise on `ndarray`.

## Acceptance criteria

All expected values from the live sklearn 1.5.2 oracle (run from `/tmp`), never
literal-copied from ferrolearn (R-CHAR-3). ferrolearn values from a throwaway
`cargo run --example` probe (since deleted). Fixture = `make_correlated_features`
(the 5×6 in `feature_agglomeration.rs` tests).

- AC-1 (REQ-3, diverges): `FeatureAgglomeration(n_clusters=3, linkage='ward').fit(X).labels_`
  = `[0,0,2,2,1,1]`; ferrolearn `feature_labels()` = `[0,0,1,1,2,2]`. Partition
  matches, label index at positions 2,3,4,5 differs → NOT value parity.
- AC-2 (REQ-1, diverges): `FeatureAgglomeration(n_clusters=3, pooling_func=np.mean).fit(X).transform(X)[0]`
  = `[1.05, 9.05, 5.05]`; ferrolearn = `[1.05, 5.05, 9.05]` → `[0,1]` is `9.05` vs
  `5.05` (column order diverges). Equal only as an unordered set.
- AC-3 (REQ-2, diverges): `pooling_func=np.max` → sklearn `transform(X)[0]` =
  `[1.1, 9.1, 5.1]`; ferrolearn `[1.1, 5.1, 9.1]`.
- AC-4 (REQ-4): `FeatureAgglomeration(n_clusters=1).fit(np.array([[1.],[2.],[3.]]))`
  → sklearn `ValueError` (min 2 features); ferrolearn `fit` SUCCEEDS — over-acceptance.
- AC-5 (REQ-6/7/8/9/10): `hasattr(FeatureAgglomeration(), 'metric')` /
  `'distance_threshold'` / `'compute_distances')` True; `FeatureAgglomeration().fit(X)`
  has `n_leaves_`/`children_`/`n_connected_components_`; `fa.inverse_transform(...)`
  works. ferrolearn has none of these, no callable `pooling_func`, no
  `ferrolearn.FeatureAgglomeration` (no binding).

## REQ status table

Binary (R-DEFER-2). `FeatureAgglomeration` / `FittedFeatureAgglomeration` are existing
pub APIs re-exported at the crate root (the only non-test consumer; grandfathered
S5/R-DEFER-1). Cites use symbol anchors (ferrolearn) / `file:line` (sklearn 1.5.2,
commit 156ef14). Live oracle = installed sklearn 1.5.2, run from `/tmp`. **Update
(post-#938, commit 3e001cf4b):** the AgglomerativeClustering unit now ships bit-exact
`_hc_cut` `labels_`; since `FeatureAgglomeration::fit` delegates to
`AgglomerativeClustering::fit(X.T)` (`_agglomerative.py:1339`), the now-correct
`labels_` flow through unchanged, so REQ-1/REQ-2/REQ-3/REQ-5 (transform VALUE +
column order, labels_ VALUE, four-linkage parity) and REQ-9 (delegated fitted attrs)
are now SHIPPED with end-to-end VALUE parity. Suggested blocker numbers — the director
creates the real issues; #937 is this doc's crosslink tracking issue.

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (`transform` mean VALUE) | SHIPPED | impl `fn transform` (PoolingFunc::Mean) computes per-cluster mean (`sum/count`) into `result[[i, label]]`, column j by ASCENDING label index, mirroring the `bincount` fast path (`_feature_agglomeration.py:51-57`, column = label index 0..k-1). With the now-correct delegated `labels_` (#938 shipped), the output is VALUE-EXACT and COLUMN-ORDERED. Test `value_transform_mean_column_ordered` (`tests/divergence_feature_agglomeration_value.rs`): ferrolearn `transform(X)` == sklearn `FeatureAgglomeration(n_clusters=3, pooling_func=np.mean).fit(X).transform(X)` full matrix ~1e-9 (ward+single); row0 `[1.05, 9.05, 5.05]`. Consumer: crate re-export (`lib.rs`). |
| REQ-2 (`transform` max VALUE) | SHIPPED | impl `fn transform` (PoolingFunc::Max) takes per-cluster max into `result[[i, label]]`, ascending label-index column order, mirroring the general pooling path (`_feature_agglomeration.py:58-63`, columns by sorted unique label = 0..k-1). VALUE-EXACT + COLUMN-ORDERED post-#938. Test `value_transform_max_column_ordered`: full-matrix equality vs sklearn `pooling_func=np.max` (ward+complete); row0 `[1.1, 9.1, 5.1]`. Consumer: crate re-export (`lib.rs`). |
| REQ-3 (`labels_` feature-cluster VALUE) | SHIPPED | `fit` delegates to `AgglomerativeClustering::new(n_clusters).with_linkage(...)` on `X.T` (`fn fit`) and stores `fitted_agg.labels_` verbatim into `feature_labels_` (NO post-relabel), mirroring `super()._fit(X.T)` (`_agglomerative.py:1339`). The inner unit now ships bit-exact `_hc_cut` (`agglomerative.rs fn hc_cut`, `np.searchsorted` `:1105`; #938 commit 3e001cf4b), so `feature_labels_`/`labels()` is integer-EXACT: sklearn k=3 `[0,0,2,2,1,1]` / k=2 `[1,1,0,0,0,0]` == ferrolearn. New `fn labels` accessor (sklearn name) + `fn feature_labels` alias. Test `value_labels_exact_all_linkages_k2_k3`. Consumer: crate re-export (`lib.rs`). |
| REQ-4 (validation guards + ensure_min_features=2 + error ABI) | SHIPPED (validation/shape only) | ferrolearn `fn fit` rejects `n_clusters==0`, `n_features < n_clusters`, empty X, AND a single feature (`ensure_min_features=2`, `:1338`). GAP (NOT-STARTED): raises `FerroError::InvalidParameter`/`InsufficientSamples`, not the sklearn `ValueError`/`InvalidParameterError` ABI (R-DEV-2). Pins: `divergence_feature_agglom_min_features_two` (#944) + green guards. |
| REQ-5 (linkage variants ward/complete/average/single) | SHIPPED | `fn map_linkage` maps all four `AgglomerativeLinkage` variants to `Linkage`, delegated to `AgglomerativeClustering` (mirroring `_TREE_BUILDERS`, `_agglomerative.py:720-725`/`:1290`). VALUE parity (labels_ + transform) holds across all four post-#938. Test `value_labels_exact_all_linkages_k2_k3` (all four) + `value_transform_mean_column_ordered` (ward+single) + `value_transform_max_column_ordered` (ward+complete). Consumer: crate re-export (`lib.rs`). |
| REQ-6 (`n_clusters=2` default + missing params metric/memory/connectivity/compute_full_tree/distance_threshold/compute_distances) | NOT-STARTED | open prereq blocker **#942**. sklearn `__init__` (`_agglomerative.py:1296-1319`) takes 9 params with `n_clusters=2` default. `FeatureAgglomeration<F>` REQUIRES `n_clusters` (`fn new`, no default) and has only `linkage`/`pooling_func`. `distance_threshold` (cut by distance, `n_clusters=None`, `:1281`/`:1091-1092`) is materially absent. |
| REQ-7 (`pooling_func` as arbitrary callable) | NOT-STARTED | open prereq blocker **#943**. sklearn `_parameter_constraints["pooling_func"] = [callable]` (`_agglomerative.py:1291`) accepts any callable (default `np.mean`, `:1305`). ferrolearn offers only the closed `PoolingFunc::{Mean, Max}` enum. |
| REQ-8 (`inverse_transform`) | NOT-STARTED | open prereq blocker **#944**. sklearn `AgglomerationTransform.inverse_transform` broadcasts pooled values back via `X[..., np.unique(labels_, return_inverse=True)[1]]` (`_feature_agglomeration.py:66-92`). `FittedFeatureAgglomeration` impls only `Transform` — no `inverse_transform` (Probe 6). |
| REQ-9 (fitted attrs labels_/n_leaves_/n_connected_components_/children_/distances_) | SHIPPED | `fn fit` stores the inner `AgglomerativeClustering::fit(X.T)` attrs into `FittedFeatureAgglomeration` (`children_`, `distances_`, `n_leaves_`, `n_connected_components_`), delegating exactly as sklearn `FeatureAgglomeration._fit` → `AgglomerativeClustering._fit(X.T)` (`_agglomerative.py:1339`, sets the attrs at `:1083-1095`). Accessors: `fn labels` (sklearn name) + `fn feature_labels` alias, `fn children`, `fn distances` (`Option`, `Some` iff `with_compute_distances(true)`, mirrors `compute_distances` `:1319`), `fn n_leaves` (= n_features), `fn n_connected_components` (= 1, unstructured path). Test `value_fitted_attrs_delegated`: `children_` == sklearn `FeatureAgglomeration(compute_distances=True).fit(X).children_`; `distances_` ~1e-9 == sklearn; `n_leaves_ == 6`; `n_connected_components_ == 1`. Consumer: crate re-export (`lib.rs`). |
| REQ-10 (PyO3 binding) | NOT-STARTED | open prereq blocker **#946**. `grep -rln FeatureAgglomeration ferrolearn-python/` is EMPTY — there is no `_RsFeatureAgglomeration`, so `import ferrolearn` cannot reach `FeatureAgglomeration`. The only non-test consumer of `fit`/`transform`/`feature_labels()` is the crate re-export (`lib.rs:104-105`). |
| REQ-11 (ferray substrate) | NOT-STARTED | open prereq blocker **#947**. `feature_agglomeration.rs` imports `ndarray::{Array1, Array2}` + `num_traits::Float`; the delegated `agglomerative.rs` is likewise on `ndarray`. Not migrated to `ferray-core` (R-SUBSTRATE-1/2). |

## Architecture

`feature_agglomeration.rs` follows the unfitted/fitted split (CLAUDE.md naming):
`FeatureAgglomeration<F>` (`n_clusters`, `linkage: AgglomerativeLinkage = Ward`,
`pooling_func: PoolingFunc = Mean`; builders `with_linkage`/`with_pooling_func`) →
`Fit<Array2<F>, ()>` → `FittedFeatureAgglomeration<F>` (private `feature_labels_:
Array1<usize>`, `n_clusters_`, `n_features_`, `pooling_func_`) →
`Transform<Array2<F>>`. Generic over `F: Float + Send + Sync + 'static`; every public
method returns `Result<_, FerroError>` (R-CODE-2). No `Predict` impl (it is a
transformer).

**Fit path (`fn fit`).** Validates `n_clusters > 0`, `n_features >= n_clusters`,
`n_samples >= 1`, then transposes `X` (`x.t().as_standard_layout().into_owned()` so
each feature/column becomes a row), and delegates to
`AgglomerativeClustering::<F>::new(n_clusters).with_linkage(map_linkage(linkage))`,
storing `fitted_agg.labels_` as `feature_labels_`. This mirrors sklearn's `fit` →
`super()._fit(X.T)` (`_agglomerative.py:1339`) faithfully in STRUCTURE — the divergence
is entirely in the delegated AgglomerativeClustering's label-numbering convention
(REQ-3) and the absent `ensure_min_features=2` guard (REQ-4).

**Transform path (`fn transform`).** Validates `x.ncols() == n_features_`
(`FerroError::ShapeMismatch` else), allocates `(n_samples, n_clusters)`. Mean: count
features per cluster, accumulate `result[[i, label]] += x[[i, j]]`, divide by count —
arithmetically equal to sklearn's `np.bincount(labels, X[i]) / np.bincount(labels)`
(`_feature_agglomeration.py:51-57`). Max: init `-inf`, take per-cluster max —
arithmetically equal to `np.max(X[:, labels==l], axis=1)` (`:58-63`). **The pooling
arithmetic is correct; only the column INDEX (which cluster lands in which column)
diverges**, because ferrolearn writes cluster `label` directly into column `label`
while sklearn orders columns by `np.unique(labels_)` AND its `labels_` indices differ
(REQ-3). With identical `labels_`, the two would coincide.

**Invariants held vs sklearn:** the feature PARTITION (which features co-cluster,
Probe 1 — all four linkages on the fixture); the pooling ARITHMETIC (Mean = group
mean, Max = group max); output shape `(n_samples, n_clusters)`; `feature_labels()`
length = `n_features`; labels in `[0, n_clusters)`; shape-mismatch rejection on
transform.

**Invariants NOT held vs sklearn:** `labels_` integer VALUE (REQ-3 — permuted, the
root cause); `transform` column-ordered VALUE (REQ-1/REQ-2 — cascades from REQ-3);
`ensure_min_features=2` + error ABI (REQ-4); `n_clusters=2` default + the
metric/connectivity/distance_threshold/compute_full_tree/compute_distances parameter
surface (REQ-6); callable `pooling_func` (REQ-7); `inverse_transform` (REQ-8); fitted
attributes `labels_`/`n_leaves_`/`n_connected_components_`/`children_`/`distances_`
(REQ-9); the PyO3 binding (REQ-10); the ferray substrate (REQ-11).

**Consumer wiring.** The only non-test consumer is the crate re-export
(`pub use feature_agglomeration::{AgglomerativeLinkage, FeatureAgglomeration,
FittedFeatureAgglomeration, PoolingFunc}`, `ferrolearn-cluster/src/lib.rs:104-105`).
There is no `ferrolearn-python` binding and no in-crate pipeline consumer (Probe 8).

## Verification

Library crate (green at baseline `a40dff21` for the existing structural behavior):
```
cargo test -p ferrolearn-cluster --lib feature_agglom     # 17 passed; 0 failed
cargo clippy -p ferrolearn-cluster --all-targets -- -D warnings
cargo fmt --all --check
```
The 17 in-tree `#[test]`s (`test_feature_agglom_basic`, `..._output_shape`,
`..._labels_valid_range`, `..._correlated_grouped`, `..._mean_pooling`,
`..._max_pooling`, `..._complete_linkage`, `..._average_linkage`,
`..._single_linkage`, `..._n_clusters_equals_n_features`, `..._zero_clusters_error`,
`..._too_many_clusters_error`, `..._empty_data_error`, `..._transform_shape_mismatch`,
`..._f32`, `..._getters`, `..._n_features_getter`) pin ferrolearn's current behavior:
output SHAPE, label co-membership/range, pooling arithmetic on TRIVIAL single-cluster
fixtures (`n_clusters=1`, where column order is unambiguous), and error edges. **None
compares `labels_` or column-ordered `transform` VALUE against the live sklearn
`FeatureAgglomeration` oracle** — that is why they stay green despite the REQ-1/REQ-3
divergences. (`test_feature_agglom_mean_pooling`/`..._max_pooling` use `n_clusters=1`,
a single column, so the column-ordering divergence cannot surface there.)

Live sklearn oracle (installed 1.5.2, run from `/tmp`) — the divergences a critic
should pin (R-CHAR-3 expected values). **Pin REQ-3 (the label-index ordering in
`agglomerative.rs`) FIRST** — it is the single root cause; REQ-1/REQ-2 (transform
column order) unblock once it lands:
```
# REQ-3 (DIVERGES) labels_ index — partition matches, integer labels permuted
python3 -c "import numpy as np; from sklearn.cluster import FeatureAgglomeration; \
X=np.array([[1.,1.1,5.,5.1,9.,9.1],[2.,2.1,6.,6.1,8.,8.1],[3.,3.1,7.,7.1,7.,7.1],[4.,4.1,8.,8.1,6.,6.1],[5.,5.1,9.,9.1,5.,5.1]]); \
print(FeatureAgglomeration(n_clusters=3, linkage='ward').fit(X).labels_.tolist())"
# [0, 0, 2, 2, 1, 1]   ferro: [0, 0, 1, 1, 2, 2]
# REQ-1 (DIVERGES) transform mean column order
python3 -c "import numpy as np; from sklearn.cluster import FeatureAgglomeration; \
X=np.array([[1.,1.1,5.,5.1,9.,9.1],[2.,2.1,6.,6.1,8.,8.1],[3.,3.1,7.,7.1,7.,7.1],[4.,4.1,8.,8.1,6.,6.1],[5.,5.1,9.,9.1,5.,5.1]]); \
print(FeatureAgglomeration(n_clusters=3, pooling_func=np.mean).fit(X).transform(X)[0].tolist())"
# [1.05, 9.05, 5.05]   ferro: [1.05, 5.05, 9.05]
# REQ-4 (DIVERGES) ensure_min_features=2
python3 -c "import numpy as np; from sklearn.cluster import FeatureAgglomeration; \
FeatureAgglomeration(n_clusters=1).fit(np.array([[1.],[2.],[3.]]))"
# ValueError (min 2 features); ferro fit() SUCCEEDS
# REQ-8 (MISSING) inverse_transform
python3 -c "import numpy as np; from sklearn.cluster import FeatureAgglomeration; \
X=np.array([[1.,1.1,5.,5.1,9.,9.1],[2.,2.1,6.,6.1,8.,8.1],[3.,3.1,7.,7.1,7.,7.1],[4.,4.1,8.,8.1,6.,6.1],[5.,5.1,9.,9.1,5.,5.1]]); \
fa=FeatureAgglomeration(n_clusters=3).fit(X); print(fa.inverse_transform(fa.transform(X))[0].tolist())"
# [1.05, 1.05, 5.05, 5.05, 9.05, 9.05]; ferro has no inverse_transform
```
A characterization pin (R-CHAR-3) for each NOT-STARTED REQ belongs in
`ferrolearn-cluster/tests/divergence_feature_agglomeration.rs`, asserting the
live-sklearn expected values above and FAILING against current
`feature_agglomeration.rs`. Because REQ-3's root cause is in `agglomerative.rs`, the
matching pin also belongs in that unit's divergence suite (the label-numbering
convention).

ferrolearn-python (REQ-10 binding parity, after #946 lands):
```
cd /home/doll/ferrolearn/ferrolearn-python && maturin develop
PYTHONPATH=python python3 -m pytest tests/divergence_feature_agglomeration.py -q
```
asserting `ferrolearn.FeatureAgglomeration` exists, exposes `labels_` /
`n_leaves_` / `children_` / `inverse_transform`, and value-matches
`sklearn.cluster.FeatureAgglomeration` on the AC fixtures.

## Blockers to open

(Director creates the real issues; #937 is this doc's crosslink tracking issue,
reused for the core transform-column / label-ordering divergence REQ-1/REQ-2; the
rest are SUGGESTIONS.)

- **#940** — REQ-3 / REQ-5: fix the cluster label-numbering convention in
  `agglomerative.rs::agglomerate` to mirror sklearn's `_hc_cut` dendrogram-cut order
  + `np.searchsorted(np.unique(labels), labels)` re-labeling
  (`_agglomerative.py:1099/1105`), so `labels_` value-matches (not just the partition).
  **The core root cause — pin FIRST.** Owned by the AgglomerativeClustering unit;
  REQ-1/REQ-2 (transform column order) unblock automatically once it lands.
- **#937** — REQ-1 / REQ-2: once #940 lands, confirm `transform` (mean & max) column
  order value-matches sklearn (`_feature_agglomeration.py:51-63`).
- **#941** — REQ-4: add the `ensure_min_features=2` guard (`_agglomerative.py:1338`)
  and map validation failures to the sklearn `ValueError`/`InvalidParameterError` ABI.
- **#942** — REQ-6: `n_clusters=2` default; add `metric`/`memory`/`connectivity`/
  `compute_full_tree`/`distance_threshold`/`compute_distances` params
  (`_agglomerative.py:1296-1319`), including the `n_clusters=None` + `distance_threshold`
  cut path (`:1091-1092`).
- **#943** — REQ-7: accept an arbitrary `pooling_func` callable (`:1291`), not just
  the `PoolingFunc::{Mean, Max}` enum.
- **#944** — REQ-8: add `inverse_transform` broadcasting pooled values back to
  original feature positions (`_feature_agglomeration.py:66-92`).
- **#945** — REQ-9: rename `feature_labels_` → `labels_`; add `n_leaves_`,
  `n_connected_components_`, `children_`, `distances_` (`_agglomerative.py:1083-1095`).
- **#946** — REQ-10: add `_RsFeatureAgglomeration` to `ferrolearn-python`
  (fit / transform / inverse_transform / labels_ + parameter surface).
- **#947** — REQ-11: migrate `feature_agglomeration.rs` (and the delegated
  `agglomerative.rs`) off `ndarray`/`num-traits` to `ferray-core` (R-SUBSTRATE).
