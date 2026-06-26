# Pairwise Distances & Kernels (sklearn.metrics.pairwise)

<!--
tier: 3-component
status: draft
baseline-commit: e35a9c9c
upstream: scikit-learn 1.5.2 (commit 156ef14)
upstream-paths:
  - sklearn/metrics/pairwise.py   # euclidean_distances (:268); _euclidean_distances (:375); nan_euclidean_distances (:430); haversine_distances (:973); manhattan_distances (:1034); cosine_distances (:1094); paired_euclidean_distances (:1146); paired_manhattan_distances (:1190); paired_cosine_distances (:1224); paired_distances (:1262); linear_kernel (:1351); polynomial_kernel (:1403); sigmoid_kernel (:1466); rbf_kernel (:1524); laplacian_kernel (:1580); cosine_similarity (:1634); additive_chi2_kernel (:1696); chi2_kernel (:1772); pairwise_distances_argmin_min (:692); pairwise_distances_argmin (:841); pairwise_distances_chunked (:2013); pairwise_distances (:2196); pairwise_kernels (:2469)
ferrolearn-module: ferrolearn-metrics/src/pairwise.rs
parity-ops: euclidean_distances, manhattan_distances, cosine_similarity, cosine_distances, chebyshev_distances, haversine_distances, nan_euclidean_distances, paired_euclidean_distances, paired_manhattan_distances, paired_cosine_distances, paired_distances, additive_chi2_kernel, chi2_kernel, laplacian_kernel, distance_metrics, kernel_metrics, pairwise_distances, pairwise_distances_argmin, pairwise_distances_argmin_min, pairwise_kernels
crosslink-issue: 788
-->

## Summary

`ferrolearn-metrics/src/pairwise.rs` mirrors part of scikit-learn's
`sklearn/metrics/pairwise.py`. It implements distance functions
(`euclidean_distances`, `manhattan_distances`, `cosine_distances`,
`nan_euclidean_distances`), a `chebyshev_distances` extra (sklearn exposes
Chebyshev only via the `pairwise_distances(metric="chebyshev")` string ABI, not a
named function), a `pairwise_distances` dispatcher driven by a 4-variant `Metric`
enum, the `pairwise_distances_argmin`/`pairwise_distances_argmin_min` reductions,
and a `pairwise_kernels` dispatcher driven by a `PairwiseKernel<F>` enum.

Under honest underclaim (R-HONEST-3), the present functions are **value-correct
against the live sklearn 1.5.2 oracle on the cases probed** — the core numeric
contracts (Euclidean identity + negative clamp, Manhattan L1, cosine
`1 - cosine_similarity` with the zero-vector→1 convention, the
`sqrt(n_features/n_present * sq_diff)` nan-Euclidean formula, Chebyshev,
argmin/argmin_min tie-break-to-first) all reproduce sklearn. But every present
function **diverges from sklearn on the API/ABI contract** in at least one of:

1. **No `X is Y` self-distance diagonal-zeroing.** sklearn zeros the diagonal of
   `euclidean_distances`/`nan_euclidean_distances`/`cosine_distances` **only when
   the two arguments are the same object** (`pairwise.py:414-415`, `:532-535`,
   `:1134-1137`) — a Python `is` identity check that has **no analog** when the
   ferrolearn API takes two separate `&Array2<F>` arguments. ferrolearn relies
   purely on the float `np.maximum(.,0)` clamp, so it cannot guarantee the exact
   `0.0` diagonal sklearn guarantees on a self-call. (In f64 the clamp already
   yields `0.0` on the cases probed, so this is a latent contract gap, not an
   observed f64 value divergence — but f32 / large-magnitude near-equal rows are
   where sklearn's explicit `fill_diagonal` matters.)
2. **No `squared=` option** on `euclidean_distances` / `nan_euclidean_distances`
   (sklearn `:268`, `:430` — `squared=False` default).
3. **No `missing_values=` parameter** on `nan_euclidean_distances` (sklearn
   `:430` defaults to `np.nan` but accepts any sentinel, `:508`,`:513`).
4. **The `Metric` enum is a 4-variant closed set** (Euclidean/Manhattan/Cosine/
   Chebyshev) — sklearn's `pairwise_distances` accepts a **string metric ABI**
   covering `euclidean`/`l2`/`manhattan`/`l1`/`cityblock`/`cosine`/`chebyshev`
   plus the full scipy `cdist` metric set (`:2196`, `_VALID_METRICS`).
5. **The `PairwiseKernel<F>` enum requires explicit params with no defaults** —
   sklearn `pairwise_kernels` accepts the string set
   `{linear, poly/polynomial, rbf, sigmoid, laplacian, cosine, chi2,
   additive_chi2}` with `gamma=1/n_features` / `coef0`/`degree` defaults
   (`:2469`, `PAIRWISE_KERNEL_FUNCTIONS`).

Beyond the present functions, the routed but NOT implemented pieces are now the
remaining standalone kernel helper names (`linear_kernel`/`polynomial_kernel`/
`sigmoid_kernel`/`rbf_kernel`). ferrolearn exposes their formulas through
`PairwiseKernel<F>`, but not as those Python-named free functions with sklearn's
default-parameter ABI.

All present functions are existing pub APIs re-exported at the crate root
(`lib.rs`: `pub use pairwise::{ … }`); they are grandfathered under
S5/R-DEFER-1 (the crate-root re-export is the non-test production-consumer
surface). There is **no `ferrolearn-python` binding** for any pairwise function.

## Algorithm (sklearn — the contract)

`check_pairwise_arrays(X, Y)` validates equal `n_features` and, when `Y is None`,
sets `Y = X` (preserving object identity for the diagonal special-cases below).

### euclidean_distances (`:268`) / `_euclidean_distances` (`:375`)

`||a-b||² = ||a||² + ||b||² − 2·a·bᵀ` via BLAS (`:407-409`), then:

- **`np.maximum(distances, 0, out=distances)`** (`:410`) — clamp negatives from
  catastrophic cancellation;
- **`if X is Y: np.fill_diagonal(distances, 0)`** (`:414-415`) — exact-zero the
  self-distance diagonal (object-identity check);
- `squared` (`:307`) returns the squared matrix (skip the final `np.sqrt`,
  `:417`). float32 inputs are chunk-upcast to float64 (`:401-404`).

### nan_euclidean_distances (`:430`)

Missing-aware Euclidean. With `missing_values` (default `np.nan`,`:508`): build
masks, zero the missing entries, compute `euclidean_distances(squared=True)`,
subtract the contributions of missing features (`:524-528`), clamp to 0
(`:530`), zero the `X is Y` diagonal (`:532-535`), set **all-missing pairs to
`np.nan`** (`:540`), then **scale by `n_features / present_count`** (`:541-544`)
and `sqrt` unless `squared` (`:546-547`). The scale `n_features/n_present` is the
"average over present features, rescaled to all features" weighting.

### manhattan_distances (`:1034`) / cosine_distances (`:1094`)

- **manhattan**: `sum_k |x_ik − y_jk|` (`:1034`); dense path `manhattan_distances`.
- **cosine_distances**: `S = cosine_similarity(X, Y); S = 1 − S;
  np.clip(S, 0, 2)` (`:1130-1133`); **`if X is Y or Y is None:
  np.fill_diagonal(S, 0)`** (`:1134-1137`). `cosine_similarity` (`:1634`)
  L2-normalizes rows — a **zero-norm row normalizes to 0**, so similarity 0 →
  distance 1.

### Chebyshev — string ABI only

sklearn has **no `chebyshev_distances` function**; Chebyshev (L∞) is reachable
only as `pairwise_distances(X, Y, metric="chebyshev")`, dispatched through
scipy's `cdist` (`:2196`, `_VALID_METRICS`). Value: `max_k |x_ik − y_jk|`.

### pairwise_distances dispatcher (`:2196`)

Resolves a **string** `metric` (or callable) to a distance function: built-in
fast paths for `euclidean`/`l2`, `manhattan`/`l1`/`cityblock`, `cosine`,
`haversine`, `nan_euclidean`, `precomputed`; everything else delegated to
scipy `cdist`/`pdist` (`chebyshev`, `minkowski`, `seuclidean`, `mahalanobis`,
`hamming`, `jaccard`, …). Aliases `l1↔cityblock↔manhattan`, `l2↔euclidean`.

### pairwise_distances_argmin (`:841`) / _argmin_min (`:692`)

For each row of `X`, the index of the nearest row of `Y` (and that distance for
`_min`), computed in chunks. **Ties break to the lowest index** (probed:
`argmin([[0,0]], [[1,0],[1,0],[2,0]]) == 0`).

### pairwise_kernels (`:2469`) and the named kernels

String/callable `metric` dispatch over `PAIRWISE_KERNEL_FUNCTIONS`
(`{additive_chi2, chi2, cosine, laplacian, linear, poly, polynomial, rbf,
sigmoid}`). Defaults: `gamma = 1/n_features` when `None`
(rbf/poly/sigmoid/laplacian/chi2), `coef0 = 1`, `degree = 3`.

- **linear** (`:1351`): `X·Yᵀ`.
- **polynomial** (`:1403`): `(gamma·X·Yᵀ + coef0)^degree`.
- **rbf** (`:1524`): `exp(−gamma·||x−y||²)`.
- **sigmoid** (`:1466`): `tanh(gamma·X·Yᵀ + coef0)`.
- **laplacian** (`:1580`): `exp(−gamma·||x−y||₁)`.
- **chi2** (`:1772`): `exp(−gamma·Σ (x−y)²/(x+y))`.
- **additive_chi2** (`:1696`): `−Σ (x−y)²/(x+y)`.
- **cosine** = `cosine_similarity` (`:1634`).

### Additional sklearn functions in this module

- **haversine_distances** (`:973`): great-circle distance, 2-feature
  `(lat, lon)` in radians.
- **paired_euclidean/manhattan/cosine_distances**, **paired_distances**
  (`:1146`-`:1281`): elementwise row-paired distance — `X` and `Y` same shape,
  output shape `(n_samples,)`, NOT a full `(n, m)` matrix.
- **cosine_similarity** (`:1634`): standalone normalized dot-product
  similarity.
- **pairwise_distances_chunked** (`:2013`): generator yielding distance-matrix
  chunks with an optional `reduce_func` for memory-bounded streaming.

## ferrolearn (what exists)

All public functions live in `ferrolearn-metrics/src/pairwise.rs`, generic over
`F: Float + Send + Sync + 'static`, returning `Result<Array2<F>, FerroError>`
(or `Array1` for the argmin reductions):

- **`pub fn euclidean_distances`** — `||a||²+||b||²−2a·b` identity with
  `.max(F::zero())` negative clamp; **no `squared` arg, no `X is Y` diagonal
  special-case** (two distinct `&Array2` args).
- **`pub fn manhattan_distances`** — `Σ|x−y|`.
- **`pub fn cosine_distances`** — per-pair cosine similarity, **zero-norm row →
  distance `1.0`** (matches sklearn's normalize-to-0 convention), clamps
  `cos_sim` to `[-1, 1]` (so distance ∈ `[0, 2]`, matching sklearn's
  `clip(S, 0, 2)`); **no `X is Y` diagonal special-case**.
- **`pub fn chebyshev_distances`** — `max_k |x−y|` (over private
  `fn chebyshev_distances_inner`). **No sklearn named-function analog** — this is
  the L∞ metric that sklearn reaches only via the `metric="chebyshev"` string.
- **`pub fn nan_euclidean_distances`** — per-pair: accumulate `sq_diff` over
  features where neither value is NaN, `n_valid == 0 → NaN`, else
  `sqrt(sq_sum * n_features / n_valid)`. **Hardcodes NaN as the sentinel (no
  `missing_values` arg), no `squared` arg, no `X is Y` diagonal special-case.**
- **`pub fn cosine_similarity`** — normalized dot-product similarity; zero rows
  normalize to similarity `0.0`, matching sklearn.
- **`pub fn haversine_distances`** — angular great-circle distance for
  `(latitude, longitude)` rows in radians, requiring exactly two features.
- **`pub fn paired_euclidean_distances`**, **`paired_manhattan_distances`**,
  **`paired_cosine_distances`**, **`paired_distances`** — row-paired `(n,)`
  outputs matching sklearn's paired-distance family.
- **`pub fn pairwise_distances(x, y, metric: Metric)`** — dispatcher over the
  **`Metric`** enum `{Euclidean, Manhattan, Cosine, Chebyshev}` — a closed
  4-variant set, **not** sklearn's string ABI / scipy metric set.
- **`pub fn pairwise_distances_argmin(x, y, metric)`** — argmin row-wise,
  strict `<` so **ties break to first** (matches sklearn).
- **`pub fn pairwise_distances_argmin_min(x, y, metric)`** — `(idx, mins)`.
- **`pub fn pairwise_distances_chunked(x, y, metric, working_memory_mib)`** —
  Rust-native `Vec<Array2<F>>` vertical chunks, with `y=None` self-distance
  support and sklearn's at-least-one-row chunk sizing.
- **`pub fn pairwise_distances_chunked_reduce(...)`** — callback form receiving
  `(D_chunk, start)` like sklearn's `reduce_func(D_chunk, start)`.
- **`pub fn pairwise_kernels(x, y, kernel: PairwiseKernel<F>)`** — dispatcher
  over **`PairwiseKernel<F>`** `{Linear, Polynomial{degree,gamma,coef0},
  Rbf{gamma}, Sigmoid{gamma,coef0}, Laplacian{gamma}, Cosine, AdditiveChi2,
  Chi2{gamma}}`. **All params explicit (no `gamma=1/n_features` / `coef0=1` /
  `degree=3` defaults).**
- **`pub fn additive_chi2_kernel`**, **`chi2_kernel`**, **`laplacian_kernel`** —
  standalone dense helpers for the sklearn-named chi2 and laplacian kernels.
- **`pub fn distance_metrics`**, **`kernel_metrics`** — registry-name helpers
  exposing the sklearn metric-key sets as Rust string slices.

**Traits/types:** `pub trait DistanceMetric<F>` (the dyn-friendly `DistanceMetric`
analog, with `pairwise` + a default `distance` for single points); `impl
DistanceMetric for Metric`. **Internal helpers:** `fn check_feature_dim`,
`fn check_non_empty` (the `check_pairwise_arrays` analog — but it has **no
`Y is None ⇒ Y = X` identity path**, which is exactly why the diagonal
special-cases cannot be expressed).

**Consumers (non-test):** crate re-export — `lib.rs`
`pub use pairwise::{ DistanceMetric, Metric, PairwiseKernel,
additive_chi2_kernel, chebyshev_distances, chi2_kernel, cosine_similarity,
cosine_distances, distance_metrics, euclidean_distances, haversine_distances,
kernel_metrics, laplacian_kernel, manhattan_distances, nan_euclidean_distances,
paired_cosine_distances, paired_distances, paired_euclidean_distances,
paired_manhattan_distances, pairwise_distances, pairwise_distances_argmin,
pairwise_distances_argmin_min, pairwise_distances_chunked,
pairwise_distances_chunked_reduce, pairwise_kernels }`. These are existing pub APIs
(grandfathered, S5/R-DEFER-1). **No `ferrolearn-python` binding** exposes any
pairwise function (REQ-13).

## Requirements

- REQ-1: **`euclidean_distances` parity (R-DEV-1).** Match `:268`/`:375`: the
  negative-clamp (`:410`) **and** the `X is Y` diagonal-zeroing (`:414-415`)
  **and** the `squared=` option (`:307`). 1D-pair value + clamp already match;
  the `X is Y` diagonal and `squared` are the gaps.
- REQ-2: **`manhattan_distances` parity (R-DEV-1).** Match `:1034`: `Σ|x−y|`.
  Value matches; sklearn additionally exposes a `sum_over_features` path and the
  string-ABI aliases `l1`/`cityblock` (REQ-6).
- REQ-3: **`cosine_distances` parity (R-DEV-1).** Match `:1094`: `clip(S,0,2)`
  (already equivalent via the `[-1,1]` cos clamp), zero-norm→1 (matches), **and
  the `X is Y` / `Y is None` diagonal-zeroing** (`:1134-1137`) — the gap.
- REQ-4: **Chebyshev parity (R-DEV-2 ABI).** sklearn has **no named
  `chebyshev_distances`** — it is `pairwise_distances(metric="chebyshev")` via
  scipy `cdist`. ferrolearn's free function + `Metric::Chebyshev` are value-correct
  (`max|x−y|`) but ABI-divergent (no string name, scipy-set incompleteness).
- REQ-5: **`nan_euclidean_distances` parity (R-DEV-1/2).** Match `:430`: the
  `sqrt(n_features/n_present · sq)` formula + all-missing→NaN already match; gaps
  are the **`missing_values=` sentinel** (`:508`,`:513`), **`squared=`**, and the
  `X is Y` diagonal-zeroing (`:532-535`).
- REQ-6: **`pairwise_distances` dispatcher + metric ABI (R-DEV-2/3).** Match
  `:2196`: the **string `metric` ABI** (`euclidean`/`l2`, `manhattan`/`l1`/
  `cityblock`, `cosine`, `chebyshev`, `nan_euclidean`, `haversine`,
  `precomputed`, scipy set) with aliases. ferrolearn exposes a closed 4-variant
  `Metric` enum — no string ABI, no scipy metrics, no `precomputed` dispatch.
- REQ-7: **`pairwise_kernels` dispatcher + kernel set (R-DEV-1/2).** Match
  `:2469`: the **string kernel ABI** `{linear, poly/polynomial, rbf, sigmoid,
  laplacian, cosine, chi2, additive_chi2}` with `gamma=1/n_features`/`coef0=1`/
  `degree=3` defaults. ferrolearn's `PairwiseKernel<F>` enum requires explicit
  params. All listed formulas are value-correct given matching params.
- REQ-8: **`pairwise_distances_argmin` parity (R-DEV-1).** Match `:841`:
  nearest-row index, ties→first (matches), but `metric` is the `Metric` enum not
  the string ABI; chunking is internal sklearn detail (R-DEV-7, not observable).
- REQ-9: **`pairwise_distances_argmin_min` parity (R-DEV-1/3).** Match `:692`:
  `(indices, min_distances)` — value-correct; same `metric`-ABI caveat as REQ-8.
- REQ-10: **`cosine_similarity` (R-DEV-1).** sklearn exposes a public
  `cosine_similarity` (`:1634`); ferrolearn exposes a dense public helper that
  matches sklearn's zero-row behavior.
- REQ-11: **Named kernel functions (PARTIAL, R-DEV-1).** sklearn exposes
  `linear_kernel`/`polynomial_kernel`/`sigmoid_kernel`/`rbf_kernel`/
  `laplacian_kernel`/`chi2_kernel`/`additive_chi2_kernel` (`:1351`-`:1772`) as
  standalone functions with their default-`gamma` contract; ferrolearn now
  exposes dense `laplacian_kernel`, `chi2_kernel`, and `additive_chi2_kernel`.
  The linear/polynomial/sigmoid/rbf formulas remain enum-only.
- REQ-12: **`paired_*` family + `haversine_distances` + `pairwise_distances_chunked`
  (SHIPPED, R-DEV-1/3).** sklearn `paired_euclidean/manhattan/cosine_distances`,
  `paired_distances` (`:1146`-`:1281`, output shape `(n_samples,)`),
  `haversine_distances` (`:973`), and `pairwise_distances_chunked` (`:2013`) are
  now public and value-pinned. The chunked API is Rust-native (`Vec`/closure)
  rather than Python's generator object.
- REQ-13: **PyO3 binding (R-DEFER-1).** `import sklearn.metrics.pairwise` exposes
  these; `ferrolearn-python` exposes no pairwise shim.
- REQ-14: **ferray substrate (R-SUBSTRATE).** `pairwise.rs` imports
  `ndarray::Array2`/`Array1` + `num_traits::Float`, not `ferray-core`.

## Acceptance criteria

All expected values are from the live sklearn 1.5.2 oracle (run from `/tmp`),
never literal-copied from ferrolearn (R-CHAR-3).

- AC-1 (REQ-1 value, must stay green): `euclidean_distances([[0,0],[3,0]],[[0,4]])`
  → sklearn `[[4],[5]]`; `euclidean_distances([[1,2,3]],[[1,2,3]])` diagonal
  `0.0`. ferrolearn matches.
- AC-2 (REQ-1 `squared`): `euclidean_distances(X, X, squared=True)` returns the
  **un-rooted** matrix (e.g. `[[0,27],[27,0]]` for `[[1,2,3],[4,5,6]]`).
  ferrolearn has no `squared` arg — inexpressible.
- AC-3 (REQ-1/3/5 `X is Y` diagonal): on a self-call sklearn guarantees an
  **exact-`0.0` diagonal** via `fill_diagonal` regardless of float rounding;
  ferrolearn relies on the `max(.,0)` clamp and has no object-identity path, so
  it cannot guarantee the exact-zero diagonal on f32 / near-equal large-magnitude
  rows. (On the f64 cases probed the clamp already yields `0.0`.)
- AC-4 (REQ-3 value, must stay green):
  `cosine_distances([[1,0]],[[0,0]])` → sklearn `[[1.0]]` (zero vector);
  `cosine_distances([[1,0]],[[-1,0]])` → `[[2.0]]`. ferrolearn matches.
- AC-5 (REQ-5 value, must stay green): `nan_euclidean_distances([[nan,3]],[[0,0]])`
  → sklearn `[[4.2426...]]` (`3·√2`); all-NaN pair → `nan`. ferrolearn matches.
- AC-6 (REQ-5 `missing_values`): `nan_euclidean_distances([[0,3]],[[0,0]],
  missing_values=0)` → sklearn `[[nan]]` (the only present feature, index 0, is
  the missing sentinel for both rows → no present features). ferrolearn hardcodes
  NaN, cannot express `missing_values=0`.
- AC-7 (REQ-6 metric ABI): `pairwise_distances(X,Y,metric="cityblock")` ==
  `metric="l1"` == `metric="manhattan"` (all `[3,4]` for
  `[[0,0],[3,4]]` vs `[[1,2]]`); `metric="l2"` == `metric="euclidean"`. ferrolearn
  has no string alias surface — `Metric` enum only.
- AC-8 (REQ-7 kernels, value + defaults): `rbf_kernel([[1,2],[3,4]],[[1,0]])` →
  sklearn `[0.13533528, 4.5399e-05]` (default `gamma=0.5=1/n_features`);
  `polynomial_kernel(...)` → `[3.375, 15.625]` (`gamma=0.5,coef0=1,degree=3`);
  `chi2_kernel(...)` → `[0.13533528, 0.00673795]`. ferrolearn's `Rbf{gamma}` with
  `gamma=0.5` reproduces the rbf values, but the **default `gamma=1/n_features`**
  is the caller's responsibility (no default).
- AC-9 (REQ-8/9, must stay green): `pairwise_distances_argmin([[0,0]],
  [[1,0],[1,0],[2,0]])` → `[0]` (tie→first); `_argmin_min` → `([0],[1.0])`.
  ferrolearn matches.
- AC-10 (REQ-10/11/12 shipped/partial): `cosine_similarity([[1,2],[3,4]],[[1,0]])` →
  `[0.4472136, 0.6]`; `additive_chi2_kernel(...)` → `[-2,-5]`;
  `paired_euclidean_distances([[0,0],[1,1]],[[1,0],[1,2]])` → `[1.0, 1.0]`;
  `haversine_distances` over `(lat,lon)` radians. These are now callable in
  ferrolearn and pinned in `tests/divergence_pairwise.rs`.

## REQ status table

Binary (R-DEFER-2). Present functions are existing pub APIs re-exported at the
crate root (the non-test production-consumer surface; grandfathered S5/R-DEFER-1).
Cites use symbol anchors (ferrolearn) / `file:line` (sklearn 1.5.2). Live oracle
= installed sklearn 1.5.2, run from `/tmp`. Honest underclaim (R-HONEST-3): no
present function is fully SHIPPED — each is missing at least the `X is Y`
diagonal contract and/or the string-ABI / defaults surface; the MISSING functions
are NOT-STARTED.

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (`euclidean_distances`) | NOT-STARTED | open prereq blocker #790. `pub fn euclidean_distances` value-correct + negative-clamps (`.max(F::zero())` mirrors `pairwise.py:410`); but **no `X is Y` diagonal-zeroing** (`:414-415` — Rust takes two distinct `&Array2`, no identity path) and **no `squared=` arg** (`:307`). Pin: `euclidean_distances(X,X,squared=True)` returns the un-rooted matrix — inexpressible. |
| REQ-2 (`manhattan_distances`) | NOT-STARTED | open prereq blocker #791. `pub fn manhattan_distances` `Σ|x−y|` value matches `:1034`; no `l1`/`cityblock` string alias (folded into REQ-6 dispatcher ABI). |
| REQ-3 (`cosine_distances`) | NOT-STARTED | open prereq blocker #792. `pub fn cosine_distances` clamps cos_sim to `[-1,1]` (⇒ distance `[0,2]`, equivalent to `clip(S,0,2)` `:1133`), zero-norm→1 matches `:1130`; but **no `X is Y`/`Y is None` diagonal-zeroing** (`:1134-1137`). Value AC-4 matches. |
| REQ-4 (Chebyshev / `Metric::Chebyshev`) | NOT-STARTED | open prereq blocker #793. `pub fn chebyshev_distances` (`max|x−y|`) is value-correct vs `pairwise_distances(metric="chebyshev")`, but sklearn has **no named `chebyshev_distances`** — the ABI is the `"chebyshev"` string through scipy `cdist` (`:2196`). Folded into REQ-6: the named function + the 4-variant `Metric` enum are an ABI divergence. |
| REQ-5 (`nan_euclidean_distances`) | NOT-STARTED | open prereq blocker #794. `pub fn nan_euclidean_distances` formula `sqrt(n_feat/n_present·sq)` + all-missing→NaN match `:540-547` (AC-5); but **hardcodes NaN (no `missing_values=`, `:508`,`:513`), no `squared=`, no `X is Y` diagonal** (`:532-535`). Pin AC-6: `missing_values=0` → sklearn `[[nan]]`, inexpressible. |
| REQ-6 (`pairwise_distances` + metric ABI) | NOT-STARTED | open prereq blocker #795. `pub fn pairwise_distances(x,y,metric:Metric)` dispatches a **closed 4-variant `Metric` enum** `{Euclidean,Manhattan,Cosine,Chebyshev}`; sklearn `:2196` takes a **string `metric`** with aliases (`l1`/`cityblock`/`manhattan`, `l2`/`euclidean`) + the scipy `cdist` set + `precomputed`/`haversine`/`nan_euclidean`. Pin AC-7: `metric="cityblock"`==`"l1"`==`"manhattan"` — no alias surface. |
| REQ-7 (`pairwise_kernels` + kernel set) | PARTIAL | open prereq blocker #796. `pub fn pairwise_kernels(x,y,kernel:PairwiseKernel<F>)`: linear/poly/rbf/sigmoid/laplacian/cosine/chi2/additive_chi2 formulas are value-correct **given matching params**, but the enum **requires explicit `gamma`/`coef0`/`degree` (no `gamma=1/n_features`/`coef0=1`/`degree=3` defaults, `:2469`)**. Pins cover RBF and standalone chi2 helpers. |
| REQ-8 (`pairwise_distances_argmin`) | NOT-STARTED | open prereq blocker #797. `pub fn pairwise_distances_argmin` nearest-row index, strict `<` ⇒ ties→first matches `:841` (AC-9); but `metric:Metric` not the string ABI (depends on REQ-6). |
| REQ-9 (`pairwise_distances_argmin_min`) | NOT-STARTED | open prereq blocker #798. `pub fn pairwise_distances_argmin_min` `(idx,mins)` value-correct vs `:692` (AC-9); same `metric`-ABI dependence on REQ-6. |
| REQ-10 (`cosine_similarity`) | SHIPPED | sklearn `:1634` exposes public `cosine_similarity`; `pub fn cosine_similarity` is value-pinned in `tests/divergence_pairwise.rs`, including zero-row behavior. |
| REQ-11 (named kernel functions) | PARTIAL | open prereq blocker #800. sklearn `linear_kernel`/`polynomial_kernel`/`sigmoid_kernel`/`rbf_kernel`/`laplacian_kernel`/`chi2_kernel`/`additive_chi2_kernel` (`:1351`-`:1772`) — standalone functions with default-`gamma`; ferrolearn ships dense `laplacian_kernel`, `chi2_kernel`, and `additive_chi2_kernel`, while linear/poly/sigmoid/rbf remain enum-only. |
| REQ-12 (`paired_*`/`haversine`/`chunked`) | SHIPPED | `paired_euclidean/manhattan/cosine_distances`+`paired_distances` (`:1146`-`:1281`, output `(n_samples,)`) and `haversine_distances` (`:973`) are value-pinned; `pairwise_distances_chunked` (`:2013`) now emits contiguous vertical chunks with sklearn's at-least-one-row `working_memory=0` behavior and a reducer callback that receives `(D_chunk, start)`. Verification: `tests/divergence_pairwise.rs` pins row chunks, default self-distance full chunks, and reducer start rows. |
| REQ-13 (PyO3 binding) | NOT-STARTED | open prereq blocker #802. `ferrolearn-python` exposes no pairwise shim; `import ferrolearn` cannot call what `import sklearn.metrics.pairwise` provides. |
| REQ-14 (ferray substrate) | NOT-STARTED | open prereq blocker #803. `pairwise.rs` imports `ndarray::Array2`/`Array1` + `num_traits::Float`, not `ferray-core` (R-SUBSTRATE). |

## Architecture

`pairwise.rs` is a flat module of free functions plus two dispatch enums and one
trait, each generic over `F: Float + Send + Sync + 'static` and returning
`Result<Array2<F>, FerroError>` (argmin reductions return `Array1`). There are no
fitted/unfitted types — these are stateless metrics. Three families:

1. **Distance functions** (`euclidean_distances`, `manhattan_distances`,
   `cosine_distances`, `chebyshev_distances`, `nan_euclidean_distances`):
   double-loop folds over `(i, j)` row pairs. `euclidean_distances` uses the
   `||a||²+||b||²−2a·b` identity with a per-entry `.max(F::zero())` clamp
   (mirrors `np.maximum(.,0)`, `pairwise.py:410`). The **structural divergence
   across all of these is the missing `X is Y` diagonal special-case** — sklearn's
   `check_pairwise_arrays` preserves object identity (`Y is None ⇒ Y = X`) so the
   self-call diagonal can be force-zeroed; ferrolearn's `fn check_non_empty`/
   `fn check_feature_dim` take two independent `&Array2` and have no identity
   notion, so the diagonal is whatever the float clamp produces (REQ-1/3/5). The
   secondary divergences are the absent `squared=`/`missing_values=` options
   (REQ-1/5) and `cosine`'s missing diagonal (REQ-3).
2. **Dispatchers** (`pairwise_distances` over `Metric`; `pairwise_kernels` over
   `PairwiseKernel<F>`; the `DistanceMetric` trait + its `impl for Metric`): the
   ABI divergence lives here. sklearn dispatches **strings** (and callables) over
   open metric/kernel registries with documented aliases and `gamma`/`coef0`/
   `degree` defaults; ferrolearn dispatches **closed Rust enums** with no aliases,
   no defaults, and a strict subset of metrics/kernels (REQ-6/7). The
   `DistanceMetric` trait is the dyn-friendly `sklearn.DistanceMetric` analog and
   has a default `distance` method for single-point pairs.
3. **Reductions** (`pairwise_distances_argmin`, `pairwise_distances_argmin_min`):
   materialize the full distance matrix via `pairwise_distances`, then scan each
   row for the min with strict `<` (⇒ ties→first, matching sklearn). Value-correct;
   the only divergence is inheriting the `Metric`-enum ABI (REQ-8/9) and the
   non-chunked implementation (R-DEV-7 — internal, unobservable).
4. **Chunking** (`pairwise_distances_chunked`, `pairwise_distances_chunked_reduce`):
   compute `pairwise_distances` on contiguous row slices, using sklearn's
   `row_bytes = 8 * n_samples_Y` memory calculation and the at-least-one-row
   floor for too-small memory budgets. The reducer callback receives the
   sklearn-style `start` row.

**Invariants held vs sklearn (probed):** Euclidean values + negative clamp;
Manhattan L1; cosine `1−sim` with zero-norm→1 and `[0,2]` range; nan-Euclidean
`sqrt(n_feat/n_present·sq)` + all-missing→NaN; Chebyshev `max|x−y|`; chunk
boundaries for `working_memory=0`; reducer `start` rows; argmin/min
tie-break-to-first; the rbf/poly/sigmoid/laplacian/linear kernel formulas at
matching params. **Invariants NOT held vs sklearn:** the `X is Y` self-distance
diagonal-zeroing (euclidean/cosine/nan-euclidean); `squared=`/`missing_values=`
options; the string metric ABI + aliases + scipy metric set; the string kernel
ABI + `gamma`/`coef0`/`degree` defaults.

**MISSING functions (routed, not implemented):** `linear_kernel`,
`polynomial_kernel`, `sigmoid_kernel`, `rbf_kernel` (REQ-11).

## Verification

Library crate (green at baseline `e35a9c9c` for the existing — narrower —
contract):
```
cargo test -p ferrolearn-metrics --lib pairwise     # 28 passed, 0 failed
cargo clippy -p ferrolearn-metrics --all-targets -- -D warnings
cargo fmt --all --check
```
The existing 28 `#[test]`s pin only ferrolearn's narrower behavior (value
correctness on small explicit inputs); they do NOT establish full sklearn parity
(no `X is Y` diagonal, no `squared`/`missing_values`, no string ABI, no defaults,
no MISSING functions), so they make no REQ SHIPPED.

Live sklearn oracle (installed 1.5.2, run from `/tmp`) — the deterministic
divergences a critic should pin first (R-CHAR-3 expected values):
```
# REQ-1 squared= (un-rooted matrix): inexpressible in ferro
python3 -c "import numpy as np; from sklearn.metrics.pairwise import euclidean_distances as e; print(e(np.array([[1.,2,3],[4,5,6]]),np.array([[1.,2,3],[4,5,6]]),squared=True))"   # [[0,27],[27,0]]
# REQ-5 missing_values= sentinel: ferro hardcodes NaN
python3 -c "from sklearn.metrics.pairwise import nan_euclidean_distances as n; print(n([[0.,3]],[[0.,0]],missing_values=0))"   # [[nan]]
# REQ-6 metric string aliases: ferro Metric enum has no string surface
python3 -c "import numpy as np; from sklearn.metrics import pairwise_distances as p; X=np.array([[0.,0],[3,4]]); Y=np.array([[1.,2]]); print(p(X,Y,metric='cityblock').ravel(), p(X,Y,metric='l1').ravel(), p(X,Y,metric='chebyshev').ravel())"  # [3 4] [3 4] [2 2]
# REQ-7 kernel defaults: ferro PairwiseKernel has no default gamma/coef0/degree
python3 -c "import numpy as np; from sklearn.metrics.pairwise import rbf_kernel as r, chi2_kernel as c; X=np.array([[1.,2],[3,4]]); Y=np.array([[1.,0]]); print(r(X,Y).ravel(), c(X,Y).ravel())"  # [0.1353 4.54e-5] [0.1353 0.00674]
# REQ-10 cosine_similarity baseline
python3 -c "import numpy as np; from sklearn.metrics.pairwise import cosine_similarity as cs; print(cs(np.array([[1.,2],[3,4]]),np.array([[1.,0]])).ravel())"  # [0.4472 0.6]
# REQ-11 additive_chi2 baseline
python3 -c "import numpy as np; from sklearn.metrics.pairwise import additive_chi2_kernel as a; print(a(np.array([[1.,2],[3,4]]),np.array([[1.,0]])).ravel())"  # [-2 -5]
# REQ-12 paired_euclidean baseline (output shape (n,))
python3 -c "from sklearn.metrics.pairwise import paired_euclidean_distances as pe; print(pe([[0.,0],[1,1]],[[1.,0],[1,2]]))"  # [1. 1.]
# Value baselines that must stay green (already match):
python3 -c "from sklearn.metrics.pairwise import euclidean_distances as e, cosine_distances as c, nan_euclidean_distances as n; print(e([[0.,0],[3,0]],[[0.,4]]).ravel(), c([[1.,0]],[[0.,0]]).ravel(), n([[float('nan'),3]],[[0.,0]]).ravel())"  # [4 5] [1.] [4.2426]
python3 -c "from sklearn.metrics import pairwise_distances_argmin_min as m; import numpy as np; print(m(np.array([[0.,0]]),np.array([[1.,0],[1,0],[2,0]])))"  # (array([0]), array([1.]))
```
A characterization pin (R-CHAR-3) for each NOT-STARTED REQ belongs in
`ferrolearn-metrics/tests/divergence_pairwise.rs`, asserting the live-sklearn
expected values above and FAILING against current `pairwise.rs`. Every REQ is
NOT-STARTED; each carries an open prereq blocker.

## Blockers to open

- #790 — REQ-1 (`euclidean_distances`): no `X is Y` diagonal-zeroing
  (`pairwise.py:414-415`); no `squared=` (`:307`). Pin: `squared=True` un-rooted
  matrix inexpressible.
- #791 — REQ-2 (`manhattan_distances`): no `l1`/`cityblock` string alias
  (`:2196` aliases) — value matches; alias surface folded into #795.
- #792 — REQ-3 (`cosine_distances`): no `X is Y`/`Y is None` diagonal-zeroing
  (`:1134-1137`). Value (zero-norm→1, `[0,2]`) matches.
- #793 — REQ-4 (Chebyshev): sklearn has no named `chebyshev_distances`; the ABI
  is `pairwise_distances(metric="chebyshev")` via scipy `cdist` (`:2196`).
  Value `max|x−y|` matches.
- #794 — REQ-5 (`nan_euclidean_distances`): no `missing_values=` sentinel
  (`:508`,`:513`), no `squared=`, no `X is Y` diagonal (`:532-535`). Pin:
  `missing_values=0` → sklearn `[[nan]]`. Formula matches.
- #795 — REQ-6 (`pairwise_distances` dispatcher): closed 4-variant `Metric` enum
  vs sklearn string ABI + aliases + scipy `cdist` set + `precomputed`/`haversine`/
  `nan_euclidean` (`:2196`). Pin: `metric="cityblock"`==`"l1"`==`"manhattan"`.
- #796 — REQ-7 (`pairwise_kernels`): `PairwiseKernel<F>` requires explicit
  `gamma`/`coef0`/`degree` (no `gamma=1/n_features`/`coef0=1`/`degree=3` defaults,
  `:2469`). Pin: `chi2_kernel` value.
- #797 — REQ-8 (`pairwise_distances_argmin`): inherits the `Metric`-enum ABI
  (#795); tie→first matches.
- #798 — REQ-9 (`pairwise_distances_argmin_min`): inherits the `Metric`-enum ABI
  (#795); values match.
- #799 — REQ-10 (`cosine_similarity`): public function shipped and value-pinned.
- #800 — REQ-11 (named kernels PARTIAL): standalone `laplacian_kernel`,
  `chi2_kernel`, and `additive_chi2_kernel` are shipped; `linear_kernel`/
  `polynomial_kernel`/`sigmoid_kernel`/`rbf_kernel` remain enum-only.
- #801 — REQ-12 (`paired_*`/`haversine`/`chunked`): shipped, including
  `pairwise_distances_chunked` (`:2013`) row chunks and reducer start offsets.
- #802 — REQ-13: no `ferrolearn-python` pairwise binding.
- #803 — REQ-14: migrate `pairwise.rs` off `ndarray`/`num-traits` to the ferray
  substrate (R-SUBSTRATE).
