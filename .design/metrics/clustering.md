# Clustering Metrics (sklearn.metrics.cluster supervised + unsupervised)

<!--
tier: 3-component
status: draft
baseline-commit: ec2d4b7b
upstream: scikit-learn 1.5.2 (commit 156ef14)
upstream-paths:
  - sklearn/metrics/cluster/_supervised.py     # check_clusterings(:32); _generalized_average(:78); contingency_matrix(:104); pair_confusion_matrix(:190); rand_score(:275); adjusted_rand_score(:353); homogeneity_completeness_v_measure(:463); homogeneity_score(:563); completeness_score(:639); v_measure_score(:716); mutual_info_score(:820); adjusted_mutual_info_score(:936); normalized_mutual_info_score(:1069); fowlkes_mallows_score(:1184); entropy(:1268)
  - sklearn/metrics/cluster/_unsupervised.py    # silhouette_score(:54); _silhouette_reduce(:144); silhouette_samples(:206); calinski_harabasz_score(:328); davies_bouldin_score(:399)
  - sklearn/metrics/cluster/_expected_mutual_info_fast.pyx   # expected_mutual_information (Cython; private surface, EMI for AMI)
ferrolearn-module: ferrolearn-metrics/src/clustering.rs
parity-ops: silhouette_score, silhouette_samples, adjusted_rand_score, rand_score, adjusted_mutual_info, normalized_mutual_info_score, v_measure_score, homogeneity_score, completeness_score, homogeneity_completeness_v_measure, davies_bouldin_score, calinski_harabasz_score, fowlkes_mallows_score, mutual_info_score, contingency_matrix, pair_confusion_matrix
crosslink-issue: 796
-->

## Summary

`ferrolearn-metrics/src/clustering.rs` mirrors the supervised and unsupervised
clustering-evaluation functions of scikit-learn's `sklearn/metrics/cluster/`
package — `_supervised.py` (label-vs-label agreement metrics) and
`_unsupervised.py` (feature-space cluster-quality metrics). It implements
**sixteen** public sklearn functions plus the supporting enums `AmiMethod` /
`NmiMethod`:

- **Unsupervised (feature-space):** `silhouette_score`, `silhouette_samples`,
  `davies_bouldin_score`, `calinski_harabasz_score`.
- **Supervised (label agreement):** `adjusted_rand_score`, `rand_score`,
  `adjusted_mutual_info` (+ `adjusted_mutual_info_with_method`),
  `normalized_mutual_info_score`, `v_measure_score`, `homogeneity_score`,
  `completeness_score`, `homogeneity_completeness_v_measure`,
  `fowlkes_mallows_score`, `mutual_info_score`, `contingency_matrix`,
  `pair_confusion_matrix`.

Under honest underclaim (R-HONEST-3), **most present functions value-match the
live sklearn 1.5.2 oracle on the supported numeric path** — the EMI-bearing AMI
matches to ~1e-13, NMI/v_measure honor the `arithmetic` default, ARI/FMI/MI and
the contingency/pair matrices reproduce sklearn exactly. The mirror is far closer
to parity than the regression-metrics sibling. However several functions carry
**deterministic divergences** and every function carries **API-contract gaps**
(no `metric`/`sample_size`/`random_state`/`eps`/`sparse` keyword args; no PyO3
binding; on the `ndarray` substrate rather than ferray). The three cleanest
**deterministic** divergences a critic should pin first:

1. **`rand_score` panics (debug builds) on a valid degenerate input.**
   `rand_score([0,0,1,1],[0,0,0,0])` computes `d = comb_n - sum_comb_a -
   sum_comb_b + sum_comb_c` in **`u64`**, and the subtraction underflows before
   the `+ sum_comb_c` term, so ferrolearn **panics with attempt-to-subtract-with-
   overflow** in debug. sklearn returns **0.3333333333333333**. (In release the
   wrapping happens to cancel and the value is right — a latent overflow bug, not
   a stable contract.)
2. **`calinski_harabasz_score` returns `+inf` instead of `1.0` when within-cluster
   dispersion is zero.** For coincident-per-cluster points
   (`X=[[0],[0],[10],[10]]`, `labels=[0,0,1,1]`) sklearn returns **1.0**
   (`_unsupervised.py:385-389`: `1.0 if intra_disp == 0.0 else ...`); ferrolearn
   returns **`Ok(inf)`** (`clustering.rs` `if w_ss == F::zero() { return
   Ok(F::infinity()) }`).
3. **`homogeneity_score` / `completeness_score` / `v_measure_score` /
   `homogeneity_completeness_v_measure` error on empty input;** sklearn returns
   **`(1.0, 1.0, 1.0)`** / `1.0` (`_supervised.py:531-532`). ferrolearn returns
   `Err(InsufficientSamples)`.

All sixteen functions plus `NmiMethod` are existing pub APIs re-exported at the
crate root (`lib.rs`: `pub use clustering::{...}`); that re-export is the
non-test production-consumer surface, grandfathered under S5/R-DEFER-1. (Note:
`adjusted_mutual_info_with_method` and `AmiMethod` are **not** re-exported at the
crate root — they are reachable only via the `clustering` module path.) The
in-crate `scorer.rs` does **not** yet wire clustering scorers (its own REQ-6 is
NOT-STARTED, `#784`), and `ferrolearn-python` exposes no clustering-metric shim.

## Algorithm (sklearn — the contract)

All supervised metrics route through `check_clusterings` (`_supervised.py:32`,
1-D integral labels, consistent length) and `contingency_matrix` (`:104`). All
unsupervised metrics route through `check_X_y` + `LabelEncoder` +
`check_number_of_labels` (`_unsupervised.py:26`, requires `2 <= n_labels <
n_samples`).

### Contingency / pair-confusion (the shared substrate)

- **`contingency_matrix(labels_true, labels_pred, *, eps=None, sparse=False,
  dtype=np.int64)`** (`:104`): `C[i,j]` = count in true class `i` ∩ predicted
  class `j`, over `np.unique`-sorted class/cluster orderings. `eps` adds a float
  offset (NaN-suppression); `sparse=True` returns CSR.
- **`pair_confusion_matrix(labels_true, labels_pred)`** (`:190`): `2x2`
  `[[C00,C01],[C10,C11]]` over all `(n choose 2)` pairs, from the contingency
  sums: `C11 = sum(n_ij^2) - n`, `C01 = sum_j b_j^2 - sum(n_ij^2)`,
  `C10 = sum_i a_i^2 - sum(n_ij^2)`, `C00 = n^2 - C01 - C10 - sum(n_ij^2)`
  (`:259-264`). Returns `np.int64`.

### Pair-counting metrics

- **`rand_score(labels_true, labels_pred)`** (`:275`):
  `RI = diagonal(pair_confusion).sum() / pair_confusion.sum()`; **special case
  `numerator == denominator or denominator == 0 → return 1.0`** (`:337-341`).
- **`adjusted_rand_score`** (`:353`): from `(tn,fp),(fn,tp)` of the pair-confusion
  matrix, `int`-cast to avoid overflow (`:446`); **`fn == 0 and fp == 0 → 1.0`**
  (`:449`); else `2(tp·tn − fn·fp) / ((tp+fn)(fn+tn) + (tp+fp)(fp+tn))`.
- **`fowlkes_mallows_score(..., *, sparse=False)`** (`:1184`):
  `tk = dot(c.data,c.data) − n`, `pk = sum(b_j^2) − n`, `qk = sum(a_i^2) − n`;
  `sqrt(tk/pk)·sqrt(tk/qk)` if `tk != 0` else `0.0` (`:1256-1259`). **`-n`
  correction is the pair count minus singletons.**

### Information-theoretic metrics

- **`mutual_info_score(labels_true, labels_pred, *, contingency=None)`** (`:820`):
  `MI = sum (n_ij/n)·log(n·n_ij/(a_i·b_j))`; **single-cluster either side → 0.0**
  (`:910-911`); a per-term `|mi| < eps → 0` zeroing and a final
  `np.clip(mi.sum(), 0.0, None)` (`:924-925`). Accepts a precomputed
  `contingency` (then ignores the label args).
- **`entropy(labels)`** (`:1268`): `−sum (p_i)·(log(c_i) − log(n))`; **empty →
  1.0** (`:1285-1286`), single cluster → 0.0.
- **`adjusted_mutual_info_score(..., *, average_method='arithmetic')`** (`:936`):
  `AMI = (MI − EMI) / (avg(H(U),H(V)) − EMI)` where `EMI =
  expected_mutual_information(contingency, n)` (the exact Vinh-2010 hypergeometric
  sum, Cython `_expected_mutual_info_fast.pyx`); `avg` selected by
  `_generalized_average` (`:78`: `min`/`geometric`/`arithmetic`/`max`); **default
  `arithmetic`** (changed from `max` in 0.22). Both single-cluster → 1.0
  (`:1034-1038`); denominator sign-preserving `eps` clamp (`:1053-1056`).
- **`normalized_mutual_info_score(..., *, average_method='arithmetic')`** (`:1069`):
  `NMI = MI / avg(H(U),H(V))`; **default `arithmetic`** (changed from
  `geometric` in 0.22); both single-cluster → 1.0; `mi == 0 → 0.0`.
- **`homogeneity_completeness_v_measure(..., *, beta=1.0)`** (`:463`):
  `H_C=entropy(true)`, `H_K=entropy(pred)`, `MI=mutual_info_score`;
  `homogeneity = MI/H_C` (or 1.0 if `H_C==0`), `completeness = MI/H_K`;
  `v = (1+beta)·h·c / (beta·h + c)` (0.0 if `h+c==0`). **`beta=1.0`** default.
  **Empty input → `(1.0,1.0,1.0)`** (`:531-532`). `homogeneity_score`/
  `completeness_score`/`v_measure_score` are thin `[0]`/`[1]`/`[2]` projections.
  `v_measure_score` is identical to `normalized_mutual_info_score(average_method=
  'arithmetic')`.

### Feature-space metrics

- **`silhouette_score(X, labels, *, metric='euclidean', sample_size=None,
  random_state=None, **kwds)`** (`:54`): `np.mean(silhouette_samples(...))`. With
  `sample_size`, draws `random_state.permutation(n)[:sample_size]` (**RNG-boundary
  sampling** — numpy `RandomState` permutation is not reproducible in Rust).
- **`silhouette_samples(X, labels, *, metric='euclidean', **kwds)`** (`:206`):
  per-sample `s = (b − a) / max(a, b)` with `a` = mean intra-cluster distance
  (denominator `label_freq − 1`), `b` = min over other clusters of mean distance;
  **`nan_to_num` maps size-1 clusters (a undefined) to 0** (`:317-318`). Generic
  over `metric` via `pairwise_distances_chunked`.
- **`calinski_harabasz_score(X, labels)`** (`:328`):
  `extra_disp·(n − k) / (intra_disp·(k − 1))`; **`1.0 if intra_disp == 0.0`**
  (`:385-389`).
- **`davies_bouldin_score(X, labels)`** (`:399`): per-cluster mean centroid
  distance `s_i`, pairwise centroid distances; `DB = mean_i max_{j≠i}
  (s_i+s_j)/d(c_i,c_j)`.

## ferrolearn (what exists)

All public functions live in `ferrolearn-metrics/src/clustering.rs`. The
feature-space metrics are generic over `F: Float + Send + Sync + 'static` taking
`&Array2<F>` + `&Array1<isize>`; the label-agreement metrics take
`&Array1<isize>` × 2 and return `Result<f64, FerroError>` (or `Result<Array2<u64>,
_>` for the matrices). Labels are `isize`; **`-1` is treated as a noise sentinel
excluded from the feature-space metrics** (`fn unique_cluster_labels`) — sklearn
has no such convention (it `LabelEncode`s all labels including `-1`).

- **`pub fn silhouette_score`** / **`pub fn silhouette_samples`** — exact
  `(b−a)/max(a,b)`, size-1 cluster → `a=0`; noise (`-1`) excluded from
  `silhouette_score` mean and assigned `0.0` in `silhouette_samples`. **Euclidean
  only** (private `fn row_euclidean_dist`); no `metric`, no `sample_size`,
  no `random_state`.
- **`pub fn davies_bouldin_score`** — centroid-based DB; coincident centroids →
  `+inf` (sklearn divides too, also non-finite, but never special-cased).
- **`pub fn calinski_harabasz_score`** — VRC; **`w_ss == 0 → Ok(inf)`** (sklearn
  returns **1.0**). Errors when `n == k`.
- **`pub fn adjusted_rand_score`** — combinatorial ARI from an `isize` contingency
  table; `fn n_choose_2` on `u64`; degenerate denominator → 1.0/0.0. Value matches
  sklearn (incl. `-0.5` floor on `[0,0,1,1]/[0,1,0,1]`).
- **`pub fn rand_score`** — `(sum_comb_c + d)/comb_n` with
  `d = comb_n − sum_comb_a − sum_comb_b + sum_comb_c` in **`u64`**. **Underflows
  (panics in debug) when `d` would be negative** (e.g. `[0,0,1,1]/[0,0,0,0]`);
  sklearn 0.3333. `n < 2 → InsufficientSamples`.
- **`pub fn adjusted_mutual_info`** (= `adjusted_mutual_info_with_method(...,
  Arithmetic)`) and **`pub fn adjusted_mutual_info_with_method(..., AmiMethod)`**
  with `AmiMethod::{Arithmetic,Geometric,Min,Max}`. **Default is `Arithmetic`,
  matching sklearn.** The exact EMI is computed in-crate (`fn
  expected_mutual_info` + `fn precompute_log_factorials`) and **matches sklearn's
  Cython `expected_mutual_information` to ~1e-13** (the in-code test
  `test_ami_mixed_labels_matches_sklearn_arithmetic` pins
  `[0,0,1,1,2,2]/[0,0,0,1,1,1] == 0.2987924581708901`, R-CHAR-3 from the live
  oracle). Denominator clamp is `< f64::EPSILON → Ok(1.0)` (vs sklearn's
  sign-preserving clamp — equivalent on the tested cases).
- **`pub fn normalized_mutual_info_score(..., NmiMethod)`** with
  `NmiMethod::{Geometric,Arithmetic,Min,Max}`. **`method` is a REQUIRED positional
  argument with no default** (sklearn defaults to `arithmetic`); the value matches
  sklearn when `Arithmetic` is passed (`0.5158037429793889` on the mixed case).
- **`pub fn homogeneity_score`** / **`pub fn completeness_score`** —
  `1 − H(C|K)/H(C)` / `1 − H(K|C)/H(K)`; single-class/single-cluster → 1.0.
  **Empty input → `Err(InsufficientSamples)`** (sklearn 1.0).
- **`pub fn v_measure_score`** — fixed `beta=1.0` harmonic mean `2hc/(h+c)`;
  **no `beta` parameter** (sklearn `v_measure_score(..., *, beta=1.0)`). `h+c==0
  → 0.0`. Value matches sklearn on `beta=1` cases.
- **`pub fn homogeneity_completeness_v_measure(..., beta: f64)`** — **exposes
  `beta`** (required positional; sklearn keyword-only default 1.0); validates
  `beta > 0 && finite`; `v = (1+beta)hc/(beta·h + c)`. **Empty → error** (sklearn
  `(1,1,1)`).
- **`pub fn fowlkes_mallows_score`** — `tp/sqrt((tp+fp)(tp+fn))`; `tp+fp==0 ||
  tp+fn==0 → 0.0`. Value matches sklearn (`0.4714...` on the mixed case,
  `0.0` on all-singletons).
- **`pub fn mutual_info_score`** — `sum (n_ij/n)·log(n·n_ij/(a_i·b_j))`; **no
  `contingency=` passthrough**, no per-term eps-zeroing / `clip(.,0,None)`
  (immaterial on tested cases). Value matches (`0.6931...` perfect, `0.0566...`
  mixed).
- **`pub fn contingency_matrix`** — dense `Array2<u64>`; **no `eps`/`sparse`/
  `dtype`**. Value/ordering matches sklearn dense output.
- **`pub fn pair_confusion_matrix`** — `2x2 Array2<u64>` via the same
  contingency-sum identities; `saturating_sub` guards. Value matches sklearn
  (`[[8,2],[0,2]]`, `[[8,0],[0,4]]`).

**Internal helpers:** `fn check_labels_same_length`, `fn check_x_labels_compat`,
`fn euclidean_dist`, `fn row_euclidean_dist`, `fn unique_cluster_labels` (noise
filter), `fn n_choose_2`, `fn build_contingency_table`, `fn entropy_from_counts`,
`fn expected_mutual_info`, `fn precompute_log_factorials`. The `trait LetIf` +
`partition_point` pattern maps a label to its sorted-class index.

**Consumers (non-test):** crate re-export
(`lib.rs`: `pub use clustering::{ NmiMethod, adjusted_mutual_info,
adjusted_rand_score, calinski_harabasz_score, completeness_score,
contingency_matrix, davies_bouldin_score, fowlkes_mallows_score,
homogeneity_completeness_v_measure, homogeneity_score, mutual_info_score,
normalized_mutual_info_score, pair_confusion_matrix, rand_score,
silhouette_samples, silhouette_score, v_measure_score }`). This is the
grandfathered non-test production surface (S5/R-DEFER-1). **`AmiMethod` and
`adjusted_mutual_info_with_method` are NOT re-exported at the crate root** (only
reachable via `ferrolearn_metrics::clustering::`). The `scorer.rs` registry does
not yet wire any clustering scorer (its REQ-6 NOT-STARTED, `#784`), and
`ferrolearn-python` exposes no clustering-metric binding.

## Requirements

- REQ-1: **`silhouette_samples` / `silhouette_score` core value (R-DEV-1).** Match
  the per-sample `(b−a)/max(a,b)` with the `label_freq−1` intra denominator and
  size-1→0 `nan_to_num` (`_unsupervised.py:206,310-318`) on `metric='euclidean'`.
- REQ-2: **`silhouette` `metric`/`sample_size`/`random_state` (R-DEV-2, RNG
  boundary).** sklearn `silhouette_score(..., metric='euclidean',
  sample_size=None, random_state=None)` (`:54`) supports arbitrary `metric` and
  `random_state.permutation` subsampling; ferrolearn is euclidean-only,
  unsampled. The sampling path is an **RNG-boundary divergence** (numpy
  `RandomState` permutation order is not reproducible in Rust).
- REQ-3: **`davies_bouldin_score` (R-DEV-1).** Centroid DB index (`:399`).
- REQ-4: **`calinski_harabasz_score` (R-DEV-1).** VRC; **`intra_disp == 0 → 1.0`**
  (`:385-389`) — ferrolearn returns `+inf`.
- REQ-5: **`adjusted_rand_score` (R-DEV-1).** Pair-confusion ARI with the
  `fn==0 && fp==0 → 1.0` short-circuit (`:444-452`).
- REQ-6: **`rand_score` (R-DEV-1).** `diag/total` with
  `numerator==denominator || denominator==0 → 1.0` (`:333-343`); **must not
  panic / underflow** on degenerate labelings.
- REQ-7: **`adjusted_mutual_info` default + EMI (R-DEV-1/2).** `arithmetic`
  default; exact EMI (`:936,1044-1047`); `AmiMethod` covers
  min/geometric/arithmetic/max.
- REQ-8: **`normalized_mutual_info_score` default (R-DEV-2).** sklearn defaults
  `average_method='arithmetic'` (`:1069-1070`); ferrolearn requires an explicit
  `NmiMethod` (no default).
- REQ-9: **`v_measure_score` `beta` (R-DEV-2).** sklearn
  `v_measure_score(..., *, beta=1.0)` (`:716`) exposes `beta`; ferrolearn's
  `v_measure_score` hardcodes `beta=1` (the `beta` form lives only on
  `homogeneity_completeness_v_measure`).
- REQ-10: **`homogeneity_score` (R-DEV-1).** `1 − H(C|K)/H(C)` (`:563,629`); empty
  input convention (`(1,1,1)`, `:531-532`).
- REQ-11: **`completeness_score` (R-DEV-1).** `1 − H(K|C)/H(K)` (`:639,705`); empty
  convention.
- REQ-12: **`homogeneity_completeness_v_measure` (R-DEV-1/2).** `beta`
  keyword-only default 1.0; empty → `(1,1,1)` (`:463,531-532`).
- REQ-13: **`fowlkes_mallows_score` (R-DEV-1).** `sqrt(tk/pk)·sqrt(tk/qk)` with the
  `−n` corrections (`:1184,1256-1259`).
- REQ-14: **`mutual_info_score` (R-DEV-1/2).** MI nats; **`contingency=` keyword
  passthrough** (`:820,886-894`); per-term eps-zero + `clip(.,0,None)`.
- REQ-15: **`contingency_matrix` (R-DEV-2).** `eps`/`sparse`/`dtype` keyword args
  (`:104-106`); ferrolearn is dense-`u64`-only.
- REQ-16: **`pair_confusion_matrix` (R-DEV-1).** `2x2` int64 identities (`:190`).
- REQ-17: **`entropy` public surface (R-DEV-2, MISSING).** sklearn exposes
  `entropy(labels)` (`:1268`) as a public `cluster` function; ferrolearn has only
  the private `fn entropy_from_counts`.
- REQ-18: **PyO3 binding (R-DEFER-1).** `import sklearn.metrics` exposes these
  clustering metrics; `ferrolearn-python` exposes no shim.
- REQ-19: **ferray substrate (R-SUBSTRATE).** `clustering.rs` imports
  `ndarray::{Array1,Array2}` + `num_traits::Float`, not `ferray-core`.
- REQ-20: **Crate-root re-export of `AmiMethod` + `adjusted_mutual_info_with_method`
  (R-DEV-2).** Both are pub in the module but absent from the `lib.rs` re-export,
  so the AMI averaging selector is unreachable via `ferrolearn_metrics::*`.

## Acceptance criteria

All expected values are from the live sklearn 1.5.2 oracle (run from `/tmp`),
never literal-copied from ferrolearn (R-CHAR-3).

- AC-1 (REQ-6 pin, deterministic): `rand_score([0,0,1,1],[0,0,0,0])` must equal
  sklearn **0.3333333333333333**. ferrolearn **panics (attempt to subtract with
  overflow) in debug builds** at the `d = comb_n − sum_comb_a − sum_comb_b + ...`
  line and FAILS.
- AC-2 (REQ-4 pin, deterministic): `calinski_harabasz_score` on
  `X=[[0],[0],[10],[10]]`, `labels=[0,0,1,1]` must equal sklearn **1.0**
  (`intra_disp == 0`). ferrolearn returns **`Ok(inf)`** and FAILS.
- AC-3 (REQ-10/11/12 pin, deterministic): `homogeneity_score([],[])` must equal
  sklearn **1.0** (empty → `(1,1,1)`). ferrolearn returns
  `Err(InsufficientSamples)` and FAILS. Same for `completeness_score`,
  `v_measure_score`, `homogeneity_completeness_v_measure`.
- AC-4 (REQ-7, must stay green): `adjusted_mutual_info([0,0,1,1,2,2],
  [0,0,0,1,1,1])` must equal sklearn **0.2987924581708903** within 1e-9
  (arithmetic default, exact EMI). Already pinned by
  `test_ami_mixed_labels_matches_sklearn_arithmetic`.
- AC-5 (REQ-8, value-correct when method passed): `normalized_mutual_info_score(
  [0,0,1,1,2,2],[0,0,0,1,1,1], NmiMethod::Arithmetic)` must equal sklearn's
  **default** `normalized_mutual_info_score(...)` = **0.5158037429793889** — i.e.
  ferrolearn matches only because the caller supplies `Arithmetic`; sklearn picks
  it by default.
- AC-6 (REQ-9): `v_measure_score([0,0,1,1,2,2],[0,0,0,1,1,1], beta=2.0)` must equal
  sklearn **0.5578858913022597**; ferrolearn's `v_measure_score` has no `beta` and
  cannot express it (the `beta` form is only on
  `homogeneity_completeness_v_measure`, which yields **0.5578...** at `beta=2`).
- AC-7 (REQ-5/13/14/16, must stay green — value-correct today):
  `adjusted_rand_score([0,0,1,1],[0,1,0,1]) == -0.5`;
  `fowlkes_mallows_score([0,0,1,1,2,2],[0,0,0,1,1,1]) == 0.4714045207910317`;
  `mutual_info_score([0,1,1,0,1,0],[0,1,0,0,1,1]) == 0.0566330122651324`;
  `pair_confusion_matrix([0,0,1,2],[0,0,1,1]) == [[8,2],[0,2]]`;
  `contingency_matrix([0,0,1,1,2,2],[1,0,2,1,0,2]) == [[1,1,0],[0,1,1],[1,0,1]]`.
- AC-8 (REQ-1/3, must stay green): on
  `X=[[0,0],[0.1,0.1],[10,10],[10.1,10.1]]`, `labels=[0,0,1,1]`,
  `silhouette_score ≈ 0.9899997499937521`, `davies_bouldin_score ≈
  0.009999999999997726`, `calinski_harabasz_score ≈ 20000.0` — all match sklearn
  (no noise points; euclidean). These bound the correctness that IS present.

## REQ status table

Binary (R-DEFER-2). All sixteen functions + `NmiMethod` are existing pub APIs
re-exported at the crate root (`lib.rs`); that re-export is the non-test
production-consumer surface (grandfathered, S5/R-DEFER-1). Cites use symbol
anchors (ferrolearn) / `file:line` (sklearn 1.5.2). Live oracle = installed
sklearn 1.5.2, run from `/tmp`. **SHIPPED** = value-matches the oracle on the
supported path with a non-test consumer + a characterizing test + symbol/`file:line`
anchors. Honest underclaim (R-HONEST-3).

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (`silhouette_samples`/`silhouette_score` core) | SHIPPED | impl `pub fn silhouette_samples`/`pub fn silhouette_score` in `clustering.rs` mirror `_unsupervised.py:206,310-318` (`(b-a)/max(a,b)`, size-1→0). Consumer: `lib.rs` re-export. Verification: AC-8 (`silhouette_score≈0.98999975` on the 4-point fixture matches the live oracle; `test_silhouette_samples_mean_matches_score`, `test_silhouette_perfect_clustering`). |
| REQ-2 (`silhouette` `metric`/`sample_size`/`random_state`) | NOT-STARTED | open prereq blocker #797 (RNG boundary). `silhouette_score`/`silhouette_samples` are **euclidean-only** (`fn row_euclidean_dist`) with no `metric`, `sample_size`, or `random_state`. sklearn's `sample_size` path draws `random_state.permutation(n)[:sample_size]` (`:133-141`) — numpy `RandomState` ordering is not reproducible in Rust (RNG-boundary, like the silhouette precedent in goal.md). |
| REQ-3 (`davies_bouldin_score`) | SHIPPED | impl `pub fn davies_bouldin_score` mirrors `_unsupervised.py:399`. Consumer: `lib.rs` re-export. Verification: AC-8 (`db≈0.0099999999` matches oracle; `test_db_well_separated_is_low`, `test_db_noise_points_ignored`). Caveat: ferrolearn excludes `-1` noise (no sklearn analog) — not exercised by sklearn's contract on noise-free input. |
| REQ-4 (`calinski_harabasz_score`) | NOT-STARTED | open prereq blocker #798. impl `pub fn calinski_harabasz_score` value-matches on non-degenerate input (AC-8, `ch≈20000`) but **returns `Ok(inf)` when `w_ss==0`**; sklearn returns **1.0** (`_unsupervised.py:385-389`). Pin: `X=[[0],[0],[10],[10]]`,`labels=[0,0,1,1]` → sklearn **1.0**, ferro `inf`. |
| REQ-5 (`adjusted_rand_score`) | SHIPPED | impl `pub fn adjusted_rand_score` mirrors `_supervised.py:353,444-452`. Consumer: `lib.rs` re-export. Verification: `adjusted_rand_score([0,0,1,1],[0,1,0,1]) == -0.5` matches oracle (AC-7); `test_ari_identical_labels_is_one`, `test_ari_permuted_labels_is_one`. (Combinatorial form; algebraically equal to sklearn's pair-confusion form on integer inputs.) |
| REQ-6 (`rand_score`) | NOT-STARTED | open prereq blocker #799. impl `pub fn rand_score` computes `d = comb_n − sum_comb_a − sum_comb_b + sum_comb_c` in **`u64`**; the subtraction **underflows and panics in debug** on valid degenerate input. Pin: `rand_score([0,0,1,1],[0,0,0,0])` → sklearn **0.3333333333333333**, ferro panics (attempt to subtract with overflow). sklearn special-cases `num==den || den==0 → 1.0` (`:337-341`) which ferrolearn lacks. |
| REQ-7 (`adjusted_mutual_info` + EMI) | SHIPPED | impl `pub fn adjusted_mutual_info` / `pub fn adjusted_mutual_info_with_method` mirror `_supervised.py:936,1040-1057`; **`AmiMethod::Arithmetic` default matches** sklearn `average_method='arithmetic'` (`:937`); exact EMI in `fn expected_mutual_info` matches Cython `expected_mutual_information` to ~1e-13. Consumer: `lib.rs` re-export (of `adjusted_mutual_info`). Verification: `test_ami_mixed_labels_matches_sklearn_arithmetic` pins `0.2987924581708901` from the live oracle (AC-4). |
| REQ-8 (`normalized_mutual_info_score` default) | NOT-STARTED | open prereq blocker #800. impl `pub fn normalized_mutual_info_score(..., NmiMethod)` value-matches when `Arithmetic` passed (`0.5158037429793889`, AC-5) but **`method` is a required positional with no default**; sklearn defaults `average_method='arithmetic'` (`_supervised.py:1069-1070`). The default-omission is the R-DEV-2 ABI gap. |
| REQ-9 (`v_measure_score` `beta`) | NOT-STARTED | open prereq blocker #801. `pub fn v_measure_score` hardcodes `beta=1` (`2hc/(h+c)`); sklearn `v_measure_score(..., *, beta=1.0)` exposes `beta` (`_supervised.py:716`). Pin: `beta=2.0` → sklearn **0.5578858913022597** (reachable only via `homogeneity_completeness_v_measure(...,2.0)`). Also empty-input → error (sklearn 1.0, see REQ-12). |
| REQ-10 (`homogeneity_score`) | NOT-STARTED | open prereq blocker #802. impl `pub fn homogeneity_score` value-matches on non-empty (`0.4206...` on the mixed case) but **empty input → `Err(InsufficientSamples)`**; sklearn returns **1.0** (`_supervised.py:531-532`). Pin: `homogeneity_score([],[])` → sklearn 1.0, ferro error. |
| REQ-11 (`completeness_score`) | NOT-STARTED | open prereq blocker #803. impl `pub fn completeness_score` value-matches on non-empty (`0.6666...`) but **empty → error** (sklearn 1.0, `:531-532`). |
| REQ-12 (`homogeneity_completeness_v_measure`) | NOT-STARTED | open prereq blocker #804. `pub fn homogeneity_completeness_v_measure(...,beta)` exposes `beta` (good) but **`beta` is required-positional** (sklearn keyword-only default 1.0) and **empty input → error**; sklearn returns `(1.0,1.0,1.0)` (`_supervised.py:463,531-532`). |
| REQ-13 (`fowlkes_mallows_score`) | SHIPPED | impl `pub fn fowlkes_mallows_score` mirrors `_supervised.py:1184,1256-1259`. Consumer: `lib.rs` re-export. Verification: `fowlkes_mallows_score([0,0,1,1,2,2],[0,0,0,1,1,1]) == 0.4714045207910317` matches oracle (AC-7); `test_fmi_perfect`, `test_fmi_known_value`, `test_fmi_all_singletons`. (`sparse` kwarg is internal-only in sklearn — no value effect.) |
| REQ-14 (`mutual_info_score`) | NOT-STARTED | open prereq blocker #805. impl `pub fn mutual_info_score` value-matches (`0.0566330122651324` mixed, `0.0` single-cluster, AC-7) but **omits the `contingency=` keyword passthrough** (sklearn `:820,886-894`) — a real API-surface gap (used by `homogeneity_completeness_v_measure`/AMI/NMI in sklearn). Per-term eps-zero + `clip(.,0,None)` also absent (immaterial on tested cases). |
| REQ-15 (`contingency_matrix`) | NOT-STARTED | open prereq blocker #806. impl `pub fn contingency_matrix` value/ordering matches the **dense** sklearn output (`[[1,1,0],[0,1,1],[1,0,1]]`, AC-7) but **omits `eps`/`sparse`/`dtype`** (`_supervised.py:104-106`); returns `Array2<u64>` only. |
| REQ-16 (`pair_confusion_matrix`) | SHIPPED | impl `pub fn pair_confusion_matrix` mirrors `_supervised.py:190,259-264`. Consumer: `lib.rs` re-export. Verification: `pair_confusion_matrix([0,0,1,2],[0,0,1,1]) == [[8,2],[0,2]]` and `([0,0,1,1],[1,1,0,0]) == [[8,0],[0,4]]` match oracle (AC-7); `#[test]`s exercise the identities. |
| REQ-17 (`entropy` public surface, MISSING) | NOT-STARTED | open prereq blocker #807. sklearn exposes `entropy(labels)` as a public `cluster` function (`_supervised.py:1268`, empty→1.0, single→0.0); ferrolearn has only the private `fn entropy_from_counts`. No public `entropy` is exported. |
| REQ-18 (PyO3 binding) | NOT-STARTED | open prereq blocker #808. `ferrolearn-python` exposes no clustering-metric shim; `import ferrolearn` cannot call what `import sklearn.metrics` provides. |
| REQ-19 (ferray substrate) | NOT-STARTED | open prereq blocker #809. `clustering.rs` imports `ndarray::{Array1,Array2}` + `num_traits::Float`, not `ferray-core` (R-SUBSTRATE). |
| REQ-20 (crate-root `AmiMethod`/`adjusted_mutual_info_with_method`) | NOT-STARTED | open prereq blocker #810. `lib.rs` re-exports `adjusted_mutual_info` + `NmiMethod` but **not** `AmiMethod` or `adjusted_mutual_info_with_method`, so the AMI averaging selector is unreachable as `ferrolearn_metrics::AmiMethod` (only `ferrolearn_metrics::clustering::AmiMethod`). |

## Architecture

`clustering.rs` is a flat module of free functions plus two small `Copy` enums
(`AmiMethod`, `NmiMethod`). There are no fitted/unfitted types — these are
stateless metrics. Two families:

1. **Feature-space (unsupervised)** — `silhouette_score`/`silhouette_samples`,
   `davies_bouldin_score`, `calinski_harabasz_score`: generic over `F: Float`,
   taking `&Array2<F>` + `&Array1<isize>`. They share `fn unique_cluster_labels`
   (which **drops `-1` noise** — a ferrolearn convention with no sklearn analog;
   sklearn `LabelEncode`s every label) and `fn row_euclidean_dist`
   (**euclidean-only**, so REQ-2's `metric` is structurally absent). The two
   deterministic divergences here are `calinski_harabasz_score`'s `w_ss==0 →
   inf` (sklearn 1.0, REQ-4) and the missing `sample_size`/`random_state` RNG path
   (REQ-2).
2. **Label-agreement (supervised)** — everything else, taking `&Array1<isize>`×2,
   returning `Result<f64, _>` (or `Result<Array2<u64>, _>`). All route through
   `fn build_contingency_table` (`np.unique`-sorted classes, dense `u64` table).
   The pair-counting metrics (`adjusted_rand_score`, `rand_score`,
   `fowlkes_mallows_score`) use `fn n_choose_2` over `u64`; **`rand_score`'s `d`
   computation underflows in `u64`** (REQ-6) — the single panic-class divergence.
   The information-theoretic metrics (`mutual_info_score`, `adjusted_mutual_info`,
   `normalized_mutual_info_score`, `homogeneity`/`completeness`/`v_measure`) share
   `fn entropy_from_counts`; AMI adds the exact EMI (`fn expected_mutual_info` +
   `fn precompute_log_factorials`, the most numerically delicate code in the file
   and **verified to ~1e-13 against sklearn's Cython EMI**, REQ-7).

**Invariants held vs sklearn:** the core numeric formulas for silhouette, DB,
non-degenerate CH, ARI, FMI, MI, AMI (incl. EMI + arithmetic default), NMI (when
arithmetic passed), homogeneity/completeness (non-empty), v_measure (`beta=1`,
non-empty), contingency (dense), and pair-confusion are value-correct vs the live
oracle (AC-4/5/7/8). **Invariants NOT held vs sklearn:** `rand_score` underflow
panic (REQ-6); CH `intra_disp==0 → inf` vs 1.0 (REQ-4); empty-input convention
for the homogeneity family (REQ-10/11/12); `v_measure_score` missing `beta`
(REQ-9); NMI missing arithmetic default (REQ-8); `metric`/`sample_size`/
`random_state` (REQ-2); `eps`/`sparse`/`dtype` on `contingency_matrix` (REQ-15);
`contingency=` passthrough on `mutual_info_score` (REQ-14); public `entropy`
(REQ-17); PyO3 binding (REQ-18); ferray substrate (REQ-19); crate-root re-export
of `AmiMethod` (REQ-20). The `-1` noise convention on the feature-space metrics is
a ferrolearn extension with no sklearn counterpart — benign on noise-free input
but a latent divergence if a caller mirrors sklearn's "encode `-1` as a real
cluster" expectation.

**Missing functions.** sklearn's `cluster` package additionally exposes the
public function **`entropy`** (`_supervised.py:1268`) — ferrolearn has it only as
the private `fn entropy_from_counts` (REQ-17). `expected_mutual_information`
(Cython `_expected_mutual_info_fast.pyx`) is **private** sklearn surface (imported
internally by AMI), so it is not "missing" — ferrolearn's equivalent
`fn expected_mutual_info` is correctly private. `_generalized_average`,
`check_clusterings`, `_silhouette_reduce`, `check_number_of_labels` are likewise
private helpers. No other public `cluster` function is absent.

## Verification

Library crate (green at baseline `ec2d4b7b` for the existing — narrower —
contract; 44 clustering `#[test]`s + 2 `#[cfg(kani)]` range/non-negativity
proofs):
```
cargo test -p ferrolearn-metrics --lib clustering        # 44 passed, 0 failed
cargo clippy -p ferrolearn-metrics --all-targets -- -D warnings
cargo fmt --all --check
```
The existing `#[test]`s pin ferrolearn's behavior (incl. the value-correct
`test_ami_mixed_labels_matches_sklearn_arithmetic`, the EMI/arithmetic-default
pin) but do NOT cover the divergence cases below, so they leave REQ-2/4/6/8/9/10/
11/12/14/15/17/18/19/20 NOT-STARTED.

Live sklearn oracle (installed 1.5.2, run from `/tmp`) — the deterministic
divergences a critic should pin first (R-CHAR-3 expected values):
```
# REQ-6 (rand_score u64 underflow): sklearn 0.3333..., ferro panics in debug
python3 -c "from sklearn.metrics import rand_score; print(rand_score([0,0,1,1],[0,0,0,0]))"   # 0.3333333333333333
# REQ-4 (CH intra_disp==0 -> 1.0, not inf):
python3 -c "import numpy as np; from sklearn.metrics import calinski_harabasz_score as c; print(c(np.array([[0.],[0.],[10.],[10.]]), np.array([0,0,1,1])))"  # 1.0
# REQ-10/11/12 (empty -> 1.0/(1,1,1), not error):
python3 -c "from sklearn.metrics import homogeneity_score as h; print(h([],[]))"               # 1.0
python3 -c "from sklearn.metrics import homogeneity_completeness_v_measure as f; print(f([],[]))"  # (1.0, 1.0, 1.0)
# REQ-8 (NMI default arithmetic), REQ-9 (v_measure beta):
python3 -c "from sklearn.metrics import normalized_mutual_info_score as n; print(n([0,0,1,1,2,2],[0,0,0,1,1,1]))"  # 0.5158037429793889
python3 -c "from sklearn.metrics import v_measure_score as v; print(v([0,0,1,1,2,2],[0,0,0,1,1,1],beta=2.0))"      # 0.5578858913022597
# REQ-7 (AMI arithmetic default + EMI), must stay green:
python3 -c "from sklearn.metrics import adjusted_mutual_info_score as a; print(a([0,0,1,1,2,2],[0,0,0,1,1,1]))"    # 0.2987924581708903
# AC-7/8 baselines that must stay green (value-correct today):
python3 -c "from sklearn.metrics import adjusted_rand_score as r, fowlkes_mallows_score as f, mutual_info_score as m; print(r([0,0,1,1],[0,1,0,1]), f([0,0,1,1,2,2],[0,0,0,1,1,1]), m([0,1,1,0,1,0],[0,1,0,0,1,1]))"  # -0.5 0.4714045207910317 0.0566330122651324
python3 -c "import numpy as np; from sklearn.metrics import silhouette_score as s, davies_bouldin_score as d, calinski_harabasz_score as c; X=np.array([[0.,0.],[.1,.1],[10.,10.],[10.1,10.1]]); l=np.array([0,0,1,1]); print(s(X,l), d(X,l), c(X,l))"  # 0.98999975 0.00999999 20000.0
```
A characterization pin (R-CHAR-3) for each NOT-STARTED deterministic REQ belongs
in `ferrolearn-metrics/tests/divergence_clustering.rs`, asserting the live-sklearn
expected values above and FAILING against current `clustering.rs` (REQ-6 must
catch the panic; REQ-4/10/11/12 assert the value/`Ok` sklearn returns). The
SHIPPED REQs (1,3,5,7,13,16) are pinned green by the in-`#[cfg(test)]` suite +
the oracle baselines (AC-4/7/8).

## Blockers to open

- #797 — REQ-2 (`silhouette` `metric`/`sample_size`/`random_state`): euclidean-only;
  no subsampling. RNG-boundary (`_unsupervised.py:54,133-141`).
- #798 — REQ-4 (`calinski_harabasz_score`): `w_ss==0 → Ok(inf)` vs sklearn **1.0**
  (`_unsupervised.py:385-389`). Pin: `[[0],[0],[10],[10]]`/`[0,0,1,1]` → 1.0.
- #799 — REQ-6 (`rand_score`): `u64` underflow **panics in debug** on degenerate
  labelings; sklearn 0.3333 + `num==den||den==0→1.0` (`:333-343`). Pin:
  `[0,0,1,1]/[0,0,0,0]`.
- #800 — REQ-8 (`normalized_mutual_info_score`): no `average_method='arithmetic'`
  default — `NmiMethod` is required-positional (`_supervised.py:1069-1070`).
- #801 — REQ-9 (`v_measure_score`): no `beta` parameter (sklearn `*, beta=1.0`,
  `:716`). Pin: `beta=2.0` → 0.5578858913022597.
- #802 — REQ-10 (`homogeneity_score`): empty input → error vs sklearn **1.0**
  (`:531-532`).
- #803 — REQ-11 (`completeness_score`): empty input → error vs sklearn **1.0**.
- #804 — REQ-12 (`homogeneity_completeness_v_measure`): `beta` required-positional
  (sklearn keyword-only); empty → error vs `(1.0,1.0,1.0)` (`:463,531-532`).
- #805 — REQ-14 (`mutual_info_score`): no `contingency=` keyword passthrough
  (`:820,886-894`); per-term eps-zero + `clip(.,0,None)` absent.
- #806 — REQ-15 (`contingency_matrix`): no `eps`/`sparse`/`dtype` (`:104-106`).
- #807 — REQ-17 (`entropy`): missing public `entropy(labels)` function
  (`_supervised.py:1268`); only private `fn entropy_from_counts`.
- #808 — REQ-18: no `ferrolearn-python` clustering-metric binding.
- #809 — REQ-19: migrate `clustering.rs` off `ndarray`/`num-traits` to the ferray
  substrate (R-SUBSTRATE).
- #810 — REQ-20: re-export `AmiMethod` + `adjusted_mutual_info_with_method` at the
  crate root (`lib.rs`) so the AMI averaging selector is reachable as
  `ferrolearn_metrics::*`.
