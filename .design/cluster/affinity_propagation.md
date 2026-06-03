# Affinity Propagation Clustering (sklearn.cluster.AffinityPropagation)

<!--
tier: 3-component
status: draft
baseline-commit: 1f85b0eb
upstream: scikit-learn 1.5.2 (commit 156ef14)
upstream-paths:
  - sklearn/cluster/_affinity_propagation.py   # _equal_similarities_and_preferences (:22-33); _affinity_propagation main loop (:36-177) — preference on diagonal (:70), degeneracy noise (:78-80), responsibility update via argmax Y/Y2 (:88-102), availability colsum/clip update (:104-117), convergence over e window (:119-131), exemplar refinement (:140-162), no-cluster -1 fallback + ConvergenceWarning (:163-172); affinity_propagation functional API (:191-309); class AffinityPropagation(ClusterMixin, BaseEstimator) (:312); docstring example (:431-444); _parameter_constraints (:447-460); __init__ defaults damping=0.5/max_iter=200/convergence_iter=15/copy=True/preference=None/affinity='euclidean'/verbose=False/random_state=None (:462-481); fit — affinity_matrix_=-euclidean_distances(squared) (:511), median preference (:519-520), random_state (:525), cluster_centers_ (:543); predict via pairwise_distances_argmin (:547-580)
ferrolearn-module: ferrolearn-cluster/src/affinity_propagation.rs
parity-ops: AffinityPropagation (.__init__, .fit, .fit_predict, .labels_, .cluster_centers_, .cluster_centers_indices_, .affinity_matrix_, .n_iter_, .predict, random_state, affinity, preference)
crosslink-issue: 970
-->

## Summary

`ferrolearn-cluster/src/affinity_propagation.rs` mirrors scikit-learn's
`AffinityPropagation` (`sklearn/cluster/_affinity_propagation.py`,
`class AffinityPropagation(ClusterMixin, BaseEstimator)` `:312`) — exemplar-based
clustering by passing "responsibility" and "availability" messages between all pairs
of points until a stable set of exemplars emerges, with the number of clusters
determined by the data and the `preference` parameter rather than specified up front.
It exposes the unfitted `AffinityPropagation<F>` (`damping=0.5`, `max_iter=200`,
`convergence_iter=15`, `preference=None`), the fitted `FittedAffinityPropagation<F>`
(`cluster_centers_`, `labels_`, `exemplar_indices_`, `n_iter_`, plus an
`n_clusters()` helper), a `Fit<Array2<F>, ()>` impl, and a `fit_predict` convenience
mirroring `ClusterMixin.fit_predict`. It is re-exported at the crate root
(`pub use affinity_propagation::{AffinityPropagation, FittedAffinityPropagation}` in
`ferrolearn-cluster/src/lib.rs`).

**Honest assessment (R-HONEST-3).** This unit's load-bearing contract is the
**message-passing math and the resulting PARTITION (co-membership) up to a label
permutation, in the well-separated regime** — and that contract VALUE-matches the
algorithm sklearn runs. The responsibility update in `fn fit`
(`R[i,k] = S[i,k] - max_{k'!=k}(A[i,k'] + S[i,k'])` with damping) is mathematically
equivalent to sklearn's vectorized `Y`/`Y2` argmax form (`_affinity_propagation.py:88-102`),
and the availability update (`A[k,k] = sum_{i'!=k} max(0, R[i',k])`;
`A[i,k] = min(0, R[k,k] + sum_{i'!=i,k} max(0, R[i',k]))`) is equivalent to sklearn's
column-sum / clip form (`:104-117`). On well-separated blobs the co-membership and
`n_clusters` match. **These two SHIP** — but they ship through the crate re-export and
external library callers, NOT through a CPython binding: there is **no
`_RsAffinityPropagation` in `ferrolearn-python`** (confirmed by grep — see Probe 5),
unlike the `DBSCAN`/`Birch` siblings.

The NOT-STARTED surface is large and several pieces couple together. Exact VALUE
parity of `labels_` / `cluster_centers_indices_` / `n_iter_` is **not** shipped because
ferrolearn diverges from sklearn on: (1) the **default `preference`** — sklearn uses
`np.median` over the FULL n×n affinity matrix including the n diagonal zeros
(`:519-520`, value `-9.0` on the docstring `X`), ferrolearn medians ONLY the
`n(n-1)/2` off-diagonal entries (value `-13.0` — Probe 2); (2) **degeneracy noise
injection** (`:78-80`) — sklearn perturbs `S` with `random_state.standard_normal` to
break ties, ferrolearn injects none; (3) the **`random_state` parameter** (`:472`) —
absent; (4) the **convergence criterion** — sklearn tracks the per-sample
exemplar-membership window `e` and tests `se==convergence_iter or se==0` for all n
samples (`:119-131`), ferrolearn counts consecutive iterations with an unchanged
exemplar SET; (5) the **exemplar-refinement pass** (`:149-162`) — sklearn re-picks each
cluster's exemplar as the member maximizing intra-cluster similarity and relabels via
`searchsorted`, ferrolearn assigns by raw squared distance with an argmax fallback;
(6) the **`predict` method** (`:547-580`) — absent; (7) **`affinity='precomputed'`**
(`:343`) — euclidean-only; (8) the **`affinity_matrix_` attribute** (`:508`/`:511`) —
not exposed; (9) **non-convergence behavior** (`:140-172`) — sklearn returns
`labels=[-1]*n` + `ConvergenceWarning` when no exemplars, ferrolearn returns an
`Err(FerroError::ConvergenceFailure)`; (10) the **equal-similarities short-circuit**
(`:22-67`); (11) `copy`/`verbose` params; (12) the **ferray substrate** (R-SUBSTRATE) —
the unit imports `ndarray`/`num-traits`, not `ferray-core`.

## Live oracle probes (sklearn 1.5.2, run from /tmp; ferrolearn via `cargo test -p ferrolearn-cluster --lib`)

All expected values come from the installed sklearn 1.5.2 oracle or a sklearn
`file:line`, never literal-copied from ferrolearn (R-CHAR-3).

### Probe 1 — docstring example, exact VALUE (the parity target ferrolearn does NOT meet)

The class docstring fixture (`_affinity_propagation.py:431-444`),
`X = [[1,2],[1,4],[1,0],[4,2],[4,4],[4,0]]`:
```
python3 -c "import numpy as np; from sklearn.cluster import AffinityPropagation; \
X=np.array([[1,2],[1,4],[1,0],[4,2],[4,4],[4,0]]); \
m=AffinityPropagation(random_state=5).fit(X); \
print(m.labels_.tolist(), m.cluster_centers_indices_.tolist(), m.cluster_centers_.tolist(), \
m.predict([[0,0],[4,4]]).tolist(), m.n_iter_)"
# sklearn: labels_ [0, 0, 0, 1, 1, 1]   cluster_centers_indices_ [0, 3]
#          cluster_centers_ [[1, 2], [4, 2]]   predict([[0,0],[4,4]]) [0, 1]   n_iter_ 16
```
ferrolearn cannot reproduce this exact tuple: it has no `random_state` (so the
`:78-80` noise that sklearn injects is absent), no `predict`, no
`cluster_centers_indices_` attribute, and — critically — a different default
`preference` (Probe 2). The PARTITION shape `[0,0,0,1,1,1]` (two clusters of three)
is the kind of co-membership ferrolearn CAN match on separated data, but the
absolute exemplar indices `[0, 3]` and the labeling/iteration count are governed by
the refinement pass + noise + convergence window it does not implement.

### Probe 2 — default `preference`: full-matrix median vs off-diagonal median (REQ-2, the prime fixer candidate)

sklearn: `preference = np.median(self.affinity_matrix_)` (`fit` `:519-520`) — the
median over the FULL n×n matrix, where `affinity_matrix_ = -euclidean_distances(X, squared=True)`
has a **zero diagonal** at `fit` time (preference is placed on the diagonal only
LATER, inside `_affinity_propagation` `:70`). ferrolearn (`fn fit`, the `pref`
else-branch) medians ONLY the `n(n-1)/2` off-diagonal upper-triangle entries.
```
python3 -c "import numpy as np; from sklearn.cluster import AffinityPropagation; \
X=np.array([[1,2],[1,4],[1,0],[4,2],[4,4],[4,0]]); \
m=AffinityPropagation(random_state=5).fit(X); A=m.affinity_matrix_; \
print('diag', np.diag(A).tolist()); \
print('median_full', float(np.median(A))); \
off=A[np.triu_indices(6,1)]; print('median_offdiag', float(np.median(off)))"
# diag         [-9.0, -9.0, -9.0, -9.0, -9.0, -9.0]   (== -euclidean_distances squared diagonal is 0; this is -||xi-xk||^2 for the OTHER entries)
# median_full     -9.0   <- sklearn default preference
# median_offdiag -13.0   <- ferrolearn default preference
```
The diagonal of `affinity_matrix_` is **0** (a point's squared distance to itself);
with 6 samples the matrix has 6 diagonal zeros and 30 off-diagonal negative entries,
so the full-matrix median (`-9.0`) lands well above the off-diagonal-only median
(`-13.0`). On THIS fixture both preferences happen to yield the same 2-cluster
partition (verified: `preference=-9.0` and `preference=-13.0` both give
`labels_=[0,0,0,1,1,1]`), so the divergence does not always surface in the partition
— but it IS a different default value, and on other data it changes `n_clusters`.
This is the single most-confident **minimal-fixable** divergence: replace the
upper-triangle median with a median over all n² affinity-matrix entries (the n
self-distances are 0, the off-diagonal are `-||xi-xk||²`). It still needs a blocker +
fixer pass (a fix changes the public default behavior and must land with a
sklearn-grounded test), so it is classified NOT-STARTED below, not silently patched.

### Probe 3 — well-separated blobs PARTITION (the SHIPPED co-membership contract)

```
python3 -c "import numpy as np; from sklearn.cluster import AffinityPropagation; \
from sklearn.datasets import make_blobs; \
X,_=make_blobs(n_samples=12,centers=3,cluster_std=0.4,random_state=42); \
m=AffinityPropagation(random_state=0).fit(X); \
print('n_clusters', len(m.cluster_centers_indices_)); print('labels', m.labels_.tolist())"
# sklearn: n_clusters 3   labels [0, 0, 1, 0, 2, 0, 2, 1, 2, 1, 1, 2]
```
On clearly-separated blobs ferrolearn's message-passing converges to the same
3-block co-membership partition (the `test_three_clusters` in-tree test pins this
shape: points within a blob share a label, points across blobs do not). The
co-membership and `n_clusters` SHIP up to a label permutation; the absolute label
integers / exemplar indices do NOT (governed by the refinement pass + RNG noise it
lacks).

### Probe 4 — responsibility-update equivalence (sklearn `:88-102` vs ferrolearn `fn fit`)

sklearn computes, per row `i`, `I=argmax_k(A[i,k]+S[i,k])`, `Y=max`, `Y2=second-max`,
then `Rnew[i,k] = S[i,k] - Y` for `k!=I` and `Rnew[i,I] = S[i,I] - Y2`. ferrolearn
computes `Rnew[i,k] = S[i,k] - max_{k'!=k}(A[i,k']+S[i,k'])`. For `k != I` the
`max_{k'!=k}` over all k' excluding k still includes the global argmax I, so it equals
`Y`; for `k == I` excluding I leaves the second-largest, `Y2`. The two forms are
algebraically identical. Damping `R = damping*R + (1-damping)*Rnew` matches sklearn's
`tmp *= 1-damping; R *= damping; R += tmp` (`:100-102`). This equivalence underpins
the SHIPPED math REQ.

### Probe 5 — non-test consumer: NO CPython binding

```
grep -rn "AffinityPropagation\|RsAffinity" ferrolearn-python/   # (empty — no binding)
grep -rn "AffinityPropagation" ferrolearn-cluster/src/lib.rs
# ferrolearn-cluster/src/lib.rs: pub use affinity_propagation::{AffinityPropagation, FittedAffinityPropagation};
```
Unlike `DBSCAN` (`_RsDBSCAN`) and `Birch` (`_RsBirch`), there is **no
`_RsAffinityPropagation`** in `ferrolearn-python/src/clusterers.rs` or `extras.rs`,
and no Python wrapper. The only non-test production consumer is the crate-root
re-export in `lib.rs`. Per R-DEFER-1/S-5 the boundary estimator type
`AffinityPropagation` IS the public API and is grandfathered (it predates this
audit), so the math/partition REQs SHIP on the strength of that re-export + external
library callers; a dedicated CPython-binding REQ (REQ-10) is NOT-STARTED.

## Requirements

- REQ-1 (message-passing math): the responsibility and availability updates with
  damping compute the same fixed-point messages as sklearn's vectorized
  `Y`/`Y2` argmax + column-sum/clip form (`_affinity_propagation.py:88-117`).
- REQ-2 (default preference): when `preference=None`, use the median over the FULL
  affinity matrix (n self-distances of 0 plus the off-diagonal `-||xi-xk||²`),
  matching `np.median(self.affinity_matrix_)` (`:519-520`).
- REQ-3 (partition / `n_clusters`): on well-separated data the co-membership
  partition and the number of clusters match sklearn up to a label permutation.
- REQ-4 (`random_state` + degeneracy noise): expose `random_state` and inject
  `(eps*S + tiny*100) * standard_normal` into `S` to break ties (`:78-80`), enabling
  exact `labels_`/`cluster_centers_indices_`/`n_iter_` VALUE parity.
- REQ-5 (convergence criterion): declare convergence via the per-sample
  exemplar-membership window `e` (`se==convergence_iter or se==0` for all n,
  `:119-131`), so `n_iter_` matches.
- REQ-6 (exemplar refinement + labeling): after detecting exemplars, re-pick each
  cluster's exemplar as the intra-cluster-similarity maximizer and relabel via
  `np.unique` + `searchsorted` (`:149-162`).
- REQ-7 (`predict`): assign new samples to the nearest `cluster_centers_` via
  pairwise argmin (`:547-580`); `-1` for every sample when there are no centers.
- REQ-8 (`affinity` param): support `affinity='euclidean'|'precomputed'` (`:343`,
  `:506-511`).
- REQ-9 (`affinity_matrix_` + `cluster_centers_indices_` attributes): expose the
  fitted attributes sklearn stores (`:508`/`:511`, `:528`).
- REQ-10 (CPython binding): bind `AffinityPropagation` into `ferrolearn-python` as a
  non-test consumer of the crate type (mirroring `_RsDBSCAN`/`_RsBirch`).
- REQ-11 (non-convergence semantics): on no-exemplar degeneracy return
  `labels=[-1]*n` + empty centers + a `ConvergenceWarning` (`:140-172`), not an
  `Err`.
- REQ-12 (equal-similarities short-circuit): handle the
  `_equal_similarities_and_preferences` case (`:22-67`) — n clusters or 1 cluster +
  warning.
- REQ-13 (`copy` / `verbose` params): mirror the remaining constructor params
  (`:468`/`:471`).
- REQ-14 (ferray substrate, R-SUBSTRATE): the array / RNG layer is `ferray-core` /
  `ferray::random`, not `ndarray` / (absent) `rand`.

## Acceptance criteria

- AC-1: on `make_blobs(centers=3, cluster_std=0.4)` the ferrolearn co-membership
  partition equals sklearn's up to a permutation, and `n_clusters() == 3`
  (in-tree `test_three_clusters` exercises the 3-blob shape).
- AC-2: with `preference=None` on the docstring `X`, the computed preference equals
  `np.median(-euclidean_distances(X, squared=True))` = `-9.0`, not `-13.0`.
- AC-3: `AffinityPropagation(random_state=5).fit(X).labels_` reproduces
  `[0,0,0,1,1,1]`, `cluster_centers_indices_` reproduces `[0,3]`, `n_iter_` reproduces
  `16` on the docstring `X` (Probe 1).
- AC-4: `predict([[0,0],[4,4]])` on the fitted docstring model returns `[0,1]`.
- AC-5: a degenerate run with no exemplars returns `labels_ = [-1; n]` + a
  convergence warning, not an `Err`.
- AC-6: the responsibility/availability updates match sklearn message values within a
  documented float tolerance on a small fixed `S` (Probe 4 equivalence).

## REQ status

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (message-passing math) | SHIPPED | impl `fn fit` in `affinity_propagation.rs` (`R_new[i,k] = s[[i,k]] - max_val` over `kp != k`; `A[k,k] = sum max(0, R[i',k])`, `A[i,k] = min(0, R[k,k]+sum)` with `damping*old + (1-damping)*new`) is algebraically equivalent to sklearn `_affinity_propagation.py:88-102` (`tmp[ind, I] = S[ind, I] - Y2`) and `:104-117` (`tmp -= np.sum(tmp, axis=0); ... tmp.clip(0, inf)`). Equivalence shown in Probe 4. Consumer: crate re-export `pub use affinity_propagation::{AffinityPropagation, ...}` in `lib.rs` + external library callers (grandfathered boundary type, R-DEFER-1/S-5). Verification: `cargo test -p ferrolearn-cluster --lib affinity_propagation` (19 passed) + Probe 4. |
| REQ-3 (partition / n_clusters) | SHIPPED | impl `fn fit` exemplar detection (`r[[k,k]] + a[[k,k]] > 0`) + nearest-exemplar assignment (Step 6) yields the same co-membership as sklearn on separated data (Probe 3: sklearn 3 clusters on `make_blobs(centers=3)`). In-tree `test_three_clusters` / `test_two_clusters` pin the co-membership shape. Consumer: crate re-export in `lib.rs`. Verification: `cargo test -p ferrolearn-cluster --lib` (`test_three_clusters`, `test_two_clusters` pass) + Probe 3. Underclaim: SHIPPED only for the PARTITION up to permutation in the well-separated regime — absolute label/exemplar VALUES are NOT-STARTED (REQ-4/6). |
| REQ-2 (default preference) | SHIPPED | impl `fn fit` `pref` else-branch now medians ALL `n²` entries of `s` (the n zero diagonal + each off-diagonal value twice; `all_entries` push over `for i { for j in 0..n }` then `median_of_sorted`), matching `np.median(self.affinity_matrix_)` (`_affinity_propagation.py:519-520`) — `-9.0` on docstring `X`, not `-13.0`. Consumer: crate re-export in `lib.rs`. Verification: `pin_req2_default_preference_partition` in `tests/divergence_affinity_propagation.rs` (live-oracle fixture `make_blobs(15,centers=3,cluster_std=0.5,random_state=5)`: ferrolearn default now k=3 matching sklearn, was k=2). Fixed in #971. |
| REQ-4 (random_state + degeneracy noise) | NOT-STARTED | open prereq blocker #972. No `random_state` field on `AffinityPropagation`; `fn fit` injects no noise into `s`. sklearn `:78-80` adds `(eps*S + tiny*100) * random_state.standard_normal(...)`; couples to ferray::random (REQ-14). Blocks exact `labels_`/`n_iter_` VALUE parity (AC-3). |
| REQ-5 (convergence criterion) | NOT-STARTED | open prereq blocker #973. `fn fit` counts `stable_count` over an unchanged exemplar SET; sklearn `:119-131` tracks the membership window `e` and tests `se==convergence_iter or se==0` for all n samples. Different `n_iter_`. |
| REQ-6 (exemplar refinement + labeling) | NOT-STARTED | open prereq blocker #974. `fn fit` Step 6 assigns by raw squared distance with an argmax fallback and labels by `exemplar_indices` position; sklearn `:149-162` refines each cluster's exemplar (`argmax sum S[ii,ii]`) then relabels via `np.unique` + `np.searchsorted`. Different exemplar set + label values. |
| REQ-7 (predict) | NOT-STARTED | open prereq blocker #975. No `Predict` impl (the `//!` states "does NOT implement Predict"); sklearn `predict` (`:547-580`) uses `pairwise_distances_argmin(X, cluster_centers_)`. Blocks AC-4. |
| REQ-8 (affinity param) | NOT-STARTED | open prereq blocker #976. euclidean-only: `fn fit` Step 1 hardcodes `S = -squared_euclidean`. No `affinity` field, no `'precomputed'` path (sklearn `:343`, `:506-511`). |
| REQ-9 (affinity_matrix_ / cluster_centers_indices_ attrs) | NOT-STARTED | open prereq blocker #977. `FittedAffinityPropagation` exposes `cluster_centers_`/`labels_`/`exemplar_indices_`/`n_iter_` but NOT `affinity_matrix_` (sklearn `:508`/`:511`) nor `cluster_centers_indices_` (sklearn `:528`; `exemplar_indices()` is the closest analog but is named differently and is `Vec<usize>` not an `ndarray`). |
| REQ-10 (CPython binding) | NOT-STARTED | open prereq blocker #978. No `_RsAffinityPropagation` in `ferrolearn-python` (Probe 5 grep empty); siblings `_RsDBSCAN`/`_RsBirch` exist. |
| REQ-11 (non-convergence semantics) | NOT-STARTED | open prereq blocker #979. `fn fit` returns `Err(FerroError::ConvergenceFailure)` when no exemplars; sklearn `:163-172` returns `labels=[-1]*n` + empty `cluster_centers_indices_` + `ConvergenceWarning` (exception vs warning+labels, R-DEV-2). Blocks AC-5. |
| REQ-12 (equal-similarities short-circuit) | NOT-STARTED | open prereq blocker #980. No `_equal_similarities_and_preferences` analog; `fn fit` runs the full loop for identical points (`test_identical_points` passes but via the argmax fallback, not the short-circuit). sklearn `:22-67` returns n or 1 clusters + a warning. |
| REQ-13 (copy / verbose params) | NOT-STARTED | open prereq blocker #981. `AffinityPropagation` has `damping`/`max_iter`/`convergence_iter`/`preference` only; sklearn `__init__` `:468`/`:471` also has `copy=True`/`verbose=False`. |
| REQ-14 (ferray substrate) | NOT-STARTED | open prereq blocker #982. `affinity_propagation.rs` imports `ndarray::{Array1, Array2}` + `num_traits::Float`, not `ferray-core`; no RNG layer at all (couples to REQ-4 needing `ferray::random`). R-SUBSTRATE-2. |

## Architecture

The module follows the crate's unfitted/fitted split (CLAUDE.md naming): unfitted
`AffinityPropagation<F>` holds the four hyperparameters with builder setters
(`with_damping`/`with_max_iter`/`with_convergence_iter`/`with_preference`) and a
`new()`/`Default` mirroring sklearn `__init__` defaults
(`_affinity_propagation.py:462-481`) for the four it implements. `Fit<Array2<F>, ()>`
produces `FittedAffinityPropagation<F>` storing `cluster_centers_`, `labels_`
(`Array1<isize>` — sklearn uses `int`, R-DEV-3 dtype nuance), `exemplar_indices_`, and
`n_iter_`, with introspection getters and an `n_clusters()` helper.

`fn fit` proceeds: validate `damping ∈ [0.5,1.0)` / `max_iter≥1` / `convergence_iter≥1`
(mirroring `_parameter_constraints` `:447-460`, though raised as
`FerroError::InvalidParameter` rather than sklearn's `ValueError`); short-circuit
`n==1`; build the similarity matrix `S = -||xi-xk||²` (sklearn's
`-euclidean_distances(X, squared=True)`, `:511`); set the diagonal preference (median
of off-diagonal — REQ-2 divergence vs sklearn's full-matrix median + later diagonal
placement `:70`,`:519-520`); zero-initialize `R`/`A`; iterate the
responsibility/availability message updates with damping (REQ-1, equivalent to
`:88-117`); detect exemplars where `R[k,k]+A[k,k] > 0` with a strict→argmax fallback;
assign each non-exemplar to its nearest exemplar by raw squared distance (REQ-3/6
divergence vs sklearn's refinement `:149-162`); build `cluster_centers_` from exemplar
rows (sklearn `:543` `X[cluster_centers_indices_]`).

Key invariants: `cluster_centers_.nrows() == n_clusters() == exemplar_indices_.len()`
(pinned by `test_cluster_centers_match_exemplars` / `test_exemplar_indices_are_valid`);
labels are in `[0, n_clusters)` (`test_labels_in_range`) — note this differs from
sklearn, which can emit `-1` labels on non-convergence (REQ-11). The
`exemplar_indices_` field is the structural analog of sklearn's
`cluster_centers_indices_` but is exposed under a different name and as `Vec<usize>`
(REQ-9).

The convergence loop differs structurally from sklearn (REQ-5): sklearn keeps an
`n_samples × convergence_iter` membership window `e` and breaks when every sample's
window is all-1 or all-0 with `K>0` (`:119-131`); ferrolearn breaks after
`convergence_iter` consecutive iterations with an identical exemplar SET. The absence
of the `:78-80` RNG tie-break means ferrolearn is deterministic without a
`random_state`, but cannot reproduce sklearn's exact iterate path on tie-prone data
(REQ-4).

## Verification

```bash
cargo test -p ferrolearn-cluster --lib affinity_propagation   # 19 passed, 0 failed (run this iteration)
cargo clippy -p ferrolearn-cluster --all-targets -- -D warnings
cargo fmt --all --check
```
SHIPPED-REQ oracle commands (live sklearn 1.5.2, from /tmp):
- REQ-1 (Probe 4): the responsibility/availability algebra equivalence — verified by
  expanding sklearn `:88-117` against `fn fit`; the in-tree `test_two_clusters` /
  `test_three_clusters` co-membership tests exercise the converged messages.
- REQ-3 (Probe 3): `AffinityPropagation(random_state=0).fit(make_blobs(centers=3,
  cluster_std=0.4, random_state=42))` → 3 clusters; ferrolearn `test_three_clusters`
  pins the same 3-block co-membership shape.

NOT-STARTED REQs each have a sklearn-grounded acceptance test waiting (AC-2 for REQ-2,
AC-3 for REQ-4/5/6, AC-4 for REQ-7, AC-5 for REQ-11). Until those pass, the
corresponding REQ stays NOT-STARTED — none is reframed as SHIPPED.

## Blockers to open (director assigns #NNN)

1. "Blocker for REQ-2 of affinity_propagation: default preference must median the FULL
   affinity matrix (n zero self-distances + off-diagonal `-||xi-xk||²`, value `-9.0`
   on docstring X), not the off-diagonal-only upper triangle (`-13.0`). sklearn
   `_affinity_propagation.py:519-520`." -p high -l blocker  **(prime minimal-fixable
   candidate)**
2. "Blocker for REQ-4 of affinity_propagation: add `random_state` param + inject
   `(eps*S + tiny*100)*standard_normal` degeneracy noise into S (sklearn `:78-80`);
   requires ferray::random (REQ-14)." -p high -l blocker
3. "Blocker for REQ-5 of affinity_propagation: convergence via the per-sample
   membership window `e` (`se==convergence_iter or se==0` for all n), sklearn
   `:119-131`, for matching `n_iter_`." -p high -l blocker
4. "Blocker for REQ-6 of affinity_propagation: exemplar-refinement pass + relabel via
   `np.unique`/`searchsorted`, sklearn `:149-162`, for matching exemplar set + label
   values." -p high -l blocker
5. "Blocker for REQ-7 of affinity_propagation: implement `predict` via
   pairwise-distances argmin to `cluster_centers_` (sklearn `:547-580`); `-1` when no
   centers." -p high -l blocker
6. "Blocker for REQ-8 of affinity_propagation: add `affinity='euclidean'|'precomputed'`
   param + precomputed-matrix path (sklearn `:343`, `:506-511`)." -p medium -l blocker
7. "Blocker for REQ-9 of affinity_propagation: expose `affinity_matrix_` +
   `cluster_centers_indices_` fitted attributes (sklearn `:508`/`:511`/`:528`)." -p
   medium -l blocker
8. "Blocker for REQ-10 of affinity_propagation: bind AffinityPropagation into
   ferrolearn-python as `_RsAffinityPropagation` (mirror `_RsDBSCAN`/`_RsBirch`)." -p
   medium -l blocker
9. "Blocker for REQ-11 of affinity_propagation: non-convergence returns
   `labels=[-1]*n` + empty centers + ConvergenceWarning, not
   `Err(ConvergenceFailure)` (sklearn `:163-172`, R-DEV-2)." -p high -l blocker
10. "Blocker for REQ-12 of affinity_propagation: equal-similarities short-circuit
    (`_equal_similarities_and_preferences`, sklearn `:22-67`)." -p low -l blocker
11. "Blocker for REQ-13 of affinity_propagation: add `copy`/`verbose` constructor
    params (sklearn `:468`/`:471`)." -p low -l blocker
12. "Blocker for REQ-14 of affinity_propagation: migrate array/RNG layer to
    ferray-core + ferray::random (R-SUBSTRATE-2)." -p medium -l blocker
