# Mean Shift (sklearn.cluster.MeanShift)

<!--
tier: 3-component
status: draft
baseline-commit: beb6862e
upstream: scikit-learn 1.5.2 (commit 156ef14)
upstream-paths:
  - sklearn/cluster/_mean_shift.py          # estimate_bandwidth (:43-106); _mean_shift_single_seed (:110-130); mean_shift function (:137-246); get_bin_seeds (:249-299); class MeanShift(ClusterMixin, BaseEstimator) (:302-578); _parameter_constraints (:439-447); __init__ (:449-466); fit (:468-560); predict (:562-578)
ferrolearn-module: ferrolearn-cluster/src/mean_shift.rs
parity-ops: MeanShift (.__init__, .fit, .predict, .fit_predict, .cluster_centers_, .labels_, .n_iter_); estimate_bandwidth
crosslink-issue: TBD-director
-->

## Summary

`ferrolearn-cluster/src/mean_shift.rs` mirrors scikit-learn's `MeanShift`
(`sklearn/cluster/_mean_shift.py`, `class MeanShift(ClusterMixin, BaseEstimator)`
`:302-578`) — flat-kernel mode-seeking clustering. It exposes the unfitted
`MeanShift<F>` (`bandwidth: Option<F>`, `max_iter`, `tol`), the fitted
`FittedMeanShift<F>` (stores `cluster_centers_`, `labels_`, `n_iter_`), a `Predict`
impl assigning to the nearest center, and a `fit_predict` convenience mirroring
`ClusterMixin.fit_predict`. It is re-exported at the crate root (`pub use
mean_shift::{FittedMeanShift, MeanShift}` in `ferrolearn-cluster/src/lib.rs`).

**Under honest underclaim (R-HONEST-3), the only behavior that VALUE-matches the
live sklearn 1.5.2 oracle end-to-end is the explicit-bandwidth clustering on
well-separated data — and even there only the *partition* (label co-membership)
matches, not the `cluster_centers_` values, the `labels_` integer values, nor
`n_iter_`.** The two load-bearing divergences:

1. **Auto-bandwidth is the WRONG heuristic.** sklearn's `estimate_bandwidth`
   (`_mean_shift.py:43-106`) is a **kNN** heuristic: `mean over all points of (max
   distance to its `k` nearest neighbours)`, where `k = int(n_samples * quantile)`,
   default `quantile=0.3`. ferrolearn's `fn estimate_bandwidth` computes the
   **median of all pairwise distances** — a different statistic that produces a
   different value (Probe 2). On the `two_blobs` fixture sklearn returns `0.19073`
   (→ 4 clusters); ferrolearn's median returns `14.001` (→ everything merged into
   1 cluster). They are not the same algorithm and do not value-match.
2. **De-duplication keeps the highest-intensity mode; ferrolearn averages the
   group.** sklearn sorts converged modes by `(intensity, coordinate)` descending,
   then greedily marks all modes within `bandwidth` of a retained mode as duplicates
   — the retained `cluster_centers_` are *actual converged modes* (Probe 1:
   `[[3.333…,6.0],[1.333…,0.666…]]`), sorted most-populated-first
   (`_mean_shift.py:529-546`). ferrolearn merges nearby modes and stores the
   **arithmetic mean of the group** as the center (`fn fit`, the `group` averaging
   loop) — different values — and does NOT sort by intensity, so `labels_` integer
   values are permuted relative to sklearn even when the partition agrees.

In addition, the constructor surface diverges: ferrolearn lacks `seeds`,
`bin_seeding`, `min_bin_freq`, `cluster_all`, `n_jobs` and adds a non-sklearn `tol`
parameter; there is no `cluster_all=False` orphan `-1` labelling; and there is no
`ferrolearn-python` binding (`grep -rln MeanShift ferrolearn-python/` is empty —
`RsKMeans` is registered but no `RsMeanShift`). `MeanShift` / `FittedMeanShift` are
existing pub APIs (grandfathered per S5/R-DEFER-1); their only non-test consumer is
the crate re-export.

## Live oracle probes (sklearn 1.5.2, run from /tmp)

Expected values are from the installed sklearn 1.5.2 oracle, never literal-copied
from ferrolearn (R-CHAR-3). Fixtures: `docs` = the upstream docstring fixture
`[[1,1],[2,1],[1,0],[4,7],[3,5],[3,6]]` (`_mean_shift.py:428-429`); `blobs` = the
`two_blobs()` test fixture in `mean_shift.rs` (5 points near origin + 5 near
`(10,10)`, 10×2).

### Probe 1 — cluster_centers_ / labels_ / n_iter_ / predict (the load-bearing probe)
```
python3 -c "import numpy as np; from sklearn.cluster import MeanShift; \
X=np.array([[1,1],[2,1],[1,0],[4,7],[3,5],[3,6]],dtype=float); \
m=MeanShift(bandwidth=2).fit(X); \
print(m.cluster_centers_.tolist(), m.labels_.tolist(), m.n_iter_, m.predict([[0,0],[5,5]]).tolist())"
# centers [[3.3333333333333335, 6.0], [1.3333333333333333, 0.6666666666666666]]
# labels  [1, 1, 1, 0, 0, 0]   n_iter 2   predict [1, 0]
```
**Findings:** (a) `cluster_centers_` are exact converged modes, NOT group means;
(b) they are sorted by intensity DESCENDING (`(3.33,6)` first — 3 points — then
`(1.33,0.66)` — 3 points; ties broken by coordinate); (c) labels index into that
sorted order. ferrolearn keeps the first-seen unclaimed mode and averages its group,
in seed (data) order, so both the center VALUES and the label INTEGERS diverge.

### Probe 2 — estimate_bandwidth is kNN, not median pairwise (the core auto-bw divergence)
```
python3 -c "import numpy as np; from sklearn.cluster import estimate_bandwidth; \
X=np.array([[0.,0.],[0.2,0.1],[-0.1,0.2],[0.1,-0.1],[0.,0.1],[10.,10.],[10.2,10.1],[9.9,10.2],[10.1,9.9],[10.,10.1]]); \
print(estimate_bandwidth(X), estimate_bandwidth(X,quantile=0.5))"
# default(quantile=0.3): 0.19073262114491085   quantile=0.5: 0.296910323321918
```
ferrolearn `fn estimate_bandwidth` = median of all 45 pairwise Euclidean distances on
`blobs` = **14.00142849854971** (verified via `scipy.spatial.distance.pdist`). sklearn
`estimate_bandwidth` builds `NearestNeighbors(n_neighbors=int(n*quantile))`, sums
`max(kneighbors_distance, axis=1)` and divides by `n_samples`
(`_mean_shift.py:95-106`). **Different statistic, different value** — and `bandwidth=None`
clustering therefore diverges in cluster count: sklearn auto → 4 clusters on `blobs`
(`MeanShift().fit(blobs).cluster_centers_.shape == (4,2)`), ferrolearn auto-median
(14.0) → 1 cluster. ferrolearn also has no `quantile` / `n_samples` / `random_state` /
`n_jobs` parameters, and `fn estimate_bandwidth` is a private helper (no public
`estimate_bandwidth` analog).

### Probe 3 — explicit-bandwidth partition on well-separated blobs (the one agreement)
```
python3 -c "import numpy as np; from sklearn.cluster import MeanShift; \
X=np.array([[0.,0.],[0.2,0.1],[-0.1,0.2],[0.1,-0.1],[0.,0.1],[10.,10.],[10.2,10.1],[9.9,10.2],[10.1,9.9],[10.,10.1]]); \
m=MeanShift(bandwidth=2.0).fit(X); print(np.round(m.cluster_centers_,5).tolist(), m.labels_.tolist(), m.n_iter_)"
# centers [[10.04, 10.06], [0.04, 0.06]]   labels [1,1,1,1,1,0,0,0,0,0]   n_iter 1
```
ferrolearn `MeanShift::<f64>::new().with_bandwidth(2.0).fit(blobs)` produces the SAME
2-way **partition** (points 0-4 together, 5-9 together) but with the OPPOSITE label
integers (origin → 0 in ferrolearn, → 1 in sklearn because sklearn sorts the
`(10,10)` mode first), and centers as group means rather than the sklearn convention.
Partition co-membership agrees; `labels_` / `cluster_centers_` / `n_iter_` VALUES do
not. This is why the in-tree tests assert label co-membership (`labels[0]==labels[1]`)
and never the integer values.

### Probe 4 — n_iter_ semantics
```
# docs fixture bandwidth=2 -> n_iter_ = 2 ; blobs bandwidth=2 -> n_iter_ = 1
```
sklearn `_mean_shift_single_seed` (`_mean_shift.py:110-130`) returns
`completed_iterations`, which is incremented AFTER the convergence/`max_iter` check —
i.e. the count of *additional* steps after the first mean. `n_iter_` is the `max` over
all seeds (`:514`). ferrolearn `fn mean_shift_single` sets `n_iter = iter + 1` and
`fit` takes `last_n_iter = max(n_iter)` — a different convention (off-by-one and the
break-before-increment vs increment-after-check semantics differ), so `n_iter_`
diverges (sklearn `n_iter_=2` on `docs`, ferrolearn counts the loop iterations).

### Probe 5 — cluster_all=False orphan labelling
```
python3 -c "import numpy as np; from sklearn.cluster import MeanShift; \
X=np.array([[0.,0.],[0.1,0.],[0.,0.1],[10.,10.],[10.1,10.],[5.0,5.0]]); \
print(MeanShift(bandwidth=0.5,cluster_all=False).fit(X).labels_.tolist())"
# [0, 0, 0, 1, 1, 2]   (the (5,5) point becomes its own cluster here)
```
sklearn `cluster_all=False` fills `labels` with `-1` then assigns only points within
`bandwidth` of a center (`_mean_shift.py:552-557`); orphans keep `-1`. ferrolearn has
no `cluster_all` parameter and ALWAYS assigns every point to the nearest center
(`fn fit` label loop + `Predict::predict`), so it can never produce `-1`. (Note
ferrolearn's `labels_` is `Array1<usize>`, which cannot represent `-1` at all — a
type-level barrier to the `cluster_all=False` contract.)

### Probe 6 — bandwidth validation + error ABI
```
python3 -c "from sklearn.cluster import MeanShift; import numpy as np; \
MeanShift(bandwidth=0).fit(np.array([[0.,0.],[5.,5.]]))"
# raises sklearn.utils._param_validation.InvalidParameterError (bandwidth must be in (0, inf))
```
sklearn `_parameter_constraints["bandwidth"] = [Interval(Real, 0, None,
closed="neither"), None]` (`_mean_shift.py:440`) — `(0, inf)`, so `bandwidth=0`
and `bandwidth<0` both raise `InvalidParameterError` at `fit`. ferrolearn `fn fit`
rejects `bandwidth <= 0` with `FerroError::InvalidParameter { name: "bandwidth",
reason: "must be positive" }` — the bound agrees (`(0,inf)` ≡ `> 0`) but the error
TYPE/message differ from sklearn's `InvalidParameterError` ABI (R-DEV-2). The "No
point was within bandwidth of any seed" `ValueError` (`:516-522`) cannot occur in
ferrolearn because each seed is a data point and always contains itself.

### Probe 7 — non-test consumer
`grep -rln "MeanShift\|RsMeanShift" ferrolearn-python/` is **empty** — there is no
PyO3 binding (`RsKMeans` is registered in `ferrolearn-python/src/lib.rs:24`, but no
`RsMeanShift`), so `import ferrolearn` cannot reach `MeanShift`. `grep -rn MeanShift
ferrolearn-cluster/src/` outside `mean_shift.rs` finds only the crate re-export and
doc-comment references in `lib.rs`. The sole non-test consumer of `fit` /
`fit_predict` / `predict` / the accessors is the crate re-export.

## Requirements

- REQ-1: **explicit-bandwidth clustering PARTITION (R-DEV-1).** Mirror
  `MeanShift(bandwidth=b).fit(X)` producing the same grouping of samples into modes
  for a given explicit `bandwidth`. ferrolearn's `fn fit` runs mean-shift from every
  data point as a seed and merges nearby modes (`mean_shift_single` + the merge
  loop), recovering the correct *partition* on well-separated data (Probe 3) — but
  the center VALUES (group mean vs retained mode), the `labels_` INTEGERS (no
  intensity sort), and `n_iter_` all diverge, so this is a partition-only claim, not
  a full value-parity claim.
- REQ-2: **`cluster_centers_` VALUE + intensity ordering (R-DEV-1/3 — the core
  de-dup divergence).** Mirror `_mean_shift.py:529-546`: sort converged modes by
  `(intensity, coordinate)` descending, greedily retain the highest-intensity mode in
  each `bandwidth`-ball, keep the ACTUAL converged mode (not a group mean). ferrolearn
  averages each merged group and emits centers in seed order (Probe 1).
- REQ-3: **`estimate_bandwidth` kNN heuristic (R-DEV-1 — the core auto-bw
  divergence).** Mirror `estimate_bandwidth(X, quantile=0.3, n_samples, random_state,
  n_jobs)` (`_mean_shift.py:43-106`): `n_neighbors=int(n*quantile)` (≥1),
  `NearestNeighbors`, `bandwidth = sum(max(kneighbors_dist,axis=1)) / n_samples`.
  ferrolearn's `fn estimate_bandwidth` computes the median of all pairwise distances
  — a different value (Probe 2: 0.19073 vs 14.001 on `blobs`) and a private helper
  with no `quantile`/`n_samples`/`random_state` params and no public function.
- REQ-4: **`labels_` VALUE parity (R-DEV-1/3).** Mirror
  `MeanShift.fit().labels_` (`_mean_shift.py:548-557`). ferrolearn's label integers
  diverge from the oracle because the centers are ordered differently (REQ-2,
  no intensity sort) and because `cluster_all=False` orphan `-1` labelling is absent
  (REQ-6). Partition co-membership agrees on benign fixtures (Probe 3), integer
  values do not.
- REQ-5: **constructor surface `seeds`/`bin_seeding`/`min_bin_freq`/`cluster_all`/
  `n_jobs` + drop non-sklearn `tol` (R-DEV-2).** sklearn `__init__`
  (`_mean_shift.py:449-466`) takes `bandwidth, seeds, bin_seeding, min_bin_freq,
  cluster_all, n_jobs, max_iter` (all keyword-only). ferrolearn `MeanShift<F>` has
  only `bandwidth`/`max_iter`/`tol` — missing five sklearn params and adding a `tol`
  field sklearn does not expose (sklearn hard-codes `stop_thresh = 1e-3 * bandwidth`,
  `:113`, NOT a user parameter).
- REQ-6: **`cluster_all=False` orphan `-1` labelling (R-DEV-3).** sklearn
  `_mean_shift.py:552-557` labels orphans `-1`. ferrolearn `labels_` is
  `Array1<usize>` (cannot hold `-1`) and always assigns the nearest center — the
  `-1` contract is unrepresentable without a type change.
- REQ-7: **`bin_seeding`/`get_bin_seeds` + `min_bin_freq` seeding (R-DEV-1/2).**
  sklearn `get_bin_seeds(X, bin_size=bandwidth, min_bin_freq)` (`_mean_shift.py:249-299`)
  bins points onto a `bandwidth`-grid and seeds from bins with `>= min_bin_freq`
  members. ferrolearn always seeds from every data point; no binning.
- REQ-8: **convergence stop-threshold = `1e-3 * bandwidth` (R-DEV-1).** sklearn
  `_mean_shift_single_seed` stops when `||new_mean - old_mean|| <= 1e-3 * bandwidth`
  (`_mean_shift.py:113,124-127`). ferrolearn stops when `shift < tol` with a
  *fixed* default `tol = 1e-3` (absolute, NOT scaled by bandwidth), and `tol` is a
  user knob sklearn does not have — so convergence behaviour diverges whenever
  `bandwidth != 1`.
- REQ-9: **`n_iter_` semantics (R-DEV-3).** sklearn `n_iter_` = `max` over seeds of
  `completed_iterations` (incremented after the convergence check, `:124-129,514`).
  ferrolearn `n_iter_` = `max` over seeds of `iter+1` (loop counter) — an off-by-one /
  different-convention value (Probe 4: sklearn 2 vs ferrolearn's loop count on `docs`).
- REQ-10: **error ABI `InvalidParameterError` + "No point within bandwidth"
  `ValueError` (R-DEV-2).** sklearn raises `InvalidParameterError` for `bandwidth<=0`
  (`_parameter_constraints`, `:440`) and `ValueError` when no seed has neighbours
  (`:516-522`). ferrolearn raises `FerroError::InvalidParameter` (different type/ABI)
  and cannot hit the no-neighbour case (each seed is a data point).
- REQ-11: **PyO3 binding (R-DEFER-1/3).** No `RsMeanShift` in `ferrolearn-python`
  (`RsKMeans` is registered, `lib.rs:24`, but not `MeanShift`) — `import ferrolearn`
  cannot reach `MeanShift`.
- REQ-12: **ferray substrate (R-SUBSTRATE).** `mean_shift.rs` imports
  `ndarray::{Array1, Array2}` + `num_traits::Float`, not `ferray-core` /
  `ferray::linalg` (R-SUBSTRATE-1/2).

## Acceptance criteria

All expected values from the live sklearn 1.5.2 oracle (run from `/tmp`), never
literal-copied from ferrolearn (R-CHAR-3). Fixtures: `docs` =
`[[1,1],[2,1],[1,0],[4,7],[3,5],[3,6]]`; `blobs` = `two_blobs()` (10×2).

- AC-1 (REQ-1, partition agrees / values diverge): `MeanShift(bandwidth=2).fit(docs).labels_`
  → sklearn `[1,1,1,0,0,0]`; ferrolearn recovers the same 2-way grouping
  (`{0,1,2}` vs `{3,4,5}`) but with permuted integers and group-mean centers.
- AC-2 (REQ-2, diverges): `MeanShift(bandwidth=2).fit(docs).cluster_centers_`
  → sklearn `[[3.3333333333333335,6.0],[1.3333333333333333,0.6666666666666666]]`
  (exact converged modes, intensity-sorted). ferrolearn emits group-mean centers in
  seed order — different values and order.
- AC-3 (REQ-3, diverges): `estimate_bandwidth(blobs)` → sklearn `0.19073262114491085`
  (quantile=0.3); `estimate_bandwidth(blobs, quantile=0.5)` → `0.296910323321918`.
  ferrolearn `fn estimate_bandwidth(blobs)` = `14.00142849854971` (median pairwise) —
  a different statistic; `MeanShift().fit(blobs)` (auto) → sklearn 4 clusters vs
  ferrolearn 1 cluster.
- AC-4 (REQ-9, diverges): `MeanShift(bandwidth=2).fit(docs).n_iter_` → sklearn `2`;
  ferrolearn's loop-counter convention differs.
- AC-5 (REQ-6, diverges): `MeanShift(bandwidth=0.5,cluster_all=False).fit(X)` can
  return `-1` labels for orphans (sklearn `_mean_shift.py:555`); ferrolearn
  `labels_: Array1<usize>` cannot represent `-1`.
- AC-6 (REQ-5/7/11): `MeanShift().get_params()` exposes
  `bandwidth/bin_seeding/cluster_all/max_iter/min_bin_freq/n_jobs/seeds`; ferrolearn
  has only `bandwidth/max_iter/tol`, and `import ferrolearn; ferrolearn.MeanShift`
  does not exist (no binding).

## REQ status table

Binary (R-DEFER-2). `MeanShift` / `FittedMeanShift` are existing pub APIs re-exported
at the crate root (the only non-test consumer; grandfathered S5/R-DEFER-1). Cites use
symbol anchors (ferrolearn) / `file:line` (sklearn 1.5.2, commit 156ef14). Live oracle
= installed sklearn 1.5.2, run from `/tmp`. Honest underclaim (R-HONEST-3): **no REQ
value-matches the `MeanShift` contract end-to-end with a non-test consumer** — the
explicit-bandwidth PARTITION agrees on well-separated data (REQ-1) but the
`cluster_centers_` values, the `labels_` integers, and `n_iter_` all diverge, and
the load-bearing auto-bandwidth heuristic is the wrong statistic (REQ-3). Every REQ is
NOT-STARTED. Suggested blocker numbers are SUGGESTIONS — the director creates the real
issues.

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (explicit-bw PARTITION) | NOT-STARTED | open prereq blocker **#960** (depends on #961/#963). impl `fn fit` (seed-per-point `mean_shift_single` + merge loop) recovers the correct 2-way grouping on `blobs`/`docs` (Probe 3, AC-1), so the PARTITION matches — but the `cluster_centers_` VALUES (group mean vs sklearn's retained converged mode, REQ-2), the `labels_` INTEGERS (no intensity sort, REQ-4), and `n_iter_` (REQ-9) diverge. Not a full value-parity claim; the only consumer of `fit` is the crate re-export (no binding, REQ-11). |
| REQ-2 (`cluster_centers_` value + intensity order) | NOT-STARTED | open prereq blocker **#961**. sklearn sorts converged modes by `(intensity, coord)` descending and retains the actual highest-intensity mode per `bandwidth`-ball (`_mean_shift.py:529-546`); `cluster_centers_` on `docs` = `[[3.333…,6.0],[1.333…,0.666…]]` (AC-2). ferrolearn `fn fit` averages each merged group (`group` loop → `center = mean(group)`) and emits centers in seed order with no intensity sort — different values and ordering. |
| REQ-3 (`estimate_bandwidth` kNN heuristic) | NOT-STARTED | open prereq blocker **#962**. sklearn `estimate_bandwidth` = `sum(max(kNN_dist,axis=1))/n` with `k=int(n*quantile)`, `quantile=0.3` (`_mean_shift.py:95-106`); `estimate_bandwidth(blobs)=0.19073262114491085` (AC-3). ferrolearn `fn estimate_bandwidth` computes the MEDIAN of all pairwise distances = `14.00142849854971` on `blobs` — a different statistic → auto-`MeanShift` yields 1 cluster vs sklearn's 4. No `quantile`/`n_samples`/`random_state`/`n_jobs` params; private helper, no public `estimate_bandwidth`. NOTE on `mean_shift.rs:201`: the `dists.len() % 2 == 0` two-sided median IS a correct sample median of the pairwise-distance vector, but sklearn's `estimate_bandwidth` does NOT take a median of pairwise distances at all (it is kNN, not pdist-median) — so the median computation is internally fine yet implements the wrong heuristic. |
| REQ-4 (`labels_` VALUE parity) | NOT-STARTED | open prereq blocker **#960** (depends on #961). sklearn `labels_` indexes into intensity-sorted `cluster_centers_` (`_mean_shift.py:548-557`); on `docs` = `[1,1,1,0,0,0]` (AC-1). ferrolearn assigns the nearest of its seed-ordered, group-mean centers (`fn fit` label loop) — same partition on benign fixtures, permuted integers. Gated on REQ-2 (center ordering) + REQ-6 (orphan `-1`). |
| REQ-5 (ctor surface seeds/bin_seeding/min_bin_freq/cluster_all/n_jobs; drop `tol`) | NOT-STARTED | open prereq blocker **#964**. sklearn `__init__` (`_mean_shift.py:449-466`) = `bandwidth, seeds, bin_seeding, min_bin_freq, cluster_all, n_jobs, max_iter`. ferrolearn `MeanShift<F>` = `bandwidth/max_iter/tol` (`fn new` + builders) — missing 5 sklearn params; `tol` is non-sklearn (sklearn hard-codes `1e-3*bandwidth`, `:113`). |
| REQ-6 (`cluster_all=False` orphan `-1`) | NOT-STARTED | open prereq blocker **#965**. sklearn fills `labels=-1` then assigns only within-`bandwidth` points (`_mean_shift.py:552-557`); orphans stay `-1` (AC-5). ferrolearn `labels_: Array1<usize>` (`FittedMeanShift`) cannot represent `-1` and always assigns the nearest center (`fn fit` / `Predict::predict`). Type change + `cluster_all` param required. |
| REQ-7 (`bin_seeding` / `get_bin_seeds` / `min_bin_freq`) | NOT-STARTED | open prereq blocker **#966**. sklearn `get_bin_seeds(X, bandwidth, min_bin_freq)` bins to a `bandwidth`-grid, seeds from bins with `>=min_bin_freq` members (`_mean_shift.py:249-299`); `fit` selects seeding via `bin_seeding` (`:491-495`). ferrolearn always seeds from every data point (`fn fit` seed loop) — no binning, no `bin_seeding`/`min_bin_freq`/`seeds`. |
| REQ-8 (stop-threshold `1e-3*bandwidth`) | NOT-STARTED | open prereq blocker **#967**. sklearn `stop_thresh = 1e-3 * bandwidth` (`_mean_shift.py:113`), convergence `||Δmean|| <= stop_thresh` (`:124-127`). ferrolearn `fn mean_shift_single` stops at `shift < tol` with default `tol=1e-3` (ABSOLUTE, unscaled) — diverges for any `bandwidth != 1`; `tol` is also a non-sklearn user knob (REQ-5). |
| REQ-9 (`n_iter_` semantics) | NOT-STARTED | open prereq blocker **#968**. sklearn `n_iter_ = max(completed_iterations)`, incremented AFTER the convergence/`max_iter` check (`_mean_shift.py:124-129,514`); on `docs` = `2` (AC-4). ferrolearn `fn fit` takes `max(iter+1)` from `mean_shift_single` (loop counter) — off-by-one / different convention. |
| REQ-10 (error ABI `InvalidParameterError` + no-neighbour `ValueError`) | NOT-STARTED | open prereq blocker **#969**. sklearn rejects `bandwidth<=0` with `InvalidParameterError` (`(0,inf)`, `_mean_shift.py:440`) and raises `ValueError` when no seed has neighbours (`:516-522`). ferrolearn `fn fit` raises `FerroError::InvalidParameter` (matching bound, different type/message ABI) and cannot hit the no-neighbour case (each seed is a data point so always self-neighbours). |
| REQ-11 (PyO3 binding) | NOT-STARTED | open prereq blocker **#970**. `grep -rln "MeanShift\|RsMeanShift" ferrolearn-python/` is EMPTY — `RsKMeans` is registered (`ferrolearn-python/src/lib.rs:24`) but no `RsMeanShift`, so `import ferrolearn` cannot reach `MeanShift`. The only non-test consumer of `fit`/`predict`/`fit_predict`/accessors is the crate re-export (`lib.rs`). |
| REQ-12 (ferray substrate) | NOT-STARTED | open prereq blocker **#971**. `mean_shift.rs` imports `ndarray::{Array1, Array2}` + `num_traits::Float`; not migrated to `ferray-core` / `ferray::linalg` (R-SUBSTRATE-1/2). |

## Architecture

`mean_shift.rs` follows the unfitted/fitted split (CLAUDE.md naming):
`MeanShift<F>` (`bandwidth: Option<F>`, `max_iter: usize`, `tol: F`) →
`Fit<Array2<F>, ()>` → `FittedMeanShift<F>` (private `cluster_centers_: Array2<F>`,
`labels_: Array1<usize>`, `n_iter_: usize`). Generic over
`F: Float + Send + Sync + 'static`; every public method returns
`Result<_, FerroError>` (R-CODE-2). `FittedMeanShift` implements `Predict<Array2<F>>`
(nearest-center assignment), mirroring sklearn `MeanShift.predict` /
`pairwise_distances_argmin` (`_mean_shift.py:562-578`). A `fit_predict` convenience
mirrors `ClusterMixin.fit_predict`.

**Fit path (`fn fit`).** Validates `n_samples > 0` and (if explicit) `bandwidth > 0`,
else calls `fn estimate_bandwidth`. Then: (1) run `fn mean_shift_single` from EVERY
data point as a seed (always — no `seeds`/`bin_seeding`, REQ-7); each climbs to a
local mode, stopping at `shift < tol` or `max_iter`; (2) greedily merge modes within
`bandwidth` of a first-seen unclaimed mode and store the **group mean** as the center
(REQ-2 divergence — sklearn retains the actual highest-intensity mode, not a mean,
and sorts by intensity); (3) assign each training point to the nearest center
(`Array1<usize>` — always assigns, no `cluster_all=False`/`-1`, REQ-6).

**The pipeline is a mean-shift *variant*, not the sklearn contract** — see Summary
and Probes 1-2:
- `fn estimate_bandwidth` = median of pairwise distances; sklearn `estimate_bandwidth`
  = kNN `sum(max(kNN_dist))/n` (`_mean_shift.py:95-106`) — REQ-3.
- merge stores `mean(group)`; sklearn de-dup retains the highest-`(intensity,coord)`
  converged mode (`_mean_shift.py:529-546`) — REQ-2.
- centers are seed-ordered; sklearn sorts by intensity descending → `labels_` integers
  diverge — REQ-4.
- `tol` is an absolute user knob; sklearn uses `1e-3 * bandwidth` hard-coded — REQ-8.
- `n_iter_` = `max(iter+1)`; sklearn = `max(completed_iterations)` — REQ-9.

**Invariants held vs sklearn:** explicit-bandwidth PARTITION (label co-membership on
well-separated data, Probe 3/AC-1); `cluster_centers_.shape == (n_clusters,
n_features)`; `labels_.len() == n_samples`; labels in `[0, n_clusters)`; predict
shape-mismatch error; deterministic (no RNG in the explicit-bandwidth path);
`bandwidth<=0` rejected (bound agrees, type differs — REQ-10).

**Invariants NOT held vs sklearn:** `cluster_centers_` VALUE + intensity order
(REQ-2); auto-bandwidth heuristic (REQ-3 — the core auto-bw divergence); `labels_`
integers (REQ-4); the constructor surface (REQ-5); `cluster_all=False` orphan `-1`
(REQ-6); `bin_seeding`/`get_bin_seeds`/`min_bin_freq`/`seeds` (REQ-7); the
`1e-3*bandwidth` stop-threshold (REQ-8); `n_iter_` semantics (REQ-9); the error ABI
(REQ-10); the PyO3 binding (REQ-11); the ferray substrate (REQ-12).

**Consumer wiring.** The only non-test consumer is the crate re-export
(`pub use mean_shift::{FittedMeanShift, MeanShift}`, `ferrolearn-cluster/src/lib.rs`).
There is no `ferrolearn-python` binding (Probe 7) and no other in-crate consumer.

## Verification

Library crate (green at baseline `beb6862e` for the existing variant behavior):
```
cargo test -p ferrolearn-cluster --lib mean_shift     # 19 passed; 0 failed
cargo clippy -p ferrolearn-cluster --all-targets -- -D warnings
cargo fmt --all --check
```
The 19 in-tree `#[test]`s (`test_two_blobs_correct_clusters`, `test_labels_length`,
`test_cluster_centers_shape`, `test_centers_near_blob_means`,
`test_single_cluster_large_bandwidth`, `test_auto_bandwidth_finds_two_clusters`,
`test_predict_on_new_points`, `test_predict_shape_mismatch`, `test_single_point`,
`test_empty_data_error`, `test_invalid_bandwidth_error`, `test_zero_bandwidth_error`,
`test_n_iter_non_zero`, `test_f32_support`, `test_three_clusters`,
`test_identical_points_single_cluster`, `test_center_coordinates_reasonable`,
`test_predict_labels_range`, `test_default_trait`) pin ferrolearn's current
**variant** behavior — label co-membership, shapes, ranges, error edges, f32 support.
**None compares `cluster_centers_` values, `labels_` integers, `n_iter_`, or the
auto-bandwidth value against the live sklearn `MeanShift` oracle**, so they stay
green despite the divergences. In particular `test_auto_bandwidth_finds_two_clusters`
only asserts `n_clusters() >= 1` (a deliberate hedge — auto-median actually yields 1
cluster on `blobs`, vs sklearn's 4), which masks REQ-3.

Live sklearn oracle (installed 1.5.2, run from `/tmp`) — the divergences a critic
should pin (R-CHAR-3 expected values). **Pin REQ-3 (estimate_bandwidth) and REQ-2
(center value/order) FIRST** — they are the load-bearing value divergences; REQ-4
(label integers) unblocks once REQ-2 lands:
```
# REQ-1/REQ-4 (partition agrees, integers diverge) + REQ-2 (centers) + REQ-9 (n_iter)
python3 -c "import numpy as np; from sklearn.cluster import MeanShift; \
X=np.array([[1,1],[2,1],[1,0],[4,7],[3,5],[3,6]],dtype=float); m=MeanShift(bandwidth=2).fit(X); \
print(m.cluster_centers_.tolist(), m.labels_.tolist(), m.n_iter_)"
# [[3.3333333333333335,6.0],[1.3333333333333333,0.6666666666666666]] [1,1,1,0,0,0] 2
# REQ-3 (estimate_bandwidth — wrong statistic)
python3 -c "import numpy as np; from sklearn.cluster import estimate_bandwidth; \
X=np.array([[0.,0.],[0.2,0.1],[-0.1,0.2],[0.1,-0.1],[0.,0.1],[10.,10.],[10.2,10.1],[9.9,10.2],[10.1,9.9],[10.,10.1]]); \
print(estimate_bandwidth(X), estimate_bandwidth(X,quantile=0.5))"
# 0.19073262114491085 0.296910323321918   (ferrolearn median pairwise = 14.00142849854971)
# REQ-6 (cluster_all=False -> -1)
python3 -c "import numpy as np; from sklearn.cluster import MeanShift; \
X=np.array([[0.,0.],[0.1,0.],[0.,0.1],[10.,10.],[10.1,10.],[5.0,5.0]]); \
print(MeanShift(bandwidth=0.5,cluster_all=False).fit(X).labels_.tolist())"   # [0,0,0,1,1,2]
```
A characterization pin (R-CHAR-3) for each NOT-STARTED REQ belongs in
`ferrolearn-cluster/tests/divergence_mean_shift.rs`, asserting the live-sklearn
expected values above and FAILING against current `mean_shift.rs`. Note
`test_auto_bandwidth_finds_two_clusters` must be corrected when REQ-3 lands (its
`>= 1` hedge masks the auto-bw divergence — R-HONEST-4).

ferrolearn-python (REQ-11 binding parity, after #970 lands):
```
cd /home/doll/ferrolearn/ferrolearn-python && maturin develop
PYTHONPATH=python python3 -m pytest tests/divergence_mean_shift.py -q
```
asserting `ferrolearn.MeanShift` exists and exposes `cluster_centers_` / `labels_` /
`n_iter_` / the sklearn parameter surface, matching `sklearn.cluster.MeanShift` on
the AC fixtures.

## Blockers to open

(Director creates the real issues; all numbers are SUGGESTIONS.)

- **#960** — REQ-1/REQ-4: align the explicit-bandwidth label INTEGERS with sklearn
  (depends on #961 center ordering); the partition already matches.
- **#961** — REQ-2 (**core, pin FIRST**): replace group-mean merge with sklearn's
  de-dup — sort modes by `(intensity, coord)` descending, retain the actual
  highest-intensity converged mode per `bandwidth`-ball (`_mean_shift.py:529-546`).
- **#962** — REQ-3 (**core, pin FIRST**): replace median-pairwise
  `estimate_bandwidth` with sklearn's kNN heuristic + `quantile`/`n_samples`/
  `random_state`/`n_jobs` params (`_mean_shift.py:43-106`); expose a public
  `estimate_bandwidth`.
- **#964** — REQ-5: add `seeds`/`bin_seeding`/`min_bin_freq`/`cluster_all`/`n_jobs`
  ctor params; drop the non-sklearn `tol` (`_mean_shift.py:449-466`).
- **#965** — REQ-6: support `cluster_all=False` orphan `-1` labelling (requires a
  signed label type) (`_mean_shift.py:552-557`).
- **#966** — REQ-7: implement `get_bin_seeds` + `bin_seeding`/`min_bin_freq`/`seeds`
  seeding (`_mean_shift.py:249-299,491-495`).
- **#967** — REQ-8: stop-threshold `1e-3 * bandwidth` (drop the absolute `tol`)
  (`_mean_shift.py:113,124-127`).
- **#968** — REQ-9: `n_iter_` = `max(completed_iterations)` per sklearn convention
  (`_mean_shift.py:124-129,514`).
- **#969** — REQ-10: `InvalidParameterError` error ABI + no-neighbour `ValueError`
  (`_mean_shift.py:440,516-522`).
- **#970** — REQ-11: add `RsMeanShift` to `ferrolearn-python` (fit / predict /
  fit_predict / cluster_centers_ / labels_ / n_iter_ + parameter surface).
- **#971** — REQ-12: migrate `mean_shift.rs` off `ndarray`/`num-traits` to
  `ferray-core` / `ferray::linalg` (R-SUBSTRATE).
