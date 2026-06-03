# Local Outlier Factor (sklearn.neighbors.LocalOutlierFactor)

<!--
tier: 3-component
status: draft
baseline-commit: a0cd459f
upstream: scikit-learn 1.5.2 (commit 156ef14)
upstream-paths:
  - sklearn/neighbors/_lof.py   # LocalOutlierFactor (:20); _parameter_constraints (:186-194); __init__ (:196-219); fit_predict + _check_novelty_fit_predict (:221-256); fit (:262-320); n_neighbors_ clamp + warning (:282-289); kneighbors(n_neighbors_) excluding self (:291-293); _lrd (:301-303); negative_outlier_factor_ = -mean(lrd_ratios) (:306-310); offset_ auto=-1.5 / percentile (:312-318); predict + gating (:322-354); _predict (:356-383); decision_function + gating (:385-424); score_samples + gating (:426-484); _local_reachability_density reach_dist/lrd (:486-511)
ferrolearn-module: ferrolearn-neighbors/src/local_outlier_factor.rs
parity-ops: LocalOutlierFactor
crosslink-issue: 844
-->

## Summary

`ferrolearn-neighbors/src/local_outlier_factor.rs` mirrors scikit-learn's
`sklearn.neighbors.LocalOutlierFactor` (`sklearn/neighbors/_lof.py`): the
unsupervised Local Outlier Factor anomaly detector that scores each sample by the
ratio of its local reachability density to those of its k nearest neighbors.
It provides the unfitted `LocalOutlierFactor<F>` and fitted
`FittedLocalOutlierFactor<F>` types, with `fit_predict`, `predict`,
`score_samples`, `decision_function`, and `lof_scores()`/`threshold()`
introspection.

Under honest underclaim (R-HONEST-3), **almost none of the sklearn contract is met
end-to-end**. The brute-force k-distance → reach_dist → lrd → LOF pipeline is
structurally present and the LOF *shape* is correct (the extreme outlier gets the
largest score), but it **does not value-match the live sklearn 1.5.2 oracle** — the
`negative_outlier_factor_` values diverge at ~1e-2 because of a neighbor
**tie-break** divergence, and the entire `offset_`/threshold machinery differs
(ferrolearn has **no `contamination='auto'`** path and **no `offset_` /
`negative_outlier_factor_` attributes** at all). The cleanest deterministic
divergences (no ties): the `contamination='auto'` default + `offset_=-1.5`, the
missing `negative_outlier_factor_`/`offset_` accessors, the novelty gating, and the
`decision_function` sign convention.

`LocalOutlierFactor`/`FittedLocalOutlierFactor` are existing pub APIs re-exported at
the crate root (`ferrolearn-neighbors/src/lib.rs`:
`pub use local_outlier_factor::{FittedLocalOutlierFactor, LocalOutlierFactor}`) —
the non-test production consumer (grandfathered S5/R-DEFER-1). There is **no
`ferrolearn-python` (PyO3) binding** and **no `ferrolearn` meta-crate re-export**
for this estimator (verified by `grep`); both are NOT-STARTED.

## Algorithm (sklearn — the contract)

`LocalOutlierFactor(n_neighbors=20, *, algorithm="auto", leaf_size=30,
metric="minkowski", p=2, metric_params=None, contamination="auto", novelty=False,
n_jobs=None)` (`__init__`, `:196-208`). `_parameter_constraints` (`:186-193`):
`contamination ∈ StrOptions({"auto"})` **or** `Interval(Real, 0, 0.5,
closed="right")`; `novelty ∈ {"boolean"}`; `n_neighbors` etc. inherited from
`NeighborsBase`.

**`fit(X, y=None)`** (`:262-320`):
1. `self._fit(X)`; `n_samples = self.n_samples_fit_`.
2. **n_neighbors clamp + warning** (`:282-289`): if `n_neighbors > n_samples`,
   `warnings.warn("n_neighbors (%s) is greater than the total number of samples
   (%s). n_neighbors will be set to (n_samples - 1) ...")`; then
   `self.n_neighbors_ = max(1, min(self.n_neighbors, n_samples - 1))` (`:289`).
3. `self._distances_fit_X_, _neighbors_indices_fit_X_ =
   self.kneighbors(n_neighbors=self.n_neighbors_)` (`:291-293`) — queries
   `n_neighbors_` neighbors **excluding self** (the `kneighbors(None, ...)` path
   drops the query point).
4. `self._lrd = self._local_reachability_density(...)` (`:301-303`).
5. `lrd_ratios_array = self._lrd[neighbors_indices] / self._lrd[:, np.newaxis]`
   (`:306-308`); **`self.negative_outlier_factor_ = -np.mean(lrd_ratios_array,
   axis=1)`** (`:310`) — the **negated** mean lrd ratio.
6. **`offset_`** (`:312-318`): `if contamination == "auto": offset_ = -1.5` else
   `offset_ = np.percentile(self.negative_outlier_factor_, 100.0 * contamination)`
   (a percentile in **negative-NOF space**).

**`_local_reachability_density(distances_X, neighbors_indices)`** (`:486-511`):
`dist_k = self._distances_fit_X_[neighbors_indices, self.n_neighbors_ - 1]`
(`:507`) — the neighbor's k-distance; `reach_dist_array = np.maximum(distances_X,
dist_k)` (`:508`); **`return 1.0 / (np.mean(reach_dist_array, axis=1) + 1e-10)`**
(`:511`) — inverse of the **mean** reach_dist, with a `+1e-10` damping.

**`fit_predict(X, y=None)`** (`:221-256`): gated by
`_check_novelty_fit_predict` (`available_if`, `:230`) — **raises `AttributeError`
when `novelty=True`** (`:222-227`). Returns `self.fit(X)._predict()` (`:256`).
`_predict(None)` (`:356-383`): `is_inlier = ones(n_samples_fit_)`;
`is_inlier[negative_outlier_factor_ < offset_] = -1` (`:380-381`).

**`predict(X)`** (`:333-354`): gated by `_check_novelty_predict` — **raises
`AttributeError` when `novelty=False`** (`:322-331`). `_predict(X)`:
`shifted = decision_function(X); is_inlier[shifted < 0] = -1` (`:375-378`).

**`score_samples(X)`** (`:438-484`): gated by `_check_novelty_score_samples` —
raises when `novelty=False` (`:426-436`). Queries the **training** neighbors of
new `X`, computes `X_lrd`, then `return -np.mean(self._lrd[neighbors_indices_X] /
X_lrd[:, np.newaxis], axis=1)` (`:481-484`) — the negated mean lrd ratio for new
data (self NOT in its own neighborhood).

**`decision_function(X)`** (`:398-424`): gated when `novelty=False`
(`:385-396`); returns `self.score_samples(X) - self.offset_` (`:424`). Negative →
outlier, positive → inlier.

## ferrolearn (what exists)

All in `ferrolearn-neighbors/src/local_outlier_factor.rs`, generic over
`F: Float (+ Send + Sync + 'static)`; `ndarray` substrate.

- **`pub struct LocalOutlierFactor<F> { pub n_neighbors: usize, pub contamination:
  f64, pub algorithm: Algorithm, pub novelty: bool, _marker }`** — the unfitted
  estimator. **No `leaf_size`/`metric`/`p`/`metric_params`/`n_jobs` fields.**
- **`pub fn new`** sets `n_neighbors=20`, **`contamination=0.1`** (sklearn default
  is `"auto"`), `algorithm=Auto`, `novelty=false`. Builder setters
  `with_n_neighbors`/`with_contamination`/`with_algorithm`/`with_novelty`;
  `impl Default → new()`.
- **`pub fn fit_predict(&self, x) -> Result<Array1<isize>>`** — calls `fit` then
  `predict(x)`. **No `novelty=true` `AttributeError` gating** (sklearn `:222-227`).
- **`pub struct FittedLocalOutlierFactor<F> { x_train, lof_scores: Vec<F>,
  n_neighbors, threshold: F }`** — stores **positive** LOF scores and a
  **LOF-space `threshold`** (not `negative_outlier_factor_`/`offset_`).
- **free `fn compute_lof_scores`** (the training pipeline): per-point brute-force
  k-NN excluding self (`knn_brute_force`), `effective_k = k.min(n-1)`;
  `k_dist = nn.last().1`; `lrd = nn.len() / sum(reach_dist)` where
  `reach_dist = k_dist[neighbor].max(dist)`; `lof = mean(lrd[neighbors]) / lrd[i]`.
  (Algebraically `nn.len()/sum == 1/mean`, i.e. sklearn's lrd **without** the
  `+1e-10`.)
- **`impl Fit<Array2<F>, ()> for LocalOutlierFactor<F>` / `fn fit`** — validates
  `n_neighbors != 0`, `contamination ∈ (0, 0.5]`, `n_samples >= 2`; computes
  `lof_scores`; derives `threshold` from contamination as `sorted_scores[n -
  n_outliers - 1]` with `n_outliers = ceil(contamination * n).max(1).min(n-1)` —
  a **rank index in positive-LOF space**, NOT `np.percentile` nor `offset_=-1.5`.
- **`impl Predict<Array2<F>> for FittedLocalOutlierFactor<F>` / `fn predict`** —
  `predictions[i] = if score <= threshold { 1 } else { -1 }` over
  `self.compute_lof(x)`. **No novelty gating.**
- **`fn compute_lof`** — if `x` is bit-exactly the training matrix, returns cached
  `lof_scores`; else recomputes LOF for new rows against the training set
  (brute-force, `effective_k = n_neighbors.min(n_train)`, self IS included among
  the train neighbors of each new query, consistent with sklearn's "new data"
  path).
- **`pub fn score_samples(&self, x) -> Result<Array1<F>>`** — `-lof` over
  `compute_lof(x)`. **No novelty gating.**
- **`pub fn decision_function(&self, x) -> Result<Array1<F>>`** —
  **`threshold - lof`** (LOF-space shift), NOT sklearn's `score_samples - offset_`.
  **No novelty gating.**
- **`pub fn lof_scores(&self) -> &[F]`** and **`pub fn threshold(&self) -> F`** —
  the only fitted accessors. **No `negative_outlier_factor_`, no `offset_`, no
  `n_neighbors_`, no `n_samples_fit_`.**

**Consumer (non-test).** Crate re-export only
(`ferrolearn-neighbors/src/lib.rs`:
`pub use local_outlier_factor::{FittedLocalOutlierFactor, LocalOutlierFactor}`,
and `pub mod local_outlier_factor`). Existing pub API, grandfathered (S5/R-DEFER-1).
No PyO3 binding, no meta-crate re-export (both verified absent by `grep`).

## Requirements

- REQ-1: **Defaults + parameter surface (R-DEV-2).** Mirror
  `LocalOutlierFactor(n_neighbors=20, *, algorithm="auto", leaf_size=30,
  metric="minkowski", p=2, metric_params=None, contamination="auto",
  novelty=False, n_jobs=None)` (`_lof.py:196-208`), **including the
  `contamination="auto"` default** (`:205`) and its `StrOptions({"auto"}) |
  Interval(Real, 0, 0.5, closed="right")` constraint (`:188-191`), plus
  `metric`/`p`/`leaf_size`.
- REQ-2: **LOF pipeline → `negative_outlier_factor_` (R-DEV-1/3).** Mirror the
  k-distance → reach_dist → lrd → LOF computation (`:486-511`, `:301-310`) and
  expose the negated result as `negative_outlier_factor_` (`:310`), value-matching
  the live sklearn oracle (incl. the **neighbor tie-break** sklearn uses and the
  `+1e-10` lrd damping, `:511`).
- REQ-3: **`offset_` (`'auto'`=-1.5 + percentile) (R-DEV-1/3).** Mirror
  `offset_ = -1.5` when `contamination=="auto"` else `np.percentile(
  negative_outlier_factor_, 100*contamination)` (`:312-318`), and expose `offset_`.
- REQ-4: **`fit_predict` / `predict` labels (R-DEV-1/3).** Mirror
  `is_inlier[negative_outlier_factor_ < offset_] = -1` for `fit_predict`/`_predict(None)`
  (`:380-381`) and `is_inlier[decision_function(X) < 0] = -1` for `predict(X)`
  (`:375-378`).
- REQ-5: **`score_samples` / `decision_function` (R-DEV-1/3).** Mirror
  `score_samples = -mean(lrd_train[nbrs]/X_lrd)` (`:481-484`) and
  `decision_function = score_samples - offset_` (`:424`), value-matching the oracle.
- REQ-6: **`n_neighbors >= n_samples` clamp + warning (R-DEV-1/2).** Mirror
  `n_neighbors_ = max(1, min(n_neighbors, n_samples-1))` (`:289`) AND the
  `warnings.warn(...)` (`:282-288`), exposing `n_neighbors_`.
- REQ-7: **Novelty gating (R-DEV-2).** Mirror the `available_if` gating:
  `fit_predict` unavailable (`AttributeError`) when `novelty=True` (`:222-227`);
  `predict`/`score_samples`/`decision_function` unavailable when `novelty=False`
  (`:322-331`, `:385-396`, `:426-436`).
- REQ-8: **PyO3 binding (R-DEFER-1/R-DEV-2).** `import ferrolearn` exposes
  `LocalOutlierFactor` mirroring `import sklearn`, including the
  `negative_outlier_factor_`/`offset_`/`n_neighbors_` attributes and the
  novelty-gated method surface.
- REQ-9: **ferray substrate (R-SUBSTRATE).** `local_outlier_factor.rs` is on the
  ferray array substrate (`ferray-core`), not `ndarray` + `num-traits`.

## Acceptance criteria

All expected values are from the live sklearn 1.5.2 oracle (run from `/tmp`),
never literal-copied from ferrolearn (R-CHAR-3).

- AC-1 (REQ-1 defaults): `LocalOutlierFactor()` → `n_neighbors==20`,
  **`contamination=='auto'`**, `novelty==False`, `algorithm=='auto'`,
  `metric=='minkowski'`, `p==2`, `leaf_size==30`. ferrolearn `new()` has
  `n_neighbors==20`, `novelty==false`, `algorithm==Auto` but
  **`contamination==0.1`** (no `'auto'`) and **no `metric`/`p`/`leaf_size`** —
  REQ-1 FAILS.
- AC-2 (REQ-2 NOF value-match): on `X=[[-1.1],[0.2],[101.1],[0.3]]`,
  `LocalOutlierFactor(n_neighbors=2).fit(X).negative_outlier_factor_ ==
  [-0.98214286, -1.03703704, -73.36970899, -0.98214286]`. On the 9-point fixture
  `X=[[0,0],[0.1,0],[0,0.1],[0.1,0.1],[-0.1,0],[0,-0.1],[0.05,0.05],[-0.05,-0.05],
  [10,10]]`, `n_neighbors=5`: sklearn `-nof = [1.03689756, 1.00298925, 1.00298925,
  1.04249127, 0.96923375, 0.96923375, 1.00298925, 0.96923375, 92.73011569]`;
  ferrolearn `lof_scores() = [1.00378346, 1.00747336, 1.00747336, 1.04712262,
  0.97349562, 0.97349562, 1.00747336, 0.97349562, 93.14904476]` — diverges at
  ~1e-2 (neighbor **tie-break** on the equidistant ring points) and ferrolearn has
  no `negative_outlier_factor_` accessor. REQ-2 FAILS.
- AC-3 (REQ-3 offset_ 'auto'): `LocalOutlierFactor(n_neighbors=2).fit(X).offset_
  == -1.5` (contamination='auto'). With a float, `contamination=0.1` →
  `offset_ == np.percentile(negative_outlier_factor_, 10.0)`
  (e.g. on `X=[[-1.1],[0.2],[101.1],[0.3]]` → `-51.66990740`). ferrolearn has **no
  `offset_`**; it stores a **positive-LOF-space `threshold`** computed by a rank
  index `sorted[n - n_outliers - 1]`, not `np.percentile` in NOF space, and has no
  `'auto'` (-1.5) branch. REQ-3 FAILS.
- AC-4 (REQ-4 labels): on the 9-point fixture `n_neighbors=5`, sklearn
  `fit_predict` (contamination='auto') `== [1,1,1,1,1,1,1,1,-1]` and
  (contamination=0.15) `== [1,1,1,-1,1,1,1,1,-1]`. ferrolearn `fit_predict`
  default(0.1) `== [1,1,1,1,1,1,1,1,-1]` — coincidentally matches the auto case
  here, but via a different mechanism (rank-threshold vs `offset_=-1.5`); the label
  rule and threshold value are not the sklearn contract. REQ-4 FAILS (mechanism +
  contamination='auto' parity).
- AC-5 (REQ-5 score_samples/decision_function, novelty=True): train on
  `X=[[0,0],[0.1,0],[0,0.1],[0.1,0.1],[-0.1,0],[0,-0.1],[0.05,0.05],[-0.05,-0.05]]`,
  `n_neighbors=5`; on `Xt=[[100,100],[0.02,0.02]]`, sklearn
  `score_samples == [-931.49032345, -0.91457584]`,
  `decision_function == [-929.99032345, 0.58542416]`. ferrolearn
  `score_samples == [-935.69853974, -0.91863892]` (diverges, tie-break) and
  `decision_function == [-934.69106637, 0.08883445]` (sklearn subtracts
  `offset_=-1.5`; ferrolearn subtracts the LOF-space `threshold` → wrong shift).
  REQ-5 FAILS.
- AC-6 (REQ-6 clamp + warning): on `X` with 4 samples,
  `LocalOutlierFactor(n_neighbors=20).fit(X)` → sklearn warns
  `"n_neighbors (20) is greater than the total number of samples (4) ..."` and sets
  `n_neighbors_ == 3`. ferrolearn clamps `effective_k = k.min(n-1) == 3`
  internally but **emits no warning and exposes no `n_neighbors_`**. REQ-6 FAILS
  on the warning + attribute.
- AC-7 (REQ-7 novelty gating): `hasattr(LocalOutlierFactor(novelty=False),
  'predict') == False`, `hasattr(..., 'score_samples') == False`,
  `hasattr(..., 'decision_function') == False`;
  `hasattr(LocalOutlierFactor(novelty=True), 'fit_predict') == False`. ferrolearn
  exposes all four methods regardless of `novelty` (the field is documented as
  "informational" — `with_novelty` doc). REQ-7 FAILS.

## REQ status table

Binary (R-DEFER-2). `LocalOutlierFactor`/`FittedLocalOutlierFactor` are existing
pub APIs re-exported at the crate root (the non-test production-consumer surface;
grandfathered S5/R-DEFER-1). Cites use symbol anchors (ferrolearn) / `file:line`
(sklearn 1.5.2). Live oracle = installed sklearn 1.5.2, run from `/tmp`. Honest
underclaim (R-HONEST-3).

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (defaults + param surface) | NOT-STARTED | open prereq blocker #845. `pub fn new` sets **`contamination: 0.1`** but sklearn's default is `"auto"` (`_lof.py:205`; constraint `StrOptions({"auto"}) | Interval(Real,0,0.5,closed="right")` `:188-191`). `pub struct LocalOutlierFactor` has no `metric`/`p`/`leaf_size`/`metric_params`/`n_jobs` fields (`__init__` `:196-208`) and `contamination` is `f64` (cannot represent `"auto"`). Pin (AC-1): `LocalOutlierFactor().contamination == "auto"`; ferro `0.1`. |
| REQ-2 (LOF pipeline → negative_outlier_factor_) | NOT-STARTED | open prereq blocker #846. `fn compute_lof_scores` produces positive `lof_scores` that do **not** value-match sklearn's `-negative_outlier_factor_` (`:310`): on the 9-point fixture ferro `[1.00378346,...,93.14904476]` vs sklearn `[1.03689756,...,92.73011569]` (~1e-2, neighbor **tie-break** divergence in `knn_brute_force` vs sklearn `kneighbors`; sklearn also adds `+1e-10` to the lrd denominator, `:511`). No `negative_outlier_factor_` accessor exists (only `fn lof_scores` returns the un-negated LOF). Pin (AC-2). |
| REQ-3 (offset_ auto=-1.5 + percentile) | NOT-STARTED | open prereq blocker #847. `fn fit` computes a positive-LOF-space `threshold = sorted_scores[n - n_outliers - 1]` (rank index), not sklearn's `offset_` (`:312-318`): no `contamination=="auto" → -1.5` branch, and the float path is a rank index, not `np.percentile(negative_outlier_factor_, 100*contamination)` in NOF space. No `offset_` accessor. Pin (AC-3): `offset_==-1.5` (auto); `offset_==np.percentile(nof,10)` for `contamination=0.1`. |
| REQ-4 (fit_predict / predict labels) | NOT-STARTED | open prereq blocker #848. `fn predict` / `fn fit_predict` apply `score <= threshold ? 1 : -1` in positive-LOF space, not `negative_outlier_factor_ < offset_` (`:380-381`) / `decision_function(X) < 0` (`:375-378`). Because `offset_` (REQ-3) and the NOF pipeline (REQ-2) diverge, labels are correct only by coincidence on easy fixtures (AC-4: matches the auto case on the 9-point set but via the wrong mechanism). Pin (AC-4). |
| REQ-5 (score_samples / decision_function) | NOT-STARTED | open prereq blocker #849. `fn score_samples` returns `-lof` and `fn decision_function` returns **`threshold - lof`** (LOF-space shift), not sklearn `score_samples - offset_` (`:424`). On `Xt=[[100,100],[0.02,0.02]]` (novelty-style) sklearn `decision_function==[-929.99,0.5854]`; ferro `[-934.69,0.0888]` — both the score-value (tie-break, REQ-2) and the offset shift (REQ-3) diverge. Pin (AC-5). |
| REQ-6 (n_neighbors clamp + warning) | NOT-STARTED | open prereq blocker #850. `compute_lof_scores`/`compute_lof` clamp `effective_k = k.min(n-1)` (matching sklearn `min(n_neighbors, n_samples-1)` `:289`) but emit **no `warnings.warn`** (`:282-288`) and expose **no `n_neighbors_`** attribute (`:130-131`, `:289`). Pin (AC-6): 4-sample fit with `n_neighbors=20` → sklearn warns + `n_neighbors_==3`; ferro silent, no attribute. |
| REQ-7 (novelty gating) | NOT-STARTED | open prereq blocker #851. `fn predict`/`fn score_samples`/`fn decision_function`/`fn fit_predict` are callable regardless of `novelty` (the field is "informational" per `with_novelty` doc). sklearn gates each via `available_if`: `fit_predict` raises `AttributeError` when `novelty=True` (`:222-227`); `predict`/`score_samples`/`decision_function` raise when `novelty=False` (`:322-331`,`:385-396`,`:426-436`). Pin (AC-7): `hasattr(LocalOutlierFactor(novelty=False),'predict')==False`. |
| REQ-8 (PyO3 binding) | NOT-STARTED | open prereq blocker #852. No `RsLocalOutlierFactor` (or equivalent) exists in `ferrolearn-python/src/` and no meta-crate re-export in `ferrolearn/src/` (both verified absent by `grep`). `import ferrolearn` cannot construct `LocalOutlierFactor` nor read `negative_outlier_factor_`/`offset_`. NOT-STARTED until the library REQs land and the shim exposes the gated method surface + NOF/offset attributes. |
| REQ-9 (ferray substrate) | NOT-STARTED | open prereq blocker #853. `local_outlier_factor.rs` imports `ndarray::{Array1, Array2}` + `num_traits::Float` (the wrong substrate); not migrated to `ferray-core` (R-SUBSTRATE-1/2). |

## Architecture

`local_outlier_factor.rs` follows the unfitted/fitted split (CLAUDE.md naming):
`LocalOutlierFactor<F>` (params) → `Fit` → `FittedLocalOutlierFactor<F>` (the
learned `lof_scores: Vec<F>` + `threshold: F` + cached `x_train`). Generic over
`F: Float + Send + Sync + 'static`; all public methods return
`Result<_, FerroError>` (no panics in library code).

**Fit path (`fn fit`).** Validation (`n_neighbors != 0`, `contamination ∈ (0,
0.5]`, `n_samples >= 2` → `InvalidParameter`/`InsufficientSamples`). The data is
materialized as `Vec<Vec<F>>` and `compute_lof_scores` runs the four-step LOF:
(1) per-point brute-force k-NN excluding self (`knn_brute_force`, `effective_k =
k.min(n-1)`); (2) `k_dist = nn.last()`; (3) `lrd = nn.len() / Σ reach_dist`,
`reach_dist = k_dist[neighbor].max(dist)`; (4) `lof = mean(lrd[neighbors]) /
lrd[i]`. The result is the **positive** LOF (sklearn's `-negative_outlier_factor_`).

The divergences vs sklearn live here:
- **Tie-break (REQ-2).** `knn_brute_force` sorts `(idx, dist)` by `partial_cmp` on
  distance only; with equidistant neighbors the surviving k are chosen by
  collection (index) order, which differs from sklearn's `kneighbors` ordering — on
  the equidistant-ring fixture this shifts the lrd average by ~1e-2.
- **lrd damping (REQ-2).** `nn.len()/Σ reach_dist == 1/mean(reach_dist)`; sklearn
  uses `1/(mean(reach_dist) + 1e-10)` (`:511`). ferrolearn instead guards a
  near-zero sum with a `1e10` sentinel and a `1e-15` epsilon — a different
  duplicate/NaN-avoidance scheme.
- **Threshold (REQ-3).** `fn fit` derives `threshold = sorted_scores[n -
  n_outliers - 1]` with `n_outliers = ceil(contamination*n).max(1).min(n-1)` — a
  rank index in **positive-LOF** space. sklearn computes `offset_` in
  **negative-NOF** space: `-1.5` for `"auto"`, else `np.percentile(nof,
  100*contamination)` (`:312-318`). ferrolearn has neither the `"auto"` branch nor
  `np.percentile`.

**Predict / score path.** `fn compute_lof` short-circuits to the cached
`lof_scores` when `x` is bit-exactly `x_train` (the `fit_predict`/training-eval
case); otherwise it recomputes LOF for the new rows against the stored training
set (brute-force, self included among the train neighbors — consistent with
sklearn's "new data, point in its own neighborhood" semantics for `score_samples`,
`:447-449`). `fn predict` thresholds in LOF space (`score <= threshold`); `fn
score_samples` returns `-lof`; `fn decision_function` returns `threshold - lof`
(NOT sklearn's `score_samples - offset_`, `:424`). **None of these are
novelty-gated** (REQ-7), whereas sklearn makes `predict`/`score_samples`/
`decision_function` available only under `novelty=True` and `fit_predict` only
under `novelty=False`.

**Missing fitted attributes vs sklearn (`Attributes`, `:117-162`):**
`negative_outlier_factor_` (`:119`, REQ-2), `offset_` (`:133`, REQ-3),
`n_neighbors_` (`:130`, REQ-6), `effective_metric_`/`effective_metric_params_`
(`:144-148`), `n_features_in_`/`feature_names_in_` (`:150-159`),
`n_samples_fit_` (`:161`). ferrolearn exposes only `lof_scores()` (the un-negated
LOF) and `threshold()` (a LOF-space rank threshold).

## Verification

Library crate (green at baseline `a0cd459f` for the existing — narrower, NOT
sklearn-value-matching — contract):
```
cargo test -p ferrolearn-neighbors --lib local_outlier_factor
cargo clippy -p ferrolearn-neighbors --all-targets -- -D warnings
cargo fmt --all --check
```
The existing `#[test]`s pin ferrolearn's CURRENT behavior — `test_lof_default`
pins `contamination==0.1` (the divergent default), `test_lof_outlier_has_high_score`
pins only the argmax (LOF shape), `test_lof_predict_training_data` pins
`contamination=0.15` labels. **None compares against the live sklearn oracle**, so
they establish no REQ as SHIPPED. The conformance fixture
(`ferrolearn-neighbors/tests/conformance_wave4.rs`
`conformance_local_outlier_factor`) is the pre-existing in-repo suite — **not the
contract here** (goal.md §"The verification model").

Live sklearn oracle (installed 1.5.2, run from `/tmp`) — the divergences a critic
should pin first (R-CHAR-3 expected values). **Pin the deterministic ones first**
(no tie-break ambiguity): `contamination='auto'` default + `offset_=-1.5`, the
missing `negative_outlier_factor_`/`offset_` accessors, the novelty gating, the
`decision_function` sign convention:
```
# REQ-1 (defaults — contamination='auto'):
python3 -c "from sklearn.neighbors import LocalOutlierFactor as L; c=L(); print(c.n_neighbors, c.contamination, c.novelty, c.metric, c.p, c.leaf_size)"
#   -> 20 auto False minkowski 2 30   (ferro: 20, 0.1, false; no metric/p/leaf_size)
# REQ-2 (NOF value-match):
python3 -c "
import numpy as np; from sklearn.neighbors import LocalOutlierFactor as L
X=np.array([[-1.1],[0.2],[101.1],[0.3]])
print(L(n_neighbors=2).fit(X).negative_outlier_factor_.tolist())"
#   -> [-0.98214286, -1.03703704, -73.36970899, -0.98214286]
python3 -c "
import numpy as np; from sklearn.neighbors import LocalOutlierFactor as L
X=np.array([[0,0],[0.1,0],[0,0.1],[0.1,0.1],[-0.1,0],[0,-0.1],[0.05,0.05],[-0.05,-0.05],[10,10]],dtype=float)
print((-L(n_neighbors=5).fit(X).negative_outlier_factor_).tolist())"
#   -> [1.03689756,1.00298925,1.00298925,1.04249127,0.96923375,0.96923375,1.00298925,0.96923375,92.73011569]
#      (ferro lof_scores: [1.00378346,1.00747336,...,93.14904476] — tie-break divergence)
# REQ-3 (offset_ auto + percentile):
python3 -c "
import numpy as np; from sklearn.neighbors import LocalOutlierFactor as L
X=np.array([[-1.1],[0.2],[101.1],[0.3]])
print(L(n_neighbors=2).fit(X).offset_, L(n_neighbors=2,contamination=0.1).fit(X).offset_)"
#   -> -1.5   -51.66990740   (ferro: positive-LOF rank threshold, no offset_)
# REQ-4 (labels auto vs float):
python3 -c "
import numpy as np; from sklearn.neighbors import LocalOutlierFactor as L
X=np.array([[0,0],[0.1,0],[0,0.1],[0.1,0.1],[-0.1,0],[0,-0.1],[0.05,0.05],[-0.05,-0.05],[10,10]],dtype=float)
print(L(n_neighbors=5).fit_predict(X).tolist(), L(n_neighbors=5,contamination=0.15).fit_predict(X).tolist())"
#   -> [1,1,1,1,1,1,1,1,-1]   [1,1,1,-1,1,1,1,1,-1]
# REQ-5 (score_samples / decision_function, novelty=True):
python3 -c "
import numpy as np; from sklearn.neighbors import LocalOutlierFactor as L
X=np.array([[0,0],[0.1,0],[0,0.1],[0.1,0.1],[-0.1,0],[0,-0.1],[0.05,0.05],[-0.05,-0.05]],dtype=float)
c=L(n_neighbors=5,novelty=True).fit(X); Xt=np.array([[100,100],[0.02,0.02]],dtype=float)
print(c.score_samples(Xt).tolist(), c.decision_function(Xt).tolist())"
#   -> [-931.49032345,-0.91457584]   [-929.99032345,0.58542416]
#      (ferro: [-935.69853974,-0.91863892]  [-934.69106637,0.08883445])
# REQ-6 (n_neighbors clamp + warning):
python3 -c "
import numpy as np, warnings; from sklearn.neighbors import LocalOutlierFactor as L
X=np.array([[0,0],[0.1,0],[0,0.1],[0.1,0.1]],dtype=float)
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter('always'); c=L(n_neighbors=20).fit(X)
    print(c.n_neighbors_, [str(x.message)[:50] for x in w])"
#   -> 3 ['n_neighbors (20) is greater than the total number ...']  (ferro: clamps, no warn, no n_neighbors_)
# REQ-7 (novelty gating):
python3 -c "from sklearn.neighbors import LocalOutlierFactor as L
print(hasattr(L(novelty=False),'predict'), hasattr(L(novelty=False),'score_samples'), hasattr(L(novelty=True),'fit_predict'))"
#   -> False False False   (ferro: all methods callable regardless of novelty)
```
A characterization pin (R-CHAR-3) for each NOT-STARTED REQ belongs in
`ferrolearn-neighbors/tests/divergence_local_outlier_factor.rs`, asserting the
live-sklearn expected values above and FAILING against current
`local_outlier_factor.rs`.

ferrolearn-python (REQ-8 binding parity, after the library REQs land):
```
cd /home/doll/ferrolearn/ferrolearn-python && maturin develop
PYTHONPATH=python python3 -m pytest tests/divergence_neighbors.py -q
```
asserting `ferrolearn.LocalOutlierFactor` matches `sklearn.neighbors.
LocalOutlierFactor` on `contamination`/`novelty`, `negative_outlier_factor_`,
`offset_`, `n_neighbors_`, the gated method surface, and `fit_predict`/
`decision_function` outputs.

## Blockers to open

(Director creates the real issues; numbers below are SUGGESTIONS continuing the
neighbors layer from the nearest-centroid block #801-#807.)

- #845 — REQ-1 (defaults + param surface): `contamination` default is `0.1`, not
  `"auto"` (sklearn `_lof.py:205`; constraint `:188-191`); `contamination` is
  `f64` and cannot represent `"auto"`; no `metric`/`p`/`leaf_size`/`metric_params`/
  `n_jobs` fields (`:196-208`).
- #846 — REQ-2 (LOF pipeline → negative_outlier_factor_): `lof_scores` do not
  value-match sklearn `-negative_outlier_factor_` (`:310`) — neighbor **tie-break**
  divergence (`knn_brute_force` vs `kneighbors`) ~1e-2 on equidistant points; lrd
  lacks the `+1e-10` damping (`:511`); no `negative_outlier_factor_` accessor.
  Pin: 4-point `[-0.98214286,-1.03703704,-73.36970899,-0.98214286]`.
- #847 — REQ-3 (offset_): no `contamination=="auto" → -1.5` branch; float path is
  a positive-LOF rank index, not `np.percentile(negative_outlier_factor_,
  100*contamination)` in NOF space (`:312-318`); no `offset_` accessor.
- #848 — REQ-4 (fit_predict/predict labels): thresholds in positive-LOF space
  (`score <= threshold`), not `negative_outlier_factor_ < offset_` (`:380-381`) /
  `decision_function(X) < 0` (`:375-378`); blocked on #846/#847.
- #849 — REQ-5 (score_samples/decision_function): `decision_function` returns
  `threshold - lof` not `score_samples - offset_` (`:424`); score values diverge
  (tie-break, #846). Pin: novelty=True `decision_function([[100,100],[0.02,0.02]])
  == [-929.99032345, 0.58542416]`.
- #850 — REQ-6 (n_neighbors clamp + warning): clamps but emits no
  `warnings.warn` (`:282-288`) and exposes no `n_neighbors_` (`:130-131`,`:289`).
- #851 — REQ-7 (novelty gating): `predict`/`score_samples`/`decision_function`/
  `fit_predict` callable regardless of `novelty`; sklearn gates via `available_if`
  (`:221-227`,`:322-331`,`:385-396`,`:426-436`). Pin:
  `hasattr(LocalOutlierFactor(novelty=False),'predict')==False`.
- #852 — REQ-8 (PyO3 binding): no `RsLocalOutlierFactor` in `ferrolearn-python`
  and no meta-crate re-export; expose the estimator + NOF/offset/n_neighbors_
  attributes + gated methods at sklearn parity once #845-#851 land.
- #853 — REQ-9 (ferray substrate): migrate `local_outlier_factor.rs` off
  `ndarray`/`num-traits` to `ferray-core` (R-SUBSTRATE).
