# Nearest Centroid Classifier (sklearn.neighbors.NearestCentroid)

<!--
tier: 3-component
status: draft
baseline-commit: 2013942c
upstream: scikit-learn 1.5.2 (commit 156ef14)
upstream-paths:
  - sklearn/neighbors/_nearest_centroid.py   # NearestCentroid class (:24); _parameter_constraints (:103); __init__ (:108); fit (:113); manhattan/euclidean centroid (:164-171); n_classes<2 ValueError (:147-151); shrink_threshold block (:173-196); s += median(s) (:184); zero-variance ValueError (:174-175); predict (:199)
ferrolearn-module: ferrolearn-neighbors/src/nearest_centroid.rs
parity-ops: NearestCentroid
crosslink-issue: PLACEHOLDER  # NOTE — replace with the nearest-centroid unit's own tracking issue when the director assigns it (the ferrolearn-neighbors layer block).
-->

## Summary

`ferrolearn-neighbors/src/nearest_centroid.rs` mirrors scikit-learn's
`sklearn.neighbors.NearestCentroid` (`sklearn/neighbors/_nearest_centroid.py`):
the nearest-centroid classifier that represents each class by a centroid computed
at `fit` time and classifies a test sample to the class whose centroid is nearest.
It provides the unfitted `NearestCentroid<F>` and fitted `FittedNearestCentroid<F>`
types, with `classes_`/`centroids_` introspection and an optional
`shrink_threshold` shrinkage path.

Under honest underclaim (R-HONEST-3), the **euclidean default fit/predict path,
the per-class mean centroid, and the `classes_`/`centroids_` attributes match the
live sklearn 1.5.2 oracle** and are SHIPPED. But several behaviors **diverge from
sklearn and are NOT-STARTED**:

1. **No `metric` constructor parameter at all.** sklearn's
   `NearestCentroid(metric="euclidean", *, shrink_threshold=None)` (`:108`)
   accepts `metric ∈ {"euclidean","manhattan"}` (`:104`); ferrolearn's `new()`
   exposes only `shrink_threshold` and hard-codes euclidean. `metric="manhattan"`
   (feature-wise **median** centroid, `:164-169`) is entirely absent.
2. **`shrink_threshold` shrinkage uses the wrong `s`.** sklearn adds the median of
   the pooled within-class std to itself — `s += np.median(s)` (`:184`) — before
   forming the deviation denominator; ferrolearn omits the `+ median(s)` term, so
   the shrunken `centroids_` differ numerically (oracle below).
3. **No `n_classes < 2` guard.** sklearn raises `ValueError` (`:147-151`) when a
   single class is seen; ferrolearn fits a single-class model successfully.
4. **`shrink_threshold` accepts `0` and zero-variance input.** sklearn's
   constraint is `Interval(Real, 0, None, closed="neither")` (`:105`), i.e. it
   requires `> 0` and raises on `0`; ferrolearn accepts `0`. sklearn also raises
   `ValueError("All features have zero variance. ...")` (`:174-175`) when all
   features are constant; ferrolearn substitutes `1.0` instead.

`NearestCentroid`/`FittedNearestCentroid` are existing pub APIs re-exported at the
crate root (`ferrolearn-neighbors/src/lib.rs`) and exposed to CPython by
`ferrolearn-python` (`extras.rs` `RsNearestCentroid`, registered in `lib.rs`).
Those are the non-test production consumers; they are grandfathered under
S5/R-DEFER-1.

## Algorithm (sklearn — the contract)

`NearestCentroid(metric="euclidean", *, shrink_threshold=None)` (`__init__`, `:108`).
`_parameter_constraints` (`:103-106`): `metric ∈ StrOptions({"manhattan","euclidean"})`;
`shrink_threshold ∈ Interval(Real, 0, None, closed="neither")` **or `None`** (so a
finite threshold must be strictly positive).

**`fit(X, y)`** (`:113`):
1. Validate; `manhattan` accepts only `csc` sparse, else `csr`/`csc` (`:133-136`).
   `shrink_threshold` with sparse `X` raises `ValueError` (`:138-139`).
2. `check_classification_targets(y)`; `LabelEncoder` → `classes_ = le.classes_`,
   `y_ind` the encoded indices (`:143-145`). **`n_classes < 2` raises `ValueError`**
   (`:147-151`).
3. Per class `cur_class` (`:158-171`): `center_mask = y_ind == cur_class`,
   `nk[cur_class] = sum(center_mask)`.
   - `metric == "manhattan"`: `centroids_[cur_class] = np.median(X[mask], axis=0)`
     (feature-wise **median**, `:167`).
   - else euclidean: `centroids_[cur_class] = X[mask].mean(axis=0)` (feature-wise
     **mean**, `:171`).
4. **Shrinkage** (`if self.shrink_threshold:`, `:173-196`):
   - If `np.all(np.ptp(X, axis=0) == 0)` raise `ValueError("All features have zero
     variance. Division by zero.")` (`:174-175`).
   - `dataset_centroid_ = np.mean(X, axis=0)` (`:176`).
   - `m = np.sqrt(1/nk - 1/n_samples)` (`:179`).
   - `variance = ((X - centroids_[y_ind])**2).sum(axis=0)` (`:181-182`),
     `s = np.sqrt(variance / (n_samples - n_classes))` (`:183`),
     **`s += np.median(s)`** (`:184`) — the median-of-s damping term.
   - `ms = m.reshape(-1,1) * s` (`:185-186`),
     `deviation = (centroids_ - dataset_centroid_) / ms` (`:187`).
   - Soft-threshold: `signs = sign(deviation)`;
     `deviation = clip(|deviation| - shrink_threshold, 0, None) * signs`
     (`:190-193`); `centroids_ = dataset_centroid_ + ms * deviation` (`:195-196`).

**`predict(X)`** (`:199`): `classes_[pairwise_distances_argmin(X, centroids_,
metric=self.metric)]` (`:217-219`) — nearest centroid under the configured metric.

`centroids_` has shape `(n_classes, n_features)` and dtype `float64`; `classes_`
is the sorted unique label array.

## ferrolearn (what exists)

All in `ferrolearn-neighbors/src/nearest_centroid.rs`, generic over
`F: Float + Send + Sync + 'static`.

- **`pub struct NearestCentroid<F> { pub shrink_threshold: Option<F> }`** — the
  unfitted estimator. **It has no `metric` field.**
- **`pub fn new() -> Self`** sets `shrink_threshold: None` (matches sklearn's
  euclidean/`None` default for the params it does expose).
- **`pub fn with_shrink_threshold(self, threshold) -> Self`** — builder setter.
- **`impl Default`** → `new()`.
- **`pub struct FittedNearestCentroid<F> { centroids: Array2<F>, classes: Vec<usize> }`**.
- **`impl Fit<Array2<F>, Array1<usize>> for NearestCentroid<F>` / `fn fit`** —
  validates non-empty (`InsufficientSamples`) and matching row count
  (`ShapeMismatch`); rejects negative `shrink_threshold` (`InvalidParameter`);
  collects sorted-unique `classes`; computes per-class **mean** centroids
  (euclidean only); applies the shrinkage block when `shrink_threshold` is set.
- **`impl Predict<Array2<F>> for FittedNearestCentroid<F>` / `fn predict`** —
  checks feature count (`ShapeMismatch`), then assigns each sample to the class
  with the smallest **squared euclidean** distance to its centroid (argmin;
  first-wins tie-break via strict `<`).
- **`impl HasClasses` / `fn classes`, `fn n_classes`** — the `classes_` analog.
- **`fn centroids(&self) -> &Array2<F>`** — the `centroids_` analog.
- **`fn score(&self, x, y)`** — mean accuracy (sklearn `ClassifierMixin.score`).

**Shrinkage divergence (in `fn fit`).** ferrolearn computes
`pooled_var[j] = sqrt(sum_within_class (x-mean)^2 / max(n_samples - n_classes, 1))`,
then **clamps `pooled_var[j] < 1e-10` to `1.0`** and uses it directly as the
denominator scale (`m_k * pooled_var[j]`). sklearn instead does
`s = sqrt(variance / (n_samples - n_classes))` then **`s += np.median(s)`**
(`:184`) and has **no `< 1e-10 → 1.0` clamp** — it raises on globally-constant `X`
(`:174-175`). The missing median-of-s term changes every shrunken centroid value.

**Consumers (non-test).** Crate re-export
(`ferrolearn-neighbors/src/lib.rs`: `pub use nearest_centroid::{FittedNearestCentroid,
NearestCentroid}`) and the PyO3 binding (`ferrolearn-python/src/extras.rs`:
`RsNearestCentroid` wrapping `FittedNearestCentroid<f64>`, registered in
`ferrolearn-python/src/lib.rs` via `m.add_class::<extras::RsNearestCentroid>()`).
Both are existing pub APIs (grandfathered, S5/R-DEFER-1).

## Requirements

- REQ-1: **Defaults + parameter surface (R-DEV-2).** Mirror
  `NearestCentroid(metric="euclidean", *, shrink_threshold=None)` (`:108`) incl.
  the `metric` constructor parameter (`StrOptions({"manhattan","euclidean"})`,
  `:104`) and the `shrink_threshold ∈ (0, None)` strictly-positive constraint
  (`:105`), with the euclidean/`None` defaults.
- REQ-2: **Euclidean centroid (per-class mean) + nearest-centroid predict
  (R-DEV-1).** Mirror `X[mask].mean(axis=0)` (`:171`) and
  `pairwise_distances_argmin(X, centroids_)` (`:217-219`) under euclidean.
- REQ-3: **`shrink_threshold` shrinkage (R-DEV-1).** Mirror the shrunken-centroid
  formula incl. **`s += np.median(s)`** (`:184`), the `(0, None)` strictly-positive
  constraint (`:105`), and the **zero-variance `ValueError`** (`:174-175`).
- REQ-4: **`metric="manhattan"` (median centroid, R-DEV-1/2).** Mirror the
  feature-wise median centroid (`np.median(X[mask], axis=0)`, `:167`) and
  manhattan-metric `predict` (`:218`).
- REQ-5: **`classes_` / `centroids_` attributes (R-DEV-3).** Mirror `classes_`
  (sorted unique labels, `:145`) and `centroids_` (shape `(n_classes,
  n_features)`, `:154`).
- REQ-6: **`n_classes < 2` guard (R-DEV-1/2).** Mirror the `ValueError` raised
  when fewer than two classes are seen (`:147-151`).
- REQ-7: **PyO3 binding (R-DEFER-1/R-DEV-2).** `import ferrolearn` exposes
  `NearestCentroid` mirroring `import sklearn`, including the `metric` /
  `shrink_threshold` constructor params and the `classes_`/`centroids_` attributes.
- REQ-8: **ferray substrate (R-SUBSTRATE).** `nearest_centroid.rs` is on the
  ferray array substrate (`ferray-core`), not `ndarray` + `num-traits`.

## Acceptance criteria

All expected values are from the live sklearn 1.5.2 oracle (run from `/tmp`),
never literal-copied from ferrolearn (R-CHAR-3).

- AC-1 (REQ-1 defaults): `NearestCentroid()` → `metric == "euclidean"`,
  `shrink_threshold is None`. ferrolearn `new()` matches on `shrink_threshold`
  but **has no `metric`** — REQ-1 fails on the missing parameter.
- AC-2 (REQ-2 centroids+predict): on
  `X=[[0,0],[0.5,0],[0,0.5],[0.5,0.5],[5,5],[5.5,5],[5,5.5],[5.5,5.5]]`,
  `y=[0,0,0,0,1,1,1,1]`, sklearn `centroids_ == [[0.25,0.25],[5.25,5.25]]` and
  `predict(X) == [0,0,0,0,1,1,1,1]`. ferrolearn matches both (already pinned by
  `test_nearest_centroid_centroids`/`test_nearest_centroid_fit_predict`).
- AC-3 (REQ-3 shrink formula): on the AC-2 data,
  `NearestCentroid(shrink_threshold=0.5).fit(X,y).centroids_` ==
  **`[[0.3520620726159658, 0.3520620726159658], [5.147937927384034,
  5.147937927384034]]`** (sklearn, with `s += median(s)`). ferrolearn's
  median-add-omitting formula produces
  **`[[0.30103..., 0.30103...], [5.19897..., 5.19897...]]`** and FAILS.
- AC-4 (REQ-3 constraint + zero-variance): `NearestCentroid(shrink_threshold=0)`
  raises `InvalidParameterError` (constraint `> 0`); ferrolearn accepts `0`.
  `NearestCentroid(shrink_threshold=0.5).fit([[1,1],[1,1],[1,1],[1,1]],[0,0,1,1])`
  raises `ValueError("All features have zero variance...")`; ferrolearn returns
  `Ok` (it clamps the zero std to `1.0`).
- AC-5 (REQ-4 manhattan median): on
  `X=[[0,0],[1,0],[10,0],[5,5],[5.5,5],[100,5]]`, `y=[0,0,0,1,1,1]`,
  `NearestCentroid(metric="manhattan").fit(X,y).centroids_` ==
  **`[[1.0, 0.0], [5.5, 5.0]]`** (feature-wise median), distinct from the
  euclidean mean **`[[3.6666..., 0.0], [36.8333..., 5.0]]`**. ferrolearn cannot
  express `metric="manhattan"` — REQ-4 inexpressible.
- AC-6 (REQ-5 attributes): on the AC-2 data, `classes_ == [0,1]` and
  `centroids_.shape == (2,2)`. ferrolearn `classes() == [0,1]`,
  `centroids().dim() == (2,2)` (pinned by `test_nearest_centroid_has_classes`).
  Non-contiguous labels `y=[10,10,10,20,20,20]` → sklearn `classes_ == [10,20]`;
  ferrolearn matches (`test_nearest_centroid_noncontiguous_labels`).
- AC-7 (REQ-6 single-class guard): `NearestCentroid().fit([[1,1],[1.5,1],[1,1.5]],
  [5,5,5])` raises `ValueError("The number of classes has to be greater than one;
  got 1 class")`. ferrolearn returns `Ok` (single-class fit) and FAILS.

## REQ status table

Binary (R-DEFER-2). `NearestCentroid`/`FittedNearestCentroid` are existing pub
APIs re-exported at the crate root and bound by `ferrolearn-python` (the non-test
production-consumer surface; grandfathered S5/R-DEFER-1). Cites use symbol anchors
(ferrolearn) / `file:line` (sklearn 1.5.2). Live oracle = installed sklearn 1.5.2,
run from `/tmp`. Honest underclaim (R-HONEST-3).

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (defaults + param surface) | NOT-STARTED | open prereq blocker #801. `pub struct NearestCentroid { pub shrink_threshold: Option<F> }` + `pub fn new` match sklearn's `shrink_threshold=None` default but **expose no `metric` parameter** (sklearn `__init__` `_nearest_centroid.py:108`; constraint `StrOptions({"manhattan","euclidean"})` `:104`). Also no strictly-positive `(0, None)` `shrink_threshold` constraint (`:105`). Pin: `NearestCentroid()` → sklearn `metric=="euclidean"`; ferro has no `metric`. |
| REQ-2 (euclidean centroid + predict) | SHIPPED | impl `fn fit in nearest_centroid.rs` computes per-class mean (`centroids[[ci,j]] = sum/n_c_f`) mirroring `X[mask].mean(axis=0)` (sklearn `_nearest_centroid.py:171`); `fn predict` assigns nearest squared-euclidean centroid mirroring `pairwise_distances_argmin(X, centroids_)` (`:217-219`). Non-test consumer: crate re-export `lib.rs` (`pub use nearest_centroid::{FittedNearestCentroid, NearestCentroid}`) + PyO3 `RsNearestCentroid` in `ferrolearn-python/src/extras.rs`. Oracle (AC-2): `centroids_==[[0.25,0.25],[5.25,5.25]]`, `predict==[0,0,0,0,1,1,1,1]` — ferro matches; tests `test_nearest_centroid_centroids`, `test_nearest_centroid_fit_predict`. |
| REQ-3 (shrink_threshold shrinkage) | NOT-STARTED | open prereq blocker #802. `fn fit` shrinkage omits sklearn's **`s += np.median(s)`** (`_nearest_centroid.py:184`): ferro denominator is `m_k * pooled_var[j]` with no median-of-s term, so shrunken `centroids_` diverge. Pin (AC-3): `shrink_threshold=0.5` → sklearn `[[0.35206...,0.35206...],[5.14794...,5.14794...]]`, ferro `[[0.30103...,...],[5.19897...,...]]`. Also accepts `shrink_threshold=0` (sklearn constraint `>0`, `:105`) and clamps zero-variance to `1.0` instead of raising `ValueError` (`:174-175`). |
| REQ-4 (metric="manhattan" median) | SHIPPED | `NearestCentroid<F>` gains `pub metric: NcMetric` (`enum {Euclidean (default), Manhattan}`) + `with_metric` builder, threaded into `FittedNearestCentroid`. `Fit::fit` branches: `Euclidean` per-class mean (unchanged); `Manhattan` per-class feature-wise median (`np.median(X[mask], axis=0)`, `_nearest_centroid.py:164-167`). `predict` branches: `Euclidean` L2, `Manhattan` L1 distance (`:218`), first-index tie-break. Euclidean default byte-identical (≤1e-12 guard). Shrinkage stays euclidean-only (documented). Verification (live sklearn 1.5.2, R-CHAR-3, `X=[[1,0],[2,0],[3,1],[10,5],[11,5],[12,6]]`, `y=[0,0,0,1,1,1]`): manhattan `centroids_=[[2,0],[11,5]]` (median) vs euclidean mean `[[2,0.3333],[11,5.3333]]`; manhattan `predict([[4,1],[9,5]])=[0,1]`. Tests `nearest_centroid_manhattan_median_centroids`, `nearest_centroid_manhattan_predict_matches_sklearn`, `nearest_centroid_euclidean_default_unchanged`. |
| REQ-5 (classes_ / centroids_) | SHIPPED | impl `fn classes`/`fn n_classes` (`HasClasses`) returns sorted-unique labels mirroring `classes_ = le.classes_` (sklearn `:145`); `fn centroids` returns the `(n_classes, n_features)` array mirroring `centroids_` (`:154`). Non-test consumer: PyO3 `RsNearestCentroid` exposes both, `lib.rs` re-export. Oracle (AC-6): `classes_==[0,1]`, `centroids_.shape==(2,2)`, non-contiguous `[10,20]` — ferro matches; tests `test_nearest_centroid_has_classes`, `test_nearest_centroid_noncontiguous_labels`. |
| REQ-6 (n_classes < 2 guard) | NOT-STARTED | open prereq blocker #804. `fn fit` has no minimum-classes check and fits a single-class model (see `test_nearest_centroid_single_class`, which asserts the `Ok`). sklearn raises `ValueError("The number of classes has to be greater than one; got 1 class")` (`_nearest_centroid.py:147-151`). Pin (AC-7): single-class fit → sklearn raises, ferro returns `Ok`. |
| REQ-7 (PyO3 binding) | NOT-STARTED | open prereq blocker #805. `ferrolearn-python/src/extras.rs` registers `RsNearestCentroid` over `FittedNearestCentroid<f64>` constructed via `NearestCentroid::<f64>::new()` (registered in `lib.rs`), so the estimator IS exposed — but the binding inherits the library gaps: **no `metric` / strictly-positive `shrink_threshold` constructor args** (REQ-1), and the shrink/median (REQ-3), manhattan (REQ-4), and single-class-guard (REQ-6) divergences surface through `import ferrolearn` vs `import sklearn`. NOT-STARTED until the library REQs land and the shim exposes `metric`/`shrink_threshold`/`classes_`/`centroids_` at parity. |
| REQ-8 (ferray substrate) | NOT-STARTED | open prereq blocker #806. `nearest_centroid.rs` imports `ndarray::{Array1, Array2}` + `num_traits::Float` (the wrong substrate); not migrated to `ferray-core` (R-SUBSTRATE-1/2). |

## Architecture

`nearest_centroid.rs` follows the unfitted/fitted split (CLAUDE.md naming):
`NearestCentroid<F>` (params) → `Fit` → `FittedNearestCentroid<F>` (the learned
`centroids: Array2<F>` + `classes: Vec<usize>`). Generic over
`F: Float + Send + Sync + 'static`; all public methods return
`Result<_, FerroError>` (no panics in library code).

**Fit path (`fn fit`).** Validation (non-empty → `InsufficientSamples`; row-count
→ `ShapeMismatch`; negative `shrink_threshold` → `InvalidParameter`). `classes` is
the sorted-deduped `y` (the `LabelEncoder.classes_` analog, sklearn `:143-145`).
Per-class centroids are accumulated as feature-wise means (euclidean only). The
**shrinkage block** (gated on `shrink_threshold.is_some()`) recomputes the overall
centroid, the pooled within-class variance → `pooled_var` (std), then
soft-thresholds each centroid's deviation from the overall centroid. This block is
where REQ-3 lives: ferrolearn's `pooled_var` lacks the `s += np.median(s)` damping
(sklearn `:184`) and clamps near-zero std to `1.0` rather than raising the
zero-variance `ValueError` (sklearn `:174-175`).

**Predict path (`fn predict`).** Feature-count check (`ShapeMismatch`), then an
argmin over squared euclidean distance to each centroid (first-wins tie-break via
strict `<`) — the `pairwise_distances_argmin(..., metric="euclidean")` analog
(sklearn `:217-219`). Because there is no `metric` field, there is no manhattan
(L1) predict branch (REQ-4).

**Invariants held vs sklearn (euclidean, no-shrink):** per-class mean centroids
(AC-2), nearest-centroid `predict` (AC-2), `classes_` sorted-unique incl.
non-contiguous labels (AC-6), `centroids_` shape (AC-6). **Invariants NOT held:**
the `metric` parameter and manhattan median centroid (REQ-1/4); the
`s += median(s)` shrink term + the `(0, None)` constraint + the zero-variance
`ValueError` (REQ-3); the `n_classes < 2` `ValueError` (REQ-6); the ferray
substrate (REQ-8).

**Fitted introspection** is via `HasClasses` (`classes`/`n_classes`), `centroids`,
and `score`. There is no `n_features_in_`/`feature_names_in_` analog (sklearn
`:60-69`) — minor, folded into REQ-7's binding-parity surface.

## Verification

Library crate (green at baseline `2013942c` for the existing — narrower —
euclidean contract):
```
cargo test -p ferrolearn-neighbors --lib nearest_centroid
cargo clippy -p ferrolearn-neighbors --all-targets -- -D warnings
cargo fmt --all --check
```
The existing `#[test]`s pin ferrolearn's current behavior — including some that
pin the DIVERGENT behavior (`test_nearest_centroid_single_class` asserts the
single-class `Ok` that sklearn rejects; `test_nearest_centroid_negative_shrink_threshold`
pins the negative-rejection but **not** the `0`-rejection sklearn requires). They
establish REQ-2/REQ-5 parity (AC-2/AC-6 values match the oracle) but do NOT
establish REQ-1/3/4/6 parity.

**Known crate-gauntlet blocker — clippy `collapsible_if` (#807).**
`nearest_centroid.rs:127` nests `if let Some(threshold) = self.shrink_threshold {
if threshold < F::zero() { ... } }`. On the workspace MSRV (1.88, raised by the
ferray substrate) let-chains are stable, so clippy's `collapsible_if` lint fires
under `-D warnings` (collapse to `if let Some(threshold) = self.shrink_threshold
&& threshold < F::zero()`). This blocks `cargo clippy --all-targets -- -D
warnings` for this crate and must be cleared by an acto-fixer. It is a lint-only
fix, not a behavior change. (NO `.rs` edit is made by this doc-author dispatch.)

Live sklearn oracle (installed 1.5.2, run from `/tmp`) — the divergences a critic
should pin first (R-CHAR-3 expected values):
```
# REQ-1 (defaults / metric param):
python3 -c "from sklearn.neighbors import NearestCentroid as N; c=N(); print(c.metric, c.shrink_threshold)"  # euclidean None
# REQ-2 (euclidean centroids + predict) — must stay green:
python3 -c "
import numpy as np; from sklearn.neighbors import NearestCentroid as N
X=np.array([[0,0],[0.5,0],[0,0.5],[0.5,0.5],[5,5],[5.5,5],[5,5.5],[5.5,5.5]]); y=np.array([0,0,0,0,1,1,1,1])
c=N().fit(X,y); print(c.centroids_.tolist(), c.predict(X).tolist())"  # [[0.25,0.25],[5.25,5.25]] [0,0,0,0,1,1,1,1]
# REQ-3 (shrink median-add): sklearn != ferro
python3 -c "
import numpy as np; from sklearn.neighbors import NearestCentroid as N
X=np.array([[0,0],[0.5,0],[0,0.5],[0.5,0.5],[5,5],[5.5,5],[5,5.5],[5.5,5.5]]); y=np.array([0,0,0,0,1,1,1,1])
print(N(shrink_threshold=0.5).fit(X,y).centroids_.tolist())"  # [[0.35206...,0.35206...],[5.14794...,5.14794...]]  (ferro: [[0.30103...,...],[5.19897...,...]])
# REQ-3 (constraint > 0 and zero-variance):
python3 -c "from sklearn.neighbors import NearestCentroid as N; import numpy as np
try: N(shrink_threshold=0).fit(np.array([[0,0],[1,1],[2,2],[3,3]]),np.array([0,0,1,1]))
except Exception as e: print(type(e).__name__)"  # InvalidParameterError (ferro: Ok)
python3 -c "from sklearn.neighbors import NearestCentroid as N; import numpy as np
try: N(shrink_threshold=0.5).fit(np.array([[1,1],[1,1],[1,1],[1,1]]),np.array([0,0,1,1]))
except Exception as e: print(type(e).__name__, str(e)[:30])"  # ValueError All features have zero variance (ferro: Ok)
# REQ-4 (manhattan median):
python3 -c "
import numpy as np; from sklearn.neighbors import NearestCentroid as N
X=np.array([[0,0],[1,0],[10,0],[5,5],[5.5,5],[100,5]]); y=np.array([0,0,0,1,1,1])
print(N(metric='manhattan').fit(X,y).centroids_.tolist())"  # [[1.0,0.0],[5.5,5.0]]  (euclidean mean: [[3.6666...,0.0],[36.8333...,5.0]])
# REQ-6 (n_classes<2 ValueError):
python3 -c "from sklearn.neighbors import NearestCentroid as N; import numpy as np
try: N().fit(np.array([[1,1],[1.5,1],[1,1.5]]),np.array([5,5,5]))
except Exception as e: print(type(e).__name__, str(e)[:40])"  # ValueError The number of classes has to be ... (ferro: Ok)
```
A characterization pin (R-CHAR-3) for each NOT-STARTED REQ belongs in
`ferrolearn-neighbors/tests/divergence_nearest_centroid.rs`, asserting the
live-sklearn expected values above and FAILING against current
`nearest_centroid.rs`. REQ-2 and REQ-5 are SHIPPED (oracle-matching, tested,
consumed by the re-export + PyO3 binding).

ferrolearn-python (REQ-7 binding parity, after the library REQs land):
```
cd /home/doll/ferrolearn/ferrolearn-python && maturin develop
PYTHONPATH=python python3 -m pytest tests/divergence_neighbors.py -q
```
asserting `ferrolearn.NearestCentroid` matches `sklearn.neighbors.NearestCentroid`
on `metric`/`shrink_threshold` constructor args and `classes_`/`centroids_`.

## Blockers to open

- #801 — REQ-1 (defaults + param surface): no `metric` constructor parameter
  (sklearn `_nearest_centroid.py:104,108`); no strictly-positive `(0, None)`
  `shrink_threshold` constraint (`:105`).
- #802 — REQ-3 (shrink_threshold shrinkage): omits `s += np.median(s)`
  (`:184`) so shrunken `centroids_` diverge; accepts `shrink_threshold=0`
  (sklearn `>0`, `:105`); clamps zero-variance to `1.0` instead of raising
  `ValueError` (`:174-175`). Pin: `shrink_threshold=0.5` → sklearn
  `[[0.35206...,...],[5.14794...,...]]`, ferro `[[0.30103...,...],[5.19897...,...]]`.
- #803 — REQ-4 (metric="manhattan"): no median-centroid path
  (`np.median(X[mask],axis=0)`, `:167`) nor manhattan `predict` (`:218`). Pin:
  manhattan `centroids_==[[1.0,0.0],[5.5,5.0]]`, ferro inexpressible.
- #804 — REQ-6 (n_classes < 2 guard): no `ValueError` on single class
  (`:147-151`); `test_nearest_centroid_single_class` currently asserts the
  divergent `Ok`.
- #805 — REQ-7 (PyO3 binding parity): `RsNearestCentroid` is registered but
  inherits the library gaps; expose `metric`/`shrink_threshold` constructor args
  and `classes_`/`centroids_` at sklearn parity once #801-#804 land.
- #806 — REQ-8 (ferray substrate): migrate `nearest_centroid.rs` off
  `ndarray`/`num-traits` to `ferray-core` (R-SUBSTRATE).
- #807 — Crate-gauntlet (not a sklearn divergence): clippy `collapsible_if` at
  `nearest_centroid.rs:127` (`if let Some(threshold) = ... { if threshold <
  F::zero() {...} }` → let-chain `&&`; MSRV 1.88). Blocks `cargo clippy
  --all-targets -- -D warnings`; acto-fixer clears it.
