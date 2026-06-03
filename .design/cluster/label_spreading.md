# Label Spreading (sklearn.semi_supervised.LabelSpreading)

<!--
tier: 3-component
status: draft
baseline-commit: 8be3f21c
upstream: scikit-learn 1.5.2 (commit 156ef14)
upstream-paths:
  - sklearn/semi_supervised/_label_propagation.py   # class BaseLabelPropagation(ClassifierMixin, BaseEstimator) (:76-335); _parameter_constraints (:110-118); __init__ (:120-142); _get_kernel (:144-165); predict (:173-191); predict_proba (:193-231); fit (:233-335); class LabelSpreading(BaseLabelPropagation) (:486-623); _variant="spreading" (:582); LabelSpreading._parameter_constraints["alpha"]=[Interval(Real,0,1,closed="neither")] (:584-585); LabelSpreading.__init__ alpha=0.2/max_iter=30/tol=1e-3 (:587-607); _build_graph csgraph_laplacian(normed=True) (:609-623). NOTE: class LabelPropagation (:338-483) is the SEPARATE label_propagation.rs unit; THIS unit owns ONLY LabelSpreading and the BaseLabelPropagation behavior reached through _variant="spreading".
ferrolearn-module: ferrolearn-cluster/src/label_spreading.rs
parity-ops: LabelSpreading (.__init__, .fit, .predict, .predict_proba, .score, .transduction_/labels_, .label_distributions_, .classes_, .n_iter_)
crosslink-issue: 1008
-->

## Summary

`ferrolearn-cluster/src/label_spreading.rs` mirrors scikit-learn's
`LabelSpreading` (`sklearn/semi_supervised/_label_propagation.py`,
`class LabelSpreading(BaseLabelPropagation)` `:486-623`, `_variant="spreading"`
`:582`) — graph-based semi-supervised classification that propagates known labels
(label `-1` = unlabeled) through a **normalized-graph-Laplacian** affinity graph with
soft clamping (the `alpha` factor). It shares `BaseLabelPropagation.fit` `:233-335`
with `LabelPropagation`; the `_variant` branch (`:290-292,316-320`) is what makes it
"spreading". It exposes the unfitted `LabelSpreading<F>` (`kernel`, `gamma`,
`n_neighbors`, `max_iter`, `tol`, `alpha`), the fitted `FittedLabelSpreading<F>`
(stores `labels_`, `label_distributions_`, `x_train_`, `n_classes_`), a `Predict`
impl, and `predict_proba` / `score` methods. It is re-exported at the crate root
(`pub use label_spreading::{FittedLabelSpreading, LabelSpreading,
LabelSpreadingKernel}`, `ferrolearn-cluster/src/lib.rs:111`).

**Under honest underclaim (R-HONEST-3), exactly ONE behavior VALUE-matches the live
sklearn 1.5.2 oracle and SHIPS through the crate re-export: the transduction
PARTITION (label co-membership) on well-separated data with CONTIGUOUS `0..k-1`
label sets (REQ-1).** On such data ferrolearn's `transduction_`-analog (`labels_`)
assigns the same class to each blob as sklearn (Probe A), and the in-tree tests
(`test_label_spreading_basic`, `test_three_classes`, `test_knn_kernel`,
`test_alpha_affects_results`) pin that. What does NOT ship — every numerical-value and
non-contiguous-label claim, all driven by deliberate algorithmic divergences in
`fn fit` / `fn build_rbf_affinity` / `fn normalized_laplacian` / `fn spread`:

1. **`classes_` mapping / `n_classes` (CORRECTNESS BUG on non-contiguous labels —
   IDENTICAL to the just-fixed `label_propagation` #999).** sklearn `classes_ =
   np.unique(y)` minus `-1` (`:272-274`), `n_classes = len(classes_)`, and
   `transduction_ = classes_[argmax(label_distributions_, axis=1)]` (`:333`) — the
   argmax INDEX is mapped THROUGH `classes_`. ferrolearn `fn fit` computes
   `n_classes = max(label) + 1` (the `.max()...+1` expression `~:477-483`) and emits
   `labels = best_c as isize` (the raw argmax index, `fn fit` argmax loop `:515-529`),
   ASSUMING contiguous `0..n_classes`. For a `{0,2}` label set sklearn returns
   `transduction_ ∈ {0,2}` with `n_classes=2` (Probe B); ferrolearn builds a 3-column
   distribution, can emit a phantom class `1`, and reports `n_classes()==3` — WRONG.
   The highest-value, cleanest-to-pin divergence; same fix as #999 (REQ-4).
2. **`alpha=0` validation (R-DEV-2 ABI bug).** sklearn `LabelSpreading._parameter_constraints["alpha"] =
   [Interval(Real, 0, 1, closed="neither")]` (`:584-585`) — `alpha` in the OPEN
   interval `(0,1)`: BOTH `alpha=0` AND `alpha=1` are rejected with
   `InvalidParameterError` (Probe C). ferrolearn `fn fit` rejects `alpha < 0 ||
   alpha >= 1` (`:446-451`) — it ALLOWS `alpha=0`, and `test_alpha_zero_recovers_initial`
   relies on that. Divergence: ferrolearn must reject `alpha=0` (R-HONEST-4 — the fix
   removes/updates `test_alpha_zero_recovers_initial`). Clean minimal pin (REQ-2).
3. **Normalized-Laplacian graph S VALUE — RBF self-affinity in the degree (R-DEV-1).**
   sklearn `_build_graph` (`:609-623`): `affinity = rbf_kernel(X, X)` (diagonal `=1`,
   self-affinity), `laplacian = csgraph_laplacian(affinity, normed=True)` =
   `I - D^{-1/2} A D^{-1/2}` (degree `D` INCLUDES the diagonal `1`), then
   `laplacian = -laplacian` and the diagonal is zeroed (`:617-622`) → graph
   `S[i,j] = (D^{-1/2} A D^{-1/2})[i,j]` for `i≠j`. ferrolearn `fn build_rbf_affinity`
   leaves the `W` diagonal ZERO (`:282-299`) and `fn normalized_laplacian` computes
   `S[i,j] = D^{-1/2}[i] W[i,j] D^{-1/2}[j]` with `D` = row sums EXCLUDING the (zero)
   diagonal (`:332-351`). sklearn's degree includes the self-affinity `1`; ferrolearn's
   does not → different `S` off-diagonal values → different `label_distributions_`
   (REQ-3, Probe D).
4. **Iteration — no per-iter normalization + zero-init unlabeled (R-DEV-1).** sklearn
   spreading iteration (`:316-320`): `label_distributions_ = alpha * (graph @
   label_distributions_) + y_static` where `y_static = label_distributions_ *
   (1-alpha)` (one-hot for labeled, ZERO for unlabeled, `:290-292`); there is NO row
   normalization DURING iteration — only ONCE at the END (`:328-330`). ferrolearn
   `fn spread` (`:369-405`) computes `alpha*S@f + (1-alpha)*initial_y` THEN
   row-normalizes EVERY iteration, and `initial_y` for unlabeled rows is UNIFORM
   `1/n_classes` (`:503-507`), not zero. Two divergences: per-iteration normalization +
   uniform-vs-zero unlabeled init (REQ-4 group — pinned under REQ-3).
5. **Convergence criterion + `n_iter_` (R-DEV-1).** sklearn checks at the START of the
   loop with the L1 abs-sum `np.abs(label_distributions_ - l_previous).sum() < tol`
   (`:301`). ferrolearn `fn spread` checks at the END with the L2 norm `sqrt(Σd²) <
   tol` (`:391-404`) — a different stopping rule and iteration count (Probe D
   `n_iter_=4`); no `n_iter_` exposed (REQ-5).
6. **`tol` default (R-DEV-2).** sklearn `LabelSpreading` `tol=1e-3` (`:595`/Probe D);
   ferrolearn `fn new` sets `tol=1e-4` (`:94`) — a default divergence (REQ-6).
7. **`predict` / `predict_proba` semantics (R-DEV-3).** sklearn `predict_proba`
   (`:193-231`) is a KERNEL-WEIGHTED combination over ALL training rows
   (`rbf_kernel(X_train, X_query).T @ label_distributions_`, row-normalized);
   `predict = classes_[argmax(predict_proba)]` (`:190-191`). ferrolearn `fn predict` /
   `fn predict_proba` (`:201-233,550-587`) use the NEAREST training point's
   distribution row only (REQ-7).
8. **Attribute naming + missing attributes (R-DEV-2/3).** sklearn exposes
   `transduction_`, `classes_`, `n_iter_`, `X_`, `n_features_in_` (`:529-555` docstring,
   `:264,274,300,333-334`). ferrolearn names the fitted labels `labels_` and exposes
   neither `classes_`, `n_iter_`, `X_`, nor `n_features_in_` (REQ-8).
9. **`ConvergenceWarning` + `n_iter_` on non-convergence (R-DEV-2/3).** sklearn warns
   `ConvergenceWarning` and increments `n_iter_` when `max_iter` is reached
   (`:321-326`). ferrolearn `fn spread` breaks silently (REQ-9).
10. **KNN graph — symmetrize vs connectivity (R-DEV-1).** ferrolearn
    `fn build_knn_affinity` SYMMETRIZES (`w[i,j]=w[j,i]=1`, `:322-325`); sklearn knn
    `_get_kernel` uses `kneighbors_graph(mode="connectivity")` — a DIRECTED graph
    (`:156-157`) before the normalized-Laplacian step (REQ-10).
11. **`fit` input validation / error ABI (R-DEV-2).** sklearn validates via
    `check_classification_targets` + `_validate_data` (`:258-265`) and the open-interval
    `alpha` (`:584-585`), raising `InvalidParameterError`. ferrolearn `fn fit`
    (`:446-474`) raises `FerroError::InvalidParameter` (different TYPE/ABI), rejects
    `gamma>0` (stricter than sklearn's `[0,∞)`), and has no `check_classification_targets`
    analog (REQ-11).
12. **ferray substrate (R-SUBSTRATE).** The unit imports `ndarray::{Array1, Array2}` +
    `num_traits::Float`, not `ferray-core` / `ferray::linalg` (REQ-12).

There is **no `ferrolearn-python` binding** — `grep -rn "LabelSpreading\|
RsLabelSpread" ferrolearn-python/` is EMPTY (Probe G). `LabelSpreading` /
`FittedLabelSpreading` / `LabelSpreadingKernel` are existing pub APIs re-exported at
the crate root (the only non-test consumer; grandfathered per S5/R-DEFER-1). A REQ
for the PyO3 binding is NOT-STARTED (REQ-13).

## Live oracle probes (sklearn 1.5.2, run from /tmp)

Expected values are from the installed sklearn 1.5.2 oracle, never literal-copied
from ferrolearn (R-CHAR-3). Fixtures: `blobs2` = two well-separated 4-point blobs near
origin and `(10,10)`, 8×2, one labeled point per blob; `noncontig` = `blobs2` with the
second blob's labeled point set to class `2` (label set `{0,2}`); `line` = 4 colinear
points `[[0.,0.],[0.3,0.],[0.6,0.],[1.0,0.]]` with endpoints labeled `0` and `1`.

### Probe A — contiguous-label transduction PARTITION (the one agreement)
```
python3 -c "import numpy as np; from sklearn.semi_supervised import LabelSpreading; \
X=np.array([[0.,0.],[0.1,0.],[0.,0.1],[0.1,0.1],[10.,10.],[10.1,10.],[10.,10.1],[10.1,10.1]]); \
y=np.array([0,-1,-1,-1,1,-1,-1,-1]); m=LabelSpreading().fit(X,y); \
print(m.transduction_.tolist(), m.classes_.tolist(), m.n_iter_)"
# transduction_ [0,0,0,0,1,1,1,1]   classes_ [0,1]   n_iter_ 4
```
**Finding:** on contiguous `{0,1}` labels with well-separated blobs, sklearn assigns
the whole first blob to `0` and the whole second to `1`. ferrolearn
`LabelSpreading::<f64>::new().fit(blobs2,y).labels()` produces the SAME partition
(`[0,0,0,0,1,1,1,1]`) — the only VALUE-parity claim (REQ-1). The `label_distributions_`
VALUES still diverge (Probe D), but the argmax partition agrees because each blob's
labeled seed dominates.

### Probe B — non-contiguous {0,2} labels (the correctness bug — REQ-4)
```
python3 -c "import numpy as np; from sklearn.semi_supervised import LabelSpreading; \
X=np.array([[0.,0.],[0.1,0.],[0.,0.1],[0.1,0.1],[10.,10.],[10.1,10.],[10.,10.1],[10.1,10.1]]); \
y=np.array([0,-1,-1,-1,2,-1,-1,-1]); m=LabelSpreading().fit(X,y); \
print(m.classes_.tolist(), m.transduction_.tolist(), m.label_distributions_.shape)"
# classes_ [0,2]   transduction_ [0,0,0,0,2,2,2,2]   label_distributions_ shape (8,2)
```
**Finding:** sklearn `classes_ = [0,2]`, `n_classes = 2`, an `(8,2)`
`label_distributions_`, and `transduction_` mapped through `classes_` → `{0,2}`
(NEVER `1`). ferrolearn `fn fit` sets `n_classes = max(label)+1 = 3`, builds an `(8,3)`
distribution, and emits the raw argmax index — so `n_classes()` reports `3` not `2`, a
phantom middle column `1` exists, and the contract value `2` is emitted only by index
coincidence. Cleanest single-fix pin: assert `fitted.n_classes() == 2` AND `labels ⊆
{0,2}` on `noncontig` — currently FAILS.

### Probe C — alpha open-interval validation (the ABI bug — REQ-2)
```
python3 -c "import numpy as np; from sklearn.semi_supervised import LabelSpreading; \
X=np.array([[0.,0.],[1.,1.]]); y=np.array([0,1]); \
LabelSpreading(alpha=0).fit(X,y)"
# raises sklearn.utils._param_validation.InvalidParameterError (alpha must be in (0,1))
python3 -c "... LabelSpreading(alpha=1).fit(X,y)"
# also raises InvalidParameterError
```
**Finding:** sklearn `_parameter_constraints["alpha"] = [Interval(Real, 0, 1,
closed="neither")]` (`:584-585`) → the OPEN interval `(0,1)`: BOTH endpoints
rejected. ferrolearn `fn fit` rejects `alpha < 0 || alpha >= 1` (`:446-451`) — it
ACCEPTS `alpha=0` (and `test_alpha_zero_recovers_initial` exercises that acceptance).
A characterization pin asserting `LabelSpreading::<f64>::new().with_alpha(0.0).fit(...)`
is `Err` currently FAILS. The fix flips `>=` plus the lower bound to a strict
`<= 0 || >= 1` rejection and updates `test_alpha_zero_recovers_initial` (R-HONEST-4).

### Probe D — label_distributions_ VALUES + defaults (REQ-3/5/6)
```
python3 -c "import numpy as np; from sklearn.semi_supervised import LabelSpreading; \
X=np.array([[0.,0.],[0.3,0.],[0.6,0.],[1.0,0.]]); y=np.array([0,-1,-1,1]); \
m=LabelSpreading(gamma=1.0,tol=1e-3).fit(X,y); \
print(np.round(m.label_distributions_,6).tolist(), m.transduction_.tolist(), m.n_iter_); \
print(LabelSpreading().alpha, LabelSpreading().tol, LabelSpreading().max_iter, \
LabelSpreading().gamma, LabelSpreading().n_neighbors, LabelSpreading().kernel)"
# label_distributions_ [[0.952495,0.047505],[0.571687,0.428313],[0.455792,0.544208],[0.04756,0.95244]]
# transduction_ [0,0,1,1]   n_iter_ 4
# defaults: 0.2 0.001 30 20 7 rbf
```
**Finding:** sklearn's rows are produced by (a) the normalized-Laplacian graph whose
degree INCLUDES the self-affinity `1` (RBF diagonal), (b) `y_static` for unlabeled
rows = ZERO and the soft-clamp update `alpha*graph@F + y_static` (`:316-320`), (c) NO
per-iteration normalization (only once at the end, `:328-330`), and (d) an L1-at-start
convergence test stopping at `n_iter_=4`. Note the LABELED endpoints are NOT clamped to
exactly `[1,0]`/`[0,1]` — sklearn yields `[0.952495,0.047505]` / `[0.04756,0.95244]`,
because spreading uses soft clamping (`y_static *= 1-alpha`), unlike propagation which
hard-clamps. ferrolearn `fn spread` builds the SAME graph minus the self-affinity in
the degree, starts unlabeled rows at UNIFORM `[0.5,0.5]`, and row-normalizes EVERY
iteration with an L2-at-end stop — its row values differ. Also `LabelSpreading().tol`
is `0.001` (ferrolearn `fn new` `1e-4`); `alpha=0.2`, `max_iter=30`, `gamma=20`,
`n_neighbors=7`, `kernel="rbf"` all AGREE (`fn new` `:88-97`).

### Probe G — non-test consumer
`grep -rn "LabelSpreading\|RsLabelSpread" ferrolearn-python/` is **EMPTY** — there is
no PyO3 binding, so `import ferrolearn` cannot reach `LabelSpreading`. `grep -rn
"LabelSpreading" ferrolearn-cluster/src/` outside `label_spreading.rs` finds only the
crate re-export (`pub use label_spreading::{FittedLabelSpreading, LabelSpreading,
LabelSpreadingKernel}`, `lib.rs:111`) and `//!` doc-comment references (`lib.rs:22,62`).
The sole non-test consumer of `fit`/`predict`/`predict_proba`/`score`/the accessors is
that re-export (grandfathered per S5/R-DEFER-1).

## Requirements

- REQ-1: **contiguous-label transduction PARTITION (R-DEV-1).** Mirror
  `LabelSpreading().fit(X,y).transduction_` producing the same per-blob class
  assignment on well-separated data with a CONTIGUOUS `0..k-1` label set. ferrolearn
  `fn fit` builds the RBF/KNN graph, computes the normalized Laplacian `fn
  normalized_laplacian`, runs `fn spread`, and takes the per-row argmax (`labels_`),
  recovering sklearn's partition on benign contiguous data (Probe A) — but the
  `label_distributions_` VALUES (REQ-3), non-contiguous labels (REQ-4), `alpha=0`
  validation (REQ-2), `classes_`/`n_iter_` attributes (REQ-8/9), and predict semantics
  (REQ-7) all diverge, so this is a contiguous-label PARTITION claim only.
- REQ-2: **`alpha` open-interval `(0,1)` validation (R-DEV-2 — the ABI bug).** Mirror
  sklearn `LabelSpreading._parameter_constraints["alpha"] = [Interval(Real, 0, 1,
  closed="neither")]` (`:584-585`) — reject BOTH `alpha=0` and `alpha=1` with an
  `InvalidParameterError`-analog. ferrolearn `fn fit` (`:446-451`) rejects `alpha < 0
  || alpha >= 1`, ACCEPTING `alpha=0` (Probe C). Clean minimal fix (flip the lower
  bound to strict + update `test_alpha_zero_recovers_initial`, R-HONEST-4).
- REQ-3: **`label_distributions_` VALUE — normalized-Laplacian degree + spreading
  iteration (R-DEV-1).** Mirror sklearn's graph (`csgraph_laplacian(rbf_kernel(X,X),
  normed=True)`, degree includes the self-affinity `1`, `:615-616`) and the spreading
  update (`y_static[unlabeled]=0`, `label_distributions_ = alpha*graph@F + y_static`,
  NO per-iteration normalization, `:290-292,316-320,328-330`). ferrolearn `fn
  build_rbf_affinity` zeroes the diagonal, `fn normalized_laplacian` excludes the
  diagonal from the degree, and `fn spread` inits unlabeled rows UNIFORM and
  row-normalizes EVERY iteration — different steady-state values (Probe D, rows
  `[0.571687,0.428313]` / `[0.455792,0.544208]`).
- REQ-4: **`classes_` mapping + `n_classes` on non-contiguous labels (R-DEV-1/3 — the
  correctness bug).** Mirror `classes_ = unique(y) \ {-1}`, `n_classes = len(classes_)`,
  `transduction_ = classes_[argmax(...)]` (`:272-274,333`). ferrolearn `fn fit` sets
  `n_classes = max(label)+1` and emits the raw argmax index — WRONG for any
  non-contiguous label set (Probe B: `{0,2}` → sklearn `{0,2}`/`n_classes=2`;
  ferrolearn builds 3 columns, can emit phantom `1`, reports `n_classes=3`). Same fix
  as the just-fixed `label_propagation` #999.
- REQ-5: **convergence criterion — L1-at-start + `n_iter_` (R-DEV-1).** Mirror sklearn's
  start-of-loop L1 test `np.abs(label_distributions_ - l_previous).sum() < tol`
  (`:301`) and the recorded `n_iter_`. ferrolearn `fn spread` (`:391-404`) uses an
  end-of-loop L2 test `sqrt(Σd²) < tol` — a different stopping rule and iteration count
  (Probe D `n_iter_=4`), with no `n_iter_` field.
- REQ-6: **`tol` default `1e-3` (R-DEV-2).** sklearn `LabelSpreading` `tol=1e-3`
  (`:595`/Probe D). ferrolearn `fn new` (`:94`) sets `tol=1e-4` — a default divergence.
- REQ-7: **`predict` / `predict_proba` kernel-weighted semantics (R-DEV-3).** Mirror
  `predict_proba = rbf_kernel(X_train,X).T @ label_distributions_` row-normalized,
  `predict = classes_[argmax]` (`:190-191,218-231`). ferrolearn `fn predict` /
  `fn predict_proba` (`:201-233,550-587`) return the NEAREST training point's
  distribution row only — a different inductive rule.
- REQ-8: **attribute surface `transduction_` / `classes_` / `n_iter_` / `X_` /
  `n_features_in_` (R-DEV-2/3).** sklearn exposes these fitted attributes
  (`:529-555` docstring, `:264,274,300,333-334`). ferrolearn `FittedLabelSpreading`
  names the labels `labels_` (not `transduction_`) and exposes neither `classes_`,
  `n_iter_`, `X_`, nor `n_features_in_` (it has `labels_`/`label_distributions_`/
  `n_classes_`).
- REQ-9: **`ConvergenceWarning` + `n_iter_` on non-convergence (R-DEV-2/3).** sklearn
  warns `ConvergenceWarning` and increments `n_iter_` when `max_iter` is reached
  (`:321-326`). ferrolearn `fn spread` breaks silently, exposes no `n_iter_`, and has
  no warning channel.
- REQ-10: **KNN connectivity graph — directed vs symmetrized (R-DEV-1).** Mirror knn
  `_get_kernel` → `kneighbors_graph(mode="connectivity")` (DIRECTED, `:156-157`) before
  the normalized-Laplacian step (`:615-616`). ferrolearn `fn build_knn_affinity`
  SYMMETRIZES (`w[i,j]=w[j,i]=1`, `:322-325`) — a different affinity for the directed
  case.
- REQ-11: **`fit` input validation / error ABI (R-DEV-2).** sklearn validates via
  `check_classification_targets` + `_validate_data` (`:258-265`) and the
  `_parameter_constraints` (`gamma∈[0,∞)`, `n_neighbors∈(0,∞)`, `max_iter∈(0,∞)`,
  `tol∈[0,∞)`, `alpha∈(0,1)`, `:110-118,584-585`), raising `InvalidParameterError`.
  ferrolearn `fn fit` (`:446-474`) raises `FerroError::InvalidParameter` (a different
  TYPE/ABI) and rejects `gamma>0` (stricter than sklearn's `[0,∞)`); no
  `check_classification_targets` analog.
- REQ-12: **ferray substrate (R-SUBSTRATE).** `label_spreading.rs` imports
  `ndarray::{Array1, Array2}` + `num_traits::Float`, not `ferray-core` /
  `ferray::linalg` (R-SUBSTRATE-1/2).
- REQ-13: **PyO3 binding (R-DEFER-1/3).** No `LabelSpreading`/`RsLabelSpread` in
  `ferrolearn-python` (Probe G) — `import ferrolearn` cannot reach the estimator.

## Acceptance criteria

All expected values from the live sklearn 1.5.2 oracle (run from `/tmp`), never
literal-copied from ferrolearn (R-CHAR-3). Fixtures: `blobs2`, `noncontig`, `line`
(Probes A/B/D).

- AC-1 (REQ-1, partition agrees / values diverge): `LabelSpreading().fit(blobs2,
  [0,-1,-1,-1,1,-1,-1,-1]).transduction_` → sklearn `[0,0,0,0,1,1,1,1]`; ferrolearn
  `labels()` recovers the same partition (in-tree `test_label_spreading_basic`,
  `test_three_classes`, `test_knn_kernel`, `test_alpha_affects_results` pin
  co-membership on contiguous labels).
- AC-2 (REQ-2, diverges — the ABI bug): `LabelSpreading(alpha=0).fit(X,y)` → sklearn
  raises `InvalidParameterError` (Probe C); ferrolearn
  `LabelSpreading::<f64>::new().with_alpha(0.0).fit(blobs2,y)` returns `Ok` (accepted).
- AC-3 (REQ-3, diverges): `LabelSpreading(gamma=1.0,tol=1e-3).fit(line,
  [0,-1,-1,1]).label_distributions_` → sklearn rows
  `[[0.952495,0.047505],[0.571687,0.428313],[0.455792,0.544208],[0.04756,0.95244]]`
  and `n_iter_=4`. ferrolearn's rows differ (degree excludes self-affinity, uniform
  init, per-iteration normalization, soft-clamp endpoints not reproduced).
- AC-4 (REQ-4, diverges — the correctness bug): `LabelSpreading().fit(noncontig,
  [0,-1,-1,-1,2,-1,-1,-1])` → sklearn `classes_=[0,2]`, `n_classes=2`,
  `transduction_ ∈ {0,2}` (Probe B). ferrolearn `fitted.n_classes()` returns `3` and
  `labels()` can emit a phantom `1` / cannot map to the sklearn `{0,2}` contract.
- AC-5 (REQ-6, diverges): `LabelSpreading().tol` → sklearn `0.001`; ferrolearn
  `LabelSpreading::<f64>::new().tol` = `1e-4`.
- AC-6 (REQ-7, diverges): for a query NOT equal to a training point, sklearn
  `predict_proba` is the kernel-weighted average over ALL training rows; ferrolearn
  returns the single nearest training row's distribution — the two disagree whenever
  the second-nearest row carries non-negligible kernel weight.
- AC-7 (REQ-8/9/13): `LabelSpreading().fit(...)` exposes `transduction_` / `classes_` /
  `n_iter_`; ferrolearn `FittedLabelSpreading` exposes
  `labels_`/`label_distributions_`/`n_classes_` only, and `import ferrolearn;
  ferrolearn.LabelSpreading` does not exist (no binding).

## REQ status table

Binary (R-DEFER-2). `LabelSpreading` / `FittedLabelSpreading` / `LabelSpreadingKernel`
are existing pub APIs re-exported at the crate root
(`ferrolearn-cluster/src/lib.rs:111`) — the only non-test consumer; grandfathered
(S5/R-DEFER-1). Cites use symbol anchors (ferrolearn) / `file:line` (sklearn 1.5.2,
commit 156ef14). Live oracle = installed sklearn 1.5.2, run from `/tmp`. Honest
assessment (R-HONEST-3): **THREE REQs SHIP** — the contiguous-label transduction
PARTITION (REQ-1), the `alpha ∈ (0,1)` open-interval validation (REQ-2, fixed iter
125 / #1009), and the `classes_`/`n_classes`/label-VALUE mapping (REQ-4, fixed iter
125 / #1011) — through the crate re-export; the `label_distributions_` value,
attribute-surface, inductive-predict, default-tol, binding, and substrate REQs DIVERGE.
Blocker numbers below are the real filed issues (#1009–#1020).

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (contiguous-label transduction PARTITION) | SHIPPED | impl `fn fit` (graph build → `fn normalized_laplacian` → `fn spread` → per-row argmax → `labels_`) recovers sklearn's per-blob class assignment for a CONTIGUOUS `0..k-1` label set on well-separated data (Probe A → `[0,0,0,0,1,1,1,1]`, `classes_=[0,1]`). Consumer: crate re-export `pub use label_spreading::{FittedLabelSpreading, LabelSpreading, LabelSpreadingKernel}` (`lib.rs:111`). Guards (in-tree, contiguous labels): `test_label_spreading_basic` (labels[0..3]→0, labels[4..7]→1), `test_three_classes` (3 contiguous classes), `test_knn_kernel` (labels[0]→0, labels[4]→1), `test_alpha_affects_results`. Verification: `cargo test -p ferrolearn-cluster --lib label_spreading` → 17 passed, 0 failed (baseline `8be3f21c`). Underclaim: PARTITION on CONTIGUOUS labels only — `label_distributions_` VALUES (REQ-3), non-contiguous labels (REQ-4), `alpha=0` validation (REQ-2), and predict semantics (REQ-7) diverge. |
| REQ-2 (`alpha` open-interval `(0,1)` validation) | SHIPPED | impl `fn fit` now rejects `alpha <= 0 || alpha >= 1` (reason "must be in (0, 1)"), matching sklearn `_parameter_constraints["alpha"] = [Interval(Real, 0, 1, closed="neither")]` (`:585`) — alpha=0 AND alpha=1 both rejected (Probe C). Guard: `divergence_req2_alpha_zero_rejected` + `confirm_alpha_one_already_rejected`; in-tree `test_alpha_zero_recovers_initial` rewritten → `test_alpha_zero_rejected` (R-HONEST-4). Fixed #1009. |
| REQ-3 (`label_distributions_` value — Laplacian degree + spreading iteration) | NOT-STARTED | open prereq blocker #1010. sklearn `_build_graph` degree INCLUDES the RBF self-affinity `1` (`csgraph_laplacian(rbf_kernel(X,X), normed=True)`, `:615-616`), `y_static[unlabeled]=0`, soft-clamp `alpha*graph@F + y_static` with NO per-iteration normalization (`:290-292,316-320,328-330`); `line` rows `[0.952495,0.047505]`/`[0.571687,0.428313]`/`[0.455792,0.544208]`/`[0.04756,0.95244]` (AC-3). ferrolearn `fn build_rbf_affinity` zeroes the diagonal, `fn normalized_laplacian` excludes it from the degree, and `fn spread` inits unlabeled rows UNIFORM `1/n_classes` and row-normalizes EVERY iteration — different steady-state values. |
| REQ-4 (`classes_` mapping + `n_classes` on non-contiguous labels) | SHIPPED | impl `fn fit` now builds `classes_` = sorted unique non-(-1) labels, `n_classes = classes_.len()`, one-hot indexed by class POSITION, argmax mapped through `classes_` — matching sklearn `classes_ = unique(y)\{-1}` + `transduction_ = classes_[argmax]` (`:272-274,333`). Guard: `divergence_req4_noncontiguous_classes_mapping` (`{0,2}` → `n_classes()==2`, `labels ⊆ {0,2}` = sklearn `[0,0,0,0,2,2,2,2]`; was `n_classes()==3` phantom). Fixed #1011 (same fix as `label_propagation` #999). |
| REQ-5 (convergence — L1-at-start + `n_iter_`) | NOT-STARTED | open prereq blocker #1012. sklearn checks `|Δ|.sum() < tol` at loop START (`:301`) and records `n_iter_`; ferrolearn `fn spread` checks `sqrt(Σd²) < tol` at loop END (`:391-404`) with no `n_iter_` — different stopping rule + iteration count (Probe D sklearn `n_iter_=4`). |
| REQ-6 (`tol` default `1e-3`) | NOT-STARTED | open prereq blocker #1013. sklearn `LabelSpreading` `tol=1e-3` (`:595`/Probe D); ferrolearn `fn new` (`:94`) sets `tol = F::from(1e-4)` — default divergence (R-DEV-2). |
| REQ-7 (`predict`/`predict_proba` kernel-weighted) | NOT-STARTED | open prereq blocker #1014. sklearn `predict_proba = rbf_kernel(X_train,X).T @ label_distributions_` row-normalized, `predict = classes_[argmax]` (`:190-191,218-231`). ferrolearn `fn predict` / `fn predict_proba` (`:201-233,550-587`) return the NEAREST training row's distribution only (the `best_j`/`best_dist` nearest-point loop) — a different inductive rule (AC-6). |
| REQ-8 (`transduction_`/`classes_`/`n_iter_`/`X_`/`n_features_in_` attrs) | NOT-STARTED | open prereq blocker #1015. sklearn exposes these (`:529-555` docstring, `:264,274,300,333-334`). ferrolearn `FittedLabelSpreading` exposes `labels_` (named, not `transduction_`), `label_distributions_`, `n_classes_` — no `classes_`, `n_iter_`, `X_`, or `n_features_in_` accessors. |
| REQ-9 (`ConvergenceWarning` + `n_iter_`) | NOT-STARTED | open prereq blocker #1016. sklearn warns `ConvergenceWarning` + `n_iter_ += 1` at `max_iter` (`:321-326`). ferrolearn `fn spread` breaks silently with no `n_iter_` field and no warning channel. |
| REQ-10 (KNN connectivity graph — directed vs symmetrized) | NOT-STARTED | open prereq blocker #1017. sklearn knn `_get_kernel` → `kneighbors_graph(mode="connectivity")` (DIRECTED, `:156-157`) before `csgraph_laplacian` (`:615-616`). ferrolearn `fn build_knn_affinity` SYMMETRIZES (`w[i,j]=w[j,i]=1`, `:322-325`) — a different affinity for the directed case. |
| REQ-11 (validation / error ABI) | NOT-STARTED | open prereq blocker #1018. sklearn `check_classification_targets` + `_parameter_constraints` `gamma∈[0,∞)`, `n_neighbors∈(0,∞)`, `max_iter∈(0,∞)`, `tol∈[0,∞)`, `alpha∈(0,1)` (`:110-118,258-265,584-585`), raising `InvalidParameterError`. ferrolearn `fn fit` (`:446-474`) raises `FerroError::InvalidParameter` (different TYPE/ABI) and rejects `gamma>0` (stricter than sklearn's `[0,∞)`); no `check_classification_targets` analog. |
| REQ-12 (ferray substrate) | NOT-STARTED | open prereq blocker #1019. `label_spreading.rs` imports `ndarray::{Array1, Array2}` + `num_traits::Float`; not migrated to `ferray-core` / `ferray::linalg` (R-SUBSTRATE-1/2). |
| REQ-13 (PyO3 binding) | NOT-STARTED | open prereq blocker #1020. `grep -rn "LabelSpreading\|RsLabelSpread" ferrolearn-python/` is EMPTY (Probe G) — `import ferrolearn` cannot reach `LabelSpreading`. The only non-test consumer of `fit`/`predict`/`predict_proba`/`score`/accessors is the crate re-export (`lib.rs:111`). |

## Architecture

`label_spreading.rs` follows the unfitted/fitted split (CLAUDE.md naming):
`LabelSpreading<F>` (`kernel: LabelSpreadingKernel`, `gamma: F`, `n_neighbors: usize`,
`max_iter: usize`, `tol: F`, `alpha: F`) → `Fit<Array2<F>, Array1<isize>>` →
`FittedLabelSpreading<F>` (private `labels_: Array1<isize>`, `label_distributions_:
Array2<F>`, `x_train_: Array2<F>`, `n_classes_: usize`). Generic over `F: Float + Send +
Sync + 'static`; every public method returns `Result<_, FerroError>` (R-CODE-2).
`FittedLabelSpreading` implements `Predict<Array2<F>>` plus `predict_proba` and `score`
methods. The `LabelSpreadingKernel` enum (`Rbf`/`Knn`) selects the affinity builder.
The `alpha` field is correct here — sklearn `LabelSpreading` keeps `alpha` (default
`0.2`, `:593`) with an OPEN-interval constraint (`:584-585`), unlike `LabelPropagation`
which pops `alpha` and hard-codes `_variant="propagation"`. The defaults `alpha=0.2`,
`max_iter=30`, `gamma=20`, `n_neighbors=7`, `kernel=Rbf` (`fn new` `:88-97`) all match
sklearn (`:587-607`, Probe D); only `tol` diverges (REQ-6).

**Fit path (`fn fit`, `:426-537`).** Validates `n_samples>0`, `y.len()==n_samples`,
`alpha` (`:446-451`, REQ-2 divergence — accepts `0`), kernel-param positivity
(`:453-465`, `gamma>0` stricter than sklearn, REQ-11), and `n_labeled>0`. Then: (1)
`n_classes = max(label)+1` (`:477-483`, REQ-4 divergence — sklearn uses
`len(unique(y)\{-1})`); (2) build the affinity (`fn build_rbf_affinity` — diagonal
ZERO, REQ-3; or `fn build_knn_affinity` — SYMMETRIZED, REQ-10); (3) `fn
normalized_laplacian` computes `S = D^{-1/2} W D^{-1/2}` with `D` = row sums EXCLUDING
the zero diagonal (REQ-3 divergence — sklearn's `csgraph_laplacian(normed=True)` degree
includes the self-affinity `1`); (4) build `initial_y` (`:495-509`) — one-hot for
labeled rows, UNIFORM `1/n_classes` for unlabeled (REQ-3 divergence — sklearn sets
`y_static[unlabeled]=0`); (5) `fn spread` (`:355-408`) — iterate `F ← alpha*S·F +
(1-alpha)*Y`, row-normalize EVERY iteration (REQ-3 divergence — sklearn normalizes once
at the end), stop on end-of-loop L2 `< tol` (REQ-5 divergence); (6) per-row argmax →
`labels_` (raw index, REQ-4 divergence — no `classes_` remap).

**Predict path.** `Predict::predict` (`:550-587`) and `predict_proba` (`:201-233`) find
the single NEAREST training point (squared-Euclidean) and return its label /
distribution row (REQ-7 divergence — sklearn does a kernel-weighted combination over
ALL training rows via `_get_kernel(X_, X)`). `score` (`:242-266`) is a
`ClassifierMixin.score` analog skipping `y==-1` test rows.

**Where sklearn and ferrolearn AGREE.** `kernel="rbf"` default, `gamma=20`,
`alpha=0.2`, `n_neighbors=7`, `max_iter=30` (Probe D); the soft-clamp update form
`alpha*S·F + (1-alpha)*Y` (sklearn `:316-320` — modulo the `y_static` zero-vs-uniform
init and the per-iter normalization); `-1`-as-unlabeled convention; the
contiguous-label argmax PARTITION on well-separated data (Probe A, REQ-1).

**Where they DIVERGE (load-bearing):** the `alpha=0` validation (REQ-2, an ABI bug);
the `classes_` remap / `n_classes` on non-contiguous labels (REQ-4, a correctness bug);
the normalized-Laplacian degree + spreading-iteration init/normalization (REQ-3); the
convergence rule + `n_iter_` (REQ-5); the `tol` default (REQ-6);
`predict`/`predict_proba` inductive semantics (REQ-7); the attribute surface (REQ-8);
`ConvergenceWarning`/`n_iter_` (REQ-9); the KNN graph direction (REQ-10); the error ABI
(REQ-11); the ferray substrate (REQ-12); the missing PyO3 binding (REQ-13).

**Consumer wiring.** The only non-test consumer is the crate re-export
(`pub use label_spreading::{FittedLabelSpreading, LabelSpreading, LabelSpreadingKernel}`,
`ferrolearn-cluster/src/lib.rs:111`). There is no `ferrolearn-python` binding (Probe G)
and no other in-crate consumer.

## Verification

Library crate (green at baseline `8be3f21c` for the existing partition behavior):
```
cargo test -p ferrolearn-cluster --lib label_spreading     # 17 passed, 0 failed
cargo clippy -p ferrolearn-cluster --all-targets -- -D warnings
cargo fmt --all --check
```
The in-tree `#[test]`s (`test_label_spreading_basic`, `test_alpha_zero_recovers_initial`,
`test_convergence`, `test_knn_kernel`, `test_predict_on_new_data`, `test_invalid_alpha`,
`test_no_labeled_error`, `test_label_distributions_shape`, `test_n_classes`,
`test_predict_shape_mismatch`, `test_y_length_mismatch`, `test_empty_data`,
`test_f32_support`, `test_three_classes`, `test_default_constructor`,
`test_invalid_gamma`, `test_alpha_affects_results`) pin ferrolearn's current behavior on
CONTIGUOUS-label, well-separated fixtures — partition co-membership, shapes, `n_classes`
on contiguous data, error edges, f32 support. **None compares `label_distributions_`
VALUES, non-contiguous-label `transduction_`/`classes_`, the `tol` default, the
`alpha=0` rejection, or `predict_proba` against the live sklearn `LabelSpreading`
oracle**, so they stay green despite the divergences. Note `test_alpha_zero_recovers_initial`
actively pins the WRONG behavior (accepts `alpha=0`) — it must be removed/updated when
REQ-2 lands (R-HONEST-4). `test_n_classes` checks `n_classes()==2` only on a contiguous
`{0,1}` fixture — it does NOT exercise the `{0,2}` correctness bug (REQ-4).

Live sklearn oracle (installed 1.5.2, run from `/tmp`) — the divergences a critic
should pin (R-CHAR-3 expected values). **Pin REQ-4 (non-contiguous `classes_` mapping)
and REQ-2 (`alpha=0` rejection) FIRST — both are clean single-fix pins, independent of
each other:**
```
# REQ-4 (correctness bug — non-contiguous {0,2} labels)
python3 -c "import numpy as np; from sklearn.semi_supervised import LabelSpreading; \
X=np.array([[0.,0.],[0.1,0.],[0.,0.1],[0.1,0.1],[10.,10.],[10.1,10.],[10.,10.1],[10.1,10.1]]); \
y=np.array([0,-1,-1,-1,2,-1,-1,-1]); m=LabelSpreading().fit(X,y); \
print(m.classes_.tolist(), m.transduction_.tolist())"
# [0,2] [0,0,0,0,2,2,2,2]   -> assert ferrolearn n_classes()==2 AND labels ⊆ {0,2}: currently FAILS
# REQ-2 (ABI bug — alpha=0 rejected, open interval)
python3 -c "import numpy as np; from sklearn.semi_supervised import LabelSpreading; \
LabelSpreading(alpha=0).fit(np.array([[0.,0.],[1.,1.]]), np.array([0,1]))"
# raises InvalidParameterError  -> assert ferrolearn fit(alpha=0) is Err: currently FAILS (returns Ok)
# REQ-3/REQ-5 (label_distributions_ values — Laplacian degree / zero-init / no per-iter norm / L1)
python3 -c "import numpy as np; from sklearn.semi_supervised import LabelSpreading; \
X=np.array([[0.,0.],[0.3,0.],[0.6,0.],[1.0,0.]]); y=np.array([0,-1,-1,1]); \
m=LabelSpreading(gamma=1.0,tol=1e-3).fit(X,y); print(np.round(m.label_distributions_,6).tolist(), m.n_iter_)"
# [[0.952495,0.047505],[0.571687,0.428313],[0.455792,0.544208],[0.04756,0.95244]] 4 -> assert rows within 1e-5: FAILS
# REQ-6 (tol default)
python3 -c "from sklearn.semi_supervised import LabelSpreading; print(LabelSpreading().tol)"   # 0.001 (ferrolearn 1e-4)
```
A characterization pin (R-CHAR-3) for each NOT-STARTED REQ belongs in
`ferrolearn-cluster/tests/divergence_label_spreading.rs`, asserting the live-sklearn
expected values above and FAILING against current `label_spreading.rs`.

ferrolearn-python (REQ-13 binding parity, after the binding lands):
```
cd /home/doll/ferrolearn/ferrolearn-python && maturin develop
PYTHONPATH=python python3 -m pytest tests/divergence_label_spreading.py -q
```
asserting `ferrolearn.LabelSpreading` exists and exposes `transduction_` / `classes_` /
`label_distributions_` / `n_iter_`, matching `sklearn.semi_supervised.LabelSpreading`
on the AC fixtures.

## Blockers (to open)

REQ-1 (contiguous-label PARTITION) SHIPS — no blocker. The remaining NOT-STARTED REQs
each need a `-l blocker` issue (dispatcher assigns real `#`-numbers, substituted for the
`#1011` placeholders in the table above). Recommended order — **REQ-4 and REQ-2 first
(both clean, independent single-fix pins), then REQ-3/REQ-5 (the value/iteration
divergences):**

- REQ-4 (**correctness bug, fix FIRST**): use `classes_ = sorted(unique(y) \ {-1})`,
  `n_classes = len(classes_)`, build `initial_y` one-hot indexed by class POSITION, and
  map the final argmax index THROUGH `classes_` for `labels_`
  (`_label_propagation.py:272-274,333`). Pin: `{0,2}` fixture → `n_classes()==2`,
  `labels ⊆ {0,2}`. (Identical to `label_propagation` #999; independent of REQ-2.)
- REQ-2 (**ABI bug, fix SECOND**): reject `alpha <= 0` (open interval `(0,1)`) in
  `fn fit` to match `_parameter_constraints["alpha"]=[Interval(Real,0,1,closed="neither")]`
  (`:584-585`); remove/update `test_alpha_zero_recovers_initial` (R-HONEST-4). Pin:
  `with_alpha(0.0).fit(...)` is `Err`. (Independent of REQ-4 — touches only the
  validation guard.)
- REQ-3: degree INCLUDES the RBF self-affinity (`csgraph_laplacian(rbf_kernel(X,X),
  normed=True)`, `:615-616`); `y_static[unlabeled]=0`; NO per-iteration normalization
  (`:290-292,316-320,328-330`).
- REQ-5: convergence = L1 abs-sum at loop START (`:301`), not L2 at end; record `n_iter_`.
- REQ-6: change `fn new` `tol` default to `1e-3` (`:595`).
- REQ-7: `predict_proba` = kernel-weighted combination over ALL training rows;
  `predict = classes_[argmax]` (`:190-191,218-231`).
- REQ-8: expose `transduction_` (rename/alias `labels_`), `classes_`, `n_iter_`, `X_`,
  `n_features_in_` accessors.
- REQ-9: `ConvergenceWarning` + `n_iter_` on `max_iter` (`:321-326`).
- REQ-10: KNN connectivity graph (directed `kneighbors_graph`, `:156-157`) before the
  normalized-Laplacian step.
- REQ-11: error ABI — `InvalidParameterError` analog + `gamma∈[0,∞)` bound +
  `check_classification_targets` (`:110-118,258-265,584-585`).
- REQ-12: migrate `label_spreading.rs` off `ndarray`/`num-traits` to `ferray-core` /
  `ferray::linalg` (R-SUBSTRATE).
- REQ-13: add the `ferrolearn-python` binding (fit / predict / predict_proba /
  transduction_ / classes_ / label_distributions_ / n_iter_).
