# Label Propagation (sklearn.semi_supervised.LabelPropagation)

<!--
tier: 3-component
status: draft
baseline-commit: f11e336c
upstream: scikit-learn 1.5.2 (commit 156ef14)
upstream-paths:
  - sklearn/semi_supervised/_label_propagation.py   # class BaseLabelPropagation(ClassifierMixin, BaseEstimator) (:76-335); _parameter_constraints (:110-118); __init__ (:120-142); _get_kernel (:144-165); _build_graph abstractmethod (:167-171); predict (:173-191); predict_proba (:193-231); fit (:233-335); class LabelPropagation(BaseLabelPropagation) (:338-483); _variant="propagation" (:423); __init__ (:428-446); _build_graph (:448-462). NOTE: class LabelSpreading (:486+) is the SEPARATE label_spreading.rs unit; THIS unit owns ONLY LabelPropagation.
ferrolearn-module: ferrolearn-cluster/src/label_propagation.rs
parity-ops: LabelPropagation (.__init__, .fit, .predict, .predict_proba, .score, .transduction_/labels_, .label_distributions_, .classes_, .n_iter_)
crosslink-issue: 996
-->

## Summary

`ferrolearn-cluster/src/label_propagation.rs` mirrors scikit-learn's
`LabelPropagation` (`sklearn/semi_supervised/_label_propagation.py`,
`class LabelPropagation(BaseLabelPropagation)` `:338-483`, `_variant="propagation"`
`:423`) — graph-based semi-supervised classification that propagates known labels
(label `-1` = unlabeled) through an RBF/KNN similarity graph to a steady state. It
exposes the unfitted `LabelPropagation<F>` (`kernel`, `gamma`, `n_neighbors`,
`max_iter`, `tol`), the fitted `FittedLabelPropagation<F>` (stores `labels_`,
`label_distributions_`, `x_train_`, `n_classes_`), a `Predict` impl, and
`predict_proba` / `score` methods. It is re-exported at the crate root
(`pub use label_propagation::{FittedLabelPropagation, LabelPropagation,
LabelPropagationKernel}`, `ferrolearn-cluster/src/lib.rs:110`).

**Under honest underclaim (R-HONEST-3), exactly ONE behavior VALUE-matches the live
sklearn 1.5.2 oracle and SHIPS through the crate re-export: the transduction
PARTITION (label co-membership) on well-separated data with CONTIGUOUS `0..k-1`
label sets (REQ-1).** On such data ferrolearn's `transduction_`-analog (`labels_`)
assigns the same class to each blob as sklearn (Probe A), and the in-tree tests
(`test_label_propagation_basic`, `test_three_classes`) pin that. What does NOT
ship — every numerical-value and non-contiguous-label claim, all driven by
deliberate algorithmic divergences in `fit`/`build_rbf_affinity`/`propagate`:

1. **`classes_` mapping / `n_classes` (CORRECTNESS BUG on non-contiguous labels).**
   sklearn `classes_ = np.unique(y)` minus `-1` (`:272-274`), `n_classes =
   len(classes_)`, and `transduction_ = classes_[argmax(label_distributions_,
   axis=1)]` (`:333`) — the argmax INDEX is mapped THROUGH `classes_`. ferrolearn
   computes `n_classes = max(label) + 1` (`fn fit`, the `.max()...+1` expression) and
   emits `labels = best_c as isize` (the raw argmax index, `fn fit` argmax loop),
   ASSUMING contiguous `0..n_classes`. For a `{0,2}` label set sklearn returns
   `transduction_ ∈ {0,2}` with `n_classes=2` (Probe B); ferrolearn builds a 3-column
   distribution, can emit a phantom class `1`, and never emits `2` — WRONG labels.
   This is the highest-value, cleanest-to-pin divergence (REQ-4).

2. **RBF affinity diagonal.** sklearn `_get_kernel` uses `rbf_kernel(X, X, gamma)`
   (`:147`) whose diagonal is `exp(0) = 1` (self-affinity = 1, Probe D). ferrolearn
   `fn build_rbf_affinity` explicitly leaves the diagonal ZERO ("no self-loops").
   Different graph → different `label_distributions_` (REQ-2).

3. **Unlabeled-row initialization.** sklearn `fit` (`:282-289`) initializes
   `label_distributions_ = zeros`, one-hot only for labeled rows, and for
   `_variant=="propagation"` sets `y_static[unlabeled] = 0` — unlabeled rows START AT
   ZERO. ferrolearn (`fn fit`, the `initial_y` else-branch) initializes unlabeled
   rows as UNIFORM `1/n_classes` (REQ-2).

4. **Convergence criterion.** sklearn checks at the START of the loop with the L1
   abs-sum `np.abs(label_distributions_ - l_previous).sum() < tol` (`:301`).
   ferrolearn `fn propagate` checks at the END with the L2 norm
   `sqrt(sum_of_squares) < tol` (REQ-3).

5. **`predict` / `predict_proba` semantics.** sklearn `predict_proba` (`:193-231`) is
   a KERNEL-WEIGHTED combination over ALL training rows
   (`rbf_kernel(X_train, X_query).T @ label_distributions_`, row-normalized);
   `predict = classes_[argmax(predict_proba)]` (`:190-191`). ferrolearn
   `predict`/`predict_proba` use the NEAREST training point's distribution row only
   (REQ-6).

6. **Attribute naming + missing attributes.** sklearn exposes `transduction_`,
   `classes_`, `n_iter_`, `X_`, `n_features_in_`. ferrolearn names the fitted labels
   `labels_` and exposes neither `classes_`, `n_iter_`, nor `transduction_` (REQ-7).

7. **`tol` default + `ConvergenceWarning` + `n_iter_`.** sklearn `tol=1e-3`
   (`:435`); ferrolearn `fn new` `tol=1e-4` (REQ-5). On non-convergence sklearn warns
   `ConvergenceWarning` and sets `n_iter_ += 1` (`:321-326`); ferrolearn is silent
   with no `n_iter_` (REQ-8).

8. **KNN graph + normalization direction.** sklearn knn `_get_kernel` uses
   `kneighbors_graph(mode="connectivity")` (`:156-157`) — a DIRECTED graph (each row
   `i`'s `k` neighbors) — then `_build_graph` normalizes by COLUMN sums
   (`normalizer = affinity.sum(axis=0)`, divide row `i` by column-sum `i`, `:457-461`).
   ferrolearn `fn build_knn_affinity` SYMMETRIZES (`w[i,j]=w[j,i]=1`) and `fn
   row_normalize` normalizes by ROW sums. For the SYMMETRIC RBF graph column- and
   row-normalization coincide; for the directed KNN graph they differ (REQ-9).

There is **no `ferrolearn-python` binding** — `grep -rn "LabelPropagation\|
RsLabelProp" ferrolearn-python/` is EMPTY (Probe G). `LabelPropagation` /
`FittedLabelPropagation` / `LabelPropagationKernel` are existing pub APIs
re-exported at the crate root (the only non-test consumer; grandfathered per
S5/R-DEFER-1). A REQ for the PyO3 binding is NOT-STARTED (REQ-11). The unit imports
`ndarray` + `num-traits`, not the ferray substrate (REQ-12, R-SUBSTRATE).

## Live oracle probes (sklearn 1.5.2, run from /tmp)

Expected values are from the installed sklearn 1.5.2 oracle, never literal-copied
from ferrolearn (R-CHAR-3). Fixtures: `blobs2` = two well-separated 4-point blobs
near origin and `(10,10)`, 8×2, one labeled point per blob; `noncontig` = `blobs2`
with the second blob's labeled point set to class `2` (label set `{0,2}`); `line` =
4 colinear points `[[0],[0.3],[0.6],[1.0]]` (2-D as `[[0.,0.],…]`) with endpoints
labeled `0` and `1`.

### Probe A — contiguous-label transduction PARTITION (the one agreement)
```
python3 -c "import numpy as np; from sklearn.semi_supervised import LabelPropagation; \
X=np.array([[0.,0.],[0.1,0.],[0.,0.1],[0.1,0.1],[10.,10.],[10.1,10.],[10.,10.1],[10.1,10.1]]); \
y=np.array([0,-1,-1,-1,1,-1,-1,-1]); m=LabelPropagation().fit(X,y); \
print(m.transduction_.tolist(), m.classes_.tolist(), m.n_iter_)"
# transduction_ [0,0,0,0,1,1,1,1]   classes_ [0,1]   n_iter_ 2
```
**Finding:** on contiguous `{0,1}` labels with well-separated blobs, sklearn assigns
the whole first blob to `0` and the whole second to `1`. ferrolearn
`LabelPropagation::<f64>::new().fit(blobs2,y).labels()` produces the SAME partition
(`[0,0,0,0,1,1,1,1]`) — the only VALUE-parity claim (REQ-1). The
`label_distributions_` VALUES still diverge (Probes C/F), but the argmax partition
agrees because each blob's labeled seed dominates.

### Probe B — non-contiguous {0,2} labels (the correctness bug — REQ-4)
```
python3 -c "import numpy as np; from sklearn.semi_supervised import LabelPropagation; \
X=np.array([[0.,0.],[0.1,0.],[0.,0.1],[0.1,0.1],[10.,10.],[10.1,10.],[10.,10.1],[10.1,10.1]]); \
y=np.array([0,-1,-1,-1,2,-1,-1,-1]); m=LabelPropagation().fit(X,y); \
print(m.classes_.tolist(), m.transduction_.tolist(), m.label_distributions_.shape)"
# classes_ [0,2]   transduction_ [0,0,0,0,2,2,2,2]   label_distributions_ shape (8,2)
```
**Finding:** sklearn `classes_ = [0,2]`, `n_classes = 2`, a `(8,2)`
`label_distributions_`, and `transduction_` mapped through `classes_` →
`{0,2}` (NEVER `1`). ferrolearn `fn fit` sets `n_classes = max(label)+1 = 3`, builds
a `(8,3)` distribution, and emits the raw argmax index — so the second blob's argmax
column `2` is reported as label `2` only by accident of the index coinciding, but the
phantom middle column `1` (which received the uniform-init mass and propagation from
neither seed cleanly) can win argmax for boundary points, and `n_classes()` reports
`3` not `2`. Cleanest single-fix pin: assert `fitted.n_classes() == 2` AND
`labels` ⊆ `{0,2}` on `noncontig` — currently FAILS.

### Probe C — label_distributions_ VALUES (RBF-diagonal / zero-init / L1 — REQ-2/3)
```
python3 -c "import numpy as np; from sklearn.semi_supervised import LabelPropagation; \
X=np.array([[0.,0.],[0.1,0.],[0.,0.1],[0.1,0.1],[10.,10.],[10.1,10.],[10.,10.1],[10.1,10.1]]); \
y=np.array([0,-1,-1,-1,1,-1,-1,-1]); m=LabelPropagation(gamma=1.0).fit(X,y); \
print(np.round(m.label_distributions_,6)[:3].tolist())"
# [[1.0,0.0],[1.0,0.0],[1.0,0.0]]   (labeled row 0 clamped; rows 1-2 propagate to 1-hot here)
```
On well-separated blobs the steady state is near 1-hot, masking the value
divergence. The value divergence surfaces on overlapping data:

### Probe F — overlapping `line` fixture exposes the VALUE divergence (REQ-2/3)
```
python3 -c "import numpy as np; from sklearn.semi_supervised import LabelPropagation; \
X=np.array([[0.,0.],[0.3,0.],[0.6,0.],[1.0,0.]]); y=np.array([0,-1,-1,1]); \
m=LabelPropagation(gamma=1.0,tol=1e-3).fit(X,y); \
print(np.round(m.label_distributions_,6).tolist(), m.transduction_.tolist(), m.n_iter_)"
# label_distributions_ [[1.0,0.0],[0.55811,0.44189],[0.49024,0.50976],[0.0,1.0]]
# transduction_ [0,0,1,1]   n_iter_ 4
```
**Finding:** sklearn's middle rows are `[0.55811,0.44189]` / `[0.49024,0.50976]`,
produced by (a) an RBF graph whose DIAGONAL is `1` (self-affinity), (b) unlabeled
rows STARTING at zero, and (c) an L1-at-start convergence test stopping after
`n_iter_=4`. ferrolearn builds the SAME graph minus the diagonal, starts unlabeled
rows at the UNIFORM `[0.5,0.5]`, and stops on an L2-at-end test — its middle-row
distribution VALUES differ from `[0.55811,0.44189]` / `[0.49024,0.50976]`. A
characterization pin asserting those two sklearn rows (within `1e-5`) against
`fitted.label_distributions()` FAILS (REQ-2/3).

### Probe D — rbf_kernel diagonal = 1 (REQ-2 evidence)
```
python3 -c "import numpy as np; from sklearn.metrics.pairwise import rbf_kernel; \
X=np.array([[0.,0.],[0.1,0.],[10.,10.]]); K=rbf_kernel(X,X,gamma=20.0); \
print(np.diag(K).tolist(), np.round(K[0],8).tolist())"
# diag [1.0,1.0,1.0]   K[0] [1.0,0.81873075,0.0]
```
sklearn's affinity has `K[i,i]=1`; ferrolearn `fn build_rbf_affinity` leaves
`w[i*n+i]=0`. Distinct graphs.

### Probe E — defaults (REQ-5 evidence)
```
python3 -c "from sklearn.semi_supervised import LabelPropagation; m=LabelPropagation(); \
print(m.max_iter, m.tol, m.gamma, m.n_neighbors, m.kernel)"
# 1000 0.001 20 7 rbf
```
sklearn `tol=1e-3` (`:435`); ferrolearn `fn new` sets `tol=1e-4`. `max_iter=1000`,
`gamma=20`, `n_neighbors=7`, `kernel="rbf"` all AGREE.

### Probe G — non-test consumer
`grep -rn "LabelPropagation\|RsLabelProp" ferrolearn-python/` is **EMPTY** — there is
no PyO3 binding, so `import ferrolearn` cannot reach `LabelPropagation`. `grep -rn
"LabelPropagation" ferrolearn-cluster/src/` outside `label_propagation.rs` finds only
the crate re-export (`pub use label_propagation::{FittedLabelPropagation,
LabelPropagation, LabelPropagationKernel}`, `lib.rs:110`) and `//!` doc-comment
references. The sole non-test consumer of `fit`/`predict`/`predict_proba`/`score`/the
accessors is that re-export.

## Requirements

- REQ-1: **contiguous-label transduction PARTITION (R-DEV-1).** Mirror
  `LabelPropagation().fit(X,y).transduction_` producing the same per-blob class
  assignment on well-separated data with a CONTIGUOUS `0..k-1` label set. ferrolearn
  `fn fit` builds the RBF/KNN graph, row-normalizes, runs `fn propagate`, and takes
  the per-row argmax (`labels_`), recovering sklearn's partition on benign contiguous
  data (Probe A) — but the `label_distributions_` VALUES (REQ-2/3), non-contiguous
  labels (REQ-4), `classes_`/`n_iter_` attributes (REQ-7/8), and predict semantics
  (REQ-6) all diverge, so this is a contiguous-label PARTITION claim only.
- REQ-2: **`label_distributions_` VALUE — RBF diagonal + zero-init (R-DEV-1).**
  Mirror sklearn's graph (`rbf_kernel(X,X,gamma)` diagonal `=1`, `:147`/Probe D) and
  unlabeled-row zero initialization with `y_static[unlabeled]=0` (`:282-289`).
  ferrolearn `fn build_rbf_affinity` zeroes the diagonal and `fn fit` initializes
  unlabeled rows UNIFORM `1/n_classes` — different steady-state values (Probe F).
- REQ-3: **convergence criterion — L1-at-start (R-DEV-1).** Mirror sklearn's
  start-of-loop L1 test `np.abs(label_distributions_ - l_previous).sum() < tol`
  (`:301`). ferrolearn `fn propagate` uses an end-of-loop L2 test `sqrt(Σd²) < tol` —
  a different stopping rule and iteration count (Probe F `n_iter_=4`).
- REQ-4: **`classes_` mapping + `n_classes` on non-contiguous labels (R-DEV-1/3 — the
  correctness bug).** Mirror `classes_ = unique(y) \ {-1}`, `n_classes =
  len(classes_)`, `transduction_ = classes_[argmax(...)]` (`:272-274,333`). ferrolearn
  `fn fit` sets `n_classes = max(label)+1` and emits the raw argmax index — WRONG for
  any non-contiguous label set (Probe B: `{0,2}` → sklearn `{0,2}`/`n_classes=2`;
  ferrolearn builds 3 columns, can emit phantom `1`, reports `n_classes=3`).
- REQ-5: **`tol` default `1e-3` (R-DEV-2).** sklearn `LabelPropagation` `tol=1e-3`
  (`:435`/Probe E). ferrolearn `fn new` sets `tol=1e-4` — a default divergence.
- REQ-6: **`predict` / `predict_proba` kernel-weighted semantics (R-DEV-3).** Mirror
  `predict_proba` = `rbf_kernel(X_train,X).T @ label_distributions_` row-normalized,
  `predict = classes_[argmax]` (`:190-191,218-231`). ferrolearn `predict` /
  `predict_proba` return the NEAREST training point's distribution row only — a
  different inductive rule.
- REQ-7: **attribute surface `transduction_` / `classes_` / `n_iter_` / `X_` /
  `n_features_in_` (R-DEV-2/3).** sklearn exposes these fitted attributes
  (`:264,274,300,333-334`, class docstring `:370-396`). ferrolearn names the labels
  `labels_` (not `transduction_`) and exposes neither `classes_`, `n_iter_`, `X_`,
  nor `n_features_in_` (it has `labels_`/`label_distributions_`/`n_classes_`).
- REQ-8: **`ConvergenceWarning` + `n_iter_` on non-convergence (R-DEV-2/3).** sklearn
  warns `ConvergenceWarning` and increments `n_iter_` when `max_iter` is reached
  (`:321-326`). ferrolearn `fn propagate` is silent, exposes no `n_iter_`, and has no
  warning channel.
- REQ-9: **KNN connectivity graph + column-normalization (R-DEV-1).** Mirror
  `_get_kernel` knn `kneighbors_graph(mode="connectivity")` (directed, `:156-157`)
  and `_build_graph` column-sum normalization (`affinity.sum(axis=0)`, `:457-461`).
  ferrolearn `fn build_knn_affinity` SYMMETRIZES (`w[i,j]=w[j,i]=1`) and `fn
  row_normalize` normalizes by row sums — for the directed KNN graph this is a
  different operator (for the symmetric RBF graph the two normalizations coincide).
- REQ-10: **`fit` input validation / error ABI (R-DEV-2).** sklearn validates via
  `check_classification_targets` + `_validate_data` (`:258-265`) and constrains
  `gamma ∈ [0,∞)`, `n_neighbors ∈ (0,∞)`, `max_iter ∈ (0,∞)`, `tol ∈ [0,∞)`
  (`_parameter_constraints` `:110-118`), raising `InvalidParameterError`. ferrolearn
  `fn fit` checks `n_labeled>0`, `gamma>0` (RBF), `n_neighbors>0` (KNN), shape match,
  raising `FerroError::InvalidParameter` (matching bounds where present, but a
  different error TYPE/ABI; `gamma>0` is stricter than sklearn's `[0,∞)`).
- REQ-11: **PyO3 binding (R-DEFER-1/3).** No `LabelPropagation`/`RsLabelProp` in
  `ferrolearn-python` (Probe G) — `import ferrolearn` cannot reach the estimator.
- REQ-12: **ferray substrate (R-SUBSTRATE).** `label_propagation.rs` imports
  `ndarray::{Array1, Array2}` + `num_traits::Float`, not `ferray-core` /
  `ferray::linalg` (R-SUBSTRATE-1/2).

## Acceptance criteria

All expected values from the live sklearn 1.5.2 oracle (run from `/tmp`), never
literal-copied from ferrolearn (R-CHAR-3). Fixtures: `blobs2`, `noncontig`, `line`
(Probes A/B/F).

- AC-1 (REQ-1, partition agrees / values diverge): `LabelPropagation().fit(blobs2,
  [0,-1,-1,-1,1,-1,-1,-1]).transduction_` → sklearn `[0,0,0,0,1,1,1,1]`; ferrolearn
  `labels()` recovers the same partition (in-tree `test_label_propagation_basic` /
  `test_three_classes` pin co-membership on contiguous labels).
- AC-2 (REQ-2/3, diverges): `LabelPropagation(gamma=1.0,tol=1e-3).fit(line,
  [0,-1,-1,1]).label_distributions_` → sklearn rows
  `[[1,0],[0.55811,0.44189],[0.49024,0.50976],[0,1]]` and `n_iter_=4`. ferrolearn's
  middle rows differ (zero-diagonal graph, uniform init, L2-at-end stop).
- AC-3 (REQ-4, diverges — the correctness bug): `LabelPropagation().fit(noncontig,
  [0,-1,-1,-1,2,-1,-1,-1])` → sklearn `classes_=[0,2]`, `n_classes=2`,
  `transduction_ ∈ {0,2}` (Probe B). ferrolearn `fitted.n_classes()` returns `3` and
  `labels()` can emit a phantom `1` / cannot map to the sklearn `{0,2}` contract.
- AC-4 (REQ-5, diverges): `LabelPropagation().tol` → sklearn `0.001`; ferrolearn
  `LabelPropagation::<f64>::new().tol` = `1e-4`.
- AC-5 (REQ-6, diverges): for a query NOT equal to a training point, sklearn
  `predict_proba` is the kernel-weighted average over ALL training rows; ferrolearn
  returns the single nearest training row's distribution — the two disagree whenever
  the second-nearest row carries non-negligible kernel weight.
- AC-6 (REQ-7/8/11): `LabelPropagation().fit(...)` exposes `transduction_` /
  `classes_` / `n_iter_`; ferrolearn `FittedLabelPropagation` exposes
  `labels_`/`label_distributions_`/`n_classes_` only, and `import ferrolearn;
  ferrolearn.LabelPropagation` does not exist (no binding).

## REQ status table

Binary (R-DEFER-2). `LabelPropagation` / `FittedLabelPropagation` /
`LabelPropagationKernel` are existing pub APIs re-exported at the crate root
(`ferrolearn-cluster/src/lib.rs:110`) — the only non-test consumer; grandfathered
(S5/R-DEFER-1). Cites use symbol anchors (ferrolearn) / `file:line` (sklearn 1.5.2,
commit 156ef14). Live oracle = installed sklearn 1.5.2, run from `/tmp`. Honest
assessment (R-HONEST-3): **TWO REQs SHIP** — the contiguous-label transduction
PARTITION (REQ-1) and the `classes_`/`n_classes`/label-VALUE mapping (REQ-4, fixed iter
124 / #999) — through the crate re-export; every numerical-value (label_distributions_),
attribute-surface, inductive-predict, default, binding, and substrate REQ DIVERGES.
Blocker numbers below are the real filed issues (#997–#1007; #999 CLOSED iter 124).

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (contiguous-label transduction PARTITION) | SHIPPED | impl `fn fit` (graph build → `fn row_normalize` → `fn propagate` → per-row argmax → `labels_`) recovers sklearn's per-blob class assignment for a CONTIGUOUS `0..k-1` label set on well-separated data (Probe A → `[0,0,0,0,1,1,1,1]`). Consumer: crate re-export `pub use label_propagation::{FittedLabelPropagation, LabelPropagation, LabelPropagationKernel}` (`lib.rs:110`). Guards (in-tree, contiguous labels): `test_label_propagation_basic` (labels[0..3]→0, labels[4..7]→1), `test_three_classes` (3 contiguous classes). Underclaim: PARTITION on CONTIGUOUS labels only — `label_distributions_` VALUES (REQ-2/3), non-contiguous labels (REQ-4), and predict semantics (REQ-6) diverge. |
| REQ-2 (`label_distributions_` value — RBF diag + zero-init) | NOT-STARTED | open prereq blocker #997. sklearn affinity diagonal `=1` (`rbf_kernel(X,X)`, `:147`/Probe D) and unlabeled rows START at zero (`y_static[unlabeled]=0`, `:282-289`); `line` middle rows `[0.55811,0.44189]`/`[0.49024,0.50976]` (AC-2). ferrolearn `fn build_rbf_affinity` zeroes the diagonal and `fn fit` inits unlabeled rows UNIFORM `1/n_classes` — different steady-state values. |
| REQ-3 (convergence — L1-at-start vs L2-at-end) | NOT-STARTED | open prereq blocker #998. sklearn checks `|Δ|.sum() < tol` at loop START (`:301`); ferrolearn `fn propagate` checks `sqrt(Σd²) < tol` at loop END — different stopping rule, different `n_iter_` (Probe F sklearn `n_iter_=4`) and final values. |
| REQ-4 (`classes_` mapping + `n_classes` on non-contiguous labels) | SHIPPED | impl `fn fit` now builds `classes_` = sorted unique non-(-1) labels, `n_classes = classes_.len()`, one-hot indexed by class POSITION, and maps the final argmax index through `classes_` — matching sklearn `classes_=unique(y)\{-1}` + `transduction_=classes_[argmax]` (`:272-274,333`). Guard: `divergence_req4_noncontiguous_classes_mapping` (`{0,2}` fixture → ferrolearn `n_classes()==2`, `labels ⊆ {0,2}` = sklearn `[0,0,0,0,2,2,2,2]`; was `n_classes()==3` with a phantom class). Fixed #999. |
| REQ-5 (`tol` default `1e-3`) | NOT-STARTED | open prereq blocker #1000. sklearn `tol=1e-3` (`:435`/Probe E); ferrolearn `fn new` sets `tol = F::from(1e-4)` — default divergence (R-DEV-2). |
| REQ-6 (`predict`/`predict_proba` kernel-weighted) | NOT-STARTED | open prereq blocker #1001. sklearn `predict_proba = rbf_kernel(X_train,X).T @ label_distributions_` row-normalized, `predict = classes_[argmax]` (`:190-191,218-231`). ferrolearn `fn predict` / `fn predict_proba` return the NEAREST training row's distribution only (the `best_j`/`best_dist` nearest-point loop) — a different inductive rule (AC-5). |
| REQ-7 (`transduction_`/`classes_`/`n_iter_`/`X_`/`n_features_in_` attrs) | NOT-STARTED | open prereq blocker #1002. sklearn exposes these (`:264,274,300,333-334`, docstring `:370-396`). ferrolearn `FittedLabelPropagation` exposes `labels_` (named, not `transduction_`), `label_distributions_`, `n_classes_` — no `classes_`, `n_iter_`, `X_`, or `n_features_in_` accessors. |
| REQ-8 (`ConvergenceWarning` + `n_iter_`) | NOT-STARTED | open prereq blocker #1003. sklearn warns `ConvergenceWarning` + `n_iter_ += 1` at `max_iter` (`:321-326`). ferrolearn `fn propagate` breaks silently with no `n_iter_` field and no warning channel. |
| REQ-9 (KNN connectivity graph + column-normalization) | NOT-STARTED | open prereq blocker #1004. sklearn knn `_get_kernel` → `kneighbors_graph(mode="connectivity")` (directed, `:156-157`), `_build_graph` normalizes by COLUMN sums (`affinity.sum(axis=0)`, `:457-461`). ferrolearn `fn build_knn_affinity` SYMMETRIZES (`w[i,j]=w[j,i]=1`) and `fn row_normalize` uses ROW sums — different operator for the directed KNN graph (coincides only on the symmetric RBF graph). |
| REQ-10 (validation / error ABI) | NOT-STARTED | open prereq blocker #1005. sklearn `check_classification_targets` + `_parameter_constraints` `gamma∈[0,∞)`, `n_neighbors∈(0,∞)`, `max_iter∈(0,∞)`, `tol∈[0,∞)` (`:110-118,258-265`), raising `InvalidParameterError`. ferrolearn `fn fit` raises `FerroError::InvalidParameter` (different TYPE/ABI) and rejects `gamma>0` (stricter than sklearn's `[0,∞)`); no `check_classification_targets` analog. |
| REQ-11 (PyO3 binding) | NOT-STARTED | open prereq blocker #1006. `grep -rn "LabelPropagation\|RsLabelProp" ferrolearn-python/` is EMPTY (Probe G) — `import ferrolearn` cannot reach `LabelPropagation`. The only non-test consumer of `fit`/`predict`/`predict_proba`/`score`/accessors is the crate re-export (`lib.rs:110`). |
| REQ-12 (ferray substrate) | NOT-STARTED | open prereq blocker #1007. `label_propagation.rs` imports `ndarray::{Array1, Array2}` + `num_traits::Float`; not migrated to `ferray-core` / `ferray::linalg` (R-SUBSTRATE-1/2). |

## Architecture

`label_propagation.rs` follows the unfitted/fitted split (CLAUDE.md naming):
`LabelPropagation<F>` (`kernel: LabelPropagationKernel`, `gamma: F`,
`n_neighbors: usize`, `max_iter: usize`, `tol: F`) →
`Fit<Array2<F>, Array1<isize>>` → `FittedLabelPropagation<F>` (private
`labels_: Array1<isize>`, `label_distributions_: Array2<F>`, `x_train_: Array2<F>`,
`n_classes_: usize`). Generic over `F: Float + Send + Sync + 'static`; every public
method returns `Result<_, FerroError>` (R-CODE-2). `FittedLabelPropagation` implements
`Predict<Array2<F>>` plus `predict_proba` and `score` methods. The
`LabelPropagationKernel` enum (`Rbf`/`Knn`) selects the affinity builder. There is no
`alpha` field — correct, since sklearn `LabelPropagation` pops `alpha` from its
`_parameter_constraints` (`:425-426`) and hard-codes `_variant="propagation"`
(`:423`); `alpha` belongs to `LabelSpreading` (the separate `label_spreading.rs`
unit).

**Fit path (`fn fit`).** Validates `n_samples>0`, `y.len()==n_samples`, kernel-param
positivity, and `n_labeled>0`. Then: (1) `n_classes = max(label)+1` (REQ-4
divergence — sklearn uses `len(unique(y)\{-1})`); (2) build the affinity
(`build_rbf_affinity` — diagonal ZERO, REQ-2; or `build_knn_affinity` — SYMMETRIZED,
REQ-9); (3) `row_normalize` to `T = D⁻¹W` (matches sklearn for symmetric RBF, differs
for directed KNN, REQ-9); (4) build `initial_y` — one-hot for labeled rows, UNIFORM
`1/n_classes` for unlabeled (REQ-2 divergence — sklearn zeroes them); (5) `propagate`
— iterate `F ← T·F`, clamp labeled rows to `initial_y`, row-normalize, stop on
end-of-loop L2 `< tol` (REQ-3 divergence); (6) per-row argmax → `labels_` (raw index,
REQ-4 divergence — no `classes_` remap).

**Predict path.** `Predict::predict` and `predict_proba` find the single NEAREST
training point (squared-Euclidean) and return its label / distribution row (REQ-6
divergence — sklearn does a kernel-weighted combination over ALL training rows via
`_get_kernel(X_, X)`). `score` is a `ClassifierMixin.score` analog skipping `y==-1`
test rows.

**Where sklearn and ferrolearn AGREE.** `kernel="rbf"` default, `gamma=20`,
`n_neighbors=7`, `max_iter=1000` (Probe E); `-1`-as-unlabeled convention; the
clamp-labeled-rows step; row-normalization equals sklearn's column-normalization on
the SYMMETRIC RBF graph (REQ-9 note); the contiguous-label argmax PARTITION on
well-separated data (Probe A, REQ-1).

**Where they DIVERGE (load-bearing):** the `classes_` remap / `n_classes` on
non-contiguous labels (REQ-4, a correctness bug); the RBF diagonal (REQ-2); the
unlabeled-row init (REQ-2); the convergence rule (REQ-3); `predict`/`predict_proba`
inductive semantics (REQ-6); the `tol` default (REQ-5); the attribute surface
(REQ-7); `ConvergenceWarning`/`n_iter_` (REQ-8); the KNN graph + normalization
direction (REQ-9); the error ABI (REQ-10); the missing PyO3 binding (REQ-11); the
ferray substrate (REQ-12).

**Consumer wiring.** The only non-test consumer is the crate re-export
(`pub use label_propagation::{FittedLabelPropagation, LabelPropagation,
LabelPropagationKernel}`, `ferrolearn-cluster/src/lib.rs:110`). There is no
`ferrolearn-python` binding (Probe G) and no other in-crate consumer.

## Verification

Library crate (green at baseline `f11e336c` for the existing partition behavior):
```
cargo test -p ferrolearn-cluster --lib label_propagation     # in-tree tests pass
cargo clippy -p ferrolearn-cluster --all-targets -- -D warnings
cargo fmt --all --check
```
The in-tree `#[test]`s (`test_label_propagation_basic`, `test_knn_kernel`,
`test_predict_on_new_data`, `test_all_labeled`, `test_no_labeled_error`,
`test_label_distributions_shape`, `test_n_classes`, `test_predict_shape_mismatch`,
`test_y_length_mismatch`, `test_empty_data`, `test_f32_support`, `test_three_classes`,
`test_default_constructor`, `test_invalid_gamma`) pin ferrolearn's current behavior on
CONTIGUOUS-label, well-separated fixtures — partition co-membership, shapes,
`n_classes` on contiguous data, error edges, f32 support. **None compares
`label_distributions_` VALUES, non-contiguous-label `transduction_`/`classes_`, the
`tol` default, or `predict_proba` against the live sklearn `LabelPropagation`
oracle**, so they stay green despite the divergences. In particular `test_n_classes`
only checks `n_classes()==2` on a contiguous `{0,1}` fixture — it does NOT exercise
the `{0,2}` correctness bug (REQ-4).

Live sklearn oracle (installed 1.5.2, run from `/tmp`) — the divergences a critic
should pin (R-CHAR-3 expected values). **Pin REQ-4 (non-contiguous `classes_`
mapping) FIRST — it is a CORRECTNESS bug and the cleanest single-fix pin:**
```
# REQ-4 (correctness bug — non-contiguous {0,2} labels)
python3 -c "import numpy as np; from sklearn.semi_supervised import LabelPropagation; \
X=np.array([[0.,0.],[0.1,0.],[0.,0.1],[0.1,0.1],[10.,10.],[10.1,10.],[10.,10.1],[10.1,10.1]]); \
y=np.array([0,-1,-1,-1,2,-1,-1,-1]); m=LabelPropagation().fit(X,y); \
print(m.classes_.tolist(), m.transduction_.tolist())"
# [0,2] [0,0,0,0,2,2,2,2]   -> assert ferrolearn n_classes()==2 AND labels ⊆ {0,2}: currently FAILS
# REQ-2/REQ-3 (label_distributions_ values — RBF diag / zero-init / L1)
python3 -c "import numpy as np; from sklearn.semi_supervised import LabelPropagation; \
X=np.array([[0.,0.],[0.3,0.],[0.6,0.],[1.0,0.]]); y=np.array([0,-1,-1,1]); \
m=LabelPropagation(gamma=1.0,tol=1e-3).fit(X,y); print(np.round(m.label_distributions_,6).tolist(), m.n_iter_)"
# [[1,0],[0.55811,0.44189],[0.49024,0.50976],[0,1]] 4   -> assert middle rows within 1e-5: FAILS
# REQ-5 (tol default)
python3 -c "from sklearn.semi_supervised import LabelPropagation; print(LabelPropagation().tol)"   # 0.001 (ferrolearn 1e-4)
```
A characterization pin (R-CHAR-3) for each NOT-STARTED REQ belongs in
`ferrolearn-cluster/tests/divergence_label_propagation.rs`, asserting the
live-sklearn expected values above and FAILING against current
`label_propagation.rs`.

ferrolearn-python (REQ-11 binding parity, after the binding lands):
```
cd /home/doll/ferrolearn/ferrolearn-python && maturin develop
PYTHONPATH=python python3 -m pytest tests/divergence_label_propagation.py -q
```
asserting `ferrolearn.LabelPropagation` exists and exposes `transduction_` /
`classes_` / `label_distributions_` / `n_iter_`, matching
`sklearn.semi_supervised.LabelPropagation` on the AC fixtures.

## Blockers (to open)

REQ-1 (contiguous-label PARTITION) SHIPS — no blocker. The remaining NOT-STARTED REQs
each need a `-l blocker` issue (dispatcher assigns real `#`-numbers, substituted into
the table above). Recommended order — **REQ-4 first (correctness bug, cleanest single
fix), then REQ-2/REQ-3 (the value-graph divergences):**

- #999 — REQ-4 (**correctness bug, fix FIRST**): use `classes_ = sorted(unique(y) \
  {-1})`, `n_classes = len(classes_)`, and map the argmax index THROUGH `classes_`
  for `labels_` (`_label_propagation.py:272-274,333`). Pin: `{0,2}` fixture →
  `n_classes()==2`, `labels ⊆ {0,2}`.
- #997 — REQ-2: set the RBF affinity diagonal to `1` (`rbf_kernel(X,X)`, `:147`) and
  zero-initialize unlabeled rows (`y_static[unlabeled]=0`, `:282-289`).
- #998 — REQ-3: convergence = L1 abs-sum at loop START (`:301`), not L2 at end.
- #1000 — REQ-5: change `fn new` `tol` default to `1e-3` (`:435`).
- #1001 — REQ-6: `predict_proba` = kernel-weighted combination over ALL training rows;
  `predict = classes_[argmax]` (`:190-191,218-231`).
- #1002 — REQ-7: expose `transduction_` (rename/alias `labels_`), `classes_`,
  `n_iter_`, `X_`, `n_features_in_` accessors.
- #1003 — REQ-8: `ConvergenceWarning` + `n_iter_` on `max_iter` (`:321-326`).
- #1004 — REQ-9: KNN connectivity graph (directed `kneighbors_graph`) + column-sum
  normalization (`:156-157,457-461`).
- #1005 — REQ-10: error ABI — `InvalidParameterError` analog + `gamma∈[0,∞)` bound +
  `check_classification_targets` (`:110-118,258-265`).
- #1006 — REQ-11: add the `ferrolearn-python` binding (fit / predict / predict_proba /
  transduction_ / classes_ / label_distributions_ / n_iter_).
- #1007 — REQ-12: migrate `label_propagation.rs` off `ndarray`/`num-traits` to
  `ferray-core` / `ferray::linalg` (R-SUBSTRATE).
