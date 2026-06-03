# Ranking Metrics (sklearn.metrics ranking functions)

<!--
tier: 3-component
status: draft
baseline-commit: a9c421de
upstream: scikit-learn 1.5.2 (commit 156ef14)
upstream-paths:
  - sklearn/metrics/_ranking.py   # dcg_score (:1602) + _dcg_sample_scores (:1468) + _tie_averaged_dcg (:1528); ndcg_score (:1770) + _ndcg_sample_scores (:1709); coverage_error (:1301); label_ranking_average_precision_score (:1202); label_ranking_loss (:1376); auc (:52); average_precision_score (:127); det_curve (:282); roc_auc_score (:420); precision_recall_curve (:879); roc_curve (:1053); top_k_accuracy_score (:1891)
ferrolearn-module: ferrolearn-metrics/src/ranking.rs
parity-ops: dcg_score, ndcg_score, coverage_error, label_ranking_average_precision_score, label_ranking_loss
crosslink-issue: 753
-->

## Summary

`ferrolearn-metrics/src/ranking.rs` mirrors the ranking-metric functions of
scikit-learn's `sklearn/metrics/_ranking.py`. It currently implements five of
that module's functions: `dcg_score`, `ndcg_score` (1D, gain-based) and the
multilabel `coverage_error`, `label_ranking_average_precision_score` (LRAP), and
`label_ranking_loss`. The remaining ranking surface mirrored by `_ranking.py`
— `auc`, `roc_auc_score`, `roc_curve`, `precision_recall_curve`,
`average_precision_score`, `det_curve`, `top_k_accuracy_score` — is **not absent
but MISPLACED**: those seven functions live in `ferrolearn-metrics/src/classification.rs`,
not in `ranking.rs`, even though sklearn groups them under `_ranking.py`.

**All five present functions DIVERGE from the live sklearn 1.5.2 oracle**, so —
under honest underclaim (R-HONEST-3) — every REQ in this doc is NOT-STARTED with
a concrete blocker. The two highest-value, fully-deterministic divergences a
critic should pin first:

1. **`dcg_score`/`ndcg_score` do NOT tie-average** (the default
   `ignore_ties=False` behavior). ferrolearn's `argsort_desc` resolves tied
   `y_score` by index, computing the `ignore_ties=True` answer. On the sklearn
   docstring example (`y_true=[10,0,0,1,5]`, `y_score=[1,0,0,0,1]`, `k=1`):
   sklearn `dcg_score` returns **7.5** (the average relevance of the tied top
   group, `(10+5)/2`), ferrolearn returns **10**. `ndcg_score` returns sklearn
   **0.75** vs ferrolearn **1.0**.
2. **`label_ranking_loss` normalizes by the wrong denominator** when a sample is
   degenerate (all-positive or all-negative). sklearn sets the degenerate
   sample's loss to `0.0` and averages over **all** `n_samples`
   (`_ranking.py:1463-1465`); ferrolearn skips degenerate samples and divides by
   `counted` (the non-degenerate count). On `y_true=[[1,0,0],[0,0,0]]`,
   `y_score=[[0.75,0.5,1.0],[0.9,0.8,0.7]]`: sklearn **0.25**, ferrolearn
   **0.5**.

Beyond those, every present function is **1D-or-2D-shape-restricted relative to
sklearn and omits `sample_weight`**; `dcg_score`/`ndcg_score` additionally omit
`log_base` and operate on 1D `y_true`/`y_score` rather than sklearn's
`(n_samples, n_labels)` 2D sample-mean contract, and `ndcg_score` lacks the
negative-`y_true` `ValueError` guard. The multilabel trio (`coverage_error`,
LRAP) is **value-correct on un-weighted inputs including ties and degenerate
rows** but still fails the `sample_weight` half of the sklearn signature, so it
is NOT-STARTED for the full contract.

## Algorithm (sklearn — the contract)

### `dcg_score` (`_ranking.py:1602`) over `_dcg_sample_scores` (`:1468`)

Signature: `dcg_score(y_true, y_score, *, k=None, log_base=2, sample_weight=None,
ignore_ties=False)`. Inputs are **2D `(n_samples, n_labels)`**; the return is the
`np.average` over samples (`:1701-1706`), weighted by `sample_weight`.

Discount vector (`:1511`): `discount[i] = 1 / (log(i + 2) / log(log_base))`, i.e.
`log_base`-based; with `k`, `discount[k:] = 0` (`:1512-1513`).

- `ignore_ties=True` (`:1514-1517`): `argsort(y_score)[:, ::-1]` (descending),
  gather `y_true` in that order, dot with `discount`.
- `ignore_ties=False` (**default**, `:1518-1524`): per-sample `_tie_averaged_dcg`
  (`:1528`). Tied `y_score` form a group; the group's gain is the **average
  `y_true` within the group** times the **sum of the discounts** of the ranks the
  group spans (`:1565-1573`). This is the McSherry-Najork tie-averaged DCG —
  averaging over all permutations of tied scores.

### `ndcg_score` (`:1770`) over `_ndcg_sample_scores` (`:1709`)

Signature: `ndcg_score(y_true, y_score, *, k=None, sample_weight=None,
ignore_ties=False)`. Per-sample DCG (`:1749`, honoring `ignore_ties`) divided by
the **ideal** DCG (`_dcg_sample_scores(y_true, y_true, k, ignore_ties=True)`,
`:1753`); all-irrelevant samples (ideal == 0) map to 0 (`:1754-1756`); then
`np.average` over samples weighted by `sample_weight` (`:1867-1876`, not shown in
the excerpt but mirrors `dcg_score`). `y_true` is checked for **negative values**
and raises `ValueError("ndcg_score should not be used on negative y_true values.")`.

### `coverage_error` (`:1301`)

Signature: `coverage_error(y_true, y_score, *, sample_weight=None)`. y_true must
be `multilabel-indicator` (`:1353-1355`). Mask `y_score` to the relevant labels,
take the per-row min relevant score (`:1360-1361`), `coverage = (y_score >=
y_min_relevant).sum(axis=1)` (`:1362`) — **ties are broken by maximal rank**
(every label with score `>=` the min relevant counts). Rows with no positive
label: the masked min is masked → `coverage.filled(0)` gives 0 (`:1363`). Return
`np.average(coverage, weights=sample_weight)` over **all** samples (`:1365`).

### `label_ranking_average_precision_score` (LRAP, `:1202`)

Signature: `label_ranking_average_precision_score(y_true, y_score, *,
sample_weight=None)`. y_score is negated and `rankdata(..., "max")` (max-tie
ranks) is taken per row (`:1263`, `:1277-1278`). For each sample: if
`relevant.size == 0 or == n_labels`, the per-sample score is `1.0`
(`:1271-1274`); else `aux = mean(L / rank)` where `L` is the relevant-only max-tie
rank and `rank` is the full max-tie rank at the relevant positions (`:1277-1279`).
Optionally multiply each `aux` by `sample_weight[i]` (`:1281-1282`); divide the
sum by `n_samples` or `sum(sample_weight)` (`:1285-1288`).

### `label_ranking_loss` (`:1376`)

Signature: `label_ranking_loss(y_true, y_score, *, sample_weight=None)`. Per
sample, count incorrectly-ordered (positive, negative) label pairs via a
unique-score binned reversed-rank cumulative count (`:1442-1455`), then normalize
by `(n_labels - n_positives) * n_positives` (`:1457-1459`). **Degenerate samples
(`n_positives == 0` or `== n_labels`) are set to `0.0`** (`:1461-1463`), then
`np.average(loss, weights=sample_weight)` over **all** `n_samples` (`:1465`).

### Misplaced surface mirrored by `_ranking.py` but living in `classification.rs`

`auc` (`:52`), `average_precision_score` (`:127`), `det_curve` (`:282`),
`roc_auc_score` (`:420`), `precision_recall_curve` (`:879`), `roc_curve`
(`:1053`), `top_k_accuracy_score` (`:1891`) are all defined in `_ranking.py`
upstream but implemented in `ferrolearn-metrics/src/classification.rs` (and
re-exported from the crate root from there). They are present in ferrolearn —
the divergence is a **module-placement mismatch** vs sklearn's source layout, not
an absent capability. Their numerical parity is owned by the `classification.rs`
design unit, not this one.

## ferrolearn (what exists)

- **1D gain metrics**: `pub fn dcg_score`, `pub fn ndcg_score`, both
  `<F: Float + Send + Sync + 'static>(y_true: &Array1<F>, y_score: &Array1<F>,
  k: Option<usize>) -> Result<F, FerroError>`. Signature exposes only `k`; no
  `log_base`, `sample_weight`, `ignore_ties`. `y_true`/`y_score` are **1D**.
- **Multilabel metrics**: `pub fn coverage_error`,
  `pub fn label_ranking_average_precision_score`, `pub fn label_ranking_loss`,
  all `<F>(y_true: &Array2<usize>, y_score: &Array2<F>) -> Result<F, FerroError>`.
  No `sample_weight`. `y_true` is `Array2<usize>` (binary indicator).
- **Internal helpers**: `fn argsort_desc` (descending sort, tie-break by index —
  the `ignore_ties=True` ordering), `fn compute_dcg` (fixed `log2` discount,
  `rel_i / log2(i + 2)`), `fn check_ranking_inputs` (shape/empty validation).
- **Consumers (non-test)**: crate re-export
  (`ferrolearn-metrics/src/lib.rs`: `pub use ranking::{coverage_error, dcg_score,
  label_ranking_average_precision_score, label_ranking_loss, ndcg_score}`). These
  are existing pub APIs (grandfathered, S5/R-DEFER-1); the re-export is the
  production-consumer surface. **No `ferrolearn-python` binding** exposes the
  ranking metrics (a binding gap, REQ-7).

## Requirements

- REQ-1: **`dcg_score` parity (R-DEV-1/2).** Match `dcg_score(y_true, y_score, *,
  k=None, log_base=2, sample_weight=None, ignore_ties=False)` (`:1602`):
  2D `(n_samples, n_labels)` with sample-mean, **tie-averaged DCG by default**
  (`_tie_averaged_dcg`, `:1528`), `log_base`, and `sample_weight`.
- REQ-2: **`ndcg_score` parity (R-DEV-1/2).** Match `ndcg_score(y_true, y_score,
  *, k=None, sample_weight=None, ignore_ties=False)` (`:1770`): 2D sample-mean,
  tie-averaging by default, ideal-DCG normalization, `sample_weight`, and the
  negative-`y_true` `ValueError` guard.
- REQ-3: **`coverage_error` parity (R-DEV-1/2).** Value logic matches sklearn
  (`:1301`) incl. tie max-rank and empty-row → 0; **missing `sample_weight`** is
  the remaining gap.
- REQ-4: **`label_ranking_average_precision_score` parity (R-DEV-1/2).** Value
  logic matches sklearn (`:1202`) incl. max-tie ranks and degenerate → 1.0;
  **missing `sample_weight`** is the remaining gap.
- REQ-5: **`label_ranking_loss` parity (R-DEV-1/2).** Two gaps: (a) **degenerate-
  sample normalization divides by the non-degenerate count `counted` instead of
  all `n_samples`** (`:1463-1465`) — a deterministic value divergence when any
  row is degenerate; (b) missing `sample_weight`.
- REQ-6: **Ranking surface placement / completeness (R-DEV-2).** `auc`,
  `roc_auc_score`, `roc_curve`, `precision_recall_curve`,
  `average_precision_score`, `det_curve`, `top_k_accuracy_score` are mirrored by
  `_ranking.py` but implemented in `classification.rs`. Their numerical parity is
  owned by the `classification.rs` unit; this REQ tracks the module-placement
  mismatch vs sklearn's source layout.
- REQ-7: **PyO3 binding (R-DEFER-1).** `import sklearn.metrics` exposes the
  ranking metrics; `ferrolearn-python` exposes no `dcg_score`/`ndcg_score`/
  `coverage_error`/LRAP/`label_ranking_loss` shim.
- REQ-8: **ferray substrate (R-SUBSTRATE).** `ranking.rs` imports
  `ndarray::{Array1, Array2}` + `num-traits`, not `ferray-core`.

## Acceptance criteria

- AC-1 (REQ-1 pin, R-CHAR-3): `dcg_score([[10,0,0,1,5]], [[1,0,0,0,1]], k=1)` must
  equal sklearn **7.5** (tie-averaged). ferrolearn returns **10** and FAILS.
- AC-2 (REQ-2 pin, R-CHAR-3): `ndcg_score([[10,0,0,1,5]], [[1,0,0,0,1]], k=1)`
  must equal sklearn **0.75**. ferrolearn returns **1.0** and FAILS. Plus:
  `ndcg_score` on negative `y_true` must raise the sklearn `ValueError`.
- AC-3 (REQ-1/2 shape+weight): `dcg_score`/`ndcg_score` accept 2D
  `(n_samples, n_labels)` and return the (optionally `sample_weight`-) averaged
  per-sample score; e.g. `dcg_score([[3,2,3,0,1,2],[1,0,2,3,1,0]],
  [[6,5,4,3,2,1],[1,2,3,4,5,6]])` == sklearn **5.1048...**, and with
  `sample_weight=[2,1]` == **5.6902...**.
- AC-4 (REQ-3): `coverage_error` matches sklearn on basic (**1.5**), all-tied
  (`[[1,0,1,0]]` all-0.5 → **4.0**), and empty-row (**0.5**) cases (ferrolearn
  already matches these); with `sample_weight=[2,1]` on the basic case ==
  sklearn **1.333...** (ferrolearn FAILS — no `sample_weight`).
- AC-5 (REQ-4): LRAP matches sklearn basic (**0.4166...**) and empty-row (**1.0**)
  (ferrolearn already matches); with `sample_weight=[2,1]` == sklearn **0.4444...**
  (ferrolearn FAILS — no `sample_weight`).
- AC-6 (REQ-5 pin, R-CHAR-3): `label_ranking_loss([[1,0,0],[0,0,0]],
  [[0.75,0.5,1.0],[0.9,0.8,0.7]])` must equal sklearn **0.25** (degenerate row
  averaged over `n_samples=2`). ferrolearn returns **0.5** (divides by
  `counted=1`) and FAILS.

## REQ status table

Binary (R-DEFER-2). All five present functions are existing pub APIs re-exported
at the crate root (the non-test production-consumer surface; grandfathered per
S5/R-DEFER-1). Cites use symbol anchors (ferrolearn) / `file:line`
(sklearn 1.5.2). Live oracle = installed sklearn 1.5.2, run from `/tmp`.

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (`dcg_score`) | NOT-STARTED | open prereq blocker #754. `pub fn dcg_score` ranks via `fn argsort_desc` (tie-break by index) + `fn compute_dcg` (fixed `log2`) — this is sklearn's `ignore_ties=True` path, NOT the default tie-averaged `_tie_averaged_dcg` (`_ranking.py:1518-1524`, `:1528`). It also takes **1D** `Array1` (sklearn is 2D `(n_samples,n_labels)` with sample-mean, `:1701-1706`) and omits `log_base`/`sample_weight`. Pin: `dcg_score([[10,0,0,1,5]],[[1,0,0,0,1]],k=1)` → sklearn **7.5**, ferro **10**. |
| REQ-2 (`ndcg_score`) | NOT-STARTED | open prereq blocker #755. `pub fn ndcg_score` reuses `fn argsort_desc`/`fn compute_dcg` (no tie-averaging), is **1D**, and omits `sample_weight` and the negative-`y_true` `ValueError` guard (`_ranking.py:1770`, ideal-DCG at `:1753`). Pin: `ndcg_score([[10,0,0,1,5]],[[1,0,0,0,1]],k=1)` → sklearn **0.75**, ferro **1.0**; negative `y_true` → sklearn raises, ferro does not. |
| REQ-3 (`coverage_error`) | NOT-STARTED | open prereq blocker #756. `pub fn coverage_error` value logic matches sklearn (`_ranking.py:1301`) incl. tie max-rank (`row_score[j] >= min_pos_score`) and empty-row → 0 (verified: basic 1.5, all-tied 4.0, empty-row 0.5 all match), but the signature omits `sample_weight` (sklearn `:1301`, `np.average(..., weights=sample_weight)` `:1365`). With `sample_weight=[2,1]` on the basic case sklearn returns 1.333..., ferro cannot express it. |
| REQ-4 (LRAP) | NOT-STARTED | open prereq blocker #757. `pub fn label_ranking_average_precision_score` value logic matches sklearn (`_ranking.py:1202`) incl. `>=`-as-max-tie-rank and degenerate → 1.0 (verified: basic 0.4166..., empty-row 1.0 match), but omits `sample_weight` (`:1281-1288`). With `sample_weight=[2,1]` sklearn returns 0.4444..., ferro cannot express it. |
| REQ-5 (`label_ranking_loss`) | NOT-STARTED | open prereq blocker #758. `pub fn label_ranking_loss` diverges on **degenerate-row normalization**: it accumulates `counted` (non-degenerate rows) and divides `totals / counted`, but sklearn sets degenerate-row loss to 0 and averages over **all** `n_samples` (`_ranking.py:1461-1465`). Pin: `label_ranking_loss([[1,0,0],[0,0,0]],[[0.75,0.5,1.0],[0.9,0.8,0.7]])` → sklearn **0.25**, ferro **0.5**. Also omits `sample_weight` (`:1465`). (Basic non-degenerate case 0.75 and all-tied 1.0 match.) |
| REQ-6 (ranking surface placement) | NOT-STARTED | open prereq blocker #759. `auc`/`roc_auc_score`/`roc_curve`/`precision_recall_curve`/`average_precision_score`/`det_curve`/`top_k_accuracy_score` are mirrored by `_ranking.py` (`:52`,`:420`,`:1053`,`:879`,`:127`,`:282`,`:1891`) but defined in `ferrolearn-metrics/src/classification.rs` (re-exported from there in `lib.rs`). Module-placement mismatch vs sklearn's source layout; their numerical parity is owned by the `classification.rs` unit. |
| REQ-7 (PyO3 binding) | NOT-STARTED | open prereq blocker #760. `ferrolearn-python` exposes no ranking-metric shim; `import ferrolearn` cannot call `dcg_score`/`ndcg_score`/`coverage_error`/LRAP/`label_ranking_loss` that `import sklearn.metrics` provides. |
| REQ-8 (ferray substrate) | NOT-STARTED | open prereq blocker #761. `ranking.rs` imports `ndarray::{Array1, Array2}` + `num_traits::Float`, not `ferray-core` (R-SUBSTRATE). |

## Architecture

`ranking.rs` has two families. The **1D gain metrics** (`dcg_score`,
`ndcg_score`) take `Array1<F>` and rank with `fn argsort_desc` — a descending
sort with index tie-break — then accumulate `fn compute_dcg`'s fixed
`rel_i / log2(i + 2)` discount. This is structurally the sklearn
`ignore_ties=True` path; sklearn's **default** averages tied groups
(`_tie_averaged_dcg`, `:1528`) and operates on 2D `(n_samples, n_labels)`,
returning the per-sample mean (`np.average(..., weights=sample_weight)`).
ferrolearn's single-sample 1D shape, fixed `log2` discount, and index tie-break
are three independent gaps from the sklearn contract (REQ-1/2).

The **multilabel metrics** (`coverage_error`, `label_ranking_average_precision_score`,
`label_ranking_loss`) take `Array2<usize>` (binary indicator) and `Array2<F>`,
validated by `fn check_ranking_inputs` (shape + non-empty). `coverage_error` and
LRAP reproduce sklearn's value logic faithfully — the `>= min_pos_score` /
`>= pos_score` comparisons reproduce `rankdata(..., "max")` max-tie ranks, and
degenerate rows are handled with sklearn's conventions (coverage 0 / LRAP 1.0).
`label_ranking_loss` reproduces the pairwise bad-pair count but **normalizes by
`counted` (non-degenerate sample count) rather than `n_samples`** — the one
deterministic value divergence in the multilabel trio (REQ-5). None of the three
accept `sample_weight`, which sklearn requires (`np.average(..., weights=...)`),
so all three are NOT-STARTED for the full signature.

**Invariants held:** shape/empty validation; coverage tie max-rank and empty-row
→ 0; LRAP max-tie precision and degenerate → 1.0; ndcg all-irrelevant → 0.
**Invariants NOT held vs sklearn:** tie-averaged DCG/NDCG default; 2D sample-mean
for DCG/NDCG; `log_base`; `sample_weight` (all five); ndcg negative-`y_true`
guard; `label_ranking_loss` degenerate-row denominator. **Placement:** seven
`_ranking.py` functions live in `classification.rs` (REQ-6).

## Verification

Library crate (green at baseline `a9c421de` for the existing — narrower —
contract):
```
cargo test -p ferrolearn-metrics --lib ranking   # existing ranking unit tests pass
cargo clippy -p ferrolearn-metrics --all-targets -- -D warnings
cargo fmt --all --check
```
The existing `#[test]`s pin only ferrolearn's narrower (1D, no-tie-averaging,
no-`sample_weight`) behavior; they do NOT establish sklearn parity, so they do
not make any REQ SHIPPED.

Live sklearn oracle (installed 1.5.2, run from `/tmp`) — the divergences:
```
# REQ-1 (dcg tie-averaging): sklearn 7.5 vs ferro 10
python3 -c "import numpy as np; from sklearn.metrics import dcg_score; \
print(dcg_score(np.array([[10,0,0,1,5]]), np.array([[1,0,0,0,1]]), k=1))"   # 7.5
# REQ-2 (ndcg tie-averaging): sklearn 0.75 vs ferro 1.0; negative-y_true guard
python3 -c "import numpy as np; from sklearn.metrics import ndcg_score; \
print(ndcg_score(np.array([[10,0,0,1,5]]), np.array([[1,0,0,0,1]]), k=1))"   # 0.75
python3 -c "import numpy as np; from sklearn.metrics import ndcg_score; \
ndcg_score(np.array([[-1,2,3]]), np.array([[1,2,3]]))"   # ValueError: negative y_true
# REQ-1/2 (2D sample-mean + sample_weight)
python3 -c "import numpy as np; from sklearn.metrics import dcg_score; \
yt=np.array([[3,2,3,0,1,2],[1,0,2,3,1,0]]); ys=np.array([[6,5,4,3,2,1],[1,2,3,4,5,6]]); \
print(dcg_score(yt,ys), dcg_score(yt,ys,sample_weight=np.array([2.,1.])))"   # 5.1048..., 5.6902...
# REQ-5 (label_ranking_loss degenerate denominator): sklearn 0.25 vs ferro 0.5
python3 -c "import numpy as np; from sklearn.metrics import label_ranking_loss; \
print(label_ranking_loss(np.array([[1,0,0],[0,0,0]]), np.array([[0.75,0.5,1.0],[0.9,0.8,0.7]])))"  # 0.25
# REQ-3/4 (sample_weight gap): coverage 1.333..., lrap 0.4444...
python3 -c "import numpy as np; from sklearn.metrics import coverage_error, \
label_ranking_average_precision_score as lrap; \
yt=np.array([[1,0,0],[0,1,1]]); ys=np.array([[1,0,0],[0,1,1]]); \
print(coverage_error(yt,ys,sample_weight=np.array([2.,1.])))"   # 1.333...
```
Every REQ is NOT-STARTED; each carries an open prereq blocker. A characterization
pin (R-CHAR-3) for each belongs in
`ferrolearn-metrics/tests/divergence_ranking.rs`, asserting the live-sklearn
expected values above and FAILING against current `ranking.rs`.

## Blockers to open

- #754 — REQ-1 (`dcg_score`): no tie-averaging (default `ignore_ties=False`,
  `_ranking.py:1518-1524`/`:1528`); 1D not 2D sample-mean (`:1701-1706`); no
  `log_base`/`sample_weight`. Pin: tied `k=1` → sklearn 7.5, ferro 10.
- #755 — REQ-2 (`ndcg_score`): same tie/2D/`sample_weight` gaps as #754 plus the
  missing negative-`y_true` `ValueError` guard. Pin: tied `k=1` → sklearn 0.75,
  ferro 1.0.
- #756 — REQ-3 (`coverage_error`): value-correct but missing `sample_weight`
  (`_ranking.py:1365`).
- #757 — REQ-4 (LRAP): value-correct but missing `sample_weight`
  (`_ranking.py:1281-1288`).
- #758 — REQ-5 (`label_ranking_loss`): degenerate-row normalization divides by
  `counted` not `n_samples` (`_ranking.py:1461-1465`) — pin: degenerate row →
  sklearn 0.25, ferro 0.5; also missing `sample_weight`.
- #759 — REQ-6: `auc`/`roc_auc_score`/`roc_curve`/`precision_recall_curve`/
  `average_precision_score`/`det_curve`/`top_k_accuracy_score` mirrored by
  `_ranking.py` but implemented in `classification.rs` (module-placement
  mismatch).
- #760 — REQ-7: no `ferrolearn-python` ranking-metric binding.
- #761 — REQ-8: migrate `ranking.rs` off `ndarray`/`num-traits` to the ferray
  substrate (R-SUBSTRATE).
