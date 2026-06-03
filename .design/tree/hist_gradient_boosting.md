# Histogram Gradient Boosting (HistGradientBoostingClassifier / HistGradientBoostingRegressor)

<!--
tier: 3-component
status: draft
baseline-commit: 9dabd1b5
upstream: scikit-learn 1.5.2 (commit 156ef14)
upstream-paths:
  - sklearn/ensemble/_hist_gradient_boosting/gradient_boosting.py   # BaseHistGradientBoosting.fit; HistGradientBoostingRegressor (:1430) __init__ defaults loss='squared_error'/learning_rate=0.1/max_iter=100/max_leaf_nodes=31/max_depth=None/min_samples_leaf=20/l2_regularization=0.0/max_features=1.0/max_bins=255/early_stopping='auto'/validation_fraction=0.1/n_iter_no_change=10/tol=1e-7 (:1706-1725); HistGradientBoostingClassifier (:1817) loss='log_loss' (:2086); grower construction max_leaf_nodes/max_depth/min_samples_leaf/l2_regularization (:911-914)
  - sklearn/ensemble/_hist_gradient_boosting/binning.py   # _find_binning_thresholds (:23): n_unique<=max_bins -> midpoints distinct[:-1]+distinct[1:] *0.5 (:53-55); else np.percentile(method='midpoint') (:61-63); _BinMapper subsample=2e5 with random_state (:157,:199-200)
  - sklearn/ensemble/_hist_gradient_boosting/grower.py   # compute_node_value = -sum_gradients/(sum_hessians+l2) (via splitting.compute_node_value); _apply_shrinkage leaf.value *= shrinkage (:391-401); best-first heap split_next (:11,:136-137); min_gain_to_split default 0 (:253)
  - sklearn/ensemble/_hist_gradient_boosting/splitting.pyx   # _split_gain = loss_current - loss_left - loss_right (:1118-1124); _loss_from_value = sum_gradient*value (:1131-1138) with value=-G/(H+l2) -> loss=-G^2/(H+l2); find_node_split best_gain scan (:667-717)
ferrolearn-module: ferrolearn-tree/src/hist_gradient_boosting.rs
parity-ops: HistGradientBoostingClassifier, HistGradientBoostingRegressor
crosslink-issue: 745
-->

## Summary

`ferrolearn-tree/src/hist_gradient_boosting.rs` mirrors scikit-learn's
`sklearn.ensemble.HistGradientBoostingRegressor` (`gradient_boosting.py:1430`)
and `HistGradientBoostingClassifier` (`:1817`) — LightGBM-style gradient
boosting that (1) discretises every feature into `max_bins` bins once up front,
then (2) each boosting round accumulates per-bin gradient/hessian histograms,
grows a tree best-first by split gain `G²/(H+λ)`, sets each leaf to
`-G/(H+λ)`, scales by `learning_rate`, and adds it to the running raw scores.

**HEADLINE divergence (R-DEV-1) — binning thresholds differ.** sklearn's
`_BinMapper` places a feature's bin thresholds at the **midpoints between
consecutive distinct values** when `n_unique <= max_bins`
(`binning.py:53-55`), and at `np.percentile(..., method="midpoint")` otherwise
(`:61-63`). ferrolearn's `compute_bin_edges` instead uses **quantile-position
linear interpolation** over the distinct values and appends the **raw maximum**
as the final edge — neither the midpoint rule nor `np.percentile`. The
threshold *locations* therefore diverge even on tiny data, which moves
prediction boundaries (live-pinned below). This is the linchpin for end-to-end
parity (REQ-2, REQ-7).

**Secondary divergence (R-DEV-2) — parameter name.** ferrolearn exposes the
boosting-round count as `n_estimators` / `with_n_estimators`. sklearn's
parameter is `max_iter` (`gradient_boosting.py:1710`); sklearn HGB has **no**
`n_estimators` parameter. The default *value* (100) matches.

## Requirements

- REQ-1: Constructor parameter surface and default *values* match sklearn:
  `learning_rate=0.1`, `max_iter=100`, `max_leaf_nodes=31`, `max_depth=None`,
  `min_samples_leaf=20`, `l2_regularization=0.0`, `max_bins=255`,
  loss `squared_error` (regressor) / `log_loss` (classifier).
  Parameter *name* `max_iter` (sklearn) vs `n_estimators` (ferrolearn).
- REQ-2: Feature binning thresholds match sklearn `_BinMapper.bin_thresholds_`
  (midpoint rule for `n_unique <= max_bins`; percentile-midpoint otherwise).
- REQ-3: Per-bin gradient/hessian histogram accumulation plus the
  parent-minus-sibling subtraction trick.
- REQ-4: Split gain `G_L²/(H_L+λ) + G_R²/(H_R+λ) − G_p²/(H_p+λ)`.
- REQ-5: Leaf value `-G/(H+λ)` then per-leaf shrinkage by `learning_rate`.
- REQ-6: Best-first (leaf-wise) growth bounded by `max_leaf_nodes`, honoring
  `min_samples_leaf` and `max_depth`.
- REQ-7: End-to-end prediction parity with sklearn on a small dataset
  (`early_stopping=False`, equal hyperparameters) within a documented tolerance.
- REQ-8: `early_stopping='auto'` (enabled only when `n_samples > 10000`) plus
  the internal validation split (`validation_fraction`, `n_iter_no_change`,
  `tol`).
- REQ-9: `max_features` per-node feature subsampling.
- REQ-10: Native missing-value (NaN) routing: a dedicated missing bin and a
  learned left/right direction per split, matching sklearn's
  `missing_values_bin_idx` routing.
- REQ-11: Binary and multiclass log-loss classification with
  `predict`/`predict_proba`/`decision_function`, matching sklearn's baseline
  raw predictions (`fit_intercept_only`) and label ordering.

## Acceptance criteria

- AC-1: `HistGradientBoostingRegressor::new()` and
  `HistGradientBoostingClassifier::new()` report `n_estimators == 100`,
  `learning_rate == 0.1`, `max_leaf_nodes == Some(31)`, `min_samples_leaf == 20`,
  `max_bins == 255`, `l2_regularization == 0.0` (REQ-1).
- AC-2: For `X = [0,2,4,6]ᵀ`, ferrolearn's feature bin thresholds equal
  `[1.0, 3.0, 5.0]` (sklearn `_BinMapper(n_bins=256).fit(X).bin_thresholds_[0]`).
  **Currently FAILS** (REQ-2).
- AC-3: `subtract_histograms(parent, sibling)` equals
  `build_histograms(child_indices)` bin-for-bin (REQ-3).
- AC-4: Split gain equals `G_L²/(H_L+λ)+G_R²/(H_R+λ)−G_p²/(H_p+λ)` (REQ-4).
- AC-5: A pure single-group leaf returns `-ΣG/(ΣH+λ)`; the running update adds
  `learning_rate * leaf_value` (REQ-5).
- AC-6: With `max_leaf_nodes=Some(k)`, no tree exceeds `k` leaves; the
  highest-gain pending node is split first (REQ-6).
- AC-7: `HistGradientBoostingRegressor(loss='squared_error', max_iter=10,
  max_leaf_nodes=7, min_samples_leaf=1, learning_rate=0.1,
  early_stopping=False, random_state=0).fit(X,y).predict(X)` matches ferrolearn
  (equal params) within 1e-6 on `X=[1..8]ᵀ, y=[1,1,1,1,5,5,5,5]`; AND for
  `X=[0,2,4,6]ᵀ, y=[10,20,30,40]`, test points `[1.2, 4.8]` predict the same
  leaf as sklearn (REQ-7). **Second case currently FAILS.**
- AC-8: With `n_samples <= 10000` and `early_stopping='auto'`, training runs
  all `max_iter` rounds (no validation split) (REQ-8).
- AC-9: With `max_features < 1.0`, each node considers a sampled feature subset
  (REQ-9).
- AC-10: A feature with NaN trains and predicts finite values; NaN routes to
  the split direction sklearn learns (REQ-10).
- AC-11: Binary/multiclass `predict_proba` rows sum to 1 and the argmax label
  ordering follows sorted `classes_` (REQ-11).

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (defaults+param surface) | SHIPPED (value match) / NOT-STARTED (name) | Default *values* match: `pub fn new in hist_gradient_boosting.rs` sets `n_estimators:100, learning_rate:0.1, min_samples_leaf:20, max_bins:255, l2_regularization:0.0, max_leaf_nodes:Some(31)`, mirroring sklearn `gradient_boosting.py:1706-1725` (`max_iter=100`, `max_leaf_nodes=31`, …). Non-test consumer: `RsHistGradientBoostingRegressor::new` in `ferrolearn-python/src/extras.rs` defaults `n_estimators=100, learning_rate=0.1`. Tests: `test_hgbr_default_trait`, `test_hgbc_default_trait`. The **R-DEV-2 parameter NAME** `n_estimators`/`with_n_estimators` diverges from sklearn `max_iter` (sklearn HGB has no `n_estimators`) — single-file-fixer-able rename; open prereq blocker #746. |
| REQ-2 (binning thresholds) | NOT-STARTED | open prereq blocker #747. `fn compute_bin_edges in hist_gradient_boosting.rs` uses quantile-position interpolation `unique[lo]*(1-t)+unique[hi]*t` and appends the raw max as the last edge; sklearn `binning.py:53-55` uses **midpoints** `(distinct[i]+distinct[i+1])/2` (n_unique<=max_bins) and `np.percentile(method='midpoint')` otherwise (`:61-63`). Live: for `X=[0,2,4,6]`, sklearn `bin_thresholds_=[1.0,3.0,5.0]`, ferrolearn first edge ≈1.5 / last edge =6.0. Deep fix (rewrite binning + map_to_bin convention). |
| REQ-3 (histograms + subtraction trick) | SHIPPED | impl `fn build_histograms` + `fn subtract_histograms in hist_gradient_boosting.rs` accumulate per-bin `grad_sum/hess_sum/count`; subtraction trick mirrors sklearn `grower.py` smaller-child-then-subtract. Non-test consumer: both reached via `fn build_hist_tree_best_first` called from `HistGradientBoostingRegressor::fit`, surfaced through `RsHistGradientBoostingRegressor::fit` in `ferrolearn-python/src/extras.rs`. Test: `test_subtraction_trick` (subtraction equals direct build). Verification: `cargo test -p ferrolearn-tree hist_gradient` → 39 passed, 0 failed. |
| REQ-4 (split gain G²/(H+λ)) | SHIPPED | impl `fn find_best_split_from_histograms in hist_gradient_boosting.rs`: `gain = lg*lg/(lh+l2) + rg*rg/(rh+l2) − parent_gain`, mirroring sklearn `splitting.pyx:1118-1138` (`gain = loss_current − loss_left − loss_right`, `loss = sum_gradient*value`, `value=-G/(H+l2)` ⇒ `loss=-G²/(H+l2)`). Non-test consumer via `fit` → PyO3 `fit`. Tests exercise it end-to-end (`test_hgbr_simple_least_squares`). Verification: 39 passed. |
| REQ-5 (leaf -G/(H+λ) + shrinkage) | SHIPPED | impl `fn compute_leaf_value in hist_gradient_boosting.rs` returns `-grad_sum/(hess_sum+l2_reg)`, mirroring sklearn `compute_node_value` (`grower.py`/`splitting.pyx`); shrinkage applied as `f_vals[i] += lr * value` in `HistGradientBoostingRegressor::fit` (sklearn `_apply_shrinkage` `grower.py:391-401`). Non-test consumer: PyO3 `fit`/`predict`. Tests: `test_hgbr_l2_regularization`, `test_hgbr_simple_least_squares`. Verification: 39 passed. |
| REQ-6 (best-first grower / max_leaf_nodes / min_samples_leaf) | SHIPPED | impl `fn build_hist_tree_best_first in hist_gradient_boosting.rs` pops the highest-gain pending node (`max_by` on `gain`), stops at `max_leaf_nodes`, gates children on `>= 2*min_samples_leaf` and `max_depth`, mirroring sklearn best-first heap (`grower.py:11,136-137`). Non-test consumer: default `max_leaf_nodes=Some(31)` path through `fit` → PyO3. Test: `test_hgbr_max_leaf_nodes`. Verification: 39 passed. NOTE: sklearn pops by min-`-gain` (max gain) AND applies `min_gain_to_split` (default 0, `grower.py:253`); ferrolearn additionally requires `gain > 0` strictly. |
| REQ-7 (end-to-end small-data parity) | NOT-STARTED | open prereq blocker #748 (depends on #747). Live: `X=[1..8]ᵀ, y=[1,1,1,1,5,5,5,5]`, `max_iter=10, max_leaf_nodes=7, min_samples_leaf=1, lr=0.1, early_stopping=False` — ferrolearn `[1.6973568802,…,4.3026431198]` MATCHES sklearn `[1.697357,…,4.302643]` to ~1e-9 (binning coincides on this split). But `X=[0,2,4,6]ᵀ, y=[10,20,30,40]`, test points `[1.2,4.8]` → ferrolearn `[10.0,40.0]` vs sklearn `[20.0,30.0]`. Root cause: REQ-2 binning thresholds (`1.2 > 1.0` ⇒ sklearn bin 1; `1.2 <= ~1.5` ⇒ ferrolearn bin 0). Deep fix (resolve REQ-2 first). This is the cleanest deterministic divergence to pin. |
| REQ-8 (early_stopping + validation split) | NOT-STARTED | open prereq blocker #749. ferrolearn has no `early_stopping`, `validation_fraction`, `n_iter_no_change`, or `tol` fields and always runs all `n_estimators` rounds; sklearn `gradient_boosting.py:1721-1725` defaults `early_stopping='auto'` (on only when `n_samples>10000`), `validation_fraction=0.1`. Deep: the validation split is an RNG boundary (numpy `train_test_split` shuffle vs Rust draw diverge) and changes the fitted forest. |
| REQ-9 (max_features) | NOT-STARTED | open prereq blocker #750. ferrolearn has no `max_features` field; every split scans all features. sklearn `gradient_boosting.py:1715` defaults `max_features=1.0` (all features), so the *default* path agrees, but `<1.0` per-node subsampling (another RNG boundary) is absent. |
| REQ-10 (missing-value / NaN routing) | SHIPPED (structural) / NOT-STARTED (exact parity) | impl: `fn map_to_bin` assigns NaN to `NAN_BIN`; `fn find_best_split_from_histograms` tries `nan_goes_left ∈ {true,false}` and picks the higher-gain direction; `fn traverse_hist_tree` routes NaN accordingly — structurally mirrors sklearn `missing_values_bin_idx` + learned direction (`grower.py:228-286`). Non-test consumer: PyO3 `fit`/`predict`. Tests: `test_hgbr_nan_handling`, `test_hgbc_nan_handling` (finite predictions). Exact-value parity untested (gated by REQ-2 binning + sklearn's tie-break placing the missing bin at index `n_bins`); narrow follow-up blocker #751. |
| REQ-11 (binary + multiclass log-loss) | SHIPPED (structural) / NOT-STARTED (baseline parity) | impl: `fn fit_binary` (log-odds init, `grad=p−y`, `hess=p(1−p)`) and `fn fit_multiclass` (K trees/round, `softmax_matrix`, `grad=p_k−y_k`); `predict_proba`/`decision_function`/`predict` present; `classes_` sorted via `HasClasses`. Mirrors sklearn `loss='log_loss'` (`gradient_boosting.py:2086`). Non-test consumer: `RsHistGradientBoostingClassifier` in `ferrolearn-python/src/extras.rs`. Tests: `test_hgbc_binary_simple`, `test_hgbc_multiclass`, `test_hgbc_has_classes`. BASELINE divergence — multiclass init uses `ln(p_k)` (log-prior); sklearn uses `HalfMultinomialLoss.fit_intercept_only` (log-prior shifted to sum-zero), and binary uses `HalfBinomialLoss.fit_intercept_only`. Exact raw-score parity is gated by REQ-2 and the baseline; blocker #752. |

## Architecture

**Two estimators, shared internals.** `HistGradientBoostingRegressor<F>` and
`HistGradientBoostingClassifier<F>` (both `#[derive(Serialize, Deserialize)]`)
hold the hyperparameters; their `Fit` impls produce
`FittedHistGradientBoostingRegressor<F>` /
`FittedHistGradientBoostingClassifier<F>` which store `bin_infos`, the per-class
`init`, `learning_rate`, the tree forest, and normalised
`feature_importances`. This is the unfitted/fitted split per CLAUDE.md naming.

**Binning (REQ-2 — divergent).** `compute_bin_edges` builds a
`FeatureBinInfo<F> { edges, n_bins, has_nan }` per feature; `map_to_bin`
binary-searches the first edge `>= value` (NaN ⇒ `NAN_BIN = u16::MAX`).
sklearn’s contract (`binning.py:18` docstring) is `thresholds[i-1] < x <=
thresholds[i]`; sklearn places thresholds at distinct-value **midpoints**
(`:53-55`). ferrolearn’s `edges` are quantile-interpolated with a trailing
raw-max edge — a different threshold set. sklearn additionally subsamples
`200000` rows with a `random_state` before thresholding (`binning.py:157,199`),
an RNG boundary ferrolearn does not model (irrelevant below the subsample size).

**Per-round tree building.** For each of `n_estimators` rounds: compute
gradients/hessians (`grad=F−y, hess=1` for least-squares; `p−y, p(1−p)` for
log-loss), call `build_hist_tree` (best-first via `build_hist_tree_best_first`
when `max_leaf_nodes.is_some()`, else depth-first `build_hist_tree_recursive`),
then add `learning_rate * leaf_value` to the running raw scores. Histograms use
`build_histograms` + the `subtract_histograms` trick on the smaller child.
Gain (`find_best_split_from_histograms`) and leaf value (`compute_leaf_value`)
follow sklearn’s XGBoost equations (`splitting.pyx:1118-1138`).

**Classification.** `fit_binary` trains one tree sequence on log-odds residuals;
`fit_multiclass` trains `K` trees per round and couples them through
`softmax_matrix`. `predict_proba` applies the logistic/softmax link;
`decision_function` returns raw scores; `predict` argmaxes. Baseline `init`
differs from sklearn’s `fit_intercept_only` (REQ-11).

**Not modeled.** No `early_stopping`/validation split (REQ-8), no `max_features`
(REQ-9) — both are RNG-boundary features whose absence is honest NOT-STARTED.

**Substrate (R-SUBSTRATE).** The module computes on `ndarray::Array2`/`Array1`,
not `ferray-core`; this is grandfathered for the not-yet-migrated tree crate and
tracked workspace-wide. No new wrong-substrate APIs are introduced by this doc.

## Verification

Commands establishing the SHIPPED claims (run at baseline `9dabd1b5`):

- `cargo test -p ferrolearn-tree hist_gradient` → **39 passed, 0 failed**
  (covers REQ-1, REQ-3, REQ-4, REQ-5, REQ-6, REQ-10, REQ-11 structurally).
- REQ-7 (partial) live oracle, MATCH to ~1e-9:
  `python3 -c "from sklearn.ensemble import HistGradientBoostingRegressor; import numpy as np; X=np.arange(1,9.).reshape(-1,1); y=np.array([1.,1,1,1,5,5,5,5]); print(HistGradientBoostingRegressor(loss='squared_error',max_iter=10,max_leaf_nodes=7,min_samples_leaf=1,learning_rate=0.1,early_stopping=False,random_state=0).fit(X,y).predict(X))"`
  → `[1.697357 … 4.302643]`; ferrolearn (same params) → `[1.6973568802 … 4.3026431198]`.
- REQ-2 / REQ-7 (the divergence pin) live oracle, MISMATCH:
  `python3 -c "from sklearn.ensemble._hist_gradient_boosting.binning import _BinMapper; import numpy as np; print(_BinMapper(n_bins=256).fit(np.array([0.,2,4,6]).reshape(-1,1)).bin_thresholds_[0])"`
  → `[1. 3. 5.]` (ferrolearn edges ≈ `[1.5, 3.0, 6.0]`); and end-to-end on
  `X=[0,2,4,6]ᵀ, y=[10,20,30,40]` test `[1.2,4.8]` → sklearn `[20,30]` vs
  ferrolearn `[10,40]`. **This is the cleanest deterministic pin for the critic.**

A REQ that is NOT-STARTED has no green verification by construction; its blocker
number names the missing piece.

## Blockers to open

- #746 — REQ-1 name: rename `n_estimators`/`with_n_estimators` → `max_iter`/`with_max_iter` to match sklearn (single-file fixer; `-l blocker`).
- #747 — REQ-2: rewrite `compute_bin_edges`/`map_to_bin` to sklearn’s midpoint + percentile-midpoint thresholds (deep; linchpin).
- #748 — REQ-7: end-to-end small-data prediction parity (depends on #747).
- #749 — REQ-8: `early_stopping='auto'` + validation split (`validation_fraction`/`n_iter_no_change`/`tol`); RNG boundary.
- #750 — REQ-9: `max_features` per-node feature subsampling; RNG boundary.
- #751 — REQ-10: exact NaN-routing/missing-bin-index parity (depends on #747).
- #752 — REQ-11: classifier baseline raw-prediction parity (`fit_intercept_only`) + multiclass sum-zero init.
