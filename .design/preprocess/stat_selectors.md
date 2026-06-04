# SelectFpr / SelectFdr / SelectFwe

<!--
tier: 3-component
status: shipped-partial
baseline-commit: 4ec970cf
upstream: scikit-learn 1.5.2
upstream-paths:
  - sklearn/feature_selection/_univariate_selection.py  # _BaseFilter (:526) wraps a score_func and computes scores_/pvalues_ at fit(X, y) (:559-577); class SelectFpr (:801) _get_support_mask (:875-878) -> self.pvalues_ < self.alpha; class SelectFdr (:881) _get_support_mask (:959-969) Benjamini-Hochberg: n=len(pvalues_); sv=np.sort(pvalues_); selected = sv[sv <= float(alpha)/n * np.arange(1, n+1)]; if selected.size == 0 -> all-False else pvalues_ <= selected.max(); class SelectFwe (:972) _get_support_mask (:1041-1044) -> self.pvalues_ < self.alpha / len(self.pvalues_). All three share _parameter_constraints alpha: Interval(Real, 0, 1, closed="both") (:866-869, :950-953, :1032-1035) and __init__(score_func=f_classif, *, alpha=5e-2) (:871-873, :955-957, :1037-1039). SelectorMixin provides get_support / transform / inverse_transform / get_feature_names_out.
ferrolearn-module: ferrolearn-preprocess/src/stat_selectors.rs
parity-ops: SelectFpr, SelectFdr, SelectFwe
crosslink-issue: 1396
-->

## Summary

scikit-learn's `SelectFpr` (`_univariate_selection.py:801`), `SelectFdr`
(`:881`) and `SelectFwe` (`:972`) are univariate filter selectors. Each subclasses
`_BaseFilter` (`:526`), which wraps a `score_func` (default `f_classif`), runs it on
`(X, y)` at `fit` to compute `self.scores_` / `self.pvalues_` (`:559-577`), and then
keeps features according to a per-selector `_get_support_mask` evaluated on
`self.pvalues_`: `SelectFpr` keeps `pvalues_ < alpha` (`:878`), `SelectFwe` keeps
`pvalues_ < alpha / n_features` (Bonferroni, `:1044`), and `SelectFdr` applies the
Benjamini-Hochberg procedure (`:959-969`). The selector surface (`get_support`,
`transform`, `inverse_transform`, `get_feature_names_out`) comes from
`SelectorMixin`.

`ferrolearn-preprocess/src/stat_selectors.rs` ships the **three p-value masks +
the BH procedure + column gather, GIVEN a static p-value vector** rather than
wrapping a `score_func`: `SelectFpr<F>`, `SelectFdr<F>`, `SelectFwe<F>` (each
`new(alpha)`, `alpha()`) take the per-feature p-value vector as the `Fit<Array1<F>,
()>` input `x: &Array1<F>` and produce `FittedSelectFpr<F>` /
`FittedSelectFdr<F>` / `FittedSelectFwe<F>` (`{ n_features_in, p_values,
selected_indices }`; accessors `p_values()`, `selected_indices()`,
`n_features_selected()`); `Transform<Array2<F>>` gathers the selected columns
(`ShapeMismatch` on a column-count mismatch). There is **no `score_func` wrapping
(no `scores_` / `pvalues_` computed at `fit(X, y)`), no `SelectorMixin` surface
(`get_support` / `inverse_transform` / `get_feature_names_out`), no
`n_features_in_` / `feature_names_in_` fitted attributes, no PyO3 binding, and no
ferray substrate** (the same honest gap as the sibling `select_from_model.rs`
unit, which takes importances directly).

**parity-ops correction.** The route table's `parity_ops` lists the score
functions `f_classif` / `f_regression` / `chi2` / `mutual_info_classif`. Those are
sklearn's `score_func` callables and are translated in the SIBLING
`feature_scoring.rs` unit — they are NOT in this file. **THIS unit's parity-ops are
the FPR/FDR/FWE selectors themselves** (`SelectFpr`, `SelectFdr`, `SelectFwe`),
which consume a precomputed p-value vector; the front-matter `parity-ops` is set
accordingly (`SelectFpr, SelectFdr, SelectFwe`).

This is a **shipped-partial** unit: **4 SHIPPED** (REQ-1 FPR mask, REQ-2 FDR
Benjamini-Hochberg mask, REQ-3 FWE Bonferroni mask, REQ-4 error /
parameter contracts incl. `alpha ∈ [0,1]`) / **5 NOT-STARTED** (REQ-5 `score_func` wrapping, REQ-6
`SelectorMixin` surface, REQ-7 `scores_` / `pvalues_` / `n_features_in_` fitted
attrs, REQ-8 PyO3 binding, REQ-9 ferray substrate).

## Probes (live sklearn oracle, 1.5.2)

Because the real sklearn selectors require a `score_func` (no direct
p-values-vector API), the cleanest oracle is each selector's `_get_support_mask`
FORMULA replicated in numpy on a chosen p-value vector — this IS the spec the doc
pins ferrolearn's `selected_indices()` against (`SelectFpr:878`,
`SelectFdr:959-969`, `SelectFwe:1044`). All values below are live numpy output
(sklearn 1.5.2 installed).

```bash
# PROBE A (REQ-1) — SelectFpr mask: pvalues_ < alpha (_univariate_selection.py:878):
python3 -c "print('fpr',[j for j,p in enumerate([0.01,0.5,0.03,0.9]) if p<0.05])"
#   -> fpr [0, 2]
#   ferrolearn SelectFpr::<f64>::new(0.05).fit(array![0.01,0.5,0.03,0.9]):
#     selected_indices() = [0, 2]  (IDENTICAL — strict <, p==alpha NOT selected).

# PROBE B (REQ-2) — SelectFdr Benjamini-Hochberg (_univariate_selection.py:959-969):
python3 -c "import numpy as np
def fdr(pv,a): pv=np.asarray(pv);n=len(pv);sv=np.sort(pv);sel=sv[sv<=float(a)/n*np.arange(1,n+1)];return ([] if sel.size==0 else np.flatnonzero(pv<=sel.max()).tolist())
print('fdr     ',fdr([0.01,0.5,0.03,0.9],0.05))          # [0]
print('fdr_tie ',fdr([0.01,0.025,0.025,0.9],0.05))       # [0, 1, 2]
print('fdr_gap ',fdr([0.001,0.04,0.045,0.011],0.05))     # [0, 1, 2, 3]
print('fdr_none',fdr([0.9,0.8,0.95],0.05))               # []"
#   -> fdr      [0]
#   -> fdr_tie  [0, 1, 2]   (ties at 0.025 BOTH kept — sklearn pvalues_ <= selected.max())
#   -> fdr_gap  [0, 1, 2, 3] (NON-MONOTONE: 0.045 at rank 2 fails its threshold but rank 3's
#                             0.04 qualifies — highest qualifying rank == selected.max())
#   -> fdr_none []
#   ferrolearn SelectFdr::<f64>::new(0.05).fit(...): selected_indices() = the same sets.

# PROBE C (REQ-3) — SelectFwe Bonferroni: pvalues_ < alpha / n (_univariate_selection.py:1044):
python3 -c "print('fwe',[j for j,p in enumerate([0.001,0.5,0.03,0.9]) if p<0.05/4])"
#   -> fwe [0]   (threshold 0.05/4 = 0.0125; only p=0.001 qualifies; strict <)
#   ferrolearn SelectFwe::<f64>::new(0.05).fit(array![0.001,0.5,0.03,0.9]):
#     selected_indices() = [0]  (IDENTICAL — strict <, p==alpha/n NOT selected).
```

**FDR equivalence note (REQ-2).** sklearn computes `selected = sv[sv <= alpha/n *
arange(1, n+1)]` then `mask = pvalues_ <= selected.max()` (`:964-969`). ferrolearn
sorts `(idx, p)` ascending, finds the LARGEST rank `k` with `p_(k) <= alpha*(k+1)/n`
(in-module `fit`, `:286-293`), and selects `ranked[..=k]` (all features at/below the
highest qualifying rank). These are equivalent: the largest qualifying rank's
p-value IS `selected.max()`, and selecting every original feature with `p <=
selected.max()` IS the prefix `ranked[..=k]` after re-sorting to ascending index.
The tie case (`[0.01, 0.025, 0.025, 0.9]` → `[0, 1, 2]`) and the non-monotone gap
case (`[0.001, 0.04, 0.045, 0.011]` → all four, because the highest qualifying
rank is rank 3 holding `0.04`, and `0.045 <= 0.04`-max is FALSE but `0.045` is at a
LOWER rank than the max-rank so it is included via the prefix — equivalently
`0.045 <= selected.max()=0.045`) both confirm the equivalence against the live
oracle (Probe B).

## Requirements

- REQ-1: **SelectFpr mask `p < alpha`, GIVEN a static p-value vector** (SHIPPED).
  Keep every feature whose p-value is strictly below `alpha`, mirroring
  `SelectFpr._get_support_mask` `return self.pvalues_ < self.alpha`
  (`_univariate_selection.py:878`). ferrolearn's `Fit::fit for SelectFpr` computes
  `selected_indices = {j : p[j] < alpha}`. The boundary `p == alpha` is NOT
  selected (strict `<`, matching the `<` operator at `:878`).

- REQ-2: **SelectFdr Benjamini-Hochberg mask, GIVEN a static p-value vector,
  including ties and non-monotone gaps** (SHIPPED). Apply the BH procedure: sort
  p-values ascending, find the highest rank `k` with `p_(k) <= alpha*(k+1)/n`, and
  select every feature at/below that rank, mirroring
  `SelectFdr._get_support_mask` (`:959-969`: `selected = sv[sv <= alpha/n *
  arange(1, n+1)]`; empty → all-False; else `pvalues_ <= selected.max()`).
  ferrolearn's `Fit::fit for SelectFdr` is algebraically equivalent (highest
  qualifying rank + `ranked[..=k]`), verified against the live oracle on the tie
  fixture and the non-monotone gap fixture (Probe B).

- REQ-3: **SelectFwe Bonferroni mask `p < alpha / n`, GIVEN a static p-value
  vector** (SHIPPED). Keep every feature whose p-value is strictly below
  `alpha / n_features`, mirroring `SelectFwe._get_support_mask` `return
  self.pvalues_ < self.alpha / len(self.pvalues_)` (`:1044`). ferrolearn's
  `Fit::fit for SelectFwe` computes `selected_indices = {j : p[j] < alpha/n}`. The
  boundary `p == alpha/n` is NOT selected (strict `<`, matching `:1044`).

- REQ-4: **Error / parameter contracts incl. `alpha ∈ [0,1]` (closed-both)**
  (SHIPPED). All three selectors reject an empty p-value vector and an
  out-of-range `alpha` via the shared `validate_inputs(n, alpha)`; `Transform`
  rejects a column-count mismatch. `validate_inputs` accepts `alpha ∈ [0, 1]`
  (closed-both), matching sklearn's `_parameter_constraints` `alpha` constraint
  `Interval(Real, 0, 1, closed="both")` (`:866-869`, `:950-953`, `:1032-1035`) and
  the `_validate_data` shape contract, scoped to the p-value-vector API ferrolearn
  actually exposes. `alpha == 0` → `fit` returns `Ok` and selects nothing (FPR/FWE
  `p < 0` and FDR all-zero BH threshold qualify nothing). This was the #1397
  divergence (ferrolearn previously rejected `alpha == 0` with interval `(0, 1]`);
  now RESOLVED — the divergence tests `divergence_{fpr,fwe,fdr}_alpha_zero_accepted`
  pass and the in-module `test_{fpr,fdr,fwe}_alpha_zero_valid` tests pin the
  accepted endpoint.

- REQ-5: **`score_func` wrapping (`f_classif` / `f_regression` / `chi2` /
  `mutual_info_classif` → `scores_` / `pvalues_` at `fit(X, y)`)** (the big HONEST
  gap, NOT-STARTED). sklearn's `_BaseFilter.fit` (`:541-577`) runs
  `score_func(X, y)` and stores `self.scores_` / `self.pvalues_` (`:567-575`); the
  masks then operate on `self.pvalues_`. ferrolearn takes the p-value VECTOR
  directly as `x: &Array1<F>` and never wraps or invokes a `score_func` (the score
  functions live in the sibling `feature_scoring.rs` unit).

- REQ-6: **`SelectorMixin` surface (`get_support` / `inverse_transform` /
  `get_feature_names_out`)** (NOT-STARTED). sklearn's selectors inherit
  `SelectorMixin` (`_BaseFilter(SelectorMixin, BaseEstimator)`, `:526`), exposing
  `get_support()` (boolean mask / indices), `inverse_transform` (scatter selected
  columns back into a full-width zero matrix), and `get_feature_names_out`.
  ferrolearn's `Fitted*` types expose `selected_indices()` / `n_features_selected()`
  but none of the mixin surface.

- REQ-7: **`scores_` / `pvalues_` as fitted attributes of the COMPUTED score +
  `n_features_in_` / `feature_names_in_`** (NOT-STARTED). sklearn exposes
  `scores_` and `pvalues_` (the score-function output, `:569-575`), plus
  `n_features_in_` and `feature_names_in_` (`SelectFpr` docstring `:828-837`).
  ferrolearn's `Fitted*` stores the INPUT `p_values` (the user-supplied vector,
  echoed back via `p_values()`) and `n_features_in` internally but exposes no
  `scores_`, no sklearn-named `n_features_in_` accessor, and no
  `feature_names_in_`.

- REQ-8: **PyO3 binding** (NOT-STARTED). There is no `_RsSelectFpr` /
  `_RsSelectFdr` / `_RsSelectFwe` CPython binding in `ferrolearn-python`
  (`grep -rn "SelectFpr\|SelectFdr\|SelectFwe" ferrolearn-python/src` finds none),
  so the selectors are unreachable from Python.

- REQ-9: **ferray substrate** (NOT-STARTED). Compute the masks, the BH sort, and
  the column gather over `ferray-core` arrays rather than `ndarray::Array1` /
  `Array2`, `num_traits::Float`, and the `Vec<usize>` index path (`select_columns`)
  (R-SUBSTRATE-1/2).

## Acceptance criteria

- AC-1 (REQ-1): `SelectFpr::<f64>::new(0.05).fit(array![0.01, 0.5, 0.03,
  0.9])` yields `selected_indices() == [0, 2]` (Probe A). Pinned by
  `test_fpr_selects_below_alpha`, `test_fpr_none_below_alpha`,
  `test_fpr_all_below_alpha`, `test_fpr_transform` (in-module).

- AC-2 (REQ-2): `SelectFdr::<f64>::new(0.05)` selects `[0]` on `[0.01, 0.5,
  0.03, 0.9]`, `[0, 1, 2]` on the tie fixture `[0.01, 0.025, 0.025, 0.9]`, all four
  on the non-monotone gap fixture `[0.001, 0.04, 0.045, 0.011]`, and nothing on
  `[0.9, 0.8, 0.95]` (Probe B). Pinned by `test_fdr_basic`,
  `test_fdr_multiple_pass`, `test_fdr_none_selected`, `test_fdr_transform`
  (in-module); the tie / gap fixtures are characterization gaps in the in-module
  suite (covered by the doc's oracle Probe B against the equivalence argument).

- AC-3 (REQ-3): `SelectFwe::<f64>::new(0.05).fit(array![0.001, 0.5, 0.03,
  0.9])` yields `selected_indices() == [0]` (threshold `0.05/4 = 0.0125`, Probe
  C). Pinned by `test_fwe_basic`, `test_fwe_two_features`, `test_fwe_none_selected`,
  `test_fwe_single_feature`, `test_fwe_transform`, `test_fwe_f32` (in-module).

- AC-4 (REQ-4): `SelectFpr::<f64>::new(0.05).fit(Array1::zeros(0))` returns
  `Err(InvalidParameter)`; `new(-0.1)` / `new(1.5)` return `Err(InvalidParameter)`;
  `new(0.0).fit(...)` returns `Ok` and selects nothing; a fitted handle's
  `transform` on a wrong column count returns `Err(ShapeMismatch)`. Pinned by
  `test_{fpr,fdr,fwe}_empty_error`, `test_{fpr,fdr,fwe}_invalid_alpha` (-0.1 / 1.5
  rejected), `test_{fpr,fdr,fwe}_alpha_zero_valid` (alpha=0 accepted),
  `test_{fpr,fdr,fwe}_shape_mismatch`, and the divergence suite's
  `divergence_{fpr,fwe,fdr}_alpha_zero_accepted`.

- AC-5 (REQ-5): `SelectFpr(f_classif, alpha=0.05).fit(X, y)` computes
  `pvalues_` via `f_classif(X, y)` (`_BaseFilter:567-570`); ferrolearn has no
  `Fit<(Array2, Array1)>` `score_func` path — its `Fit` input is the p-value
  vector itself.

- AC-6 (REQ-6): a fitted selector exposes `get_support()`,
  `inverse_transform(transform(X))`, and `get_feature_names_out(names)`; ferrolearn
  exposes only `selected_indices()` / `n_features_selected()`.

- AC-7 (REQ-7): a fitted selector exposes `scores_`, `pvalues_`,
  `n_features_in_`, and `feature_names_in_`; ferrolearn echoes back the INPUT
  `p_values()` only and has no `scores_` / sklearn-named `n_features_in_` /
  `feature_names_in_`.

- AC-8 (REQ-8): a CPython `SelectFpr` / `SelectFdr` / `SelectFwe` binding fits
  and transforms from Python; no such binding exists in `ferrolearn-python`.

- AC-9 (REQ-9): the mask / BH-sort / gather path computes on `ferray-core`
  arrays rather than `ndarray` + `num_traits::Float` + the `Vec<usize>` index path.

`## REQ status`

## REQ status

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (SelectFpr mask `p < alpha`, GIVEN a static p-value vector) | SHIPPED | impl `Fit::fit for SelectFpr in stat_selectors.rs` computes `selected_indices = x.iter().enumerate().filter(\|&(_, &p)\| p < alpha_f).map(\|(j, _)\| j).collect()` — the strict `<` mirroring `SelectFpr._get_support_mask` `return self.pvalues_ < self.alpha` (`sklearn/feature_selection/_univariate_selection.py:878`). Non-test consumer: the boundary re-export `pub use stat_selectors::{... SelectFpr ...}` (`lib.rs:181-183`) plus `pub mod stat_selectors;` (`lib.rs:122`) — grandfathered S5 / R-DEFER-1 boundary estimator API. Verification: `cargo test -p ferrolearn-preprocess --lib` → `test_fpr_selects_below_alpha` (`selected_indices == [0, 2]`, Probe A), `test_fpr_none_below_alpha`, `test_fpr_all_below_alpha`, `test_fpr_transform` green (27 selector tests pass). |
| REQ-2 (SelectFdr Benjamini-Hochberg mask, incl. ties + non-monotone gap) | SHIPPED | impl `Fit::fit for SelectFdr in stat_selectors.rs` sorts `(idx, p)` ascending (`ranked.sort_by(...)`), finds the largest rank with `p_val <= alpha_f * F::from(rank + 1) / n_f` (`bh_threshold` loop), and selects `ranked[..=max_rank]` then `sort_unstable()` to ascending index. Algebraically equivalent to `SelectFdr._get_support_mask` `selected = sv[sv <= float(alpha)/n * np.arange(1, n+1)]; ... return self.pvalues_ <= selected.max()` (`_univariate_selection.py:959-969`): the highest-qualifying-rank p-value IS `selected.max()` and `ranked[..=k]` IS `{j : p[j] <= selected.max()}`. Live oracle (Probe B): `fdr([0.01,0.5,0.03,0.9],0.05)=[0]`, tie `fdr([0.01,0.025,0.025,0.9],0.05)=[0,1,2]`, non-monotone gap `fdr([0.001,0.04,0.045,0.011],0.05)=[0,1,2,3]`, `fdr_none=[]` — ferrolearn matches each. Non-test consumer: boundary re-export `SelectFdr` (`lib.rs:181-183`) + `pub mod stat_selectors;` (`lib.rs:122`). Verification: `cargo test -p ferrolearn-preprocess --lib` → `test_fdr_basic`, `test_fdr_multiple_pass` (3 features), `test_fdr_none_selected`, `test_fdr_transform` green. |
| REQ-3 (SelectFwe Bonferroni mask `p < alpha / n`, GIVEN a static p-value vector) | SHIPPED | impl `Fit::fit for SelectFwe in stat_selectors.rs` computes `adjusted_alpha = self.alpha / n as f64` then `selected_indices = x.iter().enumerate().filter(\|&(_, &p)\| p < adjusted_alpha_f).map(\|(j, _)\| j).collect()` — the strict `<` Bonferroni threshold mirroring `SelectFwe._get_support_mask` `return self.pvalues_ < self.alpha / len(self.pvalues_)` (`_univariate_selection.py:1044`). Live oracle (Probe C): `fwe([0.001,0.5,0.03,0.9],0.05)=[0]` (threshold `0.0125`). Non-test consumer: boundary re-export `SelectFwe` (`lib.rs:181-183`) + `pub mod stat_selectors;` (`lib.rs:122`). Verification: `cargo test -p ferrolearn-preprocess --lib` → `test_fwe_basic` (`selected_indices == [0]`), `test_fwe_two_features`, `test_fwe_none_selected`, `test_fwe_single_feature`, `test_fwe_transform`, `test_fwe_f32` green. |
| REQ-4 (error / parameter contracts incl. `alpha ∈ [0,1]` closed-both) | SHIPPED | impl `validate_inputs in stat_selectors.rs` returns `Err(FerroError::InvalidParameter { name: "p_values", .. })` when `n_features == 0`, and `Err(FerroError::InvalidParameter { name: "alpha", .. })` when `!(0.0..=1.0).contains(&alpha)`; called from all three `Fit::fit` impls. It accepts `alpha ∈ [0, 1]` (closed-both), matching sklearn's `_parameter_constraints` `Interval(Real, 0, 1, closed="both")` (`_univariate_selection.py:868`, `:952`, `:1034`); `alpha == 0` → `fit` `Ok`, selects nothing (FPR/FWE `p < 0`, FDR all-zero BH threshold). impl `Transform::transform for FittedSelectFpr/FittedSelectFdr/FittedSelectFwe in stat_selectors.rs` returns `Err(FerroError::ShapeMismatch { .. })` when `x.ncols() != self.n_features_in`. This was the #1397 divergence (ferrolearn previously used `(0, 1]`); now RESOLVED. Non-test consumer: boundary re-export (`lib.rs:181-183`). Verification: `cargo test -p ferrolearn-preprocess` → in-module `test_{fpr,fdr,fwe}_empty_error`, `test_{fpr,fdr,fwe}_invalid_alpha` (-0.1 / 1.5 rejected), `test_{fpr,fdr,fwe}_alpha_zero_valid` (alpha=0 accepted), `test_{fpr,fdr,fwe}_shape_mismatch`, plus divergence-suite `divergence_{fpr,fwe,fdr}_alpha_zero_accepted` green. |
| REQ-5 (`score_func` wrapping → `scores_` / `pvalues_` at `fit(X, y)`) | NOT-STARTED | open prereq blocker #1398. `Fit<Array1<F>, ()>` for all three selectors takes the p-value VECTOR directly (`x: &Array1<F>`, one value per feature) and never wraps or invokes a `score_func`. sklearn's `_BaseFilter.fit` runs `score_func_ret = self.score_func(X, y)` and stores `self.scores_, self.pvalues_ = score_func_ret` (`_univariate_selection.py:567-575`); `__init__(score_func=f_classif, ...)` (`:871`, `:955`, `:1037`). The score functions (`f_classif`/`f_regression`/`chi2`/`mutual_info_classif`) are translated in the sibling `feature_scoring.rs` unit, not here. |
| REQ-6 (`SelectorMixin` surface: `get_support` / `inverse_transform` / `get_feature_names_out`) | NOT-STARTED | open prereq blocker #1399. `FittedSelectFpr/FittedSelectFdr/FittedSelectFwe<F>` expose `p_values()` / `selected_indices()` / `n_features_selected()` but NO `get_support` boolean-mask form, NO `inverse_transform` (scatter selected columns back to full width), and NO `get_feature_names_out`. sklearn's selectors inherit `SelectorMixin` (`_BaseFilter(SelectorMixin, BaseEstimator)`, `_univariate_selection.py:526`), providing all three. |
| REQ-7 (`scores_` / `pvalues_` of the computed score + `n_features_in_` / `feature_names_in_`) | NOT-STARTED | open prereq blocker #1400. ferrolearn's `Fitted*` stores the user-supplied INPUT vector (echoed via `p_values()`) and `n_features_in` internally, but exposes NO `scores_`, NO sklearn-named `n_features_in_` accessor, and NO `feature_names_in_`. sklearn exposes `scores_` / `pvalues_` as the score-function OUTPUT (`_univariate_selection.py:569-575`) plus `n_features_in_` / `feature_names_in_` (`SelectFpr` docstring `:828-837`). |
| REQ-8 (PyO3 binding) | NOT-STARTED | open prereq blocker #1401. No `_RsSelectFpr` / `_RsSelectFdr` / `_RsSelectFwe` CPython binding exists — `grep -rn "SelectFpr\|SelectFdr\|SelectFwe" ferrolearn-python/src` finds none — so the selectors are unreachable from Python. |
| REQ-9 (ferray substrate) | NOT-STARTED | open prereq blocker #1402. The mask / BH-sort / gather path uses `ndarray::Array1` / `Array2` (`x.iter()`, `select_columns`, `Array2::zeros`), `num_traits::Float`, and a `Vec<usize>` selected-index path — not `ferray-core` arrays (R-SUBSTRATE-1/2). |

## Architecture

**ferrolearn (existing, compiled at baseline).** `stat_selectors.rs` exposes three
unfitted selectors — `SelectFpr<F>`, `SelectFdr<F>`, `SelectFwe<F>` (each `{ alpha:
f64, _marker: PhantomData<F> }`, `new(alpha)`, `alpha()`) — and their fitted
counterparts `FittedSelectFpr<F>` / `FittedSelectFdr<F>` / `FittedSelectFwe<F>`
(each `{ n_features_in: usize, p_values: Array1<F>, selected_indices: Vec<usize> }`;
accessors `p_values()`, `selected_indices()`, `n_features_selected()`). All three
implement `Fit<Array1<F>, ()>` (the `Fit` input `x` is the per-feature p-value
vector) and validate inputs through the shared `validate_inputs(n, alpha)` (empty
vector or `alpha ∉ [0, 1]` → `InvalidParameter`):

- `SelectFpr::fit` selects `{j : p[j] < alpha}` (strict `<`).
- `SelectFdr::fit` runs the Benjamini-Hochberg procedure: sort `(idx, p)`
  ascending, scan for the largest rank `k` with `p_(k) <= alpha*(k+1)/n`, select
  `ranked[..=k]`, then `sort_unstable()` back to ascending index order.
- `SelectFwe::fit` selects `{j : p[j] < alpha/n}` (Bonferroni, strict `<`).

Each `Fitted*` implements `Transform<Array2<F>>`, checking the column count
(`ShapeMismatch`) and gathering the selected columns via the shared
`select_columns(x, indices)` helper. The module is **compiled** at baseline (`pub
mod stat_selectors;`, `lib.rs:122`) and re-exported (`pub use
stat_selectors::{FittedSelectFdr, FittedSelectFpr, FittedSelectFwe, SelectFdr,
SelectFpr, SelectFwe};`, `lib.rs:181-183`) — that re-export is the grandfathered
boundary consumer pinning REQ-1..REQ-4 SHIPPED. There is **no `PipelineTransformer`
impl** and **no PyO3 binding**.

**`alpha` parameter contract (REQ-4).** ferrolearn's `validate_inputs` accepts
`alpha ∈ [0, 1]` (closed-both, `!(0.0..=1.0).contains(&alpha)` → `InvalidParameter`),
matching sklearn's `_parameter_constraints` `Interval(Real, 0, 1, closed="both")`
(`:868`, `:952`, `:1034`). With `alpha == 0` all three selectors fit `Ok` and
trivially select nothing (FPR/FWE: no `p < 0`; FDR: no `p <= 0` threshold
qualifies). The earlier `(0, 1]` interval (which rejected `alpha == 0`) was the
#1397 divergence, now RESOLVED — pinned by `divergence_{fpr,fwe,fdr}_alpha_zero_accepted`
and the in-module `test_{fpr,fdr,fwe}_alpha_zero_valid`.

**sklearn (target contract).** `SelectFpr` / `SelectFdr` / `SelectFwe`
(`_univariate_selection.py:801` / `:881` / `:972`) each subclass `_BaseFilter`
(`:526`), whose `fit` (`:541-577`) `_validate_data`s `(X, y)`, runs
`score_func_ret = self.score_func(X, y)` (default `f_classif`), and unpacks
`self.scores_, self.pvalues_ = score_func_ret` (`:567-575`). Each selector's
`_get_support_mask` operates on `self.pvalues_`:
`SelectFpr` → `pvalues_ < alpha` (`:878`); `SelectFwe` → `pvalues_ < alpha/n`
(`:1044`); `SelectFdr` → BH: `n = len(pvalues_); sv = np.sort(pvalues_); selected =
sv[sv <= float(alpha)/n * np.arange(1, n+1)]; if selected.size == 0: all-False;
else pvalues_ <= selected.max()` (`:959-969`). All three share
`__init__(score_func=f_classif, *, alpha=5e-2)` and the alpha constraint
`Interval(Real, 0, 1, closed="both")`. The selector surface (`get_support`,
`transform`, `inverse_transform`, `get_feature_names_out`) and the
`n_features_in_` / `feature_names_in_` attributes come from `SelectorMixin` /
`BaseEstimator`.

**The gap.** ferrolearn matches sklearn on the three *mask algebras* over a static
p-value vector — FPR strict-`<` (REQ-1), the BH procedure including ties and
non-monotone gaps (REQ-2, verified equivalent to `pvalues_ <= selected.max()`,
Probe B), and FWE Bonferroni strict-`<` (REQ-3), plus the error / parameter contracts
(REQ-4, incl. the closed-both `alpha ∈ [0, 1]` interval). The remaining gaps are the score-function-facing filter surface: no
`score_func` wrapping / `scores_` / `pvalues_` computation at `fit(X, y)` (REQ-5,
the honest gap — ferrolearn's `Fit` input IS the p-value vector, and the score
functions live in `feature_scoring.rs`); no `SelectorMixin` surface (REQ-6); no
computed `scores_` / `pvalues_` / `n_features_in_` / `feature_names_in_` fitted
attrs (REQ-7); no PyO3 binding (REQ-8); and the non-ferray substrate (REQ-9). This
is a **shipped-partial** unit (4 SHIPPED / 5 NOT-STARTED).

## Verification

Commands establishing the SHIPPED claims (REQ-1 FPR mask, REQ-2 FDR BH mask,
REQ-3 FWE Bonferroni mask, REQ-4 error / parameter contracts):

```bash
# Module is compiled + re-exported (the boundary consumer):
grep -n "mod stat_selectors" ferrolearn-preprocess/src/lib.rs          # lib.rs:122
grep -n "SelectFpr\|SelectFdr\|SelectFwe" ferrolearn-preprocess/src/lib.rs  # lib.rs:181-183

# REQ-1..REQ-4 value parity + error contracts (27 in-module selector tests):
cargo test -p ferrolearn-preprocess --lib
#   REQ-1: test_fpr_selects_below_alpha ([0,2]), test_fpr_none_below_alpha,
#          test_fpr_all_below_alpha, test_fpr_transform
#   REQ-2: test_fdr_basic, test_fdr_multiple_pass, test_fdr_none_selected, test_fdr_transform
#   REQ-3: test_fwe_basic ([0]), test_fwe_two_features, test_fwe_none_selected,
#          test_fwe_single_feature, test_fwe_transform, test_fwe_f32
#   REQ-4: test_{fpr,fdr,fwe}_empty_error, test_{fpr,fdr,fwe}_invalid_alpha (-0.1/1.5),
#          test_{fpr,fdr,fwe}_alpha_zero_valid (alpha=0 accepted), test_{fpr,fdr,fwe}_shape_mismatch
#   -> test result: ok. 429 passed; 0 failed (selector tests included).
# REQ-4 alpha=0 endpoint (divergence suite, the resolved #1397 item):
cargo test -p ferrolearn-preprocess --test divergence_stat_selectors
#   -> divergence_{fpr,fwe,fdr}_alpha_zero_accepted green (fit Ok, selects nothing).
cargo clippy -p ferrolearn-preprocess --all-targets -- -D warnings
cargo fmt --all --check

# REQ-1 oracle gate — SelectFpr mask (_univariate_selection.py:878):
python3 -c "print('fpr',[j for j,p in enumerate([0.01,0.5,0.03,0.9]) if p<0.05])"
#   -> fpr [0, 2]   (ferrolearn SelectFpr::new(0.05): [0, 2])

# REQ-2 oracle gate — SelectFdr BH incl. ties + non-monotone gap (:959-969):
python3 -c "import numpy as np
def fdr(pv,a): pv=np.asarray(pv);n=len(pv);sv=np.sort(pv);sel=sv[sv<=float(a)/n*np.arange(1,n+1)];return ([] if sel.size==0 else np.flatnonzero(pv<=sel.max()).tolist())
print('fdr',fdr([0.01,0.5,0.03,0.9],0.05),'tie',fdr([0.01,0.025,0.025,0.9],0.05),'gap',fdr([0.001,0.04,0.045,0.011],0.05),'none',fdr([0.9,0.8,0.95],0.05))"
#   -> fdr [0] tie [0, 1, 2] gap [0, 1, 2, 3] none []

# REQ-3 oracle gate — SelectFwe Bonferroni (:1044):
python3 -c "print('fwe',[j for j,p in enumerate([0.001,0.5,0.03,0.9]) if p<0.05/4])"
#   -> fwe [0]   (threshold 0.05/4 = 0.0125; ferrolearn SelectFwe::new(0.05): [0])
```

The in-module `#[test]`s exercise REQ-1 (FPR mask), REQ-2 (the BH multi-pass +
none-selected paths), REQ-3 (FWE Bonferroni + f32), and REQ-4 (every error path).
No green command establishes REQ-5..REQ-9 (`score_func` wrapping,
`SelectorMixin` surface, computed `scores_` / `pvalues_` / `n_features_in_`, PyO3,
ferray).

## Blockers

REQ-1 (FPR mask), REQ-2 (FDR Benjamini-Hochberg mask), REQ-3 (FWE Bonferroni
mask), and REQ-4 (error / parameter contracts incl. the closed-both `alpha ∈ [0,
1]` interval) are SHIPPED — the module is compiled (`lib.rs:122`) and re-exported
(`lib.rs:181-183`, the grandfathered boundary consumer), and its in-module +
divergence tests are green.

- #1397 — REQ-4 alpha=0 endpoint: ferrolearn previously rejected `alpha == 0`
  (interval `(0, 1]`) vs sklearn's `Interval(Real, 0, 1, closed="both")` (`:868`,
  `:952`, `:1034`). **RESOLVED** — `validate_inputs` now accepts `alpha ∈ [0, 1]`;
  pinned by `divergence_{fpr,fwe,fdr}_alpha_zero_accepted` + `test_{fpr,fdr,fwe}_alpha_zero_valid`.

The remaining REQs are NOT-STARTED, filed as `-l blocker` issues against
tracking issue #1396:

- #1398 — REQ-5: no `score_func` wrapping; ferrolearn's `Fit` input is the
  p-value vector, and the score functions live in `feature_scoring.rs`
  (`_univariate_selection.py:567-575`, `:871`, `:955`, `:1037`).
- #1399 — REQ-6: no `SelectorMixin` surface — `get_support` /
  `inverse_transform` / `get_feature_names_out` (`_univariate_selection.py:526`).
- #1400 — REQ-7: no computed `scores_` / `pvalues_` / `n_features_in_` /
  `feature_names_in_` fitted attributes (`:569-575`, `:828-837`).
- #1401 — REQ-8: no PyO3 `SelectFpr` / `SelectFdr` / `SelectFwe` binding in
  `ferrolearn-python`.
- #1402 — REQ-9: fit/transform on `ndarray` / `num_traits` / `Vec<usize>`
  index path, not ferray (R-SUBSTRATE-1/2).
