# GenericUnivariateSelect

<!--
tier: 3-component
status: shipped-partial
baseline-commit: e6e8d1da
upstream: scikit-learn 1.5.2
upstream-paths:
  - sklearn/feature_selection/_univariate_selection.py  # class GenericUnivariateSelect (:1062); _selection_modes (:1127-1133); __init__(score_func=f_classif, *, mode="percentile", param=1e-5) (:1141-1144); _make_selector (:1146-1155); _get_support_mask (:1165-1171)
ferrolearn-module: ferrolearn-preprocess/src/generic_univariate_select.rs
parity-ops: GenericUnivariateSelect
crosslink-issue: 1429
-->

## Summary

`GenericUnivariateSelect` is sklearn's configurable univariate selector wrapper:
it computes `scores_`/`pvalues_` through a `score_func`, then delegates the support
mask to one of `SelectPercentile`, `SelectKBest`, `SelectFpr`, `SelectFdr`, or
`SelectFwe` based on `mode`.

`ferrolearn_preprocess::GenericUnivariateSelect` ships the dense Rust analogue for
the existing `ScoreFunc::FClassif` scorer. The Rust API uses typed
`GenericUnivariateMode` and `GenericUnivariateParam` enums instead of stringly
Python parameters, including the k-best `"all"` sentinel. It stores scores,
p-values, selected indices, and participates in `SelectorMixin`.

## REQ Status

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 public exact-name surface | SHIPPED scoped | `pub struct GenericUnivariateSelect<F>` is re-exported from `ferrolearn_preprocess`; boundary tests name it directly. |
| REQ-2 mode dispatch | SHIPPED scoped | `GenericUnivariateMode::{Percentile,KBest,Fpr,Fdr,Fwe}` maps to sklearn `_selection_modes`; `tests/divergence_generic_univariate_select.rs` pins all five support masks against live sklearn 1.5.2. |
| REQ-3 fitted scores/p-values | SHIPPED scoped | `FittedGenericUnivariateSelect::scores()` and `p_values()` expose the arrays computed by `f_classif`; oracle test pins values including a NaN constant feature. |
| REQ-4 SelectorMixin dense helpers | SHIPPED scoped | `SelectorMixin` impl provides support masks, inverse zero-fill, and selected feature names; covered by `divergence_generic_univariate_select` and `divergence_selector_mixin`. |
| REQ-5 full sklearn protocol | NOT-STARTED | Callable score functions, `chi2`/`f_regression`/`mutual_info_*` dispatch, sparse/pandas output, exact Python error ABI, and PyO3 wrapper remain open under #1429/#1421/#1432. |

## Verification

```bash
cargo test -p ferrolearn-preprocess --test divergence_generic_univariate_select
cargo test -p ferrolearn-preprocess --test divergence_selector_mixin
cargo test -p ferrolearn-preprocess --test api_proof api_proof_feature_selection
cargo test -p ferrolearn-preprocess --test divergence_lib
cargo test -p ferrolearn-preprocess --test conformance_surface_coverage
```
