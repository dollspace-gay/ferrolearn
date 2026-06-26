# SelectorMixin — dense feature-selector support surface

<!--
tier: 3-component
status: shipped-partial
baseline-commit: accfa1695
upstream: scikit-learn 1.5.2
upstream-paths:
  - sklearn/feature_selection/_base.py  # class SelectorMixin (:27); get_support (:54); inverse_transform (:136); get_feature_names_out (:176)
ferrolearn-module: ferrolearn-preprocess/src/selector_mixin.rs
parity-ops: SelectorMixin
crosslink-issue: 1430
-->

## Summary

`SelectorMixin` is sklearn's shared feature-selector base surface. It turns a
selector's support mask into `get_support`, `transform`, `inverse_transform`, and
`get_feature_names_out`.

ferrolearn ships the dense Rust analogue as `ferrolearn_preprocess::SelectorMixin`.
Implementors provide `n_features_in()` and `selected_indices()`, and the trait
supplies boolean support masks, support indices, dense inverse-transform zero-fill,
and selected feature names.

Implemented for: `FittedVarianceThreshold`, `FittedSelectKBest`,
`SelectFromModel`, `FittedSelectPercentile`, `FittedSelectFpr`,
`FittedSelectFdr`, `FittedSelectFwe`, `FittedSelectFromModelExt`, `RFE`,
`RFECV`, and `FittedSequentialFeatureSelector`.

Residual gaps: sparse matrices, pandas output, Python `BaseEstimator`/fitted-state
checks, sklearn attribute names such as `feature_names_in_`, and estimator
delegation surfaces remain owned by the individual selector docs.

## Verification

- `cargo test -p ferrolearn-preprocess --test divergence_selector_mixin`
- `cargo test -p ferrolearn-preprocess --test api_proof api_proof_feature_selection`
- `cargo test -p ferrolearn-preprocess --test divergence_lib`
- `cargo test -p ferrolearn-preprocess --test conformance_surface_coverage`

The oracle fixture uses sklearn `VarianceThreshold` on
`[[1,10,100],[2,10,200],[3,10,300]]`: support `[True, False, True]`, indices
`[0, 2]`, inverse transform zero-fills column 1, and feature names filter to
`["x0", "x2"]` or `["a", "c"]`.
