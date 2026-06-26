# ferrolearn-linear — SVM Bounds Helpers

<!--
tier: 2-helper
status: draft
upstream-paths:
  - sklearn/svm/_bounds.py
-->

## Summary

`ferrolearn-linear/src/svm_bounds.rs` mirrors `sklearn.svm.l1_min_c`, the small helper that computes the lowest `C` value above which L1-penalized linear classifiers can select a non-zero coefficient or intercept. The Rust API is `l1_min_c(&Array2<F>, &Array1<usize>, L1MinCLoss, fit_intercept, intercept_scaling)`.

## Requirements

- REQ-1: compute `den = max(abs(LabelBinarizer(neg_label=-1).fit_transform(y).T @ X))`, then include the synthetic intercept column when `fit_intercept=true`, matching `sklearn/svm/_bounds.py:80-91`.
- REQ-2: return `0.5 / den` for `loss="squared_hinge"` and `2.0 / den` for `loss="log"`, matching `_bounds.py:97-99`.
- REQ-3: reject inconsistent lengths, empty inputs, non-finite `X`, non-positive `intercept_scaling`, and `den == 0` ill-posed inputs.

## Status

| REQ | Status | Evidence |
|---|---|---|
| REQ-1/2 (formula) | SHIPPED | `l1_min_c` builds binary and multiclass label-binarizer rows and computes the max absolute feature/intercept dot product before applying the sklearn loss constants. Tests: `tests/divergence_svm_bounds.rs::{l1_min_c_binary_matches_sklearn_no_intercept,l1_min_c_binary_intercept_scaling_matches_sklearn,l1_min_c_multiclass_matches_label_binarizer_oracle}`. |
| REQ-3 (validation) | SHIPPED | `validate_l1_min_c_input` covers shape, empty, non-finite, and intercept-scaling contracts; `den == 0` returns `InvalidParameter`. Test: `l1_min_c_rejects_ill_posed_and_invalid_inputs`. |
| REQ-substrate | NOT-STARTED | Uses `ndarray::{Array1, Array2}` rather than ferray arrays. |

## Verification

- `cargo test -p ferrolearn-linear --test divergence_svm_bounds --test api_proof`
