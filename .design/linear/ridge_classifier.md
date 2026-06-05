# Ridge Classifier

<!--
tier: 3-component
status: draft
baseline-commit: daf8c0aa3635d9dd84b7d416172eefcf39d42d51
upstream-paths:
  - sklearn/linear_model/_ridge.py
-->

## Summary
`ridge_classifier.rs` mirrors scikit-learn's `class RidgeClassifier`
(`sklearn/linear_model/_ridge.py:1344`), a classifier that binarizes the target with a
`LabelBinarizer(pos_label=1, neg_label=-1)` (`_ridge.py:1300`), fits multi-output Ridge
regression on the encoded indicator matrix, and predicts via the sign of the decision
function (binary) or its argmax (multiclass). The ferrolearn implementation exposes
`RidgeClassifier<F>` / `FittedRidgeClassifier<F>` and covers the default dense path —
`±1`/one-hot encoding, per-class closed-form Ridge fit (via `crate::linalg::solve_ridge`),
`fit_intercept` by centering, and `coef_`/`intercept_`/`classes_` introspection. It does
NOT cover `class_weight`, the `solver` variants and the `solver_` attribute, `positive`,
`max_iter`/`tol`/`n_iter_`, `sample_weight`, the separate `RidgeClassifierCV` estimator, or
the ferray substrate.

## Requirements
- REQ-1: Encode the target as an indicator matrix — binary as a single `{-1, +1}` column,
  multiclass as one-hot columns (mirrors sklearn's `LabelBinarizer(pos_label=1,
  neg_label=-1)`, `_ridge.py:1300-1301`) — then solve multi-output Ridge per column on the
  encoded matrix.
- REQ-2: Predict by the decision function `X @ coef + intercept`: binary takes the sign
  (`classes_[1]` if score >= 0 else `classes_[0]`), multiclass takes the argmax over class
  columns (mirrors `LinearClassifierMixin.predict` reached via `super().predict(X)` at
  `_ridge.py:1333`).
- REQ-3: `fit_intercept` (default `true`); when `true`, center `X`/`Y` and recover the
  intercept from the means; when `false` the intercept is `0` (sklearn `_BaseRidge`
  default `fit_intercept=True`, `_ridge.py:1363`).
- REQ-4: `alpha` (default `1.0`) regularization strength; reject negative `alpha` with a
  parameter error (sklearn `_ridge.py:1355`, "must be a positive float").
- REQ-5: Expose fitted attributes for introspection — `coef_` (the per-class coefficient
  matrix) via `HasCoefficients`/`coef_matrix`, `intercept_` via `intercept`/`intercept_vec`,
  and `classes_` (sorted unique labels) via `HasClasses` (sklearn attributes `coef_`,
  `intercept_`, `classes_`, `_ridge.py:1455-1469`), plus a `decision_function` returning
  `(n_samples, n_classes)`.
- REQ-6: At least two distinct classes are required to fit; reject single-class input
  (sklearn binarization requires >= 2 classes).
- REQ-7: `class_weight`, the `solver` parameter and its variants (`auto`/`svd`/`cholesky`/
  `lsqr`/`sparse_cg`/`sag`/`saga`/`lbfgs`) plus the resolved `solver_` fitted attribute, and
  the `positive` (non-negative coefficient) constraint (sklearn `_ridge.py:1398-1484`).
- REQ-8: `max_iter`/`tol` controls for iterative solvers and the `n_iter_` fitted attribute
  (sklearn `_ridge.py:1371-1396`, `:1464`).
- REQ-9: `sample_weight` support (per-sample weighting, sklearn `_ridge.py:1551-1556`,
  applied in `_prepare_data` at `_ridge.py:1305-1307`).
- REQ-10: `RidgeClassifierCV` — the separate cross-validated classifier
  (`class RidgeClassifierCV`, `_ridge.py:2676`) declared in this route's `parity_ops` but with
  no ferrolearn type.
- REQ-11 (substrate): run the owned array/linalg computation on ferray (`ferray-core` array
  type, `ferray::linalg`) rather than `ndarray` + the in-crate `linalg` solver.

## Acceptance criteria
- AC-1: For binary `X=[[1,1],[1,2],[2,1],[2,2],[8,8],[8,9],[9,8],[9,9]]`,
  `y=[0,0,0,0,1,1,1,1]`, `alpha=1.0`, `fit_intercept=true`, the decision function is a single
  column and `predict` returns `[0,0,0,0,1,1,1,1]` (sklearn oracle: `coef_` shape `(1,2)`,
  `predict` `[0,0,0,0,1,1,1,1]`, `intercept_ ≈ -1.40703517`).
- AC-2: For 3-class well-separated `X3`/`y3` with `alpha=0.1`, `coef_` is `(3, n_features)`,
  `classes_ == [0,1,2]`, and `predict` recovers the labels (sklearn oracle: `coef_` shape
  `(3,2)`, `predict` `[0,0,0,1,1,1,2,2,2]`).
- AC-3: Negative `alpha`, sample-count mismatch, and single-class input each return `Err`.
- AC-4: `coefficients().len() == n_features` and `classes() == sorted unique labels`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (indicator encoding + per-class Ridge fit) | SHIPPED | impl `pub fn fit in ridge_classifier.rs` builds `y_indicator` as `{-1,+1}` (binary) or one-hot (multiclass) and calls `linalg::solve_ridge` per target column, mirroring sklearn `LabelBinarizer(pos_label=1, neg_label=-1)` (`_ridge.py:1300`, `Y = self._label_binarizer.fit_transform(y)`). Non-test consumer: `RsRidgeClassifier` in `ferrolearn-python/src/extras.rs` (`py_classifier!(RsRidgeClassifier, ..., ferrolearn_linear::FittedRidgeClassifier<f64>, ... RidgeClassifier::<f64>::new())`) wrapped by `class RidgeClassifier` in `_extras.py`. Verification: sklearn oracle `RidgeClassifier(alpha=1.0).fit(X,y)` yields `coef_` shape `(1,2)`; `cargo test -p ferrolearn-linear` (`test_binary_classification`, `test_multiclass_classification`) PASS. |
| REQ-2 (sign/argmax predict) | SHIPPED | impl `pub fn predict in ridge_classifier.rs` (`FittedRidgeClassifier`) computes `scores = x.dot(coef_matrix) + intercept_vec`, sign for binary / argmax for multiclass — mirrors `super().predict(X)` (`LinearClassifierMixin.predict`, `_ridge.py:1333`). Also `pub fn decision_function in ridge_classifier.rs`. Non-test consumer: `RsRidgeClassifier::predict` path in `ferrolearn-python/src/extras.rs` via the `py_classifier!` macro. Verification: oracle `predict` `[0,0,0,0,1,1,1,1]` (binary) / `[0,0,0,1,1,1,2,2,2]` (multi) matches; `cargo test -p ferrolearn-linear` PASS. |
| REQ-3 (fit_intercept by centering) | SHIPPED | impl `pub fn fit in ridge_classifier.rs` centers `X`/`Y` by their means when `self.fit_intercept`, recovers `intercept_vec = ym - xm·coef`, else zeros; `with_fit_intercept` setter present. Mirrors sklearn default `fit_intercept=True` (`_ridge.py:1363`). Non-test consumer: `RsRidgeClassifier` passes `fit_intercept` (`extras.rs`, `with_fit_intercept(fit_intercept)`). Verification: oracle `intercept_ ≈ -1.40703517`; `cargo test -p ferrolearn-linear` PASS. |
| REQ-4 (alpha + non-negative guard) | SHIPPED | impl `pub fn fit in ridge_classifier.rs` returns `FerroError::InvalidParameter` for `alpha < 0`; `RidgeClassifier::new` defaults `alpha = 1.0`; `with_alpha` setter present — mirrors sklearn `alpha=1.0` "positive float" (`_ridge.py:1355`). Non-test consumer: `RsRidgeClassifier` passes `alpha` (`extras.rs`, `with_alpha(alpha)`). Verification: `cargo test -p ferrolearn-linear` (`test_negative_alpha`, `test_alpha_zero`) PASS. |
| REQ-5 (coef_/intercept_/classes_ + decision_function) | SHIPPED | impl `impl HasCoefficients for FittedRidgeClassifier in ridge_classifier.rs` (`coefficients`/`intercept`), `pub fn coef_matrix`/`pub fn intercept_vec`, `impl HasClasses for FittedRidgeClassifier` (`classes`/`n_classes`), and `pub fn decision_function in ridge_classifier.rs`. Mirrors sklearn `coef_`/`intercept_`/`classes_` (`_ridge.py:1455-1469`, `classes_` property `_ridge.py:1336`). Non-test consumer: `RsRidgeClassifier` exposes `classes_`/coefficients to the `_ClassifierWrapper` in `_extras.py`. Verification: `cargo test -p ferrolearn-linear` (`test_has_coefficients`, `test_has_classes`) PASS. |
| REQ-6 (>= 2 classes required) | SHIPPED | impl `pub fn fit in ridge_classifier.rs` returns `FerroError::InsufficientSamples` when `classes.len() < 2`. sklearn's `LabelBinarizer` likewise requires >= 2 classes to produce a usable indicator matrix. Non-test consumer: error surfaces through `RsRidgeClassifier::fit` (`extras.rs`). Verification: `cargo test -p ferrolearn-linear` (`test_single_class_error`) PASS. |
| REQ-7a (positive=True) | SHIPPED | `RidgeClassifier<F>` adds `pub positive: bool` (default `false`, sklearn `_ridge.py:902`/`:911`) + `with_positive(bool)` builder. `pub fn fit_with_sample_weight in ridge_classifier.rs` routes EACH indicator-target column through `crate::linalg::nonneg_ridge_cd` (shared projected-coordinate-descent kernel, `max_iter`/`tol`) when `self.positive`, on the same centered/√w-rescaled design; intercept recovery unchanged. Mirrors sklearn's `positive=True` L-BFGS-B optimum (`_ridge.py:329`). `n_iter_=Some(iters)` on the positive path. Oracle tests `ridge_classifier_positive_matches_sklearn` (alpha=1 binary coef `[0.52631579, 0.0]`, intercept `-1.43609023`, all ≥ 0, differs from unconstrained `[0.35294118, -0.23529412]`), `ridge_classifier_positive_false_unchanged`. `cargo test -p ferrolearn-linear` PASS. Split from REQ-7 of blocker #393. |
| REQ-7b-i (class_weight) | SHIPPED | `RidgeClassifier<F>` adds `pub class_weight: ClassWeight<F>` (`enum {None (default), Balanced, Explicit(HashMap<usize,F>)}`) + `with_class_weight` builder, mirroring sklearn `class_weight` (`_ridge.py:1398`) + `compute_class_weight`. In `fit_with_sample_weight`, after the class set is known, a per-class weight is derived (`Balanced`: `n_samples/(n_classes·count[c])`; `Explicit`: `map[c]` else `1.0`; `None`: synthesizes no vector — byte-identical fast path) and multiplied into the base sample weight (`effective[i] = w_class[label_i]·base[i]`), then fed into the EXISTING weighted ridge body. `class_weight=None && sample_weight=None` is byte-identical to the prior fit. Verification (live sklearn 1.5.2, R-CHAR-3, imbalanced 6:2 binary, alpha=1): `balanced` coef `[0.2692307692, 0.2692307692]` intercept `-2.0192307692`; `{0:1,1:3}` coef `[0.2709677419, 0.2709677419]` intercept `-2.0322580645`; `None` `[0.2576687117, 0.2576687117]` `-1.981595092`. Tests `ridge_classifier_class_weight_balanced_matches_sklearn`, `ridge_classifier_class_weight_dict_matches_sklearn`, `ridge_classifier_class_weight_none_unchanged`, `ridge_classifier_class_weight_explicit_equals_balanced`. |
| REQ-7b-ii (solver + solver_) | NOT-STARTED | open prereq blocker #393 (class_weight done — see REQ-7b-i). `RidgeClassifier<F>` has no `solver` selection or `solver_` attribute (`_ridge.py:1406-1484`) — the unconstrained fit still uses the dense centered closed-form path. Needs the iterative-solver (lsqr/sparse_cg/sag/saga/lbfgs) substrate. |
| REQ-8 (max_iter/tol/n_iter_) | SHIPPED | `RidgeClassifier<F>` adds `pub max_iter: Option<usize>` (default `None`) and `pub tol: F` (default `1e-4`) with `with_max_iter`/`with_tol` builders; `FittedRidgeClassifier<F>` adds `n_iter_: Option<usize>` (always `None` for the direct solver) with `pub fn n_iter(&self) -> Option<usize>`. Mirrors sklearn ctor `max_iter=None, tol=1e-4` (`_ridge.py:1520-1521`) and `n_iter_` (`_ridge.py:1464`); params are no-ops for the direct solver matching sklearn's `n_iter_=None` on the direct path. Test: `ridge_classifier_max_iter_tol_niter_defaults_and_builders`. Closes #394. |
| REQ-9 (sample_weight) | SHIPPED | `pub fn fit_with_sample_weight in ridge_classifier.rs` (`RidgeClassifier::fit_with_sample_weight(x, y, sample_weight: Option<&Array1<F>>)`) forwards per-sample weights into the underlying weighted ridge on the indicator matrix `Y`: weighted offsets `x_off[j]=Σwᵢx[i,j]/Σwᵢ`, `y_off[t]=Σwᵢ·Y[i,t]/Σwᵢ` (fit_intercept), centering, then `√wᵢ` row-rescale of `X`/`Y` (sklearn `_rescale_data`, `_ridge.py:682-688`), per-target `linalg::solve_ridge` with `alpha` UNSCALED, `intercept[t]=y_off[t]−Σⱼ x_off[j]·coef[j,t]`; `fit_intercept=false` skips centering (raw `√w`-rescale, intercept 0). `Fit::fit` delegates `fit_with_sample_weight(x, y, None)` (None byte-identical to the historic centering + `solve_ridge` body). Mirrors `RidgeClassifier.fit(X, y, sample_weight=None)` (`_ridge.py:1220`) forwarding through `_prepare_data` (`_ridge.py:1305-1307`) into `_BaseRidge.fit`. Oracle tests `ridge_classifier_sample_weight_matches_sklearn` (alpha=1 binary coef `[0.25333333, 0.36]`, intercept `-1.70666667`, differs from unweighted `[0.31840796, 0.31840796]`), `ridge_classifier_none_sample_weight_equals_unweighted` (byte-identical guard). `cargo test -p ferrolearn-linear` PASS. Closes #395. |
| REQ-10 (RidgeClassifierCV) | NOT-STARTED | open prereq blocker #395. `class RidgeClassifierCV` (`_ridge.py:2676`) is in this route's `parity_ops` but no ferrolearn type exists for it. |
| REQ-11 (ferray substrate) | NOT-STARTED | open prereq blocker #396. The module computes on `ndarray` (`Array1`/`Array2`) and the in-crate `crate::linalg::solve_ridge`, not on `ferray-core`/`ferray::linalg` (R-SUBSTRATE-2). |

## Architecture
The unfitted estimator `RidgeClassifier<F>` (fields `alpha`, `fit_intercept`; builders `new`/
`with_alpha`/`with_fit_intercept`) implements `Fit<Array2<F>, Array1<usize>>`. `fit` collects
and sorts the unique labels (`classes`), sets `is_binary = classes.len() == 2`, and builds
`y_indicator` of shape `(n_samples, n_targets)` where `n_targets == 1` for binary (`±1`) or
`classes.len()` for multiclass (one-hot) — the structural mirror of sklearn's
`LabelBinarizer(pos_label=1, neg_label=-1)` (`_ridge.py:1300`). When `fit_intercept`, `X` and
`y_indicator` are centered by their column means; the per-class coefficient matrix is solved
column-by-column with `crate::linalg::solve_ridge` (`linalg.rs`, a `pub(crate)` closed-form
solver), and `intercept_vec = y_mean - x_mean·coef`.

`FittedRidgeClassifier<F>` stores `coef_matrix` `(n_features, n_targets)`, `intercept_vec`,
the first-column `coefficients`/`intercept` (for `HasCoefficients`), `classes`, `is_binary`,
and `n_features`. `Predict<Array2<F>>` returns `Array1<usize>` via sign (binary) / argmax
(multiclass) of `X @ coef_matrix + intercept_vec`. `decision_function` exposes the raw
`(n_samples, n_classes)` scores. Invariant: `argmax` of a `decision_function` row equals the
`predict` label.

Divergence note for the critic: sklearn's binary `coef_` has shape `(1, n_features)` and
`classes_` is the sorted label array; ferrolearn stores `coef_matrix` as
`(n_features, n_targets)` (transposed relative to sklearn's `(n_classes, n_features)` `coef_`)
and `classes` as `Vec<usize>`. The Python wrapper's attribute exposure (`coef_` orientation,
`classes_` dtype) should be checked against sklearn at the binding boundary.

## Verification
- `cargo test -p ferrolearn-linear` — `ridge_classifier::tests` (`test_binary_classification`,
  `test_multiclass_classification`, `test_negative_alpha`, `test_alpha_zero`,
  `test_single_class_error`, `test_has_coefficients`, `test_has_classes`,
  `test_shape_mismatch`, `test_predict_feature_mismatch`) — all PASS at baseline.
- sklearn oracle (live, 1.5.2): `python3 -c "import numpy as np; from sklearn.linear_model import RidgeClassifier; X=np.array([[1,1],[1,2],[2,1],[2,2],[8,8],[8,9],[9,8],[9,9]],float); y=np.array([0,0,0,0,1,1,1,1]); m=RidgeClassifier(alpha=1.0).fit(X,y); print(m.coef_.shape, m.predict(X).tolist(), m.intercept_.tolist())"`
  → `(1, 2) [0, 0, 0, 0, 1, 1, 1, 1] [-1.407035175879397]`.
- Multiclass oracle: 3-class separated data, `alpha=0.1` → `coef_` shape `(3,2)`, `predict`
  `[0,0,0,1,1,1,2,2,2]`, `classes_ == [0,1,2]`.
- REQ-7..11 have no green verification (no implementation); each is NOT-STARTED with the
  blocker cited above.
