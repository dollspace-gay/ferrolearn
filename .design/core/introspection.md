# Introspection Traits

<!--
tier: 3-component
status: draft
baseline-commit: 9494fa1d958fe81342a8d2114819a86c3f073644
upstream-paths:
  - sklearn/base.py
  - sklearn/utils/_param_validation.py
-->

## Summary
`ferrolearn-core`'s `introspection` module defines the cross-crate vocabulary by which a
*fitted* estimator exposes its learned state for uniform inspection. It mirrors scikit-learn's
de-facto convention of trailing-underscore fitted attributes (`coef_`, `intercept_`,
`feature_importances_`, `classes_`, `n_classes_`) that estimators set during `fit` and that
downstream code reads back through duck-typed attribute access (`sklearn/base.py`). Where
scikit-learn relies on dynamically-assigned attributes guarded by `check_is_fitted`, ferrolearn
encodes the same contract as three traits implemented on `Fitted*` types: `HasCoefficients<F>`,
`HasFeatureImportances<F>`, and `HasClasses`.

## Requirements
- REQ-1: A fitted linear estimator exposes its learned coefficient vector and scalar intercept
  through a uniform interface, mirroring scikit-learn's `coef_` / `intercept_` fitted attributes
  set during `fit`.
- REQ-2: A fitted tree-based or ensemble estimator exposes per-feature importance scores through
  a uniform interface, mirroring scikit-learn's `feature_importances_` fitted property.
- REQ-3: A fitted classifier exposes the sorted set of class labels it was trained on and their
  count, mirroring scikit-learn's `classes_` / `n_classes_` fitted attributes derived via
  `np.unique(y)`.
- REQ-4: The introspection return types are carried by the ferray array substrate (ferray
  `Array1`) rather than the legacy `ndarray::Array1`, per R-SUBSTRATE-1/2.

## Acceptance criteria
- AC-1: A value of type `FittedLinearRegression<f64>` can be passed by reference where
  `&dyn HasCoefficients<f64>` or a generic `T: HasCoefficients<f64>` is expected, and
  `coefficients()` / `intercept()` return the same numbers sklearn's `LinearRegression.coef_` /
  `.intercept_` produce on the same `(X, y)` (within R-DEV-1 tolerance). Verified by
  `cargo test -p ferrolearn-linear` + live sklearn oracle.
- AC-2: A `FittedDecisionTreeClassifier<f64>` / `FittedRandomForestClassifier<f64>` satisfies
  `HasFeatureImportances<f64>` and `feature_importances()` returns a non-negative vector summing
  to ~1.0, matching sklearn's `feature_importances_`. Verified by `cargo test -p ferrolearn-tree`.
- AC-3: A `FittedLogisticRegression<f64>` (and other classifiers) satisfies `HasClasses` and
  `classes()` returns the sorted unique label set with `n_classes() == classes().len()`, matching
  sklearn's `classes_` ordering. Verified by `cargo test -p ferrolearn-linear`.
- AC-4: Every introspection trait return type that today is `ndarray::Array1<F>` is the ferray
  `Array1<F>` analog; no `ndarray` appears in `introspection.rs`'s signatures.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (HasCoefficients ↔ coef_/intercept_) | SHIPPED | impl `pub trait HasCoefficients in introspection.rs` (`fn coefficients(&self) -> &Array1<F>; fn intercept(&self) -> F`) mirrors sklearn `LinearRegression` setting `self.coef_` at `sklearn/linear_model/_base.py:691` (`self.coef_ = np.ravel(self.coef_)`) and `self.intercept_` via `_set_intercept` at `sklearn/linear_model/_base.py:692`. Cross-crate non-test consumer: `impl HasCoefficients for FittedLinearRegression in linear_regression.rs` (ferrolearn-linear), with further impls `impl HasCoefficients for FittedRidge in ridge.rs` and `impl HasCoefficients for FittedLogisticRegression in logistic_regression.rs`. The trait is a `pub use` of `ferrolearn_core::lib` (external estimator crates + downstream users are the consumers). Verification: `cargo build -p ferrolearn-linear` (green). |
| REQ-2 (HasFeatureImportances ↔ feature_importances_) | SHIPPED | impl `pub trait HasFeatureImportances in introspection.rs` (`fn feature_importances(&self) -> &Array1<F>`) mirrors sklearn `BaseDecisionTree.feature_importances_` property at `sklearn/tree/_classes.py:671` (`def feature_importances_(self): ... return self.tree_.compute_feature_importances()`) and the forest analog `BaseForest.feature_importances_` at `sklearn/ensemble/_forest.py:653`. Cross-crate non-test consumer: `impl HasFeatureImportances for FittedDecisionTreeClassifier in decision_tree.rs` (ferrolearn-tree), with `impl HasFeatureImportances for FittedRandomForestClassifier in random_forest.rs` and `impl HasFeatureImportances for FittedGradientBoostingClassifier in gradient_boosting.rs`. Verification: `cargo build -p ferrolearn-tree` (green). |
| REQ-3 (HasClasses ↔ classes_/n_classes_) | SHIPPED | impl `pub trait HasClasses in introspection.rs` (`fn classes(&self) -> &[usize]; fn n_classes(&self) -> usize`) mirrors the scikit-learn `ClassifierMixin` convention (`sklearn/base.py:703`) of deriving `self.classes_ = np.unique(y)` — e.g. `LogisticRegression.fit` at `sklearn/linear_model/_logistic.py:1232` (`self.classes_ = np.unique(y)`). Cross-crate non-test consumers: `impl HasClasses for FittedLogisticRegression in logistic_regression.rs` and `impl HasClasses for FittedDecisionTreeClassifier in decision_tree.rs`, plus `impl HasClasses for FittedRandomForestClassifier in random_forest.rs`, `impl HasClasses for FittedGaussianNB in gaussian.rs` (ferrolearn-bayes), and `impl HasClasses for FittedKNeighborsClassifier in knn.rs` (ferrolearn-neighbors). Verification: `cargo build -p ferrolearn-linear -p ferrolearn-tree` (green). |
| REQ-4 (ferray substrate for return types) | NOT-STARTED | open prereq blocker #359. The trait return types `coefficients(&self) -> &Array1<F>` and `feature_importances(&self) -> &Array1<F>` use `use ndarray::Array1` (the legacy numpy-like substrate, R-SUBSTRATE-1); the destination is ferray's `Array1`. Migrating the *return type* forces every implementing `Fitted*` estimator (which today stores `ndarray::Array1` fields) onto ferray simultaneously, so it cascades through ferrolearn-linear + ferrolearn-tree (+ bayes/neighbors for the storage of `classes`) and cannot be done in isolation at the trait level. The `ndarray::Array1` return type is grandfathered-transitional per R-SUBSTRATE-4 until those crates reach their ferray iteration. |

## Architecture

The module is a pure trait-vocabulary unit: it declares behavior and owns no estimator state.
The three traits are re-exported at the crate root (`pub use introspection::{HasClasses,
HasCoefficients, HasFeatureImportances}` in `lib.rs`), making them the public boundary API that
every downstream estimator crate implements on its `Fitted*` types.

**Mapping to scikit-learn.** scikit-learn has no equivalent trait layer — it uses Python's
duck typing: an estimator sets `self.coef_` during `fit` (`sklearn/linear_model/_base.py:691`),
and any consumer reads `estimator.coef_` directly, with `check_is_fitted` raising
`NotFittedError` if absent. ferrolearn moves the "fitted-ness" guarantee into the type system:
the trait is implemented only on `Fitted*` types produced by a successful `fit`, so the
unfitted→fitted distinction (sklearn's runtime `check_is_fitted`) becomes a compile-time
distinction. This is an R-DEV-7 deviation (Rust analog materially better) that preserves the
observable contract: the *values* returned match sklearn's fitted attributes array-by-array.

- `HasCoefficients<F>` (`pub trait HasCoefficients in introspection.rs`): `coefficients(&self)
  -> &Array1<F>` mirrors `coef_`; `intercept(&self) -> F` mirrors `intercept_`. Note the scalar
  intercept return type pins the *single-output* contract: sklearn's `intercept_` is a scalar for
  single-output regressors and an array for multi-output / multinomial. The single-output trait is
  the grandfathered boundary API; multi-output introspection (where sklearn's `coef_` is 2-D and
  `intercept_` is a vector, `sklearn/linear_model/_base.py:290`) is owned by the individual
  estimators, not this trait.
- `HasFeatureImportances<F>` (`pub trait HasFeatureImportances in introspection.rs`):
  `feature_importances(&self) -> &Array1<F>` mirrors the `feature_importances_` *property*
  (`sklearn/tree/_classes.py:671`, `sklearn/ensemble/_forest.py:653`). sklearn computes this
  lazily from `tree_.compute_feature_importances()`; ferrolearn estimators store the precomputed
  vector and return a reference. The invariant — non-negative, sums to 1.0 — is the estimator's
  responsibility, documented on the trait method.
- `HasClasses` (`pub trait HasClasses in introspection.rs`): non-generic over `F` because labels
  are integer class indices. `classes(&self) -> &[usize]` mirrors `classes_` (sorted unique labels
  from `np.unique(y)`, `sklearn/linear_model/_logistic.py:1232`); `n_classes(&self) -> usize`
  mirrors `len(classes_)` / per-estimator `n_classes_`. The sorted ordering is load-bearing
  (R-DEV-3): `predict_proba` column order and `decision_function` sign convention are defined
  relative to `classes_` ordering in sklearn, so consumers rely on `classes()` being sorted.

**Substrate.** Per R-SUBSTRATE-1, the destination array type is ferray's `Array1`, not
`ndarray::Array1`. The current `use ndarray::Array1` in `introspection.rs` is the wrong substrate
but is grandfathered-transitional (R-SUBSTRATE-4): because the return types are references into
estimator-owned storage, the trait signature cannot migrate ahead of the estimators that own that
storage. Tracked as REQ-4 / blocker #359.

## Verification

The introspection module defines no executable logic of its own (no unit tests in
`introspection.rs`; `cargo test -p ferrolearn-core --lib introspection` runs 0 tests by design —
a trait declaration has nothing to assert). Its contract is exercised entirely through its
cross-crate implementations:

- `cargo build -p ferrolearn-linear -p ferrolearn-tree` — green at baseline
  `9494fa1d958fe81342a8d2114819a86c3f073644`; confirms `HasCoefficients` /
  `HasFeatureImportances` / `HasClasses` are implementable and implemented across crates.
- `cargo test -p ferrolearn-linear` — pins REQ-1/REQ-3 value-contracts (coefficients, intercept,
  classes ordering) against the live sklearn oracle, e.g.
  `python3 -c "from sklearn.linear_model import LinearRegression; import numpy as np; m=LinearRegression().fit(X,y); print(m.coef_.tolist(), m.intercept_)"`
  compared to `FittedLinearRegression::coefficients()` / `::intercept()`.
- `cargo test -p ferrolearn-tree` — pins REQ-2 (feature importances non-negative, sum≈1, matching
  sklearn `DecisionTreeClassifier.feature_importances_`).
- `cargo test -p ferrolearn-bayes -p ferrolearn-neighbors` — additional `HasClasses` consumers
  (REQ-3) for naive-Bayes and KNN classifiers.

REQ-4 has no green verification (blocker #359 open); it remains NOT-STARTED until ferrolearn-linear
and ferrolearn-tree are migrated to the ferray substrate, at which point AC-4 becomes checkable as
`grep -L ndarray ferrolearn-core/src/introspection.rs`.
