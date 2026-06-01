# Estimator Protocol Traits

<!--
tier: 3-component
status: draft
baseline-commit: 5fec18bc95cd2c289c05c0f25f5d19b0d5ba1d5a
upstream-paths:
  - sklearn/base.py
-->

## Summary

`ferrolearn-core/src/traits.rs` is the Rust typestate translation of scikit-learn's
estimator contract defined in `sklearn/base.py`. It defines the five traits that every
ferrolearn estimator and transformer implements — `Fit`, `Predict`, `Transform`,
`FitTransform`, `PartialFit` — mirroring sklearn's `fit`/`predict`/`transform`/
`fit_transform`/`partial_fit` protocol and the `BaseEstimator` + `ClassifierMixin` /
`RegressorMixin` / `TransformerMixin` / `OutlierMixin` mixin surface. The deliberate
deviation (R-DEV-4): ferrolearn encodes "fit before predict/transform" in the type
system — `Fit::fit` returns a distinct `Fitted` associated type, and only that fitted
type implements `Predict`/`Transform` — eliminating sklearn's runtime
`check_is_fitted` / `NotFittedError` footgun entirely. This is a Rust-eliminates-a-footgun
deviation, NOT a numerical/contract divergence.

## Requirements

- REQ-1: A `fit` → fitted typestate that mirrors sklearn's estimator/predict protocol
  (`BaseEstimator.fit` returns the estimator, `predict` is then callable). The unfitted
  configuration struct implements `Fit` and returns a distinct `Fitted` type; the
  unfitted struct never implements `Predict`/`Transform`, so "predict before fit" — which
  sklearn defends against at runtime via `check_is_fitted` raising `NotFittedError` — is a
  compile-time type error. (R-DEV-4 deviation, documented.)
- REQ-2: A `Transform` contract mirroring `TransformerMixin.transform` — a fitted
  transformer maps `X` to a transformed `X_new`.
- REQ-3: A `FitTransform` contract mirroring `TransformerMixin.fit_transform`'s
  "fit then transform" semantics — a single call that fits to `X` and returns the
  transformed `X`, defaulting to `fit(X).transform(X)` in sklearn.
- REQ-4: A `PartialFit` contract mirroring sklearn's `partial_fit` incremental protocol —
  callable repeatedly, each call updating the model with a new batch, chainable into
  predict.
- REQ-5: A `Predict` contract mirroring `BaseEstimator.predict` as exercised by
  `ClassifierMixin.score` (classification labels), `RegressorMixin.score` (regression
  values), and `OutlierMixin.fit_predict` (inlier/outlier labels) — only fitted types
  implement it.

## Acceptance criteria

- AC-1: `cargo test -p ferrolearn-core` passes, including the `trybuild` compile-fail
  fixtures `tests/compile_fail/predict_unfitted_linear.rs`,
  `predict_unfitted_classifier.rs`, `predict_unfitted_kmeans.rs`,
  `transform_unfitted_pca.rs` — proving an unfitted estimator cannot call
  `predict`/`transform` (the typestate analog of `NotFittedError`).
- AC-2: For each trait, at least one non-test production estimator in another workspace
  crate implements it, and at least one is exposed through `ferrolearn-python`.
- AC-3: `Fit::fit(&self, ...)` returns `Result<Self::Fitted, Self::Error>` where
  `Self::Fitted` is a type distinct from `Self`, and `Self::Fitted` (not `Self`) carries
  the `Predict`/`Transform` impl.
- AC-4: `FitTransform<X>: Transform<X>` — the supertrait bound enforces that any
  fit-transform type is also a transform type, mirroring that `fit_transform` returns the
  same `X_new` as `transform`.
- AC-5: `PartialFit::FitResult: Predict<X> + PartialFit<X, Y>` — the associated result is
  bounded to be both predictable and further-partial-fittable, mirroring sklearn's
  chainable `partial_fit(...).partial_fit(...).predict(...)`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (fit→fitted typestate) | SHIPPED | impl `pub trait Fit in traits.rs` (`fn fit(&self, x: &X, y: &Y) -> Result<Self::Fitted, Self::Error>`) mirrors `sklearn/base.py:180` (`BaseEstimator.fit` example returns `self`, then `predict` is callable) and replaces the runtime guard `sklearn/base.py:35` (`check_is_fitted` import) / `sklearn/base.py:1143` (`check_is_fitted(self, "n_features_in_")`). Non-test consumer: `pub struct LinearRegression in linear_regression.rs` returns `FittedLinearRegression`; only the fitted type carries `impl Predict in linear_regression.rs`. Python consumer: `extras.rs` (`let fitted = model.fit(&x_nd, &y_nd)`). Verification: `cargo test -p ferrolearn-core` — `tests/compile_fail/predict_unfitted_linear.rs` (and `_classifier`, `_kmeans`) compile-fail, proving unfitted `predict` is rejected at compile time. |
| REQ-2 (Transform contract) | SHIPPED | impl `pub trait Transform in traits.rs` (`fn transform(&self, x: &X) -> Result<Self::Output, Self::Error>`) mirrors `TransformerMixin` `sklearn/base.py:1010` / its `transform` usage in `fit_transform` `sklearn/base.py:1098`. Non-test consumer: `impl Transform for FittedStandardScaler in standard_scaler.rs`; also `impl Transform for FittedKMeans in kmeans.rs`. Python consumer: registered transformers in `transformers.rs`. Verification: `cargo test -p ferrolearn-core` (`tests/compile_fail/transform_unfitted_pca.rs` pins that unfitted transform is rejected). |
| REQ-3 (FitTransform "fit then transform") | SHIPPED | impl `pub trait FitTransform in traits.rs` with supertrait `FitTransform<X>: Transform<X>` and `fn fit_transform(&self, x: &X) -> Result<Self::Output, Self::FitError>` mirrors `TransformerMixin.fit_transform` `sklearn/base.py:1043` (`return self.fit(X, **fit_params).transform(X)` `sklearn/base.py:1098`). Non-test consumer: `impl FitTransform for StandardScaler in standard_scaler.rs`; 17 preprocess transformers implement it (e.g. `MinMaxScaler in min_max_scaler.rs`, `SimpleImputer in imputer.rs`, `OneHotEncoder in one_hot_encoder.rs`). Verification: `cargo test -p ferrolearn-preprocess`. |
| REQ-4 (PartialFit incremental) | SHIPPED | impl `pub trait PartialFit in traits.rs` (`fn partial_fit(self, x: &X, y: &Y) -> Result<Self::FitResult, Self::Error>`, `FitResult: Predict<X> + PartialFit<X, Y>`) mirrors sklearn's `partial_fit` protocol and the `partial_fit && _is_fitted` re-call path in `_fit_context` `sklearn/base.py:1461-1462` (`fit_method.__name__ == "partial_fit" and _is_fitted(estimator)`). Non-test consumer: `impl PartialFit for SGDClassifier in sgd.rs` (unfitted entry) and `impl PartialFit for FittedSGDClassifier in sgd.rs` (chained re-call); likewise `SGDRegressor`/`FittedSGDRegressor`. Verification: `cargo test -p ferrolearn-linear`. |
| REQ-5 (Predict contract) | SHIPPED | impl `pub trait Predict in traits.rs` (`fn predict(&self, x: &X) -> Result<Self::Output, Self::Error>`, implemented only on fitted types) mirrors `BaseEstimator.predict` as consumed by `ClassifierMixin.score` `sklearn/base.py:764` (`accuracy_score(y, self.predict(X), ...)`), `RegressorMixin.score` `sklearn/base.py:848` (`y_pred = self.predict(X)`), and `OutlierMixin.fit_predict` `sklearn/base.py:1311` (`return self.fit(X, **kwargs).predict(X)`). Non-test consumer: `impl Predict for FittedLinearRegression in linear_regression.rs` (regressor), `impl Predict for FittedGaussianNB in gaussian.rs` (classifier), `impl Predict for FittedEllipticEnvelope in covariance.rs` (outlier detector). Python consumer: `regressors.rs` / `classifiers.rs`. Verification: `cargo test -p ferrolearn-core` (compile-fail fixtures pin unfitted-predict rejection). |

## Architecture

The module defines five object-safe-shaped traits, all generic over the data types they
operate on, threading a numeric parameter `F: Float + Send + Sync + 'static` through each
concrete estimator (per CLAUDE.md). The structural core is a two-state type machine:

- **Unfitted state**: a configuration struct (e.g. `LinearRegression`, `StandardScaler`,
  `KMeans`) holding hyperparameters. It implements `Fit<X, Y>` and — for stateless or
  fit-transform transformers — `FitTransform<X>`. It does **not** implement `Predict` or
  the fitted `Transform`. This is the typestate analog of a freshly-constructed sklearn
  estimator before `fit`, where `predict` would raise `NotFittedError` via
  `check_is_fitted` (`sklearn/base.py:35`, used at `sklearn/base.py:1143`,
  `sklearn/base.py:1189`).
- **Fitted state**: a distinct type (`FittedLinearRegression`, `FittedStandardScaler`,
  `FittedKMeans`) named by the `Fitted` associated type of `Fit`. Only this type
  implements `Predict<X>` / `Transform<X>`. This corresponds to a sklearn estimator after
  `fit` has set its learned attributes (`coef_`, `mean_`, `cluster_centers_`).

`Fit` (`pub trait Fit in traits.rs`) is the entry transition: `fn fit(&self, x: &X, y: &Y)
-> Result<Self::Fitted, Self::Error>`. The `Y = ()` convention encodes unsupervised
estimators (sklearn's `fit(self, X, y=None)`, e.g. `TransformerMixin` example
`sklearn/base.py:1033`). Because `fit` takes `&self` and returns a new owned `Fitted`, the
unfitted config can be reused — matching sklearn's "construct once, `fit` many" usage and
the `clone` semantics in `sklearn/base.py:40` where a fitted estimator's params reproduce
a fresh unfitted clone.

`Predict` (`pub trait Predict in traits.rs`) carries `type Output` so the same trait
serves regression (`Array1<F>`), classification (`Array1<usize>` labels, mirroring the
`classes_`-indexed output of `ClassifierMixin`), and outlier detection (inlier/outlier
labels per `OutlierMixin`, `sklearn/base.py:1235`). sklearn distinguishes these via the
runtime `_estimator_type` tag (`sklearn/base.py:736`, `:803`, `:1259`) and the `score`
method each mixin attaches; ferrolearn distinguishes them by the concrete `Output` type
and the introspection traits (`HasClasses`, etc.) defined elsewhere in the crate.

`Transform` (`pub trait Transform in traits.rs`) mirrors `TransformerMixin.transform`. A
stateful transformer implements it only on the fitted type; a purely stateless transformer
may implement it on the unfitted type directly (the module doc-comment notes this, and
`StandardScaler` itself carries both an unfitted-passthrough `Transform` and the fitted
`Transform` — see `standard_scaler.rs`).

`FitTransform` (`pub trait FitTransform in traits.rs`) is `FitTransform<X>: Transform<X>`,
encoding at the type level the invariant that `fit_transform` and `transform` produce the
same `Output` shape — exactly sklearn's `TransformerMixin.fit_transform` default of
`self.fit(X).transform(X)` (`sklearn/base.py:1098`). The separate `FitError` associated
type allows fit-time failures (bad shape, non-convergence) to differ from transform-time
failures.

`PartialFit` (`pub trait PartialFit in traits.rs`) is `Sized` and consumes `self`,
returning `FitResult: Predict<X> + PartialFit<X, Y>`. This bound makes the chained
incremental protocol — `m.partial_fit(b1,y1)?.partial_fit(b2,y2)?.predict(x)?` — a
type-level guarantee, mirroring sklearn's reusable `partial_fit` whose re-entry on an
already-fitted estimator skips param re-validation (`_fit_context`,
`sklearn/base.py:1461`). Both the unfitted estimator (first batch) and the fitted result
(subsequent batches) implement it, so `SGDClassifier` and `FittedSGDClassifier` each carry
an impl (`sgd.rs`).

Invariants:
- The `Fitted` type is distinct from the unfitted type; the compiler's acceptance of a
  program is the proof certificate that every `predict`/`transform` is preceded by a
  compatible `fit`.
- The numeric type `F` threads through `Fit`/`Predict`/`Transform`, so a mixed-precision
  workflow (fit on `f32`, predict on `f64`) is a compile error — no sklearn analog is
  needed because sklearn casts at runtime (`check_array` dtype coercion).
- No trait method panics; all return `Result<_, FerroError>` (CLAUDE.md, R-CODE-2).

## Verification

The SHIPPED claims are established by:

- `cargo test -p ferrolearn-core` — 93 unit + 7 integration + 1 trybuild-suite + doctests
  green at baseline. The trybuild compile-fail fixtures are the mechanical proof of REQ-1
  / REQ-2 / REQ-5's typestate guarantee (the analog of sklearn's runtime
  `check_is_fitted`):
  - `tests/compile_fail/predict_unfitted_linear.rs` — calling `predict` on an unfitted
    regressor fails to compile.
  - `tests/compile_fail/predict_unfitted_classifier.rs` — same for a classifier.
  - `tests/compile_fail/predict_unfitted_kmeans.rs` — same for a clusterer.
  - `tests/compile_fail/transform_unfitted_pca.rs` — calling `transform` on an unfitted
    transformer fails to compile.
- `cargo test -p ferrolearn-preprocess` — exercises the `FitTransform` consumers (REQ-3).
- `cargo test -p ferrolearn-linear` — exercises the `PartialFit` consumers
  `SGDClassifier`/`SGDRegressor` and their fitted re-call impls (REQ-4).
- sklearn-oracle cross-check of the protocol shape (the traits carry no numerics of their
  own; the per-estimator design docs pin numerical parity against the live sklearn
  oracle). The contract this doc pins is structural: unfitted has no `predict`/`transform`;
  fitted does; `fit_transform == fit().transform()`; `partial_fit` chains into `predict`.

All four trait-level commands are green at baseline commit
`5fec18bc95cd2c289c05c0f25f5d19b0d5ba1d5a`; therefore all five REQs are SHIPPED. No prereq
blockers are open against this unit.
