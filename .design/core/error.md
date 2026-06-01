# FerroError — the unified error/warning contract

<!--
tier: 3-component
status: draft
baseline-commit: 69824090c63b9fe8e7ac3f7a35e3ed591fb1663a
upstream-paths:
  - sklearn/exceptions.py
-->

## Summary
`ferrolearn-core/src/error.rs` is the Rust-idiom translation of scikit-learn's
`sklearn/exceptions.py`. sklearn exposes a class hierarchy split between *errors*
(`NotFittedError`, `UnsetMetadataPassedError`) and *warnings* (`ConvergenceWarning`,
`DataConversionWarning`, `DataDimensionalityWarning`, `EfficiencyWarning`,
`FitFailedWarning`, `UndefinedMetricWarning`, `PositiveSpectrumWarning`,
`InconsistentVersionWarning`). ferrolearn collapses the *error* cases into a single
`#[non_exhaustive] enum FerroError` (thiserror) returned everywhere as
`Result<T, FerroError>`, eliminates `NotFittedError` at compile time via the
traits-crate typestate, and has no warnings channel (a Rust library that returns
`Result` does not emit advisory warnings). This document records that mapping
faithfully — it does not propose code changes.

## Requirements
- REQ-1: ShapeMismatch — a variant carrying `expected`/`actual` dimensions plus a
  `context` string, returned when array dimensions disagree (sklearn raises
  `ValueError` from input validation in this case).
- REQ-2: InsufficientSamples — a variant returned when fewer samples are supplied
  than an operation requires (sklearn raises `ValueError`).
- REQ-3: ConvergenceFailure — a variant whose *name* mirrors sklearn's
  `ConvergenceWarning` (`exceptions.py:64`), carrying `iterations` + `message`.
  error.rs only *defines* the variant; whether a given estimator should error vs
  warn-and-return-best-so-far is sklearn-behavior owned by each estimator crate,
  not by error.rs (see Architecture).
- REQ-4: InvalidParameter — a variant carrying `name` + `reason`, the Rust analog of
  the `ValueError` sklearn raises from `_parameter_constraints` / `_param_validation`.
- REQ-5: NumericalInstability — a variant for NaN/Inf/singular-matrix conditions.
- REQ-6: IoError + SerdeError — variants for persistence and (de)serialization
  failures during data loading / model persistence.
- REQ-7: FerroResult<T> — the `Result<T, FerroError>` alias that is the uniform
  fallible-return contract for the workspace.
- REQ-8: NotFittedError elimination — sklearn's `NotFittedError(ValueError, AttributeError)`
  (`exceptions.py:42`) is a *runtime* exception on predict-before-fit. ferrolearn
  intentionally has NO `FerroError::NotFitted` variant: the typestate in
  `ferrolearn-core/src/traits.rs` makes predict-before-fit a compile error. This is
  a sanctioned R-DEV-4 deviation.
- REQ-9: ShapeMismatchContext — the builder-style constructor for
  `FerroError::ShapeMismatch` (`new`/`expected`/`actual`/`build`).
- REQ-10: Advisory-warning mapping — sklearn's pure `UserWarning`/`RuntimeWarning`
  subclasses (`DataConversionWarning`, `DataDimensionalityWarning`,
  `EfficiencyWarning`, `FitFailedWarning`, `UndefinedMetricWarning`,
  `PositiveSpectrumWarning`, `InconsistentVersionWarning`) have no `FerroError`
  analog because they are non-fatal advisories not part of any `Result` contract.

## Acceptance criteria
- AC-1: `FerroError::ShapeMismatch { expected, actual, context }` exists and its
  `Display` contains the context string and both shapes.
- AC-2: `FerroError::InsufficientSamples { required, actual, context }` exists with a
  `Display` naming both counts.
- AC-3: `FerroError::ConvergenceFailure { iterations, message }` exists; a production
  estimator returns it on non-convergence.
- AC-4: `FerroError::InvalidParameter { name, reason }` exists; a production estimator
  returns it for an out-of-domain hyperparameter (parity with sklearn `KMeans(n_clusters=0)`
  raising `ValueError`).
- AC-5: `FerroError::NumericalInstability { message }` exists and is returned from a
  production linear-algebra path.
- AC-6: `FerroError::IoError(std::io::Error)` (with `#[from]`) and
  `FerroError::SerdeError { message }` exist and are returned from production I/O / serde paths.
- AC-7: `FerroResult<T>` is defined as `Result<T, FerroError>` and `FerroError: Send + Sync`.
- AC-8: There is no `FerroError::NotFitted`; unfitted estimator structs do not implement
  `Predict`/`Transform`, so predict-before-fit fails to compile (trybuild coverage in traits.rs).
- AC-9: `ShapeMismatchContext::new(..).expected(..).actual(..).build()` yields a
  `FerroError::ShapeMismatch` equivalent to the hand-built variant.
- AC-10: No `FerroError` variant corresponds to a sklearn advisory `*Warning`; the
  enum is `#[non_exhaustive]` so a warnings facility could be added later without a break.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (ShapeMismatch) | SHIPPED | impl `FerroError::ShapeMismatch in error.rs` (`#[error("Shape mismatch in {context}: expected {expected:?}, got {actual:?}")]`). Non-test consumer: `fn predict in mean_shift.rs` (`return Err(FerroError::ShapeMismatch { … })` on feature-count mismatch); also `bisecting_kmeans.rs`. Maps sklearn input-validation `ValueError`. Verification: `cargo test -p ferrolearn-core --lib error` (`test_shape_mismatch_display ... ok`). |
| REQ-2 (InsufficientSamples) | SHIPPED | impl `FerroError::InsufficientSamples in error.rs` (`#[error("Insufficient samples: need at least {required}, got {actual} ({context})")]`). Non-test consumer: `fn fit in gaussian.rs` of `ferrolearn-bayes` (`return Err(FerroError::InsufficientSamples { … })` on zero samples); also `affinity_propagation.rs`. Verification: `test_insufficient_samples_display ... ok`. |
| REQ-3 (ConvergenceFailure) | SHIPPED | impl `FerroError::ConvergenceFailure in error.rs` (`#[error("Convergence failure after {iterations} iterations: {message}")]`); name mirrors sklearn `ConvergenceWarning` (`sklearn/exceptions.py:64`). Non-test consumer: `fn jacobi_eigen_symmetric in kernel_pca.rs` of `ferrolearn-decomp` (`Err(FerroError::ConvergenceFailure { … })`); also `incremental_pca.rs`, `lasso.rs`, `affinity_propagation.rs`. error.rs ships the *variant*; warn-vs-error per estimator is pinned downstream, not here. Verification: `test_convergence_failure_display ... ok`. |
| REQ-4 (InvalidParameter) | SHIPPED | impl `FerroError::InvalidParameter in error.rs` (`#[error("Invalid parameter `{name}`: {reason}")]`). Non-test consumer: `fn fit in mini_batch_kmeans.rs` of `ferrolearn-cluster` (`return Err(FerroError::InvalidParameter { name: "n_clusters".into(), reason: "must be at least 1".into() })`). Mirrors sklearn `ValueError` from `_parameter_constraints` (oracle: `KMeans(n_clusters=0).fit(...)` → `ValueError: The 'n_clusters' parameter of KMeans must be an int in the range [1, inf). Got 0`). Verification: `test_invalid_parameter_display ... ok`. |
| REQ-5 (NumericalInstability) | SHIPPED | impl `FerroError::NumericalInstability in error.rs` (`#[error("Numerical instability: {message}")]`). Non-test consumer: `fn cholesky_gpc in gp_classifier.rs` of `ferrolearn-kernel` (`return Err(FerroError::NumericalInstability { … })` on a non-PD matrix); also `birch.rs`, `mini_batch_kmeans.rs`. Verification: `test_numerical_instability_display ... ok`. |
| REQ-6 (IoError + SerdeError) | SHIPPED | impl `FerroError::IoError(#[from] std::io::Error)` and `FerroError::SerdeError { message } in error.rs`. Non-test consumers: `pub fn save_pmml in pmml.rs` of `ferrolearn-io` (`fs::write(path, xml).map_err(FerroError::IoError)`) for IoError; `pub fn fetch_openml in openml.rs` of `ferrolearn-fetch` (`serde_json::from_str(...).map_err(|e| FerroError::SerdeError { … })`) for SerdeError. Verification: `test_io_error_from ... ok`, `test_serde_error_display ... ok`. |
| REQ-7 (FerroResult alias) | SHIPPED | impl `pub type FerroResult<T> = Result<T, FerroError> in error.rs`. Non-test consumers: every routed `fit`/`predict`/`transform` returns `Result<_, FerroError>` (the alias's expansion) — e.g. `fn fit in mini_batch_kmeans.rs`, `fn predict in mean_shift.rs`. `FerroError: Send + Sync` is the typestate prerequisite for `F: Float + Send + Sync + 'static`. Verification: `test_ferro_error_is_send_sync ... ok`. |
| REQ-8 (NotFittedError eliminated, R-DEV-4) | SHIPPED (as deviation) | Sanctioned R-DEV-4 deviation: there is intentionally NO `FerroError::NotFitted` variant in error.rs. Replacement: the typestate in `traits.rs` — `pub trait Predict` / `pub trait Transform` are implemented ONLY on fitted types (`FittedLinearRegression`, `FittedStandardScaler`), never on the unfitted struct, so predict-before-fit is a compile error. Mirrors sklearn `NotFittedError(ValueError, AttributeError)` (`sklearn/exceptions.py:42`; oracle MRO `(NotFittedError, ValueError, AttributeError)`) which is a *runtime* guard via `check_is_fitted`. Non-test consumer of the typestate: `LinearRegression`→`FittedLinearRegression` in `linear_regression.rs` (documented in `.design/core/traits.md` REQ-1, trybuild `predict_unfitted_*`). |
| REQ-9 (ShapeMismatchContext builder) | NOT-STARTED | open prereq blocker #351. impl `struct ShapeMismatchContext` + `new`/`expected`/`actual`/`build in error.rs` exists and is unit-tested, but a workspace-wide grep finds NO non-test production consumer: every shape-mismatch site (e.g. `mean_shift.rs`, `bisecting_kmeans.rs`) constructs `FerroError::ShapeMismatch { … }` directly rather than via the builder. The `.build()` hits in `extra_trees_ensemble.rs` are `rayon::ThreadPoolBuilder::build`, unrelated. Per R-HONEST-2 the builder is dead public API until a real consumer routes through it. |
| REQ-10 (advisory-warning mapping) | SHIPPED (non-applicable mapping) | Documented non-applicable: sklearn's `DataConversionWarning`/`DataDimensionalityWarning`/`EfficiencyWarning`/`FitFailedWarning`/`UndefinedMetricWarning`/`PositiveSpectrumWarning`/`InconsistentVersionWarning` (`sklearn/exceptions.py:64-188`) are all `UserWarning`/`RuntimeWarning` subclasses emitted via Python's `warnings` module — non-fatal advisories outside any return value. ferrolearn returns `Result<T, FerroError>` and has no warnings channel, so these map to nothing in the `Result` contract. The absence is not an observable `Result`-contract gap. `FerroError` is `#[non_exhaustive]`, so a future warnings facility can be added without a breaking change. (Reassess per-warning if a downstream estimator's *value* contract diverges because it suppresses a behavior sklearn warns about — that would be a divergence owned by the estimator crate, not error.rs.) |

## Architecture
error.rs defines one public enum, one alias, and one builder, all `#[cfg(feature = "std")]`-aware
(`IoError` is gated; the crate is otherwise `no_std`-capable via `alloc`).

`pub enum FerroError` is `#[derive(Debug, thiserror::Error)]` and `#[non_exhaustive]`. The
`#[non_exhaustive]` attribute is the structural counterpart to sklearn's open-ended exception
hierarchy: downstream code must use a wildcard match arm, and new variants (including a future
warnings facility for REQ-10) may be added in a minor release. Each variant carries diagnostic
context per the project's "every error variant carries diagnostic context" rule:

- structural variants (`ShapeMismatch`, `InsufficientSamples`) carry the offending dimensions/counts
  plus a `context` string — the Rust analog of the message sklearn's `check_array` /
  `check_consistent_length` bake into a `ValueError`;
- `InvalidParameter { name, reason }` is the analog of the structured message sklearn's
  `_parameter_constraints` machinery composes (`sklearn/utils/_param_validation.py`), where the
  oracle shows `KMeans(n_clusters=0)` → `ValueError: The 'n_clusters' parameter … Got 0`;
- `ConvergenceFailure { iterations, message }` borrows the *name* of sklearn's
  `ConvergenceWarning` (`exceptions.py:64`) but NOT its severity semantics. sklearn's
  `ConvergenceWarning` is a `UserWarning`: the estimator warns and returns the best-so-far fit.
  Whether a ferrolearn estimator should error (return `ConvergenceFailure`) or warn-and-continue
  to match sklearn is a per-estimator behavioral contract owned by that estimator's crate and its
  design doc — error.rs only supplies the vocabulary. This document does not assert any estimator
  matches sklearn's warn-and-continue; those claims live downstream (e.g. `lasso.rs`,
  `affinity_propagation.rs` design docs).

`pub type FerroResult<T> = Result<T, FerroError>` is the uniform fallible-return alias.
`FerroError` is `Send + Sync` (pinned by `test_ferro_error_is_send_sync`), which the generic
bound `F: Float + Send + Sync + 'static` requires for error propagation across rayon boundaries.

The most important structural decision is the *absence* of a `NotFitted` variant (REQ-8). sklearn's
`NotFittedError(ValueError, AttributeError)` (`exceptions.py:42`) is raised at runtime by
`check_is_fitted` when `predict`/`transform` is called before `fit`. ferrolearn moves this guard
to compile time: `traits.rs` implements `Predict`/`Transform` only on the `Fitted*` associated
type returned by `Fit::fit`, so an unfitted struct has no `predict` method at all. This is an
intentional R-DEV-4 deviation (Rust's typestate eliminates a CPython-only footgun); the contract
sklearn users observe — "you cannot predict before fitting" — is preserved, strengthened from a
runtime exception to a compile error.

`ShapeMismatchContext` (REQ-9) is a builder that *would* let call sites accumulate
`context`/`expected`/`actual` and emit `FerroError::ShapeMismatch` via `.build()`. It is fully
implemented and unit-tested but currently has zero non-test callers — call sites construct the
variant directly — so it is classified NOT-STARTED against blocker #351 rather than SHIPPED.

## Verification
Commands establishing the SHIPPED claims (all run from repo root):

```
cargo test -p ferrolearn-core --lib error
# -> 15 passed; 0 failed (test_shape_mismatch_display, test_insufficient_samples_display,
#    test_convergence_failure_display, test_invalid_parameter_display,
#    test_numerical_instability_display, test_io_error_from, test_serde_error_display,
#    test_shape_mismatch_context_builder, test_ferro_error_is_send_sync, ...)
```

Non-test consumer existence (each SHIPPED REQ):
```
grep -rn "FerroError::InvalidParameter"   ferrolearn-cluster/src/mini_batch_kmeans.rs   # REQ-4: fn fit
grep -rn "FerroError::ShapeMismatch"      ferrolearn-cluster/src/mean_shift.rs          # REQ-1: fn predict
grep -rn "FerroError::InsufficientSamples" ferrolearn-bayes/src/gaussian.rs             # REQ-2: fn fit
grep -rn "FerroError::ConvergenceFailure" ferrolearn-decomp/src/kernel_pca.rs           # REQ-3: jacobi_eigen_symmetric
grep -rn "FerroError::NumericalInstability" ferrolearn-kernel/src/gp_classifier.rs      # REQ-5: cholesky_gpc
grep -rn "FerroError::IoError"            ferrolearn-io/src/pmml.rs                      # REQ-6: save_pmml
grep -rn "FerroError::SerdeError"         ferrolearn-fetch/src/openml.rs                # REQ-6: fetch_openml
```

sklearn oracle (REQ-4 / REQ-8 parity):
```
python3 -c "from sklearn.cluster import KMeans; \
  KMeans(n_clusters=0).fit([[1,2],[3,4]])"
# -> ValueError: The 'n_clusters' parameter of KMeans must be an int in the range [1, inf). Got 0
python3 -c "from sklearn.svm import LinearSVC; from sklearn.exceptions import NotFittedError; \
  print([c.__name__ for c in NotFittedError.__mro__[:3]])"
# -> ['NotFittedError', 'ValueError', 'AttributeError']
```

REQ-8 typestate guard is pinned by trybuild in `ferrolearn-core` (`predict_unfitted_*`, see
`.design/core/traits.md`); REQ-9 has no green verification (the builder is unused in production) and
is therefore NOT-STARTED against blocker #351. REQ-10 is a documented non-applicable mapping with no
`Result`-contract obligation to verify.
