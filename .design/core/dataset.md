# Dataset — shape introspection trait

<!--
tier: 3-component
status: draft
baseline-commit: 12bae51
upstream-paths:
  - sklearn/utils/validation.py
  - sklearn/base.py
-->

## Summary

`ferrolearn-core/src/dataset.rs` defines the `Dataset` trait, a thin
shape-introspection interface (`n_samples`, `n_features`, `is_sparse`) over
tabular data. It mirrors **only the shape slice** of scikit-learn's input
validation: the helpers `_num_samples` (`sklearn/utils/validation.py:364`) and
`_num_features` (`sklearn/utils/validation.py:311`), and the
`n_features_in_` / sample-count concept that sklearn estimators expose via
`sklearn/base.py`. It deliberately does **not** implement the rest of
`validation.py` — `check_array`, `check_X_y`, `check_consistent_length`,
`assert_all_finite`, or `column_or_1d`; those are NOT-STARTED at this unit and
tracked by blocker #356.

## Requirements

- REQ-1: Shape introspection — a `Dataset` trait exposing `n_samples()`
  (row count) and `n_features()` (column count) that mirrors sklearn's
  `_num_samples` / `_num_features`, with a concrete implementation for the
  dense array type and a real (non-test) production consumer.
- REQ-2: Sparse/dense discrimination — `is_sparse()` returning `false` for
  dense arrays and `true` for the sparse matrix type, with a real production
  consumer that branches on it.
- REQ-3: `Dataset` implemented on the ferray substrate — `Dataset` (or an
  equivalent shape trait) implemented for ferray's array type (`ferray-core`,
  whose array exposes `.shape() -> &[usize]`), per R-SUBSTRATE-1.
- REQ-4: Full input-validation surface — `check_array`, `check_X_y`,
  `check_consistent_length`, `assert_all_finite`, and `column_or_1d` mirroring
  `sklearn/utils/validation.py`, covering dtype coercion, finite/NaN/inf
  checking, `ensure_2d`, `ensure_min_samples` / `ensure_min_features`, copy /
  `force_writeable`, `accept_sparse`, `multi_output`, and `y_numeric`.

## Acceptance criteria

- AC-1: For a dense `m × n` array, `n_samples() == m` and `n_features() == n`,
  matching `_num_samples(X) == X.shape[0]` (`validation.py:388`) and
  `_num_features(X) == X.shape[1]` (`validation.py:346`) on the live sklearn
  oracle. Edge cases `(0, 0)` and `(1, n)` agree.
- AC-2: A dense array reports `is_sparse() == false`; the sparse matrix type
  reports `is_sparse() == true`. At least one non-test code path selects
  behavior based on this value.
- AC-3: `Dataset` (or the shape contract) is implemented for the ferray array
  type and `ferrolearn-core` declares a dependency on `ferray`.
- AC-4: `check_X_y(X, y)` raises on inconsistent lengths (mirroring
  `check_consistent_length`, `validation.py:436`) and `check_array` rejects
  non-finite values when `force_all_finite=True` (mirroring `assert_all_finite`,
  `validation.py:175`), verified against the live sklearn oracle.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (shape introspection) | NOT-STARTED | open prereq blocker #354. The `Dataset` trait (`pub trait Dataset in dataset.rs`) and its blanket `impl<F> Dataset for Array2<F> in dataset.rs` exist and are oracle-correct (`n_samples` → `self.nrows()`, `n_features` → `self.ncols()`; live oracle `_num_samples(zeros(100,10)) == 100`, `_num_features == 10`), but **no non-test production code consumes the `Dataset` trait** — no trait bound, no `&dyn Dataset`, no method call outside `#[cfg(test)]` anywhere in the workspace. The `.n_features()` calls in `ferrolearn-io/src/onnx.rs` and elsewhere are inherent fitted-estimator methods (e.g. `FittedGradientBoostingRegressor::n_features`), not `Dataset::n_features`. SHIPPED requires impl + real consumer (R-HONEST-2); the consumer is absent. |
| REQ-2 (is_sparse) | NOT-STARTED | open prereq blocker #354. Both impls exist and are correct: `is_sparse` returns `false` in `impl Dataset for Array2<F> in dataset.rs` and `true` in `impl<F> Dataset for CsrMatrix<F> in csr.rs` (`ferrolearn-sparse`). No non-test code branches on `Dataset::is_sparse()`; the only callers are unit tests. Same missing-consumer gap as REQ-1. |
| REQ-3 (ferray substrate) | NOT-STARTED | open prereq blocker #355. `Dataset` is implemented only for `ndarray::Array2<F>` (the grandfathered transitional substrate, R-SUBSTRATE-4). `ferrolearn-core` does not depend on `ferray`, and no `impl Dataset for` ferray's array type exists. The destination substrate is `ferray-core`'s array type (`.shape() -> &[usize]`), per R-SUBSTRATE-1. |
| REQ-4 (validation surface) | NOT-STARTED | open prereq blocker #356. `dataset.rs` implements only the shape slice. `check_array` (`validation.py:718`), `check_X_y` (`validation.py:1154`), `check_consistent_length` (`validation.py:436`), `assert_all_finite` (`validation.py:175`), and `column_or_1d` (`validation.py:1348`) have no counterpart in `ferrolearn-core`. |

## Architecture

### What dataset.rs is

A single trait with three methods and two implementations:

```text
pub trait Dataset {
    fn n_samples(&self) -> usize;
    fn n_features(&self) -> usize;
    fn is_sparse(&self) -> bool;
}
```

- `pub trait Dataset in dataset.rs` — the introspection contract. It is
  object-safe (verified by `test_dataset_trait_is_object_safe`).
- `impl<F> Dataset for Array2<F> in dataset.rs` (bound `F: Float + Send + Sync +
  'static`) — the dense implementation. `n_samples` delegates to
  `ndarray::Array2::nrows`, `n_features` to `ncols`, and `is_sparse` is a
  constant `false`.
- `impl<F> Dataset for CsrMatrix<F> in csr.rs` (`ferrolearn-sparse`) — the
  sparse implementation. `n_samples`/`n_features` delegate to the CSR row/column
  counts and `is_sparse` is a constant `true`.

### Mapping to scikit-learn

scikit-learn has no `Dataset` type; shape introspection lives in free helper
functions. `Dataset::n_samples` mirrors `_num_samples`
(`validation.py:364`), whose dense path returns `x.shape[0]`
(`validation.py:388`: `return x.shape[0]`). `Dataset::n_features` mirrors
`_num_features` (`validation.py:311`), whose dense path returns `X.shape[1]`
(`validation.py:346`: `return X.shape[1]`). The sklearn helpers carry extra
behavior ferrolearn does not need here — dataframe-interchange detection, list
heuristics, `TypeError` on estimators or 1-D/0-D arrays — because the Rust type
system makes a `Dataset` impl on a 2-D array total. The `n_features_in_`
estimator attribute (`sklearn/base.py`) is the same column-count concept exposed
on fitted estimators; in ferrolearn that lives as inherent methods on the
`Fitted*` structs and is out of scope for this trait.

### Why every REQ is NOT-STARTED here

This unit is an honest underclaim (R-HONEST-3). The trait and both impls are
written and numerically correct against the live sklearn oracle, but:

1. **No production consumer (REQ-1, REQ-2).** SHIPPED requires impl **and** a
   non-test production consumer (R-HONEST-2, R-DOC-1). A workspace-wide search
   for `Dataset` trait bounds, `dyn Dataset`, and `Dataset::`-method calls finds
   only `#[cfg(test)]` sites. Estimators accept `ndarray::Array2` directly and
   use inherent `.nrows()`/`.ncols()`. The trait is currently unconsumed
   plumbing. Tracked by blocker #354.

2. **Wrong substrate (REQ-3).** The impl is on `ndarray::Array2`, the
   transitional substrate (R-SUBSTRATE-4), not ferray's array type. Tracked by
   blocker #355.

3. **Validation surface absent (REQ-4).** The bulk of `validation.py` is not
   translated. Tracked by blocker #356.

The dense `ndarray::Array2` impl is grandfathered as the transitional substrate
per R-SUBSTRATE-4; it stays until the ferray migration iteration (REQ-3 /
#355). When estimator `fit` paths begin routing input-shape checks through
`Dataset`, REQ-1/REQ-2 acquire a real consumer and can move to SHIPPED.

## Verification

Commands establishing the (correct-but-unconsumed) state of the impl and the
oracle agreement that grounds the acceptance criteria:

```bash
# dataset unit tests are green (impl is correct, but tests are not consumers)
cargo test -p ferrolearn-core --lib dataset::
#   test dataset::tests::test_array2_f64_dataset ... ok
#   test dataset::tests::test_array2_f32_dataset ... ok
#   test dataset::tests::test_dataset_trait_is_object_safe ... ok
#   test dataset::tests::test_empty_array_dataset ... ok
#   test dataset::tests::test_single_sample_dataset ... ok
#   test result: ok. 5 passed; 0 failed

# live sklearn oracle for the shape slice (grounds AC-1)
python3 -c "from sklearn.utils.validation import _num_samples, _num_features; import numpy as np; X=np.zeros((100,10)); print(_num_samples(X), _num_features(X))"
#   100 10

# proof of the missing consumer (grounds the REQ-1/REQ-2 NOT-STARTED claim):
# every Dataset-trait usage outside dataset.rs is a #[cfg(test)] site.
grep -rn ": Dataset\|dyn Dataset\|impl Dataset\|Dataset::" ferrolearn-*/src/ | grep -v 'dataset.rs' | grep -v '#\[cfg(test'
#   (only the csr.rs `impl<F> Dataset for CsrMatrix<F>` — itself test-only consumed)
```

A REQ moves to SHIPPED only when its blocker closes:

- REQ-1 / REQ-2 → blocker #354: a non-test production consumer of the `Dataset`
  trait lands (e.g. estimator `fit` input-shape validation routed through it),
  plus a characterization test pinning the shape semantics against the live
  oracle.
- REQ-3 → blocker #355: `ferrolearn-core` depends on `ferray` and
  `impl Dataset for` ferray's array type exists.
- REQ-4 → blocker #356: a validation module mirrors `check_array` /
  `check_X_y` / `check_consistent_length` with sklearn-grounded divergence
  tests (length-mismatch `ValueError`, non-finite rejection per
  `assert_all_finite`).
