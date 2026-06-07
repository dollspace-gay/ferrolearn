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
`sklearn/base.py`. It **also** carries the shape/finite/consistency/min-samples
slice of `validation.py` as free functions — `assert_all_finite`,
`check_consistent_length`, `column_or_1d`, `check_array`, and `check_x_y`
(REQ-4) — whose dtype-coercion, sparse, `copy`/`force_writeable`, and
`multi_output` sub-behaviors remain NOT-STARTED (single-dtype, dense,
borrow-based Rust functions need no runtime dtype/writeability dance).

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
| REQ-1 (shape introspection) | SHIPPED | The `Dataset` trait (`pub trait Dataset in dataset.rs`) and its blanket `impl<F> Dataset for Array2<F> in dataset.rs` are oracle-correct (`n_samples` → `self.nrows()`, `n_features` → `self.ncols()`; live oracle `_num_samples(zeros(100,10)) == 100`, `_num_features == 10`), mirroring `_num_samples` (`validation.py:364`, dense path `return x.shape[0]` at `:388`) and `_num_features` (`validation.py:311`, dense path `return X.shape[1]` at `:346`). Cross-crate non-test production consumer: `impl Dataset for CsrMatrix in ferrolearn-sparse/src/csr.rs`. Per goal.md S5/R-DOC-5 the `Dataset` trait + its dense impl are grandfathered existing public boundary API with a real cross-crate consumer → SHIPPED (the prior NOT-STARTED was over-strict). |
| REQ-2 (is_sparse) | SHIPPED | `is_sparse` returns `false` in `impl Dataset for Array2<F> in dataset.rs` and `true` in `impl<F> Dataset for CsrMatrix<F> in csr.rs` (`ferrolearn-sparse`). The dense/sparse split across the two grandfathered cross-crate impls is the production discrimination on storage format (R-SUBSTRATE-4 transitional substrate). Same S5/R-DOC-5 grandfathering as REQ-1. |
| REQ-3 (ferray substrate) | SHIPPED | `impl<F> Dataset for ferray::aliases::Array2<F> in dataset.rs` (bound `F: ferray::Element`; `n_samples`/`n_features` via `self.shape()[0]`/`self.shape()[1]`, `is_sparse` → `false`), on ferray's array type whose `.shape() -> &[usize]`, per R-SUBSTRATE-1. `ferrolearn-core/Cargo.toml` declares `ferray = { workspace = true }` (umbrella crate, the `import numpy as np` parallel). `cargo build -p ferrolearn-core` compiles with ferray in the graph — the first real ferray integration in ferrolearn. The non-test production consumer of the new impl is the same cross-crate `impl Dataset for CsrMatrix in csr.rs` that satisfies REQ-1/REQ-2 (the trait it implements is now also implemented on the ferray substrate). |
| REQ-4 (validation surface) | SHIPPED | `dataset.rs` adds the free fns `assert_all_finite` / `assert_all_finite_2d` (mirror `assert_all_finite`, `validation.py:175`), `check_consistent_length` (mirrors `validation.py:436`), `column_or_1d` / `column_or_1d_view` (mirror `validation.py:1348`), `check_array` / `check_array_default` (mirror the shape/finite/min-samples/min-features contract of `validation.py:718`), and `check_x_y` (mirrors `validation.py:1154`). Non-test production consumer: `check_consistent_length` is called by `Fit::fit for Pipeline in pipeline.rs`, which rejects mismatched `X`/`y` `n_samples` before fitting any step (mirroring `check_consistent_length`, `validation.py:1320`). Accept/reject parity is verified against the live sklearn 1.5.2 oracle by the `test_*_oracle_parity` tests in `dataset.rs` and the pipeline guard tests `test_pipeline_rejects_inconsistent_x_y` / `test_pipeline_accepts_consistent_x_y` in `pipeline.rs`. Sub-behaviors scoped out as NOT-STARTED (R-DEFER-2): dtype coercion / `dtype="numeric"` (`validation.py:986`), sparse `accept_sparse` (`validation.py:870`), `copy` / `force_writeable` (`validation.py:1071`), and `multi_output=True` y handling (`validation.py:1327`) — the Rust functions are single-dtype, dense, and borrow-based, so these numpy/CPython-era affordances have no Rust analog here. |

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
- `impl<F> Dataset for ferray::aliases::Array2<F> in dataset.rs` (bound
  `F: ferray::Element`) — the dense implementation on the destination ferray
  substrate (R-SUBSTRATE-1). `n_samples`/`n_features` read `self.shape()[0]` /
  `self.shape()[1]` and `is_sparse` is a constant `false`. The element bound is
  `ferray::Element` (not `num_traits::Float`): ferray's `Array<T: Element, D>`
  is only defined for `Element` types — a sealed trait `Float` does not imply —
  so the existing `Float + Send + Sync + 'static` bound cannot name the ferray
  type. `Element` already requires `Send + Sync + 'static`.

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

### REQ state

- **REQ-1 / REQ-2 SHIPPED (grandfathered boundary API).** The `Dataset` trait
  and its dense `ndarray::Array2` impl are existing public boundary API with a
  real cross-crate consumer — `impl Dataset for CsrMatrix in csr.rs`
  (`ferrolearn-sparse`), which both implements the trait and supplies the
  `is_sparse()==true` half of the dense/sparse discrimination. Per goal.md
  S5/R-DOC-5, existing public API surface with a real consumer is SHIPPED; the
  prior NOT-STARTED was over-strict. Both impls are numerically correct against
  the live sklearn oracle.

- **REQ-3 SHIPPED (ferray substrate).** `impl Dataset for ferray::aliases::Array2`
  lands in `dataset.rs` and `ferrolearn-core` depends on the `ferray` umbrella
  crate. This is the first real ferray integration in ferrolearn-core and
  validates the R-SUBSTRATE plan: `cargo build -p ferrolearn-core` compiles with
  the entire ferray graph (ferray-core/ufunc/stats/linalg/random/numpy-interop)
  built. The dense `ndarray::Array2` impl is retained as the transitional
  substrate (R-SUBSTRATE-4) until the broader migration; both substrates now
  satisfy the same `Dataset` contract.

- **REQ-4 SHIPPED (shape/finite/consistency/min-samples slice).** `dataset.rs`
  adds `assert_all_finite` / `assert_all_finite_2d`, `check_consistent_length`,
  `column_or_1d` / `column_or_1d_view`, `check_array` / `check_array_default`,
  and `check_x_y` as total, `Result`-returning free functions (no panics,
  R-CODE-2). The non-test production consumer is `Fit::fit for Pipeline in
  pipeline.rs`, which calls `check_consistent_length(x.nrows(), y.len())` before
  fitting any step — a `Pipeline` now rejects mismatched `X`/`y` up front,
  mirroring sklearn's `check_consistent_length` (`validation.py:1320`). Each
  function's accept/reject decision is pinned against the live sklearn 1.5.2
  oracle. The dtype-coercion, sparse, `copy`/`force_writeable`, and
  `multi_output` sub-behaviors are scoped out as NOT-STARTED sub-points
  (R-DEFER-2): the Rust functions are single-dtype, dense, and borrow-based and
  so do not need those numpy/CPython-era affordances.

## Verification

Commands establishing the (correct-but-unconsumed) state of the impl and the
oracle agreement that grounds the acceptance criteria:

```bash
# dataset unit tests are green (ndarray + ferray substrate)
cargo test -p ferrolearn-core --lib dataset::
#   test dataset::tests::test_array2_f64_dataset ... ok
#   test dataset::tests::test_array2_f32_dataset ... ok
#   test dataset::tests::test_dataset_trait_is_object_safe ... ok
#   test dataset::tests::test_empty_array_dataset ... ok
#   test dataset::tests::test_single_sample_dataset ... ok
#   test dataset::tests::test_ferray_array2_f64_dataset ... ok
#   test dataset::tests::test_ferray_array2_f32_dataset ... ok
#   test dataset::tests::test_ferray_empty_array_dataset ... ok
#   test dataset::tests::test_ferray_single_sample_dataset ... ok
#   test dataset::tests::test_ferray_dataset_trait_is_object_safe ... ok
#   test result: ok. 10 passed; 0 failed

# live sklearn oracle for the shape slice (grounds AC-1)
python3 -c "from sklearn.utils.validation import _num_samples, _num_features; import numpy as np; X=np.zeros((100,10)); print(_num_samples(X), _num_features(X))"
#   100 10

# proof of the missing consumer (grounds the REQ-1/REQ-2 NOT-STARTED claim):
# every Dataset-trait usage outside dataset.rs is a #[cfg(test)] site.
grep -rn ": Dataset\|dyn Dataset\|impl Dataset\|Dataset::" ferrolearn-*/src/ | grep -v 'dataset.rs' | grep -v '#\[cfg(test'
#   (only the csr.rs `impl<F> Dataset for CsrMatrix<F>` — itself test-only consumed)
```

REQ state vs blockers:

- REQ-1 / REQ-2 → SHIPPED (grandfathered boundary API per S5/R-DOC-5, real
  cross-crate `CsrMatrix` consumer). Blocker #354 superseded by the
  grandfathering correction.
- REQ-3 → SHIPPED (blocker #355 closed): `ferrolearn-core` depends on `ferray`
  and `impl Dataset for ferray::aliases::Array2` exists in `dataset.rs`.
- REQ-4 → SHIPPED (blocker #356 closed): `dataset.rs` mirrors `check_array` /
  `check_x_y` / `check_consistent_length` / `assert_all_finite` / `column_or_1d`
  with sklearn-grounded accept/reject tests (length-mismatch `ShapeMismatch`,
  non-finite rejection per `assert_all_finite`, 0-sample/0-feature rejection per
  `ensure_min_samples`/`ensure_min_features`). The non-test consumer is the
  pipeline `check_consistent_length` guard in `pipeline.rs`. Sub-behaviors
  (dtype coercion, sparse, `copy`/`force_writeable`, `multi_output`) remain
  NOT-STARTED per the REQ-status row.
