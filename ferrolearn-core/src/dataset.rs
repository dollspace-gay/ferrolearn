//! Dataset trait and implementations for common array types.
//!
//! The [`Dataset`] trait provides a uniform interface for querying the
//! shape of tabular data, regardless of the underlying storage format
//! (dense or sparse). Implementations are provided for the transitional
//! `ndarray::Array2<F>` substrate and the destination ferray substrate
//! (`ferray::aliases::Array2<F>`, per R-SUBSTRATE-1).
//!
//! ## REQ status
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (shape introspection) | SHIPPED | `impl Dataset for ndarray::Array2 in dataset.rs` (`n_samples` → `self.nrows()`, `n_features` → `self.ncols()`), mirroring `_num_samples` (`validation.py:364`, dense path `return x.shape[0]` at `:388`) and `_num_features` (`validation.py:311`, dense path `return X.shape[1]` at `:346`). Cross-crate production consumer: `impl Dataset for CsrMatrix in ferrolearn-sparse/src/csr.rs`. |
//! | REQ-2 (is_sparse) | SHIPPED | `is_sparse` returns `false` in `impl Dataset for ndarray::Array2 in dataset.rs` and `true` in `impl Dataset for CsrMatrix in ferrolearn-sparse/src/csr.rs`; the dense/sparse split across the two impls is the production discrimination. |
//! | REQ-3 (ferray substrate) | SHIPPED | `impl Dataset for ferray::aliases::Array2 in dataset.rs` (`n_samples`/`n_features` via `self.shape()[0]`/`self.shape()[1]`, `is_sparse` → `false`), on ferray's array type (`.shape() -> &[usize]`), per R-SUBSTRATE-1. `ferrolearn-core` declares `ferray = { workspace = true }`. |
//! | REQ-4 (validation surface) | SHIPPED | Free fns in `dataset.rs`: `assert_all_finite`/`assert_all_finite_2d` (mirror `assert_all_finite`, `validation.py:175`/`_assert_all_finite` element check `:147`), `check_consistent_length` (mirrors `validation.py:436`), `column_or_1d`/`column_or_1d_view` (mirror `validation.py:1348`, `(n,1)`→ravel, else Err), `check_array`/`check_array_default` (mirror the shape/finite/min-samples/min-features contract of `validation.py:718` — finite `:1063`, min-samples `:1084`, min-features `:1093`), `check_x_y` (mirrors `validation.py:1154` = `check_array`+`column_or_1d`+`assert_all_finite(y)`+`check_consistent_length`). Non-test production consumer: `check_consistent_length` called in `Fit::fit for Pipeline in pipeline.rs` (rejects mismatched X/y `n_samples` before fitting any step, mirroring `sklearn/utils/validation.py:1320`). Accept/reject parity verified against the live sklearn 1.5.2 oracle (`test_*_oracle_parity` in `dataset.rs`). NOT-STARTED sub-behaviors (scoped out, R-DEFER-2): dtype coercion (`validation.py:986`), sparse `accept_sparse` (`:870`), `copy`/`force_writeable` (`:1071`), `multi_output=True` y (`:1327`) — the Rust functions are single-dtype/dense/borrow-based and so do not need these. |

#[cfg(not(feature = "std"))]
use alloc::{format, string::String, vec, vec::Vec};

use ferray::aliases::Array2 as FerrayArray2;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use num_traits::Float;

use crate::error::FerroError;

/// A trait for types that represent tabular datasets.
///
/// Provides basic shape information that algorithms need to validate
/// inputs and allocate output buffers.
///
/// # Examples
///
/// ```
/// use ndarray::Array2;
/// use ferrolearn_core::Dataset;
///
/// let data = Array2::<f64>::zeros((100, 10));
/// assert_eq!(data.n_samples(), 100);
/// assert_eq!(data.n_features(), 10);
/// assert!(!data.is_sparse());
/// ```
pub trait Dataset {
    /// Returns the number of samples (rows) in the dataset.
    fn n_samples(&self) -> usize;

    /// Returns the number of features (columns) in the dataset.
    fn n_features(&self) -> usize;

    /// Returns `true` if the dataset uses a sparse representation.
    fn is_sparse(&self) -> bool;
}

/// Blanket implementation of [`Dataset`] for `ndarray::Array2<F>` where
/// `F` is any floating-point type satisfying the ferrolearn float bound.
impl<F> Dataset for Array2<F>
where
    F: Float + Send + Sync + 'static,
{
    fn n_samples(&self) -> usize {
        self.nrows()
    }

    fn n_features(&self) -> usize {
        self.ncols()
    }

    fn is_sparse(&self) -> bool {
        false
    }
}

/// Implementation of [`Dataset`] for the ferray dense array type
/// (`ferray::aliases::Array2<F>`), the destination numpy substrate
/// (R-SUBSTRATE-1).
///
/// ferray's array exposes shape via `.shape() -> &[usize]`; the dense
/// 2-D path mirrors scikit-learn's `_num_samples`
/// (`sklearn/utils/validation.py:364`, returning `x.shape[0]` at `:388`)
/// and `_num_features` (`validation.py:311`, returning `X.shape[1]` at
/// `:346`). The element bound is `F: ferray::Element` — ferray's
/// `Array<T: Element, D>` is only defined for `Element` types, a sealed
/// trait that `num_traits::Float` does not imply, so the existing
/// `Float + Send + Sync + 'static` bound cannot name the ferray type;
/// `Element` itself already requires `Send + Sync + 'static`.
impl<F> Dataset for FerrayArray2<F>
where
    F: ferray::Element,
{
    fn n_samples(&self) -> usize {
        self.shape()[0]
    }

    fn n_features(&self) -> usize {
        self.shape()[1]
    }

    fn is_sparse(&self) -> bool {
        false
    }
}

// ---------------------------------------------------------------------------
// Input validation (mirrors `sklearn/utils/validation.py` @ 1.5.2)
// ---------------------------------------------------------------------------
//
// The functions below translate scikit-learn's input-validation surface
// (`assert_all_finite`, `check_consistent_length`, `column_or_1d`,
// `check_array`, `check_X_y`) into total, `Result`-returning Rust functions.
// sklearn raises `ValueError`; the Rust analog returns `Err(FerroError::...)`
// (no panics, per goal.md R-CODE-2). Each function carries the sklearn
// `file:line` cite for the exact behavior it mirrors.
//
// SCOPED OUT (documented as NOT-STARTED sub-behaviors of REQ-4, with cites):
//   * dtype coercion / `dtype="numeric"` (`validation.py:986-1012`) — the Rust
//     functions are generic over a single float type `F`; there is no runtime
//     dtype to coerce. Mirrored structurally by the `F: Float` bound.
//   * sparse acceptance (`accept_sparse`, `validation.py:870-930`) — these
//     operate on dense `ArrayView`s; the sparse path is owned by
//     `ferrolearn-sparse`.
//   * `copy` / `force_writeable` (`validation.py:1071-1127`) — Rust borrow
//     semantics make the writeability dance unnecessary; callers clone
//     explicitly when they need ownership.
//   * `multi_output=True` y handling (`validation.py:1327-1336`) — `check_x_y`
//     here mirrors the default single-output path (`column_or_1d` on `y`).

/// Throw an error if `x` contains NaN or (±)infinity.
///
/// Mirrors `assert_all_finite` / `_assert_all_finite`
/// (`sklearn/utils/validation.py:175`, element-wise check at `:147`). With the
/// default `allow_nan=False`, sklearn raises `ValueError("Input contains NaN")`
/// for any NaN and `ValueError("Input ... contains infinity or a value too
/// large ...")` for any ±inf; an all-finite array returns without error.
///
/// This Rust analog always disallows both NaN and inf (the sklearn default) and
/// returns [`FerroError::NumericalInstability`] echoing sklearn's wording on the
/// first offending value, or `Ok(())` when every entry is finite.
///
/// # Errors
///
/// Returns [`FerroError::NumericalInstability`] if any element is NaN or
/// infinite.
pub fn assert_all_finite<F>(x: ArrayView1<F>) -> Result<(), FerroError>
where
    F: Float + Send + Sync + 'static,
{
    // sklearn checks NaN before infinity when constructing the message
    // (`validation.py:148-152`): a NaN yields "Input contains NaN", an inf
    // yields "Input contains infinity or a value too large ...".
    if x.iter().any(|v| v.is_nan()) {
        return Err(FerroError::NumericalInstability {
            message: "Input contains NaN".into(),
        });
    }
    if x.iter().any(|v| v.is_infinite()) {
        return Err(FerroError::NumericalInstability {
            message: "Input contains infinity or a value too large for dtype.".into(),
        });
    }
    Ok(())
}

/// Throw an error if `x` (2-D) contains NaN or (±)infinity.
///
/// The 2-D companion of [`assert_all_finite`], used by [`check_array`]. sklearn
/// flattens before checking (`validation.py:141` `X.reshape(-1)`); this iterates
/// every element in row-major order with the same NaN-before-inf precedence.
///
/// # Errors
///
/// Returns [`FerroError::NumericalInstability`] if any element is NaN or
/// infinite.
pub fn assert_all_finite_2d<F>(x: ArrayView2<F>) -> Result<(), FerroError>
where
    F: Float + Send + Sync + 'static,
{
    if x.iter().any(|v| v.is_nan()) {
        return Err(FerroError::NumericalInstability {
            message: "Input contains NaN".into(),
        });
    }
    if x.iter().any(|v| v.is_infinite()) {
        return Err(FerroError::NumericalInstability {
            message: "Input contains infinity or a value too large for dtype.".into(),
        });
    }
    Ok(())
}

/// Check that two arrays have a consistent number of samples (first dimension).
///
/// Mirrors `check_consistent_length` (`sklearn/utils/validation.py:436`):
/// `lengths = [_num_samples(X) for X in arrays]`; if the set of lengths has more
/// than one distinct value it raises `ValueError("Found input variables with
/// inconsistent numbers of samples: %r")`. ferrolearn takes the two sample
/// counts directly (callers pass `X.n_samples()` / `y.len()`), keeping the
/// function total over any [`Dataset`]-like pair without committing to a single
/// array shape.
///
/// # Errors
///
/// Returns [`FerroError::ShapeMismatch`] when `len_a != len_b`, echoing
/// sklearn's "inconsistent numbers of samples" wording in the context field.
pub fn check_consistent_length(len_a: usize, len_b: usize) -> Result<(), FerroError> {
    if len_a != len_b {
        return Err(FerroError::ShapeMismatch {
            expected: vec![len_a],
            actual: vec![len_b],
            context: format!(
                "Found input variables with inconsistent numbers of samples: {:?}",
                [len_a, len_b]
            ),
        });
    }
    Ok(())
}

/// Ravel a column vector or 1-D array to 1-D, else error.
///
/// Mirrors `column_or_1d` (`sklearn/utils/validation.py:1348`): a `(n,)` 1-D
/// array is returned as-is and a `(n, 1)` column vector is raveled to `(n,)`;
/// any other 2-D shape (including a `(1, n)` row vector for `n != 1`) raises
/// `ValueError("y should be a 1d array, got an array of shape ...")`.
///
/// This 2-D entry point accepts an [`ArrayView2`] and applies the column-or-1d
/// rule: shape `(n, 1)` → `Ok` raveled to `(n,)`; anything else → `Err`. (A
/// genuine 1-D input is already the desired shape; callers holding an
/// `Array1<F>` need no validation — use [`column_or_1d_view`] for the
/// already-1-D fast path.)
///
/// # Errors
///
/// Returns [`FerroError::ShapeMismatch`] when the second dimension is not 1,
/// echoing sklearn's "y should be a 1d array" wording.
pub fn column_or_1d<F>(y: ArrayView2<F>) -> Result<Array1<F>, FerroError>
where
    F: Float + Send + Sync + 'static,
{
    let (n_rows, n_cols) = (y.nrows(), y.ncols());
    if n_cols == 1 {
        // `(n, 1)` column vector → ravel to `(n,)` (`validation.py:1393`).
        return Ok(y.column(0).to_owned());
    }
    Err(FerroError::ShapeMismatch {
        expected: vec![n_rows, 1],
        actual: vec![n_rows, n_cols],
        context: format!(
            "y should be a 1d array, got an array of shape {:?} instead.",
            [n_rows, n_cols]
        ),
    })
}

/// The already-1-D fast path of [`column_or_1d`].
///
/// A `(n,)` array is always a valid 1-D target (`validation.py:1391-1392`), so
/// this simply clones the view. Provided so callers that already hold an
/// `ArrayView1<F>` need not reshape into 2-D just to validate.
///
/// # Errors
///
/// Never errors; returns `Ok` with a cloned [`Array1`]. The `Result` return
/// keeps a uniform signature with [`column_or_1d`].
pub fn column_or_1d_view<F>(y: ArrayView1<F>) -> Result<Array1<F>, FerroError>
where
    F: Float + Send + Sync + 'static,
{
    Ok(y.to_owned())
}

/// Validate a 2-D feature matrix: finite, ≥ `ensure_min_samples` rows,
/// ≥ `ensure_min_features` columns.
///
/// Mirrors the shape/finite/min-samples/min-features contract of `check_array`
/// (`sklearn/utils/validation.py:718`): it calls `_assert_all_finite` when
/// `force_all_finite` (`:1063`), rejects fewer than `ensure_min_samples` rows
/// (`:1084`, default 1) with `ValueError("Found array with %d sample(s) ...")`,
/// and rejects fewer than `ensure_min_features` columns (`:1093`, default 1)
/// with `ValueError("Found array with %d feature(s) ...")`. sklearn checks
/// min-samples before min-features; this matches that order.
///
/// Use [`check_array_default`] for the sklearn defaults
/// (`ensure_min_samples=1`, `ensure_min_features=1`, finite required).
///
/// dtype coercion, sparse acceptance, `ensure_2d`, `copy`, and `force_writeable`
/// are scoped out (see the module-level NOT-STARTED notes); the Rust type system
/// makes a `(n, m)` `ArrayView2<F>` already 2-D and single-dtype.
///
/// # Errors
///
/// Returns [`FerroError::NumericalInstability`] for a non-finite entry (when
/// `force_all_finite`), or [`FerroError::InsufficientSamples`] /
/// [`FerroError::InvalidParameter`] when the row / column count is below the
/// requested minimum.
pub fn check_array<F>(
    x: ArrayView2<F>,
    force_all_finite: bool,
    ensure_min_samples: usize,
    ensure_min_features: usize,
) -> Result<(), FerroError>
where
    F: Float + Send + Sync + 'static,
{
    let (n_samples, n_features) = (x.nrows(), x.ncols());

    // sklearn checks finiteness (`:1063`) before the min-samples/min-features
    // gates (`:1084`, `:1093`).
    if force_all_finite {
        assert_all_finite_2d(x)?;
    }

    // ensure_min_samples (`validation.py:1084-1091`).
    if ensure_min_samples > 0 && n_samples < ensure_min_samples {
        return Err(FerroError::InsufficientSamples {
            required: ensure_min_samples,
            actual: n_samples,
            context: format!(
                "Found array with {} sample(s) (shape=[{}, {}]) while a minimum of {} is required.",
                n_samples, n_samples, n_features, ensure_min_samples
            ),
        });
    }

    // ensure_min_features (`validation.py:1093-1100`).
    if ensure_min_features > 0 && n_features < ensure_min_features {
        return Err(FerroError::InvalidParameter {
            name: "ensure_min_features".into(),
            reason: format!(
                "Found array with {} feature(s) (shape=[{}, {}]) while a minimum of {} is required.",
                n_features, n_samples, n_features, ensure_min_features
            ),
        });
    }

    Ok(())
}

/// [`check_array`] with scikit-learn's defaults: finite required,
/// `ensure_min_samples=1`, `ensure_min_features=1`
/// (`sklearn/utils/validation.py:718` signature defaults).
///
/// # Errors
///
/// See [`check_array`].
pub fn check_array_default<F>(x: ArrayView2<F>) -> Result<(), FerroError>
where
    F: Float + Send + Sync + 'static,
{
    check_array(x, true, 1, 1)
}

/// Input validation for standard estimators: validate `X`, validate `y` as a
/// 1-D target, and check `X`/`y` have consistent sample counts.
///
/// Mirrors `check_X_y` (`sklearn/utils/validation.py:1154`): `X = check_array(X,
/// ...)` (`:1301`), `y = _check_y(y, ...)` which for the default single-output
/// path is `column_or_1d(y)` + `_assert_all_finite(y)` (`:1337-1340`), then
/// `check_consistent_length(X, y)` (`:1320`). The headline contract: it rejects
/// (a) non-finite `X`, (b) `X` with 0 samples / 0 features, (c) `y` whose shape
/// is not column-or-1d, (d) `len(X) != len(y)`.
///
/// This entry point takes an already-1-D `y` ([`ArrayView1`]); for a `(n, 1)`
/// column-vector `y`, call [`column_or_1d`] first. `X` is validated with the
/// sklearn defaults ([`check_array_default`]); `y` is checked finite. Returns
/// the validated, owned `(X, y)` pair so callers can take ownership without a
/// second clone.
///
/// # Errors
///
/// Returns [`FerroError::NumericalInstability`] for non-finite `X` or `y`,
/// [`FerroError::InsufficientSamples`] / [`FerroError::InvalidParameter`] for an
/// `X` below the sample/feature minimum, and [`FerroError::ShapeMismatch`] when
/// `X` and `y` have different sample counts.
pub fn check_x_y<F>(
    x: ArrayView2<F>,
    y: ArrayView1<F>,
) -> Result<(Array2<F>, Array1<F>), FerroError>
where
    F: Float + Send + Sync + 'static,
{
    // check_array(X) with sklearn defaults (`validation.py:1301`).
    check_array_default(x)?;
    // _check_y: column_or_1d(y) + _assert_all_finite(y) (`validation.py:1339-1340`).
    let y_validated = column_or_1d_view(y)?;
    assert_all_finite(y_validated.view())?;
    // check_consistent_length(X, y) (`validation.py:1320`).
    check_consistent_length(x.nrows(), y_validated.len())?;
    Ok((x.to_owned(), y_validated))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_array2_f64_dataset() {
        let data = Array2::<f64>::zeros((50, 12));
        assert_eq!(data.n_samples(), 50);
        assert_eq!(data.n_features(), 12);
        assert!(!data.is_sparse());
    }

    #[test]
    fn test_array2_f32_dataset() {
        let data = Array2::<f32>::zeros((200, 5));
        assert_eq!(data.n_samples(), 200);
        assert_eq!(data.n_features(), 5);
        assert!(!data.is_sparse());
    }

    #[test]
    fn test_empty_array_dataset() {
        let data = Array2::<f64>::zeros((0, 0));
        assert_eq!(data.n_samples(), 0);
        assert_eq!(data.n_features(), 0);
    }

    #[test]
    fn test_single_sample_dataset() {
        let data = Array2::<f64>::zeros((1, 100));
        assert_eq!(data.n_samples(), 1);
        assert_eq!(data.n_features(), 100);
    }

    #[test]
    fn test_dataset_trait_is_object_safe() {
        // Verify Dataset can be used as a trait object.
        let data = Array2::<f64>::zeros((10, 3));
        let _: &dyn Dataset = &data;
    }

    // ----- ferray substrate (REQ-3) -----
    //
    // These exercise the real ferray array type via the umbrella crate.
    // Expected shapes are derived from sklearn's dense shape contract:
    // `_num_samples(X) == X.shape[0]` (`validation.py:388`) and
    // `_num_features(X) == X.shape[1]` (`validation.py:346`). The test
    // fns return `FerrayResult` so the `?` operator propagates the
    // fallible ferray constructor without an `.unwrap()` escape hatch.

    use ferray::aliases::Array2 as FArray2;
    use ferray::{FerrayResult, Ix2, zeros as ferray_zeros};

    #[test]
    fn test_ferray_array2_f64_dataset() -> FerrayResult<()> {
        let data: FArray2<f64> = ferray_zeros(Ix2::new([50, 12]))?;
        assert_eq!(data.n_samples(), 50);
        assert_eq!(data.n_features(), 12);
        assert!(!data.is_sparse());
        Ok(())
    }

    #[test]
    fn test_ferray_array2_f32_dataset() -> FerrayResult<()> {
        let data: FArray2<f32> = ferray_zeros(Ix2::new([200, 5]))?;
        assert_eq!(data.n_samples(), 200);
        assert_eq!(data.n_features(), 5);
        assert!(!data.is_sparse());
        Ok(())
    }

    #[test]
    fn test_ferray_empty_array_dataset() -> FerrayResult<()> {
        let data: FArray2<f64> = ferray_zeros(Ix2::new([0, 0]))?;
        assert_eq!(data.n_samples(), 0);
        assert_eq!(data.n_features(), 0);
        Ok(())
    }

    #[test]
    fn test_ferray_single_sample_dataset() -> FerrayResult<()> {
        let data: FArray2<f64> = ferray_zeros(Ix2::new([1, 100]))?;
        assert_eq!(data.n_samples(), 1);
        assert_eq!(data.n_features(), 100);
        assert!(!data.is_sparse());
        Ok(())
    }

    #[test]
    fn test_ferray_dataset_trait_is_object_safe() -> FerrayResult<()> {
        let data: FArray2<f64> = ferray_zeros(Ix2::new([10, 3]))?;
        let _: &dyn Dataset = &data;
        Ok(())
    }

    // ----- Input validation (REQ-4) -----
    //
    // Expected accept/reject decisions come from the LIVE sklearn 1.5.2 oracle
    // (R-CHAR-3), recorded by running the corresponding `try: ...; print('OK')
    // except ValueError: print('RAISE')` snippet against the installed sklearn
    // and asserting ferrolearn returns Ok where sklearn returns, Err where
    // sklearn raises. Each test names the oracle call it mirrors. The test fns
    // return `Result<(), FerroError>` so accepted cases propagate via `?`
    // (no unwrap/expect escape hatch).

    use ndarray::{Array1, array};

    // --- assert_all_finite ---
    // Oracle:
    //   assert_all_finite([1,2,3])   -> OK
    //   assert_all_finite([1, nan])  -> RAISE
    //   assert_all_finite([1, inf])  -> RAISE
    //   assert_all_finite([1, -inf]) -> RAISE
    #[test]
    fn test_assert_all_finite_oracle_parity() -> Result<(), FerroError> {
        assert_all_finite(array![1.0_f64, 2.0, 3.0].view())?; // sklearn: OK
        assert!(assert_all_finite(array![1.0_f64, f64::NAN].view()).is_err()); // RAISE
        assert!(assert_all_finite(array![1.0_f64, f64::INFINITY].view()).is_err()); // RAISE
        assert!(assert_all_finite(array![1.0_f64, f64::NEG_INFINITY].view()).is_err()); // RAISE
        Ok(())
    }

    #[test]
    fn test_assert_all_finite_messages_mirror_sklearn() {
        // sklearn message text (`validation.py:149,152,154`): "Input contains
        // NaN" vs "Input ... contains infinity or a value too large ...".
        let nan_msg = matches!(
            assert_all_finite(array![f64::NAN].view()),
            Err(FerroError::NumericalInstability { message }) if message.contains("NaN")
        );
        assert!(nan_msg, "NaN error must mention NaN");
        let inf_msg = matches!(
            assert_all_finite(array![f64::INFINITY].view()),
            Err(FerroError::NumericalInstability { message }) if message.contains("infinity")
        );
        assert!(inf_msg, "inf error must mention infinity");
    }

    #[test]
    fn test_assert_all_finite_f32_parity() -> Result<(), FerroError> {
        assert!(assert_all_finite(array![1.0_f32, f32::NAN].view()).is_err());
        assert_all_finite(array![1.0_f32, 2.0].view())?;
        Ok(())
    }

    // --- check_consistent_length ---
    // Oracle:
    //   check_consistent_length(zeros((3,2)), zeros(3)) -> OK
    //   check_consistent_length(zeros((3,2)), zeros(4)) -> RAISE
    #[test]
    fn test_check_consistent_length_oracle_parity() -> Result<(), FerroError> {
        check_consistent_length(3, 3)?; // sklearn: OK
        assert!(check_consistent_length(3, 4).is_err()); // sklearn: RAISE
        check_consistent_length(0, 0)?;
        Ok(())
    }

    #[test]
    fn test_check_consistent_length_message() {
        let ok = matches!(
            check_consistent_length(3, 4),
            Err(FerroError::ShapeMismatch { context, .. })
                if context.contains("inconsistent numbers of samples")
        );
        assert!(ok, "message must echo sklearn wording");
    }

    // --- column_or_1d ---
    // Oracle:
    //   column_or_1d(zeros(5))     -> OK shape (5,)
    //   column_or_1d(zeros((5,1))) -> OK shape (5,)
    //   column_or_1d(zeros((5,3))) -> RAISE
    //   column_or_1d(zeros((1,5))) -> RAISE   (row vector, not column)
    #[test]
    fn test_column_or_1d_oracle_parity() -> Result<(), FerroError> {
        // (5, 1) column vector -> Ok raveled to (5,)
        let col: Array2<f64> = Array2::zeros((5, 1));
        let raveled = column_or_1d(col.view())?;
        assert_eq!(raveled.len(), 5); // sklearn: OK shape (5,)

        // (5, 3) general 2-D -> Err
        let two_d: Array2<f64> = Array2::zeros((5, 3));
        assert!(column_or_1d(two_d.view()).is_err()); // sklearn: RAISE

        // (1, 5) row vector -> Err (sklearn only accepts (n,1) columns)
        let row: Array2<f64> = Array2::zeros((1, 5));
        assert!(column_or_1d(row.view()).is_err()); // sklearn: RAISE

        // already-1-D (5,) -> Ok unchanged
        let one_d: Array1<f64> = Array1::zeros(5);
        let out = column_or_1d_view(one_d.view())?;
        assert_eq!(out.len(), 5); // sklearn: OK shape (5,)
        Ok(())
    }

    #[test]
    fn test_column_or_1d_ravel_preserves_values() -> Result<(), FerroError> {
        let col = array![[1.0_f64], [2.0], [3.0]];
        let raveled = column_or_1d(col.view())?;
        assert_eq!(raveled, array![1.0, 2.0, 3.0]);
        Ok(())
    }

    // --- check_array ---
    // Oracle (defaults: finite, ensure_min_samples=1, ensure_min_features=1):
    //   check_array(zeros((3,2))) -> OK
    //   check_array(zeros((0,3))) -> RAISE  (0 samples)
    //   check_array(zeros((3,0))) -> RAISE  (0 features)
    //   check_array(zeros((1,5))) -> OK     (1 sample is enough)
    //   check_array(nan in (3,2)) -> RAISE
    //   check_array(inf in (3,2)) -> RAISE
    #[test]
    fn test_check_array_default_oracle_parity() -> Result<(), FerroError> {
        check_array_default(Array2::<f64>::zeros((3, 2)).view())?; // sklearn: OK
        assert!(check_array_default(Array2::<f64>::zeros((0, 3)).view()).is_err()); // RAISE
        assert!(check_array_default(Array2::<f64>::zeros((3, 0)).view()).is_err()); // RAISE
        check_array_default(Array2::<f64>::zeros((1, 5)).view())?; // sklearn: OK

        let mut with_nan: Array2<f64> = Array2::zeros((3, 2));
        with_nan[[0, 0]] = f64::NAN;
        assert!(check_array_default(with_nan.view()).is_err()); // RAISE

        let mut with_inf: Array2<f64> = Array2::zeros((3, 2));
        with_inf[[0, 0]] = f64::INFINITY;
        assert!(check_array_default(with_inf.view()).is_err()); // RAISE
        Ok(())
    }

    #[test]
    fn test_check_array_min_samples_error_kind() {
        let zero_samples: Array2<f64> = Array2::zeros((0, 3));
        assert!(matches!(
            check_array_default(zero_samples.view()),
            Err(FerroError::InsufficientSamples { actual: 0, .. })
        ));
    }

    #[test]
    fn test_check_array_force_all_finite_false_allows_nan() -> Result<(), FerroError> {
        // With force_all_finite=false, sklearn's check_array does not call
        // assert_all_finite (`validation.py:1063`), so a NaN passes the finite
        // gate (still subject to min-samples/min-features).
        let mut with_nan: Array2<f64> = Array2::zeros((3, 2));
        with_nan[[0, 0]] = f64::NAN;
        check_array(with_nan.view(), false, 1, 1)?;
        Ok(())
    }

    // --- check_x_y ---
    // Oracle (defaults):
    //   check_X_y(zeros((3,2)), zeros(3))   -> OK
    //   check_X_y(nan-X, zeros(3))          -> RAISE
    //   check_X_y(zeros((0,2)), zeros(0))   -> RAISE  (0 samples)
    //   check_X_y(zeros((3,0)), zeros(3))   -> RAISE  (0 features)
    //   check_X_y(zeros((3,2)), zeros(4))   -> RAISE  (len mismatch)
    #[test]
    fn test_check_x_y_good_case() -> Result<(), FerroError> {
        let x: Array2<f64> = Array2::zeros((3, 2));
        let y: Array1<f64> = Array1::zeros(3);
        let (xo, yo) = check_x_y(x.view(), y.view())?;
        assert_eq!(xo.dim(), (3, 2));
        assert_eq!(yo.len(), 3);
        Ok(())
    }

    #[test]
    fn test_check_x_y_failure_modes_oracle_parity() {
        // (a) non-finite X -> Err
        let mut nan_x: Array2<f64> = Array2::zeros((3, 2));
        nan_x[[1, 0]] = f64::NAN;
        let y3: Array1<f64> = Array1::zeros(3);
        assert!(check_x_y(nan_x.view(), y3.view()).is_err()); // sklearn: RAISE

        // (b) 0 samples -> Err
        let x0s: Array2<f64> = Array2::zeros((0, 2));
        let y0: Array1<f64> = Array1::zeros(0);
        assert!(check_x_y(x0s.view(), y0.view()).is_err()); // sklearn: RAISE

        // (b') 0 features -> Err
        let x0f: Array2<f64> = Array2::zeros((3, 0));
        assert!(check_x_y(x0f.view(), y3.view()).is_err()); // sklearn: RAISE

        // (d) length mismatch -> Err
        let x32: Array2<f64> = Array2::zeros((3, 2));
        let y4: Array1<f64> = Array1::zeros(4);
        assert!(check_x_y(x32.view(), y4.view()).is_err()); // sklearn: RAISE
    }

    #[test]
    fn test_check_x_y_length_mismatch_is_shape_mismatch() {
        let x: Array2<f64> = Array2::zeros((3, 2));
        let y: Array1<f64> = Array1::zeros(4);
        assert!(matches!(
            check_x_y(x.view(), y.view()),
            Err(FerroError::ShapeMismatch { .. })
        ));
    }

    #[test]
    fn test_check_x_y_nonfinite_y() {
        // sklearn's _check_y calls _assert_all_finite on y
        // (`validation.py:1340`); a NaN target raises.
        let x: Array2<f64> = Array2::zeros((3, 2));
        let mut y: Array1<f64> = Array1::zeros(3);
        y[1] = f64::NAN;
        assert!(check_x_y(x.view(), y.view()).is_err());
    }
}
