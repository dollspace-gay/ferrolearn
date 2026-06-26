//! Dummy-feature helper: prepend a constant feature column.
//!
//! This mirrors the dense-array path of scikit-learn's
//! `sklearn.preprocessing.add_dummy_feature` (`_data.py:2594`): validate a
//! finite 2D input and return `np.hstack((full((n_samples, 1), value), X))`.
//!
//! ## REQ status
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (dense value/shape parity) | SHIPPED | `add_dummy_feature` returns an owned `Array2` with shape `(n_samples, n_features + 1)`, first column filled with `value`, and original features shifted right. Live-oracle tests in `tests/divergence_add_dummy_feature.rs`: default value, custom value, and input-preservation guards. |
//! | REQ-2 (check_array-style validation) | SHIPPED | zero rows -> `InsufficientSamples`, zero features/non-finite X -> `InvalidParameter`, matching sklearn `check_array(..., dtype=FLOAT_DTYPES)` defaults (`_data.py:2623`). |
//! | REQ-3 (sparse CSR/CSC/COO path) | NOT-STARTED | sklearn preserves sparse format and prepends sparse dummy values (`_data.py:2626-2648`); ferrolearn exposes dense `Array2` only here. |

use ferrolearn_core::error::FerroError;
use ndarray::{Array2, s};
use num_traits::Float;

/// Prepend a constant dummy feature column to a dense feature matrix.
///
/// The returned matrix has one additional first column filled with `value`;
/// the original columns are copied to columns `1..`.
///
/// # Errors
///
/// Returns [`FerroError::InsufficientSamples`] for zero rows and
/// [`FerroError::InvalidParameter`] for zero features or any non-finite input
/// element. The dummy `value` itself is not validated, matching sklearn's
/// parameter behavior after input validation.
#[must_use = "add_dummy_feature returns a new array; use the returned value"]
pub fn add_dummy_feature<F: Float>(x: &Array2<F>, value: F) -> Result<Array2<F>, FerroError> {
    let n_samples = x.nrows();
    if n_samples == 0 {
        return Err(FerroError::InsufficientSamples {
            required: 1,
            actual: 0,
            context: "add_dummy_feature".into(),
        });
    }

    let n_features = x.ncols();
    if n_features == 0 {
        return Err(FerroError::InvalidParameter {
            name: "X".into(),
            reason: "Found array with 0 feature(s); a minimum of 1 is required.".into(),
        });
    }

    if x.iter().any(|v| !v.is_finite()) {
        return Err(FerroError::InvalidParameter {
            name: "X".into(),
            reason: "Input X contains NaN or infinity.".into(),
        });
    }

    let mut out = Array2::from_elem((n_samples, n_features + 1), value);
    out.slice_mut(s![.., 1..]).assign(x);
    Ok(out)
}
