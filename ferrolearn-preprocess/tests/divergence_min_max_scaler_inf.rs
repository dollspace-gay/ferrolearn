//! Divergence test: ferrolearn `MinMaxScaler::fit` accepts non-finite (±inf)
//! input where scikit-learn 1.5.2 REJECTS it.
//!
//! sklearn `MinMaxScaler.partial_fit` validates with
//! `force_all_finite="allow-nan"` (`sklearn/preprocessing/_data.py:494`):
//!
//! ```text
//! X = self._validate_data(
//!     X, ..., force_all_finite="allow-nan",
//! )
//! ```
//!
//! `allow-nan` permits NaN but STILL rejects ±inf (`check_array`,
//! `sklearn/utils/validation.py`): the live 1.5.2 oracle raises
//! `ValueError: Input X contains infinity or a value too large for
//! dtype('float64').` on a `+inf`/`-inf` entry in `fit`.
//!
//! ferrolearn's `Fit::fit` (`min_max_scaler.rs:330-347`) only skips `NaN`
//! (`if v.is_nan() { continue; }`) and otherwise keeps the value in the
//! min/max reduction. `+inf` is therefore retained as `data_max` and `fit`
//! returns `Ok(..)` — accepting input sklearn rejects.
//!
//! Expected value from the LIVE sklearn 1.5.2 oracle (R-CHAR-3), never copied
//! from the ferrolearn side:
//!
//! ```text
//! $ python3 -c "from sklearn.preprocessing import MinMaxScaler
//!   try:
//!       MinMaxScaler().fit([[1.0],[float('inf')],[3.0]])
//!   except Exception as e:
//!       print(type(e).__name__)"
//! ValueError
//! ```
//!
//! sklearn raises (returns no model); ferrolearn returns `Ok`. This test pins
//! that `fit` on inf-containing input must be an `Err` (any input-validation
//! error variant, mirroring the Binarizer non-finite precedent
//! `binarizer.rs:173`). It FAILS today because ferrolearn returns `Ok`.
//!
//! Tracking: #2200

use ferrolearn_core::traits::Fit;
use ferrolearn_preprocess::MinMaxScaler;
use ndarray::array;

/// Divergence: `MinMaxScaler::fit` accepts `+inf` where sklearn
/// `_data.py:494` (`force_all_finite="allow-nan"`) raises `ValueError`.
/// Live oracle: `MinMaxScaler().fit([[1.],[inf],[3.]])` -> `ValueError`.
/// ferrolearn: returns `Ok` (only NaN is skipped; inf is kept as data_max).
/// Tracking: #2200
#[test]
fn divergence_fit_rejects_positive_inf() {
    // Live sklearn 1.5.2 oracle: this input raises ValueError, so a faithful
    // translation must return Err (no fitted model). Asserting `is_err()` is
    // the parity contract; the value comes from the oracle, not ferrolearn.
    let scaler = MinMaxScaler::<f64>::new();
    let x = array![[1.0], [f64::INFINITY], [3.0]];
    let result = scaler.fit(&x, &());
    assert!(
        result.is_err(),
        "sklearn raises ValueError on +inf in fit (force_all_finite='allow-nan', \
         _data.py:494); ferrolearn returned Ok with data_max={:?}",
        result.map(|f| f.data_max().clone()).ok()
    );
}

/// Divergence: `MinMaxScaler::fit` accepts `-inf` where sklearn raises.
/// Live oracle: `MinMaxScaler().fit([[1.],[-inf],[3.]])` -> `ValueError`.
/// Tracking: #2200
#[test]
fn divergence_fit_rejects_negative_inf() {
    let scaler = MinMaxScaler::<f64>::new();
    let x = array![[1.0], [f64::NEG_INFINITY], [3.0]];
    let result = scaler.fit(&x, &());
    assert!(
        result.is_err(),
        "sklearn raises ValueError on -inf in fit (force_all_finite='allow-nan', \
         _data.py:494); ferrolearn returned Ok with data_min={:?}",
        result.map(|f| f.data_min().clone()).ok()
    );
}
