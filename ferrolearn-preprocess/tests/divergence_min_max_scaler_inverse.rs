//! Divergence test: ferrolearn `FittedMinMaxScaler::inverse_transform` was NOT
//! updated by the #2200 (inf-reject) / #2201 (constant-col affine) fixes, which
//! only touched `fit` and `transform`. Two divergences from scikit-learn 1.5.2
//! survive in `inverse_transform`.
//!
//! sklearn `MinMaxScaler.inverse_transform` (`sklearn/preprocessing/_data.py:574-575`):
//!
//! ```text
//! X -= self.min_
//! X /= self.scale_
//! ```
//!
//! validated with `force_all_finite="allow-nan"` (`_data.py:571`).
//!
//! # Divergence 1 — constant-column held-out forced to `data_min`
//!
//! For a column constant at `5.0` (default range), `data_range_ = 0` is replaced
//! by `1` via `_handle_zeros_in_scale` (`_data.py:88`), giving `scale_ = 1.0`,
//! `min_ = 0 - 5*1 = -5.0` (`_data.py:508`,`:511`). So `inverse_transform([[y]])`
//! is `(y - (-5)) / 1 = y + 5`, NOT a collapse to `data_min`. Live oracle:
//!
//! ```text
//! $ python3 -c "from sklearn.preprocessing import MinMaxScaler; import numpy as np
//!   m = MinMaxScaler().fit(np.array([[5.],[5.],[5.]]))
//!   print(m.inverse_transform(np.array([[0.],[0.5],[2.]])).ravel().tolist())"
//! [5.0, 5.5, 7.0]
//! ```
//!
//! ferrolearn's `inverse_transform` (`min_max_scaler.rs:281-288`) computes
//! `(v - range_min) * span / range_width + data_min` with `span = data_max -
//! data_min`. For a constant column `span == 0`, so EVERY row collapses to
//! `data_min = 5.0`, ignoring the input value: probed output `[5.0, 5.0, 5.0]`.
//!
//! # Divergence 2 — `+/-inf` not rejected in `inverse_transform`
//!
//! sklearn's `inverse_transform` uses `force_all_finite="allow-nan"`
//! (`_data.py:571`): NaN passes, `+/-inf` raises `ValueError`. Live oracle:
//!
//! ```text
//! $ python3 -c "from sklearn.preprocessing import MinMaxScaler; import numpy as np
//!   m = MinMaxScaler().fit(np.array([[1.],[2.],[3.]]))
//!   try: m.inverse_transform(np.array([[np.inf]]))
//!   except ValueError as e: print('ValueError')"
//! ValueError
//! ```
//!
//! ferrolearn's `inverse_transform` has NO `is_infinite` guard (unlike `fit`
//! `min_max_scaler.rs:319` and `transform` `:423`): probed output for an `inf`
//! input is `Ok([inf])`, accepting input sklearn rejects.
//!
//! Expected values come from the LIVE sklearn 1.5.2 oracle (R-CHAR-3), never
//! copied from the ferrolearn side.
//!
//! Tracking: #2202

use ferrolearn_core::traits::Fit;
use ferrolearn_preprocess::MinMaxScaler;
use ndarray::array;

/// Divergence 1: `inverse_transform` of a fit-constant column with held-out
/// scaled data. sklearn applies the affine inverse `(y - min_)/scale_` with
/// `scale_=1`, `min_=-5` (`_data.py:574-575`,`:508`,`:511`,`:88`), giving
/// `[5.0, 5.5, 7.0]`. ferrolearn forces every row to `data_min=5.0` because its
/// `span==0` formula (`min_max_scaler.rs:284-287`) multiplies the input away.
/// Live oracle:
///   `MinMaxScaler().fit([[5.],[5.],[5.]]).inverse_transform([[0.],[0.5],[2.]])`
///   -> `[5.0, 5.5, 7.0]`.
/// Tracking: #2202
#[test]
#[ignore = "divergence: inverse_transform forces fit-constant col to data_min for held-out data (sklearn uses affine scale_=1,min_=-5); tracking #2202"]
fn divergence_inverse_constant_column_holdout_uses_affine() {
    // Live sklearn 1.5.2 oracle, never copied from ferrolearn.
    let sk = [5.0_f64, 5.5, 7.0];

    let train = array![[5.0_f64], [5.0], [5.0]];
    let fitted = MinMaxScaler::<f64>::new().fit(&train, &()).unwrap();

    let scaled = array![[0.0_f64], [0.5], [2.0]];
    let inv = fitted.inverse_transform(&scaled).unwrap();
    for i in 0..3 {
        assert!(
            (inv[[i, 0]] - sk[i]).abs() < 1e-12,
            "inverse held-out row {i}: sklearn {} ferrolearn {} \
             (constant-fit col must use affine (y-min_)/scale_ with scale_=1, min_=-5, \
             not collapse to data_min)",
            sk[i],
            inv[[i, 0]]
        );
    }
}

/// Divergence 1b (custom range `(2,5)`): a constant col gives `scale_=3`,
/// `min_=-13`, so `inverse_transform([[2.],[3.5],[8.]]) = (y+13)/3` ->
/// `[5.0, 5.5, 7.0]`. Live oracle:
///   `MinMaxScaler(feature_range=(2,5)).fit([[5.],[5.],[5.]])
///    .inverse_transform([[2.],[3.5],[8.]])` -> `[5.0, 5.5, 7.0]`.
/// Tracking: #2202
#[test]
#[ignore = "divergence: inverse_transform constant col custom-range held-out collapses to data_min; tracking #2202"]
fn divergence_inverse_constant_column_custom_range_holdout() {
    // Live sklearn 1.5.2 oracle.
    let sk = [5.0_f64, 5.5, 7.0];

    let train = array![[5.0_f64], [5.0], [5.0]];
    let fitted = MinMaxScaler::<f64>::with_feature_range(2.0, 5.0)
        .unwrap()
        .fit(&train, &())
        .unwrap();

    let scaled = array![[2.0_f64], [3.5], [8.0]];
    let inv = fitted.inverse_transform(&scaled).unwrap();
    for i in 0..3 {
        assert!(
            (inv[[i, 0]] - sk[i]).abs() < 1e-12,
            "inverse custom-range held-out row {i}: sklearn {} ferrolearn {}",
            sk[i],
            inv[[i, 0]]
        );
    }
}

/// Divergence 2: `inverse_transform` accepts `+inf` where sklearn
/// `_data.py:571` (`force_all_finite="allow-nan"`) raises `ValueError`.
/// Live oracle: `MinMaxScaler().fit([[1.],[2.],[3.]]).inverse_transform([[inf]])`
///   -> `ValueError`. ferrolearn returns `Ok([inf])` (no inf-check, unlike its
/// own `fit`/`transform`). A faithful translation must return `Err`.
/// Tracking: #2202
#[test]
#[ignore = "divergence: inverse_transform accepts +/-inf where sklearn rejects (force_all_finite=allow-nan); tracking #2202"]
fn divergence_inverse_rejects_inf() {
    let train = array![[1.0_f64], [2.0], [3.0]];
    let fitted = MinMaxScaler::<f64>::new().fit(&train, &()).unwrap();

    let scaled = array![[f64::INFINITY]];
    let result = fitted.inverse_transform(&scaled);
    assert!(
        result.is_err(),
        "sklearn raises ValueError on +inf in inverse_transform \
         (force_all_finite='allow-nan', _data.py:571); ferrolearn returned {:?}",
        result.map(|a| a.iter().copied().collect::<Vec<_>>()).ok()
    );
}
