//! Divergence audit (REQ-5, #1160): `OrdinalEncoder` `handle_unknown='use_encoded_value'`
//! non-integer `unknown_value` validation vs scikit-learn 1.5.2.
//!
//! sklearn `sklearn/preprocessing/_encoders.py:1481-1487` (`OrdinalEncoder.fit`):
//! ```text
//!     elif not isinstance(self.unknown_value, numbers.Integral):
//!         raise TypeError(
//!             "unknown_value should be an integer or "
//!             "np.nan when "
//!             "handle_unknown is 'use_encoded_value', "
//!             f"got {self.unknown_value}."
//!         )
//! ```
//! This `isinstance(.., Integral)` guard fires BEFORE the range/collision check
//! (`:1518-1526`), so ANY non-integer, non-nan `unknown_value` (e.g. `1.5`,
//! `2.5`, `-1.5`, even out-of-range `100.5`) raises `TypeError` — regardless of
//! magnitude or whether it would collide with an encoding index.
//!
//! ferrolearn (`ferrolearn-preprocess/src/ordinal_encoder.rs:329-345`) takes an
//! `f64` `unknown_value` and only runs its collision branch when
//! `v.fract() == 0.0`; a non-integer float skips ALL validation and `fit`
//! returns `Ok`. There is no analog of sklearn's `:1481` integrality rejection,
//! so ferrolearn OVER-ACCEPTS (R-DEV-2 over-rejection's mirror: over-acceptance).
//!
//! LIVE sklearn 1.5.2 oracle (run from /tmp):
//! ```text
//!   $ python3 -c "import numpy as np; from sklearn.preprocessing import OrdinalEncoder
//!     for v in [1.5, 2.5, -1.5, 100.5]:
//!         try:
//!             OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=v)\
//!                 .fit([['cat'],['dog'],['cat']]); print(repr(v),'OK')
//!         except Exception as e: print(repr(v), type(e).__name__)"
//!   -> 1.5 TypeError
//!      2.5 TypeError      # out-of-range [0,2) and still TypeError (Integral check first)
//!      -1.5 TypeError
//!      100.5 TypeError
//! ```
//! Expected: every non-integer non-nan `unknown_value` makes `fit` return `Err`.
//! Actual (ferrolearn): `fit` returns `Ok` for all of them.
//!
//! Tracking: #2221

use ferrolearn_core::traits::Fit;
use ferrolearn_preprocess::{HandleUnknown, OrdinalEncoder};
use ndarray::Array2;

fn make_1col(vals: &[&str]) -> Array2<String> {
    Array2::from_shape_vec(
        (vals.len(), 1),
        vals.iter().map(std::string::ToString::to_string).collect(),
    )
    .unwrap()
}

/// Divergence: ferrolearn's `OrdinalEncoder::fit` diverges from
/// `sklearn/preprocessing/_encoders.py:1481` for a non-integer, non-nan
/// `unknown_value` under `handle_unknown='use_encoded_value'`.
/// sklearn raises `TypeError` ("unknown_value should be an integer or np.nan");
/// ferrolearn returns `Ok` because its only check is `v.fract() == 0.0` for the
/// range/collision branch and it has no integrality guard.
/// Tracking: #2221
#[test]
fn divergence_noninteger_unknown_value_accepted() {
    // categories_[0] = ['cat','dog'] -> max_cardinality = 2.
    let x_train = make_1col(&["cat", "dog", "cat"]);

    // Live sklearn 1.5.2 oracle: ALL of these raise TypeError in
    // OrdinalEncoder.fit (Integral check at _encoders.py:1481, BEFORE the
    // range/collision check). So ferrolearn's fit MUST be Err for each.
    for v in [1.5_f64, 2.5, -1.5, 100.5] {
        let enc = OrdinalEncoder::new()
            .with_handle_unknown(HandleUnknown::UseEncodedValue)
            .with_unknown_value(v);
        let result = enc.fit(&x_train, &());
        assert!(
            result.is_err(),
            "sklearn _encoders.py:1481 raises TypeError for non-integer \
             unknown_value={v}; ferrolearn must Err but returned Ok"
        );
    }
}
