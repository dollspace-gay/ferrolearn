//! Divergence test: ferrolearn `MinMaxScaler::transform` forces EVERY row of a
//! column that was constant at fit time to `feature_range[0]`, but scikit-learn
//! 1.5.2 applies the learned affine map `X *= scale_; X += min_` to held-out
//! data — so a held-out value DIFFERENT from the fitted constant does NOT map
//! to `feature_range[0]`.
//!
//! sklearn `MinMaxScaler.transform` (`sklearn/preprocessing/_data.py:543-544`):
//!
//! ```text
//! X *= self.scale_
//! X += self.min_
//! ```
//!
//! For a column that was constant at `5.0` (default range), `data_range_ = 0`
//! is replaced by `1` via `_handle_zeros_in_scale` (`_data.py:88`), giving
//! `scale_ = (1-0)/1 = 1.0` and `min_ = 0 - 5*1 = -5.0` (`_data.py:508`,`:511`).
//! Transforming a held-out `9.0` therefore gives `9*1 + (-5) = 4.0`, NOT
//! `feature_range[0] = 0.0`. The "constant column maps to feature_range[0]"
//! identity only holds for the fitted constant value itself (`5 -> 0`).
//!
//! ferrolearn's `Transform::transform` (`min_max_scaler.rs:433-441`) takes a
//! `if span == F::zero()` branch that overwrites EVERY element of the column
//! with `range_min`, ignoring the input value — so held-out `9.0 -> 0.0`.
//!
//! Expected values from the LIVE sklearn 1.5.2 oracle (R-CHAR-3), never copied
//! from ferrolearn:
//!
//! ```text
//! $ python3 -c "from sklearn.preprocessing import MinMaxScaler
//!   m = MinMaxScaler().fit([[5.,1.],[5.,2.],[5.,3.]])
//!   print(m.transform([[4.,2.],[9.,2.]]).tolist())"
//! [[-1.0, 0.5], [4.0, 0.5]]        # col 0 held-out 4 -> -1.0, 9 -> 4.0
//! ```
//!
//! With `clip=True` the held-out `9.0` (affine -> 4.0) clamps to `1.0`, and
//! held-out `4.0` (affine -> -1.0) clamps to `0.0`:
//!
//! ```text
//! $ python3 -c "from sklearn.preprocessing import MinMaxScaler
//!   m = MinMaxScaler(clip=True).fit([[5.,1.],[5.,2.],[5.,3.]])
//!   print(m.transform([[9.,2.]]).tolist())"
//! [[1.0, 0.5]]
//! ```
//!
//! Tracking: #2201

use ferrolearn_core::traits::Fit;
use ferrolearn_core::traits::Transform;
use ferrolearn_preprocess::MinMaxScaler;
use ndarray::array;

/// Divergence: held-out values on a fit-constant column. sklearn applies the
/// affine map (`scale_=1`, `min_=-5`): `4 -> -1.0`, `9 -> 4.0`
/// (`_data.py:543-544`,`:508`,`:511`,`:88`). ferrolearn forces the whole
/// column to `range_min=0.0` (`min_max_scaler.rs:433-441`).
/// Live oracle: `MinMaxScaler().fit([[5.,1.],[5.,2.],[5.,3.]]).transform([[4.,2.],[9.,2.]])`
///   -> `[[-1.0, 0.5], [4.0, 0.5]]`.
/// Tracking: #2201
#[test]
#[ignore = "divergence: transform forces constant col to range_min, sklearn applies affine to held-out; tracking #2201"]
fn divergence_constant_column_holdout_uses_affine() {
    // Live sklearn 1.5.2 oracle for col 0 of the held-out rows.
    let sk_col0 = [-1.0_f64, 4.0];

    let train = array![[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]];
    let fitted = MinMaxScaler::<f64>::new().fit(&train, &()).unwrap();

    let holdout = array![[4.0, 2.0], [9.0, 2.0]];
    let out = fitted.transform(&holdout).unwrap();
    for i in 0..2 {
        assert!(
            (out[[i, 0]] - sk_col0[i]).abs() < 1e-12,
            "held-out row {i} col0: sklearn {} ferrolearn {} \
             (constant-fit col must use affine scale_=1, min_=-5, not range_min)",
            sk_col0[i],
            out[[i, 0]]
        );
    }
}

/// Divergence + clip: held-out `9.0` on a fit-constant column maps (affine) to
/// `4.0`, then clip=True clamps to `feature_range[1]=1.0`. ferrolearn forces it
/// to `range_min=0.0` before any clip.
/// Live oracle: `MinMaxScaler(clip=True).fit([[5.,1.],[5.,2.],[5.,3.]]).transform([[9.,2.]])`
///   -> `[[1.0, 0.5]]`.
/// Tracking: #2201
#[test]
#[ignore = "divergence: constant-col held-out + clip; tracking #2201"]
fn divergence_constant_column_holdout_with_clip() {
    // Live sklearn 1.5.2 oracle: affine 9 -> 4.0, clipped to 1.0.
    let sk_col0 = 1.0_f64;

    let train = array![[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]];
    let fitted = MinMaxScaler::<f64>::new()
        .with_clip(true)
        .fit(&train, &())
        .unwrap();

    let holdout = array![[9.0, 2.0]];
    let out = fitted.transform(&holdout).unwrap();
    assert!(
        (out[[0, 0]] - sk_col0).abs() < 1e-12,
        "held-out 9.0 on constant-fit col with clip: sklearn {} ferrolearn {} \
         (affine 9->4.0 then clip to 1.0)",
        sk_col0,
        out[[0, 0]]
    );
}
