//! Oracle pins for `ferrolearn_preprocess::r_regression` against
//! `sklearn.feature_selection.r_regression`
//! (`sklearn/feature_selection/_univariate_selection.py:301-393`).

use approx::assert_relative_eq;
use ferrolearn_preprocess::{r_regression, r_regression_with_options};
use ndarray::{Array1, Array2, array};

fn fixture_x() -> Array2<f64> {
    array![
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 1.0],
        [2.0, 1.0, 9.0],
        [5.0, 3.0, 2.0],
        [8.0, 9.0, 4.0]
    ]
}

fn fixture_y() -> Array1<f64> {
    array![1.5, 2.0, 3.5, 4.0, 5.5, 6.0]
}

#[test]
fn r_regression_centered_default_matches_sklearn() {
    let corr = r_regression(&fixture_x(), &fixture_y()).unwrap();
    let expected = [
        0.655_763_253_996_913_7,
        0.355_303_332_401_016_47,
        -0.103_836_753_919_982_34,
    ];
    for (actual, expected) in corr.iter().zip(expected) {
        assert_relative_eq!(*actual, expected, epsilon = 1e-15);
    }
}

#[test]
fn r_regression_uncentered_matches_sklearn() {
    let corr = r_regression_with_options(&fixture_x(), &fixture_y(), false, true).unwrap();
    let expected = [
        0.928_360_763_528_088_2,
        0.848_302_106_257_209_8,
        0.747_756_501_105_966,
    ];
    for (actual, expected) in corr.iter().zip(expected) {
        assert_relative_eq!(*actual, expected, epsilon = 1e-15);
    }
}

#[test]
fn r_regression_force_finite_constant_feature_and_target() {
    let x_const_feature = array![[2.0_f64, 1.0], [2.0, 0.0], [2.0, 10.0], [2.0, 4.0]];
    let y = array![0.0_f64, 1.0, 1.0, 0.0];
    let forced = r_regression_with_options(&x_const_feature, &y, true, true).unwrap();
    assert_relative_eq!(forced[0], 0.0, epsilon = 0.0);
    assert_relative_eq!(forced[1], 0.320_750_149_549_792_1, epsilon = 1e-15);

    let unforced = r_regression_with_options(&x_const_feature, &y, true, false).unwrap();
    assert!(unforced[0].is_nan());
    assert_relative_eq!(unforced[1], 0.320_750_149_549_792_1, epsilon = 1e-15);

    let x = array![[5.0_f64, 1.0], [3.0, 0.0], [2.0, 10.0], [8.0, 4.0]];
    let y_const = array![0.0_f64, 0.0, 0.0, 0.0];
    let forced = r_regression_with_options(&x, &y_const, true, true).unwrap();
    assert_eq!(forced.to_vec(), vec![0.0, 0.0]);
    let unforced = r_regression_with_options(&x, &y_const, true, false).unwrap();
    assert!(unforced.iter().all(|value| (*value).is_nan()));
}

#[test]
fn r_regression_validation_contracts() {
    let x = fixture_x();
    let y_short = array![1.0, 2.0];
    assert!(r_regression(&x, &y_short).is_err());

    let x_empty = Array2::<f64>::zeros((0, 2));
    let y_empty = Array1::<f64>::zeros(0);
    assert!(r_regression(&x_empty, &y_empty).is_err());

    let x_zero_features = Array2::<f64>::zeros((2, 0));
    let y = array![1.0, 2.0];
    assert!(r_regression(&x_zero_features, &y).is_err());

    let x_nan = array![[1.0, f64::NAN], [2.0, 3.0]];
    assert!(r_regression(&x_nan, &y).is_err());

    let y_nan = array![1.0, f64::NAN];
    let x = array![[1.0, 2.0], [2.0, 3.0]];
    assert!(r_regression(&x, &y_nan).is_err());
}
