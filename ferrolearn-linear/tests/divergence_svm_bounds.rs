//! Oracle pins for `ferrolearn_linear::l1_min_c` against
//! `sklearn.svm.l1_min_c` (`sklearn/svm/_bounds.py:26-99`).

use approx::assert_relative_eq;
use ferrolearn_linear::{L1MinCLoss, l1_min_c};
use ndarray::{Array1, Array2, array};

fn binary_fixture() -> (Array2<f64>, Array1<usize>) {
    let x = array![[-1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 1.0]];
    let y = array![0usize, 1, 1, 1];
    (x, y)
}

#[test]
fn l1_min_c_binary_matches_sklearn_no_intercept() {
    let (x, y) = binary_fixture();

    let squared = l1_min_c(&x, &y, L1MinCLoss::SquaredHinge, false, 1.0).unwrap();
    let log = l1_min_c(&x, &y, L1MinCLoss::Log, false, 1.0).unwrap();

    // Live sklearn 1.5.2:
    // l1_min_c(X, y, loss="squared_hinge", fit_intercept=False) -> 1/6
    // l1_min_c(X, y, loss="log", fit_intercept=False) -> 2/3
    assert_relative_eq!(squared, 0.16666666666666666, epsilon = 1e-15);
    assert_relative_eq!(log, 0.6666666666666666, epsilon = 1e-15);
}

#[test]
fn l1_min_c_binary_intercept_scaling_matches_sklearn() {
    let (x, y) = binary_fixture();

    let squared = l1_min_c(&x, &y, L1MinCLoss::SquaredHinge, true, 10.0).unwrap();
    let log = l1_min_c(&x, &y, L1MinCLoss::Log, true, 10.0).unwrap();

    // Live sklearn 1.5.2, same fixture, intercept_scaling=10:
    // squared_hinge -> 0.025, log -> 0.1
    assert_relative_eq!(squared, 0.025, epsilon = 1e-15);
    assert_relative_eq!(log, 0.1, epsilon = 1e-15);
}

#[test]
fn l1_min_c_multiclass_matches_label_binarizer_oracle() {
    let x = array![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 0.0]];
    let y = array![0usize, 1, 2, 1];

    let squared = l1_min_c(&x, &y, L1MinCLoss::SquaredHinge, false, 1.0).unwrap();
    let squared_bias = l1_min_c(&x, &y, L1MinCLoss::SquaredHinge, true, 2.0).unwrap();
    let log = l1_min_c(&x, &y, L1MinCLoss::Log, false, 1.0).unwrap();
    let log_bias = l1_min_c(&x, &y, L1MinCLoss::Log, true, 2.0).unwrap();

    // Live sklearn 1.5.2:
    // LabelBinarizer(neg_label=-1) creates one row per class for this fixture.
    assert_relative_eq!(squared, 0.25, epsilon = 1e-15);
    assert_relative_eq!(squared_bias, 0.125, epsilon = 1e-15);
    assert_relative_eq!(log, 1.0, epsilon = 1e-15);
    assert_relative_eq!(log_bias, 0.5, epsilon = 1e-15);
}

#[test]
fn l1_min_c_rejects_ill_posed_and_invalid_inputs() {
    let zero_x = Array2::zeros((2, 2));
    let y = array![0usize, 1];
    assert!(l1_min_c(&zero_x, &y, L1MinCLoss::SquaredHinge, true, 1.0).is_err());

    let (x, y) = binary_fixture();
    let short_y = array![0usize, 1, 1];
    assert!(l1_min_c(&x, &short_y, L1MinCLoss::SquaredHinge, true, 1.0).is_err());
    assert!(l1_min_c(&x, &y, L1MinCLoss::SquaredHinge, true, 0.0).is_err());

    let x_nan = array![[1.0, f64::NAN], [2.0, 3.0]];
    let y_nan = array![0usize, 1];
    assert!(l1_min_c(&x_nan, &y_nan, L1MinCLoss::SquaredHinge, true, 1.0).is_err());
}
