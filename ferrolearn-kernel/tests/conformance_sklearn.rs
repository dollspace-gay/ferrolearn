//! Conformance tests for ferrolearn-kernel vs scikit-learn.
//!
//! Currently covers `KernelRidge` (linear kernel). RBF / polynomial kernels
//! and GaussianProcess estimators will be added once fixtures are generated.

use ferrolearn_core::{Fit, Predict};
use ferrolearn_kernel::{KernelRidge, KernelType};
use ferrolearn_test_oracle::{
    TOL_KERNEL_ABS, TOL_KERNEL_REL, assert_close_slice, json_to_array1, json_to_array2,
    load_fixture,
};

#[test]
fn conformance_kernel_ridge_linear() {
    let fx = load_fixture("kernel_ridge");
    let x = json_to_array2(&fx.input["X"]);
    let y = json_to_array1(&fx.input["y"]);
    let (rel, abs) = fx.tolerance(TOL_KERNEL_REL, TOL_KERNEL_ABS);

    let alpha = fx.params["alpha"].as_f64().unwrap();
    let kernel = match fx.params["kernel"].as_str().unwrap() {
        "linear" => KernelType::Linear,
        "rbf" => KernelType::Rbf,
        "polynomial" => KernelType::Polynomial,
        other => panic!("unsupported kernel: {other}"),
    };
    let model = KernelRidge::<f64>::new()
        .with_alpha(alpha)
        .with_kernel(kernel);
    let fitted = model.fit(&x, &y).expect("KernelRidge fit");

    let preds = fitted.predict(&x).expect("KernelRidge predict");
    let expected_preds = json_to_array1(&fx.expected["predictions"]);
    assert_close_slice(
        preds.as_slice().unwrap(),
        expected_preds.as_slice().unwrap(),
        rel,
        abs,
        "KernelRidge.predict",
    );

    let expected_dual = json_to_array1(&fx.expected["dual_coef"]);
    assert_close_slice(
        fitted.dual_coef().as_slice().unwrap(),
        expected_dual.as_slice().unwrap(),
        rel,
        abs,
        "KernelRidge.dual_coef",
    );
}
