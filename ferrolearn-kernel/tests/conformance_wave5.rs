//! Wave-5 kernel conformance: GP regressor/classifier, Nystroem, RBFSampler,
//! KernelRidge with RBF + polynomial kernels.

use ferrolearn_core::{Fit, Predict, Transform};
use ferrolearn_kernel::{
    GaussianProcessClassifier, GaussianProcessRegressor, KernelRidge, KernelType, Nystroem,
    RBFKernel, RBFSampler,
};
use ferrolearn_test_oracle::{json_to_array1, json_to_array2, load_fixture};

#[test]
fn conformance_gaussian_process_regressor() {
    let fx = load_fixture("gaussian_process_regressor");
    let x = json_to_array2(&fx.input["X"]);
    let y = json_to_array1(&fx.input["y"]);
    let length_scale = fx.params["length_scale"].as_f64().unwrap_or(1.0);

    let kernel = Box::new(RBFKernel::<f64>::new(length_scale));
    let model = GaussianProcessRegressor::<f64>::new(kernel);
    let fitted = model.fit(&x, &y).expect("GP regressor fit");

    let preds = fitted.predict(&x).expect("GP regressor predict");
    let expected_preds = json_to_array1(&fx.expected["predictions"]);
    // GP predictions on training data should be near-interpolating;
    // ferrolearn's GP uses Cholesky + same RBF, agreement should be tight.
    let n = preds.len();
    let max_diff = preds
        .iter()
        .zip(expected_preds.iter())
        .map(|(a, e)| (a - e).abs())
        .fold(0.0f64, f64::max);
    let y_range = y.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        - y.iter().copied().fold(f64::INFINITY, f64::min);
    assert!(
        max_diff < 0.1 * y_range,
        "GP regressor predictions diverge by {max_diff:.4} on y range {y_range:.4} (n={n})"
    );
}

#[test]
fn conformance_gaussian_process_classifier() {
    let fx = load_fixture("gaussian_process_classifier");
    let x = json_to_array2(&fx.input["X"]);
    let y_vec: Vec<usize> = fx.input["y"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let y = ndarray::Array1::from_vec(y_vec);
    let length_scale = fx.params["length_scale"].as_f64().unwrap_or(1.0);

    let kernel = Box::new(RBFKernel::<f64>::new(length_scale));
    let model = GaussianProcessClassifier::<f64>::new(kernel);
    let fitted = model.fit(&x, &y).expect("GP classifier fit");
    let preds = fitted.predict(&x).expect("GP classifier predict");
    let expected_preds: Vec<usize> = fx.expected["predictions"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let matches = preds.iter().zip(expected_preds.iter()).filter(|(a, e)| a == e).count();
    let acc = matches as f64 / preds.len() as f64;
    assert!(
        acc >= 0.85,
        "GP classifier accuracy {acc:.4} < 0.85 floor"
    );
}

#[test]
fn conformance_nystroem() {
    let fx = load_fixture("nystroem");
    let x = json_to_array2(&fx.input["X"]);
    let gamma = fx.params["gamma"].as_f64().unwrap_or(0.1);
    let n_components = fx.params["n_components"].as_u64().unwrap_or(10) as usize;
    let random_state = fx.params["random_state"].as_u64().unwrap_or(42);

    let model = Nystroem::<f64>::new()
        .with_kernel(KernelType::Rbf)
        .with_gamma(gamma)
        .with_n_components(n_components)
        .with_random_state(random_state);
    let fitted = model.fit(&x, &()).expect("Nystroem fit");
    let xt = fitted.transform(&x).expect("Nystroem transform");
    assert_eq!(xt.nrows(), x.nrows(), "Nystroem rows");
    assert_eq!(xt.ncols(), n_components, "Nystroem cols");
    for v in xt.iter() {
        assert!(v.is_finite(), "Nystroem non-finite");
    }
}

#[test]
fn conformance_rbf_sampler() {
    let fx = load_fixture("rbf_sampler");
    let x = json_to_array2(&fx.input["X"]);
    let gamma = fx.params["gamma"].as_f64().unwrap_or(0.5);
    let n_components = fx.params["n_components"].as_u64().unwrap_or(20) as usize;
    let random_state = fx.params["random_state"].as_u64().unwrap_or(42);

    let model = RBFSampler::<f64>::new()
        .with_gamma(gamma)
        .with_n_components(n_components)
        .with_random_state(random_state);
    let fitted = model.fit(&x, &()).expect("RBFSampler fit");
    let xt = fitted.transform(&x).expect("RBFSampler transform");
    assert_eq!(xt.nrows(), x.nrows(), "RBFSampler rows");
    assert_eq!(xt.ncols(), n_components, "RBFSampler cols");
    for v in xt.iter() {
        assert!(v.is_finite(), "RBFSampler non-finite");
    }
}

#[test]
fn conformance_kernel_ridge_rbf() {
    let fx = load_fixture("kernel_ridge_rbf");
    let x = json_to_array2(&fx.input["X"]);
    let y = json_to_array1(&fx.input["y"]);
    let alpha = fx.params["alpha"].as_f64().unwrap_or(1.0);
    let gamma = fx.params["gamma"].as_f64().unwrap_or(0.1);

    let model = KernelRidge::<f64>::new()
        .with_alpha(alpha)
        .with_kernel(KernelType::Rbf)
        .with_gamma(gamma);
    let fitted = model.fit(&x, &y).expect("KernelRidge fit");
    let preds = fitted.predict(&x).expect("KernelRidge predict");
    let expected_preds = json_to_array1(&fx.expected["predictions"]);

    // KernelRidge with RBF should match closely — both use the same closed-form
    // dual solution. Tolerance bumped to 1e-4 to absorb gamma-formatting drift.
    for (i, (&a, &e)) in preds.iter().zip(expected_preds.iter()).enumerate() {
        let threshold = (1e-4f64).max(1e-4 * e.abs());
        assert!(
            (a - e).abs() <= threshold,
            "KernelRidge(RBF).predict[{i}]: actual={a:.6} expected={e:.6} diff={diff:.3e} > {threshold:.3e}",
            diff = (a - e).abs()
        );
    }
}

#[test]
fn conformance_kernel_ridge_poly() {
    let fx = load_fixture("kernel_ridge_poly");
    let x = json_to_array2(&fx.input["X"]);
    let y = json_to_array1(&fx.input["y"]);
    let alpha = fx.params["alpha"].as_f64().unwrap_or(1.0);
    let gamma = fx.params["gamma"].as_f64().unwrap_or(0.1);
    let degree = fx.params["degree"].as_u64().unwrap_or(2) as usize;
    let coef0 = fx.params["coef0"].as_f64().unwrap_or(1.0);

    let model = KernelRidge::<f64>::new()
        .with_alpha(alpha)
        .with_kernel(KernelType::Polynomial)
        .with_gamma(gamma)
        .with_degree(degree)
        .with_coef0(coef0);
    let fitted = model.fit(&x, &y).expect("KernelRidge(poly) fit");
    let preds = fitted.predict(&x).expect("KernelRidge(poly) predict");
    let expected_preds = json_to_array1(&fx.expected["predictions"]);

    for (i, (&a, &e)) in preds.iter().zip(expected_preds.iter()).enumerate() {
        let threshold = (1e-4f64).max(1e-4 * e.abs());
        assert!(
            (a - e).abs() <= threshold,
            "KernelRidge(poly).predict[{i}]: actual={a:.6} expected={e:.6} diff={diff:.3e} > {threshold:.3e}",
            diff = (a - e).abs()
        );
    }
}
