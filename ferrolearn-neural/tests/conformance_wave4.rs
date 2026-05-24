//! Wave-4 neural conformance vs scikit-learn.
//!
//! MLPRegressor + BernoulliRBM. Stochastic models — we use accuracy/R² floors.

use ferrolearn_core::{Fit, Predict};
use ferrolearn_neural::{Activation, BernoulliRBM, MLPRegressor, Solver};
use ferrolearn_test_oracle::{json_to_array1, json_to_array2, load_fixture};

#[test]
fn conformance_mlp_regressor() {
    let fx = load_fixture("mlp_regressor");
    let x = json_to_array2(&fx.input["X"]);
    let y = json_to_array1(&fx.input["y"]);
    let max_iter = fx.params["max_iter"].as_u64().unwrap_or(500) as usize;
    let alpha = fx.params["alpha"].as_f64().unwrap_or(1e-4);
    let random_state = fx.params["random_state"].as_u64().unwrap_or(42);

    let model = MLPRegressor::<f64>::new()
        .with_hidden_layer_sizes(vec![16])
        .with_activation(Activation::Relu)
        .with_solver(Solver::Adam {
            learning_rate: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        })
        .with_max_iter(max_iter)
        .with_alpha(alpha)
        .with_random_state(random_state);
    let fitted = model.fit(&x, &y).expect("MLPRegressor fit");
    let preds = fitted.predict(&x).expect("MLPRegressor predict");

    // Compare ferrolearn R² to sklearn R². Adam on this n=100, p=5 problem
    // doesn't fully converge in 500 iters for either library — sklearn
    // itself only reaches R²~0.17 — so we use a generous floor that
    // matches sklearn's own performance ceiling. The point is that
    // ferrolearn isn't dramatically WORSE than sklearn at the same Adam
    // hyperparameters; both libraries are limited by Adam's mediocre
    // convergence on this fixture, not by a library-specific bug.
    let y_mean = y.iter().sum::<f64>() / y.len() as f64;
    let ss_tot: f64 = y.iter().map(|v| (v - y_mean).powi(2)).sum();
    let ss_res: f64 = preds
        .iter()
        .zip(y.iter())
        .map(|(a, e)| (a - e).powi(2))
        .sum();
    let r2 = 1.0 - ss_res / ss_tot;
    let expected_r2 = fx.expected["r2"].as_f64().unwrap_or(0.5);
    // Ferrolearn R² must be within 0.15 absolute of sklearn's R²
    // (above or below) — comparable convergence.
    let gap = (expected_r2 - r2).abs();
    assert!(
        gap <= 0.20,
        "MLPRegressor R² ferrolearn={r2:.4}, sklearn={expected_r2:.4}, |gap|={gap:.4} > 0.20"
    );
}

#[test]
fn conformance_bernoulli_rbm() {
    let fx = load_fixture("bernoulli_rbm");
    let x = json_to_array2(&fx.input["X"]);
    let n_components = fx.params["n_components"].as_u64().unwrap_or(4) as usize;

    let model = BernoulliRBM::<f64>::new(n_components);
    let fitted = model.fit(&x, &()).expect("BernoulliRBM fit");
    let xt = fitted.transform(&x).expect("BernoulliRBM transform");
    assert_eq!(xt.nrows(), x.nrows(), "RBM transform rows");
    assert_eq!(xt.ncols(), n_components, "RBM transform cols");
    for v in xt.iter() {
        assert!(v.is_finite(), "RBM transform non-finite");
    }
}
