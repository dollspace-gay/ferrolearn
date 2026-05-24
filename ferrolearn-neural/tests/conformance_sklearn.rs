//! Conformance tests for ferrolearn-neural vs scikit-learn.
//!
//! Both libraries use Adam since ferrolearn doesn't ship an LBFGS solver yet.
//! Adam is stochastic; conformance is on the prediction accuracy floor.

use ferrolearn_core::{Fit, Predict};
use ferrolearn_neural::{Activation, MLPClassifier, Solver};
use ferrolearn_test_oracle::{json_to_array2, load_fixture};

#[test]
fn conformance_mlp_classifier() {
    let fx = load_fixture("mlp_classifier");
    let x = json_to_array2(&fx.input["X"]);
    let y_vec: Vec<usize> = fx.input["y"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let y = ndarray::Array1::from_vec(y_vec);
    let max_iter = fx.params["max_iter"].as_u64().unwrap_or(500) as usize;
    let alpha = fx.params["alpha"].as_f64().unwrap_or(1e-4);
    let random_state = fx.params["random_state"].as_u64().unwrap_or(42);

    let model = MLPClassifier::<f64>::new()
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
    let fitted = model.fit(&x, &y).expect("MLPClassifier fit");

    let preds = fitted.predict(&x).expect("MLPClassifier predict");
    let expected_acc = fx.expected["accuracy"].as_f64().unwrap_or(0.5);
    let matches: usize = preds.iter().zip(y.iter()).filter(|(a, e)| a == e).count();
    let acc = matches as f64 / y.len() as f64;
    // sklearn at Adam defaults also doesn't fully converge on this n=100,
    // p=5, 3-class problem; both libraries achieve similar mediocre Adam
    // accuracy. Floor at 0.6× sklearn's accuracy.
    assert!(
        acc >= 0.6 * expected_acc,
        "MLPClassifier accuracy {acc:.4} < 0.6 * sklearn {expected_acc:.4}"
    );
}
