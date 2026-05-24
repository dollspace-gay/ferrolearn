//! Wave-4 bayes conformance vs scikit-learn: CategoricalNB.

use ferrolearn_bayes::CategoricalNB;
use ferrolearn_core::{Fit, Predict};
use ferrolearn_test_oracle::{json_to_array2, load_fixture};

#[test]
fn conformance_categorical_nb() {
    let fx = load_fixture("categorical_nb");
    // X is integer-valued, but ferrolearn takes f64; cast.
    let x = json_to_array2(&fx.input["X"]);
    let y_vec: Vec<usize> = fx.input["y"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let y = ndarray::Array1::from_vec(y_vec);

    let alpha = fx.params["alpha"].as_f64().unwrap_or(1.0);
    let model = CategoricalNB::<f64>::new().with_alpha(alpha);
    let fitted = model.fit(&x, &y).expect("CategoricalNB fit");
    let preds = fitted.predict(&x).expect("CategoricalNB predict");
    let expected: Vec<usize> = fx.expected["predictions"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let matches = preds
        .iter()
        .zip(expected.iter())
        .filter(|(a, e)| a == e)
        .count();
    let acc = matches as f64 / preds.len() as f64;
    assert!(acc >= 0.95, "CategoricalNB accuracy {acc:.4} < 0.95 floor");
}
