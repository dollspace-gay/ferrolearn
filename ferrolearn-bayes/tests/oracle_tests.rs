//! Oracle tests comparing ferrolearn Naive Bayes models against scikit-learn
//! reference outputs stored in `fixtures/*.json`.

use ferrolearn_core::{Fit, Predict};
use ndarray::{Array1, Array2};

fn json_to_array2(value: &serde_json::Value) -> Array2<f64> {
    let rows: Vec<Vec<f64>> = value
        .as_array()
        .unwrap()
        .iter()
        .map(|row| {
            row.as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap())
                .collect()
        })
        .collect();
    let n_rows = rows.len();
    let n_cols = rows[0].len();
    let flat: Vec<f64> = rows.into_iter().flatten().collect();
    Array2::from_shape_vec((n_rows, n_cols), flat).unwrap()
}

fn json_to_labels(value: &serde_json::Value) -> Array1<usize> {
    let vec: Vec<usize> = value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    Array1::from_vec(vec)
}

// ---------------------------------------------------------------------------
// Gaussian Naive Bayes
// ---------------------------------------------------------------------------

#[test]
fn test_gaussian_nb_oracle() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../../fixtures/gaussian_nb.json")).unwrap();

    let x = json_to_array2(&fixture["input"]["X"]);
    let y = json_to_labels(&fixture["input"]["y"]);

    let expected_preds: Vec<usize> = fixture["expected"]["predictions"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let sklearn_accuracy = fixture["expected"]["accuracy"].as_f64().unwrap();

    let model = ferrolearn_bayes::GaussianNB::<f64>::new();
    let fitted = model.fit(&x, &y).unwrap();
    let preds = fitted.predict(&x).unwrap();

    // GaussianNB is a simple algorithm — predictions should match sklearn closely.
    let n_match: usize = preds
        .iter()
        .zip(expected_preds.iter())
        .filter(|(a, b)| a == b)
        .count();
    let match_rate = n_match as f64 / expected_preds.len() as f64;

    assert!(
        match_rate >= 0.95,
        "GaussianNB prediction match rate {match_rate:.4} < 0.95 vs sklearn"
    );

    // Accuracy should be very close to sklearn's.
    let n_correct: usize = preds.iter().zip(y.iter()).filter(|(a, b)| a == b).count();
    let accuracy = n_correct as f64 / y.len() as f64;
    assert!(
        (accuracy - sklearn_accuracy).abs() < 0.05,
        "GaussianNB accuracy {accuracy:.4} too far from sklearn {sklearn_accuracy:.4}"
    );
}
