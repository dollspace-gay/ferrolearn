//! Statistical equivalence tests comparing ferrolearn metrics against
//! scikit-learn reference values stored in `tests/fixtures/sklearn_reference/`.
//!
//! These tests verify that ferrolearn's metric computations produce numerically
//! identical results to scikit-learn, with very tight tolerances (< 1e-12 for
//! most metrics).

use approx::assert_relative_eq;
use ndarray::{Array1, Array2};

// ---------------------------------------------------------------------------
// JSON helpers
// ---------------------------------------------------------------------------

fn json_to_array1_f64(value: &serde_json::Value) -> Array1<f64> {
    let vec: Vec<f64> = value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    Array1::from_vec(vec)
}

fn json_to_array1_usize(value: &serde_json::Value) -> Array1<usize> {
    let vec: Vec<usize> = value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    Array1::from_vec(vec)
}

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

// ---------------------------------------------------------------------------
// Classification Metrics
// ---------------------------------------------------------------------------

#[test]
fn sklearn_equiv_classification_accuracy() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../tests/fixtures/sklearn_reference/classification_metrics.json"
    ))
    .unwrap();

    let y_true = json_to_array1_usize(&fixture["y_true"]);
    let y_pred = json_to_array1_usize(&fixture["y_pred"]);
    let expected = fixture["accuracy"].as_f64().unwrap();

    let actual = ferrolearn_metrics::accuracy_score(&y_true, &y_pred).unwrap();
    assert_relative_eq!(actual, expected, epsilon = 1e-12);
}

#[test]
fn sklearn_equiv_classification_precision_binary() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../tests/fixtures/sklearn_reference/classification_metrics.json"
    ))
    .unwrap();

    let y_true = json_to_array1_usize(&fixture["y_true"]);
    let y_pred = json_to_array1_usize(&fixture["y_pred"]);
    let expected = fixture["precision_binary"].as_f64().unwrap();

    let actual =
        ferrolearn_metrics::precision_score(&y_true, &y_pred, ferrolearn_metrics::Average::Binary)
            .unwrap();
    assert_relative_eq!(actual, expected, epsilon = 1e-12);
}

#[test]
fn sklearn_equiv_classification_recall_binary() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../tests/fixtures/sklearn_reference/classification_metrics.json"
    ))
    .unwrap();

    let y_true = json_to_array1_usize(&fixture["y_true"]);
    let y_pred = json_to_array1_usize(&fixture["y_pred"]);
    let expected = fixture["recall_binary"].as_f64().unwrap();

    let actual =
        ferrolearn_metrics::recall_score(&y_true, &y_pred, ferrolearn_metrics::Average::Binary)
            .unwrap();
    assert_relative_eq!(actual, expected, epsilon = 1e-12);
}

#[test]
fn sklearn_equiv_classification_f1_binary() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../tests/fixtures/sklearn_reference/classification_metrics.json"
    ))
    .unwrap();

    let y_true = json_to_array1_usize(&fixture["y_true"]);
    let y_pred = json_to_array1_usize(&fixture["y_pred"]);
    let expected = fixture["f1_binary"].as_f64().unwrap();

    let actual =
        ferrolearn_metrics::f1_score(&y_true, &y_pred, ferrolearn_metrics::Average::Binary)
            .unwrap();
    assert_relative_eq!(actual, expected, epsilon = 1e-12);
}

#[test]
fn sklearn_equiv_classification_precision_macro() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../tests/fixtures/sklearn_reference/classification_metrics.json"
    ))
    .unwrap();

    let y_true = json_to_array1_usize(&fixture["y_true"]);
    let y_pred = json_to_array1_usize(&fixture["y_pred"]);
    let expected = fixture["precision_macro"].as_f64().unwrap();

    let actual =
        ferrolearn_metrics::precision_score(&y_true, &y_pred, ferrolearn_metrics::Average::Macro)
            .unwrap();
    assert_relative_eq!(actual, expected, epsilon = 1e-12);
}

#[test]
fn sklearn_equiv_classification_recall_macro() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../tests/fixtures/sklearn_reference/classification_metrics.json"
    ))
    .unwrap();

    let y_true = json_to_array1_usize(&fixture["y_true"]);
    let y_pred = json_to_array1_usize(&fixture["y_pred"]);
    let expected = fixture["recall_macro"].as_f64().unwrap();

    let actual =
        ferrolearn_metrics::recall_score(&y_true, &y_pred, ferrolearn_metrics::Average::Macro)
            .unwrap();
    assert_relative_eq!(actual, expected, epsilon = 1e-12);
}

#[test]
fn sklearn_equiv_classification_f1_macro() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../tests/fixtures/sklearn_reference/classification_metrics.json"
    ))
    .unwrap();

    let y_true = json_to_array1_usize(&fixture["y_true"]);
    let y_pred = json_to_array1_usize(&fixture["y_pred"]);
    let expected = fixture["f1_macro"].as_f64().unwrap();

    let actual =
        ferrolearn_metrics::f1_score(&y_true, &y_pred, ferrolearn_metrics::Average::Macro).unwrap();
    assert_relative_eq!(actual, expected, epsilon = 1e-12);
}

#[test]
fn sklearn_equiv_classification_precision_weighted() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../tests/fixtures/sklearn_reference/classification_metrics.json"
    ))
    .unwrap();

    let y_true = json_to_array1_usize(&fixture["y_true"]);
    let y_pred = json_to_array1_usize(&fixture["y_pred"]);
    let expected = fixture["precision_weighted"].as_f64().unwrap();

    let actual = ferrolearn_metrics::precision_score(
        &y_true,
        &y_pred,
        ferrolearn_metrics::Average::Weighted,
    )
    .unwrap();
    assert_relative_eq!(actual, expected, epsilon = 1e-12);
}

#[test]
fn sklearn_equiv_classification_recall_weighted() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../tests/fixtures/sklearn_reference/classification_metrics.json"
    ))
    .unwrap();

    let y_true = json_to_array1_usize(&fixture["y_true"]);
    let y_pred = json_to_array1_usize(&fixture["y_pred"]);
    let expected = fixture["recall_weighted"].as_f64().unwrap();

    let actual =
        ferrolearn_metrics::recall_score(&y_true, &y_pred, ferrolearn_metrics::Average::Weighted)
            .unwrap();
    assert_relative_eq!(actual, expected, epsilon = 1e-12);
}

#[test]
fn sklearn_equiv_classification_f1_weighted() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../tests/fixtures/sklearn_reference/classification_metrics.json"
    ))
    .unwrap();

    let y_true = json_to_array1_usize(&fixture["y_true"]);
    let y_pred = json_to_array1_usize(&fixture["y_pred"]);
    let expected = fixture["f1_weighted"].as_f64().unwrap();

    let actual =
        ferrolearn_metrics::f1_score(&y_true, &y_pred, ferrolearn_metrics::Average::Weighted)
            .unwrap();
    assert_relative_eq!(actual, expected, epsilon = 1e-12);
}

#[test]
fn sklearn_equiv_confusion_matrix() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../tests/fixtures/sklearn_reference/classification_metrics.json"
    ))
    .unwrap();

    let y_true = json_to_array1_usize(&fixture["y_true"]);
    let y_pred = json_to_array1_usize(&fixture["y_pred"]);
    let expected_cm: Vec<Vec<usize>> = fixture["confusion_matrix"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| {
            row.as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect()
        })
        .collect();

    let cm = ferrolearn_metrics::confusion_matrix(&y_true, &y_pred).unwrap();
    for (i, expected_row) in expected_cm.iter().enumerate() {
        for (j, &expected_val) in expected_row.iter().enumerate() {
            assert_eq!(
                cm[[i, j]],
                expected_val,
                "confusion_matrix[{i},{j}]: actual={}, expected={expected_val}",
                cm[[i, j]]
            );
        }
    }
}

#[test]
fn sklearn_equiv_roc_auc() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../tests/fixtures/sklearn_reference/classification_metrics.json"
    ))
    .unwrap();

    let y_true = json_to_array1_usize(&fixture["y_true"]);
    let y_scores = json_to_array1_f64(&fixture["y_scores"]);
    let expected = fixture["roc_auc"].as_f64().unwrap();

    let actual = ferrolearn_metrics::roc_auc_score(&y_true, &y_scores).unwrap();
    assert_relative_eq!(actual, expected, epsilon = 1e-12);
}

#[test]
fn sklearn_equiv_log_loss() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../tests/fixtures/sklearn_reference/classification_metrics.json"
    ))
    .unwrap();

    let y_true = json_to_array1_usize(&fixture["y_true"]);
    let y_proba = json_to_array2(&fixture["y_proba"]);
    let expected = fixture["log_loss"].as_f64().unwrap();

    let actual = ferrolearn_metrics::log_loss(&y_true, &y_proba).unwrap();
    // Log loss may have slightly larger numerical differences due to
    // log computation, so use a tolerance of 1e-10.
    assert_relative_eq!(actual, expected, epsilon = 1e-10);
}

// ---------------------------------------------------------------------------
// Regression Metrics
// ---------------------------------------------------------------------------

#[test]
fn sklearn_equiv_regression_mae() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../tests/fixtures/sklearn_reference/regression_metrics.json"
    ))
    .unwrap();

    let y_true = json_to_array1_f64(&fixture["y_true"]);
    let y_pred = json_to_array1_f64(&fixture["y_pred"]);
    let expected = fixture["mae"].as_f64().unwrap();

    let actual: f64 = ferrolearn_metrics::mean_absolute_error(&y_true, &y_pred).unwrap();
    assert_relative_eq!(actual, expected, epsilon = 1e-12);
}

#[test]
fn sklearn_equiv_regression_mse() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../tests/fixtures/sklearn_reference/regression_metrics.json"
    ))
    .unwrap();

    let y_true = json_to_array1_f64(&fixture["y_true"]);
    let y_pred = json_to_array1_f64(&fixture["y_pred"]);
    let expected = fixture["mse"].as_f64().unwrap();

    let actual: f64 = ferrolearn_metrics::mean_squared_error(&y_true, &y_pred).unwrap();
    assert_relative_eq!(actual, expected, epsilon = 1e-12);
}

#[test]
fn sklearn_equiv_regression_rmse() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../tests/fixtures/sklearn_reference/regression_metrics.json"
    ))
    .unwrap();

    let y_true = json_to_array1_f64(&fixture["y_true"]);
    let y_pred = json_to_array1_f64(&fixture["y_pred"]);
    let expected = fixture["rmse"].as_f64().unwrap();

    let actual: f64 = ferrolearn_metrics::root_mean_squared_error(&y_true, &y_pred).unwrap();
    assert_relative_eq!(actual, expected, epsilon = 1e-12);
}

#[test]
fn sklearn_equiv_regression_r2() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../tests/fixtures/sklearn_reference/regression_metrics.json"
    ))
    .unwrap();

    let y_true = json_to_array1_f64(&fixture["y_true"]);
    let y_pred = json_to_array1_f64(&fixture["y_pred"]);
    let expected = fixture["r2"].as_f64().unwrap();

    let actual: f64 = ferrolearn_metrics::r2_score(&y_true, &y_pred).unwrap();
    assert_relative_eq!(actual, expected, epsilon = 1e-12);
}

#[test]
fn sklearn_equiv_regression_mape() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../tests/fixtures/sklearn_reference/regression_metrics.json"
    ))
    .unwrap();

    let y_true = json_to_array1_f64(&fixture["y_true"]);
    let y_pred = json_to_array1_f64(&fixture["y_pred"]);
    let expected = fixture["mape"].as_f64().unwrap();

    let actual: f64 = ferrolearn_metrics::mean_absolute_percentage_error(&y_true, &y_pred).unwrap();
    // ferrolearn returns MAPE as a percentage (multiplied by 100),
    // while sklearn returns it as a fraction. Adjust for comparison.
    let actual_fraction = actual / 100.0;
    assert_relative_eq!(actual_fraction, expected, epsilon = 1e-10);
}

#[test]
fn sklearn_equiv_regression_explained_variance() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../tests/fixtures/sklearn_reference/regression_metrics.json"
    ))
    .unwrap();

    let y_true = json_to_array1_f64(&fixture["y_true"]);
    let y_pred = json_to_array1_f64(&fixture["y_pred"]);
    let expected = fixture["explained_variance"].as_f64().unwrap();

    let actual: f64 = ferrolearn_metrics::explained_variance_score(&y_true, &y_pred).unwrap();
    assert_relative_eq!(actual, expected, epsilon = 1e-12);
}

// ---------------------------------------------------------------------------
// Clustering Metrics
// ---------------------------------------------------------------------------

#[test]
fn sklearn_equiv_silhouette_score() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../tests/fixtures/sklearn_reference/clustering_metrics.json"
    ))
    .unwrap();

    let x = json_to_array2(&fixture["X"]);
    let labels_true: Vec<usize> = fixture["labels_true"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let labels =
        ndarray::Array1::from_vec(labels_true.iter().map(|&l| l as isize).collect::<Vec<_>>());
    let expected = fixture["silhouette_score"].as_f64().unwrap();

    let actual = ferrolearn_metrics::silhouette_score(&x, &labels).unwrap();
    // Silhouette involves many distance calculations, so allow slightly
    // looser tolerance.
    assert_relative_eq!(actual, expected, epsilon = 1e-10);
}

#[test]
fn sklearn_equiv_adjusted_rand_score() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../tests/fixtures/sklearn_reference/clustering_metrics.json"
    ))
    .unwrap();

    let labels_true: Vec<isize> = fixture["labels_true"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap() as isize)
        .collect();
    let labels_pred: Vec<isize> = fixture["labels_pred"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap() as isize)
        .collect();
    let labels_a = ndarray::Array1::from_vec(labels_true);
    let labels_b = ndarray::Array1::from_vec(labels_pred);
    let expected = fixture["adjusted_rand_score"].as_f64().unwrap();

    let actual = ferrolearn_metrics::adjusted_rand_score(&labels_a, &labels_b).unwrap();
    assert_relative_eq!(actual, expected, epsilon = 1e-10);
}

#[test]
fn sklearn_equiv_adjusted_mutual_info() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../tests/fixtures/sklearn_reference/clustering_metrics.json"
    ))
    .unwrap();

    let labels_true: Vec<isize> = fixture["labels_true"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap() as isize)
        .collect();
    let labels_pred: Vec<isize> = fixture["labels_pred"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap() as isize)
        .collect();
    let labels_a = ndarray::Array1::from_vec(labels_true);
    let labels_b = ndarray::Array1::from_vec(labels_pred);
    let expected = fixture["adjusted_mutual_info"].as_f64().unwrap();

    let actual = ferrolearn_metrics::adjusted_mutual_info(&labels_a, &labels_b).unwrap();
    // AMI involves log computations, allow moderate tolerance.
    assert_relative_eq!(actual, expected, epsilon = 1e-6);
}

#[test]
fn sklearn_equiv_davies_bouldin_score() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../tests/fixtures/sklearn_reference/clustering_metrics.json"
    ))
    .unwrap();

    let x = json_to_array2(&fixture["X"]);
    let labels_true: Vec<usize> = fixture["labels_true"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let labels =
        ndarray::Array1::from_vec(labels_true.iter().map(|&l| l as isize).collect::<Vec<_>>());
    let expected = fixture["davies_bouldin_score"].as_f64().unwrap();

    let actual = ferrolearn_metrics::davies_bouldin_score(&x, &labels).unwrap();
    assert_relative_eq!(actual, expected, epsilon = 1e-10);
}
