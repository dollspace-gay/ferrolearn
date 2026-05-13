//! Conformance tests for ferrolearn-metrics vs scikit-learn.
//!
//! Each test loads a fixture from `fixtures/<name>.json`, runs the matching
//! ferrolearn metric on the input, and compares against sklearn's value via
//! `ferrolearn-test-oracle` helpers.
//!
//! Metrics are closed-form reductions over inputs — agreement should be tight
//! (default `TOL_METRIC_REL = 1e-9`, `TOL_METRIC_ABS = 1e-12`). When a fixture
//! drifts further than that it is almost certainly a real implementation bug.

use ferrolearn_metrics::{
    Average, accuracy_score, adjusted_mutual_info, adjusted_rand_score, confusion_matrix,
    davies_bouldin_score, explained_variance_score, f1_score, log_loss, mean_absolute_error,
    mean_absolute_percentage_error, mean_squared_error, precision_score, r2_score, recall_score,
    roc_auc_score, root_mean_squared_error, silhouette_score,
};
use ferrolearn_test_oracle::{
    TOL_METRIC_ABS, TOL_METRIC_REL, assert_close, json_to_array1, json_to_array2, load_fixture,
};
use ndarray::{Array1, Array2};

// ---------------------------------------------------------------------------
// Small local helpers — convert JSON-loaded arrays into the integer label
// types expected by the metric functions (Array1<usize> for classification,
// Array1<isize> for clustering).
// ---------------------------------------------------------------------------

fn array1_to_usize(arr: &Array1<f64>) -> Array1<usize> {
    arr.iter().map(|&v| v as usize).collect()
}

fn array1_to_isize(arr: &Array1<f64>) -> Array1<isize> {
    arr.iter().map(|&v| v as isize).collect()
}

// ---------------------------------------------------------------------------
// 1. Binary classification metrics — accuracy, precision, recall, F1,
//    confusion matrix.
// ---------------------------------------------------------------------------

#[test]
fn conformance_classification_metrics() {
    let fx = load_fixture("classification_metrics");
    let (rel, abs) = fx.tolerance(TOL_METRIC_REL, TOL_METRIC_ABS);

    let y_true = array1_to_usize(&json_to_array1(&fx.input["y_true"]));
    let y_pred = array1_to_usize(&json_to_array1(&fx.input["y_pred"]));

    // Accuracy.
    let acc = accuracy_score(&y_true, &y_pred).expect("accuracy_score");
    assert_close(
        acc,
        fx.expected["accuracy"].as_f64().unwrap(),
        rel,
        abs,
        "accuracy_score",
    );

    // Precision / recall / F1 in binary mode (per fixture params.average).
    let prec = precision_score(&y_true, &y_pred, Average::Binary).expect("precision_score");
    assert_close(
        prec,
        fx.expected["precision"].as_f64().unwrap(),
        rel,
        abs,
        "precision_score",
    );

    let rec = recall_score(&y_true, &y_pred, Average::Binary).expect("recall_score");
    assert_close(
        rec,
        fx.expected["recall"].as_f64().unwrap(),
        rel,
        abs,
        "recall_score",
    );

    let f1 = f1_score(&y_true, &y_pred, Average::Binary).expect("f1_score");
    assert_close(
        f1,
        fx.expected["f1"].as_f64().unwrap(),
        rel,
        abs,
        "f1_score",
    );

    // Confusion matrix — counts must match exactly (integer equality).
    let cm = confusion_matrix(&y_true, &y_pred).expect("confusion_matrix");
    let expected_cm = &fx.expected["confusion_matrix"]
        .as_array()
        .expect("expected confusion_matrix array");
    for (i, row) in expected_cm.iter().enumerate() {
        for (j, val) in row.as_array().unwrap().iter().enumerate() {
            let expected = val.as_u64().unwrap() as usize;
            assert_eq!(
                cm[[i, j]],
                expected,
                "confusion_matrix[{i},{j}]: actual={}, expected={expected}",
                cm[[i, j]]
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 2. Clustering metrics — silhouette, ARI, AMI, Davies-Bouldin.
// ---------------------------------------------------------------------------

#[test]
fn conformance_clustering_metrics() {
    let fx = load_fixture("clustering_metrics");
    let (rel, abs) = fx.tolerance(TOL_METRIC_REL, TOL_METRIC_ABS);

    let x: Array2<f64> = json_to_array2(&fx.input["X"]);
    let labels_true = array1_to_isize(&json_to_array1(&fx.input["labels_true"]));
    let labels_pred = array1_to_isize(&json_to_array1(&fx.input["labels_pred"]));

    let sil = silhouette_score(&x, &labels_pred).expect("silhouette_score");
    assert_close(
        sil,
        fx.expected["silhouette"].as_f64().unwrap(),
        rel,
        abs,
        "silhouette_score",
    );

    let ari = adjusted_rand_score(&labels_true, &labels_pred).expect("adjusted_rand_score");
    assert_close(
        ari,
        fx.expected["adjusted_rand"].as_f64().unwrap(),
        rel,
        abs,
        "adjusted_rand_score",
    );

    let ami = adjusted_mutual_info(&labels_true, &labels_pred).expect("adjusted_mutual_info");
    assert_close(
        ami,
        fx.expected["adjusted_mutual_info"].as_f64().unwrap(),
        rel,
        abs,
        "adjusted_mutual_info",
    );

    let dbi = davies_bouldin_score(&x, &labels_pred).expect("davies_bouldin_score");
    assert_close(
        dbi,
        fx.expected["davies_bouldin"].as_f64().unwrap(),
        rel,
        abs,
        "davies_bouldin_score",
    );
}

// ---------------------------------------------------------------------------
// 3. Regression metrics (core) — MAE, MSE, RMSE, R².
// ---------------------------------------------------------------------------

#[test]
fn conformance_regression_metrics() {
    let fx = load_fixture("regression_metrics");
    let (rel, abs) = fx.tolerance(TOL_METRIC_REL, TOL_METRIC_ABS);

    let y_true = json_to_array1(&fx.input["y_true"]);
    let y_pred = json_to_array1(&fx.input["y_pred"]);

    let mae = mean_absolute_error(&y_true, &y_pred).expect("mean_absolute_error");
    assert_close(
        mae,
        fx.expected["mae"].as_f64().unwrap(),
        rel,
        abs,
        "mean_absolute_error",
    );

    let mse = mean_squared_error(&y_true, &y_pred).expect("mean_squared_error");
    assert_close(
        mse,
        fx.expected["mse"].as_f64().unwrap(),
        rel,
        abs,
        "mean_squared_error",
    );

    let rmse = root_mean_squared_error(&y_true, &y_pred).expect("root_mean_squared_error");
    assert_close(
        rmse,
        fx.expected["rmse"].as_f64().unwrap(),
        rel,
        abs,
        "root_mean_squared_error",
    );

    let r2 = r2_score(&y_true, &y_pred).expect("r2_score");
    assert_close(
        r2,
        fx.expected["r2"].as_f64().unwrap(),
        rel,
        abs,
        "r2_score",
    );
}

// ---------------------------------------------------------------------------
// 4. Regression metrics (extended) — MAPE, explained variance.
//
// `mean_absolute_percentage_error` is documented in `regression.rs` as
// returning MAPE *as a percentage* (multiplied by 100), explicitly diverging
// from sklearn's *fraction* convention. The conformance test divides by 100
// before comparing so that the documented convention divergence does not
// produce a false-positive failure — the underlying numbers must still
// agree to metric tolerance.
// ---------------------------------------------------------------------------

#[test]
fn conformance_regression_metrics_extended() {
    let fx = load_fixture("regression_metrics_extended");
    let (rel, abs) = fx.tolerance(TOL_METRIC_REL, TOL_METRIC_ABS);

    let y_true = json_to_array1(&fx.input["y_true"]);
    let y_pred = json_to_array1(&fx.input["y_pred"]);

    let mape =
        mean_absolute_percentage_error(&y_true, &y_pred).expect("mean_absolute_percentage_error");
    // #335 fixed: ferrolearn now returns fraction (no ×100), matching sklearn.
    assert_close(
        mape,
        fx.expected["mape"].as_f64().unwrap(),
        rel,
        abs,
        "mean_absolute_percentage_error",
    );

    let evs = explained_variance_score(&y_true, &y_pred).expect("explained_variance_score");
    assert_close(
        evs,
        fx.expected["explained_variance"].as_f64().unwrap(),
        rel,
        abs,
        "explained_variance_score",
    );
}

// ---------------------------------------------------------------------------
// 5. ROC AUC — binary, single-score input.
// ---------------------------------------------------------------------------

#[test]
fn conformance_roc_auc() {
    let fx = load_fixture("roc_auc");
    let (rel, abs) = fx.tolerance(TOL_METRIC_REL, TOL_METRIC_ABS);

    let y_true = array1_to_usize(&json_to_array1(&fx.input["y_true"]));
    let y_score = json_to_array1(&fx.input["y_score"]);

    let auc = roc_auc_score(&y_true, &y_score).expect("roc_auc_score");
    assert_close(
        auc,
        fx.expected["auc"].as_f64().unwrap(),
        rel,
        abs,
        "roc_auc_score",
    );
}

// ---------------------------------------------------------------------------
// 6. Log loss — binary, requires (n_samples, n_classes) probability matrix.
// ---------------------------------------------------------------------------

#[test]
fn conformance_log_loss() {
    let fx = load_fixture("log_loss");
    let (rel, abs) = fx.tolerance(TOL_METRIC_REL, TOL_METRIC_ABS);

    let y_true = array1_to_usize(&json_to_array1(&fx.input["y_true"]));
    let y_prob = json_to_array2(&fx.input["y_prob"]);

    let loss = log_loss(&y_true, &y_prob).expect("log_loss");
    assert_close(
        loss,
        fx.expected["loss"].as_f64().unwrap(),
        rel,
        abs,
        "log_loss",
    );
}
