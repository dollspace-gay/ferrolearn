//! Green-guard tests for threshold metrics against sklearn 1.9 source
//! `sklearn.metrics._ranking`.
//!
//! The installed local oracle is sklearn 1.5.2, which predates
//! `confusion_matrix_at_thresholds` and `metric_at_thresholds`. Expected values
//! below are therefore pinned from the 1.9 source doc examples and tests in the
//! local sklearn mirror (`.sklearn-ref/scikit-learn`, `_ranking.py:934` and
//! `tests/test_ranking.py:2417`).

use ferrolearn_metrics::{
    accuracy_score, confusion_matrix_at_thresholds,
    confusion_matrix_at_thresholds_with_sample_weight, metric_at_thresholds,
    metric_at_thresholds_with_sample_weight,
};
use ndarray::{Array1, array};

fn assert_close_array(actual: &Array1<f64>, expected: &[f64]) {
    assert_eq!(actual.len(), expected.len(), "array length");
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!((a - e).abs() < 1e-12, "value[{i}] actual={a} expected={e}");
    }
}

fn weighted_accuracy(
    y_true: &Array1<usize>,
    y_pred: &Array1<usize>,
    sample_weight: &Array1<f64>,
) -> Result<f64, ferrolearn_core::FerroError> {
    let total: f64 = sample_weight.sum();
    let correct: f64 = y_true
        .iter()
        .zip(y_pred.iter())
        .zip(sample_weight.iter())
        .filter_map(|((&t, &p), &w)| (t == p).then_some(w))
        .sum();
    Ok(correct / total)
}

#[test]
fn green_confusion_matrix_at_thresholds_doc_example() {
    // sklearn 1.9 `_ranking.py` doc example:
    // confusion_matrix_at_thresholds([0,0,1,1], [0.1,0.4,0.35,0.8])
    let y_true = array![0usize, 0, 1, 1];
    let y_score = array![0.1_f64, 0.4, 0.35, 0.8];
    let (tns, fps, fns, tps, thresholds) =
        confusion_matrix_at_thresholds(&y_true, &y_score).unwrap();

    assert_close_array(&tns, &[2.0, 1.0, 1.0, 0.0]);
    assert_close_array(&fps, &[0.0, 1.0, 1.0, 2.0]);
    assert_close_array(&fns, &[1.0, 1.0, 0.0, 0.0]);
    assert_close_array(&tps, &[1.0, 1.0, 2.0, 2.0]);
    assert_close_array(&thresholds, &[0.8, 0.4, 0.35, 0.1]);
}

#[test]
fn green_confusion_matrix_at_thresholds_zero_weight_filtering() {
    // Mirrors sklearn `test_confusion_matrix_at_thresholds_zero_sample_weight`:
    // zero-weighted samples are removed before unique thresholds are extracted.
    let y_true = array![0usize, 0, 1, 1, 1];
    let y_score = array![0.1_f64, 0.2, 0.3, 0.4, 0.5];
    let sample_weight = array![1.0_f64, 1.0, 1.0, 0.5, 0.0];
    let (tns, fps, fns, tps, thresholds) =
        confusion_matrix_at_thresholds_with_sample_weight(&y_true, &y_score, &sample_weight)
            .unwrap();

    assert_close_array(&tns, &[2.0, 2.0, 1.0, 0.0]);
    assert_close_array(&fps, &[0.0, 0.0, 1.0, 2.0]);
    assert_close_array(&fns, &[1.0, 0.0, 0.0, 0.0]);
    assert_close_array(&tps, &[0.5, 1.5, 1.5, 1.5]);
    assert_close_array(&thresholds, &[0.4, 0.3, 0.2, 0.1]);
}

#[test]
fn green_metric_at_thresholds_accuracy_doc_example() {
    // sklearn 1.9 `_ranking.py` doc example:
    // metric_at_thresholds(..., accuracy_score) returns thresholds
    // [0.8, 0.4, 0.35, 0.1] and scores [0.75, 0.5, 0.75, 0.5].
    let y_true = array![0usize, 0, 1, 1];
    let y_score = array![0.1_f64, 0.4, 0.35, 0.8];
    let (metric_values, thresholds) =
        metric_at_thresholds(&y_true, &y_score, accuracy_score).unwrap();

    assert_close_array(&thresholds, &[0.8, 0.4, 0.35, 0.1]);
    assert_close_array(&metric_values, &[0.75, 0.5, 0.75, 0.5]);
}

#[test]
fn green_metric_at_thresholds_duplicate_scores_are_batched() {
    let y_true = array![0usize, 0, 1, 1, 1];
    let y_score = array![0.1_f64, 0.6, 0.4, 0.9, 0.4];
    let (metric_values, thresholds) =
        metric_at_thresholds(&y_true, &y_score, accuracy_score).unwrap();

    assert_close_array(&thresholds, &[0.9, 0.6, 0.4, 0.1]);
    assert_close_array(&metric_values, &[0.6, 0.4, 0.8, 0.6]);
}

#[test]
fn green_metric_at_thresholds_weighted_accuracy() {
    let y_true = array![0usize, 0, 1, 1, 1];
    let y_score = array![0.1_f64, 0.6, 0.4, 0.9, 0.4];
    let sample_weight = array![1.0_f64, 2.0, 3.0, 1.0, 2.0];
    let (metric_values, thresholds) = metric_at_thresholds_with_sample_weight(
        &y_true,
        &y_score,
        &sample_weight,
        weighted_accuracy,
    )
    .unwrap();

    assert_close_array(&thresholds, &[0.9, 0.6, 0.4, 0.1]);
    assert_close_array(
        &metric_values,
        &[4.0 / 9.0, 2.0 / 9.0, 7.0 / 9.0, 6.0 / 9.0],
    );
}

#[test]
fn green_threshold_helpers_validation_boundaries() {
    let y_true = array![0usize, 1];
    let y_score = array![0.1_f64, 0.2];
    assert!(confusion_matrix_at_thresholds(&array![0usize, 1, 2], &array![0.1, 0.2, 0.3]).is_err());
    assert!(confusion_matrix_at_thresholds(&y_true, &array![0.1]).is_err());
    assert!(confusion_matrix_at_thresholds(&y_true, &array![f64::NAN, 0.2]).is_err());
    assert!(metric_at_thresholds(&y_true, &array![f64::INFINITY, 0.2], accuracy_score).is_err());
    assert!(
        confusion_matrix_at_thresholds_with_sample_weight(&y_true, &y_score, &array![0.0, 0.0])
            .is_err()
    );
}
