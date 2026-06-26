//! Green guards for metric display data containers against sklearn display
//! classes.
//!
//! Live sklearn 1.5.2 oracle for the shared binary fixture:
//!
//! ```text
//! y_true = [0, 0, 1, 1]
//! y_pred = [0, 1, 1, 1]
//! y_score = [0.1, 0.4, 0.35, 0.8]
//! ConfusionMatrixDisplay.from_predictions(...).confusion_matrix == [[1, 1], [0, 2]]
//! roc_curve(y_true, y_score) -> fpr=[0, 0, .5, .5, 1], tpr=[0, .5, .5, 1, 1]
//! precision_recall_curve(...) -> precision=[.5, .66666667, .5, 1, 1],
//!                                recall=[1, 1, .5, .5, 0]
//! det_curve(...) -> fpr=[.5, .5, 0], fnr=[0, .5, .5]
//! ```
//!
//! ferrolearn display types intentionally return renderer-neutral data rather
//! than matplotlib artists. Remaining gaps include `from_estimator`, plotting
//! keyword surfaces, sample weights, custom `pos_label`, and Python artist/axis
//! attributes.

use approx::assert_relative_eq;
use ferrolearn_metrics::{
    ConfusionMatrixDisplay, DetCurveDisplay, PrecisionRecallDisplay, PredictionErrorDisplay,
    PredictionErrorKind, RocCurveDisplay,
};
use ndarray::array;

#[test]
fn green_confusion_matrix_display_from_predictions() {
    let y_true = array![0usize, 0, 1, 1];
    let y_pred = array![0usize, 1, 1, 1];

    let display = ConfusionMatrixDisplay::from_predictions(&y_true, &y_pred).unwrap();

    assert_eq!(display.confusion_matrix, array![[1usize, 1], [0, 2]]);
    assert_eq!(display.display_labels, array![0usize, 1]);
}

#[test]
fn green_curve_displays_from_predictions() {
    let y_true = array![0usize, 0, 1, 1];
    let y_score = array![0.1_f64, 0.4, 0.35, 0.8];

    let roc = RocCurveDisplay::from_predictions(&y_true, &y_score).unwrap();
    assert_eq!(roc.fpr.len(), 5);
    assert_relative_eq!(roc.fpr[0], 0.0, epsilon = 1e-12);
    assert_relative_eq!(roc.fpr[2], 0.5, epsilon = 1e-12);
    assert_relative_eq!(roc.tpr[3], 1.0, epsilon = 1e-12);
    assert_relative_eq!(roc.roc_auc.unwrap(), 0.75, epsilon = 1e-12);

    let pr = PrecisionRecallDisplay::from_predictions(&y_true, &y_score).unwrap();
    assert_eq!(pr.precision.len(), 5);
    assert_relative_eq!(pr.precision[0], 0.5, epsilon = 1e-12);
    assert_relative_eq!(pr.precision[1], 2.0 / 3.0, epsilon = 1e-12);
    assert_relative_eq!(pr.recall[2], 0.5, epsilon = 1e-12);
    assert_relative_eq!(pr.average_precision.unwrap(), 5.0 / 6.0, epsilon = 1e-12);

    let det = DetCurveDisplay::from_predictions(&y_true, &y_score).unwrap();
    assert_eq!(det.fpr.len(), 3);
    assert_relative_eq!(det.fpr[0], 0.5, epsilon = 1e-12);
    assert_relative_eq!(det.fnr[0], 0.0, epsilon = 1e-12);
    assert_relative_eq!(det.fnr[2], 0.5, epsilon = 1e-12);
}

#[test]
fn green_prediction_error_display_data_matches_sklearn_semantics() {
    let y_true = array![3.0, -0.5, 2.0, 7.0];
    let y_pred = array![2.5, 0.0, 2.1, 7.8];

    let actual_vs_predicted = PredictionErrorDisplay::from_predictions(
        &y_true,
        &y_pred,
        PredictionErrorKind::ActualVsPredicted,
    )
    .unwrap();
    let (x, y) = actual_vs_predicted.plot_data();
    assert_eq!(x, &y_true);
    assert_eq!(y, &y_pred);

    let residual = PredictionErrorDisplay::from_predictions(
        &y_true,
        &y_pred,
        PredictionErrorKind::ResidualVsPredicted,
    )
    .unwrap();
    let expected_residuals = array![0.5, -0.5, -0.1, -0.8];
    for (actual, expected) in residual.residuals.iter().zip(expected_residuals.iter()) {
        assert_relative_eq!(*actual, *expected, epsilon = 1e-12);
    }
    let (x, y) = residual.plot_data();
    assert_eq!(x, &y_pred);
    assert_eq!(y, &residual.residuals);

    assert!(
        PredictionErrorDisplay::from_predictions(
            &array![1.0, 2.0],
            &array![1.0],
            PredictionErrorKind::ActualVsPredicted
        )
        .is_err()
    );
}
