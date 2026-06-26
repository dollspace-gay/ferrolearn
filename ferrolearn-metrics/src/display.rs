//! Data-backed display helpers for metric visualizations.
//!
//! These types mirror sklearn's public display classes at the data layer. They
//! intentionally do not render with matplotlib; instead they store the arrays a
//! renderer needs so Rust callers can choose their own plotting backend.

use std::collections::BTreeSet;

use ferrolearn_core::FerroError;
use ndarray::{Array1, Array2};
use num_traits::Float;

use crate::classification::{auc, confusion_matrix, det_curve, precision_recall_curve, roc_curve};

/// Data container equivalent to `sklearn.metrics.ConfusionMatrixDisplay`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfusionMatrixDisplay {
    /// Confusion matrix where rows are true labels and columns are predicted labels.
    pub confusion_matrix: Array2<usize>,
    /// Labels displayed on both axes.
    pub display_labels: Array1<usize>,
}

impl ConfusionMatrixDisplay {
    /// Build a display from an already-computed confusion matrix and labels.
    #[must_use]
    pub fn new(confusion_matrix: Array2<usize>, display_labels: Array1<usize>) -> Self {
        Self {
            confusion_matrix,
            display_labels,
        }
    }

    /// Compute the confusion matrix and labels from predictions.
    pub fn from_predictions(
        y_true: &Array1<usize>,
        y_pred: &Array1<usize>,
    ) -> Result<Self, FerroError> {
        let matrix = confusion_matrix(y_true, y_pred)?;
        let labels = unique_labels(y_true, y_pred);
        Ok(Self::new(matrix, labels))
    }
}

/// Data container equivalent to `sklearn.metrics.RocCurveDisplay`.
#[derive(Debug, Clone, PartialEq)]
pub struct RocCurveDisplay<F> {
    /// False-positive rates.
    pub fpr: Array1<F>,
    /// True-positive rates.
    pub tpr: Array1<F>,
    /// Score thresholds.
    pub thresholds: Array1<F>,
    /// Area under the ROC curve.
    pub roc_auc: Option<F>,
    /// Positive class label. This scoped implementation supports sklearn's default `1`.
    pub pos_label: usize,
    /// Optional estimator name for a renderer legend.
    pub estimator_name: Option<String>,
}

impl<F> RocCurveDisplay<F>
where
    F: Float + Send + Sync + 'static,
{
    /// Build a display from already-computed curve arrays.
    #[must_use]
    pub fn new(
        fpr: Array1<F>,
        tpr: Array1<F>,
        thresholds: Array1<F>,
        roc_auc: Option<F>,
        pos_label: usize,
        estimator_name: Option<String>,
    ) -> Self {
        Self {
            fpr,
            tpr,
            thresholds,
            roc_auc,
            pos_label,
            estimator_name,
        }
    }

    /// Compute ROC display data from binary labels and positive-class scores.
    pub fn from_predictions(
        y_true: &Array1<usize>,
        y_score: &Array1<F>,
    ) -> Result<Self, FerroError> {
        let (fpr, tpr, thresholds) = roc_curve(y_true, y_score)?;
        let roc_auc = Some(auc(&fpr, &tpr)?);
        Ok(Self::new(fpr, tpr, thresholds, roc_auc, 1, None))
    }
}

/// Data container equivalent to `sklearn.metrics.PrecisionRecallDisplay`.
#[derive(Debug, Clone, PartialEq)]
pub struct PrecisionRecallDisplay<F> {
    /// Precision values.
    pub precision: Array1<F>,
    /// Recall values.
    pub recall: Array1<F>,
    /// Score thresholds.
    pub thresholds: Array1<F>,
    /// Average precision computed from the curve.
    pub average_precision: Option<F>,
    /// Positive class label. This scoped implementation supports sklearn's default `1`.
    pub pos_label: usize,
    /// Optional estimator name for a renderer legend.
    pub estimator_name: Option<String>,
}

impl<F> PrecisionRecallDisplay<F>
where
    F: Float + Send + Sync + 'static,
{
    /// Build a display from already-computed curve arrays.
    #[must_use]
    pub fn new(
        precision: Array1<F>,
        recall: Array1<F>,
        thresholds: Array1<F>,
        average_precision: Option<F>,
        pos_label: usize,
        estimator_name: Option<String>,
    ) -> Self {
        Self {
            precision,
            recall,
            thresholds,
            average_precision,
            pos_label,
            estimator_name,
        }
    }

    /// Compute precision-recall display data from binary labels and scores.
    pub fn from_predictions(
        y_true: &Array1<usize>,
        y_score: &Array1<F>,
    ) -> Result<Self, FerroError> {
        let (precision, recall, thresholds) = precision_recall_curve(y_true, y_score)?;
        let average_precision = Some(average_precision_from_curve(&precision, &recall));
        Ok(Self::new(
            precision,
            recall,
            thresholds,
            average_precision,
            1,
            None,
        ))
    }
}

/// Data container equivalent to `sklearn.metrics.DetCurveDisplay`.
#[derive(Debug, Clone, PartialEq)]
pub struct DetCurveDisplay<F> {
    /// False-positive rates.
    pub fpr: Array1<F>,
    /// False-negative rates.
    pub fnr: Array1<F>,
    /// Score thresholds.
    pub thresholds: Array1<F>,
    /// Positive class label. This scoped implementation supports sklearn's default `1`.
    pub pos_label: usize,
    /// Optional estimator name for a renderer legend.
    pub estimator_name: Option<String>,
}

impl<F> DetCurveDisplay<F>
where
    F: Float + Send + Sync + 'static,
{
    /// Build a display from already-computed DET arrays.
    #[must_use]
    pub fn new(
        fpr: Array1<F>,
        fnr: Array1<F>,
        thresholds: Array1<F>,
        pos_label: usize,
        estimator_name: Option<String>,
    ) -> Self {
        Self {
            fpr,
            fnr,
            thresholds,
            pos_label,
            estimator_name,
        }
    }

    /// Compute DET display data from binary labels and scores.
    pub fn from_predictions(
        y_true: &Array1<usize>,
        y_score: &Array1<F>,
    ) -> Result<Self, FerroError> {
        let (fpr, fnr, thresholds) = det_curve(y_true, y_score)?;
        Ok(Self::new(fpr, fnr, thresholds, 1, None))
    }
}

/// Plot kind for [`PredictionErrorDisplay`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PredictionErrorKind {
    /// Plot true targets on the x-axis and predictions on the y-axis.
    ActualVsPredicted,
    /// Plot predictions on the x-axis and residuals on the y-axis.
    ResidualVsPredicted,
}

/// Data container equivalent to `sklearn.metrics.PredictionErrorDisplay`.
#[derive(Debug, Clone, PartialEq)]
pub struct PredictionErrorDisplay<F> {
    /// Ground-truth regression targets.
    pub y_true: Array1<F>,
    /// Predicted regression targets.
    pub y_pred: Array1<F>,
    /// Residuals, computed as `y_true - y_pred`.
    pub residuals: Array1<F>,
    /// Display kind.
    pub kind: PredictionErrorKind,
    /// Optional estimator name for a renderer legend.
    pub estimator_name: Option<String>,
}

impl<F> PredictionErrorDisplay<F>
where
    F: Float + Send + Sync + 'static,
{
    /// Build a display from predictions.
    pub fn from_predictions(
        y_true: &Array1<F>,
        y_pred: &Array1<F>,
        kind: PredictionErrorKind,
    ) -> Result<Self, FerroError> {
        validate_regression_inputs(y_true, y_pred)?;
        let residuals = y_true - y_pred;
        Ok(Self {
            y_true: y_true.clone(),
            y_pred: y_pred.clone(),
            residuals,
            kind,
            estimator_name: None,
        })
    }

    /// Return the x/y arrays a renderer should draw for this display kind.
    #[must_use]
    pub fn plot_data(&self) -> (&Array1<F>, &Array1<F>) {
        match self.kind {
            PredictionErrorKind::ActualVsPredicted => (&self.y_true, &self.y_pred),
            PredictionErrorKind::ResidualVsPredicted => (&self.y_pred, &self.residuals),
        }
    }
}

fn unique_labels(y_true: &Array1<usize>, y_pred: &Array1<usize>) -> Array1<usize> {
    let labels = y_true
        .iter()
        .chain(y_pred.iter())
        .copied()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    Array1::from_vec(labels)
}

fn average_precision_from_curve<F>(precision: &Array1<F>, recall: &Array1<F>) -> F
where
    F: Float,
{
    let mut ap = F::zero();
    for i in 1..recall.len() {
        ap = ap + (recall[i - 1] - recall[i]).abs() * precision[i - 1];
    }
    ap
}

fn validate_regression_inputs<F>(y_true: &Array1<F>, y_pred: &Array1<F>) -> Result<(), FerroError>
where
    F: Float,
{
    if y_true.len() != y_pred.len() {
        return Err(FerroError::ShapeMismatch {
            expected: vec![y_true.len()],
            actual: vec![y_pred.len()],
            context: "PredictionErrorDisplay: y_true vs y_pred".into(),
        });
    }
    if y_true.is_empty() {
        return Err(FerroError::InsufficientSamples {
            required: 1,
            actual: 0,
            context: "PredictionErrorDisplay".into(),
        });
    }
    if y_true.iter().chain(y_pred.iter()).any(|v| !v.is_finite()) {
        return Err(FerroError::InvalidParameter {
            name: "y_true/y_pred".into(),
            reason: "values must be finite".into(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn confusion_matrix_display_from_predictions() {
        let y_true = array![0usize, 0, 1, 1];
        let y_pred = array![0usize, 1, 1, 1];
        let display = ConfusionMatrixDisplay::from_predictions(&y_true, &y_pred).unwrap();
        assert_eq!(display.confusion_matrix, array![[1usize, 1], [0, 2]]);
        assert_eq!(display.display_labels, array![0usize, 1]);
    }

    #[test]
    fn curve_displays_from_predictions() {
        let y_true = array![0usize, 0, 1, 1];
        let y_score = array![0.1_f64, 0.4, 0.35, 0.8];
        let roc = RocCurveDisplay::from_predictions(&y_true, &y_score).unwrap();
        assert_eq!(roc.fpr.len(), roc.tpr.len());
        assert_eq!(roc.thresholds.len(), roc.fpr.len());
        assert_relative_eq!(roc.roc_auc.unwrap(), 0.75, epsilon = 1e-12);

        let pr = PrecisionRecallDisplay::from_predictions(&y_true, &y_score).unwrap();
        assert_eq!(pr.precision.len(), pr.recall.len());
        assert_eq!(pr.thresholds.len() + 1, pr.precision.len());
        assert_relative_eq!(pr.average_precision.unwrap(), 5.0 / 6.0, epsilon = 1e-12);

        let det = DetCurveDisplay::from_predictions(&y_true, &y_score).unwrap();
        assert_eq!(det.fpr.len(), det.fnr.len());
        assert_eq!(det.thresholds.len(), det.fpr.len());
    }

    #[test]
    fn prediction_error_display_plot_data() {
        let y_true = array![3.0, -0.5, 2.0];
        let y_pred = array![2.5, 0.0, 2.1];
        let display = PredictionErrorDisplay::from_predictions(
            &y_true,
            &y_pred,
            PredictionErrorKind::ResidualVsPredicted,
        )
        .unwrap();
        let expected_residuals = array![0.5, -0.5, -0.1];
        for (actual, expected) in display.residuals.iter().zip(expected_residuals.iter()) {
            assert_relative_eq!(*actual, *expected, epsilon = 1e-12);
        }
        let (x, y) = display.plot_data();
        assert_eq!(x, &display.y_pred);
        assert_eq!(y, &display.residuals);
    }
}
