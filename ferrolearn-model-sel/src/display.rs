//! Data-backed display helpers for model-selection and inspection results.
//!
//! These structs mirror sklearn's public display class names at the stored-data
//! layer. They do not render with matplotlib and intentionally avoid Python
//! estimator protocols; callers can hand the exposed arrays to a Rust plotting
//! backend of their choice.

use ferrolearn_core::FerroError;
use ndarray::{Array1, Array2};

use crate::calibration::{CalibrationStrategy, calibration_curve};
use crate::inspection::PartialDependenceResult;
use crate::learning_curve::LearningCurveResult;
use crate::validation_curve::ValidationCurveResult;

/// Data container equivalent to `sklearn.model_selection.LearningCurveDisplay`.
#[derive(Debug, Clone, PartialEq)]
pub struct LearningCurveDisplay {
    /// Absolute training sizes used to generate the curve.
    pub train_sizes: Vec<usize>,
    /// Training scores with shape `(n_ticks, n_cv_folds)`.
    pub train_scores: Array2<f64>,
    /// Test scores with shape `(n_ticks, n_cv_folds)`.
    pub test_scores: Array2<f64>,
    /// Optional score name for renderer labels.
    pub score_name: Option<String>,
}

impl LearningCurveDisplay {
    /// Build a display from already-computed learning-curve arrays.
    #[must_use]
    pub fn new(
        train_sizes: Vec<usize>,
        train_scores: Array2<f64>,
        test_scores: Array2<f64>,
        score_name: Option<String>,
    ) -> Self {
        Self {
            train_sizes,
            train_scores,
            test_scores,
            score_name,
        }
    }

    /// Build a display from a [`LearningCurveResult`].
    #[must_use]
    pub fn from_result(result: LearningCurveResult, score_name: Option<String>) -> Self {
        Self::new(
            result.train_sizes,
            result.train_scores,
            result.test_scores,
            score_name,
        )
    }

    /// Mean training score for each training-size tick.
    #[must_use]
    pub fn train_score_mean(&self) -> Array1<f64> {
        row_mean(&self.train_scores)
    }

    /// Mean test score for each training-size tick.
    #[must_use]
    pub fn test_score_mean(&self) -> Array1<f64> {
        row_mean(&self.test_scores)
    }

    /// Standard deviation of training scores for each training-size tick.
    #[must_use]
    pub fn train_score_std(&self) -> Array1<f64> {
        row_std(&self.train_scores)
    }

    /// Standard deviation of test scores for each training-size tick.
    #[must_use]
    pub fn test_score_std(&self) -> Array1<f64> {
        row_std(&self.test_scores)
    }
}

/// Data container equivalent to `sklearn.model_selection.ValidationCurveDisplay`.
#[derive(Debug, Clone, PartialEq)]
pub struct ValidationCurveDisplay {
    /// Name of the varied parameter.
    pub param_name: String,
    /// Parameter values that were evaluated.
    pub param_range: Array1<f64>,
    /// Training scores with shape `(n_ticks, n_cv_folds)`.
    pub train_scores: Array2<f64>,
    /// Test scores with shape `(n_ticks, n_cv_folds)`.
    pub test_scores: Array2<f64>,
    /// Optional score name for renderer labels.
    pub score_name: Option<String>,
}

impl ValidationCurveDisplay {
    /// Build a display from already-computed validation-curve arrays.
    #[must_use]
    pub fn new(
        param_name: impl Into<String>,
        param_range: Array1<f64>,
        train_scores: Array2<f64>,
        test_scores: Array2<f64>,
        score_name: Option<String>,
    ) -> Self {
        Self {
            param_name: param_name.into(),
            param_range,
            train_scores,
            test_scores,
            score_name,
        }
    }

    /// Build a display from a [`ValidationCurveResult`].
    #[must_use]
    pub fn from_result(
        param_name: impl Into<String>,
        result: ValidationCurveResult,
        score_name: Option<String>,
    ) -> Self {
        Self::new(
            param_name,
            Array1::from_vec(result.param_values),
            result.train_scores,
            result.test_scores,
            score_name,
        )
    }

    /// Mean training score for each parameter tick.
    #[must_use]
    pub fn train_score_mean(&self) -> Array1<f64> {
        row_mean(&self.train_scores)
    }

    /// Mean test score for each parameter tick.
    #[must_use]
    pub fn test_score_mean(&self) -> Array1<f64> {
        row_mean(&self.test_scores)
    }

    /// Standard deviation of training scores for each parameter tick.
    #[must_use]
    pub fn train_score_std(&self) -> Array1<f64> {
        row_std(&self.train_scores)
    }

    /// Standard deviation of test scores for each parameter tick.
    #[must_use]
    pub fn test_score_std(&self) -> Array1<f64> {
        row_std(&self.test_scores)
    }
}

/// Data container equivalent to `sklearn.calibration.CalibrationDisplay`.
#[derive(Debug, Clone, PartialEq)]
pub struct CalibrationDisplay {
    /// Fraction of positives in each non-empty probability bin.
    pub prob_true: Array1<f64>,
    /// Mean predicted probability in each non-empty probability bin.
    pub prob_pred: Array1<f64>,
    /// Probability estimates for the positive class.
    pub y_prob: Array1<f64>,
    /// Optional estimator name for renderer legends.
    pub estimator_name: Option<String>,
    /// Positive class label used to compute the curve.
    pub pos_label: Option<usize>,
}

impl CalibrationDisplay {
    /// Build a display from already-computed calibration arrays.
    #[must_use]
    pub fn new(
        prob_true: Array1<f64>,
        prob_pred: Array1<f64>,
        y_prob: Array1<f64>,
        estimator_name: Option<String>,
        pos_label: Option<usize>,
    ) -> Self {
        Self {
            prob_true,
            prob_pred,
            y_prob,
            estimator_name,
            pos_label,
        }
    }

    /// Compute calibration display data from labels and positive-class probabilities.
    pub fn from_predictions(
        y_true: &Array1<usize>,
        y_prob: &Array1<f64>,
        n_bins: usize,
        strategy: CalibrationStrategy,
        pos_label: Option<usize>,
    ) -> Result<Self, FerroError> {
        let (prob_true, prob_pred) =
            calibration_curve(y_true, y_prob, n_bins, strategy, pos_label)?;
        Ok(Self::new(
            prob_true,
            prob_pred,
            y_prob.clone(),
            None,
            pos_label,
        ))
    }
}

/// Data container equivalent to `sklearn.inspection.DecisionBoundaryDisplay`.
#[derive(Debug, Clone, PartialEq)]
pub struct DecisionBoundaryDisplay {
    /// First meshgrid coordinate.
    pub xx0: Array2<f64>,
    /// Second meshgrid coordinate.
    pub xx1: Array2<f64>,
    /// Response values over the mesh.
    pub response: Array2<f64>,
    /// Expected number of classes represented by `response`.
    pub n_classes: usize,
    /// Optional color labels for multiclass rendering.
    pub multiclass_colors: Option<Vec<String>>,
    /// Optional x-axis label.
    pub xlabel: Option<String>,
    /// Optional y-axis label.
    pub ylabel: Option<String>,
}

impl DecisionBoundaryDisplay {
    /// Build a decision-boundary display, validating mesh and response shapes.
    pub fn new(
        xx0: Array2<f64>,
        xx1: Array2<f64>,
        response: Array2<f64>,
        n_classes: usize,
        multiclass_colors: Option<Vec<String>>,
        xlabel: Option<String>,
        ylabel: Option<String>,
    ) -> Result<Self, FerroError> {
        if xx0.dim() != xx1.dim() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![xx0.nrows(), xx0.ncols()],
                actual: vec![xx1.nrows(), xx1.ncols()],
                context: "DecisionBoundaryDisplay: xx0 vs xx1".into(),
            });
        }
        if response.dim() != xx0.dim() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![xx0.nrows(), xx0.ncols()],
                actual: vec![response.nrows(), response.ncols()],
                context: "DecisionBoundaryDisplay: mesh vs response".into(),
            });
        }
        if n_classes < 2 {
            return Err(FerroError::InvalidParameter {
                name: "n_classes".into(),
                reason: "must be >= 2".into(),
            });
        }
        Ok(Self {
            xx0,
            xx1,
            response,
            n_classes,
            multiclass_colors,
            xlabel,
            ylabel,
        })
    }
}

/// Display kind for [`PartialDependenceDisplay`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PartialDependenceKind {
    /// Plot averaged partial dependence.
    Average,
    /// Plot individual conditional expectation data.
    Individual,
    /// Plot both averaged partial dependence and ICE data.
    Both,
}

/// Data container equivalent to `sklearn.inspection.PartialDependenceDisplay`.
#[derive(Debug, Clone)]
pub struct PartialDependenceDisplay {
    /// Partial dependence results, one entry per requested feature set.
    pub pd_results: Vec<PartialDependenceResult>,
    /// Feature index groups corresponding to `pd_results`.
    pub features: Vec<Vec<usize>>,
    /// Feature names.
    pub feature_names: Vec<String>,
    /// Target index for multiclass or multioutput displays.
    pub target_idx: usize,
    /// Feature deciles available to a renderer.
    pub deciles: Vec<(usize, Array1<f64>)>,
    /// Display kind.
    pub kind: PartialDependenceKind,
    /// Optional ICE subsample cap.
    pub subsample: Option<usize>,
    /// Optional random-state seed for renderer-side sampling.
    pub random_state: Option<u64>,
    /// Optional categorical-feature mask.
    pub is_categorical: Option<Vec<bool>>,
}

impl PartialDependenceDisplay {
    /// Build a partial-dependence display from precomputed result data.
    #[allow(
        clippy::too_many_arguments,
        reason = "mirrors sklearn's display constructor attributes"
    )]
    pub fn new(
        pd_results: Vec<PartialDependenceResult>,
        features: Vec<Vec<usize>>,
        feature_names: Vec<String>,
        target_idx: usize,
        deciles: Vec<(usize, Array1<f64>)>,
        kind: PartialDependenceKind,
        subsample: Option<usize>,
        random_state: Option<u64>,
        is_categorical: Option<Vec<bool>>,
    ) -> Result<Self, FerroError> {
        if pd_results.len() != features.len() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![pd_results.len()],
                actual: vec![features.len()],
                context: "PartialDependenceDisplay: pd_results vs features".into(),
            });
        }
        if pd_results.is_empty() {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "PartialDependenceDisplay".into(),
            });
        }
        Ok(Self {
            pd_results,
            features,
            feature_names,
            target_idx,
            deciles,
            kind,
            subsample,
            random_state,
            is_categorical,
        })
    }

    /// Build a one-feature average partial-dependence display.
    pub fn from_single_feature(
        result: PartialDependenceResult,
        feature_idx: usize,
        feature_name: impl Into<String>,
    ) -> Result<Self, FerroError> {
        Self::new(
            vec![result],
            vec![vec![feature_idx]],
            vec![feature_name.into()],
            0,
            Vec::new(),
            PartialDependenceKind::Average,
            None,
            None,
            None,
        )
    }
}

fn row_mean(matrix: &Array2<f64>) -> Array1<f64> {
    let n_cols = matrix.ncols() as f64;
    Array1::from_iter(matrix.rows().into_iter().map(|row| {
        if row.is_empty() {
            f64::NAN
        } else {
            row.iter().sum::<f64>() / n_cols
        }
    }))
}

fn row_std(matrix: &Array2<f64>) -> Array1<f64> {
    let means = row_mean(matrix);
    let n_cols = matrix.ncols() as f64;
    Array1::from_iter(
        matrix
            .rows()
            .into_iter()
            .zip(means.iter())
            .map(|(row, &mean)| {
                if row.is_empty() {
                    f64::NAN
                } else {
                    let var = row.iter().map(|value| (value - mean).powi(2)).sum::<f64>() / n_cols;
                    var.sqrt()
                }
            }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn learning_curve_display_summarizes_rows() {
        let display = LearningCurveDisplay::new(
            vec![2, 4],
            array![[1.0, 3.0], [2.0, 4.0]],
            array![[0.0, 2.0], [1.0, 5.0]],
            Some("Score".into()),
        );
        assert_eq!(display.train_score_mean(), array![2.0, 3.0]);
        assert_eq!(display.test_score_mean(), array![1.0, 3.0]);
        assert_eq!(display.train_score_std(), array![1.0, 1.0]);
    }

    #[test]
    fn calibration_display_from_predictions() {
        let y_true = array![0usize, 0, 0, 0, 1, 1, 1, 1, 1];
        let y_prob = array![0.1, 0.2, 0.3, 0.4, 0.65, 0.7, 0.8, 0.9, 1.0];
        let display = CalibrationDisplay::from_predictions(
            &y_true,
            &y_prob,
            3,
            CalibrationStrategy::Uniform,
            None,
        )
        .unwrap();
        for (actual, expected) in display.prob_true.iter().zip(array![0.0, 0.5, 1.0].iter()) {
            assert_relative_eq!(*actual, *expected, epsilon = 1e-12);
        }
        for (actual, expected) in display
            .prob_pred
            .iter()
            .zip(array![0.2, 0.525, 0.85].iter())
        {
            assert_relative_eq!(*actual, *expected, epsilon = 1e-12);
        }
    }

    #[test]
    fn decision_boundary_display_validates_shapes() {
        let xx0 = array![[0.0, 1.0], [0.0, 1.0]];
        let xx1 = array![[0.0, 0.0], [1.0, 1.0]];
        let response = array![[0.0, 1.0], [1.0, 0.0]];
        let display =
            DecisionBoundaryDisplay::new(xx0, xx1, response, 2, None, None, None).unwrap();
        assert_eq!(display.response.dim(), (2, 2));
    }

    #[test]
    fn partial_dependence_display_requires_matching_features() {
        let result = PartialDependenceResult {
            grid: array![0.0, 1.0],
            averaged_predictions: array![2.0, 3.0],
        };
        let display = PartialDependenceDisplay::from_single_feature(result, 0, "x0").unwrap();
        assert_eq!(display.features, vec![vec![0]]);
        assert_eq!(display.feature_names, vec!["x0".to_string()]);
    }
}
