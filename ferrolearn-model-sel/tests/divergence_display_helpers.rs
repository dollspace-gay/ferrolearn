//! Green guards for sklearn display-class data containers.
//!
//! These tests pin the data that sklearn display classes store before handing
//! off to matplotlib. ferrolearn intentionally stops at renderer-neutral data:
//! no matplotlib axes/artists, no `from_estimator` classmethods, and no Python
//! estimator protocol.

use approx::assert_relative_eq;
use ferrolearn_core::FerroError;
use ferrolearn_model_sel::{
    CalibrationDisplay, CalibrationStrategy, DecisionBoundaryDisplay, LearningCurveDisplay,
    PartialDependenceDisplay, PartialDependenceKind, ValidationCurveDisplay, calibration_curve,
    partial_dependence,
};
use ndarray::{Array1, Array2, array};

#[test]
fn green_calibration_display_from_predictions_matches_sklearn_doc_example() {
    // sklearn 1.9 local mirror doc example:
    // y_true=[0,0,0,0,1,1,1,1,1], y_prob=[.1,.2,.3,.4,.65,.7,.8,.9,1]
    // calibration_curve(..., n_bins=3) -> prob_true=[0,.5,1],
    // prob_pred=[.2,.525,.85].
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

    let expected_true = array![0.0, 0.5, 1.0];
    let expected_pred = array![0.2, 0.525, 0.85];
    for (actual, expected) in display.prob_true.iter().zip(expected_true.iter()) {
        assert_relative_eq!(*actual, *expected, epsilon = 1e-12);
    }
    for (actual, expected) in display.prob_pred.iter().zip(expected_pred.iter()) {
        assert_relative_eq!(*actual, *expected, epsilon = 1e-12);
    }
    assert_eq!(display.y_prob, y_prob);
}

#[test]
fn green_calibration_curve_quantile_matches_sklearn_oracle() {
    // Live sklearn 1.5.2 oracle:
    // calibration_curve(y, p, n_bins=3, strategy="quantile") ->
    // prob_true=[1/3,.5,2/3], prob_pred=[.07333333333333333,.125,.8833333333333334].
    let y_true = array![0usize, 0, 1, 1, 0, 1, 0, 1];
    let y_prob = array![0.01, 0.1, 0.11, 0.12, 0.13, 0.8, 0.9, 0.95];

    let (prob_true, prob_pred) =
        calibration_curve(&y_true, &y_prob, 3, CalibrationStrategy::Quantile, None).unwrap();

    let expected_true = array![1.0 / 3.0, 0.5, 2.0 / 3.0];
    let expected_pred = array![0.073_333_333_333_333_33, 0.125, 0.883_333_333_333_333_4];
    for (actual, expected) in prob_true.iter().zip(expected_true.iter()) {
        assert_relative_eq!(*actual, *expected, epsilon = 1e-12);
    }
    for (actual, expected) in prob_pred.iter().zip(expected_pred.iter()) {
        assert_relative_eq!(*actual, *expected, epsilon = 1e-12);
    }
}

#[test]
fn green_curve_displays_store_plot_inputs_and_row_statistics() {
    // sklearn LearningCurveDisplay/ValidationCurveDisplay store tick values and
    // `(n_ticks, n_cv_folds)` score matrices; plot summaries are row-wise
    // means/stds over folds.
    let learning = LearningCurveDisplay::new(
        vec![5, 10],
        array![[1.0, 3.0], [2.0, 4.0]],
        array![[0.0, 2.0], [1.0, 5.0]],
        Some("Score".into()),
    );
    assert_eq!(learning.train_sizes, vec![5, 10]);
    assert_eq!(learning.train_score_mean(), array![2.0, 3.0]);
    assert_eq!(learning.test_score_mean(), array![1.0, 3.0]);
    assert_eq!(learning.test_score_std(), array![1.0, 2.0]);

    let validation = ValidationCurveDisplay::new(
        "alpha",
        array![0.1, 1.0],
        array![[1.0, 1.0], [0.0, 2.0]],
        array![[0.5, 0.7], [0.2, 0.4]],
        Some("Score".into()),
    );
    assert_eq!(validation.param_name, "alpha");
    assert_eq!(validation.param_range, array![0.1, 1.0]);
    assert_eq!(validation.train_score_mean(), array![1.0, 1.0]);
    let expected_test_mean = array![0.6, 0.3];
    for (actual, expected) in validation
        .test_score_mean()
        .iter()
        .zip(expected_test_mean.iter())
    {
        assert_relative_eq!(*actual, *expected, epsilon = 1e-12);
    }
}

#[test]
fn green_inspection_displays_store_mesh_and_partial_dependence_data() {
    // DecisionBoundaryDisplay stores meshgrid outputs and the shaped response.
    let xx0 = array![[0.0, 1.0], [0.0, 1.0]];
    let xx1 = array![[0.0, 0.0], [1.0, 1.0]];
    let response = array![[0.0, 1.0], [1.0, 0.0]];
    let boundary = DecisionBoundaryDisplay::new(
        xx0.clone(),
        xx1.clone(),
        response.clone(),
        2,
        None,
        Some("x0".into()),
        Some("x1".into()),
    )
    .unwrap();
    assert_eq!(boundary.xx0, xx0);
    assert_eq!(boundary.xx1, xx1);
    assert_eq!(boundary.response, response);
    assert_eq!(boundary.xlabel.as_deref(), Some("x0"));

    // Live sklearn 1.5.2 brute-PD oracle from divergence_inspection.rs:
    // predict(X)=X*[0.5,0.5]-1 over grid [0,1,2,10] -> [1.5,2,2.5,6.5].
    let predict = |x: &Array2<f64>| -> Result<Array1<f64>, FerroError> {
        let mut out = Array1::<f64>::zeros(x.nrows());
        for row in 0..x.nrows() {
            out[row] = 0.5 * x[[row, 0]] + 0.5 * x[[row, 1]] - 1.0;
        }
        Ok(out)
    };
    let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
    let grid = array![0.0, 1.0, 2.0, 10.0];
    let pd = partial_dependence(predict, &x, 0, &grid).unwrap();
    let display = PartialDependenceDisplay::from_single_feature(pd, 0, "x0").unwrap();

    assert_eq!(display.kind, PartialDependenceKind::Average);
    assert_eq!(display.features, vec![vec![0]]);
    let expected = array![1.5, 2.0, 2.5, 6.5];
    for (actual, expected) in display.pd_results[0]
        .averaged_predictions
        .iter()
        .zip(expected.iter())
    {
        assert_relative_eq!(*actual, *expected, epsilon = 1e-12);
    }
}
