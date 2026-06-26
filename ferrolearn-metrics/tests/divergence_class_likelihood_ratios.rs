//! Oracle pins for `class_likelihood_ratios` vs scikit-learn.
//!
//! Expected values mirror sklearn's `metrics._classification.class_likelihood_ratios`
//! examples and tests. The metric is binary-only, with optional
//! `[negative, positive]` labels, optional sample weights, and configurable
//! replacement values for undefined ratios.

use ferrolearn_metrics::classification::{
    ClassLikelihoodUndefined, class_likelihood_ratios, class_likelihood_ratios_with_options,
};
use ndarray::{Array1, array};

fn lab(values: &[usize]) -> Array1<usize> {
    Array1::from(values.to_vec())
}

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1e-12,
        "actual={actual}, expected={expected}"
    );
}

#[test]
fn likelihood_ratios_match_sklearn_oracle_and_weighted_path() {
    // sklearn test fixture:
    // tn=9, fp=8, fn=1, tp=2 -> LR+=34/24, LR-=17/27.
    let mut y_true_values = vec![1usize; 3];
    y_true_values.extend(std::iter::repeat_n(0usize, 17));
    let y_true = Array1::from(y_true_values);

    let mut y_pred_values = vec![1usize; 2];
    y_pred_values.extend(std::iter::repeat_n(0usize, 10));
    y_pred_values.extend(std::iter::repeat_n(1usize, 8));
    let y_pred = Array1::from(y_pred_values);

    let (lr_pos, lr_neg) = class_likelihood_ratios(&y_true, &y_pred).unwrap();
    assert_close(lr_pos, 34.0 / 24.0);
    assert_close(lr_neg, 17.0 / 27.0);

    let (perfect_pos, perfect_neg) = class_likelihood_ratios(&y_true, &y_true).unwrap();
    assert!(perfect_pos.is_nan(), "perfect-classifier LR+ should be NaN");
    assert_close(perfect_neg, 0.0);

    // Ignore the last five false positives: tn=9, fp=3, fn=1, tp=2.
    let mut sample_weight_values = vec![1.0_f64; 15];
    sample_weight_values.extend(std::iter::repeat_n(0.0_f64, 5));
    let sample_weight = Array1::from(sample_weight_values);
    let (weighted_pos, weighted_neg) = class_likelihood_ratios_with_options(
        &y_true,
        &y_pred,
        None,
        Some(&sample_weight),
        ClassLikelihoodUndefined::Nan,
    )
    .unwrap();
    assert_close(weighted_pos, 24.0 / 9.0);
    assert_close(weighted_neg, 12.0 / 27.0);
}

#[test]
fn likelihood_ratios_support_explicit_labels_and_non_contiguous_classes() {
    // Same values as sklearn's public example, but using usize labels 2/3.
    let y_true = lab(&[2, 3, 2, 3, 2]);
    let y_pred = lab(&[3, 3, 2, 2, 2]);

    let (lr_pos, lr_neg) = class_likelihood_ratios_with_options(
        &y_true,
        &y_pred,
        Some([2, 3]),
        None,
        ClassLikelihoodUndefined::Nan,
    )
    .unwrap();

    assert_close(lr_pos, 1.5);
    assert_close(lr_neg, 0.75);
}

#[test]
fn likelihood_ratios_replacement_policies_match_sklearn_edges() {
    // fp=0 makes LR+ undefined; LR- is still computed as 0.5.
    let y_true = lab(&[1, 1, 0]);
    let y_pred = lab(&[1, 0, 0]);
    let (lr_pos, lr_neg) = class_likelihood_ratios_with_options(
        &y_true,
        &y_pred,
        None,
        None,
        ClassLikelihoodUndefined::Worst,
    )
    .unwrap();
    assert_close(lr_pos, 1.0);
    assert_close(lr_neg, 0.5);

    let (lr_pos_inf, _) = class_likelihood_ratios_with_options(
        &y_true,
        &y_pred,
        None,
        None,
        ClassLikelihoodUndefined::Values {
            lr_positive: f64::INFINITY,
            lr_negative: 0.0,
        },
    )
    .unwrap();
    assert!(lr_pos_inf.is_infinite() && lr_pos_inf.is_sign_positive());

    // tn=0 makes LR- undefined; LR+ is still computed as 1.0.
    let y_true = lab(&[1, 0, 0]);
    let y_pred = lab(&[1, 1, 1]);
    let (lr_pos, lr_neg) = class_likelihood_ratios_with_options(
        &y_true,
        &y_pred,
        None,
        None,
        ClassLikelihoodUndefined::Values {
            lr_positive: 2.0,
            lr_negative: 0.5,
        },
    )
    .unwrap();
    assert_close(lr_pos, 1.0);
    assert_close(lr_neg, 0.5);
}

#[test]
fn likelihood_ratios_reject_non_binary_and_invalid_options() {
    let multiclass_y_true = lab(&[0, 1, 0, 1, 0]);
    let multiclass_y_pred = lab(&[1, 1, 0, 0, 2]);
    assert!(class_likelihood_ratios(&multiclass_y_true, &multiclass_y_pred).is_err());

    let y_true = array![1usize, 0];
    let y_pred = array![1usize, 0];
    let bad_replacement = class_likelihood_ratios_with_options(
        &y_true,
        &y_pred,
        None,
        None,
        ClassLikelihoodUndefined::Values {
            lr_positive: 0.0,
            lr_negative: 2.0,
        },
    );
    assert!(bad_replacement.is_err());

    let bad_weights = array![1.0_f64];
    let bad_weight_len = class_likelihood_ratios_with_options(
        &y_true,
        &y_pred,
        None,
        Some(&bad_weights),
        ClassLikelihoodUndefined::Nan,
    );
    assert!(bad_weight_len.is_err());
}
