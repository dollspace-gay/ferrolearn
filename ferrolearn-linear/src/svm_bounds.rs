//! Parameter-bound helpers for linear L1 classifiers.
//!
//! Mirrors `sklearn.svm._bounds.l1_min_c`: compute the smallest `C` above
//! which an L1-penalized linear classifier is guaranteed to have at least one
//! non-zero coefficient or intercept for the given data.
//!
//! ## REQ status
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (`l1_min_c` formula) | SHIPPED | `l1_min_c` computes `den = max(abs(LabelBinarizer(neg_label=-1).fit_transform(y).T @ X))`, includes the synthetic intercept column when requested, and returns `0.5 / den` for `squared_hinge` or `2.0 / den` for `log`, matching `sklearn/svm/_bounds.py:80-99`. Oracle tests live in `tests/divergence_svm_bounds.rs`. |
//! | REQ-2 (validation) | SHIPPED | rejects inconsistent lengths, empty inputs, non-positive `intercept_scaling`, non-finite `X`, and ill-posed `den == 0`, mirroring sklearn's `check_array`, `check_consistent_length`, parameter validation, and `ValueError` branch. |
//! | REQ-substrate | NOT-STARTED | helper accepts `ndarray::Array2`/`Array1`, not ferray arrays. |

use ferrolearn_core::error::FerroError;
use ndarray::{Array1, Array2};
use num_traits::Float;

/// Loss selector for [`l1_min_c`].
///
/// Mirrors sklearn's `loss={"squared_hinge", "log"}` parameter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum L1MinCLoss {
    /// Squared hinge loss, the default used by `LinearSVC(loss="squared_hinge")`.
    #[default]
    SquaredHinge,
    /// Logistic-regression loss.
    Log,
}

/// Return the lowest useful `C` for L1-penalized linear classifiers.
///
/// This is the Rust analogue of `sklearn.svm.l1_min_c(X, y, loss=...,
/// fit_intercept=..., intercept_scaling=...)`.
///
/// # Errors
///
/// Returns [`FerroError::ShapeMismatch`] when `X.nrows() != y.len()`.
/// Returns [`FerroError::InvalidParameter`] for empty feature matrices,
/// non-finite `X`, non-positive `intercept_scaling`, or an ill-posed all-zero
/// bound.
pub fn l1_min_c<F>(
    x: &Array2<F>,
    y: &Array1<usize>,
    loss: L1MinCLoss,
    fit_intercept: bool,
    intercept_scaling: F,
) -> Result<F, FerroError>
where
    F: Float,
{
    validate_l1_min_c_input(x, y, intercept_scaling)?;

    let rows = label_binarizer_rows(y);
    let mut den = F::zero();

    for label_row in &rows {
        for feature in 0..x.ncols() {
            let mut dot = F::zero();
            for sample in 0..x.nrows() {
                dot = dot + label_row[sample] * x[[sample, feature]];
            }
            den = den.max(dot.abs());
        }

        if fit_intercept {
            let bias_dot = label_row
                .iter()
                .copied()
                .fold(F::zero(), |acc, value| acc + value)
                * intercept_scaling;
            den = den.max(bias_dot.abs());
        }
    }

    if den == F::zero() {
        return Err(FerroError::InvalidParameter {
            name: "X".into(),
            reason: "Ill-posed l1_min_c calculation: l1 will always select zero coefficients for this data".into(),
        });
    }

    match loss {
        L1MinCLoss::SquaredHinge => Ok(cst::<F>(0.5) / den),
        L1MinCLoss::Log => Ok(cst::<F>(2.0) / den),
    }
}

fn validate_l1_min_c_input<F>(
    x: &Array2<F>,
    y: &Array1<usize>,
    intercept_scaling: F,
) -> Result<(), FerroError>
where
    F: Float,
{
    if x.nrows() != y.len() {
        return Err(FerroError::ShapeMismatch {
            expected: vec![x.nrows()],
            actual: vec![y.len()],
            context: "l1_min_c requires X and y to have consistent lengths".into(),
        });
    }
    if x.nrows() == 0 {
        return Err(FerroError::InsufficientSamples {
            required: 1,
            actual: 0,
            context: "l1_min_c".into(),
        });
    }
    if x.ncols() == 0 {
        return Err(FerroError::InvalidParameter {
            name: "X".into(),
            reason: "l1_min_c requires at least one feature".into(),
        });
    }
    if intercept_scaling <= F::zero() {
        return Err(FerroError::InvalidParameter {
            name: "intercept_scaling".into(),
            reason: "must be strictly positive".into(),
        });
    }
    if x.iter().any(|value| !value.is_finite()) {
        return Err(FerroError::InvalidParameter {
            name: "X".into(),
            reason: "Input X contains NaN or infinity.".into(),
        });
    }
    Ok(())
}

fn label_binarizer_rows<F>(y: &Array1<usize>) -> Vec<Vec<F>>
where
    F: Float,
{
    let mut classes = y.to_vec();
    classes.sort_unstable();
    classes.dedup();

    if classes.len() <= 2 {
        let positive = classes.get(1).copied();
        vec![
            y.iter()
                .map(|&label| {
                    if positive.is_some_and(|class| label == class) {
                        F::one()
                    } else {
                        -F::one()
                    }
                })
                .collect(),
        ]
    } else {
        classes
            .iter()
            .map(|&class| {
                y.iter()
                    .map(|&label| if label == class { F::one() } else { -F::one() })
                    .collect()
            })
            .collect()
    }
}

fn cst<F: Float>(value: f64) -> F {
    F::from(value).unwrap_or_else(F::zero)
}
