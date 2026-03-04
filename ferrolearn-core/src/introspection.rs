//! Introspection traits for fitted models.
//!
//! These traits allow downstream code to inspect the internal state of
//! fitted models (coefficients, feature importances, class labels) in
//! a uniform way, enabling generic model-inspection utilities.

use ndarray::Array1;

/// A fitted model that exposes linear coefficients and an intercept.
///
/// Implemented by linear models such as `FittedLinearRegression`,
/// `FittedLogisticRegression`, `FittedRidge`, etc.
///
/// # Type Parameters
///
/// - `F`: The floating-point type (e.g., `f64`).
pub trait HasCoefficients<F> {
    /// Returns a reference to the learned coefficient vector.
    fn coefficients(&self) -> &Array1<F>;

    /// Returns the learned intercept (bias) term.
    fn intercept(&self) -> F;
}

/// A fitted model that exposes per-feature importance scores.
///
/// Implemented by tree-based models such as `FittedDecisionTree`,
/// `FittedRandomForest`, `FittedGradientBoosting`, etc.
///
/// # Type Parameters
///
/// - `F`: The floating-point type (e.g., `f64`).
pub trait HasFeatureImportances<F> {
    /// Returns a reference to the feature importance array.
    ///
    /// Importances are non-negative and typically sum to 1.0.
    fn feature_importances(&self) -> &Array1<F>;
}

/// A fitted classifier that knows the set of classes it was trained on.
///
/// Implemented by all classifiers after fitting, to allow introspection
/// of the label space.
pub trait HasClasses {
    /// Returns the sorted list of unique class labels.
    fn classes(&self) -> &[usize];

    /// Returns the number of distinct classes.
    fn n_classes(&self) -> usize;
}
