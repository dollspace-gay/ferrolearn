//! Introspection traits for fitted models.
//!
//! These traits allow downstream code to inspect the internal state of
//! fitted models (coefficients, feature importances, class labels) in
//! a uniform way, enabling generic model-inspection utilities.
//!
//! ## REQ status (per `.design/core/introspection.md`, mirrors sklearn fitted attrs @ 1.5.2)
//!
//! These traits encode scikit-learn's trailing-underscore fitted-attribute
//! convention (`coef_`/`intercept_`/`feature_importances_`/`classes_`) as
//! compile-time traits implemented on `Fitted*` types across the workspace.
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (HasCoefficients ↔ coef_/intercept_) | SHIPPED | `HasCoefficients` trait; consumers `impl HasCoefficients for FittedLinearRegression in linear_regression.rs` (+ `FittedRidge`, `FittedLogisticRegression`). Mirrors `sklearn/linear_model/_base.py:691-692`. |
//! | REQ-2 (HasFeatureImportances ↔ feature_importances_) | SHIPPED | `HasFeatureImportances` trait; consumers `impl ... for FittedDecisionTreeClassifier in decision_tree.rs` (+ random_forest, gradient_boosting). Mirrors `sklearn/tree/_classes.py:671`. |
//! | REQ-3 (HasClasses ↔ classes_/n_classes_) | SHIPPED | `HasClasses` trait; consumers `impl ... for FittedLogisticRegression in logistic_regression.rs` (+ tree/bayes/neighbors classifiers). Mirrors `classes_ = np.unique(y)` at `sklearn/linear_model/_logistic.py:1232`. |
//! | REQ-4 (ferray substrate for return types) | NOT-STARTED | open blocker #359 — `coefficients()`/`feature_importances()` return `&ndarray::Array1<F>`; migrating to ferray `Array1` cascades through every implementing estimator (R-SUBSTRATE-4, grandfathered-transitional). |
//!
//! acto-critic audit: NO DIVERGENCE FOUND (pure trait definitions; the multi-output
//! `coef_`/`intercept_` shape and `classes_` label-value contracts are observable only
//! through implementing estimators and are pinned when the loop audits linear/tree/bayes/
//! neighbors). Two states only per goal.md R-DEFER-2.

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
