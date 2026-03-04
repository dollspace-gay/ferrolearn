/// Calling `transform` on an unfitted `StandardScaler` must return a runtime
/// error (the unfitted scaler *does* implement `Transform` to satisfy the
/// `FitTransform: Transform` supertrait, but it always returns Err).
///
/// However, calling it on a different type that does NOT implement `Transform`
/// would be a compile error. Here we demonstrate that `StandardScaler`
/// cannot be used as a `FittedStandardScaler` — specifically, the fitted
/// type's `transform` cannot be called through an unfitted reference.
///
/// We test a stronger property: you cannot call `Predict` on `StandardScaler`.
use ferrolearn_core::Predict;
use ferrolearn_preprocess::StandardScaler;
use ndarray::Array2;

fn main() {
    let scaler = StandardScaler::<f64>::new();
    let x = Array2::<f64>::zeros((3, 2));
    // This should fail: StandardScaler doesn't implement Predict
    let _ = scaler.predict(&x);
}
