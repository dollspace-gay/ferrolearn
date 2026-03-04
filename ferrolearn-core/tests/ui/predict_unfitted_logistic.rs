/// Calling `predict` on an unfitted `LogisticRegression` must be a compile error.
/// `LogisticRegression` implements `Fit` but NOT `Predict`.
use ferrolearn_core::Predict;
use ferrolearn_linear::LogisticRegression;
use ndarray::Array2;

fn main() {
    let model = LogisticRegression::<f64>::new();
    let x = Array2::<f64>::zeros((3, 2));
    // This should fail: LogisticRegression doesn't implement Predict
    let _ = model.predict(&x);
}
