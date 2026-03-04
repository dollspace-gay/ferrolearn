/// Calling `predict` on an unfitted `LinearRegression` must be a compile error.
/// `LinearRegression` implements `Fit` but NOT `Predict`.
use ferrolearn_core::Predict;
use ferrolearn_linear::LinearRegression;
use ndarray::Array2;

fn main() {
    let model = LinearRegression::<f64>::new();
    let x = Array2::<f64>::zeros((3, 2));
    // This should fail: LinearRegression doesn't implement Predict
    let _ = model.predict(&x);
}
