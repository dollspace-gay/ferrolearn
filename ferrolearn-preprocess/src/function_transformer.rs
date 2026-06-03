//! Function transformer: apply a user-provided function element-wise.
//!
//! Wraps any `Fn(F) -> F` callable and applies it to every element in the
//! input matrix. This is useful for applying non-standard transformations
//! such as `ln`, `sqrt`, or custom domain-specific functions.
//!
//! This transformer is **stateless** — no fitting is required. Call
//! [`Transform::transform`] directly.
//!
//! # `## REQ status`
//!
//! Binary (R-DEFER-2), translating `sklearn/preprocessing/_function_transformer.py`
//! (`class FunctionTransformer(TransformerMixin, BaseEstimator)`). Design doc:
//! `.design/preprocess/function_transformer.md`. Expected values from the live sklearn 1.5.2
//! oracle (R-CHAR-3). HONEST (R-HONEST-3): ferrolearn ships a THIN element-wise wrapper —
//! `func` is a scalar `Fn(F) -> F` applied via `mapv`, NOT sklearn's whole-array `func(X)`;
//! matches sklearn only on the element-wise/ufunc subset. Consumer: crate re-export
//! (`lib.rs`, grandfathered S5).
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (element-wise forward transform) | SHIPPED (scoped) | `Transform::transform` = `x.mapv(\|v\| (self.func)(v))`, shape-preserving, infallible; mirrors sklearn `_transform` (`_function_transformer.py:375-379`) for element-wise ufunc `func`. Critic-verified bit-identical to live sklearn: `guard_log1p_/expm1_/sqrt_/log_nan_inf_/empty_matrix_*` (5 green) in `tests/divergence_function_transformer.rs`. Consumer: `pub use function_transformer::FunctionTransformer` (`lib.rs:114`). Caveat: scalar `Fn(F)->F`, not array `Fn(X)->X`. |
//! | REQ-2 (func=None identity default) | NOT-STARTED | open prereq blocker #1112. `new` requires a closure; no identity default (`_identity`, `:22-24`). |
//! | REQ-3 (whole-array func, headline) | NOT-STARTED | open prereq blocker #1113. `Box<dyn Fn(F)->F>` cannot read the array / change shape; sklearn `func(X)` is array→array (`:375-379`). |
//! | REQ-4 (inverse_func / inverse_transform) | NOT-STARTED | open prereq blocker #1114. No inverse path (sklearn `:309-325`). |
//! | REQ-5 (validate / accept_sparse) | NOT-STARTED | open prereq blocker #1115. No input validation (sklearn `:173-182`). |
//! | REQ-6 (fit / check_inverse / is_fitted) | NOT-STARTED | open prereq blocker #1116. No fit/check_inverse (sklearn `:213-235`, `:184-210`). |
//! | REQ-7 (feature_names_out / n_features_in_) | NOT-STARTED | open prereq blocker #1117. None (sklearn `:327-373`). |
//! | REQ-8 (kw_args / inv_kw_args) | NOT-STARTED | open prereq blocker #1118. No kwarg forwarding (sklearn `:93-101`,`:379`). |
//! | REQ-9 (ctor surface + _parameter_constraints) | NOT-STARTED | open prereq blocker #1119. Only `func`; 7 params + validation missing (R-DEV-2, sklearn `:141-171`). |
//! | REQ-10 (PyO3 binding) | NOT-STARTED | open prereq blocker #1120. No `ferrolearn-python` registration. |
//! | REQ-11 (ferray substrate) | NOT-STARTED | open prereq blocker #1121. `ndarray`/`num_traits`, not `ferray-core`/`ferray-ufunc` (R-SUBSTRATE-1/2). |

use ferrolearn_core::error::FerroError;
use ferrolearn_core::traits::Transform;
use ndarray::Array2;
use num_traits::Float;

// ---------------------------------------------------------------------------
// FunctionTransformer
// ---------------------------------------------------------------------------

/// A stateless element-wise function transformer.
///
/// Wraps a boxed `Fn(F) -> F` closure and applies it to every element in
/// the input matrix.
///
/// # Examples
///
/// ```
/// use ferrolearn_preprocess::function_transformer::FunctionTransformer;
/// use ferrolearn_core::traits::Transform;
/// use ndarray::array;
///
/// // Apply natural logarithm element-wise (values must be > 0)
/// let ft = FunctionTransformer::<f64>::new(|v| v.ln());
/// let x = array![[1.0, 2.0], [3.0, 4.0]];
/// let out = ft.transform(&x).unwrap();
/// ```
pub struct FunctionTransformer<F> {
    func: Box<dyn Fn(F) -> F + Send + Sync>,
}

impl<F: Float + Send + Sync + 'static> FunctionTransformer<F> {
    /// Create a new `FunctionTransformer` with the given function.
    ///
    /// The function will be applied element-wise to the input matrix.
    pub fn new<Func>(func: Func) -> Self
    where
        Func: Fn(F) -> F + Send + Sync + 'static,
    {
        Self {
            func: Box::new(func),
        }
    }
}

impl<F: Float + Send + Sync + 'static> std::fmt::Debug for FunctionTransformer<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FunctionTransformer")
            .field("func", &"<fn(F) -> F>")
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Trait implementations
// ---------------------------------------------------------------------------

impl<F: Float + Send + Sync + 'static> Transform<Array2<F>> for FunctionTransformer<F> {
    type Output = Array2<F>;
    type Error = FerroError;

    /// Apply the stored function to every element of `x`.
    ///
    /// # Errors
    ///
    /// This implementation never returns an error for well-formed inputs.
    /// Note: if the user-provided function produces NaN or infinity for
    /// certain inputs, those values will appear in the output without error.
    fn transform(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let out = x.mapv(|v| (self.func)(v));
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_identity_function() {
        let ft = FunctionTransformer::<f64>::new(|v| v);
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let out = ft.transform(&x).unwrap();
        for (a, b) in x.iter().zip(out.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-15);
        }
    }

    #[test]
    fn test_sqrt_function() {
        let ft = FunctionTransformer::<f64>::new(|v: f64| v.sqrt());
        let x = array![[1.0, 4.0], [9.0, 16.0]];
        let out = ft.transform(&x).unwrap();
        assert_abs_diff_eq!(out[[0, 0]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(out[[0, 1]], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(out[[1, 0]], 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(out[[1, 1]], 4.0, epsilon = 1e-10);
    }

    #[test]
    fn test_ln_function() {
        let ft = FunctionTransformer::<f64>::new(|v: f64| v.ln());
        let x = array![[1.0, 2.0]];
        let out = ft.transform(&x).unwrap();
        assert_abs_diff_eq!(out[[0, 0]], 0.0, epsilon = 1e-10); // ln(1) = 0
        assert_abs_diff_eq!(out[[0, 1]], 2.0_f64.ln(), epsilon = 1e-10);
    }

    #[test]
    fn test_negate_function() {
        let ft = FunctionTransformer::<f64>::new(|v| -v);
        let x = array![[1.0, -2.0, 3.0]];
        let out = ft.transform(&x).unwrap();
        assert_abs_diff_eq!(out[[0, 0]], -1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(out[[0, 1]], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(out[[0, 2]], -3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_constant_function() {
        let ft = FunctionTransformer::<f64>::new(|_| 42.0);
        let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let out = ft.transform(&x).unwrap();
        for v in &out {
            assert_abs_diff_eq!(*v, 42.0, epsilon = 1e-15);
        }
    }

    #[test]
    fn test_preserves_shape() {
        let ft = FunctionTransformer::<f64>::new(|v| v * 2.0);
        let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let out = ft.transform(&x).unwrap();
        assert_eq!(out.shape(), x.shape());
    }

    #[test]
    fn test_clamp_function() {
        let ft = FunctionTransformer::<f64>::new(|v: f64| v.clamp(0.0, 1.0));
        let x = array![[-1.0, 0.5, 2.0]];
        let out = ft.transform(&x).unwrap();
        assert_abs_diff_eq!(out[[0, 0]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(out[[0, 1]], 0.5, epsilon = 1e-10);
        assert_abs_diff_eq!(out[[0, 2]], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_f32_function() {
        let ft = FunctionTransformer::<f32>::new(|v: f32| v * 2.0);
        let x: Array2<f32> = array![[1.0f32, 2.0], [3.0, 4.0]];
        let out = ft.transform(&x).unwrap();
        assert!((out[[0, 0]] - 2.0f32).abs() < 1e-6);
        assert!((out[[1, 1]] - 8.0f32).abs() < 1e-6);
    }

    #[test]
    fn test_closure_captures_environment() {
        let scale = 3.0_f64;
        let ft = FunctionTransformer::<f64>::new(move |v| v * scale);
        let x = array![[1.0, 2.0]];
        let out = ft.transform(&x).unwrap();
        assert_abs_diff_eq!(out[[0, 0]], 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(out[[0, 1]], 6.0, epsilon = 1e-10);
    }

    #[test]
    fn test_empty_matrix() {
        let ft = FunctionTransformer::<f64>::new(|v| v);
        let x: Array2<f64> = Array2::zeros((0, 3));
        let out = ft.transform(&x).unwrap();
        assert_eq!(out.shape(), &[0, 3]);
    }
}
