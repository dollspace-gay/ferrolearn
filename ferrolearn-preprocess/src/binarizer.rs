//! Binarizer: threshold features to binary values.
//!
//! Values strictly greater than the threshold are set to `1.0`; all other
//! values are set to `0.0`.
//!
//! This transformer is **stateless** — no fitting is required. Call
//! [`Transform::transform`] directly.
//!
//! # `## REQ status`
//!
//! Binary (R-DEFER-2), translating `sklearn/preprocessing/_data.py` (`class Binarizer`
//! `:2177`, `binarize` `:2120`). Design doc: `.design/preprocess/binarizer.md`. Expected
//! values from the live sklearn 1.5.2 oracle (R-CHAR-3). Consumer: crate re-export
//! (`lib.rs:106`, grandfathered S5). Two SHIPPED REQs are critic-verified vs the oracle;
//! the remaining surface is NOT-STARTED with concrete blockers.
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (dense strict-greater transform) | SHIPPED | `Transform::transform` = `x.mapv(\|v\| if v > threshold { 1 } else { 0 })`, strict `>`, shape-preserving; `Default` threshold 0.0. Mirrors sklearn `binarize` dense path (`_data.py:2170-2173`). Critic-verified bit-identical to live sklearn (`guard_binarizer_*` in `tests/divergence_binarizer.rs`: thr 0.5 → `[[0,1,0],[1,0,0]]`, default, negative, f32). Consumer: `pub use binarizer::Binarizer` (`lib.rs:106`). |
//! | REQ-9 (transform input validation per check_array) | SHIPPED | FIXED #1123/#1124/#1125. `transform` rejects (in sklearn order) zero-samples → `InsufficientSamples` (`validation.py:1084`), zero-features → `InvalidParameter` (`:1093`), non-finite NaN/±inf → `InvalidParameter` (`:1063`, force_all_finite=True) — matching sklearn `Binarizer.transform` `_validate_data` (`_data.py:2301`). 13 live-oracle tests green; finite extremes (1e308/-0.0/subnormal) not over-rejected. Two-round critic-verified CLEAN. |
//! | REQ-2 (copy param) | NOT-STARTED | open prereq blocker #1126. No `copy` ctor/transform param (sklearn `:2253`,`:2298-2307`). |
//! | REQ-3 (fit + parameter-constraints validation) | NOT-STARTED | open prereq blocker #1127. No `fit`/`_parameter_constraints` (threshold is Real → InvalidParameterError; sklearn `:2248-2278`). Transform-time INPUT validation is REQ-9; this is fit-time PARAMETER validation. |
//! | REQ-4 (binarize free function) | SHIPPED | FIXED #1128. Standalone [`binarize`] (`x.mapv(\|v\| if v > threshold { 1 } else { 0 })`, strict `>`, shape-preserving) mirrors sklearn `binarize` dense path (`_data.py:2120-2174`); keyword default `threshold=0.0` documented. `Transform::transform` delegates its mapping to `binarize`, so the two are byte-identical. Critic-verified vs the live sklearn 1.5.2 oracle (`binarize_*_matches_sklearn`). |
//! | REQ-5 (n_features_in_ / feature names) | NOT-STARTED | open prereq blocker #1129. No `n_features_in_`/`get_feature_names_out` (OneToOneFeatureMixin; sklearn `:2277`). Depends on REQ-3. |
//! | REQ-6 (sparse support) | NOT-STARTED | open prereq blocker #1130. Dense-only; no CSR/CSC path, no `threshold<0` guard, no `eliminate_zeros` (sklearn `:2161-2168`). |
//! | REQ-7 (PyO3 binding) | NOT-STARTED | open prereq blocker #1131. No `ferrolearn-python` registration. |
//! | REQ-8 (ferray substrate) | NOT-STARTED | open prereq blocker #1132. `ndarray`/`num_traits`, not `ferray-core`/`ferray-ufunc` (R-SUBSTRATE-1/2). |

use ferrolearn_core::error::FerroError;
use ferrolearn_core::traits::Transform;
use ndarray::Array2;
use num_traits::Float;

// ---------------------------------------------------------------------------
// binarize (free function)
// ---------------------------------------------------------------------------

/// Boolean thresholding of a dense array, element by element.
///
/// Values **strictly greater** than `threshold` become `1.0`; all other values
/// (less than *or equal to* the threshold) become `0.0`. The result is a new,
/// shape-preserving array.
///
/// This is the estimator-less functional form of [`Binarizer`], mirroring
/// scikit-learn's `binarize(X, *, threshold=0.0, copy=True)`
/// (`sklearn/preprocessing/_data.py:2120-2174`), whose dense path is
/// `cond = X > threshold; X[cond] = 1; X[not_cond] = 0` (`:2170-2173`) — the
/// load-bearing strict greater-than. scikit-learn's keyword default is
/// `threshold=0.0` (only positive values map to `1.0`); here the caller passes
/// the threshold explicitly.
///
/// [`Binarizer`]'s [`Transform::transform`] delegates its element mapping to
/// this function, so the two share one implementation.
///
/// # Examples
///
/// ```
/// use ferrolearn_preprocess::binarizer::binarize;
/// use ndarray::array;
///
/// let x = array![[0.4, 0.6, 0.5], [0.6, 0.1, 0.2]];
/// let out = binarize(&x, 0.5);
/// // out = [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]
/// ```
#[must_use]
pub fn binarize<F>(x: &Array2<F>, threshold: F) -> Array2<F>
where
    F: Float,
{
    x.mapv(|v| if v > threshold { F::one() } else { F::zero() })
}

// ---------------------------------------------------------------------------
// Binarizer
// ---------------------------------------------------------------------------

/// A stateless feature binarizer.
///
/// Values strictly greater than `threshold` become `1.0`; all other values
/// become `0.0`. The default threshold is `0.0`.
///
/// This transformer is stateless — no fitting is needed. Call
/// [`Transform::transform`] directly.
///
/// # Examples
///
/// ```
/// use ferrolearn_preprocess::binarizer::Binarizer;
/// use ferrolearn_core::traits::Transform;
/// use ndarray::array;
///
/// let binarizer = Binarizer::<f64>::new(0.5);
/// let x = array![[0.0, 0.5, 1.0]];
/// let out = binarizer.transform(&x).unwrap();
/// // out = [[0.0, 0.0, 1.0]]
/// ```
#[derive(Debug, Clone)]
pub struct Binarizer<F> {
    /// The threshold value. Values strictly greater than this become 1.0.
    pub(crate) threshold: F,
}

impl<F: Float + Send + Sync + 'static> Binarizer<F> {
    /// Create a new `Binarizer` with the given threshold.
    #[must_use]
    pub fn new(threshold: F) -> Self {
        Self { threshold }
    }

    /// Return the configured threshold.
    #[must_use]
    pub fn threshold(&self) -> F {
        self.threshold
    }
}

impl<F: Float + Send + Sync + 'static> Default for Binarizer<F> {
    fn default() -> Self {
        Self::new(F::zero())
    }
}

// ---------------------------------------------------------------------------
// Trait implementations
// ---------------------------------------------------------------------------

impl<F: Float + Send + Sync + 'static> Transform<Array2<F>> for Binarizer<F> {
    type Output = Array2<F>;
    type Error = FerroError;

    /// Apply the threshold: values > threshold become `1.0`, others become `0.0`.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::InsufficientSamples`] if `x` has zero rows. This
    /// mirrors scikit-learn's `Binarizer.transform`
    /// (`sklearn/preprocessing/_data.py:2301`), whose `_validate_data` ->
    /// `check_array` min-samples check raises `ValueError: Found array with 0
    /// sample(s) ... while a minimum of 1 is required by Binarizer.`
    ///
    /// Returns [`FerroError::InvalidParameter`] if `x` has zero features
    /// (columns). This mirrors scikit-learn's `Binarizer.transform`
    /// (`sklearn/preprocessing/_data.py:2301`), whose `_validate_data` ->
    /// `check_array` min-features check (`utils/validation.py:1093`,
    /// `ensure_min_features=1`) raises `ValueError: Found array with 0
    /// feature(s) (shape=(3, 0)) while a minimum of 1 is required by Binarizer.`
    ///
    /// Returns [`FerroError::InvalidParameter`] if `x` contains any non-finite
    /// value (NaN, +inf, or -inf). This mirrors scikit-learn's
    /// `Binarizer.transform` (`sklearn/preprocessing/_data.py:2301`), which
    /// validates input via `check_array(force_all_finite=True)` and raises
    /// `ValueError: Input X contains NaN.` / `Input X contains infinity ...`
    /// before applying the threshold comparison.
    fn transform(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        if x.nrows() == 0 {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "Binarizer::transform".into(),
            });
        }
        if x.ncols() == 0 {
            return Err(FerroError::InvalidParameter {
                name: "X".to_string(),
                reason: "Found array with 0 feature(s); a minimum of 1 is required \
                         by Binarizer"
                    .to_string(),
            });
        }
        if x.iter().any(|v| !v.is_finite()) {
            return Err(FerroError::InvalidParameter {
                name: "X".to_string(),
                reason: "Input X contains non-finite values (NaN or infinity); \
                         Binarizer requires all-finite input"
                    .to_string(),
            });
        }
        Ok(binarize(x, self.threshold))
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
    fn test_binarizer_default_threshold() {
        let b = Binarizer::<f64>::default();
        assert_eq!(b.threshold(), 0.0);
        let x = array![[-1.0, 0.0, 0.5, 1.0]];
        let out = b.transform(&x).unwrap();
        assert_abs_diff_eq!(out[[0, 0]], 0.0, epsilon = 1e-10); // -1 <= 0
        assert_abs_diff_eq!(out[[0, 1]], 0.0, epsilon = 1e-10); // 0 not > 0
        assert_abs_diff_eq!(out[[0, 2]], 1.0, epsilon = 1e-10); // 0.5 > 0
        assert_abs_diff_eq!(out[[0, 3]], 1.0, epsilon = 1e-10); // 1.0 > 0
    }

    #[test]
    fn test_binarizer_custom_threshold() {
        let b = Binarizer::<f64>::new(0.5);
        let x = array![[0.0, 0.5, 1.0]];
        let out = b.transform(&x).unwrap();
        assert_abs_diff_eq!(out[[0, 0]], 0.0, epsilon = 1e-10); // 0.0 not > 0.5
        assert_abs_diff_eq!(out[[0, 1]], 0.0, epsilon = 1e-10); // 0.5 not > 0.5 (strict)
        assert_abs_diff_eq!(out[[0, 2]], 1.0, epsilon = 1e-10); // 1.0 > 0.5
    }

    #[test]
    fn test_binarizer_all_zeros() {
        let b = Binarizer::<f64>::new(0.0);
        let x = array![[0.0, 0.0, 0.0]];
        let out = b.transform(&x).unwrap();
        for v in &out {
            assert_abs_diff_eq!(*v, 0.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_binarizer_all_ones() {
        let b = Binarizer::<f64>::new(0.0);
        let x = array![[1.0, 2.0, 3.0]];
        let out = b.transform(&x).unwrap();
        for v in &out {
            assert_abs_diff_eq!(*v, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_binarizer_negative_threshold() {
        let b = Binarizer::<f64>::new(-1.0);
        let x = array![[-2.0, -1.0, -0.5, 0.0]];
        let out = b.transform(&x).unwrap();
        assert_abs_diff_eq!(out[[0, 0]], 0.0, epsilon = 1e-10); // -2 <= -1
        assert_abs_diff_eq!(out[[0, 1]], 0.0, epsilon = 1e-10); // -1 not > -1
        assert_abs_diff_eq!(out[[0, 2]], 1.0, epsilon = 1e-10); // -0.5 > -1
        assert_abs_diff_eq!(out[[0, 3]], 1.0, epsilon = 1e-10); // 0.0 > -1
    }

    #[test]
    fn test_binarizer_multiple_rows() {
        let b = Binarizer::<f64>::new(2.0);
        let x = array![[1.0, 3.0], [2.0, 4.0], [5.0, 0.0]];
        let out = b.transform(&x).unwrap();
        assert_eq!(out.shape(), &[3, 2]);
        assert_abs_diff_eq!(out[[0, 0]], 0.0, epsilon = 1e-10); // 1 <= 2
        assert_abs_diff_eq!(out[[0, 1]], 1.0, epsilon = 1e-10); // 3 > 2
        assert_abs_diff_eq!(out[[1, 0]], 0.0, epsilon = 1e-10); // 2 not > 2
        assert_abs_diff_eq!(out[[1, 1]], 1.0, epsilon = 1e-10); // 4 > 2
        assert_abs_diff_eq!(out[[2, 0]], 1.0, epsilon = 1e-10); // 5 > 2
        assert_abs_diff_eq!(out[[2, 1]], 0.0, epsilon = 1e-10); // 0 <= 2
    }

    #[test]
    fn test_binarizer_preserves_shape() {
        let b = Binarizer::<f64>::default();
        let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let out = b.transform(&x).unwrap();
        assert_eq!(out.shape(), x.shape());
    }

    #[test]
    fn test_binarizer_f32() {
        let b = Binarizer::<f32>::new(0.0f32);
        let x: Array2<f32> = array![[1.0f32, -1.0, 0.0]];
        let out = b.transform(&x).unwrap();
        assert!((out[[0, 0]] - 1.0f32).abs() < 1e-6);
        assert!((out[[0, 1]] - 0.0f32).abs() < 1e-6);
        assert!((out[[0, 2]] - 0.0f32).abs() < 1e-6);
    }

    // -- binarize free function (REQ-4) -- oracle-grounded vs live sklearn 1.5.2 --
    // X = [[1,-1,2],[2,0,0],[0,1,-1]]
    // python3 -c "from sklearn.preprocessing import binarize; import numpy as np; \
    //   print(binarize(np.array([[1.,-1,2],[2,0,0],[0,1,-1]])).tolist())"
    //   -> [[1,0,1],[1,0,0],[0,1,0]]   (threshold 0.0, strict >)
    // python3 -c "from sklearn.preprocessing import binarize; import numpy as np; \
    //   print(binarize(np.array([[1.,-1,2],[2,0,0],[0,1,-1]]), threshold=-0.5).tolist())"
    //   -> [[1,0,1],[1,1,1],[1,1,0]]

    #[test]
    fn binarize_default_threshold_matches_sklearn() {
        let x = array![[1.0, -1.0, 2.0], [2.0, 0.0, 0.0], [0.0, 1.0, -1.0]];
        let out = binarize(&x, 0.0);
        let expected = array![[1.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        assert_eq!(out, expected);
    }

    #[test]
    fn binarize_negative_threshold_matches_sklearn() {
        let x = array![[1.0, -1.0, 2.0], [2.0, 0.0, 0.0], [0.0, 1.0, -1.0]];
        let out = binarize(&x, -0.5);
        let expected = array![[1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 0.0]];
        assert_eq!(out, expected);
    }

    #[test]
    fn binarize_matches_estimator_transform() {
        let x = array![[1.0, -1.0, 2.0], [2.0, 0.0, 0.0], [0.0, 1.0, -1.0]];
        let free = binarize(&x, 0.5);
        let est = Binarizer::<f64>::new(0.5).transform(&x).ok();
        assert_eq!(est, Some(free));
    }

    #[test]
    fn test_output_values_are_zero_or_one() {
        let b = Binarizer::<f64>::new(0.0);
        let x = array![[-5.0, -1.0, 0.0, 0.001, 1.0, 100.0]];
        let out = b.transform(&x).unwrap();
        for v in &out {
            assert!(*v == 0.0 || *v == 1.0, "expected 0 or 1, got {v}");
        }
    }
}
