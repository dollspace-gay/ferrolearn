//! Target encoder: encode categorical features using target statistics.
//!
//! [`TargetEncoder`] replaces each category with the mean of the target variable
//! for that category, regularised toward the global mean using smoothing.
//!
//! This is especially useful for high-cardinality categorical features where
//! one-hot encoding would produce too many columns.
//!
//! # Smoothing
//!
//! The encoded value for category `c` is (matching scikit-learn
//! `_target_encoder_fast.pyx:60-75` — the accumulator is seeded with
//! `smooth * global_mean` then the category's targets are added, divided by
//! `smooth + count(c)`):
//!
//! ```text
//! encoded(c) = (smooth * global_mean + sum_of_targets(c)) / (smooth + count(c))
//! ```
//!
//! where `smooth` controls the degree of regularisation.
//!
//! Translation target: scikit-learn 1.5.2 `class TargetEncoder`
//! (`sklearn/preprocessing/_target_encoder.py`). Design:
//! `.design/preprocess/target_encoder.md`. Tracking: #1260.
//!
//! `## REQ status`
//!
//! | REQ | Status | Anchor |
//! |---|---|---|
//! | REQ-1 manual-`smooth` m-estimate value match (f64, bit-exact) | SHIPPED | `TargetEncoder::fit` / `transform`; sklearn `_target_encoder_fast.pyx:60-75`, `_target_encoder.py:289`,`:383` (#1261 pairwise sum, #1262 formula) |
//! | REQ-2 unseen category → `target_mean_` (global mean) | SHIPPED | `transform` `unwrap_or(global_mean)`; sklearn `_target_encoder.py:324-345` |
//! | REQ-3 InsufficientSamples / ShapeMismatch / InvalidParameter errors | SHIPPED | `fit` / `transform` guards; sklearn `_target_encoder.py:189` |
//! | REQ-4 `smooth="auto"` empirical-Bayes default | NOT-STARTED (#1264) | sklearn `_target_encoder.py:85-89`,`:189` |
//! | REQ-5 cross-fitting `fit_transform` (KFold/StratifiedKFold) | NOT-STARTED (#1265) | sklearn `_target_encoder.py:232`,`:254-303` |
//! | REQ-6 `target_type` binary/multiclass | NOT-STARTED (#1266) | sklearn `_target_encoder.py:269-273`,`:376-379` |
//! | REQ-7 `categories` param + `categories_`/`target_type_`/`classes_` | NOT-STARTED (#1267) | sklearn `_target_encoder.py:197`,`:358-381` |
//! | REQ-8 `cv`/`shuffle`/`random_state` params | NOT-STARTED (#1268) | sklearn `_target_encoder.py:200-209` |
//! | REQ-9 string/object categories | NOT-STARTED (#1269) | usize-only, R-DEV-3 |
//! | REQ-10 `get_feature_names_out`/`n_features_in_` | NOT-STARTED (#1270) | sklearn `OneToOneFeatureMixin` |
//! | REQ-11 PyO3 binding | NOT-STARTED (#1271) | `ferrolearn-python/src/` (absent) |
//! | REQ-12 ferray substrate | NOT-STARTED (#1272) | R-SUBSTRATE |
//! | REQ-13 f32 accumulates in lower precision than sklearn's f64 | NOT-STARTED (#1263) | sklearn `_target_encoder_fast.pyx:42,44,68` (always f64); f64 path CLEAN |

use ferrolearn_core::error::FerroError;
use ferrolearn_core::traits::{Fit, Transform};
use ndarray::{Array1, Array2};
use num_traits::Float;
use std::collections::HashMap;

/// Sum a slice reproducing NumPy's pairwise summation (the algorithm behind
/// `np.add.reduce` / `np.mean`), so a ferrolearn mean bit-matches sklearn's
/// `np.mean` on ill-conditioned mixed-magnitude inputs.
///
/// sklearn sets `target_mean_ = np.mean(y, axis=0)`
/// (`sklearn/preprocessing/_target_encoder.py:383`), and `np.mean` reduces via
/// NumPy pairwise summation, which rounds differently from a naive left-fold on
/// targets that mix magnitudes.
///
/// Mirrors NumPy `pairwise_sum` (numpy/_core/src/umath/loops_utils.h.src):
/// - `n < 8`        : straight sequential sum seeded from the first element.
/// - `8 <= n <= 128`: 8 partial accumulators, unrolled by 8, combined as
///   `((r0+r1)+(r2+r3)) + ((r4+r5)+(r6+r7))`, then the tail.
/// - `n > 128`      : split at `n2 = (n/2)` rounded DOWN to a multiple of 8, recurse.
fn pairwise_sum<F: Float>(data: &[F]) -> F {
    let n = data.len();
    if n == 0 {
        return F::zero();
    }
    if n < 8 {
        // Seed from the first element, then fold the rest left-to-right (numpy).
        data[1..].iter().fold(data[0], |a, &v| a + v)
    } else if n <= 128 {
        let mut r0 = data[0];
        let mut r1 = data[1];
        let mut r2 = data[2];
        let mut r3 = data[3];
        let mut r4 = data[4];
        let mut r5 = data[5];
        let mut r6 = data[6];
        let mut r7 = data[7];
        let bound = n - (n % 8);
        let mut i = 8;
        while i < bound {
            r0 = r0 + data[i];
            r1 = r1 + data[i + 1];
            r2 = r2 + data[i + 2];
            r3 = r3 + data[i + 3];
            r4 = r4 + data[i + 4];
            r5 = r5 + data[i + 5];
            r6 = r6 + data[i + 6];
            r7 = r7 + data[i + 7];
            i += 8;
        }
        let res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7));
        // Add the remainder (indices `bound..n`) left-to-right (numpy tail).
        data[bound..].iter().fold(res, |a, &v| a + v)
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        pairwise_sum(&data[..n2]) + pairwise_sum(&data[n2..])
    }
}

// ---------------------------------------------------------------------------
// TargetEncoder (unfitted)
// ---------------------------------------------------------------------------

/// An unfitted target encoder.
///
/// Takes a matrix of categorical integer features and a continuous (or binary)
/// target vector at fit time. Each category is encoded as the smoothed mean of
/// the target for that category.
///
/// # Parameters
///
/// - `smooth` — smoothing factor (default 1.0). Higher values regularise more
///   toward the global mean. Set to 0 for no smoothing.
///
/// # Examples
///
/// ```
/// use ferrolearn_preprocess::target_encoder::TargetEncoder;
/// use ferrolearn_core::traits::{Fit, Transform};
/// use ndarray::array;
///
/// let enc = TargetEncoder::<f64>::new(1.0);
/// let x = array![[0usize, 1], [0, 0], [1, 1], [1, 0]];
/// let y = array![1.0, 2.0, 3.0, 4.0];
/// let fitted = enc.fit(&x, &y).unwrap();
/// let out = fitted.transform(&x).unwrap();
/// assert_eq!(out.shape(), &[4, 2]);
/// ```
#[must_use]
#[derive(Debug, Clone)]
pub struct TargetEncoder<F> {
    /// Smoothing factor.
    smooth: F,
}

impl<F: Float + Send + Sync + 'static> TargetEncoder<F> {
    /// Create a new `TargetEncoder` with the given smoothing factor.
    pub fn new(smooth: F) -> Self {
        Self { smooth }
    }

    /// Return the smoothing factor.
    #[must_use]
    pub fn smooth(&self) -> F {
        self.smooth
    }
}

impl<F: Float + Send + Sync + 'static> Default for TargetEncoder<F> {
    fn default() -> Self {
        Self::new(F::one())
    }
}

// ---------------------------------------------------------------------------
// FittedTargetEncoder
// ---------------------------------------------------------------------------

/// A fitted target encoder holding per-feature, per-category encoding values.
///
/// Created by calling [`Fit::fit`] on a [`TargetEncoder`].
#[derive(Debug, Clone)]
pub struct FittedTargetEncoder<F> {
    /// Per-feature mapping from category → encoded value.
    category_maps: Vec<HashMap<usize, F>>,
    /// Global target mean (used for unseen categories).
    global_mean: F,
}

impl<F: Float + Send + Sync + 'static> FittedTargetEncoder<F> {
    /// Return the encoding maps per feature.
    #[must_use]
    pub fn category_maps(&self) -> &[HashMap<usize, F>] {
        &self.category_maps
    }

    /// Return the global target mean.
    #[must_use]
    pub fn global_mean(&self) -> F {
        self.global_mean
    }
}

// ---------------------------------------------------------------------------
// Trait implementations
// ---------------------------------------------------------------------------

impl<F: Float + Send + Sync + 'static> Fit<Array2<usize>, Array1<F>> for TargetEncoder<F> {
    type Fitted = FittedTargetEncoder<F>;
    type Error = FerroError;

    /// Fit the encoder by computing smoothed target means per category.
    ///
    /// # Errors
    ///
    /// - [`FerroError::InsufficientSamples`] if the input has zero rows.
    /// - [`FerroError::ShapeMismatch`] if `x` rows and `y` length differ.
    /// - [`FerroError::InvalidParameter`] if `smooth` is negative.
    fn fit(&self, x: &Array2<usize>, y: &Array1<F>) -> Result<FittedTargetEncoder<F>, FerroError> {
        let n_samples = x.nrows();
        if n_samples == 0 {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "TargetEncoder::fit".into(),
            });
        }
        if y.len() != n_samples {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_samples],
                actual: vec![y.len()],
                context: "TargetEncoder::fit — y must have same length as x rows".into(),
            });
        }
        if self.smooth < F::zero() {
            return Err(FerroError::InvalidParameter {
                name: "smooth".into(),
                reason: "smoothing factor must be non-negative".into(),
            });
        }

        let n_features = x.ncols();
        // sklearn: target_mean_ = np.mean(y, axis=0) (_target_encoder.py:383),
        // which reduces via NumPy pairwise summation. Reproduce it bit-for-bit so
        // the mean matches on mixed-magnitude targets.
        let global_mean = if let Some(slice) = y.as_slice() {
            pairwise_sum(slice)
        } else {
            let v: Vec<F> = y.iter().copied().collect();
            pairwise_sum(&v)
        } / F::from(n_samples).unwrap_or_else(F::one);

        let mut category_maps = Vec::with_capacity(n_features);

        for j in 0..n_features {
            // Collect (sum, count) per category. Match sklearn's
            // `_target_encoder_fast.pyx:60-75` bit-for-bit: seed each category's
            // accumulator with `smooth * y_mean` FIRST (`:60` `sums[cat]=smooth_sum`),
            // then add each target sequentially in row order (`:68` `sums[cat]+=y[i]`).
            // So the accumulated F is `smooth*y_mean + y[0] + y[1] + ...`.
            let mut cat_stats: HashMap<usize, (F, usize)> = HashMap::new();
            for i in 0..n_samples {
                let cat = x[[i, j]];
                let entry = cat_stats
                    .entry(cat)
                    .or_insert_with(|| (self.smooth * global_mean, 0usize));
                entry.0 = entry.0 + y[i];
                entry.1 += 1;
            }

            // Per category: `encoding = sums[cat] / counts[cat]`
            // (`_target_encoder_fast.pyx:75`), where `counts[cat] = smooth + count`
            // (count seeded from `smooth` at `:61`) and `sums[cat]` already includes
            // the `smooth*y_mean` seed. i.e. `(smooth*y_mean + Σyᵢ) / (smooth + count)`.
            let mut cat_map: HashMap<usize, F> = HashMap::new();
            for (&cat, &(sum, count)) in &cat_stats {
                let count_f = F::from(count).unwrap_or_else(F::one);
                let encoded = sum / (self.smooth + count_f);
                cat_map.insert(cat, encoded);
            }

            category_maps.push(cat_map);
        }

        Ok(FittedTargetEncoder {
            category_maps,
            global_mean,
        })
    }
}

impl<F: Float + Send + Sync + 'static> Transform<Array2<usize>> for FittedTargetEncoder<F> {
    type Output = Array2<F>;
    type Error = FerroError;

    /// Encode categorical features using the learned target statistics.
    ///
    /// Unseen categories are encoded as the global target mean.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of columns differs
    /// from the number of features seen during fitting.
    fn transform(&self, x: &Array2<usize>) -> Result<Array2<F>, FerroError> {
        let n_features = self.category_maps.len();
        if x.ncols() != n_features {
            return Err(FerroError::ShapeMismatch {
                expected: vec![x.nrows(), n_features],
                actual: vec![x.nrows(), x.ncols()],
                context: "FittedTargetEncoder::transform".into(),
            });
        }

        let n_samples = x.nrows();
        let mut out = Array2::zeros((n_samples, n_features));

        for j in 0..n_features {
            let cat_map = &self.category_maps[j];
            for i in 0..n_samples {
                let cat = x[[i, j]];
                out[[i, j]] = *cat_map.get(&cat).unwrap_or(&self.global_mean);
            }
        }

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
    fn test_target_encoder_basic() {
        let enc = TargetEncoder::<f64>::new(0.0); // no smoothing
        // Category 0: targets [1.0, 2.0], mean = 1.5
        // Category 1: targets [3.0, 4.0], mean = 3.5
        let x = array![[0usize], [0], [1], [1]];
        let y = array![1.0, 2.0, 3.0, 4.0];
        let fitted = enc.fit(&x, &y).unwrap();
        let out = fitted.transform(&x).unwrap();
        assert_abs_diff_eq!(out[[0, 0]], 1.5, epsilon = 1e-10);
        assert_abs_diff_eq!(out[[1, 0]], 1.5, epsilon = 1e-10);
        assert_abs_diff_eq!(out[[2, 0]], 3.5, epsilon = 1e-10);
        assert_abs_diff_eq!(out[[3, 0]], 3.5, epsilon = 1e-10);
    }

    #[test]
    fn test_target_encoder_smoothing() {
        let enc = TargetEncoder::<f64>::new(2.0);
        // Category 0: targets [1.0], mean = 1.0, count = 1
        // Category 1: targets [3.0, 5.0], mean = 4.0, count = 2
        // Global mean = (1 + 3 + 5) / 3 = 3.0
        let x = array![[0usize], [1], [1]];
        let y = array![1.0, 3.0, 5.0];
        let fitted = enc.fit(&x, &y).unwrap();
        let out = fitted.transform(&x).unwrap();
        // Cat 0: (1 * 1.0 + 2 * 3.0) / (1 + 2) = 7/3 ≈ 2.333
        let expected_0 = (1.0 * 1.0 + 2.0 * 3.0) / (1.0 + 2.0);
        assert_abs_diff_eq!(out[[0, 0]], expected_0, epsilon = 1e-10);
        // Cat 1: (2 * 4.0 + 2 * 3.0) / (2 + 2) = 14/4 = 3.5
        let expected_1 = (2.0 * 4.0 + 2.0 * 3.0) / (2.0 + 2.0);
        assert_abs_diff_eq!(out[[1, 0]], expected_1, epsilon = 1e-10);
    }

    #[test]
    fn test_target_encoder_unseen_category() {
        let enc = TargetEncoder::<f64>::new(1.0);
        let x = array![[0usize], [0], [1], [1]];
        let y = array![1.0, 2.0, 3.0, 4.0];
        let fitted = enc.fit(&x, &y).unwrap();
        // Transform with unseen category 2
        let x_new = array![[2usize]];
        let out = fitted.transform(&x_new).unwrap();
        // Unseen category → global mean = 2.5
        assert_abs_diff_eq!(out[[0, 0]], 2.5, epsilon = 1e-10);
    }

    #[test]
    fn test_target_encoder_multi_feature() {
        let enc = TargetEncoder::<f64>::new(0.0);
        let x = array![[0usize, 1], [0, 0], [1, 1], [1, 0]];
        let y = array![1.0, 2.0, 3.0, 4.0];
        let fitted = enc.fit(&x, &y).unwrap();
        let out = fitted.transform(&x).unwrap();
        assert_eq!(out.shape(), &[4, 2]);
    }

    #[test]
    fn test_target_encoder_zero_rows_error() {
        let enc = TargetEncoder::<f64>::new(1.0);
        let x: Array2<usize> = Array2::zeros((0, 2));
        let y: Array1<f64> = Array1::zeros(0);
        assert!(enc.fit(&x, &y).is_err());
    }

    #[test]
    fn test_target_encoder_shape_mismatch_fit() {
        let enc = TargetEncoder::<f64>::new(1.0);
        let x = array![[0usize], [1]];
        let y = array![1.0]; // wrong length
        assert!(enc.fit(&x, &y).is_err());
    }

    #[test]
    fn test_target_encoder_shape_mismatch_transform() {
        let enc = TargetEncoder::<f64>::new(1.0);
        let x = array![[0usize, 1], [1, 0]];
        let y = array![1.0, 2.0];
        let fitted = enc.fit(&x, &y).unwrap();
        let x_bad = array![[0usize]]; // wrong number of columns
        assert!(fitted.transform(&x_bad).is_err());
    }

    #[test]
    fn test_target_encoder_negative_smooth_error() {
        let enc = TargetEncoder::<f64>::new(-1.0);
        let x = array![[0usize]];
        let y = array![1.0];
        assert!(enc.fit(&x, &y).is_err());
    }

    #[test]
    fn test_target_encoder_default() {
        let enc = TargetEncoder::<f64>::default();
        assert_abs_diff_eq!(enc.smooth(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_target_encoder_global_mean_accessor() {
        let enc = TargetEncoder::<f64>::new(0.0);
        let x = array![[0usize], [1]];
        let y = array![2.0, 4.0];
        let fitted = enc.fit(&x, &y).unwrap();
        assert_abs_diff_eq!(fitted.global_mean(), 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_target_encoder_f32() {
        let enc = TargetEncoder::<f32>::new(1.0f32);
        let x = array![[0usize], [0], [1]];
        let y: Array1<f32> = array![1.0f32, 2.0, 3.0];
        let fitted = enc.fit(&x, &y).unwrap();
        let out = fitted.transform(&x).unwrap();
        assert!(!out[[0, 0]].is_nan());
    }
}
