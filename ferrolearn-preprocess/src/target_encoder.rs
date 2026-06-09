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
//! | REQ-4 `smooth="auto"` empirical-Bayes encoding + DEFAULT | SHIPPED | `Smooth` enum `{ Auto, Fixed(F) }` (`Default`/`TargetEncoder::default` → `Auto`); `fit_feature_encoding` Auto branch (two-pass means/ssd, `lambda_ = y_variance*count/(y_variance*count+ssd/count)`, NaN→y_mean), `population_variance_f64` (ddof=0) computed once in `fit`; sklearn `_target_encoder_fast.pyx:140-165`, `_target_encoder.py:199`,`:416`. Consumer: `TargetEncoder::fit`/`fit_transform`/`default` (the `Smooth` field drives the encoding branch) + the public module path `ferrolearn_preprocess::target_encoder::Smooth` (`pub mod target_encoder` in `lib.rs`). Verify: pins `divergence_default_smooth_is_auto`/`divergence_smooth_auto_empirical_bayes` green (#2342 #2343) |
//! | REQ-5 cross-fitting `fit_transform` (deterministic KFold) | SHIPPED | `TargetEncoder::fit_transform` cross-fits over `kfold_test_ranges` (contiguous no-shuffle folds, `cv` default 5), per-fold `fit_feature_encoding` on TRAIN rows → encode TEST rows (unseen-in-train → `y_train_mean`); sklearn `_target_encoder.py:232`,`:254-303`, `_split.py:521-534`. Consumer: crate re-export (`lib.rs`). Verify: pin `divergence_crossfit_fit_transform` green (#2344). NOTE: `shuffle`/`random_state` (REQ-8 NOT-STARTED) absent → deterministic `shuffle=False` KFold only |
//! | REQ-6 `target_type` binary/multiclass | NOT-STARTED (#1266) | sklearn `_target_encoder.py:269-273`,`:376-379` |
//! | REQ-7 `categories` param + `categories_`/`target_type_`/`classes_` | NOT-STARTED (#1267) | sklearn `_target_encoder.py:197`,`:358-381` |
//! | REQ-8 `cv`/`shuffle`/`random_state` params | NOT-STARTED (#1268) | sklearn `_target_encoder.py:200-209` |
//! | REQ-9 string/object categories | NOT-STARTED (#1269) | usize-only, R-DEV-3 |
//! | REQ-10 `get_feature_names_out`/`n_features_in_` | NOT-STARTED (#1270) | sklearn `OneToOneFeatureMixin` |
//! | REQ-11 PyO3 binding | NOT-STARTED (#1271) | `ferrolearn-python/src/` (absent) |
//! | REQ-12 ferray substrate | NOT-STARTED (#1272) | R-SUBSTRATE |
//! | REQ-13 per-category sums accumulate in f64 (always), matching sklearn's C `double` | SHIPPED | `fit` accumulates `cat_stats: HashMap<usize,(f64,usize)>` seeded with `smooth_f64*global_mean_f64`, `+= y[i].to_f64()`, then `F::from(sum/(smooth_f64+count))`; sklearn `_target_encoder_fast.pyx:42,44,68` (`double sums[]`/`counts[]`, `sums[cat]+=y[i]` regardless of `Y_DTYPE`), `encodings_` always float64 (`_target_encoder.py:335`). f64 path identity (bit-exact unchanged); `TargetEncoder<f32>` now captures `2^24+1` (#1263) |

use ferrolearn_core::error::FerroError;
use ferrolearn_core::traits::{Fit, Transform};
use ndarray::{Array1, Array2};
use num_traits::Float;
use std::collections::HashMap;

/// The smoothing strategy for [`TargetEncoder`].
///
/// Mirrors scikit-learn's `smooth` parameter
/// (`sklearn/preprocessing/_target_encoder.py:189`,
/// `"smooth": [StrOptions({"auto"}), Interval(Real, 0, None, closed="left")]`),
/// whose DEFAULT is the string `"auto"` (an empirical-Bayes estimate,
/// `_target_encoder.py:85-89`) rather than a fixed numeric value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Smooth<F> {
    /// `smooth="auto"` — the empirical-Bayes shrinkage estimate
    /// (`_target_encoder_fast.pyx:140-165`): per category blend the category
    /// mean toward the global mean by a `lambda_` derived from the
    /// within-category sum-of-squared deviations vs the overall target variance.
    Auto,
    /// A fixed numeric smoothing factor `m` driving the m-estimate
    /// `(smooth * y_mean + Σyᵢ) / (smooth + count)`
    /// (`_target_encoder_fast.pyx:60-75`). Must be non-negative.
    Fixed(F),
}

impl<F: Float> Default for Smooth<F> {
    /// The default matches scikit-learn's constructor default `smooth="auto"`
    /// (`_target_encoder.py:199`).
    fn default() -> Self {
        Smooth::Auto
    }
}

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

/// `np.mean(y)` over the first `n` elements via NumPy pairwise summation
/// (`_target_encoder.py:383` `target_mean_ = np.mean(y, axis=0)`).
fn mean_pairwise<F: Float>(y: &Array1<F>, n: usize) -> F {
    let total = if let Some(slice) = y.as_slice() {
        pairwise_sum(slice)
    } else {
        let v: Vec<F> = y.iter().copied().collect();
        pairwise_sum(&v)
    };
    total / F::from(n).unwrap_or_else(F::one)
}

/// The POPULATION variance of `y` (`np.var(y)`, ddof=0), computed in f64 to
/// match scikit-learn's C `double` accumulation. sklearn evaluates
/// `y_variance = np.var(y)` once per fit (`_target_encoder.py:416`) and feeds it
/// into the empirical-Bayes `lambda_` (`_target_encoder_fast.pyx:152-156`).
///
/// `mean_f64` is the already-computed `np.mean(y)` (`np.var` subtracts the same
/// mean); the squared deviations are reduced via NumPy pairwise summation, which
/// `np.var` uses internally.
fn population_variance_f64<F: Float>(y: &Array1<F>, mean_f64: f64) -> f64 {
    let n = y.len();
    if n == 0 {
        return 0.0;
    }
    let sq: Vec<f64> = y
        .iter()
        .map(|&v| {
            let d = v.to_f64().unwrap_or(0.0) - mean_f64;
            d * d
        })
        .collect();
    pairwise_sum(&sq) / n as f64
}

/// Learn the per-category encoding for ONE feature column.
///
/// Dispatches on the [`Smooth`] strategy. All arithmetic is done in f64
/// (matching sklearn's C `double` accumulators, `_target_encoder_fast.pyx:42,44`)
/// then cast to `F`; for `F = f64` the round-trip is the identity.
///
/// - [`Smooth::Fixed`] reproduces `_fit_encoding_fast` (`:55-77`): seed each
///   category with `(smooth*y_mean, smooth)`, add `(yᵢ, 1)` per sample, then
///   `encoding = sum/count`, or `y_mean` when `count == 0`.
/// - [`Smooth::Auto`] reproduces `_fit_encoding_fast_auto_smooth`
///   (`:120-165`): two passes (mean, then sum-of-squared-diffs), a per-category
///   `lambda_ = y_variance*count / (y_variance*count + ssd/count)`, blended as
///   `lambda_*mean + (1-lambda_)*y_mean`; a NaN `lambda_` (count 0, or
///   `y_variance == 0 && ssd == 0`) falls back to `y_mean`.
fn fit_feature_encoding<F: Float>(
    col: &[usize],
    y: &Array1<F>,
    smooth: Smooth<F>,
    y_mean_f64: f64,
    y_variance_f64: Option<f64>,
) -> HashMap<usize, F> {
    match smooth {
        Smooth::Fixed(s) => {
            let smooth_f64 = s.to_f64().unwrap_or(0.0);
            // Seed each category's accumulator with `(smooth*y_mean, smooth)`,
            // add each sample's `(yᵢ, 1)` in row order, then `sum/count`
            // (`_target_encoder_fast.pyx:60-75`).
            let mut stats: HashMap<usize, (f64, f64)> = HashMap::new();
            for (i, &cat) in col.iter().enumerate() {
                let entry = stats
                    .entry(cat)
                    .or_insert((smooth_f64 * y_mean_f64, smooth_f64));
                entry.0 += y[i].to_f64().unwrap_or(0.0);
                entry.1 += 1.0;
            }
            let mut map: HashMap<usize, F> = HashMap::new();
            for (&cat, &(sum, count)) in &stats {
                // `count` is `smooth + n_cat`; it is 0 only when smooth==0 AND
                // the category has no rows — which cannot happen here since a
                // category key exists only if a sample produced it. Guard anyway
                // to mirror sklearn's `if counts[cat]==0 -> y_mean` (`:72-73`).
                let encoded = if count == 0.0 {
                    y_mean_f64
                } else {
                    sum / count
                };
                map.insert(cat, F::from(encoded).unwrap_or_else(F::zero));
            }
            map
        }
        Smooth::Auto => {
            let y_variance = y_variance_f64.unwrap_or(0.0);
            // First pass: per-category sum + count (-> means).
            let mut sums: HashMap<usize, f64> = HashMap::new();
            let mut counts: HashMap<usize, f64> = HashMap::new();
            for (i, &cat) in col.iter().enumerate() {
                *sums.entry(cat).or_insert(0.0) += y[i].to_f64().unwrap_or(0.0);
                *counts.entry(cat).or_insert(0.0) += 1.0;
            }
            let means: HashMap<usize, f64> = sums
                .iter()
                .map(|(&cat, &s)| (cat, s / counts[&cat]))
                .collect();
            // Second pass: per-category sum of squared deviations from the mean
            // (`_target_encoder_fast.pyx:143-149`).
            let mut ssd: HashMap<usize, f64> = HashMap::new();
            for (i, &cat) in col.iter().enumerate() {
                let diff = y[i].to_f64().unwrap_or(0.0) - means[&cat];
                *ssd.entry(cat).or_insert(0.0) += diff * diff;
            }
            let mut map: HashMap<usize, F> = HashMap::new();
            for (&cat, &mean) in &means {
                let count = counts[&cat];
                let ssd_cat = ssd[&cat];
                // lambda_ = y_variance*count / (y_variance*count + ssd/count)
                // (`_target_encoder_fast.pyx:152-156`).
                let denom = y_variance * count + ssd_cat / count;
                let lambda = (y_variance * count) / denom;
                let encoded = if lambda.is_nan() {
                    // NaN when count==0 OR (y_variance==0 AND ssd==0): -> y_mean
                    // (`_target_encoder_fast.pyx:157-161`).
                    y_mean_f64
                } else {
                    lambda * mean + (1.0 - lambda) * y_mean_f64
                };
                map.insert(cat, F::from(encoded).unwrap_or_else(F::zero));
            }
            map
        }
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
/// - `smooth` — the smoothing strategy ([`Smooth`]). The DEFAULT is
///   [`Smooth::Auto`] (empirical Bayes), matching scikit-learn's constructor
///   default `smooth="auto"` (`_target_encoder.py:199`). [`Smooth::Fixed`]
///   selects the fixed m-estimate; higher values regularise more toward the
///   global mean, `Fixed(0)` is no smoothing.
/// - `cv` — the number of cross-fitting folds used by
///   [`fit_transform`](TargetEncoder::fit_transform) (default 5, matching
///   scikit-learn's `cv=5`, `_target_encoder.py:200`).
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
    /// Smoothing strategy.
    smooth: Smooth<F>,
    /// Number of cross-fitting folds for `fit_transform`.
    cv: usize,
}

impl<F: Float + Send + Sync + 'static> TargetEncoder<F> {
    /// Create a new `TargetEncoder` with a FIXED smoothing factor.
    ///
    /// This is shorthand for [`with_smooth`](Self::with_smooth) with
    /// [`Smooth::Fixed`] and `cv = 5`.
    pub fn new(smooth: F) -> Self {
        Self {
            smooth: Smooth::Fixed(smooth),
            cv: 5,
        }
    }

    /// Create a new `TargetEncoder` with the given smoothing strategy and
    /// `cv = 5` (matching scikit-learn's default).
    pub fn with_smooth(smooth: Smooth<F>) -> Self {
        Self { smooth, cv: 5 }
    }

    /// Set the number of cross-fitting folds used by
    /// [`fit_transform`](Self::fit_transform).
    pub fn with_cv(mut self, cv: usize) -> Self {
        self.cv = cv;
        self
    }

    /// Return the smoothing strategy.
    #[must_use]
    pub fn smooth(&self) -> Smooth<F> {
        self.smooth
    }

    /// Return the number of cross-fitting folds.
    #[must_use]
    pub fn cv(&self) -> usize {
        self.cv
    }
}

impl<F: Float + Send + Sync + 'static> Default for TargetEncoder<F> {
    /// The default uses [`Smooth::Auto`] (empirical Bayes) and `cv = 5`,
    /// matching scikit-learn's `TargetEncoder()` (`smooth="auto"`, `cv=5`,
    /// `_target_encoder.py:199-200`).
    fn default() -> Self {
        Self {
            smooth: Smooth::Auto,
            cv: 5,
        }
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
        if let Smooth::Fixed(s) = self.smooth
            && s < F::zero()
        {
            return Err(FerroError::InvalidParameter {
                name: "smooth".into(),
                reason: "smoothing factor must be non-negative".into(),
            });
        }

        let n_features = x.ncols();
        // sklearn: target_mean_ = np.mean(y, axis=0) (_target_encoder.py:383),
        // which reduces via NumPy pairwise summation. Reproduce it bit-for-bit so
        // the mean matches on mixed-magnitude targets.
        let global_mean = mean_pairwise(y, n_samples);
        let global_mean_f64 = global_mean.to_f64().unwrap_or(0.0);

        // For `smooth="auto"` (empirical Bayes) sklearn needs the POPULATION
        // variance of the full target, computed once per fit
        // (`_target_encoder.py:416` `y_variance = np.var(y)`).
        let y_variance_f64 = match self.smooth {
            Smooth::Auto => Some(population_variance_f64(y, global_mean_f64)),
            Smooth::Fixed(_) => None,
        };

        let mut category_maps = Vec::with_capacity(n_features);
        for j in 0..n_features {
            let col: Vec<usize> = (0..n_samples).map(|i| x[[i, j]]).collect();
            category_maps.push(fit_feature_encoding(
                &col,
                y,
                self.smooth,
                global_mean_f64,
                y_variance_f64,
            ));
        }

        Ok(FittedTargetEncoder {
            category_maps,
            global_mean,
        })
    }
}

/// The contiguous (un-shuffled) KFold test-index folds over `n` samples.
///
/// Mirrors scikit-learn's `KFold._iter_test_indices`
/// (`sklearn/model_selection/_split.py:521-534`) with `shuffle=False`: the
/// indices are `0..n` in order, split into `k` consecutive folds where the
/// first `n % k` folds have size `n // k + 1` and the rest `n // k`. Returns a
/// vec of `(test_start, test_end)` half-open ranges.
fn kfold_test_ranges(n: usize, k: usize) -> Vec<(usize, usize)> {
    let base = n / k;
    let rem = n % k;
    let mut ranges = Vec::with_capacity(k);
    let mut current = 0usize;
    for fold in 0..k {
        let size = base + usize::from(fold < rem);
        ranges.push((current, current + size));
        current += size;
    }
    ranges
}

impl<F: Float + Send + Sync + 'static> TargetEncoder<F> {
    /// Cross-fitting `fit_transform`: encode each row using encodings learned on
    /// the OTHER folds, preventing target leakage.
    ///
    /// Mirrors scikit-learn's `TargetEncoder.fit_transform`
    /// (`sklearn/preprocessing/_target_encoder.py:232-303`): for the
    /// continuous/binary single-output case it uses a deterministic `KFold`
    /// (`cv` folds, NO shuffle — ferrolearn exposes no `shuffle`/`random_state`,
    /// so this is sklearn's reproducible `shuffle=False` path, `:262`); for each
    /// `(train, test)` fold it fits the per-feature encodings on the TRAIN rows
    /// (with that fold's `y_train_mean`) and writes the TEST rows through those
    /// train-encodings (`:277-302`). A category unseen in the train fold encodes
    /// to `y_train_mean` (the `count == 0 -> y_mean` rule, mirroring
    /// `_transform_X_ordinal`'s unknown-category fallback, `:494-497`).
    ///
    /// Note `fit(X,y).transform(X)` does NOT equal `fit_transform(X,y)`
    /// (`:235-238`): `transform` uses the full-data `encodings_`, `fit_transform`
    /// is cross-fit.
    ///
    /// # Errors
    ///
    /// - [`FerroError::InsufficientSamples`] if the input has zero rows.
    /// - [`FerroError::ShapeMismatch`] if `x` rows and `y` length differ.
    /// - [`FerroError::InvalidParameter`] if a [`Smooth::Fixed`] factor is
    ///   negative, or if `cv < 2` / `cv` exceeds the sample count (sklearn
    ///   requires `cv >= 2`, `_target_encoder.py:190`, and `KFold` rejects more
    ///   splits than samples, `_split.py:408-414`).
    pub fn fit_transform(&self, x: &Array2<usize>, y: &Array1<F>) -> Result<Array2<F>, FerroError> {
        let n_samples = x.nrows();
        if n_samples == 0 {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "TargetEncoder::fit_transform".into(),
            });
        }
        if y.len() != n_samples {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_samples],
                actual: vec![y.len()],
                context: "TargetEncoder::fit_transform — y must have same length as x rows".into(),
            });
        }
        if let Smooth::Fixed(s) = self.smooth
            && s < F::zero()
        {
            return Err(FerroError::InvalidParameter {
                name: "smooth".into(),
                reason: "smoothing factor must be non-negative".into(),
            });
        }
        // sklearn `_parameter_constraints` requires `cv >= 2`
        // (`_target_encoder.py:190`); `KFold` additionally rejects more splits
        // than samples (`_split.py:408-414`).
        if self.cv < 2 {
            return Err(FerroError::InvalidParameter {
                name: "cv".into(),
                reason: "cv must be at least 2".into(),
            });
        }
        if self.cv > n_samples {
            return Err(FerroError::InvalidParameter {
                name: "cv".into(),
                reason: "cv cannot exceed the number of samples".into(),
            });
        }

        let n_features = x.ncols();
        let mut out = Array2::zeros((n_samples, n_features));

        for (test_start, test_end) in kfold_test_ranges(n_samples, self.cv) {
            // Train indices are everything OUTSIDE the contiguous test fold.
            let train_idx: Vec<usize> = (0..n_samples)
                .filter(|&i| i < test_start || i >= test_end)
                .collect();

            // y_train_mean = np.mean(y[train]) (`_target_encoder.py:279`).
            let y_train: Vec<F> = train_idx.iter().map(|&i| y[i]).collect();
            let y_train_arr = Array1::from(y_train);
            let train_mean = mean_pairwise(&y_train_arr, train_idx.len());
            let train_mean_f64 = train_mean.to_f64().unwrap_or(0.0);
            let train_var_f64 = match self.smooth {
                Smooth::Auto => Some(population_variance_f64(&y_train_arr, train_mean_f64)),
                Smooth::Fixed(_) => None,
            };

            for j in 0..n_features {
                // Fit this fold's per-feature encoding on the TRAIN rows.
                let train_col: Vec<usize> = train_idx.iter().map(|&i| x[[i, j]]).collect();
                let enc = fit_feature_encoding(
                    &train_col,
                    &y_train_arr,
                    self.smooth,
                    train_mean_f64,
                    train_var_f64,
                );
                // Encode the TEST rows; a category unseen in the train fold ->
                // the train y_mean (`_transform_X_ordinal`, `:494-497`).
                for i in test_start..test_end {
                    let cat = x[[i, j]];
                    out[[i, j]] = *enc.get(&cat).unwrap_or(&train_mean);
                }
            }
        }

        Ok(out)
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
        // sklearn's DEFAULT is smooth="auto" (`_target_encoder.py:199`), NOT a
        // fixed value; `new(F)` is the explicit fixed-smooth constructor.
        let enc = TargetEncoder::<f64>::default();
        assert_eq!(enc.smooth(), Smooth::Auto);
        assert_eq!(enc.cv(), 5);
        let fixed = TargetEncoder::<f64>::new(1.0);
        assert_eq!(fixed.smooth(), Smooth::Fixed(1.0));
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
