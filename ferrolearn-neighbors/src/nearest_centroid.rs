//! Nearest Centroid classifier.
//!
//! This module provides [`NearestCentroid`], which classifies samples by
//! computing the mean of each class during training and assigning new
//! samples to the class with the nearest centroid (Euclidean distance).
//!
//! An optional `shrink_threshold` parameter allows centroid shrinkage: each
//! class centroid is moved toward the overall centroid by subtracting a
//! threshold-dependent amount from the per-class offsets.
//!
//! # Examples
//!
//! ```
//! use ferrolearn_neighbors::nearest_centroid::NearestCentroid;
//! use ferrolearn_core::{Fit, Predict};
//! use ndarray::{array, Array2};
//!
//! let x = Array2::from_shape_vec((6, 2), vec![
//!     0.0, 0.0, 0.5, 0.0, 0.0, 0.5,
//!     5.0, 5.0, 5.5, 5.0, 5.0, 5.5,
//! ]).unwrap();
//! let y = array![0, 0, 0, 1, 1, 1];
//!
//! let model = NearestCentroid::<f64>::new();
//! let fitted = model.fit(&x, &y).unwrap();
//! let preds = fitted.predict(&x).unwrap();
//! assert_eq!(preds.len(), 6);
//! ```
//!
//! ## REQ status
//!
//! Mirrors `sklearn.neighbors.NearestCentroid`
//! (`sklearn/neighbors/_nearest_centroid.py`). See
//! `.design/neighbors/nearest_centroid.md`. Non-test consumer: crate re-export
//! + the `RsNearestCentroid` PyO3 binding (`ferrolearn-python/src/extras.rs`).
//!
//! | REQ | Description | Status |
//! |-----|-------------|--------|
//! | REQ-1 | Euclidean per-class-mean `centroids_` + nearest-centroid `predict` + `classes_` ordering (`_nearest_centroid.py:171,:217`) | SHIPPED |
//! | REQ-2 | `shrink_threshold` shrunken centroids: `s = sqrt(var/(n-K))`, `s += median(s)` (`:183-184`), soft-threshold `sign·max(\|dev\|−thr,0)`, `centroid = dataset_centroid + m·s·signed_dev`; verified on 3-class/full/partial/partial-constant inputs | SHIPPED |
//! | REQ-3 | `n_classes < 2` → `ValueError` (`:147-151`) | SHIPPED |
//! | REQ-4 | All-features-zero-variance + shrink → `ValueError` (`:174-175`) | SHIPPED |
//! | REQ-5 | `metric` param + `metric='manhattan'` median centroid (`NcMetric` + `with_metric`; feature-wise median `:167`, L1 predict `:218`) | SHIPPED (#841) |
//! | REQ-6 | `shrink_threshold > 0` constraint (sklearn `InvalidParameterError` on 0/negative) | NOT-STARTED (#842) |
//! | REQ-7 | PyO3 binding fidelity + ferray substrate | NOT-STARTED (#843) |

use ferrolearn_core::error::FerroError;
use ferrolearn_core::introspection::HasClasses;
use ferrolearn_core::traits::{Fit, Predict};
use ndarray::{Array1, Array2};
use num_traits::Float;

// ---------------------------------------------------------------------------
// NearestCentroid
// ---------------------------------------------------------------------------

/// Distance metric used by [`NearestCentroid`].
///
/// Mirrors sklearn's `metric` parameter
/// (`_nearest_centroid.py:104`: `StrOptions({"manhattan", "euclidean"})`).
/// The metric governs BOTH how each class centroid is computed during `fit`
/// and how `predict` measures distance to the centroids.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NcMetric {
    /// `metric="euclidean"` (sklearn default, `_nearest_centroid.py:104`).
    ///
    /// Centroid = per-class arithmetic MEAN (`:171`); `predict` assigns by
    /// (squared) L2 distance.
    #[default]
    Euclidean,
    /// `metric="manhattan"` (`_nearest_centroid.py:104`).
    ///
    /// Centroid = per-class feature-wise MEDIAN (`:167`,
    /// `np.median(X[mask], axis=0)`); `predict` assigns by L1 distance.
    Manhattan,
}

/// Nearest Centroid classifier.
///
/// Classifies samples by assigning them to the class with the nearest
/// centroid in the configured [`NcMetric`]. For [`NcMetric::Euclidean`] the
/// centroid is the per-class mean and distance is L2; for
/// [`NcMetric::Manhattan`] the centroid is the per-class feature-wise median
/// and distance is L1.
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
#[derive(Debug, Clone)]
pub struct NearestCentroid<F> {
    /// Distance metric. Default: [`NcMetric::Euclidean`] (sklearn default,
    /// `_nearest_centroid.py:108`).
    pub metric: NcMetric,
    /// Optional shrinkage threshold. If set, each class centroid is moved
    /// toward the overall centroid by this amount. Default: `None`.
    ///
    /// Shrinkage is defined for the euclidean/mean path only. When `metric ==
    /// NcMetric::Manhattan`, `shrink_threshold` is currently IGNORED: sklearn's
    /// shrinkage formula (`_nearest_centroid.py:173-196`) is derived for the
    /// mean centroid, and the manhattan-with-shrink combination is not yet
    /// translated.
    pub shrink_threshold: Option<F>,
}

impl<F: Float> NearestCentroid<F> {
    /// Create a new `NearestCentroid` with the default ([`NcMetric::Euclidean`])
    /// metric and no shrinkage.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metric: NcMetric::Euclidean,
            shrink_threshold: None,
        }
    }

    /// Set the distance [`NcMetric`].
    ///
    /// Mirrors sklearn's `metric` constructor parameter
    /// (`_nearest_centroid.py:108`).
    #[must_use]
    pub fn with_metric(mut self, metric: NcMetric) -> Self {
        self.metric = metric;
        self
    }

    /// Set the shrinkage threshold.
    ///
    /// When set, each per-class centroid deviation from the overall centroid
    /// is soft-thresholded (shifted toward zero), which can improve
    /// generalization and implicitly perform feature selection.
    #[must_use]
    pub fn with_shrink_threshold(mut self, threshold: F) -> Self {
        self.shrink_threshold = Some(threshold);
        self
    }
}

impl<F: Float> Default for NearestCentroid<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted Nearest Centroid classifier.
///
/// Stores the class centroids and class labels computed during fitting.
#[derive(Debug, Clone)]
pub struct FittedNearestCentroid<F> {
    /// Per-class centroids, shape `(n_classes, n_features)`.
    centroids: Array2<F>,
    /// Sorted unique class labels.
    classes: Vec<usize>,
    /// Distance metric used to assign samples in `predict`
    /// (`_nearest_centroid.py:218`, `metric=self.metric`).
    metric: NcMetric,
}

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, Array1<usize>> for NearestCentroid<F> {
    type Fitted = FittedNearestCentroid<F>;
    type Error = FerroError;

    /// Fit the Nearest Centroid classifier by computing class means.
    ///
    /// # Errors
    ///
    /// - [`FerroError::ShapeMismatch`] if `x` and `y` have different numbers of rows.
    /// - [`FerroError::InsufficientSamples`] if there are no samples.
    /// - [`FerroError::InvalidParameter`] if `shrink_threshold` is negative.
    fn fit(
        &self,
        x: &Array2<F>,
        y: &Array1<usize>,
    ) -> Result<FittedNearestCentroid<F>, FerroError> {
        let (n_samples, n_features) = x.dim();

        if n_samples == 0 {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "NearestCentroid requires at least one sample".into(),
            });
        }

        if n_samples != y.len() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_samples],
                actual: vec![y.len()],
                context: "y length must match number of samples in X".into(),
            });
        }

        if let Some(threshold) = self.shrink_threshold
            && threshold < F::zero()
        {
            return Err(FerroError::InvalidParameter {
                name: "shrink_threshold".into(),
                reason: "must be non-negative".into(),
            });
        }

        // Collect sorted unique classes.
        let mut classes: Vec<usize> = y.to_vec();
        classes.sort_unstable();
        classes.dedup();
        let n_classes = classes.len();

        // sklearn `_nearest_centroid.py:147-151`: raise ValueError when fewer
        // than two classes are present.
        if n_classes < 2 {
            return Err(FerroError::InvalidParameter {
                name: "y".into(),
                reason: format!(
                    "The number of classes has to be greater than one; got {n_classes} class"
                ),
            });
        }

        // Compute per-class centroids.
        let mut centroids = Array2::<F>::zeros((n_classes, n_features));

        for (ci, &class_label) in classes.iter().enumerate() {
            let class_indices: Vec<usize> = y
                .iter()
                .enumerate()
                .filter_map(|(i, &label)| if label == class_label { Some(i) } else { None })
                .collect();

            let n_c = class_indices.len();

            match self.metric {
                // `_nearest_centroid.py:171`: euclidean → per-class feature-wise
                // arithmetic mean `X[mask].mean(axis=0)`.
                NcMetric::Euclidean => {
                    let n_c_f = F::from(n_c).unwrap_or_else(F::one);
                    for j in 0..n_features {
                        let sum = class_indices
                            .iter()
                            .fold(F::zero(), |acc, &i| acc + x[[i, j]]);
                        centroids[[ci, j]] = sum / n_c_f;
                    }
                }
                // `_nearest_centroid.py:167`: manhattan → per-class feature-wise
                // MEDIAN `np.median(X[mask], axis=0)`. numpy convention: sort the
                // column values; odd count → middle element, even count →
                // average of the two middle elements.
                NcMetric::Manhattan => {
                    let two = F::one() + F::one();
                    for j in 0..n_features {
                        let mut col: Vec<F> = class_indices.iter().map(|&i| x[[i, j]]).collect();
                        col.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                        let mid = col.len() / 2;
                        centroids[[ci, j]] = if col.len() % 2 == 1 {
                            col[mid]
                        } else {
                            (col[mid - 1] + col[mid]) / two
                        };
                    }
                }
            }
        }

        // Apply shrinkage if requested.
        if let Some(threshold) = self.shrink_threshold {
            // sklearn `_nearest_centroid.py:174-175`: raise ValueError when all
            // features have zero variance (ptp == 0 for every feature).
            // `np.ptp(X, axis=0)` is per-feature range (max - min); the check
            // `np.all(... == 0)` fires only when EVERY feature is constant.
            let all_zero_variance = (0..n_features).all(|j| {
                let (col_min, col_max) =
                    (0..n_samples).fold((x[[0, j]], x[[0, j]]), |(mn, mx), i| {
                        (
                            if x[[i, j]] < mn { x[[i, j]] } else { mn },
                            if x[[i, j]] > mx { x[[i, j]] } else { mx },
                        )
                    });
                col_max - col_min == F::zero()
            });
            if all_zero_variance {
                return Err(FerroError::InvalidParameter {
                    name: "X".into(),
                    reason: "All features have zero variance. Division by zero.".into(),
                });
            }

            // Compute overall centroid.
            let mut overall = Array1::<F>::zeros(n_features);
            for j in 0..n_features {
                let sum = (0..n_samples).fold(F::zero(), |acc, i| acc + x[[i, j]]);
                overall[j] = sum / F::from(n_samples).unwrap();
            }

            // Compute within-class standard deviation per feature.
            let mut pooled_var = Array1::<F>::zeros(n_features);
            for (ci, &class_label) in classes.iter().enumerate() {
                let class_indices: Vec<usize> = y
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &label)| if label == class_label { Some(i) } else { None })
                    .collect();

                for j in 0..n_features {
                    let mean = centroids[[ci, j]];
                    let var_sum = class_indices.iter().fold(F::zero(), |acc, &i| {
                        let d = x[[i, j]] - mean;
                        acc + d * d
                    });
                    pooled_var[j] = pooled_var[j] + var_sum;
                }
            }

            let denom = F::from(n_samples - n_classes).unwrap().max(F::one());
            for j in 0..n_features {
                // sklearn `_nearest_centroid.py:183`: `s = np.sqrt(variance/(n_samples-n_classes))`.
                // No clamp: a constant feature yields s=0 here, which is correct.
                // The all-zero-variance guard above prevents the all-constant case
                // (where median(s)=0 would produce a zero denominator).
                pooled_var[j] = (pooled_var[j] / denom).sqrt();
            }

            // sklearn `_nearest_centroid.py:184`: `s += np.median(s)` — add the
            // median of the per-feature pooled std to every feature's std "to
            // deter outliers from affecting the results." `np.median` averages
            // the two middle elements for an even-length vector.
            let mut sorted_s: Vec<F> = pooled_var.iter().copied().collect();
            sorted_s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let median_s = if sorted_s.is_empty() {
                F::zero()
            } else if sorted_s.len() % 2 == 1 {
                sorted_s[sorted_s.len() / 2]
            } else {
                let mid = sorted_s.len() / 2;
                let two = F::one() + F::one();
                (sorted_s[mid - 1] + sorted_s[mid]) / two
            };
            for j in 0..n_features {
                pooled_var[j] = pooled_var[j] + median_s;
            }

            // Soft-threshold the deviation of each class centroid from the overall centroid.
            for ci in 0..n_classes {
                let class_indices: Vec<usize> = y
                    .iter()
                    .enumerate()
                    .filter_map(
                        |(i, &label)| {
                            if label == classes[ci] { Some(i) } else { None }
                        },
                    )
                    .collect();

                let n_c_f = F::from(class_indices.len()).unwrap();
                let m_k = (F::one() / n_c_f - F::one() / F::from(n_samples).unwrap()).sqrt();

                for j in 0..n_features {
                    let delta = (centroids[[ci, j]] - overall[j]) / (m_k * pooled_var[j]);
                    let sign = if delta > F::zero() {
                        F::one()
                    } else if delta < F::zero() {
                        -F::one()
                    } else {
                        F::zero()
                    };
                    let shrunk = (delta.abs() - threshold).max(F::zero()) * sign;
                    centroids[[ci, j]] = overall[j] + shrunk * m_k * pooled_var[j];
                }
            }
        }

        Ok(FittedNearestCentroid {
            centroids,
            classes,
            metric: self.metric,
        })
    }
}

impl<F: Float + Send + Sync + 'static> Predict<Array2<F>> for FittedNearestCentroid<F> {
    type Output = Array1<usize>;
    type Error = FerroError;

    /// Predict class labels by finding the nearest centroid.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of features
    /// does not match the fitted model.
    fn predict(&self, x: &Array2<F>) -> Result<Array1<usize>, FerroError> {
        let n_features = x.ncols();
        let n_features_fitted = self.centroids.ncols();

        if n_features != n_features_fitted {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_features_fitted],
                actual: vec![n_features],
                context: "number of features must match fitted NearestCentroid".into(),
            });
        }

        let n_samples = x.nrows();
        let n_classes = self.classes.len();
        let mut predictions = Array1::<usize>::zeros(n_samples);

        for i in 0..n_samples {
            let mut best_class = 0;
            let mut best_dist = F::infinity();

            for ci in 0..n_classes {
                // `_nearest_centroid.py:218`: `pairwise_distances_argmin(X,
                // centroids_, metric=self.metric)`. Euclidean uses (squared) L2;
                // manhattan uses L1 `Σ|x_j − c_j|`. Squaring is monotone so the
                // argmin is unchanged vs true L2 for the euclidean path.
                let dist: F = match self.metric {
                    NcMetric::Euclidean => (0..n_features)
                        .map(|j| {
                            let d = x[[i, j]] - self.centroids[[ci, j]];
                            d * d
                        })
                        .fold(F::zero(), |a, b| a + b),
                    NcMetric::Manhattan => (0..n_features)
                        .map(|j| (x[[i, j]] - self.centroids[[ci, j]]).abs())
                        .fold(F::zero(), |a, b| a + b),
                };

                if dist < best_dist {
                    best_dist = dist;
                    best_class = ci;
                }
            }

            predictions[i] = self.classes[best_class];
        }

        Ok(predictions)
    }
}

impl<F: Float + Send + Sync + 'static> HasClasses for FittedNearestCentroid<F> {
    fn classes(&self) -> &[usize] {
        &self.classes
    }

    fn n_classes(&self) -> usize {
        self.classes.len()
    }
}

impl<F: Float + Send + Sync + 'static> FittedNearestCentroid<F> {
    /// Get the class centroids.
    ///
    /// Returns an array of shape `(n_classes, n_features)`.
    #[must_use]
    pub fn centroids(&self) -> &Array2<F> {
        &self.centroids
    }

    /// Mean accuracy on the given test data and labels.
    /// Equivalent to sklearn's `ClassifierMixin.score`.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if `x.nrows() != y.len()` or
    /// the feature count does not match the training data.
    pub fn score(&self, x: &Array2<F>, y: &Array1<usize>) -> Result<F, FerroError> {
        if x.nrows() != y.len() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![x.nrows()],
                actual: vec![y.len()],
                context: "y length must match number of samples in X".into(),
            });
        }
        let preds = self.predict(x)?;
        let n = y.len();
        if n == 0 {
            return Ok(F::zero());
        }
        let correct = preds.iter().zip(y.iter()).filter(|(p, t)| p == t).count();
        Ok(F::from(correct).unwrap() / F::from(n).unwrap())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    fn make_2class_data() -> (Array2<f64>, Array1<usize>) {
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                0.0, 0.0, 0.5, 0.0, 0.0, 0.5, 0.5, 0.5, 5.0, 5.0, 5.5, 5.0, 5.0, 5.5, 5.5, 5.5,
            ],
        )
        .unwrap();
        let y = array![0usize, 0, 0, 0, 1, 1, 1, 1];
        (x, y)
    }

    #[test]
    fn test_nearest_centroid_fit_predict() {
        let (x, y) = make_2class_data();
        let model = NearestCentroid::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        let correct: usize = preds.iter().zip(y.iter()).filter(|(p, a)| p == a).count();
        assert_eq!(correct, 8);
    }

    #[test]
    fn test_nearest_centroid_centroids() {
        let (x, y) = make_2class_data();
        let model = NearestCentroid::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();
        let centroids = fitted.centroids();

        assert_eq!(centroids.nrows(), 2);
        assert_eq!(centroids.ncols(), 2);

        // Class 0: mean of (0,0), (0.5,0), (0,0.5), (0.5,0.5) = (0.25, 0.25)
        assert_relative_eq!(centroids[[0, 0]], 0.25, epsilon = 1e-10);
        assert_relative_eq!(centroids[[0, 1]], 0.25, epsilon = 1e-10);

        // Class 1: mean of (5,5), (5.5,5), (5,5.5), (5.5,5.5) = (5.25, 5.25)
        assert_relative_eq!(centroids[[1, 0]], 5.25, epsilon = 1e-10);
        assert_relative_eq!(centroids[[1, 1]], 5.25, epsilon = 1e-10);
    }

    #[test]
    fn test_nearest_centroid_has_classes() {
        let (x, y) = make_2class_data();
        let model = NearestCentroid::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();
        assert_eq!(fitted.classes(), &[0, 1]);
        assert_eq!(fitted.n_classes(), 2);
    }

    #[test]
    fn test_nearest_centroid_three_classes() {
        let x = Array2::from_shape_vec(
            (9, 2),
            vec![
                0.0, 0.0, 0.5, 0.0, 0.0, 0.5, 5.0, 0.0, 5.5, 0.0, 5.0, 0.5, 0.0, 5.0, 0.5, 5.0,
                0.0, 5.5,
            ],
        )
        .unwrap();
        let y = array![0usize, 0, 0, 1, 1, 1, 2, 2, 2];

        let model = NearestCentroid::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();
        assert_eq!(fitted.n_classes(), 3);

        let preds = fitted.predict(&x).unwrap();
        let correct: usize = preds.iter().zip(y.iter()).filter(|(p, a)| p == a).count();
        assert_eq!(correct, 9);
    }

    #[test]
    fn test_nearest_centroid_with_shrinkage() {
        let (x, y) = make_2class_data();
        let model = NearestCentroid::<f64>::new().with_shrink_threshold(0.5);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        let correct: usize = preds.iter().zip(y.iter()).filter(|(p, a)| p == a).count();
        assert!(
            correct >= 6,
            "Expected at least 6 correct with shrinkage, got {correct}"
        );
    }

    #[test]
    fn test_nearest_centroid_shrinkage_high_threshold() {
        // With a very high threshold, centroids collapse to the overall mean.
        let (x, y) = make_2class_data();
        let model = NearestCentroid::<f64>::new().with_shrink_threshold(1000.0);
        let fitted = model.fit(&x, &y).unwrap();
        let centroids = fitted.centroids();

        // Both centroids should be very close to the overall mean.
        let overall_mean_0 = (0.0 + 0.5 + 0.0 + 0.5 + 5.0 + 5.5 + 5.0 + 5.5) / 8.0;
        assert_relative_eq!(centroids[[0, 0]], overall_mean_0, epsilon = 0.1);
        assert_relative_eq!(centroids[[1, 0]], overall_mean_0, epsilon = 0.1);
    }

    #[test]
    fn test_nearest_centroid_shape_mismatch_fit() {
        let x = Array2::from_shape_vec((4, 2), vec![1.0; 8]).unwrap();
        let y = array![0usize, 1]; // Wrong length
        let model = NearestCentroid::<f64>::new();
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_nearest_centroid_shape_mismatch_predict() {
        let (x, y) = make_2class_data();
        let model = NearestCentroid::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();
        let x_bad = Array2::from_shape_vec((3, 3), vec![1.0; 9]).unwrap();
        assert!(fitted.predict(&x_bad).is_err());
    }

    #[test]
    fn test_nearest_centroid_empty_data() {
        let x = Array2::<f64>::zeros((0, 2));
        let y = Array1::<usize>::zeros(0);
        let model = NearestCentroid::<f64>::new();
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_nearest_centroid_single_class() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 1.0, 1.5, 1.0, 1.0, 1.5]).unwrap();
        let y = array![5usize, 5, 5];
        let model = NearestCentroid::<f64>::new();
        // sklearn `_nearest_centroid.py:147-151` raises ValueError on single class (R-CHAR-3).
        let result = model.fit(&x, &y);
        assert!(
            result.is_err(),
            "sklearn raises ValueError on a single class (:147-151); \
             ferrolearn must return Err, got Ok"
        );
    }

    #[test]
    fn test_nearest_centroid_default() {
        let model = NearestCentroid::<f64>::default();
        assert!(model.shrink_threshold.is_none());
    }

    #[test]
    fn test_nearest_centroid_negative_shrink_threshold() {
        let (x, y) = make_2class_data();
        let model = NearestCentroid::<f64>::new().with_shrink_threshold(-1.0);
        assert!(model.fit(&x, &y).is_err());
    }

    // R-CHAR-3 oracle: sklearn 1.5.2 `NearestCentroid`, run from /tmp on
    // X=[[1,0],[2,0],[3,1],[10,5],[11,5],[12,6]], y=[0,0,0,1,1,1]:
    //   metric='manhattan' centroids_ = [[2.0, 0.0], [11.0, 5.0]] (feature-wise median)
    //   metric='manhattan' predict([[4,1],[9,5]]) = [0, 1]
    //   default (euclidean) centroids_ = [[2.0, 0.3333333333333333],
    //                                     [11.0, 5.333333333333333]] (mean)
    fn manhattan_oracle_data() -> (Array2<f64>, Array1<usize>) {
        let x = array![
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 1.0],
            [10.0, 5.0],
            [11.0, 5.0],
            [12.0, 6.0]
        ];
        let y = array![0usize, 0, 0, 1, 1, 1];
        (x, y)
    }

    #[test]
    fn nearest_centroid_manhattan_median_centroids() -> Result<(), FerroError> {
        let (x, y) = manhattan_oracle_data();
        let fitted = NearestCentroid::<f64>::new()
            .with_metric(NcMetric::Manhattan)
            .fit(&x, &y)?;
        let centroids = fitted.centroids();

        // sklearn `_nearest_centroid.py:167` np.median(X[mask], axis=0).
        assert_relative_eq!(centroids[[0, 0]], 2.0, epsilon = 1e-9);
        assert_relative_eq!(centroids[[0, 1]], 0.0, epsilon = 1e-9);
        assert_relative_eq!(centroids[[1, 0]], 11.0, epsilon = 1e-9);
        assert_relative_eq!(centroids[[1, 1]], 5.0, epsilon = 1e-9);
        Ok(())
    }

    #[test]
    fn nearest_centroid_manhattan_predict_matches_sklearn() -> Result<(), FerroError> {
        let (x, y) = manhattan_oracle_data();
        let fitted = NearestCentroid::<f64>::new()
            .with_metric(NcMetric::Manhattan)
            .fit(&x, &y)?;

        let query = array![[4.0, 1.0], [9.0, 5.0]];
        let preds = fitted.predict(&query)?;
        // sklearn `_nearest_centroid.py:218` manhattan pairwise_distances_argmin.
        assert_eq!(preds[0], 0);
        assert_eq!(preds[1], 1);
        Ok(())
    }

    #[test]
    fn nearest_centroid_euclidean_default_unchanged() -> Result<(), FerroError> {
        let (x, y) = manhattan_oracle_data();
        // Default metric is Euclidean (sklearn `_nearest_centroid.py:108`).
        let fitted = NearestCentroid::<f64>::new().fit(&x, &y)?;
        let centroids = fitted.centroids();

        // sklearn `_nearest_centroid.py:171` X[mask].mean(axis=0) — the per-class
        // mean path is unchanged by this commit (≤1e-12).
        assert_relative_eq!(centroids[[0, 0]], 2.0, epsilon = 1e-12);
        assert_relative_eq!(centroids[[0, 1]], 1.0 / 3.0, epsilon = 1e-12);
        assert_relative_eq!(centroids[[1, 0]], 11.0, epsilon = 1e-12);
        assert_relative_eq!(centroids[[1, 1]], 16.0 / 3.0, epsilon = 1e-12);
        Ok(())
    }

    #[test]
    fn test_nearest_centroid_noncontiguous_labels() {
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![0.0, 0.0, 0.5, 0.0, 0.0, 0.5, 5.0, 5.0, 5.5, 5.0, 5.0, 5.5],
        )
        .unwrap();
        let y = array![10usize, 10, 10, 20, 20, 20];

        let model = NearestCentroid::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();
        assert_eq!(fitted.classes(), &[10, 20]);

        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds[0], 10);
        assert_eq!(preds[5], 20);
    }
}
