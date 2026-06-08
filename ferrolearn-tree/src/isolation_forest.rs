//! Isolation forest anomaly detection.
//!
//! This module provides [`IsolationForest`], an unsupervised anomaly detection
//! algorithm that isolates observations by randomly selecting features and split
//! points. Anomalies are the points that require fewer random splits to isolate,
//! resulting in shorter average path lengths across the ensemble of isolation trees.
//!
//! # Examples
//!
//! ```
//! use ferrolearn_tree::IsolationForest;
//! use ferrolearn_core::{Fit, Predict};
//! use ndarray::Array2;
//!
//! let x = Array2::from_shape_vec((8, 2), vec![
//!     1.0, 2.0,  2.0, 3.0,  3.0, 3.0,  4.0, 4.0,
//!     5.0, 6.0,  6.0, 7.0,  7.0, 8.0,  100.0, 100.0,
//! ]).unwrap();
//!
//! let model = IsolationForest::<f64>::new()
//!     .with_n_estimators(100)
//!     .with_contamination(0.1)
//!     .with_random_state(42);
//! let fitted = model.fit(&x, &()).unwrap();
//! let preds = fitted.predict(&x).unwrap();
//! // Normal points get 1, anomalies get -1
//! assert!(preds.iter().all(|&v| v == 1 || v == -1));
//! ```
//!
//! ## REQ status
//!
//! Mirrors `sklearn.ensemble.IsolationForest` (`sklearn/ensemble/_iforest.py`).
//! See `.design/tree/isolation_forest.md`. Non-test consumer: crate re-export
//! + pipeline adapter (no PyO3 binding).
//!
//! **RNG boundary:** subsample + split draws use `StdRng` where sklearn draws
//! from numpy MT19937 — exact tree/score-value parity is infeasible (#730);
//! ferrolearn's fit is internally reproducible.
//!
//! | REQ | Description | Status |
//! |-----|-------------|--------|
//! | REQ-1 | Param defaults: `n_estimators=100`, `max_samples` effective `min(256,n)` == sklearn `'auto'`, `contamination=Auto`, `random_state=None` | SHIPPED |
//! | REQ-2 | Isolation-tree build + `max_depth=ceil(log2(max_samples))` (structural; RNG boundary) | SHIPPED |
//! | REQ-3 | `c(n)` average path length incl. `n<=1→0`, `n==2→1.0` special-cases (`_iforest.py:558-562`) | SHIPPED |
//! | REQ-4 | `score_samples = -2^(-mean/c)` ∈ [-1,0], higher = normal (`_iforest.py:451`) | SHIPPED |
//! | REQ-5 | `decision_function = score_samples - offset_` (`_iforest.py:410`) | SHIPPED |
//! | REQ-6 | `offset_` = `-0.5` for `Contamination::Auto`, else numpy-percentile of train scores (`_iforest.py:341-353`); `Contamination{Auto,Value}` enum | SHIPPED |
//! | REQ-8 | `predict` = `-1 where decision_function(X) < 0 else 1` (`_iforest.py:374-378`) | SHIPPED |
//! | REQ-9 | `random_state` reproducibility (ferrolearn-internal; numpy-MT parity = RNG boundary #730) | SHIPPED |
//! | REQ-7a | `max_features` + `bootstrap` params + `max_samples` int/'auto'-string representation | NOT-STARTED (#728) |
//! | REQ-7b | Subsample WITHOUT replacement (`bootstrap=False`) | NOT-STARTED (#729) |
//! | REQ-10 | ferray substrate migration | NOT-STARTED (#731) |
//! | REQ-11 | Reject non-finite input (NaN+Inf): `fn reject_non_finite` at the top of `IsolationForest::fit` rejects NaN AND infinity. sklearn validates X up front (`_iforest.py:291`, default `force_all_finite=True`) BEFORE any isolation tree ⇒ `ValueError`; the `ExtraTreeRegressor(splitter='random')` base learner has no missing-value support. Consumer: the existing `fit` entry (crate-root re-export + pipeline adapter). Pinned by `divergence_isolation_forest_nan_not_rejected` (live sklearn 1.5.2 raises). | SHIPPED |

use ferrolearn_core::error::FerroError;
use ferrolearn_core::traits::{Fit, Predict};
use ndarray::{Array1, Array2};
use num_traits::Float;
use rand::SeedableRng;
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};

/// Reject `X` containing any non-finite value (NaN or infinity).
///
/// sklearn's `IsolationForest.fit` validates X up front via
/// `_validate_data(X, accept_sparse=["csc"], dtype=tree_dtype)` with the default
/// `force_all_finite=True` (`sklearn/ensemble/_iforest.py:291`), raising
/// `ValueError("Input X contains NaN.")` (`validation.py:147-154`) BEFORE any
/// isolation tree is built. The `ExtraTreeRegressor(splitter='random')` base
/// learner does NOT support missing values, so NaN AND infinity are both
/// rejected. Never panics (R-CODE-2).
fn reject_non_finite<F: Float>(x: &Array2<F>) -> Result<(), FerroError> {
    if x.iter().any(|v| !v.is_finite()) {
        return Err(FerroError::InvalidParameter {
            name: "X".into(),
            reason: "Input X contains NaN or infinity.".into(),
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// IsolationTree node representation
// ---------------------------------------------------------------------------

/// A single node in an isolation tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
enum IsoNode<F> {
    /// An internal split node.
    Split {
        /// Feature index used for the split.
        feature: usize,
        /// Split threshold; samples with `x[feature] <= threshold` go left.
        threshold: F,
        /// Index of the left child in the flat node vector.
        left: usize,
        /// Index of the right child in the flat node vector.
        right: usize,
        /// Number of samples that reached this node during training.
        n_samples: usize,
    },
    /// An external (leaf) node.
    Leaf {
        /// Number of samples that reached this node during training.
        n_samples: usize,
    },
}

// ---------------------------------------------------------------------------
// Contamination
// ---------------------------------------------------------------------------

/// The expected proportion of outliers in the data, used to set `offset_`.
///
/// Mirrors scikit-learn's `contamination` parameter, which is either the
/// string `'auto'` or a float in `(0, 0.5]`
/// (`sklearn/ensemble/_iforest.py:199` `_parameter_constraints`,
/// `:221` `__init__` default `contamination="auto"`).
///
/// - [`Contamination::Auto`] reproduces sklearn's default `'auto'` path:
///   `offset_ = -0.5`, the threshold from the original isolation-forest paper
///   (`_iforest.py:341-345`).
/// - [`Contamination::Value`] reproduces the numeric path:
///   `offset_ = np.percentile(score_samples(X_train), 100 * contamination)`
///   (`_iforest.py:353`).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Contamination {
    /// sklearn `contamination='auto'`: use the paper threshold `offset_ = -0.5`.
    Auto,
    /// sklearn `contamination=<float>`: `offset_` is the `100*value` percentile
    /// of the training scores. `value` must be in `(0.0, 0.5]`.
    Value(f64),
}

// ---------------------------------------------------------------------------
// IsolationForest
// ---------------------------------------------------------------------------

/// Isolation forest anomaly detector.
///
/// Builds an ensemble of isolation trees, each trained on a random subsample.
/// Anomaly scores are derived from the average path length: points that are
/// isolated in fewer splits are more anomalous.
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IsolationForest<F> {
    /// Number of isolation trees in the forest.
    pub n_estimators: usize,
    /// Number of samples to draw for each tree.
    pub max_samples: usize,
    /// Contamination parameter controlling `offset_` (sklearn `contamination`,
    /// `_iforest.py:221`). Defaults to [`Contamination::Auto`].
    pub contamination: Contamination,
    /// Random seed for reproducibility. `None` means non-deterministic.
    pub random_state: Option<u64>,
    _marker: std::marker::PhantomData<F>,
}

impl<F: Float> IsolationForest<F> {
    /// Create a new `IsolationForest` with default settings.
    ///
    /// Defaults: `n_estimators = 100`, `max_samples = 256`,
    /// `contamination = Contamination::Auto`, `random_state = None`
    /// (sklearn `__init__`, `_iforest.py:221`, default `contamination="auto"`).
    #[must_use]
    pub fn new() -> Self {
        Self {
            n_estimators: 100,
            max_samples: 256,
            contamination: Contamination::Auto,
            random_state: None,
            _marker: std::marker::PhantomData,
        }
    }

    /// Set the number of isolation trees.
    #[must_use]
    pub fn with_n_estimators(mut self, n_estimators: usize) -> Self {
        self.n_estimators = n_estimators;
        self
    }

    /// Set the number of samples to draw for each tree.
    #[must_use]
    pub fn with_max_samples(mut self, max_samples: usize) -> Self {
        self.max_samples = max_samples;
        self
    }

    /// Set the contamination fraction (proportion of anomalies) as
    /// [`Contamination::Value`].
    ///
    /// `contamination` must be in `(0.0, 0.5]`; an out-of-range value is
    /// rejected with [`FerroError::InvalidParameter`] at `fit` time (sklearn
    /// `_parameter_constraints` `Interval(Real, 0, 0.5, closed="right")`,
    /// `_iforest.py:199`).
    #[must_use]
    pub fn with_contamination(mut self, contamination: f64) -> Self {
        self.contamination = Contamination::Value(contamination);
        self
    }

    /// Set the contamination to [`Contamination::Auto`] (sklearn
    /// `contamination='auto'`, `offset_ = -0.5`, `_iforest.py:341-345`).
    #[must_use]
    pub fn with_contamination_auto(mut self) -> Self {
        self.contamination = Contamination::Auto;
        self
    }

    /// Set the random seed for reproducibility.
    #[must_use]
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }
}

impl<F: Float> Default for IsolationForest<F> {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// FittedIsolationForest
// ---------------------------------------------------------------------------

/// A fitted isolation forest anomaly detector.
///
/// Stores the ensemble of isolation trees and the decision offset `offset_`
/// derived from the contamination parameter (sklearn `offset_`,
/// `_iforest.py:341-353`).
#[derive(Debug, Clone)]
pub struct FittedIsolationForest<F> {
    /// Individual isolation trees, each stored as a flat node vector.
    trees: Vec<Vec<IsoNode<F>>>,
    /// Number of features the model was trained on.
    n_features: usize,
    /// Decision offset: `decision_function(X) = score_samples(X) - offset_`;
    /// a sample is an outlier (`-1`) when `decision_function(X) < 0`
    /// (sklearn `offset_`, `_iforest.py:341-353`).
    offset_: f64,
    /// Effective number of samples used per tree.
    max_samples: usize,
}

impl<F: Float + Send + Sync + 'static> FittedIsolationForest<F> {
    /// Returns the number of isolation trees.
    #[must_use]
    pub fn n_estimators(&self) -> usize {
        self.trees.len()
    }

    /// Returns the number of features the model was trained on.
    #[must_use]
    pub fn n_features(&self) -> usize {
        self.n_features
    }

    /// Returns the decision offset `offset_`.
    ///
    /// `decision_function(X) = score_samples(X) - offset_`; samples with a
    /// negative decision function are outliers (sklearn `offset_`,
    /// `_iforest.py:341-353`).
    #[must_use]
    pub fn offset(&self) -> f64 {
        self.offset_
    }

    /// Compute the opposite of the paper anomaly score for each sample.
    ///
    /// Returns `-2^(-mean_path_length / c(max_samples))`, where `c(n)` is the
    /// average path length of an unsuccessful search in a binary search tree.
    /// This mirrors sklearn's sign convention (`_score_samples =
    /// -_compute_chunked_score_samples`, `_iforest.py:451`): scores lie in
    /// `[-1, 0]`, where **higher (closer to 0) means more NORMAL** and lower
    /// (more negative) means more anomalous. "The lower, the more abnormal.
    /// Negative scores represent outliers, positive scores represent inliers"
    /// (`_iforest.py:404-405`).
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of features does
    /// not match the training data.
    pub fn score_samples(&self, x: &Array2<F>) -> Result<Array1<f64>, FerroError> {
        if x.ncols() != self.n_features {
            return Err(FerroError::ShapeMismatch {
                expected: vec![self.n_features],
                actual: vec![x.ncols()],
                context: "number of features must match fitted model".into(),
            });
        }

        let n_samples = x.nrows();
        let c_n = average_path_length(self.max_samples);
        let n_trees = self.trees.len() as f64;
        let mut scores = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let row = x.row(i);
            let mut total_path = 0.0;
            for tree_nodes in &self.trees {
                total_path += path_length(tree_nodes, &row);
            }
            let mean_path = total_path / n_trees;
            // Guard the division: when c_n == 0 (max_samples <= 1,
            // average_path_length(1) == 0), sklearn's np.divide with
            // `out=ones, where=denominator != 0` keeps the ratio at 1.0
            // so the score is -2^(-1) = -0.5 (_iforest.py:519-522).
            let ratio = if c_n != 0.0 { mean_path / c_n } else { 1.0 };
            // Take the opposite of the paper score so that "bigger is better"
            // (less abnormal), matching sklearn (_iforest.py:451).
            scores[i] = -f64::powf(2.0, -ratio);
        }

        Ok(scores)
    }

    /// Compute the anomaly decision function for each sample.
    ///
    /// `decision_function(X) = score_samples(X) - offset_` (sklearn
    /// `_iforest.py:410`). Subtracting `offset_` makes `0` the outlier
    /// threshold: a sample is an outlier when the result is `< 0`. "The lower,
    /// the more abnormal. Negative scores represent outliers, positive scores
    /// represent inliers" (`_iforest.py:404-405`).
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of features does
    /// not match the training data.
    pub fn decision_function(&self, x: &Array2<F>) -> Result<Array1<f64>, FerroError> {
        let scores = self.score_samples(x)?;
        Ok(scores.mapv(|s| s - self.offset_))
    }
}

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, ()> for IsolationForest<F> {
    type Fitted = FittedIsolationForest<F>;
    type Error = FerroError;

    /// Fit the isolation forest on the training data.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::InsufficientSamples`] if there are no samples.
    /// Returns [`FerroError::InvalidParameter`] if hyperparameters are invalid.
    fn fit(&self, x: &Array2<F>, _y: &()) -> Result<FittedIsolationForest<F>, FerroError> {
        let (n_samples, n_features) = x.dim();

        if n_samples == 0 {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "IsolationForest requires at least one sample".into(),
            });
        }
        if self.n_estimators == 0 {
            return Err(FerroError::InvalidParameter {
                name: "n_estimators".into(),
                reason: "must be at least 1".into(),
            });
        }
        if self.max_samples == 0 {
            return Err(FerroError::InvalidParameter {
                name: "max_samples".into(),
                reason: "must be at least 1".into(),
            });
        }
        if let Contamination::Value(v) = self.contamination {
            // sklearn `_parameter_constraints`: Interval(Real, 0, 0.5,
            // closed="right") (_iforest.py:199) — 0 < v <= 0.5.
            if !(v > 0.0 && v <= 0.5) {
                return Err(FerroError::InvalidParameter {
                    name: "contamination".into(),
                    reason: "must be in (0.0, 0.5]".into(),
                });
            }
        }
        // Reject non-finite X up front, before building any isolation tree,
        // matching sklearn (`_iforest.py:291`).
        reject_non_finite(x)?;

        let effective_max_samples = self.max_samples.min(n_samples);
        let max_depth = (effective_max_samples as f64).log2().ceil() as usize;

        let mut rng = if let Some(seed) = self.random_state {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::from_os_rng()
        };

        let mut trees = Vec::with_capacity(self.n_estimators);
        for _ in 0..self.n_estimators {
            // Sample indices with replacement.
            let sample_indices: Vec<usize> = (0..effective_max_samples)
                .map(|_| {
                    use rand::RngCore;
                    (rng.next_u64() as usize) % n_samples
                })
                .collect();

            let mut nodes = Vec::new();
            let indices: Vec<usize> = (0..sample_indices.len()).collect();
            // Build a view of the subsampled data.
            build_isolation_tree(
                x,
                &sample_indices,
                &indices,
                &mut nodes,
                0,
                max_depth,
                n_features,
                &mut rng,
            );
            trees.push(nodes);
        }

        // Build a provisional fitted model (offset_ filled in below) so we can
        // score the training rows to derive offset_.
        let mut fitted = FittedIsolationForest {
            trees,
            n_features,
            offset_: 0.0,
            max_samples: effective_max_samples,
        };

        // offset_ (sklearn _iforest.py:341-353).
        fitted.offset_ = match self.contamination {
            // contamination == "auto": the paper threshold, on the opposite
            // (negated) score convention (_iforest.py:341-345).
            Contamination::Auto => -0.5,
            // Else: np.percentile(_score_samples(X), 100 * contamination)
            // (_iforest.py:353).
            Contamination::Value(v) => {
                let train_scores = fitted.score_samples(x)?;
                percentile(train_scores.as_slice().unwrap_or(&[]), 100.0 * v)
            }
        };

        Ok(fitted)
    }
}

/// numpy `np.percentile(a, q)` with the default `'linear'` interpolation.
///
/// Sorts `a` ascending and returns the value at fractional rank
/// `q/100 * (n - 1)`, linearly interpolating between the floor and ceil ranks
/// (numpy's `_lerp` default, mirrored to match sklearn's `offset_` computation
/// `np.percentile(..., 100*contamination)`, `_iforest.py:353`). For an empty
/// input it returns `0.0` (sklearn never calls this on an empty array, as `fit`
/// rejects zero-sample input first).
fn percentile(a: &[f64], q: f64) -> f64 {
    let n = a.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return a[0];
    }
    let mut sorted: Vec<f64> = a.to_vec();
    sorted.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));

    let rank = (q / 100.0) * ((n - 1) as f64);
    let lo = rank.floor();
    let hi = rank.ceil();
    let lo_idx = (lo as usize).min(n - 1);
    let hi_idx = (hi as usize).min(n - 1);
    if lo_idx == hi_idx {
        return sorted[lo_idx];
    }
    let frac = rank - lo;
    sorted[lo_idx] + (sorted[hi_idx] - sorted[lo_idx]) * frac
}

impl<F: Float + Send + Sync + 'static> Predict<Array2<F>> for FittedIsolationForest<F> {
    type Output = Array1<isize>;
    type Error = FerroError;

    /// Predict anomaly labels for the given feature matrix.
    ///
    /// Returns `1` for inliers and `-1` for outliers, where a sample is an
    /// outlier when `decision_function(X) < 0` (sklearn `predict`:
    /// `is_inlier = ones; is_inlier[decision_function(X) < 0] = -1`,
    /// `_iforest.py:374-378`).
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of features does
    /// not match the fitted model.
    fn predict(&self, x: &Array2<F>) -> Result<Array1<isize>, FerroError> {
        let decision = self.decision_function(x)?;
        let predictions = decision.mapv(|d| if d < 0.0 { -1 } else { 1 });
        Ok(predictions)
    }
}

// ---------------------------------------------------------------------------
// Internal: isolation tree building
// ---------------------------------------------------------------------------

/// Average path length of an unsuccessful search in a BST with `n` elements.
///
/// Mirrors sklearn `_average_path_length` (`_iforest.py:557-566`):
/// `c(n) = 0` for `n <= 1`, `c(2) = 1`, else
/// `2 * (ln(n-1) + euler_gamma) - 2*(n-1)/n`.
fn average_path_length(n: usize) -> f64 {
    if n <= 1 {
        // mask_1: n_samples_leaf <= 1 -> 0.0 (_iforest.py:557, :561)
        return 0.0;
    }
    if n == 2 {
        // mask_2: n_samples_leaf == 2 -> 1.0 (_iforest.py:558, :562)
        return 1.0;
    }
    let n_f = n as f64;
    // Euler-Mascheroni constant == np.euler_gamma
    2.0 * ((n_f - 1.0).ln() + 0.5772156649015329) - 2.0 * (n_f - 1.0) / n_f
}

/// Compute the path length for a single sample through an isolation tree.
fn path_length<F: Float>(nodes: &[IsoNode<F>], sample: &ndarray::ArrayView1<F>) -> f64 {
    let mut idx = 0;
    let mut depth: f64 = 0.0;
    loop {
        match &nodes[idx] {
            IsoNode::Split {
                feature,
                threshold,
                left,
                right,
                ..
            } => {
                if sample[*feature] <= *threshold {
                    idx = *left;
                } else {
                    idx = *right;
                }
                depth += 1.0;
            }
            IsoNode::Leaf { n_samples } => {
                // Add the expected path length for the remaining samples.
                return depth + average_path_length(*n_samples);
            }
        }
    }
}

/// Generate a uniform random float in `[min_val, max_val]`.
fn random_threshold<F: Float>(rng: &mut StdRng, min_val: F, max_val: F) -> F {
    use rand::RngCore;
    let u = (rng.next_u64() as f64) / (u64::MAX as f64);
    let range = max_val - min_val;
    min_val + F::from(u).unwrap() * range
}

/// Build an isolation tree recursively.
///
/// `sample_indices` maps local indices to rows in `x`.
/// `indices` are the local indices of points currently in this node.
#[allow(clippy::too_many_arguments)]
fn build_isolation_tree<F: Float>(
    x: &Array2<F>,
    sample_indices: &[usize],
    indices: &[usize],
    nodes: &mut Vec<IsoNode<F>>,
    depth: usize,
    max_depth: usize,
    n_features: usize,
    rng: &mut StdRng,
) -> usize {
    let n = indices.len();

    // Stop if: only one sample, or max depth reached.
    if n <= 1 || depth >= max_depth {
        let idx = nodes.len();
        nodes.push(IsoNode::Leaf { n_samples: n });
        return idx;
    }

    // Try random features until we find one that can split, or exhaust attempts.
    let max_attempts = n_features * 2;
    for _ in 0..max_attempts {
        use rand::RngCore;
        let feature = (rng.next_u64() as usize) % n_features;

        // Find min and max of this feature for the current indices.
        let mut min_val = x[[sample_indices[indices[0]], feature]];
        let mut max_val = min_val;
        for &i in &indices[1..] {
            let v = x[[sample_indices[i], feature]];
            if v < min_val {
                min_val = v;
            }
            if v > max_val {
                max_val = v;
            }
        }

        // If all values are the same for this feature, try another feature.
        if min_val >= max_val {
            continue;
        }

        let threshold = random_threshold(rng, min_val, max_val);

        // Partition indices.
        let mut left_indices = Vec::new();
        let mut right_indices = Vec::new();
        for &i in indices {
            if x[[sample_indices[i], feature]] <= threshold {
                left_indices.push(i);
            } else {
                right_indices.push(i);
            }
        }

        // If the split is degenerate (all on one side), try again.
        if left_indices.is_empty() || right_indices.is_empty() {
            continue;
        }

        // Reserve a slot for this node.
        let node_idx = nodes.len();
        nodes.push(IsoNode::Leaf { n_samples: 0 }); // placeholder

        let left_child = build_isolation_tree(
            x,
            sample_indices,
            &left_indices,
            nodes,
            depth + 1,
            max_depth,
            n_features,
            rng,
        );
        let right_child = build_isolation_tree(
            x,
            sample_indices,
            &right_indices,
            nodes,
            depth + 1,
            max_depth,
            n_features,
            rng,
        );

        nodes[node_idx] = IsoNode::Split {
            feature,
            threshold,
            left: left_child,
            right: right_child,
            n_samples: n,
        };

        return node_idx;
    }

    // Could not find a splittable feature — make this a leaf.
    let idx = nodes.len();
    nodes.push(IsoNode::Leaf { n_samples: n });
    idx
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn make_normal_data() -> Array2<f64> {
        // 10 normal points clustered around (5, 5)
        Array2::from_shape_vec(
            (10, 2),
            vec![
                4.5, 4.8, 5.1, 5.2, 4.9, 5.0, 5.3, 4.7, 4.8, 5.1, 5.0, 5.3, 5.2, 4.9, 4.7, 5.0,
                5.1, 4.8, 4.9, 5.2,
            ],
        )
        .unwrap()
    }

    fn make_data_with_anomaly() -> Array2<f64> {
        // 9 normal points + 1 clear anomaly at (100, 100)
        Array2::from_shape_vec(
            (10, 2),
            vec![
                4.5, 4.8, 5.1, 5.2, 4.9, 5.0, 5.3, 4.7, 4.8, 5.1, 5.0, 5.3, 5.2, 4.9, 4.7, 5.0,
                5.1, 4.8, 100.0, 100.0,
            ],
        )
        .unwrap()
    }

    #[test]
    fn test_isolation_forest_default() {
        let model = IsolationForest::<f64>::new();
        assert_eq!(model.n_estimators, 100);
        assert_eq!(model.max_samples, 256);
        // sklearn default contamination='auto' (_iforest.py:221).
        assert_eq!(model.contamination, Contamination::Auto);
        assert!(model.random_state.is_none());
    }

    #[test]
    fn test_isolation_forest_builder() {
        let model = IsolationForest::<f64>::new()
            .with_n_estimators(50)
            .with_max_samples(128)
            .with_contamination(0.05)
            .with_random_state(123);
        assert_eq!(model.n_estimators, 50);
        assert_eq!(model.max_samples, 128);
        assert_eq!(model.contamination, Contamination::Value(0.05));
        assert_eq!(model.random_state, Some(123));
    }

    #[test]
    fn test_contamination_auto_builder() {
        let model = IsolationForest::<f64>::new()
            .with_contamination(0.2)
            .with_contamination_auto();
        assert_eq!(model.contamination, Contamination::Auto);
    }

    #[test]
    fn test_fit_predict_basic() {
        let x = make_normal_data();
        let model = IsolationForest::<f64>::new()
            .with_n_estimators(50)
            .with_random_state(42);
        let fitted = model.fit(&x, &()).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 10);
        // All predictions should be either 1 or -1
        assert!(preds.iter().all(|&v| v == 1 || v == -1));
    }

    #[test]
    fn test_anomaly_detected() {
        let x = make_data_with_anomaly();
        let model = IsolationForest::<f64>::new()
            .with_n_estimators(200)
            .with_contamination(0.15)
            .with_random_state(42);
        let fitted = model.fit(&x, &()).unwrap();
        let preds = fitted.predict(&x).unwrap();

        // The last point (100, 100) should be flagged as anomaly (-1)
        assert_eq!(preds[9], -1, "outlier should be detected as anomaly");
    }

    #[test]
    fn test_anomaly_scores() {
        let x = make_data_with_anomaly();
        let model = IsolationForest::<f64>::new()
            .with_n_estimators(200)
            .with_random_state(42);
        let fitted = model.fit(&x, &()).unwrap();
        let scores = fitted.score_samples(&x).unwrap();

        assert_eq!(scores.len(), 10);
        // sklearn sign convention (_iforest.py:451, :404-405): score_samples
        // is the OPPOSITE of the paper score, so the anomaly (last point) has
        // the LOWEST (most negative) score; all scores are <= 0.
        assert!(scores.iter().all(|&s| s <= 0.0), "scores must be <= 0");
        let anomaly_score = scores[9];
        let min_normal_score = scores.iter().take(9).copied().fold(f64::INFINITY, f64::min);
        assert!(
            anomaly_score < min_normal_score,
            "anomaly score ({anomaly_score}) should be less than min normal score ({min_normal_score})"
        );
    }

    #[test]
    fn test_empty_input_error() {
        let x = Array2::<f64>::zeros((0, 2));
        let model = IsolationForest::<f64>::new();
        let result = model.fit(&x, &());
        assert!(result.is_err());
    }

    #[test]
    fn test_zero_estimators_error() {
        let x = make_normal_data();
        let model = IsolationForest::<f64>::new().with_n_estimators(0);
        let result = model.fit(&x, &());
        assert!(result.is_err());
    }

    #[test]
    fn test_zero_max_samples_error() {
        let x = make_normal_data();
        let model = IsolationForest::<f64>::new().with_max_samples(0);
        let result = model.fit(&x, &());
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_contamination_error() {
        let x = make_normal_data();
        let model = IsolationForest::<f64>::new().with_contamination(0.6);
        let result = model.fit(&x, &());
        assert!(result.is_err());
    }

    #[test]
    fn test_predict_shape_mismatch() {
        let x_train = make_normal_data();
        let model = IsolationForest::<f64>::new()
            .with_n_estimators(10)
            .with_random_state(42);
        let fitted = model.fit(&x_train, &()).unwrap();

        let x_test = Array2::<f64>::zeros((3, 5)); // wrong number of features
        let result = fitted.predict(&x_test);
        assert!(result.is_err());
    }

    #[test]
    fn test_score_shape_mismatch() {
        let x_train = make_normal_data();
        let model = IsolationForest::<f64>::new()
            .with_n_estimators(10)
            .with_random_state(42);
        let fitted = model.fit(&x_train, &()).unwrap();

        let x_test = Array2::<f64>::zeros((3, 5));
        let result = fitted.score_samples(&x_test);
        assert!(result.is_err());
    }

    #[test]
    fn test_average_path_length_values() {
        // sklearn _average_path_length special-cases (_iforest.py:561-562):
        // mask_1 (n<=1) -> 0.0, mask_2 (n==2) -> 1.0.
        // live: `_average_path_length([1]) == 0.`, `_average_path_length([2]) == 1.`
        assert!((average_path_length(1) - 0.0).abs() < 1e-10);
        assert!((average_path_length(2) - 1.0).abs() < 1e-10);
        // General branch (_iforest.py:563-566): live `_average_path_length([256])
        // == 10.2447709...`.
        let c256 = average_path_length(256);
        assert!((c256 - 10.244770920119917).abs() < 1e-9, "c(256) = {c256}");
    }

    #[test]
    fn test_reproducibility() {
        let x = make_data_with_anomaly();
        let model = IsolationForest::<f64>::new()
            .with_n_estimators(50)
            .with_random_state(999);

        let fitted1 = model.fit(&x, &()).unwrap();
        let preds1 = fitted1.predict(&x).unwrap();

        let fitted2 = model.fit(&x, &()).unwrap();
        let preds2 = fitted2.predict(&x).unwrap();

        assert_eq!(preds1, preds2);
    }

    #[test]
    fn test_max_samples_larger_than_data() {
        // max_samples > n_samples should still work (clamped to n_samples).
        let x = make_normal_data();
        let model = IsolationForest::<f64>::new()
            .with_n_estimators(10)
            .with_max_samples(10000)
            .with_random_state(42);
        let fitted = model.fit(&x, &()).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 10);
    }

    #[test]
    fn test_f32() {
        let x = Array2::<f32>::from_shape_vec(
            (6, 2),
            vec![
                1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 5.0, 6.0, 6.0, 7.0, 100.0, 100.0,
            ],
        )
        .unwrap();
        let model = IsolationForest::<f32>::new()
            .with_n_estimators(50)
            .with_contamination(0.2)
            .with_random_state(42);
        let fitted = model.fit(&x, &()).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 6);
        assert!(preds.iter().all(|&v| v == 1 || v == -1));
    }

    #[test]
    fn test_single_sample() {
        let x = Array2::<f64>::from_shape_vec((1, 3), vec![1.0, 2.0, 3.0]).unwrap();
        let model = IsolationForest::<f64>::new()
            .with_n_estimators(10)
            // contamination=0.0 is invalid (sklearn Interval(Real, 0, 0.5,
            // closed="right"), _iforest.py:199); use the 'auto' path instead.
            .with_contamination_auto()
            .with_random_state(42);
        let fitted = model.fit(&x, &()).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 1);
    }

    #[test]
    fn test_fitted_accessors() {
        let x = make_normal_data();
        let model = IsolationForest::<f64>::new()
            .with_n_estimators(10)
            .with_random_state(42);
        let fitted = model.fit(&x, &()).unwrap();
        assert_eq!(fitted.n_estimators(), 10);
        assert_eq!(fitted.n_features(), 2);
        // Default contamination='auto' => offset_ == -0.5 (_iforest.py:341-345).
        assert!((fitted.offset() - (-0.5)).abs() < 1e-12);
    }

    // -----------------------------------------------------------------------
    // Scoring-contract tests (REQ-4/5/6/8). Expected values come from sklearn
    // constants/formulas (R-CHAR-3), never copied from ferrolearn output.
    // These tests return `Result` and use `?` so no test-local `.unwrap()` is
    // introduced.
    // -----------------------------------------------------------------------

    #[test]
    fn test_score_samples_sign_in_minus_one_zero() -> Result<(), FerroError> {
        // sklearn _score_samples = -_compute_chunked_score_samples
        // (_iforest.py:451): every score is in [-1, 0]. (REQ-4)
        let x = make_data_with_anomaly();
        let model = IsolationForest::<f64>::new()
            .with_n_estimators(50)
            .with_random_state(42);
        let fitted = model.fit(&x, &())?;
        let scores = fitted.score_samples(&x)?;
        assert!(scores.iter().all(|&s| (-1.0..=0.0).contains(&s)));
        Ok(())
    }

    #[test]
    fn test_offset_auto_is_minus_half() -> Result<(), FerroError> {
        // contamination='auto' => offset_ == -0.5 (_iforest.py:341-345). (REQ-6)
        let x = make_data_with_anomaly();
        let model = IsolationForest::<f64>::new()
            .with_n_estimators(50)
            .with_contamination_auto()
            .with_random_state(42);
        let fitted = model.fit(&x, &())?;
        assert!((fitted.offset() - (-0.5)).abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn test_decision_function_equals_score_minus_offset() -> Result<(), FerroError> {
        // decision_function(X) == score_samples(X) - offset_ (_iforest.py:410).
        // (REQ-5)
        let x = make_data_with_anomaly();
        let model = IsolationForest::<f64>::new()
            .with_n_estimators(80)
            .with_contamination(0.2)
            .with_random_state(42);
        let fitted = model.fit(&x, &())?;
        let scores = fitted.score_samples(&x)?;
        let decision = fitted.decision_function(&x)?;
        let off = fitted.offset();
        for i in 0..x.nrows() {
            assert!((decision[i] - (scores[i] - off)).abs() < 1e-12);
        }
        Ok(())
    }

    #[test]
    fn test_predict_agrees_with_decision_function_sign() -> Result<(), FerroError> {
        // sklearn predict: -1 where decision_function(X) < 0 else 1
        // (_iforest.py:374-378). (REQ-8)
        let x = make_data_with_anomaly();
        let model = IsolationForest::<f64>::new()
            .with_n_estimators(80)
            .with_contamination(0.2)
            .with_random_state(42);
        let fitted = model.fit(&x, &())?;
        let decision = fitted.decision_function(&x)?;
        let preds = fitted.predict(&x)?;
        for i in 0..x.nrows() {
            let expected = if decision[i] < 0.0 { -1 } else { 1 };
            assert_eq!(preds[i], expected);
        }
        Ok(())
    }

    #[test]
    fn test_offset_value_is_percentile_of_scores() -> Result<(), FerroError> {
        // contamination=v => offset_ == np.percentile(score_samples(X), 100*v)
        // (_iforest.py:353), numpy default linear interpolation. (REQ-6)
        let x = make_data_with_anomaly();
        let model = IsolationForest::<f64>::new()
            .with_n_estimators(80)
            .with_contamination(0.2)
            .with_random_state(42);
        let fitted = model.fit(&x, &())?;
        let scores = fitted.score_samples(&x)?;
        let slice = scores.as_slice().unwrap_or(&[]);
        let expected = percentile(slice, 20.0);
        assert!((fitted.offset() - expected).abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn test_percentile_linear_interpolation() {
        // numpy np.percentile default 'linear': rank = q/100*(n-1), interp
        // between floor/ceil. Live values:
        //   np.percentile([1,2,3,4], 0.0)   == 1.0
        //   np.percentile([1,2,3,4], 25.0)  == 1.75
        //   np.percentile([1,2,3,4], 50.0)  == 2.5
        //   np.percentile([1,2,3,4], 100.0) == 4.0
        let a = [1.0, 2.0, 3.0, 4.0];
        assert!((percentile(&a, 0.0) - 1.0).abs() < 1e-12);
        assert!((percentile(&a, 25.0) - 1.75).abs() < 1e-12);
        assert!((percentile(&a, 50.0) - 2.5).abs() < 1e-12);
        assert!((percentile(&a, 100.0) - 4.0).abs() < 1e-12);
    }
}
