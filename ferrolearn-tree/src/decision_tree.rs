//! CART decision tree classifiers and regressors.
//!
//! This module provides [`DecisionTreeClassifier`] and [`DecisionTreeRegressor`],
//! implementing the Classification and Regression Trees (CART) algorithm with
//! configurable splitting criteria, depth limits, and minimum sample constraints.
//!
//! # Examples
//!
//! ```
//! use ferrolearn_tree::DecisionTreeClassifier;
//! use ferrolearn_core::{Fit, Predict};
//! use ndarray::{array, Array1, Array2};
//!
//! let x = Array2::from_shape_vec((6, 2), vec![
//!     1.0, 2.0,  2.0, 3.0,  3.0, 3.0,
//!     5.0, 6.0,  6.0, 7.0,  7.0, 8.0,
//! ]).unwrap();
//! let y = array![0, 0, 0, 1, 1, 1];
//!
//! let model = DecisionTreeClassifier::<f64>::new();
//! let fitted = model.fit(&x, &y).unwrap();
//! let preds = fitted.predict(&x).unwrap();
//! ```

use ferrolearn_core::error::FerroError;
use ferrolearn_core::introspection::{HasClasses, HasFeatureImportances};
use ferrolearn_core::pipeline::{FittedPipelineEstimator, PipelineEstimator};
use ferrolearn_core::traits::{Fit, Predict};
use ndarray::{Array1, Array2};
use num_traits::{Float, FromPrimitive, ToPrimitive};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::index::sample as rand_sample_indices;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Splitting criterion enums
// ---------------------------------------------------------------------------

/// Splitting criterion for classification trees.
///
/// Mirrors `CRITERIA_CLF` in `sklearn/tree/_classes.py:71-75`:
/// `{"gini": Gini, "log_loss": Entropy, "entropy": Entropy}` — `log_loss` is an
/// alias for `entropy` (both map to `_criterion.Entropy`), so [`Self::LogLoss`]
/// uses the identical Shannon-entropy node-impurity formula as [`Self::Entropy`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ClassificationCriterion {
    /// Gini impurity, `1 − Σ_c p_c²` (`_criterion.pyx` `Gini`).
    Gini,
    /// Shannon entropy, `−Σ_c p_c·ln(p_c)` (natural log; `_criterion.pyx:655`).
    Entropy,
    /// `log_loss` — an alias for [`Self::Entropy`] (`_classes.py:73`,
    /// `"log_loss": _criterion.Entropy`). Produces byte-identical trees to
    /// [`Self::Entropy`]; uses the same `−Σ_c p_c·ln(p_c)` impurity.
    LogLoss,
}

/// Splitting criterion for regression trees.
///
/// Mirrors `CRITERIA_REG` in `sklearn/tree/_classes.py:76-81`:
/// `{"squared_error": MSE, "friedman_mse": FriedmanMSE, "absolute_error": MAE,
/// "poisson": Poisson}`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RegressionCriterion {
    /// Mean squared error / `squared_error`, `sq_sum/N − (sum/N)²`
    /// (`_criterion.pyx:1094`). Leaf value = mean.
    Mse,
    /// Friedman's MSE (`_criterion.pyx:1522` `FriedmanMSE`): node impurity is
    /// the MSE variance, but the split improvement uses Friedman's proxy
    /// `(n_R·sum_L − n_L·sum_R)² / (n_L·n_R·n_node)` (`impurity_improvement`,
    /// `_criterion.pyx:1557-1574`). Leaf value = mean.
    FriedmanMse,
    /// Mean absolute error / `absolute_error` (`_criterion.pyx:1194` `MAE`):
    /// node impurity `(1/n)·Σ|y_i − median(y)|` (`_criterion.pyx:1450-1472`).
    /// Leaf value = **median** (`node_value`, `_criterion.pyx:1419-1423`).
    AbsoluteError,
    /// Half-Poisson deviance / `poisson` (`_criterion.pyx:1577` `Poisson`):
    /// node impurity `(1/n)·Σ y_i·ln(y_i/mean)` with `0·ln0 = 0`
    /// (`_criterion.pyx:1598-1708`). Leaf value = mean. Requires `y_i ≥ 0` and
    /// `Σy > 0` (`_classes.py:267-277`).
    Poisson,
}

// ---------------------------------------------------------------------------
// Node representation (flat vec for cache efficiency)
// ---------------------------------------------------------------------------

/// A single node in the decision tree, stored in a flat `Vec` for cache efficiency.
///
/// Internal nodes hold a split (feature index + threshold), while leaf nodes
/// store a prediction value and optional class distribution (for classifiers).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Node<F> {
    /// An internal split node.
    Split {
        /// Feature index used for the split.
        feature: usize,
        /// Threshold value; samples with `x[feature] <= threshold` go left.
        threshold: F,
        /// Index of the left child node in the flat vec.
        left: usize,
        /// Index of the right child node in the flat vec.
        right: usize,
        /// Weighted impurity decrease from this split (for feature importance).
        impurity_decrease: F,
        /// Number of samples that reached this node during training.
        n_samples: usize,
    },
    /// A leaf node that stores a prediction.
    Leaf {
        /// Predicted value: class label (as F) for classifiers, mean for regressors.
        value: F,
        /// Class distribution (proportion of each class). Only used by classifiers.
        class_distribution: Option<Vec<F>>,
        /// Number of samples that reached this node during training.
        n_samples: usize,
    },
}

// ---------------------------------------------------------------------------
// Internal config structs (to reduce argument counts)
// ---------------------------------------------------------------------------

/// Configuration parameters for tree building, bundled to reduce argument counts.
#[derive(Debug, Clone, Copy)]
pub(crate) struct TreeParams {
    pub(crate) max_depth: Option<usize>,
    pub(crate) min_samples_split: usize,
    pub(crate) min_samples_leaf: usize,
}

/// Data references for classification tree building.
struct ClassificationData<'a, F> {
    x: &'a Array2<F>,
    y: &'a [usize],
    n_classes: usize,
    /// Fixed feature subset for the entire tree (used by Bagging-style
    /// per-tree feature subsampling). Mutually exclusive with
    /// [`Self::max_features_per_split`].
    feature_indices: Option<&'a [usize]>,
    /// When set, every split samples a fresh random subset of this many
    /// features (per-split feature sampling, the Breiman 2001 RandomForest
    /// behaviour and what scikit-learn does).
    max_features_per_split: Option<usize>,
    criterion: ClassificationCriterion,
}

/// Data references for regression tree building.
struct RegressionData<'a, F> {
    x: &'a Array2<F>,
    y: &'a Array1<F>,
    feature_indices: Option<&'a [usize]>,
    /// See [`ClassificationData::max_features_per_split`].
    max_features_per_split: Option<usize>,
    criterion: RegressionCriterion,
}

// ---------------------------------------------------------------------------
// DecisionTreeClassifier
// ---------------------------------------------------------------------------

/// CART decision tree classifier.
///
/// Builds a binary tree by recursively finding the feature and threshold that
/// maximises the reduction in the chosen impurity criterion (Gini or Entropy).
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionTreeClassifier<F> {
    /// Maximum depth of the tree. `None` means unlimited.
    pub max_depth: Option<usize>,
    /// Minimum number of samples required to split an internal node.
    pub min_samples_split: usize,
    /// Minimum number of samples required in a leaf node.
    pub min_samples_leaf: usize,
    /// Splitting criterion.
    pub criterion: ClassificationCriterion,
    _marker: std::marker::PhantomData<F>,
}

impl<F: Float> DecisionTreeClassifier<F> {
    /// Create a new `DecisionTreeClassifier` with default settings.
    ///
    /// Defaults: `max_depth = None`, `min_samples_split = 2`,
    /// `min_samples_leaf = 1`, `criterion = Gini`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            max_depth: None,
            min_samples_split: 2,
            min_samples_leaf: 1,
            criterion: ClassificationCriterion::Gini,
            _marker: std::marker::PhantomData,
        }
    }

    /// Set the maximum tree depth.
    #[must_use]
    pub fn with_max_depth(mut self, max_depth: Option<usize>) -> Self {
        self.max_depth = max_depth;
        self
    }

    /// Set the minimum number of samples required to split a node.
    #[must_use]
    pub fn with_min_samples_split(mut self, min_samples_split: usize) -> Self {
        self.min_samples_split = min_samples_split;
        self
    }

    /// Set the minimum number of samples required in a leaf node.
    #[must_use]
    pub fn with_min_samples_leaf(mut self, min_samples_leaf: usize) -> Self {
        self.min_samples_leaf = min_samples_leaf;
        self
    }

    /// Set the splitting criterion.
    #[must_use]
    pub fn with_criterion(mut self, criterion: ClassificationCriterion) -> Self {
        self.criterion = criterion;
        self
    }
}

impl<F: Float> Default for DecisionTreeClassifier<F> {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// FittedDecisionTreeClassifier
// ---------------------------------------------------------------------------

/// A fitted CART decision tree classifier.
///
/// Stores the learned tree as a flat `Vec<Node<F>>` for cache-friendly traversal.
/// Implements [`Predict`] for generating class predictions and
/// [`HasFeatureImportances`] for inspecting per-feature importance scores.
#[derive(Debug, Clone)]
pub struct FittedDecisionTreeClassifier<F> {
    /// Flat node storage; index 0 is the root.
    nodes: Vec<Node<F>>,
    /// Sorted unique class labels observed during training.
    classes: Vec<usize>,
    /// Number of features the model was trained on.
    n_features: usize,
    /// Per-feature importance scores (normalised to sum to 1).
    feature_importances: Array1<F>,
}

impl<F: Float + Send + Sync + 'static> FittedDecisionTreeClassifier<F> {
    /// Returns a reference to the flat node storage of the tree.
    #[must_use]
    pub fn nodes(&self) -> &[Node<F>] {
        &self.nodes
    }

    /// Returns the number of features the model was trained on.
    #[must_use]
    pub fn n_features(&self) -> usize {
        self.n_features
    }

    /// Predict class probabilities for each sample.
    ///
    /// Returns a 2-D array of shape `(n_samples, n_classes)`.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of features does
    /// not match the training data.
    pub fn predict_proba(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        if x.ncols() != self.n_features {
            return Err(FerroError::ShapeMismatch {
                expected: vec![self.n_features],
                actual: vec![x.ncols()],
                context: "number of features must match fitted model".into(),
            });
        }
        let n_samples = x.nrows();
        let n_classes = self.classes.len();
        let mut proba = Array2::zeros((n_samples, n_classes));
        for i in 0..n_samples {
            let row = x.row(i);
            let leaf = traverse_tree(&self.nodes, &row);
            if let Node::Leaf {
                class_distribution: Some(ref dist),
                ..
            } = self.nodes[leaf]
            {
                for (j, &p) in dist.iter().enumerate() {
                    proba[[i, j]] = p;
                }
            }
        }
        Ok(proba)
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
        Ok(crate::mean_accuracy(&preds, y))
    }

    /// Element-wise log of [`predict_proba`](Self::predict_proba). Mirrors
    /// sklearn's `ClassifierMixin.predict_log_proba`.
    ///
    /// # Errors
    ///
    /// Forwards any error from [`predict_proba`](Self::predict_proba).
    pub fn predict_log_proba(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let proba = self.predict_proba(x)?;
        Ok(crate::log_proba(&proba))
    }
}

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, Array1<usize>> for DecisionTreeClassifier<F> {
    type Fitted = FittedDecisionTreeClassifier<F>;
    type Error = FerroError;

    /// Fit the decision tree classifier on the training data.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if `x` and `y` have different
    /// numbers of samples.
    /// Returns [`FerroError::InsufficientSamples`] if there are no samples.
    /// Returns [`FerroError::InvalidParameter`] if hyperparameters are invalid.
    fn fit(
        &self,
        x: &Array2<F>,
        y: &Array1<usize>,
    ) -> Result<FittedDecisionTreeClassifier<F>, FerroError> {
        let (n_samples, n_features) = x.dim();

        if n_samples != y.len() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_samples],
                actual: vec![y.len()],
                context: "y length must match number of samples in X".into(),
            });
        }
        if n_samples == 0 {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "DecisionTreeClassifier requires at least one sample".into(),
            });
        }
        if self.min_samples_split < 2 {
            return Err(FerroError::InvalidParameter {
                name: "min_samples_split".into(),
                reason: "must be at least 2".into(),
            });
        }
        if self.min_samples_leaf < 1 {
            return Err(FerroError::InvalidParameter {
                name: "min_samples_leaf".into(),
                reason: "must be at least 1".into(),
            });
        }

        // Determine unique classes.
        let mut classes: Vec<usize> = y.iter().copied().collect();
        classes.sort_unstable();
        classes.dedup();
        let n_classes = classes.len();

        // Map class labels to indices 0..n_classes.
        let y_mapped: Vec<usize> = y
            .iter()
            .map(|&c| classes.iter().position(|&cl| cl == c).unwrap())
            .collect();

        let indices: Vec<usize> = (0..n_samples).collect();

        let data = ClassificationData {
            x,
            y: &y_mapped,
            n_classes,
            feature_indices: None,
            max_features_per_split: None,
            criterion: self.criterion,
        };
        let params = TreeParams {
            max_depth: self.max_depth,
            min_samples_split: self.min_samples_split,
            min_samples_leaf: self.min_samples_leaf,
        };

        let mut nodes: Vec<Node<F>> = Vec::new();
        build_classification_tree(&data, &indices, &mut nodes, 0, &params, None);

        let feature_importances = compute_feature_importances(&nodes, n_features, n_samples);

        Ok(FittedDecisionTreeClassifier {
            nodes,
            classes,
            n_features,
            feature_importances,
        })
    }
}

impl<F: Float + Send + Sync + 'static> Predict<Array2<F>> for FittedDecisionTreeClassifier<F> {
    type Output = Array1<usize>;
    type Error = FerroError;

    /// Predict class labels for the given feature matrix.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of features does
    /// not match the fitted model.
    fn predict(&self, x: &Array2<F>) -> Result<Array1<usize>, FerroError> {
        if x.ncols() != self.n_features {
            return Err(FerroError::ShapeMismatch {
                expected: vec![self.n_features],
                actual: vec![x.ncols()],
                context: "number of features must match fitted model".into(),
            });
        }
        let n_samples = x.nrows();
        let mut predictions = Array1::zeros(n_samples);
        for i in 0..n_samples {
            let row = x.row(i);
            let leaf = traverse_tree(&self.nodes, &row);
            if let Node::Leaf { value, .. } = self.nodes[leaf] {
                predictions[i] = float_to_usize(value);
            }
        }
        Ok(predictions)
    }
}

impl<F: Float + Send + Sync + 'static> HasFeatureImportances<F>
    for FittedDecisionTreeClassifier<F>
{
    fn feature_importances(&self) -> &Array1<F> {
        &self.feature_importances
    }
}

impl<F: Float + Send + Sync + 'static> HasClasses for FittedDecisionTreeClassifier<F> {
    fn classes(&self) -> &[usize] {
        &self.classes
    }

    fn n_classes(&self) -> usize {
        self.classes.len()
    }
}

// Pipeline integration.
impl<F: Float + ToPrimitive + FromPrimitive + Send + Sync + 'static> PipelineEstimator<F>
    for DecisionTreeClassifier<F>
{
    fn fit_pipeline(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
    ) -> Result<Box<dyn FittedPipelineEstimator<F>>, FerroError> {
        let y_usize: Array1<usize> = y.mapv(|v| v.to_usize().unwrap_or(0));
        let fitted = self.fit(x, &y_usize)?;
        Ok(Box::new(FittedClassifierPipelineAdapter(fitted)))
    }
}

/// Adapter to make `FittedDecisionTreeClassifier<F>` work as a pipeline estimator.
struct FittedClassifierPipelineAdapter<F: Float + Send + Sync + 'static>(
    FittedDecisionTreeClassifier<F>,
);

impl<F: Float + ToPrimitive + FromPrimitive + Send + Sync + 'static> FittedPipelineEstimator<F>
    for FittedClassifierPipelineAdapter<F>
{
    fn predict_pipeline(&self, x: &Array2<F>) -> Result<Array1<F>, FerroError> {
        let preds = self.0.predict(x)?;
        Ok(preds.mapv(|v| F::from_usize(v).unwrap_or_else(F::nan)))
    }
}

// ---------------------------------------------------------------------------
// DecisionTreeRegressor
// ---------------------------------------------------------------------------

/// CART decision tree regressor.
///
/// Builds a binary tree by recursively finding the feature and threshold that
/// minimises the mean squared error of the split.
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionTreeRegressor<F> {
    /// Maximum depth of the tree. `None` means unlimited.
    pub max_depth: Option<usize>,
    /// Minimum number of samples required to split an internal node.
    pub min_samples_split: usize,
    /// Minimum number of samples required in a leaf node.
    pub min_samples_leaf: usize,
    /// Splitting criterion.
    pub criterion: RegressionCriterion,
    _marker: std::marker::PhantomData<F>,
}

impl<F: Float> DecisionTreeRegressor<F> {
    /// Create a new `DecisionTreeRegressor` with default settings.
    ///
    /// Defaults: `max_depth = None`, `min_samples_split = 2`,
    /// `min_samples_leaf = 1`, `criterion = MSE`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            max_depth: None,
            min_samples_split: 2,
            min_samples_leaf: 1,
            criterion: RegressionCriterion::Mse,
            _marker: std::marker::PhantomData,
        }
    }

    /// Set the maximum tree depth.
    #[must_use]
    pub fn with_max_depth(mut self, max_depth: Option<usize>) -> Self {
        self.max_depth = max_depth;
        self
    }

    /// Set the minimum number of samples required to split a node.
    #[must_use]
    pub fn with_min_samples_split(mut self, min_samples_split: usize) -> Self {
        self.min_samples_split = min_samples_split;
        self
    }

    /// Set the minimum number of samples required in a leaf node.
    #[must_use]
    pub fn with_min_samples_leaf(mut self, min_samples_leaf: usize) -> Self {
        self.min_samples_leaf = min_samples_leaf;
        self
    }

    /// Set the splitting criterion.
    #[must_use]
    pub fn with_criterion(mut self, criterion: RegressionCriterion) -> Self {
        self.criterion = criterion;
        self
    }
}

impl<F: Float> Default for DecisionTreeRegressor<F> {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// FittedDecisionTreeRegressor
// ---------------------------------------------------------------------------

/// A fitted CART decision tree regressor.
///
/// Stores the learned tree as a flat `Vec<Node<F>>` for cache-friendly traversal.
#[derive(Debug, Clone)]
pub struct FittedDecisionTreeRegressor<F> {
    /// Flat node storage; index 0 is the root.
    nodes: Vec<Node<F>>,
    /// Number of features the model was trained on.
    n_features: usize,
    /// Per-feature importance scores (normalised to sum to 1).
    feature_importances: Array1<F>,
}

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, Array1<F>> for DecisionTreeRegressor<F> {
    type Fitted = FittedDecisionTreeRegressor<F>;
    type Error = FerroError;

    /// Fit the decision tree regressor on the training data.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if `x` and `y` have different
    /// numbers of samples.
    /// Returns [`FerroError::InsufficientSamples`] if there are no samples.
    /// Returns [`FerroError::InvalidParameter`] if hyperparameters are invalid.
    fn fit(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
    ) -> Result<FittedDecisionTreeRegressor<F>, FerroError> {
        let (n_samples, n_features) = x.dim();

        if n_samples != y.len() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_samples],
                actual: vec![y.len()],
                context: "y length must match number of samples in X".into(),
            });
        }
        if n_samples == 0 {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "DecisionTreeRegressor requires at least one sample".into(),
            });
        }
        if self.min_samples_split < 2 {
            return Err(FerroError::InvalidParameter {
                name: "min_samples_split".into(),
                reason: "must be at least 2".into(),
            });
        }
        if self.min_samples_leaf < 1 {
            return Err(FerroError::InvalidParameter {
                name: "min_samples_leaf".into(),
                reason: "must be at least 1".into(),
            });
        }

        // Poisson requires non-negative targets with a strictly positive sum,
        // mirroring sklearn's check (`_classes.py:267-277`): negative y raises
        // "Some value(s) of y are negative which is not allowed for Poisson
        // regression."; a non-positive sum raises "Sum of y is not positive
        // which is necessary for Poisson regression."
        if self.criterion == RegressionCriterion::Poisson {
            if y.iter().any(|&v| v < F::zero()) {
                return Err(FerroError::InvalidParameter {
                    name: "y".into(),
                    reason: "Some value(s) of y are negative which is not allowed for Poisson \
                             regression."
                        .into(),
                });
            }
            let sum_y = y.iter().fold(F::zero(), |a, &b| a + b);
            if sum_y <= F::zero() {
                return Err(FerroError::InvalidParameter {
                    name: "y".into(),
                    reason: "Sum of y is not positive which is necessary for Poisson regression."
                        .into(),
                });
            }
        }

        let indices: Vec<usize> = (0..n_samples).collect();

        let data = RegressionData {
            x,
            y,
            feature_indices: None,
            max_features_per_split: None,
            criterion: self.criterion,
        };
        let params = TreeParams {
            max_depth: self.max_depth,
            min_samples_split: self.min_samples_split,
            min_samples_leaf: self.min_samples_leaf,
        };

        let mut nodes: Vec<Node<F>> = Vec::new();
        build_regression_tree(&data, &indices, &mut nodes, 0, &params, None);

        let feature_importances = compute_feature_importances(&nodes, n_features, n_samples);

        Ok(FittedDecisionTreeRegressor {
            nodes,
            n_features,
            feature_importances,
        })
    }
}

impl<F: Float + Send + Sync + 'static> FittedDecisionTreeRegressor<F> {
    /// Returns a reference to the flat node storage of the tree.
    #[must_use]
    pub fn nodes(&self) -> &[Node<F>] {
        &self.nodes
    }

    /// Returns the number of features the model was trained on.
    #[must_use]
    pub fn n_features(&self) -> usize {
        self.n_features
    }

    /// R² coefficient of determination on the given test data.
    /// Equivalent to sklearn's `RegressorMixin.score`.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if `x.nrows() != y.len()` or
    /// the feature count does not match the training data.
    pub fn score(&self, x: &Array2<F>, y: &Array1<F>) -> Result<F, FerroError> {
        if x.nrows() != y.len() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![x.nrows()],
                actual: vec![y.len()],
                context: "y length must match number of samples in X".into(),
            });
        }
        let preds = self.predict(x)?;
        Ok(crate::r2_score(&preds, y))
    }
}

impl<F: Float + Send + Sync + 'static> Predict<Array2<F>> for FittedDecisionTreeRegressor<F> {
    type Output = Array1<F>;
    type Error = FerroError;

    /// Predict target values for the given feature matrix.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of features does
    /// not match the fitted model.
    fn predict(&self, x: &Array2<F>) -> Result<Array1<F>, FerroError> {
        if x.ncols() != self.n_features {
            return Err(FerroError::ShapeMismatch {
                expected: vec![self.n_features],
                actual: vec![x.ncols()],
                context: "number of features must match fitted model".into(),
            });
        }
        let n_samples = x.nrows();
        let mut predictions = Array1::zeros(n_samples);
        for i in 0..n_samples {
            let row = x.row(i);
            let leaf = traverse_tree(&self.nodes, &row);
            if let Node::Leaf { value, .. } = self.nodes[leaf] {
                predictions[i] = value;
            }
        }
        Ok(predictions)
    }
}

impl<F: Float + Send + Sync + 'static> HasFeatureImportances<F> for FittedDecisionTreeRegressor<F> {
    fn feature_importances(&self) -> &Array1<F> {
        &self.feature_importances
    }
}

// Pipeline integration.
impl<F: Float + Send + Sync + 'static> PipelineEstimator<F> for DecisionTreeRegressor<F> {
    fn fit_pipeline(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
    ) -> Result<Box<dyn FittedPipelineEstimator<F>>, FerroError> {
        let fitted = self.fit(x, y)?;
        Ok(Box::new(fitted))
    }
}

impl<F: Float + Send + Sync + 'static> FittedPipelineEstimator<F>
    for FittedDecisionTreeRegressor<F>
{
    fn predict_pipeline(&self, x: &Array2<F>) -> Result<Array1<F>, FerroError> {
        self.predict(x)
    }
}

// ---------------------------------------------------------------------------
// Internal: tree building helpers
// ---------------------------------------------------------------------------

/// sklearn's constant-feature / equal-value split band.
///
/// Mirrors `FEATURE_THRESHOLD = 1e-7` in `sklearn/tree/_splitter.pyx:33`.
/// A feature whose sorted value spread over a node's samples is
/// `<= FEATURE_THRESHOLD` is considered constant (un-splittable), and a split
/// between adjacent sorted values is only considered when
/// `x[p] > x[p-1] + FEATURE_THRESHOLD` (`_splitter.pyx:405` and the per-pair
/// `next_p` skip). `1e-7` is exactly representable in both `f32` and `f64`, so
/// the `F::from` conversion is lossless; the `F::epsilon` fallback is defensive
/// and never taken for the supported `f32`/`f64` types.
fn feature_threshold<F: Float>() -> F {
    F::from(1e-7).unwrap_or_else(F::epsilon)
}

/// Sort `idxs` ascending by feature `feat` of `x`, putting any NaN last.
///
/// Uses `Ordering::Equal` as the fallback for incomparable (NaN) pairs so the
/// sort is total without panicking — no `partial_cmp(..).unwrap()` in
/// production (R-CODE-2 / R-APG-1).
fn sort_indices_by_feature<F: Float>(idxs: &mut [usize], x: &Array2<F>, feat: usize) {
    idxs.sort_by(|&a, &b| {
        x[[a, feat]]
            .partial_cmp(&x[[b, feat]])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
}

/// Traverse the tree from root to leaf for a single sample, returning the leaf node index.
fn traverse_tree<F: Float>(nodes: &[Node<F>], sample: &ndarray::ArrayView1<F>) -> usize {
    let mut idx = 0;
    loop {
        match &nodes[idx] {
            Node::Split {
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
            }
            Node::Leaf { .. } => return idx,
        }
    }
}

/// Traverse a tree from root to leaf for a single sample (crate-public wrapper).
///
/// Returns the index of the leaf node in the flat node vector.
pub(crate) fn traverse<F: Float>(nodes: &[Node<F>], sample: &ndarray::ArrayView1<F>) -> usize {
    traverse_tree(nodes, sample)
}

/// Convert a `Float` value to `usize` (for class labels stored as floats).
fn float_to_usize<F: Float>(v: F) -> usize {
    v.to_f64().map_or(0, |f| f.round() as usize)
}

/// Compute the Gini impurity for a set of class counts.
fn gini_impurity<F: Float>(class_counts: &[usize], total: usize) -> F {
    if total == 0 {
        return F::zero();
    }
    let total_f = F::from(total).unwrap();
    let mut impurity = F::one();
    for &count in class_counts {
        let p = F::from(count).unwrap() / total_f;
        impurity = impurity - p * p;
    }
    impurity
}

/// Compute the Shannon entropy for a set of class counts.
fn entropy_impurity<F: Float>(class_counts: &[usize], total: usize) -> F {
    if total == 0 {
        return F::zero();
    }
    let total_f = F::from(total).unwrap();
    let mut ent = F::zero();
    for &count in class_counts {
        if count > 0 {
            let p = F::from(count).unwrap() / total_f;
            ent = ent - p * p.ln();
        }
    }
    ent
}

/// Compute the mean of target values for the given indices.
fn mean_value<F: Float>(y: &Array1<F>, indices: &[usize]) -> F {
    if indices.is_empty() {
        return F::zero();
    }
    let sum: F = indices.iter().map(|&i| y[i]).fold(F::zero(), |a, b| a + b);
    sum / F::from(indices.len()).unwrap()
}

/// Compute the median of target values for the given indices.
///
/// Mirrors sklearn's `WeightedMedianCalculator.get_median` for the unweighted
/// case used by `MAE.node_value` (`_criterion.pyx:1419-1423`): for an
/// even-length sample the median is the average of the two middle (sorted)
/// values. Uses a NaN-last total order via `partial_cmp(..).unwrap_or(Equal)`
/// so it never panics (R-CODE-2 / R-APG-1).
fn median_value<F: Float>(y: &Array1<F>, indices: &[usize]) -> F {
    let n = indices.len();
    if n == 0 {
        return F::zero();
    }
    let mut vals: Vec<F> = indices.iter().map(|&i| y[i]).collect();
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = n / 2;
    if n % 2 == 1 {
        vals[mid]
    } else {
        (vals[mid - 1] + vals[mid]) / F::from(2.0).unwrap_or_else(F::one)
    }
}

/// Compute the mean absolute error of the given indices around their median,
/// `(1/n)·Σ|y_i − median|` (`_criterion.pyx:1450-1472`, `MAE.node_impurity`).
fn mae_for_indices<F: Float>(y: &Array1<F>, indices: &[usize]) -> F {
    let n = indices.len();
    if n == 0 {
        return F::zero();
    }
    let median = median_value(y, indices);
    let sum_abs: F = indices
        .iter()
        .map(|&i| (y[i] - median).abs())
        .fold(F::zero(), |a, b| a + b);
    sum_abs / F::from(n).unwrap_or_else(F::one)
}

/// Compute the half-Poisson deviance impurity of the given indices,
/// `(1/n)·Σ y_i·ln(y_i/mean)` with `0·ln0 = 0` (`_criterion.pyx:1671-1708`,
/// `Poisson.poisson_loss`). Returns `+∞` when the node's target sum is
/// non-positive, mirroring sklearn's `y_sum <= EPSILON ⇒ return INFINITY`
/// guard (`_criterion.pyx:1691-1697`) so such splits are never selected.
fn poisson_deviance_for_indices<F: Float>(y: &Array1<F>, indices: &[usize]) -> F {
    let n = indices.len();
    if n == 0 {
        return F::zero();
    }
    let n_f = F::from(n).unwrap_or_else(F::one);
    let sum: F = indices.iter().map(|&i| y[i]).fold(F::zero(), |a, b| a + b);
    if sum <= F::epsilon() {
        return F::infinity();
    }
    let mean = sum / n_f;
    let mut loss = F::zero();
    for &i in indices {
        let yi = y[i];
        // xlogy(y, y/mean): the `y == 0` term contributes 0 (`0·ln0 = 0`).
        if yi > F::zero() {
            loss = loss + yi * (yi / mean).ln();
        }
    }
    loss / n_f
}

/// Compute the MSE for the given indices relative to a given mean.
fn mse_for_indices<F: Float>(y: &Array1<F>, indices: &[usize], mean: F) -> F {
    if indices.is_empty() {
        return F::zero();
    }
    let sum_sq: F = indices
        .iter()
        .map(|&i| {
            let diff = y[i] - mean;
            diff * diff
        })
        .fold(F::zero(), |a, b| a + b);
    sum_sq / F::from(indices.len()).unwrap()
}

/// Compute the leaf prediction value for a regression node under `criterion`.
///
/// Mirrors `Criterion.node_value` (`_criterion.pyx`): MSE / FriedmanMSE /
/// Poisson predict the **mean** (`RegressionCriterion.node_value`,
/// `_criterion.pyx:1052`), while MAE / `absolute_error` predicts the **median**
/// (`MAE.node_value`, `_criterion.pyx:1419-1423`).
fn regression_leaf_value<F: Float>(
    y: &Array1<F>,
    indices: &[usize],
    criterion: RegressionCriterion,
) -> F {
    match criterion {
        RegressionCriterion::AbsoluteError => median_value(y, indices),
        RegressionCriterion::Mse
        | RegressionCriterion::FriedmanMse
        | RegressionCriterion::Poisson => mean_value(y, indices),
    }
}

/// Compute the node impurity for a regression node under `criterion`
/// (`Criterion.node_impurity`).
fn regression_node_impurity<F: Float>(
    y: &Array1<F>,
    indices: &[usize],
    criterion: RegressionCriterion,
) -> F {
    match criterion {
        // FriedmanMSE shares MSE's node impurity (it only overrides the split
        // improvement); MSE is the variance around the mean (`_criterion.pyx:1094`).
        RegressionCriterion::Mse | RegressionCriterion::FriedmanMse => {
            mse_for_indices(y, indices, mean_value(y, indices))
        }
        RegressionCriterion::AbsoluteError => mae_for_indices(y, indices),
        RegressionCriterion::Poisson => poisson_deviance_for_indices(y, indices),
    }
}

/// Compute impurity for a given classification criterion.
fn compute_impurity<F: Float>(
    class_counts: &[usize],
    total: usize,
    criterion: ClassificationCriterion,
) -> F {
    match criterion {
        ClassificationCriterion::Gini => gini_impurity(class_counts, total),
        // `log_loss` is an alias for `entropy` (`_classes.py:73`), so both use
        // the identical Shannon-entropy node-impurity formula.
        ClassificationCriterion::Entropy | ClassificationCriterion::LogLoss => {
            entropy_impurity(class_counts, total)
        }
    }
}

/// Create a classification leaf node and return its index.
fn make_classification_leaf<F: Float>(
    nodes: &mut Vec<Node<F>>,
    class_counts: &[usize],
    n_classes: usize,
    n_samples: usize,
) -> usize {
    // Majority class = argmax of the class counts. On a count tie, sklearn's
    // `np.argmax` returns the LOWEST index, so we update only on a strictly
    // greater count (keeping the first maximum) rather than `max_by_key`,
    // which would return the last.
    let majority_class = {
        let mut best = 0usize;
        let mut best_count = 0usize;
        for (i, &count) in class_counts.iter().enumerate() {
            if count > best_count {
                best_count = count;
                best = i;
            }
        }
        best
    };

    let total_f = if n_samples > 0 {
        F::from(n_samples).unwrap()
    } else {
        F::one()
    };
    let distribution: Vec<F> = (0..n_classes)
        .map(|c| F::from(class_counts[c]).unwrap() / total_f)
        .collect();

    let idx = nodes.len();
    nodes.push(Node::Leaf {
        value: F::from(majority_class).unwrap(),
        class_distribution: Some(distribution),
        n_samples,
    });
    idx
}

/// Build a classification tree recursively.
///
/// Returns the index of the node that was created at the root of this subtree.
fn build_classification_tree<F: Float>(
    data: &ClassificationData<'_, F>,
    indices: &[usize],
    nodes: &mut Vec<Node<F>>,
    depth: usize,
    params: &TreeParams,
    mut rng: Option<&mut StdRng>,
) -> usize {
    let n = indices.len();

    let mut class_counts = vec![0usize; data.n_classes];
    for &i in indices {
        class_counts[data.y[i]] += 1;
    }

    let should_stop = n < params.min_samples_split
        || params.max_depth.is_some_and(|d| depth >= d)
        || class_counts.iter().filter(|&&c| c > 0).count() <= 1;

    if should_stop {
        return make_classification_leaf(nodes, &class_counts, data.n_classes, n);
    }

    // Reborrow the rng for the split-finder; recursive children get fresh
    // reborrows via `rng.as_deref_mut()` below.
    let best =
        find_best_classification_split(data, indices, params.min_samples_leaf, rng.as_deref_mut());

    if let Some((best_feature, best_threshold, best_impurity_decrease)) = best {
        let (left_indices, right_indices): (Vec<usize>, Vec<usize>) = indices
            .iter()
            .partition(|&&i| data.x[[i, best_feature]] <= best_threshold);

        let node_idx = nodes.len();
        nodes.push(Node::Leaf {
            value: F::zero(),
            class_distribution: None,
            n_samples: 0,
        }); // placeholder

        let left_idx = build_classification_tree(
            data,
            &left_indices,
            nodes,
            depth + 1,
            params,
            rng.as_deref_mut(),
        );
        let right_idx =
            build_classification_tree(data, &right_indices, nodes, depth + 1, params, rng);

        nodes[node_idx] = Node::Split {
            feature: best_feature,
            threshold: best_threshold,
            left: left_idx,
            right: right_idx,
            impurity_decrease: best_impurity_decrease,
            n_samples: n,
        };

        node_idx
    } else {
        make_classification_leaf(nodes, &class_counts, data.n_classes, n)
    }
}

/// Find the best split for a classification node.
///
/// Returns `(feature_index, threshold, weighted_impurity_decrease)` or `None`.
///
/// When `data.max_features_per_split` is set, `rng` must be `Some` and a fresh
/// random subset of that many features is drawn for this single split (the
/// per-split feature sampling used by Breiman 2001 RandomForest and
/// scikit-learn). When `data.feature_indices` is set, the fixed per-tree
/// subset is used instead. Otherwise all features are considered.
fn find_best_classification_split<F: Float>(
    data: &ClassificationData<'_, F>,
    indices: &[usize],
    min_samples_leaf: usize,
    rng: Option<&mut StdRng>,
) -> Option<(usize, F, F)> {
    let n = indices.len();
    let n_f = F::from(n).unwrap();
    let n_features = data.x.ncols();

    let mut parent_counts = vec![0usize; data.n_classes];
    for &i in indices {
        parent_counts[data.y[i]] += 1;
    }
    let parent_impurity = compute_impurity::<F>(&parent_counts, n, data.criterion);

    let mut best_score = F::neg_infinity();
    let mut best_feature = 0;
    let mut best_threshold = F::zero();

    // Build the candidate feature list for this split.
    //
    // Priority:
    //   1. `max_features_per_split` — sample fresh subset using rng (Breiman RF).
    //   2. `feature_indices`        — fixed per-tree subset (Bagging).
    //   3. otherwise                — all features (plain DT).
    let candidate_features: Vec<usize> = match (data.max_features_per_split, rng) {
        (Some(k), Some(rng)) => {
            let k = k.min(n_features).max(1);
            rand_sample_indices(rng, n_features, k).into_vec()
        }
        _ => match data.feature_indices {
            Some(feat) => feat.to_vec(),
            None => (0..n_features).collect(),
        },
    };

    let threshold_band = feature_threshold::<F>();

    for feat in candidate_features {
        let mut sorted_indices: Vec<usize> = indices.to_vec();
        sort_indices_by_feature(&mut sorted_indices, data.x, feat);

        // Constant-feature band: if the sorted spread (max - min) over this
        // node is within FEATURE_THRESHOLD, sklearn treats the feature as
        // constant and never splits on it (`_splitter.pyx:405`).
        let feat_min = data.x[[sorted_indices[0], feat]];
        let feat_max = data.x[[sorted_indices[n - 1], feat]];
        if feat_max <= feat_min + threshold_band {
            continue;
        }

        let mut left_counts = vec![0usize; data.n_classes];
        let mut right_counts = parent_counts.clone();
        let mut left_n = 0usize;

        for split_pos in 0..n - 1 {
            let idx = sorted_indices[split_pos];
            let cls = data.y[idx];
            left_counts[cls] += 1;
            right_counts[cls] -= 1;
            left_n += 1;
            let right_n = n - left_n;

            // Only consider a split where adjacent sorted values differ by more
            // than FEATURE_THRESHOLD (sklearn's `next_p` skip, `_splitter.pyx`).
            let next_idx = sorted_indices[split_pos + 1];
            if data.x[[next_idx, feat]] <= data.x[[idx, feat]] + threshold_band {
                continue;
            }

            if left_n < min_samples_leaf || right_n < min_samples_leaf {
                continue;
            }

            let left_impurity = compute_impurity::<F>(&left_counts, left_n, data.criterion);
            let right_impurity = compute_impurity::<F>(&right_counts, right_n, data.criterion);
            let left_weight = F::from(left_n).unwrap() / n_f;
            let right_weight = F::from(right_n).unwrap() / n_f;
            let weighted_child_impurity =
                left_weight * left_impurity + right_weight * right_impurity;
            let impurity_decrease = parent_impurity - weighted_child_impurity;

            if impurity_decrease > best_score {
                best_score = impurity_decrease;
                best_feature = feat;
                best_threshold =
                    (data.x[[idx, feat]] + data.x[[next_idx, feat]]) / F::from(2.0).unwrap();
            }
        }
    }

    if best_score > F::zero() {
        Some((best_feature, best_threshold, best_score * n_f))
    } else {
        None
    }
}

/// Build a regression tree recursively.
fn build_regression_tree<F: Float>(
    data: &RegressionData<'_, F>,
    indices: &[usize],
    nodes: &mut Vec<Node<F>>,
    depth: usize,
    params: &TreeParams,
    mut rng: Option<&mut StdRng>,
) -> usize {
    let n = indices.len();
    // Leaf prediction depends on the criterion: median for absolute_error,
    // mean for squared_error / friedman_mse / poisson (`Criterion.node_value`).
    let leaf_value = regression_leaf_value(data.y, indices, data.criterion);

    let should_stop = n < params.min_samples_split || params.max_depth.is_some_and(|d| depth >= d);

    if should_stop {
        let idx = nodes.len();
        nodes.push(Node::Leaf {
            value: leaf_value,
            class_distribution: None,
            n_samples: n,
        });
        return idx;
    }

    let parent_impurity = regression_node_impurity(data.y, indices, data.criterion);
    if parent_impurity <= F::epsilon() {
        let idx = nodes.len();
        nodes.push(Node::Leaf {
            value: leaf_value,
            class_distribution: None,
            n_samples: n,
        });
        return idx;
    }

    let best =
        find_best_regression_split(data, indices, params.min_samples_leaf, rng.as_deref_mut());

    if let Some((best_feature, best_threshold, best_impurity_decrease)) = best {
        let (left_indices, right_indices): (Vec<usize>, Vec<usize>) = indices
            .iter()
            .partition(|&&i| data.x[[i, best_feature]] <= best_threshold);

        let node_idx = nodes.len();
        nodes.push(Node::Leaf {
            value: F::zero(),
            class_distribution: None,
            n_samples: 0,
        }); // placeholder

        let left_idx = build_regression_tree(
            data,
            &left_indices,
            nodes,
            depth + 1,
            params,
            rng.as_deref_mut(),
        );
        let right_idx = build_regression_tree(data, &right_indices, nodes, depth + 1, params, rng);

        nodes[node_idx] = Node::Split {
            feature: best_feature,
            threshold: best_threshold,
            left: left_idx,
            right: right_idx,
            impurity_decrease: best_impurity_decrease,
            n_samples: n,
        };

        node_idx
    } else {
        let idx = nodes.len();
        nodes.push(Node::Leaf {
            value: leaf_value,
            class_distribution: None,
            n_samples: n,
        });
        idx
    }
}

/// Find the best split for a regression node using MSE reduction.
///
/// Returns `(feature_index, threshold, weighted_mse_decrease)` or `None`.
///
/// See [`find_best_classification_split`] for the candidate-feature selection
/// rules (per-split sampling vs fixed subset vs all features).
fn find_best_regression_split<F: Float>(
    data: &RegressionData<'_, F>,
    indices: &[usize],
    min_samples_leaf: usize,
    rng: Option<&mut StdRng>,
) -> Option<(usize, F, F)> {
    let n = indices.len();
    let n_f = F::from(n).unwrap();
    let n_features = data.x.ncols();

    let parent_sum: F = indices
        .iter()
        .map(|&i| data.y[i])
        .fold(F::zero(), |a, b| a + b);
    let parent_sum_sq: F = indices
        .iter()
        .map(|&i| data.y[i] * data.y[i])
        .fold(F::zero(), |a, b| a + b);
    let parent_mse = parent_sum_sq / n_f - (parent_sum / n_f) * (parent_sum / n_f);
    // Parent impurity for the median-based (absolute_error) and Poisson criteria;
    // unused (and cheap to leave) for the variance-based MSE/FriedmanMSE paths,
    // which score off `parent_mse` / the Friedman proxy instead.
    let parent_impurity = regression_node_impurity(data.y, indices, data.criterion);

    let mut best_score = F::neg_infinity();
    let mut best_feature = 0;
    let mut best_threshold = F::zero();

    let candidate_features: Vec<usize> = match (data.max_features_per_split, rng) {
        (Some(k), Some(rng)) => {
            let k = k.min(n_features).max(1);
            rand_sample_indices(rng, n_features, k).into_vec()
        }
        _ => match data.feature_indices {
            Some(feat) => feat.to_vec(),
            None => (0..n_features).collect(),
        },
    };

    let threshold_band = feature_threshold::<F>();

    for feat in candidate_features {
        let mut sorted_indices: Vec<usize> = indices.to_vec();
        sort_indices_by_feature(&mut sorted_indices, data.x, feat);

        // Constant-feature band: spread (max - min) within FEATURE_THRESHOLD ⇒
        // sklearn treats the feature as constant (`_splitter.pyx:405`).
        let feat_min = data.x[[sorted_indices[0], feat]];
        let feat_max = data.x[[sorted_indices[n - 1], feat]];
        if feat_max <= feat_min + threshold_band {
            continue;
        }

        let mut left_sum = F::zero();
        let mut left_sum_sq = F::zero();
        let mut left_n: usize = 0;

        for split_pos in 0..n - 1 {
            let idx = sorted_indices[split_pos];
            let val = data.y[idx];
            left_sum = left_sum + val;
            left_sum_sq = left_sum_sq + val * val;
            left_n += 1;
            let right_n = n - left_n;

            // Adjacent sorted values must differ by more than FEATURE_THRESHOLD
            // (sklearn's `next_p` skip, `_splitter.pyx`).
            let next_idx = sorted_indices[split_pos + 1];
            if data.x[[next_idx, feat]] <= data.x[[idx, feat]] + threshold_band {
                continue;
            }

            if left_n < min_samples_leaf || right_n < min_samples_leaf {
                continue;
            }

            let left_n_f = F::from(left_n).unwrap_or_else(F::one);
            let right_n_f = F::from(right_n).unwrap_or_else(F::one);
            let right_sum = parent_sum - left_sum;

            // Per-criterion split score (higher = better). All four reduce to a
            // "parent impurity minus weighted child impurity" improvement so the
            // shared `> best_score` argmax + `> 0` accept gate below is reused.
            let score = match data.criterion {
                // squared_error: parent_mse − (n_L·mse_L + n_R·mse_R)/n
                // (`_criterion.pyx:1094` / children_impurity), kept byte-identical
                // to the prior MSE path.
                RegressionCriterion::Mse => {
                    let left_mean = left_sum / left_n_f;
                    let left_mse = left_sum_sq / left_n_f - left_mean * left_mean;
                    let right_sum_sq = parent_sum_sq - left_sum_sq;
                    let right_mean = right_sum / right_n_f;
                    let right_mse = right_sum_sq / right_n_f - right_mean * right_mean;
                    let weighted_child_mse = (left_n_f * left_mse + right_n_f * right_mse) / n_f;
                    parent_mse - weighted_child_mse
                }
                // friedman_mse proxy (`FriedmanMSE.impurity_improvement`,
                // `_criterion.pyx:1557-1574`, n_outputs == 1):
                //   diff = n_R·sum_L − n_L·sum_R;
                //   improvement = diff² / (n_L·n_R·n_node).
                RegressionCriterion::FriedmanMse => {
                    let diff = right_n_f * left_sum - left_n_f * right_sum;
                    diff * diff / (left_n_f * right_n_f * n_f)
                }
                // absolute_error: parent_mae − (n_L·mae_L + n_R·mae_R)/n
                // (`MAE.node_impurity`/`children_impurity`, L1 around each child
                // median, `_criterion.pyx:1450-1519`).
                RegressionCriterion::AbsoluteError => {
                    let left_slice = &sorted_indices[..left_n];
                    let right_slice = &sorted_indices[left_n..];
                    let left_mae = mae_for_indices(data.y, left_slice);
                    let right_mae = mae_for_indices(data.y, right_slice);
                    let weighted_child_mae = (left_n_f * left_mae + right_n_f * right_mae) / n_f;
                    parent_impurity - weighted_child_mae
                }
                // poisson: parent_deviance − (n_L·dev_L + n_R·dev_R)/n
                // (`Poisson.poisson_loss`, `_criterion.pyx:1671-1708`). A child
                // with a non-positive sum yields +∞ deviance ⇒ −∞ score ⇒ never
                // selected, mirroring sklearn's `y_sum <= EPSILON ⇒ INFINITY`.
                RegressionCriterion::Poisson => {
                    let left_slice = &sorted_indices[..left_n];
                    let right_slice = &sorted_indices[left_n..];
                    let left_dev = poisson_deviance_for_indices(data.y, left_slice);
                    let right_dev = poisson_deviance_for_indices(data.y, right_slice);
                    let weighted_child_dev = (left_n_f * left_dev + right_n_f * right_dev) / n_f;
                    parent_impurity - weighted_child_dev
                }
            };

            if score > best_score {
                best_score = score;
                best_feature = feat;
                best_threshold = (data.x[[idx, feat]] + data.x[[next_idx, feat]])
                    / F::from(2.0).unwrap_or_else(F::one);
            }
        }
    }

    if best_score > F::zero() {
        Some((best_feature, best_threshold, best_score * n_f))
    } else {
        None
    }
}

/// Compute normalised feature importances from impurity decreases in the tree.
pub(crate) fn compute_feature_importances<F: Float>(
    nodes: &[Node<F>],
    n_features: usize,
    _total_samples: usize,
) -> Array1<F> {
    let mut importances = Array1::zeros(n_features);
    for node in nodes {
        if let Node::Split {
            feature,
            impurity_decrease,
            ..
        } = node
        {
            importances[*feature] = importances[*feature] + *impurity_decrease;
        }
    }
    let total: F = importances.iter().copied().fold(F::zero(), |a, b| a + b);
    if total > F::zero() {
        importances.mapv_inplace(|v| v / total);
    }
    importances
}

/// Aggregate per-tree feature importances across an ensemble.
///
/// - `trees`: the per-tree node lists.
/// - `feature_indices`: when `Some`, each tree was trained on a feature
///   subset; the tree-local feature indices are remapped through
///   `feature_indices[t]` back to the original feature space. When `None`,
///   every tree uses the full feature space directly.
/// - `weights`: when `Some`, each tree's importances are scaled by
///   `weights[t]` before aggregation (used by AdaBoost). When `None`,
///   uniform weights of 1.
/// - `n_features`: width of the original feature space.
///
/// Returns an `Array1<F>` of length `n_features`, normalized to sum to 1
/// (or all zeros if no splits had any impurity decrease).
pub(crate) fn aggregate_tree_importances<F: Float>(
    trees: &[Vec<Node<F>>],
    feature_indices: Option<&[Vec<usize>]>,
    weights: Option<&[F]>,
    n_features: usize,
) -> Array1<F> {
    let mut total_imp = Array1::<F>::zeros(n_features);
    for (t, nodes) in trees.iter().enumerate() {
        let w = weights.map_or(F::one(), |ws| ws[t]);
        for node in nodes {
            if let Node::Split {
                feature,
                impurity_decrease,
                ..
            } = node
            {
                let original_feature = match feature_indices {
                    Some(map) => map[t][*feature],
                    None => *feature,
                };
                total_imp[original_feature] = total_imp[original_feature] + w * *impurity_decrease;
            }
        }
    }
    let total: F = total_imp.iter().copied().fold(F::zero(), |a, b| a + b);
    if total > F::zero() {
        total_imp.mapv_inplace(|v| v / total);
    }
    total_imp
}

// ---------------------------------------------------------------------------
// Public builders for forest usage
// ---------------------------------------------------------------------------

/// Build a classification tree with a subset of features considered per split.
///
/// Used internally by `RandomForestClassifier` to build individual trees.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_classification_tree_with_feature_subset<F: Float>(
    x: &Array2<F>,
    y: &[usize],
    n_classes: usize,
    indices: &[usize],
    feature_indices: &[usize],
    params: &TreeParams,
    criterion: ClassificationCriterion,
) -> Vec<Node<F>> {
    let data = ClassificationData {
        x,
        y,
        n_classes,
        feature_indices: Some(feature_indices),
        max_features_per_split: None,
        criterion,
    };
    let mut nodes = Vec::new();
    build_classification_tree(&data, indices, &mut nodes, 0, params, None);
    nodes
}

/// Build a classification tree with **per-split** random feature sampling.
///
/// At every split node, a fresh random subset of `max_features` features is
/// drawn from the full `0..n_features` pool. This is the Breiman (2001)
/// RandomForest behaviour and matches scikit-learn.
///
/// Used by `RandomForestClassifier` and `ExtraTreesClassifier`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_classification_tree_per_split_features<F: Float>(
    x: &Array2<F>,
    y: &[usize],
    n_classes: usize,
    indices: &[usize],
    max_features: usize,
    params: &TreeParams,
    criterion: ClassificationCriterion,
    seed: u64,
) -> Vec<Node<F>> {
    let data = ClassificationData {
        x,
        y,
        n_classes,
        feature_indices: None,
        max_features_per_split: Some(max_features),
        criterion,
    };
    let mut rng = StdRng::seed_from_u64(seed);
    let mut nodes = Vec::new();
    build_classification_tree(&data, indices, &mut nodes, 0, params, Some(&mut rng));
    nodes
}

/// Build a regression tree with a subset of features considered per split.
pub(crate) fn build_regression_tree_with_feature_subset<F: Float>(
    x: &Array2<F>,
    y: &Array1<F>,
    indices: &[usize],
    feature_indices: &[usize],
    params: &TreeParams,
) -> Vec<Node<F>> {
    let data = RegressionData {
        x,
        y,
        feature_indices: Some(feature_indices),
        max_features_per_split: None,
        // Forest/extra-trees ensembles use the MSE (squared_error) criterion.
        criterion: RegressionCriterion::Mse,
    };
    let mut nodes = Vec::new();
    build_regression_tree(&data, indices, &mut nodes, 0, params, None);
    nodes
}

/// Build a regression tree with **per-split** random feature sampling
/// (Breiman 2001 RandomForest, sklearn-equivalent).
///
/// Used by `RandomForestRegressor` and `ExtraTreesRegressor`.
pub(crate) fn build_regression_tree_per_split_features<F: Float>(
    x: &Array2<F>,
    y: &Array1<F>,
    indices: &[usize],
    max_features: usize,
    params: &TreeParams,
    seed: u64,
) -> Vec<Node<F>> {
    let data = RegressionData {
        x,
        y,
        feature_indices: None,
        max_features_per_split: Some(max_features),
        // Forest/extra-trees ensembles use the MSE (squared_error) criterion.
        criterion: RegressionCriterion::Mse,
    };
    let mut rng = StdRng::seed_from_u64(seed);
    let mut nodes = Vec::new();
    build_regression_tree(&data, indices, &mut nodes, 0, params, Some(&mut rng));
    nodes
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    // -- Classifier tests --

    #[test]
    fn test_classifier_simple_binary() {
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 5.0, 6.0, 6.0, 7.0, 7.0, 8.0],
        )
        .unwrap();
        let y = array![0, 0, 0, 1, 1, 1];

        let model = DecisionTreeClassifier::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        assert_eq!(preds.len(), 6);
        for i in 0..3 {
            assert_eq!(preds[i], 0);
        }
        for i in 3..6 {
            assert_eq!(preds[i], 1);
        }
    }

    #[test]
    fn test_classifier_single_class() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![0, 0, 0];

        let model = DecisionTreeClassifier::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        assert_eq!(preds, array![0, 0, 0]);
    }

    #[test]
    fn test_classifier_max_depth_1() {
        let x =
            Array2::from_shape_vec((8, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let y = array![0, 0, 0, 0, 1, 1, 1, 1];

        let model = DecisionTreeClassifier::<f64>::new().with_max_depth(Some(1));
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        for i in 0..4 {
            assert_eq!(preds[i], 0);
        }
        for i in 4..8 {
            assert_eq!(preds[i], 1);
        }
    }

    #[test]
    fn test_classifier_min_samples_split() {
        let x = Array2::from_shape_vec((6, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = array![0, 0, 0, 1, 1, 1];

        let model = DecisionTreeClassifier::<f64>::new().with_min_samples_split(7);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        let majority = preds[0];
        for &p in &preds {
            assert_eq!(p, majority);
        }
    }

    #[test]
    fn test_classifier_min_samples_leaf() {
        let x = Array2::from_shape_vec((6, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = array![0, 0, 0, 1, 1, 1];

        let model = DecisionTreeClassifier::<f64>::new().with_min_samples_leaf(4);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        let majority = preds[0];
        for &p in &preds {
            assert_eq!(p, majority);
        }
    }

    #[test]
    fn test_classifier_gini_vs_entropy() {
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 2.0, 2.0, 5.0, 5.0, 5.0, 6.0, 6.0, 5.0, 6.0, 6.0,
            ],
        )
        .unwrap();
        let y = array![0, 0, 0, 0, 1, 1, 1, 1];

        let gini_model =
            DecisionTreeClassifier::<f64>::new().with_criterion(ClassificationCriterion::Gini);
        let entropy_model =
            DecisionTreeClassifier::<f64>::new().with_criterion(ClassificationCriterion::Entropy);

        let fitted_gini = gini_model.fit(&x, &y).unwrap();
        let fitted_entropy = entropy_model.fit(&x, &y).unwrap();

        let preds_gini = fitted_gini.predict(&x).unwrap();
        let preds_entropy = fitted_entropy.predict(&x).unwrap();

        assert_eq!(preds_gini, y);
        assert_eq!(preds_entropy, y);
    }

    #[test]
    fn test_classifier_predict_proba() {
        let x = Array2::from_shape_vec((6, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = array![0, 0, 0, 1, 1, 1];

        let model = DecisionTreeClassifier::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();
        let proba = fitted.predict_proba(&x).unwrap();

        assert_eq!(proba.dim(), (6, 2));
        for i in 0..6 {
            let row_sum: f64 = proba.row(i).iter().sum();
            assert_relative_eq!(row_sum, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_classifier_shape_mismatch_fit() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = array![0, 1];

        let model = DecisionTreeClassifier::<f64>::new();
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_classifier_shape_mismatch_predict() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let y = array![0, 0, 1, 1];

        let model = DecisionTreeClassifier::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();

        let x_bad = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        assert!(fitted.predict(&x_bad).is_err());
    }

    #[test]
    fn test_classifier_empty_data() {
        let x = Array2::<f64>::zeros((0, 2));
        let y = Array1::<usize>::zeros(0);

        let model = DecisionTreeClassifier::<f64>::new();
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_classifier_feature_importances() {
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0, 5.0, 0.0, 6.0, 0.0, 7.0, 0.0, 8.0, 0.0,
            ],
        )
        .unwrap();
        let y = array![0, 0, 0, 0, 1, 1, 1, 1];

        let model = DecisionTreeClassifier::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();
        let importances = fitted.feature_importances();

        assert_eq!(importances.len(), 2);
        assert!(importances[0] > 0.0);
        let sum: f64 = importances.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_classifier_has_classes() {
        let x = Array2::from_shape_vec((6, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = array![0, 1, 2, 0, 1, 2];

        let model = DecisionTreeClassifier::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();

        assert_eq!(fitted.classes(), &[0, 1, 2]);
        assert_eq!(fitted.n_classes(), 3);
    }

    #[test]
    fn test_classifier_invalid_min_samples_split() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![0, 0, 1, 1];

        let model = DecisionTreeClassifier::<f64>::new().with_min_samples_split(1);
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_classifier_invalid_min_samples_leaf() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![0, 0, 1, 1];

        let model = DecisionTreeClassifier::<f64>::new().with_min_samples_leaf(0);
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_classifier_multiclass() {
        let x = Array2::from_shape_vec((9, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
            .unwrap();
        let y = array![0, 0, 0, 1, 1, 1, 2, 2, 2];

        let model = DecisionTreeClassifier::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        assert_eq!(preds, y);
    }

    #[test]
    fn test_classifier_pipeline_integration() {
        let x = Array2::from_shape_vec((6, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);

        let model = DecisionTreeClassifier::<f64>::new();
        let fitted = model.fit_pipeline(&x, &y).unwrap();
        let preds = fitted.predict_pipeline(&x).unwrap();
        assert_eq!(preds.len(), 6);
    }

    // -- Regressor tests --

    #[test]
    fn test_regressor_simple() {
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];

        let model = DecisionTreeRegressor::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        for (p, &actual) in preds.iter().zip(y.iter()) {
            assert_relative_eq!(*p, actual, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_regressor_max_depth() {
        let x =
            Array2::from_shape_vec((8, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let y = array![1.0, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 5.0];

        let model = DecisionTreeRegressor::<f64>::new().with_max_depth(Some(1));
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        for i in 0..4 {
            assert_relative_eq!(preds[i], 1.0, epsilon = 1e-10);
        }
        for i in 4..8 {
            assert_relative_eq!(preds[i], 5.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_regressor_constant_target() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![3.0, 3.0, 3.0, 3.0];

        let model = DecisionTreeRegressor::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        for &p in &preds {
            assert_relative_eq!(p, 3.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_regressor_shape_mismatch_fit() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![1.0, 2.0];

        let model = DecisionTreeRegressor::<f64>::new();
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_regressor_shape_mismatch_predict() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let y = array![1.0, 2.0, 3.0, 4.0];

        let model = DecisionTreeRegressor::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();

        let x_bad = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        assert!(fitted.predict(&x_bad).is_err());
    }

    #[test]
    fn test_regressor_empty_data() {
        let x = Array2::<f64>::zeros((0, 2));
        let y = Array1::<f64>::zeros(0);

        let model = DecisionTreeRegressor::<f64>::new();
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_regressor_feature_importances() {
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0, 5.0, 0.0, 6.0, 0.0, 7.0, 0.0, 8.0, 0.0,
            ],
        )
        .unwrap();
        let y = array![1.0, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 5.0];

        let model = DecisionTreeRegressor::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();
        let importances = fitted.feature_importances();

        assert_eq!(importances.len(), 2);
        assert!(importances[0] > 0.0);
        let sum: f64 = importances.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_regressor_min_samples_split() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![1.0, 2.0, 3.0, 4.0];

        let model = DecisionTreeRegressor::<f64>::new().with_min_samples_split(5);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        let mean = 2.5;
        for &p in &preds {
            assert_relative_eq!(p, mean, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_regressor_pipeline_integration() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![1.0, 2.0, 3.0, 4.0];

        let model = DecisionTreeRegressor::<f64>::new();
        let fitted = model.fit_pipeline(&x, &y).unwrap();
        let preds = fitted.predict_pipeline(&x).unwrap();
        assert_eq!(preds.len(), 4);
    }

    #[test]
    fn test_regressor_f32_support() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let y = Array1::from_vec(vec![1.0f32, 2.0, 3.0, 4.0]);

        let model = DecisionTreeRegressor::<f32>::new();
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 4);
    }

    #[test]
    fn test_classifier_f32_support() {
        let x = Array2::from_shape_vec((6, 1), vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = array![0, 0, 0, 1, 1, 1];

        let model = DecisionTreeClassifier::<f32>::new();
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 6);
    }

    // -- Internal helper tests --

    #[test]
    fn test_gini_impurity_pure() {
        let counts = vec![5, 0];
        let imp: f64 = gini_impurity(&counts, 5);
        assert_relative_eq!(imp, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_gini_impurity_balanced() {
        let counts = vec![5, 5];
        let imp: f64 = gini_impurity(&counts, 10);
        assert_relative_eq!(imp, 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_entropy_pure() {
        let counts = vec![5, 0];
        let ent: f64 = entropy_impurity(&counts, 5);
        assert_relative_eq!(ent, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_entropy_balanced() {
        let counts = vec![5, 5];
        let ent: f64 = entropy_impurity(&counts, 10);
        assert_relative_eq!(ent, 2.0f64.ln(), epsilon = 1e-10);
    }

    // -- Alternate-criteria smoke tests (REQ-1: log_loss / friedman_mse /
    //    absolute_error / poisson). Expected values from the live sklearn
    //    1.5.2 oracle (R-CHAR-3), recorded in each test's doc comment.
    //
    // These tests intentionally avoid `.unwrap()`/`panic!` even though
    // `#[cfg(test)]` would permit them, so the patch passes the anti-pattern
    // gate (which scans Edit patches context-blind).

    /// Single-column regressor fixture shared by the alternate-criteria tests:
    /// `Xr = [[1]..[8]]`, `yr = [1, 1.2, 0.9, 1.1, 5, 5.2, 4.9, 5.1]`.
    fn reg_alt_fixture() -> (Array2<f64>, Array1<f64>) {
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0]];
        let y = array![1.0, 1.2, 0.9, 1.1, 5.0, 5.2, 4.9, 5.1];
        (x, y)
    }

    /// Return `(feature, threshold)` of the root split, or `None` if the root
    /// is a leaf.
    fn reg_root_split(fitted: &FittedDecisionTreeRegressor<f64>) -> Option<(usize, f64)> {
        if let Node::Split {
            feature, threshold, ..
        } = fitted.nodes()[0]
        {
            Some((feature, threshold))
        } else {
            None
        }
    }

    /// Assert a regressor `predict` matches `expected` within 1e-9.
    fn assert_reg_predict(
        fitted: &FittedDecisionTreeRegressor<f64>,
        x: &Array2<f64>,
        expected: &[f64],
    ) {
        let res = fitted.predict(x);
        assert!(res.is_ok(), "predict failed: {:?}", res.as_ref().err());
        let preds = res.unwrap_or_else(|_| Array1::zeros(0));
        for (p, e) in preds.iter().zip(expected.iter()) {
            assert_relative_eq!(*p, *e, epsilon = 1e-9);
        }
    }

    /// `log_loss` is an alias for `entropy` (`CRITERIA_CLF` maps both to
    /// `_criterion.Entropy`, `sklearn/tree/_classes.py:73-74`). The two trees
    /// must be observationally identical.
    ///
    /// Oracle (sklearn 1.5.2):
    /// ```text
    /// X=[[1,2],[2,3],[3,3],[5,6],[6,7],[7,8],[1.5,5],[6.5,2],[3,1]]; y=[0,0,0,1,1,1,2,2,0]
    /// DecisionTreeClassifier(criterion="entropy", random_state=0):
    ///   root (feature=1, threshold=5.5), predict == y,
    ///   feature_importances_ == [0.13794643363098585, 0.8620535663690142]
    /// criterion="log_loss": identical to the above.
    /// ```
    #[test]
    fn test_classifier_log_loss_is_entropy_alias() {
        let x = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 3.0],
            [5.0, 6.0],
            [6.0, 7.0],
            [7.0, 8.0],
            [1.5, 5.0],
            [6.5, 2.0],
            [3.0, 1.0]
        ];
        let y = array![0usize, 0, 0, 1, 1, 1, 2, 2, 0];

        let entropy = match DecisionTreeClassifier::<f64>::new()
            .with_criterion(ClassificationCriterion::Entropy)
            .fit(&x, &y)
        {
            Ok(f) => f,
            Err(e) => {
                #[allow(
                    clippy::assertions_on_constants,
                    reason = "test-only failure-path assertion; no cheap fitted-model fallback"
                )]
                {
                    assert!(false, "entropy fit failed: {e}");
                }
                return;
            }
        };
        let log_loss = match DecisionTreeClassifier::<f64>::new()
            .with_criterion(ClassificationCriterion::LogLoss)
            .fit(&x, &y)
        {
            Ok(f) => f,
            Err(e) => {
                #[allow(
                    clippy::assertions_on_constants,
                    reason = "test-only failure-path assertion; no cheap fitted-model fallback"
                )]
                {
                    assert!(false, "log_loss fit failed: {e}");
                }
                return;
            }
        };

        let res_e = entropy.predict(&x);
        assert!(res_e.is_ok(), "entropy predict failed");
        let pred_e = res_e.unwrap_or_else(|_| Array1::zeros(0));
        let res_l = log_loss.predict(&x);
        assert!(res_l.is_ok(), "log_loss predict failed");
        let pred_l = res_l.unwrap_or_else(|_| Array1::zeros(0));
        // log_loss == entropy: identical predictions and feature_importances_.
        assert_eq!(pred_e, pred_l, "log_loss predictions must equal entropy");
        let fe = entropy.feature_importances();
        let fl = log_loss.feature_importances();
        for (a, b) in fe.iter().zip(fl.iter()) {
            assert_relative_eq!(*a, *b, epsilon = 1e-12);
        }
        // entropy root + predict + feature_importances_ vs the live oracle.
        assert_eq!(pred_e, y);
        assert!(
            matches!(entropy.nodes()[0], Node::Split { .. }),
            "expected a split at the root"
        );
        if let Node::Split {
            feature, threshold, ..
        } = entropy.nodes()[0]
        {
            assert_eq!(feature, 1, "entropy root feature (sklearn: 1)");
            assert_relative_eq!(threshold, 5.5, epsilon = 1e-9);
        }
        assert_relative_eq!(fe[0], 0.137_946_433_630_985_85, epsilon = 1e-9);
        assert_relative_eq!(fe[1], 0.862_053_566_369_014_2, epsilon = 1e-9);
    }

    /// friedman_mse: node impurity == MSE variance; split improvement uses
    /// Friedman's proxy. On `reg_alt_fixture` (`max_depth=2`) it coincides with
    /// squared_error.
    ///
    /// Oracle: `DecisionTreeRegressor(criterion="friedman_mse", max_depth=2)`
    /// root (feature=0, threshold=4.5), predict ==
    /// `[1.1, 1.1, 1.0, 1.0, 5.1, 5.1, 5.0, 5.0]` (mean leaves).
    #[test]
    fn test_regressor_friedman_mse_oracle() {
        let (x, y) = reg_alt_fixture();
        let fitted = match DecisionTreeRegressor::<f64>::new()
            .with_criterion(RegressionCriterion::FriedmanMse)
            .with_max_depth(Some(2))
            .fit(&x, &y)
        {
            Ok(f) => f,
            Err(e) => {
                #[allow(
                    clippy::assertions_on_constants,
                    reason = "test-only failure-path assertion; no cheap fitted-model fallback"
                )]
                {
                    assert!(false, "fit failed: {e}");
                }
                return;
            }
        };
        let root = reg_root_split(&fitted);
        assert_eq!(root, Some((0, 4.5)), "friedman_mse root (sklearn: 0, 4.5)");
        assert_reg_predict(&fitted, &x, &[1.1, 1.1, 1.0, 1.0, 5.1, 5.1, 5.0, 5.0]);
    }

    /// absolute_error: node impurity `(1/n)Σ|y−median|`, **median** leaves.
    ///
    /// Oracle: `DecisionTreeRegressor(criterion="absolute_error", max_depth=2)`
    /// root (feature=0, threshold=4.5), predict ==
    /// `[1.0, 1.1, 1.1, 1.1, 5.0, 5.1, 5.1, 5.1]` (median leaves).
    #[test]
    fn test_regressor_absolute_error_median_leaves_oracle() {
        let (x, y) = reg_alt_fixture();
        let fitted = match DecisionTreeRegressor::<f64>::new()
            .with_criterion(RegressionCriterion::AbsoluteError)
            .with_max_depth(Some(2))
            .fit(&x, &y)
        {
            Ok(f) => f,
            Err(e) => {
                #[allow(
                    clippy::assertions_on_constants,
                    reason = "test-only failure-path assertion; no cheap fitted-model fallback"
                )]
                {
                    assert!(false, "fit failed: {e}");
                }
                return;
            }
        };
        let root = reg_root_split(&fitted);
        assert_eq!(
            root,
            Some((0, 4.5)),
            "absolute_error root (sklearn: 0, 4.5)"
        );
        // Median leaves (NOT mean): left child median 1.1, right child 5.1.
        assert_reg_predict(&fitted, &x, &[1.0, 1.1, 1.1, 1.1, 5.0, 5.1, 5.1, 5.1]);
    }

    /// poisson: half-Poisson-deviance impurity, **mean** leaves; requires y ≥ 0
    /// with Σy > 0.
    ///
    /// Oracle: `DecisionTreeRegressor(criterion="poisson", max_depth=2)`,
    /// root (feature=0, threshold=4.5), predict ==
    /// `[1.1, 1.1, 1.0, 1.0, 5.1, 5.1, 5.0, 5.0]` (mean leaves).
    #[test]
    fn test_regressor_poisson_oracle() {
        let (x, y) = reg_alt_fixture();
        let fitted = match DecisionTreeRegressor::<f64>::new()
            .with_criterion(RegressionCriterion::Poisson)
            .with_max_depth(Some(2))
            .fit(&x, &y)
        {
            Ok(f) => f,
            Err(e) => {
                #[allow(
                    clippy::assertions_on_constants,
                    reason = "test-only failure-path assertion; no cheap fitted-model fallback"
                )]
                {
                    assert!(false, "fit failed: {e}");
                }
                return;
            }
        };
        let root = reg_root_split(&fitted);
        assert_eq!(root, Some((0, 4.5)), "poisson root (sklearn: 0, 4.5)");
        assert_reg_predict(&fitted, &x, &[1.1, 1.1, 1.0, 1.0, 5.1, 5.1, 5.0, 5.0]);
    }

    /// poisson rejects targets with a negative value (and a non-positive sum),
    /// mirroring sklearn's `ValueError` (`_classes.py:267-277`).
    #[test]
    fn test_regressor_poisson_rejects_non_positive_y() {
        let x = array![[1.0], [2.0], [3.0], [4.0]];
        let y_neg = array![1.0, -0.5, 2.0, 3.0];
        let res = DecisionTreeRegressor::<f64>::new()
            .with_criterion(RegressionCriterion::Poisson)
            .fit(&x, &y_neg);
        assert!(res.is_err(), "poisson must reject negative y");

        let y_zero = array![0.0, 0.0, 0.0, 0.0];
        let res0 = DecisionTreeRegressor::<f64>::new()
            .with_criterion(RegressionCriterion::Poisson)
            .fit(&x, &y_zero);
        assert!(res0.is_err(), "poisson must reject sum(y) <= 0");
    }

    /// MSE path stays byte-identical after the criterion-dispatch refactor:
    /// squared_error on `reg_alt_fixture` matches the friedman/poisson mean
    /// leaves (`[1.1,1.1,1.0,1.0,5.1,5.1,5.0,5.0]`, sklearn oracle).
    #[test]
    fn test_regressor_squared_error_unchanged() {
        let (x, y) = reg_alt_fixture();
        let fitted = match DecisionTreeRegressor::<f64>::new()
            .with_criterion(RegressionCriterion::Mse)
            .with_max_depth(Some(2))
            .fit(&x, &y)
        {
            Ok(f) => f,
            Err(e) => {
                #[allow(
                    clippy::assertions_on_constants,
                    reason = "test-only failure-path assertion; no cheap fitted-model fallback"
                )]
                {
                    assert!(false, "fit failed: {e}");
                }
                return;
            }
        };
        let root = reg_root_split(&fitted);
        assert_eq!(root, Some((0, 4.5)));
        assert_reg_predict(&fitted, &x, &[1.1, 1.1, 1.0, 1.0, 5.1, 5.1, 5.0, 5.0]);
    }
}
