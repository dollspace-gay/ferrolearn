//! AdaBoost classifier.
//!
//! This module provides [`AdaBoostClassifier`], which implements the Adaptive
//! Boosting algorithm using decision tree stumps (depth-1 trees) as base
//! estimators. Two algorithm variants are supported:
//!
//! - **SAMME**: uses discrete class predictions and works with multiclass
//!   problems directly.
//! - **SAMME.R** (default): uses class probability estimates, typically giving
//!   better performance than SAMME.
//!
//! # Examples
//!
//! ```
//! use ferrolearn_tree::AdaBoostClassifier;
//! use ferrolearn_core::{Fit, Predict};
//! use ndarray::{array, Array1, Array2};
//!
//! let x = Array2::from_shape_vec((8, 2), vec![
//!     1.0, 2.0,  2.0, 3.0,  3.0, 3.0,  4.0, 4.0,
//!     5.0, 6.0,  6.0, 7.0,  7.0, 8.0,  8.0, 9.0,
//! ]).unwrap();
//! let y = array![0, 0, 0, 0, 1, 1, 1, 1];
//!
//! let model = AdaBoostClassifier::<f64>::new()
//!     .with_n_estimators(50)
//!     .with_random_state(42);
//! let fitted = model.fit(&x, &y).unwrap();
//! let preds = fitted.predict(&x).unwrap();
//! ```
//!
//! ## REQ status
//!
//! Mirrors `sklearn.ensemble.AdaBoostClassifier` (SAMME/SAMME.R,
//! `sklearn/ensemble/_weight_boosting.py`). See `.design/tree/adaboost.md`.
//! Non-test consumer: the `RsAdaBoostClassifier` PyO3 binding
//! (`ferrolearn-python/src/extras.rs`).
//!
//! **Algorithm default (R-DEV-6 deviation):** ferrolearn defaults to `Samme`.
//! sklearn 1.5.2's literal default is `'SAMME.R'`, but 1.5.2 deprecates it with
//! a `FutureWarning` (`_weight_boosting.py:526-534`) and 1.6 removed it,
//! making SAMME the sole option — a deliberate, documented deviation (#713).
//!
//! | REQ | Description | Status |
//! |-----|-------------|--------|
//! | REQ-1 | Param surface + numeric defaults (`n_estimators=50`, `learning_rate=1.0`, `random_state=None`) | SHIPPED |
//! | REQ-2 | SAMME `estimator_weight = lr*(log((1-err)/err)+log(K-1))` + reweight `*=exp(alpha·incorrect)` (`_weight_boosting.py:696-706`) | SHIPPED |
//! | REQ-3 | SAMME weighted error `Σ w_i·[pred≠y]/Σw` (`:676`) | SHIPPED |
//! | REQ-4 | Worse-than-random stop `err >= 1 - 1/K` (`:685`) | SHIPPED |
//! | REQ-6 | Weighted base-estimator fit (deterministic, no resample) — `predict` matches sklearn SAMME end-to-end (`:664`; verified on iris N≤50, K=3) | SHIPPED |
//! | REQ-7 | Perfect-fit `err<=0 → estimator_weight=1.0` + stop, before worse-than-random check (`:679-680`) | SHIPPED |
//! | REQ-11 | End-to-end SAMME `predict` parity vs live `AdaBoostClassifier(algorithm='SAMME')` — 0 mismatches (iris K=3 N≤50, binary N≤50) | SHIPPED |
//! | REQ-10 | `feature_importances_` = weighted-normalized mean of per-stump importances | SHIPPED |
//! | REQ-5 | `decision_function`/`predict_proba` exact form (`decision/(K-1)` scaling + softmax, `:799-870`) | NOT-STARTED (#712) |
//! | REQ-8 | SAMME.R `_boost_real` reweight correct for `K>2` (full `y_coding` xlogy, `:644-656`) | NOT-STARTED (#711) |
//! | REQ-9 | Pluggable base estimator + `estimator_errors_` attribute | NOT-STARTED (#714) |
//! | REQ-1b | `algorithm` default match-1.5.2-literal (`'SAMME.R'`) — deliberate R-DEV-6 deviation | NOT-STARTED (#713) |
//! | REQ-12 | Empty-ensemble worse-than-random raises `ValueError` (`:687-692`) | NOT-STARTED (#715) |
//! | REQ-13 | ferray substrate migration | NOT-STARTED (#716) |
//! | REQ-14 | Reject non-finite input (NaN+Inf): `fn reject_non_finite` at the top of `AdaBoostClassifier::fit` rejects NaN AND infinity. sklearn validates X up front (`_weight_boosting.py:133-141`, default `force_all_finite=True`) BEFORE any base learner ⇒ `ValueError`, even though the ferrolearn `DecisionTree` base now accepts NaN (#2277). Consumer: the existing `fit` entry (`RsAdaBoostClassifier` PyO3 reg). Pinned by `divergence_adaboost_classifier_nan_not_rejected` (live sklearn 1.5.2 raises). | SHIPPED |

use crate::decision_tree::{
    self, ClassificationCriterion, Node, build_weighted_classification_tree_with_feature_subset,
};
use ferrolearn_core::error::FerroError;
use ferrolearn_core::introspection::{HasClasses, HasFeatureImportances};
use ferrolearn_core::pipeline::{FittedPipelineEstimator, PipelineEstimator};
use ferrolearn_core::traits::{Fit, Predict};
use ndarray::{Array1, Array2};
use num_traits::{Float, FromPrimitive, ToPrimitive};

/// Reject `X` containing any non-finite value (NaN or infinity).
///
/// sklearn's `AdaBoostClassifier.fit` validates X up front via
/// `_validate_data(...)` with the default `force_all_finite=True`
/// (`sklearn/ensemble/_weight_boosting.py:133-141`), raising
/// `ValueError("Input X contains NaN.")` (`validation.py:147-154`) BEFORE any
/// base learner is built — so although ferrolearn's `DecisionTree` base now
/// accepts NaN (#2277), AdaBoost rejects it at its own entry, matching sklearn.
/// NaN AND infinity are both rejected. Never panics (R-CODE-2).
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
// Algorithm enum
// ---------------------------------------------------------------------------

/// AdaBoost algorithm variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdaBoostAlgorithm {
    /// SAMME: Stagewise Additive Modeling using a Multi-class Exponential loss.
    ///
    /// Uses discrete class predictions from each base estimator.
    Samme,
    /// SAMME.R: the "real" variant that uses class probability estimates.
    ///
    /// Generally outperforms SAMME.
    SammeR,
}

// ---------------------------------------------------------------------------
// AdaBoostClassifier
// ---------------------------------------------------------------------------

/// AdaBoost classifier using decision tree stumps as base estimators.
///
/// At each boosting round a decision tree stump (max depth = 1) is fitted
/// to the weighted training data. Misclassified samples receive higher
/// weight in subsequent rounds, allowing the ensemble to focus on hard
/// examples.
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
#[derive(Debug, Clone)]
pub struct AdaBoostClassifier<F> {
    /// Number of boosting stages (stumps).
    pub n_estimators: usize,
    /// Learning rate (shrinkage). Lower values require more estimators.
    pub learning_rate: f64,
    /// Algorithm variant (`SAMME` or `SAMME.R`).
    pub algorithm: AdaBoostAlgorithm,
    /// Random seed for reproducibility.
    pub random_state: Option<u64>,
    _marker: std::marker::PhantomData<F>,
}

impl<F: Float> AdaBoostClassifier<F> {
    /// Create a new `AdaBoostClassifier` with default settings.
    ///
    /// Defaults: `n_estimators = 50`, `learning_rate = 1.0`,
    /// `algorithm = SAMME`, `random_state = None`.
    ///
    /// The default algorithm is `SAMME` to match scikit-learn ≥ 1.4,
    /// which removed `SAMME.R` in 1.6 and made `SAMME` the only option.
    #[must_use]
    pub fn new() -> Self {
        Self {
            n_estimators: 50,
            learning_rate: 1.0,
            algorithm: AdaBoostAlgorithm::Samme,
            random_state: None,
            _marker: std::marker::PhantomData,
        }
    }

    /// Set the number of boosting stages.
    #[must_use]
    pub fn with_n_estimators(mut self, n: usize) -> Self {
        self.n_estimators = n;
        self
    }

    /// Set the learning rate.
    #[must_use]
    pub fn with_learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Set the algorithm variant.
    #[must_use]
    pub fn with_algorithm(mut self, algo: AdaBoostAlgorithm) -> Self {
        self.algorithm = algo;
        self
    }

    /// Set the random seed for reproducibility.
    #[must_use]
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }
}

impl<F: Float> Default for AdaBoostClassifier<F> {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// FittedAdaBoostClassifier
// ---------------------------------------------------------------------------

/// A fitted AdaBoost classifier.
///
/// Stores the sequence of stumps and their weights. Predictions are made
/// by weighted majority vote (SAMME) or weighted probability averaging
/// (SAMME.R).
#[derive(Debug, Clone)]
pub struct FittedAdaBoostClassifier<F> {
    /// Sorted unique class labels.
    classes: Vec<usize>,
    /// Sequence of fitted tree stumps.
    estimators: Vec<Vec<Node<F>>>,
    /// Weight of each estimator (SAMME) or kept for SAMME.R bookkeeping.
    estimator_weights: Vec<F>,
    /// Number of features.
    n_features: usize,
    /// Number of classes.
    n_classes: usize,
    /// Algorithm used.
    algorithm: AdaBoostAlgorithm,
    /// Per-feature importance scores aggregated across the boosted stumps,
    /// weighted by `estimator_weights` (normalized to sum to 1).
    feature_importances: Array1<F>,
}

impl<F: Float + Send + Sync + 'static> HasFeatureImportances<F> for FittedAdaBoostClassifier<F> {
    fn feature_importances(&self) -> &Array1<F> {
        &self.feature_importances
    }
}

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, Array1<usize>> for AdaBoostClassifier<F> {
    type Fitted = FittedAdaBoostClassifier<F>;
    type Error = FerroError;

    /// Fit the AdaBoost classifier.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if `x` and `y` have different
    /// numbers of samples.
    /// Returns [`FerroError::InsufficientSamples`] if there are no samples.
    /// Returns [`FerroError::InvalidParameter`] for invalid hyperparameters.
    fn fit(
        &self,
        x: &Array2<F>,
        y: &Array1<usize>,
    ) -> Result<FittedAdaBoostClassifier<F>, FerroError> {
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
                context: "AdaBoostClassifier requires at least one sample".into(),
            });
        }
        if self.n_estimators == 0 {
            return Err(FerroError::InvalidParameter {
                name: "n_estimators".into(),
                reason: "must be at least 1".into(),
            });
        }
        if self.learning_rate <= 0.0 {
            return Err(FerroError::InvalidParameter {
                name: "learning_rate".into(),
                reason: "must be positive".into(),
            });
        }
        // Reject non-finite X up front (before building any base learner),
        // matching sklearn (`_weight_boosting.py:133-141`).
        reject_non_finite(x)?;

        // Determine unique classes.
        let mut classes: Vec<usize> = y.iter().copied().collect();
        classes.sort_unstable();
        classes.dedup();
        let n_classes = classes.len();

        if n_classes < 2 {
            return Err(FerroError::InvalidParameter {
                name: "y".into(),
                reason: "need at least 2 distinct classes".into(),
            });
        }

        let y_mapped: Vec<usize> = y
            .iter()
            .map(|&c| classes.iter().position(|&cl| cl == c).unwrap())
            .collect();

        match self.algorithm {
            AdaBoostAlgorithm::Samme => {
                self.fit_samme(x, &y_mapped, n_samples, n_features, n_classes, &classes)
            }
            AdaBoostAlgorithm::SammeR => {
                self.fit_samme_r(x, &y_mapped, n_samples, n_features, n_classes, &classes)
            }
        }
    }
}

impl<F: Float + Send + Sync + 'static> AdaBoostClassifier<F> {
    /// Fit using the SAMME algorithm (discrete predictions).
    fn fit_samme(
        &self,
        x: &Array2<F>,
        y_mapped: &[usize],
        n_samples: usize,
        n_features: usize,
        n_classes: usize,
        classes: &[usize],
    ) -> Result<FittedAdaBoostClassifier<F>, FerroError> {
        let lr = F::from(self.learning_rate).unwrap();
        let n_f = F::from(n_samples).unwrap();
        let eps = F::from(1e-10).unwrap();

        // Initialize sample weights uniformly.
        let mut weights = vec![F::one() / n_f; n_samples];

        let all_features: Vec<usize> = (0..n_features).collect();
        let stump_params = decision_tree::TreeParams {
            max_depth: Some(1),
            min_samples_split: 2,
            min_samples_leaf: 1,
        };

        let mut estimators = Vec::with_capacity(self.n_estimators);
        let mut estimator_weights = Vec::with_capacity(self.n_estimators);

        for _ in 0..self.n_estimators {
            // sklearn fits each round's stump on the WEIGHTED data directly
            // (`estimator.fit(X, y, sample_weight=sample_weight)`,
            // `_weight_boosting.py:664`), deterministically — NO bootstrap/RNG.
            let tree = build_weighted_classification_tree_with_feature_subset(
                x,
                y_mapped,
                n_classes,
                &weights,
                &all_features,
                &stump_params,
                ClassificationCriterion::Gini,
            );

            // Compute predictions and weighted error.
            let mut weighted_error = F::zero();
            let mut preds = vec![0usize; n_samples];
            for i in 0..n_samples {
                let row = x.row(i);
                let leaf_idx = decision_tree::traverse(&tree, &row);
                if let Node::Leaf { value, .. } = tree[leaf_idx] {
                    preds[i] = value.to_f64().map_or(0, |f| f.round() as usize);
                }
                if preds[i] != y_mapped[i] {
                    weighted_error = weighted_error + weights[i];
                }
            }

            // Normalise error.
            let weight_sum: F = weights.iter().copied().fold(F::zero(), |a, b| a + b);
            let err = if weight_sum > F::zero() {
                weighted_error / weight_sum
            } else {
                F::from(0.5).unwrap()
            };

            // Stop if classification is perfect: sklearn returns
            // `(sample_weight, 1.0, 0.0)` the instant `estimator_error <= 0`
            // (`_boost_discrete:679-680`) and `fit` then breaks on
            // `estimator_error == 0` (`fit:180`). This guard MUST come BEFORE the
            // worse-than-random check, matching sklearn's ordering.
            if err <= eps {
                estimators.push(tree);
                estimator_weights.push(F::one());
                break;
            }

            // If error is too high or zero, stop or skip.
            if err >= F::one() - F::one() / F::from(n_classes).unwrap() {
                // Error too high; stop boosting.
                if estimators.is_empty() {
                    // Keep at least one estimator.
                    estimators.push(tree);
                    estimator_weights.push(F::one());
                }
                break;
            }

            // Estimator weight: SAMME formula.
            let alpha = lr * ((F::one() - err).max(eps) / err.max(eps)).ln()
                + lr * (F::from(n_classes - 1).unwrap()).ln();

            // Update sample weights.
            for i in 0..n_samples {
                if preds[i] != y_mapped[i] {
                    weights[i] = weights[i] * alpha.exp();
                }
            }

            // Normalise weights.
            let new_sum: F = weights.iter().copied().fold(F::zero(), |a, b| a + b);
            if new_sum > F::zero() {
                for w in &mut weights {
                    *w = *w / new_sum;
                }
            }

            estimators.push(tree);
            estimator_weights.push(alpha);
        }

        let feature_importances = decision_tree::aggregate_tree_importances(
            &estimators,
            None,
            Some(&estimator_weights),
            n_features,
        );

        Ok(FittedAdaBoostClassifier {
            classes: classes.to_vec(),
            estimators,
            estimator_weights,
            n_features,
            n_classes,
            algorithm: AdaBoostAlgorithm::Samme,
            feature_importances,
        })
    }

    /// Fit using the SAMME.R algorithm (real-valued / probability-based).
    fn fit_samme_r(
        &self,
        x: &Array2<F>,
        y_mapped: &[usize],
        n_samples: usize,
        n_features: usize,
        n_classes: usize,
        classes: &[usize],
    ) -> Result<FittedAdaBoostClassifier<F>, FerroError> {
        let lr = F::from(self.learning_rate).unwrap();
        let n_f = F::from(n_samples).unwrap();
        let eps = F::from(1e-10).unwrap();
        let k_f = F::from(n_classes).unwrap();

        // Initialize sample weights uniformly.
        let mut weights = vec![F::one() / n_f; n_samples];

        let all_features: Vec<usize> = (0..n_features).collect();
        let stump_params = decision_tree::TreeParams {
            max_depth: Some(1),
            min_samples_split: 2,
            min_samples_leaf: 1,
        };

        let mut estimators = Vec::with_capacity(self.n_estimators);
        let mut estimator_weights = Vec::with_capacity(self.n_estimators);

        for _ in 0..self.n_estimators {
            // sklearn fits each round's stump on the WEIGHTED data directly
            // (`estimator.fit(X, y, sample_weight=sample_weight)`,
            // `_weight_boosting.py:605`), deterministically — NO bootstrap/RNG.
            // (The SAMME.R reweight math, REQ-8/#713, is unchanged here — only
            // the stump FIT is rewired off the systematic resample.)
            let tree = build_weighted_classification_tree_with_feature_subset(
                x,
                y_mapped,
                n_classes,
                &weights,
                &all_features,
                &stump_params,
                ClassificationCriterion::Gini,
            );

            // Get class probability estimates for each sample.
            let mut proba = vec![vec![F::zero(); n_classes]; n_samples];
            for (i, proba_row) in proba.iter_mut().enumerate() {
                let row = x.row(i);
                let leaf_idx = decision_tree::traverse(&tree, &row);
                if let Node::Leaf {
                    class_distribution: Some(ref dist),
                    ..
                } = tree[leaf_idx]
                {
                    for (k, &p) in dist.iter().enumerate() {
                        proba_row[k] = p.max(eps);
                    }
                } else {
                    // Fallback: uniform.
                    for val in proba_row.iter_mut() {
                        *val = F::one() / k_f;
                    }
                }
                // Normalise.
                let row_sum: F = proba_row.iter().copied().fold(F::zero(), |a, b| a + b);
                if row_sum > F::zero() {
                    for val in proba_row.iter_mut() {
                        *val = *val / row_sum;
                    }
                }
            }

            // Stop if classification is perfect: sklearn's SAMME.R `_boost_real`
            // computes `incorrect = argmax(proba) != y`,
            // `estimator_error = mean(average(incorrect, weights))`, and returns
            // `(sample_weight, 1.0, 0.0)` the instant `estimator_error <= 0`
            // (`_boost_real:616-623`); `fit` then breaks on `error == 0`
            // (`fit:180`). SAMME.R uses an estimator weight of `1.0` regardless.
            let mut weighted_error = F::zero();
            for (i, proba_row) in proba.iter().enumerate() {
                let mut argmax = 0usize;
                let mut best = proba_row[0];
                for (k, &p) in proba_row.iter().enumerate().skip(1) {
                    if p > best {
                        best = p;
                        argmax = k;
                    }
                }
                if argmax != y_mapped[i] {
                    weighted_error = weighted_error + weights[i];
                }
            }
            let weight_sum: F = weights.iter().copied().fold(F::zero(), |a, b| a + b);
            let err = if weight_sum > F::zero() {
                weighted_error / weight_sum
            } else {
                F::zero()
            };
            if err <= eps {
                estimators.push(tree);
                estimator_weights.push(F::one());
                break;
            }

            // SAMME.R weight update: based on log-probability.
            // h_k(x) = (K-1) * (log(p_k(x)) - (1/K) * sum_j log(p_j(x)))
            // Then update: w_i *= exp(-(K-1)/K * lr * sum_k y_{ik} * log(p_k(x)))
            // Simplified: w_i *= exp(-lr * (K-1)/K * log(p_{y_i}(x)))
            let factor = lr * (k_f - F::one()) / k_f;
            let mut any_update = false;

            for i in 0..n_samples {
                let p_correct = proba[i][y_mapped[i]].max(eps);
                let exponent = -factor * p_correct.ln();
                weights[i] = weights[i] * exponent.exp();
                if exponent.abs() > eps {
                    any_update = true;
                }
            }

            // Normalise weights.
            let new_sum: F = weights.iter().copied().fold(F::zero(), |a, b| a + b);
            if new_sum > F::zero() {
                for w in &mut weights {
                    *w = *w / new_sum;
                }
            }

            estimators.push(tree);
            estimator_weights.push(F::one()); // SAMME.R uses equal weight; prediction uses probabilities.

            if !any_update {
                break;
            }
        }

        let feature_importances = decision_tree::aggregate_tree_importances(
            &estimators,
            None,
            Some(&estimator_weights),
            n_features,
        );

        Ok(FittedAdaBoostClassifier {
            classes: classes.to_vec(),
            estimators,
            estimator_weights,
            n_features,
            n_classes,
            algorithm: AdaBoostAlgorithm::SammeR,
            feature_importances,
        })
    }
}

impl<F: Float + Send + Sync + 'static> Predict<Array2<F>> for FittedAdaBoostClassifier<F> {
    type Output = Array1<usize>;
    type Error = FerroError;

    /// Predict class labels.
    ///
    /// - **SAMME**: weighted majority vote using estimator weights.
    /// - **SAMME.R**: weighted average of log-probabilities.
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

        match self.algorithm {
            AdaBoostAlgorithm::Samme => self.predict_samme(x, n_samples),
            AdaBoostAlgorithm::SammeR => self.predict_samme_r(x, n_samples),
        }
    }
}

impl<F: Float + Send + Sync + 'static> FittedAdaBoostClassifier<F> {
    /// Predict using SAMME (weighted majority vote).
    fn predict_samme(&self, x: &Array2<F>, n_samples: usize) -> Result<Array1<usize>, FerroError> {
        let mut predictions = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let row = x.row(i);
            let mut class_scores = vec![F::zero(); self.n_classes];

            for (t, tree_nodes) in self.estimators.iter().enumerate() {
                let leaf_idx = decision_tree::traverse(tree_nodes, &row);
                if let Node::Leaf { value, .. } = tree_nodes[leaf_idx] {
                    let class_idx = value.to_f64().map_or(0, |f| f.round() as usize);
                    if class_idx < self.n_classes {
                        class_scores[class_idx] =
                            class_scores[class_idx] + self.estimator_weights[t];
                    }
                }
            }

            let best = class_scores
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map_or(0, |(k, _)| k);
            predictions[i] = self.classes[best];
        }

        Ok(predictions)
    }

    /// Predict using SAMME.R (weighted probability averaging).
    fn predict_samme_r(
        &self,
        x: &Array2<F>,
        n_samples: usize,
    ) -> Result<Array1<usize>, FerroError> {
        let eps = F::from(1e-10).unwrap();
        let k_f = F::from(self.n_classes).unwrap();
        let k_minus_1 = k_f - F::one();

        let mut predictions = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let row = x.row(i);
            let mut accumulated = vec![F::zero(); self.n_classes];

            for tree_nodes in &self.estimators {
                let leaf_idx = decision_tree::traverse(tree_nodes, &row);
                if let Node::Leaf {
                    class_distribution: Some(ref dist),
                    ..
                } = tree_nodes[leaf_idx]
                {
                    // h_k(x) = (K-1) * (log(p_k) - mean(log(p_j)))
                    let log_probs: Vec<F> = dist.iter().map(|&p| p.max(eps).ln()).collect();
                    let mean_log: F = log_probs.iter().copied().fold(F::zero(), |a, b| a + b) / k_f;

                    for k in 0..self.n_classes {
                        accumulated[k] = accumulated[k] + k_minus_1 * (log_probs[k] - mean_log);
                    }
                } else {
                    // Leaf without distribution: predict from value.
                    if let Node::Leaf { value, .. } = tree_nodes[leaf_idx] {
                        let class_idx = value.to_f64().map_or(0, |f| f.round() as usize);
                        if class_idx < self.n_classes {
                            accumulated[class_idx] = accumulated[class_idx] + F::one();
                        }
                    }
                }
            }

            let best = accumulated
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map_or(0, |(k, _)| k);
            predictions[i] = self.classes[best];
        }

        Ok(predictions)
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

    /// Predict class probabilities for each sample. Mirrors sklearn's
    /// `AdaBoostClassifier.predict_proba`.
    ///
    /// SAMME: normalizes the weighted-vote vector per row.
    /// SAMME.R: applies softmax to the accumulated `(K-1)*(log p_k - mean)`
    /// scores per row.
    ///
    /// Returns shape `(n_samples, n_classes)`; rows sum to 1.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of features
    /// does not match the fitted model.
    pub fn predict_proba(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        if x.ncols() != self.n_features {
            return Err(FerroError::ShapeMismatch {
                expected: vec![self.n_features],
                actual: vec![x.ncols()],
                context: "number of features must match fitted model".into(),
            });
        }
        let n_samples = x.nrows();
        let n_classes = self.n_classes;
        let mut proba = Array2::<F>::zeros((n_samples, n_classes));

        match self.algorithm {
            AdaBoostAlgorithm::Samme => {
                for i in 0..n_samples {
                    let row = x.row(i);
                    let mut scores = vec![F::zero(); n_classes];
                    for (t, tree_nodes) in self.estimators.iter().enumerate() {
                        let leaf_idx = decision_tree::traverse(tree_nodes, &row);
                        if let Node::Leaf { value, .. } = tree_nodes[leaf_idx] {
                            let class_idx = value.to_f64().map_or(0, |f| f.round() as usize);
                            if class_idx < n_classes {
                                scores[class_idx] = scores[class_idx] + self.estimator_weights[t];
                            }
                        }
                    }
                    let total: F = scores.iter().copied().fold(F::zero(), |a, b| a + b);
                    if total > F::zero() {
                        for k in 0..n_classes {
                            proba[[i, k]] = scores[k] / total;
                        }
                    } else {
                        let u = F::one() / F::from(n_classes).unwrap();
                        for k in 0..n_classes {
                            proba[[i, k]] = u;
                        }
                    }
                }
            }
            AdaBoostAlgorithm::SammeR => {
                let eps = F::from(1e-10).unwrap();
                let k_f = F::from(n_classes).unwrap();
                let k_minus_1 = k_f - F::one();
                for i in 0..n_samples {
                    let row = x.row(i);
                    let mut accumulated = vec![F::zero(); n_classes];
                    for tree_nodes in &self.estimators {
                        let leaf_idx = decision_tree::traverse(tree_nodes, &row);
                        if let Node::Leaf {
                            class_distribution: Some(ref dist),
                            ..
                        } = tree_nodes[leaf_idx]
                        {
                            let log_probs: Vec<F> = dist.iter().map(|&p| p.max(eps).ln()).collect();
                            let mean_log: F =
                                log_probs.iter().copied().fold(F::zero(), |a, b| a + b) / k_f;
                            for k in 0..n_classes {
                                accumulated[k] =
                                    accumulated[k] + k_minus_1 * (log_probs[k] - mean_log);
                            }
                        } else if let Node::Leaf { value, .. } = tree_nodes[leaf_idx] {
                            let class_idx = value.to_f64().map_or(0, |f| f.round() as usize);
                            if class_idx < n_classes {
                                accumulated[class_idx] = accumulated[class_idx] + F::one();
                            }
                        }
                    }
                    // Softmax of accumulated.
                    let max_score = accumulated
                        .iter()
                        .copied()
                        .fold(F::neg_infinity(), |a, b| if b > a { b } else { a });
                    let mut sum_exp = F::zero();
                    for k in 0..n_classes {
                        let e = (accumulated[k] - max_score).exp();
                        proba[[i, k]] = e;
                        sum_exp = sum_exp + e;
                    }
                    if sum_exp > F::zero() {
                        for k in 0..n_classes {
                            proba[[i, k]] = proba[[i, k]] / sum_exp;
                        }
                    }
                }
            }
        }
        Ok(proba)
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

    /// Per-class raw scores. Mirrors sklearn's
    /// `AdaBoostClassifier.decision_function`.
    ///
    /// SAMME: returns the cumulative weighted vote per class (unnormalized).
    /// SAMME.R: returns the accumulated `(K-1)*(log p_k - mean log p)`
    /// scores.
    ///
    /// Returns shape `(n_samples, n_classes)`. The argmax of each row
    /// agrees with [`Predict::predict`].
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of features
    /// does not match the fitted model.
    pub fn decision_function(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        if x.ncols() != self.n_features {
            return Err(FerroError::ShapeMismatch {
                expected: vec![self.n_features],
                actual: vec![x.ncols()],
                context: "number of features must match fitted model".into(),
            });
        }
        let n_samples = x.nrows();
        let n_classes = self.n_classes;
        let mut out = Array2::<F>::zeros((n_samples, n_classes));

        match self.algorithm {
            AdaBoostAlgorithm::Samme => {
                for i in 0..n_samples {
                    let row = x.row(i);
                    for (t, tree_nodes) in self.estimators.iter().enumerate() {
                        let leaf_idx = decision_tree::traverse(tree_nodes, &row);
                        if let Node::Leaf { value, .. } = tree_nodes[leaf_idx] {
                            let class_idx = value.to_f64().map_or(0, |f| f.round() as usize);
                            if class_idx < n_classes {
                                out[[i, class_idx]] =
                                    out[[i, class_idx]] + self.estimator_weights[t];
                            }
                        }
                    }
                }
            }
            AdaBoostAlgorithm::SammeR => {
                let eps = F::from(1e-10).unwrap();
                let k_f = F::from(n_classes).unwrap();
                let k_minus_1 = k_f - F::one();
                for i in 0..n_samples {
                    let row = x.row(i);
                    for tree_nodes in &self.estimators {
                        let leaf_idx = decision_tree::traverse(tree_nodes, &row);
                        if let Node::Leaf {
                            class_distribution: Some(ref dist),
                            ..
                        } = tree_nodes[leaf_idx]
                        {
                            let log_probs: Vec<F> = dist.iter().map(|&p| p.max(eps).ln()).collect();
                            let mean_log: F =
                                log_probs.iter().copied().fold(F::zero(), |a, b| a + b) / k_f;
                            for k in 0..n_classes {
                                out[[i, k]] = out[[i, k]] + k_minus_1 * (log_probs[k] - mean_log);
                            }
                        } else if let Node::Leaf { value, .. } = tree_nodes[leaf_idx] {
                            let class_idx = value.to_f64().map_or(0, |f| f.round() as usize);
                            if class_idx < n_classes {
                                out[[i, class_idx]] = out[[i, class_idx]] + F::one();
                            }
                        }
                    }
                }
            }
        }
        Ok(out)
    }
}

impl<F: Float + Send + Sync + 'static> HasClasses for FittedAdaBoostClassifier<F> {
    fn classes(&self) -> &[usize] {
        &self.classes
    }

    fn n_classes(&self) -> usize {
        self.classes.len()
    }
}

// Pipeline integration.
impl<F: Float + ToPrimitive + FromPrimitive + Send + Sync + 'static> PipelineEstimator<F>
    for AdaBoostClassifier<F>
{
    fn fit_pipeline(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
    ) -> Result<Box<dyn FittedPipelineEstimator<F>>, FerroError> {
        let y_usize: Array1<usize> = y.mapv(|v| v.to_usize().unwrap_or(0));
        let fitted = self.fit(x, &y_usize)?;
        Ok(Box::new(FittedAdaBoostPipelineAdapter(fitted)))
    }
}

/// Pipeline adapter for `FittedAdaBoostClassifier<F>`.
struct FittedAdaBoostPipelineAdapter<F: Float + Send + Sync + 'static>(FittedAdaBoostClassifier<F>);

impl<F: Float + ToPrimitive + FromPrimitive + Send + Sync + 'static> FittedPipelineEstimator<F>
    for FittedAdaBoostPipelineAdapter<F>
{
    fn predict_pipeline(&self, x: &Array2<F>) -> Result<Array1<F>, FerroError> {
        let preds = self.0.predict(x)?;
        Ok(preds.mapv(|v| F::from_usize(v).unwrap_or_else(F::nan)))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    // -- SAMME.R tests --

    #[test]
    fn test_adaboost_sammer_binary_simple() {
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 5.0, 6.0, 6.0, 7.0, 7.0, 8.0, 8.0, 9.0,
            ],
        )
        .unwrap();
        let y = array![0, 0, 0, 0, 1, 1, 1, 1];

        let model = AdaBoostClassifier::<f64>::new()
            .with_n_estimators(50)
            .with_random_state(42);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        assert_eq!(preds.len(), 8);
        for i in 0..4 {
            assert_eq!(preds[i], 0);
        }
        for i in 4..8 {
            assert_eq!(preds[i], 1);
        }
    }

    #[test]
    fn test_adaboost_sammer_multiclass() {
        let x = Array2::from_shape_vec((9, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
            .unwrap();
        let y = array![0, 0, 0, 1, 1, 1, 2, 2, 2];

        let model = AdaBoostClassifier::<f64>::new()
            .with_n_estimators(50)
            .with_random_state(42);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        assert_eq!(preds.len(), 9);
        let correct = preds.iter().zip(y.iter()).filter(|(p, t)| p == t).count();
        assert!(
            correct >= 5,
            "Expected at least 5/9 correct, got {correct}/9"
        );
    }

    // -- SAMME tests --

    #[test]
    fn test_adaboost_samme_binary_simple() {
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 5.0, 6.0, 6.0, 7.0, 7.0, 8.0, 8.0, 9.0,
            ],
        )
        .unwrap();
        let y = array![0, 0, 0, 0, 1, 1, 1, 1];

        let model = AdaBoostClassifier::<f64>::new()
            .with_n_estimators(50)
            .with_algorithm(AdaBoostAlgorithm::Samme)
            .with_random_state(42);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        assert_eq!(preds.len(), 8);
        for i in 0..4 {
            assert_eq!(preds[i], 0);
        }
        for i in 4..8 {
            assert_eq!(preds[i], 1);
        }
    }

    #[test]
    fn test_adaboost_samme_multiclass() {
        let x = Array2::from_shape_vec((9, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
            .unwrap();
        let y = array![0, 0, 0, 1, 1, 1, 2, 2, 2];

        let model = AdaBoostClassifier::<f64>::new()
            .with_n_estimators(50)
            .with_algorithm(AdaBoostAlgorithm::Samme)
            .with_random_state(42);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        assert_eq!(preds.len(), 9);
        let correct = preds.iter().zip(y.iter()).filter(|(p, t)| p == t).count();
        assert!(
            correct >= 5,
            "Expected at least 5/9 correct for SAMME multiclass, got {correct}/9"
        );
    }

    // -- Common tests --

    #[test]
    fn test_adaboost_has_classes() {
        let x = Array2::from_shape_vec((6, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = array![0, 1, 2, 0, 1, 2];

        let model = AdaBoostClassifier::<f64>::new()
            .with_n_estimators(5)
            .with_random_state(0);
        let fitted = model.fit(&x, &y).unwrap();

        assert_eq!(fitted.classes(), &[0, 1, 2]);
        assert_eq!(fitted.n_classes(), 3);
    }

    #[test]
    fn test_adaboost_reproducibility() {
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 5.0, 6.0, 6.0, 7.0, 7.0, 8.0, 8.0, 9.0,
            ],
        )
        .unwrap();
        let y = array![0, 0, 0, 0, 1, 1, 1, 1];

        let model = AdaBoostClassifier::<f64>::new()
            .with_n_estimators(10)
            .with_random_state(42);

        let fitted1 = model.fit(&x, &y).unwrap();
        let fitted2 = model.fit(&x, &y).unwrap();

        let preds1 = fitted1.predict(&x).unwrap();
        let preds2 = fitted2.predict(&x).unwrap();
        assert_eq!(preds1, preds2);
    }

    #[test]
    fn test_adaboost_shape_mismatch_fit() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = array![0, 1];

        let model = AdaBoostClassifier::<f64>::new().with_n_estimators(5);
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_adaboost_shape_mismatch_predict() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let y = array![0, 0, 1, 1];

        let model = AdaBoostClassifier::<f64>::new()
            .with_n_estimators(5)
            .with_random_state(0);
        let fitted = model.fit(&x, &y).unwrap();

        let x_bad = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        assert!(fitted.predict(&x_bad).is_err());
    }

    #[test]
    fn test_adaboost_empty_data() {
        let x = Array2::<f64>::zeros((0, 2));
        let y = Array1::<usize>::zeros(0);

        let model = AdaBoostClassifier::<f64>::new().with_n_estimators(5);
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_adaboost_single_class() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![0, 0, 0];

        let model = AdaBoostClassifier::<f64>::new().with_n_estimators(5);
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_adaboost_zero_estimators() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![0, 0, 1, 1];

        let model = AdaBoostClassifier::<f64>::new().with_n_estimators(0);
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_adaboost_invalid_learning_rate() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![0, 0, 1, 1];

        let model = AdaBoostClassifier::<f64>::new()
            .with_n_estimators(5)
            .with_learning_rate(0.0);
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_adaboost_pipeline_integration() {
        let x = Array2::from_shape_vec((6, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);

        let model = AdaBoostClassifier::<f64>::new()
            .with_n_estimators(10)
            .with_random_state(42);
        let fitted = model.fit_pipeline(&x, &y).unwrap();
        let preds = fitted.predict_pipeline(&x).unwrap();
        assert_eq!(preds.len(), 6);
    }

    #[test]
    fn test_adaboost_f32_support() {
        let x = Array2::from_shape_vec((6, 1), vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = array![0, 0, 0, 1, 1, 1];

        let model = AdaBoostClassifier::<f32>::new()
            .with_n_estimators(10)
            .with_random_state(42);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 6);
    }

    #[test]
    fn test_adaboost_default_trait() {
        let model = AdaBoostClassifier::<f64>::default();
        assert_eq!(model.n_estimators, 50);
        assert!((model.learning_rate - 1.0).abs() < 1e-10);
        assert_eq!(model.algorithm, AdaBoostAlgorithm::Samme);
    }

    #[test]
    fn test_adaboost_non_contiguous_labels() {
        let x = Array2::from_shape_vec((6, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = array![10, 10, 10, 20, 20, 20];

        let model = AdaBoostClassifier::<f64>::new()
            .with_n_estimators(20)
            .with_random_state(42);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        assert_eq!(preds.len(), 6);
        for &p in &preds {
            assert!(p == 10 || p == 20);
        }
    }

    #[test]
    fn test_adaboost_sammer_learning_rate_effect() {
        let x =
            Array2::from_shape_vec((8, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let y = array![0, 0, 0, 0, 1, 1, 1, 1];

        // Low learning rate should still work (just slower convergence).
        let model = AdaBoostClassifier::<f64>::new()
            .with_n_estimators(50)
            .with_learning_rate(0.1)
            .with_random_state(42);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 8);
    }

    #[test]
    fn test_adaboost_samme_learning_rate_effect() {
        let x =
            Array2::from_shape_vec((8, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let y = array![0, 0, 0, 0, 1, 1, 1, 1];

        let model = AdaBoostClassifier::<f64>::new()
            .with_n_estimators(50)
            .with_algorithm(AdaBoostAlgorithm::Samme)
            .with_learning_rate(0.5)
            .with_random_state(42);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 8);
    }

    #[test]
    fn test_adaboost_many_features() {
        // 4 features, only first one is informative.
        let x = Array2::from_shape_vec(
            (8, 4),
            vec![
                1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0,
                5.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 7.0, 0.0, 0.0, 0.0, 8.0, 0.0, 0.0, 0.0,
            ],
        )
        .unwrap();
        let y = array![0, 0, 0, 0, 1, 1, 1, 1];

        let model = AdaBoostClassifier::<f64>::new()
            .with_n_estimators(20)
            .with_random_state(42);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 8);
    }

    #[test]
    fn test_adaboost_4_classes() {
        let x = Array2::from_shape_vec(
            (12, 1),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        )
        .unwrap();
        let y = array![0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3];

        let model = AdaBoostClassifier::<f64>::new()
            .with_n_estimators(50)
            .with_random_state(42);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        assert_eq!(preds.len(), 12);
        assert_eq!(fitted.n_classes(), 4);
    }

    #[test]
    fn test_adaboost_sammer_single_estimator() {
        let x = Array2::from_shape_vec((6, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = array![0, 0, 0, 1, 1, 1];

        let model = AdaBoostClassifier::<f64>::new()
            .with_n_estimators(1)
            .with_random_state(42);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 6);
    }

    #[test]
    fn test_adaboost_samme_single_estimator() {
        let x = Array2::from_shape_vec((6, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = array![0, 0, 0, 1, 1, 1];

        let model = AdaBoostClassifier::<f64>::new()
            .with_n_estimators(1)
            .with_algorithm(AdaBoostAlgorithm::Samme)
            .with_random_state(42);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 6);
    }

    #[test]
    fn test_adaboost_negative_learning_rate() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![0, 0, 1, 1];

        let model = AdaBoostClassifier::<f64>::new()
            .with_n_estimators(5)
            .with_learning_rate(-0.1);
        assert!(model.fit(&x, &y).is_err());
    }
}
