//! Sequential feature selection via forward or backward search.
//!
//! [`SequentialFeatureSelector`] greedily adds (forward) or removes (backward)
//! features one at a time, evaluating each candidate subset with a
//! user-supplied scoring callback.
//!
//! # Algorithm
//!
//! **Forward**: Start with an empty feature set. At each step, try adding
//! each remaining feature, evaluate the score, and keep the addition that
//! yields the highest score. Repeat until `n_features_to_select` features
//! have been selected.
//!
//! **Backward**: Start with all features. At each step, try removing each
//! remaining feature, evaluate the score, and keep the removal that yields
//! the highest score. Repeat until `n_features_to_select` features remain.
//!
//! Translation target: scikit-learn 1.5.2 `class SequentialFeatureSelector`
//! (`sklearn/feature_selection/_sequential.py:19`). Design:
//! `.design/preprocess/sequential_feature_selector.md`. Tracking: #1283.
//!
//! Note: ferrolearn scores subsets via a user-supplied callback; sklearn uses a
//! wrapped estimator + cross-validation (`cross_val_score(...).mean()`). The
//! greedy search shape matches; the estimator/CV scoring is NOT-STARTED (#1286).
//!
//! `## REQ status`
//!
//! | REQ | Status | Anchor |
//! |---|---|---|
//! | REQ-1 greedy forward/backward search + lowest-index tie-break | SHIPPED | `fit`/`forward_search`/`backward_search`; sklearn `_sequential.py:280-294` |
//! | REQ-2 error contracts (n=0, >n_features, 0-rows, y-len, score_fn) | SHIPPED | `fit` / `transform` guards; sklearn `_sequential.py:211-216` |
//! | REQ-8 `< n_features` + `ensure_min_features=2` validation | SHIPPED (#1284, #1285) | `fit` guards; sklearn `_sequential.py:214`,`:227-228` |
//! | REQ-3 wrapped estimator + cross_val_score scoring | NOT-STARTED (#1286) | sklearn `_sequential.py:286-293` |
//! | REQ-4 `n_features_to_select="auto"` default | NOT-STARTED (#1287) | sklearn `_sequential.py:219-225` |
//! | REQ-5 `tol` early-stop + forward tol>0 validation | NOT-STARTED (#1288) | sklearn `_sequential.py:233-236`,`:258-259` |
//! | REQ-6 float `n_features_to_select` fraction | NOT-STARTED (#1289) | sklearn `_sequential.py:159`,`:230-231` |
//! | REQ-7 `cv` / `scoring` / `n_jobs` params | NOT-STARTED (#1290) | sklearn `_sequential.py:164-166` |
//! | REQ-9 SelectorMixin surface + `n_features_to_select_` | SHIPPED (scoped) / residual open (#1291) | [`crate::SelectorMixin`] supplies dense support masks, inverse zero-fill, and feature-name filtering; sklearn-named `n_features_to_select_` attr remains open |
//! | REQ-10 PyO3 binding | NOT-STARTED (#1292) | `ferrolearn-python/src/` (absent) |
//! | REQ-11 ferray substrate | NOT-STARTED (#1293) | R-SUBSTRATE |

use ferrolearn_core::error::FerroError;
use ferrolearn_core::traits::Transform;
use ndarray::{Array1, Array2};
use num_traits::Float;

// ---------------------------------------------------------------------------
// Direction
// ---------------------------------------------------------------------------

/// Search direction for [`SequentialFeatureSelector`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    /// Start with no features and greedily add one at a time.
    Forward,
    /// Start with all features and greedily remove one at a time.
    Backward,
}

// ---------------------------------------------------------------------------
// SequentialFeatureSelector (unfitted)
// ---------------------------------------------------------------------------

/// A greedy sequential feature selector.
///
/// # Examples
///
/// ```
/// use ferrolearn_preprocess::sequential_feature_selector::{
///     SequentialFeatureSelector, Direction,
/// };
/// use ndarray::{array, Array1, Array2};
///
/// let sfs = SequentialFeatureSelector::new(1, Direction::Forward);
/// let x = array![[1.0, 10.0, 0.1],
///                 [2.0, 20.0, 0.2],
///                 [3.0, 30.0, 0.3]];
/// let y = array![1.0, 2.0, 3.0];
///
/// // Score function: sum of selected column means (higher is better)
/// let score_fn = |x_sub: &Array2<f64>, _y: &Array1<f64>| -> Result<f64, _> {
///     let mean_sum: f64 = x_sub.columns().into_iter()
///         .map(|c| c.sum() / c.len() as f64)
///         .sum();
///     Ok(mean_sum)
/// };
///
/// let fitted = sfs.fit(&x, &y, score_fn).unwrap();
/// assert_eq!(fitted.selected_indices().len(), 1);
/// // Feature 1 (column means 10,20,30 → mean 20) should be selected
/// assert_eq!(fitted.selected_indices(), &[1]);
/// ```
#[must_use]
#[derive(Debug, Clone)]
pub struct SequentialFeatureSelector {
    /// Number of features to select.
    n_features_to_select: usize,
    /// Search direction (forward or backward).
    direction: Direction,
}

impl SequentialFeatureSelector {
    /// Create a new `SequentialFeatureSelector`.
    ///
    /// # Parameters
    ///
    /// - `n_features_to_select` — how many features to keep.
    /// - `direction` — [`Direction::Forward`] or [`Direction::Backward`].
    pub fn new(n_features_to_select: usize, direction: Direction) -> Self {
        Self {
            n_features_to_select,
            direction,
        }
    }

    /// Return the number of features to select.
    #[must_use]
    pub fn n_features_to_select(&self) -> usize {
        self.n_features_to_select
    }

    /// Return the search direction.
    #[must_use]
    pub fn direction(&self) -> Direction {
        self.direction
    }

    /// Fit the selector by evaluating feature subsets with a scoring function.
    ///
    /// # Parameters
    ///
    /// - `x` — the feature matrix (`n_samples x n_features`).
    /// - `y` — the target vector (`n_samples`).
    /// - `score_fn` — a callback `(&Array2<F>, &Array1<F>) -> Result<F, FerroError>`
    ///   that evaluates the quality of a feature subset.
    ///
    /// # Errors
    ///
    /// - [`FerroError::InvalidParameter`] if `n_features_to_select` is zero or
    ///   exceeds the number of features.
    /// - [`FerroError::ShapeMismatch`] if `x` and `y` have mismatched lengths.
    /// - Propagates errors from `score_fn`.
    pub fn fit<F: Float + Send + Sync + 'static>(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
        score_fn: impl Fn(&Array2<F>, &Array1<F>) -> Result<F, FerroError>,
    ) -> Result<FittedSequentialFeatureSelector<F>, FerroError> {
        let n_features = x.ncols();
        let n_samples = x.nrows();

        // sklearn validates a minimum feature count BEFORE resolving
        // `n_features_to_select`, so a 1-feature X is rejected for the
        // "minimum of 2" reason regardless of `n_features_to_select`.
        // Mirrors `_sequential.py:214`:
        //   `X = self._validate_data(X, ..., ensure_min_features=2, ...)`
        if n_features < 2 {
            return Err(FerroError::InvalidParameter {
                name: "x".into(),
                reason: format!(
                    "Found array with {n_features} feature(s) while a minimum of 2 is required by SequentialFeatureSelector"
                ),
            });
        }

        if self.n_features_to_select == 0 {
            return Err(FerroError::InvalidParameter {
                name: "n_features_to_select".into(),
                reason: "must be at least 1".into(),
            });
        }
        // sklearn requires strictly `n_features_to_select < n_features`; it
        // rejects `>= n_features` (selecting all features is disallowed).
        // Mirrors `_sequential.py:227-228`:
        //   `if self.n_features_to_select >= n_features:`
        //   `    raise ValueError("n_features_to_select must be < n_features.")`
        if self.n_features_to_select >= n_features {
            return Err(FerroError::InvalidParameter {
                name: "n_features_to_select".into(),
                reason: format!(
                    "n_features_to_select ({}) must be < number of features ({})",
                    self.n_features_to_select, n_features
                ),
            });
        }
        if n_samples == 0 {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "SequentialFeatureSelector::fit".into(),
            });
        }
        if y.len() != n_samples {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_samples],
                actual: vec![y.len()],
                context: "SequentialFeatureSelector::fit — y must match x rows".into(),
            });
        }

        let selected_indices = match self.direction {
            Direction::Forward => self.forward_search(x, y, n_features, &score_fn)?,
            Direction::Backward => self.backward_search(x, y, n_features, &score_fn)?,
        };

        Ok(FittedSequentialFeatureSelector {
            n_features_in: n_features,
            selected_indices,
            _marker: std::marker::PhantomData,
        })
    }

    /// Forward greedy search.
    #[allow(clippy::type_complexity)]
    fn forward_search<F: Float + Send + Sync + 'static>(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
        n_features: usize,
        score_fn: &dyn Fn(&Array2<F>, &Array1<F>) -> Result<F, FerroError>,
    ) -> Result<Vec<usize>, FerroError> {
        let mut selected: Vec<usize> = Vec::with_capacity(self.n_features_to_select);
        let mut remaining: Vec<usize> = (0..n_features).collect();

        for _ in 0..self.n_features_to_select {
            let mut best_score = F::neg_infinity();
            let mut best_feature = remaining[0];

            for &candidate in &remaining {
                let mut trial: Vec<usize> = selected.clone();
                trial.push(candidate);
                trial.sort_unstable();
                let x_sub = select_columns(x, &trial);
                let score = score_fn(&x_sub, y)?;
                if score > best_score {
                    best_score = score;
                    best_feature = candidate;
                }
            }

            selected.push(best_feature);
            remaining.retain(|&f| f != best_feature);
        }

        selected.sort_unstable();
        Ok(selected)
    }

    /// Backward greedy search.
    #[allow(clippy::type_complexity)]
    fn backward_search<F: Float + Send + Sync + 'static>(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
        n_features: usize,
        score_fn: &dyn Fn(&Array2<F>, &Array1<F>) -> Result<F, FerroError>,
    ) -> Result<Vec<usize>, FerroError> {
        let mut remaining: Vec<usize> = (0..n_features).collect();

        while remaining.len() > self.n_features_to_select {
            let mut best_score = F::neg_infinity();
            let mut worst_feature = remaining[0];

            for &candidate in &remaining {
                // Try removing this feature
                let trial: Vec<usize> = remaining
                    .iter()
                    .copied()
                    .filter(|&f| f != candidate)
                    .collect();
                let x_sub = select_columns(x, &trial);
                let score = score_fn(&x_sub, y)?;
                if score > best_score {
                    best_score = score;
                    worst_feature = candidate;
                }
            }

            remaining.retain(|&f| f != worst_feature);
        }

        remaining.sort_unstable();
        Ok(remaining)
    }
}

// ---------------------------------------------------------------------------
// FittedSequentialFeatureSelector
// ---------------------------------------------------------------------------

/// A fitted sequential feature selector holding the selected feature indices.
///
/// Created by calling [`SequentialFeatureSelector::fit`].
#[derive(Debug, Clone)]
pub struct FittedSequentialFeatureSelector<F> {
    /// Number of features seen during fitting.
    n_features_in: usize,
    /// Indices of the selected columns (sorted).
    selected_indices: Vec<usize>,
    _marker: std::marker::PhantomData<F>,
}

impl<F: Float + Send + Sync + 'static> FittedSequentialFeatureSelector<F> {
    /// Return the number of features seen during fitting.
    #[must_use]
    pub fn n_features_in(&self) -> usize {
        self.n_features_in
    }

    /// Return the indices of the selected features.
    #[must_use]
    pub fn selected_indices(&self) -> &[usize] {
        &self.selected_indices
    }

    /// Return the number of selected features.
    #[must_use]
    pub fn n_features_selected(&self) -> usize {
        self.selected_indices.len()
    }
}

impl<F: Float + Send + Sync + 'static> Transform<Array2<F>> for FittedSequentialFeatureSelector<F> {
    type Output = Array2<F>;
    type Error = FerroError;

    /// Return a matrix containing only the selected columns.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of columns differs
    /// from the number of features seen during fitting.
    fn transform(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        if x.ncols() != self.n_features_in {
            return Err(FerroError::ShapeMismatch {
                expected: vec![x.nrows(), self.n_features_in],
                actual: vec![x.nrows(), x.ncols()],
                context: "FittedSequentialFeatureSelector::transform".into(),
            });
        }
        Ok(select_columns(x, &self.selected_indices))
    }
}

/// Build a new `Array2<F>` containing only the columns listed in `indices`.
fn select_columns<F: Float>(x: &Array2<F>, indices: &[usize]) -> Array2<F> {
    let nrows = x.nrows();
    let ncols = indices.len();
    if ncols == 0 {
        return Array2::zeros((nrows, 0));
    }
    let mut out = Array2::zeros((nrows, ncols));
    for (new_j, &old_j) in indices.iter().enumerate() {
        for i in 0..nrows {
            out[[i, new_j]] = x[[i, old_j]];
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    /// Score function: sum of column means (higher is better).
    fn mean_sum_score(x: &Array2<f64>, _y: &Array1<f64>) -> Result<f64, FerroError> {
        let score: f64 = x
            .columns()
            .into_iter()
            .map(|c| c.sum() / c.len() as f64)
            .sum();
        Ok(score)
    }

    #[test]
    fn test_forward_selects_best() {
        let sfs = SequentialFeatureSelector::new(1, Direction::Forward);
        let x = array![[1.0, 10.0, 0.1], [2.0, 20.0, 0.2], [3.0, 30.0, 0.3]];
        let y = array![1.0, 2.0, 3.0];
        let fitted = sfs.fit(&x, &y, mean_sum_score).unwrap();
        assert_eq!(fitted.selected_indices(), &[1]); // col 1 has highest mean
    }

    #[test]
    fn test_forward_select_two() {
        let sfs = SequentialFeatureSelector::new(2, Direction::Forward);
        let x = array![[1.0, 10.0, 100.0], [2.0, 20.0, 200.0]];
        let y = array![1.0, 2.0];
        let fitted = sfs.fit(&x, &y, mean_sum_score).unwrap();
        assert_eq!(fitted.n_features_selected(), 2);
        // Top 2 by mean: col 2 (150.0) and col 1 (15.0)
        assert!(fitted.selected_indices().contains(&1));
        assert!(fitted.selected_indices().contains(&2));
    }

    #[test]
    fn test_backward_selects_best() {
        let sfs = SequentialFeatureSelector::new(1, Direction::Backward);
        let x = array![[1.0, 10.0, 0.1], [2.0, 20.0, 0.2], [3.0, 30.0, 0.3]];
        let y = array![1.0, 2.0, 3.0];
        let fitted = sfs.fit(&x, &y, mean_sum_score).unwrap();
        // Backward: remove 2 features. With sum-of-means score, removing
        // the smallest contributors yields col 1 remaining
        assert_eq!(fitted.selected_indices(), &[1]);
    }

    #[test]
    fn test_backward_select_two() {
        let sfs = SequentialFeatureSelector::new(2, Direction::Backward);
        let x = array![[1.0, 10.0, 100.0], [2.0, 20.0, 200.0]];
        let y = array![1.0, 2.0];
        let fitted = sfs.fit(&x, &y, mean_sum_score).unwrap();
        assert_eq!(fitted.n_features_selected(), 2);
        // Remove col 0 (lowest mean), keep 1 and 2
        assert_eq!(fitted.selected_indices(), &[1, 2]);
    }

    /// `n_features_to_select == n_features` is REJECTED.
    ///
    /// sklearn `_sequential.py:227-228` requires strictly
    /// `n_features_to_select < n_features` and raises
    /// `ValueError("n_features_to_select must be < n_features.")` for
    /// `>= n_features`, so selecting all features is disallowed.
    #[test]
    fn test_select_all_features_rejected() {
        let sfs = SequentialFeatureSelector::new(3, Direction::Forward);
        let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let y = array![1.0, 2.0];
        assert!(sfs.fit(&x, &y, mean_sum_score).is_err());
    }

    #[test]
    fn test_transform() {
        let sfs = SequentialFeatureSelector::new(1, Direction::Forward);
        let x = array![[1.0, 10.0], [2.0, 20.0]];
        let y = array![1.0, 2.0];
        let fitted = sfs.fit(&x, &y, mean_sum_score).unwrap();
        let out = fitted.transform(&x).unwrap();
        assert_eq!(out.ncols(), 1);
        assert_abs_diff_eq!(out[[0, 0]], 10.0, epsilon = 1e-15);
        assert_abs_diff_eq!(out[[1, 0]], 20.0, epsilon = 1e-15);
    }

    #[test]
    fn test_zero_features_error() {
        let sfs = SequentialFeatureSelector::new(0, Direction::Forward);
        let x = array![[1.0, 2.0]];
        let y = array![1.0];
        assert!(sfs.fit(&x, &y, mean_sum_score).is_err());
    }

    #[test]
    fn test_too_many_features_error() {
        let sfs = SequentialFeatureSelector::new(5, Direction::Forward);
        let x = array![[1.0, 2.0]];
        let y = array![1.0];
        assert!(sfs.fit(&x, &y, mean_sum_score).is_err());
    }

    #[test]
    fn test_zero_rows_error() {
        let sfs = SequentialFeatureSelector::new(1, Direction::Forward);
        let x: Array2<f64> = Array2::zeros((0, 3));
        let y: Array1<f64> = Array1::zeros(0);
        assert!(sfs.fit(&x, &y, mean_sum_score).is_err());
    }

    #[test]
    fn test_y_length_mismatch() {
        let sfs = SequentialFeatureSelector::new(1, Direction::Forward);
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![1.0]; // wrong length
        assert!(sfs.fit(&x, &y, mean_sum_score).is_err());
    }

    #[test]
    fn test_shape_mismatch_on_transform() {
        let sfs = SequentialFeatureSelector::new(1, Direction::Forward);
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![1.0, 2.0];
        let fitted = sfs.fit(&x, &y, mean_sum_score).unwrap();
        let x_bad = array![[1.0, 2.0, 3.0]];
        assert!(fitted.transform(&x_bad).is_err());
    }

    #[test]
    fn test_score_fn_error_propagated() {
        let sfs = SequentialFeatureSelector::new(1, Direction::Forward);
        let x = array![[1.0, 2.0]];
        let y = array![1.0];
        let bad_fn = |_x: &Array2<f64>, _y: &Array1<f64>| -> Result<f64, FerroError> {
            Err(FerroError::NumericalInstability {
                message: "test error".into(),
            })
        };
        assert!(sfs.fit(&x, &y, bad_fn).is_err());
    }

    #[test]
    fn test_indices_sorted() {
        let sfs = SequentialFeatureSelector::new(2, Direction::Forward);
        let x = array![[100.0, 1.0, 10.0], [200.0, 2.0, 20.0]];
        let y = array![1.0, 2.0];
        let fitted = sfs.fit(&x, &y, mean_sum_score).unwrap();
        let indices = fitted.selected_indices();
        assert!(indices.windows(2).all(|w| w[0] < w[1]));
    }

    #[test]
    fn test_accessors() {
        let sfs = SequentialFeatureSelector::new(2, Direction::Backward);
        assert_eq!(sfs.n_features_to_select(), 2);
        assert_eq!(sfs.direction(), Direction::Backward);
    }
}
