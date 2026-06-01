//! Logistic Regression with built-in cross-validated C selection.
//!
//! This module provides [`LogisticRegressionCV`], which automatically selects
//! the optimal regularization strength `C` by running k-fold cross-validation
//! over a grid of candidate values.
//!
//! # Examples
//!
//! ```
//! use ferrolearn_linear::logistic_regression_cv::LogisticRegressionCV;
//! use ferrolearn_core::{Fit, Predict};
//! use ndarray::{array, Array2};
//!
//! let x = Array2::from_shape_vec(
//!     (12, 2),
//!     vec![
//!         1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 2.0, 2.0, 1.5, 1.5, 1.0, 1.8,
//!         8.0, 8.0, 8.0, 9.0, 9.0, 8.0, 9.0, 9.0, 8.5, 8.5, 8.0, 8.8,
//!     ],
//! ).unwrap();
//! let y = array![0usize, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1];
//!
//! let model = LogisticRegressionCV::<f64>::new();
//! let fitted = model.fit(&x, &y).unwrap();
//! let preds = fitted.predict(&x).unwrap();
//! assert_eq!(preds.len(), 12);
//! ```

use ferrolearn_core::error::FerroError;
use ferrolearn_core::introspection::{HasClasses, HasCoefficients};
use ferrolearn_core::traits::{Fit, Predict};
use ndarray::{Array1, Array2, ScalarOperand};
use num_traits::Float;

use crate::logistic_regression::LogisticRegression;

/// Logistic Regression with cross-validated C selection.
///
/// Evaluates a grid of C values using k-fold cross-validation and refits
/// on the full dataset with the best C.
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
#[derive(Debug, Clone)]
pub struct LogisticRegressionCV<F> {
    /// Candidate C values to evaluate.
    pub cs: Vec<F>,
    /// Number of cross-validation folds.
    pub cv: usize,
    /// Maximum number of L-BFGS iterations per fit.
    pub max_iter: usize,
    /// Convergence tolerance per fit.
    pub tol: F,
}

impl<F: Float> LogisticRegressionCV<F> {
    /// Create a new `LogisticRegressionCV` with default settings.
    ///
    /// Defaults: `cs` = 10 log-spaced values from 1e-4 to 1e4,
    /// `cv = 5`, `max_iter = 1000`, `tol = 1e-4`.
    #[must_use]
    pub fn new() -> Self {
        // Build log-spaced C values from 1e-4 to 1e4.
        let mut cs = Vec::with_capacity(10);
        for i in 0..10 {
            let exp = F::from(-4.0 + i as f64 * 8.0 / 9.0).unwrap();
            let base = F::from(10.0).unwrap();
            cs.push(base.powf(exp));
        }

        Self {
            cs,
            cv: 5,
            max_iter: 1000,
            tol: F::from(1e-4).unwrap(),
        }
    }

    /// Set the candidate C values.
    #[must_use]
    pub fn with_cs(mut self, cs: Vec<F>) -> Self {
        self.cs = cs;
        self
    }

    /// Set the number of cross-validation folds.
    #[must_use]
    pub fn with_cv(mut self, cv: usize) -> Self {
        self.cv = cv;
        self
    }

    /// Set the maximum number of iterations.
    #[must_use]
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the convergence tolerance.
    #[must_use]
    pub fn with_tol(mut self, tol: F) -> Self {
        self.tol = tol;
        self
    }
}

impl<F: Float> Default for LogisticRegressionCV<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted Logistic Regression CV model.
///
/// Stores the model fitted with the best C value. Exposes `best_c()`.
#[derive(Debug, Clone)]
pub struct FittedLogisticRegressionCV<F> {
    /// The best C value found by cross-validation.
    best_c: F,
    /// The inner fitted logistic regression model.
    inner: crate::logistic_regression::FittedLogisticRegression<F>,
    /// CV scores for each C value (accuracy).
    cv_scores: Vec<F>,
    /// The C values evaluated.
    cs_evaluated: Vec<F>,
}

impl<F: Float> FittedLogisticRegressionCV<F> {
    /// Returns the best C value selected by cross-validation.
    #[must_use]
    pub fn best_c(&self) -> F {
        self.best_c
    }

    /// Returns the CV accuracy scores, one per C value.
    #[must_use]
    pub fn cv_scores(&self) -> &[F] {
        &self.cv_scores
    }

    /// Returns the C values that were evaluated.
    #[must_use]
    pub fn cs_evaluated(&self) -> &[F] {
        &self.cs_evaluated
    }
}

impl<F: Float + ndarray::ScalarOperand + Send + Sync + 'static> FittedLogisticRegressionCV<F> {
    /// Predict per-class probabilities. Mirrors sklearn
    /// `LogisticRegressionCV.predict_proba`. Delegates to the inner
    /// fitted LogisticRegression at the best C value.
    ///
    /// # Errors
    ///
    /// Forwards any error from the inner `predict_proba`.
    pub fn predict_proba(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        self.inner.predict_proba(x)
    }

    /// Element-wise log of [`predict_proba`](Self::predict_proba).
    ///
    /// # Errors
    ///
    /// Forwards any error from [`predict_proba`](Self::predict_proba).
    pub fn predict_log_proba(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let proba = self.predict_proba(x)?;
        Ok(crate::log_proba(&proba))
    }

    /// Raw decision-function scores delegated to the inner fitted
    /// LogisticRegression at the best C value. Mirrors sklearn
    /// `LogisticRegressionCV.decision_function`.
    ///
    /// # Errors
    ///
    /// Forwards any error from the inner `decision_function`.
    pub fn decision_function(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        self.inner.decision_function(x)
    }
}

/// StratifiedKFold-style split, replicating scikit-learn's non-shuffled
/// `StratifiedKFold._make_test_folds` (`sklearn/model_selection/_split.py:746-806`),
/// the default CV splitter for `LogisticRegressionCV` (#346, #456).
///
/// Each class's samples are assigned to folds in CONTIGUOUS blocks (not
/// round-robin `i % k`). The per-fold class allocation is the round-robin
/// distribution over the *sorted* class labels:
///   `allocation[j][c] = bincount(y_order[j::k])[c]` where `y_order = sort(y_encoded)`
/// (sklearn `_split.py:786-792`). Then, for each class `c`, its samples (in
/// original input order) are assigned to folds via
///   `folds_for_class = arange(k).repeat(allocation[:, c])`
/// so fold 0 takes the first `allocation[0][c]` of class-`c`'s samples, fold 1
/// the next `allocation[1][c]`, etc. (sklearn `_split.py:794-805`).
///
/// Returns `(train_indices, test_indices)` for fold number `fold`.
fn stratified_kfold_split(y: &Array1<usize>, k: usize, fold: usize) -> (Vec<usize>, Vec<usize>) {
    let n = y.len();

    // Distinct classes, sorted ascending — this is sklearn's class encoding
    // (`np.unique`); the encoded label of sample `i` is its position here.
    let mut classes: Vec<usize> = y.iter().copied().collect();
    classes.sort_unstable();
    classes.dedup();
    let n_classes = classes.len();
    let encode = |label: usize| classes.binary_search(&label).unwrap_or(0);

    // y_order = the class-encoded labels sorted ascending (all class-0 samples,
    // then class-1, ...). sklearn: `y_order = np.sort(y_encoded)`.
    let mut y_order: Vec<usize> = y.iter().map(|&label| encode(label)).collect();
    y_order.sort_unstable();

    // allocation[j][c] = bincount(y_order[j::k], minlength=n_classes)[c]
    // = number of class-c samples assigned to fold j (balances fold sizes).
    let mut allocation = vec![vec![0usize; n_classes]; k];
    for (j, alloc_row) in allocation.iter_mut().enumerate() {
        let mut idx = j;
        while idx < y_order.len() {
            alloc_row[y_order[idx]] += 1;
            idx += k;
        }
    }

    // For each class c, assign its original-order samples to folds in
    // contiguous blocks per `allocation[:, c]`, then collect those landing in
    // the requested `fold` as the test set.
    let mut test_indices: Vec<usize> = Vec::new();
    for (c, &cls) in classes.iter().enumerate() {
        let cls_indices: Vec<usize> = (0..n).filter(|&i| y[i] == cls).collect();
        let mut pos = 0usize;
        for (j, alloc_row) in allocation.iter().enumerate() {
            let count = alloc_row[c];
            if j == fold {
                for &idx in &cls_indices[pos..pos + count] {
                    test_indices.push(idx);
                }
            }
            pos += count;
        }
    }

    test_indices.sort_unstable();
    let test_set: std::collections::HashSet<usize> = test_indices.iter().copied().collect();
    let train_indices: Vec<usize> = (0..n).filter(|i| !test_set.contains(i)).collect();
    (train_indices, test_indices)
}

/// Extract rows from a 2D array by index.
fn select_rows<F: Float>(x: &Array2<F>, indices: &[usize]) -> Array2<F> {
    let n_features = x.ncols();
    let n_rows = indices.len();
    let mut result = Array2::<F>::zeros((n_rows, n_features));
    for (r, &i) in indices.iter().enumerate() {
        for j in 0..n_features {
            result[[r, j]] = x[[i, j]];
        }
    }
    result
}

/// Extract elements from a 1D array by index.
fn select_elements(y: &Array1<usize>, indices: &[usize]) -> Array1<usize> {
    Array1::from_vec(indices.iter().map(|&i| y[i]).collect())
}

impl<F: Float + Send + Sync + ScalarOperand + 'static> Fit<Array2<F>, Array1<usize>>
    for LogisticRegressionCV<F>
{
    type Fitted = FittedLogisticRegressionCV<F>;
    type Error = FerroError;

    /// Fit logistic regression with cross-validated C selection.
    ///
    /// For each candidate C, runs k-fold CV and computes accuracy.
    /// Selects the C with the highest mean accuracy and refits on the
    /// full dataset.
    ///
    /// # Errors
    ///
    /// - [`FerroError::ShapeMismatch`] — sample count mismatch.
    /// - [`FerroError::InvalidParameter`] — empty cs or cv < 2.
    /// - [`FerroError::InsufficientSamples`] — fewer than 2 classes or
    ///   too few samples for the number of folds.
    fn fit(
        &self,
        x: &Array2<F>,
        y: &Array1<usize>,
    ) -> Result<FittedLogisticRegressionCV<F>, FerroError> {
        let (n_samples, _n_features) = x.dim();

        if n_samples != y.len() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_samples],
                actual: vec![y.len()],
                context: "y length must match number of samples in X".into(),
            });
        }

        if self.cs.is_empty() {
            return Err(FerroError::InvalidParameter {
                name: "cs".into(),
                reason: "must have at least one candidate C value".into(),
            });
        }

        if self.cv < 2 {
            return Err(FerroError::InvalidParameter {
                name: "cv".into(),
                reason: "must be at least 2".into(),
            });
        }

        if n_samples < self.cv {
            return Err(FerroError::InsufficientSamples {
                required: self.cv,
                actual: n_samples,
                context: "need at least as many samples as CV folds".into(),
            });
        }

        let mut classes: Vec<usize> = y.to_vec();
        classes.sort_unstable();
        classes.dedup();

        if classes.len() < 2 {
            return Err(FerroError::InsufficientSamples {
                required: 2,
                actual: classes.len(),
                context: "LogisticRegressionCV requires at least 2 distinct classes".into(),
            });
        }

        // Evaluate each C via k-fold CV.
        let mut cv_scores = Vec::with_capacity(self.cs.len());
        let mut best_c = self.cs[0];
        let mut best_score = F::neg_infinity();

        for &c in &self.cs {
            let mut total_correct = 0usize;
            let mut total_count = 0usize;
            let mut fold_failed = false;

            for fold in 0..self.cv {
                let _ = n_samples; // computed via y.len() in the splitter
                let (train_idx, test_idx) = stratified_kfold_split(y, self.cv, fold);

                let x_train = select_rows(x, &train_idx);
                let y_train = select_elements(y, &train_idx);
                let x_test = select_rows(x, &test_idx);
                let y_test = select_elements(y, &test_idx);

                // Check that training set has at least 2 classes.
                let mut train_classes: Vec<usize> = y_train.to_vec();
                train_classes.sort_unstable();
                train_classes.dedup();
                if train_classes.len() < 2 {
                    fold_failed = true;
                    break;
                }

                let lr = LogisticRegression::<F>::new()
                    .with_c(c)
                    .with_max_iter(self.max_iter)
                    .with_tol(self.tol);

                match lr.fit(&x_train, &y_train) {
                    Ok(fitted) => match fitted.predict(&x_test) {
                        Ok(preds) => {
                            let correct = preds
                                .iter()
                                .zip(y_test.iter())
                                .filter(|(p, a)| p == a)
                                .count();
                            total_correct += correct;
                            total_count += y_test.len();
                        }
                        Err(_) => {
                            fold_failed = true;
                            break;
                        }
                    },
                    Err(_) => {
                        fold_failed = true;
                        break;
                    }
                }
            }

            let score = if fold_failed || total_count == 0 {
                F::zero()
            } else {
                F::from(total_correct).unwrap() / F::from(total_count).unwrap()
            };

            cv_scores.push(score);

            if score > best_score {
                best_score = score;
                best_c = c;
            }
        }

        // Refit on the full dataset with the best C.
        let lr = LogisticRegression::<F>::new()
            .with_c(best_c)
            .with_max_iter(self.max_iter)
            .with_tol(self.tol);

        let inner = lr.fit(x, y)?;

        Ok(FittedLogisticRegressionCV {
            best_c,
            inner,
            cv_scores,
            cs_evaluated: self.cs.clone(),
        })
    }
}

impl<F: Float + Send + Sync + ScalarOperand + 'static> Predict<Array2<F>>
    for FittedLogisticRegressionCV<F>
{
    type Output = Array1<usize>;
    type Error = FerroError;

    /// Predict class labels using the refitted model with the best C.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of features
    /// does not match the fitted model.
    fn predict(&self, x: &Array2<F>) -> Result<Array1<usize>, FerroError> {
        self.inner.predict(x)
    }
}

impl<F: Float + Send + Sync + ScalarOperand + 'static> HasCoefficients<F>
    for FittedLogisticRegressionCV<F>
{
    fn coefficients(&self) -> &Array1<F> {
        self.inner.coefficients()
    }

    fn intercept(&self) -> F {
        self.inner.intercept()
    }
}

impl<F: Float + Send + Sync + ScalarOperand + 'static> HasClasses
    for FittedLogisticRegressionCV<F>
{
    fn classes(&self) -> &[usize] {
        self.inner.classes()
    }

    fn n_classes(&self) -> usize {
        self.inner.n_classes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// #456: `stratified_kfold_split` must replicate sklearn's non-shuffled
    /// `StratifiedKFold._make_test_folds` (contiguous blocks), NOT the old
    /// `i % k` round-robin. Expected test-fold memberships come from the live
    /// oracle (R-CHAR-3):
    ///   `StratifiedKFold(3).split(zeros, y)` for
    ///   `y=[0,0,0,0,1,1,1,1,1]` -> [[0,1,4],[2,5,6],[3,7,8]]
    /// and for the 18-sample test dataset
    ///   `y=[1,1,0,1,1,0,0,0,0,1,1,0,0,0,1,1,0,1]` ->
    ///   [[0,1,2,3,5,6],[4,7,8,9,10,11],[12,13,14,15,16,17]].
    #[test]
    fn stratified_kfold_split_matches_sklearn_partition() {
        // y=[0,0,0,0,1,1,1,1,1], k=3 (the goal.md verification case).
        let y = Array1::from_vec(vec![0usize, 0, 0, 0, 1, 1, 1, 1, 1]);
        let expected: [Vec<usize>; 3] = [vec![0, 1, 4], vec![2, 5, 6], vec![3, 7, 8]];
        for (fold, exp) in expected.iter().enumerate() {
            let (_train, test) = stratified_kfold_split(&y, 3, fold);
            assert_eq!(&test, exp, "fold {fold} test indices mismatch vs sklearn");
        }

        // The 18-sample, 2-class dataset used by the divergence integration test.
        let y2 = Array1::from_vec(vec![
            1usize, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1,
        ]);
        let expected2: [Vec<usize>; 3] = [
            vec![0, 1, 2, 3, 5, 6],
            vec![4, 7, 8, 9, 10, 11],
            vec![12, 13, 14, 15, 16, 17],
        ];
        for (fold, exp) in expected2.iter().enumerate() {
            let (train, test) = stratified_kfold_split(&y2, 3, fold);
            assert_eq!(&test, exp, "18-sample fold {fold} test mismatch vs sklearn");
            // train is the disjoint complement.
            assert_eq!(train.len() + test.len(), y2.len());
        }
    }

    #[test]
    fn test_default_constructor() {
        let m = LogisticRegressionCV::<f64>::new();
        assert_eq!(m.cv, 5);
        assert_eq!(m.max_iter, 1000);
        assert_eq!(m.cs.len(), 10);
    }

    #[test]
    fn test_builder() {
        let m = LogisticRegressionCV::<f64>::new()
            .with_cs(vec![0.1, 1.0, 10.0])
            .with_cv(3)
            .with_max_iter(500)
            .with_tol(1e-6);
        assert_eq!(m.cs.len(), 3);
        assert_eq!(m.cv, 3);
        assert_eq!(m.max_iter, 500);
    }

    #[test]
    fn test_binary_cv() {
        let x = Array2::from_shape_vec(
            (12, 2),
            vec![
                1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 2.0, 2.0, 1.5, 1.5, 1.0, 1.8, 8.0, 8.0, 8.0, 9.0,
                9.0, 8.0, 9.0, 9.0, 8.5, 8.5, 8.0, 8.8,
            ],
        )
        .unwrap();
        let y = array![0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1];

        let model = LogisticRegressionCV::<f64>::new()
            .with_cs(vec![0.1, 1.0, 10.0])
            .with_cv(3);
        let fitted = model.fit(&x, &y).unwrap();

        assert!(fitted.best_c() > 0.0);
        assert_eq!(fitted.cv_scores().len(), 3);
        assert_eq!(fitted.cs_evaluated().len(), 3);

        let preds = fitted.predict(&x).unwrap();
        let correct: usize = preds.iter().zip(y.iter()).filter(|(p, a)| p == a).count();
        assert!(correct >= 8, "expected at least 8 correct, got {correct}");
    }

    #[test]
    fn test_multiclass_cv() {
        let x = Array2::from_shape_vec(
            (12, 2),
            vec![
                0.0, 0.0, 0.5, 0.0, 0.0, 0.5, 0.5, 0.5, 10.0, 0.0, 10.5, 0.0, 10.0, 0.5, 10.5, 0.5,
                0.0, 10.0, 0.5, 10.0, 0.0, 10.5, 0.5, 10.5,
            ],
        )
        .unwrap();
        let y = array![0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2];

        let model = LogisticRegressionCV::<f64>::new()
            .with_cs(vec![1.0, 10.0])
            .with_cv(2);
        let fitted = model.fit(&x, &y).unwrap();

        assert_eq!(fitted.n_classes(), 3);
        assert_eq!(fitted.classes(), &[0, 1, 2]);

        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 12);
    }

    #[test]
    fn test_best_c_in_cs() {
        let x = Array2::from_shape_vec(
            (10, 2),
            vec![
                1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 2.0, 2.0, 1.5, 1.5, 8.0, 8.0, 8.0, 9.0, 9.0, 8.0,
                9.0, 9.0, 8.5, 8.5,
            ],
        )
        .unwrap();
        let y = array![0, 0, 0, 0, 0, 1, 1, 1, 1, 1];

        let cs = vec![0.01, 0.1, 1.0, 10.0, 100.0];
        let model = LogisticRegressionCV::<f64>::new()
            .with_cs(cs.clone())
            .with_cv(2);
        let fitted = model.fit(&x, &y).unwrap();

        assert!(cs.contains(&fitted.best_c()));
    }

    #[test]
    fn test_shape_mismatch() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![0, 1]; // Wrong length

        let model = LogisticRegressionCV::<f64>::new();
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_empty_cs() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![0, 0, 1, 1];

        let model = LogisticRegressionCV::<f64>::new().with_cs(vec![]);
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_cv_too_small() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![0, 0, 1, 1];

        let model = LogisticRegressionCV::<f64>::new().with_cv(1);
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_single_class_error() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![0, 0, 0, 0];

        let model = LogisticRegressionCV::<f64>::new()
            .with_cs(vec![1.0])
            .with_cv(2);
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_has_coefficients() {
        let x = Array2::from_shape_vec(
            (10, 2),
            vec![
                1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 2.0, 2.0, 1.5, 1.5, 8.0, 8.0, 8.0, 9.0, 9.0, 8.0,
                9.0, 9.0, 8.5, 8.5,
            ],
        )
        .unwrap();
        let y = array![0, 0, 0, 0, 0, 1, 1, 1, 1, 1];

        let fitted = LogisticRegressionCV::<f64>::new()
            .with_cs(vec![1.0])
            .with_cv(2)
            .fit(&x, &y)
            .unwrap();
        assert_eq!(fitted.coefficients().len(), 2);
    }

    #[test]
    fn test_stratified_kfold_split() {
        // 10 samples, 5 folds, balanced binary labels.
        let y: Array1<usize> = ndarray::array![0, 0, 0, 0, 0, 1, 1, 1, 1, 1];
        let (train, test) = stratified_kfold_split(&y, 5, 0);
        // Each fold gets one sample per class.
        assert_eq!(test.len(), 2);
        assert_eq!(train.len(), 8);
        // The test set must contain one of each class.
        let n0 = test.iter().filter(|&&i| y[i] == 0).count();
        let n1 = test.iter().filter(|&&i| y[i] == 1).count();
        assert_eq!(n0, 1);
        assert_eq!(n1, 1);
    }

    #[test]
    fn test_stratified_kfold_uneven() {
        // 7 samples, 3 folds, 4 of class 0 + 3 of class 1.
        let y: Array1<usize> = ndarray::array![0, 0, 0, 0, 1, 1, 1];
        let (train, test) = stratified_kfold_split(&y, 3, 0);
        // train + test must partition the 7 samples.
        assert_eq!(train.len() + test.len(), 7);
        // Class-0 distributed: 4 / 3 ≈ 1-2 per fold; class-1: 3 / 3 = 1 per fold.
        let n1 = test.iter().filter(|&&i| y[i] == 1).count();
        assert!(n1 >= 1, "class 1 must be present in stratified test fold");
    }
}
