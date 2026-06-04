//! Divergence tests for `ferrolearn-model-sel::threshold` vs scikit-learn 1.5.2
//! (`sklearn/model_selection/_classification_threshold.py`).
//!
//! Expected values come from a LIVE sklearn 1.5.2 oracle (run from /tmp) or
//! from sklearn's documented thresholding FORMULA
//! (`_threshold_scores_to_class_labels`, `:57-66`) — never copied from the
//! ferrolearn side (goal.md R-CHAR-3).

use ferrolearn_model_sel::threshold::{
    FitScoreFn, FixedThresholdClassifier, ScoreFn, ThresholdScoring, TunedThresholdClassifierCV,
};
use ndarray::{Array1, Array2, array};

/// `FitScoreFn` whose `ScoreFn` returns feature column 0 verbatim as the
/// positive-class score. Mirrors a sklearn estimator whose
/// `decision_function` / `predict_proba(...)[:, 1]` is column 0, used by the
/// oracle setups below.
fn col0_fit_fn() -> FitScoreFn {
    Box::new(|_x: &Array2<f64>, _y: &Array1<usize>| {
        Ok(Box::new(|x: &Array2<f64>| {
            let n = x.nrows();
            let mut out = Array1::<f64>::zeros(n);
            for i in 0..n {
                out[i] = x[[i, 0]];
            }
            Ok(out)
        }) as ScoreFn)
    })
}

/// balanced_accuracy: the mean over the classes present in `y_true` of each
/// class's recall. Mirrors `sklearn.metrics.balanced_accuracy_score`, which is
/// `TunedThresholdClassifierCV`'s default `scoring`
/// (`sklearn/model_selection/_classification_threshold.py:805`).
fn balanced_accuracy() -> ThresholdScoring {
    Box::new(|y_true: &Array1<usize>, y_pred: &Array1<usize>| {
        let mut classes: Vec<usize> = y_true.iter().copied().collect();
        classes.sort_unstable();
        classes.dedup();
        let mut recalls = Vec::with_capacity(classes.len());
        for &c in &classes {
            let mut total = 0usize;
            let mut hit = 0usize;
            for (t, p) in y_true.iter().zip(y_pred.iter()) {
                if *t == c {
                    total += 1;
                    if *p == c {
                        hit += 1;
                    }
                }
            }
            recalls.push(hit as f64 / total as f64);
        }
        recalls.iter().sum::<f64>() / recalls.len() as f64
    })
}

/// GREEN GUARD — REQ-1: `FittedFixedThresholdClassifier::predict` parity with
/// sklearn's `_threshold_scores_to_class_labels` `(y_score >= threshold)`
/// kernel, INCLUDING the `s == threshold` edge.
///
/// Mirrors `sklearn/model_selection/_classification_threshold.py:66`
/// `classes[map[(y_score >= threshold).astype(int)]]` for the
/// `pos_label=None` -> `[0, 1]` binary case (`:60`).
///
/// Live oracle (run from /tmp, sklearn 1.5.2):
/// ```text
/// from sklearn.model_selection import FixedThresholdClassifier
/// class ColZero(ClassifierMixin, BaseEstimator):
///     def fit(self, X, y): self.classes_=np.array([0,1]); ...; return self
///     def decision_function(self, X): return np.asarray(X)[:,0]
/// X=[[0.1,9],[0.6,9],[0.4,9],[0.9,9],[0.5,9]]
/// FixedThresholdClassifier(ColZero().fit(X,y), threshold=0.5,
///     response_method='decision_function').fit(X,y).predict(X)
/// # -> [0, 1, 0, 1, 1]   (the score-0.5 == threshold-0.5 row -> class 1)
/// ```
/// This test PASSES against the current implementation (REQ-1 SHIPPED guard).
#[test]
fn fixed_threshold_predict_ge_edge_parity_green() {
    // Scores are column 0: [0.1, 0.6, 0.4, 0.9, 0.5].
    let x = array![[0.1, 9.0], [0.6, 9.0], [0.4, 9.0], [0.9, 9.0], [0.5, 9.0]];
    let y = array![0usize, 1, 0, 1, 1];
    let clf = FixedThresholdClassifier::new(col0_fit_fn(), 0.5);
    let fitted = clf.fit(&x, &y).unwrap();
    let preds = fitted.predict(&x).unwrap();

    // EXPECTED from the live sklearn 1.5.2 oracle above (NOT from ferrolearn).
    // The 0.5 == 0.5 row maps to class 1 via the `>=` comparison.
    let expected = array![0usize, 1, 0, 1, 1];
    assert_eq!(
        preds, expected,
        "FixedThresholdClassifier::predict must match sklearn \
         _threshold_scores_to_class_labels `y_score >= threshold` incl. the \
         s == threshold edge"
    );
}

/// DIVERGENCE (FAILING) — REQ-5 / REQ-6: `TunedThresholdClassifierCV::fit`
/// pools all CV folds into ONE out-of-fold score vector and scores each
/// threshold ONCE globally, whereas sklearn computes a per-fold score CURVE
/// and combines folds via `_mean_interpolated_score`
/// (`sklearn/model_selection/_classification_threshold.py:591-616`,`:928-930`)
/// before `objective_scores.argmax()` (`:931`). With a fold-sensitive scorer
/// (balanced_accuracy, sklearn's default `:805`) the pooled-OOF argmax differs
/// from sklearn's per-fold-mean argmax.
///
/// Fixture (n=8, cv=2): the unshuffled contiguous folds [0,1,2,3]/[4,5,6,7]
/// produced by ferrolearn's `KFold::new(2)` EQUAL the splits sklearn's
/// `check_cv(2, classifier=True)` (StratifiedKFold) produces for this `y`, so
/// the ONLY difference is the fold-combination arithmetic.
///
/// Live oracle (run from /tmp, sklearn 1.5.2):
/// ```text
/// s=[0.528,0.227,0.778,0.17,0.577,0.536,0.672,0.76]; y=[1,0,0,1,0,0,0,1]
/// X=column_stack([s, zeros]); thresholds=[0.2,0.4,0.6,0.8]
/// TunedThresholdClassifierCV(ColZero(), thresholds=[0.2,0.4,0.6,0.8],
///     scoring='balanced_accuracy', cv=2).fit(X,y).best_threshold_
/// # -> 0.6   (per-fold-mean scores: [0.375, 0.5, 0.5417, 0.5])
/// ```
/// ferrolearn pools the OOF vector and selects 0.8 instead
/// (pooled scores: [0.333, 0.433, 0.467, 0.5]).
///
/// Tracking: #1736 (per-REQ blocker filed below).
#[test]
fn tuned_threshold_fold_combination_divergence() {
    let x = array![
        [0.528, 0.0],
        [0.227, 0.0],
        [0.778, 0.0],
        [0.17, 0.0],
        [0.577, 0.0],
        [0.536, 0.0],
        [0.672, 0.0],
        [0.76, 0.0],
    ];
    let y = array![1usize, 0, 0, 1, 0, 0, 0, 1];
    let clf = TunedThresholdClassifierCV::new(
        col0_fit_fn(),
        2,
        vec![0.2, 0.4, 0.6, 0.8],
        balanced_accuracy(),
    );
    let fitted = clf.fit(&x, &y).unwrap();

    // EXPECTED from the live sklearn 1.5.2 oracle above (per-fold mean,
    // `best_threshold_ == 0.6`). ferrolearn's pooled-OOF argmax returns 0.8.
    let sklearn_best_threshold = 0.6_f64;
    assert!(
        (fitted.best_threshold() - sklearn_best_threshold).abs() < 1e-9,
        "TunedThresholdClassifierCV best_threshold {} must match sklearn \
         per-fold-mean-interpolated best_threshold_ {} (sklearn :928-931); \
         ferrolearn pools OOF folds and diverges",
        fitted.best_threshold(),
        sklearn_best_threshold
    );
}
