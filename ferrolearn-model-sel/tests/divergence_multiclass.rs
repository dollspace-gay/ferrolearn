//! Divergence audit: `OneVsRestClassifier` / `OneVsOneClassifier` vs
//! scikit-learn 1.5.2 (`sklearn/multiclass.py`,
//! `sklearn/utils/multiclass.py:520-562` `_ovr_decision_function`).
//!
//! All expected values come from a LIVE sklearn 1.5.2 oracle call run from
//! `/tmp` (`from sklearn.multiclass import ...` /
//! `from sklearn.utils.multiclass import _ovr_decision_function`), NEVER copied
//! from the ferrolearn side (R-CHAR-3). The oracle command that produced each
//! constant is quoted inline above the assertion.
//!
//! GREEN GUARDS (pass now, pin SHIPPED behavior against the oracle):
//! - REQ-OVR-PREDICT: OvR `predict` is LAST-on-tie, matching sklearn's
//!   `argmaxima[maxima == pred] = i` overwrite pattern
//!   (`sklearn/multiclass.py:496-500`), NOT `np.argmax` first-on-tie.
//! - REQ-OVR-MECH: K sorted classes -> K estimators, `classes()` sorted,
//!   `decision_function` shape `(n, K)` (`sklearn/multiclass.py:362-382`,
//!   `:554-582`).
//! - OvR + OvO end-to-end agreement on a clean, well-separated (NON-tie)
//!   3-class problem (confirms the OvO i/j vote convention is self-consistent).
//!
//! FAILING PIN (`#[ignore]`, #1819; doc REQ-OVO-DECISION planned #1812): OvO `predict` on a constructed VOTE TIE.
//! ferrolearn accumulates PURE integer votes and `max_by_key` (LAST-on-tie)
//! with NO confidence term; sklearn routes `predict` through
//! `decision_function` = `_ovr_decision_function(predictions, confidences, K)`
//! = `votes + sum_of_confidences/(3*(|sum|+1))`
//! (`sklearn/multiclass.py:935-939`, `:983`;
//! `sklearn/utils/multiclass.py:540-562`), breaking vote ties by summed
//! confidence. The fixer must port `_ovr_decision_function` (the OvO
//! `decision_function`, #1813) and route `predict` through it.

use ferrolearn_core::pipeline::{FittedPipelineEstimator, Pipeline, PipelineEstimator};
use ferrolearn_core::{FerroError, Predict};
use ferrolearn_model_sel::{OneVsOneClassifier, OneVsRestClassifier};
use ndarray::{Array1, Array2};

/// Local mirror of the private `PipelineFactory` alias used by the
/// multiclass meta-estimators (transparent boxed factory closure).
type PipelineFactory = Box<dyn Fn() -> Pipeline<f64> + Send + Sync>;

// ===========================================================================
// Fixture estimators
// ===========================================================================

/// A binary estimator that ignores its training data and always predicts a
/// fixed constant score for every sample. Used to build deterministic decision
/// rows (e.g. exact ties) independent of the data.
struct ConstScore(f64);

struct FittedConstScore(f64);

impl PipelineEstimator<f64> for ConstScore {
    fn fit_pipeline(
        &self,
        _x: &Array2<f64>,
        _y: &Array1<f64>,
    ) -> Result<Box<dyn FittedPipelineEstimator<f64>>, FerroError> {
        Ok(Box::new(FittedConstScore(self.0)))
    }
}

impl FittedPipelineEstimator<f64> for FittedConstScore {
    fn predict_pipeline(&self, x: &Array2<f64>) -> Result<Array1<f64>, FerroError> {
        Ok(Array1::from_elem(x.nrows(), self.0))
    }
}

/// A binary estimator whose constant predict score is the MEAN of feature
/// column 1 over its (filtered) training rows. Because each OvO pair is trained
/// on a different filtered subset, this lets a single factory closure produce a
/// DISTINCT per-pair constant score, driving an exact OvO vote pattern.
struct Col1MeanScore;

struct FittedCol1Mean(f64);

impl PipelineEstimator<f64> for Col1MeanScore {
    fn fit_pipeline(
        &self,
        x: &Array2<f64>,
        _y: &Array1<f64>,
    ) -> Result<Box<dyn FittedPipelineEstimator<f64>>, FerroError> {
        let n = x.nrows();
        let mean = if n == 0 {
            0.0
        } else {
            x.column(1).iter().copied().sum::<f64>() / n as f64
        };
        Ok(Box::new(FittedCol1Mean(mean)))
    }
}

impl FittedPipelineEstimator<f64> for FittedCol1Mean {
    fn predict_pipeline(&self, x: &Array2<f64>) -> Result<Array1<f64>, FerroError> {
        Ok(Array1::from_elem(x.nrows(), self.0))
    }
}

/// A nearest-prototype binary classifier: learns the mean row-value of the
/// positive (`1.0`) vs negative (`0.0`) class and at predict time returns a
/// score in `[0, 1]` that is higher the closer a sample is to the positive
/// mean. Deterministic; used for clean (non-tie) end-to-end agreement guards.
struct ThresholdEstimator;

struct FittedThreshold {
    pos_mean: f64,
    neg_mean: f64,
}

impl PipelineEstimator<f64> for ThresholdEstimator {
    fn fit_pipeline(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
    ) -> Result<Box<dyn FittedPipelineEstimator<f64>>, FerroError> {
        let (mut pos_sum, mut pos_count) = (0.0, 0usize);
        let (mut neg_sum, mut neg_count) = (0.0, 0usize);
        for (i, &label) in y.iter().enumerate() {
            let row_mean = x.row(i).mean().unwrap_or(0.0);
            if label > 0.5 {
                pos_sum += row_mean;
                pos_count += 1;
            } else {
                neg_sum += row_mean;
                neg_count += 1;
            }
        }
        let pos_mean = if pos_count > 0 {
            pos_sum / pos_count as f64
        } else {
            0.0
        };
        let neg_mean = if neg_count > 0 {
            neg_sum / neg_count as f64
        } else {
            0.0
        };
        Ok(Box::new(FittedThreshold { pos_mean, neg_mean }))
    }
}

impl FittedPipelineEstimator<f64> for FittedThreshold {
    fn predict_pipeline(&self, x: &Array2<f64>) -> Result<Array1<f64>, FerroError> {
        let preds: Vec<f64> = x
            .rows()
            .into_iter()
            .map(|row| {
                let val = row.mean().unwrap_or(0.0);
                let d_pos = (val - self.pos_mean).abs();
                let d_neg = (val - self.neg_mean).abs();
                let total = d_pos + d_neg;
                if total < 1e-15 { 0.5 } else { d_neg / total }
            })
            .collect();
        Ok(Array1::from_vec(preds))
    }
}

fn const_factory(val: f64) -> PipelineFactory {
    Box::new(move || Pipeline::new().estimator_step("clf", Box::new(ConstScore(val))))
}

fn col1_mean_factory() -> PipelineFactory {
    Box::new(|| Pipeline::new().estimator_step("clf", Box::new(Col1MeanScore)))
}

fn threshold_factory() -> PipelineFactory {
    Box::new(|| Pipeline::new().estimator_step("clf", Box::new(ThresholdEstimator)))
}

// ===========================================================================
// REQ-OVR-PREDICT (SHIPPED) — OvR predict is LAST-on-tie. GREEN GUARD.
// ===========================================================================

/// Divergence guard: ferrolearn OvR `predict`
/// (`ferrolearn-model-sel/src/multiclass.rs:236-253`, per-row `max_by`) must
/// pick the HIGHEST (last) class index on a decision-score tie, matching
/// sklearn's overwrite pattern `argmaxima[maxima == pred] = i`
/// (`sklearn/multiclass.py:499`) — which is LAST-on-tie, NOT `np.argmax`
/// first-on-tie.
///
/// Live oracle (sklearn 1.5.2, run from /tmp):
/// ```text
/// import numpy as np
/// from sklearn.multiclass import OneVsRestClassifier
/// from sklearn.base import BaseEstimator, ClassifierMixin
/// class ConstScore(BaseEstimator, ClassifierMixin):
///     def fit(self,X,y): self.classes_=np.unique(y); return self
///     def decision_function(self,X): return np.full(X.shape[0], 0.5)
///     def predict(self,X): return np.zeros(X.shape[0])
/// X=np.array([[0.],[1.],[2.],[3.],[4.],[5.]]); y=np.array([0,0,1,1,2,2])
/// clf=OneVsRestClassifier(ConstScore()).fit(X,y)
/// clf.predict(X[:1]).tolist()   # -> [2]   (LAST class index on a 3-way tie)
/// ```
#[test]
fn ovr_predict_last_on_tie_matches_sklearn() {
    // Constant-score base => every per-class binary classifier scores 0.5 =>
    // a 3-way decision-score tie for every sample.
    let x = Array2::from_shape_vec((6, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let y = Array1::from_vec(vec![0, 0, 1, 1, 2, 2]);

    let fitted = OneVsRestClassifier::new(const_factory(0.5))
        .fit(&x, &y)
        .unwrap();
    let preds = fitted.predict(&x).unwrap();

    // sklearn oracle: predict on the tied row == class 2 (LAST). Assert every
    // sample resolves to the HIGHEST class index, NOT class 0 (first-on-tie).
    const SK_TIE_CLASS: usize = 2;
    for (s, &p) in preds.iter().enumerate() {
        assert_eq!(
            p, SK_TIE_CLASS,
            "sample {s}: OvR tie must resolve to LAST class index (sklearn={SK_TIE_CLASS})"
        );
    }
}

// ===========================================================================
// REQ-OVR-MECH (SHIPPED) — fit/decision_function/classes_ shape. GREEN GUARD.
// ===========================================================================

/// Divergence guard: ferrolearn OvR `fit` builds one estimator per sorted
/// class, `classes()` is sorted ascending, and `decision_function` has shape
/// `(n_samples, n_classes)`
/// (`ferrolearn-model-sel/src/multiclass.rs:142-223`), mirroring sklearn
/// `fit` (`sklearn/multiclass.py:362-382`) + `decision_function` (`:554-582`).
///
/// Live oracle (sklearn 1.5.2, run from /tmp):
/// ```text
/// import numpy as np
/// from sklearn.multiclass import OneVsRestClassifier
/// from sklearn.svm import SVC
/// X=np.array([[10,10.],[8,10],[-5,5.5],[-5.4,5.5],[-20,-20],[-15,-20]])
/// y=np.array([0,0,1,1,2,2])
/// clf=OneVsRestClassifier(SVC()).fit(X,y)
/// len(clf.estimators_), clf.classes_.tolist(), clf.decision_function(X[:3]).shape
/// # -> 3, [0, 1, 2], (3, 3)
/// ```
#[test]
fn ovr_mechanic_k_estimators_sorted_classes_df_shape() {
    let x = Array2::from_shape_vec(
        (6, 2),
        vec![
            10.0, 10.0, 8.0, 10.0, -5.0, 5.5, -5.4, 5.5, -20.0, -20.0, -15.0, -20.0,
        ],
    )
    .unwrap();
    let y = Array1::from_vec(vec![0, 0, 1, 1, 2, 2]);

    let fitted = OneVsRestClassifier::new(threshold_factory())
        .fit(&x, &y)
        .unwrap();

    // Oracle: 3 estimators, classes_ == [0,1,2], decision_function (3,3).
    assert_eq!(fitted.n_estimators(), 3, "K classes => K estimators");
    assert_eq!(fitted.classes(), &[0, 1, 2], "classes() sorted ascending");
    let df = fitted.decision_function(&x).unwrap();
    assert_eq!(df.nrows(), 6);
    assert_eq!(df.ncols(), 3, "decision_function is (n_samples, n_classes)");
}

// ===========================================================================
// Clean (non-tie) end-to-end agreement: OvR and OvO both recover the obvious
// class on a well-separated 3-class problem. Confirms the OvO i/j vote
// convention is self-consistent (no convention-flip divergence). GREEN GUARD.
// ===========================================================================

/// Divergence guard: on a cleanly separated 3-class problem (no vote/score
/// tie), ferrolearn OvR and OvO `predict` both return the obvious class for
/// every sample — the same label sklearn `OneVsRestClassifier` /
/// `OneVsOneClassifier` return.
///
/// Live oracle (sklearn 1.5.2, run from /tmp):
/// ```text
/// import numpy as np
/// from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
/// from sklearn.svm import SVC
/// X=np.array([[0.,0.1],[0.1,0.],[0.2,0.1],[5.,5.1],[5.1,5.],[5.2,5.1],
///             [10.,10.1],[10.1,10.],[10.2,10.1]])
/// y=np.array([0,0,0,1,1,1,2,2,2])
/// OneVsRestClassifier(SVC()).fit(X,y).predict(X).tolist()  # -> [0,0,0,1,1,1,2,2,2]
/// OneVsOneClassifier(SVC()).fit(X,y).predict(X).tolist()   # -> [0,0,0,1,1,1,2,2,2]
/// ```
#[test]
fn ovr_ovo_clean_case_end_to_end_agree() {
    let x = Array2::from_shape_vec(
        (9, 2),
        vec![
            0.0, 0.1, 0.1, 0.0, 0.2, 0.1, 5.0, 5.1, 5.1, 5.0, 5.2, 5.1, 10.0, 10.1, 10.1, 10.0,
            10.2, 10.1,
        ],
    )
    .unwrap();
    let y = Array1::from_vec(vec![0, 0, 0, 1, 1, 1, 2, 2, 2]);
    let expected = [0usize, 0, 0, 1, 1, 1, 2, 2, 2];

    let ovr = OneVsRestClassifier::new(threshold_factory())
        .fit(&x, &y)
        .unwrap();
    let ovr_preds = ovr.predict(&x).unwrap();
    for (s, (&p, &e)) in ovr_preds.iter().zip(expected.iter()).enumerate() {
        assert_eq!(p, e, "OvR clean sample {s} should be class {e} (sklearn)");
    }

    let ovo = OneVsOneClassifier::new(threshold_factory())
        .fit(&x, &y)
        .unwrap();
    let ovo_preds = ovo.predict(&x).unwrap();
    for (s, (&p, &e)) in ovo_preds.iter().zip(expected.iter()).enumerate() {
        assert_eq!(p, e, "OvO clean sample {s} should be class {e} (sklearn)");
    }
}

// ===========================================================================
// #1819 (doc #1812) — OvO predict confidence tie-break. FAILING PIN.
// ===========================================================================

/// DIVERGENCE PIN (#1819, doc REQ-OVO-DECISION planned #1812): ferrolearn OvO `predict`
/// (`ferrolearn-model-sel/src/multiclass.rs:415-457`) tallies PURE integer
/// votes and `max_by_key` (LAST-on-tie) with NO confidence term. sklearn OvO
/// `predict` (`sklearn/multiclass.py:935-939`) is
/// `classes_[decision_function(X).argmax(1)]` where `decision_function`
/// (`:983`) = `_ovr_decision_function(predictions, confidences, K)` =
/// `votes + sum_of_confidences/(3*(|sum|+1))`
/// (`sklearn/utils/multiclass.py:540-562`), breaking vote ties by SUMMED
/// CONFIDENCE.
///
/// Fixture (classes [0,1,2], pairs (0,1),(0,2),(1,2), `Col1MeanScore` base):
/// feature column 1 per class is class0=0, class1=7, class2=0, so each pair's
/// constant predict score is the pair mean of column 1:
///   pair(0,1) = (0+7)/2 = 3.5  (> 0.5 -> ferrolearn votes LOWER class 0)
///   pair(0,2) = (0+0)/2 = 0.0  (<= 0.5 -> ferrolearn votes HIGHER class 2)
///   pair(1,2) = (7+0)/2 = 3.5  (> 0.5 -> ferrolearn votes LOWER class 1)
/// => integer votes = [1, 1, 1] (a 3-way tie). ferrolearn `max_by_key` returns
/// the LAST equal index => class 2.
///
/// The SAME outcomes map to sklearn's (predictions, confidences) under
/// `_fit_ovo_binary`'s lower->0 / higher->1 convention
/// (`sklearn/multiclass.py:626-645`) and `_predict_binary` confidence
/// (`:103-112`): each pair's `predict` = 0 when it favors the lower class
/// (score > 0.5) else 1; the per-pair confidence centers the score about the
/// 0.5 vote threshold toward the lower class. The exact (predictions,
/// confidences) yielding this vote tie with class 0 winning by confidence is
/// fed to the LIVE oracle below.
///
/// Live oracle (sklearn 1.5.2, run from /tmp) — a vote-tie where the confidence
/// term selects class 0, NOT ferrolearn's last-index class 2:
/// ```text
/// import numpy as np
/// from sklearn.utils.multiclass import _ovr_decision_function
/// predictions = np.array([[0, 1, 0]])      # votes: class0, class2, class1 -> [1,1,1] tie
/// confidences = np.array([[-3.0, 0.5, -3.0]])
/// Y = _ovr_decision_function(predictions, confidences, 3)
/// [round(v,4) for v in Y[0]]   # -> [1.238, 1.0, 0.762]
/// int(Y.argmax(1)[0])          # -> 0
/// ```
/// sklearn picks class 0; ferrolearn (pure-vote last-on-tie) picks class 2.
///
/// Un-ignore to confirm this FAILS against current ferrolearn. The fixer must
/// port `_ovr_decision_function` (OvO `decision_function`, #1813) and route
/// `predict` through it.
///
/// Tracking: #1819 (doc REQ-OVO-DECISION planned #1812)
#[test]
#[ignore = "divergence: OvO predict lacks _ovr_decision_function confidence tie-break; tracking #1819"]
fn ovo_predict_confidence_tiebreak_diverges() {
    // Column 0 is an arbitrary feature; column 1 encodes the per-class value
    // that produces the target per-pair mean score (see doc comment).
    // class0 rows: col1=0 ; class1 rows: col1=7 ; class2 rows: col1=0.
    let x = Array2::from_shape_vec(
        (6, 2),
        vec![
            0.0, 0.0, // class 0
            1.0, 0.0, // class 0
            2.0, 7.0, // class 1
            3.0, 7.0, // class 1
            4.0, 0.0, // class 2
            5.0, 0.0, // class 2
        ],
    )
    .unwrap();
    let y = Array1::from_vec(vec![0, 0, 1, 1, 2, 2]);

    let fitted = OneVsOneClassifier::new(col1_mean_factory())
        .fit(&x, &y)
        .unwrap();
    // Predict on a single arbitrary sample; the scores are constant per pair,
    // so the vote pattern is [1,1,1] regardless of the query row.
    let query = Array2::from_shape_vec((1, 2), vec![0.0, 0.0]).unwrap();
    let preds = fitted.predict(&query).unwrap();

    // sklearn oracle (above): the confidence-corrected argmax is class 0.
    const SK_PRED: usize = 0;
    assert_eq!(
        preds[0], SK_PRED,
        "OvO vote tie must break by summed confidence to class {SK_PRED} (sklearn), \
         but ferrolearn picked {} (pure-vote last-on-tie)",
        preds[0]
    );
}
