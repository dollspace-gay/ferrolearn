//! Divergence pins: `Average::Binary` positive-class selection in
//! `ferrolearn-metrics/src/classification.rs` vs scikit-learn 1.5.2.
//!
//! ferrolearn's `Average::Binary` hard-codes the positive class as
//! `classes[1]` — index 1 of the SORTED unique labels (see
//! `classification.rs` `aggregate_metric`/`aggregate_recall`/`aggregate_f1`,
//! e.g. `Ok(safe_div(tp[1] as f64, (tp[1] + fp[1]) as f64))`, and
//! `jaccard_score` `Ok(per_class[1])`).
//!
//! scikit-learn's `average="binary"` instead reports the class whose LABEL
//! VALUE equals `pos_label` (default `pos_label=1`):
//!   `sklearn/metrics/_classification.py:757` — `pos_label=1` (the default),
//!   `sklearn/metrics/_classification.py:1573` — `labels = [pos_label]`.
//! So the positive class is always the label whose value is 1, regardless of
//! its position in the sorted label set.
//!
//! For zero-based labels {0, 1} the two definitions coincide (label 1 IS the
//! index-1 element), which is why the existing happy-path binary guards pass.
//! For a binary problem with labels {1, 2}, they diverge: sklearn reports
//! label 1 (sorted index 0); ferrolearn reports label 2 (index 1).
//!
//! All expected values are from the LIVE sklearn 1.5.2 oracle (the exact
//! `python3 -c` call is quoted in each test, R-CHAR-3) — never copied from the
//! ferrolearn side. Fixture: `y_true=[1,1,1,2,2]`, `y_pred=[1,2,2,2,2]`.
//!
//! Tracking: #2292.

use ferrolearn_metrics::classification::{
    Average, f1_score, fbeta_score, jaccard_score, precision_score, recall_score,
};
use ndarray::Array1;

fn lab(v: &[usize]) -> Array1<usize> {
    Array1::from_vec(v.to_vec())
}

/// Divergence: `precision_score(.., Average::Binary)` reports `classes[1]`
/// (label 2) instead of sklearn's `pos_label=1` (label 1).
///
/// Oracle:
/// ```text
/// python3 -c "import numpy as np; from sklearn.metrics import precision_score; \
///   print(precision_score(np.array([1,1,1,2,2]), np.array([1,2,2,2,2])))"
/// # 1.0
/// ```
/// sklearn (label 1): TP=1, FP=0 -> 1.0.
/// ferrolearn (label 2): TP=2, FP=2 -> 0.5.
/// Tracking: #2292
#[test]
#[ignore = "divergence: Average::Binary uses index-1 not pos_label=1; tracking #2292"]
fn divergence_precision_binary_nonzero_labels() {
    const SK_PRECISION: f64 = 1.0;
    let yt = lab(&[1, 1, 1, 2, 2]);
    let yp = lab(&[1, 2, 2, 2, 2]);
    let got = precision_score(&yt, &yp, Average::Binary).unwrap();
    assert!(
        (got - SK_PRECISION).abs() < 1e-12,
        "precision_score binary: sklearn(pos_label=1)={SK_PRECISION}, ferrolearn={got}"
    );
}

/// Divergence: `recall_score(.., Average::Binary)` reports `classes[1]`
/// (label 2) instead of sklearn's `pos_label=1` (label 1).
///
/// Oracle:
/// ```text
/// python3 -c "import numpy as np; from sklearn.metrics import recall_score; \
///   print(recall_score(np.array([1,1,1,2,2]), np.array([1,2,2,2,2])))"
/// # 0.3333333333333333
/// ```
/// sklearn (label 1): TP=1, FN=2 -> 1/3. ferrolearn (label 2): TP=2, FN=0 -> 1.0.
/// Tracking: #2292
#[test]
#[ignore = "divergence: Average::Binary uses index-1 not pos_label=1; tracking #2292"]
fn divergence_recall_binary_nonzero_labels() {
    const SK_RECALL: f64 = 1.0 / 3.0;
    let yt = lab(&[1, 1, 1, 2, 2]);
    let yp = lab(&[1, 2, 2, 2, 2]);
    let got = recall_score(&yt, &yp, Average::Binary).unwrap();
    assert!(
        (got - SK_RECALL).abs() < 1e-12,
        "recall_score binary: sklearn(pos_label=1)={SK_RECALL}, ferrolearn={got}"
    );
}

/// Divergence: `f1_score(.., Average::Binary)` reports `classes[1]` (label 2)
/// instead of sklearn's `pos_label=1` (label 1).
///
/// Oracle:
/// ```text
/// python3 -c "import numpy as np; from sklearn.metrics import f1_score; \
///   print(f1_score(np.array([1,1,1,2,2]), np.array([1,2,2,2,2])))"
/// # 0.5
/// ```
/// sklearn (label 1): prec=1.0, rec=1/3 -> f1=0.5.
/// ferrolearn (label 2): prec=0.5, rec=1.0 -> f1=2/3.
/// Tracking: #2292
#[test]
#[ignore = "divergence: Average::Binary uses index-1 not pos_label=1; tracking #2292"]
fn divergence_f1_binary_nonzero_labels() {
    const SK_F1: f64 = 0.5;
    let yt = lab(&[1, 1, 1, 2, 2]);
    let yp = lab(&[1, 2, 2, 2, 2]);
    let got = f1_score(&yt, &yp, Average::Binary).unwrap();
    assert!(
        (got - SK_F1).abs() < 1e-12,
        "f1_score binary: sklearn(pos_label=1)={SK_F1}, ferrolearn={got}"
    );
}

/// Divergence: `fbeta_score(.., beta=2, Average::Binary)` reports `classes[1]`
/// (label 2) instead of sklearn's `pos_label=1` (label 1).
///
/// Oracle:
/// ```text
/// python3 -c "import numpy as np; from sklearn.metrics import fbeta_score; \
///   print(fbeta_score(np.array([1,1,1,2,2]), np.array([1,2,2,2,2]), beta=2.0))"
/// # 0.38461538461538464
/// ```
/// sklearn (label 1): prec=1.0, rec=1/3 -> F2 = 5*prec*rec/(4*prec+rec) = 0.3846.
/// ferrolearn (label 2): prec=0.5, rec=1.0 -> F2 = 0.8333.
/// Tracking: #2292
#[test]
#[ignore = "divergence: Average::Binary uses index-1 not pos_label=1; tracking #2292"]
fn divergence_fbeta_binary_nonzero_labels() {
    const SK_FBETA2: f64 = 0.384_615_384_615_384_64;
    let yt = lab(&[1, 1, 1, 2, 2]);
    let yp = lab(&[1, 2, 2, 2, 2]);
    let got = fbeta_score(&yt, &yp, 2.0, Average::Binary).unwrap();
    assert!(
        (got - SK_FBETA2).abs() < 1e-12,
        "fbeta_score(beta=2) binary: sklearn(pos_label=1)={SK_FBETA2}, ferrolearn={got}"
    );
}

/// Divergence: `jaccard_score(.., Average::Binary)` reports `classes[1]`
/// (label 2) instead of sklearn's `pos_label=1` (label 1).
///
/// Oracle:
/// ```text
/// python3 -c "import numpy as np; from sklearn.metrics import jaccard_score; \
///   print(jaccard_score(np.array([1,1,1,2,2]), np.array([1,2,2,2,2])))"
/// # 0.3333333333333333
/// ```
/// sklearn (label 1): TP=1, FP=0, FN=2 -> 1/3.
/// ferrolearn (label 2): TP=2, FP=2, FN=0 -> 0.5.
/// Tracking: #2292
#[test]
#[ignore = "divergence: Average::Binary uses index-1 not pos_label=1; tracking #2292"]
fn divergence_jaccard_binary_nonzero_labels() {
    const SK_JACCARD: f64 = 1.0 / 3.0;
    let yt = lab(&[1, 1, 1, 2, 2]);
    let yp = lab(&[1, 2, 2, 2, 2]);
    let got = jaccard_score(&yt, &yp, Average::Binary).unwrap();
    assert!(
        (got - SK_JACCARD).abs() < 1e-12,
        "jaccard_score binary: sklearn(pos_label=1)={SK_JACCARD}, ferrolearn={got}"
    );
}
