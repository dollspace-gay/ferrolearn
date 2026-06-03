//! Divergence pins for `ferrolearn-metrics/src/classification.rs` vs scikit-learn 1.5.2.
//!
//! All expected values come from the LIVE sklearn 1.5.2 oracle (the exact
//! `python3 -c` call is quoted in each test, R-CHAR-3) — NEVER copied from the
//! ferrolearn side. The clone at `/home/doll/scikit-learn` and the installed
//! `sklearn` package are both at tag 1.5.2 (commit 156ef14), so the source line
//! citations and the oracle values are the same version by construction.
//!
//! RED pins (deterministic DEFAULT-PATH value divergences to fix this iteration):
//!   - `red_log_loss_clip_eps` — `log_loss` hard-codes `EPS=1e-15`
//!     (classification.rs:537); sklearn clips to `np.finfo(float64).eps =
//!     2.22e-16` (_classification.py:2951). On a true-class prob of exactly 0,
//!     ferrolearn returns ~34.5388, sklearn 36.04365338911715.
//!   - `red_roc_curve_drop_intermediate_default` — `roc_curve` keeps every
//!     distinct-score point (classification.rs `pub fn roc_curve`); sklearn's
//!     default `drop_intermediate=True` drops collinear interior points
//!     (_ranking.py:1158), so the thresholds shrink from 9 to 4.
//!   - `red_det_curve_drops_inf_endpoint` — `det_curve` retains the prepended
//!     `(0,0)`/`+inf` ROC endpoint (classification.rs `pub fn det_curve`);
//!     sklearn's `det_curve` drops it (_ranking.py:362-376), giving 3 points
//!     not 4.
//!   - `red_calibration_curve_searchsorted_binning` — `calibration_curve` bins
//!     by `floor(prob * n_bins)` (classification.rs:1245); sklearn bins by
//!     `np.searchsorted(bins[1:-1], prob)` (calibration.py:1035), so a value on
//!     a bin boundary (0.5 with n_bins=2) lands in a different bin.
//!
//! GREEN guards establish the SHIPPED value contracts: each present function ==
//! a live sklearn 1.5.2 value on its supported (binary/integer-label,
//! no-`sample_weight`) signature. They must stay green.

use ferrolearn_metrics::classification::{
    Average, accuracy_score, auc, average_precision_score, balanced_accuracy_score,
    brier_score_loss, calibration_curve, cohen_kappa_score, confusion_matrix, det_curve, f1_score,
    fbeta_score, hamming_loss, hinge_loss, jaccard_score, log_loss, matthews_corrcoef,
    precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve,
    top_k_accuracy_score, zero_one_loss,
};
use ndarray::{Array1, Array2, array};

fn lab(v: &[usize]) -> Array1<usize> {
    Array1::from(v.to_vec())
}

// ===========================================================================
// RED pins — deterministic default-path value divergences to fix this iteration
// ===========================================================================

/// RED — `log_loss` clips to a hard-coded `EPS=1e-15`; sklearn clips to
/// `np.finfo(float64).eps = 2.220446049250313e-16`.
///
/// ferrolearn `pub fn log_loss` defines `const EPS: f64 = 1e-15;`
/// (classification.rs:537) and clamps the true-class probability via
/// `.clamp(EPS, 1.0 - EPS)` (classification.rs:549). sklearn 1.5.2 sets
/// `eps = np.finfo(y_pred.dtype).eps` (`_classification.py:2951`) then
/// `y_pred = np.clip(y_pred, eps, 1 - eps)` (`_classification.py:2962`), so for
/// a true-class probability of exactly 0 it returns `-ln(2.22e-16)`, not
/// `-ln(1e-15)`.
///
/// Oracle (sklearn 1.5.2, run from /tmp):
///   python3 -c "import numpy as np; from sklearn.metrics import log_loss; \
///     print(repr(log_loss([0,1],np.array([[0.,1.],[1.,0.]]),labels=[0,1])))"
///   # 36.04365338911715
///   python3 -c "import numpy as np; print(repr(-np.log(np.finfo(np.float64).eps)))"
///   # 36.04365338911715
#[test]
fn red_log_loss_clip_eps() {
    let y_true = lab(&[0, 1]);
    // Sample 0 true class 0 has prob 0; sample 1 true class 1 has prob 0.
    let y_prob = Array2::from_shape_vec((2, 2), vec![0.0, 1.0, 1.0, 0.0]).unwrap();
    let loss = log_loss(&y_true, &y_prob).expect("log_loss should not error on valid input");
    // sklearn 1.5.2 live oracle: -ln(np.finfo(float64).eps).
    const SK: f64 = 36.043_653_389_117_15;
    assert!(
        (loss - SK).abs() / SK < 1e-9,
        "log_loss([0,1],[[0,1],[1,0]]): sklearn={SK}, ferrolearn={loss}"
    );
}

/// RED — `roc_curve` does not implement `drop_intermediate` (sklearn default
/// `True`); it keeps every distinct-score threshold.
///
/// ferrolearn `pub fn roc_curve` (classification.rs) pushes one point per
/// distinct score plus the `(0,0)`/`+inf` prepend, reproducing sklearn's
/// `drop_intermediate=False` arrays. sklearn `roc_curve` defaults
/// `drop_intermediate=True` (`_ranking.py:1054`) and drops collinear suboptimal
/// interior points via `if drop_intermediate and len(fps) > 2:`
/// (`_ranking.py:1158`), so an 8-sample monotone separation collapses to
/// thresholds `[inf, 0.8, 0.5, 0.1]` (len 4) — not the 9 distinct scores.
///
/// Oracle (sklearn 1.5.2, run from /tmp):
///   python3 -c "from sklearn.metrics import roc_curve; \
///     f,t,th=roc_curve([0,0,0,0,1,1,1,1],[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]); \
///     print(f.tolist(), t.tolist(), th.tolist())"
///   # fpr [0.0, 0.0, 0.0, 1.0]
///   # tpr [0.0, 0.25, 1.0, 1.0]
///   # th  [inf, 0.8, 0.5, 0.1]
#[test]
fn red_roc_curve_drop_intermediate_default() {
    let y_true = lab(&[0, 0, 0, 0, 1, 1, 1, 1]);
    let y_score = array![0.1_f64, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
    let (fpr, tpr, th) = roc_curve(&y_true, &y_score).expect("roc_curve should not error");

    // sklearn 1.5.2 live oracle (drop_intermediate=True default): 4 points.
    let sk_fpr = [0.0_f64, 0.0, 0.0, 1.0];
    let sk_tpr = [0.0_f64, 0.25, 1.0, 1.0];
    let sk_th = [f64::INFINITY, 0.8, 0.5, 0.1];

    assert_eq!(
        th.len(),
        sk_th.len(),
        "roc_curve thresholds length: sklearn={} (drop_intermediate=True), ferrolearn={}",
        sk_th.len(),
        th.len()
    );
    for i in 0..sk_th.len() {
        assert!(
            (fpr[i] - sk_fpr[i]).abs() < 1e-12,
            "roc_curve fpr[{i}]: sklearn={}, ferrolearn={}",
            sk_fpr[i],
            fpr[i]
        );
        assert!(
            (tpr[i] - sk_tpr[i]).abs() < 1e-12,
            "roc_curve tpr[{i}]: sklearn={}, ferrolearn={}",
            sk_tpr[i],
            tpr[i]
        );
        let thr = th[i];
        if sk_th[i].is_infinite() {
            assert!(
                thr.is_infinite(),
                "roc_curve th[{i}]: sklearn=inf, ferrolearn={thr}"
            );
        } else {
            assert!(
                (thr - sk_th[i]).abs() < 1e-12,
                "roc_curve th[{i}]: sklearn={}, ferrolearn={thr}",
                sk_th[i]
            );
        }
    }
}

/// RED — `det_curve` retains the prepended `(0,0)`/`+inf` ROC endpoint that
/// sklearn drops.
///
/// ferrolearn `pub fn det_curve` (classification.rs) delegates to `roc_curve`
/// and just maps `fnr = 1 - tpr`, keeping the `(fpr=0, fnr=1, th=+inf)` ROC
/// prepend. sklearn's `det_curve` slices it off:
/// `fps[0] == 0 ... fns = (tps[-1] - tps); fpr = fps[1:] / ...` and returns
/// `fpr[sl], fnr[sl], thresholds[sl]` without the (0,0)/+inf endpoint
/// (`_ranking.py:376 (slice built at :362-373)`), yielding 3 points.
///
/// Oracle (sklearn 1.5.2, run from /tmp):
///   python3 -c "from sklearn.metrics import det_curve; \
///     f,fn,th=det_curve([0,0,1,1],[0.1,0.4,0.35,0.8]); \
///     print(f.tolist(), fn.tolist(), th.tolist())"
///   # fpr [0.5, 0.5, 0.0]
///   # fnr [0.0, 0.5, 0.5]
///   # th  [0.35, 0.4, 0.8]
#[test]
fn red_det_curve_drops_inf_endpoint() {
    let y_true = lab(&[0, 0, 1, 1]);
    let y_score = array![0.1_f64, 0.4, 0.35, 0.8];
    let (fpr, fnr, th) = det_curve(&y_true, &y_score).expect("det_curve should not error");

    // sklearn 1.5.2 live oracle: 3 points, no (0,0)/+inf endpoint.
    let sk_fpr = [0.5_f64, 0.5, 0.0];
    let sk_fnr = [0.0_f64, 0.5, 0.5];
    let sk_th = [0.35_f64, 0.4, 0.8];

    assert_eq!(
        th.len(),
        sk_th.len(),
        "det_curve length: sklearn={} (no +inf endpoint), ferrolearn={}",
        sk_th.len(),
        th.len()
    );
    for i in 0..sk_th.len() {
        assert!(
            (fpr[i] - sk_fpr[i]).abs() < 1e-12,
            "det_curve fpr[{i}]: sklearn={}, ferrolearn={}",
            sk_fpr[i],
            fpr[i]
        );
        assert!(
            (fnr[i] - sk_fnr[i]).abs() < 1e-12,
            "det_curve fnr[{i}]: sklearn={}, ferrolearn={}",
            sk_fnr[i],
            fnr[i]
        );
        assert!(
            (th[i] - sk_th[i]).abs() < 1e-12,
            "det_curve th[{i}]: sklearn={}, ferrolearn={}",
            sk_th[i],
            th[i]
        );
    }
}

/// RED — `calibration_curve` bins by `floor(prob*n_bins)`; sklearn bins by
/// `np.searchsorted(bins[1:-1], prob)`, so a value on a bin boundary lands in a
/// different bin.
///
/// ferrolearn `pub fn calibration_curve` computes
/// `let mut bin = (prob * n_bins_f).to_usize()...` (classification.rs:1245), so
/// `0.5` with `n_bins=2` → `floor(1.0) = 1` (bin 1). sklearn computes
/// `binids = np.searchsorted(bins[1:-1], y_prob)` (`calibration.py:1035`); with
/// `bins=[0,0.5,1]` and `bins[1:-1]=[0.5]`, `searchsorted([0.5], 0.5)` is `0`
/// (left side), so the two `0.5` samples land in bin **0** with the `0.0`
/// sample → `prob_true=[1/3, 1.0]`, `prob_pred=[1/3, 1.0]`.
///
/// Oracle (sklearn 1.5.2, run from /tmp):
///   python3 -c "from sklearn.calibration import calibration_curve; \
///     pt,pp=calibration_curve([0,1,0,1],[0.5,0.5,0.0,1.0],n_bins=2); \
///     print(pt.tolist(), pp.tolist())"
///   # prob_true [0.3333333333333333, 1.0]
///   # prob_pred [0.3333333333333333, 1.0]
#[test]
fn red_calibration_curve_searchsorted_binning() {
    let y_true = lab(&[0, 1, 0, 1]);
    let y_prob = array![0.5_f64, 0.5, 0.0, 1.0];
    let (prob_true, prob_pred) =
        calibration_curve(&y_true, &y_prob, 2).expect("calibration_curve should not error");

    // sklearn 1.5.2 live oracle: searchsorted puts the two 0.5s in bin 0.
    let sk_pt = [1.0_f64 / 3.0, 1.0];
    let sk_pp = [1.0_f64 / 3.0, 1.0];

    assert_eq!(
        prob_true.len(),
        sk_pt.len(),
        "calibration_curve n_bins-occupied: sklearn={}, ferrolearn={}",
        sk_pt.len(),
        prob_true.len()
    );
    for i in 0..sk_pt.len() {
        assert!(
            (prob_true[i] - sk_pt[i]).abs() < 1e-12,
            "calibration_curve prob_true[{i}]: sklearn={}, ferrolearn={}",
            sk_pt[i],
            prob_true[i]
        );
        assert!(
            (prob_pred[i] - sk_pp[i]).abs() < 1e-12,
            "calibration_curve prob_pred[{i}]: sklearn={}, ferrolearn={}",
            sk_pp[i],
            prob_pred[i]
        );
    }
}

// ===========================================================================
// GREEN guards — oracle-grounded SHIPPED value contracts (must pass now)
// ===========================================================================

/// Guard: `accuracy_score` default fraction.
///
/// Oracle (sklearn 1.5.2, run from /tmp):
///   python3 -c "from sklearn.metrics import accuracy_score; \
///     print(repr(accuracy_score([0,1,2,1,0],[0,1,2,0,0])))"
///   # 0.8
#[test]
fn green_accuracy_score() {
    let acc = accuracy_score(&lab(&[0, 1, 2, 1, 0]), &lab(&[0, 1, 2, 0, 0])).unwrap();
    const SK: f64 = 0.8;
    assert!(
        (acc - SK).abs() < 1e-12,
        "accuracy_score: sklearn={SK}, ferrolearn={acc}"
    );
}

/// Guard: `confusion_matrix` default sorted-union counts.
///
/// Oracle (sklearn 1.5.2, run from /tmp):
///   python3 -c "from sklearn.metrics import confusion_matrix; \
///     print(confusion_matrix([0,1,2,0,1,2],[0,2,2,0,0,2]).tolist())"
///   # [[2, 0, 0], [1, 0, 1], [0, 0, 2]]
#[test]
fn green_confusion_matrix() {
    let cm = confusion_matrix(&lab(&[0, 1, 2, 0, 1, 2]), &lab(&[0, 2, 2, 0, 0, 2])).unwrap();
    let sk: Array2<usize> = array![[2, 0, 0], [1, 0, 1], [0, 0, 2]];
    assert_eq!(
        cm, sk,
        "confusion_matrix: sklearn={sk:?}, ferrolearn={cm:?}"
    );
}

/// Guard: `matthews_corrcoef` value + zero-denominator → 0.0.
///
/// Oracle (sklearn 1.5.2, run from /tmp):
///   python3 -c "from sklearn.metrics import matthews_corrcoef as m; \
///     print(repr(m([0,1,1,0],[0,1,0,0])), repr(m([0,1,1,0],[1,1,1,1])))"
///   # 0.5773502691896258 0.0
#[test]
fn green_matthews_corrcoef() {
    let mcc = matthews_corrcoef(&lab(&[0, 1, 1, 0]), &lab(&[0, 1, 0, 0])).unwrap();
    const SK: f64 = 0.577_350_269_189_625_8;
    assert!(
        (mcc - SK).abs() < 1e-12,
        "matthews_corrcoef: sklearn={SK}, ferrolearn={mcc}"
    );
    // Zero-denominator convention: sklearn returns 0.0.
    let mcc0 = matthews_corrcoef(&lab(&[0, 1, 1, 0]), &lab(&[1, 1, 1, 1])).unwrap();
    assert!(
        mcc0.abs() < 1e-12,
        "matthews_corrcoef zero-denom: sklearn=0.0, ferrolearn={mcc0}"
    );
}

/// Guard: `balanced_accuracy_score` default + adjusted.
///
/// Oracle (sklearn 1.5.2, run from /tmp):
///   python3 -c "from sklearn.metrics import balanced_accuracy_score as b; \
///     print(repr(b([0,0,1,1,2,2],[0,1,1,1,2,0])), \
///           repr(b([0,0,1,1,2,2],[0,1,1,1,2,0],adjusted=True)))"
///   # 0.6666666666666666 0.49999999999999994
#[test]
fn green_balanced_accuracy_score() {
    let yt = lab(&[0, 0, 1, 1, 2, 2]);
    let yp = lab(&[0, 1, 1, 1, 2, 0]);
    let ba = balanced_accuracy_score(&yt, &yp, false).unwrap();
    const SK: f64 = 0.666_666_666_666_666_6;
    assert!(
        (ba - SK).abs() < 1e-12,
        "balanced_accuracy_score: sklearn={SK}, ferrolearn={ba}"
    );
    let ba_adj = balanced_accuracy_score(&yt, &yp, true).unwrap();
    const SK_ADJ: f64 = 0.499_999_999_999_999_94;
    assert!(
        (ba_adj - SK_ADJ).abs() < 1e-12,
        "balanced_accuracy_score(adjusted): sklearn={SK_ADJ}, ferrolearn={ba_adj}"
    );
}

/// Guard: `roc_auc_score` (binary).
///
/// Oracle (sklearn 1.5.2, run from /tmp):
///   python3 -c "from sklearn.metrics import roc_auc_score; \
///     print(repr(roc_auc_score([0,0,1,1],[0.1,0.4,0.35,0.8])))"
///   # 0.75
#[test]
fn green_roc_auc_score() {
    let a = roc_auc_score(&lab(&[0, 0, 1, 1]), &array![0.1, 0.4, 0.35, 0.8]).unwrap();
    const SK: f64 = 0.75;
    assert!(
        (a - SK).abs() < 1e-12,
        "roc_auc_score: sklearn={SK}, ferrolearn={a}"
    );
}

/// Guard: `auc` trapezoidal area.
///
/// Oracle (sklearn 1.5.2, run from /tmp):
///   python3 -c "import numpy as np; from sklearn.metrics import auc; \
///     print(repr(auc(np.array([0.,0.,1.]),np.array([0.,1.,1.]))))"
///   # 1.0
#[test]
fn green_auc() {
    let a = auc(&array![0.0_f64, 0.0, 1.0], &array![0.0_f64, 1.0, 1.0]).unwrap();
    const SK: f64 = 1.0;
    assert!((a - SK).abs() < 1e-12, "auc: sklearn={SK}, ferrolearn={a}");
}

/// Guard: `precision_recall_curve` arrays + `(P=1,R=0)` sentinel.
///
/// Oracle (sklearn 1.5.2, run from /tmp):
///   python3 -c "from sklearn.metrics import precision_recall_curve as prc; \
///     p,r,th=prc([0,0,1,1],[0.1,0.4,0.35,0.8]); \
///     print(p.tolist(), r.tolist(), th.tolist())"
///   # p  [0.5, 0.6666666666666666, 0.5, 1.0, 1.0]
///   # r  [1.0, 1.0, 0.5, 0.5, 0.0]
///   # th [0.1, 0.35, 0.4, 0.8]
#[test]
fn green_precision_recall_curve() {
    let (p, r, th) =
        precision_recall_curve(&lab(&[0, 0, 1, 1]), &array![0.1_f64, 0.4, 0.35, 0.8]).unwrap();
    let sk_p = [0.5_f64, 2.0 / 3.0, 0.5, 1.0, 1.0];
    let sk_r = [1.0_f64, 1.0, 0.5, 0.5, 0.0];
    let sk_th = [0.1_f64, 0.35, 0.4, 0.8];
    assert_eq!(p.len(), sk_p.len(), "precision_recall_curve precision len");
    assert_eq!(
        th.len(),
        sk_th.len(),
        "precision_recall_curve thresholds len"
    );
    for i in 0..sk_p.len() {
        assert!(
            (p[i] - sk_p[i]).abs() < 1e-12,
            "prc precision[{i}]: sklearn={}",
            sk_p[i]
        );
        assert!(
            (r[i] - sk_r[i]).abs() < 1e-12,
            "prc recall[{i}]: sklearn={}",
            sk_r[i]
        );
    }
    for i in 0..sk_th.len() {
        assert!(
            (th[i] - sk_th[i]).abs() < 1e-12,
            "prc threshold[{i}]: sklearn={}",
            sk_th[i]
        );
    }
}

/// Guard: `average_precision_score` step integration (the 0.8333 case).
///
/// Oracle (sklearn 1.5.2, run from /tmp):
///   python3 -c "from sklearn.metrics import average_precision_score as ap; \
///     print(repr(ap([0,0,1,1],[0.1,0.4,0.35,0.8])))"
///   # 0.8333333333333333
#[test]
fn green_average_precision_score() {
    let ap =
        average_precision_score(&lab(&[0, 0, 1, 1]), &array![0.1_f64, 0.4, 0.35, 0.8]).unwrap();
    const SK: f64 = 0.833_333_333_333_333_3;
    assert!(
        (ap - SK).abs() < 1e-12,
        "average_precision_score: sklearn={SK}, ferrolearn={ap}"
    );
}

/// Guard: `precision_score`/`recall_score`/`f1_score` (binary default + macro).
///
/// Oracle (sklearn 1.5.2, run from /tmp):
///   python3 -c "from sklearn.metrics import precision_score as p, recall_score as r, f1_score as f; \
///     print(repr(p([0,1,1,0,1],[0,1,0,1,1])), repr(r([0,1,1,0,1],[0,1,0,1,1])), repr(f([0,1,1,0,1],[0,1,0,1,1])))"
///   # 0.6666666666666666 0.6666666666666666 0.6666666666666666
///   python3 -c "from sklearn.metrics import precision_score as p, recall_score as r, f1_score as f; \
///     print(repr(p([0,1,2,0,1,2],[0,2,1,0,0,1],average='macro')), \
///           repr(r([0,1,2,0,1,2],[0,2,1,0,0,1],average='macro')), \
///           repr(f([0,1,2,0,1,2],[0,2,1,0,0,1],average='macro')))"
///   # 0.2222222222222222 0.3333333333333333 0.26666666666666666
#[test]
fn green_precision_recall_f1_binary_and_macro() {
    let yt = lab(&[0, 1, 1, 0, 1]);
    let yp = lab(&[0, 1, 0, 1, 1]);
    const SK_BIN: f64 = 0.666_666_666_666_666_6;
    assert!((precision_score(&yt, &yp, Average::Binary).unwrap() - SK_BIN).abs() < 1e-12);
    assert!((recall_score(&yt, &yp, Average::Binary).unwrap() - SK_BIN).abs() < 1e-12);
    assert!((f1_score(&yt, &yp, Average::Binary).unwrap() - SK_BIN).abs() < 1e-12);

    let yt3 = lab(&[0, 1, 2, 0, 1, 2]);
    let yp3 = lab(&[0, 2, 1, 0, 0, 1]);
    const SK_PMAC: f64 = 0.222_222_222_222_222_2;
    const SK_RMAC: f64 = 0.333_333_333_333_333_3;
    const SK_FMAC: f64 = 0.266_666_666_666_666_66;
    assert!(
        (precision_score(&yt3, &yp3, Average::Macro).unwrap() - SK_PMAC).abs() < 1e-12,
        "precision macro: sklearn={SK_PMAC}"
    );
    assert!(
        (recall_score(&yt3, &yp3, Average::Macro).unwrap() - SK_RMAC).abs() < 1e-12,
        "recall macro: sklearn={SK_RMAC}"
    );
    assert!(
        (f1_score(&yt3, &yp3, Average::Macro).unwrap() - SK_FMAC).abs() < 1e-12,
        "f1 macro: sklearn={SK_FMAC}"
    );
}

/// Guard: `top_k_accuracy_score` (default fraction, k=2).
///
/// Oracle (sklearn 1.5.2, run from /tmp):
///   python3 -c "import numpy as np; from sklearn.metrics import top_k_accuracy_score as t; \
///     print(repr(t([0,1,2,2],np.array([[.7,.2,.1],[.1,.8,.1],[.3,.3,.4],[.1,.5,.4]]),k=2)))"
///   # 1.0
#[test]
fn green_top_k_accuracy_score() {
    let y_true = lab(&[0, 1, 2, 2]);
    let y_score = Array2::from_shape_vec(
        (4, 3),
        vec![0.7, 0.2, 0.1, 0.1, 0.8, 0.1, 0.3, 0.3, 0.4, 0.1, 0.5, 0.4],
    )
    .unwrap();
    let t = top_k_accuracy_score(&y_true, &y_score, 2).unwrap();
    const SK: f64 = 1.0;
    assert!(
        (t - SK).abs() < 1e-12,
        "top_k_accuracy_score: sklearn={SK}, ferrolearn={t}"
    );
}

/// Guard: `hamming_loss` (single-label).
///
/// Oracle (sklearn 1.5.2, run from /tmp):
///   python3 -c "from sklearn.metrics import hamming_loss; \
///     print(repr(hamming_loss([0,1,2,1,0],[0,1,2,0,0])))"
///   # 0.2
#[test]
fn green_hamming_loss() {
    let h = hamming_loss(&lab(&[0, 1, 2, 1, 0]), &lab(&[0, 1, 2, 0, 0])).unwrap();
    const SK: f64 = 0.2;
    assert!(
        (h - SK).abs() < 1e-12,
        "hamming_loss: sklearn={SK}, ferrolearn={h}"
    );
}

/// Guard: `zero_one_loss` fraction + count.
///
/// Oracle (sklearn 1.5.2, run from /tmp):
///   python3 -c "from sklearn.metrics import zero_one_loss as z; \
///     print(repr(z([0,1,2,1,0],[0,1,2,0,0])), repr(z([0,1,2,1,0],[0,1,2,0,0],normalize=False)))"
///   # 0.19999999999999996 1.0
#[test]
fn green_zero_one_loss() {
    let yt = lab(&[0, 1, 2, 1, 0]);
    let yp = lab(&[0, 1, 2, 0, 0]);
    let frac = zero_one_loss(&yt, &yp, true).unwrap();
    const SK_FRAC: f64 = 0.199_999_999_999_999_96;
    assert!(
        (frac - SK_FRAC).abs() < 1e-12,
        "zero_one_loss frac: sklearn={SK_FRAC}, ferrolearn={frac}"
    );
    let count = zero_one_loss(&yt, &yp, false).unwrap();
    const SK_COUNT: f64 = 1.0;
    assert!(
        (count - SK_COUNT).abs() < 1e-12,
        "zero_one_loss count: sklearn={SK_COUNT}, ferrolearn={count}"
    );
}

/// Guard: `brier_score_loss` (binary default pos_label).
///
/// Oracle (sklearn 1.5.2, run from /tmp):
///   python3 -c "from sklearn.metrics import brier_score_loss; \
///     print(repr(brier_score_loss([0,0,1,1],[0.1,0.4,0.35,0.8])))"
///   # 0.15812500000000002
#[test]
fn green_brier_score_loss() {
    let b = brier_score_loss(&lab(&[0, 0, 1, 1]), &array![0.1, 0.4, 0.35, 0.8]).unwrap();
    const SK: f64 = 0.158_125_000_000_000_02;
    assert!(
        (b - SK).abs() < 1e-12,
        "brier_score_loss: sklearn={SK}, ferrolearn={b}"
    );
}

/// Guard: `hinge_loss` (binary; labels 0/1 → -1/+1).
///
/// Oracle (sklearn 1.5.2, run from /tmp):
///   python3 -c "from sklearn.metrics import hinge_loss; \
///     print(repr(hinge_loss([0,1,1,0],[-0.5,0.7,0.3,-0.2])))"
///   # 0.575
#[test]
fn green_hinge_loss() {
    let h = hinge_loss(&lab(&[0, 1, 1, 0]), &array![-0.5, 0.7, 0.3, -0.2]).unwrap();
    const SK: f64 = 0.575;
    assert!(
        (h - SK).abs() < 1e-12,
        "hinge_loss: sklearn={SK}, ferrolearn={h}"
    );
}

/// Guard: `cohen_kappa_score` (unweighted).
///
/// Oracle (sklearn 1.5.2, run from /tmp):
///   python3 -c "from sklearn.metrics import cohen_kappa_score as c; \
///     print(repr(c([0,1,2,2,1],[0,2,2,1,1])))"
///   # 0.375
#[test]
fn green_cohen_kappa_score() {
    let k = cohen_kappa_score(&lab(&[0, 1, 2, 2, 1]), &lab(&[0, 2, 2, 1, 1])).unwrap();
    const SK: f64 = 0.375;
    assert!(
        (k - SK).abs() < 1e-12,
        "cohen_kappa_score: sklearn={SK}, ferrolearn={k}"
    );
}

/// Guard: `jaccard_score` (binary).
///
/// Oracle (sklearn 1.5.2, run from /tmp):
///   python3 -c "from sklearn.metrics import jaccard_score; \
///     print(repr(jaccard_score([0,1,1,0,1],[0,1,0,1,1])))"
///   # 0.5
#[test]
fn green_jaccard_score() {
    let j = jaccard_score(
        &lab(&[0, 1, 1, 0, 1]),
        &lab(&[0, 1, 0, 1, 1]),
        Average::Binary,
    )
    .unwrap();
    const SK: f64 = 0.5;
    assert!(
        (j - SK).abs() < 1e-12,
        "jaccard_score: sklearn={SK}, ferrolearn={j}"
    );
}

/// Guard: `fbeta_score` (binary, beta=2).
///
/// Oracle (sklearn 1.5.2, run from /tmp):
///   python3 -c "from sklearn.metrics import fbeta_score; \
///     print(repr(fbeta_score([0,1,1,0,1],[0,1,0,1,1],beta=2.0)))"
///   # 0.6666666666666666
#[test]
fn green_fbeta_score() {
    let f = fbeta_score(
        &lab(&[0, 1, 1, 0, 1]),
        &lab(&[0, 1, 0, 1, 1]),
        2.0,
        Average::Binary,
    )
    .unwrap();
    const SK: f64 = 0.666_666_666_666_666_6;
    assert!(
        (f - SK).abs() < 1e-12,
        "fbeta_score: sklearn={SK}, ferrolearn={f}"
    );
}

// ===========================================================================
// RE-AUDIT RED pins (harder-input divergences surfaced by the #807–#810
// re-audit; the four original RED pins above are now GREEN/fixed).
// ===========================================================================

/// RED — `calibration_curve` hand-rolled `fract()==0` binning diverges from
/// `np.searchsorted(np.linspace(0,1,n_bins+1)[1:-1], prob)` when a probability
/// lies one ULP above a bin edge whose float64 `linspace` value rounds down.
///
/// ferrolearn `pub fn calibration_curve` (classification.rs:1317-1323) bins via
/// `scaled = prob * n_bins; if scaled.fract()==0 { scaled-1 } else { ... }`,
/// which assumes the bin edge is EXACTLY `k/n_bins`. sklearn
/// (`sklearn/calibration.py:1035`) uses
/// `binids = np.searchsorted(bins[1:-1], y_prob)` where
/// `bins = np.linspace(0,1,n_bins+1)` — the interior edge for `k=2,n_bins=3` is
/// `linspace`'s `0.6666666666666666` (= `2/3` rounded DOWN), but the probe
/// value `v = nextafter(2/3, 1) = 0.6666666666666667` has `v*3 == 2.0` exactly,
/// so the hand-rolled path puts `v` in bin 1 (lower), while sklearn's
/// searchsorted puts `v` in bin 2 (because `v > linspace_edge`).
///
/// Oracle (sklearn 1.5.2, run from /tmp):
///   python3 -c "import numpy as np; from sklearn.calibration import calibration_curve as c; \
///     v=float(np.nextafter(2/3,1)); \
///     pt,pp=c([0,1,1,1],[0.1,0.5,v,0.9],n_bins=3); \
///     print(pt.tolist(), pp.tolist())"
///   # prob_true [0.0, 1.0, 1.0]
///   # prob_pred [0.1, 0.5, 0.7833333333333334]
/// ferrolearn returns prob_pred [0.1, 0.5833333333333334, 0.9].
#[test]
fn red_calibration_curve_float_edge_binning() {
    // v = nextafter(2/3, 1): one ULP above 2/3; v*3 == 2.0 exactly in f64.
    let v = f64::from_bits(0x3fe5555555555556);
    let y_true = lab(&[0, 1, 1, 1]);
    let y_prob = array![0.1_f64, 0.5, v, 0.9];
    let (prob_true, prob_pred) =
        calibration_curve(&y_true, &y_prob, 3).expect("calibration_curve should not error");

    // sklearn 1.5.2 live oracle: v lands in bin 2 (with 0.9), bin 1 holds only 0.5.
    let sk_pt = [0.0_f64, 1.0, 1.0];
    let sk_pp = [0.1_f64, 0.5, 0.783_333_333_333_333_4];

    assert_eq!(
        prob_pred.len(),
        sk_pp.len(),
        "calibration_curve n_bins-occupied: sklearn={}, ferrolearn={}",
        sk_pp.len(),
        prob_pred.len()
    );
    for i in 0..sk_pp.len() {
        assert!(
            (prob_true[i] - sk_pt[i]).abs() < 1e-12,
            "calibration_curve prob_true[{i}]: sklearn={}, ferrolearn={}",
            sk_pt[i],
            prob_true[i]
        );
        assert!(
            (prob_pred[i] - sk_pp[i]).abs() < 1e-12,
            "calibration_curve prob_pred[{i}]: sklearn={}, ferrolearn={}",
            sk_pp[i],
            prob_pred[i]
        );
    }
}

/// RED — `top_k_accuracy_score` breaks score ties toward the LOWER class index;
/// sklearn breaks them toward the HIGHER class index.
///
/// ferrolearn `pub fn top_k_accuracy_score` (classification.rs:1205-1210) does a
/// stable descending `sort_by(|a,b| row[b].partial_cmp(&row[a]))`, so tied
/// classes keep ascending-index order and the LOWER index enters the top-k.
/// sklearn (`sklearn/metrics/_ranking.py:2043`) computes
/// `sorted_pred = np.argsort(y_score, axis=1, kind="mergesort")[:, ::-1]` —
/// stable ASCENDING argsort then reversed, so tied classes are ranked with the
/// HIGHER index first. For sample `[0.1, 0.1, 0.8]` with true label 0, k=2:
/// sklearn's top-2 is `{2, 1}` (MISS); ferrolearn's is `{2, 0}` (HIT).
///
/// Oracle (sklearn 1.5.2, run from /tmp):
///   python3 -c "import numpy as np; from sklearn.metrics import top_k_accuracy_score as t; \
///     ys=np.array([[0.6,0.3,0.1],[0.2,0.5,0.3],[0.3,0.3,0.4],[0.1,0.1,0.8],[0.4,0.4,0.2]]); \
///     print(repr(t([0,1,2,0,1],ys,k=2)))"
///   # 0.8
/// ferrolearn returns 1.0 (counts the tied sample 3 as a hit).
#[test]
fn red_top_k_accuracy_tie_breaking() {
    let y_true = lab(&[0, 1, 2, 0, 1]);
    let y_score = Array2::from_shape_vec(
        (5, 3),
        vec![
            0.6, 0.3, 0.1, // sample 0: true 0 -> top2 {0,1} hit
            0.2, 0.5, 0.3, // sample 1: true 1 -> hit
            0.3, 0.3, 0.4, // sample 2: true 2 -> hit
            0.1, 0.1, 0.8, // sample 3: true 0, tie at 0.1 -> sklearn MISS
            0.4, 0.4, 0.2, // sample 4: true 1 -> hit
        ],
    )
    .unwrap();
    let t = top_k_accuracy_score(&y_true, &y_score, 2).unwrap();
    // sklearn 1.5.2 live oracle: 4/5 = 0.8 (tie broken toward higher index).
    const SK: f64 = 0.8;
    assert!(
        (t - SK).abs() < 1e-12,
        "top_k_accuracy_score tie-break: sklearn={SK}, ferrolearn={t}"
    );
}
