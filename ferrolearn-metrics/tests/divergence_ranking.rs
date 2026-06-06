//! Divergence pins for `ferrolearn-metrics/src/ranking.rs` vs scikit-learn 1.5.2.
//!
//! All expected values come from the LIVE sklearn 1.5.2 oracle (the `python3 -c`
//! call is quoted in each test, R-CHAR-3) or a sklearn `file:line` symbolic
//! constant — NEVER copied from the ferrolearn side.
//!
//! ferrolearn's `dcg_score`/`ndcg_score` are 1D (single ranking); sklearn's are
//! 2D `(n_samples, n_labels)`. We call ferrolearn's 1D API with the single-row
//! vectors and compare to sklearn's single-row 2D result (= the per-row score).
//!
//! RED pins (deterministic correctness divergences — fixers must fix):
//!   - divergence_dcg_tie_averaging          (#754)
//!   - divergence_ndcg_tie_averaging         (#755)
//!   - divergence_ndcg_negative_y_true_guard (#755)
//!   - divergence_label_ranking_loss_degenerate_denominator (#758)
//!
//! GREEN guards (oracle-grounded, must pass now — guard the correct paths):
//!   - green_coverage_error_basic
//!   - green_lrap_basic
//!   - green_dcg_no_tie
//!   - green_ndcg_no_tie

use ferrolearn_metrics::ranking::{
    coverage_error, dcg_score, label_ranking_average_precision_score, label_ranking_loss,
    ndcg_score,
};
use ndarray::{Array2, array};

// ===========================================================================
// RED pins — deterministic correctness divergences
// ===========================================================================

/// Divergence: `dcg_score` does NOT tie-average (sklearn default
/// `ignore_ties=False` → `_tie_averaged_dcg`, `sklearn/metrics/_ranking.py:1528`;
/// dispatched from `_dcg_sample_scores` `:1518-1524`). ferrolearn's
/// `fn argsort_desc` resolves tied `y_score` by index — that is sklearn's
/// `ignore_ties=True` path (`:1514-1517`), the WRONG default.
///
/// Oracle (sklearn docstring example, `_ranking.py:1688-1689`):
///   python3 -c "import numpy as np; from sklearn.metrics import dcg_score; \
///     print(dcg_score(np.array([[10,0,0,1,5]]), np.array([[1,0,0,0,1]]), k=1))"
///   # np.float64(7.5)   ← average true relevance of the tied top group (10+5)/2
/// ferrolearn returns 10.0 (picks index-0 of the tie). Tracking: #754
#[test]
fn divergence_dcg_tie_averaging() {
    let y_true = array![10.0_f64, 0.0, 0.0, 1.0, 5.0];
    let y_score = array![1.0_f64, 0.0, 0.0, 0.0, 1.0];
    let got = dcg_score(&y_true, &y_score, Some(1), None).unwrap();
    // sklearn 1.5.2 live oracle:
    const SK_DCG_TIE_K1: f64 = 7.5;
    assert!(
        (got - SK_DCG_TIE_K1).abs() < 1e-9,
        "dcg_score tie-averaging: sklearn={SK_DCG_TIE_K1}, ferrolearn={got}"
    );
}

/// Divergence: `ndcg_score` does NOT tie-average (same root cause as dcg —
/// reuses `fn argsort_desc`/`fn compute_dcg`; sklearn `ndcg_score` `:1770` over
/// `_ndcg_sample_scores` `:1749` honors `ignore_ties=False` by default).
///
/// Oracle (sklearn docstring example, `_ranking.py:1855-1856`):
///   python3 -c "import numpy as np; from sklearn.metrics import ndcg_score; \
///     print(ndcg_score(np.array([[10,0,0,1,5]]), np.array([[1,0,0,0,1]]), k=1))"
///   # np.float64(0.75)
/// ferrolearn returns 1.0. Tracking: #755
#[test]
fn divergence_ndcg_tie_averaging() {
    let y_true = array![10.0_f64, 0.0, 0.0, 1.0, 5.0];
    let y_score = array![1.0_f64, 0.0, 0.0, 0.0, 1.0];
    let got = ndcg_score(&y_true, &y_score, Some(1)).unwrap();
    // sklearn 1.5.2 live oracle:
    const SK_NDCG_TIE_K1: f64 = 0.75;
    assert!(
        (got - SK_NDCG_TIE_K1).abs() < 1e-9,
        "ndcg_score tie-averaging: sklearn={SK_NDCG_TIE_K1}, ferrolearn={got}"
    );
}

/// Divergence: `ndcg_score` lacks the negative-`y_true` guard. sklearn raises
/// `ValueError("ndcg_score should not be used on negative y_true values.")` at
/// `sklearn/metrics/_ranking.py:1868-1869`. ferrolearn returns Ok with a value.
///
/// Oracle:
///   python3 -c "import numpy as np; from sklearn.metrics import ndcg_score; \
///     ndcg_score(np.array([[-1.0,2,3]]), np.array([[1.0,2,3]]))"
///   # ValueError: ndcg_score should not be used on negative y_true values.
/// Tracking: #755
#[test]
fn divergence_ndcg_negative_y_true_guard() {
    let y_true = array![-1.0_f64, 2.0, 3.0];
    let y_score = array![1.0_f64, 2.0, 3.0];
    let got = ndcg_score(&y_true, &y_score, None);
    // sklearn raises ValueError; ferrolearn must return Err (R-DEV-2).
    assert!(
        got.is_err(),
        "ndcg_score negative-y_true: sklearn raises ValueError; ferrolearn returned {got:?}"
    );
}

/// Divergence: `label_ranking_loss` normalizes by the wrong denominator on
/// degenerate rows. sklearn sets a degenerate (all-same-label) row's loss to
/// `0.0` and averages over ALL `n_samples` (`sklearn/metrics/_ranking.py:1463`
/// then `np.average(loss, ...)` `:1465`). ferrolearn skips degenerate rows and
/// divides by `counted` (the non-degenerate count).
///
/// Oracle:
///   python3 -c "import numpy as np; from sklearn.metrics import label_ranking_loss; \
///     print(label_ranking_loss(np.array([[1,0,0],[0,0,0]]), \
///       np.array([[0.75,0.5,1.0],[0.9,0.8,0.7]])))"
///   # np.float64(0.25)   ← row0 loss 0.5, row1 degenerate→0, mean over 2 rows
/// ferrolearn returns 0.5 (divides by counted=1). Tracking: #758
#[test]
fn divergence_label_ranking_loss_degenerate_denominator() {
    let y_true: Array2<usize> = array![[1, 0, 0], [0, 0, 0]];
    let y_score: Array2<f64> = array![[0.75, 0.5, 1.0], [0.9, 0.8, 0.7]];
    let got = label_ranking_loss(&y_true, &y_score).unwrap();
    // sklearn 1.5.2 live oracle:
    const SK_LRL_DEGEN: f64 = 0.25;
    assert!(
        (got - SK_LRL_DEGEN).abs() < 1e-9,
        "label_ranking_loss degenerate denominator: sklearn={SK_LRL_DEGEN}, ferrolearn={got}"
    );
}

// ===========================================================================
// GREEN guards — oracle-grounded; must pass now (protect the correct paths)
// ===========================================================================

/// Guard: `coverage_error` basic case matches sklearn.
/// Oracle:
///   python3 -c "import numpy as np; from sklearn.metrics import coverage_error; \
///     print(coverage_error(np.array([[1,0,0],[0,0,1]]), \
///       np.array([[0.1,0.2,0.3],[0.7,0.6,0.5]])))"
///   # np.float64(3.0)
#[test]
fn green_coverage_error_basic() {
    let y_true: Array2<usize> = array![[1, 0, 0], [0, 0, 1]];
    let y_score: Array2<f64> = array![[0.1, 0.2, 0.3], [0.7, 0.6, 0.5]];
    let got = coverage_error(&y_true, &y_score).unwrap();
    const SK_COV_BASIC: f64 = 3.0;
    assert!(
        (got - SK_COV_BASIC).abs() < 1e-9,
        "coverage_error basic: sklearn={SK_COV_BASIC}, ferrolearn={got}"
    );
}

/// Guard: `label_ranking_average_precision_score` basic case matches sklearn.
/// Oracle:
///   python3 -c "import numpy as np; \
///     from sklearn.metrics import label_ranking_average_precision_score as lrap; \
///     print(lrap(np.array([[1,0,0],[0,1,1]]), np.array([[0.1,0.2,0.3],[0.7,0.6,0.5]])))"
///   # np.float64(0.45833333333333326)
#[test]
fn green_lrap_basic() {
    let y_true: Array2<usize> = array![[1, 0, 0], [0, 1, 1]];
    let y_score: Array2<f64> = array![[0.1, 0.2, 0.3], [0.7, 0.6, 0.5]];
    let got = label_ranking_average_precision_score(&y_true, &y_score).unwrap();
    const SK_LRAP_BASIC: f64 = 0.458_333_333_333_333_26;
    assert!(
        (got - SK_LRAP_BASIC).abs() < 1e-9,
        "lrap basic: sklearn={SK_LRAP_BASIC}, ferrolearn={got}"
    );
}

/// Guard: `dcg_score` no-tie (distinct y_score) matches sklearn — the no-tie
/// path the fixer must NOT regress. Single-row 2D → ferrolearn 1D.
/// Oracle:
///   python3 -c "import numpy as np; from sklearn.metrics import dcg_score; \
///     print(dcg_score(np.array([[3,2,3,0,1,2]]), np.array([[6,5,4,3,2,1]])))"
///   # np.float64(6.861126688593501)
#[test]
fn green_dcg_no_tie() {
    let y_true = array![3.0_f64, 2.0, 3.0, 0.0, 1.0, 2.0];
    let y_score = array![6.0_f64, 5.0, 4.0, 3.0, 2.0, 1.0];
    let got = dcg_score(&y_true, &y_score, None, None).unwrap();
    const SK_DCG_NOTIE: f64 = 6.861_126_688_593_501;
    assert!(
        (got - SK_DCG_NOTIE).abs() < 1e-9,
        "dcg_score no-tie: sklearn={SK_DCG_NOTIE}, ferrolearn={got}"
    );
}

/// Guard: `ndcg_score` no-tie (distinct y_score) matches sklearn.
/// Oracle:
///   python3 -c "import numpy as np; from sklearn.metrics import ndcg_score; \
///     print(ndcg_score(np.array([[3,2,3,0,1,2]]), np.array([[6,5,4,3,2,1]])))"
///   # np.float64(0.9608081943360616)
#[test]
fn green_ndcg_no_tie() {
    let y_true = array![3.0_f64, 2.0, 3.0, 0.0, 1.0, 2.0];
    let y_score = array![6.0_f64, 5.0, 4.0, 3.0, 2.0, 1.0];
    let got = ndcg_score(&y_true, &y_score, None).unwrap();
    const SK_NDCG_NOTIE: f64 = 0.960_808_194_336_061_6;
    assert!(
        (got - SK_NDCG_NOTIE).abs() < 1e-9,
        "ndcg_score no-tie: sklearn={SK_NDCG_NOTIE}, ferrolearn={got}"
    );
}
