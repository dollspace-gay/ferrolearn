//! Divergence / green-guard suite for `ferrolearn-bayes/src/lib.rs`'s only
//! real behavior, the crate-level helper `log_softmax_rows`.
//!
//! `log_softmax_rows<F>(jll) -> jll - logsumexp(jll, axis=1)` is the
//! numerically stable body of sklearn `_BaseNB.predict_log_proba`
//! (`sklearn/naive_bayes.py:105-126`, `jll - np.atleast_2d(logsumexp(jll,
//! axis=1)).T`, where `logsumexp` = `scipy.special.logsumexp`, imported `:21`)
//! and feeds `_BaseNB.predict_proba = np.exp(predict_log_proba)`
//! (`naive_bayes.py:128-144`).
//!
//! `log_softmax_rows` is `pub(crate)`, so it is exercised here through its two
//! public routes:
//!   * Probe 1 — end-to-end through the public `GaussianNB` estimator
//!     (`predict_log_proba` / `predict_proba`), which delegates through
//!     `BaseNB::nb_predict_log_proba` -> `log_softmax_rows`.
//!   * Probes 2-4 — through a local `BaseNB` impl (`StubNB`) that returns an
//!     arbitrary fixed jll, so the all-`-inf`, single-column, and
//!     large-magnitude jll rows can be driven directly into `log_softmax_rows`
//!     via the public `nb_predict_log_proba` / `nb_predict_proba` methods.
//!
//! R-CHAR-3: every expected value below was obtained from a LIVE
//! sklearn 1.5.2 / scipy oracle (commands quoted at each test), never copied
//! from the ferrolearn side.

use ferrolearn_bayes::{BaseNB, GaussianNB};
use ferrolearn_core::{FerroError, Fit};
use ndarray::{Array1, Array2, array};

/// Minimal `BaseNB` implementor returning a fixed joint-log-likelihood matrix,
/// so an arbitrary `jll` can be pushed straight into `log_softmax_rows` via the
/// public `nb_predict_log_proba` / `nb_predict_proba` pipeline.
struct StubNB {
    classes: Vec<usize>,
    jll: Array2<f64>,
}

impl BaseNB<f64> for StubNB {
    fn joint_log_likelihood(&self, _x: &Array2<f64>) -> Result<Array2<f64>, FerroError> {
        Ok(self.jll.clone())
    }
    fn nb_classes(&self) -> &[usize] {
        &self.classes
    }
}

// ---------------------------------------------------------------------------
// Probe 1 — normalization correctness end-to-end through GaussianNB.
//
// Live oracle (sklearn 1.5.2):
//   python3 -c "
//   import numpy as np
//   from sklearn.naive_bayes import GaussianNB
//   X = np.array([[1.0,2.0],[1.5,2.5],[1.2,1.8],[6.0,7.0],[5.8,6.5],[6.2,7.2]])
//   y = np.array([0,0,0,1,1,1])
//   clf = GaussianNB().fit(X,y)
//   Xt = np.array([[2.0,2.0],[6.0,6.5]])
//   print(clf.predict_log_proba(Xt).tolist())
//   print(clf.predict_proba(Xt).tolist())"
//   predict_log_proba = [[0.0, -431.2711718694835], [-380.0647416360279, 0.0]]
//   predict_proba     = [[1.0, 5.0270112036202264e-188],
//                        [8.709233641855078e-166, 1.0]]
// GREEN guard: ferrolearn must reproduce these to ~1e-12 and row-sum to 1.
// ---------------------------------------------------------------------------

/// Green guard: `FittedGaussianNB::predict_log_proba` / `predict_proba`
/// (delegating through `log_softmax_rows`) match the live sklearn 1.5.2
/// oracle for the design-doc GaussianNB probe.
/// Mirrors `_BaseNB.predict_log_proba` (`sklearn/naive_bayes.py:123-126`).
#[test]
fn green_log_softmax_gaussian_matches_oracle() {
    // sklearn-oracle expected values (NOT from ferrolearn):
    const SK_LOGPROBA: [[f64; 2]; 2] = [[0.0, -431.2711718694835], [-380.0647416360279, 0.0]];
    const SK_PROBA: [[f64; 2]; 2] = [
        [1.0, 5.0270112036202264e-188],
        [8.709233641855078e-166, 1.0],
    ];

    let x = Array2::from_shape_vec(
        (6, 2),
        vec![1.0, 2.0, 1.5, 2.5, 1.2, 1.8, 6.0, 7.0, 5.8, 6.5, 6.2, 7.2],
    )
    .unwrap();
    let y: Array1<usize> = array![0, 0, 0, 1, 1, 1];
    let fitted = GaussianNB::<f64>::new().fit(&x, &y).unwrap();

    let xt = array![[2.0, 2.0], [6.0, 6.5]];
    let logproba = fitted.predict_log_proba(&xt).unwrap();
    let proba = fitted.predict_proba(&xt).unwrap();

    for i in 0..2 {
        for j in 0..2 {
            // log-proba: absolute tolerance scaled by magnitude (~1e-12 rel).
            let lp_tol = 1e-12_f64.max(SK_LOGPROBA[i][j].abs() * 1e-12);
            assert!(
                (logproba[[i, j]] - SK_LOGPROBA[i][j]).abs() <= lp_tol,
                "predict_log_proba[{i}][{j}]: sklearn {} vs ferrolearn {}",
                SK_LOGPROBA[i][j],
                logproba[[i, j]]
            );
            // proba: tiny entries are ~1e-188 -> compare relatively.
            let pdiff = (proba[[i, j]] - SK_PROBA[i][j]).abs();
            let prel = if SK_PROBA[i][j] != 0.0 {
                pdiff / SK_PROBA[i][j]
            } else {
                pdiff
            };
            assert!(
                prel <= 1e-9,
                "predict_proba[{i}][{j}]: sklearn {} vs ferrolearn {}",
                SK_PROBA[i][j],
                proba[[i, j]]
            );
        }
        let rowsum: f64 = (0..2).map(|j| proba[[i, j]]).sum();
        assert!(
            (rowsum - 1.0).abs() <= 1e-12,
            "proba row {i} sums to {rowsum}, expected 1.0"
        );
    }
}

// ---------------------------------------------------------------------------
// Probe 2 — the all-`-inf` jll row edge.
//
// Live oracle (scipy 1.x bundled with sklearn 1.5.2):
//   python3 -c "
//   import numpy as np
//   from scipy.special import logsumexp
//   jll = np.array([[-np.inf, -np.inf]])
//   print(logsumexp(jll, axis=1).tolist())                # [-inf]
//   print((jll - logsumexp(jll,axis=1)[:,None]).tolist())"# [[nan, nan]]
//   logsumexp([-inf,-inf]) = -inf -> jll - (-inf) = nan (per element).
//
// `log_softmax_rows` arithmetic: max = -inf; (-inf)-(-inf) = NaN;
//   sum_exp = NaN; log_norm = -inf + ln(NaN) = NaN; out = -inf - NaN = NaN.
// GREEN guard: ferrolearn must produce NaN (matching scipy). If it instead
// guards `max==-inf` and returns a finite value or -inf, this assertion fails
// and PINS a divergence.
// ---------------------------------------------------------------------------

/// Green guard: an all-`-inf` jll row yields NaN log-proba and NaN proba,
/// matching `scipy.special.logsumexp([-inf,-inf]) = -inf` -> `jll - (-inf) =
/// nan` as used by `_BaseNB.predict_log_proba` (`sklearn/naive_bayes.py:125`).
#[test]
fn green_log_softmax_all_neg_inf_row_is_nan() {
    let stub = StubNB {
        classes: vec![0, 1],
        jll: array![[f64::NEG_INFINITY, f64::NEG_INFINITY]],
    };
    let x = Array2::<f64>::zeros((1, 1));

    let logproba = stub.nb_predict_log_proba(&x).unwrap();
    let proba = stub.nb_predict_proba(&x).unwrap();

    for j in 0..2 {
        assert!(
            logproba[[0, j]].is_nan(),
            "all-(-inf) row: predict_log_proba[0][{j}] = {} (scipy gives NaN)",
            logproba[[0, j]]
        );
        assert!(
            proba[[0, j]].is_nan(),
            "all-(-inf) row: predict_proba[0][{j}] = {} (scipy gives NaN)",
            proba[[0, j]]
        );
    }
}

// ---------------------------------------------------------------------------
// Probe 3 — single-column jll (n_classes == 1).
//
// Live oracle:
//   python3 -c "
//   import numpy as np
//   from scipy.special import logsumexp
//   jll1 = np.array([[-3.5]])
//   print((jll1 - logsumexp(jll1,axis=1)[:,None]).tolist())"  # [[0.0]]
//   logsumexp of one element = that element -> jll - jll = 0 -> proba = 1.0.
//   Cross-checked end-to-end: a single-class GaussianNB gives
//   predict_proba [[1.0]], predict_log_proba [[0.0]].
// GREEN guard.
// ---------------------------------------------------------------------------

/// Green guard: a single-column jll row gives log-proba 0.0 (proba 1.0),
/// matching `scipy.special.logsumexp([-3.5]) = -3.5` -> `jll - jll = 0`
/// (`_BaseNB.predict_log_proba`, `sklearn/naive_bayes.py:125`).
#[test]
fn green_log_softmax_single_column_is_zero() {
    const SK_LOGPROBA: f64 = 0.0; // scipy: jll - logsumexp(jll) for n_classes==1
    const SK_PROBA: f64 = 1.0; // exp(0.0)

    let stub = StubNB {
        classes: vec![0],
        jll: array![[-3.5]],
    };
    let x = Array2::<f64>::zeros((1, 1));

    let logproba = stub.nb_predict_log_proba(&x).unwrap();
    let proba = stub.nb_predict_proba(&x).unwrap();

    assert!(
        (logproba[[0, 0]] - SK_LOGPROBA).abs() <= 1e-15,
        "single-column predict_log_proba = {} (scipy {SK_LOGPROBA})",
        logproba[[0, 0]]
    );
    assert!(
        (proba[[0, 0]] - SK_PROBA).abs() <= 1e-15,
        "single-column predict_proba = {} (scipy {SK_PROBA})",
        proba[[0, 0]]
    );
}

// ---------------------------------------------------------------------------
// Probe 4 — large-magnitude jll: the max-subtraction must prevent overflow.
//
// Live oracle:
//   python3 -c "
//   import numpy as np
//   from scipy.special import logsumexp
//   jll = np.array([[1000.0,1001.0]])
//   lse = logsumexp(jll, axis=1)
//   print((jll - lse[:,None]).tolist())          # [[-1.3132616875182066, -0.3132616875182066]]
//   print(np.exp(jll - lse[:,None]).tolist())"   # [[0.2689414213699995, 0.7310585786300168]]
//   Naive exp(1000) overflows to +inf; max-subtraction keeps results finite.
// GREEN guard: ferrolearn must match scipy and stay finite (no inf/NaN).
// ---------------------------------------------------------------------------

/// Green guard: a `[1000, 1001]` jll row matches scipy `logsumexp` and stays
/// finite — the max-subtraction stabilization in `log_softmax_rows` mirrors
/// `scipy.special.logsumexp` (`sklearn/naive_bayes.py:125`).
#[test]
fn green_log_softmax_large_magnitude_no_overflow() {
    // scipy-oracle expected values (NOT from ferrolearn):
    const SK_LOGPROBA: [f64; 2] = [-1.3132616875182066, -0.3132616875182066];
    const SK_PROBA: [f64; 2] = [0.2689414213699995, 0.7310585786300168];

    let stub = StubNB {
        classes: vec![0, 1],
        jll: array![[1000.0, 1001.0]],
    };
    let x = Array2::<f64>::zeros((1, 1));

    let logproba = stub.nb_predict_log_proba(&x).unwrap();
    let proba = stub.nb_predict_proba(&x).unwrap();

    for j in 0..2 {
        assert!(
            logproba[[0, j]].is_finite() && proba[[0, j]].is_finite(),
            "large-magnitude jll produced non-finite output: logproba {} proba {}",
            logproba[[0, j]],
            proba[[0, j]]
        );
        assert!(
            (logproba[[0, j]] - SK_LOGPROBA[j]).abs() <= 1e-12,
            "large predict_log_proba[{j}] = {} (scipy {})",
            logproba[[0, j]],
            SK_LOGPROBA[j]
        );
        assert!(
            (proba[[0, j]] - SK_PROBA[j]).abs() <= 1e-12,
            "large predict_proba[{j}] = {} (scipy {})",
            proba[[0, j]],
            SK_PROBA[j]
        );
    }
    let rowsum = proba[[0, 0]] + proba[[0, 1]];
    assert!(
        (rowsum - 1.0).abs() <= 1e-12,
        "large-magnitude proba row sums to {rowsum}, expected 1.0"
    );
}
