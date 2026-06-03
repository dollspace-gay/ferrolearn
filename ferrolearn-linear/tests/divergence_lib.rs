//! Divergence tests for the `ferrolearn-linear` crate-root score helpers
//! (`r2_score` / `mean_accuracy`) reached through the public score traits.
//!
//! Expected values are obtained from the LIVE scikit-learn 1.5.2 oracle (see
//! the per-test doc comment for the exact `python3 -c` invocation), never
//! copied from the ferrolearn side (R-CHAR-3).

use ferrolearn_core::error::FerroError;
use ferrolearn_core::traits::Predict;
use ferrolearn_linear::RegressorScore;
use ndarray::{Array1, Array2};

/// A minimal regressor whose `predict` returns predictions supplied by the
/// test, so the test fully controls `y_pred` while `score(&x, &y)` supplies
/// `y_true`. This reaches the `pub(crate)` `r2_score` helper through the
/// public `RegressorScore::score` blanket impl (the only documented surface).
struct FixedRegressor {
    preds: Array1<f64>,
}

impl Predict<Array2<f64>> for FixedRegressor {
    type Output = Array1<f64>;
    type Error = FerroError;

    fn predict(&self, _x: &Array2<f64>) -> Result<Self::Output, Self::Error> {
        Ok(self.preds.clone())
    }
}

/// Divergence: ferrolearn's `r2_score` (backing `RegressorScore::score`) at
/// `ferrolearn-linear/src/lib.rs:217-222` returns `F::neg_infinity()` for the
/// `ss_tot == 0 && ss_res != 0` branch (constant `y_true`, non-zero residual).
///
/// scikit-learn `RegressorMixin.score` delegates to `sklearn.metrics.r2_score`
/// (`sklearn/base.py:849`: `return r2_score(y, y_pred, ...)`), which sets the
/// score to `0.0` for that case
/// (`sklearn/metrics/_regression.py:891`:
/// `output_scores[nonzero_numerator & ~nonzero_denominator] = 0.0`), emitting
/// an `UndefinedMetricWarning`.
///
/// Live oracle (sklearn 1.5.2):
///   python3 -c "import warnings; from sklearn.metrics import r2_score;
///     warnings.simplefilter('ignore');
///     print(r2_score([5.,5.,5.],[4.,5.,6.]))"  -> 0.0
/// (`r2_score(y_true=[5,5,5], y_pred=[4,5,6])`).
///
/// Here `score(&x, &y)` passes `y = [5,5,5]` as the true targets and the fixed
/// predictions `[4,5,6]` as `y_pred`; sklearn returns 0.0, ferrolearn returns
/// -inf.
/// Tracking: #1104 (release-blocker; left un-ignored).
#[test]
fn divergence_r2_constant_ytrue_nonzero_residual_returns_zero() {
    // Live-oracle expected value, NOT copied from ferrolearn.
    const SK_R2_CONST_Y_RESID: f64 = 0.0;

    let reg = FixedRegressor {
        preds: Array1::from(vec![4.0_f64, 5.0, 6.0]),
    };
    let x = Array2::<f64>::zeros((3, 1)); // ignored by FixedRegressor
    let y = Array1::from(vec![5.0_f64, 5.0, 5.0]); // constant y_true

    let score = reg.score(&x, &y).expect("score should succeed");

    assert_eq!(
        score, SK_R2_CONST_Y_RESID,
        "constant y_true with non-zero residual: sklearn r2_score returns {SK_R2_CONST_Y_RESID}, \
         ferrolearn returned {score} (tracking #1104)",
    );
}

/// Companion (NOT a divergence — ferrolearn agrees here): constant `y_true`
/// with zero residual must return `1.0`, matching the live oracle
///   python3 -c "from sklearn.metrics import r2_score;
///     print(r2_score([5.,5.,5.],[5.,5.,5.]))"  -> 1.0
/// This guards against a fix that over-corrects the edge to also zero out the
/// perfect-prediction case. It is expected to PASS today.
#[test]
fn r2_constant_ytrue_zero_residual_returns_one() {
    const SK_R2_CONST_Y_PERFECT: f64 = 1.0;

    let reg = FixedRegressor {
        preds: Array1::from(vec![5.0_f64, 5.0, 5.0]),
    };
    let x = Array2::<f64>::zeros((3, 1));
    let y = Array1::from(vec![5.0_f64, 5.0, 5.0]);

    let score = reg.score(&x, &y).expect("score should succeed");

    assert_eq!(score, SK_R2_CONST_Y_PERFECT);
}

/// In-regime parity (NOT a divergence — expected to PASS): non-degenerate `y`.
/// Live oracle:
///   python3 -c "from sklearn.metrics import r2_score;
///     print(r2_score([3.,5.,2.,7.],[2.5,5.,2.,8.]))"  -> 0.9152542372881356
/// where sklearn args are (y_true=[3,5,2,7], y_pred=[2.5,5,2,8]). In
/// ferrolearn `score(&x, &y)`, `y` is y_true and the fixed predictions are
/// y_pred, so y = [3,5,2,7], preds = [2.5,5,2,8].
#[test]
fn r2_in_regime_matches_oracle() {
    const SK_R2_IN_REGIME: f64 = 0.9152542372881356;

    let reg = FixedRegressor {
        preds: Array1::from(vec![2.5_f64, 5.0, 2.0, 8.0]),
    };
    let x = Array2::<f64>::zeros((4, 1));
    let y = Array1::from(vec![3.0_f64, 5.0, 2.0, 7.0]);

    let score = reg.score(&x, &y).expect("score should succeed");

    assert!(
        (score - SK_R2_IN_REGIME).abs() < 1e-8,
        "in-regime R^2: sklearn {SK_R2_IN_REGIME}, ferrolearn {score}",
    );
}
