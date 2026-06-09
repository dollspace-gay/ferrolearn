//! Adversarial divergence pin (ACToR critic) for `GraphicalLasso` in the
//! NON-CONVERGED (max_iter-capped) regime vs scikit-learn 1.5.2.
//!
//! The converged covariance_/precision_ match sklearn to ~1e-14 (already pinned
//! green in `divergence_graphical_lasso.rs`). This file pins a REMAINING
//! divergence: when the outer loop hits `max_iter` BEFORE the dual gap drops
//! below `tol` (the case sklearn emits a `ConvergenceWarning` for,
//! `sklearn/covariance/_graph_lasso.py:190-195`), the truncated precision_
//! iterate differs from sklearn's by ~3.7e-6 — above the R-DEV-1 ~1e-6 bar.
//!
//! Oracle: live `sklearn.covariance.GraphicalLasso` v1.5.2 (ConvergenceWarning
//! suppressed). Fixture is `np.random.RandomState(42)` multivariate_normal data
//! (30x4) — the SAME bytes fed to ferrolearn. Expected values copied from the
//! sklearn oracle ONLY (R-CHAR-3). Reproduce:
//!
//! ```text
//! import warnings, numpy as np
//! from sklearn.exceptions import ConvergenceWarning
//! from sklearn.covariance import GraphicalLasso
//! warnings.simplefilter('ignore', ConvergenceWarning)
//! rng = np.random.RandomState(42)
//! A = rng.randn(4,4); cov_true = A @ A.T + np.eye(4)
//! X = rng.multivariate_normal(np.zeros(4), cov_true, size=30)
//! m = GraphicalLasso(alpha=0.01, max_iter=3, tol=1e-12).fit(X)
//! m.n_iter_           # 3 (loop exhausted, no convergence)
//! m.precision_[0,0]   # 0.3016369815423428
//! ```

use ferrolearn_core::traits::Fit;
use ferrolearn_covariance::GraphicalLasso;
use ndarray::{Array2, array};

#[allow(
    clippy::excessive_precision,
    reason = "verbatim sklearn-oracle RandomState(42).multivariate_normal bytes (R-CHAR-3); the extra decimal digits round to the same f64 and must not be truncated"
)]
fn fixture() -> Array2<f64> {
    array![
        [
            1.9729921512656956,
            0.4873844459862377,
            -1.9189333959839106,
            -2.675209895788814
        ],
        [
            -2.5771976434506896,
            -2.8542920638756892,
            -0.58902106772265417,
            3.1464328361391765
        ],
        [
            1.9257394821608118,
            -0.10666351361392168,
            0.14357677074133768,
            -1.4240998190124956
        ],
        [
            1.4633980994953495,
            0.68423238322940183,
            1.8484415667963792,
            -1.6297942259682028
        ],
        [
            -2.4626353682923101,
            -0.3220187059798062,
            -0.19888888613703776,
            -1.4051414131777602
        ],
        [
            -1.5722131384168092,
            -3.103453618243575,
            2.2606166338732203,
            -2.1500928776159753
        ],
        [
            -0.6651818977814048,
            -1.2457919435400473,
            -0.16841573090689016,
            1.9896133520238466
        ],
        [
            1.5920704735050153,
            1.6458926600027926,
            1.1841026687550842,
            -4.511251349989589
        ],
        [
            -3.1073144033236644,
            -1.6713187029265095,
            1.4359748372942087,
            -1.3112842201285426
        ],
        [
            1.0087604665243646,
            2.8630730659546613,
            -0.050179255216773543,
            -0.52834342063599238
        ],
        [
            0.63519863693187539,
            1.7248699123377582,
            0.87836426417210545,
            -2.2246260943037912
        ],
        [
            0.921830406212924,
            -0.87024210964529669,
            -1.0235693350678559,
            -1.9422971646935261
        ],
        [
            1.1601740660518396,
            -0.019896324649066658,
            -0.064326296499043048,
            3.9197283760881492
        ],
        [
            -1.0982286149320075,
            -0.2250597360862546,
            2.1642280336211552,
            0.52200847913797543
        ],
        [
            4.620471817739479,
            -1.1574320635380666,
            -0.53021480902465412,
            1.4124177297765952
        ],
        [
            -1.2071565762189225,
            -0.8792864568662152,
            -1.5503054390255495,
            -0.54184426033364419
        ],
        [
            -0.72626633923514727,
            1.9568321520328573,
            -1.0341180879637988,
            0.11391035323946427
        ],
        [
            -0.36766366317811489,
            1.9240509571169482,
            0.41988935165128216,
            -2.3859028865791307
        ],
        [
            1.5300999284369066,
            1.6212619386682205,
            0.21184663990824829,
            -0.48277281179654036
        ],
        [
            0.29850828749372199,
            0.015786146083991873,
            -1.2785188942022303,
            -2.5738094036024419
        ],
        [
            -0.098535236086197187,
            -0.33613220917334435,
            -0.36387714279916417,
            1.0229998533241413
        ],
        [
            1.2076701776127432,
            1.3243598639827778,
            -0.83343590796000255,
            -4.3432338461629714
        ],
        [
            -0.87112445386171811,
            2.5256079173534092,
            -0.44318559418292314,
            0.55695169847438863
        ],
        [
            1.2579489599266713,
            -2.4151214483006633,
            0.30214855234490423,
            0.11632284645909929
        ],
        [
            3.3858724509525162,
            1.6729187507089214,
            -2.0528544899873622,
            3.2085503304943006
        ],
        [
            -2.2309497167384795,
            0.55473397742224728,
            1.7219358222931143,
            -1.1114348480113663
        ],
        [
            -3.806387517404167,
            -0.94808686684670684,
            -0.26702228975765763,
            0.90687142573373836
        ],
        [
            2.8153159725117454,
            -0.44009551950038062,
            -2.3441418514654906,
            3.8410155804272916
        ],
        [
            0.60147746012439696,
            -2.0895505997923323,
            0.7161381115539982,
            -0.70820759914564357
        ],
        [
            3.1944712428446183,
            1.5805998939919998,
            0.68618663102916355,
            -1.9891045728605352
        ]
    ]
}

/// Divergence: ferrolearn `GraphicalLasso::fit` produces a different
/// non-converged precision_ iterate than sklearn when `max_iter` is reached
/// before the dual gap < `tol`.
///
/// sklearn `_graphical_lasso` (`sklearn/covariance/_graph_lasso.py:120-185`)
/// runs the outer loop `for i in range(max_iter)`, breaking on
/// `np.abs(d_gap) < tol` (`:184`); at `max_iter=3, tol=1e-12` it exhausts the
/// loop (`n_iter_=3`) and returns precision_[0,0] = 0.3016369815423428.
/// ferrolearn's `solve_glasso` returns precision_[0,0] ~= 0.3016407093383834
/// for the same input — a per-iterate gap of ~3.7e-6, exceeding R-DEV-1 ~1e-6.
/// (The CONVERGED outputs at default tol DO match to ~1e-14; this divergence is
/// confined to the truncated/ConvergenceWarning regime, where ferrolearn's
/// inner Gram-CD evolves the precision differently across early outer steps.)
///
/// Tracking: #2361
#[test]
fn divergence_graphical_lasso_noconverge_precision() {
    // sklearn oracle: GraphicalLasso(alpha=0.01, max_iter=3, tol=1e-12).
    const SK_PREC_00: f64 = 0.3016369815423428;
    const SK_PREC_11: f64 = 0.4836717810016965;
    const SK_PREC_22: f64 = 0.8772367433643129;
    let f = GraphicalLasso::<f64>::new(0.01)
        .max_iter(3)
        .tol(1e-12)
        .fit(&fixture(), &())
        .unwrap();
    // n_iter_ DOES match (loop exhausted at 3) — assert it to localise the gap
    // to the iterate values, not the loop count.
    assert_eq!(f.n_iter(), 3, "n_iter_ should be the loop cap 3");
    let p = f.precision();
    assert!(
        (p[[0, 0]] - SK_PREC_00).abs() < 1e-6,
        "precision_[0,0]: sklearn {SK_PREC_00}, ferrolearn {} (diff {:e})",
        p[[0, 0]],
        (p[[0, 0]] - SK_PREC_00).abs()
    );
    assert!(
        (p[[1, 1]] - SK_PREC_11).abs() < 1e-6,
        "precision_[1,1]: sklearn {SK_PREC_11}, ferrolearn {}",
        p[[1, 1]]
    );
    assert!(
        (p[[2, 2]] - SK_PREC_22).abs() < 1e-6,
        "precision_[2,2]: sklearn {SK_PREC_22}, ferrolearn {}",
        p[[2, 2]]
    );
}
