//! Divergence tests: ferrolearn `RFE` / `RFECV` vs scikit-learn 1.5.2
//! `sklearn/feature_selection/_rfe.py` (`class RFE` `:72`, `class RFECV` `:485`).
//!
//! All expected values come from the LIVE sklearn 1.5.2 oracle (run from /tmp)
//! or a sklearn `file:line` symbolic constant — NEVER copied from the
//! ferrolearn side (R-CHAR-3).
//!
//! SCOPE NOTE. ferrolearn's `RFE` takes a STATIC importance vector, not a
//! wrapped estimator. The estimator + per-round re-fit + squaring (REQ-3) and
//! the RFECV internal cross-validation (REQ-8) are structurally NOT-STARTED and
//! are deliberately NOT pinned here (R-DEFER-3). These tests pin only:
//!   * the ONE fixable boundary divergence DIV-1 (REQ-4, blocker #1296) — RED;
//!   * GREEN guards locking the SHIPPED static-importance ranking (REQ-1),
//!     RFECV optimal-count selection (REQ-9), and the scoped error contracts
//!     (REQ-2).
//!
//! The GREEN ranking guards solve the verification problem (sklearn re-fits the
//! estimator each round so per-round importances move) by using a fixture where
//! sklearn's per-round squared importances are STABLE: `X[:, j] = y * scale_j`
//! with widely separated scales, so dropping the weakest feature never reorders
//! the survivors. The importances fed to ferrolearn are sklearn's
//! `LinearRegression().fit(X, y).coef_ ** 2` (the squared-coef importance source
//! sklearn uses, `_rfe.py:326-330`), and the asserted ranking/support are
//! sklearn's `RFE(...).ranking_` / `.support_`.

use ferrolearn_core::traits::Transform;
use ferrolearn_preprocess::{RFE, RFECV};
use ndarray::{Array1, array};

// ===========================================================================
// RED PIN — DIV-1: n_features_to_select > n_features = warn + keep-all (REQ-4)
// ===========================================================================

/// Divergence: ferrolearn's `RFE::new` diverges from
/// `sklearn/feature_selection/_rfe.py:290-297` (the `warnings.warn(... "There
/// will be no feature selection and all features will be kept.")` branch) and
/// `:314` (the `while np.sum(support_) > n_features_to_select` loop, which never
/// runs when `n_features_to_select > n_features`).
///
/// Input: 2-feature importance vector, `n_features_to_select = 5` (> 2).
///
/// sklearn does NOT raise — it warns and KEEPS ALL features. Live oracle:
///   python3 -c "import warnings,numpy as np; \
///     from sklearn.feature_selection import RFE; \
///     from sklearn.linear_model import LinearRegression; \
///     X=np.array([[1.,10.],[2.,20.],[3.,30.],[4.,5.]]); y=np.array([1.,2.,3.,4.]); \
///     warnings.simplefilter('ignore'); \
///     s=RFE(LinearRegression(), n_features_to_select=5).fit(X,y); \
///     print(s.support_.tolist(), s.ranking_.tolist(), int(s.n_features_))"
///   -> support_ [True, True]  ranking_ [1, 1]  n_features_ 2   (ALL kept)
///
/// ferrolearn's `RFE::new(&imp_2feat, 5, 1)` instead returns
/// `Err(FerroError::InvalidParameter)` (`rfe.rs:105`).
///
/// Tracking: blocker #1296. (Do NOT file a new blocker — referenced per task.)
///
/// FIXER NOTE: remove the `n_features_to_select > n_features` arm from the error
/// condition and CLAMP `n_features_to_select = n_features_to_select.min(n_features)`
/// so the elimination loop is a no-op (all features kept). The UserWarning has no
/// Rust analog (no log facade) — keeping all features silently is the fix.
#[test]
fn divergence_rfe_n_features_to_select_gt_n_features_keeps_all() {
    // sklearn oracle (from /tmp, values above): all features kept.
    const SK_SUPPORT: [bool; 2] = [true, true];
    const SK_RANKING: [usize; 2] = [1, 1];
    const SK_N_FEATURES: usize = 2;

    let imp_2feat: Array1<f64> = array![0.5, 0.3];
    let rfe = RFE::<f64>::new(&imp_2feat, 5, 1).expect(
        "sklearn keeps all features (warns, does not raise) when n_features_to_select > n_features",
    );

    assert_eq!(rfe.support(), &SK_SUPPORT, "all features kept");
    assert_eq!(rfe.ranking(), &SK_RANKING, "all features rank 1");
    assert_eq!(rfe.n_features_selected(), SK_N_FEATURES);
}

// ===========================================================================
// GREEN GUARD — RFE static-importance ranking reproduces sklearn ranking_ (REQ-1)
//
// Fixture: X[:, j] = y * scale_j, scales = [100, 10, 1, 0.1]. With this design
// LinearRegression coef_[j] ~= 1/scale_j, so squared coefs are widely separated
// and STABLE across re-fits → dropping the weakest never reorders survivors.
//
// Live oracle (from /tmp):
//   y=[1..6]; scales=[100,10,1,0.1]; X=np.outer(y,scales)
//   lr=LinearRegression().fit(X,y)
//   (lr.coef_**2)            -> [9.80100019602e-05, 9.80100019602001e-07,
//                                9.801000196020004e-09, 9.801000196020007e-11]
//   RFE(LinearRegression(), n_features_to_select=1, step=1).fit(X,y)
//     ranking_  -> [1, 2, 3, 4]   support_ -> [True, False, False, False]
//   RFE(LinearRegression(), n_features_to_select=1, step=2).fit(X,y)
//     ranking_  -> [1, 2, 3, 3]   support_ -> [True, False, False, False]
// ===========================================================================

/// sklearn `LinearRegression().fit(X, y).coef_ ** 2` for the stable fixture
/// (the squared-coef importances sklearn ranks by, `_rfe.py:326-330`). These
/// are the importances we feed to ferrolearn — sourced from the sklearn oracle,
/// NOT the ferrolearn side (R-CHAR-3).
const SK_SQUARED_COEFS: [f64; 4] = [
    9.801_000_196_02e-05,
    9.801_000_196_020_01e-07,
    9.801_000_196_020_004e-09,
    9.801_000_196_020_007e-11,
];

/// Green guard (REQ-1, step=1): ferrolearn's static-importance ranking
/// reproduces sklearn's `RFE(...).ranking_` / `support_` when fed sklearn's
/// squared-coef importances on a stable fixture.
/// Mirrors `sklearn/feature_selection/_rfe.py:331,337,345-346`.
#[test]
fn green_rfe_ranking_matches_sklearn_step1() {
    // sklearn oracle (from /tmp).
    const SK_RANKING: [usize; 4] = [1, 2, 3, 4];
    const SK_SUPPORT: [bool; 4] = [true, false, false, false];

    let imp = Array1::from(SK_SQUARED_COEFS.to_vec());
    let rfe = RFE::<f64>::new(&imp, 1, 1).unwrap();
    assert_eq!(rfe.ranking(), &SK_RANKING);
    assert_eq!(rfe.support(), &SK_SUPPORT);
    assert_eq!(rfe.selected_indices(), &[0]);
}

/// Green guard (REQ-1, step=2): a multi-feature-per-round elimination reproduces
/// sklearn's `ranking_` (two features share the last-removed-round rank).
/// Mirrors `sklearn/feature_selection/_rfe.py:337` (`threshold = min(step, ...)`).
#[test]
fn green_rfe_ranking_matches_sklearn_step2() {
    // sklearn oracle (from /tmp): ranking_ [1, 2, 3, 3], support_ [T,F,F,F].
    const SK_RANKING: [usize; 4] = [1, 2, 3, 3];
    const SK_SUPPORT: [bool; 4] = [true, false, false, false];

    let imp = Array1::from(SK_SQUARED_COEFS.to_vec());
    let rfe = RFE::<f64>::new(&imp, 1, 2).unwrap();
    assert_eq!(rfe.ranking(), &SK_RANKING);
    assert_eq!(rfe.support(), &SK_SUPPORT);
}

// ===========================================================================
// GREEN GUARD — RFECV optimal-count selection (REQ-9)
//
// sklearn picks `step_n_features_rev[np.argmax(scores_sum_rev)]`
// (`_rfe.py:786-788`); the `[::-1]` reversal makes argmax resolve ties to the
// LOWEST feature count. With cv_scores indexed ascending by count (index 0 = 1
// feature, ferrolearn's convention), this equals first-max-on-ascending-count.
//
// Live oracle (from /tmp, faithful ascending-count simulation of :786-788):
//   [0.85, 0.95, 0.90] -> count 2
//   [0.85, 0.95, 0.95] -> count 2   (tie resolves to LOWER count)
//   [0.9, 0.8]         -> count 1
// ===========================================================================

/// Green guard (REQ-9): clear-max CV scores select the argmax count.
/// Mirrors `sklearn/feature_selection/_rfe.py:786-788`.
#[test]
fn green_rfecv_optimal_count_clear_max() {
    // sklearn oracle (from /tmp): [0.85,0.95,0.90] -> count 2.
    const SK_OPTIMAL: usize = 2;

    let imp = array![0.5, 0.3, 0.2];
    let cv_scores = vec![0.85, 0.95, 0.90];
    let rfecv = RFECV::<f64>::new(&imp, &cv_scores, 1).unwrap();
    assert_eq!(rfecv.optimal_n_features(), SK_OPTIMAL);
    assert_eq!(rfecv.n_features_selected(), SK_OPTIMAL);
}

/// Green guard (REQ-9): a trailing tie resolves to the LOWER feature count,
/// matching sklearn's reversed-argmax tie-break (`_rfe.py:786-788`).
#[test]
fn green_rfecv_optimal_count_tie_picks_lower() {
    // sklearn oracle (from /tmp): [0.85,0.95,0.95] -> count 2 (lower of the tie).
    const SK_OPTIMAL: usize = 2;

    let imp = array![0.5, 0.3, 0.2];
    let cv_scores = vec![0.85, 0.95, 0.95];
    let rfecv = RFECV::<f64>::new(&imp, &cv_scores, 1).unwrap();
    assert_eq!(rfecv.optimal_n_features(), SK_OPTIMAL);
    assert_eq!(rfecv.n_features_selected(), SK_OPTIMAL);
}

// ===========================================================================
// GREEN GUARD — scoped error contracts (REQ-2)
// ===========================================================================

/// Green guard (REQ-2): empty importances -> Err. Analog of sklearn
/// `_validate_data(ensure_min_features=2, ...)` (`_rfe.py:275-282`).
#[test]
fn green_rfe_empty_importances_err() {
    let imp: Array1<f64> = Array1::zeros(0);
    assert!(RFE::<f64>::new(&imp, 1, 1).is_err());
}

/// Green guard (REQ-2): step == 0 -> Err. Analog of sklearn's `step`
/// `_parameter_constraints` (`Interval(..., closed="neither")`, `_rfe.py:208`).
#[test]
fn green_rfe_zero_step_err() {
    let imp = array![0.5, 0.3];
    assert!(RFE::<f64>::new(&imp, 1, 0).is_err());
}

/// Green guard (REQ-2): n_features_to_select == 0 -> Err. Analog of sklearn's
/// `n_features_to_select` constraint (`Interval(..., closed="neither")`,
/// `_rfe.py:204`).
#[test]
fn green_rfe_zero_n_features_err() {
    let imp = array![0.5, 0.3];
    assert!(RFE::<f64>::new(&imp, 0, 1).is_err());
}

/// Green guard (REQ-2): transform column-count mismatch -> Err (ShapeMismatch).
#[test]
fn green_rfe_transform_shape_mismatch_err() {
    let imp = array![0.5, 0.3];
    let rfe = RFE::<f64>::new(&imp, 1, 1).unwrap();
    let x_bad = array![[1.0, 2.0, 3.0]];
    assert!(rfe.transform(&x_bad).is_err());
}

/// Green guard (REQ-2): RFECV cv_scores length mismatch -> Err.
#[test]
fn green_rfecv_cv_scores_length_mismatch_err() {
    let imp = array![0.5, 0.3, 0.2];
    let cv_scores = vec![0.85, 0.95]; // wrong length
    assert!(RFECV::<f64>::new(&imp, &cv_scores, 1).is_err());
}

// ===========================================================================
// GREEN GUARD — clamp boundary (RE-AUDIT of blocker #1296 fix, REQ-4)
//
// The fix removed the `n_features_to_select > n_features` error arm and replaced
// it with `n_features_to_select = n_features_to_select.min(n_features)` (clamp)
// before the elimination loop (`rfe.rs:117`), mirroring sklearn's keep-all
// behavior (`sklearn/feature_selection/_rfe.py:290-297` warn-only branch; `:314`
// loop never runs). These guards stress the novel clamp control flow against the
// LIVE sklearn 1.5.2 oracle (`warnings.simplefilter('ignore')`, run from /tmp).
//
// sklearn LinearRegression squared coefs for the stable keep-all fixtures
// `X[:,j] = y*scale_j`, y=[1..6] (R-CHAR-3: these are the oracle's coef_**2,
// NOT copied from ferrolearn):
//   scales=[100,10,1]   -> [9.801019602029405e-05, 9.801019602029405e-07,
//                           9.8010196020294e-09]
//   scales=[100,10,1,.1]-> [9.80100019602e-05, 9.80100019602001e-07,
//                           9.801000196020004e-09, 9.801000196020007e-11]
// ===========================================================================

/// sklearn `coef_**2` for the 3-feature stable fixture (scales [100,10,1]).
/// Sourced from the live oracle, NOT the ferrolearn side (R-CHAR-3).
const SK_SQUARED_COEFS_3: [f64; 3] = [
    9.801_019_602_029_405e-05,
    9.801_019_602_029_405e-07,
    9.801_019_602_029_4e-09,
];

/// Green guard (REQ-4): `n_features_to_select == n_features` exactly (3 of 3).
/// sklearn's `while np.sum(support_) > n_features_to_select` never runs, so all
/// features are kept (`_rfe.py:314`). Confirms the clamp fix did NOT perturb the
/// already-allowed exact-equality path.
/// Live oracle (from /tmp): support [True,True,True], ranking [1,1,1], n 3.
#[test]
fn green_rfe_clamp_n_equals_n_features_keeps_all() {
    const SK_SUPPORT: [bool; 3] = [true, true, true];
    const SK_RANKING: [usize; 3] = [1, 1, 1];

    let imp = Array1::from(SK_SQUARED_COEFS_3.to_vec());
    let rfe = RFE::<f64>::new(&imp, 3, 1).expect("n_features_to_select == n_features must be Ok");
    assert_eq!(rfe.support(), &SK_SUPPORT, "all 3 features kept");
    assert_eq!(rfe.ranking(), &SK_RANKING, "all rank 1");
    assert_eq!(rfe.n_features_selected(), 3);
}

/// Green guard (REQ-4): `n_features_to_select == n_features + 1` (4 of 3) — just
/// over the boundary. sklearn warns and keeps all (`_rfe.py:290-297`); ferrolearn
/// clamps to 3 so the loop is a no-op.
/// Live oracle (from /tmp): support [True,True,True], ranking [1,1,1], n 3.
#[test]
fn green_rfe_clamp_n_just_over_keeps_all() {
    const SK_SUPPORT: [bool; 3] = [true, true, true];
    const SK_RANKING: [usize; 3] = [1, 1, 1];

    let imp = Array1::from(SK_SQUARED_COEFS_3.to_vec());
    let rfe = RFE::<f64>::new(&imp, 4, 1).expect("n_features_to_select == n+1 must keep all (Ok)");
    assert_eq!(rfe.support(), &SK_SUPPORT);
    assert_eq!(rfe.ranking(), &SK_RANKING);
    assert_eq!(rfe.n_features_selected(), 3);
}

/// Green guard (REQ-4): a much larger `n_features_to_select` (100 of 4). sklearn
/// warns and keeps all (`_rfe.py:290-297`); ferrolearn clamps to 4.
/// Live oracle (from /tmp): support [True,True,True,True], ranking [1,1,1,1], n 4.
#[test]
fn green_rfe_clamp_n_far_over_keeps_all() {
    const SK_SUPPORT: [bool; 4] = [true, true, true, true];
    const SK_RANKING: [usize; 4] = [1, 1, 1, 1];

    let imp = Array1::from(SK_SQUARED_COEFS.to_vec());
    let rfe = RFE::<f64>::new(&imp, 100, 1).expect("n_features_to_select=100 must keep all (Ok)");
    assert_eq!(rfe.support(), &SK_SUPPORT);
    assert_eq!(rfe.ranking(), &SK_RANKING);
    assert_eq!(rfe.n_features_selected(), 4);
}

/// Green guard (REQ-4 / REQ-1): a VALID `n_features_to_select < n_features` (2 of
/// 4) — confirms the clamp did NOT alter normal elimination. ferrolearn is fed
/// sklearn's `coef_**2` and must reproduce sklearn's `RFE(...).ranking_`.
/// Mirrors `sklearn/feature_selection/_rfe.py:331,337,345-346`.
/// Live oracle (from /tmp): support [True,True,False,False], ranking [1,1,2,3].
#[test]
fn green_rfe_clamp_does_not_perturb_valid_count() {
    const SK_SUPPORT: [bool; 4] = [true, true, false, false];
    const SK_RANKING: [usize; 4] = [1, 1, 2, 3];

    let imp = Array1::from(SK_SQUARED_COEFS.to_vec());
    let rfe = RFE::<f64>::new(&imp, 2, 1).unwrap();
    assert_eq!(rfe.support(), &SK_SUPPORT, "weakest two eliminated");
    assert_eq!(
        rfe.ranking(),
        &SK_RANKING,
        "valid-count ranking unchanged by clamp"
    );
    assert_eq!(rfe.selected_indices(), &[0, 1]);
}

/// Green guard (REQ-2): `n_features_to_select == 0` still errors — the clamp must
/// not swallow the zero-check. sklearn rejects 0 via the
/// `Interval(Integral, 1, None, closed="left")` constraint (`_rfe.py:204`), so 0
/// is never a valid keep-all input.
#[test]
fn green_rfe_clamp_does_not_swallow_zero_check() {
    let imp = array![0.5, 0.3, 0.2, 0.1];
    assert!(
        RFE::<f64>::new(&imp, 0, 1).is_err(),
        "n_features_to_select == 0 must still error after the clamp fix"
    );
}
