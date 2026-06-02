//! Critic divergence pins: ferrolearn-tree `DecisionTreeClassifier` /
//! `DecisionTreeRegressor` (CART) vs the LIVE scikit-learn 1.5.2 oracle.
//!
//! All expected values were derived by live-calling sklearn 1.5.2 (the installed
//! oracle == the `/home/doll/scikit-learn` source clone at tag 1.5.2). The exact
//! python invocations are recorded above each pin (R-CHAR-3 — no value is
//! literal-copied from the ferrolearn side).
//!
//! Introspection: `FittedDecisionTree*::nodes()` returns `&[Node<F>]` and the
//! `Node::Split { feature, threshold, .. }` fields are public, so the tree
//! structure (node_count, root split feature+threshold) IS observable — there is
//! NO missing-accessor blocker for this estimator.
//!
//! FINDING SUMMARY (see per-test docs):
//!   * The four design-doc "oracle dataset" pins (feature_importances_, tree
//!     structure, predict/predict_proba, regressor) all PASS — ferrolearn
//!     matches sklearn on those specific inputs. They are retained as
//!     oracle-grounded characterization tests (NOT divergences).
//!   * Two RED pins expose genuine algorithmic divergences on adversarial inputs:
//!       - `divergence_clf_tiebreak_random_state` — missing RNG feature-order
//!         tie-break (`random_state`-permuted visitation, _splitter.pyx).
//!       - `divergence_clf_feature_threshold_band` — missing
//!         `FEATURE_THRESHOLD = 1e-7` constant-feature band (_splitter.pyx:33).

use ferrolearn_core::introspection::HasFeatureImportances;
use ferrolearn_core::traits::{Fit, Predict};
use ferrolearn_tree::{DecisionTreeClassifier, DecisionTreeRegressor, Node};
use ndarray::{Array1, Array2, array};

/// Build the design-doc oracle classifier dataset.
fn clf_dataset() -> (Array2<f64>, Array1<usize>) {
    let x = Array2::from_shape_vec(
        (9, 2),
        vec![
            1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 5.0, 6.0, 6.0, 7.0, 7.0, 8.0, 1.5, 5.0, 6.5, 2.0, 3.0,
            1.0,
        ],
    )
    .unwrap();
    let y = array![0usize, 0, 0, 1, 1, 1, 2, 2, 0];
    (x, y)
}

/// Build the design-doc oracle regressor dataset.
fn reg_dataset() -> (Array2<f64>, Array1<f64>) {
    let x = Array2::from_shape_vec((8, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
    let y = array![1.0, 1.2, 0.9, 1.1, 5.0, 5.2, 4.9, 5.1];
    (x, y)
}

/// Locate the root node; return `Some((feature, threshold))` if it is a Split,
/// `None` if the root is a Leaf (i.e. the tree did not split at all).
fn root_split(nodes: &[Node<f64>]) -> Option<(usize, f64)> {
    match nodes.first() {
        Some(Node::Split {
            feature, threshold, ..
        }) => Some((*feature, *threshold)),
        _ => None,
    }
}

// ===========================================================================
// GREEN — oracle-grounded characterization (ferrolearn matches sklearn here).
// Retained per R-CHAR-1 as live-oracle-derived assertions; they are NOT
// divergences (they pass against the current implementation).
// ===========================================================================

/// Oracle (sklearn 1.5.2):
///   python3 -c "import numpy as np; from sklearn.tree import DecisionTreeClassifier; \
///     X=np.array([[1,2],[2,3],[3,3],[5,6],[6,7],[7,8],[1.5,5],[6.5,2],[3,1]],dtype=float); \
///     y=np.array([0,0,0,1,1,1,2,2,0]); \
///     print(DecisionTreeClassifier(random_state=0).fit(X,y).feature_importances_.tolist())"
///   => [0.18461538461538465, 0.8153846153846153]   (`sklearn/tree/_classes.py:671`)
#[test]
fn clf_feature_importances_oracle() {
    const SK_IMPORTANCE_0: f64 = 0.18461538461538465;
    const SK_IMPORTANCE_1: f64 = 0.8153846153846153;

    let (x, y) = clf_dataset();
    let fitted = DecisionTreeClassifier::<f64>::new().fit(&x, &y).unwrap();
    let imp = fitted.feature_importances();

    assert_eq!(imp.len(), 2);
    assert!(
        (imp[0] - SK_IMPORTANCE_0).abs() < 1e-12,
        "feature_importances_[0]: ferrolearn={} sklearn={}",
        imp[0],
        SK_IMPORTANCE_0
    );
    assert!(
        (imp[1] - SK_IMPORTANCE_1).abs() < 1e-12,
        "feature_importances_[1]: ferrolearn={} sklearn={}",
        imp[1],
        SK_IMPORTANCE_1
    );
}

/// Oracle (sklearn 1.5.2): `c.tree_.node_count => 7`,
/// `c.tree_.feature[0] => 1`, `c.tree_.threshold[0] => 5.5`.
#[test]
fn clf_tree_structure_oracle() {
    let (x, y) = clf_dataset();
    let fitted = DecisionTreeClassifier::<f64>::new().fit(&x, &y).unwrap();
    let nodes = fitted.nodes();

    assert_eq!(nodes.len(), 7, "node_count: ferrolearn={}", nodes.len());
    let (feat, thr) = root_split(nodes).expect("root must be a Split");
    assert_eq!(feat, 1, "root feature: ferrolearn={feat}");
    assert!((thr - 5.5).abs() < 1e-9, "root threshold: ferrolearn={thr}");
}

/// Oracle (sklearn 1.5.2): `c.predict(X) => [0,0,0,1,1,1,2,2,0]`;
/// `c.predict_proba(X)` one-hot per leaf (every leaf is pure on this set).
#[test]
fn clf_predict_and_proba_oracle() {
    let sk_predict: [usize; 9] = [0, 0, 0, 1, 1, 1, 2, 2, 0];
    let sk_proba: [[f64; 3]; 9] = [
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
    ];

    let (x, y) = clf_dataset();
    let fitted = DecisionTreeClassifier::<f64>::new().fit(&x, &y).unwrap();

    let preds = fitted.predict(&x).unwrap();
    for (i, &exp) in sk_predict.iter().enumerate() {
        assert_eq!(preds[i], exp, "predict[{i}]: ferrolearn={}", preds[i]);
    }
    let proba = fitted.predict_proba(&x).unwrap();
    for (i, row) in sk_proba.iter().enumerate() {
        for (j, &exp) in row.iter().enumerate() {
            assert!(
                (proba[[i, j]] - exp).abs() < 1e-9,
                "predict_proba[{i}][{j}]: ferrolearn={}",
                proba[[i, j]]
            );
        }
    }
}

/// Oracle (sklearn 1.5.2): `r.tree_.node_count => 15`, root `(feature=0,
/// threshold=4.5)`, `r.predict(X) => y`.
#[test]
fn reg_tree_structure_and_predict_oracle() {
    let sk_predict = [1.0, 1.2, 0.9, 1.1, 5.0, 5.2, 4.9, 5.1];

    let (x, y) = reg_dataset();
    let fitted = DecisionTreeRegressor::<f64>::new().fit(&x, &y).unwrap();
    let nodes = fitted.nodes();

    assert_eq!(
        nodes.len(),
        15,
        "REG node_count: ferrolearn={}",
        nodes.len()
    );
    let (feat, thr) = root_split(nodes).expect("REG root must be a Split");
    assert_eq!(feat, 0, "REG root feature: ferrolearn={feat}");
    assert!(
        (thr - 4.5).abs() < 1e-9,
        "REG root threshold: ferrolearn={thr}"
    );

    let preds = fitted.predict(&x).unwrap();
    for (i, &exp) in sk_predict.iter().enumerate() {
        assert!(
            (preds[i] - exp).abs() < 1e-9,
            "REG predict[{i}]: ferrolearn={}",
            preds[i]
        );
    }
}

// ===========================================================================
// RED — genuine divergences (these FAIL against the current implementation).
// ===========================================================================

/// DOCUMENTED RNG BOUNDARY (not a fixable divergence): on an exact
/// improvement-TIE between features, the chosen split feature is
/// `random_state`-dependent in sklearn. `node_split_best`
/// (`sklearn/tree/_splitter.pyx:293`) visits features in a `random_state`-permuted
/// (Fisher-Yates) order and keeps a `>=` reservoir split, so the winning feature
/// among equal-improvement candidates depends on the numpy RNG (with
/// `random_state=None` sklearn is itself non-deterministic on such ties).
/// ferrolearn scans `0..n_features` deterministically and keeps the first
/// max-improvement feature. Matching sklearn's exact feature on a measure-zero
/// improvement-tie would require replicating numpy's RNG permutation — infeasible
/// (the same RNG boundary documented for SGD shuffle / libsvm CV folds). The
/// random_state PARAMETER (which also drives `max_features` subsampling) is a
/// separate NOT-STARTED feature.
///
/// What IS deterministic and pinnable here is the OBSERVABLE contract: BOTH
/// features perfectly separate the classes, so `predict` is identical regardless
/// of which feature the root splits on — and it matches sklearn's `predict`.
/// Oracle (sklearn 1.5.2, random_state-invariant on this set):
///   python3 -c "import numpy as np; from sklearn.tree import DecisionTreeClassifier; \
///     X=np.array([[0,0],[0,0],[1,1],[1,1]],dtype=float); y=np.array([0,0,1,1]); \
///     print([DecisionTreeClassifier(random_state=s).fit(X,y).predict(X).tolist() for s in (0,1,2)])"
///   => [[0,0,1,1],[0,0,1,1],[0,0,1,1]]   (predict invariant; root feature varies with seed)
#[test]
fn clf_tiebreak_predict_invariant_rng_boundary() {
    let x = Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]).unwrap();
    let y = array![0usize, 0, 1, 1];
    let fitted = DecisionTreeClassifier::<f64>::new().fit(&x, &y).unwrap();

    // The root split is deterministic in ferrolearn (some equal-improvement
    // feature); the EXACT feature index is the documented random_state boundary,
    // so we do NOT pin it. We pin the observable predict, which is
    // random_state-invariant on this set and matches sklearn.
    let (feat, _thr) = root_split(fitted.nodes()).expect("root must be a Split");
    assert!(feat < 2, "root splits on a valid feature: {feat}");

    let sk_predict = [0usize, 0, 1, 1];
    let preds = fitted.predict(&x).unwrap();
    for (i, &exp) in sk_predict.iter().enumerate() {
        assert_eq!(
            preds[i], exp,
            "predict[{i}] (random_state-invariant): ferrolearn={}",
            preds[i]
        );
    }
}

/// Divergence: ferrolearn's split-finder skips a split point only on EXACT
/// equality (`x[idx] == x[next]` in `find_best_classification_split`,
/// decision_tree.rs) and has no constant-feature guard, so it will split on a
/// feature whose total spread is below 1e-7.
///
/// sklearn treats a feature as constant when
/// `feature_values[end-1] <= feature_values[start] + FEATURE_THRESHOLD` with
/// `FEATURE_THRESHOLD = 1e-7` (`sklearn/tree/_splitter.pyx:33`, `:404`), so a
/// feature varying by < 1e-7 is never split on.
///
/// On this set feature 0 separates the classes but spans only 3e-8 (< 1e-7) and
/// feature 1 is exactly constant, so sklearn cannot split at all → a single-leaf
/// root predicting the majority class (0).
/// Oracle (sklearn 1.5.2):
///   python3 -c "import numpy as np; from sklearn.tree import DecisionTreeClassifier; \
///     X=np.array([[1.0,5.0],[1.00000001,5.0],[1.00000002,5.0],[1.00000003,5.0]]); \
///     y=np.array([0,0,1,1]); c=DecisionTreeClassifier(random_state=0).fit(X,y); \
///     print(c.tree_.node_count, c.predict(X).tolist())"
///   => 1 [0, 0, 0, 0]      (ferrolearn: node_count 3, predict [0,0,1,1])
#[test]
fn divergence_clf_feature_threshold_band() {
    const SK_NODE_COUNT: usize = 1;
    let sk_predict: [usize; 4] = [0, 0, 0, 0];

    let x = Array2::from_shape_vec(
        (4, 2),
        vec![
            1.0,
            5.0,
            1.000_000_01,
            5.0,
            1.000_000_02,
            5.0,
            1.000_000_03,
            5.0,
        ],
    )
    .unwrap();
    let y = array![0usize, 0, 1, 1];
    let fitted = DecisionTreeClassifier::<f64>::new().fit(&x, &y).unwrap();

    assert_eq!(
        fitted.nodes().len(),
        SK_NODE_COUNT,
        "FEATURE_THRESHOLD node_count: ferrolearn={} sklearn={SK_NODE_COUNT}",
        fitted.nodes().len()
    );

    let preds = fitted.predict(&x).unwrap();
    for (i, &exp) in sk_predict.iter().enumerate() {
        assert_eq!(
            preds[i], exp,
            "FEATURE_THRESHOLD predict[{i}]: ferrolearn={} sklearn={exp}",
            preds[i]
        );
    }
}

// ===========================================================================
// REQ-1 ALT-CRITERIA CERTIFYING PINS (#661 / iter REQ-1).
// log_loss alias, friedman_mse, absolute_error (median leaves), poisson.
// All expected values derived LIVE from the sklearn 1.5.2 oracle (R-CHAR-3);
// the exact python invocations are recorded above each pin. Tolerance 1e-9.
// ===========================================================================

use ferrolearn_tree::{ClassificationCriterion, RegressionCriterion};

/// PIN A — `log_loss` is the `entropy` alias (classification).
///
/// Oracle (sklearn 1.5.2):
///   python3 -c "import numpy as np; from sklearn.tree import DecisionTreeClassifier; \
///     X=np.array([[1,2],[2,3],[3,3],[5,6],[6,7],[7,8],[1.5,5],[6.5,2],[3,1]],dtype=float); \
///     y=np.array([0,0,0,1,1,1,2,2,0]); \
///     c=DecisionTreeClassifier(criterion='log_loss',random_state=0).fit(X,y); \
///     print(c.tree_.feature[0], c.tree_.threshold[0], \
///           c.feature_importances_.tolist(), c.predict(X).tolist())"
///   => root (1, 5.5);
///      feature_importances_ [0.13794643363098585, 0.8620535663690142];
///      predict [0,0,0,1,1,1,2,2,0]
///   and `criterion='entropy'` produces the IDENTICAL fi + predict (alias).
#[test]
fn req1_clf_log_loss_is_entropy_alias_oracle() {
    const SK_LOGLOSS_FI_0: f64 = 0.13794643363098585;
    const SK_LOGLOSS_FI_1: f64 = 0.8620535663690142;
    let sk_predict: [usize; 9] = [0, 0, 0, 1, 1, 1, 2, 2, 0];

    let (x, y) = clf_dataset();

    // LogLoss criterion vs the oracle absolute values.
    let log_loss = DecisionTreeClassifier::<f64>::new()
        .with_criterion(ClassificationCriterion::LogLoss)
        .fit(&x, &y)
        .unwrap();

    let (feat, thr) = root_split(log_loss.nodes()).expect("LogLoss root must be a Split");
    assert_eq!(feat, 1, "LogLoss root feature: ferrolearn={feat} sklearn=1");
    assert!(
        (thr - 5.5).abs() < 1e-9,
        "LogLoss root threshold: ferrolearn={thr} sklearn=5.5"
    );

    let ll_imp = log_loss.feature_importances();
    assert!(
        (ll_imp[0] - SK_LOGLOSS_FI_0).abs() < 1e-9,
        "LogLoss fi[0]: ferrolearn={} sklearn={SK_LOGLOSS_FI_0}",
        ll_imp[0]
    );
    assert!(
        (ll_imp[1] - SK_LOGLOSS_FI_1).abs() < 1e-9,
        "LogLoss fi[1]: ferrolearn={} sklearn={SK_LOGLOSS_FI_1}",
        ll_imp[1]
    );

    let ll_preds = log_loss.predict(&x).unwrap();
    for (i, &exp) in sk_predict.iter().enumerate() {
        assert_eq!(
            ll_preds[i], exp,
            "LogLoss predict[{i}]: ferrolearn={} sklearn={exp}",
            ll_preds[i]
        );
    }

    // Alias invariant: LogLoss == Entropy, feature-importances + predict identical.
    let entropy = DecisionTreeClassifier::<f64>::new()
        .with_criterion(ClassificationCriterion::Entropy)
        .fit(&x, &y)
        .unwrap();
    let en_imp = entropy.feature_importances();
    assert!(
        (ll_imp[0] - en_imp[0]).abs() < 1e-12 && (ll_imp[1] - en_imp[1]).abs() < 1e-12,
        "LogLoss/Entropy fi must be identical: log_loss=[{}, {}] entropy=[{}, {}]",
        ll_imp[0],
        ll_imp[1],
        en_imp[0],
        en_imp[1]
    );
    let en_preds = entropy.predict(&x).unwrap();
    assert_eq!(
        ll_preds, en_preds,
        "LogLoss/Entropy predict must be identical (alias invariant)"
    );
}

/// PIN B — `friedman_mse` (regression, mean leaves).
///
/// Oracle (sklearn 1.5.2):
///   python3 -c "import numpy as np; from sklearn.tree import DecisionTreeRegressor; \
///     Xr=np.array([[1],[2],[3],[4],[5],[6],[7],[8]],dtype=float); \
///     yr=np.array([1,1.2,0.9,1.1,5,5.2,4.9,5.1]); \
///     r=DecisionTreeRegressor(criterion='friedman_mse',max_depth=2,random_state=0).fit(Xr,yr); \
///     print(r.tree_.feature[0], r.tree_.threshold[0], r.predict(Xr).tolist())"
///   => root (0, 4.5); predict [1.1,1.1,1.0,1.0,5.1,5.1,5.0,5.0]  (MEAN leaves)
#[test]
fn req1_reg_friedman_mse_oracle() {
    let sk_predict = [1.1, 1.1, 1.0, 1.0, 5.1, 5.1, 5.0, 5.0];

    let (x, y) = reg_dataset();
    let fitted = DecisionTreeRegressor::<f64>::new()
        .with_criterion(RegressionCriterion::FriedmanMse)
        .with_max_depth(Some(2))
        .fit(&x, &y)
        .unwrap();

    let (feat, thr) = root_split(fitted.nodes()).expect("friedman_mse root must be a Split");
    assert_eq!(
        feat, 0,
        "friedman_mse root feature: ferrolearn={feat} sklearn=0"
    );
    assert!(
        (thr - 4.5).abs() < 1e-9,
        "friedman_mse root threshold: ferrolearn={thr} sklearn=4.5"
    );

    let preds = fitted.predict(&x).unwrap();
    for (i, &exp) in sk_predict.iter().enumerate() {
        assert!(
            (preds[i] - exp).abs() < 1e-9,
            "friedman_mse predict[{i}] (mean leaf): ferrolearn={} sklearn={exp}",
            preds[i]
        );
    }
}

/// PIN B' — friedman_mse's improvement formula picks a DIFFERENT split than
/// squared_error on this set, proving the Friedman proxy is actually distinct
/// (not a no-op that coincides on every input).
///
/// Oracle (sklearn 1.5.2):
///   python3 -c "import numpy as np; from sklearn.tree import DecisionTreeRegressor; \
///     X=np.arange(5,dtype=float).reshape(-1,1); yr=np.array([0.9,6.3,6.0,3.7,0.9]); \
///     rf=DecisionTreeRegressor(criterion='friedman_mse',max_depth=1,random_state=0).fit(X,yr); \
///     rs=DecisionTreeRegressor(criterion='squared_error',max_depth=1,random_state=0).fit(X,yr); \
///     print(rf.tree_.threshold[0], rf.predict(X).tolist()); \
///     print(rs.tree_.threshold[0], rs.predict(X).tolist())"
///   => friedman_mse root thr 3.5, predict [4.225,4.225,4.225,4.225,0.9]
///      squared_error root thr 0.5, predict [0.9,4.225,4.225,4.225,4.225]
#[test]
fn req1_reg_friedman_mse_differs_from_squared_error() {
    const SK_FRIEDMAN_THR: f64 = 3.5;
    const SK_SQUARED_THR: f64 = 0.5;
    let sk_friedman_predict = [4.225, 4.225, 4.225, 4.225, 0.9];
    let sk_squared_predict = [0.9, 4.225, 4.225, 4.225, 4.225];

    let x = Array2::from_shape_vec((5, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0]).unwrap();
    let y = array![0.9, 6.3, 6.0, 3.7, 0.9];

    let friedman = DecisionTreeRegressor::<f64>::new()
        .with_criterion(RegressionCriterion::FriedmanMse)
        .with_max_depth(Some(1))
        .fit(&x, &y)
        .unwrap();
    let squared = DecisionTreeRegressor::<f64>::new()
        .with_criterion(RegressionCriterion::Mse)
        .with_max_depth(Some(1))
        .fit(&x, &y)
        .unwrap();

    let (_ff, ft) = root_split(friedman.nodes()).expect("friedman root must be a Split");
    let (_sf, st) = root_split(squared.nodes()).expect("squared root must be a Split");
    assert!(
        (ft - SK_FRIEDMAN_THR).abs() < 1e-9,
        "friedman_mse root threshold: ferrolearn={ft} sklearn={SK_FRIEDMAN_THR}"
    );
    assert!(
        (st - SK_SQUARED_THR).abs() < 1e-9,
        "squared_error root threshold: ferrolearn={st} sklearn={SK_SQUARED_THR}"
    );
    // The two criteria must DISAGREE on the split (proof the proxy differs).
    assert!(
        (ft - st).abs() > 1e-9,
        "friedman_mse and squared_error must pick DIFFERENT splits here: \
         friedman={ft} squared={st}"
    );

    let fpred = friedman.predict(&x).unwrap();
    let spred = squared.predict(&x).unwrap();
    for (i, &exp) in sk_friedman_predict.iter().enumerate() {
        assert!(
            (fpred[i] - exp).abs() < 1e-9,
            "friedman predict[{i}]: ferrolearn={} sklearn={exp}",
            fpred[i]
        );
    }
    for (i, &exp) in sk_squared_predict.iter().enumerate() {
        assert!(
            (spred[i] - exp).abs() < 1e-9,
            "squared predict[{i}]: ferrolearn={} sklearn={exp}",
            spred[i]
        );
    }
}

/// PIN C — `absolute_error` (regression, MEDIAN leaves).
///
/// Oracle (sklearn 1.5.2):
///   python3 -c "import numpy as np; from sklearn.tree import DecisionTreeRegressor; \
///     Xr=np.array([[1],[2],[3],[4],[5],[6],[7],[8]],dtype=float); \
///     yr=np.array([1,1.2,0.9,1.1,5,5.2,4.9,5.1]); \
///     r=DecisionTreeRegressor(criterion='absolute_error',max_depth=2,random_state=0).fit(Xr,yr); \
///     print(r.predict(Xr).tolist())"
///   => predict [1.0,1.1,1.1,1.1,5.0,5.1,5.1,5.1]  (MEDIAN leaves)
/// The DISTINCTNESS from PIN B's mean-leaf predict is the proof of the
/// median-leaf path (`MAE.node_value`, `_criterion.pyx:1419-1423`).
#[test]
fn req1_reg_absolute_error_median_leaves_oracle() {
    let sk_predict = [1.0, 1.1, 1.1, 1.1, 5.0, 5.1, 5.1, 5.1];
    // PIN B's friedman_mse (mean-leaf) predictions on the SAME set — must differ.
    let friedman_mean_predict = [1.1, 1.1, 1.0, 1.0, 5.1, 5.1, 5.0, 5.0];

    let (x, y) = reg_dataset();
    let fitted = DecisionTreeRegressor::<f64>::new()
        .with_criterion(RegressionCriterion::AbsoluteError)
        .with_max_depth(Some(2))
        .fit(&x, &y)
        .unwrap();

    let preds = fitted.predict(&x).unwrap();
    for (i, &exp) in sk_predict.iter().enumerate() {
        assert!(
            (preds[i] - exp).abs() < 1e-9,
            "absolute_error predict[{i}] (median leaf): ferrolearn={} sklearn={exp}",
            preds[i]
        );
    }

    // Median leaves are DISTINCT from friedman_mse's mean leaves on this set.
    let any_diff = sk_predict
        .iter()
        .zip(friedman_mean_predict.iter())
        .any(|(a, b)| (a - b).abs() > 1e-9);
    assert!(
        any_diff,
        "median-leaf predict must differ from mean-leaf predict (proof of MEDIAN path)"
    );
}

/// PIN D — `poisson` (regression).
///
/// Oracle (sklearn 1.5.2):
///   python3 -c "import numpy as np; from sklearn.tree import DecisionTreeRegressor; \
///     Xr=np.array([[1],[2],[3],[4],[5],[6],[7],[8]],dtype=float); \
///     yr=np.array([1,1.2,0.9,1.1,5,5.2,4.9,5.1]); \
///     r=DecisionTreeRegressor(criterion='poisson',max_depth=2,random_state=0).fit(Xr,yr); \
///     print(r.tree_.feature[0], r.tree_.threshold[0], r.predict(Xr).tolist())"
///   => root (0, 4.5); predict [1.1,1.1,1.0,1.0,5.1,5.1,5.0,5.0]  (MEAN leaves)
/// AND a set with a y <= 0 must Err (sklearn raises ValueError "Some value(s)
/// of y are negative which is not allowed for Poisson regression.",
/// `_classes.py:267-277`).
#[test]
fn req1_reg_poisson_oracle_and_negative_y_errors() {
    let sk_predict = [1.1, 1.1, 1.0, 1.0, 5.1, 5.1, 5.0, 5.0];

    let (x, y) = reg_dataset();
    let fitted = DecisionTreeRegressor::<f64>::new()
        .with_criterion(RegressionCriterion::Poisson)
        .with_max_depth(Some(2))
        .fit(&x, &y)
        .unwrap();

    let (feat, thr) = root_split(fitted.nodes()).expect("poisson root must be a Split");
    assert_eq!(feat, 0, "poisson root feature: ferrolearn={feat} sklearn=0");
    assert!(
        (thr - 4.5).abs() < 1e-9,
        "poisson root threshold: ferrolearn={thr} sklearn=4.5"
    );

    let preds = fitted.predict(&x).unwrap();
    for (i, &exp) in sk_predict.iter().enumerate() {
        assert!(
            (preds[i] - exp).abs() < 1e-9,
            "poisson predict[{i}] (mean leaf): ferrolearn={} sklearn={exp}",
            preds[i]
        );
    }

    // A set with a negative y must be rejected (sklearn raises ValueError).
    let x_neg = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let y_neg = array![1.0, -0.5, 2.0, 3.0];
    let res = DecisionTreeRegressor::<f64>::new()
        .with_criterion(RegressionCriterion::Poisson)
        .with_max_depth(Some(2))
        .fit(&x_neg, &y_neg);
    assert!(
        res.is_err(),
        "poisson fit on a set with negative y must return Err (sklearn raises ValueError)"
    );
}
