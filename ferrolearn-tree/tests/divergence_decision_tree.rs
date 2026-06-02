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

// ===========================================================================
// REQ-3 STOPPING-PARAM CERTIFYING PINS (#662 min_impurity_decrease / #664
// min_weight_fraction_leaf). All expected values derived LIVE from the
// sklearn 1.5.2 oracle (R-CHAR-3); the exact python invocations are recorded
// above each pin. node_count = `tree_.node_count`; predict = `predict(X)`.
// Tests run in integration-test (test) context — the .unwrap()/.expect() idiom
// matches the 11 prior pins above (R-APG-2: test context is gate-exempt).
// ===========================================================================

/// PIN A — `min_impurity_decrease` gates the CLASSIFIER tree (#662).
///
/// Oracle (sklearn 1.5.2):
///   python3 -c "import numpy as np; from sklearn.tree import DecisionTreeClassifier; \
///     X=np.array([[1,2],[2,3],[3,3],[5,6],[6,7],[7,8],[1.5,5],[6.5,2],[3,1]],dtype=float); \
///     y=np.array([0,0,0,1,1,1,2,2,0]); \
///     [print(V, DecisionTreeClassifier(min_impurity_decrease=V,random_state=0).fit(X,y).tree_.node_count, \
///       DecisionTreeClassifier(min_impurity_decrease=V,random_state=0).fit(X,y).predict(X).tolist()) \
///       for V in (0.0,0.2,0.5)]"
///   => V=0.0 node_count 7  predict [0,0,0,1,1,1,2,2,0]
///      V=0.2 node_count 3  predict [0,0,0,1,1,1,0,0,0]
///      V=0.5 node_count 1  predict [0,0,0,0,0,0,0,0,0]
#[test]
fn req3_clf_min_impurity_decrease_oracle() {
    let (x, y) = clf_dataset();

    // Default (V=0.0): impurity gate is OFF — full oracle tree (node_count 7).
    let default = DecisionTreeClassifier::<f64>::new().fit(&x, &y).unwrap();
    assert_eq!(
        default.nodes().len(),
        7,
        "min_impurity_decrease default node_count: ferrolearn={} sklearn=7",
        default.nodes().len()
    );
    assert_eq!(
        default.predict(&x).unwrap().to_vec().as_slice(),
        &[0, 0, 0, 1, 1, 1, 2, 2, 0],
        "mid=0.0 predict"
    );

    // V=0.2: prunes the class-2 split — node_count 3.
    let v02 = DecisionTreeClassifier::<f64>::new()
        .with_min_impurity_decrease(0.2)
        .fit(&x, &y)
        .unwrap();
    assert_eq!(
        v02.nodes().len(),
        3,
        "min_impurity_decrease=0.2 node_count: ferrolearn={} sklearn=3",
        v02.nodes().len()
    );
    assert_eq!(
        v02.predict(&x).unwrap().to_vec().as_slice(),
        &[0, 0, 0, 1, 1, 1, 0, 0, 0],
        "mid=0.2 predict"
    );

    // V=0.5: prunes the root — single leaf, all-majority (0).
    let v05 = DecisionTreeClassifier::<f64>::new()
        .with_min_impurity_decrease(0.5)
        .fit(&x, &y)
        .unwrap();
    assert_eq!(
        v05.nodes().len(),
        1,
        "min_impurity_decrease=0.5 node_count: ferrolearn={} sklearn=1",
        v05.nodes().len()
    );
    assert!(
        matches!(v05.nodes().first(), Some(Node::Leaf { .. })),
        "mid=0.5 root must be a Leaf"
    );
    assert_eq!(
        v05.predict(&x).unwrap().to_vec().as_slice(),
        &[0, 0, 0, 0, 0, 0, 0, 0, 0],
        "mid=0.5 predict"
    );
}

/// PIN B — `min_weight_fraction_leaf` gates the CLASSIFIER tree (#664).
///
/// Oracle (sklearn 1.5.2):
///   python3 -c "import numpy as np; from sklearn.tree import DecisionTreeClassifier; \
///     X=np.array([[1,2],[2,3],[3,3],[5,6],[6,7],[7,8],[1.5,5],[6.5,2],[3,1]],dtype=float); \
///     y=np.array([0,0,0,1,1,1,2,2,0]); \
///     [print(W, DecisionTreeClassifier(min_weight_fraction_leaf=W,random_state=0).fit(X,y).tree_.node_count, \
///       DecisionTreeClassifier(min_weight_fraction_leaf=W,random_state=0).fit(X,y).predict(X).tolist()) \
///       for W in (0.0,0.25)]"
///   => W=0.0  node_count 7  predict [0,0,0,1,1,1,2,2,0]
///      W=0.25 node_count 5  predict [0,0,0,1,1,1,0,0,0]
#[test]
fn req3_clf_min_weight_fraction_leaf_oracle() {
    let (x, y) = clf_dataset();

    // Default (W=0.0): full oracle tree (node_count 7).
    let default = DecisionTreeClassifier::<f64>::new().fit(&x, &y).unwrap();
    assert_eq!(
        default.nodes().len(),
        7,
        "min_weight_fraction_leaf default node_count: ferrolearn={} sklearn=7",
        default.nodes().len()
    );
    assert_eq!(
        default.predict(&x).unwrap().to_vec().as_slice(),
        &[0, 0, 0, 1, 1, 1, 2, 2, 0],
        "mwfl=0.0 predict"
    );

    // W=0.25: effective per-child minimum becomes ceil(0.25*9)=3 — node_count 5.
    let w025 = DecisionTreeClassifier::<f64>::new()
        .with_min_weight_fraction_leaf(0.25)
        .fit(&x, &y)
        .unwrap();
    assert_eq!(
        w025.nodes().len(),
        5,
        "min_weight_fraction_leaf=0.25 node_count: ferrolearn={} sklearn=5",
        w025.nodes().len()
    );
    assert_eq!(
        w025.predict(&x).unwrap().to_vec().as_slice(),
        &[0, 0, 0, 1, 1, 1, 0, 0, 0],
        "mwfl=0.25 predict"
    );
}

/// PIN C — `min_impurity_decrease` gates the REGRESSOR tree (#662). Exercises
/// the regressor finder's split-accept path with the impurity gate (the builder
/// noted the regressor finder keeps a strict `>0` accept — this pins a value
/// where the oracle measurably reduces the tree, and a value that collapses it
/// to a single leaf).
///
/// Oracle (sklearn 1.5.2):
///   python3 -c "import numpy as np; from sklearn.tree import DecisionTreeRegressor; \
///     Xr=np.array([[1],[2],[3],[4],[5],[6],[7],[8]],dtype=float); \
///     yr=np.array([1.0,1.2,0.9,1.1,5.0,5.2,4.9,5.1]); \
///     [print(V, DecisionTreeRegressor(min_impurity_decrease=V,random_state=0).fit(Xr,yr).tree_.node_count, \
///       DecisionTreeRegressor(min_impurity_decrease=V,random_state=0).fit(Xr,yr).predict(Xr).tolist()) \
///       for V in (0.0,0.5,5.0)]"
///   => V=0.0 node_count 15 predict [1.0,1.2,0.9,1.1,5.0,5.2,4.9,5.1]
///      V=0.5 node_count 3  root (0,4.5) predict [1.05,1.05,1.05,1.05,5.05,5.05,5.05,5.05]
///      V=5.0 node_count 1  predict all 3.05 (mean of all y = 24.4/8)
#[test]
fn req3_reg_min_impurity_decrease_oracle() {
    let (x, y) = reg_dataset();

    // Default (V=0.0): impurity gate OFF — full oracle tree (node_count 15).
    let default = DecisionTreeRegressor::<f64>::new().fit(&x, &y).unwrap();
    assert_eq!(
        default.nodes().len(),
        15,
        "REG min_impurity_decrease default node_count: ferrolearn={} sklearn=15",
        default.nodes().len()
    );

    // V=0.5: gate prunes 15 -> 3, root (0, 4.5), mean leaves 1.05 / 5.05.
    let v05 = DecisionTreeRegressor::<f64>::new()
        .with_min_impurity_decrease(0.5)
        .fit(&x, &y)
        .unwrap();
    assert_eq!(
        v05.nodes().len(),
        3,
        "REG min_impurity_decrease=0.5 node_count: ferrolearn={} sklearn=3",
        v05.nodes().len()
    );
    let (feat, thr) = root_split(v05.nodes()).expect("REG mid=0.5 root must be a Split");
    assert_eq!(
        feat, 0,
        "REG mid=0.5 root feature: ferrolearn={feat} sklearn=0"
    );
    assert!(
        (thr - 4.5).abs() < 1e-9,
        "REG mid=0.5 root threshold: ferrolearn={thr} sklearn=4.5"
    );
    let sk_v05_predict = [1.05, 1.05, 1.05, 1.05, 5.05, 5.05, 5.05, 5.05];
    let preds = v05.predict(&x).unwrap();
    for (i, &exp) in sk_v05_predict.iter().enumerate() {
        assert!(
            (preds[i] - exp).abs() < 1e-9,
            "REG mid=0.5 predict[{i}]: ferrolearn={} sklearn={exp}",
            preds[i]
        );
    }

    // V=5.0: gate prunes the root — single leaf, mean of all y = 3.05.
    let v50 = DecisionTreeRegressor::<f64>::new()
        .with_min_impurity_decrease(5.0)
        .fit(&x, &y)
        .unwrap();
    assert_eq!(
        v50.nodes().len(),
        1,
        "REG min_impurity_decrease=5.0 node_count: ferrolearn={} sklearn=1",
        v50.nodes().len()
    );
    assert!(
        matches!(v50.nodes().first(), Some(Node::Leaf { .. })),
        "REG mid=5.0 root must be a Leaf"
    );
    let preds50 = v50.predict(&x).unwrap();
    for i in 0..preds50.len() {
        assert!(
            (preds50[i] - 3.05).abs() < 1e-9,
            "REG mid=5.0 predict[{i}] (mean of all y): ferrolearn={} sklearn=3.05",
            preds50[i]
        );
    }
}

// ===========================================================================
// REQ-3 ccp_alpha (minimal cost-complexity pruning) CERTIFYING PINS (#663).
// Breiman weakest-link pruning applied at the END of fit when ccp_alpha > 0
// (`fn prune_ccp` / `fn rebuild_pruned_tree`, the native analog of sklearn's
// `_cost_complexity_prune` + `_build_pruned_tree_ccp`, `_tree.pyx`;
// stop_pruning when `ccp_alpha < effective_alpha`, `_tree.pyx:1617`).
// All expected values derived LIVE from the sklearn 1.5.2 oracle (R-CHAR-3);
// the exact python invocations are recorded above each pin. Tolerance 1e-9.
// node_count = `tree_.node_count`; predict = `predict(X)`.
// PIN C (cost_complexity_pruning_path values) is NOT pinned: ferrolearn exposes
// NO `cost_complexity_pruning_path`/`ccp_alphas` accessor (verified by
// `grep cost_complexity_pruning_path decision_tree.rs lib.rs` => none public);
// the pruning is therefore certified through the OBSERVABLE node_count/predict
// contract, which is exactly what `ccp_alpha` controls. The oracle ccp_alphas
// path is recorded here for traceability:
//   clf cost_complexity_pruning_path(X,y).ccp_alphas
//     => [0.0, 0.14814814814814814, 0.345679012345679]
//   reg cost_complexity_pruning_path(Xr,yr).ccp_alphas
//     => [0.0, 0.0020833333…, 0.0020833333…, 3.9999999999999996]
// ===========================================================================

/// PIN A — `ccp_alpha` prunes the CLASSIFIER tree (#663).
///
/// On the 9x2 oracle set the weakest-link effective_alpha is 0.14814814…
/// (the class-2 subtree). So `ccp_alpha=0.1 < 0.14815` keeps the full tree,
/// and `ccp_alpha=0.3 >= 0.14815` (but `< 0.34568`) collapses exactly that
/// subtree → node_count 3. Default `ccp_alpha=0.0` ⇒ no pruning (node_count 7).
///
/// Oracle (sklearn 1.5.2):
///   python3 -c "import numpy as np; from sklearn.tree import DecisionTreeClassifier; \
///     X=np.array([[1,2],[2,3],[3,3],[5,6],[6,7],[7,8],[1.5,5],[6.5,2],[3,1]],dtype=float); \
///     y=np.array([0,0,0,1,1,1,2,2,0]); \
///     [print(A, DecisionTreeClassifier(ccp_alpha=A,random_state=0).fit(X,y).tree_.node_count, \
///       DecisionTreeClassifier(ccp_alpha=A,random_state=0).fit(X,y).predict(X).tolist()) \
///       for A in (0.0,0.1,0.3)]"
///   => A=0.0 node_count 7 predict [0,0,0,1,1,1,2,2,0]
///      A=0.1 node_count 7 predict [0,0,0,1,1,1,2,2,0]   (0.1 < weakest-link 0.14815)
///      A=0.3 node_count 3 predict [0,0,0,1,1,1,0,0,0]   (prunes the 0.14815 subtree)
#[test]
fn req3_clf_ccp_alpha_oracle() {
    let (x, y) = clf_dataset();

    // Default (ccp_alpha=0.0): no pruning — full oracle tree (node_count 7).
    let default = DecisionTreeClassifier::<f64>::new().fit(&x, &y).unwrap();
    assert_eq!(
        default.nodes().len(),
        7,
        "ccp_alpha default node_count: ferrolearn={} sklearn=7",
        default.nodes().len()
    );
    assert_eq!(
        default.predict(&x).unwrap().to_vec().as_slice(),
        &[0, 0, 0, 1, 1, 1, 2, 2, 0],
        "ccp_alpha=0.0 predict"
    );

    // A=0.1: below the weakest-link effective_alpha (0.14815) — no prune.
    let a01 = DecisionTreeClassifier::<f64>::new()
        .with_ccp_alpha(0.1)
        .fit(&x, &y)
        .unwrap();
    assert_eq!(
        a01.nodes().len(),
        7,
        "ccp_alpha=0.1 node_count: ferrolearn={} sklearn=7 (0.1 < 0.14815)",
        a01.nodes().len()
    );
    assert_eq!(
        a01.predict(&x).unwrap().to_vec().as_slice(),
        &[0, 0, 0, 1, 1, 1, 2, 2, 0],
        "ccp_alpha=0.1 predict"
    );

    // A=0.3: prunes the 0.14815 subtree (class-2 split) — node_count 3.
    let a03 = DecisionTreeClassifier::<f64>::new()
        .with_ccp_alpha(0.3)
        .fit(&x, &y)
        .unwrap();
    assert_eq!(
        a03.nodes().len(),
        3,
        "ccp_alpha=0.3 node_count: ferrolearn={} sklearn=3",
        a03.nodes().len()
    );
    assert_eq!(
        a03.predict(&x).unwrap().to_vec().as_slice(),
        &[0, 0, 0, 1, 1, 1, 0, 0, 0],
        "ccp_alpha=0.3 predict"
    );
}

/// PIN B — `ccp_alpha` prunes the REGRESSOR tree (#663).
///
/// On the 8x1 oracle set the full tree has node_count 15 (each sample its own
/// leaf). `ccp_alpha=0.05` collapses the tree to the two top-level clusters →
/// node_count 3, root `(0, 4.5)`, MEAN leaves `[1.05x4, 5.05x4]`.
/// Default `ccp_alpha=0.0` ⇒ no pruning (node_count 15).
///
/// Oracle (sklearn 1.5.2):
///   python3 -c "import numpy as np; from sklearn.tree import DecisionTreeRegressor; \
///     Xr=np.array([[1],[2],[3],[4],[5],[6],[7],[8]],dtype=float); \
///     yr=np.array([1.0,1.2,0.9,1.1,5.0,5.2,4.9,5.1]); \
///     [print(A, DecisionTreeRegressor(ccp_alpha=A).fit(Xr,yr).tree_.node_count, \
///       DecisionTreeRegressor(ccp_alpha=A).fit(Xr,yr).predict(Xr).tolist()) \
///       for A in (0.0,0.05)]"
///   => A=0.0  node_count 15 predict [1.0,1.2,0.9,1.1,5.0,5.2,4.9,5.1]
///      A=0.05 node_count 3  predict [1.05,1.05,1.05,1.05,5.05,5.05,5.05,5.05]
#[test]
fn req3_reg_ccp_alpha_oracle() {
    let (x, y) = reg_dataset();

    // Default (ccp_alpha=0.0): no pruning — full oracle tree (node_count 15).
    let default = DecisionTreeRegressor::<f64>::new().fit(&x, &y).unwrap();
    assert_eq!(
        default.nodes().len(),
        15,
        "REG ccp_alpha default node_count: ferrolearn={} sklearn=15",
        default.nodes().len()
    );
    let def_pred = default.predict(&x).unwrap();
    for (i, &exp) in [1.0, 1.2, 0.9, 1.1, 5.0, 5.2, 4.9, 5.1].iter().enumerate() {
        assert!(
            (def_pred[i] - exp).abs() < 1e-9,
            "REG ccp_alpha=0.0 predict[{i}]: ferrolearn={} sklearn={exp}",
            def_pred[i]
        );
    }

    // A=0.05: prunes 15 -> 3, root (0, 4.5), mean leaves 1.05 / 5.05.
    let a05 = DecisionTreeRegressor::<f64>::new()
        .with_ccp_alpha(0.05)
        .fit(&x, &y)
        .unwrap();
    assert_eq!(
        a05.nodes().len(),
        3,
        "REG ccp_alpha=0.05 node_count: ferrolearn={} sklearn=3",
        a05.nodes().len()
    );
    let (feat, thr) = root_split(a05.nodes()).expect("REG ccp_alpha=0.05 root must be a Split");
    assert_eq!(
        feat, 0,
        "REG ccp_alpha=0.05 root feature: ferrolearn={feat} sklearn=0"
    );
    assert!(
        (thr - 4.5).abs() < 1e-9,
        "REG ccp_alpha=0.05 root threshold: ferrolearn={thr} sklearn=4.5"
    );
    let sk_a05_predict = [1.05, 1.05, 1.05, 1.05, 5.05, 5.05, 5.05, 5.05];
    let preds = a05.predict(&x).unwrap();
    for (i, &exp) in sk_a05_predict.iter().enumerate() {
        assert!(
            (preds[i] - exp).abs() < 1e-9,
            "REG ccp_alpha=0.05 predict[{i}] (mean leaf): ferrolearn={} sklearn={exp}",
            preds[i]
        );
    }
}

// ===========================================================================
// REQ-3 max_leaf_nodes (best-first growth) CERTIFYING PINS (#664).
// `with_max_leaf_nodes(Some(k))` dispatches `fit` to the best-first builders
// (`fn build_classification_tree_best_first` / `fn build_regression_tree_best_first`),
// the native analog of sklearn's `BestFirstTreeBuilder.build` (`_tree.pyx:427`):
// a max-heap frontier on tree-normalized improvement (`fn pop_best_frontier`),
// `max_split_nodes = k - 1` (`_tree.pyx:457`), reusing the split-finder +
// `ImpurityGate`. The regressor heap key uses a two-pass centered variance
// (`fn stable_regression_improvement`) so the ~2e-17 near-tie at k=4 keeps
// sklearn's frontier order. All expected values derived LIVE from the
// sklearn 1.5.2 oracle (R-CHAR-3); python invocations recorded above each pin.
// node_count = `tree_.node_count`; n_leaves = `get_n_leaves()`; predict =
// `predict(X)`. Tests run in integration-test context — the `.unwrap()` idiom
// matches the prior pins above (R-APG-2: test context is gate-exempt).
// ===========================================================================

/// Count `Node::Leaf` entries in a serialized tree (sklearn `get_n_leaves()`).
fn count_leaves(nodes: &[Node<f64>]) -> usize {
    nodes
        .iter()
        .filter(|n| matches!(n, Node::Leaf { .. }))
        .count()
}

/// PIN A — `max_leaf_nodes` best-first growth on the CLASSIFIER (#664).
///
/// Oracle (sklearn 1.5.2):
///   python3 -c "import numpy as np; from sklearn.tree import DecisionTreeClassifier; \
///     X=np.array([[1,2],[2,3],[3,3],[5,6],[6,7],[7,8],[1.5,5],[6.5,2],[3,1]],dtype=float); \
///     y=np.array([0,0,0,1,1,1,2,2,0]); \
///     [print(k, DecisionTreeClassifier(max_leaf_nodes=k,random_state=0).fit(X,y).tree_.node_count, \
///       DecisionTreeClassifier(max_leaf_nodes=k,random_state=0).fit(X,y).get_n_leaves(), \
///       DecisionTreeClassifier(max_leaf_nodes=k,random_state=0).fit(X,y).predict(X).tolist()) \
///       for k in (2,3,4)]; \
///     print('None', DecisionTreeClassifier(random_state=0).fit(X,y).tree_.node_count)"
///   => k=2  node_count 3  n_leaves 2  predict [0,0,0,1,1,1,0,0,0]
///      k=3  node_count 5  n_leaves 3  predict [0,0,0,1,1,1,0,2,0]
///      k=4  node_count 7  n_leaves 4  predict [0,0,0,1,1,1,2,2,0]
///      None node_count 7
#[test]
fn req3_clf_max_leaf_nodes_oracle() {
    let (x, y) = clf_dataset();

    // k=2 -> node_count 3 / 2 leaves; the best-first frontier expands only the
    // single highest-improvement split (the class-0/1 separation).
    let k2 = DecisionTreeClassifier::<f64>::new()
        .with_max_leaf_nodes(Some(2))
        .fit(&x, &y)
        .unwrap();
    assert_eq!(
        k2.nodes().len(),
        3,
        "CLF k=2 node_count: ferrolearn={} sklearn=3",
        k2.nodes().len()
    );
    assert_eq!(
        count_leaves(k2.nodes()),
        2,
        "CLF k=2 n_leaves: ferrolearn={} sklearn=2",
        count_leaves(k2.nodes())
    );
    assert_eq!(
        k2.predict(&x).unwrap().to_vec().as_slice(),
        &[0, 0, 0, 1, 1, 1, 0, 0, 0],
        "CLF k=2 predict"
    );

    // k=3 -> node_count 5 / 3 leaves; predict separates one class-2 sample.
    let k3 = DecisionTreeClassifier::<f64>::new()
        .with_max_leaf_nodes(Some(3))
        .fit(&x, &y)
        .unwrap();
    assert_eq!(
        k3.nodes().len(),
        5,
        "CLF k=3 node_count: ferrolearn={} sklearn=5",
        k3.nodes().len()
    );
    assert_eq!(
        count_leaves(k3.nodes()),
        3,
        "CLF k=3 n_leaves: ferrolearn={} sklearn=3",
        count_leaves(k3.nodes())
    );
    assert_eq!(
        k3.predict(&x).unwrap().to_vec().as_slice(),
        &[0, 0, 0, 1, 1, 1, 0, 2, 0],
        "CLF k=3 predict"
    );

    // k=4 -> node_count 7 / 4 leaves == the unlimited (depth-first) tree.
    let k4 = DecisionTreeClassifier::<f64>::new()
        .with_max_leaf_nodes(Some(4))
        .fit(&x, &y)
        .unwrap();
    assert_eq!(
        k4.nodes().len(),
        7,
        "CLF k=4 node_count: ferrolearn={} sklearn=7",
        k4.nodes().len()
    );
    assert_eq!(
        count_leaves(k4.nodes()),
        4,
        "CLF k=4 n_leaves: ferrolearn={} sklearn=4",
        count_leaves(k4.nodes())
    );
    assert_eq!(
        k4.predict(&x).unwrap().to_vec().as_slice(),
        &[0, 0, 0, 1, 1, 1, 2, 2, 0],
        "CLF k=4 predict"
    );

    // None -> depth-first builder, node_count 7 (unchanged).
    let none = DecisionTreeClassifier::<f64>::new()
        .with_max_leaf_nodes(None)
        .fit(&x, &y)
        .unwrap();
    assert_eq!(
        none.nodes().len(),
        7,
        "CLF None node_count: ferrolearn={} sklearn=7",
        none.nodes().len()
    );
}

/// PIN B — `max_leaf_nodes` best-first growth on the REGRESSOR (#664).
///
/// SCRUTINY (k=4 near-tie): the depth-2 frontier holds the [5,6]-node
/// (y=[5.0,5.2], split-improvement 0.0025000000000000044) and the [7,8]-node
/// (y=[4.9,5.1], split-improvement 0.0024999999999999823) — differing by
/// ~2.2e-17. sklearn's running-sum criterion expands the [5,6]-node first
/// (higher improvement); the naive `Sum(y^2)/n - mean^2` variance can flip this
/// order, so the builder orders the heap via `fn stable_regression_improvement`
/// (two-pass centered variance). This pin asserts ferrolearn reproduces
/// sklearn's exact frontier order at the near-tie: predict X=6 -> 5.2 (its own
/// leaf) and X=7,8 -> 5.0 (collapsed). The result is random_state-invariant
/// (single feature => no subsampling), verified across seeds.
///
/// Oracle (sklearn 1.5.2):
///   python3 -c "import numpy as np; from sklearn.tree import DecisionTreeRegressor; \
///     Xr=np.array([[1],[2],[3],[4],[5],[6],[7],[8]],dtype=float); \
///     yr=np.array([1.0,1.2,0.9,1.1,5.0,5.2,4.9,5.1]); \
///     [print(k, DecisionTreeRegressor(max_leaf_nodes=k).fit(Xr,yr).tree_.node_count, \
///       DecisionTreeRegressor(max_leaf_nodes=k).fit(Xr,yr).get_n_leaves(), \
///       DecisionTreeRegressor(max_leaf_nodes=k).fit(Xr,yr).predict(Xr).tolist()) \
///       for k in (2,3,4)]; \
///     print('None', DecisionTreeRegressor().fit(Xr,yr).tree_.node_count)"
///   => k=2  node_count 3  n_leaves 2  predict [1.05,1.05,1.05,1.05,5.05,5.05,5.05,5.05]
///      k=3  node_count 5  n_leaves 3  predict [1.05,1.05,1.05,1.05,5.1,5.1,5.0,5.0]
///      k=4  node_count 7  n_leaves 4  predict [1.05,1.05,1.05,1.05,5.0,5.2,5.0,5.0]
///      None node_count 15
#[test]
fn req3_reg_max_leaf_nodes_oracle() {
    let (x, y) = reg_dataset();

    // k=2 -> node_count 3 / 2 leaves; root (0,4.5), mean leaves 1.05 / 5.05.
    let k2 = DecisionTreeRegressor::<f64>::new()
        .with_max_leaf_nodes(Some(2))
        .fit(&x, &y)
        .unwrap();
    assert_eq!(
        k2.nodes().len(),
        3,
        "REG k=2 node_count: ferrolearn={} sklearn=3",
        k2.nodes().len()
    );
    assert_eq!(
        count_leaves(k2.nodes()),
        2,
        "REG k=2 n_leaves: ferrolearn={} sklearn=2",
        count_leaves(k2.nodes())
    );
    let k2_pred = k2.predict(&x).unwrap();
    for (i, &exp) in [1.05, 1.05, 1.05, 1.05, 5.05, 5.05, 5.05, 5.05]
        .iter()
        .enumerate()
    {
        assert!(
            (k2_pred[i] - exp).abs() < 1e-9,
            "REG k=2 predict[{i}]: ferrolearn={} sklearn={exp}",
            k2_pred[i]
        );
    }

    // k=3 -> node_count 5 / 3 leaves; predict [1.05x4, 5.1, 5.1, 5.0, 5.0].
    let k3 = DecisionTreeRegressor::<f64>::new()
        .with_max_leaf_nodes(Some(3))
        .fit(&x, &y)
        .unwrap();
    assert_eq!(
        k3.nodes().len(),
        5,
        "REG k=3 node_count: ferrolearn={} sklearn=5",
        k3.nodes().len()
    );
    assert_eq!(
        count_leaves(k3.nodes()),
        3,
        "REG k=3 n_leaves: ferrolearn={} sklearn=3",
        count_leaves(k3.nodes())
    );
    let k3_pred = k3.predict(&x).unwrap();
    for (i, &exp) in [1.05, 1.05, 1.05, 1.05, 5.1, 5.1, 5.0, 5.0]
        .iter()
        .enumerate()
    {
        assert!(
            (k3_pred[i] - exp).abs() < 1e-9,
            "REG k=3 predict[{i}]: ferrolearn={} sklearn={exp}",
            k3_pred[i]
        );
    }

    // k=4 (NEAR-TIE) -> node_count 7 / 4 leaves; the [5,6]-node is expanded
    // before [7,8] becomes a leaf, so predict [1.05x4, 5.0, 5.2, 5.0, 5.0].
    let k4 = DecisionTreeRegressor::<f64>::new()
        .with_max_leaf_nodes(Some(4))
        .fit(&x, &y)
        .unwrap();
    assert_eq!(
        k4.nodes().len(),
        7,
        "REG k=4 node_count: ferrolearn={} sklearn=7",
        k4.nodes().len()
    );
    assert_eq!(
        count_leaves(k4.nodes()),
        4,
        "REG k=4 n_leaves: ferrolearn={} sklearn=4",
        count_leaves(k4.nodes())
    );
    let k4_pred = k4.predict(&x).unwrap();
    for (i, &exp) in [1.05, 1.05, 1.05, 1.05, 5.0, 5.2, 5.0, 5.0]
        .iter()
        .enumerate()
    {
        assert!(
            (k4_pred[i] - exp).abs() < 1e-9,
            "REG k=4 (near-tie) predict[{i}]: ferrolearn={} sklearn={exp}",
            k4_pred[i]
        );
    }

    // None -> depth-first builder, node_count 15 (each sample its own leaf).
    let none = DecisionTreeRegressor::<f64>::new()
        .with_max_leaf_nodes(None)
        .fit(&x, &y)
        .unwrap();
    assert_eq!(
        none.nodes().len(),
        15,
        "REG None node_count: ferrolearn={} sklearn=15",
        none.nodes().len()
    );
}

// ===========================================================================
// REQ-7 class_weight CERTIFYING PINS (#665).
// `with_class_weight(ClassWeight::{None,Balanced,Explicit})` on
// `DecisionTreeClassifier`. sklearn expands `class_weight` to per-sample
// weights (`compute_sample_weight`, `_classes.py:310-367`) and folds them into
// node class counts, the gini/entropy split (`fn weighted_compute_impurity`),
// AND the leaf `value_`/`predict_proba` (`fn weighted_classification_node_value`).
// `ClassWeight` is NOT re-exported at the crate root, so it is imported via the
// module path `ferrolearn_tree::decision_tree::ClassWeight`.
// All expected values derived LIVE from the sklearn 1.5.2 oracle (R-CHAR-3):
//   python3 -c "import numpy as np; from sklearn.tree import DecisionTreeClassifier; \
//     X=np.array([[1,0],[1.5,0],[2,0],[1.2,0],[2.2,0],[5,0],[6,0],[7,0]],dtype=float); \
//     y=np.array([0,0,0,1,1,1,1,1]); \
//     [print(cw, DecisionTreeClassifier(max_depth=1,class_weight=cw,random_state=0).fit(X,y).tree_.feature[0], \
//       DecisionTreeClassifier(max_depth=1,class_weight=cw,random_state=0).fit(X,y).tree_.threshold[0], \
//       DecisionTreeClassifier(max_depth=1,class_weight=cw,random_state=0).fit(X,y).predict(X).tolist(), \
//       DecisionTreeClassifier(max_depth=1,class_weight=cw,random_state=0).fit(X,y).predict_proba(X)[0].tolist()) \
//       for cw in (None,{0:1.0,1:5.0},'balanced')]"
//   => None       root (0, 2.1) predict [0,0,0,0,1,1,1,1] proba0 [0.75, 0.25]
//      {0:1,1:5}  root (0, 1.1) predict [0,1,1,1,1,1,1,1] proba0 [1.0, 0.0]
//      balanced   root (0, 2.1) predict [0,0,0,0,1,1,1,1] proba0 [0.8333…, 0.1667…]
//   balanced weights = compute_class_weight('balanced', [0,1], y) = [8/6, 0.8]
//   (n_samples / (n_classes * count_c): 8/(2*3)=1.333…, 8/(2*5)=0.8).
// Tests run in integration-test context — the `.unwrap()` idiom matches the
// prior pins above (R-APG-2: test context is gate-exempt).
// ===========================================================================

use ferrolearn_tree::decision_tree::ClassWeight;

/// PIN — `DecisionTreeClassifier` class_weight wired through impurity AND leaf
/// value (#665). On the 8x2 set above, `None`, `Explicit([(0,1),(1,5)])`, and
/// `Balanced` produce three MUTUALLY DISTINCT fitted trees:
///   * `None`/`Balanced` split at threshold 2.1, predict `[0,0,0,0,1,1,1,1]`;
///   * `Explicit` flips both the split (threshold 1.1) AND the leaf proba (the
///     class-1 reweighting by 5x makes the root-left leaf pure class 0 →
///     proba0 `[1.0,0.0]`, the proof class_weight reaches BOTH the split and the
///     leaf value);
///   * `Balanced` keeps `None`'s split but reweights the leaf proba to
///     `[0.8333…,0.1667…]` (`3·w0/(3·w0+1·w1)`, w0=8/6, w1=0.8), distinct from
///     `None`'s `[0.75,0.25]` (the proof the leaf value is weighted).
///
/// The pin FAILS if class_weight were ignored (the three would coincide).
/// Also asserts `Balanced == Explicit([(0,8/6),(1,0.8)])` (the balanced formula,
/// `class_weight.py:72`).
#[test]
fn req7_clf_class_weight_oracle() {
    // Oracle (live sklearn 1.5.2) — see header for invocation.
    const SK_THR_NONE: f64 = 2.1;
    const SK_THR_EXPLICIT: f64 = 1.1;
    const SK_THR_BALANCED: f64 = 2.1;
    let sk_predict_none: [usize; 8] = [0, 0, 0, 0, 1, 1, 1, 1];
    let sk_predict_explicit: [usize; 8] = [0, 1, 1, 1, 1, 1, 1, 1];
    let sk_predict_balanced: [usize; 8] = [0, 0, 0, 0, 1, 1, 1, 1];
    let sk_proba0_none: [f64; 2] = [0.75, 0.25];
    let sk_proba0_explicit: [f64; 2] = [1.0, 0.0];
    let sk_proba0_balanced: [f64; 2] = [0.833_333_333_333_333_4, 0.166_666_666_666_666_69];

    let x = Array2::from_shape_vec(
        (8, 2),
        vec![
            1.0, 0.0, 1.5, 0.0, 2.0, 0.0, 1.2, 0.0, 2.2, 0.0, 5.0, 0.0, 6.0, 0.0, 7.0, 0.0,
        ],
    )
    .unwrap();
    let y = array![0usize, 0, 0, 1, 1, 1, 1, 1];

    let fit_cw = |cw: ClassWeight<f64>| {
        DecisionTreeClassifier::<f64>::new()
            .with_max_depth(Some(1))
            .with_class_weight(cw)
            .fit(&x, &y)
            .unwrap()
    };

    let none = fit_cw(ClassWeight::None);
    let explicit = fit_cw(ClassWeight::Explicit(vec![(0, 1.0), (1, 5.0)]));
    let balanced = fit_cw(ClassWeight::Balanced);

    // --- None ---
    let (nf, nt) = root_split(none.nodes()).expect("None root must be a Split");
    assert_eq!(nf, 0, "None root feature: ferrolearn={nf} sklearn=0");
    assert!(
        (nt - SK_THR_NONE).abs() < 1e-6,
        "None root threshold: ferrolearn={nt} sklearn={SK_THR_NONE}"
    );
    let none_pred = none.predict(&x).unwrap();
    for (i, &exp) in sk_predict_none.iter().enumerate() {
        assert_eq!(
            none_pred[i], exp,
            "None predict[{i}]: ferrolearn={} sklearn={exp}",
            none_pred[i]
        );
    }
    let none_proba = none.predict_proba(&x).unwrap();
    for (j, &exp) in sk_proba0_none.iter().enumerate() {
        assert!(
            (none_proba[[0, j]] - exp).abs() < 1e-6,
            "None proba0[{j}]: ferrolearn={} sklearn={exp}",
            none_proba[[0, j]]
        );
    }

    // --- Explicit([(0,1),(1,5)]) — split AND leaf flip ---
    let (ef, et) = root_split(explicit.nodes()).expect("Explicit root must be a Split");
    assert_eq!(ef, 0, "Explicit root feature: ferrolearn={ef} sklearn=0");
    assert!(
        (et - SK_THR_EXPLICIT).abs() < 1e-6,
        "Explicit root threshold: ferrolearn={et} sklearn={SK_THR_EXPLICIT}"
    );
    let exp_pred = explicit.predict(&x).unwrap();
    for (i, &exp) in sk_predict_explicit.iter().enumerate() {
        assert_eq!(
            exp_pred[i], exp,
            "Explicit predict[{i}]: ferrolearn={} sklearn={exp}",
            exp_pred[i]
        );
    }
    let exp_proba = explicit.predict_proba(&x).unwrap();
    for (j, &exp) in sk_proba0_explicit.iter().enumerate() {
        assert!(
            (exp_proba[[0, j]] - exp).abs() < 1e-6,
            "Explicit proba0[{j}]: ferrolearn={} sklearn={exp}",
            exp_proba[[0, j]]
        );
    }

    // --- Balanced — same split as None, reweighted leaf proba ---
    let (bf, bt) = root_split(balanced.nodes()).expect("Balanced root must be a Split");
    assert_eq!(bf, 0, "Balanced root feature: ferrolearn={bf} sklearn=0");
    assert!(
        (bt - SK_THR_BALANCED).abs() < 1e-6,
        "Balanced root threshold: ferrolearn={bt} sklearn={SK_THR_BALANCED}"
    );
    let bal_pred = balanced.predict(&x).unwrap();
    for (i, &exp) in sk_predict_balanced.iter().enumerate() {
        assert_eq!(
            bal_pred[i], exp,
            "Balanced predict[{i}]: ferrolearn={} sklearn={exp}",
            bal_pred[i]
        );
    }
    let bal_proba = balanced.predict_proba(&x).unwrap();
    for (j, &exp) in sk_proba0_balanced.iter().enumerate() {
        assert!(
            (bal_proba[[0, j]] - exp).abs() < 1e-6,
            "Balanced proba0[{j}]: ferrolearn={} sklearn={exp}",
            bal_proba[[0, j]]
        );
    }

    // --- Mutual distinctness: the pin FAILS if class_weight were ignored. ---
    // Explicit's split (1.1) must differ from None's (2.1).
    assert!(
        (et - nt).abs() > 1e-6,
        "Explicit split ({et}) must differ from None ({nt}) — proof split is weighted"
    );
    // Balanced's leaf proba (0.833) must differ from None's (0.75).
    assert!(
        (bal_proba[[0, 0]] - none_proba[[0, 0]]).abs() > 1e-6,
        "Balanced proba0 ({}) must differ from None ({}) — proof leaf value is weighted",
        bal_proba[[0, 0]],
        none_proba[[0, 0]]
    );

    // --- Balanced == Explicit([(0,8/6),(1,0.8)]) (the balanced formula). ---
    let balanced_as_explicit = fit_cw(ClassWeight::Explicit(vec![(0, 8.0 / 6.0), (1, 0.8)]));
    let (baf, bat) = root_split(balanced_as_explicit.nodes())
        .expect("Balanced-as-Explicit root must be a Split");
    assert_eq!(
        baf, bf,
        "Balanced-as-Explicit root feature must match Balanced"
    );
    assert!(
        (bat - bt).abs() < 1e-9,
        "Balanced-as-Explicit threshold ({bat}) must match Balanced ({bt})"
    );
    let bae_proba = balanced_as_explicit.predict_proba(&x).unwrap();
    for j in 0..2 {
        assert!(
            (bae_proba[[0, j]] - bal_proba[[0, j]]).abs() < 1e-12,
            "Balanced == Explicit([(0,8/6),(1,0.8)]) proba0[{j}]: explicit={} balanced={}",
            bae_proba[[0, j]],
            bal_proba[[0, j]]
        );
    }
}
