//! Divergence pins for `ferrolearn-tree/src/extra_tree.rs`
//! (`ExtraTreeClassifier` / `ExtraTreeRegressor`) against the live
//! scikit-learn 1.5.2 oracle.
//!
//! ExtraTree is RNG-driven (random split thresholds). Per goal.md's RNG-boundary
//! precedent (SGD shuffle, libsvm CV, decision_tree tie-break), numpy MT19937 vs
//! Rust `StdRng` cannot bit-match, so exact node-for-node fit/predict parity at a
//! fixed `random_state` is a DOCUMENTED BOUNDARY, not a fixable divergence and is
//! NOT pinned here. Only DETERMINISTIC, matchable contracts are pinned:
//!   PIN 1 — constructor defaults match sklearn (`inspect.signature`-derived).
//!   PIN 2 — intra-ferrolearn `random_state` reproducibility (same seed => same tree).
//!   PIN 3 — `ExtraTreeRegressor` regression-criterion gap (#681): `with_criterion`
//!           is accepted but the regression builder is hard-wired to MSE/mean leaves,
//!           so `AbsoluteError` does NOT produce median leaves (sklearn does).
//!
//! Expected values come from the live sklearn oracle or sklearn `file:line`
//! symbolic constants, never copied from the ferrolearn side (R-CHAR-3).

use ferrolearn_core::traits::{Fit, Predict};
use ferrolearn_tree::Node;
use ferrolearn_tree::decision_tree::RegressionCriterion;
use ferrolearn_tree::random_forest::MaxFeatures;
use ferrolearn_tree::{ExtraTreeClassifier, ExtraTreeRegressor};
use ndarray::{Array2, array};

// ---------------------------------------------------------------------------
// PIN 1 — constructor defaults (deterministic; mirrors sklearn `_classes.py`)
// ---------------------------------------------------------------------------

/// Divergence guard: ferrolearn `ExtraTreeClassifier::new()` defaults must mirror
/// `sklearn/tree/_classes.py:1564` ExtraTreeClassifier.__init__
/// (`criterion='gini'`, `splitter='random'`, `max_features='sqrt'`,
/// `random_state=None`).
///
/// Live sklearn oracle (1.5.2):
///   ExtraTreeClassifier().criterion == 'gini'
///   ExtraTreeClassifier().max_features == 'sqrt'
///   ExtraTreeClassifier().random_state is None
///
/// ferrolearn: `max_features = MaxFeatures::Sqrt`,
/// `criterion = ClassificationCriterion::Gini`, `random_state = None`.
/// (This contract currently HOLDS; the test passes — it locks the surface.)
#[test]
fn pin1_clf_defaults_match_sklearn() {
    let m = ExtraTreeClassifier::<f64>::new();
    // sklearn 'sqrt' (max_features default for ExtraTreeClassifier).
    assert_eq!(
        m.max_features,
        MaxFeatures::Sqrt,
        "sklearn ExtraTreeClassifier max_features default is 'sqrt'"
    );
    // sklearn 'gini'.
    use ferrolearn_tree::decision_tree::ClassificationCriterion;
    assert_eq!(
        m.criterion,
        ClassificationCriterion::Gini,
        "sklearn ExtraTreeClassifier criterion default is 'gini'"
    );
    // sklearn random_state=None.
    assert_eq!(
        m.random_state, None,
        "sklearn ExtraTreeClassifier random_state default is None"
    );
}

/// Divergence guard: ferrolearn `ExtraTreeRegressor::new()` defaults must mirror
/// `sklearn/tree/_classes.py:1838` ExtraTreeRegressor.__init__
/// (`criterion='squared_error'`, `splitter='random'`, `max_features=1.0`,
/// `random_state=None`).
///
/// Live sklearn oracle (1.5.2):
///   ExtraTreeRegressor().criterion == 'squared_error'
///   ExtraTreeRegressor().max_features == 1.0   (i.e. ALL features)
///   ExtraTreeRegressor().random_state is None
///
/// sklearn's regressor default `max_features=1.0` means ALL features, which is
/// `MaxFeatures::All` in ferrolearn (NOT Sqrt). If ferrolearn defaulted the
/// regressor to Sqrt that would be a real divergence; this asserts it is All.
#[test]
fn pin1_reg_defaults_match_sklearn() {
    let m = ExtraTreeRegressor::<f64>::new();
    // sklearn max_features=1.0 == all features.
    assert_eq!(
        m.max_features,
        MaxFeatures::All,
        "sklearn ExtraTreeRegressor max_features default is 1.0 (== all features)"
    );
    // sklearn 'squared_error' == MSE.
    assert_eq!(
        m.criterion,
        RegressionCriterion::Mse,
        "sklearn ExtraTreeRegressor criterion default is 'squared_error'"
    );
    assert_eq!(
        m.random_state, None,
        "sklearn ExtraTreeRegressor random_state default is None"
    );
}

// ---------------------------------------------------------------------------
// PIN 2 — intra-ferrolearn random_state reproducibility (deterministic)
// ---------------------------------------------------------------------------

/// Matchable-determinism contract (NOT numpy parity): a fixed `random_state`
/// must produce IDENTICAL trees/predictions across two independent fits.
/// sklearn guarantees the analogous reproducibility for a fixed `random_state`
/// (`sklearn/tree/_classes.py` BaseDecisionTree seeds `check_random_state`).
#[test]
fn pin2_clf_random_state_reproducible() {
    let x = Array2::from_shape_vec(
        (8, 2),
        vec![
            1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 5.0, 6.0, 6.0, 7.0, 7.0, 8.0, 8.0, 9.0,
        ],
    )
    .unwrap();
    let y = array![0usize, 0, 0, 0, 1, 1, 1, 1];

    let f1 = ExtraTreeClassifier::<f64>::new()
        .with_random_state(42)
        .fit(&x, &y)
        .unwrap();
    let f2 = ExtraTreeClassifier::<f64>::new()
        .with_random_state(42)
        .fit(&x, &y)
        .unwrap();

    let p1 = f1.predict(&x).unwrap();
    let p2 = f2.predict(&x).unwrap();
    assert_eq!(p1, p2, "same random_state must give identical predictions");
    assert_eq!(
        f1.nodes().len(),
        f2.nodes().len(),
        "same random_state must give identical tree structure"
    );
    // And the split features/thresholds must coincide node-for-node.
    for (a, b) in f1.nodes().iter().zip(f2.nodes().iter()) {
        match (a, b) {
            (
                Node::Split {
                    feature: fa,
                    threshold: ta,
                    ..
                },
                Node::Split {
                    feature: fb,
                    threshold: tb,
                    ..
                },
            ) => {
                assert_eq!(fa, fb, "split feature must match across identical seeds");
                assert_eq!(ta, tb, "split threshold must match across identical seeds");
            }
            (Node::Leaf { .. }, Node::Leaf { .. }) => {}
            _ => panic!("node kind diverged across identical seeds"),
        }
    }
}

// ---------------------------------------------------------------------------
// PIN 3 — #681 ExtraTreeRegressor regression-criterion gap (RED, fixable)
// ---------------------------------------------------------------------------

// Both PIN 3 tests use widely-separated single-feature data so that — whenever
// the single random threshold lands in the wide (2, 100) gap — the three small-x
// rows (y={1,1,5}) and the three large-x rows (y={20,20,20}) end up in separate
// leaves. The {1,1,5} group has median 1.0 (sklearn MAE leaf, `_criterion.pyx`
// MAE.node_value) but mean 7/3 ≈ 2.333 (sklearn MSE leaf). The exact threshold
// value at a fixed seed is the documented #668 RNG boundary (numpy MT19937 vs
// Rust StdRng); the seeds here are chosen so the isolating split is drawn, which
// is what surfaces the criterion difference (median vs mean leaf).

/// Helper: sorted leaf values of a fitted regressor.
fn reg_leaves(f: &ferrolearn_tree::FittedExtraTreeRegressor<f64>) -> Vec<f64> {
    let mut v: Vec<f64> = f
        .nodes()
        .iter()
        .filter_map(|n| match n {
            Node::Leaf { value, .. } => Some(*value),
            Node::Split { .. } => None,
        })
        .collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v
}

/// Divergence: `ExtraTreeRegressor::with_criterion(RegressionCriterion::AbsoluteError)`
/// is accepted (the field + setter exist at `extra_tree.rs` `with_criterion`),
/// but the regression builder (`fn build_extra_regression_tree` /
/// `fn find_random_regression_split`) is HARD-WIRED to MSE: leaf values are always
/// `fn mean_value` and `RegressionData` carries no `criterion` field, so the
/// criterion is SILENTLY IGNORED.
///
/// sklearn mirror: `sklearn/tree/_criterion.pyx` `MAE`/`absolute_error` predicts
/// the MEDIAN at a leaf (`node_value`), distinct from MSE's mean. decision_tree.rs
/// already honors this (`fn regression_leaf_value`: `AbsoluteError => median_value`,
/// `decision_tree.rs:1452`).
///
/// Live sklearn oracle (1.5.2):
///   X=[[0],[1],[2],[100],[101],[102]], y=[1,1,5,20,20,20], max_depth=1
///   squared_error small-x leaf value (mean of {1,1,5}) = 2.333...
///   absolute_error small-x leaf value (median of {1,1,5}) = 1.0
/// => leaf-value SET differs between criteria.
///
/// Intra-ferrolearn pin: at the SAME `random_state`, `AbsoluteError` must produce
/// a DIFFERENT set of leaf values than `Mse` (median vs mean). ferrolearn produces
/// IDENTICAL leaves (means) because the criterion is ignored => FAILS, pinning #681.
#[test]
fn pin3_reg_absolute_error_differs_from_mse_same_seed() {
    let x = Array2::from_shape_vec((6, 1), vec![0.0, 1.0, 2.0, 100.0, 101.0, 102.0]).unwrap();
    let y = array![1.0f64, 1.0, 5.0, 20.0, 20.0, 20.0];

    let f_mse = ExtraTreeRegressor::<f64>::new()
        .with_max_features(MaxFeatures::All)
        .with_criterion(RegressionCriterion::Mse)
        .with_max_depth(Some(1))
        .with_random_state(7)
        .fit(&x, &y)
        .unwrap();
    let f_mae = ExtraTreeRegressor::<f64>::new()
        .with_max_features(MaxFeatures::All)
        .with_criterion(RegressionCriterion::AbsoluteError)
        .with_max_depth(Some(1))
        .with_random_state(7)
        .fit(&x, &y)
        .unwrap();

    let mse_leaves = reg_leaves(&f_mse);
    let mae_leaves = reg_leaves(&f_mae);

    // sklearn: MAE leaves (medians) differ from MSE leaves (means) on this data.
    // ferrolearn ignores the criterion => leaves are identical => this FAILS.
    assert_ne!(
        mse_leaves, mae_leaves,
        "ExtraTreeRegressor(AbsoluteError) must give median leaves distinct from \
         Mse's mean leaves (sklearn _criterion.pyx MAE.node_value); ferrolearn \
         hard-wires MSE means and ignores `criterion` (#681). \
         Mse leaves={mse_leaves:?}, AbsoluteError leaves={mae_leaves:?}"
    );
}

/// Stronger deterministic pin for #681: the AbsoluteError fit must yield the
/// MEDIAN (1.0) of the {1,1,5} group as a leaf value. With widely-separated x the
/// single random threshold isolates {1,1,5} from {20,20,20} *whenever the draw
/// lands in the (2, 100) gap* (it does at this fixed seed — exact threshold value
/// is the documented #668 RNG boundary, not asserted here). sklearn's MAE leaf
/// value for {1,1,5} is its median 1.0 (`_criterion.pyx` MAE.node_value).
/// ferrolearn returns the MEAN (2.333) regardless of criterion, so no leaf equals
/// 1.0 => FAILS.
#[test]
fn pin3_reg_absolute_error_yields_median_leaf() {
    let x = Array2::from_shape_vec((6, 1), vec![0.0, 1.0, 2.0, 100.0, 101.0, 102.0]).unwrap();
    let y = array![1.0f64, 1.0, 5.0, 20.0, 20.0, 20.0];

    let f_mae = ExtraTreeRegressor::<f64>::new()
        .with_max_features(MaxFeatures::All)
        .with_criterion(RegressionCriterion::AbsoluteError)
        .with_max_depth(Some(1))
        .with_random_state(5)
        .fit(&x, &y)
        .unwrap();

    let leaf_vals = reg_leaves(&f_mae);

    // sklearn AbsoluteError: the {1,1,5} leaf value is the MEDIAN == 1.0.
    // ferrolearn returns the MEAN 7/3 ≈ 2.333 => no leaf equals 1.0 => FAILS.
    let median_of_small = 1.0_f64; // median of [1,1,5] per _criterion.pyx MAE.node_value
    let has_median_leaf = leaf_vals
        .iter()
        .any(|&v| (v - median_of_small).abs() < 1e-9);
    assert!(
        has_median_leaf,
        "AbsoluteError must yield the median leaf value 1.0 for the y={{1,1,5}} group \
         (sklearn _criterion.pyx MAE.node_value); ferrolearn returns the mean 2.333 \
         because the regression builder ignores `criterion` (#681). leaves={leaf_vals:?}"
    );
}
