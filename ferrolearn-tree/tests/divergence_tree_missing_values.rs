//! Native missing-value (NaN) support for `DecisionTreeClassifier` /
//! `DecisionTreeRegressor` — live scikit-learn 1.5.2 parity pins.
//!
//! scikit-learn's CART base accepts NaN in `X` (`force_all_finite=False`,
//! `sklearn/tree/_classes.py:248-250`) and handles missing values NATIVELY: the
//! best-splitter (`node_split_best`, `sklearn/tree/_splitter.pyx:293`) sorts NaN
//! to the end, then for each candidate threshold evaluates sending all missing
//! samples LEFT vs RIGHT, recording the better direction in
//! `tree_.missing_go_to_left` (`_tree.pyx:746`); at predict, a NaN feature value
//! routes to that direction (`_apply_dense`, `_tree.pyx:1015-1025`).
//!
//! Every expected value here is computed by a live `python3 -c` call against the
//! installed sklearn 1.5.2 oracle (R-CHAR-3) — NEVER copied from ferrolearn. The
//! `# oracle:` comment above each test records the exact command + result.

use ferrolearn_core::{Fit, Predict};
use ferrolearn_tree::{DecisionTreeClassifier, DecisionTreeRegressor, Node};
use ndarray::{Array1, Array2, array};

fn nan() -> f64 {
    f64::NAN
}

/// The root split's `(feature, threshold)` and whether it is a 3-node stump.
fn root_split(nodes: &[Node<f64>]) -> (usize, f64) {
    match nodes[0] {
        Node::Split {
            feature, threshold, ..
        } => (feature, threshold),
        Node::Leaf { .. } => panic!("root is a leaf, expected a split"),
    }
}

// oracle: python3 -c "import numpy as np; from sklearn.tree import DecisionTreeClassifier; \
//   nan=float('nan'); X=np.array([[1.],[2.],[nan],[8.],[9.],[nan]]); y=np.array([0,0,0,1,1,1]); \
//   c=DecisionTreeClassifier(max_depth=1).fit(X,y); \
//   print(c.tree_.node_count, c.tree_.feature[0], c.tree_.threshold[0], int(c.tree_.missing_go_to_left[0]), \
//   c.predict([[nan]]).tolist(), c.predict(X).tolist())"
//   -> 3 0 5.0 0 [1] [0, 0, 1, 1, 1, 1]
#[test]
fn clf_max_depth1_missing_go_right_oracle() {
    let x = Array2::from_shape_vec((6, 1), vec![1.0, 2.0, nan(), 8.0, 9.0, nan()]).unwrap();
    let y = array![0, 0, 0, 1, 1, 1];
    let fitted = DecisionTreeClassifier::<f64>::new()
        .with_max_depth(Some(1))
        .fit(&x, &y)
        .expect("fit must accept NaN (sklearn force_all_finite=False)");

    // node_count == 3 (a depth-1 stump).
    assert_eq!(fitted.nodes().len(), 3);
    // root split (feature 0, threshold 5.0) == sklearn.
    let (feat, thr) = root_split(fitted.nodes());
    assert_eq!(feat, 0);
    assert!((thr - 5.0).abs() < 1e-12, "threshold {thr} != 5.0");

    // missing_go_to_left = 0 ⇒ a NaN query routes RIGHT (class 1).
    let mut q = Array2::zeros((1, 1));
    q[[0, 0]] = nan();
    assert_eq!(fitted.predict(&q).unwrap()[0], 1);
    // Full predict (NaN rows route right, into class 1).
    assert_eq!(fitted.predict(&x).unwrap(), array![0, 0, 1, 1, 1, 1]);
}

// oracle: python3 -c "import numpy as np; from sklearn.tree import DecisionTreeClassifier; \
//   nan=float('nan'); X=np.array([[1.],[2.],[3.],[8.],[9.],[10.],[nan],[nan]]); y=np.array([0,0,0,1,1,1,0,0]); \
//   c=DecisionTreeClassifier(max_depth=1).fit(X,y); \
//   print(c.tree_.node_count, c.tree_.feature[0], c.tree_.threshold[0], int(c.tree_.missing_go_to_left[0]), \
//   c.predict([[nan]]).tolist(), c.predict(X).tolist())"
//   -> 3 0 5.5 1 [0] [0, 0, 0, 1, 1, 1, 0, 0]
#[test]
fn clf_max_depth1_missing_go_left_oracle() {
    let x =
        Array2::from_shape_vec((8, 1), vec![1.0, 2.0, 3.0, 8.0, 9.0, 10.0, nan(), nan()]).unwrap();
    let y = array![0, 0, 0, 1, 1, 1, 0, 0];
    let fitted = DecisionTreeClassifier::<f64>::new()
        .with_max_depth(Some(1))
        .fit(&x, &y)
        .expect("fit must accept NaN");

    assert_eq!(fitted.nodes().len(), 3);
    let (feat, thr) = root_split(fitted.nodes());
    assert_eq!(feat, 0);
    assert!((thr - 5.5).abs() < 1e-12, "threshold {thr} != 5.5");

    // missing_go_to_left = 1 ⇒ a NaN query routes LEFT (class 0, the group the
    // missing samples belong to).
    let mut q = Array2::zeros((1, 1));
    q[[0, 0]] = nan();
    assert_eq!(fitted.predict(&q).unwrap()[0], 0);
    assert_eq!(fitted.predict(&x).unwrap(), array![0, 0, 0, 1, 1, 1, 0, 0]);
}

// oracle: python3 -c "import numpy as np; from sklearn.tree import DecisionTreeClassifier; \
//   nan=float('nan'); X=np.array([[1.],[2.],[nan],[8.],[9.],[nan]]); y=np.array([0,0,0,1,1,1]); \
//   c=DecisionTreeClassifier().fit(X,y); \
//   print(c.tree_.node_count, c.predict(X).tolist(), c.predict_proba([[nan]]).tolist())"
//   -> 5 [0, 0, 0, 1, 1, 0] [[0.5, 0.5]]
#[test]
fn clf_deep_tree_threshold_inf_candidate_oracle() {
    // A full (unbounded-depth) tree: the right child of the root (the high-value
    // samples + the two mixed-class missing samples) further splits the missing
    // block off via the `threshold = +∞` "all-non-missing-left, all-missing-right"
    // candidate (`_splitter.pyx:498-519`).
    let x = Array2::from_shape_vec((6, 1), vec![1.0, 2.0, nan(), 8.0, 9.0, nan()]).unwrap();
    let y = array![0, 0, 0, 1, 1, 1];
    let fitted = DecisionTreeClassifier::<f64>::new()
        .fit(&x, &y)
        .expect("fit must accept NaN");

    assert_eq!(fitted.nodes().len(), 5);
    assert_eq!(fitted.predict(&x).unwrap(), array![0, 0, 0, 1, 1, 0]);
    // The NaN query lands in a 50/50 leaf (`predict_proba == [0.5, 0.5]`).
    let mut q = Array2::zeros((1, 1));
    q[[0, 0]] = nan();
    let proba = fitted.predict_proba(&q).unwrap();
    assert!(
        (proba[[0, 0]] - 0.5).abs() < 1e-12,
        "proba0 {}",
        proba[[0, 0]]
    );
    assert!(
        (proba[[0, 1]] - 0.5).abs() < 1e-12,
        "proba1 {}",
        proba[[0, 1]]
    );
}

// oracle: python3 -c "import numpy as np; from sklearn.tree import DecisionTreeRegressor; \
//   nan=float('nan'); X=np.array([[1.],[2.],[nan],[8.],[9.],[nan]]); y=np.array([1.,1.,1.,5.,5.,5.]); \
//   r=DecisionTreeRegressor(max_depth=1).fit(X,y); \
//   print(r.tree_.node_count, r.tree_.feature[0], r.tree_.threshold[0], int(r.tree_.missing_go_to_left[0]), \
//   r.predict([[nan]]).tolist(), r.predict(X).tolist())"
//   -> 3 0 5.0 0 [4.0] [1.0, 1.0, 4.0, 4.0, 4.0, 4.0]
#[test]
fn reg_max_depth1_missing_go_right_oracle() {
    let x = Array2::from_shape_vec((6, 1), vec![1.0, 2.0, nan(), 8.0, 9.0, nan()]).unwrap();
    let y: Array1<f64> = array![1.0, 1.0, 1.0, 5.0, 5.0, 5.0];
    let fitted = DecisionTreeRegressor::<f64>::new()
        .with_max_depth(Some(1))
        .fit(&x, &y)
        .expect("fit must accept NaN");

    assert_eq!(fitted.nodes().len(), 3);
    let (feat, thr) = root_split(fitted.nodes());
    assert_eq!(feat, 0);
    assert!((thr - 5.0).abs() < 1e-12, "threshold {thr} != 5.0");

    // missing_go_to_left = 0 ⇒ NaN routes to the right child, whose mean is
    // (5+5+1+5)/4 = 4.0 (two high-value samples + the two missing samples).
    let mut q = Array2::zeros((1, 1));
    q[[0, 0]] = nan();
    assert!((fitted.predict(&q).unwrap()[0] - 4.0).abs() < 1e-12);
    let preds = fitted.predict(&x).unwrap();
    for (p, e) in preds.iter().zip([1.0, 1.0, 4.0, 4.0, 4.0, 4.0]) {
        assert!((p - e).abs() < 1e-12, "{p} != {e}");
    }
}

// oracle: python3 -c "import numpy as np; from sklearn.tree import DecisionTreeRegressor; \
//   nan=float('nan'); X=np.array([[1.],[2.],[3.],[8.],[9.],[10.],[nan],[nan]]); y=np.array([1.,1.,1.,5.,5.,5.,1.,1.]); \
//   r=DecisionTreeRegressor(max_depth=1).fit(X,y); \
//   print(r.tree_.node_count, r.tree_.feature[0], r.tree_.threshold[0], int(r.tree_.missing_go_to_left[0]), \
//   r.predict([[nan]]).tolist(), r.predict(X).tolist())"
//   -> 3 0 5.5 1 [1.0] [1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 1.0, 1.0]
#[test]
fn reg_max_depth1_missing_go_left_oracle() {
    let x =
        Array2::from_shape_vec((8, 1), vec![1.0, 2.0, 3.0, 8.0, 9.0, 10.0, nan(), nan()]).unwrap();
    let y: Array1<f64> = array![1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 1.0, 1.0];
    let fitted = DecisionTreeRegressor::<f64>::new()
        .with_max_depth(Some(1))
        .fit(&x, &y)
        .expect("fit must accept NaN");

    assert_eq!(fitted.nodes().len(), 3);
    let (feat, thr) = root_split(fitted.nodes());
    assert_eq!(feat, 0);
    assert!((thr - 5.5).abs() < 1e-12, "threshold {thr} != 5.5");

    // missing_go_to_left = 1 ⇒ NaN routes LEFT (the low-value group, mean 1.0).
    let mut q = Array2::zeros((1, 1));
    q[[0, 0]] = nan();
    assert!((fitted.predict(&q).unwrap()[0] - 1.0).abs() < 1e-12);
}

// oracle: python3 -c "import numpy as np; from sklearn.tree import DecisionTreeClassifier; \
//   nan=float('nan'); X=np.array([[nan,1.],[2.,1.],[nan,1.],[8.,9.],[9.,9.],[10.,9.]]); y=np.array([0,0,0,1,1,1]); \
//   c=DecisionTreeClassifier(max_depth=1).fit(X,y); \
//   print(c.tree_.node_count, c.tree_.feature[0], c.tree_.threshold[0], int(c.tree_.missing_go_to_left[0]), \
//   c.predict(X).tolist(), c.predict([[nan,1.],[nan,9.],[5.,5.]]).tolist())"
//   -> 3 0 5.0 1 [0, 0, 0, 1, 1, 1] [0, 0, 0]
#[test]
fn clf_multi_feature_missing_in_one_feature_oracle() {
    // Feature 0 has missing values; feature 1 is clean. sklearn still splits on
    // feature 0 (threshold 5.0, missing→left) — the missing samples belong with
    // the low-value class-0 group.
    let x = Array2::from_shape_vec(
        (6, 2),
        vec![
            nan(),
            1.0,
            2.0,
            1.0,
            nan(),
            1.0,
            8.0,
            9.0,
            9.0,
            9.0,
            10.0,
            9.0,
        ],
    )
    .unwrap();
    let y = array![0, 0, 0, 1, 1, 1];
    let fitted = DecisionTreeClassifier::<f64>::new()
        .with_max_depth(Some(1))
        .fit(&x, &y)
        .expect("fit must accept NaN");

    assert_eq!(fitted.nodes().len(), 3);
    let (feat, thr) = root_split(fitted.nodes());
    assert_eq!(feat, 0);
    assert!((thr - 5.0).abs() < 1e-12, "threshold {thr} != 5.0");

    assert_eq!(fitted.predict(&x).unwrap(), array![0, 0, 0, 1, 1, 1]);
    // Query: NaN-on-feature-0 routes left (class 0) regardless of feature 1.
    let q = Array2::from_shape_vec((3, 2), vec![nan(), 1.0, nan(), 9.0, 5.0, 5.0]).unwrap();
    assert_eq!(fitted.predict(&q).unwrap(), array![0, 0, 0]);
}

// oracle: python3 -c "import numpy as np; from sklearn.tree import DecisionTreeClassifier; \
//   nan=float('nan'); X=np.array([[1.,2.],[2.,3.],[3.,3.],[5.,6.],[6.,7.],[7.,8.]]); X[0,0]=nan; \
//   y=np.array([0,0,0,1,1,1]); c=DecisionTreeClassifier().fit(X,y); \
//   print(c.tree_.node_count, c.predict(X).tolist())"
//   -> 3 [0, 0, 0, 1, 1, 1]
#[test]
fn clf_2277_fixture_fit_succeeds_and_predicts_oracle() {
    // The exact #2277 stack-overflow fixture: a single NaN in X[0,0]. ferrolearn
    // now FITS and PREDICTS instead of aborting. On this fixture BOTH features
    // separate the classes perfectly; sklearn picks the clean feature 1 (its
    // `random_state`-permuted feature order), ferrolearn's deterministic
    // feature-0-first scan picks feature 0 — the documented RNG feature-order
    // boundary (REQ-2 / #660 / #670), NOT a missing-value defect. The CONTRACT
    // — `predict == sklearn` (a 3-node stump, perfect class separation) — holds.
    let mut x = Array2::from_shape_vec(
        (6, 2),
        vec![1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 5.0, 6.0, 6.0, 7.0, 7.0, 8.0],
    )
    .unwrap();
    x[[0, 0]] = nan();
    let y = array![0, 0, 0, 1, 1, 1];
    let fitted = DecisionTreeClassifier::<f64>::new()
        .fit(&x, &y)
        .expect("fit must accept NaN (#2277), not abort");

    assert_eq!(fitted.nodes().len(), 3);
    assert!(matches!(fitted.nodes()[0], Node::Split { .. }));
    assert_eq!(fitted.predict(&x).unwrap(), array![0, 0, 0, 1, 1, 1]);
}

// oracle: python3 -c "import numpy as np; from sklearn.tree import DecisionTreeRegressor; \
//   nan=float('nan'); X=np.array([[1.,2.],[2.,3.],[3.,3.],[5.,6.],[6.,7.],[7.,8.]]); X[0,0]=nan; \
//   y=np.array([1.,2.,3.,4.,5.,6.]); r=DecisionTreeRegressor().fit(X,y); \
//   print(r.predict(X).tolist())"
//   -> [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
#[test]
fn reg_2277_fixture_fit_succeeds_and_predicts_oracle() {
    // Same #2277 fixture as the classifier: fit succeeds (no abort), each sample
    // lands in its own leaf, `predict == y`. (Feature choice is the documented
    // RNG boundary; the prediction contract holds.)
    let mut x = Array2::from_shape_vec(
        (6, 2),
        vec![1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 5.0, 6.0, 6.0, 7.0, 7.0, 8.0],
    )
    .unwrap();
    x[[0, 0]] = nan();
    let y: Array1<f64> = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let fitted = DecisionTreeRegressor::<f64>::new()
        .fit(&x, &y)
        .expect("fit must accept NaN (#2277), not abort");

    let preds = fitted.predict(&x).unwrap();
    for (p, e) in preds.iter().zip([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) {
        assert!((p - e).abs() < 1e-12, "{p} != {e}");
    }
}

/// All-finite data is unaffected: the tree is identical to the pre-missing-value
/// build (no NaN branch ever taken). Root split + predictions match the
/// long-standing `divergence_decision_tree` oracle (`node_count == 7`, root
/// `(feature 1, threshold 5.5)`).
// oracle: python3 -c "import numpy as np; from sklearn.tree import DecisionTreeClassifier; \
//   X=np.array([[1,2],[2,3],[3,3],[5,6],[6,7],[7,8],[1.5,5],[6.5,2],[3,1]]); y=np.array([0,0,0,1,1,1,2,2,0]); \
//   c=DecisionTreeClassifier(random_state=0).fit(X,y); \
//   print(c.tree_.node_count, c.tree_.feature[0], c.tree_.threshold[0], c.predict(X).tolist())"
//   -> 7 1 5.5 [0, 0, 0, 1, 1, 1, 2, 2, 0]
#[test]
fn clf_all_finite_byte_identical_oracle() {
    let x = Array2::from_shape_vec(
        (9, 2),
        vec![
            1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 5.0, 6.0, 6.0, 7.0, 7.0, 8.0, 1.5, 5.0, 6.5, 2.0, 3.0,
            1.0,
        ],
    )
    .unwrap();
    let y = array![0, 0, 0, 1, 1, 1, 2, 2, 0];
    let fitted = DecisionTreeClassifier::<f64>::new().fit(&x, &y).unwrap();
    assert_eq!(fitted.nodes().len(), 7);
    let (feat, thr) = root_split(fitted.nodes());
    assert_eq!(feat, 1);
    assert!((thr - 5.5).abs() < 1e-12, "threshold {thr} != 5.5");
    assert_eq!(
        fitted.predict(&x).unwrap(),
        array![0, 0, 0, 1, 1, 1, 2, 2, 0]
    );
}
