//! Divergence pins for the #2139 KNN tie-break claim.
//!
//! #2139 routes every `algorithm` variant of
//! `FittedKNeighborsClassifier`/`Regressor` through
//! `kdtree::brute_force_knn`, which selects the k nearest with a STABLE
//! `sort_by (distance asc, then index asc)` and claims this reproduces
//! sklearn's `np.argpartition` + `np.argsort` selection
//! (`sklearn/neighbors/_base.py:738`, `:740-741`) with a "LOWEST-INDEX"
//! tie-break that is "identical for every `algorithm`".
//!
//! The algorithm-invariance half of the claim holds on the ferrolearn side
//! (the in-module `test_knn_tie_break_lowest_index_all_algorithms` is green).
//! The "matches sklearn (lowest index)" half is FALSE: sklearn's
//! `np.argpartition` does NOT select the lowest tied index when a strictly
//! closer (fp-distinct) point pushes the k-th boundary into an exact-distance
//! tie set. ferrolearn's lowest-index rule then selects a DIFFERENT set than
//! sklearn, flipping the observable `predict` and `predict_proba`.
//!
//! Tracking: #2141.

use ferrolearn_core::traits::{Fit, Predict};
use ferrolearn_neighbors::{Algorithm, KNeighborsClassifier};
use ndarray::{Array1, Array2, array};

/// The shared tie fixture. idx0 is the origin (distance 0). idx5 and idx6 are
/// at distance sqrt(2 * 0.7071^2) = 0.99999041 — STRICTLY closer than the four
/// unit-axis points idx1..idx4, which are at EXACTLY distance 1.0. So the k=4
/// neighbor set is { idx0, idx5, idx6, ONE of the {1,2,3,4} tie }.
fn fixture() -> (Array2<f64>, Array1<usize>, Array2<f64>) {
    let x = array![
        [0.0, 0.0],     // idx0  dist 0
        [1.0, 0.0],     // idx1  dist 1.0  (exact tie)
        [-1.0, 0.0],    // idx2  dist 1.0  (exact tie)
        [0.0, 1.0],     // idx3  dist 1.0  (exact tie)
        [0.0, -1.0],    // idx4  dist 1.0  (exact tie)
        [0.7071, 0.7071],   // idx5  dist 0.99999041  (strictly closer)
        [-0.7071, 0.7071],  // idx6  dist 0.99999041  (strictly closer)
    ];
    // Labels chosen so the tie-resolved 4th member decides the vote:
    // idx0=0, idx1=0, idx2=1, idx5=1, idx6=1 (idx3/idx4 unused at k=4).
    let y = array![0usize, 0, 1, 9, 9, 1, 1];
    let xq = array![[0.0, 0.0]];
    (x, y, xq)
}

/// Divergence: `FittedKNeighborsClassifier::kneighbors` selects the LOWEST
/// tied index (idx1) at the k=4 boundary, but sklearn's `np.argpartition`
/// (`sklearn/neighbors/_base.py:738`) selects idx2 — for brute AND kd_tree AND
/// ball_tree AND auto.
///
/// Live sklearn 1.5.2 oracle (system python3):
/// ```text
/// x=[[0,0],[1,0],[-1,0],[0,1],[0,-1],[0.7071,0.7071],[-0.7071,0.7071]]
/// for algo in brute/kd_tree/ball_tree/auto:
///   KNeighborsClassifier(n_neighbors=4, algorithm=algo).fit(x,y).kneighbors([[0,0]])
///   -> indices [0, 6, 5, 2]   (4th member = idx2, NOT idx1)
/// ```
/// ferrolearn returns the set {0, 1, 5, 6} (4th member = lowest index idx1).
///
/// Tracking: #2141
#[test]
#[ignore = "divergence: KNN tie-break is not sklearn's argpartition (picks lowest-index idx1, sklearn picks idx2); tracking #2141"]
fn divergence_knn_tiebreak_set_differs_from_sklearn() {
    let (x, y, xq) = fixture();
    // sklearn's selected SET at k=4 (oracle, all algorithms): {0, 2, 5, 6}.
    let sklearn_set: Vec<usize> = vec![0, 2, 5, 6];

    for algo in [
        Algorithm::BruteForce,
        Algorithm::KdTree,
        Algorithm::BallTree,
        Algorithm::Auto,
    ] {
        let fitted = KNeighborsClassifier::<f64>::new()
            .with_n_neighbors(4)
            .with_algorithm(algo)
            .fit(&x, &y)
            .unwrap();
        let (_d, indices) = fitted.kneighbors(&xq, Some(4)).unwrap();
        let mut got: Vec<usize> = indices.row(0).to_vec();
        got.sort_unstable();
        assert_eq!(
            got, sklearn_set,
            "{algo:?}: kneighbors must select sklearn's argpartition set \
             {sklearn_set:?} (idx2 at the tie boundary), not the lowest-index \
             set; sklearn/neighbors/_base.py:738"
        );
    }
}

/// Divergence: the tie-set difference (idx2 vs idx1) flips the OBSERVABLE
/// `predict` and `predict_proba`.
///
/// Live sklearn 1.5.2 oracle (system python3), labels y=[0,0,1,9,9,1,1]:
/// ```text
/// KNeighborsClassifier(n_neighbors=4).fit(x,y)
///   .predict([[0,0]])        -> [1]
///   .predict_proba([[0,0]])  -> [[0.25, 0.75, 0.0]]   classes_=[0,1,9]
/// ```
/// ferrolearn (all algorithms) selects {0,1,5,6} (labels {0,0,1,1}) and so
/// returns predict=0, predict_proba=[0.5, 0.5, 0.0].
///
/// Tracking: #2141
#[test]
#[ignore = "divergence: KNN tie-break flips predict/predict_proba vs sklearn; tracking #2141"]
fn divergence_knn_tiebreak_flips_predict() {
    let (x, y, xq) = fixture();
    // sklearn oracle.
    let sklearn_pred: usize = 1;
    let sklearn_proba: [f64; 3] = [0.25, 0.75, 0.0]; // classes_ = [0, 1, 9]

    for algo in [
        Algorithm::BruteForce,
        Algorithm::KdTree,
        Algorithm::BallTree,
        Algorithm::Auto,
    ] {
        let fitted = KNeighborsClassifier::<f64>::new()
            .with_n_neighbors(4)
            .with_algorithm(algo)
            .fit(&x, &y)
            .unwrap();

        let pred = fitted.predict(&xq).unwrap();
        assert_eq!(
            pred[0], sklearn_pred,
            "{algo:?}: predict must match sklearn (class 1); the tie boundary \
             selects idx2 (class 1), not lowest-index idx1 (class 0)"
        );

        let proba = fitted.predict_proba(&xq).unwrap();
        for (c, &expected) in sklearn_proba.iter().enumerate() {
            let got = proba[[0, c]];
            assert!(
                (got - expected).abs() < 1e-9,
                "{algo:?}: predict_proba[{c}] = {got}, sklearn = {expected} \
                 (tie boundary picks idx2/class1, shifting the vote)"
            );
        }
    }
}
