//! Adversarial divergence pins for `ferrolearn-neighbors/src/nearest_centroid.rs`
//! (`pub struct NearestCentroid`, `NearestCentroid::fit`, `FittedNearestCentroid::predict`,
//! `centroids`, `classes`) against the live scikit-learn 1.5.2 oracle
//! (`from sklearn.neighbors import NearestCentroid`).
//!
//! Every expected value below is captured from a live `python3 -c "..."` run of
//! sklearn 1.5.2 (R-CHAR-3 — never literal-copied from the ferrolearn side).
//! The exact oracle call and its output is quoted above each assertion.
//!
//! Module REQ table (`.design/neighbors/nearest_centroid.md`):
//!   REQ-2 (euclidean centroid + predict) SHIPPED; REQ-5 (classes_/centroids_) SHIPPED;
//!   REQ-3 (shrink `s += median(s)`) NOT-STARTED (#837); zero-variance (#839);
//!   REQ-6 (n_classes < 2 guard) NOT-STARTED (#838).
//!
//! Upstream sklearn 1.5.2 (`sklearn/neighbors/_nearest_centroid.py`):
//!   * `:171` — euclidean centroid `self.centroids_[cur_class] = X[center_mask].mean(axis=0)`.
//!   * `:183-184` — `s = np.sqrt(variance / (n_samples - n_classes))` then
//!     `s += np.median(s)  # To deter outliers from affecting the results.`
//!   * `:185-196` — `ms = m.reshape(-1,1) * s`; `deviation = (centroids_ - dataset_centroid_) / ms`;
//!     soft-threshold; `centroids_ = dataset_centroid_ + ms * deviation`.
//!   * `:147-151` — `if n_classes < 2: raise ValueError("The number of classes has
//!     to be greater than one; got %d class" % n_classes)`.
//!   * `:174-175` — `if np.all(np.ptp(X, axis=0) == 0): raise ValueError("All
//!     features have zero variance. Division by zero.")`.
//!   * `:217-219` — `predict` = `classes_[pairwise_distances_argmin(X, centroids_, metric)]`.
//!
//! ferrolearn API under test:
//!   * `NearestCentroid::<f64>::new()` / `.with_shrink_threshold(t)` (builder).
//!   * `Fit::fit(&x, &y) -> Result<FittedNearestCentroid<f64>, FerroError>`.
//!   * `FittedNearestCentroid::centroids() -> &Array2<f64>`,
//!     `FittedNearestCentroid::predict(&x) -> Result<Array1<usize>, _>`,
//!     `HasClasses::classes() -> &[usize]`.

use ferrolearn_core::introspection::HasClasses;
use ferrolearn_core::traits::{Fit, Predict};
use ferrolearn_neighbors::nearest_centroid::NearestCentroid;
use ndarray::{Array1, Array2, array};

/// AC-2 shared fixture: two well-separated 2-D clusters, 4 samples each.
fn ac2_data() -> (Array2<f64>, Array1<usize>) {
    let x = Array2::from_shape_vec(
        (8, 2),
        vec![
            0.0, 0.0, 0.5, 0.0, 0.0, 0.5, 0.5, 0.5, 5.0, 5.0, 5.5, 5.0, 5.0, 5.5, 5.5, 5.5,
        ],
    )
    .unwrap();
    let y = array![0usize, 0, 0, 0, 1, 1, 1, 1];
    (x, y)
}

// ===========================================================================
// RED pins — deterministic divergences to be fixed this iteration.
// ===========================================================================

/// Divergence (HEADLINE, R-DEV-1): ferrolearn's within-class std `pooled_var`
/// (`nearest_centroid.rs:191-197`) OMITS sklearn's `s += np.median(s)`
/// regularization (`_nearest_centroid.py:184`), so every shrunken centroid value
/// diverges.
///
/// Live oracle (sklearn 1.5.2, run from /tmp):
/// ```text
/// python3 -c "
/// import numpy as np; from sklearn.neighbors import NearestCentroid as N
/// X=np.array([[0,0],[0.5,0],[0,0.5],[0.5,0.5],[5,5],[5.5,5],[5,5.5],[5.5,5.5]],dtype=float)
/// y=np.array([0,0,0,0,1,1,1,1])
/// print(N(shrink_threshold=0.5).fit(X,y).centroids_.tolist())"
/// # [[0.3520620726159658, 0.3520620726159658], [5.147937927384034, 5.147937927384034]]
/// ```
/// ferrolearn's no-median formula instead yields
/// `[[0.30103103630798, ...], [5.19896896369202, ...]]` (verified by reproducing
/// the ferrolearn arithmetic in numpy). This test asserts the sklearn value and
/// MUST FAIL against current `nearest_centroid.rs`.
/// Tracking: #837
#[test]
fn divergence_shrink_threshold_median_add() {
    // sklearn 1.5.2 oracle values (s += median(s) path):
    const SK_C0: f64 = 0.3520620726159658;
    const SK_C1: f64 = 5.147937927384034;

    let (x, y) = ac2_data();
    let fitted = NearestCentroid::<f64>::new()
        .with_shrink_threshold(0.5)
        .fit(&x, &y)
        .unwrap();
    let c = fitted.centroids();

    // Both features of each class share the symmetric value by construction.
    let max_err = [
        (c[[0, 0]] - SK_C0).abs(),
        (c[[0, 1]] - SK_C0).abs(),
        (c[[1, 0]] - SK_C1).abs(),
        (c[[1, 1]] - SK_C1).abs(),
    ]
    .into_iter()
    .fold(0.0_f64, f64::max);

    assert!(
        max_err < 1e-9,
        "shrunken centroids must match sklearn (s += median(s), :184); \
         expected ~[[{SK_C0},..],[{SK_C1},..]], got {c:?} (max_err={max_err:e})"
    );
}

/// Divergence (R-DEV-2): sklearn raises `ValueError` when fewer than two classes
/// are seen (`_nearest_centroid.py:147-151`); ferrolearn fits a single-class model
/// and returns `Ok`.
///
/// Live oracle (sklearn 1.5.2, run from /tmp):
/// ```text
/// python3 -c "from sklearn.neighbors import NearestCentroid as N; import numpy as np
/// try: N().fit(np.array([[1,1],[1.5,1],[1,1.5]],dtype=float),np.array([5,5,5]))
/// except Exception as e: print(type(e).__name__,'|',str(e))"
/// # ValueError | The number of classes has to be greater than one; got 1 class
/// ```
/// This test asserts ferrolearn `fit` returns `Err` for a single-class `y` and
/// MUST FAIL against current `nearest_centroid.rs`.
/// Tracking: #838
#[test]
fn divergence_single_class_must_err() {
    let x = Array2::from_shape_vec((3, 2), vec![1.0, 1.0, 1.5, 1.0, 1.0, 1.5]).unwrap();
    let y = array![5usize, 5, 5];

    let result = NearestCentroid::<f64>::new().fit(&x, &y);
    assert!(
        result.is_err(),
        "sklearn raises ValueError on a single class (:147-151); \
         ferrolearn must return Err, got Ok"
    );
}

/// Divergence (R-DEV-2): with `shrink_threshold` set and all features constant,
/// sklearn raises `ValueError("All features have zero variance. Division by
/// zero.")` (`_nearest_centroid.py:174-175`). ferrolearn clamps the zero std to
/// `1.0` (`nearest_centroid.rs:194-196`) and returns `Ok`.
///
/// Live oracle (sklearn 1.5.2, run from /tmp):
/// ```text
/// python3 -c "from sklearn.neighbors import NearestCentroid as N; import numpy as np
/// try: N(shrink_threshold=0.5).fit(np.array([[1,1],[1,1],[1,1],[1,1]],dtype=float),np.array([0,0,1,1]))
/// except Exception as e: print(type(e).__name__,'|',str(e))"
/// # ValueError | All features have zero variance. Division by zero.
/// ```
/// This test asserts ferrolearn `fit` returns `Err` for constant `X` + shrink and
/// MUST FAIL against current `nearest_centroid.rs`.
/// Tracking: #839
#[test]
fn divergence_zero_variance_shrink_must_err() {
    let x = Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).unwrap();
    let y = array![0usize, 0, 1, 1];

    let result = NearestCentroid::<f64>::new()
        .with_shrink_threshold(0.5)
        .fit(&x, &y);
    assert!(
        result.is_err(),
        "sklearn raises ValueError on all-zero-variance X + shrink (:174-175); \
         ferrolearn must return Err, got Ok"
    );
}

// ===========================================================================
// GREEN guards — SHIPPED euclidean centroid + predict + classes_ contract.
// These must PASS now (oracle-grounded, R-CHAR-3).
// ===========================================================================

/// REQ-2/REQ-5 SHIPPED: euclidean per-class mean centroids + classes_.
///
/// Live oracle (sklearn 1.5.2, run from /tmp):
/// ```text
/// python3 -c "
/// import numpy as np; from sklearn.neighbors import NearestCentroid as N
/// X=np.array([[0,0],[0.5,0],[0,0.5],[0.5,0.5],[5,5],[5.5,5],[5,5.5],[5.5,5.5]],dtype=float)
/// y=np.array([0,0,0,0,1,1,1,1]); c=N().fit(X,y)
/// print(c.centroids_.tolist(), c.classes_.tolist())"
/// # [[0.25, 0.25], [5.25, 5.25]] [0, 1]
/// ```
#[test]
fn green_euclidean_centroids_and_classes() {
    let (x, y) = ac2_data();
    let fitted = NearestCentroid::<f64>::new().fit(&x, &y).unwrap();
    let c = fitted.centroids();

    assert_eq!(fitted.classes(), &[0usize, 1]);
    assert_eq!(c.dim(), (2, 2));
    for &v in &[c[[0, 0]], c[[0, 1]]] {
        assert!((v - 0.25).abs() < 1e-12, "class-0 centroid: {v}");
    }
    for &v in &[c[[1, 0]], c[[1, 1]]] {
        assert!((v - 5.25).abs() < 1e-12, "class-1 centroid: {v}");
    }
}

/// REQ-2 SHIPPED: nearest-centroid predict (euclidean argmin).
///
/// Live oracle (sklearn 1.5.2, run from /tmp):
/// ```text
/// python3 -c "
/// import numpy as np; from sklearn.neighbors import NearestCentroid as N
/// X=np.array([[0,0],[0.5,0],[0,0.5],[0.5,0.5],[5,5],[5.5,5],[5,5.5],[5.5,5.5]],dtype=float)
/// y=np.array([0,0,0,0,1,1,1,1]); print(N().fit(X,y).predict(X).tolist())"
/// # [0, 0, 0, 0, 1, 1, 1, 1]
/// ```
#[test]
fn green_euclidean_predict() {
    let (x, y) = ac2_data();
    let fitted = NearestCentroid::<f64>::new().fit(&x, &y).unwrap();
    let preds = fitted.predict(&x).unwrap();
    assert_eq!(preds.to_vec(), vec![0usize, 0, 0, 0, 1, 1, 1, 1]);
}

// ===========================================================================
// RE-AUDIT pin (post #837/#838/#839): partial-constant + shrink centroids.
// ===========================================================================

/// Divergence (R-DEV-1): with `shrink_threshold` set and a PARTIALLY-constant `X`
/// (one feature constant, others not), ferrolearn's per-feature pooled-std clamp
/// `if pooled_var[j] < 1e-10 { pooled_var[j] = 1.0 }`
/// (`nearest_centroid.rs:226-227`) runs BEFORE the `s += median(s)` step
/// (`:231-248`). sklearn has NO such clamp: it leaves a constant feature's
/// `s = sqrt(variance/(n-c)) = 0` (`_nearest_centroid.py:183`) and only then does
/// `s += np.median(s)` (`:184`). For `X` with feature-0 constant and feature-1
/// separated, sklearn's `median(s) = median([0, 0.6103...]) = 0.3052...` but
/// ferrolearn's clamp makes it `median([1.0, 0.6103...]) = 0.8052...`, scaling
/// every shrunken centroid differently — so feature-1 centroids diverge.
///
/// sklearn only raises on ALL features constant (`np.all(np.ptp==0)`, `:174`);
/// a partial-constant `X` is a valid fit that must match the oracle.
///
/// Live oracle (sklearn 1.5.2, run from /tmp):
/// ```text
/// python3 -c "
/// import numpy as np; from sklearn.neighbors import NearestCentroid as N
/// X=np.array([[3.0,0.0],[3.0,0.5],[3.0,1.0],[3.0,9.0],[3.0,9.5],[3.0,10.0]],dtype=float)
/// y=np.array([0,0,0,1,1,1])
/// print(N(shrink_threshold=0.2).fit(X,y).centroids_[:,1].tolist())"
/// # [0.5612372435695798, 9.438762756430421]
/// ```
/// ferrolearn (verified directly in Rust) yields feature-1 centroids
/// `[0.6020620726159649, 9.397937927384035]`. This test asserts the sklearn
/// values and MUST FAIL against current `nearest_centroid.rs`.
/// Tracking: #840
#[test]
fn divergence_partial_constant_shrink_centroids() {
    // sklearn 1.5.2 oracle (feature-1 of each class centroid after shrink):
    const SK_C0_F1: f64 = 0.5612372435695798;
    const SK_C1_F1: f64 = 9.438762756430421;

    // Feature 0 is constant (3.0 for all rows); feature 1 separates the classes.
    let x = Array2::from_shape_vec(
        (6, 2),
        vec![3.0, 0.0, 3.0, 0.5, 3.0, 1.0, 3.0, 9.0, 3.0, 9.5, 3.0, 10.0],
    )
    .unwrap();
    let y = array![0usize, 0, 0, 1, 1, 1];

    let fitted = NearestCentroid::<f64>::new()
        .with_shrink_threshold(0.2)
        .fit(&x, &y)
        .expect("partial-constant X is a valid fit; sklearn only errors on ALL-constant X");
    let c = fitted.centroids();

    let err0 = (c[[0, 1]] - SK_C0_F1).abs();
    let err1 = (c[[1, 1]] - SK_C1_F1).abs();

    assert!(
        err0 < 1e-9 && err1 < 1e-9,
        "partial-constant shrunken centroids must match sklearn (no pre-median \
         std clamp; :183-184); expected feature-1 [{SK_C0_F1}, {SK_C1_F1}], got \
         [{}, {}] (err0={err0:e}, err1={err1:e})",
        c[[0, 1]],
        c[[1, 1]]
    );
}
