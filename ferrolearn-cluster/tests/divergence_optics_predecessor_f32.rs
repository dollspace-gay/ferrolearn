//! Divergence pin: `ferrolearn_cluster::FittedOPTICS::predecessor()` (REQ-4,
//! `-1`-sentinel `predecessor_` int array) diverges from scikit-learn 1.5.2
//! `sklearn.cluster.OPTICS.predecessor_` on `f32` input.
//!
//! Iteration 132 (critic). The builder's REQ-4 row claims `predecessor_`
//! value-parity (incl. the `-1` seed) and that the strict-`<` improvement rule
//! makes it tie-robust — but all of its green guards
//! (`green_predecessor_three_blobs/small10/docstring` in `divergence_optics.rs`)
//! are `f64` only. On `f32` the predecessor assignment diverges.
//!
//! Expected value is the LIVE sklearn 1.5.2 oracle (R-CHAR-3), NEVER copied from
//! ferrolearn:
//!   python3 -c "import numpy as np; from sklearn.cluster import OPTICS; \
//!     X=np.array([[2.1,0.3],[1.5,0.6],[0.5,-0.8],[0.9,0.2],[-1.9,-0.6],\
//!       [-0.1,0.8],[-0.6,0.6],[-0.3,0.3],[-0.4,0.2],[0.7,0.8]],dtype=np.float32); \
//!     print(OPTICS(min_samples=2).fit(X).predecessor_.tolist())"
//!   -> [-1, 0, 3, 1, 8, 9, 7, 5, 7, 3]
//!
//! ferrolearn produces `[-1, 0, 3, 1, 8, 9, 5, 6, 7, 3]` — diverges at indices
//! 6, 7, 8. Root cause: ferrolearn's `f32` reachability for points 6 and 7 ends
//! up swapped relative to sklearn (ferro reach[6]≈0.5385/reach[7]≈0.4243; sklearn
//! reach[6]≈0.4243/reach[7]≈0.5385), so the `f32` traversal visits 6/7 in a
//! different order and the strict-`<` predecessor assignment
//! (`sklearn/cluster/_optics.py:712` `improved = np.where(rdists < ...)`,
//! `:714` `predecessor_[unproc[improved]] = point_index`) records different
//! predecessors. The same fixture matches sklearn exactly in `f64`.
//!
//! Tracking: #2195

use ferrolearn_cluster::OPTICS;
use ferrolearn_core::Fit;
use ndarray::Array2;

/// The 10-point fixture from `divergence_optics.rs::small10`, but in `f32`.
fn small10_f32() -> Array2<f32> {
    Array2::from_shape_vec(
        (10, 2),
        vec![
            2.1f32, 0.3, 1.5, 0.6, 0.5, -0.8, 0.9, 0.2, -1.9, -0.6, -0.1, 0.8, -0.6, 0.6, -0.3,
            0.3, -0.4, 0.2, 0.7, 0.8,
        ],
    )
    .unwrap()
}

/// Divergence: `FittedOPTICS::<f32>::predecessor()` diverges from
/// `sklearn/cluster/_optics.py:712-714` for the small10 fixture on `f32`.
/// sklearn returns `[-1,0,3,1,8,9,7,5,7,3]`; ferrolearn returns
/// `[-1,0,3,1,8,9,5,6,7,3]` (differs at indices 6, 7, 8).
/// Tracking: #2195
#[test]
#[ignore = "divergence: OPTICS predecessor_ diverges from sklearn on f32 (indices 6,7,8); tracking #2195"]
fn divergence_predecessor_small10_f32() {
    // LIVE sklearn 1.5.2 oracle (header), never copied from ferrolearn.
    let sk_pred: [i64; 10] = [-1, 0, 3, 1, 8, 9, 7, 5, 7, 3];

    let fitted = OPTICS::<f32>::new(2).fit(&small10_f32(), &()).unwrap();
    let pred = fitted.predecessor();

    assert_eq!(pred.len(), 10, "predecessor_ must be shape (n_samples,)");
    for (i, &sk) in sk_pred.iter().enumerate() {
        assert_eq!(
            pred[i], sk,
            "predecessor_[{i}]: ferro={} sklearn={sk} (f32, sklearn/cluster/_optics.py:714)",
            pred[i]
        );
    }
}
