//! Divergence pin: `FittedIsomap::transform` (out-of-sample) vs scikit-learn
//! 1.5.2 `Isomap.transform` (`sklearn/manifold/_isomap.py:386-435`).
//!
//! sklearn links each new point INTO the geodesic graph of the training data:
//! it finds the new point's `n_neighbors` nearest TRAINING neighbors, then
//! builds the geodesic kernel row
//!
//! ```text
//! G_X[i] = np.min(self.dist_matrix_[indices[i]] + distances[i][:, None], 0)
//! ```
//! (`_isomap.py:430`), squares + halves it (`:432-433`), and projects via
//! `kernel_pca_.transform(G_X)` (`:435`). The geodesic distance from the new
//! point to a far training point is the shortest path THROUGH the manifold
//! graph — NOT the raw Euclidean chord.
//!
//! ferrolearn's `FittedIsomap::transform` (`ferrolearn-decomp/src/isomap.rs:479-553`)
//! instead uses the RAW Euclidean distance from the new point to EVERY training
//! point (`isomap.rs:502-517`) plugged into a hand-rolled double-centering
//! Nystroem formula (`isomap.rs:535-549`). This is a different algorithm: it
//! never routes through the geodesic graph, so its output diverges from
//! sklearn's even after eigenvector-sign alignment.
//!
//! All expected values come from the live sklearn 1.5.2 oracle (run in /tmp),
//! hard-coded at full precision (R-CHAR-3), never copied from ferrolearn.
//!
//! Oracle command:
//! ```text
//! python3 -W ignore -c "import numpy as np; from sklearn.manifold import Isomap
//! X=np.array([[0.,0.,0.],[1.,0.1,0.],[2.,0.3,0.1],[3.,0.2,0.],[0.5,1.,0.2],
//!   [1.5,1.1,0.1],[2.5,0.9,0.3],[3.5,1.2,0.2],[0.2,2.,0.],[1.2,2.1,0.1]])
//! iso=Isomap(n_components=2, n_neighbors=4).fit(X)
//! Xnew=np.array([[0.5,0.5,0.1],[2.5,1.5,0.2],[1.0,1.5,0.1]])
//! print(iso.transform(Xnew).tolist())"
//! ```
//!
//! Tracking: #2401.

use ferrolearn_core::traits::{Fit, Transform};
use ferrolearn_decomp::Isomap;
use ndarray::{Array2, array};

/// The 10x3 training fixture (shared with `divergence_isomap.rs`).
fn fixture() -> Array2<f64> {
    array![
        [0.0, 0.0, 0.0],
        [1.0, 0.1, 0.0],
        [2.0, 0.3, 0.1],
        [3.0, 0.2, 0.0],
        [0.5, 1.0, 0.2],
        [1.5, 1.1, 0.1],
        [2.5, 0.9, 0.3],
        [3.5, 1.2, 0.2],
        [0.2, 2.0, 0.0],
        [1.2, 2.1, 0.1],
    ]
}

/// Out-of-sample query points.
fn x_new() -> Array2<f64> {
    array![[0.5, 0.5, 0.1], [2.5, 1.5, 0.2], [1.0, 1.5, 0.1]]
}

/// sklearn 1.5.2 oracle:
/// `Isomap(n_components=2, n_neighbors=4).fit(X).transform(Xnew)`,
/// shape `(3, 2)`, full precision (`_isomap.py:435`). R-CHAR-3.
#[allow(
    clippy::excessive_precision,
    reason = "full-precision sklearn 1.5.2 oracle value, _isomap.py:435"
)]
const SK_TRANSFORM: [[f64; 2]; 3] = [
    [-1.0473976086554222, -0.530035596827322],
    [0.9958091326147656, 0.4417260694522937],
    [-0.6138743183537207, 0.6373850940688576],
];

/// Sign of the max-absolute-value entry of a column (sklearn `svd_flip`
/// convention, `extmath.py:888-896`). Used to compare MAGNITUDES independent of
/// the per-column eigenvector sign, so the assertion below isolates a genuine
/// VALUE divergence (not a mere sign flip).
fn max_abs_positive_sign(col: &[f64]) -> f64 {
    let mut best = 0usize;
    for (i, &v) in col.iter().enumerate() {
        if v.abs() > col[best].abs() {
            best = i;
        }
    }
    if col[best] < 0.0 { -1.0 } else { 1.0 }
}

/// Divergence: `FittedIsomap::transform` (`isomap.rs:479`) diverges from
/// `sklearn/manifold/_isomap.py:435` for out-of-sample points.
///
/// ferrolearn uses raw-Euclidean-to-all-training distances + a hand-rolled
/// Nystroem (`isomap.rs:502-549`); sklearn links the point into the geodesic
/// graph (`G_X[i] = min(dist_matrix_[indices[i]] + distances[i])`,
/// `_isomap.py:430`) then `kernel_pca_.transform`. The two algorithms give
/// different coordinates even AFTER per-column sign alignment.
///
/// Input: 10x3 fixture, `n_components=2, n_neighbors=4`; 3 query points.
/// sklearn `SK_TRANSFORM`; ferrolearn (observed) col-0 sign-aligned differs by
/// ~0.14, col-1 by ~0.33 — both >> 1e-6.
/// Tracking: #2401.
#[test]
fn divergence_transform_out_of_sample() {
    let fitted = Isomap::new(2)
        .with_n_neighbors(4)
        .fit(&fixture(), &())
        .expect("fit must succeed on the 10x3 fixture");
    let got = fitted
        .transform(&x_new())
        .expect("transform must succeed on valid query points");
    assert_eq!(got.dim(), (3, 2), "shape (n_queries, n_components)");

    // Compare magnitudes after aligning each column's svd_flip sign on BOTH
    // sides, so a pure sign difference would NOT count as a divergence. Any
    // surviving difference is a genuine value divergence from the differing
    // out-of-sample algorithm.
    let mut max_diff = 0.0_f64;
    for j in 0..2 {
        let got_col: Vec<f64> = (0..3).map(|i| got[[i, j]]).collect();
        let sk_col: Vec<f64> = (0..3).map(|i| SK_TRANSFORM[i][j]).collect();
        let gs = max_abs_positive_sign(&got_col);
        let ss = max_abs_positive_sign(&sk_col);
        for i in 0..3 {
            let d = (got_col[i] * gs - sk_col[i] * ss).abs();
            if d > max_diff {
                max_diff = d;
            }
        }
    }
    assert!(
        max_diff < 1e-6,
        "Isomap::transform diverges from sklearn _isomap.py:435: \
         max sign-aligned |ferro - sklearn| = {max_diff} (>= 1e-6). \
         ferrolearn uses raw-Euclidean Nystroem; sklearn links into the \
         geodesic graph (_isomap.py:430)."
    );
}
