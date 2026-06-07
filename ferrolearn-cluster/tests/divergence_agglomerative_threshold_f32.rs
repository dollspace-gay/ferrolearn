//! Divergence: `AgglomerativeClustering::<f32>` stores `distances_` in `f32`,
//! but scikit-learn 1.5.2 keeps `distances_` in **float64** even for float32
//! input. The f32 truncation perturbs each merge height by ~1e-7, and the
//! `distance_threshold` cut compares the (truncated) f32 height against the
//! (f32-rounded) threshold, so a threshold chosen between sklearn's f64 height
//! and ferrolearn's f32 height flips `count_nonzero(distances_ >= t)` — and
//! therefore `n_clusters_` and `labels_` — by one.
//!
//! sklearn site: `sklearn/cluster/_agglomerative.py:1088`
//!   `self.distances_ = out[-1]`  (out[-1] is scipy's float64 height column;
//!   scipy `linkage`/`ward_tree` always return float64 heights regardless of
//!   `X.dtype`), and `:1091-1092`
//!   `self.n_clusters_ = np.count_nonzero(self.distances_ >= distance_threshold) + 1`.
//!
//! ferrolearn site: `ferrolearn-cluster/src/agglomerative.rs:757-760`
//!   `let distances_f: Vec<F> = raw_dists.iter().map(|&d| F::from(d)...).collect();`
//!   (the f64 heights are down-cast to `F`, i.e. f32 here) and `:766`
//!   `let count = distances_f.iter().filter(|&&d| d >= t).count();`.
//!
//! Input: 6-pt fixture, `F = f32`, ward linkage, `distance_threshold = 7.1292824`.
//!
//! Live sklearn 1.5.2 oracle (system python3):
//! ```text
//! python3 -c "import numpy as np; from sklearn.cluster import AgglomerativeClustering; \
//!   X=np.array([[0.,0.],[0.2,0.1],[0.9,1.1],[3.,3.2],[3.3,3.],[6.,0.2]],dtype=np.float32); \
//!   m=AgglomerativeClustering(n_clusters=None,distance_threshold=7.1292824,linkage='ward').fit(X); \
//!   print(m.distances_.dtype, m.n_clusters_, m.labels_.tolist(), m.distances_.tolist())"
//! # float64 1 [0, 0, 0, 0, 0, 0] [0.2236..., 0.3605..., 1.5242..., 4.6950..., 7.129282330937144]
//! ```
//!
//! sklearn: `distances_[4] = 7.129282330937144` (f64) is NOT >= 7.1292824, so the
//! top merge is treated as merged → `n_clusters_ == 1`, `labels_ == [0;6]`.
//!
//! ferrolearn: `distances_[4]` is `f32(...) == 7.129282474517822` and the
//! threshold is `f32(7.1292824) == 7.129282474517822`, so `>=` is TRUE → the top
//! merge is treated as NOT merged → `n_clusters_ == 2`,
//! `labels_ == [1, 1, 1, 0, 0, 0]`.
//!
//! Tracking: see crate-level note; failing test pins the f32 `distances_`
//! dtype/precision divergence and its threshold-count knock-on.

use ferrolearn_cluster::{AgglomerativeClustering, Linkage};
use ferrolearn_core::Fit;
use ndarray::Array2;

fn fixture6_f32() -> Array2<f32> {
    Array2::from_shape_vec(
        (6, 2),
        vec![
            0.0, 0.0, 0.2, 0.1, 0.9, 1.1, 3.0, 3.2, 3.3, 3.0, 6.0, 0.2,
        ],
    )
    .unwrap()
}

/// Divergence: ferrolearn `AgglomerativeClustering::<f32>` truncates `distances_`
/// to f32 (sklearn keeps float64), flipping the `distance_threshold` count.
/// sklearn returns `n_clusters_ == 1` / `labels_ == [0;6]`; ferrolearn returns
/// `n_clusters_ == 2` / `labels_ == [1,1,1,0,0,0]`.
#[test]
#[ignore = "divergence: f32 distances_ truncated (sklearn keeps float64) flips distance_threshold count; tracking #2185"]
fn divergence_f32_distances_threshold_count_flip_6pt_ward() {
    // Oracle values from the live sklearn 1.5.2 command in the module doc.
    const SK_N_CLUSTERS: usize = 1;
    const SK_LABELS: [usize; 6] = [0, 0, 0, 0, 0, 0];

    let x = fixture6_f32();
    let t = 7.1292824_f32;
    let fitted = AgglomerativeClustering::<f32>::new(2)
        .with_linkage(Linkage::Ward)
        .with_distance_threshold(t)
        .fit(&x, &())
        .unwrap();

    let labels: Vec<usize> = fitted.labels().iter().copied().collect();

    assert_eq!(
        fitted.n_clusters(),
        SK_N_CLUSTERS,
        "f32 ward t={t}: ferrolearn n_clusters_ {} != sklearn (float64 distances_) {SK_N_CLUSTERS}",
        fitted.n_clusters()
    );
    assert_eq!(
        labels,
        SK_LABELS.to_vec(),
        "f32 ward t={t}: ferrolearn labels_ {labels:?} != sklearn {SK_LABELS:?}"
    );
}
