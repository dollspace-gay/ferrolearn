//! Divergence pin: `ferrolearn_cluster::OPTICS` with `cluster_method='dbscan'`
//! diverges from scikit-learn 1.5.2 when `eps` is set EXACTLY equal to a stored
//! `core_distances_` value, because ferrolearn's `core_distances_` differs from
//! sklearn's by a few ULP and the `near_core = core_distances <= eps` boundary
//! (`sklearn/cluster/_optics.py:786`, INCLUSIVE `<=`) flips.
//!
//! Root cause: sklearn computes `core_distances_` via `NearestNeighbors`
//! (`_compute_core_distances_`, `sklearn/cluster/_optics.py:405-438`), whose
//! squared-distance reduction + sqrt path yields, for the `small10` fixture,
//! `core_distances_[5] = 0.53851648071345` (f64 bits `59ae20ea863be13f`).
//! ferrolearn's `fn core_distance` (`ferrolearn-cluster/src/optics.rs:380-412`,
//! plain `euclidean` = sum-of-squares then sqrt, `:346-351`) yields
//! `0.5385164807134504` (bits `5cae20ea863be13f`) — larger by ~4e-17.
//!
//! When `eps == 0.53851648071345` (the sklearn stored value, which roundtrips
//! bit-exact from the repr literal):
//!   * sklearn   `near_core[5] = 0.53851648071345  <= 0.53851648071345` -> TRUE
//!   * ferrolearn `near_core[5] = 0.5385164807134504 <= 0.53851648071345` -> FALSE
//!
//! Points 5,6,7,8 are thus core-and-far in sklearn (forming cluster 0) but are
//! treated as non-core noise in ferrolearn (all `-1`).
//!
//! sklearn `labels_`  : `[-1, -1, -1, -1, -1, 0, 0, 0, 0, -1]`
//! ferrolearn `labels_`: `[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]`
//!
//! The existing `green_dbscan_small10_eps05_noise`/`green_reachability_small10`
//! guards do NOT catch this: they use eps=0.5 (below the boundary) or a `1e-9`
//! reachability tolerance, neither of which exercises the bit-exact `<=` cut.
//!
//! This pin computes the expected `labels_` from the LIVE sklearn 1.5.2 oracle at
//! test time (`python3 -c`, run from a temp dir), so the assertion is genuinely
//! non-tautological (R-CHAR-3) — the expected value is never copied from the
//! ferrolearn side.
//!
//! Tracking: #2196

use ferrolearn_cluster::OPTICS;
use ferrolearn_cluster::optics::OpticsClusterMethod;
use ferrolearn_core::Fit;
use ndarray::Array2;
use std::process::Command;

/// The 10-point `small10` fixture (also used by the green guards).
fn small10() -> Array2<f64> {
    Array2::from_shape_vec(
        (10, 2),
        vec![
            2.1, 0.3, 1.5, 0.6, 0.5, -0.8, 0.9, 0.2, -1.9, -0.6, -0.1, 0.8, -0.6, 0.6, -0.3, 0.3,
            -0.4, 0.2, 0.7, 0.8,
        ],
    )
    .unwrap()
}

/// `eps` set EXACTLY to sklearn's stored `core_distances_[5]` value
/// (`0.53851648071345`, roundtrips bit-exact from this repr literal).
const EPS_AT_CORE: f64 = 0.53851648071345;

/// Live sklearn 1.5.2 oracle: `OPTICS(min_samples=2, cluster_method='dbscan',
/// eps=0.53851648071345).fit(small10).labels_`. Returns the labels as a `Vec`.
fn sklearn_dbscan_labels() -> Vec<isize> {
    let py = r#"
import numpy as np
from sklearn.cluster import OPTICS
X=np.array([[2.1,0.3],[1.5,0.6],[0.5,-0.8],[0.9,0.2],[-1.9,-0.6],[-0.1,0.8],[-0.6,0.6],[-0.3,0.3],[-0.4,0.2],[0.7,0.8]])
lab=OPTICS(min_samples=2, cluster_method='dbscan', eps=0.53851648071345).fit(X).labels_
print(','.join(str(int(v)) for v in lab))
"#;
    let out = Command::new("python3")
        .arg("-c")
        .arg(py)
        .current_dir("/tmp")
        .output()
        .expect("failed to spawn the sklearn oracle (python3)");
    assert!(
        out.status.success(),
        "sklearn oracle failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let s = String::from_utf8(out.stdout).expect("oracle stdout not utf8");
    s.trim()
        .split(',')
        .map(|t| t.parse::<isize>().expect("oracle label not an int"))
        .collect()
}

/// Divergence: `cluster_optics_dbscan` near_core `<=` boundary
/// (`sklearn/cluster/_optics.py:786`) flips vs ferrolearn at
/// `eps == core_distances_[5]` because ferrolearn's `core_distances_` differs
/// from sklearn's by a few ULP (`ferrolearn-cluster/src/optics.rs:380-412,520`).
///
/// sklearn returns `[-1,-1,-1,-1,-1,0,0,0,0,-1]`; ferrolearn returns all `-1`.
/// Tracking: #2196
#[test]
#[ignore = "divergence: dbscan near_core <= boundary flips at eps==core_distances_ due to sub-ULP core_distance precision (sklearn/cluster/_optics.py:786); tracking #2196"]
fn divergence_dbscan_eps_at_core_distance_boundary() {
    // Expected value: computed LIVE from the installed sklearn 1.5.2 oracle
    // (never copied from ferrolearn) — R-CHAR-3.
    let sk_labels = sklearn_dbscan_labels();

    let fitted = OPTICS::<f64>::new(2)
        .with_cluster_method(OpticsClusterMethod::Dbscan)
        .with_eps(EPS_AT_CORE)
        .fit(&small10(), &())
        .unwrap();
    let ferro: Vec<isize> = fitted.labels().to_vec();

    assert_eq!(
        ferro, sk_labels,
        "OPTICS(cluster_method='dbscan', eps={EPS_AT_CORE}) labels_ must match the \
         live sklearn 1.5.2 oracle. ferro={ferro:?} sklearn={sk_labels:?}. \
         The near_core `<=` boundary (sklearn/cluster/_optics.py:786) flips because \
         ferrolearn's core_distances_ differs from sklearn's by a few ULP \
         (ferrolearn-cluster/src/optics.rs:380-412); #2196"
    );
}
