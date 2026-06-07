//! Divergence + green-guard tests for `ferrolearn-cluster::OPTICS` vs the live
//! scikit-learn 1.5.2 oracle (`sklearn/cluster/_optics.py`).
//!
//! Iteration 131. All expected values are recorded from the LIVE installed
//! sklearn 1.5.2 oracle (`from sklearn.cluster import OPTICS`, run from `/tmp`),
//! NEVER literal-copied from ferrolearn (R-CHAR-3). OPTICS is deterministic
//! (no RNG), so genuine value-parity is checkable.
//!
//! Contents:
//!   - GREEN-GUARDS (PASS today, back the SHIPPED REQ rows):
//!       * `green_core_distances_three_blobs`  — REQ-1 core_distances_ VALUE
//!       * `green_core_distances_small10_ms2`  — REQ-1 core_distances_ VALUE (harder fixture)
//!       * `green_core_distances_small10_ms3`  — REQ-1 core_distances_ VALUE (k=3)
//!       * `green_ordering_three_blobs`        — Probe-0 ordering agreement (clean fixture)
//!       * `green_ordering_docstring`          — Probe-0 ordering agreement (docstring fixture)
//!       * `green_reachability_docstring`      — Probe-0 reachability agreement (docstring fixture)
//!       * `green_reachability_small10`        — REQ-3 reachability_ VALUE on the noisy tie fixture
//!       * `green_ordering_small10`            — REQ-2 ordering VALUE after the #1080 fix
//!         (formerly `divergence_ordering_small10`; the fix landed so the pin is green)

use ferrolearn_cluster::OPTICS;
use ferrolearn_cluster::optics::OpticsClusterMethod;
use ferrolearn_core::Fit;
use ndarray::Array2;

/// Three tight 2-D clusters — the in-tree test fixture (9x2).
fn three_blobs() -> Array2<f64> {
    Array2::from_shape_vec(
        (9, 2),
        vec![
            0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 5.0, 5.0, 5.1, 5.0, 5.0, 5.1, 10.0, 0.0, 10.1, 0.0, 10.0,
            0.1,
        ],
    )
    .unwrap()
}

/// The `_optics.py` docstring worked-example fixture (6x2).
fn docstring() -> Array2<f64> {
    Array2::from_shape_vec(
        (6, 2),
        vec![1.0, 2.0, 2.0, 5.0, 3.0, 6.0, 8.0, 7.0, 8.0, 8.0, 7.0, 3.0],
    )
    .unwrap()
}

/// A 10-point fixture (RandomState(1024).randn(10,2).round(1)) where ferrolearn's
/// former BinaryHeap traversal diverged from sklearn's linear-argmin traversal on
/// a reachability tie that sklearn's `np.around` rounding turns into an exact tie.
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

const TOL: f64 = 1e-6;

// ─────────────────────────────────────────────────────────────────────────────
// GREEN-GUARD: REQ-1 core_distances_ VALUE parity (SHIPPED)
// ─────────────────────────────────────────────────────────────────────────────

/// Green-guard: REQ-1. `OPTICS(min_samples=2).fit(three_blobs).core_distances_`
/// value-matches sklearn's `_compute_core_distances_`
/// (`sklearn/cluster/_optics.py:405-438`, k-NN distance `:437`).
///
/// LIVE ORACLE (sklearn 1.5.2, run from /tmp):
///   python3 -c "import numpy as np; from sklearn.cluster import OPTICS; \
///     X=np.array([[0.,0.],[0.1,0.],[0.,0.1],[5.,5.],[5.1,5.],[5.,5.1],[10.,0.],[10.1,0.],[10.,0.1]]); \
///     print([repr(float(c)) for c in OPTICS(min_samples=2).fit(X).core_distances_])"
///   -> all nine values == 0.1
#[test]
fn green_core_distances_three_blobs() {
    // sklearn 1.5.2 oracle (above): every point's 1-NN distance is 0.1.
    let sk_core = [0.1_f64; 9];
    let fitted = OPTICS::<f64>::new(2).fit(&three_blobs(), &()).unwrap();
    let core = fitted.core_distances();
    for (i, &sk) in sk_core.iter().enumerate() {
        assert!(
            (core[i] - sk).abs() <= TOL,
            "core_distances[{i}]: ferro={} sklearn={sk}",
            core[i]
        );
    }
}

/// Green-guard: REQ-1. `OPTICS(min_samples=2).fit(small10).core_distances_`
/// value-matches sklearn on a harder (irregular) fixture.
/// (`sklearn/cluster/_optics.py:405-438`.)
///
/// LIVE ORACLE (sklearn 1.5.2, run from /tmp):
///   python3 -c "import numpy as np; from sklearn.cluster import OPTICS; \
///     X=np.array([[2.1,0.3],[1.5,0.6],[0.5,-0.8],[0.9,0.2],[-1.9,-0.6],[-0.1,0.8],[-0.6,0.6],[-0.3,0.3],[-0.4,0.2],[0.7,0.8]]); \
///     print(OPTICS(min_samples=2).fit(X).core_distances_.tolist())"
///   -> [0.670820393249937, 0.670820393249937, 1.077032961426901, 0.632455532033676,
///       1.7, 0.53851648071345, 0.424264068711928, 0.14142135623731, 0.14142135623731,
///       0.632455532033676]
#[test]
fn green_core_distances_small10_ms2() {
    let sk_core = [
        0.670820393249937,
        0.670820393249937,
        1.077032961426901,
        0.632455532033676,
        1.7,
        0.53851648071345,
        0.424264068711928,
        0.14142135623731,
        0.14142135623731,
        0.632455532033676,
    ];
    let fitted = OPTICS::<f64>::new(2).fit(&small10(), &()).unwrap();
    let core = fitted.core_distances();
    for (i, &sk) in sk_core.iter().enumerate() {
        assert!(
            (core[i] - sk).abs() <= TOL,
            "core_distances[{i}]: ferro={} sklearn={sk}",
            core[i]
        );
    }
}

/// Green-guard: REQ-1. `OPTICS(min_samples=3).fit(small10).core_distances_`
/// value-matches sklearn (k=3 NN — a different neighbour rank).
/// (`sklearn/cluster/_optics.py:405-438`.)
///
/// LIVE ORACLE (sklearn 1.5.2, run from /tmp):
///   python3 -c "import numpy as np; from sklearn.cluster import OPTICS; \
///     X=np.array([[2.1,0.3],[1.5,0.6],[0.5,-0.8],[0.9,0.2],[-1.9,-0.6],[-0.1,0.8],[-0.6,0.6],[-0.3,0.3],[-0.4,0.2],[0.7,0.8]]); \
///     print(OPTICS(min_samples=3).fit(X).core_distances_.tolist())"
///   -> [1.20415945787923, 0.721110255092798, 1.345362404707371, 0.721110255092798,
///       1.769180601295413, 0.53851648071345, 0.447213595499958, 0.424264068711928,
///       0.447213595499958, 0.8]
#[test]
fn green_core_distances_small10_ms3() {
    let sk_core = [
        1.20415945787923,
        0.721110255092798,
        1.345362404707371,
        0.721110255092798,
        1.769180601295413,
        0.53851648071345,
        0.447213595499958,
        0.424264068711928,
        0.447213595499958,
        0.8,
    ];
    let fitted = OPTICS::<f64>::new(3).fit(&small10(), &()).unwrap();
    let core = fitted.core_distances();
    for (i, &sk) in sk_core.iter().enumerate() {
        assert!(
            (core[i] - sk).abs() <= TOL,
            "core_distances[{i}]: ferro={} sklearn={sk}",
            core[i]
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GREEN-GUARD: Probe-0 ordering/reachability agreement on clean fixtures.
// These fixtures have NO reachability ties, so heap-order == linear-argmin-order
// == sklearn-order; they STAY green after the #1080 traversal fix.
// ─────────────────────────────────────────────────────────────────────────────

/// Green-guard: Probe-0. `OPTICS(min_samples=2).fit(three_blobs).ordering_`
/// matches sklearn (clean fixture, no ties).
/// (`sklearn/cluster/_optics.py:638-659` traversal.)
///
/// LIVE ORACLE (sklearn 1.5.2, run from /tmp):
///   python3 -c "import numpy as np; from sklearn.cluster import OPTICS; \
///     X=np.array([[0.,0.],[0.1,0.],[0.,0.1],[5.,5.],[5.1,5.],[5.,5.1],[10.,0.],[10.1,0.],[10.,0.1]]); \
///     print(OPTICS(min_samples=2).fit(X).ordering_.tolist())"
///   -> [0, 1, 2, 3, 4, 5, 8, 6, 7]
#[test]
fn green_ordering_three_blobs() {
    let sk_ordering = [0usize, 1, 2, 3, 4, 5, 8, 6, 7];
    let fitted = OPTICS::<f64>::new(2).fit(&three_blobs(), &()).unwrap();
    assert_eq!(fitted.ordering(), &sk_ordering[..]);
}

/// Green-guard: Probe-0. `OPTICS(min_samples=2).fit(docstring).ordering_`
/// matches sklearn's docstring worked example
/// (`sklearn/cluster/_optics.py:585-594`).
///
/// LIVE ORACLE (sklearn 1.5.2, run from /tmp):
///   python3 -c "import numpy as np; from sklearn.cluster import OPTICS; \
///     X=np.array([[1.,2.],[2.,5.],[3.,6.],[8.,7.],[8.,8.],[7.,3.]]); \
///     print(OPTICS(min_samples=2).fit(X).ordering_.tolist())"
///   -> [0, 1, 2, 5, 3, 4]
#[test]
fn green_ordering_docstring() {
    let sk_ordering = [0usize, 1, 2, 5, 3, 4];
    let fitted = OPTICS::<f64>::new(2).fit(&docstring(), &()).unwrap();
    assert_eq!(fitted.ordering(), &sk_ordering[..]);
}

/// Green-guard: Probe-0. `OPTICS(min_samples=2).fit(docstring).reachability_`
/// matches sklearn (`sklearn/cluster/_optics.py:585-594` / `_set_reach_dist`
/// `:671-714`).
///
/// LIVE ORACLE (sklearn 1.5.2, run from /tmp):
///   python3 -c "import numpy as np; from sklearn.cluster import OPTICS; \
///     X=np.array([[1.,2.],[2.,5.],[3.,6.],[8.,7.],[8.,8.],[7.,3.]]); \
///     print([repr(float(r)) for r in OPTICS(min_samples=2).fit(X).reachability_])"
///   -> ['inf', '3.16227766016838', '1.414213562373095', '4.12310562561766', '1.0', '5.0']
/// (reachability_ is indexed by ORIGINAL point index, not ordering position.)
#[test]
#[allow(
    clippy::approx_constant,
    reason = "1.414213562373095 is the exact sklearn 1.5.2 reachability_ oracle value (R-CHAR-3), not an approximation of SQRT_2 (they differ at the 16th digit)"
)]
fn green_reachability_docstring() {
    // index 0 is inf (first seed); the rest are finite sklearn values.
    let sk_reach = [
        f64::INFINITY,
        3.16227766016838,
        1.414213562373095,
        4.12310562561766,
        1.0,
        5.0,
    ];
    let fitted = OPTICS::<f64>::new(2).fit(&docstring(), &()).unwrap();
    let reach = fitted.reachability();
    assert!(reach[0].is_infinite(), "reach[0] should be inf");
    for i in 1..6 {
        assert!(
            (reach[i] - sk_reach[i]).abs() <= TOL,
            "reachability[{i}]: ferro={} sklearn={}",
            reach[i],
            sk_reach[i]
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GREEN-GUARD: REQ-3 reachability_ VALUE on the noisy small10 tie fixture.
//
// This is the load-bearing guard for the SHIPPED REQ-3 row: the only prior
// reachability guard was on the CLEAN docstring fixture (no ties). small10 is the
// tie-prone fixture whose `np.around` collapse drives the REQ-2 ordering parity,
// so a reachability_ VALUE match here is what actually backs REQ-3 SHIPPED.
// ─────────────────────────────────────────────────────────────────────────────

/// Green-guard: REQ-3. `OPTICS(min_samples=2).fit(small10).reachability_`
/// value-matches sklearn on the noisy tie fixture
/// (`sklearn/cluster/_optics.py:671-714` `_set_reach_dist`; `np.around` `:711`).
///
/// LIVE ORACLE (sklearn 1.5.2, run from /tmp):
///   python3 -c "import numpy as np; from sklearn.cluster import OPTICS; \
///     X=np.array([[2.1,0.3],[1.5,0.6],[0.5,-0.8],[0.9,0.2],[-1.9,-0.6],[-0.1,0.8],[-0.6,0.6],[-0.3,0.3],[-0.4,0.2],[0.7,0.8]]); \
///     print([repr(float(r)) for r in OPTICS(min_samples=2).fit(X).reachability_])"
///   -> ['inf', '0.670820393249937', '1.077032961426901', '0.721110255092798', '1.7',
///       '0.8', '0.53851648071345', '0.424264068711928', '0.14142135623731',
///       '0.632455532033676']
/// (reachability_ is indexed by ORIGINAL point index, not ordering position.)
#[test]
fn green_reachability_small10() {
    // LIVE sklearn 1.5.2 oracle (above), never copied from ferrolearn.
    // index 0 is the first seed (inf reachability).
    let sk_reach = [
        f64::INFINITY,
        0.670820393249937,
        1.077032961426901,
        0.721110255092798,
        1.7,
        0.8,
        0.53851648071345,
        0.424264068711928,
        0.14142135623731,
        0.632455532033676,
    ];
    let fitted = OPTICS::<f64>::new(2).fit(&small10(), &()).unwrap();
    let reach = fitted.reachability();
    for (i, &sk) in sk_reach.iter().enumerate() {
        if sk.is_infinite() {
            assert!(
                reach[i].is_infinite(),
                "reachability[{i}]: ferro={} sklearn=inf",
                reach[i]
            );
        } else {
            assert!(
                (reach[i] - sk).abs() <= 1e-9,
                "reachability[{i}]: ferro={} sklearn={sk}",
                reach[i]
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GREEN-GUARD: REQ-2 ordering/tie-break VALUE parity (blocker #1080, NOW FIXED).
//
// Formerly `divergence_ordering_small10` (a FAILING pin). The #1080 fix landed
// (linear-argmin single-pool traversal with smallest-index tie-break +
// `np.around` reachability rounding), so this is now a GREEN guard backing the
// SHIPPED REQ-2 row.
// ─────────────────────────────────────────────────────────────────────────────

/// Green-guard: REQ-2. `OPTICS(min_samples=2).fit(small10).ordering_`
/// value-matches sklearn's linear-argmin traversal with smallest-index tie-break
/// (`sklearn/cluster/_optics.py:638-659`, `:711`).
///
/// After point 5 is processed, points 6 and 7 receive reachabilities that differ
/// only at the 16th significant digit; sklearn's `np.around(decimals=15)` collapses
/// both to an exact TIE, so its smallest-index tie-break visits 6 before 7. The
/// fixed `fn fit` applies the same rounding (`fn round_to_precision`) and linear
/// argmin, reproducing the ordering.
///
/// LIVE ORACLE (sklearn 1.5.2, run from /tmp):
///   python3 -c "import numpy as np; from sklearn.cluster import OPTICS; \
///     X=np.array([[2.1,0.3],[1.5,0.6],[0.5,-0.8],[0.9,0.2],[-1.9,-0.6],[-0.1,0.8],[-0.6,0.6],[-0.3,0.3],[-0.4,0.2],[0.7,0.8]]); \
///     print(OPTICS(min_samples=2).fit(X).ordering_.tolist())"
///   -> [0, 1, 3, 9, 5, 6, 7, 8, 2, 4]
///
/// Tracking: #1080 (fixed)
#[test]
fn green_ordering_small10() {
    // LIVE sklearn 1.5.2 oracle expected value (above), never from ferrolearn:
    let sk_ordering = [0usize, 1, 3, 9, 5, 6, 7, 8, 2, 4];
    let fitted = OPTICS::<f64>::new(2).fit(&small10(), &()).unwrap();
    assert_eq!(
        fitted.ordering(),
        &sk_ordering[..],
        "ferrolearn OPTICS ordering_ must match sklearn linear-argmin traversal \
         (sklearn/cluster/_optics.py:638-659,711); #1080"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// GREEN-GUARD: REQ-4 `predecessor_` `-1`-sentinel int array VALUE parity (#1082).
//
// sklearn's `predecessor_` is `np.full(n_samples, -1, dtype=int)` then set on a
// STRICT reachability improvement only — `improved = np.where(rdists < ...)` and
// `predecessor_[unproc[improved]] = point_index` (`sklearn/cluster/_optics.py:712-714`),
// indexed by ORIGINAL object order, seeds = -1 (`:187-189`, `:558-560`). ferrolearn's
// `fn update_seeds` records the predecessor under the SAME strict
// `new_reach < reachability[q]` condition, then `Fit::fit` maps `Some(j)->j`,
// `None->-1` into the `Array1<i64>` surfaced by `FittedOPTICS::predecessor()`.
// These guards assert element-wise integer parity (incl. the -1 seed sentinel)
// against the live sklearn 1.5.2 oracle (R-CHAR-3).
// ─────────────────────────────────────────────────────────────────────────────

/// Green-guard: REQ-4. `OPTICS(min_samples=2).fit(three_blobs).predecessor_`
/// value-matches sklearn's `-1`-sentinel int array
/// (`sklearn/cluster/_optics.py:604-605,712-714`; seeds -1 `:187-189`).
///
/// LIVE ORACLE (sklearn 1.5.2, run from /tmp):
///   python3 -c "import numpy as np; from sklearn.cluster import OPTICS; \
///     X=np.array([[0.,0.],[0.1,0.],[0.,0.1],[5.,5.],[5.1,5.],[5.,5.1],[10.,0.],[10.1,0.],[10.,0.1]]); \
///     print(OPTICS(min_samples=2).fit(X).predecessor_.tolist())"
///   -> [-1, 0, 0, 1, 3, 3, 8, 6, 4]   (exactly one -1: the single seed)
#[test]
fn green_predecessor_three_blobs() {
    // LIVE sklearn 1.5.2 oracle (above), never copied from ferrolearn.
    let sk_pred: [i64; 9] = [-1, 0, 0, 1, 3, 3, 8, 6, 4];
    let fitted = OPTICS::<f64>::new(2).fit(&three_blobs(), &()).unwrap();
    let pred = fitted.predecessor();
    assert_eq!(pred.len(), 9, "predecessor_ must be shape (n_samples,)");
    for (i, &sk) in sk_pred.iter().enumerate() {
        assert_eq!(
            pred[i], sk,
            "predecessor_[{i}]: ferro={} sklearn={sk}",
            pred[i]
        );
    }
    // Seed-sentinel count matches sklearn (exactly one -1).
    let n_neg1 = pred.iter().filter(|&&v| v == -1).count();
    let sk_n_neg1 = sk_pred.iter().filter(|&&v| v == -1).count();
    assert_eq!(n_neg1, sk_n_neg1, "count of -1 seeds must match sklearn");
    assert_eq!(
        n_neg1, 1,
        "three_blobs has a single seed (one connected order)"
    );
    // The seed sentinel sits on the first point in the ordering.
    let first = fitted.ordering()[0];
    assert_eq!(pred[first], -1, "the first ordered point is the -1 seed");
}

/// Green-guard: REQ-4. `OPTICS(min_samples=2).fit(small10).predecessor_`
/// value-matches sklearn on the TIE-PRONE fixture — the case where a `<=`
/// (rather than strict `<`) improvement condition, or unconditional assignment,
/// would diverge. The match confirms `fn update_seeds` uses the same strict
/// `rdist < reachability_[i]` rule (`sklearn/cluster/_optics.py:712-714`).
///
/// LIVE ORACLE (sklearn 1.5.2, run from /tmp):
///   python3 -c "import numpy as np; from sklearn.cluster import OPTICS; \
///     X=np.array([[2.1,0.3],[1.5,0.6],[0.5,-0.8],[0.9,0.2],[-1.9,-0.6],[-0.1,0.8],[-0.6,0.6],[-0.3,0.3],[-0.4,0.2],[0.7,0.8]]); \
///     print(OPTICS(min_samples=2).fit(X).predecessor_.tolist())"
///   -> [-1, 0, 3, 1, 8, 9, 5, 6, 7, 3]   (exactly one -1)
#[test]
fn green_predecessor_small10() {
    // LIVE sklearn 1.5.2 oracle (above), never copied from ferrolearn.
    let sk_pred: [i64; 10] = [-1, 0, 3, 1, 8, 9, 5, 6, 7, 3];
    let fitted = OPTICS::<f64>::new(2).fit(&small10(), &()).unwrap();
    let pred = fitted.predecessor();
    assert_eq!(pred.len(), 10, "predecessor_ must be shape (n_samples,)");
    for (i, &sk) in sk_pred.iter().enumerate() {
        assert_eq!(
            pred[i], sk,
            "predecessor_[{i}]: ferro={} sklearn={sk} (tie-prone strict-improvement)",
            pred[i]
        );
    }
    let n_neg1 = pred.iter().filter(|&&v| v == -1).count();
    assert_eq!(n_neg1, 1, "small10 has a single -1 seed (matches sklearn)");
    let first = fitted.ordering()[0];
    assert_eq!(pred[first], -1, "the first ordered point is the -1 seed");
}

/// Green-guard: REQ-4. `OPTICS(min_samples=2).fit(docstring).predecessor_`
/// value-matches sklearn's docstring worked example
/// (`sklearn/cluster/_optics.py:593-594` shows `predecessor array([-1,0,1,5,3,2])`).
///
/// LIVE ORACLE (sklearn 1.5.2, run from /tmp):
///   python3 -c "import numpy as np; from sklearn.cluster import OPTICS; \
///     X=np.array([[1.,2.],[2.,5.],[3.,6.],[8.,7.],[8.,8.],[7.,3.]]); \
///     print(OPTICS(min_samples=2).fit(X).predecessor_.tolist())"
///   -> [-1, 0, 1, 5, 3, 2]   (matches the `_optics.py:594` docstring exactly)
#[test]
fn green_predecessor_docstring() {
    // LIVE sklearn 1.5.2 oracle (above) == the `_optics.py:594` docstring constant.
    let sk_pred: [i64; 6] = [-1, 0, 1, 5, 3, 2];
    let fitted = OPTICS::<f64>::new(2).fit(&docstring(), &()).unwrap();
    let pred = fitted.predecessor();
    assert_eq!(pred.len(), 6, "predecessor_ must be shape (n_samples,)");
    for (i, &sk) in sk_pred.iter().enumerate() {
        assert_eq!(
            pred[i], sk,
            "predecessor_[{i}]: ferro={} sklearn={sk}",
            pred[i]
        );
    }
    let n_neg1 = pred.iter().filter(|&&v| v == -1).count();
    assert_eq!(n_neg1, 1, "docstring fixture has a single -1 seed");
}

// ─────────────────────────────────────────────────────────────────────────────
// GREEN-GUARD: REQ-6 `cluster_method='dbscan'` + `eps` + `cluster_optics_dbscan`
// VALUE parity (#1084, NOW SHIPPED).
//
// sklearn's `OPTICS(cluster_method='dbscan', eps=...).fit(X).labels_` derives the
// labels from the reachability graph by the linear-time two-step extraction
// `cluster_optics_dbscan` (`sklearn/cluster/_optics.py:781-787`):
//   far_reach = reachability > eps              (STRICT >)
//   near_core = core_distances <= eps           (INCLUSIVE <=)
//   labels[ordering] = cumsum(far_reach[ordering] & near_core[ordering]) - 1
//   labels[far_reach & ~near_core] = -1
// `eps` resolves to `max_eps` when unset (`:375-378`); `eps > max_eps` raises
// ValueError (`:380-383`). All expected values below are from the LIVE sklearn
// 1.5.2 oracle (run from /tmp), never copied from ferrolearn (R-CHAR-3). f64
// fixtures only (the OPTICS graph diverges on f32, #2195 — out of scope here).
// ─────────────────────────────────────────────────────────────────────────────

/// Green-guard: REQ-6. `OPTICS(min_samples=2, cluster_method='dbscan', eps=0.5)
/// .fit(three_blobs).labels_` value-matches sklearn — three tight clusters, no
/// noise (`sklearn/cluster/_optics.py:781-787`).
///
/// LIVE ORACLE (sklearn 1.5.2, run from /tmp):
///   python3 -c "import numpy as np; from sklearn.cluster import OPTICS; \
///     X=np.array([[0.,0.],[0.1,0.],[0.,0.1],[5.,5.],[5.1,5.],[5.,5.1],[10.,0.],[10.1,0.],[10.,0.1]]); \
///     print(OPTICS(min_samples=2, cluster_method='dbscan', eps=0.5).fit(X).labels_.tolist())"
///   -> [0, 0, 0, 1, 1, 1, 2, 2, 2]
#[test]
fn green_dbscan_three_blobs_eps05() {
    let sk_labels: [isize; 9] = [0, 0, 0, 1, 1, 1, 2, 2, 2];
    let fitted = OPTICS::<f64>::new(2)
        .with_cluster_method(OpticsClusterMethod::Dbscan)
        .with_eps(0.5)
        .fit(&three_blobs(), &())
        .unwrap();
    assert_eq!(
        fitted.labels().as_slice().unwrap(),
        &sk_labels[..],
        "dbscan labels_ must match sklearn (sklearn/cluster/_optics.py:781-787); #1084"
    );
}

/// Green-guard: REQ-6. With `eps=None` and the default `max_eps=inf`, `eps`
/// resolves to `inf`; `far_reach = reachability > inf` is FALSE everywhere, so the
/// cumsum is all-zero → `labels = cumsum - 1 = -1` everywhere (all noise). The
/// noise mask is empty (no point has `far_reach`), so the result is all `-1`.
/// (`sklearn/cluster/_optics.py:375-378,781-787`.)
///
/// LIVE ORACLE (sklearn 1.5.2, run from /tmp):
///   python3 -c "import numpy as np; from sklearn.cluster import OPTICS; \
///     X=np.array([[0.,0.],[0.1,0.],[0.,0.1],[5.,5.],[5.1,5.],[5.,5.1],[10.,0.],[10.1,0.],[10.,0.1]]); \
///     print(OPTICS(min_samples=2, cluster_method='dbscan').fit(X).labels_.tolist())"
///   -> [-1, -1, -1, -1, -1, -1, -1, -1, -1]
#[test]
fn green_dbscan_eps_none_all_noise() {
    let sk_labels: [isize; 9] = [-1, -1, -1, -1, -1, -1, -1, -1, -1];
    // eps unset → resolves to max_eps (= inf here).
    let fitted = OPTICS::<f64>::new(2)
        .with_cluster_method(OpticsClusterMethod::Dbscan)
        .fit(&three_blobs(), &())
        .unwrap();
    assert_eq!(
        fitted.labels().as_slice().unwrap(),
        &sk_labels[..],
        "eps=None,max_eps=inf must be all-noise like sklearn; #1084"
    );
}

/// Green-guard: REQ-6. A LARGE `eps` merges everything into a single cluster
/// (`sklearn/cluster/_optics.py:781-787`).
///
/// LIVE ORACLE (sklearn 1.5.2, run from /tmp):
///   python3 -c "import numpy as np; from sklearn.cluster import OPTICS; \
///     X=np.array([[0.,0.],[0.1,0.],[0.,0.1],[5.,5.],[5.1,5.],[5.,5.1],[10.,0.],[10.1,0.],[10.,0.1]]); \
///     print(OPTICS(min_samples=2, cluster_method='dbscan', eps=100.0).fit(X).labels_.tolist())"
///   -> [0, 0, 0, 0, 0, 0, 0, 0, 0]
#[test]
fn green_dbscan_three_blobs_eps_large_merge() {
    let sk_labels: [isize; 9] = [0, 0, 0, 0, 0, 0, 0, 0, 0];
    let fitted = OPTICS::<f64>::new(2)
        .with_cluster_method(OpticsClusterMethod::Dbscan)
        .with_eps(100.0)
        .fit(&three_blobs(), &())
        .unwrap();
    assert_eq!(
        fitted.labels().as_slice().unwrap(),
        &sk_labels[..],
        "large eps must merge all into one cluster like sklearn; #1084"
    );
}

/// Green-guard: REQ-6. `small10`, `eps=0.7` — a mixed labelling with NOISE points
/// (`-1`), exercising both the STRICT-`>` reachability boundary and the
/// `<=`-INCLUSIVE core boundary (`sklearn/cluster/_optics.py:781-787`).
///
/// LIVE ORACLE (sklearn 1.5.2, run from /tmp):
///   python3 -c "import numpy as np; from sklearn.cluster import OPTICS; \
///     X=np.array([[2.1,0.3],[1.5,0.6],[0.5,-0.8],[0.9,0.2],[-1.9,-0.6],[-0.1,0.8],[-0.6,0.6],[-0.3,0.3],[-0.4,0.2],[0.7,0.8]]); \
///     print(OPTICS(min_samples=2, cluster_method='dbscan', eps=0.7).fit(X).labels_.tolist())"
///   -> [0, 0, -1, 1, -1, 2, 2, 2, 2, 1]
#[test]
fn green_dbscan_small10_eps07_mixed() {
    let sk_labels: [isize; 10] = [0, 0, -1, 1, -1, 2, 2, 2, 2, 1];
    let fitted = OPTICS::<f64>::new(2)
        .with_cluster_method(OpticsClusterMethod::Dbscan)
        .with_eps(0.7)
        .fit(&small10(), &())
        .unwrap();
    assert_eq!(
        fitted.labels().as_slice().unwrap(),
        &sk_labels[..],
        "small10 eps=0.7 mixed labelling must match sklearn; #1084"
    );
}

/// Green-guard: REQ-6. `small10`, `eps=0.5` — produces several `-1` noise points
/// (small eps) with a single 3-point cluster
/// (`sklearn/cluster/_optics.py:781-787`).
///
/// LIVE ORACLE (sklearn 1.5.2, run from /tmp):
///   python3 -c "import numpy as np; from sklearn.cluster import OPTICS; \
///     X=np.array([[2.1,0.3],[1.5,0.6],[0.5,-0.8],[0.9,0.2],[-1.9,-0.6],[-0.1,0.8],[-0.6,0.6],[-0.3,0.3],[-0.4,0.2],[0.7,0.8]]); \
///     print(OPTICS(min_samples=2, cluster_method='dbscan', eps=0.5).fit(X).labels_.tolist())"
///   -> [-1, -1, -1, -1, -1, -1, 0, 0, 0, -1]
#[test]
fn green_dbscan_small10_eps05_noise() {
    let sk_labels: [isize; 10] = [-1, -1, -1, -1, -1, -1, 0, 0, 0, -1];
    let fitted = OPTICS::<f64>::new(2)
        .with_cluster_method(OpticsClusterMethod::Dbscan)
        .with_eps(0.5)
        .fit(&small10(), &())
        .unwrap();
    assert_eq!(
        fitted.labels().as_slice().unwrap(),
        &sk_labels[..],
        "small10 eps=0.5 small-eps noise must match sklearn; #1084"
    );
}

/// Green-guard: REQ-6. `small10`, `eps=2.0` — a large eps merges everything
/// (`sklearn/cluster/_optics.py:781-787`).
///
/// LIVE ORACLE (sklearn 1.5.2, run from /tmp):
///   python3 -c "import numpy as np; from sklearn.cluster import OPTICS; \
///     X=np.array([[2.1,0.3],[1.5,0.6],[0.5,-0.8],[0.9,0.2],[-1.9,-0.6],[-0.1,0.8],[-0.6,0.6],[-0.3,0.3],[-0.4,0.2],[0.7,0.8]]); \
///     print(OPTICS(min_samples=2, cluster_method='dbscan', eps=2.0).fit(X).labels_.tolist())"
///   -> [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#[test]
fn green_dbscan_small10_eps2_merge() {
    let sk_labels: [isize; 10] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    let fitted = OPTICS::<f64>::new(2)
        .with_cluster_method(OpticsClusterMethod::Dbscan)
        .with_eps(2.0)
        .fit(&small10(), &())
        .unwrap();
    assert_eq!(
        fitted.labels().as_slice().unwrap(),
        &sk_labels[..],
        "small10 eps=2.0 must merge all like sklearn; #1084"
    );
}

/// Green-guard: REQ-6. `eps > max_eps` must error (no panic), mirroring sklearn's
/// `ValueError("Specify an epsilon smaller than %s. Got %s.")`
/// (`sklearn/cluster/_optics.py:380-383`).
///
/// LIVE ORACLE (sklearn 1.5.2, run from /tmp):
///   python3 -c "import numpy as np; from sklearn.cluster import OPTICS; \
///     X=np.array([[0.,0.],[0.1,0.],[0.,0.1],[5.,5.],[5.1,5.],[5.,5.1],[10.,0.],[10.1,0.],[10.,0.1]]); \
///     OPTICS(min_samples=2, cluster_method='dbscan', eps=5.0, max_eps=1.0).fit(X)"
///   -> ValueError: Specify an epsilon smaller than 1.0. Got 5.0.
#[test]
fn green_dbscan_eps_gt_max_eps_errs() {
    let result = OPTICS::<f64>::new(2)
        .with_cluster_method(OpticsClusterMethod::Dbscan)
        .with_eps(5.0)
        .with_max_eps(1.0)
        .fit(&three_blobs(), &());
    assert!(
        result.is_err(),
        "eps > max_eps must error like sklearn's ValueError (sklearn/cluster/_optics.py:380-383); #1084"
    );
}

/// Green-guard: REQ-6. The DEFAULT `cluster_method` (Xi) path is UNCHANGED — the
/// existing three_blobs Xi labels_ still produce three clusters. Confirms the new
/// `cluster_method` dispatch did not perturb the Xi default
/// (`sklearn/cluster/_optics.py:363-372` Xi branch).
///
/// LIVE ORACLE (sklearn 1.5.2, run from /tmp):
///   python3 -c "import numpy as np; from sklearn.cluster import OPTICS; \
///     X=np.array([[0.,0.],[0.1,0.],[0.,0.1],[5.,5.],[5.1,5.],[5.,5.1],[10.,0.],[10.1,0.],[10.,0.1]]); \
///     print(OPTICS(min_samples=2).fit(X).labels_.tolist())"
///   -> [0, 0, 0, 1, 1, 1, 2, 2, 2]
#[test]
fn green_xi_default_unchanged_three_blobs() {
    let sk_labels: [isize; 9] = [0, 0, 0, 1, 1, 1, 2, 2, 2];
    // No with_cluster_method → default Xi path.
    let est = OPTICS::<f64>::new(2);
    assert_eq!(est.cluster_method, OpticsClusterMethod::Xi);
    let fitted = est.fit(&three_blobs(), &()).unwrap();
    assert_eq!(
        fitted.labels().as_slice().unwrap(),
        &sk_labels[..],
        "default Xi labels_ must be unchanged by the dbscan dispatch; #1084"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// REQ-11 — parameter-validation accept/reject BOUNDARIES match sklearn's
// `OPTICS._parameter_constraints` (`sklearn/cluster/_optics.py:242-264`).
//
// Only the accept/reject BOUNDARY is matched; the error TYPE stays the
// grandfathered crate `FerroError` ABI (NOT sklearn's `InvalidParameterError`).
// Each oracle is cited via a `python3 -c` command (run from /tmp), recorded from
// the LIVE installed sklearn 1.5.2 oracle, NEVER copied from ferrolearn (R-CHAR-3).
// ─────────────────────────────────────────────────────────────────────────────

/// REQ-11. `min_samples` int must be `>= 2`
/// (`"min_samples": [Interval(Integral, 2, None, closed="left"), ...]`,
/// `sklearn/cluster/_optics.py:243-246`). `min_samples ∈ {0, 1}` is REJECTED.
///
/// LIVE ORACLE (sklearn 1.5.2, run from /tmp): both raise InvalidParameterError
///   python3 -c "import numpy as np; from sklearn.cluster import OPTICS; \
///     X=np.array([[0.,0.],[0.1,0.],[0.,0.1],[5.,5.],[5.1,5.],[5.,5.1],[10.,0.],[10.1,0.],[10.,0.1]]); \
///     OPTICS(min_samples=1).fit(X)"
///   -> InvalidParameterError: The 'min_samples' parameter of OPTICS must be an
///      int in the range [2, inf) or a float ...
///   python3 -c "... OPTICS(min_samples=0).fit(X)"
///   -> InvalidParameterError (same: range [2, inf))
#[test]
fn green_min_samples_below_2_rejected() {
    // ferrolearn rejects both 0 and 1 (sklearn raises InvalidParameterError);
    // the BOUNDARY matches, the error TYPE is the grandfathered FerroError.
    let r0 = OPTICS::<f64>::new(0).fit(&three_blobs(), &());
    let r1 = OPTICS::<f64>::new(1).fit(&three_blobs(), &());
    assert!(r0.is_err(), "min_samples=0 must be rejected like sklearn");
    assert!(r1.is_err(), "min_samples=1 must be rejected like sklearn");
}

/// REQ-11. `min_samples == 2` is ACCEPTED (the left-closed interval lower bound).
///
/// LIVE ORACLE (sklearn 1.5.2, run from /tmp): fits without error
///   python3 -c "import numpy as np; from sklearn.cluster import OPTICS; \
///     X=np.array([[0.,0.],[0.1,0.],[0.,0.1],[5.,5.],[5.1,5.],[5.,5.1],[10.,0.],[10.1,0.],[10.,0.1]]); \
///     print(OPTICS(min_samples=2).fit(X).ordering_.tolist())"
///   -> [0, 1, 2, 3, 4, 5, 8, 6, 7]   (no error)
#[test]
fn green_min_samples_2_accepted() {
    let r = OPTICS::<f64>::new(2).fit(&three_blobs(), &());
    assert!(r.is_ok(), "min_samples=2 must be accepted like sklearn");
}

/// REQ-11. `max_eps == 0` is ACCEPTED (`"max_eps": [Interval(Real, 0, None,
/// closed="both")]`, `sklearn/cluster/_optics.py:247` — closed at 0). The fit
/// RUNS and produces the degenerate result: every `core_distances_` and
/// `reachability_` is `inf`, `ordering_` is by index, `labels_` are all `-1`.
///
/// LIVE ORACLE (sklearn 1.5.2, run from /tmp):
///   python3 -c "import numpy as np, warnings; from sklearn.cluster import OPTICS; \
///     warnings.simplefilter('ignore'); \
///     X=np.array([[0.,0.],[0.1,0.],[0.,0.1],[5.,5.],[5.1,5.],[5.,5.1],[10.,0.],[10.1,0.],[10.,0.1]]); \
///     m=OPTICS(min_samples=2, max_eps=0).fit(X); \
///     print(m.core_distances_.tolist()); print(m.reachability_.tolist()); \
///     print(m.ordering_.tolist()); print(m.labels_.tolist())"
///   -> core_distances_: [inf, inf, inf, inf, inf, inf, inf, inf, inf]
///      reachability_:   [inf, inf, inf, inf, inf, inf, inf, inf, inf]
///      ordering_:       [0, 1, 2, 3, 4, 5, 6, 7, 8]
///      labels_:         [-1, -1, -1, -1, -1, -1, -1, -1, -1]
#[test]
fn green_max_eps_zero_degenerate_matches_oracle() {
    // LIVE sklearn 1.5.2 oracle (above), never copied from ferrolearn.
    let sk_ordering: [usize; 9] = [0, 1, 2, 3, 4, 5, 6, 7, 8];
    let sk_labels: [isize; 9] = [-1; 9];
    let fitted = OPTICS::<f64>::new(2)
        .with_max_eps(0.0)
        .fit(&three_blobs(), &())
        .expect("max_eps=0 must be accepted (sklearn closed at 0)");
    let core = fitted.core_distances();
    let reach = fitted.reachability();
    for i in 0..9 {
        assert!(
            core[i].is_infinite(),
            "core_distances_[{i}]: ferro={} sklearn=inf",
            core[i]
        );
        assert!(
            reach[i].is_infinite(),
            "reachability_[{i}]: ferro={} sklearn=inf",
            reach[i]
        );
    }
    assert_eq!(
        fitted.ordering(),
        &sk_ordering[..],
        "max_eps=0 ordering_ must be by index, matching sklearn"
    );
    assert_eq!(
        fitted.labels().as_slice().unwrap(),
        &sk_labels[..],
        "max_eps=0 labels_ must be all -1, matching sklearn"
    );
}

/// REQ-11. A normal finite `max_eps == 0.5` still works as before (the boundary
/// change only opened up `max_eps == 0`; positive values are unaffected).
///
/// LIVE ORACLE (sklearn 1.5.2, run from /tmp):
///   python3 -c "import numpy as np; from sklearn.cluster import OPTICS; \
///     X=np.array([[0.,0.],[0.1,0.],[0.,0.1],[5.,5.],[5.1,5.],[5.,5.1],[10.,0.],[10.1,0.],[10.,0.1]]); \
///     print(OPTICS(min_samples=2, max_eps=0.5).fit(X).core_distances_.tolist())"
///   -> [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
#[test]
fn green_max_eps_half_unchanged() {
    let fitted = OPTICS::<f64>::new(2)
        .with_max_eps(0.5)
        .fit(&three_blobs(), &())
        .expect("max_eps=0.5 must be accepted");
    let core = fitted.core_distances();
    for i in 0..9 {
        assert!(
            (core[i] - 0.1).abs() <= TOL,
            "core_distances_[{i}]: ferro={} sklearn=0.1",
            core[i]
        );
    }
}

/// REQ-11. `max_eps < 0` is REJECTED. sklearn's `Interval(Real, 0, None,
/// closed="both")` (`sklearn/cluster/_optics.py:247`) excludes negatives.
///
/// LIVE ORACLE (sklearn 1.5.2, run from /tmp):
///   python3 -c "import numpy as np; from sklearn.cluster import OPTICS; \
///     X=np.array([[0.,0.],[0.1,0.],[0.,0.1],[5.,5.],[5.1,5.],[5.,5.1],[10.,0.],[10.1,0.],[10.,0.1]]); \
///     OPTICS(min_samples=2, max_eps=-1.0).fit(X)"
///   -> InvalidParameterError: The 'max_eps' parameter of OPTICS must be a float
///      in the range [0, inf). Got -1.0 instead.
#[test]
fn green_max_eps_negative_rejected() {
    let r = OPTICS::<f64>::new(2)
        .with_max_eps(-1.0)
        .fit(&three_blobs(), &());
    assert!(r.is_err(), "max_eps<0 must be rejected like sklearn");
}

/// REQ-11. `xi == 0` is ACCEPTED (`"xi": [Interval(Real, 0, 1, closed="both")]`,
/// `sklearn/cluster/_optics.py:253` — closed at 0). (Matching the Xi `labels_`
/// VALUE for `xi=0` is REQ-5, NOT-STARTED; only the param is ACCEPTED here.)
///
/// LIVE ORACLE (sklearn 1.5.2, run from /tmp): fits without error
///   python3 -c "import numpy as np, warnings; from sklearn.cluster import OPTICS; \
///     warnings.simplefilter('ignore'); \
///     X=np.array([[0.,0.],[0.1,0.],[0.,0.1],[5.,5.],[5.1,5.],[5.,5.1],[10.,0.],[10.1,0.],[10.,0.1]]); \
///     print(OPTICS(min_samples=2, xi=0.0).fit(X).labels_.tolist())"
///   -> [0, 0, 0, 1, 1, 1, 2, 2, 2]   (no error — accepted)
#[test]
fn green_xi_zero_accepted() {
    let r = OPTICS::<f64>::new(2).with_xi(0.0).fit(&three_blobs(), &());
    assert!(r.is_ok(), "xi=0 must be accepted like sklearn");
}

/// REQ-11. `xi == 1` is ACCEPTED by sklearn's PARAMETER VALIDATION
/// (`Interval(Real, 0, 1, closed="both")`, `sklearn/cluster/_optics.py:253` —
/// closed at 1). sklearn's `_xi_cluster` then raises a runtime
/// `ZeroDivisionError` (because `1 - xi == 0`), but that is downstream of the
/// accept/reject BOUNDARY this REQ governs. ferrolearn's Xi extraction computes
/// the same ratios in Rust float arithmetic (`1.0/0.0 == inf`, `inf/inf == NaN`,
/// all-false comparisons), so it does NOT panic (R-CODE-2) and returns labels.
///
/// LIVE ORACLE (sklearn 1.5.2, run from /tmp): parameter validation ACCEPTS
///   python3 -c "import numpy as np; from sklearn.cluster import OPTICS; \
///     X=np.array([[0.,0.],[0.1,0.],[0.,0.1],[5.,5.],[5.1,5.],[5.,5.1],[10.,0.],[10.1,0.],[10.,0.1]]); \
///     OPTICS(min_samples=2, xi=1.0).fit(X)"
///   -> ZeroDivisionError: float division by zero  (NOT InvalidParameterError —
///      the parameter passed _parameter_constraints; the error is a runtime
///      1-xi==0 divide inside _xi_cluster, not a validation rejection)
#[test]
fn green_xi_one_accepted_no_panic() {
    // The BOUNDARY (parameter validation) ACCEPTS xi=1.0; ferrolearn must not
    // reject it at validation and must not panic.
    let r = OPTICS::<f64>::new(2).with_xi(1.0).fit(&three_blobs(), &());
    assert!(
        r.is_ok(),
        "xi=1 must pass parameter validation (sklearn closed at 1) and not panic"
    );
}

/// REQ-11. `xi < 0` is REJECTED.
///
/// LIVE ORACLE (sklearn 1.5.2, run from /tmp):
///   python3 -c "import numpy as np; from sklearn.cluster import OPTICS; \
///     X=np.array([[0.,0.],[0.1,0.],[0.,0.1],[5.,5.],[5.1,5.],[5.,5.1],[10.,0.],[10.1,0.],[10.,0.1]]); \
///     OPTICS(min_samples=2, xi=-0.1).fit(X)"
///   -> InvalidParameterError: The 'xi' parameter of OPTICS must be a float in
///      the range [0.0, 1.0]. Got -0.1 instead.
#[test]
fn green_xi_negative_rejected() {
    let r = OPTICS::<f64>::new(2).with_xi(-0.1).fit(&three_blobs(), &());
    assert!(r.is_err(), "xi<0 must be rejected like sklearn");
}

/// REQ-11. `xi > 1` is REJECTED.
///
/// LIVE ORACLE (sklearn 1.5.2, run from /tmp):
///   python3 -c "import numpy as np; from sklearn.cluster import OPTICS; \
///     X=np.array([[0.,0.],[0.1,0.],[0.,0.1],[5.,5.],[5.1,5.],[5.,5.1],[10.,0.],[10.1,0.],[10.,0.1]]); \
///     OPTICS(min_samples=2, xi=1.1).fit(X)"
///   -> InvalidParameterError: The 'xi' parameter of OPTICS must be a float in
///      the range [0.0, 1.0]. Got 1.1 instead.
#[test]
fn green_xi_above_one_rejected() {
    let r = OPTICS::<f64>::new(2).with_xi(1.1).fit(&three_blobs(), &());
    assert!(r.is_err(), "xi>1 must be rejected like sklearn");
}

// ─────────────────────────────────────────────────────────────────────────────
// REQ-11 UPPER-BOUND GAP (critic pin) — min_samples > n_samples.
//
// The shipped REQ-11 validation matched sklearn's LOWER bound for `min_samples`
// (`< 2` rejected, `== 2` accepted) but MISSED the UPPER bound. sklearn's
// `compute_optics_graph` calls `_validate_size(min_samples, n_samples,
// "min_samples")` (`sklearn/cluster/_optics.py:597`), which raises
//     ValueError("%s must be no greater than the number of samples (%d). Got %d")
// (`sklearn/cluster/_optics.py:393-400`) whenever `min_samples > n_samples`.
//
// ferrolearn's `Fit::fit` has NO upper-bound check: `core_distance` (optics.rs
// :396-411) merely returns `F::infinity()` when fewer than `min_samples-1`
// other points exist, so `fit` SUCCEEDS (Ok) returning an all-`inf`
// core_distances_/reachability_ and all-`-1` labels_ — exactly where sklearn
// ERRORS. This is an observable contract divergence (sklearn raises, ferrolearn
// returns Ok).
// ─────────────────────────────────────────────────────────────────────────────

/// Divergence: `ferrolearn_cluster::OPTICS::<f64>::fit` accepts `min_samples`
/// strictly greater than `n_samples` and returns `Ok`, whereas sklearn 1.5.2
/// RAISES `ValueError` via `_validate_size` (`sklearn/cluster/_optics.py:597`
/// calls `_validate_size` defined at `:393-400`:
/// `if size > n_samples: raise ValueError(...)`).
///
/// Input: `n_samples = 5`, `min_samples = 6` (6 > 5).
///
/// LIVE ORACLE (sklearn 1.5.2):
///   python3 -c "import numpy as np; from sklearn.cluster import OPTICS; \
///     X=np.array([[0.,0.],[0.1,0.],[0.,0.1],[5.,5.],[5.1,5.]]); \
///     OPTICS(min_samples=6).fit(X)"
///   -> ValueError: min_samples must be no greater than the number of samples
///      (5). Got 6
///
/// sklearn: fit RAISES (ValueError).
/// ferrolearn: fit returns Ok (core_distances_/reachability_ all inf, labels_
///   all -1) — no error.
///
/// Tracking: #2197
#[test]
#[ignore = "divergence: OPTICS::fit accepts min_samples > n_samples (sklearn _validate_size raises); tracking #2197"]
fn divergence_min_samples_above_n_samples_rejected() {
    // n_samples = 5 fixture (subset of three_blobs).
    let x = Array2::from_shape_vec(
        (5, 2),
        vec![0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 5.0, 5.0, 5.1, 5.0],
    )
    .unwrap();

    // sklearn (oracle above) RAISES ValueError for min_samples=6 > n_samples=5.
    // The OBSERVABLE contract is "fit errors"; the error TYPE is the
    // grandfathered FerroError (not sklearn's exception class). Assert ferrolearn
    // ALSO errors. It currently returns Ok, so this assertion FAILS — pinning the
    // missing upper-bound (`_validate_size`) check.
    let r = OPTICS::<f64>::new(6).fit(&x, &());
    assert!(
        r.is_err(),
        "min_samples=6 with n_samples=5 must be rejected like sklearn \
         (_validate_size, sklearn/cluster/_optics.py:597 -> :393-400 raises \
         ValueError); ferrolearn returned Ok"
    );
}
