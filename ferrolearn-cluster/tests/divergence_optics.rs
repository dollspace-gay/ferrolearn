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
