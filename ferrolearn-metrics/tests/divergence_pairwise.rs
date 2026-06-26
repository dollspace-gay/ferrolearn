//! Divergence pins for `ferrolearn-metrics/src/pairwise.rs` vs scikit-learn 1.5.2.
//!
//! All expected values come from the LIVE sklearn 1.5.2 oracle (the exact
//! `python3 -c` call is quoted in each test, R-CHAR-3) — NEVER copied from the
//! ferrolearn side.
//!
//! VERDICT (this audit): NO deterministic VALUE divergence was found in the
//! PRESENT functions. The present functions (`euclidean_distances`,
//! `manhattan_distances`, `cosine_similarity`, `cosine_distances`,
//! `chebyshev_distances`,
//! `nan_euclidean_distances`, `pairwise_distances` via `Metric`,
//! `paired_*`, `pairwise_distances_argmin`/`_argmin_min`,
//! `pairwise_kernels` rbf)
//! reproduce sklearn's exact float64 values on every case probed — including
//! the rounding-level results (e.g. `cosine_distances([[1,1]],[[-1,-1]]) ==
//! 1.9999999999999998` matches sklearn to the ULP).
//!
//! The known divergences enumerated by the design doc (`.design/metrics/
//! pairwise.md`) are ALL ABI / missing-surface gaps that are NOT pinnable as
//! failing tests on the existing API (a missing function/param cannot be
//! called): no `squared=`, no `missing_values=`, no string-metric ABI, no
//! kernel defaults, missing `chi2`/`additive_chi2`/`cosine` kernels,
//! `haversine`/`chunked`, no PyO3 binding. They are tracked as
//! NOT-STARTED REQs (blockers #791-#803), not as RED pins.
//!
//! Two diagonal probes were run and did NOT reproduce a divergence:
//!   - euclidean `X==X` diagonal (f32 AND large-magnitude f64): ferrolearn
//!     accumulates `dot(x_i, x_i)` with the SAME fold as `||x_i||^2`, so the
//!     squared distance is EXACTLY `0.0` on the self-pair — matching sklearn's
//!     `np.fill_diagonal(., 0)` result without needing an identity check.
//!   - cosine self-pair (distinct objects): ferrolearn and sklearn both produce
//!     `~1.1e-16` rounding noise; the noise lands in different cells but at the
//!     same magnitude — symmetric and non-robust, NOT a pinnable value
//!     divergence (pinning it would be fragile/tautological).
//!
//! GREEN guards below establish the SHIPPED value contracts: each present
//! function == a live sklearn 1.5.2 value. These are the evidence the value
//! contracts are correct and must stay green.

use ferrolearn_metrics::pairwise::{
    Metric, PairwiseKernel, chebyshev_distances, cosine_distances, cosine_similarity,
    euclidean_distances, manhattan_distances, nan_euclidean_distances, paired_cosine_distances,
    paired_distances, paired_euclidean_distances, paired_manhattan_distances, pairwise_distances,
    pairwise_distances_argmin, pairwise_distances_argmin_min, pairwise_kernels,
};
use ndarray::array;

// ===========================================================================
// GREEN guards — oracle-grounded SHIPPED value contracts (must pass now)
// ===========================================================================

/// Guard: `euclidean_distances` basic values match sklearn.
///
/// Oracle (sklearn 1.5.2, run from /tmp):
///   python3 -c "import numpy as np; from sklearn.metrics.pairwise import \
///     euclidean_distances as e; \
///     print(e(np.array([[0.,0],[3,0]]), np.array([[0.,4]])).ravel().tolist())"
///   # [4.0, 5.0]
#[test]
fn green_euclidean_basic() {
    let x = array![[0.0_f64, 0.0], [3.0, 0.0]];
    let y = array![[0.0_f64, 4.0]];
    let d = euclidean_distances(&x, &y).unwrap();
    // sklearn 1.5.2 live oracle: [4.0, 5.0].
    const SK: [f64; 2] = [4.0, 5.0];
    assert!(
        (d[[0, 0]] - SK[0]).abs() < 1e-12 && (d[[1, 0]] - SK[1]).abs() < 1e-12,
        "euclidean: sklearn={SK:?}, ferrolearn=[{},{}]",
        d[[0, 0]],
        d[[1, 0]]
    );
}

/// Guard: `euclidean_distances(X, X)` diagonal is exactly 0.0, matching
/// sklearn's `np.fill_diagonal(., 0)` self-call guarantee — even for
/// large-magnitude f64 rows (ferrolearn reaches it via same-accumulation, not
/// an identity check).
///
/// Oracle (sklearn 1.5.2, run from /tmp):
///   python3 -c "import numpy as np; from sklearn.metrics.pairwise import \
///     euclidean_distances as e; X=np.array([[1e8,1e8],[1e8+1,1e8-1]]); \
///     d=e(X,X); print(repr(d[0,0]), repr(d[1,1]))"
///   # 0.0 0.0
#[test]
fn green_euclidean_self_diagonal_zero() {
    let x = array![[1e8_f64, 1e8], [1e8 + 1.0, 1e8 - 1.0]];
    let d = euclidean_distances(&x, &x).unwrap();
    // sklearn 1.5.2 live oracle: self-pair diagonal == 0.0 exactly.
    assert_eq!(d[[0, 0]], 0.0, "euclidean self-diag[0,0]: sklearn 0.0");
    assert_eq!(d[[1, 1]], 0.0, "euclidean self-diag[1,1]: sklearn 0.0");
}

/// Guard: `manhattan_distances` basic values match sklearn.
///
/// Oracle (sklearn 1.5.2, run from /tmp):
///   python3 -c "import numpy as np; from sklearn.metrics.pairwise import \
///     manhattan_distances as m; \
///     print(m(np.array([[0.,0],[1,1]]), np.array([[1.,1]])).ravel().tolist())"
///   # [2.0, 0.0]
#[test]
fn green_manhattan_basic() {
    let x = array![[0.0_f64, 0.0], [1.0, 1.0]];
    let y = array![[1.0_f64, 1.0]];
    let d = manhattan_distances(&x, &y).unwrap();
    // sklearn 1.5.2 live oracle: [2.0, 0.0].
    const SK: [f64; 2] = [2.0, 0.0];
    assert!(
        (d[[0, 0]] - SK[0]).abs() < 1e-12 && (d[[1, 0]] - SK[1]).abs() < 1e-12,
        "manhattan: sklearn={SK:?}, ferrolearn=[{},{}]",
        d[[0, 0]],
        d[[1, 0]]
    );
}

/// Guard: `cosine_distances` edge values match sklearn — orthogonal → 1.0,
/// zero-vector → 1.0, and opposite vectors → the exact `1.9999999999999998`
/// sklearn computes (NOT a rounded `2.0`).
///
/// Oracle (sklearn 1.5.2, run from /tmp):
///   python3 -c "import numpy as np; from sklearn.metrics.pairwise import \
///     cosine_distances as c; \
///     print(repr(c(np.array([[1.,0]]),np.array([[0.,1]])).ravel()[0]), \
///           repr(c(np.array([[1.,0]]),np.array([[0.,0]])).ravel()[0]), \
///           repr(c(np.array([[1.,1]]),np.array([[-1.,-1]])).ravel()[0]))"
///   # 1.0 1.0 1.9999999999999998
#[test]
fn green_cosine_edges() {
    let d_ortho = cosine_distances(&array![[1.0_f64, 0.0]], &array![[0.0_f64, 1.0]]).unwrap();
    let d_zero = cosine_distances(&array![[1.0_f64, 0.0]], &array![[0.0_f64, 0.0]]).unwrap();
    let d_opp = cosine_distances(&array![[1.0_f64, 1.0]], &array![[-1.0_f64, -1.0]]).unwrap();
    // sklearn 1.5.2 live oracle.
    const SK_ORTHO: f64 = 1.0;
    const SK_ZERO: f64 = 1.0;
    const SK_OPP: f64 = 1.9999999999999998;
    assert_eq!(d_ortho[[0, 0]], SK_ORTHO, "cosine orthogonal: sklearn 1.0");
    assert_eq!(d_zero[[0, 0]], SK_ZERO, "cosine zero-vector: sklearn 1.0");
    // Exact ULP match: sklearn produces 1.9999999999999998, NOT 2.0.
    assert_eq!(
        d_opp[[0, 0]],
        SK_OPP,
        "cosine opposite: sklearn 1.9999999999999998, ferrolearn {}",
        d_opp[[0, 0]]
    );
}

/// Guard: `cosine_similarity` matches sklearn's normalized dot product,
/// including zero-row behavior (similarity 0.0).
///
/// Oracle (sklearn 1.5.2, run from /tmp):
///   python3 -c "import numpy as np; from sklearn.metrics.pairwise import \
///     cosine_similarity as c; X=np.array([[0.,0.,0.],[1.,1.,1.]]); \
///     Y=np.array([[1.,0.,0.],[1.,1.,0.]]); print(c(X,Y).tolist())"
///   # [[0.0, 0.0], [0.5773502691896258, 0.816496580927726]]
#[test]
fn green_cosine_similarity_matches_sklearn() {
    let x = array![[0.0_f64, 0.0, 0.0], [1.0, 1.0, 1.0]];
    let y = array![[1.0_f64, 0.0, 0.0], [1.0, 1.0, 0.0]];
    let sim = cosine_similarity(&x, &y).unwrap();
    const SK: [[f64; 2]; 2] = [[0.0, 0.0], [0.5773502691896258, 0.816496580927726]];
    for i in 0..2 {
        for j in 0..2 {
            assert!(
                (sim[[i, j]] - SK[i][j]).abs() < 1e-15,
                "cosine_similarity[{i},{j}]: sklearn={}, ferrolearn={}",
                SK[i][j],
                sim[[i, j]]
            );
        }
    }
}

/// Guard: `paired_euclidean_distances` returns a length-`n` row-wise vector.
///
/// Oracle (sklearn 1.5.2, run from /tmp):
///   python3 -c "import numpy as np; from sklearn.metrics.pairwise import \
///     paired_euclidean_distances as p; X=np.array([[0.,0.,0.],[1.,1.,1.]]); \
///     Y=np.array([[1.,0.,0.],[1.,1.,0.]]); print(p(X,Y).tolist())"
///   # [1.0, 1.0]
#[test]
fn green_paired_euclidean_matches_sklearn() {
    let x = array![[0.0_f64, 0.0, 0.0], [1.0, 1.0, 1.0]];
    let y = array![[1.0_f64, 0.0, 0.0], [1.0, 1.0, 0.0]];
    let distances = paired_euclidean_distances(&x, &y).unwrap();
    const SK: [f64; 2] = [1.0, 1.0];
    for i in 0..2 {
        assert!(
            (distances[i] - SK[i]).abs() < 1e-12,
            "paired_euclidean[{i}]: sklearn={}, ferrolearn={}",
            SK[i],
            distances[i]
        );
    }
}

/// Guard: `paired_manhattan_distances` matches sklearn's row-wise L1 output.
///
/// Oracle (sklearn 1.5.2, run from /tmp):
///   python3 -c "import numpy as np; from sklearn.metrics.pairwise import \
///     paired_manhattan_distances as p; X=np.array([[1.,1.,0.],[0.,1.,0.],\
///     [0.,0.,1.]]); Y=np.array([[0.,1.,0.],[0.,0.,1.],[0.,0.,0.]]); \
///     print(p(X,Y).tolist())"
///   # [1.0, 2.0, 1.0]
#[test]
fn green_paired_manhattan_matches_sklearn() {
    let x = array![[1.0_f64, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let y = array![[0.0_f64, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]];
    let distances = paired_manhattan_distances(&x, &y).unwrap();
    const SK: [f64; 3] = [1.0, 2.0, 1.0];
    for i in 0..3 {
        assert!(
            (distances[i] - SK[i]).abs() < 1e-12,
            "paired_manhattan[{i}]: sklearn={}, ferrolearn={}",
            SK[i],
            distances[i]
        );
    }
}

/// Guard: `paired_cosine_distances` matches sklearn's normalized-difference
/// implementation, including zero-row vs nonzero-row distance `0.5`.
///
/// Oracle (sklearn 1.5.2, run from /tmp):
///   python3 -c "import numpy as np; from sklearn.metrics.pairwise import \
///     paired_cosine_distances as p; X=np.array([[0.,0.,0.],[1.,1.,1.]]); \
///     Y=np.array([[1.,0.,0.],[1.,1.,0.]]); print(p(X,Y).tolist())"
///   # [0.5, 0.183503419072274]
#[test]
fn green_paired_cosine_matches_sklearn() {
    let x = array![[0.0_f64, 0.0, 0.0], [1.0, 1.0, 1.0]];
    let y = array![[1.0_f64, 0.0, 0.0], [1.0, 1.0, 0.0]];
    let distances = paired_cosine_distances(&x, &y).unwrap();
    const SK: [f64; 2] = [0.5, 0.183503419072274];
    for i in 0..2 {
        assert!(
            (distances[i] - SK[i]).abs() < 1e-15,
            "paired_cosine[{i}]: sklearn={}, ferrolearn={}",
            SK[i],
            distances[i]
        );
    }
}

/// Guard: `paired_distances` dispatches to sklearn's supported paired metrics
/// and rejects Chebyshev, which sklearn does not allow for paired distances.
///
/// Oracle (sklearn 1.5.2, run from /tmp):
///   python3 -c "import numpy as np; from sklearn.metrics.pairwise import \
///     paired_distances as p; print(p(np.array([[0.,1.],[1.,1.]]), \
///     np.array([[0.,1.],[2.,1.]]), metric='euclidean').tolist())"
///   # [0.0, 1.0]
#[test]
fn green_paired_distances_dispatch_matches_sklearn() {
    let x = array![[0.0_f64, 1.0], [1.0, 1.0]];
    let y = array![[0.0_f64, 1.0], [2.0, 1.0]];
    let distances = paired_distances(&x, &y, Metric::Euclidean).unwrap();
    const SK: [f64; 2] = [0.0, 1.0];
    for i in 0..2 {
        assert!(
            (distances[i] - SK[i]).abs() < 1e-12,
            "paired_distances(Euclidean)[{i}]: sklearn={}, ferrolearn={}",
            SK[i],
            distances[i]
        );
    }
    assert!(
        paired_distances(&x, &y, Metric::Chebyshev).is_err(),
        "sklearn rejects metric='chebyshev' for paired_distances"
    );
}

/// Guard: `chebyshev_distances` matches `pairwise_distances(metric='chebyshev')`
/// (the only ABI sklearn exposes for Chebyshev).
///
/// Oracle (sklearn 1.5.2, run from /tmp):
///   python3 -c "import numpy as np; from sklearn.metrics import \
///     pairwise_distances; \
///     print(pairwise_distances(np.array([[0.,0],[3,4]]),np.array([[1.,2]]), \
///       metric='chebyshev').ravel().tolist())"
///   # [2.0, 2.0]
#[test]
fn green_chebyshev_basic() {
    let x = array![[0.0_f64, 0.0], [3.0, 4.0]];
    let y = array![[1.0_f64, 2.0]];
    let d = chebyshev_distances(&x, &y).unwrap();
    // sklearn 1.5.2 live oracle (via metric='chebyshev'): [2.0, 2.0].
    const SK: [f64; 2] = [2.0, 2.0];
    assert!(
        (d[[0, 0]] - SK[0]).abs() < 1e-12 && (d[[1, 0]] - SK[1]).abs() < 1e-12,
        "chebyshev: sklearn={SK:?}, ferrolearn=[{},{}]",
        d[[0, 0]],
        d[[1, 0]]
    );
}

/// Guard: `nan_euclidean_distances` `sqrt(n_features/n_present * sq)` scaling
/// matches sklearn to ~1e-12, both for a single missing feature and for a
/// disjoint-present (2-of-3 missing) pair.
///
/// Oracle (sklearn 1.5.2, run from /tmp):
///   python3 -c "import numpy as np; from sklearn.metrics.pairwise import \
///     nan_euclidean_distances as n; \
///     print(repr(n(np.array([[np.nan,3.]]),np.array([[0.,0]])).ravel()[0]), \
///       repr(n(np.array([[np.nan,1.,2.]]),np.array([[3.,np.nan,5.]])).ravel()[0]))"
///   # 4.242640687119285 5.196152422706632
#[test]
fn green_nan_euclidean_scaling() {
    let d1 = nan_euclidean_distances(&array![[f64::NAN, 3.0]], &array![[0.0_f64, 0.0]]).unwrap();
    let d2 = nan_euclidean_distances(
        &array![[f64::NAN, 1.0, 2.0]],
        &array![[3.0_f64, f64::NAN, 5.0]],
    )
    .unwrap();
    // sklearn 1.5.2 live oracle.
    const SK1: f64 = 4.242640687119285; // sqrt(9 * 2/1) = 3*sqrt(2)
    const SK2: f64 = 5.196152422706632; // sqrt(9 * 3/1) = sqrt(27)
    assert!(
        (d1[[0, 0]] - SK1).abs() < 1e-12,
        "nan_euclidean single-present: sklearn={SK1}, ferrolearn={}",
        d1[[0, 0]]
    );
    assert!(
        (d2[[0, 0]] - SK2).abs() < 1e-12,
        "nan_euclidean disjoint-present: sklearn={SK2}, ferrolearn={}",
        d2[[0, 0]]
    );
}

/// Guard: an all-missing pair yields `NaN`, matching sklearn (`pairwise.py:540`
/// sets all-missing pairs to `np.nan`).
///
/// Oracle (sklearn 1.5.2, run from /tmp):
///   python3 -c "import numpy as np; from sklearn.metrics.pairwise import \
///     nan_euclidean_distances as n; \
///     print(n(np.array([[np.nan,np.nan]]),np.array([[1.,2]])).ravel())"
///   # [nan]
#[test]
fn green_nan_euclidean_all_missing() {
    let d =
        nan_euclidean_distances(&array![[f64::NAN, f64::NAN]], &array![[1.0_f64, 2.0]]).unwrap();
    // sklearn 1.5.2 live oracle: all-missing pair == nan.
    assert!(d[[0, 0]].is_nan(), "nan_euclidean all-missing: sklearn nan");
}

/// Guard: `pairwise_distances` via `Metric::Euclidean` == the direct
/// `euclidean_distances` AND == sklearn's `euclidean` string metric values.
///
/// Oracle (sklearn 1.5.2, run from /tmp):
///   python3 -c "import numpy as np; from sklearn.metrics import \
///     pairwise_distances as p; \
///     print(p(np.array([[0.,0],[3,0]]),np.array([[0.,4]]), \
///       metric='euclidean').ravel().tolist())"
///   # [4.0, 5.0]
#[test]
fn green_pairwise_distances_dispatch_euclidean() {
    let x = array![[0.0_f64, 0.0], [3.0, 0.0]];
    let y = array![[0.0_f64, 4.0]];
    let d = pairwise_distances(&x, &y, Metric::Euclidean).unwrap();
    let direct = euclidean_distances(&x, &y).unwrap();
    assert_eq!(d, direct, "dispatcher Euclidean must equal direct function");
    // sklearn 1.5.2 live oracle (metric='euclidean'): [4.0, 5.0].
    const SK: [f64; 2] = [4.0, 5.0];
    assert!(
        (d[[0, 0]] - SK[0]).abs() < 1e-12 && (d[[1, 0]] - SK[1]).abs() < 1e-12,
        "pairwise_distances(Euclidean): sklearn={SK:?}, ferrolearn=[{},{}]",
        d[[0, 0]],
        d[[1, 0]]
    );
}

/// Guard: `pairwise_distances_argmin` ties break to the LOWEST index, matching
/// sklearn.
///
/// Oracle (sklearn 1.5.2, run from /tmp):
///   python3 -c "import numpy as np; from sklearn.metrics import \
///     pairwise_distances_argmin as a; \
///     print(a(np.array([[0.,0]]),np.array([[1.,0],[1,0],[2,0]])).tolist())"
///   # [0]
#[test]
fn green_argmin_tie_break_to_first() {
    let x = array![[0.0_f64, 0.0]];
    let y = array![[1.0_f64, 0.0], [1.0, 0.0], [2.0, 0.0]];
    let idx = pairwise_distances_argmin(&x, &y, Metric::Euclidean).unwrap();
    // sklearn 1.5.2 live oracle: [0] (tie among indices 0 and 1 -> first).
    assert_eq!(idx[0], 0, "argmin tie: sklearn 0 (lowest index)");
}

/// Guard: `pairwise_distances_argmin_min` returns the tie-broken index AND the
/// minimum distance, matching sklearn.
///
/// Oracle (sklearn 1.5.2, run from /tmp):
///   python3 -c "import numpy as np; from sklearn.metrics import \
///     pairwise_distances_argmin_min as m; \
///     r=m(np.array([[0.,0]]),np.array([[1.,0],[1,0],[2,0]])); \
///     print(r[0].tolist(), r[1].tolist())"
///   # [0] [1.0]
#[test]
fn green_argmin_min_value() {
    let x = array![[0.0_f64, 0.0]];
    let y = array![[1.0_f64, 0.0], [1.0, 0.0], [2.0, 0.0]];
    let (idx, mins) = pairwise_distances_argmin_min(&x, &y, Metric::Euclidean).unwrap();
    // sklearn 1.5.2 live oracle: ([0], [1.0]).
    assert_eq!(idx[0], 0, "argmin_min idx: sklearn 0");
    assert!(
        (mins[0] - 1.0).abs() < 1e-12,
        "argmin_min min: sklearn 1.0, ferrolearn {}",
        mins[0]
    );
}

/// Guard: `pairwise_kernels` RBF at `gamma=0.5` matches sklearn `rbf_kernel`
/// (`gamma=0.5` is sklearn's default `1/n_features` for these 2-feature rows).
///
/// Oracle (sklearn 1.5.2, run from /tmp):
///   python3 -c "import numpy as np; from sklearn.metrics.pairwise import \
///     rbf_kernel as r; \
///     print(r(np.array([[1.,2],[3,4]]),np.array([[1.,0]]),gamma=0.5).ravel().tolist())"
///   # [0.1353352832366127, 4.5399929762484854e-05]
#[test]
fn green_pairwise_kernels_rbf() {
    let x = array![[1.0_f64, 2.0], [3.0, 4.0]];
    let y = array![[1.0_f64, 0.0]];
    let k = pairwise_kernels(&x, &y, PairwiseKernel::Rbf { gamma: 0.5 }).unwrap();
    // sklearn 1.5.2 live oracle.
    const SK0: f64 = 0.1353352832366127;
    const SK1: f64 = 4.5399929762484854e-05;
    assert!(
        (k[[0, 0]] - SK0).abs() < 1e-15,
        "rbf[0]: sklearn={SK0}, ferrolearn={}",
        k[[0, 0]]
    );
    assert!(
        (k[[1, 0]] - SK1).abs() < 1e-18,
        "rbf[1]: sklearn={SK1}, ferrolearn={}",
        k[[1, 0]]
    );
}
