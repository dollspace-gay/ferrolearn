//! Adversarial divergence pins for `ferrolearn-neighbors/src/nearest_neighbors.rs`
//! (`NearestNeighbors` / `FittedNearestNeighbors`, `fit` / `kneighbors` /
//! `radius_neighbors`) against the live scikit-learn 1.5.2 oracle
//! (`from sklearn.neighbors import NearestNeighbors`, mirroring
//! `sklearn/neighbors/_unsupervised.py` + `sklearn/neighbors/_base.py`).
//!
//! Every expected value below is captured from a live `python3 -c "..."` run of
//! sklearn 1.5.2 (R-CHAR-3 — never literal-copied from the ferrolearn side). The
//! exact oracle call and its output is quoted above each assertion.
//!
//! Design doc: `.design/neighbors/nearest_neighbors.md` (commit 3ff1ca96).
//! Upstream: `sklearn/neighbors/_unsupervised.py` (the thin estimator) +
//! `sklearn/neighbors/_base.py` (`KNeighborsMixin.kneighbors`,
//! `RadiusNeighborsMixin.radius_neighbors`).
//!
//! ferrolearn API under test:
//!   * `NearestNeighbors::<F>::new().with_n_neighbors(k).with_algorithm(..)`
//!   * `fit(&x, &()) -> Result<FittedNearestNeighbors<F>>`
//!   * `kneighbors(&x, Option<k>) -> Result<(Array2<F>, Array2<usize>)>`,
//!     each row sorted nearest-first.
//!   * `radius_neighbors(&x, radius) -> Result<Vec<(Vec<F>, Vec<usize>)>>`,
//!     ALWAYS sorted ascending by distance.
//!
//! GREEN guards (must PASS — REQ-1/REQ-2 SHIPPED value contracts):
//!   * `green_kneighbors_value_explicit_x_tiefree` — REQ-1.
//!   * `green_kneighbors_self_query_includes_self`  — REQ-1 (explicit-X self).
//!   * `green_radius_neighbors_set_match`           — REQ-2.
//!   * `green_kneighbors_k_too_large_errors`        — REQ-3 (guard-exists).
//!
//! RED pin (must FAIL now — the one deterministic, single-file-fixable divergence):
//!   * `divergence_fit_does_not_error_when_n_neighbors_gt_n_samples` — REQ-4 / #872.
//!
//! ==========================================================================
//! NOT-STARTED / MISSING-SURFACE divergences — DOCUMENTED ONLY, no forced test.
//! ==========================================================================
//! Following the balltree/kdtree sibling precedent (`divergence_balltree.rs`),
//! the divergences below are NOT pinned with a committed RED test because they
//! require the generator to add API surface (a discriminator may not author
//! production code), or are inexpressible through the current public API:
//!
//! **`kneighbors(X=None)` self-exclusion** (#866): sklearn `kneighbors()` (no
//! `X`) sets `query_is_train = X is None` (`_base.py:815`), queries
//! `n_neighbors + 1` and drops each row's self-column (`_base.py:931-939`), so
//! row `i` excludes index `i`. Live oracle
//! `NearestNeighbors(n_neighbors=2).fit(X).kneighbors()` ->
//! `i=[[1,2],[3,0],[3,0],[1,2],[3,1]]`. ferrolearn `kneighbors` REQUIRES a query
//! matrix `x` — there is no `X=None` method, so the self-match (index `i`,
//! distance 0) is always returned. MISSING SURFACE — not runtime-pinnable
//! without inventing API.
//!
//! **`radius_neighbors` `sort_results` default** (#867): sklearn's default is
//! `sort_results=False` → native (tree/brute) order, NOT distance-sorted. Live
//! oracle (X2=[[10,10],[1,0],[0,1],[0,0],[1,1]], r=2.0, q=[[0.2,0.1]]): default
//! -> `i=[[1,2,3,4]]` (native, dist NOT ascending); `sort_results=True` ->
//! `i=[[3,1,2,4]]` (ascending). ferrolearn ALWAYS sorts ascending (matches
//! `sort_results=True`, diverges from the default order). A fix needs a
//! `sort_results` param API addition + consumer threading (graph.rs) — not a
//! single-file minimal fix. Documented, no forced RED.
//!
//! **k>n ValueError exact message/type** (#872 message portion): both ferrolearn
//! and sklearn ERROR when k > n_samples at `kneighbors` time, but the message
//! ("Expected n_neighbors <= n_samples_fit, ...", `_base.py:828-832`) + type
//! (`ValueError` vs `FerroError::InvalidParameter`) differ; likewise `k==0` ->
//! sklearn `ValueError("Expected n_neighbors > 0. Got 0")` (`:808`). The
//! guard-EXISTS contract is SHIPPED (pinned green by
//! `green_kneighbors_k_too_large_errors`); the EXACT-MESSAGE match is NOT-STARTED
//! (`FerroError` carries no Python message). Documented, no RED.
//!
//! **missing constructor params** `radius=1.0` / `metric='minkowski'` / `p=2` /
//! `metric_params` / `n_jobs` (#868), estimator `kneighbors_graph` `X=None`
//! (#869), PyO3 binding + meta-crate re-export (#870), ferray substrate (#871):
//! missing features / surface. A missing field/method is not runtime-pinnable
//! without inventing surface the impl does not have. Documented, no tests.

use ferrolearn_core::traits::Fit;
use ferrolearn_neighbors::NearestNeighbors;
use ndarray::{Array2, array};

/// Primary fixture: `X = [[0,0],[1,0],[0,1],[1,1],[10,10]]` (n_samples = 5).
fn five_points() -> Array2<f64> {
    array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [10.0, 10.0]]
}

// ===========================================================================
// GREEN 1 — REQ-1 (SHIPPED): explicit-query k-NN value parity (tie-free).
// ===========================================================================
//
// Query (0.2,0.1) is tie-free (distances [0.2236, 0.8062, 0.9220, 1.2042,
// 13.93], all distinct). Oracle (run from /tmp):
//   python3 -c "import numpy as np; from sklearn.neighbors import \
//   NearestNeighbors; d,i=NearestNeighbors(n_neighbors=3).fit(np.array( \
//   [[0.,0.],[1.,0.],[0.,1.],[1.,1.],[10.,10.]])).kneighbors(np.array( \
//   [[0.2,0.1]])); print(d.tolist(),i.tolist())"
//   -> [[0.223606797749979, 0.8062257748298549, 0.9219544457292888]] [[0,1,2]]
//
//   ...,n_neighbors=2) -> [[0.223606797749979, 0.8062257748298549]] [[0,1]]
#[test]
fn green_kneighbors_value_explicit_x_tiefree() {
    let x = five_points();
    let xq = array![[0.2, 0.1]];

    let nn = NearestNeighbors::<f64>::new().with_n_neighbors(3);
    let fitted = nn.fit(&x, &()).expect("fit should succeed");

    // k=3: distinct order [0, 1, 2].
    let (d3, i3) = fitted
        .kneighbors(&xq, Some(3))
        .expect("kneighbors k=3 should succeed");
    assert_eq!(d3.dim(), (1, 3), "k=3 shape");
    assert_eq!(i3[[0, 0]], 0, "k=3 idx0");
    assert_eq!(i3[[0, 1]], 1, "k=3 idx1");
    assert_eq!(i3[[0, 2]], 2, "k=3 idx2");
    let exp3 = [
        0.223_606_797_749_979,
        0.806_225_774_829_854_9,
        0.921_954_445_729_288_8,
    ];
    for (j, e) in exp3.iter().enumerate() {
        assert!(
            (d3[[0, j]] - e).abs() < 1e-12,
            "k=3 dist[{j}]: got {} expected {}",
            d3[[0, j]],
            e
        );
    }

    // k=2: distinct order [0, 1].
    let (d2, i2) = fitted
        .kneighbors(&xq, Some(2))
        .expect("kneighbors k=2 should succeed");
    assert_eq!(d2.dim(), (1, 2), "k=2 shape");
    assert_eq!(i2[[0, 0]], 0, "k=2 idx0");
    assert_eq!(i2[[0, 1]], 1, "k=2 idx1");
    let exp2 = [0.223_606_797_749_979, 0.806_225_774_829_854_9];
    for (j, e) in exp2.iter().enumerate() {
        assert!(
            (d2[[0, j]] - e).abs() < 1e-12,
            "k=2 dist[{j}]: got {} expected {}",
            d2[[0, j]],
            e
        );
    }
}

// ===========================================================================
// GREEN 2 — REQ-1 (SHIPPED): explicit-X self-query includes self.
// ===========================================================================
//
// sklearn includes the self-match when X is passed EXPLICITLY (no X=None
// self-removal): each row's nearest is itself at distance 0. Oracle (/tmp):
//   python3 -c "import numpy as np; from sklearn.neighbors import \
//   NearestNeighbors; X=np.array([[0.,0.],[1.,0.],[0.,1.],[1.,1.], \
//   [10.,10.]]); d,i=NearestNeighbors(n_neighbors=2).fit(X).kneighbors(X); \
//   print(d.tolist(),i.tolist())"
//   -> d=[[0.0,1.0],[0.0,1.0],[0.0,1.0],[0.0,1.0],[0.0,12.727922061357855]]
//      i=[[0,1],[1,0],[2,0],[3,1],[4,3]]
// Col 0 of each row is (row_index, 0.0).
#[test]
fn green_kneighbors_self_query_includes_self() {
    let x = five_points();
    let nn = NearestNeighbors::<f64>::new().with_n_neighbors(2);
    let fitted = nn.fit(&x, &()).expect("fit should succeed");

    let (d, i) = fitted
        .kneighbors(&x, Some(2))
        .expect("self-query kneighbors should succeed");
    assert_eq!(d.dim(), (5, 2), "self-query shape");

    for row in 0..5 {
        assert_eq!(
            i[[row, 0]],
            row,
            "explicit-X self-query: row {row} col-0 index must be the row itself"
        );
        assert!(
            d[[row, 0]].abs() < 1e-12,
            "explicit-X self-query: row {row} col-0 distance must be 0, got {}",
            d[[row, 0]]
        );
    }
}

// ===========================================================================
// GREEN 3 — REQ-2 (SHIPPED): radius_neighbors SET + distance-multiset parity.
// ===========================================================================
//
// ferrolearn always sorts ascending — which coincides with sklearn's
// sort_results=True. Oracle (/tmp), sort_results=True so the order is
// deterministic:
//   python3 -c "import numpy as np; from sklearn.neighbors import \
//   NearestNeighbors; X=np.array([[0.,0.],[1.,0.],[0.,1.],[1.,1.], \
//   [10.,10.]]); Xq=np.array([[0.2,0.1]]); \
//   d,i=NearestNeighbors(radius=1.5).fit(X).radius_neighbors(Xq, \
//   sort_results=True); print([a.tolist() for a in i],[a.tolist() for a in d])"
//   -> i=[[0,1,2,3]]
//      d=[[0.223606797749979, 0.8062257748298549, 0.9219544457292888,
//          1.2041594578792296]]
#[test]
fn green_radius_neighbors_set_match() {
    let x = five_points();
    let xq = array![[0.2, 0.1]];
    let nn = NearestNeighbors::<f64>::new().with_n_neighbors(2);
    let fitted = nn.fit(&x, &()).expect("fit should succeed");

    let results = fitted
        .radius_neighbors(&xq, 1.5)
        .expect("radius_neighbors should succeed");
    assert_eq!(results.len(), 1, "one query row");
    let (dists, idxs) = &results[0];

    // Compare index SET (sort both sides).
    let mut got_idx = idxs.clone();
    got_idx.sort_unstable();
    assert_eq!(
        got_idx,
        vec![0, 1, 2, 3],
        "in-radius index set must be {{0,1,2,3}}"
    );

    // ferrolearn always-sorted coincides with sklearn sort_results=True order.
    let expected = [
        0.223_606_797_749_979,
        0.806_225_774_829_854_9,
        0.921_954_445_729_288_8,
        1.204_159_457_879_229_6,
    ];
    assert_eq!(idxs.as_slice(), &[0, 1, 2, 3], "sorted index order");
    assert_eq!(dists.len(), 4, "four in-radius distances");
    for (j, e) in expected.iter().enumerate() {
        assert!(
            (dists[j] - e).abs() < 1e-12,
            "in-radius dist[{j}]: got {} expected {}",
            dists[j],
            e
        );
    }
}

// ===========================================================================
// GREEN 4 — REQ-3 (SHIPPED): k > n_samples error guard EXISTS at query time.
// ===========================================================================
//
// Both ferrolearn and sklearn ERROR when k > n_samples_fit at kneighbors time;
// only the type/message differ (REQ-4 / #872, documented above). Here we pin
// only that the guard EXISTS (the SHIPPED contract). Oracle (/tmp):
//   python3 -c "import numpy as np; from sklearn.neighbors import \
//   NearestNeighbors; NearestNeighbors(n_neighbors=2).fit(np.array( \
//   [[0.,0.],[1.,0.],[0.,1.],[1.,1.],[10.,10.]])).kneighbors(np.array( \
//   [[0.2,0.1]]),n_neighbors=100)"
//   -> ValueError: Expected n_neighbors <= n_samples_fit, but n_neighbors = 100,
//      n_samples_fit = 5, n_samples = 1
// (n_samples_fit = 5; k=100 > 5 -> error. ferrolearn returns Err likewise.)
#[test]
fn green_kneighbors_k_too_large_errors() {
    let x = five_points();
    let xq = array![[0.2, 0.1]];
    let nn = NearestNeighbors::<f64>::new().with_n_neighbors(2);
    let fitted = nn.fit(&x, &()).expect("fit should succeed");

    assert!(
        fitted.kneighbors(&xq, Some(100)).is_err(),
        "k=100 > n_samples_fit=5 must error at kneighbors time (sklearn raises \
         ValueError; ferrolearn raises FerroError — REQ-4/#872 covers the \
         type/message divergence)"
    );
}

// ===========================================================================
// RED 1 — REQ-4 / #872 (DIVERGENCE, must FAIL now): fit timing.
// ===========================================================================
//
// sklearn does NOT validate n_neighbors vs n_samples at FIT time — the fit
// succeeds and the check is deferred to kneighbors() query time. Oracle (/tmp):
//   python3 -c "
//   import numpy as np
//   from sklearn.neighbors import NearestNeighbors
//   m=NearestNeighbors(n_neighbors=3).fit(np.array([[0.,0.],[1.,1.]]))
//   print('fit ok')
//   try:
//       m.kneighbors()
//   except ValueError as e:
//       print('kneighbors raises:', e)"
//   -> fit ok
//      kneighbors raises: Expected n_neighbors < n_samples_fit, but
//      n_neighbors = 3, n_samples_fit = 2, n_samples = 2
//
// sklearn `fit` SUCCEEDS. ferrolearn `fit` returns Err(InsufficientSamples) at
// FIT time (`nearest_neighbors.rs` fn fit: `if n_samples < self.n_neighbors`)
// — a fit-time guard with NO sklearn analog. This assertion (fit is Ok) FAILS
// against current ferrolearn, pinning the divergence. Minimally fixable: remove
// the fit-time `n_samples < n_neighbors` guard; the k>n check already exists at
// kneighbors time (green_kneighbors_k_too_large_errors).
#[test]
fn divergence_fit_does_not_error_when_n_neighbors_gt_n_samples() {
    // X has 2 rows; n_neighbors = 3 > 2.
    let x_2rows = array![[0.0, 0.0], [1.0, 1.0]];
    let nn = NearestNeighbors::<f64>::new().with_n_neighbors(3);

    assert!(
        nn.fit(&x_2rows, &()).is_ok(),
        "sklearn NearestNeighbors(n_neighbors=3).fit(X_with_2_rows) SUCCEEDS \
         (no fit-time n_neighbors-vs-n_samples validation; deferred to \
         kneighbors). ferrolearn fit returns Err(InsufficientSamples) — a \
         fit-time guard with no sklearn analog. (#872)"
    );
}
