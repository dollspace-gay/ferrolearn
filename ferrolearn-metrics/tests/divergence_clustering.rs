//! Divergence pins for `ferrolearn-metrics/src/clustering.rs` vs scikit-learn 1.5.2.
//!
//! All expected values come from the LIVE sklearn 1.5.2 oracle (the exact
//! `python3 -c` call is quoted in each test, R-CHAR-3) — NEVER copied from the
//! ferrolearn side. The clone at `/home/doll/scikit-learn` and the installed
//! `sklearn` package are both at tag 1.5.2 (commit 156ef14), so the source line
//! citations and the oracle values are the same version by construction.
//!
//! RED pins (deterministic divergences to be fixed this iteration):
//!   - `red_rand_score_degenerate_pred` — `rand_score([0,0,1,1],[0,0,0,0])`
//!     PANICS in debug (`attempt to subtract with overflow`, the `u64`
//!     `d = comb_n - sum_comb_a - sum_comb_b + sum_comb_c` underflows). sklearn
//!     returns 0.3333333333333333.
//!   - `red_calinski_harabasz_intra_disp_zero` — coincident-per-cluster points
//!     give `Ok(inf)`; sklearn returns 1.0 (`intra_disp == 0.0` guard).
//!   - `red_homogeneity_family_empty` — empty input returns
//!     `Err(InsufficientSamples)`; sklearn returns 1.0 / (1.0, 1.0, 1.0).
//!
//! GREEN guards establish the SHIPPED value contracts: each present function ==
//! a live sklearn 1.5.2 value. They must stay green.

use ferrolearn_metrics::clustering::{
    NmiMethod, adjusted_mutual_info, adjusted_rand_score, calinski_harabasz_score,
    completeness_score, contingency_matrix, davies_bouldin_score, fowlkes_mallows_score,
    homogeneity_completeness_v_measure, homogeneity_score, mutual_info_score,
    normalized_mutual_info_score, pair_confusion_matrix, rand_score, silhouette_score,
    v_measure_score,
};
use ndarray::{Array1, Array2, array};

fn labels(v: &[isize]) -> Array1<isize> {
    Array1::from(v.to_vec())
}

// ===========================================================================
// RED pins — deterministic divergences to be fixed this iteration
// ===========================================================================

/// RED — `rand_score` underflows/panics on a valid degenerate labeling.
///
/// ferrolearn `rand_score` (clustering.rs `pub fn rand_score`) computes
/// `d = comb_n - sum_comb_a - sum_comb_b + sum_comb_c` in **`u64`**
/// (clustering.rs:1460); the subtraction underflows BEFORE `+ sum_comb_c`, so
/// in a debug build this PANICS with `attempt to subtract with overflow`. The
/// test FAILS by panic — that panic IS the divergence (R-CODE-2: no panic in
/// library code). sklearn never panics; it special-cases
/// `numerator == denominator or denominator == 0 -> 1.0` (_supervised.py:337-341)
/// and otherwise returns `numerator / denominator` (_supervised.py:343).
///
/// Oracle (sklearn 1.5.2, run from /tmp):
///   python3 -c "from sklearn.metrics import rand_score; \
///     print(repr(rand_score([0,0,1,1],[0,0,0,0])))"
///   # np.float64(0.3333333333333333)
///
/// Tracking: #797
#[test]
fn red_rand_score_degenerate_pred() {
    let lt = labels(&[0, 0, 1, 1]);
    let lp = labels(&[0, 0, 0, 0]);
    // In debug this panics (subtract-with-overflow) at the `d = ...` line; the
    // panic IS the divergence. Once the fixer reorders/saturates, the value
    // must equal the sklearn oracle.
    let ri = rand_score(&lt, &lp).expect("rand_score should not error on valid 4-sample input");
    // sklearn 1.5.2 live oracle: 0.3333333333333333.
    const SK: f64 = 0.333_333_333_333_333_3;
    assert!(
        (ri - SK).abs() < 1e-12,
        "rand_score([0,0,1,1],[0,0,0,0]): sklearn={SK}, ferrolearn={ri}"
    );
}

/// RED — `calinski_harabasz_score` returns `+inf` instead of `1.0` when the
/// within-cluster dispersion is zero.
///
/// ferrolearn (clustering.rs `pub fn calinski_harabasz_score`, the
/// `if w_ss == F::zero() { return Ok(F::infinity()) }` branch at clustering.rs:1110-1113)
/// returns `Ok(inf)` for coincident-per-cluster points. sklearn returns **1.0**
/// (_unsupervised.py:387-389: `if intra_disp == 0.0: return 1.0`).
///
/// Oracle (sklearn 1.5.2, run from /tmp):
///   python3 -c "import numpy as np; \
///     from sklearn.metrics import calinski_harabasz_score as c; \
///     print(repr(c(np.array([[0.],[0.],[10.],[10.]]), np.array([0,0,1,1]))))"
///   # 1.0
///
/// Tracking: #798
#[test]
fn red_calinski_harabasz_intra_disp_zero() {
    let x: Array2<f64> = array![[0.0], [0.0], [10.0], [10.0]];
    let l = labels(&[0, 0, 1, 1]);
    let ch = calinski_harabasz_score(&x, &l)
        .expect("calinski_harabasz_score should not error on this input");
    // sklearn 1.5.2 live oracle: intra_disp == 0 -> 1.0.
    const SK: f64 = 1.0;
    assert!(
        (ch - SK).abs() < 1e-12,
        "calinski_harabasz_score(coincident clusters): sklearn={SK}, ferrolearn={ch}"
    );
}

/// RED — the homogeneity family errors on empty input; sklearn returns the
/// perfect-match convention.
///
/// ferrolearn `homogeneity_score`/`completeness_score`/`v_measure_score`/
/// `homogeneity_completeness_v_measure` all return
/// `Err(InsufficientSamples)` on empty `labels_true`/`labels_pred` (the
/// `if n == 0 { return Err(...) }` guards, e.g. clustering.rs:1190-1196). sklearn
/// short-circuits empty input to `(1.0, 1.0, 1.0)`
/// (_supervised.py:531-532: `if len(labels_true) == 0: return 1.0, 1.0, 1.0`),
/// and the thin projections return 1.0 (_supervised.py:629, :705, :809).
///
/// Oracle (sklearn 1.5.2, run from /tmp):
///   python3 -c "from sklearn.metrics import homogeneity_score as h, \
///     completeness_score as c, v_measure_score as v, \
///     homogeneity_completeness_v_measure as f; \
///     print(h([],[]), c([],[]), v([],[]), f([],[]))"
///   # 1.0 1.0 1.0 (1.0, 1.0, 1.0)
///
/// Tracking: #799
#[test]
fn red_homogeneity_family_empty() {
    let empty = Array1::<isize>::from(Vec::<isize>::new());

    let h = homogeneity_score(&empty, &empty)
        .expect("homogeneity_score should return Ok(1.0) on empty input (sklearn convention)");
    let c = completeness_score(&empty, &empty)
        .expect("completeness_score should return Ok(1.0) on empty input");
    let v = v_measure_score(&empty, &empty)
        .expect("v_measure_score should return Ok(1.0) on empty input");
    let (hh, hc, hv) = homogeneity_completeness_v_measure(&empty, &empty, 1.0)
        .expect("homogeneity_completeness_v_measure should return Ok((1,1,1)) on empty input");

    // sklearn 1.5.2 live oracle: 1.0 / 1.0 / 1.0 / (1.0, 1.0, 1.0).
    assert_eq!(h, 1.0, "homogeneity_score([],[]): sklearn 1.0");
    assert_eq!(c, 1.0, "completeness_score([],[]): sklearn 1.0");
    assert_eq!(v, 1.0, "v_measure_score([],[]): sklearn 1.0");
    assert_eq!(
        (hh, hc, hv),
        (1.0, 1.0, 1.0),
        "homogeneity_completeness_v_measure([],[]): sklearn (1.0,1.0,1.0)"
    );
}

// ===========================================================================
// GREEN guards — oracle-grounded SHIPPED value contracts (must pass now)
// ===========================================================================

/// Guard: `silhouette_score` / `davies_bouldin_score` / `calinski_harabasz_score`
/// match sklearn on a clean two-cluster fixture (no noise, euclidean).
///
/// Oracle (sklearn 1.5.2, run from /tmp):
///   python3 -c "import numpy as np; from sklearn.metrics import \
///     silhouette_score as s, davies_bouldin_score as d, \
///     calinski_harabasz_score as c; \
///     X=np.array([[0.,0.],[.1,.1],[10.,10.],[10.1,10.1]]); l=np.array([0,0,1,1]); \
///     print(repr(s(X,l)), repr(d(X,l)), repr(c(X,l)))"
///   # 0.9899997499937521 0.009999999999997726 20000.000000000076
#[test]
fn green_feature_space_metrics() {
    let x: Array2<f64> = array![[0.0, 0.0], [0.1, 0.1], [10.0, 10.0], [10.1, 10.1]];
    let l = labels(&[0, 0, 1, 1]);
    let s = silhouette_score(&x, &l).unwrap();
    let d = davies_bouldin_score(&x, &l).unwrap();
    let c = calinski_harabasz_score(&x, &l).unwrap();
    // sklearn 1.5.2 live oracle.
    const SK_S: f64 = 0.989_999_749_993_752_1;
    const SK_D: f64 = 0.009_999_999_999_997_726;
    const SK_C: f64 = 20_000.000_000_000_076;
    assert!(
        (s - SK_S).abs() < 1e-9,
        "silhouette_score: sklearn={SK_S}, ferrolearn={s}"
    );
    assert!(
        (d - SK_D).abs() < 1e-9,
        "davies_bouldin_score: sklearn={SK_D}, ferrolearn={d}"
    );
    assert!(
        (c - SK_C).abs() < 1e-3,
        "calinski_harabasz_score: sklearn={SK_C}, ferrolearn={c}"
    );
}

/// Guard: `adjusted_rand_score` reproduces the -0.5 discordant floor.
///
/// Oracle (sklearn 1.5.2, run from /tmp):
///   python3 -c "from sklearn.metrics import adjusted_rand_score as r; \
///     print(repr(r([0,0,1,1],[0,1,0,1])))"
///   # -0.5
#[test]
fn green_adjusted_rand_discordant() {
    let ari = adjusted_rand_score(&labels(&[0, 0, 1, 1]), &labels(&[0, 1, 0, 1])).unwrap();
    // sklearn 1.5.2 live oracle: -0.5.
    const SK: f64 = -0.5;
    assert!(
        (ari - SK).abs() < 1e-12,
        "adjusted_rand_score([0,0,1,1],[0,1,0,1]): sklearn={SK}, ferrolearn={ari}"
    );
}

/// Guard: `adjusted_mutual_info` matches sklearn's `arithmetic`-default AMI
/// (exact EMI) on a mixed labeling.
///
/// Oracle (sklearn 1.5.2, run from /tmp):
///   python3 -c "from sklearn.metrics import adjusted_mutual_info_score as a; \
///     print(repr(a([0,0,1,1,2,2],[0,0,0,1,1,1])))"
///   # 0.2987924581708903
#[test]
fn green_adjusted_mutual_info_arithmetic_default() {
    let ami =
        adjusted_mutual_info(&labels(&[0, 0, 1, 1, 2, 2]), &labels(&[0, 0, 0, 1, 1, 1])).unwrap();
    // sklearn 1.5.2 live oracle (average_method='arithmetic' default, exact EMI).
    const SK: f64 = 0.298_792_458_170_890_3;
    assert!(
        (ami - SK).abs() < 1e-9,
        "adjusted_mutual_info(mixed): sklearn={SK}, ferrolearn={ami}"
    );
}

/// Guard: `normalized_mutual_info_score` with `NmiMethod::Arithmetic` matches
/// sklearn's `arithmetic`-DEFAULT NMI (ferrolearn matches only because the
/// caller supplies `Arithmetic`; sklearn picks it by default).
///
/// Oracle (sklearn 1.5.2, run from /tmp):
///   python3 -c "from sklearn.metrics import normalized_mutual_info_score as n; \
///     print(repr(n([0,0,1,1,2,2],[0,0,0,1,1,1])))"
///   # 0.5158037429793889
#[test]
fn green_normalized_mutual_info_arithmetic() {
    let nmi = normalized_mutual_info_score(
        &labels(&[0, 0, 1, 1, 2, 2]),
        &labels(&[0, 0, 0, 1, 1, 1]),
        NmiMethod::Arithmetic,
    )
    .unwrap();
    // sklearn 1.5.2 live oracle (average_method='arithmetic' default).
    const SK: f64 = 0.515_803_742_979_388_9;
    assert!(
        (nmi - SK).abs() < 1e-9,
        "normalized_mutual_info_score(mixed, Arithmetic): sklearn={SK}, ferrolearn={nmi}"
    );
}

/// Guard: `v_measure_score` (beta=1) matches sklearn's default v-measure, which
/// is identical to `normalized_mutual_info_score(average_method='arithmetic')`.
///
/// Oracle (sklearn 1.5.2, run from /tmp):
///   python3 -c "from sklearn.metrics import v_measure_score as v; \
///     print(repr(v([0,0,1,1,2,2],[0,0,0,1,1,1])))"
///   # 0.5158037429793889
#[test]
fn green_v_measure_default_beta() {
    let v = v_measure_score(&labels(&[0, 0, 1, 1, 2, 2]), &labels(&[0, 0, 0, 1, 1, 1])).unwrap();
    // sklearn 1.5.2 live oracle (beta=1.0 default == NMI arithmetic).
    const SK: f64 = 0.515_803_742_979_388_9;
    assert!(
        (v - SK).abs() < 1e-9,
        "v_measure_score(mixed, beta=1): sklearn={SK}, ferrolearn={v}"
    );
}

/// Guard: `homogeneity_completeness_v_measure` with `beta=2.0` matches sklearn's
/// beta-weighted v-measure (the beta form is exposed on this function in both
/// sklearn and ferrolearn).
///
/// Oracle (sklearn 1.5.2, run from /tmp):
///   python3 -c "from sklearn.metrics import \
///     homogeneity_completeness_v_measure as f; \
///     print(repr(f([0,0,1,1,2,2],[0,0,0,1,1,1], beta=2.0)[2]))"
///   # 0.5578858913022597
#[test]
fn green_hcv_beta_two() {
    let (_h, _c, v) = homogeneity_completeness_v_measure(
        &labels(&[0, 0, 1, 1, 2, 2]),
        &labels(&[0, 0, 0, 1, 1, 1]),
        2.0,
    )
    .unwrap();
    // sklearn 1.5.2 live oracle: v at beta=2.0.
    const SK: f64 = 0.557_885_891_302_259_7;
    assert!(
        (v - SK).abs() < 1e-9,
        "homogeneity_completeness_v_measure(mixed, beta=2): sklearn v={SK}, ferrolearn v={v}"
    );
}

/// Guard: `fowlkes_mallows_score` matches sklearn on a mixed labeling.
///
/// Oracle (sklearn 1.5.2, run from /tmp):
///   python3 -c "from sklearn.metrics import fowlkes_mallows_score as f; \
///     print(repr(f([0,0,1,1,2,2],[0,0,0,1,1,1])))"
///   # 0.4714045207910317
#[test]
fn green_fowlkes_mallows() {
    let fmi =
        fowlkes_mallows_score(&labels(&[0, 0, 1, 1, 2, 2]), &labels(&[0, 0, 0, 1, 1, 1])).unwrap();
    // sklearn 1.5.2 live oracle.
    const SK: f64 = 0.471_404_520_791_031_7;
    assert!(
        (fmi - SK).abs() < 1e-12,
        "fowlkes_mallows_score(mixed): sklearn={SK}, ferrolearn={fmi}"
    );
}

/// Guard: `mutual_info_score` matches sklearn (nats) on a mixed labeling.
///
/// Oracle (sklearn 1.5.2, run from /tmp):
///   python3 -c "from sklearn.metrics import mutual_info_score as m; \
///     print(repr(m([0,1,1,0,1,0],[0,1,0,0,1,1])))"
///   # 0.0566330122651324
#[test]
fn green_mutual_info_score() {
    let mi = mutual_info_score(&labels(&[0, 1, 1, 0, 1, 0]), &labels(&[0, 1, 0, 0, 1, 1])).unwrap();
    // sklearn 1.5.2 live oracle.
    const SK: f64 = 0.056_633_012_265_132_4;
    assert!(
        (mi - SK).abs() < 1e-12,
        "mutual_info_score(mixed): sklearn={SK}, ferrolearn={mi}"
    );
}

/// Guard: `contingency_matrix` matches sklearn dense output and ordering.
///
/// Oracle (sklearn 1.5.2, run from /tmp):
///   python3 -c "from sklearn.metrics.cluster import contingency_matrix as cm; \
///     print(cm([0,0,1,1,2,2],[1,0,2,1,0,2]).tolist())"
///   # [[1, 1, 0], [0, 1, 1], [1, 0, 1]]
#[test]
fn green_contingency_matrix() {
    let cm =
        contingency_matrix(&labels(&[0, 0, 1, 1, 2, 2]), &labels(&[1, 0, 2, 1, 0, 2])).unwrap();
    // sklearn 1.5.2 live oracle.
    let sk: Array2<u64> = array![[1, 1, 0], [0, 1, 1], [1, 0, 1]];
    assert_eq!(
        cm, sk,
        "contingency_matrix: sklearn={sk:?}, ferrolearn={cm:?}"
    );
}

/// Guard: `pair_confusion_matrix` matches sklearn's 2x2 int64 identities.
///
/// Oracle (sklearn 1.5.2, run from /tmp):
///   python3 -c "from sklearn.metrics.cluster import pair_confusion_matrix as p; \
///     print(p([0,0,1,2],[0,0,1,1]).tolist())"
///   # [[8, 2], [0, 2]]
#[test]
fn green_pair_confusion_matrix() {
    let pc = pair_confusion_matrix(&labels(&[0, 0, 1, 2]), &labels(&[0, 0, 1, 1])).unwrap();
    // sklearn 1.5.2 live oracle.
    let sk: Array2<u64> = array![[8, 2], [0, 2]];
    assert_eq!(
        pc, sk,
        "pair_confusion_matrix: sklearn={sk:?}, ferrolearn={pc:?}"
    );
}

// ===========================================================================
// RE-AUDIT pin (#800) — rand_score n<2 guard pre-empts sklearn's degenerate
// branch. The #797 rewrite cited sklearn's `numerator == denominator or
// denominator == 0 -> 1.0` branch (_supervised.py:337-341) but left the
// pre-existing `if n < 2 { return Err(InsufficientSamples) }` guard
// (clustering.rs `pub fn rand_score`) in front of it, so a single sample
// never reaches that branch.
// ===========================================================================

/// RED — `rand_score` errors on a single-sample labeling; sklearn returns 1.0.
///
/// sklearn `rand_score([0],[0])` builds `pair_confusion_matrix([0],[0]) ==
/// [[0,0],[0,0]]`, so `numerator == denominator == 0` and the
/// `if numerator == denominator or denominator == 0: return 1.0` branch
/// (_supervised.py:337-341) fires → 1.0. ferrolearn rejects n<2 up front with
/// `Err(InsufficientSamples)` (clustering.rs `pub fn rand_score`, the
/// `if n < 2 { return Err(...) }` guard), so the value is never computed.
///
/// Oracle (sklearn 1.5.2, run from /tmp):
///   python3 -c "from sklearn.metrics import rand_score; \
///     print(repr(rand_score([0],[0])))"
///   # 1.0
///   python3 -c "from sklearn.metrics.cluster import pair_confusion_matrix; \
///     print(pair_confusion_matrix([0],[0]).tolist())"
///   # [[0, 0], [0, 0]]
///
/// Tracking: #800
#[test]
fn red_rand_score_single_sample() {
    let lt = labels(&[0]);
    let lp = labels(&[0]);
    let ri = rand_score(&lt, &lp)
        .expect("rand_score([0],[0]) should return Ok(1.0) per sklearn, not Err");
    // sklearn 1.5.2 live oracle: 1.0 (denominator == 0 branch).
    const SK: f64 = 1.0;
    assert!(
        (ri - SK).abs() < 1e-12,
        "rand_score([0],[0]): sklearn={SK}, ferrolearn={ri}"
    );
}
