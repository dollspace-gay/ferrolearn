//! Adversarial divergence pins for `ferrolearn-metrics/src/clustering.rs` vs
//! scikit-learn 1.5.2 (audit #2300).
//!
//! Every expected value is the LIVE sklearn 1.5.2 oracle (the exact `python3 -c`
//! invocation is quoted in each test, R-CHAR-3) — NEVER copied from the
//! ferrolearn side. The installed `sklearn` package and the clone at
//! `/home/doll/scikit-learn` are both at tag 1.5.2, so the `file:line`
//! citations and the oracle values are the same version by construction.
//!
//! These pin three genuine VALUE divergences in SHIPPED functions:
//!   - silhouette singleton-cluster sample/score (ferro 1.0 vs sklearn 0.0)
//!   - NMI MI==0 single-side limit (ferro 1.0 vs sklearn 0.0)
//!   - ARI single-sample mismatched labels (ferro 0.0 vs sklearn 1.0)
//!
//! Each is `#[ignore]`d with its tracking issue: the generator must fix
//! `clustering.rs`, not this test.

use ferrolearn_metrics::clustering::{
    NmiMethod, adjusted_rand_score, normalized_mutual_info_score, silhouette_samples,
    silhouette_score,
};
use ndarray::{Array1, Array2, array};

fn labels(v: &[isize]) -> Array1<isize> {
    Array1::from(v.to_vec())
}

// ===========================================================================
// 1. silhouette singleton cluster — ferrolearn returns 1.0, sklearn 0.0.
// ===========================================================================

/// Divergence: `silhouette_samples` gives a size-1 cluster a coefficient of
/// 1.0; sklearn assigns 0.0.
///
/// For a sample that is the sole member of its cluster, ferrolearn computes
/// `a_i = 0` (clustering.rs `silhouette_samples`, the
/// `if ci_members.len() <= 1 { F::zero() }` branch) and then
/// `s_i = (b_i - 0) / max(0, b_i) = 1.0`. scikit-learn instead produces NaN for
/// size-1 clusters (intra divides by `label_freqs - 1 == 0`) and maps it to 0.0
/// via `np.nan_to_num` — `_unsupervised.py:317-318`:
///   `# nan values are for clusters of size 1, and should be 0`
///   `return np.nan_to_num(sil_samples)`
///
/// Oracle (sklearn 1.5.2, run from /tmp):
///   python3 -c "import numpy as np; from sklearn.metrics import silhouette_samples; \
///     X=np.array([[0.,0.],[1.,0.],[0.,1.],[10.,10.],[11.,10.],[20.,0.]]); \
///     print(repr(float(silhouette_samples(X, np.array([0,0,0,1,1,2]))[5])))"
///   # 0.0
///
/// ferrolearn currently returns 1.0 for sample index 5 (the singleton).
///
/// Tracking: #2301
#[test]
#[ignore = "divergence: silhouette singleton cluster sample = 1.0, sklearn = 0.0; tracking #2301"]
fn divergence_silhouette_samples_singleton() {
    let x: Array2<f64> = array![
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [10.0, 10.0],
        [11.0, 10.0],
        [20.0, 0.0]
    ];
    let lab = labels(&[0, 0, 0, 1, 1, 2]);
    let samples = silhouette_samples(&x, &lab).unwrap();
    // sklearn 1.5.2 live oracle: the singleton (index 5) is 0.0.
    const SK_SINGLETON: f64 = 0.0;
    assert!(
        (samples[5] - SK_SINGLETON).abs() < 1e-12,
        "silhouette_samples singleton: sklearn={SK_SINGLETON}, ferrolearn={}",
        samples[5]
    );
}

/// Divergence: `silhouette_score` mean is inflated by the size-1 cluster's
/// erroneous 1.0 contribution.
///
/// Because the singleton contributes 1.0 (see
/// `divergence_silhouette_samples_singleton`) instead of sklearn's 0.0, the mean
/// over all samples is larger than sklearn's. sklearn averages
/// `np.nan_to_num(sil_samples)` (`_unsupervised.py:318`, then `np.mean` in
/// `silhouette_score`).
///
/// Oracle (sklearn 1.5.2, run from /tmp):
///   python3 -c "import numpy as np; from sklearn.metrics import silhouette_score; \
///     X=np.array([[0.,0.],[1.,0.],[0.,1.],[10.,10.],[11.,10.],[20.,0.]]); \
///     print(repr(float(silhouette_score(X, np.array([0,0,0,1,1,2])))))"
///   # 0.7681491694704304
///
/// ferrolearn returns ~0.9348 (the singleton averaged in at 1.0).
///
/// Tracking: #2301
#[test]
#[ignore = "divergence: silhouette_score mean inflated by singleton=1.0 vs sklearn=0.0; tracking #2301"]
fn divergence_silhouette_score_singleton() {
    let x: Array2<f64> = array![
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [10.0, 10.0],
        [11.0, 10.0],
        [20.0, 0.0]
    ];
    let lab = labels(&[0, 0, 0, 1, 1, 2]);
    let score = silhouette_score(&x, &lab).unwrap();
    // sklearn 1.5.2 live oracle.
    const SK: f64 = 0.768_149_169_470_430_4;
    assert!(
        (score - SK).abs() < 1e-9,
        "silhouette_score(singleton fixture): sklearn={SK}, ferrolearn={score}"
    );
}

// ===========================================================================
// 2. NMI MI==0 single-side limit — ferrolearn returns 1.0, sklearn 0.0.
// ===========================================================================

/// Divergence: `normalized_mutual_info_score` returns 1.0 when one side has a
/// single label (entropy 0) and the other does not; sklearn returns 0.0.
///
/// ferrolearn short-circuits on a zero *normalizer*
/// (clustering.rs `normalized_mutual_info_score`, the
/// `if normalizer.abs() < f64::EPSILON { return Ok(1.0) }` branch): with
/// `Geometric` (or `Min`), `H(true)=0` makes the normalizer 0, so it returns
/// 1.0. scikit-learn only short-circuits to 1.0 when BOTH labelings are a
/// single cluster (`_supervised.py:1152-1156`); otherwise, having dealt with
/// that case, `_supervised.py:1163-1167`:
///   `# At this point mi = 0 can't be a perfect match ...`
///   `if mi == 0:`
///   `    return 0.0`
///
/// Oracle (sklearn 1.5.2, run from /tmp):
///   python3 -c "from sklearn.metrics.cluster import normalized_mutual_info_score as n; \
///     print(repr(float(n([0,0,0],[0,1,2],average_method='geometric'))), \
///           repr(float(n([0,0,0],[0,1,2],average_method='min'))))"
///   # 0.0 0.0
///
/// ferrolearn returns 1.0 for both `Geometric` and `Min`.
///
/// Tracking: #2302
#[test]
#[ignore = "divergence: NMI single-side mi==0 returns 1.0, sklearn 0.0; tracking #2302"]
fn divergence_nmi_single_side_geometric() {
    let lt = labels(&[0, 0, 0]);
    let lp = labels(&[0, 1, 2]);
    let nmi_geo = normalized_mutual_info_score(&lt, &lp, NmiMethod::Geometric).unwrap();
    let nmi_min = normalized_mutual_info_score(&lt, &lp, NmiMethod::Min).unwrap();
    // sklearn 1.5.2 live oracle: 0.0 for both.
    const SK: f64 = 0.0;
    assert!(
        (nmi_geo - SK).abs() < 1e-12,
        "NMI([0,0,0],[0,1,2],geometric): sklearn={SK}, ferrolearn={nmi_geo}"
    );
    assert!(
        (nmi_min - SK).abs() < 1e-12,
        "NMI([0,0,0],[0,1,2],min): sklearn={SK}, ferrolearn={nmi_min}"
    );
}

// ===========================================================================
// 3. ARI single-sample mismatched labels — ferrolearn 0.0, sklearn 1.0.
// ===========================================================================

/// Divergence: `adjusted_rand_score` of one sample with differently-named
/// labels returns 0.0; sklearn returns 1.0.
///
/// ferrolearn special-cases `C(n,2)==0` on raw label equality
/// (clustering.rs `adjusted_rand_score`, the `comb_n == 0` branch:
/// `Ok(if labels_true[0] == labels_pred[0] { 1.0 } else { 0.0 })`), so two
/// different label *names* on a single sample yield 0.0. scikit-learn treats
/// each labeling as a single cluster (no pairs ⇒ `fn == 0 and fp == 0`) and
/// returns 1.0 regardless of the label names — `_supervised.py:448-450`:
///   `# Special cases: empty data or full agreement`
///   `if fn == 0 and fp == 0:`
///   `    return 1.0`
///
/// Oracle (sklearn 1.5.2, run from /tmp):
///   python3 -c "from sklearn.metrics import adjusted_rand_score as a; \
///     print(repr(float(a([0],[5]))))"
///   # 1.0
///
/// ferrolearn returns 0.0.
///
/// Tracking: #2303
#[test]
#[ignore = "divergence: ARI single-sample mismatched labels = 0.0, sklearn = 1.0; tracking #2303"]
fn divergence_ari_single_sample_mismatched() {
    let ari = adjusted_rand_score(&labels(&[0]), &labels(&[5])).unwrap();
    // sklearn 1.5.2 live oracle: 1.0 (no pairs ⇒ fn==fp==0 ⇒ full agreement).
    const SK: f64 = 1.0;
    assert!(
        (ari - SK).abs() < 1e-12,
        "adjusted_rand_score([0],[5]): sklearn={SK}, ferrolearn={ari}"
    );
}
