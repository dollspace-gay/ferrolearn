//! Divergence pins for `AgglomerativeClustering` dendrogram / label numbering
//! (`ferrolearn-cluster/src/agglomerative.rs`) vs scikit-learn 1.5.2 / scipy
//! `sklearn/cluster/_agglomerative.py`
//! (`class AgglomerativeClustering(ClusterMixin, BaseEstimator)`, `:781`).
//!
//! Pins the two NOT-STARTED REQs sharing root cause #938 (truncated reused-slot
//! tree + ascending-slot relabel vs sklearn's full scipy dendrogram + `_hc_cut`):
//!
//!   * REQ-6 — `children_` FULL-DENDROGRAM format. sklearn delegates the default
//!     unstructured ward case to scipy: `ward_tree` returns
//!     `scipy.cluster.hierarchy.ward(X)[:, :2]` (`_agglomerative.py:314`); the
//!     result is shape `(n_samples-1, 2)` with internal-node IDs `>= n_samples`
//!     (leaves `0..n-1`; the i-th merge in scipy output-row order creates node
//!     `n+i`). ferrolearn's `children_` is instead length `n_samples-n_clusters`
//!     of reused merged-into-slot pairs (`fn agglomerate`: `children.push((ci,cj))`),
//!     a TRUNCATED tree with neither the full length nor the `>= n_samples` IDs.
//!
//!   * REQ-7 — `labels_` ABSOLUTE numbering via `_hc_cut`. sklearn numbers
//!     `labels_` with `_hc_cut(n_clusters, children_, n_leaves)`
//!     (`_agglomerative.py:731-778`), a negated-max-heap split of the top of the
//!     dendrogram. ferrolearn relabels by ascending surviving-slot order (a
//!     `HashMap` in `fn agglomerate`), giving the SAME partition (REQ-1, which
//!     ships) but PERMUTED integer labels.
//!
//! Expected values come from the LIVE scipy/sklearn 1.5.2 oracle (R-CHAR-3,
//! R-DEV-3 exact-output) — NEVER literal-copied from the ferrolearn side. The
//! exact `python3 -c` command + its output are recorded in each doc comment.
//!
//! Both are marked `#[ignore]` (tracking #938) so the orchestrator suite stays
//! green; run with `--ignored` to confirm they FAIL against current code.

use ferrolearn_cluster::{AgglomerativeClustering, Linkage};
use ferrolearn_core::Fit;
use ndarray::Array2;

/// Small generic 6-point fixture, no degenerate ties. Three loose pairs/triples
/// arranged so the scipy merge order and the sklearn `_hc_cut` label numbering
/// are both well-defined and (for `labels_`) DIFFER from ferrolearn's
/// ascending-slot numbering.
fn fixture() -> Array2<f64> {
    Array2::from_shape_vec(
        (6, 2),
        vec![0.0, 0.0, 0.2, 0.1, 0.9, 1.1, 3.0, 3.2, 3.3, 3.0, 6.0, 0.2],
    )
    .unwrap()
}

/// REQ-6: `children_` must be the FULL scipy dendrogram — shape
/// `(n_samples-1, 2)` with internal-node IDs `>= n_samples` — equal to
/// `scipy.cluster.hierarchy.ward(X)[:, :2]`.
///
/// Live scipy 1.17.1 / sklearn 1.5.2 oracle (system python3):
/// ```text
/// python3 -c "import numpy as np; from scipy.cluster.hierarchy import ward; \
///   X=np.array([[0.,0.],[0.2,0.1],[0.9,1.1],[3.,3.2],[3.3,3.],[6.,0.2]]); \
///   print(ward(X)[:, :2].astype(int))"
/// # [[0 1]
/// #  [3 4]
/// #  [2 6]
/// #  [5 7]
/// #  [8 9]]
/// ```
/// n_samples = 6, so the full dendrogram has n_samples-1 = 5 rows, and IDs
/// 6,7,8,9 (>= n_samples) name the internal merge nodes.
///
/// ferrolearn-actual (current code, truncated reused-slot tree, n_clusters=2):
/// `children_` = `[(0,1),(3,4),(0,2),(3,5)]` — only 4 rows, IDs all < 6.
#[test]
fn divergence_children_full_dendrogram_format() {
    // Live scipy ward(X)[:, :2].astype(int) (see doc comment) — the contract.
    const SK_CHILDREN: [[usize; 2]; 5] = [[0, 1], [3, 4], [2, 6], [5, 7], [8, 9]];
    let n_samples = 6usize;

    let x = fixture();
    let fitted = AgglomerativeClustering::<f64>::new(2)
        .with_linkage(Linkage::Ward)
        .fit(&x, &())
        .unwrap();

    let children = fitted.children();

    // (a) full-dendrogram LENGTH: n_samples - 1, independent of n_clusters.
    assert_eq!(
        children.len(),
        n_samples - 1,
        "REQ-6: children_ must have n_samples-1 = {} rows (full scipy dendrogram); \
         ferrolearn returned {} (truncated reused-slot tree)",
        n_samples - 1,
        children.len()
    );

    // (b) exact row-by-row match to scipy ward(X)[:, :2] (IDs incl. >= n_samples).
    for (i, &[a, b]) in SK_CHILDREN.iter().enumerate() {
        assert_eq!(
            (children[i].0, children[i].1),
            (a, b),
            "REQ-6: children_[{i}] must equal scipy row ({a},{b}); ferrolearn returned {:?}",
            children[i]
        );
    }
}

/// REQ-7: `labels_` must use sklearn's ABSOLUTE `_hc_cut` numbering, i.e. equal
/// `AgglomerativeClustering(n_clusters=2, linkage='ward').fit(X).labels_` as
/// INTEGERS (not merely up-to-permutation). The PARTITION already matches
/// (REQ-1, SHIPPED); this pins ONLY the integer label values.
///
/// Live sklearn 1.5.2 oracle (system python3):
/// ```text
/// python3 -c "import numpy as np; from sklearn.cluster import AgglomerativeClustering; \
///   X=np.array([[0.,0.],[0.2,0.1],[0.9,1.1],[3.,3.2],[3.3,3.],[6.,0.2]]); \
///   print(AgglomerativeClustering(n_clusters=2,linkage='ward').fit(X).labels_.tolist())"
/// # [1, 1, 1, 0, 0, 0]
/// ```
/// The `_hc_cut` heap split numbers the {0,1,2} group as label 1 and the
/// {3,4,5} group as label 0.
///
/// ferrolearn-actual (current code, ascending-slot relabel): `[0, 0, 0, 1, 1, 1]`
/// — SAME partition, but the two integer labels are SWAPPED.
#[test]
fn divergence_labels_absolute_hc_cut_numbering() {
    // Live sklearn labels_ (see doc comment) — the absolute-numbering contract.
    const SK_LABELS: [usize; 6] = [1, 1, 1, 0, 0, 0];

    let x = fixture();
    let fitted = AgglomerativeClustering::<f64>::new(2)
        .with_linkage(Linkage::Ward)
        .fit(&x, &())
        .unwrap();

    let labels: Vec<usize> = fitted.labels().to_vec();
    assert_eq!(
        labels,
        SK_LABELS.to_vec(),
        "REQ-7: labels_ must EXACT-equal sklearn _hc_cut numbering {SK_LABELS:?} \
         (partition matches but integers are permuted); ferrolearn returned {labels:?}"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// REQ-6 / REQ-7 bit-exact parity across all four linkages, live scipy/sklearn.
//
// All expected values are the LIVE scipy 1.17.1 / sklearn 1.5.2 oracle, computed
// in-test from `python3 -c` (R-CHAR-3 — NEVER copied from ferrolearn). The exact
// command + its raw output is recorded in each helper's doc comment.
// ─────────────────────────────────────────────────────────────────────────────

/// Larger 10-point generic fixture (no degenerate ties).
fn fixture10() -> Array2<f64> {
    Array2::from_shape_vec(
        (10, 2),
        vec![
            0.0, 0.0, 1.0, 0.2, 0.3, 1.0, 5.0, 5.0, 5.4, 4.8, 4.7, 5.5, 10.0, 0.0, 9.6, 0.5, 2.0,
            8.0, 2.5, 7.6,
        ],
    )
    .unwrap()
}

fn children_of(x: &Array2<f64>, linkage: Linkage) -> Vec<(usize, usize)> {
    AgglomerativeClustering::<f64>::new(2)
        .with_linkage(linkage)
        .fit(x, &())
        .unwrap()
        .children()
        .to_vec()
}

fn labels_of(x: &Array2<f64>, k: usize, linkage: Linkage) -> Vec<usize> {
    AgglomerativeClustering::<f64>::new(k)
        .with_linkage(linkage)
        .fit(x, &())
        .unwrap()
        .labels()
        .to_vec()
}

/// REQ-6: `children_` EXACT-equals
/// `scipy.cluster.hierarchy.linkage(X6, method, 'euclidean')[:, :2]` for the
/// nn-chain linkages (ward/complete/average), which sklearn routes verbatim
/// through `hierarchy.linkage` (`_agglomerative.py:314`/`:586`).
///
/// Live scipy 1.17.1 oracle:
/// ```text
/// python3 -c "import numpy as np; from scipy.cluster.hierarchy import linkage; \
///   X=np.array([[0.,0.],[0.2,0.1],[0.9,1.1],[3.,3.2],[3.3,3.],[6.,0.2]]); \
///   [print(m, linkage(X,method=m,metric='euclidean')[:, :2].astype(int).tolist()) \
///    for m in ['ward','complete','average']]"
/// # ward     [[0,1],[3,4],[2,6],[5,7],[8,9]]
/// # complete [[0,1],[3,4],[2,6],[5,7],[8,9]]
/// # average  [[0,1],[3,4],[2,6],[7,8],[5,9]]
/// ```
#[test]
fn children_exact_scipy_6pt_nn_chain_linkages() {
    let x = fixture();
    let expected: [(Linkage, Vec<(usize, usize)>); 3] = [
        (Linkage::Ward, vec![(0, 1), (3, 4), (2, 6), (5, 7), (8, 9)]),
        (
            Linkage::Complete,
            vec![(0, 1), (3, 4), (2, 6), (5, 7), (8, 9)],
        ),
        (
            Linkage::Average,
            vec![(0, 1), (3, 4), (2, 6), (7, 8), (5, 9)],
        ),
    ];
    for (linkage, exp) in expected {
        assert_eq!(
            children_of(&x, linkage),
            exp,
            "REQ-6 children_ mismatch (6pt) for {linkage:?} vs scipy linkage"
        );
    }
}

/// REQ-6 (single linkage, R-DEV-7): for `single`, sklearn does NOT delegate to
/// `scipy.cluster.hierarchy.linkage(method='single')`; it uses its own
/// `mst_linkage_core` + `_single_linkage_label` path
/// (`_agglomerative.py:567-584`, `_hierarchical_fast.pyx`), whose
/// `_single_linkage_label` emits each merge as `(left_root, right_root)` in MST
/// order WITHOUT scipy's min/max column swap. The CONTRACT for the fitted
/// `children_` attribute is therefore sklearn's value (which ferrolearn
/// reproduces bit-exact), NOT `scipy.linkage`'s column ordering — the two differ
/// in pair order for single linkage.
///
/// Live sklearn 1.5.2 oracle (the actual `children_` attribute):
/// ```text
/// python3 -c "import numpy as np; from sklearn.cluster import AgglomerativeClustering as AC; \
///   X=np.array([[0.,0.],[0.2,0.1],[0.9,1.1],[3.,3.2],[3.3,3.],[6.,0.2]]); \
///   print(AC(n_clusters=2,linkage='single').fit(X).children_.tolist())"
/// # [[0,1],[3,4],[6,2],[8,7],[9,5]]
/// ```
#[test]
fn children_exact_sklearn_single_6pt() {
    let x = fixture();
    let expected = vec![(0, 1), (3, 4), (6, 2), (8, 7), (9, 5)];
    assert_eq!(
        children_of(&x, Linkage::Single),
        expected,
        "REQ-6 children_ mismatch (6pt single) vs sklearn AgglomerativeClustering.children_"
    );
}

/// REQ-6: `children_` EXACT-equals scipy linkage `[:, :2]` for the nn-chain
/// linkages on the 10-point fixture; single is checked against sklearn's own
/// `children_` (see [`children_exact_sklearn_single_10pt`], the R-DEV-7 path).
///
/// Live scipy 1.17.1 oracle:
/// ```text
/// python3 -c "import numpy as np; from scipy.cluster.hierarchy import linkage; \
///   X=np.array([[0.,0.],[1.,0.2],[0.3,1.],[5.,5.],[5.4,4.8],[4.7,5.5],[10.,0.],[9.6,0.5],[2.,8.],[2.5,7.6]]); \
///   [print(m, linkage(X,method=m,metric='euclidean')[:, :2].astype(int).tolist()) \
///    for m in ['ward','complete','average']]"
/// # ward     [[3,4],[8,9],[6,7],[5,10],[0,1],[2,14],[11,13],[15,16],[12,17]]
/// # complete [[3,4],[8,9],[6,7],[5,10],[0,1],[2,14],[11,13],[15,16],[12,17]]
/// # average  [[3,4],[8,9],[6,7],[5,10],[0,1],[2,14],[11,13],[15,16],[12,17]]
/// ```
#[test]
fn children_exact_scipy_10pt_nn_chain_linkages() {
    let x = fixture10();
    let nn_chain = vec![
        (3, 4),
        (8, 9),
        (6, 7),
        (5, 10),
        (0, 1),
        (2, 14),
        (11, 13),
        (15, 16),
        (12, 17),
    ];
    let expected: [(Linkage, Vec<(usize, usize)>); 3] = [
        (Linkage::Ward, nn_chain.clone()),
        (Linkage::Complete, nn_chain.clone()),
        (Linkage::Average, nn_chain.clone()),
    ];
    for (linkage, exp) in expected {
        assert_eq!(
            children_of(&x, linkage),
            exp,
            "REQ-6 children_ mismatch (10pt) for {linkage:?} vs scipy linkage"
        );
    }
}

/// REQ-6 (single linkage, R-DEV-7): `children_` EXACT-equals sklearn's own
/// `AgglomerativeClustering(linkage='single').children_` (the `mst_linkage_core`
/// + `_single_linkage_label` path) on the 10-point fixture.
///
/// Live sklearn 1.5.2 oracle:
/// ```text
/// python3 -c "import numpy as np; from sklearn.cluster import AgglomerativeClustering as AC; \
///   X=np.array([[0.,0.],[1.,0.2],[0.3,1.],[5.,5.],[5.4,4.8],[4.7,5.5],[10.,0.],[9.6,0.5],[2.,8.],[2.5,7.6]]); \
///   print(AC(n_clusters=2,linkage='single').fit(X).children_.tolist())"
/// # [[3,4],[10,5],[9,8],[7,6],[0,1],[14,2],[11,12],[16,13],[15,17]]
/// ```
#[test]
fn children_exact_sklearn_single_10pt() {
    let x = fixture10();
    let expected = vec![
        (3, 4),
        (10, 5),
        (9, 8),
        (7, 6),
        (0, 1),
        (14, 2),
        (11, 12),
        (16, 13),
        (15, 17),
    ];
    assert_eq!(
        children_of(&x, Linkage::Single),
        expected,
        "REQ-6 children_ mismatch (10pt single) vs sklearn AgglomerativeClustering.children_"
    );
}

/// REQ-7: `labels_` EXACT-equals
/// `sklearn.cluster.AgglomerativeClustering(n_clusters=k, linkage=…).fit(X6).labels_`
/// for k in {2,3} and all four linkages.
///
/// Live sklearn 1.5.2 oracle:
/// ```text
/// python3 -c "import numpy as np; from sklearn.cluster import AgglomerativeClustering as AC; \
///   X=np.array([[0.,0.],[0.2,0.1],[0.9,1.1],[3.,3.2],[3.3,3.],[6.,0.2]]); \
///   [print(m,k,AC(n_clusters=k,linkage=m).fit(X).labels_.tolist()) \
///    for m in ['ward','complete','average','single'] for k in (2,3)]"
/// # ward     2 [1,1,1,0,0,0]   ward     3 [0,0,0,1,1,2]
/// # complete 2 [1,1,1,0,0,0]   complete 3 [0,0,0,1,1,2]
/// # average  2 [0,0,0,0,0,1]   average  3 [0,0,0,2,2,1]
/// # single   2 [0,0,0,0,0,1]   single   3 [0,0,0,2,2,1]
/// ```
#[test]
fn labels_exact_sklearn_6pt_all_linkages() {
    let x = fixture();
    let cases: [(Linkage, usize, Vec<usize>); 8] = [
        (Linkage::Ward, 2, vec![1, 1, 1, 0, 0, 0]),
        (Linkage::Ward, 3, vec![0, 0, 0, 1, 1, 2]),
        (Linkage::Complete, 2, vec![1, 1, 1, 0, 0, 0]),
        (Linkage::Complete, 3, vec![0, 0, 0, 1, 1, 2]),
        (Linkage::Average, 2, vec![0, 0, 0, 0, 0, 1]),
        (Linkage::Average, 3, vec![0, 0, 0, 2, 2, 1]),
        (Linkage::Single, 2, vec![0, 0, 0, 0, 0, 1]),
        (Linkage::Single, 3, vec![0, 0, 0, 2, 2, 1]),
    ];
    for (linkage, k, exp) in cases {
        assert_eq!(
            labels_of(&x, k, linkage),
            exp,
            "REQ-7 labels_ mismatch (6pt) for {linkage:?} k={k} vs sklearn _hc_cut"
        );
    }
}

/// REQ-7: `labels_` EXACT-equals sklearn `.fit(X10).labels_` for k in {2,3} and
/// all four linkages.
///
/// Live sklearn 1.5.2 oracle:
/// ```text
/// python3 -c "import numpy as np; from sklearn.cluster import AgglomerativeClustering as AC; \
///   X=np.array([[0.,0.],[1.,0.2],[0.3,1.],[5.,5.],[5.4,4.8],[4.7,5.5],[10.,0.],[9.6,0.5],[2.,8.],[2.5,7.6]]); \
///   [print(m,k,AC(n_clusters=k,linkage=m).fit(X).labels_.tolist()) \
///    for m in ['ward','complete','average','single'] for k in (2,3)]"
/// # ward     2 [0,0,0,0,0,0,1,1,0,0]   ward     3 [2,2,2,0,0,0,1,1,0,0]
/// # complete 2 [0,0,0,0,0,0,1,1,0,0]   complete 3 [2,2,2,0,0,0,1,1,0,0]
/// # average  2 [0,0,0,0,0,0,1,1,0,0]   average  3 [2,2,2,0,0,0,1,1,0,0]
/// # single   2 [1,1,1,0,0,0,0,0,0,0]   single   3 [1,1,1,0,0,0,2,2,0,0]
/// ```
#[test]
fn labels_exact_sklearn_10pt_all_linkages() {
    let x = fixture10();
    let cases: [(Linkage, usize, Vec<usize>); 8] = [
        (Linkage::Ward, 2, vec![0, 0, 0, 0, 0, 0, 1, 1, 0, 0]),
        (Linkage::Ward, 3, vec![2, 2, 2, 0, 0, 0, 1, 1, 0, 0]),
        (Linkage::Complete, 2, vec![0, 0, 0, 0, 0, 0, 1, 1, 0, 0]),
        (Linkage::Complete, 3, vec![2, 2, 2, 0, 0, 0, 1, 1, 0, 0]),
        (Linkage::Average, 2, vec![0, 0, 0, 0, 0, 0, 1, 1, 0, 0]),
        (Linkage::Average, 3, vec![2, 2, 2, 0, 0, 0, 1, 1, 0, 0]),
        (Linkage::Single, 2, vec![1, 1, 1, 0, 0, 0, 0, 0, 0, 0]),
        (Linkage::Single, 3, vec![1, 1, 1, 0, 0, 0, 2, 2, 0, 0]),
    ];
    for (linkage, k, exp) in cases {
        assert_eq!(
            labels_of(&x, k, linkage),
            exp,
            "REQ-7 labels_ mismatch (10pt) for {linkage:?} k={k} vs sklearn _hc_cut"
        );
    }
}
