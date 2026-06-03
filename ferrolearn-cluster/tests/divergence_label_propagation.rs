//! Divergence + green-guard tests for `LabelPropagation` vs scikit-learn 1.5.2
//! (`sklearn/semi_supervised/_label_propagation.py`,
//! `class LabelPropagation(BaseLabelPropagation)` :338-483).
//!
//! Expected values are computed by the LIVE sklearn 1.5.2 oracle (R-CHAR-3),
//! never literal-copied from ferrolearn. The oracle invocations and outputs are
//! quoted in each test's doc comment.
//!
//! Two groups:
//!   * GREEN-GUARD (PASS today): the SHIPPED REQ-1 contiguous-label transduction
//!     PARTITION on FRESH separable fixtures (not used in-tree).
//!   * PIN (FAIL until #999 fixed): REQ-4 `classes_`/`n_classes` mapping on a
//!     non-contiguous `{0,2}` label set.

use ferrolearn_cluster::LabelPropagation;
use ferrolearn_core::Fit;
use ndarray::{Array1, Array2};

// ─────────────────────────────────────────────────────────────────────────────
// GREEN-GUARD: REQ-1 SHIPPED — contiguous-label transduction PARTITION.
// ─────────────────────────────────────────────────────────────────────────────

/// Green-guard for REQ-1 (SHIPPED): contiguous `{0,1}` transduction partition.
///
/// Uses ferrolearn `new()` defaults (rbf, gamma=20, n_neighbors=7,
/// max_iter=1000, tol=1e-4); the well-separated partition is robust to the
/// tol divergence (verified at tol 1e-3 AND 1e-4).
///
/// FRESH fixture (NOT in-tree): two 3-point blobs near (1,1) and (8,3).
///
/// Live sklearn 1.5.2 oracle:
/// ```text
/// X=np.array([[1.,1.],[1.2,0.9],[0.8,1.1],[8.,3.],[8.2,3.1],[7.9,2.8]])
/// y=np.array([0,-1,-1,1,-1,-1])
/// LabelPropagation(gamma=20.0).fit(X,y).transduction_  ->  [0,0,0,1,1,1]
/// .classes_  ->  [0,1]
/// ```
/// Contiguous `{0,1}` labels make the `classes_` mapping the identity, so the
/// raw argmax index equals `transduction_` — direct comparison is valid here.
#[test]
fn green_guard_req1_contiguous_partition_2blob() {
    // sklearn transduction_ (live oracle, gamma=20.0, tol in {1e-3,1e-4}).
    const SK_TRANSDUCTION: [isize; 6] = [0, 0, 0, 1, 1, 1];
    const SK_N_CLASSES: usize = 2; // len(sklearn classes_ == [0,1])

    let x = Array2::from_shape_vec(
        (6, 2),
        vec![1., 1., 1.2, 0.9, 0.8, 1.1, 8., 3., 8.2, 3.1, 7.9, 2.8],
    )
    .unwrap();
    let y = Array1::from_vec(vec![0, -1, -1, 1, -1, -1]);

    let fitted = LabelPropagation::<f64>::new().fit(&x, &y).unwrap();

    assert_eq!(
        fitted.n_classes(),
        SK_N_CLASSES,
        "contiguous {{0,1}}: n_classes must match sklearn len(classes_)"
    );
    assert_eq!(
        fitted.labels().to_vec(),
        SK_TRANSDUCTION.to_vec(),
        "contiguous {{0,1}} partition must match sklearn transduction_"
    );
}

/// Green-guard for REQ-1 (SHIPPED): contiguous `{0,1,2}` transduction partition.
///
/// FRESH fixture (NOT in-tree): three 3-point blobs near (0,0), (6,6), (0,6),
/// one labeled seed per blob.
///
/// Live sklearn 1.5.2 oracle:
/// ```text
/// X=np.array([[0.,0.],[0.2,0.1],[-0.1,0.2],[6.,6.],[6.1,5.9],[5.8,6.2],
///             [0.,6.],[0.1,6.1],[-0.2,5.9]])
/// y=np.array([0,-1,-1,1,-1,-1,2,-1,-1])
/// LabelPropagation(gamma=20.0).fit(X,y).transduction_ -> [0,0,0,1,1,1,2,2,2]
/// .classes_ -> [0,1,2]
/// ```
#[test]
fn green_guard_req1_contiguous_partition_3blob() {
    const SK_TRANSDUCTION: [isize; 9] = [0, 0, 0, 1, 1, 1, 2, 2, 2];
    const SK_N_CLASSES: usize = 3; // len(sklearn classes_ == [0,1,2])

    let x = Array2::from_shape_vec(
        (9, 2),
        vec![
            0., 0., 0.2, 0.1, -0.1, 0.2, 6., 6., 6.1, 5.9, 5.8, 6.2, 0., 6., 0.1, 6.1, -0.2, 5.9,
        ],
    )
    .unwrap();
    let y = Array1::from_vec(vec![0, -1, -1, 1, -1, -1, 2, -1, -1]);

    let fitted = LabelPropagation::<f64>::new().fit(&x, &y).unwrap();

    assert_eq!(
        fitted.n_classes(),
        SK_N_CLASSES,
        "contiguous {{0,1,2}}: n_classes must match sklearn len(classes_)"
    );
    assert_eq!(
        fitted.labels().to_vec(),
        SK_TRANSDUCTION.to_vec(),
        "contiguous {{0,1,2}} partition must match sklearn transduction_"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// PIN: REQ-4 divergence — classes_/n_classes mapping on non-contiguous labels.
// ─────────────────────────────────────────────────────────────────────────────

/// Divergence: ferrolearn's `LabelPropagation::fit` diverges from
/// `sklearn/semi_supervised/_label_propagation.py:272-274` (`classes =
/// np.unique(y); classes = classes[classes != -1]; self.classes_ = classes`)
/// and `:333` (`transduction = self.classes_[np.argmax(...)]`) for a
/// NON-CONTIGUOUS label set `{0,2}`.
///
/// ferrolearn computes `n_classes = max(label)+1` (`fn fit`, the `.max()...+1`
/// expression at :460-466) and emits the raw argmax INDEX as the label
/// (`best_c as isize`, :498-512), assuming contiguous `0..k` labels.
///
/// Live sklearn 1.5.2 oracle (FRESH fixture; gamma=20.0, tol in {1e-3,1e-4}
/// give the same result):
/// ```text
/// X=np.array([[0.,0.],[0.1,0.],[0.,0.1],[0.1,0.1],
///             [10.,10.],[10.1,10.],[10.,10.1],[10.1,10.1]])
/// y=np.array([0,-1,-1,-1,2,-1,-1,-1])
/// m=LabelPropagation(gamma=20.0).fit(X,y)
/// m.classes_            -> [0, 2]
/// len(m.classes_)       -> 2
/// m.transduction_       -> [0, 0, 0, 0, 2, 2, 2, 2]
/// m.label_distributions_.shape -> (8, 2)
/// ```
/// sklearn returns `n_classes=2` and labels ⊆ `{0,2}`. ferrolearn returns
/// `n_classes()==3` (a `(8,3)` distribution with a phantom class-1 column) —
/// so this assertion FAILS today. The well-separated partition is already
/// correct in ferrolearn (`labels()==[0,0,0,0,2,2,2,2]`), so a single fix that
/// (i) sets `classes_ = sorted unique non-(-1) labels`, (ii) `n_classes =
/// len(classes_)`, (iii) maps the final argmax index through `classes_` flips
/// this pin green WITHOUT needing the entangled RBF-diagonal/zero-init/L2
/// value divergences (REQ-2/3).
///
/// Tracking: #999
#[test]
fn divergence_req4_noncontiguous_classes_mapping() {
    // Live sklearn 1.5.2 oracle (see doc comment above) — NOT copied from ferrolearn.
    const SK_N_CLASSES: usize = 2; // len(classes_ == [0,2])
    const SK_CLASSES: [isize; 2] = [0, 2];
    const SK_TRANSDUCTION: [isize; 8] = [0, 0, 0, 0, 2, 2, 2, 2];

    let x = Array2::from_shape_vec(
        (8, 2),
        vec![
            0., 0., 0.1, 0., 0., 0.1, 0.1, 0.1, 10., 10., 10.1, 10., 10., 10.1, 10.1, 10.1,
        ],
    )
    .unwrap();
    // Non-contiguous label set {0, 2}: second blob's seed is class 2.
    let y = Array1::from_vec(vec![0, -1, -1, -1, 2, -1, -1, -1]);

    let fitted = LabelPropagation::<f64>::new().fit(&x, &y).unwrap();

    // sklearn: n_classes == len(classes_) == 2. ferrolearn: max(label)+1 == 3.
    assert_eq!(
        fitted.n_classes(),
        SK_N_CLASSES,
        "REQ-4: n_classes must be len(unique(y)\\{{-1}}) = 2 (sklearn classes_ == {:?}), \
         not max(label)+1 = 3",
        SK_CLASSES
    );

    // Every emitted label must be one of sklearn's classes_ {0,2}; no phantom 1.
    for &lab in fitted.labels().iter() {
        assert!(
            SK_CLASSES.contains(&lab),
            "REQ-4: label {lab} not in sklearn classes_ {SK_CLASSES:?} (phantom class)"
        );
    }

    // Partition must match sklearn transduction_ exactly.
    assert_eq!(
        fitted.labels().to_vec(),
        SK_TRANSDUCTION.to_vec(),
        "REQ-4: labels must match sklearn transduction_ mapped through classes_"
    );
}
