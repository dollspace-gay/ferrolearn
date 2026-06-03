//! Divergence + green-guard tests for `LabelSpreading` vs scikit-learn 1.5.2
//! (`sklearn/semi_supervised/_label_propagation.py`,
//! `class LabelSpreading(BaseLabelPropagation)` :486-623, `_variant="spreading"`
//! :582).
//!
//! Expected values are computed by the LIVE sklearn 1.5.2 oracle (R-CHAR-3),
//! never literal-copied from ferrolearn. The oracle invocations and outputs are
//! quoted in each test's doc comment.
//!
//! Three groups:
//!   * GREEN-GUARD (PASS today): the SHIPPED REQ-1 contiguous-label transduction
//!     PARTITION on well-separated fixtures, including a FRESH fixture not used
//!     in-tree.
//!   * PIN (FAIL until #1011 fixed): REQ-4 `classes_`/`n_classes` mapping on a
//!     non-contiguous `{0,2}` label set (identical bug to label_propagation #999).
//!   * PIN (FAIL until #1009 fixed): REQ-2 `alpha=0` open-interval validation.
//!
//! REQ-4 (class-mapping) and REQ-2 (validation guard) touch DISJOINT code paths
//! and are independent — neither pin interferes with the other.

use ferrolearn_cluster::LabelSpreading;
use ferrolearn_core::Fit;
use ndarray::{Array1, Array2};

// ─────────────────────────────────────────────────────────────────────────────
// GREEN-GUARD: REQ-1 SHIPPED — contiguous-label transduction PARTITION.
// ─────────────────────────────────────────────────────────────────────────────

/// Green-guard for REQ-1 (SHIPPED): contiguous `{0,1}` transduction partition
/// on a well-separated two-blob fixture, using ferrolearn `new()` defaults
/// (rbf, gamma=20, alpha=0.2, max_iter=30, tol=1e-4).
///
/// NOTE: ferrolearn default tol=1e-4 vs sklearn 1e-3 (REQ-6), and the
/// `label_distributions_` VALUES diverge (REQ-3) — but on well-separated blobs
/// the argmax PARTITION is robust to both (verified at tol 1e-3 AND 1e-4 below).
/// Contiguous `{0,1}` labels make sklearn's `classes_` mapping the identity, so
/// the raw argmax index equals `transduction_` — direct comparison is valid.
///
/// Live sklearn 1.5.2 oracle:
/// ```text
/// X=np.array([[0.,0.],[0.1,0.],[0.,0.1],[0.1,0.1],
///             [10.,10.],[10.1,10.],[10.,10.1],[10.1,10.1]])
/// y=np.array([0,-1,-1,-1,1,-1,-1,-1])
/// LabelSpreading().fit(X,y).transduction_  ->  [0,0,0,0,1,1,1,1]
/// .classes_                                 ->  [0,1]
/// ```
#[test]
fn green_guard_req1_contiguous_partition_2blob() {
    // sklearn transduction_ (live oracle, default params, tol in {1e-3,1e-4}).
    const SK_TRANSDUCTION: [isize; 8] = [0, 0, 0, 0, 1, 1, 1, 1];
    const SK_N_CLASSES: usize = 2; // len(sklearn classes_ == [0,1])

    let x = Array2::from_shape_vec(
        (8, 2),
        vec![
            0., 0., 0.1, 0., 0., 0.1, 0.1, 0.1, 10., 10., 10.1, 10., 10., 10.1, 10.1, 10.1,
        ],
    )
    .unwrap();
    let y = Array1::from_vec(vec![0, -1, -1, -1, 1, -1, -1, -1]);

    let fitted = LabelSpreading::<f64>::new().fit(&x, &y).unwrap();

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

/// Green-guard for REQ-1 (SHIPPED): contiguous `{0,1,2}` transduction partition
/// on a well-separated three-blob fixture (default params).
///
/// Live sklearn 1.5.2 oracle:
/// ```text
/// X=np.array([[0.,0.],[0.1,0.],[0.,0.1],
///             [10.,10.],[10.1,10.],[10.,10.1],
///             [0.,10.],[0.1,10.],[0.,10.1]])
/// y=np.array([0,-1,-1,1,-1,-1,2,-1,-1])
/// LabelSpreading().fit(X,y).transduction_ -> [0,0,0,1,1,1,2,2,2]
/// .classes_ -> [0,1,2]
/// ```
#[test]
fn green_guard_req1_contiguous_partition_3blob() {
    const SK_TRANSDUCTION: [isize; 9] = [0, 0, 0, 1, 1, 1, 2, 2, 2];
    const SK_N_CLASSES: usize = 3; // len(sklearn classes_ == [0,1,2])

    let x = Array2::from_shape_vec(
        (9, 2),
        vec![
            0., 0., 0.1, 0., 0., 0.1, 10., 10., 10.1, 10., 10., 10.1, 0., 10., 0.1, 10., 0., 10.1,
        ],
    )
    .unwrap();
    let y = Array1::from_vec(vec![0, -1, -1, 1, -1, -1, 2, -1, -1]);

    let fitted = LabelSpreading::<f64>::new().fit(&x, &y).unwrap();

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

/// Green-guard for REQ-1 (SHIPPED): FRESH separable contiguous-label fixture
/// not used anywhere in-tree (two 3-point blobs near (2,2) and (15,9)).
///
/// Live sklearn 1.5.2 oracle (default params; tol in {1e-3,1e-4} agree):
/// ```text
/// X=np.array([[2.,2.],[2.3,1.8],[1.7,2.2],[15.,9.],[15.2,9.3],[14.8,8.7]])
/// y=np.array([0,-1,-1,1,-1,-1])
/// LabelSpreading().fit(X,y).transduction_ -> [0,0,0,1,1,1]
/// .classes_ -> [0,1]
/// ```
#[test]
fn green_guard_req1_contiguous_partition_fresh() {
    const SK_TRANSDUCTION: [isize; 6] = [0, 0, 0, 1, 1, 1];
    const SK_N_CLASSES: usize = 2; // len(sklearn classes_ == [0,1])

    let x = Array2::from_shape_vec(
        (6, 2),
        vec![2., 2., 2.3, 1.8, 1.7, 2.2, 15., 9., 15.2, 9.3, 14.8, 8.7],
    )
    .unwrap();
    let y = Array1::from_vec(vec![0, -1, -1, 1, -1, -1]);

    let fitted = LabelSpreading::<f64>::new().fit(&x, &y).unwrap();

    assert_eq!(
        fitted.n_classes(),
        SK_N_CLASSES,
        "fresh contiguous {{0,1}}: n_classes must match sklearn len(classes_)"
    );
    assert_eq!(
        fitted.labels().to_vec(),
        SK_TRANSDUCTION.to_vec(),
        "fresh contiguous {{0,1}} partition must match sklearn transduction_"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// PIN: REQ-4 divergence — classes_/n_classes mapping on non-contiguous labels.
// (Tracking #1011 — identical bug to the just-fixed label_propagation #999.)
// ─────────────────────────────────────────────────────────────────────────────

/// Divergence: ferrolearn's `LabelSpreading::fit` diverges from
/// `sklearn/semi_supervised/_label_propagation.py:272-274` (`classes =
/// np.unique(y); classes = classes[classes != -1]; self.classes_ = classes`)
/// and `:333` (`transduction = self.classes_[np.argmax(...)]`) for a
/// NON-CONTIGUOUS label set `{0,2}`.
///
/// ferrolearn computes `n_classes = max(label)+1` (`fn fit`, the `.max()...+1`
/// expression) and emits the raw argmax INDEX as the label (`best_c as isize`,
/// the argmax loop), assuming contiguous `0..k` labels.
///
/// Live sklearn 1.5.2 oracle (default params):
/// ```text
/// X=np.array([[0.,0.],[0.1,0.],[0.,0.1],[0.1,0.1],
///             [10.,10.],[10.1,10.],[10.,10.1],[10.1,10.1]])
/// y=np.array([0,-1,-1,-1,2,-1,-1,-1])
/// m=LabelSpreading().fit(X,y)
/// m.classes_            -> [0, 2]
/// len(m.classes_)       -> 2
/// m.transduction_       -> [0, 0, 0, 0, 2, 2, 2, 2]
/// m.label_distributions_.shape -> (8, 2)
/// ```
/// sklearn returns `n_classes=2` and labels ⊆ `{0,2}`. ferrolearn returns
/// `n_classes()==3` (a `(8,3)` distribution with a phantom class-1 column) — so
/// this assertion FAILS today. The well-separated partition is already correct,
/// so the fix that (i) sets `classes_ = sorted unique non-(-1) labels`,
/// (ii) `n_classes = len(classes_)`, (iii) maps the final argmax index through
/// `classes_` flips this pin green WITHOUT needing the entangled
/// RBF-diagonal/zero-init/L2 value divergences (REQ-3/5).
///
/// Tracking: #1011
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

    let fitted = LabelSpreading::<f64>::new().fit(&x, &y).unwrap();

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

// ─────────────────────────────────────────────────────────────────────────────
// PIN: REQ-2 divergence — alpha=0 open-interval (0,1) validation.
// (Tracking #1009.)
// ─────────────────────────────────────────────────────────────────────────────

/// Divergence: ferrolearn's `LabelSpreading::fit` diverges from
/// `sklearn/semi_supervised/_label_propagation.py:585`
/// (`_parameter_constraints["alpha"] = [Interval(Real, 0, 1, closed="neither")]`)
/// for `alpha=0`.
///
/// sklearn's constraint is the OPEN interval `(0,1)` — BOTH `alpha=0` and
/// `alpha=1` are rejected with `InvalidParameterError`. ferrolearn `fn fit`
/// rejects only `alpha < 0 || alpha >= 1`, ALLOWING `alpha=0`.
///
/// Live sklearn 1.5.2 oracle:
/// ```text
/// X=np.array([[0.,0.],[1.,1.]]); y=np.array([0,1])
/// LabelSpreading(alpha=0).fit(X,y)  ->  raises InvalidParameterError
/// LabelSpreading(alpha=1).fit(X,y)  ->  raises InvalidParameterError
/// ```
/// sklearn rejects `alpha=0`; ferrolearn returns `Ok` — this assertion FAILS
/// today. The fix flips the lower bound to a strict reject (`alpha <= 0`) and
/// requires removing/updating the in-tree `test_alpha_zero_recovers_initial`
/// (which currently relies on `alpha=0` being accepted — R-HONEST-4, flagged
/// for the fixer).
///
/// Tracking: #1009
#[test]
fn divergence_req2_alpha_zero_rejected() {
    let x = Array2::from_shape_vec(
        (8, 2),
        vec![
            0., 0., 0.1, 0., 0., 0.1, 0.1, 0.1, 10., 10., 10.1, 10., 10., 10.1, 10.1, 10.1,
        ],
    )
    .unwrap();
    let y = Array1::from_vec(vec![0, -1, -1, -1, 1, -1, -1, -1]);

    // sklearn: alpha=0 is OUTSIDE the open interval (0,1) -> InvalidParameterError.
    let result = LabelSpreading::<f64>::new().with_alpha(0.0).fit(&x, &y);
    assert!(
        result.is_err(),
        "REQ-2: alpha=0 must be rejected (sklearn open interval (0,1), \
         _label_propagation.py:585); ferrolearn currently returns Ok"
    );
}

/// Companion confirmation (PASS today): `alpha=1` is ALREADY rejected by
/// ferrolearn (`alpha >= 1` guard), matching sklearn's open-interval upper
/// bound — so only the `alpha=0` side (REQ-2) diverges. This isolates the
/// divergence to the lower bound.
///
/// Live sklearn 1.5.2 oracle: `LabelSpreading(alpha=1).fit(X,y)` raises
/// `InvalidParameterError` (`_label_propagation.py:585`).
#[test]
fn confirm_alpha_one_already_rejected() {
    let x = Array2::from_shape_vec(
        (8, 2),
        vec![
            0., 0., 0.1, 0., 0., 0.1, 0.1, 0.1, 10., 10., 10.1, 10., 10., 10.1, 10.1, 10.1,
        ],
    )
    .unwrap();
    let y = Array1::from_vec(vec![0, -1, -1, -1, 1, -1, -1, -1]);

    let result = LabelSpreading::<f64>::new().with_alpha(1.0).fit(&x, &y);
    assert!(
        result.is_err(),
        "alpha=1 must be rejected (sklearn open interval (0,1)); \
         confirms only the alpha=0 lower bound diverges"
    );
}
