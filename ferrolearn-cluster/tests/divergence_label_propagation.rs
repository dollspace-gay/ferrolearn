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
use ferrolearn_core::{Fit, Predict};
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

// ─────────────────────────────────────────────────────────────────────────────
// VALUE PARITY: REQ-2/3 — label_distributions_ + n_iter_ bit-exact.
// ─────────────────────────────────────────────────────────────────────────────

/// `line` fixture: 4 collinear points, endpoints labeled `{0,1}`, the two
/// interior points unlabeled — overlapping enough to make the soft-label
/// VALUES (not just the partition) load-bearing.
fn line_fixture() -> (Array2<f64>, Array1<isize>) {
    let x = Array2::from_shape_vec((4, 2), vec![0., 0., 0.3, 0., 0.6, 0., 1.0, 0.]).unwrap();
    let y = Array1::from_vec(vec![0, -1, -1, 1]);
    (x, y)
}

/// REQ-2 (`label_distributions_` VALUES) + REQ-3 (`n_iter_`, L1-at-start).
///
/// Live sklearn 1.5.2 oracle:
/// ```text
/// X=np.array([[0.,0.],[0.3,0.],[0.6,0.],[1.0,0.]]); y=np.array([0,-1,-1,1])
/// m=LabelPropagation(gamma=1.0,tol=1e-3).fit(X,y)
/// m.label_distributions_ -> [[1.0,0.0],[0.55810978,0.44189022],
///                            [0.49024013,0.50975987],[0.0,1.0]]
/// m.n_iter_              -> 4
/// m.classes_            -> [0,1]
/// ```
/// These values come from (a) the `rbf_kernel(X,X)` diagonal `= 1`,
/// (b) unlabeled rows STARTING at zero, (c) column-sum graph normalization, and
/// (d) the L1-at-start convergence loop stopping at `n_iter_=4`. Asserted to
/// 1e-6 — NOT copied from ferrolearn (R-CHAR-3).
#[test]
fn divergence_req2_3_label_distributions_line() {
    // Live sklearn 1.5.2 oracle (see doc comment).
    const SK_LD: [[f64; 2]; 4] = [
        [1.0, 0.0],
        [0.558_109_78, 0.441_890_22],
        [0.490_240_13, 0.509_759_87],
        [0.0, 1.0],
    ];
    const SK_N_ITER: usize = 4;

    let (x, y) = line_fixture();
    let fitted = LabelPropagation::<f64>::new()
        .with_gamma(1.0)
        .with_tol(1e-3)
        .fit(&x, &y)
        .unwrap();

    let ld = fitted.label_distributions();
    assert_eq!(ld.nrows(), 4);
    assert_eq!(ld.ncols(), 2);
    for (i, sk_row) in SK_LD.iter().enumerate() {
        for (c, &sk_v) in sk_row.iter().enumerate() {
            let got = ld[[i, c]];
            assert!(
                (got - sk_v).abs() < 1e-6,
                "REQ-2/3: label_distributions_[{i},{c}] = {got} but sklearn = {sk_v}"
            );
        }
        // Each row sums to 1.
        let s: f64 = (0..2).map(|c| ld[[i, c]]).sum();
        assert!(
            (s - 1.0).abs() < 1e-9,
            "REQ-2/3: row {i} must sum to 1, got {s}"
        );
    }

    assert_eq!(
        fitted.n_iter(),
        SK_N_ITER,
        "REQ-3: n_iter_ must match sklearn's L1-at-start convergence count"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// DEFAULT PARITY: REQ-5 — tol default 1e-3.
// ─────────────────────────────────────────────────────────────────────────────

/// REQ-5: the `tol` default is `1e-3`.
///
/// Live sklearn 1.5.2 oracle:
/// ```text
/// LabelPropagation().tol -> 0.001
/// ```
#[test]
fn divergence_req5_tol_default() {
    // sklearn LabelPropagation().tol == 0.001 (_label_propagation.py:435).
    const SK_TOL: f64 = 1e-3;
    let model = LabelPropagation::<f64>::new();
    assert!(
        (model.tol - SK_TOL).abs() < 1e-18,
        "REQ-5: tol default must be sklearn's 1e-3, got {}",
        model.tol
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// PREDICT PARITY: REQ-6 — kernel-weighted predict_proba / predict.
// ─────────────────────────────────────────────────────────────────────────────

/// REQ-6: `predict_proba` is the kernel-weighted combination over ALL training
/// rows (`rbf_kernel(X_train, X).T @ label_distributions_`, row-normalized),
/// and `predict = classes_[argmax(predict_proba)]`.
///
/// Live sklearn 1.5.2 oracle (query points NOT equal to any training row, so
/// the second-nearest training row carries non-negligible weight — a
/// nearest-neighbor rule would disagree):
/// ```text
/// X=np.array([[0.,0.],[0.3,0.],[0.6,0.],[1.0,0.]]); y=np.array([0,-1,-1,1])
/// m=LabelPropagation(gamma=1.0,tol=1e-3).fit(X,y)
/// q=np.array([[0.5,0.0],[0.2,0.0]])
/// m.predict_proba(q) -> [[0.51315926,0.48684074],[0.57986224,0.42013776]]
/// m.predict(q)       -> [0,0]
/// ```
/// Asserted to 1e-6 — NOT copied from ferrolearn (R-CHAR-3).
#[test]
fn divergence_req6_predict_proba_kernel_weighted() {
    // Live sklearn 1.5.2 oracle (see doc comment).
    const SK_PROBA: [[f64; 2]; 2] = [[0.513_159_26, 0.486_840_74], [0.579_862_24, 0.420_137_76]];
    const SK_PRED: [isize; 2] = [0, 0];

    let (x, y) = line_fixture();
    let fitted = LabelPropagation::<f64>::new()
        .with_gamma(1.0)
        .with_tol(1e-3)
        .fit(&x, &y)
        .unwrap();

    let q = Array2::from_shape_vec((2, 2), vec![0.5, 0.0, 0.2, 0.0]).unwrap();
    let proba = fitted.predict_proba(&q).unwrap();
    assert_eq!(proba.nrows(), 2);
    assert_eq!(proba.ncols(), 2);
    for (i, sk_row) in SK_PROBA.iter().enumerate() {
        for (c, &sk_v) in sk_row.iter().enumerate() {
            let got = proba[[i, c]];
            assert!(
                (got - sk_v).abs() < 1e-6,
                "REQ-6: predict_proba[{i},{c}] = {got} but sklearn = {sk_v}"
            );
        }
        let s: f64 = (0..2).map(|c| proba[[i, c]]).sum();
        assert!(
            (s - 1.0).abs() < 1e-9,
            "REQ-6: predict_proba row {i} must sum to 1, got {s}"
        );
    }

    let pred = fitted.predict(&q).unwrap();
    assert_eq!(
        pred.to_vec(),
        SK_PRED.to_vec(),
        "REQ-6: predict must be classes_[argmax(predict_proba)] matching sklearn"
    );
}

/// REQ-6 cross-check on a non-contiguous `{0,2}` label set: `predict` must map
/// the argmax INDEX through `classes_` (never emit a phantom label).
///
/// Live sklearn 1.5.2 oracle:
/// ```text
/// X=np.array([[0.,0.],[0.1,0.],[0.,0.1],[0.1,0.1],
///             [10.,10.],[10.1,10.],[10.,10.1],[10.1,10.1]])
/// y=np.array([0,-1,-1,-1,2,-1,-1,-1]); m=LabelPropagation().fit(X,y)
/// m.predict(np.array([[0.05,0.05],[10.05,10.05]])) -> [0, 2]
/// ```
#[test]
fn divergence_req6_predict_noncontiguous_classes() {
    const SK_PRED: [isize; 2] = [0, 2];

    let x = Array2::from_shape_vec(
        (8, 2),
        vec![
            0., 0., 0.1, 0., 0., 0.1, 0.1, 0.1, 10., 10., 10.1, 10., 10., 10.1, 10.1, 10.1,
        ],
    )
    .unwrap();
    let y = Array1::from_vec(vec![0, -1, -1, -1, 2, -1, -1, -1]);
    let fitted = LabelPropagation::<f64>::new().fit(&x, &y).unwrap();

    let q = Array2::from_shape_vec((2, 2), vec![0.05, 0.05, 10.05, 10.05]).unwrap();
    let pred = fitted.predict(&q).unwrap();
    assert_eq!(
        pred.to_vec(),
        SK_PRED.to_vec(),
        "REQ-6: predict on {{0,2}} labels must map through classes_ ({{0,2}}), no phantom"
    );
    assert_eq!(fitted.classes(), &[0, 2], "REQ-4/6: classes_ must be [0,2]");
}
