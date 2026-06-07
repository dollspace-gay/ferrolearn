//! Divergence + green-guard tests for `LabelSpreading` vs scikit-learn 1.5.2
//! (`sklearn/semi_supervised/_label_propagation.py`,
//! `class LabelSpreading(BaseLabelPropagation)` :486-623, `_variant="spreading"`
//! :582).
//!
//! Expected values are computed by the LIVE sklearn 1.5.2 oracle (R-CHAR-3),
//! never literal-copied from ferrolearn. The oracle invocations and outputs are
//! quoted in each test's doc comment.
//!
//! Groups (all GREEN now that REQ-2/3/4/5/6/7 ship):
//!   * GREEN-GUARD: the REQ-1 contiguous-label transduction PARTITION on
//!     well-separated fixtures, including a FRESH fixture not used in-tree.
//!   * REQ-4 (`classes_`/`n_classes` mapping on non-contiguous `{0,2}` labels).
//!   * REQ-2 (`alpha=0` open-interval validation).
//!   * REQ-3/5: `label_distributions_` VALUES + `n_iter_` (normalized-Laplacian
//!     graph, soft-clamp spreading iteration, L1-at-start convergence) — bit-exact
//!     against the live sklearn 1.5.2 oracle across gamma {1,20} and
//!     alpha {0.2,0.5,0.8}, 2- and 3-class, plus a max_iter-hit case.
//!   * REQ-6: `tol` default 1e-3.
//!   * REQ-7: kernel-weighted `predict_proba` / `predict`.

use ferrolearn_cluster::{LabelSpreading, LabelSpreadingKernel};
use ferrolearn_core::{Fit, Predict};
use ndarray::{Array1, Array2};

/// Assert two f64 row-major distribution matrices match within `eps`.
fn assert_ld_close(got: &Array2<f64>, expected: &[[f64; 2]], eps: f64, ctx: &str) {
    assert_eq!(got.nrows(), expected.len(), "{ctx}: row count");
    assert_eq!(got.ncols(), 2, "{ctx}: col count");
    for (i, exp_row) in expected.iter().enumerate() {
        for (c, &exp) in exp_row.iter().enumerate() {
            let g = got[[i, c]];
            assert!(
                (g - exp).abs() <= eps,
                "{ctx}: ld[{i},{c}] = {g} != sklearn {exp} (eps {eps})"
            );
        }
    }
}

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

// ─────────────────────────────────────────────────────────────────────────────
// PARITY: REQ-3 `label_distributions_` VALUES (normalized-Laplacian graph +
// soft-clamp spreading iteration, no per-iter norm) + REQ-5 `n_iter_`.
// (Closes #1010 / #1014's value portion. Live sklearn 1.5.2 oracle.)
//
// The `line` fixture = 4 colinear points [[0,0],[0.3,0],[0.6,0],[1.0,0]] with
// endpoints labeled 0 and 1. The graph matrix is the symmetric normalized
// Laplacian `S = D^{-1/2} W D^{-1/2}` (degree D = OFF-diagonal row sums; scipy
// `csgraph_laplacian` ignores self-loops) with its diagonal zeroed; the
// iteration is `ld = alpha*(graph@ld) + y_static`, y_static = one-hot*(1-alpha),
// NO per-iteration normalization, one final row-normalize.
// ─────────────────────────────────────────────────────────────────────────────

/// REQ-3/5: `label_distributions_` VALUES + `n_iter_` for the default alpha=0.2,
/// gamma=1, tol=1e-3 on the `line` fixture.
///
/// Live sklearn 1.5.2 oracle:
/// ```text
/// X=np.array([[0.,0.],[0.3,0.],[0.6,0.],[1.0,0.]]); y=np.array([0,-1,-1,1])
/// m=LabelSpreading(gamma=1.0,alpha=0.2,tol=1e-3).fit(X,y)
/// m.label_distributions_ ->
///   [[0.95249527,0.04750473],[0.57168677,0.42831323],
///    [0.4557925,0.5442075],[0.04756047,0.95243953]]
/// m.n_iter_ -> 4
/// ```
#[test]
fn parity_req3_label_distributions_line_default_alpha() {
    const SK_LD: [[f64; 2]; 4] = [
        [0.95249527, 0.04750473],
        [0.57168677, 0.42831323],
        [0.4557925, 0.5442075],
        [0.04756047, 0.95243953],
    ];
    const SK_N_ITER: usize = 4;

    let x = Array2::from_shape_vec((4, 2), vec![0., 0., 0.3, 0., 0.6, 0., 1.0, 0.]).unwrap();
    let y = Array1::from_vec(vec![0, -1, -1, 1]);

    let fitted = LabelSpreading::<f64>::new()
        .with_gamma(1.0)
        .with_alpha(0.2)
        .with_tol(1e-3)
        .fit(&x, &y)
        .unwrap();

    assert_ld_close(
        fitted.label_distributions(),
        &SK_LD,
        1e-6,
        "REQ-3 line gamma=1 alpha=0.2",
    );
    assert_eq!(
        fitted.n_iter(),
        SK_N_ITER,
        "REQ-5 n_iter_ must match sklearn (L1-at-start convergence)"
    );
    // Rows sum to 1 (final normalization, sklearn :328-330).
    for i in 0..4 {
        let s: f64 = (0..2).map(|c| fitted.label_distributions()[[i, c]]).sum();
        assert!((s - 1.0).abs() < 1e-12, "row {i} must sum to 1");
    }
}

/// REQ-3/5: `line` fixture at gamma=20 (default-ish gamma), alpha=0.2.
///
/// Live sklearn 1.5.2 oracle:
/// ```text
/// LabelSpreading(gamma=20.0,alpha=0.2,tol=1e-3).fit(line,[0,-1,-1,1])
/// .label_distributions_ ->
///   [[0.9982805,0.0017195],[0.92534976,0.07465024],
///    [0.17625253,0.82374747],[0.00174093,0.99825907]]
/// .n_iter_ -> 5
/// ```
#[test]
fn parity_req3_label_distributions_line_gamma20() {
    const SK_LD: [[f64; 2]; 4] = [
        [0.9982805, 0.0017195],
        [0.92534976, 0.07465024],
        [0.17625253, 0.82374747],
        [0.00174093, 0.99825907],
    ];
    const SK_N_ITER: usize = 5;

    let x = Array2::from_shape_vec((4, 2), vec![0., 0., 0.3, 0., 0.6, 0., 1.0, 0.]).unwrap();
    let y = Array1::from_vec(vec![0, -1, -1, 1]);

    let fitted = LabelSpreading::<f64>::new()
        .with_gamma(20.0)
        .with_alpha(0.2)
        .with_tol(1e-3)
        .fit(&x, &y)
        .unwrap();

    assert_ld_close(
        fitted.label_distributions(),
        &SK_LD,
        1e-6,
        "REQ-3 line gamma=20 alpha=0.2",
    );
    assert_eq!(fitted.n_iter(), SK_N_ITER, "REQ-5 n_iter_ gamma=20");
}

/// REQ-3/5: `line` fixture at alpha=0.5 (gamma=1).
///
/// Live sklearn 1.5.2 oracle:
/// ```text
/// LabelSpreading(gamma=1.0,alpha=0.5,tol=1e-3).fit(line,[0,-1,-1,1])
/// .label_distributions_ ->
///   [[0.84245051,0.15754949],[0.5468566,0.4531434],
///    [0.47719173,0.52280827],[0.15880994,0.84119006]]
/// .n_iter_ -> 6
/// ```
#[test]
fn parity_req3_label_distributions_line_alpha05() {
    const SK_LD: [[f64; 2]; 4] = [
        [0.84245051, 0.15754949],
        [0.5468566, 0.4531434],
        [0.47719173, 0.52280827],
        [0.15880994, 0.84119006],
    ];
    const SK_N_ITER: usize = 6;

    let x = Array2::from_shape_vec((4, 2), vec![0., 0., 0.3, 0., 0.6, 0., 1.0, 0.]).unwrap();
    let y = Array1::from_vec(vec![0, -1, -1, 1]);

    let fitted = LabelSpreading::<f64>::new()
        .with_gamma(1.0)
        .with_alpha(0.5)
        .with_tol(1e-3)
        .fit(&x, &y)
        .unwrap();

    assert_ld_close(
        fitted.label_distributions(),
        &SK_LD,
        1e-6,
        "REQ-3 line gamma=1 alpha=0.5",
    );
    assert_eq!(fitted.n_iter(), SK_N_ITER, "REQ-5 n_iter_ alpha=0.5");
}

/// REQ-3/5: `line` fixture at alpha=0.8 (gamma=1).
///
/// Live sklearn 1.5.2 oracle:
/// ```text
/// LabelSpreading(gamma=1.0,alpha=0.8,tol=1e-3).fit(line,[0,-1,-1,1])
/// .label_distributions_ ->
///   [[0.67096594,0.32903406],[0.52398538,0.47601462],
///    [0.49707557,0.50292443],[0.33656799,0.66343201]]
/// .n_iter_ -> 9
/// ```
#[test]
fn parity_req3_label_distributions_line_alpha08() {
    const SK_LD: [[f64; 2]; 4] = [
        [0.67096594, 0.32903406],
        [0.52398538, 0.47601462],
        [0.49707557, 0.50292443],
        [0.33656799, 0.66343201],
    ];
    const SK_N_ITER: usize = 9;

    let x = Array2::from_shape_vec((4, 2), vec![0., 0., 0.3, 0., 0.6, 0., 1.0, 0.]).unwrap();
    let y = Array1::from_vec(vec![0, -1, -1, 1]);

    let fitted = LabelSpreading::<f64>::new()
        .with_gamma(1.0)
        .with_alpha(0.8)
        .with_tol(1e-3)
        .fit(&x, &y)
        .unwrap();

    assert_ld_close(
        fitted.label_distributions(),
        &SK_LD,
        1e-6,
        "REQ-3 line gamma=1 alpha=0.8",
    );
    assert_eq!(fitted.n_iter(), SK_N_ITER, "REQ-5 n_iter_ alpha=0.8");
}

/// REQ-3/5: a NON-degenerate 3-class fixture (close clusters so the
/// `label_distributions_` rows are genuinely mixed, not 1.0/0.0).
///
/// Live sklearn 1.5.2 oracle:
/// ```text
/// X=np.array([[0.,0.],[1.,0.],[2.,0.],[2.,2.],[0.,2.],[1.,1.]])
/// y=np.array([0,-1,1,-1,2,-1])
/// m=LabelSpreading(gamma=1.0,alpha=0.2,tol=1e-3).fit(X,y)
/// m.label_distributions_ ->
///  [[0.9675461,0.01756497,0.01488893],[0.47612568,0.47612568,0.04774864],
///   [0.01775611,0.97807469,0.0041692],[0.09525009,0.34385578,0.56089413],
///   [0.01518431,0.00420614,0.98060955],[0.28927374,0.28927374,0.42145253]]
/// m.classes_ -> [0,1,2]   m.transduction_ -> [0,0,1,2,2,2]   m.n_iter_ -> 5
/// ```
#[test]
fn parity_req3_label_distributions_three_class() {
    // 3-column oracle rows (live sklearn 1.5.2).
    const SK_LD: [[f64; 3]; 6] = [
        [0.9675461, 0.01756497, 0.01488893],
        [0.47612568, 0.47612568, 0.04774864],
        [0.01775611, 0.97807469, 0.0041692],
        [0.09525009, 0.34385578, 0.56089413],
        [0.01518431, 0.00420614, 0.98060955],
        [0.28927374, 0.28927374, 0.42145253],
    ];
    const SK_TRANSDUCTION: [isize; 6] = [0, 0, 1, 2, 2, 2];
    const SK_CLASSES: [isize; 3] = [0, 1, 2];
    const SK_N_ITER: usize = 5;

    let x = Array2::from_shape_vec((6, 2), vec![0., 0., 1., 0., 2., 0., 2., 2., 0., 2., 1., 1.])
        .unwrap();
    let y = Array1::from_vec(vec![0, -1, 1, -1, 2, -1]);

    let fitted = LabelSpreading::<f64>::new()
        .with_gamma(1.0)
        .with_alpha(0.2)
        .with_tol(1e-3)
        .fit(&x, &y)
        .unwrap();

    assert_eq!(fitted.n_classes(), 3, "3 classes");
    assert_eq!(fitted.classes(), SK_CLASSES, "classes_ must be [0,1,2]");
    let ld = fitted.label_distributions();
    for (i, exp_row) in SK_LD.iter().enumerate() {
        for (c, &exp) in exp_row.iter().enumerate() {
            let g = ld[[i, c]];
            assert!(
                (g - exp).abs() <= 1e-6,
                "REQ-3 three-class ld[{i},{c}] = {g} != sklearn {exp}"
            );
        }
    }
    assert_eq!(
        fitted.labels().to_vec(),
        SK_TRANSDUCTION.to_vec(),
        "REQ-3 three-class transduction_"
    );
    assert_eq!(fitted.n_iter(), SK_N_ITER, "REQ-5 three-class n_iter_");
}

/// REQ-5: a max_iter-hit case → `n_iter_ == max_iter` (sklearn loop `else:`
/// increments `n_iter_` on non-convergence, `_label_propagation.py:321-326`).
///
/// Live sklearn 1.5.2 oracle:
/// ```text
/// LabelSpreading(gamma=1.0,alpha=0.8,tol=1e-12,max_iter=5).fit(line,[0,-1,-1,1])
/// .label_distributions_ ->
///   [[0.67100643,0.32899357],[0.52569192,0.47430808],
///    [0.49564182,0.50435818],[0.3363551,0.6636449]]
/// .n_iter_ -> 5   (== max_iter; never converged at tol=1e-12)
/// ```
#[test]
fn parity_req5_n_iter_max_iter_hit() {
    const SK_LD: [[f64; 2]; 4] = [
        [0.67100643, 0.32899357],
        [0.52569192, 0.47430808],
        [0.49564182, 0.50435818],
        [0.3363551, 0.6636449],
    ];
    const SK_MAX_ITER: usize = 5;

    let x = Array2::from_shape_vec((4, 2), vec![0., 0., 0.3, 0., 0.6, 0., 1.0, 0.]).unwrap();
    let y = Array1::from_vec(vec![0, -1, -1, 1]);

    let fitted = LabelSpreading::<f64>::new()
        .with_gamma(1.0)
        .with_alpha(0.8)
        .with_tol(1e-12)
        .with_max_iter(SK_MAX_ITER)
        .fit(&x, &y)
        .unwrap();

    assert_eq!(
        fitted.n_iter(),
        SK_MAX_ITER,
        "REQ-5: n_iter_ must equal max_iter on non-convergence (sklearn :321-326)"
    );
    assert_ld_close(
        fitted.label_distributions(),
        &SK_LD,
        1e-6,
        "REQ-5 max_iter-hit ld",
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// PARITY: REQ-6 `tol` default 1e-3 (sklearn LabelSpreading tol=1e-3, :595).
// ─────────────────────────────────────────────────────────────────────────────

/// REQ-6: ferrolearn `new()` default `tol` matches sklearn `LabelSpreading().tol`.
///
/// Live sklearn 1.5.2 oracle:
/// ```text
/// from sklearn.semi_supervised import LabelSpreading
/// LabelSpreading().tol       -> 0.001
/// LabelSpreading().alpha     -> 0.2
/// LabelSpreading().max_iter  -> 30
/// LabelSpreading().gamma     -> 20
/// LabelSpreading().n_neighbors -> 7
/// ```
#[test]
fn parity_req6_default_tol_and_params() {
    let m = LabelSpreading::<f64>::new();
    assert_eq!(m.tol, 1e-3, "REQ-6: default tol must be sklearn 1e-3");
    assert_eq!(m.alpha, 0.2, "default alpha 0.2");
    assert_eq!(m.max_iter, 30, "default max_iter 30");
    assert_eq!(m.gamma, 20.0, "default gamma 20");
    assert_eq!(m.n_neighbors, 7, "default n_neighbors 7");
    assert_eq!(m.kernel, LabelSpreadingKernel::Rbf, "default kernel rbf");
}

// ─────────────────────────────────────────────────────────────────────────────
// PARITY: REQ-7 kernel-weighted predict_proba / predict.
// (sklearn predict_proba = rbf_kernel(X_train,X).T @ ld, row-normalized; :218-231.)
// ─────────────────────────────────────────────────────────────────────────────

/// REQ-7: `predict_proba` is the kernel-weighted combination over ALL training
/// rows (NOT the nearest training row), row-normalized; `predict` is the argmax
/// mapped through `classes_`.
///
/// Live sklearn 1.5.2 oracle:
/// ```text
/// X=np.array([[0.,0.],[0.3,0.],[0.6,0.],[1.0,0.]]); y=np.array([0,-1,-1,1])
/// m=LabelSpreading(gamma=1.0,alpha=0.2,tol=1e-3).fit(X,y)
/// Xq=np.array([[0.15,0.],[0.5,0.],[0.9,0.]])
/// m.predict_proba(Xq) ->
///   [[0.57880954,0.42119046],[0.5071689,0.4928311],[0.42219637,0.57780363]]
/// m.predict(Xq) -> [0, 0, 1]
/// ```
#[test]
fn parity_req7_predict_proba_kernel_weighted() {
    const SK_PP: [[f64; 2]; 3] = [
        [0.57880954, 0.42119046],
        [0.5071689, 0.4928311],
        [0.42219637, 0.57780363],
    ];
    const SK_PRED: [isize; 3] = [0, 0, 1];

    let x = Array2::from_shape_vec((4, 2), vec![0., 0., 0.3, 0., 0.6, 0., 1.0, 0.]).unwrap();
    let y = Array1::from_vec(vec![0, -1, -1, 1]);

    let fitted = LabelSpreading::<f64>::new()
        .with_gamma(1.0)
        .with_alpha(0.2)
        .with_tol(1e-3)
        .fit(&x, &y)
        .unwrap();

    let xq = Array2::from_shape_vec((3, 2), vec![0.15, 0., 0.5, 0., 0.9, 0.]).unwrap();
    let pp = fitted.predict_proba(&xq).unwrap();
    assert_ld_close(&pp, &SK_PP, 1e-6, "REQ-7 predict_proba kernel-weighted");

    // Each row sums to 1 (sklearn :229-230).
    for i in 0..3 {
        let s: f64 = (0..2).map(|c| pp[[i, c]]).sum();
        assert!((s - 1.0).abs() < 1e-12, "predict_proba row {i} sums to 1");
    }

    let pred = fitted.predict(&xq).unwrap();
    assert_eq!(
        pred.to_vec(),
        SK_PRED.to_vec(),
        "REQ-7 predict = classes_[argmax(predict_proba)]"
    );
}

/// REQ-4 mapping survives the value rewrite: non-contiguous `{0,2}` labels still
/// map through `classes_` (no phantom class 1). Live oracle: Probe B.
#[test]
fn parity_req4_noncontiguous_preserved_after_value_fix() {
    const SK_N_CLASSES: usize = 2;
    const SK_CLASSES: [isize; 2] = [0, 2];
    const SK_TRANSDUCTION: [isize; 8] = [0, 0, 0, 0, 2, 2, 2, 2];

    let x = Array2::from_shape_vec(
        (8, 2),
        vec![
            0., 0., 0.1, 0., 0., 0.1, 0.1, 0.1, 10., 10., 10.1, 10., 10., 10.1, 10.1, 10.1,
        ],
    )
    .unwrap();
    let y = Array1::from_vec(vec![0, -1, -1, -1, 2, -1, -1, -1]);

    let fitted = LabelSpreading::<f64>::new().fit(&x, &y).unwrap();
    assert_eq!(fitted.n_classes(), SK_N_CLASSES);
    assert_eq!(fitted.classes(), SK_CLASSES);
    assert_eq!(fitted.labels().to_vec(), SK_TRANSDUCTION.to_vec());
}
