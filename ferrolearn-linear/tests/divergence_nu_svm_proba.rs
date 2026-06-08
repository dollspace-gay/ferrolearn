//! Divergence pins for `ferrolearn-linear` `NuSVC(probability=true)`
//! `predict_proba` (REQ-9, #653) against the LIVE scikit-learn 1.5.2 oracle.
//!
//! These pin a STRUCTURAL-CONTRACT divergence that is DISTINCT from the
//! documented R-DEV-4 RNG-CV value non-determinism: sklearn's `probA_`/`probB_`
//! VALUES vary slightly with `random_state`, but the structural properties
//! asserted here (a non-degenerate sigmoid; `predict_proba` of a clearly
//! in-cluster query peaking at that query's class) hold for EVERY
//! `random_state` in sklearn. ferrolearn's DETERMINISTIC contiguous 5-fold split
//! produces single-class training folds for small per-ovo-pair sample counts,
//! so the held-out decisions are perfectly-separable constants and
//! `sigmoid_train` returns the degenerate `(A,B)=(0,0)`. With `A=B=0`,
//! `sigmoid_predict` is identically `0.5` for every pair, so the Wu-Lin-Weng
//! coupling collapses `predict_proba` to the uniform `1/n_classes` regardless of
//! the decision value — a broken probability distribution that does not track
//! the cluster, unlike sklearn.
//!
//! This is NOT the bit-value mismatch the builder documented (R-DEV-4): the
//! invariant breaks across ALL random_states, not just at specific seeds.

use ferrolearn_core::{Fit, Predict};
use ferrolearn_linear::nu_svm::NuSVC;
use ferrolearn_linear::svm::LinearKernel;
use ndarray::{Array2, array};

/// Divergence: `FittedNuSVC::predict_proba` (delegating to
/// `FittedSVC::predict_proba`, `ferrolearn-linear/src/svm.rs:1696`, via the
/// per-ovo-pair `platt_cv_sigmoid` at `svm.rs:1026`) diverges from
/// `sklearn.svm.NuSVC(probability=True)`
/// (`sklearn/svm/src/libsvm/svm.cpp:2107-2203` `svm_binary_svc_probability`).
///
/// Input: a 3-class linear-separable set (3 samples/class), `nu=0.5`. The query
/// `[5.0, 0.0]` sits in the heart of the class-1 cluster.
///
/// sklearn (live 1.5.2, STABLE across random_state 0/1/42):
/// ```text
/// X=[[0,0],[0.5,0],[0,0.5],[5,0],[5.5,0],[5,0.5],[0,5],[0.5,5],[0,5.5]]
/// y=[0,0,0,1,1,1,2,2,2]
/// NuSVC(kernel='linear',nu=0.5,probability=True,random_state=rs)
///   .predict_proba([[0,0],[5,0],[0,5]]).argmax(1) == [0, 1, 2]  for rs in {0,1,42}
///   row 1 (q=[5,0]) proba ~ [0.19, 0.70, 0.11]  -> argmax = class 1
/// ```
/// ferrolearn: `probA_ = probB_ = [0,0,0]` (degenerate); every pairwise sigmoid
/// is 0.5; `predict_proba` row 1 = `[0.3333, 0.3333, 0.3333]`, argmax = class 0.
///
/// The asserted invariant (argmax of a clearly in-cluster query == its class) is
/// sklearn's contract, NOT a copied ferrolearn value (R-CHAR-3); it holds for
/// every sklearn random_state.
/// Tracking: #2245
#[test]
#[ignore = "divergence: NuSVC predict_proba degenerates to uniform 1/N (probA_=0) on small per-pair folds; tracking #2245"]
fn divergence_nusvc_predict_proba_multiclass_argmax_orientation() {
    let x = Array2::from_shape_vec(
        (9, 2),
        vec![
            0.0, 0.0, 0.5, 0.0, 0.0, 0.5, 5.0, 0.0, 5.5, 0.0, 5.0, 0.5, 0.0, 5.0, 0.5, 5.0, 0.0,
            5.5,
        ],
    )
    .unwrap();
    let y = array![0usize, 0, 0, 1, 1, 1, 2, 2, 2];

    // sklearn oracle: predict_proba argmax for in-cluster queries (stable across
    // random_state) — NOT copied from ferrolearn.
    let oracle_argmax = [0usize, 1, 2];

    let ft = NuSVC::<f64, LinearKernel>::new(LinearKernel)
        .with_nu(0.5)
        .with_tol(1e-6)
        .with_max_iter(200_000)
        .with_probability(true)
        .fit(&x, &y)
        .unwrap();

    // predict (decision path) is correct — only predict_proba diverges.
    let q = Array2::from_shape_vec((3, 2), vec![0.0, 0.0, 5.0, 0.0, 0.0, 5.0]).unwrap();
    assert_eq!(
        ft.predict(&q).unwrap(),
        array![0usize, 1, 2],
        "predict (decision path) must match the cluster classes"
    );

    let p = ft.predict_proba(&q).unwrap();
    for (s, &expect) in oracle_argmax.iter().enumerate() {
        let mut best = 0usize;
        for c in 1..3 {
            if p[[s, c]] > p[[s, best]] {
                best = c;
            }
        }
        assert_eq!(
            best, expect,
            "predict_proba row {s} argmax={best} != sklearn cluster class {expect}; \
             ferrolearn proba={:?} (uniform 1/3 means probA_/probB_ degenerated to 0; \
             sklearn argmax is {expect} for every random_state, svm.cpp:2107-2203)",
            (p[[s, 0]], p[[s, 1]], p[[s, 2]])
        );
    }
}

/// Divergence: `FittedNuSVC::prob_a`/`prob_b` (the per-ovo-pair Platt sigmoid
/// `(A, B)`, sklearn's `probA_`/`probB_`) degenerate to `0` for a small
/// per-pair sample count, where sklearn fits a non-degenerate sigmoid.
///
/// Input: a binary 6-sample linear-separable pair (3/class), `nu=0.5` — the size
/// of ONE ovo pair of the 3-class set above.
///
/// sklearn (live 1.5.2): `probA_ ~ -1.43` (and `predict_proba([[5,0]]) ~
/// [0.20, 0.80]`, oriented toward class 1) for random_state 0/1/42; the
/// magnitude is stable, never 0. ferrolearn: `probA_ = probB_ = 0` (the
/// contiguous 5-fold split makes the training folds single-class, so the
/// out-of-fold decisions are perfectly-separable constants and `sigmoid_train`
/// returns `(0,0)`).
///
/// Invariant (sklearn contract): a separable binary pair yields a sigmoid with
/// `|A| > 0` (a real slope), so `predict_proba` of an in-cluster query exceeds
/// 0.5 for that class. Asserted as `|probA_| > 0.01` AND
/// `P(class1 | [5,0]) > 0.5`; both are sklearn invariants, not copied values.
/// Tracking: #2245
#[test]
#[ignore = "divergence: NuSVC probA_/probB_ degenerate to 0 on a 6-sample separable pair (contiguous folds single-class); tracking #2245"]
fn divergence_nusvc_prob_a_degenerate_small_pair() {
    let x = Array2::from_shape_vec(
        (6, 2),
        vec![0.0, 0.0, 0.5, 0.0, 0.0, 0.5, 5.0, 0.0, 5.5, 0.0, 5.0, 0.5],
    )
    .unwrap();
    let y = array![0usize, 0, 0, 1, 1, 1];

    let ft = NuSVC::<f64, LinearKernel>::new(LinearKernel)
        .with_nu(0.5)
        .with_tol(1e-6)
        .with_max_iter(200_000)
        .with_probability(true)
        .fit(&x, &y)
        .unwrap();

    let a = ft.prob_a();
    assert_eq!(a.len(), 1, "one ovo pair");
    assert!(
        a[0].abs() > 1e-2,
        "probA_={} is degenerate (sklearn fits |probA_| ~ 1.43, stable across \
         random_state; svm.cpp:2107-2203 svm_binary_svc_probability)",
        a[0]
    );

    // predict_proba of the class-1 query must favor class 1 (P > 0.5).
    let q = Array2::from_shape_vec((1, 2), vec![5.0, 0.0]).unwrap();
    let p = ft.predict_proba(&q).unwrap();
    assert!(
        p[[0, 1]] > 0.5,
        "P(class1 | [5,0]) = {} must exceed 0.5 (sklearn ~ 0.80); \
         a uniform 0.5 means the sigmoid degenerated",
        p[[0, 1]]
    );
}
