//! Divergence pins + value-contract guards for `GaussianNB` / `FittedGaussianNB`
//! (`ferrolearn-bayes/src/gaussian.rs`) against the LIVE scikit-learn 1.5.2
//! oracle (`from sklearn.naive_bayes import GaussianNB`, mirroring
//! `sklearn/naive_bayes.py`).
//!
//! Every expected value below is a LIVE `sklearn` 1.5.2 oracle value (computed
//! via `python3 -c "..."` run from `/tmp`, quoted above each block) — NEVER
//! copied from the ferrolearn side (goal.md R-CHAR-3).
//!
//! Mirrors `sklearn/naive_bayes.py`:
//! `GaussianNB._partial_fit` epsilon_ `= var_smoothing * np.var(X, axis=0).max()`
//! (naive_bayes.py:431); `var_ += epsilon_` (naive_bayes.py:497);
//! `GaussianNB._joint_log_likelihood` (naive_bayes.py:506-515); priors
//! validation ValueErrors (naive_bayes.py:448-455).
//!
//! Design doc: `.design/bayes/gaussian.md` (commit 68afd3a0).
//!
//! Test taxonomy — RED pins FAIL now and go green when the generator lands the
//! fix; GREEN guards PASS now and protect the parts that are already correct:
//! `divergence_gaussian_epsilon_global_var_no_floor` (RED, #891);
//! `divergence_gaussian_priors_sum_not_one_rejected` (RED, #893);
//! `green_gaussian_predict_labels`, `green_gaussian_predict_proba_sums_to_one`,
//! `green_gaussian_score_accuracy` (GREEN).
//!
//! Documented-not-pinned (missing surface / multi-file — kdtree/knn precedent):
//!
//! sample_weight on fit/partial_fit (#894): sklearn `fit(X,y,sample_weight)`
//! supports weighted theta_/var_/class_count_ via `_update_mean_variance`
//! (naive_bayes.py:319-320); ferrolearn's `Fit` trait is `fn fit(&self,x,y)` —
//! no `sample_weight` parameter. Missing surface; no forced test.
//!
//! partial_fit epsilon-once (#895): sklearn fixes `epsilon_` at the first fit
//! and does the subtract-before / re-add-after dance (naive_bayes.py:465 /
//! :497); ferrolearn `partial_fit` recomputes `epsilon` from the current `sigma`
//! each call. Compounds #891; deferred until #891 lands.
//!
//! fitted accessors theta_/var_/epsilon_/class_count_/class_prior_ (#896):
//! sklearn exposes these (naive_bayes.py:171-202); `FittedGaussianNB` keeps them
//! private with no accessor (only `classes()` via `HasClasses`). Missing
//! surface; cannot be reached from Rust to assert a value.
//!
//! PyO3 surface (#897): `_RsGaussianNB` exposes only fit/predict/predict_proba/
//! classes_ — no priors kwarg, no theta_/var_/epsilon_ getters, no
//! predict_log_proba/score/partial_fit. Lives in ferrolearn-python; a pytest
//! divergence belongs there, not in this crate.
//!
//! ferray substrate (#898): `gaussian.rs` imports `ndarray` + `num-traits`, not
//! `ferray-core`. Substrate migration; no observable-value pin.

use ferrolearn_bayes::GaussianNB;
use ferrolearn_core::{Fit, Predict};
use ndarray::{Array1, Array2, array};

// ===========================================================================
// RED #891 — epsilon_ global per-feature variance, NO 1.0 floor.
//
// sklearn `GaussianNB._partial_fit` (naive_bayes.py:431):
//     self.epsilon_ = self.var_smoothing * np.var(X, axis=0).max()
// then (naive_bayes.py:497):  self.var_[:, :] += self.epsilon_
// This is the GLOBAL per-feature variance over ALL X (across all classes),
// reduced by .max() over features, times var_smoothing — NO floor.
//
// ferrolearn `fn fit` (gaussian.rs ~:195-198):
//     let max_var = sigma.iter().fold(...max per-class sigma entry...);
//     let epsilon = self.var_smoothing * max_var.max(F::one());
// Two bugs: (a) max over the PER-CLASS variance matrix, not the global
// per-feature variance, and (b) a spurious `.max(1.0)` floor.
//
// `FittedGaussianNB` exposes no public `epsilon_` accessor, so we pin the
// OBSERVABLE consequence: `predict_joint_log_proba` (which consumes the
// smoothed `var_`). On this fixture sklearn `epsilon_ = 6.416666666666667e-9`
// (= 1e-9 * np.var(X,axis=0).max(), where np.var(X,axis=0)=[6.4167,6.3367]);
// ferrolearn's per-class max variance is 0.16667 (< 1.0), floored to 1.0, so it
// uses epsilon = 1e-9 — a different smoothing constant. The jll shifts by ~1e-8
// beyond the 1e-9 parity bar.
//
// Live oracle (sklearn 1.5.2, run from /tmp):
//   python3 -c "import numpy as np; from sklearn.naive_bayes import GaussianNB; \
//     X=np.array([[1.,2.],[1.5,1.8],[2.,2.5],[6.,7.],[6.5,6.8],[7.,7.5]]); \
//     y=np.array([0,0,0,1,1,1]); m=GaussianNB().fit(X,y); \
//     q=np.array([[1.2,2.1],[6.6,7.1]]); \
//     jll=m.predict_joint_log_proba(q); print(repr(jll[0,0]), repr(jll[1,1]))"
//   ->  np.float64(-0.6823015899121332) np.float64(-0.44230159915213274)
//
// (Cross-check of epsilon_ itself:
//   python3 -c "import numpy as np; from sklearn.naive_bayes import GaussianNB; \
//     X=np.array([[1.,2.],[1.5,1.8],[2.,2.5],[6.,7.],[6.5,6.8],[7.,7.5]]); \
//     y=np.array([0,0,0,1,1,1]); m=GaussianNB().fit(X,y); print(repr(m.epsilon_))"
//   ->  np.float64(6.416666666666667e-09) )
//
// ferrolearn currently returns jll[0][0] = -0.6823015511871344 (~3.9e-8 off),
// jll[1][1] = -0.44230155262713433 (~4.7e-8 off): FAILS the 1e-9 bar below.
// After the fixer corrects epsilon_ this goes green.
// ===========================================================================

/// Divergence: ferrolearn's `GaussianNB::fit` smoothing diverges from
/// `sklearn/naive_bayes.py:431` (`epsilon_ = var_smoothing * np.var(X,
/// axis=0).max()`, no floor) for the fixture below. sklearn
/// `predict_joint_log_proba(q)[0][0] = -0.6823015899121332`; ferrolearn returns
/// `-0.6823015511871344` (~3.9e-8 off) because it floors a per-class max
/// variance at 1.0 instead of taking the global per-feature variance.
/// Tracking: #891
#[test]
fn divergence_gaussian_epsilon_global_var_no_floor() {
    let x = Array2::from_shape_vec(
        (6, 2),
        vec![
            1.0, 2.0, 1.5, 1.8, 2.0, 2.5, // class 0
            6.0, 7.0, 6.5, 6.8, 7.0, 7.5, // class 1
        ],
    )
    .unwrap();
    let y = array![0usize, 0, 0, 1, 1, 1];
    let fitted = GaussianNB::<f64>::new().fit(&x, &y).unwrap();

    let q = Array2::from_shape_vec((2, 2), vec![1.2, 2.1, 6.6, 7.1]).unwrap();
    let jll = fitted.predict_joint_log_proba(&q).unwrap();

    // sklearn 1.5.2 oracle values (quoted above) — NOT copied from ferrolearn.
    const SK_JLL_00: f64 = -0.6823015899121332;
    const SK_JLL_11: f64 = -0.44230159915213274;
    const TOL: f64 = 1e-9;

    assert!(
        (jll[[0, 0]] - SK_JLL_00).abs() <= TOL,
        "predict_joint_log_proba[0][0]: ferrolearn {} vs sklearn {} (epsilon_ #891)",
        jll[[0, 0]],
        SK_JLL_00
    );
    assert!(
        (jll[[1, 1]] - SK_JLL_11).abs() <= TOL,
        "predict_joint_log_proba[1][1]: ferrolearn {} vs sklearn {} (epsilon_ #891)",
        jll[[1, 1]],
        SK_JLL_11
    );
}

// ===========================================================================
// RED #893 — priors validation: sum ~= 1 (and non-negativity).
//
// sklearn `GaussianNB._partial_fit` (naive_bayes.py:451-452):
//     if not np.isclose(priors.sum(), 1.0):
//         raise ValueError("The sum of the priors should be 1.")
// and (naive_bayes.py:454-455):
//     if (priors < 0).any():
//         raise ValueError("Priors must be non-negative.")
//
// ferrolearn `fn fit` (gaussian.rs ~:202-216) validates ONLY length
// (`priors.len() != n_classes`) and then `log_prior[ci] = priors[ci].ln()` —
// no sum-to-1 check, no non-negativity check. A `priors=[0.5,0.3]` (sum 0.8) is
// accepted silently.
//
// Live oracle (sklearn 1.5.2, run from /tmp):
//   python3 -c "import numpy as np; from sklearn.naive_bayes import GaussianNB; \
//     X=np.array([[1.,2.],[1.5,1.8],[2.,2.5],[6.,7.],[6.5,6.8],[7.,7.5]]); \
//     y=np.array([0,0,0,1,1,1]); \
//     GaussianNB(priors=np.array([0.5,0.3])).fit(X,y)"
//   ->  ValueError: The sum of the priors should be 1.
//
// We pin the OBSERVABLE: `fit` must return `Err` (sklearn raises). ferrolearn
// currently returns `Ok` (only length is checked): FAILS now. This is a clean
// single-file fix inside `gaussian.rs::fit` (add the sum/non-neg checks), so it
// is pinned rather than merely documented.
// ===========================================================================

/// Divergence: ferrolearn's `GaussianNB::fit` accepts class priors that do not
/// sum to 1, whereas `sklearn/naive_bayes.py:451-452` raises
/// `ValueError("The sum of the priors should be 1.")`. With
/// `class_prior=[0.5,0.3]` (sum 0.8) sklearn errors; ferrolearn returns `Ok`.
/// Tracking: #893
#[test]
fn divergence_gaussian_priors_sum_not_one_rejected() {
    let x = Array2::from_shape_vec(
        (6, 2),
        vec![
            1.0, 2.0, 1.5, 1.8, 2.0, 2.5, // class 0
            6.0, 7.0, 6.5, 6.8, 7.0, 7.5, // class 1
        ],
    )
    .unwrap();
    let y = array![0usize, 0, 0, 1, 1, 1];

    // priors sum to 0.8, not 1.0 — sklearn raises ValueError (oracle above).
    let model = GaussianNB::<f64>::new().with_class_prior(vec![0.5, 0.3]);
    let result = model.fit(&x, &y);

    assert!(
        result.is_err(),
        "fit with priors=[0.5,0.3] (sum 0.8) should error (sklearn: \
         'The sum of the priors should be 1.'), got Ok (#893)"
    );
}

// ===========================================================================
// GREEN — predict LABELS on a well-separated fixture (argmax robust to the
// epsilon_ bug; correct even pre-fix).
//
// Live oracle (sklearn 1.5.2, run from /tmp):
//   python3 -c "import numpy as np; from sklearn.naive_bayes import GaussianNB; \
//     X=np.array([[1.,2.],[1.5,1.8],[2.,2.5],[6.,7.],[6.5,6.8],[7.,7.5]]); \
//     y=np.array([0,0,0,1,1,1]); m=GaussianNB().fit(X,y); \
//     q=np.array([[1.2,2.1],[6.6,7.1]]); print(m.predict(q).tolist())"
//   ->  [0, 1]
// ===========================================================================

/// Guard: `predict` labels match sklearn `GaussianNB().fit(X,y).predict(q)` =
/// `[0,1]` (naive_bayes.py:506-515 via `_BaseNB.predict`). Argmax is robust to
/// the `epsilon_` shift, so this is correct pre-fix.
#[test]
fn green_gaussian_predict_labels() {
    let x = Array2::from_shape_vec(
        (6, 2),
        vec![1.0, 2.0, 1.5, 1.8, 2.0, 2.5, 6.0, 7.0, 6.5, 6.8, 7.0, 7.5],
    )
    .unwrap();
    let y = array![0usize, 0, 0, 1, 1, 1];
    let fitted = GaussianNB::<f64>::new().fit(&x, &y).unwrap();

    let q = Array2::from_shape_vec((2, 2), vec![1.2, 2.1, 6.6, 7.1]).unwrap();
    let preds: Array1<usize> = fitted.predict(&q).unwrap();

    // sklearn 1.5.2 oracle: [0, 1].
    assert_eq!(preds, array![0usize, 1]);
}

// ===========================================================================
// GREEN — predict_proba rows sum to 1 (a structural contract independent of the
// epsilon_ value; sklearn normalizes per row). This guards the BaseNB
// normalization delegation. (An epsilon-independent absolute proba value is not
// achievable here: even on a large-variance fixture ferrolearn's per-class
// floored epsilon differs from sklearn's global epsilon, so absolute proba still
// diverges. Per the task's honest fallback, we guard the sum-to-one invariant.)
//
// Live oracle (sklearn 1.5.2, run from /tmp):
//   python3 -c "import numpy as np; from sklearn.naive_bayes import GaussianNB; \
//     X=np.array([[1.,2.],[1.5,1.8],[2.,2.5],[6.,7.],[6.5,6.8],[7.,7.5]]); \
//     y=np.array([0,0,0,1,1,1]); m=GaussianNB().fit(X,y); \
//     q=np.array([[1.2,2.1],[6.6,7.1]]); \
//     print([float(r.sum()) for r in m.predict_proba(q)])"
//   ->  [1.0, 1.0]
// ===========================================================================

/// Guard: `predict_proba` rows sum to 1, matching sklearn (each row is a
/// normalized posterior, `_BaseNB.predict_proba`, naive_bayes.py:144). This
/// invariant holds regardless of the `epsilon_` value.
#[test]
fn green_gaussian_predict_proba_sums_to_one() {
    let x = Array2::from_shape_vec(
        (6, 2),
        vec![1.0, 2.0, 1.5, 1.8, 2.0, 2.5, 6.0, 7.0, 6.5, 6.8, 7.0, 7.5],
    )
    .unwrap();
    let y = array![0usize, 0, 0, 1, 1, 1];
    let fitted = GaussianNB::<f64>::new().fit(&x, &y).unwrap();

    let q = Array2::from_shape_vec((2, 2), vec![1.2, 2.1, 6.6, 7.1]).unwrap();
    let proba = fitted.predict_proba(&q).unwrap();

    // sklearn 1.5.2 oracle: each row sums to 1.0.
    const SK_ROW_SUM: f64 = 1.0;
    const TOL: f64 = 1e-12;
    for i in 0..proba.nrows() {
        let s = proba.row(i).sum();
        assert!(
            (s - SK_ROW_SUM).abs() <= TOL,
            "predict_proba row {i} sum = {s}, want {SK_ROW_SUM}"
        );
    }
}

// ===========================================================================
// GREEN — score (mean accuracy) on a separable fixture.
//
// Live oracle (sklearn 1.5.2, run from /tmp):
//   python3 -c "import numpy as np; from sklearn.naive_bayes import GaussianNB; \
//     X=np.array([[1.,2.],[1.5,1.8],[2.,2.5],[6.,7.],[6.5,6.8],[7.,7.5]]); \
//     y=np.array([0,0,0,1,1,1]); m=GaussianNB().fit(X,y); print(m.score(X,y))"
//   ->  1.0
// ===========================================================================

/// Guard: `score(X,y)` matches sklearn `GaussianNB().fit(X,y).score(X,y)` =
/// `1.0` on this separable fixture (`ClassifierMixin.score` analog). Robust to
/// the `epsilon_` shift (labels are correct).
#[test]
fn green_gaussian_score_accuracy() {
    let x = Array2::from_shape_vec(
        (6, 2),
        vec![1.0, 2.0, 1.5, 1.8, 2.0, 2.5, 6.0, 7.0, 6.5, 6.8, 7.0, 7.5],
    )
    .unwrap();
    let y = array![0usize, 0, 0, 1, 1, 1];
    let fitted = GaussianNB::<f64>::new().fit(&x, &y).unwrap();

    let acc = fitted.score(&x, &y).unwrap();

    // sklearn 1.5.2 oracle: 1.0.
    const SK_SCORE: f64 = 1.0;
    assert!(
        (acc - SK_SCORE).abs() <= 1e-12,
        "score = {acc}, want {SK_SCORE}"
    );
}
