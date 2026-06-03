//! Divergence pins + value-contract guards for `ComplementNB` /
//! `FittedComplementNB` (`ferrolearn-bayes/src/complement.rs`) against the LIVE
//! scikit-learn 1.5.2 oracle (`from sklearn.naive_bayes import ComplementNB`,
//! mirroring `sklearn/naive_bayes.py` `ComplementNB` + `_BaseDiscreteNB`).
//!
//! Every expected value below is a LIVE `sklearn` 1.5.2 oracle value (computed
//! via `python3 -c "..."` run from `/tmp`, quoted above each block) — NEVER
//! copied from the ferrolearn side (goal.md R-CHAR-3).
//!
//! Mirrors `sklearn/naive_bayes.py`:
//! `_BaseDiscreteNB._parameter_constraints` `alpha: [Interval(Real, 0, None,
//! closed="left"), "array-like"]` (naive_bayes.py:530), inherited by
//! `ComplementNB._parameter_constraints` (naive_bayes.py:1000-1003) — the
//! `>= 0` HARD reject enforced at `fit` by `_validate_params`;
//! `_BaseDiscreteNB._check_alpha` floor 1e-10 unless `force_alpha`
//! (naive_bayes.py:604-626); `ComplementNB._update_feature_log_prob` —
//! `comp_count = feature_all_ + alpha - feature_count_`,
//! `logged = log(comp_count / comp_count.sum(axis=1, keepdims=True))`,
//! `norm? logged/logged.sum : -logged` (naive_bayes.py:1032-1042);
//! `_update_class_log_prior` LENGTH-only check then `log(class_prior)`
//! (naive_bayes.py:580-602, NO sum/non-neg check for discrete NB);
//! `ComplementNB._joint_log_likelihood = X @ feature_log_prob_.T` +
//! single-class `class_log_prior_` add (naive_bayes.py:1044-1049);
//! `_count` → `check_non_negative(X, "ComplementNB (input X)")`
//! (naive_bayes.py:1027).
//!
//! Design doc: `.design/bayes/complement.md` (commit adc9b9d6).
//!
//! Test taxonomy — the RED pin FAILS now and goes green when the generator
//! lands the fix; the GREEN guards PASS now and protect the parts already
//! correct:
//! `divergence_complement_negative_alpha_rejected` (RED, #918);
//! `green_complement_predict_value_norm_false`,
//! `green_complement_predict_value_norm_true`,
//! `green_complement_class_prior_length_only`,
//! `green_complement_score_accuracy`,
//! `green_complement_negative_features_rejected` (GREEN).
//!
//! Documented-not-pinned (missing surface / benign — no forced test):
//!
//! single-class jll `class_log_prior_` add (BENIGN): sklearn's
//! `ComplementNB._joint_log_likelihood` ADDS `class_log_prior_` ONLY when
//! `len(classes_) == 1` (naive_bayes.py:1047-1048); ferrolearn's `X @
//! weights.T` does NOT add a prior. This is NOT observable: with a single
//! class `class_log_prior_ = [0.0]` (`log(class_count_/class_count_.sum()) =
//! log(1) = 0`), so the add is literally `+0.0`; and even were it nonzero,
//! softmax over one column is always `[[1.0]]`. The design doc (AC-8) confirmed
//! both sides COINCIDE (`predict_proba = [[1.0]]`, `predict = [0]`). Benign; not
//! pinned. (The in-tree `test_complement_nb_single_class` already covers the
//! label path.)
//!
//! sample_weight on fit/partial_fit + partial_fit `classes=` (#915): sklearn
//! `fit(X, y, sample_weight=None)` (naive_bayes.py:712) weights the binarized
//! `Y` so `feature_count_ = Y.T@X` / `class_count_ = Y.sum(axis=0)` /
//! `feature_all_ = feature_count_.sum(axis=0)` become weighted counts
//! (naive_bayes.py:1025-1030); the shared `partial_fit` (naive_bayes.py:628-709)
//! takes the full `classes` list on the first call and binarizes against it.
//! ferrolearn's `Fit` trait is `fn fit(&self, x, y)` — no `sample_weight`
//! parameter on `fit` or `partial_fit`, and `partial_fit` has no `classes`
//! argument (it loops only over the already-fitted `self.classes`, silently
//! dropping a brand-new later-chunk label). The same-classes incremental path
//! IS value-correct (in-tree `test_complement_nb_partial_fit`). Missing
//! surface; documented, not pinned.
//!
//! fitted accessors feature_log_prob_/feature_all_/feature_count_/
//! class_count_/class_log_prior_ + the PyO3 surface (#916): sklearn exposes
//! these (naive_bayes.py:937-970; `hasattr(fitted,'coef_') == False`);
//! `FittedComplementNB` keeps `weights`/`feature_counts`/`class_counts` private
//! with no accessor (only `classes()` via `HasClasses`; no stored
//! `feature_all_`/`class_log_prior_`), and `_RsComplementNB`
//! (`ferrolearn-python/src/extras.rs`, the `py_classifier!` macro) exposes only
//! `new(alpha, fit_prior, norm)` + `fit` + `predict` — no `class_prior`/
//! `force_alpha` kwargs, no `predict_proba`/`predict_log_proba`/
//! `predict_joint_log_proba`/`score`/`partial_fit` (which the library HAS), no
//! fitted-attr getters. This also subsumes the negative-feature MESSAGE/TYPE
//! sub-item (ferrolearn `InvalidParameter` vs sklearn `ValueError`) and the
//! `class_prior` wrong-length TYPE sub-item. Missing surface; a pytest
//! divergence belongs in ferrolearn-python, not in this crate.
//!
//! ferray substrate (#917): `complement.rs` imports `ndarray::{Array1,
//! Array2}` + `num_traits::{Float, FromPrimitive, ToPrimitive}`, not
//! `ferray-core`. Substrate migration; no observable-value pin.

use ferrolearn_bayes::ComplementNB;
use ferrolearn_core::{Fit, Predict};
use ndarray::{Array1, Array2, array};

/// The shared count fixture from `.design/bayes/complement.md`:
/// `X = [[5,1,0],[4,2,0],[6,0,1],[0,1,5],[1,0,4],[0,2,6]]`, `y = [0,0,0,1,1,1]`.
fn count_fixture() -> (Array2<f64>, Array1<usize>) {
    let x = Array2::from_shape_vec(
        (6, 3),
        vec![
            5.0, 1.0, 0.0, // class 0
            4.0, 2.0, 0.0, // class 0
            6.0, 0.0, 1.0, // class 0
            0.0, 1.0, 5.0, // class 1
            1.0, 0.0, 4.0, // class 1
            0.0, 2.0, 6.0, // class 1
        ],
    )
    .unwrap();
    let y = array![0usize, 0, 0, 1, 1, 1];
    (x, y)
}

/// The query `q = [[3,1,1],[0,1,4]]`.
fn query() -> Array2<f64> {
    Array2::from_shape_vec((2, 3), vec![3.0, 1.0, 1.0, 0.0, 1.0, 4.0]).unwrap()
}

// ===========================================================================
// RED #918 — alpha >= 0 is a HARD reject at fit.
//
// sklearn `_BaseDiscreteNB._parameter_constraints` (naive_bayes.py:530):
//     "alpha": [Interval(Real, 0, None, closed="left"), "array-like"],
// inherited by `ComplementNB._parameter_constraints` (naive_bayes.py:1000-1003,
// `**_BaseDiscreteNB._parameter_constraints`) → alpha must be >= 0, enforced
// at `fit` by `_validate_params`. `ComplementNB(alpha=-0.5).fit(X,y)` raises
// `InvalidParameterError` (a ValueError subclass). This is DISTINCT from
// `_check_alpha`'s 1e-10 floor (naive_bayes.py:604-626), which only fires for
// alpha < 1e-10 when force_alpha=False.
//
// ferrolearn `fn fit` (complement.rs) computes
//     let alpha = crate::clamp_alpha(self.alpha, self.force_alpha);
// where `clamp_alpha` (= base::check_alpha) only FLOORS under
// force_alpha=false. With the default force_alpha=true, `clamp_alpha(-0.5,
// true)` returns -0.5 unchanged, so `fit` proceeds and computes
// `-log((complement_count - 0.5)/denom)` — negative-smoothed garbage / NaN,
// NO error.
//
// Live oracle (sklearn 1.5.2, run from /tmp):
//   python3 -c "from sklearn.naive_bayes import ComplementNB; import numpy as np
//   try:
//       ComplementNB(alpha=-0.5).fit(np.array([[5.,1,0],[4,2,0],[0,1,5],[1,0,4]]), np.array([0,0,1,1]))
//       print('no error')
//   except Exception as e:
//       print(type(e).__name__)"
//   ->  InvalidParameterError
//       (full message: "The 'alpha' parameter of ComplementNB must be a float
//        in the range [0.0, inf) or an array-like. Got -0.5 instead.")
//
// We pin the OBSERVABLE: `fit` must return `Err` (sklearn raises). ferrolearn
// currently returns `Ok`: FAILS now. Minimally fixable in `complement.rs`
// `fn fit` — reject `alpha < 0` (InvalidParameter) before/around clamp_alpha.
// ===========================================================================

/// Divergence: ferrolearn's `ComplementNB::fit` accepts a negative `alpha`,
/// whereas `sklearn/naive_bayes.py:530`
/// (`alpha: Interval(Real, 0, None, closed="left")`, inherited at
/// naive_bayes.py:1000-1003) makes `alpha < 0` a HARD reject at `fit`
/// (`InvalidParameterError`). `ComplementNB(alpha=-0.5).fit` raises in sklearn;
/// ferrolearn returns `Ok` because `clamp_alpha(-0.5, true)` passes -0.5
/// through unchanged (the floor only fires under force_alpha=false).
/// Tracking: #918
#[test]
fn divergence_complement_negative_alpha_rejected() {
    let (x, y) = count_fixture();

    // alpha = -0.5 — sklearn raises InvalidParameterError (oracle above).
    let model = ComplementNB::<f64>::new().with_alpha(-0.5);
    let result = model.fit(&x, &y);

    assert!(
        result.is_err(),
        "fit with alpha=-0.5 should error (sklearn: InvalidParameterError, \
         'The 'alpha' parameter of ComplementNB must be a float in the range \
         [0.0, inf) ...'), got Ok (#918)"
    );
}

// ===========================================================================
// GREEN — predict_proba / predict_joint_log_proba / predict VALUE (norm=False)
// on the count fixture. Mirrors `ComplementNB._update_feature_log_prob`
// norm=False branch (naive_bayes.py:1032-1042, `feature_log_prob_ = -logged`)
// feeding `_joint_log_likelihood = X @ feature_log_prob_.T`
// (naive_bayes.py:1046) through the _BaseNB pipeline.
//
// Live oracle (sklearn 1.5.2, run from /tmp):
//   python3 -c "from sklearn.naive_bayes import ComplementNB; import numpy as np; \
//     X=np.array([[5.,1,0],[4,2,0],[6,0,1],[0,1,5],[1,0,4],[0,2,6]]); \
//     y=np.array([0,0,0,1,1,1]); q=np.array([[3.,1,1],[0,1,4]]); \
//     m=ComplementNB().fit(X,y); \
//     print(m.predict_proba(q).tolist()); \
//     print(m.predict_joint_log_proba(q).tolist()); \
//     print(m.predict(q).tolist())"
//   -> predict_proba             [[0.9846153846153846, 0.015384615384615375],
//                                 [0.0002440810349035878, 0.9997559189650967]]
//   -> predict_joint_log_proba   [[9.216887641752072, 5.058004558392399],
//                                 [2.9785630167125636, 11.296329183431908]]
//   -> predict                   [0, 1]
// ===========================================================================

/// Guard: `predict_proba` / `predict_joint_log_proba` / `predict` match sklearn
/// `ComplementNB().fit(X,y)` to ~1e-12 (the `_update_feature_log_prob`
/// norm=False branch + `_joint_log_likelihood` VALUE, naive_bayes.py:1032-1046).
#[test]
fn green_complement_predict_value_norm_false() {
    let (x, y) = count_fixture();
    let q = query();
    let fitted = ComplementNB::<f64>::new().fit(&x, &y).unwrap();

    let proba = fitted.predict_proba(&q).unwrap();
    let jll = fitted.predict_joint_log_proba(&q).unwrap();
    let preds: Array1<usize> = fitted.predict(&q).unwrap();

    // sklearn 1.5.2 oracle (quoted above) — NOT copied from ferrolearn.
    const SK_PROBA: [[f64; 2]; 2] = [
        [0.9846153846153846, 0.015384615384615375],
        [0.0002440810349035878, 0.9997559189650967],
    ];
    const SK_JLL: [[f64; 2]; 2] = [
        [9.216887641752072, 5.058004558392399],
        [2.9785630167125636, 11.296329183431908],
    ];
    const TOL: f64 = 1e-12;

    for i in 0..2 {
        for c in 0..2 {
            assert!(
                (proba[[i, c]] - SK_PROBA[i][c]).abs() <= TOL,
                "predict_proba[{i}][{c}]: ferrolearn {} vs sklearn {}",
                proba[[i, c]],
                SK_PROBA[i][c]
            );
            assert!(
                (jll[[i, c]] - SK_JLL[i][c]).abs() <= TOL,
                "predict_joint_log_proba[{i}][{c}]: ferrolearn {} vs sklearn {}",
                jll[[i, c]],
                SK_JLL[i][c]
            );
        }
    }
    // sklearn 1.5.2 oracle: predict(q) == [0, 1].
    assert_eq!(preds, array![0usize, 1]);
}

// ===========================================================================
// GREEN — predict_proba / predict VALUE (norm=True). Mirrors
// `ComplementNB._update_feature_log_prob` norm=True branch
// (naive_bayes.py:1037-1039, `feature_log_prob_ = logged / logged.sum(axis=1,
// keepdims=True)`). ferrolearn stores `weights = -logged` then divides each
// row by its sum: `(-logged)/sum(-logged)`, the algebraic identity of
// `logged/sum(logged)` (the two minus signs cancel). This LOCKS that
// equivalence.
//
// Live oracle (sklearn 1.5.2, run from /tmp):
//   python3 -c "from sklearn.naive_bayes import ComplementNB; import numpy as np; \
//     X=np.array([[5.,1,0],[4,2,0],[6,0,1],[0,1,5],[1,0,4],[0,2,6]]); \
//     y=np.array([0,0,0,1,1,1]); q=np.array([[3.,1,1],[0,1,4]]); \
//     mn=ComplementNB(norm=True).fit(X,y); \
//     print(mn.predict_proba(q).tolist()); print(mn.predict(q).tolist())"
//   -> predict_proba   [[0.7192390704948571, 0.2807609295051429],
//                       [0.13223037910101987, 0.8677696208989801]]
//   -> predict         [0, 1]
// ===========================================================================

/// Guard: `with_norm(true)` `predict_proba` / `predict` match sklearn
/// `ComplementNB(norm=True).fit(X,y)` to ~1e-12 — locking the
/// `(-logged)/sum(-logged) == logged/sum(logged)` norm equivalence
/// (naive_bayes.py:1037-1039).
#[test]
fn green_complement_predict_value_norm_true() {
    let (x, y) = count_fixture();
    let q = query();
    let fitted = ComplementNB::<f64>::new()
        .with_norm(true)
        .fit(&x, &y)
        .unwrap();

    let proba = fitted.predict_proba(&q).unwrap();
    let preds: Array1<usize> = fitted.predict(&q).unwrap();

    // sklearn 1.5.2 oracle (quoted above) — NOT copied from ferrolearn.
    const SK_PROBA: [[f64; 2]; 2] = [
        [0.7192390704948571, 0.2807609295051429],
        [0.13223037910101987, 0.8677696208989801],
    ];
    const TOL: f64 = 1e-12;

    for i in 0..2 {
        for c in 0..2 {
            assert!(
                (proba[[i, c]] - SK_PROBA[i][c]).abs() <= TOL,
                "norm=True predict_proba[{i}][{c}]: ferrolearn {} vs sklearn {}",
                proba[[i, c]],
                SK_PROBA[i][c]
            );
        }
    }
    // sklearn 1.5.2 oracle: predict(q) == [0, 1].
    assert_eq!(preds, array![0usize, 1]);
}

// ===========================================================================
// GREEN — class_prior LENGTH-only validation (REQ-4, the MATCH, NOT a
// divergence). sklearn `_update_class_log_prior` (naive_bayes.py:589-591)
// checks ONLY length then `class_log_prior_ = np.log(class_prior)` — discrete
// NB has NO sum-to-1 and NO non-negativity check. So `class_prior=[0.5,0.3]`
// (sum 0.8) is ACCEPTED on the sklearn side. ferrolearn `fn fit` checks ONLY
// `priors.len() != n_classes` — must ALSO accept.
//
// For ComplementNB `class_prior` is "Not used" in multi-class predict
// (naive_bayes.py:929) — only the accept/reject DECISION is observable, so we
// assert `.is_ok()` (not a downstream value).
//
// Live oracle (sklearn 1.5.2, run from /tmp):
//   python3 -c "from sklearn.naive_bayes import ComplementNB; import numpy as np; \
//     X=np.array([[5.,1,0],[4,2,0],[6,0,1],[0,1,5],[1,0,4],[0,2,6]]); \
//     y=np.array([0,0,0,1,1,1]); \
//     ComplementNB(class_prior=[0.5,0.3]).fit(X,y); print('ok')"
//   ->  ok   (sum 0.8 ACCEPTED; class_log_prior_ = log([0.5,0.3]); NO error)
// ===========================================================================

/// Guard: ferrolearn `with_class_prior([0.5,0.3]).fit` SUCCEEDS, matching
/// sklearn `_update_class_log_prior` LENGTH-only validation
/// (naive_bayes.py:589-591) — discrete NB has NO sum/non-neg check. The
/// length-validation decision is the only observable (`class_prior` is "Not
/// used" in ComplementNB multi-class predict, naive_bayes.py:929).
#[test]
fn green_complement_class_prior_length_only() {
    let (x, y) = count_fixture();

    // sum 0.8 — sklearn accepts (oracle above). ferrolearn must too.
    let model = ComplementNB::<f64>::new().with_class_prior(vec![0.5, 0.3]);
    let result = model.fit(&x, &y);

    assert!(
        result.is_ok(),
        "fit with class_prior=[0.5,0.3] (sum 0.8) should SUCCEED \
         (discrete NB has no sum check; sklearn ACCEPTS, class_log_prior_ = \
         log([0.5,0.3])), got Err"
    );
}

// ===========================================================================
// GREEN — score (mean accuracy) on the separable count fixture
// (`ClassifierMixin.score` analog).
//
// Live oracle (sklearn 1.5.2, run from /tmp):
//   python3 -c "from sklearn.naive_bayes import ComplementNB; import numpy as np; \
//     X=np.array([[5.,1,0],[4,2,0],[6,0,1],[0,1,5],[1,0,4],[0,2,6]]); \
//     y=np.array([0,0,0,1,1,1]); print(ComplementNB().fit(X,y).score(X,y))"
//   ->  1.0
// ===========================================================================

/// Guard: `score(X,y)` matches sklearn `ComplementNB().fit(X,y).score(X,y)` =
/// `1.0` on this separable fixture (`ClassifierMixin.score` analog).
#[test]
fn green_complement_score_accuracy() {
    let (x, y) = count_fixture();
    let fitted = ComplementNB::<f64>::new().fit(&x, &y).unwrap();

    let acc = fitted.score(&x, &y).unwrap();

    // sklearn 1.5.2 oracle: 1.0.
    const SK_SCORE: f64 = 1.0;
    assert!(
        (acc - SK_SCORE).abs() <= 1e-12,
        "score = {acc}, want {SK_SCORE}"
    );
}

// ===========================================================================
// GREEN — negative features rejected by BOTH (the reject DECISION matches;
// the exact ValueError message/type is documented NOT-pinned in the header).
//
// sklearn `ComplementNB._count` (naive_bayes.py:1027):
//     check_non_negative(X, "ComplementNB (input X)")
// → `ValueError("Negative values in data passed to ComplementNB (input X)")`.
// ferrolearn `fn fit` rejects with `InvalidParameter { name: "X" }`.
//
// Live oracle (sklearn 1.5.2, run from /tmp):
//   python3 -c "from sklearn.naive_bayes import ComplementNB; import numpy as np; \
//     X=np.array([[1.,2],[-0.5,3],[2,1],[0,4]]); y=np.array([0,0,1,1]); \
//     ComplementNB().fit(X,y)"
//   ->  ValueError: Negative values in data passed to ComplementNB (input X)
// ===========================================================================

/// Guard: ferrolearn `ComplementNB().fit(X_with_neg, y)` returns `Err`,
/// matching sklearn's `check_non_negative` reject (naive_bayes.py:1027). Both
/// REJECT; the exact message/type divergence is documented NOT-pinned.
#[test]
fn green_complement_negative_features_rejected() {
    let x = Array2::from_shape_vec((4, 2), vec![1.0, 2.0, -0.5, 3.0, 2.0, 1.0, 0.0, 4.0]).unwrap();
    let y = array![0usize, 0, 1, 1];

    let result = ComplementNB::<f64>::new().fit(&x, &y);

    assert!(
        result.is_err(),
        "fit with a negative feature should error (sklearn: ValueError \
         'Negative values in data passed to ComplementNB (input X)'), got Ok"
    );
}
