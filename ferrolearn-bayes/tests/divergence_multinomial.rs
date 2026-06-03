//! Divergence pins + value-contract guards for `MultinomialNB` /
//! `FittedMultinomialNB` (`ferrolearn-bayes/src/multinomial.rs`) against the
//! LIVE scikit-learn 1.5.2 oracle (`from sklearn.naive_bayes import
//! MultinomialNB`, mirroring `sklearn/naive_bayes.py` `MultinomialNB` +
//! `_BaseDiscreteNB`).
//!
//! Every expected value below is a LIVE `sklearn` 1.5.2 oracle value (computed
//! via `python3 -c "..."` run from `/tmp`, quoted above each block) — NEVER
//! copied from the ferrolearn side (goal.md R-CHAR-3).
//!
//! Mirrors `sklearn/naive_bayes.py`:
//! `MultinomialNB._parameter_constraints` `alpha: [Interval(Real, 0, None,
//! closed="left"), "array-like"]` (naive_bayes.py:530, the `>= 0` HARD reject
//! enforced at `fit` by `_validate_params`); `_BaseDiscreteNB._check_alpha`
//! floor 1e-10 unless `force_alpha` (naive_bayes.py:604-626);
//! `_update_feature_log_prob = log(fc+alpha) - log((fc+alpha).sum(axis=1))`
//! (naive_bayes.py:885-892); `_update_class_log_prior` LENGTH-only check then
//! `log(class_prior)` (naive_bayes.py:580-602, NO sum/non-neg check for
//! discrete NB); `_joint_log_likelihood = X @ feature_log_prob_.T +
//! class_log_prior_` (naive_bayes.py:894-896); `_count` →
//! `check_non_negative(X, "MultinomialNB (input X)")` (naive_bayes.py:881).
//!
//! Design doc: `.design/bayes/multinomial.md` (commit 303f552d).
//!
//! Test taxonomy — RED pins FAIL now and go green when the generator lands the
//! fix; GREEN guards PASS now and protect the parts that are already correct:
//! `divergence_multinomial_negative_alpha_rejected` (RED, #904);
//! `green_multinomial_predict_proba_log_proba_value`,
//! `green_multinomial_class_prior_length_only_accepts_non_unit_sum`,
//! `green_multinomial_score_accuracy`,
//! `green_multinomial_negative_features_rejected`,
//! `green_multinomial_partial_fit_equals_fit` (GREEN).
//!
//! Documented-not-pinned (missing surface / multi-file — gaussian #894-#898
//! precedent):
//!
//! sample_weight on fit/partial_fit (#901): sklearn `fit(X, y, sample_weight)`
//! (naive_bayes.py:712) weights the binarized `Y` (`Y *= sample_weight.T`,
//! naive_bayes.py:751) so `feature_count_ = Y.T@X` / `class_count_ =
//! Y.sum(axis=0)` become weighted — `fit(X,y,sample_weight=[1,2,1,1,1,3])`
//! gives `feature_count_ = [[11,3,2],[1,7,22]]`, `class_count_ = [4,5]`.
//! ferrolearn's `Fit` trait is `fn fit(&self,x,y)` — no `sample_weight`
//! parameter on `fit` or `partial_fit`. Missing surface; no forced test.
//!
//! fitted accessors feature_log_prob_/class_log_prior_/feature_count_/
//! class_count_ + the `_BaseDiscreteNB` coef_/intercept_ properties + the PyO3
//! surface (#902): sklearn exposes these (naive_bayes.py); `FittedMultinomialNB`
//! keeps `log_theta`/`log_prior`/`feature_counts`/`class_counts` private with no
//! accessor (only `classes()` via `HasClasses`), and `_RsMultinomialNB`
//! (`ferrolearn-python/src/extras.rs`, the `py_classifier!` macro) exposes only
//! `new(alpha, fit_prior)` + `fit` + `predict` — no `class_prior`/`force_alpha`
//! kwargs, no `predict_proba`/`predict_log_proba`/`score`/`partial_fit`, no
//! fitted-attr getters. Missing surface; a pytest divergence belongs in
//! ferrolearn-python, not in this crate.
//!
//! negative-feature exact ValueError message (#904-adjacent / #902): sklearn
//! raises `ValueError("Negative values in data passed to MultinomialNB (input
//! X)")` (`check_non_negative`, naive_bayes.py:881); ferrolearn raises
//! `FerroError::InvalidParameter { name: "X", reason: "MultinomialNB requires
//! non-negative feature values" }`. Both REJECT (guarded green below); the exact
//! message/type differs — documented, not pinned.
//!
//! partial_fit `classes=` / unseen-label handling: sklearn's shared
//! `_BaseDiscreteNB.partial_fit` (naive_bayes.py:628-709) requires `classes=`
//! at the FIRST call (`ValueError("classes must be passed on the first call to
//! partial_fit.")`) and thereafter binarizes `y` against the FIXED `classes_`.
//! ferrolearn `FittedMultinomialNB::partial_fit(&mut self, x, y)` has NO
//! `classes` argument: it loops only over the already-fitted `self.classes`, so
//! a brand-new label is SILENTLY DROPPED (no error, no new class). NOTE:
//! against the live oracle this is NOT a clean "silent-drop vs error" pin in the
//! binary 2-class case — sklearn's `label_binarize([2], classes=[0,1])` yields a
//! single-column `[[0]]` (binary quirk), so sklearn ALSO does not error on an
//! unseen label `2`; it mis-assigns the row's counts to class 0 rather than
//! dropping it. Because both sides differ from the obvious contract (sklearn
//! mis-counts, ferrolearn drops) and the divergence needs the `classes=`
//! partial_fit API (a multi-file surface change, #902-adjacent), it is
//! DOCUMENTED here, not pinned RED. The same-classes incremental path IS
//! value-correct and is guarded green
//! (`green_multinomial_partial_fit_equals_fit`).
//!
//! ferray substrate (#903): `multinomial.rs` imports `ndarray` + `num-traits`,
//! not `ferray-core`. Substrate migration; no observable-value pin.

use ferrolearn_bayes::MultinomialNB;
use ferrolearn_core::{Fit, Predict};
use ndarray::{Array1, Array2, array};

/// The shared count fixture from `.design/bayes/multinomial.md`:
/// `X = [[3,1,0],[2,0,1],[4,2,0],[0,1,4],[1,0,3],[0,2,5]]`, `y = [0,0,0,1,1,1]`.
fn count_fixture() -> (Array2<f64>, Array1<usize>) {
    let x = Array2::from_shape_vec(
        (6, 3),
        vec![
            3.0, 1.0, 0.0, // class 0
            2.0, 0.0, 1.0, // class 0
            4.0, 2.0, 0.0, // class 0
            0.0, 1.0, 4.0, // class 1
            1.0, 0.0, 3.0, // class 1
            0.0, 2.0, 5.0, // class 1
        ],
    )
    .unwrap();
    let y = array![0usize, 0, 0, 1, 1, 1];
    (x, y)
}

/// The query `q = [[2,1,1],[0,1,3]]`.
fn query() -> Array2<f64> {
    Array2::from_shape_vec((2, 3), vec![2.0, 1.0, 1.0, 0.0, 1.0, 3.0]).unwrap()
}

// ===========================================================================
// RED #904 — alpha >= 0 is a HARD reject at fit.
//
// sklearn `MultinomialNB._parameter_constraints` (naive_bayes.py:530):
//     "alpha": [Interval(Real, 0, None, closed="left"), "array-like"],
// → alpha must be >= 0, enforced at `fit` by `_validate_params`.
// `MultinomialNB(alpha=-0.5).fit(X,y)` raises `InvalidParameterError` (a
// ValueError subclass). This is DISTINCT from `_check_alpha`'s 1e-10 floor
// (naive_bayes.py:604-626), which only fires for alpha < 1e-10 when
// force_alpha=False.
//
// ferrolearn `fn fit` (multinomial.rs) computes
//     let alpha = crate::clamp_alpha(self.alpha, self.force_alpha);
// where `clamp_alpha` (= base::check_alpha) only FLOORS under
// force_alpha=false. With the default force_alpha=true, `clamp_alpha(-0.5,
// true)` returns -0.5 unchanged, so `fit` proceeds and computes
// `log((count - 0.5)/denom)` — negative-smoothed garbage / NaN, NO error.
//
// Live oracle (sklearn 1.5.2, run from /tmp):
//   python3 -c "from sklearn.naive_bayes import MultinomialNB; import numpy as np; \
//     X=np.array([[3.,1.,0.],[2.,0.,1.],[4.,2.,0.],[0.,1.,4.],[1.,0.,3.],[0.,2.,5.]]); \
//     y=np.array([0,0,0,1,1,1]); \
//     MultinomialNB(alpha=-0.5).fit(X,y)"
//   ->  InvalidParameterError :: The 'alpha' parameter of MultinomialNB must be
//       a float in the range [0.0, inf) or an array-like. Got -0.5 instead.
//
// We pin the OBSERVABLE: `fit` must return `Err` (sklearn raises). ferrolearn
// currently returns `Ok`: FAILS now. Minimally fixable in `multinomial.rs`
// `fn fit` — reject `alpha < 0` (InvalidParameter) before/around clamp_alpha.
// ===========================================================================

/// Divergence: ferrolearn's `MultinomialNB::fit` accepts a negative `alpha`,
/// whereas `sklearn/naive_bayes.py:530`
/// (`alpha: Interval(Real, 0, None, closed="left")`) makes `alpha < 0` a HARD
/// reject at `fit` (`InvalidParameterError`). `MultinomialNB(alpha=-0.5).fit`
/// raises in sklearn; ferrolearn returns `Ok` because `clamp_alpha(-0.5, true)`
/// passes -0.5 through unchanged (the floor only fires under force_alpha=false).
/// Tracking: #904
#[test]
fn divergence_multinomial_negative_alpha_rejected() {
    let (x, y) = count_fixture();

    // alpha = -0.5 — sklearn raises InvalidParameterError (oracle above).
    let model = MultinomialNB::<f64>::new().with_alpha(-0.5);
    let result = model.fit(&x, &y);

    assert!(
        result.is_err(),
        "fit with alpha=-0.5 should error (sklearn: InvalidParameterError, \
         'The 'alpha' parameter of MultinomialNB must be a float in the range \
         [0.0, inf) ...'), got Ok (#904)"
    );
}

// ===========================================================================
// GREEN — predict_proba / predict_log_proba / predict VALUE on the count
// fixture. Mirrors `_update_feature_log_prob` (naive_bayes.py:885-892) feeding
// `_joint_log_likelihood = X @ feature_log_prob_.T + class_log_prior_`
// (naive_bayes.py:894-896) through the _BaseNB pipeline.
//
// Live oracle (sklearn 1.5.2, run from /tmp):
//   python3 -c "from sklearn.naive_bayes import MultinomialNB; import numpy as np; \
//     X=np.array([[3.,1.,0.],[2.,0.,1.],[4.,2.,0.],[0.,1.,4.],[1.,0.,3.],[0.,2.,5.]]); \
//     y=np.array([0,0,0,1,1,1]); q=np.array([[2.,1.,1.],[0.,1.,3.]]); \
//     m=MultinomialNB().fit(X,y); \
//     print(m.predict_proba(q).tolist()); print(m.predict_log_proba(q).tolist()); \
//     print(m.predict(q).tolist())"
//   -> predict_proba    [[0.8843694464372913, 0.11563055356270838],
//                        [0.007188876743869827, 0.9928111232561301]]
//   -> predict_log_proba [[-0.12288037781713079, -2.1573550534903774],
//                        [-4.935220344228254, -0.007214841230117397]]
//   -> predict          [0, 1]
// ===========================================================================

/// Guard: `predict_proba` / `predict_log_proba` / `predict` match sklearn
/// `MultinomialNB().fit(X,y)` to ~1e-12 (the `_update_feature_log_prob` +
/// `_joint_log_likelihood` VALUE, naive_bayes.py:885-896).
#[test]
fn green_multinomial_predict_proba_log_proba_value() {
    let (x, y) = count_fixture();
    let q = query();
    let fitted = MultinomialNB::<f64>::new().fit(&x, &y).unwrap();

    let proba = fitted.predict_proba(&q).unwrap();
    let log_proba = fitted.predict_log_proba(&q).unwrap();
    let preds: Array1<usize> = fitted.predict(&q).unwrap();

    // sklearn 1.5.2 oracle (quoted above) — NOT copied from ferrolearn.
    const SK_PROBA: [[f64; 2]; 2] = [
        [0.8843694464372913, 0.11563055356270838],
        [0.007188876743869827, 0.9928111232561301],
    ];
    const SK_LOG_PROBA: [[f64; 2]; 2] = [
        [-0.12288037781713079, -2.1573550534903774],
        [-4.935220344228254, -0.007214841230117397],
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
                (log_proba[[i, c]] - SK_LOG_PROBA[i][c]).abs() <= TOL,
                "predict_log_proba[{i}][{c}]: ferrolearn {} vs sklearn {}",
                log_proba[[i, c]],
                SK_LOG_PROBA[i][c]
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
// NB has NO sum-to-1 and NO non-negativity check (UNLIKE GaussianNB, which
// rejects a non-unit-sum prior, see divergence_gaussian.rs #893). So
// `class_prior=[0.5,0.3]` (sum 0.8) is ACCEPTED on the sklearn side. ferrolearn
// `fn fit` checks ONLY length then `log_prior[ci] = p.ln()` — must ALSO accept.
//
// Live oracle (sklearn 1.5.2, run from /tmp):
//   python3 -c "from sklearn.naive_bayes import MultinomialNB; import numpy as np; \
//     X=np.array([[3.,1.,0.],[2.,0.,1.],[4.,2.,0.],[0.,1.,4.],[1.,0.,3.],[0.,2.,5.]]); \
//     y=np.array([0,0,0,1,1,1]); \
//     print(MultinomialNB(class_prior=[0.5,0.3]).fit(X,y).class_log_prior_.tolist())"
//   ->  [-0.6931471805599453, -1.2039728043259361]   (= log([0.5,0.3]); NO error)
//
// The class-prior term enters `_joint_log_likelihood` ADDITIVELY (X @ flp.T +
// class_log_prior_). On this balanced fixture the empirical prior is log(0.5)
// for BOTH classes, so its inter-class log-prior gap is 0. The explicit prior
// [0.5,0.3] contributes an inter-class gap of class_log_prior_[0] -
// class_log_prior_[1]. Hence the observable
//   (jll_explicit[i,0]-jll_explicit[i,1]) - (jll_emp[i,0]-jll_emp[i,1])
// equals that explicit prior gap. From the oracle class_log_prior_ above,
// the gap = -0.6931471805599453 - (-1.2039728043259361) = 0.5108256237659908.
//
//   python3 -c "import numpy as np; clp=np.log(np.array([0.5,0.3])); \
//     print(float(clp[0]-clp[1]))"   ->  0.5108256237659908
// ===========================================================================

/// Guard: ferrolearn `with_class_prior([0.5,0.3]).fit` SUCCEEDS, matching
/// sklearn `_update_class_log_prior` LENGTH-only validation
/// (naive_bayes.py:589-591) — discrete NB has NO sum/non-neg check (contrast
/// GaussianNB, which rejects this; divergence_gaussian.rs #893). The resulting
/// `class_log_prior_` = `log([0.5,0.3])`; we assert its inter-class gap VALUE
/// via the joint-log-likelihood column differences (the only observable, since
/// `log_prior` is private).
#[test]
fn green_multinomial_class_prior_length_only_accepts_non_unit_sum() {
    let (x, y) = count_fixture();

    // sum 0.8 — sklearn accepts (oracle above). ferrolearn must too.
    let model = MultinomialNB::<f64>::new().with_class_prior(vec![0.5, 0.3]);
    let result = model.fit(&x, &y);

    assert!(
        result.is_ok(),
        "fit with class_prior=[0.5,0.3] (sum 0.8) should SUCCEED \
         (discrete NB has no sum check; sklearn class_log_prior_ = \
         log([0.5,0.3]) = [-0.6931..,-1.2040..]), got Err"
    );

    let fitted = model.fit(&x, &y).unwrap();
    let empirical = MultinomialNB::<f64>::new().fit(&x, &y).unwrap();
    let q = query();
    let jll_prior = fitted.predict_joint_log_proba(&q).unwrap();
    let jll_emp = empirical.predict_joint_log_proba(&q).unwrap();

    // sklearn 1.5.2 oracle: class_log_prior_[0] - class_log_prior_[1] for
    // class_prior=[0.5,0.3] (= log(0.5) - log(0.3)); the balanced empirical
    // prior gap is 0, so the difference-of-gaps isolates this value.
    const SK_PRIOR_GAP: f64 = 0.5108256237659908;
    const TOL: f64 = 1e-12;

    for i in 0..2 {
        let gap_prior = jll_prior[[i, 0]] - jll_prior[[i, 1]];
        let gap_emp = jll_emp[[i, 0]] - jll_emp[[i, 1]];
        let observed = gap_prior - gap_emp;
        assert!(
            (observed - SK_PRIOR_GAP).abs() <= TOL,
            "row {i}: explicit-vs-empirical prior gap = {observed}, want \
             class_log_prior_[0]-class_log_prior_[1] = {SK_PRIOR_GAP}"
        );
    }
}

// ===========================================================================
// GREEN — score (mean accuracy) on the separable count fixture.
//
// Live oracle (sklearn 1.5.2, run from /tmp):
//   python3 -c "from sklearn.naive_bayes import MultinomialNB; import numpy as np; \
//     X=np.array([[3.,1.,0.],[2.,0.,1.],[4.,2.,0.],[0.,1.,4.],[1.,0.,3.],[0.,2.,5.]]); \
//     y=np.array([0,0,0,1,1,1]); print(MultinomialNB().fit(X,y).score(X,y))"
//   ->  1.0
// ===========================================================================

/// Guard: `score(X,y)` matches sklearn `MultinomialNB().fit(X,y).score(X,y)` =
/// `1.0` on this separable fixture (`ClassifierMixin.score` analog).
#[test]
fn green_multinomial_score_accuracy() {
    let (x, y) = count_fixture();
    let fitted = MultinomialNB::<f64>::new().fit(&x, &y).unwrap();

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
// sklearn `_count` (naive_bayes.py:881):
//     check_non_negative(X, "MultinomialNB (input X)")
// → `ValueError("Negative values in data passed to MultinomialNB (input X)")`.
// ferrolearn `fn fit` rejects with `InvalidParameter { name: "X" }`.
//
// Live oracle (sklearn 1.5.2, run from /tmp):
//   python3 -c "from sklearn.naive_bayes import MultinomialNB; import numpy as np; \
//     X=np.array([[1.,2.],[-1.,3.],[2.,1.],[0.,4.]]); y=np.array([0,0,1,1]); \
//     MultinomialNB().fit(X,y)"
//   ->  ValueError: Negative values in data passed to MultinomialNB (input X)
// ===========================================================================

/// Guard: ferrolearn `MultinomialNB().fit(X_with_neg, y)` returns `Err`,
/// matching sklearn's `check_non_negative` reject (naive_bayes.py:881). Both
/// REJECT; the exact message/type divergence is documented NOT-pinned.
#[test]
fn green_multinomial_negative_features_rejected() {
    let x = Array2::from_shape_vec((4, 2), vec![1.0, 2.0, -1.0, 3.0, 2.0, 1.0, 0.0, 4.0]).unwrap();
    let y = array![0usize, 0, 1, 1];

    let result = MultinomialNB::<f64>::new().fit(&x, &y);

    assert!(
        result.is_err(),
        "fit with a negative feature should error (sklearn: ValueError \
         'Negative values in data passed to MultinomialNB (input X)'), got Ok"
    );
}

// ===========================================================================
// GREEN — partial_fit over chunks == fit on the whole (same-classes path,
// REQ-8). sklearn's shared `_BaseDiscreteNB.partial_fit` (naive_bayes.py:628-709)
// reapplies the smoothing to accumulated counts each call, so chunked
// partial_fit equals whole fit.
//
// Live oracle (sklearn 1.5.2, run from /tmp):
//   python3 -c "from sklearn.naive_bayes import MultinomialNB; import numpy as np; \
//     X=np.array([[3.,1.,0.],[2.,0.,1.],[4.,2.,0.],[0.,1.,4.],[1.,0.,3.],[0.,2.,5.]]); \
//     y=np.array([0,0,0,1,1,1]); q=np.array([[2.,1.,1.],[0.,1.,3.]]); \
//     m=MultinomialNB(); m.partial_fit(X[:4],y[:4],classes=[0,1]); m.partial_fit(X[4:],y[4:]); \
//     print(m.predict_proba(q).tolist())"
//   ->  [[0.8843694464372913, 0.11563055356270838],
//        [0.007188876743869827, 0.9928111232561301]]
//      (identical to the whole-fit predict_proba above; np.allclose(flp)==True)
// ===========================================================================

/// Guard: ferrolearn's two-chunk `partial_fit` (same classes both chunks)
/// reproduces sklearn's whole-`fit` `predict_proba` to ~1e-12
/// (`_BaseDiscreteNB.partial_fit` accumulate-then-resmooth, naive_bayes.py:628-709).
#[test]
fn green_multinomial_partial_fit_equals_fit() {
    let (x, y) = count_fixture();
    let q = query();

    // Chunk 1: rows 0..4 (both classes present so the incremental path covers
    // every fitted class). Chunk 2: rows 4..6.
    let x1 = x.slice(ndarray::s![0..4, ..]).to_owned();
    let y1 = y.slice(ndarray::s![0..4]).to_owned();
    let x2 = x.slice(ndarray::s![4..6, ..]).to_owned();
    let y2 = y.slice(ndarray::s![4..6]).to_owned();

    let mut fitted = MultinomialNB::<f64>::new().fit(&x1, &y1).unwrap();
    fitted.partial_fit(&x2, &y2).unwrap();
    let proba = fitted.predict_proba(&q).unwrap();

    // sklearn 1.5.2 oracle (chunked partial_fit == whole fit), quoted above.
    const SK_PROBA: [[f64; 2]; 2] = [
        [0.8843694464372913, 0.11563055356270838],
        [0.007188876743869827, 0.9928111232561301],
    ];
    const TOL: f64 = 1e-12;

    for i in 0..2 {
        for c in 0..2 {
            assert!(
                (proba[[i, c]] - SK_PROBA[i][c]).abs() <= TOL,
                "partial_fit predict_proba[{i}][{c}]: ferrolearn {} vs sklearn {}",
                proba[[i, c]],
                SK_PROBA[i][c]
            );
        }
    }
}
