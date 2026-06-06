//! Divergence pins + value-contract guards for `CategoricalNB` /
//! `FittedCategoricalNB` (`ferrolearn-bayes/src/categorical.rs`) against the LIVE
//! scikit-learn 1.5.2 oracle (`from sklearn.naive_bayes import CategoricalNB`,
//! mirroring `sklearn/naive_bayes.py` `CategoricalNB` + `_BaseDiscreteNB`).
//!
//! Every expected value below is a LIVE `sklearn` 1.5.2 oracle value (computed
//! via `python3 -c "..."` run from `/tmp`, quoted above each block) — NEVER
//! copied from the ferrolearn side (goal.md R-CHAR-3).
//!
//! Mirrors `sklearn/naive_bayes.py`:
//! `CategoricalNB._parameter_constraints` OVERRIDES `alpha` to
//! `Interval(Real, 0, None, closed="left")` (naive_bayes.py:1333) — `alpha >= 0`
//! is ALLOWED so `alpha = 0` is ACCEPTED at `fit` (emits a `divide by zero
//! encountered in log` RuntimeWarning where a count is zero, NO error); only
//! `alpha < 0` is a HARD reject (`InvalidParameterError`).
//! `CategoricalNB._check_X` / `_check_X_y` validate via `_validate_data(...,
//! dtype="int")` + `check_non_negative(X, "CategoricalNB (input X)")`
//! (naive_bayes.py:1427-1440) — a negative value raises
//! `ValueError("Negative values in data passed to CategoricalNB (input X)")`.
//! `CategoricalNB._update_feature_log_prob`
//! (naive_bayes.py:1498-1506) — `smoothed_cat_count = category_count_[i] +
//! alpha`; `feature_log_prob_[i] = log(smoothed_cat_count) -
//! log(smoothed_cat_count.sum(axis=1).reshape(-1,1))`.
//! `CategoricalNB._joint_log_likelihood` (naive_bayes.py:1508-1515) — `jll +=
//! feature_log_prob_[i][:, X[:,i]].T; jll += class_log_prior_`.
//! `CategoricalNB._validate_n_categories` (naive_bayes.py:1446-1466) —
//! `n_categories_ = max(X.max(0)+1, min_categories)`.
//! shared `_update_class_log_prior` LENGTH-only check then `log(class_prior)`
//! (naive_bayes.py:580-602, NO sum/non-neg check for discrete NB).
//!
//! Design doc: `.design/bayes/categorical.md` (commit 0994f03b).
//!
//! Test taxonomy — the RED pins FAIL now and go green when the generator lands
//! the (single-file) fix; the GREEN guards PASS now and protect the parts
//! already correct:
//! `divergence_categorical_alpha_zero_allowed` (RED, #921);
//! `divergence_categorical_negative_features_rejected` (RED, #922);
//! `green_categorical_predict_value`,
//! `green_categorical_min_categories`,
//! `green_categorical_class_prior_length_only`,
//! `green_categorical_score_accuracy` (GREEN).
//!
//! Documented-not-pinned (behavioral / larger change / missing surface — no
//! forced test this iteration):
//!
//! unseen-category at predict (#920): sklearn requires category indices `<
//! n_categories_[i]`; the `feature_log_prob_[i][:, X[:,i]]` fancy-index
//! (naive_bayes.py:1513) raises `IndexError("index 5 is out of bounds for axis 1
//! with size 2")` for an index beyond `n_categories_`. ferrolearn's
//! `log_prob_for` (categorical.rs:508-518) returns a uniform
//! `(1/(n_known_cats+1)).ln()` fallback for any unknown category — NO error
//! (`predict([[5,0]])` → `[0]`, `predict_proba` → `[[0.5,0.5]]`). This is a real
//! R-DEV-3 output-contract divergence, but matching sklearn's `IndexError`
//! requires threading a `Result` through `log_prob_for` / `joint_log_likelihood`
//! and reconsidering the graceful-degradation design — NOT a one-line fix like
//! the two RED pins. Documented NOT-STARTED #920; not pinned RED this iteration.
//!
//! sample_weight on fit + partial_fit new-category/new-class EXTENSION (#924):
//! sklearn `fit(X, y, sample_weight=None)` (naive_bayes.py:712) weights the
//! binarized `Y` so `class_count_` / the `np.bincount` per-category counts become
//! weighted (naive_bayes.py:1468-1496); the shared `partial_fit`
//! (naive_bayes.py:628-709) keeps `n_categories_` fixed (a category `>=
//! n_categories_` in a later chunk `IndexError`s) and binarizes against the full
//! `classes` list. ferrolearn's `impl Fit` is `fn fit(&self, x, y)` — no
//! `sample_weight`; `FittedCategoricalNB::partial_fit` (no `classes` arg) APPENDS
//! never-seen categories and inserts never-seen class labels — a non-sklearn
//! flexibility (documented in the method doc-comment). Missing surface /
//! deliberate deviation; not pinned.
//!
//! fitted accessors + PyO3 binding (#923): sklearn exposes `category_count_`,
//! `feature_log_prob_`, `class_count_`, `class_log_prior_`, `n_categories_`,
//! `classes_`, `n_features_in_` (naive_bayes.py:1266-1303). `FittedCategoricalNB`
//! exposes ONLY `classes()` (via `HasClasses`); the rest are PRIVATE fields with
//! no accessor. **CategoricalNB has NO PyO3 binding** — there is no
//! `_RsCategoricalNB` in `ferrolearn-python/src/extras.rs` and no
//! `ferrolearn.CategoricalNB`, so `import ferrolearn` cannot reach it (unlike its
//! discrete-NB siblings). A pytest divergence belongs in ferrolearn-python, not
//! here. Missing surface; documented, not pinned.
//!
//! ferray substrate (#925): `categorical.rs` imports `ndarray::{Array1, Array2}`
//! plus `num_traits::{Float, FromPrimitive, ToPrimitive}`, not `ferray-core` —
//! a substrate migration with no observable-value pin.

use ferrolearn_bayes::CategoricalNB;
use ferrolearn_core::error::FerroError;
use ferrolearn_core::{Fit, Predict};
use ndarray::{Array1, Array2, array};

/// The shared categorical fixture (feature values 0..K-1):
/// `X = [[0,1],[1,0],[0,0],[2,1],[2,0],[1,1]]`, `y = [0,0,0,1,1,1]`.
fn categorical_fixture() -> (Array2<f64>, Array1<usize>) {
    let x = Array2::from_shape_vec(
        (6, 2),
        vec![
            0.0, 1.0, // class 0
            1.0, 0.0, // class 0
            0.0, 0.0, // class 0
            2.0, 1.0, // class 1
            2.0, 0.0, // class 1
            1.0, 1.0, // class 1
        ],
    )
    .unwrap();
    let y = array![0usize, 0, 0, 1, 1, 1];
    (x, y)
}

// ===========================================================================
// RED #921 — alpha = 0 is ALLOWED at fit (over-rejection in ferrolearn).
//
// sklearn `CategoricalNB._parameter_constraints` OVERRIDES `alpha` to
// `Interval(Real, 0, None, closed="left")` (naive_bayes.py:1333) → `alpha >= 0`
// is allowed, so `alpha = 0` is INSIDE the interval and ACCEPTED at `fit`.
// `_check_alpha` (naive_bayes.py:604-626) only FLOORS to 1e-10 when `alpha <
// 1e-10 and not force_alpha`; with the default `force_alpha=True`, `alpha = 0`
// is used as-is, producing `-inf` log-probs where a count is zero, with a
// `divide by zero encountered in log` RuntimeWarning — NOT an error.
//
// ferrolearn `fn fit` (categorical.rs:238) rejects
//     self.alpha <= F::zero() && self.force_alpha
// → with the default `force_alpha=true` it OVER-REJECTS `alpha = 0`
// (`InvalidParameter`).
//
// Live oracle (sklearn 1.5.2, run from /tmp):
//   python3 -c "import warnings; warnings.simplefilter('ignore'); \
//     from sklearn.naive_bayes import CategoricalNB; import numpy as np; \
//     CategoricalNB(alpha=0.0).fit( \
//       np.array([[0,1],[1,0],[0,0],[2,1],[2,0],[1,1]]), \
//       np.array([0,0,0,1,1,1])); print('fit ok')"
//   ->  fit ok   (alpha=0 ACCEPTED; -inf where a count is zero; RuntimeWarning,
//                 NOT an error)
//
// We pin the OBSERVABLE: `fit` must return `Ok` (sklearn accepts). ferrolearn
// currently returns `Err`: FAILS now. Minimally fixable in `categorical.rs`
// `fn fit` — change the line-238 guard from `alpha <= 0` to `alpha < 0`
// (allow zero, reject only negatives).
// ===========================================================================

/// Divergence: ferrolearn's `CategoricalNB::fit` REJECTS `alpha = 0`, whereas
/// `sklearn/naive_bayes.py:1333`
/// (`alpha: Interval(Real, 0, None, closed="left")`) makes `alpha = 0` ALLOWED
/// (inside the closed-left interval). `CategoricalNB(alpha=0.0).fit(X,y)`
/// SUCCEEDS in sklearn (emits a divide-by-zero RuntimeWarning where a count is
/// zero, NO error); ferrolearn returns `Err` because the line-238 guard
/// `alpha <= 0 && force_alpha` fires under the default `force_alpha=true`.
/// Tracking: #921
#[test]
fn divergence_categorical_alpha_zero_allowed() {
    let (x, y) = categorical_fixture();

    // alpha = 0.0 — sklearn ACCEPTS (oracle above: "fit ok").
    let model = CategoricalNB::<f64>::new().with_alpha(0.0);
    let result = model.fit(&x, &y);

    assert!(
        result.is_ok(),
        "fit with alpha=0.0 should SUCCEED (sklearn: alpha Interval closed-left \
         at 0 → 0 ACCEPTED, divide-by-zero RuntimeWarning but NO error), got Err \
         (#921)"
    );
}

// ===========================================================================
// RED #922 — negative feature values are REJECTED at fit.
//
// sklearn `CategoricalNB._check_X_y` validates `X` via `_validate_data(X, y,
// dtype="int", ...)` then `check_non_negative(X, "CategoricalNB (input X)")`
// (naive_bayes.py:1435-1440) → a negative value raises
// `ValueError("Negative values in data passed to CategoricalNB (input X)")`.
//
// ferrolearn `fn fit` maps every value via `x[[i,j]].to_usize().unwrap_or(0)`
// (categorical.rs:286, 312) — a negative value silently becomes category `0`
// (`unwrap_or(0)`), so `fit` proceeds with NO error.
//
// Live oracle (sklearn 1.5.2, run from /tmp):
//   python3 -c "from sklearn.naive_bayes import CategoricalNB; import numpy as np
//   try:
//       CategoricalNB().fit(np.array([[-1,0],[0,1],[1,0],[1,1]]), np.array([0,0,1,1]))
//       print('no error')
//   except Exception as e:
//       print(type(e).__name__, '::', e)"
//   ->  ValueError :: Negative values in data passed to CategoricalNB (input X)
//
// We pin the OBSERVABLE: `fit` must return `Err` (sklearn raises). ferrolearn
// currently returns `Ok`: FAILS now. Minimally fixable in `categorical.rs`
// `fn fit` — add a `x.iter().any(|&v| v < F::zero())` reject (like
// MultinomialNB / ComplementNB).
// ===========================================================================

/// Divergence: ferrolearn's `CategoricalNB::fit` accepts a negative feature
/// value (silently mapping it to category 0 via `to_usize().unwrap_or(0)`,
/// categorical.rs:286/312), whereas `sklearn/naive_bayes.py:1435-1440`
/// (`check_non_negative(X, "CategoricalNB (input X)")`) makes a negative value a
/// HARD reject at `fit`. `CategoricalNB().fit(X_with_neg, y)` raises
/// `ValueError("Negative values in data passed to CategoricalNB (input X)")` in
/// sklearn; ferrolearn returns `Ok`.
/// Tracking: #922
#[test]
fn divergence_categorical_negative_features_rejected() {
    // X[0,0] = -1.0 — sklearn raises ValueError (oracle above).
    let x = Array2::from_shape_vec((4, 2), vec![-1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]).unwrap();
    let y = array![0usize, 0, 1, 1];

    let result = CategoricalNB::<f64>::new().fit(&x, &y);

    assert!(
        result.is_err(),
        "fit with a negative feature should error (sklearn: ValueError \
         'Negative values in data passed to CategoricalNB (input X)'), got Ok \
         (#922)"
    );
}

// ===========================================================================
// #920 — predict-path validation: negative + unseen-category at predict.
//
// sklearn `CategoricalNB._check_X` runs `check_non_negative(X, "CategoricalNB
// (input X)")` on the predict path (naive_bayes.py:1432) → a negative value
// raises `ValueError`; and `_joint_log_likelihood`'s fancy-index
// `feature_log_prob_[i][:, X[:,i]]` (naive_bayes.py:1513) raises `IndexError`
// for a category index `>= n_categories_[i]`.
//
// On `X = [[0,1],[1,0],[0,0],[2,1],[2,0],[1,1]]`, `y = [0,0,0,1,1,1]`
// (`n_categories_ = [3,2]`):
//
//   python3 -c "from sklearn.naive_bayes import CategoricalNB; import numpy as np
//   X=np.array([[0,1],[1,0],[0,0],[2,1],[2,0],[1,1]]); y=np.array([0,0,0,1,1,1])
//   m=CategoricalNB().fit(X,y)
//   for q in ([[-1,0]],[[5,0]]):
//       try: m.predict(np.array(q)); print('no error')
//       except Exception as e: print(type(e).__name__,'::',e)"
//   ->  predict([[-1,0]])  ValueError :: Negative values in data passed to
//                          CategoricalNB (input X)
//   ->  predict([[5,0]])   IndexError :: index 5 is out of bounds for axis 1
//                          with size 3
//
// ferrolearn maps both to `FerroError::InvalidParameter` (the workspace error
// analog) — the predict path now rejects instead of silently mapping `-1 → 0`
// or returning a uniform fallback for the unseen category. Tracking: #920
// ===========================================================================

/// Pin #920: ferrolearn's predict path must REJECT a negative feature value
/// (sklearn `_check_X` → `check_non_negative`, naive_bayes.py:1432 →
/// `ValueError`). ferrolearn returns `Err(FerroError::InvalidParameter)`.
#[test]
fn categorical_predict_negative_rejected() -> Result<(), FerroError> {
    let (x, y) = categorical_fixture();
    let fitted = CategoricalNB::<f64>::new().fit(&x, &y)?;

    // q[0,0] = -1.0 — sklearn raises ValueError (oracle above).
    let q = array![[-1.0_f64, 0.0]];
    let result = fitted.predict(&q);

    assert!(
        matches!(result, Err(FerroError::InvalidParameter { .. })),
        "predict on a negative feature must return Err(InvalidParameter) \
         (sklearn: ValueError 'Negative values in data passed to CategoricalNB \
         (input X)'), got {result:?} (#920)"
    );
    Ok(())
}

/// Pin #920: ferrolearn's predict path must REJECT an unseen category index
/// `>= n_categories_[i]` (sklearn `_joint_log_likelihood` fancy-index,
/// naive_bayes.py:1513 → `IndexError`). Category 5 ≥ `n_categories_[0] = 3`.
/// ferrolearn returns `Err(FerroError::InvalidParameter)`.
#[test]
fn categorical_predict_unseen_category_rejected() -> Result<(), FerroError> {
    let (x, y) = categorical_fixture();
    let fitted = CategoricalNB::<f64>::new().fit(&x, &y)?;

    // q[0,0] = 5.0, but n_categories_[0] = 3 — sklearn raises IndexError.
    let q = array![[5.0_f64, 0.0]];
    let result = fitted.predict(&q);

    assert!(
        matches!(result, Err(FerroError::InvalidParameter { .. })),
        "predict on an unseen category (5 >= n_categories_[0]=3) must return \
         Err(InvalidParameter) (sklearn: IndexError 'index 5 is out of bounds \
         for axis 1 with size 3'), got {result:?} (#920)"
    );
    Ok(())
}

// ===========================================================================
// GREEN — predict_proba / predict_joint_log_proba / predict VALUE on the
// categorical fixture. Mirrors `CategoricalNB._update_feature_log_prob`
// (naive_bayes.py:1498-1506) feeding `_joint_log_likelihood`
// (naive_bayes.py:1508-1515) through the `_BaseNB` pipeline.
//
// Live oracle (sklearn 1.5.2, run from /tmp):
//   python3 -c "from sklearn.naive_bayes import CategoricalNB; import numpy as np; \
//     X=np.array([[0,1],[1,0],[0,0],[2,1],[2,0],[1,1]]); y=np.array([0,0,0,1,1,1]); \
//     q=np.array([[0,0],[2,1]]); m=CategoricalNB().fit(X,y); \
//     print(m.predict_proba(q).tolist()); \
//     print(m.predict_joint_log_proba(q).tolist()); \
//     print(m.predict(q).tolist())"
//   -> predict_proba             [[0.8181818181818182, 0.18181818181818182],
//                                 [0.18181818181818182, 0.8181818181818182]]
//   -> predict_joint_log_proba   [[-1.8971199848858809, -3.401197381662155],
//                                 [-3.401197381662155, -1.8971199848858809]]
//   -> predict                   [0, 1]
// ===========================================================================

/// Guard: `predict_proba` / `predict_joint_log_proba` / `predict` match sklearn
/// `CategoricalNB().fit(X,y)` to ~1e-12 (the `_update_feature_log_prob` +
/// `_joint_log_likelihood` VALUE, naive_bayes.py:1498-1515).
#[test]
fn green_categorical_predict_value() {
    let (x, y) = categorical_fixture();
    let q = Array2::from_shape_vec((2, 2), vec![0.0, 0.0, 2.0, 1.0]).unwrap();
    let fitted = CategoricalNB::<f64>::new().fit(&x, &y).unwrap();

    let proba = fitted.predict_proba(&q).unwrap();
    let jll = fitted.predict_joint_log_proba(&q).unwrap();
    let preds: Array1<usize> = fitted.predict(&q).unwrap();

    // sklearn 1.5.2 oracle (quoted above) — NOT copied from ferrolearn.
    const SK_PROBA: [[f64; 2]; 2] = [
        [0.8181818181818182, 0.18181818181818182],
        [0.18181818181818182, 0.8181818181818182],
    ];
    const SK_JLL: [[f64; 2]; 2] = [
        [-1.8971199848858809, -3.401197381662155],
        [-3.401197381662155, -1.8971199848858809],
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
// GREEN — min_categories n_categories_ VALUE on an allocated-but-unobserved
// category. Mirrors `CategoricalNB._validate_n_categories`
// (naive_bayes.py:1446-1466, `n_categories_ = max(X.max(0)+1, min_categories)`)
// + the `_count` padding (naive_bayes.py:1491-1493) so the unobserved-but-
// allocated category 3 gets the smoothed weight `alpha/(N_c+alpha*K_j)`.
//
// Live oracle (sklearn 1.5.2, run from /tmp):
//   python3 -c "from sklearn.naive_bayes import CategoricalNB; import numpy as np; \
//     X=np.array([[0,1],[1,0],[0,0],[2,1],[2,0],[1,1]]); y=np.array([0,0,0,1,1,1]); \
//     m=CategoricalNB(min_categories=4).fit(X,y); \
//     print(m.n_categories_.tolist()); \
//     print(m.predict_joint_log_proba(np.array([[3,0]])).tolist()); \
//     print(m.predict_proba(np.array([[3,0]])).tolist()); \
//     print(m.predict(np.array([[3,0]])).tolist())"
//   -> n_categories_             [4, 4]
//   -> predict_joint_log_proba   [[-3.4863551900024623, -3.891820298110627]]
//   -> predict_proba             [[0.6000000000000001, 0.39999999999999997]]
//   -> predict                   [0]
// (feature 0 value 3 is the allocated-but-unobserved category; feature 1 value 0
//  is observed, so the two classes are NOT symmetric for this query.)
// ===========================================================================

/// Guard: `with_min_categories(4)` `predict_joint_log_proba` / `predict_proba` /
/// `predict` for a query with an allocated-but-unobserved category (3) match
/// sklearn `CategoricalNB(min_categories=4).fit(X,y)` to ~1e-12 (the
/// `_validate_n_categories` + `_count` padding VALUE, naive_bayes.py:1446-1493).
#[test]
fn green_categorical_min_categories() {
    let (x, y) = categorical_fixture();
    let q = Array2::from_shape_vec((1, 2), vec![3.0, 0.0]).unwrap();
    let fitted = CategoricalNB::<f64>::new()
        .with_min_categories(4)
        .fit(&x, &y)
        .unwrap();

    let jll = fitted.predict_joint_log_proba(&q).unwrap();
    let proba = fitted.predict_proba(&q).unwrap();
    let preds: Array1<usize> = fitted.predict(&q).unwrap();

    // sklearn 1.5.2 oracle (quoted above) — NOT copied from ferrolearn.
    const SK_JLL: [f64; 2] = [-3.4863551900024623, -3.891820298110627];
    const SK_PROBA: [f64; 2] = [0.6000000000000001, 0.39999999999999997];
    const TOL: f64 = 1e-12;

    for c in 0..2 {
        assert!(
            (jll[[0, c]] - SK_JLL[c]).abs() <= TOL,
            "min_categories jll[0][{c}]: ferrolearn {} vs sklearn {}",
            jll[[0, c]],
            SK_JLL[c]
        );
        assert!(
            (proba[[0, c]] - SK_PROBA[c]).abs() <= TOL,
            "min_categories proba[0][{c}]: ferrolearn {} vs sklearn {}",
            proba[[0, c]],
            SK_PROBA[c]
        );
    }
    // sklearn 1.5.2 oracle: predict([[3,0]]) == [0].
    assert_eq!(preds, array![0usize]);
}

// ===========================================================================
// GREEN — class_prior LENGTH-only validation (REQ-2 MATCH, NOT a divergence).
// sklearn `_update_class_log_prior` (naive_bayes.py:589-591) checks ONLY length
// then `class_log_prior_ = np.log(class_prior)` — discrete NB has NO sum-to-1
// and NO non-negativity check. So `class_prior=[0.5,0.3]` (sum 0.8) is ACCEPTED.
// ferrolearn `resolve_class_log_prior` checks ONLY `priors.len() != n_classes`
// (categorical.rs:386) — must ALSO accept.
//
// Live oracle (sklearn 1.5.2, run from /tmp):
//   python3 -c "from sklearn.naive_bayes import CategoricalNB; import numpy as np; \
//     X=np.array([[0,1],[1,0],[0,0],[2,1],[2,0],[1,1]]); y=np.array([0,0,0,1,1,1]); \
//     m=CategoricalNB(class_prior=[0.5,0.3]).fit(X,y); \
//     print(m.class_log_prior_.tolist())"
//   ->  [-0.6931471805599453, -1.2039728043259361]
//       (sum 0.8 ACCEPTED; class_log_prior_ = log([0.5,0.3]); NO error)
// ===========================================================================

/// Guard: ferrolearn `with_class_prior([0.5,0.3]).fit` SUCCEEDS, matching
/// sklearn `_update_class_log_prior` LENGTH-only validation
/// (naive_bayes.py:589-591) — discrete NB has NO sum/non-neg check (sklearn
/// accepts sum-0.8 priors, `class_log_prior_ = log([0.5,0.3])`).
#[test]
fn green_categorical_class_prior_length_only() {
    let (x, y) = categorical_fixture();

    // sum 0.8 — sklearn accepts (oracle above). ferrolearn must too.
    let model = CategoricalNB::<f64>::new().with_class_prior(vec![0.5, 0.3]);
    let result = model.fit(&x, &y);

    assert!(
        result.is_ok(),
        "fit with class_prior=[0.5,0.3] (sum 0.8) should SUCCEED (discrete NB \
         has no sum check; sklearn ACCEPTS, class_log_prior_ = log([0.5,0.3])), \
         got Err"
    );
}

// ===========================================================================
// GREEN — score (mean accuracy) on the separable categorical fixture
// (`ClassifierMixin.score` analog).
//
// Live oracle (sklearn 1.5.2, run from /tmp):
//   python3 -c "from sklearn.naive_bayes import CategoricalNB; import numpy as np; \
//     X=np.array([[0,1],[1,0],[0,0],[2,1],[2,0],[1,1]]); y=np.array([0,0,0,1,1,1]); \
//     print(CategoricalNB().fit(X,y).score(X,y))"
//   ->  1.0
// ===========================================================================

/// Guard: `score(X,y)` matches sklearn `CategoricalNB().fit(X,y).score(X,y)` =
/// `1.0` on this separable fixture (`ClassifierMixin.score` analog).
#[test]
fn green_categorical_score_accuracy() {
    let (x, y) = categorical_fixture();
    let fitted = CategoricalNB::<f64>::new().fit(&x, &y).unwrap();

    let acc = fitted.score(&x, &y).unwrap();

    // sklearn 1.5.2 oracle: 1.0.
    const SK_SCORE: f64 = 1.0;
    assert!(
        (acc - SK_SCORE).abs() <= 1e-12,
        "score = {acc}, want {SK_SCORE}"
    );
}
