//! Conformance tests for ferrolearn-bayes vs scikit-learn.
//!
//! Each test loads a fixture from `fixtures/<name>.json`, runs the matching
//! ferrolearn Naive Bayes model on the input, and compares against sklearn's
//! output via `ferrolearn-test-oracle` helpers.
//!
//! Naive Bayes is closed-form: predictions and learned parameters should
//! agree to numerical-noise level (`TOL_NAIVE_BAYES_REL = 1e-9`,
//! `TOL_NAIVE_BAYES_ABS = 1e-12`).
//!
//! ## Parameter introspection
//!
//! `class_log_prior` / `feature_log_prob` / `theta` / `var` are stored
//! privately on the fitted structs and are not (yet) exposed via accessor
//! methods. Where the public API permits, we *derive* the relevant
//! parameters via `predict_joint_log_proba` on probe inputs:
//!
//! * `MultinomialNB::joint_log_proba(zeros) = class_log_prior`
//! * `MultinomialNB::joint_log_proba(eye[j]) - class_log_prior = feature_log_prob[:, j]`
//! * `ComplementNB::joint_log_proba(eye[j]) = -feature_log_prob[:, j]`
//!
//! For `BernoulliNB` and `GaussianNB` the parameter values cannot be cleanly
//! recovered through the public surface, so those checks are limited to
//! predictions and accuracy. (The fixtures still ship the sklearn-side
//! parameter expectations for the day accessor methods are added.)

use ferrolearn_bayes::{BernoulliNB, ComplementNB, GaussianNB, MultinomialNB};
use ferrolearn_core::{Fit, Predict};
use ferrolearn_test_oracle::{
    TOL_NAIVE_BAYES_ABS, TOL_NAIVE_BAYES_REL, assert_close, assert_close_slice,
    assert_labels_equal, json_to_array1, json_to_array2, json_to_labels, load_fixture,
};
use ndarray::{Array1, Array2};

// ---------------------------------------------------------------------------
// Small local helpers.
// ---------------------------------------------------------------------------

fn array1_to_usize(arr: &Array1<f64>) -> Array1<usize> {
    arr.iter().map(|&v| v as usize).collect()
}

/// Compute prediction accuracy in the same way sklearn's `.score()` does.
fn accuracy(preds: &Array1<usize>, y: &Array1<usize>) -> f64 {
    let n_correct = preds.iter().zip(y.iter()).filter(|(a, b)| a == b).count();
    n_correct as f64 / preds.len() as f64
}

/// Identity matrix of shape (n, n) as f64. Used as probe input to
/// `predict_joint_log_proba` to extract per-feature log-probability columns.
fn eye_f64(n: usize) -> Array2<f64> {
    let mut a = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        a[[i, i]] = 1.0;
    }
    a
}

// ---------------------------------------------------------------------------
// 1. BernoulliNB — binary features, alpha=1.0.
//
// Predictions and accuracy must match sklearn exactly. Parameter values
// (`class_log_prior`, `feature_log_prob`) are not recoverable through the
// public surface — see file-level docs.
// ---------------------------------------------------------------------------

#[test]
fn conformance_bernoulli_nb() {
    let fx = load_fixture("bernoulli_nb");
    let (rel, abs) = fx.tolerance(TOL_NAIVE_BAYES_REL, TOL_NAIVE_BAYES_ABS);

    let x = json_to_array2(&fx.input["X"]);
    let y = array1_to_usize(&json_to_array1(&fx.input["y"]));

    let alpha = fx.params["alpha"].as_f64().unwrap_or(1.0);

    let model = BernoulliNB::<f64>::new().with_alpha(alpha);
    let fitted = model.fit(&x, &y).expect("BernoulliNB fit");

    let preds = fitted.predict(&x).expect("BernoulliNB predict");
    let preds_i64: Vec<i64> = preds.iter().map(|&v| v as i64).collect();
    let expected_preds = json_to_labels(&fx.expected["predictions"]);
    assert_labels_equal(&preds_i64, &expected_preds, "BernoulliNB.predict");

    let acc = accuracy(&preds, &y);
    assert_close(
        acc,
        fx.expected["accuracy"].as_f64().unwrap(),
        rel,
        abs,
        "BernoulliNB.accuracy",
    );
}

// ---------------------------------------------------------------------------
// 2. ComplementNB — non-negative counts, alpha=1.0.
//
// In sklearn's convention `ComplementNB._joint_log_likelihood(X) = X @
// feature_log_prob_.T` where `feature_log_prob_ = -log(p_complement)` is
// **positive**. ferrolearn stores `weights = log(p_complement)`
// (**negative**) and negates inside `predict_joint_log_proba`. The two
// libraries therefore agree on predictions (argmax invariant) but the
// extracted `feature_log_prob` differs in sign — see candidate divergence
// note in the final report.
// ---------------------------------------------------------------------------

#[test]
fn conformance_complement_nb() {
    let fx = load_fixture("complement_nb");
    let (rel, abs) = fx.tolerance(TOL_NAIVE_BAYES_REL, TOL_NAIVE_BAYES_ABS);

    let x = json_to_array2(&fx.input["X"]);
    let y = array1_to_usize(&json_to_array1(&fx.input["y"]));

    let alpha = fx.params["alpha"].as_f64().unwrap_or(1.0);

    let model = ComplementNB::<f64>::new().with_alpha(alpha);
    let fitted = model.fit(&x, &y).expect("ComplementNB fit");

    let preds = fitted.predict(&x).expect("ComplementNB predict");
    let preds_i64: Vec<i64> = preds.iter().map(|&v| v as i64).collect();
    let expected_preds = json_to_labels(&fx.expected["predictions"]);
    assert_labels_equal(&preds_i64, &expected_preds, "ComplementNB.predict");

    let acc = accuracy(&preds, &y);
    assert_close(
        acc,
        fx.expected["accuracy"].as_f64().unwrap(),
        rel,
        abs,
        "ComplementNB.accuracy",
    );

    // Derive feature_log_prob from `predict_joint_log_proba` on the identity
    // matrix. Magnitudes match sklearn to numerical-noise, but the **sign**
    // is flipped: sklearn stores `feature_log_prob_ = -log(p_complement)`
    // (positive), while ferrolearn's internal `weights` field is
    // `+log(p_complement)` (negative) and is negated inside
    // `predict_joint_log_proba`, so `jll(eye[j])[j, c]` equals
    // `+log(p_complement)` instead of sklearn's `-log(p_complement)`. The
    // sign-aware comparison below documents this convention divergence:
    // predictions/accuracy are correct, but extracted parameter values
    // disagree in sign. The hard check is left ignored pending reconciliation
    // (see candidate-divergence note in the final report).
    let expected_flp = json_to_array2(&fx.expected["feature_log_prob"]);
    let n_features = x.ncols();
    let eye = eye_f64(n_features);
    let jll = fitted
        .predict_joint_log_proba(&eye)
        .expect("ComplementNB predict_joint_log_proba");
    let n_classes = jll.ncols();
    let mut derived = Array2::<f64>::zeros((n_classes, n_features));
    for ci in 0..n_classes {
        for j in 0..n_features {
            derived[[ci, j]] = -jll[[j, ci]];
        }
    }
    // Sanity check: magnitudes agree with sklearn to numerical-noise even
    // though the sign convention differs.
    for ((&a, &e), idx) in derived
        .iter()
        .zip(expected_flp.iter())
        .zip(0..derived.len())
    {
        let diff = (a.abs() - e.abs()).abs();
        let threshold = abs.max(rel * e.abs());
        assert!(
            diff <= threshold,
            "ComplementNB.feature_log_prob[{idx}]: |actual|={a_abs:.6} |expected|={e_abs:.6} \
             diff={diff:.3e} threshold={threshold:.3e}",
            a_abs = a.abs(),
            e_abs = e.abs()
        );
    }
}

/// Strict sign-matching `feature_log_prob` check for ComplementNB.
///
/// Both sklearn and ferrolearn now store the positive form
/// `feature_log_prob_ = -log(complement_prob)` — see #346. We derive the
/// fitted weights by feeding the identity matrix through
/// `predict_joint_log_proba`: jll[j, ci] = sum_k I[j, k] * w[ci, k] = w[ci, j].
#[test]
fn conformance_complement_nb_feature_log_prob_strict() {
    let fx = load_fixture("complement_nb");
    let (rel, abs) = fx.tolerance(TOL_NAIVE_BAYES_REL, TOL_NAIVE_BAYES_ABS);

    let x = json_to_array2(&fx.input["X"]);
    let y = array1_to_usize(&json_to_array1(&fx.input["y"]));
    let alpha = fx.params["alpha"].as_f64().unwrap_or(1.0);

    let fitted = ComplementNB::<f64>::new()
        .with_alpha(alpha)
        .fit(&x, &y)
        .expect("ComplementNB fit");

    let n_features = x.ncols();
    let eye = eye_f64(n_features);
    let jll = fitted
        .predict_joint_log_proba(&eye)
        .expect("ComplementNB predict_joint_log_proba");
    let n_classes = jll.ncols();
    let mut derived = Array2::<f64>::zeros((n_classes, n_features));
    for ci in 0..n_classes {
        for j in 0..n_features {
            derived[[ci, j]] = jll[[j, ci]];
        }
    }
    let expected_flp = json_to_array2(&fx.expected["feature_log_prob"]);
    assert_close_slice(
        derived.as_slice().unwrap(),
        expected_flp.as_slice().unwrap(),
        rel,
        abs,
        "ComplementNB.feature_log_prob (sign-strict)",
    );
}

// ---------------------------------------------------------------------------
// 3. GaussianNB — iris (no params).
//
// Predictions and accuracy are compared exactly. `class_prior`, `theta`,
// `var` are not exposed by the public API; the fixture's parameter
// expectations are kept for the future accessor surface.
// ---------------------------------------------------------------------------

#[test]
fn conformance_gaussian_nb() {
    let fx = load_fixture("gaussian_nb");
    let (rel, abs) = fx.tolerance(TOL_NAIVE_BAYES_REL, TOL_NAIVE_BAYES_ABS);

    let x = json_to_array2(&fx.input["X"]);
    let y = array1_to_usize(&json_to_array1(&fx.input["y"]));

    let model = GaussianNB::<f64>::new();
    let fitted = model.fit(&x, &y).expect("GaussianNB fit");

    let preds = fitted.predict(&x).expect("GaussianNB predict");
    let preds_i64: Vec<i64> = preds.iter().map(|&v| v as i64).collect();
    let expected_preds = json_to_labels(&fx.expected["predictions"]);
    assert_labels_equal(&preds_i64, &expected_preds, "GaussianNB.predict");

    let acc = accuracy(&preds, &y);
    assert_close(
        acc,
        fx.expected["accuracy"].as_f64().unwrap(),
        rel,
        abs,
        "GaussianNB.accuracy",
    );
}

// ---------------------------------------------------------------------------
// 4. MultinomialNB — non-negative counts, alpha=1.0.
//
// Both `class_log_prior` and `feature_log_prob` are recoverable via the
// public `predict_joint_log_proba` because the formula is
// `jll(X) = X @ feature_log_prob.T + class_log_prior`:
//
//   jll(zeros)[:, c] = class_log_prior[c]
//   jll(eye[j])[j, c] - class_log_prior[c] = feature_log_prob[c, j]
// ---------------------------------------------------------------------------

#[test]
fn conformance_multinomial_nb() {
    let fx = load_fixture("multinomial_nb");
    let (rel, abs) = fx.tolerance(TOL_NAIVE_BAYES_REL, TOL_NAIVE_BAYES_ABS);

    let x = json_to_array2(&fx.input["X"]);
    let y = array1_to_usize(&json_to_array1(&fx.input["y"]));

    let alpha = fx.params["alpha"].as_f64().unwrap_or(1.0);

    let model = MultinomialNB::<f64>::new().with_alpha(alpha);
    let fitted = model.fit(&x, &y).expect("MultinomialNB fit");

    let preds = fitted.predict(&x).expect("MultinomialNB predict");
    let preds_i64: Vec<i64> = preds.iter().map(|&v| v as i64).collect();
    let expected_preds = json_to_labels(&fx.expected["predictions"]);
    assert_labels_equal(&preds_i64, &expected_preds, "MultinomialNB.predict");

    let acc = accuracy(&preds, &y);
    assert_close(
        acc,
        fx.expected["accuracy"].as_f64().unwrap(),
        rel,
        abs,
        "MultinomialNB.accuracy",
    );

    // Derive class_log_prior from jll on a single all-zero row.
    let expected_clp = json_to_array1(&fx.expected["class_log_prior"]);
    let n_features = x.ncols();
    let n_classes = expected_clp.len();
    let zeros = Array2::<f64>::zeros((1, n_features));
    let jll_zero = fitted
        .predict_joint_log_proba(&zeros)
        .expect("MultinomialNB predict_joint_log_proba(zeros)");
    let derived_clp: Vec<f64> = (0..n_classes).map(|c| jll_zero[[0, c]]).collect();
    assert_close_slice(
        &derived_clp,
        expected_clp.as_slice().unwrap(),
        rel,
        abs,
        "MultinomialNB.class_log_prior",
    );

    // Derive feature_log_prob from jll on the identity matrix.
    let expected_flp = json_to_array2(&fx.expected["feature_log_prob"]);
    let eye = eye_f64(n_features);
    let jll_eye = fitted
        .predict_joint_log_proba(&eye)
        .expect("MultinomialNB predict_joint_log_proba(eye)");
    let mut derived_flp = Array2::<f64>::zeros((n_classes, n_features));
    for ci in 0..n_classes {
        for j in 0..n_features {
            derived_flp[[ci, j]] = jll_eye[[j, ci]] - derived_clp[ci];
        }
    }
    assert_close_slice(
        derived_flp.as_slice().unwrap(),
        expected_flp.as_slice().unwrap(),
        rel,
        abs,
        "MultinomialNB.feature_log_prob",
    );
}
