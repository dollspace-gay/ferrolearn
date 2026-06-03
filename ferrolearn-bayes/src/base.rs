//! Shared naive-Bayes base — `_BaseNB` / `_BaseDiscreteNB` analogs.
//!
//! This module mirrors scikit-learn's abstract naive-Bayes class hierarchy
//! (`sklearn/naive_bayes.py`): the abstract `_BaseNB` (the prediction pipeline
//! shared by every NB variant) and the `_BaseDiscreteNB` smoothing helper
//! `_check_alpha`. ferrolearn expresses the `_BaseNB` contract as the
//! [`BaseNB`] trait whose provided methods implement the exact prediction
//! pipeline — argmax over the joint log-likelihood, the `jll - logsumexp(jll)`
//! log-probability normalization, and `exp(...)` for probabilities — leaving
//! only `joint_log_likelihood` (sklearn's abstract `_joint_log_likelihood`)
//! and `nb_classes` (the `classes_` attribute) for each variant to provide.
//!
//! The five Naive Bayes variants ([`crate::GaussianNB`],
//! [`crate::MultinomialNB`], [`crate::BernoulliNB`], [`crate::ComplementNB`],
//! [`crate::CategoricalNB`]) implement [`BaseNB`] for their fitted types and
//! delegate their inherent `predict_proba` / `predict_log_proba` /
//! `predict_joint_log_proba` methods and their [`Predict`](ferrolearn_core::Predict)
//! impls to the trait defaults. They are the non-test production consumers of
//! this base (R-DEFER-1).
//!
//! # `## REQ status`
//!
//! Binary classification (R-DEFER-2): SHIPPED needs impl + a non-test
//! production consumer + green verification. The non-test consumers are the
//! five `Fitted*NB` types whose predict pipeline delegates here; the green
//! verification is the existing in-tree variant test suite (91 lib tests),
//! which exercises the delegated pipeline unchanged, plus the live sklearn
//! oracle sanity-check below. Cites use symbol anchors (ferrolearn) /
//! `file:line` (sklearn 1.5.2, commit 156ef14). Live oracle = installed
//! sklearn 1.5.2.
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (`_BaseNB.predict` — `classes_[argmax(jll)]`) | SHIPPED | provided method `BaseNB::nb_predict` (per-row argmax over `joint_log_likelihood`, first-max/smallest-index tie-break) mirrors `_BaseNB.predict` (`sklearn/naive_bayes.py:103`, `self.classes_[np.argmax(jll, axis=1)]`). Non-test consumers: `impl Predict for FittedGaussianNB`/`FittedMultinomialNB`/`FittedBernoulliNB`/`FittedComplementNB`/`FittedCategoricalNB`'s `fn predict` delegate to `BaseNB::nb_predict`. Verified: live oracle `MultinomialNB().fit([[1,2],[0,3],[4,0],[3,1]],[0,0,1,1]).predict([[2,2]])` → `[0]`; ferrolearn matches; the 91 in-tree variant tests stay green. |
//! | REQ-2 (`_BaseNB.predict_log_proba` — `jll - logsumexp(jll)`) | SHIPPED | provided method `BaseNB::nb_predict_log_proba` (calls `crate::log_softmax_rows`) mirrors `_BaseNB.predict_log_proba` (`sklearn/naive_bayes.py:123-126`, `jll - np.atleast_2d(logsumexp(jll, axis=1)).T`). Non-test consumers: each variant's `pub fn predict_log_proba` delegates here. Verified: live oracle predict_log_proba `[[2,2]]` → `[[-0.5470675457484475, -0.8642776061017265]]`; ferrolearn matches to ~1e-12. |
//! | REQ-3 (`_BaseNB.predict_proba` — `exp(predict_log_proba)`) | SHIPPED | provided method `BaseNB::nb_predict_proba` (`exp` of `nb_predict_log_proba`) mirrors `_BaseNB.predict_proba` (`sklearn/naive_bayes.py:144`, `np.exp(self.predict_log_proba(X))`). Non-test consumers: each variant's `pub fn predict_proba` (the rows-sum-to-1 softmax) — value-identical to `exp(jll - logsumexp)`. Verified: live oracle predict_proba `[[2,2]]` → `[[0.5786441724102462, 0.4213558275897536]]`; ferrolearn matches; `*_predict_proba_sums_to_one` tests stay green. |
//! | REQ-4 (`_BaseDiscreteNB._check_alpha` — floor 1e-10 unless `force_alpha`) | SHIPPED | `check_alpha` (re-homed from `lib.rs::clamp_alpha`) mirrors `_BaseDiscreteNB._check_alpha` (`sklearn/naive_bayes.py:604-626`: `alpha_lower_bound = 1e-10`; `np.maximum(alpha, alpha_lower_bound)` when `alpha_min < alpha_lower_bound and not self.force_alpha`). Non-test consumers: `MultinomialNB`/`BernoulliNB`/`ComplementNB`/`CategoricalNB` `fn fit` call `crate::clamp_alpha` (re-exported `pub(crate) use base::check_alpha as clamp_alpha`). Verified: `*_alpha_smoothing_effect` and `*_default` tests stay green. |
//! | REQ-5 (`_BaseDiscreteNB.coef_` / `intercept_`) | NOT-STARTED | open prereq blocker. sklearn exposes `coef_ = feature_log_prob_[1:]` (binary collapses to one row) and `intercept_ = class_log_prior_[1:]` (`sklearn/naive_bayes.py` `_BaseDiscreteNB.coef_`/`intercept_` properties). No ferrolearn analog: the discrete variants store `log_theta`/`log_prob`/`weights` and `log_prior` but expose no `coef_`/`intercept_` accessor on the fitted types or the trait. |
//! | REQ-6 (`_BaseDiscreteNB.partial_fit` / `_count` / `_update_feature_log_prob`) | NOT-STARTED | open prereq blocker. sklearn factors fitting through abstract `_count`/`_update_feature_log_prob` driven by a shared `partial_fit` (`sklearn/naive_bayes.py:628-709`). ferrolearn implements `partial_fit` per variant (each `Fitted*NB::partial_fit`) with no shared `_count`/`_update_feature_log_prob` seam on this base; this base covers only the predict pipeline + `_check_alpha`. |
//! | REQ-7 (`_BaseDiscreteNB._update_class_log_prior` / `class_prior` handling) | NOT-STARTED | open prereq blocker. sklearn's `_update_class_log_prior` (`sklearn/naive_bayes.py:580-602`) centralizes empirical-vs-uniform-vs-explicit prior selection driven by `fit_prior`/`class_prior`. ferrolearn duplicates this logic per discrete variant (`fit`/`partial_fit` prior blocks); it is not lifted onto this base. |

use ferrolearn_core::error::FerroError;
use ndarray::{Array1, Array2};
use num_traits::Float;

/// Smoothing-floor mirroring scikit-learn `_BaseDiscreteNB._check_alpha`
/// (`sklearn/naive_bayes.py:604-626`).
///
/// When `force_alpha = false` and `alpha < 1e-10`, the alpha is raised to the
/// `1e-10` lower bound (sklearn's legacy "alpha too small will result in
/// numeric errors" guard, `np.maximum(alpha, alpha_lower_bound)`). When
/// `force_alpha = true`, the user-supplied alpha is returned unchanged, even
/// if zero.
///
/// This is the re-homed `clamp_alpha`; `lib.rs` re-exports it as `clamp_alpha`
/// so the discrete variants' call sites are unchanged.
#[must_use]
pub(crate) fn check_alpha<F: Float>(alpha: F, force_alpha: bool) -> F {
    if force_alpha {
        alpha
    } else {
        let floor = F::from(1e-10).unwrap_or_else(F::epsilon);
        if alpha < floor { floor } else { alpha }
    }
}

/// Shared naive-Bayes prediction pipeline — the `_BaseNB` analog.
///
/// Mirrors scikit-learn's abstract `_BaseNB` (`sklearn/naive_bayes.py`). An
/// implementor supplies the two abstract pieces — [`joint_log_likelihood`]
/// (sklearn's abstract `_joint_log_likelihood`, the unnormalized
/// `log P(c) + log P(x|c)`) and [`nb_classes`] (the sorted `classes_`) — and
/// gets the full prediction pipeline for free:
///
/// - [`nb_predict`] — `classes_[argmax(jll)]` (`sklearn/naive_bayes.py:103`),
/// - [`nb_predict_log_proba`] — `jll - logsumexp(jll)`
///   (`sklearn/naive_bayes.py:123-126`),
/// - [`nb_predict_proba`] — `exp(predict_log_proba)`
///   (`sklearn/naive_bayes.py:144`),
/// - [`nb_predict_joint_log_proba`] — the unnormalized joint log-probability
///   (`sklearn/naive_bayes.py:62-84`).
///
/// [`joint_log_likelihood`]: BaseNB::joint_log_likelihood
/// [`nb_classes`]: BaseNB::nb_classes
/// [`nb_predict`]: BaseNB::nb_predict
/// [`nb_predict_log_proba`]: BaseNB::nb_predict_log_proba
/// [`nb_predict_proba`]: BaseNB::nb_predict_proba
/// [`nb_predict_joint_log_proba`]: BaseNB::nb_predict_joint_log_proba
pub trait BaseNB<F: Float> {
    /// Compute the unnormalized joint log-likelihood `log P(c) + log P(x|c)`.
    ///
    /// Abstract — mirrors sklearn `_BaseNB._joint_log_likelihood`
    /// (`sklearn/naive_bayes.py:42-53`). Returns shape
    /// `(n_samples, n_classes)`; column `c` corresponds to the class in
    /// [`nb_classes`](BaseNB::nb_classes) at index `c`.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`](ferrolearn_core::error::FerroError)
    /// if the feature count of `x` does not match the fitted model.
    fn joint_log_likelihood(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError>;

    /// The sorted class labels — the `classes_` attribute.
    fn nb_classes(&self) -> &[usize];

    /// Predict class labels: `classes_[argmax(jll, axis=1)]`.
    ///
    /// Mirrors sklearn `_BaseNB.predict` (`sklearn/naive_bayes.py:86-103`,
    /// `return self.classes_[np.argmax(jll, axis=1)]`). The argmax uses
    /// `np.argmax`'s **first-max** rule — on ties the smallest column index
    /// wins, and because `classes_` is sorted that is the smallest class
    /// label. The scan replaces the running best only on a strict `>`,
    /// reproducing first-max exactly.
    ///
    /// # Errors
    ///
    /// Propagates any error from [`joint_log_likelihood`](BaseNB::joint_log_likelihood).
    fn nb_predict(&self, x: &Array2<F>) -> Result<Array1<usize>, FerroError> {
        let scores = self.joint_log_likelihood(x)?;
        let classes = self.nb_classes();
        let n_samples = scores.nrows();
        let n_classes = scores.ncols();

        let mut predictions = Array1::<usize>::zeros(n_samples);
        for i in 0..n_samples {
            // First-max argmax (np.argmax): start at column 0, replace only on
            // a strict greater-than so ties keep the earlier (smaller) index.
            let best_class = (1..n_classes).fold(0usize, |best, ci| {
                match scores[[i, ci]].partial_cmp(&scores[[i, best]]) {
                    Some(core::cmp::Ordering::Greater) => ci,
                    _ => best,
                }
            });
            predictions[i] = classes[best_class];
        }
        Ok(predictions)
    }

    /// Return log-probability estimates: `jll - logsumexp(jll, axis=1)`.
    ///
    /// Mirrors sklearn `_BaseNB.predict_log_proba`
    /// (`sklearn/naive_bayes.py:105-126`). Computed via
    /// `crate::log_softmax_rows`, the numerically stable row-wise
    /// log-softmax, so the result is bit-identical to the variants' prior
    /// inherent implementation.
    ///
    /// # Errors
    ///
    /// Propagates any error from [`joint_log_likelihood`](BaseNB::joint_log_likelihood).
    fn nb_predict_log_proba(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let jll = self.joint_log_likelihood(x)?;
        Ok(crate::log_softmax_rows(&jll))
    }

    /// Return probability estimates: `exp(predict_log_proba)`.
    ///
    /// Mirrors sklearn `_BaseNB.predict_proba` (`sklearn/naive_bayes.py:128-144`,
    /// `np.exp(self.predict_log_proba(X))`). Each row sums to 1.
    ///
    /// # Errors
    ///
    /// Propagates any error from [`joint_log_likelihood`](BaseNB::joint_log_likelihood).
    fn nb_predict_proba(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let mut log_proba = self.nb_predict_log_proba(x)?;
        log_proba.mapv_inplace(|v| v.exp());
        Ok(log_proba)
    }

    /// Return the unnormalized joint log-probability estimates.
    ///
    /// Mirrors sklearn `_BaseNB.predict_joint_log_proba`
    /// (`sklearn/naive_bayes.py:62-84`, returns `self._joint_log_likelihood(X)`).
    ///
    /// # Errors
    ///
    /// Propagates any error from [`joint_log_likelihood`](BaseNB::joint_log_likelihood).
    fn nb_predict_joint_log_proba(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        self.joint_log_likelihood(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// A minimal `BaseNB` implementor over a fixed joint-log-likelihood matrix,
    /// used to exercise the provided pipeline methods in isolation.
    struct StubNB {
        classes: Vec<usize>,
        jll: Array2<f64>,
    }

    impl BaseNB<f64> for StubNB {
        fn joint_log_likelihood(&self, _x: &Array2<f64>) -> Result<Array2<f64>, FerroError> {
            Ok(self.jll.clone())
        }
        fn nb_classes(&self) -> &[usize] {
            &self.classes
        }
    }

    #[test]
    fn test_nb_predict_argmax_to_classes() {
        // Row 0: class 0 wins; row 1: class 1 wins.
        let stub = StubNB {
            classes: vec![3, 7],
            jll: array![[1.0, 0.5], [0.2, 0.9]],
        };
        let x = Array2::<f64>::zeros((2, 1));
        let preds = stub.nb_predict(&x).unwrap();
        assert_eq!(preds[0], 3);
        assert_eq!(preds[1], 7);
    }

    #[test]
    fn test_nb_predict_tie_breaks_to_smallest_index() {
        // Equal scores -> np.argmax first-max -> column 0 -> smallest label.
        let stub = StubNB {
            classes: vec![5, 9],
            jll: array![[2.0, 2.0]],
        };
        let x = Array2::<f64>::zeros((1, 1));
        let preds = stub.nb_predict(&x).unwrap();
        assert_eq!(preds[0], 5);
    }

    #[test]
    fn test_nb_predict_proba_is_exp_of_log_proba_and_sums_to_one() {
        let stub = StubNB {
            classes: vec![0, 1],
            jll: array![[1.0, -0.5], [0.0, 0.0]],
        };
        let x = Array2::<f64>::zeros((2, 1));
        let log_proba = stub.nb_predict_log_proba(&x).unwrap();
        let proba = stub.nb_predict_proba(&x).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                assert!((proba[[i, j]] - log_proba[[i, j]].exp()).abs() < 1e-15);
            }
            assert!((proba.row(i).sum() - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn test_nb_predict_joint_log_proba_is_jll() {
        let jll = array![[1.0, 2.0], [3.0, 4.0]];
        let stub = StubNB {
            classes: vec![0, 1],
            jll: jll.clone(),
        };
        let x = Array2::<f64>::zeros((2, 1));
        let out = stub.nb_predict_joint_log_proba(&x).unwrap();
        assert_eq!(out, jll);
    }

    #[test]
    fn test_check_alpha_force_alpha_keeps_value() {
        // force_alpha = true -> returned unchanged, even below the floor.
        assert_eq!(check_alpha::<f64>(0.0, true), 0.0);
        assert_eq!(check_alpha::<f64>(1e-20, true), 1e-20);
    }

    #[test]
    fn test_check_alpha_floors_when_not_forced() {
        // force_alpha = false -> raised to the 1e-10 lower bound
        // (sklearn naive_bayes.py:618 alpha_lower_bound = 1e-10).
        assert_eq!(check_alpha::<f64>(0.0, false), 1e-10);
        assert_eq!(check_alpha::<f64>(1e-20, false), 1e-10);
        // Above the floor -> unchanged.
        assert_eq!(check_alpha::<f64>(1.0, false), 1.0);
    }
}
