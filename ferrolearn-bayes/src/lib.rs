//! # ferrolearn-bayes
//!
//! Bayesian methods for the ferrolearn machine learning framework.
//!
//! Two families of tools live here:
//!
//! 1. **Naive Bayes classifiers** — five variants for classification.
//! 2. **Conjugate priors** — closed-form posterior updates for parameter
//!    estimation. See [`conjugate`].
//!
//! This crate provides five Naive Bayes variants:
//!
//! - **[`GaussianNB`]** — Assumes Gaussian-distributed features. Suitable for
//!   continuous data.
//! - **[`MultinomialNB`]** — For discrete count data (e.g., word counts).
//!   Features must be non-negative.
//! - **[`BernoulliNB`]** — For binary/boolean features. Optional binarization
//!   threshold.
//! - **[`CategoricalNB`]** — For categorical features where each column takes
//!   on one of several discrete values. Laplace-smoothed.
//! - **[`ComplementNB`]** — A Multinomial NB variant that uses complement-class
//!   statistics; better suited for imbalanced datasets.
//!
//! # Design
//!
//! Each classifier follows the compile-time safety pattern:
//!
//! - The unfitted struct (e.g., `GaussianNB<F>`) holds hyperparameters and
//!   implements [`Fit`](ferrolearn_core::Fit).
//! - Calling `fit()` produces a fitted type (e.g., `FittedGaussianNB<F>`) that
//!   implements [`Predict`](ferrolearn_core::Predict) and
//!   [`HasClasses`](ferrolearn_core::introspection::HasClasses).
//! - The fitted types also expose a `predict_proba` method returning class
//!   probabilities as `Array2<F>`.
//!
//! # Examples
//!
//! ```
//! use ferrolearn_bayes::GaussianNB;
//! use ferrolearn_core::{Fit, Predict};
//! use ndarray::{array, Array2};
//!
//! let x = Array2::from_shape_vec(
//!     (6, 2),
//!     vec![1.0, 2.0, 1.5, 2.5, 1.2, 1.8, 6.0, 7.0, 5.8, 6.5, 6.2, 7.2],
//! ).unwrap();
//! let y = array![0usize, 0, 0, 1, 1, 1];
//!
//! let model = GaussianNB::<f64>::new();
//! let fitted = model.fit(&x, &y).unwrap();
//! let preds = fitted.predict(&x).unwrap();
//! assert_eq!(preds.len(), 6);
//! ```
//!
//! # `## REQ status`
//!
//! Binary (R-DEFER-2) for the crate-root RE-EXPORT BOUNDARY (this file is the public-API
//! surface + one cross-cutting helper, not an estimator). Mirrors `sklearn/naive_bayes.py`
//! `__all__` (`:30-36`) + `_BaseNB.predict_log_proba` (`:105-126`, `jll − logsumexp(jll,axis=1)`).
//! Design doc: `.design/bayes/lib.md`. Per-variant REQs live in the sibling modules' routed docs
//! (`.design/bayes/{gaussian,multinomial,bernoulli,categorical,complement,base}.md`). The
//! `clamp_alpha = base::check_alpha` re-export alias is owned by `base.rs` (REQ in `base.md`).
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (re-export boundary) | SHIPPED | the `pub use` block re-exports `BaseNB` + the 5 NB variants (`GaussianNB`/`MultinomialNB`/`BernoulliNB`/`CategoricalNB`/`ComplementNB` + their `Fitted*`), mirroring sklearn `naive_bayes.__all__` (`naive_bayes.py:30-36`). Consumers: meta-crate `pub use ferrolearn_bayes as bayes` + PyO3 pyclasses `RsGaussianNB` (`classifiers.rs`), `RsMultinomialNB`/`RsBernoulliNB`/`RsComplementNB` (`extras.rs`). Verification: `cargo test -p ferrolearn-bayes` green + `tests/api_proof.rs`. |
//! | REQ-2 (`log_softmax_rows` == jll − logsumexp(jll)) | SHIPPED | `pub(crate) fn log_softmax_rows` is the numerically stable (max-subtraction) form of `_BaseNB.predict_log_proba` (`naive_bayes.py:105-126`; `logsumexp` = `scipy.special.logsumexp`). Consumers: `BaseNB::nb_predict_log_proba in base.rs` + every `Fitted*NB::predict_log_proba`/`predict_proba`. Critic-verified vs live oracle: `green_log_softmax_*` in `tests/divergence_lib.rs` (4 green) — GaussianNB end-to-end ~1e-12, all-`-inf` row → NaN (matches scipy), single-col → 0.0, large-magnitude `[[1000,1001]]` finite (no `exp` overflow). |
//! | REQ-substrate (ferray) | NOT-STARTED | open prereq blocker #1110. `log_softmax_rows` + the boundary run on `ndarray::Array2` + `num_traits::Float`, not `ferray-core`/`ferray-ufunc` (R-SUBSTRATE-1); the helper migrates with the NB variants' per-variant blockers (#898/#903/#910/#917/#925). |

pub mod base;
pub mod bernoulli;
pub mod categorical;
pub mod complement;
pub mod conjugate;
pub mod gaussian;
pub mod multinomial;

// Re-export all public types at the crate root.
pub use base::BaseNB;
pub use bernoulli::{BernoulliNB, FittedBernoulliNB};
pub use categorical::{CategoricalNB, FittedCategoricalNB};
pub use complement::{ComplementNB, FittedComplementNB};
pub use gaussian::{FittedGaussianNB, GaussianNB};
pub use multinomial::{FittedMultinomialNB, MultinomialNB};

// The `_BaseDiscreteNB._check_alpha` smoothing floor lives in `base.rs`
// (re-homed from the former `clamp_alpha`). Re-export under the original
// `clamp_alpha` name so the discrete variants' fit call sites are unchanged.
pub(crate) use base::check_alpha as clamp_alpha;

use ndarray::Array2;
use num_traits::Float;

/// Numerically stable row-wise log-softmax: returns `jll - logsumexp(jll, axis=1)`.
///
/// Used by every Fitted*NB to convert joint log-likelihoods into log
/// probabilities. The subtraction-of-row-max trick keeps the exponentials
/// bounded by 1, avoiding overflow.
pub(crate) fn log_softmax_rows<F: Float>(jll: &Array2<F>) -> Array2<F> {
    let n_samples = jll.nrows();
    let n_classes = jll.ncols();
    let mut log_proba = Array2::<F>::zeros((n_samples, n_classes));
    for i in 0..n_samples {
        let max_score = jll.row(i).iter().fold(F::neg_infinity(), |a, &b| a.max(b));
        let mut sum_exp = F::zero();
        for ci in 0..n_classes {
            sum_exp = sum_exp + (jll[[i, ci]] - max_score).exp();
        }
        let log_norm = max_score + sum_exp.ln();
        for ci in 0..n_classes {
            log_proba[[i, ci]] = jll[[i, ci]] - log_norm;
        }
    }
    log_proba
}
