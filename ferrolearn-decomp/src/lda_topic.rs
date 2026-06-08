//! Latent Dirichlet Allocation (LDA) topic model.
//!
//! [`LatentDirichletAllocation`] discovers latent topics in a document-term
//! matrix using variational inference. This is the *topic model* LDA, **not**
//! Linear Discriminant Analysis (which lives in `ferrolearn-linear`).
//!
//! # Algorithm
//!
//! Two solvers are supported:
//!
//! - **Batch** variational EM: iterates over the full corpus each step.
//!   E-step updates per-document topic distributions; M-step updates the
//!   global topic-word distributions.
//! - **Online** variational Bayes (Hoffman et al. 2010): processes mini-batches
//!   and uses a decaying learning rate to update global parameters
//!   incrementally.
//!
//! # Examples
//!
//! ```
//! use ferrolearn_decomp::LatentDirichletAllocation;
//! use ferrolearn_core::traits::{Fit, Transform};
//! use ndarray::array;
//!
//! // Simple 4-document, 6-word corpus
//! let dtm = array![
//!     [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
//!     [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
//!     [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
//!     [0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
//! ];
//! let lda = LatentDirichletAllocation::new(2).with_random_state(42);
//! let fitted = lda.fit(&dtm, &()).unwrap();
//! let topics = fitted.transform(&dtm).unwrap();
//! assert_eq!(topics.dim(), (4, 2));
//! ```
//!
//! ## REQ status
//!
//! Design: `.design/decomp/lda_topic.md`. Tracking: #1540. Each REQ is BINARY —
//! SHIPPED (impl + non-test consumer + tests + green verification) or NOT-STARTED
//! (concrete open blocker). Non-test consumer: crate re-export (`lib.rs:91`); there
//! is NO PyO3 binding. Oracle = live sklearn 1.5.2 (`_lda.py`,
//! `class LatentDirichletAllocation` — topic model, NOT LDA-discriminant), run from
//! `/tmp` (R-CHAR-3). ferrolearn is a SIMPLIFIED f64-only variational Bayes
//! reimplementation; exact component/topic VALUES are a carve-out (Uniform+beta init
//! vs sklearn Gamma(100,0.01) + numpy RNG).
//!
//! | REQ | Scope | Status | Evidence / Blocker |
//! |---|---|---|---|
//! | REQ-1 | Structural: `components_` shape `(n_topics,n_words)`, `n_iter_`==max_iter, seed-determinism, digamma accuracy | SHIPPED (scoped) | `fit` stores `components_=lambda` (`:440`), `n_iter_=max_iter` (`:443`, matches sklearn default `evaluate_every=-1` `_lda.py:695`); `digamma` (`:277`) matches scipy.special.psi ~1.17e-10; green-guards + in-module tests. STRUCTURAL, NOT values (REQ-4) |
//! | REQ-2 | `components_` non-negativity | SHIPPED | M-step adds non-negative suff-stats to non-negative init; `test_lda_components_non_negative` + green-guard |
//! | REQ-3 | transform doc-topic shape + each row sums to 1 + topic separation + error contracts (incl. NON-FINITE rejection) | SHIPPED (scoped) | `transform` normalizes gamma rows (= sklearn `_lda.py:745`); fit/transform guards. NON-FINITE: `fit`+`transform` call `reject_non_finite` (`lda_topic.rs` symbol `reject_non_finite`) BEFORE the non-negativity check and the VB iterations, returning the CLEAN finiteness `InvalidParameter{name:"X", reason:"Input X contains NaN or infinity."}` = sklearn `_check_non_neg_array`'s `_validate_data(force_all_finite=True)` finiteness-before-non-negativity (`_lda.py:566` before `:572`, `utils/validation.py:147-154`). `tests/divergence_nonfinite_spillover.rs::divergence_lda_fit_nan` matches the live sklearn 1.5.2 oracle (#2290). FLAG: sklearn raises `ValueError`, defaults n_components=10, doesn't pre-reject 0 words |
//! | REQ-4 | EXACT `components_` value parity | NOT-STARTED | CARVE-OUT (R-DEFER-3): Uniform+beta/Xoshiro init vs Gamma(100,0.01)/numpy RandomState VI (`_lda.py:419-421`) — blocker #1541 |
//! | REQ-5 | transform doc-topic VALUE parity | NOT-STARTED | CARVE-OUT, folds into REQ-4 (downstream of components_, no injectable API) — blocker #1542 |
//! | REQ-6 | Gamma(100,0.01) init (components + per-doc gamma) | NOT-STARTED | sklearn `_lda.py:96-99,:419-421` — blocker #1543 |
//! | REQ-7 | `exp_dirichlet_component_` representation/attr | NOT-STARTED | sklearn `_lda.py:424`; ferrolearn log-space on the fly — blocker #1544 |
//! | REQ-8 | `perplexity`/`score`/`_approx_bound` | NOT-STARTED | sklearn `_lda.py:748,:827,:896` — blocker #1545 |
//! | REQ-9 | `evaluate_every`/`perp_tol` perplexity early stop | NOT-STARTED | sklearn `_lda.py:676-691`; ferrolearn fixed max_iter loop — blocker #1546 |
//! | REQ-10 | `batch_size`/`total_samples` online mini-batching | NOT-STARTED | sklearn `_lda.py:662,:535-538`; ferrolearn batch-of-1 — blocker #1547 |
//! | REQ-11 | fitted attrs `n_features_in_`/`bound_` | NOT-STARTED | sklearn `_lda.py:701-703` — blocker #1548 |
//! | REQ-12 | `n_jobs`/`verbose` | NOT-STARTED | sklearn `_lda.py:378-379` — blocker #1549 |
//! | REQ-13 | generic `F` (f32+f64) | NOT-STARTED | f64-only — blocker #1550 |
//! | REQ-14 | PyO3 binding | NOT-STARTED | absent; only consumer re-export `lib.rs:91` — blocker #1551 |
//! | REQ-15 | ferray substrate | NOT-STARTED | `ndarray`+`rand`+hand-rolled digamma — blocker #1552 |
//!
//! Count: **3 SHIPPED (REQ-1,2,3) / 12 NOT-STARTED (REQ-4..15)**.

use ferrolearn_core::error::FerroError;
use ferrolearn_core::traits::{Fit, Transform};
use ndarray::Array2;
use rand::SeedableRng;
use rand_distr::{Distribution, Uniform};
use rand_xoshiro::Xoshiro256PlusPlus;

/// Reject non-finite input the way sklearn's `_check_non_neg_array` does.
///
/// sklearn's `LatentDirichletAllocation` runs `_check_non_neg_array` which calls
/// `_validate_data` with the default `force_all_finite=True`
/// (`sklearn/decomposition/_lda.py:566`) BEFORE the non-negativity check
/// (`check_non_negative`, `:572`) and any variational-Bayes math, raising
/// `ValueError("Input X contains NaN.")` / `"... contains infinity ..."`
/// (`sklearn/utils/validation.py:147-154`). NaN AND infinity are both rejected,
/// finiteness BEFORE non-negativity. The message names "NaN" and "infinity" to
/// mirror sklearn's `ValueError`. Never panics (R-CODE-2).
fn reject_non_finite(x: &Array2<f64>) -> Result<(), FerroError> {
    if x.iter().any(|v| !v.is_finite()) {
        return Err(FerroError::InvalidParameter {
            name: "X".into(),
            reason: "Input X contains NaN or infinity.".into(),
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Learning method enum
// ---------------------------------------------------------------------------

/// The learning method for LDA.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LdaLearningMethod {
    /// Batch variational EM — iterates over the full corpus each step.
    Batch,
    /// Online variational Bayes (Hoffman et al. 2010).
    Online,
}

// ---------------------------------------------------------------------------
// LatentDirichletAllocation (unfitted)
// ---------------------------------------------------------------------------

/// Latent Dirichlet Allocation configuration.
///
/// Holds hyperparameters for the LDA topic model. Calling [`Fit::fit`]
/// learns topic-word distributions and returns a
/// [`FittedLatentDirichletAllocation`].
#[derive(Debug, Clone)]
pub struct LatentDirichletAllocation {
    /// Number of topics to extract.
    n_components: usize,
    /// Maximum number of E-M iterations (batch) or passes (online).
    max_iter: usize,
    /// Learning method.
    learning_method: LdaLearningMethod,
    /// Offset for learning rate in online mode (default 10.0).
    learning_offset: f64,
    /// Decay for learning rate in online mode (default 0.7).
    learning_decay: f64,
    /// Document-topic prior (Dirichlet alpha). None = 1/n_components.
    doc_topic_prior: Option<f64>,
    /// Topic-word prior (Dirichlet beta). None = 1/n_components.
    topic_word_prior: Option<f64>,
    /// Maximum E-step iterations per document.
    max_doc_update_iter: usize,
    /// Optional random seed.
    random_state: Option<u64>,
}

impl LatentDirichletAllocation {
    /// Create a new `LatentDirichletAllocation` with `n_components` topics.
    ///
    /// Defaults: `max_iter=10`, `learning_method=Batch`,
    /// `learning_offset=10.0`, `learning_decay=0.7`,
    /// priors=`1/n_components`, `max_doc_update_iter=100`.
    #[must_use]
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            max_iter: 10,
            learning_method: LdaLearningMethod::Batch,
            learning_offset: 10.0,
            learning_decay: 0.7,
            doc_topic_prior: None,
            topic_word_prior: None,
            max_doc_update_iter: 100,
            random_state: None,
        }
    }

    /// Set the maximum number of iterations.
    #[must_use]
    pub fn with_max_iter(mut self, n: usize) -> Self {
        self.max_iter = n;
        self
    }

    /// Set the learning method.
    #[must_use]
    pub fn with_learning_method(mut self, m: LdaLearningMethod) -> Self {
        self.learning_method = m;
        self
    }

    /// Set the learning offset (online mode).
    #[must_use]
    pub fn with_learning_offset(mut self, v: f64) -> Self {
        self.learning_offset = v;
        self
    }

    /// Set the learning decay (online mode).
    #[must_use]
    pub fn with_learning_decay(mut self, v: f64) -> Self {
        self.learning_decay = v;
        self
    }

    /// Set the document-topic prior (alpha).
    #[must_use]
    pub fn with_doc_topic_prior(mut self, v: f64) -> Self {
        self.doc_topic_prior = Some(v);
        self
    }

    /// Set the topic-word prior (beta).
    #[must_use]
    pub fn with_topic_word_prior(mut self, v: f64) -> Self {
        self.topic_word_prior = Some(v);
        self
    }

    /// Set the random seed.
    #[must_use]
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Set the maximum E-step iterations per document.
    #[must_use]
    pub fn with_max_doc_update_iter(mut self, n: usize) -> Self {
        self.max_doc_update_iter = n;
        self
    }

    /// Return the configured number of topics.
    #[must_use]
    pub fn n_components(&self) -> usize {
        self.n_components
    }

    /// Return the configured maximum iterations.
    #[must_use]
    pub fn max_iter(&self) -> usize {
        self.max_iter
    }

    /// Return the configured learning method.
    #[must_use]
    pub fn learning_method(&self) -> LdaLearningMethod {
        self.learning_method
    }

    /// Return the configured learning offset.
    #[must_use]
    pub fn learning_offset(&self) -> f64 {
        self.learning_offset
    }

    /// Return the configured learning decay.
    #[must_use]
    pub fn learning_decay(&self) -> f64 {
        self.learning_decay
    }

    /// Return the configured document-topic prior, if explicitly set.
    #[must_use]
    pub fn doc_topic_prior(&self) -> Option<f64> {
        self.doc_topic_prior
    }

    /// Return the configured topic-word prior, if explicitly set.
    #[must_use]
    pub fn topic_word_prior(&self) -> Option<f64> {
        self.topic_word_prior
    }

    /// Return the configured random state, if any.
    #[must_use]
    pub fn random_state(&self) -> Option<u64> {
        self.random_state
    }
}

// ---------------------------------------------------------------------------
// FittedLatentDirichletAllocation
// ---------------------------------------------------------------------------

/// A fitted LDA model holding the learned topic-word distributions.
///
/// Created by calling [`Fit::fit`] on a [`LatentDirichletAllocation`].
/// Implements [`Transform<Array2<f64>>`] to compute document-topic
/// distributions for new documents.
#[derive(Debug, Clone)]
pub struct FittedLatentDirichletAllocation {
    /// Topic-word distribution (un-normalised), shape `(n_topics, n_words)`.
    /// The `components_[k][w]` entry is proportional to the probability
    /// of word `w` in topic `k`.
    components_: Array2<f64>,
    /// Document-topic prior (alpha).
    alpha_: f64,
    /// Topic-word prior (beta).
    beta_: f64,
    /// Number of iterations performed.
    n_iter_: usize,
    /// Maximum E-step iterations per document.
    max_doc_update_iter_: usize,
}

impl FittedLatentDirichletAllocation {
    /// Topic-word distribution, shape `(n_topics, n_words)`.
    ///
    /// Each row is a (possibly un-normalised) distribution over the
    /// vocabulary for one topic.
    #[must_use]
    pub fn components(&self) -> &Array2<f64> {
        &self.components_
    }

    /// Number of iterations performed during fitting.
    #[must_use]
    pub fn n_iter(&self) -> usize {
        self.n_iter_
    }

    /// The document-topic prior used during fitting.
    #[must_use]
    pub fn alpha(&self) -> f64 {
        self.alpha_
    }

    /// The topic-word prior used during fitting.
    #[must_use]
    pub fn beta(&self) -> f64 {
        self.beta_
    }
}

// ---------------------------------------------------------------------------
// Internal: digamma approximation
// ---------------------------------------------------------------------------

/// Approximate digamma function (psi) using the asymptotic expansion.
///
/// For x >= 6 uses the series; for x < 6 uses the recurrence
/// psi(x) = psi(x+1) - 1/x.
fn digamma(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::NAN;
    }
    let mut val = x;
    let mut result = 0.0;
    // Use recurrence to bring val >= 6.
    while val < 6.0 {
        result -= 1.0 / val;
        val += 1.0;
    }
    // Asymptotic expansion.
    result += val.ln() - 0.5 / val;
    let inv2 = 1.0 / (val * val);
    result -=
        inv2 * (1.0 / 12.0 - inv2 * (1.0 / 120.0 - inv2 * (1.0 / 252.0 - inv2 * 1.0 / 240.0)));
    result
}

/// Compute the E-step for a single document.
///
/// Given the document word counts `doc` (length V) and the current
/// topic-word log expectations `e_log_beta` (shape K x V), compute the
/// variational parameters `gamma` (length K, document-topic).
///
/// Returns the gamma vector (un-normalised document-topic distribution).
fn e_step_doc(doc: &[f64], e_log_beta: &Array2<f64>, alpha: f64, max_iter: usize) -> Vec<f64> {
    let n_topics = e_log_beta.nrows();
    let n_words = e_log_beta.ncols();

    // Initialise gamma uniformly.
    let mut gamma = vec![alpha + (n_words as f64) / (n_topics as f64); n_topics];

    for _iter in 0..max_iter {
        let e_log_theta: Vec<f64> = gamma.iter().map(|&g| digamma(g)).collect();
        let gamma_sum_dig = digamma(gamma.iter().sum::<f64>());

        let mut new_gamma = vec![alpha; n_topics];

        for w in 0..n_words {
            if doc[w] < 1e-16 {
                continue;
            }
            // Compute log of un-normalised phi for each topic.
            let mut log_phi = Vec::with_capacity(n_topics);
            let mut max_log = f64::NEG_INFINITY;
            for k in 0..n_topics {
                let v = e_log_theta[k] - gamma_sum_dig + e_log_beta[[k, w]];
                log_phi.push(v);
                if v > max_log {
                    max_log = v;
                }
            }
            // Normalise in log space.
            let mut sum_phi = 0.0;
            let mut phi = Vec::with_capacity(n_topics);
            for lp in &log_phi {
                let p = (lp - max_log).exp();
                phi.push(p);
                sum_phi += p;
            }
            if sum_phi < 1e-16 {
                sum_phi = 1e-16;
            }
            for k in 0..n_topics {
                new_gamma[k] += doc[w] * phi[k] / sum_phi;
            }
        }

        // Check convergence.
        let mut diff = 0.0;
        for k in 0..n_topics {
            diff += (new_gamma[k] - gamma[k]).abs();
        }
        gamma = new_gamma;
        if diff < 1e-3 {
            break;
        }
    }

    gamma
}

// ---------------------------------------------------------------------------
// Trait implementations
// ---------------------------------------------------------------------------

impl Fit<Array2<f64>, ()> for LatentDirichletAllocation {
    type Fitted = FittedLatentDirichletAllocation;
    type Error = FerroError;

    /// Fit the LDA model on a document-term matrix.
    ///
    /// # Errors
    ///
    /// - [`FerroError::InvalidParameter`] if `n_components` is zero or
    ///   any entry of the input is negative.
    /// - [`FerroError::InsufficientSamples`] if there are zero documents or
    ///   zero words.
    fn fit(&self, x: &Array2<f64>, _y: &()) -> Result<FittedLatentDirichletAllocation, FerroError> {
        let (n_docs, n_words) = x.dim();

        // Validate.
        if self.n_components == 0 {
            return Err(FerroError::InvalidParameter {
                name: "n_components".into(),
                reason: "must be at least 1".into(),
            });
        }
        if n_docs == 0 {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "LatentDirichletAllocation::fit".into(),
            });
        }
        if n_words == 0 {
            return Err(FerroError::InvalidParameter {
                name: "X".into(),
                reason: "document-term matrix must have at least 1 word".into(),
            });
        }
        // Reject NaN/Inf BEFORE the non-negativity check and the VB iterations
        // (sklearn's `_check_non_neg_array` runs `_validate_data(force_all_finite
        // =True)` at `_lda.py:566` before `check_non_negative` at `:572`,
        // `utils/validation.py:147-154`).
        reject_non_finite(x)?;
        for &val in x {
            if val < 0.0 {
                return Err(FerroError::InvalidParameter {
                    name: "X".into(),
                    reason: "LDA requires non-negative entries in the document-term matrix".into(),
                });
            }
        }

        let n_topics = self.n_components;
        let alpha = self.doc_topic_prior.unwrap_or(1.0 / n_topics as f64);
        let beta = self.topic_word_prior.unwrap_or(1.0 / n_topics as f64);
        let seed = self.random_state.unwrap_or(0);

        // Initialise lambda (topic-word variational parameters) randomly.
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        let uniform = Uniform::new(0.5, 1.5).unwrap();
        let mut lambda = Array2::<f64>::zeros((n_topics, n_words));
        for elem in &mut lambda {
            *elem = uniform.sample(&mut rng) + beta;
        }

        match self.learning_method {
            LdaLearningMethod::Batch => {
                self.fit_batch(x, &mut lambda, alpha, beta, n_docs, n_words, n_topics);
            }
            LdaLearningMethod::Online => {
                self.fit_online(
                    x,
                    &mut lambda,
                    alpha,
                    beta,
                    n_docs,
                    n_words,
                    n_topics,
                    &mut rng,
                );
            }
        }

        Ok(FittedLatentDirichletAllocation {
            components_: lambda,
            alpha_: alpha,
            beta_: beta,
            n_iter_: self.max_iter,
            max_doc_update_iter_: self.max_doc_update_iter,
        })
    }
}

impl LatentDirichletAllocation {
    /// Batch variational EM.
    #[allow(clippy::too_many_arguments)]
    fn fit_batch(
        &self,
        x: &Array2<f64>,
        lambda: &mut Array2<f64>,
        alpha: f64,
        beta: f64,
        n_docs: usize,
        n_words: usize,
        n_topics: usize,
    ) {
        for _outer in 0..self.max_iter {
            // Compute E[log beta] from current lambda.
            let e_log_beta = compute_e_log_beta(lambda, n_topics, n_words);

            // Accumulate sufficient statistics.
            let mut ss = Array2::<f64>::zeros((n_topics, n_words));

            for d in 0..n_docs {
                let doc: Vec<f64> = (0..n_words).map(|w| x[[d, w]]).collect();
                let gamma = e_step_doc(&doc, &e_log_beta, alpha, self.max_doc_update_iter);

                // Compute phi for this document and accumulate.
                let e_log_theta: Vec<f64> = gamma.iter().map(|&g| digamma(g)).collect();
                let gamma_sum_dig = digamma(gamma.iter().sum::<f64>());

                for w in 0..n_words {
                    if doc[w] < 1e-16 {
                        continue;
                    }
                    let mut log_phi = Vec::with_capacity(n_topics);
                    let mut max_log = f64::NEG_INFINITY;
                    for k in 0..n_topics {
                        let v = e_log_theta[k] - gamma_sum_dig + e_log_beta[[k, w]];
                        log_phi.push(v);
                        if v > max_log {
                            max_log = v;
                        }
                    }
                    let mut phi = Vec::with_capacity(n_topics);
                    let mut sum_phi = 0.0;
                    for lp in &log_phi {
                        let p = (lp - max_log).exp();
                        phi.push(p);
                        sum_phi += p;
                    }
                    if sum_phi < 1e-16 {
                        sum_phi = 1e-16;
                    }
                    for k in 0..n_topics {
                        ss[[k, w]] += doc[w] * phi[k] / sum_phi;
                    }
                }
            }

            // M-step: update lambda.
            for k in 0..n_topics {
                for w in 0..n_words {
                    lambda[[k, w]] = beta + ss[[k, w]];
                }
            }
        }
    }

    /// Online variational Bayes (Hoffman et al. 2010).
    #[allow(clippy::too_many_arguments)]
    fn fit_online(
        &self,
        x: &Array2<f64>,
        lambda: &mut Array2<f64>,
        alpha: f64,
        beta: f64,
        n_docs: usize,
        n_words: usize,
        n_topics: usize,
        _rng: &mut Xoshiro256PlusPlus,
    ) {
        let mut update_count = 0u64;

        for _outer in 0..self.max_iter {
            // Process each document as a mini-batch of size 1.
            for d in 0..n_docs {
                let doc: Vec<f64> = (0..n_words).map(|w| x[[d, w]]).collect();

                let e_log_beta = compute_e_log_beta(lambda, n_topics, n_words);
                let gamma = e_step_doc(&doc, &e_log_beta, alpha, self.max_doc_update_iter);

                // Compute sufficient statistics for this document.
                let e_log_theta: Vec<f64> = gamma.iter().map(|&g| digamma(g)).collect();
                let gamma_sum_dig = digamma(gamma.iter().sum::<f64>());

                let mut ss = Array2::<f64>::zeros((n_topics, n_words));
                for w in 0..n_words {
                    if doc[w] < 1e-16 {
                        continue;
                    }
                    let mut log_phi = Vec::with_capacity(n_topics);
                    let mut max_log = f64::NEG_INFINITY;
                    for k in 0..n_topics {
                        let v = e_log_theta[k] - gamma_sum_dig + e_log_beta[[k, w]];
                        log_phi.push(v);
                        if v > max_log {
                            max_log = v;
                        }
                    }
                    let mut phi = Vec::with_capacity(n_topics);
                    let mut sum_phi = 0.0;
                    for lp in &log_phi {
                        let p = (lp - max_log).exp();
                        phi.push(p);
                        sum_phi += p;
                    }
                    if sum_phi < 1e-16 {
                        sum_phi = 1e-16;
                    }
                    for k in 0..n_topics {
                        ss[[k, w]] += doc[w] * phi[k] / sum_phi;
                    }
                }

                // Online update with decaying step size.
                update_count += 1;
                let rho = (self.learning_offset + update_count as f64).powf(-self.learning_decay);

                // lambda_new = (1-rho)*lambda + rho*(beta + n_docs * ss)
                let n_docs_f = n_docs as f64;
                for k in 0..n_topics {
                    for w in 0..n_words {
                        let target = beta + n_docs_f * ss[[k, w]];
                        lambda[[k, w]] = (1.0 - rho) * lambda[[k, w]] + rho * target;
                    }
                }
            }
        }
    }
}

/// Compute E[log beta] from lambda (the variational parameters for topic-word).
fn compute_e_log_beta(lambda: &Array2<f64>, n_topics: usize, n_words: usize) -> Array2<f64> {
    let mut e_log_beta = Array2::<f64>::zeros((n_topics, n_words));
    for k in 0..n_topics {
        let row_sum: f64 = (0..n_words).map(|w| lambda[[k, w]]).sum();
        let dig_sum = digamma(row_sum);
        for w in 0..n_words {
            e_log_beta[[k, w]] = digamma(lambda[[k, w]]) - dig_sum;
        }
    }
    e_log_beta
}

impl Transform<Array2<f64>> for FittedLatentDirichletAllocation {
    type Output = Array2<f64>;
    type Error = FerroError;

    /// Compute the document-topic distribution for new documents.
    ///
    /// Returns an array of shape `(n_docs, n_topics)` where each row sums
    /// approximately to 1.
    ///
    /// # Errors
    ///
    /// - [`FerroError::ShapeMismatch`] if the number of words does not match
    ///   the vocabulary size from fitting.
    /// - [`FerroError::InvalidParameter`] if any entry is negative.
    fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>, FerroError> {
        let n_words = self.components_.ncols();
        if x.ncols() != n_words {
            return Err(FerroError::ShapeMismatch {
                expected: vec![x.nrows(), n_words],
                actual: vec![x.nrows(), x.ncols()],
                context: "FittedLatentDirichletAllocation::transform".into(),
            });
        }
        // Reject NaN/Inf BEFORE the non-negativity check and the E-step (sklearn
        // re-validates via `_check_non_neg_array` `_validate_data(force_all_finite
        // =True)` `_lda.py:566` before `check_non_negative`, `utils/validation.py:147-154`).
        reject_non_finite(x)?;
        for &val in x {
            if val < 0.0 {
                return Err(FerroError::InvalidParameter {
                    name: "X".into(),
                    reason: "LDA requires non-negative entries".into(),
                });
            }
        }

        let n_docs = x.nrows();
        let n_topics = self.components_.nrows();
        let e_log_beta = compute_e_log_beta(&self.components_, n_topics, n_words);

        let mut result = Array2::<f64>::zeros((n_docs, n_topics));
        for d in 0..n_docs {
            let doc: Vec<f64> = (0..n_words).map(|w| x[[d, w]]).collect();
            let gamma = e_step_doc(&doc, &e_log_beta, self.alpha_, self.max_doc_update_iter_);

            // Normalise gamma to get document-topic proportions.
            let gamma_sum: f64 = gamma.iter().sum();
            if gamma_sum > 1e-16 {
                for k in 0..n_topics {
                    result[[d, k]] = gamma[k] / gamma_sum;
                }
            } else {
                // Uniform fallback.
                let uniform = 1.0 / n_topics as f64;
                for k in 0..n_topics {
                    result[[d, k]] = uniform;
                }
            }
        }

        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    /// Simple two-topic corpus.
    fn two_topic_corpus() -> Array2<f64> {
        array![
            [5.0, 5.0, 5.0, 0.0, 0.0, 0.0],
            [4.0, 6.0, 3.0, 0.0, 0.0, 0.0],
            [5.0, 4.0, 6.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 5.0, 5.0, 5.0],
            [0.0, 0.0, 0.0, 6.0, 4.0, 3.0],
            [0.0, 0.0, 0.0, 4.0, 6.0, 5.0],
        ]
    }

    #[test]
    fn test_lda_basic_shape() {
        let dtm = two_topic_corpus();
        let lda = LatentDirichletAllocation::new(2).with_random_state(42);
        let fitted = lda.fit(&dtm, &()).unwrap();
        assert_eq!(fitted.components().dim(), (2, 6));
    }

    #[test]
    fn test_lda_transform_shape() {
        let dtm = two_topic_corpus();
        let lda = LatentDirichletAllocation::new(2).with_random_state(42);
        let fitted = lda.fit(&dtm, &()).unwrap();
        let topics = fitted.transform(&dtm).unwrap();
        assert_eq!(topics.dim(), (6, 2));
    }

    #[test]
    fn test_lda_topic_proportions_sum_to_one() {
        let dtm = two_topic_corpus();
        let lda = LatentDirichletAllocation::new(2)
            .with_max_iter(20)
            .with_random_state(42);
        let fitted = lda.fit(&dtm, &()).unwrap();
        let topics = fitted.transform(&dtm).unwrap();
        for i in 0..topics.nrows() {
            let sum: f64 = topics.row(i).sum();
            assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-5);
        }
    }

    #[test]
    fn test_lda_topics_distinguish_groups() {
        let dtm = two_topic_corpus();
        let lda = LatentDirichletAllocation::new(2)
            .with_max_iter(30)
            .with_random_state(42);
        let fitted = lda.fit(&dtm, &()).unwrap();
        let topics = fitted.transform(&dtm).unwrap();

        // First 3 docs should cluster on one topic, last 3 on another.
        // Check that the dominant topic differs between the two groups.
        let first_group_topic: Vec<usize> = (0..3)
            .map(|i| {
                if topics[[i, 0]] > topics[[i, 1]] {
                    0
                } else {
                    1
                }
            })
            .collect();
        let second_group_topic: Vec<usize> = (3..6)
            .map(|i| {
                if topics[[i, 0]] > topics[[i, 1]] {
                    0
                } else {
                    1
                }
            })
            .collect();

        // At least 2 out of 3 in each group should agree on the topic.
        let fg_mode = if first_group_topic.iter().filter(|&&t| t == 0).count() >= 2 {
            0
        } else {
            1
        };
        let sg_mode = if second_group_topic.iter().filter(|&&t| t == 0).count() >= 2 {
            0
        } else {
            1
        };

        assert_ne!(
            fg_mode, sg_mode,
            "the two document groups should be assigned to different topics"
        );
    }

    #[test]
    fn test_lda_online_learning() {
        let dtm = two_topic_corpus();
        let lda = LatentDirichletAllocation::new(2)
            .with_learning_method(LdaLearningMethod::Online)
            .with_max_iter(10)
            .with_random_state(42);
        let fitted = lda.fit(&dtm, &()).unwrap();
        assert_eq!(fitted.components().dim(), (2, 6));
        let topics = fitted.transform(&dtm).unwrap();
        // Each row should sum to ~1.
        for i in 0..topics.nrows() {
            let sum: f64 = topics.row(i).sum();
            assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-5);
        }
    }

    #[test]
    fn test_lda_components_non_negative() {
        let dtm = two_topic_corpus();
        let lda = LatentDirichletAllocation::new(2).with_random_state(42);
        let fitted = lda.fit(&dtm, &()).unwrap();
        for &val in fitted.components() {
            assert!(val >= 0.0, "component should be non-negative, got {val}");
        }
    }

    #[test]
    fn test_lda_transform_shape_mismatch() {
        let dtm = two_topic_corpus();
        let lda = LatentDirichletAllocation::new(2).with_random_state(42);
        let fitted = lda.fit(&dtm, &()).unwrap();
        let bad = array![[1.0, 2.0, 3.0]]; // 3 words instead of 6
        assert!(fitted.transform(&bad).is_err());
    }

    #[test]
    fn test_lda_transform_negative_rejected() {
        let dtm = two_topic_corpus();
        let lda = LatentDirichletAllocation::new(2).with_random_state(42);
        let fitted = lda.fit(&dtm, &()).unwrap();
        let bad = array![[1.0, -1.0, 0.0, 0.0, 0.0, 0.0]];
        assert!(fitted.transform(&bad).is_err());
    }

    #[test]
    fn test_lda_invalid_n_components_zero() {
        let dtm = two_topic_corpus();
        let lda = LatentDirichletAllocation::new(0);
        assert!(lda.fit(&dtm, &()).is_err());
    }

    #[test]
    fn test_lda_negative_input_rejected() {
        let dtm = array![[1.0, -1.0], [2.0, 3.0]];
        let lda = LatentDirichletAllocation::new(1);
        assert!(lda.fit(&dtm, &()).is_err());
    }

    #[test]
    fn test_lda_empty_corpus() {
        let dtm = Array2::<f64>::zeros((0, 5));
        let lda = LatentDirichletAllocation::new(2);
        assert!(lda.fit(&dtm, &()).is_err());
    }

    #[test]
    fn test_lda_zero_words() {
        let dtm = Array2::<f64>::zeros((5, 0));
        let lda = LatentDirichletAllocation::new(2);
        assert!(lda.fit(&dtm, &()).is_err());
    }

    #[test]
    fn test_lda_getters() {
        let lda = LatentDirichletAllocation::new(5)
            .with_max_iter(20)
            .with_learning_method(LdaLearningMethod::Online)
            .with_learning_offset(15.0)
            .with_learning_decay(0.5)
            .with_doc_topic_prior(0.1)
            .with_topic_word_prior(0.01)
            .with_random_state(99);
        assert_eq!(lda.n_components(), 5);
        assert_eq!(lda.max_iter(), 20);
        assert_eq!(lda.learning_method(), LdaLearningMethod::Online);
        assert!((lda.learning_offset() - 15.0).abs() < 1e-10);
        assert!((lda.learning_decay() - 0.5).abs() < 1e-10);
        assert_eq!(lda.doc_topic_prior(), Some(0.1));
        assert_eq!(lda.topic_word_prior(), Some(0.01));
        assert_eq!(lda.random_state(), Some(99));
    }

    #[test]
    fn test_lda_fitted_accessors() {
        let dtm = two_topic_corpus();
        let lda = LatentDirichletAllocation::new(2)
            .with_doc_topic_prior(0.5)
            .with_topic_word_prior(0.1)
            .with_random_state(42);
        let fitted = lda.fit(&dtm, &()).unwrap();
        assert!((fitted.alpha() - 0.5).abs() < 1e-10);
        assert!((fitted.beta() - 0.1).abs() < 1e-10);
        assert!(fitted.n_iter() > 0);
    }

    #[test]
    fn test_lda_single_topic() {
        let dtm = two_topic_corpus();
        let lda = LatentDirichletAllocation::new(1).with_random_state(42);
        let fitted = lda.fit(&dtm, &()).unwrap();
        let topics = fitted.transform(&dtm).unwrap();
        assert_eq!(topics.ncols(), 1);
        // With 1 topic, all documents should have proportion ~1.
        for i in 0..topics.nrows() {
            assert_abs_diff_eq!(topics[[i, 0]], 1.0, epsilon = 1e-3);
        }
    }

    #[test]
    fn test_digamma_basic() {
        // digamma(1) = -gamma (Euler-Mascheroni constant) ~ -0.5772
        let val = digamma(1.0);
        assert!((val - (-0.5772156649)).abs() < 1e-4, "digamma(1) = {val}");
    }

    #[test]
    fn test_digamma_large() {
        // digamma(10) ~ 2.2517525890
        let val = digamma(10.0);
        assert!((val - 2.2517525890).abs() < 1e-4, "digamma(10) = {val}");
    }
}
