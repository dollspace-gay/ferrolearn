//! Divergence audit for `LatentDirichletAllocation` /
//! `FittedLatentDirichletAllocation` (`ferrolearn-decomp/src/lda_topic.rs`) against
//! scikit-learn 1.5.2 `class LatentDirichletAllocation`
//! (`sklearn/decomposition/_lda.py:163`) — the TOPIC-MODEL LDA (online variational
//! Bayes, Hoffman et al. 2010), NOT Linear Discriminant Analysis.
//!
//! Design doc: `.design/decomp/lda_topic.md` (3 SHIPPED / 12 NOT-STARTED, tracking
//! #1540).
//!
//! # Verdict: NO FIXABLE DIVERGENCE FOUND.
//!
//! Every value-parity candidate is gated on the RNG-coupled `components_`
//! (carve-out, same class as `minibatch_nmf` / `dictionary_learning` / `sparse_pca`):
//! ferrolearn inits `lambda`/`components_` from `Uniform(0.5,1.5)+beta` via Xoshiro256++
//! (`lda_topic.rs:414-419`) and the per-doc gamma uniformly at `alpha + V/K`
//! (`lda_topic.rs:308`), whereas sklearn inits `components_ = random_state.gamma(100,
//! 0.01)` (`_lda.py:419-421`) and the per-doc gamma from `np.ones` (transform path,
//! `random_init=False`, `_lda.py:100-101`) or `gamma(100, 0.01)` (M-step path,
//! `_lda.py:96-99`), with numpy `RandomState`. Different init distribution + RNG ⇒
//! `components_` and `transform` VALUES diverge element-wise. The `FittedLatentDirichlet
//! Allocation` fields are PRIVATE with no injectable-components constructor (only
//! `new` + builders + accessors `components()`/`n_iter()`/`alpha()`/`beta()`), so a
//! `transform`-value pin against sklearn's E-step is unreachable without the carved-out
//! `components_`. No component/topic VALUE pin is asserted (goal.md R-DEFER-3).
//!
//! INVESTIGATE outcomes (all confirmed against the live sklearn 1.5.2 oracle, run from
//! /tmp, R-CHAR-3):
//!   - `n_iter_` semantics: ferrolearn `n_iter_ == max_iter` MATCHES sklearn under the
//!     default `evaluate_every=-1` (no perplexity early-stop fires, `_lda.py:660`/`:695`).
//!     Oracle: `LatentDirichletAllocation(n_components=2, random_state=0).fit(dtm)`
//!     reports `n_iter_=10` (== default `max_iter`); with `max_iter=20`, `n_iter_=20`.
//!     Green-guarded by `green_guard_n_iter_equals_max_iter` below.
//!   - `doc_topic_prior_` default `1/n_components`: MATCHES. Oracle reports
//!     `doc_topic_prior_=0.5` / `topic_word_prior_=0.5` for `n_components=2`
//!     (`_lda.py:406-414`); ferrolearn resolves `alpha = beta = 1.0/n_topics = 0.5`
//!     (`lda_topic.rs:409-410`). Green-guarded by `green_guard_default_priors`.
//!   - digamma: the doc-author verified `digamma` matches `scipy.special.psi` to
//!     ~1.17e-10 over x in [0.1, 100] (NOT a divergence source).
//!   - transform/component VALUES: carve-out-gated on the RNG (see above); no
//!     injectable-components API exists, so no value pin is reachable.
//!
//! All tests below are STRUCTURAL GREEN-GUARDS: they assert sklearn-confirmed structural
//! properties and MUST PASS against the current implementation. They are NOT
//! `#[ignore]`d. There is no fixable divergence to track.

use approx::assert_abs_diff_eq;
use ferrolearn_core::traits::{Fit, Transform};
use ferrolearn_decomp::{LatentDirichletAllocation, LdaLearningMethod};
use ndarray::{Array2, array};

/// Well-separated two-topic corpus (matches the in-module `two_topic_corpus()` fixture
/// and the sklearn oracle probe in `.design/decomp/lda_topic.md`).
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

// ---------------------------------------------------------------------------
// REQ-1/2 — components_ shape, non-negativity (structural).
// ---------------------------------------------------------------------------

/// Green-guard: `components_` has shape `(n_topics, n_words)`.
///
/// sklearn `_init_latent_vars` sets `self.components_ = random_state_.gamma(100., 0.01,
/// (n_components, n_features))` (`_lda.py:419-421`); oracle reports
/// `components_.shape == (2, 6)` for `n_components=2` on the 6-word corpus.
#[test]
fn green_guard_components_shape() {
    let dtm = two_topic_corpus();
    let lda = LatentDirichletAllocation::new(2).with_random_state(0);
    let fitted = lda.fit(&dtm, &()).expect("fit should succeed");
    // sklearn oracle: components_.shape == (2, 6).
    assert_eq!(fitted.components().dim(), (2, 6));
}

/// Green-guard: every `components_` entry is non-negative.
///
/// sklearn M-step is `components_ = topic_word_prior_ + suff_stats` (`_lda.py:528`),
/// a non-negative Gamma init plus non-negative sufficient statistics; oracle reports
/// `(m.components_ >= 0).all() == True`.
#[test]
fn green_guard_components_non_negative() {
    let dtm = two_topic_corpus();
    let lda = LatentDirichletAllocation::new(2).with_random_state(0);
    let fitted = lda.fit(&dtm, &()).expect("fit should succeed");
    for &val in fitted.components() {
        assert!(val >= 0.0, "component must be non-negative, got {val}");
    }
}

// ---------------------------------------------------------------------------
// REQ-3 — transform doc-topic shape + each row sums to 1 (structural).
// ---------------------------------------------------------------------------

/// Green-guard: `transform` returns shape `(n_docs, n_topics)` and EACH ROW sums to 1.
///
/// sklearn `transform` normalises `doc_topic_distr /= doc_topic_distr.sum(axis=1)[:,
/// np.newaxis]` (`_lda.py:745`); oracle reports `transform(dtm).sum(axis=1) == [1.0]*6`.
#[test]
fn green_guard_transform_shape_and_row_sums() {
    let dtm = two_topic_corpus();
    let lda = LatentDirichletAllocation::new(2)
        .with_max_iter(20)
        .with_random_state(0);
    let fitted = lda.fit(&dtm, &()).expect("fit should succeed");
    let topics = fitted.transform(&dtm).expect("transform should succeed");
    // sklearn oracle: transform shape (6, 2).
    assert_eq!(topics.dim(), (6, 2));
    for i in 0..topics.nrows() {
        let sum: f64 = topics.row(i).sum();
        // sklearn oracle: each row sum == 1.0 (`_lda.py:745`).
        assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-5);
    }
}

/// Green-guard: on a well-separated two-topic corpus the two document groups load on
/// DIFFERENT dominant topics (structural topic separation, NOT value parity).
///
/// sklearn oracle `transform` row0 `[0.968663, 0.031337]` (dominant topic 0) vs the
/// last-group rows dominate the other topic; the structural property is that the two
/// groups separate, which is RNG-/label-permutation invariant.
#[test]
fn green_guard_topic_separation() {
    let dtm = two_topic_corpus();
    let lda = LatentDirichletAllocation::new(2)
        .with_max_iter(30)
        .with_random_state(0);
    let fitted = lda.fit(&dtm, &()).expect("fit should succeed");
    let topics = fitted.transform(&dtm).expect("transform should succeed");

    let dominant = |i: usize| -> usize {
        if topics[[i, 0]] >= topics[[i, 1]] {
            0
        } else {
            1
        }
    };
    let fg_mode = if (0..3).filter(|&i| dominant(i) == 0).count() >= 2 {
        0
    } else {
        1
    };
    let sg_mode = if (3..6).filter(|&i| dominant(i) == 0).count() >= 2 {
        0
    } else {
        1
    };
    assert_ne!(
        fg_mode, sg_mode,
        "the two well-separated document groups must load on different dominant topics"
    );
}

/// Green-guard: with `n_components=1` every document proportion is ~1.
///
/// A single-topic model degenerately assigns all mass to the one topic; sklearn's
/// row-normalisation (`_lda.py:745`) of a length-1 gamma is exactly 1.0.
#[test]
fn green_guard_single_topic_all_one() {
    let dtm = two_topic_corpus();
    let lda = LatentDirichletAllocation::new(1).with_random_state(0);
    let fitted = lda.fit(&dtm, &()).expect("fit should succeed");
    let topics = fitted.transform(&dtm).expect("transform should succeed");
    assert_eq!(topics.ncols(), 1);
    for i in 0..topics.nrows() {
        assert_abs_diff_eq!(topics[[i, 0]], 1.0, epsilon = 1e-3);
    }
}

// ---------------------------------------------------------------------------
// REQ-1 — n_iter_ == max_iter (matches sklearn under default evaluate_every=-1).
// ---------------------------------------------------------------------------

/// Green-guard: `n_iter()` equals the configured `max_iter`.
///
/// sklearn's `fit` loop runs `for i in range(max_iter)` and only breaks early when
/// `evaluate_every > 0` (`_lda.py:660`/`:676-691`); under the DEFAULT
/// `evaluate_every=-1` no perplexity break fires, so `self.n_iter_ += 1`
/// (`_lda.py:695`) runs exactly `max_iter` times. Oracle: `n_iter_=10` at default
/// `max_iter`, `n_iter_=20` at `max_iter=20`. ferrolearn sets `n_iter_ = self.max_iter`
/// (`lda_topic.rs:443`).
#[test]
fn green_guard_n_iter_equals_max_iter() {
    let dtm = two_topic_corpus();
    // Default max_iter == 10; sklearn oracle n_iter_ == 10.
    let fitted_default = LatentDirichletAllocation::new(2)
        .with_random_state(0)
        .fit(&dtm, &())
        .expect("fit should succeed");
    assert_eq!(fitted_default.n_iter(), 10);

    // max_iter == 20; sklearn oracle n_iter_ == 20.
    let fitted20 = LatentDirichletAllocation::new(2)
        .with_max_iter(20)
        .with_random_state(0)
        .fit(&dtm, &())
        .expect("fit should succeed");
    assert_eq!(fitted20.n_iter(), 20);
}

/// Green-guard: the resolved priors default to `1/n_components`.
///
/// sklearn: `doc_topic_prior_ = doc_topic_prior or 1/n_components`,
/// `topic_word_prior_ = topic_word_prior or 1/n_components` (`_lda.py:406-414`); oracle
/// reports `doc_topic_prior_ == 0.5` and `topic_word_prior_ == 0.5` for `n_components=2`.
/// ferrolearn resolves `alpha = beta = 1.0/n_topics` (`lda_topic.rs:409-410`), exposed
/// via `alpha()` / `beta()`.
#[test]
fn green_guard_default_priors() {
    let dtm = two_topic_corpus();
    let fitted = LatentDirichletAllocation::new(2)
        .with_random_state(0)
        .fit(&dtm, &())
        .expect("fit should succeed");
    // sklearn oracle: doc_topic_prior_ == 0.5, topic_word_prior_ == 0.5 (n_components=2).
    assert_abs_diff_eq!(fitted.alpha(), 0.5, epsilon = 1e-12);
    assert_abs_diff_eq!(fitted.beta(), 0.5, epsilon = 1e-12);
}

// ---------------------------------------------------------------------------
// REQ-1 — determinism given a fixed random_state.
// ---------------------------------------------------------------------------

/// Green-guard: two fits with the same `random_state` produce IDENTICAL components and
/// transforms.
///
/// sklearn is deterministic given a fixed integer `random_state` (numpy `RandomState`);
/// ferrolearn is deterministic given a fixed seed (Xoshiro256++). This is a structural
/// reproducibility property, not value parity with sklearn.
#[test]
fn green_guard_determinism_same_seed() {
    let dtm = two_topic_corpus();
    let a = LatentDirichletAllocation::new(2)
        .with_random_state(7)
        .fit(&dtm, &())
        .expect("fit should succeed");
    let b = LatentDirichletAllocation::new(2)
        .with_random_state(7)
        .fit(&dtm, &())
        .expect("fit should succeed");
    assert_eq!(a.components().dim(), b.components().dim());
    for (x, y) in a.components().iter().zip(b.components().iter()) {
        assert_eq!(x.to_bits(), y.to_bits(), "components must be bit-identical");
    }
    let ta = a.transform(&dtm).expect("transform should succeed");
    let tb = b.transform(&dtm).expect("transform should succeed");
    for (x, y) in ta.iter().zip(tb.iter()) {
        assert_eq!(x.to_bits(), y.to_bits(), "transform must be bit-identical");
    }
}

// ---------------------------------------------------------------------------
// REQ-3 — error / parameter contracts.
// ---------------------------------------------------------------------------

/// Green-guard: `fit` rejects `n_components == 0`, negative input, 0 docs, 0 words; and
/// `transform` rejects a column-count mismatch and negative input.
///
/// sklearn `_check_non_neg_array` (`_lda.py:554`/`:644`) raises `ValueError` on negative
/// input; ferrolearn raises `FerroError` (the divergence is the error TYPE only, an
/// idiomatic-Rust mapping, not a behavior divergence — both reject the same inputs).
#[test]
fn green_guard_error_contracts() {
    let dtm = two_topic_corpus();

    // n_components == 0 -> Err.
    assert!(LatentDirichletAllocation::new(0).fit(&dtm, &()).is_err());

    // negative input -> Err.
    let neg = array![[1.0, -1.0], [2.0, 3.0]];
    assert!(LatentDirichletAllocation::new(1).fit(&neg, &()).is_err());

    // 0 docs -> Err.
    let empty = Array2::<f64>::zeros((0, 5));
    assert!(LatentDirichletAllocation::new(2).fit(&empty, &()).is_err());

    // 0 words -> Err.
    let zero_words = Array2::<f64>::zeros((5, 0));
    assert!(
        LatentDirichletAllocation::new(2)
            .fit(&zero_words, &())
            .is_err()
    );

    // transform column mismatch -> Err.
    let fitted = LatentDirichletAllocation::new(2)
        .with_random_state(0)
        .fit(&dtm, &())
        .expect("fit should succeed");
    let bad_cols = array![[1.0, 2.0, 3.0]]; // 3 words vs fitted 6
    assert!(fitted.transform(&bad_cols).is_err());

    // transform negative -> Err.
    let bad_neg = array![[1.0, -1.0, 0.0, 0.0, 0.0, 0.0]];
    assert!(fitted.transform(&bad_neg).is_err());
}

// ---------------------------------------------------------------------------
// Online learning method — structural row-sum-to-1.
// ---------------------------------------------------------------------------

/// Green-guard: the Online learning method also yields `(n_topics, n_words)` components
/// and a transform whose rows sum to 1.
#[test]
fn green_guard_online_method_structural() {
    let dtm = two_topic_corpus();
    let fitted = LatentDirichletAllocation::new(2)
        .with_learning_method(LdaLearningMethod::Online)
        .with_max_iter(10)
        .with_random_state(0)
        .fit(&dtm, &())
        .expect("fit should succeed");
    assert_eq!(fitted.components().dim(), (2, 6));
    let topics = fitted.transform(&dtm).expect("transform should succeed");
    for i in 0..topics.nrows() {
        let sum: f64 = topics.row(i).sum();
        assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-5);
    }
}
