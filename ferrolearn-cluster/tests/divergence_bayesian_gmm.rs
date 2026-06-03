//! Green-guard tests for `ferrolearn-cluster` `BayesianGaussianMixture` against
//! the live scikit-learn 1.5.2 oracle (`sklearn/mixture/_bayesian_mixture.py`,
//! `class BayesianGaussianMixture(BaseMixture)` :75).
//!
//! `bayesian_gmm.rs` is a **heuristic plain-EM approximation, NOT scikit-learn's
//! variational Bayes** (see `.design/cluster/bayesian_gmm.md`). The entire VB
//! algorithm, every Bayesian fitted-attribute value, the true ELBO, the
//! full-covariance score, and the PyO3 binding DIVERGE and are documented as
//! NOT-STARTED blockers #1057-#1066 — those are builder-scale, NOT minimal pins,
//! so this file pins NO RED test for them (a RED test would red the gauntlet,
//! R-DEFER-6).
//!
//! What this file green-guards (PASS today), all expected values from the live
//! sklearn 1.5.2 oracle (R-CHAR-3), never literal-copied from ferrolearn:
//!
//! 1. **REQ-1 is NOT-STARTED (#1067).** ferrolearn does NOT replicate sklearn's
//!    automatic DP component pruning, so the partition diverges whenever sklearn
//!    prunes: on two blobs at (0,0)/(20,20) with `n_components=5`, live sklearn
//!    prunes to a SINGLE dominant component (`fit_predict -> [1;8]`,
//!    `weights_ ~ [0.12, 0.86, 0.016, ...]`) while ferrolearn keeps two. The
//!    `char_*` test below is a regime CHARACTERIZATION (not a SHIPPED contract): on
//!    a 3-blob fixture where sklearn happens to retain all 3 components, the
//!    partitions coincide. The general partition-match is gated on the VB/DP
//!    algorithm (#1057) and is NOT pinned RED here (R-DEFER-6).
//! 2. **REQ-2** — API / output-shape + row-stochastic contracts:
//!    `predict_proba(X)` is `(n, k)` with each row summing to 1 (mirroring
//!    `BaseMixture.predict_proba` `_base.py:387-404` + `_estimate_log_prob_resp`
//!    logsumexp normalization `:507-531`); `weights()` sums to 1; `means()` is
//!    `(k, d)`; `score_samples(X)` is `(n,)`. These are ferrolearn-internal
//!    contract guards (shapes + stochasticity), NOT exact value parity vs
//!    sklearn (the algorithm is heuristic).
//! 3. **REQ-9** — matching defaults. Observable: `new(k)` defaults to `Full`
//!    covariance + `DirichletProcess` prior, matching sklearn `__init__`
//!    (`_bayesian_mixture.py:373,:379`). The config fields are PRIVATE; the
//!    constants below are anchored to the sklearn `file:line` defaults (R-CHAR-3),
//!    asserted via observable fit behavior plus the fitted `weight_prior_type()`.
//!
//! **REQ-1 honesty (R-HONEST-3/4).** A prior draft of this file green-guarded a
//! 2-blob "partition matches sklearn" claim with a FABRICATED expected value
//! (ferrolearn's own `[[0,1,2,3],[4,5,6,7]]`); the live oracle actually prunes that
//! fixture to one cluster. That guard was removed and REQ-1 reclassified NOT-STARTED
//! (#1067). The remaining `char_*` 3-blob test is honest (live-oracle-verified) but
//! only documents the no-prune regime.

use ferrolearn_cluster::{BayesianCovType, BayesianGaussianMixture, WeightPriorType};
use ferrolearn_core::traits::{Fit, Predict};
use ndarray::Array2;

/// Canonicalize a label vector to a set of sorted index-groups, ignoring the
/// integer label values (and thus any permutation / pruning-driven relabel).
fn canonical_partition(labels: &[usize]) -> Vec<Vec<usize>> {
    use std::collections::BTreeMap;
    let mut groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (i, &l) in labels.iter().enumerate() {
        groups.entry(l).or_default().push(i);
    }
    let mut out: Vec<Vec<usize>> = groups.into_values().collect();
    for g in &mut out {
        g.sort_unstable();
    }
    out.sort();
    out
}

/// Fresh 3-blob fixture (not in `bayesian_gmm.rs` tests): a symmetric triangle of
/// three 4-point blobs at (0,0), (30,0), (15,26) — mutually ~30 apart, so the
/// diagonal-only heuristic score still separates them (cf. the module-doc note).
fn three_blobs() -> Array2<f64> {
    Array2::from_shape_vec(
        (12, 2),
        vec![
            0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 0.05, 0.05, // blob A @ (0,0)
            30.0, 0.0, 30.1, 0.0, 30.0, 0.1, 30.05, 0.05, // blob B @ (30,0)
            15.0, 26.0, 15.1, 26.0, 15.0, 26.1, 15.05, 26.05, // blob C @ (15,26)
        ],
    )
    .unwrap()
}

// ─────────────────────────────────────────────────────────────────────────────
// REQ-1 (NOT-STARTED #1067) — regime CHARACTERIZATION, not a SHIPPED contract.
// On a no-prune isotropic 3-blob fixture ferrolearn's partition coincides with
// sklearn's. The general partition-match diverges because ferrolearn lacks
// sklearn's DP component pruning (see the module doc + #1057/#1067).
// ─────────────────────────────────────────────────────────────────────────────

/// Characterization (REQ-1 regime boundary): on the no-prune 3-blob fixture
/// ferrolearn's `fit_predict` recovers the SAME co-membership partition (up to a
/// permutation) as sklearn's
/// `BayesianGaussianMixture(n_components=5, random_state=0).fit_predict(X)`.
///
/// Live sklearn 1.5.2 oracle (run from /tmp):
/// ```text
/// b1=[[0,0],[.1,0],[0,.1],[.05,.05]]; b2=b1+[30,0]; b3=[[15,26],...];
/// X=vstack([b1,b2,b3]);
/// BayesianGaussianMixture(n_components=5,random_state=0).fit_predict(X)
///   -> [2,2,2,2, 3,3,3,3, 1,1,1,1]
/// canonical_partition -> [[0,1,2,3],[4,5,6,7],[8,9,10,11]]
/// ```
/// NOTE: this is NOT a general partition contract — on a 2-blob fixture sklearn
/// prunes to one component while ferrolearn keeps two (#1067).
#[test]
fn char_bgm_three_blob_no_prune_partition_matches_sklearn() {
    // From the live sklearn oracle (canonicalized, permutation-invariant).
    let sklearn_partition: Vec<Vec<usize>> =
        vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7], vec![8, 9, 10, 11]];

    let x = three_blobs();
    let model = BayesianGaussianMixture::<f64>::new(5).with_random_state(0);
    let labels = model.fit_predict(&x).unwrap();
    let ferro_partition = canonical_partition(labels.as_slice().unwrap());

    assert_eq!(
        ferro_partition, sklearn_partition,
        "ferrolearn 3-blob partition {ferro_partition:?} must equal sklearn's \
         {sklearn_partition:?} up to a label permutation"
    );
}

// (Removed: a 2-blob "partition matches sklearn" guard that fabricated its
// expected value to match ferrolearn — live sklearn prunes that fixture to one
// component. REQ-1 is NOT-STARTED #1067; see the module doc.)

// ─────────────────────────────────────────────────────────────────────────────
// REQ-2 — API / output-shape + row-stochastic contracts (GREEN-GUARD)
// ─────────────────────────────────────────────────────────────────────────────

/// Green-guard REQ-2: `predict_proba(X)` is `(n_samples, n_components)` and each
/// row sums to 1 — the row-stochastic contract of `BaseMixture.predict_proba`
/// (`_base.py:387-404`) + `_estimate_log_prob_resp` logsumexp normalization
/// (`_base.py:507-531`). Row-sum==1 is a sklearn structural invariant, not a
/// ferrolearn literal.
#[test]
fn green_req2_predict_proba_shape_and_row_stochastic() {
    let x = three_blobs();
    let k = 5;
    let fitted = BayesianGaussianMixture::<f64>::new(k)
        .with_random_state(0)
        .fit(&x, &())
        .unwrap();

    let proba = fitted.predict_proba(&x).unwrap();
    assert_eq!(
        proba.dim(),
        (x.nrows(), k),
        "predict_proba must be (n_samples, n_components)"
    );
    for (i, row) in proba.rows().into_iter().enumerate() {
        let s: f64 = row.iter().sum();
        assert!(
            (s - 1.0).abs() < 1e-9,
            "predict_proba row {i} must sum to 1 (got {s})"
        );
    }
}

/// Green-guard REQ-2: `weights()` sums to 1 (sklearn `weights_` is a probability
/// vector; `_set_parameters` derives `weights_ = weight_concentration_ / sum`,
/// `_bayesian_mixture.py:870-873`).
#[test]
fn green_req2_weights_sum_to_one() {
    let x = three_blobs();
    let fitted = BayesianGaussianMixture::<f64>::new(5)
        .with_random_state(0)
        .fit(&x, &())
        .unwrap();
    let s: f64 = fitted.weights().iter().sum();
    assert!((s - 1.0).abs() < 1e-9, "weights must sum to 1 (got {s})");
}

/// Green-guard REQ-2: `means()` is `(n_components, n_features)` and
/// `score_samples(X)` is `(n_samples,)` and finite. Shapes mirror sklearn's
/// `means_` (`(k, d)`) and `score_samples` (`(n,)`, `_base.py:331-347`).
#[test]
fn green_req2_means_and_score_samples_shapes() {
    let x = three_blobs();
    let k = 5;
    let fitted = BayesianGaussianMixture::<f64>::new(k)
        .with_random_state(0)
        .fit(&x, &())
        .unwrap();

    assert_eq!(
        fitted.means().dim(),
        (k, x.ncols()),
        "means must be (n_components, n_features)"
    );

    let scores = fitted.score_samples(&x).unwrap();
    assert_eq!(
        scores.len(),
        x.nrows(),
        "score_samples must be (n_samples,)"
    );
    assert!(
        scores.iter().all(|v| v.is_finite()),
        "score_samples must be finite"
    );

    let labels = fitted.predict(&x).unwrap();
    assert_eq!(labels.len(), x.nrows(), "predict must be (n_samples,)");
}

// ─────────────────────────────────────────────────────────────────────────────
// REQ-9 — matching defaults (GREEN-GUARD; sklearn file:line constants)
// ─────────────────────────────────────────────────────────────────────────────

// sklearn `BayesianGaussianMixture.__init__` defaults
// (`sklearn/mixture/_bayesian_mixture.py`):
//   covariance_type="full"                               (:373)
//   tol=1e-3                                              (:374)
//   max_iter=100                                          (:376)
//   weight_concentration_prior_type="dirichlet_process"  (:379)
// Anchored as named constants traceable to the sklearn source (R-CHAR-3), NOT
// copied from ferrolearn.
const SK_DEFAULT_COVARIANCE_TYPE: BayesianCovType = BayesianCovType::Full; // :373
const SK_DEFAULT_WEIGHT_PRIOR_TYPE: WeightPriorType = WeightPriorType::DirichletProcess; // :379

/// Green-guard REQ-9: the ferrolearn default `BayesianGaussianMixture::new(k)`
/// behaves identically to explicitly setting the sklearn-default
/// `covariance_type="full"` and `weight_concentration_prior_type="dirichlet_process"`
/// (the config fields are private; this asserts the *observable* default matches
/// the sklearn `file:line` defaults). If `new` defaulted to a different
/// covariance type or prior, the two fitted partitions would diverge.
#[test]
fn green_req9_defaults_match_sklearn_observably() {
    let x = three_blobs();

    // The default-constructed model.
    let default_labels = BayesianGaussianMixture::<f64>::new(5)
        .with_random_state(0)
        .fit_predict(&x)
        .unwrap();

    // Explicitly re-set the sklearn defaults (from the source line constants).
    let explicit_labels = BayesianGaussianMixture::<f64>::new(5)
        .with_random_state(0)
        .with_covariance_type(SK_DEFAULT_COVARIANCE_TYPE)
        .with_weight_prior_type(SK_DEFAULT_WEIGHT_PRIOR_TYPE)
        .fit_predict(&x)
        .unwrap();

    assert_eq!(
        canonical_partition(default_labels.as_slice().unwrap()),
        canonical_partition(explicit_labels.as_slice().unwrap()),
        "new(k) default must behave as covariance_type=Full + \
         weight_prior_type=DirichletProcess (sklearn defaults :373,:379)"
    );

    // weight_prior_type of the default fit is observable on the fitted model and
    // must equal the sklearn default (dirichlet_process, :379).
    let fitted = BayesianGaussianMixture::<f64>::new(5)
        .with_random_state(0)
        .fit(&x, &())
        .unwrap();
    assert_eq!(
        fitted.weight_prior_type(),
        SK_DEFAULT_WEIGHT_PRIOR_TYPE,
        "default weight_prior_type must be DirichletProcess (sklearn :379)"
    );
}
