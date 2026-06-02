//! Divergence pins for `ferrolearn-tree/src/random_trees_embedding.rs`
//! (`RandomTreesEmbedding`) against the live scikit-learn 1.5.2 oracle.
//!
//! `RandomTreesEmbedding` is an inherently RNG-driven embedding: the per-tree
//! feature pick (`next_u64() % n_features`) and uniform threshold draw are seeded
//! through Rust's `StdRng`, while sklearn draws the fabricated random target, the
//! split feature, and the threshold from numpy's MT19937. The two streams cannot
//! bit-match, so EXACT embedding-for-embedding parity at a given `random_state`
//! is a documented RNG boundary (goal.md RNG-boundary precedent, shared with
//! `extra_tree`/`random_forest`/SGD; `.design/tree/random_trees_embedding.md`
//! REQ-6, blocker #1844). These tests therefore pin only the DETERMINISTIC
//! contract: the param/default ABI surface and the structural one-hot invariants
//! (every entry in {0,1}, each row sums to exactly `n_estimators`, column count
//! equals `n_output_features`), which hold for ANY seed/input.
//!
//! Reference: scikit-learn 1.5.2 (commit 156ef14),
//! `sklearn/ensemble/_forest.py` (`RandomTreesEmbedding`, class at :2623).
//!
//! `tests/*.rs` is anti-pattern-gate-exempt: `.unwrap()`/`assert!` are used
//! deliberately (no `panic!`/`unreachable!`).

use ferrolearn_core::traits::{Fit, Transform};
use ferrolearn_tree::RandomTreesEmbedding;
use ndarray::Array2;

/// Deterministic 30x4 dataset with spread-out feature values so totally-random
/// splits can find splittable features. The structural one-hot invariants pinned
/// below are RNG-robust, so the exact values only need to be non-degenerate.
fn make_data() -> Array2<f64> {
    let mut data = Vec::with_capacity(30 * 4);
    // Simple deterministic generator (portable, no RNG-stream dependency): the
    // pin asserts structure, not a numpy-matched embedding.
    let mut state: u64 = 0x243F_6A88_85A3_08D3;
    for _ in 0..(30 * 4) {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        let z = state.wrapping_mul(0x2545_F491_4F6C_DD1D);
        data.push((z >> 11) as f64 / (1u64 << 53) as f64 * 10.0);
    }
    Array2::from_shape_vec((30, 4), data).unwrap()
}

/// PIN 1 (RED — the headline, R-DEV-2 ABI divergence, #687).
///
/// Divergence: `RandomTreesEmbedding::<F>::new` (`fn new` in
/// `random_trees_embedding.rs`) sets `n_estimators = 10`. sklearn's
/// `RandomTreesEmbedding.__init__` defaults `n_estimators=100`
/// (`sklearn/ensemble/_forest.py:2820`):
///   `n_estimators=100,`
/// Verified live: `RandomTreesEmbedding().get_params()['n_estimators'] == 100`.
///
/// Expected value is sklearn's documented default (NOT copied from ferrolearn,
/// which is 10). This MUST currently FAIL (ferrolearn returns 10).
///
/// Tracking: #687
#[test]
fn divergence_n_estimators_default_is_100() {
    // sklearn/ensemble/_forest.py:2820 — `n_estimators=100`; live get_params == 100.
    const SK_N_ESTIMATORS: usize = 100;

    let model = RandomTreesEmbedding::<f64>::new();
    assert_eq!(
        model.n_estimators, SK_N_ESTIMATORS,
        "RandomTreesEmbedding::new() sets n_estimators = {}; sklearn default is \
         100 (_forest.py:2820, live get_params)",
        model.n_estimators
    );
}

/// PIN 2 (GREEN) — the constructor defaults that ARE correct (R-DEV-2 guard).
///
/// sklearn `RandomTreesEmbedding.__init__` (`sklearn/ensemble/_forest.py`):
///   `max_depth=5` (:2822), `min_samples_split=2` (:2823),
///   `min_samples_leaf=1` (:2824), `random_state=None` (default).
/// Verified live: `get_params()` yields `max_depth=5, min_samples_split=2,
/// min_samples_leaf=1, random_state=None`.
///
/// Expected values are sklearn's documented defaults. Guards the matched surface
/// against regression. (ferrolearn has no `min_samples_leaf` field — REQ-1
/// ABSENT, #1842 — so only the params ferrolearn exposes are pinned here.)
#[test]
fn defaults_match_sklearn() {
    const SK_MAX_DEPTH: usize = 5; // _forest.py:2822 max_depth=5
    const SK_MIN_SAMPLES_SPLIT: usize = 2; // _forest.py:2823 min_samples_split=2

    let model = RandomTreesEmbedding::<f64>::new();
    assert_eq!(model.max_depth, Some(SK_MAX_DEPTH));
    assert_eq!(model.min_samples_split, SK_MIN_SAMPLES_SPLIT);
    assert!(model.random_state.is_none()); // random_state=None
}

/// PIN 3 (GREEN) — structural one-hot contract: every entry in {0,1} and each
/// row sums to EXACTLY K (one active leaf per tree).
///
/// sklearn `fit_transform` one-hot-encodes `apply(X)` — one categorical column
/// per tree (`sklearn/ensemble/_forest.py:2982`,
/// `one_hot_encoder_.transform(self.apply(X))`), so each row has exactly one `1`
/// per tree => row-sum == n_estimators, all entries in {0,1}. Verified live:
/// `E(n_estimators=5,max_depth=3,random_state=42).fit_transform(X)` rows all sum
/// to 5.0. RNG-robust: holds for any seed/input. (Expected value `K` is sklearn's
/// per-tree one-hot semantics, NOT copied from ferrolearn.)
#[test]
fn one_hot_row_sum_equals_n_estimators_and_binary() {
    const K: usize = 7;
    let x = make_data();
    let model = RandomTreesEmbedding::<f64>::new()
        .with_n_estimators(K)
        .with_max_depth(Some(3))
        .with_random_state(42);
    let fitted = model.fit(&x, &()).unwrap();
    let embedded = fitted.transform(&x).unwrap();

    assert_eq!(embedded.nrows(), x.nrows());

    for i in 0..embedded.nrows() {
        let mut row_sum = 0.0;
        for &v in embedded.row(i).iter() {
            assert!(
                (v - 0.0).abs() < 1e-12 || (v - 1.0).abs() < 1e-12,
                "entry [{i}] = {v} not in {{0,1}} (sklearn one-hot of apply is binary)"
            );
            row_sum += v;
        }
        assert!(
            (row_sum - K as f64).abs() < 1e-12,
            "row {i} sums to {row_sum}, expected exactly {K} (one active leaf per \
             tree, sklearn _forest.py:2982)"
        );
    }
}

/// PIN 4 (GREEN) — column count equals `n_output_features` (Sigma per-tree leaf
/// counts).
///
/// sklearn's output column count is the sum of per-tree distinct leaf counts
/// (one-hot of `apply`, `sklearn/ensemble/_forest.py:2982`). ferrolearn exposes
/// `n_output_features()` = total leaves across trees. We pin the API-exposed
/// consistency: `transform(X).ncols() == fitted.n_output_features()`. (The
/// fitted accessor is the only per-tree leaf-count surface the API exposes.)
#[test]
fn ncols_equals_n_output_features() {
    let x = make_data();
    let model = RandomTreesEmbedding::<f64>::new()
        .with_n_estimators(6)
        .with_max_depth(Some(3))
        .with_random_state(7);
    let fitted = model.fit(&x, &()).unwrap();
    let embedded = fitted.transform(&x).unwrap();

    assert_eq!(embedded.ncols(), fitted.n_output_features());
    // n_output_features must be at least one leaf per tree (no empty trees).
    assert!(fitted.n_output_features() >= fitted.n_estimators());
}

/// PIN 5 (GREEN) — `random_state` reproducibility (ferrolearn-internal
/// determinism, NOT numpy parity).
///
/// Two `fit` calls with the same `with_random_state(42)` on the same data must
/// produce identical transform output. The exact embedding at a seed vs numpy-MT
/// is the documented RNG boundary (`.design/tree/random_trees_embedding.md`
/// REQ-6, #1844); this only asserts ferrolearn is deterministic w.r.t. its seed.
#[test]
fn random_state_reproducible() {
    let x = make_data();
    let model = RandomTreesEmbedding::<f64>::new()
        .with_n_estimators(6)
        .with_max_depth(Some(3))
        .with_random_state(42);

    let f1 = model.fit(&x, &()).unwrap();
    let e1 = f1.transform(&x).unwrap();

    let f2 = model.fit(&x, &()).unwrap();
    let e2 = f2.transform(&x).unwrap();

    assert_eq!(e1, e2, "same seed must produce identical embedding");
}
