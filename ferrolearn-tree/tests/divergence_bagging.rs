//! Divergence pins for `ferrolearn-tree/src/bagging.rs`
//! (`BaggingClassifier` / `BaggingRegressor`) against the live
//! scikit-learn 1.5.2 oracle.
//!
//! These tests pin DETERMINISTIC / intra-ferrolearn contracts only — the exact
//! per-tree bootstrap-sample + feature-subset draw at a given `random_state` is
//! the documented numpy-MT19937 vs `StdRng` RNG boundary (goal.md /
//! `.design/tree/bagging.md` #719-RNG) and is intentionally NOT pinned here. The
//! pinned contracts (soft-vote consistency, defaults, proba normalization,
//! regressor mean, RNG reproducibility) hold regardless of the RNG stream.
//!
//! Reference: scikit-learn 1.5.2 (commit 156ef14),
//! `sklearn/ensemble/_bagging.py`.
//!
//! `tests/*.rs` is anti-pattern-gate-exempt: `.unwrap()`/`assert!` are used
//! deliberately (no `panic!`/`unreachable!`).

use ferrolearn_core::introspection::HasClasses;
use ferrolearn_core::traits::{Fit, Predict};
use ferrolearn_tree::{BaggingClassifier, BaggingRegressor};
use ndarray::{Array1, Array2};

/// Build a deterministic 300-sample, 6-feature binary dataset mirroring the
/// structure of the design-doc oracle
/// (`np.random.RandomState(3).randn(300, 6)`, label `X[:,0] + X[:,1] +
/// 0.8*noise > 0`). Generated with a tiny self-contained xorshift so the Rust
/// test is reproducible without an RNG-stream dependency — the goal is only to
/// obtain a dataset on which shallow trees (bootstrap row sampling alone) emit
/// *impure* leaves (soft vote != hard vote), a structural property, not an
/// RNG-parity claim. (Live oracle on the real `RandomState(3)` data with
/// `max_depth=2`, `max_features=1.0`, `n_estimators=11`: soft==predict True,
/// soft vs hard differ on 14/300 rows; with `max_features=0.5`: 21/300.)
fn make_dataset() -> (Array2<f64>, Array1<usize>) {
    let n = 300usize;
    let p = 6usize;
    let mut state: u64 = 0x9E3779B97F4A7C15;
    let mut next = || {
        // xorshift64* — deterministic, portable.
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        let z = state.wrapping_mul(0x2545F4914F6CDD1D);
        // Map to roughly N(0,1)-ish in [-3, 3) via uniform spread.
        ((z >> 11) as f64 / (1u64 << 53) as f64) * 6.0 - 3.0
    };
    let mut data = Vec::with_capacity(n * p);
    let mut labels = Vec::with_capacity(n);
    for _ in 0..n {
        let mut row = Vec::with_capacity(p);
        for _ in 0..p {
            row.push(next());
        }
        let noise = next() * 0.8;
        let label = if row[0] + row[1] + noise > 0.0 {
            1usize
        } else {
            0usize
        };
        data.extend_from_slice(&row);
        labels.push(label);
    }
    (
        Array2::from_shape_vec((n, p), data).unwrap(),
        Array1::from_vec(labels),
    )
}

/// Argmax of a probability row (lowest index on ties — matches numpy
/// `np.argmax`, used by `BaggingClassifier.predict` `_bagging.py:914`).
fn argmax(row: &[f64]) -> usize {
    let mut best = 0usize;
    let mut best_v = row[0];
    for (j, &v) in row.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = j;
        }
    }
    best
}

/// PIN 1 (RED — the headline, #718) — SOFT-vote consistency.
///
/// Divergence: `FittedBaggingClassifier::predict` (`predict` in `bagging.rs`,
/// the `votes[class_idx] += 1` / `max_by_key` loop ~476-496) HARD-votes — it
/// tallies one predicted label per tree and returns the argmax of integer vote
/// counts (`max_by_key`, last-index on ties). sklearn
/// `BaggingClassifier.predict` (`sklearn/ensemble/_bagging.py:913-914`):
///   `predicted_probabilitiy = self.predict_proba(X)`
///   `return self.classes_.take((np.argmax(predicted_probabilitiy, axis=1)), axis=0)`
/// where `predict_proba` is the *mean of per-estimator `predict_proba`*
/// (`_bagging.py:964`: `proba = sum(all_proba) / self.n_estimators`) — a SOFT
/// vote.
///
/// Pins the intra-ferrolearn invariant that IS sklearn's documented contract
/// (R-CHAR-3-valid: the asserted relation `predict == argmax(predict_proba)` is
/// sklearn's behavior at `:913-914`, NOT a value copied from the ferrolearn
/// side): for every sample, `predict(X)[i] == classes_[argmax(predict_proba(X)[i])]`.
/// ferrolearn already computes `predict_proba` as the soft mean
/// (`_bagging.py:964`); its hard-voting `predict` is internally inconsistent
/// with it. `max_features=1.0` (full feature set) avoids the separate
/// feature-subset panic (PIN 6 / #719); with `max_depth=2` bootstrap row
/// sampling alone makes the trees impure → soft and hard voting disagree (live
/// oracle: 14/300 rows), so this MUST currently FAIL.
///
/// Tracking: #718
#[test]
fn divergence_predict_is_soft_vote_argmax_of_proba() {
    let (x, y) = make_dataset();

    let model = BaggingClassifier::<f64>::new()
        .with_n_estimators(11)
        .with_max_depth(Some(2))
        .with_random_state(1);
    let fitted = model.fit(&x, &y).unwrap();

    let preds = fitted.predict(&x).unwrap();
    let proba = fitted.predict_proba(&x).unwrap();
    let classes = HasClasses::classes(&fitted).to_vec();

    let mut disagree = 0usize;
    for i in 0..x.nrows() {
        let row: Vec<f64> = proba.row(i).to_vec();
        let soft_label = classes[argmax(&row)];
        if preds[i] != soft_label {
            disagree += 1;
        }
    }

    assert_eq!(
        disagree,
        0,
        "predict diverges from argmax(predict_proba) on {disagree}/{} rows \
         (sklearn _bagging.py:913-914 = SOFT vote = classes_.take(argmax(mean proba)); \
         ferrolearn hard-votes one label per tree via max_by_key)",
        x.nrows()
    );
}

/// PIN 2 (GREEN) — classifier constructor defaults (R-DEV-2, deterministic).
///
/// Live sklearn 1.5.2 `BaggingClassifier().get_params()` (`__init__`
/// `sklearn/ensemble/_bagging.py:813`; defaults block `:314`):
///   n_estimators=10, max_samples=1.0, max_features=1.0, bootstrap=True,
///   bootstrap_features=False, oob_score=False, random_state=None.
/// Verified live: `{'bootstrap': True, 'bootstrap_features': False,
/// 'max_features': 1.0, 'max_samples': 1.0, 'n_estimators': 10,
/// 'oob_score': False, 'random_state': None, ...}`.
///
/// Expected values are sklearn's documented defaults (NOT copied from
/// ferrolearn). Guards against regression of the exposed surface.
#[test]
fn defaults_classifier_match_sklearn() {
    const SK_N_ESTIMATORS: usize = 10; // _bagging.py:314 n_estimators=10

    let clf = BaggingClassifier::<f64>::new();
    assert_eq!(clf.n_estimators, SK_N_ESTIMATORS);
    assert!((clf.max_samples - 1.0).abs() < f64::EPSILON); // max_samples=1.0
    assert!((clf.max_features - 1.0).abs() < f64::EPSILON); // max_features=1.0
    assert!(clf.bootstrap); // bootstrap=True
    assert!(!clf.bootstrap_features); // bootstrap_features=False
    assert!(clf.random_state.is_none()); // random_state=None
}

/// PIN 2b (GREEN) — regressor constructor defaults.
///
/// Live sklearn 1.5.2 `BaggingRegressor().get_params()` (`__init__`
/// `sklearn/ensemble/_bagging.py:1230`) shares the identical defaults block as
/// the classifier (`:314`): n_estimators=10, max_samples=1.0, max_features=1.0,
/// bootstrap=True, bootstrap_features=False, random_state=None.
#[test]
fn defaults_regressor_match_sklearn() {
    const SK_N_ESTIMATORS: usize = 10; // _bagging.py:314 n_estimators=10

    let reg = BaggingRegressor::<f64>::new();
    assert_eq!(reg.n_estimators, SK_N_ESTIMATORS);
    assert!((reg.max_samples - 1.0).abs() < f64::EPSILON);
    assert!((reg.max_features - 1.0).abs() < f64::EPSILON);
    assert!(reg.bootstrap);
    assert!(!reg.bootstrap_features);
    assert!(reg.random_state.is_none());
}

/// PIN 3 (GREEN) — `predict_proba` rows sum to 1 and lie in [0, 1].
///
/// Deterministic invariant from `sklearn/ensemble/_bagging.py:964`
/// (`proba = sum(all_proba) / self.n_estimators` — mean of per-estimator
/// distributions, each of which sums to 1 over classes). Holds regardless of
/// the RNG stream. `max_features=1.0` avoids the feature-subset panic (PIN 6).
#[test]
fn predict_proba_rows_sum_to_one() {
    let (x, y) = make_dataset();

    let model = BaggingClassifier::<f64>::new()
        .with_n_estimators(10)
        .with_max_depth(Some(2))
        .with_random_state(7);
    let fitted = model.fit(&x, &y).unwrap();
    let proba = fitted.predict_proba(&x).unwrap();

    for i in 0..x.nrows() {
        let mut s = 0.0;
        for j in 0..proba.ncols() {
            let v = proba[[i, j]];
            assert!(
                (0.0..=1.0).contains(&v),
                "proba[{i},{j}] = {v} outside [0,1]"
            );
            s += v;
        }
        assert!((s - 1.0).abs() < 1e-9, "row {i} sums to {s}, expected 1.0");
    }
}

/// PIN 4 (GREEN) — regressor `predict` is a MEAN of per-tree outputs
/// (`sklearn/ensemble/_bagging.py:1299`: `y_hat /= self.n_estimators`).
///
/// `traverse` / leaf access is `pub(crate)`, not reachable from an external
/// test crate. We pin the deterministic mean STRUCTURE instead: on a
/// constant-target dataset every per-tree leaf value is that constant, so the
/// ensemble mean must equal it exactly. A `sum`-without-`/n_trees` bug or any
/// non-mean aggregation would break this.
#[test]
fn regressor_predict_is_mean_constant_target() {
    let x = Array2::from_shape_vec((8, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
    // Constant target: every tree's leaf value is 3.0, so the ensemble mean is
    // exactly 3.0 regardless of the (RNG-driven) bootstrap draws.
    let y = Array1::from_vec(vec![3.0; 8]);

    let model = BaggingRegressor::<f64>::new()
        .with_n_estimators(10)
        .with_random_state(3);
    let fitted = model.fit(&x, &y).unwrap();
    let preds = fitted.predict(&x).unwrap();

    for i in 0..x.nrows() {
        assert!(
            (preds[i] - 3.0).abs() < 1e-12,
            "predict[{i}] = {} != mean 3.0 (mean-aggregation broken)",
            preds[i]
        );
    }
}

/// PIN 5 (GREEN) — `random_state` reproducibility (REQ-10, ferrolearn-internal
/// determinism, NOT numpy parity).
///
/// Two `fit` calls with the same `with_random_state(42)` on the same data must
/// produce identical `predict` AND `predict_proba`. The exact ensemble at a
/// seed vs numpy-MT is the documented RNG boundary; this only asserts
/// ferrolearn is deterministic w.r.t. its own seed. `max_features=1.0` avoids
/// the feature-subset panic (PIN 6).
#[test]
fn random_state_reproducible() {
    let (x, y) = make_dataset();

    let model = BaggingClassifier::<f64>::new()
        .with_n_estimators(12)
        .with_max_depth(Some(3))
        .with_random_state(42);

    let f1 = model.fit(&x, &y).unwrap();
    let f2 = model.fit(&x, &y).unwrap();

    let p1 = f1.predict(&x).unwrap();
    let p2 = f2.predict(&x).unwrap();
    assert_eq!(p1, p2);

    let pp1 = f1.predict_proba(&x).unwrap();
    let pp2 = f2.predict_proba(&x).unwrap();
    assert_eq!(pp1, pp2);
}

/// PIN 6 (RED — #719) — feature subsampling (`max_features < 1.0`) must not
/// panic.
///
/// Divergence: `BaggingClassifier::fit` with `max_features < 1.0` draws a
/// per-tree feature subset (`n_feature_draw = ceil(n_features * max_features)`),
/// then calls `aggregate_tree_importances(&trees, Some(&feature_indices), ...)`.
/// Each tree's `Node::Split.feature` ALREADY holds the ORIGINAL feature index
/// (the candidate features in `find_best_split` are the subset's original
/// indices), but `aggregate_tree_importances` (`decision_tree.rs`, the
/// `Some(map) => map[t][*feature]` branch) treats `feature` as a SUBSET-relative
/// index and re-maps it through `feature_indices[t]` — a double mapping that
/// indexes out of bounds and PANICS (`index out of bounds: the len is 3 but the
/// index is 4`) inside `fit`, before any prediction.
///
/// sklearn's `BaggingClassifier(max_features=0.5).fit(X, y)` returns normally
/// (feature subsampling is a supported, default-adjacent code path —
/// `_generate_bagging_indices` `_bagging.py:67`, `_max_features` sizing
/// `:487-498`). This pin asserts `fit` simply SUCCEEDS (no panic) for
/// `max_features=0.5` on a 6-feature dataset where the subset size (3) is
/// smaller than an original split-feature index — it currently aborts the test
/// process with a panic, so this MUST currently FAIL.
///
/// (R-CHAR-3-valid: the asserted behavior — "fit with feature subsampling does
/// not crash" — is sklearn's observable contract, not a value copied from
/// ferrolearn.)
///
/// Tracking: #719
#[test]
fn divergence_feature_subsample_fit_does_not_panic() {
    let (x, y) = make_dataset();

    let model = BaggingClassifier::<f64>::new()
        .with_n_estimators(11)
        .with_max_depth(Some(2))
        .with_max_features(0.5)
        .with_random_state(1);

    let result = model.fit(&x, &y);
    assert!(
        result.is_ok(),
        "BaggingClassifier::fit with max_features=0.5 must succeed (sklearn \
         supports feature subsampling, _bagging.py:67/:487-498); ferrolearn \
         double-maps the original split-feature index through feature_indices[t] \
         in aggregate_tree_importances and panics"
    );
}
