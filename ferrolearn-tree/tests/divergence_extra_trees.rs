//! Divergence pins for `ferrolearn-tree/src/extra_trees_ensemble.rs`
//! (`ExtraTreesClassifier` / `ExtraTreesRegressor`) against the live
//! scikit-learn 1.5.2 oracle.
//!
//! These tests pin DETERMINISTIC contracts only — the exact per-tree random
//! split thresholds + optional bootstrap at a given `random_state` is the
//! documented numpy-MT19937 vs `StdRng` RNG boundary
//! (goal.md / `.design/tree/extra_trees_ensemble.md` #693) and is intentionally
//! NOT pinned here. The pinned contracts (soft-vote consistency, defaults,
//! proba normalization, regressor mean, RNG reproducibility) hold regardless of
//! the RNG stream.
//!
//! Reference: scikit-learn 1.5.2 (commit 156ef14),
//! `sklearn/ensemble/_forest.py` (`ExtraTrees*` inherit `Forest*` aggregation).
//!
//! `tests/*.rs` is anti-pattern-gate-exempt: `.unwrap()`/`assert!` are used
//! deliberately (no `panic!`/`unreachable!`).

use ferrolearn_core::introspection::HasClasses;
use ferrolearn_core::traits::{Fit, Predict};
use ferrolearn_tree::{ExtraTreesClassifier, ExtraTreesRegressor, MaxFeatures};
use ndarray::{Array1, Array2};

/// Build a deterministic 200-sample, 8-feature binary dataset matching the
/// structure of the design-doc oracle
/// (`np.random.RandomState(7).randn(200, 8)`, label `X[:,0] + 0.5*noise > 0`).
/// Generated with a tiny self-contained xorshift so the Rust test is
/// reproducible without an RNG-stream dependency — the goal is only to obtain a
/// dataset on which shallow trees emit *impure* leaves (soft vote != hard
/// vote), a structural property, not an RNG-parity claim. (Live oracle on the
/// real `RandomState(7)` data with `max_depth=2`: soft==predict True,
/// hard==predict False, 54/200 rows differ.)
fn make_dataset() -> (Array2<f64>, Array1<usize>) {
    let n = 200usize;
    let p = 8usize;
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
        let noise = next() * 0.5;
        let label = if row[0] + noise > 0.0 { 1usize } else { 0usize };
        data.extend_from_slice(&row);
        labels.push(label);
    }
    (
        Array2::from_shape_vec((n, p), data).unwrap(),
        Array1::from_vec(labels),
    )
}

/// Argmax of a probability row (lowest index on ties — matches numpy argmax,
/// `np.argmax`, used by `ForestClassifier.predict` `_forest.py:907`).
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

/// PIN 1 (RED — the headline, #679) — SOFT-vote consistency.
///
/// Divergence: `FittedExtraTreesClassifier::predict` (`predict` in
/// `extra_trees_ensemble.rs`) HARD-votes — it tallies one predicted label per
/// tree (`votes[class_idx] += 1`) and returns the argmax of integer vote
/// counts. sklearn `ExtraTreesClassifier.predict` (inherited from
/// `ForestClassifier.predict`, `sklearn/ensemble/_forest.py:907`):
///   `return self.classes_.take(np.argmax(proba, axis=1), axis=0)`
/// where `proba` is the *mean of per-tree `predict_proba`*
/// (`_forest.py:963`: `proba /= len(self.estimators_)`) — a SOFT vote.
///
/// Pins the intra-ferrolearn invariant that IS sklearn's documented contract
/// (R-CHAR-3-valid: the asserted relation `predict == argmax(predict_proba)` is
/// sklearn's behavior, NOT a value copied from the ferrolearn side): for every
/// sample, `predict(X)[i] == classes_[argmax(predict_proba(X)[i])]`. ferrolearn
/// already computes `predict_proba` as the soft mean (`:963`); its hard-voting
/// `predict` is internally inconsistent with it. With `max_depth=2` the trees
/// are impure → soft and hard voting disagree, so this MUST currently FAIL.
///
/// Tracking: #679
#[test]
fn divergence_predict_is_soft_vote_argmax_of_proba() {
    let (x, y) = make_dataset();

    let model = ExtraTreesClassifier::<f64>::new()
        .with_n_estimators(25)
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
         (sklearn _forest.py:907 = SOFT vote = classes_.take(argmax(mean proba)); \
         ferrolearn hard-votes one label per tree)",
        x.nrows()
    );
}

/// PIN 2 (GREEN) — classifier constructor defaults (R-DEV-2, deterministic).
///
/// Live sklearn 1.5.2 `ExtraTreesClassifier().get_params()`
/// (`__init__` `sklearn/ensemble/_forest.py:2224`):
///   n_estimators=100 (`:2225`), criterion='gini' (`:2228`),
///   max_features='sqrt' (`:2233`), bootstrap=False (`:2236`).
/// Verified live: `{'n_estimators': 100, 'criterion': 'gini',
/// 'max_features': 'sqrt', 'bootstrap': False}`.
///
/// Expected values are sklearn's documented defaults (NOT copied from
/// ferrolearn). Guards against regression of the exposed surface.
#[test]
fn defaults_classifier_match_sklearn() {
    const SK_N_ESTIMATORS: usize = 100; // _forest.py:2225 n_estimators=100

    let clf = ExtraTreesClassifier::<f64>::new();
    assert_eq!(clf.n_estimators, SK_N_ESTIMATORS);
    assert_eq!(clf.max_features, MaxFeatures::Sqrt); // 'sqrt' :2233
    assert!(!clf.bootstrap); // bootstrap=False :2236 (asymmetry vs RandomForest)
    assert!(clf.random_state.is_none()); // random_state=None
}

/// PIN 2b (GREEN) — regressor constructor defaults.
///
/// Live sklearn 1.5.2 `ExtraTreesRegressor().get_params()`
/// (`__init__` `sklearn/ensemble/_forest.py:2564`):
///   n_estimators=100, criterion='squared_error' (`:2568`),
///   max_features=1.0 (`:2573`, = all features), bootstrap=False.
/// Verified live: `{'n_estimators': 100, 'criterion': 'squared_error',
/// 'max_features': 1.0, 'bootstrap': False}`.
#[test]
fn defaults_regressor_match_sklearn() {
    const SK_N_ESTIMATORS: usize = 100; // _forest.py:2564 n_estimators=100

    let reg = ExtraTreesRegressor::<f64>::new();
    assert_eq!(reg.n_estimators, SK_N_ESTIMATORS);
    // sklearn max_features=1.0 means "use all features" -> MaxFeatures::All.
    assert_eq!(reg.max_features, MaxFeatures::All); // :2573
    assert!(!reg.bootstrap); // bootstrap=False
    assert!(reg.random_state.is_none());
}

/// PIN 3 (GREEN) — `predict_proba` rows sum to 1 and lie in [0, 1].
///
/// Deterministic invariant from `sklearn/ensemble/_forest.py:963`
/// (`proba /= len(self.estimators_)` — mean of per-tree distributions, each of
/// which sums to 1 over classes). Holds regardless of the RNG stream.
#[test]
fn predict_proba_rows_sum_to_one() {
    let (x, y) = make_dataset();

    let model = ExtraTreesClassifier::<f64>::new()
        .with_n_estimators(20)
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
/// (`sklearn/ensemble/_forest.py:1081`: `y_hat /= len(self.estimators_)`).
///
/// `traverse` is `pub(crate)`, so per-tree leaf access is not reachable from an
/// external test crate. We pin the deterministic mean STRUCTURE instead: on a
/// constant-target dataset every per-tree leaf value is that constant, so the
/// forest mean must equal it exactly. A `sum`-without-`/n_trees` bug or any
/// non-mean aggregation would break this.
#[test]
fn regressor_predict_is_mean_constant_target() {
    let x = Array2::from_shape_vec((8, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
    // Constant target: every tree's leaf value is 3.0, so the forest mean is
    // exactly 3.0 regardless of the (RNG-driven) ensemble.
    let y = Array1::from_vec(vec![3.0; 8]);

    let model = ExtraTreesRegressor::<f64>::new()
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

/// PIN 5 (GREEN) — `random_state` reproducibility (REQ-9, ferrolearn-internal
/// determinism, NOT numpy parity).
///
/// Two `fit` calls with the same `with_random_state(42)` on the same data must
/// produce identical `predict` AND `predict_proba`. The exact ensemble at a
/// seed vs numpy-MT is the documented RNG boundary (#693); this only asserts
/// ferrolearn is deterministic w.r.t. its own seed.
#[test]
fn random_state_reproducible() {
    let (x, y) = make_dataset();

    let model = ExtraTreesClassifier::<f64>::new()
        .with_n_estimators(15)
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
