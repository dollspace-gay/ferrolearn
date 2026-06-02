//! Divergence pins for `ferrolearn-tree/src/random_forest.rs`
//! (`RandomForestClassifier` / `RandomForestRegressor`) against the live
//! scikit-learn 1.5.2 oracle.
//!
//! These tests pin DETERMINISTIC contracts only — the exact bootstrap +
//! per-tree ensemble at a given `random_state` is the documented numpy-MT19937
//! vs `StdRng` RNG boundary (goal.md / `.design/tree/random_forest.md` #674)
//! and is intentionally NOT pinned here.
//!
//! Reference: scikit-learn 1.5.2 (commit 156ef14),
//! `sklearn/ensemble/_forest.py`.
//!
//! `tests/*.rs` is anti-pattern-gate-exempt: `.unwrap()`/`assert!` are used
//! deliberately (no `panic!`/`unreachable!`).

use ferrolearn_core::traits::{Fit, Predict};
use ferrolearn_tree::{MaxFeatures, RandomForestClassifier, RandomForestRegressor};
use ndarray::{Array1, Array2};

/// Build a deterministic 200-sample, 8-feature binary dataset matching the
/// structure of the design-doc oracle
/// (`np.random.RandomState(7).randn(200, 8)`, label `X[:,0] + noise > 0`).
/// Generated with a tiny self-contained LCG so the Rust test is reproducible
/// without an RNG-stream dependency — the goal is only to obtain a dataset on
/// which shallow trees emit *impure* leaves (soft vote != hard vote), which is
/// a structural property, not an RNG-parity claim.
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
        // Map to roughly N(0,1)-ish in [-3, 3) via uniform sum.
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

/// Argmax of a probability row (lowest index on ties — matches numpy argmax).
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

/// PIN 1 — #670 SOFT-vote consistency (the fixable divergence).
///
/// Divergence: `FittedRandomForestClassifier::predict` (`predict` in
/// `random_forest.rs`) HARD-votes — it tallies per-tree predicted labels
/// (`votes[class_idx] += 1`) and returns the argmax of integer vote counts.
/// sklearn `ForestClassifier.predict`
/// (`sklearn/ensemble/_forest.py:904-907`):
///   `return self.classes_.take(np.argmax(proba, axis=1), axis=0)`
/// is a SOFT vote — argmax of the *mean of per-tree `predict_proba`*.
///
/// This pins the intra-ferrolearn invariant that IS sklearn's documented
/// contract (R-CHAR-3-valid: the asserted relation `predict == argmax(
/// predict_proba)` is sklearn's behavior, NOT a value copied from ferrolearn):
/// for every sample, `predict(X)[i] == classes_[argmax(predict_proba(X)[i])]`.
/// With `max_depth=2` the trees are impure → soft and hard voting disagree, so
/// hard-voting `predict` FAILS this invariant on the disagreement rows.
///
/// Tracking: #670
#[test]
fn divergence_predict_is_soft_vote_argmax_of_proba() {
    let (x, y) = make_dataset();

    let model = RandomForestClassifier::<f64>::new()
        .with_n_estimators(25)
        .with_max_depth(Some(2))
        .with_random_state(1);
    let fitted = model.fit(&x, &y).unwrap();

    let preds = fitted.predict(&x).unwrap();
    let proba = fitted.predict_proba(&x).unwrap();
    let classes = ferrolearn_core::introspection::HasClasses::classes(&fitted).to_vec();

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
         (sklearn _forest.py:904-907 = SOFT vote; ferrolearn hard-votes)",
        x.nrows()
    );
}

/// PIN 2 — constructor defaults (R-DEV-2, deterministic).
///
/// Live sklearn 1.5.2 (`RandomForestClassifier().get_params()`,
/// `sklearn/ensemble/_forest.py:1170`):
///   n_estimators=100, criterion='gini', max_features='sqrt', random_state=None.
///
/// Pins ferrolearn's exposed default surface against the live oracle values
/// (NOT copied from ferrolearn — these are sklearn's documented defaults).
/// Expected to PASS; guards against regression. ABSENT params (regressor
/// `criterion`, `bootstrap`, `max_samples`, `oob_score`, `class_weight`, …)
/// are filed as blockers in the report, not pinned as non-compiling tests.
#[test]
fn defaults_classifier_match_sklearn() {
    const SK_N_ESTIMATORS: usize = 100; // _forest.py:1170 n_estimators=100

    let clf = RandomForestClassifier::<f64>::new();
    assert_eq!(clf.n_estimators, SK_N_ESTIMATORS);
    assert_eq!(clf.max_features, MaxFeatures::Sqrt); // 'sqrt'
    assert!(clf.random_state.is_none()); // None
}

/// PIN 2b — regressor defaults.
///
/// Live sklearn 1.5.2 (`RandomForestRegressor().get_params()`,
/// `sklearn/ensemble/_forest.py:1555`): n_estimators=100,
/// criterion='squared_error', max_features=1.0 (= all features),
/// random_state=None.
#[test]
fn defaults_regressor_match_sklearn() {
    const SK_N_ESTIMATORS: usize = 100; // _forest.py:1555 n_estimators=100

    let reg = RandomForestRegressor::<f64>::new();
    assert_eq!(reg.n_estimators, SK_N_ESTIMATORS);
    // sklearn max_features=1.0 means "use all features" -> MaxFeatures::All.
    assert_eq!(reg.max_features, MaxFeatures::All);
    assert!(reg.random_state.is_none());
}

/// PIN 3 — `random_state` reproducibility (deterministic, NOT numpy parity).
///
/// Two fits with the same `with_random_state(42)` on the same data must give
/// identical `predict`. This is ferrolearn-internal reproducibility (REQ-9),
/// not numpy-MT parity.
#[test]
fn random_state_reproducible() {
    let (x, y) = make_dataset();

    let model = RandomForestClassifier::<f64>::new()
        .with_n_estimators(15)
        .with_max_depth(Some(3))
        .with_random_state(42);

    let f1 = model.fit(&x, &y).unwrap();
    let f2 = model.fit(&x, &y).unwrap();

    let p1 = f1.predict(&x).unwrap();
    let p2 = f2.predict(&x).unwrap();
    assert_eq!(p1, p2);
}

/// PIN 4 — `predict_proba` rows sum to 1 and lie in [0, 1] (deterministic
/// invariant from `sklearn/ensemble/_forest.py:962-963`, mean of per-tree
/// distributions).
#[test]
fn predict_proba_rows_sum_to_one() {
    let (x, y) = make_dataset();

    let model = RandomForestClassifier::<f64>::new()
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

/// PIN 4b — regressor `predict` is a MEAN of per-tree outputs
/// (`sklearn/ensemble/_forest.py:1081`: `y_hat /= len(self.estimators_)`).
///
/// `traverse` is `pub(crate)`, so per-tree leaf access is not available from an
/// external test crate. We pin the deterministic mean STRUCTURE instead: on a
/// constant-target dataset, the mean of any per-tree predictions (each a leaf
/// mean of a bootstrap subset of identical targets) must equal that constant.
/// A `sum`-without-`/n_trees` bug or any non-mean aggregation would break this.
#[test]
fn regressor_predict_is_mean_constant_target() {
    let x = Array2::from_shape_vec((8, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
    // Constant target: every tree's leaf value (mean of a bootstrap subset) is
    // 3.0, so the forest mean is exactly 3.0 regardless of the ensemble.
    let y = Array1::from_vec(vec![3.0; 8]);

    let model = RandomForestRegressor::<f64>::new()
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
