//! Divergence pins for `ferrolearn-tree/src/isolation_forest.rs`
//! (`IsolationForest`) against the live scikit-learn 1.5.2 oracle.
//!
//! These tests pin DETERMINISTIC / intra-ferrolearn contracts only — the exact
//! per-tree subsample draw + split stream at a given `random_state` is the
//! documented numpy-MT19937-vs-`StdRng` RNG boundary (goal.md /
//! `.design/tree/isolation_forest.md` #732) and is intentionally NOT pinned
//! here. The headline pin (`score_samples` sign/range) holds regardless of the
//! RNG stream: sklearn ALWAYS returns `<= 0`, ferrolearn ALWAYS returns `> 0`.
//!
//! Reference: scikit-learn 1.5.2 (commit 156ef14),
//! `sklearn/ensemble/_iforest.py`.
//!
//! `tests/*.rs` is anti-pattern-gate-exempt: `.unwrap()`/`assert!` are used
//! deliberately (no `panic!`/`unreachable!`).

use ferrolearn_core::traits::{Fit, Predict};
use ferrolearn_tree::{Contamination, IsolationForest};
use ndarray::Array2;

/// Deterministic 52-row, 2-feature dataset: 50 normal points around the origin
/// plus two clear outliers — mirrors the design-doc oracle
/// (`np.vstack([RandomState(0).randn(50,2), [[100,100],[-80,90]]])`). Generated
/// with a tiny self-contained xorshift so the Rust test is reproducible without
/// an RNG-stream dependency — the point is only to obtain a small fitted forest
/// to assert the sign/range contract on, NOT an RNG-parity claim.
fn make_dataset() -> Array2<f64> {
    let n_normal = 50usize;
    let p = 2usize;
    let mut state: u64 = 0x9E3779B97F4A7C15;
    let mut next = || {
        // xorshift64* — deterministic, portable.
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        let z = state.wrapping_mul(0x2545F4914F6CDD1D);
        // Map to roughly N(0,1)-ish in [-3, 3).
        ((z >> 11) as f64 / (1u64 << 53) as f64) * 6.0 - 3.0
    };
    let mut data = Vec::with_capacity((n_normal + 2) * p);
    for _ in 0..n_normal {
        data.push(next());
        data.push(next());
    }
    // Two clear outliers.
    data.extend_from_slice(&[100.0, 100.0, -80.0, 90.0]);
    Array2::from_shape_vec((n_normal + 2, p), data).unwrap()
}

/// PIN 1 (RED — the HEADLINE, #726, R-DEV-3) — `score_samples` sign/range.
///
/// Divergence: `FittedIsolationForest::score_samples` (`score_samples` in
/// `isolation_forest.rs`, `scores[i] = f64::powf(2.0, -mean_path / c_n)` — the
/// `scores[i] = ...` line ~212) returns the POSITIVE paper anomaly score
/// `+2^(-mean/c) ∈ (0, 1]` (HIGHER = more ANOMALOUS).
///
/// sklearn `IsolationForest.score_samples` is the OPPOSITE
/// (`sklearn/ensemble/_iforest.py:451`):
///   `# Take the opposite of the scores as bigger is better (here less abnormal)`
///   `return -self._compute_chunked_score_samples(X)`
/// so `score_samples(X) = -2^(-mean/c) ∈ [-1, 0]` — ALWAYS `<= 0`, HIGHER =
/// more NORMAL. Docstring (`:404-405`): "The lower, the more abnormal. Negative
/// scores represent outliers, positive scores represent inliers."
///
/// Live oracle fact (R-CHAR-3 — the asserted relation is sklearn's behavior,
/// NOT a value copied from ferrolearn):
///   python3 -c "import numpy as np; from sklearn.ensemble import IsolationForest;
///     rng=np.random.RandomState(0); X=np.vstack([rng.randn(50,2),[[100.,100.],[-80.,90.]]]);
///     s=IsolationForest(random_state=0).fit(X).score_samples(X);
///     print(bool((s<=0).all()), float(s.min()), float(s.max()))"
///   => True -0.8495421126357149 -0.3566161713654292   (every score <= 0)
///
/// ferrolearn returns strictly POSITIVE scores, so this MUST currently FAIL.
///
/// Tracking: #726
#[test]
fn divergence_score_samples_sign_le_zero() {
    let x = make_dataset();
    let model = IsolationForest::<f64>::new()
        .with_n_estimators(100)
        .with_random_state(0);
    let fitted = model.fit(&x, &()).unwrap();
    let scores = fitted.score_samples(&x).unwrap();

    assert!(
        scores.iter().all(|&s| s <= 0.0),
        "score_samples must be <= 0 for every row (sklearn _iforest.py:451 \
         `return -self._compute_chunked_score_samples(X)`, range [-1,0], live \
         oracle: all scores <= 0, min -0.8495); ferrolearn returns the POSITIVE \
         paper score `+2^(-mean/c)` ∈ (0,1] — inverted sign. scores = {scores:?}"
    );
}

/// PIN 2 (GREEN) — constructor defaults (R-DEV-2, deterministic).
///
/// Live sklearn 1.5.2 `IsolationForest().get_params()` (`__init__`
/// `sklearn/ensemble/_iforest.py:221`; defaults `:224-232`):
///   n_estimators=100, max_samples='auto', random_state=None.
/// Verified live: `{'n_estimators': 100, 'max_samples': 'auto',
/// 'random_state': None, ...}`.
///
/// ferrolearn cannot express the `'auto'` string; it defaults `max_samples=256`
/// then applies `.min(n_samples)` in `fit`, which is numerically equivalent to
/// sklearn `'auto' = min(256, n_samples)` (`:304`). We pin the matched parts:
/// `n_estimators == 100`, `max_samples == 256` (the `min(256, n)` cap value),
/// `random_state == None`. Expected values are sklearn's documented defaults,
/// NOT copied from ferrolearn.
#[test]
fn defaults_match_sklearn() {
    const SK_N_ESTIMATORS: usize = 100; // _iforest.py:224 n_estimators=100
    const SK_MAX_SAMPLES_AUTO_CAP: usize = 256; // _iforest.py:304 min(256, n_samples)

    let model = IsolationForest::<f64>::new();
    assert_eq!(model.n_estimators, SK_N_ESTIMATORS);
    assert_eq!(model.max_samples, SK_MAX_SAMPLES_AUTO_CAP);
    assert!(model.random_state.is_none()); // random_state=None
}

/// PIN 2b (GREEN) — effective `max_samples` is `min(256, n_samples)` for the
/// 'auto'-equivalent default.
///
/// sklearn `'auto' -> max_samples = min(256, n_samples)` (`_iforest.py:303-304`).
/// On a 52-row X, sklearn `max_samples_ = 52`; ferrolearn `256.min(52) = 52`.
/// `max_samples` is not exposed on `FittedIsolationForest`, so we pin the
/// observable consequence: with a default `max_samples` (256) on a smaller
/// dataset, `fit` succeeds and `predict` returns one label per row — the
/// `.min(n_samples)` clamp is exercised (the build would otherwise draw 256
/// indices into a 52-row matrix, which is fine, but a default forest on small
/// data must still produce n_samples predictions).
#[test]
fn max_samples_auto_clamps_to_n_samples() {
    let x = make_dataset(); // 52 rows
    let model = IsolationForest::<f64>::new() // default max_samples = 256 > 52
        .with_n_estimators(20)
        .with_random_state(0);
    let fitted = model.fit(&x, &()).unwrap();
    let preds = fitted.predict(&x).unwrap();
    assert_eq!(preds.len(), x.nrows());
}

/// PIN 3 (GREEN) — `random_state` reproducibility (REQ-9, ferrolearn-internal
/// determinism, NOT numpy parity).
///
/// Two `fit` calls with the same `with_random_state(7)` on the same data must
/// produce identical `score_samples` AND `predict`. The exact ensemble at a
/// seed vs numpy-MT is the documented RNG boundary (#732); this only asserts
/// ferrolearn is deterministic w.r.t. its own seed.
#[test]
fn random_state_reproducible() {
    let x = make_dataset();
    let model = IsolationForest::<f64>::new()
        .with_n_estimators(40)
        .with_random_state(7);

    let f1 = model.fit(&x, &()).unwrap();
    let f2 = model.fit(&x, &()).unwrap();

    let s1 = f1.score_samples(&x).unwrap();
    let s2 = f2.score_samples(&x).unwrap();
    assert_eq!(s1, s2, "score_samples not reproducible at a fixed seed");

    let p1 = f1.predict(&x).unwrap();
    let p2 = f2.predict(&x).unwrap();
    assert_eq!(p1, p2, "predict not reproducible at a fixed seed");
}

/// PIN 4 (CONTRACT) — `offset_ == -0.5` for `contamination='auto'`
/// (REQ-6, R-DEV-2/R-DEV-3, deterministic).
///
/// sklearn `fit` (`sklearn/ensemble/_iforest.py:341-345`):
///   `if self.contamination == "auto":`
///   `    # 0.5 plays a special role as described in the original paper.`
///   `    # we take the opposite as we consider the opposite of their score.`
///   `    self.offset_ = -0.5`
/// Live oracle fact (R-CHAR-3): `IsolationForest(random_state=0).fit(X).offset_`
/// is exactly `-0.5` (the 'auto' default). The value `-0.5` is a sklearn
/// symbolic constant, not copied from ferrolearn.
#[test]
fn contract_offset_auto_is_minus_half() {
    let x = make_dataset();
    let model = IsolationForest::<f64>::new()
        .with_n_estimators(100)
        .with_contamination_auto()
        .with_random_state(0);
    let fitted = model.fit(&x, &()).unwrap();
    assert!(
        (fitted.offset() - (-0.5)).abs() < 1e-12,
        "offset_ must be -0.5 for contamination='auto' (sklearn \
         _iforest.py:341-345 `self.offset_ = -0.5`); got {}",
        fitted.offset()
    );
    // `Contamination::Auto` is the default (sklearn contamination='auto').
    assert_eq!(
        IsolationForest::<f64>::new().contamination,
        Contamination::Auto
    );
}

/// PIN 5 (CONTRACT) — `decision_function(X) == score_samples(X) - offset_`
/// (REQ-5, R-DEV-3, deterministic regardless of RNG stream).
///
/// sklearn `decision_function` (`sklearn/ensemble/_iforest.py:410`):
///   `return self.score_samples(X) - self.offset_`
/// Live oracle fact (R-CHAR-3): for any fitted model,
/// `np.allclose(m.decision_function(X), m.score_samples(X) - m.offset_)` is
/// `True`. The asserted identity is sklearn's definition, not a ferrolearn
/// value.
#[test]
fn contract_decision_function_equals_score_minus_offset() {
    let x = make_dataset();
    let model = IsolationForest::<f64>::new()
        .with_n_estimators(100)
        .with_contamination(0.1)
        .with_random_state(0);
    let fitted = model.fit(&x, &()).unwrap();
    let scores = fitted.score_samples(&x).unwrap();
    let decision = fitted.decision_function(&x).unwrap();
    let off = fitted.offset();
    for i in 0..x.nrows() {
        assert!(
            (decision[i] - (scores[i] - off)).abs() < 1e-12,
            "decision_function[{i}] must equal score_samples[{i}] - offset_ \
             (sklearn _iforest.py:410); df={}, score={}, offset_={off}",
            decision[i],
            scores[i]
        );
    }
}

/// PIN 6 (CONTRACT) — `predict(X) == -1 where decision_function(X) < 0 else 1`
/// (REQ-8, R-DEV-3, deterministic regardless of RNG stream).
///
/// sklearn `predict` (`sklearn/ensemble/_iforest.py:374-378`):
///   `decision_func = self.decision_function(X)`
///   `is_inlier = np.ones_like(decision_func, dtype=int)`
///   `is_inlier[decision_func < 0] = -1`
///   `return is_inlier`
/// Live oracle fact (R-CHAR-3):
/// `(m.predict(X) == np.where(m.decision_function(X) < 0, -1, 1)).all()` is
/// `True`. The asserted equality is sklearn's definition.
#[test]
fn contract_predict_agrees_with_decision_function_sign() {
    let x = make_dataset();
    let model = IsolationForest::<f64>::new()
        .with_n_estimators(100)
        .with_contamination(0.1)
        .with_random_state(0);
    let fitted = model.fit(&x, &()).unwrap();
    let decision = fitted.decision_function(&x).unwrap();
    let preds = fitted.predict(&x).unwrap();
    for i in 0..x.nrows() {
        let expected = if decision[i] < 0.0 { -1 } else { 1 };
        assert_eq!(
            preds[i], expected,
            "predict[{i}] must be -1 where decision_function < 0 else 1 \
             (sklearn _iforest.py:374-378); df={}",
            decision[i]
        );
    }
}

/// PIN 7 (RED — NEW divergence, #732) — single-sample `score_samples` /
/// `decision_function` must be finite (`-0.5` / `0.0`), not `NaN`.
///
/// When a model is fit on a SINGLE training sample, `max_samples_ == 1`, so
/// `c(max_samples_) = _average_path_length([1]) = 0` and the denominator
/// `n_estimators * c(max_samples_) == 0`. sklearn guards this division
/// (`sklearn/ensemble/_iforest.py:516-522`):
///   `scores = 2 ** (`
///   `    -np.divide(`
///   `        depths, denominator, out=np.ones_like(depths), where=denominator != 0`
///   `    )`
///   `)`
/// With `denominator == 0` the ratio stays at the `out=ones` default, so
/// `raw = 2**(-1) = 0.5` and `_score_samples = -0.5`.
///
/// Live oracle fact (R-CHAR-3 — value is sklearn's, NOT copied from ferrolearn):
///   python3 -c "import numpy as np; from sklearn.ensemble import IsolationForest;
///     X=np.array([[1.,2.,3.]]);
///     m=IsolationForest(n_estimators=10, random_state=0).fit(X);
///     print(m.score_samples(X).tolist(), m.decision_function(X).tolist())"
///   => [-0.5] [0.0]
///
/// ferrolearn `score_samples` (`isolation_forest.rs`) computes
/// `-2^(-mean_path / c_n)` with `c_n = average_path_length(self.max_samples)
/// = c(1) = 0`, giving `0.0 / 0.0 = NaN`; both `score_samples` and the
/// `decision_function` derived from it return `NaN` (probed: `score=[NaN]`,
/// `decision=[NaN]`). This MUST currently FAIL.
///
/// Tracking: #732
#[test]
fn divergence_single_sample_score_is_minus_half() {
    // sklearn symbolic constants (_iforest.py:516-522 + :561 c(1)=0 guard).
    const SK_SINGLE_SCORE: f64 = -0.5; // 2**(-1) default, negated (_iforest.py:519-522)
    const SK_SINGLE_DECISION: f64 = 0.0; // score_samples - offset_ = -0.5 - (-0.5)

    let x = Array2::<f64>::from_shape_vec((1, 3), vec![1.0, 2.0, 3.0]).unwrap();
    let model = IsolationForest::<f64>::new()
        .with_n_estimators(10)
        .with_random_state(0);
    let fitted = model.fit(&x, &()).unwrap();
    let score = fitted.score_samples(&x).unwrap()[0];
    let decision = fitted.decision_function(&x).unwrap()[0];

    assert!(
        score.is_finite(),
        "single-sample score_samples must be finite (-0.5), not NaN/inf \
         (sklearn _iforest.py:519-522 `np.divide(..., out=ones, where=denominator!=0)` \
         keeps ratio=1 when c(max_samples_=1)=0 -> 2^-1 -> -0.5); got {score}"
    );
    assert!(
        (score - SK_SINGLE_SCORE).abs() < 1e-12,
        "single-sample score_samples must equal sklearn -0.5; got {score}"
    );
    assert!(
        (decision - SK_SINGLE_DECISION).abs() < 1e-12,
        "single-sample decision_function must equal sklearn 0.0; got {decision}"
    );
}
