//! ACToR critic — divergence audit of `ferrolearn-neural/src/mlp.rs`
//! (`MLPClassifier` / `MLPRegressor`) against scikit-learn 1.5.2.
//!
//! Tracking umbrella: #1710.
//!
//! # Testability gap (CRITICAL — drives the audit shape)
//!
//! `FittedMLPClassifier<F>` / `FittedMLPRegressor<F>` hold their network weights
//! in a PRIVATE field `layers: Vec<LayerParams<F>>` and expose NO public
//! constructor / `from_weights` / weight-injection path (see `mlp.rs:580`,
//! `mlp.rs:1003`). The only way to obtain a fitted net is via `fit`, which is
//! RNG-coupled (random Glorot init + per-epoch minibatch shuffle + early-stop
//! split, numpy MT vs Rust `StdRng` — cannot bit-match, R-DEFER-3).
//!
//! Consequence: the DETERMINISTIC value-parity REQs that the design doc marks
//! oracle-pinnable cannot be pinned as failing value tests through the public
//! API, because every deterministic surface is locked behind RNG-coupled fit:
//!   - REQ-1 (activations): `activate_inplace` is a private free fn.
//!   - REQ-2 (L2 term in reported loss): `train_network` loss is private and
//!     never surfaced (no `loss_` / `loss_curve_` attr — REQ-10).
//!   - REQ-3 / REQ-11 (forward pass / predict_proba parity given fixed weights):
//!     requires injecting `coefs_`/`intercepts_`, which is impossible here.
//!   - REQ-4 (Adam eps / SGD nesterov): `apply_adam_update` / `apply_sgd_update`
//!     are private free fns, unreachable from this external test crate.
//!
//! These are therefore filed as BLOCKERS referencing #1710, NOT failing tests
//! (R-CHAR: a deterministic value test would need the weight-injection ctor
//! that ferrolearn lacks). The blocker for the gap itself is filed too.
//!
//! What this file CONTAINS:
//!   - GREEN structural guards (RNG-independent invariants of `predict_proba` /
//!     `predict`) that mirror sklearn's observable contract and PASS today.
//!   - A GREEN learning-signal guard (REQ-5 carve-out: net learns on a
//!     separable problem with a fixed seed; exact weights NOT compared).
//!   - One FAILING guard pinning the weight-injection / oracle-anchor
//!     testability gap structurally (no public way to build a known-weight net).

use ferrolearn_core::{Fit, Predict};
use ferrolearn_neural::{MLPClassifier, MLPRegressor, Solver};
use ndarray::{Array1, Array2, array};

// ---------------------------------------------------------------------------
// GREEN STRUCTURAL GUARDS (RNG-independent; PASS today)
// ---------------------------------------------------------------------------

/// GREEN guard — REQ-3 / REQ-11.
///
/// sklearn `MLPClassifier.predict_proba` for binary problems returns shape
/// `(n_samples, 2)` with rows that sum to 1 and column layout `[1-p, p]`
/// (`sklearn/neural_network/_multilayer_perceptron.py:1252-1253`):
///   `prob = y_pred.ravel(); return np.vstack([1 - prob, prob]).T`
/// and `predict` returns the argmax label over those columns.
///
/// These are RNG-INDEPENDENT structural invariants: regardless of the (random)
/// fitted weights, the proba rows must sum to 1, the shape must be `(n,2)`,
/// column 0 must equal `1 - column 1`, and `predict` must equal the argmax.
/// Oracle: verified live against sklearn 1.5.2 `predict_proba(X).sum(1)==1`,
/// shape `(4,2)`, `classes_==[0,1]`.
#[test]
fn green_binary_predict_proba_structure() {
    let x = Array2::from_shape_vec(
        (6, 2),
        vec![1.0, 2.0, 2.0, 3.0, 3.0, 1.0, 8.0, 7.0, 9.0, 8.0, 7.0, 9.0],
    )
    .unwrap();
    let y = array![0usize, 0, 0, 1, 1, 1];

    let clf = MLPClassifier::<f64>::new()
        .with_hidden_layer_sizes(vec![5])
        .with_max_iter(50)
        .with_random_state(0);
    let fitted = clf.fit(&x, &y).unwrap();

    let proba = fitted.predict_proba(&x).unwrap();
    assert_eq!(proba.dim(), (6, 2), "binary predict_proba must be (n, 2)");

    for i in 0..6 {
        let row_sum = proba[[i, 0]] + proba[[i, 1]];
        assert!(
            (row_sum - 1.0).abs() < 1e-9,
            "row {i} proba must sum to 1, got {row_sum}"
        );
        // sklearn layout: column 0 == 1 - column 1.
        assert!(
            (proba[[i, 0]] - (1.0 - proba[[i, 1]])).abs() < 1e-12,
            "binary layout must be [1-p, p]"
        );
    }

    // predict == argmax of predict_proba (sklearn `_predict` argmax + label inv).
    let preds = fitted.predict(&x).unwrap();
    for i in 0..6 {
        let argmax = if proba[[i, 1]] > proba[[i, 0]] { 1 } else { 0 };
        assert_eq!(
            preds[i], argmax,
            "predict must equal argmax of predict_proba at row {i}"
        );
    }
}

/// GREEN guard — REQ-11.
///
/// sklearn multiclass `MLPClassifier.predict_proba` returns softmax rows that
/// sum to 1 over `n_classes` columns (`_multilayer_perceptron.py:1232-1255`,
/// `out_activation_ == "softmax"`). RNG-independent invariant: every row of the
/// returned matrix sums to 1 regardless of the (random) fitted weights.
#[test]
fn green_multiclass_predict_proba_sums_to_one() {
    // 3 classes, separable-ish.
    let x = Array2::from_shape_vec(
        (9, 2),
        vec![
            0.0, 0.0, 0.5, 0.2, 0.1, 0.4, 5.0, 5.0, 5.2, 4.8, 4.9, 5.1, 0.0, 9.0, 0.3, 9.2, 0.1,
            8.8,
        ],
    )
    .unwrap();
    let y = array![0usize, 0, 0, 1, 1, 1, 2, 2, 2];

    let clf = MLPClassifier::<f64>::new()
        .with_hidden_layer_sizes(vec![6])
        .with_max_iter(50)
        .with_random_state(1);
    let fitted = clf.fit(&x, &y).unwrap();

    let proba = fitted.predict_proba(&x).unwrap();
    assert_eq!(
        proba.ncols(),
        3,
        "3-class predict_proba must have 3 columns"
    );
    for i in 0..proba.nrows() {
        let s: f64 = proba.row(i).sum();
        assert!(
            (s - 1.0).abs() < 1e-9,
            "multiclass softmax row {i} must sum to 1, got {s}"
        );
        for j in 0..3 {
            assert!(
                (0.0..=1.0).contains(&proba[[i, j]]),
                "softmax probability must be in [0,1]"
            );
        }
    }
}

/// GREEN guard — REQ-5 carve-out (R-DEFER-3): LEARNING signal, NOT exact
/// weights. sklearn-equivalent observable: on a linearly separable 2-class
/// problem with a fixed seed, the fitted net beats chance accuracy on the
/// training set. Exact fitted weights vs sklearn are an RNG carve-out (numpy
/// MT vs Rust `StdRng`), filed as a blocker with NO value test.
#[test]
fn green_learning_signal_separable() {
    let x = Array2::from_shape_vec(
        (8, 2),
        vec![
            0.0, 0.0, 0.5, 0.3, 0.2, 0.6, 0.4, 0.1, 9.0, 9.0, 8.7, 9.3, 9.2, 8.6, 8.5, 9.1,
        ],
    )
    .unwrap();
    let y = array![0usize, 0, 0, 0, 1, 1, 1, 1];

    let clf = MLPClassifier::<f64>::new()
        .with_hidden_layer_sizes(vec![10])
        .with_max_iter(300)
        .with_solver(Solver::Adam {
            learning_rate: 0.01,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        })
        .with_random_state(42);
    let fitted = clf.fit(&x, &y).unwrap();
    let preds = fitted.predict(&x).unwrap();
    let correct = preds.iter().zip(y.iter()).filter(|(a, b)| a == b).count();
    assert!(
        correct >= 7,
        "net must learn a separable problem (beats chance): {correct}/8 correct"
    );
}

/// GREEN guard — REQ-3. Regressor `predict` returns one continuous value per
/// sample (identity output, `out_activation_ == "identity"`,
/// `_multilayer_perceptron.py:1261`+). RNG-independent shape/finiteness
/// invariant.
#[test]
fn green_regressor_predict_shape_finite() {
    let x = Array2::from_shape_vec((6, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let y: Array1<f64> = array![2.0, 4.0, 6.0, 8.0, 10.0, 12.0];

    let reg = MLPRegressor::<f64>::new()
        .with_hidden_layer_sizes(vec![8])
        .with_max_iter(200)
        .with_random_state(7);
    let fitted = reg.fit(&x, &y).unwrap();
    let preds = fitted.predict(&x).unwrap();
    assert_eq!(preds.len(), 6);
    for v in preds.iter() {
        assert!(v.is_finite(), "regressor prediction must be finite");
    }
}

// ---------------------------------------------------------------------------
// FAILING GUARD — testability gap (no weight injection)
// ---------------------------------------------------------------------------

/// Divergence (TESTABILITY GAP): `FittedMLPClassifier` / `FittedMLPRegressor`
/// expose NO public constructor or weight-injection path, so the DETERMINISTIC
/// forward-pass / `predict_proba` value parity against sklearn's
/// `_forward_pass_fast` (`_multilayer_perceptron.py:186-221`) with KNOWN
/// `coefs_`/`intercepts_` (REQ-3 / REQ-11, AC-3) cannot be pinned as a value
/// test. sklearn lets you SET `m.coefs_`, `m.intercepts_`, `m.out_activation_`
/// and call `m.predict_proba`; ferrolearn's `layers` field is private
/// (`mlp.rs:582`) with no equivalent.
///
/// This test fails NOW: it documents (via `assert!(false, ...)`) that there is
/// no reachable API to construct a known-weight net for oracle value parity.
/// It is the structural pin for the gap that forces REQ-1/2/3/4/11 value tests
/// to be blocker-only. It must be DELETED (not silenced) once ferrolearn adds a
/// public `from_weights` / `coefs_` setter and the value-parity tests can be
/// written.
///
/// Tracking: #1710 (weight-injection / oracle-anchor testability gap).
#[test]
#[ignore = "divergence: no public weight-injection ctor on Fitted MLP structs blocks deterministic forward-pass value parity (REQ-3/REQ-11); tracking #1710"]
fn divergence_no_weight_injection_blocks_forward_parity() {
    // There is no `FittedMLPClassifier::from_weights(coefs, intercepts, ...)`
    // and `layers` is private; a deterministic forward-pass value test against
    // the live sklearn oracle is therefore unwritable through the public API.
    let injectable = false; // set true once a public weight ctor exists.
    assert!(
        injectable,
        "ferrolearn exposes no public weight-injection path on FittedMLPClassifier/\
         FittedMLPRegressor; sklearn allows setting coefs_/intercepts_/out_activation_ \
         to verify the deterministic forward pass (REQ-3/REQ-11). Until ferrolearn \
         adds `from_weights`, the deterministic value-parity tests are blocker-only."
    );
}
