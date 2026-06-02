//! Divergence pins for `ferrolearn-tree/src/adaboost_regressor.rs`
//! (`AdaBoostRegressor` — AdaBoost.R2) against scikit-learn 1.5.2.
//!
//! Reference: scikit-learn 1.5.2 (commit 156ef14),
//! `sklearn/ensemble/_weight_boosting.py` (`AdaBoostRegressor._boost`
//! :1117-1213; `_get_median_predict` :1215-1230; `__init__`/`_validate_estimator`
//! :1094/:1115).
//!
//! Boundary note (`.design/tree/adaboost_regressor.md`, REQ-7): sklearn fits each
//! round's tree on a *stochastic numpy weighted bootstrap*
//! (`random_state.choice(replace=True, p=sample_weight)`, :1162-1167) while
//! ferrolearn uses a *deterministic systematic resample* with no RNG. End-to-end
//! `.fit()` parity at a seed is therefore INFEASIBLE, so the headline REQ-5 pin
//! is grounded in the sklearn SOURCE FORMULA (:1206/:1210), not a sklearn run.
//! The construction below is chosen so BOTH rounds resample to the full dataset
//! (`[0,1,2,3,4]`) under either exponent, making the per-round tree identical and
//! the only varying quantity the reweighted distribution — which isolates the
//! `* learning_rate` exponent factor cleanly through `estimator_weights()`.
//!
//! `tests/*.rs` is anti-pattern-gate-exempt: `.unwrap()`/`assert!` are used
//! deliberately (no `panic!`/`unreachable!`).

use ferrolearn_core::traits::{Fit, Predict};
use ferrolearn_tree::decision_tree::Node;
use ferrolearn_tree::{AdaBoostLoss, AdaBoostRegressor};
use ndarray::{Array1, Array2};

/// PIN 1 (RED — the headline, #703) — sample-weight reweight exponent is
/// MISSING the `* learning_rate` factor.
///
/// Divergence: `AdaBoostRegressor::fit` (`adaboost_regressor.rs`, the reweight
/// loop) does
///   `weights[i] *= beta.powf(F::one() - losses[i])`   // exponent = (1 - loss_i)
/// whereas sklearn `_weight_boosting.py:1209-1211` does
///   `sample_weight[mask] *= np.power(beta, (1.0 - error_vect) * self.learning_rate)`
/// — exponent `(1 - loss_i) * learning_rate`. The two agree ONLY at
/// `learning_rate == 1.0`; for any other value the reweighted distribution
/// differs, which feeds the next round's weighted-average loss → `beta` →
/// `estimator_weight`, making it observable through `estimator_weights()`.
///
/// Construction (fully analytic; every number grounded in the sklearn source,
/// NOT copied from ferrolearn — R-CHAR-3):
///   X = [[0],[1],[2],[3],[4]], y = [6,0,4,8,7], `max_depth=2`,
///   `learning_rate=0.5`, `n_estimators=2`, `loss=linear`.
///
/// ITER 1 (uniform weights → systematic resample = identity → full data):
///   depth-2 regression tree predicts `[6,2,2,8,7]`
///     (live-confirmed `DecisionTreeRegressor(max_depth=2).fit(X,y).predict(X)`;
///      this tree is inherited from oracle-verified `decision_tree.rs`),
///   abs error `[0,2,2,0,0]`, `error_max=2`, linear loss `[0,1,1,0,0]`,
///   weighted avg loss `= 0.4`, `beta1 = 0.4/(1-0.4) = 2/3`
///     (`:1203`), `estimator_weight[0] = 0.5 * ln(1/beta1) = 0.2027325540540821`
///     (`:1206`).  ── this is computed BEFORE the reweight, so it is the SAME
///   under both exponents and is GREEN-anchored below.
///
/// REWEIGHT (the divergent step):
///   sklearn (`:1210`, exponent `(1-loss)*lr`):
///     w ∝ [1/5·β^(1·0.5), 1/5·β^0, 1/5·β^0, 1/5·β^(1·0.5), 1/5·β^(1·0.5)]
///       → normalized `[0.18350342, 0.22474487, 0.22474487, 0.18350342, 0.18350342]`.
///   ferrolearn (exponent `(1-loss)`):
///     → normalized `[0.16666667, 0.25, 0.25, 0.16666667, 0.16666667]`.
///
/// ITER 2 (both still resample to the full set → identical depth-2 tree, preds
/// `[6,2,2,8,7]`, loss `[0,1,1,0,0]`); only the weighted-avg loss differs because
/// the WEIGHTS differ:
///   sklearn:    avg2 = (w_skl · loss).sum()/w_skl.sum() = 0.4494897427831781
///               → beta2 = 0.816496580927726
///               → `estimator_weight[1] = 0.5 * ln(1/beta2) = 0.10136627702704105`.
///   ferrolearn: avg2 = 0.4999999999999999 → beta2 ≈ 1.0
///               → `estimator_weight[1] ≈ 2.22e-16` (≈ 0).
///
/// Expected value `0.10136627702704105` is the sklearn `:1206`/`:1210` closed
/// form; ferrolearn currently returns ≈0. This MUST currently FAIL.
///
/// Tracking: #703
#[test]
fn divergence_reweight_exponent_missing_learning_rate() {
    let x = Array2::from_shape_vec((5, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0]).unwrap();
    let y = Array1::from(vec![6.0, 0.0, 4.0, 8.0, 7.0]);

    let model = AdaBoostRegressor::<f64>::new()
        .with_n_estimators(2)
        .with_max_depth(Some(2))
        .with_learning_rate(0.5)
        .with_loss(AdaBoostLoss::Linear);
    let fitted = model.fit(&x, &y).unwrap();
    let ew = fitted.estimator_weights();

    assert_eq!(
        ew.len(),
        2,
        "construction expects two surviving estimators; got {ew:?}"
    );

    // GREEN anchor (computed pre-reweight, identical under both exponents):
    // estimator_weight[0] = lr * ln(1/beta1), beta1 = 0.4/0.6 = 2/3.
    assert!(
        (ew[0] - 0.202_732_554_054_082_1).abs() < 1e-12,
        "estimator_weight[0] = {} != sklearn closed form 0.2027325540540821 \
         (lr*ln(1/(0.4/0.6)), _weight_boosting.py:1206)",
        ew[0]
    );

    // RED: estimator_weight[1] under the sklearn exponent (:1210).
    // sklearn beta2 = 0.816496580927726 → 0.5*ln(1/beta2) = 0.10136627702704105.
    // ferrolearn (missing *learning_rate) yields ≈ 2.22e-16.
    let sklearn_estimator_weight_1 = 0.101_366_277_027_041_05;
    assert!(
        (ew[1] - sklearn_estimator_weight_1).abs() < 1e-12,
        "estimator_weight[1] = {} != sklearn closed form {sklearn_estimator_weight_1} \
         (reweight exponent must be (1-loss_i)*learning_rate, \
         _weight_boosting.py:1209-1211); ferrolearn omits the *learning_rate factor",
        ew[1]
    );
}

/// PIN 2 (GREEN) — `n_estimators=1` predict equals the single tree's prediction
/// (`_get_median_predict` degenerate, :1215-1230).
///
/// With a single estimator the weighted-median CDF (`weight_cdf >= 0.5*total`,
/// :1224) trivially selects that estimator's prediction for every sample.
/// ferrolearn `weighted_median` with one `(value, weight)` pair returns `value`.
/// Anchored to the per-tree prediction recomputed from the oracle-verified
/// `DecisionTreeRegressor` (NOT from the AdaBoost path — R-CHAR-3).
#[test]
fn single_estimator_predict_equals_tree_prediction() {
    use ferrolearn_tree::DecisionTreeRegressor;

    let x = Array2::from_shape_vec((5, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0]).unwrap();
    let y = Array1::from(vec![6.0, 0.0, 4.0, 8.0, 7.0]);

    let ada = AdaBoostRegressor::<f64>::new()
        .with_n_estimators(1)
        .with_max_depth(Some(2));
    let fitted = ada.fit(&x, &y).unwrap();
    let preds = fitted.predict(&x).unwrap();

    // Independent reference: a depth-2 regression tree on the full data
    // (iter-1 resample on uniform weights is the identity).
    let tree = DecisionTreeRegressor::<f64>::new()
        .with_max_depth(Some(2))
        .fit(&x, &y)
        .unwrap();
    let tree_preds = tree.predict(&x).unwrap();

    for i in 0..x.nrows() {
        assert!(
            (preds[i] - tree_preds[i]).abs() < 1e-12,
            "n_estimators=1 predict[{i}] = {} != single-tree prediction {} \
             (_get_median_predict degenerate, _weight_boosting.py:1215)",
            preds[i],
            tree_preds[i]
        );
    }
}

/// PIN 3 (GREEN) — weighted-median predict returns one of the constituent tree
/// leaf values, within `[min, max]` of them (`_get_median_predict` :1215-1230
/// returns `predictions[..., median_estimators]` — always an actual member, not
/// a blend).
///
/// Structural invariant of the median rule: the output is selected from the set
/// of per-tree leaf values, so every prediction must be one of those leaf values
/// (membership) and within their range. A mean/blend bug (returning an average
/// not present in the leaf set) breaks the membership check. The leaf-value set
/// is read from the public `Node::Leaf { value }` of `estimators()` — not from
/// the predict path (R-CHAR-3).
#[test]
fn predict_is_a_member_of_tree_leaf_values() {
    let x = Array2::from_shape_vec((6, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let y = Array1::from(vec![1.0, 3.0, 2.0, 8.0, 5.0, 9.0]);

    let model = AdaBoostRegressor::<f64>::new()
        .with_n_estimators(10)
        .with_max_depth(Some(2));
    let fitted = model.fit(&x, &y).unwrap();
    let preds = fitted.predict(&x).unwrap();

    let estimators = fitted.estimators();
    assert!(!estimators.is_empty());

    // The set of all leaf values present across every tree in the ensemble.
    let mut leaf_values: Vec<f64> = Vec::new();
    for tree_nodes in estimators {
        for node in tree_nodes {
            if let Node::Leaf { value, .. } = node {
                leaf_values.push(*value);
            }
        }
    }
    let lo = leaf_values.iter().cloned().fold(f64::INFINITY, f64::min);
    let hi = leaf_values
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);

    for i in 0..x.nrows() {
        assert!(
            preds[i] >= lo - 1e-12 && preds[i] <= hi + 1e-12,
            "predict[{i}] = {} outside ensemble leaf-value range [{lo}, {hi}] \
             (_get_median_predict selects a member, _weight_boosting.py:1230)",
            preds[i]
        );
        assert!(
            leaf_values.iter().any(|&v| (v - preds[i]).abs() < 1e-12),
            "predict[{i}] = {} is not any tree leaf value \
             (weighted median returns a member, not a blend, _weight_boosting.py:1230)",
            preds[i]
        );
    }
}

/// PIN 4 (GREEN) — deterministic reproducibility (REQ-7 / AC-6).
///
/// ferrolearn's fit uses NO RNG (deterministic systematic resample), so two fits
/// with identical params must produce identical `estimator_weights` and
/// `predict`. (End-to-end equality to sklearn at a seed is NOT asserted — the
/// numpy weighted-bootstrap boundary, `:1162-1167`.)
#[test]
fn fit_is_deterministic_and_reproducible() {
    let x = Array2::from_shape_vec((6, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let y = Array1::from(vec![1.0, 3.0, 2.0, 8.0, 5.0, 9.0]);

    let model = AdaBoostRegressor::<f64>::new()
        .with_n_estimators(20)
        .with_max_depth(Some(2))
        .with_random_state(7);

    let f1 = model.fit(&x, &y).unwrap();
    let f2 = model.fit(&x, &y).unwrap();

    assert_eq!(
        f1.estimator_weights(),
        f2.estimator_weights(),
        "estimator_weights must be identical across fits (no RNG)"
    );
    assert_eq!(
        f1.predict(&x).unwrap(),
        f2.predict(&x).unwrap(),
        "predict must be identical across fits (no RNG)"
    );
}

/// PIN 5 (GREEN) — constructor defaults match `AdaBoostRegressor().get_params()`
/// (sklearn 1.5.2) and the base `DecisionTreeRegressor(max_depth=3)`
/// (`_validate_estimator`, :1115).
///
/// Live oracle: `AdaBoostRegressor().get_params()` →
///   `{'learning_rate': 1.0, 'loss': 'linear', 'n_estimators': 50,
///     'random_state': None}` (`__init__`, :1094).
#[test]
fn defaults_match_sklearn_get_params() {
    let m = AdaBoostRegressor::<f64>::default();
    assert_eq!(
        m.n_estimators, 50,
        "sklearn default n_estimators=50 (:1094)"
    );
    assert!(
        (m.learning_rate - 1.0).abs() < 1e-15,
        "sklearn default learning_rate=1.0 (:1094)"
    );
    assert_eq!(
        m.loss,
        AdaBoostLoss::Linear,
        "sklearn default loss='linear' (:1094)"
    );
    assert_eq!(
        m.max_depth,
        Some(3),
        "sklearn base DecisionTreeRegressor(max_depth=3) (_validate_estimator, :1115)"
    );
    assert!(
        m.random_state.is_none(),
        "sklearn default random_state=None (:1094)"
    );
}
