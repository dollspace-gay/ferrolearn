//! Divergence pins for `ferrolearn-tree/src/voting.rs`
//! (`VotingClassifier` / `VotingRegressor`) against the live scikit-learn 1.5.2
//! oracle.
//!
//! Reference: scikit-learn 1.5.2 (commit 156ef14),
//! `sklearn/ensemble/_voting.py`.
//!
//! Scope note: ferrolearn's `Voting*` are homogeneous decision-tree ensembles
//! keyed by `max_depths`, not sklearn's heterogeneous `estimators=[(name,est)]`
//! meta-estimators (the architectural headline, `.design/tree/voting.md` #693).
//! These pins exercise the aggregation MATH on a fixed, deterministic
//! homogeneous tree set (`weights=None`) — the only contract ferrolearn claims.
//!
//! `tests/*.rs` is anti-pattern-gate-exempt: `.unwrap()`/`assert!` are used
//! deliberately (no `panic!`/`unreachable!`).

use ferrolearn_core::traits::{Fit, Predict};
use ferrolearn_tree::{DecisionTreeClassifier, VotingClassifier, VotingRegressor};
use ndarray::{Array1, Array2, array};

/// PIN 1 (RED — the headline, #694) — hard-vote TIE-BREAK direction.
///
/// Divergence: `FittedVotingClassifier::predict` (`predict` in `voting.rs`)
/// aggregates per-tree votes with
///   `votes.iter().enumerate().max_by_key(|&(_, &count)| count)`
/// and Rust's `Iterator::max_by_key` returns the LAST maximal element on a tie
/// — so on an exact vote tie ferrolearn returns the HIGHEST-index class.
///
/// sklearn `VotingClassifier(voting='hard').predict`
/// (`sklearn/ensemble/_voting.py:443`-`:448`):
///   `np.argmax(np.bincount(x, weights=self._weights_not_none))`
/// `np.argmax` returns the LOWEST index on a tie (live-confirmed:
/// `np.argmax(np.bincount([1,0])) == 0`).
///
/// Construction (deterministic, no RNG — voting trees see the full dataset):
/// X = [[0],[1],[2],[3]], y = [0,1,0,1], two trees at `max_depth` 1 and 2.
///   depth-1 tree predicts [0,1,1,1]; depth-2 tree predicts [0,1,0,0]
///   (live-confirmed for both sklearn and ferrolearn decision trees).
/// On sample index 2 and 3 the two trees split the vote 1-1 between class 0
/// and class 1 — an EXACT tie. sklearn returns class 0 (lowest index);
/// ferrolearn returns class 1 (highest index, `max_by_key` last-max).
///
/// Live sklearn oracle on this exact (X, y, estimators):
///   `VotingClassifier(estimators=[('a',DT(max_depth=1)),('b',DT(max_depth=2))],
///    voting='hard').fit(X,y).predict(X) == [0, 1, 0, 0]`.
///
/// Expected value is the sklearn contract (lowest-index argmax), NOT copied
/// from ferrolearn (R-CHAR-3). This MUST currently FAIL.
///
/// Tracking: #694
#[test]
fn divergence_hard_vote_tie_breaks_to_lowest_index() {
    let x = Array2::from_shape_vec((4, 1), vec![0.0, 1.0, 2.0, 3.0]).unwrap();
    let y = array![0usize, 1, 0, 1];

    // Sanity-anchor the construction to the per-tree predictions that create
    // the tie (these mirror the live sklearn DecisionTreeClassifier outputs).
    let t1 = DecisionTreeClassifier::<f64>::new()
        .with_max_depth(Some(1))
        .fit(&x, &y)
        .unwrap();
    let t2 = DecisionTreeClassifier::<f64>::new()
        .with_max_depth(Some(2))
        .fit(&x, &y)
        .unwrap();
    assert_eq!(
        t1.predict(&x).unwrap(),
        array![0usize, 1, 1, 1],
        "depth-1 tree must predict [0,1,1,1] for the tie construction"
    );
    assert_eq!(
        t2.predict(&x).unwrap(),
        array![0usize, 1, 0, 0],
        "depth-2 tree must predict [0,1,0,0] for the tie construction"
    );

    // Two-tree VotingClassifier => rows 2 and 3 are exact 1-1 ties.
    let model = VotingClassifier::<f64>::new().with_max_depths(vec![Some(1), Some(2)]);
    let fitted = model.fit(&x, &y).unwrap();
    let preds = fitted.predict(&x).unwrap();

    // sklearn VotingClassifier(voting='hard').predict(X) -> [0, 1, 0, 0]
    // (np.argmax(np.bincount) breaks ties to the LOWEST index, _voting.py:445).
    let sklearn_expected = array![0usize, 1, 0, 0];
    assert_eq!(
        preds, sklearn_expected,
        "hard-vote tie must break to LOWEST-index class (sklearn _voting.py:445 \
         np.argmax(np.bincount)); ferrolearn max_by_key returns the highest \
         index on ties (got {preds:?})"
    );
}

/// PIN 2 (GREEN) — soft `predict_proba` rows sum to 1.0, all entries in [0,1].
///
/// Deterministic invariant from the `voting='soft'`, `weights=None` path of
/// sklearn `_voting.py:480` (`np.average(_collect_probas(X), axis=0)` — the mean
/// of per-estimator `predict_proba`, each of which sums to 1 over classes).
/// ferrolearn `predict_proba` averages per-tree proba (`sum / n_trees`); the row
/// sum is 1 by construction. Holds for any tree set.
#[test]
fn predict_proba_rows_sum_to_one() {
    let x = Array2::from_shape_vec(
        (8, 2),
        vec![
            1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 10.0, 6.0, 11.0, 7.0, 12.0, 8.0, 13.0, 9.0,
        ],
    )
    .unwrap();
    let y = array![0usize, 0, 0, 0, 1, 1, 1, 1];

    let model =
        VotingClassifier::<f64>::new().with_max_depths(vec![Some(1), Some(2), Some(3), None]);
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
        assert!((s - 1.0).abs() < 1e-12, "row {i} sums to {s}, expected 1.0");
    }
}

/// PIN 3 (GREEN) — regressor `predict` == arithmetic MEAN of per-tree
/// predictions (the `weights=None` case of sklearn `_voting.py:716`,
/// `np.average(self._predict(X), axis=1)`).
///
/// Structural pin: recompute the mean of each constituent tree's `predict` and
/// require the ensemble `predict` to equal it. A `sum`-without-`/n` bug or any
/// non-mean aggregation breaks this. Expected value is the documented
/// averaging contract, recomputed from per-tree outputs (NOT copied from the
/// ferrolearn ensemble path — R-CHAR-3).
#[test]
fn regressor_predict_equals_mean_of_per_tree() {
    use ferrolearn_tree::DecisionTreeRegressor;

    let x = Array2::from_shape_vec(
        (6, 2),
        vec![1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 5.0, 6.0, 6.0, 7.0, 7.0, 8.0],
    )
    .unwrap();
    let y = array![1.0, 2.0, 3.0, 5.0, 6.0, 7.0];

    let depths = vec![Some(1usize), Some(2), None];
    let model = VotingRegressor::<f64>::new().with_max_depths(depths.clone());
    let fitted = model.fit(&x, &y).unwrap();
    let preds = fitted.predict(&x).unwrap();

    // Independently recompute the per-tree mean (sklearn _voting.py:716 contract).
    let mut per_tree: Vec<Array1<f64>> = Vec::new();
    for &d in &depths {
        let t = DecisionTreeRegressor::<f64>::new()
            .with_max_depth(d)
            .fit(&x, &y)
            .unwrap();
        per_tree.push(t.predict(&x).unwrap());
    }
    let n = per_tree.len() as f64;
    for i in 0..x.nrows() {
        let mean: f64 = per_tree.iter().map(|p| p[i]).sum::<f64>() / n;
        assert!(
            (preds[i] - mean).abs() < 1e-12,
            "regressor predict[{i}] = {} != per-tree mean {mean} (_voting.py:716)",
            preds[i]
        );
    }
}

/// PIN 4 (GREEN) — clear (non-tie) majority hard vote returns the strict
/// majority class. No tie => unambiguous; `np.argmax(np.bincount)` and
/// ferrolearn `max_by_key` agree (the tie divergence is PIN 1).
///
/// Well-separated 2-class data; three trees all predict the obvious label.
/// Live sklearn oracle on this exact (X, y, 3 estimators):
///   `VotingClassifier(..., voting='hard').predict(X) == [0,0,0,0,1,1,1,1]`.
/// Expected value is the sklearn contract (NOT copied from ferrolearn).
#[test]
fn hard_vote_clear_majority_returns_majority_class() {
    let x =
        Array2::from_shape_vec((8, 1), vec![0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0]).unwrap();
    let y = array![0usize, 0, 0, 0, 1, 1, 1, 1];

    let model = VotingClassifier::<f64>::new().with_max_depths(vec![Some(2), Some(3), None]);
    let fitted = model.fit(&x, &y).unwrap();
    let preds = fitted.predict(&x).unwrap();

    let sklearn_expected = array![0usize, 0, 0, 0, 1, 1, 1, 1];
    assert_eq!(
        preds, sklearn_expected,
        "clear-majority hard vote must return the majority class \
         (sklearn _voting.py:443; live-confirmed [0,0,0,0,1,1,1,1])"
    );
}
