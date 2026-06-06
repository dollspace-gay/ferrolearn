//! Divergence pins for issue #674:
//! `RandomForestClassifier::feature_importances()` aggregate over per-tree
//! importances, matching scikit-learn 1.5.2 `sklearn/ensemble/_forest.py:681`.
//!
//! sklearn (`_forest.py:675-685`) computes the forest importances as the
//! `np.mean` over ONLY the trees whose `tree_.node_count > 1` (i.e. it excludes
//! single-node "stump" trees), then normalizes the mean to sum to 1. When NO
//! tree has a split (`if not all_importances`), it returns `np.zeros(
//! n_features_in_)`.
//!
//! ferrolearn (`random_forest.rs:366-377`) sums the per-tree *already-
//! normalized* importance vectors (a stump yields an all-zero vector, since
//! `compute_feature_importances` at `decision_tree.rs:2371-2374` leaves zeros
//! when the impurity total is 0) and divides by `imp_sum`. The hypothesis under
//! audit is that this is algebraically identical to the sklearn formulation:
//! stumps contribute zero vectors so they do not perturb the mean, and the
//! all-stump case yields `imp_sum == 0`, leaving the result all-zeros.
//!
//! Live sklearn 1.5.2 oracle (re-run 2026-06):
//!   * all-stump: `RandomForestClassifier(n_estimators=5, random_state=0)`
//!     `.fit([[1,2],[3,4],[5,6],[7,8]], [0,0,0,0])` ->
//!     `feature_importances_ == [0., 0.]`, `n_features_in_ == 2`.
//!   * normal (2-class, real split): `feature_importances_.sum() == 1.0`,
//!     all entries `>= 0`.
//!
//! `tests/*.rs` is anti-pattern-gate-exempt; this file additionally uses
//! `-> Result<(), FerroError>` + `?` on the fit, with only `assert!`/`assert_eq!`
//! (no `panic!`/`unwrap`/`expect`).

use ferrolearn_core::error::FerroError;
use ferrolearn_core::introspection::HasFeatureImportances;
use ferrolearn_core::traits::Fit;
use ferrolearn_tree::RandomForestClassifier;
use ndarray::{Array1, Array2};

/// Divergence guard for #674 (all-stump branch, `_forest.py:681-682`):
///
///   `if not all_importances: return np.zeros(self.n_features_in_, ...)`
///
/// With `y = [0,0,0,0]` (all one class) EVERY bootstrap sample is single-class,
/// so every tree is a root-only stump regardless of RNG. sklearn returns the
/// all-zeros vector of length `n_features_in_ == 2`. This case is RNG-
/// independent, so an exact `assert_eq!` on `0.0` is valid.
///
/// Expected (sklearn oracle): `[0.0, 0.0]`.
#[test]
fn rf_feature_importances_all_stumps_are_zero() -> Result<(), FerroError> {
    let x: Array2<f64> =
        Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).map_err(
            |e| FerroError::InvalidParameter {
                name: "x".into(),
                reason: e.to_string(),
            },
        )?;
    let y: Array1<usize> = Array1::from_vec(vec![0, 0, 0, 0]);

    let fitted = RandomForestClassifier::<f64>::new()
        .with_n_estimators(5)
        .with_random_state(0)
        .fit(&x, &y)?;

    let importances = fitted.feature_importances();

    // sklearn `_forest.py:682`: zeros of shape (n_features_in_,) == (2,).
    assert_eq!(
        importances.len(),
        2,
        "feature_importances length must equal n_features_in_ (=2); got {importances:?}"
    );
    // Zeros are exact: every tree is a stump => empty `all_importances` => zeros.
    assert_eq!(
        importances[0], 0.0,
        "all-stump forest: feature_importances_[0] must be exactly 0.0; got {importances:?}"
    );
    assert_eq!(
        importances[1], 0.0,
        "all-stump forest: feature_importances_[1] must be exactly 0.0; got {importances:?}"
    );

    Ok(())
}

/// Divergence guard for #674 (normal branch, `_forest.py:684-685`):
///
///   `all_importances = np.mean(all_importances, axis=0, ...)`
///   `return all_importances / np.sum(all_importances)`
///
/// On a 2-class separable dataset that produces at least one real split, the
/// sklearn oracle yields `feature_importances_.sum() == 1.0` with all entries
/// `>= 0`. Per-feature values are NOT pinned (the StdRng vs numpy-MT19937
/// bootstrap stream is the documented #673 boundary); only the structural
/// invariants are asserted.
#[test]
fn rf_feature_importances_normal_sums_to_one() -> Result<(), FerroError> {
    let x: Array2<f64> = Array2::from_shape_vec(
        (6, 2),
        vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
    )
    .map_err(|e| FerroError::InvalidParameter {
        name: "x".into(),
        reason: e.to_string(),
    })?;
    let y: Array1<usize> = Array1::from_vec(vec![0, 0, 1, 1, 0, 1]);

    let fitted = RandomForestClassifier::<f64>::new()
        .with_n_estimators(10)
        .with_random_state(42)
        .fit(&x, &y)?;

    let importances = fitted.feature_importances();

    assert_eq!(
        importances.len(),
        2,
        "feature_importances length must equal n_features (=2); got {importances:?}"
    );

    // sklearn `_forest.py:685`: normalized to sum to 1.
    let sum: f64 = importances.iter().copied().sum();
    assert!(
        (sum - 1.0).abs() < 1e-9,
        "feature_importances_ must sum to 1.0 (sklearn _forest.py:685); got sum={sum} ({importances:?})"
    );

    // sklearn: impurity importances are non-negative.
    for (i, &v) in importances.iter().enumerate() {
        assert!(
            v >= 0.0,
            "feature_importances_[{i}] must be >= 0; got {v} ({importances:?})"
        );
    }

    Ok(())
}
