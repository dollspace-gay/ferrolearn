//! Divergence pins for `VarianceThreshold` vs scikit-learn 1.5.2.
//!
//! Audited under #2349. Each test below pins a VALUE/mask/error divergence
//! between `ferrolearn_preprocess::feature_selection::VarianceThreshold` and
//! `sklearn.feature_selection.VarianceThreshold`. Expected values come from the
//! LIVE sklearn 1.5.2 oracle (cited inline), never copied from ferrolearn.

use ferrolearn_core::traits::{Fit, Transform};
use ferrolearn_preprocess::feature_selection::VarianceThreshold;
use ndarray::array;

/// Divergence: ferrolearn's `VarianceThreshold::fit` propagates NaN through its
/// Welford loop so a column containing a NaN yields `var = NaN`, and
/// `NaN > threshold` is false → the column is DROPPED.
///
/// sklearn computes the variance with `np.nanvar`
/// (`sklearn/feature_selection/_variance_threshold.py:112`), which IGNORES NaN.
///
/// Live oracle (sklearn 1.5.2):
/// ```text
/// X = [[1,7],[2,7],[nan,7]]
/// VarianceThreshold(0.0).fit(X).variances_      -> [0.25, 0.0]
/// VarianceThreshold(0.0).fit(X).get_support(indices=True) -> [0]
/// ```
/// Column 0 (`[1,2,NaN]`) has nanvar 0.25 and is KEPT.
/// ferrolearn computes var = NaN for column 0 and DROPS it (support == []).
/// Tracking: #2350
#[test]
#[ignore = "divergence: VarianceThreshold uses raw Welford not np.nanvar, drops NaN columns; tracking #2350"]
fn divergence_variance_threshold_nanvar_keeps_column() {
    let sel = VarianceThreshold::<f64>::new(0.0);
    let x = array![[1.0, 7.0], [2.0, 7.0], [f64::NAN, 7.0]];
    let fitted = sel.fit(&x, &()).unwrap();
    // sklearn keeps column 0 (nanvar = 0.25) and drops constant column 1.
    assert_eq!(
        fitted.selected_indices(),
        &[0usize],
        "sklearn np.nanvar keeps col 0 (var=0.25); ferrolearn NaN-propagates and drops it"
    );
    // And the stored variance for column 0 should be 0.25, not NaN.
    assert!(
        (fitted.variances()[0] - 0.25).abs() < 1e-12,
        "sklearn variances_[0] == 0.25 (np.nanvar), got {}",
        fitted.variances()[0]
    );
}

/// Divergence: when NO feature meets the variance threshold, sklearn RAISES
/// `ValueError("No feature in X meets the variance threshold ...")`
/// (`sklearn/feature_selection/_variance_threshold.py:121-125`).
///
/// ferrolearn's `fit` returns `Ok` with an empty `selected_indices` (no error),
/// then `transform` yields a zero-column matrix.
///
/// Live oracle (sklearn 1.5.2):
/// ```text
/// X = [[5,3],[5,3],[5,3]]  (both columns constant)
/// VarianceThreshold(0.0).fit(X)  -> raises ValueError
/// ```
/// Tracking: #2351
#[test]
#[ignore = "divergence: VarianceThreshold does not raise when all features removed; tracking #2351"]
fn divergence_variance_threshold_all_removed_raises() {
    let sel = VarianceThreshold::<f64>::new(0.0);
    let x = array![[5.0, 3.0], [5.0, 3.0], [5.0, 3.0]];
    // sklearn raises ValueError here; ferrolearn must surface an error too.
    assert!(
        sel.fit(&x, &()).is_err(),
        "sklearn raises ValueError when no feature meets the threshold; ferrolearn returned Ok"
    );
}

/// Divergence: a single-sample input has zero variance everywhere, so sklearn
/// RAISES `ValueError("No feature ... (X contains only one sample)")`
/// (`sklearn/feature_selection/_variance_threshold.py:121-125`).
///
/// ferrolearn's `fit` succeeds (n_samples == 1 passes the zero-row guard) and
/// returns an empty selection.
///
/// Live oracle (sklearn 1.5.2):
/// ```text
/// X = [[1,2,3]]
/// VarianceThreshold(0.0).fit(X)  -> raises ValueError
/// ```
/// Tracking: #2351
#[test]
#[ignore = "divergence: VarianceThreshold single-sample does not raise; tracking #2351"]
fn divergence_variance_threshold_single_sample_raises() {
    let sel = VarianceThreshold::<f64>::new(0.0);
    let x = array![[1.0, 2.0, 3.0]];
    assert!(
        sel.fit(&x, &()).is_err(),
        "sklearn raises ValueError for a single-sample X; ferrolearn returned Ok"
    );
}
