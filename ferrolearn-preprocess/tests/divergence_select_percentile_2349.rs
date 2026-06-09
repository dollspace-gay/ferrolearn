//! Divergence pins for `SelectPercentile` vs scikit-learn 1.5.2 (#2349).
//!
//! Expected values come from the LIVE sklearn 1.5.2 oracle (cited inline).

use ferrolearn_core::traits::Fit;
use ferrolearn_preprocess::feature_selection::ScoreFunc;
use ferrolearn_preprocess::select_percentile::SelectPercentile;
use ndarray::{Array1, array};

/// Divergence: sklearn's `SelectPercentile._get_support_mask`
/// (`sklearn/feature_selection/_univariate_selection.py:678-679`) computes the
/// threshold with `np.percentile(scores, 100 - percentile)` AFTER `_clean_nans`
/// maps constant-feature NaN scores to `np.finfo.min`.
///
/// When the cleaned score array contains both `+inf` (a perfectly separating
/// feature) and `finfo.min` (a constant feature), `np.percentile`'s linear
/// interpolation evaluates `inf - finfo.min`, which poisons the threshold to
/// `NaN`. `scores > NaN` is False for every feature, so sklearn selects NOTHING.
///
/// ferrolearn's `numpy_percentile` (`select_percentile.rs:101`) takes a clean
/// branch (no `inf`-arithmetic) and returns a finite threshold (`0.2`), so it
/// selects the `+inf` feature.
///
/// Live oracle (sklearn 1.5.2):
/// ```text
/// X = [[0,1,7],[0,2,7],[5,1,7],[5,3,7]], y=[0,0,1,1]
/// f_classif scores -> [inf, 0.2, nan]
/// SelectPercentile(f_classif, percentile=50).fit(X,y).get_support(indices=True) -> []
/// ```
/// ferrolearn returns `[0]`.
/// Tracking: #2352
#[test]
fn divergence_select_percentile_inf_constant_nan_threshold() {
    let sel = SelectPercentile::<f64>::new(50, ScoreFunc::FClassif);
    let x = array![
        [0.0, 1.0, 7.0],
        [0.0, 2.0, 7.0],
        [5.0, 1.0, 7.0],
        [5.0, 3.0, 7.0]
    ];
    let y: Array1<usize> = array![0, 0, 1, 1];
    let fitted = sel.fit(&x, &y).unwrap();
    // sklearn selects nothing because np.percentile is NaN-poisoned by inf.
    assert_eq!(
        fitted.selected_indices(),
        &[] as &[usize],
        "sklearn np.percentile(inf,finfo.min) is NaN -> empty mask; ferrolearn selected {:?}",
        fitted.selected_indices()
    );
}
