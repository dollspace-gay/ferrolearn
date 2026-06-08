//! Divergence pin for `QuantileTransformer` single-sample (`n_quantiles_ == 1`)
//! Normal forward transform vs scikit-learn 1.5.2.
//!
//! Context: the #2218 fix made `fit` accept `n_samples == 1` (clamping
//! `n_quantiles_ = max(1, min(n_quantiles, n_samples)) = 1`, sklearn
//! `_data.py:2790`, with `references_ = [0.0]`). The fit, the attributes, and
//! the *uniform* transform all match sklearn. This pin isolates the ONE
//! single-sample case that still diverges: the **Normal** forward transform of
//! HELD-OUT data above the single landmark.
//!
//! sklearn `_transform_col` (no-inverse) builds `lower_bound_x = quantiles[0]`,
//! `upper_bound_x = quantiles[-1]` (`_data.py:2809-2810`) — for a single
//! landmark both equal the lone quantile value. For `output_distribution ==
//! "normal"` it then sets (`_data.py:2826-2828`):
//!   `lower_bounds_idx = X_col - BOUNDS_THRESHOLD < lower_bound_x`
//!   `upper_bounds_idx = X_col + BOUNDS_THRESHOLD > upper_bound_x`
//! and overrides `X_col[upper_bounds_idx] = upper_bound_y (= 1)` /
//! `X_col[lower_bounds_idx] = lower_bound_y (= 0)` (`_data.py:2850-2851`),
//! finally applying `stats.norm.ppf` + clip (`_data.py:2856-2862`). So a value
//! ABOVE the single landmark hits the upper bound -> `ppf(1)` -> `+clip`.
//!
//! ferrolearn's forward `transform` never applies those exact-bound overrides;
//! it relies on `interpolate_cdf`, which for a single-landmark column always
//! returns `references_[0] = 0.0`, so `probit(0.0)` -> `-clip` for EVERY
//! held-out value (regardless of side).
//!
//! R-CHAR-3: every expected value below is a LIVE sklearn 1.5.2 call (see the
//! `// oracle:` comments), never copied from the ferrolearn side.
//!
//! Tracking: #2219

use ferrolearn_core::traits::{Fit, Transform};
use ferrolearn_preprocess::quantile_transformer::{OutputDistribution, QuantileTransformer};
use ndarray::array;

/// Divergence: `FittedQuantileTransformer::transform`
/// (`ferrolearn-preprocess/src/quantile_transformer.rs:569-608`, via
/// `interpolate_cdf` `:442-451` which clamps a single-landmark column to
/// `references_[0] = 0.0`) diverges from sklearn `_transform_col`
/// (`sklearn/preprocessing/_data.py:2826-2851`,`:2856-2862`).
///
/// Input: `fit([[7.0]])`, `n_quantiles=5`, `output=Normal`
/// (=> `n_quantiles_ == 1`, `references_ == [0.0]`, `quantiles_ == [[7.0]]`);
/// then `transform([[50.0]])` — a held-out value ABOVE the single landmark.
///
/// sklearn returns `+5.19933758270342` (upper-bound override -> `ppf(1)` ->
/// `+clip`); ferrolearn returns `-5.199337582605575` (single-landmark interp ->
/// `0.0` -> `probit(0.0)` -> `-clip`). The fitted-row value AND the uniform
/// case both match sklearn — this pin is only the held-out Normal mismatch.
///
/// Tracking: #2219
#[test]
#[ignore = "divergence: single-sample Normal forward transform ignores sklearn exact upper/lower bound override, held-out value above landmark -> -clip instead of +clip; tracking #2219"]
fn divergence_single_sample_normal_heldout_above_landmark() {
    // oracle: python3 -c "import numpy as np, warnings; warnings.simplefilter('ignore'); \
    //   from sklearn.preprocessing import QuantileTransformer; \
    //   qt=QuantileTransformer(n_quantiles=5, output_distribution='normal', subsample=None).fit([[7.]]); \
    //   print(qt.transform([[50.]])[0,0])"
    //   -> 5.19933758270342
    const SK_HELDOUT_ABOVE: f64 = 5.19933758270342;

    let x = array![[7.0_f64]];
    let qt = QuantileTransformer::<f64>::new(5, OutputDistribution::Normal, 0);
    let fitted = qt.fit(&x, &()).unwrap();

    // Sanity (these MATCH sklearn; not the divergence): fit clamps to 1 landmark.
    // oracle: qt.n_quantiles_ == 1, qt.references_ == [0.0]
    assert_eq!(fitted.n_quantiles(), 1, "n_quantiles_ should clamp to 1");
    assert_eq!(fitted.references(), &[0.0_f64], "references_ should be [0.0]");

    let out = fitted.transform(&array![[50.0_f64]]).unwrap();
    let actual = out[[0, 0]];

    // sklearn maps the above-landmark held-out value to +clip; ferrolearn -> -clip.
    assert!(
        (actual - SK_HELDOUT_ABOVE).abs() < 1e-6,
        "single-sample Normal held-out above landmark: ferrolearn {actual} != sklearn {SK_HELDOUT_ABOVE}"
    );
}
