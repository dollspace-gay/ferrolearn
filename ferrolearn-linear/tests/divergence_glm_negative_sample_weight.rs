//! Divergence pin: the GLM family (`PoissonRegressor` / `GammaRegressor` /
//! `TweedieRegressor`, all routing through the shared `fn fit_glm_irls`) REJECTS
//! a *finite negative* `sample_weight` with
//! `FerroError::InvalidParameter { name: "sample_weight",
//!  reason: "sample weights must be non-negative" }`, but scikit-learn 1.5.2's
//! GLM regressors do NOT reject finite negative weights — they FIT and return a
//! model.
//!
//! This was surfaced while auditing the #2261 batch-3 non-finite-input work,
//! which reordered the finiteness guard above the pre-existing non-negative
//! guard in `fn fit_glm_irls` (`glm.rs:1574-1602`). The reorder is correct for
//! NaN/Inf weights (sklearn raises a finiteness `ValueError`), but the
//! non-negative guard ITSELF diverges from sklearn: sklearn's GLM calls
//! `_check_sample_weight(sample_weight, X, dtype=loss_dtype)` WITHOUT
//! `only_non_negative=True` (`sklearn/linear_model/_glm/glm.py:208-211`), so a
//! finite negative weight passes validation and reaches the optimizer.
//!
//! sklearn cite — `sklearn/linear_model/_glm/glm.py:211`:
//!   `sample_weight = _check_sample_weight(sample_weight, X, dtype=loss_dtype)`
//! (no `only_non_negative=...` argument => negative weights are accepted; the
//! `_check_sample_weight` default does not enforce non-negativity).
//!
//! ferrolearn cite — `ferrolearn-linear/src/glm.rs:1595-1602`:
//! ```ignore
//! for &si in sample_weight.iter() {
//!     if si < F::zero() {
//!         return Err(FerroError::InvalidParameter {
//!             name: "sample_weight".into(),
//!             reason: "sample weights must be non-negative".into(),
//!         });
//!     }
//! }
//! ```
//!
//! Live sklearn 1.5.2 oracle (R-CHAR-3 — expected behavior is sklearn's, never
//! copied from ferrolearn). Negative weight on the last sample, all others 1.0:
//! ```text
//! cd /tmp && python3 -c "
//! import numpy as np, warnings; warnings.simplefilter('ignore')
//! from sklearn.linear_model import (PoissonRegressor, GammaRegressor,
//!     TweedieRegressor)
//! X=np.array([[1.,2.],[3.,4.],[5.,6.],[7.,8.],[2.,1.],[4.,3.]])
//! y=np.array([1.,2.,3.,4.,1.5,2.5])
//! w=np.array([1.,1,1,1,1,-1.0])
//! print('Poisson', PoissonRegressor().fit(X,y,sample_weight=w).coef_)
//! print('Gamma',   GammaRegressor().fit(X,y,sample_weight=w).coef_)
//! print('Tweedie', TweedieRegressor(power=1.5).fit(X,y,sample_weight=w).coef_)
//! "
//! # Poisson [-0.04268902  0.24080026]   (NO RAISE)
//! # Gamma   [ 0.038       0.14459 ]     (NO RAISE)
//! # Tweedie [ 0.00672     0.18325 ]     (NO RAISE)
//! ```
//!
//! Expected (sklearn): `fit_with_sample_weight` returns `Ok(_)` (a fitted model).
//! Actual (ferrolearn): `Err(InvalidParameter { name: "sample_weight",
//! reason: "sample weights must be non-negative" })`.
//!
//! NOTE on scope: BayesianRidge and RidgeClassifier DO raise in sklearn for a
//! finite negative weight (the `sqrt(sample_weight)` rescale yields NaN, caught
//! downstream by `check_array` → `ValueError: array must not contain infs or
//! NaNs`), so a ferrolearn rejection there is exception parity (R-DEV-2), not a
//! divergence — those are deliberately NOT pinned. Only the GLM family, which
//! sklearn fits without raising, diverges.

use ferrolearn_core::Fit;
use ferrolearn_linear::{GammaRegressor, PoissonRegressor, TweedieRegressor};
use ndarray::{array, Array1, Array2};

fn finite_xy() -> (Array2<f64>, Array1<f64>) {
    let x: Array2<f64> = array![
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
        [7.0, 8.0],
        [2.0, 1.0],
        [4.0, 3.0]
    ];
    let y: Array1<f64> = array![1.0, 2.0, 3.0, 4.0, 1.5, 2.5];
    (x, y)
}

/// A weight vector that is all-ones except a single FINITE negative entry.
fn finite_negative_weight() -> Array1<f64> {
    let mut w = Array1::<f64>::ones(6);
    w[5] = -1.0;
    w
}

#[test]
#[ignore = "divergence: GLM family rejects finite negative sample_weight; sklearn fits (no only_non_negative); tracking #2262"]
fn poisson_accepts_finite_negative_sample_weight_like_sklearn() {
    // Oracle: PoissonRegressor().fit(X, y, sample_weight=[..,-1.0]) returns a
    // fitted model (coef_ ~= [-0.04268902, 0.24080026]); sklearn does NOT reject
    // finite negative weights (`glm.py:211`, no `only_non_negative`).
    let (x, y) = finite_xy();
    let w = finite_negative_weight();
    let res = PoissonRegressor::<f64>::new().fit_with_sample_weight(&x, &y, &w);
    assert!(
        res.is_ok(),
        "PoissonRegressor: sklearn FITS on a finite negative sample_weight \
         (does not reject), got {res:?}"
    );
}

#[test]
#[ignore = "divergence: GLM family rejects finite negative sample_weight; sklearn fits (no only_non_negative); tracking #2262"]
fn gamma_accepts_finite_negative_sample_weight_like_sklearn() {
    // Oracle: GammaRegressor().fit(X, y, sample_weight=[..,-1.0]) returns a
    // fitted model (coef_ ~= [0.038, 0.14459]); sklearn does NOT reject.
    let (x, y) = finite_xy();
    let w = finite_negative_weight();
    let res = GammaRegressor::<f64>::new().fit_with_sample_weight(&x, &y, &w);
    assert!(
        res.is_ok(),
        "GammaRegressor: sklearn FITS on a finite negative sample_weight \
         (does not reject), got {res:?}"
    );
}

#[test]
#[ignore = "divergence: GLM family rejects finite negative sample_weight; sklearn fits (no only_non_negative); tracking #2262"]
fn tweedie_accepts_finite_negative_sample_weight_like_sklearn() {
    // Oracle: TweedieRegressor(power=1.5).fit(X, y, sample_weight=[..,-1.0])
    // returns a fitted model (coef_ ~= [0.00672, 0.18325]); sklearn does NOT
    // reject.
    let (x, y) = finite_xy();
    let w = finite_negative_weight();
    let res = TweedieRegressor::<f64>::new()
        .with_power(1.5)
        .fit_with_sample_weight(&x, &y, &w);
    assert!(
        res.is_ok(),
        "TweedieRegressor: sklearn FITS on a finite negative sample_weight \
         (does not reject), got {res:?}"
    );
}
