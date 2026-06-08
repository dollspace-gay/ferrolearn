//! Non-finite input validation pins for the seven distance-based
//! `ferrolearn-neighbors` estimators (`KNeighbors{Classifier,Regressor}`,
//! `RadiusNeighbors{Classifier,Regressor}`, `NearestCentroid`,
//! `NearestNeighbors`, `LocalOutlierFactor`) against the LIVE scikit-learn
//! 1.5.2 oracle (#2272).
//!
//! sklearn validates X (and float y) finiteness up-front in `_validate_data`
//! (`force_all_finite=True`):
//!   - The supervised/unsupervised neighbor estimators route through
//!     `NeighborsBase._fit` → `self._validate_data(X, y, ...)` (`requires_y`,
//!     `sklearn/neighbors/_base.py:475`) or `X = self._validate_data(X,
//!     accept_sparse="csr", order="C")` (unsupervised, `:517`).
//!   - `NearestCentroid.fit` → `self._validate_data(X, y, ...)`
//!     (`_nearest_centroid.py:134`/`:136`) BEFORE the `n_classes < 2` check.
//!   - `NearestNeighbors.fit` → `self._fit(X)` (`_unsupervised.py:176`);
//!     `LocalOutlierFactor.fit` → `self._fit(X)` (`_lof.py:279`);
//!     `LocalOutlierFactor.fit_predict` → `self.fit(X)._predict()`
//!     (`_lof.py:231-256`), so it inherits the same up-front validation.
//!
//! Any NaN/+/-inf in X (or in the float `y` of the two regressors) raises
//! `ValueError("Input X contains NaN.")` or `"... contains infinity ..."` (or
//! the `y`-named variants) before the neighbor index / centroid math.
//!
//! Every expected behavior below is the LIVE `sklearn` 1.5.2 oracle (computed
//! via `python3 -c "..."` run from `/tmp`, quoted below) — NEVER copied from
//! the ferrolearn side (goal.md R-CHAR-3).
//!
//! Live oracle (run from `/tmp`):
//! ```text
//! import numpy as np
//! from sklearn.neighbors import (KNeighborsClassifier, KNeighborsRegressor,
//!     RadiusNeighborsClassifier, RadiusNeighborsRegressor, NearestCentroid,
//!     NearestNeighbors, LocalOutlierFactor)
//! Xg = [[0,0],[1,0],[0,1],[5,5],[6,5],[5,6]]; yc=[0,0,0,1,1,1]; yr=[0.,0,0,1,1,1]
//! # fit/fit_predict, NaN / +inf / -inf in X -> ValueError (for ALL seven):
//! KNeighborsClassifier(n_neighbors=3).fit(X_nan, yc)   -> ValueError: Input X contains NaN.
//! KNeighborsRegressor(n_neighbors=3).fit(X_inf, yr)    -> ValueError: Input X contains infinity ...
//! RadiusNeighborsClassifier(radius=3).fit(X_neg_inf,yc)-> ValueError: Input X contains infinity ...
//! RadiusNeighborsRegressor(radius=3).fit(X_nan, yr)    -> ValueError: Input X contains NaN.
//! NearestCentroid().fit(X_inf, yc)                     -> ValueError: Input X contains infinity ...
//! NearestNeighbors(n_neighbors=2).fit(X_nan)           -> ValueError: Input X contains NaN.
//! LocalOutlierFactor(n_neighbors=3).fit(X_inf)         -> ValueError: Input X contains infinity ...
//! LocalOutlierFactor(n_neighbors=3).fit_predict(X_nan) -> ValueError: Input X contains NaN.
//! # NaN / inf in (float) y for the two regressors -> ValueError:
//! KNeighborsRegressor(n_neighbors=3).fit(Xg, y_nan)    -> ValueError: Input y contains NaN.
//! KNeighborsRegressor(n_neighbors=3).fit(Xg, y_inf)    -> ValueError: Input y contains infinity ...
//! RadiusNeighborsRegressor(radius=3).fit(Xg, y_nan)    -> ValueError: Input y contains NaN.
//! # all-finite known fits (no false positive, no regression):
//! KNeighborsClassifier(n_neighbors=3).fit(Xg,yc).predict([[0,0],[5,5]]) -> [0, 1]
//! KNeighborsRegressor(n_neighbors=2).fit(Xg,yr).predict([[0,0]])        -> [0.0]
//! NearestCentroid().fit(Xg,yc).predict([[0,0],[5,5]])                   -> [0, 1]
//! NearestNeighbors(n_neighbors=2).fit(Xg).kneighbors([[0,0]])[1]        -> [[0, 1]]
//! LocalOutlierFactor(n_neighbors=3).fit_predict(Xg)                     -> finite labels (no raise)
//! ```

use ferrolearn_core::{Fit, Predict};
use ferrolearn_neighbors::{
    KNeighborsClassifier, KNeighborsRegressor, LocalOutlierFactor, NearestCentroid,
    NearestNeighbors, RadiusNeighborsClassifier, RadiusNeighborsRegressor,
};
use ndarray::{Array1, Array2, array};

const NAN: f64 = f64::NAN;
const POS_INF: f64 = f64::INFINITY;
const NEG_INF: f64 = f64::NEG_INFINITY;

/// All-finite 6-point, 2-feature fixture shared across estimators.
fn finite_x() -> Array2<f64> {
    array![
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [5.0, 5.0],
        [6.0, 5.0],
        [5.0, 6.0]
    ]
}

fn labels() -> Array1<usize> {
    array![0usize, 0, 0, 1, 1, 1]
}

fn float_y() -> Array1<f64> {
    array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
}

/// Return a copy of `finite_x` with `x[0][0]` replaced by `v`.
fn x_with(v: f64) -> Array2<f64> {
    let mut x = finite_x();
    x[[0, 0]] = v;
    x
}

// ---------------------------------------------------------------------------
// KNeighborsClassifier — X only (integer labels).
// ---------------------------------------------------------------------------

#[test]
fn knn_classifier_fit_rejects_nonfinite_x() {
    // sklearn KNeighborsClassifier().fit(X_nonfinite, y) -> ValueError.
    let y = labels();
    for v in [NAN, POS_INF, NEG_INF] {
        assert!(
            KNeighborsClassifier::<f64>::new()
                .with_n_neighbors(3)
                .fit(&x_with(v), &y)
                .is_err(),
            "KNeighborsClassifier.fit must reject {v} in X (sklearn raises ValueError)"
        );
    }
}

#[test]
fn knn_classifier_fit_accepts_finite() {
    // No false positive + sklearn predict([[0,0],[5,5]]) == [0, 1].
    let fitted = KNeighborsClassifier::<f64>::new()
        .with_n_neighbors(3)
        .fit(&finite_x(), &labels())
        .expect("finite fit must succeed");
    let preds = fitted.predict(&array![[0.0, 0.0], [5.0, 5.0]]).unwrap();
    assert_eq!(preds[0], 0);
    assert_eq!(preds[1], 1);
}

// ---------------------------------------------------------------------------
// KNeighborsRegressor — X AND float y.
// ---------------------------------------------------------------------------

#[test]
fn knn_regressor_fit_rejects_nonfinite_x() {
    let y = float_y();
    for v in [NAN, POS_INF, NEG_INF] {
        assert!(
            KNeighborsRegressor::<f64>::new()
                .with_n_neighbors(3)
                .fit(&x_with(v), &y)
                .is_err(),
            "KNeighborsRegressor.fit must reject {v} in X (sklearn raises ValueError)"
        );
    }
}

#[test]
fn knn_regressor_fit_rejects_nonfinite_y() {
    // sklearn KNeighborsRegressor().fit(Xg, y_nan/y_inf) -> ValueError: Input y ...
    let x = finite_x();
    for v in [NAN, POS_INF, NEG_INF] {
        let mut y = float_y();
        y[0] = v;
        assert!(
            KNeighborsRegressor::<f64>::new()
                .with_n_neighbors(3)
                .fit(&x, &y)
                .is_err(),
            "KNeighborsRegressor.fit must reject {v} in y (sklearn raises ValueError)"
        );
    }
}

#[test]
fn knn_regressor_fit_accepts_finite() {
    let fitted = KNeighborsRegressor::<f64>::new()
        .with_n_neighbors(2)
        .fit(&finite_x(), &float_y())
        .expect("finite fit must succeed");
    let preds = fitted.predict(&array![[0.0, 0.0]]).unwrap();
    // Two nearest of (0,0) are training rows 0,1 (both y=0.0) -> mean 0.0.
    assert!((preds[0] - 0.0).abs() < 1e-12);
}

// ---------------------------------------------------------------------------
// RadiusNeighborsClassifier — X only (integer labels).
// ---------------------------------------------------------------------------

#[test]
fn radius_classifier_fit_rejects_nonfinite_x() {
    let y = labels();
    for v in [NAN, POS_INF, NEG_INF] {
        assert!(
            RadiusNeighborsClassifier::<f64>::new()
                .with_radius(3.0)
                .fit(&x_with(v), &y)
                .is_err(),
            "RadiusNeighborsClassifier.fit must reject {v} in X (sklearn raises ValueError)"
        );
    }
}

#[test]
fn radius_classifier_fit_accepts_finite() {
    let fitted = RadiusNeighborsClassifier::<f64>::new()
        .with_radius(3.0)
        .fit(&finite_x(), &labels())
        .expect("finite fit must succeed");
    let preds = fitted.predict(&array![[0.0, 0.0], [5.0, 5.0]]).unwrap();
    assert_eq!(preds[0], 0);
    assert_eq!(preds[1], 1);
}

// ---------------------------------------------------------------------------
// RadiusNeighborsRegressor — X AND float y.
// ---------------------------------------------------------------------------

#[test]
fn radius_regressor_fit_rejects_nonfinite_x() {
    let y = float_y();
    for v in [NAN, POS_INF, NEG_INF] {
        assert!(
            RadiusNeighborsRegressor::<f64>::new()
                .with_radius(3.0)
                .fit(&x_with(v), &y)
                .is_err(),
            "RadiusNeighborsRegressor.fit must reject {v} in X (sklearn raises ValueError)"
        );
    }
}

#[test]
fn radius_regressor_fit_rejects_nonfinite_y() {
    let x = finite_x();
    for v in [NAN, POS_INF, NEG_INF] {
        let mut y = float_y();
        y[0] = v;
        assert!(
            RadiusNeighborsRegressor::<f64>::new()
                .with_radius(3.0)
                .fit(&x, &y)
                .is_err(),
            "RadiusNeighborsRegressor.fit must reject {v} in y (sklearn raises ValueError)"
        );
    }
}

#[test]
fn radius_regressor_fit_accepts_finite() {
    let fitted = RadiusNeighborsRegressor::<f64>::new()
        .with_radius(3.0)
        .fit(&finite_x(), &float_y())
        .expect("finite fit must succeed");
    let preds = fitted.predict(&array![[0.0, 0.0]]).unwrap();
    assert!(preds[0].is_finite());
}

// ---------------------------------------------------------------------------
// NearestCentroid — X only (integer labels; no sample_weight).
// ---------------------------------------------------------------------------

#[test]
fn nearest_centroid_fit_rejects_nonfinite_x() {
    let y = labels();
    for v in [NAN, POS_INF, NEG_INF] {
        assert!(
            NearestCentroid::<f64>::new().fit(&x_with(v), &y).is_err(),
            "NearestCentroid.fit must reject {v} in X (sklearn raises ValueError)"
        );
    }
}

#[test]
fn nearest_centroid_fit_accepts_finite() {
    let fitted = NearestCentroid::<f64>::new()
        .fit(&finite_x(), &labels())
        .expect("finite fit must succeed");
    let preds = fitted.predict(&array![[0.0, 0.0], [5.0, 5.0]]).unwrap();
    assert_eq!(preds[0], 0);
    assert_eq!(preds[1], 1);
}

// ---------------------------------------------------------------------------
// NearestNeighbors — unsupervised, X only.
// ---------------------------------------------------------------------------

#[test]
fn nearest_neighbors_fit_rejects_nonfinite_x() {
    for v in [NAN, POS_INF, NEG_INF] {
        assert!(
            NearestNeighbors::<f64>::new()
                .with_n_neighbors(2)
                .fit(&x_with(v), &())
                .is_err(),
            "NearestNeighbors.fit must reject {v} in X (sklearn raises ValueError)"
        );
    }
}

#[test]
fn nearest_neighbors_fit_accepts_finite() {
    let fitted = NearestNeighbors::<f64>::new()
        .with_n_neighbors(2)
        .fit(&finite_x(), &())
        .expect("finite fit must succeed");
    let (_dists, idxs) = fitted.kneighbors(&array![[0.0, 0.0]], None).unwrap();
    // The 2 nearest of (0,0) are training rows 0 (self) and 1.
    assert_eq!(idxs[[0, 0]], 0);
    assert_eq!(idxs[[0, 1]], 1);
}

// ---------------------------------------------------------------------------
// LocalOutlierFactor — unsupervised, X only; guard fit AND fit_predict.
// ---------------------------------------------------------------------------

#[test]
fn lof_fit_rejects_nonfinite_x() {
    for v in [NAN, POS_INF, NEG_INF] {
        assert!(
            LocalOutlierFactor::<f64>::new()
                .with_n_neighbors(3)
                .fit(&x_with(v), &())
                .is_err(),
            "LocalOutlierFactor.fit must reject {v} in X (sklearn raises ValueError)"
        );
    }
}

#[test]
fn lof_fit_predict_rejects_nonfinite_x() {
    // sklearn LocalOutlierFactor().fit_predict(X_nonfinite) -> ValueError
    // (fit_predict routes through fit, _lof.py:231-256). The separate ferrolearn
    // `fit_predict` arm must inherit the guard (the SGD partial_fit lesson).
    for v in [NAN, POS_INF, NEG_INF] {
        assert!(
            LocalOutlierFactor::<f64>::new()
                .with_n_neighbors(3)
                .fit_predict(&x_with(v))
                .is_err(),
            "LocalOutlierFactor.fit_predict must reject {v} in X (sklearn raises ValueError)"
        );
    }
}

#[test]
fn lof_fit_predict_accepts_finite() {
    let labels = LocalOutlierFactor::<f64>::new()
        .with_n_neighbors(3)
        .fit_predict(&finite_x())
        .expect("finite fit_predict must succeed");
    assert_eq!(labels.len(), 6);
    // Labels are inlier(+1)/outlier(-1) — every entry finite/in-range.
    assert!(labels.iter().all(|&l| l == 1 || l == -1));
}
