//! Predict/kneighbors-time non-finite validation pins vs the LIVE scikit-learn
//! 1.5.2 oracle.
//!
//! The #2272 sweep added finiteness guards at FIT entry only. sklearn ALSO
//! validates the query X at predict/kneighbors time: `KNeighborsMixin
//! .kneighbors` calls `X = self._validate_data(X, ..., reset=False)`
//! (`sklearn/neighbors/_base.py:824`), keeping `force_all_finite=True`, so a
//! NaN/inf query raises `ValueError("Input X contains NaN.")` /
//! `"... contains infinity ..."` BEFORE the neighbor search.
//!
//! ferrolearn's fitted `predict` / `kneighbors` have NO finiteness guard. A NaN
//! query flows into the kd_tree/heap search where the NaN comparison yields an
//! out-of-bounds neighbor index (`usize::MAX`), then `y_train[idx]` PANICS:
//!   `ndarray: index 18446744073709551615 is out of bounds for array of
//!    shape [4]` (knn.rs:944). This violates goal.md R-CODE-2 (no panic in
//! library code) AND diverges from sklearn's clean `ValueError`.
//!
//! Live sklearn 1.5.2 oracle (run from `/tmp`):
//! ```text
//! import numpy as np
//! from sklearn.neighbors import KNeighborsClassifier
//! Xf=[[0.,0.],[1.,1.],[5.,5.],[6.,6.]]; y=[0,0,1,1]
//! clf=KNeighborsClassifier(n_neighbors=2).fit(Xf,y)
//! clf.predict(np.array([[np.nan,0.]]))
//!   -> ValueError: Input X contains NaN.
//! clf.predict(np.array([[np.inf,0.]]))
//!   -> ValueError: Input X contains infinity or a value too large ...
//! ```
//!
//! Every expected behavior is the LIVE sklearn oracle (goal.md R-CHAR-3).

use ferrolearn_core::{Fit, Predict};
use ferrolearn_neighbors::KNeighborsClassifier;
use ndarray::{array, Array2};
use std::panic::{catch_unwind, AssertUnwindSafe};

fn fitted() -> ferrolearn_neighbors::FittedKNeighborsClassifier<f64> {
    let x = array![[0.0f64, 0.0], [1.0, 1.0], [5.0, 5.0], [6.0, 6.0]];
    let y = array![0usize, 0, 1, 1];
    KNeighborsClassifier::<f64>::new()
        .with_n_neighbors(2)
        .fit(&x, &y)
        .expect("finite fit succeeds")
}

/// Divergence + R-CODE-2 no-panic violation: `FittedKNeighborsClassifier
/// ::predict` with a NaN query must return a clean `Err` (sklearn raises
/// `ValueError("Input X contains NaN.")`, `sklearn/neighbors/_base.py:824`
/// `_validate_data(..., reset=False)`). ferrolearn currently PANICS at
/// knn.rs:944 (`y_train[usize::MAX]` OOB) because the NaN flows unvalidated
/// into the kd_tree search.
/// Tracking: #2274
#[test]
#[ignore = "release-blocker: predict-time NaN panics (knn.rs:944) instead of Err; sklearn ValueError; tracking #2274"]
fn knn_predict_nan_query_returns_err_not_panic() {
    let f = fitted();
    let q: Array2<f64> = array![[f64::NAN, 0.0]];
    // sklearn raises ValueError; the ferrolearn contract is Result<_, FerroError>
    // (no panic, R-CODE-2). Today this PANICS, so catch_unwind reports the
    // divergence as a failed test even though predict itself does not return.
    let result = catch_unwind(AssertUnwindSafe(|| f.predict(&q)));
    match result {
        Ok(Ok(p)) => panic!(
            "sklearn raises ValueError(Input X contains NaN.) on a NaN query; \
             ferrolearn returned Ok({p:?}) (silent wrong answer)"
        ),
        Ok(Err(_)) => { /* desired post-fix behavior: clean Err */ }
        Err(_) => panic!(
            "ferrolearn PANICKED on a NaN predict query (R-CODE-2 no-panic \
             violation); sklearn raises a clean ValueError"
        ),
    }
}

/// Same divergence for an +inf query: sklearn raises
/// `ValueError("Input X contains infinity ...")`; ferrolearn does not validate
/// the query, so the +inf flows into the search.
/// Tracking: #2274
#[test]
#[ignore = "release-blocker: predict-time inf query unvalidated; sklearn ValueError; tracking #2274"]
fn knn_predict_inf_query_returns_err_not_panic() {
    let f = fitted();
    let q: Array2<f64> = array![[f64::INFINITY, 0.0]];
    let result = catch_unwind(AssertUnwindSafe(|| f.predict(&q)));
    match result {
        Ok(Ok(p)) => panic!(
            "sklearn raises ValueError(Input X contains infinity ...) on an inf \
             query; ferrolearn returned Ok({p:?})"
        ),
        Ok(Err(_)) => { /* desired post-fix behavior */ }
        Err(_) => panic!(
            "ferrolearn PANICKED on an inf predict query (R-CODE-2); sklearn \
             raises a clean ValueError"
        ),
    }
}
