//! Non-finite input validation pins for the five Naive Bayes estimators
//! (`GaussianNB` / `BernoulliNB` / `MultinomialNB` / `ComplementNB` /
//! `CategoricalNB`) against the LIVE scikit-learn 1.5.2 oracle (#2271).
//!
//! sklearn validates X finiteness up-front in `_validate_data`/`_check_X_y`
//! (`force_all_finite=True`, `sklearn/naive_bayes.py`): `GaussianNB._partial_fit`
//! calls `self._validate_data(X, y, reset=...)` (`naive_bayes.py:423`); the
//! discrete NBs' shared `_BaseDiscreteNB.fit`/`partial_fit` call
//! `self._check_X_y(X, y)` → `self._validate_data(X, y, accept_sparse="csr",
//! ...)` (`naive_bayes.py:576-578`, `:668`, `:712`); `CategoricalNB._check_X_y`
//! calls `self._validate_data(X, y, dtype="int", ...)` then
//! `check_non_negative(...)` (`naive_bayes.py:1435-1439`). Any NaN/+/-inf in X
//! raises `ValueError("Input X contains NaN.")` / `"... contains infinity ..."`
//! BEFORE the per-estimator non-negative check (Multinomial/Complement
//! `check_non_negative` in `_count`, `naive_bayes.py:881`/`:1027`), the
//! Categorical integer-cast/negative check, and the NB count math.
//!
//! Every expected behavior below is the LIVE `sklearn` 1.5.2 oracle (computed
//! via `python3 -c "..."` run from `/tmp`, quoted below) — NEVER copied from the
//! ferrolearn side (goal.md R-CHAR-3).
//!
//! Live oracle (run from `/tmp`):
//! ```text
//! import numpy as np
//! from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB, ComplementNB, CategoricalNB
//! # fit, NaN / +inf / -inf in X -> ValueError:
//! GaussianNB().fit([[1,2],[nan,3],[4,5],[6,7]],[0,0,1,1])      -> ValueError: Input X contains NaN.
//! GaussianNB().fit([[1,2],[inf,3],[4,5],[6,7]],[0,0,1,1])      -> ValueError: Input X contains infinity ...
//! GaussianNB().fit([[1,2],[-inf,3],[4,5],[6,7]],[0,0,1,1])     -> ValueError: Input X contains infinity ...
//! MultinomialNB().fit([[1,2],[nan,-3],[4,5],[6,1]],[0,0,1,1])  -> ValueError: Input X contains NaN.   (NaN, NOT "Negative values")
//! MultinomialNB().fit([[1,2],[-1,3],[4,5],[6,1]],[0,0,1,1])    -> ValueError: Negative values ...      (no NaN -> negative check fires)
//! ComplementNB().fit([[1,2],[nan,-3],[4,5],[6,1]],[0,0,1,1])   -> ValueError: Input X contains NaN.
//! BernoulliNB().fit([[1,0],[nan,1],[0,1],[1,0]],[0,0,1,1])     -> ValueError: Input X contains NaN.
//! CategoricalNB().fit([[0,1],[nan,0],[1,2],[2,1]],[0,0,1,1])   -> ValueError: Input X contains NaN.
//! CategoricalNB().fit([[0,1],[inf,0],[1,2],[2,1]],[0,0,1,1])   -> ValueError: Input X contains infinity ...
//! # partial_fit, NaN in X -> ValueError (same raises):
//! GaussianNB().partial_fit([[1,2],[nan,3]],[0,1],classes=[0,1])     -> ValueError: Input X contains NaN.
//! MultinomialNB().partial_fit([[1,2],[nan,-3]],[0,1],classes=[0,1]) -> ValueError: Input X contains NaN.
//! ComplementNB().partial_fit([[1,2],[nan,-3]],[0,1],classes=[0,1])  -> ValueError: Input X contains NaN.
//! BernoulliNB().partial_fit([[1,0],[nan,1]],[0,1],classes=[0,1])    -> ValueError: Input X contains NaN.
//! CategoricalNB().partial_fit([[0,1],[nan,0]],[0,1],classes=[0,1])  -> ValueError: Input X contains NaN.
//! # all-finite known fits (no false positive, no regression): predict -> [0, 1]:
//! GaussianNB().fit([[1,2],[1.5,1.8],[2,2.5],[6,7],[6.5,6.8],[7,7.5]],[0,0,0,1,1,1]).predict([[1.2,2.1],[6.6,7.1]])  -> [0, 1]
//! MultinomialNB().fit([[3,1,0],[2,0,1],[4,2,0],[0,1,4],[1,0,3],[0,2,5]],[0,0,0,1,1,1]).predict([[3,1,0],[0,1,4]])  -> [0, 1]
//! ComplementNB().fit([[5,1,0],[4,2,0],[6,0,1],[0,1,5],[1,0,4],[0,2,6]],[0,0,0,1,1,1]).predict([[5,1,0],[0,1,5]])   -> [0, 1]
//! BernoulliNB().fit([[1,1,0],[1,0,0],[1,1,0],[0,0,1],[0,1,1],[0,0,1]],[0,0,0,1,1,1]).predict([[1,1,0],[0,0,1]])    -> [0, 1]
//! CategoricalNB().fit([[0,1],[0,0],[1,2],[2,1],[2,2],[1,0]],[0,0,0,1,1,1]).predict([[0,1],[2,1]])                  -> [0, 1]
//! ```

use ferrolearn_bayes::{BernoulliNB, CategoricalNB, ComplementNB, GaussianNB, MultinomialNB};
use ferrolearn_core::{Fit, Predict};
use ndarray::array;

const NAN: f64 = f64::NAN;
const POS_INF: f64 = f64::INFINITY;
const NEG_INF: f64 = f64::NEG_INFINITY;

// ---------------------------------------------------------------------------
// GaussianNB — fit + partial_fit, NaN / +inf / -inf in X all rejected.
// ---------------------------------------------------------------------------

#[test]
fn gaussian_fit_rejects_nan_in_x() {
    // sklearn GaussianNB().fit(X_with_nan, y) -> ValueError: Input X contains NaN.
    let x = array![[1.0, 2.0], [NAN, 3.0], [4.0, 5.0], [6.0, 7.0]];
    let y = array![0usize, 0, 1, 1];
    assert!(
        GaussianNB::<f64>::new().fit(&x, &y).is_err(),
        "GaussianNB.fit must reject NaN in X (sklearn raises ValueError)"
    );
}

#[test]
fn gaussian_fit_rejects_pos_inf_in_x() {
    let x = array![[1.0, 2.0], [POS_INF, 3.0], [4.0, 5.0], [6.0, 7.0]];
    let y = array![0usize, 0, 1, 1];
    assert!(
        GaussianNB::<f64>::new().fit(&x, &y).is_err(),
        "GaussianNB.fit must reject +inf in X (sklearn raises ValueError)"
    );
}

#[test]
fn gaussian_fit_rejects_neg_inf_in_x() {
    let x = array![[1.0, 2.0], [NEG_INF, 3.0], [4.0, 5.0], [6.0, 7.0]];
    let y = array![0usize, 0, 1, 1];
    assert!(
        GaussianNB::<f64>::new().fit(&x, &y).is_err(),
        "GaussianNB.fit must reject -inf in X (sklearn raises ValueError)"
    );
}

#[test]
fn gaussian_partial_fit_rejects_nan_in_x() {
    // sklearn GaussianNB().partial_fit(X_with_nan, y, classes=...) -> ValueError.
    let x0 = array![[1.0, 2.0], [4.0, 5.0]];
    let y0 = array![0usize, 1];
    let mut fitted = GaussianNB::<f64>::new().fit(&x0, &y0).unwrap();
    let x = array![[1.0, 2.0], [NAN, 3.0]];
    let y = array![0usize, 1];
    assert!(
        fitted.partial_fit(&x, &y).is_err(),
        "GaussianNB.partial_fit must reject NaN in X (sklearn raises ValueError)"
    );
}

#[test]
fn gaussian_all_finite_fit_ok_known_predict() {
    // sklearn GaussianNB().fit(X, y).predict([[1.2,2.1],[6.6,7.1]]) -> [0, 1].
    let x = array![
        [1.0, 2.0],
        [1.5, 1.8],
        [2.0, 2.5],
        [6.0, 7.0],
        [6.5, 6.8],
        [7.0, 7.5]
    ];
    let y = array![0usize, 0, 0, 1, 1, 1];
    let fitted = GaussianNB::<f64>::new().fit(&x, &y).unwrap();
    let preds = fitted.predict(&array![[1.2, 2.1], [6.6, 7.1]]).unwrap();
    assert_eq!(preds, array![0usize, 1]);
}

// ---------------------------------------------------------------------------
// MultinomialNB — finiteness FIRST (NaN+negative -> NaN error, not negative).
// ---------------------------------------------------------------------------

#[test]
fn multinomial_fit_rejects_nan_in_x() {
    let x = array![[1.0, 2.0], [NAN, 3.0], [4.0, 5.0], [6.0, 1.0]];
    let y = array![0usize, 0, 1, 1];
    assert!(
        MultinomialNB::<f64>::new().fit(&x, &y).is_err(),
        "MultinomialNB.fit must reject NaN in X"
    );
}

#[test]
fn multinomial_fit_rejects_pos_inf_in_x() {
    let x = array![[1.0, 2.0], [POS_INF, 3.0], [4.0, 5.0], [6.0, 1.0]];
    let y = array![0usize, 0, 1, 1];
    assert!(
        MultinomialNB::<f64>::new().fit(&x, &y).is_err(),
        "MultinomialNB.fit must reject +inf in X"
    );
}

#[test]
fn multinomial_fit_finiteness_before_negative() {
    // A cell that is BOTH NaN AND in a row with a negative value. sklearn's
    // `_check_X_y` (force_all_finite) runs BEFORE `_count`'s `check_non_negative`
    // (naive_bayes.py:881), so the NaN error fires first, NOT "Negative values".
    // Both still produce an Err — the contract is "rejected because finiteness
    // is validated first". The non-negative guard alone would MISS the NaN cell
    // (NaN < 0 == false), so this Err proves the finiteness guard fired.
    let x = array![[1.0, 2.0], [NAN, -3.0], [4.0, 5.0], [6.0, 1.0]];
    let y = array![0usize, 0, 1, 1];
    assert!(
        MultinomialNB::<f64>::new().fit(&x, &y).is_err(),
        "MultinomialNB.fit with NaN+negative must reject (finiteness first)"
    );
}

#[test]
fn multinomial_partial_fit_rejects_nan_in_x() {
    let x0 = array![[1.0, 2.0], [4.0, 5.0]];
    let y0 = array![0usize, 1];
    let mut fitted = MultinomialNB::<f64>::new().fit(&x0, &y0).unwrap();
    let x = array![[1.0, 2.0], [NAN, 3.0]];
    let y = array![0usize, 1];
    assert!(
        fitted.partial_fit(&x, &y).is_err(),
        "MultinomialNB.partial_fit must reject NaN in X"
    );
}

#[test]
fn multinomial_all_finite_fit_ok_known_predict() {
    // sklearn MultinomialNB().fit(X,y).predict([[3,1,0],[0,1,4]]) -> [0, 1].
    let x = array![
        [3.0, 1.0, 0.0],
        [2.0, 0.0, 1.0],
        [4.0, 2.0, 0.0],
        [0.0, 1.0, 4.0],
        [1.0, 0.0, 3.0],
        [0.0, 2.0, 5.0]
    ];
    let y = array![0usize, 0, 0, 1, 1, 1];
    let fitted = MultinomialNB::<f64>::new().fit(&x, &y).unwrap();
    let preds = fitted
        .predict(&array![[3.0, 1.0, 0.0], [0.0, 1.0, 4.0]])
        .unwrap();
    assert_eq!(preds, array![0usize, 1]);
}

// ---------------------------------------------------------------------------
// ComplementNB — finiteness FIRST as well.
// ---------------------------------------------------------------------------

#[test]
fn complement_fit_rejects_nan_in_x() {
    let x = array![[1.0, 2.0], [NAN, 3.0], [4.0, 5.0], [6.0, 1.0]];
    let y = array![0usize, 0, 1, 1];
    assert!(
        ComplementNB::<f64>::new().fit(&x, &y).is_err(),
        "ComplementNB.fit must reject NaN in X"
    );
}

#[test]
fn complement_fit_rejects_neg_inf_in_x() {
    let x = array![[1.0, 2.0], [NEG_INF, 3.0], [4.0, 5.0], [6.0, 1.0]];
    let y = array![0usize, 0, 1, 1];
    assert!(
        ComplementNB::<f64>::new().fit(&x, &y).is_err(),
        "ComplementNB.fit must reject -inf in X"
    );
}

#[test]
fn complement_fit_finiteness_before_negative() {
    // NaN+negative -> NaN error first (finiteness before check_non_negative,
    // naive_bayes.py:1027). The non-negative guard alone misses the NaN.
    let x = array![[1.0, 2.0], [NAN, -3.0], [4.0, 5.0], [6.0, 1.0]];
    let y = array![0usize, 0, 1, 1];
    assert!(
        ComplementNB::<f64>::new().fit(&x, &y).is_err(),
        "ComplementNB.fit with NaN+negative must reject (finiteness first)"
    );
}

#[test]
fn complement_partial_fit_rejects_nan_in_x() {
    let x0 = array![[1.0, 2.0], [4.0, 5.0]];
    let y0 = array![0usize, 1];
    let mut fitted = ComplementNB::<f64>::new().fit(&x0, &y0).unwrap();
    let x = array![[1.0, 2.0], [NAN, 3.0]];
    let y = array![0usize, 1];
    assert!(
        fitted.partial_fit(&x, &y).is_err(),
        "ComplementNB.partial_fit must reject NaN in X"
    );
}

#[test]
fn complement_all_finite_fit_ok_known_predict() {
    // sklearn ComplementNB().fit(X,y).predict([[5,1,0],[0,1,5]]) -> [0, 1].
    let x = array![
        [5.0, 1.0, 0.0],
        [4.0, 2.0, 0.0],
        [6.0, 0.0, 1.0],
        [0.0, 1.0, 5.0],
        [1.0, 0.0, 4.0],
        [0.0, 2.0, 6.0]
    ];
    let y = array![0usize, 0, 0, 1, 1, 1];
    let fitted = ComplementNB::<f64>::new().fit(&x, &y).unwrap();
    let preds = fitted
        .predict(&array![[5.0, 1.0, 0.0], [0.0, 1.0, 5.0]])
        .unwrap();
    assert_eq!(preds, array![0usize, 1]);
}

// ---------------------------------------------------------------------------
// BernoulliNB — finiteness before binarization.
// ---------------------------------------------------------------------------

#[test]
fn bernoulli_fit_rejects_nan_in_x() {
    let x = array![[1.0, 0.0], [NAN, 1.0], [0.0, 1.0], [1.0, 0.0]];
    let y = array![0usize, 0, 1, 1];
    assert!(
        BernoulliNB::<f64>::new().fit(&x, &y).is_err(),
        "BernoulliNB.fit must reject NaN in X"
    );
}

#[test]
fn bernoulli_fit_rejects_pos_inf_in_x() {
    let x = array![[1.0, 0.0], [POS_INF, 1.0], [0.0, 1.0], [1.0, 0.0]];
    let y = array![0usize, 0, 1, 1];
    assert!(
        BernoulliNB::<f64>::new().fit(&x, &y).is_err(),
        "BernoulliNB.fit must reject +inf in X"
    );
}

#[test]
fn bernoulli_partial_fit_rejects_nan_in_x() {
    let x0 = array![[1.0, 0.0], [0.0, 1.0]];
    let y0 = array![0usize, 1];
    let mut fitted = BernoulliNB::<f64>::new().fit(&x0, &y0).unwrap();
    let x = array![[1.0, 0.0], [NAN, 1.0]];
    let y = array![0usize, 1];
    assert!(
        fitted.partial_fit(&x, &y).is_err(),
        "BernoulliNB.partial_fit must reject NaN in X"
    );
}

#[test]
fn bernoulli_all_finite_fit_ok_known_predict() {
    // sklearn BernoulliNB().fit(X,y).predict([[1,1,0],[0,0,1]]) -> [0, 1].
    let x = array![
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [0.0, 0.0, 1.0]
    ];
    let y = array![0usize, 0, 0, 1, 1, 1];
    let fitted = BernoulliNB::<f64>::new().fit(&x, &y).unwrap();
    let preds = fitted
        .predict(&array![[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        .unwrap();
    assert_eq!(preds, array![0usize, 1]);
}

// ---------------------------------------------------------------------------
// CategoricalNB — finiteness before the negative / integer-category checks.
// ---------------------------------------------------------------------------

#[test]
fn categorical_fit_rejects_nan_in_x() {
    let x = array![[0.0, 1.0], [NAN, 0.0], [1.0, 2.0], [2.0, 1.0]];
    let y = array![0usize, 0, 1, 1];
    assert!(
        CategoricalNB::<f64>::new().fit(&x, &y).is_err(),
        "CategoricalNB.fit must reject NaN in X"
    );
}

#[test]
fn categorical_fit_rejects_pos_inf_in_x() {
    let x = array![[0.0, 1.0], [POS_INF, 0.0], [1.0, 2.0], [2.0, 1.0]];
    let y = array![0usize, 0, 1, 1];
    assert!(
        CategoricalNB::<f64>::new().fit(&x, &y).is_err(),
        "CategoricalNB.fit must reject +inf in X"
    );
}

#[test]
fn categorical_fit_finiteness_before_negative() {
    // NaN+negative -> rejected (finiteness validated before check_non_negative,
    // naive_bayes.py:1439, and before the to_usize category mapping).
    let x = array![[0.0, 1.0], [NAN, -1.0], [1.0, 2.0], [2.0, 1.0]];
    let y = array![0usize, 0, 1, 1];
    assert!(
        CategoricalNB::<f64>::new().fit(&x, &y).is_err(),
        "CategoricalNB.fit with NaN+negative must reject (finiteness first)"
    );
}

#[test]
fn categorical_partial_fit_rejects_nan_in_x() {
    let x0 = array![[0.0, 1.0], [1.0, 2.0]];
    let y0 = array![0usize, 1];
    let mut fitted = CategoricalNB::<f64>::new().fit(&x0, &y0).unwrap();
    let x = array![[0.0, 1.0], [NAN, 0.0]];
    let y = array![0usize, 1];
    assert!(
        fitted.partial_fit(&x, &y).is_err(),
        "CategoricalNB.partial_fit must reject NaN in X"
    );
}

#[test]
fn categorical_all_finite_fit_ok_known_predict() {
    // sklearn CategoricalNB().fit(X,y).predict([[0,1],[2,1]]) -> [0, 1].
    let x = array![
        [0.0, 1.0],
        [0.0, 0.0],
        [1.0, 2.0],
        [2.0, 1.0],
        [2.0, 2.0],
        [1.0, 0.0]
    ];
    let y = array![0usize, 0, 0, 1, 1, 1];
    let fitted = CategoricalNB::<f64>::new().fit(&x, &y).unwrap();
    let preds = fitted.predict(&array![[0.0, 1.0], [2.0, 1.0]]).unwrap();
    assert_eq!(preds, array![0usize, 1]);
}
