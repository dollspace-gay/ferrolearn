//! Divergence audit (ACToR critic) for `ferrolearn-model-sel::inspection`
//! against scikit-learn 1.5.2.
//!
//! Oracle (R-CHAR-3): every expected value below is produced by a LIVE
//! sklearn 1.5.2 call run from /tmp, NEVER copied from ferrolearn. The
//! commands are reproduced in each test's doc-comment.
//!
//! Coverage:
//! - GREEN guard 1 (REQ-1): PD brute-averaging value parity vs a live
//!   `LinearRegression` brute PD `[1.5, 2.0, 2.5, 6.5]`.
//! - GREEN guard 2 (REQ-6): permutation_importance structure — shapes,
//!   the `baseline - permuted` sign convention, and `np.std` ddof=0.
//!
//! No real divergence was found; both guards PASS. The RNG carve-out
//! (REQ-7) and the feature/API gaps (REQ-3/4/5/8/9) are blocker-only per
//! R-DEFER-3 / R-DEFER-2 and carry NO failing test.

use ferrolearn_core::FerroError;
use ferrolearn_model_sel::{partial_dependence, permutation_importance};
use ndarray::{Array1, Array2, array};

/// GREEN guard — REQ-1 (PD brute-averaging value parity).
///
/// Divergence target: ferrolearn's `partial_dependence` brute averaging vs
/// `sklearn/inspection/_partial_dependence.py:308`
/// `averaged_predictions.append(np.average(pred, axis=0, weights=sample_weight))`,
/// with the per-grid column overwrite at `:294-295` and `X_eval = X.copy()`
/// at `:292`.
///
/// Live oracle (sklearn 1.5.2, run from /tmp):
/// ```text
/// python3 -c "import numpy as np; from sklearn.linear_model import LinearRegression; \
/// X=np.array([[1.,2.],[3.,4.],[5.,6.],[7.,8.]]); y=X@np.array([2.,-1.])+0.5; \
/// m=LinearRegression().fit(X,y); grid=[0.,1.,2.,10.]; \
/// print([float(m.predict(np.column_stack([np.full(4,v),X[:,1]])).mean()) for v in grid])"
/// # -> [1.5, 2.0, 2.5, 6.5]
/// # fitted coef_ == [0.5, 0.5], intercept_ == -1.0
/// ```
///
/// The closure below encodes the fitted predictor `predict(X) = X·[0.5,0.5] - 1.0`.
/// sklearn returns `[1.5, 2.0, 2.5, 6.5]`; ferrolearn must match within 1e-9.
#[test]
fn divergence_pd_brute_value_parity_linear_oracle() {
    // predict(X) = X·[0.5, 0.5] - 1.0  (the live LinearRegression fit).
    let predict = |x: &Array2<f64>| -> Result<Array1<f64>, FerroError> {
        let n = x.nrows();
        let mut out = Array1::<f64>::zeros(n);
        for i in 0..n {
            out[i] = 0.5 * x[[i, 0]] + 0.5 * x[[i, 1]] - 1.0;
        }
        Ok(out)
    };
    let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
    let grid = array![0.0, 1.0, 2.0, 10.0];

    let res = partial_dependence(predict, &x, 0, &grid).unwrap();

    // Live-oracle brute PD (sklearn 1.5.2): never copied from ferrolearn.
    let sk_pd = [1.5_f64, 2.0, 2.5, 6.5];
    assert_eq!(res.averaged_predictions.len(), sk_pd.len());
    for (got, &want) in res.averaged_predictions.iter().zip(sk_pd.iter()) {
        assert!(
            (got - want).abs() < 1e-9,
            "PD brute averaging diverges: got {got}, sklearn oracle {want}"
        );
    }
    // sklearn also returns the grid it evaluated on (here: the explicit grid).
    for (got, &want) in res.grid.iter().zip(grid.iter()) {
        assert!((got - want).abs() < 1e-12);
    }
}

/// GREEN guard — REQ-1 corollary: PD averages (mean), not sums.
///
/// Pins the `np.average(pred, axis=0)` reduction at
/// `sklearn/inspection/_partial_dependence.py:308` against a sum-vs-mean
/// divergence. With a predictor that returns the (constant) overwritten
/// column value, the brute PD at grid value `v` over an n-row `X` is the
/// MEAN `v`, not the sum `n*v`.
///
/// Live oracle (sklearn 1.5.2, run from /tmp):
/// ```text
/// python3 -c "import numpy as np; from sklearn.linear_model import LinearRegression; \
/// X=np.array([[1.,9.],[2.,9.],[3.,9.],[4.,9.],[5.,9.]]); y=X[:,0].copy(); \
/// m=LinearRegression().fit(X,y); grid=[10.,20.]; \
/// print([float(m.predict(np.column_stack([np.full(5,v),X[:,1]])).mean()) for v in grid])"
/// # -> [10.0, 20.0]   (mean of a constant column == the grid value; sum would be 50/100)
/// ```
#[test]
fn divergence_pd_uses_mean_not_sum() {
    // predict returns column 0 exactly (a fit of y = x0 on a constant col1).
    let predict =
        |x: &Array2<f64>| -> Result<Array1<f64>, FerroError> { Ok(x.column(0).to_owned()) };
    let x = array![[1.0, 9.0], [2.0, 9.0], [3.0, 9.0], [4.0, 9.0], [5.0, 9.0]];
    let grid = array![10.0, 20.0];

    let res = partial_dependence(predict, &x, 0, &grid).unwrap();

    // sklearn oracle: mean of a 5-row constant column == the grid value.
    assert!((res.averaged_predictions[0] - 10.0).abs() < 1e-9);
    assert!((res.averaged_predictions[1] - 20.0).abs() < 1e-9);
}

/// GREEN guard — REQ-6 (permutation_importance structure: shapes + sign + ddof=0).
///
/// Pins three observable invariants of `_create_importances_bunch`
/// (`sklearn/inspection/_permutation_importance.py:103-107`):
///  1. `importances = baseline_score - permuted_score` (`:103`) — a feature
///     whose shuffle LOWERS the score has POSITIVE importance.
///  2. shapes `(n_features, n_repeats)` / `(n_features,)` / `(n_features,)`.
///  3. `importances_std = np.std(importances, axis=1)` (`:106`, ddof=0).
///
/// Here `score(X, y)` is maximal at the identity row ordering, so any shuffle
/// of column 0 (strictly ascending) lowers the score. Because the deterministic
/// seed drives a real shuffle we assert only the SIGN and the structure, not
/// the exact per-repeat values (REQ-7 carve-out).
///
/// Live oracle for the SIGN (sklearn 1.5.2, run from /tmp): a permuted score
/// below baseline yields a positive importance.
/// ```text
/// python3 -c "import numpy as np; \
/// from sklearn.inspection._permutation_importance import _create_importances_bunch; \
/// b=_create_importances_bunch(np.array([1.0]), np.array([[0.2,0.3,0.1,0.4]])); \
/// print(b.importances_mean.tolist())  # baseline 1.0 > permuted -> [0.75] (>0)"
/// ```
#[test]
fn divergence_permutation_sign_and_shape() {
    // A score that is maximal at the identity ordering: sum_i (x[i,0] * i).
    // Any shuffle of column 0 (sorted ascending) lowers this score, so the
    // baseline-minus-permuted importance for column 0 must be >= 0.
    let score = |x: &Array2<f64>, _y: &Array1<f64>| -> Result<f64, FerroError> {
        let mut acc = 0.0;
        for i in 0..x.nrows() {
            acc += x[[i, 0]] * (i as f64);
        }
        Ok(acc)
    };
    // Column 0 strictly ascending so the identity ordering is the unique max.
    let x = array![[0.0, 5.0], [1.0, 5.0], [2.0, 5.0], [3.0, 5.0], [4.0, 5.0]];
    let y = Array1::<f64>::zeros(5);
    let res = permutation_importance(score, &x, &y, 4, Some(123)).unwrap();

    // Shapes (sklearn :96-100 / :108).
    assert_eq!(res.importances.dim(), (2, 4));
    assert_eq!(res.importances_mean.len(), 2);
    assert_eq!(res.importances_std.len(), 2);

    // Sign convention (sklearn :103): shuffling the score-driving column 0
    // can only lower the score, so importance >= 0 (positive for useful).
    assert!(
        res.importances_mean[0] >= 0.0,
        "permutation sign inverted: col-0 importance {} should be >= 0 (baseline - permuted)",
        res.importances_mean[0]
    );
    // Column 1 is constant -> shuffling never changes the score -> exactly 0.
    assert!(
        res.importances_mean[1].abs() < 1e-12,
        "constant column 1 should have 0 importance, got {}",
        res.importances_mean[1]
    );
    for r in 0..4 {
        assert!(res.importances[[1, r]].abs() < 1e-12);
    }
}

/// GREEN guard — REQ-6 (std ddof=0 reduction), oracle-grounded.
///
/// Pins `importances_std = np.std(importances, axis=1)`
/// (`sklearn/inspection/_permutation_importance.py:106`, numpy default ddof=0)
/// against an off-by-one ddof divergence (dividing by `n_repeats - 1`).
///
/// We drive a DETERMINISTIC score whose per-repeat importances are exactly
/// `[1, 2, 4, 8]` regardless of the shuffle: `score` returns a value read from
/// a shared counter, so the recorded `baseline - s` walks `1, 2, 4, 8`. The
/// per-repeat sequence is chosen so neither the ddof=0 nor the ddof=1 std is a
/// recognizable math constant (avoids the SQRT_2 collision of `[1..5]`). The
/// mean must be `3.75` and the std the ddof=0 value.
///
/// Live oracle (sklearn 1.5.2, run from /tmp):
/// ```text
/// python3 -c "import numpy as np; \
/// from sklearn.inspection._permutation_importance import _create_importances_bunch; \
/// b=_create_importances_bunch(np.array([0.0]), -np.array([[1.,2.,4.,8.]])); \
/// print(b.importances_mean.tolist(), repr(b.importances_std[0])); \
/// print(repr(np.std([1.,2.,4.,8.], ddof=1)))"
/// # -> [3.75] 2.680951323690902   (ddof=0; ddof=1 would be 3.095695936834452)
/// ```
#[test]
fn divergence_permutation_std_ddof0_oracle() {
    use std::cell::Cell;
    // Counter of how many score calls have happened. The baseline call
    // (counter 0) returns 0.0; permuted calls return -1, -2, -4, -8 so that
    // importance = baseline - s = 0 - (-k) = k.
    let perm_vals = [1.0_f64, 2.0, 4.0, 8.0];
    let counter = Cell::new(0usize);
    let score = |_x: &Array2<f64>, _y: &Array1<f64>| -> Result<f64, FerroError> {
        let c = counter.get();
        counter.set(c + 1);
        if c == 0 {
            Ok(0.0) // baseline
        } else {
            Ok(-perm_vals[c - 1]) // permuted: -1, -2, -4, -8
        }
    };
    // Single feature so the 4 repeats consume permuted calls 1..=4 in order.
    let x = array![[0.0], [0.0], [0.0]];
    let y = Array1::<f64>::zeros(3);
    let res = permutation_importance(score, &x, &y, 4, Some(0)).unwrap();

    // Per-repeat importances are the sequence [1,2,4,8]; sklearn oracle:
    let sk_mean = 3.75_f64;
    let sk_std_ddof0 = 2.680951323690902_f64;
    let sk_std_ddof1 = 3.095695936834452_f64; // what a divergence would give

    assert!(
        (res.importances_mean[0] - sk_mean).abs() < 1e-12,
        "permutation mean diverges: got {}, sklearn {sk_mean}",
        res.importances_mean[0]
    );
    assert!(
        (res.importances_std[0] - sk_std_ddof0).abs() < 1e-12,
        "permutation std uses wrong ddof: got {}, sklearn ddof=0 is {sk_std_ddof0} \
         (ddof=1 would be {sk_std_ddof1})",
        res.importances_std[0]
    );
    // Guard the divergent value explicitly: must NOT equal the ddof=1 result.
    assert!(
        (res.importances_std[0] - sk_std_ddof1).abs() > 1e-9,
        "permutation std appears to use ddof=1 ({sk_std_ddof1}), diverging from sklearn ddof=0"
    );
}

/// GREEN guard — REQ-6 (std ddof=0), two-value sanity per the audit brief.
///
/// For per-repeat values `[a, b]`, the ddof=0 std is `|a-b|/2`, NOT the
/// sample (ddof=1) std `|a-b|/sqrt(2)`. Pins `:106` for `n_repeats=2`.
///
/// Live oracle (sklearn 1.5.2, run from /tmp):
/// ```text
/// python3 -c "import numpy as np; \
/// from sklearn.inspection._permutation_importance import _create_importances_bunch; \
/// b=_create_importances_bunch(np.array([0.0]), -np.array([[3.,7.]])); \
/// print(b.importances_std.tolist())  # -> [2.0]  == |3-7|/2
/// ```
#[test]
fn divergence_permutation_std_two_value_oracle() {
    use std::cell::Cell;
    let counter = Cell::new(0usize);
    let score = |_x: &Array2<f64>, _y: &Array1<f64>| -> Result<f64, FerroError> {
        let c = counter.get();
        counter.set(c + 1);
        match c {
            0 => Ok(0.0),  // baseline
            1 => Ok(-3.0), // importance 3
            _ => Ok(-7.0), // importance 7
        }
    };
    let x = array![[0.0], [0.0]];
    let y = Array1::<f64>::zeros(2);
    let res = permutation_importance(score, &x, &y, 2, Some(0)).unwrap();

    // sklearn ddof=0 std of [3,7] is |3-7|/2 = 2.0.
    let sk_std = 2.0_f64;
    assert!(
        (res.importances_std[0] - sk_std).abs() < 1e-12,
        "two-value std diverges: got {}, sklearn ddof=0 |a-b|/2 = {sk_std}",
        res.importances_std[0]
    );
}
