//! Divergence pins for `IsotonicRegression` against the live scikit-learn
//! 1.5.2 oracle (`sklearn/isotonic.py`, commit 156ef14).
//!
//! IsotonicRegression is fully DETERMINISTIC (no RNG): the PAVA pool, the
//! `_make_unique` weighted duplicate-X collapse, and the piecewise-linear
//! interpolant are all closed-form functions of `(X, y)`. So full
//! predicted-value parity is assertable — these tests assert exact predicted
//! values (the fitted thresholds surface through `predict`, the only public
//! observable, since ferrolearn exposes no `X_thresholds_`/`y_thresholds_`
//! accessors yet — REQ-9 / #570).
//!
//! Every expected value is produced by RUNNING scikit-learn 1.5.2 (the live
//! oracle), never copied from ferrolearn (goal.md R-CHAR-3). The exact python
//! invocation is recorded above each oracle constant.
//!
//! # Path confirmed ALREADY CORRECT — no failing test written
//!
//! REQ-1 (increasing PAVA pool on DISTINCT, equal-weight X). Oracle:
//! ```text
//! python3 -c "import numpy as np; from sklearn.isotonic import IsotonicRegression; \
//!   m=IsotonicRegression(out_of_bounds='clip').fit(np.array([1.,2.,3.,4.,5.,6.]).reshape(-1,1), np.array([1.,4.,2.,5.,3.,7.])); \
//!   print(m.predict(np.array([1.,2.,3.,4.,5.,6.]).reshape(-1,1)).tolist())"
//!   # -> [1.0, 3.0, 3.0, 4.0, 4.0, 7.0]
//! ```
//! ferrolearn `predict([1..6])` on the same fit returns `[1.0, 3.0, 3.0, 4.0,
//! 4.0, 7.0]` (probed directly) — it MATCHES the oracle. A passing path is not
//! a divergence (task instruction #3), so no `isotonic_pava_pooled_values` test
//! is written; REQ-1's core unweighted PAVA on distinct X is correct.

use ferrolearn_core::traits::{Fit, Predict};
use ferrolearn_linear::isotonic::{IsotonicRegression, check_increasing, isotonic_regression};
use ndarray::{Array2, array};

/// Divergence: a DEFAULT-constructed `IsotonicRegression` clips out-of-range
/// predictions instead of returning NaN.
///
/// sklearn's constructor defaults `out_of_bounds="nan"`
/// (`sklearn/isotonic.py:274`: `def __init__(self, *, y_min=None, y_max=None,
/// increasing=True, out_of_bounds="nan")`; `_parameter_constraints`
/// `"out_of_bounds": [StrOptions({"nan", "clip", "raise"})]`,
/// `isotonic.py:271`), so a default fit predicting BELOW/ABOVE the training
/// range returns `NaN`. ferrolearn's `IsotonicRegression::new` defaults
/// `out_of_bounds: OutOfBounds::Clip` (`ferrolearn-linear/src/isotonic.rs:82`),
/// so it clips to the boundary y instead — a default-ABI divergence (R-DEV-2).
///
/// Oracle:
/// ```text
/// python3 -c "import numpy as np; from sklearn.isotonic import IsotonicRegression; \
///   print(repr(IsotonicRegression().out_of_bounds)); \
///   m=IsotonicRegression().fit(np.array([1.,2.,3.]).reshape(-1,1), np.array([1.,2.,3.])); \
///   print(m.predict(np.array([0.,4.]).reshape(-1,1)).tolist())"
///   # -> 'nan'
///   # -> [nan, nan]
/// ```
/// sklearn returns `[nan, nan]`; ferrolearn (default = Clip) returns `[1.0,
/// 3.0]`. Asserting `.is_nan()` fails because both are finite boundary values.
///
/// Tracking: #565
#[test]
fn isotonic_default_out_of_bounds_nan() {
    // Fit on X=[1,2,3], y=[1,2,3] with the DEFAULT constructor (no
    // .with_out_of_bounds() — exercising the default exactly as sklearn's
    // `IsotonicRegression()` does).
    let x = Array2::from_shape_vec((3, 1), vec![1.0_f64, 2.0, 3.0]).unwrap();
    let y = array![1.0_f64, 2.0, 3.0];

    let fitted = IsotonicRegression::<f64>::new().fit(&x, &y).unwrap();

    // Predict at X=[0.0, 4.0] — below and above the training range [1, 3].
    let x_oob = Array2::from_shape_vec((2, 1), vec![0.0_f64, 4.0]).unwrap();
    let preds = fitted.predict(&x_oob).unwrap();

    // sklearn 1.5.2 oracle: predict([0., 4.]) == [nan, nan] (default
    // out_of_bounds="nan"). Pin the default behavior via NaN checks.
    assert!(
        preds[0].is_nan(),
        "default-constructed IsotonicRegression predict(0.0) below range: \
         sklearn returns NaN (out_of_bounds default 'nan', isotonic.py:274), \
         ferrolearn returned {} (default Clip, isotonic.rs:82)",
        preds[0]
    );
    assert!(
        preds[1].is_nan(),
        "default-constructed IsotonicRegression predict(4.0) above range: \
         sklearn returns NaN (out_of_bounds default 'nan', isotonic.py:274), \
         ferrolearn returned {} (default Clip, isotonic.rs:82)",
        preds[1]
    );
}

/// Divergence: ferrolearn does not perform `_make_unique` weighted
/// duplicate-X collapse BEFORE the PAVA pool, so duplicate-X inputs fit a
/// different monotone function.
///
/// sklearn sorts `np.lexsort((y, X))` then calls `_make_unique`
/// (`sklearn/isotonic.py:317-319`: `order = np.lexsort((y, X)); ...; unique_X,
/// unique_y, unique_sample_weight = _make_unique(X, y, sample_weight)`), which
/// collapses each run of equal X into ONE point whose y is the weighted mean of
/// the run, BEFORE running `isotonic_regression` (`isotonic.py:322`). For
/// X=[1,1,2,3], y=[1,3,2,4] the duplicate X=1 collapses to y=mean(1,3)=2, and
/// the PAVA then leaves [2,2,4] (the (2,...) at X=2 pools with the leading 2).
/// ferrolearn's `pav_increasing` sorts by X ONLY (no `_make_unique`,
/// `isotonic.rs:213`) and pools the raw [1,3,2,4], yielding y_thr ≈ [1,2.5,4].
///
/// Oracle:
/// ```text
/// python3 -c "import numpy as np; from sklearn.isotonic import IsotonicRegression; \
///   m=IsotonicRegression(out_of_bounds='clip').fit(np.array([1.,1.,2.,3.]).reshape(-1,1), np.array([1.,3.,2.,4.])); \
///   print(m.X_thresholds_.tolist(), m.y_thresholds_.tolist()); \
///   print(m.predict(np.array([1.,2.,3.]).reshape(-1,1)).tolist())"
///   # -> [1.0, 2.0, 3.0] [2.0, 2.0, 4.0]
///   # -> [2.0, 2.0, 4.0]
/// ```
/// sklearn `predict([1,2,3]) == [2.0, 2.0, 4.0]`; ferrolearn returns
/// `[1.0, 2.5, 4.0]` (probed directly).
///
/// Tracking: #569
#[test]
fn isotonic_make_unique_duplicate_x() {
    let x = Array2::from_shape_vec((4, 1), vec![1.0_f64, 1.0, 2.0, 3.0]).unwrap();
    let y = array![1.0_f64, 3.0, 2.0, 4.0];

    // out_of_bounds=Clip so the in-range query points are unaffected by the
    // default-NaN divergence (#565) — this isolates the _make_unique gap.
    let fitted = IsotonicRegression::<f64>::new()
        .with_out_of_bounds(ferrolearn_linear::isotonic::OutOfBounds::Clip)
        .fit(&x, &y)
        .unwrap();

    let x_q = Array2::from_shape_vec((3, 1), vec![1.0_f64, 2.0, 3.0]).unwrap();
    let preds = fitted.predict(&x_q).unwrap();

    // sklearn 1.5.2 oracle: predict([1.,2.,3.]) == [2.0, 2.0, 4.0].
    let oracle = [2.0_f64, 2.0, 4.0];
    for (i, &exp) in oracle.iter().enumerate() {
        assert!(
            (preds[i] - exp).abs() < 1e-10,
            "duplicate-X fit predict({}): sklearn (_make_unique, isotonic.py:319) \
             returns {exp}, ferrolearn (no _make_unique, isotonic.rs:213) returned {}",
            [1.0, 2.0, 3.0][i],
            preds[i]
        );
    }
}

/// Parity: `IsotonicRegression::fit_with_sample_weight` reproduces sklearn's
/// weighted fit. `sample_weight` enters `IsotonicRegression.fit(X, y,
/// sample_weight)` (`sklearn/isotonic.py:251`, dispatched into `_build_y` at
/// `isotonic.py:300-328`): the weighted `_make_unique` collapses equal-X runs
/// to the sample-weighted mean and summed weight, and `isotonic_regression`
/// pools with those weights. Up-weighting the X=3 sample (`w=5` vs `1`) drags
/// the pooled [2,3] block toward y=2, giving 2.1666… instead of the unweighted
/// 2.5.
///
/// Oracle:
/// ```text
/// python3 -c "import numpy as np; from sklearn.isotonic import IsotonicRegression; \
///   m=IsotonicRegression().fit(np.array([1.,2.,3.,4.]),np.array([1.,3.,2.,5.]),sample_weight=np.array([1.,1.,5.,1.])); \
///   print(repr(m.predict([1.,2.,3.,4.]).tolist()))"
///   # -> [1.0, 2.1666666666666665, 2.1666666666666665, 5.0]
/// ```
/// (unweighted on the same X,y is `[1.0, 2.5, 2.5, 5.0]`). All query X are
/// in-range, so the default out_of_bounds='nan' never fires.
///
/// Tracking: #568
#[test]
fn isotonic_sample_weight() {
    let x = Array2::from_shape_vec((4, 1), vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    let y = array![1.0_f64, 3.0, 2.0, 5.0];
    let sample_weight = array![1.0_f64, 1.0, 5.0, 1.0];

    let fitted = IsotonicRegression::<f64>::new()
        .fit_with_sample_weight(&x, &y, &sample_weight)
        .unwrap();

    let preds = fitted.predict(&x).unwrap();

    // sklearn 1.5.2 oracle (recorded above): all four X are in the training
    // range [1, 4], so default 'nan' out_of_bounds is irrelevant.
    let oracle = [
        1.0_f64,
        2.166_666_666_666_666_5,
        2.166_666_666_666_666_5,
        5.0,
    ];
    for (i, &exp) in oracle.iter().enumerate() {
        assert!(
            (preds[i] - exp).abs() < 1e-9,
            "weighted fit predict({}): sklearn (fit sample_weight, isotonic.py:251) \
             returns {exp}, ferrolearn fit_with_sample_weight returned {}",
            [1.0, 2.0, 3.0, 4.0][i],
            preds[i]
        );
    }
}

/// Parity: `y_min`/`y_max` clip the pooled `y_thresholds` after PAVA.
///
/// sklearn's `IsotonicRegression(y_min=..., y_max=...)` clips the pooled `y`
/// (the `y_thresholds_`) to `[y_min, y_max]` AFTER the PAVA pool
/// (`sklearn/isotonic.py:163-170`: `if y_min is not None or y_max is not None:
/// ...; np.clip(y, y_min, y_max, y)`; unset bounds default to `±inf`,
/// `isotonic.py:165-168`); the constructor declares `y_min=None, y_max=None`
/// (`isotonic.py:274`). On `X=[1,2,3,4,5], y=[1,2,3,4,5]` (already monotone, so
/// `y_thresholds_ == [1,2,3,4,5]`), clipping the thresholds caps the linear
/// interpolant, so `predict` at the training X reflects the clipped thresholds.
///
/// Oracle:
/// ```text
/// python3 -c "import numpy as np; from sklearn.isotonic import IsotonicRegression; \
///   X=np.array([1.,2.,3.,4.,5.]); y=np.array([1.,2.,3.,4.,5.]); \
///   print(IsotonicRegression(y_min=2.0).fit(X,y).predict(X).tolist(), \
///         IsotonicRegression(y_max=4.0).fit(X,y).predict(X).tolist(), \
///         IsotonicRegression(y_min=2.0,y_max=4.0).fit(X,y).predict(X).tolist())"
///   # -> [2.0, 2.0, 3.0, 4.0, 5.0] [1.0, 2.0, 3.0, 4.0, 4.0] [2.0, 2.0, 3.0, 4.0, 4.0]
/// ```
/// All query X are in-range, so the default out_of_bounds='nan' never fires.
///
/// Tracking: #566
#[test]
fn isotonic_y_min_y_max() {
    let x = Array2::from_shape_vec((5, 1), vec![1.0_f64, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let y = array![1.0_f64, 2.0, 3.0, 4.0, 5.0];

    // y_min=2.0 -> [2,2,3,4,5]
    let fitted_min = IsotonicRegression::<f64>::new()
        .with_y_min(2.0)
        .fit(&x, &y)
        .unwrap();
    let preds_min = fitted_min.predict(&x).unwrap();
    let oracle_min = [2.0_f64, 2.0, 3.0, 4.0, 5.0];

    // y_max=4.0 -> [1,2,3,4,4]
    let fitted_max = IsotonicRegression::<f64>::new()
        .with_y_max(4.0)
        .fit(&x, &y)
        .unwrap();
    let preds_max = fitted_max.predict(&x).unwrap();
    let oracle_max = [1.0_f64, 2.0, 3.0, 4.0, 4.0];

    // y_min=2.0, y_max=4.0 -> [2,2,3,4,4]
    let fitted_both = IsotonicRegression::<f64>::new()
        .with_y_min(2.0)
        .with_y_max(4.0)
        .fit(&x, &y)
        .unwrap();
    let preds_both = fitted_both.predict(&x).unwrap();
    let oracle_both = [2.0_f64, 2.0, 3.0, 4.0, 4.0];

    for i in 0..5 {
        assert!(
            (preds_min[i] - oracle_min[i]).abs() < 1e-9,
            "y_min=2.0 predict({}): sklearn (np.clip, isotonic.py:163-170) returns {}, \
             ferrolearn returned {}",
            [1.0, 2.0, 3.0, 4.0, 5.0][i],
            oracle_min[i],
            preds_min[i]
        );
        assert!(
            (preds_max[i] - oracle_max[i]).abs() < 1e-9,
            "y_max=4.0 predict({}): sklearn (np.clip, isotonic.py:163-170) returns {}, \
             ferrolearn returned {}",
            [1.0, 2.0, 3.0, 4.0, 5.0][i],
            oracle_max[i],
            preds_max[i]
        );
        assert!(
            (preds_both[i] - oracle_both[i]).abs() < 1e-9,
            "y_min=2.0,y_max=4.0 predict({}): sklearn (np.clip, isotonic.py:163-170) returns {}, \
             ferrolearn returned {}",
            [1.0, 2.0, 3.0, 4.0, 5.0][i],
            oracle_both[i],
            preds_both[i]
        );
    }
}

/// Parity: `increasing='auto'` resolves the monotonicity direction from the
/// data via a Spearman correlation test (#567).
///
/// sklearn's `_build_y` runs `self.increasing_ = check_increasing(X, y)` when
/// `self.increasing == "auto"` (`sklearn/isotonic.py:306-307`), where
/// `check_increasing` returns `rho >= 0` for the Spearman rho between `x` and
/// `y` (`isotonic.py:76-77`). On strictly DECREASING data the rho is `-1`, so
/// the fit goes decreasing and `increasing_` is `False`; on the increasing
/// scatter `[1,3,2,4]` rho > 0 so `increasing_` is `True`.
///
/// Oracle:
/// ```text
/// python3 -c "import numpy as np; from sklearn.isotonic import IsotonicRegression; \
///   m=IsotonicRegression(increasing='auto',out_of_bounds='clip').fit(np.array([1.,2.,3.,4.]).reshape(-1,1), np.array([4.,3.,2.,1.])); \
///   print(m.predict(np.array([1.,2.,3.,4.]).reshape(-1,1)).tolist(), m.increasing_); \
///   m2=IsotonicRegression(increasing='auto').fit(np.array([1.,2.,3.,4.]).reshape(-1,1), np.array([1.,3.,2.,4.])); \
///   print(m2.increasing_)"
///   # -> [4.0, 3.0, 2.0, 1.0] False
///   # -> True
/// ```
/// (The Spearman-CI warning sklearn emits is advisory and does not change the
/// returned direction — see `check_increasing`'s doc.)
///
/// Tracking: #567
#[test]
fn isotonic_increasing_auto() {
    // Decreasing data: X=[1,2,3,4], y=[4,3,2,1] -> resolves decreasing.
    let x = Array2::from_shape_vec((4, 1), vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    let y_dec = array![4.0_f64, 3.0, 2.0, 1.0];

    let fitted_dec = IsotonicRegression::<f64>::new()
        .with_increasing_auto()
        .fit(&x, &y_dec)
        .unwrap();

    // increasing_ == False (oracle).
    assert!(
        !fitted_dec.increasing(),
        "increasing='auto' on decreasing data: sklearn increasing_=False \
         (check_increasing, isotonic.py:306-307), ferrolearn returned {}",
        fitted_dec.increasing()
    );

    // predict([1,2,3,4]) == [4,3,2,1] (oracle).
    let preds = fitted_dec.predict(&x).unwrap();
    let oracle = [4.0_f64, 3.0, 2.0, 1.0];
    for (i, &exp) in oracle.iter().enumerate() {
        assert!(
            (preds[i] - exp).abs() < 1e-9,
            "increasing='auto' decreasing predict({}): sklearn returns {exp}, \
             ferrolearn returned {}",
            [1.0, 2.0, 3.0, 4.0][i],
            preds[i]
        );
    }

    // Increasing scatter: X=[1,2,3,4], y=[1,3,2,4] -> resolves increasing.
    let y_inc = array![1.0_f64, 3.0, 2.0, 4.0];
    let fitted_inc = IsotonicRegression::<f64>::new()
        .with_increasing_auto()
        .fit(&x, &y_inc)
        .unwrap();
    assert!(
        fitted_inc.increasing(),
        "increasing='auto' on increasing data: sklearn increasing_=True \
         (check_increasing, isotonic.py:306-307), ferrolearn returned {}",
        fitted_inc.increasing()
    );
}

/// Parity: the fitted `X_min_`/`X_max_`/`X_thresholds_`/`y_thresholds_`/
/// `increasing_` attributes are exposed and match sklearn (#570).
///
/// sklearn sets these in `fit`/`_build_y` (`sklearn/isotonic.py:331` `X_min_`/
/// `X_max_`; `:393` `X_thresholds_`/`y_thresholds_`; `:307-309` `increasing_`),
/// after the `trim_duplicates` interior-plateau trim (`:333-341`).
///
/// Oracle:
/// ```text
/// python3 -c "import numpy as np; from sklearn.isotonic import IsotonicRegression; \
///   m=IsotonicRegression().fit(np.array([1.,2.,3.,4.]).reshape(-1,1), np.array([1.,3.,2.,4.])); \
///   print(m.X_min_, m.X_max_, m.X_thresholds_.tolist(), m.y_thresholds_.tolist(), m.increasing_)"
///   # -> 1.0 4.0 [1.0, 2.0, 3.0, 4.0] [1.0, 2.5, 2.5, 4.0] True
/// ```
///
/// Tracking: #570
#[test]
fn isotonic_fitted_attributes() {
    let x = Array2::from_shape_vec((4, 1), vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    let y = array![1.0_f64, 3.0, 2.0, 4.0];

    let fitted = IsotonicRegression::<f64>::new().fit(&x, &y).unwrap();

    // sklearn 1.5.2 oracle (recorded above).
    assert!(
        (fitted.x_min() - 1.0).abs() < 1e-12,
        "X_min_: sklearn=1.0 (isotonic.py:331), ferrolearn={}",
        fitted.x_min()
    );
    assert!(
        (fitted.x_max() - 4.0).abs() < 1e-12,
        "X_max_: sklearn=4.0 (isotonic.py:331), ferrolearn={}",
        fitted.x_max()
    );

    let x_thr_oracle = [1.0_f64, 2.0, 3.0, 4.0];
    let y_thr_oracle = [1.0_f64, 2.5, 2.5, 4.0];
    assert_eq!(
        fitted.x_thresholds().len(),
        x_thr_oracle.len(),
        "X_thresholds_ length: sklearn={} (isotonic.py:393), ferrolearn={}",
        x_thr_oracle.len(),
        fitted.x_thresholds().len()
    );
    for (i, (&xt, &xo)) in fitted.x_thresholds().iter().zip(&x_thr_oracle).enumerate() {
        assert!(
            (xt - xo).abs() < 1e-12,
            "X_thresholds_[{i}]: sklearn={xo} (isotonic.py:393), ferrolearn={xt}"
        );
    }
    for (i, (&yt, &yo)) in fitted.y_thresholds().iter().zip(&y_thr_oracle).enumerate() {
        assert!(
            (yt - yo).abs() < 1e-12,
            "y_thresholds_[{i}]: sklearn={yo} (isotonic.py:393), ferrolearn={yt}"
        );
    }
    assert!(
        fitted.increasing(),
        "increasing_: sklearn=True (isotonic.py:307-309), ferrolearn={}",
        fitted.increasing()
    );
}

/// Parity: the free `check_increasing(x, y)` function (#571).
///
/// sklearn's `check_increasing` returns `rho >= 0` for the Spearman rho between
/// `x` and `y` (`sklearn/isotonic.py:32-98`, esp. `:76-77`).
///
/// Oracle:
/// ```text
/// python3 -c "from sklearn.isotonic import check_increasing; \
///   print(check_increasing([1,2,3,4],[1,2,3,4]), check_increasing([1,2,3,4],[4,3,2,1]))"
///   # -> True False
/// ```
///
/// Tracking: #571
#[test]
fn isotonic_free_check_increasing() {
    assert!(
        check_increasing(&[1.0_f64, 2.0, 3.0, 4.0], &[1.0_f64, 2.0, 3.0, 4.0]),
        "check_increasing([1,2,3,4],[1,2,3,4]): sklearn=True (isotonic.py:76-77)"
    );
    assert!(
        !check_increasing(&[1.0_f64, 2.0, 3.0, 4.0], &[4.0_f64, 3.0, 2.0, 1.0]),
        "check_increasing([1,2,3,4],[4,3,2,1]): sklearn=False (isotonic.py:76-77)"
    );
}

/// Parity: the free `isotonic_regression(y, ...)` PAVA function (#571).
///
/// sklearn's `isotonic_regression` runs the contiguous-sequence PAVA on `y`
/// (`sklearn/isotonic.py:111-171`).
///
/// Oracle:
/// ```text
/// python3 -c "from sklearn.isotonic import isotonic_regression; \
///   print(isotonic_regression([5,3,1,2,8,10,7,9,6,4]).tolist())"
///   # -> [2.75, 2.75, 2.75, 2.75, 7.333333333333333, 7.333333333333333,
///   #     7.333333333333333, 7.333333333333333, 7.333333333333333, 7.333333333333333]
/// ```
///
/// Tracking: #571
#[test]
fn isotonic_free_isotonic_regression() {
    let y = [5.0_f64, 3.0, 1.0, 2.0, 8.0, 10.0, 7.0, 9.0, 6.0, 4.0];
    let out = isotonic_regression(&y, None, None, None, true);

    let third = 7.0_f64 + 1.0 / 3.0;
    let oracle = [
        2.75_f64, 2.75, 2.75, 2.75, third, third, third, third, third, third,
    ];
    assert_eq!(out.len(), oracle.len());
    for (i, (&got, &exp)) in out.iter().zip(&oracle).enumerate() {
        assert!(
            (got - exp).abs() < 1e-9,
            "isotonic_regression[{i}]: sklearn={exp} (isotonic.py:111-171), ferrolearn={got}"
        );
    }
}
