//! Divergence + green-guard suite for `IterativeImputer` against scikit-learn
//! 1.5.2 `class IterativeImputer` (`sklearn/impute/_iterative.py`).
//!
//! EXPERIMENTAL upstream — imported in oracles via
//! `from sklearn.experimental import enable_iterative_imputer`.
//!
//! HEADLINE FIXABLE divergence DIV-1 (REQ-3): `max_iter == 0` handling.
//! sklearn `fit_transform` (`_iterative.py:750-752`) treats `max_iter == 0` as
//! VALID -> sets `n_iter_ = 0` and returns the INITIAL imputation
//! (`SimpleImputer(strategy=initial_strategy)` result, `_initial_imputation`
//! `:743`), with NO regression rounds. ferrolearn `fit`
//! (`iterative_imputer.rs:400-405`) REJECTS `max_iter == 0` with
//! `InvalidParameter`. Parity IS achievable: sklearn's max_iter=0 output equals
//! the per-column mean/median initial fill, which ferrolearn's `initial_fill`
//! already computes (matches SimpleImputer). Tracking #1403, blocker #1404.
//!
//! All sklearn expected values are hard-coded from the LIVE sklearn 1.5.2 oracle
//! (run from /tmp) — see each test's doc comment for the probe. The iterated
//! imputed VALUES for `max_iter >= 1` are a documented CARVE-OUT (REQ-4:
//! ferrolearn Ridge(alpha=1) != sklearn default BayesianRidge, plus order/tol/clip
//! differences) and are NOT asserted here.

use ferrolearn_core::traits::{Fit, FitTransform, Transform};
use ferrolearn_preprocess::iterative_imputer::{
    FittedIterativeImputer, InitialStrategy, IterativeImputer,
};
use ndarray::{Array1, Array2, array};

/// Per-column mean of the non-NaN values for the DIV-1 fixture
/// `X = [[1,2],[NaN,3],[5,NaN],[7,8]]`.
///
/// sklearn `SimpleImputer(strategy='mean').statistics_` (live, /tmp):
/// col0 observed {1,5,7} -> 13/3; col1 observed {2,3,8} -> 13/3.
/// Both columns coincide at 4.333333333333333.
const SK_MEAN_FILL_DIV1: f64 = 13.0 / 3.0;

/// Build the fitted handle for the DIV-1 fixture, mapping any error into a
/// descriptive panic (no bare `.unwrap()`/`.expect()` in the divergence path).
fn fit_div1_fixture(
    imputer: &IterativeImputer<f64>,
    x: &Array2<f64>,
) -> Result<FittedIterativeImputer<f64>, ferrolearn_core::error::FerroError> {
    imputer.fit(x, &())
}

// ===========================================================================
// DIV-1 (HEADLINE, FIXABLE, REQ-3) — max_iter == 0 returns the initial fill
// ===========================================================================

/// Divergence DIV-1: `IterativeImputer::fit` diverges from
/// `sklearn/impute/_iterative.py:750-752`
/// (`if self.max_iter == 0 ... self.n_iter_ = 0; return ...`) for `max_iter == 0`.
///
/// Oracle (sklearn 1.5.2, EXPERIMENTAL gate, run from /tmp):
///   X = [[1,2],[nan,3],[5,nan],[7,8]]
///   IterativeImputer(max_iter=0).fit(X) -> n_iter_ = 0
///   transform(X) = [[1.0, 2.0],
///                   [4.333333333333333, 3.0],
///                   [5.0, 4.333333333333333],
///                   [7.0, 8.0]]
/// i.e. the per-column mean initial fill, observed values preserved.
///
/// sklearn ACCEPTS max_iter=0 and returns the initial fill.
/// ferrolearn REJECTS max_iter=0 with `InvalidParameter`
/// (`iterative_imputer.rs:400-405`), so `fit` returns `Err`.
///
/// Tracking #1403, blocker #1404.
#[test]
fn divergence_max_iter_zero_returns_initial_fill() {
    let imputer = IterativeImputer::<f64>::new(0, 1e-3, InitialStrategy::Mean);
    let x = array![[1.0, 2.0], [f64::NAN, 3.0], [5.0, f64::NAN], [7.0, 8.0]];

    // sklearn ACCEPTS max_iter=0; ferrolearn currently returns Err here.
    let fitted = match fit_div1_fixture(&imputer, &x) {
        Ok(f) => f,
        Err(e) => {
            panic!(
                "DIV-1: ferrolearn rejected max_iter=0 with {e:?}; sklearn \
                 (_iterative.py:750-752) accepts it and returns the initial fill"
            )
        }
    };

    // sklearn: n_iter_ == 0 for max_iter == 0.
    assert_eq!(
        fitted.n_iter(),
        0,
        "DIV-1: sklearn sets n_iter_=0 for max_iter=0 (_iterative.py:751)"
    );

    let out = match fitted.transform(&x) {
        Ok(o) => o,
        Err(e) => panic!("DIV-1: transform after max_iter=0 fit failed: {e:?}"),
    };

    // sklearn output == per-column mean initial fill, observed values preserved.
    let expected = array![
        [1.0, 2.0],
        [SK_MEAN_FILL_DIV1, 3.0],
        [5.0, SK_MEAN_FILL_DIV1],
        [7.0, 8.0]
    ];
    for (got, want) in out.iter().zip(expected.iter()) {
        assert!(
            (got - want).abs() < 1e-9,
            "DIV-1: max_iter=0 output {got} != sklearn initial fill {want}"
        );
    }
}

// ===========================================================================
// GREEN GUARDS — the SHIPPED structural / contract claims (must PASS)
// ===========================================================================

/// GREEN GUARD (REQ-1): the Mean initial fill equals the per-column mean of the
/// non-NaN values, matching `SimpleImputer(strategy='mean').statistics_`.
///
/// Oracle (sklearn 1.5.2, /tmp): X = [[1,2],[3,nan],[nan,6]],
/// `SimpleImputer(strategy='mean').statistics_ = [2.0, 4.0]`.
#[test]
fn green_initial_fill_mean_matches_simpleimputer() {
    let imputer = IterativeImputer::<f64>::new(10, 1e-3, InitialStrategy::Mean);
    let x = array![[1.0, 2.0], [3.0, f64::NAN], [f64::NAN, 6.0]];
    let fitted = imputer.fit(&x, &()).expect("fit should succeed");
    let fill = fitted.initial_fill();
    let sk_mean: Array1<f64> = array![2.0, 4.0]; // SimpleImputer(mean).statistics_
    assert_eq!(fill.len(), 2);
    for (got, want) in fill.iter().zip(sk_mean.iter()) {
        assert!(
            (got - want).abs() < 1e-12,
            "Mean initial fill {got} != SimpleImputer mean {want}"
        );
    }
}

/// GREEN GUARD (REQ-1): the Median initial fill equals the per-column median of
/// the non-NaN values, matching `SimpleImputer(strategy='median').statistics_`.
///
/// Oracle (sklearn 1.5.2, /tmp): X = [[1,2],[3,nan],[nan,6]],
/// `SimpleImputer(strategy='median').statistics_ = [2.0, 4.0]`.
#[test]
fn green_initial_fill_median_matches_simpleimputer() {
    let imputer = IterativeImputer::<f64>::new(10, 1e-3, InitialStrategy::Median);
    let x = array![[1.0, 2.0], [3.0, f64::NAN], [f64::NAN, 6.0]];
    let fitted = imputer.fit(&x, &()).expect("fit should succeed");
    let fill = fitted.initial_fill();
    let sk_median: Array1<f64> = array![2.0, 4.0]; // SimpleImputer(median).statistics_
    assert_eq!(fill.len(), 2);
    for (got, want) in fill.iter().zip(sk_median.iter()) {
        assert!(
            (got - want).abs() < 1e-12,
            "Median initial fill {got} != SimpleImputer median {want}"
        );
    }
}

/// GREEN GUARD (REQ-1): originally-non-missing entries are preserved after
/// `fit_transform` (sklearn `_assign_where(Xt, X, cond=~mask)`,
/// `_iterative.py:829`). Structural property — no sklearn value needed.
#[test]
fn green_non_missing_preserved() {
    let imputer = IterativeImputer::<f64>::new(10, 1e-3, InitialStrategy::Mean);
    let x = array![[1.0, 2.0], [3.0, f64::NAN], [f64::NAN, 6.0], [5.0, 8.0]];
    let out = imputer
        .fit_transform(&x)
        .expect("fit_transform should succeed");
    for ((i, j), &v) in x.indexed_iter() {
        if !v.is_nan() {
            assert!(
                (out[[i, j]] - v).abs() < 1e-12,
                "observed entry [{i},{j}] changed: input {v} -> output {}",
                out[[i, j]]
            );
        }
    }
}

/// GREEN GUARD (REQ-1): output shape is `(n_samples, n_features)` and no NaN
/// remains. Structural property.
#[test]
fn green_output_shape_and_no_nan() {
    let imputer = IterativeImputer::<f64>::new(10, 1e-3, InitialStrategy::Mean);
    let x = array![[1.0, 2.0], [3.0, f64::NAN], [f64::NAN, 6.0]];
    let out = imputer
        .fit_transform(&x)
        .expect("fit_transform should succeed");
    assert_eq!(
        out.dim(),
        (3, 2),
        "output shape must be (n_samples, n_features)"
    );
    for v in &out {
        assert!(!v.is_nan(), "output must contain no NaN");
    }
}

/// GREEN GUARD (REQ-2): the fit->transform path is deterministic (no RNG).
/// Two independent runs produce bit-identical output.
#[test]
fn green_determinism() {
    let imputer = IterativeImputer::<f64>::new(10, 1e-3, InitialStrategy::Mean);
    let x = array![[1.0, 2.0], [3.0, f64::NAN], [f64::NAN, 6.0], [5.0, 8.0]];
    let out1 = imputer.fit_transform(&x).expect("run 1");
    let out2 = imputer.fit_transform(&x).expect("run 2");
    for (a, b) in out1.iter().zip(out2.iter()) {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "non-deterministic output: {a} vs {b}"
        );
    }
}

/// GREEN GUARD (REQ-2): for `max_iter >= 1` on a fixture with missing values,
/// `n_iter()` is in `[1, max_iter]` (bounded termination, sklearn
/// `for self.n_iter_ in range(1, self.max_iter + 1)`, `_iterative.py:781`).
#[test]
fn green_termination_bounded() {
    let max_iter = 10usize;
    let imputer = IterativeImputer::<f64>::new(max_iter, 1e-3, InitialStrategy::Mean);
    let x = array![[1.0, 2.0], [3.0, f64::NAN], [f64::NAN, 6.0]];
    let fitted = imputer.fit(&x, &()).expect("fit should succeed");
    let n = fitted.n_iter();
    assert!(
        n >= 1,
        "n_iter() = {n} must be >= 1 for max_iter >= 1 with missing values"
    );
    assert!(
        n <= max_iter,
        "n_iter() = {n} must be <= max_iter = {max_iter}"
    );
}

/// GREEN GUARD (REQ-3): `n_samples == 0` -> `fit` returns `Err`
/// (sklearn raises a validation `ValueError`).
#[test]
fn green_error_zero_rows() {
    let imputer = IterativeImputer::<f64>::new(10, 1e-3, InitialStrategy::Mean);
    let x: Array2<f64> = Array2::zeros((0, 3));
    assert!(imputer.fit(&x, &()).is_err(), "zero rows must error");
}

/// GREEN GUARD (REQ-3): `transform` with a mismatched column count -> `Err`.
#[test]
fn green_error_transform_ncols_mismatch() {
    let imputer = IterativeImputer::<f64>::new(10, 1e-3, InitialStrategy::Mean);
    let x_train = array![[1.0, 2.0], [3.0, 4.0]];
    let fitted = imputer.fit(&x_train, &()).expect("fit should succeed");
    let x_bad = array![[1.0, 2.0, 3.0]];
    assert!(
        fitted.transform(&x_bad).is_err(),
        "ncols mismatch must error"
    );
}

/// GREEN GUARD (REQ-3): calling `transform` on the UNFITTED imputer -> `Err`
/// (mirrors sklearn `check_is_fitted`).
#[test]
fn green_error_unfitted_transform() {
    let imputer = IterativeImputer::<f64>::new(10, 1e-3, InitialStrategy::Mean);
    let x = array![[1.0, 2.0]];
    assert!(
        imputer.transform(&x).is_err(),
        "unfitted transform must error"
    );
}

/// GREEN GUARD (REQ-1): f32 path produces finite (non-NaN) output.
#[test]
fn green_f32_finite_output() {
    let imputer = IterativeImputer::<f32>::new(10, 1e-3, InitialStrategy::Mean);
    let x: Array2<f32> = array![[1.0f32, 2.0], [3.0, f32::NAN], [f32::NAN, 6.0]];
    let out = imputer
        .fit_transform(&x)
        .expect("f32 fit_transform should succeed");
    for v in &out {
        assert!(v.is_finite(), "f32 output must be finite, got {v}");
    }
}

/// GREEN GUARD: single-feature input with missing values imputes to the column
/// mean (both ferrolearn and sklearn agree). sklearn returns the initial fill
/// with `n_iter_ = 0` for a single feature (`_iterative.py:755-757`); ferrolearn's
/// round-robin loop skips it (`n_predictors == 0`), leaving the initial mean fill.
///
/// Oracle (sklearn 1.5.2, /tmp): X = [[1],[nan],[5],[7]],
/// IterativeImputer(max_iter=10).fit_transform -> [[1],[4.333..],[5],[7]],
/// where 4.333.. = mean of {1,5,7} = 13/3.
#[test]
fn green_single_feature_imputes_to_column_mean() {
    let imputer = IterativeImputer::<f64>::new(10, 1e-3, InitialStrategy::Mean);
    let x = array![[1.0], [f64::NAN], [5.0], [7.0]];
    let out = imputer
        .fit_transform(&x)
        .expect("single-feature fit_transform");
    let expected = array![[1.0], [SK_MEAN_FILL_DIV1], [5.0], [7.0]];
    for (got, want) in out.iter().zip(expected.iter()) {
        assert!(
            (got - want).abs() < 1e-9,
            "single-feature output {got} != column mean fill {want}"
        );
    }
}
