//! Divergence audit: `ferrolearn-preprocess` `TargetEncoder` /
//! `FittedTargetEncoder` vs scikit-learn 1.5.2
//! `sklearn/preprocessing/_target_encoder.py::class TargetEncoder` (`:18`).
//!
//! VERIFY-AND-DOCUMENT unit. EVERY expected value below is grounded in a LIVE
//! sklearn 1.5.2 oracle call (run from /tmp) — NEVER copied from the ferrolearn
//! side (R-CHAR-3).
//!
//! Oracle session (sklearn 1.5.2, run from /tmp):
//! ```text
//! # Probe 1 (REQ-1, REQ-2) — manual smooth=2.0 m-estimate + unseen -> target_mean_:
//! >>> e = TargetEncoder(smooth=2.0, target_type='continuous')\
//! ...       .fit([[0],[0],[1],[1]], [1,2,3,4])
//! >>> [c.tolist() for c in e.encodings_]     -> [[2.0, 3.0]]
//! >>> float(e.target_mean_)                  -> 2.5
//! >>> e.transform([[0],[0],[1],[1]]).ravel() -> [2.0, 2.0, 3.0, 3.0]
//! >>> e.transform([[5]]).ravel()             -> [2.5]                  (unseen)
//!
//! # Probe RICH (REQ-1) — multi-feature, smooth=1.0:
//! >>> X = [[0,1],[0,0],[1,1],[1,0]]; y = [10,20,30,40]
//! >>> e = TargetEncoder(smooth=1.0, target_type='continuous').fit(X, y)
//! >>> [c.tolist() for c in e.encodings_]
//!     -> [[18.333333333333332, 31.666666666666668],
//!         [28.333333333333332, 21.666666666666668]]
//! >>> float(e.target_mean_)                  -> 25.0
//! >>> e.transform(X).tolist()
//!     -> [[18.333333333333332, 21.666666666666668],
//!         [18.333333333333332, 28.333333333333332],
//!         [31.666666666666668, 21.666666666666668],
//!         [31.666666666666668, 28.333333333333332]]
//!
//! # Probe BIN (REQ-1, target_type='auto'->binary, mean of 0/1 == P(y=1)):
//! >>> X = [[0],[0],[0],[1],[1],[1]]; yb = [0,0,1,1,1,0]
//! >>> e = TargetEncoder(smooth=2.0).fit(X, yb)   # auto -> binary, LabelEncode
//! >>> e.target_type_                         -> 'binary'
//! >>> [c.tolist() for c in e.encodings_]      -> [[0.4, 0.6]]
//! >>> float(e.target_mean_)                   -> 0.5
//! >>> e.transform(X).ravel()                  -> [0.4,0.4,0.4,0.6,0.6,0.6]
//!
//! # Probe C1 (REQ-1, count=1 category m-estimate edge, smooth=3.0):
//! >>> X = [[0],[1],[1],[1]]; y = [7,1,2,3]
//! >>> e = TargetEncoder(smooth=3.0, target_type='continuous').fit(X, y)
//! >>> [c.tolist() for c in e.encodings_]      -> [[4.1875, 2.625]]
//! >>> float(e.target_mean_)                   -> 3.25
//!
//! # Probe MEAN (HUNT) — global mean summation order, y = [1e16, 1.0 x100]:
//! >>> import numpy as np
//! >>> y = np.array([1e16] + [1.0]*100); X = np.zeros((101,1), dtype=int)
//! >>> e = TargetEncoder(smooth=1.0, target_type='continuous').fit(X, y)
//! >>> float(e.target_mean_)  -> 99009900990099.84   (== float(np.mean(y)), pairwise)
//! >>>   naive left-fold sum/n -> 99009900990099.02   (ferrolearn), diff ~0.83
//! ```

use ferrolearn_core::error::FerroError;
use ferrolearn_core::traits::{Fit, Transform};
use ferrolearn_preprocess::TargetEncoder;
use ndarray::{Array1, Array2, array};

// ---------------------------------------------------------------------------
// GREEN GUARDS (verify-and-document; oracle-grounded, R-CHAR-3)
// ---------------------------------------------------------------------------

/// GREEN guard — REQ-1/REQ-2. Probe 1 manual `smooth=2.0` m-estimate value
/// match: sklearn `encodings_=[[2.0,3.0]]`, `target_mean_=2.5`,
/// `transform([[0],[0],[1],[1]])=[2,2,3,3]`, unseen -> 2.5
/// (`_target_encoder.py:289` m-estimate, `:383` target_mean_, `:324-345` transform).
#[test]
fn green_req1_probe1_smooth2_value_match() {
    // Oracle (Probe 1, sklearn 1.5.2): encodings_=[[2.0,3.0]], target_mean_=2.5.
    const SK_TARGET_MEAN: f64 = 2.5;
    const SK_CAT0: f64 = 2.0;
    const SK_CAT1: f64 = 3.0;
    let x = array![[0usize], [0], [1], [1]];
    let y: Array1<f64> = array![1.0, 2.0, 3.0, 4.0];

    let fitted = TargetEncoder::<f64>::new(2.0).fit(&x, &y).unwrap();

    assert!((fitted.global_mean() - SK_TARGET_MEAN).abs() < 1e-12);
    let m0 = &fitted.category_maps()[0];
    assert!((m0[&0] - SK_CAT0).abs() < 1e-12);
    assert!((m0[&1] - SK_CAT1).abs() < 1e-12);

    let out = fitted.transform(&x).unwrap();
    // sklearn transform -> [2,2,3,3]
    assert!((out[[0, 0]] - SK_CAT0).abs() < 1e-12);
    assert!((out[[1, 0]] - SK_CAT0).abs() < 1e-12);
    assert!((out[[2, 0]] - SK_CAT1).abs() < 1e-12);
    assert!((out[[3, 0]] - SK_CAT1).abs() < 1e-12);
}

/// GREEN guard — REQ-2. Probe 1 unseen category at transform -> `target_mean_`
/// (`_target_encoder.py:324-345`, handle_unknown="ignore"). Oracle: 2.5.
#[test]
fn green_req2_probe1_unseen_to_target_mean() {
    const SK_TARGET_MEAN: f64 = 2.5; // oracle e.transform([[5]]) -> [2.5]
    let x = array![[0usize], [0], [1], [1]];
    let y: Array1<f64> = array![1.0, 2.0, 3.0, 4.0];
    let fitted = TargetEncoder::<f64>::new(2.0).fit(&x, &y).unwrap();
    let out = fitted.transform(&array![[5usize]]).unwrap();
    assert!((out[[0, 0]] - SK_TARGET_MEAN).abs() < 1e-12);
}

/// GREEN guard — REQ-1. Richer multi-feature fixture, smooth=1.0. Exercises the
/// per-category m-estimate AND the global-mean fallback on BOTH columns.
/// Oracle (Probe RICH): target_mean_=25.0, transform per cell as below.
#[test]
fn green_req1_rich_multifeature_value_match() {
    // sklearn 1.5.2 live oracle (Probe RICH):
    const SK_TARGET_MEAN: f64 = 25.0;
    // encodings_ col0: cat0->18.333..., cat1->31.666...
    const SK_C0_CAT0: f64 = 18.333_333_333_333_332;
    const SK_C0_CAT1: f64 = 31.666_666_666_666_668;
    // encodings_ col1: cat0->28.333..., cat1->21.666...
    const SK_C1_CAT0: f64 = 28.333_333_333_333_332;
    const SK_C1_CAT1: f64 = 21.666_666_666_666_668;
    // transform(X) rows for X = [[0,1],[0,0],[1,1],[1,0]]
    let sk_transform = [
        [SK_C0_CAT0, SK_C1_CAT1],
        [SK_C0_CAT0, SK_C1_CAT0],
        [SK_C0_CAT1, SK_C1_CAT1],
        [SK_C0_CAT1, SK_C1_CAT0],
    ];

    let x = array![[0usize, 1], [0, 0], [1, 1], [1, 0]];
    let y: Array1<f64> = array![10.0, 20.0, 30.0, 40.0];
    let fitted = TargetEncoder::<f64>::new(1.0).fit(&x, &y).unwrap();

    assert!((fitted.global_mean() - SK_TARGET_MEAN).abs() < 1e-12);
    let maps = fitted.category_maps();
    assert!((maps[0][&0] - SK_C0_CAT0).abs() < 1e-12);
    assert!((maps[0][&1] - SK_C0_CAT1).abs() < 1e-12);
    assert!((maps[1][&0] - SK_C1_CAT0).abs() < 1e-12);
    assert!((maps[1][&1] - SK_C1_CAT1).abs() < 1e-12);

    let out = fitted.transform(&x).unwrap();
    for i in 0..4 {
        for j in 0..2 {
            assert!(
                (out[[i, j]] - sk_transform[i][j]).abs() < 1e-12,
                "cell [{i},{j}] ferro={} sk={}",
                out[[i, j]],
                sk_transform[i][j]
            );
        }
    }
}

/// GREEN guard — REQ-1. Binary target (y in {0,1}): sklearn target_type='auto'
/// LabelEncodes 0/1 and mean(0/1)=P(y=1), so ferrolearn's continuous-mean formula
/// must reproduce sklearn's binary encodings (`_target_encoder.py:371-375,:383`).
/// Oracle (Probe BIN): encodings_=[[0.4,0.6]], target_mean_=0.5.
#[test]
fn green_req1_binary_target_matches_continuous_formula() {
    const SK_TARGET_MEAN: f64 = 0.5;
    const SK_CAT0: f64 = 0.4;
    const SK_CAT1: f64 = 0.6;
    let x = array![[0usize], [0], [0], [1], [1], [1]];
    let y: Array1<f64> = array![0.0, 0.0, 1.0, 1.0, 1.0, 0.0];
    let fitted = TargetEncoder::<f64>::new(2.0).fit(&x, &y).unwrap();

    assert!((fitted.global_mean() - SK_TARGET_MEAN).abs() < 1e-12);
    let m0 = &fitted.category_maps()[0];
    assert!((m0[&0] - SK_CAT0).abs() < 1e-12);
    assert!((m0[&1] - SK_CAT1).abs() < 1e-12);
}

/// GREEN guard — REQ-1. count=1 category m-estimate edge, smooth=3.0.
/// Oracle (Probe C1): encodings_=[[4.1875, 2.625]], target_mean_=3.25.
#[test]
fn green_req1_count_one_category_edge() {
    const SK_TARGET_MEAN: f64 = 3.25;
    const SK_CAT0: f64 = 4.1875; // count=1: (1*7 + 3*3.25)/(1+3)
    const SK_CAT1: f64 = 2.625; // count=3: (3*2 + 3*3.25)/(3+3)
    let x = array![[0usize], [1], [1], [1]];
    let y: Array1<f64> = array![7.0, 1.0, 2.0, 3.0];
    let fitted = TargetEncoder::<f64>::new(3.0).fit(&x, &y).unwrap();

    assert!((fitted.global_mean() - SK_TARGET_MEAN).abs() < 1e-12);
    let m0 = &fitted.category_maps()[0];
    assert!((m0[&0] - SK_CAT0).abs() < 1e-12);
    assert!((m0[&1] - SK_CAT1).abs() < 1e-12);
}

/// GREEN guard — REQ-3. Error contracts: 0-row fit, y-length mismatch fit,
/// transform ncols mismatch, negative smooth. sklearn rejects smooth<0 via
/// `Interval(Real, 0, None, closed="left")` (`_target_encoder.py:189`).
#[test]
fn green_req3_error_contracts() {
    // 0-row fit -> Err(InsufficientSamples)
    let x0: Array2<usize> = Array2::zeros((0, 2));
    let y0: Array1<f64> = Array1::zeros(0);
    assert!(matches!(
        TargetEncoder::<f64>::new(1.0).fit(&x0, &y0),
        Err(FerroError::InsufficientSamples { .. })
    ));

    // y-length mismatch -> Err(ShapeMismatch)
    let xm = array![[0usize], [1]];
    let ym: Array1<f64> = array![1.0];
    assert!(matches!(
        TargetEncoder::<f64>::new(1.0).fit(&xm, &ym),
        Err(FerroError::ShapeMismatch { .. })
    ));

    // transform ncols mismatch -> Err(ShapeMismatch)
    let xt = array![[0usize, 1], [1, 0]];
    let yt: Array1<f64> = array![1.0, 2.0];
    let fitted = TargetEncoder::<f64>::new(1.0).fit(&xt, &yt).unwrap();
    assert!(matches!(
        fitted.transform(&array![[0usize]]),
        Err(FerroError::ShapeMismatch { .. })
    ));

    // negative smooth -> Err(InvalidParameter)  (sklearn Interval closed="left" at 0)
    let xn = array![[0usize]];
    let yn: Array1<f64> = array![1.0];
    assert!(matches!(
        TargetEncoder::<f64>::new(-1.0).fit(&xn, &yn),
        Err(FerroError::InvalidParameter { .. })
    ));
}

// ---------------------------------------------------------------------------
// PINNED DIVERGENCE (genuine, on the implemented manual-smooth path)
// ---------------------------------------------------------------------------

/// Divergence: ferrolearn's `TargetEncoder::fit` computes
/// `global_mean = y.iter().fold(0, |a,v| a+v) / n` — a sequential left-fold —
/// at `ferrolearn-preprocess/src/target_encoder.rs:148-149`:
///   `let global_mean = y.iter().copied().fold(F::zero(), |a, v| a + v)`
///   `    / F::from(n_samples)...;`
/// sklearn sets `target_mean_ = np.mean(y, axis=0)` at
/// `sklearn/preprocessing/_target_encoder.py:383`, and `np.mean` uses NumPy
/// pairwise summation, which differs from a naive left-fold for inputs that mix
/// magnitudes.
///
/// Live oracle (sklearn 1.5.2, run from /tmp), y = [1e16, 1.0 x100]:
///   float(e.target_mean_) -> 99009900990099.84   (== float(np.mean(y)))
///   ferrolearn left-fold sum/n -> 99009900990099.02   (diff ~0.83)
///
/// This breaks the REQ-1 "value match within ULP tolerance" SHIPPED claim for
/// mixed-magnitude targets: the divergence is ~0.83 absolute, ~12 ULP.
/// Tracking: #1261
#[test]
#[ignore = "divergence: global_mean naive left-fold vs np.mean pairwise summation; tracking #1261"]
fn divergence_global_mean_summation_order() {
    // Oracle: sklearn target_mean_ for y = [1e16, 1.0 x100] (== float(np.mean(y))).
    const SK_TARGET_MEAN: f64 = 99_009_900_990_099.84;

    let n = 101usize;
    let x: Array2<usize> = Array2::zeros((n, 1));
    let mut y: Array1<f64> = Array1::ones(n);
    y[0] = 1e16;

    let fitted = TargetEncoder::<f64>::new(1.0).fit(&x, &y).unwrap();

    // sklearn's target_mean_ (pairwise) is the contract; ferrolearn's left-fold
    // diverges by ~0.83. A 1e-6 tolerance is generous and still fails.
    assert!(
        (fitted.global_mean() - SK_TARGET_MEAN).abs() < 1e-6,
        "ferrolearn global_mean()={} diverges from sklearn target_mean_={} (np.mean pairwise)",
        fitted.global_mean(),
        SK_TARGET_MEAN
    );
}
