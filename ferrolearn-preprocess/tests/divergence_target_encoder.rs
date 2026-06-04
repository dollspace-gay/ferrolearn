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

/// GREEN guard — `pairwise_sum` branch coverage. NumPy is the oracle:
/// `float(np.mean(y))` for n=5 ill-conditioned mixed-magnitude y (seed 1).
/// sklearn `target_mean_ = np.mean(y, axis=0)` (`_target_encoder.py:383`).
/// Asserts ferrolearn `global_mean()` bit-matches `np.mean` (exact f64 equality).
#[test]
fn green_pairwise_n_lt_8() {
    const SK_MEAN: f64 = 22674627783.60881f64; // float(np.mean(y)), sklearn 1.5.2 / numpy
    let y: Array1<f64> = array![
        23643249400.51343f64,
        90.09273926518706f64,
        -0.007116807745607325f64,
        89729889427.44878f64,
        -0.003763370959790291f64
    ];
    let n = y.len();
    let x: Array2<usize> = Array2::zeros((n, 1));
    let fitted = TargetEncoder::<f64>::new(0.0).fit(&x, &y).unwrap();
    assert_eq!(
        fitted.global_mean().to_bits(),
        SK_MEAN.to_bits(),
        "n={n}: ferro global_mean()={} (bits {:#x}) != np.mean={} (bits {:#x})",
        fitted.global_mean(),
        fitted.global_mean().to_bits(),
        SK_MEAN,
        SK_MEAN.to_bits()
    );
}

/// GREEN guard — `pairwise_sum` branch coverage. NumPy is the oracle:
/// `float(np.mean(y))` for n=8 ill-conditioned mixed-magnitude y (seed 1).
/// sklearn `target_mean_ = np.mean(y, axis=0)` (`_target_encoder.py:383`).
/// Asserts ferrolearn `global_mean()` bit-matches `np.mean` (exact f64 equality).
#[test]
fn green_pairwise_n_eq_8() {
    const SK_MEAN: f64 = 34691868168.815254f64; // float(np.mean(y)), sklearn 1.5.2 / numpy
    let y: Array1<f64> = array![
        236432.49400513433f64,
        90092.73926518706f64,
        -7.116807745607325e-07f64,
        8.972988942744878e-09f64,
        -376337095979.0291f64,
        -1533471020.548487f64,
        655405187640.8835f64,
        -1816.0172726167746f64
    ];
    let n = y.len();
    let x: Array2<usize> = Array2::zeros((n, 1));
    let fitted = TargetEncoder::<f64>::new(0.0).fit(&x, &y).unwrap();
    assert_eq!(
        fitted.global_mean().to_bits(),
        SK_MEAN.to_bits(),
        "n={n}: ferro global_mean()={} (bits {:#x}) != np.mean={} (bits {:#x})",
        fitted.global_mean(),
        fitted.global_mean().to_bits(),
        SK_MEAN,
        SK_MEAN.to_bits()
    );
}

/// GREEN guard — `pairwise_sum` branch coverage. NumPy is the oracle:
/// `float(np.mean(y))` for n=13 ill-conditioned mixed-magnitude y (seed 2).
/// sklearn `target_mean_ = np.mean(y, axis=0)` (`_target_encoder.py:383`).
/// Asserts ferrolearn `global_mean()` bit-matches `np.mean` (exact f64 equality).
#[test]
fn green_pairwise_n_in_8_128() {
    const SK_MEAN: f64 = -68440884286730.24f64; // float(np.mean(y)), sklearn 1.5.2 / numpy
    let y: Array1<f64> = array![
        -47677573.150136724f64,
        -40301771.317175336f64,
        62845148118856.06f64,
        -81.61681157298062f64,
        0.000200201051931308f64,
        4571210.536235892f64,
        -62419785326679.31f64,
        -889706745333863.6f64,
        -450061264187.9238f64,
        31486602.975118518f64,
        1.24531325560856f64,
        -6.998754733893278f64,
        -1.3473841839042567e-09f64
    ];
    let n = y.len();
    let x: Array2<usize> = Array2::zeros((n, 1));
    let fitted = TargetEncoder::<f64>::new(0.0).fit(&x, &y).unwrap();
    assert_eq!(
        fitted.global_mean().to_bits(),
        SK_MEAN.to_bits(),
        "n={n}: ferro global_mean()={} (bits {:#x}) != np.mean={} (bits {:#x})",
        fitted.global_mean(),
        fitted.global_mean().to_bits(),
        SK_MEAN,
        SK_MEAN.to_bits()
    );
}

/// GREEN guard — `pairwise_sum` branch coverage. NumPy is the oracle:
/// `float(np.mean(y))` for n=101 ill-conditioned mixed-magnitude y (seed 4).
/// sklearn `target_mean_ = np.mean(y, axis=0)` (`_target_encoder.py:383`).
/// Asserts ferrolearn `global_mean()` bit-matches `np.mean` (exact f64 equality).
#[test]
fn green_pairwise_n_101() {
    const SK_MEAN: f64 = 7075924318325.883f64; // float(np.mean(y)), sklearn 1.5.2 / numpy
    let y: Array1<f64> = array![
        8861122.111447353f64,
        2.2655105628723196e-07f64,
        9524.874114154083f64,
        -83832795.22087957f64,
        2.1471166399005928e-05f64,
        -0.024702683124545488f64,
        603802.4139716144f64,
        -6509443677.119431f64,
        7.432705483753128e-09f64,
        878.8280152699634f64,
        8044301.594319767f64,
        -0.4569295232158743f64,
        -1.3900744454117375e-08f64,
        577893435.0886588f64,
        9683059998.622427f64,
        -2.6054841469514248e-05f64,
        937865738632.4698f64,
        0.00858052775530839f64,
        -0.6446148284760378f64,
        0.02177032336890952f64,
        0.0004097294911294851f64,
        8.856073582583344f64,
        3313.1483393150684f64,
        -7332084890966.138f64,
        -4.264802429251091f64,
        -1276033208.7770276f64,
        0.045237869749170656f64,
        91716457221.84944f64,
        -30012520.03554904f64,
        -5524577.06443568f64,
        44174004.42289532f64,
        2.8234184793005634e-09f64,
        87821410.68480176f64,
        0.0016403160129409167f64,
        -46433321.9416823f64,
        8595.493906161339f64,
        -1.654943077504689e-09f64,
        3516017298670.5176f64,
        -47922179.049196154f64,
        -0.5660399471560098f64,
        3.851040894211213f64,
        54126.096618348754f64,
        -61842298948292.48f64,
        -80167974503.0368f64,
        -2.7636988019618157e-08f64,
        -0.6585403611074503f64,
        -5.572966802884256e-06f64,
        925014822.6527568f64,
        7684461032.635626f64,
        -235416.34761044162f64,
        4789464862700.879f64,
        -9160.63925637459f64,
        8301144.0248034485f64,
        743417.0242820448f64,
        6405356.070358876f64,
        -448.12348486965294f64,
        -0.24775237364186453f64,
        -3.03912562176339e-07f64,
        944879.3354225118f64,
        -1410802341545.3252f64,
        -0.0006145189360067249f64,
        911957860037452.5f64,
        806391.1767014762f64,
        -20997897947.973022f64,
        -385284160541.92883f64,
        0.6140958722700762f64,
        -7.85056820261437e-09f64,
        -2156221.7260676f64,
        75286763436655.02f64,
        8352848296.653064f64,
        -53304746.26766963f64,
        -8.425596989786902e-08f64,
        5.754549969164102e-05f64,
        2435561307.920473f64,
        -0.5249903647523255f64,
        1.178638842201285e-08f64,
        -85716355.58495669f64,
        40017464104562.68f64,
        5345091193211.271f64,
        0.000409295728880031f64,
        -82366125398084.19f64,
        289630046602507.25f64,
        -259328233.71047497f64,
        -1.3828680127838667e-06f64,
        -0.00032903500048296744f64,
        2681.7305271753876f64,
        136521870.1440136f64,
        23422951418.24442f64,
        -8.732407586442848e-06f64,
        -549716797977532.8f64,
        -8.561908784572198e-08f64,
        -23340430.03724011f64,
        -0.0037983012448637335f64,
        -775316114474.923f64,
        3351781697.8533087f64,
        -0.2528648741392381f64,
        6.59424129722419e-12f64,
        1949316820.056346f64,
        86975679220823.92f64,
        1.777299333167972f64,
        -0.008130348730986763f64
    ];
    let n = y.len();
    let x: Array2<usize> = Array2::zeros((n, 1));
    let fitted = TargetEncoder::<f64>::new(0.0).fit(&x, &y).unwrap();
    assert_eq!(
        fitted.global_mean().to_bits(),
        SK_MEAN.to_bits(),
        "n={n}: ferro global_mean()={} (bits {:#x}) != np.mean={} (bits {:#x})",
        fitted.global_mean(),
        fitted.global_mean().to_bits(),
        SK_MEAN,
        SK_MEAN.to_bits()
    );
}

/// GREEN guard — `pairwise_sum` branch coverage. NumPy is the oracle:
/// `float(np.mean(y))` for n=128 ill-conditioned mixed-magnitude y (seed 5).
/// sklearn `target_mean_ = np.mean(y, axis=0)` (`_target_encoder.py:383`).
/// Asserts ferrolearn `global_mean()` bit-matches `np.mean` (exact f64 equality).
#[test]
fn green_pairwise_n_eq_128() {
    const SK_MEAN: f64 = 9376590335716.943f64; // float(np.mean(y)), sklearn 1.5.2 / numpy
    let y: Array1<f64> = array![
        6100058474.907603f64,
        61588157.94729875f64,
        0.30651122084284f64,
        -4.283972398237168e-09f64,
        -892138595.2366872f64,
        -2.3326223842896354e-09f64,
        -18305358916.00027f64,
        -9.094496121951097e-06f64,
        -0.0902484578545664f64,
        0.000998352230130143f64,
        3047.382231759754f64,
        -5.309795966603521e-05f64,
        -0.0013010489554971594f64,
        9.483723865185107e-08f64,
        795355216217097.6f64,
        0.0006884620752174819f64,
        -2.1519067133044368e-05f64,
        -13953.962536514819f64,
        3.533787036621321e-06f64,
        -8783945740838.879f64,
        1111922338.4144683f64,
        -0.45709679093979694f64,
        7.593023466698443f64,
        -0.871571125375618f64,
        358363066.0427299f64,
        74.01770046550067f64,
        -0.005453629496781838f64,
        79089647.8828252f64,
        7.443909360486701e-06f64,
        -9629655646.595785f64,
        41.499113467435464f64,
        -99760063282634.27f64,
        6.727931107328945e-05f64,
        -1.2666589564869457f64,
        -593494327770.8704f64,
        -35011471.08487879f64,
        6.124306620540872e-09f64,
        -0.3670958251910197f64,
        -7.019228329328884e-09f64,
        39702398063.662155f64,
        -1.0291179847924758f64,
        0.000597878981926268f64,
        -528967085.3876571f64,
        -360430691363.4274f64,
        5.997590521099067e-07f64,
        141.3627784678262f64,
        0.012770002843141892f64,
        -527611743.208273f64,
        -97092743927237.34f64,
        8664.478004508674f64,
        -828348417.8263401f64,
        0.068985360294722f64,
        -26423.852217696654f64,
        902.0459623915989f64,
        -201149476.84238714f64,
        8.728841236840061e-08f64,
        0.00011231955106779412f64,
        -5.197293433944483e-08f64,
        482843340055625.56f64,
        3487.794297735256f64,
        368.4104247691966f64,
        -72.35127983326551f64,
        -5.562228816710906e-09f64,
        28188028296.26219f64,
        -785821007725.4271f64,
        3.844445492896597e-09f64,
        270773.5997814216f64,
        -24697.494882461557f64,
        5.97046691612211f64,
        -611.9490479859109f64,
        -2190821724045.5686f64,
        595867.772194042f64,
        -239.0492587133528f64,
        426515728.2489043f64,
        22503560.833062373f64,
        8820.019504852036f64,
        98335343398.03926f64,
        44735.250930159265f64,
        617687.6180693398f64,
        -694.2699537086543f64,
        425780.5251331368f64,
        695248.7605090678f64,
        -1975485040147.371f64,
        0.00010650007958708185f64,
        -0.041024496478275774f64,
        91704.5999598743f64,
        -0.36544431944656863f64,
        -19583094.612446826f64,
        -99816042177.82684f64,
        -1.5963093061373914e-05f64,
        2.628722122645801e-08f64,
        869922188373300.6f64,
        8473.531695154692f64,
        -345300373.3518043f64,
        9777220512.380384f64,
        -62465011953096.24f64,
        6.465040360720631e-09f64,
        -0.06854810258319109f64,
        -189802694277680.94f64,
        -85.30533763854449f64,
        7.160778623960347f64,
        0.006575961419530448f64,
        -7.204214029145675e-09f64,
        5.421295311416507e-09f64,
        -483717006167387.25f64,
        -1.6448599962327616e-10f64,
        1.064120118984977e-06f64,
        -0.0007874758976024857f64,
        0.00728751126202265f64,
        -442607361931.376f64,
        -0.010575335685125344f64,
        -8.854828065685027e-06f64,
        -9.945095871835242f64,
        -60968095.68060275f64,
        -31.664963176331874f64,
        0.0008562268420291474f64,
        7.794565770190948e-07f64,
        -3.899796670770716e-09f64,
        -90.48036283483741f64,
        3.339860877077614e-05f64,
        717509.8049197082f64,
        -32540498.8502602f64,
        5.873090555173121f64,
        -2020.2549382391232f64,
        1.881077647754148e-08f64,
        47493.39655804539f64,
        2.7144088342523354e-07f64,
        38.155154241339886f64
    ];
    let n = y.len();
    let x: Array2<usize> = Array2::zeros((n, 1));
    let fitted = TargetEncoder::<f64>::new(0.0).fit(&x, &y).unwrap();
    assert_eq!(
        fitted.global_mean().to_bits(),
        SK_MEAN.to_bits(),
        "n={n}: ferro global_mean()={} (bits {:#x}) != np.mean={} (bits {:#x})",
        fitted.global_mean(),
        fitted.global_mean().to_bits(),
        SK_MEAN,
        SK_MEAN.to_bits()
    );
}

/// GREEN guard — `pairwise_sum` branch coverage. NumPy is the oracle:
/// `float(np.mean(y))` for n=300 ill-conditioned mixed-magnitude y (seed 6).
/// sklearn `target_mean_ = np.mean(y, axis=0)` (`_target_encoder.py:383`).
/// Asserts ferrolearn `global_mean()` bit-matches `np.mean` (exact f64 equality).
#[test]
fn green_pairwise_n_gt_128_300() {
    const SK_MEAN: f64 = 2059875317792.7341f64; // float(np.mean(y)), sklearn 1.5.2 / numpy
    let y: Array1<f64> = array![
        76328.70294388638f64,
        -3.1345826037332314e-07f64,
        -2.618655204092435e-07f64,
        -2.510064688242353e-07f64,
        0.0009748899803729332f64,
        26551.25452142921f64,
        348.64786100898715f64,
        -34.00730892290833f64,
        359835322328044.6f64,
        -7540552.502256199f64,
        -89654153603.24496f64,
        7003.826936175741f64,
        -0.09822083889291307f64,
        9575351243.705935f64,
        65400605163.81072f64,
        57042191.86527428f64,
        -90383302.5994907f64,
        -5.851214675860506e-09f64,
        0.6997312621904237f64,
        -1.350103766899384e-07f64,
        25.49358674061477f64,
        -7554352.023780724f64,
        -627.327362644341f64,
        -5464174517.052722f64,
        518933.1536446702f64,
        7971440868.887214f64,
        -78623984286191.5f64,
        7.323721570297519e-07f64,
        -7.206821808517019e-05f64,
        -120684.48497542072f64,
        1736989468889.2783f64,
        -3.6170099520718262e-06f64,
        -16389166.82322198f64,
        0.07739289816532435f64,
        -9172859682.77457f64,
        78494225010.40926f64,
        0.009409783042816212f64,
        -448033.727132809f64,
        0.0007895803757755689f64,
        -276288957142.47046f64,
        -3.7591796518133514e-05f64,
        762578.3888978781f64,
        -8.690065036591272e-07f64,
        7333102438218.284f64,
        6555352.368534399f64,
        0.007894296822011045f64,
        -8.804854651403103f64,
        -522.3888108911993f64,
        5.503072304198458e-06f64,
        -12629982434136.732f64,
        -0.9893736827447424f64,
        3852392420.58941f64,
        59953934.79322269f64,
        9.472498804670932e-08f64,
        -69867.17411829508f64,
        -0.0004960900501440824f64,
        6827211840729.668f64,
        -0.8163817527416735f64,
        -45145886336325.414f64,
        -964745.8664798791f64,
        -16331433.821416397f64,
        0.0008546785877058127f64,
        7373685251215.263f64,
        262526707037.7919f64,
        3080001585.116363f64,
        18444313682.1814f64,
        -578.3427096522311f64,
        -10.068178209355283f64,
        9750.76781872698f64,
        0.0081846085245546f64,
        5.348888254470612e-09f64,
        6.736333012742591e-05f64,
        7377315.412707073f64,
        0.0002617649324668552f64,
        -0.0020486146775473f64,
        52.63932621165779f64,
        -602832959.9372972f64,
        14564.028262826523f64,
        -372750.94463037275f64,
        5627986496282.409f64,
        65504084529.65768f64,
        987.5926180412562f64,
        6699462371968.241f64,
        -88147.327806156f64,
        4.486513545607873f64,
        0.0006163590058603647f64,
        7444844419647.032f64,
        -16655676487959.07f64,
        -4.8176779509982114e-08f64,
        61.62722721301983f64,
        -6.901334443268267e-08f64,
        9.139107409064477e-06f64,
        974008545013821.6f64,
        -0.0047403379016066885f64,
        9146722813997.611f64,
        -583563795.0616819f64,
        -0.08305125495915576f64,
        0.0046445950251217385f64,
        632812291.8353858f64,
        -600542927610402.0f64,
        90.7514696329497f64,
        -20053.465632352374f64,
        -6.282638065857256f64,
        -39885106598.36171f64,
        -77437779.27484383f64,
        543641.8306682125f64,
        -429374333056872.0f64,
        38659218.87462995f64,
        -78191815586.56752f64,
        -7.500539345192623f64,
        17449089669.306873f64,
        93435089394.85223f64,
        -0.06302589916274515f64,
        92514910640.47182f64,
        -76385920.00782494f64,
        15285508434.341133f64,
        -6528453974885.094f64,
        9919443.20454867f64,
        995785602852.5902f64,
        74344386910.52199f64,
        415007.0838781228f64,
        -10287586578.252617f64,
        -4.807752334537081e-08f64,
        0.000744208964714964f64,
        7.705488411087074e-08f64,
        5.680668953948009e-08f64,
        6137332097.9344845f64,
        -91652591757010.75f64,
        -56195.434317215055f64,
        1.0907656541650068f64,
        -842876.2986250628f64,
        1.804965880398053e-08f64,
        8945101.024220515f64,
        -292415974555.50726f64,
        0.3807517850377453f64,
        -596207072375.9639f64,
        5929.43784702727f64,
        792883.9897440562f64,
        0.06304693081189423f64,
        -1.1851120585360419e-07f64,
        -3.035092352562141e-07f64,
        -832504852540.6438f64,
        6959.628202697403f64,
        807.6405413001632f64,
        1169.630589435875f64,
        5.282527163242527e-07f64,
        -544218554448260.75f64,
        5.119035319294043e-05f64,
        -50046.93146223143f64,
        4.58129488104776e-06f64,
        -9.760563130617816f64,
        -230605666903.77814f64,
        -0.052361815270625445f64,
        -9191579523413.953f64,
        -48102937.66728466f64,
        83710780603573.7f64,
        -0.003949213074881057f64,
        -0.07671445780000169f64,
        3.336100639651449f64,
        544550.0792640856f64,
        -310417.8631758947f64,
        846451.826064262f64,
        3942587752.2144346f64,
        0.07509205324222257f64,
        7931876.700410094f64,
        6.179826270406577e-09f64,
        2.0851365447312654e-05f64,
        -4.9743498620080293e-08f64,
        4.4997516466914744e-08f64,
        2.4114286192461498e-08f64,
        0.07937573557102223f64,
        87.77585656477696f64,
        -8.771993878560887e-06f64,
        0.5347456188869293f64,
        8.743908005307891e-06f64,
        -1.3796060622033579e-09f64,
        507590632.71596336f64,
        -5.4049377976245535e-08f64,
        742303607.7249725f64,
        -6.512581298785058f64,
        -4.7623122615913014e-08f64,
        -5083653.417506278f64,
        -7158039138644.079f64,
        3388.864885516389f64,
        6855203790.742903f64,
        -6848774839.744241f64,
        -6.015976572798172e-05f64,
        -20522.939413527653f64,
        -3.9823395152272243e-07f64,
        2.759490949008099e-07f64,
        -0.007658819696993804f64,
        1.652480550100257e-07f64,
        7341684.847301577f64,
        -0.7650953824787872f64,
        -10373.341318340223f64,
        -250448.62095811026f64,
        -8650755095580.929f64,
        -348240923673.6372f64,
        2853.943158544849f64,
        7505934227837.6455f64,
        0.9052546003504909f64,
        473149314346555.06f64,
        -1836618.3623318078f64,
        -702327472671.3817f64,
        366238.0183049483f64,
        1.4620316592804294f64,
        7.223468109991731e-08f64,
        -48774.5185632797f64,
        31379407975492.656f64,
        -8.13024334357978e-08f64,
        -3809.0122750308233f64,
        -13772903659507.14f64,
        6.107109498188671e-07f64,
        -304586.4688921551f64,
        237.7796250955615f64,
        9.72052490127728e-08f64,
        8925.486123230088f64,
        -447757165565.7853f64,
        -581394826.9831331f64,
        230975566.8463389f64,
        58706841.98102527f64,
        187749453388077.47f64,
        0.000495385389685754f64,
        -5.733489892517696e-07f64,
        -4.283847102653988e-05f64,
        -68.75919096645833f64,
        -8848.597043356009f64,
        -7.901919773786004e-08f64,
        54635084.58597655f64,
        -25118456124222.61f64,
        0.0009062818927639413f64,
        -2.972141606377965e-06f64,
        -238.51397121260564f64,
        704722623.6108431f64,
        3.205598704981658e-07f64,
        6870475.25394f64,
        -5.277709748473354e-06f64,
        -824863.265730895f64,
        -6724849962.4278345f64,
        -669.2433198918952f64,
        -1621968479665.0635f64,
        -33.857549061156234f64,
        942910.6741187021f64,
        -48032237238.689125f64,
        36549984009561.06f64,
        892266.6695545122f64,
        6.3562890160068f64,
        0.1530334712860577f64,
        -4.8270480687063814e-06f64,
        581973325.7287456f64,
        6.72365000888684e-06f64,
        -81173494.34364012f64,
        692858100912179.9f64,
        84.04097459183954f64,
        -5.9515541266010286e-05f64,
        -63305640.81992833f64,
        4527223685692.386f64,
        -3.917386016859803e-07f64,
        -8.520767261323243e-06f64,
        22.678036823947735f64,
        658387.3316539078f64,
        4583896135.657728f64,
        -8982255567701.453f64,
        -9146506030.116245f64,
        3595137.5859473613f64,
        3154440011.289528f64,
        8412875607.346859f64,
        422073830309750.0f64,
        355995.1060965563f64,
        7.504400069708965f64,
        -1.8636376988078008e-05f64,
        -6561235872119.2705f64,
        -7.700245935205512f64,
        -2857884045.374808f64,
        7.28862098610392e-05f64,
        979538051.4418639f64,
        -9.128861094051125e-07f64,
        0.08141780131033971f64,
        64744483748.28943f64,
        -8.883783354314407e-07f64,
        0.005719475158528882f64,
        -6351115602.457842f64,
        -8.994451264201982e-05f64,
        -2.087911177280437e-08f64,
        4.1110808380026077e-05f64,
        -4619719.142251566f64,
        -0.1362992579540947f64,
        6.78489855386621e-09f64,
        -757083175097988.2f64,
        2.808906303252765e-05f64,
        -3699.7383932191806f64,
        -87.69720263620164f64,
        5.271209374790509f64,
        348729072542.0711f64,
        -77731.76092819912f64,
        -0.0066065184725282f64,
        -2.232470194836105e-05f64,
        35990893502951.35f64,
        -0.009811932075143198f64,
        -78267145986511.2f64
    ];
    let n = y.len();
    let x: Array2<usize> = Array2::zeros((n, 1));
    let fitted = TargetEncoder::<f64>::new(0.0).fit(&x, &y).unwrap();
    assert_eq!(
        fitted.global_mean().to_bits(),
        SK_MEAN.to_bits(),
        "n={n}: ferro global_mean()={} (bits {:#x}) != np.mean={} (bits {:#x})",
        fitted.global_mean(),
        fitted.global_mean().to_bits(),
        SK_MEAN,
        SK_MEAN.to_bits()
    );
}

/// GREEN guard — `pairwise_sum` branch coverage. NumPy is the oracle:
/// `float(np.mean(y))` for n=1000 ill-conditioned mixed-magnitude y (seed 7).
/// sklearn `target_mean_ = np.mean(y, axis=0)` (`_target_encoder.py:383`).
/// Asserts ferrolearn `global_mean()` bit-matches `np.mean` (exact f64 equality).
#[test]
fn green_pairwise_n_gt_128_1000() {
    const SK_MEAN: f64 = 5200671267529.136f64; // float(np.mean(y)), sklearn 1.5.2 / numpy
    let y: Array1<f64> = array![
        2501909.3320933394f64,
        794427601939.151f64,
        0.00551371380490387f64,
        -5495856200188.163f64,
        -3.996674301775491e-08f64,
        747106.8907925237f64,
        -9.894693908688506f64,
        6424.568367655325f64,
        594138.8575040925f64,
        -6.413009431255845e-06f64,
        -3.9393514636137294e-08f64,
        -4431.487757984533f64,
        -49026082469175.08f64,
        -0.0010984738823470687f64,
        909.6517915906599f64,
        1.0699470414898493e-06f64,
        0.9910005668687853f64,
        5.853238384275062e-06f64,
        2.4435845888232533e-07f64,
        97792029536.37698f64,
        -5693826035288.021f64,
        -67957593228431.09f64,
        225079208.54606155f64,
        -9.121159840772333e-06f64,
        -92863944245280.77f64,
        2.977764054274057e-10f64,
        -0.000675879493494218f64,
        0.008343355463857045f64,
        258452508.98202088f64,
        2.8235293199027736e-07f64,
        -6253129212991.481f64,
        -5.049701559453384e-05f64,
        -976411948914.9883f64,
        -61519.57120293787f64,
        0.0038406424176367837f64,
        -59.87865520260096f64,
        -2.6092737879558656e-07f64,
        -99.2531515895848f64,
        66009545.96034911f64,
        -6.910778378771203e-05f64,
        -464801390872429.1f64,
        7606643.079616573f64,
        0.019581619736846356f64,
        0.006943004927317387f64,
        2794343338.8505244f64,
        4.835418947237143f64,
        -81700878987390.88f64,
        82.2876427529775f64,
        1554447.260069991f64,
        7426787533857.613f64,
        -27747188197168.48f64,
        0.00019636813441442614f64,
        -88.14967153089927f64,
        -224736397.7785426f64,
        -3.539273074835867f64,
        -0.06996005418590963f64,
        632676.2076381514f64,
        -2411076568993.751f64,
        957.4957688224432f64,
        1.7998338602122055f64,
        2101125076.5970256f64,
        2759.9316157666444f64,
        0.03529004876255766f64,
        -6984239616632.626f64,
        -119373.06562362493f64,
        -52087207.63409533f64,
        -19500340379.203674f64,
        -8.065918121365088e-07f64,
        935656102097.6428f64,
        -5.6999192528824e-06f64,
        3.4353032522256987e-09f64,
        -399159837041859.44f64,
        0.007481540522990089f64,
        3.2442947667690757e-09f64,
        -73676836.83833885f64,
        0.06901486417491058f64,
        88.98963422899591f64,
        807833576391853.6f64,
        13943.82957185545f64,
        -7.090800924781462e-09f64,
        -61507301.00633352f64,
        8.55811369489049e-06f64,
        0.010465297533452756f64,
        -0.0006388950031021767f64,
        7681137883929.398f64,
        2.831434104449615e-07f64,
        1393.8854894761587f64,
        -247.42432774015975f64,
        -17808.94356857403f64,
        -521021.5746363103f64,
        -9.238854266175218e-07f64,
        0.0752437616218542f64,
        -6453.95663718078f64,
        9.527039842747054e-08f64,
        -35567338043.95498f64,
        50264983970.98558f64,
        -9496.062583964793f64,
        -25562945486.95921f64,
        -93.92994112317767f64,
        -75421.57955899813f64,
        934.2964707947355f64,
        3.1552146007702887e-06f64,
        -1.4355950722103739e-05f64,
        4748.021582096063f64,
        7.456184171295488e-08f64,
        -3115786.660079476f64,
        0.018058196457942932f64,
        36.73687466790874f64,
        -2891724459529.759f64,
        38196972991.98022f64,
        530.4947665570452f64,
        8.183586279811003f64,
        -69787544384.6369f64,
        8668.387846148571f64,
        -989642268.3691357f64,
        5.059550071955079e-07f64,
        621053660.6344903f64,
        -72647.1896914837f64,
        -0.0001621926981736297f64,
        6305125559.900912f64,
        -9714.576206307787f64,
        256923895.73258147f64,
        58604731619.60682f64,
        0.00026007165644416965f64,
        45169884110.21743f64,
        -547153034605711.9f64,
        -6029.577040146543f64,
        -273746098.8992968f64,
        -6411879448.657069f64,
        -30.78771217750431f64,
        89624812282.16295f64,
        1.4666543522224185f64,
        -3198638651583.203f64,
        -4569507603.824294f64,
        9040.789814150445f64,
        -1.1104358070577348e-05f64,
        96078950176.11789f64,
        310453.3821254196f64,
        4233.225805244789f64,
        793081047.4320016f64,
        485534836430079.1f64,
        1613.057277056451f64,
        -1.4670099232748335f64,
        756375718836.5609f64,
        -1767077.0018622316f64,
        845.519152336935f64,
        -8.62569295756277e-05f64,
        -140.0062761976739f64,
        3902963975571.794f64,
        0.009018763911636996f64,
        -498001506670483.7f64,
        61207829452421.76f64,
        352942399997.18677f64,
        0.04341718085622459f64,
        25924436568.04071f64,
        94312141.69231601f64,
        -3346370815773.503f64,
        -2.034488806412589f64,
        -594176.4955729676f64,
        -89859188945545.89f64,
        -5741836100493.216f64,
        83092879424781.66f64,
        6.803376343167792e-07f64,
        -7.75188517014842e-09f64,
        2.0755805063373066e-07f64,
        -41607010220750.414f64,
        1.8936974097655712f64,
        318550.0132181265f64,
        -38.66810182222562f64,
        9.227016894729139e-08f64,
        -683199298777.5015f64,
        256.20170972981373f64,
        0.02704523677994264f64,
        -6.322212069806176e-08f64,
        -8.762691636872516e-07f64,
        -17696.635300405018f64,
        5280601768.769373f64,
        6.304435765835435e-05f64,
        4599.7849800413f64,
        -0.773590094949554f64,
        8267097.23062091f64,
        0.06040731559945676f64,
        7553827332990.67f64,
        46608.30595035459f64,
        8.312708702014649e-05f64,
        -90669552409.22256f64,
        -9394223.32136796f64,
        -95956885328770.62f64,
        -49446264409508.72f64,
        -0.05028604535820343f64,
        -62499333.12880669f64,
        13411.16367707913f64,
        -92.20283179339093f64,
        1.8077573566970351e-09f64,
        -66797769390.556595f64,
        0.3557474171346189f64,
        -9.578492911122178e-08f64,
        -378859605893610.25f64,
        8.76682573729228f64,
        7.67927597544893e-08f64,
        62.31748054935016f64,
        316052163223.45166f64,
        221501571.2270525f64,
        -6174946.398302052f64,
        0.014878949651561203f64,
        -0.09206271630820001f64,
        0.006033288101837467f64,
        9201418340.523521f64,
        70801814268.36905f64,
        -0.008985807080678776f64,
        -0.032267983323956306f64,
        -3639936042754.355f64,
        -774566016554260.0f64,
        2.532236386562421e-08f64,
        0.005949163507326667f64,
        -3.725570559955122e-07f64,
        0.7256184698374695f64,
        5942538256.922594f64,
        -741724.1170513504f64,
        53.37183134763106f64,
        7.652414398038366e-05f64,
        -6.05434860336507e-06f64,
        1472823589.9704623f64,
        2774.9993204420575f64,
        2.186685149851997e-09f64,
        -0.08075086447439131f64,
        32238292.20298984f64,
        26390.944935903728f64,
        647770978.0040286f64,
        607025398.2606566f64,
        -3.4566358413468863e-07f64,
        444094665853.1077f64,
        734546.775144712f64,
        7.858955163815791e-08f64,
        -67697534610600.15f64,
        -946595.2998274307f64,
        0.00030161488684445726f64,
        -570647454512473.4f64,
        1.2741944853413844e-09f64,
        889.6090596136819f64,
        -2.413607514948646e-09f64,
        -4944508992.313866f64,
        -869799000.7626122f64,
        3.144878265414672e-07f64,
        -0.07978020642906825f64,
        -238830472998.91458f64,
        -0.007325576223229546f64,
        32489.24432285076f64,
        66110.50645263678f64,
        -2462924371824.351f64,
        -256552094.45116344f64,
        790433178.9486907f64,
        -5.698844407238128e-07f64,
        -505180816766.4582f64,
        -34.029543343517176f64,
        -8.514860372356513e-09f64,
        -8.369370793948531e-07f64,
        505464272.7340113f64,
        1581091488585.0847f64,
        -4.006122542189976e-05f64,
        -0.08449068552979944f64,
        0.005263619028029624f64,
        -7378420234173.586f64,
        -0.07335867153006435f64,
        -7386315.083854467f64,
        -837475.3777680191f64,
        8.12784730800514f64,
        -4615114.00653219f64,
        -3.871788415862796e-05f64,
        66558862000413.52f64,
        2398469194236.0283f64,
        -6.257132269185624e-09f64,
        -130.37317524682223f64,
        7.678449294442934e-05f64,
        -24.925196596107256f64,
        0.42176310639306736f64,
        -806384582327.0638f64,
        45464955347.70878f64,
        0.00552947048130493f64,
        65153.356340426515f64,
        3.484019808704224f64,
        -2585.344103474949f64,
        -8715759996677.928f64,
        375524681.20589876f64,
        5.149198127644041e-08f64,
        -6183235.930206456f64,
        -4675255467.741128f64,
        0.0007224008475874433f64,
        496663125.7958385f64,
        7.931734615183589e-07f64,
        -7.485169649316616e-07f64,
        -631.4595742670068f64,
        5.990739427762857f64,
        2890432460469.494f64,
        44194558361.99548f64,
        993542.3300626273f64,
        87835.99173817622f64,
        6.860501336831723e-08f64,
        5.5423161810636404e-05f64,
        -20.996156958712618f64,
        28.245822121963027f64,
        -6.311087151068422f64,
        5189.889305616118f64,
        51538.132903677746f64,
        4.425904272694381f64,
        -110401.95114644202f64,
        -2.436393424183578e-07f64,
        -16045813526011.732f64,
        -9333.215297775068f64,
        0.6886402328229859f64,
        8.47397551095428f64,
        -2.2496272199653733f64,
        0.0009605753653207772f64,
        443280.43703932443f64,
        -2370782612032.085f64,
        6612813468239.838f64,
        8.389266926107741e-05f64,
        -22513540.53787111f64,
        -7243676913.001378f64,
        520.7465919803644f64,
        98589771.74972121f64,
        -70402370247.58707f64,
        4.253513521229297e-07f64,
        650.6468006000805f64,
        8.411438899927222e-09f64,
        -753237171444851.4f64,
        -8163.801736967811f64,
        0.009757431636930672f64,
        -7.664870297968233e-08f64,
        -6.46384881726208e-06f64,
        0.0014990586765803804f64,
        -10745392742.073069f64,
        500784382655665.94f64,
        -0.06188855036837733f64,
        0.8288555231198389f64,
        -565610307384.555f64,
        5.382219803732795e-07f64,
        -864792.9494518787f64,
        -53194.824485669254f64,
        -9348.833467096141f64,
        -372378587205.79407f64,
        -3.755386516499419e-07f64,
        0.004394968782771512f64,
        -899650336.9520025f64,
        -0.08864512752141665f64,
        9.907233188579938e-08f64,
        77.73986134166655f64,
        832647869946806.1f64,
        -5068489398.5272255f64,
        -2117794905500.6406f64,
        -0.0054564099817329325f64,
        -0.07501872356446687f64,
        -9.339521506685287e-06f64,
        6.672895972813954e-08f64,
        -0.007537326878672206f64,
        -6473912572637.457f64,
        72095136.09152418f64,
        -31.514462732137318f64,
        -632.5929579595013f64,
        3.3972911963462437e-07f64,
        -468.27024461036103f64,
        53.87440010011435f64,
        -4340942642.753112f64,
        3.232237502188773e-05f64,
        257067181498.30344f64,
        0.7241615908253562f64,
        -2.087917854101988e-08f64,
        5816275.975410583f64,
        746874.8982913689f64,
        -6412574726235.873f64,
        -72738.25503639391f64,
        -77361671843.59146f64,
        0.0009591934608966347f64,
        8.831826057118356e-06f64,
        -5386656.353977952f64,
        939.8151471643425f64,
        -58436466575.3192f64,
        12.952032376261524f64,
        -522991156.09695244f64,
        829911805477680.1f64,
        -918942280210447.0f64,
        -3.692983711312372e-07f64,
        199949396931454.97f64,
        -8672.03562993442f64,
        -0.5268988471879565f64,
        -698730918049.1908f64,
        7.617140537384555e-05f64,
        5.219190433155756f64,
        65796093912.34829f64,
        0.005221395450586248f64,
        41544155209956.6f64,
        69938.66852928934f64,
        3629.533814733776f64,
        0.047136611862720224f64,
        -3.967109944955716e-06f64,
        -66471855420251.67f64,
        5.130499583115769e-09f64,
        -6.683254084626775e-07f64,
        8.389117030399381e-08f64,
        1.932856747122427e-05f64,
        -341131774.25634956f64,
        8.732863133047017e-07f64,
        -6.897394561243105e-06f64,
        28932675.390492823f64,
        -8.168918535675278e-09f64,
        0.09308554181691475f64,
        15075212362.325785f64,
        6073331956001.08f64,
        -436.1551627045943f64,
        6035932533370.773f64,
        4056839311781.959f64,
        28736293958537.05f64,
        0.000901125629814064f64,
        -0.0013301732380951404f64,
        -1698231.328157458f64,
        3.842422575700335e-09f64,
        0.006701106122860789f64,
        -3.2984795993241134f64,
        3393141093914.784f64,
        -581925184271787.4f64,
        0.10339994730576607f64,
        5383434037.362658f64,
        -86937.00976996604f64,
        4.556841308885431e-08f64,
        -9.691350460989934f64,
        91670005.7990582f64,
        -6266039.900314224f64,
        -18185828743724.23f64,
        4408893.017354376f64,
        46597739932183.305f64,
        46.16977106465268f64,
        -830484210.5325787f64,
        125583.74132574523f64,
        116217385699.17528f64,
        86422778341764.02f64,
        -920788843822.7363f64,
        -9436071388718.47f64,
        0.2621458243200654f64,
        101493902536503.14f64,
        -8.51677804365186e-07f64,
        0.18645678637129115f64,
        -555606229.5454713f64,
        -608901.641326304f64,
        7.5738176949530716e-06f64,
        -60431578.4520627f64,
        -0.9141849598705654f64,
        50057376447.84065f64,
        414630772049188.44f64,
        1.0691801184259698e-05f64,
        61409.149385278084f64,
        -6.836765159200931e-10f64,
        2407.7086747886246f64,
        637833281437.8722f64,
        35618327202592.64f64,
        28365673341.475662f64,
        -0.0018774255252576788f64,
        1.166469161867243f64,
        -207834672047325.66f64,
        4885845105.942979f64,
        -2391396877714.995f64,
        -6.675769136111054e-05f64,
        0.5109141799957033f64,
        731159694.536987f64,
        -0.03239348586360724f64,
        6536378777.388465f64,
        -2.3807481173419445e-06f64,
        6.89458201348977e-08f64,
        5704808053974.239f64,
        -0.010542235245817056f64,
        4.257818794596555e-05f64,
        -0.009312205994983233f64,
        -2.2040319223632564e-09f64,
        7206673723.921477f64,
        1590787150367.1877f64,
        1.1565165244679566e-09f64,
        329477.77110709506f64,
        0.3553432639485117f64,
        16719650.209424874f64,
        -15874137.898014108f64,
        -633032623.626407f64,
        -4.147159297282674e-08f64,
        -4.141556459309399e-08f64,
        -13851015071953.742f64,
        9.981175109596316e-07f64,
        -2948.961162426573f64,
        -105.24143636146289f64,
        -2.567448723470944f64,
        1.6285547904974228e-08f64,
        8.957971589516252e-05f64,
        7781.457874370381f64,
        -2.4572250859093444e-07f64,
        -466197715.91613436f64,
        768797208702.7988f64,
        -123333.22171682148f64,
        37108229423.631325f64,
        -853417.1543406941f64,
        73311539.37724961f64,
        -3.740696747661487e-08f64,
        -1094.6085425061813f64,
        -0.000603957836481086f64,
        -1.59514754547579e-05f64,
        6560978345935.7705f64,
        65515849574.73972f64,
        -5.265597386323239e-08f64,
        535821402357030.0f64,
        -838.6626854685386f64,
        -253287.31964158325f64,
        9.177795316090731e-09f64,
        -5973184656810.157f64,
        -36687695023986.45f64,
        4130575426.467569f64,
        5.4102796306936085e-05f64,
        -8865.837294094908f64,
        4.65652280105497f64,
        6.348598713964249f64,
        -10961.285350346705f64,
        -875092174.2781936f64,
        365826510.8910981f64,
        -0.0005076726807376553f64,
        28671465784443.95f64,
        -24.67549818649919f64,
        274327040.77578986f64,
        -483730239844.7074f64,
        -0.046925954850743246f64,
        49349478068.97538f64,
        2.62026364861589e-08f64,
        -0.05936643168495897f64,
        92410215.04049815f64,
        -4217238485.13287f64,
        -387959736465039.8f64,
        -48.737523367970816f64,
        -8.650474788638126e-09f64,
        -15.415417419699228f64,
        5436726226.334301f64,
        4.1913387336969694e-07f64,
        -6.088550939193678e-05f64,
        -7.261361949333347f64,
        -357.1441191968594f64,
        53.65103067801456f64,
        -238.4372829554926f64,
        846368.364881372f64,
        312767.5318538734f64,
        2.41813561224268f64,
        535382252094644.3f64,
        500.51649538592534f64,
        -31232374.997121282f64,
        -4859831.965446398f64,
        -1.2475722054863558f64,
        -215056.38457376408f64,
        -716554459.1521617f64,
        -117.33560991310887f64,
        279.6465117633249f64,
        -728570.9900330406f64,
        -7.2392422388209f64,
        -6525571749.734565f64,
        -0.43333718928349585f64,
        9.234185497588846e-07f64,
        -948.417414809077f64,
        -6325217634376.745f64,
        -0.006699043212455989f64,
        -0.005130394969277541f64,
        -0.0009698544429391667f64,
        -8365355891525.605f64,
        -6.741551001730005e-06f64,
        -46.071955467383205f64,
        58.48812115207953f64,
        -63186.55570461789f64,
        0.023425766613792788f64,
        -1873754602.17174f64,
        -164808861528.0603f64,
        63.36908351958701f64,
        -678170091625.6257f64,
        8.47620450244157e-07f64,
        0.7869817145224203f64,
        3366439256754.0747f64,
        -901105.3585767883f64,
        3.266170569819827e-05f64,
        2006953647280.183f64,
        5444118172665.422f64,
        -0.5723368353464395f64,
        9.067120207957112e-05f64,
        77239.42995281654f64,
        -771285.7191711613f64,
        -36.959628282274146f64,
        -935111622.1720799f64,
        6618944.252161427f64,
        8301894933.358889f64,
        979618502751803.6f64,
        2238500352.8349643f64,
        3.319711576975575e-06f64,
        3.568110669922251e-09f64,
        140296869625865.73f64,
        90484582269.52267f64,
        2157224.5745697315f64,
        -0.003285479216057197f64,
        2575447986.906787f64,
        -9.626742502375888e-07f64,
        4482526.09649449f64,
        3.8784215238906716f64,
        -157897717506.16443f64,
        70296218.09374335f64,
        0.05110521917005988f64,
        -3266308032.481675f64,
        -84500406.05253786f64,
        -0.007355539503422475f64,
        -0.006314555413303864f64,
        -3.921639419761229f64,
        -1049.6855155880191f64,
        0.008559931565801752f64,
        -44316420.42408361f64,
        7971.713260862026f64,
        3.609243602131385e-05f64,
        70698583.00398715f64,
        -1.9349321062395153e-08f64,
        -99344651617270.97f64,
        5739347241.313426f64,
        -5.090482975616821e-07f64,
        -71503958847.49858f64,
        0.05989831965689174f64,
        3726572.99995649f64,
        -0.0005751695983141727f64,
        -0.046483637055893895f64,
        8.784396628659143e-09f64,
        -4629240.498154659f64,
        404811.75608771294f64,
        -7.40893506713619e-07f64,
        0.0063007173841566465f64,
        -7648199.775647134f64,
        3208225356.152172f64,
        -92956.47468655405f64,
        1.0015709119468741e-06f64,
        693243.2279694217f64,
        -0.0008312029761709093f64,
        -45.14868412189181f64,
        529.3861267854625f64,
        0.00021619663891447272f64,
        -37440618690.58639f64,
        257807758.90823036f64,
        -9.057935915532766e-08f64,
        724.7125314108957f64,
        -4.1412019500424853e-07f64,
        515461206.30464286f64,
        2.713483101396428f64,
        0.4808573216205583f64,
        -0.00024347677260142664f64,
        4.49842037860829e-07f64,
        3.968521934784415e-06f64,
        -22502.22157946431f64,
        -947095539.3642191f64,
        925826.636760392f64,
        0.0004272007921529368f64,
        -59052270314239.96f64,
        4616144.271128253f64,
        -4426332728523.881f64,
        -6.142921426237246e-10f64,
        7.197763212270907e-08f64,
        -1.038557941011855e-08f64,
        -4239868543.195089f64,
        -0.3890177767719729f64,
        -0.008459654666906536f64,
        -560988.8407245886f64,
        -45778754.13907986f64,
        -65.40617746370468f64,
        -8.738685707997288f64,
        -0.6842167405574306f64,
        -0.05712077378043399f64,
        4.77541022168968f64,
        845.0555256058459f64,
        -0.04540306005528192f64,
        1444.9658155307122f64,
        134513.29689529113f64,
        -84978292.2525931f64,
        -7.502512352571604e-09f64,
        7695589.158009703f64,
        -0.0051695924334409955f64,
        60921982.06120771f64,
        -866838510451692.9f64,
        1.8962627316051495f64,
        7.306239085498125e-07f64,
        73413219.34717077f64,
        -0.0988555141253177f64,
        0.04344020899795087f64,
        -261990.5014364874f64,
        520084.0581980983f64,
        -8.535620063743626e-09f64,
        -470.14687565013037f64,
        6.252855706443359f64,
        -6088.755492631992f64,
        -936993.5701658472f64,
        9.371172961553723e-09f64,
        -3.602186753525416e-08f64,
        -916.5558827092606f64,
        -215598331881.58594f64,
        -0.0006932528768607386f64,
        -8.362162876046334e-07f64,
        1.357207135624321f64,
        0.009071512321168007f64,
        1.5881514855275736e-08f64,
        29422909511437.246f64,
        -3527422.76721556f64,
        14224571390.71176f64,
        5044890272830.32f64,
        1785697.1070359084f64,
        0.6055567483749777f64,
        1.00941360913408e-09f64,
        -6.037118870579379e-07f64,
        0.016545250846958858f64,
        -18128.30577833324f64,
        -6681289.350700148f64,
        2.3833170232095836e-06f64,
        0.006339210848708639f64,
        -7298246130444.206f64,
        0.008989172327601148f64,
        4.6737426558654294e-07f64,
        -0.00019396480763493408f64,
        -0.4599075175765177f64,
        -255.43917865977937f64,
        83.39703767421902f64,
        3.733207464258532e-07f64,
        1.1835044351153367e-07f64,
        -7.85949930014664e-05f64,
        0.06357727864895328f64,
        847719428443624.0f64,
        -0.007974143151163553f64,
        -0.5009610758237424f64,
        -652637623.3801743f64,
        0.6983103298599533f64,
        820557447235675.8f64,
        -911.4524914399223f64,
        -0.33516117207074547f64,
        -6150107739.031354f64,
        -609345862.2133174f64,
        8431808.54571201f64,
        -8.59653516699414e-06f64,
        -0.0657441774016887f64,
        -2594.143703725704f64,
        0.04806938952561701f64,
        49.10025659883941f64,
        -0.0012462792290494251f64,
        -0.007177264971209394f64,
        -292.77625638990077f64,
        96418623653.6396f64,
        0.5014442930754461f64,
        -767311256.9304541f64,
        616418357677979.8f64,
        -3.7814707203703306e-06f64,
        56279296422271.25f64,
        -0.09745571016099988f64,
        7.425403080335053e-07f64,
        -67537935238.946465f64,
        -8.849880868459459e-10f64,
        323511196581688.8f64,
        1.573913648925318e-06f64,
        -669204.4648852086f64,
        -0.0007217629735828799f64,
        -430931214152.0738f64,
        -89521648.59645833f64,
        -566910465779922.6f64,
        -7397899674.49491f64,
        -70405970.97475399f64,
        -28698014131465.156f64,
        1736644.9740094247f64,
        -0.00043824227073532374f64,
        -0.0009321779029529365f64,
        -3377796.768866357f64,
        0.5103614082593144f64,
        -4883236797.655752f64,
        -190903.13314708695f64,
        5.099487609357964e-09f64,
        -18663743191133.332f64,
        774248263898906.5f64,
        0.08361370118970522f64,
        -0.0037894072365488875f64,
        4.719273061772196f64,
        -49376603425.68764f64,
        -7.209532589434118e-09f64,
        -1.2423791737699964e-05f64,
        20494485.800163243f64,
        87.1181784102064f64,
        -70.81969619347805f64,
        42472535065293.65f64,
        -4907099026.087014f64,
        -7.365814047624632e-07f64,
        0.0009453160945994708f64,
        0.003915378969146299f64,
        -7422826.484724774f64,
        81.15108372454888f64,
        -79381167.88955307f64,
        -0.0008146815306002735f64,
        -336.8980089111826f64,
        0.8759183204020633f64,
        -9.33951037041791e-05f64,
        8.364144821604048e-05f64,
        2.23223157390253f64,
        -86589672.37594838f64,
        137183450.85702178f64,
        -824185549.6345218f64,
        1.2796909773602794e-05f64,
        47440.70821593442f64,
        3.572381943681253e-07f64,
        9.94455096134396e-06f64,
        -686.6618727322507f64,
        62571083.75054663f64,
        -0.00017781402619085386f64,
        5.666150421208185e-05f64,
        78355834759.0016f64,
        983996.3759624053f64,
        -46791631007.21322f64,
        -9.412345792345977e-08f64,
        88555854712697.14f64,
        89147675991.87741f64,
        -254252314254.21014f64,
        8.804189089686127e-07f64,
        -1882176.8944242722f64,
        -2504.6780204938423f64,
        5.1323879054629296e-05f64,
        -575763401179.6075f64,
        -10360308763.959503f64,
        -22776201264.335037f64,
        -134392136.18482038f64,
        -7.203647071790771e-05f64,
        -877925705.0944855f64,
        507.9862893708955f64,
        12899989.916184507f64,
        343781.29591639707f64,
        60467090428.53194f64,
        -5397.876750178705f64,
        -0.05324541435099209f64,
        -4441827.342369127f64,
        -39215001.76967687f64,
        0.005213818356968012f64,
        -2516.368275097449f64,
        -3536222.7424093294f64,
        0.2726031398919464f64,
        -6.268589972598018e-05f64,
        -0.07871596998276453f64,
        394266476385.26746f64,
        5.128906640029965e-06f64,
        -4.072707941262328f64,
        272.08235370392964f64,
        -93184686421.4809f64,
        -0.191362038840883f64,
        155916725711753.3f64,
        -49.31629458739819f64,
        -303.5701961892241f64,
        219099127621.72888f64,
        17.97956888341643f64,
        -69161852.5067514f64,
        0.4131637724248556f64,
        -83994144.84462418f64,
        939449.3482815328f64,
        -595276083652772.6f64,
        -4.92903714661812f64,
        -0.0005993989073785598f64,
        9.82997919055162e-06f64,
        -4268.101590691922f64,
        2105.845489776339f64,
        9.812683048873295e-07f64,
        9.00135711524052e-09f64,
        -9.153003418862069f64,
        4.2368137650443893e-10f64,
        20260935271251.566f64,
        4047901231.232116f64,
        -3.281813700283569e-07f64,
        -3441827797.3838763f64,
        -9914319687.827698f64,
        -929526597.5479974f64,
        100114337038.00577f64,
        -84822579.74573126f64,
        -5950.375768610041f64,
        -333456167.559107f64,
        -85905748300.76587f64,
        -938280.3200242093f64,
        85.38408988263764f64,
        -12958.274084261757f64,
        -0.5902310787195926f64,
        4.169036026983353f64,
        -9811.093174432235f64,
        -509965504.7083411f64,
        9.214992781108798e-05f64,
        79.53803458499395f64,
        0.0271850500810215f64,
        -8.13594457262022e-06f64,
        -0.03638983094723767f64,
        -0.0944134899684612f64,
        -561617991031010.6f64,
        -9.491317528339993f64,
        -0.7476726420645425f64,
        2.6642908234081797f64,
        -0.0007991143114841064f64,
        -6884738485039.843f64,
        80561596607606.58f64,
        -3.290951529385595f64,
        -3.3400615094075038e-06f64,
        -8.26921059199875e-09f64,
        591861043894299.5f64,
        35275448091.322815f64,
        94265600720.5278f64,
        235825.6837689039f64,
        -7992386.151921493f64,
        -4793878663.399304f64,
        16.728303173162896f64,
        90446387684.5867f64,
        6.342638841567416e-05f64,
        28.14527172209489f64,
        3.7637157647604e-08f64,
        -9697356465.247557f64,
        7.814325350907782e-07f64,
        -8662.918092188535f64,
        -0.7418378982042293f64,
        386143476876898.56f64,
        6.242879618917748f64,
        5.7310132773524085e-06f64,
        -9519175940001.533f64,
        -64588.844466603914f64,
        -9293.955668401892f64,
        7027592.203695883f64,
        955394926163735.1f64,
        -5.117289904321571e-09f64,
        732152655.2181339f64,
        -19.493816607492388f64,
        -3.773124800494854e-06f64,
        103086742367.05562f64,
        -98366.36365570723f64,
        -1145710615822.293f64,
        -6.008900628344079e-06f64,
        42491438.71603536f64,
        5.1427593913171424e-11f64,
        737705259864159.6f64,
        79329638177489.69f64,
        -3664821404998.302f64,
        -0.4392323886058409f64,
        87826076.7491611f64,
        7.99511859165647e-07f64,
        -806968947352.0432f64,
        -9.082297379161028e-07f64,
        980851048.3241415f64,
        1583272.5805559943f64,
        -0.0891014053612319f64,
        22.51089674226825f64,
        8.209005982792839f64,
        -1.2922431637325138e-09f64,
        9.025607970180795e-08f64,
        897544061.4676766f64,
        -75006464.34524073f64,
        -0.0069655859579855275f64,
        -18.50227884390325f64,
        6680529423.464685f64,
        24750519.026670802f64,
        459.6602557989442f64,
        152116764743.74646f64,
        -5798.861272083977f64,
        6.3553085756188874e-09f64,
        -97.19580358758012f64,
        0.00039696851579129876f64,
        -61132076.84874322f64,
        -5.7817040332562625e-08f64,
        950266410824154.2f64,
        78071.71503033428f64,
        -88610235776.21884f64,
        -142.4403402989256f64,
        3.2053421166384344e-08f64,
        -88.6586615196158f64,
        -494.69254699758045f64,
        -96.3327724147745f64,
        8.013650251957327e-05f64,
        -2.3704484164227924e-05f64,
        7.380033322657686e-09f64,
        -408.3263683144167f64,
        3.136140275078041e-05f64,
        -6538276.891571022f64,
        0.0008133040478049831f64,
        92.01629102437587f64,
        -14012685526482.604f64,
        -692.1974815035848f64,
        719233411.6703259f64,
        -42032440.48446546f64,
        606211360813.8541f64,
        -4.253847815181888e-08f64,
        43992958.90479504f64,
        -0.00033197731634833594f64,
        0.02052033425135165f64,
        8517364925769.129f64,
        -8.913504292284923f64,
        6145.092552855141f64,
        8.369113816565843f64,
        -2.418647392213482e-07f64,
        -0.02994490857504857f64,
        -8.611759534891803e-06f64,
        -48.32426050986609f64,
        2.4627404959047515e-09f64,
        -8048200811.876906f64,
        3388.1468073450715f64,
        9.245066782746847e-08f64,
        961552207096382.0f64,
        0.0077848243486898115f64,
        0.6132217085196707f64,
        -644.0933776050548f64,
        9403.119074031883f64,
        -59455359.474166736f64
    ];
    let n = y.len();
    let x: Array2<usize> = Array2::zeros((n, 1));
    let fitted = TargetEncoder::<f64>::new(0.0).fit(&x, &y).unwrap();
    assert_eq!(
        fitted.global_mean().to_bits(),
        SK_MEAN.to_bits(),
        "n={n}: ferro global_mean()={} (bits {:#x}) != np.mean={} (bits {:#x})",
        fitted.global_mean(),
        fitted.global_mean().to_bits(),
        SK_MEAN,
        SK_MEAN.to_bits()
    );
}

// ---------------------------------------------------------------------------
// PINNED DIVERGENCE — per-category encoding (sequential, rearranged formula)
// ---------------------------------------------------------------------------

/// Divergence: ferrolearn's per-category encoded value diverges at the bit level
/// from sklearn's `_fit_encoding_fast`.
///
/// sklearn (`sklearn/preprocessing/_target_encoder_fast.pyx`) computes, per
/// category, with `smooth_sum = smooth * y_mean`:
///   `:60`  `sums[cat_idx] = smooth_sum`            (seed = smooth * y_mean)
///   `:68`  `sums[X_int_tmp] += y[sample_idx]`       (SEQUENTIAL accumulation)
///   `:69`  `counts[X_int_tmp] += 1.0`               (count seeded from `smooth` at `:61`)
///   `:75`  `encoding = sums[cat_idx] / counts[cat_idx]`
/// i.e. `encoded = (smooth*y_mean + Σ y_i) / (smooth + count)`.
///
/// ferrolearn (`ferrolearn-preprocess/src/target_encoder.rs:219-230`) instead
/// computes a zero-seeded sequential sum, then the ALGEBRAICALLY-REARRANGED form:
///   `:220` `entry.0 = entry.0 + y[i];`              (sum seeded from 0.0)
///   `:228` `let cat_mean = sum / count_f;`          (divide first)
///   `:230` `(count_f * cat_mean + smooth*global_mean) / (count_f + smooth)`
/// The `count*(sum/count)` round-trip + separate `smooth*global_mean` product is
/// NOT bit-identical to sklearn's `(smooth*y_mean + Σ y_i)/(smooth+count)`.
///
/// Live oracle (sklearn 1.5.2 from /tmp), single feature, smooth=1.0,
/// cat0 (mixed magnitude) y = [6653563921156.749, 0.40053020400449824,
///  -0.03752667172359179, 664519602.7904022, 60952871.499360375,
///  -0.0002250432419396511], cat1 y = [0.13607512177992703, 3.9559294497604784,
///  3.3577691973009576, 3.555310355269042]:
///   TargetEncoder(smooth=1.0, target_type="continuous").fit(X, y)
///   float(e.encodings_[0][0]) -> 1045674047570.8059  (0x1.e6ee0f70a59cap+39)
/// ferrolearn computes                1045674047570.8058  (0x1.e6ee0f70a59c9p+39),
/// a 1-ULP divergence on the implemented manual-smooth path.
/// Tracking: #1262
#[test]
fn divergence_per_category_sum_formula() {
    // sklearn 1.5.2 live oracle: encodings_[0][0] for the cat0 fixture below.
    const SK_ENC_CAT0: f64 = 1045674047570.8059f64; // sklearn encodings_[0][0]
    let sk_enc_cat0_bits = SK_ENC_CAT0.to_bits();
    // cat0 (rows 0..6) then cat1 (rows 6..10); smooth = 1.0.
    let y: Array1<f64> = array![
        6653563921156.749f64,
        0.40053020400449824f64,
        -0.03752667172359179f64,
        664519602.7904022f64,
        60952871.499360375f64,
        -0.0002250432419396511f64,
        0.13607512177992703f64,
        3.9559294497604784f64,
        3.3577691973009576f64,
        3.555310355269042f64
    ];
    let x: Array2<usize> = array![[0usize], [0], [0], [0], [0], [0], [1], [1], [1], [1]];

    let fitted = TargetEncoder::<f64>::new(1.0).fit(&x, &y).unwrap();
    let ferro_cat0 = fitted.category_maps()[0][&0];

    assert_eq!(
        ferro_cat0.to_bits(),
        sk_enc_cat0_bits,
        "per-category encoding diverges: ferro cat0={} (bits {:#x}) != sklearn \
         encodings_[0][0]={} (bits {:#x})",
        ferro_cat0,
        ferro_cat0.to_bits(),
        f64::from_bits(sk_enc_cat0_bits),
        sk_enc_cat0_bits
    );
}

// ---------------------------------------------------------------------------
// FRESH HUNT (re-audit after #1262) — order-sensitive accumulation, multi-
// feature 3x3 full-matrix transform, unknown cell, and the f32 path.
// All expected values are LIVE sklearn 1.5.2 oracle bits (run from /tmp),
// never copied from ferrolearn (R-CHAR-3).
//
// Oracle session (sklearn 1.5.2, run from /tmp):
// ```text
// # HUNT-1 — per-category accumulation ORDER matters. cat0 targets in ROW
// #          ORDER are mixed-magnitude [1e15,1.0,-1e15,2.0] interleaved with cat1:
// >>> y = [1e15, 7.0, 1.0, 8.0, -1e15, 9.0, 2.0, 10.0]
// >>> X = [[0],[1],[0],[1],[0],[1],[0],[1]]
// >>> e = TargetEncoder(smooth=1.0, target_type='continuous').fit(X, y)
// >>> float(e.target_mean_)           -> 4.625
// >>> [float(v) for v in e.encodings_[0]] -> [1.525, 7.725]
// #   pyx accumulates sums[cat] in float64 in row order, seeded smooth*y_mean
// #   FIRST (pyx:60), so cat0 sum = 4.625 + 1e15 + 1.0 + (-1e15) + 2.0 = 7.625,
// #   /(1+4) = 1.525. ferrolearn does the identical f64 row-order fold -> match.
//
// # HUNT-3 — multi-feature (3 features x 3 categories), float smooth=1.5:
// >>> X = [[0,1,2],[1,0,1],[2,2,0],[0,2,1],[1,1,2],[2,0,0],[0,0,1],[1,2,2],[2,1,0]]
// >>> y = [3.5,-1.25,100.0,0.5,7.0,-3.0,2.0,50.0,-10.0]
// >>> e = TargetEncoder(smooth=1.5, target_type='continuous').fit(X, y)
// >>> float(e.target_mean_)    -> 16.52777777777778  (hex 0x1.0871c71c71c72p+4)
// >>> encodings_ (per feat)    -> see hex consts below
// >>> e.transform([[9,9,9]])   -> [[TM, TM, TM]] (all unknown -> target_mean_)
//
// # HUNT-f32 — sklearn computes encodings_ in FLOAT64 regardless of y dtype
// #            (pyx `double sums/counts`). y float32 = [2^24, 1, 1, 1], smooth=0:
// >>> y = np.array([16777216.0,1.0,1.0,1.0], dtype=np.float32)
// >>> e = TargetEncoder(smooth=0.0, target_type='continuous').fit([[0]]*4, y)
// >>> float(e.encodings_[0][0]) -> 4194304.75   (f64 accum: (2^24+1+1+1)/4)
// >>> e.transform([[0],[0],[0],[0]]).ravel() -> [4194304.75]*4
// #   ferrolearn TargetEncoder<f32> accumulates in f32: 2^24+1 == 2^24 (lost),
// #   so sum stays 2^24, /4 = 4194304.0  -> diverges by 0.75.
// ```

/// GREEN guard — HUNT-1. Per-category accumulation ORDER (row order) with
/// mixed-magnitude interleaved targets bit-matches sklearn's pyx float64
/// row-order fold seeded with `smooth*y_mean` (`_target_encoder_fast.pyx:60-69`).
/// Oracle: target_mean_=4.625, encodings_[0]=[1.525, 7.725].
#[test]
fn green_hunt1_interleaved_order_accumulation() {
    // sklearn 1.5.2 live oracle bits:
    const SK_TM: f64 = 4.625; // 0x1.2800000000000p+2
    const SK_CAT0: f64 = 1.525; // 0x1.8666666666666p+0
    const SK_CAT1: f64 = 7.725; // 0x1.ee66666666666p+2
    let y: Array1<f64> = array![1e15, 7.0, 1.0, 8.0, -1e15, 9.0, 2.0, 10.0];
    let x: Array2<usize> = array![[0], [1], [0], [1], [0], [1], [0], [1]];
    let fitted = TargetEncoder::<f64>::new(1.0).fit(&x, &y).unwrap();

    assert_eq!(
        fitted.global_mean().to_bits(),
        SK_TM.to_bits(),
        "target_mean_: ferro={} sk={}",
        fitted.global_mean(),
        SK_TM
    );
    let m0 = &fitted.category_maps()[0];
    assert_eq!(
        m0[&0].to_bits(),
        SK_CAT0.to_bits(),
        "cat0 (row-order fold): ferro={} (bits {:#x}) != sk={} (bits {:#x})",
        m0[&0],
        m0[&0].to_bits(),
        SK_CAT0,
        SK_CAT0.to_bits()
    );
    assert_eq!(
        m0[&1].to_bits(),
        SK_CAT1.to_bits(),
        "cat1: ferro={} != sk={}",
        m0[&1],
        SK_CAT1
    );
}

/// GREEN guard — HUNT-3. Multi-feature (3 features x 3 categories) float
/// smooth=1.5, FULL transform matrix bit-match + every encodings_ cell, plus
/// unknown category cell -> target_mean_ (`_target_encoder_fast.pyx:60-75`,
/// `_target_encoder.py:383` mean, transform handle_unknown="ignore").
#[test]
fn green_hunt3_multifeature_3x3_full_matrix() {
    // sklearn 1.5.2 live oracle bits (hex -> f64):
    const SK_TM: f64 = f64::from_bits(0x4030871c71c71c72); // 16.52777777777778
    // feat0
    const F0C0: f64 = f64::from_bits(0x401b5ed097b425ed);
    const F0C1: f64 = f64::from_bits(0x4031e5ed097b425f);
    const F0C2: f64 = f64::from_bits(0x4038d7b425ed097c);
    // feat1
    const F1C0: f64 = f64::from_bits(0x4014097b425ed098);
    const F1C1: f64 = f64::from_bits(0x40167b425ed097b5);
    const F1C2: f64 = f64::from_bits(0x40437a12f684bda2);
    // feat2
    const F2C0: f64 = f64::from_bits(0x4038d7b425ed097c);
    const F2C1: f64 = f64::from_bits(0x401725ed097b425f);
    const F2C2: f64 = f64::from_bits(0x4032f425ed097b43);

    let x: Array2<usize> = array![
        [0, 1, 2],
        [1, 0, 1],
        [2, 2, 0],
        [0, 2, 1],
        [1, 1, 2],
        [2, 0, 0],
        [0, 0, 1],
        [1, 2, 2],
        [2, 1, 0],
    ];
    let y: Array1<f64> = array![3.5, -1.25, 100.0, 0.5, 7.0, -3.0, 2.0, 50.0, -10.0];
    let fitted = TargetEncoder::<f64>::new(1.5).fit(&x, &y).unwrap();

    assert_eq!(fitted.global_mean().to_bits(), SK_TM.to_bits());
    let maps = fitted.category_maps();
    let enc = [[F0C0, F0C1, F0C2], [F1C0, F1C1, F1C2], [F2C0, F2C1, F2C2]];
    for f in 0..3 {
        for c in 0..3 {
            assert_eq!(
                maps[f][&c].to_bits(),
                enc[f][c].to_bits(),
                "enc[feat={f}][cat={c}]: ferro={} (bits {:#x}) != sk={} (bits {:#x})",
                maps[f][&c],
                maps[f][&c].to_bits(),
                enc[f][c],
                enc[f][c].to_bits()
            );
        }
    }

    // FULL transform matrix: each cell maps row's category through enc[feat].
    let out = fitted.transform(&x).unwrap();
    for i in 0..x.nrows() {
        for f in 0..3 {
            let cat = x[[i, f]];
            assert_eq!(
                out[[i, f]].to_bits(),
                enc[f][cat].to_bits(),
                "transform[{i}][{f}] cat={cat}: ferro={} != sk={}",
                out[[i, f]],
                enc[f][cat]
            );
        }
    }

    // Unknown category in every column -> target_mean_ (oracle: [TM, TM, TM]).
    let unk = fitted.transform(&array![[9usize, 9, 9]]).unwrap();
    for f in 0..3 {
        assert_eq!(
            unk[[0, f]].to_bits(),
            SK_TM.to_bits(),
            "unknown[{f}]: ferro={} != sk target_mean_={}",
            unk[[0, f]],
            SK_TM
        );
    }
}
