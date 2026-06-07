//! ACToR divergence + oracle-grounded guard tests for `Normalizer`.
//!
//! Mirrors scikit-learn 1.5.2 `sklearn/preprocessing/_data.py`:
//!   - free fn `normalize` (`:1866`)
//!   - `class Normalizer` (`:1980`)
//!
//! ALL expected values come from a LIVE sklearn 1.5.2 call run from /tmp
//! (recorded in each test's doc comment), never copied from the ferrolearn
//! side (R-CHAR-3).
//!
//! Two kinds of tests:
//!   * GREEN GUARDS (REQ-1) — assert ferrolearn already matches the live
//!     oracle for the dense row-wise L1/L2/Max transform + zero-row-unchanged.
//!   * FAILING PINS (REQ-2) — assert the input-validation behavior sklearn's
//!     `check_array` (`force_all_finite=True`, min-samples/min-features = 1)
//!     enforces; ferrolearn's `transform` does no validation so these FAIL
//!     today (the orchestrator fixes this iteration).

use ferrolearn_core::error::FerroError;
use ferrolearn_core::traits::{Fit, Transform};
use ferrolearn_preprocess::Normalizer;
use ferrolearn_preprocess::normalizer::{FittedNormalizer, NormType};
use ndarray::{Array2, array};

// ===========================================================================
// REQ-1 — oracle-grounded GREEN guards
// ===========================================================================

/// Oracle (live sklearn 1.5.2, run from /tmp):
/// `normalize([[-2.,1.,2.],[-1.,0.,1.]], norm='l1').tolist()`
/// -> `[[-0.4, 0.2, 0.4], [-0.5, 0.0, 0.5]]`
#[test]
fn guard_l1_matches_oracle() {
    const SK: [[f64; 3]; 2] = [[-0.4, 0.2, 0.4], [-0.5, 0.0, 0.5]];
    let x = array![[-2.0, 1.0, 2.0], [-1.0, 0.0, 1.0]];
    let out = Normalizer::<f64>::l1().transform(&x).unwrap();
    assert_eq!(out.dim(), (2, 3));
    for i in 0..2 {
        for j in 0..3 {
            assert!(
                (out[[i, j]] - SK[i][j]).abs() < 1e-12,
                "l1[{i},{j}]: ferro={} sklearn={}",
                out[[i, j]],
                SK[i][j]
            );
        }
    }
}

/// Oracle (live sklearn 1.5.2, run from /tmp):
/// `normalize([[-2.,1.,2.],[-1.,0.,1.]], norm='l2').tolist()`
/// -> `[[-0.6666666666666666, 0.3333333333333333, 0.6666666666666666],
///      [-0.7071067811865475, 0.0, 0.7071067811865475]]`
///
/// Also confirms ferrolearn's L2 (`sqrt(sum v^2)`) agrees with sklearn's
/// `row_norms` (einsum) to 1e-12: a live `row_norms` vs naive-sqrt comparison
/// on this fixture showed maxdiff 0.0, so no L2-method divergence to pin.
#[test]
fn guard_l2_matches_oracle() {
    const SK: [[f64; 3]; 2] = [
        [
            -0.666_666_666_666_666_6,
            0.333_333_333_333_333_3,
            0.666_666_666_666_666_6,
        ],
        [-0.707_106_781_186_547_5, 0.0, 0.707_106_781_186_547_5],
    ];
    let x = array![[-2.0, 1.0, 2.0], [-1.0, 0.0, 1.0]];
    let out = Normalizer::<f64>::l2().transform(&x).unwrap();
    assert_eq!(out.dim(), (2, 3));
    for i in 0..2 {
        for j in 0..3 {
            assert!(
                (out[[i, j]] - SK[i][j]).abs() < 1e-12,
                "l2[{i},{j}]: ferro={} sklearn={}",
                out[[i, j]],
                SK[i][j]
            );
        }
    }
}

/// Oracle (live sklearn 1.5.2, run from /tmp):
/// `Normalizer(norm='max').transform([[-5.,3.,1.]]).tolist()`
/// -> `[[-1.0, 0.6, 0.2]]`
#[test]
fn guard_max_matches_oracle() {
    const SK: [f64; 3] = [-1.0, 0.6, 0.2];
    let x = array![[-5.0, 3.0, 1.0]];
    let out = Normalizer::<f64>::new(NormType::Max).transform(&x).unwrap();
    assert_eq!(out.dim(), (1, 3));
    for j in 0..3 {
        assert!(
            (out[[0, j]] - SK[j]).abs() < 1e-12,
            "max[0,{j}]: ferro={} sklearn={}",
            out[[0, j]],
            SK[j]
        );
    }
}

/// Oracle (live sklearn 1.5.2, run from /tmp):
/// `Normalizer(norm='l2').transform([[0.,0.,0.],[3.,4.,0.]]).tolist()`
/// -> `[[0.0, 0.0, 0.0], [0.6, 0.8, 0.0]]`
/// (`_handle_zeros_in_scale` maps the zero divisor to 1, `_data.py:1968`).
#[test]
fn guard_zero_row_unchanged_matches_oracle() {
    const SK: [[f64; 3]; 2] = [[0.0, 0.0, 0.0], [0.6, 0.8, 0.0]];
    let x = array![[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]];
    let out = Normalizer::<f64>::l2().transform(&x).unwrap();
    for i in 0..2 {
        for j in 0..3 {
            assert!(
                (out[[i, j]] - SK[i][j]).abs() < 1e-12,
                "zero[{i},{j}]: ferro={} sklearn={}",
                out[[i, j]],
                SK[i][j]
            );
        }
    }
}

/// Oracle (live sklearn 1.5.2, run from /tmp):
/// `normalize(np.array([[3.,4.]],dtype=np.float32), norm='l2').tolist()`
/// -> `[[0.6000000238418579, 0.800000011920929]]`
#[test]
fn guard_f32_l2_matches_oracle() {
    // 0.6_f32 == 0.6000000238418579, 0.8_f32 == 0.800000011920929 (the live-oracle f32 values, bit-exact).
    const SK: [f32; 2] = [0.6, 0.8];
    let x: Array2<f32> = array![[3.0f32, 4.0]];
    let out = Normalizer::<f32>::l2().transform(&x).unwrap();
    for j in 0..2 {
        assert!(
            (out[[0, j]] - SK[j]).abs() < 1e-6,
            "f32[0,{j}]: ferro={} sklearn={}",
            out[[0, j]],
            SK[j]
        );
    }
}

// ===========================================================================
// REQ-2 — input-validation DIVERGENCE (FAILING pins, un-ignored)
//
// sklearn `Normalizer.transform` -> `normalize` -> `check_array`
// (`_data.py:1933-1940`) with default `force_all_finite=True` and default
// `ensure_min_samples=1` / `ensure_min_features=1`. Each input below makes the
// LIVE oracle raise `ValueError`; ferrolearn's `transform` does NO validation
// and returns `Ok`, so each assertion FAILS today.
//
// check_array order (confirmed live; mirrors binarizer.rs REQ-9):
//   min-samples BEFORE min-features BEFORE finite. So:
//     (0,3) -> samples error, (2,0) -> features error, (0,0) -> samples error.
// Expected FerroError mapping (consistent with binarizer.rs):
//   zero samples  -> FerroError::InsufficientSamples
//   zero features -> FerroError::InvalidParameter
//   non-finite    -> FerroError::InvalidParameter
// ===========================================================================

/// Divergence: `Normalizer::transform` diverges from
/// `sklearn/preprocessing/_data.py:1933` (`check_array`,
/// `force_all_finite=True`) for a NaN-containing row.
///
/// Oracle (live sklearn 1.5.2, run from /tmp):
/// `Normalizer().transform(np.array([[float('nan'),1.0]]))`
/// -> `ValueError: Input X contains NaN. Normalizer does not accept missing
///     values encoded as NaN natively...`
///
/// ferrolearn returns `Ok` with a NaN row.
/// Tracking: see crate report / blocker for NORM-VALIDATE.
#[test]
fn divergence_transform_rejects_nan() {
    let x = array![[f64::NAN, 1.0]];
    let result = Normalizer::<f64>::l2().transform(&x);
    assert!(
        result.is_err(),
        "sklearn raises ValueError on NaN input; ferrolearn returned Ok"
    );
}

/// Divergence: `Normalizer::transform` diverges from
/// `sklearn/preprocessing/_data.py:1933` for a +inf-containing row.
///
/// Oracle (live sklearn 1.5.2, run from /tmp):
/// `Normalizer().transform(np.array([[float('inf'),1.0]]))`
/// -> `ValueError: Input X contains infinity or a value too large for
///     dtype('float64').`
///
/// ferrolearn returns `Ok` with an inf row.
/// Tracking: see crate report / blocker for NORM-VALIDATE.
#[test]
fn divergence_transform_rejects_pos_inf() {
    let x = array![[f64::INFINITY, 1.0]];
    let result = Normalizer::<f64>::l2().transform(&x);
    assert!(
        result.is_err(),
        "sklearn raises ValueError on +inf input; ferrolearn returned Ok"
    );
}

/// Divergence: `Normalizer::transform` diverges from
/// `sklearn/preprocessing/_data.py:1933` for a -inf-containing row.
///
/// Oracle (live sklearn 1.5.2, run from /tmp):
/// `Normalizer().transform(np.array([[float('-inf'),1.0]]))`
/// -> `ValueError: Input X contains infinity or a value too large for
///     dtype('float64').`
///
/// ferrolearn returns `Ok` with a -inf row.
/// Tracking: see crate report / blocker for NORM-VALIDATE.
#[test]
fn divergence_transform_rejects_neg_inf() {
    let x = array![[f64::NEG_INFINITY, 1.0]];
    let result = Normalizer::<f64>::l2().transform(&x);
    assert!(
        result.is_err(),
        "sklearn raises ValueError on -inf input; ferrolearn returned Ok"
    );
}

/// Divergence: `Normalizer::transform` diverges from
/// `sklearn/utils/validation.py:1084` (`ensure_min_samples=1`) for a
/// zero-sample `(0, 3)` array.
///
/// Oracle (live sklearn 1.5.2, run from /tmp):
/// `Normalizer().transform(np.zeros((0,3)))`
/// -> `ValueError: Found array with 0 sample(s) (shape=(0, 3)) while a minimum
///     of 1 is required by Normalizer.`
///
/// ferrolearn returns `Ok` of an empty array.
/// Expected mapping: `FerroError::InsufficientSamples` (samples checked first).
/// Tracking: see crate report / blocker for NORM-VALIDATE.
#[test]
fn divergence_transform_rejects_zero_samples() {
    let x: Array2<f64> = Array2::zeros((0, 3));
    let result = Normalizer::<f64>::l2().transform(&x);
    assert!(
        result.is_err(),
        "sklearn raises ValueError (0 samples); ferrolearn returned Ok"
    );
    assert!(
        matches!(result, Err(FerroError::InsufficientSamples { .. })),
        "zero-sample input should map to FerroError::InsufficientSamples"
    );
}

/// Divergence: `Normalizer::transform` diverges from
/// `sklearn/utils/validation.py:1093` (`ensure_min_features=1`) for a
/// zero-feature `(2, 0)` array.
///
/// Oracle (live sklearn 1.5.2, run from /tmp):
/// `Normalizer().transform(np.zeros((2,0)))`
/// -> `ValueError: Found array with 0 feature(s) (shape=(2, 0)) while a minimum
///     of 1 is required by Normalizer.`
///
/// ferrolearn returns `Ok` of the `(2, 0)` array.
/// Expected mapping: `FerroError::InvalidParameter` (features after samples).
/// Tracking: see crate report / blocker for NORM-VALIDATE.
#[test]
fn divergence_transform_rejects_zero_features() {
    let x: Array2<f64> = Array2::zeros((2, 0));
    let result = Normalizer::<f64>::l2().transform(&x);
    assert!(
        result.is_err(),
        "sklearn raises ValueError (0 features); ferrolearn returned Ok"
    );
    assert!(
        matches!(result, Err(FerroError::InvalidParameter { .. })),
        "zero-feature input should map to FerroError::InvalidParameter"
    );
}

/// Divergence: `Normalizer::transform` order check — for a fully-empty
/// `(0, 0)` array sklearn checks min-samples BEFORE min-features, so it reports
/// the SAMPLES error.
///
/// Oracle (live sklearn 1.5.2, run from /tmp):
/// `Normalizer().transform(np.zeros((0,0)))`
/// -> `ValueError: Found array with 0 sample(s) (shape=(0, 0)) while a minimum
///     of 1 is required by Normalizer.`
///
/// ferrolearn returns `Ok`. Pins the sklearn ordering so the fixer reports
/// `InsufficientSamples` (not the feature error) for `(0, 0)`.
/// Tracking: see crate report / blocker for NORM-VALIDATE.
#[test]
fn divergence_transform_empty_reports_samples_first() {
    let x: Array2<f64> = Array2::zeros((0, 0));
    let result = Normalizer::<f64>::l2().transform(&x);
    assert!(
        matches!(result, Err(FerroError::InsufficientSamples { .. })),
        "(0,0) should report samples error first (sklearn check_array order)"
    );
}

// ===========================================================================
// CONVERGENCE GREEN GUARDS — no over-rejection of finite, non-empty,
// >=1-feature input (#1140 re-audit). Each pins a LIVE sklearn 1.5.2 oracle
// value that the new transform guards must NOT reject. All currently GREEN.
// ===========================================================================

/// Oracle (live sklearn 1.5.2, run from /tmp):
/// `Normalizer().transform([[0.,0.,0.],[3.,4.,0.]]).tolist()`
/// -> `[[0.0, 0.0, 0.0], [0.6, 0.8, 0.0]]`
///
/// A finite all-zero ROW (zero NORM, not zero shape) in a (2,3) array is
/// normalized by leaving it unchanged via `_handle_zeros_in_scale`
/// (`sklearn/preprocessing/_data.py:1968`) — it is NOT an error. The new
/// `nrows()==0` guard must not fire (nrows=2), so ferrolearn returns Ok and
/// matches the oracle. Pins that the zero-NORM-row case is not over-rejected.
#[test]
fn guard_zero_norm_row_not_rejected_matches_oracle() {
    const SK: [[f64; 3]; 2] = [[0.0, 0.0, 0.0], [0.6, 0.8, 0.0]];
    let x = array![[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]];
    let out = Normalizer::<f64>::l2()
        .transform(&x)
        .expect("zero-NORM row in a (2,3) array must NOT be rejected by guards");
    assert_eq!(out.dim(), (2, 3));
    for i in 0..2 {
        for j in 0..3 {
            assert!(
                (out[[i, j]] - SK[i][j]).abs() < 1e-12,
                "zeronorm[{i},{j}]: ferro={} sklearn={}",
                out[[i, j]],
                SK[i][j]
            );
        }
    }
}

/// Oracle (live sklearn 1.5.2, run from /tmp):
/// `normalize(np.array([[1e308, 0.0]]), norm='max').tolist()` -> `[[1.0, 0.0]]`
///
/// A large-but-finite value (1e308) is accepted by sklearn `check_array`
/// (`force_all_finite=True` rejects only NaN/inf, `sklearn/utils/validation.py:1064`)
/// — 1e308 is finite, so it is NOT rejected. Confirms the new finite guard does
/// not over-reject large-finite input.
#[test]
fn guard_large_finite_not_rejected_matches_oracle() {
    const SK: [f64; 2] = [1.0, 0.0];
    let x = array![[1e308, 0.0]];
    let out = Normalizer::<f64>::max()
        .transform(&x)
        .expect("large-but-finite 1e308 must NOT be rejected by the finite guard");
    for j in 0..2 {
        assert!(
            (out[[0, j]] - SK[j]).abs() < 1e-12,
            "largefinite[0,{j}]: ferro={} sklearn={}",
            out[[0, j]],
            SK[j]
        );
    }
}

/// Oracle (live sklearn 1.5.2, run from /tmp):
/// `normalize(np.array([[5e-324, 0.0]]), norm='l2').tolist()` -> `[[5e-324, 0.0]]`
///
/// A subnormal value (smallest positive f64 denormal, 5e-324) is finite, so
/// sklearn accepts it; here its L2 norm equals itself, so normalize leaves it
/// at 5e-324. Confirms the finite guard does not over-reject subnormals.
#[test]
fn guard_subnormal_not_rejected_matches_oracle() {
    const SK: [f64; 2] = [5e-324, 0.0];
    let x = array![[5e-324, 0.0]];
    let out = Normalizer::<f64>::l2()
        .transform(&x)
        .expect("subnormal 5e-324 is finite and must NOT be rejected");
    assert!(
        (out[[0, 0]] - SK[0]).abs() <= f64::from_bits(1),
        "subnormal[0,0]: ferro={} sklearn={}",
        out[[0, 0]],
        SK[0]
    );
    assert_eq!(out[[0, 1]], SK[1]);
}

/// Oracle (live sklearn 1.5.2, run from /tmp):
/// `Normalizer().transform([[-0.0, -0.0]]).tolist()` -> `[[-0.0, -0.0]]`
///
/// A row of negative zeros has L2 norm 0.0, so `_handle_zeros_in_scale` leaves
/// it unchanged; `-0.0` is finite, so it is NOT rejected. Confirms the finite
/// guard does not over-reject signed zero.
#[test]
fn guard_neg_zero_not_rejected_matches_oracle() {
    let x = array![[-0.0_f64, -0.0_f64]];
    let out = Normalizer::<f64>::l2()
        .transform(&x)
        .expect("-0.0 is finite and must NOT be rejected");
    // sklearn returns the row unchanged (zero-norm); values are zero-valued.
    assert_eq!(out[[0, 0]], 0.0);
    assert_eq!(out[[0, 1]], 0.0);
}

// ===========================================================================
// REQ-3 / REQ-5 / REQ-6 — stateful `Fit` -> `FittedNormalizer` path.
//
// sklearn `Normalizer(norm, copy).fit(X).transform(X)`:
//   - fit "Only validates estimator's parameters" + records n_features_in_
//     (`_data.py:2062-2083`); _validate_data default force_all_finite=True
//     REJECTS NaN/±inf.
//   - transform delegates to `normalize(X, norm=self.norm, axis=1)` (`:2106`),
//     so the fitted path is bit-identical to the stateless `Normalizer.transform`.
//
// Shared fixture X = [[1,2,2],[0,3,4],[-5,3,1]]. ALL expected values come from a
// LIVE sklearn 1.5.2 call run from /tmp (recorded in each test's doc comment),
// never copied from the ferrolearn side (R-CHAR-3).
// ===========================================================================

/// REQ-3 / REQ-6 oracle (live sklearn 1.5.2, run from /tmp):
/// ```text
/// python3 -c "import numpy as np; from sklearn.preprocessing import Normalizer
/// X=np.array([[1.,2.,2.],[0.,3.,4.],[-5.,3.,1.]])
/// print(Normalizer(norm='l1').fit(X).transform(X).tolist())
/// print(Normalizer(norm='l2').fit(X).transform(X).tolist())
/// print(Normalizer(norm='max').fit(X).transform(X).tolist())"
/// ```
/// l1  -> [[0.2,0.4,0.4],[0.0,0.42857142857142855,0.5714285714285714],
///         [-0.5555555555555556,0.3333333333333333,0.1111111111111111]]
/// l2  -> [[0.3333333333333333,0.6666666666666666,0.6666666666666666],
///         [0.0,0.6,0.8],
///         [-0.8451542547285166,0.50709255283711,0.1690308509457033]]
/// max -> [[0.5,1.0,1.0],[0.0,0.75,1.0],[-1.0,0.6,0.2]]
///
/// The fitted-path output must equal the live oracle AND the stateless
/// `Normalizer::transform` output (the Fit path must not diverge).
#[test]
fn fit_l1_l2_max_matches_oracle_and_stateless() {
    let x = array![[1.0, 2.0, 2.0], [0.0, 3.0, 4.0], [-5.0, 3.0, 1.0]];

    let l1: [[f64; 3]; 3] = [
        [0.2, 0.4, 0.4],
        [0.0, 0.428_571_428_571_428_55, 0.571_428_571_428_571_4],
        [
            -0.555_555_555_555_555_6,
            0.333_333_333_333_333_3,
            0.111_111_111_111_111_1,
        ],
    ];
    let l2: [[f64; 3]; 3] = [
        [
            0.333_333_333_333_333_3,
            0.666_666_666_666_666_6,
            0.666_666_666_666_666_6,
        ],
        [0.0, 0.6, 0.8],
        [
            -0.845_154_254_728_516_6,
            0.507_092_552_837_11,
            0.169_030_850_945_703_3,
        ],
    ];
    let max: [[f64; 3]; 3] = [[0.5, 1.0, 1.0], [0.0, 0.75, 1.0], [-1.0, 0.6, 0.2]];

    for (norm, sk) in [(NormType::L1, l1), (NormType::L2, l2), (NormType::Max, max)] {
        let fitted: FittedNormalizer<f64> = Normalizer::<f64>::new(norm).fit(&x, &()).unwrap();
        let fit_out = fitted.transform(&x).unwrap();
        let stateless_out = Normalizer::<f64>::new(norm).transform(&x).unwrap();
        assert_eq!(fit_out.dim(), (3, 3));
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (fit_out[[i, j]] - sk[i][j]).abs() < 1e-12,
                    "fit {norm:?}[{i},{j}]: ferro={} sklearn={}",
                    fit_out[[i, j]],
                    sk[i][j]
                );
                // Fit path must be bit-identical to the stateless path.
                assert_eq!(
                    fit_out[[i, j]].to_bits(),
                    stateless_out[[i, j]].to_bits(),
                    "fit-path != stateless-path at {norm:?}[{i},{j}]"
                );
            }
        }
    }
}

/// REQ-3 / REQ-6 oracle (live sklearn 1.5.2, run from /tmp):
/// ```text
/// python3 -c "import numpy as np; from sklearn.preprocessing import Normalizer
/// print(Normalizer().fit(np.array([[1.,2.,2.],[0.,3.,4.],[-5.,3.,1.]])).n_features_in_)"
/// ```
/// -> `3`
#[test]
fn fit_n_features_in_matches_ncols() {
    let x = array![[1.0, 2.0, 2.0], [0.0, 3.0, 4.0], [-5.0, 3.0, 1.0]];
    let fitted = Normalizer::<f64>::l2().fit(&x, &()).unwrap();
    assert_eq!(fitted.n_features_in(), 3);
    // 2-column fit records 2.
    let x2 = array![[1.0, 2.0], [3.0, 4.0]];
    assert_eq!(
        Normalizer::<f64>::l2()
            .fit(&x2, &())
            .unwrap()
            .n_features_in(),
        2
    );
}

/// REQ-3 oracle (live sklearn 1.5.2, run from /tmp):
/// ```text
/// python3 -c "import numpy as np; from sklearn.preprocessing import Normalizer
/// Normalizer().fit(np.array([[float('nan'),1.0]]))"
/// ```
/// -> `ValueError: Input X contains NaN. Normalizer does not accept missing
///     values encoded as NaN natively...`
///
/// sklearn `fit` -> `_validate_data` default `force_all_finite=True` REJECTS
/// NaN; ferrolearn `fit` must return Err.
#[test]
fn fit_rejects_nan() {
    let x = array![[f64::NAN, 1.0]];
    assert!(
        Normalizer::<f64>::l2().fit(&x, &()).is_err(),
        "sklearn Normalizer().fit raises ValueError on NaN; ferrolearn fit must Err"
    );
}

/// REQ-3 oracle (live sklearn 1.5.2, run from /tmp):
/// ```text
/// python3 -c "import numpy as np; from sklearn.preprocessing import Normalizer
/// Normalizer().fit(np.array([[float('inf'),1.0]]))"
/// ```
/// -> `ValueError: Input X contains infinity or a value too large for
///     dtype('float64').`
#[test]
fn fit_rejects_pos_inf() {
    let x = array![[f64::INFINITY, 1.0]];
    assert!(
        Normalizer::<f64>::l2().fit(&x, &()).is_err(),
        "sklearn Normalizer().fit raises ValueError on +inf; ferrolearn fit must Err"
    );
}

/// REQ-3 oracle (live sklearn 1.5.2, run from /tmp):
/// ```text
/// python3 -c "import numpy as np; from sklearn.preprocessing import Normalizer
/// Normalizer().fit(np.array([[float('-inf'),1.0]]))"
/// ```
/// -> `ValueError: Input X contains infinity or a value too large for
///     dtype('float64').`
#[test]
fn fit_rejects_neg_inf() {
    let x = array![[f64::NEG_INFINITY, 1.0]];
    assert!(
        Normalizer::<f64>::l2().fit(&x, &()).is_err(),
        "sklearn Normalizer().fit raises ValueError on -inf; ferrolearn fit must Err"
    );
}

/// REQ-5 oracle (live sklearn 1.5.2, run from /tmp):
/// ```text
/// python3 -c "import numpy as np; from sklearn.preprocessing import Normalizer
/// X=np.array([[1.,2.,2.],[0.,3.,4.],[-5.,3.,1.]])
/// a=Normalizer(norm='l2',copy=True).fit(X).transform(X)
/// b=Normalizer(norm='l2',copy=False).fit(X).transform(X.copy())
/// print(np.array_equal(a,b))"
/// ```
/// -> `True`
///
/// `copy` is an ACCEPT-AND-DOCUMENT no-op in ferrolearn: `copy=True` and
/// `copy=False` produce bit-identical output.
#[test]
fn fit_copy_true_false_identical() {
    let x = array![[1.0, 2.0, 2.0], [0.0, 3.0, 4.0], [-5.0, 3.0, 1.0]];
    let a = Normalizer::<f64>::l2()
        .with_copy(true)
        .fit(&x, &())
        .unwrap();
    let b = Normalizer::<f64>::l2()
        .with_copy(false)
        .fit(&x, &())
        .unwrap();
    assert!(a.copy());
    assert!(!b.copy());
    let out_a = a.transform(&x).unwrap();
    let out_b = b.transform(&x).unwrap();
    for (va, vb) in out_a.iter().zip(out_b.iter()) {
        assert_eq!(va.to_bits(), vb.to_bits(), "copy flag changed the output");
    }
}

/// REQ-1-through-Fit oracle (live sklearn 1.5.2, run from /tmp):
/// ```text
/// python3 -c "import numpy as np; from sklearn.preprocessing import Normalizer
/// Z=np.array([[0.,0.,0.],[3.,4.,0.]])
/// print(Normalizer(norm='l2').fit(Z).transform(Z).tolist())"
/// ```
/// -> `[[0.0, 0.0, 0.0], [0.6, 0.8, 0.0]]`
///
/// A zero-NORM row is left unchanged through the Fit path (REQ-1 preserved via
/// `_handle_zeros_in_scale`, `_data.py:1968`).
#[test]
fn fit_zero_row_unchanged() {
    const SK: [[f64; 3]; 2] = [[0.0, 0.0, 0.0], [0.6, 0.8, 0.0]];
    let z = array![[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]];
    let out = Normalizer::<f64>::l2()
        .fit(&z, &())
        .unwrap()
        .transform(&z)
        .unwrap();
    for i in 0..2 {
        for j in 0..3 {
            assert!(
                (out[[i, j]] - SK[i][j]).abs() < 1e-12,
                "fit zero[{i},{j}]: ferro={} sklearn={}",
                out[[i, j]],
                SK[i][j]
            );
        }
    }
}

/// REQ-6 oracle (live sklearn 1.5.2, run from /tmp):
/// ```text
/// python3 -c "import numpy as np; from sklearn.preprocessing import Normalizer
/// m=Normalizer().fit(np.ones((2,3)))
/// m.transform(np.ones((2,5)))"
/// ```
/// -> `ValueError: X has 5 features, but Normalizer is expecting 3 features as
///     input.`
///
/// sklearn `transform` -> `_validate_data(reset=False)` (`_data.py:2104`) checks
/// the column count against the fitted `n_features_in_`. ferrolearn maps the
/// mismatch to `FerroError::ShapeMismatch`.
#[test]
fn fitted_transform_shape_mismatch() {
    let x_fit = array![[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]];
    let fitted = Normalizer::<f64>::l2().fit(&x_fit, &()).unwrap();
    let x_wrong = array![[1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0]];
    let result = fitted.transform(&x_wrong);
    assert!(
        matches!(result, Err(FerroError::ShapeMismatch { .. })),
        "wrong column count after fit should map to FerroError::ShapeMismatch"
    );
}

/// REQ-3 — the Fit path equals the stateless path equals the `normalize` free
/// fn on a richer fixture, bit-for-bit (no divergence between the three public
/// entry points). Oracle (live sklearn 1.5.2, run from /tmp):
/// ```text
/// python3 -c "import numpy as np; from sklearn.preprocessing import Normalizer, normalize
/// X=np.array([[-2.,1.,2.],[-1.,0.,1.],[10.,-3.,0.5]])
/// print(np.array_equal(Normalizer(norm='l2').fit(X).transform(X), normalize(X, norm='l2')))"
/// ```
/// -> `True`
#[test]
fn fit_path_equals_stateless_path() {
    let x = array![[-2.0, 1.0, 2.0], [-1.0, 0.0, 1.0], [10.0, -3.0, 0.5]];
    for norm in [NormType::L1, NormType::L2, NormType::Max] {
        let fit_out = Normalizer::<f64>::new(norm)
            .fit(&x, &())
            .unwrap()
            .transform(&x)
            .unwrap();
        let stateless_out = Normalizer::<f64>::new(norm).transform(&x).unwrap();
        for (a, b) in fit_out.iter().zip(stateless_out.iter()) {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "fit-path diverged from stateless-path for {norm:?}"
            );
        }
    }
}

/// REQ-3 — f32 Fit path matches the stateless f32 path. Oracle (live sklearn
/// 1.5.2, run from /tmp):
/// ```text
/// python3 -c "import numpy as np; from sklearn.preprocessing import Normalizer
/// X=np.array([[3.,4.]],dtype=np.float32)
/// print(Normalizer(norm='l2').fit(X).transform(X).tolist())"
/// ```
/// -> `[[0.6000000238418579, 0.800000011920929]]`
#[test]
fn fit_f32_matches_oracle() {
    const SK: [f32; 2] = [0.6, 0.8];
    let x: Array2<f32> = array![[3.0f32, 4.0]];
    let out = Normalizer::<f32>::l2()
        .fit(&x, &())
        .unwrap()
        .transform(&x)
        .unwrap();
    for j in 0..2 {
        assert!(
            (out[[0, j]] - SK[j]).abs() < 1e-6,
            "fit f32[0,{j}]: ferro={} sklearn={}",
            out[[0, j]],
            SK[j]
        );
    }
}
