//! Divergence pins for `RidgeClassifierCV` vs scikit-learn 1.5.2
//! `sklearn.linear_model.RidgeClassifierCV`.
//!
//! Each test constructs an input scikit-learn handles a specific, documented
//! way and asserts that observable behavior against the live sklearn 1.5.2
//! oracle (R-CHAR-3: expected values are oracle-derived or sklearn
//! `file:line` symbolic, NEVER copied from the ferrolearn side). The tests are
//! `#[ignore]`d with their tracking issue; un-ignore once the generator fixes
//! the divergence.

use ferrolearn_core::error::FerroError;
use ferrolearn_core::traits::{Fit, Predict};
use ferrolearn_linear::RidgeClassifierCV;
use ndarray::{Array2, array};

/// Divergence: `RidgeClassifierCV::fit` accepts non-finite `X` on the
/// eigen/wide path (`n_samples <= n_features`) and returns `Ok(_)` with NaN
/// coefficients, whereas sklearn validates up front and RAISES.
///
/// sklearn cite: `RidgeClassifierCV.fit` → `_BaseRidge._prepare_data` calls
/// `self._validate_data(..., force_all_finite=True[default], ...)`
/// (`sklearn/linear_model/_ridge.py:1291`), so any non-finite entry raises
/// `ValueError("Input X contains infinity or a value too large ...")`.
///
/// Live oracle (sklearn 1.5.2):
/// ```text
/// python3 -c "import numpy as np; from sklearn.linear_model import RidgeClassifierCV; \
///   Xw=np.array([[1.,2.,3.,np.inf],[2.,1.,0.,1.],[3.,1.,1.,2.]]); yw=np.array([0,0,1]); \
///   RidgeClassifierCV(alphas=[0.1,1.,10.]).fit(Xw,yw)"
///   -> ValueError: Input X contains infinity or a value too large for dtype('float64').
/// ```
///
/// ferrolearn actual (`ridge_classifier_cv.rs:355` `select_alpha_gcv` →
/// `gcv_scores_eigen`, NO non-finite guard): returns `Ok(_)` whose `coef_` is
/// `[[NaN, NaN, NaN, NaN]]`. A clean otherwise-identical fit (the same `X` with
/// the `inf` replaced by `0.5`) succeeds, so the data is valid apart from the
/// non-finite entry — the divergence is purely the missing finite check.
///
/// Tracking: #2246
#[test]
fn divergence_nonfinite_x_wide_eigen_accepts_nan() {
    // n_samples (3) <= n_features (4) → eigen/wide GCV path.
    let x = Array2::from_shape_vec(
        (3, 4),
        vec![
            1.0,
            2.0,
            3.0,
            f64::INFINITY,
            2.0,
            1.0,
            0.0,
            1.0,
            3.0,
            1.0,
            1.0,
            2.0,
        ],
    )
    .unwrap();
    let y = array![0usize, 0, 1];

    let model = RidgeClassifierCV::<f64>::new().with_alphas(vec![0.1, 1.0, 10.0]);
    let result = model.fit(&x, &y);

    // sklearn raises ValueError on the non-finite entry; ferrolearn must reject
    // too. Today it returns Ok with NaN coefficients → this assertion FAILS.
    assert!(
        result.is_err(),
        "sklearn raises ValueError on non-finite X (_ridge.py:1291); ferrolearn must reject, \
         got Ok with coef_ = {:?}",
        result.map(|f| f.coefficients().clone())
    );
}

/// Divergence: `RidgeClassifierCV::fit` accepts non-finite `X` on the SVD path
/// (`n_samples > n_features`) by surfacing an INCIDENTAL linalg-convergence
/// error rather than a clean finite-validation rejection.
///
/// This pins the observable-error divergence: sklearn raises a `ValueError`
/// from input validation (`_ridge.py:1291`) BEFORE any decomposition; ferrolearn
/// has no finite guard and instead lets the SVD fail, returning a
/// `NumericalInstability { message: "... SVD failed to converge ..." }`. Both
/// are `Err`, but the SVD path masks the missing validation — kept distinct
/// from the wide-path `Ok(NaN)` pin so the generator adds the up-front check
/// that handles BOTH paths uniformly (R-CODE-2 / R-DEV-2).
///
/// Live oracle (sklearn 1.5.2):
/// ```text
/// python3 -c "import numpy as np; from sklearn.linear_model import RidgeClassifierCV; \
///   X=np.array([[1.,2.],[2.,1.],[3.,1.],[1.,np.nan],[6.,5.],[5.,6.]]); y=np.array([0,0,0,0,1,1]); \
///   RidgeClassifierCV(alphas=[0.1,1.,10.]).fit(X,y)"
///   -> ValueError: Input X contains NaN.
/// ```
///
/// Tracking: #2246
#[test]
fn divergence_nonfinite_x_svd_incidental_error() {
    // n_samples (6) > n_features (2) → SVD GCV path.
    let x = Array2::from_shape_vec(
        (6, 2),
        vec![
            1.0,
            2.0,
            2.0,
            1.0,
            3.0,
            1.0,
            1.0,
            f64::NAN,
            6.0,
            5.0,
            5.0,
            6.0,
        ],
    )
    .unwrap();
    let y = array![0usize, 0, 0, 0, 1, 1];

    let model = RidgeClassifierCV::<f64>::new().with_alphas(vec![0.1, 1.0, 10.0]);
    let err = model
        .fit(&x, &y)
        .expect_err("non-finite X must be rejected");

    // sklearn raises a clean input-validation error. ferrolearn should reject
    // with a validation/parameter-style error (mirroring `Input X contains
    // NaN`), NOT an incidental linalg "SVD failed to converge". This assertion
    // FAILS today: the error is `NumericalInstability { SVD failed ... }`.
    let is_validation_style = matches!(err, FerroError::InvalidParameter { .. });
    assert!(
        is_validation_style,
        "non-finite X should yield a clean validation error mirroring sklearn's \
         `Input X contains NaN` (_ridge.py:1291), got incidental {err:?}"
    );
}

/// Divergence: `RidgeClassifierCV::fit` accepts `alphas` containing `0.0`,
/// whereas sklearn rejects any non-strictly-positive alpha on the GCV path.
///
/// sklearn cite: when `cv is None` (the GCV path — ferrolearn's only path),
/// `_BaseRidgeCV.fit` validates each alpha with
/// `include_boundaries="neither"` over `[0, +inf)`
/// (`sklearn/linear_model/_ridge.py:2354-2360`, constraint
/// `Interval(Real, 0, None, closed="neither")` at `:2259`):
/// > `# _RidgeGCV does not work for alpha = 0`
///
/// Live oracle (sklearn 1.5.2):
/// ```text
/// python3 -c "import numpy as np; from sklearn.linear_model import RidgeClassifierCV; \
///   X=np.array([[1,2],[2,1],[3,1],[1,3],[2,2],[6,5],[5,6],[7,7]],float); \
///   y=np.array([0,0,0,0,0,1,1,1]); RidgeClassifierCV(alphas=[0.0,1.0,10.0]).fit(X,y)"
///   -> ValueError: alphas[0] == 0.0, must be > 0.0.
/// ```
///
/// ferrolearn actual (`ridge_classifier_cv.rs:242`): the alphas guard only
/// rejects `a < 0`, so `alpha == 0.0` passes and `fit` returns `Ok(_)`
/// (selecting `alpha_ == 10.0` on this fixture).
///
/// Tracking: #2247
#[test]
fn divergence_alphas_zero_accepted() {
    let x = Array2::from_shape_vec(
        (8, 2),
        vec![
            1.0, 2.0, 2.0, 1.0, 3.0, 1.0, 1.0, 3.0, 2.0, 2.0, 6.0, 5.0, 5.0, 6.0, 7.0, 7.0,
        ],
    )
    .unwrap();
    let y = array![0usize, 0, 0, 0, 0, 1, 1, 1];

    let model = RidgeClassifierCV::<f64>::new().with_alphas(vec![0.0, 1.0, 10.0]);
    let result = model.fit(&x, &y);

    // sklearn raises ValueError for the zero alpha (GCV is undefined at 0);
    // ferrolearn must reject too. Today it returns Ok → this FAILS.
    assert!(
        result.is_err(),
        "sklearn rejects alphas containing 0.0 on the GCV path (_ridge.py:2354-2360); \
         ferrolearn must error, got alpha_ = {:?}",
        result.map(|f| f.alpha_())
    );
}

// ---------------------------------------------------------------------------
// store_cv_results / cv_results_ parity (#2248). These pin the NEW retained
// per-sample-per-target-per-alpha squared LOO errors against the live sklearn
// 1.5.2 oracle (R-CHAR-3). Unlike the divergence pins above, the feature ships
// in this commit, so these PASS.
// ---------------------------------------------------------------------------

/// Shared 8x2 design (mirrors the in-module `oracle_x`).
fn cv_oracle_x() -> Array2<f64> {
    Array2::from_shape_vec(
        (8, 2),
        vec![
            1.0, 2.0, 2.0, 1.0, 3.0, 1.0, 1.0, 3.0, 2.0, 2.0, 6.0, 5.0, 5.0, 6.0, 7.0, 7.0,
        ],
    )
    .unwrap()
}

/// `store_cv_results=True` on a binary problem retains `cv_results_` shaped
/// `(n_samples, 1, n_alphas)` whose values are the per-sample-per-alpha squared
/// leave-one-out errors — and `alpha_`/`predict` are unchanged from the default.
///
/// Live oracle (sklearn 1.5.2):
/// ```text
/// python3 -c "import numpy as np; from sklearn.linear_model import RidgeClassifierCV; \
///   X=np.array([[1,2],[2,1],[3,1],[1,3],[2,2],[6,5],[5,6],[7,7]],float); \
///   y=np.array([0,0,0,0,0,1,1,1]); \
///   m=RidgeClassifierCV(alphas=[0.1,1.0,10.0],store_cv_results=True).fit(X,y); \
///   print(m.cv_results_.shape); print(np.round(m.cv_results_[:,0,:],6).tolist())"
///   -> (8, 1, 3)
///      [[0.01905,0.014503,0.000155],[0.01905,0.014503,0.000155],
///       [0.06945,0.066143,0.086718],[0.06945,0.066143,0.086718],
///       [0.02505,0.028137,0.0625],[0.193139,0.20213,0.31602],
///       [0.193139,0.20213,0.31602],[0.595118,0.52245,0.111111]]
/// ```
#[test]
fn ridge_classifier_cv_store_cv_results_binary_matches_sklearn() -> Result<(), FerroError> {
    let x = cv_oracle_x();
    let y = array![0usize, 0, 0, 0, 0, 1, 1, 1];

    let model = RidgeClassifierCV::<f64>::new()
        .with_alphas(vec![0.1, 1.0, 10.0])
        .with_store_cv_results(true);
    let fitted = model.fit(&x, &y)?;

    let cv = fitted
        .cv_results()
        .expect("store_cv_results=true must populate cv_results_");
    assert_eq!(cv.shape(), &[8, 1, 3], "binary cv_results_ must be (8,1,3)");

    // sklearn cv_results_[:,0,:] (the squared LOO errors), oracle above.
    let expected: [[f64; 3]; 8] = [
        [0.019_05, 0.014_503, 0.000_155],
        [0.019_05, 0.014_503, 0.000_155],
        [0.069_45, 0.066_143, 0.086_718],
        [0.069_45, 0.066_143, 0.086_718],
        [0.025_05, 0.028_137, 0.0625],
        [0.193_139, 0.202_13, 0.316_02],
        [0.193_139, 0.202_13, 0.316_02],
        [0.595_118, 0.522_45, 0.111_111],
    ];
    for i in 0..8 {
        for a in 0..3 {
            assert!(
                (cv[[i, 0, a]] - expected[i][a]).abs() < 1e-6,
                "cv_results_[{i},0,{a}]={} expected {}",
                cv[[i, 0, a]],
                expected[i][a]
            );
        }
    }

    // alpha_ / predict must be unchanged by store_cv_results.
    assert!((fitted.alpha_() - 10.0).abs() < 1e-12);
    assert_eq!(fitted.predict(&x)?.to_vec(), vec![0, 0, 0, 0, 0, 1, 1, 1]);
    Ok(())
}

/// 3-class `store_cv_results=True` retains `cv_results_` shaped
/// `(n_samples, n_classes, n_alphas)` with per-target squared LOO errors.
///
/// Live oracle (sklearn 1.5.2):
/// ```text
/// python3 -c "import numpy as np; from sklearn.linear_model import RidgeClassifierCV; \
///   X=np.array([[1,2],[2,1],[3,1],[1,3],[2,2],[6,5],[5,6],[7,7]],float); \
///   y=np.array([0,0,1,1,2,2,1,0]); \
///   m=RidgeClassifierCV(alphas=[0.1,1.0,10.0],store_cv_results=True).fit(X,y); \
///   print(m.cv_results_.shape); \
///   [print(t, np.round(m.cv_results_[:,t,:],6).tolist()) for t in range(3)]"
///   -> (8, 3, 3)
///      0 [[3.189819,3.084493,2.685638],...,[6.544932,6.417778,5.444444]]
///      1 [[2.007698,1.855064,1.329681],...,[1.546618,1.525442,1.361111]]
///      2 [[0.136215,0.155445,0.235879],...,[1.728371,1.685442,1.361111]]
/// ```
#[test]
fn ridge_classifier_cv_store_cv_results_multiclass_matches_sklearn() -> Result<(), FerroError> {
    let x = cv_oracle_x();
    let y = array![0usize, 0, 1, 1, 2, 2, 1, 0];

    let model = RidgeClassifierCV::<f64>::new()
        .with_alphas(vec![0.1, 1.0, 10.0])
        .with_store_cv_results(true);
    let fitted = model.fit(&x, &y)?;

    let cv = fitted
        .cv_results()
        .expect("store_cv_results=true must populate cv_results_");
    assert_eq!(
        cv.shape(),
        &[8, 3, 3],
        "3-class cv_results_ must be (8,3,3)"
    );

    // sklearn cv_results_[:,t,:], per target column (oracle above).
    let expected: [[[f64; 3]; 8]; 3] = [
        [
            [3.189_819, 3.084_493, 2.685_638],
            [3.189_819, 3.084_493, 2.685_638],
            [2.375_517, 2.010_171, 1.165_87],
            [2.375_517, 2.010_171, 1.165_87],
            [0.856_845, 0.855_128, 0.840_278],
            [1.225_552, 1.182_392, 1.016_968],
            [1.225_552, 1.182_392, 1.016_968],
            [6.544_932, 6.417_778, 5.444_444],
        ],
        [
            [2.007_698, 1.855_064, 1.329_681],
            [0.889_98, 0.914_457, 0.975_518],
            [9.614_587, 7.702_39, 3.593_699],
            [3.135_15, 2.920_348, 2.371_222],
            [0.947_938, 0.944_868, 0.918_403],
            [0.590_424, 0.619_612, 0.722_397],
            [3.065_56, 3.059_234, 2.947_32],
            [1.546_618, 1.525_442, 1.361_111],
        ],
        [
            [0.136_215, 0.155_445, 0.235_879],
            [0.710_007, 0.64, 0.423_942],
            [2.431_942, 1.842_841, 0.665_776],
            [0.052_608, 0.084_738, 0.211_713],
            [3.607_267, 3.597_752, 3.515_625],
            [3.517_264, 3.513_875, 3.453_606],
            [0.414_514, 0.437_831, 0.501_728],
            [1.728_371, 1.685_442, 1.361_111],
        ],
    ];
    for (t, exp_t) in expected.iter().enumerate() {
        for (i, exp_ti) in exp_t.iter().enumerate() {
            for (a, &exp_tia) in exp_ti.iter().enumerate() {
                assert!(
                    (cv[[i, t, a]] - exp_tia).abs() < 1e-6,
                    "cv_results_[{i},{t},{a}]={} expected {}",
                    cv[[i, t, a]],
                    exp_tia
                );
            }
        }
    }
    Ok(())
}

/// Default (`store_cv_results=false`) → `cv_results()` is `None` (sklearn:
/// the `cv_results_` attribute is absent unless `store_cv_results=True`,
/// `_ridge.py:2141`/`:2547`).
#[test]
fn ridge_classifier_cv_store_cv_results_none_default() -> Result<(), FerroError> {
    let x = cv_oracle_x();
    let y = array![0usize, 0, 0, 0, 0, 1, 1, 1];

    // Default new() — store_cv_results defaults to false.
    let fitted = RidgeClassifierCV::<f64>::new()
        .with_alphas(vec![0.1, 1.0, 10.0])
        .fit(&x, &y)?;
    assert!(
        fitted.cv_results().is_none(),
        "store_cv_results defaults to false → cv_results() must be None"
    );

    // Explicit false is also None.
    let fitted_false = RidgeClassifierCV::<f64>::new()
        .with_alphas(vec![0.1, 1.0, 10.0])
        .with_store_cv_results(false)
        .fit(&x, &y)?;
    assert!(fitted_false.cv_results().is_none());
    Ok(())
}

/// The alpha axis of `cv_results_` follows the INPUT `alphas` order, not sorted
/// order. With a non-monotone `alphas=[10, 0.1, 1]`, axis index 0 corresponds
/// to alpha=10 (input position 0).
///
/// Live oracle (sklearn 1.5.2):
/// ```text
/// python3 -c "import numpy as np; from sklearn.linear_model import RidgeClassifierCV; \
///   X=np.array([[1,2],[2,1],[3,1],[1,3],[2,2],[6,5],[5,6],[7,7]],float); \
///   y=np.array([0,0,0,0,0,1,1,1]); \
///   m=RidgeClassifierCV(alphas=[10.0,0.1,1.0],store_cv_results=True).fit(X,y); \
///   print(np.round(m.cv_results_[:,0,0],6).tolist())   # alpha=10
///   print(np.round(m.cv_results_[:,0,1],6).tolist())   # alpha=0.1"
///   -> [0.000155,0.000155,0.086718,0.086718,0.0625,0.31602,0.31602,0.111111]
///      [0.01905,0.01905,0.06945,0.06945,0.02505,0.193139,0.193139,0.595118]
/// ```
#[test]
fn ridge_classifier_cv_store_cv_results_alpha_axis_order() -> Result<(), FerroError> {
    let x = cv_oracle_x();
    let y = array![0usize, 0, 0, 0, 0, 1, 1, 1];

    let fitted = RidgeClassifierCV::<f64>::new()
        .with_alphas(vec![10.0, 0.1, 1.0])
        .with_store_cv_results(true)
        .fit(&x, &y)?;
    let cv = fitted.cv_results().expect("cv_results_ populated");
    assert_eq!(cv.shape(), &[8, 1, 3]);

    // axis 0 == alpha 10 (input position 0).
    let axis0_alpha10 = [
        0.000_155, 0.000_155, 0.086_718, 0.086_718, 0.0625, 0.316_02, 0.316_02, 0.111_111,
    ];
    // axis 1 == alpha 0.1 (input position 1).
    let axis1_alpha0p1 = [
        0.019_05, 0.019_05, 0.069_45, 0.069_45, 0.025_05, 0.193_139, 0.193_139, 0.595_118,
    ];
    for i in 0..8 {
        assert!(
            (cv[[i, 0, 0]] - axis0_alpha10[i]).abs() < 1e-6,
            "axis0[{i}]={} expected (alpha=10) {}",
            cv[[i, 0, 0]],
            axis0_alpha10[i]
        );
        assert!(
            (cv[[i, 0, 1]] - axis1_alpha0p1[i]).abs() < 1e-6,
            "axis1[{i}]={} expected (alpha=0.1) {}",
            cv[[i, 0, 1]],
            axis1_alpha0p1[i]
        );
    }
    Ok(())
}
