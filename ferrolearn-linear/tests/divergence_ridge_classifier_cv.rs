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
use ferrolearn_core::traits::Fit;
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
