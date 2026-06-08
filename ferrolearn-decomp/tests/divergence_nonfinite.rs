//! Divergence pins: ferrolearn-decomp's NaN/Inf input handling vs scikit-learn 1.5.2.
//!
//! scikit-learn's decomposition estimators have NO missing-value support: every
//! `fit`/`transform` runs `self._validate_data(...)` whose `check_array`
//! `force_all_finite=True` default raises
//! `ValueError("Input X contains NaN.")` (NaN) /
//! `ValueError("Input X contains infinity ...")` (Inf) BEFORE any decomposition
//! math (`sklearn/utils/validation.py:108` `raise ValueError("Input contains NaN")`,
//! reached from `check_array`).
//!
//! Per-estimator `_validate_data` sites (read at tag 1.5.2, commit 156ef14):
//!   - PCA            `sklearn/decomposition/_pca.py:511-518`
//!   - TruncatedSVD   `sklearn/decomposition/_truncated_svd.py:228` (fit), `:294` (transform)
//!   - NMF            `sklearn/decomposition/_nmf.py:1652-1653` (BEFORE check_non_negative `:1706`)
//!   - FastICA        `sklearn/decomposition/_fastica.py:564` (fit), `:756` (transform)
//!   - KernelPCA      `sklearn/decomposition/_kernel_pca.py:438` (fit), `:499` (transform)
//!   - FactorAnalysis `sklearn/decomposition/_factor_analysis.py:222` (fit), `:332`/`:402` (transform)
//!
//! Live sklearn 1.5.2 oracle (run from /tmp, R-CHAR-3) — for EACH of the six,
//! `fit(X_with_nan)` and `fit(X_with_inf)` raise `ValueError`; first message line
//! is exactly `Input X contains NaN.` (NaN) / `Input X contains infinity or a
//! value too large for dtype('float64').` (Inf). NMF raises the NaN finiteness
//! error EVEN when a negative value is also present (finiteness fires before the
//! non-negativity check). transform(X_nan) after a finite fit likewise raises.
//!
//! ferrolearn (probed empirically): NaN/Inf flow unchecked into the SVD/eigen/NMF
//! math. PCA & FactorAnalysis raise an INCIDENTAL `NumericalInstability` (LAPACK
//! gesdd illegal-argument / faer NoConvergence — the wrong error, R-DEV-2), NOT a
//! deliberate finiteness rejection; TruncatedSVD, NMF, FastICA, KernelPCA return
//! `Ok(_)` with a silent garbage decomposition; every probed transform returns
//! `Ok(_)` with non-finite output.
//!
//! Tracking: fit-time #2288, transform-time #2289.

use ferrolearn_core::error::FerroError;
use ferrolearn_core::traits::{Fit, Transform};
use ferrolearn_decomp::{FactorAnalysis, FastICA, KernelPCA, NMF, PCA, TruncatedSVD};
use ndarray::{Array2, array};

fn x_finite() -> Array2<f64> {
    array![
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [10.0, 11.0, 12.0],
    ]
}
fn x_nan() -> Array2<f64> {
    array![
        [1.0, 2.0, 3.0],
        [4.0, f64::NAN, 6.0],
        [7.0, 8.0, 9.0],
        [10.0, 11.0, 12.0],
    ]
}
fn x_inf() -> Array2<f64> {
    array![
        [1.0, 2.0, 3.0],
        [4.0, f64::INFINITY, 6.0],
        [7.0, 8.0, 9.0],
        [10.0, 11.0, 12.0],
    ]
}

/// A `fit` result "rejects for non-finiteness like sklearn" only if it is an
/// `Err` carrying a finiteness signal (the message names NaN / infinite /
/// non-finite). An `Ok(_)` (silent garbage) and an incidental error that does
/// NOT mention finiteness (e.g. an SVD `NumericalInstability`) both FAIL this —
/// matching the R-DEV-2 contract that the rejection is a deliberate finiteness
/// `ValueError`, not a downstream linalg accident.
fn rejects_for_nonfinite<T>(r: &Result<T, FerroError>) -> bool {
    match r {
        Ok(_) => false,
        Err(e) => {
            let m = format!("{e}").to_lowercase();
            m.contains("nan")
                || m.contains("infinit")
                || m.contains("non-finite")
                || m.contains("finite")
        }
    }
}

// ---------------------------------------------------------------------------
// fit-time NaN — sklearn ValueError "Input X contains NaN." (tracking #2288)
// ---------------------------------------------------------------------------

#[test]
fn divergence_pca_fit_nan_rejects_for_finiteness() {
    // sklearn: PCA().fit(X_nan) -> ValueError("Input X contains NaN.")
    // (`_pca.py:511`). ferrolearn: incidental NumericalInstability from gesdd,
    // not a finiteness rejection.
    assert!(rejects_for_nonfinite(
        &PCA::<f64>::new(1).fit(&x_nan(), &())
    ));
}

#[test]
fn divergence_truncated_svd_fit_nan_rejects_for_finiteness() {
    // sklearn: TruncatedSVD().fit(X_nan) -> ValueError("Input X contains NaN.")
    // (`_truncated_svd.py:228`). ferrolearn: Ok(silent garbage).
    assert!(rejects_for_nonfinite(
        &TruncatedSVD::<f64>::new(1).fit(&x_nan(), &())
    ));
}

#[test]
fn divergence_nmf_fit_nan_rejects_for_finiteness() {
    // sklearn: NMF().fit(X_nan) -> ValueError("Input X contains NaN.")
    // (`_nmf.py:1652`). ferrolearn: Ok(silent garbage). Non-negative input.
    let x = array![[1.0, 2.0], [3.0, f64::NAN], [5.0, 6.0], [7.0, 8.0]];
    assert!(rejects_for_nonfinite(&NMF::<f64>::new(1).fit(&x, &())));
}

#[test]
fn divergence_nmf_fit_nan_and_negative_finiteness_fires_first() {
    // sklearn: NMF with a NaN AND a negative entry -> ValueError("Input X
    // contains NaN.") — `_validate_data` (`_nmf.py:1652`) runs BEFORE
    // `check_non_negative` (`_nmf.py:1706`), so the finiteness error wins.
    // ferrolearn: Ok(silent garbage) — neither check exists.
    let x = array![[1.0, 2.0], [3.0, f64::NAN], [-5.0, 6.0], [7.0, 8.0]];
    let r = NMF::<f64>::new(1).fit(&x, &());
    // Must reject, and the reason must be the NaN (not the negative value).
    assert!(rejects_for_nonfinite(&r));
}

#[test]
fn divergence_fast_ica_fit_nan_rejects_for_finiteness() {
    // sklearn: FastICA().fit(X_nan) -> ValueError("Input X contains NaN.")
    // (`_fastica.py:564`). ferrolearn: Ok(silent garbage).
    assert!(rejects_for_nonfinite(
        &FastICA::<f64>::new(1).fit(&x_nan(), &())
    ));
}

#[test]
fn divergence_kernel_pca_fit_nan_rejects_for_finiteness() {
    // sklearn: KernelPCA().fit(X_nan) -> ValueError("Input X contains NaN.")
    // (`_kernel_pca.py:438`). ferrolearn: Ok(silent garbage).
    assert!(rejects_for_nonfinite(
        &KernelPCA::<f64>::new(1).fit(&x_nan(), &())
    ));
}

#[test]
fn divergence_factor_analysis_fit_nan_rejects_for_finiteness() {
    // sklearn: FactorAnalysis().fit(X_nan) -> ValueError("Input X contains NaN.")
    // (`_factor_analysis.py:222`). ferrolearn: incidental NumericalInstability
    // (faer NoConvergence), not a finiteness rejection.
    assert!(rejects_for_nonfinite(
        &FactorAnalysis::<f64>::new(1).fit(&x_nan(), &())
    ));
}

// ---------------------------------------------------------------------------
// fit-time +Inf — sklearn ValueError "Input X contains infinity ..." (#2288)
// ---------------------------------------------------------------------------

#[test]
fn divergence_truncated_svd_fit_inf_rejects_for_finiteness() {
    assert!(rejects_for_nonfinite(
        &TruncatedSVD::<f64>::new(1).fit(&x_inf(), &())
    ));
}

#[test]
fn divergence_fast_ica_fit_inf_rejects_for_finiteness() {
    assert!(rejects_for_nonfinite(
        &FastICA::<f64>::new(1).fit(&x_inf(), &())
    ));
}

#[test]
fn divergence_kernel_pca_fit_inf_rejects_for_finiteness() {
    assert!(rejects_for_nonfinite(
        &KernelPCA::<f64>::new(1).fit(&x_inf(), &())
    ));
}

#[test]
fn divergence_pca_fit_inf_rejects_for_finiteness() {
    // sklearn: ValueError("Input X contains infinity ..."). ferrolearn: incidental
    // gesdd NumericalInstability — wrong error type (R-DEV-2).
    assert!(rejects_for_nonfinite(
        &PCA::<f64>::new(1).fit(&x_inf(), &())
    ));
}

// ---------------------------------------------------------------------------
// transform-time NaN after a finite fit — sklearn ValueError (tracking #2289)
// ---------------------------------------------------------------------------

#[test]
fn divergence_pca_transform_nan_rejects_for_finiteness() {
    // sklearn: PCA().fit(Xf).transform(X_nan) -> ValueError (`_pca.py:824`,
    // _validate_data reset=False). ferrolearn: Ok(non-finite garbage).
    let f = PCA::<f64>::new(1).fit(&x_finite(), &()).unwrap();
    let xq = array![[1.0, f64::NAN, 3.0]];
    assert!(rejects_for_nonfinite(&f.transform(&xq)));
}

#[test]
fn divergence_truncated_svd_transform_nan_rejects_for_finiteness() {
    // sklearn: `_truncated_svd.py:294`. ferrolearn: Ok(non-finite garbage).
    let f = TruncatedSVD::<f64>::new(1).fit(&x_finite(), &()).unwrap();
    let xq = array![[1.0, f64::NAN, 3.0]];
    assert!(rejects_for_nonfinite(&f.transform(&xq)));
}

#[test]
fn divergence_factor_analysis_transform_nan_rejects_for_finiteness() {
    // sklearn: `_factor_analysis.py:332`. ferrolearn: Ok(non-finite garbage).
    let f = FactorAnalysis::<f64>::new(1).fit(&x_finite(), &()).unwrap();
    let xq = array![[1.0, f64::NAN, 3.0]];
    assert!(rejects_for_nonfinite(&f.transform(&xq)));
}

#[test]
fn divergence_kernel_pca_transform_nan_rejects_for_finiteness() {
    // sklearn: `_kernel_pca.py:499`. ferrolearn: Ok(non-finite garbage).
    let f = KernelPCA::<f64>::new(1).fit(&x_finite(), &()).unwrap();
    let xq = array![[1.0, f64::NAN, 3.0]];
    assert!(rejects_for_nonfinite(&f.transform(&xq)));
}
