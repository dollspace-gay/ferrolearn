//! Divergence + green-guard suite for `SelectFpr` / `SelectFdr` / `SelectFwe`
//! against scikit-learn 1.5.2 `_univariate_selection.py`.
//!
//! Each selector consumes a STATIC per-feature p-value vector
//! (`Fit<Array1<F>, ()>`); the sklearn selectors wrap a `score_func` and
//! compute `pvalues_` internally — that wrapping is the documented HONEST gap
//! (REQ-5, NOT-STARTED) and is NOT pinned here. These tests pin the selection
//! MASKS + the alpha parameter contract on static p-value vectors.
//!
//! Expected values are the sklearn `_get_support_mask` FORMULA replicated in
//! numpy over each chosen p-value vector (live sklearn 1.5.2 oracle); they are
//! NOT copied from the ferrolearn side (R-CHAR-3).
//!
//! Tracking: #1396.

use ferrolearn_core::traits::{Fit, Transform};
use ferrolearn_preprocess::stat_selectors::{SelectFdr, SelectFpr, SelectFwe};
use ndarray::{Array1, array};

// ===========================================================================
// SelectFpr — mask `pvalues_ < alpha` (_univariate_selection.py:878)
// ===========================================================================

/// Green-guard: ferrolearn's `SelectFpr` mirrors
/// `sklearn/feature_selection/_univariate_selection.py:878`
/// `return self.pvalues_ < self.alpha` for `[0.01,0.5,0.03,0.9]`, alpha=0.05.
///
/// Oracle (live sklearn 1.5.2): `[j for j,p in enumerate(pv) if p<a]` == [0, 2].
/// Tracking: #1396
#[test]
fn green_fpr_basic() {
    let sel = SelectFpr::<f64>::new(0.05);
    let p = array![0.01, 0.5, 0.03, 0.9];
    match sel.fit(&p, &()) {
        Ok(fitted) => assert_eq!(fitted.selected_indices(), &[0, 2]),
        Err(e) => panic!("fit should succeed, got {e:?}"),
    }
}

/// Green-guard: STRICT `<` at `_univariate_selection.py:878` — a p-value
/// EXACTLY equal to alpha is NOT selected.
///
/// Oracle: `fpr([0.05, 0.04], 0.05)` == [1] (0.05 == alpha excluded).
/// Tracking: #1396
#[test]
fn green_fpr_strict_boundary() {
    let sel = SelectFpr::<f64>::new(0.05);
    let p = array![0.05, 0.04];
    match sel.fit(&p, &()) {
        Ok(fitted) => assert_eq!(fitted.selected_indices(), &[1]),
        Err(e) => panic!("fit should succeed, got {e:?}"),
    }
}

// ===========================================================================
// SelectFwe — mask `pvalues_ < alpha / n` (_univariate_selection.py:1044)
// ===========================================================================

/// Green-guard: ferrolearn's `SelectFwe` mirrors
/// `sklearn/feature_selection/_univariate_selection.py:1044`
/// `return self.pvalues_ < self.alpha / len(self.pvalues_)` for
/// `[0.001,0.5,0.03,0.9]`, alpha=0.05 (threshold 0.05/4 = 0.0125).
///
/// Oracle: `fwe([0.001,0.5,0.03,0.9], 0.05)` == [0].
/// Tracking: #1396
#[test]
fn green_fwe_basic() {
    let sel = SelectFwe::<f64>::new(0.05);
    let p = array![0.001, 0.5, 0.03, 0.9];
    match sel.fit(&p, &()) {
        Ok(fitted) => assert_eq!(fitted.selected_indices(), &[0]),
        Err(e) => panic!("fit should succeed, got {e:?}"),
    }
}

/// Green-guard: STRICT `<` at `_univariate_selection.py:1044` — a p-value
/// EXACTLY equal to alpha/n is NOT selected. n=2, alpha=0.05 → thr=0.025.
///
/// Oracle: `fwe([0.025, 0.01], 0.05)` == [1] (0.025 == alpha/n excluded).
/// Tracking: #1396
#[test]
fn green_fwe_strict_boundary() {
    let sel = SelectFwe::<f64>::new(0.05);
    let p = array![0.025, 0.01];
    match sel.fit(&p, &()) {
        Ok(fitted) => assert_eq!(fitted.selected_indices(), &[1]),
        Err(e) => panic!("fit should succeed, got {e:?}"),
    }
}

// ===========================================================================
// SelectFdr — Benjamini-Hochberg (_univariate_selection.py:959-969)
//   n = len(pvalues_); sv = np.sort(pvalues_)
//   selected = sv[sv <= float(alpha)/n * np.arange(1, n+1)]
//   if selected.size == 0: all-False  else: pvalues_ <= selected.max()
// ===========================================================================

/// Green-guard: SelectFdr basic. Oracle:
/// `fdr([0.01,0.5,0.03,0.9], 0.05)` == [0].
/// Tracking: #1396
#[test]
fn green_fdr_basic() {
    let sel = SelectFdr::<f64>::new(0.05);
    let p = array![0.01, 0.5, 0.03, 0.9];
    match sel.fit(&p, &()) {
        Ok(fitted) => assert_eq!(fitted.selected_indices(), &[0]),
        Err(e) => panic!("fit should succeed, got {e:?}"),
    }
}

/// Green-guard: SelectFdr TIE at the BH boundary — both 0.025 values kept
/// because the mask is `pvalues_ <= selected.max()` (:969). Oracle:
/// `fdr([0.01,0.025,0.025,0.9], 0.05)` == [0, 1, 2].
/// Tracking: #1396
#[test]
fn green_fdr_tie_boundary() {
    let sel = SelectFdr::<f64>::new(0.05);
    let p = array![0.01, 0.025, 0.025, 0.9];
    match sel.fit(&p, &()) {
        Ok(fitted) => assert_eq!(fitted.selected_indices(), &[0, 1, 2]),
        Err(e) => panic!("fit should succeed, got {e:?}"),
    }
}

/// Green-guard: SelectFdr NON-MONOTONE GAP — a feature failing its OWN rank
/// threshold is still selected because a higher rank qualifies and the mask is
/// `pvalues_ <= selected.max()` (:969). Oracle:
/// `fdr([0.001,0.04,0.045,0.011], 0.05)` == [0, 1, 2, 3].
/// Tracking: #1396
#[test]
fn green_fdr_nonmonotone_gap() {
    let sel = SelectFdr::<f64>::new(0.05);
    let p = array![0.001, 0.04, 0.045, 0.011];
    match sel.fit(&p, &()) {
        Ok(fitted) => assert_eq!(fitted.selected_indices(), &[0, 1, 2, 3]),
        Err(e) => panic!("fit should succeed, got {e:?}"),
    }
}

/// Green-guard: SelectFdr NONE qualify → empty (`selected.size == 0` branch,
/// :967-968). Oracle: `fdr([0.9,0.8,0.95], 0.05)` == [].
/// Tracking: #1396
#[test]
fn green_fdr_none_qualify() {
    let sel = SelectFdr::<f64>::new(0.05);
    let p = array![0.9, 0.8, 0.95];
    match sel.fit(&p, &()) {
        Ok(fitted) => {
            let empty: &[usize] = &[];
            assert_eq!(fitted.selected_indices(), empty);
        }
        Err(e) => panic!("fit should succeed, got {e:?}"),
    }
}

/// Green-guard: SelectFdr ALL qualify. Oracle:
/// `fdr([0.001,0.002,0.003], 0.05)` == [0, 1, 2].
/// Tracking: #1396
#[test]
fn green_fdr_all_qualify() {
    let sel = SelectFdr::<f64>::new(0.05);
    let p = array![0.001, 0.002, 0.003];
    match sel.fit(&p, &()) {
        Ok(fitted) => assert_eq!(fitted.selected_indices(), &[0, 1, 2]),
        Err(e) => panic!("fit should succeed, got {e:?}"),
    }
}

// ===========================================================================
// Parameter contract — alpha boundary
// sklearn `_parameter_constraints` alpha: Interval(Real, 0, 1, closed="both")
//   (:866-869, :950-953, :1032-1035) → alpha=0 AND alpha=1 are BOTH VALID.
// ferrolearn `validate_inputs` rejects `alpha <= 0.0 || alpha > 1.0`
//   (stat_selectors.rs:49) → alpha=0 ERRORS.
// ===========================================================================

/// DIVERGENCE: ferrolearn's `SelectFpr::new(0.0).fit(..)` errors, but sklearn's
/// `_parameter_constraints` allows `alpha == 0`
/// (`sklearn/feature_selection/_univariate_selection.py:868`
/// `Interval(Real, 0, 1, closed="both")`); with alpha=0 the FPR mask
/// `pvalues_ < 0` selects nothing for any positive p-value.
///
/// ferrolearn `stat_selectors.rs:49`: `if alpha <= 0.0 || alpha > 1.0 { Err }`.
///
/// Input: `[0.01,0.5,0.03]`, alpha=0.0.
/// sklearn: accepts, selects [] (oracle `fpr([0.01,0.5,0.03], 0.0)` == []).
/// ferrolearn: returns `Err(InvalidParameter)`.
/// Tracking: #1396
#[test]
fn divergence_fpr_alpha_zero_accepted() {
    let sel = SelectFpr::<f64>::new(0.0);
    let p = array![0.01, 0.5, 0.03];
    match sel.fit(&p, &()) {
        Ok(fitted) => {
            let empty: &[usize] = &[];
            assert_eq!(fitted.selected_indices(), empty);
        }
        Err(e) => panic!(
            "sklearn accepts alpha=0 (Interval closed=both, :868) and selects \
             nothing; ferrolearn rejected it with {e:?}"
        ),
    }
}

/// DIVERGENCE: ferrolearn's `SelectFwe::new(0.0).fit(..)` errors, but sklearn
/// allows `alpha == 0` (`:1034`). Oracle `fwe([0.01,0.5,0.03], 0.0)` == [].
/// Tracking: #1396
#[test]
fn divergence_fwe_alpha_zero_accepted() {
    let sel = SelectFwe::<f64>::new(0.0);
    let p = array![0.01, 0.5, 0.03];
    match sel.fit(&p, &()) {
        Ok(fitted) => {
            let empty: &[usize] = &[];
            assert_eq!(fitted.selected_indices(), empty);
        }
        Err(e) => panic!(
            "sklearn accepts alpha=0 (Interval closed=both, :1034) and selects \
             nothing; ferrolearn rejected it with {e:?}"
        ),
    }
}

/// DIVERGENCE: ferrolearn's `SelectFdr::new(0.0).fit(..)` errors, but sklearn
/// allows `alpha == 0` (`:952`). With alpha=0 and no zero-valued p-values, the
/// BH threshold is all-zero so nothing qualifies. Oracle
/// `fdr([0.01,0.5,0.03], 0.0)` == [].
/// Tracking: #1396
#[test]
fn divergence_fdr_alpha_zero_accepted() {
    let sel = SelectFdr::<f64>::new(0.0);
    let p = array![0.01, 0.5, 0.03];
    match sel.fit(&p, &()) {
        Ok(fitted) => {
            let empty: &[usize] = &[];
            assert_eq!(fitted.selected_indices(), empty);
        }
        Err(e) => panic!(
            "sklearn accepts alpha=0 (Interval closed=both, :952) and selects \
             nothing; ferrolearn rejected it with {e:?}"
        ),
    }
}

/// Green-guard: alpha == 1 is the UPPER endpoint of sklearn's
/// `Interval(Real, 0, 1, closed="both")` (:868) and ferrolearn's `(0, 1]` both
/// ACCEPT it. Oracle `fpr([0.5,1.0,0.99], 1.0)` == [0, 2] (p < 1.0 strict).
/// Tracking: #1396
#[test]
fn green_fpr_alpha_one_accepted() {
    let sel = SelectFpr::<f64>::new(1.0);
    let p = array![0.5, 1.0, 0.99];
    match sel.fit(&p, &()) {
        Ok(fitted) => assert_eq!(fitted.selected_indices(), &[0, 2]),
        Err(e) => panic!("alpha=1 must be accepted (both intervals), got {e:?}"),
    }
}

// ===========================================================================
// transform — column projection + shape contract
// ===========================================================================

/// Green-guard: transform projects exactly the selected columns in order.
/// FPR on `[0.01,0.5,0.03]`, alpha=0.05 selects [0, 2]; transform of a
/// 2x3 matrix yields columns 0 and 2.
/// Tracking: #1396
#[test]
fn green_fpr_transform_projection() {
    let sel = SelectFpr::<f64>::new(0.05);
    let p = array![0.01, 0.5, 0.03];
    let fitted = match sel.fit(&p, &()) {
        Ok(f) => f,
        Err(e) => panic!("fit failed: {e:?}"),
    };
    let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    match fitted.transform(&x) {
        Ok(out) => {
            assert_eq!(out.shape(), &[2, 2]);
            assert_eq!(out, array![[1.0, 3.0], [4.0, 6.0]]);
        }
        Err(e) => panic!("transform failed: {e:?}"),
    }
}

/// Green-guard: transform with wrong column count → `ShapeMismatch`.
/// Tracking: #1396
#[test]
fn green_fpr_transform_ncols_mismatch() {
    let sel = SelectFpr::<f64>::new(0.05);
    let p = array![0.01, 0.5];
    let fitted = match sel.fit(&p, &()) {
        Ok(f) => f,
        Err(e) => panic!("fit failed: {e:?}"),
    };
    let x_bad = array![[1.0, 2.0, 3.0]];
    assert!(
        fitted.transform(&x_bad).is_err(),
        "transform must reject a column-count mismatch"
    );
}

// ===========================================================================
// f32 path
// ===========================================================================

/// Green-guard: f32 SelectFdr matches the same oracle as f64.
/// Oracle `fdr([0.001,0.002,0.003], 0.05)` == [0, 1, 2].
/// Tracking: #1396
#[test]
fn green_fdr_f32_path() {
    let sel = SelectFdr::<f32>::new(0.05);
    let p: Array1<f32> = array![0.001f32, 0.002, 0.003];
    match sel.fit(&p, &()) {
        Ok(fitted) => assert_eq!(fitted.selected_indices(), &[0, 1, 2]),
        Err(e) => panic!("fit should succeed, got {e:?}"),
    }
}
