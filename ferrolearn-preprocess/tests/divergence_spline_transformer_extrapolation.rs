//! ACToR critic: oracle-grounded audit of `ferrolearn-preprocess`'s
//! `SplineTransformer` DEFAULT out-of-range behavior + NaN handling against
//! scikit-learn 1.5.2 `sklearn/preprocessing/_polynomial.py`
//! `class SplineTransformer` (`:580`).
//!
//! Companion to `divergence_spline_transformer.rs` (which re-audits the
//! in-base-interval Uniform basis VALUE parity, REQ-2). This file pins the
//! DEFAULT-API divergences that the existing critic file did not cover:
//!
//!   * DIV-A (REQ-3, tracking #2326): ferrolearn has NO `extrapolation` param,
//!     so it always evaluates Cox-de Boor over the EXTENDED knot vector for
//!     out-of-base-interval `x`. sklearn's DEFAULT is `extrapolation="constant"`
//!     (`__init__`, `_polynomial.py:721`), which for `x < xmin` / `x > xmax`
//!     CLAMPS the first/last `degree` basis columns to the boundary spline
//!     values and zeroes the rest (`_polynomial.py:1059-1087`). Because the
//!     defaults differ, `fit_transform` on data containing out-of-range query
//!     values produces DIFFERENT numbers (not just a missing knob): ferrolearn
//!     returns garbage extrapolated (often negative) values where sklearn
//!     returns the held boundary basis.
//!
//!   * DIV-B (REQ-6 / validation, tracking #2327): sklearn's `fit` calls
//!     `_validate_data(..., ensure_2d=True)` which rejects non-finite input —
//!     `ValueError("Input X contains NaN. ...")` (`_polynomial.py:833-839`).
//!     ferrolearn fits NaN silently and emits NaN basis values.
//!
//! All expected VALUES come from the LIVE sklearn 1.5.2 oracle (run from /tmp,
//! R-CHAR-3) — NEVER literal-copied from the ferrolearn side.
//!
//! Oracle reproduction (sklearn 1.5.2):
//! ```text
//! cd /tmp && python3 -c "import numpy as np; \
//!   from sklearn.preprocessing import SplineTransformer; \
//!   X=np.array([[0.],[0.3],[0.6],[1.0]]); \
//!   st=SplineTransformer(n_knots=4, degree=3).fit(X); \
//!   print(np.round(st.transform(np.array([[-0.5],[1.5]])),9).tolist())"
//! # -> [[0.166666667, 0.666666667, 0.166666667, 0.0, 0.0, 0.0],
//! #     [0.0, 0.0, 0.0, 0.166666667, 0.666666667, 0.166666667]]
//! ```

use ferrolearn_core::traits::{Fit, Transform};
use ferrolearn_preprocess::spline_transformer::{KnotStrategy, SplineTransformer};
use ndarray::{Array2, array};

/// Helper: fit on `x_fit`, transform `x_query`, assert each cell matches the
/// sklearn oracle matrix within `tol`. Error arms fail loudly (no panic!/unwrap
/// in the assertion path — anti-pattern gate).
#[allow(
    clippy::assertions_on_constants,
    reason = "error arm fails loudly without panic!/unwrap (anti-pattern gate)"
)]
fn assert_query_matches_oracle(
    x_fit: &Array2<f64>,
    x_query: &Array2<f64>,
    n_knots: usize,
    degree: usize,
    oracle: &[&[f64]],
    tol: f64,
    label: &str,
) {
    let st = SplineTransformer::<f64>::new(n_knots, degree, KnotStrategy::Uniform);
    let fitted = match st.fit(x_fit, &()) {
        Ok(f) => f,
        Err(e) => {
            assert!(false, "[{label}] fit failed: {e}");
            return;
        }
    };
    let out = match fitted.transform(x_query) {
        Ok(o) => o,
        Err(e) => {
            assert!(false, "[{label}] transform failed: {e}");
            return;
        }
    };
    assert_eq!(out.nrows(), oracle.len(), "[{label}] row count");
    assert_eq!(out.ncols(), oracle[0].len(), "[{label}] col count");
    for (i, orow) in oracle.iter().enumerate() {
        for (j, &b) in orow.iter().enumerate() {
            let a = out[[i, j]];
            assert!(
                (a - b).abs() < tol,
                "[{label}] row {i} col {j}: ferrolearn {a} vs sklearn {b} (diff {})",
                (a - b).abs()
            );
        }
    }
}

/// DIV-A (REQ-3, HEADLINE): DEFAULT `extrapolation="constant"` — out-of-range
/// query for degree=3, n_knots=4.
///
/// sklearn `_polynomial.py:721` default `extrapolation="constant"`; the
/// constant clamp lives at `_polynomial.py:1059-1087`:
///   `mask = X[:, i] < xmin; XBS[mask, ...:degree] = f_min[:degree]`
///   `mask = X[:, i] > xmax; XBS[mask, -degree:] = f_max[-degree:]`
///
/// Live sklearn 1.5.2 ORACLE (run from /tmp; X fit = [[0],[0.3],[0.6],[1.0]],
/// query = [[-0.5],[1.5]]):
///   [[1/6, 2/3, 1/6, 0, 0, 0],   # x=-0.5 < xmin: held to f_min boundary basis
///    [0,   0,   0,   1/6, 2/3, 1/6]]  # x=1.5 > xmax: held to f_max boundary basis
///
/// ferrolearn (no extrapolation param) instead evaluates Cox-de Boor over the
/// EXTENDED knot vector: for x=-0.5 it returns ~[0.479, 0.0208, 0, 0, 0, 0],
/// and for x=1.5 it returns ~[..., -0.5625, 2.229, -3.271, 2.604] (negative,
/// extrapolated). Drastic VALUE divergence in the DEFAULT API.
///
/// Tracking: #2326.
#[test]
fn divergence_extrapolation_constant_default_degree3() {
    let x_fit = array![[0.0], [0.3], [0.6], [1.0]];
    let x_query = array![[-0.5], [1.5]];
    let frac = 1.0 / 6.0;
    let two3 = 2.0 / 3.0;
    #[rustfmt::skip]
    let oracle: [[f64; 6]; 2] = [
        [frac, two3, frac, 0.0,  0.0,  0.0],
        [0.0,  0.0,  0.0,  frac, two3, frac],
    ];
    let rows: Vec<&[f64]> = oracle.iter().map(|r| r.as_slice()).collect();
    assert_query_matches_oracle(&x_fit, &x_query, 4, 3, &rows, 1e-6, "extrap_constant_d3");
}

/// DIV-A (REQ-3): DEFAULT `extrapolation="constant"`, degree=2, n_knots=3.
///
/// Live sklearn 1.5.2 ORACLE (X fit = [[0],[0.5],[1.0]], query = [[-1.0],[2.0]]):
///   [[0.5, 0.5, 0, 0],   # x=-1.0 < xmin -> held boundary basis
///    [0,   0,   0.5, 0.5]] # x=2.0 > xmax -> held boundary basis
///
/// ferrolearn returns [[0,0,0,0],[2.0,-5.5,4.5,...]] (extended-knot Cox-de Boor,
/// including a negative extrapolated value). Tracking: #2326.
#[test]
fn divergence_extrapolation_constant_default_degree2() {
    let x_fit = array![[0.0], [0.5], [1.0]];
    let x_query = array![[-1.0], [2.0]];
    #[rustfmt::skip]
    let oracle: [[f64; 4]; 2] = [
        [0.5, 0.5, 0.0, 0.0],
        [0.0, 0.0, 0.5, 0.5],
    ];
    let rows: Vec<&[f64]> = oracle.iter().map(|r| r.as_slice()).collect();
    assert_query_matches_oracle(&x_fit, &x_query, 3, 2, &rows, 1e-6, "extrap_constant_d2");
}

/// DIV-A (REQ-3): DEFAULT `extrapolation="constant"`, degree=1, n_knots=3.
///
/// Live sklearn 1.5.2 ORACLE (X fit = [[0],[0.5],[1.0]], query = [[-1.0],[2.0]]):
///   [[1.0, 0.0, 0.0],   # x=-1.0 < xmin -> first basis held at 1.0
///    [0.0, 0.0, 1.0]]   # x=2.0  > xmax -> last basis held at 1.0
///
/// ferrolearn returns [[0,0,0],[-2.0,3.0,...]] (linear extrapolation over the
/// extended knots, including a negative value). Tracking: #2326.
#[test]
fn divergence_extrapolation_constant_default_degree1() {
    let x_fit = array![[0.0], [0.5], [1.0]];
    let x_query = array![[-1.0], [2.0]];
    #[rustfmt::skip]
    let oracle: [[f64; 3]; 2] = [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ];
    let rows: Vec<&[f64]> = oracle.iter().map(|r| r.as_slice()).collect();
    assert_query_matches_oracle(&x_fit, &x_query, 3, 1, &rows, 1e-6, "extrap_constant_d1");
}

/// DIV-B (validation, tracking #2327): NaN rejection.
///
/// sklearn `fit` calls `_validate_data(...)` (`_polynomial.py:833-839`) which
/// raises `ValueError("Input X contains NaN. ...")`. Live oracle:
/// ```text
/// SplineTransformer(n_knots=3, degree=2).fit(np.array([[0.],[np.nan],[1.0]]))
/// # -> ValueError: Input X contains NaN.
/// ```
/// ferrolearn fits NaN silently (returns Ok) and emits NaN basis values on
/// transform. This test asserts the sklearn contract: fit MUST error on NaN.
#[test]
fn divergence_nan_input_must_error() {
    let st = SplineTransformer::<f64>::new(3, 2, KnotStrategy::Uniform);
    let x = array![[0.0], [f64::NAN], [1.0]];
    let res = st.fit(&x, &());
    assert!(
        res.is_err(),
        "sklearn raises ValueError(Input X contains NaN) at _polynomial.py:833-839; \
         ferrolearn must reject NaN input on fit, got Ok"
    );
}
