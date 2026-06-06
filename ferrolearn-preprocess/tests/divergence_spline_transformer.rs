//! ACToR critic: oracle-grounded audit of `ferrolearn-preprocess`'s
//! `SplineTransformer` / `FittedSplineTransformer` against scikit-learn 1.5.2
//! `sklearn/preprocessing/_polynomial.py` `class SplineTransformer` (`:580`).
//!
//! All expected VALUES are derived from the LIVE sklearn 1.5.2 oracle, run from
//! `/tmp` (R-CHAR-3): NEVER literal-copied from the ferrolearn side.
//!
//! HEADLINE DIVERGENCE — DIV-1 (REQ-2), tracking #1331, blocker #1332:
//! a prior fixer rewrote the knot construction from CLAMPED (boundary repeated
//! `degree+1` times — the `np.tile` construction sklearn EXPLICITLY REJECTS in
//! the commented-out block at `_polynomial.py:898-906`) to sklearn's EXTENDED
//! edge-spacing knot vector (`_polynomial.py:908-923`:
//!   `dist_min = base_knots[1] - base_knots[0]`,
//!   `dist_max = base_knots[-1] - base_knots[-2]`,
//!   `knots = np.r_[np.linspace(base[0]-degree*dist_min, base[0]-dist_min, degree),
//!                  base_knots,
//!                  np.linspace(base[-1]+dist_max, base[-1]+degree*dist_max, degree)]`)
//! and re-keyed the right-endpoint handling in `fn bspline_basis`
//! (`ferrolearn-preprocess/src/spline_transformer.rs:155`) to the base-interval
//! right endpoint `knots[n_basis]` (closed on the right, matching scipy
//! `BSpline.design_matrix`).
//!
//! This file RE-AUDITS that fix across MANY Uniform-knot configs (degree
//! 1/2/3, multi-feature, both base endpoints). The probes are PASSING
//! green-guard VALUE tests (live-oracle matrices) confirming REQ-2 parity.
//!
//! Oracle commands (sklearn 1.5.2 == `python3 -c "import sklearn; sklearn.__version__"`):
//! ```text
//! cd /tmp && python3 -c "import numpy as np; from sklearn.preprocessing import SplineTransformer; \
//!   X=np.array([[0.],[0.3],[0.6],[1.0]]); st=SplineTransformer(n_knots=4, degree=3).fit(X); \
//!   print(np.round(st.transform(X),9).tolist())"
//! ```

use ferrolearn_core::error::FerroError;
use ferrolearn_core::traits::{Fit, Transform};
use ferrolearn_preprocess::spline_transformer::{KnotStrategy, SplineTransformer};
use ndarray::{Array2, array};

/// Helper: fit + transform a Uniform `SplineTransformer` and assert the full
/// output matrix matches a hard-coded sklearn 1.5.2 oracle matrix elementwise.
/// No bare `.unwrap()`/`.expect()`/`panic!` in caller — errors surface as
/// assertion failures with diagnostic context.
#[allow(
    clippy::assertions_on_constants,
    reason = "error arm fails loudly without panic!/unwrap (anti-pattern gate)"
)]
fn assert_matches_oracle(
    x: &Array2<f64>,
    n_knots: usize,
    degree: usize,
    oracle: &[&[f64]],
    tol: f64,
    label: &str,
) {
    let st = SplineTransformer::<f64>::new(n_knots, degree, KnotStrategy::Uniform);
    let fitted = match st.fit(x, &()) {
        Ok(f) => f,
        Err(e) => {
            assert!(false, "[{label}] fit failed: {e}");
            return;
        }
    };
    let out = match fitted.transform(x) {
        Ok(o) => o,
        Err(e) => {
            assert!(false, "[{label}] transform failed: {e}");
            return;
        }
    };
    assert_eq!(out.nrows(), oracle.len(), "[{label}] row count");
    assert_eq!(
        out.ncols(),
        oracle[0].len(),
        "[{label}] col count (n_knots+degree-1 per feature)"
    );
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

/// DIV-1 (REQ-2, HEADLINE): after the fix, ferrolearn's EXTENDED edge-spacing
/// knots + Cox-de Boor basis must MATCH sklearn's `_polynomial.py:908-923` +
/// scipy `BSpline` design matrix (`:925-940`).
///
/// Input: X = [[0.],[0.3],[0.6],[1.0]], n_knots=4, degree=3, Uniform.
/// sklearn (live 1.5.2 oracle) returns the full 4x6 matrix below. This was the
/// SINGLE pinned fixture the fix verified; it remains green.
///
/// Tracking: #1331, blocker #1332.
#[test]
fn divergence_spline_basis_value_asymmetric_fixture() {
    // sklearn 1.5.2 LIVE ORACLE (R-CHAR-3 — hard-coded from /tmp run, NOT ferrolearn):
    //   SplineTransformer(n_knots=4, degree=3).fit(X).transform(X), np.round(...,9)
    #[rustfmt::skip]
    let sklearn: [[f64; 6]; 4] = [
        [0.166666667, 0.666666667, 0.166666667, 0.0,         0.0,         0.0],
        [0.000166667, 0.221166667, 0.657166667, 0.1215,      0.0,         0.0],
        [0.0,         0.001333333, 0.282666667, 0.630666667, 0.085333333, 0.0],
        [0.0,         0.0,         0.0,         0.166666667, 0.666666667, 0.166666667],
    ];

    let x = array![[0.0], [0.3], [0.6], [1.0]];
    let rows: Vec<&[f64]> = sklearn.iter().map(|r| r.as_slice()).collect();
    assert_matches_oracle(&x, 4, 3, &rows, 1e-6, "asymmetric_fixture");
}

// ---------------------------------------------------------------------------
// RE-AUDIT PROBES (REQ-2): Uniform-knot VALUE parity across more configs.
// All matrices are LIVE sklearn 1.5.2 oracle output (rounded to 9 dp), run
// from /tmp. These are PASSING green-guard tests confirming the fix is faithful.
// ---------------------------------------------------------------------------

/// (a) degree=1 (linear), n_knots=4. degree==1 exercises the `linspace num==1`
/// special case in the EXTENDED knot construction (`spline_transformer.rs:321`,
/// numpy `linspace(a,b,num=1) -> [a]`, sklearn `_polynomial.py:911-916`).
/// Oracle: `SplineTransformer(n_knots=4, degree=1).fit_transform(X)`.
#[test]
fn probe_a_degree1_linear_nknots4() {
    let x = array![[0.0], [0.2], [0.5], [0.9], [1.0]];
    #[rustfmt::skip]
    let oracle: [[f64; 4]; 5] = [
        [1.0, 0.0, 0.0, 0.0],
        [0.4, 0.6, 0.0, 0.0],
        [0.0, 0.5, 0.5, 0.0],
        [0.0, 0.0, 0.3, 0.7],
        [0.0, 0.0, 0.0, 1.0],
    ];
    let rows: Vec<&[f64]> = oracle.iter().map(|r| r.as_slice()).collect();
    assert_matches_oracle(&x, 4, 1, &rows, 1e-6, "a_degree1");
}

/// (b) degree=2, n_knots=3 (the doctest-shaped quadratic basis).
/// Oracle: `SplineTransformer(n_knots=3, degree=2).fit_transform(X)`.
#[test]
fn probe_b_degree2_nknots3() {
    let x = array![[0.0], [0.25], [0.5], [0.75], [1.0]];
    #[rustfmt::skip]
    let oracle: [[f64; 4]; 5] = [
        [0.5,   0.5,   0.0,   0.0],
        [0.125, 0.75,  0.125, 0.0],
        [0.0,   0.5,   0.5,   0.0],
        [0.0,   0.125, 0.75,  0.125],
        [0.0,   0.0,   0.5,   0.5],
    ];
    let rows: Vec<&[f64]> = oracle.iter().map(|r| r.as_slice()).collect();
    assert_matches_oracle(&x, 3, 2, &rows, 1e-6, "b_degree2");
}

/// (c) degree=3, n_knots=6, denser interior points (off-knot interior values).
/// Oracle: `SplineTransformer(n_knots=6, degree=3).fit_transform(X)`.
#[test]
fn probe_c_degree3_nknots6_dense() {
    let x = array![
        [0.0],
        [0.1],
        [0.2],
        [0.35],
        [0.5],
        [0.65],
        [0.8],
        [0.95],
        [1.0]
    ];
    #[rustfmt::skip]
    let oracle: [[f64; 8]; 9] = [
        [0.166666667, 0.666666667, 0.166666667, 0.0,         0.0,         0.0,         0.0,         0.0],
        [0.020833333, 0.479166667, 0.479166667, 0.020833333, 0.0,         0.0,         0.0,         0.0],
        [0.0,         0.166666667, 0.666666667, 0.166666667, 0.0,         0.0,         0.0,         0.0],
        [0.0,         0.002604167, 0.315104167, 0.611979167, 0.0703125,   0.0,         0.0,         0.0],
        [0.0,         0.0,         0.020833333, 0.479166667, 0.479166667, 0.020833333, 0.0,         0.0],
        [0.0,         0.0,         0.0,         0.0703125,   0.611979167, 0.315104167, 0.002604167, 0.0],
        [0.0,         0.0,         0.0,         0.0,         0.166666667, 0.666666667, 0.166666667, 0.0],
        [0.0,         0.0,         0.0,         0.0,         0.002604167, 0.315104167, 0.611979167, 0.0703125],
        [0.0,         0.0,         0.0,         0.0,         0.0,         0.166666667, 0.666666667, 0.166666667],
    ];
    let rows: Vec<&[f64]> = oracle.iter().map(|r| r.as_slice()).collect();
    assert_matches_oracle(&x, 6, 3, &rows, 1e-6, "c_degree3_dense");
}

/// (d) MULTI-FEATURE: 2 columns with different ranges (col0 in [0,1],
/// col1 in [-5,5]). Verifies per-feature knot vectors are computed
/// independently and the column blocks (`n_knots+degree-1` cols each) are
/// concatenated (`spline_transformer.rs:381-392`, sklearn `_polynomial.py:992`
/// `XBS[:, (i*n_splines):((i+1)*n_splines)]`).
/// Oracle: `SplineTransformer(n_knots=4, degree=2).fit_transform(X)`.
#[test]
fn probe_d_multi_feature_distinct_ranges() {
    let x = array![[0.0, -5.0], [0.3, -2.0], [0.5, 0.0], [0.7, 2.5], [1.0, 5.0]];
    // n_basis per feature = 4 + 2 - 1 = 5; total cols = 10.
    #[rustfmt::skip]
    let oracle: [[f64; 10]; 5] = [
        [0.5,   0.5,   0.0,     0.0,    0.0,   0.5,   0.5,   0.0,     0.0,    0.0],
        [0.005, 0.59,  0.405,   0.0,    0.0,   0.005, 0.59,  0.405,   0.0,    0.0],
        [0.0,   0.125, 0.75,    0.125,  0.0,   0.0,   0.125, 0.75,    0.125,  0.0],
        [0.0,   0.0,   0.405,   0.59,   0.005, 0.0,   0.0,   0.28125, 0.6875, 0.03125],
        [0.0,   0.0,   0.0,     0.5,    0.5,   0.0,   0.0,   0.0,     0.5,    0.5],
    ];
    let rows: Vec<&[f64]> = oracle.iter().map(|r| r.as_slice()).collect();
    assert_matches_oracle(&x, 4, 2, &rows, 1e-6, "d_multi_feature");
}

/// (e) RIGHT ENDPOINT x==max specifically — the boundary the fix re-keyed to the
/// base-interval right endpoint `knots[n_basis]`. scipy `BSpline.design_matrix`
/// includes the right endpoint of the base interval (closed on the right), so
/// the last row must be NON-zero (NOT all-zero under a naive half-open rule).
/// Two configs (degree=2/n_knots=3 and degree=3/n_knots=4); we check the FULL
/// matrix but the load-bearing assertion is the last (x==max) row.
/// Oracle: `SplineTransformer(...).fit_transform(X)` last row.
#[test]
fn probe_e_right_endpoint_xmax_degree2() {
    let x = array![[0.0], [0.5], [1.0]];
    #[rustfmt::skip]
    let oracle: [[f64; 4]; 3] = [
        [0.5, 0.5, 0.0, 0.0],
        [0.0, 0.5, 0.5, 0.0],
        [0.0, 0.0, 0.5, 0.5], // x==max==1.0: NON-zero, closed-right base interval
    ];
    let rows: Vec<&[f64]> = oracle.iter().map(|r| r.as_slice()).collect();
    assert_matches_oracle(&x, 3, 2, &rows, 1e-6, "e_xmax_degree2");
}

/// (e2) RIGHT ENDPOINT x==max, degree=3, n_knots=4. Last row (x==1.0) must be
/// [0,0,0, 1/6, 2/3, 1/6] per scipy, NOT all-zero.
#[test]
fn probe_e_right_endpoint_xmax_degree3() {
    let x = array![[0.0], [0.3], [0.6], [1.0]];
    #[rustfmt::skip]
    let oracle: [[f64; 6]; 4] = [
        [0.166666667, 0.666666667, 0.166666667, 0.0,         0.0,         0.0],
        [0.000166667, 0.221166667, 0.657166667, 0.1215,      0.0,         0.0],
        [0.0,         0.001333333, 0.282666667, 0.630666667, 0.085333333, 0.0],
        [0.0,         0.0,         0.0,         0.166666667, 0.666666667, 0.166666667], // x==max
    ];
    let rows: Vec<&[f64]> = oracle.iter().map(|r| r.as_slice()).collect();
    assert_matches_oracle(&x, 4, 3, &rows, 1e-6, "e_xmax_degree3");
}

/// (f) INTERIOR sample at the LEFT endpoint x==min, with a NON-[0,1] shifted
/// range ([2,5]) to ensure the min offset is honoured. degree=1, n_knots=3.
/// First row (x==min==2.0) must be [1, 0, 0].
/// Oracle: `SplineTransformer(n_knots=3, degree=1).fit_transform(X)`.
#[test]
fn probe_f_left_endpoint_xmin_shifted_range() {
    let x = array![[2.0], [3.0], [5.0]];
    #[rustfmt::skip]
    let oracle: [[f64; 3]; 3] = [
        [1.0,         0.0,         0.0], // x==min==2.0
        [0.333333333, 0.666666667, 0.0],
        [0.0,         0.0,         1.0],
    ];
    let rows: Vec<&[f64]> = oracle.iter().map(|r| r.as_slice()).collect();
    assert_matches_oracle(&x, 3, 1, &rows, 1e-6, "f_xmin_shifted");
}

// ---------------------------------------------------------------------------
// REQ-6 — parameter-contract DIVERGENCE (blocker #1336): sklearn allows
// `degree == 0` (piecewise-constant B-spline) via `_parameter_constraints`
// `degree: Interval(Integral, 0, None, closed="left")` (`_polynomial.py:705`),
// but ferrolearn previously rejected it. ferrolearn's `n_samples >= 2`
// requirement is NOT a divergence — sklearn also enforces it via
// `_validate_data(..., ensure_min_samples=2)` (`_polynomial.py:830`).
// ---------------------------------------------------------------------------

/// DIV (REQ-6, blocker #1336): `degree == 0` must be ACCEPTED and produce the
/// scipy piecewise-constant B-spline design matrix, not a rejected parameter.
///
/// Live sklearn 1.5.2 ORACLE (run from /tmp, R-CHAR-3 — NOT copied from
/// ferrolearn):
/// ```text
/// import numpy as np; from sklearn.preprocessing import SplineTransformer
/// X = np.array([[0.],[1.],[2.],[3.],[4.]])
/// st = SplineTransformer(n_knots=3, degree=0, knots='uniform').fit(X)
/// st.transform(np.array([[0.5],[2.5]]))
/// # -> shape (2, 2); [[1.0, 0.0], [0.0, 1.0]]
/// ```
/// Columns = `n_knots + degree - 1 = 3 + 0 - 1 = 2`. 0.5 falls in the first
/// knot interval -> `[1, 0]`; 2.5 in the second -> `[0, 1]`.
#[test]
fn spline_degree_zero_allowed_matches_sklearn() -> Result<(), FerroError> {
    let st = SplineTransformer::<f64>::new(3, 0, KnotStrategy::Uniform);
    let x = array![[0.0], [1.0], [2.0], [3.0], [4.0]];
    let fitted = st.fit(&x, &())?;

    let q = array![[0.5], [2.5]];
    let out = fitted.transform(&q)?;

    // n_knots + degree - 1 = 3 + 0 - 1 = 2 columns.
    assert_eq!(out.ncols(), 2, "degree-0 column count (n_knots+degree-1)");
    assert_eq!(out.nrows(), 2, "row count");

    // sklearn 1.5.2 live oracle.
    let oracle: [[f64; 2]; 2] = [[1.0, 0.0], [0.0, 1.0]];
    for (i, orow) in oracle.iter().enumerate() {
        for (j, &b) in orow.iter().enumerate() {
            let a = out[[i, j]];
            assert!(
                (a - b).abs() < 1e-9,
                "row {i} col {j}: ferrolearn {a} vs sklearn {b} (diff {})",
                (a - b).abs()
            );
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// GREEN GUARDS — structural B-spline PROPERTIES that genuinely HOLD (REQ-1).
// These pin what is SHIPPED: column count, partition of unity, non-negativity.
// ---------------------------------------------------------------------------

/// GREEN GUARD (REQ-1): output column count == n_knots + degree - 1 per feature.
/// sklearn `n_splines = n_knots + self.degree - 1` (`_polynomial.py:875`).
#[test]
#[allow(
    clippy::assertions_on_constants,
    reason = "error arm fails loudly without panic!/unwrap (anti-pattern gate)"
)]
fn green_guard_column_count_per_feature() {
    let x = array![[0.0], [0.25], [0.5], [0.75], [1.0]];
    let st = SplineTransformer::<f64>::new(5, 3, KnotStrategy::Uniform);
    let fitted = match st.fit(&x, &()) {
        Ok(f) => f,
        Err(e) => {
            assert!(false, "fit failed: {e}");
            return;
        }
    };
    let out = match fitted.transform(&x) {
        Ok(o) => o,
        Err(e) => {
            assert!(false, "transform failed: {e}");
            return;
        }
    };
    assert_eq!(out.ncols(), 7, "single feature -> 7 basis columns");

    let x2 = array![[0.0, 10.0], [0.5, 15.0], [1.0, 20.0]];
    let st2 = SplineTransformer::<f64>::new(3, 2, KnotStrategy::Uniform);
    let fitted2 = match st2.fit(&x2, &()) {
        Ok(f) => f,
        Err(e) => {
            assert!(false, "fit failed: {e}");
            return;
        }
    };
    let out2 = match fitted2.transform(&x2) {
        Ok(o) => o,
        Err(e) => {
            assert!(false, "transform failed: {e}");
            return;
        }
    };
    assert_eq!(out2.ncols(), 8, "two features -> 8 basis columns");
}

/// GREEN GUARD (REQ-1): partition of unity — each row sums to ~1.0 over the
/// base interval. Holds for any valid B-spline basis.
#[test]
#[allow(
    clippy::assertions_on_constants,
    reason = "error arm fails loudly without panic!/unwrap (anti-pattern gate)"
)]
fn green_guard_partition_of_unity() {
    let x = array![[0.0], [0.25], [0.5], [0.75], [1.0]];
    let st = SplineTransformer::<f64>::new(5, 3, KnotStrategy::Uniform);
    let fitted = match st.fit(&x, &()) {
        Ok(f) => f,
        Err(e) => {
            assert!(false, "fit failed: {e}");
            return;
        }
    };
    let out = match fitted.transform(&x) {
        Ok(o) => o,
        Err(e) => {
            assert!(false, "transform failed: {e}");
            return;
        }
    };
    for i in 0..out.nrows() {
        let row_sum: f64 = out.row(i).iter().sum();
        assert!(
            (row_sum - 1.0).abs() < 1e-9,
            "row {i} sum {row_sum} should be ~1.0 (partition of unity)"
        );
    }
}

/// GREEN GUARD (REQ-1): non-negativity — every basis value >= 0.
#[test]
#[allow(
    clippy::assertions_on_constants,
    reason = "error arm fails loudly without panic!/unwrap (anti-pattern gate)"
)]
fn green_guard_non_negativity() {
    let x = array![[0.0], [0.3], [0.6], [1.0]];
    let st = SplineTransformer::<f64>::new(4, 3, KnotStrategy::Uniform);
    let fitted = match st.fit(&x, &()) {
        Ok(f) => f,
        Err(e) => {
            assert!(false, "fit failed: {e}");
            return;
        }
    };
    let out = match fitted.transform(&x) {
        Ok(o) => o,
        Err(e) => {
            assert!(false, "transform failed: {e}");
            return;
        }
    };
    for v in &out {
        assert!(*v >= -1e-12, "basis value should be non-negative, got {v}");
    }
}
