//! Divergence tests: ferrolearn `MinMaxScaler` vs scikit-learn 1.5.2
//! `sklearn/preprocessing/_data.py` `class MinMaxScaler` (`:291`).
//!
//! All expected values come from the LIVE sklearn 1.5.2 oracle (run from /tmp)
//! or a sklearn `file:line` symbolic constant — never copied from the
//! ferrolearn side (R-CHAR-3).
//!
//! Two flavors here:
//!   * GREEN guards — pin the SHIPPED, in-regime (non-constant) REQ-1/REQ-3
//!     behavior; these PASS today and lock parity.
//!   * RED pins — the constant/zero-range column divergence; these FAIL today
//!     and must be flipped by the generator (tracking blocker #1170).

use ferrolearn_core::traits::{Fit, FitTransform};
use ferrolearn_preprocess::MinMaxScaler;
use ndarray::array;

// ===========================================================================
// RED PINS — constant / zero-range column divergence (the fixable one)
// ===========================================================================

/// Divergence: ferrolearn's `FittedMinMaxScaler::transform` diverges from
/// `sklearn/preprocessing/_data.py:508,511,543-544` for a constant
/// (zero-range) column.
///
/// sklearn: `data_range = data_max - data_min = 0`, then
/// `_handle_zeros_in_scale(data_range) -> 1` (`_data.py:88`), so
/// `scale_ = (fr[1]-fr[0]) / 1 = range_width` (`:508`),
/// `min_ = fr[0] - data_min*scale_` (`:511`), and on transform
/// `X *= scale_; X += min_` (`:543-544`) gives
/// `transform(data_min) = data_min*range_width + fr[0] - data_min*range_width = fr[0]`.
/// Live oracle (default range `(0,1)`):
///   `MinMaxScaler().fit_transform([[5.,1.],[5.,2.],[5.,3.]])`
///   -> `[[0.0,0.0],[0.0,0.5],[0.0,1.0]]` (constant col 0 -> 0.0 = feature_range[0]).
/// ferrolearn: `if span == F::zero() { continue }` leaves the column UNCHANGED,
/// returning the original 5.0 (`min_max_scaler.rs:202-205`).
/// Tracking: #1170
#[test]
fn divergence_constant_column_default_range_maps_to_zero() {
    // Live sklearn 1.5.2 oracle value: feature_range[0] = 0.0 (default).
    const SK_FEATURE_RANGE_LOW: f64 = 0.0;

    let scaler = MinMaxScaler::<f64>::new();
    let x = array![[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]];
    let scaled = scaler.fit_transform(&x).unwrap();

    // Constant column 0 must map to feature_range[0] = 0.0 for every row.
    for i in 0..3 {
        assert!(
            (scaled[[i, 0]] - SK_FEATURE_RANGE_LOW).abs() < 1e-12,
            "constant col 0 row {i}: sklearn maps to {SK_FEATURE_RANGE_LOW}, ferrolearn returned {}",
            scaled[[i, 0]]
        );
    }
    // Non-constant column 1 is unaffected by the bug; sanity per oracle.
    assert!((scaled[[0, 1]] - 0.0).abs() < 1e-12);
    assert!((scaled[[1, 1]] - 0.5).abs() < 1e-12);
    assert!((scaled[[2, 1]] - 1.0).abs() < 1e-12);
}

/// Divergence (custom range): the constant column maps to `feature_range[0]`,
/// not to 0 and not to its original value.
/// Live oracle:
///   `MinMaxScaler(feature_range=(-1,1)).fit_transform([[5.,1.],[5.,2.],[5.,3.]])`
///   -> `[[-1.0,-1.0],[-1.0,0.0],[-1.0,1.0]]` (constant col 0 -> -1.0).
/// sklearn cite: `_data.py:511` (`min_ = fr[0] - data_min*scale_`) +
/// `:543-544` give `transform(data_min) = fr[0] = -1.0`.
/// ferrolearn leaves col 0 unchanged at 5.0.
/// Tracking: #1170
#[test]
fn divergence_constant_column_custom_range_maps_to_range_low() {
    // Live sklearn 1.5.2 oracle value: feature_range[0] = -1.0 for (-1, 1).
    const SK_FEATURE_RANGE_LOW: f64 = -1.0;

    let scaler = MinMaxScaler::<f64>::with_feature_range(-1.0, 1.0).unwrap();
    let x = array![[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]];
    let scaled = scaler.fit_transform(&x).unwrap();

    for i in 0..3 {
        assert!(
            (scaled[[i, 0]] - SK_FEATURE_RANGE_LOW).abs() < 1e-12,
            "constant col 0 row {i}: sklearn maps to {SK_FEATURE_RANGE_LOW}, ferrolearn returned {}",
            scaled[[i, 0]]
        );
    }
}

// ===========================================================================
// GREEN GUARDS — shipped in-regime parity (REQ-1, REQ-3). Pass today.
// ===========================================================================

/// REQ-1 green guard: non-constant default-range value match.
/// Live oracle: `MinMaxScaler().fit_transform([[1.,10.],[2.,20.],[3.,30.]])`
///   -> `[[0.0,0.0],[0.5,0.5],[1.0,1.0]]`.
#[test]
fn req1_default_range_value_match() {
    // Live sklearn 1.5.2 oracle.
    let sk = [[0.0_f64, 0.0], [0.5, 0.5], [1.0, 1.0]];

    let scaler = MinMaxScaler::<f64>::new();
    let x = array![[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]];
    let scaled = scaler.fit_transform(&x).unwrap();
    for i in 0..3 {
        for j in 0..2 {
            assert!(
                (scaled[[i, j]] - sk[i][j]).abs() < 1e-12,
                "[{i},{j}] sklearn {} ferrolearn {}",
                sk[i][j],
                scaled[[i, j]]
            );
        }
    }
}

/// REQ-1 green guard: custom range `(-1, 1)` value match.
/// Live oracle: `MinMaxScaler(feature_range=(-1,1)).fit_transform([[0.],[5.],[10.]])`
///   -> `[[-1.0],[0.0],[1.0]]`.
#[test]
fn req1_custom_range_value_match() {
    // Live sklearn 1.5.2 oracle.
    let sk = [-1.0_f64, 0.0, 1.0];

    let scaler = MinMaxScaler::<f64>::with_feature_range(-1.0, 1.0).unwrap();
    let x = array![[0.0], [5.0], [10.0]];
    let scaled = scaler.fit_transform(&x).unwrap();
    for i in 0..3 {
        assert!(
            (scaled[[i, 0]] - sk[i]).abs() < 1e-12,
            "[{i}] sklearn {} ferrolearn {}",
            sk[i],
            scaled[[i, 0]]
        );
    }
}

/// REQ-1 green guard: `data_min_` / `data_max_` fitted attributes.
/// Live oracle: `MinMaxScaler().fit([[1.,10.],[2.,20.],[3.,30.]])`
///   -> `data_min_ = [1.0, 10.0]`, `data_max_ = [3.0, 30.0]`.
#[test]
fn req1_data_min_max_attributes() {
    // Live sklearn 1.5.2 oracle.
    let sk_min = [1.0_f64, 10.0];
    let sk_max = [3.0_f64, 30.0];

    let scaler = MinMaxScaler::<f64>::new();
    let x = array![[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]];
    let fitted = scaler.fit(&x, &()).unwrap();
    for j in 0..2 {
        assert!((fitted.data_min()[j] - sk_min[j]).abs() < 1e-12);
        assert!((fitted.data_max()[j] - sk_max[j]).abs() < 1e-12);
    }
}

/// REQ-1 green guard: non-trivial fixture with negative + decimal values.
/// Live oracle:
///   `MinMaxScaler().fit_transform([[-2.5,0.5],[-1.0,1.5],[3.0,-4.0]])`
///   -> `[[0.0, 0.8181818181818182],
///        [0.27272727272727276, 1.0],
///        [1.0, 0.0]]`.
#[test]
fn req1_nontrivial_negative_decimal_value_match() {
    // Live sklearn 1.5.2 oracle.
    let sk = [
        [0.0_f64, 0.818_181_818_181_818_2],
        [0.272_727_272_727_272_76, 1.0],
        [1.0, 0.0],
    ];

    let scaler = MinMaxScaler::<f64>::new();
    let x = array![[-2.5, 0.5], [-1.0, 1.5], [3.0, -4.0]];
    let scaled = scaler.fit_transform(&x).unwrap();
    for i in 0..3 {
        for j in 0..2 {
            assert!(
                (scaled[[i, j]] - sk[i][j]).abs() < 1e-12,
                "[{i},{j}] sklearn {} ferrolearn {}",
                sk[i][j],
                scaled[[i, j]]
            );
        }
    }
}

/// REQ-3 green guard: feature_range validation. sklearn raises ValueError when
/// `feature_range[0] >= feature_range[1]` (`_data.py:475-479`,
/// "Minimum of desired feature range must be smaller than maximum").
/// Live oracle confirms both `(1,0)` and `(1,1)` raise. ferrolearn
/// `with_feature_range` returns `Err` for both.
#[test]
fn req3_feature_range_validation_rejects() {
    // sklearn: (1,0) -> ValueError; (1,1) -> ValueError (_data.py:475-479).
    assert!(MinMaxScaler::<f64>::with_feature_range(1.0, 0.0).is_err());
    assert!(MinMaxScaler::<f64>::with_feature_range(1.0, 1.0).is_err());
}

// ===========================================================================
// CONVERGENCE GREEN GUARDS — constant-column fix completeness (blocker #1170)
// Added by the critic to verify the fix has no over-reach and covers the
// edge cases (positive custom range, mixed fixture, negative/zero constant,
// single-row fit). All expected values come from the LIVE sklearn 1.5.2
// oracle (run from /tmp). sklearn maps a constant column to feature_range[0]
// via `_handle_zeros_in_scale` (data_range 0 -> 1, `_data.py:88`,`:507-511`).
// ===========================================================================

/// Convergence guard: POSITIVE custom range `(2, 5)`. The constant column maps
/// to `feature_range[0] = 2.0` (NOT 0, NOT its original 5.0); the non-constant
/// column still scales across the full range.
/// Live oracle: `MinMaxScaler(feature_range=(2,5)).fit_transform([[5.,1.],[5.,2.],[5.,3.]])`
///   -> `[[2.0,2.0],[2.0,3.5],[2.0,5.0]]`.
/// sklearn cite: `_data.py:511` (`min_ = fr[0] - data_min*scale_`) -> `fr[0]=2.0`
/// for the constant col; `:507-508` affine for col 1.
#[test]
fn req2_constant_column_positive_custom_range_maps_to_range_low() {
    // Live sklearn 1.5.2 oracle.
    let sk = [[2.0_f64, 2.0], [2.0, 3.5], [2.0, 5.0]];

    let scaler = MinMaxScaler::<f64>::with_feature_range(2.0, 5.0).unwrap();
    let x = array![[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]];
    let scaled = scaler.fit_transform(&x).unwrap();
    for i in 0..3 {
        for j in 0..2 {
            assert!(
                (scaled[[i, j]] - sk[i][j]).abs() < 1e-12,
                "[{i},{j}] sklearn {} ferrolearn {}",
                sk[i][j],
                scaled[[i, j]]
            );
        }
    }
}

/// Convergence guard: MIXED fixture — one constant col + one non-constant col,
/// both columns verified simultaneously so the fix neither over-reaches onto
/// the scaling column nor under-reaches on the constant column.
/// Live oracle: `MinMaxScaler().fit_transform([[5.,1.],[5.,2.],[5.,3.]])`
///   -> `[[0.0,0.0],[0.0,0.5],[0.0,1.0]]`.
#[test]
fn req2_mixed_constant_and_scaling_columns() {
    // Live sklearn 1.5.2 oracle.
    let sk = [[0.0_f64, 0.0], [0.0, 0.5], [0.0, 1.0]];

    let scaler = MinMaxScaler::<f64>::new();
    let x = array![[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]];
    let scaled = scaler.fit_transform(&x).unwrap();
    for i in 0..3 {
        for j in 0..2 {
            assert!(
                (scaled[[i, j]] - sk[i][j]).abs() < 1e-12,
                "[{i},{j}] sklearn {} ferrolearn {}",
                sk[i][j],
                scaled[[i, j]]
            );
        }
    }
}

/// Convergence guard: constant column at a NON-fr[0] value (0.0 and -3.0) maps
/// to `feature_range[0]` regardless of the constant's value. This pins that the
/// fix sets the column to `range_min`, not to the data value or to 0 by
/// accident of the constant happening to be 0.
/// Live oracle:
///   `MinMaxScaler().fit_transform([[0.,1.],[0.,2.]])` -> `[[0.0,0.0],[0.0,1.0]]`
///   `MinMaxScaler().fit_transform([[-3.],[-3.]])`     -> `[[0.0],[0.0]]`
#[test]
fn req2_negative_and_zero_constant_map_to_range_low() {
    // Case A: constant col 0 at value 0.0 -> fr[0] = 0.0; col 1 scales.
    let sk_a = [[0.0_f64, 0.0], [0.0, 1.0]];
    let scaler = MinMaxScaler::<f64>::new();
    let xa = array![[0.0, 1.0], [0.0, 2.0]];
    let sa = scaler.fit_transform(&xa).unwrap();
    for i in 0..2 {
        for j in 0..2 {
            assert!(
                (sa[[i, j]] - sk_a[i][j]).abs() < 1e-12,
                "A[{i},{j}] sklearn {} ferrolearn {}",
                sk_a[i][j],
                sa[[i, j]]
            );
        }
    }

    // Case B: constant col at -3.0 (NEGATIVE, != fr[0]) -> fr[0] = 0.0.
    let sk_b = [0.0_f64, 0.0];
    let scaler2 = MinMaxScaler::<f64>::new();
    let xb = array![[-3.0], [-3.0]];
    let sb = scaler2.fit_transform(&xb).unwrap();
    for i in 0..2 {
        assert!(
            (sb[[i, 0]] - sk_b[i]).abs() < 1e-12,
            "B[{i}] sklearn {} ferrolearn {} (constant -3 must map to fr[0]=0)",
            sk_b[i],
            sb[[i, 0]]
        );
    }
}

/// Convergence guard: SINGLE-ROW fit (n_samples == 1) makes EVERY column
/// constant (data_max == data_min), so all columns map to `feature_range[0]`.
/// Live oracle: `MinMaxScaler().fit_transform([[5.,7.]])` -> `[[0.0,0.0]]`.
#[test]
fn req2_single_row_fit_all_columns_map_to_range_low() {
    // Live sklearn 1.5.2 oracle.
    let sk = [0.0_f64, 0.0];

    let scaler = MinMaxScaler::<f64>::new();
    let x = array![[5.0, 7.0]];
    let scaled = scaler.fit_transform(&x).unwrap();
    for j in 0..2 {
        assert!(
            (scaled[[0, j]] - sk[j]).abs() < 1e-12,
            "col {j} sklearn {} ferrolearn {} (single-row fit -> constant -> fr[0])",
            sk[j],
            scaled[[0, j]]
        );
    }
}
