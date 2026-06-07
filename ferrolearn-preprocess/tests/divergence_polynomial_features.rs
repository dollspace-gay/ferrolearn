//! ACToR critic: oracle-grounded audit of `ferrolearn-preprocess`'s
//! `PolynomialFeatures` against scikit-learn 1.5.2
//! `sklearn/preprocessing/_polynomial.py` `class PolynomialFeatures` (`:99`).
//!
//! All expected values are derived from the LIVE sklearn 1.5.2 oracle, run from
//! `/tmp` (R-CHAR-3): NEVER literal-copied from the ferrolearn side.
//!
//! Oracle commands (sklearn 1.5.2 == `python3 -c "import sklearn; sklearn.__version__"`):
//! ```text
//! # REQ-1 VALUE + COLUMN ORDER (the subtle part = sklearn _combinations itertools order):
//! python3 -c "from sklearn.preprocessing import PolynomialFeatures; \
//!   print(PolynomialFeatures(2).fit_transform([[2.,3.]]).tolist())"
//!   -> [[1.0, 2.0, 3.0, 4.0, 6.0, 9.0]]
//! python3 -c "from sklearn.preprocessing import PolynomialFeatures; \
//!   print(PolynomialFeatures(2).fit_transform([[2.,3.,5.]]).tolist())"
//!   -> [[1.0, 2.0, 3.0, 5.0, 4.0, 6.0, 10.0, 9.0, 15.0, 25.0]]
//! python3 -c "from sklearn.preprocessing import PolynomialFeatures; \
//!   print(PolynomialFeatures(2, interaction_only=True).fit_transform([[2.,3.,5.]]).tolist())"
//!   -> [[1.0, 2.0, 3.0, 5.0, 6.0, 10.0, 15.0]]
//! python3 -c "from sklearn.preprocessing import PolynomialFeatures; \
//!   print(PolynomialFeatures(3, interaction_only=True, include_bias=False).fit_transform([[2.,3.,5.]]).tolist())"
//!   -> [[2.0, 3.0, 5.0, 6.0, 10.0, 15.0, 30.0]]
//! python3 -c "from sklearn.preprocessing import PolynomialFeatures; \
//!   print(PolynomialFeatures(3, include_bias=False).fit_transform([[2.]]).tolist())"
//!   -> [[2.0, 4.0, 8.0]]
//! python3 -c "from sklearn.preprocessing import PolynomialFeatures; \
//!   print(PolynomialFeatures(2, include_bias=False).fit_transform([[1.,2.],[3.,4.]]).tolist())"
//!   -> [[1.0, 2.0, 1.0, 2.0, 4.0], [3.0, 4.0, 9.0, 12.0, 16.0]]
//!
//! # REQ-8 input validation — sklearn transform routes through _validate_data
//! # (_polynomial.py:433-435, defaults force_all_finite=True / ensure_min_samples=1):
//! python3 -c "import numpy as np; from sklearn.preprocessing import PolynomialFeatures; \
//!   PolynomialFeatures().fit_transform(np.array([[float('nan'),1.]]))"
//!   -> ValueError: ... does not accept missing values encoded as NaN ...
//! python3 -c "import numpy as np; from sklearn.preprocessing import PolynomialFeatures; \
//!   PolynomialFeatures().fit_transform(np.array([[float('inf'),1.]]))"
//!   -> ValueError: Input X contains infinity or a value too large for dtype('float64').
//! python3 -c "import numpy as np; from sklearn.preprocessing import PolynomialFeatures; \
//!   PolynomialFeatures().fit_transform(np.array([[float('-inf'),1.]]))"
//!   -> ValueError: Input X contains infinity or a value too large for dtype('float64').
//! python3 -c "import numpy as np; from sklearn.preprocessing import PolynomialFeatures; \
//!   PolynomialFeatures().fit_transform(np.empty((0,2)))"
//!   -> ValueError: Found array with 0 sample(s) (shape=(0, 2)) while a minimum of 1 is required ...
//! python3 -c "import numpy as np; from sklearn.preprocessing import PolynomialFeatures; \
//!   PolynomialFeatures().fit_transform(np.empty((2,0)))"
//!   -> ValueError: Found array with 0 feature(s) (shape=(2, 0)) while a minimum of 1 is required ...
//! ```
//!
//! sklearn's `check_array` enforces validation in the ORDER:
//!   samples (`ensure_min_samples`) -> features (`ensure_min_features`) -> finite
//!   (`force_all_finite`). The fixer should match that order (consistent with
//!   binarizer.rs REQ-9): 0-rows -> `InsufficientSamples`, 0-features ->
//!   `InvalidParameter` (already present), non-finite -> `InvalidParameter`.

use ferrolearn_core::error::FerroError;
use ferrolearn_core::traits::{Fit, Transform};
use ferrolearn_preprocess::PolynomialFeatures;
use ndarray::{Array2, array};

// ---------------------------------------------------------------------------
// GREEN guards — REQ-1 dense int-degree VALUE + COLUMN ORDER vs the LIVE
// sklearn oracle (R-CHAR-3: expected values are the oracle outputs quoted
// above). The column ORDER (sklearn `_combinations` itertools order, bias ->
// degree-ascending -> lexicographic-within-degree, `_polynomial.py:209-220`)
// is the load-bearing assertion — values are pinned column-for-column.
// ---------------------------------------------------------------------------

/// Guard: default (degree=2, full, bias), 2 features. Live oracle
/// `PolynomialFeatures(2).fit_transform([[2.,3.]])` == `[[1,2,3,4,6,9]]`
/// (bias, a, b, a^2, a*b, b^2). ferrolearn `new(2,false,true)` matches
/// column-for-column.
#[test]
fn guard_degree2_two_features_full_bias_matches_sklearn_oracle() {
    // Live oracle: PolynomialFeatures(2).fit_transform([[2.,3.]])
    let sklearn_expected: Array2<f64> = array![[1.0, 2.0, 3.0, 4.0, 6.0, 9.0]];
    let poly = PolynomialFeatures::<f64>::new(2, false, true).unwrap();
    let x = array![[2.0, 3.0]];
    let out = poly.transform(&x).unwrap();
    assert_eq!(out, sklearn_expected);
}

/// Guard: degree=2, 3 features, full+bias — ORDER-SENSITIVE. Live oracle
/// `PolynomialFeatures(2).fit_transform([[2.,3.,5.]])` ==
/// `[[1,2,3,5,4,6,10,9,15,25]]` (bias, a, b, c, a^2, ab, ac, b^2, bc, c^2).
/// ferrolearn `new(2,false,true)` must reproduce the itertools
/// `combinations_with_replacement` order column-for-column.
#[test]
fn guard_degree2_three_features_full_bias_order_matches_sklearn_oracle() {
    // Live oracle: PolynomialFeatures(2).fit_transform([[2.,3.,5.]])
    let sklearn_expected: Array2<f64> =
        array![[1.0, 2.0, 3.0, 5.0, 4.0, 6.0, 10.0, 9.0, 15.0, 25.0]];
    let poly = PolynomialFeatures::<f64>::new(2, false, true).unwrap();
    let x = array![[2.0, 3.0, 5.0]];
    let out = poly.transform(&x).unwrap();
    assert_eq!(out, sklearn_expected);
}

/// Guard: degree=2, 3 features, interaction_only + bias — ORDER-SENSITIVE.
/// Live oracle
/// `PolynomialFeatures(2, interaction_only=True).fit_transform([[2.,3.,5.]])`
/// == `[[1,2,3,5,6,10,15]]` (bias, a, b, c, ab, ac, bc). ferrolearn
/// `new(2,true,true)` must reproduce the itertools `combinations` order.
#[test]
fn guard_degree2_three_features_interaction_only_order_matches_sklearn_oracle() {
    // Live oracle: PolynomialFeatures(2, interaction_only=True).fit_transform([[2.,3.,5.]])
    let sklearn_expected: Array2<f64> = array![[1.0, 2.0, 3.0, 5.0, 6.0, 10.0, 15.0]];
    let poly = PolynomialFeatures::<f64>::new(2, true, true).unwrap();
    let x = array![[2.0, 3.0, 5.0]];
    let out = poly.transform(&x).unwrap();
    assert_eq!(out, sklearn_expected);
}

/// Guard: degree=3, 3 features, interaction_only, NO bias — ORDER-SENSITIVE.
/// Live oracle
/// `PolynomialFeatures(3, interaction_only=True, include_bias=False).fit_transform([[2.,3.,5.]])`
/// == `[[2,3,5,6,10,15,30]]` (a, b, c, ab, ac, bc, abc). ferrolearn
/// `new(3,true,false)` matches.
#[test]
fn guard_degree3_interaction_only_no_bias_order_matches_sklearn_oracle() {
    // Live oracle:
    // PolynomialFeatures(3, interaction_only=True, include_bias=False).fit_transform([[2.,3.,5.]])
    let sklearn_expected: Array2<f64> = array![[2.0, 3.0, 5.0, 6.0, 10.0, 15.0, 30.0]];
    let poly = PolynomialFeatures::<f64>::new(3, true, false).unwrap();
    let x = array![[2.0, 3.0, 5.0]];
    let out = poly.transform(&x).unwrap();
    assert_eq!(out, sklearn_expected);
}

/// Guard: no-bias + degree=3, single feature — VALUE + ORDER. Live oracle
/// `PolynomialFeatures(3, include_bias=False).fit_transform([[2.]])` ==
/// `[[2,4,8]]` (a, a^2, a^3). ferrolearn `new(3,false,false)` matches.
#[test]
fn guard_degree3_single_feature_no_bias_matches_sklearn_oracle() {
    // Live oracle: PolynomialFeatures(3, include_bias=False).fit_transform([[2.]])
    let sklearn_expected: Array2<f64> = array![[2.0, 4.0, 8.0]];
    let poly = PolynomialFeatures::<f64>::new(3, false, false).unwrap();
    let x = array![[2.0]];
    let out = poly.transform(&x).unwrap();
    assert_eq!(out, sklearn_expected);
}

/// Guard: multi-row, degree=2, 2 features, no bias — VALUE + ORDER per row.
/// Live oracle
/// `PolynomialFeatures(2, include_bias=False).fit_transform([[1.,2.],[3.,4.]])`
/// == `[[1,2,1,2,4],[3,4,9,12,16]]` (a, b, a^2, ab, b^2). ferrolearn
/// `new(2,false,false)` matches.
#[test]
fn guard_multi_row_degree2_no_bias_matches_sklearn_oracle() {
    // Live oracle:
    // PolynomialFeatures(2, include_bias=False).fit_transform([[1.,2.],[3.,4.]])
    let sklearn_expected: Array2<f64> =
        array![[1.0, 2.0, 1.0, 2.0, 4.0], [3.0, 4.0, 9.0, 12.0, 16.0]];
    let poly = PolynomialFeatures::<f64>::new(2, false, false).unwrap();
    let x = array![[1.0, 2.0], [3.0, 4.0]];
    let out = poly.transform(&x).unwrap();
    assert_eq!(out, sklearn_expected);
}

// ---------------------------------------------------------------------------
// GREEN guard — REQ-8 zero-FEATURES: both sklearn and ferrolearn reject.
// This is NOT a pin (ferrolearn already errors), but a guard confirming the
// live oracle raises and that ferrolearn matches "raises" (note the ORDER:
// sklearn checks min-features AFTER min-samples; here n_samples=2>0 so the
// features check fires).
// ---------------------------------------------------------------------------

/// Guard: 0 features (2 rows, 0 cols). Live oracle
/// `PolynomialFeatures().fit_transform(np.empty((2,0)))` raises
/// `ValueError: Found array with 0 feature(s) ... minimum of 1 is required`
/// (`check_array` `ensure_min_features=1`). ferrolearn `transform` ALREADY
/// returns `Err(FerroError::InvalidParameter)` (`polynomial_features.rs:184`).
/// Both reject — GREEN guard.
#[test]
fn guard_zero_features_errors_like_sklearn_oracle() {
    // Live oracle: PolynomialFeatures().fit_transform(np.empty((2,0))) -> ValueError
    let poly = PolynomialFeatures::<f64>::new(2, false, true).unwrap();
    let x: Array2<f64> = Array2::zeros((2, 0));
    let result = poly.transform(&x);
    assert!(
        matches!(result, Err(FerroError::InvalidParameter { .. })),
        "sklearn raises ValueError (0 feature(s), check_array ensure_min_features); \
         ferrolearn must reject with InvalidParameter, got {result:?}"
    );
}

// ---------------------------------------------------------------------------
// RED pins — REQ-8 input-validation DIVERGENCE (un-ignored release blocker,
// tracking #1180). sklearn `transform` routes through `_validate_data`
// (`_polynomial.py:433-435`) whose `check_array` defaults reject non-finite
// (NaN / +/-inf, `force_all_finite=True`) and zero-row (`ensure_min_samples=1`)
// input with `ValueError`. ferrolearn's `transform` has NO finite check and NO
// min-samples check — NaN/inf flow through the product loop and zero-row input
// returns `Ok` with an empty matrix. These tests assert ferrolearn returns
// `Err`; they FAIL against the current implementation (the divergence).
// ---------------------------------------------------------------------------

/// Divergence: `ferrolearn::PolynomialFeatures::transform` diverges from
/// `sklearn/preprocessing/_polynomial.py:433` (`self._validate_data(...)`,
/// whose `check_array` defaults reject non-finite) for input containing NaN.
/// sklearn raises `ValueError` (does not accept NaN); ferrolearn returns
/// `Ok` (NaN flows through `acc * x[[i,j]]` -> NaN products,
/// `polynomial_features.rs:198`).
/// Tracking: #1180
#[test]
fn divergence_nan_should_error_like_sklearn() {
    // Live oracle: PolynomialFeatures().fit_transform([[nan,1.]]) -> ValueError
    let poly = PolynomialFeatures::<f64>::new(2, false, true).unwrap();
    let x = array![[f64::NAN, 1.0]];
    let result = poly.transform(&x);
    assert!(
        result.is_err(),
        "sklearn raises ValueError on NaN input (_polynomial.py:433 _validate_data, \
         force_all_finite=True); ferrolearn returned Ok({:?})",
        result.ok()
    );
}

/// Divergence: `ferrolearn::PolynomialFeatures::transform` diverges from
/// `sklearn/preprocessing/_polynomial.py:433` for input containing +inf.
/// sklearn raises `ValueError: Input X contains infinity ...`; ferrolearn
/// returns `Ok` (inf flows through the product loop).
/// Tracking: #1180
#[test]
fn divergence_pos_inf_should_error_like_sklearn() {
    // Live oracle: PolynomialFeatures().fit_transform([[inf,1.]]) -> ValueError
    let poly = PolynomialFeatures::<f64>::new(2, false, true).unwrap();
    let x = array![[f64::INFINITY, 1.0]];
    let result = poly.transform(&x);
    assert!(
        result.is_err(),
        "sklearn raises ValueError on +inf input (_polynomial.py:433 _validate_data, \
         force_all_finite=True); ferrolearn returned Ok({:?})",
        result.ok()
    );
}

/// Divergence: `ferrolearn::PolynomialFeatures::transform` diverges from
/// `sklearn/preprocessing/_polynomial.py:433` for input containing -inf.
/// sklearn raises `ValueError: Input X contains infinity ...`; ferrolearn
/// returns `Ok` (-inf flows through the product loop).
/// Tracking: #1180
#[test]
fn divergence_neg_inf_should_error_like_sklearn() {
    // Live oracle: PolynomialFeatures().fit_transform([[-inf,1.]]) -> ValueError
    let poly = PolynomialFeatures::<f64>::new(2, false, true).unwrap();
    let x = array![[f64::NEG_INFINITY, 1.0]];
    let result = poly.transform(&x);
    assert!(
        result.is_err(),
        "sklearn raises ValueError on -inf input (_polynomial.py:433 _validate_data, \
         force_all_finite=True); ferrolearn returned Ok({:?})",
        result.ok()
    );
}

/// Divergence: `ferrolearn::PolynomialFeatures::transform` diverges from
/// `sklearn/preprocessing/_polynomial.py:433` (`check_array`
/// `ensure_min_samples=1`) for a zero-row input. sklearn raises
/// `ValueError: Found array with 0 sample(s) (shape=(0, 2)) while a minimum of
/// 1 is required`; ferrolearn returns `Ok` of an empty `(0, n_out)` matrix
/// (the product loop iterates over 0 samples). The fixer should reject with
/// `InsufficientSamples` (consistent with binarizer.rs REQ-9), checked BEFORE
/// the features check per sklearn `check_array` order.
/// Tracking: #1180
#[test]
fn divergence_zero_rows_should_error_like_sklearn() {
    // Live oracle: PolynomialFeatures().fit_transform(np.empty((0,2))) -> ValueError
    let poly = PolynomialFeatures::<f64>::new(2, false, true).unwrap();
    let x: Array2<f64> = Array2::zeros((0, 2));
    let result = poly.transform(&x);
    assert!(
        result.is_err(),
        "sklearn raises ValueError on 0-sample input (_polynomial.py:433 _validate_data, \
         ensure_min_samples=1); ferrolearn returned Ok({:?})",
        result.ok().map(|a| a.shape().to_vec())
    );
}

// ---------------------------------------------------------------------------
// GREEN guard — REQ-8 finite extremes must NOT be over-rejected. sklearn
// passes 1e308 / -0.0 (finite) through; the fixer's finite guard must only
// reject NaN/inf. Live oracle:
//   PolynomialFeatures(1, include_bias=False).fit_transform([[1e308,-0.0]])
//   -> [[1e+308, -0.0]]
// ---------------------------------------------------------------------------

/// Guard: finite extremes (1e308, -0.0) are NOT rejected. Live oracle
/// `PolynomialFeatures(1, include_bias=False).fit_transform([[1e308,-0.0]])`
/// == `[[1e308, -0.0]]`. ferrolearn `new(1,false,false)` matches and must NOT
/// over-reject once the finite guard lands.
#[test]
fn guard_finite_extremes_not_over_rejected_matches_sklearn_oracle() {
    // Live oracle:
    // PolynomialFeatures(1, include_bias=False).fit_transform([[1e308,-0.0]])
    let sklearn_expected: Array2<f64> = array![[1e308, -0.0]];
    let poly = PolynomialFeatures::<f64>::new(1, false, false).unwrap();
    let x = array![[1e308, -0.0]];
    let out = poly.transform(&x).unwrap();
    assert_eq!(out, sklearn_expected);
}

// ---------------------------------------------------------------------------
// GREEN guard — REQ-8 finite-INPUT / overflowing-PRODUCT: the polynomial-
// specific over-rejection risk. The finite guard checks INPUT finiteness via
// `x.iter()`, NOT the polynomial OUTPUT. A finite input (1e308) can produce an
// inf product (a^2 = 1e616 overflows to inf). sklearn validates only the INPUT
// (check_array force_all_finite over X, NOT over the expanded output), so it
// ACCEPTS the finite input and RETURNS inf in the a^2 column. ferrolearn must
// likewise accept (NOT error) and emit inf. Live oracle:
//   PolynomialFeatures(2, include_bias=False).fit_transform([[1e308]])
//   -> [[1e+308, inf]]   (a = 1e308 finite-input; a^2 overflows -> inf-output)
// ---------------------------------------------------------------------------

/// Guard: finite INPUT 1e308 whose degree-2 PRODUCT overflows to inf is NOT
/// rejected (the guard checks input finiteness, not output). Live oracle
/// `PolynomialFeatures(2, include_bias=False).fit_transform([[1e308]])` ==
/// `[[1e308, inf]]`. ferrolearn `new(2,false,false)` must ACCEPT the finite
/// input and return `inf` for the a^2 column (NOT `Err`).
#[test]
fn guard_finite_input_overflowing_product_not_over_rejected_matches_sklearn_oracle() {
    // Live oracle:
    // PolynomialFeatures(2, include_bias=False).fit_transform([[1e308]]) -> [[1e308, inf]]
    let poly = PolynomialFeatures::<f64>::new(2, false, false).unwrap();
    let x = array![[1e308]];
    let out = poly
        .transform(&x)
        .expect("finite input 1e308 must be ACCEPTED even though a^2 overflows to inf");
    assert_eq!(out.shape(), &[1, 2]);
    // Column 0 = a = 1e308 (finite input preserved).
    assert_eq!(out[[0, 0]], 1e308);
    // Column 1 = a^2 = 1e616 -> overflows to +inf in the OUTPUT (sklearn matches).
    assert!(
        out[[0, 1]].is_infinite() && out[[0, 1]] > 0.0,
        "sklearn returns inf for the overflowing a^2 column (input was finite, so \
         accepted); ferrolearn returned {}",
        out[[0, 1]]
    );
}

// ---------------------------------------------------------------------------
// REQ-4 / REQ-5 — stateful fit -> FittedPolynomialFeatures: n_features_in_,
// n_output_features_, powers_, and the transform-time feature-count check.
// All expected values are from the LIVE sklearn 1.5.2 oracle (R-CHAR-3), run
// from /tmp; NEVER literal-copied from ferrolearn.
//
// Oracle commands:
//   python3 -c "from sklearn.preprocessing import PolynomialFeatures as P; \
//     import numpy as np; \
//     print(P(2).fit(np.array([[2.,3.]])).powers_.tolist())"
//     -> [[0,0],[1,0],[0,1],[2,0],[1,1],[0,2]]
//   python3 -c "from sklearn.preprocessing import PolynomialFeatures as P; \
//     import numpy as np; print(P(2).fit(np.array([[2.,3.,5.]])).powers_.tolist())"
//     -> [[0,0,0],[1,0,0],[0,1,0],[0,0,1],[2,0,0],[1,1,0],[1,0,1],[0,2,0],[0,1,1],[0,0,2]]
//   python3 -c "from sklearn.preprocessing import PolynomialFeatures as P; \
//     import numpy as np; \
//     print(P(2,interaction_only=True).fit(np.array([[2.,3.]])).powers_.tolist())"
//     -> [[0,0],[1,0],[0,1],[1,1]]
//   python3 -c "from sklearn.preprocessing import PolynomialFeatures as P; \
//     import numpy as np; \
//     print(P(2,include_bias=False).fit(np.array([[2.,3.]])).powers_.tolist())"
//     -> [[1,0],[0,1],[2,0],[1,1],[0,2]]
//   python3 -c "from sklearn.preprocessing import PolynomialFeatures as P; \
//     import numpy as np; \
//     print(P(2,interaction_only=True,include_bias=False).fit(np.array([[2.,3.,5.]])).powers_.tolist())"
//     -> [[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,0,1],[0,1,1]]
//   python3 -c "from sklearn.preprocessing import PolynomialFeatures as P; \
//     import numpy as np; m=P(2).fit(np.array([[2.,3.]])); \
//     print(m.n_output_features_, m.n_features_in_)"   -> 6 2
//   python3 -c "from sklearn.preprocessing import PolynomialFeatures as P; \
//     import numpy as np; print(P(3,interaction_only=True,include_bias=False) \
//     .fit(np.array([[2.,3.,5.]])).n_output_features_)"   -> 7
//   python3 -c "from sklearn.preprocessing import PolynomialFeatures as P; \
//     import numpy as np; m=P(2).fit(np.array([[2.,3.]])); m.transform([[1.,2.,3.]])"
//     -> ValueError: X has 3 features, but PolynomialFeatures is expecting 2 features as input.
//   python3 -c "from sklearn.preprocessing import PolynomialFeatures as P; \
//     import numpy as np; m=P(2).fit(np.array([[2.,3.]])); m.transform([[1.]])"
//     -> ValueError: X has 1 features, but PolynomialFeatures is expecting 2 features as input.
//   python3 -c "from sklearn.preprocessing import PolynomialFeatures as P; \
//     import numpy as np; m=P(2).fit(np.array([[2.,3.]])); \
//     m.transform(np.array([[float('nan'),1.,2.]]))"
//     -> ValueError: Input X contains NaN.   (check_array fires BEFORE the n_features mismatch)
// ---------------------------------------------------------------------------

/// REQ-5 powers_: 2-feat, degree=2, default (full + bias). Live oracle
/// `PolynomialFeatures(2).fit([[2.,3.]]).powers_`
/// == `[[0,0],[1,0],[0,1],[2,0],[1,1],[0,2]]`, shape (6, 2). ferrolearn
/// `new(2,false,true).fit` must match every entry and the shape.
#[test]
fn fit_powers_2feat_degree2_default_matches_sklearn_oracle() {
    // Live oracle: PolynomialFeatures(2).fit([[2.,3.]]).powers_
    let sklearn_powers: Array2<usize> = array![[0, 0], [1, 0], [0, 1], [2, 0], [1, 1], [0, 2]];
    let fitted = PolynomialFeatures::<f64>::new(2, false, true)
        .unwrap()
        .fit(&array![[2.0, 3.0]], &())
        .unwrap();
    assert_eq!(fitted.powers().shape(), &[6, 2]);
    assert_eq!(fitted.powers(), &sklearn_powers);
    // Shape consistency: (n_output_features_, n_features_in_).
    assert_eq!(
        fitted.powers().shape(),
        &[fitted.n_output_features(), fitted.n_features_in()]
    );
}

/// REQ-5 powers_: 3-feat, degree=2, default. Live oracle
/// `PolynomialFeatures(2).fit([[2.,3.,5.]]).powers_` ==
/// `[[0,0,0],[1,0,0],[0,1,0],[0,0,1],[2,0,0],[1,1,0],[1,0,1],[0,2,0],[0,1,1],[0,0,2]]`,
/// shape (10, 3).
#[test]
fn fit_powers_3feat_degree2_default_matches_sklearn_oracle() {
    // Live oracle: PolynomialFeatures(2).fit([[2.,3.,5.]]).powers_
    let sklearn_powers: Array2<usize> = array![
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [2, 0, 0],
        [1, 1, 0],
        [1, 0, 1],
        [0, 2, 0],
        [0, 1, 1],
        [0, 0, 2]
    ];
    let fitted = PolynomialFeatures::<f64>::new(2, false, true)
        .unwrap()
        .fit(&array![[2.0, 3.0, 5.0]], &())
        .unwrap();
    assert_eq!(fitted.powers().shape(), &[10, 3]);
    assert_eq!(fitted.powers(), &sklearn_powers);
}

/// REQ-5 powers_: 2-feat, degree=2, interaction_only (bias present). Live oracle
/// `PolynomialFeatures(2, interaction_only=True).fit([[2.,3.]]).powers_` ==
/// `[[0,0],[1,0],[0,1],[1,1]]` (no pure-power `[2,0]`/`[0,2]` rows).
#[test]
fn fit_powers_2feat_interaction_only_matches_sklearn_oracle() {
    // Live oracle: PolynomialFeatures(2, interaction_only=True).fit([[2.,3.]]).powers_
    let sklearn_powers: Array2<usize> = array![[0, 0], [1, 0], [0, 1], [1, 1]];
    let fitted = PolynomialFeatures::<f64>::new(2, true, true)
        .unwrap()
        .fit(&array![[2.0, 3.0]], &())
        .unwrap();
    assert_eq!(fitted.powers(), &sklearn_powers);
}

/// REQ-5 powers_: 2-feat, degree=2, include_bias=False — the all-zeros bias row
/// is ABSENT. Live oracle
/// `PolynomialFeatures(2, include_bias=False).fit([[2.,3.]]).powers_` ==
/// `[[1,0],[0,1],[2,0],[1,1],[0,2]]` (no leading `[0,0]`).
#[test]
fn fit_powers_2feat_no_bias_omits_zero_row_matches_sklearn_oracle() {
    // Live oracle: PolynomialFeatures(2, include_bias=False).fit([[2.,3.]]).powers_
    let sklearn_powers: Array2<usize> = array![[1, 0], [0, 1], [2, 0], [1, 1], [0, 2]];
    let fitted = PolynomialFeatures::<f64>::new(2, false, false)
        .unwrap()
        .fit(&array![[2.0, 3.0]], &())
        .unwrap();
    assert_eq!(fitted.powers(), &sklearn_powers);
}

/// REQ-5 powers_: 3-feat, degree=2, interaction_only + no bias. Live oracle
/// `PolynomialFeatures(2, interaction_only=True, include_bias=False).fit([[2.,3.,5.]]).powers_`
/// == `[[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,0,1],[0,1,1]]`.
#[test]
fn fit_powers_3feat_interaction_only_no_bias_matches_sklearn_oracle() {
    // Live oracle:
    // PolynomialFeatures(2,interaction_only=True,include_bias=False).fit([[2.,3.,5.]]).powers_
    let sklearn_powers: Array2<usize> = array![
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1]
    ];
    let fitted = PolynomialFeatures::<f64>::new(2, true, false)
        .unwrap()
        .fit(&array![[2.0, 3.0, 5.0]], &())
        .unwrap();
    assert_eq!(fitted.powers(), &sklearn_powers);
}

/// REQ-4 n_features_in_ == ncols seen during fit (here 3). Live oracle
/// `PolynomialFeatures(2).fit([[2.,3.,5.]]).n_features_in_` == `3`.
#[test]
fn fit_n_features_in_matches_ncols() {
    let fitted = PolynomialFeatures::<f64>::new(2, false, true)
        .unwrap()
        .fit(&array![[2.0, 3.0, 5.0]], &())
        .unwrap();
    assert_eq!(fitted.n_features_in(), 3);
}

/// REQ-4 n_output_features_ == sklearn `.n_output_features_` AND == the transform
/// output column count, across several (degree, interaction_only, include_bias)
/// combos. Live oracle counts:
///   P(2).fit([[2,3]]).n_output_features_                              -> 6
///   P(2).fit([[2,3,5]]).n_output_features_                            -> 10
///   P(2,interaction_only=True).fit([[2,3,5]]).n_output_features_      -> 7
///   P(3,interaction_only=True,include_bias=False).fit([[2,3,5]])...   -> 7
///   P(2,include_bias=False).fit([[2,3]]).n_output_features_           -> 5
#[test]
fn fit_n_output_features_matches_sklearn_oracle_and_transform_ncols() {
    // (degree, interaction_only, include_bias, n_features, sklearn n_output_features_)
    let cases: &[(usize, bool, bool, usize, usize)] = &[
        (2, false, true, 2, 6),
        (2, false, true, 3, 10),
        (2, true, true, 3, 7),
        (3, true, false, 3, 7),
        (2, false, false, 2, 5),
    ];
    for &(deg, inter, bias, nfeat, expected_n_out) in cases {
        let poly = PolynomialFeatures::<f64>::new(deg, inter, bias).unwrap();
        let x: Array2<f64> = Array2::from_elem((1, nfeat), 2.0);
        let fitted = poly.fit(&x, &()).unwrap();
        assert_eq!(
            fitted.n_output_features(),
            expected_n_out,
            "n_output_features_ mismatch for deg={deg} inter={inter} bias={bias} nfeat={nfeat}"
        );
        // n_output_features_ must equal the transform output column count.
        let out = fitted.transform(&x).unwrap();
        assert_eq!(out.shape()[1], expected_n_out);
        // And the stateless path agrees on the column count.
        assert_eq!(poly.transform(&x).unwrap().shape()[1], expected_n_out);
    }
}

/// REQ-4: `fit(X).transform(X)` == the stateless `PolynomialFeatures::transform(X)`
/// == sklearn `.fit_transform(X)` (full matrix, bit-exact, REQ-1 column order).
/// Live oracle (REQ-1 reuse): `PolynomialFeatures(2).fit_transform([[2.,3.,5.]])`
/// == `[[1,2,3,5,4,6,10,9,15,25]]`.
#[test]
fn fitted_transform_matches_stateless_and_sklearn_oracle() {
    // Live oracle: PolynomialFeatures(2).fit_transform([[2.,3.,5.]])
    let sklearn_expected: Array2<f64> =
        array![[1.0, 2.0, 3.0, 5.0, 4.0, 6.0, 10.0, 9.0, 15.0, 25.0]];
    let poly = PolynomialFeatures::<f64>::new(2, false, true).unwrap();
    let x = array![[2.0, 3.0, 5.0]];
    let stateless = poly.transform(&x).unwrap();
    let fitted = poly.fit(&x, &()).unwrap();
    let stateful = fitted.transform(&x).unwrap();
    // fitted == stateless == sklearn, byte-for-byte.
    assert_eq!(stateful, stateless);
    assert_eq!(stateful, sklearn_expected);
}

/// REQ-4 transform-time feature-count check: MORE features than fitted →
/// ShapeMismatch (Err, no panic). Live oracle
/// `PolynomialFeatures(2).fit([[2.,3.]]).transform([[1.,2.,3.]])` raises
/// `ValueError: X has 3 features, but PolynomialFeatures is expecting 2 features
/// as input.`
#[test]
fn fitted_transform_more_features_shape_mismatch_like_sklearn() {
    // Live oracle: fit([[2.,3.]]) then transform([[1.,2.,3.]]) -> ValueError (3 vs 2)
    let fitted = PolynomialFeatures::<f64>::new(2, false, true)
        .unwrap()
        .fit(&array![[2.0, 3.0]], &())
        .unwrap();
    let result = fitted.transform(&array![[1.0, 2.0, 3.0]]);
    assert!(
        matches!(result, Err(FerroError::ShapeMismatch { .. })),
        "sklearn raises ValueError (X has 3 features, expecting 2); ferrolearn must \
         return ShapeMismatch, got {result:?}"
    );
}

/// REQ-4 transform-time feature-count check: FEWER features than fitted →
/// ShapeMismatch (Err, no panic). Live oracle
/// `PolynomialFeatures(2).fit([[2.,3.]]).transform([[1.]])` raises
/// `ValueError: X has 1 features, but PolynomialFeatures is expecting 2 features
/// as input.`
#[test]
fn fitted_transform_fewer_features_shape_mismatch_like_sklearn() {
    // Live oracle: fit([[2.,3.]]) then transform([[1.]]) -> ValueError (1 vs 2)
    let fitted = PolynomialFeatures::<f64>::new(2, false, true)
        .unwrap()
        .fit(&array![[2.0, 3.0]], &())
        .unwrap();
    let result = fitted.transform(&array![[1.0]]);
    assert!(
        matches!(result, Err(FerroError::ShapeMismatch { .. })),
        "sklearn raises ValueError (X has 1 features, expecting 2); ferrolearn must \
         return ShapeMismatch, got {result:?}"
    );
}

/// REQ-4 #2207 order: a NaN-containing input that ALSO has the wrong feature
/// count must raise the check_array (finite) error, NOT ShapeMismatch — sklearn
/// runs `check_array` (force_all_finite) BEFORE the n_features consistency check
/// (`_polynomial.py:433-435`). Live oracle
/// `PolynomialFeatures(2).fit([[2.,3.]]).transform([[nan,1.,2.]])` raises
/// `ValueError: Input X contains NaN.` (the finite error), not the 3-vs-2
/// feature-count error.
#[test]
fn fitted_transform_nan_validation_before_n_features_like_sklearn() {
    // Live oracle: fit([[2.,3.]]) then transform([[nan,1.,2.]]) -> "Input X contains NaN."
    // (check_array finite check fires BEFORE the 3-vs-2 n_features mismatch.)
    let fitted = PolynomialFeatures::<f64>::new(2, false, true)
        .unwrap()
        .fit(&array![[2.0, 3.0]], &())
        .unwrap();
    let result = fitted.transform(&array![[f64::NAN, 1.0, 2.0]]);
    // The NaN/finite check (REQ-8 InvalidParameter) must win over the
    // feature-count check (ShapeMismatch), matching sklearn's validation order.
    assert!(
        matches!(result, Err(FerroError::InvalidParameter { .. })),
        "sklearn validates finiteness (Input X contains NaN) BEFORE the n_features \
         check (#2207); ferrolearn must return InvalidParameter, not ShapeMismatch, \
         got {result:?}"
    );
}

/// REQ-4: the fitted path enforces REQ-8 input validation too (shared guard) —
/// a NaN input with the CORRECT feature count still errors (InvalidParameter),
/// matching the stateless path. Live oracle
/// `PolynomialFeatures(2).fit([[2.,3.]]).transform([[nan,1.]])` raises
/// `ValueError: Input X contains NaN.`
#[test]
fn fitted_transform_rejects_nan_like_sklearn() {
    let fitted = PolynomialFeatures::<f64>::new(2, false, true)
        .unwrap()
        .fit(&array![[2.0, 3.0]], &())
        .unwrap();
    let result = fitted.transform(&array![[f64::NAN, 1.0]]);
    assert!(
        matches!(result, Err(FerroError::InvalidParameter { .. })),
        "sklearn raises ValueError on NaN (force_all_finite); ferrolearn must reject \
         with InvalidParameter, got {result:?}"
    );
}

/// REQ-4: `fit` itself runs the REQ-8 guard — a NaN in the fit data is rejected
/// (sklearn `fit` -> `_validate_data` default force_all_finite=True). Live oracle
/// `PolynomialFeatures(2).fit(np.array([[float('nan'),1.]]))` raises
/// `ValueError: Input X contains NaN.`
#[test]
fn fit_rejects_nan_like_sklearn() {
    let result = PolynomialFeatures::<f64>::new(2, false, true)
        .unwrap()
        .fit(&array![[f64::NAN, 1.0]], &());
    assert!(
        matches!(result, Err(FerroError::InvalidParameter { .. })),
        "sklearn fit rejects NaN (force_all_finite); ferrolearn must return \
         InvalidParameter, got {result:?}"
    );
}

/// REQ-4: `fit` rejects a zero-row input (sklearn `_validate_data`
/// ensure_min_samples=1). Live oracle
/// `PolynomialFeatures(2).fit(np.empty((0,2)))` raises
/// `ValueError: Found array with 0 sample(s) ... a minimum of 1 is required`.
#[test]
fn fit_rejects_zero_rows_like_sklearn() {
    let x: Array2<f64> = Array2::zeros((0, 2));
    let result = PolynomialFeatures::<f64>::new(2, false, true)
        .unwrap()
        .fit(&x, &());
    assert!(
        matches!(result, Err(FerroError::InsufficientSamples { .. })),
        "sklearn fit rejects 0-sample input (ensure_min_samples=1); ferrolearn must \
         return InsufficientSamples, got {result:?}"
    );
}

/// REQ-5 f32 generic: powers_ is dtype-independent (always usize exponents) and
/// matches the oracle for an f32 estimator. Live oracle (same as the f64 2-feat
/// default) `PolynomialFeatures(2).fit([[2.,3.]]).powers_` ==
/// `[[0,0],[1,0],[0,1],[2,0],[1,1],[0,2]]`.
#[test]
fn fit_powers_f32_matches_sklearn_oracle() {
    let sklearn_powers: Array2<usize> = array![[0, 0], [1, 0], [0, 1], [2, 0], [1, 1], [0, 2]];
    let x: Array2<f32> = array![[2.0f32, 3.0]];
    let fitted = PolynomialFeatures::<f32>::new(2, false, true)
        .unwrap()
        .fit(&x, &())
        .unwrap();
    assert_eq!(fitted.powers(), &sklearn_powers);
    assert_eq!(fitted.n_output_features(), 6);
    assert_eq!(fitted.n_features_in(), 2);
}
