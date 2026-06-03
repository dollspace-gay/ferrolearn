//! Divergence audit: `FunctionTransformer` (forward element-wise transform,
//! REQ-1) vs scikit-learn 1.5.2 `sklearn/preprocessing/_function_transformer.py`
//! `class FunctionTransformer` `_transform` (`:375-379`:
//! `return func(X, **(kw_args if kw_args else {}))`).
//!
//! ferrolearn ships a thin scalar wrapper applying `Fn(F) -> F` element-wise
//! via `Array2::mapv` (`function_transformer.rs:77-80`). For element-wise
//! ufuncs this is numerically identical to sklearn's whole-array `func(X)`.
//!
//! EVERY expected value below is taken from a LIVE sklearn 1.5.2 oracle call
//! (run from /tmp), pasted in the `// oracle:` comment above the constant, NOT
//! copied from the ferrolearn side (R-CHAR-3). The scalar-vs-array signature
//! mismatch (REQ-2/REQ-3) is a structural NOT-STARTED missing feature, not a
//! runtime value divergence, and is intentionally NOT pinned here (there is no
//! whole-array `func` callable through the `Fn(F) -> F` API).

use ferrolearn_core::traits::Transform;
use ferrolearn_preprocess::FunctionTransformer;
use ndarray::{Array2, array};

/// Green guard (REQ-1 / AC-1): element-wise `np.log1p` ufunc.
///
/// Mirrors `sklearn/preprocessing/_function_transformer.py:375-379`
/// (`func(X)` with `func = np.log1p`).
///
/// Oracle (live sklearn 1.5.2, run from /tmp):
/// ```text
/// python3 -c "import numpy as np; from sklearn.preprocessing import \
///   FunctionTransformer; import json; \
///   print(json.dumps(FunctionTransformer(np.log1p).transform(\
///   np.array([[0.,1.],[2.,3.]])).tolist()))"
/// -> [[0.0, 0.6931471805599453], [1.0986122886681096, 1.3862943611198906]]
/// ```
/// np.log1p(x) = ln(1+x); Rust analog `f64::ln_1p`.
#[test]
#[allow(
    clippy::approx_constant,
    reason = "exact live-sklearn-1.5.2 oracle value ln(2)=log1p(1); R-CHAR-3, not the std constant"
)]
fn guard_log1p_matches_sklearn_oracle() {
    // oracle: FunctionTransformer(np.log1p).transform([[0,1],[2,3]])
    let expected: [[f64; 2]; 2] = [
        [0.0, 0.693_147_180_559_945_3],
        [1.098_612_288_668_109_6, 1.386_294_361_119_890_6],
    ];

    let ft = FunctionTransformer::<f64>::new(|v: f64| v.ln_1p());
    let x = array![[0.0, 1.0], [2.0, 3.0]];
    let out = ft.transform(&x).unwrap();

    assert_eq!(out.shape(), x.shape(), "shape must be preserved");
    for ((r, c), &want) in expected
        .iter()
        .enumerate()
        .flat_map(|(i, row)| row.iter().enumerate().map(move |(j, v)| ((i, j), v)))
    {
        // Bit-exact: numpy and Rust both delegate log1p to libm.
        assert_eq!(
            out[[r, c]].to_bits(),
            want.to_bits(),
            "log1p mismatch at [{r},{c}]: ferro={} sklearn={}",
            out[[r, c]],
            want
        );
    }
}

/// Green guard (REQ-1): element-wise `np.expm1` ufunc cross-check.
///
/// Oracle (live sklearn 1.5.2, run from /tmp):
/// ```text
/// python3 -c "import numpy as np; from sklearn.preprocessing import \
///   FunctionTransformer; import json; \
///   print(json.dumps(FunctionTransformer(np.expm1).transform(\
///   np.array([[0.,1.],[2.,3.]])).tolist()))"
/// -> [[0.0, 1.718281828459045], [6.38905609893065, 19.085536923187668]]
/// ```
/// np.expm1(x) = exp(x) - 1; Rust analog `f64::exp_m1`.
#[test]
fn guard_expm1_matches_sklearn_oracle() {
    // oracle: FunctionTransformer(np.expm1).transform([[0,1],[2,3]])
    let expected: [[f64; 2]; 2] = [
        [0.0, 1.718_281_828_459_045],
        [6.389_056_098_930_65, 19.085_536_923_187_668],
    ];

    let ft = FunctionTransformer::<f64>::new(|v: f64| v.exp_m1());
    let x = array![[0.0, 1.0], [2.0, 3.0]];
    let out = ft.transform(&x).unwrap();

    for ((r, c), &want) in expected
        .iter()
        .enumerate()
        .flat_map(|(i, row)| row.iter().enumerate().map(move |(j, v)| ((i, j), v)))
    {
        assert_eq!(
            out[[r, c]].to_bits(),
            want.to_bits(),
            "expm1 mismatch at [{r},{c}]: ferro={} sklearn={}",
            out[[r, c]],
            want
        );
    }
}

/// Green guard (REQ-1): element-wise `np.sqrt` ufunc cross-check.
///
/// Oracle (live sklearn 1.5.2, run from /tmp):
/// ```text
/// python3 -c "import numpy as np; from sklearn.preprocessing import \
///   FunctionTransformer; import json; \
///   print(json.dumps(FunctionTransformer(np.sqrt).transform(\
///   np.array([[1.,4.],[9.,16.]])).tolist()))"
/// -> [[1.0, 2.0], [3.0, 4.0]]
/// ```
#[test]
fn guard_sqrt_matches_sklearn_oracle() {
    // oracle: FunctionTransformer(np.sqrt).transform([[1,4],[9,16]])
    let expected: [[f64; 2]; 2] = [[1.0, 2.0], [3.0, 4.0]];

    let ft = FunctionTransformer::<f64>::new(|v: f64| v.sqrt());
    let x = array![[1.0, 4.0], [9.0, 16.0]];
    let out = ft.transform(&x).unwrap();

    for ((r, c), &want) in expected
        .iter()
        .enumerate()
        .flat_map(|(i, row)| row.iter().enumerate().map(move |(j, v)| ((i, j), v)))
    {
        assert_eq!(
            out[[r, c]].to_bits(),
            want.to_bits(),
            "sqrt mismatch at [{r},{c}]: ferro={} sklearn={}",
            out[[r, c]],
            want
        );
    }
}

/// Green guard (REQ-1): NaN / -inf propagation through `np.log` on a 0.0 and a
/// negative input matches sklearn (sklearn forwards the whole array to np.log;
/// `_transform`, `:375-379`).
///
/// Oracle (live sklearn 1.5.2, run from /tmp, np.seterr(all='ignore')):
/// ```text
/// python3 -c "import numpy as np; from sklearn.preprocessing import \
///   FunctionTransformer; import json; np.seterr(all='ignore'); \
///   print(json.dumps(FunctionTransformer(np.log).transform(\
///   np.array([[0.,1.],[-1.,2.]])).tolist()))"
/// -> [[-Infinity, 0.0], [NaN, 0.6931471805599453]]
/// ```
#[test]
#[allow(
    clippy::approx_constant,
    reason = "exact live-sklearn-1.5.2 oracle value ln(2)=log(2); R-CHAR-3, not the std constant"
)]
fn guard_log_nan_inf_propagation_matches_sklearn_oracle() {
    let ft = FunctionTransformer::<f64>::new(|v: f64| v.ln());
    let x = array![[0.0, 1.0], [-1.0, 2.0]];
    let out = ft.transform(&x).unwrap();

    // oracle: np.log(0.0) -> -inf
    assert!(
        out[[0, 0]].is_infinite() && out[[0, 0]].is_sign_negative(),
        "expected -inf at [0,0], got {}",
        out[[0, 0]]
    );
    // oracle: np.log(1.0) -> 0.0
    assert_eq!(out[[0, 1]].to_bits(), 0.0_f64.to_bits());
    // oracle: np.log(-1.0) -> NaN
    assert!(
        out[[1, 0]].is_nan(),
        "expected NaN at [1,0], got {}",
        out[[1, 0]]
    );
    // oracle: np.log(2.0) -> 0.6931471805599453
    assert_eq!(
        out[[1, 1]].to_bits(),
        0.693_147_180_559_945_3_f64.to_bits(),
        "ln(2) mismatch: {}",
        out[[1, 1]]
    );
}

/// Green guard (REQ-1): empty-matrix `(0, n)` shape preservation matches
/// sklearn (whole-array func on an empty array returns the same shape).
///
/// Oracle (live sklearn 1.5.2, run from /tmp):
/// ```text
/// python3 -c "import numpy as np; from sklearn.preprocessing import \
///   FunctionTransformer; \
///   print(FunctionTransformer(np.log1p).transform(np.zeros((0,3))).shape)"
/// -> (0, 3)
/// ```
#[test]
fn guard_empty_matrix_shape_matches_sklearn_oracle() {
    // oracle: transform(zeros((0,3))).shape == (0, 3)
    let expected_shape = [0_usize, 3];

    let ft = FunctionTransformer::<f64>::new(|v: f64| v.ln_1p());
    let x: Array2<f64> = Array2::zeros((0, 3));
    let out = ft.transform(&x).unwrap();

    assert_eq!(out.shape(), &expected_shape);
}
