//! ACToR critic: oracle-grounded audit of `ferrolearn-preprocess`'s
//! `Binarizer` against scikit-learn 1.5.2 `sklearn/preprocessing/_data.py`
//! `class Binarizer` (`:2177`) / `binarize` (`:2120`).
//!
//! All expected values are derived from the LIVE sklearn 1.5.2 oracle, run from
//! `/tmp` (R-CHAR-3): NEVER literal-copied from the ferrolearn side.
//!
//! Oracle commands (sklearn 1.5.2 == `python3 -c "import sklearn; sklearn.__version__"`):
//! ```text
//! python3 -c "from sklearn.preprocessing import Binarizer; \
//!   print(Binarizer(threshold=0.5).transform([[0.4,0.6,0.5],[0.6,0.1,0.2]]).tolist())"
//!   -> [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]
//! python3 -c "from sklearn.preprocessing import Binarizer; \
//!   print(Binarizer().transform([[-1.0,0.0,0.5,1.0]]).tolist())"
//!   -> [[0.0, 0.0, 1.0, 1.0]]
//! python3 -c "from sklearn.preprocessing import Binarizer; \
//!   print(Binarizer(threshold=-1.0).transform([[-2.0,-1.0,-0.5,0.0]]).tolist())"
//!   -> [[0.0, 0.0, 1.0, 1.0]]
//! ```

use ferrolearn_core::error::FerroError;
use ferrolearn_core::traits::{Fit, Transform};
use ferrolearn_preprocess::Binarizer;
use ferrolearn_preprocess::binarizer::{FittedBinarizer, binarize};
use ndarray::{Array2, array};

// ---------------------------------------------------------------------------
// GREEN guards — REQ-1 dense strict-greater transform vs the LIVE sklearn
// oracle (R-CHAR-3: expected values are the oracle outputs quoted above).
// ---------------------------------------------------------------------------

/// Guard: `Binarizer::new(0.5).transform(X)` matches the live sklearn oracle
/// `Binarizer(threshold=0.5).transform([[0.4,0.6,0.5],[0.6,0.1,0.2]])`
/// == `[[0.0,1.0,0.0],[1.0,0.0,0.0]]`. The boundary `0.5` is NOT > `0.5`
/// (strict greater-than, sklearn `_data.py:2170` `cond = X > threshold`).
#[test]
fn guard_binarizer_threshold_0_5_matches_sklearn_oracle() {
    // Live oracle: Binarizer(threshold=0.5).transform([[0.4,0.6,0.5],[0.6,0.1,0.2]])
    let sklearn_expected: Array2<f64> = array![[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]];
    let b = Binarizer::<f64>::new(0.5);
    let x = array![[0.4, 0.6, 0.5], [0.6, 0.1, 0.2]];
    let out = b.transform(&x).unwrap();
    assert_eq!(out, sklearn_expected);
    assert_eq!(out.shape(), x.shape());
}

/// Guard: default threshold 0.0 — only strictly-positive values map to 1
/// (`0.0` is NOT > `0.0`). Live oracle:
/// `Binarizer().transform([[-1.0,0.0,0.5,1.0]])` == `[[0.0,0.0,1.0,1.0]]`.
#[test]
fn guard_binarizer_default_threshold_matches_sklearn_oracle() {
    // Live oracle: Binarizer().transform([[-1.0,0.0,0.5,1.0]])
    let sklearn_expected: Array2<f64> = array![[0.0, 0.0, 1.0, 1.0]];
    let b = Binarizer::<f64>::default();
    let x = array![[-1.0, 0.0, 0.5, 1.0]];
    let out = b.transform(&x).unwrap();
    assert_eq!(out, sklearn_expected);
}

/// Guard: negative threshold -1.0 (dense, no sparse restriction). Live oracle:
/// `Binarizer(threshold=-1.0).transform([[-2.0,-1.0,-0.5,0.0]])`
/// == `[[0.0,0.0,1.0,1.0]]` (`-1.0` is NOT > `-1.0`).
#[test]
fn guard_binarizer_negative_threshold_matches_sklearn_oracle() {
    // Live oracle: Binarizer(threshold=-1.0).transform([[-2.0,-1.0,-0.5,0.0]])
    let sklearn_expected: Array2<f64> = array![[0.0, 0.0, 1.0, 1.0]];
    let b = Binarizer::<f64>::new(-1.0);
    let x = array![[-2.0, -1.0, -0.5, 0.0]];
    let out = b.transform(&x).unwrap();
    assert_eq!(out, sklearn_expected);
}

/// Guard: f32 path, default threshold 0.0. Live oracle:
/// `Binarizer(threshold=0.0).transform(np.array([[1.0,-1.0,0.0]], dtype=np.float32))`
/// == `[[1.0, 0.0, 0.0]]`.
#[test]
fn guard_binarizer_f32_matches_sklearn_oracle() {
    // Live oracle (float32): Binarizer(threshold=0.0).transform([[1.0,-1.0,0.0]])
    let sklearn_expected: Array2<f32> = array![[1.0f32, 0.0, 0.0]];
    let b = Binarizer::<f32>::new(0.0f32);
    let x: Array2<f32> = array![[1.0f32, -1.0, 0.0]];
    let out = b.transform(&x).unwrap();
    assert_eq!(out, sklearn_expected);
}

// ---------------------------------------------------------------------------
// DIVERGENCE — non-finite input.
//
// sklearn `Binarizer.transform` (`_data.py:2301`) calls `_validate_data(...)`
// (no `force_all_finite="allow-nan"` override at :2300-2307, so it defaults to
// rejecting non-finite values) BEFORE the threshold comparison at `:2170`.
// The LIVE oracle therefore RAISES `ValueError` for NaN, +inf, and -inf:
//
//   python3 -c "import numpy as np; from sklearn.preprocessing import Binarizer; \
//     Binarizer(threshold=0.0).transform(np.array([[float('nan'),1.0]]))"
//   -> ValueError: Input X contains NaN. ...
//   ... transform(np.array([[float('inf'),1.0]]))
//   -> ValueError: Input X contains infinity or a value too large ...
//   ... transform(np.array([[float('-inf'),1.0]]))
//   -> ValueError: Input X contains infinity or a value too large ...
//
// ferrolearn's `transform` does NO validation; it applies
// `mapv(|v| v > threshold)` directly and ALWAYS returns `Ok`. For NaN,
// `NaN > 0.0` is `false` -> `0.0`; for `+inf`, `inf > 0.0` -> `1.0`; for
// `-inf`, `-inf > 0.0` is `false` -> `0.0`. ferrolearn thus silently produces
// a finite output where sklearn raises an error — an observable divergence in
// the `transform` contract for inputs both implementations accept a call on.
// ---------------------------------------------------------------------------

/// Divergence: `ferrolearn::Binarizer::transform` diverges from
/// `sklearn/preprocessing/_data.py:2301` (`self._validate_data(...)`, which
/// defaults to rejecting non-finite values) for input containing NaN.
/// sklearn raises `ValueError: Input X contains NaN.`; ferrolearn returns
/// `Ok([[0.0, ...]])` (because Rust `NaN > 0.0` is `false`).
/// Tracking: #1123
#[test]
fn divergence_binarizer_nan_should_error_like_sklearn() {
    // Live oracle: Binarizer(threshold=0.0).transform([[nan, 1.0]]) -> ValueError
    let b = Binarizer::<f64>::new(0.0);
    let x = array![[f64::NAN, 1.0]];
    let result = b.transform(&x);
    assert!(
        result.is_err(),
        "sklearn raises ValueError on NaN input (_data.py:2301 _validate_data); \
         ferrolearn returned Ok({:?})",
        result.ok()
    );
}

/// Divergence: `ferrolearn::Binarizer::transform` diverges from
/// `sklearn/preprocessing/_data.py:2301` for input containing +inf.
/// sklearn raises `ValueError: Input X contains infinity ...`; ferrolearn
/// returns `Ok([[1.0, ...]])` (Rust `inf > 0.0` is `true`).
/// Tracking: #1123
#[test]
fn divergence_binarizer_pos_inf_should_error_like_sklearn() {
    // Live oracle: Binarizer(threshold=0.0).transform([[inf, 1.0]]) -> ValueError
    let b = Binarizer::<f64>::new(0.0);
    let x = array![[f64::INFINITY, 1.0]];
    let result = b.transform(&x);
    assert!(
        result.is_err(),
        "sklearn raises ValueError on +inf input (_data.py:2301 _validate_data); \
         ferrolearn returned Ok({:?})",
        result.ok()
    );
}

/// Divergence: `ferrolearn::Binarizer::transform` diverges from
/// `sklearn/preprocessing/_data.py:2301` for input containing -inf.
/// sklearn raises `ValueError: Input X contains infinity ...`; ferrolearn
/// returns `Ok([[0.0, ...]])` (Rust `-inf > 0.0` is `false`).
/// Tracking: #1123
#[test]
fn divergence_binarizer_neg_inf_should_error_like_sklearn() {
    // Live oracle: Binarizer(threshold=0.0).transform([[-inf, 1.0]]) -> ValueError
    let b = Binarizer::<f64>::new(0.0);
    let x = array![[f64::NEG_INFINITY, 1.0]];
    let result = b.transform(&x);
    assert!(
        result.is_err(),
        "sklearn raises ValueError on -inf input (_data.py:2301 _validate_data); \
         ferrolearn returned Ok({:?})",
        result.ok()
    );
}

// ---------------------------------------------------------------------------
// GREEN guards — REQ-1 finite extremes MUST still be binarized, NOT rejected.
// The #1123 guard uses `!v.is_finite()`, which is true ONLY for NaN/±inf, so
// all finite values (incl. 1e308, -0.0, subnormals) must pass through.
// Expected values are the LIVE sklearn 1.5.2 oracle (run from /tmp):
//   Binarizer().transform([[1e308]])        -> [[1.0]]
//   Binarizer().transform([[-0.0, 0.0]])    -> [[0.0, 0.0]]
//   Binarizer().transform([[5e-324]])       -> [[1.0]]  (subnormal > 0.0)
// ---------------------------------------------------------------------------

/// Guard: large finite value 1e308 is binarized (1.0 > 0.0), NOT rejected.
/// Confirms the #1123 `!is_finite()` guard does not over-reject finite
/// extremes. Live oracle: `Binarizer().transform([[1e308]])` -> `[[1.0]]`.
#[test]
fn guard_binarizer_large_finite_not_rejected() {
    // Live oracle: Binarizer().transform([[1e308]]) -> [[1.0]] (no error)
    let sklearn_expected: Array2<f64> = array![[1.0]];
    let b = Binarizer::<f64>::default();
    let x = array![[1e308_f64]];
    let out = b
        .transform(&x)
        .expect("1e308 is finite; sklearn accepts it, ferrolearn must not reject");
    assert_eq!(out, sklearn_expected);
}

/// Guard: negative-zero `-0.0` is finite and NOT > 0.0 (IEEE `-0.0 == 0.0`),
/// so it maps to `0.0`, not rejected. Live oracle:
/// `Binarizer().transform([[-0.0, 0.0]])` -> `[[0.0, 0.0]]`.
#[test]
fn guard_binarizer_negative_zero_not_rejected() {
    // Live oracle: Binarizer().transform([[-0.0, 0.0]]) -> [[0.0, 0.0]]
    let sklearn_expected: Array2<f64> = array![[0.0, 0.0]];
    let b = Binarizer::<f64>::default();
    let x = array![[-0.0_f64, 0.0_f64]];
    let out = b.transform(&x).expect("-0.0 is finite; sklearn accepts it");
    assert_eq!(out, sklearn_expected);
}

/// Guard: smallest positive subnormal `5e-324` is finite and `> 0.0`, so it
/// maps to `1.0`, not rejected. Live oracle:
/// `Binarizer().transform([[5e-324]])` -> `[[1.0]]`.
#[test]
fn guard_binarizer_subnormal_not_rejected() {
    // Live oracle: Binarizer().transform([[5e-324]]) -> [[1.0]]
    let sklearn_expected: Array2<f64> = array![[1.0]];
    let b = Binarizer::<f64>::default();
    let x = array![[5e-324_f64]];
    let out = b
        .transform(&x)
        .expect("subnormal 5e-324 is finite; sklearn accepts it");
    assert_eq!(out, sklearn_expected);
}

// ---------------------------------------------------------------------------
// DIVERGENCE — empty input (0 rows).
//
// The parent-audit assumption that "sklearn accepts empty (0,3) -> Ok(empty)"
// is FALSE. sklearn `Binarizer.transform` (`_data.py:2301` `_validate_data`,
// reaching `utils/validation.py:1087` `check_array` min-samples check) REJECTS
// a (0,3) array. LIVE oracle (sklearn 1.5.2, from /tmp):
//
//   python3 -c "import numpy as np; from sklearn.preprocessing import Binarizer; \
//     Binarizer().transform(np.zeros((0,3)))"
//   -> ValueError: Found array with 0 sample(s) (shape=(0, 3)) while a minimum
//      of 1 is required by Binarizer.
//
// ferrolearn's `transform` runs `x.iter().any(|v| !v.is_finite())`, which is
// `false` for an empty array, so it skips the guard and returns `Ok` of an
// empty (0,3) array — diverging from sklearn, which raises ValueError. Sibling
// preprocess transformers (`robust_scaler.rs:125`, `one_hot_encoder.rs:106`)
// DO reject zero-row input via `FerroError::InsufficientSamples`, so this is
// also a cross-transformer consistency gap.
// ---------------------------------------------------------------------------

/// Divergence: `ferrolearn::Binarizer::transform` diverges from
/// `sklearn/preprocessing/_data.py:2301` (`_validate_data` ->
/// `utils/validation.py:1087` min-samples check) for empty (0,3) input.
/// sklearn raises `ValueError: Found array with 0 sample(s) ... minimum of 1
/// is required by Binarizer.`; ferrolearn returns `Ok` of an empty array
/// (the `!is_finite()` guard's `.any()` is `false` on empty, and there is no
/// min-samples check). Sibling `RobustScaler`/`OneHotEncoder` reject zero rows
/// via `InsufficientSamples`.
/// Tracking: #1124
#[test]
fn divergence_binarizer_empty_input_should_error_like_sklearn() {
    // Live oracle: Binarizer().transform(np.zeros((0,3))) -> ValueError (min 1 sample)
    let b = Binarizer::<f64>::default();
    let x: Array2<f64> = Array2::zeros((0, 3));
    let result = b.transform(&x);
    assert!(
        result.is_err(),
        "sklearn raises ValueError on empty (0,3) input \
         (_data.py:2301 _validate_data, validation.py:1087 min-samples); \
         ferrolearn returned Ok({:?})",
        result.ok()
    );
}

// ---------------------------------------------------------------------------
// DIVERGENCE — zero features (n_rows > 0, 0 columns).
//
// sklearn `Binarizer.transform` (`_data.py:2301` `_validate_data`, defaults to
// `check_array(... ensure_min_features=1)`) reaches the min-FEATURES check at
// `utils/validation.py:1093-1099`:
//   `if ensure_min_features > 0 and array.ndim == 2:`
//   `    n_features = array.shape[1]`
//   `    if n_features < ensure_min_features:`
//   `        raise ValueError("Found array with %d feature(s) (shape=%s) ...`
// This is SEPARATE from the min-samples check (`:1084-1090`), which only fires
// for 0 rows. A `(3, 0)` array has >0 samples but 0 features, so it passes the
// samples check and is REJECTED by the features check. LIVE oracle
// (sklearn 1.5.2, from /tmp):
//
//   python3 -c "import numpy as np; from sklearn.preprocessing import Binarizer; \
//     Binarizer().transform(np.zeros((3,0)))"
//   -> ValueError: Found array with 0 feature(s) (shape=(3, 0)) while a minimum
//      of 1 is required by Binarizer.
//
// ferrolearn's `transform` only guards `x.nrows() == 0` (the #1124 SAMPLES
// guard); it has NO min-FEATURES check. For `(3, 0)`: `nrows() == 3` (guard
// skipped), `iter().any(|v| !v.is_finite())` is `false` on an empty array
// (guard skipped), and `mapv` returns `Ok` of an empty `(3, 0)` array.
// ferrolearn thus accepts a zero-feature input where sklearn raises — a real
// validation-contract divergence not covered by #1123 (non-finite) or #1124
// (zero rows).
// ---------------------------------------------------------------------------

/// Divergence: `ferrolearn::Binarizer::transform` diverges from
/// `sklearn/preprocessing/_data.py:2301` (`_validate_data` ->
/// `utils/validation.py:1093` min-FEATURES check, `ensure_min_features=1`)
/// for a `(3, 0)` zero-feature input. sklearn raises
/// `ValueError: Found array with 0 feature(s) (shape=(3, 0)) while a minimum
/// of 1 is required by Binarizer.`; ferrolearn returns `Ok` of an empty
/// `(3, 0)` array (it guards 0 rows via #1124 but has no min-features check,
/// and the `!is_finite()` `.any()` is `false` on an empty array).
/// Tracking: #1125
#[test]
fn divergence_binarizer_zero_features_should_error_like_sklearn() {
    // Live oracle: Binarizer().transform(np.zeros((3,0))) -> ValueError (min 1 feature)
    let b = Binarizer::<f64>::default();
    let x: Array2<f64> = Array2::zeros((3, 0));
    let result = b.transform(&x);
    assert!(
        result.is_err(),
        "sklearn raises ValueError on zero-feature (3,0) input \
         (_data.py:2301 _validate_data, validation.py:1093 min-features); \
         ferrolearn returned Ok({:?})",
        result.ok()
    );
}

// ---------------------------------------------------------------------------
// GREEN guard — single element (1,1) MUST still be accepted (1 sample,
// 1 feature both meet the minimum-of-1 thresholds). Confirms a future
// min-features fix does not over-reject. Live oracle (sklearn 1.5.2):
//   Binarizer().transform(np.zeros((1,1))) -> [[0.0]] (no error)
// ---------------------------------------------------------------------------

/// Guard: `(1,1)` single-element input is accepted (1 sample, 1 feature both
/// satisfy `ensure_min_samples=1` / `ensure_min_features=1`). Live oracle:
/// `Binarizer().transform(np.zeros((1,1)))` -> `[[0.0]]`. Guards against an
/// over-eager min-features fix.
#[test]
fn guard_binarizer_single_element_not_rejected() {
    // Live oracle: Binarizer().transform(np.zeros((1,1))) -> [[0.0]]
    let sklearn_expected: Array2<f64> = array![[0.0]];
    let b = Binarizer::<f64>::default();
    let x: Array2<f64> = Array2::zeros((1, 1));
    let out = b
        .transform(&x)
        .expect("(1,1) has 1 sample and 1 feature; sklearn accepts it");
    assert_eq!(out, sklearn_expected);
}

// ===========================================================================
// REQ-2 / REQ-3 / REQ-5 — stateful `Fit` -> `FittedBinarizer` path.
//
// sklearn `Binarizer.fit` (`_data.py:2257-2278`) "Only validates estimator's
// parameters": `_validate_data(X, accept_sparse="csr")` sets `n_features_in_`,
// validates X (default `force_all_finite=True` REJECTS NaN/±inf), returns self.
// `Binarizer.transform` (`:2280-2308`) `_validate_data(reset=False)` then
// `binarize(X, threshold=self.threshold)` — strict `>`, identical to the
// stateless / free-fn path.
//
// All expected values below are the LIVE sklearn 1.5.2 oracle (R-CHAR-3, run
// from /tmp), NEVER literal-copied from ferrolearn. The mixed-sign fixture is
//   X = [[1,-1,2],[2,0,0],[0,1,-1]]   (same as the REQ-4 free-fn fixture).
// ===========================================================================

/// REQ-3 oracle (live sklearn 1.5.2, run from /tmp), mixed-sign fixture
/// `X = [[1,-1,2],[2,0,0],[0,1,-1]]`:
/// ```text
/// python3 -c "import numpy as np; from sklearn.preprocessing import Binarizer
/// X=np.array([[1.,-1.,2.],[2.,0.,0.],[0.,1.,-1.]])
/// print(Binarizer().fit(X).transform(X).tolist())"          # -> [[1,0,1],[1,0,0],[0,1,0]]
/// print(Binarizer(threshold=0.5).fit(X).transform(X).tolist())"  # -> [[1,0,1],[1,0,0],[0,1,0]]
/// print(Binarizer(threshold=-0.5).fit(X).transform(X).tolist())" # -> [[1,0,1],[1,1,1],[1,1,0]]
/// ```
/// The `fit().transform()` path must equal `binarize(X, t)` AND the stateless
/// `Binarizer::new(t).transform(X)` AND the live sklearn oracle, bit-for-bit.
#[test]
fn fit_transform_matches_sklearn_and_stateless() {
    let x = array![[1.0, -1.0, 2.0], [2.0, 0.0, 0.0], [0.0, 1.0, -1.0]];

    // Live oracle outputs, keyed by threshold.
    let cases: [(f64, [[f64; 3]; 3]); 3] = [
        // Binarizer().fit(X).transform(X)  (default threshold 0.0)
        (0.0, [[1.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        // Binarizer(threshold=0.5).fit(X).transform(X)
        (0.5, [[1.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        // Binarizer(threshold=-0.5).fit(X).transform(X)
        (-0.5, [[1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 0.0]]),
    ];

    for (t, sk) in cases {
        let sklearn_expected =
            Array2::from_shape_vec((3, 3), sk.iter().flat_map(|r| r.iter().copied()).collect())
                .unwrap();

        let fit_out = Binarizer::<f64>::new(t).fit(&x, &()).unwrap().transform(&x);
        let fit_out = fit_out.unwrap();
        // Fit path == live sklearn oracle.
        assert_eq!(
            fit_out, sklearn_expected,
            "fit-path != sklearn at threshold {t}"
        );
        // Fit path == stateless `Binarizer::transform`, bit-for-bit.
        let stateless = Binarizer::<f64>::new(t).transform(&x).unwrap();
        assert_eq!(fit_out, stateless, "fit-path != stateless at threshold {t}");
        // Fit path == `binarize` free fn, bit-for-bit.
        let free = binarize(&x, t).unwrap();
        assert_eq!(fit_out, free, "fit-path != binarize() at threshold {t}");
    }
}

/// REQ-3 boundary: strict greater-than preserved through the Fit path — a value
/// EQUAL to the threshold maps to 0, a value just above maps to 1. Live oracle
/// (sklearn 1.5.2, run from /tmp):
/// ```text
/// python3 -c "import numpy as np; from sklearn.preprocessing import Binarizer
/// X=np.array([[0.5, 0.5000000001]])
/// print(Binarizer(threshold=0.5).fit(X).transform(X).tolist())"
/// ```
/// -> `[[0.0, 1.0]]`  (`0.5` is NOT > `0.5`; `0.5000000001` is > `0.5`)
#[test]
fn fit_strict_greater_boundary_preserved() {
    // Live oracle: Binarizer(threshold=0.5).fit(X).transform([[0.5, 0.5000000001]]) -> [[0,1]]
    let sklearn_expected: Array2<f64> = array![[0.0, 1.0]];
    let x = array![[0.5, 0.500_000_000_1]];
    let out = Binarizer::<f64>::new(0.5)
        .fit(&x, &())
        .unwrap()
        .transform(&x)
        .unwrap();
    assert_eq!(out, sklearn_expected);
}

/// REQ-5 oracle (live sklearn 1.5.2, run from /tmp):
/// ```text
/// python3 -c "import numpy as np; from sklearn.preprocessing import Binarizer
/// X=np.array([[1.,-1.,2.],[2.,0.,0.],[0.,1.,-1.]])
/// print(Binarizer().fit(X).n_features_in_)"
/// ```
/// -> `3`
///
/// `n_features_in_` equals the number of columns seen during `fit`.
#[test]
fn fit_n_features_in_matches_ncols() {
    let x = array![[1.0, -1.0, 2.0], [2.0, 0.0, 0.0], [0.0, 1.0, -1.0]];
    let fitted: FittedBinarizer<f64> = Binarizer::<f64>::default().fit(&x, &()).unwrap();
    assert_eq!(fitted.n_features_in(), 3);
    // A 2-column fit records 2.
    let x2 = array![[1.0, 2.0], [3.0, 4.0]];
    assert_eq!(
        Binarizer::<f64>::default()
            .fit(&x2, &())
            .unwrap()
            .n_features_in(),
        2
    );
}

/// REQ-3 oracle (live sklearn 1.5.2, run from /tmp):
/// ```text
/// python3 -c "import numpy as np; from sklearn.preprocessing import Binarizer
/// Binarizer().fit(np.array([[float('nan')]]))"
/// ```
/// -> `ValueError: Input X contains NaN. ...`
///
/// sklearn `fit` -> `_validate_data` default `force_all_finite=True` REJECTS
/// NaN; ferrolearn `fit` must return Err.
#[test]
fn fit_rejects_nan() {
    let x = array![[f64::NAN, 1.0]];
    assert!(
        Binarizer::<f64>::default().fit(&x, &()).is_err(),
        "sklearn Binarizer().fit raises ValueError on NaN; ferrolearn fit must Err"
    );
}

/// REQ-3 oracle (live sklearn 1.5.2, run from /tmp):
/// ```text
/// python3 -c "import numpy as np; from sklearn.preprocessing import Binarizer
/// Binarizer().fit(np.array([[float('inf')]]))"
/// ```
/// -> `ValueError: Input X contains infinity or a value too large ...`
#[test]
fn fit_rejects_pos_inf() {
    let x = array![[f64::INFINITY, 1.0]];
    assert!(
        Binarizer::<f64>::default().fit(&x, &()).is_err(),
        "sklearn Binarizer().fit raises ValueError on +inf; ferrolearn fit must Err"
    );
}

/// REQ-3 oracle (live sklearn 1.5.2, run from /tmp):
/// ```text
/// python3 -c "import numpy as np; from sklearn.preprocessing import Binarizer
/// Binarizer().fit(np.array([[float('-inf')]]))"
/// ```
/// -> `ValueError: Input X contains infinity or a value too large ...`
#[test]
fn fit_rejects_neg_inf() {
    let x = array![[f64::NEG_INFINITY, 1.0]];
    assert!(
        Binarizer::<f64>::default().fit(&x, &()).is_err(),
        "sklearn Binarizer().fit raises ValueError on -inf; ferrolearn fit must Err"
    );
}

/// REQ-2 oracle (live sklearn 1.5.2, run from /tmp):
/// ```text
/// python3 -c "import numpy as np; from sklearn.preprocessing import Binarizer
/// X=np.array([[1.,-1.,2.],[2.,0.,0.],[0.,1.,-1.]])
/// a=Binarizer(threshold=0.0,copy=True).fit(X).transform(X.copy())
/// b=Binarizer(threshold=0.0,copy=False).fit(X).transform(X.copy())
/// print(np.array_equal(a,b))"
/// ```
/// -> `True`
///
/// `copy` is an ACCEPT-AND-DOCUMENT no-op in ferrolearn: `copy=True` and
/// `copy=False` produce bit-identical output.
#[test]
fn fit_copy_true_false_identical_matches_sklearn() {
    let x = array![[1.0, -1.0, 2.0], [2.0, 0.0, 0.0], [0.0, 1.0, -1.0]];
    let a = Binarizer::<f64>::default()
        .with_copy(true)
        .fit(&x, &())
        .unwrap();
    let b = Binarizer::<f64>::default()
        .with_copy(false)
        .fit(&x, &())
        .unwrap();
    assert!(a.copy());
    assert!(!b.copy());
    let out_a = a.transform(&x).unwrap();
    let out_b = b.transform(&x).unwrap();
    assert_eq!(
        out_a, out_b,
        "copy flag changed the output (must be a no-op)"
    );
}

/// REQ-5 oracle (live sklearn 1.5.2, run from /tmp):
/// ```text
/// python3 -c "import numpy as np; from sklearn.preprocessing import Binarizer
/// m=Binarizer().fit(np.ones((2,3)))
/// m.transform(np.ones((2,5)))"
/// ```
/// -> `ValueError: X has 5 features, but Binarizer is expecting 3 features as
///     input.`
///
/// sklearn `transform` -> `_validate_data(reset=False)` (`_data.py:2301`) checks
/// the column count against the fitted `n_features_in_`. ferrolearn maps the
/// CLEAN (all-finite, non-empty) mismatch to `FerroError::ShapeMismatch`.
#[test]
fn fitted_transform_shape_mismatch() {
    let x_fit = array![[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]];
    let fitted = Binarizer::<f64>::default().fit(&x_fit, &()).unwrap();
    let x_wrong = array![[1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0]];
    let result = fitted.transform(&x_wrong);
    assert!(
        matches!(result, Err(FerroError::ShapeMismatch { .. })),
        "clean wrong-column-count after fit should map to FerroError::ShapeMismatch, got {result:?}"
    );
}

/// REQ-5 / #2207 ORDER oracle (live sklearn 1.5.2, run from /tmp). sklearn's
/// `_validate_data(reset=False)` runs `check_array` (finite / min checks) BEFORE
/// `_check_n_features` (`base.py:633` then `:654`). So an input that is BOTH
/// non-finite AND the wrong column count raises the `check_array` (finite)
/// error, NOT the n_features mismatch:
/// ```text
/// python3 -c "import numpy as np; from sklearn.preprocessing import Binarizer
/// m=Binarizer().fit(np.ones((2,3)))
/// m.transform(np.array([[float('nan'),1.0]]))"   # 2 cols (wrong) AND NaN
/// ```
/// -> `ValueError: Input X contains NaN. ...`   (the FINITE error, not n_features)
///
/// ferrolearn must therefore return the `check_array` error
/// (`FerroError::InvalidParameter`), NOT `ShapeMismatch`, for this input.
#[test]
fn fitted_transform_check_array_before_n_features() {
    // fit on 3 columns.
    let fitted = Binarizer::<f64>::default()
        .fit(&array![[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], &())
        .unwrap();
    // Held-out input: 2 columns (wrong count) AND a NaN.
    let x_bad = array![[f64::NAN, 1.0]];
    let result = fitted.transform(&x_bad);
    // sklearn raises the FINITE (check_array) error first, not the n_features one.
    assert!(
        matches!(result, Err(FerroError::InvalidParameter { .. })),
        "sklearn check_array (finite) runs BEFORE _check_n_features (#2207); \
         expected InvalidParameter (the finite error), got {result:?}"
    );
    assert!(
        !matches!(result, Err(FerroError::ShapeMismatch { .. })),
        "must NOT be ShapeMismatch — the finite check fires first (#2207)"
    );
}

/// REQ-3 — f32 Fit path matches the stateless f32 path AND the live oracle.
/// Oracle (sklearn 1.5.2, run from /tmp):
/// ```text
/// python3 -c "import numpy as np; from sklearn.preprocessing import Binarizer
/// X=np.array([[1.0,-1.0,0.0]],dtype=np.float32)
/// print(Binarizer(threshold=0.0).fit(X).transform(X).tolist())"
/// ```
/// -> `[[1.0, 0.0, 0.0]]`
#[test]
fn fit_f32_matches_oracle_and_stateless() {
    let sklearn_expected: Array2<f32> = array![[1.0f32, 0.0, 0.0]];
    let x: Array2<f32> = array![[1.0f32, -1.0, 0.0]];
    let fit_out = Binarizer::<f32>::new(0.0f32)
        .fit(&x, &())
        .unwrap()
        .transform(&x)
        .unwrap();
    assert_eq!(fit_out, sklearn_expected);
    let stateless = Binarizer::<f32>::new(0.0f32).transform(&x).unwrap();
    assert_eq!(fit_out, stateless);
}

// ===========================================================================
// DIVERGENCE — NaN / +-inf THRESHOLD over-acceptance.
//
// The builder's REQ-3/REQ-4 doc claims sklearn's `threshold:[Real]`
// (`_data.py:2249`) "accepts any float, incl. unusual ones — ferrolearn does
// NOT over-reject." That is FALSE for the binarize/transform path. The free
// function `binarize` is decorated with `@validate_params({"threshold":
// [Interval(Real, None, None, closed="neither")]})` (`_data.py:2112-2118`) — an
// OPEN interval `(-inf, inf)` that EXCLUDES NaN and +-inf. `Binarizer.transform`
// (`_data.py:2308`) calls `binarize(X, threshold=self.threshold, copy=False)`,
// so a NaN/inf threshold raises `InvalidParameterError`. LIVE oracle
// (sklearn 1.5.2, from /tmp):
//
//   python3 -c "import numpy as np; from sklearn.preprocessing import binarize
//   binarize(np.array([[5.,-1.,0.]]), threshold=float('nan'))"
//   -> InvalidParameterError: The 'threshold' parameter of binarize must be a
//      float in the range (-inf, inf). Got nan instead.
//   ... Binarizer(threshold=np.nan).transform(...)  -> InvalidParameterError (routes through binarize)
//   ... binarize(..., threshold=float('inf'))        -> InvalidParameterError
//
// ferrolearn's `binarize`/`Binarizer::transform` accept ANY `F` threshold:
// `mapv(|v| v > threshold)` runs unchecked. For threshold = NaN, every
// `x > NaN` is `false` -> all-zeros `Ok`; for threshold = +inf, nothing is
// `> inf` -> all-zeros `Ok`. ferrolearn returns Ok where sklearn raises.
// ===========================================================================

/// Divergence: `ferrolearn::binarize` (and `Binarizer::transform`, which would
/// call the same threshold path) diverges from
/// `sklearn/preprocessing/_data.py:2112-2118` (`@validate_params`,
/// `threshold: [Interval(Real, None, None, closed="neither")]`) for a NaN
/// threshold. sklearn raises `InvalidParameterError: The 'threshold' parameter
/// of binarize must be a float in the range (-inf, inf). Got nan instead.`;
/// ferrolearn accepts the NaN threshold and returns `Ok` of an all-zeros array
/// (every `x > NaN` is false). The builder's "[Real] accepts any float —
/// ferrolearn does NOT over-reject" claim is wrong: sklearn's open interval
/// EXCLUDES NaN/+-inf.
/// Tracking: #2208
#[test]
fn divergence_binarize_nan_threshold_should_error_like_sklearn() {
    // Live oracle: binarize([[5,-1,0]], threshold=nan) -> InvalidParameterError
    let x = array![[5.0_f64, -1.0, 0.0]];
    // Stateless `Binarizer::transform` with a NaN threshold. sklearn raises
    // (InvalidParameterError) because the threshold leaves the open (-inf, inf)
    // interval; ferrolearn must therefore also reject, not return Ok.
    let result = Binarizer::<f64>::new(f64::NAN).transform(&x);
    assert!(
        result.is_err(),
        "sklearn raises InvalidParameterError on a NaN threshold \
         (_data.py:2114 Interval(Real, closed=neither)); ferrolearn returned Ok({:?})",
        result.ok()
    );
    // The free `binarize` fn (the @validate_params boundary) rejects directly.
    assert!(
        binarize(&x, f64::NAN).is_err(),
        "free binarize(X, nan) must Err (sklearn @validate_params, _data.py:2114)"
    );
    // NOTE: `Binarizer.fit` does NOT reject a non-finite threshold — its
    // `_parameter_constraints {threshold: [Real]}` (`_data.py:2249`) is a bare
    // type check that accepts NaN/+-inf; only `transform`/`binarize` reject it
    // (#2209). That accept-at-fit behavior is pinned by
    // `divergence_binarizer_fit_accepts_nonfinite_threshold_like_sklearn`.
}

/// Divergence: `ferrolearn::binarize` diverges from
/// `sklearn/preprocessing/_data.py:2112-2118` for a `+inf` threshold. sklearn
/// raises `InvalidParameterError` (the open `(-inf, inf)` interval excludes
/// `inf`); ferrolearn accepts it and returns `Ok` of an all-zeros array
/// (nothing is `> inf`). Pins the threshold-domain over-acceptance for the
/// free-function entry point used by every Binarizer path.
/// Tracking: #2208
#[test]
fn divergence_binarize_inf_threshold_should_error_like_sklearn() {
    // Live oracle: binarize([[5,-1,1e308]], threshold=inf) -> InvalidParameterError
    let x = array![[5.0_f64, -1.0, 1e308]];
    // ferrolearn `binarize` free fn returns Array2 unconditionally — no Result.
    // sklearn raises. Pin via the estimator `transform`, which is the closest
    // Result-returning surface and shares the unchecked threshold path.
    let result = Binarizer::<f64>::new(f64::INFINITY).transform(&x);
    assert!(
        result.is_err(),
        "sklearn raises InvalidParameterError on a +inf threshold \
         (_data.py:2114 Interval(Real, closed=neither)); ferrolearn returned Ok({:?})",
        result.ok()
    );
}

// ===========================================================================
// DIVERGENCE (#2209) — Binarizer::fit OVER-REJECTS a non-finite threshold.
//
// REGRESSION introduced by the #2208 fix. The #2208 fix correctly added a
// non-finite-threshold rejection to the FREE function `binarize` and to
// `Binarizer::transform` (which both mirror sklearn's `binarize`
// `@validate_params({"threshold": [Interval(Real, None, None,
// closed="neither")]})`, `_data.py:2112-2118` — the OPEN interval excludes
// NaN/+-inf). BUT it ALSO added the same rejection to `Binarizer::fit`
// (`binarizer.rs:304-309`), where sklearn does NOT reject.
//
// sklearn's *estimator* `Binarizer` uses a DIFFERENT, looser constraint than
// the `binarize` free function: its `_parameter_constraints` is
// `{"threshold": [Real], "copy": ["boolean"]}` (`_data.py:2248-2251`). `[Real]`
// is the bare Real type-check, which ACCEPTS NaN and +-inf (only the OPEN
// `Interval` on `binarize` excludes them). `Binarizer.fit` is decorated
// `@_fit_context` (`_data.py:2257`) and its body only calls
// `self._validate_data(X, accept_sparse="csr")` (`:2277`) — it validates X but
// NOT the threshold against an interval. So a non-finite threshold passes fit.
//
// LIVE oracle (sklearn 1.5.2, run from /tmp):
//   python3 -c "import numpy as np; from sklearn.preprocessing import Binarizer
//   m=Binarizer(threshold=float('nan')).fit(np.array([[1.,-1.,2.],[2.,0.,0.],[0.,1.,-1.]]))
//   print('OK', m.n_features_in_)"
//   -> OK 3            # fit RETURNS a fitted estimator; does NOT raise
//   ... threshold=float('inf')  -> OK 3
//   ... threshold=float('-inf') -> OK 3
//
// ferrolearn `Binarizer::fit` checks `self.threshold.is_nan() ||
// self.threshold.is_infinite()` FIRST (`binarizer.rs:304-309`) and returns
// `Err(FerroError::InvalidParameter)` — rejecting an input sklearn accepts.
// This is an over-rejection divergence (R-DEV-2: ferrolearn must not reject
// inputs sklearn accepts). The in-tree #2208 tests `fit_rejects_nan` /
// `fit_rejects_pos_inf` / `fit_rejects_neg_inf` assert the WRONG (rejecting)
// behavior and are themselves divergent (tracked under #2209).
// ===========================================================================

/// Divergence: `ferrolearn::Binarizer::fit` over-rejects a non-finite
/// `threshold`, diverging from `sklearn/preprocessing/_data.py:2248-2251`
/// (`Binarizer._parameter_constraints = {"threshold": [Real], ...}` — the bare
/// `[Real]` type-check ACCEPTS NaN/+-inf; only the `binarize` free function uses
/// the OPEN `Interval(Real, None, None, closed="neither")` at `:2115`). sklearn
/// `Binarizer(threshold=nan).fit(goodX)` RETURNS a fitted estimator with
/// `n_features_in_ == 3` (live oracle); ferrolearn `fit` returns
/// `Err(InvalidParameter)` (`binarizer.rs:304-309`). Over-rejection regression
/// from the #2208 fix.
/// Tracking: #2209
#[test]
fn divergence_binarizer_fit_accepts_nonfinite_threshold_like_sklearn() {
    // Live oracle (sklearn 1.5.2, /tmp):
    //   Binarizer(threshold=nan).fit([[1,-1,2],[2,0,0],[0,1,-1]]) -> OK, n_features_in_=3
    //   ... threshold=inf  -> OK, n_features_in_=3
    //   ... threshold=-inf -> OK, n_features_in_=3
    let x = array![[1.0, -1.0, 2.0], [2.0, 0.0, 0.0], [0.0, 1.0, -1.0]];

    for thr in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let fitted = Binarizer::<f64>::new(thr).fit(&x, &());
        assert!(
            fitted.is_ok(),
            "sklearn Binarizer(threshold={thr:?}).fit(goodX) returns a fitted \
             estimator (_data.py:2248-2251 [Real] accepts NaN/+-inf); \
             ferrolearn fit returned {:?}",
            fitted.as_ref().err()
        );
        // sklearn records n_features_in_ == ncols == 3 even with a non-finite threshold.
        let n = fitted.unwrap().n_features_in();
        assert_eq!(
            n, 3,
            "sklearn fit records n_features_in_=3 (live oracle) for threshold {thr:?}; got {n}"
        );
    }
}
