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

use ferrolearn_core::traits::Transform;
use ferrolearn_preprocess::Binarizer;
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
