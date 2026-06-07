//! Binarizer: threshold features to binary values.
//!
//! Values strictly greater than the threshold are set to `1.0`; all other
//! values are set to `0.0`.
//!
//! This transformer is **stateless** — no fitting is required. Call
//! [`Transform::transform`] directly. For scikit-learn API parity it ALSO
//! supports the stateful [`Fit`](ferrolearn_core::traits::Fit) →
//! [`FittedBinarizer`] path, which records `n_features_in_` and (like sklearn)
//! validates the input in `fit`; the fitted type's `transform` reuses the very
//! same strict-greater logic as the stateless path, so both paths are
//! bit-identical.
//!
//! # `## REQ status`
//!
//! Binary (R-DEFER-2), translating `sklearn/preprocessing/_data.py` (`class Binarizer`
//! `:2177`, `binarize` `:2120`). Design doc: `.design/preprocess/binarizer.md`. Expected
//! values from the live sklearn 1.5.2 oracle (R-CHAR-3). Consumers: the in-file
//! `FittedBinarizer::transform` (the stateful fit→transform path) + crate re-export
//! (`lib.rs:106`, grandfathered S5). The SHIPPED REQs are critic-verified vs the oracle;
//! the remaining surface is NOT-STARTED with concrete blockers.
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (dense strict-greater transform) | SHIPPED | `Transform::transform` = `x.mapv(\|v\| if v > threshold { 1 } else { 0 })`, strict `>`, shape-preserving; `Default` threshold 0.0. Mirrors sklearn `binarize` dense path (`_data.py:2170-2173`). Critic-verified bit-identical to live sklearn (`guard_binarizer_*` in `tests/divergence_binarizer.rs`: thr 0.5 → `[[0,1,0],[1,0,0]]`, default, negative, f32). Consumer: `pub use binarizer::Binarizer` (`lib.rs:106`). |
//! | REQ-9 (transform input validation per check_array) | SHIPPED | FIXED #1123/#1124/#1125. `transform` rejects (in sklearn order) zero-samples → `InsufficientSamples` (`validation.py:1084`), zero-features → `InvalidParameter` (`:1093`), non-finite NaN/±inf → `InvalidParameter` (`:1063`, force_all_finite=True) — matching sklearn `Binarizer.transform` `_validate_data` (`_data.py:2301`). 13 live-oracle tests green; finite extremes (1e308/-0.0/subnormal) not over-rejected. Two-round critic-verified CLEAN. |
//! | REQ-2 (copy param) | SHIPPED | FIXED #1126. `Binarizer<F>` gains a `copy: bool` field (default `true`) + `#[must_use] with_copy` builder + `copy()` getter, threaded onto `FittedBinarizer`, mirroring sklearn `__init__(*, threshold=0.0, copy=True)` (`_data.py:2253-2255`, `_parameter_constraints {copy:["boolean"]}` `:2250`). ACCEPT-AND-DOCUMENT no-op: ferrolearn's [`Transform`] always returns a freshly allocated array (`binarize` → `mapv`), so `copy` has no observable effect — `copy=True`/`copy=False` produce identical output (sklearn's `copy=False` does in-place binarization, an optimization Rust's ownership makes moot here). Live-oracle test `fit_copy_true_false_identical_matches_sklearn`. Consumers: `FittedBinarizer` carries the flag + crate re-export `lib.rs:106`. |
//! | REQ-3 (fit + parameter-constraints validation) | SHIPPED | FIXED #1127. `impl Fit<Array2<F>, ()> for Binarizer<F>` (`fit`): runs the SAME `validate_binarize_input` guard as `Transform::transform`/`binarize` (REQ-9: zero-samples → `InsufficientSamples`, zero-features/non-finite NaN±inf → `InvalidParameter`; sklearn `_validate_data` default `force_all_finite=True` REJECTS NaN/inf — confirmed `Binarizer().fit([[nan]])`/`[[inf]]` raise `ValueError`, `:2277`, `utils/validation.py:1063/1084/1093`), records `n_features_in_ = x.ncols()`, returns `FittedBinarizer { threshold, copy, n_features_in_ }` (no fitted statistics — Binarizer is stateless, sklearn fit "Only validates", `:2257-2278`). THRESHOLD domain (#2209, R-HONEST-4): `Binarizer.fit` does NOT validate the threshold against an interval — its `_parameter_constraints {threshold: [Real]}` (`_data.py:2249`) is a BARE `Real` type check that ACCEPTS `NaN`/`±inf`, and `fit` (`:2257-2278`) only runs `_validate_data` on the DATA. So `Binarizer(threshold=nan/inf).fit(X)` is ACCEPTED here; the non-finite threshold is only rejected later by `transform`/`binarize` (whose `@validate_params` uses the OPEN `Interval(Real, None, None, closed="neither")`, `:2114-2115`). The `fit_rejects_nan/pos_inf/neg_inf` tests reject NaN/inf in the DATA `X` (REQ-9), not the threshold. Live-oracle tests: `fit_transform_matches_sklearn_and_stateless`, `fit_n_features_in_matches_ncols`, `fit_rejects_nan/pos_inf/neg_inf` (data), `fit_strict_greater_boundary_preserved`, `fitted_transform_check_array_before_n_features`, `fitted_transform_shape_mismatch`, `divergence_binarizer_fit_accepts_nonfinite_threshold_like_sklearn` (#2209) in `tests/divergence_binarizer.rs`. Consumers: `FittedBinarizer::transform` (the fitted path) + crate re-export `lib.rs:106`. |
//! | REQ-4 (binarize free function) | SHIPPED | FIXED #1128, #2208. Standalone [`binarize`] returns `Result<Array2<F>, FerroError>`: it FIRST rejects a non-finite `threshold` (`NaN`/`±inf` → `InvalidParameter`), mirroring sklearn's `@validate_params({"threshold": [Interval(Real, None, None, closed="neither")]})` (`_data.py:2112-2118`) — the OPEN interval `(-inf, inf)` EXCLUDES `NaN`/`±inf`, so `binarize(X, threshold=nan/inf)` raises `InvalidParameterError`; then `x.mapv(\|v\| if v > threshold { 1 } else { 0 })`, strict `>`, shape-preserving, mirroring the dense path (`_data.py:2120-2174`). Keyword default `threshold=0.0` documented. `Transform::transform` delegates to `binarize` (propagating the `Result`), so the two are byte-identical. FIXED #2208 (R-HONEST-4): the prior infallible signature + "[Real] accepts any float — ferrolearn does NOT over-reject" claim was WRONG; the non-finite threshold is now REJECTED. Critic-verified vs the live sklearn 1.5.2 oracle (`binarize_*_matches_sklearn`, `divergence_binarize_nan/inf_threshold_should_error_like_sklearn`). |
//! | REQ-5 (n_features_in_ / feature names) | PARTIAL | `n_features_in_` SHIPPED (FIXED #1129), `get_feature_names_out` NOT-STARTED. `FittedBinarizer<F>` records `n_features_in_ = x.ncols()` in `fit` and exposes `pub fn n_features_in(&self) -> usize`, mirroring sklearn's `_validate_data` setting `n_features_in_` (`:2277`); `FittedBinarizer::transform` validates the input column count against it (`ShapeMismatch`, sklearn `_validate_data(reset=False)` `:2301`), AFTER the `check_array` finite/min checks (#2207 order). The `OneToOneFeatureMixin.get_feature_names_out` / `feature_names_in_` string-name plumbing is OUT OF SCOPE for this build (no string feature-name infrastructure in ferrolearn yet) — keep prereq blocker #1129 open for the feature-name half. Live-oracle tests `fit_n_features_in_matches_ncols`, `fitted_transform_shape_mismatch`. |
//! | REQ-6 (sparse support) | NOT-STARTED | open prereq blocker #1130. Dense-only; no CSR/CSC path, no `threshold<0` guard, no `eliminate_zeros` (sklearn `:2161-2168`). |
//! | REQ-7 (PyO3 binding) | SHIPPED | FIXED #1131. `ferrolearn-python` registers `_RsBinarizer` (`py_transformer!` macro, `ferrolearn-python/src/extras.rs`, ctor `threshold: f64 = 0.0` + `copy: bool = true` mirroring sklearn `Binarizer.__init__(*, threshold=0.0, copy=True)` `_data.py:2253`; builds `Binarizer::<f64>::new(threshold).with_copy(copy)`, `fit(x)`→`FittedBinarizer`, `transform(x)`→binarized `PyArray2<f64>`; `FerroError`→`PyValueError`), wired in `ferrolearn-python/src/lib.rs` (`m.add_class::<extras::RsBinarizer>()`). Non-test production consumer (R-DEFER-1): `ferrolearn-python/python/ferrolearn/_extras.py::class Binarizer(_TransformerWrapper)` (keyword-only `__init__(*, threshold=0.0, copy=True)`, `_make_rs → _RsBinarizer(threshold, copy)`, inherits `fit`/`transform`/`fit_transform`) re-exported as `ferrolearn.Binarizer` (`ferrolearn-python/python/ferrolearn/__init__.py`). The non-finite-threshold accept-at-fit (#2209) / reject-at-transform (#2208) and NaN/±inf-input rejection (REQ-9) surface naturally as Python `ValueError`. Verification (model B, R-CHAR-3): `ferrolearn-python/tests/divergence_binarizer.py` — `fit_transform`/`fit`-then-`transform` value parity vs the live sklearn 1.5.2 oracle for thresholds 0.0/0.5/-1.0 on a mixed-sign fixture, strict-greater boundary (value == threshold → 0), default threshold 0.0, NaN/±inf input → `ValueError`, non-finite threshold rejected at `transform` / accepted at `fit`, `get_params`/`set_params`/`clone` round-trip of `threshold`/`copy`, `copy=True`/`copy=False` identical output. |
//! | REQ-8 (ferray substrate) | NOT-STARTED | open prereq blocker #1132. `ndarray`/`num_traits`, not `ferray-core`/`ferray-ufunc` (R-SUBSTRATE-1/2). |

use ferrolearn_core::error::FerroError;
use ferrolearn_core::traits::{Fit, Transform};
use ndarray::Array2;
use num_traits::Float;

// ---------------------------------------------------------------------------
// binarize (free function)
// ---------------------------------------------------------------------------

/// Boolean thresholding of a dense array, element by element.
///
/// Values **strictly greater** than `threshold` become `1.0`; all other values
/// (less than *or equal to* the threshold) become `0.0`. The result is a new,
/// shape-preserving array.
///
/// This is the estimator-less functional form of [`Binarizer`], mirroring
/// scikit-learn's `binarize(X, *, threshold=0.0, copy=True)`
/// (`sklearn/preprocessing/_data.py:2120-2174`), whose dense path is
/// `cond = X > threshold; X[cond] = 1; X[not_cond] = 0` (`:2170-2173`) — the
/// load-bearing strict greater-than. scikit-learn's keyword default is
/// `threshold=0.0` (only positive values map to `1.0`); here the caller passes
/// the threshold explicitly.
///
/// `binarize` is decorated `@validate_params({"threshold": [Interval(Real,
/// None, None, closed="neither")]})` (`_data.py:2112-2118`), an OPEN interval
/// `(-inf, inf)` that EXCLUDES `NaN` and `±inf`. A non-finite `threshold`
/// therefore raises `InvalidParameterError` (a `ValueError`) BEFORE any element
/// comparison; this function mirrors that by returning
/// [`FerroError::InvalidParameter`] for a non-finite threshold.
///
/// [`Binarizer`]'s [`Transform::transform`] delegates its element mapping to
/// this function, so the two share one implementation.
///
/// # Errors
///
/// Returns [`FerroError::InvalidParameter`] if `threshold` is `NaN` or `±inf`
/// (sklearn `Interval(Real, None, None, closed="neither")`, `_data.py:2114`).
///
/// # Examples
///
/// ```
/// use ferrolearn_preprocess::binarizer::binarize;
/// use ndarray::array;
///
/// let x = array![[0.4, 0.6, 0.5], [0.6, 0.1, 0.2]];
/// let out = binarize(&x, 0.5).unwrap();
/// // out = [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]
/// ```
pub fn binarize<F>(x: &Array2<F>, threshold: F) -> Result<Array2<F>, FerroError>
where
    F: Float,
{
    // sklearn `@validate_params` rejects a non-finite threshold at the
    // `binarize` boundary (`Interval(Real, None, None, closed="neither")`,
    // `_data.py:2114-2115`): the open `(-inf, inf)` interval excludes NaN/±inf.
    if threshold.is_nan() || threshold.is_infinite() {
        return Err(FerroError::InvalidParameter {
            name: "threshold".into(),
            reason: "must be a finite real number (got NaN or infinity)".into(),
        });
    }
    Ok(x.mapv(|v| if v > threshold { F::one() } else { F::zero() }))
}

/// Run the shared `check_array` input validation (REQ-9) used by both
/// [`Binarizer`]'s [`Transform::transform`] and [`Binarizer`]'s [`Fit::fit`], in
/// sklearn's `check_array` order: zero-samples → zero-features → non-finite
/// (`sklearn/utils/validation.py:1084`, `:1093`, `:1063`). Mirrors sklearn
/// `Binarizer.fit`/`.transform` → `_validate_data` (`_data.py:2277`, `:2301`),
/// whose default `force_all_finite=True` REJECTS NaN/±inf.
///
/// `context` names the calling site for diagnostics (e.g. `"Binarizer::transform"`
/// vs `"Binarizer::fit"`).
fn validate_binarize_input<F: Float>(x: &Array2<F>, context: &str) -> Result<(), FerroError> {
    if x.nrows() == 0 {
        return Err(FerroError::InsufficientSamples {
            required: 1,
            actual: 0,
            context: context.into(),
        });
    }
    if x.ncols() == 0 {
        return Err(FerroError::InvalidParameter {
            name: "X".to_string(),
            reason: "Found array with 0 feature(s); a minimum of 1 is required \
                     by Binarizer"
                .to_string(),
        });
    }
    if x.iter().any(|v| !v.is_finite()) {
        return Err(FerroError::InvalidParameter {
            name: "X".to_string(),
            reason: "Input X contains non-finite values (NaN or infinity); \
                     Binarizer requires all-finite input"
                .to_string(),
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Binarizer
// ---------------------------------------------------------------------------

/// A stateless feature binarizer.
///
/// Values strictly greater than `threshold` become `1.0`; all other values
/// become `0.0`. The default threshold is `0.0`.
///
/// This transformer is stateless — no fitting is needed. Call
/// [`Transform::transform`] directly.
///
/// # Examples
///
/// ```
/// use ferrolearn_preprocess::binarizer::Binarizer;
/// use ferrolearn_core::traits::Transform;
/// use ndarray::array;
///
/// let binarizer = Binarizer::<f64>::new(0.5);
/// let x = array![[0.0, 0.5, 1.0]];
/// let out = binarizer.transform(&x).unwrap();
/// // out = [[0.0, 0.0, 1.0]]
/// ```
#[derive(Debug, Clone)]
pub struct Binarizer<F> {
    /// The threshold value. Values strictly greater than this become 1.0.
    pub(crate) threshold: F,
    /// sklearn's `copy` constructor parameter (`__init__(*, threshold=0.0,
    /// copy=True)`, `_data.py:2253-2255`; `_parameter_constraints
    /// {copy:["boolean"]}` `:2248-2251`). ACCEPT-AND-DOCUMENT no-op: ferrolearn's
    /// [`Transform`] always returns a freshly allocated array, so `copy` has no
    /// observable effect on the output. Retained for API parity. Defaults to
    /// `true`.
    pub(crate) copy: bool,
}

impl<F: Float + Send + Sync + 'static> Binarizer<F> {
    /// Create a new `Binarizer` with the given threshold (and the default
    /// `copy = true`).
    ///
    /// sklearn constrains `threshold` to `Interval(Real, None, None,
    /// closed="neither")` on `binarize` (`_data.py:2114-2115`) — an OPEN
    /// interval `(-inf, inf)` that EXCLUDES `NaN`/`±inf`. A non-finite threshold
    /// is NOT rejected by `new` (no validation at construction, matching
    /// sklearn's `__init__`, which stores params unchecked); it is rejected
    /// later by [`Fit::fit`] / [`Transform::transform`] / [`binarize`]
    /// (`InvalidParameter`), matching sklearn's `_fit_context` /
    /// `@validate_params` raising `InvalidParameterError` at `fit`/`binarize`.
    #[must_use]
    pub fn new(threshold: F) -> Self {
        Self {
            threshold,
            copy: true,
        }
    }

    /// Return the configured threshold.
    #[must_use]
    pub fn threshold(&self) -> F {
        self.threshold
    }

    /// Set the `copy` parameter (sklearn `Binarizer(copy=...)`,
    /// `_data.py:2253`, `_parameter_constraints {copy:["boolean"]}` `:2250`).
    ///
    /// This is an ACCEPT-AND-DOCUMENT no-op: ferrolearn's [`Transform`] always
    /// returns a freshly allocated array, so `copy` has no observable effect on
    /// the output. It is retained for API parity with scikit-learn.
    #[must_use]
    pub fn with_copy(mut self, copy: bool) -> Self {
        self.copy = copy;
        self
    }

    /// Return the configured `copy` flag (sklearn `Binarizer.copy`).
    #[must_use]
    pub fn copy(&self) -> bool {
        self.copy
    }
}

impl<F: Float + Send + Sync + 'static> Default for Binarizer<F> {
    fn default() -> Self {
        Self::new(F::zero())
    }
}

// ---------------------------------------------------------------------------
// FittedBinarizer (sklearn stateful `fit` -> fitted estimator path)
// ---------------------------------------------------------------------------

/// A fitted [`Binarizer`].
///
/// `Binarizer` is stateless — its `fit` (sklearn `Binarizer.fit`,
/// `_data.py:2257-2278`, "Only validates estimator's parameters") learns NO
/// statistics; it merely validates the input and records `n_features_in_`. The
/// fitted type therefore carries only the configured `threshold`, the `copy`
/// flag, and the recorded feature count. Its [`Transform::transform`] reuses the
/// very same strict-greater logic as the stateless [`Binarizer`]/[`binarize`]
/// path, so the two paths are bit-identical.
#[derive(Debug, Clone)]
pub struct FittedBinarizer<F> {
    /// The threshold value. Values strictly greater than this become 1.0.
    pub(crate) threshold: F,
    /// The `copy` flag carried from the unfitted [`Binarizer`] (no-op; see
    /// [`Binarizer::with_copy`]).
    pub(crate) copy: bool,
    /// Number of features (columns) seen during [`Fit::fit`] — sklearn's
    /// `n_features_in_` (`_data.py:2277`, set by `_validate_data`).
    pub(crate) n_features_in_: usize,
}

impl<F: Float + Send + Sync + 'static> FittedBinarizer<F> {
    /// Return the number of features (columns) seen during [`Fit::fit`].
    ///
    /// Mirrors scikit-learn's `Binarizer.n_features_in_` (`_data.py:2277`).
    #[must_use]
    pub fn n_features_in(&self) -> usize {
        self.n_features_in_
    }

    /// Return the configured threshold.
    #[must_use]
    pub fn threshold(&self) -> F {
        self.threshold
    }

    /// Return the configured `copy` flag (no-op; see [`Binarizer::with_copy`]).
    #[must_use]
    pub fn copy(&self) -> bool {
        self.copy
    }
}

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, ()> for Binarizer<F> {
    type Fitted = FittedBinarizer<F>;
    type Error = FerroError;

    /// Validate the input and record `n_features_in_`, returning a
    /// [`FittedBinarizer`].
    ///
    /// `Binarizer` is stateless: like scikit-learn's `Binarizer.fit`
    /// (`sklearn/preprocessing/_data.py:2257-2278`, "Only validates estimator's
    /// parameters"), this learns NO statistics. It runs the SAME `check_array`
    /// validation as [`Transform::transform`] / [`binarize`] (REQ-9, via the
    /// shared [`validate_binarize_input`] helper) and records
    /// `n_features_in_ = x.ncols()`. sklearn's `_validate_data` uses the default
    /// `force_all_finite=True`, so NaN/±inf are REJECTED in `fit`
    /// (`Binarizer().fit([[nan]])` / `[[inf]]` raise `ValueError`). sklearn's
    /// `_fit_context` validates `_parameter_constraints` (`:2249`) BEFORE the
    /// data, and `threshold` is constrained to `Interval(Real, None, None,
    /// closed="neither")` on `binarize` (`_data.py:2114`) — an OPEN interval
    /// `(-inf, inf)` that EXCLUDES `NaN`/`±inf`. A non-finite `threshold` is
    /// therefore rejected here (param-check first, matching `_fit_context`).
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::InvalidParameter`] if `threshold` is non-finite
    /// (`NaN`/`±inf`, sklearn `Interval(Real, None, None, closed="neither")`,
    /// `_data.py:2114`), [`FerroError::InsufficientSamples`] for zero rows, and
    /// [`FerroError::InvalidParameter`] for zero features or any non-finite
    /// value (NaN, +inf, -inf) — matching `check_array`
    /// (`sklearn/utils/validation.py:1084`, `:1093`, `:1063`) as routed through
    /// `Binarizer.fit` -> `_validate_data` (`_data.py:2277`).
    fn fit(&self, x: &Array2<F>, _y: &()) -> Result<FittedBinarizer<F>, FerroError> {
        // sklearn `Binarizer._parameter_constraints = {"threshold": [Real], ...}`
        // (`_data.py:2249`) is a bare `Real` TYPE check that ACCEPTS NaN/+-inf —
        // UNLIKE the free `binarize`'s `Interval(Real, None, None,
        // closed="neither")` (`:2115`). `Binarizer.fit` (`:2257-2278`) validates
        // ONLY the data (`_validate_data`), never the threshold against an
        // interval, so a non-finite threshold is accepted here and only rejected
        // later by `transform` (which calls `binarize`). #2209.
        validate_binarize_input(x, "Binarizer::fit")?;
        Ok(FittedBinarizer {
            threshold: self.threshold,
            copy: self.copy,
            n_features_in_: x.ncols(),
        })
    }
}

impl<F: Float + Send + Sync + 'static> Transform<Array2<F>> for FittedBinarizer<F> {
    type Output = Array2<F>;
    type Error = FerroError;

    /// Threshold each element of `x`, delegating to the SAME strict-greater
    /// logic as the stateless [`Binarizer`] / [`binarize`] path.
    ///
    /// First applies the REQ-9 `check_array` guards (finite / min-samples /
    /// min-features) and binarizes, THEN validates that `x` has the same number
    /// of columns recorded during [`Fit::fit`]. This ORDER matches sklearn's
    /// `_validate_data(reset=False)`, which runs `check_array` BEFORE
    /// `_check_n_features` (`base.py:633` then `:654`, #2207): a NaN / ±inf /
    /// zero-sample / zero-feature input raises its `check_array` error EVEN when
    /// the column count is also wrong. Only after that does the `n_features`
    /// comparison fire. The output is therefore byte-identical to
    /// `Binarizer::transform` / `binarize(x, threshold)`.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the column count differs from
    /// `n_features_in_`. Returns [`FerroError::InsufficientSamples`] for zero
    /// rows and [`FerroError::InvalidParameter`] for zero features or any
    /// non-finite value (REQ-9, via [`validate_binarize_input`]).
    fn transform(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        // sklearn `_validate_data(reset=False)` runs `check_array` (finite /
        // min-samples / min-features) BEFORE `_check_n_features` (#2207). So
        // validate + binarize FIRST; a NaN / ±inf / zero-sample / zero-feature
        // input must raise its check_array error EVEN when the column count is
        // also wrong. Only then does the n_features comparison fire.
        validate_binarize_input(x, "FittedBinarizer::transform")?;
        let out = binarize(x, self.threshold)?;
        if x.ncols() != self.n_features_in_ {
            return Err(FerroError::ShapeMismatch {
                expected: vec![x.nrows(), self.n_features_in_],
                actual: vec![x.nrows(), x.ncols()],
                context: "FittedBinarizer::transform".into(),
            });
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Trait implementations
// ---------------------------------------------------------------------------

impl<F: Float + Send + Sync + 'static> Transform<Array2<F>> for Binarizer<F> {
    type Output = Array2<F>;
    type Error = FerroError;

    /// Apply the threshold: values > threshold become `1.0`, others become `0.0`.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::InsufficientSamples`] if `x` has zero rows. This
    /// mirrors scikit-learn's `Binarizer.transform`
    /// (`sklearn/preprocessing/_data.py:2301`), whose `_validate_data` ->
    /// `check_array` min-samples check raises `ValueError: Found array with 0
    /// sample(s) ... while a minimum of 1 is required by Binarizer.`
    ///
    /// Returns [`FerroError::InvalidParameter`] if `x` has zero features
    /// (columns). This mirrors scikit-learn's `Binarizer.transform`
    /// (`sklearn/preprocessing/_data.py:2301`), whose `_validate_data` ->
    /// `check_array` min-features check (`utils/validation.py:1093`,
    /// `ensure_min_features=1`) raises `ValueError: Found array with 0
    /// feature(s) (shape=(3, 0)) while a minimum of 1 is required by Binarizer.`
    ///
    /// Returns [`FerroError::InvalidParameter`] if `x` contains any non-finite
    /// value (NaN, +inf, or -inf). This mirrors scikit-learn's
    /// `Binarizer.transform` (`sklearn/preprocessing/_data.py:2301`), which
    /// validates input via `check_array(force_all_finite=True)` and raises
    /// `ValueError: Input X contains NaN.` / `Input X contains infinity ...`
    /// before applying the threshold comparison.
    fn transform(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        validate_binarize_input(x, "Binarizer::transform")?;
        binarize(x, self.threshold)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_binarizer_default_threshold() {
        let b = Binarizer::<f64>::default();
        assert_eq!(b.threshold(), 0.0);
        let x = array![[-1.0, 0.0, 0.5, 1.0]];
        let out = b.transform(&x).unwrap();
        assert_abs_diff_eq!(out[[0, 0]], 0.0, epsilon = 1e-10); // -1 <= 0
        assert_abs_diff_eq!(out[[0, 1]], 0.0, epsilon = 1e-10); // 0 not > 0
        assert_abs_diff_eq!(out[[0, 2]], 1.0, epsilon = 1e-10); // 0.5 > 0
        assert_abs_diff_eq!(out[[0, 3]], 1.0, epsilon = 1e-10); // 1.0 > 0
    }

    #[test]
    fn test_binarizer_custom_threshold() {
        let b = Binarizer::<f64>::new(0.5);
        let x = array![[0.0, 0.5, 1.0]];
        let out = b.transform(&x).unwrap();
        assert_abs_diff_eq!(out[[0, 0]], 0.0, epsilon = 1e-10); // 0.0 not > 0.5
        assert_abs_diff_eq!(out[[0, 1]], 0.0, epsilon = 1e-10); // 0.5 not > 0.5 (strict)
        assert_abs_diff_eq!(out[[0, 2]], 1.0, epsilon = 1e-10); // 1.0 > 0.5
    }

    #[test]
    fn test_binarizer_all_zeros() {
        let b = Binarizer::<f64>::new(0.0);
        let x = array![[0.0, 0.0, 0.0]];
        let out = b.transform(&x).unwrap();
        for v in &out {
            assert_abs_diff_eq!(*v, 0.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_binarizer_all_ones() {
        let b = Binarizer::<f64>::new(0.0);
        let x = array![[1.0, 2.0, 3.0]];
        let out = b.transform(&x).unwrap();
        for v in &out {
            assert_abs_diff_eq!(*v, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_binarizer_negative_threshold() {
        let b = Binarizer::<f64>::new(-1.0);
        let x = array![[-2.0, -1.0, -0.5, 0.0]];
        let out = b.transform(&x).unwrap();
        assert_abs_diff_eq!(out[[0, 0]], 0.0, epsilon = 1e-10); // -2 <= -1
        assert_abs_diff_eq!(out[[0, 1]], 0.0, epsilon = 1e-10); // -1 not > -1
        assert_abs_diff_eq!(out[[0, 2]], 1.0, epsilon = 1e-10); // -0.5 > -1
        assert_abs_diff_eq!(out[[0, 3]], 1.0, epsilon = 1e-10); // 0.0 > -1
    }

    #[test]
    fn test_binarizer_multiple_rows() {
        let b = Binarizer::<f64>::new(2.0);
        let x = array![[1.0, 3.0], [2.0, 4.0], [5.0, 0.0]];
        let out = b.transform(&x).unwrap();
        assert_eq!(out.shape(), &[3, 2]);
        assert_abs_diff_eq!(out[[0, 0]], 0.0, epsilon = 1e-10); // 1 <= 2
        assert_abs_diff_eq!(out[[0, 1]], 1.0, epsilon = 1e-10); // 3 > 2
        assert_abs_diff_eq!(out[[1, 0]], 0.0, epsilon = 1e-10); // 2 not > 2
        assert_abs_diff_eq!(out[[1, 1]], 1.0, epsilon = 1e-10); // 4 > 2
        assert_abs_diff_eq!(out[[2, 0]], 1.0, epsilon = 1e-10); // 5 > 2
        assert_abs_diff_eq!(out[[2, 1]], 0.0, epsilon = 1e-10); // 0 <= 2
    }

    #[test]
    fn test_binarizer_preserves_shape() {
        let b = Binarizer::<f64>::default();
        let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let out = b.transform(&x).unwrap();
        assert_eq!(out.shape(), x.shape());
    }

    #[test]
    fn test_binarizer_f32() {
        let b = Binarizer::<f32>::new(0.0f32);
        let x: Array2<f32> = array![[1.0f32, -1.0, 0.0]];
        let out = b.transform(&x).unwrap();
        assert!((out[[0, 0]] - 1.0f32).abs() < 1e-6);
        assert!((out[[0, 1]] - 0.0f32).abs() < 1e-6);
        assert!((out[[0, 2]] - 0.0f32).abs() < 1e-6);
    }

    // -- binarize free function (REQ-4) -- oracle-grounded vs live sklearn 1.5.2 --
    // X = [[1,-1,2],[2,0,0],[0,1,-1]]
    // python3 -c "from sklearn.preprocessing import binarize; import numpy as np; \
    //   print(binarize(np.array([[1.,-1,2],[2,0,0],[0,1,-1]])).tolist())"
    //   -> [[1,0,1],[1,0,0],[0,1,0]]   (threshold 0.0, strict >)
    // python3 -c "from sklearn.preprocessing import binarize; import numpy as np; \
    //   print(binarize(np.array([[1.,-1,2],[2,0,0],[0,1,-1]]), threshold=-0.5).tolist())"
    //   -> [[1,0,1],[1,1,1],[1,1,0]]

    #[test]
    fn binarize_default_threshold_matches_sklearn() {
        let x = array![[1.0, -1.0, 2.0], [2.0, 0.0, 0.0], [0.0, 1.0, -1.0]];
        let out = binarize(&x, 0.0).ok();
        let expected = array![[1.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        assert_eq!(out, Some(expected));
    }

    #[test]
    fn binarize_negative_threshold_matches_sklearn() {
        let x = array![[1.0, -1.0, 2.0], [2.0, 0.0, 0.0], [0.0, 1.0, -1.0]];
        let out = binarize(&x, -0.5).ok();
        let expected = array![[1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 0.0]];
        assert_eq!(out, Some(expected));
    }

    #[test]
    fn binarize_matches_estimator_transform() {
        let x = array![[1.0, -1.0, 2.0], [2.0, 0.0, 0.0], [0.0, 1.0, -1.0]];
        let free = binarize(&x, 0.5).ok();
        let est = Binarizer::<f64>::new(0.5).transform(&x).ok();
        assert_eq!(est, free);
    }

    #[test]
    fn test_output_values_are_zero_or_one() {
        let b = Binarizer::<f64>::new(0.0);
        let x = array![[-5.0, -1.0, 0.0, 0.001, 1.0, 100.0]];
        let out = b.transform(&x).unwrap();
        for v in &out {
            assert!(*v == 0.0 || *v == 1.0, "expected 0 or 1, got {v}");
        }
    }
}
