//! Normalizer: scale each sample (row) to unit norm.
//!
//! Unlike column-wise scalers, the `Normalizer` operates row-wise: each
//! sample is scaled independently so that its chosen norm equals 1.
//!
//! Supported norms:
//! - **L1**: divide by the sum of absolute values
//! - **L2**: divide by the Euclidean norm (default)
//! - **Max**: divide by the maximum absolute value
//!
//! Samples that already have a zero norm are left unchanged.
//!
//! This transformer is **stateless** — no fitting is required. Call
//! [`Transform::transform`] directly. For scikit-learn API parity it ALSO
//! supports the stateful [`Fit`](ferrolearn_core::traits::Fit) →
//! [`FittedNormalizer`] path, which records `n_features_in_` and (like
//! sklearn) validates the input in `fit`; the fitted type's `transform`
//! reuses the very same row-norm logic as the stateless path, so both paths
//! are bit-identical.
//!
//! # `## REQ status`
//!
//! Binary (R-DEFER-2), translating `sklearn/preprocessing/_data.py` (`class Normalizer`
//! `:1980`, `normalize` `:1866`). Design doc: `.design/preprocess/normalizer.md`. Expected
//! values from the live sklearn 1.5.2 oracle (R-CHAR-3). Consumers: the in-file
//! `PipelineTransformer`/`FittedPipelineTransformer` impls (pipeline integration) + crate
//! re-export (`lib.rs:119`, grandfathered S5). No PyO3 binding.
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (row-wise L1/L2/Max transform) | SHIPPED | `Transform::transform` divides each row by its norm (L1=Σ\|v\|, L2=√Σv², Max=max\|v\|; zero-norm row unchanged), default L2; mirrors sklearn dense `normalize` (`_data.py:1962-1969`, `_handle_zeros_in_scale` `:1968`). Critic-verified bit-identical to live oracle: `guard_l1/l2/max/zero_row/f32_matches_oracle` in `tests/divergence_normalizer.rs`. Consumers: `FittedPipelineTransformer::transform_pipeline` + crate re-export `lib.rs:119`. |
//! | REQ-2 (transform input validation per check_array) | SHIPPED | FIXED #1140. `transform` guards (sklearn order) zero-samples → `InsufficientSamples` (`validation.py:1084`), zero-features → `InvalidParameter` (`:1093`), non-finite NaN/±inf → `InvalidParameter` (`:1063`) — matching `Normalizer.transform` → `normalize` → `check_array` (`_data.py:1933-1940`). Mirrors converged `binarizer.rs`. Critic two-round CLEAN: 6 rejection pins + finite-not-over-rejected guards (zero-NORM-row/1e308/subnormal/-0.0); pipeline consumer inherits validation. |
//! | REQ-3 (validating fit + parameter constraints) | SHIPPED | FIXED #1141. `impl Fit<Array2<F>, ()> for Normalizer` (`fit`): runs the SAME `validate_normalize_input` guard as `Transform::transform`/`normalize` (REQ-2: zero-samples → `InsufficientSamples`, zero-features/non-finite NaN±inf → `InvalidParameter`, sklearn `_validate_data` default `force_all_finite=True` REJECTS NaN/inf — confirmed `Normalizer().fit([[nan]])`/`[[inf]]` raise ValueError, `:2082`,`utils/validation.py:1063/1084/1093`), records `n_features_in_ = x.ncols()`, returns `FittedNormalizer { norm, copy, n_features_in_ }` (no fitted statistics — Normalizer is stateless, sklearn fit "Only validates", `:2062-2083`). sklearn's `_parameter_constraints {norm:[StrOptions{l1,l2,max}]}` (`:2053-2055`) has NO ferrolearn analog: `NormType` is a closed Rust enum, so an out-of-domain norm is UNREPRESENTABLE rather than runtime-rejected — the type system satisfies the param-domain check. Live-oracle tests: `fit_l1/l2/max_matches_oracle_and_stateless`, `fit_rejects_nan/pos_inf/neg_inf`, `fit_zero_row_unchanged`, `fitted_transform_shape_mismatch`, `fit_path_equals_stateless_path` in `tests/divergence_normalizer.rs`. Consumers: `FittedNormalizer::transform` (the fitted path) + crate re-export `lib.rs:140`. |
//! | REQ-4 (normalize free fn: axis / return_norm) | SHIPPED | FIXED #1142. `pub fn normalize` + `pub fn normalize_with_norms` (free fns) mirror sklearn `normalize(X, norm, *, axis=1, copy=True, return_norm=False)` (`_data.py:1866`). Shared `row_norm` helper computes L1=Σ\|v\|, L2=√Σv², Max=max\|v\| (`:1962-1967`); `_handle_zeros_in_scale` zero→1 (`:1968`); `X /= norms` (`:1969`). `axis=1` row-normalizes; `axis=0` column-normalizes (sklearn transpose `:1926-1942`,`:1971-1972`); `axis ∉ {0,1}` → `InvalidParameter`. `normalize_with_norms` returns `(normalized, raw_norms)` (return_norm `:1974-1975`; raw, NOT zero-handled). Same validation as `Transform::transform` (REQ-2). Oracle-grounded tests in `#[cfg(test)]`: `normalize_l2/l1/max_axis1_matches_sklearn`, `normalize_l2_axis0_matches_sklearn`, `normalize_return_norm_l2_and_l1`, `normalize_invalid_axis_errors`. |
//! | REQ-5 (copy parameter) | SHIPPED | FIXED #1143. `Normalizer<F>` gains a `copy: bool` field (default `true`) + `#[must_use] with_copy` builder + `copy()` getter, threaded onto `FittedNormalizer`, mirroring sklearn `__init__(norm='l2', *, copy=True)` (`_data.py:2058-2060`, `_parameter_constraints {copy:["boolean"]}` `:2055`). ACCEPT-AND-DOCUMENT no-op: ferrolearn's `Transform` always returns a freshly allocated array (`to_owned()`), so `copy` has no observable effect — `copy=True`/`copy=False` produce identical output (sklearn's `copy=False` does in-place row normalization, an optimization Rust's ownership makes moot here). Live-oracle test `fit_copy_true_false_identical`. Consumers: `FittedNormalizer` carries the flag + crate re-export `lib.rs:140`. |
//! | REQ-6 (n_features_in_ / feature names) | PARTIAL | `n_features_in_` SHIPPED, `get_feature_names_out` NOT-STARTED. `FittedNormalizer<F>` records `n_features_in_ = x.ncols()` in `fit` and exposes `pub fn n_features_in(&self) -> usize`, mirroring sklearn's `_validate_data` setting `n_features_in_` (`:2082`); `FittedNormalizer::transform` validates the input column count against it (`ShapeMismatch`, sklearn `_validate_data(reset=False)` `:2104`). The `OneToOneFeatureMixin.get_feature_names_out` / `feature_names_in_` string-name plumbing is OUT OF SCOPE for this build (no string feature-name infrastructure in ferrolearn yet) — open prereq blocker #1144 for the feature-name half. Live-oracle test `fit_n_features_in_matches_ncols`. |
//! | REQ-7 (sparse support) | NOT-STARTED | open prereq blocker #1145. Dense-only; no CSR `inplace_csr_row_normalize_l1/l2` / `min_max_axis` Max (`:1944-1960`). |
//! | REQ-8 (PyO3 binding) | SHIPPED | FIXED #1146. `ferrolearn-python` surfaces `Normalizer` as `ferrolearn.Normalizer`: the hand-written `_RsNormalizer` `#[pyclass]` (`ferrolearn-python/src/extras.rs`, registered `lib.rs`) maps sklearn's `norm` STRING ('l1'/'l2'/'max') to the closed Rust `NormType` enum via `RsNormalizer::resolve_norm` — a bad string → `PyValueError` (sklearn `_parameter_constraints {norm: StrOptions({"l1","l2","max"})}`, `_data.py:2055`, `InvalidParameterError` ⊂ ValueError), builds `Normalizer::<f64>::new(normtype).with_copy(copy)`, runs the validating `Fit` (NaN/±inf → `PyValueError`, REQ-3) and delegates `transform` to `FittedNormalizer`. The non-test production consumer is `_extras.py::Normalizer(_TransformerWrapper)` with sklearn's `__init__(self, norm="l2", *, copy=True)` ABI (norm positional-or-keyword, copy keyword-only, `_data.py:2058`) + an overridden STATELESS `transform` (build-on-demand without fit, `_more_tags stateless=True` `_data.py:2110`, #2213) doing a FLOAT-ONLY dtype cast-back (float32→float32, float64→float64, int64→float64 UPCAST per `check_array(dtype=FLOAT_DTYPES)` `_data.py:2104`, #2214-analog — DIFFERS from Binarizer's number-preserving cast); re-exported in `__init__.py`. Verified vs the live sklearn 1.5.2 oracle: `tests/divergence_normalizer.py` (l1/l2/max values, default-l2, positional-norm, stateless, dtype, NaN/±inf, zero-norm, bad-norm, clone/get_params/set_params, copy no-op, pipeline). **Reduced-precision caveat (#2215, tracked):** sklearn `normalize` casts X to the INPUT float precision via `check_array(dtype=FLOAT_DTYPES)` (`_data.py:1933`) and computes the norm + division IN that precision (float16/float32), but the f64-only binding ABI (shared by EVERY `_Rs*` transformer) computes the norm in float64 then casts the result back — so float32 (~6e-8) and float16 (~5e-4) VALUES diverge slightly (dtype LABELS match; the float64 path is bit-exact, <1e-12). Same class as the generic-F precision caveats #2205/#2206; float16 is fundamentally unmatchable (the Rust core has no f16). Pinned `#[skip]` in `tests/divergence_normalizer_reduced_precision.py`. |
//! | REQ-9 (ferray substrate) | NOT-STARTED | open prereq blocker #1147. `ndarray::Array2` + `num_traits::Float`, not `ferray-core`/`ferray-ufunc` (R-SUBSTRATE-1/2). |

use ferrolearn_core::error::FerroError;
use ferrolearn_core::pipeline::{FittedPipelineTransformer, PipelineTransformer};
use ferrolearn_core::traits::{Fit, Transform};
use ndarray::{Array1, Array2, ArrayView1};
use num_traits::Float;

// ---------------------------------------------------------------------------
// NormType
// ---------------------------------------------------------------------------

/// The norm used by [`Normalizer`] when scaling each sample.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NormType {
    /// L1 norm: sum of absolute values.
    L1,
    /// L2 norm: Euclidean norm (square root of sum of squares). This is the default.
    #[default]
    L2,
    /// Max norm: maximum absolute value in the sample.
    Max,
}

// ---------------------------------------------------------------------------
// Normalizer
// ---------------------------------------------------------------------------

/// A stateless row-wise normalizer.
///
/// Each sample (row) is independently scaled so that its chosen norm equals 1.
/// Samples with a zero norm are left unchanged.
///
/// This transformer is stateless — no [`Fit`](ferrolearn_core::traits::Fit)
/// step is needed. Call [`Transform::transform`] directly.
///
/// # Examples
///
/// ```
/// use ferrolearn_preprocess::normalizer::{Normalizer, NormType};
/// use ferrolearn_core::traits::Transform;
/// use ndarray::array;
///
/// let normalizer = Normalizer::<f64>::new(NormType::L2);
/// let x = array![[3.0, 4.0], [1.0, 0.0]];
/// let out = normalizer.transform(&x).unwrap();
/// // Row 0: [3/5, 4/5], Row 1: [1.0, 0.0]
/// ```
#[derive(Debug, Clone)]
pub struct Normalizer<F> {
    /// The norm to use for normalisation.
    pub(crate) norm: NormType,
    /// sklearn's `copy` constructor parameter (`__init__(norm='l2', *, copy=True)`,
    /// `_data.py:2058-2060`; `_parameter_constraints {copy:["boolean"]}` `:2055`).
    /// ACCEPT-AND-DOCUMENT no-op: ferrolearn's [`Transform`] always returns a
    /// freshly allocated array, so `copy` has no observable effect. Retained for
    /// API parity. Defaults to `true`.
    pub(crate) copy: bool,
    _marker: std::marker::PhantomData<F>,
}

impl<F: Float + Send + Sync + 'static> Normalizer<F> {
    /// Create a new `Normalizer` with the specified norm type.
    #[must_use]
    pub fn new(norm: NormType) -> Self {
        Self {
            norm,
            copy: true,
            _marker: std::marker::PhantomData,
        }
    }

    /// Create a new `Normalizer` using the default L2 norm.
    #[must_use]
    pub fn l2() -> Self {
        Self::new(NormType::L2)
    }

    /// Create a new `Normalizer` using the L1 norm.
    #[must_use]
    pub fn l1() -> Self {
        Self::new(NormType::L1)
    }

    /// Create a new `Normalizer` using the Max norm.
    #[must_use]
    pub fn max() -> Self {
        Self::new(NormType::Max)
    }

    /// Return the configured norm type.
    #[must_use]
    pub fn norm(&self) -> NormType {
        self.norm
    }

    /// Set the `copy` parameter (sklearn `Normalizer(copy=...)`,
    /// `_data.py:2058`, `_parameter_constraints {copy:["boolean"]}` `:2055`).
    ///
    /// This is an ACCEPT-AND-DOCUMENT no-op: ferrolearn's [`Transform`] always
    /// returns a freshly allocated array, so `copy` has no observable effect on
    /// the output. It is retained for API parity with scikit-learn.
    #[must_use]
    pub fn with_copy(mut self, copy: bool) -> Self {
        self.copy = copy;
        self
    }

    /// Return the configured `copy` flag (sklearn `Normalizer.copy`).
    #[must_use]
    pub fn copy(&self) -> bool {
        self.copy
    }
}

impl<F: Float + Send + Sync + 'static> Default for Normalizer<F> {
    fn default() -> Self {
        Self::new(NormType::L2)
    }
}

// ---------------------------------------------------------------------------
// FittedNormalizer (sklearn stateful `fit` -> fitted estimator path)
// ---------------------------------------------------------------------------

/// A fitted [`Normalizer`].
///
/// `Normalizer` is stateless — its `fit` (sklearn `Normalizer.fit`,
/// `_data.py:2062-2083`, "Only validates estimator's parameters") learns NO
/// statistics; it merely validates the input and records `n_features_in_`. The
/// fitted type therefore carries only the configured `norm`, the `copy` flag,
/// and the recorded feature count. Its [`Transform::transform`] reuses the very
/// same row-norm logic as the stateless [`Normalizer`]/[`normalize`] path, so
/// the two paths are bit-identical.
#[derive(Debug, Clone)]
pub struct FittedNormalizer<F> {
    /// The norm to use for normalisation.
    pub(crate) norm: NormType,
    /// The `copy` flag carried from the unfitted [`Normalizer`] (no-op; see
    /// [`Normalizer::with_copy`]).
    pub(crate) copy: bool,
    /// Number of features (columns) seen during [`Fit::fit`] — sklearn's
    /// `n_features_in_` (`_data.py:2082`, set by `_validate_data`).
    pub(crate) n_features_in_: usize,
    _marker: std::marker::PhantomData<F>,
}

impl<F: Float + Send + Sync + 'static> FittedNormalizer<F> {
    /// Return the number of features (columns) seen during [`Fit::fit`].
    ///
    /// Mirrors scikit-learn's `Normalizer.n_features_in_` (`_data.py:2082`).
    #[must_use]
    pub fn n_features_in(&self) -> usize {
        self.n_features_in_
    }

    /// Return the configured norm type.
    #[must_use]
    pub fn norm(&self) -> NormType {
        self.norm
    }

    /// Return the configured `copy` flag (no-op; see [`Normalizer::with_copy`]).
    #[must_use]
    pub fn copy(&self) -> bool {
        self.copy
    }
}

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, ()> for Normalizer<F> {
    type Fitted = FittedNormalizer<F>;
    type Error = FerroError;

    /// Validate the input and record `n_features_in_`, returning a
    /// [`FittedNormalizer`].
    ///
    /// `Normalizer` is stateless: like scikit-learn's `Normalizer.fit`
    /// (`sklearn/preprocessing/_data.py:2062-2083`, "Only validates estimator's
    /// parameters"), this learns NO statistics. It runs the SAME `check_array`
    /// validation as [`Transform::transform`] / [`normalize`] (REQ-2, via the
    /// shared `validate_normalize_input` helper) and records
    /// `n_features_in_ = x.ncols()`. sklearn's `_validate_data` uses the default
    /// `force_all_finite=True`, so NaN/±inf are REJECTED in `fit`
    /// (`Normalizer().fit([[nan]])` / `[[inf]]` raise `ValueError`).
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::InsufficientSamples`] for zero rows and
    /// [`FerroError::InvalidParameter`] for zero features or any non-finite
    /// value (NaN, +inf, -inf) — matching `check_array`
    /// (`sklearn/utils/validation.py:1084`, `:1093`, `:1063`) as routed through
    /// `Normalizer.fit` -> `_validate_data` (`_data.py:2082`).
    fn fit(&self, x: &Array2<F>, _y: &()) -> Result<FittedNormalizer<F>, FerroError> {
        validate_normalize_input(x)?;
        Ok(FittedNormalizer {
            norm: self.norm,
            copy: self.copy,
            n_features_in_: x.ncols(),
            _marker: std::marker::PhantomData,
        })
    }
}

impl<F: Float + Send + Sync + 'static> Transform<Array2<F>> for FittedNormalizer<F> {
    type Output = Array2<F>;
    type Error = FerroError;

    /// Normalize each row of `x` to unit norm, delegating to the SAME row-norm
    /// logic as the stateless [`Normalizer`] / [`normalize`] path.
    ///
    /// First validates that `x` has the same number of columns recorded during
    /// [`Fit::fit`] (sklearn `_validate_data(reset=False)`,
    /// `sklearn/preprocessing/_data.py:2104`) and applies the REQ-2
    /// `check_array` guards, then calls the shared [`normalize`] free function
    /// with `axis=1` (sklearn `Normalizer.transform` ->
    /// `normalize(X, norm=self.norm, axis=1)`, `:2106`). The output is therefore
    /// byte-identical to `Normalizer::transform`.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the column count differs from
    /// `n_features_in_`. Returns [`FerroError::InsufficientSamples`] for zero
    /// rows and [`FerroError::InvalidParameter`] for zero features or any
    /// non-finite value (REQ-2, via [`normalize`]).
    fn transform(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        // sklearn `_validate_data(reset=False)` runs `check_array` (finite /
        // min-samples / min-features) BEFORE `_check_n_features` (`base.py:633`
        // then `:654`, #2207). So validate + normalize FIRST (this is
        // `check_array`'s job via the shared REQ-2 guard in `normalize`); a NaN /
        // +-inf / zero-sample / zero-feature input must raise its check_array
        // error EVEN when the column count is also wrong. Only after that does
        // the n_features comparison fire.
        let normalized = normalize(x, self.norm, 1)?;
        if x.ncols() != self.n_features_in_ {
            return Err(FerroError::ShapeMismatch {
                expected: vec![x.nrows(), self.n_features_in_],
                actual: vec![x.nrows(), x.ncols()],
                context: "FittedNormalizer::transform".into(),
            });
        }
        Ok(normalized)
    }
}

// ---------------------------------------------------------------------------
// Trait implementations
// ---------------------------------------------------------------------------

impl<F: Float + Send + Sync + 'static> Transform<Array2<F>> for Normalizer<F> {
    type Output = Array2<F>;
    type Error = FerroError;

    /// Normalize each row of `x` to unit norm.
    ///
    /// Rows with a zero norm value are left unchanged.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::InsufficientSamples`] if `x` has zero rows. This
    /// mirrors scikit-learn's `Normalizer.transform` ->
    /// `normalize` -> `check_array` (`sklearn/preprocessing/_data.py:1933`),
    /// whose min-samples check (`utils/validation.py:1084`,
    /// `ensure_min_samples=1`) raises `ValueError: Found array with 0 sample(s)
    /// ... while a minimum of 1 is required by Normalizer.`
    ///
    /// Returns [`FerroError::InvalidParameter`] if `x` has zero features
    /// (columns). This mirrors the same `check_array` min-features check
    /// (`utils/validation.py:1093`, `ensure_min_features=1`) which raises
    /// `ValueError: Found array with 0 feature(s) ... while a minimum of 1 is
    /// required by Normalizer.`
    ///
    /// Returns [`FerroError::InvalidParameter`] if `x` contains any non-finite
    /// value (NaN, +inf, or -inf). This mirrors `check_array(force_all_finite=
    /// True)` (`utils/validation.py:1063`), which raises `ValueError: Input X
    /// contains NaN.` / `Input X contains infinity ...` before normalizing.
    fn transform(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        if x.nrows() == 0 {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "Normalizer::transform".into(),
            });
        }
        if x.ncols() == 0 {
            return Err(FerroError::InvalidParameter {
                name: "X".to_string(),
                reason: "Found array with 0 feature(s); a minimum of 1 is required \
                         by Normalizer"
                    .to_string(),
            });
        }
        if x.iter().any(|v| !v.is_finite()) {
            return Err(FerroError::InvalidParameter {
                name: "X".to_string(),
                reason: "Input X contains non-finite values (NaN or infinity); \
                         Normalizer requires all-finite input"
                    .to_string(),
            });
        }
        let mut out = x.to_owned();
        for mut row in out.rows_mut() {
            let norm_val =
                match self.norm {
                    NormType::L1 => row.iter().copied().fold(F::zero(), |acc, v| acc + v.abs()),
                    NormType::L2 => row
                        .iter()
                        .copied()
                        .fold(F::zero(), |acc, v| acc + v * v)
                        .sqrt(),
                    NormType::Max => row.iter().copied().fold(F::zero(), |acc, v| {
                        if v.abs() > acc { v.abs() } else { acc }
                    }),
                };
            if norm_val == F::zero() {
                // Zero-norm row: leave unchanged.
                continue;
            }
            for v in &mut row {
                *v = *v / norm_val;
            }
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Standalone `normalize` free function (sklearn `normalize`, `_data.py:1866`)
// ---------------------------------------------------------------------------

/// Compute the `norm` of a single 1-D slice (one row or one column).
///
/// Mirrors sklearn's dense `normalize` per-vector norms (`_data.py:1962-1967`):
/// L1 = Σ|v|, L2 = √Σv², Max = max|v|.
fn row_norm<F: Float>(row: ArrayView1<F>, norm: NormType) -> F {
    match norm {
        NormType::L1 => row.iter().copied().fold(F::zero(), |acc, v| acc + v.abs()),
        NormType::L2 => row
            .iter()
            .copied()
            .fold(F::zero(), |acc, v| acc + v * v)
            .sqrt(),
        NormType::Max => {
            row.iter().copied().fold(
                F::zero(),
                |acc, v| {
                    if v.abs() > acc { v.abs() } else { acc }
                },
            )
        }
    }
}

/// Run the shared `check_array` input validation (REQ-2) used by both
/// [`Normalizer`]'s `transform` and the free [`normalize`]/[`normalize_with_norms`]
/// functions, in sklearn's `check_array` order (zero-samples → zero-features →
/// non-finite; `sklearn/utils/validation.py:1084`, `:1093`, `:1063`).
fn validate_normalize_input<F: Float>(x: &Array2<F>) -> Result<(), FerroError> {
    if x.nrows() == 0 {
        return Err(FerroError::InsufficientSamples {
            required: 1,
            actual: 0,
            context: "normalize".into(),
        });
    }
    if x.ncols() == 0 {
        return Err(FerroError::InvalidParameter {
            name: "X".to_string(),
            reason: "Found array with 0 feature(s); a minimum of 1 is required \
                     by the normalize function"
                .to_string(),
        });
    }
    if x.iter().any(|v| !v.is_finite()) {
        return Err(FerroError::InvalidParameter {
            name: "X".to_string(),
            reason: "Input X contains non-finite values (NaN or infinity); \
                     the normalize function requires all-finite input"
                .to_string(),
        });
    }
    Ok(())
}

/// Shared core of [`normalize`] / [`normalize_with_norms`]: validate `axis` and
/// input, then return the normalized array plus the per-axis **raw** norm vector.
///
/// The returned `norms` are the actual computed norms (NOT zero-handled): a
/// zero-norm row/column appears as `0.0` even though the division used `1`
/// (`_handle_zeros_in_scale`, `_data.py:1968`) to leave it unchanged. This
/// matches sklearn's `normalize(..., return_norm=True)` (`:1974-1975`).
fn normalize_inner<F: Float>(
    x: &Array2<F>,
    norm: NormType,
    axis: usize,
) -> Result<(Array2<F>, Array1<F>), FerroError> {
    if axis != 0 && axis != 1 {
        return Err(FerroError::InvalidParameter {
            name: "axis".into(),
            reason: "must be 0 or 1".into(),
        });
    }
    validate_normalize_input(x)?;

    let mut out = x.to_owned();
    if axis == 1 {
        // Row-normalize (sklearn default axis=1).
        let mut norms = Array1::<F>::zeros(out.nrows());
        for (i, mut row) in out.rows_mut().into_iter().enumerate() {
            let n = row_norm(row.view(), norm);
            norms[i] = n;
            // _handle_zeros_in_scale: a zero norm divides by 1 (row unchanged).
            let eff = if n == F::zero() { F::one() } else { n };
            for v in &mut row {
                *v = *v / eff;
            }
        }
        Ok((out, norms))
    } else {
        // axis == 0: column-normalize. sklearn transposes, runs the axis=1
        // path, then transposes back (`_data.py:1926-1942`, `:1971-1972`).
        let mut norms = Array1::<F>::zeros(out.ncols());
        for (j, mut col) in out.columns_mut().into_iter().enumerate() {
            let n = row_norm(col.view(), norm);
            norms[j] = n;
            let eff = if n == F::zero() { F::one() } else { n };
            for v in &mut col {
                *v = *v / eff;
            }
        }
        Ok((out, norms))
    }
}

/// Scale input vectors individually to unit norm — the standalone, estimator-less
/// API mirroring scikit-learn's `normalize` free function
/// (`sklearn/preprocessing/_data.py:1866`).
///
/// With `axis == 1` (sklearn's default) each **row** (sample) is divided by its
/// `norm` (L1 = Σ|v|, L2 = √Σv², Max = max|v|); with `axis == 0` each **column**
/// (feature) is normalized instead (sklearn transposes, row-normalizes, and
/// transposes back — `:1926-1942`, `:1971-1972`). A row/column whose norm is zero
/// is left unchanged, matching `_handle_zeros_in_scale` (`:1968`).
///
/// # Errors
///
/// Returns [`FerroError::InvalidParameter`] if `axis` is not `0` or `1`. Also
/// applies the same `check_array` input validation as [`Normalizer`]'s
/// `transform` (REQ-2): [`FerroError::InsufficientSamples`] for zero rows, and
/// [`FerroError::InvalidParameter`] for zero features or any non-finite value
/// (`_data.py:1933-1940`).
#[must_use = "normalize returns a new array; the input is not modified"]
pub fn normalize<F: Float>(
    x: &Array2<F>,
    norm: NormType,
    axis: usize,
) -> Result<Array2<F>, FerroError> {
    let (out, _norms) = normalize_inner(x, norm, axis)?;
    Ok(out)
}

/// Like [`normalize`] but also returns the per-axis norm vector — the
/// `return_norm=True` form of scikit-learn's `normalize`
/// (`sklearn/preprocessing/_data.py:1971-1975`).
///
/// Returns `(normalized, norms)` where `norms` is the per-row vector for
/// `axis == 1` (length = n_rows) or the per-column vector for `axis == 0`
/// (length = n_cols). The norms are the **raw** computed norms, NOT
/// zero-handled: a zero norm appears as `0.0` in the returned vector even though
/// the division used `1` to leave that row/column unchanged (sklearn returns the
/// raw `norms` array — `:1974-1975`).
///
/// # Errors
///
/// Same as [`normalize`].
#[must_use = "normalize_with_norms returns a new array and the norm vector"]
pub fn normalize_with_norms<F: Float>(
    x: &Array2<F>,
    norm: NormType,
    axis: usize,
) -> Result<(Array2<F>, Array1<F>), FerroError> {
    normalize_inner(x, norm, axis)
}

// ---------------------------------------------------------------------------
// Pipeline integration (generic)
// ---------------------------------------------------------------------------

impl<F: Float + Send + Sync + 'static> PipelineTransformer<F> for Normalizer<F> {
    /// Fit the normalizer using the pipeline interface.
    ///
    /// Because `Normalizer` is stateless, this simply boxes `self` as a
    /// [`FittedPipelineTransformer`].
    ///
    /// # Errors
    ///
    /// This implementation never returns an error.
    fn fit_pipeline(
        &self,
        _x: &Array2<F>,
        _y: &Array1<F>,
    ) -> Result<Box<dyn FittedPipelineTransformer<F>>, FerroError> {
        Ok(Box::new(self.clone()))
    }
}

impl<F: Float + Send + Sync + 'static> FittedPipelineTransformer<F> for Normalizer<F> {
    /// Transform data using the pipeline interface.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`Transform::transform`].
    fn transform_pipeline(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        self.transform(x)
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
    fn test_l2_norm_basic() {
        let norm = Normalizer::<f64>::l2();
        // Row [3, 4] has L2 norm 5.
        let x = array![[3.0, 4.0]];
        let out = norm.transform(&x).unwrap();
        assert_abs_diff_eq!(out[[0, 0]], 0.6, epsilon = 1e-10);
        assert_abs_diff_eq!(out[[0, 1]], 0.8, epsilon = 1e-10);
    }

    #[test]
    fn test_l2_unit_norm_after_transform() {
        let norm = Normalizer::<f64>::l2();
        let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let out = norm.transform(&x).unwrap();
        for row in out.rows() {
            let row_norm: f64 = row.iter().map(|v| v * v).sum::<f64>().sqrt();
            assert_abs_diff_eq!(row_norm, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_l1_norm_basic() {
        let norm = Normalizer::<f64>::l1();
        // Row [1, 2, 3] has L1 norm 6.
        let x = array![[1.0, 2.0, 3.0]];
        let out = norm.transform(&x).unwrap();
        assert_abs_diff_eq!(out[[0, 0]], 1.0 / 6.0, epsilon = 1e-10);
        assert_abs_diff_eq!(out[[0, 1]], 2.0 / 6.0, epsilon = 1e-10);
        assert_abs_diff_eq!(out[[0, 2]], 3.0 / 6.0, epsilon = 1e-10);
    }

    #[test]
    fn test_l1_unit_norm_after_transform() {
        let norm = Normalizer::<f64>::l1();
        let x = array![[1.0, 2.0, 3.0], [-4.0, 5.0, 6.0]];
        let out = norm.transform(&x).unwrap();
        for row in out.rows() {
            let row_norm: f64 = row.iter().map(|v| v.abs()).sum();
            assert_abs_diff_eq!(row_norm, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_max_norm_basic() {
        let norm = Normalizer::<f64>::max();
        // Row [-5, 3, 1] has max norm 5.
        let x = array![[-5.0, 3.0, 1.0]];
        let out = norm.transform(&x).unwrap();
        assert_abs_diff_eq!(out[[0, 0]], -1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(out[[0, 1]], 0.6, epsilon = 1e-10);
        assert_abs_diff_eq!(out[[0, 2]], 0.2, epsilon = 1e-10);
    }

    #[test]
    fn test_zero_row_unchanged() {
        let norm = Normalizer::<f64>::l2();
        let x = array![[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]];
        let out = norm.transform(&x).unwrap();
        // Zero row stays zero
        assert_abs_diff_eq!(out[[0, 0]], 0.0, epsilon = 1e-15);
        assert_abs_diff_eq!(out[[0, 1]], 0.0, epsilon = 1e-15);
        assert_abs_diff_eq!(out[[0, 2]], 0.0, epsilon = 1e-15);
    }

    #[test]
    fn test_negative_values_l2() {
        let norm = Normalizer::<f64>::l2();
        let x = array![[-3.0, -4.0]];
        let out = norm.transform(&x).unwrap();
        assert_abs_diff_eq!(out[[0, 0]], -0.6, epsilon = 1e-10);
        assert_abs_diff_eq!(out[[0, 1]], -0.8, epsilon = 1e-10);
    }

    #[test]
    fn test_default_is_l2() {
        let norm = Normalizer::<f64>::default();
        assert_eq!(norm.norm(), NormType::L2);
    }

    #[test]
    fn test_multiple_rows_independent() {
        let norm = Normalizer::<f64>::l2();
        let x = array![[3.0, 4.0], [0.0, 5.0]];
        let out = norm.transform(&x).unwrap();
        // Row 0: L2 norm = 5
        assert_abs_diff_eq!(out[[0, 0]], 0.6, epsilon = 1e-10);
        assert_abs_diff_eq!(out[[0, 1]], 0.8, epsilon = 1e-10);
        // Row 1: L2 norm = 5
        assert_abs_diff_eq!(out[[1, 0]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(out[[1, 1]], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_pipeline_integration() {
        use ferrolearn_core::pipeline::PipelineTransformer;
        let norm = Normalizer::<f64>::l2();
        let x = array![[3.0, 4.0], [0.0, 2.0]];
        let y = Array1::zeros(2);
        let fitted = norm.fit_pipeline(&x, &y).unwrap();
        let result = fitted.transform_pipeline(&x).unwrap();
        assert_abs_diff_eq!(result[[0, 0]], 0.6, epsilon = 1e-10);
        assert_abs_diff_eq!(result[[0, 1]], 0.8, epsilon = 1e-10);
    }

    #[test]
    fn test_f32_normalizer() {
        let norm = Normalizer::<f32>::l2();
        let x: Array2<f32> = array![[3.0f32, 4.0]];
        let out = norm.transform(&x).unwrap();
        assert!((out[[0, 0]] - 0.6f32).abs() < 1e-6);
        assert!((out[[0, 1]] - 0.8f32).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // REQ-4 — standalone `normalize` / `normalize_with_norms` free functions.
    // Oracle: live sklearn 1.5.2 (R-CHAR-3), X = [[1,2,2],[0,3,4]].
    //   normalize(X, l2, axis=1) -> [[.33333333,.66666667,.66666667],[0,.6,.8]]
    //   normalize(X, l1, axis=1) -> [[.2,.4,.4],[0,.42857143,.57142857]]
    //   normalize(X, max,axis=1) -> [[.5,1,1],[0,.75,1]]
    //   normalize(X, l2, axis=0) -> [[1,.5547002,.4472136],[0,.83205029,.89442719]]
    //   return_norm l2 axis=1 norms -> [3,5]; l1 axis=1 norms -> [5,7]
    // -----------------------------------------------------------------------

    #[test]
    fn normalize_l2_axis1_matches_sklearn() -> Result<(), FerroError> {
        let x = array![[1.0, 2.0, 2.0], [0.0, 3.0, 4.0]];
        let out = normalize(&x, NormType::L2, 1)?;
        let expected = array![[0.33333333, 0.66666667, 0.66666667], [0.0, 0.6, 0.8]];
        for (a, b) in out.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-7);
        }
        Ok(())
    }

    #[test]
    fn normalize_l1_axis1_matches_sklearn() -> Result<(), FerroError> {
        let x = array![[1.0, 2.0, 2.0], [0.0, 3.0, 4.0]];
        let out = normalize(&x, NormType::L1, 1)?;
        let expected = array![[0.2, 0.4, 0.4], [0.0, 0.42857143, 0.57142857]];
        for (a, b) in out.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-7);
        }
        Ok(())
    }

    #[test]
    fn normalize_max_axis1_matches_sklearn() -> Result<(), FerroError> {
        let x = array![[1.0, 2.0, 2.0], [0.0, 3.0, 4.0]];
        let out = normalize(&x, NormType::Max, 1)?;
        let expected = array![[0.5, 1.0, 1.0], [0.0, 0.75, 1.0]];
        for (a, b) in out.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-7);
        }
        Ok(())
    }

    #[test]
    fn normalize_l2_axis0_matches_sklearn() -> Result<(), FerroError> {
        let x = array![[1.0, 2.0, 2.0], [0.0, 3.0, 4.0]];
        let out = normalize(&x, NormType::L2, 0)?;
        let expected = array![[1.0, 0.5547002, 0.4472136], [0.0, 0.83205029, 0.89442719]];
        for (a, b) in out.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-7);
        }
        Ok(())
    }

    #[test]
    fn normalize_return_norm_l2_and_l1() -> Result<(), FerroError> {
        let x = array![[1.0, 2.0, 2.0], [0.0, 3.0, 4.0]];

        let (_out_l2, norms_l2) = normalize_with_norms(&x, NormType::L2, 1)?;
        assert_abs_diff_eq!(norms_l2[0], 3.0, epsilon = 1e-9);
        assert_abs_diff_eq!(norms_l2[1], 5.0, epsilon = 1e-9);

        let (_out_l1, norms_l1) = normalize_with_norms(&x, NormType::L1, 1)?;
        assert_abs_diff_eq!(norms_l1[0], 5.0, epsilon = 1e-9);
        assert_abs_diff_eq!(norms_l1[1], 7.0, epsilon = 1e-9);
        Ok(())
    }

    #[test]
    fn normalize_invalid_axis_errors() {
        let x = array![[1.0, 2.0, 2.0], [0.0, 3.0, 4.0]];
        let err = normalize(&x, NormType::L2, 2);
        assert!(matches!(err, Err(FerroError::InvalidParameter { .. })));
    }
}
