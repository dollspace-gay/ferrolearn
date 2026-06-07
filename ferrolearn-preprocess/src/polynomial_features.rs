//! Polynomial features: generate polynomial and interaction features.
//!
//! Given input features `[a, b]` and degree 2, this transformer generates:
//! - `[1, a, b, a², a·b, b²]` (default — `interaction_only = false`)
//! - `[1, a, b, a·b]` (with `interaction_only = true`)
//!
//! With `include_bias = false`, the constant column `1` is omitted.
//!
//! This transformer is stateless for value generation — call
//! [`Transform::transform`] directly. For scikit-learn API parity it ALSO
//! supports the stateful [`Fit`](ferrolearn_core::traits::Fit) →
//! [`FittedPolynomialFeatures`] path, which records `n_features_in_` /
//! `n_output_features_` / `powers_` and validates the input feature count in
//! `transform`; the fitted type's `transform` REUSES the very same value math as
//! the stateless path, so both paths are bit-identical to each other. That value
//! math (FIXED #2210) reproduces scikit-learn's INCREMENTAL degree-by-degree
//! column build (`_polynomial.py:525-564`) — each degree-`d` column is a stored
//! degree-`(d-1)` column times a single feature, in sklearn's `np.multiply` arg
//! order — so the polynomial VALUES are now bit-identical to the sklearn oracle
//! at EVERY degree, including degree≥3 (the prior independent per-column left-fold
//! diverged by 1 ULP on degree≥3 terms because FP multiplication is
//! non-associative).
//!
//! # `## REQ status`
//!
//! Binary (R-DEFER-2), translating `sklearn/preprocessing/_polynomial.py` (`class
//! PolynomialFeatures` `:99`). Design doc: `.design/preprocess/polynomial_features.md`. Expected
//! values from the live sklearn 1.5.2 oracle (R-CHAR-3). Consumers: `PipelineTransformer` impl +
//! crate re-export (`lib.rs`, grandfathered S5) + the `FittedPolynomialFeatures::transform` fitted
//! path. HONEST (R-HONEST-3): a dense int-degree transformer — the polynomial VALUES + column ORDER
//! match sklearn exactly, and the stateful `fit` → `FittedPolynomialFeatures` surface
//! (`n_features_in_`/`n_output_features_`/`powers_` + the transform feature-count check) NOW SHIPS
//! (REQ-4/REQ-5); `get_feature_names_out`, degree-tuple, `order`, sparse, full-ctor, PyO3, ferray
//! stay NOT-STARTED.
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (int-degree dense values + column order) | SHIPPED | Column order: `fn feature_combinations` (bias=empty combo, DFS interaction_only strictly-increasing / else non-decreasing, sort by degree-then-lex) reproduces sklearn `_combinations` itertools order (`_polynomial.py:209-220`) column-for-column (this drives `powers_`/`n_output_features_`). Value math: `fn generate_poly_features` builds the columns DEGREE-BY-DEGREE via sklearn's incremental block recurrence (`_polynomial.py:525-564`) — each degree-`d` column = a stored degree-`(d-1)` column × a single feature in sklearn's `np.multiply` arg order — so values are bit-identical to the live oracle at EVERY degree. FIXED #2210: the prior independent per-column left-fold (`combo.iter().fold(1, |acc,j| acc*x[j])`) diverged by 1 ULP on degree≥3 terms (FP multiplication is non-associative); the incremental build now matches bit-for-bit (pin `divergence_polynomial_higher_degree_term_ulp_vs_sklearn`, x1*x2*x3 == `0x1.7b645a1cac082p+2`). Green guards (`tests/divergence_polynomial_features.rs`): 2-feat default `[1,2,3,4,6,9]`, 3-feat default `[1,2,3,5,4,6,10,9,15,25]`, 3-feat interaction `[1,2,3,5,6,10,15]`, deg-3 interaction no-bias `[2,3,5,6,10,15,30]`, multi-row. Consumers: `FittedPipelineTransformer` + re-export. |
//! | REQ-8 (input validation per check_array) | SHIPPED | FIXED #1180. `transform` guards (sklearn order) zero-samples → `InsufficientSamples`, zero-features → `InvalidParameter`, non-finite NaN/±inf → `InvalidParameter` — matching sklearn `transform` → `_validate_data` (`_polynomial.py:433-435`). Mirrors converged binarizer/normalizer. Critic two-round CLEAN: 13 tests incl. finite input (1e308) with overflowing product correctly ACCEPTED (input-only validation). |
//! | REQ-2 (degree tuple / min_degree) | NOT-STARTED | open prereq blocker #1181. Single `usize` degree, always starts at degree 1 (sklearn `:334-360`). |
//! | REQ-3 (order C/F param) | NOT-STARTED | open prereq blocker #1182. No `order` (sklearn `:201`,`:132-134`). |
//! | REQ-4 (stateful fit + n_features_in_/n_output_features_ + count check) | SHIPPED | FIXED #1183. `impl Fit<Array2<F>, ()> for PolynomialFeatures<F>` (`fit`): runs the SAME REQ-8 `validate_poly_input` guard (zero-samples → `InsufficientSamples`, zero-features/non-finite → `InvalidParameter`, sklearn `_validate_data` `_polynomial.py:323`), enumerates `feature_combinations(x.ncols())` (REUSED, not reimplemented), records `n_features_in_ = x.ncols()` and `n_output_features_ = combinations.len()` (== `_num_combinations`, sklearn `:362`), returns `FittedPolynomialFeatures`. `FittedPolynomialFeatures::transform` validates X (REQ-8) FIRST, generates via the SHARED `generate_poly_features` value math (delegated — byte-identical to the stateless path), THEN checks `x.ncols() != n_features_in_` → `ShapeMismatch` (sklearn `X has N features, but ... expecting M`, `:402-435`). The X-validation-before-n_features ORDER matches sklearn `_validate_data(reset=False)` → `check_array` before `_check_n_features` (#2207). Accessors `n_features_in()`/`n_output_features()`. Live-oracle tests: `fit_n_features_in_matches_ncols`, `fit_n_output_features_*`, `fitted_transform_matches_stateless_and_sklearn`, `fitted_transform_more/fewer_features_shape_mismatch`, `fitted_transform_nan_validation_before_n_features`. Consumers: `FittedPolynomialFeatures::transform` (the fitted path) + crate re-export `pub use polynomial_features::{PolynomialFeatures, FittedPolynomialFeatures}` (`lib.rs`). |
//! | REQ-5 (powers_ attribute) | SHIPPED | FIXED #1184. `FittedPolynomialFeatures<F>` carries `powers_: Array2<usize>` of shape `(n_output_features_, n_features_in_)` (matching `n_features_in_`'s `usize` storage idiom; sklearn `powers_` is int), built in `fit` from the SAME `feature_combinations` the value math uses (so row order == output column order): each combination → a `bincount` row counting occurrences of each feature index `0..n_features_in_`, the bias empty-combo → an all-zeros row (sklearn `np.vstack([np.bincount(c, minlength=n_features_in_) for c in combinations])`, `_polynomial.py:262-264`). Accessor `powers(&self) -> &Array2<usize>`. Live-oracle tests: `fit_powers_2feat/3feat_*` EXACT-match sklearn `.powers_` for default / `interaction_only` / `include_bias=False`, asserting shape + every entry. Consumer: the `powers_` accessor on the re-exported `FittedPolynomialFeatures`. |
//! | REQ-6 (get_feature_names_out) | NOT-STARTED | open prereq blocker #1185. None (sklearn `:266-303`). Depends on REQ-5. |
//! | REQ-7 (sparse CSR/CSC) | NOT-STARTED | open prereq blocker #1186. Dense-only (sklearn `:402-`,`:38-96`). |
//! | REQ-9 (full ctor + _parameter_constraints) | NOT-STARTED | open prereq blocker #1187. Positional `new`, degree==0 only (sklearn `:194-207`). |
//! | REQ-10 (PyO3 binding) | NOT-STARTED | open prereq blocker #1188. No `ferrolearn-python` registration (R-DEFER-1). |
//! | REQ-11 (ferray substrate) | NOT-STARTED | open prereq blocker #1189. `ndarray`+`num_traits`, not `ferray-core` (R-SUBSTRATE-1/2). |

use ferrolearn_core::error::FerroError;
use ferrolearn_core::pipeline::{FittedPipelineTransformer, PipelineTransformer};
use ferrolearn_core::traits::{Fit, Transform};
use ndarray::{Array1, Array2};
use num_traits::Float;

// ---------------------------------------------------------------------------
// PolynomialFeatures
// ---------------------------------------------------------------------------

/// A stateless polynomial feature generator.
///
/// Generates all polynomial combinations of the input features up to the
/// specified degree.
///
/// # Configuration
///
/// - `degree`: maximum polynomial degree (default `2`).
/// - `interaction_only`: if `true`, only cross-product terms are generated
///   (no pure powers like `a²`). Default `false`.
/// - `include_bias`: if `true`, a constant column of ones is prepended.
///   Default `true`.
///
/// # Examples
///
/// ```
/// use ferrolearn_preprocess::polynomial_features::PolynomialFeatures;
/// use ferrolearn_core::traits::Transform;
/// use ndarray::array;
///
/// let poly = PolynomialFeatures::<f64>::new(2, false, true).unwrap();
/// let x = array![[2.0, 3.0]];
/// let out = poly.transform(&x).unwrap();
/// // out = [[1, 2, 3, 4, 6, 9]]
/// ```
#[derive(Debug, Clone)]
pub struct PolynomialFeatures<F> {
    /// Maximum polynomial degree.
    pub(crate) degree: usize,
    /// If `true`, only interaction terms are produced (no pure powers).
    pub(crate) interaction_only: bool,
    /// If `true`, prepend a bias (constant ones) column.
    pub(crate) include_bias: bool,
    _marker: std::marker::PhantomData<F>,
}

impl<F: Float + Send + Sync + 'static> PolynomialFeatures<F> {
    /// Create a new `PolynomialFeatures` transformer.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::InvalidParameter`] if `degree == 0`.
    pub fn new(
        degree: usize,
        interaction_only: bool,
        include_bias: bool,
    ) -> Result<Self, FerroError> {
        if degree == 0 {
            return Err(FerroError::InvalidParameter {
                name: "degree".into(),
                reason: "degree must be at least 1".into(),
            });
        }
        Ok(Self {
            degree,
            interaction_only,
            include_bias,
            _marker: std::marker::PhantomData,
        })
    }

    /// Create a `PolynomialFeatures` with default settings:
    /// degree=2, interaction_only=false, include_bias=true.
    #[must_use]
    pub fn default_config() -> Self {
        Self {
            degree: 2,
            interaction_only: false,
            include_bias: true,
            _marker: std::marker::PhantomData,
        }
    }

    /// Return the configured degree.
    #[must_use]
    pub fn degree(&self) -> usize {
        self.degree
    }

    /// Return whether only interaction terms are generated.
    #[must_use]
    pub fn interaction_only(&self) -> bool {
        self.interaction_only
    }

    /// Return whether a bias column is included.
    #[must_use]
    pub fn include_bias(&self) -> bool {
        self.include_bias
    }

    /// Generate all combinations (with repetition unless `interaction_only`)
    /// of feature indices up to `degree`.
    ///
    /// Returns a list of index-tuples, where each tuple specifies which
    /// feature indices to multiply together to produce one output column.
    fn feature_combinations(&self, n_features: usize) -> Vec<Vec<usize>> {
        let mut combos: Vec<Vec<usize>> = Vec::new();

        // Bias term: empty product = 1
        if self.include_bias {
            combos.push(vec![]);
        }

        // Generate combinations of degrees 1..=self.degree
        let mut stack: Vec<(Vec<usize>, usize)> = Vec::new();

        // Start with each feature at degree 1
        for i in 0..n_features {
            stack.push((vec![i], i));
        }

        while let Some((combo, last_idx)) = stack.pop() {
            combos.push(combo.clone());

            if combo.len() < self.degree {
                // Extend with another feature
                let start = if self.interaction_only {
                    // Strictly increasing indices — no repeated features
                    last_idx + 1
                } else {
                    // Non-decreasing indices — allows repeated features (pure powers)
                    last_idx
                };
                for i in start..n_features {
                    let mut new_combo = combo.clone();
                    new_combo.push(i);
                    stack.push((new_combo, i));
                }
            }
        }

        // Sort: bias first, then by combo length, then lexicographically
        combos.sort_by(|a, b| a.len().cmp(&b.len()).then_with(|| a.cmp(b)));

        combos
    }
}

/// Run the shared `check_array` input validation (REQ-8) used by the stateless
/// [`Transform::transform`], the stateful [`Fit::fit`], and the
/// [`FittedPolynomialFeatures`] transform path, in sklearn's `check_array`
/// order: zero-samples → zero-features → non-finite
/// (`sklearn/utils/validation.py:1084`, `:1093`, `:1063`). Mirrors sklearn
/// `PolynomialFeatures.fit`/`.transform` → `_validate_data`
/// (`_polynomial.py:323`, `:433-435`), whose default `force_all_finite=True`
/// REJECTS NaN/±inf.
///
/// `context` names the calling site for diagnostics (e.g.
/// `"PolynomialFeatures::transform"` vs `"PolynomialFeatures::fit"`).
fn validate_poly_input<F: Float>(x: &Array2<F>, context: &str) -> Result<(), FerroError> {
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
                     by PolynomialFeatures"
                .to_string(),
        });
    }
    if x.iter().any(|v| !v.is_finite()) {
        return Err(FerroError::InvalidParameter {
            name: "X".to_string(),
            reason: "Input X contains non-finite values (NaN or infinity); \
                     PolynomialFeatures requires all-finite input"
                .to_string(),
        });
    }
    Ok(())
}

/// Generate the dense polynomial feature matrix (REQ-1 value math), shared by
/// the stateless [`Transform::transform`] and the stateful
/// [`FittedPolynomialFeatures`] transform path so the two are byte-identical.
///
/// Builds the output columns DEGREE-BY-DEGREE using scikit-learn's incremental
/// block recurrence (`_polynomial.py:525-564`), NOT an independent per-column
/// left-fold: the degree-1 block is `X` (copied in feature order); each degree-`d`
/// block is built feature-by-feature, where for `feature_idx` the new columns are
/// the just-built degree-`(d-1)` columns in slice `[start, end)` each multiplied
/// by `X[:, feature_idx]` — i.e. `previous_degree_column * X[:, feature_idx]` in
/// THAT multiply order, matching `np.multiply(XP[:, start:end], X[:, feature_idx],
/// casting="no")` bit-for-bit (FP multiplication is non-associative, so the
/// association order is load-bearing for the last ULP on degree≥3 terms). The
/// resulting column order is exactly sklearn's (bias, then degree-1, then each
/// degree-`d` block by `feature_idx`), which equals the [`feature_combinations`]
/// order the `powers_`/REQ-1 guards pin. The bias column (if `include_bias`) is
/// `1`. The input is assumed already validated by [`validate_poly_input`].
fn generate_poly_features<F: Float>(
    x: &Array2<F>,
    degree: usize,
    interaction_only: bool,
    include_bias: bool,
) -> Array2<F> {
    let n_samples = x.nrows();
    let n_features = x.ncols();
    let n_out = num_output_columns(n_features, degree, interaction_only, include_bias);
    let mut out = Array2::zeros((n_samples, n_out));

    // degree 0 (bias) term
    let mut current_col = 0usize;
    if include_bias {
        for i in 0..n_samples {
            out[[i, 0]] = F::one();
        }
        current_col = 1;
    }

    if n_features == 0 {
        return out;
    }

    // degree 1 block: copy X column-for-column.
    // `index[j]` is the output column where degree-(d-1) block for feature `j`
    // begins; the trailing `index[n_features]` marks one past the block end.
    let mut index: Vec<usize> = Vec::with_capacity(n_features + 1);
    for j in 0..n_features {
        index.push(current_col);
        for i in 0..n_samples {
            out[[i, current_col]] = x[[i, j]];
        }
        current_col += 1;
    }
    index.push(current_col);

    // degree >= 2 blocks, built incrementally from the previous degree block.
    for _ in 2..=degree {
        let mut new_index: Vec<usize> = Vec::with_capacity(n_features + 1);
        // sklearn `end = index[-1]` (`_polynomial.py:544`): the LAST element of
        // `index`, NOT `index[n_features]`. When `interaction_only` runs out of
        // combinations at a high degree the inner loop `break`s early, leaving
        // `index` (the previous `new_index`) with FEWER than `n_features+1`
        // entries — so indexing `[n_features]` would panic (#2211, R-CODE-2). The
        // last element is the correct degree-(d-1) block end either way.
        let end = index.last().copied().unwrap_or(0);
        for feature_idx in 0..n_features {
            let mut start = index[feature_idx];
            new_index.push(current_col);
            if interaction_only {
                start += index[feature_idx + 1] - index[feature_idx];
            }
            if end <= start {
                // next_col <= current_col → no new columns for this feature.
                break;
            }
            // new column = (stored degree-(d-1) column) * X[:, feature_idx],
            // in that np.multiply arg order (prev FIRST, single feature SECOND).
            for src_col in start..end {
                for i in 0..n_samples {
                    out[[i, current_col]] = out[[i, src_col]] * x[[i, feature_idx]];
                }
                current_col += 1;
            }
        }
        new_index.push(current_col);
        index = new_index;
    }

    out
}

/// Number of output columns for the incremental build (== number of
/// [`feature_combinations`] entries / `n_output_features_`): the count of
/// degree-`1..=degree` combinations (with repetition unless `interaction_only`)
/// over `n_features` features, plus one for the bias if `include_bias`.
fn num_output_columns(
    n_features: usize,
    degree: usize,
    interaction_only: bool,
    include_bias: bool,
) -> usize {
    let mut total = usize::from(include_bias);
    if n_features == 0 {
        return total;
    }
    // Count combinations per degree by mirroring the incremental block sizes.
    // degree-1 block has `n_features` columns; subsequent block sizes follow the
    // same start/end recurrence as the value build.
    let mut index: Vec<usize> = (0..=n_features).collect();
    total += n_features;
    let mut current = n_features;
    for _ in 2..=degree {
        let mut new_index: Vec<usize> = Vec::with_capacity(n_features + 1);
        // sklearn `index[-1]` (the last element), not `index[n_features]` —
        // robust to the `interaction_only` high-degree early break (#2211).
        let end = index.last().copied().unwrap_or(0);
        for feature_idx in 0..n_features {
            let mut start = index[feature_idx];
            new_index.push(current);
            if interaction_only {
                start += index[feature_idx + 1] - index[feature_idx];
            }
            if end <= start {
                break;
            }
            current += end - start;
            total += end - start;
        }
        new_index.push(current);
        index = new_index;
    }
    total
}

/// Build the `powers_` exponent matrix from a precomputed combination list
/// (REQ-5). Returns an `Array2<usize>` of shape `(combos.len(), n_features)`
/// where `powers_[k, j]` is the number of occurrences of input feature `j` in
/// combination `k` (its exponent) — sklearn's
/// `np.vstack([np.bincount(c, minlength=n_features_in_) for c in combinations])`
/// (`_polynomial.py:262-264`). The bias empty-combo → an all-zeros row. Built
/// from the SAME combinations the value math uses, so the row order equals the
/// output column order.
fn build_powers(combos: &[Vec<usize>], n_features: usize) -> Array2<usize> {
    let mut powers = Array2::<usize>::zeros((combos.len(), n_features));
    for (k, combo) in combos.iter().enumerate() {
        for &j in combo {
            powers[[k, j]] += 1;
        }
    }
    powers
}

impl<F: Float + Send + Sync + 'static> Default for PolynomialFeatures<F> {
    fn default() -> Self {
        Self::default_config()
    }
}

// ---------------------------------------------------------------------------
// Trait implementations
// ---------------------------------------------------------------------------

impl<F: Float + Send + Sync + 'static> Transform<Array2<F>> for PolynomialFeatures<F> {
    type Output = Array2<F>;
    type Error = FerroError;

    /// Generate polynomial and interaction features.
    ///
    /// # Errors
    ///
    /// Mirrors scikit-learn's `PolynomialFeatures.transform`
    /// (`sklearn/preprocessing/_polynomial.py:433`, `self._validate_data(...)` ->
    /// `check_array`), validating in `check_array` order
    /// (samples -> features -> finite):
    ///
    /// - Returns [`FerroError::InsufficientSamples`] if `x` has zero rows
    ///   (`check_array` `ensure_min_samples=1`; sklearn raises `ValueError:
    ///   Found array with 0 sample(s) ... a minimum of 1 is required`).
    /// - Returns [`FerroError::InvalidParameter`] if `x` has zero columns
    ///   (`check_array` `ensure_min_features=1`; sklearn raises `ValueError:
    ///   Found array with 0 feature(s) ... a minimum of 1 is required`).
    /// - Returns [`FerroError::InvalidParameter`] if `x` contains any non-finite
    ///   value (NaN, +inf, or -inf; `check_array` `force_all_finite=True`;
    ///   sklearn raises `ValueError: Input X contains NaN` / `infinity ...`).
    fn transform(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        validate_poly_input(x, "PolynomialFeatures::transform")?;
        Ok(generate_poly_features(
            x,
            self.degree,
            self.interaction_only,
            self.include_bias,
        ))
    }
}

// ---------------------------------------------------------------------------
// FittedPolynomialFeatures (sklearn stateful `fit` -> fitted estimator path)
// ---------------------------------------------------------------------------

/// A fitted [`PolynomialFeatures`].
///
/// `PolynomialFeatures.fit` (sklearn `_polynomial.py:306-400`, "Compute number
/// of output features") learns NO numeric statistics — it records only the
/// shape metadata derivable from `n_features_in_` and the configured params:
/// `n_features_in_`, `n_output_features_` (the number of output columns), and
/// the `powers_` exponent matrix. The fitted type's [`Transform::transform`]
/// REUSES the very same combination enumeration ([`PolynomialFeatures::
/// feature_combinations`]) and value math ([`generate_poly_features`]) as the
/// stateless [`PolynomialFeatures`] path, so the two paths are bit-identical;
/// it additionally enforces the transform-time feature-count check against
/// `n_features_in_` (sklearn `X has N features, but ... expecting M`).
#[derive(Debug, Clone)]
pub struct FittedPolynomialFeatures<F> {
    /// Maximum polynomial degree (carried from the unfitted [`PolynomialFeatures`]).
    pub(crate) degree: usize,
    /// If `true`, only interaction terms are produced (no pure powers).
    pub(crate) interaction_only: bool,
    /// If `true`, a bias (constant ones) column is prepended.
    pub(crate) include_bias: bool,
    /// Number of features (columns) seen during [`Fit::fit`] — sklearn's
    /// `n_features_in_` (`_polynomial.py:323`, set by `_validate_data`).
    pub(crate) n_features_in_: usize,
    /// Total number of polynomial output columns — sklearn's
    /// `n_output_features_` (`_polynomial.py:362`, == `_num_combinations`).
    pub(crate) n_output_features_: usize,
    /// The exponent matrix of shape `(n_output_features_, n_features_in_)`:
    /// `powers_[i, j]` is the exponent of input feature `j` in output feature
    /// `i` — sklearn's `powers_` (`_polynomial.py:250-264`). The bias empty-combo
    /// row is all zeros.
    pub(crate) powers_: Array2<usize>,
    _marker: std::marker::PhantomData<F>,
}

impl<F: Float + Send + Sync + 'static> FittedPolynomialFeatures<F> {
    /// Return the number of features (columns) seen during [`Fit::fit`].
    ///
    /// Mirrors scikit-learn's `PolynomialFeatures.n_features_in_`
    /// (`_polynomial.py:323`).
    #[must_use]
    pub fn n_features_in(&self) -> usize {
        self.n_features_in_
    }

    /// Return the total number of polynomial output columns.
    ///
    /// Mirrors scikit-learn's `PolynomialFeatures.n_output_features_`
    /// (`_polynomial.py:362`).
    #[must_use]
    pub fn n_output_features(&self) -> usize {
        self.n_output_features_
    }

    /// Return the `powers_` exponent matrix of shape
    /// `(n_output_features_, n_features_in_)`.
    ///
    /// `powers_[i, j]` is the exponent of input feature `j` in output feature
    /// `i`, mirroring scikit-learn's `PolynomialFeatures.powers_`
    /// (`_polynomial.py:250-264`).
    #[must_use]
    pub fn powers(&self) -> &Array2<usize> {
        &self.powers_
    }

    /// Return the configured degree.
    #[must_use]
    pub fn degree(&self) -> usize {
        self.degree
    }

    /// Return whether only interaction terms are generated.
    #[must_use]
    pub fn interaction_only(&self) -> bool {
        self.interaction_only
    }

    /// Return whether a bias column is included.
    #[must_use]
    pub fn include_bias(&self) -> bool {
        self.include_bias
    }
}

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, ()> for PolynomialFeatures<F> {
    type Fitted = FittedPolynomialFeatures<F>;
    type Error = FerroError;

    /// Validate the input and record `n_features_in_` / `n_output_features_` /
    /// `powers_`, returning a [`FittedPolynomialFeatures`].
    ///
    /// `PolynomialFeatures.fit` (sklearn `_polynomial.py:306-400`, "Compute
    /// number of output features") learns NO numeric statistics. This runs the
    /// SAME REQ-8 `check_array` validation as [`Transform::transform`] (via the
    /// shared [`validate_poly_input`] helper; sklearn's `_validate_data` default
    /// `force_all_finite=True` REJECTS NaN/±inf), enumerates the polynomial
    /// combinations by REUSING [`PolynomialFeatures::feature_combinations`] (no
    /// reimplementation), and records `n_features_in_ = x.ncols()`,
    /// `n_output_features_ = combinations.len()` (== sklearn's
    /// `_num_combinations`, `:362`), and the `powers_` exponent matrix
    /// (`build_powers` = sklearn's `np.bincount` over the combinations, `:262`).
    /// `powers_` is built from the SAME combinations the transform value math
    /// uses, so its row order equals the output column order.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::InsufficientSamples`] for zero rows and
    /// [`FerroError::InvalidParameter`] for zero features or any non-finite
    /// value (NaN, +inf, -inf) — matching `check_array`
    /// (`sklearn/utils/validation.py:1084`, `:1093`, `:1063`) as routed through
    /// `PolynomialFeatures.fit` → `_validate_data` (`_polynomial.py:323`).
    fn fit(&self, x: &Array2<F>, _y: &()) -> Result<FittedPolynomialFeatures<F>, FerroError> {
        validate_poly_input(x, "PolynomialFeatures::fit")?;
        let n_features_in_ = x.ncols();
        // REUSE the SHIPPED combination enumeration (REQ-1) — do NOT reimplement.
        let combos = self.feature_combinations(n_features_in_);
        let n_output_features_ = combos.len();
        let powers_ = build_powers(&combos, n_features_in_);
        Ok(FittedPolynomialFeatures {
            degree: self.degree,
            interaction_only: self.interaction_only,
            include_bias: self.include_bias,
            n_features_in_,
            n_output_features_,
            powers_,
            _marker: std::marker::PhantomData,
        })
    }
}

impl<F: Float + Send + Sync + 'static> Transform<Array2<F>> for FittedPolynomialFeatures<F> {
    type Output = Array2<F>;
    type Error = FerroError;

    /// Generate polynomial features, delegating to the SAME combination
    /// enumeration + value math as the stateless [`PolynomialFeatures`] path.
    ///
    /// First applies the REQ-8 `check_array` guards (min-samples / min-features
    /// / finite) and generates the polynomial matrix, THEN validates that `x`
    /// has the same number of columns recorded during [`Fit::fit`]. This ORDER
    /// matches sklearn's `_validate_data(reset=False)`, which runs `check_array`
    /// BEFORE the `n_features_in_` consistency check (`_polynomial.py:433-435`,
    /// #2207): a NaN / ±inf / zero-sample / zero-feature input raises its
    /// `check_array` error EVEN when the column count is also wrong. Only after
    /// that does the feature-count comparison fire. Because the combinations are
    /// re-enumerated from the SAME params (degree / interaction_only /
    /// include_bias) and the value math is the shared [`generate_poly_features`],
    /// the output is byte-identical to `PolynomialFeatures::transform`.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the column count differs from
    /// `n_features_in_` (sklearn `ValueError: X has N features, but
    /// PolynomialFeatures is expecting M features as input.`,
    /// `_polynomial.py:402-435`). Returns [`FerroError::InsufficientSamples`]
    /// for zero rows and [`FerroError::InvalidParameter`] for zero features or
    /// any non-finite value (REQ-8, via [`validate_poly_input`]).
    fn transform(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        // sklearn `_validate_data(reset=False)` runs `check_array` (finite /
        // min-samples / min-features) BEFORE the `n_features_in_` check (#2207).
        // So validate + generate FIRST; a NaN / ±inf / zero-sample / zero-feature
        // input must raise its check_array error EVEN when the column count is
        // also wrong. Only then does the feature-count comparison fire.
        validate_poly_input(x, "FittedPolynomialFeatures::transform")?;
        // REUSE the SHARED incremental value math — build from the configured
        // params so the column order matches the stateless path (and `powers_`)
        // exactly.
        let out = generate_poly_features(x, self.degree, self.interaction_only, self.include_bias);
        if x.ncols() != self.n_features_in_ {
            return Err(FerroError::ShapeMismatch {
                expected: vec![x.nrows(), self.n_features_in_],
                actual: vec![x.nrows(), x.ncols()],
                context: "FittedPolynomialFeatures::transform".into(),
            });
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Pipeline integration (generic)
// ---------------------------------------------------------------------------

impl<F: Float + Send + Sync + 'static> PipelineTransformer<F> for PolynomialFeatures<F> {
    /// Fit the polynomial features transformer using the pipeline interface.
    ///
    /// Because `PolynomialFeatures` is stateless, this simply boxes `self`
    /// as a [`FittedPipelineTransformer`].
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

impl<F: Float + Send + Sync + 'static> FittedPipelineTransformer<F> for PolynomialFeatures<F> {
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
    fn test_degree2_two_features_with_bias() {
        // degree=2, interaction_only=false, include_bias=true
        // Expected: [1, a, b, a², a·b, b²]
        let poly = PolynomialFeatures::<f64>::new(2, false, true).unwrap();
        let x = array![[2.0, 3.0]];
        let out = poly.transform(&x).unwrap();
        assert_eq!(out.shape()[0], 1);
        assert_eq!(out.shape()[1], 6); // 1 + 2 + 3 combinations
        assert_abs_diff_eq!(out[[0, 0]], 1.0, epsilon = 1e-10); // bias
        assert_abs_diff_eq!(out[[0, 1]], 2.0, epsilon = 1e-10); // a
        assert_abs_diff_eq!(out[[0, 2]], 3.0, epsilon = 1e-10); // b
        assert_abs_diff_eq!(out[[0, 3]], 4.0, epsilon = 1e-10); // a²
        assert_abs_diff_eq!(out[[0, 4]], 6.0, epsilon = 1e-10); // a·b
        assert_abs_diff_eq!(out[[0, 5]], 9.0, epsilon = 1e-10); // b²
    }

    #[test]
    fn test_degree2_interaction_only() {
        // degree=2, interaction_only=true, include_bias=true
        // Expected: [1, a, b, a·b]
        let poly = PolynomialFeatures::<f64>::new(2, true, true).unwrap();
        let x = array![[2.0, 3.0]];
        let out = poly.transform(&x).unwrap();
        assert_eq!(out.shape()[1], 4);
        assert_abs_diff_eq!(out[[0, 0]], 1.0, epsilon = 1e-10); // bias
        assert_abs_diff_eq!(out[[0, 1]], 2.0, epsilon = 1e-10); // a
        assert_abs_diff_eq!(out[[0, 2]], 3.0, epsilon = 1e-10); // b
        assert_abs_diff_eq!(out[[0, 3]], 6.0, epsilon = 1e-10); // a·b
    }

    #[test]
    fn test_no_bias() {
        // degree=2, interaction_only=false, include_bias=false
        // Expected: [a, b, a², a·b, b²]
        let poly = PolynomialFeatures::<f64>::new(2, false, false).unwrap();
        let x = array![[2.0, 3.0]];
        let out = poly.transform(&x).unwrap();
        assert_eq!(out.shape()[1], 5);
        assert_abs_diff_eq!(out[[0, 0]], 2.0, epsilon = 1e-10); // a
    }

    #[test]
    fn test_degree1_only_linear() {
        let poly = PolynomialFeatures::<f64>::new(1, false, true).unwrap();
        let x = array![[2.0, 3.0]];
        let out = poly.transform(&x).unwrap();
        // [1, a, b]
        assert_eq!(out.shape()[1], 3);
        assert_abs_diff_eq!(out[[0, 0]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(out[[0, 1]], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(out[[0, 2]], 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_multiple_rows() {
        let poly = PolynomialFeatures::<f64>::new(2, false, false).unwrap();
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let out = poly.transform(&x).unwrap();
        assert_eq!(out.shape(), &[2, 5]);
        // Row 0: a=1, b=2 → [1, 2, 1, 2, 4]
        assert_abs_diff_eq!(out[[0, 0]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(out[[0, 1]], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(out[[0, 2]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(out[[0, 3]], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(out[[0, 4]], 4.0, epsilon = 1e-10);
    }

    #[test]
    fn test_single_feature_degree2() {
        // [a] → [1, a, a²]
        let poly = PolynomialFeatures::<f64>::new(2, false, true).unwrap();
        let x = array![[3.0]];
        let out = poly.transform(&x).unwrap();
        assert_eq!(out.shape()[1], 3);
        assert_abs_diff_eq!(out[[0, 0]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(out[[0, 1]], 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(out[[0, 2]], 9.0, epsilon = 1e-10);
    }

    #[test]
    fn test_invalid_degree_zero() {
        assert!(PolynomialFeatures::<f64>::new(0, false, true).is_err());
    }

    #[test]
    fn test_default_config() {
        let poly = PolynomialFeatures::<f64>::default();
        assert_eq!(poly.degree(), 2);
        assert!(!poly.interaction_only());
        assert!(poly.include_bias());
    }

    #[test]
    fn test_pipeline_integration() {
        use ferrolearn_core::pipeline::PipelineTransformer;
        let poly = PolynomialFeatures::<f64>::new(2, false, true).unwrap();
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = Array1::zeros(2);
        let fitted = poly.fit_pipeline(&x, &y).unwrap();
        let result = fitted.transform_pipeline(&x).unwrap();
        assert_eq!(result.shape(), &[2, 6]);
    }

    #[test]
    fn test_degree3_single_feature() {
        // [a] with degree=3, no bias → [a, a², a³]
        let poly = PolynomialFeatures::<f64>::new(3, false, false).unwrap();
        let x = array![[2.0]];
        let out = poly.transform(&x).unwrap();
        assert_eq!(out.shape()[1], 3);
        assert_abs_diff_eq!(out[[0, 0]], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(out[[0, 1]], 4.0, epsilon = 1e-10);
        assert_abs_diff_eq!(out[[0, 2]], 8.0, epsilon = 1e-10);
    }
}
