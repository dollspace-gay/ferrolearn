//! One-hot encoder for categorical numeric features.
//!
//! `fit` learns, for each input column, `categories_[j]` = the **sorted unique
//! set** of values in that column (matching scikit-learn's
//! `OneHotEncoder.categories_`, `_BaseEncoder._fit:99`, `categories_ =
//! _unique(Xi)`). `transform` emits a dense binary matrix where each learned
//! category gets its own output column; the per-feature blocks are concatenated
//! left-to-right (column 0's categories first, then column 1's, …), and a value
//! is one-hot by **category membership** (the value's index within
//! `categories_[j]`), NOT by an assumed contiguous `0..max` integer layout.
//!
//! # Example
//!
//! ```text
//! Input column with the (non-contiguous) categories {2, 5, 9}:
//!   [2, 5, 9]  →  [[1,0,0],[0,1,0],[0,0,1]]   (3 columns, one per unique value)
//! ```
//!
//! # `## REQ status`
//!
//! Binary (R-DEFER-2), translating `sklearn/preprocessing/_encoders.py` (`class OneHotEncoder`
//! `:458`). Design doc: `.design/preprocess/one_hot_encoder.md`. Expected values from the live
//! sklearn 1.5.2 oracle (R-CHAR-3). Consumer: crate re-export (`lib.rs`, grandfathered S5).
//! HONEST (R-HONEST-3): ferrolearn ships a numeric (`F`-input) DENSE encoder whose `categories_`
//! and column layout now match sklearn's `sparse_output=False` output for ANY finite numeric
//! columns (contiguous or not); sparse-by-default output, string/object categories, `drop`,
//! `handle_unknown='ignore'`, infrequent grouping, `inverse_transform`/`get_feature_names_out`,
//! the full ctor surface, the PyO3 binding, and the ferray substrate stay NOT-STARTED.
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (dense one-hot via per-feature category blocks) | SHIPPED | `Transform::transform for FittedOneHotEncoder` zero-fills an `Array2<F>` of width `n_output()` then, for each value, sets `out[[i, offsets[j]+idx]]=1` where `idx` is the value's index in `categories_[j]` (membership), mirroring `_BaseEncoder._transform` (`_encoders.py:206-240`) + the one-hot block expansion. Consumer: crate re-export `lib.rs`. |
//! | REQ-2 (sparse-by-default output) | NOT-STARTED | open prereq blocker #1149. Dense `Array2<F>` only; sklearn defaults `sparse_output=True` → scipy CSR (`:531`,`:748`). |
//! | REQ-3 (categories_ = sorted unique set) | SHIPPED | `Fit::fit` computes `categories_[j]` = per-column values sorted via `partial_cmp` then exact-equality deduped to the sorted-unique set (`_BaseEncoder._fit:99` `categories_=_unique(Xi)`); precomputes `offsets` (prefix sums of `categories_[j].len()`) + `n_output`; rejects 0 rows (`InsufficientSamples`). `categories()` accessor exposes the learned sets. Transform is membership-based (value's index in `categories_[j]`), so non-contiguous integers (`[2,5,9]` → 3 columns, NOT 10) and arbitrary finite floats encode correctly — bit-exact to live sklearn 1.5.2 `sparse_output=False`: `categories_`/`transform`/non-contiguous-headline/offsets guards in `tests/divergence_one_hot_encoder.rs`. Consumer: crate re-export `lib.rs`. SCOPE: numeric `F` input; exact float equality for membership (np.unique semantics — documented); NaN-as-a-category is HANDLED (#2223): NaN sorts LAST + collapses to one category (sklearn `_encode.py:70-74`), a NaN row one-hots its column; string/object input is REQ-3-string (NOT-STARTED, no String path). |
//! | REQ-4 (handle_unknown + set-membership error) | NOT-STARTED | open prereq blocker #1151. `transform` raises `InvalidParameter` ("Found unknown categories …") via `categories_` MEMBERSHIP (no longer a `max+1` comparison) — the default `handle_unknown='error'` ValueError mechanism now MATCHES (`_encoders.py:206-214`), but the `'ignore'`/`'infrequent_if_exist'` modes (`:541`) are still absent. R-DEV-2. |
//! | REQ-5 (drop + infrequent grouping) | NOT-STARTED | open prereq blocker #1152. No `drop` (`:498-516`) / `min_frequency`/`max_categories` (`:566-`). |
//! | REQ-6 (inverse_transform + get_feature_names_out) | SHIPPED | `FittedOneHotEncoder::inverse_transform` reduces each per-feature block `x[:, offsets[j]..offsets[j]+len(categories_[j])]` via **argmax** (numpy first-max-on-ties) to `categories_[j][argmax]`, then errors on an ALL-ZERO block (`block_sum == 0`) with `InvalidParameter` ("Samples can not be inverted when drop=None and handle_unknown='error' because they contain all zeros"), mirroring sklearn's two-step argmax-then-all-zero-check (`_encoders.py:1136-1168`); 0-row → `InsufficientSamples`, `ncols != n_output` → `ShapeMismatch` (`:1100-1104`). Never panics (block slices bounds-checked, R-CODE-2). `FittedOneHotEncoder::get_feature_names_out` emits `format!("x{j}_{cat}")` over `categories_` with default `input_features=["x0",..]` + the `"concat"` combiner (`feature+"_"+str(category)`, `:1217,1224`) → `["x0_2.0","x0_5.0","x0_9.0","x1_0.0","x1_1.0"]`; the float label via `category_label` appends `.0` to whole-valued floats (Python `str(np.float64)`: `2.0`/`-3.0`/`2.5`), `NaN→"nan"`. Live-oracle parity (roundtrip incl. non-contiguous `{2,5,9}`, held-out `[[0,1,0,1,0]]→[[5,0]]`, all-zero/ncols/0-row errors, feature names whole+fractional+negative) in `tests/divergence_one_hot_encoder.rs`. Consumer: crate re-export (`lib.rs:141`). DOCUMENTED DIVERGENCE (R-HONEST-3): the float label uses Rust `Display` for non-whole values, so it diverges from Python's scientific notation at `|v|>=1e16` / `0<|v|<1e-4` (`1e+20`/`1e-07` vs full decimal) — not a plausible category. STILL NOT-STARTED within REQ-6: the `input_features=`/`feature_name_combiner=` params (`:1192,1222`) and the drop-aware / `handle_unknown='ignore'` inverse (None for all-zeros, `:1141-1158`). |
//! | REQ-7 (ctor + dtype + _parameter_constraints) | NOT-STARTED | open prereq blocker #1154. `new()` takes no params/validates nothing (`:728-762`). |
//! | REQ-8 (PyO3 binding) | NOT-STARTED | open prereq blocker #1155. No `ferrolearn-python` registration (R-DEFER-1). |
//! | REQ-9 (ferray substrate) | NOT-STARTED | open prereq blocker #1156. `ndarray::Array2`, not `ferray-core` (R-SUBSTRATE-1/2). |

use ferrolearn_core::error::FerroError;
use ferrolearn_core::traits::{Fit, FitTransform, Transform};
use ndarray::Array2;
use num_traits::Float;
use std::cmp::Ordering;

// ---------------------------------------------------------------------------
// OneHotEncoder (unfitted)
// ---------------------------------------------------------------------------

/// An unfitted one-hot encoder for multi-column numeric categorical data.
///
/// Input: `Array2<F>` where each column contains the (finite) numeric category
/// values. Calling [`Fit::fit`] learns, per column, the **sorted unique set** of
/// values (`categories_`) and returns a [`FittedOneHotEncoder`]. The output of
/// [`Transform::transform`] is a dense binary matrix with one column per learned
/// category, the per-feature blocks concatenated left-to-right.
///
/// # Examples
///
/// ```
/// use ferrolearn_preprocess::OneHotEncoder;
/// use ferrolearn_core::traits::{Fit, Transform};
/// use ndarray::array;
///
/// let enc = OneHotEncoder::<f64>::new();
/// // Non-contiguous categories {2, 5, 9} in column 0, {0, 1} in column 1.
/// let x = array![[2.0_f64, 0.0], [5.0, 1.0], [9.0, 0.0], [5.0, 1.0]];
/// let fitted = enc.fit(&x, &()).unwrap();
/// assert_eq!(fitted.categories(), &[vec![2.0, 5.0, 9.0], vec![0.0, 1.0]]);
/// let encoded = fitted.transform(&x).unwrap();
/// assert_eq!(encoded.ncols(), 5); // 3 + 2 category columns
/// ```
#[derive(Debug, Clone)]
pub struct OneHotEncoder<F> {
    _marker: std::marker::PhantomData<F>,
}

impl<F: Float + Send + Sync + 'static> OneHotEncoder<F> {
    /// Create a new `OneHotEncoder`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            _marker: std::marker::PhantomData,
        }
    }
}

impl<F: Float + Send + Sync + 'static> Default for OneHotEncoder<F> {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// FittedOneHotEncoder
// ---------------------------------------------------------------------------

/// A fitted one-hot encoder holding the sorted-unique category set per input
/// column, plus the precomputed output-column layout.
///
/// Created by calling [`Fit::fit`] on a [`OneHotEncoder`]. Mirrors
/// scikit-learn's `OneHotEncoder.categories_` (a list of arrays of the actual
/// sorted-unique values, `_BaseEncoder._fit:99`).
#[derive(Debug, Clone)]
pub struct FittedOneHotEncoder<F> {
    /// Per-column sorted-unique category values (`categories_`). `categories_[j]`
    /// is the sorted set of distinct values seen in input column `j`; its length
    /// is the number of output columns devoted to that feature's block.
    pub(crate) categories_: Vec<Vec<F>>,
    /// Per-column output-block start offsets (prefix sums of
    /// `categories_[j].len()`). Output column `offsets[j] + idx` is the one-hot
    /// bit for `categories_[j][idx]`. Has length `categories_.len()`.
    pub(crate) offsets: Vec<usize>,
    /// Total number of output columns (`Σ categories_[j].len()`).
    pub(crate) n_output: usize,
}

impl<F: Float + Send + Sync + 'static> FittedOneHotEncoder<F> {
    /// Return the learned sorted-unique category set for each input column
    /// (`categories_`).
    ///
    /// `categories()[j][idx]` is the value encoded by output column
    /// `offsets[j] + idx`. Mirrors scikit-learn's `OneHotEncoder.categories_`.
    #[must_use]
    pub fn categories(&self) -> &[Vec<F>] {
        &self.categories_
    }

    /// Return the number of distinct categories for each input feature column,
    /// i.e. the width of each per-feature one-hot block.
    #[must_use]
    pub fn n_categories(&self) -> Vec<usize> {
        self.categories_.iter().map(Vec::len).collect()
    }

    /// Return the number of input feature columns.
    #[must_use]
    pub fn n_features(&self) -> usize {
        self.categories_.len()
    }

    /// Return the total number of output columns (`Σ categories_[j].len()`).
    #[must_use]
    pub fn n_output_features(&self) -> usize {
        self.n_output
    }

    /// Invert a one-hot encoded matrix back to the original category values.
    ///
    /// For each input feature `j` the per-feature block
    /// `x[:, offsets[j] .. offsets[j] + categories_[j].len()]` is reduced to a
    /// single category via **argmax** (the index of the maximum value in the
    /// block, first-max on ties — numpy `argmax` semantics), and the original
    /// value `categories_[j][argmax]` is written to `out[[i, j]]`. This mirrors
    /// scikit-learn's `OneHotEncoder.inverse_transform`
    /// (`sklearn/preprocessing/_encoders.py:1136-1139`):
    /// `labels = sub.argmax(axis=1); X_tr[:, i] = cats[labels]`.
    ///
    /// After the argmax, an **all-zero block** (a row whose per-feature block
    /// sums to zero) cannot be inverted. With no `drop` and the default
    /// `handle_unknown='error'` (the only mode ferrolearn ships — REQ-4/5), this
    /// is an error, matching sklearn's
    /// `ValueError("Samples [...] can not be inverted when drop=None and
    /// handle_unknown='error' because they contain all zeros")`
    /// (`_encoders.py:1160-1168`). A proper one-hot row from
    /// [`Transform::transform`] has exactly one `1` per block, so argmax always
    /// finds it and the block sum is never zero.
    ///
    /// # Errors
    ///
    /// - [`FerroError::InsufficientSamples`] if `x` has zero rows (sklearn
    ///   `check_array` requires a minimum of 1 sample).
    /// - [`FerroError::ShapeMismatch`] if `x.ncols() != n_output` (sklearn's
    ///   "Shape of the passed X data is not correct" `ValueError`,
    ///   `_encoders.py:1100-1104`).
    /// - [`FerroError::InvalidParameter`] if any per-feature block is all-zero
    ///   (the sklearn all-zeros `ValueError`, `_encoders.py:1164-1168`).
    ///
    /// Never panics: every block slice is bounds-checked (R-CODE-2).
    pub fn inverse_transform(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let n_samples = x.nrows();
        if n_samples == 0 {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "FittedOneHotEncoder::inverse_transform".into(),
            });
        }
        // sklearn `inverse_transform` -> `check_array(X, accept_sparse="csr")`
        // (`_encoders.py:1092`) with the DEFAULT `force_all_finite=True`, so a
        // NaN or +/-inf cell in the one-hot matrix raises BEFORE the argmax
        // (#2224). A valid one-hot row is all 0/1 (finite); a non-finite cell is
        // invalid input.
        if x.iter().any(|v| !v.is_finite()) {
            return Err(FerroError::InvalidParameter {
                name: "X".into(),
                reason: "Input X contains NaN or infinity.".into(),
            });
        }
        if x.ncols() != self.n_output {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_samples, self.n_output],
                actual: vec![n_samples, x.ncols()],
                context: "FittedOneHotEncoder::inverse_transform".into(),
            });
        }

        let n_features = self.categories_.len();
        let mut out = Array2::zeros((n_samples, n_features));

        for j in 0..n_features {
            let cats = &self.categories_[j];
            let block_len = cats.len();
            // A unique-category column has an empty `categories_[j]` only if the
            // training column was empty, which `fit` rejects (0 rows); a
            // 1-category column has block_len == 1. Guard anyway (R-CODE-2).
            if block_len == 0 {
                continue;
            }
            let offset = self.offsets[j];
            for i in 0..n_samples {
                // Argmax over the per-feature block (numpy `argmax`: index of the
                // maximum, FIRST on ties). Track the block sum to detect the
                // all-zero (uninvertible) case separately, mirroring sklearn's
                // two-step argmax-then-all-zero-check (`_encoders.py:1138-1168`).
                let mut argmax: usize = 0;
                let mut max_val = x[[i, offset]];
                let mut block_sum = max_val;
                for k in 1..block_len {
                    let v = x[[i, offset + k]];
                    block_sum = block_sum + v;
                    if v > max_val {
                        max_val = v;
                        argmax = k;
                    }
                }
                // All-zero block: cannot be inverted with drop=None and
                // handle_unknown='error' (sklearn `_encoders.py:1164-1168`). We
                // have no drop/ignore mode (REQ-4/5), so this is always an error.
                if block_sum == F::zero() {
                    return Err(FerroError::InvalidParameter {
                        name: "X".into(),
                        reason: "Samples can not be inverted when drop=None and \
                                 handle_unknown='error' because they contain all zeros"
                            .into(),
                    });
                }
                out[[i, j]] = cats[argmax];
            }
        }

        Ok(out)
    }

    /// Return the output feature names, one per output column.
    ///
    /// For each input feature `j`, for each category `c` in `categories_[j]`,
    /// emits `format!("x{j}_{c}")` where `c` is rendered to match Python's
    /// `str(np.float64(c))`. This mirrors scikit-learn's
    /// `OneHotEncoder.get_feature_names_out` with the default `input_features`
    /// (`["x0", "x1", ...]`) and the `"concat"` name combiner
    /// (`feature + "_" + str(category)`, `_encoders.py:1217,1224`). For the
    /// whole-number fixture `[[2,0],[5,1],[9,0],[5,1]]` this yields
    /// `["x0_2.0", "x0_5.0", "x0_9.0", "x1_0.0", "x1_1.0"]`.
    ///
    /// # Float-rendering divergence (HONEST, R-HONEST-3)
    ///
    /// The category is rendered via [`Self::category_label`], which appends `.0`
    /// to integer-valued floats (`2.0 → "2.0"`, `-3.0 → "-3.0"`, matching
    /// Python) and uses Rust's shortest round-trip `Display` otherwise
    /// (`2.5 → "2.5"`). For category values in the usual categorical range
    /// (small whole or fractional numbers) this is byte-identical to Python.
    /// It DIVERGES for extreme magnitudes: Python's `repr`/`str` switches to
    /// scientific notation at `|v| >= 1e16` and `0 < |v| < 1e-4`
    /// (`1e+20`, `1e-07`), while Rust's `Display` prints the full decimal
    /// (`100000000000000000000`, `0.0000001`). Such values are not plausible
    /// one-hot categories; the divergence is documented rather than papered over.
    /// `NaN` renders as `"nan"` (matching Python's `str(nan)`).
    #[must_use]
    pub fn get_feature_names_out(&self) -> Vec<String> {
        let mut names = Vec::with_capacity(self.n_output);
        for (j, cats) in self.categories_.iter().enumerate() {
            for &c in cats {
                names.push(format!("x{j}_{}", Self::category_label(c)));
            }
        }
        names
    }

    /// Render a category value to a string matching Python's `str(np.float64(v))`
    /// for the categorical-value range (see [`Self::get_feature_names_out`] for
    /// the documented extreme-magnitude divergence).
    ///
    /// Python's `str(float)` always shows a decimal point for whole floats
    /// (`2.0`, not `2`), so an integer-valued finite float gets a `.0` suffix;
    /// otherwise Rust's shortest round-trip `Display` is used. `NaN → "nan"`.
    fn category_label(v: F) -> String {
        let Some(f) = v.to_f64() else {
            return "nan".to_string();
        };
        if f.is_nan() {
            return "nan".to_string();
        }
        if f.is_finite() && f == f.trunc() {
            // Whole-valued finite float: Python prints e.g. "2.0", "-3.0".
            format!("{f:.1}")
        } else {
            // Fractional or non-finite: shortest round-trip Display ("2.5").
            format!("{f}")
        }
    }
}

// ---------------------------------------------------------------------------
// Trait implementations
// ---------------------------------------------------------------------------

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, ()> for OneHotEncoder<F> {
    type Fitted = FittedOneHotEncoder<F>;
    type Error = FerroError;

    /// Fit the encoder by learning the **sorted-unique category set** per column.
    ///
    /// For each input column `j`, `categories_[j]` is the distinct values of that
    /// column, sorted ascending via `partial_cmp` and deduped by **exact
    /// equality** — mirroring scikit-learn's `categories_ = _unique(Xi)`
    /// (`sklearn/preprocessing/_encoders.py:99`, `np.unique` per column).
    /// The output-column layout (`offsets`, `n_output`) is precomputed as the
    /// prefix sums / total of the per-column category counts.
    ///
    /// Exact float equality is what `np.unique` does, so two values that differ
    /// by an ULP are distinct categories here, exactly as in sklearn.
    ///
    /// # NaN handling (#2223)
    ///
    /// `NaN` is treated as a valid category, matching sklearn's `_unique_np`
    /// (`_encode.py:70-74`): it sorts LAST and a run of duplicate NaNs collapses
    /// to a SINGLE sorted-last category (the sort orders `NaN` after every finite
    /// value and `dedup_by` collapses consecutive NaNs, since `NaN != NaN`). A
    /// NaN cell at `transform` then one-hots that trailing category. `fit` never
    /// panics (R-CODE-2).
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::InsufficientSamples`] if the input has zero rows
    /// (matching sklearn's `check_array` minimum-of-1-sample requirement).
    fn fit(&self, x: &Array2<F>, _y: &()) -> Result<FittedOneHotEncoder<F>, FerroError> {
        let n_samples = x.nrows();
        if n_samples == 0 {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "OneHotEncoder::fit".into(),
            });
        }

        let n_features = x.ncols();
        let mut categories_: Vec<Vec<F>> = Vec::with_capacity(n_features);
        let mut offsets: Vec<usize> = Vec::with_capacity(n_features);
        let mut n_output: usize = 0;

        for j in 0..n_features {
            // Collect this column's values, sort ascending (sklearn `np.unique`
            // sorts), then dedup by EXACT equality to the sorted-unique set.
            let mut col: Vec<F> = x.column(j).iter().copied().collect();
            // Sort ascending with NaN LAST (sklearn `_unique_np` keeps any NaN at
            // the end, `_encode.py:70-74`); `partial_cmp` alone returns None for
            // NaN and would leave it unmoved (#2223).
            col.sort_by(|a, b| match (a.is_nan(), b.is_nan()) {
                (true, true) => Ordering::Equal,
                (true, false) => Ordering::Greater,
                (false, true) => Ordering::Less,
                (false, false) => a.partial_cmp(b).unwrap_or(Ordering::Equal),
            });
            // Dedup to the sorted-unique set: consecutive EXACT-equal values
            // collapse (an ULP-apart pair stays distinct, like `np.unique`), AND
            // consecutive NaNs collapse to ONE — `dedup` alone would keep every
            // NaN (`NaN != NaN`), so handle it explicitly (sklearn collapses the
            // trailing NaN run to a single sorted-last category, #2223).
            col.dedup_by(|a, b| a == b || (a.is_nan() && b.is_nan()));

            offsets.push(n_output);
            n_output += col.len();
            categories_.push(col);
        }

        Ok(FittedOneHotEncoder {
            categories_,
            offsets,
            n_output,
        })
    }
}

impl<F: Float + Send + Sync + 'static> Transform<Array2<F>> for FittedOneHotEncoder<F> {
    type Output = Array2<F>;
    type Error = FerroError;

    /// Transform numeric categorical data into a dense one-hot encoded matrix.
    ///
    /// Each value is one-hot by **category membership**: for input column `j` the
    /// value `x[[i, j]]` is matched (by exact equality) against `categories_[j]`,
    /// and the bit at output column `offsets[j] + idx` is set, where `idx` is the
    /// value's position in the sorted-unique set. The per-feature one-hot blocks
    /// are concatenated left-to-right, matching scikit-learn's
    /// `OneHotEncoder(sparse_output=False)` output column layout
    /// (`_BaseEncoder._transform`, `_encoders.py:206-240`).
    ///
    /// A value not present in `categories_[j]` is an **unknown category**;
    /// matching sklearn's default `handle_unknown='error'` it returns an error
    /// (sklearn raises `ValueError("Found unknown categories … during
    /// transform")`, `_encoders.py:209-214`).
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of columns does not
    /// match the number of features seen during fitting.
    ///
    /// Returns [`FerroError::InvalidParameter`] if any value is an unknown
    /// category (not in the learned `categories_[j]` set).
    fn transform(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let n_features = self.categories_.len();
        if x.ncols() != n_features {
            return Err(FerroError::ShapeMismatch {
                expected: vec![x.nrows(), n_features],
                actual: vec![x.nrows(), x.ncols()],
                context: "FittedOneHotEncoder::transform".into(),
            });
        }

        let n_samples = x.nrows();
        let mut out = Array2::zeros((n_samples, self.n_output));

        for j in 0..n_features {
            let cats = &self.categories_[j];
            let offset = self.offsets[j];
            for i in 0..n_samples {
                let value = x[[i, j]];
                // Membership lookup: find the value's index in the sorted-unique
                // `categories_[j]` by EXACT equality (np.unique / `_encode`
                // semantics). A small linear scan over the per-feature category
                // set — bounds-safe (no unchecked indexing; R-CODE-2).
                match cats
                    .iter()
                    .position(|&c| c == value || (c.is_nan() && value.is_nan()))
                {
                    Some(idx) => out[[i, offset + idx]] = F::one(),
                    None => {
                        // Unknown category under handle_unknown='error' (the
                        // sklearn default): ValueError "Found unknown categories
                        // … during transform" (`_encoders.py:209-214`). Format the
                        // value via its `Display` impl when available; `F: Float`
                        // is not `Display`, so report position + column.
                        let v = value.to_f64();
                        let shown = match v {
                            Some(f) => format!("[{f}]"),
                            None => "[<non-finite>]".to_string(),
                        };
                        return Err(FerroError::InvalidParameter {
                            name: format!("x[{i},{j}]"),
                            reason: format!(
                                "Found unknown categories {shown} in column {j} during transform"
                            ),
                        });
                    }
                }
            }
        }

        Ok(out)
    }
}

/// Implement `Transform` on the unfitted encoder to satisfy the `FitTransform: Transform`
/// supertrait bound. Calling `transform` on an unfitted encoder always returns an error.
impl<F: Float + Send + Sync + 'static> Transform<Array2<F>> for OneHotEncoder<F> {
    type Output = Array2<F>;
    type Error = FerroError;

    /// Always returns an error — the encoder must be fitted first.
    ///
    /// Use [`Fit::fit`] to produce a [`FittedOneHotEncoder`], then call
    /// [`Transform::transform`] on that.
    fn transform(&self, _x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        Err(FerroError::InvalidParameter {
            name: "OneHotEncoder".into(),
            reason: "encoder must be fitted before calling transform; use fit() first".into(),
        })
    }
}

impl<F: Float + Send + Sync + 'static> FitTransform<Array2<F>> for OneHotEncoder<F> {
    type FitError = FerroError;

    /// Fit the encoder on `x` and return the one-hot encoded output in one step.
    ///
    /// # Errors
    ///
    /// Returns an error if fitting or transformation fails.
    fn fit_transform(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let fitted = self.fit(x, &())?;
        fitted.transform(x)
    }
}

/// Convenience: encode a 1-D array of numeric categories.
///
/// This wraps the input in a single-column `Array2<F>` and returns the encoded
/// result with one-hot columns for that single feature, matching the membership
/// encoding of [`Transform::transform`].
impl<F: Float + Send + Sync + 'static> FittedOneHotEncoder<F> {
    /// Transform a 1-D slice of numeric category values.
    ///
    /// # Errors
    ///
    /// Returns an error if the encoder was fitted on more than one column, or if
    /// any value is an unknown category (not in the learned `categories_[0]`).
    pub fn transform_1d(&self, x: &[F]) -> Result<Array2<F>, FerroError> {
        if self.categories_.len() != 1 {
            return Err(FerroError::InvalidParameter {
                name: "transform_1d".into(),
                reason: "encoder was fitted on more than one column; use transform instead".into(),
            });
        }
        let col = Array2::from_shape_vec((x.len(), 1), x.to_vec()).map_err(|e| {
            FerroError::InvalidParameter {
                name: "x".into(),
                reason: e.to_string(),
            }
        })?;
        self.transform(&col)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_one_hot_single_column() {
        let enc = OneHotEncoder::<f64>::new();
        let x = array![[0.0_f64], [1.0], [2.0]];
        let fitted = enc.fit(&x, &()).unwrap();
        assert_eq!(fitted.categories(), &[vec![0.0, 1.0, 2.0]]);
        assert_eq!(fitted.n_categories(), vec![3]);
        assert_eq!(fitted.n_output_features(), 3);

        let out = fitted.transform(&x).unwrap();
        assert_eq!(out.shape(), &[3, 3]);
        // Row 0: category 0 → [1, 0, 0]
        assert_eq!(out[[0, 0]], 1.0);
        assert_eq!(out[[0, 1]], 0.0);
        assert_eq!(out[[0, 2]], 0.0);
        // Row 1: category 1 → [0, 1, 0]
        assert_eq!(out[[1, 0]], 0.0);
        assert_eq!(out[[1, 1]], 1.0);
        assert_eq!(out[[1, 2]], 0.0);
        // Row 2: category 2 → [0, 0, 1]
        assert_eq!(out[[2, 0]], 0.0);
        assert_eq!(out[[2, 1]], 0.0);
        assert_eq!(out[[2, 2]], 1.0);
    }

    #[test]
    fn test_one_hot_multi_column() {
        let enc = OneHotEncoder::<f64>::new();
        // Two columns: col0 has 3 categories, col1 has 2 categories
        let x = array![[0.0_f64, 0.0], [1.0, 1.0], [2.0, 0.0]];
        let fitted = enc.fit(&x, &()).unwrap();
        assert_eq!(fitted.categories(), &[vec![0.0, 1.0, 2.0], vec![0.0, 1.0]]);
        assert_eq!(fitted.n_categories(), vec![3, 2]);
        assert_eq!(fitted.n_output_features(), 5);

        let out = fitted.transform(&x).unwrap();
        assert_eq!(out.shape(), &[3, 5]);
        // Row 0: (0, 0) → [1,0,0, 1,0]
        assert_eq!(out.row(0).to_vec(), vec![1.0, 0.0, 0.0, 1.0, 0.0]);
        // Row 1: (1, 1) → [0,1,0, 0,1]
        assert_eq!(out.row(1).to_vec(), vec![0.0, 1.0, 0.0, 0.0, 1.0]);
        // Row 2: (2, 0) → [0,0,1, 1,0]
        assert_eq!(out.row(2).to_vec(), vec![0.0, 0.0, 1.0, 1.0, 0.0]);
    }

    #[test]
    fn test_non_contiguous_single_column() {
        // The REQ-3 headline: non-contiguous integers {2,5,9} must yield 3
        // category columns (one per unique value), NOT max+1 == 10.
        let enc = OneHotEncoder::<f64>::new();
        let x = array![[2.0_f64], [5.0], [9.0]];
        let fitted = enc.fit(&x, &()).unwrap();
        assert_eq!(fitted.categories(), &[vec![2.0, 5.0, 9.0]]);
        assert_eq!(fitted.n_output_features(), 3);
        let out = fitted.transform(&x).unwrap();
        assert_eq!(out.shape(), &[3, 3]);
        assert_eq!(out.row(0).to_vec(), vec![1.0, 0.0, 0.0]);
        assert_eq!(out.row(1).to_vec(), vec![0.0, 1.0, 0.0]);
        assert_eq!(out.row(2).to_vec(), vec![0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_unknown_category_error() {
        let enc = OneHotEncoder::<f64>::new();
        let x_train = array![[0.0_f64], [1.0]];
        let fitted = enc.fit(&x_train, &()).unwrap();
        // Value 2.0 was not seen during fitting → unknown category.
        let x_bad = array![[2.0_f64]];
        assert!(fitted.transform(&x_bad).is_err());
    }

    #[test]
    fn test_fit_transform_equivalence() {
        let enc = OneHotEncoder::<f64>::new();
        let x = array![[0.0_f64, 1.0], [1.0, 0.0], [2.0, 1.0]];
        let via_fit_transform: Array2<f64> = enc.fit_transform(&x).unwrap();
        let fitted = enc.fit(&x, &()).unwrap();
        let via_separate = fitted.transform(&x).unwrap();
        for (a, b) in via_fit_transform.iter().zip(via_separate.iter()) {
            assert!((a - b).abs() < 1e-15);
        }
    }

    #[test]
    fn test_shape_mismatch_error() {
        let enc = OneHotEncoder::<f64>::new();
        let x_train = array![[0.0_f64, 1.0], [1.0, 0.0]];
        let fitted = enc.fit(&x_train, &()).unwrap();
        let x_bad = array![[0.0_f64]];
        assert!(fitted.transform(&x_bad).is_err());
    }
}
