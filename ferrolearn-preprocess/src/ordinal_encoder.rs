//! Ordinal encoder: map string categories to integer indices.
//!
//! Each column's categories are mapped to integers `0, 1, 2, ...` in
//! **lexicographic order** (matching scikit-learn's `OrdinalEncoder`).
//! Unknown categories seen during `transform` produce an error by default
//! (`handle_unknown='error'`); with `handle_unknown='use_encoded_value'` they
//! are instead encoded as a configurable `unknown_value` sentinel
//! (matching scikit-learn's `OrdinalEncoder`).
//!
//! # `## REQ status`
//!
//! Binary (R-DEFER-2), translating `sklearn/preprocessing/_encoders.py` (`class OrdinalEncoder`
//! `:1235`). Design doc: `.design/preprocess/ordinal_encoder.md`. Expected values from the live
//! sklearn 1.5.2 oracle (R-CHAR-3). Consumer: crate re-export (`lib.rs:121`, grandfathered S5).
//! HONEST (R-HONEST-3): a FAITHFUL String-only ordinal encoder — `categories_`=sorted-unique and
//! the ordinal VALUES match sklearn bit-for-bit on the string path; the output container is now
//! `Array2<f64>` (sklearn's `dtype=np.float64` default, `:1262`); remaining divergences are
//! String-only input, the absent configurable `dtype` param, and the rest of the param/feature
//! surface.
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (string fit → sorted-unique categories_) | SHIPPED | `Fit::fit` per column → `categories_`=sorted-unique (`Vec<String>::sort`, lexicographic) + index map; rejects 0 rows (`InsufficientSamples`, matches sklearn `check_array`). Mirrors `_BaseEncoder._fit` `categories_=_unique(Xi)` (`_encoders.py:99`). Critic-verified vs live oracle: `green_value_match_and_categories` (`[['bird','cat','dog'],['large','medium','small']]`), `green_lexicographic_sort_matches_np_unique` + `green_non_ascii_codepoint_order` (== `np.unique`), `green_empty_fit_rejected_matches_sklearn`. Consumer: re-export `lib.rs:121`. |
//! | REQ-2 (transform + fit_transform, ordinal values + unknown rejection) | SHIPPED | `Transform::transform` maps category→ordinal index (now cast to `f64` via `ordinal_index_to_f64`), unknown → `InvalidParameter` (matches `handle_unknown='error'` default `ValueError`), ncols-mismatch → `ShapeMismatch`. The unknown/ncols-mismatch LOGIC is byte-for-byte UNCHANGED by the dtype fix. Critic-verified: ordinal VALUES `[[1.,2.],[2.,0.],[1.,1.],[0.,2.]]` == live oracle, `green_unknown_category_rejected`, `green_fit_transform_equals_oracle`. Consumer: re-export `lib.rs:142`. |
//! | REQ-3 (output dtype float64) | SHIPPED | `Transform::Output = Array2<f64>` on BOTH `Transform` impls (`FittedOrdinalEncoder` + the unfitted `OrdinalEncoder` shim) and `FitTransform::fit_transform`; each cell is the ordinal index cast via `ordinal_index_to_f64` (`idx as f64`, lossless < 2^53), matching sklearn's default `dtype=np.float64` output container (`_encoders.py:1262`, `transform` casts `X_int.astype(self.dtype)`). The REQ-1/REQ-2 fit + unknown-rejection LOGIC is unchanged. Critic-verified vs live oracle: `green_fit_transform_f64_oracle` (multi-feature f64 matrix), `green_exact_integer_index_to_f64` (index 10 → `10.0`), plus the value guards over `Array2<f64>`. A CONFIGURABLE non-float64 output dtype (`int32` etc.) is a FOLLOW-ON (blocker #1158 remains open for the `dtype` ctor param); ferrolearn's output is fixed to sklearn's `float64` DEFAULT. This unblocks REQ-5's float `unknown_value` sentinel. Consumer: crate re-export `lib.rs:142`. |
//! | REQ-4 (numeric/mixed-dtype input) | NOT-STARTED | open prereq blocker #1159. `Array2<String>`-only; sklearn accepts int/str/object (`np.unique` numeric sort). |
//! | REQ-5 (handle_unknown='use_encoded_value' + unknown_value) | SHIPPED | `HandleUnknown` enum `{ Error, UseEncodedValue }` (default `Error`) + `unknown_value: Option<f64>` on `OrdinalEncoder`, threaded into `FittedOrdinalEncoder` via `with_handle_unknown`/`with_unknown_value` builders. `Fit::fit` runs the 3 sklearn validations (`_encoders.py:1473-1526`) AFTER the unchanged `categories_` compute, mapping sklearn's `TypeError`/`ValueError` → `FerroError::InvalidParameter`: (a) `UseEncodedValue` && `unknown_value is None` (sklearn `:1481` `not isinstance(.,Integral)` TypeError); (b) `Error` && `unknown_value is Some` (sklearn `:1488` TypeError); (c) `UseEncodedValue` && non-nan integer `v` with `0 <= v < max_cardinality` (sklearn `:1518-1526` ValueError collision). `Transform::transform` branches unknown categories: `UseEncodedValue` → write `unknown_value` (incl. nan) (sklearn `:1591` `X_trans[~X_mask] = self.unknown_value`); `Error` → `InvalidParameter` (the SHIPPED REQ-2 default, UNCHANGED). Seen categories still map to `idx as f64` (UNCHANGED). NEVER panics (R-CODE-2). Critic-verified vs live sklearn 1.5.2 oracle: `green_use_encoded_value_minus_one`, `green_use_encoded_value_nan`, `green_use_encoded_value_multifeature`, `red_uev_requires_unknown_value`, `red_error_mode_forbids_unknown_value`, `red_unknown_value_collision_in_range`, `green_unknown_value_negative_or_oob_or_nan_ok`, `green_error_mode_unknown_still_rejected` (`tests/divergence_ordinal_encoder.rs`). Configurable `dtype`/`encoded_missing_value` interplay stays OUT OF SCOPE (REQ-3/REQ-6). Consumer: crate re-export `lib.rs:142`. |
//! | REQ-6 (encoded_missing_value / NaN) | NOT-STARTED | open prereq blocker #1161. No missing-value concept (`:1283`). |
//! | REQ-7 (explicit categories param) | SHIPPED | `Categories` enum `{ Auto, Explicit(Vec<Vec<String>>) }` (default `Auto`) + `#[must_use] OrdinalEncoder::with_categories(Vec<Vec<String>>)` builder + `categories_param()` getter (named to avoid colliding with `FittedOrdinalEncoder::categories`). `Fit::fit` branches on the param AFTER the 0-row guard: `Auto` → the SHIPPED REQ-1 sorted-unique compute (UNCHANGED); `Explicit(lists)` → use each `lists[j]` AS-GIVEN for `categories_[j]` (GIVEN order, NOT re-sorted) + the index map in that order, mirroring sklearn `_encoders.py:114` `cats = np.array(self.categories[i])`. Validations match `_BaseEncoder._fit`: list-count ≠ n_features → `ShapeMismatch` ("Shape mismatch: if categories is an array, it has to be of shape (n_features,)." `:85-89`); an EMPTY list → `InvalidParameter` (sklearn indexes `cats[0]` -> IndexError in both modes, `:114-117`, #2229); a list with duplicate elements → `InvalidParameter` ("In column {j}, the predefined categories contain duplicate elements." `:136-141`); under [`HandleUnknown::Error`] (default) a data value not in its column's list → `InvalidParameter` ("Found unknown categories [{v}] in column {j} during fit" `:153-160`), while under [`HandleUnknown::UseEncodedValue`] this fit-time subset check is SKIPPED (out-of-set data is encoded to `unknown_value` at transform). The REQ-5 unknown_value validations still apply (the `max_cardinality` collision check now keys off the explicit list lengths). `Transform`/`inverse_transform`/`categories()`/`get_feature_names_out` are UNCHANGED — they already read `categories_`/`category_to_index`, which now reflect the explicit given-order set. NEVER panics (R-CODE-2). Critic-verified vs live sklearn 1.5.2 oracle (`tests/divergence_ordinal_encoder.rs`): `green_explicit_given_order_not_sorted`, `green_explicit_unsorted_accepted`, `red_explicit_error_mode_data_not_in_cats_fits_err`, `green_explicit_use_encoded_value_out_of_set_ok`, `red_explicit_n_features_mismatch`, `green_explicit_multifeature_each_own_order`, `red_explicit_duplicate_categories`, `green_explicit_inverse_roundtrip_given_order`, `green_explicit_auto_still_default`. Consumer: crate re-export (`lib.rs:142`, `Categories` re-exported). Configurable numeric/`bytes` categories + the nan-last rule stay OUT OF SCOPE (String-only path, REQ-4/REQ-6). |
//! | REQ-8 (min_frequency/max_categories infrequent) | SHIPPED | #1163: `OrdinalEncoder::with_min_frequency`/`with_max_categories` (+`min_frequency()`/`max_categories()` getters) add the integer-count infrequent thresholds (`_encoders.py:1289-1315`). The OrdinalEncoder ANALOG of the SHIPPED OneHotEncoder REQ-5b (`one_hot_encoder.rs`): the SAME `_identify_infrequent` algorithm (reused as `identify_infrequent` + `build_infrequent_map`, mirroring `_BaseEncoder._identify_infrequent` `:275-318` + `_default_to_infrequent_mappings` `:373-400`: min_frequency `count < min_freq` FIRST, then max_categories keeps top `max_categories-1` by count via a STABLE argsort over the full count array — ties favor the LARGER index; `max_categories==1` → all infrequent), but the infrequent categories collapse to a single shared ORDINAL CODE `n_frequent` (NOT a one-hot column): frequent categories keep codes `0..n_frequent` in their original sorted order, every infrequent category emits `n_frequent`. `Fit::fit` runs the `_parameter_constraints` check FIRST (`min_frequency`/`max_categories` `Some(0)` → `InvalidParameter` "must be an int in the range [1, inf)", BEFORE the data, `Interval(Integral,1,None)`), then (after the SHIPPED `categories_` compute, UNCHANGED — `categories_` keeps ALL categories) builds per-feature `infrequent_indices_`/`infrequent_map`/`n_frequent` from the fit-data category counts. `FittedOrdinalEncoder::infrequent_categories()` exposes the infrequent VALUES per feature (`infrequent_categories_`, `:255-262`). `Transform` routes a found category index through `infrequent_map[j]` then casts to f64 (frequent → own code, infrequent → shared trailing code; `_map_infrequent_categories`, `:402-452`); with grouping DISABLED the map is the identity so REQ-2 is UNCHANGED. `inverse_transform`: a code `< n_frequent` → the frequent category at that remapped slot (via the frequent-only category list); a code `== n_frequent` (exact float equality on the raw label) → the REAL String `"infrequent_sklearn"` (`:1644`,`:1675-1677`) — representable, UNLIKE OneHotEncoder's NaN proxy; the truncate+wrap numpy index logic applies over the frequent-only list (SHIPPED REQ-9 path UNCHANGED when disabled). The `unknown_value` collision check now keys off the EFFECTIVE code count `n_frequent + 1` (verified live: `min_frequency=2` over 4 cats → 3 codes → `unknown_value=3` accepted, `=2` collides). `get_feature_names_out` is UNCHANGED (OrdinalEncoder is one-to-one — infrequent does NOT add columns). NEVER panics (R-CODE-2). Critic-verified vs live sklearn 1.5.2 oracle (`tests/divergence_ordinal_encoder.rs`): `req8_min_frequency_two_categories_transform_inverse`, `req8_max_categories_keeps_top_k_minus_one`, `req8_max_categories_tiebreak_favors_larger_index`, `req8_both_set_multifeature_some_without_infrequent`, `req8_zero_thresholds_rejected`, `req8_infrequent_plus_use_encoded_value_distinct_codes`, `req8_unknown_value_collision_uses_effective_code_count`, `req8_disabled_default_unchanged`, `req8_inverse_infrequent_non_roundtrip`. Consumer: crate re-export `lib.rs:142`. STILL NOT-STARTED (R-HONEST-3): the FLOAT-fraction `min_frequency` (`:1296-1297`,`:297-299`) and the explicit-`categories`+infrequent interaction stay unimplemented. |
//! | REQ-9 (inverse_transform) | SHIPPED | `FittedOrdinalEncoder::inverse_transform(&Array2<f64>) -> Array2<String>` reuses the SHIPPED `categories_` (REQ-1): each cell is an ordinal index into `categories[j]`, mirroring sklearn `X_tr[:, i] = self.categories_[i][labels]` (`_encoders.py:1595-1679`). Validates the index BEFORE lookup (no panic, R-CODE-2): an exact non-negative integer in `[0, len)` → `categories[j][index].clone()`; 0-row → `InsufficientSamples` (symmetry with the #2220 transform guard); ncols-mismatch → `ShapeMismatch` (sklearn `:1619`). FAITHFUL to numpy: mirrors `labels.astype("int64")` (truncate toward zero, Rust `as i64`) + numpy fancy indexing (negative WRAP, `-1.0` → last category, `-2.0` → `len-2`), raising only once the wrapped index leaves `[0, len)` (`_encoders.py:1664`,`:1679`). Non-finite (NaN/±inf) → `InvalidParameter` (sklearn IndexError/ValueError; guarded because Rust `f64 as i64` saturates NaN→0). Critic-verified vs live sklearn 1.5.2 oracle: `green_inverse_roundtrip_multifeature`, `green_inverse_held_out_valid_ordinals`, `green_inverse_negative_wraps_like_numpy` (`-1.0`→'dog', `-2.0`→'cat', `-3.0`→Err), `green_inverse_non_integer_truncates_like_numpy` (`1.5`→'dog', `0.7`→'cat'), `red_inverse_out_of_range_positive` (`9.0`→Err), `red_inverse_ncols_mismatch`, `red_inverse_zero_row`, `red_inverse_use_encoded_value_unknown_cell` (`tests/divergence_ordinal_encoder.rs`). SCOPE LIMITATION (R-HONEST-3): the `unknown_value`-cell → `None` inverse (sklearn `:1673`) is unrepresentable in `Array2<String>` (would need `Array2<Option<String>>`), so a `use_encoded_value` cell equal to `unknown_value` ERRORS (checked BEFORE the index logic so the sentinel is not silently wrapped) instead of yielding `None`; the default `Error`-mode encoder has only valid ordinals so its inverse is COMPLETE and bit-exact. Consumer: crate re-export `lib.rs:142`. |
//! | REQ-10 (get_feature_names_out + n_features_in_) | SHIPPED | `FittedOrdinalEncoder::n_features_in()` (= `n_features()`, sklearn `n_features_in_`) + `get_feature_names_out(input_features)` — `OneToOneFeatureMixin` (one output col per input col) returns the INPUT names unchanged: `None` -> `["x0","x1",..]` (`_check_feature_names_in`), `Some(names)` -> verbatim, a wrong-length `input_features` -> `ShapeMismatch` (sklearn ValueError). Live-oracle test `req10_feature_names_out_and_n_features_in` (`['x0','x1']`, `['a','b']`, wrong-length Err). feature_names_in_ (string input-name capture) stays NOT-STARTED (ferrolearn fit takes positional columns, no input names). Consumer: crate re-export `lib.rs:142`. |
//! | REQ-11 (full ctor + _parameter_constraints) | NOT-STARTED | open prereq blocker #1166. `new()` takes no params (`:1320-1386`). |
//! | REQ-12 (PyO3 binding) | SHIPPED | `_RsOrdinalEncoder` (hand `#[pyclass]`, `ferrolearn-python/src/extras.rs`) over `OrdinalEncoder`/`FittedOrdinalEncoder`/`HandleUnknown`/`Categories` — the FIRST STRING-INPUT binding: `fit(rows)`/`transform(rows)` take a Python `list[list[str]]` (PyO3 `Vec<Vec<String>>` extraction, NOT a numpy f64 array), validate rectangular rows (ragged → `PyValueError`), build `Array2<String>` via `Array2::from_shape_vec`, and `transform` returns `PyArray2<f64>`; `inverse_transform(PyReadonlyArray2<f64>)` returns the `Array2<String>` rows as `Vec<Vec<String>>` (the `use_encoded_value`→None inverse ERRORS, REQ-9 scope → `PyValueError`). Ctor knobs `handle_unknown="error"` (`resolve_handle_unknown`: "error"→`Error`, "use_encoded_value"→`UseEncodedValue`, bad→`PyValueError` per `_encoders.py:1425`), `unknown_value: Option<f64>=None`, `categories: Option<Vec<Vec<String>>>=None` (None→`Auto`, Some→`Explicit`); the REQ-5/REQ-7 fit validations (`OrdinalEncoder::fit`) surface as `FerroError`→`PyValueError`. `#[getter]`s `categories_` (PyList of str lists), `n_features_in_`, `feature_names_out` (`get_feature_names_out(None)`). Registered `lib.rs` `m.add_class::<extras::RsOrdinalEncoder>()`. Non-test production consumer (R-DEFER-1): `_extras.py::OrdinalEncoder(BaseEstimator)` — a CUSTOM class (NOT `_TransformerWrapper`, input is str), full 7-key keyword-only ctor (`categories`/`dtype`/`handle_unknown`/`unknown_value`/`encoded_missing_value`/`min_frequency`/`max_categories`, `_encoders.py:1435-1452`) for `get_params`/`clone`, `_to_rows` (numpy str/object array OR list-of-lists → `list[list[str]]` via `np.asarray(X).astype(str).tolist()`), `_check_unsupported` (non-NaN `encoded_missing_value` REQ-6 / `min_frequency`/`max_categories` REQ-8 / non-f64 `dtype` REQ-3 → `NotImplementedError`), `fit`/`transform`/`fit_transform`/`inverse_transform` (→ numpy object array)/`get_feature_names_out`, `@property` `categories_`/`n_features_in_`, pre-fit access → `NotFittedError` (`check_is_fitted(self, "_rs")`); re-exported in `ferrolearn/__init__.py` as `ferrolearn.OrdinalEncoder`. Live-oracle parity (R-CHAR-3, sklearn 1.5.2, `tests/divergence_ordinal_encoder_py.py`, 19 pass): `fit_transform([['cat'],['dog'],['cat']])==[[0.],[1.],[0.]]`==sklearn, `categories_`==sklearn sorted-unique, multi-feature, inverse_transform roundtrip==original, `use_encoded_value`/`unknown_value=-1`→-1.0, explicit `categories=[['dog','cat','bird']]`→given-order index, `n_features_in_`, `get_feature_names_out`→`['x0','x1']` (+ input_features pass-through), pre-fit `NotFittedError`, bad `handle_unknown`→`ValueError`, unsupported (`encoded_missing_value`/`min_frequency`/`max_categories`/`dtype`)→`NotImplementedError`, 7-key get_params==sklearn, `clone`, numpy object/str-array input. STRING-only input (REQ-4 #1159), the `use_encoded_value`→None inverse (REQ-9), and the rest of the param surface stay OUT OF SCOPE (R-HONEST-3). |
//! | REQ-13 (ferray substrate) | NOT-STARTED | open prereq blocker #1168. `ndarray`+`HashMap`, not `ferray-core` (R-SUBSTRATE-1/2). |

use ferrolearn_core::error::FerroError;
use ferrolearn_core::traits::{Fit, FitTransform, Transform};
use ndarray::Array2;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// HandleUnknown
// ---------------------------------------------------------------------------

/// How [`OrdinalEncoder`] treats categories at `transform` time that were not
/// seen during `fit`.
///
/// Mirrors scikit-learn's `OrdinalEncoder(handle_unknown=...)` parameter
/// (`sklearn/preprocessing/_encoders.py:1262`), which accepts `'error'` and
/// `'use_encoded_value'`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum HandleUnknown {
    /// Raise an error on any unknown category (scikit-learn's default
    /// `handle_unknown='error'`). This is also the default here.
    #[default]
    Error,
    /// Encode unknown categories with the configured `unknown_value` sentinel
    /// (scikit-learn's `handle_unknown='use_encoded_value'`). Requires
    /// `unknown_value` to be set.
    UseEncodedValue,
}

// ---------------------------------------------------------------------------
// Categories
// ---------------------------------------------------------------------------

/// How [`OrdinalEncoder`] determines, per column, the ordered category set used
/// to assign ordinal indices.
///
/// Mirrors scikit-learn's `OrdinalEncoder(categories=...)` parameter
/// (`sklearn/preprocessing/_encoders.py:1252`), which accepts `'auto'` or a
/// list of per-feature category lists.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum Categories {
    /// Determine the categories automatically from the training data as the
    /// sorted-unique values per column (scikit-learn's default `categories='auto'`).
    #[default]
    Auto,
    /// Use the explicit, user-provided category lists. `Explicit(lists)[j]` is
    /// the ordered category set for column `j`, used **as given** (the order is
    /// preserved, NOT re-sorted), mirroring scikit-learn's
    /// `categories=[list, ...]` (`_encoders.py:114`, the categories are used
    /// `np.array(self.categories[i])` as-is).
    Explicit(Vec<Vec<String>>),
}

// ---------------------------------------------------------------------------
// OrdinalEncoder (unfitted)
// ---------------------------------------------------------------------------

/// An unfitted ordinal encoder.
///
/// Calling [`Fit::fit`] on an `Array2<String>` learns, for each column, a
/// mapping from the unique string categories (sorted lexicographically)
/// to consecutive integers `0, 1, 2, ...`, and returns a
/// [`FittedOrdinalEncoder`].
///
/// Unknown categories at `transform` time are, by default, rejected
/// ([`HandleUnknown::Error`]). Configuring
/// [`with_handle_unknown`](OrdinalEncoder::with_handle_unknown) with
/// [`HandleUnknown::UseEncodedValue`] plus
/// [`with_unknown_value`](OrdinalEncoder::with_unknown_value) instead encodes
/// unknown categories as the supplied sentinel (which may be `f64::NAN`),
/// matching scikit-learn's `OrdinalEncoder(handle_unknown='use_encoded_value')`.
///
/// # Examples
///
/// ```
/// use ferrolearn_preprocess::ordinal_encoder::OrdinalEncoder;
/// use ferrolearn_core::traits::{Fit, Transform};
/// use ndarray::Array2;
///
/// let enc = OrdinalEncoder::new();
/// let data = Array2::from_shape_vec(
///     (3, 2),
///     vec![
///         "cat".to_string(), "small".to_string(),
///         "dog".to_string(), "large".to_string(),
///         "cat".to_string(), "small".to_string(),
///     ],
/// ).unwrap();
/// let fitted = enc.fit(&data, &()).unwrap();
/// let encoded = fitted.transform(&data).unwrap();
/// // Output is `Array2<f64>`, matching sklearn's `dtype=np.float64` default.
/// assert_eq!(encoded[[0, 0]], 0.0); // "cat" is index 0 in col 0
/// assert_eq!(encoded[[1, 0]], 1.0); // "dog" is index 1 in col 0
/// ```
#[derive(Debug, Clone, Default)]
pub struct OrdinalEncoder {
    /// How the per-column category sets are determined ([`Categories::Auto`] =
    /// sorted-unique from the data, the default; [`Categories::Explicit`] =
    /// user-provided lists used in the given order).
    categories: Categories,
    /// Strategy for unknown categories at `transform` time.
    handle_unknown: HandleUnknown,
    /// Sentinel written for unknown categories when `handle_unknown` is
    /// [`HandleUnknown::UseEncodedValue`]. May be `f64::NAN`.
    unknown_value: Option<f64>,
    /// Minimum frequency (count) below which a category is grouped into the
    /// single trailing "infrequent" ordinal index for that feature
    /// (`min_frequency`). `None` (the default) disables the min-frequency
    /// threshold. Mirrors scikit-learn's `OrdinalEncoder(min_frequency=...)`
    /// (`sklearn/preprocessing/_encoders.py:1289-1297`). SCOPE (R-HONEST-3):
    /// only the integer-count form is supported — sklearn also accepts a FLOAT
    /// fraction `min_frequency * n_samples` (`:1296-1297`,`:297-299`), which is
    /// NOT-STARTED here.
    min_frequency: Option<usize>,
    /// Upper limit on the number of output ordinal codes per feature when
    /// grouping infrequent categories (`max_categories`); the infrequent group
    /// itself counts toward this limit. `None` (the default) imposes no limit.
    /// Mirrors scikit-learn's `OrdinalEncoder(max_categories=...)`
    /// (`sklearn/preprocessing/_encoders.py:1301-1315`).
    max_categories: Option<usize>,
}

impl OrdinalEncoder {
    /// Create a new `OrdinalEncoder` with scikit-learn's defaults
    /// (`handle_unknown='error'`, no `unknown_value`).
    #[must_use]
    pub fn new() -> Self {
        Self {
            categories: Categories::Auto,
            handle_unknown: HandleUnknown::Error,
            unknown_value: None,
            min_frequency: None,
            max_categories: None,
        }
    }

    /// Set the explicit per-column category lists (`categories=[list, ...]`).
    ///
    /// Each `lists[j]` is the ordered category set for column `j`, used **as
    /// given** at `fit` time — the order is preserved (NOT re-sorted), so the
    /// assigned ordinal indices follow the supplied order, matching
    /// scikit-learn's `OrdinalEncoder(categories=...)`
    /// (`sklearn/preprocessing/_encoders.py:114`).
    ///
    /// At `fit` time the number of lists must equal the number of input columns,
    /// no list may contain duplicates, and (under the default
    /// `handle_unknown='error'`) every value seen in the data must appear in its
    /// column's list; otherwise [`Fit::fit`] returns an error. See [`Fit::fit`]
    /// for the exact validation contract.
    #[must_use]
    pub fn with_categories(mut self, categories: Vec<Vec<String>>) -> Self {
        self.categories = Categories::Explicit(categories);
        self
    }

    /// Return the configured `categories` strategy ([`Categories::Auto`] or
    /// [`Categories::Explicit`]).
    ///
    /// Named `categories_param` to avoid colliding with
    /// [`FittedOrdinalEncoder::categories`], which returns the *learned*
    /// per-column category lists after fitting.
    #[must_use]
    pub fn categories_param(&self) -> &Categories {
        &self.categories
    }

    /// Set the unknown-category strategy (`handle_unknown`).
    ///
    /// With [`HandleUnknown::UseEncodedValue`] an `unknown_value` must also be
    /// supplied via [`with_unknown_value`](OrdinalEncoder::with_unknown_value);
    /// otherwise [`Fit::fit`] returns an error (matching scikit-learn's
    /// validation).
    #[must_use]
    pub fn with_handle_unknown(mut self, handle_unknown: HandleUnknown) -> Self {
        self.handle_unknown = handle_unknown;
        self
    }

    /// Set the sentinel written for unknown categories under
    /// [`HandleUnknown::UseEncodedValue`]. May be `f64::NAN`.
    ///
    /// Setting this while `handle_unknown` is [`HandleUnknown::Error`] causes
    /// [`Fit::fit`] to return an error (matching scikit-learn's validation).
    #[must_use]
    pub fn with_unknown_value(mut self, unknown_value: f64) -> Self {
        self.unknown_value = Some(unknown_value);
        self
    }

    /// Return the configured unknown-category strategy.
    #[must_use]
    pub fn handle_unknown(&self) -> HandleUnknown {
        self.handle_unknown
    }

    /// Return the configured unknown-category sentinel, if any.
    #[must_use]
    pub fn unknown_value(&self) -> Option<f64> {
        self.unknown_value
    }

    /// Set the minimum-frequency threshold for infrequent grouping
    /// (`min_frequency`, integer count).
    ///
    /// At `fit` time a category whose count in the training data is **strictly
    /// less than** `min_frequency` is grouped with the other infrequent
    /// categories into a single trailing ordinal index `n_frequent` for that
    /// feature (the frequent categories keep ordinal indices `0..n_frequent` in
    /// their original sorted order), matching scikit-learn's
    /// `OrdinalEncoder(min_frequency=...)` integer form
    /// (`sklearn/preprocessing/_encoders.py:1289-1297`, `_identify_infrequent`
    /// `:295-296` `category_count < self.min_frequency`).
    ///
    /// Unlike [`crate::OneHotEncoder`], the infrequent group collapses to ONE
    /// **ordinal index** (not a one-hot column), so `categories_` is unchanged
    /// (all categories retained) — only the emitted ordinal code is shared.
    ///
    /// SCOPE (R-HONEST-3): only the integer-count form is supported. sklearn
    /// also accepts a FLOAT `min_frequency` interpreted as the fraction
    /// `min_frequency * n_samples` (`_encoders.py:1296-1297`,`:297-299`); the
    /// float-fraction form is NOT-STARTED here.
    #[must_use]
    pub fn with_min_frequency(mut self, min_frequency: usize) -> Self {
        self.min_frequency = Some(min_frequency);
        self
    }

    /// Set the maximum number of output ordinal codes per feature for infrequent
    /// grouping (`max_categories`).
    ///
    /// At `fit` time, if a feature would otherwise produce more than
    /// `max_categories` distinct ordinal codes, the least-frequent categories
    /// are grouped into the single trailing infrequent index so the number of
    /// codes is at most `max_categories` (the infrequent group itself counts
    /// toward the limit). Mirrors scikit-learn's
    /// `OrdinalEncoder(max_categories=...)`
    /// (`sklearn/preprocessing/_encoders.py:1301-1315`, `_identify_infrequent`
    /// `:303-315`).
    #[must_use]
    pub fn with_max_categories(mut self, max_categories: usize) -> Self {
        self.max_categories = Some(max_categories);
        self
    }

    /// Return the configured minimum-frequency threshold (`min_frequency`), or
    /// `None` if infrequent grouping by frequency is disabled.
    #[must_use]
    pub fn min_frequency(&self) -> Option<usize> {
        self.min_frequency
    }

    /// Return the configured maximum ordinal-code limit (`max_categories`), or
    /// `None` if no limit is imposed.
    #[must_use]
    pub fn max_categories(&self) -> Option<usize> {
        self.max_categories
    }

    /// Whether infrequent grouping is enabled (either `min_frequency` or
    /// `max_categories` is set). Mirrors scikit-learn's `_infrequent_enabled`
    /// (`_encoders.py:271-273`: `(max_categories is not None and
    /// max_categories >= 1) or min_frequency is not None`).
    fn infrequent_enabled(&self) -> bool {
        self.min_frequency.is_some() || self.max_categories.is_some_and(|m| m >= 1)
    }
}

// ---------------------------------------------------------------------------
// FittedOrdinalEncoder
// ---------------------------------------------------------------------------

/// A fitted ordinal encoder holding per-column category-to-index mappings.
///
/// Created by calling [`Fit::fit`] on an [`OrdinalEncoder`].
#[derive(Debug, Clone)]
pub struct FittedOrdinalEncoder {
    /// Per-column ordered category lists (index = integer value).
    pub(crate) categories: Vec<Vec<String>>,
    /// Per-column category-to-index maps.
    pub(crate) category_to_index: Vec<HashMap<String, usize>>,
    /// Strategy for unknown categories at `transform` time (threaded from the
    /// unfitted [`OrdinalEncoder`]).
    pub(crate) handle_unknown: HandleUnknown,
    /// Sentinel for unknown categories under
    /// [`HandleUnknown::UseEncodedValue`] (threaded from the unfitted encoder;
    /// validated to be present in that mode during `fit`).
    pub(crate) unknown_value: Option<f64>,
    /// Per-feature indices into `categories[j]` of the categories grouped as
    /// **infrequent** (`min_frequency`/`max_categories`), sorted ascending.
    /// Mirrors scikit-learn's private `_infrequent_indices[j]`
    /// (`_encoders.py:336-340`,`:367-370`). Empty when feature `j` has no
    /// infrequent categories (sklearn's `None`); with infrequent grouping
    /// disabled every entry is empty. Length `categories.len()`. The categories
    /// themselves are NOT removed from `categories[j]` (unlike one-hot column
    /// dropping) — only their emitted ordinal code is folded.
    pub(crate) infrequent_indices_: Vec<Vec<usize>>,
    /// Per-feature mapping from a `categories[j]` index to its emitted ORDINAL
    /// code. Mirrors scikit-learn's `_default_to_infrequent_mappings[j]`
    /// (`_encoders.py:373-400`): a frequent category maps to its remapped slot
    /// `0..n_frequent` (frequent categories keep their original sorted order),
    /// every infrequent category maps to the single trailing index
    /// `n_frequent`. When feature `j` has no infrequent categories the mapping
    /// is the identity `0..len` (sklearn stores `None`; the identity is the
    /// representable equivalent). Length `categories.len()`, with
    /// `infrequent_map[j].len() == categories[j].len()`. Used by `transform`
    /// and `inverse_transform`.
    pub(crate) infrequent_map: Vec<Vec<usize>>,
    /// Per-feature number of frequent categories (`n_frequent`): the trailing
    /// infrequent ordinal index when feature `j` has infrequent categories.
    /// Equals `categories[j].len() - infrequent_indices_[j].len()`. When feature
    /// `j` has no infrequent categories this equals `categories[j].len()` (the
    /// identity map's range). Length `categories.len()`. Used by
    /// `inverse_transform` to recognise the shared infrequent code.
    pub(crate) n_frequent: Vec<usize>,
}

impl FittedOrdinalEncoder {
    /// Return the ordered category list for each column.
    ///
    /// `categories()[j][i]` is the category that maps to integer `i` in column `j`.
    #[must_use]
    pub fn categories(&self) -> &[Vec<String>] {
        &self.categories
    }

    /// Return the infrequent category **values** for each feature
    /// (`infrequent_categories_`).
    ///
    /// `infrequent_categories()[j]` is the sorted list of category values from
    /// `categories[j]` that were grouped into the single trailing "infrequent"
    /// ordinal code (because their training count fell below `min_frequency`
    /// and/or beyond the `max_categories` limit). An EMPTY inner `Vec` means
    /// feature `j` had no infrequent categories (scikit-learn returns `None`
    /// there; an empty list is the representable equivalent). With infrequent
    /// grouping disabled every entry is empty. Mirrors scikit-learn's
    /// `OrdinalEncoder.infrequent_categories_` (`_encoders.py:255-262`):
    /// `category[indices]` over `_infrequent_indices`.
    #[must_use]
    pub fn infrequent_categories(&self) -> Vec<Vec<String>> {
        self.infrequent_indices_
            .iter()
            .enumerate()
            .map(|(j, idxs)| {
                idxs.iter()
                    .filter_map(|&idx| self.categories.get(j).and_then(|c| c.get(idx)).cloned())
                    .collect()
            })
            .collect()
    }

    /// Return the number of input columns (features).
    #[must_use]
    pub fn n_features(&self) -> usize {
        self.categories.len()
    }

    /// Return the number of features seen during `fit`.
    ///
    /// Mirrors scikit-learn's `n_features_in_` attribute (set by `_validate_data`
    /// at fit, `sklearn/base.py`). Equal to [`n_features`](Self::n_features); the
    /// distinct name matches sklearn's fitted-attribute surface (REQ-10).
    #[must_use]
    pub fn n_features_in(&self) -> usize {
        self.categories.len()
    }

    /// Return the output feature names, one per input feature.
    ///
    /// `OrdinalEncoder` is a `OneToOneFeatureMixin` (one output column per input
    /// column), so `get_feature_names_out` returns the INPUT feature names
    /// unchanged (`sklearn/utils/_set_output` / `OneToOneFeatureMixin.
    /// get_feature_names_out`): with `input_features = None` the default names
    /// `["x0", "x1", ...]` (`_check_feature_names_in`), otherwise the supplied
    /// names verbatim.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if `input_features` is `Some` but its
    /// length differs from [`n_features_in`](Self::n_features_in) (sklearn raises
    /// `ValueError("input_features should have length equal to number of features
    /// ...")`).
    pub fn get_feature_names_out(
        &self,
        input_features: Option<&[String]>,
    ) -> Result<Vec<String>, FerroError> {
        let n = self.categories.len();
        match input_features {
            None => Ok((0..n).map(|j| format!("x{j}")).collect()),
            Some(names) => {
                if names.len() != n {
                    return Err(FerroError::ShapeMismatch {
                        expected: vec![n],
                        actual: vec![names.len()],
                        context: "FittedOrdinalEncoder::get_feature_names_out (input_features \
                                  length must equal n_features_in_)"
                            .into(),
                    });
                }
                Ok(names.to_vec())
            }
        }
    }

    /// Return the configured unknown-category strategy.
    #[must_use]
    pub fn handle_unknown(&self) -> HandleUnknown {
        self.handle_unknown
    }

    /// Return the configured unknown-category sentinel, if any.
    #[must_use]
    pub fn unknown_value(&self) -> Option<f64> {
        self.unknown_value
    }

    /// Convert ordinal indices back to the original category strings.
    ///
    /// This is the inverse of [`Transform::transform`]: each `f64` cell is read
    /// as an ordinal index into the per-column `categories_` learned at `fit`
    /// time, and the corresponding category string is returned. Reusing the
    /// SHIPPED `categories_` (REQ-1), `inverse_transform(transform(X)) == X` for
    /// any `X` whose every category was seen during `fit` (a bit-exact roundtrip
    /// on the default `Error`-mode encoder). Mirrors scikit-learn's
    /// `OrdinalEncoder.inverse_transform` (`sklearn/preprocessing/_encoders.py:1595`),
    /// `X_tr[:, i] = self.categories_[i][labels]`.
    ///
    /// # Index contract (faithful to sklearn / numpy)
    ///
    /// Mirrors sklearn's `labels.astype("int64")` (`_encoders.py:1664`) followed
    /// by numpy fancy indexing `categories_[j][labels]` (`:1679`):
    /// - **truncates non-integers toward zero** (`1.5` → index `1` → that
    ///   category; `0.7` → `0`) — Rust `f64 as i64` matches the C-style cast.
    /// - **wraps small negatives** via numpy negative indexing (`-1.0` →
    ///   `categories_[j][len-1]`, the LAST category; `-2.0` → `len-2`), raising
    ///   only once the wrapped index still leaves `[0, len)` (`-3.0` with 2
    ///   categories → `IndexError`).
    /// - **errors** on an out-of-range positive ordinal (`9.0` with 2 categories
    ///   → sklearn `IndexError`) and on a non-finite cell (NaN/±inf overflow the
    ///   `astype("int64")` cast → sklearn `IndexError`/`ValueError`; guarded
    ///   explicitly because Rust's `f64 as i64` saturates NaN→0, which would
    ///   diverge).
    ///
    /// The roundtrip, held-out valid-ordinal, truncation, and negative-wrap paths
    /// all match sklearn; out-of-range / non-finite both error.
    ///
    /// # `use_encoded_value` → `None` (SCOPE LIMITATION, R-HONEST-3)
    ///
    /// With [`HandleUnknown::UseEncodedValue`], sklearn maps a cell equal to
    /// `unknown_value` back to `None` (`_encoders.py:1673`,
    /// `X_tr[mask, idx] = None`). ferrolearn's `Array2<String>` output container
    /// **cannot represent `None`** (it would require `Array2<Option<String>>`).
    /// The configured `unknown_value` is itself out of the valid `[0, len)`
    /// range (e.g. `-1`), so such a cell hits the out-of-range error path: this
    /// inverse therefore ERRORS where sklearn returns `[[None, ...]]`. This is a
    /// documented divergence, not a silent wrong-string — the honest behavior is
    /// to error rather than fabricate a category. The default `Error`-mode
    /// encoder produces only valid ordinals, so its inverse is COMPLETE and
    /// bit-exact.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::InsufficientSamples`] if the input has zero rows
    /// (symmetry with `transform`'s #2220 guard and sklearn's `check_array`).
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of columns does not
    /// match the number of features seen during fitting (sklearn's
    /// `_encoders.py:1619` "Shape of the passed X data is not correct").
    ///
    /// Returns [`FerroError::InvalidParameter`] if any cell is not an exact
    /// non-negative integer in `[0, categories_[j].len())` (sklearn's
    /// `IndexError`, plus the strict negative/non-integer contract above).
    pub fn inverse_transform(&self, x: &Array2<f64>) -> Result<Array2<String>, FerroError> {
        let n_features = self.categories.len();
        // Symmetric with `transform`'s 0-row guard (#2220) and sklearn's
        // `check_array` minimum-of-1-sample (`_encoders.py:1610`): a 0-row input
        // raises "Found array with 0 sample(s) ... a minimum of 1 is required".
        if x.nrows() == 0 {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "FittedOrdinalEncoder::inverse_transform".into(),
            });
        }
        // sklearn validates the column count (`_encoders.py:1619`) -> ValueError.
        if x.ncols() != n_features {
            return Err(FerroError::ShapeMismatch {
                expected: vec![x.nrows(), n_features],
                actual: vec![x.nrows(), x.ncols()],
                context: "FittedOrdinalEncoder::inverse_transform".into(),
            });
        }

        let n_samples = x.nrows();
        // `Array2::default` fills with the empty String; every cell is overwritten
        // on the Ok path, so the default is never observed by the caller.
        let mut out = Array2::<String>::default((n_samples, n_features));

        for j in 0..n_features {
            let cats = &self.categories[j];
            // Infrequent grouping (REQ-8). When feature `j` has infrequent
            // categories, the valid ordinal codes are `0..=n_frequent[j]`: codes
            // `0..n_frequent` index the FREQUENT-only category list (the original
            // `categories[j]` with the infrequent entries removed, order
            // preserved — sklearn `frequent_categories_mask`,
            // `_encoders.py:1648-1652`), and the shared trailing code
            // `n_frequent` inverts to the literal String `"infrequent_sklearn"`
            // (`_encoders.py:1675-1677` `X_tr[mask, idx] = "infrequent_sklearn"`).
            // UNLIKE `OneHotEncoder`'s NaN proxy, this is a REAL representable
            // String. With grouping disabled `infrequent_indices_[j]` is empty,
            // so this branch is skipped and the SHIPPED REQ-9 path runs unchanged.
            let frequent_only: Option<Vec<String>> = if self
                .infrequent_indices_
                .get(j)
                .is_some_and(|v| !v.is_empty())
            {
                let map = &self.infrequent_map[j];
                let nf = self.n_frequent[j];
                // Slot `s` (in `0..nf`) → the `categories[j]` element whose
                // remapped code is `s` (frequent categories keep their order).
                let mut fo: Vec<String> = Vec::with_capacity(nf);
                for s in 0..nf {
                    if let Some(orig) = map.iter().position(|&c| c == s)
                        && let Some(cat) = cats.get(orig)
                    {
                        fo.push(cat.clone());
                    }
                }
                Some(fo)
            } else {
                None
            };
            // The category list the numpy index logic indexes into: the
            // frequent-only list when grouping is active for this feature, else
            // the full `categories[j]` (SHIPPED REQ-9, UNCHANGED).
            let index_cats: &[String] = frequent_only.as_deref().unwrap_or(cats);
            let len = index_cats.len() as i64;
            for i in 0..n_samples {
                let v = x[[i, j]];
                // `use_encoded_value`: sklearn maps a cell equal to
                // `unknown_value` back to `None` (`_encoders.py:1673`) BEFORE the
                // int cast / indexing. `Array2<String>` cannot hold `None`, so
                // this cell errors (documented scope limitation, R-HONEST-3) —
                // checked first so the configured sentinel (e.g. `-1`) is NOT
                // silently wrapped to a real category by the numpy index logic.
                if self.handle_unknown == HandleUnknown::UseEncodedValue
                    && let Some(uv) = self.unknown_value
                    && (v == uv || (v.is_nan() && uv.is_nan()))
                {
                    return Err(FerroError::InvalidParameter {
                        name: "X".into(),
                        reason: format!(
                            "value {v} at row {i}, feature {j} equals unknown_value; \
                             sklearn inverts it to None, which Array2<String> cannot \
                             represent (would need Array2<Option<String>>)"
                        ),
                    });
                }
                // Infrequent: a cell EXACTLY equal to the shared trailing code
                // `n_frequent` (a float equality, computed on the RAW label
                // BEFORE the int cast — sklearn `labels == infrequent_encoding_value`,
                // `_encoders.py:1644`) inverts to `"infrequent_sklearn"`. A cell
                // that merely truncates to `n_frequent` (e.g. `2.5`) does NOT —
                // it falls through to the frequent-only index logic and errors out
                // of range, matching the live oracle.
                if frequent_only.is_some() && v == self.n_frequent[j] as f64 {
                    out[[i, j]] = "infrequent_sklearn".to_string();
                    continue;
                }
                // sklearn does `labels.astype('int64')` then `categories_[j][idx]`
                // (`_encoders.py:1664`,`:1679`). A non-finite cell overflows the
                // cast (NaN/+-inf -> IndexError/ValueError); reject it (R-CODE-2:
                // Rust's `f64 as i64` would saturate NaN->0, diverging from numpy,
                // so guard explicitly).
                if !v.is_finite() {
                    return Err(FerroError::InvalidParameter {
                        name: "X".into(),
                        reason: format!(
                            "value {v} at row {i}, feature {j} is not a finite ordinal \
                             index (sklearn raises on NaN/inf astype('int64'))"
                        ),
                    });
                }
                // `astype('int64')` truncates toward zero (Rust `as i64` matches
                // for finite values); numpy indexing then WRAPS a negative index
                // by `+= len` (`-1` -> last category), raising only once the
                // wrapped index still leaves `[0, len)`.
                let mut idx = v as i64;
                if idx < 0 {
                    idx += len;
                }
                if idx < 0 || idx >= len {
                    return Err(FerroError::InvalidParameter {
                        name: "X".into(),
                        reason: format!(
                            "ordinal index {} at row {i} is out of bounds for the {len} \
                             categories of feature {j} (sklearn IndexError)",
                            v as i64
                        ),
                    });
                }
                // `idx` is now provably in `[0, len)` (checked above) — no panic.
                // `index_cats` is the frequent-only list under infrequent
                // grouping (so a frequent code maps to its frequent category),
                // else the full `categories[j]` (SHIPPED REQ-9, UNCHANGED).
                out[[i, j]] = index_cats[idx as usize].clone();
            }
        }

        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Cast an ordinal category index to `f64`, matching scikit-learn's default
/// `OrdinalEncoder(dtype=np.float64)` output container
/// (`sklearn/preprocessing/_encoders.py:1262`).
///
/// `f64` exactly represents every integer up to `2^53`, so this is lossless for
/// any realistic category count. Indices above `2^53` (astronomically more
/// categories than memory could hold) round to the nearest `f64`, never panic
/// (R-CODE-2) — the same silent float rounding numpy performs.
#[inline]
fn ordinal_index_to_f64(idx: usize) -> f64 {
    idx as f64
}

/// Identify the indices of infrequent categories for one feature, given the
/// per-category training `counts` (aligned with `categories[j]`) and the
/// `min_frequency`/`max_categories` thresholds.
///
/// Mirrors scikit-learn's `_BaseEncoder._identify_infrequent`
/// (`_encoders.py:275-318`). This is the SAME algorithm the SHIPPED
/// `OneHotEncoder` REQ-5b uses (`one_hot_encoder.rs::identify_infrequent`):
/// 1. min_frequency: a category with `count < min_frequency` is infrequent
///    (`:295-296`, integer form only).
/// 2. max_categories: if (after step 1) the feature would still produce more
///    than `max_categories` ordinal codes — counted as `n_remaining_frequent +
///    1` for the infrequent group (`:303`) — the least-frequent categories are
///    additionally marked infrequent until only `max_categories - 1` frequent
///    categories remain (`:304-315`). Ties broken by a STABLE sort over the
///    FULL count array, so among equal counts the SMALLER category index is
///    marked infrequent first (sklearn `np.argsort(kind="mergesort")[:-k]`),
///    i.e. the LARGER index is favoured to stay frequent. `max_categories == 1`
///    (frequent_category_count 0) makes every category infrequent (`:307-309`).
///
/// Returns the sorted-ascending infrequent indices (empty if none — sklearn's
/// `None`). Never panics (R-CODE-2).
fn identify_infrequent(
    counts: &[usize],
    min_frequency: Option<usize>,
    max_categories: Option<usize>,
) -> Vec<usize> {
    let n = counts.len();
    let mut infrequent_mask = vec![false; n];

    // Step 1: min_frequency (integer count). `count < min_frequency`.
    if let Some(min_freq) = min_frequency {
        for (idx, &c) in counts.iter().enumerate() {
            if c < min_freq {
                infrequent_mask[idx] = true;
            }
        }
    }

    // Step 2: max_categories on the survivors. `n_current_features` counts the
    // remaining frequent categories PLUS 1 for the infrequent group
    // (`_encoders.py:303`).
    if let Some(max_cat) = max_categories {
        let n_infreq = infrequent_mask.iter().filter(|&&m| m).count();
        let n_current_features = n - n_infreq + 1;
        if max_cat < n_current_features {
            // `max_categories` includes the one infrequent category.
            let frequent_category_count = max_cat - 1;
            if frequent_category_count == 0 {
                // All categories are infrequent (`:307-309`).
                infrequent_mask.iter_mut().for_each(|m| *m = true);
            } else {
                // Stable argsort over the FULL count array (ascending by count,
                // ties by ascending index), then mark the smallest
                // `n - frequent_category_count` levels infrequent — i.e. keep the
                // top `frequent_category_count` by count, with ties resolved in
                // favor of the LARGER index (`np.argsort(kind="mergesort")[:-k]`,
                // `:312-315`).
                let mut order: Vec<usize> = (0..n).collect();
                order.sort_by(|&a, &b| counts[a].cmp(&counts[b]).then(a.cmp(&b)));
                let keep = frequent_category_count.min(n);
                let cut = n - keep;
                for &idx in &order[..cut] {
                    infrequent_mask[idx] = true;
                }
            }
        }
    }

    infrequent_mask
        .iter()
        .enumerate()
        .filter_map(|(idx, &m)| if m { Some(idx) } else { None })
        .collect()
}

/// Build the per-feature mapping from a `categories[j]` index to its emitted
/// ORDINAL code.
///
/// Mirrors scikit-learn's `_default_to_infrequent_mappings[j]`
/// (`_encoders.py:373-400`): frequent categories take codes `0..n_frequent` in
/// their original (ascending-index) order; every infrequent category maps to
/// the single trailing code `n_frequent`. With no infrequent categories the
/// mapping is the identity `0..n`. `infrequent` must be sorted ascending. Never
/// panics (R-CODE-2): every index is bounds-checked.
fn build_infrequent_map(n: usize, infrequent: &[usize]) -> Vec<usize> {
    if infrequent.is_empty() {
        return (0..n).collect();
    }
    let n_frequent = n - infrequent.len();
    let mut map = vec![n_frequent; n];
    let mut next_frequent = 0usize;
    for (idx, slot) in map.iter_mut().enumerate() {
        if infrequent.binary_search(&idx).is_ok() {
            // Infrequent → the trailing code (already set to `n_frequent`).
        } else {
            *slot = next_frequent;
            next_frequent += 1;
        }
    }
    map
}

// ---------------------------------------------------------------------------
// Trait implementations
// ---------------------------------------------------------------------------

impl Fit<Array2<String>, ()> for OrdinalEncoder {
    type Fitted = FittedOrdinalEncoder;
    type Error = FerroError;

    /// Fit the encoder by building per-column category-to-index mappings.
    ///
    /// With the default `categories='auto'` ([`Categories::Auto`]), categories
    /// are recorded in **lexicographic order** in each column, matching
    /// scikit-learn's `OrdinalEncoder.categories_`.
    ///
    /// With explicit categories ([`Categories::Explicit`], set via
    /// [`OrdinalEncoder::with_categories`]), the user-provided lists are used in
    /// the **given order** (NOT re-sorted), and the ordinal indices follow that
    /// order, mirroring scikit-learn (`sklearn/preprocessing/_encoders.py:114`).
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::InsufficientSamples`] if the input has zero rows.
    ///
    /// Returns [`FerroError::ShapeMismatch`] if explicit categories are set but
    /// the number of category lists differs from the number of input columns
    /// (sklearn `_encoders.py:85-89` "Shape mismatch: if categories is an array,
    /// it has to be of shape (n_features,).").
    ///
    /// Returns [`FerroError::InvalidParameter`] if an explicit category list
    /// contains duplicate elements (sklearn `_encoders.py:136-141`), or — under
    /// the default [`HandleUnknown::Error`] — if a value seen in the data is not
    /// in its column's explicit list (sklearn `_encoders.py:153-160` "Found
    /// unknown categories ... during fit"; SKIPPED under
    /// [`HandleUnknown::UseEncodedValue`]).
    ///
    /// Returns [`FerroError::InvalidParameter`] for the `handle_unknown` /
    /// `unknown_value` validation failures (mirroring scikit-learn's
    /// `TypeError`/`ValueError` at `_encoders.py:1473-1526`): selecting
    /// [`HandleUnknown::UseEncodedValue`] without an `unknown_value`; setting an
    /// `unknown_value` while in [`HandleUnknown::Error`] mode; or an
    /// `unknown_value` that collides with an already-used encoding index.
    fn fit(&self, x: &Array2<String>, _y: &()) -> Result<FittedOrdinalEncoder, FerroError> {
        // sklearn `_parameter_constraints` (`@_fit_context`, validated BEFORE the
        // data): `min_frequency` and `max_categories` are each
        // `Interval(Integral, 1, None)` — a value of 0 raises
        // `InvalidParameterError` ("must be an int in the range [1, inf)").
        // REQ-8, verified live: `OrdinalEncoder(min_frequency=0).fit` →
        // InvalidParameterError. (handle_unknown is a type-safe Rust enum, so its
        // StrOptions constraint is provided by the type system.)
        if self.min_frequency == Some(0) {
            return Err(FerroError::InvalidParameter {
                name: "min_frequency".into(),
                reason: "must be an int in the range [1, inf)".into(),
            });
        }
        if self.max_categories == Some(0) {
            return Err(FerroError::InvalidParameter {
                name: "max_categories".into(),
                reason: "must be an int in the range [1, inf)".into(),
            });
        }

        let n_samples = x.nrows();
        if n_samples == 0 {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "OrdinalEncoder::fit".into(),
            });
        }

        // Validation (a)/(b) on the param SHAPE — independent of the data, but
        // matching sklearn these are evaluated in `fit`, AFTER the 0-row
        // `check_array` guard above and (for the collision check) AFTER the
        // categories_ compute below. (a)/(b) map sklearn's `TypeError`
        // (`_encoders.py:1481-1493`).
        match (self.handle_unknown, self.unknown_value) {
            // (a) use_encoded_value REQUIRES an unknown_value (an int or nan).
            // sklearn: `not isinstance(unknown_value, Integral)` -> TypeError
            // (`:1481`); `unknown_value is None` falls into that branch.
            (HandleUnknown::UseEncodedValue, None) => {
                return Err(FerroError::InvalidParameter {
                    name: "unknown_value".into(),
                    reason: "unknown_value should be set (an integer or NaN) when \
                             handle_unknown is 'use_encoded_value'"
                        .into(),
                });
            }
            // (b) error-mode forbids a set unknown_value. sklearn: `:1488`
            // `elif self.unknown_value is not None` -> TypeError.
            (HandleUnknown::Error, Some(v)) => {
                return Err(FerroError::InvalidParameter {
                    name: "unknown_value".into(),
                    reason: format!(
                        "unknown_value should only be set when handle_unknown is \
                         'use_encoded_value', got {v}"
                    ),
                });
            }
            _ => {}
        }

        let n_features = x.ncols();
        let mut categories = Vec::with_capacity(n_features);
        let mut category_to_index = Vec::with_capacity(n_features);

        match &self.categories {
            // `categories='auto'` (default): per column, sorted-unique from the
            // data (SHIPPED REQ-1, UNCHANGED). sklearn `_encoders.py:98-99`
            // `result = _unique(Xi)`.
            Categories::Auto => {
                for j in 0..n_features {
                    // Collect unique categories then sort lexicographically so the
                    // assigned indices match sklearn's `OrdinalEncoder`, which
                    // documents `categories_ = sorted(unique(X[:, j]))`. (Older
                    // ferrolearn versions used first-seen order — #344.)
                    let mut unique: Vec<String> = Vec::new();
                    let mut seen_set: std::collections::HashSet<String> =
                        std::collections::HashSet::new();
                    for i in 0..n_samples {
                        let cat = &x[[i, j]];
                        if seen_set.insert(cat.clone()) {
                            unique.push(cat.clone());
                        }
                    }
                    unique.sort();

                    let map: HashMap<String, usize> = unique
                        .iter()
                        .enumerate()
                        .map(|(idx, s)| (s.clone(), idx))
                        .collect();

                    categories.push(unique);
                    category_to_index.push(map);
                }
            }
            // `categories=[list, ...]` (explicit): use the user-provided lists in
            // the GIVEN order (NOT re-sorted), mirroring sklearn `_encoders.py:84-160`.
            Categories::Explicit(lists) => {
                // sklearn (`_encoders.py:85-89`): the list count must match
                // n_features, else ValueError -> map to `ShapeMismatch`.
                if lists.len() != n_features {
                    return Err(FerroError::ShapeMismatch {
                        expected: vec![n_features],
                        actual: vec![lists.len()],
                        context: "Shape mismatch: if categories is an array, it has to be of \
                                  shape (n_features,)."
                            .into(),
                    });
                }

                for (j, list) in lists.iter().enumerate() {
                    // sklearn (`_encoders.py:114-117`) indexes `cats[0]` on the
                    // provided list BEFORE the duplicate/subset checks, so an
                    // EMPTY explicit list raises `IndexError` at fit in BOTH
                    // handle_unknown modes (#2229). Reject it here (the
                    // use_encoded_value path would otherwise skip the subset
                    // check and silently fit an empty category set).
                    if list.is_empty() {
                        return Err(FerroError::InvalidParameter {
                            name: "categories".into(),
                            reason: format!(
                                "column {j} has an empty predefined category list; \
                                 each feature needs at least one category"
                            ),
                        });
                    }
                    // sklearn (`_encoders.py:136-141`): a list with duplicate
                    // elements raises ValueError. Build the index map detecting
                    // duplicates in one pass (R-CODE-2: never panic).
                    let mut map: HashMap<String, usize> = HashMap::with_capacity(list.len());
                    for (idx, cat) in list.iter().enumerate() {
                        if map.insert(cat.clone(), idx).is_some() {
                            return Err(FerroError::InvalidParameter {
                                name: "categories".into(),
                                reason: format!(
                                    "In column {j}, the predefined categories contain \
                                     duplicate elements."
                                ),
                            });
                        }
                    }

                    // sklearn (`_encoders.py:153-160`): under handle_unknown='error'
                    // every value seen in the data must be present in the
                    // predefined list, else ValueError. Under 'use_encoded_value'
                    // this fit-time subset check is SKIPPED (out-of-set data is
                    // fine — encoded to `unknown_value` later at transform time).
                    if self.handle_unknown == HandleUnknown::Error {
                        for i in 0..n_samples {
                            let cat = &x[[i, j]];
                            if !map.contains_key(cat) {
                                return Err(FerroError::InvalidParameter {
                                    name: "X".into(),
                                    reason: format!(
                                        "Found unknown categories [{cat}] in column {j} \
                                         during fit"
                                    ),
                                });
                            }
                        }
                    }

                    // Use the list AS-GIVEN (preserve order — do NOT sort).
                    categories.push(list.clone());
                    category_to_index.push(map);
                }
            }
        }

        // Infrequent grouping (REQ-8). When `min_frequency`/`max_categories` are
        // set, fold the least-frequent categories of each feature into a single
        // shared trailing ORDINAL code (the frequent categories keep codes
        // `0..n_frequent` in their original sorted order). `categories` is NOT
        // changed (all categories retained, sklearn keeps `categories_` whole and
        // only remaps the emitted index, `_encoders.py:1289-1370`) — only the
        // per-feature `infrequent_map` / `infrequent_indices_` / `n_frequent` are
        // built. With grouping disabled the map is the identity and every feature
        // has no infrequent categories.
        let mut infrequent_indices_: Vec<Vec<usize>> = Vec::with_capacity(n_features);
        let mut infrequent_map: Vec<Vec<usize>> = Vec::with_capacity(n_features);
        let mut n_frequent: Vec<usize> = Vec::with_capacity(n_features);
        if self.infrequent_enabled() {
            for (j, cats) in categories.iter().enumerate() {
                // Per-category training counts ALIGNED with `categories[j]`
                // (sklearn `_unique(Xi, return_counts=True)`,
                // `_encoders.py:99-102`). Built from the fit data through the
                // category→index map, so it works for BOTH the Auto and Explicit
                // category sets. (A datum not in an explicit list contributes no
                // count — under `handle_unknown='error'` the subset check above
                // already rejected it; under `use_encoded_value` it is an unknown
                // that does not affect category frequencies.)
                let map = &category_to_index[j];
                let mut counts = vec![0usize; cats.len()];
                for i in 0..n_samples {
                    if let Some(&idx) = map.get(&x[[i, j]]) {
                        counts[idx] += 1;
                    }
                }
                let infreq = identify_infrequent(&counts, self.min_frequency, self.max_categories);
                let imap = build_infrequent_map(cats.len(), &infreq);
                n_frequent.push(cats.len() - infreq.len());
                infrequent_indices_.push(infreq);
                infrequent_map.push(imap);
            }
        } else {
            for cats in &categories {
                infrequent_indices_.push(Vec::new());
                infrequent_map.push((0..cats.len()).collect());
                n_frequent.push(cats.len());
            }
        }

        // Validation (a'): sklearn (`_encoders.py:1481-1487`) requires
        // `unknown_value` to be an INTEGER or `np.nan` when
        // `handle_unknown='use_encoded_value'` — a non-integer float raises
        // `TypeError` BEFORE the range/collision check (#2221). `f64` cannot
        // express "integral", so a non-nan value with a fractional part is
        // rejected here.
        if self.handle_unknown == HandleUnknown::UseEncodedValue
            && let Some(v) = self.unknown_value
            && !v.is_nan()
            && v.fract() != 0.0
        {
            return Err(FerroError::InvalidParameter {
                name: "unknown_value".into(),
                reason: format!(
                    "unknown_value should be an integer or np.nan when \
                     handle_unknown is 'use_encoded_value', got {v}"
                ),
            });
        }

        // Validation (c): collision of a non-nan integer unknown_value with an
        // already-used encoding index. sklearn (`_encoders.py:1518-1526`) loops
        // each column's cardinality and raises `ValueError` if
        // `0 <= unknown_value < cardinality`; that is equivalent to comparing
        // against the maximum cardinality. The earlier sklearn check
        // (`:1481`) already guaranteed `unknown_value` is an int or nan, so a
        // non-integer / nan value is fine here, as is a negative value or one
        // `>= max_cardinality`.
        if self.handle_unknown == HandleUnknown::UseEncodedValue
            && let Some(v) = self.unknown_value
            && !v.is_nan()
            && v.fract() == 0.0
        {
            // sklearn's collision check keys off the EFFECTIVE number of distinct
            // output codes per feature: with infrequent grouping a feature emits
            // `n_frequent + 1` codes (the shared infrequent index), so its
            // cardinality for the unknown_value collision is `n_frequent + 1`, NOT
            // `len(categories_)` (verified live: `min_frequency=2` over 4 cats →
            // 3 codes → `unknown_value=3` is accepted, `=2` collides). With
            // grouping disabled `n_frequent[j] == categories[j].len()` and there
            // is no infrequent code, so this reduces to the SHIPPED REQ-5 check.
            let max_cardinality = (0..n_features)
                .map(|j| n_frequent[j] + usize::from(!infrequent_indices_[j].is_empty()))
                .max()
                .unwrap_or(0);
            // `0 <= v < max_cardinality` with v an integer-valued f64.
            if v >= 0.0 && v < max_cardinality as f64 {
                return Err(FerroError::InvalidParameter {
                    name: "unknown_value".into(),
                    reason: format!(
                        "The used value for unknown_value {v} is one of the \
                         values already used for encoding the seen categories"
                    ),
                });
            }
        }

        Ok(FittedOrdinalEncoder {
            categories,
            category_to_index,
            handle_unknown: self.handle_unknown,
            unknown_value: self.unknown_value,
            infrequent_indices_,
            infrequent_map,
            n_frequent,
        })
    }
}

impl Transform<Array2<String>> for FittedOrdinalEncoder {
    type Output = Array2<f64>;
    type Error = FerroError;

    /// Transform string categories to ordinal indices, returned as `f64`.
    ///
    /// Each cell is the (lexicographic) category index cast to `f64`. The
    /// ordinal VALUES are unchanged from the integer mapping; only the output
    /// container dtype is `f64`, matching scikit-learn's
    /// `OrdinalEncoder(dtype=np.float64)` default
    /// (`sklearn/preprocessing/_encoders.py:1262`). A configurable non-float64
    /// output dtype (e.g. `int32`) is OUT OF SCOPE here — ferrolearn's output is
    /// the fixed sklearn DEFAULT `f64`; a `dtype` param is a follow-on design
    /// (blocker #1158). `f64` exactly represents every integer up to `2^53`, so
    /// the cast is lossless for any realistic category count.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of columns does not
    /// match the number of features seen during fitting.
    ///
    /// Returns [`FerroError::InvalidParameter`] if any category was not seen
    /// during fitting AND `handle_unknown` is [`HandleUnknown::Error`] (the
    /// default). Under [`HandleUnknown::UseEncodedValue`], unknown categories
    /// are instead encoded as the configured `unknown_value` sentinel (which may
    /// be `f64::NAN`), matching sklearn `_encoders.py:1591`.
    fn transform(&self, x: &Array2<String>) -> Result<Array2<f64>, FerroError> {
        let n_features = self.categories.len();
        // sklearn `OrdinalEncoder.transform` -> `_transform` -> `_check_X` ->
        // `check_array` (`_encoders.py:45`) enforces a minimum of 1 sample BEFORE
        // the n_features comparison (#2220, symmetric with the 0-row fit guard).
        // A 0-row input raises "Found array with 0 sample(s) ... minimum of 1".
        if x.nrows() == 0 {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "FittedOrdinalEncoder::transform".into(),
            });
        }
        if x.ncols() != n_features {
            return Err(FerroError::ShapeMismatch {
                expected: vec![x.nrows(), n_features],
                actual: vec![x.nrows(), x.ncols()],
                context: "FittedOrdinalEncoder::transform".into(),
            });
        }

        let n_samples = x.nrows();
        let mut out = Array2::zeros((n_samples, n_features));

        for j in 0..n_features {
            let map = &self.category_to_index[j];
            // Per-feature infrequent remapping (REQ-8): a found category's
            // `categories[j]` index is routed through `infrequent_map[j]` to its
            // emitted ordinal code (a frequent category → its remapped slot
            // `0..n_frequent`, an infrequent category → the shared trailing code
            // `n_frequent`), mirroring sklearn `_map_infrequent_categories`
            // (`_encoders.py:402-452`: `X_int = np.take(mapping, X_int)`). With
            // grouping DISABLED `infrequent_map[j]` is the identity, so the code
            // equals `idx` — the SHIPPED REQ-2 behaviour is UNCHANGED.
            let imap = self.infrequent_map.get(j);
            for i in 0..n_samples {
                let cat = &x[[i, j]];
                match map.get(cat) {
                    // Route the category index through the infrequent map, then
                    // cast the resulting ordinal code to f64 (sklearn's float64
                    // default, `_encoders.py:1262`). Lossless: codes are < 2^53.
                    // Bounds-safe: `imap.get(idx)` falls back to the raw `idx`
                    // (R-CODE-2) — `imap` always has `categories[j].len()` entries.
                    Some(&idx) => {
                        let code = imap.and_then(|m| m.get(idx)).copied().unwrap_or(idx);
                        out[[i, j]] = ordinal_index_to_f64(code);
                    }
                    None => match self.handle_unknown {
                        // handle_unknown='use_encoded_value': write the sentinel
                        // (which may be NaN). sklearn `_encoders.py:1591`
                        // `X_trans[~X_mask] = self.unknown_value`. `fit`
                        // guaranteed `unknown_value` is `Some` in this mode, but
                        // we never panic (R-CODE-2): fall back to the Error path
                        // if it were somehow `None`.
                        HandleUnknown::UseEncodedValue => match self.unknown_value {
                            Some(v) => out[[i, j]] = v,
                            None => {
                                return Err(FerroError::InvalidParameter {
                                    name: format!("x[{i},{j}]"),
                                    reason: format!(
                                        "unknown category \"{cat}\" in column {j} and \
                                         no unknown_value configured"
                                    ),
                                });
                            }
                        },
                        // handle_unknown='error' (default): reject (SHIPPED
                        // REQ-2, UNCHANGED). sklearn raises ValueError
                        // "Found unknown categories ... during transform".
                        HandleUnknown::Error => {
                            return Err(FerroError::InvalidParameter {
                                name: format!("x[{i},{j}]"),
                                reason: format!("unknown category \"{cat}\" in column {j}"),
                            });
                        }
                    },
                }
            }
        }

        Ok(out)
    }
}

/// Implement `Transform` on the unfitted encoder to satisfy the
/// `FitTransform: Transform` supertrait bound.
impl Transform<Array2<String>> for OrdinalEncoder {
    type Output = Array2<f64>;
    type Error = FerroError;

    /// Always returns an error — the encoder must be fitted first.
    fn transform(&self, _x: &Array2<String>) -> Result<Array2<f64>, FerroError> {
        Err(FerroError::InvalidParameter {
            name: "OrdinalEncoder".into(),
            reason: "encoder must be fitted before calling transform; use fit() first".into(),
        })
    }
}

impl FitTransform<Array2<String>> for OrdinalEncoder {
    type FitError = FerroError;

    /// Fit the encoder on `x` and return the encoded output in one step.
    ///
    /// # Errors
    ///
    /// Returns an error if fitting or transformation fails.
    fn fit_transform(&self, x: &Array2<String>) -> Result<Array2<f64>, FerroError> {
        let fitted = self.fit(x, &())?;
        fitted.transform(x)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn make_2col(rows: &[(&str, &str)]) -> Array2<String> {
        let flat: Vec<String> = rows
            .iter()
            .flat_map(|(a, b)| [a.to_string(), b.to_string()])
            .collect();
        Array2::from_shape_vec((rows.len(), 2), flat).unwrap()
    }

    #[test]
    fn test_ordinal_encoder_basic() {
        let enc = OrdinalEncoder::new();
        let x = make_2col(&[
            ("cat", "small"),
            ("dog", "large"),
            ("cat", "medium"),
            ("bird", "small"),
        ]);
        let fitted = enc.fit(&x, &()).unwrap();

        // Categories are sorted lexicographically (sklearn convention).
        assert_eq!(fitted.categories()[0], vec!["bird", "cat", "dog"]);
        assert_eq!(fitted.categories()[1], vec!["large", "medium", "small"]);

        let encoded = fitted.transform(&x).unwrap();
        // Output container is `Array2<f64>` (sklearn's `dtype=np.float64`).
        assert_eq!(encoded[[0, 0]], 1.0); // "cat"  -> 1 (lex pos)
        assert_eq!(encoded[[1, 0]], 2.0); // "dog"  -> 2
        assert_eq!(encoded[[2, 0]], 1.0); // "cat"  -> 1
        assert_eq!(encoded[[3, 0]], 0.0); // "bird" -> 0
        assert_eq!(encoded[[0, 1]], 2.0); // "small"  -> 2
        assert_eq!(encoded[[1, 1]], 0.0); // "large"  -> 0
        assert_eq!(encoded[[2, 1]], 1.0); // "medium" -> 1
        assert_eq!(encoded[[3, 1]], 2.0); // "small"  -> 2
    }

    #[test]
    fn test_fit_transform_equivalence() {
        let enc = OrdinalEncoder::new();
        let x = make_2col(&[("a", "x"), ("b", "y"), ("a", "z")]);
        let via_ft = enc.fit_transform(&x).unwrap();
        let fitted = enc.fit(&x, &()).unwrap();
        let via_sep = fitted.transform(&x).unwrap();
        assert_eq!(via_ft, via_sep);
    }

    #[test]
    fn test_unknown_category_error() {
        let enc = OrdinalEncoder::new();
        let x_train = make_2col(&[("cat", "small"), ("dog", "large")]);
        let fitted = enc.fit(&x_train, &()).unwrap();
        let x_test = make_2col(&[("fish", "small")]);
        assert!(fitted.transform(&x_test).is_err());
    }

    #[test]
    fn test_shape_mismatch_error() {
        let enc = OrdinalEncoder::new();
        let x_train = make_2col(&[("a", "x")]);
        let fitted = enc.fit(&x_train, &()).unwrap();
        // Single-column input when 2 cols expected
        let x_bad = Array2::from_shape_vec((1, 1), vec!["a".to_string()]).unwrap();
        assert!(fitted.transform(&x_bad).is_err());
    }

    #[test]
    fn test_insufficient_samples_error() {
        let enc = OrdinalEncoder::new();
        let x: Array2<String> = Array2::from_shape_vec((0, 2), vec![]).unwrap();
        assert!(enc.fit(&x, &()).is_err());
    }

    #[test]
    fn test_unfitted_transform_error() {
        let enc = OrdinalEncoder::new();
        let x = make_2col(&[("a", "x")]);
        assert!(enc.transform(&x).is_err());
    }

    #[test]
    fn test_single_column() {
        let enc = OrdinalEncoder::new();
        let flat = vec![
            "red".to_string(),
            "green".to_string(),
            "blue".to_string(),
            "red".to_string(),
        ];
        let x = Array2::from_shape_vec((4, 1), flat).unwrap();
        let fitted = enc.fit(&x, &()).unwrap();
        // Lex order: blue (0), green (1), red (2)
        assert_eq!(fitted.categories()[0], vec!["blue", "green", "red"]);
        let encoded = fitted.transform(&x).unwrap();
        assert_eq!(encoded[[0, 0]], 2.0); // red
        assert_eq!(encoded[[1, 0]], 1.0); // green
        assert_eq!(encoded[[2, 0]], 0.0); // blue
        assert_eq!(encoded[[3, 0]], 2.0); // red
    }

    #[test]
    fn test_n_features() {
        let enc = OrdinalEncoder::new();
        let x = make_2col(&[("a", "x")]);
        let fitted = enc.fit(&x, &()).unwrap();
        assert_eq!(fitted.n_features(), 2);
    }

    #[test]
    fn test_lexicographic_order() {
        // Categories are sorted lexicographically to match sklearn (#344).
        let enc = OrdinalEncoder::new();
        let flat = vec!["zebra".to_string(), "ant".to_string(), "moose".to_string()];
        let x = Array2::from_shape_vec((3, 1), flat).unwrap();
        let fitted = enc.fit(&x, &()).unwrap();
        // ant < moose < zebra
        assert_eq!(fitted.categories()[0][0], "ant");
        assert_eq!(fitted.categories()[0][1], "moose");
        assert_eq!(fitted.categories()[0][2], "zebra");
    }
}
