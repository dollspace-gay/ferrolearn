//! Parameter grid types and the [`macro@crate::param_grid`] macro.
//!
//! This module defines:
//!
//! - [`ParameterGrid`] — an explicit sklearn-named Cartesian-product grid.
//! - [`ParamValue`] — a dynamically-typed hyperparameter value.
//! - [`ParamSet`] — a single parameter configuration (`HashMap<String, ParamValue>`).
//! - [`macro@crate::param_grid`] — a macro that builds the Cartesian product
//!   of parameter lists as `Vec<ParamSet>`.
//!
//! # Example
//!
//! ```rust
//! use ferrolearn_model_sel::{param_grid, ParamValue};
//!
//! let grid = param_grid! {
//!     "alpha" => [0.01, 0.1, 1.0],
//!     "fit_intercept" => [true, false],
//! };
//! // 3 alphas × 2 fit_intercept values = 6 combinations.
//! assert_eq!(grid.len(), 6);
//! ```
//!
//! Mirrors scikit-learn's `sklearn/model_selection/_search.py` `ParameterGrid`
//! (`:63`, tag 1.5.2). Deterministic Cartesian-product enumeration.
//!
//! ## REQ status
//!
//! | REQ | Behavior | Status | Evidence |
//! |-----|----------|--------|----------|
//! | REQ-1 | Cartesian-product contents | SHIPPED | `param_grid!` iterative product = `itertools.product` contents; `guard_cartesian_product_contents_match` (set equality vs live oracle) |
//! | REQ-2 | enumeration order (sorted keys) | SHIPPED | macro and `ParameterGrid::new` sort axes by key = sklearn `sorted(p.items())` (`_search.py:157`, fixed #1698); `green_enumeration_order_sorted_keys`, `green_parameter_grid_public_surface_matches_sklearn` |
//! | REQ-3 | list-of-dicts grid (union of sub-grids) | NOT-STARTED | `param_grid!` is single-dict only (`_search.py:114-117`) — blocker #1700 |
//! | REQ-4 | `len`/indexing surface | SHIPPED | native `Vec` `.len()`/`[i]` (R-DEV-7); sklearn's lazy O(1) `__getitem__` not ported (eager materialization) |
//! | REQ-5 | empty-value-list rejection | SHIPPED for `ParameterGrid`; NOT-STARTED for macro | `ParameterGrid::new` rejects empty value lists with `FerroError::InvalidParameter`; `param_grid!{ "a"=>[] }` still yields `[]` because a macro has no `Result` channel — blocker #1699 (empty-dict → `[{}]` matches) |
//! | REQ-6 | `ParamValue` type coverage | SHIPPED | enum + `From` for f64/f32/i64/i32/usize/bool/String/&str |
//! | REQ-7 | non-test production consumer | SHIPPED | `ParameterGrid`/`ParamSet`/`ParamValue` are re-exported; `ParamSet`/`param_grid!` consumed by `grid_search.rs` + `random_search.rs` |
//! | REQ-8 | ferray substrate | SHIPPED (N/A) | no array layer (strings/values only) |
//!
//! Reference: scikit-learn 1.5.2 (commit 156ef14).

use std::collections::HashMap;

use ferrolearn_core::FerroError;

/// A dynamically-typed hyperparameter value.
///
/// Used as the value type in a [`ParamSet`].
#[derive(Debug, Clone, PartialEq)]
pub enum ParamValue {
    /// A 64-bit floating-point value.
    Float(f64),
    /// A 64-bit signed integer value.
    Int(i64),
    /// A boolean value.
    Bool(bool),
    /// A string value.
    String(String),
}

impl std::fmt::Display for ParamValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParamValue::Float(v) => write!(f, "{v}"),
            ParamValue::Int(v) => write!(f, "{v}"),
            ParamValue::Bool(v) => write!(f, "{v}"),
            ParamValue::String(v) => write!(f, "{v}"),
        }
    }
}

/// A single set of hyperparameter name–value pairs.
///
/// Created by [`macro@crate::param_grid`] or built manually.
pub type ParamSet = HashMap<String, ParamValue>;

/// Explicit sklearn-named parameter-grid surface.
///
/// This is the eager Rust analog of `sklearn.model_selection.ParameterGrid`:
/// it materializes every Cartesian-product combination as a [`ParamSet`],
/// enumerating axes in sorted-key order. Unlike the [`macro@crate::param_grid`]
/// macro, this constructor has a [`Result`] channel and rejects empty value
/// lists, matching sklearn's `ValueError` for invalid grids.
#[derive(Debug, Clone, PartialEq)]
pub struct ParameterGrid {
    params: Vec<ParamSet>,
}

impl ParameterGrid {
    /// Build a single-dict Cartesian-product parameter grid.
    ///
    /// The input is a list of `(name, values)` axes. Keys are sorted before
    /// enumeration so the sequence matches sklearn's `sorted(p.items())`
    /// order. An empty axis list yields one empty [`ParamSet`].
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::InvalidParameter`] when any axis has no values.
    pub fn new(mut axes: Vec<(String, Vec<ParamValue>)>) -> Result<Self, FerroError> {
        for (name, values) in &axes {
            if values.is_empty() {
                return Err(FerroError::InvalidParameter {
                    name: name.clone(),
                    reason: "parameter grid values must be a non-empty sequence".to_string(),
                });
            }
        }

        axes.sort_by(|a, b| a.0.cmp(&b.0));

        let mut result: Vec<ParamSet> = vec![ParamSet::new()];
        for (name, values) in axes {
            let mut next = Vec::with_capacity(result.len() * values.len());
            for existing in &result {
                for val in &values {
                    let mut entry = existing.clone();
                    entry.insert(name.clone(), val.clone());
                    next.push(entry);
                }
            }
            result = next;
        }

        Ok(Self { params: result })
    }

    /// Wrap already-materialized parameter combinations.
    pub fn from_params(params: Vec<ParamSet>) -> Self {
        Self { params }
    }

    /// Return the materialized combinations.
    pub fn as_slice(&self) -> &[ParamSet] {
        &self.params
    }

    /// Consume the grid and return the materialized combinations.
    pub fn into_vec(self) -> Vec<ParamSet> {
        self.params
    }

    /// Number of combinations.
    pub fn len(&self) -> usize {
        self.params.len()
    }

    /// Whether the materialized grid contains no combinations.
    pub fn is_empty(&self) -> bool {
        self.params.is_empty()
    }

    /// Iterate over the materialized combinations.
    pub fn iter(&self) -> std::slice::Iter<'_, ParamSet> {
        self.params.iter()
    }
}

impl IntoIterator for ParameterGrid {
    type Item = ParamSet;
    type IntoIter = std::vec::IntoIter<ParamSet>;

    fn into_iter(self) -> Self::IntoIter {
        self.params.into_iter()
    }
}

impl<'a> IntoIterator for &'a ParameterGrid {
    type Item = &'a ParamSet;
    type IntoIter = std::slice::Iter<'a, ParamSet>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl From<ParameterGrid> for Vec<ParamSet> {
    fn from(grid: ParameterGrid) -> Self {
        grid.into_vec()
    }
}

// ---------------------------------------------------------------------------
// param_grid! macro
// ---------------------------------------------------------------------------

/// Build a `Vec<ParamSet>` containing every Cartesian-product combination of
/// the supplied parameter lists.
///
/// Each entry in the macro is `"param_name" => [val1, val2, ...]`. The values
/// are automatically converted to [`ParamValue`] via the `Into<ParamValue>`
/// trait implementations defined in this module.
///
/// # Example
///
/// ```rust
/// use ferrolearn_model_sel::{param_grid, ParamValue};
///
/// let grid = param_grid! {
///     "alpha" => [0.01_f64, 0.1_f64, 1.0_f64],
///     "fit_intercept" => [true, false],
/// };
/// assert_eq!(grid.len(), 6); // 3 × 2
/// ```
///
/// The macro expands to code that iterates the Cartesian product of all lists
/// and collects the results into a `Vec<ParamSet>`.
#[macro_export]
macro_rules! param_grid {
    // Entry point: collect all (name, values) pairs into a vec, then
    // compute the Cartesian product.
    ( $( $key:expr => [ $( $val:expr ),* $(,)? ] ),* $(,)? ) => {{
        // Build a Vec of (name, Vec<ParamValue>) entries.
        let mut axes: Vec<(String, Vec<$crate::ParamValue>)> = vec![
            $(
                (
                    $key.to_string(),
                    vec![ $( $crate::param_grid!(@into $val) ),* ],
                )
            ),*
        ];

        // Sort axes by key name to match scikit-learn's `sorted(p.items())`
        // (sklearn/model_selection/_search.py:157) before the Cartesian
        // product, so the enumeration order matches `ParameterGrid`.
        axes.sort_by(|a, b| a.0.cmp(&b.0));

        // Compute Cartesian product iteratively.
        let mut result: Vec<$crate::ParamSet> = vec![$crate::ParamSet::new()];
        for (name, values) in axes {
            let mut next = Vec::with_capacity(result.len() * values.len());
            for existing in &result {
                for val in &values {
                    let mut entry = existing.clone();
                    entry.insert(name.clone(), val.clone());
                    next.push(entry);
                }
            }
            result = next;
        }
        result
    }};

    // Helper arm: convert a literal to ParamValue.
    (@into $val:expr) => {
        $crate::ParamValue::from($val)
    };
}

// ---------------------------------------------------------------------------
// From conversions for ParamValue
// ---------------------------------------------------------------------------

impl From<f64> for ParamValue {
    fn from(v: f64) -> Self {
        ParamValue::Float(v)
    }
}

impl From<f32> for ParamValue {
    fn from(v: f32) -> Self {
        ParamValue::Float(f64::from(v))
    }
}

impl From<i64> for ParamValue {
    fn from(v: i64) -> Self {
        ParamValue::Int(v)
    }
}

impl From<i32> for ParamValue {
    fn from(v: i32) -> Self {
        ParamValue::Int(i64::from(v))
    }
}

impl From<usize> for ParamValue {
    fn from(v: usize) -> Self {
        ParamValue::Int(v as i64)
    }
}

impl From<bool> for ParamValue {
    fn from(v: bool) -> Self {
        ParamValue::Bool(v)
    }
}

impl From<String> for ParamValue {
    fn from(v: String) -> Self {
        ParamValue::String(v)
    }
}

impl From<&str> for ParamValue {
    fn from(v: &str) -> Self {
        ParamValue::String(v.to_owned())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_param_grid_single_axis() {
        let grid = param_grid! {
            "alpha" => [0.01_f64, 0.1_f64, 1.0_f64],
        };
        assert_eq!(grid.len(), 3);
        for set in &grid {
            assert!(set.contains_key("alpha"));
        }
    }

    #[test]
    fn test_param_grid_two_axes_cartesian() {
        let grid = param_grid! {
            "alpha" => [0.01_f64, 0.1_f64, 1.0_f64],
            "fit_intercept" => [true, false],
        };
        // 3 × 2 = 6 combinations.
        assert_eq!(grid.len(), 6);
    }

    #[test]
    fn test_param_grid_three_axes() {
        let grid = param_grid! {
            "a" => [1_i64, 2_i64],
            "b" => [true, false],
            "c" => [0.5_f64, 1.0_f64, 2.0_f64],
        };
        // 2 × 2 × 3 = 12 combinations.
        assert_eq!(grid.len(), 12);
    }

    #[test]
    fn test_param_grid_all_keys_present() {
        let grid = param_grid! {
            "alpha" => [0.1_f64],
            "normalize" => [true, false],
        };
        for set in &grid {
            assert!(set.contains_key("alpha"), "missing 'alpha'");
            assert!(set.contains_key("normalize"), "missing 'normalize'");
        }
    }

    #[test]
    fn test_param_grid_values_correct() {
        let grid = param_grid! {
            "x" => [1_f64, 2_f64],
        };
        let vals: Vec<f64> = grid
            .iter()
            .map(|s| match s["x"] {
                ParamValue::Float(v) => v,
                _ => panic!("expected Float"),
            })
            .collect();
        assert!(vals.contains(&1.0));
        assert!(vals.contains(&2.0));
    }

    #[test]
    fn test_param_value_display() {
        assert_eq!(ParamValue::Float(1.5).to_string(), "1.5");
        assert_eq!(ParamValue::Int(42).to_string(), "42");
        assert_eq!(ParamValue::Bool(true).to_string(), "true");
        assert_eq!(ParamValue::String("foo".into()).to_string(), "foo");
    }

    #[test]
    fn test_param_value_from_conversions() {
        assert_eq!(ParamValue::from(1.0_f64), ParamValue::Float(1.0));
        assert_eq!(ParamValue::from(2_i64), ParamValue::Int(2));
        assert_eq!(ParamValue::from(true), ParamValue::Bool(true));
        assert_eq!(
            ParamValue::from("hello"),
            ParamValue::String("hello".into())
        );
    }
}
