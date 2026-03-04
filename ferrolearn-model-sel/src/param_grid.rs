//! Parameter grid types and the [`param_grid!`] macro.
//!
//! This module defines:
//!
//! - [`ParamValue`] — a dynamically-typed hyperparameter value.
//! - [`ParamSet`] — a single parameter configuration (`HashMap<String, ParamValue>`).
//! - [`param_grid!`] — a macro that builds the Cartesian product of parameter
//!   lists as `Vec<ParamSet>`.
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

use std::collections::HashMap;

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
/// Created by [`param_grid!`] or built manually.
pub type ParamSet = HashMap<String, ParamValue>;

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
        let axes: Vec<(String, Vec<$crate::ParamValue>)> = vec![
            $(
                (
                    $key.to_string(),
                    vec![ $( $crate::param_grid!(@into $val) ),* ],
                )
            ),*
        ];

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
