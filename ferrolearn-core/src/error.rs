//! Error types for the ferrolearn framework.
//!
//! This module defines [`FerroError`], the unified error type used throughout
//! all ferrolearn crates. Each variant carries diagnostic context to help
//! users identify and fix problems.

#[cfg(not(feature = "std"))]
use alloc::{string::String, vec::Vec};

use core::fmt;

/// The unified error type for all ferrolearn operations.
///
/// Every public function in ferrolearn returns `Result<T, FerroError>`.
/// The enum is `#[non_exhaustive]` so that new variants can be added in
/// future minor releases without breaking downstream code.
///
/// # Examples
///
/// ```
/// use ferrolearn_core::FerroError;
///
/// let err = FerroError::ShapeMismatch {
///     expected: vec![100, 10],
///     actual: vec![100, 5],
///     context: "feature matrix".into(),
/// };
/// assert!(err.to_string().contains("Shape mismatch"));
/// ```
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum FerroError {
    /// Array dimensions do not match the expected shape.
    #[error("Shape mismatch in {context}: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        /// The expected dimensions.
        expected: Vec<usize>,
        /// The actual dimensions encountered.
        actual: Vec<usize>,
        /// Human-readable description of where the mismatch occurred.
        context: String,
    },

    /// Not enough samples were provided for the requested operation.
    #[error("Insufficient samples: need at least {required}, got {actual} ({context})")]
    InsufficientSamples {
        /// The minimum number of samples required.
        required: usize,
        /// The actual number of samples provided.
        actual: usize,
        /// Human-readable description of the operation.
        context: String,
    },

    /// An iterative algorithm did not converge within the allowed iterations.
    #[error("Convergence failure after {iterations} iterations: {message}")]
    ConvergenceFailure {
        /// The number of iterations that were attempted.
        iterations: usize,
        /// A description of the convergence issue.
        message: String,
    },

    /// A hyperparameter or configuration value is invalid.
    #[error("Invalid parameter `{name}`: {reason}")]
    InvalidParameter {
        /// The name of the parameter.
        name: String,
        /// Why the value is invalid.
        reason: String,
    },

    /// A numerical computation produced NaN, infinity, or other instability.
    #[error("Numerical instability: {message}")]
    NumericalInstability {
        /// A description of the numerical issue.
        message: String,
    },

    /// An I/O error occurred during data loading or model persistence.
    #[cfg(feature = "std")]
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    /// A serialization or deserialization error occurred.
    #[error("Serialization error: {message}")]
    SerdeError {
        /// A description of the serialization issue.
        message: String,
    },
}

/// A convenience type alias for `Result<T, FerroError>`.
pub type FerroResult<T> = Result<T, FerroError>;

/// Diagnostic context attached to shape-mismatch errors.
///
/// This struct provides a builder-style API for constructing
/// descriptive [`FerroError::ShapeMismatch`] errors.
///
/// # Examples
///
/// ```
/// use ferrolearn_core::error::ShapeMismatchContext;
///
/// let ctx = ShapeMismatchContext::new("predict input")
///     .expected(&[100, 10])
///     .actual(&[100, 5]);
/// let err = ctx.build();
/// assert!(err.to_string().contains("predict input"));
/// ```
#[derive(Debug, Clone)]
pub struct ShapeMismatchContext {
    context: String,
    expected: Vec<usize>,
    actual: Vec<usize>,
}

impl ShapeMismatchContext {
    /// Create a new context with the given description.
    pub fn new(context: impl Into<String>) -> Self {
        Self {
            context: context.into(),
            expected: Vec::new(),
            actual: Vec::new(),
        }
    }

    /// Set the expected shape.
    #[must_use]
    pub fn expected(mut self, shape: &[usize]) -> Self {
        self.expected = shape.to_vec();
        self
    }

    /// Set the actual shape.
    #[must_use]
    pub fn actual(mut self, shape: &[usize]) -> Self {
        self.actual = shape.to_vec();
        self
    }

    /// Build the [`FerroError::ShapeMismatch`] error.
    pub fn build(self) -> FerroError {
        FerroError::ShapeMismatch {
            expected: self.expected,
            actual: self.actual,
            context: self.context,
        }
    }
}

impl fmt::Display for ShapeMismatchContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ShapeMismatchContext({}, expected {:?}, actual {:?})",
            self.context, self.expected, self.actual
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shape_mismatch_display() {
        let err = FerroError::ShapeMismatch {
            expected: vec![100, 10],
            actual: vec![100, 5],
            context: "feature matrix".into(),
        };
        let msg = err.to_string();
        assert!(msg.contains("Shape mismatch"));
        assert!(msg.contains("feature matrix"));
        assert!(msg.contains("[100, 10]"));
        assert!(msg.contains("[100, 5]"));
    }

    #[test]
    fn test_insufficient_samples_display() {
        let err = FerroError::InsufficientSamples {
            required: 10,
            actual: 3,
            context: "cross-validation".into(),
        };
        let msg = err.to_string();
        assert!(msg.contains("10"));
        assert!(msg.contains("3"));
        assert!(msg.contains("cross-validation"));
    }

    #[test]
    fn test_convergence_failure_display() {
        let err = FerroError::ConvergenceFailure {
            iterations: 1000,
            message: "loss did not decrease".into(),
        };
        let msg = err.to_string();
        assert!(msg.contains("1000"));
        assert!(msg.contains("loss did not decrease"));
    }

    #[test]
    fn test_invalid_parameter_display() {
        let err = FerroError::InvalidParameter {
            name: "n_clusters".into(),
            reason: "must be positive".into(),
        };
        let msg = err.to_string();
        assert!(msg.contains("n_clusters"));
        assert!(msg.contains("must be positive"));
    }

    #[test]
    fn test_numerical_instability_display() {
        let err = FerroError::NumericalInstability {
            message: "matrix is singular".into(),
        };
        assert!(err.to_string().contains("matrix is singular"));
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_io_error_from() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let ferro_err: FerroError = io_err.into();
        assert!(ferro_err.to_string().contains("file not found"));
    }

    #[test]
    fn test_serde_error_display() {
        let err = FerroError::SerdeError {
            message: "invalid JSON".into(),
        };
        assert!(err.to_string().contains("invalid JSON"));
    }

    #[test]
    fn test_shape_mismatch_context_builder() {
        let err = ShapeMismatchContext::new("test context")
            .expected(&[3, 4])
            .actual(&[3, 5])
            .build();
        let msg = err.to_string();
        assert!(msg.contains("test context"));
        assert!(msg.contains("[3, 4]"));
        assert!(msg.contains("[3, 5]"));
    }

    #[test]
    fn test_ferro_error_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<FerroError>();
    }
}
