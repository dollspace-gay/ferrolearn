//! PMML (Predictive Model Markup Language) export for ferrolearn fitted models.
//!
//! This module provides the [`ToPmml`] trait and supporting functions for
//! converting fitted linear models into PMML 4.4 XML format.
//!
//! PMML is an XML-based interchange format for predictive models, widely
//! supported by data mining tools and scoring engines.
//!
//! # Supported Models
//!
//! - [`FittedLinearRegression`](ferrolearn_linear::FittedLinearRegression) — PMML `RegressionModel`
//! - [`FittedRidge`](ferrolearn_linear::FittedRidge) — PMML `RegressionModel`
//!
//! # Example
//!
//! ```no_run
//! use ferrolearn_io::pmml::{ToPmml, save_pmml};
//! use ferrolearn_linear::LinearRegression;
//! use ferrolearn_core::Fit;
//! use ndarray::{Array2, array};
//!
//! let x = Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
//! let y = array![1.0, 2.0, 3.0, 4.0];
//! let model = LinearRegression::<f64>::new();
//! let fitted = model.fit(&x, &y).unwrap();
//!
//! let pmml_xml = fitted.to_pmml().unwrap();
//! save_pmml(&pmml_xml, "/tmp/model.pmml").unwrap();
//! ```

use std::fs;
use std::path::Path;

use ferrolearn_core::error::FerroError;
use ferrolearn_core::introspection::HasCoefficients;
use num_traits::Float;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Trait for converting a fitted model into a PMML XML string.
pub trait ToPmml {
    /// Convert this fitted model to a PMML 4.4 XML string.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::SerdeError`] if the conversion fails.
    fn to_pmml(&self) -> Result<String, FerroError>;
}

/// Write a PMML XML string to a file.
///
/// # Errors
///
/// Returns [`FerroError::IoError`] if the file cannot be written.
pub fn save_pmml(xml: &str, path: impl AsRef<Path>) -> Result<(), FerroError> {
    fs::write(path, xml).map_err(FerroError::IoError)
}

// ---------------------------------------------------------------------------
// XML helpers
// ---------------------------------------------------------------------------

/// Escape a string for safe inclusion in XML text content.
fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

// ---------------------------------------------------------------------------
// ToPmml for linear regression models
// ---------------------------------------------------------------------------

/// Generate a PMML 4.4 `RegressionModel` for a linear model.
///
/// The model has `n_features` input fields named `x0`, `x1`, etc.
/// The output field is named `y`.
fn linear_model_to_pmml<F: Float>(
    model: &dyn HasCoefficients<F>,
    model_name: &str,
) -> Result<String, FerroError> {
    let coefficients = model.coefficients();
    let intercept = model.intercept();
    let n_features = coefficients.len();

    let intercept_f64 = intercept.to_f64().ok_or_else(|| FerroError::SerdeError {
        message: "failed to convert intercept to f64".to_owned(),
    })?;

    let mut xml = String::new();

    // XML header
    xml.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    xml.push_str("<PMML xmlns=\"http://www.dmg.org/PMML-4_4\" version=\"4.4\">\n");

    // Header
    xml.push_str("  <Header copyright=\"ferrolearn\">\n");
    xml.push_str("    <Application name=\"ferrolearn\" />\n");
    xml.push_str("  </Header>\n");

    // DataDictionary
    xml.push_str("  <DataDictionary>\n");
    for i in 0..n_features {
        xml.push_str(&format!(
            "    <DataField name=\"x{i}\" optype=\"continuous\" dataType=\"double\" />\n"
        ));
    }
    xml.push_str("    <DataField name=\"y\" optype=\"continuous\" dataType=\"double\" />\n");
    xml.push_str("  </DataDictionary>\n");

    // RegressionModel
    xml.push_str(&format!(
        "  <RegressionModel modelName=\"{}\" functionName=\"regression\" algorithmName=\"{}\">\n",
        xml_escape(model_name),
        xml_escape(model_name)
    ));

    // MiningSchema
    xml.push_str("    <MiningSchema>\n");
    for i in 0..n_features {
        xml.push_str(&format!(
            "      <MiningField name=\"x{i}\" usageType=\"active\" />\n"
        ));
    }
    xml.push_str("      <MiningField name=\"y\" usageType=\"predicted\" />\n");
    xml.push_str("    </MiningSchema>\n");

    // RegressionTable
    xml.push_str(&format!(
        "    <RegressionTable intercept=\"{intercept_f64}\">\n"
    ));
    for i in 0..n_features {
        let coef_f64 = coefficients[i]
            .to_f64()
            .ok_or_else(|| FerroError::SerdeError {
                message: format!("failed to convert coefficient {i} to f64"),
            })?;
        xml.push_str(&format!(
            "      <NumericPredictor name=\"x{i}\" coefficient=\"{coef_f64}\" />\n"
        ));
    }
    xml.push_str("    </RegressionTable>\n");
    xml.push_str("  </RegressionModel>\n");
    xml.push_str("</PMML>\n");

    Ok(xml)
}

impl ToPmml for ferrolearn_linear::FittedLinearRegression<f64> {
    fn to_pmml(&self) -> Result<String, FerroError> {
        linear_model_to_pmml(self, "LinearRegression")
    }
}

impl ToPmml for ferrolearn_linear::FittedLinearRegression<f32> {
    fn to_pmml(&self) -> Result<String, FerroError> {
        linear_model_to_pmml(self, "LinearRegression")
    }
}

impl ToPmml for ferrolearn_linear::FittedRidge<f64> {
    fn to_pmml(&self) -> Result<String, FerroError> {
        linear_model_to_pmml(self, "Ridge")
    }
}

impl ToPmml for ferrolearn_linear::FittedRidge<f32> {
    fn to_pmml(&self) -> Result<String, FerroError> {
        linear_model_to_pmml(self, "Ridge")
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ferrolearn_core::traits::Fit;
    use ndarray::{Array2, array};

    #[test]
    fn test_linear_regression_to_pmml() {
        let x = Array2::from_shape_vec(
            (5, 2),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        )
        .unwrap();
        let y = array![5.0, 11.0, 17.0, 23.0, 29.0];

        let model = ferrolearn_linear::LinearRegression::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();

        let pmml = fitted.to_pmml().unwrap();

        // Check XML structure
        assert!(pmml.starts_with("<?xml"));
        assert!(pmml.contains("<PMML"));
        assert!(pmml.contains("version=\"4.4\""));
        assert!(pmml.contains("<RegressionModel"));
        assert!(pmml.contains("functionName=\"regression\""));
        assert!(pmml.contains("<DataDictionary>"));
        assert!(pmml.contains("<MiningSchema>"));
        assert!(pmml.contains("<RegressionTable"));
        assert!(pmml.contains("<NumericPredictor"));
        assert!(pmml.contains("name=\"x0\""));
        assert!(pmml.contains("name=\"x1\""));
        assert!(pmml.contains("</PMML>"));
    }

    #[test]
    fn test_linear_regression_pmml_coefficients() {
        // y = 2*x + 1
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = array![3.0, 5.0, 7.0, 9.0, 11.0];

        let model = ferrolearn_linear::LinearRegression::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();

        let pmml = fitted.to_pmml().unwrap();

        // Check that the intercept and coefficient appear
        // The model should learn approximately y = 2*x + 1
        assert!(pmml.contains("intercept=\""), "missing intercept attribute");
        assert!(
            pmml.contains("coefficient=\""),
            "missing coefficient attribute"
        );

        // Parse the coefficient value from the XML
        let coef_start = pmml.find("coefficient=\"").unwrap() + "coefficient=\"".len();
        let coef_end = pmml[coef_start..].find('"').unwrap() + coef_start;
        let coef_str = &pmml[coef_start..coef_end];
        let coef: f64 = coef_str.parse().unwrap();
        assert!(
            (coef - 2.0).abs() < 1e-6,
            "expected coefficient ~2.0, got {coef}"
        );

        // Parse the intercept value
        let int_start = pmml.find("intercept=\"").unwrap() + "intercept=\"".len();
        let int_end = pmml[int_start..].find('"').unwrap() + int_start;
        let int_str = &pmml[int_start..int_end];
        let intercept: f64 = int_str.parse().unwrap();
        assert!(
            (intercept - 1.0).abs() < 1e-6,
            "expected intercept ~1.0, got {intercept}"
        );
    }

    #[test]
    fn test_ridge_to_pmml() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![2.0, 4.0, 6.0, 8.0];

        let model = ferrolearn_linear::Ridge::<f64>::new().with_alpha(0.01);
        let fitted = model.fit(&x, &y).unwrap();

        let pmml = fitted.to_pmml().unwrap();

        assert!(pmml.contains("<RegressionModel"));
        assert!(pmml.contains("algorithmName=\"Ridge\""));
        assert!(pmml.contains("<NumericPredictor"));
    }

    #[test]
    fn test_linear_regression_f32_to_pmml() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let y = ndarray::Array1::from_vec(vec![2.0f32, 4.0, 6.0, 8.0]);

        let model = ferrolearn_linear::LinearRegression::<f32>::new();
        let fitted = model.fit(&x, &y).unwrap();

        let pmml = fitted.to_pmml().unwrap();
        assert!(pmml.contains("<PMML"));
        assert!(pmml.contains("<NumericPredictor"));
    }

    #[test]
    fn test_pmml_is_valid_xml_structure() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = array![1.0, 2.0, 3.0];

        let model = ferrolearn_linear::LinearRegression::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();
        let pmml = fitted.to_pmml().unwrap();

        // Verify all tags are properly opened and closed
        let open_pmml = pmml.matches("<PMML").count();
        let close_pmml = pmml.matches("</PMML>").count();
        assert_eq!(open_pmml, 1);
        assert_eq!(close_pmml, 1);

        let open_dd = pmml.matches("<DataDictionary>").count();
        let close_dd = pmml.matches("</DataDictionary>").count();
        assert_eq!(open_dd, 1);
        assert_eq!(close_dd, 1);

        let open_ms = pmml.matches("<MiningSchema>").count();
        let close_ms = pmml.matches("</MiningSchema>").count();
        assert_eq!(open_ms, 1);
        assert_eq!(close_ms, 1);

        // 2 features -> 2 NumericPredictor elements
        let predictors = pmml.matches("<NumericPredictor").count();
        assert_eq!(predictors, 2);
    }

    #[test]
    fn test_save_pmml_file() {
        let dir = tempdir();
        let path = dir.join("model.pmml");

        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![2.0, 4.0, 6.0, 8.0];

        let model = ferrolearn_linear::LinearRegression::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();
        let pmml = fitted.to_pmml().unwrap();

        save_pmml(&pmml, &path).unwrap();

        let contents = std::fs::read_to_string(&path).unwrap();
        assert!(contents.starts_with("<?xml"));
        assert!(contents.contains("<PMML"));
    }

    #[test]
    fn test_xml_escape() {
        assert_eq!(xml_escape("hello"), "hello");
        assert_eq!(xml_escape("a<b"), "a&lt;b");
        assert_eq!(xml_escape("a&b"), "a&amp;b");
        assert_eq!(xml_escape("a\"b"), "a&quot;b");
    }

    fn tempdir() -> std::path::PathBuf {
        use std::time::{SystemTime, UNIX_EPOCH};
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .subsec_nanos();
        let dir = std::env::temp_dir().join(format!("ferrolearn_pmml_test_{nanos}"));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }
}
