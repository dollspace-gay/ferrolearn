//! DictVectorizer: dense vectorization of feature-value mappings.
//!
//! Translation target: scikit-learn 1.5.2 `sklearn.feature_extraction.DictVectorizer`
//! (`feature_extraction/_dict_vectorizer.py`). Design:
//! `.design/preprocess/dict_vectorizer.md`.
//!
//! `## REQ status`
//!
//! | REQ | Status | Anchor |
//! |---|---|---|
//! | REQ-1 numeric mappings | SHIPPED scoped | [`DictVectorizer::fit`] + [`FittedDictVectorizer::transform`] for `DictValue::Number` |
//! | REQ-2 string categorical values | SHIPPED scoped | `DictValue::Text` -> `feature=value` one-hot features |
//! | REQ-3 iterable string categorical values | SHIPPED scoped | `DictValue::Texts` counts repeated categorical values |
//! | REQ-4 feature names / vocabulary / inverse transform | SHIPPED scoped | [`FittedDictVectorizer::get_feature_names_out`], [`FittedDictVectorizer::vocabulary_map`], [`FittedDictVectorizer::inverse_transform`] |
//! | REQ-5 sparse CSR + dtype/Python protocol/restrict | NOT-STARTED | dense `Array2<f64>` only; no dtype/PyO3/metadata routing/restrict |

use ferrolearn_core::error::FerroError;
use ndarray::Array2;
use std::collections::HashMap;

/// Supported values inside a [`DictVectorizer`] sample mapping.
#[derive(Debug, Clone, PartialEq)]
pub enum DictValue {
    /// Numeric feature value. The output feature name is the original key.
    Number(f64),
    /// String categorical value. The output feature name is `key=value`.
    Text(String),
    /// Iterable of string categorical values. Repeated values are counted.
    Texts(Vec<String>),
    /// Missing value. Mirrors sklearn's `None` path as a numeric NaN value.
    None,
}

impl From<f64> for DictValue {
    fn from(value: f64) -> Self {
        Self::Number(value)
    }
}

impl From<&str> for DictValue {
    fn from(value: &str) -> Self {
        Self::Text(value.to_string())
    }
}

impl From<String> for DictValue {
    fn from(value: String) -> Self {
        Self::Text(value)
    }
}

/// Dense analogue of sklearn's `DictVectorizer`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DictVectorizer {
    /// Separator used for categorical `key=value` feature names.
    pub separator: String,
}

impl DictVectorizer {
    /// Create a new `DictVectorizer` with sklearn-like defaults.
    #[must_use]
    pub fn new() -> Self {
        Self {
            separator: "=".to_string(),
        }
    }

    /// Set the separator used for categorical feature names.
    #[must_use]
    pub fn separator(mut self, separator: impl Into<String>) -> Self {
        self.separator = separator.into();
        self
    }

    /// Fit the vectorizer on mapping-style samples.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::InvalidParameter`] when the sample slice is empty.
    pub fn fit(
        &self,
        samples: &[HashMap<String, DictValue>],
    ) -> Result<FittedDictVectorizer, FerroError> {
        if samples.is_empty() {
            return Err(empty_sequence_error());
        }

        let mut feature_names = Vec::new();
        let mut vocabulary = HashMap::new();
        for sample in samples {
            for (feature, value) in sample {
                for (feature_name, _) in self.entries(feature, value) {
                    if !vocabulary.contains_key(&feature_name) {
                        vocabulary.insert(feature_name.clone(), feature_names.len());
                        feature_names.push(feature_name);
                    }
                }
            }
        }

        feature_names.sort();
        vocabulary = feature_names
            .iter()
            .enumerate()
            .map(|(index, feature_name)| (feature_name.clone(), index))
            .collect();

        Ok(FittedDictVectorizer {
            vocabulary,
            feature_names,
            separator: self.separator.clone(),
        })
    }

    /// Fit and transform mapping-style samples in one call.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`Self::fit`] and
    /// [`FittedDictVectorizer::transform`].
    pub fn fit_transform(
        &self,
        samples: &[HashMap<String, DictValue>],
    ) -> Result<(FittedDictVectorizer, Array2<f64>), FerroError> {
        let fitted = self.fit(samples)?;
        let transformed = fitted.transform(samples)?;
        Ok((fitted, transformed))
    }

    fn entries(&self, feature: &str, value: &DictValue) -> Vec<(String, f64)> {
        match value {
            DictValue::Number(value) => vec![(feature.to_string(), *value)],
            DictValue::Text(value) => {
                vec![(format!("{}{}{}", feature, self.separator, value), 1.0)]
            }
            DictValue::Texts(values) => values
                .iter()
                .map(|value| (format!("{}{}{}", feature, self.separator, value), 1.0))
                .collect(),
            DictValue::None => vec![(feature.to_string(), f64::NAN)],
        }
    }
}

impl Default for DictVectorizer {
    fn default() -> Self {
        Self::new()
    }
}

/// A fitted [`DictVectorizer`] with learned feature names and vocabulary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FittedDictVectorizer {
    vocabulary: HashMap<String, usize>,
    feature_names: Vec<String>,
    separator: String,
}

impl FittedDictVectorizer {
    /// Return transformed feature names in column order.
    #[must_use]
    pub fn get_feature_names_out(&self) -> &[String] {
        &self.feature_names
    }

    /// Return transformed feature names in column order.
    #[must_use]
    pub fn feature_names(&self) -> &[String] {
        &self.feature_names
    }

    /// Return the vocabulary mapping from feature name to column index.
    #[must_use]
    pub fn vocabulary_map(&self) -> &HashMap<String, usize> {
        &self.vocabulary
    }

    /// Transform mapping-style samples into a dense feature matrix.
    ///
    /// Unseen feature names and unseen categorical values are silently ignored,
    /// matching sklearn's transform behavior.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::InvalidParameter`] when the sample slice is empty.
    pub fn transform(
        &self,
        samples: &[HashMap<String, DictValue>],
    ) -> Result<Array2<f64>, FerroError> {
        if samples.is_empty() {
            return Err(empty_sequence_error());
        }

        let mut matrix = Array2::<f64>::zeros((samples.len(), self.feature_names.len()));
        for (row, sample) in samples.iter().enumerate() {
            for (feature, value) in sample {
                for (feature_name, numeric_value) in self.entries(feature, value) {
                    if let Some(&column) = self.vocabulary.get(&feature_name) {
                        matrix[[row, column]] += numeric_value;
                    }
                }
            }
        }
        Ok(matrix)
    }

    /// Invert a dense feature matrix back to non-zero feature mappings.
    ///
    /// For one-hot categorical values, sklearn returns constructed feature
    /// names such as `"city=Dubai"` as keys; this method follows that contract.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] when the matrix width differs from
    /// the fitted feature count.
    pub fn inverse_transform(
        &self,
        matrix: &Array2<f64>,
    ) -> Result<Vec<HashMap<String, f64>>, FerroError> {
        if matrix.ncols() != self.feature_names.len() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![matrix.nrows(), self.feature_names.len()],
                actual: vec![matrix.nrows(), matrix.ncols()],
                context: "DictVectorizer::inverse_transform".into(),
            });
        }

        let mut rows = Vec::with_capacity(matrix.nrows());
        for row in matrix.rows() {
            let mut sample = HashMap::new();
            for (column, value) in row.iter().enumerate() {
                if *value != 0.0 {
                    sample.insert(self.feature_names[column].clone(), *value);
                }
            }
            rows.push(sample);
        }
        Ok(rows)
    }

    fn entries(&self, feature: &str, value: &DictValue) -> Vec<(String, f64)> {
        match value {
            DictValue::Number(value) => vec![(feature.to_string(), *value)],
            DictValue::Text(value) => {
                vec![(format!("{}{}{}", feature, self.separator, value), 1.0)]
            }
            DictValue::Texts(values) => values
                .iter()
                .map(|value| (format!("{}{}{}", feature, self.separator, value), 1.0))
                .collect(),
            DictValue::None => vec![(feature.to_string(), f64::NAN)],
        }
    }
}

fn empty_sequence_error() -> FerroError {
    FerroError::InvalidParameter {
        name: "X".into(),
        reason: "Sample sequence X is empty.".into(),
    }
}
