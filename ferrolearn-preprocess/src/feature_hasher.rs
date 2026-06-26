//! FeatureHasher: dense feature hashing for symbolic feature names.
//!
//! Translation target: scikit-learn 1.5.2 `sklearn.feature_extraction.FeatureHasher`
//! (`feature_extraction/_hash.py`). Design: `.design/preprocess/feature_hasher.md`.
//!
//! `## REQ status`
//!
//! | REQ | Status | Anchor |
//! |---|---|---|
//! | REQ-1 dict input hashing | SHIPPED scoped | [`FeatureHasher::transform_dict`] hashes `HashMap<String, f64>` samples with signed MurmurHash3 x86-32 |
//! | REQ-2 pair input hashing | SHIPPED scoped | [`FeatureHasher::transform_pairs`] hashes `(String, f64)` sample pairs |
//! | REQ-3 string input hashing | SHIPPED scoped | [`FeatureHasher::transform_strings`] hashes string samples with implied value 1 |
//! | REQ-4 parameter validation | SHIPPED scoped | `n_features` range and input-type consistency checks |
//! | REQ-5 sparse CSR output + dtype/Python protocol | NOT-STARTED | dense `Array2<f64>` only; no dtype/PyO3/metadata routing |

use crate::hashing::{murmurhash3_32_signed, signed_hash_index};
use ferrolearn_core::error::FerroError;
use ndarray::Array2;
use std::collections::HashMap;

/// Input representation selected for [`FeatureHasher`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FeatureHasherInputType {
    /// Samples are mappings from feature name to numeric value.
    #[default]
    Dict,
    /// Samples are iterables of `(feature name, value)` pairs.
    Pair,
    /// Samples are iterables of feature names with implied value 1.
    String,
}

/// Stateless feature hasher using signed 32-bit MurmurHash3.
///
/// This is a dense Rust analogue of sklearn's `FeatureHasher`. It preserves the
/// hashed-column and `alternate_sign` semantics for string feature names, while
/// returning `Array2<f64>` instead of sklearn's sparse CSR matrix.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FeatureHasher {
    /// Number of hashed output columns.
    pub n_features: usize,
    /// Expected input representation.
    pub input_type: FeatureHasherInputType,
    /// If `true`, negative MurmurHash3 values flip the feature value sign.
    pub alternate_sign: bool,
}

impl FeatureHasher {
    /// Create a new `FeatureHasher` with sklearn-like defaults.
    #[must_use]
    pub fn new() -> Self {
        Self {
            n_features: 1 << 20,
            input_type: FeatureHasherInputType::Dict,
            alternate_sign: true,
        }
    }

    /// Set the number of hashed output columns.
    #[must_use]
    pub fn n_features(mut self, n_features: usize) -> Self {
        self.n_features = n_features;
        self
    }

    /// Set the expected input representation.
    #[must_use]
    pub fn input_type(mut self, input_type: FeatureHasherInputType) -> Self {
        self.input_type = input_type;
        self
    }

    /// Enable or disable alternating signs.
    #[must_use]
    pub fn alternate_sign(mut self, alternate_sign: bool) -> Self {
        self.alternate_sign = alternate_sign;
        self
    }

    /// Stateless fit: validate parameters and return `self`.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::InvalidParameter`] when `n_features` is outside
    /// sklearn's accepted `fit` range `[1, 2147483647]`.
    pub fn fit(&self) -> Result<Self, FerroError> {
        self.validate_n_features()?;
        Ok(self.clone())
    }

    /// Transform mapping-style samples into a dense hashed feature matrix.
    ///
    /// # Errors
    ///
    /// Returns an error for empty sample slices, invalid `n_features`, or an
    /// `input_type` other than [`FeatureHasherInputType::Dict`].
    pub fn transform_dict(
        &self,
        samples: &[HashMap<String, f64>],
    ) -> Result<Array2<f64>, FerroError> {
        self.validate_for(FeatureHasherInputType::Dict)?;
        let mut matrix = self.empty_output(samples.len())?;
        for (row, sample) in samples.iter().enumerate() {
            for (feature, value) in sample {
                self.add_feature(&mut matrix, row, feature, *value);
            }
        }
        Ok(matrix)
    }

    /// Transform pair-style samples into a dense hashed feature matrix.
    ///
    /// # Errors
    ///
    /// Returns an error for empty sample slices, invalid `n_features`, or an
    /// `input_type` other than [`FeatureHasherInputType::Pair`].
    pub fn transform_pairs(
        &self,
        samples: &[Vec<(String, f64)>],
    ) -> Result<Array2<f64>, FerroError> {
        self.validate_for(FeatureHasherInputType::Pair)?;
        let mut matrix = self.empty_output(samples.len())?;
        for (row, sample) in samples.iter().enumerate() {
            for (feature, value) in sample {
                self.add_feature(&mut matrix, row, feature, *value);
            }
        }
        Ok(matrix)
    }

    /// Transform string-style samples into a dense hashed feature matrix.
    ///
    /// # Errors
    ///
    /// Returns an error for empty sample slices, invalid `n_features`, or an
    /// `input_type` other than [`FeatureHasherInputType::String`].
    pub fn transform_strings(&self, samples: &[Vec<String>]) -> Result<Array2<f64>, FerroError> {
        self.validate_for(FeatureHasherInputType::String)?;
        let mut matrix = self.empty_output(samples.len())?;
        for (row, sample) in samples.iter().enumerate() {
            for feature in sample {
                self.add_feature(&mut matrix, row, feature, 1.0);
            }
        }
        Ok(matrix)
    }

    /// Fit and transform mapping-style samples.
    ///
    /// # Errors
    ///
    /// Propagates validation errors from [`Self::fit`] and
    /// [`Self::transform_dict`].
    pub fn fit_transform_dict(
        &self,
        samples: &[HashMap<String, f64>],
    ) -> Result<Array2<f64>, FerroError> {
        self.fit()?.transform_dict(samples)
    }

    /// Fit and transform pair-style samples.
    ///
    /// # Errors
    ///
    /// Propagates validation errors from [`Self::fit`] and
    /// [`Self::transform_pairs`].
    pub fn fit_transform_pairs(
        &self,
        samples: &[Vec<(String, f64)>],
    ) -> Result<Array2<f64>, FerroError> {
        self.fit()?.transform_pairs(samples)
    }

    /// Fit and transform string-style samples.
    ///
    /// # Errors
    ///
    /// Propagates validation errors from [`Self::fit`] and
    /// [`Self::transform_strings`].
    pub fn fit_transform_strings(
        &self,
        samples: &[Vec<String>],
    ) -> Result<Array2<f64>, FerroError> {
        self.fit()?.transform_strings(samples)
    }

    fn validate_for(&self, expected: FeatureHasherInputType) -> Result<(), FerroError> {
        self.validate_n_features()?;
        if self.input_type != expected {
            return Err(FerroError::InvalidParameter {
                name: "input_type".into(),
                reason: format!("expected {:?}, got {:?}", expected, self.input_type),
            });
        }
        Ok(())
    }

    fn validate_n_features(&self) -> Result<(), FerroError> {
        if self.n_features == 0 || self.n_features > i32::MAX as usize {
            return Err(FerroError::InvalidParameter {
                name: "n_features".into(),
                reason: "must be in [1, 2147483647]".into(),
            });
        }
        Ok(())
    }

    fn empty_output(&self, n_samples: usize) -> Result<Array2<f64>, FerroError> {
        if n_samples == 0 {
            return Err(FerroError::InvalidParameter {
                name: "raw_X".into(),
                reason: "Cannot vectorize empty sequence.".into(),
            });
        }
        Ok(Array2::<f64>::zeros((n_samples, self.n_features)))
    }

    fn add_feature(&self, matrix: &mut Array2<f64>, row: usize, feature: &str, value: f64) {
        let hash = murmurhash3_32_signed(feature.as_bytes(), 0);
        let col = signed_hash_index(hash, self.n_features);
        let sign = if self.alternate_sign && hash < 0 {
            -1.0
        } else {
            1.0
        };
        matrix[[row, col]] += sign * value;
    }
}

impl Default for FeatureHasher {
    fn default() -> Self {
        Self::new()
    }
}
