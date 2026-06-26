//! Polynomial kernel approximation via Tensor Sketch.
//!
//! [`PolynomialCountSketch`] mirrors sklearn's
//! `sklearn.kernel_approximation.PolynomialCountSketch`: it hashes signed
//! copies of `sqrt(gamma) * X` (plus an optional `sqrt(coef0)` bias feature)
//! and multiplies the count sketches by circular convolution. sklearn performs
//! that multiplication with FFTs; this implementation uses direct circular
//! convolution to keep the dependency surface small while preserving the dense
//! transform formula.
//!
//! Exact `indexHash_` / `bitHash_` values are an RNG-substrate gap (Xoshiro vs
//! numpy `RandomState`), matching the existing random sampler carve-outs.

use ferrolearn_core::error::FerroError;
use ferrolearn_core::traits::{Fit, Transform};
use ndarray::{Array1, Array2};
use num_traits::Float;
use rand::{Rng, SeedableRng};

/// Polynomial kernel approximation via Tensor Sketch.
#[derive(Debug, Clone)]
pub struct PolynomialCountSketch<F> {
    /// Polynomial kernel multiplier. Default: 1.0.
    gamma: F,
    /// Polynomial degree. Default: 2.
    degree: usize,
    /// Polynomial kernel constant term. Default: 0.0.
    coef0: F,
    /// Output dimensionality. Default: 100.
    n_components: usize,
    /// Optional random seed.
    random_state: Option<u64>,
}

impl<F: Float + Send + Sync + 'static> PolynomialCountSketch<F> {
    /// Create a new sampler with sklearn defaults.
    #[must_use]
    pub fn new() -> Self {
        Self {
            gamma: F::one(),
            degree: 2,
            coef0: F::zero(),
            n_components: 100,
            random_state: None,
        }
    }

    /// Set the polynomial kernel multiplier.
    #[must_use]
    pub fn with_gamma(mut self, gamma: F) -> Self {
        self.gamma = gamma;
        self
    }

    /// Set the polynomial degree.
    #[must_use]
    pub fn with_degree(mut self, degree: usize) -> Self {
        self.degree = degree;
        self
    }

    /// Set the polynomial kernel constant term.
    #[must_use]
    pub fn with_coef0(mut self, coef0: F) -> Self {
        self.coef0 = coef0;
        self
    }

    /// Set the output dimensionality.
    #[must_use]
    pub fn with_n_components(mut self, n_components: usize) -> Self {
        self.n_components = n_components;
        self
    }

    /// Set the random seed.
    #[must_use]
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Return the configured `gamma`.
    #[must_use]
    pub fn gamma(&self) -> F {
        self.gamma
    }

    /// Return the configured degree.
    #[must_use]
    pub fn degree(&self) -> usize {
        self.degree
    }

    /// Return the configured `coef0`.
    #[must_use]
    pub fn coef0(&self) -> F {
        self.coef0
    }

    /// Return the configured output dimensionality.
    #[must_use]
    pub fn n_components(&self) -> usize {
        self.n_components
    }
}

impl<F: Float + Send + Sync + 'static> Default for PolynomialCountSketch<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted Tensor Sketch polynomial feature map.
#[derive(Debug, Clone)]
pub struct FittedPolynomialCountSketch<F> {
    index_hash: Array2<usize>,
    bit_hash: Array2<F>,
    gamma: F,
    degree: usize,
    coef0: F,
    n_components: usize,
    n_features_in: usize,
}

impl<F: Float + Send + Sync + 'static> FittedPolynomialCountSketch<F> {
    /// Index hash array of shape `(degree, n_features [+ bias])`.
    #[must_use]
    pub fn index_hash(&self) -> &Array2<usize> {
        &self.index_hash
    }

    /// Sign hash array of shape `(degree, n_features [+ bias])`.
    #[must_use]
    pub fn bit_hash(&self) -> &Array2<F> {
        &self.bit_hash
    }

    /// Number of features observed during fitting, excluding the optional bias.
    #[must_use]
    pub fn n_features_in(&self) -> usize {
        self.n_features_in
    }

    /// Output dimensionality.
    #[must_use]
    pub fn n_components(&self) -> usize {
        self.n_components
    }

    /// Polynomial degree used by the transform.
    #[must_use]
    pub fn degree(&self) -> usize {
        self.degree
    }

    /// Polynomial kernel multiplier used by the transform.
    #[must_use]
    pub fn gamma(&self) -> F {
        self.gamma
    }

    /// Polynomial kernel constant term used by the transform.
    #[must_use]
    pub fn coef0(&self) -> F {
        self.coef0
    }
}

fn validate_finite_matrix<F: Float>(x: &Array2<F>, context: &str) -> Result<(), FerroError> {
    for &value in x {
        if !value.is_finite() {
            return Err(FerroError::InvalidParameter {
                name: "X".into(),
                reason: format!("{context} requires finite input"),
            });
        }
    }
    Ok(())
}

fn add_feature_to_count_sketches<F: Float>(
    sketches: &mut Array2<F>,
    index_hash: &Array2<usize>,
    bit_hash: &Array2<F>,
    feature_idx: usize,
    value: F,
) {
    if value == F::zero() {
        return;
    }
    for degree_idx in 0..sketches.nrows() {
        let component_idx = index_hash[[degree_idx, feature_idx]];
        sketches[[degree_idx, component_idx]] =
            sketches[[degree_idx, component_idx]] + bit_hash[[degree_idx, feature_idx]] * value;
    }
}

fn circular_convolve<F: Float>(left: &Array1<F>, right: &Array1<F>) -> Array1<F> {
    let n = left.len();
    let mut out = Array1::<F>::zeros(n);
    for (i, &left_value) in left.iter().enumerate() {
        if left_value == F::zero() {
            continue;
        }
        for (j, &right_value) in right.iter().enumerate() {
            if right_value == F::zero() {
                continue;
            }
            let out_idx = (i + j) % n;
            out[out_idx] = out[out_idx] + left_value * right_value;
        }
    }
    out
}

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, ()> for PolynomialCountSketch<F> {
    type Fitted = FittedPolynomialCountSketch<F>;
    type Error = FerroError;

    /// Fit by sampling index and sign hash arrays.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::InvalidParameter`] for invalid hyperparameters,
    /// non-finite input, or zero input features. Returns
    /// [`FerroError::InsufficientSamples`] when no rows are provided.
    fn fit(&self, x: &Array2<F>, _y: &()) -> Result<FittedPolynomialCountSketch<F>, FerroError> {
        if !self.gamma.is_finite() || self.gamma < F::zero() {
            return Err(FerroError::InvalidParameter {
                name: "gamma".into(),
                reason: "must be finite and non-negative".into(),
            });
        }
        if self.degree == 0 {
            return Err(FerroError::InvalidParameter {
                name: "degree".into(),
                reason: "must be at least 1".into(),
            });
        }
        if !self.coef0.is_finite() {
            return Err(FerroError::InvalidParameter {
                name: "coef0".into(),
                reason: "must be finite".into(),
            });
        }
        if self.n_components == 0 {
            return Err(FerroError::InvalidParameter {
                name: "n_components".into(),
                reason: "must be at least 1".into(),
            });
        }
        if x.nrows() == 0 {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "PolynomialCountSketch::fit".into(),
            });
        }
        if x.ncols() == 0 {
            return Err(FerroError::InvalidParameter {
                name: "X".into(),
                reason: "must contain at least one feature".into(),
            });
        }
        validate_finite_matrix(x, "PolynomialCountSketch::fit")?;

        let n_features = if self.coef0 != F::zero() {
            x.ncols()
                .checked_add(1)
                .ok_or_else(|| FerroError::ShapeMismatch {
                    expected: vec![x.nrows(), x.ncols()],
                    actual: vec![x.nrows(), usize::MAX],
                    context: "PolynomialCountSketch::fit bias feature count overflow".into(),
                })?
        } else {
            x.ncols()
        };
        let n_hashes =
            self.degree
                .checked_mul(n_features)
                .ok_or_else(|| FerroError::ShapeMismatch {
                    expected: vec![self.degree, n_features],
                    actual: vec![usize::MAX],
                    context: "PolynomialCountSketch::fit hash matrix size overflow".into(),
                })?;

        let mut rng = match self.random_state {
            Some(seed) => rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(seed),
            None => rand_xoshiro::Xoshiro256PlusPlus::from_os_rng(),
        };
        let mut index_data = Vec::with_capacity(n_hashes);
        let mut bit_data = Vec::with_capacity(n_hashes);
        for _ in 0..n_hashes {
            index_data.push(rng.random_range(0..self.n_components));
            let sign = if rng.random_range(0..2) == 0 {
                -F::one()
            } else {
                F::one()
            };
            bit_data.push(sign);
        }

        let index_hash =
            Array2::from_shape_vec((self.degree, n_features), index_data).map_err(|e| {
                FerroError::NumericalInstability {
                    message: format!("failed to create index hash matrix: {e}"),
                }
            })?;
        let bit_hash =
            Array2::from_shape_vec((self.degree, n_features), bit_data).map_err(|e| {
                FerroError::NumericalInstability {
                    message: format!("failed to create bit hash matrix: {e}"),
                }
            })?;

        Ok(FittedPolynomialCountSketch {
            index_hash,
            bit_hash,
            gamma: self.gamma,
            degree: self.degree,
            coef0: self.coef0,
            n_components: self.n_components,
            n_features_in: x.ncols(),
        })
    }
}

impl<F: Float + Send + Sync + 'static> Transform<Array2<F>> for FittedPolynomialCountSketch<F> {
    type Output = Array2<F>;
    type Error = FerroError;

    /// Generate the Tensor Sketch feature map.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] when feature counts differ from
    /// fit time, [`FerroError::InvalidParameter`] for non-finite input, and
    /// [`FerroError::InsufficientSamples`] for zero-row input.
    fn transform(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        if x.nrows() == 0 {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "FittedPolynomialCountSketch::transform".into(),
            });
        }
        if x.ncols() != self.n_features_in {
            return Err(FerroError::ShapeMismatch {
                expected: vec![x.nrows(), self.n_features_in],
                actual: vec![x.nrows(), x.ncols()],
                context: "FittedPolynomialCountSketch::transform feature count must match fit data"
                    .into(),
            });
        }
        validate_finite_matrix(x, "FittedPolynomialCountSketch::transform")?;

        let n_components = self.n_components;
        let mut out = Array2::<F>::zeros((x.nrows(), n_components));
        let sqrt_gamma = self.gamma.sqrt();
        let bias_value = (self.coef0 != F::zero()).then(|| self.coef0.sqrt());

        for sample_idx in 0..x.nrows() {
            let mut count_sketches = Array2::<F>::zeros((self.degree, n_components));
            for feature_idx in 0..self.n_features_in {
                add_feature_to_count_sketches(
                    &mut count_sketches,
                    &self.index_hash,
                    &self.bit_hash,
                    feature_idx,
                    sqrt_gamma * x[[sample_idx, feature_idx]],
                );
            }
            if let Some(bias) = bias_value {
                add_feature_to_count_sketches(
                    &mut count_sketches,
                    &self.index_hash,
                    &self.bit_hash,
                    self.n_features_in,
                    bias,
                );
            }

            let mut sketch_product = count_sketches.row(0).to_owned();
            for degree_idx in 1..self.degree {
                sketch_product =
                    circular_convolve(&sketch_product, &count_sketches.row(degree_idx).to_owned());
            }
            out.row_mut(sample_idx).assign(&sketch_product);
        }

        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::{Array2, array};

    #[test]
    fn defaults_match_sklearn() {
        let sampler = PolynomialCountSketch::<f64>::new();
        assert_abs_diff_eq!(sampler.gamma(), 1.0, epsilon = 1e-15);
        assert_eq!(sampler.degree(), 2);
        assert_abs_diff_eq!(sampler.coef0(), 0.0, epsilon = 1e-15);
        assert_eq!(sampler.n_components(), 100);
    }

    #[test]
    fn transform_formula_matches_sklearn_with_fixed_hashes() {
        // sklearn 1.5.2, /tmp:
        // X = [[1,2],[-1,0.5]]
        // PolynomialCountSketch(gamma=0.25, degree=3, coef0=2.25,
        //                       n_components=5)
        // indexHash_ = [[0,1,2],[2,3,4],[1,0,3]]
        // bitHash_ = [[1,-1,1],[-1,1,-1],[1,1,-1]]
        let fitted = FittedPolynomialCountSketch {
            index_hash: Array2::from_shape_vec((3, 3), vec![0, 1, 2, 2, 3, 4, 1, 0, 3]).unwrap(),
            bit_hash: Array2::from_shape_vec(
                (3, 3),
                vec![1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0],
            )
            .unwrap(),
            gamma: 0.25,
            degree: 3,
            coef0: 2.25,
            n_components: 5,
            n_features_in: 2,
        };
        let z = fitted.transform(&array![[1.0, 2.0], [-1.0, 0.5]]).unwrap();
        let expected = [
            [2.125, -2.25, 2.375, -3.625, 1.375],
            [-0.15625, -0.5625, -1.09375, -1.0625, 3.859375],
        ];
        for i in 0..2 {
            for j in 0..5 {
                assert_abs_diff_eq!(z[[i, j]], expected[i][j], epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn fit_shapes_and_reproducible_hashes() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let fitted_a = PolynomialCountSketch::<f64>::new()
            .with_gamma(0.5)
            .with_degree(3)
            .with_coef0(2.0)
            .with_n_components(7)
            .with_random_state(42)
            .fit(&x, &())
            .unwrap();
        let fitted_b = PolynomialCountSketch::<f64>::new()
            .with_gamma(0.5)
            .with_degree(3)
            .with_coef0(2.0)
            .with_n_components(7)
            .with_random_state(42)
            .fit(&x, &())
            .unwrap();

        assert_eq!(fitted_a.n_features_in(), 2);
        assert_eq!(fitted_a.degree(), 3);
        assert_eq!(fitted_a.index_hash().dim(), (3, 3));
        assert_eq!(fitted_a.bit_hash().dim(), (3, 3));
        assert_eq!(fitted_a.index_hash(), fitted_b.index_hash());
        assert_eq!(fitted_a.bit_hash(), fitted_b.bit_hash());
        for &index in fitted_a.index_hash() {
            assert!(index < 7);
        }
        for &sign in fitted_a.bit_hash() {
            assert!(sign == -1.0 || sign == 1.0);
        }

        let z = fitted_a.transform(&x).unwrap();
        assert_eq!(z.dim(), (2, 7));
        for &value in &z {
            assert!(value.is_finite());
        }
    }

    #[test]
    fn rejects_invalid_parameters_and_input() {
        let x = array![[1.0, 2.0]];
        assert!(
            PolynomialCountSketch::<f64>::new()
                .with_gamma(-1.0)
                .fit(&x, &())
                .is_err()
        );
        assert!(
            PolynomialCountSketch::<f64>::new()
                .with_degree(0)
                .fit(&x, &())
                .is_err()
        );
        assert!(
            PolynomialCountSketch::<f64>::new()
                .with_n_components(0)
                .fit(&x, &())
                .is_err()
        );
        assert!(
            PolynomialCountSketch::<f64>::new()
                .fit(&Array2::<f64>::zeros((0, 2)), &())
                .is_err()
        );
        assert!(
            PolynomialCountSketch::<f64>::new()
                .fit(&Array2::<f64>::zeros((2, 0)), &())
                .is_err()
        );

        let fitted = PolynomialCountSketch::<f64>::new()
            .with_n_components(4)
            .fit(&x, &())
            .unwrap();
        assert!(fitted.transform(&array![[1.0, 2.0, 3.0]]).is_err());
        assert!(fitted.transform(&array![[f64::NAN, 2.0]]).is_err());
        assert!(fitted.transform(&Array2::<f64>::zeros((0, 2))).is_err());
    }

    #[test]
    fn negative_coef0_matches_sklearn_nan_bias_behavior() {
        let x = array![[1.0, 2.0]];
        let fitted = PolynomialCountSketch::<f64>::new()
            .with_coef0(-1.0)
            .with_n_components(4)
            .with_random_state(1)
            .fit(&x, &())
            .unwrap();
        let z = fitted.transform(&x).unwrap();
        assert!(z.iter().any(|value| value.is_nan()));
    }
}
