//! Skewed chi-squared random Fourier feature map.
//!
//! [`SkewedChi2Sampler`] mirrors sklearn's
//! `sklearn.kernel_approximation.SkewedChi2Sampler`: it samples random weights
//! from the hyperbolic secant distribution via the inverse CDF and maps
//! `log(X + skewedness)` through random cosine features.
//!
//! Exact random draws are an RNG-substrate gap (Xoshiro vs numpy RandomState),
//! but the transform formula, validation boundaries, shapes, and sampling
//! support are pinned against sklearn behavior.

use ferrolearn_core::error::FerroError;
use ferrolearn_core::traits::{Fit, Transform};
use ndarray::{Array1, Array2};
use num_traits::Float;
use rand::SeedableRng;
use rand_distr::{Distribution, Uniform};

/// Approximate feature map for the skewed chi-squared kernel.
#[derive(Debug, Clone)]
pub struct SkewedChi2Sampler<F> {
    /// Skew added to every input before taking logs. Default: 1.0.
    skewedness: F,
    /// Number of Monte Carlo features. Default: 100.
    n_components: usize,
    /// Optional random seed for reproducibility.
    random_state: Option<u64>,
}

impl<F: Float + Send + Sync + 'static> SkewedChi2Sampler<F> {
    /// Create a new sampler with sklearn defaults.
    #[must_use]
    pub fn new() -> Self {
        Self {
            skewedness: F::one(),
            n_components: 100,
            random_state: None,
        }
    }

    /// Set the skewedness parameter.
    #[must_use]
    pub fn with_skewedness(mut self, skewedness: F) -> Self {
        self.skewedness = skewedness;
        self
    }

    /// Set the number of output components.
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

    /// Return the configured skewedness.
    #[must_use]
    pub fn skewedness(&self) -> F {
        self.skewedness
    }

    /// Return the configured component count.
    #[must_use]
    pub fn n_components(&self) -> usize {
        self.n_components
    }
}

impl<F: Float + Send + Sync + 'static> Default for SkewedChi2Sampler<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted skewed chi-squared random Fourier map.
#[derive(Debug, Clone)]
pub struct FittedSkewedChi2Sampler<F> {
    random_weights: Array2<F>,
    random_offset: Array1<F>,
    skewedness: F,
    scale: F,
}

impl<F: Float + Send + Sync + 'static> FittedSkewedChi2Sampler<F> {
    /// Random weight matrix of shape `(n_features, n_components)`.
    #[must_use]
    pub fn random_weights(&self) -> &Array2<F> {
        &self.random_weights
    }

    /// Random offset vector of shape `(n_components,)`.
    #[must_use]
    pub fn random_offset(&self) -> &Array1<F> {
        &self.random_offset
    }

    /// Number of features observed during fitting.
    #[must_use]
    pub fn n_features_in(&self) -> usize {
        self.random_weights.nrows()
    }

    /// Number of output components.
    #[must_use]
    pub fn n_components(&self) -> usize {
        self.random_weights.ncols()
    }

    /// Skewedness used during transformation.
    #[must_use]
    pub fn skewedness(&self) -> F {
        self.skewedness
    }
}

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, ()> for SkewedChi2Sampler<F> {
    type Fitted = FittedSkewedChi2Sampler<F>;
    type Error = FerroError;

    /// Fit by sampling random weights and offsets.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::InvalidParameter`] when `n_components == 0`.
    /// Returns [`FerroError::InsufficientSamples`] when no rows are provided.
    fn fit(&self, x: &Array2<F>, _y: &()) -> Result<FittedSkewedChi2Sampler<F>, FerroError> {
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
                context: "SkewedChi2Sampler::fit".into(),
            });
        }

        let n_features = x.ncols();
        let n_components = self.n_components;
        let mut rng = match self.random_state {
            Some(seed) => rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(seed),
            None => rand_xoshiro::Xoshiro256PlusPlus::from_os_rng(),
        };
        let unit = Uniform::new(0.0, 1.0).map_err(|e| FerroError::InvalidParameter {
            name: "weight_distribution".into(),
            reason: format!("failed to create uniform distribution: {e}"),
        })?;
        let offset_dist = Uniform::new(0.0, 2.0 * std::f64::consts::PI).map_err(|e| {
            FerroError::InvalidParameter {
                name: "offset_distribution".into(),
                reason: format!("failed to create uniform distribution: {e}"),
            }
        })?;

        let pi = std::f64::consts::PI;
        let mut weights = Vec::with_capacity(n_features * n_components);
        for _ in 0..(n_features * n_components) {
            let u: f64 = unit.sample(&mut rng);
            let weight = (0.5 * pi * u).tan().ln() / pi;
            weights.push(
                F::from(weight).ok_or_else(|| FerroError::NumericalInstability {
                    message: "SkewedChi2Sampler: f64->F conversion of sampled weight failed".into(),
                })?,
            );
        }
        let random_weights =
            Array2::from_shape_vec((n_features, n_components), weights).map_err(|e| {
                FerroError::NumericalInstability {
                    message: format!("failed to create weight matrix: {e}"),
                }
            })?;

        let mut offsets = Vec::with_capacity(n_components);
        for _ in 0..n_components {
            offsets.push(F::from(offset_dist.sample(&mut rng)).ok_or_else(|| {
                FerroError::NumericalInstability {
                    message: "SkewedChi2Sampler: f64->F conversion of sampled offset failed".into(),
                }
            })?);
        }

        let n_components_f =
            F::from(n_components).ok_or_else(|| FerroError::NumericalInstability {
                message: "SkewedChi2Sampler: usize->F conversion of n_components failed".into(),
            })?;
        let scale = F::from(2.0).unwrap().sqrt() / n_components_f.sqrt();

        Ok(FittedSkewedChi2Sampler {
            random_weights,
            random_offset: Array1::from_vec(offsets),
            skewedness: self.skewedness,
            scale,
        })
    }
}

impl<F: Float + Send + Sync + 'static> Transform<Array2<F>> for FittedSkewedChi2Sampler<F> {
    type Output = Array2<F>;
    type Error = FerroError;

    /// Apply the random Fourier map to `log(X + skewedness)`.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] when feature counts differ from
    /// fit time, and [`FerroError::InvalidParameter`] when any entry is not
    /// strictly greater than `-skewedness`.
    fn transform(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        if x.ncols() != self.random_weights.nrows() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![x.nrows(), self.random_weights.nrows()],
                actual: vec![x.nrows(), x.ncols()],
                context: "FittedSkewedChi2Sampler::transform feature count must match fit data"
                    .into(),
            });
        }

        let lower_bound = -self.skewedness;
        for &value in x {
            if value <= lower_bound {
                return Err(FerroError::InvalidParameter {
                    name: "X".into(),
                    reason: "may not contain entries smaller than or equal to -skewedness".into(),
                });
            }
        }

        let shifted_log = x.mapv(|value| (value + self.skewedness).ln());
        let projection = shifted_log.dot(&self.random_weights) + &self.random_offset;
        Ok(projection.mapv(|value| self.scale * value.cos()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::{Array1, Array2, array};

    #[test]
    fn defaults_match_sklearn() {
        let sampler = SkewedChi2Sampler::<f64>::new();
        assert_abs_diff_eq!(sampler.skewedness(), 1.0, epsilon = 1e-15);
        assert_eq!(sampler.n_components(), 100);
    }

    #[test]
    fn transform_formula_matches_sklearn_with_fixed_components() {
        // sklearn 1.5.2, /tmp:
        // SkewedChi2Sampler(skewedness=0.5, n_components=3) with
        // random_weights_ = [[0.25,-0.5,1.0],[1.25,0.0,-0.75]]
        // random_offset_ = [0.1,1.5,2.5]
        // transform([[0,1],[2,3]])
        let fitted = FittedSkewedChi2Sampler {
            random_weights: Array2::from_shape_vec((2, 3), vec![0.25, -0.5, 1.0, 1.25, 0.0, -0.75])
                .unwrap(),
            random_offset: Array1::from_vec(vec![0.1, 1.5, 2.5]),
            skewedness: 0.5,
            scale: (2.0_f64 / 3.0).sqrt(),
        };
        let z = fitted.transform(&array![[0.0, 1.0], [2.0, 3.0]]).unwrap();
        let expected = [
            [
                0.740_956_282_178_009_7,
                -0.222_327_865_319_947_6,
                0.055_513_477_864_574_54,
            ],
            [
                -0.260_118_716_121_996_7,
                0.412_020_458_355_585_3,
                -0.642_578_266_993_953_3,
            ],
        ];
        for i in 0..2 {
            for j in 0..3 {
                assert_abs_diff_eq!(z[[i, j]], expected[i][j], epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn fit_shapes_and_sampling_ranges() {
        let x = array![[0.0, 1.0], [2.0, 3.0]];
        let fitted = SkewedChi2Sampler::<f64>::new()
            .with_skewedness(0.5)
            .with_n_components(4)
            .with_random_state(42)
            .fit(&x, &())
            .unwrap();
        assert_eq!(fitted.random_weights().dim(), (2, 4));
        assert_eq!(fitted.random_offset().len(), 4);
        assert_eq!(fitted.n_features_in(), 2);
        assert_eq!(fitted.n_components(), 4);
        assert_abs_diff_eq!(fitted.skewedness(), 0.5, epsilon = 1e-15);
        for &offset in fitted.random_offset() {
            assert!(offset >= 0.0);
            assert!(offset < 2.0 * std::f64::consts::PI);
        }
        let z = fitted.transform(&x).unwrap();
        assert_eq!(z.dim(), (2, 4));
        for &value in &z {
            assert!(value.is_finite());
        }
    }

    #[test]
    fn rejects_zero_components_and_bad_transform_input() {
        let x = array![[0.0, 1.0]];
        assert!(
            SkewedChi2Sampler::<f64>::new()
                .with_n_components(0)
                .fit(&x, &())
                .is_err()
        );
        let fitted = SkewedChi2Sampler::<f64>::new()
            .with_skewedness(0.5)
            .fit(&x, &())
            .unwrap();
        assert!(fitted.transform(&array![[-0.5, 0.0]]).is_err());
        assert!(fitted.transform(&array![[0.0, 1.0, 2.0]]).is_err());
    }
}
