//! Spline transformer: generate B-spline basis functions for each feature.
//!
//! [`SplineTransformer`] expands each input feature into a set of B-spline
//! basis columns. This is a nonlinear feature expansion technique that
//! represents each feature as a combination of piecewise polynomial functions.
//!
//! # Knot Placement
//!
//! - [`KnotStrategy::Uniform`] — knots are evenly spaced between min and max.
//! - [`KnotStrategy::Quantile`] — knots are placed at quantiles of the data.
//!
//! ## REQ status
//!
//! Translation target: scikit-learn 1.5.2 `class SplineTransformer`
//! (`sklearn/preprocessing/_polynomial.py:580`). Tracking: #1331.
//! Each REQ is BINARY — SHIPPED (impl + non-test consumer + tests + green
//! verification) or NOT-STARTED (with a concrete open blocker).
//!
//! | REQ | Scope | Status | Evidence / Blocker |
//! |-----|-------|--------|--------------------|
//! | REQ-1 | Output dimensions (`n_knots+degree-1` cols/feature) + B-spline structural properties (partition-of-unity, non-negativity) | SHIPPED | [`FittedSplineTransformer::transform`]; sklearn `n_splines` `_polynomial.py:875`; tests `green_guard_column_count_per_feature` / `_partition_of_unity` / `_non_negativity` |
//! | REQ-2 | Uniform-knot basis VALUE parity — EXTENDED edge-spacing knots + scipy `BSpline` design matrix | SHIPPED | [`FittedSplineTransformer`] knot construction matches sklearn `_polynomial.py:908-923` + `:925-940`; verified across degree∈{1,2,3}, multi-feature, both base endpoints in `tests/divergence_spline_transformer.rs` (was DIV-1 #1332) |
//! | REQ-3 | `extrapolation` param (`constant`/`linear`/`continue`/`periodic`/`error`) for out-of-base-interval values | NOT-STARTED | no param; sklearn `_polynomial.py:619-633,721,1023-1123` — blocker #1333 |
//! | REQ-4 | `include_bias` param (drop one column when `false`) | NOT-STARTED | no param; sklearn `_polynomial.py:635,942` — blocker #1334 |
//! | REQ-5 | Quantile knots via `np.percentile`-exact (ferrolearn uses linear-interp percentile) | NOT-STARTED | `spline_transformer.rs` Quantile path; sklearn `_polynomial.py:747-753` — blocker #1335 |
//! | REQ-6 | Error/parameter contracts (`n_samples<2`, `n_knots<2`, `degree==0`, transform ncols, unfitted) | SHIPPED (scoped) | [`SplineTransformer::fit`]; NOTE divergence — sklearn allows `degree==0` & has no `n_samples>=2` requirement (`_parameter_constraints`) — see blocker #1336 |
//! | REQ-7 | `sparse_output` + `order` params | NOT-STARTED | no params; sklearn `_polynomial.py:716-730` — blocker #1337 |
//! | REQ-8 | `sample_weight` (weighted knot placement) | NOT-STARTED | sklearn `fit(X, y=None, sample_weight=None)` `_polynomial.py:811` — blocker #1338 |
//! | REQ-9 | `get_feature_names_out` (`{feat}_sp_{j}`) + `bsplines_`/`n_features_out_` fitted attrs | NOT-STARTED | sklearn `_polynomial.py:781-809,942` — blocker #1339 |
//! | REQ-10 | PyO3 binding | NOT-STARTED | no `ferrolearn-python` binding — blocker #1340 |
//! | REQ-11 | ferray substrate | NOT-STARTED | dense `Array2` only — blocker #1341 |

use ferrolearn_core::error::FerroError;
use ferrolearn_core::traits::{Fit, FitTransform, Transform};
use ndarray::Array2;
use num_traits::Float;

// ---------------------------------------------------------------------------
// KnotStrategy
// ---------------------------------------------------------------------------

/// Strategy for placing knots in the spline transformer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KnotStrategy {
    /// Knots are evenly spaced between the min and max of each feature.
    Uniform,
    /// Knots are placed at quantiles of the data.
    Quantile,
}

// ---------------------------------------------------------------------------
// SplineTransformer (unfitted)
// ---------------------------------------------------------------------------

/// An unfitted spline transformer.
///
/// Calling [`Fit::fit`] computes the knot positions and returns a
/// [`FittedSplineTransformer`] that generates B-spline basis functions.
///
/// # Parameters
///
/// - `n_knots` — number of interior knots (default 5).
/// - `degree` — degree of the B-spline (default 3, i.e. cubic).
/// - `knots` — knot placement strategy (default `Uniform`).
///
/// The number of output columns per feature is `n_knots + degree - 1`.
///
/// # Examples
///
/// ```
/// use ferrolearn_preprocess::spline_transformer::{SplineTransformer, KnotStrategy};
/// use ferrolearn_core::traits::{Fit, Transform};
/// use ndarray::array;
///
/// let st = SplineTransformer::<f64>::new(5, 3, KnotStrategy::Uniform);
/// let x = array![[0.0], [0.25], [0.5], [0.75], [1.0]];
/// let fitted = st.fit(&x, &()).unwrap();
/// let out = fitted.transform(&x).unwrap();
/// // 5 + 3 - 1 = 7 basis columns per feature
/// assert_eq!(out.ncols(), 7);
/// ```
#[must_use]
#[derive(Debug, Clone)]
pub struct SplineTransformer<F> {
    /// Number of interior knots.
    n_knots: usize,
    /// Degree of the B-spline.
    degree: usize,
    /// Knot placement strategy.
    knots: KnotStrategy,
    _marker: std::marker::PhantomData<F>,
}

impl<F: Float + Send + Sync + 'static> SplineTransformer<F> {
    /// Create a new `SplineTransformer`.
    pub fn new(n_knots: usize, degree: usize, knots: KnotStrategy) -> Self {
        Self {
            n_knots,
            degree,
            knots,
            _marker: std::marker::PhantomData,
        }
    }

    /// Return the number of interior knots.
    #[must_use]
    pub fn n_knots(&self) -> usize {
        self.n_knots
    }

    /// Return the B-spline degree.
    #[must_use]
    pub fn degree(&self) -> usize {
        self.degree
    }

    /// Return the knot placement strategy.
    #[must_use]
    pub fn knot_strategy(&self) -> KnotStrategy {
        self.knots
    }
}

impl<F: Float + Send + Sync + 'static> Default for SplineTransformer<F> {
    fn default() -> Self {
        Self::new(5, 3, KnotStrategy::Uniform)
    }
}

// ---------------------------------------------------------------------------
// FittedSplineTransformer
// ---------------------------------------------------------------------------

/// A fitted spline transformer holding per-feature knot positions.
///
/// Created by calling [`Fit::fit`] on a [`SplineTransformer`].
#[derive(Debug, Clone)]
pub struct FittedSplineTransformer<F> {
    /// Full knot vector per feature (including boundary knots with multiplicity).
    knot_vectors: Vec<Vec<F>>,
    /// Degree of the B-spline.
    degree: usize,
    /// Number of basis functions per feature.
    n_basis: usize,
}

impl<F: Float + Send + Sync + 'static> FittedSplineTransformer<F> {
    /// Return the knot vectors.
    #[must_use]
    pub fn knot_vectors(&self) -> &[Vec<F>] {
        &self.knot_vectors
    }

    /// Return the number of basis functions per feature.
    #[must_use]
    pub fn n_basis_per_feature(&self) -> usize {
        self.n_basis
    }

    /// Return the total number of output columns.
    #[must_use]
    pub fn n_output_features(&self) -> usize {
        self.knot_vectors.len() * self.n_basis
    }
}

// ---------------------------------------------------------------------------
// B-spline evaluation (Cox-de Boor recursion)
// ---------------------------------------------------------------------------

/// Evaluate all B-spline basis functions at a given value `x` using the
/// Cox-de Boor recursion.
///
/// `knots` is the full knot vector of length `n_basis + degree + 1`.
/// Returns a vector of length `n_basis` containing the basis values.
fn bspline_basis<F: Float>(x: F, knots: &[F], degree: usize, n_basis: usize) -> Vec<F> {
    // Start with degree-0 basis functions
    let n_intervals = knots.len() - 1;
    let mut basis = vec![F::zero(); n_intervals];

    // Degree 0: indicator functions using half-open intervals [t_i, t_{i+1}).
    // Special case: with sklearn's EXTENDED knot vector the base interval is
    // `[knots[degree], knots[n_basis]]` (knots[n_basis] is the right end of the
    // base support, NOT the rightmost extended knot). scipy's `design_matrix`
    // includes the right endpoint of the base interval, so a value at
    // `x == knots[n_basis]` must be evaluated as the limit from the left rather
    // than returning all-zero under a naive half-open `t_i <= x < t_{i+1}`.
    // Activate the last non-degenerate interval that LIES AT OR BEFORE the base
    // right endpoint so the Cox-de Boor recursion propagates a non-zero value.
    let base_right = knots[n_basis];
    if x >= base_right {
        // Find the last interval ending at the base right endpoint with
        // non-zero width and activate it (the closed-right base span).
        let mut found = false;
        for i in (0..n_intervals).rev() {
            if knots[i + 1] <= base_right && knots[i] < knots[i + 1] {
                basis[i] = F::one();
                found = true;
                break;
            }
        }
        // Fallback: if all such intervals are degenerate, activate the last one
        if !found {
            basis[n_intervals - 1] = F::one();
        }
    } else {
        for i in 0..n_intervals {
            // Half-open: [t_i, t_{i+1})
            basis[i] = if x >= knots[i] && x < knots[i + 1] {
                F::one()
            } else {
                F::zero()
            };
        }
    }

    // Build up to the desired degree
    for d in 1..=degree {
        let n_current = n_intervals - d;
        let mut new_basis = vec![F::zero(); n_current];
        for i in 0..n_current {
            let denom1 = knots[i + d] - knots[i];
            let denom2 = knots[i + d + 1] - knots[i + 1];

            let left = if denom1 > F::zero() {
                (x - knots[i]) / denom1 * basis[i]
            } else {
                F::zero()
            };

            let right = if denom2 > F::zero() {
                (knots[i + d + 1] - x) / denom2 * basis[i + 1]
            } else {
                F::zero()
            };

            new_basis[i] = left + right;
        }
        basis = new_basis;
    }

    // Truncate or pad to n_basis
    basis.truncate(n_basis);
    while basis.len() < n_basis {
        basis.push(F::zero());
    }

    basis
}

// ---------------------------------------------------------------------------
// Trait implementations
// ---------------------------------------------------------------------------

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, ()> for SplineTransformer<F> {
    type Fitted = FittedSplineTransformer<F>;
    type Error = FerroError;

    /// Fit by computing knot positions for each feature.
    ///
    /// # Errors
    ///
    /// - [`FerroError::InsufficientSamples`] if the input has fewer than 2 rows.
    /// - [`FerroError::InvalidParameter`] if `n_knots` < 2 or `degree` is 0.
    fn fit(&self, x: &Array2<F>, _y: &()) -> Result<FittedSplineTransformer<F>, FerroError> {
        let n_samples = x.nrows();
        if n_samples < 2 {
            return Err(FerroError::InsufficientSamples {
                required: 2,
                actual: n_samples,
                context: "SplineTransformer::fit".into(),
            });
        }
        if self.n_knots < 2 {
            return Err(FerroError::InvalidParameter {
                name: "n_knots".into(),
                reason: "n_knots must be at least 2".into(),
            });
        }
        if self.degree == 0 {
            return Err(FerroError::InvalidParameter {
                name: "degree".into(),
                reason: "degree must be at least 1".into(),
            });
        }

        let n_features = x.ncols();
        let n_basis = self.n_knots + self.degree - 1;
        let mut knot_vectors = Vec::with_capacity(n_features);

        for j in 0..n_features {
            let mut col_vals: Vec<F> = x.column(j).iter().copied().collect();
            col_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let min_val = col_vals[0];
            let max_val = col_vals[col_vals.len() - 1];

            // Compute interior knots
            let interior_knots: Vec<F> = match self.knots {
                KnotStrategy::Uniform => (0..self.n_knots)
                    .map(|i| {
                        min_val
                            + (max_val - min_val) * F::from(i).unwrap()
                                / F::from(self.n_knots - 1).unwrap()
                    })
                    .collect(),
                KnotStrategy::Quantile => {
                    let n = col_vals.len();
                    (0..self.n_knots)
                        .map(|i| {
                            let frac = F::from(i).unwrap()
                                / F::from(self.n_knots - 1).unwrap_or_else(F::one);
                            let pos = frac * F::from(n.saturating_sub(1)).unwrap();
                            let lo = pos.floor().to_usize().unwrap_or(0).min(n - 1);
                            let hi = pos.ceil().to_usize().unwrap_or(0).min(n - 1);
                            let f = pos - F::from(lo).unwrap();
                            col_vals[lo] * (F::one() - f) + col_vals[hi] * f
                        })
                        .collect()
                }
            };

            // Build full knot vector using sklearn's EXTENDED edge-spacing
            // construction (`_polynomial.py:908-923`). sklearn explicitly
            // REJECTS the clamped/`np.tile` repeated-boundary construction
            // (`:898-906`, Eilers & Marx) in favour of reusing the spacing of
            // the two first/last base knots:
            //   dist_min = base[1] - base[0]; dist_max = base[-1] - base[-2]
            //   left  = linspace(base[0] - degree*dist_min, base[0] - dist_min, degree)
            //   right = linspace(base[-1] + dist_max, base[-1] + degree*dist_max, degree)
            //   knots = [left, base, right]
            // numpy `linspace(a, b, num)` is inclusive of both endpoints.
            let base = &interior_knots;
            let nb = base.len();
            let dist_min = base[1] - base[0];
            let dist_max = base[nb - 1] - base[nb - 2];
            let degree = self.degree;
            let deg_f = F::from(degree).unwrap_or_else(F::one);

            // numpy linspace with `num` inclusive endpoints. For num == 1 numpy
            // returns just [a]; for num >= 2 it includes both a and b.
            let linspace = |a: F, b: F, num: usize| -> Vec<F> {
                if num == 1 {
                    return vec![a];
                }
                let denom = F::from(num - 1).unwrap_or_else(F::one);
                (0..num)
                    .map(|i| {
                        let t = F::from(i).unwrap_or_else(F::zero) / denom;
                        a + (b - a) * t
                    })
                    .collect()
            };

            let left = linspace(base[0] - deg_f * dist_min, base[0] - dist_min, degree);
            let right = linspace(
                base[nb - 1] + dist_max,
                base[nb - 1] + deg_f * dist_max,
                degree,
            );

            let mut full_knots = Vec::with_capacity(left.len() + nb + right.len());
            full_knots.extend_from_slice(&left);
            full_knots.extend_from_slice(base);
            full_knots.extend_from_slice(&right);

            knot_vectors.push(full_knots);
        }

        Ok(FittedSplineTransformer {
            knot_vectors,
            degree: self.degree,
            n_basis,
        })
    }
}

impl<F: Float + Send + Sync + 'static> Transform<Array2<F>> for FittedSplineTransformer<F> {
    type Output = Array2<F>;
    type Error = FerroError;

    /// Generate B-spline basis functions for each feature.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of columns differs
    /// from the number of features seen during fitting.
    fn transform(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let n_features = self.knot_vectors.len();
        if x.ncols() != n_features {
            return Err(FerroError::ShapeMismatch {
                expected: vec![x.nrows(), n_features],
                actual: vec![x.nrows(), x.ncols()],
                context: "FittedSplineTransformer::transform".into(),
            });
        }

        let n_samples = x.nrows();
        let n_out = n_features * self.n_basis;
        let mut out = Array2::zeros((n_samples, n_out));

        for j in 0..n_features {
            let knots = &self.knot_vectors[j];
            let col_offset = j * self.n_basis;

            for i in 0..n_samples {
                let val = x[[i, j]];
                let basis_vals = bspline_basis(val, knots, self.degree, self.n_basis);
                for (k, &bv) in basis_vals.iter().enumerate() {
                    out[[i, col_offset + k]] = bv;
                }
            }
        }

        Ok(out)
    }
}

/// Implement `Transform` on the unfitted transformer.
impl<F: Float + Send + Sync + 'static> Transform<Array2<F>> for SplineTransformer<F> {
    type Output = Array2<F>;
    type Error = FerroError;

    /// Always returns an error — the transformer must be fitted first.
    fn transform(&self, _x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        Err(FerroError::InvalidParameter {
            name: "SplineTransformer".into(),
            reason: "transformer must be fitted before calling transform; use fit() first".into(),
        })
    }
}

impl<F: Float + Send + Sync + 'static> FitTransform<Array2<F>> for SplineTransformer<F> {
    type FitError = FerroError;

    /// Fit and transform in one step.
    ///
    /// # Errors
    ///
    /// Returns an error if fitting fails.
    fn fit_transform(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
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
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_spline_output_dimensions() {
        let st = SplineTransformer::<f64>::new(5, 3, KnotStrategy::Uniform);
        let x = array![[0.0], [0.25], [0.5], [0.75], [1.0]];
        let fitted = st.fit(&x, &()).unwrap();
        let out = fitted.transform(&x).unwrap();
        // n_basis = n_knots + degree - 1 = 5 + 3 - 1 = 7
        assert_eq!(out.ncols(), 7);
        assert_eq!(out.nrows(), 5);
    }

    #[test]
    fn test_spline_partition_of_unity() {
        // B-spline basis functions should sum to 1 at any interior point
        let st = SplineTransformer::<f64>::new(5, 3, KnotStrategy::Uniform);
        let x = array![[0.0], [0.25], [0.5], [0.75], [1.0]];
        let fitted = st.fit(&x, &()).unwrap();
        let out = fitted.transform(&x).unwrap();
        for i in 0..out.nrows() {
            let row_sum: f64 = out.row(i).iter().sum();
            assert_abs_diff_eq!(row_sum, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_spline_non_negative() {
        let st = SplineTransformer::<f64>::new(5, 3, KnotStrategy::Uniform);
        let x = array![[0.0], [0.1], [0.5], [0.9], [1.0]];
        let fitted = st.fit(&x, &()).unwrap();
        let out = fitted.transform(&x).unwrap();
        for v in &out {
            assert!(*v >= -1e-10, "Basis value should be non-negative, got {v}");
        }
    }

    #[test]
    fn test_spline_quantile_knots() {
        let st = SplineTransformer::<f64>::new(5, 3, KnotStrategy::Quantile);
        let x = array![[0.0], [0.1], [0.2], [0.5], [1.0]];
        let fitted = st.fit(&x, &()).unwrap();
        let out = fitted.transform(&x).unwrap();
        assert_eq!(out.ncols(), 7);
        // Partition of unity should still hold
        for i in 0..out.nrows() {
            let row_sum: f64 = out.row(i).iter().sum();
            assert_abs_diff_eq!(row_sum, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_spline_multi_feature() {
        let st = SplineTransformer::<f64>::new(3, 2, KnotStrategy::Uniform);
        let x = array![[0.0, 10.0], [0.5, 15.0], [1.0, 20.0]];
        let fitted = st.fit(&x, &()).unwrap();
        let out = fitted.transform(&x).unwrap();
        // n_basis per feature = 3 + 2 - 1 = 4, total = 2 * 4 = 8
        assert_eq!(out.ncols(), 8);
    }

    #[test]
    fn test_spline_fit_transform() {
        let st = SplineTransformer::<f64>::new(5, 3, KnotStrategy::Uniform);
        let x = array![[0.0], [0.5], [1.0]];
        let out = st.fit_transform(&x).unwrap();
        assert_eq!(out.ncols(), 7);
    }

    #[test]
    fn test_spline_insufficient_samples_error() {
        let st = SplineTransformer::<f64>::new(5, 3, KnotStrategy::Uniform);
        let x = array![[1.0]];
        assert!(st.fit(&x, &()).is_err());
    }

    #[test]
    fn test_spline_too_few_knots_error() {
        let st = SplineTransformer::<f64>::new(1, 3, KnotStrategy::Uniform);
        let x = array![[0.0], [1.0]];
        assert!(st.fit(&x, &()).is_err());
    }

    #[test]
    fn test_spline_zero_degree_error() {
        let st = SplineTransformer::<f64>::new(5, 0, KnotStrategy::Uniform);
        let x = array![[0.0], [1.0]];
        assert!(st.fit(&x, &()).is_err());
    }

    #[test]
    fn test_spline_shape_mismatch_error() {
        let st = SplineTransformer::<f64>::new(5, 3, KnotStrategy::Uniform);
        let x_train = array![[0.0, 1.0], [0.5, 1.5]];
        let fitted = st.fit(&x_train, &()).unwrap();
        let x_bad = array![[0.0]];
        assert!(fitted.transform(&x_bad).is_err());
    }

    #[test]
    fn test_spline_unfitted_error() {
        let st = SplineTransformer::<f64>::new(5, 3, KnotStrategy::Uniform);
        let x = array![[0.0]];
        assert!(st.transform(&x).is_err());
    }

    #[test]
    fn test_spline_default() {
        let st = SplineTransformer::<f64>::default();
        assert_eq!(st.n_knots(), 5);
        assert_eq!(st.degree(), 3);
        assert_eq!(st.knot_strategy(), KnotStrategy::Uniform);
    }

    #[test]
    fn test_spline_degree1() {
        // Linear splines: should produce piecewise linear basis
        let st = SplineTransformer::<f64>::new(3, 1, KnotStrategy::Uniform);
        let x = array![[0.0], [0.5], [1.0]];
        let fitted = st.fit(&x, &()).unwrap();
        let out = fitted.transform(&x).unwrap();
        // n_basis = 3 + 1 - 1 = 3
        assert_eq!(out.ncols(), 3);
        // Partition of unity
        for i in 0..out.nrows() {
            let row_sum: f64 = out.row(i).iter().sum();
            assert_abs_diff_eq!(row_sum, 1.0, epsilon = 1e-10);
        }
    }
}
