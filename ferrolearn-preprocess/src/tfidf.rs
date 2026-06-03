//! TF-IDF transformer: weight a term-count matrix by inverse document frequency.
//!
//! Applies TF-IDF weighting to a term-count matrix produced by
//! [`CountVectorizer`](crate::count_vectorizer::CountVectorizer).
//!
//! # `## REQ status`
//!
//! Binary (R-DEFER-2), translating `sklearn/feature_extraction/text.py` (`class
//! TfidfTransformer` `:1483`, `TfidfVectorizer` `:1721`). Design doc:
//! `.design/preprocess/tfidf.md`. Expected values from the live sklearn 1.5.2 oracle (R-CHAR-3).
//! Consumer: crate re-export (`lib.rs:142`, grandfathered S5). HONEST (R-HONEST-3): a faithful
//! DENSE `TfidfTransformer` — the full numeric contract (IDF formula, sublinear-tf, l1/l2/None
//! norm) value-matches sklearn; sparse, `TfidfVectorizer`, param-validation, PyO3 are NOT-STARTED.
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (default smooth idf_) | SHIPPED | `fit` smooth branch `idf = ln((1+n)/(1+df))+1` = sklearn `log(n/df)+1` after `df+=1;n+=1` (`text.py:1660-1666`). Critic-verified vs live oracle (`[1.0,1.6931471805599454,...]`); `green_req1_*` in `tests/divergence_tfidf.rs`. Consumer: re-export `lib.rs:142`. |
//! | REQ-2 (default transform: idf×tf + l2) | SHIPPED | `transform` ×idf per column then L2 row-normalize, mirroring sklearn `X.data *= idf_[indices]` + `normalize` (`text.py:1705-1708`). Critic-verified FULL oracle vector `[[0.50854232,0.86103700,0],...]` (`green_req2_default_fit_transform_l2_full_vector`). |
//! | REQ-3 (smooth_idf=False) | SHIPPED | `fit` else-branch `ln(n/df)+1` (`text.py:1660-1666`); oracle `[1.0,2.09861228866811,...]`. DEVIATION (R-DEV-4): `df==0` edge → ferrolearn `1.0` vs sklearn `inf` (avoids the `0*inf=nan` footgun; a CountVectorizer never emits all-zero columns). |
//! | REQ-4 (norm l1/l2/None) | SHIPPED | `match self.norm` mirrors sklearn `normalize(X, norm)` (`text.py:1707-1708`); l1/None full vectors critic-verified vs oracle. |
//! | REQ-5 (sublinear_tf) | SHIPPED | `1 + ln(tf)` for tf>0, mirroring sklearn `log(X.data); +=1` (`text.py:1698-1700`); oracle `[[2.386294361119891,1.0]]`. |
//! | REQ-6 (use_idf=False) | SHIPPED | `idf = None` path → TF+norm only, mirroring sklearn `if self.use_idf` (`text.py:1654`,`:1702`); critic-verified value-match + `idf()==None`. |
//! | REQ-7 (idf_ attribute exposure) | SHIPPED | `idf() -> Option<&Array1<F>>` mirrors sklearn fitted attr `idf_` (`text.py:1666`). |
//! | REQ-8 (sparse CSR I/O) | NOT-STARTED | open prereq blocker #1211. Dense `Array2<F>` only; sklearn accept_sparse + sparse output (`text.py:1648`). |
//! | REQ-9 (ctor surface + _parameter_constraints) | NOT-STARTED | open prereq blocker #1212. Closed `TfidfNorm` enum; no runtime validation / InvalidParameterError (`text.py:1614-1625`, R-DEV-2). |
//! | REQ-10 (TfidfVectorizer) | NOT-STARTED | open prereq blocker #1213. No raw-text→TF-IDF chain (`text.py:1721`). |
//! | REQ-11 (PyO3 binding) | NOT-STARTED | open prereq blocker #1214. No `ferrolearn-python` registration (R-DEFER-1). |
//! | REQ-12 (ferray substrate) | NOT-STARTED | open prereq blocker #1215. `ndarray`+`num_traits`, not `ferray-core` (R-SUBSTRATE-1/2). |

use ferrolearn_core::error::FerroError;
use ndarray::{Array1, Array2};
use num_traits::Float;

// ---------------------------------------------------------------------------
// TfidfNorm
// ---------------------------------------------------------------------------

/// Row-normalization mode for the TF-IDF transformer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TfidfNorm {
    /// Normalize rows to unit L1 norm.
    L1,
    /// Normalize rows to unit L2 norm (default).
    #[default]
    L2,
    /// No row normalization.
    None,
}

// ---------------------------------------------------------------------------
// TfidfTransformer (unfitted)
// ---------------------------------------------------------------------------

/// An unfitted TF-IDF transformer.
///
/// Fits IDF weights from a term-count matrix and transforms new count
/// matrices into TF-IDF weighted matrices.
///
/// # Examples
///
/// ```
/// use ferrolearn_preprocess::tfidf::{TfidfTransformer, TfidfNorm};
/// use ndarray::array;
///
/// let counts = array![
///     [3.0_f64, 0.0, 1.0],
///     [2.0, 0.0, 0.0],
///     [3.0, 0.0, 0.0],
///     [4.0, 0.0, 0.0],
///     [3.0, 2.0, 0.0],
///     [3.0, 0.0, 2.0],
/// ];
/// let tfidf = TfidfTransformer::<f64>::new();
/// let fitted = tfidf.fit(&counts).unwrap();
/// let result = fitted.transform(&counts).unwrap();
/// assert_eq!(result.shape(), counts.shape());
/// ```
#[derive(Debug, Clone)]
pub struct TfidfTransformer<F> {
    /// Row normalization mode.
    pub norm: TfidfNorm,
    /// Whether to use IDF weighting.
    pub use_idf: bool,
    /// Whether to smooth IDF: `ln((1+n)/(1+df)) + 1`.
    pub smooth_idf: bool,
    /// Whether to apply sublinear TF scaling: `1 + ln(tf)`.
    pub sublinear_tf: bool,
    _marker: std::marker::PhantomData<F>,
}

impl<F: Float + Send + Sync + 'static> TfidfTransformer<F> {
    /// Create a new `TfidfTransformer` with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self {
            norm: TfidfNorm::L2,
            use_idf: true,
            smooth_idf: true,
            sublinear_tf: false,
            _marker: std::marker::PhantomData,
        }
    }

    /// Set the row normalization mode.
    #[must_use]
    pub fn norm(mut self, norm: TfidfNorm) -> Self {
        self.norm = norm;
        self
    }

    /// Set whether to use IDF weighting.
    #[must_use]
    pub fn use_idf(mut self, use_idf: bool) -> Self {
        self.use_idf = use_idf;
        self
    }

    /// Set whether to smooth IDF.
    #[must_use]
    pub fn smooth_idf(mut self, smooth: bool) -> Self {
        self.smooth_idf = smooth;
        self
    }

    /// Set whether to apply sublinear TF scaling.
    #[must_use]
    pub fn sublinear_tf(mut self, sublinear: bool) -> Self {
        self.sublinear_tf = sublinear;
        self
    }

    /// Fit the transformer by computing IDF from a term-count matrix.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::InsufficientSamples`] if the matrix has zero rows.
    pub fn fit(&self, counts: &Array2<F>) -> Result<FittedTfidfTransformer<F>, FerroError> {
        let n_docs = counts.nrows();
        if n_docs == 0 {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "TfidfTransformer::fit".into(),
            });
        }

        let n_features = counts.ncols();
        let n_f = F::from(n_docs).unwrap();

        let idf = if self.use_idf {
            let mut idf_vec = Array1::zeros(n_features);
            for j in 0..n_features {
                // df = number of documents where feature j is non-zero
                let df = counts.column(j).iter().filter(|&&v| v > F::zero()).count();
                let df_f = F::from(df).unwrap();

                if self.smooth_idf {
                    // idf = ln((1 + n) / (1 + df)) + 1
                    idf_vec[j] = ((F::one() + n_f) / (F::one() + df_f)).ln() + F::one();
                } else {
                    // idf = ln(n / df) + 1
                    if df > 0 {
                        idf_vec[j] = (n_f / df_f).ln() + F::one();
                    } else {
                        idf_vec[j] = F::one();
                    }
                }
            }
            Some(idf_vec)
        } else {
            None
        };

        Ok(FittedTfidfTransformer {
            idf,
            norm: self.norm,
            sublinear_tf: self.sublinear_tf,
        })
    }
}

impl<F: Float + Send + Sync + 'static> Default for TfidfTransformer<F> {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// FittedTfidfTransformer
// ---------------------------------------------------------------------------

/// A fitted TF-IDF transformer holding learned IDF weights.
///
/// Created by calling [`TfidfTransformer::fit`].
#[derive(Debug, Clone)]
pub struct FittedTfidfTransformer<F> {
    /// Per-feature IDF weights, if `use_idf` was `true`.
    idf: Option<Array1<F>>,
    /// Row normalization mode.
    norm: TfidfNorm,
    /// Whether to apply sublinear TF.
    sublinear_tf: bool,
}

impl<F: Float + Send + Sync + 'static> FittedTfidfTransformer<F> {
    /// Return the IDF weights, if computed.
    #[must_use]
    pub fn idf(&self) -> Option<&Array1<F>> {
        self.idf.as_ref()
    }

    /// Transform a term-count matrix into a TF-IDF matrix.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of columns does not
    /// match the fitted vocabulary size.
    /// Returns [`FerroError::InsufficientSamples`] if the matrix has zero rows.
    pub fn transform(&self, counts: &Array2<F>) -> Result<Array2<F>, FerroError> {
        if counts.nrows() == 0 {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "FittedTfidfTransformer::transform".into(),
            });
        }

        if let Some(ref idf) = self.idf
            && counts.ncols() != idf.len()
        {
            return Err(FerroError::ShapeMismatch {
                expected: vec![counts.nrows(), idf.len()],
                actual: vec![counts.nrows(), counts.ncols()],
                context: "FittedTfidfTransformer::transform".into(),
            });
        }

        let mut result = counts.to_owned();

        // Sublinear TF: replace tf with 1 + ln(tf) for tf > 0.
        if self.sublinear_tf {
            result.mapv_inplace(|v| if v > F::zero() { F::one() + v.ln() } else { v });
        }

        // Multiply by IDF.
        if let Some(ref idf) = self.idf {
            for mut row in result.rows_mut() {
                for (j, v) in row.iter_mut().enumerate() {
                    *v = *v * idf[j];
                }
            }
        }

        // Row normalization.
        match self.norm {
            TfidfNorm::L1 => {
                for mut row in result.rows_mut() {
                    let norm: F = row.iter().map(|v| v.abs()).fold(F::zero(), |a, b| a + b);
                    if norm > F::zero() {
                        for v in &mut row {
                            *v = *v / norm;
                        }
                    }
                }
            }
            TfidfNorm::L2 => {
                for mut row in result.rows_mut() {
                    let norm_sq: F = row.iter().map(|v| *v * *v).fold(F::zero(), |a, b| a + b);
                    let norm = norm_sq.sqrt();
                    if norm > F::zero() {
                        for v in &mut row {
                            *v = *v / norm;
                        }
                    }
                }
            }
            TfidfNorm::None => {}
        }

        Ok(result)
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
    fn test_tfidf_basic() {
        // 3 docs, 3 features
        let counts = array![[1.0_f64, 1.0, 0.0], [1.0, 0.0, 1.0], [1.0, 0.0, 0.0],];
        let transformer = TfidfTransformer::<f64>::new();
        let fitted = transformer.fit(&counts).unwrap();
        let result = fitted.transform(&counts).unwrap();
        assert_eq!(result.shape(), &[3, 3]);

        // Each row should have L2 norm ≈ 1
        for i in 0..3 {
            let row_norm: f64 = result.row(i).iter().map(|v| v * v).sum::<f64>().sqrt();
            assert_abs_diff_eq!(row_norm, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_tfidf_no_idf() {
        let counts = array![[3.0_f64, 1.0], [0.0, 2.0]];
        let transformer = TfidfTransformer::<f64>::new().use_idf(false);
        let fitted = transformer.fit(&counts).unwrap();
        let result = fitted.transform(&counts).unwrap();
        // Should just normalize rows (L2)
        for i in 0..2 {
            let row_norm: f64 = result.row(i).iter().map(|v| v * v).sum::<f64>().sqrt();
            assert_abs_diff_eq!(row_norm, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_tfidf_l1_norm() {
        let counts = array![[3.0_f64, 1.0], [0.0, 2.0]];
        let transformer = TfidfTransformer::<f64>::new()
            .use_idf(false)
            .norm(TfidfNorm::L1);
        let fitted = transformer.fit(&counts).unwrap();
        let result = fitted.transform(&counts).unwrap();
        for i in 0..2 {
            let row_l1: f64 = result.row(i).iter().map(|v| v.abs()).sum();
            assert_abs_diff_eq!(row_l1, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_tfidf_no_norm() {
        let counts = array![[1.0_f64, 0.0], [1.0, 1.0]];
        let transformer = TfidfTransformer::<f64>::new()
            .use_idf(false)
            .norm(TfidfNorm::None);
        let fitted = transformer.fit(&counts).unwrap();
        let result = fitted.transform(&counts).unwrap();
        // Should be unchanged
        for (a, b) in counts.iter().zip(result.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_tfidf_sublinear_tf() {
        let counts = array![[4.0_f64, 1.0]];
        let transformer = TfidfTransformer::<f64>::new()
            .use_idf(false)
            .sublinear_tf(true)
            .norm(TfidfNorm::None);
        let fitted = transformer.fit(&counts).unwrap();
        let result = fitted.transform(&counts).unwrap();
        // tf=4 -> 1+ln(4), tf=1 -> 1+ln(1) = 1
        assert_abs_diff_eq!(result[[0, 0]], 1.0 + 4.0_f64.ln(), epsilon = 1e-10);
        assert_abs_diff_eq!(result[[0, 1]], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_tfidf_smooth_idf() {
        // 3 docs, feature 0 in all docs, feature 1 in 1 doc
        let counts = array![[1.0_f64, 1.0], [1.0, 0.0], [1.0, 0.0]];
        let transformer = TfidfTransformer::<f64>::new().norm(TfidfNorm::None);
        let fitted = transformer.fit(&counts).unwrap();
        let idf = fitted.idf().unwrap();

        // idf[0]: ln((1+3)/(1+3)) + 1 = ln(1) + 1 = 1.0
        assert_abs_diff_eq!(idf[0], 1.0, epsilon = 1e-10);
        // idf[1]: ln((1+3)/(1+1)) + 1 = ln(2) + 1
        assert_abs_diff_eq!(idf[1], 2.0_f64.ln() + 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_tfidf_no_smooth_idf() {
        let counts = array![[1.0_f64, 1.0], [1.0, 0.0], [1.0, 0.0]];
        let transformer = TfidfTransformer::<f64>::new()
            .smooth_idf(false)
            .norm(TfidfNorm::None);
        let fitted = transformer.fit(&counts).unwrap();
        let idf = fitted.idf().unwrap();

        // idf[0]: ln(3/3) + 1 = 1.0
        assert_abs_diff_eq!(idf[0], 1.0, epsilon = 1e-10);
        // idf[1]: ln(3/1) + 1
        assert_abs_diff_eq!(idf[1], 3.0_f64.ln() + 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_tfidf_empty() {
        let counts = Array2::<f64>::zeros((0, 3));
        let transformer = TfidfTransformer::<f64>::new();
        assert!(transformer.fit(&counts).is_err());
    }

    #[test]
    fn test_tfidf_shape_mismatch() {
        let train = array![[1.0_f64, 0.0], [0.0, 1.0]];
        let fitted = TfidfTransformer::<f64>::new().fit(&train).unwrap();
        let bad = array![[1.0_f64, 0.0, 0.0]];
        assert!(fitted.transform(&bad).is_err());
    }

    #[test]
    fn test_tfidf_f32() {
        let counts = array![[1.0_f32, 0.0], [0.0, 1.0]];
        let transformer = TfidfTransformer::<f32>::new();
        let fitted = transformer.fit(&counts).unwrap();
        let result = fitted.transform(&counts).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
    }
}
