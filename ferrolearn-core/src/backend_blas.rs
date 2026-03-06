//! BLAS/LAPACK backend implementation using `ndarray-linalg`.
//!
//! [`BLASBackend`] implements the [`Backend`](crate::backend::Backend) trait by
//! delegating to system BLAS (for GEMM) and LAPACK (for SVD, QR, Cholesky,
//! eigendecomposition, solve, determinant, and inverse) via the `ndarray-linalg`
//! crate.
//!
//! This backend is enabled by the `blas` feature flag. It links against
//! OpenBLAS (static) by default; other BLAS implementations (Intel MKL,
//! Netlib, system OpenBLAS) can be selected via the `ndarray-linalg` feature
//! flags in `Cargo.toml`.
//!
//! # When to use
//!
//! The BLAS backend can be faster than the default `faer` backend for large
//! matrices on hardware with optimized BLAS libraries (e.g., Intel MKL on
//! x86). For small-to-medium matrices, the default `faer` backend is often
//! competitive or faster due to lower overhead.
//!
//! # Example
//!
//! ```ignore
//! use ferrolearn_core::backend::Backend;
//! use ferrolearn_core::backend_blas::BLASBackend;
//! use ndarray::array;
//!
//! let a = array![[1.0, 2.0], [3.0, 4.0]];
//! let b = array![[5.0, 6.0], [7.0, 8.0]];
//! let c = BLASBackend::gemm(&a, &b).unwrap();
//! ```

use crate::backend::Backend;
use crate::error::{FerroError, FerroResult};
use ndarray::{Array1, Array2};

/// A BLAS/LAPACK backend for linear algebra operations.
///
/// Delegates to system BLAS (via `ndarray-linalg`) for high-performance
/// matrix operations. This is a zero-sized type used solely as a type
/// parameter on algorithms generic over [`Backend`].
///
/// Requires the `blas` feature flag to be enabled.
pub struct BLASBackend;

impl Backend for BLASBackend {
    fn gemm(a: &Array2<f64>, b: &Array2<f64>) -> FerroResult<Array2<f64>> {
        if a.ncols() != b.nrows() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![a.nrows(), a.ncols()],
                actual: vec![b.nrows(), b.ncols()],
                context: format!(
                    "gemm: A is {}x{} but B is {}x{} (inner dimensions {} != {})",
                    a.nrows(),
                    a.ncols(),
                    b.nrows(),
                    b.ncols(),
                    a.ncols(),
                    b.nrows()
                ),
            });
        }
        // ndarray's dot delegates to BLAS when the blas feature is enabled
        // on ndarray (which ndarray-linalg enables).
        Ok(a.dot(b))
    }

    fn svd(a: &Array2<f64>) -> FerroResult<(Array2<f64>, Array1<f64>, Array2<f64>)> {
        use ndarray_linalg::SVD;

        let (u_opt, s, vt_opt) =
            a.svd(true, true)
                .map_err(|e| FerroError::NumericalInstability {
                    message: format!("SVD failed: {e}"),
                })?;

        let u = u_opt.ok_or_else(|| FerroError::NumericalInstability {
            message: "SVD did not produce U matrix".into(),
        })?;
        let vt = vt_opt.ok_or_else(|| FerroError::NumericalInstability {
            message: "SVD did not produce Vt matrix".into(),
        })?;

        Ok((u, s, vt))
    }

    fn qr(a: &Array2<f64>) -> FerroResult<(Array2<f64>, Array2<f64>)> {
        use ndarray_linalg::QR;

        let (q, r) = a.qr().map_err(|e| FerroError::NumericalInstability {
            message: format!("QR decomposition failed: {e}"),
        })?;

        Ok((q, r))
    }

    fn cholesky(a: &Array2<f64>) -> FerroResult<Array2<f64>> {
        use ndarray_linalg::cholesky::*;

        let (nrows, ncols) = a.dim();
        if nrows != ncols {
            return Err(FerroError::ShapeMismatch {
                expected: vec![nrows, nrows],
                actual: vec![nrows, ncols],
                context: "cholesky: matrix must be square".into(),
            });
        }

        let l = a
            .cholesky(UPLO::Lower)
            .map_err(|e| FerroError::NumericalInstability {
                message: format!(
                    "Cholesky decomposition failed (matrix not positive definite): {e}"
                ),
            })?;

        Ok(l)
    }

    fn solve(a: &Array2<f64>, b: &Array1<f64>) -> FerroResult<Array1<f64>> {
        use ndarray_linalg::Solve;

        let (nrows, ncols) = a.dim();
        if nrows != ncols {
            return Err(FerroError::ShapeMismatch {
                expected: vec![nrows, nrows],
                actual: vec![nrows, ncols],
                context: "solve: coefficient matrix must be square".into(),
            });
        }
        if b.len() != nrows {
            return Err(FerroError::ShapeMismatch {
                expected: vec![nrows],
                actual: vec![b.len()],
                context: format!("solve: b has length {} but A has {} rows", b.len(), nrows),
            });
        }

        let x = a.solve(b).map_err(|e| FerroError::NumericalInstability {
            message: format!("Linear solve failed (matrix may be singular): {e}"),
        })?;

        Ok(x)
    }

    fn eigh(a: &Array2<f64>) -> FerroResult<(Array1<f64>, Array2<f64>)> {
        use ndarray_linalg::Eigh;

        let (nrows, ncols) = a.dim();
        if nrows != ncols {
            return Err(FerroError::ShapeMismatch {
                expected: vec![nrows, nrows],
                actual: vec![nrows, ncols],
                context: "eigh: matrix must be square".into(),
            });
        }

        let (eigenvalues, eigenvectors) =
            a.eigh(ndarray_linalg::UPLO::Lower)
                .map_err(|e| FerroError::NumericalInstability {
                    message: format!("Symmetric eigendecomposition failed: {e}"),
                })?;

        Ok((eigenvalues, eigenvectors))
    }

    fn det(a: &Array2<f64>) -> FerroResult<f64> {
        use ndarray_linalg::Determinant;

        let (nrows, ncols) = a.dim();
        if nrows != ncols {
            return Err(FerroError::ShapeMismatch {
                expected: vec![nrows, nrows],
                actual: vec![nrows, ncols],
                context: "det: matrix must be square".into(),
            });
        }

        let d = a.det().map_err(|e| FerroError::NumericalInstability {
            message: format!("Determinant computation failed: {e}"),
        })?;

        Ok(d)
    }

    fn inv(a: &Array2<f64>) -> FerroResult<Array2<f64>> {
        use ndarray_linalg::Inverse;

        let (nrows, ncols) = a.dim();
        if nrows != ncols {
            return Err(FerroError::ShapeMismatch {
                expected: vec![nrows, nrows],
                actual: vec![nrows, ncols],
                context: "inv: matrix must be square".into(),
            });
        }

        let inv_mat = a.inv().map_err(|e| FerroError::NumericalInstability {
            message: format!("Matrix inversion failed (matrix may be singular): {e}"),
        })?;

        Ok(inv_mat)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    // -----------------------------------------------------------------------
    // Helper: shorthand alias for the backend
    // -----------------------------------------------------------------------
    type B = BLASBackend;

    /// Assert that two 2D arrays are element-wise approximately equal.
    fn assert_mat_eq(actual: &Array2<f64>, expected: &Array2<f64>, eps: f64) {
        assert_eq!(actual.dim(), expected.dim(), "shape mismatch");
        for ((i, j), &val) in actual.indexed_iter() {
            assert_relative_eq!(val, expected[[i, j]], epsilon = eps);
        }
    }

    /// Assert that two 1D arrays are element-wise approximately equal.
    fn assert_vec_eq(actual: &Array1<f64>, expected: &Array1<f64>, eps: f64) {
        assert_eq!(actual.len(), expected.len(), "length mismatch");
        for (i, &val) in actual.iter().enumerate() {
            assert_relative_eq!(val, expected[i], epsilon = eps);
        }
    }

    // -----------------------------------------------------------------------
    // gemm tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_gemm_identity() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let eye = array![[1.0, 0.0], [0.0, 1.0]];
        let c = B::gemm(&a, &eye).unwrap();
        assert_mat_eq(&c, &a, 1e-12);
    }

    #[test]
    fn test_gemm_known_result() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let b = array![[5.0, 6.0], [7.0, 8.0]];
        let c = B::gemm(&a, &b).unwrap();
        let expected = array![[19.0, 22.0], [43.0, 50.0]];
        assert_mat_eq(&c, &expected, 1e-12);
    }

    #[test]
    fn test_gemm_rectangular() {
        let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let b = array![[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]];
        let c = B::gemm(&a, &b).unwrap();
        let expected = array![[58.0, 64.0], [139.0, 154.0]];
        assert_mat_eq(&c, &expected, 1e-12);
    }

    #[test]
    fn test_gemm_shape_mismatch() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let b = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let result = B::gemm(&a, &b);
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // svd tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_svd_identity() {
        let eye = array![[1.0, 0.0], [0.0, 1.0]];
        let (u, s, vt) = B::svd(&eye).unwrap();
        for &val in s.iter() {
            assert_relative_eq!(val, 1.0, epsilon = 1e-12);
        }
        let reconstructed = reconstruct_svd(&u, &s, &vt);
        assert_mat_eq(&reconstructed, &eye, 1e-12);
    }

    #[test]
    fn test_svd_reconstruction() {
        let a = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let (u, s, vt) = B::svd(&a).unwrap();
        let reconstructed = reconstruct_svd(&u, &s, &vt);
        assert_mat_eq(&reconstructed, &a, 1e-10);
    }

    #[test]
    fn test_svd_singular_values_descending() {
        let a = array![[3.0, 1.0], [1.0, 3.0]];
        let (_, s, _) = B::svd(&a).unwrap();
        assert!(s[0] >= s[1], "singular values should be non-increasing");
    }

    // -----------------------------------------------------------------------
    // qr tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_qr_reconstruction() {
        let a = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let (q, r) = B::qr(&a).unwrap();
        let reconstructed = q.dot(&r);
        assert_mat_eq(&reconstructed, &a, 1e-10);
    }

    #[test]
    fn test_qr_identity() {
        let eye = array![[1.0, 0.0], [0.0, 1.0]];
        let (q, r) = B::qr(&eye).unwrap();
        let reconstructed = q.dot(&r);
        assert_mat_eq(&reconstructed, &eye, 1e-12);
    }

    // -----------------------------------------------------------------------
    // cholesky tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_cholesky_known() {
        let a = array![[4.0, 2.0], [2.0, 3.0]];
        let l = B::cholesky(&a).unwrap();
        let reconstructed = l.dot(&l.t());
        assert_mat_eq(&reconstructed, &a, 1e-10);
    }

    #[test]
    fn test_cholesky_identity() {
        let eye = array![[1.0, 0.0], [0.0, 1.0]];
        let l = B::cholesky(&eye).unwrap();
        assert_mat_eq(&l, &eye, 1e-12);
    }

    #[test]
    fn test_cholesky_not_positive_definite() {
        let a = array![[-1.0, 0.0], [0.0, -1.0]];
        let result = B::cholesky(&a);
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // solve tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_solve_simple() {
        let a = array![[2.0, 1.0], [1.0, 3.0]];
        let b = array![5.0, 7.0];
        let x = B::solve(&a, &b).unwrap();
        assert_relative_eq!(x[0], 1.6, epsilon = 1e-10);
        assert_relative_eq!(x[1], 1.8, epsilon = 1e-10);
    }

    #[test]
    fn test_solve_identity() {
        let eye = array![[1.0, 0.0], [0.0, 1.0]];
        let b = array![3.0, 7.0];
        let x = B::solve(&eye, &b).unwrap();
        assert_vec_eq(&x, &b, 1e-12);
    }

    #[test]
    fn test_solve_3x3() {
        let a = array![[1.0, 2.0, 3.0], [0.0, 1.0, 4.0], [5.0, 6.0, 0.0]];
        let b = array![1.0, 2.0, 3.0];
        let x = B::solve(&a, &b).unwrap();
        let ax = a.dot(&x);
        assert_vec_eq(&ax, &b, 1e-10);
    }

    #[test]
    fn test_solve_shape_mismatch() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let b = array![1.0, 2.0, 3.0];
        let result = B::solve(&a, &b);
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // eigh tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_eigh_identity() {
        let eye = array![[1.0, 0.0], [0.0, 1.0]];
        let (eigenvalues, eigenvectors) = B::eigh(&eye).unwrap();
        for &val in eigenvalues.iter() {
            assert_relative_eq!(val, 1.0, epsilon = 1e-12);
        }
        let vvt = eigenvectors.dot(&eigenvectors.t());
        assert_mat_eq(&vvt, &eye, 1e-12);
    }

    #[test]
    fn test_eigh_symmetric() {
        let a = array![[2.0, 1.0], [1.0, 2.0]];
        let (eigenvalues, eigenvectors) = B::eigh(&a).unwrap();
        // LAPACK returns eigenvalues in non-decreasing order
        assert_relative_eq!(eigenvalues[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(eigenvalues[1], 3.0, epsilon = 1e-10);

        let reconstructed = reconstruct_eigh(&eigenvalues, &eigenvectors);
        assert_mat_eq(&reconstructed, &a, 1e-10);
    }

    #[test]
    fn test_eigh_not_square() {
        let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let result = B::eigh(&a);
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // det tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_det_identity() {
        let eye = array![[1.0, 0.0], [0.0, 1.0]];
        let d = B::det(&eye).unwrap();
        assert_relative_eq!(d, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_det_known() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let d = B::det(&a).unwrap();
        assert_relative_eq!(d, -2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_det_singular() {
        let a = array![[1.0, 2.0], [2.0, 4.0]];
        let d = B::det(&a).unwrap();
        assert_relative_eq!(d, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_det_not_square() {
        let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let result = B::det(&a);
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // inv tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_inv_identity() {
        let eye = array![[1.0, 0.0], [0.0, 1.0]];
        let inv = B::inv(&eye).unwrap();
        assert_mat_eq(&inv, &eye, 1e-12);
    }

    #[test]
    fn test_inv_known() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let inv = B::inv(&a).unwrap();
        let expected = array![[-2.0, 1.0], [1.5, -0.5]];
        assert_mat_eq(&inv, &expected, 1e-10);
    }

    #[test]
    fn test_inv_roundtrip() {
        let a = array![[4.0, 7.0], [2.0, 6.0]];
        let inv = B::inv(&a).unwrap();
        let product = a.dot(&inv);
        let eye = array![[1.0, 0.0], [0.0, 1.0]];
        assert_mat_eq(&product, &eye, 1e-10);
    }

    #[test]
    fn test_inv_not_square() {
        let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let result = B::inv(&a);
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // Cross-operation consistency tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_solve_via_inv() {
        let a = array![[2.0, 1.0], [1.0, 3.0]];
        let b = array![5.0, 7.0];
        let x_solve = B::solve(&a, &b).unwrap();
        let a_inv = B::inv(&a).unwrap();
        let x_inv = a_inv.dot(&b);
        assert_vec_eq(&x_solve, &x_inv, 1e-10);
    }

    #[test]
    fn test_det_via_eigh() {
        let a = array![[4.0, 2.0], [2.0, 3.0]];
        let det_direct = B::det(&a).unwrap();
        let (eigenvalues, _) = B::eigh(&a).unwrap();
        let det_from_eig: f64 = eigenvalues.iter().product();
        assert_relative_eq!(det_direct, det_from_eig, epsilon = 1e-10);
    }

    #[test]
    fn test_backend_is_send_sync() {
        fn assert_send_sync<T: Send + Sync + 'static>() {}
        assert_send_sync::<BLASBackend>();
    }

    // -----------------------------------------------------------------------
    // Helpers for reconstruction
    // -----------------------------------------------------------------------

    /// Reconstruct a matrix from its SVD: A = U * diag(S) * Vt.
    fn reconstruct_svd(u: &Array2<f64>, s: &Array1<f64>, vt: &Array2<f64>) -> Array2<f64> {
        let m = u.nrows();
        let n = vt.ncols();
        let k = s.len();
        let mut result = Array2::zeros((m, n));
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for l in 0..k {
                    sum += u[[i, l]] * s[l] * vt[[l, j]];
                }
                result[[i, j]] = sum;
            }
        }
        result
    }

    /// Reconstruct a matrix from symmetric eigendecomposition.
    fn reconstruct_eigh(eigenvalues: &Array1<f64>, v: &Array2<f64>) -> Array2<f64> {
        let n = eigenvalues.len();
        let mut result = Array2::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += v[[i, k]] * eigenvalues[k] * v[[j, k]];
                }
                result[[i, j]] = sum;
            }
        }
        result
    }
}
