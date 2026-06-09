//! Divergence pins for `FastICA` fitted-attribute SEMANTICS vs scikit-learn
//! 1.5.2 (`sklearn/decomposition/_fastica.py`), audit #2410.
//!
//! The existing `divergence_fast_ica.rs` concludes (lines 38-60) that the
//! `mixing_`/`components_` differences are "attribute-semantics, not a transform
//! divergence" and that "NO deterministic, observable, non-RNG-coupled,
//! non-identifiability-gated numeric divergence ... was found". This audit
//! REFUTES that for `mixing_` and `components_`: sklearn pins two DETERMINISTIC,
//! observable, RNG- and sign/permutation/scale-INVARIANT contracts that
//! ferrolearn's exposed attributes violate by O(1) (errors ~8.7 and ~3.5 vs
//! sklearn ~1e-15). These hold for ANY converged ICA solution and so are not
//! gated by the `w_init` RNG, the eigh-vs-svd whitening convention, or ICA
//! identifiability.
//!
//! Expected values come from the LIVE sklearn 1.5.2 oracle (R-CHAR-3); the
//! oracle scripts are quoted in each test. Each pin is `#[ignore]`d with its
//! tracking issue. Source-recovery quality (abs-corr ≈ 1) IS clean and is NOT
//! pinned (genuine ICA carve-out, REQ-4 #1572).
//!
//! Fixture: a deterministic 3-source mixed-signal problem (sine / square /
//! sawtooth) mixed by a fixed 3×3 matrix `A`, so the parallel iteration
//! converges. No RNG enters the fixture. Reproduced byte-for-byte from:
//! ```python
//! n=60; t=np.arange(n)*0.1
//! s1=np.sin(2*t); s2=np.sign(np.sin(3*t+0.3)); s3=((t%1.0)-0.5)
//! S=np.c_[s1,s2,s3]
//! A=np.array([[1.,1.,.5],[.5,2.,1.],[1.5,1.,2.]]); X=S.dot(A.T)
//! ```

use ferrolearn_core::traits::{Fit, Transform};
use ferrolearn_decomp::fast_ica::{Algorithm, FastICA, FittedFastICA, NonLinearity};
use ndarray::Array2;

/// Deterministic convergent mixed-signal fixture: 3 independent sources mixed
/// by a fixed 3×3 matrix. No RNG; closed-form. Matches the Python oracle above
/// (verified `np.allclose` against the saved `.npy`).
fn mixed_fixture() -> Array2<f64> {
    let n = 60usize;
    let mut x = Array2::<f64>::zeros((n, 3));
    let a = [[1.0, 1.0, 0.5], [0.5, 2.0, 1.0], [1.5, 1.0, 2.0]];
    for i in 0..n {
        let t = i as f64 * 0.1;
        let s1 = (2.0 * t).sin();
        let sv = (3.0 * t + 0.3).sin();
        let s2 = if sv > 0.0 {
            1.0
        } else if sv < 0.0 {
            -1.0
        } else {
            0.0
        };
        let s3 = (t % 1.0) - 0.5;
        let s = [s1, s2, s3];
        for r in 0..3 {
            x[[i, r]] = a[r][0] * s[0] + a[r][1] * s[1] + a[r][2] * s[2];
        }
    }
    x
}

fn fit(x: &Array2<f64>) -> FittedFastICA<f64> {
    FastICA::<f64>::new(3)
        .with_algorithm(Algorithm::Parallel)
        .with_fun(NonLinearity::LogCosh)
        .with_random_state(0)
        .fit(x, &())
        .unwrap()
}

/// Divergence: `FittedFastICA::mixing` diverges from sklearn `mixing_` at
/// `sklearn/decomposition/_fastica.py:689`
/// (`self.mixing_ = linalg.pinv(self.components_, check_finite=False)`).
///
/// sklearn's `mixing_` is the pseudo-inverse of `components_`, so it satisfies
/// the RNG/sign/permutation/scale-INVARIANT reconstruction contract
/// `X - mean_ == sources @ mixing_.T`. LIVE oracle:
/// ```python
/// from sklearn.decomposition import FastICA
/// W0=np.array([[.3,-.2,.7],[.1,.9,-.4],[-.6,.2,.5]])
/// ica=FastICA(3,algorithm='parallel',fun='logcosh',whiten='unit-variance',w_init=W0).fit(X)
/// S=ica.transform(X)
/// np.allclose(X-ica.mean_, S.dot(ica.mixing_.T))   # -> True, max err 1.8e-15
/// ```
/// ferrolearn computes `mixing = Kᵀ Wᵀ` (`fast_ica.rs:716`), where `K` is the
/// covariance-eigh whitening matrix (missing sklearn's `sqrt(n)` /
/// unit-variance scale). For this fixture the reconstruction error is ~8.71 —
/// `mixing` is NOT `pinv(components)`. RNG/sign/permutation INVARIANT.
///
/// Tracking: #2411
#[test]
#[ignore = "divergence: mixing != pinv(components), X_centered != S@mixing.T (err ~8.7 vs sklearn 1.8e-15); tracking #2411"]
fn divergence_fast_ica_mixing_pinv_contract() {
    let x = mixed_fixture();
    let f = fit(&x);
    let s = f.transform(&x).unwrap();
    let mean = f.mean();
    let mixing = f.mixing(); // ferrolearn: (n_features, k)
    let recon = s.dot(&mixing.t()); // (n_samples, n_features)

    let mut max_err = 0.0f64;
    for i in 0..x.nrows() {
        for j in 0..x.ncols() {
            let xc = x[[i, j]] - mean[j];
            let e = (xc - recon[[i, j]]).abs();
            if e > max_err {
                max_err = e;
            }
        }
    }
    // sklearn satisfies this to machine precision (max err ~1.8e-15).
    assert!(
        max_err < 1e-6,
        "sklearn mixing_ satisfies X_centered == S @ mixing_.T (max err ~1.8e-15); \
         ferrolearn max err = {max_err}"
    );
}

/// Divergence: `FittedFastICA::components` diverges from sklearn `components_`
/// at `sklearn/decomposition/_fastica.py:683` (`self.components_ = np.dot(W, K)`),
/// `:676-681` (the `unit-variance` `W /= S_std.T` rescale).
///
/// sklearn's `components_` (= `W @ K`, shape `(n_components, n_features)`,
/// unit-variance-rescaled) is the unmixing applied to *raw centered* data, so it
/// satisfies the RNG/sign/permutation-INVARIANT transform contract
/// `sources == (X - mean_) @ components_.T`. LIVE oracle:
/// ```python
/// ica=FastICA(3,algorithm='parallel',fun='logcosh',whiten='unit-variance',w_init=W0).fit(X)
/// S=ica.transform(X)
/// np.allclose(S, (X-ica.mean_).dot(ica.components_.T))   # -> True (atol 1e-10)
/// ```
/// ferrolearn's `components()` returns the k×k whitened-space unmixing `W`
/// (`fast_ica.rs:719`), NOT `W @ K`; therefore `(X-mean) @ components.T` does
/// not equal `transform(X)`. For this fixture the contract error is ~3.52.
///
/// Tracking: #2412
#[test]
#[ignore = "divergence: components is W not W@K; S != X_centered@components.T (err ~3.5 vs sklearn 1e-10); tracking #2412"]
fn divergence_fast_ica_components_transform_contract() {
    let x = mixed_fixture();
    let f = fit(&x);
    let s = f.transform(&x).unwrap();
    let mean = f.mean();
    let comp = f.components();

    let mut xc = x.clone();
    for mut row in xc.rows_mut() {
        for (v, &m) in row.iter_mut().zip(mean.iter()) {
            *v -= m;
        }
    }
    let pred = xc.dot(&comp.t());

    let mut max_err = 0.0f64;
    for i in 0..s.nrows() {
        for j in 0..s.ncols() {
            let e = (s[[i, j]] - pred[[i, j]]).abs();
            if e > max_err {
                max_err = e;
            }
        }
    }
    // sklearn satisfies sources == X_centered @ components_.T to ~1e-10.
    assert!(
        max_err < 1e-6,
        "sklearn components_ satisfies sources == X_centered @ components_.T; \
         ferrolearn max err = {max_err}"
    );
}
