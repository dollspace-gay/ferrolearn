//! VALUE-divergence pins for `ferrolearn-decomp::NMF` on the DETERMINISTIC
//! `init='nndsvd'`, `solver='cd'` path vs scikit-learn 1.5.2
//! `class NMF` (`sklearn/decomposition/_nmf.py:912`). Tracking: #2393.
//!
//! Unlike `divergence_nmf.rs` (structural green-guards that carve out exact
//! VALUES under REQ-5), this file PINS the exact values on the fully
//! deterministic sklearn-default path: `init='nndsvd'` (SVD-based, no RNG once
//! `random_state` is fixed) + `solver='cd'`. Every expected number below is
//! the LIVE sklearn 1.5.2 oracle (run from `/tmp`, R-CHAR-3), NEVER copied
//! from the ferrolearn side.
//!
//! Oracle reproduction (sklearn 1.5.2):
//! ```python
//! import numpy as np
//! from sklearn.decomposition import NMF
//! rng = np.random.RandomState(0)
//! X = (rng.rand(12, 6) * 5).round(3)        # the FIXTURE below
//! m = NMF(n_components=3, init='nndsvd', solver='cd',
//!         max_iter=200, tol=1e-4, random_state=0)
//! W = m.fit_transform(X)
//! m.components_; m.reconstruction_err_; m.n_iter_; W
//! ```
//!
//! Findings (all FAIL against current `nmf.rs`):
//!  - #2394 components_ (H) diverges ~5x (NNDSVD init has no SVD pos/neg split;
//!    `init_nndsvd` uses Jacobi(X'X)+clamp, `nmf.rs:341`).
//!  - #2395 n_iter_ = 50 vs sklearn 151 (recon-err-delta tol vs sklearn's
//!    violation-based CD stopping rule, `_nmf.py:410`).
//!  - #2396 reconstruction_err_ off > 1e-6 (5.5147 vs 5.5136).
//!  - #2397 transform W ~5.4x off (0.1-seeded MU loop, `nmf.rs:795`, vs the
//!    sklearn NNLS W of `transform`, `_nmf.py:1213`).
//!
//! NMF is identifiable only up to permutation/scaling — BUT init='nndsvd' is
//! deterministic, so sklearn lands on ONE specific local optimum and the
//! VALUES are a contract, not a carve-out. The ferrolearn W is uniformly
//! ~5.4x larger and H ~5.2x smaller (so W@H — hence recon_err — is only
//! mildly off), which is exactly the signature of a missing init/transform
//! scaling, not a benign permutation.

use ferrolearn_core::traits::{Fit, Transform};
use ferrolearn_decomp::{NMF, NMFInit, NMFSolver};
use ndarray::{Array2, array};

/// Deterministic 12x6 non-negative fixture: `(RandomState(0).rand(12,6)*5).round(3)`.
fn fixture_12x6() -> Array2<f64> {
    array![
        [2.744, 3.576, 3.014, 2.724, 2.118, 3.229],
        [2.188, 4.459, 4.818, 1.917, 3.959, 2.644],
        [2.84, 4.628, 0.355, 0.436, 0.101, 4.163],
        [3.891, 4.35, 4.893, 3.996, 2.307, 3.903],
        [0.591, 3.2, 0.717, 4.723, 2.609, 2.073],
        [1.323, 3.871, 2.281, 2.842, 0.094, 3.088],
        [3.06, 3.085, 4.719, 3.409, 1.798, 2.185],
        [3.488, 0.301, 3.334, 3.353, 1.052, 0.645],
        [1.577, 1.819, 2.851, 2.193, 4.942, 0.51],
        [1.044, 0.807, 3.266, 1.266, 2.332, 1.222],
        [0.795, 0.552, 3.282, 0.691, 0.983, 1.844],
        [4.105, 0.486, 4.19, 0.48, 4.882, 2.343],
    ]
}

/// Fit on the fixture with the deterministic sklearn-default CD+NNDSVD path.
fn fit_cd_nndsvd() -> ferrolearn_decomp::FittedNMF<f64> {
    NMF::<f64>::new(3)
        .with_solver(NMFSolver::CoordinateDescent)
        .with_init(NMFInit::Nndsvd)
        .with_max_iter(200)
        .with_tol(1e-4)
        .with_random_state(0)
        .fit(&fixture_12x6(), &())
        .expect("fit should succeed on the 12x6 non-negative fixture")
}

/// Divergence (#2394): `NMF(n_components=3, init='nndsvd', solver='cd')`
/// `.components_` diverges from sklearn `_initialize_nmf` NNDSVD
/// (`sklearn/decomposition/_nmf.py:321-360`: `randomized_svd` + pos/neg-part
/// split + `lbd = sqrt(S[j]*sigma)` scaling) feeding `_fit_coordinate_descent`.
/// ferrolearn `init_nndsvd` (`nmf.rs:341`) instead does Jacobi(X'X)+clamp with
/// no SVD scaling, so the CD lands on a differently-scaled factorization.
/// sklearn components_[0] = [1.6531963561786642, 0.03253488093304504,
/// 2.504223159640572, 0.07478966190269389, 2.2876705631651566,
/// 0.6092353143399691]; ferrolearn returns ~[0.318, 0.0147, 0.487, ...] (~5.2x
/// smaller). Expected values are the LIVE sklearn 1.5.2 oracle.
#[test]
fn divergence_components_cd_nndsvd() {
    let fitted = fit_cd_nndsvd();
    // sklearn 1.5.2 oracle, m.components_.tolist() (row 0):
    let sk_comp_row0 = [
        1.6531963561786642,
        0.03253488093304504,
        2.504223159640572,
        0.07478966190269389,
        2.2876705631651566,
        0.6092353143399691,
    ];
    let h = fitted.components();
    assert_eq!(h.dim(), (3, 6), "components_ shape");
    for (j, &expected) in sk_comp_row0.iter().enumerate() {
        let actual = h[[0, j]];
        assert!(
            (actual - expected).abs() <= 1e-6,
            "components_[0][{j}]: sklearn {expected}, ferrolearn {actual}"
        );
    }
}

/// Divergence (#2395): `n_iter_` for CD+NNDSVD. sklearn's
/// `_fit_coordinate_descent` (`sklearn/decomposition/_nmf.py:410`) stops on a
/// VIOLATION-based criterion (`violation / violation_init <= tol`), reaching
/// `n_iter_ == 151` on this fixture. ferrolearn `solve_coordinate_descent`
/// (`nmf.rs:563`) stops when the reconstruction-error DELTA drops below `tol`,
/// reaching only 50. Expected value is the LIVE sklearn 1.5.2 oracle.
#[test]
fn divergence_n_iter_cd_nndsvd() {
    let fitted = fit_cd_nndsvd();
    // sklearn 1.5.2 oracle: m.n_iter_ == 151.
    let sk_n_iter = 151usize;
    assert_eq!(
        fitted.n_iter(),
        sk_n_iter,
        "n_iter_: sklearn {sk_n_iter}, ferrolearn {}",
        fitted.n_iter()
    );
}

/// Divergence (#2396): `reconstruction_err_` for CD+NNDSVD. sklearn defines it
/// as `sqrt(2 * _beta_divergence(X, W, H, 2)) == ||X - W H||_F`
/// (`sklearn/decomposition/_nmf.py:1657`), yielding 5.513563243249451 on this
/// fixture. ferrolearn returns 5.514708952320657 — off by ~1.15e-3, well
/// beyond the ~1e-6 parity bar (R-DEV-1) because the CD converges to a
/// different local optimum (downstream of the wrong NNDSVD init / stopping
/// rule). Expected value is the LIVE sklearn 1.5.2 oracle.
#[test]
fn divergence_reconstruction_err_cd_nndsvd() {
    let fitted = fit_cd_nndsvd();
    // sklearn 1.5.2 oracle: m.reconstruction_err_ == 5.513563243249451.
    let sk_recon = 5.513563243249451_f64;
    let actual = fitted.reconstruction_err();
    assert!(
        (actual - sk_recon).abs() <= 1e-6,
        "reconstruction_err_: sklearn {sk_recon}, ferrolearn {actual} (diff {})",
        (actual - sk_recon).abs()
    );
}

/// Divergence (#2397): `transform(X)` (the W document-topic matrix) for
/// CD+NNDSVD. sklearn `NMF.transform` (`sklearn/decomposition/_nmf.py:1213`)
/// solves `W` with `H` fixed via `_fit_transform(X, H=components_,
/// update_H=False)`: for the `cd` solver `W` starts at ZEROS (`_nmf.py:1254`)
/// and only `W` is updated, re-solving the NNLS `min_{W>=0} ||X - W·H||²`.
///
/// NOTE: sklearn's own `transform(X)` does NOT equal `fit_transform`'s W on the
/// training data — the re-solve has its own (looser) violation-ratio stop, so
/// sklearn's `transform W[0] = [0.7208388473417567, 1.29584414848054,
/// 0.7663490877861501]` differs by ~2e-5 from `fit_transform W[0] =
/// [0.7208617..., 1.2958572..., 0.7663393...]`. We pin to sklearn's `TRANSFORM`
/// output (the contract `NMF.transform` actually returns), the faithful target.
/// ferrolearn previously ran 200 multiplicative-update steps from a constant
/// `0.1` seed (`nmf.rs`), producing a W uniformly ~5.4x too large (~[3.92, 4.28,
/// 2.64]). Expected values are the LIVE sklearn 1.5.2 oracle
/// (`m.transform(X)[0].tolist()`, run from /tmp, R-CHAR-3).
#[test]
fn divergence_transform_w_cd_nndsvd() {
    let fitted = fit_cd_nndsvd();
    let w = fitted
        .transform(&fixture_12x6())
        .expect("transform should succeed");
    assert_eq!(w.dim(), (12, 3), "transform W shape");
    // sklearn 1.5.2 oracle: m.transform(X)[0].tolist() (the re-solve W, the
    // value sklearn's NMF.transform actually returns — NOT fit_transform's W).
    let sk_w_row0 = [0.7208388473417567, 1.29584414848054, 0.7663490877861501];
    for (k, &expected) in sk_w_row0.iter().enumerate() {
        let actual = w[[0, k]];
        assert!(
            (actual - expected).abs() <= 1e-6,
            "transform W[0][{k}]: sklearn {expected}, ferrolearn {actual}"
        );
    }
}
