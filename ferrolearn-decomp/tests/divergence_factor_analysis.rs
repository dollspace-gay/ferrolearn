//! Divergence + green-guard suite for `ferrolearn-decomp` `FactorAnalysis`
//! vs scikit-learn 1.5.2 `sklearn.decomposition.FactorAnalysis`
//! (`/home/doll/scikit-learn/sklearn/decomposition/_factor_analysis.py:42`).
//!
//! Both fit the FA model `Σ = WWᵀ + diag(ψ)` by EM but via different
//! formulations: sklearn an SVD-on-whitened-data EM
//! (`_factor_analysis.py:278-297`), ferrolearn a random-init posterior-mean EM
//! (`factor_analysis.rs:466-572`). The loading matrix `W` is identifiable only
//! up to an orthogonal rotation `W → WR` (CARVE-OUT, REQ-1) — but the
//! rotation-INVARIANT quantities `noise_variance_`, the implied covariance
//! `C = WWᵀ + diag(ψ)` (= sklearn `get_covariance()`), and the converged
//! log-likelihood MUST match if ferrolearn's EM reaches the same optimum
//! (REQ-2).
//!
//! All oracle expectations are from the live sklearn 1.5.2 LAPACK (deterministic)
//! solver run from /tmp (R-CHAR-3), never copied from ferrolearn.

use ferrolearn_core::traits::{Fit, Transform};
use ferrolearn_decomp::FactorAnalysis;
use ndarray::{Array1, Array2};

/// Probe 1 fixture (well-conditioned 10×4, 2-factor structure, distinct
/// per-feature noise). Matches `.design/decomp/factor_analysis.md` Probe 1.
fn probe1() -> Array2<f64> {
    Array2::from_shape_vec(
        (10, 4),
        vec![
            1.20, 0.50, 2.10, -0.30, -0.80, 1.40, -1.20, 0.90, 0.30, -0.60, 0.80, 1.10, 2.10, 0.20,
            3.40, -1.00, -1.50, 2.20, -2.10, 1.80, 0.60, -1.10, 1.50, 0.40, 1.80, 0.90, 2.60,
            -0.70, -0.40, 1.80, -0.90, 1.30, 0.90, -0.30, 1.20, 0.20, -1.10, 2.50, -1.70, 2.00,
        ],
    )
    .expect("probe1 fixture shape")
}

/// In-module `simple_data()` fixture reproduced (10×4). The headline task also
/// asked for this fixture; it is near-degenerate under sklearn LAPACK (noise
/// variances collapse to the `SMALL=1e-12` floor) so the well-conditioned
/// Probe 1 is the primary discriminator.
fn simple_data() -> Array2<f64> {
    Array2::from_shape_vec(
        (10, 4),
        vec![
            1.0, 2.0, 1.5, 3.0, 1.1, 2.1, 1.6, 3.1, 0.9, 1.9, 1.4, 2.9, 2.0, 4.0, 3.0, 6.0, 2.1,
            4.1, 3.1, 6.1, 1.9, 3.9, 2.9, 5.9, 0.5, 1.0, 0.7, 1.5, 0.4, 0.9, 0.6, 1.4, 0.6, 1.1,
            0.8, 1.6, 1.5, 3.0, 2.2, 4.5,
        ],
    )
    .expect("simple_data fixture shape")
}

/// Reconstruct the rotation-invariant implied covariance
/// `C = components·componentsᵀ + diag(noise_variance)` from ferrolearn's fitted
/// state. ferrolearn stores `components` as `(n_features, n_components)`, so
/// `C[i,j] = Σ_c W[i,c]·W[j,c] (+ ψ_i on the diagonal)` — equal to sklearn's
/// `get_covariance() = components_ᵀ·components_ + diag(noise_variance_)` with
/// `components_` shape `(n_components, n_features)`.
fn implied_covariance(w: &Array2<f64>, psi: &Array1<f64>) -> Array2<f64> {
    let (p, k) = (w.nrows(), w.ncols());
    let mut cov = Array2::<f64>::zeros((p, p));
    for i in 0..p {
        for j in 0..p {
            let mut s = 0.0;
            for c in 0..k {
                s += w[[i, c]] * w[[j, c]];
            }
            cov[[i, j]] = s;
        }
        cov[[i, i]] += psi[i];
    }
    cov
}

// ===========================================================================
// DIV-1 — the rotation-invariant parity check (REQ-2). FAILING.
// ===========================================================================
//
// sklearn `FactorAnalysis(n_components=2, svd_method='lapack').fit(probe1)`
// (`_factor_analysis.py:278-297`) and ferrolearn `FactorAnalysis::new(2)`
// (`factor_analysis.rs:466-572`) are fitting the SAME likelihood, so the
// rotation-INVARIANT implied covariance, noise_variance, and converged
// log-likelihood MUST agree. They do not: ferrolearn's posterior-mean EM
// converges (at n_iter=181, robustly across seeds {42, 7, None} and even with
// max_iter=100_000) to a STRICTLY LOWER-likelihood optimum, and the
// rotation-invariant implied covariance differs by tenths (not by a rotation,
// which Probe 2 of the design doc shows leaves the covariance bit-identical to
// 3.3e-16).
//
// Live sklearn 1.5.2 LAPACK oracle (R-CHAR-3), run from /tmp:
//   tol=1e-3 (matches ferrolearn default tol): n_iter_=26
//     noise_variance_ = [0.009962591317, 0.158130095209, 0.012643079339, 0.063252661937]
//     get_covariance() diag = [1.364839477029, 1.343938974198, 3.336181722712, 0.948086111619]
//     loglike_[-1] = -27.61873891552277,  score(X) = -2.7618738915776566
//   tol=1e-2 (sklearn default tol): n_iter_=15
//     noise_variance_ = [0.009165047048, 0.18929822983, 0.014331031175, 0.062762007729]
//     get_covariance() diag = [1.364869529838, 1.347659729737, 3.33640246802, 0.947929114262]
//     loglike_[-1] = -27.644987554315705, score(X) = -2.7644987554564664
// (The tol=1e-3 vs tol=1e-2 oracles differ only at the ~1e-2 level — both
//  describe the SAME optimum; ferrolearn diverges from BOTH by tenths.)
//
// ferrolearn `FactorAnalysis::<f64>::new(2).fit(probe1)`:
//     noise_variance ≈ [0.010907532780, 0.136719066001, 0.011264677579, 0.063639386477]
//     implied cov diag ≈ [1.131978459400, 1.232109886392, 2.761788280384, 0.800025331122]
//     log_likelihood ≈ -27.70783015220203  (LOWER than sklearn's optimum)
// Worst covariance-diag gap: |2.7618 − 3.3362| ≈ 0.574 (~17%).
//
// Candidate roots for the FIXER (do NOT fix here): the M-step psi update
// (`factor_analysis.rs:527-558`) vs sklearn `psi = max(var − sum(W², axis=0),
// SMALL)` (`_factor_analysis.py:297`); the `1e-6` psi floor (`:553`) vs sklearn
// `SMALL=1e-12` (`:252`/`:297`); the convergence criterion `|Δll|<tol`
// (`:565-567`) vs sklearn one-sided `(ll−old)<tol` (`:293`); the
// `compute_log_likelihood` formula (`:271`).
//
// Tracking: #1527

/// Divergence: ferrolearn `FactorAnalysis::fit`
/// (`ferrolearn-decomp/src/factor_analysis.rs:466-572`) converges to a
/// DIFFERENT (lower-likelihood) optimum than sklearn LAPACK
/// (`sklearn/decomposition/_factor_analysis.py:278-297`). The rotation-INVARIANT
/// implied covariance `WWᵀ + diag(ψ)` (= sklearn `get_covariance()`) diverges
/// by ~0.57 on the diagonal — far beyond rotation freedom (which is exact to
/// machine precision). Tracking: #1527
#[test]
fn divergence_fa_rotation_invariant_covariance() {
    // sklearn 1.5.2 LAPACK oracle, tol=1e-2 (== ferrolearn DEFAULT tol, now
    // matching sklearn `_factor_analysis.py:185`; was tol=1e-3 before #2392).
    #[allow(
        clippy::excessive_precision,
        reason = "live sklearn 1.5.2 oracle (R-CHAR-3)"
    )]
    let sk_cov_diag = [
        1.364869529838_f64,
        1.347659729737,
        3.33640246802,
        0.947929114262,
    ];
    #[allow(
        clippy::excessive_precision,
        reason = "live sklearn 1.5.2 oracle (R-CHAR-3)"
    )]
    let sk_noise = [
        0.009165047048_f64,
        0.18929822983,
        0.014331031175,
        0.062762007729,
    ];
    #[allow(
        clippy::excessive_precision,
        reason = "live sklearn 1.5.2 oracle (R-CHAR-3)"
    )]
    let sk_loglike: f64 = -27.644987554315705;

    let x = probe1();
    let fitted = FactorAnalysis::<f64>::new(2)
        .with_random_state(42)
        .fit(&x, &())
        .expect("fit probe1");
    let cov = implied_covariance(fitted.components(), fitted.noise_variance());
    let psi = fitted.noise_variance();

    // Tolerance 1e-2: loose enough that two EM runs converged to the SAME
    // optimum (sklearn tol=1e-3 vs tol=1e-2 oracles agree at ~1e-2) would pass,
    // but tight enough to catch the genuine ~0.57 covariance divergence.
    for (i, &exp) in sk_cov_diag.iter().enumerate() {
        let got = cov[[i, i]];
        assert!(
            (got - exp).abs() < 1e-2,
            "covariance diag[{i}]: sklearn {exp}, ferrolearn {got} (diff {})",
            (got - exp).abs()
        );
    }
    for (i, &exp) in sk_noise.iter().enumerate() {
        let got = psi[i];
        assert!(
            (got - exp).abs() < 1e-2,
            "noise_variance[{i}]: sklearn {exp}, ferrolearn {got} (diff {})",
            (got - exp).abs()
        );
    }
    let got_ll = fitted.log_likelihood();
    assert!(
        (got_ll - sk_loglike).abs() < 1e-2,
        "log_likelihood: sklearn {sk_loglike}, ferrolearn {got_ll} (diff {})",
        (got_ll - sk_loglike).abs()
    );
}

/// Divergence (same root, simple_data fixture, near-degenerate). sklearn LAPACK
/// drives the noise variances to the `SMALL=1e-12` floor and reaches
/// `loglike_[-1] = 147.56994485222165` (n_iter_=62, identical at tol=1e-3 and
/// tol=1e-2); the implied covariance diag is `[0.366, 1.446, 0.8436, 3.246]`.
/// ferrolearn does not reach that optimum. Tracking: #1527
#[test]
fn divergence_fa_simple_data_loglike() {
    // sklearn 1.5.2 LAPACK oracle (identical at tol=1e-3 and tol=1e-2).
    #[allow(
        clippy::excessive_precision,
        reason = "live sklearn 1.5.2 oracle (R-CHAR-3)"
    )]
    let sk_loglike: f64 = 147.56994485222165;
    #[allow(
        clippy::excessive_precision,
        reason = "live sklearn 1.5.2 oracle (R-CHAR-3)"
    )]
    let sk_cov_diag = [0.366_f64, 1.446, 0.8436, 3.246];

    let x = simple_data();
    let fitted = FactorAnalysis::<f64>::new(2)
        .with_random_state(42)
        .fit(&x, &())
        .expect("fit simple_data");
    let cov = implied_covariance(fitted.components(), fitted.noise_variance());

    let got_ll = fitted.log_likelihood();
    assert!(
        (got_ll - sk_loglike).abs() < 1e-1,
        "log_likelihood: sklearn {sk_loglike}, ferrolearn {got_ll} (diff {})",
        (got_ll - sk_loglike).abs()
    );
    for (i, &exp) in sk_cov_diag.iter().enumerate() {
        let got = cov[[i, i]];
        assert!(
            (got - exp).abs() < 1e-1,
            "covariance diag[{i}]: sklearn {exp}, ferrolearn {got} (diff {})",
            (got - exp).abs()
        );
    }
}

// ===========================================================================
// CARVE-OUT (REQ-1) — the W-VALUE divergence is NOT a defect: assert ONLY the
// rotation-invariant property structurally (the covariance is symmetric PSD and
// the loadings live in (n_features, n_components)). This is a PASSING guard that
// documents that we do NOT pin W element-wise (rotation freedom, R-DEFER-3).
// ===========================================================================

#[test]
fn carveout_fa_loadings_only_rotation_invariant() {
    // We deliberately do NOT compare `components()` element-wise to sklearn's
    // `components_ row0 = [-1.162373, 0.790449, -1.821601, 0.927488]`: those
    // loadings are identifiable only up to W→WR (design-doc Probe 2). We only
    // assert the implied covariance is symmetric (the rotation-invariant form).
    let x = probe1();
    let fitted = FactorAnalysis::<f64>::new(2)
        .with_random_state(42)
        .fit(&x, &())
        .expect("fit probe1");
    let cov = implied_covariance(fitted.components(), fitted.noise_variance());
    let p = cov.nrows();
    for i in 0..p {
        for j in 0..p {
            assert!(
                (cov[[i, j]] - cov[[j, i]]).abs() < 1e-12,
                "implied covariance must be symmetric"
            );
        }
    }
}

// ===========================================================================
// STRUCTURAL GREEN-GUARDS — must PASS against current code.
// ===========================================================================

#[test]
fn green_components_shape_is_features_by_components() {
    // ferrolearn stores W as (n_features, n_components) = sklearn components_.T.
    let x = probe1();
    let fitted = FactorAnalysis::<f64>::new(2).fit(&x, &()).expect("fit");
    assert_eq!(fitted.components().dim(), (4, 2));
}

#[test]
fn green_transform_scores_shape() {
    let x = probe1();
    let fitted = FactorAnalysis::<f64>::new(2).fit(&x, &()).expect("fit");
    let scores = fitted.transform(&x).expect("transform");
    assert_eq!(scores.dim(), (10, 2));
}

#[test]
fn green_noise_variance_strictly_positive() {
    let x = probe1();
    let fitted = FactorAnalysis::<f64>::new(2).fit(&x, &()).expect("fit");
    for &v in fitted.noise_variance() {
        assert!(v > 0.0, "noise variance must be strictly positive, got {v}");
    }
}

#[test]
fn green_log_likelihood_finite_and_n_iter_in_range() {
    let x = probe1();
    let max_iter = 1000;
    let fitted = FactorAnalysis::<f64>::new(2)
        .with_max_iter(max_iter)
        .fit(&x, &())
        .expect("fit");
    assert!(fitted.log_likelihood().is_finite());
    assert!(fitted.n_iter() >= 1 && fitted.n_iter() <= max_iter);
}

#[test]
fn green_determinism_same_seed_identical() {
    let x = probe1();
    let f1 = FactorAnalysis::<f64>::new(2)
        .with_random_state(42)
        .fit(&x, &())
        .expect("fit1");
    let f2 = FactorAnalysis::<f64>::new(2)
        .with_random_state(42)
        .fit(&x, &())
        .expect("fit2");
    for (a, b) in f1.components().iter().zip(f2.components().iter()) {
        assert!(
            (a - b).abs() < 1e-15,
            "components must be identical for same seed"
        );
    }
    for (a, b) in f1.noise_variance().iter().zip(f2.noise_variance().iter()) {
        assert!(
            (a - b).abs() < 1e-15,
            "noise_variance must be identical for same seed"
        );
    }
}

#[test]
fn green_error_zero_components() {
    let x = probe1();
    assert!(FactorAnalysis::<f64>::new(0).fit(&x, &()).is_err());
}

#[test]
fn green_error_too_many_components() {
    let x = probe1();
    assert!(FactorAnalysis::<f64>::new(5).fit(&x, &()).is_err());
}

#[test]
fn green_error_insufficient_samples() {
    let x = Array2::<f64>::from_shape_vec((1, 4), vec![1.0, 2.0, 3.0, 4.0]).expect("shape");
    assert!(FactorAnalysis::<f64>::new(1).fit(&x, &()).is_err());
}

#[test]
fn green_error_transform_col_mismatch() {
    let x = probe1();
    let fitted = FactorAnalysis::<f64>::new(2).fit(&x, &()).expect("fit");
    let bad = Array2::<f64>::zeros((3, 7));
    assert!(fitted.transform(&bad).is_err());
}

#[test]
fn green_inverse_transform_shape_and_col_mismatch() {
    let x = probe1();
    let fitted = FactorAnalysis::<f64>::new(2).fit(&x, &()).expect("fit");
    // Correct shape: z is (n_samples, n_components) -> (n_samples, n_features).
    let z = Array2::<f64>::zeros((5, 2));
    let recon = fitted.inverse_transform(&z).expect("inverse_transform");
    assert_eq!(recon.dim(), (5, 4));
    // Column mismatch -> Err.
    let z_bad = Array2::<f64>::zeros((5, 3));
    assert!(fitted.inverse_transform(&z_bad).is_err());
}

// ===========================================================================
// DIV-3 — DEFAULT `tol` parity (REQ-16). FAILING.
// ===========================================================================
//
// sklearn `FactorAnalysis` documents and constructs with `tol=1e-2`
// (`sklearn/decomposition/_factor_analysis.py:72` "tol : float, default=1e-2",
// `:185` `tol=1e-2`). ferrolearn's `FactorAnalysis::new`
// (`ferrolearn-decomp/src/factor_analysis.rs:134-144`) constructs with
// `tol: 1e-3`. Because the EM stops on the one-sided criterion
// `(ll - old_ll) < tol` (`_factor_analysis.py:293`), the wrong default tol
// makes the DEFAULT-constructed estimator iterate further and stop at a
// different fitted state: it reports a DIFFERENT `n_iter_` and a DIFFERENT
// final `loglike_`/`log_likelihood` than the default-constructed sklearn
// estimator on the same data (R-DEV-1 fitted-attribute parity / R-DEV-2
// constructor-default parity).
//
// Live sklearn 1.5.2 LAPACK oracle (R-CHAR-3), run from /tmp, sklearn DEFAULTS
// `FactorAnalysis(n_components=2, svd_method='lapack')` on probe1:
//   n_iter_       = 15
//   loglike_[-1]  = -27.644987554315705
//   (noise_variance_ = [0.009165047048366848, 0.1892982298298851,
//                       0.014331031175260911, 0.06276200772936313])
//
// ferrolearn DEFAULT `FactorAnalysis::<f64>::new(2)` on probe1 (tol=1e-3):
//   n_iter()         = 26
//   log_likelihood() = -27.618738915522783
// (Cross-check: when ferrolearn's tol is explicitly set to sklearn's default
//  1e-2 via `.with_tol(1e-2)` the EM matches sklearn to ~1e-13 — n_iter=15,
//  ll=-27.644987554315730 — confirming the divergence is the DEFAULT tol value,
//  not the EM math. The FIXER must change `new()`'s default tol to 1e-2; do NOT
//  fix here.)
//
// Tracking: #2392 (relates #1536, #2391)

/// Divergence: ferrolearn's default-constructed `FactorAnalysis::new`
/// (`ferrolearn-decomp/src/factor_analysis.rs:140`, `tol: 1e-3`) diverges from
/// sklearn's default `FactorAnalysis` (`_factor_analysis.py:185`, `tol=1e-2`).
/// On probe1 sklearn DEFAULT reports `n_iter_=15`,
/// `loglike_[-1]=-27.644987554315705`; ferrolearn DEFAULT reports `n_iter()=26`,
/// `log_likelihood()=-27.618738915522783`. Tracking: #2392
#[test]
fn divergence_fa_default_tol_n_iter_and_loglike() {
    // Live sklearn 1.5.2 LAPACK oracle, sklearn DEFAULTS (tol=1e-2). R-CHAR-3.
    let sk_n_iter: usize = 15;
    #[allow(
        clippy::excessive_precision,
        reason = "live sklearn 1.5.2 oracle (R-CHAR-3)"
    )]
    let sk_loglike: f64 = -27.644987554315705;

    let x = probe1();
    // DEFAULT-constructed ferrolearn estimator (no `.with_tol`) — must mirror a
    // default-constructed sklearn `FactorAnalysis`.
    let fitted = FactorAnalysis::<f64>::new(2)
        .fit(&x, &())
        .expect("fit probe1 default");

    assert_eq!(
        fitted.n_iter(),
        sk_n_iter,
        "n_iter_: sklearn default (tol=1e-2) {sk_n_iter}, ferrolearn default {}",
        fitted.n_iter()
    );
    let got_ll = fitted.log_likelihood();
    assert!(
        (got_ll - sk_loglike).abs() < 1e-6,
        "final loglike: sklearn default (tol=1e-2) {sk_loglike}, ferrolearn default {got_ll} (diff {})",
        (got_ll - sk_loglike).abs()
    );
}

// ---------------------------------------------------------------------------
// GREEN GUARD — the EM math itself is CLEAN when tol is matched to sklearn's
// default. This PASSES today and documents that the DIV-3 divergence is purely
// the default tol value, not the SVD-EM / noise_variance / loglike update.
// ---------------------------------------------------------------------------

/// Green guard: with `tol` explicitly set to sklearn's default 1e-2,
/// ferrolearn's EM matches the live sklearn 1.5.2 LAPACK oracle (n_iter,
/// final loglike, noise_variance) to ~1e-6. Confirms the EM update is correct.
#[test]
fn green_fa_matched_tol_matches_sklearn() {
    // Live sklearn 1.5.2 LAPACK oracle, tol=1e-2. R-CHAR-3.
    let sk_n_iter: usize = 15;
    #[allow(
        clippy::excessive_precision,
        reason = "live sklearn 1.5.2 oracle (R-CHAR-3)"
    )]
    let sk_loglike: f64 = -27.644987554315705;
    #[allow(
        clippy::excessive_precision,
        reason = "live sklearn 1.5.2 oracle (R-CHAR-3)"
    )]
    let sk_noise = [
        0.009165047048366848_f64,
        0.1892982298298851,
        0.014331031175260911,
        0.06276200772936313,
    ];

    let x = probe1();
    let fitted = FactorAnalysis::<f64>::new(2)
        .with_tol(1e-2)
        .fit(&x, &())
        .expect("fit probe1 tol=1e-2");

    assert_eq!(fitted.n_iter(), sk_n_iter);
    assert!((fitted.log_likelihood() - sk_loglike).abs() < 1e-6);
    for (i, &exp) in sk_noise.iter().enumerate() {
        assert!(
            (fitted.noise_variance()[i] - exp).abs() < 1e-6,
            "noise_variance[{i}]: sklearn {exp}, ferrolearn {}",
            fitted.noise_variance()[i]
        );
    }
}
