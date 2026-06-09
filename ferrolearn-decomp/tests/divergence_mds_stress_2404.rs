//! Divergence audit (#2404): `ferrolearn_decomp::MDS::stress()` vs scikit-learn
//! 1.5.2 `MDS.stress_` (`sklearn/manifold/_mds.py`).
//!
//! DETERMINISTIC surface only (the SMACOF init is RNG / non-parity; classical
//! MDS is RNG-free). The fixtures below are PRECOMPUTED non-Euclidean distance
//! matrices with DISTINCT top-2 eigenvalues, so the classical-MDS embedding
//! (`classical_mds`, `mds.rs:239`) is uniquely determined up to per-axis sign,
//! and every quantity asserted (a pairwise-distance functional) is sign- and
//! eigen-basis-invariant — i.e. independent of faer vs numpy eigenvector
//! conventions.
//!
//! Oracle: live sklearn 1.5.2 (`python3 -c`), NOT copied from ferrolearn
//! (R-CHAR-3). The expected values are sklearn's stress FORMULAS evaluated on
//! the SAME classical-MDS embedding ferrolearn produces (so the only thing
//! under test is the stress DEFINITION, not the embedding algorithm).

use ferrolearn_core::traits::Fit;
use ferrolearn_decomp::{Dissimilarity, MDS};
use ndarray::{Array2, array};

/// The shared discriminating fixture: a 4x4 symmetric non-Euclidean distance
/// matrix whose double-centred Gram matrix `B` has DISTINCT eigenvalues
/// (≈ 40.888547, 8.999393, ~0, -7.137939 — live `np.linalg.eigh`), so the
/// rank-2 classical-MDS embedding is unique up to sign and the embedded
/// pairwise distances are deterministic across eigensolvers.
fn fixture_d() -> Array2<f64> {
    array![
        [0.0, 2.0, 5.0, 9.0],
        [2.0, 0.0, 3.0, 4.0],
        [5.0, 3.0, 0.0, 6.0],
        [9.0, 4.0, 6.0, 0.0],
    ]
}

/// Divergence: ferrolearn's `FittedMDS::stress()` diverges from sklearn 1.5.2
/// `MDS.stress_` for the METRIC default.
///
/// sklearn `_smacof_single` (`sklearn/manifold/_mds.py:147`):
///     `stress = ((dis.ravel() - disparities.ravel()) ** 2).sum() / 2`
/// with `disparities = dissimilarities` for `metric=True` (`_mds.py:131`) and
/// `normalized_stress='auto'` → `False` for metric MDS (`_mds.py:331-332`).
/// So `MDS().stress_` is the RAW sum-of-squared-residuals / 2 over the FULL
/// matrix.
///
/// ferrolearn `kruskal_stress` (`mds.rs:195`) returns the Kruskal STRESS-1
/// `sqrt(Σ_{i<j}(d_o-d_e)² / Σ_{i<j} d_o²)` — a DIFFERENT, normalized scalar.
///
/// Live sklearn 1.5.2 oracle on the SAME classical-MDS embedding of `fixture_d`
/// (`python3 -c`, _mds.py:147 formula):
///     RAW metric `stress_` = 5.33696816976651
/// ferrolearn `stress()` returns ≈ 0.176664 (Kruskal-1) for this fixture.
///
/// Tracking: #1455 (REQ-8 `stress_` raw-SSR definition).
#[test]
#[ignore = "divergence: MDS::stress() returns Kruskal-1 not sklearn raw SSR/2; tracking #1455"]
fn divergence_mds_stress_metric_raw_ssr() {
    // sklearn 1.5.2 raw metric stress_ on this embedding (_mds.py:147).
    const SK_RAW_STRESS: f64 = 5.336_968_169_766_51;

    let fitted = MDS::new(2)
        .with_dissimilarity(Dissimilarity::Precomputed)
        .fit(&fixture_d(), &())
        .expect("fit must succeed on the precomputed fixture");

    let got = fitted.stress();
    assert!(
        (got - SK_RAW_STRESS).abs() <= 1e-6,
        "MDS::stress() = {got}, sklearn metric MDS.stress_ (raw SSR/2, \
         _mds.py:147) = {SK_RAW_STRESS}; |diff| = {}",
        (got - SK_RAW_STRESS).abs()
    );
}

/// Divergence (corroborating): ferrolearn's `stress()` is also NOT sklearn's
/// `normalized_stress=True` Stress-1 in general. sklearn's normalized stress
/// (`_mds.py:149`) is `sqrt(stress / ((disparities**2).sum() / 2))` over the
/// FULL matrix; ferrolearn's Kruskal-1 (`mds.rs:213`) sums only `i<j`. They
/// coincide ONLY when `dis`/`disparities` are symmetric with zero diagonal
/// (a coincidence of the metric case). This test pins the RAW-vs-anything gap
/// via a second, independent fixture to guard against fixture-specific luck.
///
/// Live sklearn 1.5.2 oracle (`python3 -c`) on the classical-MDS embedding of
/// the regular-simplex distance matrix `[[0,1,1,1],[1,0,1,1],[1,1,0,1],
/// [1,1,1,0]]` reduced to 2D is eigen-basis dependent (degenerate), so this
/// test instead reuses the DISTINCT-eigenvalue `fixture_d`: sklearn raw
/// `stress_` = 5.33696816976651 (metric default), confirming ferrolearn's
/// ~0.1767 is the normalized value, not the reported default.
///
/// Tracking: #1455.
#[test]
#[ignore = "divergence: MDS reports normalized stress; sklearn metric default reports raw; tracking #1455"]
fn divergence_mds_stress_default_is_raw_not_normalized() {
    // The sklearn metric DEFAULT (normalized_stress='auto' -> False) value.
    const SK_RAW_STRESS: f64 = 5.336_968_169_766_51;
    // What ferrolearn actually reports happens to equal sklearn's
    // normalized_stress=True value here (0.176664...), NOT the default.
    const SK_NORMALIZED_STRESS: f64 = 0.176_664_484_755_916;

    let fitted = MDS::new(2)
        .with_dissimilarity(Dissimilarity::Precomputed)
        .fit(&fixture_d(), &())
        .expect("fit must succeed");
    let got = fitted.stress();

    // sklearn's metric MDS reports the RAW value by default. ferrolearn must
    // match that default; it instead reports the normalized value.
    assert!(
        (got - SK_RAW_STRESS).abs() <= 1e-6,
        "MDS::stress() = {got} matches sklearn NORMALIZED ({SK_NORMALIZED_STRESS}), \
         not the metric DEFAULT raw stress_ {SK_RAW_STRESS} (_mds.py:147,331-332)"
    );
}
