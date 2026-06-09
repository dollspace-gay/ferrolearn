//! Parity (#2404/#2406): `ferrolearn_decomp::MDS::stress()` vs scikit-learn
//! 1.5.2 `MDS.stress_` (`sklearn/manifold/_mds.py`).
//!
//! CORRECTION (#2405 was pinned with the WRONG oracle): the prior pin used
//! `5.336968` — the CLASSICAL-MDS raw SSR over the OLD eigendecomposition
//! embedding. ferrolearn's `MDS` is now SMACOF (matching sklearn), so the
//! correct oracle is sklearn's FIXED-INIT SMACOF stress: with `init=X0` and
//! `n_init=1` the Guttman trajectory is deterministic and the final
//! `stress_` is `3.148219331054871` — NOT the classical `5.336968`.
//!
//! These are now GREEN-GUARDS (the divergence is FIXED): with a fixed init,
//! `MDS::with_init(X0).fit(D).stress()` matches sklearn `smacof(D, init=X0,
//! n_init=1, normalized_stress=False).stress` to ~1e-6.
//!
//! Oracle: live sklearn 1.5.2 (`python3 -c`), NOT copied from ferrolearn
//! (R-CHAR-3):
//! ```text
//! smacof(D, metric=True, init=X0, n_init=1, normalized_stress=False,
//!        max_iter=300, eps=1e-3, return_n_iter=True)
//!   -> stress = 3.148219331054871, n_iter = 13
//! # normalized_stress=True on the SAME run -> Kruskal-1 in (0, 1).
//! ```

use ferrolearn_core::traits::Fit;
use ferrolearn_decomp::{Dissimilarity, MDS};
use ndarray::{Array2, array};

/// The shared discriminating fixture: a 4x4 symmetric non-Euclidean distance
/// matrix.
fn fixture_d() -> Array2<f64> {
    array![
        [0.0, 2.0, 5.0, 9.0],
        [2.0, 0.0, 3.0, 4.0],
        [5.0, 3.0, 0.0, 6.0],
        [9.0, 4.0, 6.0, 0.0],
    ]
}

/// A fixed SMACOF init (the deterministic parity path).
fn fixed_init() -> Array2<f64> {
    array![[0.1, 0.2], [0.3, -0.1], [-0.2, 0.4], [0.5, 0.05]]
}

/// CORRECTED ORACLE (#2405): `MDS::with_init(X0).fit(D).stress()` matches
/// sklearn's FIXED-INIT SMACOF raw `stress_` (the metric default,
/// `normalized_stress='auto'` -> `False`, `_mds.py:147,331-332`).
///
/// Live sklearn 1.5.2 oracle (`python3 -c`):
///   `smacof(D, metric=True, init=X0, n_init=1, normalized_stress=False).stress`
///   = 3.148219331054871   (NOT the classical-MDS 5.336968).
#[test]
fn mds_stress_metric_raw_ssr_fixed_init_parity() {
    // sklearn 1.5.2 fixed-init SMACOF raw stress_ (_mds.py:147).
    const SK_RAW_STRESS: f64 = 3.148_219_331_054_871;

    let fitted = MDS::new(2)
        .with_dissimilarity(Dissimilarity::Precomputed)
        .with_init(fixed_init())
        .fit(&fixture_d(), &())
        .expect("fit must succeed on the precomputed fixture");

    let got = fitted.stress();
    assert!(
        (got - SK_RAW_STRESS).abs() <= 1e-6,
        "MDS::stress() = {got}, sklearn fixed-init SMACOF stress_ (raw SSR/2, \
         _mds.py:147) = {SK_RAW_STRESS}; |diff| = {}",
        (got - SK_RAW_STRESS).abs()
    );
}

/// The metric DEFAULT reports the RAW stress (not the normalized Stress-1).
/// With `with_normalized_stress(true)` the SAME run reports Kruskal Stress-1
/// `sqrt(raw / (Σ disparities²/2))` (`_mds.py:148-149`), a value in `(0, 1)`.
///
/// Live sklearn 1.5.2 oracle: raw `3.148...` (`> 1`) vs normalized Stress-1
/// `0.1356...` (`< 1`) — the two definitions are distinct magnitude classes.
#[test]
fn mds_stress_default_is_raw_not_normalized() {
    const SK_RAW_STRESS: f64 = 3.148_219_331_054_871;

    let raw = MDS::new(2)
        .with_dissimilarity(Dissimilarity::Precomputed)
        .with_init(fixed_init())
        .fit(&fixture_d(), &())
        .expect("fit must succeed");
    let norm = MDS::new(2)
        .with_dissimilarity(Dissimilarity::Precomputed)
        .with_init(fixed_init())
        .with_normalized_stress(true)
        .fit(&fixture_d(), &())
        .expect("fit must succeed");

    // The metric default reports the RAW value.
    assert!(
        (raw.stress() - SK_RAW_STRESS).abs() <= 1e-6,
        "default MDS::stress() = {} must be the raw SMACOF stress {SK_RAW_STRESS} \
         (_mds.py:147,331-332)",
        raw.stress()
    );
    // The normalized toggle reports Stress-1 in (0, 1).
    assert!(
        norm.stress() > 0.0 && norm.stress() < 1.0,
        "normalized_stress=true must report Kruskal-1 in (0,1), got {}",
        norm.stress()
    );
}
