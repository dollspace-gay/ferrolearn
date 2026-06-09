//! EDGE-case re-audit of the deterministic `init='nndsvd'`, `solver='cd'` NMF
//! path vs scikit-learn 1.5.2 (`sklearn/decomposition/_nmf.py:912`).
//! Tracking: #2398. Probes k=1, k=n_features, zero-row, rank-deficient (zero
//! singular value), max_iter=1, tol=1e-3 — n_iter_/recon/components vs the LIVE
//! sklearn 1.5.2 oracle (run from /tmp, R-CHAR-3), NEVER copied from ferrolearn.
//! Also asserts no-panic + finiteness (R-CODE-2) on every config.

use ferrolearn_core::traits::Fit;
use ferrolearn_decomp::{NMF, NMFInit, NMFSolver};
use ndarray::{Array2, array};

fn fit_cd(x: &Array2<f64>, k: usize, max_iter: usize, tol: f64) -> ferrolearn_decomp::FittedNMF<f64> {
    NMF::<f64>::new(k)
        .with_solver(NMFSolver::CoordinateDescent)
        .with_init(NMFInit::Nndsvd)
        .with_max_iter(max_iter)
        .with_tol(tol)
        .with_random_state(0)
        .fit(x, &())
        .expect("fit should succeed")
}

fn assert_close(name: &str, actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= 1e-6,
        "{name}: sklearn {expected}, ferrolearn {actual} (diff {})",
        (actual - expected).abs()
    );
}

fn assert_finite_nonneg(name: &str, h: &Array2<f64>) {
    for &v in h {
        assert!(v.is_finite() && v >= 0.0, "{name}: non-finite/neg {v}");
    }
}

fn x_8x5() -> Array2<f64> {
    array![
        [1.11, 4.354, 1.034, 4.593, 2.442],
        [3.059, 3.83, 2.592, 1.484, 0.939],
        [0.404, 3.692, 2.207, 0.792, 4.4],
        [1.37, 2.071, 1.48, 3.144, 2.899],
        [3.0, 1.329, 1.423, 1.268, 1.638],
        [0.721, 0.828, 4.82, 4.801, 0.942],
        [0.122, 1.023, 3.499, 3.898, 0.115],
        [2.888, 0.008, 2.577, 3.199, 4.928],
    ]
}

/// DIVERGENCE (#2398 residual): `n_iter_` for k=1 CD+NNDSVD on the 8x5 fixture.
/// sklearn `_fit_coordinate_descent` (`sklearn/decomposition/_nmf.py:498-525`)
/// stops only when `violation / violation_init <= tol`; for k=1 the violation
/// ratio bounces (verbose trace: 1.0, 0.419, 0.194, 0.935, 0.645, 0.290, 0.0)
/// and crosses the threshold only when it reaches EXACTLY 0.0 at iteration 7, so
/// sklearn reports `n_iter_ == 7`. ferrolearn's `solve_coordinate_descent`
/// (`nmf.rs:863`) accumulates the violation via `ndarray` `dot` in a different
/// FP order, so its ratio reaches 0 two sweeps earlier and it reports
/// `n_iter_ == 5`. The factorization itself converges to the SAME optimum
/// (components_/reconstruction_err_ match to ~1e-15, asserted below), so this is
/// a PURE `n_iter_` (public fitted-attribute) divergence, not a values drift.
/// Expected `n_iter_ == 7` is the LIVE sklearn 1.5.2 oracle (`m.n_iter_`).
#[test]
#[ignore = "divergence: NMF cd+nndsvd k=1 n_iter_ 5 vs sklearn 7 (violation-ratio FP-order near zero); tracking #2399"]
fn divergence_edge_k1_n_iter() {
    let x = x_8x5();
    let fitted = fit_cd(&x, 1, 200, 1e-4);
    // Context: the factorization matches sklearn to ~1e-6 — only n_iter_ drifts.
    assert_finite_nonneg("k1 H", fitted.components());
    assert_close("recon k1", fitted.reconstruction_err(), 8.418315267888007);
    let sk_comp = [
        1.1229694214551988,
        1.5717105090798411,
        1.8003013708645514,
        2.2037127235269947,
        1.7335357287714142,
    ];
    let h = fitted.components();
    for (j, &e) in sk_comp.iter().enumerate() {
        assert_close(&format!("comp k1 [0][{j}]"), h[[0, j]], e);
    }
    // The divergence: sklearn n_iter_ == 7, ferrolearn == 5.
    assert_eq!(fitted.n_iter(), 7, "n_iter_ k1 (sklearn 7, ferrolearn 5)");
}

/// k = n_features (6x4, k=4): n_iter_=120, recon, components_ row0 vs sklearn.
#[test]
fn divergence_edge_k_eq_nfeat() {
    let x: Array2<f64> = array![
        [4.464, 1.66, 4.106, 0.208],
        [0.538, 2.975, 2.649, 2.094],
        [1.677, 3.113, 2.191, 3.679],
        [2.59, 2.894, 3.227, 4.951],
        [4.099, 2.066, 4.381, 4.119],
        [0.272, 3.593, 4.011, 3.682],
    ];
    let fitted = fit_cd(&x, 4, 200, 1e-4);
    assert_finite_nonneg("k_eq_nfeat H", fitted.components());
    assert_eq!(fitted.n_iter(), 120, "n_iter_ k_eq_nfeat");
    assert_close("recon k_eq_nfeat", fitted.reconstruction_err(), 0.001043623244759193);
    let sk_comp0 = [0.12653082328725238, 1.0789382480526568, 1.3503991478817718, 2.6461854326059155];
    let h = fitted.components();
    for (j, &e) in sk_comp0.iter().enumerate() {
        assert_close(&format!("comp k_eq_nfeat [0][{j}]"), h[[0, j]], e);
    }
}

/// Zero-row input (no-panic + n_iter_/recon parity, R-CODE-2).
#[test]
fn divergence_edge_zero_row() {
    let x: Array2<f64> = array![
        [0.382, 3.9, 2.192, 3.617],
        [4.89, 2.692, 2.506, 0.36],
        [0.0, 0.0, 0.0, 0.0],
        [1.905, 0.33, 1.441, 4.548],
        [1.067, 2.261, 4.656, 0.124],
        [3.003, 4.751, 1.152, 2.742],
    ];
    let fitted = fit_cd(&x, 2, 200, 1e-4);
    assert_finite_nonneg("zero_row H", fitted.components());
    assert_eq!(fitted.n_iter(), 50, "n_iter_ zero_row");
    assert_close("recon zero_row", fitted.reconstruction_err(), 4.688901827868638);
    let sk_comp0 = [1.9048037407615392, 1.8464340319158339, 1.8971743171661242, 0.0];
    let h = fitted.components();
    for (j, &e) in sk_comp0.iter().enumerate() {
        assert_close(&format!("comp zero_row [0][{j}]"), h[[0, j]], e);
    }
}

/// Rank-deficient X (col 3 == col 0 -> a zero singular value); NNDSVD must not
/// panic/NaN and the CD must reach sklearn's n_iter_/recon.
#[test]
fn divergence_edge_rank_deficient() {
    let x: Array2<f64> = array![
        [4.367, 4.843, 4.346, 4.367],
        [1.164, 0.057, 2.152, 1.164],
        [2.613, 2.392, 2.777, 2.613],
        [3.804, 3.562, 3.098, 3.804],
        [1.445, 4.869, 1.669, 1.445],
        [0.329, 4.914, 0.639, 0.329],
    ];
    let fitted = fit_cd(&x, 3, 200, 1e-4);
    assert_finite_nonneg("rank_def H", fitted.components());
    assert_eq!(fitted.n_iter(), 120, "n_iter_ rank_def");
    assert_close("recon rank_def", fitted.reconstruction_err(), 0.000752377404029985);
    let sk_comp0 = [1.7157150970719655, 1.606666382320096, 1.3975170714934835, 1.7157150970719655];
    let h = fitted.components();
    for (j, &e) in sk_comp0.iter().enumerate() {
        assert_close(&format!("comp rank_def [0][{j}]"), h[[0, j]], e);
    }
}

/// max_iter=1 (n_iter_=1) and tol=1e-3 (n_iter_=48) on the same 6x4 fixture.
#[test]
fn divergence_edge_max_iter1_and_tol() {
    let x: Array2<f64> = array![
        [0.052, 2.509, 2.479, 0.669],
        [0.711, 1.093, 2.093, 1.241],
        [0.42, 1.727, 0.834, 4.393],
        [4.755, 0.194, 3.496, 2.864],
        [4.49, 3.334, 2.739, 3.512],
        [1.932, 3.472, 4.124, 2.328],
    ];
    let f1 = fit_cd(&x, 2, 1, 1e-4);
    assert_eq!(f1.n_iter(), 1, "n_iter_ max_iter1");
    assert_close("recon max_iter1", f1.reconstruction_err(), 4.598985199724485);
    let sk_c0_mi1 = [1.322311062457787, 1.5818174457309582, 1.9781893091479412, 1.9118066011108688];
    let h1 = f1.components();
    for (j, &e) in sk_c0_mi1.iter().enumerate() {
        assert_close(&format!("comp max_iter1 [0][{j}]"), h1[[0, j]], e);
    }
    let f2 = fit_cd(&x, 2, 200, 1e-3);
    assert_eq!(f2.n_iter(), 48, "n_iter_ tol1e3");
    assert_close("recon tol1e3", f2.reconstruction_err(), 3.7268772029842276);
    let sk_c0_t = [0.0, 2.151375968331528, 1.5556591488971483, 1.3735811534648343];
    let h2 = f2.components();
    for (j, &e) in sk_c0_t.iter().enumerate() {
        assert_close(&format!("comp tol1e3 [0][{j}]"), h2[[0, j]], e);
    }
}
