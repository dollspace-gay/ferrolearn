//! Divergence pins for `TruncatedSVD` / `FittedTruncatedSVD` against
//! scikit-learn 1.5.2 `class TruncatedSVD`
//! (`/home/doll/scikit-learn/sklearn/decomposition/_truncated_svd.py:30`).
//!
//! Tracking: #1553 (doc) — per-divergence blockers cited on each `#[ignore]`.
//!
//! All expected values come from the live sklearn 1.5.2 oracle (run from /tmp)
//! or are computed from ferrolearn's OWN fitted components fed through the
//! sklearn FORMULA (centered variance) — never literal-copied from the
//! ferrolearn side (R-CHAR-3).
//!
//! Three FIXABLE value divergences are pinned below as `#[ignore]`'d failing
//! tests, plus a block of GREEN-GUARD structural tests that must PASS against
//! the current code. The no-power-iteration accuracy question (REQ-4) is a
//! CARVE-OUT (see `div4_*` note) — ferrolearn's randomized-without-power-iter
//! singular values/components match the TRUE SVD to ~8 sig figs even on a
//! slowly-decaying spectrum, so it is NOT pinned.

use approx::assert_abs_diff_eq;
use ferrolearn_core::traits::{Fit, Transform};
use ferrolearn_decomp::TruncatedSVD;
use ndarray::{Array1, Array2, array};

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// Fixture A: 5x3, large column means (~100/200/300). Doc Probe 1. Used for
/// the explained_variance_ / ratio centering divergences — the transformed
/// columns have a huge nonzero mean, so the centered vs uncentered formulas
/// differ by ~5 orders of magnitude.
fn fixture_a() -> Array2<f64> {
    array![
        [100., 200., 300.],
        [101., 202., 298.],
        [99., 201., 303.],
        [102., 199., 301.],
        [98., 203., 299.],
    ]
}

/// Fixture B: 8x6, slowly-decaying spectrum (true singular values
/// 345.67, 9.04, 7.19) with a large mean offset (~50). On seed 42 the raw
/// Jacobi eigenvector signs disagree with sklearn's `svd_flip` on ALL THREE
/// component rows (each ferrolearn row's max-abs entry is NEGATIVE), so the
/// sign divergence is observable on every row. Also used to confirm the
/// no-power-iteration accuracy carve-out (REQ-4).
fn fixture_b() -> Array2<f64> {
    array![
        [
            52.325753, 48.807097, 49.960143, 50.82559, 48.706457, 50.145166
        ],
        [
            50.389219, 46.949279, 52.244265, 51.211077, 49.132879, 49.880222
        ],
        [
            50.783175, 49.156477, 49.224875, 47.294565, 51.258186, 50.499185
        ],
        [
            50.83212, 46.794883, 53.403384, 50.317271, 49.788047, 53.858485
        ],
        [
            50.380122, 47.189255, 48.977126, 45.594834, 52.267655, 49.767635
        ],
        [
            48.884506, 52.570946, 46.551405, 51.443129, 46.59302, 48.40303
        ],
        [
            47.83578, 52.526051, 53.235021, 49.986119, 51.603006, 49.545917
        ],
        [
            51.350758, 48.192038, 46.530565, 45.990544, 50.960224, 54.276556
        ],
    ]
}

/// Centered column variance (population, ddof=0) of `col` — the numpy
/// `np.var(.., axis=0)` formula sklearn applies (`_truncated_svd.py:269,274`).
/// Implemented as `mean(x^2) - mean(x)^2`. This is the sklearn FORMULA applied
/// to ferrolearn's own data (R-CHAR-3), not a copied ferrolearn value.
fn centered_var(col: &Array1<f64>) -> f64 {
    let n = col.len() as f64;
    let mean = col.sum() / n;
    let mean_sq = col.iter().map(|&v| v * v).sum::<f64>() / n;
    mean_sq - mean * mean
}

// ---------------------------------------------------------------------------
// DIV-1: components_ sign via svd_flip(u_based_decision=False)
// sklearn `_truncated_svd.py:255` + `extmath.py:897-905`:
//   max_abs_v_rows = argmax(abs(VT), axis=1); signs = sign(VT[row, max_abs]);
//   VT *= signs[:, newaxis]  =>  each component row's max-abs entry POSITIVE.
// ferrolearn `truncated_svd.rs:584-588` copies raw Jacobi eigenvector rows with
// NO sign convention. On fixture B + seed 42 all three rows' max-abs entries are
// NEGATIVE. sklearn arpack oracle (Probe-2 style): comp max-abs entries are
// +0.415685 (idx5), +0.562754 (idx3), +0.740641 (idx2) — all POSITIVE.
// Tracking: #1556
// ---------------------------------------------------------------------------

#[test]
fn divergence_components_max_abs_positive_svd_flip() {
    let x = fixture_b();
    let fitted = TruncatedSVD::<f64>::new(3)
        .with_random_state(42)
        .fit(&x, &())
        .expect("fit fixture_b");
    let comp = fitted.components();
    // sklearn invariant: the max-abs entry of EACH component row is POSITIVE.
    for i in 0..comp.nrows() {
        let row = comp.row(i);
        let (max_idx, _) = row
            .iter()
            .enumerate()
            .fold((0usize, 0.0f64), |acc, (j, &v)| {
                if v.abs() > acc.1 { (j, v.abs()) } else { acc }
            });
        let max_abs_entry = row[max_idx];
        assert!(
            max_abs_entry > 0.0,
            "component row {i} max-abs entry must be POSITIVE per svd_flip \
             (_truncated_svd.py:255); got {max_abs_entry} at idx {max_idx}"
        );
    }
}

// ---------------------------------------------------------------------------
// DIV-2: explained_variance_ CENTERING.
// sklearn: explained_variance_ = np.var(X_transformed, axis=0)
// (`_truncated_svd.py:269`), X_transformed = X @ components_.T (`:264`).
// np.var SUBTRACTS the transformed column mean (ddof=0).
// ferrolearn: explained_variance = singular_values^2 / n_samples
// (`truncated_svd.rs:595-596`) — uncentered 2nd moment, NO mean subtraction.
// R-CHAR-3 oracle: np.var(X @ ferro_components.T, axis=0), computed in-Rust from
// ferrolearn's OWN fitted components (sign-invariant). On fixture A this is
// ~1.099 / ~3.839 vs ferrolearn's sigma^2/n ~140522 / ~3.839.
// Tracking: #1554
// ---------------------------------------------------------------------------

#[test]
fn divergence_explained_variance_centered() {
    let x = fixture_a();
    let fitted = TruncatedSVD::<f64>::new(2)
        .with_random_state(42)
        .fit(&x, &())
        .expect("fit fixture_a");
    // Oracle: centered variance of ferrolearn's OWN transformed columns.
    let transformed = fitted.transform(&x).expect("transform");
    let ev = fitted.explained_variance();
    for i in 0..ev.len() {
        let col = transformed.column(i).to_owned();
        let expected = centered_var(&col); // sklearn np.var formula, R-CHAR-3
        assert_abs_diff_eq!(ev[i], expected, epsilon = 1e-6);
    }
}

// ---------------------------------------------------------------------------
// DIV-3: explained_variance_ratio_ DENOMINATOR.
// sklearn: full_var = np.var(X, axis=0).sum() (`_truncated_svd.py:274`),
//          explained_variance_ratio_ = exp_var / full_var (`:275`).
// ferrolearn: total_var = Sum(x^2)/n_samples (`truncated_svd.rs:599-605`),
//             ratio = explained_variance / total_var (`:607-611`).
// R-CHAR-3 oracle: ferrolearn's OWN explained_variance() divided by the CENTERED
// per-feature variance sum np.var(X, axis=0).sum(). On fixture A the sklearn
// denominator is ~6.96 vs ferrolearn's uncentered ~140528.
// Tracking: #1555
// ---------------------------------------------------------------------------

#[test]
fn divergence_explained_variance_ratio_centered_denominator() {
    let x = fixture_a();
    let fitted = TruncatedSVD::<f64>::new(2)
        .with_random_state(42)
        .fit(&x, &())
        .expect("fit fixture_a");
    // Centered per-feature variance sum (sklearn full_var formula, R-CHAR-3).
    let full_var: f64 = (0..x.ncols())
        .map(|j| centered_var(&x.column(j).to_owned()))
        .sum();
    let ev = fitted.explained_variance();
    let ratio = fitted.explained_variance_ratio();
    for i in 0..ratio.len() {
        // ratio[i] should equal ferrolearn's OWN explained_variance / centered full_var.
        let expected = ev[i] / full_var;
        assert_abs_diff_eq!(ratio[i], expected, epsilon = 1e-9);
    }
}

// ---------------------------------------------------------------------------
// DIV-4 (CARVE-OUT, NOT pinned): no power iterations.
// ferrolearn does ZERO power iterations (`truncated_svd.rs:564`, single
// Y = X @ Omega) where sklearn passes n_iter=5. The decision per the brief: if
// ferrolearn's top-k singular values match the TRUE SVD within a reasonable tol
// even on a slowly-decaying spectrum, the missing power iterations are an
// accuracy-margin note and NOT a divergence. This test asserts that accuracy
// holds (it PASSES) — documenting the carve-out. The oracle is numpy's exact
// top-k singular values (deterministic, == arpack to 8 figs, doc Probe 2).
// ---------------------------------------------------------------------------

/// Exact top-3 singular values of fixture B from `np.linalg.svd(X)` (and
/// `TruncatedSVD(algorithm='arpack')`), live sklearn/numpy oracle (R-CHAR-3).
#[allow(
    clippy::excessive_precision,
    reason = "live sklearn/numpy 1.5.2 oracle (R-CHAR-3)"
)]
const SK_TRUE_SV_B: [f64; 3] = [345.6655195709838, 9.035375462312068, 7.190742878457003];

#[test]
fn div4_carveout_no_power_iter_singular_values_match_true_svd() {
    // GREEN: ferrolearn's randomized-without-power-iteration singular values
    // match the TRUE SVD even on a slowly-decaying spectrum => carve-out.
    let x = fixture_b();
    let fitted = TruncatedSVD::<f64>::new(3)
        .with_random_state(42)
        .fit(&x, &())
        .expect("fit fixture_b");
    let sv = fitted.singular_values();
    for i in 0..3 {
        assert_abs_diff_eq!(sv[i], SK_TRUE_SV_B[i], epsilon = 1e-5);
    }
}

// ---------------------------------------------------------------------------
// GREEN GUARDS (structural SHIPPED contract — must PASS against current code).
// ---------------------------------------------------------------------------

#[test]
fn green_components_shape_and_singular_values_descending_nonneg() {
    let x = fixture_b();
    let fitted = TruncatedSVD::<f64>::new(3)
        .with_random_state(42)
        .fit(&x, &())
        .expect("fit");
    assert_eq!(fitted.components().dim(), (3, 6));
    let sv = fitted.singular_values();
    assert_eq!(sv.len(), 3);
    for &s in sv {
        assert!(s >= 0.0, "singular value must be non-negative, got {s}");
    }
    for i in 1..sv.len() {
        assert!(
            sv[i - 1] >= sv[i] - 1e-10,
            "singular values not descending: {} < {}",
            sv[i - 1],
            sv[i]
        );
    }
}

#[test]
fn green_components_rows_unit_norm() {
    let x = fixture_b();
    let fitted = TruncatedSVD::<f64>::new(3)
        .with_random_state(42)
        .fit(&x, &())
        .expect("fit");
    let c = fitted.components();
    for i in 0..c.nrows() {
        let norm: f64 = c.row(i).iter().map(|v| v * v).sum::<f64>().sqrt();
        assert_abs_diff_eq!(norm, 1.0, epsilon = 1e-6);
    }
}

#[test]
fn green_transform_equals_x_dot_components_t_no_centering() {
    let x = fixture_a();
    let fitted = TruncatedSVD::<f64>::new(2)
        .with_random_state(42)
        .fit(&x, &())
        .expect("fit");
    let t = fitted.transform(&x).expect("transform");
    assert_eq!(t.dim(), (5, 2));
    // transform == X @ components.T (algebra, REQ-10).
    let manual = x.dot(&fitted.components().t());
    for (a, b) in t.iter().zip(manual.iter()) {
        assert_abs_diff_eq!(a, b, epsilon = 1e-9);
    }
    // NO centering: large-mean input => large-magnitude transform output.
    let max_abs = t.iter().fold(0.0f64, |m, &v| m.max(v.abs()));
    assert!(
        max_abs > 100.0,
        "uncentered transform of large-mean X should be large; got {max_abs}"
    );
}

#[test]
fn green_inverse_transform_shape_and_algebra() {
    let x = fixture_a();
    let fitted = TruncatedSVD::<f64>::new(2)
        .with_random_state(42)
        .fit(&x, &())
        .expect("fit");
    let t = fitted.transform(&x).expect("transform");
    let inv = fitted.inverse_transform(&t).expect("inverse_transform");
    assert_eq!(inv.dim(), (5, 3));
    // inverse_transform == X_reduced @ components (algebra, REQ-10).
    let manual = t.dot(fitted.components());
    for (a, b) in inv.iter().zip(manual.iter()) {
        assert_abs_diff_eq!(a, b, epsilon = 1e-9);
    }
}

#[test]
fn green_explained_variance_ratio_sums_le_1() {
    // Structural bound only (REQ-9): the ratio VALUES diverge (DIV-2/3), but
    // sum <= 1 still holds because retained energy <= total energy.
    let x = fixture_b();
    let fitted = TruncatedSVD::<f64>::new(3)
        .with_random_state(42)
        .fit(&x, &())
        .expect("fit");
    let sum: f64 = fitted.explained_variance_ratio().iter().sum();
    assert!(sum <= 1.0 + 1e-6, "ratio sum exceeds 1: {sum}");
}

#[test]
fn green_error_contracts() {
    // n_components == 0 => Err
    let x = fixture_a();
    assert!(TruncatedSVD::<f64>::new(0).fit(&x, &()).is_err());
    // n_components > min(n, p) => Err
    assert!(TruncatedSVD::<f64>::new(10).fit(&x, &()).is_err());
    // n_samples == 0 => Err
    let empty: Array2<f64> = Array2::zeros((0, 3));
    assert!(TruncatedSVD::<f64>::new(1).fit(&empty, &()).is_err());
    // transform column mismatch => Err
    let fitted = TruncatedSVD::<f64>::new(2)
        .with_random_state(42)
        .fit(&x, &())
        .expect("fit");
    let bad = array![[1.0, 2.0]];
    assert!(fitted.transform(&bad).is_err());
    // inverse_transform column mismatch => Err
    let bad_reduced = array![[1.0, 2.0, 3.0]];
    assert!(fitted.inverse_transform(&bad_reduced).is_err());
}

#[test]
fn green_determinism_same_seed() {
    let x = fixture_b();
    let f1 = TruncatedSVD::<f64>::new(3)
        .with_random_state(42)
        .fit(&x, &())
        .expect("fit");
    let f2 = TruncatedSVD::<f64>::new(3)
        .with_random_state(42)
        .fit(&x, &())
        .expect("fit");
    for (a, b) in f1.singular_values().iter().zip(f2.singular_values().iter()) {
        assert_abs_diff_eq!(a, b, epsilon = 1e-12);
    }
    for (a, b) in f1.components().iter().zip(f2.components().iter()) {
        assert_abs_diff_eq!(a, b, epsilon = 1e-12);
    }
}

#[test]
fn green_f32_fits_without_panic() {
    let x: Array2<f32> = array![[1.0f32, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
    let fitted = TruncatedSVD::<f32>::new(1)
        .with_random_state(42)
        .fit(&x, &())
        .expect("f32 fit");
    let t = fitted.transform(&x).expect("transform");
    assert_eq!(t.ncols(), 1);
}
