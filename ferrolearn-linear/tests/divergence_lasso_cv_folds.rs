//! Regression guard for `LassoCV` cross-validation fold strategy (#421).
//!
//! ferrolearn's `kfold_indices` (`ferrolearn-linear/src/lasso_cv.rs`) formerly
//! assigned sample `i` to fold `i % k` (round-robin / interleaved). scikit-learn's
//! `LassoCV` uses `check_cv(5) -> KFold(5)` non-shuffled, which produces
//! **contiguous** blocks (`sklearn/model_selection/_split.py:521-534`
//! `KFold._iter_test_indices`; routed via
//! `sklearn/linear_model/_coordinate_descent.py:1729` `check_cv(self.cv)`).
//! The two partitions induced different per-alpha CV MSE, hence a different
//! selected `alpha_`. `kfold_indices` now mirrors the contiguous partition,
//! so the selected `alpha_` matches sklearn exactly. The residual
//! `coef_`/`intercept_` mismatch (~4e-5) is the coordinate-descent stopping
//! criterion #412 (dual-gap vs max-coef-change in `lasso.rs`), NOT the fold
//! strategy, and is tracked separately.
//!
//! Oracle: scikit-learn 1.5.2 (commit 156ef14), computed live (see header of
//! each test). Expected values are NEVER copied from the ferrolearn side
//! (R-CHAR-3).

use ferrolearn_core::introspection::HasCoefficients;
use ferrolearn_core::traits::Fit;
use ferrolearn_linear::LassoCV;
use ndarray::{Array1, Array2};

/// The seed-1 RandomState(1) dataset (n=12, p=3, beta=[3,0,-2], 0.5*noise)
/// for which sklearn's contiguous KFold and ferrolearn's round-robin folds
/// select DIFFERENT alphas. Captured verbatim from the live oracle:
///
/// ```text
/// rng=np.random.RandomState(1); X=rng.randn(12,3)
/// y = X @ [3,0,-2] + 0.5*rng.randn(12)
/// ```
#[rustfmt::skip]
fn seed1_data() -> (Array2<f64>, Array1<f64>) {
    let x = Array2::from_shape_vec((12, 3), vec![
         1.6243453636632417, -0.6117564136500754, -0.5281717522634557,
        -1.0729686221561705,  0.8654076293246785, -2.3015386968802827,
         1.74481176421648,   -0.7612069008951028,  0.31903909605709857,
        -0.2493703754774101,  1.462107937044974,  -2.060140709497654,
        -0.3224172040135075, -0.38405435466841564, 1.1337694423354374,
        -1.0998912673140309, -0.17242820755043575,-0.8778584179213718,
         0.04221374671559283, 0.5828152137158222, -1.1006191772129212,
         1.1447237098396141,  0.9015907205927955,  0.5024943389018682,
         0.9008559492644118, -0.6837278591743331, -0.12289022551864817,
        -0.9357694342590688, -0.2678880796260159,  0.530355466738186,
        -0.691660751725309,  -0.39675352685597737,-0.6871727001195994,
        -0.8452056414987196, -0.671246130836819,  -0.01266459891890136,
    ]).unwrap();
    let y = Array1::from(vec![
        5.370724421198997, 1.5013793762006, 5.426258189090179, 3.7431923728517456,
        -3.330708272892205, -1.9877714481417672, 1.9543004476972023, 3.2754097522289793,
        2.973752176218546, -4.1865170595382555, -0.6051791126029951, -1.460160158418935,
    ]);
    (x, y)
}

/// Fold-strategy parity (#421): ferrolearn `LassoCV(n_alphas=10, cv=3)` must
/// select the SAME `alpha_` as `sklearn.linear_model.LassoCV` on the seed-1
/// dataset. The divergence was that `kfold_indices` (`lasso_cv.rs`) used
/// round-robin `i % k` folds; sklearn's `LassoCV` routes through
/// `check_cv(self.cv)` (`_coordinate_descent.py:1729`) to a non-shuffled
/// `KFold`, whose `_iter_test_indices` (`sklearn/model_selection/_split.py:521-534`)
/// yields **contiguous** index blocks (sizes `n//k`, first `n%k` folds +1).
///
/// With the contiguous-fold fix the per-alpha CV MSE ordering matches sklearn,
/// so the selected `alpha_` matches to floating-point round-off (~2e-17). This
/// test pins that exact `alpha_` parity as the regression guard.
///
/// Live oracle (sklearn 1.5.2):
/// ```text
/// m = LassoCV(n_alphas=10, cv=3).fit(X, y)
/// m.alpha_     == 0.02561299415267483
/// m.coef_      == [3.03624696385305, 0.1618877055852443, -1.826054712871375]
/// m.intercept_ == 0.20532096730384264
/// ```
/// Before the fix (round-robin folds) ferrolearn selected
/// `alpha_ == 0.055181523118106465`, a different grid point.
///
/// COEF/INTERCEPT bound is #412, NOT the fold strategy. The refit `coef_` and
/// `intercept_` match sklearn only to ~1e-4, not to ULP: sklearn's oracle
/// `coef_` is taken at its default `tol=1e-4` (an *under-converged* Lasso
/// solution), while ferrolearn's coordinate-descent stopping criterion is
/// max-coef-change rather than sklearn's dual-gap (#412, `lasso.rs`). Asserting
/// exact coef parity here is unachievable without #412 — tightening
/// ferrolearn's `tol` moves the solution AWAY from sklearn's under-converged
/// oracle. So coef/intercept are guarded at `1e-4` (observed worst residual
/// ~4e-5); #412 owns closing that residual, and the fold fix does not affect it.
///
/// Tracking: #421 (fold strategy, fixed here); #412 (CD dual-gap stopping
/// criterion bounding the refit `coef_`/`intercept_` parity).
#[test]
fn divergence_lasso_cv_fold_strategy_selects_alpha() {
    let (x, y) = seed1_data();

    let fitted = LassoCV::<f64>::new()
        .with_n_alphas(10)
        .with_cv(3)
        .fit(&x, &y)
        .expect("fit should succeed");

    // sklearn KFold(3) (contiguous) selects this alpha; with the fold fix
    // ferrolearn selects the same grid point (round-robin chose 0.0551815...).
    const SK_ALPHA: f64 = 0.02561299415267483;
    const SK_COEF: [f64; 3] = [3.03624696385305, 0.1618877055852443, -1.826054712871375];
    const SK_INTERCEPT: f64 = 0.20532096730384264;

    // Fold strategy → alpha_ selection parity: exact (the contiguous-fold fix
    // makes the selected grid point match sklearn to floating-point round-off).
    assert!(
        (fitted.best_alpha() - SK_ALPHA).abs() < 1e-9,
        "LassoCV alpha_: ferrolearn={} sklearn={}",
        fitted.best_alpha(),
        SK_ALPHA,
    );

    // coef_/intercept_ parity is bounded by #412 (CD dual-gap stopping
    // criterion), NOT the fold strategy. Guard at 1e-4 (observed residual
    // ~4e-5); exact parity is #412's job, not this divergence's.
    let coef = fitted.coefficients();
    for k in 0..3 {
        assert!(
            (coef[k] - SK_COEF[k]).abs() < 1e-4,
            "LassoCV coef_[{k}]: ferrolearn={} sklearn={} (bound is #412, not folds)",
            coef[k],
            SK_COEF[k],
        );
    }

    assert!(
        (fitted.intercept() - SK_INTERCEPT).abs() < 1e-4,
        "LassoCV intercept_: ferrolearn={} sklearn={} (bound is #412, not folds)",
        fitted.intercept(),
        SK_INTERCEPT,
    );
}

/// Isolation check (NOT a separate divergence): confirm the auto alpha GRID
/// ferrolearn evaluates equals sklearn's `_alpha_grid` output (`alphas_`) for
/// the seed-1 dataset. If this passes, the alpha grid (alpha_max centering,
/// eps=1e-3, n_alphas, geomspace) is correct and the ONLY cause of the
/// `alpha_`/`coef_` divergence above is the fold partition.
///
/// Live oracle (sklearn 1.5.2):
/// ```text
/// m = LassoCV(n_alphas=10, cv=3).fit(X, y); m.alphas_ ==
/// [2.5612994152674826, 1.1888498765444488, 0.5518152311810645,
///  0.2561299415267483, 0.11888498765444491, 0.055181523118106465,
///  0.02561299415267483, 0.011888498765444497, 0.005518152311810647,
///  0.0025612994152674827]
/// ```
///
/// This test is expected to PASS (the grid matches); it is the isolation
/// evidence accompanying #421, not itself a pinned divergence.
#[test]
fn isolation_lasso_cv_alpha_grid_matches_sklearn() {
    let (x, y) = seed1_data();

    let fitted = LassoCV::<f64>::new()
        .with_n_alphas(10)
        .with_cv(3)
        .fit(&x, &y)
        .expect("fit should succeed");

    // sklearn _alpha_grid(X, y, n_alphas=10, eps=1e-3, fit_intercept=True).
    const SK_ALPHAS: [f64; 10] = [
        2.5612994152674826,
        1.1888498765444488,
        0.5518152311810645,
        0.2561299415267483,
        0.11888498765444491,
        0.055181523118106465,
        0.02561299415267483,
        0.011888498765444497,
        0.005518152311810647,
        0.0025612994152674827,
    ];

    let grid = fitted.alphas();
    assert_eq!(grid.len(), 10, "grid length");
    for (i, &expected) in SK_ALPHAS.iter().enumerate() {
        let rel = (grid[i] - expected).abs() / expected;
        assert!(
            rel < 1e-9,
            "alpha grid[{i}]: ferrolearn={} sklearn={} (rel {rel})",
            grid[i],
            expected,
        );
    }
}
