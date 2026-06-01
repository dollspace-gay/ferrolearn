//! Divergence guards for `ElasticNetCV` against scikit-learn 1.5.2.
//!
//! ferrolearn's [`ElasticNetCV`] (`ferrolearn-linear/src/elastic_net_cv.rs`)
//! diverges from `sklearn.linear_model.ElasticNetCV`
//! (`sklearn/linear_model/_coordinate_descent.py:2131`) in three observable
//! ways pinned here:
//!
//! 1. **Default `l1_ratio` (#432).** sklearn's `__init__`
//!    (`_coordinate_descent.py:2328`) defaults `l1_ratio=0.5` — a SINGLE value.
//!    ferrolearn's `new()` defaults to a 7-element grid
//!    `[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]`. With the wider search ferrolearn
//!    can select a non-0.5 `l1_ratio_` and a different `alpha_`.
//!
//! 2. **Fold strategy (#431).** sklearn routes `cv` through `check_cv`
//!    (`_coordinate_descent.py:1729`) to a non-shuffled `KFold`, whose
//!    `_iter_test_indices` (`sklearn/model_selection/_split.py:521-534`)
//!    produces **contiguous** index blocks. ferrolearn's `kfold_indices`
//!    (`elastic_net_cv.rs` line 178) assigns sample `i` to fold `i % k`
//!    (round-robin). The two partitions induce different per-alpha CV MSE and
//!    therefore a different selected `alpha_`.
//!
//! 3. **`l1_ratio=0` validation (#440).** sklearn's `_alpha_grid`
//!    (`_coordinate_descent.py:140`) raises `ValueError` for `l1_ratio=0`
//!    ("Automatic alpha grid generation is not supported for l1_ratio=0").
//!    ferrolearn silently computes a fallback grid
//!    (`compute_alpha_max_enet` else-branch, `elastic_net_cv.rs`) and fits.
//!
//! Oracle: scikit-learn 1.5.2 (commit 156ef14), computed live (values quoted
//! in each test). Expected values are NEVER copied from the ferrolearn side
//! (R-CHAR-3).
//!
//! `coef_`/`intercept_` parity is bounded by the coordinate-descent stopping
//! criterion (#412, `enet.rs`/`lasso.rs`), NOT by these divergences, so coef is
//! asserted only at ~1e-4 with a #412 note (mirrors `divergence_lasso_cv_folds`).

use ferrolearn_core::introspection::HasCoefficients;
use ferrolearn_core::traits::Fit;
use ferrolearn_linear::ElasticNetCV;
use ndarray::{Array1, Array2};

/// The seed-1 `RandomState(1)` dataset (n=12, p=3, beta=[3,0,-2], 0.5*noise),
/// captured verbatim from the live oracle:
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

/// Divergence #432 — default `l1_ratio`. `ElasticNetCV()` with all defaults:
/// sklearn fixes `l1_ratio=0.5` (single value, `_coordinate_descent.py:2328`)
/// and the default `cv=None` resolves to 5 folds. ferrolearn's `new()` searches
/// a 7-element `l1_ratios` grid (`elastic_net_cv.rs` `new()` lines 74-82) and
/// uses `cv=5`; the wider grid lets it pick a different `(alpha_, l1_ratio_)`.
///
/// Live oracle (sklearn 1.5.2), full defaults:
/// ```text
/// m = ElasticNetCV().fit(X, y)
/// m.l1_ratio_  == 0.5
/// m.alpha_     == 0.012689068444348958
/// m.coef_      == [3.0392792963711335, 0.20550076850906293, -1.8157039482607618]
/// m.intercept_ == 0.2102488166554194
/// ```
///
/// Tracking: #432 (default `l1_ratios=[0.5]`). Also gated by #431 (folds) and
/// #412 (CD stopping); this test pins the default-`l1_ratio` selection divergence.
#[test]
fn divergence_elastic_net_cv_default_l1_ratio_selection() {
    let (x, y) = seed1_data();

    let fitted = ElasticNetCV::<f64>::new()
        .fit(&x, &y)
        .expect("fit should succeed");

    // sklearn ElasticNetCV() default: l1_ratio_ == 0.5 (single-value default).
    const SK_L1_RATIO: f64 = 0.5;
    const SK_ALPHA: f64 = 0.012689068444348958;

    assert!(
        (fitted.best_l1_ratio() - SK_L1_RATIO).abs() < 1e-12,
        "ElasticNetCV() l1_ratio_: ferrolearn={} sklearn={} (default must be 0.5)",
        fitted.best_l1_ratio(),
        SK_L1_RATIO,
    );

    assert!(
        (fitted.best_alpha() - SK_ALPHA).abs() < 1e-9,
        "ElasticNetCV() alpha_: ferrolearn={} sklearn={}",
        fitted.best_alpha(),
        SK_ALPHA,
    );
}

/// Divergence #431 — fold strategy, ISOLATED from #432. Both sides use a single
/// `l1_ratio=0.5` (ferrolearn via `with_l1_ratios([0.5])`, sklearn via
/// `ElasticNetCV(l1_ratio=0.5)`), so the ONLY difference is the CV fold
/// partition: ferrolearn `kfold_indices` round-robin `i % k`
/// (`elastic_net_cv.rs` line 178) vs sklearn's contiguous non-shuffled `KFold`
/// (`sklearn/model_selection/_split.py:521-534`, routed via
/// `_coordinate_descent.py:1729`). The different partition selects a different
/// grid point.
///
/// Live oracle (sklearn 1.5.2):
/// ```text
/// m = ElasticNetCV(l1_ratio=0.5, n_alphas=10, cv=3).fit(X, y)
/// m.l1_ratio_  == 0.5
/// m.alpha_     == 0.023776997530888997   # alphas_[7]
/// m.coef_      == [3.01041949573582, 0.2048226153688424, -1.791458411079841]
/// m.intercept_ == 0.22133476798379514
/// m.alphas_    == [5.122598830534965, 2.3776997530888977, 1.103630462362129,
///                  0.5122598830534966, 0.2377699753088898, 0.11036304623621292,
///                  0.05122598830534967, 0.023776997530888997,
///                  0.011036304623621295, 0.005122598830534965]
/// ```
///
/// COEF/INTERCEPT bound is #412 (CD stopping), NOT folds — guarded at 1e-4.
///
/// Tracking: #431 (contiguous KFold, same fix as LassoCV #421); #412 (coef).
#[test]
fn divergence_elastic_net_cv_fold_strategy_selects_alpha() {
    let (x, y) = seed1_data();

    let fitted = ElasticNetCV::<f64>::new()
        .with_l1_ratios(vec![0.5])
        .with_n_alphas(10)
        .with_cv(3)
        .fit(&x, &y)
        .expect("fit should succeed");

    const SK_L1_RATIO: f64 = 0.5;
    const SK_ALPHA: f64 = 0.023776997530888997;
    const SK_COEF: [f64; 3] = [3.01041949573582, 0.2048226153688424, -1.791458411079841];
    const SK_INTERCEPT: f64 = 0.22133476798379514;

    // l1_ratio is matched on both sides, so it must agree exactly.
    assert!(
        (fitted.best_l1_ratio() - SK_L1_RATIO).abs() < 1e-12,
        "l1_ratio_: ferrolearn={} sklearn={}",
        fitted.best_l1_ratio(),
        SK_L1_RATIO,
    );

    // Fold strategy -> alpha_ selection parity. With contiguous folds ferrolearn
    // should select sklearn's grid point; round-robin selects a different one.
    assert!(
        (fitted.best_alpha() - SK_ALPHA).abs() < 1e-9,
        "ElasticNetCV alpha_: ferrolearn={} sklearn={} (fold strategy)",
        fitted.best_alpha(),
        SK_ALPHA,
    );

    // coef_/intercept_ parity bounded by #412, guarded at 1e-4.
    let coef = fitted.coefficients();
    for k in 0..3 {
        assert!(
            (coef[k] - SK_COEF[k]).abs() < 1e-4,
            "coef_[{k}]: ferrolearn={} sklearn={} (bound is #412, not folds)",
            coef[k],
            SK_COEF[k],
        );
    }
    assert!(
        (fitted.intercept() - SK_INTERCEPT).abs() < 1e-4,
        "intercept_: ferrolearn={} sklearn={} (bound is #412, not folds)",
        fitted.intercept(),
        SK_INTERCEPT,
    );
}

/// Divergence #440 — `l1_ratio=0` validation. sklearn's `_alpha_grid`
/// (`_coordinate_descent.py:140-146`) raises `ValueError` when `l1_ratio=0`
/// with no explicit `alphas` ("Automatic alpha grid generation is not supported
/// for l1_ratio=0. Please supply a grid ..."). ferrolearn's
/// `compute_alpha_max_enet` (`elastic_net_cv.rs` else-branch ~line 244) instead
/// silently returns `max|Xᵀy| / n` and fits, never raising.
///
/// Live oracle (sklearn 1.5.2):
/// ```text
/// _alpha_grid(X, y, l1_ratio=0.0, n_alphas=10) -> ValueError
/// ```
///
/// This pins the observable contract divergence: ferrolearn must ERROR for
/// `l1_ratios=[0.0]` (auto grid) where sklearn raises. Currently `fit` returns
/// `Ok`, so this assertion fails.
///
/// Tracking: #440 (l1_ratio=0 validation; raise on auto-grid l1_ratio=0).
#[test]
fn divergence_elastic_net_cv_l1_ratio_zero_must_error() {
    let (x, y) = seed1_data();

    let result = ElasticNetCV::<f64>::new()
        .with_l1_ratios(vec![0.0])
        .with_n_alphas(10)
        .with_cv(3)
        .fit(&x, &y);

    // sklearn raises ValueError for automatic alpha grid with l1_ratio=0.
    assert!(
        result.is_err(),
        "ElasticNetCV(l1_ratios=[0.0]) auto-grid: ferrolearn returned Ok, \
         sklearn raises ValueError (_coordinate_descent.py:140)",
    );
}
