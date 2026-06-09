//! Divergence audit (#2341): `ferrolearn-preprocess` `TargetEncoder` vs
//! scikit-learn 1.5.2 on the THREE unaudited subtle bug sites flagged as
//! NOT-STARTED in the src REQ table but observable through the shipped public
//! API: the DEFAULT `smooth` value, the `smooth='auto'` empirical-Bayes
//! encodings, and the cross-fit `fit_transform`.
//!
//! Every expected value below is a LIVE sklearn 1.5.2 oracle result (run from
//! /tmp), NEVER copied from the ferrolearn side (R-CHAR-3). These tests pinned
//! NOT-STARTED divergences; they are now GREEN after the `Smooth::Auto`
//! empirical-Bayes encoding, the `smooth="auto"` default, and the cross-fit
//! `fit_transform` shipped (#2342 #2343 #2344). NOTE (R-HONEST-4): the C cross-fit
//! pin originally captured sklearn's DEFAULT non-reproducible `shuffle=True`
//! output; it is now pinned against the REPRODUCIBLE deterministic `shuffle=False`
//! KFold oracle (see `divergence_crossfit_fit_transform`).
//!
//! Oracle session (sklearn 1.5.2):
//! ```text
//! # DEFAULT smooth — sklearn TargetEncoder().smooth == 'auto', NOT 1.0:
//! >>> from sklearn.preprocessing import TargetEncoder
//! >>> TargetEncoder().smooth   -> 'auto'
//! >>> TargetEncoder().cv       -> 5
//!
//! # A — DEFAULT encoder (smooth='auto'), X=[[0],[0],[1],[1],[2]], y=[1,2,3,4,10]:
//! >>> e = TargetEncoder(target_type='continuous').fit(X, y)
//! >>> [float(v) for v in e.encodings_[0]]
//!       -> [1.5308641975308643, 3.506172839506173, 10.0]
//! >>> float(e.target_mean_)  -> 4.0
//!   ferrolearn default smooth=1.0 -> fixed-smooth m-estimate, NOT this.
//!
//! # B — explicit smooth='auto', X=[[0],[0],[0],[1],[1],[2]], y=[1,3,5,2,8,4]:
//! >>> e = TargetEncoder(smooth='auto', target_type='continuous').fit(X, y)
//! >>> [float(v) for v in e.encodings_[0]]
//!       -> [3.122887864823349, 4.455331412103746, 4.0]
//! >>> float(e.target_mean_)  -> 3.8333333333333335
//!   (lambda_ = var*count / (var*count + ssd/count); enc = lam*mean + (1-lam)*ymean)
//!
//! # C — cross-fit fit_transform, cv=5, smooth=10.0, DETERMINISTIC (shuffle=False),
//! #     X=[[0],[1],[0],[1],[0],[1],[0],[1],[0],[1]], y=1..10:
//! >>> e = TargetEncoder(smooth=10.0, target_type='continuous', shuffle=False)
//! >>> e.fit_transform(X, y).ravel()
//!       -> [6.357142857142857, 6.642857142857143, 5.857142857142857,
//!           6.142857142857143, 5.357142857142857, 5.642857142857143,
//!           4.857142857142857, 5.142857142857143, 4.357142857142857,
//!           4.642857142857143]
//!   ferrolearn's `fit_transform` cross-fits over the same deterministic KFold;
//!   sklearn's DEFAULT `shuffle=True` output is non-reproducible (no seed) so the
//!   reproducible `shuffle=False` oracle is pinned (R-HONEST-4). Fit+Transform
//!   (full encodings) differs: [5.3333..,5.6666.. alternating] == sklearn's
//!   .transform() (NOT .fit_transform()).
//! ```

use ferrolearn_core::traits::{Fit, Transform};
use ferrolearn_preprocess::TargetEncoder;
use ndarray::{Array1, Array2, array};

/// Divergence (#2342): sklearn `TargetEncoder`'s DEFAULT `smooth` is the string
/// `"auto"` (empirical Bayes) — `sklearn/preprocessing/_target_encoder.py:189`
/// (`"smooth": [StrOptions({"auto"}), Interval(Real, 0, None, closed="left")]`,
/// default constructor `:200` `smooth="auto"`). ferrolearn's
/// `TargetEncoder::default()` sets `smooth = 1.0`
/// (`ferrolearn-preprocess/src/target_encoder.rs:154-156`), a fixed-smooth
/// m-estimate. So a DEFAULT-constructed ferrolearn encoder produces different
/// `encodings_` than a DEFAULT-constructed sklearn encoder.
///
/// Live oracle (sklearn 1.5.2): `TargetEncoder(target_type='continuous')`
/// (default smooth='auto'), X=[[0],[0],[1],[1],[2]], y=[1,2,3,4,10]:
///   encodings_[0] -> [1.5308641975308643, 3.506172839506173, 10.0]
/// ferrolearn default (smooth=1.0) yields the fixed-smooth values, which differ.
#[test]
fn divergence_default_smooth_is_auto() {
    // sklearn 1.5.2 live oracle — DEFAULT encoder (smooth='auto'):
    const SK_CAT0: f64 = 1.530_864_197_530_864_3;
    const SK_CAT1: f64 = 3.506_172_839_506_173;
    const SK_CAT2: f64 = 10.0;

    let x: Array2<usize> = array![[0usize], [0], [1], [1], [2]];
    let y: Array1<f64> = array![1.0, 2.0, 3.0, 4.0, 10.0];

    // ferrolearn's Default == smooth 1.0 (NOT auto).
    let fitted = TargetEncoder::<f64>::default().fit(&x, &y).unwrap();
    let m0 = &fitted.category_maps()[0];

    assert!(
        (m0[&0] - SK_CAT0).abs() < 1e-7,
        "cat0: ferro(default)={} != sklearn(default smooth='auto')={}",
        m0[&0],
        SK_CAT0
    );
    assert!(
        (m0[&1] - SK_CAT1).abs() < 1e-7,
        "cat1: ferro(default)={} != sklearn(default smooth='auto')={}",
        m0[&1],
        SK_CAT1
    );
    assert!(
        (m0[&2] - SK_CAT2).abs() < 1e-7,
        "cat2: ferro(default)={} != sklearn(default smooth='auto')={}",
        m0[&2],
        SK_CAT2
    );
}

/// Divergence (#2343): ferrolearn has NO `smooth='auto'` empirical-Bayes path
/// (src REQ-4 is NOT-STARTED, `target_encoder.rs:33`). sklearn computes a
/// per-category shrinkage `lambda_` from the within-category sum-of-squared-
/// deviations vs the global variance and blends the category mean toward the
/// global mean — `sklearn/preprocessing/_target_encoder_fast.pyx:151-165`:
///   `lambda_ = y_variance*count / (y_variance*count + ssd/count)`
///   `enc      = lambda_*means[cat] + (1 - lambda_)*y_mean`.
/// No `smooth` setting on the ferrolearn fixed-smooth m-estimate can reproduce
/// this. The DEFAULT path additionally takes this branch (#2342).
///
/// Live oracle (sklearn 1.5.2): `TargetEncoder(smooth='auto', ...)`,
/// X=[[0],[0],[0],[1],[1],[2]], y=[1,3,5,2,8,4]:
///   encodings_[0] -> [3.122887864823349, 4.455331412103746, 4.0]
/// We pin against ferrolearn's BEST fixed-smooth approximation (the default
/// smooth=1.0); the empirical-Bayes values are unreachable, so this FAILS.
#[test]
fn divergence_smooth_auto_empirical_bayes() {
    // sklearn 1.5.2 live oracle — smooth='auto':
    const SK_CAT0: f64 = 3.122_887_864_823_349;
    const SK_CAT1: f64 = 4.455_331_412_103_746;
    const SK_CAT2: f64 = 4.0;

    let x: Array2<usize> = array![[0usize], [0], [0], [1], [1], [2]];
    let y: Array1<f64> = array![1.0, 3.0, 5.0, 2.0, 8.0, 4.0];

    // ferrolearn cannot express auto-smoothing; default (smooth=1.0) is the
    // closest fixed-smooth and does NOT match the empirical-Bayes encodings.
    let fitted = TargetEncoder::<f64>::default().fit(&x, &y).unwrap();
    let m0 = &fitted.category_maps()[0];

    assert!(
        (m0[&0] - SK_CAT0).abs() < 1e-7,
        "cat0: ferro={} != sklearn(smooth='auto')={}",
        m0[&0],
        SK_CAT0
    );
    assert!(
        (m0[&1] - SK_CAT1).abs() < 1e-7,
        "cat1: ferro={} != sklearn(smooth='auto')={}",
        m0[&1],
        SK_CAT1
    );
    assert!(
        (m0[&2] - SK_CAT2).abs() < 1e-7,
        "cat2: ferro={} != sklearn(smooth='auto')={}",
        m0[&2],
        SK_CAT2
    );
}

/// Divergence (#2344): sklearn `TargetEncoder.fit_transform` uses a CROSS-FIT
/// (deterministic KFold, default cv=5, NOT shuffled) so each fold's rows are
/// encoded with the OTHER folds' statistics, preventing target leakage —
/// `sklearn/preprocessing/_target_encoder.py:232` (`cv = KFold(self.cv)`),
/// `:254-303` (per-fold `_fit_encoding_binary_or_continuous` on `y_train` then
/// `_transform_X_ordinal` on the test rows). The `cv` is deterministic, so the
/// out-of-fold output is parity-matchable.
///
/// ferrolearn exposes NO `fit_transform`; the only path is `Fit` then
/// `Transform`, which uses the FULL `encodings_` (sklearn's `.transform()`,
/// explicitly NOT equal to `.fit_transform()` per `_target_encoder.py:308-311`).
///
/// Live oracle (sklearn 1.5.2): smooth=10.0, cv=5, DETERMINISTIC KFold
/// (`shuffle=False`), X=[[0],[1],[0],[1],[0],[1],[0],[1],[0],[1]], y=1..10:
///   TargetEncoder(smooth=10.0, target_type='continuous', shuffle=False)
///       .fit_transform(X,y).ravel() ->
///     [6.357142857142857, 6.642857142857143, 5.857142857142857,
///      6.142857142857143, 5.357142857142857, 5.642857142857143,
///      4.857142857142857, 5.142857142857143, 4.357142857142857,
///      4.642857142857143]
///
/// NOTE (R-HONEST-4): the original pin captured sklearn's DEFAULT `shuffle=True`
/// `fit_transform` output ([5.821.., …]) which is NON-REPRODUCIBLE (no seed),
/// so it could not be matched. ferrolearn's `fit_transform` exposes no
/// `shuffle`/`random_state` (REQ-8 NOT-STARTED), so it implements sklearn's
/// deterministic `shuffle=False` KFold (`_split.py:521-534`), whose oracle output
/// is the stable values above (verified bit-stable across 20 runs). This still
/// pins the cross-fit DIVERGENCE: `fit_transform` (cross-fit) != `fit().transform()`
/// (full encodings), per `_target_encoder.py:235-238`.
#[test]
fn divergence_crossfit_fit_transform() {
    // sklearn 1.5.2 live oracle — fit_transform (cross-fit, cv=5, smooth=10.0,
    // shuffle=False — the reproducible deterministic-KFold path):
    let sk_fit_transform: [f64; 10] = [
        6.357_142_857_142_857,
        6.642_857_142_857_143,
        5.857_142_857_142_857,
        6.142_857_142_857_143,
        5.357_142_857_142_857,
        5.642_857_142_857_143,
        4.857_142_857_142_857,
        5.142_857_142_857_143,
        4.357_142_857_142_857,
        4.642_857_142_857_143,
    ];

    let x: Array2<usize> = array![[0], [1], [0], [1], [0], [1], [0], [1], [0], [1]];
    let y: Array1<f64> = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

    // ferrolearn's cross-fit fit_transform (deterministic KFold, cv=5).
    let out = TargetEncoder::<f64>::new(10.0)
        .fit_transform(&x, &y)
        .unwrap();

    for i in 0..10 {
        assert!(
            (out[[i, 0]] - sk_fit_transform[i]).abs() < 1e-7,
            "row {i}: ferro fit_transform={} != sklearn fit_transform (cross-fit, shuffle=False)={}",
            out[[i, 0]],
            sk_fit_transform[i]
        );
    }

    // Cross-fit MUST differ from the full-encoding fit().transform() (the leakage
    // distinction, `_target_encoder.py:235-238`).
    let full = TargetEncoder::<f64>::new(10.0)
        .fit(&x, &y)
        .unwrap()
        .transform(&x)
        .unwrap();
    let differs = (0..10).any(|i| (out[[i, 0]] - full[[i, 0]]).abs() > 1e-9);
    assert!(
        differs,
        "fit_transform (cross-fit) must differ from fit().transform() (full encodings)"
    );
}
