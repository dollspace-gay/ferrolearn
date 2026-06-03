//! Divergence / parity tests for `GaussianMixture` (`ferrolearn-cluster/src/gmm.rs`)
//! against the live scikit-learn 1.5.2 oracle (`sklearn/mixture/_gaussian_mixture.py`).
//!
//! All expected values are produced by a LIVE `from sklearn.mixture import
//! GaussianMixture` call (run from /tmp, `random_state` fixed) and recorded in the
//! doc-comment of each test — NEVER literal-copied from the ferrolearn side (R-CHAR-3).
//!
//! Green-guards (PASS against current code): REQ-1 partition, REQ-2 predict_proba
//! contract, REQ-3 well-separated weights_/means_/covariances_ value-match, REQ-4
//! defaults, plus a Diag-covariance score control (proves the Diag arm is correct).
//!
//! REQ-5 pin (FAILS until the `Full`/`Tied` `log|Σ|` double-count is fixed):
//! `divergence_score_full_double_counts_log_det`.

use ferrolearn_cluster::{CovarianceType, GaussianMixture};
use ferrolearn_core::traits::{Fit, Predict};
use ndarray::Array2;

/// The in-tree 2-blob fixture (two 6-point blobs at ~(0,0) and ~(10,10)).
/// Identical array used in the live sklearn oracle calls below.
fn two_blobs() -> Array2<f64> {
    Array2::from_shape_vec(
        (12, 2),
        vec![
            0.0, 0.0, 0.1, 0.0, 0.0, 0.1, -0.1, 0.0, 0.0, -0.1, 0.1, 0.1, 10.0, 10.0, 10.1, 10.0,
            10.0, 10.1, 9.9, 10.0, 10.0, 9.9, 10.1, 10.1,
        ],
    )
    .unwrap()
}

/// Canonicalize a 2-component fit so component order is independent of the
/// (RNG-dependent) label assignment: sort the two components by their mean's
/// first coordinate. Returns the permutation `[low_idx, high_idx]`.
fn order_by_mean0(means: &Array2<f64>) -> [usize; 2] {
    if means[[0, 0]] <= means[[1, 0]] {
        [0, 1]
    } else {
        [1, 0]
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// REQ-1 — well-separated PARTITION up-to-permutation (GREEN-GUARD)
// ─────────────────────────────────────────────────────────────────────────────

/// Mirrors sklearn `BaseMixture.predict` argmax of weighted log-prob
/// (`sklearn/mixture/_base.py:369-385`); final e-step labels (`:286-288`).
///
/// Live oracle (sklearn 1.5.2, from /tmp):
/// ```text
/// X=np.array([[0.,0.],[0.1,0.],[0.,0.1],[-0.1,0.],[0.,-0.1],[0.1,0.1],
///             [10.,10.],[10.1,10.],[10.,10.1],[9.9,10.],[10.,9.9],[10.1,10.1]])
/// GaussianMixture(n_components=2,random_state=42,max_iter=200).fit_predict(X)
///   -> [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
/// ```
/// i.e. the first 6 points co-cluster, the last 6 co-cluster, and the two
/// groups differ. We assert co-membership (up to a label permutation).
#[test]
fn req1_partition_two_blobs_up_to_permutation() {
    let x = two_blobs();
    let fitted = GaussianMixture::<f64>::new(2)
        .with_random_state(42)
        .with_max_iter(200)
        .fit(&x, &())
        .unwrap();
    let labels = fitted.predict(&x).unwrap();

    // sklearn partition: {0..6} together, {6..12} together, groups distinct.
    for i in 0..6 {
        assert_eq!(labels[i], labels[0], "first blob must co-cluster (idx {i})");
    }
    for i in 6..12 {
        assert_eq!(
            labels[i], labels[6],
            "second blob must co-cluster (idx {i})"
        );
    }
    assert_ne!(
        labels[0], labels[6],
        "the two blobs must be distinct clusters"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// REQ-2 — predict_proba / transform contract (GREEN-GUARD)
// ─────────────────────────────────────────────────────────────────────────────

/// Mirrors `BaseMixture.predict_proba = exp(log_resp)` (`_base.py:387-404`) +
/// `_estimate_log_prob_resp` logsumexp normalization (`:507-531`).
///
/// Live oracle (sklearn 1.5.2, from /tmp), same fixture/params:
/// ```text
/// gm.predict_proba(X).shape          -> (12, 2)
/// gm.predict_proba(X)[0].tolist()    -> [1.0, 0.0]
/// gm.predict_proba(X)[6].tolist()    -> [0.0, 1.0]
/// each row sums to exactly 1.
/// ```
#[test]
fn req2_predict_proba_shape_and_rows_sum_to_one() {
    let x = two_blobs();
    let fitted = GaussianMixture::<f64>::new(2)
        .with_random_state(42)
        .with_max_iter(200)
        .fit(&x, &())
        .unwrap();
    let proba = fitted.predict_proba(&x).unwrap();

    assert_eq!(proba.dim(), (12, 2), "predict_proba shape must be (n, k)");
    for (n, row) in proba.rows().into_iter().enumerate() {
        let s: f64 = row.iter().sum();
        assert!(
            (s - 1.0).abs() < 1e-9,
            "row {n} of predict_proba must sum to 1 (got {s})"
        );
        for &v in row {
            assert!(
                (-1e-12..=1.0 + 1e-12).contains(&v),
                "predict_proba entries must be probabilities"
            );
        }
    }

    // Hard-assigned points: sklearn predict_proba[0]=[1,0], [6]=[0,1].
    // Up to permutation: row 0 is ~one-hot on the component opposite row 6.
    let perm = order_by_mean0(fitted.means());
    // row 0 belongs to the low-mean (near origin) component.
    assert!(
        (proba[[0, perm[0]]] - 1.0).abs() < 1e-6,
        "point 0 should be ~certain in the near-origin component"
    );
    assert!(
        (proba[[6, perm[1]]] - 1.0).abs() < 1e-6,
        "point 6 should be ~certain in the far component"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// REQ-3 — well-separated weights_/means_/covariances_ VALUE-match (GREEN-GUARD)
// ─────────────────────────────────────────────────────────────────────────────

/// Mirrors `_estimate_gaussian_parameters` (`_gaussian_mixture.py:259-296`).
///
/// Live oracle (sklearn 1.5.2, from /tmp), same fixture/params:
/// ```text
/// gm.weights_     -> [0.5, 0.5]
/// gm.means_       -> [[0.016666666666666663, 0.016666666666666663],
///                     [10.016666666666664, 10.016666666666664]]
/// gm.covariances_ -> [[[0.0047232222222222215, 0.0013888888888888885],
///                      [0.0013888888888888885, 0.0047232222222222215]],
///                     [[0.004723222222222187,  0.0013888888888888785],
///                      [0.0013888888888888785, 0.004723222222222187]]]
/// ```
/// Canonicalized by sorting components by mean[0]; assert ferrolearn matches
/// to a tight tolerance up to that permutation.
#[test]
fn req3_well_separated_value_match() {
    let x = two_blobs();
    let fitted = GaussianMixture::<f64>::new(2)
        .with_random_state(42)
        .with_max_iter(200)
        .fit(&x, &())
        .unwrap();

    // Live sklearn values (recorded above).
    const SK_WEIGHTS: [f64; 2] = [0.5, 0.5];
    const SK_MEANS: [[f64; 2]; 2] = [
        [0.016_666_666_666_666_663, 0.016_666_666_666_666_663],
        [10.016_666_666_666_664, 10.016_666_666_666_664],
    ];
    // Covariance block (both components essentially identical here).
    const SK_COV: [[f64; 2]; 2] = [
        [0.004_723_222_222_222_221_5, 0.001_388_888_888_888_888_5],
        [0.001_388_888_888_888_888_5, 0.004_723_222_222_222_221_5],
    ];

    let perm = order_by_mean0(fitted.means());
    let w = fitted.weights();
    let m = fitted.means();
    let cov = fitted.covariances();

    for c in 0..2 {
        let fc = perm[c]; // ferrolearn index matching sklearn component c
        assert!(
            (w[fc] - SK_WEIGHTS[c]).abs() < 1e-9,
            "weight[{c}] mismatch: ferro {} sklearn {}",
            w[fc],
            SK_WEIGHTS[c]
        );
        for j in 0..2 {
            assert!(
                (m[[fc, j]] - SK_MEANS[c][j]).abs() < 1e-9,
                "mean[{c}][{j}] mismatch: ferro {} sklearn {}",
                m[[fc, j]],
                SK_MEANS[c][j]
            );
        }
        // Full covariance stored as (k*d, d) stacked blocks: block fc occupies
        // rows [fc*d .. fc*d+d).
        for i in 0..2 {
            for j in 0..2 {
                let v = cov[[fc * 2 + i, j]];
                assert!(
                    (v - SK_COV[i][j]).abs() < 1e-8,
                    "cov[{c}][{i}][{j}] mismatch: ferro {} sklearn {}",
                    v,
                    SK_COV[i][j]
                );
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// REQ-4 — matching defaults (GREEN-GUARD)
// ─────────────────────────────────────────────────────────────────────────────

/// sklearn `GaussianMixture.__init__` defaults (`_gaussian_mixture.py:705-709`)
/// and `BaseMixture._parameter_constraints` (`_base.py:50-63`) require
/// `covariance_type='full'`, `max_iter=100`, `tol=1e-3`, and `n_init=1`.
#[test]
fn req4_defaults_match_sklearn() {
    let gmm = GaussianMixture::<f64>::new(2);
    assert_eq!(gmm.covariance_type, CovarianceType::Full);
    assert_eq!(gmm.max_iter, 100);
    assert_eq!(gmm.n_init, 1);
    assert!((gmm.tol - 1e-3).abs() < 1e-18);
}

// ─────────────────────────────────────────────────────────────────────────────
// Diag-covariance score CONTROL (GREEN-GUARD) — proves the Diag arm is correct
// ─────────────────────────────────────────────────────────────────────────────

/// Control: the Diag arm folds `log|Σ|` into `log_norm` once and adds only
/// `-0.5·maha`, so its absolute `score(X)` already matches sklearn. This is the
/// contrast against the Full/Tied double-count (next test).
///
/// Live oracle (sklearn 1.5.2, from /tmp), same fixture, `covariance_type='diag'`:
/// ```text
/// GaussianMixture(n_components=2,covariance_type='diag',random_state=42,
///                 max_iter=200).fit(X).score(X)
///   -> 1.8244515110025812
/// (means_ -> [[0.0166.., 0.0166..],[10.0166.., 10.0166..]], weights_ -> [0.5,0.5])
/// ```
#[test]
fn diag_score_control_matches_sklearn() {
    let x = two_blobs();
    let fitted = GaussianMixture::<f64>::new(2)
        .with_covariance_type(CovarianceType::Diag)
        .with_random_state(42)
        .with_max_iter(200)
        .fit(&x, &())
        .unwrap();
    let score = fitted.score(&x).unwrap();
    const SK_DIAG_SCORE: f64 = 1.824_451_511_002_581_2;
    assert!(
        (score - SK_DIAG_SCORE).abs() < 1e-6,
        "Diag score should already match sklearn: ferro {score} sklearn {SK_DIAG_SCORE}"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// REQ-5 — Full/Tied score double-counts log|Σ| (PIN — FAILS until fixed)
// ─────────────────────────────────────────────────────────────────────────────

/// Divergence: `ferrolearn_cluster::gmm`'s Full/Tied Gaussian log-density
/// double-counts `log|Σ|`. `fn log_det_and_norm_full` returns
/// `log_norm = -(d/2)·ln(2π) - 0.5·log_det`, and the `Full | Tied` arm of
/// `fn log_responsibilities` then computes
/// `log_w + log_norm - 0.5·(log_det + maha)` — subtracting `0.5·log_det` TWICE.
/// sklearn's `_estimate_log_gaussian_prob`
/// (`sklearn/mixture/_gaussian_mixture.py:507`):
/// `-0.5*(n_features*log(2*pi) + log_prob) + log_det` counts `log|Σ|` once.
///
/// On the 2-blob fixture the fitted params VALUE-match sklearn (REQ-3 above), so
/// `score` is the ONLY divergence: ferrolearn's per-sample log-density is shifted
/// by `+0.5·log|Σ|_k = -5.40...` (log_det ≈ -10.8005), i.e. ferrolearn `score`
/// ≈ sklearn + 5.40.
///
/// Live oracle (sklearn 1.5.2, from /tmp), same fixture (full covariance):
/// ```text
/// gm = GaussianMixture(n_components=2,random_state=42,max_iter=200).fit(X)
/// gm.score(X)          -> 1.8696902967180025
/// gm.score_samples(X)[0] -> 2.8240114282539723
/// gm.lower_bound_      -> 1.8696902967180025
/// gm.aic(X)            -> -22.87256712123206
/// gm.bic(X)            -> -17.538593973564055
/// ```
/// ferrolearn currently returns ~7.27 / ~8.22 / ~7.27 / ~-152.48 / ~-147.14.
///
/// Tracking: #1102 (REQ-5 blocker). Un-ignored: release-blocker score
/// divergence on the default covariance_type.
#[test]
fn req5_score_full_matches_sklearn() {
    let x = two_blobs();
    let fitted = GaussianMixture::<f64>::new(2)
        .with_random_state(42)
        .with_max_iter(200)
        .fit(&x, &())
        .unwrap();

    // Live sklearn values (recorded above).
    const SK_SCORE: f64 = 1.869_690_296_718_002_5;
    const SK_SCORE_SAMPLES_0: f64 = 2.824_011_428_253_972_3;
    const SK_AIC: f64 = -22.872_567_121_232_06;
    const SK_BIC: f64 = -17.538_593_973_564_055;

    let score = fitted.score(&x).unwrap();
    let score_samples = fitted.score_samples(&x).unwrap();
    let aic = fitted.aic(&x).unwrap();
    let bic = fitted.bic(&x).unwrap();
    let lower_bound = fitted.lower_bound();

    assert!(
        (score - SK_SCORE).abs() < 1e-6,
        "Full score must match sklearn: ferro {score} sklearn {SK_SCORE}"
    );
    assert!(
        (score_samples[0] - SK_SCORE_SAMPLES_0).abs() < 1e-6,
        "Full score_samples[0] must match sklearn: ferro {} sklearn {SK_SCORE_SAMPLES_0}",
        score_samples[0]
    );
    assert!(
        (lower_bound - SK_SCORE).abs() < 1e-6,
        "lower_bound_ must match sklearn: ferro {lower_bound} sklearn {SK_SCORE}"
    );
    assert!(
        (aic - SK_AIC).abs() < 1e-3,
        "Full aic must match sklearn: ferro {aic} sklearn {SK_AIC}"
    );
    assert!(
        (bic - SK_BIC).abs() < 1e-3,
        "Full bic must match sklearn: ferro {bic} sklearn {SK_BIC}"
    );
}
