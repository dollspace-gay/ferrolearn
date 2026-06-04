//! Divergence tests for `ferrolearn_decomp::IncrementalPCA` vs scikit-learn
//! 1.5.2 `class IncrementalPCA(_BasePCA)`
//! (`sklearn/decomposition/_incremental_pca.py:19`).
//!
//! All expected values are produced by the live sklearn 1.5.2 oracle, run from
//! `/tmp` (R-CHAR-3); they are NEVER literal-copied from the ferrolearn side.
//!
//! Tracking:
//! - DIV-1 svd_flip sign: #1585
//! - DIV-2 multi-batch mean_correction: #1586
//! - DIV-3 explained_variance_ratio_ denominator: #1587
//! - Candidate (flagged, not pinned): n_components == n_features rejected: #1590
//!
//! Layout: the three FIXABLE divergences are `#[ignore]`'d failing pins; the
//! GREEN-GUARD tests must PASS against the current code and confirm that the
//! sign-invariant single-batch values (explained_variance_, singular_values_,
//! mean_) already match sklearn, cleanly isolating the three divergences.

use ferrolearn_core::traits::{Fit, Transform};
use ferrolearn_decomp::IncrementalPCA;
use ndarray::{Array2, array};

// ---------------------------------------------------------------------------
// Live sklearn 1.5.2 oracle constants (run from /tmp).
// ---------------------------------------------------------------------------

/// Fixture A: 6x2, used for single-batch sign-invariant green guards + DIV-3.
fn fixture_a() -> Array2<f64> {
    array![
        [2.5, 2.4],
        [0.5, 0.7],
        [2.2, 2.9],
        [1.9, 2.2],
        [3.1, 3.0],
        [2.3, 2.7]
    ]
}

/// Fixture B: 6x3, used for DIV-1 (single-batch sign), DIV-2 (multi-batch),
/// and multi-batch/single-batch green guards.
fn fixture_b() -> Array2<f64> {
    array![
        [2.5, 2.4, 1.0],
        [0.5, 0.7, 3.0],
        [2.2, 2.9, 1.5],
        [1.9, 2.2, 2.0],
        [3.1, 3.0, 0.5],
        [2.3, 2.7, 1.2]
    ]
}

// --- Oracle: IncrementalPCA(n_components=1, batch_size=6) on fixture_b ------
// (single batch isolates from DIV-2; exercises DIV-1 sign)
#[allow(
    clippy::excessive_precision,
    reason = "live sklearn 1.5.2 oracle (R-CHAR-3)"
)]
const SK_B_SINGLE_COMPONENTS: [f64; 3] = [0.5910710003, 0.5550545694, -0.5852772827];
#[allow(
    clippy::excessive_precision,
    reason = "live sklearn 1.5.2 oracle (R-CHAR-3)"
)]
const SK_B_SINGLE_EXPLAINED_VARIANCE: f64 = 2.1557819448;
#[allow(
    clippy::excessive_precision,
    reason = "live sklearn 1.5.2 oracle (R-CHAR-3)"
)]
const SK_B_SINGLE_SINGULAR_VALUE: f64 = 3.2831249936;
#[allow(
    clippy::excessive_precision,
    reason = "live sklearn 1.5.2 oracle (R-CHAR-3)"
)]
const SK_B_MEAN: [f64; 3] = [2.0833333333, 2.3166666667, 1.5333333333];

// --- Oracle: IncrementalPCA(n_components=1, batch_size=2) on fixture_b ------
// (multi-batch; exercises DIV-2)
#[allow(
    clippy::excessive_precision,
    reason = "live sklearn 1.5.2 oracle (R-CHAR-3)"
)]
const SK_B_MULTI_EXPLAINED_VARIANCE: f64 = 2.1555395654;
#[allow(
    clippy::excessive_precision,
    reason = "live sklearn 1.5.2 oracle (R-CHAR-3)"
)]
const SK_B_MULTI_SINGULAR_VALUE: f64 = 3.2829404239;

// --- Oracle: IncrementalPCA(n_components=1) on fixture_a (single batch) -----
#[allow(
    clippy::excessive_precision,
    reason = "live sklearn 1.5.2 oracle (R-CHAR-3)"
)]
const SK_A_EXPLAINED_VARIANCE: f64 = 1.4323494544;
#[allow(
    clippy::excessive_precision,
    reason = "live sklearn 1.5.2 oracle (R-CHAR-3)"
)]
const SK_A_SINGULAR_VALUE: f64 = 2.6761441053;
#[allow(
    clippy::excessive_precision,
    reason = "live sklearn 1.5.2 oracle (R-CHAR-3)"
)]
const SK_A_EXPLAINED_VARIANCE_RATIO: f64 = 0.9682398295;
#[allow(
    clippy::excessive_precision,
    reason = "live sklearn 1.5.2 oracle (R-CHAR-3)"
)]
const SK_A_MEAN: [f64; 2] = [2.0833333333, 2.3166666667];

// ===========================================================================
// DIV-1 (#1585): svd_flip(u_based_decision=False) SIGN convention.
// ===========================================================================

/// Divergence: ferrolearn's `partial_fit_batch` copies the raw Jacobi
/// `thin_svd` Vt rows into `components_`
/// (`ferrolearn-decomp/src/incremental_pca.rs:290-295`) with NO sign
/// convention, whereas sklearn applies
/// `U, Vt = svd_flip(U, Vt, u_based_decision=False)`
/// (`sklearn/decomposition/_incremental_pca.py:357`), which makes each
/// component (Vt) ROW's max-abs entry POSITIVE
/// (`sklearn/utils/extmath.py:897-905`, numpy argmax first-on-ties).
///
/// Input: `IncrementalPCA::<f64>::new(1).with_batch_size(6)` on the 6x3
/// fixture_b (single batch, so DIV-2/mean_correction does NOT apply — isolates
/// the sign). sklearn `components_` row = `[+0.5910710003, +0.5550545694,
/// -0.5852772827]` (max-abs at idx 0, POSITIVE). ferrolearn returns the same
/// magnitudes but with the WHOLE row negated: `[-0.591..., -0.555..., +0.585...]`.
/// Tracking: #1585
#[test]
fn divergence_svd_flip_sign() {
    let x = fixture_b();
    let fitted = IncrementalPCA::<f64>::new(1)
        .with_batch_size(6)
        .fit(&x, &())
        .expect("fit single-batch should succeed");
    let comp = fitted.components();

    // Element-wise (including sign) match to the sklearn oracle.
    for j in 0..3 {
        let diff = (comp[[0, j]] - SK_B_SINGLE_COMPONENTS[j]).abs();
        assert!(
            diff < 1e-6,
            "components[0][{j}] = {} but sklearn = {} (diff {diff})",
            comp[[0, j]],
            SK_B_SINGLE_COMPONENTS[j]
        );
    }

    // Independent of the oracle values: svd_flip guarantees the max-abs entry
    // of each component row is positive.
    let mut max_abs = 0.0_f64;
    let mut max_val = 0.0_f64;
    for j in 0..3 {
        let v = comp[[0, j]];
        if v.abs() > max_abs {
            max_abs = v.abs();
            max_val = v;
        }
    }
    assert!(
        max_val > 0.0,
        "max-abs entry of component row 0 should be positive (svd_flip), got {max_val}"
    );
}

// ===========================================================================
// DIV-2 (#1586): multi-batch mean_correction row + batch-mean centring.
// ===========================================================================

/// Divergence: for `n_samples_seen > 0` sklearn centres the batch by the BATCH
/// mean and stacks a `mean_correction` row
/// `sqrt((n_samples_seen/n_total)*n_samples)*(mean_ - col_batch_mean)`
/// (`sklearn/decomposition/_incremental_pca.py:342-354`). ferrolearn centres
/// by the running mean and OMITS the mean_correction row
/// (`ferrolearn-decomp/src/incremental_pca.rs:245-272`), biasing multi-batch
/// fits.
///
/// Input: `IncrementalPCA::<f64>::new(1).with_batch_size(2)` on 6x3 fixture_b
/// (three batches with differing batch means). Compared via the SIGN-INVARIANT
/// `explained_variance_` (and `singular_values_`) so the assertion is not
/// contaminated by DIV-1. sklearn multi-batch `explained_variance_ =
/// 2.1555395654`; ferrolearn (no mean_correction) produces ~1.826, a gross
/// divergence. Note the single-batch value (2.15578, green-guarded below) DOES
/// match sklearn, cleanly isolating DIV-2 as the multi-batch bias.
/// Tracking: #1586
#[test]
fn divergence_multibatch_mean_correction() {
    let x = fixture_b();
    let fitted = IncrementalPCA::<f64>::new(1)
        .with_batch_size(2)
        .fit(&x, &())
        .expect("fit multi-batch should succeed");

    let ev = fitted.explained_variance()[0];
    let sv = fitted.singular_values()[0];

    assert!(
        (ev - SK_B_MULTI_EXPLAINED_VARIANCE).abs() < 1e-6,
        "multi-batch explained_variance_ = {ev} but sklearn = {SK_B_MULTI_EXPLAINED_VARIANCE}"
    );
    assert!(
        (sv - SK_B_MULTI_SINGULAR_VALUE).abs() < 1e-6,
        "multi-batch singular_values_ = {sv} but sklearn = {SK_B_MULTI_SINGULAR_VALUE}"
    );
}

// ===========================================================================
// DIV-3 (#1587): explained_variance_ratio_ denominator = total feature variance.
// ===========================================================================

/// Divergence: sklearn computes
/// `explained_variance_ratio_ = S**2 / sum(col_var * n_total_samples)`
/// (`sklearn/decomposition/_incremental_pca.py:359`) — a fraction of the TOTAL
/// feature variance (< 1 when components are dropped). ferrolearn divides by
/// the sum of the RETAINED explained variances
/// (`ferrolearn-decomp/src/incremental_pca.rs:307-322`), so the ratio sums to
/// EXACTLY 1.0 over the retained components.
///
/// Input: `IncrementalPCA::<f64>::new(1)` on 6x2 fixture_a (single batch, so
/// DIV-2 does not apply; n_components=1 < n_features=2 so the difference is
/// observable). sklearn `explained_variance_ratio_[0] = 0.9682398295`;
/// ferrolearn returns 1.0.
/// Tracking: #1587
#[test]
fn divergence_explained_variance_ratio_denominator() {
    let x = fixture_a();
    let fitted = IncrementalPCA::<f64>::new(1)
        .fit(&x, &())
        .expect("fit single-batch should succeed");
    let ratio = fitted.explained_variance_ratio()[0];
    assert!(
        (ratio - SK_A_EXPLAINED_VARIANCE_RATIO).abs() < 1e-6,
        "explained_variance_ratio_[0] = {ratio} but sklearn = {SK_A_EXPLAINED_VARIANCE_RATIO}"
    );
}

// ===========================================================================
// GREEN GUARDS — must PASS against the current code.
// These confirm the SHIPPED structural/value behavior and, critically, that
// the sign-invariant single-batch values ALREADY match sklearn (isolating the
// divergences to sign + ratio-denom + multi-batch).
// ===========================================================================

/// GREEN: single-batch `explained_variance_` matches sklearn element-wise
/// (`S**2/(n-1)`, sign-invariant). Confirms DIV-2 is purely a multi-batch bug.
/// sklearn fixture_b single-batch `explained_variance_[0] = 2.1557819448`.
#[test]
fn green_single_batch_explained_variance_matches() {
    let x = fixture_b();
    let fitted = IncrementalPCA::<f64>::new(1)
        .with_batch_size(6)
        .fit(&x, &())
        .expect("fit");
    let ev = fitted.explained_variance()[0];
    assert!(
        (ev - SK_B_SINGLE_EXPLAINED_VARIANCE).abs() < 1e-6,
        "single-batch explained_variance_ = {ev} but sklearn = {SK_B_SINGLE_EXPLAINED_VARIANCE}"
    );
}

/// GREEN: single-batch `singular_values_` matches sklearn (sign-invariant).
/// sklearn fixture_b single-batch `singular_values_[0] = 3.2831249936`.
#[test]
fn green_single_batch_singular_values_matches() {
    let x = fixture_b();
    let fitted = IncrementalPCA::<f64>::new(1)
        .with_batch_size(6)
        .fit(&x, &())
        .expect("fit");
    let sv = fitted.singular_values()[0];
    assert!(
        (sv - SK_B_SINGLE_SINGULAR_VALUE).abs() < 1e-6,
        "single-batch singular_values_ = {sv} but sklearn = {SK_B_SINGLE_SINGULAR_VALUE}"
    );
}

/// GREEN: fixture_a single-batch `explained_variance_`/`singular_values_`
/// match sklearn (the case ferrolearn IS value-equal on; sign-invariant).
#[test]
fn green_fixture_a_explained_variance_and_singular_values_match() {
    let x = fixture_a();
    let fitted = IncrementalPCA::<f64>::new(1).fit(&x, &()).expect("fit");
    let ev = fitted.explained_variance()[0];
    let sv = fitted.singular_values()[0];
    assert!(
        (ev - SK_A_EXPLAINED_VARIANCE).abs() < 1e-6,
        "explained_variance_ = {ev} but sklearn = {SK_A_EXPLAINED_VARIANCE}"
    );
    assert!(
        (sv - SK_A_SINGULAR_VALUE).abs() < 1e-6,
        "singular_values_ = {sv} but sklearn = {SK_A_SINGULAR_VALUE}"
    );
}

/// GREEN: running `mean_` matches sklearn `mean_` element-wise on the
/// multi-batch fit (the streaming mean is exact regardless of mean_correction).
#[test]
fn green_multibatch_mean_matches_sklearn() {
    let x = fixture_b();
    let fitted = IncrementalPCA::<f64>::new(1)
        .with_batch_size(2)
        .fit(&x, &())
        .expect("fit");
    let mean = fitted.mean();
    for j in 0..3 {
        assert!(
            (mean[j] - SK_B_MEAN[j]).abs() < 1e-9,
            "mean_[{j}] = {} but sklearn = {}",
            mean[j],
            SK_B_MEAN[j]
        );
    }
}

/// GREEN: fixture_a running `mean_` matches sklearn.
#[test]
fn green_fixture_a_mean_matches_sklearn() {
    let x = fixture_a();
    let fitted = IncrementalPCA::<f64>::new(1).fit(&x, &()).expect("fit");
    let mean = fitted.mean();
    for j in 0..2 {
        assert!(
            (mean[j] - SK_A_MEAN[j]).abs() < 1e-9,
            "mean_[{j}] = {} but sklearn = {}",
            mean[j],
            SK_A_MEAN[j]
        );
    }
}

/// GREEN: components shape `(n_components, n_features)`; rows unit-norm;
/// transform shape; inverse_transform shape.
#[test]
fn green_shapes_and_unit_norm() {
    let x = fixture_b();
    let fitted = IncrementalPCA::<f64>::new(2)
        .with_batch_size(6)
        .fit(&x, &())
        .expect("fit");
    assert_eq!(fitted.components().dim(), (2, 3));

    for i in 0..2 {
        let norm: f64 = fitted
            .components()
            .row(i)
            .iter()
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-6,
            "component row {i} 2-norm = {norm}, expected 1.0"
        );
    }

    let projected = fitted.transform(&x).expect("transform");
    assert_eq!(projected.dim(), (6, 2));

    let inv = fitted
        .inverse_transform(&projected)
        .expect("inverse_transform");
    assert_eq!(inv.dim(), (6, 3));
}

/// GREEN: `n_samples_seen_` accumulates; partial_fit chaining gives the same
/// `n_samples_seen_` as a single fit.
#[test]
fn green_n_samples_seen_accumulation() {
    let x = fixture_b();
    let single = IncrementalPCA::<f64>::new(1)
        .with_batch_size(6)
        .fit(&x, &())
        .expect("fit");
    assert_eq!(single.n_samples_seen(), 6);

    let ipca = IncrementalPCA::<f64>::new(1);
    let b1 = array![[2.5, 2.4, 1.0], [0.5, 0.7, 3.0]];
    let b2 = array![[2.2, 2.9, 1.5], [1.9, 2.2, 2.0]];
    let b3 = array![[3.1, 3.0, 0.5], [2.3, 2.7, 1.2]];
    let s1 = ipca.partial_fit(&b1, None).expect("pf1");
    assert_eq!(s1.n_samples_seen(), 2);
    let s2 = ipca.partial_fit(&b2, Some(s1)).expect("pf2");
    assert_eq!(s2.n_samples_seen(), 4);
    let s3 = ipca.partial_fit(&b3, Some(s2)).expect("pf3");
    assert_eq!(s3.n_samples_seen(), 6);
}

/// GREEN: error / parameter contracts (ferrolearn's CURRENT behavior).
#[test]
fn green_error_contracts() {
    let x = fixture_b();
    // n_components == 0 -> Err
    assert!(IncrementalPCA::<f64>::new(0).fit(&x, &()).is_err());
    // n_components >= n_features -> Err (current ferrolearn behavior; see #1590 flag)
    assert!(IncrementalPCA::<f64>::new(3).fit(&x, &()).is_err());
    // n_samples < 2 -> Err
    let one = array![[1.0, 2.0, 3.0]];
    assert!(IncrementalPCA::<f64>::new(1).fit(&one, &()).is_err());
    // batch_size == 0 -> Err
    assert!(
        IncrementalPCA::<f64>::new(1)
            .with_batch_size(0)
            .fit(&x, &())
            .is_err()
    );
    // transform feature mismatch -> Err
    let fitted = IncrementalPCA::<f64>::new(1)
        .with_batch_size(6)
        .fit(&x, &())
        .expect("fit");
    let bad = array![[1.0, 2.0]];
    assert!(fitted.transform(&bad).is_err());
}

/// CANDIDATE DIV flag (#1590), pinned ONLY as the CURRENT ferrolearn behavior
/// so it stays green: sklearn allows `n_components == n_features`
/// (`sklearn/decomposition/_incremental_pca.py:297-308`,
/// `n_components <= min(n_samples, n_features)`), but ferrolearn rejects
/// `n_components >= n_features`. This is recorded as a candidate divergence in
/// the report; it is NOT pinned as a failing test because the resolution is
/// uncertain (the green-guard above already asserts the current reject).
#[test]
fn green_n_components_eq_n_features_currently_rejected() {
    // Documents the CURRENT (divergent) behavior; see report / #1590.
    let x = fixture_b();
    assert!(
        IncrementalPCA::<f64>::new(3).fit(&x, &()).is_err(),
        "ferrolearn currently rejects n_components == n_features (sklearn allows it; #1590)"
    );
}

/// GREEN: determinism — IncrementalPCA is a deterministic algorithm (no RNG);
/// two fits on identical input produce identical output.
#[test]
fn green_determinism() {
    let x = fixture_b();
    let f1 = IncrementalPCA::<f64>::new(1)
        .with_batch_size(2)
        .fit(&x, &())
        .expect("fit");
    let f2 = IncrementalPCA::<f64>::new(1)
        .with_batch_size(2)
        .fit(&x, &())
        .expect("fit");
    for j in 0..3 {
        assert_eq!(f1.components()[[0, j]], f2.components()[[0, j]]);
    }
    assert_eq!(f1.explained_variance()[0], f2.explained_variance()[0]);
}

/// GREEN: f32 path fits and transforms.
#[test]
fn green_f32_path() {
    let x: Array2<f32> = array![
        [2.5f32, 2.4, 1.0],
        [0.5, 0.7, 3.0],
        [2.2, 2.9, 1.5],
        [1.9, 2.2, 2.0],
        [3.1, 3.0, 0.5],
        [2.3, 2.7, 1.2]
    ];
    let fitted = IncrementalPCA::<f32>::new(1)
        .with_batch_size(6)
        .fit(&x, &())
        .expect("f32 fit");
    let projected = fitted.transform(&x).expect("f32 transform");
    assert_eq!(projected.ncols(), 1);
}
