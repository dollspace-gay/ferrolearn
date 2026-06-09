//! Divergence pins for `ferrolearn_decomp::IncrementalPCA` vs scikit-learn
//! 1.5.2 `IncrementalPCA.partial_fit`
//! (`sklearn/decomposition/_incremental_pca.py:356-367`), focused on the FULL
//! multi-batch incremental merge (n_components=3, multiple batches), which the
//! existing `divergence_incremental_pca.rs` (single-batch / n_components=1) does
//! NOT exercise.
//!
//! All expected values come from the LIVE sklearn 1.5.2 oracle, run from `/tmp`
//! (R-CHAR-3), on a deterministic 20x5 fixture `X = RandomState(42).randn(20,5)`
//! fit with `IncrementalPCA(n_components=3, batch_size=6)` (four batches: 6+6+6+2,
//! the last a partial batch). These are NEVER copied from the ferrolearn side.
//!
//! Oracle command (reproducible):
//! ```text
//! python3 -c "import numpy as np; from sklearn.decomposition import IncrementalPCA; \
//!   X=np.random.RandomState(42).randn(20,5); \
//!   m=IncrementalPCA(n_components=3,batch_size=6).fit(X); \
//!   print(m.components_, m.singular_values_, m.explained_variance_, \
//!         m.explained_variance_ratio_, m.transform(X)[0])"
//! ```
//!
//! Finding: ferrolearn's running `mean_`/`var_`/`n_samples_seen_` match sklearn
//! to ~1e-12 (green guards below), but the incremental-SVD outputs
//! (`components_`, `singular_values_`, `explained_variance_`,
//! `explained_variance_ratio_`, and hence `transform`) diverge by ~1e-2 / ~1e-3,
//! FAR beyond R-DEV-1 (~1e-6). Root cause: the hand-rolled Jacobi `thin_svd`
//! (`incremental_pca.rs:720`) does not reproduce sklearn/LAPACK's SVD of the
//! recursively-stacked 3-block merge matrix to working precision; the error
//! accumulates across batches.
//!
//! Tracking: #2386 (blocker).

use ferrolearn_core::traits::{Fit, Transform};
use ferrolearn_decomp::IncrementalPCA;
use ndarray::{Array2, array};

/// Deterministic fixture: `np.random.RandomState(42).randn(20, 5)`.
fn fixture() -> Array2<f64> {
    array![
        [
            0.496714153011,
            -0.138264301171,
            0.647688538101,
            1.523029856408,
            -0.234153374723
        ],
        [
            -0.234136956949,
            1.579212815507,
            0.767434729153,
            -0.469474385935,
            0.542560043586
        ],
        [
            -0.463417692812,
            -0.465729753570,
            0.241962271566,
            -1.913280244658,
            -1.724917832513
        ],
        [
            -0.562287529241,
            -1.012831120334,
            0.314247332595,
            -0.908024075521,
            -1.412303701335
        ],
        [
            1.465648768922,
            -0.225776300487,
            0.067528204688,
            -1.424748186213,
            -0.544382724525
        ],
        [
            0.110922589710,
            -1.150993577422,
            0.375698018346,
            -0.600638689919,
            -0.291693749793
        ],
        [
            -0.601706612229,
            1.852278184509,
            -0.013497224738,
            -1.057710928956,
            0.822544912103
        ],
        [
            -1.220843649971,
            0.208863595005,
            -1.959670123880,
            -1.328186048898,
            0.196861235869
        ],
        [
            0.738466579995,
            0.171368281190,
            -0.115648282388,
            -0.301103695589,
            -1.478521990367
        ],
        [
            -0.719844208395,
            -0.460638770960,
            1.057122226219,
            0.343618289568,
            -1.763040155363
        ],
        [
            0.324083969395,
            -0.385082280416,
            -0.676922000306,
            0.611676288841,
            1.030999522496
        ],
        [
            0.931280119116,
            -0.839217523223,
            -0.309212375851,
            0.331263431404,
            0.975545127122
        ],
        [
            -0.479174237845,
            -0.185658976664,
            -1.106334974006,
            -1.196206624081,
            0.812525822394
        ],
        [
            1.356240028571,
            -0.072010121580,
            1.003532897892,
            0.361636025048,
            -0.645119754605
        ],
        [
            0.361395605508,
            1.538036566466,
            -0.035826039110,
            1.564643655814,
            -2.619745104090
        ],
        [
            0.821902504375,
            0.087047068238,
            -0.299007350466,
            0.091760776536,
            -1.987568914601
        ],
        [
            -0.219671887838,
            0.357112571512,
            1.477894044742,
            -0.518270218274,
            -0.808493602893
        ],
        [
            -0.501757043585,
            0.915402117702,
            0.328751109660,
            -0.529760203767,
            0.513267433113
        ],
        [
            0.097077549348,
            0.968644990533,
            -0.702053093877,
            -0.327662146598,
            -0.392108153132
        ],
        [
            -1.463514948132,
            0.296120277065,
            0.261055272180,
            0.005113456642,
            -0.234587133375
        ]
    ]
}

// --- Live sklearn 1.5.2 oracle: IncrementalPCA(n_components=3, batch_size=6) ---
#[allow(
    clippy::excessive_precision,
    reason = "live sklearn 1.5.2 oracle (R-CHAR-3)"
)]
const SK_COMPONENTS: [[f64; 5]; 3] = [
    [
        -0.2719106452,
        0.1205454777,
        -0.3913343135,
        -0.4298222842,
        0.7573926643,
    ],
    [
        0.3127223204,
        0.301967885,
        0.0933154904,
        0.7261018314,
        0.5244887953,
    ],
    [
        -0.4342979184,
        0.850086047,
        0.2090483171,
        -0.0432758243,
        -0.2077619671,
    ],
];
#[allow(
    clippy::excessive_precision,
    reason = "live sklearn 1.5.2 oracle (R-CHAR-3)"
)]
const SK_SINGULAR_VALUES: [f64; 3] = [5.193408652, 4.0762856276, 3.7613633675];
#[allow(
    clippy::excessive_precision,
    reason = "live sklearn 1.5.2 oracle (R-CHAR-3)"
)]
const SK_EXPLAINED_VARIANCE: [f64; 3] = [1.4195522856, 0.8745318167, 0.7446239149];
#[allow(
    clippy::excessive_precision,
    reason = "live sklearn 1.5.2 oracle (R-CHAR-3)"
)]
const SK_EXPLAINED_VARIANCE_RATIO: [f64; 3] = [0.3536833204, 0.2178907532, 0.1855240285];
#[allow(
    clippy::excessive_precision,
    reason = "live sklearn 1.5.2 oracle (R-CHAR-3)"
)]
const SK_TRANSFORM_ROW0: [f64; 3] = [-0.9997372, 1.55217634, -0.4613732];
// Sign-invariant green-guard oracles (exact).
#[allow(
    clippy::excessive_precision,
    reason = "live sklearn 1.5.2 oracle (R-CHAR-3)"
)]
const SK_MEAN: [f64; 5] = [
    0.011868855,
    0.1518941871,
    0.066237159,
    -0.2871161834,
    -0.4621166047,
];
#[allow(
    clippy::excessive_precision,
    reason = "live sklearn 1.5.2 oracle (R-CHAR-3)"
)]
const SK_VAR: [f64; 5] = [
    0.6102427073,
    0.6821506344,
    0.5961971503,
    0.8008722733,
    1.12348117,
];

// ===========================================================================
// DIV-A (#2386): multi-batch components_ (the incremental merge / SVD).
// ===========================================================================

/// Divergence: `IncrementalPCA(n_components=3, batch_size=6).fit(X)` on the 20x5
/// fixture (four batches). sklearn applies `svd_flip(u_based_decision=False)`
/// (`sklearn/decomposition/_incremental_pca.py:357`) so each component row's
/// max-abs entry is positive — ferrolearn matches that sign convention, so a
/// sign-aware compare is valid here. The MAGNITUDES diverge: sklearn row 2 =
/// `[-0.4342979, 0.8500860, 0.2090483, -0.0432758, -0.2077620]`, ferrolearn =
/// `[-0.4320390, 0.8487626, 0.2228008, -0.0485329, -0.2023579]` (max-abs diff
/// ~1.4e-2, ~1e4x the R-DEV-1 ~1e-6 budget). The running mean_/var_ are exact
/// (green guards), so the divergence is in the hand-rolled `thin_svd`
/// (`incremental_pca.rs:720`) of the recursively-stacked merge matrix.
/// Tracking: #2386
#[test]
fn divergence_multibatch_components_n3() {
    let f = IncrementalPCA::<f64>::new(3)
        .with_batch_size(6)
        .fit(&fixture(), &())
        .expect("fit");
    let comp = f.components();
    for k in 0..3 {
        for j in 0..5 {
            let diff = (comp[[k, j]] - SK_COMPONENTS[k][j]).abs();
            assert!(
                diff < 1e-6,
                "components[{k}][{j}] = {} but sklearn = {} (diff {diff})",
                comp[[k, j]],
                SK_COMPONENTS[k][j]
            );
        }
    }
}

// ===========================================================================
// DIV-B (#2386): multi-batch singular_values_.
// ===========================================================================

/// Divergence: `singular_values_` from the multi-batch merge. sklearn (line
/// `_incremental_pca.py:363`) = `[5.193408652, 4.0762856276, 3.7613633675]`;
/// ferrolearn diverges by up to ~1.0e-3 (3rd value), beyond R-DEV-1 ~1e-6.
/// Sign-invariant, so isolates the SVD magnitude error from svd_flip.
/// Tracking: #2386
#[test]
fn divergence_multibatch_singular_values_n3() {
    let f = IncrementalPCA::<f64>::new(3)
        .with_batch_size(6)
        .fit(&fixture(), &())
        .expect("fit");
    let sv = f.singular_values();
    for k in 0..3 {
        let diff = (sv[k] - SK_SINGULAR_VALUES[k]).abs();
        assert!(
            diff < 1e-6,
            "singular_values_[{k}] = {} but sklearn = {} (diff {diff})",
            sv[k],
            SK_SINGULAR_VALUES[k]
        );
    }
}

// ===========================================================================
// DIV-C (#2386): multi-batch explained_variance_.
// ===========================================================================

/// Divergence: `explained_variance_ = S**2/(n_total-1)`
/// (`_incremental_pca.py:358`). sklearn = `[1.4195522856, 0.8745318167,
/// 0.7446239149]`; ferrolearn diverges by up to ~4.0e-4, beyond R-DEV-1 ~1e-6.
/// Tracking: #2386
#[test]
fn divergence_multibatch_explained_variance_n3() {
    let f = IncrementalPCA::<f64>::new(3)
        .with_batch_size(6)
        .fit(&fixture(), &())
        .expect("fit");
    let ev = f.explained_variance();
    for k in 0..3 {
        let diff = (ev[k] - SK_EXPLAINED_VARIANCE[k]).abs();
        assert!(
            diff < 1e-6,
            "explained_variance_[{k}] = {} but sklearn = {} (diff {diff})",
            ev[k],
            SK_EXPLAINED_VARIANCE[k]
        );
    }
}

// ===========================================================================
// DIV-D (#2386): multi-batch explained_variance_ratio_.
// ===========================================================================

/// Divergence: `explained_variance_ratio_ = S**2/sum(col_var*n_total)`
/// (`_incremental_pca.py:359`). sklearn = `[0.3536833204, 0.2178907532,
/// 0.1855240285]`; ferrolearn diverges by up to ~1.0e-4, beyond R-DEV-1 ~1e-6.
/// (The denominator/`var_` is exact — see green guard — so the divergence is
/// inherited from the S^2 numerator's SVD error.)
/// Tracking: #2386
#[test]
fn divergence_multibatch_explained_variance_ratio_n3() {
    let f = IncrementalPCA::<f64>::new(3)
        .with_batch_size(6)
        .fit(&fixture(), &())
        .expect("fit");
    let evr = f.explained_variance_ratio();
    for k in 0..3 {
        let diff = (evr[k] - SK_EXPLAINED_VARIANCE_RATIO[k]).abs();
        assert!(
            diff < 1e-6,
            "explained_variance_ratio_[{k}] = {} but sklearn = {} (diff {diff})",
            evr[k],
            SK_EXPLAINED_VARIANCE_RATIO[k]
        );
    }
}

// ===========================================================================
// DIV-E (#2386): multi-batch transform (inherits the components_ error).
// ===========================================================================

/// Divergence: `transform(X) = (X - mean_) @ components_.T`
/// (`_incremental_pca.py` `_BasePCA.transform`). Because `components_` diverges
/// (DIV-A), the projection does too. sklearn `transform(X)[0] = [-0.9997372,
/// 1.55217634, -0.4613732]`; ferrolearn diverges by up to ~2.9e-3, beyond
/// R-DEV-1 ~1e-6.
/// Tracking: #2386
#[test]
fn divergence_multibatch_transform_row0_n3() {
    let f = IncrementalPCA::<f64>::new(3)
        .with_batch_size(6)
        .fit(&fixture(), &())
        .expect("fit");
    let t = f.transform(&fixture()).expect("transform");
    for k in 0..3 {
        let diff = (t[[0, k]] - SK_TRANSFORM_ROW0[k]).abs();
        assert!(
            diff < 1e-6,
            "transform[0][{k}] = {} but sklearn = {} (diff {diff})",
            t[[0, k]],
            SK_TRANSFORM_ROW0[k]
        );
    }
}

// ===========================================================================
// GREEN GUARDS — must PASS: isolate the divergence to the SVD, not mean/var.
// ===========================================================================

/// GREEN: running `mean_` matches sklearn exactly across the four batches.
#[test]
fn green_multibatch_mean_matches() {
    let f = IncrementalPCA::<f64>::new(3)
        .with_batch_size(6)
        .fit(&fixture(), &())
        .expect("fit");
    let mean = f.mean();
    for j in 0..5 {
        assert!(
            (mean[j] - SK_MEAN[j]).abs() < 1e-9,
            "mean_[{j}] = {} but sklearn = {}",
            mean[j],
            SK_MEAN[j]
        );
    }
}

/// GREEN: running `var_` (Youngs-Cramer) matches sklearn exactly across batches.
#[test]
fn green_multibatch_var_matches() {
    let f = IncrementalPCA::<f64>::new(3)
        .with_batch_size(6)
        .fit(&fixture(), &())
        .expect("fit");
    let var = f.var();
    for j in 0..5 {
        assert!(
            (var[j] - SK_VAR[j]).abs() < 1e-9,
            "var_[{j}] = {} but sklearn = {}",
            var[j],
            SK_VAR[j]
        );
    }
}

/// GREEN: `n_samples_seen_` accumulates to 20 (last batch is a partial 2 rows).
#[test]
fn green_multibatch_n_samples_seen() {
    let f = IncrementalPCA::<f64>::new(3)
        .with_batch_size(6)
        .fit(&fixture(), &())
        .expect("fit");
    assert_eq!(f.n_samples_seen(), 20);
}
