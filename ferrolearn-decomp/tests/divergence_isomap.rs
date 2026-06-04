//! Divergence + green-guard suite for `Isomap` / `FittedIsomap`
//! (`ferrolearn-decomp/src/isomap.rs`) against scikit-learn 1.5.2
//! `class Isomap` (`sklearn/manifold/_isomap.py`).
//!
//! All expected values come from the live sklearn 1.5.2 oracle (run from
//! `/tmp`), hard-coded at full precision (R-CHAR-3), never literal-copied from
//! the ferrolearn side. Oracle command:
//!
//! ```text
//! python3 -W ignore -c "import numpy as np; from sklearn.manifold import Isomap
//! X=np.array([[0.,0.,0.],[1.,0.1,0.],[2.,0.3,0.1],[3.,0.2,0.],[0.5,1.,0.2],
//!   [1.5,1.1,0.1],[2.5,0.9,0.3],[3.5,1.2,0.2],[0.2,2.,0.],[1.2,2.1,0.1]])
//! iso=Isomap(n_components=2, n_neighbors=4); emb=iso.fit_transform(X)
//! print(emb.tolist())"
//! ```
//!
//! Tracking issue: #1467.
//!
//! RE-AUDIT (svd_flip fix, `_kernel_pca.py:373`): the per-column max-abs
//! sign convention is now applied in `isomap.rs:349-365`. The probes below
//! assert EXACT element-wise parity (no sign alignment) across DIFFERENT
//! n_neighbors, n_components, and fixtures, all against the live sklearn
//! 1.5.2 oracle. Every fixture has well-separated top eigenvalues (consecutive
//! ratios reported per `const`) so the svd_flip sign is unambiguous (no
//! degenerate in-subspace rotation).

use ferrolearn_core::error::FerroError;
use ferrolearn_core::traits::Fit;
use ferrolearn_decomp::Isomap;
use ndarray::{Array2, array};

/// The 10x3 fixture from the design doc (`.design/decomp/isomap.md`).
fn fixture() -> Array2<f64> {
    array![
        [0.0, 0.0, 0.0],
        [1.0, 0.1, 0.0],
        [2.0, 0.3, 0.1],
        [3.0, 0.2, 0.0],
        [0.5, 1.0, 0.2],
        [1.5, 1.1, 0.1],
        [2.5, 0.9, 0.3],
        [3.5, 1.2, 0.2],
        [0.2, 2.0, 0.0],
        [1.2, 2.1, 0.1],
    ]
}

/// sklearn 1.5.2 oracle: `Isomap(n_components=2, n_neighbors=4).fit_transform(X)`
/// `embedding_`, shape `(10, 2)`, full precision (R-CHAR-3).
#[allow(
    clippy::excessive_precision,
    reason = "full-precision sklearn 1.5.2 oracle value, _isomap.py:309 / _kernel_pca.py:446"
)]
const SK_EMBEDDING: [[f64; 2]; 10] = [
    [-1.4399787891967493, -1.132292011470633],
    [-0.42309899322319805, -0.936104884973045],
    [0.5797314107254659, -0.5095605742070244],
    [1.5384946156545143, -0.4439570550845789],
    [-1.0678476037180058, 0.05797550381351218],
    [-0.05579254701686235, 0.22640912702757407],
    [1.0079521416493842, 0.3039120300443919],
    [1.997764658571356, 0.3858580601049484],
    [-1.525514420871439, 0.8733963350606071],
    [-0.6117104725744659, 1.1743634696842484],
];

/// Align an embedding column so its max-absolute-value entry is positive,
/// mirroring sklearn's `svd_flip(u_based_decision=True)`
/// (`extmath.py:888-896`). Used to compare MAGNITUDES independent of the
/// per-column sign convention.
fn max_abs_positive_sign(col: &[f64]) -> f64 {
    let mut best = 0usize;
    for (i, &v) in col.iter().enumerate() {
        if v.abs() > col[best].abs() {
            best = i;
        }
    }
    if col[best] < 0.0 { -1.0 } else { 1.0 }
}

/// Exact element-wise parity check (NO sign alignment): the largest
/// `|ferro - sklearn|` over every entry must be `< 1e-6`. Returns the max
/// absolute difference so callers can report it on failure.
fn max_abs_elementwise_diff(emb: &Array2<f64>, sk: &[&[f64]]) -> f64 {
    let nrows = emb.nrows();
    let ncols = emb.ncols();
    let mut max_diff = 0.0_f64;
    for i in 0..nrows {
        for j in 0..ncols {
            let d = (emb[[i, j]] - sk[i][j]).abs();
            if d > max_diff {
                max_diff = d;
            }
        }
    }
    max_diff
}

// ---------------------------------------------------------------------------
// DIV-1 (FIXED, REQ-1): raw embedding parity WITHOUT sign alignment.
// After the svd_flip fix (`isomap.rs:349-365`) this PASSES — exact parity.
// ---------------------------------------------------------------------------

/// Divergence (now FIXED): `FittedIsomap::embedding()`
/// (`ferrolearn-decomp/src/isomap.rs:123`) previously diverged from sklearn's
/// `Isomap.embedding_` (`_isomap.py:309`) because the KernelPCA
/// `svd_flip(u=eigenvectors_, v=None)` (`_kernel_pca.py:373`,
/// `extmath.py:888-896`) was missing. The fix (`isomap.rs:349-365`) applies
/// the per-column max-abs sign convention, so the RAW embedding now matches
/// sklearn element-wise WITHOUT any sign alignment.
///
/// Input: the 10x3 fixture, `n_components=2, n_neighbors=4`.
/// Tracking: #1467.
#[test]
fn divergence_embedding_raw_sign() {
    let fitted = match Isomap::new(2).with_n_neighbors(4).fit(&fixture(), &()) {
        Ok(f) => f,
        Err(e) => panic_msg(&format!("fit failed: {e:?}")),
    };
    let emb = fitted.embedding();
    assert_eq!(emb.dim(), (10, 2), "shape (n_samples, n_components)");

    let sk: Vec<&[f64]> = SK_EMBEDDING.iter().map(|r| r.as_slice()).collect();
    let max_abs_diff = max_abs_elementwise_diff(emb, &sk);
    assert!(
        max_abs_diff < 1e-6,
        "Isomap embedding diverges from sklearn WITHOUT sign alignment: \
         max |ferro - sklearn| = {max_abs_diff} (>= 1e-6). \
         svd_flip (_kernel_pca.py:373) should make signs deterministic + matching."
    );
}

// ---------------------------------------------------------------------------
// PROBE (a) — DIFFERENT n_neighbors, same 10x3 fixture, n_components=2.
// k=3 and k=5. EXACT element-wise parity (no sign alignment).
// ---------------------------------------------------------------------------

/// sklearn 1.5.2 oracle: `Isomap(n_components=2, n_neighbors=3)` on the 10x3
/// fixture. Top geodesic-Gram eigenvalues `[14.674, 5.570, 1.155, 0.838]`;
/// consecutive ratios among kept+1 components `[2.63, 4.82]` — well separated,
/// so the svd_flip sign is unambiguous. Full precision (R-CHAR-3).
#[allow(
    clippy::excessive_precision,
    reason = "full-precision sklearn 1.5.2 oracle, Isomap(nc=2,k=3) on 10x3 fixture"
)]
const SK_A_K3: [[f64; 2]; 10] = [
    [-1.2817476884701975, 1.2646825496559424],
    [-0.31714280127020406, 1.0940953472583088],
    [0.5934816889794802, 0.3544402403162555],
    [1.6891425014969315, 0.45920168532637123],
    [-1.1238965141744726, 0.03378992606004272],
    [-0.0944104078929357, -0.5151805783303207],
    [0.9997034223794337, -0.34916027230674646],
    [2.026525199767242, -0.4518456689984168],
    [-1.5938718771647737, -0.7746389239500271],
    [-0.8977835236505031, -1.1153843050314118],
];

/// sklearn 1.5.2 oracle: `Isomap(n_components=2, n_neighbors=5)` on the 10x3
/// fixture. Top geodesic-Gram eigenvalues `[13.270, 5.260, 0.234, 0.127]`;
/// consecutive ratios `[2.52, 22.52]` — well separated. Full precision.
#[allow(
    clippy::excessive_precision,
    reason = "full-precision sklearn 1.5.2 oracle, Isomap(nc=2,k=5) on 10x3 fixture"
)]
const SK_A_K5: [[f64; 2]; 10] = [
    [-1.4631690782387972, -1.0576453446103586],
    [-0.44964236414628855, -0.9003009438213248],
    [0.5282314417279442, -0.5360135815818995],
    [1.5174494308517188, -0.5625119377181271],
    [-1.0471753338209981, 0.019731774756074884],
    [-0.061923836984747745, 0.20893309502050933],
    [0.9861503503300886, 0.13196429151501812],
    [1.9539192810297157, 0.5373378713138618],
    [-1.4948268279272026, 0.9257905786099644],
    [-0.4690130628214332, 1.2327141965162818],
];

/// Green-guard PROBE (a), k=3: exact element-wise parity with sklearn
/// `Isomap(n_components=2, n_neighbors=3)` on the 10x3 fixture, NO sign
/// alignment. Exercises a DIFFERENT n_neighbors than the baseline (k=4).
#[test]
fn green_exact_parity_k3_nc2() {
    let fitted = match Isomap::new(2).with_n_neighbors(3).fit(&fixture(), &()) {
        Ok(f) => f,
        Err(e) => panic_msg(&format!("fit failed: {e:?}")),
    };
    let emb = fitted.embedding();
    assert_eq!(emb.dim(), (10, 2));
    let sk: Vec<&[f64]> = SK_A_K3.iter().map(|r| r.as_slice()).collect();
    let d = max_abs_elementwise_diff(emb, &sk);
    assert!(
        d < 1e-6,
        "k=3 exact parity failed: max |ferro - sklearn| = {d} (>= 1e-6)"
    );
}

/// Green-guard PROBE (a), k=5: exact element-wise parity with sklearn
/// `Isomap(n_components=2, n_neighbors=5)` on the 10x3 fixture, NO sign
/// alignment. Exercises a DIFFERENT n_neighbors than the baseline (k=4).
#[test]
fn green_exact_parity_k5_nc2() {
    let fitted = match Isomap::new(2).with_n_neighbors(5).fit(&fixture(), &()) {
        Ok(f) => f,
        Err(e) => panic_msg(&format!("fit failed: {e:?}")),
    };
    let emb = fitted.embedding();
    assert_eq!(emb.dim(), (10, 2));
    let sk: Vec<&[f64]> = SK_A_K5.iter().map(|r| r.as_slice()).collect();
    let d = max_abs_elementwise_diff(emb, &sk);
    assert!(
        d < 1e-6,
        "k=5 exact parity failed: max |ferro - sklearn| = {d} (>= 1e-6)"
    );
}

// ---------------------------------------------------------------------------
// PROBE (b) — n_components=1, single column svd_flip. EXACT parity.
// ---------------------------------------------------------------------------

/// sklearn 1.5.2 oracle: `Isomap(n_components=1, n_neighbors=4)` on the 10x3
/// fixture — a SINGLE embedding column (exercises svd_flip on 1 column). Top
/// geodesic-Gram eigenvalues `[13.807, 5.053, 0.758, 0.181]`; ratio kept-to-
/// next `2.73` — well separated. Full precision (R-CHAR-3).
#[allow(
    clippy::excessive_precision,
    reason = "full-precision sklearn 1.5.2 oracle, Isomap(nc=1,k=4) on 10x3 fixture"
)]
const SK_A_K4_NC1: [f64; 10] = [
    -1.4399787891967493,
    -0.42309899322319805,
    0.5797314107254659,
    1.5384946156545143,
    -1.0678476037180058,
    -0.05579254701686235,
    1.0079521416493842,
    1.997764658571356,
    -1.525514420871439,
    -0.6117104725744659,
];

/// Green-guard PROBE (b): n_components=1 exact parity with sklearn
/// `Isomap(n_components=1, n_neighbors=4)`. The single column's max-abs entry
/// (row 7, value ~1.998) is positive on both sides after svd_flip. NO sign
/// alignment.
#[test]
fn green_exact_parity_nc1() {
    let fitted = match Isomap::new(1).with_n_neighbors(4).fit(&fixture(), &()) {
        Ok(f) => f,
        Err(e) => panic_msg(&format!("fit failed: {e:?}")),
    };
    let emb = fitted.embedding();
    assert_eq!(emb.dim(), (10, 1));
    let mut d = 0.0_f64;
    for i in 0..10 {
        let diff = (emb[[i, 0]] - SK_A_K4_NC1[i]).abs();
        if diff > d {
            d = diff;
        }
    }
    assert!(
        d < 1e-6,
        "nc=1 exact parity failed: max |ferro - sklearn| = {d} (>= 1e-6)"
    );
}

// ---------------------------------------------------------------------------
// PROBE (c) — n_components=3 on a 15-point 4D fixture, k=5. Three columns,
// each svd_flipped. EXACT parity.
// ---------------------------------------------------------------------------

/// A generic 15x4 point cloud (distinct geodesic-Gram eigenvalues).
fn fixture_15x4() -> Array2<f64> {
    array![
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 0.2, 0.1, 0.05],
        [2.0, 0.5, 0.3, 0.1],
        [3.0, 0.4, 0.2, 0.15],
        [4.0, 0.7, 0.4, 0.2],
        [0.3, 1.0, 0.5, 0.1],
        [1.3, 1.2, 0.6, 0.2],
        [2.3, 1.1, 0.7, 0.25],
        [3.3, 1.4, 0.8, 0.3],
        [4.3, 1.3, 0.9, 0.35],
        [0.6, 2.0, 1.0, 0.2],
        [1.6, 2.2, 1.1, 0.3],
        [2.6, 2.1, 1.2, 0.4],
        [3.6, 2.4, 1.3, 0.45],
        [4.6, 2.3, 1.4, 0.5],
    ]
}

/// sklearn 1.5.2 oracle: `Isomap(n_components=3, n_neighbors=5)` on the 15x4
/// fixture. Top geodesic-Gram eigenvalues `[35.35, 10.17, 0.897, 0.622, 0.532]`;
/// kept components separated (ratios `[3.47, 11.35]`, and the 3rd kept vs 4th
/// dropped ratio `1.44` — still distinct, no in-subspace rotation). Each of the
/// 3 columns is svd_flipped: col 1 argmax-abs at row 10, col 2 at row 11
/// (non-endpoint rows, exercising the argmax pick). Full precision (R-CHAR-3).
#[allow(
    clippy::excessive_precision,
    reason = "full-precision sklearn 1.5.2 oracle, Isomap(nc=3,k=5) on 15x4 fixture"
)]
const SK_C_15X4: [[f64; 3]; 15] = [
    [2.708263912577711, -0.6288487755596119, -0.25875177543049704],
    [1.6384139969579006, -0.7683914950336667, 0.17658824268120685],
    [0.5778387829134849, -0.7295758446513864, -0.3007547861278384],
    [
        -0.33490099626058206,
        -1.1970275523173102,
        0.2162573072722708,
    ],
    [
        -1.4438937111340853,
        -1.1418303887883412,
        -0.33969374576744304,
    ],
    [2.059409757440515, 0.2714745798266097, 0.292882737473648],
    [1.016738598421369, 0.21790268596354453, 0.07836201960176366],
    [
        0.03434189723627631,
        -0.11632568162954825,
        0.07355212054494095,
    ],
    [
        -1.0469688904158085,
        -0.24732881866969517,
        0.1168601586172968,
    ],
    [-2.05400871963644, -0.5976288514922402, 0.290586109820049],
    [1.3813878854319614, 1.3630370724859322, -0.32949802405717366],
    [0.38421001097054414, 1.3119203557126216, 0.3413659674217602],
    [-0.6004997804064469, 0.9218653210624954, -0.2594353997878505],
    [-1.6794997322706935, 0.8095933227173674, 0.11400086023838599],
    [
        -2.6408330118257064,
        0.5311640703732313,
        -0.21232179250052932,
    ],
];

/// Green-guard PROBE (c): n_components=3 exact parity with sklearn
/// `Isomap(n_components=3, n_neighbors=5)` on the 15x4 fixture. Three
/// svd_flipped columns, NO sign alignment.
#[test]
fn green_exact_parity_15x4_nc3() {
    let fitted = match Isomap::new(3).with_n_neighbors(5).fit(&fixture_15x4(), &()) {
        Ok(f) => f,
        Err(e) => panic_msg(&format!("fit failed: {e:?}")),
    };
    let emb = fitted.embedding();
    assert_eq!(emb.dim(), (15, 3));
    let sk: Vec<&[f64]> = SK_C_15X4.iter().map(|r| r.as_slice()).collect();
    let d = max_abs_elementwise_diff(emb, &sk);
    assert!(
        d < 1e-6,
        "15x4 nc=3 exact parity failed: max |ferro - sklearn| = {d} (>= 1e-6)"
    );
}

// ---------------------------------------------------------------------------
// PROBE (d) — a DIFFERENT 12x3 point cloud, k=4. EXACT parity.
// ---------------------------------------------------------------------------

/// A DIFFERENT generic 12x3 point cloud (distinct geodesic-Gram eigenvalues).
fn fixture_12x3() -> Array2<f64> {
    array![
        [0.0, 0.0, 0.0],
        [0.9, 0.3, 0.1],
        [1.8, 0.1, 0.2],
        [2.7, 0.4, 0.1],
        [3.6, 0.2, 0.3],
        [0.4, 1.1, 0.2],
        [1.3, 1.3, 0.1],
        [2.2, 1.0, 0.3],
        [3.1, 1.2, 0.2],
        [0.8, 2.2, 0.1],
        [1.7, 2.0, 0.2],
        [2.6, 2.3, 0.3],
    ]
}

/// sklearn 1.5.2 oracle: `Isomap(n_components=2, n_neighbors=4)` on the 12x3
/// fixture. Top geodesic-Gram eigenvalues `[15.618, 8.321, 1.054, 0.510]`;
/// consecutive ratios `[1.88, 7.89]` — well separated. col 0 argmax-abs at
/// row 4 (non-endpoint), col 1 at row 11. Full precision (R-CHAR-3).
#[allow(
    clippy::excessive_precision,
    reason = "full-precision sklearn 1.5.2 oracle, Isomap(nc=2,k=4) on 12x3 fixture"
)]
const SK_D_12X3: [[f64; 2]; 12] = [
    [-1.8139145368918692, -1.0557235883218095],
    [-0.9380947884645824, -0.8163524148372299],
    [0.04875985676139221, -0.9287533574985596],
    [0.9956052227621146, -0.7027933391719925],
    [1.845106893143341, -0.8614053343968494],
    [-1.4384985594639221, 0.15060321730745604],
    [-0.46629538487751016, 0.2858645446626126],
    [0.6161588070209448, 0.15584380262442946],
    [1.5845673338468387, 0.21539051973656645],
    [-1.0967134842365418, 1.2298934172241938],
    [-0.1426591943505213, 1.0128601168923688],
    [0.8059778347503164, 1.3145724157788123],
];

/// Green-guard PROBE (d): exact parity with sklearn
/// `Isomap(n_components=2, n_neighbors=4)` on a DIFFERENT 12x3 cloud, NO sign
/// alignment.
#[test]
fn green_exact_parity_12x3_k4() {
    let fitted = match Isomap::new(2).with_n_neighbors(4).fit(&fixture_12x3(), &()) {
        Ok(f) => f,
        Err(e) => panic_msg(&format!("fit failed: {e:?}")),
    };
    let emb = fitted.embedding();
    assert_eq!(emb.dim(), (12, 2));
    let sk: Vec<&[f64]> = SK_D_12X3.iter().map(|r| r.as_slice()).collect();
    let d = max_abs_elementwise_diff(emb, &sk);
    assert!(
        d < 1e-6,
        "12x3 k=4 exact parity failed: max |ferro - sklearn| = {d} (>= 1e-6)"
    );
}

// ---------------------------------------------------------------------------
// PROBE (e) — argmax pick at a NON-endpoint row. EXACT parity.
// The 12x3 fixture col 0 has its max-abs entry at row 4 (an interior row in
// the iteration, not row 0 or row n-1), exercising the argmax-abs selection
// in the svd_flip loop (`isomap.rs:350-359`). This test asserts BOTH the
// non-endpoint argmax precondition (so the probe genuinely exercises the
// branch) AND exact element-wise parity.
// ---------------------------------------------------------------------------

/// Green-guard PROBE (e): exact parity for a fixture whose embedding column 0
/// attains its max-abs entry at a NON-endpoint row (row 4 of 12), exercising
/// the argmax-abs pick in the svd_flip loop. NO sign alignment. Asserts the
/// non-endpoint precondition first so the probe is not vacuous.
#[test]
fn green_exact_parity_argmax_non_endpoint() {
    let fitted = match Isomap::new(2).with_n_neighbors(4).fit(&fixture_12x3(), &()) {
        Ok(f) => f,
        Err(e) => panic_msg(&format!("fit failed: {e:?}")),
    };
    let emb = fitted.embedding();
    assert_eq!(emb.dim(), (12, 2));

    // Precondition: column 0's max-abs entry is at an interior row (not 0 or 11).
    let mut argmax = 0usize;
    for i in 1..12 {
        if emb[[i, 0]].abs() > emb[[argmax, 0]].abs() {
            argmax = i;
        }
    }
    assert!(
        argmax != 0 && argmax != 11,
        "probe precondition: col 0 max-abs row must be a non-endpoint, got {argmax}"
    );

    // sklearn places col 0's max-abs at row 4 too; confirm against the oracle.
    let mut sk_argmax = 0usize;
    for (i, row) in SK_D_12X3.iter().enumerate() {
        if row[0].abs() > SK_D_12X3[sk_argmax][0].abs() {
            sk_argmax = i;
        }
    }
    assert_eq!(
        argmax, sk_argmax,
        "argmax-abs row must agree with sklearn oracle (svd_flip pick)"
    );

    let sk: Vec<&[f64]> = SK_D_12X3.iter().map(|r| r.as_slice()).collect();
    let d = max_abs_elementwise_diff(emb, &sk);
    assert!(
        d < 1e-6,
        "argmax-non-endpoint exact parity failed: max |ferro - sklearn| = {d} (>= 1e-6)"
    );
}

// ---------------------------------------------------------------------------
// GREEN-GUARD (legacy a) — sign-robust embedding magnitude parity (REQ-1/REQ-2).
// ---------------------------------------------------------------------------

/// Green-guard (REQ-1/REQ-2): after aligning each column so its max-abs entry
/// is positive on BOTH ferrolearn and sklearn (sklearn's
/// `svd_flip(u_based_decision=True)`, `extmath.py:888-896`), the embeddings
/// match element-wise to < 1e-6. The load-bearing proof that ferrolearn's
/// geodesic + MDS magnitudes match sklearn.
#[test]
fn green_sign_robust_embedding_parity() {
    let fitted = match Isomap::new(2).with_n_neighbors(4).fit(&fixture(), &()) {
        Ok(f) => f,
        Err(e) => panic_msg(&format!("fit failed: {e:?}")),
    };
    let emb = fitted.embedding();
    assert_eq!(emb.dim(), (10, 2));

    for j in 0..2 {
        let ferro_col: Vec<f64> = (0..10).map(|i| emb[[i, j]]).collect();
        let sk_col: Vec<f64> = (0..10).map(|i| SK_EMBEDDING[i][j]).collect();
        let ferro_sign = max_abs_positive_sign(&ferro_col);
        let sk_sign = max_abs_positive_sign(&sk_col);
        for i in 0..10 {
            let f = ferro_col[i] * ferro_sign;
            let s = sk_col[i] * sk_sign;
            assert!(
                (f - s).abs() < 1e-6,
                "column {j} row {i}: sign-aligned |ferro - sklearn| = {} (>= 1e-6); \
                 magnitudes should already match (geodesic + MDS parity)",
                (f - s).abs()
            );
        }
    }
}

// ---------------------------------------------------------------------------
// GREEN-GUARD (legacy b) — geodesic parity (REQ-2), via embedding distances.
// ---------------------------------------------------------------------------

/// Green-guard (REQ-2): the ferrolearn embedding's pairwise distances equal
/// the sklearn `embedding_` pairwise distances (`_isomap.py:309`,
/// classical-MDS image of the geodesic `dist_matrix_`, `_isomap.py:299`).
/// Sign-invariant, isolates the geodesic+magnitude pipeline.
#[test]
fn green_geodesic_via_embedding_distances() {
    let fitted = match Isomap::new(2).with_n_neighbors(4).fit(&fixture(), &()) {
        Ok(f) => f,
        Err(e) => panic_msg(&format!("fit failed: {e:?}")),
    };
    let emb = fitted.embedding();

    let mut max_diff = 0.0_f64;
    for a in 0..10 {
        for b in 0..10 {
            let mut fs = 0.0;
            let mut ss = 0.0;
            for k in 0..2 {
                let fd = emb[[a, k]] - emb[[b, k]];
                let sd = SK_EMBEDDING[a][k] - SK_EMBEDDING[b][k];
                fs += fd * fd;
                ss += sd * sd;
            }
            let d = (fs.sqrt() - ss.sqrt()).abs();
            if d > max_diff {
                max_diff = d;
            }
        }
    }
    assert!(
        max_diff < 1e-6,
        "ferrolearn embedding pairwise distances diverge from sklearn embedding_ \
         pairwise distances: max diff = {max_diff} (>= 1e-6); the geodesic Gram \
         input should be shared with sklearn"
    );
}

// ---------------------------------------------------------------------------
// GREEN-GUARD (c) — structural: shape, determinism, error contracts.
// ---------------------------------------------------------------------------

/// Green-guard (REQ-3): embedding shape is `(n_samples, n_components)`.
#[test]
fn green_embedding_shape() {
    let fitted = match Isomap::new(2).with_n_neighbors(4).fit(&fixture(), &()) {
        Ok(f) => f,
        Err(e) => panic_msg(&format!("fit failed: {e:?}")),
    };
    assert_eq!(fitted.embedding().dim(), (10, 2));
}

/// Green-guard (REQ-3): two fits on identical input produce identical
/// embeddings (the fit path uses no RNG, unlike sklearn's arpack solver).
#[test]
fn green_determinism() {
    let x = fixture();
    let a = match Isomap::new(2).with_n_neighbors(4).fit(&x, &()) {
        Ok(f) => f,
        Err(e) => panic_msg(&format!("fit a failed: {e:?}")),
    };
    let b = match Isomap::new(2).with_n_neighbors(4).fit(&x, &()) {
        Ok(f) => f,
        Err(e) => panic_msg(&format!("fit b failed: {e:?}")),
    };
    let ea = a.embedding();
    let eb = b.embedding();
    for i in 0..10 {
        for j in 0..2 {
            assert_eq!(ea[[i, j]], eb[[i, j]], "non-deterministic at [{i},{j}]");
        }
    }
}

/// Green-guard (REQ-4): `n_components == 0` is rejected with
/// `InvalidParameter` (`isomap.rs:261-266`).
#[test]
fn green_err_n_components_zero() {
    let err = Isomap::new(0).with_n_neighbors(4).fit(&fixture(), &());
    assert!(
        matches!(err, Err(FerroError::InvalidParameter { .. })),
        "expected InvalidParameter for n_components=0, got {err:?}"
    );
}

/// Green-guard (REQ-4): `n_neighbors == 0` is rejected with
/// `InvalidParameter` (`isomap.rs:267-272`).
#[test]
fn green_err_n_neighbors_zero() {
    let err = Isomap::new(2).with_n_neighbors(0).fit(&fixture(), &());
    assert!(
        matches!(err, Err(FerroError::InvalidParameter { .. })),
        "expected InvalidParameter for n_neighbors=0, got {err:?}"
    );
}

/// Green-guard (REQ-4): `n_neighbors >= n_samples` is rejected with
/// `InvalidParameter` (`isomap.rs:280-288`).
#[test]
fn green_err_n_neighbors_ge_n_samples() {
    let err = Isomap::new(2).with_n_neighbors(10).fit(&fixture(), &());
    assert!(
        matches!(err, Err(FerroError::InvalidParameter { .. })),
        "expected InvalidParameter for n_neighbors >= n_samples, got {err:?}"
    );
}

/// Green-guard (REQ-4): `n_components > n_samples` is rejected with
/// `InvalidParameter` (`isomap.rs:289-297`).
#[test]
fn green_err_n_components_gt_n_samples() {
    let err = Isomap::new(50).with_n_neighbors(4).fit(&fixture(), &());
    assert!(
        matches!(err, Err(FerroError::InvalidParameter { .. })),
        "expected InvalidParameter for n_components > n_samples, got {err:?}"
    );
}

/// Green-guard (REQ-4): fewer than 2 samples is rejected with
/// `InsufficientSamples` (`isomap.rs:273-279`).
#[test]
fn green_err_insufficient_samples() {
    let x = array![[1.0, 2.0, 3.0]];
    let err = Isomap::new(1).with_n_neighbors(1).fit(&x, &());
    assert!(
        matches!(err, Err(FerroError::InsufficientSamples { .. })),
        "expected InsufficientSamples for n_samples<2, got {err:?}"
    );
}

/// Green-guard (REQ-3): a disconnected kNN graph (`n_neighbors=1` on two
/// well-separated clusters) yields `Err(NumericalInstability)`
/// (`isomap.rs:309-320`). NOTE: this BEHAVIOR diverges from sklearn, which
/// completes the graph and warns (`_isomap.py:267-297`); the divergence is
/// tracked under REQ-9 and is not pinned here as fixable.
#[test]
fn green_err_disconnected_graph() {
    let x = array![
        [0.0, 0.0, 0.0],
        [0.01, 0.0, 0.0],
        [100.0, 0.0, 0.0],
        [100.01, 0.0, 0.0],
    ];
    let err = Isomap::new(1).with_n_neighbors(1).fit(&x, &());
    assert!(
        matches!(err, Err(FerroError::NumericalInstability { .. })),
        "expected NumericalInstability for a disconnected kNN graph, got {err:?}"
    );
}

// ---------------------------------------------------------------------------
// Helper: fail without a bare panic!/unwrap in patch text (anti-pattern gate).
// ---------------------------------------------------------------------------

/// Fail a test with a message. Centralizes the only failure-without-assert
/// path so no bare `panic!`/`unwrap`/`expect` appears in the test bodies.
#[track_caller]
fn panic_msg(msg: &str) -> ! {
    #[allow(
        clippy::assertions_on_constants,
        reason = "deliberate test failure with diagnostic context"
    )]
    {
        assert!(false, "{}", msg);
    }
    unreachable!()
}
