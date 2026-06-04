//! Divergence / green-guard audit for `ferrolearn_decomp::lle`
//! (`LLE` / `FittedLLE` / `compute_weights`) against scikit-learn 1.5.2
//! `sklearn/manifold/_locally_linear.py` (`LocallyLinearEmbedding`,
//! `barycenter_weights`, `null_space`).
//!
//! HISTORY (tracking #1459, blocker #1460): ferrolearn's local-covariance
//! regularization previously used `reg_val = reg * trace / k` (DIVIDED by k),
//! whereas sklearn `barycenter_weights` uses
//!   `trace = np.trace(G); R = reg * trace if trace > 0 else reg`
//! (`sklearn/manifold/_locally_linear.py:72-77`) with NO division by
//! `n_neighbors`. That bug has since been FIXED in
//! `ferrolearn-decomp/src/lle.rs:185-191`:
//!   `let reg_val = if trace > 0.0 { reg * trace } else { reg };`
//! This file RE-AUDITS the fix for faithfulness across many configurations
//! against the LIVE sklearn 1.5.2 oracle.
//!
//! SIGN HANDLING: sklearn `null_space` (`_locally_linear.py:192-196`) applies NO
//! deterministic sign flip, so the per-component sign of each eigenvector is
//! solver-arbitrary (faer vs scipy `eigh`). Every embedding comparison here is
//! SIGN-ROBUST: each column is flipped so its max-abs entry is positive on BOTH
//! ferrolearn and sklearn before the element-wise tolerance check (tol 1e-6).
//!
//! All expected matrices are the LIVE sklearn 1.5.2 oracle output, generated
//! from /tmp with warnings suppressed and hard-coded at FULL f64 precision
//! (R-CHAR-3) — never literal-copied from ferrolearn. Each fixture was chosen so
//! the bottom `n_components` eigenvalues of `M = (I-W)^T(I-W)` are DISTINCT (no
//! degenerate-subspace rotation ambiguity in the sign-robust comparison).
//!
//! The GREEN-GUARDS (shape / determinism / error contracts / column centering)
//! and every parity probe MUST PASS against the current (fixed) code; a FAIL
//! marks a residual divergence (-> blocker).

use ferrolearn_core::traits::Fit;
use ferrolearn_decomp::LLE;
use ndarray::{Array2, array};

/// The 10x3 fixture used by the design doc (`.design/decomp/lle.md`).
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

/// Flip a column in place so that its entry of MAXIMUM ABSOLUTE VALUE is positive.
/// Applied identically to ferrolearn and sklearn columns so the comparison is
/// invariant to the solver-arbitrary per-component sign (REQ-7 carve-out).
fn sign_align_column(col: &mut [f64]) {
    let mut max_abs = 0.0_f64;
    let mut max_val = 0.0_f64;
    for &v in col.iter() {
        if v.abs() > max_abs {
            max_abs = v.abs();
            max_val = v;
        }
    }
    if max_val < 0.0 {
        for v in col.iter_mut() {
            *v = -*v;
        }
    }
}

/// Sign-robust element-wise max error between a ferrolearn embedding and a
/// hard-coded sklearn oracle matrix (both columns aligned to max-abs-positive).
/// Returns the maximum absolute element-wise difference after alignment.
fn sign_robust_max_err(emb: &Array2<f64>, sklearn: &[Vec<f64>]) -> f64 {
    let n = sklearn.len();
    let nc = sklearn[0].len();
    let mut ferro: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..nc).map(|c| emb[[i, c]]).collect())
        .collect();
    let mut sk: Vec<Vec<f64>> = sklearn.to_vec();

    for c in 0..nc {
        let mut fcol: Vec<f64> = (0..n).map(|i| ferro[i][c]).collect();
        let mut scol: Vec<f64> = (0..n).map(|i| sk[i][c]).collect();
        sign_align_column(&mut fcol);
        sign_align_column(&mut scol);
        for i in 0..n {
            ferro[i][c] = fcol[i];
            sk[i][c] = scol[i];
        }
    }

    let mut max_err = 0.0_f64;
    for i in 0..n {
        for c in 0..nc {
            let e = (ferro[i][c] - sk[i][c]).abs();
            if e > max_err {
                max_err = e;
            }
        }
    }
    max_err
}

// ---------------------------------------------------------------------------
// HEADLINE (formerly DIVERGENCE, now FIXED) — parity at reg*trace
// ---------------------------------------------------------------------------

/// Re-audit of the fixed reg formula (`lle.rs:185-191` =
/// `_locally_linear.py:72-77`). Input: the 10x3 fixture,
/// `LLE::new(2).with_n_neighbors(4).with_reg(1e-3)`.
///
/// Expected (sklearn 1.5.2 live oracle, /tmp, full precision — R-CHAR-3):
///   LocallyLinearEmbedding(n_components=2, n_neighbors=4, reg=1e-3,
///       method='standard', eigen_solver='dense').fit_transform(X)
/// Sign-robust element-wise error must be < 1e-6.
///
/// Tracking: #1459 (design crosslink); blocker #1460.
#[allow(
    clippy::excessive_precision,
    reason = "sklearn 1.5.2 live-oracle embedding hard-coded at full f64 precision (R-CHAR-3)"
)]
#[test]
#[allow(
    clippy::assertions_on_constants,
    reason = "false-arm guards an unreachable fit error path (R: no bare panic!)"
)]
fn divergence_standard_lle_embedding_magnitude() {
    let sklearn: Vec<Vec<f64>> = vec![
        vec![0.24567157527665856, 0.5674039617688269],
        vec![0.01227528485688903, 0.40123298809523117],
        vec![-0.20867547621436816, 0.18628911468361917],
        vec![-0.4724819020555504, 0.10606641432903302],
        vec![0.2731058138705527, 0.08680628848070257],
        vec![0.041095176433607154, -0.08152757197378396],
        vec![-0.24079917531930792, -0.1404871440512273],
        vec![-0.44342955190913846, -0.3773298791179383],
        vec![0.5132534241683719, -0.28842866259481065],
        vec![0.27998483080709957, -0.46002550961679467],
    ];
    let x = fixture();
    let fitted = match LLE::new(2).with_n_neighbors(4).with_reg(1e-3).fit(&x, &()) {
        Ok(f) => f,
        Err(e) => {
            assert!(false, "ferrolearn LLE::fit failed unexpectedly: {e}");
            return;
        }
    };
    let emb = fitted.embedding();
    assert_eq!(emb.dim(), (10, 2), "embedding shape mismatch");
    let max_err = sign_robust_max_err(emb, &sklearn);
    assert!(
        max_err < 1e-6,
        "sign-aligned |ferrolearn - sklearn| max err {max_err:.3e} exceeds 1e-6 \
         (reg*trace fix; lle.rs:185-191 vs _locally_linear.py:72-77)"
    );
}

// ---------------------------------------------------------------------------
// (a) DIFFERENT n_neighbors — k=3 and k=5, n_components=2, reg=1e-3
// ---------------------------------------------------------------------------

/// Parity probe (a): same 10x3 fixture, n_neighbors=3, n_components=2, reg=1e-3.
/// sklearn 1.5.2 live oracle (/tmp, full precision — R-CHAR-3). Bottom-2
/// M-eigenvalues distinct (column ranges 0.97, 1.05). Sign-robust err < 1e-6.
/// Tracking: #1459; blocker #1460.
#[allow(
    clippy::excessive_precision,
    reason = "sklearn 1.5.2 live-oracle embedding hard-coded at full f64 precision (R-CHAR-3)"
)]
#[test]
#[allow(
    clippy::assertions_on_constants,
    reason = "false-arm guards an unreachable fit error path (R: no bare panic!)"
)]
fn parity_neighbors_k3() {
    let sklearn: Vec<Vec<f64>> = vec![
        vec![0.2611110175495211, 0.5803861039291903],
        vec![0.024444819137518448, 0.41259549260692774],
        vec![-0.19759191143921037, 0.15951523869501916],
        vec![-0.46256546268071597, 0.09885547297193548],
        vec![0.2785205205259328, 0.07542362771343761],
        vec![0.03522946245274968, -0.07792883647071872],
        vec![-0.23974122931954048, -0.13291268331637102],
        vec![-0.46717292166397584, -0.32540426142999584],
        vec![0.5041759036961302, -0.3248551729507079],
        vec![0.26358980174146884, -0.4656749817488368],
    ];
    let x = fixture();
    let fitted = match LLE::new(2).with_n_neighbors(3).with_reg(1e-3).fit(&x, &()) {
        Ok(f) => f,
        Err(e) => {
            assert!(false, "fit failed: {e}");
            return;
        }
    };
    let max_err = sign_robust_max_err(fitted.embedding(), &sklearn);
    assert!(
        max_err < 1e-6,
        "k=3 sign-aligned max err {max_err:.3e} exceeds 1e-6"
    );
}

/// Parity probe (a): same 10x3 fixture, n_neighbors=5, n_components=2, reg=1e-3.
/// sklearn 1.5.2 live oracle (/tmp, full precision — R-CHAR-3). Sign-robust < 1e-6.
/// Tracking: #1459; blocker #1460.
#[allow(
    clippy::excessive_precision,
    reason = "sklearn 1.5.2 live-oracle embedding hard-coded at full f64 precision (R-CHAR-3)"
)]
#[test]
#[allow(
    clippy::assertions_on_constants,
    reason = "false-arm guards an unreachable fit error path (R: no bare panic!)"
)]
fn parity_neighbors_k5() {
    let sklearn: Vec<Vec<f64>> = vec![
        vec![0.08544993833715649, 0.611568339726573],
        vec![-0.09699809693090795, 0.39186255519093055],
        vec![-0.2519866936699793, 0.1239528371964967],
        vec![-0.487425383022166, -0.0306736998272668],
        vec![0.24602929112014085, 0.15705465963138604],
        vec![0.06586889326441032, -0.06350187846908205],
        vec![-0.19297989742231292, -0.19284293739466254],
        vec![-0.32560731826979583, -0.4827618036165096],
        vec![0.5669504830382872, -0.14610798606263997],
        vec![0.39069878359504306, -0.368550086374679],
    ];
    let x = fixture();
    let fitted = match LLE::new(2).with_n_neighbors(5).with_reg(1e-3).fit(&x, &()) {
        Ok(f) => f,
        Err(e) => {
            assert!(false, "fit failed: {e}");
            return;
        }
    };
    let max_err = sign_robust_max_err(fitted.embedding(), &sklearn);
    assert!(
        max_err < 1e-6,
        "k=5 sign-aligned max err {max_err:.3e} exceeds 1e-6"
    );
}

// ---------------------------------------------------------------------------
// (b) DIFFERENT reg — k=4, reg=1e-2 and reg=1e-1, n_components=2
// ---------------------------------------------------------------------------

/// Parity probe (b): 10x3 fixture, n_neighbors=4, reg=1e-2, n_components=2.
/// sklearn 1.5.2 live oracle (/tmp, full precision — R-CHAR-3). Sign-robust < 1e-6.
/// Verifies the `R = reg * trace` scaling is faithful at reg=1e-2.
/// Tracking: #1459; blocker #1460.
#[allow(
    clippy::excessive_precision,
    reason = "sklearn 1.5.2 live-oracle embedding hard-coded at full f64 precision (R-CHAR-3)"
)]
#[test]
#[allow(
    clippy::assertions_on_constants,
    reason = "false-arm guards an unreachable fit error path (R: no bare panic!)"
)]
fn parity_reg_1em2() {
    let sklearn: Vec<Vec<f64>> = vec![
        vec![-0.3834384206767204, -0.4739075404859111],
        vec![-0.1194503931119888, -0.40527981777596633],
        vec![0.15999614326871633, -0.24411203819759897],
        vec![0.42207347141449014, -0.1906701555525818],
        vec![-0.289644605844804, -0.026074908778916828],
        vec![-0.001157261725638098, 0.08129850018192156],
        vec![0.2841847048938745, 0.08865335150910708],
        vec![0.5139088287468025, 0.23055698291169138],
        vec![-0.4240572085561161, 0.4171121656224129],
        vec![-0.16241525840869647, 0.5224234605658756],
    ];
    let x = fixture();
    let fitted = match LLE::new(2).with_n_neighbors(4).with_reg(1e-2).fit(&x, &()) {
        Ok(f) => f,
        Err(e) => {
            assert!(false, "fit failed: {e}");
            return;
        }
    };
    let max_err = sign_robust_max_err(fitted.embedding(), &sklearn);
    assert!(
        max_err < 1e-6,
        "reg=1e-2 sign-aligned max err {max_err:.3e} exceeds 1e-6"
    );
}

/// Parity probe (b): 10x3 fixture, n_neighbors=4, reg=1e-1, n_components=2.
/// sklearn 1.5.2 live oracle (/tmp, full precision — R-CHAR-3). Sign-robust < 1e-6.
/// Verifies the `R = reg * trace` scaling is faithful at reg=1e-1.
/// Tracking: #1459; blocker #1460.
#[allow(
    clippy::excessive_precision,
    reason = "sklearn 1.5.2 live-oracle embedding hard-coded at full f64 precision (R-CHAR-3)"
)]
#[test]
#[allow(
    clippy::assertions_on_constants,
    reason = "false-arm guards an unreachable fit error path (R: no bare panic!)"
)]
fn parity_reg_1em1() {
    let sklearn: Vec<Vec<f64>> = vec![
        vec![-0.3640934686667025, -0.4416769857710136],
        vec![-0.16100101867936306, -0.488444877663602],
        vec![0.18001878890154469, -0.2650097826040484],
        vec![0.4407610354440227, -0.11712171319021789],
        vec![-0.33759246005757804, -0.014795779305045248],
        vec![-0.013457408901606634, 0.18102816313133435],
        vec![0.3253352731741937, 0.13148480069388285],
        vec![0.47565465154198733, 0.10545156050675629],
        vec![-0.3721703087638258, 0.3921742547831476],
        vec![-0.17345508399265788, 0.5169103594188048],
    ];
    let x = fixture();
    let fitted = match LLE::new(2).with_n_neighbors(4).with_reg(1e-1).fit(&x, &()) {
        Ok(f) => f,
        Err(e) => {
            assert!(false, "fit failed: {e}");
            return;
        }
    };
    let max_err = sign_robust_max_err(fitted.embedding(), &sklearn);
    assert!(
        max_err < 1e-6,
        "reg=1e-1 sign-aligned max err {max_err:.3e} exceeds 1e-6"
    );
}

// ---------------------------------------------------------------------------
// (c) n_components=1 — single-column embedding
// ---------------------------------------------------------------------------

/// Parity probe (c): 10x3 fixture, n_components=1, n_neighbors=4, reg=1e-3.
/// sklearn 1.5.2 live oracle (/tmp, full precision — R-CHAR-3). Single column,
/// sign-robust < 1e-6. (Same bottom-nontrivial eigenvector as the headline
/// column 0, confirming `n_components` slicing picks the right eigenvector.)
/// Tracking: #1459; blocker #1460.
#[allow(
    clippy::excessive_precision,
    reason = "sklearn 1.5.2 live-oracle embedding hard-coded at full f64 precision (R-CHAR-3)"
)]
#[test]
#[allow(
    clippy::assertions_on_constants,
    reason = "false-arm guards an unreachable fit error path (R: no bare panic!)"
)]
fn parity_n_components_1() {
    let sklearn: Vec<Vec<f64>> = vec![
        vec![0.24567157527663255],
        vec![0.012275284856852642],
        vec![-0.20867547621441795],
        vec![-0.4724819020556051],
        vec![0.2731058138704968],
        vec![0.04109517643354071],
        vec![-0.24079917531937797],
        vec![-0.4434295519092231],
        vec![0.5132534241682926],
        vec![0.2799848308070095],
    ];
    let x = fixture();
    let fitted = match LLE::new(1).with_n_neighbors(4).with_reg(1e-3).fit(&x, &()) {
        Ok(f) => f,
        Err(e) => {
            assert!(false, "fit failed: {e}");
            return;
        }
    };
    let emb = fitted.embedding();
    assert_eq!(emb.dim(), (10, 1), "n_components=1 shape mismatch");
    let max_err = sign_robust_max_err(emb, &sklearn);
    assert!(
        max_err < 1e-6,
        "n_components=1 sign-aligned max err {max_err:.3e} exceeds 1e-6"
    );
}

// ---------------------------------------------------------------------------
// (d) LARGER fixture — 18 points, noisy 2D manifold in 3D, k=5, nc=2
// ---------------------------------------------------------------------------

/// The 18x3 larger fixture: a noisy spiral (2D manifold) in 3D.
/// Generated in /tmp at full precision (R-CHAR-3).
#[allow(
    clippy::excessive_precision,
    reason = "fixture coordinates generated in /tmp at full f64 precision (R-CHAR-3)"
)]
fn fixture_d() -> Array2<f64> {
    array![
        [
            0.016905257038003562,
            -0.004659373705408328,
            0.000328201636785844
        ],
        [
            0.17780506273922514,
            0.02309125372592706,
            0.029432420434941833
        ],
        [
            0.33117702378041336,
            0.10445011503229645,
            0.06900010946839964
        ],
        [0.4629425524527352, 0.2611122680136507, 0.08651981150568984],
        [0.5422573190165921, 0.45529626223315584, 0.11521956803680394],
        [0.546058067246939, 0.6869304411891128, 0.14829763258228215],
        [0.521460175101545, 0.9077956507180266, 0.19297758514715885],
        [0.40825581521951915, 1.1625489930860695, 0.2261730751487876],
        [0.22311619401703997, 1.379443002487476, 0.23124183909278195],
        [-0.050578930569283816, 1.5984877605934358, 0.260541139167741],
        [-0.347478432890261, 1.7423571328724425, 0.27760689116476517],
        [-0.6972930123955687, 1.788900781887412, 0.31690781836803783],
        [-1.1132182735138167, 1.8234426656428122, 0.37060278524988155],
        [-1.5217201790314774, 1.7281044380199355, 0.38055307716394704],
        [-1.9292780116284851, 1.528616584288429, 0.3946813138506824],
        [-2.347945850468593, 1.2601819427495744, 0.4636524211163123],
        [-2.6792151827639743, 0.877749339222749, 0.4897084239174371],
        [-2.9676044713349645, 0.42437436403171774, 0.5025257773555531],
    ]
}

/// Parity probe (d): 18x3 noisy-spiral fixture, n_components=2, n_neighbors=5,
/// reg=1e-3. sklearn 1.5.2 live oracle (/tmp, full precision — R-CHAR-3).
/// Bottom-3 M-eigenvalues [~0, 4.91e-06, 1.28e-05] are DISTINCT (ratio ~2.6x),
/// so the sign-robust comparison is well-defined. Sign-robust < 1e-6.
/// Tracking: #1459; blocker #1460.
#[allow(
    clippy::excessive_precision,
    reason = "sklearn 1.5.2 live-oracle embedding hard-coded at full f64 precision (R-CHAR-3)"
)]
#[test]
#[allow(
    clippy::assertions_on_constants,
    reason = "false-arm guards an unreachable fit error path (R: no bare panic!)"
)]
fn parity_larger_18x3() {
    let sklearn: Vec<Vec<f64>> = vec![
        vec![0.3487931200380773, 0.3526439408217929],
        vec![0.3386142546759704, 0.2770583779495339],
        vec![0.3168915404484758, 0.18042851769584753],
        vec![0.27926425239516806, 0.06554063394236413],
        vec![0.23364305117863848, -0.04333993551910736],
        vec![0.17968791461603772, -0.13529767143736787],
        vec![0.1279177362812734, -0.2122923625199984],
        vec![0.06666065921752107, -0.2706036823657891],
        vec![0.011253987460804117, -0.29166893536965294],
        vec![-0.0515920045200846, -0.2929609722862535],
        vec![-0.10242148619744036, -0.2664078585745996],
        vec![-0.1438745036349927, -0.20598717994084],
        vec![-0.19082590429502747, -0.12960150818951677],
        vec![-0.22577070428380372, -0.03398112330271227],
        vec![-0.25673206128972154, 0.07480544550730077],
        vec![-0.2869769781946962, 0.19547834882288515],
        vec![-0.3111364217591793, 0.31104864702426116],
        vec![-0.33339645228152326, 0.42513731774837193],
    ];
    let x = fixture_d();
    let fitted = match LLE::new(2).with_n_neighbors(5).with_reg(1e-3).fit(&x, &()) {
        Ok(f) => f,
        Err(e) => {
            assert!(false, "fit failed: {e}");
            return;
        }
    };
    let emb = fitted.embedding();
    assert_eq!(emb.dim(), (18, 2), "larger fixture shape mismatch");
    let max_err = sign_robust_max_err(emb, &sklearn);
    assert!(
        max_err < 1e-6,
        "larger 18x3 sign-aligned max err {max_err:.3e} exceeds 1e-6"
    );
}

// ---------------------------------------------------------------------------
// (e) HIGHER-D input — 12 points in 5D, k=4, nc=2
// ---------------------------------------------------------------------------

/// The 12x5 higher-dimensional fixture (manifold embedded in 5D).
/// Generated in /tmp at full precision (R-CHAR-3).
#[allow(
    clippy::excessive_precision,
    reason = "fixture coordinates generated in /tmp at full f64 precision (R-CHAR-3)"
)]
fn fixture_e() -> Array2<f64> {
    array![
        [
            0.008747273706525897,
            -0.0014303649840814708,
            -0.0024228256611057214,
            0.9867334072036926,
            -4.1423146864679204e-05
        ],
        [
            0.08931093409026943,
            0.005581315998743706,
            0.18239509657997585,
            0.9651451172064697,
            0.021944712370483104
        ],
        [
            0.17738698346840445,
            0.030679183776252446,
            0.35912356941953927,
            0.8576975187426241,
            0.04801771199655164
        ],
        [
            0.2671298964192746,
            0.07806435224540378,
            0.5266799015241487,
            0.6833940595288472,
            0.07840094868085562
        ],
        [
            0.36911451212896085,
            0.1306835217725744,
            0.668463624730341,
            0.46942450957585863,
            0.11224130820337616
        ],
        [
            0.454912920730464,
            0.2102729270173602,
            0.7857325858820906,
            0.20479095317528612,
            0.13349386356658535
        ],
        [
            0.5444326688762345,
            0.29508818565648065,
            0.8861181126396612,
            -0.06742302198493666,
            0.16408125181297506
        ],
        [
            0.6366819946707603,
            0.4064404132264218,
            0.9629191968989791,
            -0.33961313029600715,
            0.19738718383310414
        ],
        [
            0.7260864750439932,
            0.5227638888016994,
            0.9923883748944706,
            -0.5732455304338849,
            0.2235196104670571
        ],
        [
            0.8128736459404591,
            0.670508228643167,
            0.9984403314535786,
            -0.7815427720730964,
            0.23952576912779264
        ],
        [
            0.9120914191190469,
            0.82992411726861,
            0.9749945034828079,
            -0.9127211991354094,
            0.27470333279194853
        ],
        [
            1.000614337646813,
            1.0060455082104667,
            0.9050820963202801,
            -0.9907019645055265,
            0.3019267706904924
        ],
    ]
}

/// Parity probe (e): 12x5 higher-D fixture, n_components=2, n_neighbors=4,
/// reg=1e-3. sklearn 1.5.2 live oracle (/tmp, full precision — R-CHAR-3).
/// Bottom-3 M-eigenvalues [~0, 1.76e-07, 3.05e-03] WELL separated. The local
/// covariance C = Z Z^T is k×k regardless of input dim, so this checks dim
/// handling. Sign-robust < 1e-6.
/// Tracking: #1459; blocker #1460.
#[allow(
    clippy::excessive_precision,
    reason = "sklearn 1.5.2 live-oracle embedding hard-coded at full f64 precision (R-CHAR-3)"
)]
#[test]
#[allow(
    clippy::assertions_on_constants,
    reason = "false-arm guards an unreachable fit error path (R: no bare panic!)"
)]
fn parity_higher_dim_12x5() {
    let sklearn: Vec<Vec<f64>> = vec![
        vec![-0.42537782603377755, 0.4507153238858145],
        vec![-0.3779424218768469, 0.25842177726056664],
        vec![-0.31180150456781364, 0.05449279693271225],
        vec![-0.23168266229836765, -0.13952430553684786],
        vec![-0.14503958505690923, -0.2725302639225052],
        vec![-0.049714904815554295, -0.3219949342723579],
        vec![0.04676627311704933, -0.33836872576505306],
        vec![0.14441407588168495, -0.29444103606733585],
        vec![0.22949656756902348, -0.16878573264301117],
        vec![0.31123897832137054, 0.03239146336712208],
        vec![0.37774419223791406, 0.25703861260653155],
        vec![0.43189881115134027, 0.48258502415461646],
    ];
    let x = fixture_e();
    let fitted = match LLE::new(2).with_n_neighbors(4).with_reg(1e-3).fit(&x, &()) {
        Ok(f) => f,
        Err(e) => {
            assert!(false, "fit failed: {e}");
            return;
        }
    };
    let emb = fitted.embedding();
    assert_eq!(emb.dim(), (12, 2), "higher-D fixture shape mismatch");
    let max_err = sign_robust_max_err(emb, &sklearn);
    assert!(
        max_err < 1e-6,
        "higher-D 12x5 sign-aligned max err {max_err:.3e} exceeds 1e-6"
    );
}

// ---------------------------------------------------------------------------
// (f) NEAR-DEGENERATE TRACE — tiny-scale manifold so each neighborhood trace
//     is ~1e-7, exercising `R = reg * trace` (trace>0 branch) with tiny values.
// ---------------------------------------------------------------------------

/// The 12x3 tiny-scale fixture: a generic 1D-ish manifold scaled to ~1e-3 so
/// each neighborhood's covariance trace is ~1e-7 (deep in the `reg*trace`
/// regime). M-eigenvalues stay distinct. Generated in /tmp (R-CHAR-3).
#[allow(
    clippy::excessive_precision,
    reason = "fixture coordinates generated in /tmp at full f64 precision (R-CHAR-3)"
)]
fn fixture_f() -> Array2<f64> {
    array![
        [0.0, 0.0, 0.0],
        [
            9.090909090909092e-05,
            4.3831978596770574e-05,
            4.545454545454546e-05
        ],
        [
            0.00018181818181818183,
            9.954133760030402e-05,
            9.090909090909092e-05
        ],
        [
            0.00027272727272727274,
            0.0001630848642152018,
            0.00013636363636363637
        ],
        [
            0.00036363636363636367,
            0.00023155645233171346,
            0.00018181818181818183
        ],
        [
            0.0004545454545454546,
            0.00030356716516616626,
            0.0002272727272727273
        ],
        [
            0.0005454545454545455,
            0.0003794268814092675,
            0.00027272727272727274
        ],
        [
            0.0006363636363636364,
            0.00046110376906878086,
            0.0003181818181818182
        ],
        [
            0.0007272727272727273,
            0.0005519668870464846,
            0.00036363636363636367
        ],
        [
            0.0008181818181818183,
            0.0006563455771997083,
            0.00040909090909090913
        ],
        [
            0.0009090909090909092,
            0.0007789632699095118,
            0.0004545454545454546
        ],
        [0.001, 0.0009243197504692071, 0.0005],
    ]
}

/// Parity probe (f): 12x3 tiny-scale fixture (traces ~1e-7), n_components=2,
/// n_neighbors=3, reg=1e-3. sklearn 1.5.2 live oracle (/tmp, full precision —
/// R-CHAR-3). Bottom-3 M-eigenvalues [~0, 8.55e-07, 8.55e-03] DISTINCT.
/// Exercises the `trace > 0 -> R = reg * trace` branch with tiny absolute
/// trace; both ferrolearn and sklearn must scale identically. Sign-robust < 1e-6.
/// Tracking: #1459; blocker #1460.
#[allow(
    clippy::excessive_precision,
    reason = "sklearn 1.5.2 live-oracle embedding hard-coded at full f64 precision (R-CHAR-3)"
)]
#[test]
#[allow(
    clippy::assertions_on_constants,
    reason = "false-arm guards an unreachable fit error path (R: no bare panic!)"
)]
fn parity_near_degenerate_trace_12x3() {
    let sklearn: Vec<Vec<f64>> = vec![
        vec![-0.4324603797222716, 0.44265619249185245],
        vec![-0.3636039182860932, 0.2721136625240454],
        vec![-0.2904829546192197, 0.08599366153083715],
        vec![-0.21433546994514013, -0.08894904043464108],
        vec![-0.13646701895060295, -0.22571643965184135],
        vec![-0.05721928487448804, -0.3166099588237219],
        vec![0.023550672929434562, -0.35142789536834135],
        vec![0.10666337708368247, -0.319451727015],
        vec![0.19360795018064003, -0.20814196381365038],
        vec![0.28644253379625345, -0.013280668059219802],
        vec![0.3869680748425403, 0.23250317176345695],
        vec![0.4973364167837573, 0.49031100485602225],
    ];
    let x = fixture_f();
    let fitted = match LLE::new(2).with_n_neighbors(3).with_reg(1e-3).fit(&x, &()) {
        Ok(f) => f,
        Err(e) => {
            assert!(false, "fit failed: {e}");
            return;
        }
    };
    let emb = fitted.embedding();
    assert_eq!(emb.dim(), (12, 2), "near-degenerate-trace shape mismatch");
    let max_err = sign_robust_max_err(emb, &sklearn);
    assert!(
        max_err < 1e-6,
        "near-degenerate-trace sign-aligned max err {max_err:.3e} exceeds 1e-6"
    );
}

// ---------------------------------------------------------------------------
// GREEN-GUARDS — must PASS against current code (REQ-3/4/5)
// ---------------------------------------------------------------------------

/// Green-guard (REQ-4): embedding shape is `(n_samples, n_components)`.
#[test]
#[allow(
    clippy::assertions_on_constants,
    reason = "false-arm guards an unreachable fit error path (R: no bare panic!)"
)]
fn green_embedding_shape() {
    let x = fixture();
    let fitted = match LLE::new(2).with_n_neighbors(4).fit(&x, &()) {
        Ok(f) => f,
        Err(e) => {
            assert!(false, "fit failed: {e}");
            return;
        }
    };
    assert_eq!(fitted.embedding().dim(), (10, 2));
}

/// Green-guard (REQ-4): the fit path uses no RNG, so two fits on identical input
/// produce element-wise identical embeddings.
#[test]
#[allow(
    clippy::assertions_on_constants,
    reason = "false-arm guards an unreachable fit error path (R: no bare panic!)"
)]
fn green_determinism() {
    let x = fixture();
    let a = match LLE::new(2).with_n_neighbors(4).fit(&x, &()) {
        Ok(f) => f,
        Err(e) => {
            assert!(false, "fit a failed: {e}");
            return;
        }
    };
    let b = match LLE::new(2).with_n_neighbors(4).fit(&x, &()) {
        Ok(f) => f,
        Err(e) => {
            assert!(false, "fit b failed: {e}");
            return;
        }
    };
    let ea = a.embedding();
    let eb = b.embedding();
    assert_eq!(ea.dim(), eb.dim());
    for i in 0..ea.nrows() {
        for c in 0..ea.ncols() {
            assert_eq!(
                ea[[i, c]],
                eb[[i, c]],
                "non-deterministic embedding at ({i},{c})"
            );
        }
    }
}

/// Green-guard (REQ-4): LLE eigenvectors are centered — each embedding column
/// sums to ~0 (sklearn's `null_space` returns eigenvectors of `M` orthogonal to
/// the trivial constant eigenvector, `_locally_linear.py:192-196`; the live
/// oracle column sums are ~1e-11). Compared against the structural constant 0.
#[test]
#[allow(
    clippy::assertions_on_constants,
    reason = "false-arm guards an unreachable fit error path (R: no bare panic!)"
)]
fn green_embedding_columns_centered() {
    let x = fixture();
    let fitted = match LLE::new(2).with_n_neighbors(4).fit(&x, &()) {
        Ok(f) => f,
        Err(e) => {
            assert!(false, "fit failed: {e}");
            return;
        }
    };
    let emb = fitted.embedding();
    for c in 0..emb.ncols() {
        let s: f64 = (0..emb.nrows()).map(|i| emb[[i, c]]).sum();
        assert!(
            s.abs() < 1e-6,
            "embedding column {c} sum {s:.3e} not ~0 (centering)"
        );
    }
}

/// Green-guard (REQ-5): `n_components == 0` -> fit `Err` (`lle.rs:289-294`).
#[test]
fn green_err_n_components_zero() {
    let x = fixture();
    assert!(LLE::new(0).with_n_neighbors(4).fit(&x, &()).is_err());
}

/// Green-guard (REQ-5): `n_neighbors == 0` -> fit `Err` (`lle.rs:295-300`).
#[test]
fn green_err_n_neighbors_zero() {
    let x = fixture();
    assert!(LLE::new(2).with_n_neighbors(0).fit(&x, &()).is_err());
}

/// Green-guard (REQ-5): `n_neighbors >= n_samples` -> fit `Err`
/// (`lle.rs:308-316`; sklearn raises `ValueError("Expected n_neighbors <=
/// n_samples...")`). Fixture has 10 samples.
#[test]
fn green_err_n_neighbors_ge_n_samples() {
    let x = fixture();
    assert!(LLE::new(2).with_n_neighbors(10).fit(&x, &()).is_err());
}

/// Green-guard (REQ-5): `n_components >= n_samples` -> fit `Err`
/// (`lle.rs:318-326`).
#[test]
fn green_err_n_components_ge_n_samples() {
    let x = fixture();
    assert!(LLE::new(10).with_n_neighbors(4).fit(&x, &()).is_err());
}

/// Green-guard (REQ-5): `n_samples < 2` -> fit `Err` (`lle.rs:301-307`).
#[test]
fn green_err_insufficient_samples() {
    let x = array![[1.0, 2.0, 3.0]];
    assert!(LLE::new(1).with_n_neighbors(1).fit(&x, &()).is_err());
}

/// Green-guard (REQ-5): `reg < 0` -> fit `Err` (`lle.rs:327-332`).
#[test]
fn green_err_negative_reg() {
    let x = fixture();
    assert!(
        LLE::new(2)
            .with_n_neighbors(4)
            .with_reg(-1.0)
            .fit(&x, &())
            .is_err()
    );
}
