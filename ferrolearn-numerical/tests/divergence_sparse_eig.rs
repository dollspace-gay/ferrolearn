//! Divergence audit + GREEN GUARDS for `ferrolearn-numerical/src/sparse_eig.rs`
//! (the `scipy.sparse.linalg.eigsh` Lanczos symmetric eigensolver analog)
//! against the LIVE scipy 1.17.1 / numpy 2.4.5 oracle. See
//! `.design/numerical/sparse_eig.md` (crosslink unit #1978).
//!
//! ## R-CHAR-3 provenance (expected values are NEVER copied from ferrolearn)
//!
//! Symmetric-matrix eigenvalues are a UNIQUE deterministic property of the
//! matrix, so they have a strict ground truth. Two oracle matrices:
//!
//! 1. A diagonal matrix `diag([1,2,3,4,5])` — its eigenvalues ARE its diagonal
//!    `[1,2,3,4,5]` by definition (no copy from anywhere).
//! 2. A non-trivial INDEFINITE dense-symmetric `M` constructed as
//!    `0.5*((Q@D@Q.T)+(Q@D@Q.T).T)` with `Q` the seed-0 QR factor and
//!    `D = diag([-3,-1,0.5,2,4,7])`. By construction (orthogonal similarity)
//!    the full spectrum of `M` is exactly `[-3,-1,0.5,2,4,7]` — that IS the
//!    construction, not a copied value. The 6x6 entries `M_ENTRIES` below were
//!    printed by the seed-0 numpy oracle (`M.tolist()`), so the SAME matrix is
//!    fed to ferrolearn and to scipy.
//!
//! The single oracle command (run from `/tmp`, scipy 1.17.1 / numpy 2.4.5) that
//! confirmed every expected subset below:
//!
//! ```text
//! cd /tmp && python3 -c "
//! import numpy as np, scipy.sparse as sp
//! from scipy.sparse.linalg import eigsh
//! A=sp.csr_matrix(np.diag([1.,2.,3.,4.,5.]))
//! for w in ['LA','SA','LM']:
//!     print('diag',w,np.sort(eigsh(A,k=2,which=w)[0]).tolist())
//! np.random.seed(0); Q,_=np.linalg.qr(np.random.randn(6,6))
//! D=np.diag([-3.,-1.,0.5,2.,4.,7.]); M=0.5*((Q@D@Q.T)+(Q@D@Q.T).T)
//! Ms=sp.csr_matrix(M)
//! for w in ['LA','SA','LM','SM']:
//!     print('dense',w,np.sort(eigsh(Ms,k=3,which=w)[0]).tolist())
//! print('full', np.sort(np.linalg.eigvalsh(M)).tolist())
//! "
//! ```
//! → diag LA `[3.999…, 4.999…]`; SA `[1.0, 1.999…]`; LM `[3.999…, 5.0]`;
//!   dense LA `[2.0, 4.0, 7.0]`; SA `[-3.0, -1.0, 0.5]`; LM `[-3.0, 4.0, 7.0]`;
//!   SM `[-1.0, 0.5, 2.0]`; full `[-3.0,-1.0,0.5,2.0,4.0,7.0]`.
//!
//! ## VERDICT (this audit)
//!
//! GREEN — every numerical claim matches the live scipy oracle to <= 1e-7:
//!   * LargestAlgebraic / SmallestAlgebraic / LargestMagnitude eigenvalues on
//!     BOTH the diagonal and the indefinite matrix.
//!   * The STRICT indefinite-LM check: `eigsh(Ms,3,LM)` returns `{-3,4,7}` —
//!     ferrolearn correctly selects the negative `-3.0` over the positive `2.0`
//!     by magnitude. No `which`-selection divergence (NO RED pin).
//!   * SmallestMagnitude on the indefinite matrix returns `{-1,0.5,2}` — matches
//!     scipy 'SM'. (For this 6x6 the full-reorthogonalised Lanczos basis spans
//!     the WHOLE spectrum before `select_indices` runs, so the interior-magnitude
//!     selection happens to be exact here; the design doc flags this as the
//!     fragile case for larger/genuinely-truncated spectra — that risk is the
//!     REQ-5/REQ-6 shift-invert blocker, NOT a divergence on this input.)
//!   * Eigenvector residual `||Av - λv||` and orthonormality `||VᵀV - I||`.
//!
//! Structural gaps (missing 'BE', missing sigma/shift-invert/v0/M/
//! return_eigenvectors ABI, missing eigs/svds/lobpcg companions, k>=n
//! acceptance, k==0 / non-convergence error-type, Result<_,String> vs
//! FerroError, no production consumer, ferray substrate) are filed as `-l
//! blocker` issues, NOT pinned as failing tests — they are structural facts,
//! not numerical assertions on this module's current numerical surface.

use ferrolearn_numerical::sparse_eig::{EigenResult, WhichEigenvalues, eigsh};
use ndarray::Array2;

// ---------------------------------------------------------------------------
// Oracle matrices (R-CHAR-3): construction-true eigenvalues + seed-0 entries.
// ---------------------------------------------------------------------------

/// Entries of the indefinite dense-symmetric `M = 0.5*((Q@D@Q.T)+(Q@D@Q.T).T)`,
/// `Q` = seed-0 QR factor, `D = diag([-3,-1,0.5,2,4,7])`. Printed by the numpy
/// oracle (`M.tolist()`); the SAME matrix is given to scipy and to ferrolearn.
/// Its full spectrum is `[-3,-1,0.5,2,4,7]` BY CONSTRUCTION.
#[rustfmt::skip]
const M_ENTRIES: [[f64; 6]; 6] = [
    [0.0822783030067192, -2.2799456120160078, -2.198709630181292, 0.5219271901483403, -0.7512927142519789, -0.3613987842839974],
    [-2.2799456120160078, 4.73553360968558, 0.5498749768318343, -0.43958052000294456, -1.5291065303298899, -0.8884174769417486],
    [-2.198709630181292, 0.5498749768318343, 4.236216509062989, 0.37364228210623235, -0.9465734756150448, -0.7700064897573321],
    [0.5219271901483403, -0.43958052000294456, 0.37364228210623235, 0.41672146242113195, -0.8010868870003683, -0.25051636215092027],
    [-0.7512927142519789, -1.5291065303298899, -0.9465734756150448, -0.8010868870003683, -1.392097265818038, 0.6436632782753133],
    [-0.3613987842839974, -0.8884174769417486, -0.7700064897573321, -0.25051636215092027, 0.6436632782753133, 1.4213473816416198],
];

/// Full spectrum of `M`, BY CONSTRUCTION (= `D`'s diagonal). Cross-checked live
/// against `np.linalg.eigvalsh(M)` → `[-3,-1,0.5,2,4,7]`.
const M_SPECTRUM: [f64; 6] = [-3.0, -1.0, 0.5, 2.0, 4.0, 7.0];

/// Diagonal matrix `diag([1,2,3,4,5])` — eigenvalues ARE the diagonal.
const DIAG5: [f64; 5] = [1.0, 2.0, 3.0, 4.0, 5.0];

fn m_dense() -> Array2<f64> {
    let mut a = Array2::<f64>::zeros((6, 6));
    for i in 0..6 {
        for j in 0..6 {
            a[[i, j]] = M_ENTRIES[i][j];
        }
    }
    a
}

fn dense_to_sparse(a: &Array2<f64>) -> sprs::CsMat<f64> {
    let (rows, cols) = (a.nrows(), a.ncols());
    let mut indptr = vec![0];
    let mut indices = Vec::new();
    let mut data = Vec::new();
    for i in 0..rows {
        for j in 0..cols {
            let v = a[[i, j]];
            if v.abs() > 1e-300 {
                indices.push(j);
                data.push(v);
            }
        }
        indptr.push(indices.len());
    }
    sprs::CsMat::new((rows, cols), indptr, indices, data)
}

fn sparse_diag(values: &[f64]) -> sprs::CsMat<f64> {
    let n = values.len();
    let mut indptr = vec![0];
    let mut indices = Vec::new();
    let mut data = Vec::new();
    for (i, &v) in values.iter().enumerate() {
        indices.push(i);
        data.push(v);
        indptr.push(i + 1);
    }
    sprs::CsMat::new((n, n), indptr, indices, data)
}

fn ms_sparse() -> sprs::CsMat<f64> {
    dense_to_sparse(&m_dense())
}

/// Sorted (ascending) eigenvalues of an `EigenResult`.
fn sorted_eigvals(r: &EigenResult) -> Vec<f64> {
    let mut v: Vec<f64> = r.eigenvalues.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v
}

fn assert_close_set(got: &[f64], expected: &[f64], tol: f64, label: &str) {
    assert_eq!(
        got.len(),
        expected.len(),
        "{label}: length mismatch got {got:?} expected {expected:?}"
    );
    for (g, e) in got.iter().zip(expected.iter()) {
        assert!(
            (g - e).abs() <= tol,
            "{label}: got {got:?} expected {expected:?} (|{g} - {e}| > {tol})"
        );
    }
}

// ---------------------------------------------------------------------------
// GREEN GUARD — REQ-1: LargestAlgebraic ('LA') eigenvalue parity.
// ---------------------------------------------------------------------------

/// `eigsh(Ms,3,LargestAlgebraic)` must match scipy `eigsh(Ms,k=3,which='LA')`
/// = the three largest-algebraic eigenvalues `[2,4,7]` of the indefinite `M`
/// (construction-true subset of `M_SPECTRUM`, cross-checked live: scipy →
/// `[2.0, 4.0, 7.0]`).
#[test]
fn green_largest_algebraic_dense_indefinite() {
    let r = eigsh(&ms_sparse(), 3, WhichEigenvalues::LargestAlgebraic).unwrap();
    let expected = [M_SPECTRUM[3], M_SPECTRUM[4], M_SPECTRUM[5]]; // [2,4,7]
    assert_close_set(&sorted_eigvals(&r), &expected, 1e-7, "LA dense");
}

/// `eigsh(diag5,2,LargestAlgebraic)` = `[4,5]` (scipy 'LA' → `[3.999…,4.999…]`).
#[test]
fn green_largest_algebraic_diag() {
    let r = eigsh(&sparse_diag(&DIAG5), 2, WhichEigenvalues::LargestAlgebraic).unwrap();
    assert_close_set(&sorted_eigvals(&r), &[4.0, 5.0], 1e-7, "LA diag");
}

// ---------------------------------------------------------------------------
// GREEN GUARD — REQ-2: SmallestAlgebraic ('SA') eigenvalue parity.
// ---------------------------------------------------------------------------

/// `eigsh(Ms,3,SmallestAlgebraic)` = `[-3,-1,0.5]` (scipy 'SA' →
/// `[-3.0,-1.0,0.5]`) — the smallest-algebraic subset incl. both negatives.
#[test]
fn green_smallest_algebraic_dense_indefinite() {
    let r = eigsh(&ms_sparse(), 3, WhichEigenvalues::SmallestAlgebraic).unwrap();
    let expected = [M_SPECTRUM[0], M_SPECTRUM[1], M_SPECTRUM[2]]; // [-3,-1,0.5]
    assert_close_set(&sorted_eigvals(&r), &expected, 1e-7, "SA dense");
}

/// `eigsh(diag5,2,SmallestAlgebraic)` = `[1,2]` (scipy 'SA' → `[1.0,1.999…]`).
#[test]
fn green_smallest_algebraic_diag() {
    let r = eigsh(&sparse_diag(&DIAG5), 2, WhichEigenvalues::SmallestAlgebraic).unwrap();
    assert_close_set(&sorted_eigvals(&r), &[1.0, 2.0], 1e-7, "SA diag");
}

// ---------------------------------------------------------------------------
// GREEN GUARD — REQ-3: LargestMagnitude ('LM') eigenvalue parity.
// The STRICT indefinite check: |-3| > |2|, so '-3' beats '2' into the top-3.
// ---------------------------------------------------------------------------

/// `eigsh(Ms,3,LargestMagnitude)` = `{-3,4,7}` (scipy 'LM' → `[-3.0,4.0,7.0]`).
/// The negative `-3.0` is selected OVER the positive `2.0` by magnitude — this
/// is the indefinite-spectrum guard the design doc flags as fragile. ferrolearn
/// selects the correct subset here (full Lanczos basis spans the whole spectrum
/// before `select_indices`), so this is GREEN.
#[test]
fn green_largest_magnitude_dense_indefinite_strict() {
    let r = eigsh(&ms_sparse(), 3, WhichEigenvalues::LargestMagnitude).unwrap();
    let expected = [M_SPECTRUM[0], M_SPECTRUM[4], M_SPECTRUM[5]]; // [-3,4,7]
    assert_close_set(
        &sorted_eigvals(&r),
        &expected,
        1e-7,
        "LM dense (indefinite)",
    );
}

/// `eigsh(diag5,2,LargestMagnitude)` = `[4,5]` (scipy 'LM' → `[3.999…,5.0]`).
#[test]
fn green_largest_magnitude_diag() {
    let r = eigsh(&sparse_diag(&DIAG5), 2, WhichEigenvalues::LargestMagnitude).unwrap();
    assert_close_set(&sorted_eigvals(&r), &[4.0, 5.0], 1e-7, "LM diag");
}

// ---------------------------------------------------------------------------
// GREEN GUARD — REQ-4: eigenvector residual + orthonormality.
// Contract is the RESIDUAL (eigenvectors are sign-/rotation-ambiguous), not
// element-wise vs scipy. λ ground truth: M_SPECTRUM (= np.linalg.eigvalsh(M)).
// ---------------------------------------------------------------------------

/// For each returned pair `(λᵢ, vᵢ)`: `||M·vᵢ − λᵢ·vᵢ|| ≤ 1e-7`, and the
/// returned eigenvector columns are orthonormal `||VᵀV − I||_max ≤ 1e-8`.
#[test]
fn green_eigenvector_residual_and_orthonormality() {
    let md = m_dense();
    let r = eigsh(&ms_sparse(), 3, WhichEigenvalues::LargestAlgebraic).unwrap();

    let k = r.eigenvectors.ncols();
    assert_eq!(k, 3);

    // Residual ||M v - λ v|| per returned pair.
    for c in 0..k {
        let v = r.eigenvectors.column(c).to_owned();
        let lam = r.eigenvalues[c];
        let resid_vec = &md.dot(&v) - &(lam * &v);
        let resid = resid_vec.dot(&resid_vec).sqrt();
        assert!(resid <= 1e-7, "residual col {c} (λ={lam}) = {resid} > 1e-7");
    }

    // Orthonormality: VᵀV ≈ I.
    let vtv = r.eigenvectors.t().dot(&r.eigenvectors);
    for i in 0..k {
        for j in 0..k {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!(
                (vtv[[i, j]] - expected).abs() <= 1e-8,
                "VᵀV[{i},{j}] = {} expected {expected}",
                vtv[[i, j]]
            );
        }
    }
}

// ---------------------------------------------------------------------------
// GREEN GUARD — fragile case verified: SmallestMagnitude on the indefinite M.
// scipy `eigsh(Ms,k=3,which='SM')` → [-1, 0.5, 2] (the 3 eigenvalues closest to
// 0 by |λ|). ferrolearn's `SmallestMagnitude` returns the SAME subset here.
// This is a GREEN guard, NOT a red pin: for this fully-spanned 6x6 the
// interior-magnitude selection is exact. (The shift-invert structural gap for
// genuinely-truncated interior spectra is the REQ-5/REQ-6 blocker.)
// ---------------------------------------------------------------------------

#[test]
fn green_smallest_magnitude_dense_indefinite() {
    let r = eigsh(&ms_sparse(), 3, WhichEigenvalues::SmallestMagnitude).unwrap();
    // [-1, 0.5, 2] — the 3 closest to zero by magnitude.
    let expected = [M_SPECTRUM[1], M_SPECTRUM[2], M_SPECTRUM[3]];
    assert_close_set(
        &sorted_eigvals(&r),
        &expected,
        1e-7,
        "SM dense (indefinite)",
    );
}

// ---------------------------------------------------------------------------
// GREEN GUARD — documented deviation (R-DEV-2): k==n edge.
// scipy RAISES `TypeError("Cannot use scipy.linalg.eigh for sparse A with
// k >= N")` for k>=n. ferrolearn instead ACCEPTS k==n (full-Lanczos path) and
// returns all n eigenvalues. This guard PINS ferrolearn's actual documented
// behavior (returns the full, correct spectrum) so the deviation is locked;
// the ABI divergence vs scipy's TypeError is filed as a `-l blocker`, NOT a
// failing test (a doomed-by-design assertion is not pinned — task §2).
// ---------------------------------------------------------------------------

#[test]
fn green_k_equals_n_returns_full_spectrum() {
    // scipy would raise here; ferrolearn returns the full (correct) spectrum.
    let r = eigsh(&sparse_diag(&DIAG5), 5, WhichEigenvalues::LargestAlgebraic).unwrap();
    assert_close_set(&sorted_eigvals(&r), &DIAG5, 1e-7, "k==n full spectrum");
}
