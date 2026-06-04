//! Divergence pins for `SpectralEmbedding` / `FittedSpectralEmbedding`
//! against scikit-learn 1.5.2 `sklearn.manifold.SpectralEmbedding`
//! (`/home/doll/scikit-learn/sklearn/manifold/_spectral_embedding.py`).
//!
//! Tracking: #1443 (RBF embedding VALUE parity), blocker #1444.
//!
//! DEGENERATE-EIGENVALUE CARVE-OUT (REQ-7): the previously-pinned two-cluster
//! fixture (`[[0,0],[0.1,0],[0,0.1],[5,5],[5.1,5],[5,5.1]]`) is a near-disconnected
//! RBF graph with a REPEATED 0-eigenvalue (2D null space). faer and sklearn's
//! arpack pick DIFFERENT orthonormal bases of that subspace, so element-wise
//! parity is impossible there. Per goal.md R-DEFER-3 a documented carve-out has
//! NO committed failing test — that fixture is intentionally NOT pinned here.
//!
//! What IS pinned below: a CONNECTED-graph, DISTINCT-eigenvalue, ASYMMETRIC
//! fixture (`[[0,0],[1.2,0.3],[2,1.1],[3.5,0.2],[4.1,2]]`), where the bottom
//! eigenvectors are unique up to sign AND have a UNIQUE max-abs entry, so the
//! deterministic sign-flip is unambiguous and parity against sklearn is
//! well-defined (unlike the degenerate two-cluster carve-out documented above).
//! The fix (RBF diag=0 matching scipy's `csgraph_laplacian`, `/dd` rescale,
//! deterministic sign-flip) makes ferrolearn's `embedding()` MATCH sklearn
//! element-wise on BOTH the 2-component and 1-component cases — see
//! `green_rbf_embedding_value_parity_2` / `_1`.
//!
//! SIGN-TIE AVOIDANCE: the prior SYMMETRIC fixture (`[[0,0]..[4,0]]`) produced
//! ANTISYMMETRIC eigenvectors with `|v[0]| == |v[4]|`, a ULP-level tie for the
//! sign-flip argmax (faer and numpy round it oppositely → opposite global sign
//! on the 1-component embedding). Using an ASYMMETRIC fixture gives a UNIQUE
//! max-abs entry per eigenvector, so the sign is unambiguous and both 1- and
//! 2-component embeddings match sklearn to ~1e-15.

use ferrolearn_core::traits::Fit;
use ferrolearn_decomp::{Affinity, FittedSpectralEmbedding, SpectralEmbedding};
use ndarray::{Array2, array};

/// Connected-graph, DISTINCT-eigenvalue, ASYMMETRIC fixture: 5 points in 2-D.
/// With RBF(gamma=0.3) the L_sym eigenvalues are distinct, so the bottom
/// eigenvectors are unique up to sign; the asymmetry further gives each
/// eigenvector a UNIQUE max-abs entry, so the deterministic sign-flip argmax is
/// unambiguous (no faer-vs-numpy ULP tie) and element-wise parity against
/// sklearn is well-defined (unlike the degenerate two-cluster carve-out
/// documented in the module header).
fn line5() -> Array2<f64> {
    array![[0.0, 0.0], [1.2, 0.3], [2.0, 1.1], [3.5, 0.2], [4.1, 2.0]]
}

/// Fit, returning the fitted model or failing the test with the error.
#[allow(
    clippy::assertions_on_constants,
    reason = "assert!(false,..) is the no-bare-panic failure path for a test helper"
)]
fn fit_or_fail(se: &SpectralEmbedding, x: &Array2<f64>) -> FittedSpectralEmbedding {
    match se.fit(x, &()) {
        Ok(f) => f,
        Err(e) => {
            assert!(false, "fit failed: {e:?}");
            unreachable!()
        }
    }
}

/// Live sklearn 1.5.2 oracle (R-CHAR-3), 2-component embedding on `line5()`:
///
/// ```text
/// import numpy as np
/// from sklearn.manifold import SpectralEmbedding
/// X = np.array([[0.,0.],[1.2,0.3],[2.,1.1],[3.5,0.2],[4.1,2.]])
/// SpectralEmbedding(n_components=2, affinity='rbf', gamma=0.3,
///                   eigen_solver='arpack', random_state=0).fit_transform(X)
/// ```
#[allow(
    clippy::excessive_precision,
    reason = "live sklearn 1.5.2 oracle values (R-CHAR-3); Rust rounds to nearest f64"
)]
const SKLEARN_EMBEDDING_2: [[f64; 2]; 5] = [
    [-0.5492950432380824, 0.5911304325996134],
    [-0.32478442062147483, -0.047104629065039066],
    [0.0371154760143516, -0.564016870974581],
    [0.5056202555986625, 0.053627532619619465],
    [0.745795887914657, 0.6045965227471306],
];

/// Live sklearn 1.5.2 oracle (R-CHAR-3), 1-component embedding on `line5()`
/// (same fixture, `n_components=1`). Note sklearn's `_deterministic_vector_sign_flip`
/// runs per-column independently, so the 1-component sign need NOT match the
/// first column of the 2-component embedding.
#[allow(
    clippy::excessive_precision,
    reason = "live sklearn 1.5.2 oracle values (R-CHAR-3); Rust rounds to nearest f64"
)]
const SKLEARN_EMBEDDING_1: [f64; 5] = [
    -0.5492950432380823,
    -0.32478442062147495,
    0.037115476014351646,
    0.5056202555986625,
    0.7457958879146575,
];

/// Green-guard (HEADLINE, REQ-1): ferrolearn's RBF `SpectralEmbedding` embedding
/// MATCHES `sklearn.manifold.SpectralEmbedding(affinity='rbf', gamma=0.3)`
/// element-wise on the CONNECTED, DISTINCT-eigenvalue, ASYMMETRIC `line5` fixture
/// (tol 1e-6), after the fix (RBF diag=0 matching scipy's `csgraph_laplacian`,
/// `/dd` rescale, deterministic sign-flip).
///
/// sklearn returns `SKLEARN_EMBEDDING_2` (live oracle, R-CHAR-3). scipy's
/// `csgraph_laplacian(W, normed=True)` ignores the matrix diagonal (no
/// self-loops), so with `W_ii=0` the degree/`dd`/Laplacian — and hence the
/// embedding magnitudes — reproduce sklearn's `embedding = embedding / dd`
/// (`_spectral_embedding.py:443`) exactly.
///
/// Tracking: #1443 (blocker #1444) — RESOLVED.
#[test]
fn green_rbf_embedding_value_parity_2() {
    let se = SpectralEmbedding::new(2).with_affinity(Affinity::RBF { gamma: 0.3 });
    let fitted = fit_or_fail(&se, &line5());
    let emb = fitted.embedding();
    assert_eq!(emb.dim(), (5, 2), "embedding shape");

    for i in 0..5 {
        for j in 0..2 {
            let got = emb[[i, j]];
            let want = SKLEARN_EMBEDDING_2[i][j];
            assert!(
                (got - want).abs() < 1e-6,
                "embedding[{i},{j}]: ferrolearn={got} sklearn={want} (|diff|={})",
                (got - want).abs()
            );
        }
    }
}

/// Green-guard (REQ-1, 1-component bound): same ASYMMETRIC `line5` fixture with
/// `n_components=1`. On the asymmetric fixture the bottom eigenvector has a UNIQUE
/// max-abs entry, so the deterministic sign-flip argmax is unambiguous (no
/// faer-vs-numpy ULP tie as on the prior symmetric fixture) and ferrolearn's
/// embedding MATCHES sklearn `SKLEARN_EMBEDDING_1` element-wise (~1e-15, tol 1e-6).
///
/// sklearn's `_deterministic_vector_sign_flip` runs per-column independently, so
/// the 1-component sign need NOT match the first column of the 2-component
/// embedding — both are pinned against their own live oracles.
///
/// Tracking: #1443 (blocker #1444) — RESOLVED (sign-tie avoided via asymmetry).
#[test]
fn green_rbf_embedding_value_parity_1() {
    let se = SpectralEmbedding::new(1).with_affinity(Affinity::RBF { gamma: 0.3 });
    let fitted = fit_or_fail(&se, &line5());
    let emb = fitted.embedding();
    assert_eq!(emb.dim(), (5, 1), "embedding shape");

    for i in 0..5 {
        let got = emb[[i, 0]];
        let want = SKLEARN_EMBEDDING_1[i];
        assert!(
            (got - want).abs() < 1e-6,
            "embedding[{i},0]: ferrolearn={got} sklearn={want} (|diff|={})",
            (got - want).abs()
        );
    }
}

// ---------------------------------------------------------------------------
// GREEN-GUARDS — structure + error contracts (must PASS against current code).
// Scale/sign-invariant: they hold regardless of the residual magnitude DIV.
// ---------------------------------------------------------------------------

/// Green-guard: embedding shape is `(n_samples, n_components)`.
#[test]
fn green_embedding_shape() {
    let se = SpectralEmbedding::new(2).with_affinity(Affinity::RBF { gamma: 0.5 });
    let fitted = fit_or_fail(&se, &line5());
    assert_eq!(fitted.embedding().dim(), (5, 2));
}

/// Green-guard: the 1-component RBF embedding of a simple COLINEAR fixture is
/// monotone along the line (a scale/sign-invariant ordering property: colinear
/// points map to a monotone 1-D coordinate up to a global sign). Kept on its own
/// colinear fixture so the ordering property is independent of the asymmetric
/// `line5()` value-parity fixture.
#[test]
fn green_rbf_orders_line() {
    let colinear: Array2<f64> = array![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]];
    let se = SpectralEmbedding::new(1).with_affinity(Affinity::RBF { gamma: 0.5 });
    let fitted = fit_or_fail(&se, &colinear);
    let emb = fitted.embedding();
    let v: Vec<f64> = (0..5).map(|i| emb[[i, 0]]).collect();
    let ascending = v.windows(2).all(|w| w[0] <= w[1]);
    let descending = v.windows(2).all(|w| w[0] >= w[1]);
    assert!(
        ascending || descending,
        "1-D embedding of colinear points should be monotone (up to sign): {v:?}"
    );
}

/// Green-guard: deterministic — same input yields identical embedding twice.
#[test]
fn green_determinism() {
    let se = SpectralEmbedding::new(2).with_affinity(Affinity::RBF { gamma: 0.5 });
    let a = fit_or_fail(&se, &line5());
    let b = fit_or_fail(&se, &line5());
    assert_eq!(a.embedding(), b.embedding());
}

/// Green-guard: error contracts (n_components / sample / affinity-param bounds).
#[test]
fn green_error_contracts() {
    let x = line5(); // 5 samples

    // n_components == 0 -> Err.
    assert!(SpectralEmbedding::new(0).fit(&x, &()).is_err());

    // n_components >= n_samples -> Err.
    assert!(SpectralEmbedding::new(5).fit(&x, &()).is_err());

    // n < 2 samples -> Err.
    let one = array![[1.0, 2.0]];
    assert!(SpectralEmbedding::new(1).fit(&one, &()).is_err());

    // RBF gamma <= 0 -> Err.
    assert!(
        SpectralEmbedding::new(1)
            .with_affinity(Affinity::RBF { gamma: 0.0 })
            .fit(&x, &())
            .is_err()
    );
    assert!(
        SpectralEmbedding::new(1)
            .with_affinity(Affinity::RBF { gamma: -1.0 })
            .fit(&x, &())
            .is_err()
    );

    // kNN n_neighbors == 0 -> Err.
    assert!(
        SpectralEmbedding::new(1)
            .with_affinity(Affinity::NearestNeighbors { n_neighbors: 0 })
            .fit(&x, &())
            .is_err()
    );

    // kNN n_neighbors >= n_samples -> Err.
    assert!(
        SpectralEmbedding::new(1)
            .with_affinity(Affinity::NearestNeighbors { n_neighbors: 5 })
            .fit(&x, &())
            .is_err()
    );
}
