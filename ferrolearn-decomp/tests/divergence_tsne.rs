// Index-based loops over parallel centroid/spread arrays for clarity.
#![allow(clippy::needless_range_loop)]

//! Divergence audit for `Tsne` / `FittedTsne`
//! (`ferrolearn-decomp/src/tsne.rs`) against scikit-learn 1.5.2
//! `class TSNE` (`sklearn/manifold/_t_sne.py:563`).
//!
//! Design doc: `.design/decomp/tsne.md` (4 SHIPPED / 11 NOT-STARTED,
//! tracking #1596).
//!
//! ## Embedding VALUE carve-out (REQ-5, #1597 — NO failing test, R-DEFER-3)
//!
//! Exact `embedding_` / `kl_divergence_` VALUES diverge element-wise from sklearn
//! because of FOUR independent factors, NONE of which is a fixable numeric bug:
//!   (a) init: sklearn defaults `init='pca'` (deterministic randomized PCA rescaled
//!       `std(PC1)=1e-4`, `_t_sne.py:1019-1030`); ferrolearn ALWAYS draws a random
//!       Gaussian `Normal(0, 1e-4)` from a Rust `Xoshiro256PlusPlus`
//!       (`tsne.rs:601-608`) — a different start AND a different PRNG than even
//!       sklearn's `init='random'` numpy `standard_normal` (`_t_sne.py:1031-1036`).
//!   (b) non-convexity: the t-SNE objective is explicitly non-convex
//!       (`_t_sne.py:570-571`) — different starts reach different local minima.
//!   (c) learning_rate: ferrolearn fixes `200.0` (`tsne.rs:87`); sklearn defaults
//!       `'auto' = max(N/early_exaggeration/4, 50)` (`_t_sne.py:876-879`).
//!   (d) Barnes-Hut: ferrolearn's dense-P custom k-d-tree (`BHTree` `tsne.rs:242`)
//!       vs sklearn's k-NN-sparse-P Cython `_barnes_hut_tsne.gradient`
//!       (`_t_sne.py:284`).
//! The embedding is identifiable only up to rotation/reflection/translation and the
//! local optimum reached. Element-wise parity is NOT pinned (same class as the
//! fast_ica / lda / minibatch_nmf RNG + local-optimum carve-outs). The MEANINGFUL,
//! observable correctness check is CLUSTER SEPARATION on well-separated data
//! (see `separates_clusters_knn_recovery`).
//!
//! ## `/2n` vs `/sum(P)` normalisation (REQ-10, #1602) — EQUIVALENT, NOT a divergence
//!
//! ferrolearn symmetrises `P_ij = (P_i|j + P_j|i)/(2n)` (`tsne.rs:480-483`); sklearn
//! does `P = cond_P + cond_P.T; P /= max(sum(P), eps)` (`_t_sne.py:68-70`).
//! ferrolearn's `compute_pij_row` (`tsne.rs:410`) normalises each conditional row to
//! sum 1 (`p[j] *= inv_sum`, `tsne.rs:443`), EXACTLY as sklearn's
//! `_binary_search_perplexity` does. The live oracle confirms each sklearn
//! conditional row sums to 1, hence `sum(cond_P + cond_P.T) = 2n`, so
//! `/sum(P) == /2n` EXACTLY. The post-normalisation `MACHINE_EPSILON`/`1e-12` clamp
//! is applied to off-diagonal entries on both sides; the diagonal is zeroed on both.
//! Therefore the two normalisations COINCIDE — this is NOT a fixable divergence. It
//! is also NOT directly testable: `compute_joint_probabilities` (`tsne.rs:467`) and
//! `compute_pij_row` (`tsne.rs:410`) are PRIVATE `fn`s internal to `fit`, and
//! `FittedTsne` exposes NO P accessor (only `embedding()`/`kl_divergence()`/
//! `n_iter()`, `tsne.rs:215-228`). P-affinity correctness is observable ONLY through
//! the structural cluster-separation check below.
//!
//! ## Conclusion
//!
//! Every VALUE candidate is carve-out gated (init RNG + non-convexity +
//! learning_rate + Barnes-Hut) or unobservable (private P). The `/2n` vs `/sum(P)`
//! candidate is EQUIVALENT (rows sum to 1). NO deterministic, observable,
//! non-RNG/non-local-optimum-gated numeric divergence with a clean R-CHAR-3 oracle
//! was found. This is a verify-and-document unit (same class as fast_ica / lda).
//! Everything below is a STRUCTURAL GREEN-GUARD that PASSES against current code and
//! pins contracts the generator must not regress.
//!
//! All expected structural facts come from the live sklearn 1.5.2 oracle (run from
//! `/tmp`), never literal-copied from ferrolearn (R-CHAR-3). See the module-level
//! oracle notes per test.

use ferrolearn_core::traits::Fit;
use ferrolearn_decomp::Tsne;
use ndarray::Array2;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use rand_xoshiro::Xoshiro256PlusPlus;

/// Build 3 well-separated Gaussian blobs in 5-D: `n_per` points each.
///
/// Mirrors the geometry of the sklearn oracle probe (3 centres at
/// `0`, `5`, `10`-patterned coordinates, sigma 0.3) for which sklearn
/// `TSNE(perplexity=5, init='random', learning_rate='auto')` yields a 3-NN
/// label-recovery accuracy of 1.0 on the embedding (live oracle, 2026, from /tmp):
///
/// ```text
/// sklearn knn acc on separated blobs: 1.0
/// ```
fn make_blobs(seed: u64, n_per: usize) -> (Array2<f64>, Vec<usize>) {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let normal = Normal::new(0.0, 0.3).expect("valid normal");
    let centers = [
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [5.0, 5.0, 5.0, 5.0, 5.0],
        [10.0, 0.0, 10.0, 0.0, 10.0],
    ];
    let n = centers.len() * n_per;
    let mut x = Array2::<f64>::zeros((n, 5));
    let mut labels = Vec::with_capacity(n);
    for (c_idx, center) in centers.iter().enumerate() {
        for i in 0..n_per {
            let row = c_idx * n_per + i;
            for (f, &c) in center.iter().enumerate() {
                x[[row, f]] = c + normal.sample(&mut rng);
            }
            labels.push(c_idx);
        }
    }
    (x, labels)
}

/// k-NN (k=3) majority-vote label-recovery accuracy on a 2-D embedding.
fn knn_accuracy(emb: &Array2<f64>, labels: &[usize]) -> f64 {
    let n = emb.nrows();
    let mut correct = 0usize;
    for i in 0..n {
        let mut dists: Vec<(f64, usize)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| {
                let mut d = 0.0;
                for dd in 0..emb.ncols() {
                    let diff = emb[[i, dd]] - emb[[j, dd]];
                    d += diff * diff;
                }
                (d, labels[j])
            })
            .collect();
        dists.sort_by(|a, b| a.0.partial_cmp(&b.0).expect("finite distances"));
        let mut votes = [0usize; 3];
        for &(_, lbl) in dists.iter().take(3) {
            votes[lbl] += 1;
        }
        let pred = votes
            .iter()
            .enumerate()
            .max_by_key(|&(_, v)| v)
            .map(|(idx, _)| idx)
            .expect("non-empty votes");
        if pred == labels[i] {
            correct += 1;
        }
    }
    correct as f64 / n as f64
}

// ---------------------------------------------------------------------------
// REQ-1 (structural) — embedding shape (n_samples, n_components).
// Oracle: sklearn TSNE(n_components=k).fit_transform(X).shape == (n_samples, k)
// (`_t_sne.py:1176-1177`, ClassNamePrefixFeaturesOutMixin embedding_).
// GREEN-GUARD: PASSES against current code.
// ---------------------------------------------------------------------------

#[test]
fn embedding_shape_is_n_samples_by_n_components_2d() {
    let x = Array2::<f64>::from_shape_fn((20, 5), |(i, j)| (i + 2 * j) as f64);
    let fitted = Tsne::new()
        .with_perplexity(5.0)
        .with_n_iter(50)
        .with_random_state(42)
        .fit(&x, &())
        .expect("fit succeeds on valid input");
    assert_eq!(fitted.embedding().dim(), (20, 2));
}

#[test]
fn embedding_shape_respects_n_components_3() {
    let x = Array2::<f64>::from_shape_fn((15, 4), |(i, j)| (i * j) as f64);
    let fitted = Tsne::new()
        .with_n_components(3)
        .with_perplexity(4.0)
        .with_n_iter(50)
        .with_random_state(42)
        .fit(&x, &())
        .expect("fit succeeds on valid input");
    assert_eq!(fitted.embedding().dim(), (15, 3));
}

// ---------------------------------------------------------------------------
// REQ-4 (structural) — kl_divergence_ finite & >= 0; n_iter_ >= 1.
// Oracle (live, /tmp): sklearn m.kl_divergence_ is a finite non-negative float
// and m.n_iter_ >= 1 (`_t_sne.py:1115`,`:1124`).
// GREEN-GUARD: PASSES against current code.
// ---------------------------------------------------------------------------

#[test]
fn kl_divergence_is_finite_and_non_negative() {
    let (x, _) = make_blobs(42, 12);
    let fitted = Tsne::new()
        .with_perplexity(5.0)
        .with_n_iter(100)
        .with_random_state(42)
        .fit(&x, &())
        .expect("fit succeeds on valid input");
    let kl = fitted.kl_divergence();
    assert!(kl.is_finite(), "kl_divergence must be finite, got {kl}");
    assert!(kl >= 0.0, "kl_divergence must be >= 0, got {kl}");
}

#[test]
fn n_iter_is_at_least_one() {
    let (x, _) = make_blobs(42, 10);
    let fitted = Tsne::new()
        .with_perplexity(5.0)
        .with_n_iter(50)
        .with_random_state(42)
        .fit(&x, &())
        .expect("fit succeeds on valid input");
    assert!(
        fitted.n_iter() >= 1,
        "n_iter_ must be >= 1, got {}",
        fitted.n_iter()
    );
}

// ---------------------------------------------------------------------------
// REQ-1/REQ-4 (structural — the MEANINGFUL "did t-SNE work" check).
// On well-separated high-D clusters the embedding SEPARATES them. This is the
// neighbour-preservation correctness signal (analogous to fast_ica's source
// recovery), NOT a value pin. Live sklearn oracle on the same geometry gives a
// 3-NN label-recovery accuracy of 1.0 (see `make_blobs` doc); sklearn's embedding
// satisfies the same structural property. Asserting > 0.8 (sklearn-confirmed
// threshold, the embedding values themselves are carve-out REQ-5).
// GREEN-GUARD: PASSES against current code.
// ---------------------------------------------------------------------------

#[test]
fn separates_clusters_knn_recovery() {
    let (x, labels) = make_blobs(42, 15);
    let fitted = Tsne::new()
        .with_perplexity(5.0)
        .with_n_iter(500)
        .with_random_state(42)
        .fit(&x, &())
        .expect("fit succeeds on valid input");
    let acc = knn_accuracy(fitted.embedding(), &labels);
    assert!(
        acc > 0.8,
        "t-SNE should separate well-separated clusters (sklearn oracle: 1.0); \
         got 3-NN accuracy {acc}"
    );
}

#[test]
fn separates_clusters_inter_centroid_dominates_intra_spread() {
    // Complementary structural separation metric: mean inter-centroid distance
    // >> mean intra-cluster spread. sklearn's embedding satisfies the same
    // (Probe 1: inter-centroid >> intra spread).
    let (x, labels) = make_blobs(7, 15);
    let fitted = Tsne::new()
        .with_perplexity(5.0)
        .with_n_iter(500)
        .with_random_state(7)
        .fit(&x, &())
        .expect("fit succeeds on valid input");
    let emb = fitted.embedding();
    let n = emb.nrows();
    let dim = emb.ncols();

    // Per-cluster centroids.
    let mut centroids = [[0.0f64; 2]; 3];
    let mut counts = [0usize; 3];
    for i in 0..n {
        for d in 0..dim {
            centroids[labels[i]][d] += emb[[i, d]];
        }
        counts[labels[i]] += 1;
    }
    for c in 0..3 {
        for d in 0..dim {
            centroids[c][d] /= counts[c] as f64;
        }
    }

    // Mean intra-cluster spread (point-to-own-centroid distance).
    let mut intra_sum = 0.0;
    for i in 0..n {
        let mut d2 = 0.0;
        for d in 0..dim {
            let diff = emb[[i, d]] - centroids[labels[i]][d];
            d2 += diff * diff;
        }
        intra_sum += d2.sqrt();
    }
    let intra = intra_sum / n as f64;

    // Mean inter-centroid distance.
    let mut inter_sum = 0.0;
    let mut pairs = 0;
    for a in 0..3 {
        for b in (a + 1)..3 {
            let mut d2 = 0.0;
            for d in 0..dim {
                let diff = centroids[a][d] - centroids[b][d];
                d2 += diff * diff;
            }
            inter_sum += d2.sqrt();
            pairs += 1;
        }
    }
    let inter = inter_sum / pairs as f64;

    assert!(
        inter > 3.0 * intra,
        "inter-centroid distance ({inter}) should dominate intra spread ({intra})"
    );
}

// ---------------------------------------------------------------------------
// REQ-1 (structural) — determinism: same random_state -> identical embedding.
// sklearn TSNE(random_state=k) is deterministic; ferrolearn's seeded Xoshiro
// init + deterministic GD must reproduce bit-for-bit.
// GREEN-GUARD: PASSES against current code.
// ---------------------------------------------------------------------------

#[test]
fn same_random_state_gives_identical_embedding() {
    let x = Array2::<f64>::from_shape_fn((12, 3), |(i, j)| (i + j) as f64);
    let f1 = Tsne::new()
        .with_perplexity(3.0)
        .with_n_iter(80)
        .with_random_state(42)
        .fit(&x, &())
        .expect("fit succeeds");
    let f2 = Tsne::new()
        .with_perplexity(3.0)
        .with_n_iter(80)
        .with_random_state(42)
        .fit(&x, &())
        .expect("fit succeeds");
    assert_eq!(f1.embedding().dim(), f2.embedding().dim());
    for (a, b) in f1.embedding().iter().zip(f2.embedding().iter()) {
        assert!(
            (a - b).abs() < 1e-12,
            "embeddings must be bit-identical for the same seed: {a} vs {b}"
        );
    }
}

// ---------------------------------------------------------------------------
// REQ-3 — error / parameter contracts. Pinning the ACTUAL ferrolearn guards.
// sklearn raises InvalidParameterError/ValueError for these; ferrolearn returns
// Err(FerroError). We pin that each guard rejects (not the error TYPE — that is a
// known NOT-STARTED contract divergence, REQ-3 FLAG, not pinned as failing).
// GREEN-GUARD: PASSES against current code.
// ---------------------------------------------------------------------------

#[test]
fn rejects_n_components_zero() {
    let x = Array2::<f64>::zeros((10, 3));
    // ferrolearn guard: tsne.rs:554-559 InvalidParameter{name:"n_components"}.
    assert!(Tsne::new().with_n_components(0).fit(&x, &()).is_err());
}

#[test]
fn rejects_fewer_than_two_samples() {
    let x = Array2::<f64>::zeros((1, 3));
    // ferrolearn guard: tsne.rs:560-566 InsufficientSamples{required:2}.
    assert!(Tsne::new().fit(&x, &()).is_err());
}

#[test]
fn rejects_non_positive_perplexity() {
    let x = Array2::<f64>::zeros((10, 3));
    // ferrolearn guard: tsne.rs:567-572 perplexity must be positive.
    assert!(Tsne::new().with_perplexity(0.0).fit(&x, &()).is_err());
}

#[test]
fn rejects_perplexity_not_less_than_n_samples() {
    let x = Array2::<f64>::zeros((10, 3));
    // ferrolearn guard: tsne.rs:573-581 perplexity must be < n_samples.
    // Mirrors sklearn _check_params_vs_input "perplexity must be less than
    // n_samples" (`_t_sne.py:862-864`). perplexity == n_samples must reject.
    assert!(Tsne::new().with_perplexity(10.0).fit(&x, &()).is_err());
}

#[test]
fn rejects_non_positive_learning_rate() {
    let x = Array2::<f64>::zeros((10, 3));
    // ferrolearn guard: tsne.rs:582-587 learning_rate must be positive.
    assert!(Tsne::new().with_learning_rate(-1.0).fit(&x, &()).is_err());
}

#[test]
fn rejects_negative_theta() {
    let x = Array2::<f64>::zeros((10, 3));
    // ferrolearn guard: tsne.rs:588-593 theta must be non-negative.
    assert!(Tsne::new().with_theta(-0.1).fit(&x, &()).is_err());
}

// ---------------------------------------------------------------------------
// REQ-10 (structural) — theta=0 exact-gradient path still produces a valid
// (n, n_components) embedding (exercises exact_gradient, tsne.rs:759).
// GREEN-GUARD: PASSES against current code.
// ---------------------------------------------------------------------------

#[test]
fn exact_gradient_path_produces_valid_embedding() {
    let x = Array2::<f64>::from_shape_fn((10, 3), |(i, j)| (i + j) as f64);
    let fitted = Tsne::new()
        .with_theta(0.0)
        .with_perplexity(3.0)
        .with_n_iter(50)
        .with_random_state(42)
        .fit(&x, &())
        .expect("fit succeeds");
    assert_eq!(fitted.embedding().dim(), (10, 2));
    assert!(fitted.kl_divergence().is_finite());
}
