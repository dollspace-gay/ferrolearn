//! Divergence pin: ferrolearn's brute `parallel_on_Y` reduction assumes
//! `chunks_n_threads == Y_n_chunks` (one OpenMP thread per Y-chunk), so it
//! merges the per-chunk heaps CHUNK-MAJOR (chunk 0's heap, then chunk 1's, ...).
//! sklearn instead sets
//!
//! ```text
//! chunks_n_threads = min(Y_n_chunks, effective_n_threads)
//! ```
//!
//! (`sklearn/metrics/_pairwise_distances_reduction/_base.pyx.tp:201-203`) and
//! drives the Y-chunks with `prange(Y_n_chunks, schedule='static')` over
//! `chunks_n_threads` threads (`_base.pyx.tp:312`). When
//! `Y_n_chunks > effective_n_threads` each thread accumulates a CONTIGUOUS BLOCK
//! of several Y-chunks into ONE heap, and `_parallel_on_Y_synchronize` merges
//! the main heap THREAD-MAJOR over `chunks_n_threads` threads, NOT chunk-major
//! over `Y_n_chunks` chunks
//! (`sklearn/metrics/_pairwise_distances_reduction/_argkmin.pyx.tp:237-261`):
//!
//! ```text
//! for thread_num in range(self.chunks_n_threads):   # NOT Y_n_chunks
//!     for jdx in range(self.k):
//!         heap_push(main, ..., heaps[thread_num][idx*k + jdx], ...)
//! ```
//!
//! ferrolearn's `brute_parallel_on_y` (`ferrolearn-neighbors/src/knn.rs`) does
//! `for c in 0..n_chunks { build hc; for jdx in 0..k push hc[jdx] }` — i.e. one
//! heap per chunk merged chunk-major. That coincides with sklearn ONLY when
//! `Y_n_chunks <= effective_n_threads` (then `chunks_n_threads == Y_n_chunks`,
//! one chunk per thread, thread order == chunk order). For
//! `Y_n_chunks > effective_n_threads` the grouping/merge order differs and the
//! exact-tie SET at the k-boundary diverges from THIS machine's default sklearn.
//!
//! ## Live sklearn 1.5.2 / numpy 2.4.5 oracle, this fixture
//!
//! ```text
//! effective_n_threads (this machine) = 14.
//! n = 3585, d = 20, k = 6  ->  Y_n_chunks = ceil(3585/256) = 15 > 14.
//! X = zeros((3585,20)); X[:,0] = 10.0 ;
//!   winners X[2850,0]=1.0  X[1824,0]=1.5  X[1090,0]=2.0  X[2955,0]=2.5 ;
//!   tie group X[418,0]=X[937,0]=X[2421,0]=X[3284,0]=5.0 .   Q = zeros((1,20)).
//! KNeighborsClassifier(n_neighbors=6).fit(X,y)._fit_method == 'brute'.
//! .kneighbors(Q)[1][0]:
//!
//!   DEFAULT (env unset, chunks_n_threads = min(15,14) = 14)
//!                       -> [2850, 1824, 1090, 2955, 418, 2421]
//!   OMP_NUM_THREADS=1   -> [2850, 1824, 1090, 2955, 937, 2421]
//!   OMP_NUM_THREADS>=15 -> [2850, 1824, 1090, 2955, 937, 2421]
//! ```
//!
//! The four strict winners (1.0/1.5/2.0/2.5) and all six distances are identical
//! across thread counts; only the two surviving members of the four-way 5.0 tie
//! at the k-boundary differ. ferrolearn returns
//! `[2850, 1824, 1090, 2955, 937, 2421]` — the one-chunk-per-thread answer that
//! sklearn produces only for `OMP_NUM_THREADS >= Y_n_chunks`, NOT the default a
//! user on a 14-core box sees.
//!
//! Because this is the `Y_n_chunks > effective_n_threads` regime, sklearn's
//! tie SET is a function of `effective_n_threads`. This pin therefore computes
//! the expected SET from the LIVE sklearn oracle at runtime (system `python3`)
//! on whatever machine runs it, and asserts ferrolearn matches that default —
//! it does NOT hardcode a machine-specific constant, and the expected value is
//! NEVER copied from ferrolearn (R-CHAR-3). On a machine whose
//! `effective_n_threads >= 15` ferrolearn would coincide with sklearn and the
//! test self-skips (it asserts only when the oracle is in the divergence regime).
//!
//! Tracking: #2145.

use std::collections::HashSet;
use std::process::Command;

use ferrolearn_core::traits::Fit;
use ferrolearn_neighbors::{Algorithm, KNeighborsClassifier};
use ndarray::{Array1, Array2};

const N: usize = 3585;
const D: usize = 20;
const K: usize = 6;
const WINNERS: [(usize, f64); 4] = [(2850, 1.0), (1824, 1.5), (1090, 2.0), (2955, 2.5)];
const TIES: [usize; 4] = [418, 937, 2421, 3284];

/// Explicit (no-RNG) reconstruction of the oracle fixture, identical on the
/// ferrolearn and sklearn sides.
fn build_x() -> Array2<f64> {
    let mut x = Array2::<f64>::zeros((N, D));
    for i in 0..N {
        x[[i, 0]] = 10.0;
    }
    for (idx, val) in WINNERS {
        x[[idx, 0]] = val;
    }
    for idx in TIES {
        x[[idx, 0]] = 5.0;
    }
    x
}

/// Live sklearn 1.5.2 oracle: build the SAME fixture in numpy, fit a default
/// `KNeighborsClassifier(K)`, and return `kneighbors(Q)` indices + the machine's
/// `effective_n_threads` and `Y_n_chunks`. Expected value comes from sklearn,
/// never from ferrolearn (R-CHAR-3).
fn sklearn_oracle() -> Option<(Vec<usize>, usize, usize)> {
    let winners_py: String = WINNERS
        .iter()
        .map(|(i, v)| format!("({i},{v})"))
        .collect::<Vec<_>>()
        .join(",");
    let ties_py: String = TIES
        .iter()
        .map(|i| i.to_string())
        .collect::<Vec<_>>()
        .join(",");
    let script = format!(
        r#"
import math
import numpy as np
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads as eff
assert sklearn.__version__ == "1.5.2", sklearn.__version__
n, d, k = {N}, {D}, {K}
X = np.zeros((n, d), dtype=np.float64)
X[:, 0] = 10.0
for ix, v in [{winners_py}]:
    X[int(ix), 0] = v
for ix in [{ties_py}]:
    X[int(ix), 0] = 5.0
clf = KNeighborsClassifier(n_neighbors=k).fit(X, np.zeros(n, dtype=int))
assert clf._fit_method == "brute", clf._fit_method
ind = clf.kneighbors(np.zeros((1, d), dtype=np.float64))[1][0].tolist()
C = eff()
Y_n_chunks = math.ceil(n / 256)
print(",".join(str(i) for i in ind))
print(C)
print(Y_n_chunks)
"#
    );
    let out = Command::new("python3").arg("-c").arg(script).output().ok()?;
    if !out.status.success() {
        eprintln!(
            "sklearn oracle unavailable; skipping. stderr:\n{}",
            String::from_utf8_lossy(&out.stderr)
        );
        return None;
    }
    let text = String::from_utf8(out.stdout).ok()?;
    let mut lines = text.lines();
    let ind: Vec<usize> = lines
        .next()?
        .split(',')
        .map(|s| s.trim().parse().unwrap())
        .collect();
    let c: usize = lines.next()?.trim().parse().ok()?;
    let y_n_chunks: usize = lines.next()?.trim().parse().ok()?;
    Some((ind, c, y_n_chunks))
}

/// Divergence: in the `Y_n_chunks > effective_n_threads` regime ferrolearn's
/// chunk-major per-chunk-heap merge resolves the k-boundary exact tie
/// differently from THIS machine's DEFAULT sklearn (which groups multiple
/// Y-chunks per thread and merges thread-major).
///
/// Tracking: #2145.
#[test]
fn divergence_knn_multichunk_chunks_gt_cores_tie_set() {
    let Some((sklearn_ind, c, y_n_chunks)) = sklearn_oracle() else {
        // Oracle missing — cannot pin; treat as inconclusive rather than green.
        panic!("live sklearn 1.5.2 oracle required to pin this divergence");
    };

    // This divergence only manifests when sklearn groups chunks (chunks > cores).
    // If this machine has enough cores that Y_n_chunks <= C, sklearn coincides
    // with ferrolearn's one-chunk-per-thread merge — nothing to pin here.
    if y_n_chunks <= c {
        eprintln!(
            "machine has effective_n_threads={c} >= Y_n_chunks={y_n_chunks}; \
             chunks>cores regime UNREACHABLE on this box — divergence not exercised."
        );
        return;
    }

    let x = build_x();
    let y = Array1::from_iter(0..N);
    let q = Array2::<f64>::zeros((1, D));

    let fitted = KNeighborsClassifier::<f64>::new()
        .with_n_neighbors(K)
        .with_algorithm(Algorithm::Auto)
        .fit(&x, &y)
        .unwrap();
    let (_dist, indices) = fitted.kneighbors(&q, Some(K)).unwrap();
    let ferro_ind: Vec<usize> = indices.row(0).to_vec();

    let ferro_set: HashSet<usize> = ferro_ind.iter().copied().collect();
    let sk_set: HashSet<usize> = sklearn_ind.iter().copied().collect();

    assert_eq!(
        ferro_set, sk_set,
        "Y_n_chunks={y_n_chunks} > effective_n_threads={c} (chunks>cores regime): \
         ferrolearn kneighbors {ferro_ind:?} must equal THIS machine's default \
         sklearn KNeighborsClassifier({K}).kneighbors() {sklearn_ind:?}. \
         ferrolearn assumes one chunk per thread (chunk-major merge) and resolves \
         the four-way distance-5.0 tie at the k-boundary as the OMP>=Y_n_chunks \
         answer, but sklearn's default groups Y-chunks across {c} threads and \
         merges thread-major, keeping a different pair of the tied indices."
    );
}
