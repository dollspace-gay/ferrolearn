//! Divergence pin: ferrolearn's brute-regime `kneighbors` ALWAYS uses the
//! `parallel_on_Y` reduction (`brute_parallel_on_y`, one heap per Y-chunk-thread
//! over the training set, merged thread-major). sklearn's `ArgKmin.compute` is
//! called with `strategy="auto"` (`sklearn/neighbors/_base.py:848-856`), and the
//! `auto` resolver (`sklearn/metrics/_pairwise_distances_reduction/_base.pyx.tp:169-197`)
//! switches to `parallel_on_X` whenever
//!
//! ```text
//!   n_samples_Y < n_samples_X                         (line 183-189)   OR
//!   4 * chunk_size * effective_n_threads < n_samples_X (line 190-196)
//! ```
//!
//! where, for `kneighbors`, `X = the queries` and `Y = self._fit_X` (the
//! training set) (`_base.py:849-851`). i.e. when the number of QUERIES exceeds
//! the number of TRAINING samples (or exceeds `4*256*effective_n_threads`),
//! sklearn parallelises over X and each query is reduced with a SINGLE global
//! max-heap over all of Y in training-index order (`parallel_on_X`), which
//! resolves an exact distance tie at the k-th boundary DIFFERENTLY from the
//! per-chunk-thread `parallel_on_Y` merge.
//!
//! ferrolearn ignores the query count entirely: `brute_parallel_on_y`
//! (`ferrolearn-neighbors/src/knn.rs`) is applied per query regardless of how
//! many queries are passed, so it returns the `parallel_on_Y` order even in the
//! regime where sklearn's DEFAULT `auto` strategy reaches `parallel_on_X`.
//!
//! ## Live sklearn 1.5.2 / numpy 2.4.5 oracle (system python3), this fixture
//!
//! RNG-free fixture: `n_train = 3585`, `d = 20`, palette period 5 in coord 0
//! (`X[i,0] = [1,2,3,4,5][i % 5]`, all other coords 0), `Q = zeros((n_q, 20))`.
//! `k = 7`, so the 7 nearest are all at squared distance 1.0 (the 717 members of
//! the `1.0` level, indices 0,5,10,...): a 717-way exact tie whose surviving 7
//! members + order is entirely strategy-determined.
//!
//! ```text
//! KNeighborsClassifier(n_neighbors=7).fit(X, y)._fit_method == 'brute'.
//! .kneighbors(Q)[1][0]:
//!
//!   n_q = 1     (auto -> parallel_on_Y)  -> [10, 5, 25, 20, 15, 0, 30]
//!   n_q = 4000  (auto -> parallel_on_X)  -> [ 5, 30, 15, 25, 10, 20, 0]
//! ```
//!
//! `n_q = 4000 > n_train = 3585` triggers the `n_samples_Y < n_samples_X` switch
//! to `parallel_on_X`. The SET `{0,5,10,15,20,25,30}` is identical for both
//! strategies; only the ORDER differs. ferrolearn returns the `parallel_on_Y`
//! order `[10, 5, 25, 20, 15, 0, 30]` for EVERY query regardless of `n_q`, so it
//! diverges from the DEFAULT sklearn order whenever `n_q` crosses the
//! parallel_on_X threshold.
//!
//! The expected ORDER is taken from the LIVE sklearn oracle at runtime on the
//! machine that runs this test (R-CHAR-3: never copied from ferrolearn). The
//! pin asserts only in the divergence regime: it queries the oracle for both
//! `n_q = 1` (parallel_on_Y baseline, which ferrolearn must already match) and
//! the large `n_q` (parallel_on_X), and requires ferrolearn's large-`n_q`
//! output to equal the oracle's large-`n_q` (parallel_on_X) order. On a machine
//! where `4 * 256 * effective_n_threads < n_q` does NOT hold AND `n_q <=
//! n_train` (so sklearn would NOT switch), the test self-skips.
//!
//! Tracking: #2146.

use std::process::Command;

use ferrolearn_core::traits::Fit;
use ferrolearn_neighbors::{Algorithm, KNeighborsClassifier};
use ndarray::{Array1, Array2};

const N_TRAIN: usize = 3585;
const D: usize = 20;
const K: usize = 7;
const N_Q: usize = 4000; // > N_TRAIN -> sklearn auto switches to parallel_on_X
const PALETTE: [f64; 5] = [1.0, 2.0, 3.0, 4.0, 5.0];

/// RNG-free reconstruction of the training matrix, identical on both sides.
fn build_x() -> Array2<f64> {
    let mut x = Array2::<f64>::zeros((N_TRAIN, D));
    for i in 0..N_TRAIN {
        x[[i, 0]] = PALETTE[i % PALETTE.len()];
    }
    x
}

/// Live sklearn oracle: build the SAME fixture in numpy, fit a default
/// `KNeighborsClassifier(K)`, and return the `kneighbors` order for row 0 at the
/// given `n_q`, plus the resolved strategy + `effective_n_threads`. Expected
/// value comes from sklearn, never from ferrolearn (R-CHAR-3).
fn sklearn_oracle(n_q: usize) -> Option<(Vec<usize>, String, usize)> {
    let script = format!(
        r#"
import numpy as np, sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads as eff
assert sklearn.__version__ == "1.5.2", sklearn.__version__
n, d, k = {N_TRAIN}, {D}, {K}
pal = {PALETTE:?}
X = np.zeros((n, d), dtype=np.float64)
for i in range(n):
    X[i, 0] = pal[i % len(pal)]
clf = KNeighborsClassifier(n_neighbors=k).fit(X, np.arange(n) % 2)
assert clf._fit_method == "brute", clf._fit_method
Q = np.ascontiguousarray(np.zeros(({n_q}, d), dtype=np.float64))
ind = clf.kneighbors(Q)[1][0].tolist()
E = eff()
nY, nX = n, {n_q}
strat = "parallel_on_Y"
if nY < nX or 4 * 256 * E < nX:
    strat = "parallel_on_X"
print(",".join(str(i) for i in ind))
print(strat)
print(E)
"#
    );
    let out = Command::new("python3")
        .arg("-c")
        .arg(script)
        .output()
        .ok()?;
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
    let strat = lines.next()?.trim().to_string();
    let eff: usize = lines.next()?.trim().parse().ok()?;
    Some((ind, strat, eff))
}

fn ferro_order(n_q: usize) -> Vec<usize> {
    let x = build_x();
    let y = Array1::from_iter((0..N_TRAIN).map(|i| i % 2));
    let fitted = KNeighborsClassifier::<f64>::new()
        .with_n_neighbors(K)
        .with_algorithm(Algorithm::Auto)
        .fit(&x, &y)
        .unwrap();
    let xq = Array2::<f64>::zeros((n_q, D));
    let (_dist, ind) = fitted.kneighbors(&xq, Some(K)).unwrap();
    ind.row(0).to_vec()
}

/// Divergence: with `n_q > n_train`, sklearn's `auto` strategy resolves to
/// `parallel_on_X` and returns a tie ORDER that ferrolearn's always-on
/// `parallel_on_Y` reduction does not reproduce.
#[test]
fn divergence_kneighbors_strategy_parallel_on_x_order() {
    let Some((sk_big, strat_big, eff)) = sklearn_oracle(N_Q) else {
        eprintln!("oracle unavailable; skipping");
        return;
    };

    // Only assert in the divergence regime: sklearn must actually be in
    // parallel_on_X for this n_q on this machine.
    if strat_big != "parallel_on_X" {
        eprintln!(
            "machine eff={eff}: n_q={N_Q} does not trigger parallel_on_X \
             (strat={strat_big}); skipping"
        );
        return;
    }

    // Sanity: at n_q=1 sklearn is parallel_on_Y and ferrolearn already matches
    // it (the #2145 fix). This documents that the divergence is purely the
    // strategy switch, not a broken parallel_on_Y.
    if let Some((sk_small, strat_small, _)) = sklearn_oracle(1) {
        assert_eq!(strat_small, "parallel_on_Y");
        assert_eq!(
            ferro_order(1),
            sk_small,
            "baseline: ferrolearn must match sklearn parallel_on_Y at n_q=1"
        );
    }

    let ferro_big = ferro_order(N_Q);

    // SET must match (this is a pure ORDER divergence).
    let mut sk_sorted = sk_big.clone();
    sk_sorted.sort_unstable();
    let mut ferro_sorted = ferro_big.clone();
    ferro_sorted.sort_unstable();
    assert_eq!(
        ferro_sorted, sk_sorted,
        "SET must match; only ORDER should diverge"
    );

    assert_eq!(
        ferro_big, sk_big,
        "n_q={N_Q} > n_train={N_TRAIN}: sklearn auto -> parallel_on_X order \
         {sk_big:?}, but ferrolearn returns the parallel_on_Y order \
         {ferro_big:?} (it never switches strategy on query count)."
    );
}
