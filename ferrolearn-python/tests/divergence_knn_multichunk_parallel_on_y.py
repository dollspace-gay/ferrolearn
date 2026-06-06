"""Divergence pin: `ferrolearn.KNeighborsClassifier` diverges from the DEFAULT
multi-core `sklearn.neighbors.KNeighborsClassifier` in the BRUTE regime for
`n_samples > 256` on exact distance ties.

Audit of the #2143 default-backend fix
(`ferrolearn-neighbors/src/knn.rs` `brute_parallel_on_y` + `find_neighbors`).
The fix ports sklearn's `parallel_on_Y` reduction
(`sklearn/metrics/_pairwise_distances_reduction/_argkmin.pyx.tp`) as a SINGLE
per-query max-heap filled over ALL `n` training points in global index order,
then a second pass through that heap's slots
(`_parallel_on_Y_synchronize`-style double heap).

That single-global-heap model is only correct when sklearn uses ONE
chunk-thread, i.e. `n_samples <= pairwise_dist_chunk_size` (256) OR
`OMP_NUM_THREADS=1`. For `n_samples > 256` sklearn splits Y into
`Y_n_chunks = ceil(n / 256)` chunks and, on any machine with >= 2 hardware
threads (the universal default), builds ONE heap PER chunk-thread and merges
them in `_parallel_on_Y_synchronize`
(`_argkmin.pyx.tp:237-261`, thread_num-major then jdx). That per-chunk-then-
merge reduction resolves an exact distance tie at the k-th boundary differently
from a single global-index-order heap.

The builder's `find_neighbors` doc-comment claims the brute path "matches the
live sklearn 1.5.2 oracle ... bit-for-bit" and asserts (for `n < 256`) "there is
exactly one Y-chunk and one thread, so the result is fully deterministic" — true
for `n < 256`, but the implementation extends that single-heap model to
`n > 256`, where sklearn's default is the multi-chunk merge. ferrolearn returns
the `OMP_NUM_THREADS=1` tie resolution that a default multi-core sklearn user
never sees.

Verification model B (goal.md): the expected value is the LIVE sklearn 1.5.2
oracle computed in-test (default thread pool — the multi-core contract); nothing
is literal-copied from the ferrolearn side (R-CHAR-3).

Tracking: #2.
"""

import numpy as np

import ferrolearn
from sklearn.neighbors import KNeighborsClassifier as SkKNN


def _fixture():
    """Explicit (no-RNG) brute-regime fixture, n=400 > 256 -> 2 Y-chunks.

    X is all zero except column 0 = 10.0, with four "sure winner" overrides at
    indices 5/250/260/390 (distances 1/2/3/4 from the origin query). Every other
    training point is at distance exactly 10.0 — a large exact tie, of which
    k - 4 members fill the neighbor list. n_features = 20 > 15 forces `brute`
    (`sklearn/neighbors/_base.py:610-618`).
    """
    n, d = 400, 20
    X = np.zeros((n, d), dtype=np.float64)
    X[:, 0] = 10.0
    for idx, val in [(5, 1.0), (250, 2.0), (260, 3.0), (390, 4.0)]:
        X[idx, 0] = val
    # Labels: distinct classes on the two members the backends disagree on, so
    # the tie-SET difference is observable in predict_proba. sklearn keeps idx 3
    # (class 1), ferrolearn keeps idx 0 (class 0); the six shared neighbors are
    # class 2.
    y = np.full(n, 2, dtype=np.int64)
    y[0] = 0
    y[3] = 1
    Q = np.zeros((1, d), dtype=np.float64)
    return X, y, Q


def test_divergence_multichunk_predict_matches_sklearn():
    """sklearn DEFAULT (multi-core) brute classifier at n=400>256 produces a
    full class-probability vector that the n>256 tie SET determines; this guards
    the observable downstream of the kneighbors tie SET via the binding's
    public surface (`predict` / `predict_proba`; `kneighbors` is not bound).

    `predict` (argmax) happens to agree here (the six shared neighbors dominate),
    so this asserts the full neighbor-derived class counts via `predict_proba` in
    the companion test; here we additionally assert ferrolearn's predicted label
    equals sklearn's so the pin documents the argmax-stable / proba-flipped
    split explicitly."""
    X, y, Q = _fixture()
    k = 8

    sk = SkKNN(n_neighbors=k).fit(X, y)
    assert sk._fit_method == "brute"
    sk_pred = sk.predict(Q)

    fl = ferrolearn.KNeighborsClassifier(n_neighbors=k).fit(X, y)
    fl_pred = np.asarray(fl.predict(Q))

    # predict argmax is class 2 for both (the 6 shared neighbors); the divergence
    # is in the *probabilities* (companion test). This assert documents that the
    # label is the stable part of the split.
    np.testing.assert_array_equal(fl_pred, sk_pred)


def test_divergence_multichunk_predict_proba_flip():
    """The tie-SET difference flips `predict_proba` between ferrolearn and the
    DEFAULT multi-core sklearn classifier."""
    X, y, Q = _fixture()
    k = 8

    sk = SkKNN(n_neighbors=k).fit(X, y)
    sk_proba = sk.predict_proba(Q)

    fl = ferrolearn.KNeighborsClassifier(n_neighbors=k).fit(X, y)
    fl_proba = np.asarray(fl.predict_proba(Q))

    np.testing.assert_allclose(
        fl_proba,
        sk_proba,
        rtol=0,
        atol=0,
        err_msg=(
            "predict_proba diverges: sklearn DEFAULT multi-core keeps tie member "
            f"idx 3 (class 1) -> {sk_proba.tolist()}; ferrolearn keeps idx 0 "
            f"(class 0) -> {fl_proba.tolist()}. The n>256 brute parallel_on_Y "
            "multi-chunk tie resolution differs from ferrolearn's single-heap "
            "(OMP_NUM_THREADS=1) resolution."
        ),
    )
