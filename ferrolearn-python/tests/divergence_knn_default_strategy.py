"""Divergence pins: `ferrolearn.KNeighborsClassifier` diverges from the DEFAULT
`sklearn.neighbors.KNeighborsClassifier` on exact distance ties.

Re-audit of the #2139/#2141 `NeighborsHeap` fix
(`ferrolearn-neighbors/src/knn.rs` `heap_push` + `simultaneous_sort` +
`neighbors_heap_select`, the f64 path of `find_neighbors`). The fix ports
sklearn's single max-heap fill (`sklearn/utils/_heap.pyx:6-85` +
`sklearn/utils/_sorting.pyx:18-93`). That reproduces sklearn's `ArgKmin`
`parallel_on_X` order and the `kd_tree`/`ball_tree` order.

BUT the DEFAULT sklearn user (`algorithm='auto'`) does NOT get that order/set:

  * For small n the default resolves to `brute`, and brute's own `auto`
    strategy resolves to `parallel_on_Y` (verified via
    `ArgKmin.compute(..., strategy='auto')`), giving a DIFFERENT tie ORDER than
    the single-heap (`parallel_on_X`) path ferrolearn ports.
  * For n >= ~50 the default resolves to `kd_tree`, whose tie SET differs from
    the single-heap brute set, flipping `predict`/`predict_proba`.

The builder's doc-comment caveat claims the differing (`parallel_on_Y`) order
is "a non-canonical, chunk/thread-size dependent artifact" that is "not a stable
contract". That is REFUTED here: the live sklearn DEFAULT output is bit-stable
across `OMP_NUM_THREADS in {1,4,8}` and `n_jobs in {1,4,8}` (see the
adversarial-audit evidence). It is the user-facing contract, and ferrolearn must
match it.

Verification model B (goal.md): every expected value is the LIVE sklearn 1.5.2
oracle computed in-test; nothing is literal-copied from the ferrolearn side
(R-CHAR-3).

Tracking: #2143.
"""

import numpy as np

import ferrolearn
from sklearn.neighbors import KNeighborsClassifier as SkKNN


# n=81 integer-grid fixture (auto -> kd_tree for sklearn). At query [0,0] the
# k=6 boundary lands in a many-way exact tie at squared-distance 2.0. sklearn's
# DEFAULT keeps idx68; ferrolearn's single-heap keeps idx36 instead — a SET
# difference. Labels are chosen so that swapped member flips the majority vote.
# This fixture was discovered by the adversarial audit; the EXPECTED prediction
# is computed live from sklearn below, not copied.
_X = np.array(
    [[0, -3], [1, -1], [0, -1], [-1, -3], [-1, 2], [-3, 3], [-1, 0], [-1, 0],
     [2, 2], [3, 1], [-1, 2], [1, -1], [-1, 1], [3, 3], [0, 1], [2, 3], [2, -3],
     [1, 1], [-3, -2], [1, -1], [2, 1], [2, -1], [3, 2], [1, -3], [0, 2], [2, 1],
     [-1, 2], [2, 2], [-2, -2], [3, 0], [2, -3], [-2, -3], [0, 2], [-2, 0],
     [-1, 3], [2, 3], [1, 0], [3, -2], [2, 0], [3, 3], [-3, 3], [0, 2], [3, -3],
     [1, -1], [-2, 3], [-1, 0], [-1, 2], [2, -1], [1, -2], [3, 1], [-1, -2],
     [0, -2], [-1, -1], [0, -2], [2, 1], [3, 2], [3, -1], [-1, 1], [1, -2],
     [0, 3], [-1, 1], [-2, -1], [1, 1], [2, -3], [0, -2], [-1, 3], [2, -2],
     [0, 0], [0, -1], [-2, 0], [2, -3], [-1, 1], [3, 3], [3, -1], [-3, 0],
     [-2, -3], [1, 3], [3, 2], [3, -1], [1, -2], [2, -3]],
    dtype=float,
)
_Y = np.array(
    [0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0],
    dtype=int,
)
_Q = np.array([[0.0, 0.0]], dtype=float)
_K = 6


def test_divergence_knn_kd_regime_set_flips_predict():
    """ferrolearn's single-heap k-NN SET (== sklearn brute) differs from the
    DEFAULT sklearn (kd_tree) SET on n>=50 tied data, flipping predict.

    sklearn default `auto` resolves to `kd_tree` here; its tie set keeps idx68,
    ferrolearn keeps idx36 — a different label, so the majority vote flips.
    Tracking: #2143.
    """
    sk = SkKNN(n_neighbors=_K).fit(_X, _Y)
    assert sk._fit_method == "kd_tree", (
        "fixture precondition: default sklearn must resolve to kd_tree for "
        "this n so the SET divergence regime is exercised"
    )
    sk_pred = sk.predict(_Q)

    f = ferrolearn.KNeighborsClassifier(n_neighbors=_K).fit(_X, _Y)
    f_pred = np.asarray(f.predict(_Q))

    assert f_pred.tolist() == sk_pred.tolist(), (
        f"ferrolearn predict {f_pred.tolist()} must match DEFAULT sklearn "
        f"predict {sk_pred.tolist()}; ferrolearn's single-heap (parallel_on_X) "
        f"k-NN set diverges from sklearn's default kd_tree set on this exact-tie "
        f"fixture (idx36 vs idx68)"
    )


def test_divergence_knn_kd_regime_proba_differs():
    """predict_proba likewise diverges: the SET difference shifts the class
    weights. Expected proba is the live sklearn default oracle. Tracking: #2143.
    """
    sk_proba = SkKNN(n_neighbors=_K).fit(_X, _Y).predict_proba(_Q)
    f = ferrolearn.KNeighborsClassifier(n_neighbors=_K).fit(_X, _Y)
    f_proba = np.asarray(f.predict_proba(_Q))

    np.testing.assert_allclose(
        f_proba,
        sk_proba,
        atol=1e-9,
        err_msg=(
            "ferrolearn predict_proba must match DEFAULT sklearn (kd_tree) "
            "predict_proba; the single-heap tie SET diverges (idx36 vs idx68)"
        ),
    )


# Small-n ORDER divergence (the builder's own example fixture). auto -> brute ->
# parallel_on_Y for sklearn; ferrolearn ports parallel_on_X. SET matches so
# uniform predict agrees, but predict_proba ordering is unaffected — this pin is
# documented via the Rust kneighbors test
# (ferrolearn-neighbors/tests/divergence_knn_order_parallel_on_y.rs). The
# pytest surface here cannot observe order alone because the shim exposes no
# `kneighbors`; recorded for completeness.
def test_default_strategy_is_stable_oracle_fact():
    """Oracle fact backing the audit: sklearn's DEFAULT k-NN order on the
    builder's caveat fixture is deterministic (NOT thread-dependent), refuting
    the 'parallel_on_Y is a non-stable artifact' caveat. This asserts only the
    sklearn side (it is a fact about the oracle, used to justify #2143); it is
    not a ferrolearn comparison and always passes. Tracking: #2143.
    """
    X = np.array(
        [[1, 0], [-1, 0], [0, 1], [0, -1], [2, 0], [-2, 0], [0, 2]], dtype=float
    )
    y = np.arange(7)
    q = np.array([[0.0, 0.0]])
    orders = {
        tuple(SkKNN(n_neighbors=5, n_jobs=nj).fit(X, y).kneighbors(q)[1][0].tolist())
        for nj in (1, 4, 8)
    }
    assert orders == {(0, 2, 3, 1, 4)}, (
        "sklearn default kneighbors order must be the deterministic "
        "parallel_on_Y order [0,2,3,1,4] across n_jobs"
    )
