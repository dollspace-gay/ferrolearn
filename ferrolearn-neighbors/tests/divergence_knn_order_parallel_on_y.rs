//! Divergence pin: ferrolearn's `kneighbors` exact-tie ORDER diverges from
//! sklearn's DEFAULT user-facing `KNeighborsClassifier(...).kneighbors()`.
//!
//! The builder's #2139/#2141 fix ports sklearn's `NeighborsHeap`
//! (`heap_push` + `simultaneous_sort`) as a single max-heap filled in index
//! order. That reproduces sklearn's `ArgKmin` `parallel_on_X` order AND the
//! `kd_tree`/`ball_tree` (`BinaryTree.query`) order. The builder's doc-comment
//! claims this is "the single-heap result used by BOTH `ArgKmin.compute`
//! (brute, `parallel_on_X`) and `BinaryTree.query`" and that the differing
//! order is sklearn's `parallel_on_Y`, a "non-canonical, chunk/thread-size
//! dependent artifact" that is "not a stable contract".
//!
//! That justification is FALSE for the DEFAULT user surface. Live sklearn
//! 1.5.2 / numpy 2.4.5 oracle (system python3), verified deterministic over 10
//! repeated runs and across `OMP_NUM_THREADS in {1,2,4,8}` and
//! `n_jobs in {1,4,8}`:
//!
//! ```text
//! X = [[1,0],[-1,0],[0,1],[0,-1],[2,0],[-2,0],[0,2]] ; q = [[0,0]]
//!
//! KNeighborsClassifier(n_neighbors=5).fit(X,arange(7))
//!   ._fit_method                                   -> 'brute'   (auto picks brute)
//!   .kneighbors([[0,0]])[1]                         -> [[0, 2, 3, 1, 4]]   (DEFAULT, STABLE)
//!
//! ArgKmin.compute(q, X, k=5, strategy='auto')       -> [0, 2, 3, 1, 4]     (== parallel_on_Y)
//! ArgKmin.compute(q, X, k=5, strategy='parallel_on_X') -> [3, 1, 0, 2, 4]  (single-heap)
//! ArgKmin.compute(q, X, k=5, strategy='parallel_on_Y') -> [0, 2, 3, 1, 4]
//! ```
//!
//! i.e. for the small-n surface that `import ferrolearn` / the test suite
//! exercises, `algorithm='auto'` resolves to `brute`, brute's `auto` strategy
//! resolves to `parallel_on_Y`, and the DEFAULT, deterministic, user-facing
//! `.kneighbors()` order is `[0, 2, 3, 1, 4]`. The builder's kernel returns the
//! `parallel_on_X` order `[3, 1, 0, 2, 4]`, which sklearn's default user NEVER
//! sees on this input. The "parallel_on_Y is thread-dependent" claim is
//! refuted: the output is bit-identical across every thread count tested.
//!
//! The k-NN SET matches at small n (so uniform-weight `predict` is unaffected
//! there), but the ORDER is an observable contract: `kneighbors` (REQ-6)
//! returns `(distances, indices)` and the index order is the documented return
//! value; `kneighbors_graph` and any distance-tie-order-sensitive consumer
//! inherit it. (The n>=50 regime additionally diverges on the SET — see
//! `ferrolearn-python/tests/divergence_knn_default_strategy.py`.)
//!
//! Expected order is the LIVE sklearn oracle, NOT copied from ferrolearn
//! (R-CHAR-3). Tracking: #2143.

use ferrolearn_core::traits::Fit;
use ferrolearn_neighbors::{Algorithm, KNeighborsClassifier};
use ndarray::{Array1, Array2, array};

/// The builder's own example fixture. Seven points; `q=[0,0]`.
/// idx0..idx3 are the unit-axis points (dist 1), idx4/idx5 at dist 2 on the
/// x-axis, idx6 at dist 2 on the y-axis.
fn caveat_fixture() -> (Array2<f64>, Array1<usize>, Array2<f64>) {
    let x = array![
        [1.0, 0.0],
        [-1.0, 0.0],
        [0.0, 1.0],
        [0.0, -1.0],
        [2.0, 0.0],
        [-2.0, 0.0],
        [0.0, 2.0],
    ];
    let y = Array1::from_iter(0..7usize);
    let xq = array![[0.0, 0.0]];
    (x, y, xq)
}

/// Divergence: `FittedKNeighborsClassifier::kneighbors` returns the
/// `parallel_on_X`/`kd_tree` single-heap ORDER, but sklearn's DEFAULT
/// `KNeighborsClassifier.kneighbors()` (which resolves `auto -> brute ->
/// parallel_on_Y` for this small input) returns a different, fully
/// deterministic order.
///
/// sklearn default order (live oracle, all stable):
///   k=5 -> [0, 2, 3, 1, 4]
///   k=3 -> [1, 2, 0]
///
/// ferrolearn (NeighborsHeap single-heap) returns:
///   k=5 -> [3, 1, 0, 2, 4]
///   k=3 -> [2, 0, 1]
///
/// Tracking: #2143
#[test]
#[ignore = "divergence: KNN kneighbors tie ORDER is parallel_on_X single-heap, sklearn default is parallel_on_Y (auto->brute); tracking #2143"]
fn divergence_kneighbors_order_differs_from_sklearn_default() {
    let (x, y, xq) = caveat_fixture();

    // (k, sklearn DEFAULT .kneighbors() index order). Live sklearn 1.5.2 oracle:
    //   q=[0,0]; KNeighborsClassifier(n_neighbors=k).fit(x,arange(7)).kneighbors(q)[1]
    let cases: [(usize, Vec<usize>); 2] = [(5, vec![0, 2, 3, 1, 4]), (3, vec![1, 2, 0])];

    for (k, sklearn_default_order) in cases {
        // Use `auto`, which is what a default `import ferrolearn` /
        // KNeighborsClassifier user gets, mirroring sklearn's default.
        let fitted = KNeighborsClassifier::<f64>::new()
            .with_n_neighbors(k)
            .with_algorithm(Algorithm::Auto)
            .fit(&x, &y)
            .unwrap();

        let (_d, indices) = fitted.kneighbors(&xq, Some(k)).unwrap();
        let got: Vec<usize> = indices.row(0).to_vec();

        assert_eq!(
            got, sklearn_default_order,
            "k={k}: ferrolearn kneighbors order {got:?} must equal sklearn's \
             DEFAULT .kneighbors() order {sklearn_default_order:?} (auto -> \
             brute -> parallel_on_Y, deterministic across thread counts). \
             ferrolearn returns the parallel_on_X single-heap order instead."
        );
    }
}
