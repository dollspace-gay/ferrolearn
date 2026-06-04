//! Divergence tests for `train_test_split` vs scikit-learn 1.5.2.
//!
//! Audited against `sklearn/model_selection/_split.py` (tag 1.5.2):
//! `train_test_split` (`:2686`), `_validate_shuffle_split` (`:2343`).
//!
//! Expected SIZES come from the live sklearn 1.5.2 oracle (run from /tmp,
//! `from sklearn.model_selection import train_test_split`) — never copied from
//! ferrolearn (R-CHAR-3). The split SIZES are deterministic regardless of the
//! shuffle PRNG, so these assertions are oracle-pinnable despite the RNG
//! carve-out on exact membership (REQ-3).

use ferrolearn_model_sel::train_test_split;
use ndarray::{Array1, Array2};

/// Build an (x, y) pair with `n` distinguishable rows: y[i] == i.
fn make_data(n: usize) -> (Array2<f64>, Array1<f64>) {
    let x = Array2::from_shape_fn((n, 3), |(i, j)| (i * 3 + j) as f64);
    let y = Array1::from_iter((0..n).map(|i| i as f64));
    (x, y)
}

// ---------------------------------------------------------------------------
// REQ-2 (FIXABLE) — test-set sizing rule: sklearn CEIL vs ferrolearn ROUND.
// ---------------------------------------------------------------------------

/// Divergence: ferrolearn's `train_test_split` sizes the test set with
/// `round(test_size * n)` (`ferrolearn-model-sel/src/split.rs:101`:
/// `let n_test = ((n_samples as f64) * test_size).round() as usize;`), whereas
/// sklearn uses `n_test = ceil(test_size * n_samples)`
/// (`sklearn/model_selection/_split.py:2390`).
///
/// Input `n=7, test_size=0.3`: `0.3 * 7 = 2.1`.
/// Live sklearn 1.5.2 oracle: `len(x_test) == 3`, `len(x_train) == 4`
/// (`train_test_split(np.arange(14).reshape(7,2), np.arange(7),
/// test_size=0.3, random_state=0)` → n_train=4, n_test=3).
/// ferrolearn: `round(2.1) = 2` → `len(x_test) == 2`, `len(x_train) == 5`.
/// The SIZES are seed-independent, so any fixed `random_state` pins them.
/// Tracking: #1723
#[test]
fn divergence_n_test_sizing_ceil_n7_ts03() {
    // Live sklearn 1.5.2 oracle (from /tmp), seed-independent SIZES:
    //   n_train = 4, n_test = ceil(0.3 * 7) = ceil(2.1) = 3
    const SK_N_TRAIN: usize = 4;
    const SK_N_TEST: usize = 3;

    let (x, y) = make_data(7);
    let (x_train, x_test, _, _) = train_test_split(&x, &y, 0.3, Some(0)).unwrap();

    assert_eq!(
        x_test.nrows(),
        SK_N_TEST,
        "n_test: sklearn ceil(0.3*7)=3, ferrolearn round(2.1)=2"
    );
    assert_eq!(
        x_train.nrows(),
        SK_N_TRAIN,
        "n_train: sklearn 7-3=4, ferrolearn 7-2=5"
    );
}

/// Divergence: same ROUND-vs-CEIL sizing bug for `n=10, test_size=0.33`.
/// `0.33 * 10 = 3.3`.
/// Live sklearn 1.5.2 oracle: `len(x_test) == 4`, `len(x_train) == 6`
/// (`train_test_split(np.arange(20).reshape(10,2), np.arange(10),
/// test_size=0.33, random_state=0)` → n_train=6, n_test=4).
/// ferrolearn: `round(3.3) = 3` → `len(x_test) == 3`, `len(x_train) == 7`.
/// Mirrors `sklearn/model_selection/_split.py:2390`.
/// Tracking: #1723
#[test]
fn divergence_n_test_sizing_ceil_n10_ts033() {
    // Live sklearn 1.5.2 oracle (from /tmp), seed-independent SIZES:
    //   n_train = 6, n_test = ceil(0.33 * 10) = ceil(3.3) = 4
    const SK_N_TRAIN: usize = 6;
    const SK_N_TEST: usize = 4;

    let (x, y) = make_data(10);
    let (x_train, x_test, _, _) = train_test_split(&x, &y, 0.33, Some(0)).unwrap();

    assert_eq!(
        x_test.nrows(),
        SK_N_TEST,
        "n_test: sklearn ceil(0.33*10)=4, ferrolearn round(3.3)=3"
    );
    assert_eq!(
        x_train.nrows(),
        SK_N_TRAIN,
        "n_train: sklearn 10-4=6, ferrolearn 10-3=7"
    );
}

// ---------------------------------------------------------------------------
// REQ-10 (FIXABLE) — empty-split: sklearn RAISES vs ferrolearn CLAMPS to Ok.
// ---------------------------------------------------------------------------

/// Divergence: when the resulting train set would be empty, sklearn RAISES
/// `ValueError` (`sklearn/model_selection/_split.py:2414`:
/// `if n_train == 0: raise ValueError("... the resulting train set will be
/// empty ...")`), whereas ferrolearn CLAMPS the test count
/// (`ferrolearn-model-sel/src/split.rs:102`:
/// `let n_test = n_test.max(1).min(n_samples - 1);`) and returns `Ok`.
///
/// Input `n=10, test_size=0.99`: `ceil(0.99 * 10) = ceil(9.9) = 10` → n_train=0.
/// Live sklearn 1.5.2 oracle: RAISES `ValueError` with message
/// "With n_samples=10, test_size=0.99 and train_size=None, the resulting
/// train set will be empty.".
/// ferrolearn: clamps to `n_test=9`, returns `Ok` with `n_train=1`.
/// Tracking: #1723
#[test]
fn divergence_empty_train_set_raises_n10_ts099() {
    let (x, y) = make_data(10);
    // Live sklearn 1.5.2 oracle: this call RAISES ValueError (empty train set).
    // ferrolearn must therefore return Err, not Ok.
    let result = train_test_split(&x, &y, 0.99, Some(0));
    assert!(
        result.is_err(),
        "sklearn raises ValueError for n=10,test_size=0.99 (empty train set, \
         _split.py:2414); ferrolearn clamped to Ok"
    );
}

/// Divergence: empty-train-set raise for a second input `n=2, test_size=0.9`.
/// `ceil(0.9 * 2) = ceil(1.8) = 2` → n_train=0.
/// Live sklearn 1.5.2 oracle: RAISES `ValueError`
/// ("With n_samples=2, test_size=0.9 and train_size=None, the resulting
/// train set will be empty.").
/// ferrolearn: clamps to `n_test=1`, returns `Ok` with `n_train=1`.
/// Mirrors `sklearn/model_selection/_split.py:2414`.
/// Tracking: #1723
#[test]
fn divergence_empty_train_set_raises_n2_ts09() {
    let (x, y) = make_data(2);
    // Live sklearn 1.5.2 oracle: RAISES ValueError (empty train set).
    let result = train_test_split(&x, &y, 0.9, Some(0));
    assert!(
        result.is_err(),
        "sklearn raises ValueError for n=2,test_size=0.9 (empty train set, \
         _split.py:2414); ferrolearn clamped to Ok"
    );
}

// ---------------------------------------------------------------------------
// REQ-4 (SHIPPED) — GREEN GUARD: structural partition. Should PASS.
// ---------------------------------------------------------------------------

/// Green guard (REQ-4): for a fixed seed the train/test sets partition
/// `0..n` — disjoint, exhaustive, sizes sum to `n`. Mirrors sklearn's
/// train/test index sets covering all samples
/// (`sklearn/model_selection/_split.py:2806`, `:2810-2814`).
/// Seed-independent; passes today.
#[test]
fn guard_structural_partition_req4() {
    let (x, y) = make_data(10);
    let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.3, Some(0)).unwrap();

    // Sizes sum to n.
    assert_eq!(x_train.nrows() + x_test.nrows(), 10);
    assert_eq!(y_train.len() + y_test.len(), 10);

    // Every original row (identified by its y value == original index) appears
    // exactly once across train ∪ test.
    let mut all: Vec<u64> = y_train
        .iter()
        .chain(y_test.iter())
        .map(|&v| v as u64)
        .collect();
    all.sort_unstable();
    let expected: Vec<u64> = (0..10).collect();
    assert_eq!(all, expected, "train ∪ test must be a partition of 0..n");
}

// ---------------------------------------------------------------------------
// REQ-1 (SHIPPED) — GREEN GUARD: float test_size validation (0,1). Should PASS.
// ---------------------------------------------------------------------------

/// Green guard (REQ-1): `test_size <= 0` and `test_size >= 1` both return
/// `Err`, plus an out-of-range float `1.5`. Mirrors sklearn's float guard
/// `test_size <= 0 or test_size >= 1`
/// (`sklearn/model_selection/_split.py:2358`); live oracle
/// `train_test_split(X, y, test_size=1.5)` raises `ValueError`.
#[test]
fn guard_test_size_validation_req1() {
    let (x, y) = make_data(10);
    assert!(
        train_test_split(&x, &y, 0.0, None).is_err(),
        "test_size=0 rejected"
    );
    assert!(
        train_test_split(&x, &y, 1.0, None).is_err(),
        "test_size=1 rejected"
    );
    assert!(
        train_test_split(&x, &y, 1.5, None).is_err(),
        "test_size=1.5 rejected (sklearn _split.py:2358)"
    );
    assert!(
        train_test_split(&x, &y, -0.1, None).is_err(),
        "test_size=-0.1 rejected"
    );
}
