//! Adversarial divergence tests for `ferrolearn-datasets/src/svmlight.rs`
//! against scikit-learn 1.5.2 (live oracle). Each test pins a REAL, observable
//! divergence from sklearn. Expected values are computed from a live sklearn
//! 1.5.2 call (run from /tmp) or a sklearn `file:line` constant — NEVER copied
//! from the ferrolearn side (goal.md R-CHAR-3).
//!
//! Tracking umbrella: #1640. Per-divergence blockers: #1641-#1646.

use ferrolearn_datasets::{dump_svmlight_file, load_files, load_svmlight_files, load_svmlight_str};
use ndarray::{Array1, Array2};
use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};

fn tmpdir(tag: &str) -> std::path::PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("ferrolearn_divsvm_{tag}_{nanos}"));
    fs::create_dir_all(&dir).unwrap();
    dir
}

/// Divergence (REQ-6, HEADLINE): ferrolearn's `dump_svmlight_file` writes
/// ONE-based `idx:val` (`svmlight.rs` fn `dump_svmlight_file`:
/// `format!("{}:{}", j + 1, v)`), whereas sklearn's default `zero_based=True`
/// (`sklearn/datasets/_svmlight_format_io.py:479` `zero_based=True`, `:596`
/// `one_based = not zero_based`) writes ZERO-based indices.
///
/// Live oracle (sklearn 1.5.2, from /tmp):
/// ```python
/// import io, numpy as np
/// from sklearn.datasets import dump_svmlight_file
/// b=io.BytesIO()
/// dump_svmlight_file(np.array([[1,0,2],[0,3,0],[4,5,6]],float),
///                    np.array([1,0,1],float), b)
/// b.getvalue().decode()  # -> '1 0:1 2:2\n0 1:3\n1 0:4 1:5 2:6\n'
/// ```
/// ferrolearn emits `'1 1:1 3:2\n0 2:3\n1 1:4 2:5 3:6\n'` (indices off by one).
/// Tracking: #1641
#[test]
fn divergence_dump_index_base_zero_based_default() {
    // sklearn default-dump text for the SAME matrix (from the live oracle above).
    const SKLEARN_DUMP: &str = "1 0:1 2:2\n0 1:3\n1 0:4 1:5 2:6\n";

    let x: Array2<f64> = ndarray::array![[1.0, 0.0, 2.0], [0.0, 3.0, 0.0], [4.0, 5.0, 6.0]];
    let y: Array1<f64> = ndarray::array![1.0, 0.0, 1.0];
    let dir = tmpdir("dumpbase");
    let path = dir.join("out.svmlight");
    dump_svmlight_file(&x, &y, &path).unwrap();
    let ferro_text = fs::read_to_string(&path).unwrap();

    assert_eq!(
        ferro_text, SKLEARN_DUMP,
        "ferrolearn dump must be byte-for-byte equal to sklearn's default \
         zero-based dump"
    );
}

/// Divergence (REQ-2): ferrolearn's `parse_line` HARD-rejects index 0
/// (`svmlight.rs`: `if idx_one_based == 0 { return Err(...) }`), whereas
/// sklearn's default `zero_based="auto"` keeps 0-based files (min index 0)
/// as-is (`sklearn/datasets/_svmlight_format_io.py:402-409`).
///
/// Live oracle (sklearn 1.5.2, from /tmp):
/// ```python
/// import io
/// from sklearn.datasets import load_svmlight_file
/// X,y=load_svmlight_file(io.BytesIO(b'1 0:1 2:2\n0 1:3\n'))
/// X.shape          # -> (2, 3)
/// X.toarray()      # -> [[1, 0, 2], [0, 3, 0]]
/// ```
/// ferrolearn returns `Err` on the `0:` token.
/// Tracking: #1642
#[test]
fn divergence_load_zero_based_auto() {
    // sklearn dense X for the 0-based content (from the live oracle above).
    let sklearn_x: Array2<f64> = ndarray::array![[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]];

    let res = load_svmlight_str("1 0:1 2:2\n0 1:3\n", None);
    let (x, _y) = res.expect(
        "sklearn loads 0-based svmlight content under zero_based=auto; ferrolearn \
         must too (currently errors on the '0:' index)",
    );
    assert_eq!(x.dim(), (2, 3), "shape must match sklearn (2, 3)");
    assert_eq!(x, sklearn_x, "dense X must match sklearn's toarray()");
}

/// Divergence (REQ-4): ferrolearn's `load_svmlight_files` infers `n_features`
/// PER FILE (`svmlight.rs`: `paths.iter().map(|p| load_svmlight_file(p, n_features))`),
/// whereas sklearn enforces a SINGLE common `n_features = max col index + 1`
/// across ALL files (`sklearn/datasets/_svmlight_format_io.py:410-419`).
///
/// Live oracle (sklearn 1.5.2, from /tmp):
/// ```python
/// import io
/// from sklearn.datasets import load_svmlight_files
/// a=io.BytesIO(b'1 1:1.0 2:2.0\n'); b=io.BytesIO(b'0 1:3.0 4:9.0\n')
/// Xa,ya,Xb,yb=load_svmlight_files([a,b])
/// Xa.shape, Xb.shape   # -> ((1, 4), (1, 4))
/// ```
/// ferrolearn infers file A -> 2 cols, file B -> 4 cols (unequal widths).
/// Tracking: #1643
#[test]
fn divergence_load_files_common_n_features() {
    // sklearn common width for BOTH files (from the live oracle above).
    const SKLEARN_COMMON_NCOLS: usize = 4;

    let dir = tmpdir("multifile");
    let pa = dir.join("a.svmlight");
    let pb = dir.join("b.svmlight");
    fs::write(&pa, "1 1:1.0 2:2.0\n").unwrap();
    fs::write(&pb, "0 1:3.0 4:9.0\n").unwrap();

    let results = load_svmlight_files(&[pa, pb], None).unwrap();
    let xa_cols = results[0].0.ncols();
    let xb_cols = results[1].0.ncols();

    assert_eq!(
        xa_cols, SKLEARN_COMMON_NCOLS,
        "file A must use the common n_features sklearn computes"
    );
    assert_eq!(
        xb_cols, SKLEARN_COMMON_NCOLS,
        "file B must use the common n_features sklearn computes"
    );
    assert_eq!(
        xa_cols, xb_cols,
        "sklearn gives both files the same n_features"
    );
}

/// Divergence (REQ-7): ferrolearn formats dump values with Rust's
/// `format!("{}", f64)` (`svmlight.rs`: `format!("{}:{}", j + 1, v)`), which
/// diverges from sklearn/numpy's `repr` text for large/small magnitudes.
/// sklearn writes values via numpy's float repr (`_svmlight_format_fast.pyx`,
/// reached from `sklearn/datasets/_svmlight_format_io.py:450` `_dump_svmlight_file`).
///
/// Live oracle (sklearn 1.5.2, from /tmp):
/// ```python
/// import io, numpy as np
/// from sklearn.datasets import dump_svmlight_file
/// b=io.BytesIO()
/// dump_svmlight_file(np.array([[1e20, 1e-20]],float), np.array([2.5],float), b)
/// b.getvalue().decode()  # -> '2.5 0:1e+20 1:9.999999999999999e-21\n'
/// ```
/// ferrolearn emits `'2.5 1:100000000000000000000 2:0.00000000000000000001\n'`
/// (no scientific notation; and `1e-20` reprs differently).
/// Tracking: #1644
#[test]
fn divergence_dump_value_formatting_scientific() {
    // sklearn default-dump text for [[1e20, 1e-20]]/[2.5] (live oracle above).
    const SKLEARN_DUMP: &str = "2.5 0:1e+20 1:9.999999999999999e-21\n";

    let x: Array2<f64> = ndarray::array![[1e20, 1e-20]];
    let y: Array1<f64> = ndarray::array![2.5];
    let dir = tmpdir("fmtsci");
    let path = dir.join("out.svmlight");
    dump_svmlight_file(&x, &y, &path).unwrap();
    let ferro_text = fs::read_to_string(&path).unwrap();

    assert_eq!(
        ferro_text, SKLEARN_DUMP,
        "ferrolearn value text must match sklearn/numpy float repr (scientific \
         notation for large/small magnitudes)"
    );
}

/// Divergence (REQ-5): a `qid:N` token. sklearn with `query_id=False` (default)
/// SILENTLY IGNORES `qid:` tokens (`sklearn/datasets/_svmlight_format_io.py:64`
/// `query_id=False` default; the fast parser skips qid when not requested).
/// ferrolearn's `parse_line` tries to parse `qid` as a usize index and ERRORS
/// (`svmlight.rs`: `idx_tok.parse::<usize>()` -> "bad index 'qid'").
///
/// Live oracle (sklearn 1.5.2, from /tmp):
/// ```python
/// import io
/// from sklearn.datasets import load_svmlight_file
/// X,y=load_svmlight_file(io.BytesIO(b'1 qid:1 1:1.0 2:2.0\n'))
/// X.shape          # -> (1, 2)
/// X.toarray()      # -> [[1.0, 2.0]]
/// ```
/// ferrolearn returns `Err` on the `qid:1` token.
/// Tracking: #1645
#[test]
fn divergence_load_qid_token_ignored() {
    // sklearn dense X with qid ignored (from the live oracle above).
    let sklearn_x: Array2<f64> = ndarray::array![[1.0, 2.0]];

    let res = load_svmlight_str("1 qid:1 1:1.0 2:2.0\n", None);
    let (x, _y) = res.expect(
        "sklearn silently ignores qid: tokens when query_id=False (default); \
         ferrolearn must too (currently errors parsing 'qid' as an index)",
    );
    assert_eq!(x.dim(), (1, 2), "shape must match sklearn (1, 2)");
    assert_eq!(x, sklearn_x, "dense X must match sklearn's toarray()");
}

/// Divergence (REQ-9): ferrolearn's `load_files` returns a deterministic,
/// lexically-sorted 3-tuple `(docs, labels, target_names)` with NO shuffle
/// (`svmlight.rs`: `files.sort()`, no `shuffle`/`random_state`). sklearn's
/// `load_files` default `shuffle=True` PERMUTES the order
/// (`sklearn/datasets/_base.py` `load_files`, default `shuffle=True`,
/// `random_state=0`).
///
/// Live oracle (sklearn 1.5.2, from /tmp), classes alpha/beta with doc0..2 each:
/// ```python
/// from sklearn.datasets import load_files
/// b=load_files(root, random_state=0)   # default shuffle=True
/// b.target.tolist()  # -> [1, 0, 0, 1, 0, 1]   (NOT sorted)
/// ```
/// ferrolearn returns target `[0, 0, 0, 1, 1, 1]` (all class 0 then all class 1).
/// Tracking: #1646
#[test]
fn divergence_load_files_shuffle_default() {
    // sklearn's default-shuffle target order (random_state=0; live oracle above).
    const SKLEARN_TARGET: [usize; 6] = [1, 0, 0, 1, 0, 1];

    let dir = tmpdir("loadfiles");
    for cls in ["alpha", "beta"] {
        let sub = dir.join(cls);
        fs::create_dir_all(&sub).unwrap();
        for i in 0..3 {
            fs::write(sub.join(format!("doc{i}.txt")), format!("{cls}-doc{i}")).unwrap();
        }
    }

    let (_docs, labels, _names) = load_files(&dir).unwrap();
    let ferro_target: Vec<usize> = labels.to_vec();

    assert_eq!(
        ferro_target.as_slice(),
        &SKLEARN_TARGET,
        "load_files default shuffle=True must permute order to match sklearn \
         (random_state=0); ferrolearn returns sorted, unshuffled order"
    );
}
