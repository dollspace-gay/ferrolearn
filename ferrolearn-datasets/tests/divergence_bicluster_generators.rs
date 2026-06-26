//! Divergence guards for the scoped dataset generators added from
//! `sklearn.datasets._samples_generator.py`:
//! `make_sparse_coded_signal`, `make_biclusters`, and `make_checkerboard`.
//!
//! These tests pin structure that is independent of the numpy RNG bit-stream:
//! output shapes, sparse-code row cardinality, dictionary normalization,
//! `data = code @ dictionary`, and bicluster/checkerboard membership masks.
//! Element-wise stochastic value parity remains blocked by the crate-wide
//! `SmallRng` vs numpy `RandomState` gap documented in
//! `.design/datasets/generators.md`.

use ferrolearn_datasets::{make_biclusters, make_checkerboard, make_sparse_coded_signal};

#[test]
fn sparse_coded_signal_matches_sklearn_shapes_and_sparse_code_contract() {
    let (data, dictionary, code) = make_sparse_coded_signal::<f64>(8, 5, 3, 2, Some(11)).unwrap();

    assert_eq!((data.nrows(), data.ncols()), (8, 3));
    assert_eq!((dictionary.nrows(), dictionary.ncols()), (5, 3));
    assert_eq!((code.nrows(), code.ncols()), (8, 5));

    for sample in 0..code.nrows() {
        let nnz = code
            .row(sample)
            .iter()
            .filter(|&&value| value != 0.0)
            .count();
        assert_eq!(nnz, 2, "code row {sample} must have exactly two nonzeros");
    }

    for component in 0..dictionary.nrows() {
        let norm = dictionary
            .row(component)
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-12,
            "dictionary component {component} norm was {norm}"
        );
    }

    let recomputed = code.dot(&dictionary);
    for i in 0..data.nrows() {
        for j in 0..data.ncols() {
            assert!(
                (data[[i, j]] - recomputed[[i, j]]).abs() < 1e-12,
                "data[{i},{j}] should equal code @ dictionary"
            );
        }
    }

    assert!(make_sparse_coded_signal::<f64>(4, 3, 2, 4, Some(0)).is_err());
}

#[test]
fn biclusters_match_diagonal_block_membership_contract() {
    let (x, rows, cols) = make_biclusters::<f64>(6, 5, 2, 0.0, 7.0, 7.0, false, Some(0)).unwrap();

    assert_eq!((x.nrows(), x.ncols()), (6, 5));
    assert_eq!((rows.nrows(), rows.ncols()), (2, 6));
    assert_eq!((cols.nrows(), cols.ncols()), (2, 5));

    for row in 0..x.nrows() {
        assert_eq!(
            (0..rows.nrows())
                .filter(|&cluster| rows[[cluster, row]])
                .count(),
            1,
            "row {row} must belong to exactly one bicluster row mask"
        );
    }
    for col in 0..x.ncols() {
        assert_eq!(
            (0..cols.nrows())
                .filter(|&cluster| cols[[cluster, col]])
                .count(),
            1,
            "col {col} must belong to exactly one bicluster col mask"
        );
    }

    for i in 0..x.nrows() {
        let row_cluster = (0..rows.nrows())
            .find(|&cluster| rows[[cluster, i]])
            .unwrap();
        for j in 0..x.ncols() {
            let col_cluster = (0..cols.nrows())
                .find(|&cluster| cols[[cluster, j]])
                .unwrap();
            let expected = if row_cluster == col_cluster { 7.0 } else { 0.0 };
            assert_eq!(
                x[[i, j]],
                expected,
                "cell ({i},{j}) should follow diagonal bicluster membership"
            );
        }
    }
}

#[test]
fn checkerboard_masks_enumerate_row_column_cluster_pairs() {
    let (x, rows, cols) =
        make_checkerboard::<f64>(6, 5, 3, 2, 0.0, 10.0, 20.0, false, Some(3)).unwrap();

    assert_eq!((x.nrows(), x.ncols()), (6, 5));
    assert_eq!((rows.nrows(), rows.ncols()), (6, 6));
    assert_eq!((cols.nrows(), cols.ncols()), (6, 5));

    for block in 0..rows.nrows() {
        let mut block_value: Option<f64> = None;
        for i in 0..x.nrows() {
            if !rows[[block, i]] {
                continue;
            }
            for j in 0..x.ncols() {
                if !cols[[block, j]] {
                    continue;
                }
                match block_value {
                    Some(value) => assert!(
                        (x[[i, j]] - value).abs() < 1e-12,
                        "checkerboard block {block} is not constant"
                    ),
                    None => block_value = Some(x[[i, j]]),
                }
            }
        }
        let value = block_value.expect("each checkerboard block should contain at least one cell");
        assert!(
            (10.0..=20.0).contains(&value),
            "checkerboard block {block} value {value} should come from [10, 20]"
        );
    }

    for row in 0..x.nrows() {
        let memberships = (0..rows.nrows())
            .filter(|&block| rows[[block, row]])
            .count();
        assert_eq!(
            memberships, 2,
            "each row participates once for every column cluster"
        );
    }
    for col in 0..x.ncols() {
        let memberships = (0..cols.nrows())
            .filter(|&block| cols[[block, col]])
            .count();
        assert_eq!(
            memberships, 3,
            "each column participates once for every row cluster"
        );
    }
}
