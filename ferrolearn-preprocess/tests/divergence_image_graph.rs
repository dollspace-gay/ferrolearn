//! Divergence guards for dense image graph helpers vs scikit-learn 1.5.2
//! `sklearn.feature_extraction.image.grid_to_graph` and `img_to_graph`.
//!
//! Oracle snippets:
//! ```text
//! import numpy as np
//! from sklearn.feature_extraction.image import grid_to_graph, img_to_graph
//! print(grid_to_graph(2, 2, return_as=np.ndarray).tolist())
//! print(img_to_graph(np.array([[0., 0.], [0., 1.]]), return_as=np.ndarray).tolist())
//! mask=np.array([[True, False, True], [False, True, True]])
//! print(grid_to_graph(2, 3, 1, mask=mask, return_as=np.ndarray).tolist())
//! print(img_to_graph(np.array([[0., 2., 5.], [7., 11., 13.]]), mask=mask,
//!                    return_as=np.ndarray).tolist())
//! ```

use ferrolearn_preprocess::{grid_to_graph, img_to_graph};
use ndarray::{Array2, array};

fn assert_matrix_eq(actual: &Array2<f64>, expected: &Array2<f64>) {
    assert_eq!(actual.dim(), expected.dim(), "shape mismatch");
    for i in 0..actual.nrows() {
        for j in 0..actual.ncols() {
            assert!(
                (actual[[i, j]] - expected[[i, j]]).abs() < 1e-12,
                "entry ({i},{j}) got {} expected {}",
                actual[[i, j]],
                expected[[i, j]]
            );
        }
    }
}

#[test]
fn grid_to_graph_dense_matches_sklearn_oracle() {
    let graph = grid_to_graph::<f64>(2, 2, 1, None).unwrap();
    let expected = array![
        [1.0_f64, 1.0, 1.0, 0.0],
        [1.0, 1.0, 0.0, 1.0],
        [1.0, 0.0, 1.0, 1.0],
        [0.0, 1.0, 1.0, 1.0],
    ];
    assert_matrix_eq(&graph, &expected);
}

#[test]
fn img_to_graph_dense_matches_sklearn_documented_oracle() {
    let img = array![[0.0_f64, 0.0], [0.0, 1.0]];
    let graph = img_to_graph(&img, None).unwrap();
    let expected = array![
        [0.0_f64, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 1.0, 1.0],
    ];
    assert_matrix_eq(&graph, &expected);
}

#[test]
fn graph_mask_renumbering_matches_sklearn_oracles() {
    let mask = array![[true, false, true], [false, true, true]];
    let flat_mask: Vec<bool> = mask.iter().copied().collect();

    let grid = grid_to_graph::<f64>(2, 3, 1, Some(&flat_mask)).unwrap();
    let expected_grid = array![
        [1.0_f64, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 1.0, 1.0, 1.0],
    ];
    assert_matrix_eq(&grid, &expected_grid);

    let img = array![[0.0_f64, 2.0, 5.0], [7.0, 11.0, 13.0]];
    let image_graph = img_to_graph(&img, Some(&mask)).unwrap();
    let expected_image_graph = array![
        [0.0_f64, 0.0, 0.0, 0.0],
        [0.0, 5.0, 0.0, 8.0],
        [0.0, 0.0, 11.0, 2.0],
        [0.0, 8.0, 2.0, 13.0],
    ];
    assert_matrix_eq(&image_graph, &expected_image_graph);
}

#[test]
fn graph_helpers_validate_shapes() {
    assert!(grid_to_graph::<f64>(0, 2, 1, None).is_err());
    assert!(grid_to_graph::<f64>(2, 2, 1, Some(&[true, false])).is_err());

    let img = array![[0.0_f64, 1.0]];
    let wrong_mask = array![[true], [false]];
    assert!(img_to_graph(&img, Some(&wrong_mask)).is_err());
}
