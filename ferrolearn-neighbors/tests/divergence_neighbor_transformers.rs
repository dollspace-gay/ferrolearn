//! Green-guard tests for neighbor graph transformers against scikit-learn
//! `KNeighborsTransformer` / `RadiusNeighborsTransformer`.
//!
//! Expected dense matrices and `nnz` values below come from live sklearn 1.5.2
//! calls in this workspace. The remaining constructor surface (`metric`,
//! `p`, `metric_params`, `n_jobs`, sparse/precomputed input) remains a documented
//! contract gap.

use ferrolearn_core::{Fit, Transform};
use ferrolearn_neighbors::{GraphMode, KNeighborsTransformer, RadiusNeighborsTransformer};
use ndarray::{Array2, array};

fn diagonal_three() -> Array2<f64> {
    array![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]
}

fn assert_dense_close(actual: &Array2<f64>, expected: &Array2<f64>) {
    assert_eq!(actual.dim(), expected.dim(), "dense shape");
    for ((i, j), &value) in actual.indexed_iter() {
        let expected_value = expected[[i, j]];
        assert!(
            (value - expected_value).abs() < 1e-12,
            "dense[{i},{j}] actual={value} expected={expected_value}"
        );
    }
}

#[test]
fn green_kneighbors_transformer_distance_includes_self_plus_one_neighbor() {
    // sklearn 1.5.2:
    // KNeighborsTransformer(n_neighbors=1, mode="distance", algorithm="brute")
    //   .fit_transform([[0,0],[1,1],[2,2]]).toarray()
    let x = diagonal_three();
    let fitted = KNeighborsTransformer::<f64>::new()
        .with_n_neighbors(1)
        .with_mode(GraphMode::Distance)
        .fit(&x, &())
        .unwrap();
    let graph = fitted.transform(&x).unwrap();
    assert_eq!(graph.n_rows(), 3);
    assert_eq!(graph.n_cols(), 3);
    assert_eq!(graph.nnz(), 6);
    assert_eq!(fitted.n_features_in(), 2);
    assert_eq!(fitted.n_samples_fit(), 3);
    assert_eq!(fitted.n_neighbors(), 1);

    let expected = array![
        [0.0, std::f64::consts::SQRT_2, 0.0],
        [std::f64::consts::SQRT_2, 0.0, 0.0],
        [0.0, std::f64::consts::SQRT_2, 0.0],
    ];
    assert_dense_close(&graph.to_dense(), &expected);
}

#[test]
fn green_kneighbors_transformer_connectivity_self_only_for_one_neighbor() {
    // sklearn 1.5.2:
    // KNeighborsTransformer(n_neighbors=1, mode="connectivity")
    //   .fit_transform([[0,0],[1,1],[2,2]]).toarray()
    let x = diagonal_three();
    let graph = KNeighborsTransformer::<f64>::new()
        .with_n_neighbors(1)
        .with_mode(GraphMode::Connectivity)
        .fit(&x, &())
        .unwrap()
        .transform(&x)
        .unwrap();
    assert_eq!(graph.nnz(), 3);
    assert_dense_close(
        &graph.to_dense(),
        &array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    );
}

#[test]
fn green_radius_neighbors_transformer_distance_and_connectivity() {
    let x = diagonal_three();

    // sklearn 1.5.2:
    // RadiusNeighborsTransformer(radius=1.5, mode="distance", algorithm="brute")
    let distance_graph = RadiusNeighborsTransformer::<f64>::new()
        .with_radius(1.5)
        .with_mode(GraphMode::Distance)
        .fit(&x, &())
        .unwrap()
        .transform(&x)
        .unwrap();
    assert_eq!(distance_graph.nnz(), 7);
    assert_dense_close(
        &distance_graph.to_dense(),
        &array![
            [0.0, std::f64::consts::SQRT_2, 0.0],
            [std::f64::consts::SQRT_2, 0.0, std::f64::consts::SQRT_2],
            [0.0, std::f64::consts::SQRT_2, 0.0],
        ],
    );

    // sklearn 1.5.2:
    // RadiusNeighborsTransformer(radius=1.5, mode="connectivity")
    let connectivity_fitted = RadiusNeighborsTransformer::<f64>::new()
        .with_radius(1.5)
        .with_mode(GraphMode::Connectivity)
        .fit(&x, &())
        .unwrap();
    assert_eq!(connectivity_fitted.n_features_in(), 2);
    assert_eq!(connectivity_fitted.n_samples_fit(), 3);
    assert_eq!(connectivity_fitted.mode(), GraphMode::Connectivity);
    assert!((connectivity_fitted.radius() - 1.5).abs() < 1e-12);
    let connectivity_graph = connectivity_fitted.transform(&x).unwrap();
    assert_eq!(connectivity_graph.nnz(), 7);
    assert_dense_close(
        &connectivity_graph.to_dense(),
        &array![[1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]],
    );
}

#[test]
fn green_neighbor_transformers_validation_boundaries() {
    let x = diagonal_three();
    assert!(
        KNeighborsTransformer::<f64>::new()
            .with_n_neighbors(0)
            .fit(&x, &())
            .is_err()
    );
    assert!(
        RadiusNeighborsTransformer::<f64>::new()
            .with_radius(-1.0)
            .fit(&x, &())
            .is_err()
    );

    let fitted = KNeighborsTransformer::<f64>::new()
        .with_n_neighbors(1)
        .fit(&x, &())
        .unwrap();
    assert!(fitted.transform(&array![[0.0, 0.0, 0.0]]).is_err());
}
