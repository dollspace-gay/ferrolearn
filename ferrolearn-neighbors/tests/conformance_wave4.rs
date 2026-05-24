//! Wave-4 neighbors conformance vs scikit-learn.

use ferrolearn_core::{Fit, Predict};
use ferrolearn_neighbors::{
    LocalOutlierFactor, NearestCentroid, NearestNeighbors, RadiusNeighborsClassifier,
    RadiusNeighborsRegressor,
};
use ferrolearn_test_oracle::{json_to_array1, json_to_array2, load_fixture};

#[test]
fn conformance_nearest_centroid() {
    let fx = load_fixture("nearest_centroid");
    let x = json_to_array2(&fx.input["X"]);
    let y_vec: Vec<usize> = fx.input["y"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let y = ndarray::Array1::from_vec(y_vec);

    let model = NearestCentroid::<f64>::new();
    let fitted = model.fit(&x, &y).expect("NearestCentroid fit");
    let preds = fitted.predict(&x).expect("NearestCentroid predict");
    let expected: Vec<usize> = fx.expected["predictions"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let matches = preds
        .iter()
        .zip(expected.iter())
        .filter(|(a, e)| a == e)
        .count();
    let acc = matches as f64 / preds.len() as f64;
    assert!(
        acc >= 0.95,
        "NearestCentroid label-agreement {acc:.4} < 0.95 floor"
    );
}

#[test]
fn conformance_nearest_neighbors() {
    let fx = load_fixture("nearest_neighbors");
    let x = json_to_array2(&fx.input["X"]);
    let n_neighbors = fx.params["n_neighbors"].as_u64().unwrap_or(5) as usize;

    let model = NearestNeighbors::<f64>::new().with_n_neighbors(n_neighbors);
    let fitted = model.fit(&x, &()).expect("NearestNeighbors fit");
    // Query the first 10 rows.
    let query = x.slice(ndarray::s![..10, ..]).to_owned();
    let (dists, idxs) = fitted
        .kneighbors(&query, Some(n_neighbors))
        .expect("kneighbors");
    let expected_dists = fx.expected["distances"].as_array().unwrap();
    assert_eq!(dists.nrows(), expected_dists.len(), "kneighbors dists rows");
    assert_eq!(dists.ncols(), n_neighbors, "kneighbors dists cols");
    assert_eq!(idxs.nrows(), expected_dists.len(), "kneighbors idxs rows");
    // Distance to closest neighbor (self) should be ~0; verify monotone increasing.
    for row in dists.rows() {
        let mut prev = -1.0;
        for &d in row.iter() {
            assert!(
                d >= prev - 1e-9,
                "kneighbors dists not monotone: {d} after {prev}"
            );
            prev = d;
        }
    }
}

#[test]
fn conformance_local_outlier_factor() {
    let fx = load_fixture("local_outlier_factor");
    let x = json_to_array2(&fx.input["X"]);
    let n_neighbors = fx.params["n_neighbors"].as_u64().unwrap_or(20) as usize;
    let contamination = fx.params["contamination"].as_f64().unwrap_or(0.1);

    let model = LocalOutlierFactor::<f64>::new()
        .with_n_neighbors(n_neighbors)
        .with_contamination(contamination);
    let preds = model.fit_predict(&x).expect("LOF fit_predict");
    let expected: Vec<i64> = fx.expected["predictions"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap())
        .collect();
    let matches = preds
        .iter()
        .zip(expected.iter())
        .filter(|&(&a, &e)| a as i64 == e)
        .count();
    let frac = matches as f64 / preds.len() as f64;
    assert!(frac >= 0.80, "LOF +1/-1 agreement {frac:.4} < 0.80 floor");
}

#[test]
fn conformance_radius_neighbors_classifier() {
    let fx = load_fixture("radius_neighbors_classifier");
    let x = json_to_array2(&fx.input["X"]);
    let y_vec: Vec<usize> = fx.input["y"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let y = ndarray::Array1::from_vec(y_vec);
    let radius = fx.params["radius"].as_f64().unwrap_or(4.0);

    let model = RadiusNeighborsClassifier::<f64>::new().with_radius(radius);
    let fitted = model.fit(&x, &y).expect("RadiusClassifier fit");
    let preds = fitted.predict(&x).expect("RadiusClassifier predict");
    let expected: Vec<usize> = fx.expected["predictions"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let matches = preds
        .iter()
        .zip(expected.iter())
        .filter(|(a, e)| a == e)
        .count();
    let acc = matches as f64 / preds.len() as f64;
    assert!(
        acc >= 0.90,
        "RadiusNeighborsClassifier accuracy {acc:.4} < 0.90 floor"
    );
}

#[test]
fn conformance_radius_neighbors_regressor() {
    let fx = load_fixture("radius_neighbors_regressor");
    let x = json_to_array2(&fx.input["X"]);
    let y = json_to_array1(&fx.input["y"]);
    let radius = fx.params["radius"].as_f64().unwrap_or(4.0);

    let model = RadiusNeighborsRegressor::<f64>::new().with_radius(radius);
    let fitted = model.fit(&x, &y).expect("RadiusRegressor fit");
    let preds = fitted.predict(&x).expect("RadiusRegressor predict");

    let y_mean = y.iter().sum::<f64>() / y.len() as f64;
    let ss_tot: f64 = y.iter().map(|v| (v - y_mean).powi(2)).sum();
    let ss_res: f64 = preds
        .iter()
        .zip(y.iter())
        .map(|(a, e)| (a - e).powi(2))
        .sum();
    let r2 = 1.0 - ss_res / ss_tot;
    let expected_r2 = fx.expected["r2"].as_f64().unwrap_or(0.5);
    assert!(
        r2 >= expected_r2 - 0.10,
        "RadiusNeighborsRegressor R² {r2:.4} < sklearn - 0.10 ({expected_r2:.4})"
    );
}
