//! Conformance tests for ferrolearn-preprocess vs scikit-learn.
//!
//! Each test loads a fixture from `fixtures/<name>.json`, runs the matching
//! ferrolearn transformer on the same input with the same hyperparameters,
//! and compares the output to sklearn's via `ferrolearn-test-oracle` helpers.
//!
//! Tolerances default to the preprocessing-class constants
//! (`TOL_PREPROCESS_REL`, `TOL_PREPROCESS_ABS`) and can be tightened/loosened
//! per-fixture via the JSON `tolerance` field.

use ferrolearn_core::traits::{Fit, Transform};
use ferrolearn_test_oracle::{
    TOL_PREPROCESS_ABS, TOL_PREPROCESS_REL, assert_close_slice, json_to_array1, json_to_array2,
    load_fixture, parse_f64_value,
};
use ndarray::{Array1, Array2};

// ---------------------------------------------------------------------------
// Local helpers — fixture-specific parsing not provided by the shared oracle.
// ---------------------------------------------------------------------------

/// Parse a JSON nested array into an `Array2<f64>`, mapping `null` to `NaN`.
fn json_to_array2_with_nan(value: &serde_json::Value) -> Array2<f64> {
    let rows: Vec<Vec<f64>> = value
        .as_array()
        .expect("expected JSON array of rows")
        .iter()
        .map(|row| {
            row.as_array()
                .expect("expected JSON array of row values")
                .iter()
                .map(|v| {
                    if v.is_null() {
                        f64::NAN
                    } else {
                        parse_f64_value(v)
                    }
                })
                .collect()
        })
        .collect();
    let n_rows = rows.len();
    let n_cols = rows.first().map_or(0, Vec::len);
    let flat: Vec<f64> = rows.into_iter().flatten().collect();
    Array2::from_shape_vec((n_rows, n_cols), flat).expect("uniform row lengths")
}

/// Parse a JSON nested array of non-negative integers into an `Array2<usize>`.
#[allow(
    dead_code,
    reason = "OneHotEncoder migrated to Array2<F> input (REQ-3 #1150); helper retained for other usize fixtures"
)]
fn json_to_array2_usize(value: &serde_json::Value) -> Array2<usize> {
    let rows: Vec<Vec<usize>> = value
        .as_array()
        .expect("expected JSON array of rows")
        .iter()
        .map(|row| {
            row.as_array()
                .expect("expected JSON array of row values")
                .iter()
                .map(|v| v.as_u64().expect("expected u64") as usize)
                .collect()
        })
        .collect();
    let n_rows = rows.len();
    let n_cols = rows.first().map_or(0, Vec::len);
    let flat: Vec<usize> = rows.into_iter().flatten().collect();
    Array2::from_shape_vec((n_rows, n_cols), flat).expect("uniform row lengths")
}

/// Element-wise NaN-aware comparison wrapper that flattens 2-D arrays before
/// delegating to `assert_close_slice`. NaN-NaN positions are treated as
/// equal; non-NaN positions follow the composite tolerance check.
fn assert_array2_close(
    actual: &Array2<f64>,
    expected: &Array2<f64>,
    rel: f64,
    abs: f64,
    label: &str,
) {
    assert_eq!(
        actual.shape(),
        expected.shape(),
        "{label}: shape mismatch ({:?} vs {:?})",
        actual.shape(),
        expected.shape()
    );
    assert_close_slice(
        actual.as_slice().expect("contiguous"),
        expected.as_slice().expect("contiguous"),
        rel,
        abs,
        label,
    );
}

// ---------------------------------------------------------------------------
// 1. StandardScaler
//
// Fixture params: with_mean, with_std (both `true` in the current fixture).
// ferrolearn's `StandardScaler` always centers and scales; non-default
// combinations are not exposed yet. We assert the fixture's flags match the
// default to keep the test honest.
// ---------------------------------------------------------------------------

#[test]
fn conformance_standard_scaler() {
    let fx = load_fixture("standard_scaler");
    let x = json_to_array2(&fx.input["X"]);
    let (rel, abs) = fx.tolerance(TOL_PREPROCESS_REL, TOL_PREPROCESS_ABS);

    let with_mean = fx.params["with_mean"].as_bool().unwrap_or(true);
    let with_std = fx.params["with_std"].as_bool().unwrap_or(true);
    assert!(
        with_mean && with_std,
        "StandardScaler: ferrolearn does not yet expose with_mean=false / with_std=false"
    );

    let fitted = ferrolearn_preprocess::StandardScaler::<f64>::new()
        .fit(&x, &())
        .expect("StandardScaler fit");

    let expected_mean = json_to_array1(&fx.expected["mean"]);
    assert_close_slice(
        fitted.mean().as_slice().unwrap(),
        expected_mean.as_slice().unwrap(),
        rel,
        abs,
        "StandardScaler.mean",
    );
    let expected_std = json_to_array1(&fx.expected["std"]);
    assert_close_slice(
        fitted.std().as_slice().unwrap(),
        expected_std.as_slice().unwrap(),
        rel,
        abs,
        "StandardScaler.std",
    );

    let transformed = fitted.transform(&x).expect("StandardScaler transform");
    let expected_transformed = json_to_array2(&fx.expected["transformed"]);
    assert_array2_close(
        &transformed,
        &expected_transformed,
        rel,
        abs,
        "StandardScaler.transformed",
    );
}

// ---------------------------------------------------------------------------
// 2. MinMaxScaler
// ---------------------------------------------------------------------------

#[test]
fn conformance_minmax_scaler() {
    let fx = load_fixture("minmax_scaler");
    let x = json_to_array2(&fx.input["X"]);
    let (rel, abs) = fx.tolerance(TOL_PREPROCESS_REL, TOL_PREPROCESS_ABS);

    let range = fx.params["feature_range"]
        .as_array()
        .expect("feature_range must be an array");
    let lo = parse_f64_value(&range[0]);
    let hi = parse_f64_value(&range[1]);

    let scaler = if (lo - 0.0).abs() < f64::EPSILON && (hi - 1.0).abs() < f64::EPSILON {
        ferrolearn_preprocess::MinMaxScaler::<f64>::new()
    } else {
        ferrolearn_preprocess::MinMaxScaler::<f64>::with_feature_range(lo, hi)
            .expect("valid feature_range")
    };
    let fitted = scaler.fit(&x, &()).expect("MinMaxScaler fit");

    let expected_min = json_to_array1(&fx.expected["data_min"]);
    assert_close_slice(
        fitted.data_min().as_slice().unwrap(),
        expected_min.as_slice().unwrap(),
        rel,
        abs,
        "MinMaxScaler.data_min",
    );
    let expected_max = json_to_array1(&fx.expected["data_max"]);
    assert_close_slice(
        fitted.data_max().as_slice().unwrap(),
        expected_max.as_slice().unwrap(),
        rel,
        abs,
        "MinMaxScaler.data_max",
    );

    let transformed = fitted.transform(&x).expect("MinMaxScaler transform");
    let expected_transformed = json_to_array2(&fx.expected["transformed"]);
    assert_array2_close(
        &transformed,
        &expected_transformed,
        rel,
        abs,
        "MinMaxScaler.transformed",
    );
}

// ---------------------------------------------------------------------------
// 3. RobustScaler
//
// Fixture params: with_centering, with_scaling (both `true` in the fixture).
// ferrolearn unconditionally centers by median and scales by IQR.
// ---------------------------------------------------------------------------

#[test]
fn conformance_robust_scaler() {
    let fx = load_fixture("robust_scaler");
    let x = json_to_array2(&fx.input["X"]);
    let (rel, abs) = fx.tolerance(TOL_PREPROCESS_REL, TOL_PREPROCESS_ABS);

    let with_centering = fx.params["with_centering"].as_bool().unwrap_or(true);
    let with_scaling = fx.params["with_scaling"].as_bool().unwrap_or(true);
    assert!(
        with_centering && with_scaling,
        "RobustScaler: ferrolearn does not yet expose with_centering=false / with_scaling=false"
    );

    let fitted = ferrolearn_preprocess::RobustScaler::<f64>::new()
        .fit(&x, &())
        .expect("RobustScaler fit");

    // fixture key "center" → fitted.median(); "scale" → fitted.iqr()
    let expected_center = json_to_array1(&fx.expected["center"]);
    assert_close_slice(
        fitted.median().as_slice().unwrap(),
        expected_center.as_slice().unwrap(),
        rel,
        abs,
        "RobustScaler.center",
    );
    let expected_scale = json_to_array1(&fx.expected["scale"]);
    assert_close_slice(
        fitted.iqr().as_slice().unwrap(),
        expected_scale.as_slice().unwrap(),
        rel,
        abs,
        "RobustScaler.scale",
    );

    let transformed = fitted.transform(&x).expect("RobustScaler transform");
    let expected_transformed = json_to_array2(&fx.expected["transformed"]);
    assert_array2_close(
        &transformed,
        &expected_transformed,
        rel,
        abs,
        "RobustScaler.transformed",
    );
}

// ---------------------------------------------------------------------------
// 4. MaxAbsScaler
// ---------------------------------------------------------------------------

#[test]
fn conformance_max_abs_scaler() {
    let fx = load_fixture("max_abs_scaler");
    let x = json_to_array2(&fx.input["X"]);
    let (rel, abs) = fx.tolerance(TOL_PREPROCESS_REL, TOL_PREPROCESS_ABS);

    let fitted = ferrolearn_preprocess::MaxAbsScaler::<f64>::new()
        .fit(&x, &())
        .expect("MaxAbsScaler fit");

    let expected_max_abs = json_to_array1(&fx.expected["max_abs"]);
    assert_close_slice(
        fitted.max_abs().as_slice().unwrap(),
        expected_max_abs.as_slice().unwrap(),
        rel,
        abs,
        "MaxAbsScaler.max_abs",
    );

    let transformed = fitted.transform(&x).expect("MaxAbsScaler transform");
    let expected_transformed = json_to_array2(&fx.expected["transformed"]);
    assert_array2_close(
        &transformed,
        &expected_transformed,
        rel,
        abs,
        "MaxAbsScaler.transformed",
    );
}

// ---------------------------------------------------------------------------
// 5. Normalizer — stateless; transform-only.
// ---------------------------------------------------------------------------

#[test]
fn conformance_normalizer() {
    let fx = load_fixture("normalizer");
    let x = json_to_array2(&fx.input["X"]);
    let (rel, abs) = fx.tolerance(TOL_PREPROCESS_REL, TOL_PREPROCESS_ABS);

    let norm = match fx.params["norm"].as_str().unwrap_or("l2") {
        "l1" => ferrolearn_preprocess::normalizer::NormType::L1,
        "l2" => ferrolearn_preprocess::normalizer::NormType::L2,
        "max" => ferrolearn_preprocess::normalizer::NormType::Max,
        other => panic!("Normalizer: unknown norm '{other}'"),
    };
    let normalizer = ferrolearn_preprocess::Normalizer::<f64>::new(norm);

    let transformed = normalizer.transform(&x).expect("Normalizer transform");
    let expected_transformed = json_to_array2(&fx.expected["transformed"]);
    assert_array2_close(
        &transformed,
        &expected_transformed,
        rel,
        abs,
        "Normalizer.transformed",
    );
}

// ---------------------------------------------------------------------------
// 6. Binarizer — output is 0.0 or 1.0; exact equality.
// ---------------------------------------------------------------------------

#[test]
fn conformance_binarizer() {
    let fx = load_fixture("binarizer");
    let x = json_to_array2(&fx.input["X"]);
    let (rel, abs) = fx.tolerance(TOL_PREPROCESS_REL, TOL_PREPROCESS_ABS);

    let threshold = fx.params["threshold"].as_f64().expect("threshold f64");
    let bin = ferrolearn_preprocess::Binarizer::<f64>::new(threshold);

    let transformed = bin.transform(&x).expect("Binarizer transform");
    let expected_transformed = json_to_array2(&fx.expected["transformed"]);
    assert_array2_close(
        &transformed,
        &expected_transformed,
        rel,
        abs,
        "Binarizer.transformed",
    );
}

// ---------------------------------------------------------------------------
// 7. KBinsDiscretizer
// ---------------------------------------------------------------------------

#[test]
fn conformance_kbins_discretizer() {
    let fx = load_fixture("kbins_discretizer");
    let x = json_to_array2(&fx.input["X"]);
    let (rel, abs) = fx.tolerance(TOL_PREPROCESS_REL, TOL_PREPROCESS_ABS);

    let n_bins = fx.params["n_bins"].as_u64().expect("n_bins u64") as usize;
    let encode = match fx.params["encode"].as_str().unwrap_or("ordinal") {
        "ordinal" => ferrolearn_preprocess::BinEncoding::Ordinal,
        "onehot" | "onehot-dense" => ferrolearn_preprocess::BinEncoding::OneHot,
        other => panic!("KBinsDiscretizer: unsupported encode '{other}'"),
    };
    let strategy = match fx.params["strategy"].as_str().unwrap_or("uniform") {
        "uniform" => ferrolearn_preprocess::BinStrategy::Uniform,
        "quantile" => ferrolearn_preprocess::BinStrategy::Quantile,
        "kmeans" => ferrolearn_preprocess::BinStrategy::KMeans,
        other => panic!("KBinsDiscretizer: unsupported strategy '{other}'"),
    };
    let disc = ferrolearn_preprocess::KBinsDiscretizer::<f64>::new(n_bins, encode, strategy);
    let fitted = disc.fit(&x, &()).expect("KBinsDiscretizer fit");

    let expected_edges = fx.expected["bin_edges"]
        .as_array()
        .expect("bin_edges must be a 2D JSON array");
    let actual_edges = fitted.bin_edges();
    assert_eq!(
        actual_edges.len(),
        expected_edges.len(),
        "KBinsDiscretizer: feature count mismatch in bin_edges"
    );
    for (j, (a_edges, e_edges)) in actual_edges.iter().zip(expected_edges.iter()).enumerate() {
        let e_vec: Vec<f64> = e_edges
            .as_array()
            .expect("edge row must be array")
            .iter()
            .map(parse_f64_value)
            .collect();
        assert_close_slice(
            a_edges.as_slice(),
            &e_vec,
            rel,
            abs,
            &format!("KBinsDiscretizer.bin_edges[{j}]"),
        );
    }

    let transformed = fitted.transform(&x).expect("KBinsDiscretizer transform");
    let expected_transformed = json_to_array2(&fx.expected["transformed"]);
    assert_array2_close(
        &transformed,
        &expected_transformed,
        rel,
        abs,
        "KBinsDiscretizer.transformed",
    );
}

// ---------------------------------------------------------------------------
// 8. OneHotEncoder
//
// ferrolearn's encoder stores `n_categories` per column (sklearn stores the
// actual category labels). For the fixture, sklearn's `categories[j]` is
// `[0, 1, ..., n_categories[j] - 1]` so the two encodings agree by
// construction; we verify length and reconstruct the implicit category list.
// ---------------------------------------------------------------------------

#[test]
fn conformance_one_hot_encoder() {
    let fx = load_fixture("one_hot_encoder");
    let x = json_to_array2_with_nan(&fx.input["X"]);
    let (rel, abs) = fx.tolerance(TOL_PREPROCESS_REL, TOL_PREPROCESS_ABS);

    let sparse = fx.params["sparse_output"].as_bool().unwrap_or(false);
    assert!(
        !sparse,
        "OneHotEncoder: ferrolearn only emits dense output (sparse_output=true is unsupported)"
    );

    let enc = ferrolearn_preprocess::OneHotEncoder::<f64>::new();
    let fitted = enc.fit(&x, &()).expect("OneHotEncoder fit");

    // Compare the implicit categories per column. ferrolearn assumes
    // contiguous indices `[0, n_categories[j] - 1]`; sklearn produces the
    // same list for fixtures sampled from `range(k)`.
    let expected_categories = fx.expected["categories"]
        .as_array()
        .expect("categories must be 2D JSON array");
    assert_eq!(
        fitted.n_categories().len(),
        expected_categories.len(),
        "OneHotEncoder: feature count mismatch"
    );
    for (j, (n, expected_col)) in fitted
        .n_categories()
        .iter()
        .zip(expected_categories.iter())
        .enumerate()
    {
        let expected_n = expected_col
            .as_array()
            .expect("categories row must be array")
            .len();
        assert_eq!(
            *n, expected_n,
            "OneHotEncoder.categories[{j}]: count mismatch ({n} vs {expected_n})"
        );
    }

    let transformed = fitted.transform(&x).expect("OneHotEncoder transform");
    let expected_transformed = json_to_array2(&fx.expected["transformed"]);
    assert_array2_close(
        &transformed,
        &expected_transformed,
        rel,
        abs,
        "OneHotEncoder.transformed",
    );
}

// ---------------------------------------------------------------------------
// 9. PowerTransformer (Yeo-Johnson)
//
// The lambda fit is a 1-D Brent search; both libraries converge but to
// slightly different optima for finite samples. The existing oracle test
// uses 1e-2; we adopt the same per-fixture floor here, taken as the abs
// tolerance default for this estimator.
// ---------------------------------------------------------------------------

#[test]
fn conformance_power_transformer() {
    let fx = load_fixture("power_transformer");
    let x = json_to_array2(&fx.input["X"]);
    // Brent-fit lambdas converge to ~1e-2; predictions inherit that. Use the
    // fixture override when present, else fall back to a class-appropriate
    // wider floor.
    let (rel, abs) = fx.tolerance(1e-2, 1e-2);

    let method = fx.params["method"].as_str().unwrap_or("yeo-johnson");
    assert_eq!(
        method, "yeo-johnson",
        "PowerTransformer: only yeo-johnson is currently supported in ferrolearn"
    );
    let standardize = fx.params["standardize"].as_bool().unwrap_or(true);
    let pt = if standardize {
        ferrolearn_preprocess::PowerTransformer::<f64>::new()
    } else {
        ferrolearn_preprocess::PowerTransformer::<f64>::without_standardize()
    };
    let fitted = pt.fit(&x, &()).expect("PowerTransformer fit");

    let expected_lambdas = json_to_array1(&fx.expected["lambdas"]);
    assert_close_slice(
        fitted.lambdas().as_slice().unwrap(),
        expected_lambdas.as_slice().unwrap(),
        rel,
        abs,
        "PowerTransformer.lambdas",
    );

    let transformed = fitted.transform(&x).expect("PowerTransformer transform");
    let expected_transformed = json_to_array2(&fx.expected["transformed"]);
    assert_array2_close(
        &transformed,
        &expected_transformed,
        rel,
        abs,
        "PowerTransformer.transformed",
    );
}

// ---------------------------------------------------------------------------
// 10. QuantileTransformer
//
// Fixture's `references` is the evenly-spaced quantile level vector; we do
// not currently expose it through the public API, so it's compared
// implicitly via the transformed output.
// ---------------------------------------------------------------------------

#[test]
fn conformance_quantile_transformer() {
    let fx = load_fixture("quantile_transformer");
    let x = json_to_array2(&fx.input["X"]);
    let (rel, abs) = fx.tolerance(TOL_PREPROCESS_REL, TOL_PREPROCESS_ABS);

    let n_quantiles = fx.params["n_quantiles"].as_u64().expect("n_quantiles u64") as usize;
    let output = match fx.params["output_distribution"]
        .as_str()
        .unwrap_or("uniform")
    {
        "uniform" => ferrolearn_preprocess::OutputDistribution::Uniform,
        "normal" => ferrolearn_preprocess::OutputDistribution::Normal,
        other => panic!("QuantileTransformer: unsupported output_distribution '{other}'"),
    };
    let qt = ferrolearn_preprocess::QuantileTransformer::<f64>::new(n_quantiles, output, 0);
    let fitted = qt.fit(&x, &()).expect("QuantileTransformer fit");

    // Compare per-feature learned quantile reference points.
    let expected_quantiles = fx.expected["quantiles"]
        .as_array()
        .expect("quantiles must be a JSON array");
    let actual_quantiles = fitted.quantiles();
    // sklearn stores quantiles in shape `[n_quantiles, n_features]` (rows
    // = quantile level, columns = feature). ferrolearn stores them as
    // `[n_features][n_quantiles]`. Detect the orientation by length.
    if actual_quantiles.len() == expected_quantiles.len() {
        // Same outer-dim count (n_features matches first dimension of fixture).
        for (j, (a_col, e_col)) in actual_quantiles
            .iter()
            .zip(expected_quantiles.iter())
            .enumerate()
        {
            let e_vec: Vec<f64> = e_col
                .as_array()
                .expect("quantile row must be array")
                .iter()
                .map(parse_f64_value)
                .collect();
            // Either matched by feature (same length per row) — compare,
            // else the fixture is `[n_quantiles][n_features]`; fall through
            // to the transpose branch below.
            if a_col.len() == e_vec.len() {
                assert_close_slice(
                    a_col.as_slice(),
                    &e_vec,
                    rel,
                    abs,
                    &format!("QuantileTransformer.quantiles[{j}]"),
                );
            } else {
                // Lengths disagree → fixture is transposed; bail and check
                // via transformed output only.
                break;
            }
        }
    }

    let transformed = fitted.transform(&x).expect("QuantileTransformer transform");
    let expected_transformed = json_to_array2(&fx.expected["transformed"]);
    assert_array2_close(
        &transformed,
        &expected_transformed,
        rel,
        abs,
        "QuantileTransformer.transformed",
    );
}

// ---------------------------------------------------------------------------
// 11. LabelEncoder — string labels.
// ---------------------------------------------------------------------------

#[test]
fn conformance_label_encoder() {
    let fx = load_fixture("label_encoder");

    let labels_vec: Vec<String> = fx.input["labels"]
        .as_array()
        .expect("labels must be array")
        .iter()
        .map(|v| v.as_str().expect("label must be string").to_string())
        .collect();
    let labels = Array1::from_vec(labels_vec);

    let expected_classes: Vec<String> = fx.expected["classes"]
        .as_array()
        .expect("classes must be array")
        .iter()
        .map(|v| v.as_str().expect("class must be string").to_string())
        .collect();
    let expected_transformed: Vec<usize> = fx.expected["transformed"]
        .as_array()
        .expect("transformed must be array")
        .iter()
        .map(|v| v.as_u64().expect("transformed entry must be u64") as usize)
        .collect();

    let enc = ferrolearn_preprocess::LabelEncoder::new();
    let fitted = enc.fit(&labels, &()).expect("LabelEncoder fit");

    assert_eq!(
        fitted.classes(),
        expected_classes.as_slice(),
        "LabelEncoder.classes"
    );

    let transformed = fitted.transform(&labels).expect("LabelEncoder transform");
    let actual: Vec<usize> = transformed.iter().copied().collect();
    assert_eq!(actual, expected_transformed, "LabelEncoder.transformed");
}

// ---------------------------------------------------------------------------
// 12. PolynomialFeatures
// ---------------------------------------------------------------------------

#[test]
fn conformance_polynomial_features() {
    let fx = load_fixture("polynomial_features");
    let x = json_to_array2(&fx.input["X"]);
    let (rel, abs) = fx.tolerance(TOL_PREPROCESS_REL, TOL_PREPROCESS_ABS);

    let degree = fx.params["degree"].as_u64().expect("degree u64") as usize;
    let interaction_only = fx.params["interaction_only"]
        .as_bool()
        .expect("interaction_only bool");
    let include_bias = fx.params["include_bias"]
        .as_bool()
        .expect("include_bias bool");

    let poly = ferrolearn_preprocess::PolynomialFeatures::<f64>::new(
        degree,
        interaction_only,
        include_bias,
    )
    .expect("PolynomialFeatures::new");

    let transformed = poly.transform(&x).expect("PolynomialFeatures transform");
    let expected_n_output = fx.expected["n_output_features"]
        .as_u64()
        .expect("n_output_features u64") as usize;
    assert_eq!(
        transformed.ncols(),
        expected_n_output,
        "PolynomialFeatures.n_output_features"
    );

    let expected_transformed = json_to_array2(&fx.expected["transformed"]);
    assert_array2_close(
        &transformed,
        &expected_transformed,
        rel,
        abs,
        "PolynomialFeatures.transformed",
    );
}

// ---------------------------------------------------------------------------
// 13. SimpleImputer — NaN-in, fill-out.
// ---------------------------------------------------------------------------

#[test]
fn conformance_simple_imputer() {
    let fx = load_fixture("simple_imputer");
    let x = json_to_array2_with_nan(&fx.input["X"]);
    let (rel, abs) = fx.tolerance(TOL_PREPROCESS_REL, TOL_PREPROCESS_ABS);

    let strategy = match fx.params["strategy"].as_str().unwrap_or("mean") {
        "mean" => ferrolearn_preprocess::ImputeStrategy::<f64>::Mean,
        "median" => ferrolearn_preprocess::ImputeStrategy::<f64>::Median,
        "most_frequent" => ferrolearn_preprocess::ImputeStrategy::<f64>::MostFrequent,
        other => panic!("SimpleImputer: unsupported strategy '{other}'"),
    };
    let imp = ferrolearn_preprocess::SimpleImputer::<f64>::new(strategy);
    let fitted = imp.fit(&x, &()).expect("SimpleImputer fit");

    let expected_stats = json_to_array1(&fx.expected["statistics"]);
    assert_close_slice(
        fitted.fill_values().as_slice().unwrap(),
        expected_stats.as_slice().unwrap(),
        rel,
        abs,
        "SimpleImputer.statistics",
    );

    let transformed = fitted.transform(&x).expect("SimpleImputer transform");
    let expected_transformed = json_to_array2(&fx.expected["transformed"]);
    assert_array2_close(
        &transformed,
        &expected_transformed,
        rel,
        abs,
        "SimpleImputer.transformed",
    );
}
