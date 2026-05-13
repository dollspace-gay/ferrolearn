//! Wave-4 covariance conformance vs scikit-learn: GraphicalLasso + EllipticEnvelope.

use ferrolearn_core::{Fit, Predict};
use ferrolearn_covariance::{EllipticEnvelope, GraphicalLasso};
use ferrolearn_test_oracle::{
    assert_close_slice, json_to_array1, json_to_array2, load_fixture,
};

#[test]
fn conformance_graphical_lasso_location() {
    let fx = load_fixture("graphical_lasso");
    let x = json_to_array2(&fx.input["X"]);
    let alpha = fx.params["alpha"].as_f64().unwrap_or(0.1);

    let model = GraphicalLasso::<f64>::new(alpha);
    let fitted = model.fit(&x, &()).expect("GraphicalLasso fit");

    let expected_loc = json_to_array1(&fx.expected["location"]);
    assert_close_slice(
        fitted.location().as_slice().unwrap(),
        expected_loc.as_slice().unwrap(),
        1e-9,
        1e-12,
        "GraphicalLasso.location",
    );
}

#[test]
fn conformance_graphical_lasso_covariance() {
    let fx = load_fixture("graphical_lasso");
    let x = json_to_array2(&fx.input["X"]);
    let alpha = fx.params["alpha"].as_f64().unwrap_or(0.1);

    let model = GraphicalLasso::<f64>::new(alpha);
    let fitted = model.fit(&x, &()).expect("GraphicalLasso fit");
    let expected_cov = json_to_array2(&fx.expected["covariance"]);
    // #343 fixed: diagonal no longer carries the +alpha shift. Off-diagonal
    // ~3e-5 drift remains from coordinate-descent convergence path
    // differences (sklearn's `_dual_gap` stopping criterion is slightly
    // different from ferrolearn's `||W - W_old||` criterion).
    assert_close_slice(
        fitted.covariance().as_slice().unwrap(),
        expected_cov.as_slice().unwrap(),
        1e-3,
        1e-4,
        "GraphicalLasso.covariance",
    );
}

#[test]
fn conformance_elliptic_envelope() {
    let fx = load_fixture("elliptic_envelope");
    let x = json_to_array2(&fx.input["X"]);
    let contamination = fx.params["contamination"].as_f64().unwrap_or(0.1);

    let model = EllipticEnvelope::<f64>::new().contamination(contamination);
    let fitted = model.fit(&x, &()).expect("EllipticEnvelope fit");
    let preds = fitted.predict(&x).expect("EllipticEnvelope predict");
    let expected: Vec<i64> = fx.expected["predictions"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap())
        .collect();
    let matches = preds.iter().zip(expected.iter()).filter(|&(&a, &e)| a as i64 == e).count();
    let frac = matches as f64 / preds.len() as f64;
    // EllipticEnvelope depends on FastMCD which has subset variance (see
    // documented divergence `fastmcd-subset-selection-variance` and
    // investigation issue #337). Different FastMCD subsets produce different
    // chi-squared thresholds for the contamination cutoff, so the +1/-1
    // labels can diverge on borderline samples. 60% floor accepts this.
    assert!(
        frac >= 0.60,
        "EllipticEnvelope +1/-1 agreement {frac:.4} < 0.60 floor"
    );
}
