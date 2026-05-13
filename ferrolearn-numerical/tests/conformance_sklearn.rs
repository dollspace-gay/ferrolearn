//! Conformance tests for ferrolearn-numerical vs scipy.
//!
//! Three families: cubic-spline interpolation, statistical distributions,
//! sparse symmetric eigenvalues. Eigenvalues and distribution scalars use
//! tight tolerances; eigenvectors are sign-ambiguous and compared per-row.

use ferrolearn_numerical::distributions::{
    Beta, ChiSquared, ContinuousDistribution, FDist, Gamma, Normal, StudentsT,
};
use ferrolearn_numerical::interpolate::{BoundaryCondition, CubicSpline};
use ferrolearn_numerical::sparse_eig::{LanczosSolver, WhichEigenvalues};
use ferrolearn_test_oracle::{
    assert_close, assert_close_slice, load_fixture, parse_f64_value, TOL_NUMERICAL_ABS,
    TOL_NUMERICAL_REL,
};

fn json_f64_vec(v: &serde_json::Value) -> Vec<f64> {
    v.as_array()
        .expect("expected JSON array")
        .iter()
        .map(parse_f64_value)
        .collect()
}

fn dense_to_sparse(rows: &[Vec<f64>]) -> sprs::CsMat<f64> {
    let n = rows.len();
    let m = rows[0].len();
    let mut indptr = Vec::with_capacity(n + 1);
    let mut indices = Vec::new();
    let mut data = Vec::new();
    indptr.push(0);
    for row in rows {
        for (j, &val) in row.iter().enumerate() {
            if val.abs() > 1e-15 {
                indices.push(j);
                data.push(val);
            }
        }
        indptr.push(indices.len());
    }
    sprs::CsMat::new((n, m), indptr, indices, data)
}

#[test]
fn conformance_cubic_spline() {
    let fx = load_fixture("cubic_spline");
    let (rel, abs) = fx.tolerance(TOL_NUMERICAL_REL, TOL_NUMERICAL_ABS);

    let x_knots = json_f64_vec(&fx.input["x_knots"]);
    let y_knots = json_f64_vec(&fx.input["y_knots"]);
    let bc = match fx.params["bc_type"].as_str().unwrap_or("not-a-knot") {
        "not-a-knot" => BoundaryCondition::NotAKnot,
        "natural" => BoundaryCondition::Natural,
        other => panic!("unsupported bc_type: {other} (ferrolearn supports natural / not-a-knot)"),
    };
    let spline = CubicSpline::new(&x_knots, &y_knots, bc).expect("CubicSpline::new");

    let eval_points = json_f64_vec(&fx.expected["eval_points"]);
    let computed = spline.eval_array(&eval_points);
    let computed_slice: &[f64] = computed.as_slice().expect("contiguous");
    let expected_values = json_f64_vec(&fx.expected["eval_values"]);

    assert_close_slice(
        computed_slice,
        &expected_values,
        rel,
        abs,
        "CubicSpline.eval_array",
    );
}

#[test]
fn conformance_distributions() {
    let fx = load_fixture("distributions");
    let (rel, abs) = fx.tolerance(TOL_NUMERICAL_REL, TOL_NUMERICAL_ABS);

    // Normal — standard N(0, 1) per the fixture convention.
    {
        let block = &fx.expected["normal"];
        let points = json_f64_vec(&block["points"]);
        let expected_pdf = json_f64_vec(&block["pdf"]);
        let expected_cdf = json_f64_vec(&block["cdf"]);
        let expected_sf = json_f64_vec(&block["sf"]);
        let dist = Normal::new(0.0, 1.0).expect("Normal::new");
        let pdf: Vec<f64> = points.iter().map(|&x| dist.pdf(x)).collect();
        let cdf: Vec<f64> = points.iter().map(|&x| dist.cdf(x)).collect();
        let sf: Vec<f64> = points.iter().map(|&x| dist.sf(x)).collect();
        assert_close_slice(&pdf, &expected_pdf, rel, abs, "Normal.pdf");
        assert_close_slice(&cdf, &expected_cdf, rel, abs, "Normal.cdf");
        assert_close_slice(&sf, &expected_sf, rel, abs, "Normal.sf");
    }

    // ChiSquared
    {
        let block = &fx.expected["chi_squared"];
        let df = block["df"].as_f64().expect("chi_squared.df");
        let points = json_f64_vec(&block["points"]);
        let dist = ChiSquared::new(df).expect("ChiSquared::new");
        let pdf: Vec<f64> = points.iter().map(|&x| dist.pdf(x)).collect();
        let cdf: Vec<f64> = points.iter().map(|&x| dist.cdf(x)).collect();
        let sf: Vec<f64> = points.iter().map(|&x| dist.sf(x)).collect();
        assert_close_slice(&pdf, &json_f64_vec(&block["pdf"]), rel, abs, "ChiSquared.pdf");
        assert_close_slice(&cdf, &json_f64_vec(&block["cdf"]), rel, abs, "ChiSquared.cdf");
        assert_close_slice(&sf, &json_f64_vec(&block["sf"]), rel, abs, "ChiSquared.sf");
    }

    // F
    {
        let block = &fx.expected["f_distribution"];
        let df1 = block["df1"].as_f64().unwrap();
        let df2 = block["df2"].as_f64().unwrap();
        let points = json_f64_vec(&block["points"]);
        let dist = FDist::new(df1, df2).expect("FDist::new");
        let pdf: Vec<f64> = points.iter().map(|&x| dist.pdf(x)).collect();
        let cdf: Vec<f64> = points.iter().map(|&x| dist.cdf(x)).collect();
        let sf: Vec<f64> = points.iter().map(|&x| dist.sf(x)).collect();
        assert_close_slice(&pdf, &json_f64_vec(&block["pdf"]), rel, abs, "FDist.pdf");
        assert_close_slice(&cdf, &json_f64_vec(&block["cdf"]), rel, abs, "FDist.cdf");
        assert_close_slice(&sf, &json_f64_vec(&block["sf"]), rel, abs, "FDist.sf");
    }

    // Student's t
    {
        let block = &fx.expected["students_t"];
        let df = block["df"].as_f64().unwrap();
        let points = json_f64_vec(&block["points"]);
        let dist = StudentsT::new(df).expect("StudentsT::new");
        let pdf: Vec<f64> = points.iter().map(|&x| dist.pdf(x)).collect();
        let cdf: Vec<f64> = points.iter().map(|&x| dist.cdf(x)).collect();
        let sf: Vec<f64> = points.iter().map(|&x| dist.sf(x)).collect();
        assert_close_slice(&pdf, &json_f64_vec(&block["pdf"]), rel, abs, "StudentsT.pdf");
        assert_close_slice(&cdf, &json_f64_vec(&block["cdf"]), rel, abs, "StudentsT.cdf");
        assert_close_slice(&sf, &json_f64_vec(&block["sf"]), rel, abs, "StudentsT.sf");
    }

    // Beta — sub-keys: a, b
    {
        let block = &fx.expected["beta"];
        let a = block["a"].as_f64().unwrap();
        let b = block["b"].as_f64().unwrap();
        let points = json_f64_vec(&block["points"]);
        let dist = Beta::new(a, b).expect("Beta::new");
        let pdf: Vec<f64> = points.iter().map(|&x| dist.pdf(x)).collect();
        let cdf: Vec<f64> = points.iter().map(|&x| dist.cdf(x)).collect();
        let sf: Vec<f64> = points.iter().map(|&x| dist.sf(x)).collect();
        assert_close_slice(&pdf, &json_f64_vec(&block["pdf"]), rel, abs, "Beta.pdf");
        assert_close_slice(&cdf, &json_f64_vec(&block["cdf"]), rel, abs, "Beta.cdf");
        assert_close_slice(&sf, &json_f64_vec(&block["sf"]), rel, abs, "Beta.sf");
    }

    // Gamma — sub-keys: shape, scale. ferrolearn's Gamma takes (shape, rate)
    // where rate = 1/scale. Convert.
    {
        let block = &fx.expected["gamma"];
        let shape = block["shape"].as_f64().unwrap();
        let scale = block["scale"].as_f64().unwrap();
        let rate = 1.0 / scale;
        let points = json_f64_vec(&block["points"]);
        let dist = Gamma::new(shape, rate).expect("Gamma::new");
        let pdf: Vec<f64> = points.iter().map(|&x| dist.pdf(x)).collect();
        let cdf: Vec<f64> = points.iter().map(|&x| dist.cdf(x)).collect();
        let sf: Vec<f64> = points.iter().map(|&x| dist.sf(x)).collect();
        assert_close_slice(&pdf, &json_f64_vec(&block["pdf"]), rel, abs, "Gamma.pdf");
        assert_close_slice(&cdf, &json_f64_vec(&block["cdf"]), rel, abs, "Gamma.cdf");
        assert_close_slice(&sf, &json_f64_vec(&block["sf"]), rel, abs, "Gamma.sf");
    }
}

#[test]
fn conformance_sparse_eigsh() {
    let fx = load_fixture("sparse_eigsh");
    // Eigsh uses Lanczos with iterative refinement — eigenvalues should
    // agree to within 1e-9 absolute when the solver is run to tol=1e-12.
    let (rel, _abs) = fx.tolerance(1e-9, 1e-9);

    let n = fx.input["n"].as_u64().unwrap() as usize;
    let matrix_rows: Vec<Vec<f64>> = fx.input["matrix"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| json_f64_vec(row))
        .collect();
    assert_eq!(matrix_rows.len(), n, "matrix must be n x n");

    let mat = dense_to_sparse(&matrix_rows);
    let k = fx.params["k"].as_u64().unwrap() as usize;
    let which = match fx.params["which"].as_str().unwrap_or("LM") {
        "LM" => WhichEigenvalues::LargestMagnitude,
        "LA" => WhichEigenvalues::LargestAlgebraic,
        "SM" => WhichEigenvalues::SmallestMagnitude,
        "SA" => WhichEigenvalues::SmallestAlgebraic,
        other => panic!("unsupported `which`: {other}"),
    };
    let solver = LanczosSolver::new(k)
        .with_which(which)
        .with_tol(1e-12)
        .with_max_iter(500);
    let result = solver.solve_sparse(&mat).expect("Lanczos must converge");

    let mut computed: Vec<f64> = result.eigenvalues.to_vec();
    let mut expected = json_f64_vec(&fx.expected["eigenvalues"]);
    // Sort both by descending magnitude for `which=LM` (most common); fall
    // back to a generic descending sort for other modes.
    let key = |x: &f64| -x.abs();
    computed.sort_by(|a, b| key(a).partial_cmp(&key(b)).unwrap());
    expected.sort_by(|a, b| key(a).partial_cmp(&key(b)).unwrap());
    for (i, (a, e)) in computed.iter().zip(expected.iter()).enumerate() {
        assert_close(*a, *e, rel, 1e-9, &format!("eigenvalue[{i}]"));
    }
}
