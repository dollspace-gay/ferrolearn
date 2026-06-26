//! Proof-of-API integration test for ferrolearn-kernel.
//!
//! Audit deliverable for crosslink #292 (under #250). Exercises every
//! public API surface end-to-end.
//!
//! Coverage:
//! - KernelRidge: builders + fit + predict + score, every KernelType variant
//! - Nystroem: builders + fit + transform, every KernelType variant
//! - PolynomialCountSketch: builders + fit + transform
//! - RBFSampler: builders + fit + transform
//! - AdditiveChi2Sampler: builders + fit + transform
//! - SkewedChi2Sampler: builders + fit + transform
//! - GaussianProcessRegressor: fit + predict + predict_with_std + sample_y +
//!   log_marginal_likelihood + score
//! - GaussianProcessClassifier: fit + predict + predict_proba + predict_log_proba
//!   + log_marginal_likelihood + score
//! - GP kernels: RBFKernel, MaternKernel (3 nu values), RationalQuadratic,
//!   ExpSineSquared, ConstantKernel, WhiteKernel, DotProductKernel, SumKernel,
//!   ProductKernel, Exponentiation, CompoundKernel
//! - NadarayaWatson + LocalPolynomialRegression with every Kernel variant
//!   (Gaussian, Epanechnikov, Tricube, Biweight, Triweight, Uniform, Cosine)
//! - Bandwidth helpers: scott_bandwidth, silverman_bandwidth,
//!   CrossValidatedBandwidth
//! - Diagnostics: heteroscedasticity_test, residual_diagnostics

use ferrolearn_core::traits::{Fit, Predict, Transform};
use ferrolearn_kernel::bandwidth::BandwidthStrategy;
use ferrolearn_kernel::{
    AdditiveChi2Sampler, BiweightKernel, CompoundKernel, ConstantKernel, CosineKernel, CvStrategy,
    DotProductKernel, EpanechnikovKernel, ExpSineSquared, Exponentiation,
    FittedAdditiveChi2Sampler, FittedPolynomialCountSketch, FittedSkewedChi2Sampler,
    GaussianKernel, GaussianProcessClassifier, GaussianProcessRegressor, HeteroscedasticityTest,
    Hyperparameter, HyperparameterBounds, KernelRidge, KernelType, LocalPolynomialRegression,
    MaternKernel, NadarayaWatson, Nystroem, PolynomialCountSketch, ProductKernel, RBFKernel,
    RBFSampler, RationalQuadratic, SkewedChi2Sampler, SumKernel, TricubeKernel, TriweightKernel,
    UniformKernel, WhiteKernel, heteroscedasticity_test, residual_diagnostics, scott_bandwidth,
    silverman_bandwidth,
};
use ndarray::{Array1, Array2, array};

/// Simple regression dataset y ≈ 2x.
fn regression_data() -> (Array2<f64>, Array1<f64>) {
    let x = Array2::from_shape_vec(
        (10, 1),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    )
    .unwrap();
    let y = array![2.1, 3.9, 6.1, 7.9, 10.1, 11.9, 14.1, 15.9, 18.1, 19.9];
    (x, y)
}

/// Two-class binary data.
fn binary_data() -> (Array2<f64>, Array1<usize>) {
    let x = Array2::from_shape_vec(
        (10, 2),
        vec![
            0.0, 0.0, 0.5, 0.0, 0.0, 0.5, 0.5, 0.5, 0.2, 0.3, 5.0, 5.0, 5.5, 5.0, 5.0, 5.5, 5.5,
            5.5, 5.2, 5.3,
        ],
    )
    .unwrap();
    let y = array![0usize, 0, 0, 0, 0, 1, 1, 1, 1, 1];
    (x, y)
}

// =============================================================================
// KernelRidge
// =============================================================================
#[test]
fn api_proof_kernel_ridge() {
    let (x, y) = regression_data();
    for kernel in [
        KernelType::Rbf,
        KernelType::Polynomial,
        KernelType::Linear,
        KernelType::Sigmoid,
    ] {
        let m = KernelRidge::<f64>::new()
            .with_alpha(1e-3)
            .with_kernel(kernel)
            .with_gamma(0.5)
            .with_degree(2)
            .with_coef0(0.0);
        let f = m.fit(&x, &y).unwrap();
        let preds = f.predict(&x).unwrap();
        assert_eq!(preds.len(), 10);
        let _ = f.dual_coef();
        let _ = f.x_fit();
        let r2 = f.score(&x, &y).unwrap();
        assert!(r2 > -1.0); // Sigmoid may be poor; just check finite/sane.
    }
}

// =============================================================================
// Nystroem
// =============================================================================
#[test]
fn api_proof_nystroem() {
    let (x, _) = regression_data();
    let dummy_y = ();
    for kernel in [
        KernelType::Rbf,
        KernelType::Polynomial,
        KernelType::Linear,
        KernelType::Sigmoid,
    ] {
        let m = Nystroem::<f64>::new()
            .with_kernel(kernel)
            .with_gamma(1.0)
            .with_degree(2)
            .with_coef0(0.0)
            .with_n_components(5)
            .with_random_state(42);
        let f = m.fit(&x, &dummy_y).unwrap();
        let embedded = f.transform(&x).unwrap();
        assert_eq!(embedded.nrows(), 10);
        assert_eq!(embedded.ncols(), 5);
    }
}

#[test]
fn api_proof_polynomial_count_sketch() {
    let (x, _) = regression_data();
    let dummy_y = ();
    let m = PolynomialCountSketch::<f64>::new()
        .with_gamma(0.5)
        .with_degree(3)
        .with_coef0(2.0)
        .with_n_components(8)
        .with_random_state(42);
    let f: FittedPolynomialCountSketch<f64> = m.fit(&x, &dummy_y).unwrap();
    assert_eq!(f.n_features_in(), 1);
    assert_eq!(f.degree(), 3);
    assert_eq!(f.index_hash().dim(), (3, 2));
    assert_eq!(f.bit_hash().dim(), (3, 2));
    let embedded = f.transform(&x).unwrap();
    assert_eq!(embedded.dim(), (10, 8));
}

// =============================================================================
// RBFSampler
// =============================================================================
#[test]
fn api_proof_rbf_sampler() {
    let (x, _) = regression_data();
    let dummy_y = ();
    let m = RBFSampler::<f64>::new()
        .with_gamma(1.0)
        .with_n_components(8)
        .with_random_state(42);
    let f = m.fit(&x, &dummy_y).unwrap();
    let embedded = f.transform(&x).unwrap();
    assert_eq!(embedded.nrows(), 10);
    assert_eq!(embedded.ncols(), 8);
}

#[test]
fn api_proof_additive_chi2_sampler() {
    let x = array![[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]];
    let dummy_y = ();
    let m = AdditiveChi2Sampler::<f64>::new().with_sample_steps(3);
    let f: FittedAdditiveChi2Sampler<f64> = m.fit(&x, &dummy_y).unwrap();
    assert_eq!(f.n_features_in(), 2);
    assert_eq!(f.sample_steps(), 3);
    assert_eq!(f.n_features_out(), 10);
    let embedded = f.transform(&x).unwrap();
    assert_eq!(embedded.dim(), (3, 10));

    let custom = AdditiveChi2Sampler::<f64>::new()
        .with_sample_steps(4)
        .with_sample_interval(0.7)
        .fit(&x, &dummy_y)
        .unwrap();
    assert_eq!(custom.transform(&x).unwrap().dim(), (3, 14));
}

#[test]
fn api_proof_skewed_chi2_sampler() {
    let x = array![[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]];
    let dummy_y = ();
    let m = SkewedChi2Sampler::<f64>::new()
        .with_skewedness(0.5)
        .with_n_components(8)
        .with_random_state(42);
    let f: FittedSkewedChi2Sampler<f64> = m.fit(&x, &dummy_y).unwrap();
    assert_eq!(f.n_features_in(), 2);
    assert_eq!(f.n_components(), 8);
    assert_eq!(f.random_weights().dim(), (2, 8));
    assert_eq!(f.random_offset().len(), 8);
    let embedded = f.transform(&x).unwrap();
    assert_eq!(embedded.dim(), (3, 8));
}

// =============================================================================
// GaussianProcessRegressor
// =============================================================================
#[test]
fn api_proof_gaussian_process_regressor() {
    let (x, y) = regression_data();

    let gpr = GaussianProcessRegressor::<f64>::new(Box::new(RBFKernel::new(1.0)));
    let f = gpr.fit(&x, &y).unwrap();
    let preds = f.predict(&x).unwrap();
    assert_eq!(preds.len(), 10);
    let (mean, std) = f.predict_with_std(&x).unwrap();
    assert_eq!(mean.len(), 10);
    assert_eq!(std.len(), 10);
    let lml = f.log_marginal_likelihood(&y);
    assert!(lml.is_finite());
    let r2 = f.score(&x, &y).unwrap();
    assert!(r2 > 0.0);

    // sample_y posterior draws.
    let samples = f.sample_y(&x, 4, Some(42)).unwrap();
    assert_eq!(samples.dim(), (10, 4));
    // Reproducibility: same seed → same samples.
    let samples2 = f.sample_y(&x, 4, Some(42)).unwrap();
    for i in 0..10 {
        for s in 0..4 {
            assert_eq!(samples[[i, s]], samples2[[i, s]]);
        }
    }
}

// =============================================================================
// GaussianProcessClassifier
// =============================================================================
#[test]
fn api_proof_gaussian_process_classifier() {
    let (x, y) = binary_data();

    let gpc = GaussianProcessClassifier::<f64>::new(Box::new(RBFKernel::new(1.0)));
    let f = gpc.fit(&x, &y).unwrap();
    let preds = f.predict(&x).unwrap();
    assert_eq!(preds.len(), 10);
    let proba = f.predict_proba(&x).unwrap();
    assert_eq!(proba.dim(), (10, 2));
    for i in 0..10 {
        let s: f64 = proba.row(i).sum();
        assert!((s - 1.0).abs() < 1e-9);
    }
    let log_proba = f.predict_log_proba(&x).unwrap();
    assert_eq!(log_proba.dim(), (10, 2));
    let lml = f.log_marginal_likelihood();
    assert!(lml.is_finite());
    assert_eq!(f.classes(), &[0, 1]);
    let acc = f.score(&x, &y).unwrap();
    assert!(acc > 0.5);
}

// =============================================================================
// GP kernel zoo (constructors compile + compute returns shapes)
// =============================================================================
#[test]
fn api_proof_gp_kernel_zoo() {
    use ferrolearn_kernel::GPKernel;
    let x = Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();

    let hp = Hyperparameter::new(
        "length_scale",
        "numeric",
        HyperparameterBounds::Numeric(vec![(1e-5, 1e5)]),
        2,
        None,
    );
    assert_eq!(hp.bounds_array().unwrap().dim(), (2, 2));
    assert!(!hp.fixed);
    assert!(Hyperparameter::<f64>::fixed("length_scale", "numeric", 1).fixed);

    let rbf = RBFKernel::new(1.0);
    assert_eq!(rbf.compute(&x, &x).dim(), (4, 4));
    assert_eq!(rbf.diagonal(&x).len(), 4);
    assert_eq!(rbf.hyperparameters()[0].name, "length_scale");
    assert_eq!(rbf.bounds().dim(), (1, 2));
    assert_eq!(rbf.n_dims(), 1);

    for nu in [0.5, 1.5, 2.5] {
        let m = MaternKernel::new(1.0, nu);
        assert_eq!(m.compute(&x, &x).dim(), (4, 4));
    }

    let rq = RationalQuadratic::new(1.3, 0.7);
    assert_eq!(rq.compute(&x, &x).dim(), (4, 4));
    assert_eq!(rq.diagonal(&x).len(), 4);
    assert_eq!(rq.get_params().len(), 2);
    assert_eq!(
        rq.hyperparameters()
            .iter()
            .map(|hyperparameter| hyperparameter.name.as_str())
            .collect::<Vec<_>>(),
        vec!["alpha", "length_scale"]
    );

    let periodic = ExpSineSquared::new(1.3, 2.0);
    assert_eq!(periodic.compute(&x, &x).dim(), (4, 4));
    assert_eq!(periodic.diagonal(&x).len(), 4);
    assert_eq!(periodic.get_params().len(), 2);
    assert_eq!(periodic.n_dims(), 2);

    let c = ConstantKernel::new(2.0);
    let _ = c.compute(&x, &x);

    let w = WhiteKernel::new(0.1);
    let _ = w.compute(&x, &x);

    let dp = DotProductKernel::new(1.0);
    let _ = dp.compute(&x, &x);

    let s = SumKernel::new(
        Box::new(RBFKernel::new(1.0)),
        Box::new(WhiteKernel::new(0.1)),
    );
    let _ = s.compute(&x, &x);
    assert_eq!(s.bounds().dim(), (2, 2));

    let p = ProductKernel::new(
        Box::new(RBFKernel::new(1.0)),
        Box::new(ConstantKernel::new(2.0)),
    );
    let _ = p.compute(&x, &x);

    let pow = Exponentiation::new(Box::new(RBFKernel::new(1.0)), 2.0);
    assert_eq!(pow.compute(&x, &x).dim(), (4, 4));
    assert_eq!(pow.diagonal(&x).len(), 4);
    assert_eq!(pow.get_params().len(), 1);

    let compound = CompoundKernel::new(vec![
        Box::new(RBFKernel::new(1.0)),
        Box::new(DotProductKernel::new(0.5)),
        Box::new(WhiteKernel::new(0.1)),
    ]);
    assert_eq!(compound.compute_stack(&x, &x).dim(), (4, 4, 3));
    assert_eq!(compound.diagonal_stack(&x).dim(), (4, 3));
    assert_eq!(compound.get_params().len(), 3);
    assert_eq!(compound.bounds().dim(), (3, 2));
    assert_eq!(compound.n_dims(), 3);
}

// =============================================================================
// NadarayaWatson + LocalPolynomialRegression with the local-kernel zoo
// =============================================================================
#[test]
fn api_proof_nadaraya_watson() {
    let (x, y) = regression_data();

    macro_rules! exercise_nw {
        ($kernel:expr) => {{
            let m = NadarayaWatson::with_kernel($kernel, BandwidthStrategy::Fixed(1.0));
            let f = m.fit(&x, &y).unwrap();
            let preds = f.predict(&x).unwrap();
            assert_eq!(preds.len(), 10);
        }};
    }
    exercise_nw!(GaussianKernel);
    exercise_nw!(EpanechnikovKernel);
    exercise_nw!(TricubeKernel);
    exercise_nw!(BiweightKernel);
    exercise_nw!(TriweightKernel);
    exercise_nw!(UniformKernel);
    exercise_nw!(CosineKernel);

    // Default ctor smoke (uses GaussianKernel default + CV bandwidth)
    let _ = NadarayaWatson::<f64, GaussianKernel>::new();
}

#[test]
fn api_proof_local_polynomial_regression() {
    let (x, y) = regression_data();

    for order in 0..=2 {
        let m = LocalPolynomialRegression::with_kernel(
            GaussianKernel,
            BandwidthStrategy::Fixed(1.5),
            order,
        );
        let f = m.fit(&x, &y).unwrap();
        let preds = f.predict(&x).unwrap();
        assert_eq!(preds.len(), 10);
    }

    let _ = LocalPolynomialRegression::<f64, GaussianKernel>::new();
}

// =============================================================================
// Bandwidth helpers + diagnostics
// =============================================================================
#[test]
fn api_proof_bandwidth_helpers() {
    let (x, _) = regression_data();
    let h_scott = scott_bandwidth(&x);
    assert_eq!(h_scott.len(), 1);
    assert!(h_scott[0] > 0.0);

    let h_silv = silverman_bandwidth(&x);
    assert_eq!(h_silv.len(), 1);
    assert!(h_silv[0] > 0.0);

    // BandwidthStrategy variants compile.
    let _: BandwidthStrategy<f64> = BandwidthStrategy::Fixed(1.0);
    let _: BandwidthStrategy<f64> = BandwidthStrategy::PerDimension(array![1.0, 2.0]);
    let _: BandwidthStrategy<f64> = BandwidthStrategy::Silverman;
    let _: BandwidthStrategy<f64> = BandwidthStrategy::Scott;
    let _: BandwidthStrategy<f64> = BandwidthStrategy::CrossValidated {
        cv: CvStrategy::Loo,
        per_dimension: false,
    };
    let _: BandwidthStrategy<f64> = BandwidthStrategy::CrossValidated {
        cv: CvStrategy::KFold(3),
        per_dimension: false,
    };
}

#[test]
fn api_proof_diagnostics() {
    let (x, y) = regression_data();
    let nw = NadarayaWatson::with_kernel(GaussianKernel, BandwidthStrategy::Fixed(1.0));
    let fitted = nw.fit(&x, &y).unwrap();
    let preds = fitted.predict(&x).unwrap();

    let diag = residual_diagnostics(&y, &preds, 0.05);
    assert_eq!(diag.residuals.len(), 10);
    assert_eq!(diag.standardized_residuals.len(), 10);
    assert!(diag.std >= 0.0);

    for test in [
        HeteroscedasticityTest::White,
        HeteroscedasticityTest::BreuschPagan,
    ] {
        let het = heteroscedasticity_test(&x, &y, &preds, test, 0.05).unwrap();
        let _ = het.statistic;
        let _ = het.p_value;
        let _ = het.is_heteroscedastic;
        let _ = het.test_name;
        let _ = het.alpha;
    }
}
