//! Divergence pin: `IncrementalPCA` `batch_size=None` default.
//!
//! sklearn `IncrementalPCA.fit` sets `self.batch_size_ = 5 * n_features` when
//! `batch_size is None` (`sklearn/decomposition/_incremental_pca.py:236-237`),
//! so the data is processed in mini-batches of `5*n_features` rows. ferrolearn's
//! `IncrementalPCA` defaults `batch_size = None -> n_samples`
//! (`incremental_pca.rs:626`: `None => n_samples`), i.e. a SINGLE full-data
//! batch (equivalent to plain PCA). When `5*n_features < n_samples` these
//! produce DIFFERENT incremental-SVD spectra.
//!
//! This is OUT OF SCOPE for the #2386 fix (svd_lapack engine + gen_batches merge
//! given an EXPLICIT batch_size, both verified clean across the config grid);
//! it is the pre-documented REQ-13 default-batch_size gap.
//!
//! Live sklearn 1.5.2 oracle (R-CHAR-3), run from `/tmp`:
//! ```text
//! X = RandomState(0).randn(30, 3)
//! IncrementalPCA(n_components=2).fit(X)   # batch_size_ = 5*3 = 15 -> batches 15+15
//!   singular_values_ = [6.37341146, 5.27463845]
//! ```
//! ferrolearn (single 30-row batch) gives `[6.38417462, 5.30464741]` — a
//! ~1.1e-2 / ~3e-2 divergence, ~1e4x the R-DEV-1 ~1e-6 budget.
//!
//! Tracking: #2387 (blocker) / REQ-13 #1589 (NOT-STARTED).

use ferrolearn_core::traits::Fit;
use ferrolearn_decomp::IncrementalPCA;
use ndarray::Array2;

/// `np.random.RandomState(0).randn(30, 3)`, row-major.
#[allow(clippy::excessive_precision, reason = "live sklearn 1.5.2 oracle (R-CHAR-3)")]
const X: [f64; 90] = [
    1.764052345968, 0.400157208367, 0.978737984106, 2.240893199201, 1.86755799015, -0.977277879876,
    0.950088417526, -0.151357208298, -0.103218851794, 0.410598501938, 0.144043571161,
    1.454273506963, 0.761037725147, 0.121675016493, 0.443863232745, 0.333674327374,
    1.494079073158, -0.205158263766, 0.313067701651, -0.854095739302, -2.552989815834,
    0.65361859544, 0.86443619886, -0.742165020406, 2.269754623988, -1.454365674599,
    0.045758517301, -0.187183850026, 1.532779214358, 1.4693587699, 0.154947425697, 0.378162519602,
    -0.88778574763, -1.980796468224, -0.347912149326, 0.156348969104, 1.230290680728,
    1.202379848784, -0.387326817408, -0.302302750575, -1.048552965067, -1.420017937179,
    -1.706270190625, 1.950775395232, -0.509652181752, -0.438074301611, -1.25279536005,
    0.777490355832, -1.613897847558, -0.212740280214, -0.895466561194, 0.386902497859,
    -0.510805137569, -1.180632184122, -0.028182228339, 0.42833187053, 0.066517222383, 0.30247189774,
    -0.634322093681, -0.362741165987, -0.672460447776, -0.359553161541, -0.813146282044,
    -1.726282602332, 0.177426142254, -0.401780936208, -1.630198346966, 0.462782255526,
    -0.907298364383, 0.051945395796, 0.729090562178, 0.128982910757, 1.139400684543,
    -1.234825820354, 0.402341641178, -0.68481009094, -0.870797149182, -0.578849664764,
    -0.311552532127, 0.05616534223, -1.165149840783, 0.900826486954, 0.46566243973,
    -1.536243686277, 1.488252193796, 1.895889176031, 1.17877957116, -0.179924835812,
    -1.070752621511, 1.054451726931,
];

/// Divergence: `IncrementalPCA(n_components=2).fit(X)` with the DEFAULT
/// `batch_size` (None). sklearn uses `batch_size_ = 5*n_features = 15`
/// (`_incremental_pca.py:236-237`) → batches 15+15, giving
/// `singular_values_ = [6.37341146, 5.27463845]`. ferrolearn defaults to a
/// single full-data batch (`incremental_pca.rs:626`), giving
/// `[6.38417462, 5.30464741]` — diverges ~1.1e-2 / ~3e-2 (>> 1e-6).
/// Tracking: #2387 (REQ-13 #1589)
#[test]
#[ignore = "divergence: IncrementalPCA batch_size=None default is n_samples, sklearn uses 5*n_features; tracking #2387 (REQ-13 #1589)"]
fn divergence_batch_size_default_5x_n_features() {
    let x = Array2::from_shape_vec((30, 3), X.to_vec()).unwrap();
    let f = IncrementalPCA::<f64>::new(2).fit(&x, &()).unwrap();
    // sklearn batch_size_ = 5*3 = 15 -> batches 15+15.
    #[allow(clippy::excessive_precision, reason = "oracle")]
    let sk_sv = [6.37341146, 5.27463845];
    let sv = f.singular_values();
    for k in 0..2 {
        let diff = (sv[k] - sk_sv[k]).abs();
        assert!(
            diff < 1e-6,
            "singular_values_[{k}] = {} but sklearn (batch_size_=15) = {} (diff {diff})",
            sv[k],
            sk_sv[k]
        );
    }
}
