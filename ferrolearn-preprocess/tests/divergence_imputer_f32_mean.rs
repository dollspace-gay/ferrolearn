//! Divergence pin: `SimpleImputer<f32>` Mean accumulates the per-column mean in
//! f32, whereas scikit-learn 1.5.2 computes it via `np.ma.mean`
//! (`sklearn/impute/_base.py:497-501`:
//! `mean_masked = np.ma.mean(masked_X, axis=0)`) as a float32 pairwise sum
//! (numpy ma `filled(0).sum()`, `numpy/ma/core.py:5242,5251`), then a float64
//! division by the observed count (bit-equivalent to f32 division here). On
//! `transform`, sklearn fills the missing entries from that statistic, rounding
//! to f32 only at the assignment into the float32 output array
//! (`_base.py:625-636`). For a column whose naive sequential f32 accumulation
//! loses precision, ferrolearn's fill value diverges from sklearn's by many f32
//! ULPs (NOT a 1-ULP rounding difference).
//!
//! All expected values come from the LIVE sklearn 1.5.2 oracle. The 200-element
//! f32 column below is `np.random.seed(0); np.random.uniform(-1e6,1e6,200)
//! .astype(np.float32)`; each decimal literal round-trips EXACTLY to the same
//! f32 bit pattern as the oracle's input (verified via `.view(np.uint32)`), so
//! the Rust input column is bit-identical to sklearn's.
//!
//! Oracle probe (run from /tmp):
//! ```text
//! import numpy as np; from sklearn.impute import SimpleImputer
//! np.random.seed(0)
//! a = np.random.uniform(-1e6, 1e6, 200).astype(np.float32)
//! imp = SimpleImputer(strategy='mean').fit(a.reshape(-1,1))
//! Xn = np.vstack([a.reshape(-1,1), [[np.nan]]]).astype(np.float32)
//! out = imp.transform(Xn)
//! # imp.statistics_  -> float64  np.float64(875.595)
//! # out[-1,0]        -> float32  875.594970703125   (bits 1146807828)
//! ```
//! ferrolearn `SimpleImputer::<f32>::new(Mean)` fold-accumulates in f32:
//! fill value = 875.5890502929688 (bits 1146807731) — diverges by ~6e-3.
//!
//! Tracking: #2308.

use ferrolearn_core::traits::{Fit, Transform};
use ferrolearn_preprocess::imputer::{ImputeStrategy, SimpleImputer};
use ndarray::{Array2, Axis, array, concatenate};

/// The 200-element f32 column (seed 0). Bit-identical to the sklearn oracle's
/// `np.random.uniform(-1e6,1e6,200).astype(np.float32)`.
#[rustfmt::skip]
#[allow(
    clippy::excessive_precision,
    reason = "each literal is the EXACT decimal expansion of the oracle's f32 bit pattern; truncating to fewer digits (as clippy suggests) would round to a DIFFERENT f32 and break bit-identity with sklearn's input column"
)]
const COL: [f32; 200] = [
    97627.0078125f32, 430378.71875f32, 205526.75f32, 89766.3671875f32, -152690.40625f32, 291788.21875f32, -124825.578125f32, 783546.0f32, 927325.5f32, -233116.96875f32, 583450.0625f32, 57789.83984375f32, 136089.125f32, 851193.25f32, -857927.875f32, -825741.375f32, -959563.1875f32, 665239.6875f32, 556313.5f32, 740024.3125f32, 957236.6875f32, 598317.125f32, -77041.2734375f32, 561058.375f32, -763451.125f32, 279842.03125f32, -713293.4375f32, 889337.8125f32, 43696.64453125f32, -170676.125f32, -470888.78125f32, 548467.375f32, -87699.3359375f32, 136867.890625f32, -962420.375f32, 235271.0f32, 224191.453125f32, 233868.0f32, 887496.1875f32, 363640.59375f32, -280984.1875f32, -125936.09375f32, 395262.40625f32, -879549.0625f32, 333533.4375f32, 341275.75f32, -579234.875f32, -742147.375f32, -369143.3125f32, -272578.46875f32, 140393.546875f32, -122796.9765625f32, 976747.6875f32, -795910.375f32, -582246.5f32, -677380.9375f32, 306216.65625f32, -493416.78125f32, -67378.453125f32, -511148.8125f32, -682060.8125f32, -779249.6875f32, 312659.1875f32, -723634.125f32, -606835.25f32, -262549.65625f32, 641986.4375f32, -805797.4375f32, 675889.8125f32, -807803.1875f32, 952918.9375f32, -62697.59765625f32, 953522.1875f32, 209691.046875f32, 478527.15625f32, -921624.4375f32, -434386.0625f32, -759606.875f32, -407719.59375f32, -762544.5625f32, -364033.65625f32, -171474.015625f32, -871705.0f32, 384944.25f32, 133202.90625f32, -469221.03125f32, 46496.10546875f32, -812119.0f32, 151892.984375f32, 858592.375f32, -362862.09375f32, 334820.75f32, -736404.25f32, 432654.40625f32, -421187.8125f32, -633617.25f32, 173025.875f32, -959784.9375f32, 657880.0625f32, -990609.0625f32, 355633.0625f32, -459984.0625f32, 470388.03125f32, 924377.0625f32, -502493.71875f32, 152314.671875f32, 184083.859375f32, 144503.8125f32, -553836.75f32, 905498.0f32, -105749.2421875f32, 692817.375f32, 398958.5625f32, -405126.09375f32, 627595.625f32, -206988.515625f32, 762206.375f32, 162545.75f32, 763470.75f32, 385063.1875f32, 450508.5625f32, 2648.763916015625f32, 912167.25f32, 287980.40625f32, -152289.90625f32, 212786.421875f32, -961613.625f32, -396850.375f32, 320347.0625f32, -419844.78125f32, 236030.859375f32, -142462.59375f32, -729051.875f32, -403435.34375f32, 139929.828125f32, 181745.515625f32, 148650.5f32, 306401.625f32, 304206.53125f32, -137163.125f32, 793093.1875f32, -264876.25f32, -128270.1484375f32, 783846.6875f32, 612388.0f32, 407777.15625f32, -799546.25f32, 838965.25f32, 428482.59375f32, 997694.0f32, -701103.375f32, 736252.125f32, -675014.125f32, 231119.125f32, -752360.0625f32, 696016.4375f32, 614637.9375f32, 138201.484375f32, -185633.40625f32, -861666.0f32, 394857.53125f32, -92914.6328125f32, 444111.1875f32, 732764.625f32, 951043.0f32, 711606.6875f32, -976571.8125f32, -280043.875f32, 459981.125f32, -656740.625f32, 42073.2109375f32, -891324.0f32, -600006.9375f32, -962956.4375f32, 587395.4375f32, -552150.625f32, -309296.625f32, 856162.5625f32, 408828.8125f32, -936322.125f32, -670611.6875f32, 242956.796875f32, 154457.171875f32, -524214.34375f32, 868428.0f32, 227931.90625f32, 71265.609375f32, 179819.953125f32, 460244.0625f32, -376110.0f32, -203557.875f32, -580312.5f32, -627614.0f32, 888744.75f32, 479101.59375f32, -19082.3828125f32, -545170.75f32, -491287.03125f32, -883941.6875f32, -131166.75f32,
];

/// Divergence: `SimpleImputer::<f32>` Mean transform-fill diverges from
/// `sklearn/impute/_base.py:497-501` (float64 `np.ma.mean`).
///
/// Oracle (sklearn 1.5.2): the seed-0 f32 column above, `strategy='mean'`,
/// transform a trailing NaN row -> fill = `875.594970703125f32`
/// (bits `1146807828`). ferrolearn fold-accumulates in f32 ->
/// `875.5890502929688f32` (bits `1146807731`), differing by ~6e-3 (>> f32 ULP).
#[test]
fn divergence_f32_mean_transform_fill_float64_accumulation() {
    // sklearn oracle: exact f32 bit pattern of out[-1,0].
    const SKLEARN_FILL_BITS: u32 = 1_146_807_828; // 875.594970703125f32

    let col: Array2<f32> = Array2::from_shape_vec((200, 1), COL.to_vec()).unwrap();
    let nan_row: Array2<f32> = array![[f32::NAN]];
    let x_train = col.clone();
    let x_transform = concatenate(Axis(0), &[col.view(), nan_row.view()]).unwrap();

    let fitted = SimpleImputer::<f32>::new(ImputeStrategy::Mean)
        .fit(&x_train, &())
        .expect("fit should succeed");
    let out = fitted
        .transform(&x_transform)
        .expect("transform should succeed");

    let ferro_fill: f32 = out[[200, 0]];
    let sklearn_fill = f32::from_bits(SKLEARN_FILL_BITS);

    // sklearn fills with the float64-mean value rounded to f32; ferrolearn's
    // f32 accumulation diverges by many ULPs. Asserting bit-exact match
    // (the documented R-DEV-1 numerical-parity contract) FAILS today.
    assert_eq!(
        ferro_fill.to_bits(),
        sklearn_fill.to_bits(),
        "f32 mean transform fill: ferrolearn={ferro_fill} (bits {}) != sklearn={sklearn_fill} (bits {SKLEARN_FILL_BITS})",
        ferro_fill.to_bits(),
    );
}
