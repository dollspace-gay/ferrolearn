//! Divergence pin: `SimpleImputer<f32>` Mean computes the per-column mean by
//! COMPRESSING the observed (non-NaN) values into a contiguous slice and running
//! numpy's pairwise sum over that compressed slice. scikit-learn 1.5.2 computes
//! it via `np.ma.mean(masked_X, axis=0)` (`sklearn/impute/_base.py:498`), whose
//! reduction is `MaskedArray.sum`, which "set[s] masked elements to 0
//! internally" and sums the FULL-LENGTH array
//! (`numpy/ma/core.py:5242,5251`: `result = self.filled(0).sum(...)`).
//!
//! numpy's pairwise summation tree shape depends on the array LENGTH and the
//! POSITIONS of the elements (the split points + 8-way base-case grouping are
//! computed over N total positions, with NaN positions contributing a 0). When
//! NaN is present, the full-length tree (length N, masked->0) and ferrolearn's
//! compressed tree (length = observed count) round to DIFFERENT f32 values: the
//! zeros sit at different positions in the balanced tree, so the f32 partial
//! sums differ by 1-6 ULPs.
//!
//! When there is NO NaN in a column the two agree (the compressed length equals
//! N), which is why the no-NaN size sweep (probe) passes; this divergence only
//! manifests for columns with scattered NaN.
//!
//! All expected values come from the LIVE sklearn 1.5.2 / numpy 2.x oracle. The
//! 200-element f32 column below (130 observed, 70 NaN) is
//! `np.random.RandomState(330)`: `a = uniform(-1e7,1e7,200).astype(f32)`;
//! `nan_pos = choice(200,70,replace=False)`; `col=a; col[nan_pos]=nan`. Each
//! entry is reconstructed from its EXACT f32 bit pattern via `f32::from_bits`
//! (NaN bits 2143289344 round-trip to NaN, filtered by `is_nan()`), so the Rust
//! input column is bit-identical to sklearn's input.
//!
//! Oracle probe (run from /tmp):
//! ```text
//! import numpy as np; from sklearn.impute import SimpleImputer
//! rng = np.random.RandomState(330)
//! a = rng.uniform(-1e7,1e7,200).astype(np.float32)
//! nan_pos = rng.choice(200,70,replace=False)
//! col = a.copy(); col[nan_pos]=np.nan
//! imp = SimpleImputer(strategy='mean').fit(col.reshape(-1,1))
//! Xn = np.vstack([col.reshape(-1,1),[[np.nan]]]).astype(np.float32)
//! out = imp.transform(Xn)
//! # out[-1,0] -> float32 -189490.19   (bits 3359181964)
//! ```
//! ferrolearn `SimpleImputer::<f32>::new(Mean)` pairwise-sums the COMPRESSED 130
//! observed values: fill = -189490.1 (bits 3359181958) — diverges by 6 f32 ULPs.
//!
//! Tracking: #2309.

use ferrolearn_core::traits::{Fit, Transform};
use ferrolearn_preprocess::imputer::{ImputeStrategy, SimpleImputer};
use ndarray::{Array2, Axis, array, concatenate};

/// 200-element f32 column (RandomState(330), 130 observed + 70 NaN), as exact
/// f32 bit patterns. NaN bits `2143289344` round-trip via `f32::from_bits`.
#[rustfmt::skip]
const COL_BITS: [u32; 200] = [2143289344,2143289344,2143289344,2143289344,2143289344,2143289344,2143289344,1245981046,3397970134,3404951866,2143289344,1253657175,1253535557,2143289344,3401050465,1255531825,1258529118,3385837090,3376457779,2143289344,1250236165,2143289344,2143289344,3400592630,2143289344,2143289344,1252462337,1212260169,3370007227,3392047562,2143289344,3406637318,3395881310,2143289344,3388459280,1239623381,2143289344,1258109635,2143289344,2143289344,1256848778,1243305197,3372225111,1254975330,2143289344,2143289344,1226870622,1253516291,3397956189,1256907903,1246631043,1237180798,2143289344,2143289344,1254926823,1195430509,3382997629,2143289344,2143289344,1255144124,3375607864,2143289344,3400784762,1250117190,3395542974,3396502241,1254924441,1259891921,3387290252,1217403170,1255154132,1230929956,2143289344,1243932160,1259836190,3402979872,3405068918,2143289344,2143289344,2143289344,3405859536,2143289344,3399084296,2143289344,3404568778,1249024076,1228733634,3407026842,1250031437,3401394544,3388504378,2143289344,1259814627,3400158075,2143289344,1244670311,3404344287,1243920424,1225716671,3349424317,1256085100,2143289344,2143289344,1243998245,2143289344,3403580709,1247007472,2143289344,2143289344,2143289344,3395935825,2143289344,3407359381,2143289344,3400608720,3389281837,3398611691,2143289344,1235618883,2143289344,2143289344,2143289344,3388112431,1221381732,1254348655,2143289344,3403674716,1250771374,2143289344,2143289344,2143289344,3401814824,3376133887,3396050236,1235719179,3406714748,1243638291,2143289344,2143289344,2143289344,2143289344,1255907316,3384185290,2143289344,2143289344,3406729337,3406604974,3402064689,1251985747,3406799684,3396506780,1255183648,2143289344,3385799440,3397583046,2143289344,3399986254,2143289344,1240842418,3390450854,3406219063,1256200811,2143289344,2143289344,1251104981,1243842834,2143289344,3403072680,3402068794,3406956839,2143289344,3400961031,1245321829,1250261014,1236542519,1253897767,1254847719,3395643710,3406774069,2143289344,1249629668,2143289344,1253727427,2143289344,1258669449,2143289344,2143289344,3395187604,2143289344,3395592069,1256589592,1244663828,3372438982,1241339762,1255200361,3388972339,1256010286,2143289344,3406609421,3392233714];

/// Divergence: `SimpleImputer::<f32>` Mean pairwise-sums the COMPRESSED observed
/// values, whereas `sklearn/impute/_base.py:498` (`np.ma.mean`) sums the
/// FULL-LENGTH masked array with NaN->0 (`numpy/ma/core.py:5242,5251`,
/// `self.filled(0).sum(...)`). The pairwise tree shape differs when NaN is
/// present.
///
/// Oracle (sklearn 1.5.2): the 200-element column above (130 observed),
/// `strategy='mean'`, transform a trailing NaN row -> fill = `-189490.19f32`
/// (bits `3359181964`). ferrolearn compresses then pairwise-sums -> `-189490.1f32`
/// (bits `3359181958`), differing by 6 f32 ULPs.
#[ignore = "divergence: SimpleImputer<f32> mean compresses observed values before pairwise_sum, sklearn np.ma.mean sums full masked array (filled-0); tracking #2309"]
#[test]
fn divergence_f32_mean_nan_scatter_compressed_vs_full_pairwise() {
    // sklearn oracle: exact f32 bit pattern of out[-1,0].
    const SKLEARN_FILL_BITS: u32 = 3_359_181_964; // -189490.19f32

    let col: Vec<f32> = COL_BITS.iter().map(|&b| f32::from_bits(b)).collect();
    let col2: Array2<f32> = Array2::from_shape_vec((200, 1), col).unwrap();
    let nan_row: Array2<f32> = array![[f32::NAN]];
    let x_transform = concatenate(Axis(0), &[col2.view(), nan_row.view()]).unwrap();

    let fitted = SimpleImputer::<f32>::new(ImputeStrategy::Mean)
        .fit(&col2, &())
        .expect("fit should succeed");
    let out = fitted
        .transform(&x_transform)
        .expect("transform should succeed");

    let ferro_fill: f32 = out[[200, 0]];
    let sklearn_fill = f32::from_bits(SKLEARN_FILL_BITS);

    assert_eq!(
        ferro_fill.to_bits(),
        sklearn_fill.to_bits(),
        "f32 mean nan-scatter fill: ferrolearn={ferro_fill} (bits {}) != sklearn={sklearn_fill} (bits {SKLEARN_FILL_BITS})",
        ferro_fill.to_bits(),
    );
}
