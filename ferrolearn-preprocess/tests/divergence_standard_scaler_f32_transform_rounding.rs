//! Divergence pin: ferrolearn `FittedStandardScaler::<f32>::transform` rounds the
//! center+scale ONCE (fused f64 expression `(x_f64 - mean_)/scale_`, then a single
//! cast to f32), whereas scikit-learn 1.5.2 applies the centering and the scaling
//! as TWO SEPARATE in-place float32 operations, each rounded back to f32.
//!
//! sklearn `StandardScaler.transform` (`sklearn/preprocessing/_data.py:1064-1067`):
//! ```python
//! if self.with_mean:
//!     X -= self.mean_     # float32 array -= float64 scalar -> result rounded to f32
//! if self.with_std:
//!     X /= self.scale_    # float32 array /= float64 scalar -> result rounded to f32
//! ```
//! Because `X` is a float32 ndarray, numpy performs each in-place op in float64 and
//! stores the result back into the f32 array, producing TWO float32 roundings.
//!
//! ferrolearn (`ferrolearn-preprocess/src/standard_scaler.rs:519-528`):
//! ```rust
//! let mut acc = v.to_f64();   // f32 -> f64 (exact)
//! if self.with_mean { acc -= m; }   // f64
//! if self.with_std  { acc /= s; }   // f64
//! *v = F::from(acc);          // ONE rounding f64 -> f32 at the very end
//! ```
//! i.e. the fused expression `(x_f64 - mean_)/scale_` rounded a SINGLE time.
//!
//! On a large-magnitude f32 column near 2^24 (the exact regime #2205 was fixed
//! for) the two paths differ by 1 ULP per element. The existing pinned test
//! `divergence_f32_uses_float64_accumulators` uses tolerance `1e-5`, which is too
//! loose to catch this 1-ULP gap; this test pins it bit-exactly.
//!
//! Live sklearn 1.5.2 oracle (run from /tmp):
//!   X = np.array([[16777216.],[16777216.],[16777220.]], dtype=np.float32)
//!   m = StandardScaler().fit(X)
//!   m.transform(X).ravel()  ->
//!     array([-0.7071068, -0.7071068, 1.4142137], dtype=float32)
//!   exact f32 reprs:
//!     -0.7071068286895752  (bits 0x{neg}40cafb0c)
//!     -0.7071068286895752
//!      1.4142136573791504  (bits 0x3fb504f4)
//! ferrolearn's fused-1-cast path instead yields:
//!     -0.7071067690849304  (bits 0x{neg}40cafb0d)   <- 1 ULP below sklearn
//!     -0.7071067690849304
//!      1.4142135381698608  (bits 0x3fb504f3)        <- 1 ULP below sklearn
//!
//! Tracking: #2305

use ferrolearn_core::traits::{Fit, Transform};
use ferrolearn_preprocess::StandardScaler;
use ndarray::{array, Array2};

/// Pin: f32 transform must match sklearn's two-step (per-op f32-rounded) result
/// bit-for-bit. ferrolearn's single-rounding fused path is 1 ULP off.
#[test]
#[ignore = "divergence: StandardScaler<f32>::transform rounds once (fused f64) vs sklearn's two per-op f32 roundings (1 ULP off near 2^24); tracking #2305"]
fn divergence_f32_transform_per_op_f32_rounding() {
    // Live sklearn 1.5.2 oracle f32 transform output (exact f32 values).
    let sk_transform: [f32; 3] = [
        -0.7071068286895752_f32,
        -0.7071068286895752_f32,
        1.4142136573791504_f32,
    ];

    let scaler = StandardScaler::<f32>::new();
    let x: Array2<f32> = array![[16777216.0f32], [16777216.0f32], [16777220.0f32]];
    let fitted = scaler.fit(&x, &()).unwrap();
    let scaled = fitted.transform(&x).unwrap();

    for i in 0..3 {
        let got = scaled[[i, 0]];
        assert_eq!(
            got.to_bits(),
            sk_transform[i].to_bits(),
            "transform[{i}]: ferrolearn={got:?} (bits {:#x}), sklearn={:?} (bits {:#x})",
            got.to_bits(),
            sk_transform[i],
            sk_transform[i].to_bits()
        );
    }
}
