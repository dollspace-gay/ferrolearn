//! Divergence pins for `QuantileTransformer` vs scikit-learn 1.5.2.
//!
//! Oracle = installed sklearn 1.5.2 (`QuantileTransformer`,
//! `sklearn/preprocessing/_data.py`). To neutralise the NOT-STARTED random
//! subsample path (REQ-6), every fixture uses `subsample` larger than
//! `n_samples` (sklearn `subsample=10**9`) / ferrolearn `subsample=0` (= all),
//! and matches `n_quantiles` so both sides build identical reference grids.
//!
//! R-CHAR-3: every expected value below is produced by a LIVE sklearn call (see
//! the `// oracle:` comment above each constant) or a named sklearn `file:line`
//! symbolic constant — never literal-copied from the ferrolearn side.

use ferrolearn_core::traits::{Fit, FitTransform, Transform};
use ferrolearn_preprocess::quantile_transformer::{OutputDistribution, QuantileTransformer};
use ndarray::array;

// ===========================================================================
// DIV-B (HEADLINE) — Normal output accuracy: exact stats.norm.ppf + clip.
// ===========================================================================

/// Divergence: ferrolearn's `probit` (Abramowitz-Stegun rational approx, ~1e-4
/// error, `quantile_transformer.rs:151-188`) diverges from sklearn's exact
/// `stats.norm.ppf` + `BOUNDS_THRESHOLD` clip
/// (`sklearn/preprocessing/_data.py:2856-2862`:
/// `X_col = stats.norm.ppf(X_col)` then
/// `np.clip(X_col, norm.ppf(1e-7 - spacing(1)), norm.ppf(1 - (1e-7 - spacing(1))))`).
///
/// Input: `[[1],[2],[3],[4],[5]]`, n_quantiles=5, output=Normal.
/// sklearn returns `[-5.199337582605575, -0.6744897501960817, 0.0,
///                   0.6744897501960817, 5.19933758270342]`.
/// ferrolearn's A&S `probit` of {0.25,0.75} returns ~+/-0.6744 off by ~1e-4,
/// and clamps the extremes to `probit(1e-7) ~ -5.199...` only by luck of the
/// A&S formula (the quartiles are the hard fail at 1e-6).
/// Tracking: #1320
#[test]
fn divergence_normal_ppf_accuracy() {
    // oracle: python3 -c "import numpy as np; from sklearn.preprocessing import QuantileTransformer; \
    //   print(QuantileTransformer(n_quantiles=5, output_distribution='normal', subsample=10**9)\
    //   .fit_transform([[1.],[2.],[3.],[4.],[5.]]).ravel().tolist())"
    //   -> [-5.199337582605575, -0.6744897501960817, 0.0, 0.6744897501960817, 5.19933758270342]
    let sk_normal: [f64; 5] = [
        -5.199337582605575,
        -0.6744897501960817,
        0.0,
        0.6744897501960817,
        5.19933758270342,
    ];

    let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
    let qt = QuantileTransformer::<f64>::new(5, OutputDistribution::Normal, 0);
    let out = qt.fit_transform(&x).unwrap();

    for (i, &expected) in sk_normal.iter().enumerate() {
        let got = out[[i, 0]];
        assert!(
            (got - expected).abs() <= 1e-6,
            "row {i}: ferrolearn Normal ppf = {got}, sklearn norm.ppf = {expected} (diff {})",
            (got - expected).abs()
        );
    }
}

// ===========================================================================
// DIV-A — forward+reversed AVERAGED interpolation (plateau -> midpoint).
// ===========================================================================

/// Divergence: ferrolearn's `interpolate_cdf` (forward-only binary search,
/// `quantile_transformer.rs:192-225`) diverges from sklearn's averaged interp
/// `0.5 * (np.interp(X, q, refs) - np.interp(-X, -q[::-1], -refs[::-1]))`
/// (`sklearn/preprocessing/_data.py:2843-2846`).
///
/// Input: `[[1],[2],[2],[2],[3]]`, n_quantiles=5, output=Uniform.
/// quantiles_ = [1,2,2,2,3], references_ = [0,0.25,0.5,0.75,1.0].
/// The value 2.0 sits on the [0.25,0.75] plateau; sklearn averages ascending
/// and descending interp -> the MIDPOINT 0.5. ferrolearn's forward-only interp
/// lands on the upper extreme 0.75.
/// sklearn fit_transform -> [0.0, 0.5, 0.5, 0.5, 1.0].
/// Tracking: #1321
#[test]
fn divergence_averaged_interpolation_plateau() {
    // oracle: python3 -c "import numpy as np; from sklearn.preprocessing import QuantileTransformer; \
    //   print(QuantileTransformer(n_quantiles=5, output_distribution='uniform', subsample=10**9)\
    //   .fit_transform([[1.],[2.],[2.],[2.],[3.]]).ravel().tolist())"
    //   -> [0.0, 0.5, 0.5, 0.5, 1.0]
    let sk_plateau: [f64; 5] = [0.0, 0.5, 0.5, 0.5, 1.0];

    let x = array![[1.0], [2.0], [2.0], [2.0], [3.0]];
    let qt = QuantileTransformer::<f64>::new(5, OutputDistribution::Uniform, 0);
    let out = qt.fit_transform(&x).unwrap();

    for (i, &expected) in sk_plateau.iter().enumerate() {
        let got = out[[i, 0]];
        assert!(
            (got - expected).abs() <= 1e-9,
            "row {i}: ferrolearn forward-only interp = {got}, sklearn averaged = {expected}"
        );
    }
}

// ===========================================================================
// GREEN GUARDS — SHIPPED behavior (must PASS). Live-oracle grounded.
// ===========================================================================

/// REQ-1: uniform forward-CDF on STRICTLY DISTINCT data. On distinct data
/// forward == averaged interp (DIV-A dormant) and Uniform == identity
/// (DIV-B dormant), so ferrolearn matches sklearn exactly.
/// sklearn `_data.py:2795` references linspace + `:2702` nanpercentile.
#[test]
fn green_uniform_distinct_single_feature() {
    // oracle: python3 -c "...QuantileTransformer(n_quantiles=5, output_distribution='uniform', \
    //   subsample=10**9).fit_transform([[1.],[2.],[3.],[4.],[5.]]).ravel().tolist()"
    //   -> [0.0, 0.25, 0.5, 0.75, 1.0]
    let sk: [f64; 5] = [0.0, 0.25, 0.5, 0.75, 1.0];

    let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
    let qt = QuantileTransformer::<f64>::new(5, OutputDistribution::Uniform, 0);
    let out = qt.fit_transform(&x).unwrap();

    for (i, &expected) in sk.iter().enumerate() {
        let got = out[[i, 0]];
        assert!(
            (got - expected).abs() <= 1e-9,
            "row {i}: ferrolearn = {got}, sklearn = {expected}"
        );
    }
}

/// REQ-1: richer distinct multi-feature uniform value-match.
/// sklearn `_data.py:2702-2795` (per-column independent CDF).
#[test]
fn green_uniform_distinct_multi_feature() {
    // oracle: python3 -c "import numpy as np; from sklearn.preprocessing import QuantileTransformer; \
    //   X=np.array([[1.,10.],[3.,40.],[7.,20.],[2.,80.],[9.,30.],[5.,60.]]); \
    //   print(QuantileTransformer(n_quantiles=6, output_distribution='uniform', subsample=10**9)\
    //   .fit_transform(X).tolist())"
    //   -> [[0.0,0.0],[0.4,0.6],[0.8,0.2],[0.2,1.0],[1.0,0.4],[0.6,0.8]]
    let sk: [[f64; 2]; 6] = [
        [0.0, 0.0],
        [0.4, 0.6],
        [0.8, 0.2],
        [0.2, 1.0],
        [1.0, 0.4],
        [0.6, 0.8],
    ];

    let x = array![
        [1.0, 10.0],
        [3.0, 40.0],
        [7.0, 20.0],
        [2.0, 80.0],
        [9.0, 30.0],
        [5.0, 60.0]
    ];
    let qt = QuantileTransformer::<f64>::new(6, OutputDistribution::Uniform, 0);
    let out = qt.fit_transform(&x).unwrap();

    for (i, row) in sk.iter().enumerate() {
        for (j, &expected) in row.iter().enumerate() {
            let got = out[[i, j]];
            assert!(
                (got - expected).abs() <= 1e-9,
                "[{i},{j}]: ferrolearn = {got}, sklearn = {expected}"
            );
        }
    }
}

/// REQ-5 (scoped error contracts): ferrolearn's own guards must hold.
/// These are ferrolearn-internal contracts (not sklearn parity — see design
/// doc REQ-5 flagged DIVs), so the assertion is on `is_err()` shape only.
#[test]
fn green_error_contracts() {
    // n_samples < 2 -> Err
    let qt = QuantileTransformer::<f64>::new(5, OutputDistribution::Uniform, 0);
    assert!(
        qt.fit(&array![[1.0]], &()).is_err(),
        "single-sample fit should Err"
    );

    // n_quantiles < 2 -> Err
    let qt1 = QuantileTransformer::<f64>::new(1, OutputDistribution::Uniform, 0);
    assert!(
        qt1.fit(&array![[1.0], [2.0], [3.0]], &()).is_err(),
        "n_quantiles=1 should Err"
    );

    // transform column-count mismatch -> Err
    let qt = QuantileTransformer::<f64>::new(5, OutputDistribution::Uniform, 0);
    let fitted = qt.fit(&array![[1.0, 2.0], [3.0, 4.0]], &()).unwrap();
    assert!(
        fitted.transform(&array![[1.0, 2.0, 3.0]]).is_err(),
        "ncols mismatch should Err"
    );

    // unfitted transform -> Err
    let qt = QuantileTransformer::<f64>::new(5, OutputDistribution::Uniform, 0);
    assert!(
        qt.transform(&array![[1.0], [2.0]]).is_err(),
        "unfitted transform should Err"
    );
}

// ===========================================================================
// RE-AUDIT (2026-06): fixes #1320 (Acklam probit + clip) and #1321 (averaged
// np_interp) confirmed by `divergence_normal_ppf_accuracy` and
// `divergence_averaged_interpolation_plateau` now passing. The fixtures below
// are FRESH live-oracle fixtures (not the prior pinned inputs) verifying the
// implemented Uniform + Normal forward-transform surface end-to-end.
// ===========================================================================

/// DIV-C (NEW) — fit landmark FP divergence: sklearn computes quantile
/// landmarks via `references = self.references_ * 100`
/// (`sklearn/preprocessing/_data.py:2694`) then
/// `self.quantiles_ = np.nanpercentile(X, references, axis=0)` (`:2702`).
/// numpy's `nanpercentile` recomputes the interpolation position as
/// `pos = q/100 * (n-1)`; the `*100` (sklearn) then `/100` (numpy) round-trip
/// means e.g. the 2nd of 7 references gives `pos = 0.9999999999999998`, so
/// numpy interpolates `col[0] + 0.999...*(col[1]-col[0]) = 1.9999999999999998`
/// rather than landing exactly on the landmark.
///
/// ferrolearn (`quantile_transformer.rs:372`) keeps the reference as a fraction
/// `i/(K-1)` and computes `pos = ref_level * (n-1)` directly, yielding the EXACT
/// integer index `1.0` → landmark value `2.0`. On TIED/PLATEAU data this 1-ULP
/// landmark shift moves a query value from the interior of a plateau to its
/// edge, producing a LARGE (~0.083) divergence in the Uniform transform output.
///
/// Train `[[1],[2],[2],[2],[5],[5],[9]]` (n_quantiles=7, uniform).
/// sklearn quantiles_ = [1, 1.9999999999999998, 2, 2, 4.999999999999997, 5, 9].
/// ferrolearn quantiles  = [1, 2, 2, 2, 5, 5, 9].
/// transform([0,1,2,3.5,5,7,9,12]):
///   sklearn -> [0, 0, 0.41666666666666663, 0.5833333333333334,
///               0.8333333333333333, 0.9166666666666666, 1, 1]
///   ferrolearn -> [0, 0, 0.3333333333333333, 0.5833333333333333,
///               0.75, 0.9166666666666667, 1, 1]   (rows 2 and 4 diverge ~0.083)
/// Tracking: #1322
#[test]
fn divergence_landmark_nanpercentile_roundtrip_plateau() {
    // oracle: python3 -c "import numpy as np; from sklearn.preprocessing import QuantileTransformer; \
    //   qt=QuantileTransformer(n_quantiles=7, output_distribution='uniform', subsample=10**9)\
    //   .fit(np.array([[1.],[2.],[2.],[2.],[5.],[5.],[9.]])); \
    //   print(qt.transform(np.array([[0.],[1.],[2.],[3.5],[5.],[7.],[9.],[12.]])).ravel().tolist())"
    //   -> [0.0, 0.0, 0.41666666666666663, 0.5833333333333334,
    //       0.8333333333333333, 0.9166666666666666, 1.0, 1.0]
    let sk: [f64; 8] = [
        0.0,
        0.0,
        0.416_666_666_666_666_63,
        0.583_333_333_333_333_4,
        0.833_333_333_333_333_3,
        0.916_666_666_666_666_6,
        1.0,
        1.0,
    ];

    let xtrain = array![[1.0], [2.0], [2.0], [2.0], [5.0], [5.0], [9.0]];
    let qt = QuantileTransformer::<f64>::new(7, OutputDistribution::Uniform, 0);
    let fitted = qt.fit(&xtrain, &()).unwrap();
    let queries = array![[0.0], [1.0], [2.0], [3.5], [5.0], [7.0], [9.0], [12.0]];
    let out = fitted.transform(&queries).unwrap();

    for (i, &expected) in sk.iter().enumerate() {
        let got = out[[i, 0]];
        assert!(
            (got - expected).abs() <= 1e-9,
            "row {i}: ferrolearn = {got}, sklearn = {expected} (diff {})",
            (got - expected).abs()
        );
    }
}

// ===========================================================================
// FRESH GREEN GUARDS — implemented surface that NOW matches sklearn 1.5.2
// after #1320 + #1321. Every expected value is a LIVE sklearn 1.5.2 call
// (R-CHAR-3). These complement the headline DIV-A/DIV-B (now green).
// ===========================================================================

/// REQ-2 (post-fix #1321): Uniform on data with MULTIPLE DISTINCT PLATEAUS,
/// n_quantiles == n_samples == 10. Stresses the averaged interpolation on
/// several plateaus (the [2,2,2] run AND the [5,5] run) at once.
/// sklearn `_data.py:2843-2846` (averaged interp).
#[test]
fn green_uniform_multi_plateau() {
    // oracle: QuantileTransformer(n_quantiles=10, output_distribution='uniform', subsample=10**9)\
    //   .fit_transform([[1.],[2.],[2.],[2.],[3.],[5.],[5.],[7.],[8.],[9.]]).ravel().tolist()
    let sk: [f64; 10] = [
        0.0,
        0.222_222_222_222_222_2,
        0.222_222_222_222_222_2,
        0.222_222_222_222_222_2,
        0.444_444_444_444_444_4,
        0.611_111_111_111_111_2,
        0.611_111_111_111_111_2,
        0.777_777_777_777_777_9,
        0.888_888_888_888_888_8,
        1.0,
    ];
    let x = array![
        [1.0],
        [2.0],
        [2.0],
        [2.0],
        [3.0],
        [5.0],
        [5.0],
        [7.0],
        [8.0],
        [9.0]
    ];
    let qt = QuantileTransformer::<f64>::new(10, OutputDistribution::Uniform, 0);
    let out = qt.fit_transform(&x).unwrap();
    for (i, &expected) in sk.iter().enumerate() {
        let got = out[[i, 0]];
        assert!(
            (got - expected).abs() <= 1e-9,
            "row {i}: ferrolearn = {got}, sklearn = {expected}"
        );
    }
}

/// REQ-1 (post-fix): n_quantiles < n_samples (K=5, n=20). Landmarks are
/// INTERPOLATED percentiles (not raw data); exercises `pos = ref*(n-1)` against
/// `np.nanpercentile`. Data are integers 1..20 so the landmarks are exact even
/// through the *100 round-trip; full transform matches.
/// sklearn `_data.py:2702` (nanpercentile) + `:2843-2846`.
#[test]
fn green_uniform_n_quantiles_lt_n_samples() {
    // oracle: X=[[3],[7],[1],[9],[2],[8],[5],[6],[4],[10],[12],[15],[11],[13],[14],[20],[18],[16],[17],[19]] (float); \
    //   QuantileTransformer(n_quantiles=5, output_distribution='uniform', subsample=10**9).fit_transform(X)
    let sk: [f64; 20] = [
        0.105_263_157_894_736_84,
        0.315_789_473_684_210_5,
        0.0,
        0.421_052_631_578_947_35,
        0.052_631_578_947_368_42,
        0.368_421_052_631_579,
        0.210_526_315_789_473_67,
        0.263_157_894_736_842_15,
        0.157_894_736_842_105_25,
        0.473_684_210_526_315_8,
        0.578_947_368_421_052_7,
        0.736_842_105_263_157_9,
        0.526_315_789_473_684_3,
        0.631_578_947_368_421,
        0.684_210_526_315_789_5,
        1.0,
        0.894_736_842_105_263_2,
        0.789_473_684_210_526_3,
        0.842_105_263_157_894_7,
        0.947_368_421_052_631_6,
    ];
    let x = array![
        [3.0],
        [7.0],
        [1.0],
        [9.0],
        [2.0],
        [8.0],
        [5.0],
        [6.0],
        [4.0],
        [10.0],
        [12.0],
        [15.0],
        [11.0],
        [13.0],
        [14.0],
        [20.0],
        [18.0],
        [16.0],
        [17.0],
        [19.0]
    ];
    let qt = QuantileTransformer::<f64>::new(5, OutputDistribution::Uniform, 0);
    let out = qt.fit_transform(&x).unwrap();
    for (i, &expected) in sk.iter().enumerate() {
        let got = out[[i, 0]];
        assert!(
            (got - expected).abs() <= 1e-9,
            "row {i}: ferrolearn = {got}, sklearn = {expected}"
        );
    }
}

/// REQ-3 (post-fix #1320): Normal output, 11 distinct rows, n_quantiles=11 so
/// references hit non-round probabilities 0.0,0.1,...,1.0. Verifies Acklam
/// `norm.ppf` accuracy across the range (0.1→-1.28155, 0.3→-0.52440,
/// 0.7→0.52440, 0.9→1.28155) AND the extreme clip ±5.199337582605575.
/// sklearn `_data.py:2856-2862`.
#[test]
fn green_normal_nonround_probabilities() {
    // oracle: QuantileTransformer(n_quantiles=11, output_distribution='normal', subsample=10**9)\
    //   .fit_transform([[0.],[1.],...,[10.]]).ravel().tolist()
    let sk: [f64; 11] = [
        -5.199_337_582_605_575,
        -1.281_551_565_544_600_4,
        -0.841_621_233_572_914_2,
        -0.524_400_512_708_040_9,
        -0.253_347_103_135_799_7,
        0.0,
        0.253_347_103_135_799_7,
        0.524_400_512_708_041,
        0.841_621_233_572_914_3,
        1.281_551_565_544_600_4,
        5.199_337_582_703_42,
    ];
    let x = array![
        [0.0],
        [1.0],
        [2.0],
        [3.0],
        [4.0],
        [5.0],
        [6.0],
        [7.0],
        [8.0],
        [9.0],
        [10.0]
    ];
    let qt = QuantileTransformer::<f64>::new(11, OutputDistribution::Normal, 0);
    let out = qt.fit_transform(&x).unwrap();
    for (i, &expected) in sk.iter().enumerate() {
        let got = out[[i, 0]];
        assert!(
            (got - expected).abs() <= 1e-6,
            "row {i}: ferrolearn = {got}, sklearn = {expected} (diff {})",
            (got - expected).abs()
        );
    }
}

/// REQ-3 (post-fix #1320): Normal output on TIED/PLATEAU data. The plateau maps
/// to a single non-round probability whose `norm.ppf` must match.
/// sklearn `_data.py:2843-2846` + `:2856`.
#[test]
fn green_normal_plateau() {
    // oracle: QuantileTransformer(n_quantiles=10, output_distribution='normal', subsample=10**9)\
    //   .fit_transform([[1.],[2.],[2.],[2.],[3.],[5.],[5.],[7.],[8.],[9.]]).ravel().tolist()
    let sk: [f64; 10] = [
        -5.199_337_582_605_575,
        -0.764_709_673_786_387_1,
        -0.764_709_673_786_387_1,
        -0.764_709_673_786_387_1,
        -0.139_710_298_881_862_12,
        0.282_216_147_062_508_25,
        0.282_216_147_062_508_25,
        0.764_709_673_786_387_5,
        1.220_640_348_847_349_6,
        5.199_337_582_703_42,
    ];
    let x = array![
        [1.0],
        [2.0],
        [2.0],
        [2.0],
        [3.0],
        [5.0],
        [5.0],
        [7.0],
        [8.0],
        [9.0]
    ];
    let qt = QuantileTransformer::<f64>::new(10, OutputDistribution::Normal, 0);
    let out = qt.fit_transform(&x).unwrap();
    for (i, &expected) in sk.iter().enumerate() {
        let got = out[[i, 0]];
        assert!(
            (got - expected).abs() <= 1e-6,
            "row {i}: ferrolearn = {got}, sklearn = {expected} (diff {})",
            (got - expected).abs()
        );
    }
}

/// np_interp faithfulness vs numpy: query points below range, at a landmark,
/// exactly on a plateau value, between landmarks, above range — driven through
/// a fitted transformer whose quantiles_ contain DUPLICATES (the [2,2] and
/// [5,5] plateaus). Expected = live sklearn `_transform_col` averaged output.
/// sklearn `_data.py:2843-2846` (averaged `np.interp`).
#[test]
fn green_np_interp_faithfulness_duplicates() {
    // oracle: Xtrain=[[1.],[2.],[2.],[2.],[5.],[5.],[9.]]; \
    //   qt=QuantileTransformer(n_quantiles=7, output_distribution='uniform', subsample=10**9).fit(Xtrain); \
    //   qt.transform([[0.],[1.],[2.],[3.5],[5.],[7.],[9.],[12.]]).ravel().tolist()
    //   -> [0.0, 0.0, 0.41666666666666663, 0.5833333333333334,
    //       0.8333333333333333, 0.9166666666666666, 1.0, 1.0]
    //
    // NOTE: this is the SAME end-to-end pipeline as the ignored DIV-C pin
    // (#1322). Because ferrolearn's fit landmarks diverge from
    // np.nanpercentile, the full-pipeline values diverge at rows 2 and 4.
    // The np_interp helper itself is faithful, but it is fed different
    // landmarks. We therefore assert np_interp faithfulness on the
    // landmark grid ferrolearn ACTUALLY produces, by reading it from the
    // fitted model and re-deriving the expected averaged-interp from numpy
    // on THAT grid (R-CHAR-3: numpy is the oracle for np.interp).
    use ferrolearn_preprocess::quantile_transformer::FittedQuantileTransformer;

    let xtrain = array![[1.0], [2.0], [2.0], [2.0], [5.0], [5.0], [9.0]];
    let qt = QuantileTransformer::<f64>::new(7, OutputDistribution::Uniform, 0);
    let fitted: FittedQuantileTransformer<f64> = qt.fit(&xtrain, &()).unwrap();
    let landmarks = fitted.quantiles()[0].clone();

    // Live numpy oracle on the EXACT landmark grid ferrolearn produced.
    let refs: Vec<f64> = (0..landmarks.len())
        .map(|i| i as f64 / (landmarks.len() - 1) as f64)
        .collect();
    let queries = [0.0_f64, 1.0, 2.0, 3.5, 5.0, 7.0, 9.0, 12.0];

    let py = format!(
        "import numpy as np\nq=np.array({:?})\nr=np.array({:?})\nx=np.array({:?})\nv=0.5*(np.interp(x,q,r)-np.interp(-x,-q[::-1],-r[::-1]))\n# replicate sklearn uniform exact-bound override\nv[x==q[0]]=0.0\nv[x==q[-1]]=1.0\nprint(' '.join(repr(float(t)) for t in v))",
        landmarks, refs, queries
    );
    let output = std::process::Command::new("python3")
        .arg("-c")
        .arg(&py)
        .output()
        .expect("python3 oracle call failed");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let expected: Vec<f64> = stdout
        .split_whitespace()
        .map(|s| s.parse().unwrap())
        .collect();
    assert_eq!(
        expected.len(),
        queries.len(),
        "oracle parse failed: {stdout}"
    );

    let qarr = array![[0.0], [1.0], [2.0], [3.5], [5.0], [7.0], [9.0], [12.0]];
    let out = fitted.transform(&qarr).unwrap();
    for (i, &exp) in expected.iter().enumerate() {
        let got = out[[i, 0]];
        assert!(
            (got - exp).abs() <= 1e-12,
            "np_interp row {i}: ferrolearn = {got}, numpy averaged interp = {exp}"
        );
    }
}

/// Fully-distinct no-tie matrix: Uniform transforms identically and Normal
/// median maps to exactly 0. sklearn `_data.py:2843-2846`,`:2856`.
#[test]
fn green_distinct_uniform_and_normal_median_zero() {
    // oracle uniform: QuantileTransformer(5,'uniform',10**9).fit_transform([[3.],[7.],[1.],[9.],[5.]])
    //   -> [0.25, 0.75, 0.0, 1.0, 0.5]
    let sk_uniform: [f64; 5] = [0.25, 0.75, 0.0, 1.0, 0.5];
    let x = array![[3.0], [7.0], [1.0], [9.0], [5.0]];
    let qt = QuantileTransformer::<f64>::new(5, OutputDistribution::Uniform, 0);
    let out = qt.fit_transform(&x).unwrap();
    for (i, &expected) in sk_uniform.iter().enumerate() {
        assert!(
            (out[[i, 0]] - expected).abs() <= 1e-9,
            "uniform row {i}: ferrolearn = {}, sklearn = {expected}",
            out[[i, 0]]
        );
    }
    // Normal median (the value at CDF=0.5) -> exactly 0.
    let qtn = QuantileTransformer::<f64>::new(5, OutputDistribution::Normal, 0);
    let outn = qtn.fit_transform(&x).unwrap();
    // sorted [1,3,5,7,9]; median value is 5.0 at row index 4.
    assert!(
        outn[[4, 0]].abs() <= 1e-12,
        "Normal median should map to 0, got {}",
        outn[[4, 0]]
    );
}

/// f32 fixture: distinct data, Uniform, tolerance ~1e-4.
/// sklearn `_data.py:2843-2846` (sklearn computes in f64 then casts).
#[test]
fn green_f32_uniform_distinct() {
    // oracle: QuantileTransformer(6,'uniform',10**9).fit_transform(
    //   np.array([[1],[2],[3],[4],[5],[6]],dtype=np.float32))
    //   -> [0.0, 0.20000000298023224, 0.4000000059604645, 0.6000000238418579,
    //       0.800000011920929, 1.0]
    let sk: [f32; 6] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0];
    let x: ndarray::Array2<f32> = array![[1.0f32], [2.0], [3.0], [4.0], [5.0], [6.0]];
    let qt = QuantileTransformer::<f32>::new(6, OutputDistribution::Uniform, 0);
    let out = qt.fit_transform(&x).unwrap();
    for (i, &expected) in sk.iter().enumerate() {
        assert!(
            (out[[i, 0]] - expected).abs() <= 1e-4,
            "f32 row {i}: ferrolearn = {}, sklearn = {expected}",
            out[[i, 0]]
        );
    }
}

/// Multi-feature Uniform with internal plateaus, n_quantiles == n_samples == 8.
/// Two independent columns each with their own plateaus.
/// sklearn `_data.py:2702-2799` (per-column) + `:2843-2846`.
#[test]
fn green_uniform_multifeature_plateau() {
    // oracle: X=[[1.,5.],[1.,5.],[3.,5.],[4.,8.],[6.,8.],[6.,9.],[7.,9.],[9.,12.]]; \
    //   QuantileTransformer(8,'uniform',10**9).fit_transform(X).tolist()
    let sk: [[f64; 2]; 8] = [
        [0.0, 0.0],
        [0.0, 0.0],
        [0.285_714_285_714_285_7, 0.0],
        [0.428_571_428_571_428_55, 0.5],
        [0.642_857_142_857_142_8, 0.5],
        [0.642_857_142_857_142_8, 0.785_714_285_714_285_6],
        [0.857_142_857_142_857_1, 0.785_714_285_714_285_6],
        [1.0, 1.0],
    ];
    let x = array![
        [1.0, 5.0],
        [1.0, 5.0],
        [3.0, 5.0],
        [4.0, 8.0],
        [6.0, 8.0],
        [6.0, 9.0],
        [7.0, 9.0],
        [9.0, 12.0]
    ];
    let qt = QuantileTransformer::<f64>::new(8, OutputDistribution::Uniform, 0);
    let out = qt.fit_transform(&x).unwrap();
    for (i, row) in sk.iter().enumerate() {
        for (j, &expected) in row.iter().enumerate() {
            let got = out[[i, j]];
            assert!(
                (got - expected).abs() <= 1e-9,
                "[{i},{j}]: ferrolearn = {got}, sklearn = {expected}"
            );
        }
    }
}

// ===========================================================================
// REQ-7 — inverse_transform (reverse interp + norm.cdf + bounds + NaN).
// sklearn `_data.py:2947` -> `_transform_col(inverse=True)` (`:2813-2851`):
// normal -> `stats.norm.cdf` (`:2821`) then plain `np.interp(rank, references_,
// quantiles)` (`:2848`); bounds override to quantiles[0]/quantiles[-1]
// (`:2850-2851`); NaN passes through (`:2833`).
// Every expected value below is a LIVE sklearn 1.5.2 / scipy call (R-CHAR-3).
// ===========================================================================

/// REQ-7: UNIFORM round-trip recovers the original data. `inverse_transform(
/// transform(X))` ≈ X. The forward transform is monotone; on the EXACT-landmark
/// integer fixture the round-trip is bit-exact. Oracle = live sklearn
/// `inverse_transform(transform(X))`.
#[test]
fn green_inverse_uniform_roundtrip() {
    // oracle: python3 -c "import numpy as np; from sklearn.preprocessing import QuantileTransformer; \
    //   X=np.arange(1.,11.).reshape(-1,1); \
    //   qt=QuantileTransformer(n_quantiles=10, output_distribution='uniform', subsample=10**9).fit(X); \
    //   print(qt.inverse_transform(qt.transform(X)).ravel().tolist())"
    //   -> [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    let sk: [f64; 10] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let x = array![
        [1.0],
        [2.0],
        [3.0],
        [4.0],
        [5.0],
        [6.0],
        [7.0],
        [8.0],
        [9.0],
        [10.0]
    ];
    let qt = QuantileTransformer::<f64>::new(10, OutputDistribution::Uniform, 0);
    let fitted = qt.fit(&x, &()).unwrap();
    let fwd = fitted.transform(&x).unwrap();
    let inv = fitted.inverse_transform(&fwd).unwrap();
    for (i, &expected) in sk.iter().enumerate() {
        let got = inv[[i, 0]];
        assert!(
            (got - expected).abs() <= 1e-9,
            "row {i}: ferrolearn inverse roundtrip = {got}, sklearn/original = {expected}"
        );
    }
}

/// REQ-7: UNIFORM held-out arbitrary ranks Y in [0,1] match sklearn
/// `inverse_transform(Y)`. The fixture is strictly-distinct integer data so
/// ferrolearn's landmarks and sklearn's `np.nanpercentile` landmarks agree to
/// f64 ULP (#1322 dormant), making the comparison valid to 1e-9.
/// sklearn `_data.py:2848` (plain reverse `np.interp`).
#[test]
fn green_inverse_uniform_heldout() {
    // oracle: python3 -c "import numpy as np; from sklearn.preprocessing import QuantileTransformer; \
    //   X=np.arange(1.,11.).reshape(-1,1); \
    //   qt=QuantileTransformer(n_quantiles=10, output_distribution='uniform', subsample=10**9).fit(X); \
    //   Y=np.array([[0.0],[0.05],[0.15],[0.33],[0.5],[0.66],[0.8],[0.95],[1.0]]); \
    //   print(qt.inverse_transform(Y).ravel().tolist())"
    //   -> [1.0, 1.45, 2.35, 3.9699999999999998, 5.5, 6.939999999999999,
    //       8.200000000000001, 9.55, 10.0]
    let sk: [f64; 9] = [
        1.0,
        1.45,
        2.35,
        3.969_999_999_999_999_8,
        5.5,
        6.939_999_999_999_999,
        8.200_000_000_000_001,
        9.55,
        10.0,
    ];
    let y = array![
        [0.0],
        [0.05],
        [0.15],
        [0.33],
        [0.5],
        [0.66],
        [0.8],
        [0.95],
        [1.0]
    ];
    let x = array![
        [1.0],
        [2.0],
        [3.0],
        [4.0],
        [5.0],
        [6.0],
        [7.0],
        [8.0],
        [9.0],
        [10.0]
    ];
    let qt = QuantileTransformer::<f64>::new(10, OutputDistribution::Uniform, 0);
    let fitted = qt.fit(&x, &()).unwrap();
    let inv = fitted.inverse_transform(&y).unwrap();
    for (i, &expected) in sk.iter().enumerate() {
        let got = inv[[i, 0]];
        assert!(
            (got - expected).abs() <= 1e-9,
            "row {i}: ferrolearn inverse = {got}, sklearn inverse = {expected} (diff {})",
            (got - expected).abs()
        );
    }
}

/// REQ-7: NORMAL round-trip recovers the original data. `inverse_transform(
/// transform(X))` ≈ X within ~1e-7 (the forward Acklam probit + inverse ndtr
/// each ~1e-9, compounded through the interp). Oracle = live sklearn.
/// sklearn `_data.py:2821`,`:2848`.
#[test]
fn green_inverse_normal_roundtrip() {
    // oracle: python3 -c "import numpy as np; from sklearn.preprocessing import QuantileTransformer; \
    //   X=np.arange(1.,11.).reshape(-1,1); \
    //   qt=QuantileTransformer(n_quantiles=10, output_distribution='normal', subsample=10**9).fit(X); \
    //   print(qt.inverse_transform(qt.transform(X)).ravel().tolist())"
    //   -> [1.0, 2.0, 3.0, 3.9999999999999996, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    let sk: [f64; 10] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let x = array![
        [1.0],
        [2.0],
        [3.0],
        [4.0],
        [5.0],
        [6.0],
        [7.0],
        [8.0],
        [9.0],
        [10.0]
    ];
    let qt = QuantileTransformer::<f64>::new(10, OutputDistribution::Normal, 0);
    let fitted = qt.fit(&x, &()).unwrap();
    let fwd = fitted.transform(&x).unwrap();
    let inv = fitted.inverse_transform(&fwd).unwrap();
    for (i, &expected) in sk.iter().enumerate() {
        let got = inv[[i, 0]];
        assert!(
            (got - expected).abs() <= 1e-7,
            "row {i}: ferrolearn inverse normal roundtrip = {got}, original = {expected} (diff {})",
            (got - expected).abs()
        );
    }
}

/// REQ-7: NORMAL held-out values in normal space match sklearn
/// `inverse_transform(Y)` to ~1e-7 (the inverse ndtr `norm.cdf` ~1e-9 + interp).
/// sklearn `_data.py:2821` (`stats.norm.cdf`) + `:2848` (reverse interp).
#[test]
fn green_inverse_normal_heldout() {
    // oracle: python3 -c "import numpy as np; from sklearn.preprocessing import QuantileTransformer; \
    //   X=np.arange(1.,11.).reshape(-1,1); \
    //   qt=QuantileTransformer(n_quantiles=10, output_distribution='normal', subsample=10**9).fit(X); \
    //   Y=np.array([[-2.0],[-1.0],[-0.5],[0.0],[0.5],[1.0],[2.0]]); \
    //   print(qt.inverse_transform(Y).ravel().tolist())"
    //   -> [1.2047511875336128, 2.427897285383114, 3.7768378485338814, 5.5,
    //       7.223162151466117, 8.572102714616886, 9.795248812466387]
    let sk: [f64; 7] = [
        1.204_751_187_533_612_8,
        2.427_897_285_383_114,
        3.776_837_848_533_881_4,
        5.5,
        7.223_162_151_466_117,
        8.572_102_714_616_886,
        9.795_248_812_466_387,
    ];
    let y = array![[-2.0], [-1.0], [-0.5], [0.0], [0.5], [1.0], [2.0]];
    let x = array![
        [1.0],
        [2.0],
        [3.0],
        [4.0],
        [5.0],
        [6.0],
        [7.0],
        [8.0],
        [9.0],
        [10.0]
    ];
    let qt = QuantileTransformer::<f64>::new(10, OutputDistribution::Normal, 0);
    let fitted = qt.fit(&x, &()).unwrap();
    let inv = fitted.inverse_transform(&y).unwrap();
    for (i, &expected) in sk.iter().enumerate() {
        let got = inv[[i, 0]];
        assert!(
            (got - expected).abs() <= 1e-7,
            "row {i}: ferrolearn inverse normal = {got}, sklearn inverse = {expected} (diff {})",
            (got - expected).abs()
        );
    }
}

/// REQ-7: bounds clipping. UNIFORM rank exactly 0 -> column min (quantiles[0]),
/// rank exactly 1 -> column max (quantiles[-1]). NORMAL very-negative ->
/// column min, very-positive -> column max. sklearn `_data.py:2850-2851` +
/// the bound masks `:2827-2831`.
#[test]
fn green_inverse_bounds_clipping() {
    // oracle: python3 -c "import numpy as np; from sklearn.preprocessing import QuantileTransformer; \
    //   X=np.arange(1.,11.).reshape(-1,1); \
    //   qu=QuantileTransformer(n_quantiles=10, output_distribution='uniform', subsample=10**9).fit(X); \
    //   qn=QuantileTransformer(n_quantiles=10, output_distribution='normal', subsample=10**9).fit(X); \
    //   print(float(qu.inverse_transform([[0.0]])[0,0]), float(qu.inverse_transform([[1.0]])[0,0]), \
    //         float(qn.inverse_transform([[-10.0]])[0,0]), float(qn.inverse_transform([[10.0]])[0,0]))"
    //   -> 1.0 10.0 1.0 10.0   (column min=1.0, column max=10.0)
    let x = array![
        [1.0],
        [2.0],
        [3.0],
        [4.0],
        [5.0],
        [6.0],
        [7.0],
        [8.0],
        [9.0],
        [10.0]
    ];
    let qu = QuantileTransformer::<f64>::new(10, OutputDistribution::Uniform, 0)
        .fit(&x, &())
        .unwrap();
    let lo = qu.inverse_transform(&array![[0.0]]).unwrap();
    let hi = qu.inverse_transform(&array![[1.0]]).unwrap();
    assert!(
        (lo[[0, 0]] - 1.0).abs() <= 1e-12,
        "uniform rank 0 -> column min 1.0, got {}",
        lo[[0, 0]]
    );
    assert!(
        (hi[[0, 0]] - 10.0).abs() <= 1e-12,
        "uniform rank 1 -> column max 10.0, got {}",
        hi[[0, 0]]
    );

    let qn = QuantileTransformer::<f64>::new(10, OutputDistribution::Normal, 0)
        .fit(&x, &())
        .unwrap();
    let nlo = qn.inverse_transform(&array![[-10.0]]).unwrap();
    let nhi = qn.inverse_transform(&array![[10.0]]).unwrap();
    assert!(
        (nlo[[0, 0]] - 1.0).abs() <= 1e-12,
        "normal -10 -> column min 1.0, got {}",
        nlo[[0, 0]]
    );
    assert!(
        (nhi[[0, 0]] - 10.0).abs() <= 1e-12,
        "normal +10 -> column max 10.0, got {}",
        nhi[[0, 0]]
    );
}

/// REQ-7: NaN passes through inverse_transform unchanged (sklearn
/// `isfinite_mask`, `_data.py:2833`; `_more_tags` `allow_nan=True`, `:2970`).
#[test]
fn green_inverse_nan_passthrough() {
    // oracle: python3 -c "import numpy as np; from sklearn.preprocessing import QuantileTransformer; \
    //   X=np.arange(1.,11.).reshape(-1,1); \
    //   qt=QuantileTransformer(n_quantiles=10, output_distribution='uniform', subsample=10**9).fit(X); \
    //   print(qt.inverse_transform(np.array([[0.3],[np.nan],[0.7]])).ravel().tolist())"
    //   -> [3.6999999999999993, nan, 7.299999999999999]
    let x = array![
        [1.0],
        [2.0],
        [3.0],
        [4.0],
        [5.0],
        [6.0],
        [7.0],
        [8.0],
        [9.0],
        [10.0]
    ];
    let qt = QuantileTransformer::<f64>::new(10, OutputDistribution::Uniform, 0)
        .fit(&x, &())
        .unwrap();
    let y = array![[0.3], [f64::NAN], [0.7]];
    let inv = qt.inverse_transform(&y).unwrap();
    assert!(inv[[1, 0]].is_nan(), "NaN row should stay NaN");
    // finite rows match sklearn (ferrolearn exact landmarks vs sklearn ~ULP).
    assert!(
        (inv[[0, 0]] - 3.699_999_999_999_999_3).abs() <= 1e-9,
        "row 0 = {}, sklearn = 3.6999999999999993",
        inv[[0, 0]]
    );
    assert!(
        (inv[[2, 0]] - 7.299_999_999_999_999).abs() <= 1e-9,
        "row 2 = {}, sklearn = 7.299999999999999",
        inv[[2, 0]]
    );
}

/// REQ-7: shape mismatch (wrong n_features) -> Err.
#[test]
fn green_inverse_shape_mismatch() {
    let qt = QuantileTransformer::<f64>::new(5, OutputDistribution::Uniform, 0);
    let fitted = qt.fit(&array![[1.0, 2.0], [3.0, 4.0]], &()).unwrap();
    assert!(
        fitted.inverse_transform(&array![[0.5, 0.5, 0.5]]).is_err(),
        "inverse_transform ncols mismatch should Err"
    );
}

/// REQ-7: `norm_cdf` (scipy `ndtr` / `stats.norm.cdf`) sanity vs the live scipy
/// oracle, exposed through the public `inverse_transform` Normal path. A
/// transformer fitted on `[0,1]` (n_quantiles=2) has `quantiles_=[0,1]`,
/// `references_=[0,1]`, so the Normal inverse maps `norm_cdf(x)` straight
/// through `np.interp(rank,[0,1],[0,1]) = rank` (identity), exposing
/// `norm_cdf(x)` as the output. The chosen `x` give ranks strictly inside
/// `(1e-7, 1-1e-7)` so the bound masks do not fire.
#[test]
fn green_norm_cdf_sanity_via_inverse() {
    // oracle: python3 -c "import scipy.stats as st; \
    //   print(st.norm.cdf(0.0), st.norm.cdf(1.96), st.norm.cdf(-1.96), st.norm.cdf(3.0), st.norm.cdf(-3.0))"
    //   -> 0.5 0.9750021048517795 0.024997895148220435 0.9986501019683699 0.0013498980316300933
    let cases: [(f64, f64); 5] = [
        (0.0, 0.5),
        (1.96, 0.975_002_104_851_779_5),
        (-1.96, 0.024_997_895_148_220_435),
        (3.0, 0.998_650_101_968_369_9),
        (-3.0, 0.001_349_898_031_630_093_3),
    ];
    let qt = QuantileTransformer::<f64>::new(2, OutputDistribution::Normal, 0)
        .fit(&array![[0.0], [1.0]], &())
        .unwrap();
    for (x, expected_cdf) in cases {
        let out = qt.inverse_transform(&array![[x]]).unwrap();
        let got = out[[0, 0]];
        assert!(
            (got - expected_cdf).abs() <= 1e-9,
            "norm_cdf({x}) = {got}, scipy = {expected_cdf} (diff {})",
            (got - expected_cdf).abs()
        );
    }
}

/// REQ-7: NORMAL inverse on TIED/PLATEAU training data. The plateau landmarks
/// repeat; the reverse plain interp must hit the right landmark span. The
/// expected values come from the live sklearn `_transform_col` inverse computed
/// on the EXACT landmark grid ferrolearn produced (R-CHAR-3), neutralising the
/// #1322 landmark FP divergence the same way `green_np_interp_faithfulness`
/// does. sklearn `_data.py:2821`,`:2848`.
#[test]
fn green_inverse_normal_plateau_heldout() {
    use ferrolearn_preprocess::quantile_transformer::FittedQuantileTransformer;
    let x = array![[1.0], [2.0], [2.0], [2.0], [5.0], [7.0], [9.0], [12.0]];
    let qt = QuantileTransformer::<f64>::new(8, OutputDistribution::Normal, 0);
    let fitted: FittedQuantileTransformer<f64> = qt.fit(&x, &()).unwrap();
    let landmarks = fitted.quantiles()[0].clone();
    let refs: Vec<f64> = (0..landmarks.len())
        .map(|i| i as f64 / (landmarks.len() - 1) as f64)
        .collect();
    let ys = [-1.0_f64, 0.0, 1.0];
    let py = format!(
        "import numpy as np\nimport scipy.stats as st\nq=np.array({:?})\nr=np.array({:?})\nx=np.array({:?})\nrank=st.norm.cdf(x)\nv=np.interp(rank,r,q)\nv[rank<=0]=q[0]\nv[rank>=1]=q[-1]\nprint(' '.join(repr(float(t)) for t in v))",
        landmarks, refs, ys
    );
    let output = std::process::Command::new("python3")
        .arg("-c")
        .arg(&py)
        .output()
        .expect("python3 oracle call failed");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let expected: Vec<f64> = stdout
        .split_whitespace()
        .map(|s| s.parse().unwrap())
        .collect();
    assert_eq!(expected.len(), ys.len(), "oracle parse failed: {stdout}");

    let yarr = array![[-1.0], [0.0], [1.0]];
    let inv = fitted.inverse_transform(&yarr).unwrap();
    for (i, &exp) in expected.iter().enumerate() {
        let got = inv[[i, 0]];
        assert!(
            (got - exp).abs() <= 1e-7,
            "normal plateau inverse row {i}: ferrolearn = {got}, sklearn = {exp} (diff {})",
            (got - exp).abs()
        );
    }
}

// ===========================================================================
// DIV-D — inverse_transform must REJECT +/-inf (force_all_finite="allow-nan").
// ===========================================================================

/// Divergence: ferrolearn's `inverse_transform`
/// (`quantile_transformer.rs:192-262`) ACCEPTS `+/-inf` and silently maps it
/// through `norm_cdf`/`np_interp` (it only special-cases `NaN`,
/// `:221-224`), whereas sklearn `QuantileTransformer.inverse_transform`
/// (`sklearn/preprocessing/_data.py:2947`) runs
/// `_check_inputs(X, in_fit=False, ...)` (`:2965`) which validates with
/// `force_all_finite="allow-nan"` (`:2876`) — NaN is allowed but `+/-inf` is
/// REJECTED with `ValueError: Input X contains infinity ...`.
///
/// Input: a model fitted on `arange(1,11)`, then `inverse_transform([[inf]])`
/// (and `[[-inf]]`), Uniform output.
/// sklearn raises `ValueError` (Input X contains infinity) for BOTH inf signs
/// and BOTH output distributions (verified live, see oracle below).
/// ferrolearn returns `Ok` (no error): for Uniform, `np_interp(inf, refs, q)`
/// clamps to `q[-1]` for `+inf` and `q[0]` for `-inf` (a finite value, not an
/// error). So the contract divergence is: sklearn `Err`, ferrolearn `Ok`.
/// Tracking: #2212
#[test]
fn divergence_inverse_rejects_inf() {
    // oracle: python3 -c "import numpy as np; from sklearn.preprocessing import QuantileTransformer; \
    //   X=np.arange(1.,11.).reshape(-1,1); \
    //   qt=QuantileTransformer(n_quantiles=10, output_distribution='uniform', subsample=10**9).fit(X); \
    //   [print('ERROR') if (lambda f: ([f(np.array([[v]])) for v in (np.inf,-np.inf)]))(qt.inverse_transform) else None]"
    //   -> raises ValueError('Input X contains infinity or a value too large for dtype('float64').')
    //      for +inf AND -inf (uniform AND normal). NaN, by contrast, is allowed
    //      (allow-nan) and passes through.
    let x = array![
        [1.0],
        [2.0],
        [3.0],
        [4.0],
        [5.0],
        [6.0],
        [7.0],
        [8.0],
        [9.0],
        [10.0]
    ];
    let qt = QuantileTransformer::<f64>::new(10, OutputDistribution::Uniform, 0)
        .fit(&x, &())
        .unwrap();

    // sklearn rejects +inf with ValueError; ferrolearn must also Err.
    assert!(
        qt.inverse_transform(&array![[f64::INFINITY]]).is_err(),
        "inverse_transform(+inf) must Err (sklearn raises ValueError via \
         force_all_finite=allow-nan), but ferrolearn returned Ok"
    );
    // sklearn rejects -inf with ValueError; ferrolearn must also Err.
    assert!(
        qt.inverse_transform(&array![[f64::NEG_INFINITY]]).is_err(),
        "inverse_transform(-inf) must Err (sklearn raises ValueError via \
         force_all_finite=allow-nan), but ferrolearn returned Ok"
    );
}

// ===========================================================================
// REQ-9 — `quantile_transform` free function (sklearn
// `quantile_transform`, `_data.py:2978`,`:3107-3119`).
//
// The free fn delegates to `QuantileTransformer::new(...).fit(...).transform`
// (no quantile math duplicated). Oracle = live sklearn 1.5.2
// `quantile_transform` with `subsample=None` (the deterministic no-subsample
// path), matching n_quantiles so both sides build identical reference grids
// (R-CHAR-3). Every expected constant is produced by the `// oracle:`
// `python3 -c` call printed above it — never copied from ferrolearn.
// ===========================================================================

use ferrolearn_preprocess::quantile_transformer::quantile_transform;

/// `quantile_transform(X, axis=0, n_quantiles=5, Uniform)` on a 2-feature
/// distinct fixture matches sklearn column-by-column, AND equals the SHIPPED
/// estimator's `fit_transform` (the free fn must be a faithful wrapper).
#[test]
fn free_fn_axis0_uniform_matches_sklearn_and_estimator() {
    // oracle: python3 -c "import numpy as np; from sklearn.preprocessing import quantile_transform; \
    //   X=np.array([[1.,10.],[2.,20.],[3.,5.],[4.,40.],[5.,15.]]); \
    //   print(quantile_transform(X, axis=0, n_quantiles=5, output_distribution='uniform', subsample=None).tolist())"
    //   -> [[0.0,0.25],[0.25,0.75],[0.5,0.0],[0.75,1.0],[1.0,0.5]]
    let sk: [[f64; 2]; 5] = [
        [0.0, 0.25],
        [0.25, 0.75],
        [0.5, 0.0],
        [0.75, 1.0],
        [1.0, 0.5],
    ];
    let x = array![
        [1.0, 10.0],
        [2.0, 20.0],
        [3.0, 5.0],
        [4.0, 40.0],
        [5.0, 15.0]
    ];

    let out = quantile_transform(&x, 0, 5, OutputDistribution::Uniform, 0).unwrap();
    for (i, row) in sk.iter().enumerate() {
        for (j, &expected) in row.iter().enumerate() {
            assert!(
                (out[[i, j]] - expected).abs() <= 1e-9,
                "[{i},{j}] free fn = {}, sklearn = {expected}",
                out[[i, j]]
            );
        }
    }

    // The free fn MUST equal QuantileTransformer::new(...).fit(X).transform(X).
    let est = QuantileTransformer::<f64>::new(5, OutputDistribution::Uniform, 0)
        .fit(&x, &())
        .unwrap()
        .transform(&x)
        .unwrap();
    for (a, b) in out.iter().zip(est.iter()) {
        assert!(
            (a - b).abs() <= 1e-15,
            "free fn != estimator fit_transform: {a} vs {b}"
        );
    }
}

/// `quantile_transform(X, axis=0, Uniform)` on a column with TIED values matches
/// sklearn (the forward/reversed-averaged plateau mapping survives the wrapper).
#[test]
fn free_fn_axis0_uniform_tied_matches_sklearn() {
    // oracle: python3 -c "import numpy as np; from sklearn.preprocessing import quantile_transform; \
    //   print(quantile_transform(np.array([[1.],[2.],[2.],[2.],[5.]]), axis=0, n_quantiles=5, \
    //   output_distribution='uniform', subsample=None).ravel().tolist())"
    //   -> [0.0, 0.5, 0.5, 0.5, 1.0]
    let sk: [f64; 5] = [0.0, 0.5, 0.5, 0.5, 1.0];
    let x = array![[1.0], [2.0], [2.0], [2.0], [5.0]];

    let out = quantile_transform(&x, 0, 5, OutputDistribution::Uniform, 0).unwrap();
    for (i, &expected) in sk.iter().enumerate() {
        assert!(
            (out[[i, 0]] - expected).abs() <= 1e-9,
            "row {i}: free fn = {}, sklearn = {expected}",
            out[[i, 0]]
        );
    }
}

/// `quantile_transform(X, axis=0, Normal)` matches sklearn to ~1e-7 (inherits the
/// REQ-3 Acklam ppf accuracy through the estimator).
#[test]
fn free_fn_axis0_normal_matches_sklearn() {
    // oracle: python3 -c "import numpy as np; from sklearn.preprocessing import quantile_transform; \
    //   X=np.array([[1.,10.],[2.,20.],[3.,5.],[4.,40.],[5.,15.]]); \
    //   print(quantile_transform(X, axis=0, n_quantiles=5, output_distribution='normal', subsample=None).tolist())"
    //   -> [[-5.199337582605575,-0.6744897501960817],[-0.6744897501960817,0.6744897501960817],
    //       [0.0,-5.199337582605575],[0.6744897501960817,5.19933758270342],[5.19933758270342,0.0]]
    let sk: [[f64; 2]; 5] = [
        [-5.199337582605575, -0.6744897501960817],
        [-0.6744897501960817, 0.6744897501960817],
        [0.0, -5.199337582605575],
        [0.6744897501960817, 5.19933758270342],
        [5.19933758270342, 0.0],
    ];
    let x = array![
        [1.0, 10.0],
        [2.0, 20.0],
        [3.0, 5.0],
        [4.0, 40.0],
        [5.0, 15.0]
    ];

    let out = quantile_transform(&x, 0, 5, OutputDistribution::Normal, 0).unwrap();
    for (i, row) in sk.iter().enumerate() {
        for (j, &expected) in row.iter().enumerate() {
            assert!(
                (out[[i, j]] - expected).abs() <= 1e-7,
                "[{i},{j}] free fn Normal = {}, sklearn = {expected}",
                out[[i, j]]
            );
        }
    }
}

/// `quantile_transform(X, axis=1)` on a NON-SQUARE fixture (3x4) matches sklearn
/// `quantile_transform(X, axis=1)` (= `fit_transform(X.T).T`). A non-square
/// fixture makes a transpose bug observable. Also equals the manual
/// transpose-fit_transform-transpose path.
#[test]
fn free_fn_axis1_nonsquare_matches_sklearn_and_manual_transpose() {
    // oracle: python3 -c "import numpy as np; from sklearn.preprocessing import quantile_transform; \
    //   X=np.array([[10.,5.,8.,1.],[2.,9.,4.,7.],[6.,3.,11.,0.]]); \
    //   print(quantile_transform(X, axis=1, n_quantiles=4, output_distribution='uniform', subsample=None).tolist())"
    //   -> [[1.0,0.3333333333333334,0.666666666666667,0.0],
    //       [0.0,1.0,0.33333333333333337,0.666666666666667],
    //       [0.6666666666666667,0.3333333333333334,1.0,0.0]]
    let sk: [[f64; 4]; 3] = [
        [1.0, 0.333_333_333_333_333_4, 0.666_666_666_666_667, 0.0],
        [0.0, 1.0, 0.333_333_333_333_333_37, 0.666_666_666_666_667],
        [0.666_666_666_666_666_7, 0.333_333_333_333_333_4, 1.0, 0.0],
    ];
    let x = array![
        [10.0, 5.0, 8.0, 1.0],
        [2.0, 9.0, 4.0, 7.0],
        [6.0, 3.0, 11.0, 0.0]
    ];

    let out = quantile_transform(&x, 1, 4, OutputDistribution::Uniform, 0).unwrap();
    assert_eq!(
        out.dim(),
        (3, 4),
        "axis=1 must preserve original (3,4) shape"
    );
    for (i, row) in sk.iter().enumerate() {
        for (j, &expected) in row.iter().enumerate() {
            assert!(
                (out[[i, j]] - expected).abs() <= 1e-9,
                "[{i},{j}] free fn axis=1 = {}, sklearn = {expected}",
                out[[i, j]]
            );
        }
    }

    // Manual transpose-fit_transform-transpose must equal the free fn.
    let xt = x.t().to_owned();
    let manual_t = QuantileTransformer::<f64>::new(4, OutputDistribution::Uniform, 0)
        .fit(&xt, &())
        .unwrap()
        .transform(&xt)
        .unwrap();
    let manual = manual_t.t().to_owned();
    for (a, b) in out.iter().zip(manual.iter()) {
        assert!(
            (a - b).abs() <= 1e-15,
            "axis=1 free fn != manual transpose path: {a} vs {b}"
        );
    }
}

/// `quantile_transform(X, axis=2)` (any axis ∉ {0,1}) returns `Err(InvalidParameter)`,
/// mirroring sklearn's `ValueError("axis should be either equal to 0 or 1...")`.
#[test]
fn free_fn_invalid_axis_errors() {
    let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
    let err = quantile_transform(&x, 2, 3, OutputDistribution::Uniform, 0);
    match err {
        Err(ferrolearn_core::error::FerroError::InvalidParameter { name, reason }) => {
            assert_eq!(name, "axis");
            assert!(
                reason.contains("axis should be either equal to 0 or 1"),
                "reason was: {reason}"
            );
        }
        other => panic!("expected InvalidParameter for axis=2, got {other:?}"),
    }
}

/// f32 path: the free fn is generic and produces the same uniform ranks (~1e-6).
#[test]
fn free_fn_f32_axis0_uniform() {
    // oracle (same as f64): quantile_transform([[1.],[2.],[3.],[4.],[5.]], axis=0,
    //   n_quantiles=5, uniform, subsample=None) -> [0.0,0.25,0.5,0.75,1.0]
    let sk: [f32; 5] = [0.0, 0.25, 0.5, 0.75, 1.0];
    let x: ndarray::Array2<f32> = array![[1.0f32], [2.0], [3.0], [4.0], [5.0]];
    let out = quantile_transform(&x, 0, 5, OutputDistribution::Uniform, 0).unwrap();
    for (i, &expected) in sk.iter().enumerate() {
        assert!(
            (out[[i, 0]] - expected).abs() <= 1e-6,
            "row {i}: f32 free fn = {}, sklearn = {expected}",
            out[[i, 0]]
        );
    }
}

// ===========================================================================
// DIV-E — quantile_transform axis=1 on a single-column X (transpose -> 1 sample)
// rejects where sklearn succeeds. sklearn `quantile_transform`
// (`sklearn/preprocessing/_data.py:3117-3118` `X = n.fit_transform(X.T).T`)
// transposes a (n,1) matrix to (1,n) -> fits on n_samples=1, clamps
// n_quantiles to 1 (`:2785` warning + `:2790` `max(1, min(n_quantiles,
// n_samples))`), and returns all-zeros. ferrolearn's `QuantileTransformer::fit`
// (`quantile_transformer.rs:469`) hard-rejects `n_samples < 2` with
// `InsufficientSamples`, so the free fn returns `Err` where sklearn returns Ok.
// This also affects axis=0 on a single-row (1, n_features) input.
// ===========================================================================

/// Divergence: `quantile_transform(X, axis=1, ...)` on a `(5, 1)` matrix
/// diverges from sklearn `quantile_transform` (`_data.py:3117-3118`).
///
/// axis=1 transposes X to its transpose `(1, 5)` and runs `fit_transform` on it
/// (`X.T` has n_samples=1). sklearn clamps `n_quantiles` to `n_samples=1`
/// (`_data.py:2785`,`:2790`) and succeeds, returning the original `(5, 1)` shape
/// filled with `0.0`. ferrolearn's `fit` (`quantile_transformer.rs:469`:
/// `if n_samples < 2 { return Err(InsufficientSamples ...) }`) rejects the
/// single-sample transposed matrix, so the free fn returns `Err`.
///
/// Input: `[[1.],[2.],[3.],[4.],[5.]]`, axis=1, n_quantiles=5, Uniform.
/// sklearn returns `Ok` shape `(5,1)` = `[[0.],[0.],[0.],[0.],[0.]]`.
/// ferrolearn returns `Err(InsufficientSamples)`.
/// Tracking: #2218
#[test]
#[ignore = "divergence: quantile_transform axis=1 on single-column rejects (n_samples<2 guard) where sklearn clamps n_quantiles and succeeds; tracking #2218"]
fn divergence_free_fn_axis1_single_column_rejected() {
    // oracle: python3 -c "import numpy as np; from sklearn.preprocessing import quantile_transform; \
    //   import warnings; warnings.filterwarnings('ignore'); \
    //   X=np.array([[1.],[2.],[3.],[4.],[5.]]); \
    //   o=quantile_transform(X, axis=1, n_quantiles=5, output_distribution='uniform', subsample=None); \
    //   print(o.shape, o.ravel().tolist())"
    //   -> (5, 1) [0.0, 0.0, 0.0, 0.0, 0.0]
    let sk: [f64; 5] = [0.0, 0.0, 0.0, 0.0, 0.0];
    let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];

    let out = quantile_transform(&x, 1, 5, OutputDistribution::Uniform, 0)
        .expect("sklearn returns Ok((5,1)) for axis=1 single-column; ferrolearn must too");
    assert_eq!(
        out.dim(),
        (5, 1),
        "axis=1 single-column must preserve original (5,1) shape"
    );
    for (i, &expected) in sk.iter().enumerate() {
        assert!(
            (out[[i, 0]] - expected).abs() <= 1e-9,
            "row {i}: ferrolearn = {}, sklearn = {expected}",
            out[[i, 0]]
        );
    }
}
