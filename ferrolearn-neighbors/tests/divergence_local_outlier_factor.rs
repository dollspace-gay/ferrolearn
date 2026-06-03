//! Adversarial divergence pins for `ferrolearn-neighbors/src/local_outlier_factor.rs`
//! (`pub struct LocalOutlierFactor`, `FittedLocalOutlierFactor`, `fit`/`predict`/
//! `fit_predict`/`score_samples`/`decision_function`/`lof_scores`/`threshold`)
//! against the live scikit-learn 1.5.2 oracle
//! (`from sklearn.neighbors import LocalOutlierFactor`).
//!
//! Every expected value below is captured from a live `python3 -c "..."` run of
//! sklearn 1.5.2, run from `/tmp` (R-CHAR-3 — NEVER literal-copied from the
//! ferrolearn side). The exact oracle call and its output is quoted above each
//! assertion. Design doc: `.design/neighbors/local_outlier_factor.md`.
//!
//! Upstream (read-only tree at tag 1.5.2, `sklearn/neighbors/_lof.py`):
//!   * `_lof.py:205` — `contamination="auto"` is the constructor default.
//!   * `_lof.py:188-191` — `_parameter_constraints`: `contamination ∈
//!     StrOptions({"auto"}) | Interval(Real, 0, 0.5, closed="right")`.
//!   * `_lof.py:310` — `negative_outlier_factor_ = -mean(lrd_ratios_array, axis=1)`.
//!   * `_lof.py:312-318` — `offset_ = -1.5` if `contamination=="auto"` else
//!     `np.percentile(negative_outlier_factor_, 100.0 * contamination)`.
//!   * `_lof.py:380-381` — `_predict(None)`: `is_inlier[nof < offset_] = -1`.
//!   * `_lof.py:424` — `decision_function(X) = score_samples(X) - offset_`.
//!   * `_lof.py:481-484` — `score_samples = -mean(lrd[nbrs] / X_lrd, axis=1)`.
//!   * `_lof.py:511` — lrd `= 1.0 / (mean(reach_dist) + 1e-10)`.
//!
//! ferrolearn API under test (the signatures the fixer consumes — read from
//! `local_outlier_factor.rs`):
//!   * `LocalOutlierFactor::<F>::new()` / `with_n_neighbors` / `with_contamination`
//!     / `with_novelty` — builder; `pub contamination: f64` field (no `"auto"`).
//!   * `LocalOutlierFactor::fit_predict(&self, x: &Array2<F>) -> Result<Array1<isize>>`.
//!   * `<LocalOutlierFactor as Fit>::fit(&self, x, &()) -> FittedLocalOutlierFactor`.
//!   * `FittedLocalOutlierFactor::lof_scores(&self) -> &[F]`   (POSITIVE LOF).
//!   * `FittedLocalOutlierFactor::threshold(&self) -> F`        (positive-LOF space).
//!   * `FittedLocalOutlierFactor::score_samples(&self, x) -> Result<Array1<F>>`.
//!   * `FittedLocalOutlierFactor::decision_function(&self, x) -> Result<Array1<F>>`.
//!
//! RED pins (must FAIL against current local_outlier_factor.rs — deterministic,
//! tie-free divergences):
//!   * `divergence_offset_and_decision_function_novelty`   — #847 + #849:
//!     `decision_function = score_samples - offset_(=-1.5)`; ferrolearn subtracts
//!     a positive-LOF-space `threshold` → wrong shift.
//!   * `divergence_default_contamination_mislabels_mild_point` — #845 + #848:
//!     sklearn default `contamination="auto"` (`offset_=-1.5`) labels a MILD
//!     elevated point `+1` (its `nof=-1.489 > -1.5`); ferrolearn `new()` default
//!     `contamination=0.1` rank-thresholds the single highest-LOF point and
//!     mislabels it `-1`.
//!
//! GREEN guards (must PASS now — pin behavior that already value-matches the
//! oracle on a tie-free fixture, so a future fix must not regress it):
//!   * `green_negative_outlier_factor_value_via_lof_scores_tiefree` — #846: on the
//!     1-D, all-distinct-distance 4-point fixture (NO equidistant ties), sklearn
//!     `negative_outlier_factor_ == -lof_scores()` to ~1e-7 (the only residual is
//!     the `+1e-10` lrd damping, `_lof.py:511`, negligible at these magnitudes).
//!   * `green_score_samples_inlier_sign` — #849: `score_samples` of a clear inlier
//!     is negative and near -1 (LOF shape), matching the oracle sign convention.
//!
//! SKIPPED blockers (NOT expressible through the current public API — noted per
//! the critic brief, no forced test):
//!   * #845 default *value* `contamination=="auto"`: the ferrolearn field is
//!     `pub contamination: f64` and CANNOT hold the string `"auto"`; a
//!     `model.contamination == "auto"` assertion does not type-check. The
//!     OBSERVABLE consequence of the wrong default IS pinned via
//!     `divergence_default_contamination_mislabels_mild_point`.
//!   * #846 `negative_outlier_factor_` ACCESSOR: ferrolearn exposes no
//!     `negative_outlier_factor_` method — only `lof_scores()` (the un-negated
//!     LOF). The missing-accessor surface is not runtime-testable; the VALUE
//!     contract is pinned via `-lof_scores()` (GREEN on the tie-free fixture).
//!   * #846 tie-break VALUE divergence: the ~1e-2 divergence on the equidistant
//!     9-point ring fixture is NON-deterministic (depends on `knn_brute_force`
//!     index order vs sklearn `kneighbors` order on EXACTLY-equidistant points) —
//!     not pinned (the brief mandates tie-free, deterministic fixtures only).
//!   * #847 `offset_` ACCESSOR: ferrolearn exposes no `offset_` method (only a
//!     positive-LOF-space `threshold()`). The `offset_` VALUE contract surfaces
//!     observably through `decision_function` and IS pinned there (#849).
//!   * #850 `n_neighbors_` + `warnings.warn` on clamp: ferrolearn exposes no
//!     `n_neighbors_` attribute and Rust emits no Python warning — neither is
//!     expressible as a runtime assertion through this API.
//!   * #851 novelty `available_if` gating: sklearn makes `predict`/`score_samples`
//!     /`decision_function` raise `AttributeError` when `novelty=False` (and
//!     `fit_predict` when `novelty=True`). In Rust these are inherent methods that
//!     compile and run regardless of the `novelty` field — a typestate question,
//!     not a runtime divergence. Not expressible; SKIP.

use ferrolearn_core::traits::Fit;
use ferrolearn_neighbors::local_outlier_factor::LocalOutlierFactor;
use ndarray::{Array2, array};

/// Tie-free 1-D fixture: `X = [[-1.1],[0.2],[101.1],[0.3]]` (n_samples = 4).
/// All pairwise distances are DISTINCT (verified live:
///   0 -> nbrs (1, d=1.3), (3, d=1.4)
///   1 -> (3, d=0.1), (0, d=1.3)
///   2 -> (3, d=100.8), (1, d=100.9)
///   3 -> (1, d=0.1), (0, d=1.4)
/// ), so neighbor selection is deterministic with NO tie-break ambiguity.
fn four_points_1d() -> Array2<f64> {
    array![[-1.1], [0.2], [101.1], [0.3]]
}

/// 8-point inlier cluster used as the novelty-mode training set (matches the
/// design doc AC-5 fixture).
fn eight_point_cluster() -> Array2<f64> {
    array![
        [0.0, 0.0],
        [0.1, 0.0],
        [0.0, 0.1],
        [0.1, 0.1],
        [-0.1, 0.0],
        [0.0, -0.1],
        [0.05, 0.05],
        [-0.05, -0.05],
    ]
}

/// Tight cluster + ONE mildly-elevated point at `(0.17, 0.17)` (index 8).
/// Verified live: that point has the UNIQUELY highest LOF (lowest
/// `negative_outlier_factor_ = -1.48882`), but `-1.48882 > -1.5`, so sklearn's
/// default `contamination="auto"` (offset_=-1.5) labels it — and every other
/// point — an inlier (`+1`). This separates the two label mechanisms: a
/// positive-LOF *rank* threshold (ferrolearn) flags the highest-LOF point even
/// though its NOF is above sklearn's -1.5 offset.
fn cluster_with_mild_point() -> Array2<f64> {
    array![
        [0.0, 0.0],
        [0.1, 0.0],
        [0.0, 0.1],
        [0.1, 0.1],
        [-0.1, 0.0],
        [0.0, -0.1],
        [0.05, 0.05],
        [-0.05, -0.05],
        [0.17, 0.17],
    ]
}

// ===========================================================================
// GREEN 1 — #846 (REQ-2) value contract via `-lof_scores()` on the TIE-FREE
// fixture. The neighbor sets are unambiguous (all distinct distances), so the
// ONLY divergence source between ferrolearn's `1/mean` lrd and sklearn's
// `1/(mean + 1e-10)` (_lof.py:511) is the 1e-10 damping — negligible here.
// This GUARDS the value parity that already holds; a future fix must not break
// it. (The missing `negative_outlier_factor_` ACCESSOR is the un-testable part,
// noted in the header.)
// ===========================================================================
//
// Oracle (run from /tmp):
//   python3 -c "import numpy as np; from sklearn.neighbors import \
//   LocalOutlierFactor as L; X=np.array([[-1.1],[0.2],[101.1],[0.3]]); \
//   print(L(n_neighbors=2).fit(X).negative_outlier_factor_.tolist())"
//   -> [-0.9821428571441326, -1.0370370370342936, -73.36970898944223,
//       -0.9821428571441326]
#[test]
fn green_negative_outlier_factor_value_via_lof_scores_tiefree() {
    const SK_NOF: [f64; 4] = [
        -0.9821428571441326,
        -1.0370370370342936,
        -73.36970898944223,
        -0.9821428571441326,
    ];

    let x = four_points_1d();
    let model = LocalOutlierFactor::<f64>::new().with_n_neighbors(2);
    let fitted = model.fit(&x, &()).unwrap();
    let lof = fitted.lof_scores();
    assert_eq!(lof.len(), 4, "one LOF score per training sample");

    // sklearn `negative_outlier_factor_` == -lof (positive LOF). Tie-free, so
    // the only residual is the +1e-10 lrd damping → ~1e-7 relative.
    for (i, (&l, &nof)) in lof.iter().zip(SK_NOF.iter()).enumerate() {
        let ferro_nof = -l;
        let tol = 1e-6 * nof.abs().max(1.0);
        assert!(
            (ferro_nof - nof).abs() < tol,
            "sample {i}: -lof_scores()={ferro_nof}, sklearn negative_outlier_factor_={nof}"
        );
    }
}

// ===========================================================================
// GREEN 2 — #849 (REQ-5) sign convention: `score_samples` of a clear inlier is
// negative and near -1 (the LOF-shape contract sklearn shares). Pins only the
// sign/magnitude class, which already holds; the OFFSET shift is pinned RED
// below.
// ===========================================================================
//
// Oracle (run from /tmp):
//   python3 -c "import numpy as np; from sklearn.neighbors import \
//   LocalOutlierFactor as L; \
//   X=np.array([[0,0],[0.1,0],[0,0.1],[0.1,0.1],[-0.1,0],[0,-0.1],[0.05,0.05],\
//   [-0.05,-0.05]],dtype=float); c=L(n_neighbors=5,novelty=True).fit(X); \
//   print(c.score_samples(np.array([[0.02,0.02]],dtype=float)).tolist())"
//   -> [-0.9145758389282299]
#[test]
fn green_score_samples_inlier_sign() {
    let x = eight_point_cluster();
    let model = LocalOutlierFactor::<f64>::new().with_n_neighbors(5);
    let fitted = model.fit(&x, &()).unwrap();

    // A point inside the cluster (NOT in the training set, so the new-data path
    // runs). sklearn score_samples ~= -0.9146 (an inlier: negative, near -1).
    let xt = array![[0.02, 0.02]];
    let ss = fitted.score_samples(&xt).unwrap();
    assert_eq!(ss.len(), 1);
    assert!(
        ss[0] < 0.0,
        "score_samples of an inlier must be negative (sklearn -0.9146), got {}",
        ss[0]
    );
    assert!(
        ss[0] > -2.0,
        "score_samples of an inlier must be near -1 (sklearn -0.9146), got {}",
        ss[0]
    );
}

// ===========================================================================
// RED 1 — #847 (offset_) + #849 (decision_function): sklearn computes
// `decision_function(X) = score_samples(X) - offset_` (_lof.py:424) with
// `offset_ = -1.5` under the default `contamination="auto"` (_lof.py:312-314).
// ferrolearn returns `threshold - lof` where `threshold` is a positive-LOF-space
// rank index — the WRONG shift. RED: must FAIL until offset_=-1.5 lands.
// ===========================================================================
//
// Oracle (run from /tmp):
//   python3 -c "import numpy as np; from sklearn.neighbors import \
//   LocalOutlierFactor as L; \
//   X=np.array([[0,0],[0.1,0],[0,0.1],[0.1,0.1],[-0.1,0],[0,-0.1],[0.05,0.05],\
//   [-0.05,-0.05]],dtype=float); c=L(n_neighbors=5,novelty=True).fit(X); \
//   Xt=np.array([[100,100],[0.02,0.02]],dtype=float); \
//   print(c.offset_, c.decision_function(Xt).tolist())"
//   -> -1.5   [-929.9903234474953, 0.5854241610717701]
//
// The second value is the discriminating one: an INLIER must get a POSITIVE
// decision_function (sklearn 0.5854). ferrolearn shifts by the positive-LOF
// `threshold()` (~1.0..1.04) instead of `-offset_=+1.5`, so its inlier
// decision_function ~= 0.0888 — below sklearn's 0.5854 by ~0.5 (the offset gap).
#[test]
fn divergence_offset_and_decision_function_novelty() {
    // sklearn decision_function on the inlier query point (offset_ = -1.5).
    const SK_DECISION_INLIER: f64 = 0.5854241610717701;

    let x = eight_point_cluster();
    let model = LocalOutlierFactor::<f64>::new().with_n_neighbors(5);
    let fitted = model.fit(&x, &()).unwrap();

    let xt = array![[100.0, 100.0], [0.02, 0.02]];
    let df = fitted.decision_function(&xt).unwrap();
    assert_eq!(df.len(), 2);

    // The inlier (second point) must match sklearn's decision_function, which
    // equals `score_samples - (-1.5)`. ferrolearn subtracts the positive-LOF
    // `threshold` instead → diverges by ~0.5 (the |offset_| - threshold gap).
    assert!(
        (df[1] - SK_DECISION_INLIER).abs() < 1e-6,
        "decision_function inlier: ferrolearn={}, sklearn (score_samples - offset_=-1.5)={}",
        df[1],
        SK_DECISION_INLIER
    );
}

// ===========================================================================
// RED 2 — #845 (default contamination) + #848 (fit_predict labels): sklearn's
// DEFAULT `LocalOutlierFactor` uses `contamination="auto"` → `offset_=-1.5`
// (_lof.py:312-314); `_predict(None)` flags `nof < offset_` (_lof.py:380-381).
// The mild point (index 8) has `nof = -1.48882 > -1.5`, so sklearn labels EVERY
// point `+1`. ferrolearn `new()` defaults to `contamination=0.1` (NOT "auto");
// `fn fit` derives a positive-LOF *rank* threshold (`sorted[n - n_outliers - 1]`,
// `n_outliers = ceil(0.1*9).max(1) = 1`) and `fn predict` flags the single
// highest-LOF point — index 8 — as `-1`. RED: must FAIL until the `"auto"`
// default (offset_=-1.5, NOF-space rule) lands.
// ===========================================================================
//
// Oracle (run from /tmp):
//   python3 -c "import numpy as np; from sklearn.neighbors import \
//   LocalOutlierFactor as L; \
//   X=np.array([[0,0],[0.1,0],[0,0.1],[0.1,0.1],[-0.1,0],[0,-0.1],[0.05,0.05],\
//   [-0.05,-0.05],[0.17,0.17]],dtype=float); \
//   print(L(n_neighbors=3).fit_predict(X).tolist(), \
//         L(n_neighbors=3).fit(X).negative_outlier_factor_[8])"
//   -> [1, 1, 1, 1, 1, 1, 1, 1, 1]   -1.4888216...   (-1.4888 > -1.5 → inlier)
#[test]
fn divergence_default_contamination_mislabels_mild_point() {
    // sklearn default-constructed (contamination="auto") labels: ALL inliers.
    const SK_LABELS_AUTO: [isize; 9] = [1, 1, 1, 1, 1, 1, 1, 1, 1];

    let x = cluster_with_mild_point();
    // ferrolearn `new()` (default contamination = 0.1, NOT "auto").
    let model = LocalOutlierFactor::<f64>::new().with_n_neighbors(3);
    let labels = model.fit_predict(&x).unwrap();
    assert_eq!(labels.len(), 9);

    let got: Vec<isize> = labels.to_vec();
    assert_eq!(
        got,
        SK_LABELS_AUTO.to_vec(),
        "default-constructed fit_predict labels: ferrolearn (contamination=0.1, rank-threshold \
         flags highest-LOF point)={got:?}, sklearn (contamination='auto', offset_=-1.5, \
         nof=-1.489 > -1.5 → all inliers)={SK_LABELS_AUTO:?}"
    );
}
