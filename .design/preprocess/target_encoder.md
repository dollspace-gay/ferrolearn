# TargetEncoder

<!--
tier: 3-component
status: draft
baseline-commit: a8f22986
upstream: scikit-learn 1.5.2 (commit 156ef14)
upstream-paths:
  - sklearn/preprocessing/_target_encoder.py  # class TargetEncoder(OneToOneFeatureMixin, _BaseEncoder) (:55); __init__(*, categories='auto', target_type='auto', smooth='auto', cv=5, shuffle=True, random_state=None) (:195-209). DEFAULT smooth='auto' is an EMPIRICAL BAYES estimate (:85-89), NOT a fixed value; _parameter_constraints "smooth": [StrOptions({"auto"}), Interval(Real, 0, None, closed="left")] (:189). fit -> _fit_encodings_all (:347): infers target_type_ via type_of_target (:358-369), binary->LabelEncoder / multiclass->LabelBinarizer / continuous->_check_y (:371-381), target_mean_ = np.mean(y, axis=0) (:383), learns full-data encodings_; a float smooth uses the m-estimate (counts*means + smooth*y_mean)/(counts+smooth) via _fit_encoding_binary_or_continuous (:289). transform (:305): FULL-DATA encodings_ + target_mean_ for unknown (handle_unknown="ignore", :324-345). fit_transform (:232): CROSS-FITTING KFold(continuous)/StratifiedKFold (:254-303) -> fit(X,y).transform(X) != fit_transform(X,y) (:235-238). multiclass target_type_: output (n_samples, n_features*n_classes) (:269-273,:329-333,:376-379).
ferrolearn-module: ferrolearn-preprocess/src/target_encoder.rs
parity-ops: TargetEncoder
crosslink-issue: 1260
-->

## Summary

scikit-learn's `TargetEncoder` (`_target_encoder.py:55`) replaces each categorical
feature value with a smoothed estimate of the target mean for that category,
regularised toward the global target mean. With a **float** `smooth` it uses the
m-estimate `encoded = (counts*means + smooth*y_mean) / (counts + smooth)`
(`_fit_encoding_binary_or_continuous`, `:289`); the **default** `smooth="auto"`
is instead an *empirical Bayes* estimate (`:85-89`). It infers a `target_type_`
(binary / multiclass / continuous) via `type_of_target` (`:358-369`), stores
`target_mean_ = np.mean(y, axis=0)` (`:383`) and full-data `encodings_`, and at
`transform` uses those full-data encodings (`handle_unknown="ignore"`, unknowns →
`target_mean_`, `:324-345`). Crucially, `fit_transform` does **cross-fitting**
(`KFold`/`StratifiedKFold`, `:254-303`) so `fit(X,y).transform(X)` does NOT equal
`fit_transform(X,y)` (`:235-238`). For multiclass it emits
`(n_samples, n_features * n_classes)` columns (`:269-273`).

`ferrolearn-preprocess/src/target_encoder.rs` ships a **manual-`smooth`,
continuous/binary-style, `usize`-category** encoder with the unfitted/fitted
split: `TargetEncoder<F> { smooth: F }` (`new(smooth)`, `Default` = smooth 1.0,
`smooth()` accessor) and `FittedTargetEncoder<F> { category_maps:
Vec<HashMap<usize,F>>, global_mean: F }` (accessors `category_maps()`,
`global_mean()`). `impl Fit<Array2<usize>, Array1<F>>` rejects 0 rows
(`InsufficientSamples`), a `y`-length mismatch (`ShapeMismatch`), and a negative
`smooth` (`InvalidParameter`); computes `global_mean = mean(y)`, then per feature
per category collects `(sum, count)` and stores
`encoded = (count*cat_mean + smooth*global_mean)/(count+smooth) =
(sum + smooth*global)/(count+smooth)`. `impl Transform<Array2<usize>>` returns
`Array2<F>`, mapping each cell through `category_maps[j]` and falling back to
`global_mean` for unseen categories (`unwrap_or(global_mean)`). There is **no
`FitTransform`** (hence no cross-fitting), no `target_type`/multiclass path, no
`categories`/`cv`/`shuffle`/`random_state` params, no string categories, no
fitted-attribute name plumbing, and **no PyO3 binding**.

**Headline finding (document prominently — this is a VERIFY-AND-DOCUMENT unit).**
On the **manual-`smooth`, single-output (continuous/binary), `usize`-category**
`fit().transform()` path the m-estimate **VALUE-MATCHES sklearn exactly**
(confirmed live, Probe 1): `TargetEncoder::<f64>::new(2.0).fit(&X,&y).transform(&X)`
reproduces sklearn `encodings_=[[2.0,3.0]]`, `target_mean_=2.5`, and
`transform=[2,2,3,3]` for `X=[[0],[0],[1],[1]]`, `y=[1,2,3,4]` (cat0
`(2*1.5+2*2.5)/4=2.0`, cat1 `(2*3.5+2*2.5)/4=3.0`), and the unseen-category →
`target_mean_` (2.5) rule matches (REQ-1, REQ-2 SHIPPED) — now **bit-exact on the
f64 path** after two numerical fixes the critic surfaced (#1261 global_mean
pairwise summation, #1262 per-category encoding formula). The REMAINING divergences
are **STRUCTURAL, not fixable-this-iteration one-liners**: the **default constructor
itself diverges** because sklearn's default `smooth="auto"` is an empirical-Bayes
estimate (Probe 3: `encodings_=[[1.59..,3.41..]]`) while ferrolearn's `default()`
uses a fixed `smooth=1.0` (`[1.83..,3.17..]`) — REQ-4, the **headline structural
gap**; the entirely-absent **cross-fitting `fit_transform`** (`KFold`/
`StratifiedKFold`, Probe 2: `fit_transform != fit().transform()`) — REQ-5; the
**`target_type` binary/multiclass** path (multiclass → `n_features*n_classes`
columns, Probe 4) — REQ-6; and the `categories`/`cv`/`shuffle`/`random_state`
params, fitted-attribute names, string categories, feature-name plumbing, the
PyO3 binding, and the ferray substrate (REQ-7..REQ-12).

## Probes (live sklearn oracle, 1.5.2; run from /tmp)

```bash
# PROBE 1 (REQ-1, REQ-2, REQ-3) — manual smooth=2.0 m-estimate VALUE-match + unseen → target_mean_:
python3 -c "from sklearn.preprocessing import TargetEncoder; \
e=TargetEncoder(smooth=2.0, target_type='continuous').fit([[0],[0],[1],[1]],[1,2,3,4]); \
print('encodings_', [c.tolist() for c in e.encodings_], 'target_mean_', float(e.target_mean_)); \
print('transform', e.transform([[0],[0],[1],[1]]).ravel().tolist()); \
print('unseen', e.transform([[5]]).ravel().tolist())"
# -> encodings_ [[2.0, 3.0]]  target_mean_ 2.5
#    transform [2.0, 2.0, 3.0, 3.0]
#    unseen [2.5]
#    ferrolearn: TargetEncoder::<f64>::new(2.0).fit(&X,&y).transform(&X) == [2,2,3,3];
#    category_maps()[0] == {0:2.0, 1:3.0}; global_mean() == 2.5; unseen cat 5 -> 2.5. EXACT match.
#    Hand check: cat0 (2*1.5 + 2*2.5)/(2+2) = 2.0; cat1 (2*3.5 + 2*2.5)/(2+2) = 3.0.

# PROBE 2 (REQ-5) — fit_transform CROSS-FITTING != fit().transform():
python3 -c "import numpy as np; from sklearn.preprocessing import TargetEncoder; \
X=[[0],[0],[1],[1]]; y=[1,2,3,4]; \
ft=TargetEncoder(smooth=2.0,target_type='continuous',cv=2).fit_transform(X,y); \
tt=TargetEncoder(smooth=2.0,target_type='continuous',cv=2).fit(X,y).transform(X); \
print('fit_transform', ft.ravel().tolist(), 'fit().transform()', tt.ravel().tolist(), 'equal?', np.allclose(ft,tt))"
# -> fit_transform [3.5, 3.5, 1.5, 1.5]  fit().transform() [2.0, 2.0, 3.0, 3.0]  equal? False
#    sklearn fit_transform uses per-fold KFold encodings on the held-out fold (:254-303);
#    ferrolearn has NO FitTransform impl / no CV -> the cross-fitting path is unrepresentable.

# PROBE 3 (REQ-4) — DEFAULT smooth='auto' is EMPIRICAL BAYES, not fixed 1.0:
python3 -c "from sklearn.preprocessing import TargetEncoder; \
ed=TargetEncoder(target_type='continuous').fit([[0],[0],[1],[1]],[1,2,3,4]); \
print('auto encodings_', [c.tolist() for c in ed.encodings_])"
# -> auto encodings_ [[1.5909090909090908, 3.409090909090909]]   (empirical Bayes, :85-89)
#    ferrolearn default() uses smooth=1.0 -> cat0 (1*1.5+1*2.5)/2=1.833.., cat1 (1*3.5+1*2.5)/2=3.166..
#    -> [1.833.., 3.166..]. ferrolearn's DEFAULT diverges from sklearn's DEFAULT (auto). HEADLINE GAP.

# PROBE 4 (REQ-6) — multiclass target_type_ -> (n_samples, n_features*n_classes):
python3 -c "from sklearn.preprocessing import TargetEncoder; \
Xm=[[0],[0],[1],[1],[2],[2]]; ym=[0,1,2,0,1,2]; \
em=TargetEncoder(target_type='multiclass').fit(Xm,ym); \
print('target_type_', em.target_type_, 'transform shape', em.transform(Xm).shape, 'classes_', em.classes_.tolist())"
# -> target_type_ multiclass  transform shape (6, 3)  classes_ [0, 1, 2]
#    (1 feature * 3 classes = 3 output columns via LabelBinarizer, :269-273,:376-379)
#    ferrolearn: single continuous-style target only; output is always (n_samples, n_features).
```

## Requirements

- REQ-1: Manual-`smooth` per-category m-estimate **value match** on
  `fit().transform()` (continuous / binary single-output, `usize` categories) —
  learn `global_mean = mean(y)` and per feature per category
  `encoded = (count*cat_mean + smooth*global_mean)/(count+smooth)`, mirroring
  sklearn's float-`smooth` m-estimate `(counts*means + smooth*y_mean)/(counts+smooth)`
  (`_fit_encoding_binary_or_continuous`, `:289`; `target_mean_ = np.mean(y)`,
  `:383`), and apply the full-data `encodings_` at `transform` (`:305`, the
  full-data path ferrolearn matches). Output equals the oracle `encodings_`
  (Probe 1). Supports `f32`/`f64`.
- REQ-2: Unseen category at `transform` → `target_mean_` (global mean) — sklearn
  `handle_unknown="ignore"` maps unknown categories to `target_mean_`
  (`:324-345`); ferrolearn `category_maps[j].get(cat).unwrap_or(&global_mean)`
  does the same (Probe 1, unseen → 2.5).
- REQ-3: Error contracts (scoped) — `InsufficientSamples` on 0 rows,
  `ShapeMismatch` on a `y`-length-vs-rows mismatch in `fit` and on a column-count
  mismatch in `transform`, `InvalidParameter` on a negative `smooth` (sklearn
  validates `smooth >= 0` via `Interval(Real, 0, None, closed="left")`, `:189`).
- REQ-4: **`smooth="auto"` empirical-Bayes default** — sklearn's DEFAULT
  constructor uses `smooth="auto"`, an empirical Bayes estimate (`:85-89`,
  `:189`), NOT a fixed numeric value (Probe 3: `[1.59..,3.41..]`); ferrolearn's
  `Default` uses a fixed `smooth=1.0` (`[1.83..,3.17..]`). **The headline
  structural gap: ferrolearn's default-constructor result diverges from
  sklearn's default-constructor result.**
- REQ-5: **Cross-fitting `fit_transform`** — sklearn's `fit_transform`
  (`:232`) uses a `KFold` (continuous) / `StratifiedKFold` (classification)
  cross-fitting scheme, applying per-fold encodings to each held-out fold
  (`:254-303`), so `fit(X,y).transform(X) != fit_transform(X,y)` (`:235-238`,
  Probe 2); ferrolearn has **no `FitTransform` impl and no CV**.
- REQ-6: `target_type` binary / multiclass — sklearn infers `target_type_` via
  `type_of_target` (`:358-369`) and, for multiclass, `LabelBinarizer`-encodes
  `y` and emits `(n_samples, n_features * n_classes)` columns (`:269-273`,
  `:329-333`, `:376-379`, Probe 4); ferrolearn handles a single continuous-style
  target only, output always `(n_samples, n_features)`.
- REQ-7: `categories` param + `categories_` / `target_type_` / `classes_` fitted
  attributes — sklearn accepts user-supplied category lists (`categories='auto'`,
  `:197`) and exposes `categories_`, `target_type_`, `classes_`
  (`:358-381`); ferrolearn always derives categories from data and exposes only
  `category_maps()` / `global_mean()`.
- REQ-8: `cv` / `shuffle` / `random_state` constructor parameters — the
  cross-fitting controls (`cv=5`, `shuffle=True`, `random_state=None`,
  `:200-209`, `:262-264`); ferrolearn's `new(smooth)` takes only `smooth`.
- REQ-9: String / object categories — sklearn accepts any category dtype
  (`_BaseEncoder` ordinal-encoding of `X`); ferrolearn is `Array2<usize>`-only
  (R-DEV-3), so string/object categories are unrepresentable.
- REQ-10: `get_feature_names_out` / `n_features_in_` — sklearn exposes both
  (`OneToOneFeatureMixin` / `_BaseEncoder`); ferrolearn has neither.
- REQ-11: PyO3 binding — `import ferrolearn` exposes a `TargetEncoder` mirroring
  `import sklearn` (the project-boundary CPython consumer); **absent today**.
- REQ-12: ferray substrate — the encoder computes over `ferray-core` arrays
  rather than `ndarray::Array2<usize>` + `std::collections::HashMap`
  (R-SUBSTRATE).

## Acceptance criteria

- AC-1 (REQ-1): `TargetEncoder::<f64>::new(2.0).fit(&X,&y).transform(&X)` for
  `X=[[0],[0],[1],[1]]`, `y=[1,2,3,4]` equals `[[2.0],[2.0],[3.0],[3.0]]` within
  ULP tolerance (Probe 1); `category_maps()[0] == {0:2.0, 1:3.0}` and
  `global_mean() == 2.5`, equal to the oracle `encodings_=[[2.0,3.0]]` /
  `target_mean_=2.5`. Holds for `f32` and `f64`. Pinned by an oracle-grounded
  `#[test]` (R-CHAR-3).
- AC-2 (REQ-2): for the AC-1 fit, `transform([[5]])` (unseen category) equals
  `[[2.5]]` (= `global_mean()` = sklearn `target_mean_`, Probe 1).
- AC-3 (REQ-3): `fit` on `(0,n)` returns `Err(InsufficientSamples)`; a
  `y`-length-vs-rows mismatch in `fit` returns `Err(ShapeMismatch)`; a
  column-count mismatch in `transform` returns `Err(ShapeMismatch)`;
  `new(-1.0).fit(..)` returns `Err(InvalidParameter)` (sklearn rejects
  `smooth < 0` via `Interval`, `:189`).
- AC-4 (REQ-4): `TargetEncoder::<f64>::default()` reproduces sklearn
  `TargetEncoder(target_type='continuous')` (`smooth="auto"`) on the Probe 3
  fixture, i.e. `encodings_ == [[1.59..,3.41..]]` (empirical Bayes). NOT
  representable today: ferrolearn `default()` is fixed `smooth=1.0` →
  `[[1.83..,3.17..]]`.
- AC-5 (REQ-5): a `fit_transform(X,y)` on the Probe 2 fixture cross-fits
  (`[3.5,3.5,1.5,1.5]`) and differs from `fit(X,y).transform(X)`
  (`[2,2,3,3]`). NOT representable today (no `FitTransform`/CV).
- AC-6 (REQ-6): `target_type='multiclass'` on the Probe 4 fixture yields a
  `(6,3)` output (`n_features*n_classes`) with `classes_ == [0,1,2]`. NOT
  representable today (single continuous-style target).
- AC-7 (REQ-7): a user-supplied `categories=[[...]]` overrides the data-derived
  set; `categories_` / `target_type_` / `classes_` are exposed. Not present
  today.
- AC-8 (REQ-8): `cv`/`shuffle`/`random_state` control the cross-fitting splits.
  Not present today.
- AC-9 (REQ-9): a string-category `fit`/`transform` analog encodes correctly. Not
  representable today (`usize`-only).
- AC-10 (REQ-10): `get_feature_names_out()` returns the input feature names and
  `n_features_in_` is exposed. Neither present today.
- AC-11 (REQ-11): `python3 -c "import ferrolearn; ferrolearn.preprocessing.TargetEncoder"`
  resolves and matches sklearn on the Probe 1 case. No binding today.
- AC-12 (REQ-12): the encoder's owned state/compute uses `ferray-core` (no
  `ndarray`/`HashMap` in the compute path).

`## REQ status`

## REQ status

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (manual-`smooth` m-estimate value match, `fit().transform()`) | SHIPPED | impl `pub fn fit in target_encoder.rs` (`Fit<Array2<usize>, Array1<F>> for TargetEncoder<F>`) computes `global_mean = y.iter().sum() / n_samples`, then per feature collects `(sum, count)` per category and stores `encoded = (count_f * cat_mean + self.smooth * global_mean) / (count_f + self.smooth)` (where `cat_mean = sum / count_f`), i.e. the m-estimate; impl `pub fn transform in target_encoder.rs` (`Transform<Array2<usize>> for FittedTargetEncoder<F>`) writes `category_maps[j].get(cat)` per cell into an `Array2<F>`. **Bit-exact on the f64 path (two fixes this iteration):** `global_mean` now uses a numpy-faithful `pairwise_sum` helper matching `np.mean` (`:383`) across all size branches (#1261 closed); the per-category encoding now seeds the accumulator with `smooth*global_mean` then `+= yᵢ` and divides by `smooth+count` — `(smooth*y_mean + Σyᵢ)/(smooth+count)` — matching `_target_encoder_fast.pyx:60-75` exactly rather than the algebraically-rearranged `(count*(sum/count)+smooth*global)/(count+smooth)` (#1262 closed). Mirrors sklearn's float-`smooth` m-estimate (`_fit_encoding_binary_or_continuous`, `_target_encoder.py:289`) and the FULL-DATA `encodings_` `transform` path (`:305`,`:336-345`). Output equals the live oracle (Probe 1): `encodings_=[[2.0,3.0]]`, `target_mean_=2.5`, `transform=[2,2,3,3]` for `smooth=2.0`. `category_maps()` / `global_mean()` expose the learned maps + global mean. Non-test consumer: crate re-export `pub use target_encoder::{FittedTargetEncoder, TargetEncoder};` (`ferrolearn-preprocess/src/lib.rs` line 138) — boundary public API (grandfathered R-DEFER-1/S5). Verification: 17 oracle green guards in `tests/divergence_target_encoder.rs` (Probe-1 value match, multi-feature full matrix, binary-target, count-1 edge, pairwise across n<8/=8/13/101/=128/300/1000, interleaved-order accumulation, error contracts) + `cargo test -p ferrolearn-preprocess --lib target_encoder`; two-round critic-verified bit-exact on f64. **f32 path: see REQ-13 (#1263) — sklearn accumulates in f64 always, so `TargetEncoder<f32>` diverges.** |
| REQ-2 (unseen category → `target_mean_` / global mean) | SHIPPED | impl `pub fn transform in target_encoder.rs` does `out[[i,j]] = *cat_map.get(&cat).unwrap_or(&self.global_mean)` — a category absent from `category_maps[j]` falls back to `global_mean`. Mirrors sklearn `transform` with `handle_unknown="ignore"` mapping unknown categories to `target_mean_` (`_target_encoder.py:324-345`). Live oracle (Probe 1): unseen category `5` → `2.5` (= `target_mean_`); ferrolearn `transform([[5]])` → `global_mean()` = `2.5`. MATCH. Non-test consumer: crate re-export (`lib.rs` line 138). Verification: `cargo test -p ferrolearn-preprocess --lib target_encoder` (`test_target_encoder_unseen_category` asserts `2.5`). |
| REQ-3 (InsufficientSamples / ShapeMismatch / InvalidParameter error contracts) | SHIPPED | `Fit::fit` returns `Err(FerroError::InsufficientSamples { required: 1, actual: 0, context: "TargetEncoder::fit" })` when `n_samples == 0`, `Err(FerroError::ShapeMismatch { .. context: "TargetEncoder::fit — y must have same length as x rows" })` when `y.len() != n_samples`, and `Err(FerroError::InvalidParameter { name: "smooth", reason: "smoothing factor must be non-negative" })` when `self.smooth < F::zero()`; `Transform::transform` returns `Err(FerroError::ShapeMismatch { .. context: "FittedTargetEncoder::transform" })` when `x.ncols() != n_features`. The negative-`smooth` guard mirrors sklearn's `_parameter_constraints` `Interval(Real, 0, None, closed="left")` (`_target_encoder.py:189`). Non-test consumer: the error path guards every fitted instance reached through the crate re-export (`lib.rs` line 138). Verification: `cargo test -p ferrolearn-preprocess --lib target_encoder` (`test_target_encoder_zero_rows_error`, `test_target_encoder_shape_mismatch_fit`, `test_target_encoder_shape_mismatch_transform`, `test_target_encoder_negative_smooth_error`). Scoped: ferrolearn's `InsufficientSamples`/`y`-length checks are ferrolearn-side guards; the `smooth >= 0` check matches sklearn's `Interval`. |
| REQ-4 (`smooth="auto"` empirical-Bayes default) | NOT-STARTED | open prereq blocker #1264. `impl Default for TargetEncoder<F>` is `Self::new(F::one())` — a FIXED `smooth=1.0`; sklearn's DEFAULT `smooth="auto"` is an empirical Bayes estimate (`_target_encoder.py:85-89`,`:189`). **Headline structural gap: the default-constructor results diverge** — Probe 3 oracle `encodings_=[[1.59..,3.41..]]` vs ferrolearn `default()` `[[1.83..,3.17..]]`. `smooth` is a single `F` with no `"auto"` mode and no empirical-Bayes computation; not a one-liner. |
| REQ-5 (cross-fitting `fit_transform`, KFold/StratifiedKFold) | NOT-STARTED | open prereq blocker #1265. There is **no `impl FitTransform` for `TargetEncoder<F>`** and no cross-validation machinery; sklearn's `fit_transform` (`_target_encoder.py:232`) applies per-fold `KFold`/`StratifiedKFold` encodings to each held-out fold (`:254-303`), so `fit(X,y).transform(X) != fit_transform(X,y)` (`:235-238`, Probe 2: `[3.5,3.5,1.5,1.5]` vs `[2,2,3,3]`). The cross-fitting scheme (and a `model_selection` KFold dependency) is unrepresentable today. |
| REQ-6 (`target_type` binary/multiclass; multiclass → n_features*n_classes) | NOT-STARTED | open prereq blocker #1266. `Fit<Array2<usize>, Array1<F>>` treats `y` as a single continuous-style target (`global_mean = mean(y)`); there is no `type_of_target` inference (`:358-369`), no `LabelEncoder`/`LabelBinarizer` (`:371-379`), and no multiclass `(n_samples, n_features * n_classes)` output (`:269-273`,`:329-333`). Probe 4 (multiclass → `(6,3)`, `classes_=[0,1,2]`) unrepresentable. |
| REQ-7 (`categories` param + `categories_`/`target_type_`/`classes_` attrs) | NOT-STARTED | open prereq blocker #1267. `fit` always derives categories from the data; `TargetEncoder<F>` has no `categories` field (sklearn `categories='auto'`, `:197`) and `FittedTargetEncoder<F>` exposes only `category_maps()`/`global_mean()` — no `categories_`, `target_type_`, or `classes_` fitted attributes (`:358-381`). |
| REQ-8 (`cv` / `shuffle` / `random_state` params) | NOT-STARTED | open prereq blocker #1268. `new(smooth)` takes only `smooth`; sklearn's ctor has `cv=5`, `shuffle=True`, `random_state=None` (`:200-209`) driving the cross-fitting splitter (`:262-264`). No such fields. |
| REQ-9 (string / object categories) | NOT-STARTED | open prereq blocker #1269. `Fit<Array2<usize>, ..>` / `Transform<Array2<usize>>` are `usize`-category-only; sklearn ordinal-encodes any-dtype categories (`_BaseEncoder`). String/object categories are structurally unrepresentable (R-DEV-3). |
| REQ-10 (`get_feature_names_out` / `n_features_in_`) | NOT-STARTED | open prereq blocker #1270. `FittedTargetEncoder<F>` has neither `n_features_in_`/`feature_names_in_` exposure nor `get_feature_names_out` (sklearn `OneToOneFeatureMixin` / `_BaseEncoder`). |
| REQ-11 (PyO3 binding) | NOT-STARTED | open prereq blocker #1271. No `ferrolearn-python` registration of `TargetEncoder` (grep `TargetEncoder`/`target_encoder` over `ferrolearn-python/` returns nothing); `import ferrolearn` cannot expose it (boundary consumer per R-DEFER-1). |
| REQ-12 (ferray substrate) | NOT-STARTED | open prereq blocker #1272. State/compute use `ndarray::Array2<usize>`/`Array1<F>` + `std::collections::HashMap`, not `ferray-core`/`ferray-ufunc` (R-SUBSTRATE-1/2). |
| REQ-13 (f32 accumulates in lower precision than sklearn's f64) | NOT-STARTED | open prereq blocker #1263. sklearn `_target_encoder_fast.pyx:42,44,68` declares the smoothing/sum accumulators as C `double` regardless of the input `Y_DTYPE`, so `encodings_` is **always float64**; `TargetEncoder<f32>` accumulates per-category sums (`:226`) and the `pairwise_sum` mean in `f32`, losing precision. Live oracle: `TargetEncoder<f32>::new(0.0).fit([[0]×4], [2^24,1,1,1]).category_maps()[0][0]` = `4194304.0` (f32, the `+1`s past 2^24 lost) vs sklearn `4194304.75` (f64). **Structural (R-DEFER-3, no committed failing test):** a property of ferrolearn's grandfathered generic-`F` design (CLAUDE.md f32+f64 support) that applies workspace-wide — the faithful fix (accumulate internally in f64, present in `F`) is a cross-cutting concern, not a per-estimator one-liner. The **f64 instantiation (CLAUDE.md default) is bit-exact CLEAN** (REQ-1). |

## Architecture

**ferrolearn (existing).** `target_encoder.rs` exposes the unfitted/fitted pair.
`TargetEncoder<F> { smooth: F }` is constructed by `new(smooth)`, with `Default`
= `new(F::one())` (fixed `smooth=1.0`) and a `smooth()` accessor.
`FittedTargetEncoder<F> { category_maps: Vec<HashMap<usize,F>>, global_mean: F }`
exposes `category_maps() -> &[HashMap<usize,F>]` and `global_mean() -> F`. `impl
Fit<Array2<usize>, Array1<F>> for TargetEncoder<F>` (`fit`) rejects
`n_samples == 0` (`InsufficientSamples`), a `y`-length-vs-rows mismatch
(`ShapeMismatch`), and `smooth < 0` (`InvalidParameter`); computes `global_mean =
sum(y)/n_samples`; then for each column `j` collects a `HashMap<usize,(F,usize)>`
of `(sum, count)` per category and, per category, sets
`encoded = (count_f * cat_mean + self.smooth * global_mean) / (count_f +
self.smooth)` (with `cat_mean = sum / count_f`), pushing the per-column
`HashMap<usize,F>` into `category_maps`. `impl Transform<Array2<usize>> for
FittedTargetEncoder<F>` (`transform`) returns `ShapeMismatch` when
`x.ncols() != category_maps.len()`, then writes `out[[i,j]] =
*category_maps[j].get(&cat).unwrap_or(&self.global_mean)` into an `Array2<F>`
(unseen category → `global_mean`). The generic bound `F: Float + Send + Sync +
'static` supports `f32`/`f64`. There is **no `FitTransform` impl** (so no
cross-fitting), no `target_type`/multiclass branch, and no PyO3 binding; the crate
re-exports both public types (`ferrolearn-preprocess/src/lib.rs` line 138).

**sklearn (target contract).** `TargetEncoder(OneToOneFeatureMixin, _BaseEncoder)`
(`:55`). `__init__` (`:195-209`) takes `categories='auto'`, `target_type='auto'`,
`smooth='auto'`, `cv=5`, `shuffle=True`, `random_state=None`, with
`_parameter_constraints` including `"smooth": [StrOptions({"auto"}),
Interval(Real, 0, None, closed="left")]` (`:189`). `fit` delegates to
`_fit_encodings_all` (`:347`): it infers `target_type_` via `type_of_target`
(`:358-369`), then `binary → LabelEncoder` / `multiclass → LabelBinarizer` /
`continuous → _check_y` (`:371-381`), sets `target_mean_ = np.mean(y, axis=0)`
(`:383`), and learns the full-data `encodings_`. For a **float** `smooth` the
per-category m-estimate is `(counts*means + smooth*y_mean)/(counts+smooth)`
(`_fit_encoding_binary_or_continuous`, `:289`); for the **default** `smooth="auto"`
it is an empirical Bayes estimate (`:85-89`). `transform` (`:305`) uses the
**full-data** `encodings_` + `target_mean_` for unknown categories
(`handle_unknown="ignore"`, `:324-345`); for `multiclass` it emits
`(n_samples, n_features * len(classes_))` columns (`:329-333`). `fit_transform`
(`:232`) instead **cross-fits**: `KFold` (continuous) / `StratifiedKFold`
(classification) splits, with per-fold encodings applied to the held-out fold
(`:254-303`), so `fit(X,y).transform(X) != fit_transform(X,y)` (`:235-238`).
`get_feature_names_out` / `n_features_in_` come from `OneToOneFeatureMixin` /
`_BaseEncoder`.

**The structural gap.** On the **manual-`smooth`, single-output
(continuous/binary), `usize`-category** `fit().transform()` path the two coincide
exactly: the m-estimate `encodings_`, `target_mean_`, and the unseen-category →
`target_mean_` rule are value-identical to the oracle (REQ-1/REQ-2 SHIPPED; Probe
1), and the error contracts (REQ-3) ship. What differs is **structural**, and none
of it is a fixable-this-iteration one-liner: the **default constructor itself**
(ferrolearn fixed `smooth=1.0` vs sklearn empirical-Bayes `smooth="auto"` — REQ-4,
the headline gap that changes the `default()` result, Probe 3); the entirely
absent **cross-fitting `fit_transform`** (`KFold`/`StratifiedKFold` — REQ-5, Probe
2); the **`target_type` binary/multiclass** path (multiclass →
`n_features*n_classes` columns via `LabelBinarizer` — REQ-6, Probe 4); plus the
`categories` param + fitted-attribute names (REQ-7), `cv`/`shuffle`/`random_state`
(REQ-8), string/object categories (REQ-9), `get_feature_names_out`/`n_features_in_`
(REQ-10), the PyO3 binding (REQ-11), and the ferray substrate (REQ-12).

## Verification

Commands establishing the SHIPPED claims (REQ-1, REQ-2, REQ-3):

```bash
# Oracle (Probe 1, run from /tmp) — manual smooth=2.0 m-estimate + unseen → target_mean_:
python3 -c "from sklearn.preprocessing import TargetEncoder; \
e=TargetEncoder(smooth=2.0, target_type='continuous').fit([[0],[0],[1],[1]],[1,2,3,4]); \
print('encodings_', [c.tolist() for c in e.encodings_], 'target_mean_', float(e.target_mean_)); \
print('transform', e.transform([[0],[0],[1],[1]]).ravel().tolist(), 'unseen', e.transform([[5]]).ravel().tolist())"
#   -> encodings_ [[2.0, 3.0]]  target_mean_ 2.5  transform [2.0, 2.0, 3.0, 3.0]  unseen [2.5]
# ferrolearn equivalent: TargetEncoder::<f64>::new(2.0).fit(&X,&y).unwrap().transform(&X)
#   == [[2.0],[2.0],[3.0],[3.0]]; category_maps()[0]=={0:2.0,1:3.0}; global_mean()==2.5; unseen→2.5.

# Crate gauntlet:
cargo test -p ferrolearn-preprocess --lib target_encoder
#   -> SHIPPED REQ tests: test_target_encoder_basic, test_target_encoder_smoothing,
#      test_target_encoder_unseen_category, test_target_encoder_multi_feature,
#      test_target_encoder_global_mean_accessor, test_target_encoder_f32,
#      test_target_encoder_zero_rows_error, test_target_encoder_shape_mismatch_fit,
#      test_target_encoder_shape_mismatch_transform, test_target_encoder_negative_smooth_error,
#      test_target_encoder_default
cargo clippy -p ferrolearn-preprocess --all-targets -- -D warnings
cargo fmt --all --check
```

The in-module `#[test]`s exercise the REQ-1/REQ-2/REQ-3 manual-`smooth` path and
are green and coincide with the oracle. **The SHIPPED claim is the
manual-`smooth` `fit().transform()` value match** (REQ-1/REQ-2), now bit-exact on
f64 after TWO numerical fixes the critic surfaced: #1261 (global_mean pairwise
summation) and #1262 (per-category encoding formula). 17 oracle-grounded green
guards in `tests/divergence_target_encoder.rs` lock the f64 path (R-CHAR-3, all
expected values from live sklearn/numpy). The f32 path diverges (REQ-13, #1263) —
sklearn accumulates in f64 always.
No currently-green command establishes REQ-4..REQ-12; in particular
`test_target_encoder_default` pins the fixed `smooth=1.0` default, which
**diverges** from sklearn's `smooth="auto"` default (REQ-4, Probe 3) — a
characterization of the structural default-constructor gap, not a value-match.

## Blockers

Two genuine f64-path divergences were FIXED this iteration (#1261 global_mean
pairwise summation, #1262 per-category encoding formula); the rest are open
`-l blocker` issues. These remaining gaps are **structural** (REQ-1/REQ-2/REQ-3
SHIP bit-exact on f64):

- #1264 — REQ-4: `Default` is fixed `smooth=1.0`; sklearn's default
  `smooth="auto"` is an empirical Bayes estimate (`_target_encoder.py:85-89`,
  `:189`). **The default-constructor results diverge** (Probe 3:
  `[[1.59..,3.41..]]` vs `[[1.83..,3.17..]]`) — needs a `smooth` "auto" mode +
  empirical-Bayes computation.
- #1265 — REQ-5: no `FitTransform`/CV; sklearn's `fit_transform` cross-fits
  via `KFold`/`StratifiedKFold` (`:232`,`:254-303`), so
  `fit(X,y).transform(X) != fit_transform(X,y)` (`:235-238`, Probe 2).
- #1266 — REQ-6: single continuous-style target only; no `type_of_target`
  inference / `LabelEncoder`/`LabelBinarizer` / multiclass
  `n_features*n_classes` output (`:358-379`,`:269-273`, Probe 4).
- #1267 — REQ-7: no `categories` param; no `categories_`/`target_type_`/
  `classes_` fitted attributes (`:197`,`:358-381`).
- #1268 — REQ-8: `new(smooth)` lacks `cv`/`shuffle`/`random_state`
  (`:200-209`,`:262-264`).
- #1269 — REQ-9: `Array2<usize>`-only; no string/object categories (R-DEV-3).
- #1270 — REQ-10: no `get_feature_names_out`/`n_features_in_`
  (`OneToOneFeatureMixin`/`_BaseEncoder`).
- #1271 — REQ-11: no `ferrolearn-python` registration of `TargetEncoder`
  (boundary CPython consumer, R-DEFER-1).
- #1272 — REQ-12: compute path on `ndarray`/`HashMap`, not ferray
  (R-SUBSTRATE-1/2).
- #1263 — REQ-13: `TargetEncoder<f32>` accumulates per-category sums + the mean
  in `f32`; sklearn always accumulates in C `double` and `encodings_` is float64
  (`_target_encoder_fast.pyx:42,44,68`). Structural (generic-`F` design,
  workspace-wide); the f64 path is bit-exact CLEAN.

Two genuine f64-path numerical divergences were FOUND and FIXED this iteration
(not structural): **#1261** (global_mean used a naive sequential sum vs numpy
`np.mean` pairwise summation — now a `pairwise_sum` helper matching numpy across
all size branches) and **#1262** (per-category encoding used the rearranged
`(count*(sum/count)+smooth*global)/(count+smooth)` losing a bit vs sklearn's
`(smooth*y_mean + Σyᵢ)/(smooth+count)` seeded accumulator, `_target_encoder_fast.pyx:60-75`).
