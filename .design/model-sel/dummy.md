# DummyClassifier & DummyRegressor

<!--
tier: 3-component
status: draft
baseline-commit: d8c7a47fef52baf299f0c19cf3d0c8b753487796
upstream-paths:
  - sklearn/dummy.py   # class DummyClassifier (:35-:462), class DummyRegressor (:465-:702)
-->

## Summary

`ferrolearn-model-sel/src/dummy.rs` mirrors scikit-learn's two baseline
estimators in `sklearn/dummy.py`: `DummyClassifier` (`sklearn/dummy.py:35`) and
`DummyRegressor` (`sklearn/dummy.py:465`). Both ignore the input features `X`
and predict a rule-of-thumb value derived from the training targets `y` — a
constant or a marginal-distribution sample.

ferrolearn ships the **deterministic, single-output** slice of both: the
`most_frequent`/`prior`/`constant` classifier strategies (predict), and the
`mean`/`median`/`quantile`/`constant` regressor strategies. The RNG-coupled
classifier strategies (`stratified`, `uniform`) exist but cannot match sklearn
exactly because the random substrate differs (`SmallRng` vs numpy
`RandomState`) — R-DEFER-3 carve-out. `predict_proba`/`predict_log_proba`,
`sample_weight`, and multi-output `y` are all absent — NOT-STARTED. The
R-SUBSTRATE migration to `ferray-core` is also NOT-STARTED.

## Upstream reference (scikit-learn 1.5.2, commit 156ef14)

### DummyClassifier
- `sklearn/dummy.py:35` — `class DummyClassifier(MultiOutputMixin, ClassifierMixin, BaseEstimator)`.
- `:149-155` — `_parameter_constraints`: `strategy` is
  `StrOptions({"most_frequent","prior","stratified","uniform","constant"})`,
  `random_state`, `constant` is `[Integral, str, "array-like", None]`.
- `:157` — `__init__(self, *, strategy="prior", random_state=None, constant=None)`
  (default `strategy="prior"`).
- `:182-251` — `fit`: validates `X`, builds `(classes_, n_classes_, class_prior_)`
  via `class_distribution(y, sample_weight)` (`:228-230`); constant-strategy
  membership check raises `ValueError` if the constant is not in `classes_`
  (`:232-244`); `sample_weight` folded into `class_distribution` (`:211-212`);
  1D `y` reshaped to `(-1,1)` and `n_outputs_` set (`:204-207`).
- `:253-338` — `predict`: for `most_frequent`/`prior`,
  `classes_[k][class_prior_[k].argmax()]` tiled over `n_samples` (`:308-315`);
  `constant` tiled (`:332-333`); `stratified` takes `proba.argmax(axis=1)`
  (`:317-323`); `uniform` is `rs.randint(n_classes_, size=n_samples)`
  (`:325-330`). `argmax()` returns the FIRST maximal index → first-class tie-break.
- `:340-401` — `predict_proba`: `most_frequent` → one-hot on the argmax column
  (`:376-379`); `prior` → `class_prior_` broadcast over rows (`:380-381`);
  `stratified` → `rs.multinomial(1, class_prior_, size=n_samples)` (`:383-385`);
  `uniform` → `1/n_classes_` (`:387-389`); `constant` → one-hot on the constant
  column (`:391-394`).
- `:403-423` — `predict_log_proba` = `np.log(predict_proba(X))`.

### DummyRegressor
- `sklearn/dummy.py:465` — `class DummyRegressor(MultiOutputMixin, RegressorMixin, BaseEstimator)`.
- `:531-539` — `_parameter_constraints`: `strategy` is
  `StrOptions({"mean","median","quantile","constant"})`,
  `quantile` is `Interval(Real, 0.0, 1.0, closed="both")` or `None`,
  `constant` is an open `Interval` / array-like / `None`.
- `:541` — `__init__(self, *, strategy="mean", constant=None, quantile=None)`
  (default `strategy="mean"`).
- `:546-628` — `fit`: empty-`y` raises `ValueError("y must not be empty.")`
  (`:569-570`); `mean` → `np.average(y, axis=0, weights=sample_weight)` (`:581-582`);
  `median` → `np.median` or `_weighted_percentile(...,50.0)` (`:584-591`);
  `quantile` → `np.percentile(y, q=quantile*100)` or `_weighted_percentile`,
  raising `ValueError` if `quantile is None` (`:593-606`); `constant` → user
  value, raising `TypeError` if `None` (`:608-620`). `constant_` reshaped to
  `(1, -1)` (`:627`).
- `:630-666` — `predict`: `np.full((n_samples, n_outputs_), constant_)` (`:655-659`),
  ravelled to 1D when `n_outputs_ == 1` (`:662-664`); optional all-zero `y_std`
  via `return_std` (`:660`, `:666`).

## Requirements

- REQ-1: Deterministic classifier predict (`most_frequent`/`prior`/`constant`).
  `fit` counts classes (sorted ascending), computes `class_priors`; `predict`
  returns the argmax-prior class for `most_frequent`/`prior` and the constant for
  `constant`, tiled over every test row. Mirrors `sklearn/dummy.py:308-315`,
  `:332-333`. First-class tie-break matches sklearn's `argmax()`
  (`:311`). DETERMINISTIC / oracle-pinnable.
- REQ-2: `predict_proba` / `predict_log_proba`. sklearn exposes per-strategy
  probability output (`sklearn/dummy.py:340-423`); for `prior` this is the
  `class_prior_` row, the ONLY behavioral difference from `most_frequent` (whose
  `predict` is identical). `FittedDummyClassifier` has no `predict_proba`.
- REQ-3: `stratified` / `uniform` classifier strategies (RNG carve-out). Both
  draw from the empirical/uniform class distribution
  (`sklearn/dummy.py:317-330`, `:383-389`). ferrolearn implements them on
  `SmallRng`; exact sample-by-sample parity with numpy `RandomState` is
  impossible (different PRNG). R-DEFER-3 carve-out: blocker, NO failing test.
- REQ-4: Constant-membership validation (classifier). sklearn raises a
  `ValueError` when the `constant` is absent from `classes_`
  (`sklearn/dummy.py:232-244`). ferrolearn rejects it in `fit`; the error VARIANT
  diverges (`FerroError::InvalidParameter` vs `ValueError`).
- REQ-5: Classifier fitted attributes (`classes_`, `n_classes_`, `class_prior_`).
  sklearn exposes all three (`sklearn/dummy.py:101-113`, `:228-249`). ferrolearn
  exposes `classes_`/`n_classes_` via `HasClasses` and the most-frequent class via
  `most_frequent()`; `class_prior_` is stored but not publicly accessible.
- REQ-6: Deterministic regressor predict value parity
  (`mean`/`median`/`quantile`/`constant`). `fit` computes the constant; `predict`
  tiles it over every row. Mirrors `sklearn/dummy.py:581-628`, `:655-664`.
  Median and quantile match numpy's linear-interpolation percentile.
  DETERMINISTIC / oracle-pinnable.
- REQ-7: Regressor `quantile`/`constant` parameter validation. sklearn requires
  `quantile in [0,1]` and that it be set for `strategy="quantile"`
  (`sklearn/dummy.py:533`, `:593-598`), and raises `TypeError` if a `constant`
  strategy has no constant (`:608-613`). ferrolearn validates q ∈ [0,1] in
  `quantile_value`; the "quantile required" / "constant required" states are
  type-eliminated (q and the constant are constructor payloads, never `None`).
- REQ-8: `sample_weight` (both estimators). sklearn folds weights into
  `class_distribution` / `np.average` / `_weighted_percentile`
  (`sklearn/dummy.py:211-212`, `:582`, `:588-606`). ferrolearn `fit` takes no
  `sample_weight`.
- REQ-9: Multi-output `y` (both estimators). sklearn reshapes 1D→2D, tracks
  `n_outputs_`, and predicts `(n_samples, n_outputs)` (`sklearn/dummy.py:204-207`,
  `:574`, `:655-664`). ferrolearn's `Fit<.., Array1<..>>` is single-output only.
- REQ-10: R-SUBSTRATE — array type on `ferray-core` rather than `ndarray`, and
  random sampling on `ferray::random` rather than `rand`'s `SmallRng`.
- REQ-11: Non-test production consumer.

## Acceptance criteria

- AC-1 (REQ-1): with `y=[0,0,1,1,2]` (priors `[0.4,0.4,0.2]`, tie on classes 0/1),
  `DummyClassifier::new(MostFrequent)` and `new(Prior)` both predict all-`0`,
  matching the live oracle `DummyClassifier(strategy=...).fit(X,y).predict(Xt)`
  → `[0,0,0]`. Constant strategy returns the constant for every row.
  DETERMINISTIC / oracle-pinnable.
- AC-2 (REQ-2): `predict_proba` for `prior` returns the `class_prior_` row
  (`[0.4,0.4,0.2]`), for `most_frequent` the one-hot `[1,0,0]`, for `uniform`
  `[1/3,1/3,1/3]` — matching the live oracle.
- AC-3 (REQ-3): `stratified`/`uniform` predictions are in-range but are NOT
  asserted equal to sklearn (different PRNG); a carve-out blocker is filed.
- AC-4 (REQ-4): a constant absent from training labels errors at `fit`; the error
  TYPE matches sklearn's `ValueError` semantics (currently
  `FerroError::InvalidParameter`).
- AC-5 (REQ-5): `classes_`, `n_classes_`, and `class_prior_` are all publicly
  readable and equal the live-oracle attributes.
- AC-6 (REQ-6): for `y=[1,2,3,4,5]`, `mean`→3.0, `median`→3.0,
  `quantile(0.25)`→2.0; for even `y=[1,2,3,4]`, `median`→2.5 — all matching the
  live oracle. DETERMINISTIC / oracle-pinnable.
- AC-7 (REQ-7): `quantile=1.5` errors; `quantile`-strategy with a missing quantile
  / `constant`-strategy with a missing constant error with sklearn-equivalent
  semantics.
- AC-8 (REQ-8): a weighted `fit` yields the weighted mean/percentile matching the
  live oracle's `sample_weight` path.
- AC-9 (REQ-9): 2D `y` (`n×t`) predicts `n×t`; 1D `y` predicts 1D.
- AC-10 (REQ-10): owned computation runs on `ferray-core` + `ferray::random`, no
  `ndarray`/`rand` in owned computation.
- AC-11 (REQ-11): the estimators are constructed from non-test production code.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (deterministic classifier predict) | SHIPPED | impl `Predict<Array2<F>> for FittedDummyClassifier` in `dummy.rs` returns `Array1::from_elem(n, self.most_frequent())` for `MostFrequent`/`Prior` and `Array1::from_elem(n, *c)` for `Constant`; `most_frequent()` in `dummy.rs` scans `class_priors` keeping the FIRST maximum (`if p > best_p`), and `classes` is `sort_unstable`-sorted in `impl Fit … for DummyClassifier::fit`. Mirrors sklearn `classes_[k][class_prior_[k].argmax()]` tiled (`sklearn/dummy.py:308-315`) and `np.tile(self.constant,…)` (`:332-333`); first-class tie-break matches `argmax()` (`:311`). DETERMINISTIC / oracle-pinnable: `y=[0,0,1,1,2]` → live oracle `predict` `[0,0,0]` for BOTH `most_frequent` and `prior` (classes_ `[0,1,2]`, class_prior_ `[0.4,0.4,0.2]`; Verification). Tests: `dummy_classifier_most_frequent`, `dummy_classifier_constant`. CAVEAT: existing `#[test]`s use a single-mode `y` (no 0/1 prior tie) and assert hand-derived constants — the critic should pin an oracle-grounded tie-break `#[test]` on `y=[0,0,1,1,2]` expecting the live-oracle vector above (R-CHAR-3). Non-test consumer: REQ-11. |
| REQ-2 (predict_proba / predict_log_proba) | SHIPPED | `FittedDummyClassifier::predict_proba(&self, x) -> Array2<f64>` returns `(n_samples, n_classes)`: `Prior` → `class_priors` row broadcast (`dummy.py:380-381`); `MostFrequent`/`Constant` → one-hot at the class INDEX `argmax(class_prior_)` / `position(constant)` (`:376-379`); `Uniform` → `1/n_classes` (`:387-389`); `Stratified` → per-row random one-hot via the existing `make_rng`/`weighted_choice` (R-DEFER-3 RNG carve-out, not oracle-pinned). `predict_log_proba(x) = predict_proba(x).mapv(f64::ln)` (`0.0 → −inf`, `:391-423`). Verification (live sklearn 1.5.2, R-CHAR-3, `y=[0,0,1,1,2]`, priors `[0.4,0.4,0.2]`, classes `[0,1,2]`): `prior` rows `[0.4,0.4,0.2]`, `most_frequent` `[1,0,0]`, `uniform` `[1/3,1/3,1/3]`; log_proba prior `ln([0.4,0.4,0.2])`, most_frequent zeros `−inf`. Tests `dummy_predict_proba_prior_matches_sklearn`, `dummy_predict_proba_most_frequent_matches_sklearn`, `dummy_predict_proba_uniform_matches_sklearn`, `dummy_predict_log_proba_matches_sklearn`. |
| REQ-3 (stratified / uniform — RNG carve-out) | NOT-STARTED | open prereq blocker (tracking #1690; R-DEFER-3 carve-out — NO failing test). impl `Predict … for FittedDummyClassifier` implements `Stratified` via `weighted_choice` over an `rng.random::<f64>()` cumulative-prior draw and `Uniform` via `self.classes.choose(&mut rng)`, seeded by `make_rng` (`SmallRng::seed_from_u64`). sklearn draws `rs.multinomial`/`rs.randint` from numpy `RandomState` (`sklearn/dummy.py:317-330`, `:383-385`). Because the PRNG substrates differ, exact per-sample predictions CANNOT match even with equal seeds; `random_state=None` is non-deterministic in both. Per R-DEFER-3 this is a documented carve-out (distributional, not exact, parity) — a blocker is filed but NO failing `#[test]` is pinned. Existing in-range tests `dummy_classifier_stratified_in_range`, `dummy_classifier_uniform_in_range` assert only membership, not oracle equality. |
| REQ-4 (constant-membership validation, classifier) | SHIPPED | impl `Fit … for DummyClassifier::fit` in `dummy.rs` returns `FerroError::InvalidParameter { name: "constant", reason: "… constant {c} not present in training labels" }` when `!classes.contains(&c)` for `Constant(c)`. Mirrors sklearn's membership guard (`sklearn/dummy.py:232-244`). Live oracle: `DummyClassifier(strategy='constant',constant=99).fit(X,y)` → `ValueError` (Verification). DIVERGENCE (honest underclaim): the error VARIANT is `FerroError::InvalidParameter`, not sklearn's `ValueError`; R-DEV-2 maps `ValueError` to ferrolearn's parameter/validation error family, so this is SHIPPED on the rejection behavior, with the type-name mapping noted (critic may pin the exact `FerroError` variant if a stricter contract is wanted). Test: `dummy_classifier_constant_invalid` (`fit` `is_err()`). Non-test consumer: REQ-11. |
| REQ-5 (classifier fitted attributes) | SHIPPED | `impl HasClasses for FittedDummyClassifier` in `dummy.rs` exposes `classes()` (sorted `&[usize]`) and `n_classes()`; `fit` populates `classes`/`class_priors`. Mirrors sklearn `classes_`/`n_classes_`/`class_prior_` (`sklearn/dummy.py:228-249`); live oracle: `classes_=[0,1,2]`, `n_classes_=3`, `class_prior_=[0.4,0.4,0.2]` (Verification). CAVEAT (honest underclaim): `class_prior_` is computed and stored in the `class_priors` field but is NOT publicly readable — there is no `class_prior()`/`class_prior_` accessor and the field is private. SHIPPED on `classes_`/`n_classes_` parity; the `class_prior_` ACCESSOR is a gap (folded into REQ-2's blocker, since prior exposure and `predict_proba` share the same missing surface). Test: `n_classes()` asserted in `dummy_classifier_most_frequent`. Non-test consumer: REQ-11. |
| REQ-6 (deterministic regressor predict value parity) | SHIPPED | impl `Fit … for DummyRegressor::fit` in `dummy.rs` computes the constant: `Mean` = `y.iter().fold(zero, +) / n`, `Median` = `quantile_value(y, 0.5)`, `Quantile(q)` = `quantile_value(y, q)`, `Constant(c)` = `c`; `quantile_value` in `dummy.rs` sorts, then linear-interpolates at `pos = q*(n-1)` (numpy's default `linear` percentile). impl `Predict … for FittedDummyRegressor` returns `Array1::from_elem(x.nrows(), self.constant)`. Mirrors sklearn `np.average`/`np.median`/`np.percentile`/constant (`sklearn/dummy.py:581-628`) tiled (`:655-664`). DETERMINISTIC / oracle-pinnable: `y=[1,2,3,4,5]` → `mean`/`median` 3.0, `quantile(0.25)` 2.0; even `y=[1,2,3,4]` → `median` 2.5 (all live-oracle, Verification). Tests: `dummy_regressor_mean`, `dummy_regressor_median`, `dummy_regressor_quantile_25`, `dummy_regressor_constant`. CAVEAT: existing tests assert hand-computed values; the critic should re-anchor median(even-n) and quantile(linear-interp) to the live-oracle numbers above (R-CHAR-3). Non-test consumer: REQ-11. |
| REQ-7 (regressor quantile/constant validation) | SHIPPED | `quantile_value` in `dummy.rs` returns `FerroError::InvalidParameter { name: "quantile", reason: "must be in [0,1], got {q}" }` when `!(0.0..=1.0).contains(&q)`, matching sklearn's `Interval(Real,0.0,1.0,closed="both")` (`sklearn/dummy.py:533`; live oracle `quantile=1.5` → `InvalidParameterError`, Verification). DIVERGENCE / structurally-N-A (honest underclaim): sklearn additionally raises `ValueError` when `strategy="quantile"` has `quantile is None` (`:593-598`) and `TypeError` when `strategy="constant"` has `constant is None` (`:608-613`). ferrolearn's `DummyRegressorStrategy::Quantile(f64)` and `Constant(F)` carry their payload by construction — the `None` states are type-eliminated, so those two errors are unreachable-by-design rather than ported. SHIPPED on the range check; the missing-payload errors are N/A under the enum encoding (R-DEV-4-adjacent: Rust's type system eliminates the footgun). Test: `quantile_value` range path exercised indirectly. Non-test consumer: REQ-11. |
| REQ-8 (sample_weight) | NOT-STARTED | open prereq blocker (tracking #1690). `DummyClassifier::fit`/`DummyRegressor::fit` signatures are `fit(&self, _x, y)` — no `sample_weight` argument. sklearn folds weights into `class_distribution` (`sklearn/dummy.py:211-212`, `:228-230`), `np.average(..., weights=sample_weight)` (`:582`), and `_weighted_percentile` for median/quantile (`:588-606`). Absent end-to-end. |
| REQ-9 (multi-output y) | NOT-STARTED | open prereq blocker (tracking #1690). `impl Fit<Array2<F>, Array1<usize>>` (classifier) / `impl Fit<Array2<F>, Array1<F>>` (regressor) with `Output = Array1<…>` are single-output only; no `n_outputs_`, no 1D→2D reshape, no `(n_samples, n_outputs)` predict. sklearn (both `MultiOutputMixin`) reshapes 1D→2D and tracks `n_outputs_` (`sklearn/dummy.py:204-207`, `:574`, `:655-664`). Absent. |
| REQ-10 (ferray substrate) | NOT-STARTED | open prereq blocker (tracking #1690; R-SUBSTRATE-2/3). `dummy.rs` uses `ndarray::{Array1, Array2}` and `rand::rngs::SmallRng`/`rand::prelude::IndexedRandom`. Destination substrate is the `ferray-core` array type and `ferray::random` (R-SUBSTRATE-1). Not migrated; the `SmallRng` usage is also the proximate cause of the REQ-3 RNG carve-out. |
| REQ-11 (consumer) | SHIPPED | Crate re-export: `lib.rs` (`pub use dummy::{DummyClassifier, DummyClassifierStrategy, DummyRegressor, DummyRegressorStrategy, FittedDummyClassifier, FittedDummyRegressor}`) and `pub mod dummy`. R-DEFER-1 / S5: boundary baseline-estimator types ARE the public API; existing pub surface grandfathered. CAVEAT (honest underclaim): grep finds NO non-test, non-re-export caller and NO `ferrolearn-python` binding for either estimator (the only other `Dummy*` matches are an unrelated `DummyEstimator` test fixture in `ferrolearn-core/src/traits.rs` and a doc-comment in `calibration.rs`). SHIPPED on the boundary re-export per S5, not a dedicated production caller; the narrower-than-sklearn surface (no Python binding) is noted. |

## Architecture

ferrolearn splits each estimator into an unfitted/Fitted pair (CLAUDE.md naming).
`DummyClassifier { strategy: DummyClassifierStrategy, random_state: Option<u64> }`
fits to `FittedDummyClassifier { strategy, classes: Vec<usize>, class_priors:
Vec<f64>, random_state }`; `DummyRegressor<F> { strategy:
DummyRegressorStrategy<F> }` fits to `FittedDummyRegressor<F> { constant: F }`.
sklearn keeps one class each whose post-`fit` state is `classes_`/`class_prior_`
or `constant_` (`sklearn/dummy.py:228-249`, `:627`).

Strategies are encoded as Rust enums, not sklearn's `strategy=` string:
`DummyClassifierStrategy::{MostFrequent, Prior, Stratified, Uniform,
Constant(usize)}` (default `Prior` via `Default`, matching
`sklearn/dummy.py:157` `strategy="prior"`) and `DummyRegressorStrategy::{Mean,
Median, Quantile(f64), Constant(F)}` (default `Mean`, matching `:541`
`strategy="mean"`). The enum payloads (`Constant(usize)`, `Quantile(f64)`,
`Constant(F)`) carry what sklearn keeps in the separate `constant`/`quantile`
constructor parameters; this type-eliminates the "constant/quantile is `None`
for the wrong strategy" error paths (REQ-7).

The deterministic core: `DummyClassifier::fit` counts labels into a `HashMap`,
sorts the unique classes ascending, and stores per-class priors; `most_frequent()`
returns the first argmax (matching sklearn's `argmax()` first-max tie-break,
`sklearn/dummy.py:311`). `DummyRegressor::fit` reduces `y` to a single constant
through `quantile_value` (sort + `pos = q*(n-1)` linear interpolation, numpy's
default percentile method). Both `predict` impls broadcast the stored value over
`x.nrows()` via `Array1::from_elem`, mirroring sklearn's `np.tile`/`np.full`
(`:309`, `:655`).

The stochastic strategies (`Stratified`, `Uniform`) run on `rand`'s `SmallRng`
(seeded by `make_rng` = `SmallRng::seed_from_u64(seed + salt)`), with
`weighted_choice` doing a cumulative-prior inversion and `Uniform` using
`slice::choose`. This is observably correct DISTRIBUTIONALLY but cannot reproduce
numpy `RandomState`'s exact stream (REQ-3 carve-out; also the REQ-10 substrate
gap).

What is structurally absent vs sklearn: `predict_proba`/`predict_log_proba` and
the `class_prior_` accessor (REQ-2/REQ-5, `sklearn/dummy.py:340-423`),
exact-parity `stratified`/`uniform` (REQ-3, RNG carve-out), `sample_weight`
(REQ-8, `:211-212`, `:582-606`), multi-output `y` (REQ-9, `:204-207`,
`:655-664`), and the `ferray` substrate (REQ-10). The classifier
constant-membership rejection IS shipped (REQ-4), with the `FerroError` variant
diverging from `ValueError`.

Invariants: `classes` is always sorted ascending; `class_priors` sums to 1 over a
non-empty `y` (empty `y` → `FerroError::InsufficientSamples`); `Constant(c)` for
the classifier requires `c ∈ classes`; `Quantile(q)` requires `q ∈ [0,1]`;
predictions have the same row count as `X`.

## Verification

Commands establishing the SHIPPED claims (baseline
`d8c7a47fef52baf299f0c19cf3d0c8b753487796`):

- `cargo test -p ferrolearn-model-sel --lib dummy` → the 9 `dummy_*` tests pass
  (REQ-1 deterministic predict; REQ-4 constant-membership rejection; REQ-5
  `n_classes`; REQ-6 regressor value parity; REQ-7 quantile range).
- REQ-1 / REQ-5 oracle (deterministic classifier predict + attributes, live oracle):
  ```
  python3 -c "import numpy as np; from sklearn.dummy import DummyClassifier; \
  X=np.zeros((5,2)); y=np.array([0,0,1,1,2]); \
  c=DummyClassifier(strategy='most_frequent').fit(X,y); \
  print(c.predict(np.zeros((3,2))).tolist(), c.classes_.tolist(), c.class_prior_.tolist())"
  # -> [0, 0, 0] [0, 1, 2] [0.4, 0.4, 0.2]   (prior strategy: identical predict)
  ```
  Pin a tie-break `#[test]` on `y=[0,0,1,1,2]` expecting `[0,0,0]` and
  `classes()==[0,1,2]` (R-CHAR-3: from the live oracle, never copied from
  ferrolearn).
- REQ-2 oracle (the proba surface the gap omits, live oracle):
  ```
  python3 -c "import numpy as np; from sklearn.dummy import DummyClassifier; \
  X=np.zeros((5,2)); y=np.array([0,0,1,1,2]); \
  print(DummyClassifier(strategy='prior').fit(X,y).predict_proba(np.zeros((2,2))).tolist()); \
  print(DummyClassifier(strategy='most_frequent').fit(X,y).predict_proba(np.zeros((2,2))).tolist()); \
  print(DummyClassifier(strategy='uniform').fit(X,y).predict_proba(np.zeros((1,2))).tolist())"
  # -> prior [[0.4,0.4,0.2],[0.4,0.4,0.2]]; most_frequent [[1.0,0.0,0.0],[1.0,0.0,0.0]]; uniform [[1/3,1/3,1/3]]
  ```
  ferrolearn has no `predict_proba` — REQ-2 NOT-STARTED.
- REQ-4 oracle (classifier constant membership, live oracle):
  `DummyClassifier(strategy='constant',constant=99).fit(X,y)` → `ValueError`.
  ferrolearn `fit` → `FerroError::InvalidParameter` (`dummy_classifier_constant_invalid`).
- REQ-6 oracle (regressor value parity, live oracle):
  ```
  python3 -c "import numpy as np; from sklearn.dummy import DummyRegressor; \
  X=np.zeros((5,2)); y=np.array([1.,2.,3.,4.,5.]); \
  print(DummyRegressor(strategy='mean').fit(X,y).predict(np.zeros((3,2)))[0], \
        DummyRegressor(strategy='median').fit(X,y).predict(np.zeros((1,2)))[0], \
        DummyRegressor(strategy='quantile',quantile=0.25).fit(X,y).predict(np.zeros((1,2)))[0]); \
  print(DummyRegressor(strategy='median').fit(np.zeros((4,2)),np.array([1.,2.,3.,4.])).predict(np.zeros((1,2)))[0])"
  # -> 3.0 3.0 2.0  (and even-n median) 2.5
  ```
- REQ-7 oracle (regressor quantile range, live oracle):
  `DummyRegressor(strategy='quantile',quantile=1.5).fit(X,y)` →
  `InvalidParameterError`; ferrolearn `quantile_value` → `FerroError::InvalidParameter`.
  (sklearn's missing-quantile `ValueError` / missing-constant `TypeError`
  (`sklearn/dummy.py:595`, `:610`) are unreachable under the enum encoding.)
- REQ-8 oracle (the weighted path the gap omits, live oracle):
  `DummyRegressor(strategy='mean').fit(X,y,sample_weight=[3,1,1,1,1]).predict(...)`
  gives the weighted mean; ferrolearn `fit` has no `sample_weight` — REQ-8 NOT-STARTED.

Commands that establish the NOT-STARTED REQs are absent: no `predict_proba`
method (REQ-2), no numpy-stream parity for `stratified`/`uniform` (REQ-3,
carve-out), no `sample_weight` parameter (REQ-8), no 2D-`y` `Output` (REQ-9), no
`ferray-core`/`ferray::random` usage (REQ-10). Per R-DEFER-2 the table is binary
SHIPPED/NOT-STARTED.

SHIPPED: REQ-1 (deterministic classifier predict + first-max tie-break),
REQ-4 (classifier constant-membership rejection; `FerroError` variant diverges
from `ValueError`), REQ-5 (`classes_`/`n_classes_` via `HasClasses`;
`class_prior_` accessor gap noted), REQ-6 (regressor mean/median/quantile/constant
value parity), REQ-7 (regressor quantile range check; missing-payload errors N/A
under the enum), REQ-11 (boundary re-export consumer; no dedicated non-test
caller, no Python binding — honest underclaim). NOT-STARTED (tracking #1690; the
critic files per-REQ blockers): REQ-2 (`predict_proba`/`predict_log_proba` +
`class_prior_` exposure), REQ-3 (`stratified`/`uniform` exact parity — RNG
carve-out, no failing test), REQ-8 (`sample_weight`), REQ-9 (multi-output),
REQ-10 (ferray substrate).
