# Toy datasets (`load_*`)

<!--
tier: 3-component
status: draft
baseline-commit: 8f4439d251b0058ed419c4070766e0beca462317
upstream-paths:
  - sklearn/datasets/_base.py
-->

## Summary

`ferrolearn-datasets/src/toy.rs` mirrors scikit-learn's bundled toy-dataset
loaders (`sklearn/datasets/_base.py`): `load_iris`, `load_wine`,
`load_breast_cancer`, `load_diabetes`, `load_digits`, `load_linnerud`. Each
ferrolearn loader returns a bare `(X, y)` tuple, equivalent to sklearn's
`return_X_y=True` path. sklearn ships the real data under
`sklearn/datasets/data/` (`iris.csv`, `wine_data.csv`, `breast_cancer.csv`,
`diabetes_data_raw.csv.gz`, `diabetes_target.csv.gz`, `digits.csv.gz`,
`linnerud_exercise.csv`, `linnerud_physiological.csv`).

This is a **data-parity** contract: a loader is SHIPPED only when its returned
values match the live sklearn 1.5.2 oracle element-wise. The current code embeds
the real data for **iris** and **linnerud** only; **wine, breast_cancer,
diabetes, digits** return synthetic random data with the correct shape but wrong
values, and **digits** is additionally truncated to 200 of 1797 samples. A
loader that returns random data with the correct shape is NOT-STARTED for value
parity (R-HONEST-1/3) — shape correctness is not value parity.

`load_olivetti_faces` is also present but has **no toy-loader analog** in
`_base.py`: upstream it is `fetch_olivetti_faces` (network download/cache, in
`sklearn/datasets/_olivetti_faces.py`), out of scope for this translation unit.

Upstream reference: scikit-learn 1.5.2 (commit 156ef14),
`sklearn/datasets/_base.py` — `load_iris` (:620), `load_wine` (:496),
`load_breast_cancer` (:750), `load_digits` (:907), `load_diabetes` (:1044),
`load_linnerud` (:1171).

## Requirements

- REQ-1 (iris parity): `load_iris` returns `(X, y)` matching
  `sklearn.datasets.load_iris(return_X_y=True)` element-wise — `X` shape
  `(150, 4)`, `y` 3 classes (50 each), labels 0=setosa/1=versicolor/2=virginica.
- REQ-2 (linnerud parity): `load_linnerud` returns multi-output `(X, y)` matching
  `sklearn.datasets.load_linnerud(return_X_y=True)` — `X` (exercise) shape
  `(20, 3)`, `y` (physiological) shape `(20, 3)`, element-wise.
- REQ-3 (wine parity): `load_wine` returns `(X, y)` matching
  `sklearn.datasets.load_wine(return_X_y=True)` — shape `(178, 13)`, 3 classes,
  real wine measurements, element-wise.
- REQ-4 (breast_cancer parity): `load_breast_cancer` returns `(X, y)` matching
  `sklearn.datasets.load_breast_cancer(return_X_y=True)` — shape `(569, 30)`,
  2 classes (0=malignant/1=benign), element-wise.
- REQ-5 (diabetes parity, scaled default): `load_diabetes` returns `(X, y)`
  matching `sklearn.datasets.load_diabetes(return_X_y=True)` whose default
  `scaled=True` mean-centers and scales each feature by `std * sqrt(n_samples)`
  (`_base.py:1132-1134`) — shape `(442, 10)`, continuous target 25–346,
  element-wise.
- REQ-6 (digits full + n_class): `load_digits` returns all 1797 samples (not
  200) matching `sklearn.datasets.load_digits(return_X_y=True)` — shape
  `(1797, 64)`, 10 classes — with the upstream `n_class=10` default param
  (`_base.py:907`) selecting how many leading digit classes are returned.
- REQ-7 (Bunch metadata contract): expose the upstream `Bunch` attributes
  (`feature_names`, `target_names`, `DESCR`, `frame`, `data_filename`) and the
  `return_X_y` switch. ferrolearn returns bare tuples only.
- REQ-8 (olivetti): provide the real Olivetti faces dataset
  (`fetch_olivetti_faces`, `_olivetti_faces.py`), which requires a fetch+cache
  mechanism. The current synthetic stub is a documented deviation.
- REQ-9 (ferray substrate): toy loaders compute on the ferray array substrate
  (`ferray-core`) per R-SUBSTRATE-1/2, not `ndarray`.
- REQ-10 (production consumer): a non-test production consumer of the loaders
  exists (the loaders are existing boundary pub APIs).

## Acceptance criteria

- AC-1 (REQ-1): `load_iris::<f64>()` `X` equals `load_iris(return_X_y=True)[0]`
  with max abs diff `0.0` over all 150×4 cells; `y` array-equal; class counts
  50/50/50. (Verified: embedded `data/iris.csv` vs oracle — max abs diff `0.0`.)
- AC-2 (REQ-2): `load_linnerud::<f64>()` `X` and `y` each equal the oracle 20×3
  matrices element-wise; `X[0]==[5,162,60]`, `y[0]==[191,36,50]`.
- AC-3 (REQ-3): `load_wine::<f64>()` `X[0]` equals
  `[14.23,1.71,2.43,15.6,127.0,2.8,3.06,0.28,2.29,5.64,1.04,3.92,1065.0]`
  (live oracle). FAILS today: random data.
- AC-4 (REQ-4): `load_breast_cancer::<f64>()` `X[0,:3]` equals
  `[17.99,10.38,122.8]` (live oracle). FAILS today: random data.
- AC-5 (REQ-5): `load_diabetes::<f64>()` `X[0,:3]` equals
  `[0.038075906433423026,0.05068011873981862,0.061696206518683294]` and
  `y[:3]==[151.0,75.0,141.0]` (live oracle, default `scaled=True`). FAILS today:
  random data, no scaling transform.
- AC-6 (REQ-6): `load_digits::<f64>()` `X.shape==(1797,64)` and `X[0,:8]` equals
  `[0,0,5,13,9,1,0,0]` (live oracle). FAILS today: shape `(200,64)`, random data.
- AC-7 (REQ-7): a `Bunch`-equivalent return path carries `feature_names` /
  `target_names` / `DESCR`. No such path exists today.
- AC-8 (REQ-8): a real Olivetti loader sourced from the AT&T data, not a seeded
  RNG. No such path exists today.
- AC-9 (REQ-9): toy.rs imports `ferray-core` array types, not `ndarray`.
- AC-10 (REQ-10): a non-test caller of a `load_*` symbol exists in workspace
  production code (not under `#[cfg(test)]` / `tests/`).

## REQ status

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (iris parity) | SHIPPED | impl `pub fn load_iris in toy.rs` → `parse_classification_csv(IRIS_CSV, 150, 4)`, where `const IRIS_CSV` embeds `data/iris.csv` via `include_str!`. Mirrors `sklearn/datasets/_base.py:620` (`load_iris`). Oracle parity: live `load_iris(return_X_y=True)` X vs embedded CSV — `X max abs diff 0.0`, `y array_equal True`, 150 rows, class counts 50/50/50. Pinned by `#[test] test_load_iris_first_row`/`test_load_iris_last_row`/`test_load_iris_class_balance` (expected values are the live-oracle iris rows). Verification: `cargo test -p ferrolearn-datasets --lib toy` → 21 passed, 0 failed. |
| REQ-2 (linnerud parity) | SHIPPED | impl `pub fn load_linnerud in toy.rs` reads `const LINNERUD_FEATURES` (20×3 exercise) and `const LINNERUD_TARGETS` (20×3 physiological) into multi-output `(Array2, Array2)`. Mirrors `sklearn/datasets/_base.py:1171` (`load_linnerud`). Oracle parity: live `load_linnerud(return_X_y=True)` → `X[0]==[5.0,162.0,60.0]`, `y[0]==[191.0,36.0,50.0]`, shapes `(20,3)`/`(20,3)` — match the embedded consts. Pinned by `#[test] test_load_linnerud_first_row`/`test_load_linnerud_last_row`. Verification: same `cargo test` run. |
| REQ-3 (wine parity) | NOT-STARTED | `pub fn load_wine in toy.rs` returns `synthetic_classification(178, 13, 3)` — deterministic offset-blob random data, correct shape `(178,13)`, WRONG values. Live oracle `load_wine(return_X_y=True)[0][0] == [14.23,1.71,2.43,15.6,127.0,2.8,3.06,0.28,2.29,5.64,1.04,3.92,1065.0]`; ferrolearn produces `class*5 + idx*0.001`. Prereq: embed `sklearn/datasets/data/wine_data.csv` and parse it; match the live oracle element-wise. Open prereq blocker: critic to file (tracking #1652). Deterministic, oracle-pinnable. |
| REQ-4 (breast_cancer parity) | NOT-STARTED | `pub fn load_breast_cancer in toy.rs` returns `synthetic_classification(569, 30, 2)` — random, correct shape `(569,30)`, WRONG values. Live oracle `X[0,:3]==[17.99,10.38,122.8]`. Prereq: embed `sklearn/datasets/data/breast_cancer.csv` and parse it; match the live oracle element-wise. Open prereq blocker: critic to file (tracking #1652). Deterministic, oracle-pinnable. |
| REQ-5 (diabetes parity, scaled default) | NOT-STARTED | `pub fn load_diabetes in toy.rs` returns `synthetic_regression(442, 10)` — random, correct shape `(442,10)`, WRONG values, and NO scaling transform. sklearn's default `scaled=True` (`_base.py:1132-1134`) mean-centers then divides by `std * sqrt(n_samples)`; live oracle `X[0,:3]==[0.038075906433423026,0.05068011873981862,0.061696206518683294]`, `y[:3]==[151.0,75.0,141.0]`. Prereq: embed `sklearn/datasets/data/diabetes_data_raw.csv.gz` + `diabetes_target.csv.gz`, gunzip+parse, apply the `scale(...)/sqrt(n_samples)` transform; match the live oracle element-wise. Open prereq blocker: critic to file (tracking #1652). Deterministic, oracle-pinnable. |
| REQ-6 (digits full + n_class) | NOT-STARTED | `pub fn load_digits in toy.rs` returns `synthetic_classification(200, 64, 10)` — TRUNCATED to 200 samples (not 1797) AND random values. Live oracle `load_digits(return_X_y=True)` is shape `(1797,64)` with `X[0,:8]==[0,0,5,13,9,1,0,0]`; upstream signature has `n_class=10` default (`_base.py:907`). Prereq: embed `sklearn/datasets/data/digits.csv.gz`, gunzip+parse all 1797 rows, add the `n_class` selection param; match the live oracle element-wise. Open prereq blocker: critic to file (tracking #1652). Deterministic, oracle-pinnable. |
| REQ-7 (Bunch metadata contract) | NOT-STARTED | All loaders return bare `(X, y)` tuples (the `return_X_y=True` shape) — see signatures `pub fn load_iris`/`load_wine`/... `-> Result<(Array2<F>, Array1<usize>), FerroError>`. No `Bunch`-equivalent carrying `feature_names`/`target_names`/`DESCR`/`frame`/`data_filename` (`_base.py` `Bunch(...)` returns, e.g. `:1152`), and no `return_X_y` switch. Prereq: define a `Bunch`-equivalent return type and the `feature_names`/`target_names`/`DESCR` payloads per loader. Open prereq blocker: critic to file (tracking #1652). |
| REQ-8 (olivetti real dataset) | NOT-STARTED | `pub fn load_olivetti_faces in toy.rs` generates seeded-RNG synthetic 64×64 "faces" (`Xoshiro256PlusPlus::seed_from_u64(0xFACE_DA7A)`), explicitly "**not** the real Olivetti dataset". Upstream has NO toy loader for this — it is `fetch_olivetti_faces` in `sklearn/datasets/_olivetti_faces.py` (network download + on-disk cache), out of scope for `_base.py`/this unit. Prereq: a fetch+cache mechanism (separate fetch module) sourcing the real AT&T faces. Open prereq blocker: critic to file (tracking #1652). Not a toy value-parity REQ. |
| REQ-9 (ferray substrate) | NOT-STARTED | `toy.rs` uses `use ndarray::{Array1, Array2}` and `rand`/`rand_xoshiro` — the wrong substrate per R-SUBSTRATE-1. Loaders return `ndarray::Array2`/`Array1`, not `ferray-core` array types. Prereq: migrate the loader return types and parsing to `ferray-core` (bridge via `into_ndarray()` at the boundary during transition). Open prereq blocker: critic to file (tracking #1652). |
| REQ-10 (production consumer) | NOT-STARTED | No non-test production consumer: the only callers of `load_*` outside `toy.rs` are the lib.rs re-exports (`pub use toy::{...}` in `ferrolearn-datasets/src/lib.rs`) and `ferrolearn/tests/integration_tests.rs` (a `tests/` integration target — test-only). The loaders are existing grandfathered boundary pub APIs (R-DEFER-1/S5: their consumers are external users + the future `ferrolearn-python` binding), so this does not demote REQ-1/REQ-2; but no in-workspace production consumer exists today. Prereq: register the toy loaders in `ferrolearn-python` (the `import ferrolearn.datasets` surface). Open prereq blocker: critic to file (tracking #1652). |

## Architecture

Two loader families exist in `toy.rs`:

- **Real-data loaders (iris, linnerud).** `load_iris` calls
  `parse_classification_csv(IRIS_CSV, 150, 4)`, where `const IRIS_CSV: &str =
  include_str!("../data/iris.csv")` compiles in the canonical UCI Iris CSV
  (header row + 150 data rows, last column = integer label). The parser
  validates the `(150, 4)` shape and 3-class structure. `load_linnerud` reads
  two `[[f64; 3]; 20]` `const` arrays (`LINNERUD_FEATURES`,
  `LINNERUD_TARGETS`) into a multi-output `(Array2<F>, Array2<F>)` — the only
  loader whose `y` is 2-D, matching sklearn's multi-output Linnerud
  (`_base.py:1171`, exercise→X, physiological→y). Both reproduce the live
  oracle element-wise.

- **Synthetic-stub loaders (wine, breast_cancer, diabetes, digits, olivetti).**
  `synthetic_classification` and `synthetic_regression` emit deterministic but
  fabricated values (`class*5 + (i*n_feat+j)*0.001`), giving the right shape and
  class cardinality but values unrelated to the real datasets. `load_digits`
  additionally truncates to 200 of 1797 samples. These are placeholders, not
  translations: the upstream loaders read the bundled CSV/gz under
  `sklearn/datasets/data/`.

Key upstream behaviors to mirror when the synthetic loaders are replaced:
- **diabetes default scaling** (`_base.py:1132-1134`): `data = scale(data,
  copy=False); data /= data.shape[0] ** 0.5` — per-column mean-center, divide by
  population std, then divide by `sqrt(442)`. The default `scaled=True` path is
  the contract; `scaled=False` returns raw (`X[0,:3]==[59.0,2.0,32.1]`).
- **digits `n_class`** (`_base.py:907`, `def load_digits(*, n_class=10, ...)`):
  the param trims to the first `n_class` digit classes; the default returns all
  10 classes / 1797 samples.

`FittedLinearRegression`-style fitted/unfitted split does not apply here — these
are free functions, not estimators. The crate-level public API
(`ferrolearn-datasets/src/lib.rs` re-exports) is the boundary surface.

## Verification

Commands that establish the SHIPPED claims (REQ-1, REQ-2):

```bash
cargo test -p ferrolearn-datasets --lib toy   # 21 passed, 0 failed (baseline 8f4439d)
```

Live sklearn-oracle comparisons used to ground the REQ table (R-CHAR-3 — expected
values are the live oracle, never copied from ferrolearn):

```bash
# REQ-1 iris: embedded CSV vs oracle, element-wise — confirmed max abs diff 0.0, y array-equal
python3 -c "from sklearn.datasets import load_iris; import numpy as np; \
X,y=load_iris(return_X_y=True); print(X.shape, np.unique(y, return_counts=True))"
# REQ-2 linnerud: X[0]==[5,162,60], y[0]==[191,36,50]
python3 -c "from sklearn.datasets import load_linnerud; \
X,y=load_linnerud(return_X_y=True); print(X[0].tolist(), y[0].tolist())"
# REQ-3..6 (NOT-STARTED) the critic pins FAILING #[test]s comparing load_*()
# to these oracle values:
python3 -c "from sklearn.datasets import load_wine, load_breast_cancer, load_diabetes, load_digits; \
print(load_wine(return_X_y=True)[0][0].tolist()); \
print(load_breast_cancer(return_X_y=True)[0][0,:3].tolist()); \
print(load_diabetes(return_X_y=True)[0][0,:3].tolist(), load_diabetes(return_X_y=True)[1][:3].tolist()); \
print(load_digits(return_X_y=True)[0].shape, load_digits(return_X_y=True)[0][0,:8].tolist())"
```

REQ-3, REQ-4, REQ-5, REQ-6 are deterministic and oracle-pinnable: the critic must
add failing `#[test]`s comparing `load_wine`/`load_breast_cancer`/`load_diabetes`/
`load_digits` to the live `sklearn.datasets.load_*(return_X_y=True)` values
above; they fail today because the loaders return synthetic data. The shared
prerequisite for all four is: embed the real sklearn-bundled dataset
(`sklearn/datasets/data/<file>`) and parse it (gunzip for the `.csv.gz` files),
matching the live oracle element-wise.
