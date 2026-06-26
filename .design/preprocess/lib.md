# ferrolearn-preprocess crate root (re-export boundary)

<!--
tier: 3-component
status: shipped-partial
baseline-commit: 51f60dc0
upstream: scikit-learn 1.5.2
upstream-paths:
  - sklearn/preprocessing/__init__.py     # __all__ (:30-60)
  - sklearn/feature_selection/__init__.py  # __all__ (:27-47)
  - sklearn/feature_extraction/__init__.py # __all__ (DictVectorizer, FeatureHasher, image/text helpers)
  - sklearn/feature_extraction/_hash.py    # class FeatureHasher
  - sklearn/feature_extraction/text.py     # __all__ (:34-43)
  - sklearn/feature_extraction/image.py    # __all__ (:23-26); scoped dense helper surface
  - sklearn/impute/__init__.py             # __all__ (:13)
  - sklearn/random_projection.py           # __all__ (:50-54)
  - sklearn/compose/__init__.py            # __all__ (:15-20)
ferrolearn-module: ferrolearn-preprocess/src/lib.rs
parity-ops: re-export boundary
crosslink-issue: 1361
-->

## Summary

`ferrolearn-preprocess/src/lib.rs` is the crate's **public-API surface** — the
analog of the `__all__` re-export boundary of the covered scikit-learn modules plus
a scoped dense `sklearn.feature_extraction` / `feature_extraction.image` helper surface
(`sklearn.preprocessing`, `sklearn.feature_selection`,
`sklearn.feature_extraction`, `sklearn.feature_extraction.text`, `sklearn.impute`,
`sklearn.random_projection`, `sklearn.compose`, plus `grid_to_graph`/`img_to_graph`/patch helpers) collapsed under one crate root. It is **not** an estimator: it
owns no `fit`/`transform` logic. Two things live in this translation unit:

1. **Module declarations** — the `pub mod` block for the public submodules
   (one per transformer / selector / encoder / imputer family, plus the shared
   `feature_scoring`, `image`, and `stat_selectors` helpers).
2. **The re-export block** — the crate-root `pub use` blocks that surface each
   implemented estimator's unfitted + `Fitted*` type pair (plus supporting enums
   like `ImputeStrategy`, `BinStrategy`, `OutputDistribution`, `KnotStrategy`,
   `ThresholdStrategy`, `Direction`, `TfidfNorm`, `MaxPatches`, `NComponents`,
   and the `chi2`/`f_classif`/`f_regression`/`r_regression`/JL/image free
   functions) at the crate root.

**This doc covers the boundary only.** Per-estimator value parity — each
transformer's `fit`/`transform` matching its sklearn analog to ULPs — lives in the
sibling routed module docs: `.design/preprocess/{standard_scaler, min_max_scaler,
max_abs_scaler, robust_scaler, normalizer, power_transformer, quantile_transformer,
binarizer, function_transformer, polynomial_features, one_hot_encoder,
ordinal_encoder, label_encoder, label_binarizer, multi_label_binarizer,
kbins_discretizer, target_encoder, spline_transformer, knn_imputer,
select_from_model, select_percentile, rfe, sequential_feature_selector,
count_vectorizer, tfidf, ...}.md`. This doc does **not** re-litigate per-estimator
parity; it covers only (a) the re-export surface mirroring the covered sklearn
`__all__` surfaces, and (b) the ferray-substrate gap.

## Probes (live sklearn 1.5.2 oracle)

The PRESENT/ABSENT accounting below is anchored to the **live** `__all__` of each
mirrored module (R-CHAR-3 — the boundary is checked against the running sklearn,
not a transcribed list). Run from `/tmp` (outside the workspace):

```bash
python3 -c "import sklearn; print(sklearn.__version__)"   # 1.5.2
python3 -c "import sklearn.preprocessing as p;       print(sorted(p.__all__))"
python3 -c "import sklearn.feature_selection as f;    print(sorted(f.__all__))"
python3 -c "import sklearn.impute as i;               print(sorted(i.__all__))"
python3 -c "import sklearn.random_projection as r;    print(sorted(r.__all__))"
python3 -c "import sklearn.compose as c;              print(sorted(c.__all__))"
python3 -c "import sklearn.feature_extraction as fe;  print(sorted(fe.__all__))"
python3 -c "import sklearn.feature_extraction.text as t; print(sorted(n for n in dir(t) if n[0].isupper()))"
python3 -c "import sklearn.feature_extraction.image as im; print(sorted(im.__all__))"
```

Observed (sklearn 1.5.2):

- `preprocessing.__all__` = `['Binarizer', 'FunctionTransformer',
  'KBinsDiscretizer', 'KernelCenterer', 'LabelBinarizer', 'LabelEncoder',
  'MaxAbsScaler', 'MinMaxScaler', 'MultiLabelBinarizer', 'Normalizer',
  'OneHotEncoder', 'OrdinalEncoder', 'PolynomialFeatures', 'PowerTransformer',
  'QuantileTransformer', 'RobustScaler', 'SplineTransformer', 'StandardScaler',
  'TargetEncoder', 'add_dummy_feature', 'binarize', 'label_binarize',
  'maxabs_scale', 'minmax_scale', 'normalize', 'power_transform',
  'quantile_transform', 'robust_scale', 'scale']`.
- `feature_selection.__all__` = `['GenericUnivariateSelect', 'RFE', 'RFECV',
  'SelectFdr', 'SelectFpr', 'SelectFromModel', 'SelectFwe', 'SelectKBest',
  'SelectPercentile', 'SelectorMixin', 'SequentialFeatureSelector',
  'VarianceThreshold', 'chi2', 'f_classif', 'f_oneway', 'f_regression',
  'mutual_info_classif', 'mutual_info_regression', 'r_regression']`.
- `impute.__all__` = `['MissingIndicator', 'SimpleImputer', 'KNNImputer']`
  (literal at `impute/__init__.py:13`). `IterativeImputer` is namespaced into
  `sklearn.impute` only after `from sklearn.experimental import
  enable_iterative_imputer` (still experimental in 1.5.2), so it is reachable as
  `sklearn.impute.IterativeImputer` but is **not** in the literal `__all__` list.
- `random_projection.__all__` = `['GaussianRandomProjection',
  'SparseRandomProjection', 'johnson_lindenstrauss_min_dim']`.
- `compose.__all__` = `['ColumnTransformer', 'TransformedTargetRegressor',
  'make_column_selector', 'make_column_transformer']`.
- `feature_extraction.__all__` = `['DictVectorizer', 'FeatureHasher',
  'grid_to_graph', 'image', 'img_to_graph', 'text']`; ferrolearn ships
  `DictVectorizer`, `FeatureHasher`, `grid_to_graph`, and `img_to_graph` at the crate root.
- `feature_extraction.text` public classes = `CountVectorizer`,
  `HashingVectorizer`, `TfidfTransformer`, `TfidfVectorizer` (`text.py:34-43`).
- `feature_extraction.image.__all__` = `['PatchExtractor',
  'extract_patches_2d', 'grid_to_graph', 'img_to_graph',
  'reconstruct_from_patches_2d']`; ferrolearn ships all five names as scoped
  dense Rust helpers (`grid_to_graph`, `img_to_graph`, `extract_patches_2d`,
  `reconstruct_from_patches_2d`, `PatchExtractor`).

Boundary integrity (the re-export surface resolves):

```bash
cargo build -p ferrolearn-preprocess     # Finished — every pub use resolves
cargo doc  -p ferrolearn-preprocess --no-deps
```

A passing `cargo build` is the load-bearing check for REQ-1: every name in the
crate-root `pub use` blocks must name a type that actually exists in its
submodule, or the crate fails to compile.

## Requirements

- REQ-1 (re-export boundary): the crate root surfaces, via its `pub use` blocks,
  exactly the estimators ferrolearn implements that mirror the covered sklearn
  `__all__` surfaces — each as an unfitted + `Fitted*` pair per the project naming
  convention (CLAUDE.md) — plus the supporting enums and the
  `chi2`/`f_classif`/`f_regression`/`r_regression` scoring functions,
  `johnson_lindenstrauss_min_dim`, and the scoped image helpers. The surfaced set is a
  documented subset of the union of the six `__all__` lists plus image helpers (the boundary ships
  exactly what is implemented; the not-yet-translated names are enumerated in
  Architecture). It has non-test production consumers: the meta-crate re-export
  alias and the `ferrolearn-python` PyO3 pyclasses.
- REQ-substrate (ferray): `lib.rs` and every submodule it declares are on the
  ferray array substrate (`ferray-core` for `Array1`/`Array2`, `ferray-ufunc` for
  elementwise ops) rather than `ndarray` + `num_traits`, per R-SUBSTRATE-1.

## Acceptance criteria

- AC-1 (REQ-1): `cargo build -p ferrolearn-preprocess` succeeds — every name in
  the crate-root `pub use` blocks resolves to an existing type/function. Each
  re-exported estimator is reachable as `ferrolearn_preprocess::<Type>` and is
  routed by its own `.design/preprocess/<doc>.md`. The surfaced estimator set is a
  subset of the union of the six `__all__` lists plus the image helper surface above; every PRESENT entry in the
  Architecture mapping resolves, and no ABSENT entry is claimed re-exported.
- AC-2 (REQ-1, consumers): `cargo build -p ferrolearn` (meta-crate) and
  `cargo build -p ferrolearn-python` succeed against the boundary — i.e. the
  `pub use ferrolearn_preprocess as preprocess;` alias and the eight PyO3 scaler /
  transformer pyclasses all link against the re-exported types.
- AC-substrate: `grep -nE "ndarray|num-traits" ferrolearn-preprocess/Cargo.toml`
  returns nothing (the crate would depend on `ferray-core` / `ferray-ufunc`, not
  `ndarray` / `num-traits`). Currently it returns matches — AC-substrate is unmet.

## `## REQ status`

## REQ status

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (re-export boundary — covered `__all__` lists plus image helper surface) | SHIPPED | impl: the `pub use` block in `lib.rs` surfaces every implemented estimator/helper at the crate root — scalers (`StandardScaler`, `MinMaxScaler`, `MaxAbsScaler`, `RobustScaler`, `Normalizer`, `PowerTransformer`, `QuantileTransformer`), encoders (`OneHotEncoder`, `OrdinalEncoder`, `LabelEncoder`, `LabelBinarizer`, `MultiLabelBinarizer`, `TargetEncoder`, `KBinsDiscretizer`), feature engineering (`Binarizer`, `FunctionTransformer`, `PolynomialFeatures`, `SplineTransformer`), imputers (`SimpleImputer`, `MissingIndicator`, `KNNImputer`, `IterativeImputer`), selectors (`VarianceThreshold`, `SelectKBest`, `GenericUnivariateSelect`, `SelectPercentile`, `SelectFromModel`/`SelectFromModelExt`, `RFE`, `RFECV`, `SequentialFeatureSelector`, `SelectFdr`/`SelectFpr`/`SelectFwe`, `SelectorMixin`, `chi2`/`f_classif`/`f_regression`/`r_regression`), root feature extraction (`DictVectorizer`, `FeatureHasher`), text (`CountVectorizer`, `HashingVectorizer`, `TfidfTransformer`, `TfidfVectorizer`), image helpers (`PatchExtractor`, `extract_patches_2d`, `grid_to_graph`, `img_to_graph`, `reconstruct_from_patches_2d`), projection (`GaussianRandomProjection`, `SparseRandomProjection`, `johnson_lindenstrauss_min_dim`), and compose (`ColumnTransformer`, `make_column_selector`, `make_column_transformer`) — each unfitted + `Fitted*` where applicable. Mirrors the covered `__all__` surfaces listed in Probes. Non-test consumers: meta-crate `pub use ferrolearn_preprocess as preprocess;` (`ferrolearn/src/lib.rs:36`); PyO3 pyclasses `RsStandardScaler` (`ferrolearn-python/src/transformers.rs`, `#[pyclass(name = "_RsStandardScaler")]`, registered `ferrolearn-python/src/lib.rs:22`) and `RsMinMaxScaler`/`RsMaxAbsScaler`/`RsRobustScaler`/`RsPowerTransformer` (`ferrolearn-python/src/extras.rs`, `_RsMinMaxScaler`/`_RsMaxAbsScaler`/`_RsRobustScaler`/`_RsPowerTransformer`, registered `ferrolearn-python/src/lib.rs:81-84`). Verification: `cargo test -p ferrolearn-preprocess --test divergence_dict_vectorizer`; `cargo test -p ferrolearn-preprocess --test divergence_feature_hasher`; `cargo test -p ferrolearn-preprocess --test divergence_image_graph`; `cargo test -p ferrolearn-preprocess --test divergence_image_patches`; `cargo test -p ferrolearn-preprocess --test divergence_lib`; `cargo build -p ferrolearn-preprocess` resolves every re-export. ABSENT (not-yet-translated) sklearn names enumerated in Architecture — the boundary correctly exports only what exists (honest underclaim). |
| REQ-substrate (ferray) | NOT-STARTED | open prereq blocker **#1362** (crate-root substrate, R-SUBSTRATE-1). `lib.rs`'s doc-comment examples and submodules operate on `ndarray::{Array1, Array2}` with `F: num_traits::Float + Send + Sync + 'static`; `ferrolearn-preprocess/Cargo.toml` declares `ndarray.workspace = true` + `num-traits.workspace = true`, not `ferray-core` / `ferray-ufunc`. The crate is on the wrong substrate; migration cascades through the module surface. Not migrated. |

## Architecture

**This is a boundary, not an estimator.** `lib.rs` has no transformer state. Its
surface, in source order, is the `pub mod` block followed by the
`pub use` re-export blocks. The re-export block hoists each
implemented estimator's unfitted + `Fitted*` type pair (and supporting
enums/functions) to the crate root, mirroring the *function* of the covered
sklearn `__all__` surfaces: defining the importable public surface. Below is the
module-by-module accounting of which `__all__` names are PRESENT (re-exported)
vs ABSENT (untranslated, not a boundary defect — the boundary ships exactly what
is implemented).

**`sklearn.preprocessing` (`preprocessing/__init__.py:30-60`).**
PRESENT: `Binarizer`, `FunctionTransformer`, `KBinsDiscretizer`,
`KernelCenterer`, `LabelBinarizer`, `LabelEncoder`, `MultiLabelBinarizer`,
`MinMaxScaler`, `MaxAbsScaler`, `QuantileTransformer`, `Normalizer`,
`OneHotEncoder`, `OrdinalEncoder`, `PowerTransformer`, `RobustScaler`,
`SplineTransformer`, `StandardScaler`, `TargetEncoder`, `PolynomialFeatures`,
plus function-form shortcuts `add_dummy_feature`, `maxabs_scale`, `minmax_scale`,
`label_binarize`, `quantile_transform`, and `power_transform`.
ABSENT: `binarize`, `normalize`, `scale`, and `robust_scale`.

**`sklearn.feature_selection` (`feature_selection/__init__.py:27-47`).**
PRESENT: `SequentialFeatureSelector`, `RFE`, `RFECV`, `SelectFdr`, `SelectFpr`,
`SelectFwe`, `SelectKBest`, `GenericUnivariateSelect`, `SelectFromModel` (surfaced *both* as
`feature_selection::SelectFromModel` — the importance-threshold form — *and* as
`SelectFromModelExt`/`FittedSelectFromModelExt`, the strategy-parameterised
extension), `SelectPercentile`, `VarianceThreshold`, `SelectorMixin`, `chi2`,
`f_classif`, `f_regression`, `r_regression`.
ABSENT: `mutual_info_classif`, `mutual_info_regression`, and `f_oneway`.

**`sklearn.feature_extraction` (`feature_extraction/__init__.py`).**
PRESENT: `DictVectorizer`, `FeatureHasher`, `grid_to_graph`, and `img_to_graph` at the crate root;
the `text` and `image` module contents are accounted for below.
ABSENT exact names: none. Residual contracts remain in
`.design/preprocess/dict_vectorizer.md` and
`.design/preprocess/feature_hasher.md` for sparse output, dtype/Python ABI,
`restrict`, and object-key/value support.

**`sklearn.feature_extraction.text` (`text.py:34-43`).**
PRESENT: `CountVectorizer`, `HashingVectorizer`, `TfidfTransformer`,
`TfidfVectorizer`.
ABSENT exact names: none. Residual contracts remain in
`.design/preprocess/count_vectorizer.md` and `.design/preprocess/tfidf.md` for
sparse output, analyzer/tokenizer hooks, constructor validation, feature names,
and Python ABI.

**`sklearn.feature_extraction.image` (`image.py:23-26`).**
PRESENT scoped dense helper surface: `PatchExtractor`, `extract_patches_2d`,
`grid_to_graph`, `img_to_graph`, and `reconstruct_from_patches_2d`.
ABSENT exact names: none. Residual contracts remain in `.design/preprocess/image.md`
for sparse graph returns, color/channel patch tensors, exact NumPy RNG stream
parity, Python ABI, and 3D image helper support.

**`sklearn.impute` (`impute/__init__.py:13`).**
PRESENT: `MissingIndicator`, `SimpleImputer`, `KNNImputer`, `IterativeImputer`
in the crate-root re-export block. Note `IterativeImputer` is experimental in
sklearn 1.5.2 (reached via `enable_iterative_imputer`, hence not in the literal
`__all__` at `:13`); ferrolearn surfaces it unconditionally.
ABSENT: none from the literal `impute.__all__`.

**`sklearn.random_projection` (`random_projection.py:50-54`).**
PRESENT: `GaussianRandomProjection`, `SparseRandomProjection`, and
`johnson_lindenstrauss_min_dim`.
ABSENT exact names: none.

**`sklearn.compose` (`compose/__init__.py:15-20`).**
PRESENT: `ColumnTransformer`, `make_column_selector`, `make_column_transformer`
in the crate-root re-export block.
ABSENT: `TransformedTargetRegressor`.

**Extension (no sklearn analog).** `BinaryEncoder`/`FittedBinaryEncoder` (`:169`)
is a `category_encoders`-style base-N categorical encoder with **no** scikit-learn
`__all__` entry. It is documented as a ferrolearn extension of the boundary, not a
sklearn-parity item and not an ABSENT gap — the same treatment
`.design/preprocess/select_from_model.md` gives `ThresholdStrategy::Percentile`.

The non-test consumers of the boundary are the meta-crate
(`ferrolearn/src/lib.rs:36`, `pub use ferrolearn_preprocess as preprocess;`) and
the `ferrolearn-python` PyO3 shim. The shim exposes eight scaler/transformer
pyclasses backed by re-exported types: `RsStandardScaler`
(`transformers.rs`, registered `python/src/lib.rs:22`) and `RsMinMaxScaler` /
`RsMaxAbsScaler` / `RsRobustScaler` / `RsPowerTransformer` (`extras.rs`, registered
`python/src/lib.rs:81-84`), each constructing a `ferrolearn_preprocess::<Scaler>`
and holding its `Fitted*` form. The remaining re-exported estimators are
grandfathered boundary public API (goal.md S5 / R-DEFER-1: boundary estimator
types *are* the public surface; their consumers are external users + the binding —
classifying them NOT-STARTED for lack of an internal caller would over-apply the
rule).

## Verification

Commands establishing the SHIPPED claim:

- `cargo build -p ferrolearn-preprocess` — every crate-root `pub use` resolves;
  a build failure here would mean a re-export names a non-existent type (REQ-1
  boundary integrity).
- `cargo test -p ferrolearn-preprocess --test divergence_lib` — compile-pins
  every crate-root re-export named in this boundary.
- `cargo test -p ferrolearn-preprocess --test divergence_image_graph` and
  `cargo test -p ferrolearn-preprocess --test divergence_image_patches` — pin the
  scoped image helper values and shapes against live sklearn 1.5.2 oracles.
- `cargo build -p ferrolearn` — the meta-crate compiles against the re-export
  boundary, exercising `pub use ferrolearn_preprocess as preprocess;`
  (`ferrolearn/src/lib.rs:36`).
- `cargo build -p ferrolearn-python` — the PyO3 shim compiles the eight scaler /
  transformer pyclasses against the re-exported types (the production consumers of
  REQ-1).
- Re-export surface (REQ-1): every estimator type in the `pub use` block is named
  by its own routed doc under `.design/preprocess/`.
- sklearn-side enumeration (REQ-1 PRESENT/ABSENT accounting): the six `__all__`
  probes plus `feature_extraction.image.__all__` above, run live against sklearn
  1.5.2 — the authoritative source for what the boundary is measured against
  (R-CHAR-3).

REQ-1 is SHIPPED. REQ-substrate is NOT-STARTED until the ferray migration lands;
`cargo test` is green on the current `ndarray` substrate, but that does not satisfy
R-SUBSTRATE-1, so AC-substrate has no green verification yet.

## Blockers

- **#1362 (REQ-substrate)** — placeholder. The entire `ferrolearn-preprocess`
  crate (the `lib.rs` doc-comment examples and submodules behind the
  `pub use` boundary) is on `ndarray` + `num-traits`, not `ferray-core` /
  `ferray-ufunc` (R-SUBSTRATE-1). Migrating the substrate cascades through every
  transformer's `Fit`/`Transform`/`FitTransform` signature; REQ-substrate is
  NOT-STARTED until it lands. (Filing of the concrete blocker number is left to
  the dispatcher per the markdown-only scope of this doc.)
