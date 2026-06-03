# TfidfTransformer

<!--
tier: 3-component
status: draft
baseline-commit: 16de779d
upstream: scikit-learn 1.5.2 (commit 156ef14)
upstream-paths:
  - sklearn/feature_extraction/text.py  # class TfidfTransformer(OneToOneFeatureMixin, TransformerMixin, BaseEstimator) (:1483); _parameter_constraints {norm: {l1,l2}|None, use_idf/smooth_idf/sublinear_tf: boolean} (:1614-1619); __init__(*, norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False) (:1621-1625); fit -> df=_document_frequency(X); df+=smooth_idf; n+=smooth_idf; idf_=log(n/df)+1 (:1628-1668); transform -> sublinear (log(X.data); +=1) (:1698-1700); X.data*=idf_[X.indices] (:1702-1705); normalize(norm) (:1707-1708) ; idf_ set as attr (:1666). class TfidfVectorizer(CountVectorizer) = CountVectorizer + TfidfTransformer (:1721); idf_ property (:1991-2006); fit/fit_transform/transform (:2037-2097)
ferrolearn-module: ferrolearn-preprocess/src/tfidf.rs
parity-ops: TfidfTransformer, TfidfVectorizer
crosslink-issue: 1210
-->

## Summary

scikit-learn's `TfidfTransformer` (`text.py:1483`) reweights a term-count matrix
into a TF-IDF representation: `fit` learns `idf_ = log(n/df) + 1` (with optional
`+1` smoothing of both numerator and denominator), and `transform` applies
optional sublinear TF scaling (`1 + log(tf)`), multiplies each column by its IDF,
then row-normalizes (`l1`/`l2`/`None`). It operates on **sparse CSR** matrices and
exposes `idf_` as a fitted NumPy attribute. `TfidfVectorizer` (`text.py:1721`)
chains `CountVectorizer` + `TfidfTransformer` over raw text documents.

`ferrolearn-preprocess/src/tfidf.rs` ships a faithful **dense** `TfidfTransformer<F>`
/ `FittedTfidfTransformer<F>` pair that implements the full numeric contract for
the default and configured paths: all four parameters (`norm`, `use_idf`,
`smooth_idf`, `sublinear_tf`) are present and behaviorally correct, the IDF formula
and l2/l1/None normalization value-match the live sklearn oracle to ULP tolerance
(see Probes), and `idf()` exposes the learned weights. What is **absent**: sparse
(CSR) input/output (ferrolearn computes on `ndarray::Array2`), `TfidfVectorizer`
(no raw-text → TF-IDF chain type — only `CountVectorizer` and `TfidfTransformer`
exist separately), `_parameter_constraints`-style validation / typed param errors
(R-DEV-2), a PyO3 binding, and the ferray substrate. The non-test boundary consumer
is the crate re-export `pub use tfidf::{FittedTfidfTransformer, TfidfNorm,
TfidfTransformer}` (`lib.rs` line 142).

## Probes (live sklearn oracle, 1.5.2)

```bash
# Shared count matrix: 3 docs x 3 features
#   counts = [[1,1,0],[1,0,1],[1,0,0]]   (feature 0 in all 3 docs; features 1,2 in 1 doc each)

# REQ-1 — default idf_ (smooth_idf=True): idf = log((1+3)/(1+df)) + 1
python3 -c "import numpy as np; from sklearn.feature_extraction.text import TfidfTransformer; \
print(TfidfTransformer().fit(np.array([[1.,1.,0.],[1.,0.,1.],[1.,0.,0.]])).idf_.tolist())"
# -> [1.0, 1.6931471805599454, 1.6931471805599454]
#    ferrolearn fit: idf[0]=ln(4/4)+1=1.0 ; idf[1]=idf[2]=ln(4/2)+1=ln2+1=1.6931471805599454  (MATCH)

# REQ-2 — default fit_transform (l2, smooth, use_idf): tf*idf then l2-normalize each row
python3 -c "import numpy as np; from sklearn.feature_extraction.text import TfidfTransformer; \
print(np.round(TfidfTransformer().fit_transform(np.array([[1.,1.,0.],[1.,0.,1.],[1.,0.,0.]])).toarray(),8).tolist())"
# -> [[0.50854232, 0.861037, 0.0], [0.50854232, 0.0, 0.861037], [1.0, 0.0, 0.0]]   (MATCHES ferrolearn dense output)

# REQ-3 — smooth_idf=False: idf = log(3/df) + 1 (no +1 on n or df)
python3 -c "import numpy as np; from sklearn.feature_extraction.text import TfidfTransformer; \
print(TfidfTransformer(smooth_idf=False).fit(np.array([[1.,1.,0.],[1.,0.,1.],[1.,0.,0.]])).idf_.tolist())"
# -> [1.0, 2.09861228866811, 2.09861228866811]   (ferrolearn: ln(3/1)+1=2.0986...; MATCH)

# REQ-4 — norm='l1' and norm=None
python3 -c "import numpy as np; from sklearn.feature_extraction.text import TfidfTransformer as T; c=np.array([[1.,1.,0.],[1.,0.,1.],[1.,0.,0.]]); \
print('l1', np.round(T(norm='l1').fit_transform(c).toarray(),8).tolist()); \
print('none', np.round(T(norm=None).fit_transform(c).toarray(),8).tolist())"
# -> l1   [[0.37131279, 0.62868721, 0.0], [0.37131279, 0.0, 0.62868721], [1.0, 0.0, 0.0]]
# -> none [[1.0, 1.69314718, 0.0], [1.0, 0.0, 1.69314718], [1.0, 0.0, 0.0]]   (both MATCH ferrolearn)

# REQ-5 — sublinear_tf=True: tf=4 -> 1+log(4)=2.3862943611; tf=1 -> 1.0
python3 -c "import numpy as np; from sklearn.feature_extraction.text import TfidfTransformer as T; \
print(np.round(T(sublinear_tf=True,use_idf=False,norm=None).fit_transform(np.array([[4.,1.]])).toarray(),8).tolist())"
# -> [[2.38629436, 1.0]]   (ferrolearn test_tfidf_sublinear_tf pins exactly this; MATCH)

# REQ-9 — _parameter_constraints reject bad norm (R-DEV-2 error-type contract)
python3 -c "from sklearn.feature_extraction.text import TfidfTransformer; import numpy as np; \
TfidfTransformer(norm='l3').fit(np.array([[1.,0.]]))"
# -> InvalidParameterError: The 'norm' parameter ... must be a str among {'l1','l2'} or None. Got 'l3' instead.
```

## Requirements

- REQ-1: `fit` learns the IDF vector. Default (`smooth_idf=True`):
  `idf = ln((1+n)/(1+df)) + 1` per feature (`df` = #docs where the feature is
  non-zero), matching sklearn `idf_` (`text.py:1660-1666`).
- REQ-2: `transform` default path — multiply each column by its IDF then
  l2-normalize each row, matching sklearn `transform` (`text.py:1702-1708`) for
  the default config (`use_idf=True, smooth_idf=True, norm='l2', sublinear_tf=False`).
- REQ-3: `smooth_idf=False` IDF variant — `idf = ln(n/df) + 1`
  (`text.py:1660-1666` with `smooth_idf` falsey).
- REQ-4: `norm` variants `'l1'`, `'l2'`, and `None` row normalization
  (`text.py:1615`, `:1707-1708`; `normalize`).
- REQ-5: `sublinear_tf=True` — replace `tf` with `1 + ln(tf)` for `tf > 0`
  (`text.py:1698-1700`).
- REQ-6: `use_idf=False` path — TF (optionally sublinear) + normalization with
  **no** IDF multiply and **no** `idf_` attribute (`text.py:1654`, `:1702`).
- REQ-7: `idf_` fitted-attribute exposure — read access to the learned IDF
  vector (sklearn fitted attr `idf_`, `text.py:1666`; also the `TfidfVectorizer.idf_`
  property `:1991-2006`).
- REQ-8: Sparse (CSR) input/output — operate on sparse term-count matrices and
  return a sparse TF-IDF matrix (`text.py:1648`, `:1684`, `:1696`).
- REQ-9: Constructor parameter surface + `_parameter_constraints` validation —
  `*`-only kwargs and typed param rejection (`InvalidParameterError`) for bad
  `norm`/non-boolean flags (`text.py:1614-1625`) per R-DEV-2.
- REQ-10: `TfidfVectorizer` — raw-documents → TF-IDF chain
  (`CountVectorizer` + `TfidfTransformer`), incl. its `idf_` property and
  `fit`/`fit_transform`/`transform` (`text.py:1721`, `:2037-2097`).
- REQ-11: PyO3 binding — `import ferrolearn` exposes `TfidfTransformer`
  (and `TfidfVectorizer`) mirroring `import sklearn` (project boundary consumer).
- REQ-12: ferray substrate — compute on `ferray-core` arrays rather than
  `ndarray::Array2` + `num_traits::Float` (R-SUBSTRATE).

## Acceptance criteria

- AC-1 (REQ-1): `TfidfTransformer::<f64>::new().fit(&counts).idf()` for the Probe
  matrix equals `[1.0, 1.6931471805599454, 1.6931471805599454]` to ULP tolerance
  (REQ-1 Probe). Pinned by `test_tfidf_smooth_idf`.
- AC-2 (REQ-2): `new().fit(&counts).transform(&counts)` equals the REQ-2 Probe
  `[[0.50854232, 0.861037, 0], [0.50854232, 0, 0.861037], [1, 0, 0]]` to 1e-8;
  every output row has unit l2 norm. Partly pinned by `test_tfidf_basic`
  (norm-only), needs an oracle-grounded value `#[test]`.
- AC-3 (REQ-3): `new().smooth_idf(false).fit(&counts).idf()` equals
  `[1.0, 2.09861228866811, 2.09861228866811]` (REQ-3 Probe). Pinned by
  `test_tfidf_no_smooth_idf`.
- AC-4 (REQ-4): l1/None outputs match the REQ-4 Probe; l1 rows sum to 1, None is
  `tf*idf` unchanged. Partly pinned by `test_tfidf_l1_norm`, `test_tfidf_no_norm`.
- AC-5 (REQ-5): `sublinear_tf(true)` on `[[4,1]]` yields `[[2.38629436, 1.0]]`
  (REQ-5 Probe). Pinned by `test_tfidf_sublinear_tf`.
- AC-6 (REQ-6): `use_idf(false)` produces a `FittedTfidfTransformer` with
  `idf() == None` and `transform` applies only TF + normalization. Pinned by
  `test_tfidf_no_idf`.
- AC-7 (REQ-7): `fitted.idf()` returns `Some(&Array1<F>)` when `use_idf`, `None`
  otherwise. Pinned by `test_tfidf_smooth_idf` / `test_tfidf_no_idf`.
- AC-8 (REQ-8): a sparse `sprs`/CSR count matrix can be fit/transformed; output
  is sparse. (No sparse signature exists today.)
- AC-9 (REQ-9): constructing/fitting with `norm='l3'` (or a non-boolean flag)
  yields the sklearn-matching error type (REQ-9 Probe → `InvalidParameterError`).
- AC-10 (REQ-10): a `TfidfVectorizer` type accepts `Vec<String>` and produces a
  TF-IDF matrix equal to `CountVectorizer` + `TfidfTransformer`; exposes `idf_`.
- AC-11 (REQ-11): `python3 -c "import ferrolearn; ferrolearn.feature_extraction.TfidfTransformer"`
  resolves and `.transform` matches sklearn on the REQ-2 Probe.
- AC-12 (REQ-12): the compute path uses `ferray-core` (no `ndarray`/`num_traits`
  in fit/transform).

## REQ status

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (default smooth idf_) | SHIPPED | impl `pub fn fit in tfidf.rs` computes `idf_vec[j] = ((F::one()+n_f)/(F::one()+df_f)).ln() + F::one()` (smooth branch) mirroring sklearn `idf_ = np.log(n_samples/df)+1.0` after `df += smooth_idf; n_samples += smooth_idf` (`text.py:1660-1666`). Non-test consumer: crate re-export `pub use tfidf::{FittedTfidfTransformer, TfidfNorm, TfidfTransformer}` (`lib.rs` line 142) — the crate boundary public API (grandfathered under S5/R-DEFER-1). Verification: REQ-1 Probe `[1.0, 1.6931471805599454, 1.6931471805599454]` equals ferrolearn `idf()`; pinned by `test_tfidf_smooth_idf` (`cargo test -p ferrolearn-preprocess`). |
| REQ-2 (default transform: idf multiply + l2) | SHIPPED | impl `pub fn transform in FittedTfidfTransformer (tfidf.rs)` does `*v = *v * idf[j]` per column then the `TfidfNorm::L2` arm divides each row by `sqrt(sum v^2)`, mirroring sklearn `X.data *= self.idf_[X.indices]` + `normalize(X, norm=self.norm)` (`text.py:1705`, `:1708`). Non-test consumer: `lib.rs` line 142 re-export. Verification: REQ-2 Probe `[[0.50854232,0.861037,0],…]` value-matches the dense ferrolearn output (oracle confirmed); `test_tfidf_basic` pins unit-l2 rows. Least-confident claim: only the norm property (not the full oracle value vector) is currently pinned by a `#[test]` — see Verification. |
| REQ-3 (smooth_idf=False) | SHIPPED | impl `fit in tfidf.rs` else-branch `idf_vec[j] = (n_f/df_f).ln() + F::one()` mirrors sklearn no-smooth path (`text.py:1660-1666`). Non-test consumer: `lib.rs` line 142. Critic-verified value-match: REQ-3 Probe `[1.0, 2.09861228866811, 2.09861228866811]` equals ferrolearn `idf()` (`green_req3_*` in `tests/divergence_tfidf.rs`). **DEVIATION (R-DEV-4, documented):** for the `df==0` edge (an all-zero column, which a `CountVectorizer` never emits) the else-branch guards `df==0 -> 1.0`, whereas sklearn computes `log(n/0) = inf` → `idf_=inf` → `tf*idf = 0*inf = nan` (a footgun). ferrolearn's `1.0` guard avoids the NaN; this is the only df-regime divergence and it is a deliberate Rust-analog-better deviation, not a gap. |
| REQ-4 (norm l1/l2/None) | SHIPPED | impl `transform in tfidf.rs` `match self.norm { TfidfNorm::L1 => …/sum\|v\|, L2 => …/sqrt(sum v^2), None => {} }` mirrors sklearn `normalize(X, norm=self.norm)` for `{'l1','l2',None}` (`text.py:1707-1708`); `TfidfNorm` enum default `L2`. Non-test consumer: `lib.rs` line 142. Verification: REQ-4 Probe l1/None vectors match; `test_tfidf_l1_norm`, `test_tfidf_no_norm`. |
| REQ-5 (sublinear_tf) | SHIPPED | impl `transform in tfidf.rs` `if self.sublinear_tf { result.mapv_inplace(\|v\| if v>0 {1+v.ln()} else {v}) }` mirrors sklearn `np.log(X.data); X.data += 1.0` (`text.py:1698-1700`). Non-test consumer: `lib.rs` line 142. Verification: REQ-5 Probe `[[2.38629436,1.0]]` equals ferrolearn; pinned by `test_tfidf_sublinear_tf`. |
| REQ-6 (use_idf=False) | SHIPPED | impl `fit in tfidf.rs` `let idf = if self.use_idf { Some(..) } else { None }`; `transform` guards the IDF multiply behind `if let Some(ref idf) = self.idf`, so falsey `use_idf` yields TF+norm only and no IDF — mirrors sklearn `if self.use_idf` (`text.py:1654`) + `if hasattr(self,'idf_')` (`text.py:1702`). Non-test consumer: `lib.rs` line 142. Verification: `test_tfidf_no_idf` (idf absent; rows still unit-l2). |
| REQ-7 (idf_ attribute exposure) | SHIPPED | impl `pub fn idf(&self) -> Option<&Array1<F>> in FittedTfidfTransformer (tfidf.rs)` returns the learned vector (or `None` when `use_idf=false`), mirroring the sklearn fitted attr `idf_` (`text.py:1666`). Non-test consumer: `lib.rs` line 142 (the getter is part of the re-exported fitted type's public surface). Verification: `test_tfidf_smooth_idf` reads `fitted.idf().unwrap()`. Note: ferrolearn exposes IDF via a `fn idf()` accessor, not a public field named `idf_`. |
| REQ-8 (sparse CSR I/O) | NOT-STARTED | open prereq blocker #1211. `fit`/`transform` take `&Array2<F>` (dense) and return `Array2<F>`; there is no `sprs`/CSR signature. sklearn validates `accept_sparse=('csr','csc')` and returns a sparse matrix (`text.py:1648`, `:1684`). |
| REQ-9 (ctor surface + _parameter_constraints) | NOT-STARTED | open prereq blocker #1212. All four params exist as builder setters, but `norm` is a closed Rust enum (`TfidfNorm`) so there is no runtime string-validation / typed `InvalidParameterError` path; no `_parameter_constraints` analog (REQ-9 Probe → `InvalidParameterError`) (R-DEV-2). |
| REQ-10 (TfidfVectorizer) | NOT-STARTED | open prereq blocker #1213. No `TfidfVectorizer` type exists — only `CountVectorizer` and `TfidfTransformer` separately; the raw-text→TF-IDF chain and its `idf_` property (`text.py:1721`, `:1991-2006`, `:2037-2097`) are unimplemented. |
| REQ-11 (PyO3 binding) | NOT-STARTED | open prereq blocker #1214. No `ferrolearn-python` registration; `import ferrolearn` cannot expose `TfidfTransformer`/`TfidfVectorizer` (boundary consumer per R-DEFER-1). |
| REQ-12 (ferray substrate) | NOT-STARTED | open prereq blocker #1215. Compute path uses `ndarray::Array2`/`Array1` + `num_traits::Float` + `mapv_inplace`/`rows_mut`, not `ferray-core` (R-SUBSTRATE-1/2). |

## Architecture

**ferrolearn (existing).** Two generic types in `tfidf.rs`. The unfitted
`TfidfTransformer<F>` holds the four public params (`norm: TfidfNorm`, `use_idf:
bool`, `smooth_idf: bool`, `sublinear_tf: bool`) plus a `PhantomData<F>`, with a
builder API (`new`, `norm`, `use_idf`, `smooth_idf`, `sublinear_tf`) and
`Default`. `fit(&Array2<F>) -> Result<FittedTfidfTransformer<F>, FerroError>`
errors on zero rows (`FerroError::InsufficientSamples`) and, when `use_idf`,
loops each column computing `df` (count of non-zero entries) and the smooth or
unsmooth IDF, returning `Some(idf_vec)` (else `None`). The fitted
`FittedTfidfTransformer<F>` carries `idf: Option<Array1<F>>`, `norm`,
`sublinear_tf`; `idf()` exposes the vector; `transform` (a) errors on zero rows,
(b) shape-checks `counts.ncols() == idf.len()` when IDF present
(`FerroError::ShapeMismatch`), (c) applies sublinear TF in place, (d) multiplies
each row element-wise by `idf[j]`, (e) row-normalizes per `TfidfNorm`. Generic
bound `F: Float + Send + Sync + 'static` supports f32/f64.

**sklearn (target contract).** `TfidfTransformer(OneToOneFeatureMixin,
TransformerMixin, BaseEstimator)` (`text.py:1483`) with `_parameter_constraints`
(`:1614-1619`) and `__init__(*, norm='l2', use_idf=True, smooth_idf=True,
sublinear_tf=False)` (`:1621-1625`). `fit` (`:1628-1668`) validates sparse CSR/CSC
input, computes `df = _document_frequency(X)`, then `df += smooth_idf; n_samples
+= smooth_idf; self.idf_ = log(n_samples/df) + 1.0` — i.e. the smooth flag adds
`1` to **both** `n` and `df` (ferrolearn's `(1+n)/(1+df)` is algebraically
identical). `transform` (`:1670-1710`) operates on `X.data` of a CSR matrix:
sublinear (`log` then `+1`), `X.data *= idf_[X.indices]`, then
`normalize(X, norm)` if `norm is not None`. `idf_` is a fitted attribute.
`TfidfVectorizer(CountVectorizer)` (`:1721`) wires `CountVectorizer` to an internal
`_tfidf = TfidfTransformer(...)` and re-exposes `idf_` as a property
(`:1991-2006`).

**Numeric equivalence.** The IDF formula, sublinear TF, IDF multiply, and l1/l2/None
normalization all value-match the live oracle to ULP tolerance (Probes REQ-1..5).
The substantive divergences are structural, not numeric: dense-only I/O (REQ-8),
enum-typed `norm` with no runtime validation contract (REQ-9), the missing
`TfidfVectorizer` chain (REQ-10), and the absent binding/substrate (REQ-11/12).

## Verification

Commands establishing the SHIPPED claims (REQ-1..7):

```bash
# Oracle (Probes REQ-1..5) — see Probes block above; all value-match dense ferrolearn output.

# Crate gauntlet:
cargo test -p ferrolearn-preprocess   # incl. test_tfidf_basic, test_tfidf_no_idf,
                                       # test_tfidf_l1_norm, test_tfidf_no_norm,
                                       # test_tfidf_sublinear_tf, test_tfidf_smooth_idf,
                                       # test_tfidf_no_smooth_idf, test_tfidf_empty,
                                       # test_tfidf_shape_mismatch, test_tfidf_f32
cargo clippy -p ferrolearn-preprocess --all-targets -- -D warnings
cargo fmt --all --check
```

The existing `#[test]`s pin REQ-1/3/5/6/7 against **exact oracle values**
(`test_tfidf_smooth_idf`: `idf[1] == 2f64.ln()+1`; `test_tfidf_no_smooth_idf`:
`3f64.ln()+1`; `test_tfidf_sublinear_tf`: `1+4f64.ln()`). REQ-2 and REQ-4 are
currently pinned only by their **norm property** (`row_norm ≈ 1`,
`row_l1 ≈ 1`, identity for None) — NOT by the full oracle value vector
`[[0.50854232,0.861037,0],…]`. This is the single most fixable gap: the critic
should add an oracle-grounded `#[test]` asserting the REQ-2 dense output equals
the REQ-2 Probe to 1e-8 (R-CHAR-3). `api_proof.rs` (`tests/`) is a test-only
consumer and does not count toward R-DEFER-1; the lib.rs re-export is the
non-test boundary consumer. No currently-green command establishes REQ-8..12.

## Blockers

Each NOT-STARTED REQ files a `-l blocker` issue (the orchestrator assigns
`#`-numbers); reference them in the REQ status table:

- #1211 — REQ-8: dense `Array2<F>` only; no `sprs`/CSR fit/transform
  signature (sklearn `accept_sparse=('csr','csc')`, sparse output).
- #1212 — REQ-9: closed `TfidfNorm` enum with no runtime
  string-validation / `InvalidParameterError` path; no `_parameter_constraints`
  analog (R-DEV-2).
- #1213 — REQ-10: no `TfidfVectorizer` type chaining
  `CountVectorizer` + `TfidfTransformer`; no `idf_` property re-exposure.
- #1214 — REQ-11: no `ferrolearn-python` registration of `TfidfTransformer`/
  `TfidfVectorizer`.
- #1215 — REQ-12: compute path on `ndarray`/`num_traits`, not `ferray-core`.
```
