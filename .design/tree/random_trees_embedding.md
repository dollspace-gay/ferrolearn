# Random Trees Embedding (RandomTreesEmbedding)

<!--
tier: 3-component
status: draft
baseline-commit: c4fe2b21
upstream: scikit-learn 1.5.2 (commit 156ef14)
upstream-paths:
  - sklearn/ensemble/_forest.py     # RandomTreesEmbedding (:2623); class-level criterion='squared_error'/max_features=1 (:2815-2816); __init__ defaults n_estimators=100/max_depth=5/min_samples_split=2/min_samples_leaf=1/min_weight_fraction_leaf=0.0/max_leaf_nodes=None/min_impurity_decrease=0.0/sparse_output=True/random_state=None (:2818-2833); estimator=ExtraTreeRegressor()/bootstrap=False (:2834-2848); fit (:2868, delegates to fit_transform); fit_transform (:2899) rnd.uniform(size=n_samples) random y (:2924-2925), one_hot_encoder_ = OneHotEncoder(sparse_output) (:2928), output = one_hot_encoder_.fit_transform(self.apply(X)) (:2929); transform (:2965) = one_hot_encoder_.transform(self.apply(X)) (:2982); _parameter_constraints sparse_output ["boolean"] (:2810); BaseForest.apply (:265)
  - sklearn/tree/_classes.py        # ExtraTreeRegressor — the per-tree base estimator (splitter='random', criterion='squared_error', max_features=1.0); the totally-random splits ignore y, shared with .design/tree/extra_tree.md
ferrolearn-module: ferrolearn-tree/src/random_trees_embedding.rs
parity-ops: RandomTreesEmbedding
crosslink-issue: 686
-->

## Summary

`ferrolearn-tree/src/random_trees_embedding.rs` mirrors scikit-learn's
`sklearn.ensemble.RandomTreesEmbedding` (`_forest.py:2623`) — an **unsupervised**
feature transform. sklearn builds a forest of **totally-random trees** (a
`BaseForest` of `ExtraTreeRegressor`s with `bootstrap=False`,
`max_features=1`, `splitter='random'`) on a **fabricated uniform `[0,1)`
random target** (`fit_transform`, `:2924-2925`); the transform is the **one-hot
encoding** of the leaf each sample falls into, **concatenated across all trees**
(`one_hot_encoder_.transform(self.apply(X))`, `:2982`). The result is a
high-dimensional sparse binary code feeding downstream linear models.

ferrolearn re-implements this natively without ever materializing a random
target. The module ships unfitted `RandomTreesEmbedding<F>` and
`FittedRandomTreesEmbedding<F>`, a per-tree purely-random split builder
(`fn build_random_tree`, `fn random_threshold`, `fn traverse_tree` in
`random_trees_embedding.rs`), `Fit<Array2<F>, ()>` (the `()` target makes the
"ignored y" unsupervised contract type-level), `Transform<Array2<F>>` producing
the one-hot leaf code, fitted accessors (`fn n_estimators`, `fn n_features`,
`fn n_output_features`), and pipeline adapters
(`PipelineTransformer`/`FittedPipelineTransformer`).

**This is an inherently RNG-driven embedding.** The per-tree feature pick
(`next_u64() % n_features`) and threshold draw
(`random_threshold = min + u*(max-min)`, `u = next_u64()/u64::MAX`) are seeded
by `random_state` through Rust's `StdRng`; sklearn draws the random target, the
feature, and the threshold from numpy's MT19937 (`rnd.uniform`,
`ExtraTreeRegressor`'s `RandomSplitter`). The two streams cannot bit-match, so
**exact embedding-for-embedding parity at a given `random_state` is a documented
RNG boundary** — the same boundary already accepted for `extra_tree`,
`random_forest`, and the SGD shuffle. The **deterministic** contract — the
param/default surface and the **structural one-hot contract** (each output row
has exactly `n_estimators` ones; column count equals the sum of per-tree leaf
counts; all entries are in `{0, 1}`) — is the shippable/pinnable part; the exact
forest at a seed is not.

**Top divergences the director / fixer must see:**

1. **`n_estimators` default is 10, not 100 (R-DEV-2 violation).**
   `RandomTreesEmbedding::<F>::new` (`fn new` in `random_trees_embedding.rs`)
   sets `n_estimators = 10`; sklearn's `__init__` default is `100`
   (`_forest.py:2820`). Live: `RandomTreesEmbedding().n_estimators == 100`. This
   is a wrong-default ABI bug, not an RNG-boundary artifact, and is the fix the
   fixer will apply (REQ-1, blocker #1841).
2. **Output is DENSE, not sparse.** `FittedRandomTreesEmbedding::transform`
   (`transform` in `random_trees_embedding.rs`) returns a dense `Array2<F>`;
   sklearn's `sparse_output=True` default returns a CSR sparse matrix
   (`OneHotEncoder(sparse_output=self.sparse_output)`, `_forest.py:2928`; live
   `type(fit_transform(X)) == csr_matrix`). ferrolearn has **no `sparse_output`
   parameter** and no sparse path at all (REQ-4, NOT-STARTED, blocker #1843).

This doc adapts to the **existing** code. Under R-HONEST-3 no oracle pins exist
yet and the embedding is RNG-driven, so: the param-surface REQ for the params
ferrolearn *has* is SHIPPED with the `n_estimators` default flagged and absent
params enumerated; the totally-random per-tree build and the structural one-hot
contract are SHIPPED (deterministically verifiable, RNG-robust); the
`sparse_output` path, the missing params, exact numpy-MT embedding parity, and
the ferray substrate are NOT-STARTED with concrete blockers.

## Algorithm (sklearn — the contract)

### Estimator surface & defaults (live `get_params()`, sklearn 1.5.2)

`RandomTreesEmbedding.__init__` (`_forest.py:2818-2833`): `n_estimators=100`,
`max_depth=5`, `min_samples_split=2`, `min_samples_leaf=1`,
`min_weight_fraction_leaf=0.0`, `max_leaf_nodes=None`, `min_impurity_decrease=0.0`,
`sparse_output=True`, `n_jobs=None`, `random_state=None`, `verbose=0`,
`warm_start=False`. The per-tree base estimator is fixed:
`estimator=ExtraTreeRegressor()` with `bootstrap=False` (`:2834-2848`), and the
class fixes `criterion='squared_error'` and `max_features=1` at class scope
(`:2815-2816`) — every split inspects exactly **one** randomly chosen feature.
Note `max_features`, `ccp_alpha`, `splitter`, `monotonic_cst` are explicitly
**popped** from `_parameter_constraints` (`:2812-2813`), so they are NOT
user-tunable on this estimator.

**The defaults ferrolearn matches:** `max_depth=Some(5)` (= sklearn `5`),
`min_samples_split=2`, `random_state=None`.

**The default ferrolearn gets WRONG:** `n_estimators` — ferrolearn `new()` sets
`10`, sklearn defaults `100` (`_forest.py:2820`). R-DEV-2 violation (REQ-1).

**Params ABSENT in ferrolearn** (REQ-1 flags each, REQ-4 the sparse one):
`min_samples_leaf` (sklearn `1`, `:2824` — ferrolearn has no per-leaf minimum,
only `min_samples_split`); `min_weight_fraction_leaf`; `max_leaf_nodes`;
`min_impurity_decrease`; **`sparse_output`** (sklearn `True`, `:2828` — the
dense-vs-sparse divergence, REQ-4); `n_jobs`/`verbose`/`warm_start` (Python
ergonomics / R-DEV-4 — `n_jobs` is a non-observable parallelism choice, not an
API gap).

### fit / fit_transform — random-y then forest (`:2868`, `:2899`)

sklearn's `fit` (`:2868`) just calls `fit_transform` (`:2895`). `fit_transform`
(`:2899`): `rnd = check_random_state(self.random_state)`; **`y = rnd.uniform(size=n_samples)`** — a uniform `[0,1)` random target is fabricated
(`:2924-2925`); `super().fit(X, y, ...)` builds the `ExtraTreeRegressor` forest
against that random `y` (`:2926`). Because the trees are
`splitter='random'`/`max_features=1`, the split feature and threshold are
**purely random** and the random `y` only feeds the (irrelevant) leaf-value /
impurity bookkeeping — the partitioning is unsupervised in effect.

**ferrolearn deviates correctly (R-DEV-4).** `Fit<Array2<F>, ()>` takes a
**unit `()` target** and `build_random_tree` (`random_trees_embedding.rs`)
ignores it entirely: a random feature is drawn (`next_u64() % n_features`), the
node-local feature min/max is computed, and a uniform threshold in `[min, max)`
is drawn (`random_threshold`); leaf "values" are stored as `F::zero()`. ferrolearn
never fabricates a random `y` because nothing downstream reads a leaf value — the
embedding depends only on **which leaf** a sample reaches, not its value. The
observable contract (leaf partitioning) is preserved; the fabricated-target
mechanism is a CPython-side scaffold Rust elides (REQ-5).

### apply + one-hot (`BaseForest.apply:265`, `OneHotEncoder`, `:2928-2982`)

sklearn `apply(X)` returns an `(n_samples, n_estimators)` int matrix of the leaf
**node index** each sample reaches in each tree. `fit_transform` then fits a
`OneHotEncoder(sparse_output=self.sparse_output)` on that matrix
(`:2928-2929`) — one categorical column per tree, expanded to one binary column
per distinct leaf — and `transform` re-applies it (`:2982`). The output column
count is therefore `Σ_t (#distinct leaves reached in tree t)`, and each row has
exactly **`n_estimators` ones** (one active leaf per tree). Output dtype is a
CSR sparse matrix by default (`sparse_output=True`).

**ferrolearn (`transform`)** enumerates **all** leaf nodes per tree at fit time
(`leaf_maps`, `leaf_counts`, `total_leaves` in `FittedRandomTreesEmbedding`),
traverses each sample to its leaf (`traverse_tree`), and sets a single `F::one()`
at `col_offset + leaf_pos`, concatenating across trees. Output is a **dense**
`Array2<F>` of shape `(n_samples, total_leaves)`. Two structural differences
from sklearn, both RNG-robust and deterministically checkable:
- **dense vs CSR** (REQ-4) — ferrolearn returns dense; sklearn CSR.
- **leaf-column basis**: ferrolearn enumerates *every* leaf node of each tree
  (`total_leaves`), whereas sklearn's `OneHotEncoder` columns are the *distinct
  leaf indices actually observed at fit*. On the training `X` these coincide
  (every leaf is reachable by construction), so the **`Σleaves` column count and
  the `n_estimators`-ones-per-row contract hold identically**; they could differ
  only for a leaf reachable by no fit-time sample (not constructible here).

### random_state determinism

sklearn seeds the random `y`, the per-split feature pick, and the threshold from
one MT19937 stream (`check_random_state(self.random_state)`). ferrolearn seeds
`StdRng::seed_from_u64(random_state)` (or `StdRng::from_os_rng()` when `None`)
and draws the feature/threshold from it. Same seed ⇒ identical ferrolearn forest
⇒ identical embedding (`test_reproducibility`). Exact cross-impl bit-parity is
the RNG boundary (REQ-6).

## ferrolearn (what exists)

- **Unfitted**: `pub struct RandomTreesEmbedding<F>` with public fields
  `n_estimators`, `max_depth: Option<usize>`, `min_samples_split`,
  `random_state: Option<u64>`; builder setters `with_n_estimators`,
  `with_max_depth`, `with_min_samples_split`, `with_random_state`;
  `Default`/`fn new`.
- **Fitted**: `pub struct FittedRandomTreesEmbedding<F>` (private fields `trees:
  Vec<Vec<Node<F>>>`, `leaf_counts`, `leaf_maps`, `total_leaves`, `n_features`);
  accessors `fn n_estimators`, `fn n_features`, `fn n_output_features`.
- **Traits**: `Fit<Array2<F>, ()>` (unsupervised, unit target); `Transform<Array2<F>>`
  (`Output = Array2<F>` dense); `PipelineTransformer<F>`/`FittedPipelineTransformer<F>`.
- **Internals**: `fn build_random_tree` (recursive purely-random builder),
  `fn random_threshold` (uniform `[min,max)` draw), `fn traverse_tree`
  (root→leaf). `Node<F>` is shared from `decision_tree.rs`.
- **Consumers**: crate re-export (`lib.rs` `pub use
  random_trees_embedding::{FittedRandomTreesEmbedding, RandomTreesEmbedding}`);
  the `PipelineTransformer`/`FittedPipelineTransformer` impls, consumed by
  `ferrolearn-core`'s `Pipeline`/`FittedPipeline` machinery (the same non-test
  consumer surface `.design/core/pipeline.md` cites for `KernelPCA`). **No PyO3
  binding exists** for `RandomTreesEmbedding` (REQ-7 gap).

## Requirements

- REQ-1: **Param surface + defaults (R-DEV-2).** Constructor params + defaults
  match sklearn `get_params()` (`_forest.py:2818-2833`): `n_estimators=100`,
  `max_depth=5`, `min_samples_split=2`, `min_samples_leaf=1`,
  `min_weight_fraction_leaf=0.0`, `max_leaf_nodes=None`,
  `min_impurity_decrease=0.0`, `random_state=None`. **DIVERGENCE**: ferrolearn
  `new()` sets `n_estimators=10` (≠ sklearn `100`). ABSENT: `min_samples_leaf`,
  `min_weight_fraction_leaf`, `max_leaf_nodes`, `min_impurity_decrease`.
- REQ-2: **Totally-random per-tree build.** Each tree is built with purely
  random splits (one random feature per node via `max_features=1`, one uniform
  random threshold in the node's `[min,max)`), ignoring the target — mirroring
  the `ExtraTreeRegressor(splitter='random', max_features=1)` base estimator
  (`_forest.py:2834`, `:2816`). Exact forest at a seed = RNG boundary.
- REQ-3: **Transform one-hot contract.** `transform` produces the concatenated
  one-hot leaf encoding (`_forest.py:2982`): each output row has **exactly
  `n_estimators` ones**, the column count equals **`Σ_t leaves_t`**, and every
  entry is in `{0, 1}`. SHIPPED on the **dense** contract (the structural
  invariants are RNG-robust and hold for any seed/input).
- REQ-4: **`sparse_output` parameter + CSR output (R-DEV-3).** sklearn defaults
  `sparse_output=True` and returns a CSR matrix (`_forest.py:2828`, `:2928`);
  ferrolearn has no `sparse_output` param and `transform` returns a dense
  `Array2<F>`. NOT-STARTED.
- REQ-5: **Fit-on-random-y semantics (unsupervised).** sklearn fabricates
  `y = rnd.uniform(size=n_samples)` and fits the forest against it
  (`_forest.py:2924-2926`); ferrolearn's `Fit<Array2<F>, ()>` takes a unit
  target and ignores it — the observable leaf partitioning is identical because
  the random `y` only feeds leaf bookkeeping the embedding never reads (R-DEV-4).
- REQ-6: **`random_state` determinism (RNG boundary).** Same seed ⇒ identical
  ferrolearn forest ⇒ identical embedding; exact node/embedding parity with
  sklearn at a given seed is the documented numpy-MT-vs-StdRng RNG boundary, NOT
  a parity requirement.
- REQ-7: **PyO3 binding (`import ferrolearn` surface).** sklearn exposes
  `RandomTreesEmbedding` as a public transformer; ferrolearn has no
  `ferrolearn-python` binding for it. NOT-STARTED.
- REQ-8: **ferray substrate (R-SUBSTRATE).** Owned computation on `ferray-core`
  arrays + `ferray::random` for the threshold/feature draws, not `ndarray` +
  `rand`/`StdRng`; sparse output via ferray's sparse analog, not `sprs`.
  NOT-STARTED.

## Acceptance criteria

- AC-1: live `RandomTreesEmbedding().get_params()` equals the REQ-1 defaults for
  the exposed params; `n_estimators` is `100` (ferrolearn currently `10` — the
  flagged bug); absent params enumerated.
- AC-2: on a fixed-seed `fit`, the totally-random builder yields trees where
  every internal node splits on a single feature with a threshold strictly
  inside the node-local `[min,max)` (structure, not bit-parity vs sklearn).
- AC-3: for any `random_state`/input, `transform(X)` has `embedded.nrows() ==
  X.nrows()`, **every row sums to exactly `n_estimators`**, `embedded.ncols() ==
  fitted.n_output_features() == Σleaves`, and every entry ∈ `{0.0, 1.0}`
  (`test_fit_transform_basic`, `test_output_is_binary`,
  `test_total_leaves_matches_output_cols`).
- AC-4: ferrolearn `transform` returns a sparse CSR-equivalent matching
  `RandomTreesEmbedding(sparse_output=True).transform(X)` in nnz/shape
  (currently FAILS — dense only; NOT-STARTED).
- AC-5: `fit(X, &())` succeeds without any target argument; the resulting
  embedding row-sum/column contract (AC-3) is independent of any `y` (mirrors
  sklearn's ignored `y`).
- AC-6: two `fit` calls with the same `random_state` produce identical
  embeddings (`test_reproducibility`); exact sklearn-seed parity is declared
  RNG-boundary.
- AC-7: `import ferrolearn; ferrolearn.RandomTreesEmbedding` exists and mirrors
  `sklearn.ensemble.RandomTreesEmbedding` on the deterministic surface
  (currently FAILS — no binding; NOT-STARTED).
- AC-8: `rg "use ndarray|use rand|StdRng" random_trees_embedding.rs` is empty
  (ferray substrate — currently FAILS).

## REQ status table

Binary (R-DEFER-2). `RandomTreesEmbedding`/`FittedRandomTreesEmbedding` are
boundary transformer types re-exported at the crate root and wired into the
pipeline via `PipelineTransformer`/`FittedPipelineTransformer` (S5/R-DEFER-1
non-test consumer surface — the `ferrolearn-core` pipeline machinery consumes
that trait, the same surface `.design/core/pipeline.md` credits for `KernelPCA`).
Cites use symbol anchors (ferrolearn) / `file:line` (sklearn 1.5.2).

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (param surface + defaults) | NOT-STARTED | open prereq blocker #1841. `fn new` on `RandomTreesEmbedding` sets `n_estimators = 10`, but sklearn `__init__` defaults `n_estimators=100` (`_forest.py:2820`) — verified live `RandomTreesEmbedding().n_estimators == 100`. Wrong-default R-DEV-2 violation. Also ABSENT vs sklearn: `min_samples_leaf` (`:2824`), `min_weight_fraction_leaf`, `max_leaf_nodes`, `min_impurity_decrease` (#1842). The matched params (`max_depth=5`, `min_samples_split=2`, `random_state=None`) are correct, but the wrong `n_estimators` default makes the surface non-conforming as a whole. |
| REQ-2 (totally-random per-tree build) | SHIPPED | impl `fn build_random_tree` in `random_trees_embedding.rs` draws one random feature (`next_u64() % n_features`) and one uniform threshold in `[min,max)` (`fn random_threshold`) per node, ignoring the target — mirroring `ExtraTreeRegressor(splitter='random', max_features=1)` (`_forest.py:2834`, class `max_features=1` `:2816`). Consumer: `RandomTreesEmbedding::fit` → crate re-export + pipeline `fit_pipeline`. Tests: `test_fit_transform_basic`, `test_deeper_trees_more_leaves`, `test_unlimited_depth`, `test_single_sample`. Exact forest at a seed = documented RNG boundary (#1844). |
| REQ-3 (transform one-hot contract) | SHIPPED | impl `FittedRandomTreesEmbedding::transform` (`transform` in `random_trees_embedding.rs`) sets one `F::one()` per tree at `col_offset + leaf_pos`, concatenating across trees → mirrors sklearn `one_hot_encoder_.transform(self.apply(X))` (`_forest.py:2982`) on the structural contract. Consumer: crate re-export + `FittedPipelineTransformer::transform_pipeline` (pipeline machinery). Tests: `test_fit_transform_basic` (every row sums to `n_estimators`), `test_output_is_binary` (entries ∈ {0,1}), `test_total_leaves_matches_output_cols` (`ncols == n_output_features == Σleaves`). **Verified live**: `RandomTreesEmbedding(n_estimators=5,max_depth=3,random_state=42).fit_transform(X)` → CSR shape `(6,28)`, every row sum `== 5.0`, `28 == Σleaves` — the same structural invariants ferrolearn pins. RNG-robust; SHIPPED on the dense basis (the sparse dtype is REQ-4). |
| REQ-4 (sparse_output + CSR) | NOT-STARTED | open prereq blocker #1843. `transform` returns dense `Array2<F>` (`type Output = Array2<F>`); sklearn defaults `sparse_output=True` and returns CSR (`OneHotEncoder(sparse_output=self.sparse_output)`, `_forest.py:2928`; live `type(fit_transform(X)) == csr_matrix`). No `sparse_output` constructor param and no sparse path exist. |
| REQ-5 (fit-on-random-y semantics) | SHIPPED | impl: `Fit<Array2<F>, ()>` for `RandomTreesEmbedding` takes a unit `()` target; `fit` ignores `_y` and `build_random_tree` never reads any target (leaf values are `F::zero()`). This deviates correctly (R-DEV-4) from sklearn's `y = rnd.uniform(size=n_samples)` scaffold (`_forest.py:2924-2925`): because the embedding reads only *which* leaf a sample reaches, the fabricated random target is unobservable and elided. Consumer: crate re-export + pipeline `fit_pipeline` (which passes a `&Array1<F>` y that is likewise ignored). Tests: every fit test (`test_fit_transform_basic`, etc.) fits with `&()`. |
| REQ-6 (random_state determinism — RNG boundary) | SHIPPED | impl: `random_state: Option<u64>` seeds `StdRng::seed_from_u64` in `fit` (`StdRng::from_os_rng()` when `None`); same seed ⇒ identical feature/threshold stream ⇒ identical forest ⇒ identical embedding. Consumer: `with_random_state` on the re-exported type + pipeline. Test: `test_reproducibility` (two fits at seed 42 produce equal embeddings). **Documented RNG boundary**: exact embedding parity with sklearn at a given `random_state` is INFEASIBLE — numpy MT19937 (`rnd.uniform` + `ExtraTreeRegressor`'s `RandomSplitter`) vs Rust `StdRng` are different streams, the same boundary accepted for `extra_tree`/`random_forest`/SGD (#1844). Ships reproducibility, not cross-impl bit-parity. |
| REQ-7 (PyO3 binding) | NOT-STARTED | open prereq blocker #1845. No `RsRandomTreesEmbedding` in `ferrolearn-python/src/` (grep empty); `import ferrolearn` cannot mirror `sklearn.ensemble.RandomTreesEmbedding`. The library transformer + pipeline adapter exist, but the binding shim is absent. |
| REQ-8 (ferray substrate) | NOT-STARTED | open prereq blocker #1846. Module imports `use ndarray::{Array2, ArrayView1}` + `use rand::{SeedableRng, rngs::StdRng}` — wrong substrate per R-SUBSTRATE-1 (array → `ferray-core`, RNG → `ferray::random`, sparse output → ferray's sparse analog, not `sprs`). No `ferray` usage. |

## Architecture

`random_trees_embedding.rs` reuses the `Node<F>` enum from `decision_tree.rs`
(`Split { feature, threshold, left, right, .. }` / `Leaf { value, .. }`, a flat
`Vec<Node<F>>` with index-0 root) but implements its **own** totally-random
builder rather than the CART best-split scan. `RandomTreesEmbedding<F>` is the
unfitted boundary transformer (public fields + `with_*` builders + `Default`).
`fit` validates inputs (`InsufficientSamples` for empty `X`, `InvalidParameter`
for `n_estimators == 0` and `min_samples_split < 2`), seeds a single `StdRng`,
and loops `n_estimators` times: each iteration calls `build_random_tree` (random
feature → node-local min/max → uniform threshold in `[min,max)` → partition;
leaf on too-few-samples / max-depth / no-splittable-feature), then enumerates the
tree's leaf nodes into a `leaf_map` (node index → contiguous leaf position) and
accumulates `total_leaves`.

`FittedRandomTreesEmbedding<F>` stores `trees`, `leaf_counts`, `leaf_maps`,
`total_leaves`, `n_features`. `transform` allocates a dense
`(n_samples, total_leaves)` zero matrix, traverses each sample through each tree
(`traverse_tree`, `x[feature] <= threshold` → left), and writes `F::one()` at
the tree's `col_offset + leaf_pos`. The per-row invariant — exactly
`n_estimators` ones — holds because every sample reaches exactly one leaf per
tree; the column invariant — `Σ leaf_counts` — holds because each tree
contributes its full leaf enumeration. These are the RNG-robust structural
guarantees that make REQ-3 SHIPPED without numpy parity.

The exact forest at a `random_state` is NOT reproducible against sklearn: the
random target (`rnd.uniform`), the per-node feature pick, and the threshold all
come from numpy MT19937 in sklearn vs `StdRng` here — the documented RNG
boundary, consistent with `extra_tree`/`random_forest`/`decision_tree`
`splitter='random'`.

Pipeline integration: `PipelineTransformer<F>::fit_pipeline` boxes a
`FittedRandomTreesEmbedding<F>` (ignoring the passed `y`),
`FittedPipelineTransformer<F>::transform_pipeline` forwards to `transform` — the
non-test consumer surface for R-DEFER-1.

## Verification

Library crate (green at baseline `c4fe2b21`):
```
cargo test -p ferrolearn-tree --lib random_trees_embedding   # 15 passed; 0 failed
cargo clippy -p ferrolearn-tree --all-targets -- -D warnings
cargo fmt --all --check
```
Live sklearn oracle (installed 1.5.2, run from `/tmp`):
```
# REQ-1 defaults (n_estimators == 100, NOT 10) + sparse_output True
python3 -c "from sklearn.ensemble import RandomTreesEmbedding as E; p=E().get_params(); print({k:p[k] for k in ['n_estimators','max_depth','min_samples_split','min_samples_leaf','min_weight_fraction_leaf','max_leaf_nodes','min_impurity_decrease','sparse_output','random_state']})"
# REQ-3 structural one-hot contract + REQ-4 CSR dtype divergence
python3 -c "import numpy as np; from sklearn.ensemble import RandomTreesEmbedding as E; X=np.array([[1.,2.],[2.,3.],[3.,3.],[5.,6.],[6.,7.],[7.,8.]]); o=E(n_estimators=5,max_depth=3,random_state=42).fit_transform(X); print(type(o).__name__, o.shape, np.asarray(o.sum(1)).ravel().tolist())"
# -> csr_matrix (6, 28) [5.0,5.0,5.0,5.0,5.0,5.0]
```
The NOT-STARTED REQs (1, 4, 7, 8) have no green verification by construction —
each carries an open prereq blocker. REQ-2/3/5/6 are verified by the in-crate
`#[test]`s named above (deterministic structural invariants, or ferrolearn-internal
reproducibility); the structural one-hot contract (REQ-3) is the RNG-robust pin
that survives the embedding's randomness. A characterization pin for REQ-1
(R-CHAR-3, `n_estimators == 100`) belongs in
`ferrolearn-tree/tests/divergence_random_trees_embedding.rs` once the fixer
corrects the default.

## Blockers to open

(Suggested numbers; the director creates the real crosslink issues.)

- **#1841** (top — the wrong-default fix the fixer applies) — REQ-1:
  `RandomTreesEmbedding::new` sets `n_estimators = 10`; sklearn defaults `100`
  (`_forest.py:2820`). One-field change in `fn new` + update `test_default`'s
  `assert_eq!(model.n_estimators, 10)` and the doc-comment default. R-DEV-2.
- **#1842** — REQ-1: missing constructor params `min_samples_leaf` (`:2824`),
  `min_weight_fraction_leaf`, `max_leaf_nodes`, `min_impurity_decrease` vs
  sklearn `_forest.py:2818-2833`.
- **#1843** — REQ-4: `sparse_output` parameter + CSR output path; `transform`
  returns dense `Array2<F>` where sklearn returns a CSR `csr_matrix`
  (`_forest.py:2928`). Requires a sparse return type (ferray sparse analog per
  R-SUBSTRATE).
- **#1844** — REQ-2/REQ-6: exact embedding parity at `random_state` is a
  numpy-MT (`rnd.uniform` + `RandomSplitter`) vs `StdRng` RNG boundary
  (documented like `extra_tree`/`random_forest`, NOT a fixable divergence).
- **#1845** — REQ-7: add a `ferrolearn-python` `RsRandomTreesEmbedding` binding
  mirroring `sklearn.ensemble.RandomTreesEmbedding` (`fit`/`transform`/
  `fit_transform`, defaults, attributes), registered in `ferrolearn-python/src/lib.rs`.
- **#1846** — REQ-8: migrate `random_trees_embedding.rs` off `ndarray`/`rand`
  (`StdRng`) to the ferray substrate (array → `ferray-core`, RNG →
  `ferray::random`, sparse output → ferray sparse), jointly with the tree-crate
  substrate migration.

## Top 3 divergences (director / critic's first pins)

1. **`n_estimators` default 10 ≠ 100 (#1841) — the wrong-default bug.**
   `fn new` sets `n_estimators = 10`; sklearn `__init__` defaults `100`
   (`_forest.py:2820`, live `RandomTreesEmbedding().n_estimators == 100`).
   R-DEV-2 ABI divergence — a one-field fix in `fn new` plus the `test_default`
   assertion and doc-comment update.
2. **Dense vs sparse output (#1843).** `transform` returns a dense `Array2<F>`;
   sklearn's `sparse_output=True` default returns a CSR matrix
   (`_forest.py:2928`, live `csr_matrix`). ferrolearn has no `sparse_output`
   param at all — a missing parameter *and* a wrong output dtype (R-DEV-3).
3. **Missing per-leaf / leaf-node params (#1842).** `min_samples_leaf`,
   `min_weight_fraction_leaf`, `max_leaf_nodes`, `min_impurity_decrease` are
   absent; ferrolearn's builder enforces only `min_samples_split`, so the tree
   geometry (and thus the embedding dimensionality) diverges from sklearn under
   non-default settings.
