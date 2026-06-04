# TSNE (sklearn.manifold.TSNE)

<!--
tier: 3-component
status: shipped-partial
baseline-commit: 2d8b9c24
upstream: scikit-learn 1.5.2
upstream-paths:
  - sklearn/manifold/_t_sne.py  # class TSNE(ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator) (:563). ctor (:825-860): n_components=2, *, perplexity=30.0, early_exaggeration=12.0, learning_rate="auto", max_iter=None(→1000 :1170-1171), n_iter_without_progress=300, min_grad_norm=1e-7, metric="euclidean", metric_params=None, init="pca", verbose=0, random_state=None, method="barnes_hut", angle=0.5, n_jobs=None, n_iter="deprecated". _parameter_constraints (:791-817): n_components Interval[1,inf), perplexity (0,inf), early_exaggeration [1,inf), learning_rate {"auto"}|(0,inf), max_iter [250,inf)|None, init {"pca","random"}|ndarray, method {"barnes_hut","exact"}, angle [0,1]. learning_rate=='auto' -> learning_rate_=max(N/early_exaggeration/4, 50) (_fit :876-879). _joint_probabilities (:41-71): conditional_P via _utils._binary_search_perplexity (:65), P=conditional_P+conditional_P.T (:68), sum_P=max(sum(P),MACHINE_EPSILON) (:69), P=max(squareform(P)/sum_P, MACHINE_EPSILON) (:70) — DETERMINISTIC given distances. _fit (:866-1051): learning_rate_ (:876-882), barnes_hut requires n_components<4 (:917-922), exact path pairwise_distances+_joint_probabilities (:928-964) / barnes_hut path knn+_joint_probabilities_nn (:966-1015), init: pca PCA(svd_solver=randomized) rescaled std(PC1)=1e-4 (:1019-1030) / random 1e-4*standard_normal (:1031-1036), degrees_of_freedom=max(n_components-1,1) (:1042), _tsne (:1044-1051). _tsne / _gradient_descent (:304-447, :1053-1126): early-exaggeration phase _EXPLORATION_MAX_ITER=250 P*=early_exaggeration (:820,:1095), momentum 0.5 (:1080) then 0.8 (:1110); _kl_divergence (:131) / _kl_divergence_bh (:208) gradient; gains inc 0.2 / dec *0.8 min_gain=0.01 (:407-409); early stop on n_iter_without_progress (:431) or grad_norm<=min_grad_norm (:439); n_iter_=it (:1115), kl_divergence_ (:1124). fit_transform (:1132-1178): max_iter None->1000 (:1170-1171), _check_params_vs_input perplexity<n_samples (:862-864), embedding_=_fit(X) (:1176-1177). fit (:1184-1206): self.fit_transform(X); return self. fitted attrs: embedding_, kl_divergence_, n_iter_, learning_rate_, n_features_in_. NO transform / inverse_transform (t-SNE has no out-of-sample projection).
ferrolearn-module: ferrolearn-decomp/src/tsne.rs
parity-ops: TSNE
crosslink-issue: 1596
-->

## Summary

`ferrolearn-decomp/src/tsne.rs` mirrors scikit-learn's `TSNE`
(`sklearn/manifold/_t_sne.py`, `class TSNE` `:563`): non-linear dimensionality
reduction that models pairwise high-dimensional similarities as conditional
probabilities (Gaussian kernel, per-point sigma tuned to a target perplexity),
then finds a low-dimensional embedding minimising the Kullback-Leibler divergence
between those affinities and a Student-t kernel in the embedding via non-convex
gradient descent (early-exaggeration + momentum + Barnes-Hut tree repulsion). The
exposed surface is the unfitted `Tsne { n_components (2), perplexity (30.0),
learning_rate (200.0 FIXED), n_iter (1000), early_exaggeration (12.0), theta
(0.5), random_state }` (`tsne.rs`, struct at line 59; `new()` at line 83,
builders `with_n_components`/`with_perplexity`/`with_learning_rate`/`with_n_iter`/
`with_early_exaggeration`/`with_theta`/`with_random_state` at lines 96-142,
accessors at lines 144-184) and the fitted `FittedTsne { embedding_
(n_samples, n_components), kl_divergence_, n_iter_ }` (`tsne.rs`, struct at line
203, accessors `embedding()`/`kl_divergence()`/`n_iter()` at lines 212-229),
re-exported at the crate root (`pub use tsne::{FittedTsne, Tsne}`, `lib.rs:102`).
Like sklearn, `FittedTsne` implements NO `Transform` (t-SNE has no out-of-sample
projection — module doc `tsne.rs:27-28`). `Tsne` is NOT generic — it is f64-only
(`Tsne` has no `<F>` parameter; `Fit<Array2<f64>, ()>` `tsne.rs:538`). There is NO
PyO3 binding (a `grep -rn "TSNE\|Tsne" ferrolearn-python/src/` is empty) and NO
`tests/divergence_tsne.rs`.

**EXACT `embedding_` VALUE PARITY DIVERGES (R-HONEST-3, REQ-4 NOT-STARTED,
CARVE-OUT `#1597`).** The t-SNE objective is explicitly NON-CONVEX (sklearn
docstring `_t_sne.py:570-571`: "t-SNE has a cost function that is not convex,
i.e. with different initializations we can get different results"), so the
embedding is identifiable only up to ROTATION / REFLECTION / TRANSLATION and the
particular LOCAL MINIMUM reached. FOUR independent factors make ferrolearn's
`embedding_` diverge element-wise from sklearn's: (a) **init** — sklearn defaults
`init='pca'` (`_t_sne.py:837`, a DETERMINISTIC PCA of `X` rescaled so `std(PC1)=1e-4`
`:1019-1030`), whereas ferrolearn ALWAYS uses a random Gaussian init
(`Normal(0, 1e-4)` via `Xoshiro256PlusPlus` seeded `random_state.unwrap_or(0)`
`tsne.rs:601-608`) — a different starting point AND a different RNG (Xoshiro vs
numpy) than even sklearn's `init='random'` (`1e-4 * random_state.standard_normal`
`_t_sne.py:1031-1036`); (b) **non-convexity** — different starting points + the
local-minima structure give different embeddings; (c) **`learning_rate`** —
ferrolearn fixes `200.0` (`tsne.rs:64`,`:87`) whereas sklearn defaults
`'auto' = max(N/early_exaggeration/4, 50)` (`_t_sne.py:876-879`, Probe 2:
`learning_rate_ = 50.0` for `N=100`); (d) **Barnes-Hut approximation details** —
ferrolearn's k-d-tree centre-of-mass repulsion (`BHTree` `tsne.rs:242`) is a
DENSE-P, custom-tree variant, not sklearn's k-NN-sparse-`P` Cython
`_barnes_hut_tsne.gradient` (`_t_sne.py:284`). Same class as the fast_ica / lda /
minibatch_nmf RNG/local-optimum carve-outs; no failing test is asserted
(R-DEFER-3). **The meaningful structural-correctness check (like fast_ica's
source-recovery): on well-separated clusters the embedding SEPARATES the clusters
(inter-centroid distance ≫ intra-cluster spread / high k-NN neighbour
preservation) — testable structurally, NOT by exact values** (Probe 1; pinned by
`test_tsne_separates_clusters` k-NN accuracy `> 0.8`).

**The DETERMINISTIC PARITY CANDIDATE the critic should weigh (FLAG): the
high-dim affinity matrix `P`** (the perplexity binary search + symmetrisation,
sklearn `_joint_probabilities` `_t_sne.py:41-71`) is DETERMINISTIC given `X` (no
RNG; Probe 3: `P` reproducible, full matrix sums to 1, all entries clamped `> 0`).
If ferrolearn exposed `P` it should match sklearn's `_joint_probabilities`. But
ferrolearn does NOT expose `P` — `compute_joint_probabilities` (`tsne.rs:467`) is
a private `fn` internal to `fit`, and there is NO public accessor, so a direct
`P`-parity test is NOT possible through the public API (REQ-2 ships the
STRUCTURAL valid-probability properties — symmetry / sum-to-1 / positivity — that
ARE observable through `fit`, and flags exact `P` parity as a critic question).
NOTE additionally that ferrolearn's symmetrisation `P_ij = (P_i|j + P_j|i)/(2n)`
(`tsne.rs:16` doc, `:479-483`) divides by `2n` while sklearn divides by
`sum(P) = 2` after symmetrisation (`_t_sne.py:69-70`) — they coincide only when
each conditional row sums to 1 (which the binary search enforces), a candidate
divergence for the critic to pin.

As of this iteration: the STRUCTURAL embedding shape `(n_samples, n_components)`,
the perplexity binary-search affinities producing a valid symmetric
probability structure, cluster SEPARATION on well-separated data, finite
`kl_divergence_ ≥ 0`, `n_iter_` recorded, the early-exaggeration (250) + momentum
(0.5/0.8) + Barnes-Hut/exact gradient-descent run, the error / parameter
contracts, and determinism given a seed (REQ-1,2,3,4-structural-via-cluster-sep)
are SHIPPED scoped; exact `embedding_` value parity (REQ-5, CARVE-OUT `#1597`),
`init='pca'` default + `init='random'` `1e-4*standard_normal` scaling (REQ-6,
`#1598`), `learning_rate='auto'` default (REQ-7, `#1599`),
`n_iter_without_progress` / `min_grad_norm` early stopping (REQ-8, `#1600`),
`metric` (precomputed / non-euclidean) + `metric_params` (REQ-9, `#1601`),
`method='exact'` vs barnes_hut-only + the exact-`P` / `sum(P)` normalisation
(REQ-10, `#1602`), the `n_iter`→`max_iter` rename + `n_iter` deprecation + the
`max_iter≥250` constraint (REQ-11, `#1603`), fitted attrs `learning_rate_` /
`n_features_in_` (REQ-12, `#1604`), generic `F` / f32 support (REQ-13, `#1605`),
the PyO3 binding (REQ-14, `#1606`), and the ferray substrate (REQ-15, `#1607`)
are NOT-STARTED — **4 SHIPPED / 11 NOT-STARTED**.

`Tsne` / `FittedTsne` are existing pub APIs whose non-test consumer is the crate
re-export (`lib.rs:102`, boundary public API, grandfathered S5/R-DEFER-1). There
is NO PyO3 binding (REQ-14 NOT-STARTED) and NO `Transform` impl (matches sklearn).

## Probes (live sklearn oracle, 1.5.2, run from /tmp)

```bash
# PROBE 1 (REQ-1/4 SHIPPED scoped) — the MEANINGFUL "did t-SNE work" check:
# on two well-separated clusters the embedding SEPARATES them (inter-centroid
# distance >> intra-cluster spread). STRUCTURAL — do NOT pin exact embedding
# values (REQ-5 CARVE-OUT). VALUES generated by sklearn, never copied from
# ferrolearn (R-CHAR-3).
python3 -c "
import numpy as np
from sklearn.manifold import TSNE
c0 = np.zeros((8,4)); c1 = np.ones((8,4))*10.0
X = np.vstack([c0,c1]) + 0.01*np.arange(64).reshape(16,4)
emb = TSNE(n_components=2, perplexity=5, random_state=0, init='random', learning_rate='auto').fit_transform(X)
print('emb shape', emb.shape)
c0m, c1m = emb[:8].mean(0), emb[8:].mean(0)
inter = np.linalg.norm(c0m-c1m)
intra = (np.linalg.norm(emb[:8]-c0m,axis=1).mean() + np.linalg.norm(emb[8:]-c1m,axis=1).mean())/2
print('inter-centroid', round(float(inter),4), ' mean intra spread', round(float(intra),4))
print('clusters separate (inter >> intra):', bool(inter > 3*intra))"
# -> emb shape (16, 2)
# -> inter-centroid 225.9637  mean intra spread 1.565   (exact VALUES NOT reproduced, REQ-5)
# -> clusters separate (inter >> intra): True   => structural separation, the real t-SNE check (REQ-1)

# PROBE 2 (REQ-6/7/11/12 NOT-STARTED) — ctor defaults + learning_rate_='auto' + fitted attrs.
python3 -c "
from sklearn.manifold import TSNE
m=TSNE()
for p in ['n_components','perplexity','early_exaggeration','learning_rate','max_iter','n_iter_without_progress','min_grad_norm','metric','metric_params','init','method','angle','n_iter']:
    print(f'{p} =', getattr(m,p))"
# -> n_components=2 perplexity=30.0 early_exaggeration=12.0 learning_rate=auto max_iter=None
# -> n_iter_without_progress=300 min_grad_norm=1e-07 metric=euclidean metric_params=None
# -> init=pca method=barnes_hut angle=0.5 n_iter=deprecated
#    => init DEFAULT='pca' (ferrolearn always random, REQ-6); learning_rate DEFAULT='auto'
#       (ferrolearn fixed 200.0, REQ-7); max_iter None->1000 (REQ-11); n_iter deprecated (REQ-11).
python3 -c "
import numpy as np
from sklearn.manifold import TSNE
X=np.random.RandomState(0).rand(100,5)
m=TSNE(perplexity=5, random_state=0).fit(X)
print('learning_rate_', m.learning_rate_, '(= max(N/early_exag/4, 50) = max(100/12/4, 50))')
print('n_iter_', m.n_iter_, ' kl_divergence_', round(float(m.kl_divergence_),4), ' n_features_in_', m.n_features_in_)"
# -> learning_rate_ 50.0 (= max(N/early_exag/4, 50))   => REQ-7 (ferrolearn has no learning_rate_, REQ-12)
# -> n_iter_ 999  kl_divergence_ 0.5904  n_features_in_ 5   => n_features_in_ absent in ferrolearn (REQ-12)

# PROBE 3 (DETERMINISTIC-P FLAG for the critic; REQ-2 structural / REQ-10) — _joint_probabilities
# is DETERMINISTIC given the distance matrix: symmetric joint P sums to 1, all entries > 0
# (clamped to MACHINE_EPSILON). ferrolearn does NOT expose P, so only its STRUCTURAL properties
# (symmetry/sum-to-1/positivity) are observable, NOT element-wise P parity.
python3 -c "
import numpy as np
from sklearn.manifold._t_sne import _joint_probabilities
from scipy.spatial.distance import pdist, squareform
X=np.array([[0.,0],[1,0],[0,1],[5,5]])
D=squareform(pdist(X,'sqeuclidean')).astype(np.float32)
P=_joint_probabilities(D, 2.0, 0)
print('P condensed (n*(n-1)/2=6):', np.round(P,6).tolist())
print('symmetric joint sums to 1:', round(float(2*P.sum()),6), ' all > 0:', bool((P>0).all()))
print('deterministic (re-run identical):', bool(np.array_equal(P,_joint_probabilities(D,2.0,0))))"
# -> P condensed [0.130442, 0.130442, 0.0, 0.113994, 0.06256, 0.06256]
# -> symmetric joint sums to 1: 1.0  all > 0: True    => deterministic-given-X P (critic parity candidate)
# -> deterministic (re-run identical): True
#    NOTE sklearn normalises P /= sum(P) (= 2 after symmetrisation, :69-70); ferrolearn /= 2n
#    (tsne.rs:479-483) — coincide when conditional rows sum to 1, a critic divergence candidate.

# PROBE 4 (REQ-10/11 NOT-STARTED) — barnes_hut requires n_components<4; max_iter>=250.
python3 -c "
import numpy as np
from sklearn.manifold import TSNE
X=np.random.RandomState(0).rand(20,6)
try: TSNE(n_components=4, perplexity=5).fit(X); print('no error')
except ValueError as e: print('n_components=4 barnes_hut:', str(e)[:55])
try: TSNE(max_iter=50, perplexity=5).fit(X); print('no error')
except Exception as e: print('max_iter=50:', type(e).__name__)"
# -> n_components=4 barnes_hut: 'n_components' should be inferior to 4 for the
# -> max_iter=50: InvalidParameterError     => sklearn enforces max_iter>=250 (REQ-11); ferrolearn does not
```

## Requirements

- REQ-1: **Structural: embedding shape `(n_samples, n_components)`, early-exaggeration
  (250) + momentum (0.5/0.8) + Barnes-Hut/exact gradient descent runs, determinism
  given a seed (SHIPPED scoped).** `fn fit` (`Fit<Array2<f64>, ()>` for `Tsne`,
  impl at `tsne.rs:538`) computes joint probabilities (`compute_joint_probabilities`
  `tsne.rs:467`), inits a `(n, dim)` embedding from a seeded
  `Xoshiro256PlusPlus` + `Normal(0, 1e-4)` (`tsne.rs:601-608`), then runs the
  gradient-descent loop to `n_iter` (`tsne.rs:618`): early-exaggeration for the
  first `min(250, n_iter)` iterations (`early_exag_end` `tsne.rs:614`,
  `exaggeration` `tsne.rs:620-624`, mirroring `_EXPLORATION_MAX_ITER=250`
  `_t_sne.py:820` and `P *= early_exaggeration` `:1095`), momentum `0.5` early /
  `0.8` later (`tsne.rs:619`, mirroring `_t_sne.py:1080`,`:1110`), adaptive gains
  (inc `+0.2` / dec `*0.8`, floor `0.01` `tsne.rs:643-647`, mirroring `_t_sne.py:407-409`),
  velocity update `v = momentum*v − lr*gain*g` (`tsne.rs:648`, mirroring `:411`),
  and per-axis centring (`tsne.rs:653-659`). The gradient is computed by the
  Barnes-Hut tree (`Tsne::bh_gradient` `tsne.rs:674`, attractive over non-zero `P` +
  repulsive `BHTree::compute_repulsive` `tsne.rs:345` — sklearn's `_kl_divergence_bh`
  `_t_sne.py:208`) when `theta > 0 && dim <= 3` (`use_bh` `tsne.rs:616`), else the
  exact `exact_gradient` (`tsne.rs:759`, `4·(p_ij − q_ij)·inv·(y_i − y_j)` —
  sklearn's `_kl_divergence` `_t_sne.py:131`). The embedding shape is `(n, dim)`
  (`tsne.rs:605`). The seeded RNG (`tsne.rs:601-602`) + deterministic gradient
  descent make the fit reproducible given a seed. **Scope: STRUCTURAL (shape /
  determinism / the run), NOT exact embedding VALUES (REQ-5).** Pinned by
  `test_tsne_basic_shape` `(20,2)`, `test_tsne_3d_embedding` `ncols==3`,
  `test_tsne_exact_mode` (`theta=0`, `(10,2)`), `test_tsne_reproducibility`
  (two seeded fits identical to `1e-10`), `test_tsne_n_iter_recorded`. Non-test
  consumer: re-export `lib.rs:102`.

- REQ-2: **Structural: perplexity binary-search affinities produce a valid
  symmetric joint-probability structure (SHIPPED scoped).** `compute_pij_row`
  (`tsne.rs:410`) binary-searches `beta = 1/(2σ²)` per point to match the target
  perplexity's Shannon entropy `ln(perplexity)` to `1e-5` (`tsne.rs:413`,`:449-460`,
  100 iterations), mirroring sklearn's `_binary_search_perplexity`
  (`_t_sne.py:65`). `compute_joint_probabilities` (`tsne.rs:467`) symmetrises
  `P_ij = (P_i|j + P_j|i)/(2n)` (`tsne.rs:479-483`, doc `:16`), clamps each entry to
  `≥ 1e-12` (`tsne.rs:485`, mirroring sklearn's `MACHINE_EPSILON` clamp
  `_t_sne.py:70`), and zeroes the diagonal (`tsne.rs:489`). Probe 3 confirms
  sklearn's `_joint_probabilities` produces a symmetric joint summing to 1 with all
  entries `> 0` — the STRUCTURAL valid-probability property ferrolearn's affinities
  satisfy and that drives the attractive forces in `bh_gradient`/`exact_gradient`.
  **Scope: STRUCTURAL (symmetry / positivity / per-point perplexity tuning) — NOT
  exact element-wise `P` parity, which is NOT observable through the public API
  (`P` is internal to `fit`, no accessor; see REQ-10 / the FLAG).** Exercised
  end-to-end by the cluster-separation and shape tests (the affinities are the
  attractive-force input). Non-test consumer: re-export `lib.rs:102`.

- REQ-3: **Error / parameter contracts (SHIPPED scoped).** `fn fit` (`tsne.rs:538`)
  returns `InvalidParameter { name: "n_components" }` for `n_components == 0`
  (`tsne.rs:554-559`), `InsufficientSamples { required: 2 }` for `n_samples < 2`
  (`tsne.rs:560-566`), `InvalidParameter { name: "perplexity" }` for
  `perplexity <= 0.0` (`tsne.rs:567-572`) and for `perplexity >= n_samples`
  (`tsne.rs:573-581`, mirroring sklearn's `_check_params_vs_input`
  "perplexity must be less than n_samples" `_t_sne.py:862-864`),
  `InvalidParameter { name: "learning_rate" }` for `learning_rate <= 0.0`
  (`tsne.rs:582-587`), and `InvalidParameter { name: "theta" }` for `theta < 0.0`
  (`tsne.rs:588-593`). Pinned by `test_tsne_invalid_n_components_zero`,
  `test_tsne_insufficient_samples`, `test_tsne_invalid_perplexity_zero`,
  `test_tsne_perplexity_too_large`, `test_tsne_invalid_learning_rate`,
  `test_tsne_invalid_theta`. **FLAG (candidate DIVs):** sklearn validates via
  `_parameter_constraints` (`_t_sne.py:791-817`) raising `InvalidParameterError`
  /`ValueError`, NOT `FerroError`; sklearn enforces `early_exaggeration ∈ [1,inf)`
  (`:794`), `max_iter ∈ [250,inf)` (`:799`, Probe 4), `angle ∈ [0,1]` (`:811`), and
  `n_components < 4` for `method='barnes_hut'` (`:917-922`, Probe 4) — ferrolearn
  has no such guards (`theta`/`angle` is only checked `≥ 0`, and `n_components` is
  unbounded above). Non-test consumer: re-export `lib.rs:102`.

- REQ-4: **Structural: cluster SEPARATION on well-separated data + finite
  `kl_divergence_ ≥ 0` + `n_iter_` recorded (SHIPPED scoped — the meaningful
  "did t-SNE work" check).** The whole fit pipeline produces an embedding in which
  well-separated input clusters remain separated: `test_tsne_separates_clusters`
  fits 3 blobs (`make_blobs` `tsne.rs:815`) and asserts a 3-NN label-recovery
  accuracy `> 0.8` (`tsne.rs:899-905`) — the neighbour-preservation property that is
  the real correctness signal for t-SNE (analogous to fast_ica's source-recovery).
  Probe 1 confirms sklearn's embedding separates two well-separated clusters
  (inter-centroid distance `≫` intra-cluster spread). `compute_kl_divergence`
  (`tsne.rs:496`) computes `KL(P||Q) = Σ p_ij·ln(p_ij/q_ij)` over the Student-t
  `Q = 1/(1+||y_i−y_j||²)` normalised by `Z` (`tsne.rs:508`,`:524-527`, mirroring
  sklearn's `2·P·log(P/Q)` objective `_t_sne.py:184-191`) and stores it as
  `kl_divergence_` (`tsne.rs:662-666`); `n_iter_` is recorded (`tsne.rs:667`).
  Pinned by `test_tsne_separates_clusters` (k-NN accuracy `> 0.8`),
  `test_tsne_kl_divergence_non_negative` (`kl_divergence() >= 0`),
  `test_tsne_n_iter_recorded` (`n_iter() == 50`). **Scope: STRUCTURAL (cluster
  separation / KL finiteness / iteration count), NOT exact embedding VALUES (REQ-5)
  nor the exact `kl_divergence_` value (which depends on the local optimum reached,
  REQ-5).** Non-test consumer: re-export `lib.rs:102`.

- REQ-5: **EXACT `embedding_` (and `kl_divergence_`) value parity with sklearn
  (NOT-STARTED, CARVE-OUT; `#1597`).** The t-SNE objective is NON-CONVEX
  (`_t_sne.py:570-571`), so the embedding is identifiable only up to
  rotation/reflection/translation and the local minimum reached. FOUR factors make
  ferrolearn's `embedding_` diverge element-wise: (a) **init** — sklearn defaults
  `init='pca'` (DETERMINISTIC PCA rescaled `std(PC1)=1e-4` `_t_sne.py:1019-1030`),
  ferrolearn ALWAYS uses random Gaussian `Normal(0, 1e-4)` via `Xoshiro256PlusPlus`
  (`tsne.rs:601-608`) — a different start AND a different PRNG than even sklearn's
  `init='random'` (`1e-4*standard_normal` numpy `_t_sne.py:1031-1036`);
  (b) **non-convexity** — different starts → different local minima; (c)
  **`learning_rate`** — ferrolearn fixes `200.0` (`tsne.rs:87`), sklearn defaults
  `'auto' = max(N/early_exaggeration/4, 50)` (`_t_sne.py:876-879`, Probe 2 `50.0`);
  (d) **Barnes-Hut** — ferrolearn's dense-`P` custom k-d tree (`BHTree`
  `tsne.rs:242`) vs sklearn's k-NN-sparse-`P` Cython `_barnes_hut_tsne.gradient`
  (`_t_sne.py:284`). **CARVE-OUT (R-DEFER-3):** element-wise parity requires PCA
  init + the numpy `RandomState` + `learning_rate='auto'` + the sparse-`P`
  Barnes-Hut AND a fixed rotation/reflection/translation alignment of the
  non-convex optimum; no failing test is asserted (same class as the fast_ica /
  lda / minibatch_nmf RNG/local-optimum carve-outs). The observable, meaningful
  check is cluster separation (Probe 1, REQ-4).

- REQ-6: **`init='pca'` default + `init='random'` `1e-4*standard_normal` scaling +
  init as an `ndarray` (NOT-STARTED; `#1598`).** sklearn defaults `init='pca'`
  (`_t_sne.py:837`) — a randomized-PCA embedding rescaled so `std(PC1)=1e-4`
  (`:1019-1030`) — and also accepts `init='random'` (`1e-4 * random_state
  .standard_normal(size=(n, n_components))` numpy `:1031-1036`) or an explicit
  `ndarray` (`:1017-1018`); the choice is a `StrOptions({"pca","random"})|np.ndarray`
  constraint (`:804-807`). ferrolearn's `Tsne` (`tsne.rs:59`) has NO `init` field —
  it ALWAYS draws a random Gaussian `Normal(0, 1e-4)` from a seeded
  `Xoshiro256PlusPlus` (`tsne.rs:601-608`), i.e. neither the PCA default nor the
  numpy-`standard_normal` random variant nor a user-supplied init.

- REQ-7: **`learning_rate='auto'` default (NOT-STARTED; `#1599`).** sklearn defaults
  `learning_rate='auto'` (`_t_sne.py:831`) → `learning_rate_ = max(N /
  early_exaggeration / 4, 50)` (`:876-879`, Probe 2: `50.0` for `N=100`,
  `early_exaggeration=12`), with a numeric `(0, inf)` alternative
  (`_parameter_constraints` `:795-798`). ferrolearn's `Tsne.learning_rate`
  (`tsne.rs:64`) defaults to a FIXED `200.0` (`new()` `tsne.rs:87`) with no `'auto'`
  computation and no `learning_rate_` attribute (REQ-12).

- REQ-8: **`n_iter_without_progress` / `min_grad_norm` early stopping
  (NOT-STARTED; `#1600`).** sklearn's `_gradient_descent` (`_t_sne.py:304-447`)
  early-stops when the error fails to improve for more than `n_iter_without_progress`
  iterations (default 300, `:431`; the exploration phase uses
  `_EXPLORATION_MAX_ITER=250` `:1078`) OR when `grad_norm <= min_grad_norm`
  (default `1e-7`, `:439`), checked every `_N_ITER_CHECK=50` iterations
  (`:399`,`:823`). ferrolearn's gradient-descent loop (`tsne.rs:618-660`) runs a
  FIXED `n_iter` iterations with NO progress/grad-norm early stop and NO
  `n_iter_without_progress` / `min_grad_norm` fields — `n_iter_` is therefore
  always exactly the configured `n_iter` (`tsne.rs:667`), not the stopping iteration.

- REQ-9: **`metric` (precomputed / non-euclidean) + `metric_params`
  (NOT-STARTED; `#1601`).** sklearn's `metric` (default `'euclidean'`,
  `_t_sne.py:835`) accepts any scipy/pairwise metric name, a callable, or
  `'precomputed'` (X is a square distance matrix, `:894-900`,`:931-932`), plus
  `metric_params` (`:836`,`:945-947`); the affinities are computed from those
  distances. ferrolearn's `compute_joint_probabilities` (`tsne.rs:467`) ALWAYS uses
  squared Euclidean distances (`pairwise_sq_distances` `tsne.rs:389`) — there is no
  `metric` / `metric_params` field, no precomputed-distance path, and no
  non-euclidean metric.

- REQ-10: **`method='exact'` vs barnes_hut-only + the exact-`P` / `sum(P)`
  normalisation (NOT-STARTED; `#1602`).** sklearn's `method` (default
  `'barnes_hut'`, `_t_sne.py:840`) selects either the k-NN-sparse-`P`
  Barnes-Hut path (`_joint_probabilities_nn` `:74` + `_kl_divergence_bh` `:208`,
  Cython tree) or `method='exact'` (dense `_joint_probabilities` `:41` +
  `_kl_divergence` `:131`, O(N²)), a `StrOptions({"barnes_hut","exact"})` constraint
  (`:810`). ferrolearn has NO `method` field: it ALWAYS computes the DENSE joint `P`
  (`compute_joint_probabilities` `tsne.rs:467`) and selects the tree-vs-exact
  GRADIENT internally via `use_bh = theta > 0 && dim <= 3` (`tsne.rs:616`) — i.e.
  `theta=0` gives the exact gradient but still over a dense `P`, not sklearn's
  sparse-`P` barnes_hut. **Additionally** ferrolearn's symmetrisation divides by
  `2n` (`tsne.rs:480-483`) whereas sklearn divides by `sum(P)` (= 2 after
  symmetrisation, `_t_sne.py:69-70`); they coincide only when each conditional row
  sums to 1, a candidate `P`-normalisation divergence (FLAG; Probe 3).

- REQ-11: **`n_iter`→`max_iter` rename + `n_iter` deprecation + `max_iter ≥ 250`
  constraint + `max_iter=None→1000` (NOT-STARTED; `#1603`).** sklearn renamed the
  iteration param `n_iter`→`max_iter` in 1.5 (`_t_sne.py:621-626`,`:825`,`:843`),
  defaults `max_iter=None` → `1000` (`:832`,`:1170-1171`), enforces
  `max_iter ∈ [250, inf)` (`_parameter_constraints` `:799`, Probe 4), keeps a
  deprecated `n_iter` that `FutureWarning`s and errors if set together with
  `max_iter` (`:1153-1169`). ferrolearn's field is named `n_iter` (default 1000,
  `tsne.rs:67`,`:88`, builder `with_n_iter` `:118`) with NO `max_iter` alias, NO
  deprecation, and NO `≥ 250` lower-bound constraint (any `usize` is accepted).

- REQ-12: **Fitted attrs `learning_rate_` / `n_features_in_` (NOT-STARTED;
  `#1604`).** sklearn exposes `embedding_`, `kl_divergence_`, `n_iter_`,
  `learning_rate_` (the effective learning rate, `_t_sne.py:734-737`,`:878`), and
  `n_features_in_` (Probe 2: `5`). `FittedTsne` (`tsne.rs:203`) exposes
  `embedding()` / `kl_divergence()` / `n_iter()` (`tsne.rs:212-229`) but has NO
  `learning_rate_` (ferrolearn's learning rate is the fixed input `200.0`, never an
  effective `'auto'` value) and NO `n_features_in_` accessor.

- REQ-13: **Generic `F` / f32 support (NOT-STARTED; `#1605`).** sklearn's `TSNE`
  accepts `dtype=[np.float32, np.float64]` (`_t_sne.py:888`,`:892`). ferrolearn's
  `Tsne` is NOT generic — it is f64-only: the struct has no `<F>` parameter
  (`tsne.rs:59`), `Fit` is implemented only for `Array2<f64>` (`tsne.rs:538`), and
  `FittedTsne.embedding_` is `Array2<f64>` (`tsne.rs:205`). This violates the
  CLAUDE.md generic bound `F: Float + Send + Sync + 'static` (f32 + f64) that the
  sibling decomp estimators (`fast_ica`, `minibatch_nmf`) satisfy.

- REQ-14: **PyO3 binding (NOT-STARTED; `#1606`).** sklearn exposes `TSNE` via
  `import sklearn.manifold`. ferrolearn has NO PyO3 binding for `Tsne` — a
  `grep -rn "TSNE\|Tsne" ferrolearn-python/src/` is empty; the only non-test
  consumer of `Tsne`/`FittedTsne` is the crate re-export (`lib.rs:102`). The
  CPython surface (a `_RsTSNE` class with a ctor + `fit_transform` exposing
  `embedding_`/`kl_divergence_`/`n_iter_`) is absent.

- REQ-15: **ferray substrate (NOT-STARTED; `#1607`).** `tsne.rs` computes on
  `ndarray::Array2` (`tsne.rs:45`) and uses `rand` + `rand_distr::Normal` +
  `rand_xoshiro::Xoshiro256PlusPlus` (`tsne.rs:46-48`,`:602-603`) for the embedding
  init, not `ferray-core` arrays / `ferray::random` (R-SUBSTRATE-1/2).

## Acceptance criteria

All expected values from the live sklearn 1.5.2 oracle (run from `/tmp`), never
literal-copied from ferrolearn (R-CHAR-3).

- AC-1 (REQ-1, SHIPPED scoped): `Tsne::new().with_perplexity(5.0).with_n_iter(50)
  .with_random_state(42).fit(&X).unwrap().embedding().dim()` is
  `(n_samples, n_components)`; `with_n_components(3)` gives `ncols == 3`;
  `with_theta(0.0)` runs the exact gradient; two seeded fits are identical to
  `1e-10`; `n_iter()` records the configured iteration count. Pinned by
  `test_tsne_basic_shape` `(20,2)`, `test_tsne_3d_embedding`, `test_tsne_exact_mode`,
  `test_tsne_reproducibility`, `test_tsne_n_iter_recorded`. (Structural shape /
  determinism / the run only — NOT exact embedding values, REQ-5.)

- AC-2 (REQ-2, SHIPPED scoped): the per-point perplexity binary search
  (`compute_pij_row`) + symmetrisation (`compute_joint_probabilities`) produce a
  valid symmetric joint-probability structure (positive, diagonal zero) that drives
  the attractive forces. Probe 3 confirms sklearn's `_joint_probabilities` is a
  symmetric joint summing to 1 with all entries `> 0`. (Structural valid-probability
  property only — exact element-wise `P` parity is NOT observable through the public
  API, `P` is internal to `fit`; FLAG for the critic.)

- AC-3 (REQ-3, SHIPPED scoped): `fit` returns `Err` for `n_components=0`,
  `n_samples < 2`, `perplexity <= 0`, `perplexity >= n_samples`,
  `learning_rate <= 0`, and `theta < 0`. Pinned by
  `test_tsne_invalid_n_components_zero`, `test_tsne_insufficient_samples`,
  `test_tsne_invalid_perplexity_zero`, `test_tsne_perplexity_too_large`,
  `test_tsne_invalid_learning_rate`, `test_tsne_invalid_theta`. FLAG: sklearn raises
  `InvalidParameterError`/`ValueError` (not `FerroError`), and enforces
  `early_exaggeration ∈ [1,inf)`, `max_iter ∈ [250,inf)` (Probe 4),
  `angle ∈ [0,1]`, `n_components < 4` for `barnes_hut` (Probe 4) — guards ferrolearn
  lacks.

- AC-4 (REQ-4, SHIPPED scoped — the meaningful check): on well-separated clusters
  the embedding separates them — `test_tsne_separates_clusters` asserts 3-NN
  label-recovery accuracy `> 0.8`; `kl_divergence() >= 0` and is finite;
  `n_iter()` is recorded. Probe 1 confirms sklearn's embedding separates two
  clusters (inter-centroid `≫` intra spread). Pinned by
  `test_tsne_separates_clusters`, `test_tsne_kl_divergence_non_negative`,
  `test_tsne_n_iter_recorded`. (Structural separation / KL finiteness only — NOT
  exact embedding values nor the exact `kl_divergence_` value, REQ-5.)

- AC-5 (REQ-5, NOT-STARTED, CARVE-OUT): `TSNE(random_state=0, init='random')
  .fit_transform(X)` (Probe 1) is NOT reproduced element-wise by ferrolearn (PCA
  default vs random init + numpy RNG vs Xoshiro + non-convex local minima +
  `learning_rate='auto'` vs fixed 200 + dense-tree vs sparse Barnes-Hut). No failing
  test asserts this (R-DEFER-3); the observable check is cluster separation (AC-4).

- AC-6 (REQ-6..15, DIVERGES): `TSNE()` defaults `n_components=2, perplexity=30.0,
  early_exaggeration=12.0, learning_rate='auto', max_iter=None(→1000),
  n_iter_without_progress=300, min_grad_norm=1e-7, metric='euclidean',
  metric_params=None, init='pca', method='barnes_hut', angle=0.5, n_iter=deprecated`
  (Probe 2, `_t_sne.py:825-860`); sklearn computes `learning_rate_='auto'` (Probe 2:
  `50.0`), exposes `learning_rate_`/`n_features_in_`, early-stops on
  `n_iter_without_progress`/`min_grad_norm`, supports `metric`/`metric_params`,
  `method='exact'`, the `n_iter`→`max_iter` rename, f32, and is exposed via PyO3.
  ferrolearn fixes `learning_rate=200.0`, always-random init, no `metric`/`method`/
  `max_iter`/early-stop params, no `learning_rate_`/`n_features_in_`, f64-only, and
  no PyO3 binding.

- AC-7 (REQ-13/14/15): `Tsne` is f64-only (`Fit<Array2<f64>, ()>` `tsne.rs:538`,
  no `<F>`); `import ferrolearn` exposes NO `_RsTSNE`
  (`grep -rn "TSNE\|Tsne" ferrolearn-python/src/` is empty); the only non-test
  consumer is the crate re-export (`lib.rs:102`). The module imports `ndarray`
  (`tsne.rs:45`) + `rand`/`rand_distr`/`rand_xoshiro` (`:46-48`), not ferray.

`## REQ status`

## REQ status

Binary (R-DEFER-2). `Tsne` / `FittedTsne` are existing pub APIs; the non-test
consumer is the crate re-export (`lib.rs:102`, boundary public API, grandfathered
S5/R-DEFER-1) — there is NO PyO3 binding (REQ-14 NOT-STARTED) and NO `Transform`
impl (matches sklearn — t-SNE has no out-of-sample projection). Cites use symbol
anchors (ferrolearn) / `file:line` (sklearn 1.5.2). Live oracle = installed
sklearn 1.5.2, run from `/tmp`.
**EXACT `embedding_` VALUE PARITY DIVERGES (R-HONEST-3, REQ-5 NOT-STARTED,
CARVE-OUT `#1597`):** ferrolearn's always-random `Normal(0, 1e-4)`
`Xoshiro256PlusPlus` init (`tsne.rs:601-608`) + non-convex local minima + fixed
`learning_rate=200.0` (`:87`) + dense-`P` custom k-d-tree Barnes-Hut (`BHTree`
`:242`) ≠ sklearn's deterministic PCA init (`_t_sne.py:1019-1030`) + numpy
`RandomState` + `learning_rate='auto'` (`:876-879`) + sparse-`P` Cython
`_barnes_hut_tsne.gradient` (`:284`); the objective is explicitly non-convex
(`_t_sne.py:570-571`).
**DETERMINISTIC-P PARITY CANDIDATE (FLAG for the critic):** sklearn
`_joint_probabilities` (`_t_sne.py:41-71`) is DETERMINISTIC given `X` (Probe 3:
symmetric joint sums to 1, all `> 0`, reproducible) and SHOULD match ferrolearn's
`compute_joint_probabilities` (`tsne.rs:467`) — BUT ferrolearn does NOT expose `P`
(it is a private `fn` internal to `fit`, no accessor), so a direct `P`-parity test
is NOT possible through the public API; only the STRUCTURAL valid-probability
properties (REQ-2) are observable. NOTE the `/2n` (`tsne.rs:480-483`) vs `/sum(P)`
(`_t_sne.py:69-70`) normalisation difference as a `P` divergence candidate.
**MEANINGFUL STRUCTURAL CHECK (FLAG, like fast_ica's source-recovery):**
cluster SEPARATION / neighbour preservation is the real "did t-SNE work" signal —
`test_tsne_separates_clusters` pins 3-NN accuracy `> 0.8` (REQ-4); Probe 1 shows
inter-centroid `≫` intra spread.
The least-confident SHIPPED claim is REQ-2 — the affinities' STRUCTURAL
valid-probability property is exercised only end-to-end (via cluster separation /
shape), NOT by a direct `P`-property test, and exact `P` parity is unobservable
through the public API (flagged for the critic). #1596 is this doc's crosslink
tracking issue. Count: **4 SHIPPED (REQ-1,2,3,4) / 11 NOT-STARTED
(REQ-5,6,7,8,9,10,11,12,13,14,15)**.

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (structural: embedding shape / early-exag+momentum+BH/exact GD runs / determinism) | SHIPPED | `fn fit` (`Fit<Array2<f64>, ()>` for `Tsne`, impl at `tsne.rs:538`) inits a `(n, dim)` embedding from a seeded `Xoshiro256PlusPlus` + `Normal(0, 1e-4)` (`tsne.rs:601-608`), runs GD to `n_iter` (`:618`): early-exaggeration for `min(250, n_iter)` iters (`early_exag_end` `:614`, `exaggeration` `:620-624`, = `_EXPLORATION_MAX_ITER=250` `_t_sne.py:820` + `P*=early_exaggeration` `:1095`), momentum 0.5/0.8 (`:619`, = `:1080`,`:1110`), adaptive gains inc `+0.2`/dec `*0.8` floor `0.01` (`:643-647`, = `_t_sne.py:407-409`), velocity `v=momentum*v−lr*gain*g` (`:648`, = `:411`), per-axis centring (`:653-659`); gradient via `Tsne::bh_gradient` (`:674`, = `_kl_divergence_bh` `_t_sne.py:208`) when `use_bh = theta>0 && dim<=3` (`:616`) else `exact_gradient` (`:759`, = `_kl_divergence` `:131`). **Scope: STRUCTURAL, NOT exact embedding values (REQ-5).** Non-test consumer: re-export `lib.rs:102`. Verification: `cargo test -p ferrolearn-decomp --lib tsne` → `test_tsne_basic_shape` `(20,2)`, `test_tsne_3d_embedding`, `test_tsne_exact_mode`, `test_tsne_reproducibility` (`1e-10`), `test_tsne_n_iter_recorded` PASS. |
| REQ-2 (structural: perplexity binary-search affinities → valid symmetric joint P) | SHIPPED | `compute_pij_row` (`tsne.rs:410`) binary-searches `beta=1/(2σ²)` per point to match entropy `ln(perplexity)` to `1e-5` (100 iters, `:413`,`:449-460`, = `_binary_search_perplexity` `_t_sne.py:65`); `compute_joint_probabilities` (`tsne.rs:467`) symmetrises `P_ij=(P_i\|j+P_j\|i)/(2n)` (`:479-483`, doc `:16`), clamps `≥1e-12` (`:485`, = `MACHINE_EPSILON` clamp `_t_sne.py:70`), zeroes diagonal (`:489`). Probe 3: sklearn `_joint_probabilities` symmetric joint sums to 1, all `>0`, deterministic. **Scope: STRUCTURAL valid-probability property — exact element-wise `P` parity NOT observable (P internal to `fit`, no accessor; FLAG).** Exercised end-to-end via the cluster-separation + shape tests (affinities = attractive-force input). Non-test consumer: re-export `lib.rs:102`. **FLAG (critic):** `/2n` (`tsne.rs:480-483`) vs sklearn `/sum(P)` (`_t_sne.py:69-70`) normalisation. |
| REQ-3 (error / parameter contracts, scoped) | SHIPPED | `fn fit` (`tsne.rs:538`) returns `Err(InvalidParameter{name:"n_components"})` for `==0` (`:554-559`), `Err(InsufficientSamples{required:2})` for `n_samples<2` (`:560-566`), `Err(InvalidParameter{name:"perplexity"})` for `<=0` (`:567-572`) and `>=n_samples` (`:573-581`, = `_check_params_vs_input` `_t_sne.py:862-864`), `Err(InvalidParameter{name:"learning_rate"})` for `<=0` (`:582-587`), `Err(InvalidParameter{name:"theta"})` for `<0` (`:588-593`). Non-test consumer: re-export `lib.rs:102`. Verification: `cargo test -p ferrolearn-decomp --lib tsne` (`test_tsne_invalid_n_components_zero`, `_insufficient_samples`, `_invalid_perplexity_zero`, `_perplexity_too_large`, `_invalid_learning_rate`, `_invalid_theta`) PASS. **FLAG (candidate DIVs):** sklearn raises `InvalidParameterError`/`ValueError` (`_t_sne.py:791-817`), not `FerroError`; enforces `early_exaggeration∈[1,inf)` (`:794`), `max_iter∈[250,inf)` (`:799`, Probe 4), `angle∈[0,1]` (`:811`), `n_components<4` for barnes_hut (`:917-922`, Probe 4) — ferrolearn lacks these. |
| REQ-4 (structural: cluster SEPARATION + finite kl_divergence_ ≥ 0 + n_iter_ — the "did t-SNE work" check) | SHIPPED | The fit separates well-separated input clusters: `test_tsne_separates_clusters` (`make_blobs` `tsne.rs:815`, 3 blobs) asserts 3-NN label-recovery accuracy `>0.8` (`:899-905`) — the neighbour-preservation correctness signal (analogous to fast_ica source-recovery). `compute_kl_divergence` (`tsne.rs:496`) = `Σ p_ij·ln(p_ij/q_ij)` over Student-t `Q=1/(1+\|\|y_i−y_j\|\|²)`/`Z` (`:508`,`:524-527`, = `2·P·log(P/Q)` `_t_sne.py:184-191`), stored `kl_divergence_` (`:662-666`), `n_iter_` (`:667`). Probe 1: sklearn embedding separates clusters (inter-centroid `≫` intra). **Scope: STRUCTURAL separation / KL finiteness, NOT exact embedding values nor exact `kl_divergence_` (REQ-5).** Non-test consumer: re-export `lib.rs:102`. Verification: `cargo test -p ferrolearn-decomp --lib tsne` → `test_tsne_separates_clusters` (k-NN `>0.8`), `test_tsne_kl_divergence_non_negative`, `test_tsne_n_iter_recorded` PASS. |
| REQ-5 (EXACT `embedding_` / `kl_divergence_` value parity) | NOT-STARTED | open prereq blocker **#1597** (CARVE-OUT, R-DEFER-3). Objective is NON-CONVEX (`_t_sne.py:570-571`). FOUR factors: (a) init — sklearn DETERMINISTIC PCA `std(PC1)=1e-4` (`:1019-1030`) vs ferrolearn always-random `Normal(0,1e-4)` `Xoshiro256PlusPlus` (`tsne.rs:601-608`); (b) non-convex local minima; (c) `learning_rate` — fixed `200.0` (`tsne.rs:87`) vs `'auto'=max(N/early_exag/4,50)` (`_t_sne.py:876-879`, Probe 2 `50.0`); (d) Barnes-Hut — dense-`P` custom k-d tree (`BHTree` `tsne.rs:242`) vs sparse-`P` Cython `_barnes_hut_tsne.gradient` (`_t_sne.py:284`). Embedding identifiable only up to rotation/reflection/translation + local-optimum. No failing test (same class as fast_ica/lda/minibatch_nmf carve-outs); observable check = cluster separation (REQ-4). |
| REQ-6 (`init='pca'` default + `init='random'` `1e-4*standard_normal` + ndarray init) | NOT-STARTED | open prereq blocker **#1598**. sklearn `init='pca'` DEFAULT (`_t_sne.py:837`, randomized-PCA rescaled `std(PC1)=1e-4` `:1019-1030`), also `init='random'` (`1e-4*random_state.standard_normal` `:1031-1036`) or an `ndarray` (`:1017-1018`); constraint `StrOptions({"pca","random"})\|np.ndarray` (`:804-807`). ferrolearn `Tsne` (`tsne.rs:59`) has NO `init` field — ALWAYS random Gaussian `Normal(0,1e-4)` `Xoshiro256PlusPlus` (`tsne.rs:601-608`), neither PCA default, numpy-`standard_normal`, nor user init. |
| REQ-7 (`learning_rate='auto'` default) | NOT-STARTED | open prereq blocker **#1599**. sklearn `learning_rate='auto'` DEFAULT (`_t_sne.py:831`) → `learning_rate_=max(N/early_exaggeration/4, 50)` (`:876-879`, Probe 2 `50.0`), numeric `(0,inf)` alt (`:795-798`). ferrolearn `Tsne.learning_rate` (`tsne.rs:64`) is FIXED `200.0` (`new()` `:87`) — no `'auto'`, no `learning_rate_` attr (REQ-12). |
| REQ-8 (`n_iter_without_progress` / `min_grad_norm` early stopping) | NOT-STARTED | open prereq blocker **#1600**. sklearn `_gradient_descent` (`_t_sne.py:304-447`) early-stops on `n_iter_without_progress` (default 300, `:431`; exploration `_EXPLORATION_MAX_ITER=250` `:1078`) OR `grad_norm<=min_grad_norm` (default `1e-7`, `:439`), checked every `_N_ITER_CHECK=50` (`:399`,`:823`). ferrolearn GD loop (`tsne.rs:618-660`) runs a FIXED `n_iter` with NO progress/grad-norm stop, NO `n_iter_without_progress`/`min_grad_norm` fields — `n_iter_` always = configured `n_iter` (`:667`). |
| REQ-9 (`metric` precomputed/non-euclidean + `metric_params`) | NOT-STARTED | open prereq blocker **#1601**. sklearn `metric='euclidean'` DEFAULT (`_t_sne.py:835`) accepts any pairwise metric, callable, or `'precomputed'` (square distance matrix, `:894-900`,`:931-932`) + `metric_params` (`:836`,`:945-947`). ferrolearn `compute_joint_probabilities` (`tsne.rs:467`) ALWAYS uses squared-Euclidean `pairwise_sq_distances` (`tsne.rs:389`) — no `metric`/`metric_params`, no precomputed path, no non-euclidean. |
| REQ-10 (`method='exact'` vs barnes_hut-only + exact-`P`/`sum(P)` normalisation) | NOT-STARTED | open prereq blocker **#1602**. sklearn `method='barnes_hut'` DEFAULT (`_t_sne.py:840`) selects sparse-`P` Barnes-Hut (`_joint_probabilities_nn` `:74` + `_kl_divergence_bh` `:208`) or `'exact'` (dense `_joint_probabilities` `:41` + `_kl_divergence` `:131`); constraint `StrOptions({"barnes_hut","exact"})` (`:810`). ferrolearn has NO `method` field — ALWAYS dense `P` (`compute_joint_probabilities` `tsne.rs:467`), selects tree-vs-exact GRADIENT via `use_bh=theta>0 && dim<=3` (`:616`) over the same dense `P` (NOT sklearn's sparse-`P` barnes_hut). Also `/2n` (`tsne.rs:480-483`) vs `/sum(P)` (`_t_sne.py:69-70`) normalisation (FLAG, Probe 3). |
| REQ-11 (`n_iter`→`max_iter` rename + deprecation + `max_iter≥250` + None→1000) | NOT-STARTED | open prereq blocker **#1603**. sklearn renamed `n_iter`→`max_iter` in 1.5 (`_t_sne.py:621-626`,`:825`,`:843`), `max_iter=None`→`1000` (`:832`,`:1170-1171`), enforces `max_iter∈[250,inf)` (`:799`, Probe 4), deprecated `n_iter` `FutureWarning`s + errors if both set (`:1153-1169`). ferrolearn field is `n_iter` (default 1000, `tsne.rs:67`,`:88`, `with_n_iter` `:118`) — no `max_iter` alias, no deprecation, no `≥250` constraint. |
| REQ-12 (fitted attrs `learning_rate_` / `n_features_in_`) | NOT-STARTED | open prereq blocker **#1604**. sklearn exposes `embedding_`, `kl_divergence_`, `n_iter_`, `learning_rate_` (effective LR, `_t_sne.py:734-737`,`:878`), `n_features_in_` (Probe 2 `5`). `FittedTsne` (`tsne.rs:203`) exposes `embedding()`/`kl_divergence()`/`n_iter()` (`:212-229`) — NO `learning_rate_` (ferrolearn LR is the fixed input 200.0), NO `n_features_in_`. |
| REQ-13 (generic `F` / f32 support) | NOT-STARTED | open prereq blocker **#1605**. sklearn `TSNE` accepts `dtype=[np.float32, np.float64]` (`_t_sne.py:888`,`:892`). ferrolearn `Tsne` is f64-only — no `<F>` (`tsne.rs:59`), `Fit` only for `Array2<f64>` (`:538`), `FittedTsne.embedding_` is `Array2<f64>` (`:205`); violates the CLAUDE.md `F: Float + Send + Sync + 'static` bound the sibling decomp estimators satisfy. |
| REQ-14 (PyO3 binding) | NOT-STARTED | open prereq blocker **#1606**. sklearn exposes `TSNE` via `import sklearn.manifold`. ferrolearn has NO PyO3 binding — `grep -rn "TSNE\|Tsne" ferrolearn-python/src/` is empty; the only non-test consumer of `Tsne`/`FittedTsne` is the crate re-export (`lib.rs:102`). No `_RsTSNE` class. |
| REQ-15 (ferray substrate) | NOT-STARTED | open prereq blocker **#1607**. `tsne.rs` computes on `ndarray::Array2` (`tsne.rs:45`) and uses `rand`/`rand_distr::Normal`/`rand_xoshiro::Xoshiro256PlusPlus` (`:46-48`,`:602-603`) for the embedding init, not `ferray-core` arrays / `ferray::random` (R-SUBSTRATE-1/2). |

## Architecture

`tsne.rs` follows the unfitted/fitted split (CLAUDE.md naming): `Tsne {
n_components (2), perplexity (30.0), learning_rate (200.0 FIXED), n_iter (1000),
early_exaggeration (12.0), theta (0.5), random_state }` (struct at `tsne.rs:59`;
`new()` at line 83, builders `with_n_components`/`with_perplexity`/
`with_learning_rate`/`with_n_iter`/`with_early_exaggeration`/`with_theta`/
`with_random_state` at lines 96-142, accessors at lines 144-184) →
`Fit<Array2<f64>, ()>` → `FittedTsne { embedding_ (n_samples, n_components),
kl_divergence_, n_iter_ }` (struct at `tsne.rs:203`, accessors `embedding()`/
`kl_divergence()`/`n_iter()` at lines 212-229). Unlike the sibling decomp
estimators, `Tsne` is NOT generic — it is f64-only (`Fit<Array2<f64>, ()>`
`tsne.rs:538`, REQ-13). `fit` returns `Result<FittedTsne, FerroError>` (R-CODE-2).
`Default` is `new()` (`tsne.rs:187-191`). Matching sklearn, there is NO `Transform`
impl — t-SNE has no out-of-sample projection (module doc `tsne.rs:27-28`).

**Fit path (`fn fit`, impl at `tsne.rs:538`) — REQ-1/2/3/4/5.** Validates
`n_components != 0`, `n_samples >= 2`, `perplexity > 0`, `perplexity < n_samples`,
`learning_rate > 0`, `theta >= 0` (`tsne.rs:554-593`) — REQ-3. (1) **Affinities**
(`compute_joint_probabilities` `tsne.rs:467`): squared-Euclidean pairwise distances
(`pairwise_sq_distances` `tsne.rs:389`), per-point Gaussian-kernel bandwidth tuned
by binary search to the target perplexity (`compute_pij_row` `tsne.rs:410`, =
`_binary_search_perplexity` `_t_sne.py:65`), then symmetrised `P_ij =
(P_i|j + P_j|i)/(2n)` and clamped `≥ 1e-12` (`tsne.rs:479-489`) — REQ-2. This is
sklearn's `_joint_probabilities` (`_t_sne.py:41-71`) but with a `/2n` rather than
`/sum(P)` normalisation and DENSE rather than the barnes_hut sparse-`P`
(`_joint_probabilities_nn` `:74`) — REQ-10. (2) **Init** (`tsne.rs:601-608`): a
random Gaussian `Normal(0, 1e-4)` from a seeded `Xoshiro256PlusPlus` (seed
`random_state.unwrap_or(0)`) — NOT sklearn's `init='pca'` default (`_t_sne.py:1019-1030`,
REQ-6) and NOT numpy `standard_normal` (REQ-5). (3) **Gradient descent**
(`tsne.rs:610-660`): `n_iter` iterations (FIXED, no early stop, REQ-8) with
early-exaggeration over the first `min(250, n_iter)` (`tsne.rs:614`,`:620-624`),
momentum 0.5→0.8 (`:619`), adaptive gains (`:643-647`), velocity + position update
(`:648-649`), and per-axis centring (`:653-659`). The gradient is `Tsne::bh_gradient`
(`tsne.rs:674`, Barnes-Hut tree `BHTree` `:242`, attractive over non-zero `P` +
repulsive `compute_repulsive` `:345`) when `use_bh = theta > 0 && dim <= 3`
(`:616`), else `exact_gradient` (`:759`). (4) Stores `embedding_ = y`,
`kl_divergence_ = compute_kl_divergence(&p, &y)` (`tsne.rs:496`,`:662`), and
`n_iter_ = self.n_iter` (`:667`). **This DIFFERS from sklearn's embedding values
(REQ-5):** PCA-vs-random init, numpy-vs-Xoshiro RNG, non-convex local minima,
`'auto'`-vs-200 learning rate, and dense-tree-vs-sparse Barnes-Hut produce a
different (but structurally cluster-separating) embedding.

**Barnes-Hut tree (`BHTree`, `tsne.rs:242`).** A generic k-d tree storing the
centre of mass + count per node, recursively subdividing into `2^dim` children
(`subdivide` `tsne.rs:324`); `compute_repulsive` (`tsne.rs:345`) accumulates the
repulsive force and the `Z` normaliser using the angular criterion
`width/dist < theta` (`tsne.rs:365`) — the O(n log n) analog of sklearn's Cython
`_barnes_hut_tsne.gradient` (`_t_sne.py:284`), but driven by the dense `P` rather
than the k-NN-sparse `P`.

**sklearn (target contract).** `class TSNE` (`_t_sne.py:563`) takes
`__init__(n_components=2, *, perplexity=30.0, early_exaggeration=12.0,
learning_rate="auto", max_iter=None, n_iter_without_progress=300,
min_grad_norm=1e-7, metric="euclidean", metric_params=None, init="pca", verbose=0,
random_state=None, method="barnes_hut", angle=0.5, n_jobs=None, n_iter="deprecated")`
(`:825-860`). `_fit` (`:866-1051`) computes `learning_rate_` (`:876-882`), the
joint probabilities (dense `_joint_probabilities` `:41` for `exact`, sparse
`_joint_probabilities_nn` `:74` for `barnes_hut`), the init (PCA `:1019-1030` /
random `:1031-1036` / ndarray), then `_tsne` (`:1053-1126`) runs `_gradient_descent`
(`:304-447`) in two phases (early-exaggeration `_EXPLORATION_MAX_ITER=250`
momentum 0.5; then momentum 0.8) storing `n_iter_` (`:1115`) and `kl_divergence_`
(`:1124`). `fit_transform` (`:1132-1178`) resolves `max_iter` (None→1000
`:1170-1171`), checks `perplexity < n_samples` (`:862-864`), sets `embedding_`.
Fitted attrs: `embedding_`, `kl_divergence_`, `n_iter_`, `learning_rate_`,
`n_features_in_`. NO `transform`/`inverse_transform`.

**The remaining gap.** ferrolearn ships the STRUCTURAL embedding shape +
early-exaggeration/momentum/Barnes-Hut-or-exact GD run + determinism (REQ-1), the
perplexity binary-search valid-affinity structure (REQ-2), the scoped error &
parameter contracts (REQ-3), and the meaningful cluster-separation / finite-KL
check (REQ-4). It lacks: exact `embedding_` value parity (REQ-5, CARVE-OUT
`#1597`); `init='pca'`/`'random'`/ndarray (REQ-6, `#1598`);
`learning_rate='auto'` (REQ-7, `#1599`); `n_iter_without_progress`/`min_grad_norm`
early stopping (REQ-8, `#1600`); `metric`/`metric_params` (REQ-9, `#1601`);
`method='exact'` vs barnes_hut + the `sum(P)` normalisation (REQ-10, `#1602`); the
`n_iter`→`max_iter` rename + deprecation + `≥250` constraint (REQ-11, `#1603`); the
`learning_rate_`/`n_features_in_` attrs (REQ-12, `#1604`); generic `F`/f32
(REQ-13, `#1605`); the PyO3 binding (REQ-14, `#1606`); and the ferray substrate
(REQ-15, `#1607`). This is a **structure-SHIPPED-algorithm/API-NOT-STARTED** unit
(4 SHIPPED / 11 NOT-STARTED).

## Verification

Library crate (green at baseline `2d8b9c24`):
```bash
cargo test -p ferrolearn-decomp --lib tsne                  # in-module #[test]s
cargo test -p ferrolearn-decomp tsne                        # + doctest
cargo clippy -p ferrolearn-decomp --all-targets -- -D warnings
cargo fmt --all --check
```
The in-tree `#[test]`s pin REQ-1/2/3/4 (STRUCTURAL): `test_tsne_basic_shape`
`(20,2)`, `test_tsne_3d_embedding` `ncols==3`, `test_tsne_exact_mode` (`theta=0`),
`test_tsne_reproducibility` (two seeded fits identical to `1e-10`),
`test_tsne_n_iter_recorded`, `test_tsne_getters`, `test_tsne_default` (REQ-1);
the affinity structure is exercised end-to-end (REQ-2);
`test_tsne_invalid_n_components_zero`, `test_tsne_insufficient_samples`,
`test_tsne_invalid_perplexity_zero`, `test_tsne_perplexity_too_large`,
`test_tsne_invalid_learning_rate`, `test_tsne_invalid_theta` (REQ-3);
`test_tsne_separates_clusters` (3-NN accuracy `> 0.8` — the meaningful "did t-SNE
work" check), `test_tsne_kl_divergence_non_negative` (REQ-4); plus the module
doctest (`tsne.rs:32-41`). There is NO `tests/divergence_tsne.rs` yet. REQ-5
(embedding value parity) is a CARVE-OUT (R-DEFER-3, no failing test). **FLAG for
the critic:** (i) the meaningful structural check is cluster separation /
neighbour preservation (Probe 1, REQ-4) — already pinned by
`test_tsne_separates_clusters` but worth a stronger trustworthiness-style guard;
(ii) the high-dim affinity `P` is DETERMINISTIC given `X` (Probe 3) and a `P`-parity
candidate, but ferrolearn does NOT expose `P` (internal to `fit`, no accessor) so a
direct `P`-parity test is not possible through the public API — only the structural
properties (REQ-2) are observable, plus the `/2n` vs `/sum(P)` normalisation
(REQ-10) divergence candidate.

Live sklearn oracle (installed 1.5.2, run from `/tmp`) — the REQ-5 embedding value
gap (observable only as cluster separation, REQ-4) and the deterministic-`P`
candidate (REQ-2/10):
```bash
# REQ-4 the meaningful structural check (cluster separation) — embedding VALUES NOT pinned:
python3 -c "import numpy as np; from sklearn.manifold import TSNE
c0=np.zeros((8,4)); c1=np.ones((8,4))*10.0
X=np.vstack([c0,c1])+0.01*np.arange(64).reshape(16,4)
emb=TSNE(n_components=2, perplexity=5, random_state=0, init='random', learning_rate='auto').fit_transform(X)
c0m,c1m=emb[:8].mean(0),emb[8:].mean(0)
inter=np.linalg.norm(c0m-c1m); intra=(np.linalg.norm(emb[:8]-c0m,axis=1).mean()+np.linalg.norm(emb[8:]-c1m,axis=1).mean())/2
print(emb.shape, bool(inter>3*intra))"
# -> (16, 2) True   => clusters separate (structural, the real t-SNE check); VALUES NOT reproduced (REQ-5)

# REQ-2/10 deterministic-P candidate (sklearn _joint_probabilities, P sums to 1, all > 0):
python3 -c "import numpy as np; from sklearn.manifold._t_sne import _joint_probabilities
from scipy.spatial.distance import pdist, squareform
X=np.array([[0.,0],[1,0],[0,1],[5,5]]); D=squareform(pdist(X,'sqeuclidean')).astype(np.float32)
P=_joint_probabilities(D,2.0,0); print(round(float(2*P.sum()),6), bool((P>0).all()))"
# -> 1.0 True   => deterministic-given-X (critic parity candidate; ferrolearn does NOT expose P)
```
The cluster-separation oracle uses a KNOWN well-separated two-cluster dataset; the
inter `≫` intra property (and the in-tree 3-NN accuracy `> 0.8`) is the observable
t-SNE-correctness check (R-CHAR-3), NOT copied from ferrolearn.

ferrolearn-python (REQ-14, ABSENT at baseline): there is NO `_RsTSNE` binding —
`grep -rn "TSNE\|Tsne" ferrolearn-python/src/` is empty. The only non-test consumer
of `Tsne`/`FittedTsne` is the crate re-export (`lib.rs:102`). There is also NO
`Transform` impl (matches sklearn — t-SNE has no out-of-sample projection,
`tsne.rs:27-28`).

## Blockers

(#1596 is this doc's crosslink tracking issue. The blockers below are the open work
items the dispatcher files / numbers; none are filed by this doc — markdown only.)

- **#1597** — REQ-5 (CARVE-OUT): reach EXACT `embedding_` value parity by matching
  sklearn's deterministic PCA init (`_t_sne.py:1019-1030`), the numpy `RandomState`
  for `init='random'` (`:1031-1036`), `learning_rate='auto'` (`:876-879`), the
  sparse-`P` Cython Barnes-Hut (`_barnes_hut_tsne.gradient` `:284`), AND a fixed
  rotation/reflection/translation alignment of the non-convex optimum
  (`:570-571`); inherently RNG/init/local-optimum-bound (no failing test, R-DEFER-3
  — the observable check is cluster separation, REQ-4).
- **#1598** — REQ-6: add an `init` field — `'pca'` DEFAULT (randomized PCA rescaled
  `std(PC1)=1e-4`, `_t_sne.py:1019-1030`), `'random'` (`1e-4*standard_normal`
  `:1031-1036`), and a user `ndarray` (`:1017-1018`).
- **#1599** — REQ-7: add `learning_rate='auto'` (default) computing
  `learning_rate_ = max(N/early_exaggeration/4, 50)` (`_t_sne.py:876-879`).
- **#1600** — REQ-8: add `n_iter_without_progress` (300) + `min_grad_norm` (1e-7)
  fields and the `_gradient_descent` early-stop (`_t_sne.py:431`,`:439`, checked
  every `_N_ITER_CHECK=50`), so `n_iter_` records the stopping iteration.
- **#1601** — REQ-9: add `metric` (any pairwise metric / callable / `'precomputed'`,
  `_t_sne.py:894-900`) + `metric_params` (`:945-947`) to the affinity computation.
- **#1602** — REQ-10: add a `method` field (`'barnes_hut'` sparse-`P`
  `_joint_probabilities_nn` `_t_sne.py:74` + `_kl_divergence_bh` `:208`; `'exact'`
  dense `_joint_probabilities` `:41` + `_kl_divergence` `:131`) and normalise `P` by
  `sum(P)` (`:69-70`) rather than `2n` (`tsne.rs:480-483`).
- **#1603** — REQ-11: rename `n_iter`→`max_iter` (default None→1000,
  `_t_sne.py:1170-1171`), enforce `max_iter ∈ [250, inf)` (`:799`), and add the
  deprecated `n_iter` alias + `FutureWarning` (`:1153-1169`).
- **#1604** — REQ-12: expose `learning_rate_` (the effective LR, `_t_sne.py:878`)
  and `n_features_in_` accessors on `FittedTsne`.
- **#1605** — REQ-13: make `Tsne` generic over `F: Float + Send + Sync + 'static`
  (f32 + f64, `_t_sne.py:888`) per CLAUDE.md, like the sibling decomp estimators.
- **#1606** — REQ-14: add a `_RsTSNE` PyO3 binding in `ferrolearn-python` (ctor +
  `fit_transform` exposing `embedding_`/`kl_divergence_`/`n_iter_`), registered in
  `lib.rs`, as the CPython consumer.
- **#1607** — REQ-15: migrate `tsne.rs` off `ndarray` + `rand`/`rand_distr`/
  `rand_xoshiro` to `ferray-core` arrays / `ferray::random` (R-SUBSTRATE).
