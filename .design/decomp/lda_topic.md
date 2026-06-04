# LatentDirichletAllocation (sklearn.decomposition.LatentDirichletAllocation — topic model)

<!--
tier: 3-component
status: shipped-partial
baseline-commit: 15506b3e
upstream: scikit-learn 1.5.2
upstream-paths:
  - sklearn/decomposition/_lda.py  # class LatentDirichletAllocation(ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator) (:163). TOPIC-MODEL LDA (online variational Bayes, Hoffman et al. 2010), NOT Linear Discriminant Analysis. ctor (:362-397): n_components=10, *, doc_topic_prior=None, topic_word_prior=None, learning_method="batch", learning_decay=0.7, learning_offset=10.0, max_iter=10, batch_size=128, evaluate_every=-1, total_samples=1e6, perp_tol=1e-1, mean_change_tol=1e-3, max_doc_update_iter=100, n_jobs=None, verbose=0, random_state=None. _init_latent_vars (:399-426): self.n_iter_=0 (:404); doc_topic_prior_ = doc_topic_prior or 1/n_components (:406-409); topic_word_prior_ = topic_word_prior or 1/n_components (:411-414); init_gamma=100.0, init_var=1/100 (:416-417); self.components_ = random_state_.gamma(100., 0.01, (n_components, n_features)) (:419-421) — Gamma RNG init; self.exp_dirichlet_component_ = exp(_dirichlet_expectation_2d(components_)) (:424-426). _update_doc_distribution (:88-160): per-doc gamma init random_state.gamma(100., 0.01, (n_samples, n_topics)) when random_init (:96-99) else np.ones (:100-101); exp_doc_topic = exp(_dirichlet_expectation_2d(...)) (:104); inner loop (:139-151): norm_phi = exp_doc_topic_d @ exp_topic_word_d + eps (:144), doc_topic_d = exp_doc_topic_d * ((cnts/norm_phi) @ exp_topic_word_d.T) (:146), dirichlet_expectation_1d adds doc_topic_prior in-place (:148), break on mean_change(last_d, doc_topic_d) < mean_change_tol (:150, default 1e-3). _em_step (:494-546): E-step suff-stats (:521-524); M-step batch components_ = topic_word_prior_ + suff_stats (:528) OR online EWA components_ = (1-rho)*components_ + rho*(topic_word_prior_ + doc_ratio*suff_stats) with rho = (learning_offset + n_batch_iter_)^(-learning_decay) (:532-539); updates exp_dirichlet_component_ (:542-544). fit (:625-705): _init_latent_vars (:655); for i in range(max_iter) (:660): online gen_batches(n_samples, batch_size) (:662) or batch (:671); if evaluate_every>0 and (i+1)%evaluate_every==0: bound=_perplexity_precomp_distr(...); if last_bound and |last_bound-bound|<perp_tol: break (:676-691); self.n_iter_ += 1 (:695). DEFAULT evaluate_every=-1 => NO break => n_iter_ == max_iter. final self.bound_ = _perplexity_precomp_distr(...) (:701-703). transform (:724-746): _unnormalized_transform = _e_step (:720/:744), then doc_topic_distr /= doc_topic_distr.sum(axis=1)[:,newaxis] (:745) — each row sums to 1. perplexity (:896), score (:827), _approx_bound (:748), _perplexity_precomp_distr (:790) — log-likelihood bound methods. partial_fit (:560). _more_tags preserves_dtype [np.float64, np.float32] (:548-552).
ferrolearn-module: ferrolearn-decomp/src/lda_topic.rs
parity-ops: LatentDirichletAllocation
crosslink-issue: 1540
-->

## Summary

`ferrolearn-decomp/src/lda_topic.rs` mirrors scikit-learn's `LatentDirichletAllocation`
(`sklearn/decomposition/_lda.py`, `class LatentDirichletAllocation(...)` `:163`) — the
**TOPIC-MODEL** LDA (online variational Bayes, Hoffman et al. 2010), **NOT** Linear
Discriminant Analysis (which lives in `ferrolearn-linear`). It learns a topic-word
variational parameter matrix `components_` (`lambda`, `n_topics × n_words`) and, on
`transform`, the per-document topic proportions (`gamma`, normalised to sum 1) over a
document-term matrix. The exposed surface is the unfitted `LatentDirichletAllocation {
n_components, max_iter (10), learning_method: LdaLearningMethod{Batch|Online},
learning_offset (10.0), learning_decay (0.7), doc_topic_prior (None → 1/K),
topic_word_prior (None → 1/K), max_doc_update_iter (100), random_state }`
(`lda_topic.rs`, struct at line 68; builders `with_max_iter`/`with_learning_method`/
`with_learning_offset`/`with_learning_decay`/`with_doc_topic_prior`/
`with_topic_word_prior`/`with_random_state`/`with_max_doc_update_iter`, accessors) and
the fitted `FittedLatentDirichletAllocation { components_ (n_topics, n_words), alpha_,
beta_, n_iter_, max_doc_update_iter_ }` (`lda_topic.rs`, struct at line 225, accessors
`components`/`n_iter`/`alpha`/`beta`), re-exported at the crate root (`pub use
lda_topic::{FittedLatentDirichletAllocation, LatentDirichletAllocation,
LdaLearningMethod}`, `lib.rs:91-92`). The path is **f64-ONLY** — `impl Fit<Array2<f64>,
()>` (`lda_topic.rs`, impl at line 364), NOT generic over `<F>` (contrast the sibling
`MiniBatchNMF<F>` / `SparsePCA<F>`). There is NO PyO3 binding (a `grep -rn
LatentDirichlet ferrolearn-python/src/` is empty) and NO
`tests/divergence_lda_topic.rs`.

**EXACT `components_` VALUE PARITY DIVERGES (R-HONEST-3, REQ-4 NOT-STARTED, CARVE-OUT
`#1541`).** ferrolearn's `components_` (`lambda`) VALUES diverge from sklearn through
the variational-init distribution + the RNG + the per-document gamma init: (1)
**lambda init:** ferrolearn seeds `lambda = Uniform(0.5, 1.5) + beta` via
`Xoshiro256PlusPlus::seed_from_u64(random_state.unwrap_or(0))` (`lda_topic.rs`, `fn
fit`, init at lines 414-419), whereas sklearn's `_init_latent_vars` inits `self.components_
= random_state_.gamma(100., 0.01, (n_components, n_features))` — a numpy-`RandomState`
Gamma(100, 0.01) draw (`_lda.py:419-421`). (2) **per-doc gamma init:** ferrolearn's
`e_step_doc` (`lda_topic.rs`, fn at line 303) inits `gamma = alpha + n_words/n_topics`
(uniform, `lda_topic.rs:308`), whereas sklearn's `_update_doc_distribution` inits the
per-document `doc_topic_distr` from `random_state.gamma(100., 0.01, ...)` when
`random_init` (the `_e_step` M-step path, `_lda.py:96-99`). (3) **RNG:** Xoshiro256++ vs
numpy `RandomState` — different streams. The E-step is otherwise the same fixed-point
(`norm_phi`/phi update, mean-change-< 1e-3 stop), but the init distributions and RNG
make the converged `lambda` element-wise distinct. Same class as the `minibatch_nmf` /
`dictionary_learning` / `sparse_pca` RNG carve-outs; no failing test is asserted
(R-DEFER-3). The `transform` doc-topic VALUES are downstream of `components_` + the
E-step gamma init, and the struct fields are PRIVATE (no injectable-`components_` API),
so transform value parity FOLDS INTO the carve-out (REQ-5, `#1542`).

**INVESTIGATE settled (for the critic, R-CHAR-3): the digamma approximation is NOT a
divergence source.** ferrolearn's `digamma` (`lda_topic.rs`, fn at line 277, recurrence
to `x ≥ 6` + asymptotic expansion) matches `scipy.special.psi` to a max absolute error
of `~1.17e-10` over `x ∈ [0.1, 100]` (probe 4). sklearn computes `E[log beta]` as
`exp_dirichlet_component_ = exp(_dirichlet_expectation_2d(components_))` (`_lda.py:424`,
i.e. it carries the EXPONENTIATED expectation), whereas ferrolearn carries the
LOG-space `compute_e_log_beta = digamma(lambda[k,w]) − digamma(row_sum)`
(`lda_topic.rs`, fn at line 589) and exponentiates inside the phi normalisation — an
algebraically equivalent representation of the SAME `E[log beta_{kw}]`, not a
divergence. The remaining REQ-4 gap is purely the init-distribution + RNG.

As of this iteration: the STRUCTURAL `components_` shape `(n_topics, n_words)` and
non-negativity, the `transform` doc-topic shape `(n_docs, n_topics)` with each row
NORMALISED to sum 1, topic SEPARATION on a well-separated two-topic corpus, the
single-topic degenerate (all proportions ≈ 1), the error & parameter contracts
(n_components 0, negative input, empty corpus, zero words, transform col-mismatch,
transform negative), determinism given a seed, and the digamma accuracy (REQ-1,2,3) are
SHIPPED scoped; exact `components_` value parity (REQ-4, CARVE-OUT `#1541`), the
`transform` doc-topic VALUE parity (REQ-5, CARVE-OUT — folds into REQ-4, `#1542`),
Gamma `random_state.gamma(100, 0.01)` init for both `components_` and the per-doc gamma
(REQ-6, `#1543`), the `exp_dirichlet_component_` fitted attr/representation (REQ-7,
`#1544`), `perplexity` / `score` / `_approx_bound` log-likelihood bound (REQ-8,
`#1545`), `evaluate_every` / `perp_tol` perplexity-based early stop (REQ-9, `#1546`),
`batch_size` / `total_samples` online mini-batching (REQ-10, `#1547`), fitted attrs
`n_features_in_` / `bound_` / `doc_topic_prior_` / `topic_word_prior_` (REQ-11,
`#1548`), `n_jobs` / `verbose` (REQ-12, `#1549`), generic `F` f32 + f64 (REQ-13,
`#1550`), the PyO3 binding (REQ-14, `#1551`), and the ferray substrate (REQ-15,
`#1552`) are NOT-STARTED — **3 SHIPPED / 12 NOT-STARTED**.

`LatentDirichletAllocation` / `FittedLatentDirichletAllocation` are existing pub APIs
whose non-test consumer is the crate re-export (`lib.rs:91-92`, boundary public API,
grandfathered S5/R-DEFER-1). There is NO PyO3 binding (REQ-14 NOT-STARTED).

## Probes (live sklearn oracle, 1.5.2, run from /tmp)

```bash
# PROBE 1 (REQ-1/2/3 SHIPPED scoped + REQ-4/7/8/11 NOT-STARTED) — components_ shape
# (n_topics, n_words) + non-negativity; n_iter_ == max_iter under default
# evaluate_every=-1; transform rows sum to 1; the perplexity / fitted attrs sklearn
# exposes. Two-topic corpus matches the in-module two_topic_corpus() fixture. VALUES
# generated by sklearn, never copied from ferrolearn (R-CHAR-3).
python3 -c "
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
dtm=np.array([[5.,5,5,0,0,0],[4,6,3,0,0,0],[5,4,6,0,0,0],[0,0,0,5,5,5],[0,0,0,6,4,3],[0,0,0,4,6,5]])
m=LatentDirichletAllocation(n_components=2, random_state=0).fit(dtm)
print('components_ shape:', m.components_.shape)
print('components_ non-negative:', bool((m.components_>=0).all()))
print('components_ row0:', np.round(m.components_[0],6).tolist())
print('n_iter_:', m.n_iter_)
print('has exp_dirichlet_component_:', hasattr(m,'exp_dirichlet_component_'), m.exp_dirichlet_component_.shape)
print('doc_topic_prior_:', m.doc_topic_prior_, 'topic_word_prior_:', m.topic_word_prior_)
print('n_features_in_:', m.n_features_in_, 'bound_:', round(float(m.bound_),4))
T=m.transform(dtm)
print('transform shape:', T.shape, 'row sums:', np.round(T.sum(axis=1),8).tolist())
print('transform row0:', np.round(T[0],6).tolist())
print('perplexity:', round(float(m.perplexity(dtm)),4), 'score:', round(float(m.score(dtm)),4))"
# -> components_ shape: (2, 6)                                  => structural shape (REQ-1)
# -> components_ non-negative: True                             => non-negativity (REQ-2)
# -> components_ row0: [14.49861, 15.498587, 14.498625, 0.501413, 0.501386, 0.501378]  => VALUES (REQ-4 CARVE-OUT, NOT reproduced)
# -> n_iter_: 10                                                => == max_iter, evaluate_every=-1 no early stop (REQ-9 NOT-STARTED)
# -> has exp_dirichlet_component_: True (2, 6)                  => fitted attr (REQ-7 NOT-STARTED)
# -> doc_topic_prior_: 0.5 topic_word_prior_: 0.5              => fitted prior attrs (REQ-11 NOT-STARTED)
# -> n_features_in_: 6 bound_: 4.1874                          => fitted attrs (REQ-11 NOT-STARTED)
# -> transform shape: (6, 2) row sums: [1.0, ...]              => doc-topic shape + rows sum to 1 (REQ-3 SHIPPED)
# -> transform row0: [0.968663, 0.031337]                      => doc-topic VALUES (REQ-5 CARVE-OUT, NOT reproduced)
# -> perplexity: 4.1874 score: -123.1589                       => log-likelihood bound methods (REQ-8 NOT-STARTED)

# PROBE 2 (REQ-6 NOT-STARTED) — Gamma(100, 0.01) init distribution (mean ~1, var ~0.01)
# from numpy RandomState; ferrolearn uses Uniform(0.5,1.5)+beta via Xoshiro instead.
python3 -c "
import numpy as np
rs=np.random.RandomState(0)
g=rs.gamma(100.0, 0.01, (2,6))
print('gamma(100,0.01) init row0:', np.round(g[0],6).tolist())
print('mean~1.0 var~0.01:', round(float(g.mean()),4), round(float(g.var()),4))"
# -> gamma(100,0.01) init row0: [1.183354, 1.037152, 1.194979, 0.902251, 0.986397, 1.038223]
# -> mean~1.0 var~0.01: 1.0523 0.007   => sklearn lambda + per-doc gamma init (REQ-6 NOT-STARTED)

# PROBE 3 (REQ-9/10/11/12 NOT-STARTED) — ctor defaults.
python3 -c "
from sklearn.decomposition import LatentDirichletAllocation
m=LatentDirichletAllocation()
for p in ['n_components','doc_topic_prior','topic_word_prior','learning_method','learning_decay','learning_offset','max_iter','batch_size','evaluate_every','total_samples','perp_tol','mean_change_tol','max_doc_update_iter','n_jobs','verbose','random_state']:
    print(f'{p} =', getattr(m,p))"
# -> n_components = 10  doc_topic_prior = None  topic_word_prior = None
# -> learning_method = batch  learning_decay = 0.7  learning_offset = 10.0  max_iter = 10
# -> batch_size = 128  evaluate_every = -1  total_samples = 1000000.0  perp_tol = 0.1
# -> mean_change_tol = 0.001  max_doc_update_iter = 100  n_jobs = None  verbose = 0  random_state = None
#    => ferrolearn has n_components/max_iter/learning_method/learning_offset/learning_decay/
#       doc_topic_prior/topic_word_prior/max_doc_update_iter/random_state only; NO batch_size/
#       evaluate_every/total_samples/perp_tol/mean_change_tol(hard-coded 1e-3)/n_jobs/verbose;
#       default n_components is 10 (ferrolearn requires an explicit usize).

# PROBE 4 (REQ-1 SHIPPED: digamma accuracy) — scipy.special.psi cross-check. The
# ferrolearn digamma matches scipy to ~1e-10 (NOT a divergence source).
python3 -c "
from scipy.special import psi
print('psi(1)=', round(float(psi(1)),10), 'psi(10)=', round(float(psi(10)),10), 'psi(0.5)=', round(float(psi(0.5)),10))"
# -> psi(1)= -0.5772156649  psi(10)= 2.2517525891  psi(0.5)= -1.963510026
#    (ferrolearn test_digamma_basic pins digamma(1)≈-0.5772, test_digamma_large pins digamma(10)≈2.2518)
```

## Requirements

- REQ-1: **Structural: `components_` shape `(n_topics, n_words)`, finite/non-negative
  `lambda`, positive `n_iter_`, determinism given seed, and an accurate `digamma`
  (SHIPPED scoped).** `fn fit` (`lda_topic.rs`, impl at line 364) stores `components_ =
  lambda` of shape `(n_topics, n_words)` (`lda_topic.rs`, struct field `:229`, set
  `:440`, = sklearn `components_` shape `(n_components, n_features)` `_lda.py:419-421`)
  and `n_iter_ = self.max_iter` (`lda_topic.rs:443`, = sklearn `n_iter_` which under the
  default `evaluate_every=-1` reaches `max_iter` since no perplexity break fires,
  `_lda.py:660`/`:695`; Probe 1 `n_iter_ = 10 = max_iter`). The seeded
  `Xoshiro256PlusPlus` (`lda_topic.rs`, init at line 414, seed
  `random_state.unwrap_or(0)` `:411`) plus the deterministic batch/online EM make the
  fit reproducible given a seed. The `digamma` helper (`lda_topic.rs`, fn at line 277)
  is accurate to `~1.17e-10` vs `scipy.special.psi` (Probe 4). **Scope: STRUCTURAL
  (shape / finiteness / determinism / digamma accuracy), NOT component VALUES (REQ-4).**
  Pinned by `test_lda_basic_shape` `(2,6)`, `test_lda_online_learning` `(2,6)`,
  `test_lda_fitted_accessors` (`n_iter() > 0`), `test_digamma_basic`
  (digamma(1) ≈ −0.5772), `test_digamma_large` (digamma(10) ≈ 2.2518). Non-test
  consumer: re-export `lib.rs:91-92`.

- REQ-2: **Structural: `components_` (lambda) is NON-NEGATIVE (SHIPPED scoped).** Both
  the batch M-step `lambda[k,w] = beta + ss[k,w]` (`lda_topic.rs`, `fn fit_batch`,
  `:509`) and the online EWA `lambda = (1−rho)·lambda + rho·(beta + n_docs·ss)`
  (`lda_topic.rs`, `fn fit_online`, `:579-580`) start from a non-negative
  `Uniform(0.5,1.5)+beta` init and only add non-negative sufficient statistics, so every
  `components_` entry is `≥ 0` — mirroring sklearn's `components_ = topic_word_prior_ +
  suff_stats` (`_lda.py:528`, non-negative Gamma init + non-negative suff-stats). Probe 1
  confirms sklearn `components_` are non-negative. **Scope: STRUCTURAL non-negativity,
  NOT value parity (REQ-4).** Pinned by `test_lda_components_non_negative`. Non-test
  consumer: re-export `lib.rs:91-92`.

- REQ-3: **`transform` doc-topic shape `(n_docs, n_topics)` with each row NORMALISED to
  sum 1; topic SEPARATION; single-topic degenerate; error / parameter contracts
  (SHIPPED scoped).** `transform` (`impl Transform for FittedLatentDirichletAllocation`,
  fn at line 615) runs the E-step per document via `e_step_doc` (fn at line 303) and
  NORMALISES each gamma row to sum 1 (`lda_topic.rs:642-654`) — mirroring sklearn's
  `doc_topic_distr /= doc_topic_distr.sum(axis=1)[:, np.newaxis]` (`_lda.py:745`; Probe 1
  row sums `[1.0, …]`); shape is `(n_docs, n_topics)` (`lda_topic.rs:637`). On a
  well-separated two-topic corpus the two document groups load on DIFFERENT dominant
  topics (`test_lda_topics_distinguish_groups`, structural separation), and with
  `n_components=1` every proportion is `≈ 1` (`test_lda_single_topic`). `fn fit` returns
  `InvalidParameter { name: "n_components" }` for `n_components == 0`
  (`lda_topic.rs:380-385`), `InsufficientSamples { required: 1 }` for `0` docs
  (`:386-392`), `InvalidParameter { name: "X" }` for `0` words (`:393-398`) and for any
  negative entry (`:399-406`); `transform` returns `ShapeMismatch` on a column-count
  mismatch (`:617-623`) and `InvalidParameter` on a negative entry (`:624-631`).
  **Scope: STRUCTURAL (shape / row-normalisation / topic separation / degenerate /
  error contracts), NOT doc-topic VALUE parity (REQ-5).** Pinned by
  `test_lda_transform_shape` `(6,2)`, `test_lda_topic_proportions_sum_to_one`,
  `test_lda_online_learning` (row sums ≈ 1), `test_lda_topics_distinguish_groups`,
  `test_lda_single_topic`, `test_lda_transform_shape_mismatch`,
  `test_lda_transform_negative_rejected`, `test_lda_invalid_n_components_zero`,
  `test_lda_negative_input_rejected`, `test_lda_empty_corpus`, `test_lda_zero_words`.
  **FLAG (candidate DIVs):** sklearn validates via `_check_non_neg_array`
  (`_lda.py:554`/`:644`) raising `ValueError`, NOT `FerroError`; sklearn defaults
  `n_components=10` (ferrolearn requires an explicit `usize`); sklearn does not pre-reject
  `0` words. Non-test consumer: re-export `lib.rs:91-92`.

- REQ-4: **EXACT `components_` value parity with sklearn's variational EM
  (NOT-STARTED, CARVE-OUT; `#1541`).** sklearn's `fit` (`_lda.py:625-705`) inits
  `components_ = random_state_.gamma(100., 0.01, (n_components, n_features))`
  (`_lda.py:419-421`, numpy `RandomState` Gamma), then per outer iteration runs
  `_em_step` (`:494-546`): E-step suff-stats (the per-doc gamma init is itself a
  `random_state.gamma(100., 0.01, ...)` draw when `random_init`, `_lda.py:96-99`) and
  M-step `components_ = topic_word_prior_ + suff_stats` (batch, `:528`) or the online
  EWA (`:532-539`). ferrolearn's `fn fit` (`lda_topic.rs`, impl at line 364) inits
  `lambda = Uniform(0.5, 1.5) + beta` via `Xoshiro256PlusPlus` (`lda_topic.rs:414-419`)
  — NOT Gamma(100, 0.01) — and `e_step_doc` (fn at line 303) inits the per-doc gamma
  UNIFORMLY at `alpha + n_words/n_topics` (`lda_topic.rs:308`) — NOT a Gamma draw.
  Probe 1 sklearn `components_ row0 = [14.49861, 15.498587, 14.498625, 0.501413,
  0.501386, 0.501378]` is NOT reproduced element-wise. **CARVE-OUT (R-DEFER-3):**
  matching sklearn requires the numpy-`RandomState` Gamma(100, 0.01) init for both
  `components_` and the per-doc gamma plus exact RNG-stream replication; no failing test
  is asserted (same class as the `minibatch_nmf` / `dictionary_learning` / `sparse_pca`
  RNG carve-outs). The E-step fixed-point and the `digamma` are otherwise equivalent
  (probe 4); the divergence is the init distribution + RNG only.

- REQ-5: **`transform` doc-topic VALUE parity (NOT-STARTED, CARVE-OUT — folds into
  REQ-4; `#1542`).** sklearn's `transform` (`_lda.py:724-746`) E-steps the gamma per
  document and normalises to sum 1 (Probe 1 `transform row0 = [0.968663, 0.031337]`).
  ferrolearn's `transform` (`impl Transform for FittedLatentDirichletAllocation`, fn at
  line 615) produces the SAME normalised shape (REQ-3 SHIPPED) but DIFFERENT values
  because (a) it runs on the carved-out `components_` (REQ-4) and (b) `e_step_doc` inits
  the per-doc gamma uniformly (`lda_topic.rs:308`) rather than from a Gamma draw
  (`_lda.py:96-99`). **CARVE-OUT (R-DEFER-3):** the doc-topic values are downstream of
  the carved-out `components_`, and the struct fields are PRIVATE — there is NO public
  API to construct a `FittedLatentDirichletAllocation` from an injected `lambda`, so a
  transform-value pin against sklearn's E-step is unreachable without the carved-out
  `components_`. Transform value parity therefore FOLDS INTO REQ-4 (like
  `dictionary_learning` REQ-4 transform). The critic decides whether to add an
  injectable-`components_` constructor for a deterministic transform pin.

- REQ-6: **Gamma `random_state.gamma(100, 0.01)` init for `components_` AND the per-doc
  gamma (NOT-STARTED; `#1543`).** sklearn inits the topic-word `components_` via
  `random_state_.gamma(100., 0.01, (n_components, n_features))` (`_lda.py:419-421`,
  `init_gamma=100.0`, `init_var=1/100`) and the per-document `doc_topic_distr` via
  `random_state.gamma(100., 0.01, (n_samples, n_topics))` when `random_init`
  (`_lda.py:96-99`) — both numpy `RandomState` Gamma draws (mean ≈ 1, var ≈ 0.01, Probe
  2). ferrolearn inits `lambda = Uniform(0.5, 1.5) + beta` via `Xoshiro256PlusPlus`
  (`lda_topic.rs:414-419`) and the per-doc gamma UNIFORMLY at `alpha + n_words/n_topics`
  (`e_step_doc`, `lda_topic.rs:308`) — neither is a Gamma(100, 0.01) draw, and the RNG
  is Xoshiro, not numpy `RandomState`.

- REQ-7: **`exp_dirichlet_component_` representation / fitted attr (NOT-STARTED;
  `#1544`).** sklearn maintains `self.exp_dirichlet_component_ =
  exp(_dirichlet_expectation_2d(self.components_))` (`_lda.py:424-426`/`:542-544`) — the
  EXPONENTIATED `E[log beta]` cached as a fitted attribute (Probe 1 shape `(2, 6)`) and
  used directly in the E-step `norm_phi` dot products (`_lda.py:144`/`:146`).
  ferrolearn computes the LOG-space `compute_e_log_beta = digamma(lambda[k,w]) −
  digamma(row_sum)` (`lda_topic.rs`, fn at line 589) on the fly inside `e_step_doc` and
  the M-step, and exponentiates within the phi log-sum-exp normalisation
  (`lda_topic.rs:324-337`) — algebraically the SAME `E[log beta]`, but it is NOT exposed
  as an `exp_dirichlet_component_` fitted attribute and is not cached on the fitted
  struct.

- REQ-8: **`perplexity` / `score` / `_approx_bound` log-likelihood bound
  (NOT-STARTED; `#1545`).** sklearn exposes `perplexity(X)` (`_lda.py:896`, Probe 1
  `4.1874`), `score(X)` (`_lda.py:827`, the variational lower bound, Probe 1
  `-123.1589`), and the internal `_approx_bound` (`_lda.py:748`) /
  `_perplexity_precomp_distr` (`_lda.py:790`) that estimate the per-word log-likelihood
  bound. `FittedLatentDirichletAllocation` (`lda_topic.rs`, struct at line 225) exposes
  only `components()` / `n_iter()` / `alpha()` / `beta()` (fns at lines 246-266) — there
  is NO `perplexity`, NO `score`, and NO `_approx_bound` method.

- REQ-9: **`evaluate_every` / `perp_tol` perplexity-based early stop (NOT-STARTED;
  `#1546`).** sklearn's `fit` loop (`_lda.py:660-695`), when `evaluate_every > 0` and
  `(i+1) % evaluate_every == 0`, computes the perplexity `bound` and breaks if
  `|last_bound − bound| < perp_tol` (default 1e-1, `_lda.py:676-691`). ferrolearn's
  `fn fit_batch` / `fn fit_online` (`lda_topic.rs`, fns at lines 452/517) run a FIXED
  `for _outer in 0..self.max_iter` loop (`lda_topic.rs:462`/`:530`) with NO perplexity
  evaluation and NO early stop — there is no `evaluate_every` or `perp_tol` field, so
  `n_iter_` is always exactly `max_iter` (matching sklearn ONLY under the default
  `evaluate_every=-1`).

- REQ-10: **`batch_size` / `total_samples` online mini-batching (NOT-STARTED;
  `#1547`).** sklearn's online `fit` slices the corpus into `gen_batches(n_samples,
  batch_size)` (default `batch_size=128`, `_lda.py:662`) and the online M-step EWA uses
  `doc_ratio = total_samples / X.shape[0]` (default `total_samples=1e6`,
  `_lda.py:535-538`). ferrolearn's `fn fit_online` (`lda_topic.rs`, fn at line 517)
  processes each document as a MINI-BATCH OF 1 (`for d in 0..n_docs`,
  `lda_topic.rs:532`) and uses `doc_ratio = n_docs` (`lda_topic.rs:576-579`) — there is
  NO `batch_size` field (no 128-document batching) and NO `total_samples` field (the
  doc-ratio is hard-wired to the in-memory corpus size).

- REQ-11: **Fitted attrs `n_features_in_` / `bound_` / `doc_topic_prior_` /
  `topic_word_prior_` (NOT-STARTED; `#1548`).** sklearn exposes `n_features_in_`
  (Probe 1 `6`), `bound_` (the final-fit perplexity, Probe 1 `4.1874`,
  `_lda.py:701-703`), `doc_topic_prior_` (Probe 1 `0.5`, `_lda.py:406-409`), and
  `topic_word_prior_` (Probe 1 `0.5`, `_lda.py:411-414`).
  `FittedLatentDirichletAllocation` (`lda_topic.rs`, struct at line 225) stores `alpha_`
  / `beta_` (the resolved priors, exposed as `alpha()` / `beta()` fns at lines 258/264
  — these DO mirror `doc_topic_prior_` / `topic_word_prior_`) but has NO `n_features_in_`
  and NO `bound_` accessor (no final-fit perplexity is computed at all — see REQ-8).

- REQ-12: **`n_jobs` / `verbose` (NOT-STARTED; `#1549`).** sklearn's
  `LatentDirichletAllocation(n_jobs=None, verbose=0)` (`_lda.py:378-379`) parallelises
  the E-step over `joblib.Parallel(n_jobs=...)` (`_lda.py:658-659`) and prints
  per-iteration progress when `verbose` (`_lda.py:683-694`). ferrolearn's
  `LatentDirichletAllocation` (`lda_topic.rs`, struct at line 68) has NO `n_jobs` /
  `verbose` fields — the E-step is a single-threaded `for d in 0..n_docs` loop
  (`lda_topic.rs:469`/`:532`) with no progress reporting.

- REQ-13: **Generic `F: Float` (f32 + f64) (NOT-STARTED; `#1550`).** Per CLAUDE.md
  (Numeric Generics: `F: Float + Send + Sync + 'static`, support both f32 and f64) and
  sklearn's `_more_tags` `preserves_dtype: [np.float64, np.float32]` (`_lda.py:548-552`),
  the estimator should be generic. ferrolearn's `LatentDirichletAllocation` is
  **f64-ONLY**: the struct fields are `f64` (`lda_topic.rs:76-86`), the fitted
  `components_` is `Array2<f64>` (`:229`), and the trait impls are `impl Fit<Array2<f64>,
  ()>` (`lda_topic.rs:364`) / `impl Transform<Array2<f64>>` (`:601`) — not `<F>`-generic
  (contrast the sibling `MiniBatchNMF<F>` / `SparsePCA<F>`).

- REQ-14: **PyO3 binding (NOT-STARTED; `#1551`).** sklearn exposes
  `LatentDirichletAllocation` through `import sklearn.decomposition`. ferrolearn has NO
  PyO3 binding — a `grep -rn LatentDirichlet ferrolearn-python/src/` is empty; the only
  non-test consumer of `LatentDirichletAllocation` / `FittedLatentDirichletAllocation`
  is the crate re-export (`lib.rs:91-92`). The CPython surface (a
  `_RsLatentDirichletAllocation` class with a ctor + `fit` + `transform`) is absent.

- REQ-15: **ferray substrate (NOT-STARTED; `#1552`).** `lda_topic.rs` computes on
  `ndarray::Array2` (`lda_topic.rs:40`) and uses `rand` + `rand_distr` + `rand_xoshiro`
  (`lda_topic.rs:41-43`, `Xoshiro256PlusPlus` + `Uniform`) for the `lambda` init, plus a
  hand-rolled `digamma` (`lda_topic.rs:277`), not `ferray-core` arrays / `ferray::random`
  / `ferray::stats` (R-SUBSTRATE-1/2). sklearn's digamma is `scipy.special.psi` via the
  Cython `_dirichlet_expectation_2d` (`_lda.py:104`/`:424`).

## Acceptance criteria

All expected values from the live sklearn 1.5.2 oracle (run from `/tmp`), never
literal-copied from ferrolearn (R-CHAR-3).

- AC-1 (REQ-1, SHIPPED scoped): `LatentDirichletAllocation::new(2).with_random_state(42)
  .fit(&dtm, &()).unwrap().components().dim()` is `(2, n_words)`; `n_iter() > 0`; two
  fits with the same seed are identical; `digamma(1.0) ≈ −0.5772156649` and
  `digamma(10.0) ≈ 2.2517525891` (Probe 4 / `scipy.special.psi`). Pinned by
  `test_lda_basic_shape` `(2,6)`, `test_lda_online_learning` `(2,6)`,
  `test_lda_fitted_accessors`, `test_digamma_basic`, `test_digamma_large`. (Structural
  shape / finiteness / determinism / digamma accuracy only — NOT the exact component
  values, REQ-4.)

- AC-2 (REQ-2, SHIPPED scoped): every entry of `fitted.components()` is `≥ 0`. Probe 1
  confirms sklearn `components_` are non-negative. Pinned by
  `test_lda_components_non_negative`. (Structural non-negativity only — NOT value
  parity, REQ-4.)

- AC-3 (REQ-3, SHIPPED scoped): `fitted.transform(&dtm)` has shape `(n_docs, n_topics)`
  and every row sums to `≈ 1.0` (within 1e-5, Probe 1 row sums `[1.0, …]`); the two
  well-separated document groups load on different dominant topics; with `n_components=1`
  every proportion is `≈ 1`; `fit` returns `Err` for `n_components=0`, `0` docs, `0`
  words, and negative input; `transform` returns `Err` for a column-count mismatch and
  for negative input. Pinned by `test_lda_transform_shape` `(6,2)`,
  `test_lda_topic_proportions_sum_to_one`, `test_lda_online_learning`,
  `test_lda_topics_distinguish_groups`, `test_lda_single_topic`,
  `test_lda_transform_shape_mismatch`, `test_lda_transform_negative_rejected`,
  `test_lda_invalid_n_components_zero`, `test_lda_negative_input_rejected`,
  `test_lda_empty_corpus`, `test_lda_zero_words`. FLAG: sklearn raises `ValueError` (not
  `FerroError`), defaults `n_components=10`, and does not pre-reject `0` words.

- AC-4 (REQ-4, NOT-STARTED, CARVE-OUT): `LatentDirichletAllocation(n_components=2,
  random_state=0).fit(dtm).components_` (Probe 1: shape `(2,6)`, `row0 = [14.49861,
  15.498587, 14.498625, 0.501413, 0.501386, 0.501378]`) is NOT reproduced element-wise
  by ferrolearn (Uniform(0.5,1.5)+beta + Xoshiro lambda init + uniform per-doc gamma init
  vs sklearn Gamma(100,0.01) + numpy `RandomState`). No failing test asserts this
  (R-DEFER-3).

- AC-5 (REQ-5, NOT-STARTED, CARVE-OUT — folds into REQ-4): `transform(dtm)` (Probe 1
  `row0 = [0.968663, 0.031337]`) is NOT reproduced element-wise — the doc-topic values
  are downstream of the carved-out `components_` (REQ-4) and the uniform per-doc gamma
  init; no injectable-`components_` API exists (struct fields private), so transform
  value parity folds into REQ-4. No failing test (R-DEFER-3).

- AC-6 (REQ-6..12, DIVERGES): `LatentDirichletAllocation()` defaults `n_components=10,
  doc_topic_prior=None, topic_word_prior=None, learning_method="batch",
  learning_decay=0.7, learning_offset=10.0, max_iter=10, batch_size=128,
  evaluate_every=-1, total_samples=1e6, perp_tol=1e-1, mean_change_tol=1e-3,
  max_doc_update_iter=100, n_jobs=None, verbose=0` (Probe 3, `_lda.py:362-397`); sklearn
  inits `components_` / per-doc gamma via Gamma(100, 0.01), caches
  `exp_dirichlet_component_`, exposes `perplexity` / `score` / `bound_` /
  `n_features_in_` / `doc_topic_prior_` / `topic_word_prior_`, supports
  `evaluate_every` / `perp_tol` early stop and `batch_size` / `total_samples` online
  mini-batching, and parallelises via `n_jobs`. ferrolearn has Uniform+beta/uniform init,
  no `exp_dirichlet_component_` attr, no `perplexity`/`score`/`bound_`/`n_features_in_`,
  no `evaluate_every`/`perp_tol`/`batch_size`/`total_samples`/`n_jobs`/`verbose` fields
  (and `mean_change_tol` hard-coded to 1e-3, `lda_topic.rs:352`).

- AC-7 (REQ-13/14/15): `LatentDirichletAllocation` is f64-only (`impl Fit<Array2<f64>,
  ()>`, `lda_topic.rs:364`), not `<F>`-generic; `import ferrolearn` exposes NO
  `_RsLatentDirichletAllocation` (`grep -rn LatentDirichlet ferrolearn-python/src/` is
  empty); the only non-test consumer is the crate re-export (`lib.rs:91-92`). The module
  imports `ndarray` (`lda_topic.rs:40`) + `rand`/`rand_distr`/`rand_xoshiro` (`:41-43`)
  and a hand-rolled `digamma`, not ferray.

## REQ status

Binary (R-DEFER-2). `LatentDirichletAllocation` / `FittedLatentDirichletAllocation` are
existing pub APIs; the non-test consumer is the crate re-export (`lib.rs:91-92`,
boundary public API, grandfathered S5/R-DEFER-1) — there is NO PyO3 binding (REQ-14
NOT-STARTED). This is the TOPIC-MODEL LDA, NOT Linear Discriminant Analysis. Cites use
symbol anchors (ferrolearn) / `file:line` (sklearn 1.5.2). Live oracle = installed
sklearn 1.5.2, run from `/tmp`.
**EXACT `components_` VALUE PARITY DIVERGES (R-HONEST-3, REQ-4 NOT-STARTED, CARVE-OUT
`#1541`):** ferrolearn's `lambda = Uniform(0.5,1.5)+beta` Xoshiro init + uniform per-doc
gamma init (`lda_topic.rs:414-419`/`:308`) ≠ sklearn's `random_state_.gamma(100., 0.01)`
init for both `components_` and the per-doc gamma + numpy `RandomState`
(`_lda.py:419-421`/`:96-99`). The E-step fixed-point and the `digamma` are otherwise
equivalent — the **INVESTIGATE is settled (R-CHAR-3): the digamma approximation matches
`scipy.special.psi` to ~1.17e-10 (Probe 4) and is NOT a divergence source; the
log-space `compute_e_log_beta` vs sklearn's exponentiated `exp_dirichlet_component_` is
the same `E[log beta]` re-arranged**, so REQ-4 reduces to the init-distribution + RNG
carve-out and the `transform` doc-topic VALUES (REQ-5, `#1542`) fold into it (no
injectable-`components_` API). The least-confident SHIPPED claim is REQ-3 — it is
STRUCTURAL (doc-topic shape + each row normalised to sum 1 + topic separation), NOT
oracle doc-topic value parity; the in-tree tests assert row-sums ≈ 1 and group
separation, not the per-row proportions sklearn produces. #1540 is this doc's crosslink
tracking issue. Count: **3 SHIPPED (REQ-1,2,3) / 12 NOT-STARTED
(REQ-4,5,6,7,8,9,10,11,12,13,14,15)**.

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (structural: `components_` shape / non-neg / `n_iter_` / determinism / digamma accuracy) | SHIPPED | `fn fit` (`lda_topic.rs` impl at line 364) stores `components_ = lambda` shape `(n_topics, n_words)` (field `:229`, set `:440` = sklearn `components_` `(n_components, n_features)` `_lda.py:419-421`), `n_iter_ = self.max_iter` (`:443` = sklearn `n_iter_` under default `evaluate_every=-1`, no perplexity break, `_lda.py:660`/`:695`; Probe 1 `n_iter_ = 10 = max_iter`). Seeded `Xoshiro256PlusPlus` (init at line 414, seed `unwrap_or(0)` `:411`) + deterministic EM ⇒ reproducible. `digamma` (fn at line 277) matches `scipy.special.psi` to ~1.17e-10 (Probe 4). **Scope: STRUCTURAL, NOT value parity (REQ-4).** Non-test consumer: re-export `lib.rs:91-92`. Verification: `cargo test -p ferrolearn-decomp lda` → `test_lda_basic_shape` `(2,6)`, `test_lda_online_learning` `(2,6)`, `test_lda_fitted_accessors`, `test_digamma_basic`, `test_digamma_large` PASS. |
| REQ-2 (structural: `components_` non-negativity) | SHIPPED | Batch M-step `lambda[k,w] = beta + ss[k,w]` (`fn fit_batch` `lda_topic.rs:509`) and online EWA `lambda = (1−rho)·lambda + rho·(beta + n_docs·ss)` (`fn fit_online` `:579-580`) add only non-negative suff-stats to a non-negative `Uniform(0.5,1.5)+beta` init ⇒ every entry `≥ 0` — sklearn `components_ = topic_word_prior_ + suff_stats` (`_lda.py:528`). Probe 1 sklearn `components_` non-negative. **Scope: STRUCTURAL, NOT value parity (REQ-4).** Non-test consumer: re-export `lib.rs:91-92`. Verification: `cargo test -p ferrolearn-decomp lda` → `test_lda_components_non_negative` PASS. |
| REQ-3 (transform doc-topic shape + rows sum to 1 + topic separation + error contracts) | SHIPPED | `transform` (`impl Transform for FittedLatentDirichletAllocation` fn at line 615) E-steps per doc via `e_step_doc` (fn at line 303) and NORMALISES each gamma row to sum 1 (`lda_topic.rs:642-654` = sklearn `doc_topic_distr /= doc_topic_distr.sum(axis=1)[:,newaxis]` `_lda.py:745`; Probe 1 row sums `[1.0,…]`); shape `(n_docs, n_topics)` (`:637`). Well-separated groups load on different dominant topics; `n_components=1` ⇒ all props ≈ 1. `fn fit` returns `Err(InvalidParameter{name:"n_components"})` for 0 topics (`:380-385`), `Err(InsufficientSamples{required:1})` for 0 docs (`:386-392`), `Err(InvalidParameter{name:"X"})` for 0 words (`:393-398`) / negative entry (`:399-406`); `transform` returns `Err(ShapeMismatch)` on col mismatch (`:617-623`) / `Err(InvalidParameter)` on negative (`:624-631`). **Scope: STRUCTURAL, NOT doc-topic value parity (REQ-5).** Non-test consumer: re-export `lib.rs:91-92`. Verification: `cargo test -p ferrolearn-decomp lda` → `test_lda_transform_shape` `(6,2)`, `test_lda_topic_proportions_sum_to_one`, `test_lda_online_learning`, `test_lda_topics_distinguish_groups`, `test_lda_single_topic`, `test_lda_transform_shape_mismatch`, `test_lda_transform_negative_rejected`, `test_lda_invalid_n_components_zero`, `test_lda_negative_input_rejected`, `test_lda_empty_corpus`, `test_lda_zero_words` PASS. **FLAG (candidate DIVs):** sklearn raises `ValueError` (not `FerroError`) via `_check_non_neg_array` (`_lda.py:554`/`:644`); defaults `n_components=10`; does NOT pre-reject `0` words. |
| REQ-4 (EXACT `components_` value parity) | NOT-STARTED | open prereq blocker **#1541** (CARVE-OUT, R-DEFER-3). sklearn `fit` (`_lda.py:625-705`): `components_ = random_state_.gamma(100., 0.01, (n_components, n_features))` (`:419-421`), `_em_step` (`:494-546`) M-step `components_ = topic_word_prior_ + suff_stats` (batch `:528`) / online EWA (`:532-539`), per-doc gamma init `random_state.gamma(100., 0.01, ...)` (`:96-99`). ferrolearn `fn fit` (`lda_topic.rs` impl at line 364): `lambda = Uniform(0.5,1.5)+beta` Xoshiro init (`:414-419`) — NOT Gamma — + `e_step_doc` (fn at line 303) uniform per-doc gamma `alpha + n_words/n_topics` (`:308`). Probe 1 sklearn `components_ row0 = [14.49861, 15.498587, 14.498625, 0.501413, 0.501386, 0.501378]` NOT reproduced. No failing test (same class as `minibatch_nmf` / `dictionary_learning` / `sparse_pca` RNG carve-outs). digamma (~1.17e-10 vs scipy, Probe 4) + E-step fixed-point are equivalent; divergence is init distribution + RNG only. |
| REQ-5 (transform doc-topic VALUE parity) | NOT-STARTED | open prereq blocker **#1542** (CARVE-OUT — folds into REQ-4). sklearn `transform` (`_lda.py:724-746`): E-step gamma + normalise to sum 1 (Probe 1 `row0 = [0.968663, 0.031337]`). ferrolearn `transform` (`impl Transform for FittedLatentDirichletAllocation` fn at line 615) gives the SAME normalised shape (REQ-3) but DIFFERENT values — downstream of the carved-out `components_` (REQ-4) + uniform per-doc gamma init (`lda_topic.rs:308` vs `_lda.py:96-99`). Struct fields PRIVATE — NO injectable-`components_` API, so a transform-value pin is unreachable without the carved-out `components_`; folds into REQ-4 (like `dictionary_learning` REQ-4 transform). The critic decides whether to add an injectable-`components_` constructor. |
| REQ-6 (Gamma(100,0.01) init: `components_` + per-doc gamma) | NOT-STARTED | open prereq blocker **#1543**. sklearn inits `components_` via `random_state_.gamma(100., 0.01, (n_components, n_features))` (`_lda.py:419-421`, `init_gamma=100`, `init_var=1/100`) and the per-doc `doc_topic_distr` via `random_state.gamma(100., 0.01, (n_samples, n_topics))` (`:96-99`) — numpy `RandomState` (mean ≈ 1, var ≈ 0.01, Probe 2). ferrolearn inits `lambda = Uniform(0.5,1.5)+beta` Xoshiro (`lda_topic.rs:414-419`) + uniform per-doc gamma `alpha + n_words/n_topics` (`e_step_doc` `:308`) — neither is Gamma(100,0.01), RNG is Xoshiro not numpy `RandomState`. |
| REQ-7 (`exp_dirichlet_component_` representation / fitted attr) | NOT-STARTED | open prereq blocker **#1544**. sklearn caches `exp_dirichlet_component_ = exp(_dirichlet_expectation_2d(components_))` (`_lda.py:424-426`/`:542-544`, Probe 1 shape `(2,6)`), used in the E-step `norm_phi` dots (`:144`/`:146`). ferrolearn computes log-space `compute_e_log_beta = digamma(lambda[k,w]) − digamma(row_sum)` (`lda_topic.rs` fn at line 589) on the fly + exponentiates in the phi log-sum-exp (`:324-337`) — same `E[log beta]` re-arranged, but NOT exposed/cached as `exp_dirichlet_component_`. |
| REQ-8 (`perplexity` / `score` / `_approx_bound`) | NOT-STARTED | open prereq blocker **#1545**. sklearn exposes `perplexity(X)` (`_lda.py:896`, Probe 1 `4.1874`), `score(X)` (`_lda.py:827`, Probe 1 `-123.1589`), and internal `_approx_bound` (`_lda.py:748`) / `_perplexity_precomp_distr` (`_lda.py:790`). `FittedLatentDirichletAllocation` (`lda_topic.rs` struct at line 225) exposes only `components()`/`n_iter()`/`alpha()`/`beta()` (fns at 246-266) — NO `perplexity`, NO `score`, NO `_approx_bound`. |
| REQ-9 (`evaluate_every` / `perp_tol` perplexity early stop) | NOT-STARTED | open prereq blocker **#1546**. sklearn `fit` (`_lda.py:660-695`), when `evaluate_every>0` and `(i+1)%evaluate_every==0`, computes perplexity `bound` and breaks if `|last_bound−bound|<perp_tol` (default 1e-1, `:676-691`). ferrolearn `fn fit_batch`/`fn fit_online` (`lda_topic.rs` fns at 452/517) run a FIXED `for _outer in 0..self.max_iter` loop (`:462`/`:530`) — NO `evaluate_every`/`perp_tol` fields, NO perplexity, NO early stop (`n_iter_` always == `max_iter`, matching sklearn only under default `evaluate_every=-1`). |
| REQ-10 (`batch_size` / `total_samples` online mini-batching) | NOT-STARTED | open prereq blocker **#1547**. sklearn online `fit` slices `gen_batches(n_samples, batch_size)` (default 128, `_lda.py:662`) + online EWA `doc_ratio = total_samples / X.shape[0]` (default `total_samples=1e6`, `:535-538`). ferrolearn `fn fit_online` (`lda_topic.rs` fn at line 517) processes each doc as a MINI-BATCH OF 1 (`for d in 0..n_docs` `:532`) + `doc_ratio = n_docs` (`:576-579`) — NO `batch_size` (no 128-doc batching), NO `total_samples` field. |
| REQ-11 (fitted attrs `n_features_in_` / `bound_` / `doc_topic_prior_` / `topic_word_prior_`) | NOT-STARTED | open prereq blocker **#1548**. sklearn exposes `n_features_in_` (Probe 1 `6`), `bound_` (final-fit perplexity, Probe 1 `4.1874`, `_lda.py:701-703`), `doc_topic_prior_` (`0.5`, `:406-409`), `topic_word_prior_` (`0.5`, `:411-414`). `FittedLatentDirichletAllocation` (`lda_topic.rs` struct at line 225) exposes `alpha()`/`beta()` (fns at 258/264) mirroring the resolved priors, but has NO `n_features_in_` and NO `bound_` (no final-fit perplexity is computed — see REQ-8). |
| REQ-12 (`n_jobs` / `verbose`) | NOT-STARTED | open prereq blocker **#1549**. sklearn `LatentDirichletAllocation(n_jobs=None, verbose=0)` (`_lda.py:378-379`) parallelises the E-step over `joblib.Parallel(n_jobs=...)` (`:658-659`) + prints per-iter progress when `verbose` (`:683-694`). ferrolearn `LatentDirichletAllocation` (`lda_topic.rs` struct at line 68) has NO `n_jobs`/`verbose` fields — single-threaded E-step `for d in 0..n_docs` (`:469`/`:532`), no progress reporting. |
| REQ-13 (generic `F: Float`, f32 + f64) | NOT-STARTED | open prereq blocker **#1550**. CLAUDE.md mandates `F: Float + Send + Sync + 'static` (f32 + f64); sklearn `_more_tags` `preserves_dtype: [np.float64, np.float32]` (`_lda.py:548-552`). ferrolearn is **f64-ONLY**: struct fields `f64` (`lda_topic.rs:76-86`), `components_` `Array2<f64>` (`:229`), `impl Fit<Array2<f64>, ()>` (`:364`) / `impl Transform<Array2<f64>>` (`:601`) — not `<F>`-generic (contrast `MiniBatchNMF<F>` / `SparsePCA<F>`). |
| REQ-14 (PyO3 binding) | NOT-STARTED | open prereq blocker **#1551**. sklearn exposes `LatentDirichletAllocation` via `import sklearn.decomposition`. ferrolearn has NO PyO3 binding — `grep -rn LatentDirichlet ferrolearn-python/src/` is empty; the only non-test consumer of `LatentDirichletAllocation`/`FittedLatentDirichletAllocation` is the crate re-export (`lib.rs:91-92`). No `_RsLatentDirichletAllocation` class. |
| REQ-15 (ferray substrate) | NOT-STARTED | open prereq blocker **#1552**. `lda_topic.rs` computes on `ndarray::Array2` (`lda_topic.rs:40`), uses `rand`/`rand_distr`/`rand_xoshiro` `Xoshiro256PlusPlus`+`Uniform` (`:41-43`) for the `lambda` init, and a hand-rolled `digamma` (`:277`), not `ferray-core` arrays / `ferray::random` / `ferray::stats` (R-SUBSTRATE-1/2). sklearn's digamma is `scipy.special.psi` via Cython `_dirichlet_expectation_2d` (`_lda.py:104`/`:424`). |

## Architecture

`lda_topic.rs` follows the unfitted/fitted split (CLAUDE.md naming):
`LatentDirichletAllocation { n_components, max_iter (10), learning_method:
LdaLearningMethod{Batch|Online}, learning_offset (10.0), learning_decay (0.7),
doc_topic_prior (None → 1/K), topic_word_prior (None → 1/K), max_doc_update_iter (100),
random_state }` (struct at line 68; `new(n_components)` fn at line 96, builders
`with_max_iter` `:112` / `with_learning_method` `:119` / `with_learning_offset` `:126` /
`with_learning_decay` `:133` / `with_doc_topic_prior` `:140` / `with_topic_word_prior`
`:147` / `with_random_state` `:154` / `with_max_doc_update_iter` `:161`, accessors
`n_components()`..`random_state()` fns at lines 168-212) → `Fit<Array2<f64>, ()>` →
`FittedLatentDirichletAllocation { components_, alpha_, beta_, n_iter_,
max_doc_update_iter_ }` (struct at line 225, accessors `components()`/`n_iter()`/
`alpha()`/`beta()` fns at lines 246-266). The path is **f64-ONLY** (`impl
Fit<Array2<f64>, ()>` `:364`, `impl Transform<Array2<f64>>` `:601`) — NOT `<F>`-generic
(REQ-13 NOT-STARTED); `fit`/`transform` return `Result<_, FerroError>` (R-CODE-2). This
is the TOPIC-MODEL LDA (online variational Bayes), NOT Linear Discriminant Analysis.

**Fit path (`fn fit`, impl at line 364) — REQ-1/2/3/4.** Validates `n_components != 0`,
`n_docs >= 1`, `n_words >= 1`, and non-negativity of `X` (`lda_topic.rs:380-406`) —
REQ-3. Resolves `alpha = doc_topic_prior or 1/K`, `beta = topic_word_prior or 1/K`
(`:409-410`, = sklearn `doc_topic_prior_`/`topic_word_prior_` `_lda.py:406-414`), then
inits `lambda = Uniform(0.5, 1.5) + beta` via `Xoshiro256PlusPlus::seed_from_u64(
random_state.unwrap_or(0))` (`:414-419`) — NOT sklearn's Gamma(100, 0.01) init (REQ-6
NOT-STARTED). Dispatches `fit_batch` (fn at line 452) or `fit_online` (fn at line 517)
per `learning_method`, then sets `n_iter_ = self.max_iter` (`:443`). `fit_batch` runs
`for _outer in 0..max_iter`: computes `compute_e_log_beta` (fn at line 589, the log-space
`E[log beta]`), accumulates suff-stats via the per-doc `e_step_doc` (fn at line 303,
log-space phi fixed-point with L1 mean-change-< 1e-3 break, `:352`), and M-steps
`lambda[k,w] = beta + ss[k,w]` (`:509`). `fit_online` processes each document as a
mini-batch of 1 and applies the EWA `lambda = (1−rho)·lambda + rho·(beta + n_docs·ss)`
with `rho = (learning_offset + update_count)^(−learning_decay)` (`:573-580`) — REQ-10
NOT-STARTED (no `batch_size`/`total_samples`). **This is NOT sklearn's `_em_step`
end-to-end (REQ-4):** sklearn Gamma-inits `components_` AND the per-doc gamma via numpy
`RandomState` (`_lda.py:419-421`/`:96-99`) and caches `exp_dirichlet_component_`
(`:424`/`:542`), so the converged `lambda` values diverge (CARVE-OUT). The `digamma`
(fn at line 277) and the E-step fixed-point are otherwise equivalent (Probe 4).

**Transform (`impl Transform for FittedLatentDirichletAllocation`, fn at line 615) —
REQ-3/5.** Validates the column count (`:617-623`) and non-negativity (`:624-631`) —
REQ-3, then for each document runs `e_step_doc` (fn at line 303) and NORMALISES the
gamma row to sum 1 (`:642-654`), returning `(n_docs, n_topics)` — mirroring sklearn's
`_unnormalized_transform` + the `/= sum(axis=1)` normalisation (`_lda.py:720`/`:745`).
The doc-topic VALUES are downstream of the carved-out `components_` (REQ-4) and the
uniform per-doc gamma init (REQ-5 CARVE-OUT, folds into REQ-4) — there is no injectable
`components_` (struct fields private), so a value pin is unreachable without the
carved-out fit.

**sklearn (target contract).** `class LatentDirichletAllocation(...)` (`_lda.py:163`)
takes `__init__(n_components=10, *, doc_topic_prior=None, topic_word_prior=None,
learning_method="batch", learning_decay=0.7, learning_offset=10.0, max_iter=10,
batch_size=128, evaluate_every=-1, total_samples=1e6, perp_tol=1e-1,
mean_change_tol=1e-3, max_doc_update_iter=100, n_jobs=None, verbose=0,
random_state=None)` (`:362-397`). `_init_latent_vars` (`:399-426`) sets
`n_iter_ = 0`, the resolved priors, and Gamma-inits `components_` +
`exp_dirichlet_component_`. `fit` (`:625-705`) loops `_em_step` (`:494-546`, E-step
suff-stats + batch/online M-step) over `max_iter`, optionally perplexity-early-stops
(`evaluate_every`/`perp_tol` `:676-691`), increments `n_iter_` (`:695`), and computes
the final `bound_` (`:701-703`). `transform` (`:724-746`) is the normalised E-step.
`perplexity`/`score`/`_approx_bound` (`:896`/`:827`/`:748`) are the log-likelihood
bound. `partial_fit` (`:560`) is the online out-of-core variant. Fitted attrs:
`components_`, `exp_dirichlet_component_`, `n_iter_`, `bound_`, `doc_topic_prior_`,
`topic_word_prior_`, `n_features_in_`, `n_batch_iter_`.

## Verification

Library crate (green at baseline `15506b3e`):
```bash
cargo test -p ferrolearn-decomp lda               # in-module #[test]s + doctest
cargo clippy -p ferrolearn-decomp --all-targets -- -D warnings
cargo fmt --all --check
```
The in-tree `#[test]`s pin REQ-1/2/3 (STRUCTURAL): `test_lda_basic_shape` `(2,6)`,
`test_lda_online_learning` `(2,6)`, `test_lda_fitted_accessors`, `test_digamma_basic`,
`test_digamma_large` (REQ-1); `test_lda_components_non_negative` (REQ-2);
`test_lda_transform_shape` `(6,2)`, `test_lda_topic_proportions_sum_to_one`,
`test_lda_topics_distinguish_groups`, `test_lda_single_topic`,
`test_lda_transform_shape_mismatch`, `test_lda_transform_negative_rejected`,
`test_lda_invalid_n_components_zero`, `test_lda_negative_input_rejected`,
`test_lda_empty_corpus`, `test_lda_zero_words` (REQ-3); plus `test_lda_getters` and the
module doctest. There is NO `tests/divergence_lda_topic.rs` yet. REQ-4 (components value
parity) and REQ-5 (transform doc-topic value parity) are CARVE-OUTs (R-DEFER-3, no
failing test) — the init-distribution + RNG difference, not a fixable deterministic
divergence; the `digamma` is accurate to ~1.17e-10 vs `scipy.special.psi` (Probe 4) and
the E-step fixed-point is equivalent, so there is no observable parity divergence to pin
beyond the RNG.

Live sklearn oracle (installed 1.5.2, run from `/tmp`) — the REQ-4 components value gap
and the REQ-3 structural normalisation:
```bash
# REQ-1/2/3 structural + REQ-4 value gap (sklearn components_ non-negative; transform rows sum to 1; values NOT reproduced):
python3 -c "import numpy as np; from sklearn.decomposition import LatentDirichletAllocation
dtm=np.array([[5.,5,5,0,0,0],[4,6,3,0,0,0],[5,4,6,0,0,0],[0,0,0,5,5,5],[0,0,0,6,4,3],[0,0,0,4,6,5]])
m=LatentDirichletAllocation(n_components=2, random_state=0).fit(dtm)
print(m.components_.shape, bool((m.components_>=0).all()), np.round(m.components_[0],6).tolist())
T=m.transform(dtm); print(T.shape, np.round(T.sum(axis=1),8).tolist(), np.round(T[0],6).tolist())"
# -> (2, 6) True [14.49861, 15.498587, 14.498625, 0.501413, 0.501386, 0.501378]
# -> (6, 2) [1.0, 1.0, 1.0, 1.0, 1.0, 1.0] [0.968663, 0.031337]

# REQ-1 digamma vs scipy.special.psi (~1e-10 accuracy, NOT a divergence source):
python3 -c "from scipy.special import psi; print(round(float(psi(1)),10), round(float(psi(10)),10))"
# -> -0.5772156649 2.2517525891
```
The REQ-4 components value gap is the Gamma(100,0.01) + numpy `RandomState` init
(`_lda.py:419-421`/`:96-99`) vs ferrolearn's Uniform(0.5,1.5)+beta Xoshiro init + uniform
per-doc gamma (`lda_topic.rs:414-419`/`:308`), a CARVE-OUT with no parity test (R-CHAR-3
/ R-DEFER-3). The REQ-5 transform value gap folds into REQ-4 (no injectable-`components_`
API).

ferrolearn-python (REQ-14, ABSENT at baseline): there is NO
`_RsLatentDirichletAllocation` binding — `grep -rn LatentDirichlet
ferrolearn-python/src/` is empty. The only non-test consumer of
`LatentDirichletAllocation`/`FittedLatentDirichletAllocation` is the crate re-export
(`lib.rs:91-92`).

## Blockers

(#1540 is this doc's crosslink tracking issue. The blockers below are the open work
items the dispatcher files / numbers; none are filed by this doc — markdown only.)

- **#1541** — REQ-4 (CARVE-OUT): reach EXACT `components_` value parity by initing
  `components_` via numpy-`RandomState`-equivalent `gamma(100., 0.01, (n_components,
  n_features))` (`_lda.py:419-421`) and the per-doc gamma via `gamma(100., 0.01, ...)`
  (`_lda.py:96-99`); inherently RNG/init-distribution-bound (no failing test, R-DEFER-3;
  the `digamma` and E-step fixed-point are already equivalent — Probe 4).
- **#1542** — REQ-5 (CARVE-OUT, folds into #1541): the `transform` doc-topic VALUES are
  downstream of the carved-out `components_` + the uniform per-doc gamma init; reaching
  parity needs #1541 (and the per-doc Gamma init), or an injectable-`components_`
  constructor to pin the deterministic E-step on a fixed `lambda`. No failing test
  (R-DEFER-3).
- **#1543** — REQ-6: replace the `Uniform(0.5,1.5)+beta` Xoshiro `lambda` init
  (`lda_topic.rs:414-419`) and the uniform per-doc gamma init (`e_step_doc` `:308`) with
  sklearn's `random_state.gamma(100., 0.01, ...)` for both (`_lda.py:419-421`/`:96-99`),
  on a numpy-`RandomState`-equivalent stream.
- **#1544** — REQ-7: cache `exp_dirichlet_component_ = exp(E[log beta])` as a fitted
  attribute (`_lda.py:424-426`/`:542-544`) and thread it through the E-step
  (`_lda.py:144`/`:146`) instead of recomputing `compute_e_log_beta` (`lda_topic.rs:589`)
  in log space on the fly.
- **#1545** — REQ-8: add `perplexity(X)` (`_lda.py:896`), `score(X)` (`_lda.py:827`),
  and the `_approx_bound` / `_perplexity_precomp_distr` (`_lda.py:748`/`:790`)
  log-likelihood bound to `FittedLatentDirichletAllocation`.
- **#1546** — REQ-9: add `evaluate_every` / `perp_tol` fields and the perplexity-based
  early stop in the fit loop (`_lda.py:676-691`).
- **#1547** — REQ-10: add `batch_size` (default 128) + `total_samples` (default 1e6)
  fields and `gen_batches`-style online mini-batching with `doc_ratio = total_samples /
  batch_size` (`_lda.py:662`/`:535-538`), replacing the per-document mini-batch-of-1
  online path (`lda_topic.rs:532`/`:576-579`).
- **#1548** — REQ-11: expose `n_features_in_` and `bound_` (final-fit perplexity,
  `_lda.py:701-703`) fitted attrs on `FittedLatentDirichletAllocation` (`alpha()`/`beta()`
  already mirror `doc_topic_prior_`/`topic_word_prior_`).
- **#1549** — REQ-12: add `n_jobs` (E-step parallelism, `_lda.py:658-659`) and `verbose`
  (per-iteration progress, `_lda.py:683-694`) fields.
- **#1550** — REQ-13: make `LatentDirichletAllocation` generic over `F: Float + Send +
  Sync + 'static` (f32 + f64, `_more_tags` preserves_dtype `_lda.py:548-552`), like the
  sibling `MiniBatchNMF<F>` / `SparsePCA<F>`.
- **#1551** — REQ-14: add a `_RsLatentDirichletAllocation` PyO3 binding in
  `ferrolearn-python` (ctor + `fit` + `transform`), registered in `lib.rs`, as the
  CPython consumer.
- **#1552** — REQ-15: migrate `lda_topic.rs` off `ndarray` + `rand`/`rand_distr`/
  `rand_xoshiro` + the hand-rolled `digamma` to `ferray-core` arrays / `ferray::random`
  / `ferray::stats` (R-SUBSTRATE).
