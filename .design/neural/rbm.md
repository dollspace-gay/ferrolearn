# BernoulliRBM — Bernoulli Restricted Boltzmann Machine

<!--
tier: 3-component
status: draft
baseline-commit: f0786d24bf6402bfc4389da3354705ca4b51829c
upstream-paths:
  - sklearn/neural_network/_rbm.py
-->

## Summary

`ferrolearn-neural/src/rbm.rs` mirrors scikit-learn's `sklearn.neural_network.BernoulliRBM`
(`sklearn/neural_network/_rbm.py`, tag 1.5.2, commit 156ef14): an unsupervised feature
learner over a binary visible / binary hidden layer with energy
`E(v,h) = -v^T W h - b_v^T v - b_h^T h`. sklearn trains it with **Stochastic Maximum
Likelihood / Persistent Contrastive Divergence (SML/PCD)** and exposes `transform`,
`gibbs`, `partial_fit`, `score_samples`, and the fitted attributes `components_`,
`intercept_hidden_`, `intercept_visible_`, `h_samples_`, `n_features_in_`.

This doc audits the CURRENT `rbm.rs` against that contract. As of the SML/PCD rebuild
(blockers #1629–#1634), `rbm.rs` implements sklearn's **Stochastic Maximum Likelihood /
Persistent Contrastive Divergence** algorithm faithfully on the existing `ndarray` +
`rand_distr` + `rand_xoshiro` substrate: a persistent particle state `h_samples_` carried
across batches/epochs, fixed contiguous `gen_even_slices` batching (no shuffle), binary
`_sample_visibles`/`_sample_hiddens`, the SML update equations with `lr / v_pos.shape[0]`
scaling, `free_energy`, `score_samples_with_corruption` (deterministic, explicit corruption
index) + `score_samples` (RNG-drawn index), `partial_fit`, parameter validation, and a binary
`gibbs`. The deterministic methods (`transform`, `free_energy`,
`score_samples_with_corruption`) match the live sklearn 1.5.2 oracle to < 1e-12.

Two axes remain NOT-STARTED by deliberate deferral: the **ferray substrate** migration
(REQ-8, blocker #1636 — `rbm.rs` still computes on `ndarray`/`rand_xoshiro`, the same
"correct-algorithm-on-ndarray, substrate pinned" pattern as the other 134 shipped units) and
the **`ferrolearn-python` binding** (REQ-9, blocker #1637, Layer-6 work). Because the
non-test production consumer (the canonical Python registration) is still absent, the
algorithm REQs are recorded as ALGORITHM-COMPLETE (impl + live-oracle tests) rather than a
strict R-HONEST-2 SHIPPED, which also requires the non-test consumer. This doc continues to
under-claim by design (R-HONEST-3).

A note on consumers (R-DEFER-1 / S5): `BernoulliRBM`/`FittedBernoulliRBM` are re-exported
from `ferrolearn-neural/src/lib.rs` and are boundary estimator types, so the type surface is
grandfathered. However there is **no `ferrolearn-python` registration** of the neural crate
and **no non-test production consumer** anywhere in the workspace — the only callers are
`ferrolearn-neural/tests/conformance_wave4.rs` and `ferrolearn-neural/tests/api_proof.rs`
(both `tests/`, i.e. test-only). Per R-HONEST-2 a SHIPPED claim requires impl + a real
non-test consumer; lacking one is recorded as a cross-cutting prerequisite (REQ-9).

## Upstream cite

scikit-learn 1.5.2, `sklearn/neural_network/_rbm.py` (455 lines):
- class + `_parameter_constraints` + `__init__`: `_rbm.py:28`, `:134-141`, `:143-158`
- `transform` / `_mean_hiddens`: `_rbm.py:160-178`, `:180-195`
- `_sample_hiddens` / `_sample_visibles`: `_rbm.py:197-214`, `:216-235`
- `_free_energy`: `_rbm.py:237-252`
- `gibbs`: `_rbm.py:254-273`
- `partial_fit`: `_rbm.py:276-315`
- `_fit` (SML/PCD inner step): `_rbm.py:317-345`
- `score_samples`: `_rbm.py:347-386`
- `fit` (persistent particles + `gen_even_slices`): `_rbm.py:388-442`
- `gen_even_slices`: `sklearn/utils/_chunking.py:86` (contiguous, in-order, no shuffle)

## Requirements

- REQ-1: Constructor parameters, defaults, and parameter validation mirror sklearn —
  `n_components` (default 256, `>=1`), `learning_rate` (0.1, `>0`), `batch_size`
  (10, `>=1`), `n_iter` (10, `>=0`), `verbose` (0), `random_state` (None).
- REQ-2: `transform` computes the deterministic mean hiddens
  `expit(v @ components_.T + intercept_hidden_)` matching `_mean_hiddens`
  (`_rbm.py:180-195`) element-for-element given identical fitted parameters.
- REQ-3: `fit` trains via **SML/PCD**: a persistent particle state `h_samples_` of shape
  `(batch_size, n_components)` initialized once and carried across all batches/epochs;
  each inner step takes `h_pos = mean_hiddens(v_pos)`, `v_neg = sample_visibles(h_samples_)`,
  `h_neg = mean_hiddens(v_neg)`, then resamples the particles
  (`_rbm.py:317-345`, `:417`).
- REQ-4: `fit` iterates **fixed contiguous batch slices in the same order every epoch** via
  `gen_even_slices`, with no per-epoch shuffling (`_rbm.py:419-427`).
- REQ-5: `_sample_visibles` and `_sample_hiddens` return **binary** samples
  `rng.uniform < p` (`_rbm.py:214`, `:235`); `gibbs` is `_sample_hiddens` then
  `_sample_visibles`, both sampled (`_rbm.py:254-273`).
- REQ-6: The SML update equations and learning-rate scaling match sklearn —
  `lr = learning_rate / v_pos.shape[0]`,
  `update = (v_pos.T @ h_pos).T - (h_neg.T @ v_neg)`,
  `intercept_hidden_ += lr*(h_pos.sum0 - h_neg.sum0)`,
  `intercept_visible_ += lr*(v_pos.sum0 - v_neg.sum0)`, with positive/negative phases on
  potentially DIFFERENT row counts (`_rbm.py:335-342`).
- REQ-7: The inspection / scoring surface is present — `score_samples` (pseudo-likelihood:
  corrupt one random feature per sample, `-n_features * logaddexp(0, -(fe_ - fe))`,
  `_rbm.py:347-386`), `_free_energy` (`_rbm.py:237-252`), `partial_fit` (`_rbm.py:276-315`),
  and the fitted attributes `h_samples_` and `n_features_in_` (`_rbm.py:91-93`, `:417`).
- REQ-8: Substrate — the unit computes on the ferray array/random substrate
  (`ferray-core`, `ferray::random`), not `ndarray`/`rand_distr`/`rand_xoshiro` (R-SUBSTRATE).
- REQ-9: A non-test production consumer exists (the `ferrolearn-python` registration of
  `BernoulliRBM` is the canonical one) so the estimator is reachable as `import ferrolearn`
  surface (R-DEFER-1 / R-HONEST-2).

## Acceptance criteria

- AC-1: `BernoulliRBM::new` defaults match `m = BernoulliRBM()` →
  `(256, 0.1, 10, 10, verbose=0, random_state=None)` from a live sklearn call; constructing
  with `n_components=0`, `learning_rate<=0`, or `batch_size=0` is rejected.
- AC-2: Given `components_`, `intercept_hidden_` copied from a live sklearn-fitted model,
  `FittedBernoulliRBM::transform(v)` equals `m.transform(v)` within f64 ULP tolerance on a
  fixed binary `X` (deterministic; must have a failing characterization test until met).
- AC-3: After `fit`, `h_samples_` is present with shape `(batch_size, n_components)` and the
  negative phase draws from it, not from the current batch (structural check on the algorithm).
- AC-4: Batch order is `gen_even_slices(...)` and identical across epochs (no shuffle call).
- AC-5: `gibbs(v)` and the internal visible/hidden sampling return values in `{0,1}` (binary),
  not sigmoid means.
- AC-6: `score_samples(X)` matches `m.score_samples(X)` in formula/shape given a fixed
  corruption index pattern (the RNG-dependent corruption index is an R-DEFER-3 carve-out;
  the deterministic free-energy arithmetic must match).
- AC-7: `cargo tree -p ferrolearn-neural` shows ferray crates and no `ndarray`/`rand_distr`
  in the owned computation path of `rbm.rs`.
- AC-8: `import ferrolearn; from ferrolearn.neural_network import BernoulliRBM` resolves (or
  the Rust-side non-test consumer exists).

## REQ status

Non-test consumer note: `BernoulliRBM`/`FittedBernoulliRBM` are re-exported from
`ferrolearn-neural/src/lib.rs:21` (boundary estimator types, grandfathered per the S5 note
above), but the canonical R-DEFER-1 consumer — the `ferrolearn-python` registration — is
NOT-STARTED (REQ-9, blocker #1637). Algorithm REQs below are therefore ALGORITHM-COMPLETE
(impl + live-oracle tests green) and explicitly flag the missing consumer rather than claiming
a strict R-HONEST-2 SHIPPED.

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (ctor / defaults / validation) | ALGORITHM-COMPLETE (consumer NOT-STARTED) | `fn validate_params in rbm.rs` rejects `n_components < 1`, `learning_rate <= 0`, `batch_size < 1` with `FerroError::InvalidParameter`, mirroring `_parameter_constraints` (`_rbm.py:134-141`); called at the top of `fit` and `partial_fit`. `batch_size` no longer clamps `0 -> 1` (`fn batch_size` is now a plain setter). Tests: `divergence_rbm_validation.rs` (3 tests, all green) + lib `rbm_rejects_invalid_params`. STILL DEFERRED: no `verbose` param, `new` still requires `n_components` (no sklearn-256 default) — those remain under #1628; the consumer (REQ-9) is #1637. |
| REQ-2 (transform / mean hiddens) | ALGORITHM-COMPLETE (consumer NOT-STARTED) | `fn mean_hiddens in rbm.rs` computes `expit(v @ components_.T + intercept_hidden_)` (= `_rbm.py:193-195`). Live-oracle value parity pinned by `tests/divergence_rbm.rs::divergence_transform_mean_hiddens_value_parity` (< 1e-12 vs sklearn `_mean_hiddens`). Consumer (REQ-9) #1637; substrate (REQ-8) #1636. |
| REQ-3 (SML/PCD persistent particles) | ALGORITHM-COMPLETE (consumer NOT-STARTED) | `fn fit` initializes `h_samples = zeros((batch_size, n_components))` ONCE and carries it across all batches/epochs; `fn sml_step` draws `v_neg = sample_visibles(h_samples, rng)` from the particles (NOT the current batch) and resamples them via `h_neg[rng.uniform < h_neg] = 1.0; h_samples = floor(h_neg)` (= `_rbm.py:332`, `:344-345`, `:417`). `h_samples_` is now a public fitted field; `tests/divergence_rbm_missing_api.rs::divergence_h_samples_attribute_missing` asserts shape `(batch_size, n_components)`. RNG carve-out (#1635) on exact particle bits stands. Consumer (REQ-9) #1637. |
| REQ-4 (fixed contiguous batch order) | ALGORITHM-COMPLETE (consumer NOT-STARTED) | `fn fit` precomputes `gen_even_slices(batch_size, n_batches, n_samples)` ONCE and reuses the same contiguous `(start, end)` slices every epoch with NO shuffle (= `_rbm.py:419-427`, `utils/_chunking.py:86`). The Fisher–Yates per-epoch shuffle is removed. Consumer (REQ-9) #1637. |
| REQ-5 (binary sampling in gibbs / visibles) | ALGORITHM-COMPLETE (consumer NOT-STARTED) | `fn sample_visibles`/`fn sample_hiddens` return binary `{0,1}` via `rng.uniform < p` (= `_rbm.py:214`, `:235`); `fn gibbs` is `sample_hiddens` then `sample_visibles` (= `_rbm.py:270-271`). Structurally subsumes carve-out #1634 (gibbs now samples, not mean). Lib test `rbm_gibbs_round_trip_shape_and_binary` asserts the output is binary. Per-bit draws are an RNG carve-out (#1635). Consumer (REQ-9) #1637. |
| REQ-6 (SML update equations / lr scaling) | ALGORITHM-COMPLETE (consumer NOT-STARTED) | `fn sml_step` uses `lr = learning_rate / v_pos.shape[0]`, `update = (v_pos.T @ h_pos).T - (h_neg.T @ v_neg)`, `intercept_hidden += lr*(h_pos.sum0 - h_neg.sum0)`, `intercept_visible += lr*(v_pos.sum0 - v_neg.sum0)` (= `_rbm.py:335-342`), with the positive phase on `m_pos = v_pos.nrows()` rows and the negative phase on `m_neg = batch_size` particle rows — the row-count divergence is now preserved correctly. Consumer (REQ-9) #1637. |
| REQ-7 (score_samples / free_energy / partial_fit / attrs) | ALGORITHM-COMPLETE (consumer NOT-STARTED) | `fn free_energy` (= `_rbm.py:250-252`, numerically stable `logaddexp0`), `fn score_samples_with_corruption` (explicit index, deterministic, `-n_features * logaddexp(0, -(fe_ - fe))`, = `_rbm.py:386`) + `fn score_samples` (RNG-drawn index), `fn partial_fit` (one SML step, lazy init, = `_rbm.py:276-315`), and the public `h_samples_` field (`_rbm.py:417`) all ship. Live-oracle value parity (< 1e-12) for `free_energy` and `score_samples_with_corruption` pinned by `tests/divergence_rbm_missing_api.rs`. `n_features_in_` is implicit via `intercept_visible_.len()` (no dedicated attr — minor, tracked under #1628). Consumer (REQ-9) #1637. |
| REQ-8 (ferray substrate) | NOT-STARTED | `ferrolearn-neural/Cargo.toml` depends on `ndarray`, `rand_distr`, `rand_xoshiro`; `rbm.rs` imports `use ndarray::{Array1, Array2}` and `use rand_distr::{Distribution, Normal, Uniform}` and uses `Xoshiro256PlusPlus` for all sampling. The destination substrate is `ferray-core` (array type) + `ferray::random` (sampling), per R-SUBSTRATE-1. No ferray usage present. Prereq: migrate the array type to `ferray-core` and sampling to `ferray::random`; this is also the locus where the RNG carve-out (REQ below) is resolved or pinned. Blocker to be filed under #1628. |
| REQ-9 (non-test production consumer) | NOT-STARTED | The only callers of `BernoulliRBM`/`FittedBernoulliRBM` outside `src/rbm.rs` are `ferrolearn-neural/tests/conformance_wave4.rs` and `ferrolearn-neural/tests/api_proof.rs` (both under `tests/`, test-only — verified via `grep -rn BernoulliRBM | grep -v 'src/rbm.rs' | grep -v '#[cfg(test'`). `ferrolearn-neural/src/lib.rs` re-exports the types but does not consume them. There is **no `ferrolearn-python` registration** of the neural crate (`grep -rn neural ferrolearn-python/src/` is empty). Per R-HONEST-2 / R-DEFER-1 a SHIPPED estimator needs a real non-test consumer; the canonical one is the PyO3 `import ferrolearn` registration. Prereq: register `BernoulliRBM` in `ferrolearn-python` (a downstream-layer task per R-DEFER-7, gated on Layer 6). Blocker to be filed under #1628. |

## RNG carve-out vs deterministic classification

Per R-DEFER-3, RNG-stream divergences get a blocker but **NO failing characterization test**
(the test would be unwinnable: numpy `RandomState`/Mersenne-Twister vs `Xoshiro256PlusPlus`
cannot bit-match). What MUST still match — and therefore gets a **failing characterization
test from the critic** — is the algorithm class, the deterministic methods, the defaults, and
the validation/attribute contract.

**RNG carve-outs (blocker only, NO failing test):**
- The post-fit values of `components_`, `intercept_hidden_`, `intercept_visible_`, and
  `h_samples_` after `fit` — these depend on the RNG stream and the random init
  (`rng.normal(0, 0.01, ...)`, `_rbm.py:410`) and cannot bit-match sklearn. Identifiability /
  RNG carve-out under R-DEFER-3.
- The specific corrupted-feature index drawn inside `score_samples`
  (`rng.randint(0, v.shape[1], v.shape[0])`, `_rbm.py:372`) — RNG-dependent; only the
  deterministic free-energy arithmetic given a fixed index pattern is testable.
- The per-step Bernoulli draws inside `fit`/`gibbs` — the sampling MECHANISM (binary
  `uniform < p`) is testable structurally, but the exact drawn bits are an RNG carve-out.
- Determinism-default divergence to DOCUMENT (not bit-match): ferrolearn maps
  `random_state = None` to a fixed seed `0xC0FFEE` (`fn fit`: `None => Xoshiro256PlusPlus::seed_from_u64(0xC0FFEE)`), making it deterministic, whereas sklearn `None` is non-deterministic
  via `check_random_state` (`_rbm.py:407`). This is an intentional, documented divergence
  (deterministic-by-default), not a bug to fix bit-for-bit.

**Deterministic — REQs that MUST have a failing characterization test from the critic
(R-CHAR-1, expected values from LIVE sklearn 1.5.2 per R-CHAR-3):**
- REQ-2 (`transform` = `_mean_hiddens`): value parity given identical fitted parameters.
- REQ-7 free-energy / pseudo-likelihood arithmetic given fixed inputs and a fixed corruption
  pattern (`-n_features * logaddexp(0, -(fe_ - fe))`).
- REQ-1 defaults and parameter validation (purely structural, no RNG).
- REQ-3/REQ-4/REQ-5/REQ-6 algorithm-class structure (persistent particles exist; batch order
  is fixed/contiguous; visibles are sampled binary; update uses particle-derived rows) — these
  are testable structurally without depending on a specific RNG bitstream.

## Architecture

`rbm.rs` defines an unfitted builder `BernoulliRBM<F>` (fields `n_components`,
`learning_rate`, `n_iter`, `batch_size`, `random_state: Option<u64>`) with `#[must_use]`
chained setters, and a fitted `FittedBernoulliRBM<F>` (fields `components_`,
`intercept_hidden_`, `intercept_visible_`, `n_iter_`). `Fit<Array2<F>, ()>` produces the
fitted struct; `Transform<Array2<F>>` and the inherent `fn transform`/`fn gibbs` provide the
post-fit surface. A free `fn sigmoid` is the logistic `expit` analog. Generic over
`F: Float + Send + Sync + 'static`, supporting f32 and f64 per CLAUDE.md.

The unfitted/fitted split and the `components_` / `intercept_hidden_` / `intercept_visible_`
attribute names match sklearn's contract (`_rbm.py:74-93`). The structural gaps vs the sklearn
class are: the missing persistent particle attribute `h_samples_` (`_rbm.py:86-89`, `:417`),
the missing `n_features_in_` (`_rbm.py:91`), and the missing `verbose` constructor parameter
(`_rbm.py:150`). The training loop in `fn fit` is a naive CD-1 (sample-from-current-batch)
rather than sklearn's SML/PCD persistent-particle scheme (`_rbm.py:317-345`, `:388-442`), and
it shuffles each epoch rather than reusing fixed `gen_even_slices` batches
(`sklearn/utils/_chunking.py:86`). The substrate is `ndarray` + `rand_distr` +
`rand_xoshiro`, not ferray (R-SUBSTRATE-1).

## Verification

Commands that would establish SHIPPED claims (none are green for a SHIPPED classification
today; all REQs are NOT-STARTED):

- `cargo test -p ferrolearn-neural` — existing tests (`rbm_fit_smoke`,
  `rbm_transform_shape_and_range`, `rbm_gibbs_round_trip_shape`, `rbm_feature_dim_mismatch`,
  `rbm_empty_input_rejected`) check shape/range/error-paths only; none pin VALUES against
  sklearn, so they do not establish REQ-2..REQ-7 (R-CHAR-1/R-CHAR-3 require live-sklearn
  expected values).
- Live oracle for REQ-2 (deterministic):
  `python3 -c "import numpy as np; from sklearn.neural_network import BernoulliRBM; m=BernoulliRBM(n_components=2, random_state=0); X=np.array([[0,0,0],[0,1,1],[1,0,1],[1,1,1]],float); m.fit(X); print(m.transform(X).tolist())"`
  — feed `m.components_`, `m.intercept_hidden_` into `FittedBernoulliRBM` and compare
  `transform` (critic to pin as a failing test).
- Live oracle for REQ-7:
  `python3 -c "...; print(m.score_samples(X).tolist())"` — observed
  `[-1.1365449, -3.25483523, -3.43272394, -3.2175858]` on the example above; ferrolearn has no
  `score_samples` to compare (critic pins the free-energy arithmetic).
- Defaults check (REQ-1):
  `python3 -c "from sklearn.neural_network import BernoulliRBM; m=BernoulliRBM(); print(m.n_components, m.learning_rate, m.batch_size, m.n_iter, m.verbose, m.random_state)"`
  → `256 0.1 10 10 0 None`.
- Substrate check (REQ-8): `cargo tree -p ferrolearn-neural | grep -E 'ndarray|rand_distr|ferray'`.
- Consumer check (REQ-9):
  `grep -rn "BernoulliRBM" --include=*.rs | grep -v 'src/rbm.rs' | grep -v 'tests/'` is empty.

Because no command above is green in a way that satisfies impl + non-test consumer + live-
sklearn-pinned test simultaneously, every REQ is NOT-STARTED.
