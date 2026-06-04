//! Bernoulli Restricted Boltzmann Machine.
//!
//! [`BernoulliRBM`] models the joint distribution of a binary visible layer
//! and a binary hidden layer with energy
//!
//! ```text
//! E(v, h) = -v^T W h - b_v^T v - b_h^T h
//! ```
//!
//! It is trained via **Stochastic Maximum Likelihood / Persistent Contrastive
//! Divergence (SML/PCD)**, faithfully mirroring scikit-learn 1.5.2's
//! `sklearn.neural_network.BernoulliRBM` (`sklearn/neural_network/_rbm.py`,
//! commit 156ef14). A persistent particle state [`h_samples_`] of shape
//! `(batch_size, n_components)` is initialized to zeros and carried across all
//! batches and all epochs (`_rbm.py:417`, `:332`, `:344-345`).
//!
//! Inputs are interpreted as Bernoulli probabilities; the [`transform`] method
//! returns the probability that each hidden unit is active given the visible
//! vector (sklearn `_mean_hiddens`, `_rbm.py:180-195`).
//!
//! # Determinism vs RNG carve-out
//!
//! Deterministic methods that MUST match sklearn element-for-element given
//! identical fitted parameters: [`transform`], [`free_energy`],
//! [`score_samples_with_corruption`]. The post-`fit` values of `components_`,
//! the intercepts, and [`h_samples_`] depend on the RNG stream (random init +
//! per-step Bernoulli draws) and CANNOT bit-match sklearn's numpy
//! Mersenne-Twister stream; this is an accepted RNG carve-out (R-DEFER-3).
//! ferrolearn keeps `Xoshiro256PlusPlus` and maps `random_state = None` to a
//! fixed seed `0xC0FFEE` (deterministic-by-default), a documented divergence
//! from sklearn's non-deterministic `None`.
//!
//! ## REQ status
//!
//! | REQ | Behavior | Status | Evidence |
//! |-----|----------|--------|----------|
//! | REQ-1 | Constructor defaults + parameter validation | SHIPPED | `fn validate_params` rejects `n_components < 1` / `learning_rate <= 0` / `batch_size < 1` (sklearn `_parameter_constraints`, `_rbm.py:134-141`); tests in `tests/divergence_rbm_validation.rs` |
//! | REQ-2 | `transform` / `_mean_hiddens` value parity | SHIPPED | `fn transform in FittedBernoulliRBM`; matches live sklearn to `<1e-12` (`tests/divergence_rbm.rs`, `_rbm.py:180-195`) |
//! | REQ-3 | SML/PCD persistent `h_samples_` particles | SHIPPED | `fn fit` + `fn sml_step`; particles carried across batches and epochs (`_rbm.py:317-345`, `:417`) |
//! | REQ-4 | Fixed `gen_even_slices` batching (no per-epoch shuffle) | SHIPPED | `fn gen_even_slices` matches sklearn `_chunking.py:124-135` exactly on all tested shapes |
//! | REQ-5 | Binary Bernoulli sampling of visibles/hiddens | SHIPPED | `fn sample_visibles` + `fn gibbs` use `uniform < p` (`_rbm.py:216-235`, `:254-273`) |
//! | REQ-6 | SML update + `lr / batch_rows` scaling | SHIPPED | `fn sml_step`: `(v_pos.T @ h_pos).T - h_neg.T @ v_neg`, intercepts per `_rbm.py:335-342` |
//! | REQ-7 | `free_energy` / `score_samples` / incremental `partial_fit` / `h_samples_` | SHIPPED | oracle parity `<1e-10` (`tests/divergence_rbm_missing_api.rs`); incremental `partial_fit` accumulates (`tests/divergence_rbm_reaudit.rs`, `_rbm.py:237-386`) |
//! | REQ-8 | ferray substrate (R-SUBSTRATE) | NOT-STARTED | built on `ndarray` + `rand_distr` + `rand_xoshiro`, not `ferray` — blocker #1636 |
//! | REQ-9 | non-test production consumer (`ferrolearn-python` registration) | NOT-STARTED | only test-only callers; no neural binding in `ferrolearn-python` — blocker #1637 |
//!
//! RNG carve-out (R-DEFER-3, no failing test): post-`fit` `components_`/intercepts/`h_samples_`
//! exact values (numpy Mersenne-Twister vs `Xoshiro256PlusPlus`) — blocker #1635.
//!
//! [`transform`]: FittedBernoulliRBM::transform
//! [`free_energy`]: FittedBernoulliRBM::free_energy
//! [`score_samples_with_corruption`]: FittedBernoulliRBM::score_samples_with_corruption
//! [`h_samples_`]: FittedBernoulliRBM::h_samples_

use ferrolearn_core::FerroError;
use ferrolearn_core::traits::{Fit, Transform};
use ndarray::{Array1, Array2};
use num_traits::Float;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal, Uniform};
use rand_xoshiro::Xoshiro256PlusPlus;

/// Bernoulli Restricted Boltzmann Machine (unfitted configuration).
///
/// Trained via Stochastic Maximum Likelihood / Persistent Contrastive
/// Divergence, mirroring scikit-learn's `BernoulliRBM` (`_rbm.py`). Use the
/// chained `#[must_use]` setters to configure, then [`Fit::fit`] (full SML
/// training) or [`BernoulliRBM::partial_fit`] (one incremental SML step).
#[derive(Debug, Clone)]
pub struct BernoulliRBM<F> {
    n_components: usize,
    learning_rate: F,
    n_iter: usize,
    batch_size: usize,
    random_state: Option<u64>,
    /// Persistent incremental state for [`partial_fit`](BernoulliRBM::partial_fit).
    ///
    /// `None` until the first `partial_fit` call; thereafter it carries the
    /// accumulated `components_`/intercepts/`h_samples_` AND the live RNG across
    /// calls, mirroring sklearn's `first_pass = not hasattr(self, "components_")`
    /// (`_rbm.py:292`): only the first call initializes, every later call runs one
    /// `_fit` step onto the EXISTING state (`_rbm.py:315`). `fit` does not touch
    /// this field.
    partial_state: Option<(FittedBernoulliRBM<F>, Xoshiro256PlusPlus)>,
}

impl<F: Float + Send + Sync + 'static> BernoulliRBM<F> {
    /// Construct a new [`BernoulliRBM`] with the given hidden-layer size.
    ///
    /// Defaults mirror sklearn (`_rbm.py:143-158`): `learning_rate = 0.1`,
    /// `n_iter = 10`, `batch_size = 10`, `random_state = None`. Note that
    /// invalid values (`n_components < 1`, `learning_rate <= 0`,
    /// `batch_size < 1`) are NOT rejected here; they are validated at
    /// [`fit`](Fit::fit) / [`partial_fit`](BernoulliRBM::partial_fit) time,
    /// mirroring sklearn's `_parameter_constraints` check inside `fit`
    /// (`_rbm.py:134-141`).
    #[must_use]
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            learning_rate: F::from(0.1).unwrap_or_else(F::one),
            n_iter: 10,
            batch_size: 10,
            random_state: None,
            partial_state: None,
        }
    }

    /// Set the SML learning rate (default `0.1`). Must be `> 0`; a non-positive
    /// value is rejected at fit time (sklearn `_rbm.py:136`).
    #[must_use]
    pub fn learning_rate(mut self, lr: F) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Set the number of full passes over the data (default `10`). Must be
    /// `>= 0` (sklearn `_rbm.py:138`); `usize` is always valid.
    #[must_use]
    pub fn n_iter(mut self, n: usize) -> Self {
        self.n_iter = n;
        self
    }

    /// Set the mini-batch size (default `10`). Must be `>= 1`; a value of `0`
    /// is rejected at fit time (sklearn `_rbm.py:137`).
    ///
    /// Unlike the previous implementation this no longer silently clamps `0`
    /// to `1` — a caller who asks for `batch_size = 0` gets an error from
    /// [`fit`](Fit::fit), matching sklearn's `InvalidParameterError`.
    #[must_use]
    pub fn batch_size(mut self, n: usize) -> Self {
        self.batch_size = n;
        self
    }

    /// Set the RNG seed for reproducibility.
    #[must_use]
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Validate hyperparameters against sklearn's `_parameter_constraints`
    /// (`_rbm.py:134-141`): `n_components >= 1`, `learning_rate > 0`,
    /// `batch_size >= 1`. (`n_iter >= 0` is automatic for `usize`.) Returns
    /// [`FerroError::InvalidParameter`] mirroring sklearn's
    /// `InvalidParameterError`.
    fn validate_params(&self) -> Result<(), FerroError> {
        if self.n_components < 1 {
            return Err(FerroError::InvalidParameter {
                name: "n_components".into(),
                reason: "must be >= 1 (sklearn Interval(Integral, 1, None, closed=\"left\"))"
                    .into(),
            });
        }
        if self.learning_rate <= F::zero() {
            return Err(FerroError::InvalidParameter {
                name: "learning_rate".into(),
                reason: "must be > 0 (sklearn Interval(Real, 0, None, closed=\"neither\"))".into(),
            });
        }
        if self.batch_size < 1 {
            return Err(FerroError::InvalidParameter {
                name: "batch_size".into(),
                reason: "must be >= 1 (sklearn Interval(Integral, 1, None, closed=\"left\"))"
                    .into(),
            });
        }
        Ok(())
    }

    /// Fit the model to a single segment of data `x` INCREMENTALLY, mirroring
    /// sklearn's `partial_fit` (`_rbm.py:276-315`).
    ///
    /// On the FIRST call the persistent state is initialized exactly once
    /// (`components_` from `N(0, 0.01)`, intercepts and the persistent particle
    /// state `h_samples_` to zeros, plus the RNG), mirroring
    /// `first_pass = not hasattr(self, "components_")` (`_rbm.py:292-313`). On
    /// EVERY call (including the first) one inner SML step ([`_fit`],
    /// `_rbm.py:315`) is run over the whole of `x` treated as one minibatch,
    /// ACCUMULATING onto the existing state and advancing the SAME persistent
    /// RNG. A subsequent `partial_fit` therefore continues from the prior call's
    /// non-zero parameters/particles and yields a different `components_`.
    ///
    /// Conforms to ferrolearn's immutable Fit -> Fitted world: the accumulated
    /// state lives in `self` (cf. the `Option`-threaded state of
    /// `ferrolearn-decomp::IncrementalPCA::partial_fit`); each call returns a
    /// clone of the current [`FittedBernoulliRBM`] snapshot. The `y` argument is
    /// the unit `()` placeholder (RBM is unsupervised).
    ///
    /// [`_fit`]: BernoulliRBM
    pub fn partial_fit(
        &mut self,
        x: &Array2<F>,
        _y: &(),
    ) -> Result<FittedBernoulliRBM<F>, FerroError> {
        self.validate_params()?;
        let n_samples = x.nrows();
        let n_features = x.ncols();
        if n_samples == 0 || n_features == 0 {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: n_samples.min(n_features),
                context: "BernoulliRBM::partial_fit".into(),
            });
        }

        // First pass (`_rbm.py:292`): initialize state ONLY if absent. Later
        // passes reuse the accumulated parameters, particles, and RNG.
        if self.partial_state.is_none() {
            let mut rng = self.make_rng();
            let components = init_components::<F>(self.n_components, n_features, &mut rng)?;
            let fitted = FittedBernoulliRBM {
                components_: components,
                intercept_hidden_: Array1::<F>::zeros(self.n_components),
                intercept_visible_: Array1::<F>::zeros(n_features),
                // Persistent particles, zero-initialized (sklearn `_rbm.py:313`).
                h_samples_: Array2::<F>::zeros((self.batch_size, self.n_components)),
                n_iter_: 0,
            };
            self.partial_state = Some((fitted, rng));
        }

        let (fitted, rng) =
            self.partial_state
                .as_mut()
                .ok_or_else(|| FerroError::InvalidParameter {
                    name: "partial_state".into(),
                    reason: "partial_fit state was unexpectedly absent after init".into(),
                })?;

        // One inner SML step onto the EXISTING state (sklearn `_rbm.py:315`).
        sml_step(
            x,
            self.learning_rate,
            &mut fitted.components_,
            &mut fitted.intercept_hidden_,
            &mut fitted.intercept_visible_,
            &mut fitted.h_samples_,
            rng,
        )?;
        fitted.n_iter_ += 1;

        Ok(fitted.clone())
    }

    /// Build the RNG, mapping `random_state = None` to the fixed seed
    /// `0xC0FFEE` (deterministic-by-default; documented divergence from
    /// sklearn's non-deterministic `None`, `_rbm.py:407`).
    fn make_rng(&self) -> Xoshiro256PlusPlus {
        match self.random_state {
            Some(seed) => Xoshiro256PlusPlus::seed_from_u64(seed),
            None => Xoshiro256PlusPlus::seed_from_u64(0xC0FFEE),
        }
    }
}

/// A fitted [`BernoulliRBM`].
///
/// The fitted attributes mirror sklearn (`_rbm.py:74-93`): `components_`
/// (weights), `intercept_hidden_`, `intercept_visible_`, the persistent
/// particle state `h_samples_`, and the number of training iterations.
#[derive(Debug, Clone)]
pub struct FittedBernoulliRBM<F> {
    /// Weight matrix of shape `(n_components, n_features)`.
    pub components_: Array2<F>,
    /// Hidden-layer bias of length `n_components`.
    pub intercept_hidden_: Array1<F>,
    /// Visible-layer bias of length `n_features`.
    pub intercept_visible_: Array1<F>,
    /// Persistent SML/PCD particle state of shape `(batch_size, n_components)`,
    /// initialized to zeros in `fit`/`partial_fit` and carried across all
    /// batches and epochs (sklearn `_rbm.py:417`, `:332`, `:344-345`).
    pub h_samples_: Array2<F>,
    /// Number of training iterations actually run.
    pub n_iter_: usize,
}

impl<F: Float + Send + Sync + 'static> FittedBernoulliRBM<F> {
    /// Compute the probability of each hidden unit being active given `v`.
    ///
    /// This is sklearn's `_mean_hiddens` (`_rbm.py:180-195`):
    /// `expit(v @ components_.T + intercept_hidden_)`. `v` should have shape
    /// `(n_samples, n_features)` and contain values in `[0, 1]`.
    pub fn transform(&self, v: &Array2<F>) -> Result<Array2<F>, FerroError> {
        self.check_features(v.ncols(), "BernoulliRBM::transform")?;
        Ok(self.mean_hiddens(v))
    }

    /// sklearn `_mean_hiddens` (`_rbm.py:180-195`): `expit(v @ components_.T +
    /// intercept_hidden_)`. Assumes `v.ncols() == n_features`.
    fn mean_hiddens(&self, v: &Array2<F>) -> Array2<F> {
        let n = v.nrows();
        let h = self.intercept_hidden_.len();
        let n_v = self.intercept_visible_.len();
        let mut out = Array2::<F>::zeros((n, h));
        for i in 0..n {
            for j in 0..h {
                let mut acc = self.intercept_hidden_[j];
                for k in 0..n_v {
                    acc = acc + v[[i, k]] * self.components_[[j, k]];
                }
                out[[i, j]] = sigmoid(acc);
            }
        }
        out
    }

    /// sklearn `_sample_visibles` (`_rbm.py:216-235`): `p = expit(h @
    /// components_ + intercept_visible_)`, return `rng.uniform(p.shape) < p` as
    /// binary `{0, 1}` values.
    fn sample_visibles(
        &self,
        h: &Array2<F>,
        unif: &Uniform<f64>,
        rng: &mut Xoshiro256PlusPlus,
    ) -> Array2<F> {
        let n = h.nrows();
        let n_v = self.intercept_visible_.len();
        let n_h = self.intercept_hidden_.len();
        let mut out = Array2::<F>::zeros((n, n_v));
        for i in 0..n {
            for k in 0..n_v {
                let mut acc = self.intercept_visible_[k];
                for j in 0..n_h {
                    acc = acc + h[[i, j]] * self.components_[[j, k]];
                }
                let p = sigmoid(acc).to_f64().unwrap_or(0.0);
                let r = unif.sample(rng);
                out[[i, k]] = if r < p { F::one() } else { F::zero() };
            }
        }
        out
    }

    /// sklearn `_sample_hiddens` (`_rbm.py:197-214`): sample binary `{0, 1}`
    /// hiddens from `P(h=1|v) = mean_hiddens(v)`.
    fn sample_hiddens(
        &self,
        v: &Array2<F>,
        unif: &Uniform<f64>,
        rng: &mut Xoshiro256PlusPlus,
    ) -> Array2<F> {
        let p = self.mean_hiddens(v);
        let n = p.nrows();
        let h = p.ncols();
        let mut out = Array2::<F>::zeros((n, h));
        for i in 0..n {
            for j in 0..h {
                let prob = p[[i, j]].to_f64().unwrap_or(0.0);
                let r = unif.sample(rng);
                out[[i, j]] = if r < prob { F::one() } else { F::zero() };
            }
        }
        out
    }

    /// Compute the free energy `F(v)` of the visible configurations `v`.
    ///
    /// sklearn `_free_energy` (`_rbm.py:237-252`):
    /// `F(v) = -(v @ intercept_visible_)
    ///         - sum_j logaddexp(0, (v @ components_.T + intercept_hidden_)[:, j])`.
    ///
    /// Deterministic given the fitted parameters; matches sklearn
    /// element-for-element (uses a numerically stable `logaddexp`).
    pub fn free_energy(&self, v: &Array2<F>) -> Result<Array1<F>, FerroError> {
        self.check_features(v.ncols(), "BernoulliRBM::free_energy")?;
        Ok(self.free_energy_unchecked(v))
    }

    /// `_free_energy` body (`_rbm.py:250-252`); assumes shape already checked.
    fn free_energy_unchecked(&self, v: &Array2<F>) -> Array1<F> {
        let n = v.nrows();
        let n_h = self.intercept_hidden_.len();
        let n_v = self.intercept_visible_.len();
        let mut out = Array1::<F>::zeros(n);
        for i in 0..n {
            // -(v @ intercept_visible_)
            let mut vbias = F::zero();
            for k in 0..n_v {
                vbias = vbias + v[[i, k]] * self.intercept_visible_[k];
            }
            // sum_j logaddexp(0, (v @ components_.T + intercept_hidden_)[i, j])
            let mut hidden_term = F::zero();
            for j in 0..n_h {
                let mut acc = self.intercept_hidden_[j];
                for k in 0..n_v {
                    acc = acc + v[[i, k]] * self.components_[[j, k]];
                }
                hidden_term = hidden_term + logaddexp0(acc);
            }
            out[i] = -vbias - hidden_term;
        }
        out
    }

    /// Compute the pseudo-likelihood of `X` with an EXPLICIT per-sample
    /// corruption index, making the result deterministic and oracle-comparable.
    ///
    /// This is the deterministic core of sklearn's `score_samples`
    /// (`_rbm.py:347-386`): for each row `i`, flip feature
    /// `corruption_indices[i]` (`v_[ind] = 1 - v_[ind]`), then return
    /// `-n_features * logaddexp(0, -(fe_ - fe))` where `fe = free_energy(v)` and
    /// `fe_ = free_energy(v_corrupted)`. sklearn draws the corruption index via
    /// `rng.randint` (`_rbm.py:372`); passing it explicitly here removes that
    /// RNG dependence (the random index itself is an R-DEFER-3 carve-out, but
    /// the arithmetic GIVEN the index is exact).
    pub fn score_samples_with_corruption(
        &self,
        x: &Array2<F>,
        corruption_indices: &[usize],
    ) -> Result<Array1<F>, FerroError> {
        let n_features = self.intercept_visible_.len();
        self.check_features(x.ncols(), "BernoulliRBM::score_samples_with_corruption")?;
        if corruption_indices.len() != x.nrows() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![x.nrows()],
                actual: vec![corruption_indices.len()],
                context:
                    "BernoulliRBM::score_samples_with_corruption: one corruption index per row"
                        .into(),
            });
        }
        for (row, &idx) in corruption_indices.iter().enumerate() {
            if idx >= n_features {
                return Err(FerroError::InvalidParameter {
                    name: "corruption_indices".into(),
                    reason: format!(
                        "index {idx} for row {row} is out of range for {n_features} features"
                    ),
                });
            }
        }

        // v_ = v with one feature per row flipped (v_[ind] = 1 - v_[ind]).
        let mut v_corrupt = x.clone();
        for (row, &idx) in corruption_indices.iter().enumerate() {
            v_corrupt[[row, idx]] = F::one() - v_corrupt[[row, idx]];
        }

        let fe = self.free_energy_unchecked(x);
        let fe_ = self.free_energy_unchecked(&v_corrupt);
        // -n_features * logaddexp(0, -(fe_ - fe))  (`_rbm.py:386`).
        let nf = F::from(n_features).unwrap_or_else(F::one);
        let mut out = Array1::<F>::zeros(x.nrows());
        for i in 0..x.nrows() {
            out[i] = -nf * logaddexp0(-(fe_[i] - fe[i]));
        }
        Ok(out)
    }

    /// Compute the pseudo-likelihood of `X`, drawing one random corruption
    /// index per row from `random_state` (sklearn `score_samples`,
    /// `_rbm.py:347-386`).
    ///
    /// The drawn corruption index is an R-DEFER-3 RNG carve-out and will NOT
    /// match sklearn's numpy stream; for a deterministic, oracle-comparable
    /// result use [`score_samples_with_corruption`].
    ///
    /// [`score_samples_with_corruption`]: FittedBernoulliRBM::score_samples_with_corruption
    pub fn score_samples(
        &self,
        x: &Array2<F>,
        random_state: Option<u64>,
    ) -> Result<Array1<F>, FerroError> {
        let n_features = self.intercept_visible_.len();
        self.check_features(x.ncols(), "BernoulliRBM::score_samples")?;
        if n_features == 0 {
            return Err(FerroError::InvalidParameter {
                name: "n_features".into(),
                reason: "score_samples requires at least one feature".into(),
            });
        }
        let mut rng = match random_state {
            Some(seed) => Xoshiro256PlusPlus::seed_from_u64(seed),
            None => Xoshiro256PlusPlus::seed_from_u64(0xC0FFEE),
        };
        let unif = Uniform::new(0.0_f64, 1.0_f64).map_err(|e| FerroError::InvalidParameter {
            name: "uniform".into(),
            reason: e.to_string(),
        })?;
        let indices: Vec<usize> = (0..x.nrows())
            .map(|_| {
                let r = unif.sample(&mut rng);
                ((r * n_features as f64).floor() as usize).min(n_features - 1)
            })
            .collect();
        self.score_samples_with_corruption(x, &indices)
    }

    /// Perform one Gibbs sampling step from `v`: sample binary hiddens
    /// (`_sample_hiddens`) then binary visibles (`_sample_visibles`), returning
    /// the BINARY reconstructed visible vector (sklearn `gibbs`,
    /// `_rbm.py:254-273`).
    ///
    /// Unlike the previous mean-field reconstruction, this returns `{0, 1}`
    /// samples, matching sklearn. The draws use the deterministic-default seed
    /// `0xC0FFEE` (RNG carve-out, R-DEFER-3); only the binary sampling mechanism
    /// is contractual. For a caller-controlled stream use
    /// [`gibbs_seeded`](FittedBernoulliRBM::gibbs_seeded).
    pub fn gibbs(&self, v: &Array2<F>) -> Result<Array2<F>, FerroError> {
        self.gibbs_seeded(v, None)
    }

    /// Like [`gibbs`](FittedBernoulliRBM::gibbs) but with an explicit RNG seed
    /// (`None` maps to the deterministic default `0xC0FFEE`).
    pub fn gibbs_seeded(
        &self,
        v: &Array2<F>,
        random_state: Option<u64>,
    ) -> Result<Array2<F>, FerroError> {
        self.check_features(v.ncols(), "BernoulliRBM::gibbs")?;
        let mut rng = match random_state {
            Some(seed) => Xoshiro256PlusPlus::seed_from_u64(seed),
            None => Xoshiro256PlusPlus::seed_from_u64(0xC0FFEE),
        };
        let unif = Uniform::new(0.0_f64, 1.0_f64).map_err(|e| FerroError::InvalidParameter {
            name: "uniform".into(),
            reason: e.to_string(),
        })?;
        let h = self.sample_hiddens(v, &unif, &mut rng);
        Ok(self.sample_visibles(&h, &unif, &mut rng))
    }

    /// Guard that an input has the expected feature count.
    fn check_features(&self, actual: usize, context: &str) -> Result<(), FerroError> {
        if actual != self.intercept_visible_.len() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![self.intercept_visible_.len()],
                actual: vec![actual],
                context: format!("{context}: feature count mismatch"),
            });
        }
        Ok(())
    }
}

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, ()> for BernoulliRBM<F> {
    type Fitted = FittedBernoulliRBM<F>;
    type Error = FerroError;

    fn fit(&self, x: &Array2<F>, _y: &()) -> Result<FittedBernoulliRBM<F>, FerroError> {
        self.validate_params()?;
        let n_samples = x.nrows();
        let n_features = x.ncols();
        if n_samples == 0 || n_features == 0 {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: n_samples.min(n_features),
                context: "BernoulliRBM::fit".into(),
            });
        }

        let mut rng = self.make_rng();
        let mut components = init_components::<F>(self.n_components, n_features, &mut rng)?;
        let mut intercept_hidden = Array1::<F>::zeros(self.n_components);
        let mut intercept_visible = Array1::<F>::zeros(n_features);
        // Persistent particle state, zero-initialized once and carried across
        // all batches and all epochs (sklearn `_rbm.py:417`).
        let mut h_samples = Array2::<F>::zeros((self.batch_size, self.n_components));

        // Fixed contiguous batch slices, computed once and reused every epoch
        // with NO shuffle, mirroring `gen_even_slices(n_batches * batch_size,
        // n_batches, n_samples)` (sklearn `_rbm.py:419-422`,
        // `utils/_chunking.py:86`). With `n = n_batches * batch_size` and
        // `n_packs = n_batches`, every pack is `batch_size` wide, clipped to
        // `n_samples`.
        let n_batches = n_samples.div_ceil(self.batch_size);
        let slices = gen_even_slices(self.batch_size, n_batches, n_samples);

        for _epoch in 0..self.n_iter {
            for &(start, end) in &slices {
                let v_pos = x.slice(ndarray::s![start..end, ..]).to_owned();
                sml_step(
                    &v_pos,
                    self.learning_rate,
                    &mut components,
                    &mut intercept_hidden,
                    &mut intercept_visible,
                    &mut h_samples,
                    &mut rng,
                )?;
            }
        }

        Ok(FittedBernoulliRBM {
            components_: components,
            intercept_hidden_: intercept_hidden,
            intercept_visible_: intercept_visible,
            h_samples_: h_samples,
            n_iter_: self.n_iter,
        })
    }
}

impl<F: Float + Send + Sync + 'static> Transform<Array2<F>> for FittedBernoulliRBM<F> {
    type Output = Array2<F>;
    type Error = FerroError;

    fn transform(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        FittedBernoulliRBM::transform(self, x)
    }
}

/// Initialize `components_` from `N(0, 0.01)` (sklearn `_rbm.py:409-410`).
fn init_components<F: Float + Send + Sync + 'static>(
    n_components: usize,
    n_features: usize,
    rng: &mut Xoshiro256PlusPlus,
) -> Result<Array2<F>, FerroError> {
    let init_normal = Normal::new(0.0_f64, 0.01_f64).map_err(|e| FerroError::InvalidParameter {
        name: "init normal".into(),
        reason: e.to_string(),
    })?;
    let mut components = Array2::<F>::zeros((n_components, n_features));
    for j in 0..n_components {
        for k in 0..n_features {
            let v: f64 = init_normal.sample(rng);
            components[[j, k]] = F::from(v).ok_or_else(|| FerroError::InvalidParameter {
                name: "init weight".into(),
                reason: "could not convert sampled weight to F".into(),
            })?;
        }
    }
    Ok(components)
}

/// One inner SML / Persistent Contrastive Divergence step over the minibatch
/// `v_pos`, mutating the parameters and the persistent particle state in place.
///
/// Faithful port of sklearn's `_fit` (`_rbm.py:317-345`):
/// - `h_pos = mean_hiddens(v_pos)` (rows = `v_pos.shape[0]`, possibly < batch_size);
/// - `v_neg = sample_visibles(h_samples, rng)` — binary, drawn from the
///   PERSISTENT particles (rows = batch_size);
/// - `h_neg = mean_hiddens(v_neg)`;
/// - `lr = learning_rate / v_pos.shape[0]`;
/// - `components += lr * ((v_pos.T @ h_pos).T - (h_neg.T @ v_neg))`;
/// - `intercept_hidden += lr * (h_pos.sum0 - h_neg.sum0)`;
/// - `intercept_visible += lr * (v_pos.sum0 - v_neg.sum0)`;
/// - resample particles: `h_neg[rng.uniform < h_neg] = 1.0; h_samples = floor(h_neg)`.
#[allow(
    clippy::too_many_arguments,
    reason = "faithful port of sklearn _fit; \
    splitting the mutable parameter set into a struct would obscure the \
    one-to-one correspondence with _rbm.py:317-345"
)]
fn sml_step<F: Float + Send + Sync + 'static>(
    v_pos: &Array2<F>,
    learning_rate: F,
    components: &mut Array2<F>,
    intercept_hidden: &mut Array1<F>,
    intercept_visible: &mut Array1<F>,
    h_samples: &mut Array2<F>,
    rng: &mut Xoshiro256PlusPlus,
) -> Result<(), FerroError> {
    let unif = Uniform::new(0.0_f64, 1.0_f64).map_err(|e| FerroError::InvalidParameter {
        name: "uniform".into(),
        reason: e.to_string(),
    })?;

    let n_components = intercept_hidden.len();
    let n_features = intercept_visible.len();
    let m_pos = v_pos.nrows(); // positive-phase row count (current batch)
    let m_neg = h_samples.nrows(); // negative-phase row count (= batch_size)

    let m_pos_f = F::from(m_pos).ok_or_else(|| FerroError::InvalidParameter {
        name: "batch size".into(),
        reason: "could not convert v_pos.shape[0] to F".into(),
    })?;
    let lr = learning_rate / m_pos_f;

    // Borrow as immutable views for the read-only phase computations.
    let fitted_view = FittedView {
        components,
        intercept_hidden,
        intercept_visible,
    };

    // h_pos = mean_hiddens(v_pos)            shape (m_pos, n_components)
    let h_pos = fitted_view.mean_hiddens(v_pos);
    // v_neg = sample_visibles(h_samples)     shape (m_neg, n_features), binary
    let v_neg = fitted_view.sample_visibles(h_samples, &unif, rng);
    // h_neg = mean_hiddens(v_neg)            shape (m_neg, n_components)
    let mut h_neg = fitted_view.mean_hiddens(&v_neg);

    // update = (v_pos.T @ h_pos).T - (h_neg.T @ v_neg)   shape (n_components, n_features)
    // components += lr * update
    for j in 0..n_components {
        for k in 0..n_features {
            // (v_pos.T @ h_pos).T[j, k] = sum_i v_pos[i, k] * h_pos[i, j]
            let mut pos = F::zero();
            for i in 0..m_pos {
                pos = pos + v_pos[[i, k]] * h_pos[[i, j]];
            }
            // (h_neg.T @ v_neg)[j, k] = sum_i h_neg[i, j] * v_neg[i, k]
            let mut neg = F::zero();
            for i in 0..m_neg {
                neg = neg + h_neg[[i, j]] * v_neg[[i, k]];
            }
            components[[j, k]] = components[[j, k]] + lr * (pos - neg);
        }
    }

    // intercept_hidden += lr * (h_pos.sum(0) - h_neg.sum(0))
    for j in 0..n_components {
        let mut pos = F::zero();
        for i in 0..m_pos {
            pos = pos + h_pos[[i, j]];
        }
        let mut neg = F::zero();
        for i in 0..m_neg {
            neg = neg + h_neg[[i, j]];
        }
        intercept_hidden[j] = intercept_hidden[j] + lr * (pos - neg);
    }

    // intercept_visible += lr * (v_pos.sum(0) - v_neg.sum(0))
    for k in 0..n_features {
        let mut pos = F::zero();
        for i in 0..m_pos {
            pos = pos + v_pos[[i, k]];
        }
        let mut neg = F::zero();
        for i in 0..m_neg {
            neg = neg + v_neg[[i, k]];
        }
        intercept_visible[k] = intercept_visible[k] + lr * (pos - neg);
    }

    // Resample the persistent particles:
    //   h_neg[rng.uniform < h_neg] = 1.0; h_samples = floor(h_neg)
    // i.e. binarize h_neg by a Bernoulli draw, then floor (`_rbm.py:344-345`).
    for i in 0..m_neg {
        for j in 0..n_components {
            let p = h_neg[[i, j]].to_f64().unwrap_or(0.0);
            let r = unif.sample(rng);
            if r < p {
                h_neg[[i, j]] = F::one();
            }
            h_samples[[i, j]] = h_neg[[i, j]].floor();
        }
    }

    Ok(())
}

/// A lightweight read-only view over the parameter set, so the SML phase
/// computations can reuse the same `mean_hiddens` / `sample_visibles` logic as
/// [`FittedBernoulliRBM`] without cloning the parameters.
struct FittedView<'a, F> {
    components: &'a Array2<F>,
    intercept_hidden: &'a Array1<F>,
    intercept_visible: &'a Array1<F>,
}

impl<F: Float + Send + Sync + 'static> FittedView<'_, F> {
    fn mean_hiddens(&self, v: &Array2<F>) -> Array2<F> {
        let n = v.nrows();
        let h = self.intercept_hidden.len();
        let n_v = self.intercept_visible.len();
        let mut out = Array2::<F>::zeros((n, h));
        for i in 0..n {
            for j in 0..h {
                let mut acc = self.intercept_hidden[j];
                for k in 0..n_v {
                    acc = acc + v[[i, k]] * self.components[[j, k]];
                }
                out[[i, j]] = sigmoid(acc);
            }
        }
        out
    }

    fn sample_visibles(
        &self,
        h: &Array2<F>,
        unif: &Uniform<f64>,
        rng: &mut Xoshiro256PlusPlus,
    ) -> Array2<F> {
        let n = h.nrows();
        let n_v = self.intercept_visible.len();
        let n_h = self.intercept_hidden.len();
        let mut out = Array2::<F>::zeros((n, n_v));
        for i in 0..n {
            for k in 0..n_v {
                let mut acc = self.intercept_visible[k];
                for j in 0..n_h {
                    acc = acc + h[[i, j]] * self.components[[j, k]];
                }
                let p = sigmoid(acc).to_f64().unwrap_or(0.0);
                let r = unif.sample(rng);
                out[[i, k]] = if r < p { F::one() } else { F::zero() };
            }
        }
        out
    }
}

/// Compute the contiguous batch slices `gen_even_slices(n_batches * batch_size,
/// n_batches, n_samples)` (sklearn `utils/_chunking.py:86`) as `(start, end)`
/// pairs. With `n = n_batches * batch_size` and `n_packs = n_batches`, `n //
/// n_packs == batch_size` and `n % n_packs == 0`, so every pack is `batch_size`
/// wide, with the final `end` clipped to `n_samples`.
fn gen_even_slices(batch_size: usize, n_batches: usize, n_samples: usize) -> Vec<(usize, usize)> {
    let mut slices = Vec::with_capacity(n_batches);
    let mut start = 0;
    for _ in 0..n_batches {
        if batch_size > 0 {
            let end = (start + batch_size).min(n_samples);
            slices.push((start, end));
            start = end;
        }
    }
    slices
}

/// Numerically stable `logaddexp(0, x) = ln(1 + exp(x))`, computed as
/// `max(0, x) + ln(1 + exp(-|x|))` to avoid overflow (matches numpy
/// `np.logaddexp(0, x)`).
#[inline]
fn logaddexp0<F: Float>(x: F) -> F {
    let zero = F::zero();
    let max = if x > zero { x } else { zero };
    max + (F::one() + (-x.abs()).exp()).ln()
}

/// The logistic / `expit` function `1 / (1 + exp(-x))`.
#[inline]
fn sigmoid<F: Float>(x: F) -> F {
    F::one() / (F::one() + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn data() -> Array2<f64> {
        // Two pairs of correlated features in [0, 1]
        array![
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
        ]
    }

    #[test]
    fn rbm_fit_smoke() {
        let rbm = BernoulliRBM::<f64>::new(2)
            .learning_rate(0.1)
            .n_iter(5)
            .batch_size(3)
            .random_state(7);
        let fitted = rbm.fit(&data(), &()).unwrap();
        assert_eq!(fitted.components_.dim(), (2, 4));
        assert_eq!(fitted.intercept_hidden_.len(), 2);
        assert_eq!(fitted.intercept_visible_.len(), 4);
        assert_eq!(fitted.n_iter_, 5);
        // Persistent particles carried out of fit: (batch_size, n_components).
        assert_eq!(fitted.h_samples_.dim(), (3, 2));
    }

    #[test]
    fn rbm_transform_shape_and_range() {
        let rbm = BernoulliRBM::<f64>::new(3)
            .learning_rate(0.1)
            .n_iter(2)
            .random_state(7);
        let fitted = rbm.fit(&data(), &()).unwrap();
        let h = fitted.transform(&data()).unwrap();
        assert_eq!(h.dim(), (6, 3));
        for v in h.iter() {
            assert!((0.0..=1.0).contains(v));
        }
    }

    #[test]
    fn rbm_gibbs_round_trip_shape_and_binary() {
        let rbm = BernoulliRBM::<f64>::new(2).n_iter(2).random_state(7);
        let fitted = rbm.fit(&data(), &()).unwrap();
        let v_recon = fitted.gibbs_seeded(&data(), Some(7)).unwrap();
        assert_eq!(v_recon.dim(), (6, 4));
        // sklearn `gibbs` returns BINARY samples (`_rbm.py:235,214`).
        for v in v_recon.iter() {
            assert!(*v == 0.0 || *v == 1.0, "gibbs must return binary, got {v}");
        }
    }

    #[test]
    fn rbm_feature_dim_mismatch() {
        let rbm = BernoulliRBM::<f64>::new(2).n_iter(2).random_state(7);
        let fitted = rbm.fit(&data(), &()).unwrap();
        let bad: Array2<f64> = Array2::zeros((2, 9));
        assert!(fitted.transform(&bad).is_err());
    }

    #[test]
    fn rbm_empty_input_rejected() {
        let rbm = BernoulliRBM::<f64>::new(2).n_iter(2).random_state(7);
        let bad: Array2<f64> = Array2::zeros((0, 4));
        assert!(rbm.fit(&bad, &()).is_err());
    }

    #[test]
    fn rbm_rejects_invalid_params() {
        let x = data();
        assert!(BernoulliRBM::<f64>::new(0).fit(&x, &()).is_err());
        assert!(
            BernoulliRBM::<f64>::new(2)
                .learning_rate(0.0)
                .fit(&x, &())
                .is_err()
        );
        assert!(
            BernoulliRBM::<f64>::new(2)
                .batch_size(0)
                .fit(&x, &())
                .is_err()
        );
    }

    #[test]
    fn rbm_partial_fit_shapes() {
        let mut rbm = BernoulliRBM::<f64>::new(3).batch_size(2).random_state(0);
        let x: Array2<f64> = array![[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]];
        let fitted = rbm.partial_fit(&x, &()).unwrap();
        assert_eq!(fitted.components_.dim(), (3, 4));
        assert_eq!(fitted.h_samples_.dim(), (2, 3));
        assert_eq!(fitted.n_iter_, 1);
    }

    // Deterministic free-energy / pseudo-likelihood pins (live sklearn 1.5.2
    // oracle, params set directly so the RNG is irrelevant; R-CHAR-3).
    fn known_fitted() -> FittedBernoulliRBM<f64> {
        FittedBernoulliRBM {
            components_: array![[0.5, -0.3, 0.2, 0.1], [-0.2, 0.4, -0.1, 0.6]],
            intercept_hidden_: array![0.1, -0.2],
            intercept_visible_: array![0.05, -0.15, 0.25, -0.35],
            h_samples_: Array2::zeros((0, 2)),
            n_iter_: 0,
        }
    }

    fn fixed_v() -> Array2<f64> {
        array![
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
        ]
    }

    #[test]
    fn free_energy_matches_sklearn() {
        let got = known_fitted().free_energy(&fixed_v()).unwrap();
        let expected = [
            -1.9451776501278844,
            -1.3154973260213487,
            -1.8115649346659926,
        ];
        for (g, e) in got.iter().zip(expected.iter()) {
            assert!((g - e).abs() < 1e-12, "{g} vs {e}");
        }
    }

    #[test]
    fn score_samples_with_corruption_matches_sklearn() {
        let got = known_fitted()
            .score_samples_with_corruption(&fixed_v(), &[0, 1, 2])
            .unwrap();
        let expected = [-2.240546579745612, -2.8649444947448055, -2.198651158575029];
        for (g, e) in got.iter().zip(expected.iter()) {
            assert!((g - e).abs() < 1e-12, "{g} vs {e}");
        }
    }
}
