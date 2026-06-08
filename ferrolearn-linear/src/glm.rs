//! Generalized Linear Models (GLM).
//!
//! This module provides IRLS-based GLM regressors for count and positive
//! continuous data:
//!
//! - **[`GLMRegressor`]** — Generic GLM with selectable [`GLMFamily`]
//! - **[`PoissonRegressor`]** — Convenience wrapper with Poisson family
//! - **[`GammaRegressor`]** — Convenience wrapper with Gamma family
//! - **[`TweedieRegressor`]** — Convenience wrapper with Tweedie family
//!
//! All models use Iteratively Reweighted Least Squares (IRLS) and L2
//! regularization. The link function is fixed to **log** for Poisson and Gamma
//! (their sklearn losses are log-link only); [`TweedieRegressor`] selects its
//! [`Link`] via a `link` configuration (`auto`/`identity`/`log`), matching
//! `sklearn/linear_model/_glm/glm.py:889-903`.
//!
//! # Examples
//!
//! ```
//! use ferrolearn_linear::PoissonRegressor;
//! use ferrolearn_core::{Fit, Predict};
//! use ndarray::{array, Array1, Array2};
//!
//! let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
//! let y = array![2.0, 5.0, 10.0, 20.0];
//!
//! let model = PoissonRegressor::<f64>::new().with_alpha(0.0);
//! let fitted = model.fit(&x, &y).unwrap();
//! let preds = fitted.predict(&x).unwrap();
//! assert_eq!(preds.len(), 4);
//! ```
//!
//! ## REQ status (per `.design/linear/glm.md`, mirrors `sklearn/linear_model/_glm/glm.py` @ 1.5.2, commit 156ef14)
//!
//! Binary classification (R-DEFER-2): SHIPPED = impl + tests + green oracle
//! verification; NOT-STARTED = concrete open blocker referenced by `#`-number.
//! The public estimator types re-exported at the crate root are the consumer
//! surface (R-DEFER-1; no `ferrolearn-python` GLM binding yet).
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-4 (penalized objective: mean half-deviance + ½·alpha, intercept-free) | SHIPPED | `fn weighted_ridge_solve` adds the L2 penalty `weight_sum * alpha` to feature columns only, skipping the intercept column (`intercept_col`), matching sklearn's mean-deviance objective + unpenalized intercept (`glm.py:229-258`: `obj = average(½·deviance) + ½·alpha·‖coef‖²`, `l2_reg_strength = self.alpha`). Oracle parity tests `glm_poisson_intercept_unpenalized` (alpha=1e6 → `intercept_ = log(mean y)`, coef → 0) and `glm_poisson_penalty_scaling` (alpha=1.0 → `coef_=[0.34151720,0.18859745]`, `intercept_=-0.37680132`) green in `tests/divergence_glm_fit.rs`. |
//! | REQ-1 (Poisson family + log link) | SHIPPED | #548. `fn fit_glm_irls` with `GLMFamily::Poisson` (`variance => mu`, log-link Fisher scoring) under the REQ-4 mean-deviance/unpenalized-intercept objective fits `PoissonRegressor` to sklearn's `PoissonRegressor` (`HalfPoissonLoss`, log link, `glm.py:589-590`) at BOTH alpha=0 and alpha>0. Consumer: `PoissonRegressor::fit` (crate-root export). Oracle parity tests `glm_poisson_alpha_half_parity` (alpha=0.5 → live coef `[0.38388476754733647,0.2024000617918683]`, int `-0.519356533563308`), `glm_solver_param_invariant`, `glm_poisson_sample_weight`, `glm_poisson_penalty_scaling` green in `tests/divergence_glm_fit.rs` (the alpha=0 MLE matches to <1e-9, the module-header note). |
//! | REQ-2 (Gamma family + log link) | SHIPPED | #549. `GLMFamily::Gamma` (`variance => mu²`) drives `w = mu²/V(mu)` log-link IRLS; `GammaRegressor` matches sklearn's `GammaRegressor` (`HalfGammaLoss`, log link, y-domain `0 < y`, `glm.py:721-722`) at alpha=0 and alpha>0. The `y == 0` rejection (`HalfGammaLoss` open at 0) is enforced under REQ-14 (`YDomain::Positive`). Consumer: `GammaRegressor::fit` (crate-root export). Oracle tests `glm_gamma_alpha_half_parity` (alpha=0.5 → live coef `[0.24773782526507374,0.11636425618936652]`, int `0.3599464049766692`), `glm_gamma_sample_weight`, `glm_gamma_rejects_zero_y` green. |
//! | REQ-3 (Tweedie family + power) | SHIPPED | #550. `GLMFamily::Tweedie(p)` (`variance => mu^p`) with the link resolved from `power`/`link` (REQ-8); `TweedieRegressor(power=p)` matches sklearn's `TweedieRegressor` for the log-link powers `p>0` AND the identity-link `p<=0` (`HalfTweedieLoss`/`HalfTweedieLossIdentity`, `glm.py:889-903`) at alpha=0 and alpha>0. Verified live against the oracle for `p ∈ {0,1,1.5,2,3}` to <1e-8. Consumer: `TweedieRegressor::fit` (crate-root export). Oracle tests `glm_tweedie_alpha_half_parity` (`power=1.5, alpha=0.5` → live coef `[0.25606046404981164,0.11657692670900446]`, int `0.3563978246931595`), `glm_tweedie_power0_identity_link`, `glm_tweedie_power0_predict_identity_inverse`, `glm_tweedie_power2_rejects_zero_y` green. |
//! | REQ-5 (intercept init = link(weighted_mean(y))) | SHIPPED | #552. In `fn fit_glm_irls`, when `fit_intercept` AND NOT (warm_start with `coef_init`), the intercept entry is seeded at `coef[0] = link.link(weighted_mean(y))` (feature coefs stay 0) and `eta`/`mu` are recomputed from that seed, mirroring sklearn's `coef[-1] = link.link(np.average(y, weights=sample_weight))` (`glm.py:251-256`); `weighted_mean(y) = Σ(sᵢ·yᵢ)/Σ(sᵢ)`. warm_start with an explicit `coef_init` (REQ-11) takes precedence (the warm seed overrides the init), and a non-finite seed (e.g. `log(0)` for all-zero Poisson `y`) falls back to the previous cold start (intercept 0) with NO panic/NaN (R-CODE-2). The penalized GLM objective is convex, so the converged `coef_`/`intercept_` are init-invariant — all 22 pre-existing oracle tests stay byte-identical at convergence. Consumer: each estimator's `Fit::fit` (crate-root export). Oracle tests `glm_intercept_init_matches_sklearn_first_iterate` (constant `y=7` → `max_iter=1` intercept = `log(7) = 1.9459101490553132`, == live sklearn's first iterate; feature coef 0), `glm_intercept_init_converged_optimum_unchanged` (alpha=0.5 optimum unchanged vs oracle), `glm_intercept_init_all_zero_y_no_nan` (non-finite-seed fallback, finite result) green in `tests/divergence_glm_fit.rs`. |
//! | REQ-13 (score = D², deviance-explained) | SHIPPED | #559. `#[must_use] pub fn score(&self, x, y) -> Result<F, FerroError>` on `FittedGLMRegressor` computes `D² = 1 − (deviance + constant)/(deviance_null + constant)` (`glm.py:365-438`): `μ = predict(x)`, the null model predicts the (unweighted) mean `ȳ` for every sample, and the per-family unit deviance comes from `GLMFamily::unit_deviance` (Poisson `2·(y·ln(y/μ) − y + μ)`, y=0→`2μ`; Gamma `2·(−ln(y/μ) + (y−μ)/μ)`; Tweedie p=0 `(y−μ)²`; general-p `2·(y^(2−p)/((1−p)(2−p)) − y·μ^(1−p)/(1−p) + μ^(2−p)/(2−p))`), verified term-for-term against `sklearn/_loss/loss.py` (`HalfPoissonLoss:728-742`, `HalfGammaLoss:754-773`, `HalfTweedieLoss:789-837`). `GLMFamily::constant_to_optimal_zero` restores sklearn's `+ constant` so the degenerate constant-`y` boundary matches the oracle. `score` re-validates the y-domain (`YDomain::for_power`), mirroring `glm.py:413-417`. Consumer: the crate-root-exported `FittedGLMRegressor::score` (a public method on the boundary fitted type). Oracle tests `glm_poisson_d2_score` (D²=0.7979479374534378), `glm_gamma_d2_score` (0.8987486959882107), `glm_tweedie_power0_d2_score` (0.9319946452476573, == R²), `glm_tweedie_d2_score` (0.9277805586816806), `glm_score_rejects_out_of_domain_y` green in `tests/divergence_glm_fit.rs`; all 14 pre-existing glm divergence tests stay green. |
//! | REQ-7 (predict = link.inverse) | SHIPPED | `fn predict` applies `self.link.inverse(eta)` (`Link::Log => exp`, `Link::Identity => eta`), mirroring `glm.py:362` (`y_pred = link.inverse(raw_prediction)`). Consumer: the crate-root-exported `FittedGLMRegressor::predict` used by every wrapper; oracle test `glm_tweedie_power0_predict_identity_inverse` (identity link → raw linear predictor `[0.4,6.3,12.2,18.1]`) green in `tests/divergence_glm_fit.rs`. |
//! | REQ-8 (Tweedie link='auto'/identity/log) | SHIPPED | `pub enum Link { Log, Identity }` + `pub enum LinkConfig { Auto, Log, Identity }` with `LinkConfig::resolve(power)`: Auto → identity for `power <= 0`, log otherwise (`glm.py:889-893`). `TweedieRegressor.link: LinkConfig` (default `Auto`) is resolved at fit time and threaded into `fit_glm_irls`'s link-parameterized IRLS (`w = dmu_deta^2/V(mu)`, `z = eta + (y-mu)/dmu_deta`) and the fitted struct. Consumer: `TweedieRegressor::fit` (crate-root export); oracle test `glm_tweedie_power0_identity_link` (`coef_=[5.9]`, `intercept_=-5.5`, OLS) green. Poisson/Gamma wire `Link::Log` explicitly. |
//! | REQ-10 (solver param: lbfgs/newton-cholesky) | SHIPPED | #556. **R-DEV-2 (API parity):** `pub enum Solver { Lbfgs, NewtonCholesky }` + a `pub solver: Solver` field (default `Solver::Lbfgs`) on `GLMRegressor`/`PoissonRegressor`/`GammaRegressor`/`TweedieRegressor`, plus `#[must_use] fn with_solver`, mirroring sklearn's validated `solver` constructor parameter `StrOptions({"lbfgs","newton-cholesky"})` default `"lbfgs"` (`glm.py:140-145, :155`); the two-variant enum mirrors the `StrOptions` constraint. **R-DEV-7 (implementation differs, observable contract preserved):** ferrolearn fits all GLMs via IRLS/Fisher-scoring (`fn fit_glm_irls`) regardless of `solver` — the penalized GLM objective is convex, so IRLS reaches the SAME minimizer as both sklearn solvers (verified live: `PoissonRegressor(alpha=0.5)` gives coef `[0.38388523,0.20239975]`, int `-0.51935749` for `lbfgs` AND `newton-cholesky`, identical to ~1e-9). Consumer: each estimator's `Fit::fit` (crate-root export) — the `solver` field is part of the boundary estimator ABI. Oracle test `glm_solver_param_invariant` (fits with `Solver::Lbfgs` and `Solver::NewtonCholesky`, both coef/intercept match the solver-invariant live sklearn 1.5.2 oracle to 1e-4) green in `tests/divergence_glm_fit.rs`; the 19 pre-existing glm divergence tests stay green. |
//! | REQ-9 (Tweedie default power=0.0) | SHIPPED | `TweedieRegressor::new` sets `power: 0.0` (sklearn default, `glm.py:867`). Consumer: `TweedieRegressor::default`/`new` (crate-root export); oracle test `glm_tweedie_default_power` (`new().power == 0.0`) green. |
//! | REQ-11 (warm_start) | SHIPPED | #557. **R-DEV-2 (API parity):** `pub warm_start: bool` (default `false`) + `#[must_use] fn with_warm_start` on `GLMRegressor`/`PoissonRegressor`/`GammaRegressor`/`TweedieRegressor`, mirroring sklearn's `warm_start` parameter (`"boolean"`, default `False`, `glm.py:146, :158, :576, :708, :874`). **R-DEV-7 (Rust analog — immutable-estimator design, observable contract preserved):** sklearn's `warm_start=True` reuses the stateful `self.coef_`/`self.intercept_` mutated across `fit` calls as the optimizer's start (`glm.py:243-254`); ferrolearn's estimators are immutable (`fit(&self, ...)` never mutates `self`, no `self.coef_` to reuse), so the warm-start point is supplied EXPLICITLY via `pub coef_init: Option<(Array1<F>, F)>` + `#[must_use] fn with_coef_init(coef, intercept)`. `fn fit_glm_irls` seeds the IRLS coefficient vector (and derived `eta`/`mu`) from `coef_init` when `warm_start && coef_init.is_some()` (validating `feature_coef.len() == n_features`, else `ShapeMismatch`); otherwise the cold start (`coef = 0`) is byte-for-byte preserved. The penalized GLM objective is convex, so the converged `coef_`/`intercept_` are warm-start-INVARIANT — the init only changes the starting point (and iteration count), never the optimum — so the warm fit matches the cold fit AND the sklearn oracle (`glm.py:244-256`). Consumer: each estimator's `Fit::fit` (crate-root export) — the `warm_start`/`coef_init` fields are part of the boundary estimator ABI. Oracle tests `glm_warm_start_observable_contract` (warm fit from a perturbed init == cold fit == live sklearn 1.5.2 oracle `coef_=[0.38388477,0.20240006]`, `intercept_=-0.51935653` to 1e-6/1e-4) and `glm_warm_start_init_used` (seeding the exact optimum with `max_iter=1` lands at the solution, a cold `max_iter=1` fit does not — proves the init is genuinely used) green in `tests/divergence_glm_fit.rs`; the 20 pre-existing glm divergence tests stay green (all cold-start, byte-identical). |
//! | REQ-12 (sample_weight) | SHIPPED | `fn fit_with_sample_weight` on `GLMRegressor`/`PoissonRegressor`/`GammaRegressor`/`TweedieRegressor` threads an `Array1<F>` `sample_weight` into `fn fit_glm_irls`, where the IRLS `W` diagonal becomes `s_i * w_irls,i` (`weights[i] = weights[i] * sample_weight[i]`) and the L2-penalty scale is `weight_sum = S = sum_i s_i` (`sample_weight.iter().fold(..)`), matching sklearn's `sample_weight`-averaged deviance objective normalized by `sum(sample_weight)` (`glm.py:229-242`; `_check_sample_weight`, `glm.py:208-211`). Consumer: each estimator's `Fit::fit` (crate-root export) delegates with an all-ones weight vector, so the unweighted path is byte-identical (`weight_sum = n_samples`). Oracle tests `glm_poisson_sample_weight` (coef `[0.35738828,0.19717462]`, int `-0.43719203`) and `glm_gamma_sample_weight` (coef `[0.23049054,0.11350454]`, int `0.41955357`) green in `tests/divergence_glm_fit.rs`; the 8 pre-existing unweighted oracle tests stay green. |
//! | REQ-15 (non-finite input rejected) | SHIPPED | The shared IRLS entry `fn fit_glm_irls` — which every estimator (`PoissonRegressor`/`GammaRegressor`/`TweedieRegressor`/`GLMRegressor`) routes through — rejects any NaN/+/-inf in X, y, or `sample_weight` BEFORE the y-domain check and the IRLS loop with `FerroError::InvalidParameter`, mirroring sklearn's `_validate_data(force_all_finite=True)` (`glm.py:189-196`) + `_check_sample_weight` (default `force_all_finite=True`, `glm.py:211`) → `ValueError("Input X contains NaN.")` / `"... contains infinity ..."`. Placed ONCE at the shared entry (R-DEFER-8 single instance). `.iter().any(|v| !v.is_finite())` catches both NaN and Inf; the finite path is byte-identical (the unweighted `Fit::fit` delegates an all-ones weight). Verified vs the live sklearn 1.5.2 oracle (R-CHAR-3): `PoissonRegressor`/`GammaRegressor`/`TweedieRegressor`(`.fit`) raise `ValueError` for NaN/+inf/-inf in X, NaN/inf in y, and NaN/inf in sample_weight (`tests/divergence_linear_nonfinite_batch3.rs::{poisson,gamma,tweedie}_*`). Non-test consumer: each estimator's crate-root-exported `Fit::fit`. (#2261) |
//! | REQ-14 (n_iter_ + per-family y-domain validation) | SHIPPED | #560. Per-family y-domain guard in `fn fit_glm_irls`: `YDomain::for_power(family.domain_power())` then `y.iter().any(|&yi| !y_domain.contains(yi))` → `FerroError::InvalidParameter{name:"y", reason:"Some value(s) of y are out of the valid range of the loss '<loss>'."}`, mirroring sklearn's `if not base_loss.in_y_true_range(y): raise ValueError(...)` (`glm.py:221-225`). The valid range is keyed on the family's Tweedie `power` (NOT the link — verified vs the live oracle that `HalfTweedieLoss(p).interval_y_true == HalfTweedieLossIdentity(p).interval_y_true`): `power <= 0` unconstrained (Normal), `0 < power < 2` → `y >= 0` (Poisson `power=1`), `power >= 2` → `y > 0` (Gamma `power=2`, open at 0). `FittedGLMRegressor` gains `n_iter: usize` (the IRLS iteration count captured in the convergence loop) with `#[must_use] pub fn n_iter(&self) -> usize` — sklearn's `n_iter_` is the lbfgs count (`glm.py:110-114, :283`); ferrolearn's is the IRLS count (solvers differ, both report iterations-to-convergence). Consumer: `FittedGLMRegressor::n_iter` accessor on the crate-root-exported fitted type. Oracle tests `glm_gamma_rejects_zero_y` (Gamma rejects `y==0`, accepts `y>0`), `glm_tweedie_power2_rejects_zero_y` (`power=2.0` rejects `y==0`; `power=1.5` accepts it), `glm_poisson_rejects_negative_y` (rejects `y<0`, accepts `y==0`), `glm_n_iter_exposed` (`1 <= n_iter() <= max_iter`) green in `tests/divergence_glm_fit.rs`; the 10 pre-existing glm divergence tests stay green (all their `y` are in-domain). |

use ferrolearn_core::error::FerroError;
use ferrolearn_core::introspection::HasCoefficients;
use ferrolearn_core::pipeline::{FittedPipelineEstimator, PipelineEstimator};
use ferrolearn_core::traits::{Fit, Predict};
use ndarray::{Array1, Array2, ScalarOperand};
use num_traits::{Float, FromPrimitive};

// ---------------------------------------------------------------------------
// Link
// ---------------------------------------------------------------------------

/// The link function `g` of a Generalized Linear Model, mapping the mean `mu`
/// to the linear predictor `eta = g(mu)` (and back via the inverse link `h`,
/// `mu = h(eta)`).
///
/// Mirrors the link carried by sklearn's loss classes
/// (`sklearn/linear_model/_glm/glm.py:119-131`): `HalfPoissonLoss`,
/// `HalfGammaLoss` and `HalfTweedieLoss` use the **log** link
/// (`y_pred = exp(X @ coef + intercept)`); `HalfSquaredError` and
/// `HalfTweedieLossIdentity` use the **identity** link
/// (`y_pred = X @ coef + intercept`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Link {
    /// Log link: `g(mu) = ln(mu)`, inverse `h(eta) = exp(eta)`.
    ///
    /// Used by Poisson, Gamma and Tweedie-with-`power > 0` losses.
    Log,
    /// Identity link: `g(mu) = mu`, inverse `h(eta) = eta`.
    ///
    /// Used by the Normal/least-squares loss and Tweedie-with-`power <= 0`.
    Identity,
}

impl Link {
    /// Inverse link `h(eta) = mu`: maps the linear predictor to the mean.
    ///
    /// - [`Link::Log`] → `exp(eta)`
    /// - [`Link::Identity`] → `eta`
    ///
    /// Mirrors `link.inverse(raw_prediction)` in `glm.py:362`.
    #[must_use]
    fn inverse<F: Float>(self, eta: F) -> F {
        match self {
            Link::Log => eta.exp(),
            Link::Identity => eta,
        }
    }

    /// Forward link `g(mu) = eta`: maps the mean to the linear predictor.
    ///
    /// - [`Link::Log`] → `ln(mu)`
    /// - [`Link::Identity`] → `mu`
    ///
    /// Mirrors `link.link(...)` used by sklearn to seed the intercept at
    /// `link.link(average(y))` (`glm.py:254-256`).
    #[must_use]
    fn link<F: Float>(self, mu: F) -> F {
        match self {
            Link::Log => mu.ln(),
            Link::Identity => mu,
        }
    }

    /// Link derivative of the mean w.r.t. the linear predictor, `dmu/deta`,
    /// used to form the IRLS working weight and response.
    ///
    /// - [`Link::Log`] (`mu = exp(eta)`) → `dmu/deta = mu`
    /// - [`Link::Identity`] (`mu = eta`) → `dmu/deta = 1`
    #[must_use]
    fn dmu_deta<F: Float>(self, mu: F) -> F {
        match self {
            Link::Log => mu,
            Link::Identity => F::one(),
        }
    }
}

/// Configuration of the GLM link function, resolved to a concrete [`Link`] at
/// fit time.
///
/// Mirrors sklearn's `TweedieRegressor(link={'auto','identity','log'})`
/// (`glm.py:861, :889-903`). `Auto` selects the link from the Tweedie `power`:
/// identity for `power <= 0` (Normal), log otherwise (Poisson/Gamma/etc.).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinkConfig {
    /// Resolve the link from the Tweedie `power` at fit time
    /// (`power <= 0` → identity, `power > 0` → log). The default.
    Auto,
    /// Force the log link regardless of `power`.
    Log,
    /// Force the identity link regardless of `power`.
    Identity,
}

impl LinkConfig {
    /// Resolve to a concrete [`Link`] given the Tweedie `power`.
    ///
    /// Mirrors `TweedieRegressor._get_loss` (`glm.py:889-903`):
    /// - `Auto` → identity for `power <= 0`, log for `power > 0`
    /// - `Log` → log; `Identity` → identity.
    #[must_use]
    fn resolve(self, power: f64) -> Link {
        match self {
            LinkConfig::Auto => {
                if power <= 0.0 {
                    Link::Identity
                } else {
                    Link::Log
                }
            }
            LinkConfig::Log => Link::Log,
            LinkConfig::Identity => Link::Identity,
        }
    }
}

// ---------------------------------------------------------------------------
// Solver
// ---------------------------------------------------------------------------

/// The optimization algorithm requested for fitting a GLM, mirroring sklearn's
/// `solver` constructor parameter (`sklearn/linear_model/_glm/glm.py:140-145`,
/// `StrOptions({"lbfgs", "newton-cholesky"})`, default `"lbfgs"`).
///
/// **Implementation note (R-DEV-7 — Rust analog).** ferrolearn fits all GLMs via
/// IRLS / Fisher-scoring (`fn fit_glm_irls`); the `solver` parameter is accepted
/// for scikit-learn API parity (R-DEV-2) and selects the requested optimizer's
/// *contract*. The penalized GLM objective is convex, so IRLS converges to the
/// **same** minimizer as both sklearn's `lbfgs` (scipy L-BFGS-B) and
/// `newton-cholesky` (Newton-Raphson with an inner Cholesky solve) — all three
/// are descent methods on one convex objective. Therefore the observable
/// contract (`coef_` / `intercept_`) matches sklearn for **either** `solver`
/// value, and ferrolearn does not vary the numerical path between them
/// (verified live: `PoissonRegressor(alpha=0.5)` gives the same fitted
/// attributes to ~1e-9 for `lbfgs` and `newton-cholesky`).
///
/// The type system constrains valid values to the two sklearn options, mirroring
/// the role of sklearn's `StrOptions` parameter constraint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Solver {
    /// L-BFGS-B (sklearn's default `"lbfgs"`): a quasi-Newton optimizer on the
    /// penalized loss + gradient (`glm.py:263-284`). In ferrolearn the fit is
    /// performed by IRLS, which reaches the same convex optimum (R-DEV-7).
    Lbfgs,
    /// Newton-Cholesky (sklearn's `"newton-cholesky"`): Newton-Raphson steps
    /// with an inner Cholesky solve, equivalent in exact arithmetic to iterated
    /// reweighted least squares (`glm.py:72-78, :285-296`). In ferrolearn the
    /// fit is performed by IRLS, which reaches the same convex optimum (R-DEV-7).
    NewtonCholesky,
}

// ---------------------------------------------------------------------------
// GLMFamily
// ---------------------------------------------------------------------------

/// The distributional family for a Generalized Linear Model.
///
/// Determines the variance function V(mu):
/// - **Poisson**: V(mu) = mu
/// - **Gamma**: V(mu) = mu^2
/// - **Tweedie(p)**: V(mu) = mu^p
#[derive(Debug, Clone, Copy)]
pub enum GLMFamily {
    /// Poisson family — variance proportional to the mean.
    Poisson,
    /// Gamma family — variance proportional to the squared mean.
    Gamma,
    /// Tweedie family with power parameter `p`.
    ///
    /// - `p = 0` gives Normal (constant variance)
    /// - `p = 1` gives Poisson
    /// - `p = 2` gives Gamma
    /// - `1 < p < 2` gives compound Poisson-Gamma
    Tweedie(f64),
}

impl GLMFamily {
    /// Compute the variance function V(mu) for a given mean `mu`.
    fn variance<F: Float + FromPrimitive>(&self, mu: F) -> F {
        match self {
            GLMFamily::Poisson => mu,
            GLMFamily::Gamma => mu * mu,
            GLMFamily::Tweedie(p) => {
                let power = F::from(*p).unwrap_or_else(F::zero);
                mu.powf(power)
            }
        }
    }

    /// The Tweedie power that determines this family's target (`y`) domain.
    ///
    /// The sklearn EDM losses derive their `interval_y_true` from the Tweedie
    /// `power` alone (`HalfPoissonLoss` is `power = 1`, `HalfGammaLoss` is
    /// `power = 2`); the link function does NOT change the valid `y` range
    /// (verified against the live oracle: `HalfTweedieLoss(power).interval_y_true
    /// == HalfTweedieLossIdentity(power).interval_y_true`). See
    /// [`YDomain::for_power`].
    #[must_use]
    fn domain_power(&self) -> f64 {
        match self {
            GLMFamily::Poisson => 1.0,
            GLMFamily::Gamma => 2.0,
            GLMFamily::Tweedie(p) => *p,
        }
    }

    /// The per-sample **unit deviance** `dev(y, mu)` of this family, the
    /// quantity whose `sample_weight`-weighted sum forms the D² numerator and
    /// denominator (`fn score`).
    ///
    /// This equals `2 * (loss(y, mu) + constant_to_optimal_zero(y))` of the
    /// matching sklearn EDM loss — i.e. twice the half-deviance with the
    /// `raw_prediction`-independent constant restored so a perfect prediction
    /// has zero deviance (`sklearn/_loss/loss.py`: `HalfPoissonLoss` `:728-742`,
    /// `HalfGammaLoss` `:754-773`, `HalfTweedieLoss` `:789-837`). Verified
    /// term-for-term against the live sklearn 1.5.2 oracle.
    ///
    /// Per family (`p` = Tweedie power):
    /// - **Poisson** (`p = 1`): `2·(y·ln(y/mu) − y + mu)`, with the convention
    ///   `y·ln(y/mu) → 0` at `y == 0` (so `dev = 2·mu`).
    /// - **Gamma** (`p = 2`): `2·(−ln(y/mu) + (y − mu)/mu)`.
    /// - **Tweedie `p == 0`** (Normal): `(y − mu)²`.
    /// - **Tweedie general** `p ∉ {0, 1, 2}`:
    ///   `2·( max(y,0)^(2−p)/((1−p)(2−p)) − y·mu^(1−p)/(1−p) + mu^(2−p)/(2−p) )`,
    ///   matching `HalfTweedieLoss.loss` (`loss.py:792-794`); `max(y, 0)` guards
    ///   `y == 0` (the `y^(2−p)` term is then 0).
    #[must_use]
    fn unit_deviance<F: Float + FromPrimitive>(&self, y: F, mu: F) -> F {
        let two = F::from(2.0).unwrap_or_else(|| F::one() + F::one());
        match self {
            GLMFamily::Poisson => {
                // 2·(y·ln(y/mu) − y + mu); y·ln(y/mu) → 0 at y == 0.
                let y_term = if y > F::zero() {
                    y * (y / mu).ln()
                } else {
                    F::zero()
                };
                two * (y_term - y + mu)
            }
            GLMFamily::Gamma => {
                // 2·(−ln(y/mu) + (y − mu)/mu).
                two * (F::zero() - (y / mu).ln() + (y - mu) / mu)
            }
            GLMFamily::Tweedie(p) => self.tweedie_unit_deviance(*p, y, mu),
        }
    }

    /// Tweedie unit deviance for an arbitrary power `p`, dispatching the
    /// `p ∈ {1, 2}` special cases to Poisson/Gamma and `p == 0` to the Normal
    /// squared error, matching `HalfTweedieLoss` taking the `p → 0, 1, 2` limits
    /// (`loss.py:796-797`).
    #[must_use]
    fn tweedie_unit_deviance<F: Float + FromPrimitive>(&self, p: f64, y: F, mu: F) -> F {
        let two = F::from(2.0).unwrap_or_else(|| F::one() + F::one());
        if p == 0.0 {
            // Normal: (y − mu)².
            let d = y - mu;
            return d * d;
        }
        if p == 1.0 {
            return GLMFamily::Poisson.unit_deviance(y, mu);
        }
        if p == 2.0 {
            return GLMFamily::Gamma.unit_deviance(y, mu);
        }
        // General p: 2·( max(y,0)^(2−p)/((1−p)(2−p)) − y·mu^(1−p)/(1−p)
        //                 + mu^(2−p)/(2−p) ).
        let pf = F::from(p).unwrap_or_else(F::zero);
        let one = F::one();
        let one_minus_p = one - pf;
        let two_minus_p = two - pf;
        let y_pos = if y > F::zero() { y } else { F::zero() };
        let t1 = y_pos.powf(two_minus_p) / (one_minus_p * two_minus_p);
        let t2 = y * mu.powf(one_minus_p) / one_minus_p;
        let t3 = mu.powf(two_minus_p) / two_minus_p;
        two * (t1 - t2 + t3)
    }

    /// The `sample_weight`-independent constant `constant_to_optimal_zero(y)` of
    /// the matching sklearn EDM loss — the term sklearn DROPS from its
    /// half-loss so that a perfect prediction scores zero deviance/2 (it is
    /// restored when forming the unit deviance).
    ///
    /// Mirrors `sklearn/_loss/loss.py`: `HalfPoissonLoss.constant_to_optimal_zero`
    /// `:738-742` = `xlogy(y, y) − y` (`xlogy(0, 0) = 0`); `HalfGammaLoss` `:769-773`
    /// = `−ln(y) − 1`; `HalfSquaredError`/Tweedie-identity `:453-458` = `0`;
    /// `HalfTweedieLoss.constant_to_optimal_zero` `:819-837` dispatches `p ∈ {0,1,2}`
    /// and is `max(y,0)^(2−p)/((1−p)(2−p))` otherwise.
    ///
    /// `score` adds this constant to both the model and the null half-deviance so
    /// the D² expression `1 − (deviance + constant)/(deviance_null + constant)`
    /// is byte-for-byte sklearn's (`glm.py:419-438`); it only affects the result
    /// at the degenerate constant-`y` boundary (`deviance_null == 0`), where
    /// sklearn returns `0` for Poisson (`constant ≠ 0`) and `NaN` for
    /// Gamma/Normal (`constant == 0`).
    #[must_use]
    fn constant_to_optimal_zero<F: Float + FromPrimitive>(&self, y: F) -> F {
        match self {
            GLMFamily::Poisson => {
                // xlogy(y, y) − y; xlogy(0, 0) = 0.
                let xlogy = if y > F::zero() { y * y.ln() } else { F::zero() };
                xlogy - y
            }
            GLMFamily::Gamma => {
                // −ln(y) − 1.
                F::zero() - y.ln() - F::one()
            }
            GLMFamily::Tweedie(p) => self.tweedie_constant_to_optimal_zero(*p, y),
        }
    }

    /// Tweedie `constant_to_optimal_zero` for an arbitrary power `p`, dispatching
    /// `p ∈ {0, 1, 2}` to Normal/Poisson/Gamma (`loss.py:819-831`) and using
    /// `max(y,0)^(2−p)/((1−p)(2−p))` otherwise (`loss.py:832-837`).
    #[must_use]
    fn tweedie_constant_to_optimal_zero<F: Float + FromPrimitive>(&self, p: f64, y: F) -> F {
        if p == 0.0 {
            // HalfSquaredError: 0.
            return F::zero();
        }
        if p == 1.0 {
            return GLMFamily::Poisson.constant_to_optimal_zero(y);
        }
        if p == 2.0 {
            return GLMFamily::Gamma.constant_to_optimal_zero(y);
        }
        let two = F::from(2.0).unwrap_or_else(|| F::one() + F::one());
        let pf = F::from(p).unwrap_or_else(F::zero);
        let one = F::one();
        let one_minus_p = one - pf;
        let two_minus_p = two - pf;
        let y_pos = if y > F::zero() { y } else { F::zero() };
        y_pos.powf(two_minus_p) / (one_minus_p * two_minus_p)
    }
}

/// The valid target (`y`) domain of a GLM loss, derived from its Tweedie
/// `power`, mirroring sklearn's `BaseLoss.interval_y_true` /
/// `in_y_true_range(y)` (`sklearn/linear_model/_glm/glm.py:221-225`).
///
/// The interval depends on `power` only (NOT the link); verified against the
/// live sklearn 1.5.2 oracle:
///
/// | power range          | valid `y`     | sklearn loss / interval                            |
/// |----------------------|---------------|----------------------------------------------------|
/// | `power <= 0`         | any real `y`  | `HalfSquaredError`/Tweedie identity, `(-inf, inf)` |
/// | `0 < power < 2`      | `y >= 0`      | Poisson (`power = 1`): `[0, inf)`                  |
/// | `power >= 2`         | `y > 0`       | Gamma (`power = 2`): `(0, inf)`                    |
///
/// (`power < 0` is the identity-link case, unconstrained; `0 < power < 1` is not
/// a standard Tweedie EDM but sklearn's `HalfTweedieLoss(power).interval_y_true`
/// is `[0, inf)` there, so we treat it as `y >= 0`.)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum YDomain {
    /// `y` may be any real number (`power <= 0`, Normal/identity).
    Unconstrained,
    /// `y >= 0` (closed at 0; `0 < power < 2`, e.g. Poisson `power = 1`).
    NonNegative,
    /// `y > 0` (open at 0; `power >= 2`, e.g. Gamma `power = 2`).
    Positive,
}

impl YDomain {
    /// Resolve the valid `y` domain from a Tweedie `power`.
    ///
    /// Mirrors the boundaries of `HalfTweedieLoss(power).interval_y_true`
    /// (live sklearn 1.5.2 oracle): `power <= 0` → unconstrained, `0 < power < 2`
    /// → `y >= 0`, `power >= 2` → `y > 0`.
    #[must_use]
    fn for_power(power: f64) -> Self {
        if power <= 0.0 {
            YDomain::Unconstrained
        } else if power < 2.0 {
            YDomain::NonNegative
        } else {
            YDomain::Positive
        }
    }

    /// Human-readable description of the loss whose domain this is, for the
    /// error message (mirrors sklearn's `loss.__class__.__name__`,
    /// `glm.py:224`).
    #[must_use]
    fn loss_name(self) -> &'static str {
        match self {
            YDomain::Unconstrained => "HalfSquaredError",
            YDomain::NonNegative => "HalfPoissonLoss",
            YDomain::Positive => "HalfGammaLoss",
        }
    }

    /// Whether a single target value `yi` is inside this domain.
    #[must_use]
    fn contains<F: Float>(self, yi: F) -> bool {
        match self {
            YDomain::Unconstrained => true,
            YDomain::NonNegative => yi >= F::zero(),
            YDomain::Positive => yi > F::zero(),
        }
    }
}

// ---------------------------------------------------------------------------
// GLMRegressor
// ---------------------------------------------------------------------------

/// Generalized Linear Model regressor.
///
/// Fitted via IRLS with a log link function. The [`GLMFamily`] controls
/// the assumed variance-mean relationship.
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
#[derive(Debug, Clone)]
pub struct GLMRegressor<F> {
    /// Distributional family (Poisson, Gamma, or Tweedie).
    pub family: GLMFamily,
    /// L2 regularization strength.
    pub alpha: F,
    /// Maximum number of IRLS iterations.
    pub max_iter: usize,
    /// Convergence tolerance on the maximum coefficient change.
    pub tol: F,
    /// Whether to fit an intercept (bias) term.
    pub fit_intercept: bool,
    /// Optimization algorithm requested, mirroring sklearn's `solver` parameter
    /// (`glm.py:140-145`, default `"lbfgs"`).
    ///
    /// ferrolearn fits via IRLS / Fisher-scoring regardless of this value
    /// (R-DEV-7); the parameter is accepted for sklearn API parity (R-DEV-2) and
    /// the observable `coef_` / `intercept_` match sklearn for either value
    /// because IRLS reaches the same convex optimum as both `lbfgs` and
    /// `newton-cholesky`. See [`Solver`].
    pub solver: Solver,
    /// Whether to warm-start the optimizer from an explicit initial point,
    /// mirroring sklearn's `warm_start` constructor parameter (default `false`,
    /// `glm.py:146, :158, :244`).
    ///
    /// **R-DEV-2 (API parity):** the name and default match sklearn. **R-DEV-7
    /// (Rust analog):** sklearn reuses the stateful `self.coef_` / `self.intercept_`
    /// across `fit` calls (`glm.py:244-250`); ferrolearn's estimators are immutable
    /// (`fit(&self, ...)` never mutates `self`), so the warm-start point is supplied
    /// EXPLICITLY via [`GLMRegressor::with_coef_init`]. When `warm_start == true`
    /// and an init is set, the IRLS seeds from it; otherwise it cold-starts at
    /// `coef = 0`. Because the GLM objective is convex, the converged
    /// `coef_` / `intercept_` are warm-start-invariant — identical to the cold
    /// fit and to sklearn regardless of the init.
    pub warm_start: bool,
    /// Explicit warm-start initial point `(feature_coefficients, intercept)` —
    /// the ferrolearn analog of sklearn reusing `self.coef_` / `self.intercept_`
    /// across `fit` calls (R-DEV-7, `glm.py:244-250`).
    ///
    /// Only consulted when [`GLMRegressor::warm_start`] is `true`. The
    /// feature-coefficient vector length must equal the number of features in `X`
    /// (else [`FerroError::ShapeMismatch`] at fit time). Set via
    /// [`GLMRegressor::with_coef_init`]; typically a previous fit's
    /// `coefficients()` / `intercept()`.
    pub coef_init: Option<(Array1<F>, F)>,
}

impl<F: Float + FromPrimitive> GLMRegressor<F> {
    /// Create a new `GLMRegressor` with the given family.
    ///
    /// Defaults: `alpha = 1.0`, `max_iter = 100`, `tol = 1e-4`,
    /// `fit_intercept = true`, `solver = Solver::Lbfgs` (sklearn default).
    #[must_use]
    pub fn new(family: GLMFamily) -> Self {
        Self {
            family,
            alpha: F::one(),
            max_iter: 100,
            tol: F::from(1e-4).unwrap_or_else(F::epsilon),
            fit_intercept: true,
            solver: Solver::Lbfgs,
            warm_start: false,
            coef_init: None,
        }
    }

    /// Set the L2 regularization strength.
    #[must_use]
    pub fn with_alpha(mut self, alpha: F) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set the maximum number of IRLS iterations.
    #[must_use]
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the convergence tolerance.
    #[must_use]
    pub fn with_tol(mut self, tol: F) -> Self {
        self.tol = tol;
        self
    }

    /// Set whether to fit an intercept term.
    #[must_use]
    pub fn with_fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = fit_intercept;
        self
    }

    /// Set the optimization [`Solver`], mirroring sklearn's `solver` parameter
    /// (`glm.py:140-145`, default `"lbfgs"`).
    ///
    /// ferrolearn fits via IRLS regardless of the value (R-DEV-7); the parameter
    /// is accepted for sklearn API parity (R-DEV-2) and both values produce the
    /// same observable `coef_` / `intercept_` (IRLS reaches the same convex
    /// optimum as both `lbfgs` and `newton-cholesky`).
    #[must_use]
    pub fn with_solver(mut self, solver: Solver) -> Self {
        self.solver = solver;
        self
    }

    /// Enable or disable warm-starting, mirroring sklearn's `warm_start`
    /// parameter (default `false`, `glm.py:146, :158`).
    ///
    /// **R-DEV-2 / R-DEV-7.** sklearn reuses the stateful `self.coef_` /
    /// `self.intercept_` across `fit` calls (`glm.py:244-250`); ferrolearn's
    /// immutable estimators take the warm-start point EXPLICITLY via
    /// [`GLMRegressor::with_coef_init`]. When `warm_start` is `true` and an init
    /// is set, the IRLS seeds from it; otherwise it cold-starts at `coef = 0`. The
    /// convex GLM objective makes the converged `coef_` / `intercept_`
    /// warm-start-invariant.
    #[must_use]
    pub fn with_warm_start(mut self, warm_start: bool) -> Self {
        self.warm_start = warm_start;
        self
    }

    /// Set the explicit warm-start initial point `(feature_coefficients,
    /// intercept)` — the ferrolearn analog of sklearn reusing `self.coef_` /
    /// `self.intercept_` (R-DEV-7, `glm.py:244-250`).
    ///
    /// Only consulted when [`GLMRegressor::warm_start`] is `true`. `coef`'s length
    /// must equal the number of features in `X` (validated at fit time). Pass a
    /// previous fit's `coefficients()` / `intercept()` to resume from it.
    #[must_use]
    pub fn with_coef_init(mut self, coef: Array1<F>, intercept: F) -> Self {
        self.coef_init = Some((coef, intercept));
        self
    }
}

impl<F: Float + Send + Sync + ScalarOperand + FromPrimitive + 'static> GLMRegressor<F> {
    /// Fit the GLM via IRLS with per-sample weights `sample_weight`.
    ///
    /// Mirrors sklearn's `fit(X, y, sample_weight)` (`glm.py:170`,
    /// `:229-242`): the deviance term is a `sample_weight`-weighted average
    /// (normalized by `S = sum_i s_i`), so the IRLS `W` diagonal becomes
    /// `s_i * w_irls,i` and the L2 penalty scales with `S`. The working
    /// response `z_i` is unchanged per sample. Passing an all-ones weight
    /// vector reproduces [`Fit::fit`] exactly.
    ///
    /// # Errors
    ///
    /// - [`FerroError::ShapeMismatch`] — `sample_weight.len() != n_samples`,
    ///   or `y` length mismatch.
    /// - [`FerroError::InvalidParameter`] — a negative sample weight, negative
    ///   alpha, or (log link) negative `y`.
    /// - [`FerroError::InsufficientSamples`] — zero samples.
    pub fn fit_with_sample_weight(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
        sample_weight: &Array1<F>,
    ) -> Result<FittedGLMRegressor<F>, FerroError> {
        fit_glm_irls(
            x,
            y,
            sample_weight,
            &self.family,
            Link::Log,
            self.alpha,
            self.max_iter,
            self.tol,
            self.fit_intercept,
            self.warm_start,
            self.coef_init.as_ref(),
        )
    }
}

/// Fitted GLM regressor.
///
/// Stores the learned coefficients and intercept on the link scale, together
/// with the [`Link`] used at fit time. Predictions are
/// `link.inverse(X @ coef + intercept)` — `exp(...)` for [`Link::Log`], the raw
/// linear predictor for [`Link::Identity`] (`glm.py:362`).
#[derive(Debug, Clone)]
pub struct FittedGLMRegressor<F> {
    /// Learned coefficient vector on the link scale.
    coefficients: Array1<F>,
    /// Learned intercept on the link scale.
    intercept: F,
    /// Link function applied by `predict` (inverse link maps `eta` to `mu`).
    link: Link,
    /// Distributional family, retained so [`FittedGLMRegressor::score`] can
    /// compute this family's unit deviance for the D² (deviance-explained)
    /// score (`glm.py:365-438`).
    family: GLMFamily,
    /// Number of IRLS iterations actually run (until convergence or `max_iter`).
    ///
    /// Mirrors sklearn's fitted `n_iter_` attribute (`glm.py:110-114, :283`),
    /// the solver's iteration count. ferrolearn's solver is IRLS (not lbfgs), so
    /// this is the **IRLS** iteration count; both report iterations-to-convergence
    /// but the exact value is solver-dependent.
    n_iter: usize,
}

impl<F> FittedGLMRegressor<F> {
    /// Number of IRLS iterations run during the fit
    /// (until convergence or `max_iter`).
    ///
    /// Mirrors scikit-learn's fitted `n_iter_` attribute
    /// (`sklearn/linear_model/_glm/glm.py:110-114, :283`), which reports the
    /// solver's iteration count. ferrolearn's solver is **IRLS** (sklearn's
    /// default is lbfgs), so this is the IRLS iteration count, not the lbfgs
    /// one — both report iterations-to-convergence, but the exact value differs
    /// because the solvers differ. The value is in `1..=max_iter`.
    #[must_use]
    pub fn n_iter(&self) -> usize {
        self.n_iter
    }
}

impl<F: Float + Send + Sync + ScalarOperand + FromPrimitive + 'static> FittedGLMRegressor<F> {
    /// Compute **D²**, the fraction of deviance explained — the GLM
    /// generalization of R² (which is the special case for the Normal family).
    ///
    /// Mirrors `sklearn`'s `_GeneralizedLinearRegressor.score`
    /// (`sklearn/linear_model/_glm/glm.py:365-438`):
    ///
    /// ```text
    /// D² = 1 − (deviance + constant) / (deviance_null + constant)
    /// ```
    ///
    /// where `deviance` is the `sample_weight`-mean half-deviance of the fitted
    /// model's predictions `mu = predict(X)`, `deviance_null` is the same for the
    /// constant null model that predicts the weighted mean of `y` for every
    /// sample (`y_pred = link.inverse(link(mean_w(y))) = mean_w(y)`), and
    /// `constant` is the family's `constant_to_optimal_zero(y)` mean
    /// (`glm.py:419-438`). Because the dropped factor-of-2 and the `constant`
    /// cancel whenever `deviance_null ≠ 0`, the result equals the unit-deviance
    /// ratio `1 − Σ sᵢ·dev(yᵢ, muᵢ) / Σ sᵢ·dev(yᵢ, ȳ_w)`. The best possible
    /// score is `1.0`; it can be negative (an arbitrarily worse-than-null model).
    ///
    /// This is the unweighted path (`sample_weight = None`); the weighted mean
    /// `ȳ_w` reduces to the plain mean of `y`.
    ///
    /// **Degenerate (constant-`y`) boundary.** When `y` is constant the null
    /// deviance is `0`, so sklearn evaluates `1 − (deviance + constant)/0`. The
    /// returned value is sklearn's exact algebraic form, but the result there
    /// depends on the fitted model reproducing `mu == ȳ` to full precision: with
    /// sklearn's lbfgs solver `deviance + constant == deviance_null + constant`
    /// (both equal the float roundoff of `loss + constant`), giving `0` for
    /// Poisson and `NaN` for Gamma/Normal (`constant == 0`). ferrolearn's IRLS
    /// converges `mu` to `ȳ` only to solver tolerance (not bit-exactly), so on a
    /// constant-`y` Poisson input the ratio is `~1 ± ε` (≈ `0`) rather than
    /// exactly `0`; this is a fit-precision artifact of the degenerate input, not
    /// of the D² formula (non-degenerate inputs match the oracle to `< 1e-6`).
    ///
    /// `y` is re-validated against the family's target domain, mirroring
    /// sklearn's `if not base_loss.in_y_true_range(y): raise ValueError(...)` in
    /// `score` (`glm.py:413-417`).
    ///
    /// # Errors
    ///
    /// - [`FerroError::ShapeMismatch`] — `X` feature count mismatch (via
    ///   [`Predict::predict`]) or `y` length mismatch.
    /// - [`FerroError::InvalidParameter`] — a value of `y` is out of the family's
    ///   valid target range.
    #[must_use = "the computed D² score should be used"]
    pub fn score(&self, x: &Array2<F>, y: &Array1<F>) -> Result<F, FerroError> {
        let mu = self.predict(x)?;

        if y.len() != mu.len() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![mu.len()],
                actual: vec![y.len()],
                context: "y length must match number of samples in X".into(),
            });
        }

        // Re-validate y against the family's target domain, mirroring sklearn's
        // `score`: `if not base_loss.in_y_true_range(y): raise ValueError(...)`
        // (`glm.py:413-417`). Same per-family domain as `fit` (keyed on power).
        let y_domain = YDomain::for_power(self.family.domain_power());
        if y.iter().any(|&yi| !y_domain.contains(yi)) {
            return Err(FerroError::InvalidParameter {
                name: "y".into(),
                reason: format!(
                    "Some value(s) of y are out of the valid range of the loss '{}'.",
                    y_domain.loss_name()
                ),
            });
        }

        // Weighted mean of y (unweighted => plain mean). The null model predicts
        // `ȳ` for every sample (`glm.py:431`: `link.link(average(y))` mapped back
        // through `link.inverse` in the null deviance is just `ȳ`).
        let n = F::from(y.len()).unwrap_or_else(F::one);
        let y_bar = y.iter().fold(F::zero(), |acc, &yi| acc + yi) / n;

        // sklearn forms `deviance + constant` and `deviance_null + constant`,
        // where `deviance = mean(loss)`, `deviance_null = mean(loss_null)` and
        // `constant = mean(constant_to_optimal_zero(y))` (`glm.py:419-437`).
        // Because per sample `loss + constant_to_optimal_zero = ½·unit_deviance`
        // (the `loss` drops exactly the constant the unit deviance restores), the
        // two terms sklearn divides are simply the mean half unit deviances:
        //   deviance + constant      = mean(½·unit_deviance(y, mu_model))
        //   deviance_null + constant = mean(½·unit_deviance(y, ȳ)).
        // We accumulate `½·unit_deviance` on each side directly — adding `constant`
        // again would double-count it. The factor ½ cancels in the ratio for any
        // nonzero denominator; we keep it so the constant-`y` boundary
        // (`deviance_null == 0`) reproduces sklearn's exact `½·unit_dev_model / 0`
        // → it must, however, also restore the dropped `constant` at that boundary
        // to distinguish Poisson (0) from Gamma/Normal (NaN). We therefore compute
        // sklearn's `(deviance + constant)` / `(deviance_null + constant)` form
        // with `deviance = mean(loss)`, `loss = ½·unit_deviance − constant`.
        let half = F::from(0.5).unwrap_or_else(|| F::one() / (F::one() + F::one()));
        let mut sum_loss_model = F::zero();
        let mut sum_loss_null = F::zero();
        let mut sum_const = F::zero();
        for (&yi, &mui) in y.iter().zip(mu.iter()) {
            let c = self.family.constant_to_optimal_zero(yi);
            sum_loss_model = sum_loss_model + (half * self.family.unit_deviance(yi, mui) - c);
            sum_loss_null = sum_loss_null + (half * self.family.unit_deviance(yi, y_bar) - c);
            sum_const = sum_const + c;
        }
        let deviance = sum_loss_model / n;
        let deviance_null = sum_loss_null / n;
        let constant = sum_const / n;

        // `1 − (deviance + constant) / (deviance_null + constant)` (glm.py:438).
        // Away from the constant-`y` boundary this equals the unit-deviance ratio
        // `1 − Σ dev(y, mu) / Σ dev(y, ȳ)`; at `deviance_null == 0` the restored
        // `constant` reproduces sklearn's family-dependent result — 0 for Poisson
        // (`constant ≠ 0`), NaN for Gamma/Normal (`constant == 0`) — verified
        // against the live sklearn 1.5.2 oracle.
        Ok(F::one() - (deviance + constant) / (deviance_null + constant))
    }
}

// ---------------------------------------------------------------------------
// Convenience wrappers
// ---------------------------------------------------------------------------

/// Poisson regressor — GLM with Poisson family and log link.
///
/// Suitable for modelling count data (y >= 0, integer-valued).
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
#[derive(Debug, Clone)]
pub struct PoissonRegressor<F> {
    /// L2 regularization strength.
    pub alpha: F,
    /// Maximum number of IRLS iterations.
    pub max_iter: usize,
    /// Convergence tolerance.
    pub tol: F,
    /// Whether to fit an intercept.
    pub fit_intercept: bool,
    /// Optimization algorithm requested, mirroring sklearn's `solver` parameter
    /// (`glm.py:140-145`, default `"lbfgs"`).
    ///
    /// ferrolearn fits via IRLS regardless of this value (R-DEV-7); the parameter
    /// is accepted for sklearn API parity (R-DEV-2) and the observable
    /// `coef_` / `intercept_` match sklearn for either value. See [`Solver`].
    pub solver: Solver,
    /// Whether to warm-start from an explicit initial point, mirroring sklearn's
    /// `warm_start` parameter (default `false`, `glm.py:146, :576`). See
    /// [`GLMRegressor::warm_start`] for the R-DEV-2 / R-DEV-7 rationale.
    pub warm_start: bool,
    /// Explicit warm-start initial point `(feature_coefficients, intercept)` — the
    /// ferrolearn analog of sklearn reusing `self.coef_` / `self.intercept_`
    /// (R-DEV-7, `glm.py:244-250`). Consulted only when `warm_start` is `true`. Set
    /// via [`PoissonRegressor::with_coef_init`].
    pub coef_init: Option<(Array1<F>, F)>,
}

impl<F: Float + FromPrimitive> PoissonRegressor<F> {
    /// Create a new `PoissonRegressor` with default settings.
    ///
    /// Defaults: `alpha = 1.0`, `max_iter = 100`, `tol = 1e-4`,
    /// `fit_intercept = true`, `solver = Solver::Lbfgs` (sklearn default).
    #[must_use]
    pub fn new() -> Self {
        Self {
            alpha: F::one(),
            max_iter: 100,
            tol: F::from(1e-4).unwrap_or_else(F::epsilon),
            fit_intercept: true,
            solver: Solver::Lbfgs,
            warm_start: false,
            coef_init: None,
        }
    }

    /// Set the L2 regularization strength.
    #[must_use]
    pub fn with_alpha(mut self, alpha: F) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set the maximum number of IRLS iterations.
    #[must_use]
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the convergence tolerance.
    #[must_use]
    pub fn with_tol(mut self, tol: F) -> Self {
        self.tol = tol;
        self
    }

    /// Set whether to fit an intercept.
    #[must_use]
    pub fn with_fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = fit_intercept;
        self
    }

    /// Set the optimization [`Solver`], mirroring sklearn's `solver` parameter
    /// (`glm.py:140-145`, default `"lbfgs"`).
    ///
    /// ferrolearn fits via IRLS regardless of the value (R-DEV-7); both values
    /// produce the same observable `coef_` / `intercept_` (sklearn API parity,
    /// R-DEV-2).
    #[must_use]
    pub fn with_solver(mut self, solver: Solver) -> Self {
        self.solver = solver;
        self
    }

    /// Enable or disable warm-starting, mirroring sklearn's `warm_start`
    /// parameter (default `false`, `glm.py:146, :576`). See
    /// [`GLMRegressor::with_warm_start`] for the R-DEV-2 / R-DEV-7 rationale; the
    /// warm-start point is supplied explicitly via
    /// [`PoissonRegressor::with_coef_init`].
    #[must_use]
    pub fn with_warm_start(mut self, warm_start: bool) -> Self {
        self.warm_start = warm_start;
        self
    }

    /// Set the explicit warm-start initial point `(feature_coefficients,
    /// intercept)` — the ferrolearn analog of sklearn reusing `self.coef_` /
    /// `self.intercept_` (R-DEV-7, `glm.py:244-250`). Only consulted when
    /// `warm_start` is `true`; `coef`'s length must equal the number of features.
    #[must_use]
    pub fn with_coef_init(mut self, coef: Array1<F>, intercept: F) -> Self {
        self.coef_init = Some((coef, intercept));
        self
    }
}

impl<F: Float + FromPrimitive> Default for PoissonRegressor<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Float + Send + Sync + ScalarOperand + FromPrimitive + 'static> PoissonRegressor<F> {
    /// Fit the Poisson GLM via IRLS with per-sample weights `sample_weight`.
    ///
    /// Mirrors sklearn's `PoissonRegressor.fit(X, y, sample_weight)`
    /// (`glm.py:170`, `:229-242`). See
    /// [`GLMRegressor::fit_with_sample_weight`] for the weighting semantics;
    /// an all-ones weight vector reproduces [`Fit::fit`] exactly.
    ///
    /// # Errors
    ///
    /// See [`GLMRegressor::fit_with_sample_weight`].
    pub fn fit_with_sample_weight(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
        sample_weight: &Array1<F>,
    ) -> Result<FittedGLMRegressor<F>, FerroError> {
        fit_glm_irls(
            x,
            y,
            sample_weight,
            &GLMFamily::Poisson,
            Link::Log,
            self.alpha,
            self.max_iter,
            self.tol,
            self.fit_intercept,
            self.warm_start,
            self.coef_init.as_ref(),
        )
    }
}

/// Gamma regressor — GLM with Gamma family and log link.
///
/// Suitable for modelling positive continuous data (y > 0).
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
#[derive(Debug, Clone)]
pub struct GammaRegressor<F> {
    /// L2 regularization strength.
    pub alpha: F,
    /// Maximum number of IRLS iterations.
    pub max_iter: usize,
    /// Convergence tolerance.
    pub tol: F,
    /// Whether to fit an intercept.
    pub fit_intercept: bool,
    /// Optimization algorithm requested, mirroring sklearn's `solver` parameter
    /// (`glm.py:140-145`, default `"lbfgs"`).
    ///
    /// ferrolearn fits via IRLS regardless of this value (R-DEV-7); the parameter
    /// is accepted for sklearn API parity (R-DEV-2) and the observable
    /// `coef_` / `intercept_` match sklearn for either value. See [`Solver`].
    pub solver: Solver,
    /// Whether to warm-start from an explicit initial point, mirroring sklearn's
    /// `warm_start` parameter (default `false`, `glm.py:146, :708`). See
    /// [`GLMRegressor::warm_start`] for the R-DEV-2 / R-DEV-7 rationale.
    pub warm_start: bool,
    /// Explicit warm-start initial point `(feature_coefficients, intercept)` — the
    /// ferrolearn analog of sklearn reusing `self.coef_` / `self.intercept_`
    /// (R-DEV-7, `glm.py:244-250`). Consulted only when `warm_start` is `true`. Set
    /// via [`GammaRegressor::with_coef_init`].
    pub coef_init: Option<(Array1<F>, F)>,
}

impl<F: Float + FromPrimitive> GammaRegressor<F> {
    /// Create a new `GammaRegressor` with default settings.
    ///
    /// Defaults: `alpha = 1.0`, `max_iter = 100`, `tol = 1e-4`,
    /// `fit_intercept = true`, `solver = Solver::Lbfgs` (sklearn default).
    #[must_use]
    pub fn new() -> Self {
        Self {
            alpha: F::one(),
            max_iter: 100,
            tol: F::from(1e-4).unwrap_or_else(F::epsilon),
            fit_intercept: true,
            solver: Solver::Lbfgs,
            warm_start: false,
            coef_init: None,
        }
    }

    /// Set the L2 regularization strength.
    #[must_use]
    pub fn with_alpha(mut self, alpha: F) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set the maximum number of IRLS iterations.
    #[must_use]
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the convergence tolerance.
    #[must_use]
    pub fn with_tol(mut self, tol: F) -> Self {
        self.tol = tol;
        self
    }

    /// Set whether to fit an intercept.
    #[must_use]
    pub fn with_fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = fit_intercept;
        self
    }

    /// Set the optimization [`Solver`], mirroring sklearn's `solver` parameter
    /// (`glm.py:140-145`, default `"lbfgs"`).
    ///
    /// ferrolearn fits via IRLS regardless of the value (R-DEV-7); both values
    /// produce the same observable `coef_` / `intercept_` (sklearn API parity,
    /// R-DEV-2).
    #[must_use]
    pub fn with_solver(mut self, solver: Solver) -> Self {
        self.solver = solver;
        self
    }

    /// Enable or disable warm-starting, mirroring sklearn's `warm_start`
    /// parameter (default `false`, `glm.py:146, :708`). See
    /// [`GLMRegressor::with_warm_start`] for the R-DEV-2 / R-DEV-7 rationale; the
    /// warm-start point is supplied explicitly via
    /// [`GammaRegressor::with_coef_init`].
    #[must_use]
    pub fn with_warm_start(mut self, warm_start: bool) -> Self {
        self.warm_start = warm_start;
        self
    }

    /// Set the explicit warm-start initial point `(feature_coefficients,
    /// intercept)` — the ferrolearn analog of sklearn reusing `self.coef_` /
    /// `self.intercept_` (R-DEV-7, `glm.py:244-250`). Only consulted when
    /// `warm_start` is `true`; `coef`'s length must equal the number of features.
    #[must_use]
    pub fn with_coef_init(mut self, coef: Array1<F>, intercept: F) -> Self {
        self.coef_init = Some((coef, intercept));
        self
    }
}

impl<F: Float + FromPrimitive> Default for GammaRegressor<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Float + Send + Sync + ScalarOperand + FromPrimitive + 'static> GammaRegressor<F> {
    /// Fit the Gamma GLM via IRLS with per-sample weights `sample_weight`.
    ///
    /// Mirrors sklearn's `GammaRegressor.fit(X, y, sample_weight)`
    /// (`glm.py:170`, `:229-242`). See
    /// [`GLMRegressor::fit_with_sample_weight`] for the weighting semantics;
    /// an all-ones weight vector reproduces [`Fit::fit`] exactly.
    ///
    /// # Errors
    ///
    /// See [`GLMRegressor::fit_with_sample_weight`].
    pub fn fit_with_sample_weight(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
        sample_weight: &Array1<F>,
    ) -> Result<FittedGLMRegressor<F>, FerroError> {
        fit_glm_irls(
            x,
            y,
            sample_weight,
            &GLMFamily::Gamma,
            Link::Log,
            self.alpha,
            self.max_iter,
            self.tol,
            self.fit_intercept,
            self.warm_start,
            self.coef_init.as_ref(),
        )
    }
}

/// Tweedie regressor — GLM with Tweedie family and log link.
///
/// The `power` parameter controls the variance-mean relationship:
/// V(mu) = mu^power.
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
#[derive(Debug, Clone)]
pub struct TweedieRegressor<F> {
    /// Tweedie power parameter.
    pub power: f64,
    /// Link-function configuration (`Auto`/`Log`/`Identity`).
    ///
    /// `Auto` (the default) resolves to the identity link for `power <= 0`
    /// (Normal) and the log link for `power > 0`, matching sklearn's
    /// `link='auto'` (`glm.py:889-893`).
    pub link: LinkConfig,
    /// L2 regularization strength.
    pub alpha: F,
    /// Maximum number of IRLS iterations.
    pub max_iter: usize,
    /// Convergence tolerance.
    pub tol: F,
    /// Whether to fit an intercept.
    pub fit_intercept: bool,
    /// Optimization algorithm requested, mirroring sklearn's `solver` parameter
    /// (`glm.py:140-145`, default `"lbfgs"`).
    ///
    /// ferrolearn fits via IRLS regardless of this value (R-DEV-7); the parameter
    /// is accepted for sklearn API parity (R-DEV-2) and the observable
    /// `coef_` / `intercept_` match sklearn for either value. See [`Solver`].
    pub solver: Solver,
    /// Whether to warm-start from an explicit initial point, mirroring sklearn's
    /// `warm_start` parameter (default `false`, `glm.py:146, :874`). See
    /// [`GLMRegressor::warm_start`] for the R-DEV-2 / R-DEV-7 rationale.
    pub warm_start: bool,
    /// Explicit warm-start initial point `(feature_coefficients, intercept)` — the
    /// ferrolearn analog of sklearn reusing `self.coef_` / `self.intercept_`
    /// (R-DEV-7, `glm.py:244-250`). Consulted only when `warm_start` is `true`. Set
    /// via [`TweedieRegressor::with_coef_init`].
    pub coef_init: Option<(Array1<F>, F)>,
}

impl<F: Float + FromPrimitive> TweedieRegressor<F> {
    /// Create a new `TweedieRegressor` with default settings.
    ///
    /// Defaults match sklearn's `TweedieRegressor.__init__` (`glm.py:864-887`):
    /// `power = 0.0` (Normal), `link = LinkConfig::Auto`, `alpha = 1.0`,
    /// `max_iter = 100`, `tol = 1e-4`, `fit_intercept = true`,
    /// `solver = Solver::Lbfgs` (sklearn default). With the default
    /// `power = 0.0` and `Auto` link, the model is Normal/identity-link (OLS).
    #[must_use]
    pub fn new() -> Self {
        Self {
            power: 0.0,
            link: LinkConfig::Auto,
            alpha: F::one(),
            max_iter: 100,
            tol: F::from(1e-4).unwrap_or_else(F::epsilon),
            fit_intercept: true,
            solver: Solver::Lbfgs,
            warm_start: false,
            coef_init: None,
        }
    }

    /// Set the Tweedie power parameter.
    #[must_use]
    pub fn with_power(mut self, power: f64) -> Self {
        self.power = power;
        self
    }

    /// Set the link-function configuration (`Auto`/`Log`/`Identity`).
    ///
    /// Mirrors sklearn's `link={'auto','identity','log'}` (`glm.py:861`).
    #[must_use]
    pub fn with_link(mut self, link: LinkConfig) -> Self {
        self.link = link;
        self
    }

    /// Set the L2 regularization strength.
    #[must_use]
    pub fn with_alpha(mut self, alpha: F) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set the maximum number of IRLS iterations.
    #[must_use]
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the convergence tolerance.
    #[must_use]
    pub fn with_tol(mut self, tol: F) -> Self {
        self.tol = tol;
        self
    }

    /// Set whether to fit an intercept.
    #[must_use]
    pub fn with_fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = fit_intercept;
        self
    }

    /// Set the optimization [`Solver`], mirroring sklearn's `solver` parameter
    /// (`glm.py:140-145`, default `"lbfgs"`).
    ///
    /// ferrolearn fits via IRLS regardless of the value (R-DEV-7); both values
    /// produce the same observable `coef_` / `intercept_` (sklearn API parity,
    /// R-DEV-2).
    #[must_use]
    pub fn with_solver(mut self, solver: Solver) -> Self {
        self.solver = solver;
        self
    }

    /// Enable or disable warm-starting, mirroring sklearn's `warm_start`
    /// parameter (default `false`, `glm.py:146, :874`). See
    /// [`GLMRegressor::with_warm_start`] for the R-DEV-2 / R-DEV-7 rationale; the
    /// warm-start point is supplied explicitly via
    /// [`TweedieRegressor::with_coef_init`].
    #[must_use]
    pub fn with_warm_start(mut self, warm_start: bool) -> Self {
        self.warm_start = warm_start;
        self
    }

    /// Set the explicit warm-start initial point `(feature_coefficients,
    /// intercept)` — the ferrolearn analog of sklearn reusing `self.coef_` /
    /// `self.intercept_` (R-DEV-7, `glm.py:244-250`). Only consulted when
    /// `warm_start` is `true`; `coef`'s length must equal the number of features.
    #[must_use]
    pub fn with_coef_init(mut self, coef: Array1<F>, intercept: F) -> Self {
        self.coef_init = Some((coef, intercept));
        self
    }
}

impl<F: Float + FromPrimitive> Default for TweedieRegressor<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Float + Send + Sync + ScalarOperand + FromPrimitive + 'static> TweedieRegressor<F> {
    /// Fit the Tweedie GLM via IRLS with per-sample weights `sample_weight`.
    ///
    /// Mirrors sklearn's `TweedieRegressor.fit(X, y, sample_weight)`
    /// (`glm.py:170`, `:229-242`). The link is resolved from `link`/`power`
    /// as in [`Fit::fit`] (`glm.py:889-903`); see
    /// [`GLMRegressor::fit_with_sample_weight`] for the weighting semantics.
    /// An all-ones weight vector reproduces [`Fit::fit`] exactly.
    ///
    /// # Errors
    ///
    /// See [`GLMRegressor::fit_with_sample_weight`].
    pub fn fit_with_sample_weight(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
        sample_weight: &Array1<F>,
    ) -> Result<FittedGLMRegressor<F>, FerroError> {
        let link = self.link.resolve(self.power);
        fit_glm_irls(
            x,
            y,
            sample_weight,
            &GLMFamily::Tweedie(self.power),
            link,
            self.alpha,
            self.max_iter,
            self.tol,
            self.fit_intercept,
            self.warm_start,
            self.coef_init.as_ref(),
        )
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Cholesky solve for `A x = b`.
fn cholesky_solve<F: Float>(a: &Array2<F>, b: &Array1<F>) -> Result<Array1<F>, FerroError> {
    let n = a.nrows();
    let mut l = Array2::<F>::zeros((n, n));

    for i in 0..n {
        for j in 0..=i {
            let mut s = a[[i, j]];
            for k in 0..j {
                s = s - l[[i, k]] * l[[j, k]];
            }
            if i == j {
                if s <= F::zero() {
                    return Err(FerroError::NumericalInstability {
                        message: "Cholesky: matrix not positive definite".into(),
                    });
                }
                l[[i, j]] = s.sqrt();
            } else {
                l[[i, j]] = s / l[[j, j]];
            }
        }
    }

    let mut z = Array1::<F>::zeros(n);
    for i in 0..n {
        let mut s = b[i];
        for k in 0..i {
            s = s - l[[i, k]] * z[k];
        }
        z[i] = s / l[[i, i]];
    }

    let mut x_sol = Array1::<F>::zeros(n);
    for i in (0..n).rev() {
        let mut s = z[i];
        for k in (i + 1)..n {
            s = s - l[[k, i]] * x_sol[k];
        }
        x_sol[i] = s / l[[i, i]];
    }

    Ok(x_sol)
}

/// Gaussian elimination with partial pivoting.
fn gaussian_solve<F: Float>(
    n: usize,
    a: &Array2<F>,
    b: &Array1<F>,
) -> Result<Array1<F>, FerroError> {
    let mut aug = Array2::<F>::zeros((n, n + 1));
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = a[[i, j]];
        }
        aug[[i, n]] = b[i];
    }

    for col in 0..n {
        let mut max_val = aug[[col, col]].abs();
        let mut max_row = col;
        for row in (col + 1)..n {
            let v = aug[[row, col]].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }

        if max_val < F::from(1e-12).unwrap_or_else(F::epsilon) {
            return Err(FerroError::NumericalInstability {
                message: "singular matrix in Gaussian elimination".into(),
            });
        }

        if max_row != col {
            for j in 0..=n {
                let tmp = aug[[col, j]];
                aug[[col, j]] = aug[[max_row, j]];
                aug[[max_row, j]] = tmp;
            }
        }

        let pivot = aug[[col, col]];
        for row in (col + 1)..n {
            let factor = aug[[row, col]] / pivot;
            for j in col..=n {
                let above = aug[[col, j]];
                aug[[row, j]] = aug[[row, j]] - factor * above;
            }
        }
    }

    let mut x_sol = Array1::<F>::zeros(n);
    for i in (0..n).rev() {
        let mut s = aug[[i, n]];
        for j in (i + 1)..n {
            s = s - aug[[i, j]] * x_sol[j];
        }
        if aug[[i, i]].abs() < F::from(1e-12).unwrap_or_else(F::epsilon) {
            return Err(FerroError::NumericalInstability {
                message: "near-zero pivot in back substitution".into(),
            });
        }
        x_sol[i] = s / aug[[i, i]];
    }

    Ok(x_sol)
}

/// Solve the weighted ridge system `(X^T W X + P) w = X^T W z`, where the
/// penalty matrix `P` adds the L2 regularization to the diagonal of the
/// feature columns only.
///
/// # Penalty scaling and the intercept (sklearn parity, `glm.py:229-258`)
///
/// scikit-learn minimizes the per-sample-MEAN half-deviance plus an L2 prior on
/// the feature coefficients (NOT the intercept):
///
/// ```text
/// J(w) = 1/(2*S) * sum_i s_i * deviance_i + 1/2 * alpha * ||coef||^2,
/// ```
///
/// with `S = sum_i s_i` (= `n_samples` for unweighted fits). Its stationarity
/// condition is `(1/S) * grad[sum 1/2 dev] + alpha * w_features = 0`.
///
/// The IRLS normal equations `X^T W X w = X^T W z` are the linearization of the
/// SUMMED half-deviance `sum_i 1/2 dev_i` (no `1/S` factor): `X^T W X` is the
/// summed-scale Hessian. To make those summed equations correspond to sklearn's
/// mean-scale objective we multiply the penalty by `S` (the sum of weights, =
/// `n_samples` unweighted) before adding it to the diagonal: the added penalty
/// is `weight_sum * alpha`, applied to feature columns only, leaving the
/// intercept (column 0 of the augmented design, when present) unpenalized.
///
/// `weight_sum` is `S = sum_i s_i`, the sum of the GLM `sample_weight`
/// (= `n_samples` for an all-ones / unweighted fit); `intercept_col` is
/// `Some(0)` when an intercept column was prepended to `x`, `None` otherwise.
fn weighted_ridge_solve<F: Float + FromPrimitive>(
    x: &Array2<F>,
    z: &Array1<F>,
    weights: &Array1<F>,
    alpha: F,
    weight_sum: F,
    intercept_col: Option<usize>,
) -> Result<Array1<F>, FerroError> {
    let (n_samples, n_features) = x.dim();

    let mut xtwx = Array2::<F>::zeros((n_features, n_features));
    let mut xtwz = Array1::<F>::zeros(n_features);

    for i in 0..n_samples {
        let wi = weights[i];
        let xi = x.row(i);
        for r in 0..n_features {
            xtwz[r] = xtwz[r] + wi * xi[r] * z[i];
            for c in 0..n_features {
                xtwx[[r, c]] = xtwx[[r, c]] + wi * xi[r] * xi[c];
            }
        }
    }

    // Add L2 regularization. The IRLS normal equations are at the SUMMED-deviance
    // scale, so to match sklearn's MEAN-deviance objective the diagonal penalty is
    // `weight_sum * alpha` (glm.py:229-242). The intercept column is excluded:
    // sklearn's `l2_reg_strength = self.alpha` weights only `||coef||^2`
    // (glm.py:258), never the intercept.
    let penalty = weight_sum * alpha;
    for i in 0..n_features {
        if Some(i) == intercept_col {
            continue;
        }
        xtwx[[i, i]] = xtwx[[i, i]] + penalty;
    }

    cholesky_solve(&xtwx, &xtwz).or_else(|_| gaussian_solve(n_features, &xtwx, &xtwz))
}

/// Core IRLS fitting logic shared by all GLM variants.
///
/// The IRLS update is parameterized by the [`Link`]: with linear predictor
/// `eta = X @ coef`, mean `mu = link.inverse(eta)` and link derivative
/// `dmu/deta`, the standard Fisher-scoring working weight and response are
/// `w = (dmu/deta)^2 / V(mu)` and `z = eta + (y - mu) / (dmu/deta)`
/// (`glm.py:362` for the inverse-link mapping). For [`Link::Log`]
/// (`dmu/deta = mu`) this is `w = mu^2 / V(mu)`, `z = eta + (y - mu)/mu`,
/// byte-identical to the previous log-only code. For [`Link::Identity`] with
/// `V(mu) = mu^0 = 1` (Normal/`power = 0`), `w = 1`, `z = y`, so IRLS reduces to
/// ordinary least squares.
///
/// # Warm start (R-DEV-2 + R-DEV-7)
///
/// sklearn's `warm_start=True` reuses the previous fit's `self.coef_` /
/// `self.intercept_` as the optimizer's starting point (`glm.py:243-254`).
/// ferrolearn's estimators are immutable (`fit(&self, ...)` returns a fresh
/// fitted object and never mutates `self`), so there is no `self.coef_` to reuse
/// across calls; the Rust-idiomatic analog (R-DEV-7) is an EXPLICIT initial point
/// supplied via `coef_init = Some((feature_coef, intercept))`. When
/// `warm_start == true` and `coef_init` is provided, the IRLS coefficient vector
/// (and the derived `eta` / `mu`) are seeded from `coef_init` instead of the
/// cold start (`coef = 0`, `eta = link(y)`). The feature-coefficient length must
/// equal `n_features` (else [`FerroError::ShapeMismatch`]). Otherwise the cold
/// start is kept byte-for-byte.
///
/// Because the penalized GLM objective is convex, the converged `coef_` /
/// `intercept_` are warm-start-INVARIANT: the init only changes the starting
/// point (and so the iteration count), never the optimum — so the warm-started
/// fit matches both the cold-start fit and the sklearn oracle (`glm.py:244-256`
/// reaches the same minimizer regardless of the seed).
#[allow(
    clippy::too_many_arguments,
    reason = "shared IRLS core threads the link and the warm-start init \
    alongside the family/penalty/convergence parameters; splitting into a config \
    struct would obscure the 1:1 mapping to sklearn's fit signature"
)]
fn fit_glm_irls<F: Float + Send + Sync + ScalarOperand + FromPrimitive + 'static>(
    x: &Array2<F>,
    y: &Array1<F>,
    sample_weight: &Array1<F>,
    family: &GLMFamily,
    link: Link,
    alpha: F,
    max_iter: usize,
    tol: F,
    fit_intercept: bool,
    warm_start: bool,
    coef_init: Option<&(Array1<F>, F)>,
) -> Result<FittedGLMRegressor<F>, FerroError> {
    let (n_samples, n_features_orig) = x.dim();

    if n_samples != y.len() {
        return Err(FerroError::ShapeMismatch {
            expected: vec![n_samples],
            actual: vec![y.len()],
            context: "y length must match number of samples in X".into(),
        });
    }

    if sample_weight.len() != n_samples {
        return Err(FerroError::ShapeMismatch {
            expected: vec![n_samples],
            actual: vec![sample_weight.len()],
            context: "sample_weight length must match number of samples in X".into(),
        });
    }

    // Non-finite input validation, mirroring sklearn's
    // `self._validate_data(X, y, ..., y_numeric=True)` (`glm.py:189-196`) which
    // keeps the default `force_all_finite=True`, so `check_array` rejects any
    // NaN or +/-inf in X OR y with a `ValueError` BEFORE the IRLS loop. sklearn
    // also validates `sample_weight` via `_check_sample_weight` (default
    // `force_all_finite=True`, `glm.py:211`). This is the SHARED IRLS entry that
    // every estimator (PoissonRegressor/GammaRegressor/TweedieRegressor/
    // GLMRegressor) routes through, so the check lands ONCE here.
    // `.iter().any(|v| !v.is_finite())` rejects both NaN and Inf (bounds-safe,
    // no panic, R-CODE-2), matching the crate idiom (`ridge.rs`). The finite
    // path is byte-identical (the guard never fires on finite input). The
    // sample_weight finiteness check MUST precede the non-negative check below:
    // sklearn's `_check_sample_weight` runs `check_array(force_all_finite=True)`
    // first, so a `-inf` weight raises the infinity `ValueError` (verified live),
    // NOT a non-negative error. This guard also precedes the per-family y-domain
    // validation so a non-finite y is reported as a finiteness failure
    // (`is_finite()` is `false` for NaN/Inf), matching sklearn's `check_array`
    // running before `in_y_true_range`.
    if x.iter().any(|v| !v.is_finite()) {
        return Err(FerroError::InvalidParameter {
            name: "X".into(),
            reason: "Input X contains NaN or infinity.".into(),
        });
    }
    if y.iter().any(|v| !v.is_finite()) {
        return Err(FerroError::InvalidParameter {
            name: "y".into(),
            reason: "Input y contains NaN or infinity.".into(),
        });
    }
    if sample_weight.iter().any(|v| !v.is_finite()) {
        return Err(FerroError::InvalidParameter {
            name: "sample_weight".into(),
            reason: "Input sample_weight contains NaN or infinity.".into(),
        });
    }

    // sklearn's GLM family does NOT reject finite negative `sample_weight`: it
    // calls `_check_sample_weight(sample_weight, X, dtype=loss_dtype)` WITHOUT
    // `only_non_negative=True` (`glm.py:211`), so a finite negative weight passes
    // validation and reaches the optimizer (verified live: `PoissonRegressor()
    // .fit(X, y, sample_weight=[..,-1.0])` fits, coef `[-0.04268902, 0.24080026]`).
    // Negative weights are mathematically valid for the GLM objective — the
    // deviance is averaged `sum_i s_i * deviance_i` (`glm.py:229-242`), so a
    // negative `s_i` simply NEGATES that sample's contribution. ferrolearn's IRLS
    // uses `sample_weight` LINEARLY (`weights[i] *= sample_weight[i]` feeding the
    // `X^T W X` / `X^T W z` accumulation in `weighted_ridge_solve`, NO `sqrt`), so
    // a negative weight flows through unchanged and reaches the same optimum as
    // sklearn's lbfgs. The non-finite guard above (#2261) still rejects NaN/Inf
    // weights (sklearn's `_check_sample_weight` runs `check_array(force_all_finite
    // =True)` first), matching exception parity for non-finite input.

    if n_samples == 0 {
        return Err(FerroError::InsufficientSamples {
            required: 1,
            actual: 0,
            context: "GLM requires at least one sample".into(),
        });
    }

    if alpha < F::zero() {
        return Err(FerroError::InvalidParameter {
            name: "alpha".into(),
            reason: "must be non-negative".into(),
        });
    }

    // Per-family target-domain validation, mirroring sklearn's
    //   `if not linear_loss.base_loss.in_y_true_range(y): raise ValueError(...)`
    // (`glm.py:221-225`). The valid `y` range is determined by the family's
    // Tweedie `power` (NOT the link — verified against the live oracle):
    //   * `power <= 0`  (Normal/identity)        -> y unconstrained
    //   * `0 < power < 2` (Poisson `power = 1`)  -> y >= 0   (closed at 0)
    //   * `power >= 2`    (Gamma `power = 2`)     -> y > 0    (open at 0)
    // Confirmed live-oracle behaviors: `GammaRegressor().fit(X, [0,1,2])` and
    // `TweedieRegressor(power=2.0).fit(X, [0,1,2])` raise ValueError;
    // `PoissonRegressor().fit(X, [-1,1,2])` raises; `TweedieRegressor(power=1.5)`
    // accepts y == 0.
    let min_y = F::from(1e-10).unwrap_or_else(F::epsilon);
    let y_domain = YDomain::for_power(family.domain_power());
    if y.iter().any(|&yi| !y_domain.contains(yi)) {
        return Err(FerroError::InvalidParameter {
            name: "y".into(),
            reason: format!(
                "Some value(s) of y are out of the valid range of the loss '{}'.",
                y_domain.loss_name()
            ),
        });
    }

    // Build design matrix (optionally prepend intercept column).
    let n_cols = if fit_intercept {
        n_features_orig + 1
    } else {
        n_features_orig
    };

    let mut x_design = Array2::<F>::zeros((n_samples, n_cols));
    if fit_intercept {
        for i in 0..n_samples {
            x_design[[i, 0]] = F::one();
            for j in 0..n_features_orig {
                x_design[[i, j + 1]] = x[[i, j]];
            }
        }
    } else {
        x_design.assign(x);
    }

    // The intercept (column 0 of the augmented design when fitting one) is
    // excluded from the L2 penalty, matching sklearn (glm.py:258).
    let intercept_col = if fit_intercept { Some(0) } else { None };

    // Sum of sample weights `S = sum_i s_i`. For an unweighted GLM every weight
    // is 1, so this is `n_samples` (byte-identical to the previous unweighted
    // path). It scales the L2 penalty so the summed-deviance IRLS normal
    // equations minimize sklearn's mean-deviance objective normalized by
    // `sum(sample_weight)` (glm.py:229-242).
    let weight_sum = sample_weight.iter().fold(F::zero(), |acc, &si| acc + si);

    // For the log link, clamp y away from 0 so `ln(y)` and `mu` stay finite.
    // For the identity link y is used as-is (mu = eta = y, no positivity
    // constraint).
    let y_safe: Array1<F> = match link {
        Link::Log => y.mapv(|v| if v < min_y { min_y } else { v }),
        Link::Identity => y.clone(),
    };

    // Initialise mu = y_safe, eta = g(mu) = link(mu): log link → ln(y),
    // identity link → y.
    let mut mu: Array1<F> = y_safe.clone();
    let mut eta: Array1<F> = match link {
        Link::Log => y_safe.mapv(|v| v.ln()),
        Link::Identity => y_safe.clone(),
    };

    // Coefficient initialization.
    //
    // COLD START (default, `warm_start == false` or no `coef_init`): feature
    // coefficients are 0; when `fit_intercept` the intercept entry is seeded at
    // `link.link(weighted_mean(y))` (REQ-5, `glm.py:251-256`), and `eta`/`mu`
    // are recomputed from that seed (so `eta == link(mean_w(y))` constant). When
    // `fit_intercept` is false the intercept seed is skipped and `eta`/`mu` keep
    // the per-sample `link(y)` seed above. The objective is convex, so this
    // changes only the starting point (and the iteration count), never the
    // converged optimum — the cold fit still matches the sklearn oracle.
    //
    // WARM START (R-DEV-7 analog of sklearn's `warm_start=True`,
    // `glm.py:243-254`): seed the IRLS coefficient vector from the explicit
    // `coef_init = (feature_coef, intercept)` instead, then recompute
    // `eta = X_design @ coef` and `mu = link.inverse(eta)` so the first IRLS
    // weights/working-response are formed at the supplied point (sklearn likewise
    // hands `coef` straight to the optimizer). When `warm_start && coef_init`,
    // the explicit init takes precedence over the REQ-5 intercept seed.
    let mut coef = Array1::<F>::zeros(n_cols);
    if warm_start && let Some((feature_coef, intercept_init)) = coef_init {
        if feature_coef.len() != n_features_orig {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_features_orig],
                actual: vec![feature_coef.len()],
                context: "coef_init feature-coefficient length must match \
                          number of features in X"
                    .into(),
            });
        }
        // Place the supplied coefficients into the (optionally
        // intercept-augmented) design layout: column 0 is the intercept when
        // `fit_intercept`, features follow. When `fit_intercept == false` the
        // supplied intercept is ignored (the model has no intercept term),
        // matching sklearn's `coef = self.coef_` (no intercept) branch
        // (`glm.py:248-249`).
        if fit_intercept {
            coef[0] = *intercept_init;
            for (j, &cj) in feature_coef.iter().enumerate() {
                coef[j + 1] = cj;
            }
        } else {
            for (j, &cj) in feature_coef.iter().enumerate() {
                coef[j] = cj;
            }
        }
        // Recompute eta/mu at the warm-start point so the first IRLS step is
        // formed there (link.inverse with the same clamps as the loop body).
        eta = x_design.dot(&coef);
        match link {
            Link::Log => {
                let hi = F::from(20.0).unwrap_or_else(F::max_value);
                let lo = F::zero() - hi;
                let mmu = F::from(1e-10).unwrap_or_else(F::epsilon);
                let xmu = F::from(1e10).unwrap_or_else(F::max_value);
                for i in 0..n_samples {
                    let eta_i = eta[i].max(lo).min(hi);
                    eta[i] = eta_i;
                    mu[i] = link.inverse(eta_i).max(mmu).min(xmu);
                }
            }
            Link::Identity => {
                for i in 0..n_samples {
                    mu[i] = link.inverse(eta[i]);
                }
            }
        }
    } else if fit_intercept {
        // COLD-START INTERCEPT INITIALIZATION (REQ-5, `glm.py:251-256`).
        //
        // sklearn cold-starts with `coef = init_zero_coef(X)` and, when
        // `fit_intercept`, seeds the intercept entry at
        //   `coef[-1] = link.link(np.average(y, weights=sample_weight))`
        // (the feature coefficients stay 0). For the log link this is
        // `intercept_init = log(weighted_mean(y))`; for the identity link it is
        // `weighted_mean(y)`. We mirror this exactly: the intercept is design
        // column 0, so `coef[0] = link.link(weighted_mean(y))`.
        //
        // `weighted_mean(y) = Σ(s_i·y_i) / Σ(s_i)` (= plain mean for an all-ones
        // sample_weight). `weight_sum = Σ s_i` was computed above.
        //
        // Edge case (R-CODE-2, R-DEV-1): the seed must lie in the link's domain.
        // For the log link `weighted_mean(y)` can be 0 (e.g. all-zero Poisson
        // `y`, which is IN the Poisson domain `y >= 0`), where `log(0) = -inf` —
        // sklearn likewise computes `link.link(0) = -inf` there (`glm.py:254`)
        // and its lbfgs optimum STAYS at `-inf`: the penalized data deviance is
        // minimized as `mu -> 0` (`eta -> -inf`), and the L2 penalty is on the
        // feature coefficients only (`l2_reg_strength = self.alpha`,
        // `glm.py:258`), which are 0. So sklearn returns `intercept_ = -inf`,
        // `coef_ = 0`, and `predict = inverse_link(X @ 0 - inf) = exp(-inf) =
        // 0.0` EXACTLY for every sample. We short-circuit to that same degenerate
        // optimum: IRLS cannot iterate from `mu = 0` (the Fisher weights blow up),
        // so we skip the loop entirely and return `coef_ = 0`, `intercept_ =
        // link(weighted_mean(y))` (the non-finite seed). This is alpha-INVARIANT
        // (the penalty on coef is 0 since coef = 0; the deviance drives
        // `eta -> -inf` regardless of alpha) and fires ONLY for a non-finite seed
        // (log link + `mean(y) == 0`) — the exact all-zero case. Gamma requires
        // `y > 0` so `mean(y) > 0` always (domain guard rejects `y == 0`); the
        // identity link gives `link(0) = 0` (finite), so neither hits this path.
        let weighted_y_sum = y
            .iter()
            .zip(sample_weight.iter())
            .fold(F::zero(), |acc, (&yi, &si)| acc + si * yi);
        let denom = if weight_sum > F::zero() {
            weight_sum
        } else {
            F::from(n_samples).unwrap_or_else(F::one)
        };
        let weighted_mean_y = weighted_y_sum / denom;
        let intercept_init = link.link(weighted_mean_y);

        if !intercept_init.is_finite() {
            // Degenerate optimum (all-zero-y log-link case): short-circuit to
            // sklearn's lbfgs landing point `coef_ = 0`, `intercept_ = -inf`
            // (`glm.py:254`), skipping IRLS (which cannot iterate from `mu = 0`).
            // `predict = exp(-inf) = 0.0` then matches sklearn exactly. The
            // feature coefficients are already 0 (cold start); set the intercept
            // column to the non-finite seed and return immediately. `n_iter_ = 0`
            // records that the optimum was reached without any IRLS iteration.
            coef[0] = intercept_init;
            let intercept = coef[0];
            let coefficients = Array1::from_iter(coef.iter().skip(1).copied());
            return Ok(FittedGLMRegressor {
                coefficients,
                intercept,
                link,
                family: *family,
                n_iter: 0,
            });
        }

        {
            coef[0] = intercept_init;
            // Recompute eta/mu from the seeded coef (feature coefs are 0, so
            // `eta = intercept_init` for every sample, i.e. `eta = link(mean y)`).
            eta = x_design.dot(&coef);
            match link {
                Link::Log => {
                    let hi = F::from(20.0).unwrap_or_else(F::max_value);
                    let lo = F::zero() - hi;
                    let mmu = F::from(1e-10).unwrap_or_else(F::epsilon);
                    let xmu = F::from(1e10).unwrap_or_else(F::max_value);
                    for i in 0..n_samples {
                        let eta_i = eta[i].max(lo).min(hi);
                        eta[i] = eta_i;
                        mu[i] = link.inverse(eta_i).max(mmu).min(xmu);
                    }
                }
                Link::Identity => {
                    for i in 0..n_samples {
                        mu[i] = link.inverse(eta[i]);
                    }
                }
            }
        }
        // (The non-finite-seed branch above returns early, so there is no
        // finite-init fall-through here.)
    }

    let min_mu = F::from(1e-10).unwrap_or_else(F::epsilon);
    let max_mu = F::from(1e10).unwrap_or_else(F::max_value);

    // Count the IRLS iterations actually run (sklearn's `n_iter_`, `glm.py:283`).
    // At least one iteration always runs (`max_iter >= 1`); on convergence we
    // break after the iteration that satisfied the tolerance.
    let mut n_iter = 0usize;
    for _iter in 0..max_iter {
        n_iter += 1;
        let coef_old = coef.clone();

        // Compute IRLS weights and working response.
        let mut weights = Array1::<F>::zeros(n_samples);
        let mut z = Array1::<F>::zeros(n_samples);

        for i in 0..n_samples {
            // IRLS (Fisher scoring) with the configured link:
            //   dmu/deta  : Log => mu, Identity => 1
            //   weight w  = (dmu/deta)^2 / V(mu)
            //   response z = eta + (y - mu) / (dmu/deta)
            // For Log this is `w = mu^2/V(mu)`, `z = eta + (y - mu)/mu`,
            // byte-identical to the previous log-only code (clamped `mu_i`
            // throughout). For Identity + power=0 (V=1): w=1, z=y => OLS.
            match link {
                Link::Log => {
                    let mu_i = mu[i].max(min_mu).min(max_mu);
                    let var_i = family.variance(mu_i).max(min_mu);
                    let g_prime = F::one() / mu_i; // derivative of log link
                    z[i] = eta[i] + (y_safe[i] - mu_i) * g_prime;
                    weights[i] = F::one() / (g_prime * g_prime * var_i);
                }
                Link::Identity => {
                    let mu_i = mu[i];
                    // V(mu) for the identity link can see mu <= 0 (eta is
                    // unbounded); for power=0, V(mu)=mu^0=1 always. Clamp the
                    // magnitude for non-zero powers so V stays finite/positive.
                    let var_i = family.variance(mu_i.abs().max(min_mu)).max(min_mu);
                    let dmu_deta = link.dmu_deta(mu_i); // = 1
                    z[i] = eta[i] + (y_safe[i] - mu_i) / dmu_deta;
                    weights[i] = dmu_deta * dmu_deta / var_i;
                }
            }
            // Clamp the Fisher (GLM) working weight.
            if weights[i] < min_mu {
                weights[i] = min_mu;
            }
            // Apply the per-sample weight `s_i`: the `W` diagonal entry for
            // sample i becomes `s_i * w_irls,i` (standard weighted IRLS).
            // sklearn weights the deviance average by `sample_weight`
            // (glm.py:229-242); the working response `z_i` is unchanged. For
            // all-ones weights this is a no-op (byte-identical unweighted path).
            weights[i] = weights[i] * sample_weight[i];
        }

        // Solve weighted ridge. `weights` now carry `s_i * w_irls,i`;
        // `weight_sum = S = sum_i s_i` (= n_samples for an all-ones fit). The
        // penalty is scaled by `S` so the summed-deviance normal equations
        // minimize sklearn's sample-weight-averaged deviance objective
        // (glm.py:229-242). The intercept column (column 0 of the
        // augmented design, present iff `fit_intercept`) is left unpenalized
        // (glm.py:258, `l2_reg_strength = self.alpha` weighs only `||coef||^2`).
        coef = weighted_ridge_solve(&x_design, &z, &weights, alpha, weight_sum, intercept_col)?;

        // Update eta = X @ coef and mu = link.inverse(eta).
        eta = x_design.dot(&coef);
        match link {
            Link::Log => {
                let hi = F::from(20.0).unwrap_or_else(F::max_value);
                let lo = F::zero() - hi;
                for i in 0..n_samples {
                    // Clamp eta to prevent overflow in exp.
                    let eta_i = eta[i].max(lo).min(hi);
                    eta[i] = eta_i;
                    mu[i] = link.inverse(eta_i).max(min_mu).min(max_mu);
                }
            }
            Link::Identity => {
                // Identity link: eta is unbounded; mu = eta (no exp clamp).
                for i in 0..n_samples {
                    mu[i] = link.inverse(eta[i]);
                }
            }
        }

        // Check convergence.
        let max_change = coef
            .iter()
            .zip(coef_old.iter())
            .map(|(&c, &co)| (c - co).abs())
            .fold(F::zero(), |a, b| if b > a { b } else { a });

        if max_change < tol {
            break;
        }
    }

    // Extract intercept and feature coefficients.
    let (intercept, coefficients) = if fit_intercept {
        let intercept = coef[0];
        let coefficients = Array1::from_iter(coef.iter().skip(1).copied());
        (intercept, coefficients)
    } else {
        (F::zero(), coef)
    };

    Ok(FittedGLMRegressor {
        coefficients,
        intercept,
        link,
        family: *family,
        n_iter,
    })
}

// ---------------------------------------------------------------------------
// Fit — GLMRegressor
// ---------------------------------------------------------------------------

impl<F: Float + Send + Sync + ScalarOperand + FromPrimitive + 'static> Fit<Array2<F>, Array1<F>>
    for GLMRegressor<F>
{
    type Fitted = FittedGLMRegressor<F>;
    type Error = FerroError;

    /// Fit the GLM via IRLS.
    ///
    /// # Errors
    ///
    /// - [`FerroError::ShapeMismatch`] — sample count mismatch.
    /// - [`FerroError::InsufficientSamples`] — zero samples.
    /// - [`FerroError::InvalidParameter`] — negative alpha or negative y.
    fn fit(&self, x: &Array2<F>, y: &Array1<F>) -> Result<FittedGLMRegressor<F>, FerroError> {
        // GLMRegressor's families are all log-link (Poisson/Gamma/Tweedie>0).
        // The Tweedie identity link is exposed only through TweedieRegressor's
        // `link` configuration (sklearn similarly only exposes `link` on
        // `TweedieRegressor`, `glm.py:861`). The default (no sample_weight) path
        // delegates with an all-ones weight vector, matching sklearn's
        // `_check_sample_weight` default (`glm.py:208-211`); `weight_sum` then
        // equals `n_samples`, so this is byte-identical to the unweighted fit.
        self.fit_with_sample_weight(x, y, &Array1::ones(x.nrows()))
    }
}

// ---------------------------------------------------------------------------
// Fit — PoissonRegressor
// ---------------------------------------------------------------------------

impl<F: Float + Send + Sync + ScalarOperand + FromPrimitive + 'static> Fit<Array2<F>, Array1<F>>
    for PoissonRegressor<F>
{
    type Fitted = FittedGLMRegressor<F>;
    type Error = FerroError;

    /// Fit the Poisson GLM via IRLS.
    ///
    /// # Errors
    ///
    /// See [`GLMRegressor::fit`].
    fn fit(&self, x: &Array2<F>, y: &Array1<F>) -> Result<FittedGLMRegressor<F>, FerroError> {
        // Poisson uses the log link only (`HalfPoissonLoss`, `glm.py:589-590`).
        // Default (no sample_weight) path: all-ones weights => byte-identical to
        // the unweighted fit (`weight_sum = n_samples`).
        self.fit_with_sample_weight(x, y, &Array1::ones(x.nrows()))
    }
}

// ---------------------------------------------------------------------------
// Fit — GammaRegressor
// ---------------------------------------------------------------------------

impl<F: Float + Send + Sync + ScalarOperand + FromPrimitive + 'static> Fit<Array2<F>, Array1<F>>
    for GammaRegressor<F>
{
    type Fitted = FittedGLMRegressor<F>;
    type Error = FerroError;

    /// Fit the Gamma GLM via IRLS.
    ///
    /// # Errors
    ///
    /// See [`GLMRegressor::fit`].
    fn fit(&self, x: &Array2<F>, y: &Array1<F>) -> Result<FittedGLMRegressor<F>, FerroError> {
        // Gamma uses the log link only (`HalfGammaLoss`, `glm.py:721-722`).
        // Default (no sample_weight) path: all-ones weights => byte-identical to
        // the unweighted fit (`weight_sum = n_samples`).
        self.fit_with_sample_weight(x, y, &Array1::ones(x.nrows()))
    }
}

// ---------------------------------------------------------------------------
// Fit — TweedieRegressor
// ---------------------------------------------------------------------------

impl<F: Float + Send + Sync + ScalarOperand + FromPrimitive + 'static> Fit<Array2<F>, Array1<F>>
    for TweedieRegressor<F>
{
    type Fitted = FittedGLMRegressor<F>;
    type Error = FerroError;

    /// Fit the Tweedie GLM via IRLS.
    ///
    /// # Errors
    ///
    /// See [`GLMRegressor::fit`].
    fn fit(&self, x: &Array2<F>, y: &Array1<F>) -> Result<FittedGLMRegressor<F>, FerroError> {
        // Resolve the link from the configuration and Tweedie power, mirroring
        // `TweedieRegressor._get_loss` (`glm.py:889-903`): `auto` selects
        // identity for `power <= 0` (Normal/OLS) and log for `power > 0`.
        // Default (no sample_weight) path: all-ones weights => byte-identical to
        // the unweighted fit (`weight_sum = n_samples`).
        self.fit_with_sample_weight(x, y, &Array1::ones(x.nrows()))
    }
}

// ---------------------------------------------------------------------------
// Predict / HasCoefficients / Pipeline — FittedGLMRegressor
// ---------------------------------------------------------------------------

impl<F: Float + Send + Sync + ScalarOperand + 'static> Predict<Array2<F>>
    for FittedGLMRegressor<F>
{
    type Output = Array1<F>;
    type Error = FerroError;

    /// Predict using the fitted GLM.
    ///
    /// Computes `link.inverse(X @ coefficients + intercept)` (`glm.py:362`):
    /// `exp(...)` for a [`Link::Log`] model (Poisson/Gamma/Tweedie with
    /// `power > 0`), and the raw linear predictor `X @ coef + intercept` for a
    /// [`Link::Identity`] model (Tweedie with `power <= 0`, Normal).
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of features
    /// does not match the fitted model.
    fn predict(&self, x: &Array2<F>) -> Result<Array1<F>, FerroError> {
        if x.ncols() != self.coefficients.len() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![self.coefficients.len()],
                actual: vec![x.ncols()],
                context: "number of features must match fitted model".into(),
            });
        }
        let eta = x.dot(&self.coefficients) + self.intercept;
        let link = self.link;
        Ok(eta.mapv(|v| link.inverse(v)))
    }
}

impl<F: Float + Send + Sync + ScalarOperand + 'static> HasCoefficients<F>
    for FittedGLMRegressor<F>
{
    fn coefficients(&self) -> &Array1<F> {
        &self.coefficients
    }

    fn intercept(&self) -> F {
        self.intercept
    }
}

// Pipeline integration for GLMRegressor.
impl<F> PipelineEstimator<F> for GLMRegressor<F>
where
    F: Float + FromPrimitive + ScalarOperand + Send + Sync + 'static,
{
    fn fit_pipeline(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
    ) -> Result<Box<dyn FittedPipelineEstimator<F>>, FerroError> {
        let fitted = self.fit(x, y)?;
        Ok(Box::new(fitted))
    }
}

impl<F> FittedPipelineEstimator<F> for FittedGLMRegressor<F>
where
    F: Float + ScalarOperand + Send + Sync + 'static,
{
    fn predict_pipeline(&self, x: &Array2<F>) -> Result<Array1<F>, FerroError> {
        self.predict(x)
    }
}

// Pipeline integration for PoissonRegressor.
impl<F> PipelineEstimator<F> for PoissonRegressor<F>
where
    F: Float + FromPrimitive + ScalarOperand + Send + Sync + 'static,
{
    fn fit_pipeline(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
    ) -> Result<Box<dyn FittedPipelineEstimator<F>>, FerroError> {
        let fitted = self.fit(x, y)?;
        Ok(Box::new(fitted))
    }
}

// Pipeline integration for GammaRegressor.
impl<F> PipelineEstimator<F> for GammaRegressor<F>
where
    F: Float + FromPrimitive + ScalarOperand + Send + Sync + 'static,
{
    fn fit_pipeline(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
    ) -> Result<Box<dyn FittedPipelineEstimator<F>>, FerroError> {
        let fitted = self.fit(x, y)?;
        Ok(Box::new(fitted))
    }
}

// Pipeline integration for TweedieRegressor.
impl<F> PipelineEstimator<F> for TweedieRegressor<F>
where
    F: Float + FromPrimitive + ScalarOperand + Send + Sync + 'static,
{
    fn fit_pipeline(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
    ) -> Result<Box<dyn FittedPipelineEstimator<F>>, FerroError> {
        let fitted = self.fit(x, y)?;
        Ok(Box::new(fitted))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    // ---- GLMRegressor ----

    #[test]
    fn test_glm_poisson_defaults() {
        let m = GLMRegressor::<f64>::new(GLMFamily::Poisson);
        assert_relative_eq!(m.alpha, 1.0);
        assert_eq!(m.max_iter, 100);
        assert!(m.fit_intercept);
    }

    #[test]
    fn test_glm_builder() {
        let m = GLMRegressor::<f64>::new(GLMFamily::Gamma)
            .with_alpha(0.5)
            .with_max_iter(200)
            .with_tol(1e-6)
            .with_fit_intercept(false);
        assert_relative_eq!(m.alpha, 0.5);
        assert_eq!(m.max_iter, 200);
        assert!(!m.fit_intercept);
    }

    #[test]
    fn test_glm_shape_mismatch() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![1.0, 2.0];
        assert!(
            GLMRegressor::<f64>::new(GLMFamily::Poisson)
                .fit(&x, &y)
                .is_err()
        );
    }

    #[test]
    fn test_glm_negative_alpha() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![1.0, 2.0, 3.0];
        assert!(
            GLMRegressor::<f64>::new(GLMFamily::Poisson)
                .with_alpha(-1.0)
                .fit(&x, &y)
                .is_err()
        );
    }

    #[test]
    fn test_glm_poisson_fit_predict() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![2.0, 5.0, 10.0, 20.0];

        let fitted = GLMRegressor::<f64>::new(GLMFamily::Poisson)
            .with_alpha(0.0)
            .with_max_iter(200)
            .fit(&x, &y)
            .unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 4);
        // Predictions should be positive.
        for &p in preds.iter() {
            assert!(p > 0.0);
        }
    }

    #[test]
    fn test_glm_gamma_fit_predict() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![2.0, 5.0, 10.0, 20.0];

        let fitted = GLMRegressor::<f64>::new(GLMFamily::Gamma)
            .with_alpha(0.0)
            .with_max_iter(200)
            .fit(&x, &y)
            .unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 4);
        for &p in preds.iter() {
            assert!(p > 0.0);
        }
    }

    #[test]
    fn test_glm_tweedie_fit_predict() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![2.0, 5.0, 10.0, 20.0];

        let fitted = GLMRegressor::<f64>::new(GLMFamily::Tweedie(1.5))
            .with_alpha(0.0)
            .with_max_iter(200)
            .fit(&x, &y)
            .unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 4);
        for &p in preds.iter() {
            assert!(p > 0.0);
        }
    }

    #[test]
    fn test_glm_predict_feature_mismatch() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0]).unwrap();
        let y = array![1.0, 2.0, 3.0];
        let fitted = GLMRegressor::<f64>::new(GLMFamily::Poisson)
            .fit(&x, &y)
            .unwrap();
        let x_bad = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        assert!(fitted.predict(&x_bad).is_err());
    }

    #[test]
    fn test_glm_has_coefficients() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = array![1.0, 2.0, 3.0];
        let fitted = GLMRegressor::<f64>::new(GLMFamily::Poisson)
            .fit(&x, &y)
            .unwrap();
        assert_eq!(fitted.coefficients().len(), 2);
    }

    #[test]
    fn test_glm_pipeline() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![2.0, 5.0, 10.0, 20.0];
        let model = GLMRegressor::<f64>::new(GLMFamily::Poisson).with_alpha(0.0);
        let fitted = model.fit_pipeline(&x, &y).unwrap();
        let preds = fitted.predict_pipeline(&x).unwrap();
        assert_eq!(preds.len(), 4);
    }

    // ---- PoissonRegressor ----

    #[test]
    fn test_poisson_defaults() {
        let m = PoissonRegressor::<f64>::new();
        assert_relative_eq!(m.alpha, 1.0);
        assert_eq!(m.max_iter, 100);
        assert!(m.fit_intercept);
    }

    #[test]
    fn test_poisson_fit_predict() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![2.0, 5.0, 10.0, 20.0];

        let fitted = PoissonRegressor::<f64>::new()
            .with_alpha(0.0)
            .with_max_iter(200)
            .fit(&x, &y)
            .unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 4);
        for &p in preds.iter() {
            assert!(p > 0.0);
        }
    }

    #[test]
    fn test_poisson_pipeline() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![2.0, 5.0, 10.0, 20.0];
        let fitted = PoissonRegressor::<f64>::new()
            .with_alpha(0.0)
            .fit_pipeline(&x, &y)
            .unwrap();
        let preds = fitted.predict_pipeline(&x).unwrap();
        assert_eq!(preds.len(), 4);
    }

    // ---- GammaRegressor ----

    #[test]
    fn test_gamma_defaults() {
        let m = GammaRegressor::<f64>::new();
        assert_relative_eq!(m.alpha, 1.0);
        assert_eq!(m.max_iter, 100);
    }

    #[test]
    fn test_gamma_fit_predict() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![2.0, 5.0, 10.0, 20.0];

        let fitted = GammaRegressor::<f64>::new()
            .with_alpha(0.0)
            .with_max_iter(200)
            .fit(&x, &y)
            .unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 4);
        for &p in preds.iter() {
            assert!(p > 0.0);
        }
    }

    #[test]
    fn test_gamma_pipeline() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![2.0, 5.0, 10.0, 20.0];
        let fitted = GammaRegressor::<f64>::new()
            .with_alpha(0.0)
            .fit_pipeline(&x, &y)
            .unwrap();
        let preds = fitted.predict_pipeline(&x).unwrap();
        assert_eq!(preds.len(), 4);
    }

    // ---- TweedieRegressor ----

    #[test]
    fn test_tweedie_defaults() {
        let m = TweedieRegressor::<f64>::new();
        // sklearn TweedieRegressor default power=0.0 (Normal), link='auto'
        // (glm.py:867, :870).
        assert_relative_eq!(m.power, 0.0);
        assert_eq!(m.link, LinkConfig::Auto);
        assert_relative_eq!(m.alpha, 1.0);
    }

    #[test]
    fn test_tweedie_fit_predict() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![2.0, 5.0, 10.0, 20.0];

        let fitted = TweedieRegressor::<f64>::new()
            .with_power(1.5)
            .with_alpha(0.0)
            .with_max_iter(200)
            .fit(&x, &y)
            .unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 4);
        for &p in preds.iter() {
            assert!(p > 0.0);
        }
    }

    #[test]
    fn test_tweedie_pipeline() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![2.0, 5.0, 10.0, 20.0];
        let fitted = TweedieRegressor::<f64>::new()
            .with_alpha(0.0)
            .fit_pipeline(&x, &y)
            .unwrap();
        let preds = fitted.predict_pipeline(&x).unwrap();
        assert_eq!(preds.len(), 4);
    }

    // ---- Link ----

    #[test]
    fn test_link_inverse() {
        assert_relative_eq!(Link::Log.inverse(0.0_f64), 1.0);
        assert_relative_eq!(Link::Identity.inverse(3.5_f64), 3.5);
    }

    #[test]
    fn test_link_config_resolve_auto() {
        // glm.py:889-893: auto -> identity for power<=0, log for power>0.
        assert_eq!(LinkConfig::Auto.resolve(0.0), Link::Identity);
        assert_eq!(LinkConfig::Auto.resolve(-1.0), Link::Identity);
        assert_eq!(LinkConfig::Auto.resolve(1.5), Link::Log);
        assert_eq!(LinkConfig::Log.resolve(0.0), Link::Log);
        assert_eq!(LinkConfig::Identity.resolve(2.0), Link::Identity);
    }

    #[test]
    fn test_tweedie_with_link_builder() {
        let m = TweedieRegressor::<f64>::new().with_link(LinkConfig::Log);
        assert_eq!(m.link, LinkConfig::Log);
    }

    // ---- Solver (sklearn API parity, glm.py:140-145) ----

    #[test]
    fn test_solver_default_lbfgs() {
        // sklearn default solver='lbfgs' (glm.py:155).
        assert_eq!(
            GLMRegressor::<f64>::new(GLMFamily::Poisson).solver,
            Solver::Lbfgs
        );
        assert_eq!(PoissonRegressor::<f64>::new().solver, Solver::Lbfgs);
        assert_eq!(GammaRegressor::<f64>::new().solver, Solver::Lbfgs);
        assert_eq!(TweedieRegressor::<f64>::new().solver, Solver::Lbfgs);
    }

    #[test]
    fn test_with_solver_builder() {
        assert_eq!(
            PoissonRegressor::<f64>::new()
                .with_solver(Solver::NewtonCholesky)
                .solver,
            Solver::NewtonCholesky
        );
        assert_eq!(
            GLMRegressor::<f64>::new(GLMFamily::Gamma)
                .with_solver(Solver::NewtonCholesky)
                .solver,
            Solver::NewtonCholesky
        );
    }

    // ---- Variance function ----

    #[test]
    fn test_variance_poisson() {
        let v = GLMFamily::Poisson.variance(3.0_f64);
        assert_relative_eq!(v, 3.0);
    }

    #[test]
    fn test_variance_gamma() {
        let v = GLMFamily::Gamma.variance(3.0_f64);
        assert_relative_eq!(v, 9.0);
    }

    #[test]
    fn test_variance_tweedie() {
        let v = GLMFamily::Tweedie(1.5).variance(4.0_f64);
        assert_relative_eq!(v, 4.0_f64.powf(1.5), epsilon = 1e-10);
    }

    #[test]
    fn test_glm_negative_y() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![1.0, -2.0, 3.0];
        assert!(
            GLMRegressor::<f64>::new(GLMFamily::Poisson)
                .fit(&x, &y)
                .is_err()
        );
    }
}
