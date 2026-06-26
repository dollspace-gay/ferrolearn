# f_classif / r_regression / f_regression / chi2 — univariate feature scoring

<!--
tier: 3-component
status: shipped-partial
baseline-commit: 3439ff7f
upstream: scikit-learn 1.5.2
upstream-paths:
  - sklearn/feature_selection/_univariate_selection.py  # f_oneway (:43-117) computes the one-way ANOVA F via ss_alldata/sums_args/square_of_sums_alldata -> ssbn/sswn -> msb/msw -> f = msb/msw, prob = special.fdtrc(dfbn, dfwn, f) (:116); f_classif (:127-173) groups X by np.unique(y) and calls f_oneway(*args) (:172-173); _chisquare (:176-192) chisq = ((f_obs-f_exp)**2/f_exp).sum(axis=0), p = special.chdtrc(k-1, chisq) (:192); chi2 (:202-288) observed = Y.T@X (LabelBinarizer one-hot), feature_count = X.sum(0), class_prob = Y.mean(0), expected = class_prob.T@feature_count, _chisquare(observed, expected) (:275-288); r_regression (:300-393) Pearson corr (center/force_finite); f_regression (:405-...) corr from r_regression, F = corr**2/(1-corr**2)*deg_of_freedom (deg=n-2 when center), p = stats.f.sf(F, 1, deg). p-values come from scipy.special.fdtrc / chdtrc (exact).
ferrolearn-module: ferrolearn-preprocess/src/feature_scoring.rs
parity-ops: f_classif, r_regression, f_regression, chi2
crosslink-issue: 1416
-->

## Summary

scikit-learn's univariate scoring functions `f_classif`
(`_univariate_selection.py:127`, via `f_oneway` `:43-117`), `r_regression`
(`:301`), `f_regression` (`:405`, via `r_regression`) and `chi2` (`:202`, via
`_chisquare` `:176`) compute per-feature ANOVA F-statistics, signed Pearson
correlations, Pearson-correlation-derived F-statistics, and chi-squared
statistics respectively.
The p-values come from `scipy.special.fdtrc` (the exact F-distribution survival
function) and `scipy.special.chdtrc` (the exact chi-squared survival function).

`ferrolearn-preprocess/src/feature_scoring.rs` ships the statistic
computations — `f_classif`, `r_regression`, `f_regression`, `chi2` — where
`r_regression` returns signed correlations and the others return
`(Array1<F> statistics, Array1<F> p_values)`, plus the
`compute_scores_classif` (`:656`) / `compute_scores_regression` (`:678`)
score-only dispatchers and a set of hand-rolled distribution helpers
(`f_distribution_sf` `:392`, `chi2_distribution_sf` `:419`,
`upper_regularized_gamma` `:438`, `lower_regularized_gamma_series` `:491`,
`regularized_incomplete_beta` `:522`, `ln_gamma` `:608`). This is a
**DETERMINISTIC numeric-parity unit** — no RNG; the statistics are value-verifiable
against the live oracle to ~1e-15.

This is a **shipped-partial** unit: **6 SHIPPED** (REQ-1 `f_classif` F-statistic
parity, REQ-2 `f_regression` F-statistic parity, REQ-3 `chi2` statistic parity,
REQ-4 p-value parity, REQ-5 error / parameter contracts, REQ-7 `r_regression`) /
**5 NOT-STARTED** (REQ-6 `f_regression` `center=False` + `force_finite`; REQ-8
sparse `chi2`; REQ-9 `mutual_info_*`; REQ-10 PyO3 binding; REQ-11 ferray
substrate).

**P-value parity (REQ-4, now SHIPPED).** The F-statistics and chi-squared
statistics match scipy to machine precision, and the p-values for all three
functions now match scipy as well. The F-distribution p-values flow through the
rewritten `regularized_incomplete_beta` (`:549`) + the `betacf` Lentz continued
fraction (`:574`, Numerical Recipes §6.4) and match `scipy.special.fdtrc` /
`scipy.stats.f.sf` across small-p (deep tail to ~1e-23), large-p, moderate, p≈0.5,
and varied `df1∈{1,2,4}` / `df2∈{3..48}` regimes (~13-15 significant figures). The
chi-squared p-values flow through the gamma-based `chi2_distribution_sf` (`:441`) /
`upper_regularized_gamma` (`:460`) and match `scipy.special.chdtrc`. This was the
DIV-1 finding tracked as blocker #1417, now RESOLVED (a fixer rewrote
`regularized_incomplete_beta` and added `betacf`). Pinned by 29 oracle tests in
`tests/divergence_feature_scoring.rs`.

## Probes (live sklearn oracle, scikit-learn 1.5.2 + scipy)

All values below are live oracle output; ferrolearn values are from running the
shipped functions on the identical inputs (captured at baseline `3439ff7f`). The
STATISTICS are pinned to ~1e-15; the P-VALUES now match scipy for all three
functions (REQ-4 SHIPPED; the F-distribution SF was repaired under blocker #1417 —
see REQ-4).

```bash
# ---------------------------------------------------------------------------
# PROBE A (REQ-1, REQ-4) — f_classif: ANOVA F + p-value
#   X = [[1,2,3],[4,5,6],[7,8,1],[2,1,9],[5,3,2],[8,9,4]], y = [0,0,1,1,2,2]
# ---------------------------------------------------------------------------
python3 -c "
import numpy as np; from sklearn.feature_selection import f_classif
X=np.array([[1,2,3],[4,5,6],[7,8,1],[2,1,9],[5,3,2],[8,9,4]],float)
y=np.array([0,0,1,1,2,2]); f,p=f_classif(X,y); print('F',f); print('p',p)"
#   sklearn F = [1.11627907, 0.20212766, 0.16883117]   (df1=2, df2=3)
#   sklearn p = [0.43412099, 0.82727270, 0.85215439]   (special.fdtrc, EXACT)
#
#   ferrolearn f_classif F = [1.1162790697674418, 0.2021276595744681, 0.1688311688311688]
#     -> matches sklearn to ~1e-15 (machine precision).  REQ-1 SHIPPED.
#   ferrolearn f_classif p = [0.43412099354562..., 0.82727270410..., 0.85215438837...]
#     -> matches scipy fdtrc to ~13-15 sig figs (post-#1417 fix).  REQ-4 SHIPPED.

# ---------------------------------------------------------------------------
# PROBE B (REQ-2, REQ-4) — f_regression: Pearson-F + p-value
#   X as above, y = [1.5, 2.0, 3.5, 4.0, 5.5, 6.0]
# ---------------------------------------------------------------------------
python3 -c "
import numpy as np; from sklearn.feature_selection import f_regression
X=np.array([[1,2,3],[4,5,6],[7,8,1],[2,1,9],[5,3,2],[8,9,4]],float)
y=np.array([1.5,2.0,3.5,4.0,5.5,6.0]); f,p=f_regression(X,y); print('F',f); print('p',p)"
#   sklearn F = [3.01785714, 0.57791853, 0.04359837]   (df1=1, df2=4)
#   sklearn p = [0.15735256, 0.48947183, 0.84480465]   (stats.f.sf == fdtrc, EXACT)
#
#   ferrolearn f_regression F = [3.0178571428571432, 0.5779185322703944, 0.04359836656257507]
#     -> matches sklearn to ~1e-15.  REQ-2 SHIPPED.
#   ferrolearn f_regression p = [0.15735256165..., 0.48947182910..., 0.84480465677...]
#     -> matches scipy stats.f.sf / fdtrc to ~13-15 sig figs (post-#1417 fix).  REQ-4 SHIPPED.

# ---------------------------------------------------------------------------
# PROBE C (REQ-3, REQ-4) — chi2: chi-squared statistic + p-value
#   X = [[1,0,3],[0,2,1],[2,1,0],[1,3,2]], y = [0,1,0,1]
# ---------------------------------------------------------------------------
python3 -c "
import numpy as np; from sklearn.feature_selection import chi2
X=np.array([[1,0,3],[0,2,1],[2,1,0],[1,3,2]],float)
y=np.array([0,1,0,1]); s,p=chi2(X,y); print('stat',s); print('p',p)"
#   sklearn stat = [1.0, 2.66666667, 0.0]   (df = n_classes-1 = 1)
#   sklearn p    = [0.31731051, 0.10247043, 1.0]   (special.chdtrc, EXACT)
#
#   ferrolearn chi2 stat = [1.0, 2.6666666666666665, 0.0]
#     -> matches sklearn to ~1e-15.  REQ-3 SHIPPED.
#   ferrolearn chi2 p    = [0.317310507863, 0.102470434860, 1.000000000000]
#     -> matches scipy chdtrc to ~1e-9 (gamma-based SF).  REQ-4 SHIPPED.

# ---------------------------------------------------------------------------
# PROBE D (REQ-4 verification) — the F<->beta mapping + the rewritten CF agree.
# ---------------------------------------------------------------------------
python3 -c "
from scipy import special
F=1.1162790697674418; d1=2; d2=3
z=d2/(d2+d1*F)
print('betainc(d2/2,d1/2,z) =', special.betainc(d2/2,d1/2,z))  # 0.43412099...
print('fdtrc(d1,d2,F)       =', special.fdtrc(d1,d2,F))         # 0.43412099...
"
#   -> BOTH 0.43412099.  ferrolearn uses exactly this z/a/b mapping (f_distribution_sf:425-428)
#      and the rewritten regularized_incomplete_beta:549 + betacf:574 Lentz CF now returns
#      0.43412099... here too — the CF matches special.betainc to ~13-15 sig figs (#1417 fixed).
```

**P-value tolerance summary (the critic will pin this).**

| score fn | statistic vs scipy | p-value SF path | p-value vs scipy |
|---|---|---|---|
| `f_classif` | ~1e-15 (exact) | `f_distribution_sf` -> `regularized_incomplete_beta` (`:549`) + `betacf` (`:574`) | ~13-15 sig figs (matches `fdtrc`) |
| `f_regression` | ~1e-15 (exact) | `f_distribution_sf` -> `regularized_incomplete_beta` (`:549`) + `betacf` (`:574`) | ~13-15 sig figs (matches `stats.f.sf`/`fdtrc`) |
| `chi2` | ~1e-15 (exact) | `chi2_distribution_sf` -> `upper_regularized_gamma` (`:460`) | ~1e-9 (matches `chdtrc`) |

REQ-4 is scoped as "p-values for all three" and all three columns now match scipy.
A single shared helper (`regularized_incomplete_beta` + the `betacf` Lentz CF)
underpins BOTH `f_classif` and `f_regression` p-values; after the #1417 rewrite it
matches `special.fdtrc` / `special.betainc` to ~13-15 significant figures across the
small-p (deep tail to ~1e-23), large-p, moderate, p≈0.5, and varied-df regimes
pinned in `tests/divergence_feature_scoring.rs`. The chi-squared p-value path is
independent (gamma-based) and matches `special.chdtrc` to ~1e-9. REQ-4 is SHIPPED.

## Requirements

- REQ-1: **`f_classif` F-statistic value parity (one-way ANOVA)** (SHIPPED). For
  each feature compute the between-class / within-class sums of squares and
  `F = (SSB/(k-1)) / (SSW/(n-k))`, matching `f_oneway` (`:92-117`): `ssbn`/`sswn`
  via `square_of_sums_args`/`ss_alldata`, `msb = ssbn/dfbn`, `msw = sswn/dfwn`,
  `f = msb/msw` (`:108-113`), with `dfbn = n_classes-1`, `dfwn = n_samples-n_classes`
  (`:106-107`). `f_classif` (`:127`) groups `X` by `np.unique(y)` and calls
  `f_oneway(*args)` (`:172-173`). ferrolearn's `f_classif` (`feature_scoring.rs:62`)
  produces the identical F-vector to ~1e-15 (Probe A).

- REQ-2: **`f_regression` F-statistic value parity (Pearson-F)** (SHIPPED). For each
  feature compute the Pearson correlation `r` with the centered target and
  `F = r² · (n-2) / (1-r²)`, matching `f_regression` (`:405-...`): `correlation =
  r_regression(...)` (`:498`), then `F = corr²/(1-corr²) · deg_of_freedom` with
  `deg_of_freedom = n-2` (center default). ferrolearn's `f_regression`
  (`feature_scoring.rs:181`) computes `cov / sqrt(x_var·y_var)` then
  `r²·(n-2)/(1-r²)` (`:240-245`), producing the identical F-vector to ~1e-15
  (Probe B).

- REQ-3: **`chi2` chi-squared statistic value parity** (SHIPPED). For each feature,
  observed per-class sums vs expected `class_total · feature_total / n`, with
  `chi2 = Σ_class (obs-exp)²/exp` and `df = n_classes-1`, matching `chi2` (`:202`):
  `observed = Y.T @ X`, `feature_count = X.sum(0)`, `class_prob = Y.mean(0)`,
  `expected = class_prob.T @ feature_count` (`:275-286`), then `_chisquare`
  `chisq = ((f_obs-f_exp)²/f_exp).sum(axis=0)` (`:186-191`). ferrolearn's `chi2`
  (`feature_scoring.rs:294`) computes per-class `observed`/`expected` and
  `(obs-exp)²/exp` (`:357-366`), producing the identical statistic vector to ~1e-15
  (Probe C).

- REQ-4: **p-values for all three (F/chi2 SF vs scipy `fdtrc`/`chdtrc`)**
  (SHIPPED). sklearn returns `prob = special.fdtrc(dfbn, dfwn, f)` (`f_oneway:116`),
  `p = stats.f.sf(F, 1, deg)` (`f_regression`), and `special.chdtrc(k-1, chisq)`
  (`_chisquare:192`) — all EXACT. ferrolearn's chi-squared p-value path
  (`chi2_distribution_sf:441` -> `upper_regularized_gamma:460`) matches `chdtrc` to
  ~1e-9, and the F-distribution p-value path (`f_distribution_sf:414` ->
  `regularized_incomplete_beta:549` + `betacf:574`, a Numerical Recipes §6.4 Lentz
  continued fraction) now matches `special.fdtrc` / `stats.f.sf` to ~13-15
  significant figures across small-p (deep tail to ~1e-23), large-p, moderate, p≈0.5
  and varied `df1∈{1,2,4}` / `df2∈{3..48}` regimes (Probe D; 29 oracle tests in
  `tests/divergence_feature_scoring.rs`). This was DIV-1, tracked as blocker #1417,
  now RESOLVED (the F↔beta mapping was always correct; the fixer rewrote the CF).

- REQ-5: **error / parameter contracts** (SHIPPED). `f_classif`: empty `x` →
  `InsufficientSamples` (`:67-73`), `y.len() != n_rows` → `ShapeMismatch` (`:74-80`),
  `< 2` classes → `InvalidParameter` (`:89-94`). `f_regression`: `< 3` samples →
  `InsufficientSamples` (`:186-192`), shape mismatch → `ShapeMismatch` (`:193-199`).
  `chi2`: empty `x` → `InsufficientSamples` (`:299-305`), shape mismatch →
  `ShapeMismatch` (`:306-312`), any negative feature value → `InvalidParameter`
  (`:315-326`, mirroring sklearn's `if np.any(X < 0): raise ValueError("Input X must
  be non-negative.")` `:265-266`). These are the ferrolearn error contracts scoped
  to the dense `Array2`/`Array1` API.

- REQ-6: **`f_regression` `center=False` + `force_finite` (sklearn 1.5)** (NOT-STARTED).
  sklearn's `f_regression`/`r_regression` accept `center` (`:326`) and `force_finite`
  (`:330`): `center=False` skips target/feature centering (`:369-381`), and
  `force_finite=True` (default) maps `nan` correlations (constant feature/target) to
  `0.0` (`:388-392`) and, in `f_regression`, sets `F=0.0`/`p=1.0` for the nan case and
  `F=finfo.max`/`p=0.0` for the perfectly-correlated case (`:447-461`). ferrolearn's
  `f_regression` always centers and on a constant feature returns `r=0` →`F=0`
  (`:234-237`) and on `r²>=1` returns `F=inf` (`:241-243`) — no `center`/`force_finite`
  parameters and no finite-max clamp. NOT-STARTED on blocker #1418.

- REQ-7: **`r_regression` free function returning signed Pearson correlation**
  (SHIPPED). `r_regression` mirrors sklearn defaults (`center=True`,
  `force_finite=True`) and `r_regression_with_options` exposes the explicit
  `center` / `force_finite` controls from `_univariate_selection.py:301-393`.
  The implementation uses sklearn's centered identity
  `dot(y - mean(y), X) / (||X_centered|| * ||y_centered||)` and maps undefined
  NaN correlations to `0.0` when `force_finite=true`. Verification:
  `tests/divergence_r_regression.rs` pins centered, uncentered, constant-feature,
  constant-target, and validation contracts against the live sklearn 1.5.2 oracle.

- REQ-8: **sparse `chi2` (CSR)** (NOT-STARTED). sklearn's `chi2` accepts
  `accept_sparse="csr"` (`:264`) and computes `observed = Y.T @ X` via
  `safe_sparse_dot` (`:275`) over sparse `X`. ferrolearn's `chi2` operates only on
  dense `Array2<F>` (`:294`); there is no `sprs::CsMat` path. NOT-STARTED on blocker
  #1420.

- REQ-9: **`mutual_info_classif` / `mutual_info_regression`** (NOT-STARTED /
  not-translated). The route sibling score functions `mutual_info_*` live in a
  SEPARATE sklearn module (`sklearn/feature_selection/_mutual_info.py`), are
  k-NN/entropy estimators with RNG, and are NOT part of this file's
  `_univariate_selection.py` translation unit. They are absent from
  `feature_scoring.rs`. NOT-STARTED on blocker #1421 (out of this file's scope).

- REQ-10: **PyO3 binding** (NOT-STARTED). There is no `_Rsf_classif` /
  `_Rsf_regression` / `_Rschi2` (or analogous) CPython binding in `ferrolearn-python`,
  so the scoring functions are unreachable from Python. NOT-STARTED on blocker
  #1422.

- REQ-11: **ferray substrate** (NOT-STARTED). The scoring + SF helpers compute over
  `ndarray::Array2`/`Array1` and `num_traits::Float`, not `ferray-core` arrays
  (R-SUBSTRATE-1/2). NOT-STARTED on blocker #1423.

## Acceptance criteria

- AC-1 (REQ-1): `f_classif` on Probe A's `(X, y)` yields F-stats
  `[1.11627907, 0.20212766, 0.16883117]` matching `sklearn.feature_selection.f_classif`
  to `|Δ| < 1e-9` (observed ~1e-15). Pinned by in-module `test_f_classif_basic`,
  `test_f_classif_perfect_separation` (`F = inf` on perfect separation), plus the
  Probe A oracle gate.

- AC-2 (REQ-2): `f_regression` on Probe B's `(X, y)` yields F-stats
  `[3.01785714, 0.57791853, 0.04359837]` matching `f_regression` to `|Δ| < 1e-9`
  (observed ~1e-15). Pinned by in-module `test_f_regression_perfect_correlation`,
  `test_f_regression_no_correlation`, `test_f_regression_constant_feature`, plus the
  Probe B oracle gate.

- AC-3 (REQ-3): `chi2` on Probe C's `(X, y)` yields statistics `[1.0, 2.66666667, 0.0]`
  matching `chi2` to `|Δ| < 1e-9` (observed ~1e-15). Pinned by in-module
  `test_chi2_basic`, `test_chi2_all_zeros` (`stat=0`, `p=1` on an all-zero feature),
  plus the Probe C oracle gate.

- AC-4 (REQ-4): `f_classif`/`f_regression` p-values match `special.fdtrc` /
  `stats.f.sf` and `chi2` p-values match `special.chdtrc` to a stated tolerance.
  **PASSES**: the F-distribution p-values match scipy to ~13-15 significant figures
  (abs tol 1e-6 for p≥1e-4, relative tol 1e-6 in the deep tail down to ~1e-23) and
  the chi2 p-values match `chdtrc` to ~1e-9. Pinned by the 29 oracle tests in
  `tests/divergence_feature_scoring.rs` — `divergence_f_classif_pvalues`,
  `divergence_f_regression_pvalues`, the `reaudit_fclassif_*` / `reaudit_fregression_*`
  small-p/large-p/moderate/p≈0.5/varied-df probes, and `greenguard_chi2_pvalues` /
  `reaudit_chi2_3class_df2_pvalues`. Was DIV-1 / #1417 (RESOLVED). SHIPPED.

- AC-5 (REQ-5): `f_classif(Array2::zeros((0,2)), zeros(0))` →
  `Err(InsufficientSamples)`; `f_classif` with `y.len()` mismatch → `Err(ShapeMismatch)`;
  `f_classif` with a single class → `Err(InvalidParameter)`; `f_regression` with
  `< 3` rows → `Err(InsufficientSamples)`; `chi2` with a negative value →
  `Err(InvalidParameter)`. Pinned by in-module `test_f_classif_empty_input`,
  `test_f_classif_shape_mismatch`, `test_f_classif_single_class_error`,
  `test_f_regression_too_few_samples`, `test_f_regression_shape_mismatch`,
  `test_chi2_negative_value_error`, `test_chi2_empty_input`, `test_chi2_shape_mismatch`.

- AC-6 (REQ-6): `f_regression(X, y, center=False)` and the `force_finite=False`
  `nan` path; ferrolearn `f_regression` has neither parameter. NOT-STARTED.

- AC-7 (REQ-7): `r_regression(X, y)` returns signed correlations in `[-1, 1]`
  and mirrors sklearn's `center=True, force_finite=True` defaults; the options
  variant covers `center=False` and `force_finite=False`. SHIPPED.

- AC-8 (REQ-8): `chi2(X_csr, y)` on a sparse `X` matches the dense result;
  ferrolearn `chi2` rejects/has no sparse path. NOT-STARTED.

- AC-9 (REQ-9): `mutual_info_classif(X, y)` / `mutual_info_regression(X, y)` exist;
  ferrolearn has neither (separate `_mutual_info.py` unit, out of scope). NOT-STARTED.

- AC-10 (REQ-10): a CPython `f_classif`/`f_regression`/`chi2` binding computes
  scores from Python; no such binding exists in `ferrolearn-python`. NOT-STARTED.

- AC-11 (REQ-11): the scoring + SF helpers compute on `ferray-core` arrays rather
  than `ndarray` + `num_traits::Float`. NOT-STARTED.

`## REQ status`

## REQ status

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (`f_classif` F-statistic parity) | SHIPPED | impl `pub fn f_classif in feature_scoring.rs` (`:62`) computes per-feature `ss_between = Σ n_k·(class_mean-grand_mean)²` / `ss_within = Σ (x-class_mean)²` then `f = (ss_between/df_b)/(ss_within/df_w)` (`:111-139`, `df_between = n_classes-1`, `df_within = n_samples-n_classes` `:99-100`), mirroring `f_oneway` `msb = ssbn/dfbn; msw = sswn/dfwn; f = msb/msw` (`sklearn/feature_selection/_univariate_selection.py:108-113`, `dfbn`/`dfwn` `:106-107`) reached from `f_classif` `return f_oneway(*args)` (`:173`). Non-test consumer: boundary re-export `pub use feature_scoring::{... f_classif ...}` (`lib.rs:173-175`) — grandfathered S5 / R-DEFER-1 boundary scoring API. Verification: Probe A — ferrolearn F `[1.1162790697674418, 0.2021276595744681, 0.1688311688311688]` vs sklearn `[1.11627907, 0.20212766, 0.16883117]` (`|Δ| ~ 1e-15`); `cargo test -p ferrolearn-preprocess --lib` → `test_f_classif_basic`, `test_f_classif_perfect_separation` (F=inf), `test_f_classif_p_values_bounded` green. |
| REQ-2 (`f_regression` F-statistic parity) | SHIPPED | impl `pub fn f_regression in feature_scoring.rs` (`:181`) computes `r = cov / sqrt(x_var·y_var)` then `f = r²·(n-2)/(1-r²)` (`:226-245`), mirroring `f_regression` `corr = r_regression(...)` (`:498`) then `F = corr²/(1-corr²)·deg_of_freedom` with `deg_of_freedom = n-2`. Non-test consumer: boundary re-export `pub use feature_scoring::{... f_regression ...}` (`lib.rs:173-175`); also `compute_scores_regression` (`:678`) calls `f_regression` and is itself re-exported (`lib.rs:174`). Verification: Probe B — ferrolearn F `[3.0178571428571432, 0.5779185322703944, 0.04359836656257507]` vs sklearn `[3.01785714, 0.57791853, 0.04359837]` (`|Δ| ~ 1e-15`); `cargo test -p ferrolearn-preprocess --lib` → `test_f_regression_perfect_correlation`, `test_f_regression_no_correlation`, `test_f_regression_constant_feature`, `test_f_regression_p_values_bounded` green. |
| REQ-3 (`chi2` statistic parity) | SHIPPED | impl `pub fn chi2 in feature_scoring.rs` (`:294`) computes per-class `observed = Σ_rows col[i]`, `expected = total_sum·n_k/n`, `chi2 = Σ_class (obs-exp)²/exp` (`:353-369`, `df = n_classes-1` `:371`), mirroring `chi2` `observed = Y.T@X`, `expected = class_prob.T@feature_count` (`sklearn/feature_selection/_univariate_selection.py:275-286`) into `_chisquare` `chisq = ((f_obs-f_exp)²/f_exp).sum(axis=0)` (`:186-191`). Non-test consumer: boundary re-export `pub use feature_scoring::{chi2 ...}` (`lib.rs:173-174`); also `compute_scores_classif("chi2")` (`:666-669`) calls `chi2` and is re-exported. Verification: Probe C — ferrolearn stat `[1.0, 2.6666666666666665, 0.0]` vs sklearn `[1.0, 2.66666667, 0.0]` (`|Δ| ~ 1e-15`); `cargo test -p ferrolearn-preprocess --lib` → `test_chi2_basic`, `test_chi2_all_zeros` (stat=0/p=1), green. |
| REQ-4 (p-values vs scipy `fdtrc`/`chdtrc`) | SHIPPED | impl `f_distribution_sf in feature_scoring.rs` (`:414`) maps `z = d2/(d2+d1·x)`, `a = d2/2`, `b = d1/2` (`:425-428`) into `regularized_incomplete_beta in feature_scoring.rs` (`:549`) + `betacf in feature_scoring.rs` (`:574`), a Numerical Recipes §6.4 Lentz continued fraction with the `x < (a+1)/(a+b+2)` symmetry flip (`:564-568`), now matching `special.betainc` / `special.fdtrc` / `stats.f.sf` to ~13-15 significant figures (Probe D). The chi-squared path `chi2_distribution_sf in feature_scoring.rs` (`:441`) -> `upper_regularized_gamma in feature_scoring.rs` (`:460`) matches `chdtrc` to ~1e-9. Non-test consumer: boundary re-export `pub use feature_scoring::{... f_classif, f_regression, chi2 ...}` (`lib.rs:173-175`) — the p-value vector is part of each fn's `(statistics, p_values)` return. Verification: `cargo test -p ferrolearn-preprocess --test divergence_feature_scoring` (29 tests green) — `divergence_f_classif_pvalues`, `divergence_f_regression_pvalues`, the `reaudit_fclassif_*`/`reaudit_fregression_*` small-p (tail to ~1e-23)/large-p/moderate/p≈0.5/varied-df probes, `greenguard_chi2_pvalues`, `reaudit_chi2_3class_df2_pvalues`. Was DIV-1, blocker #1417, now RESOLVED. |
| REQ-5 (error / parameter contracts) | SHIPPED | impl `f_classif` (`:62`) returns `Err(InsufficientSamples)` on empty `x` (`:67-73`), `Err(ShapeMismatch)` on `y.len() != n_rows` (`:74-80`), `Err(InvalidParameter)` on `< 2` classes (`:89-94`); `f_regression` (`:181`) returns `Err(InsufficientSamples)` on `< 3` rows (`:186-192`) and `Err(ShapeMismatch)` on length mismatch (`:193-199`); `chi2` (`:294`) returns `Err(InsufficientSamples)` on empty `x` (`:299-305`), `Err(ShapeMismatch)` on length mismatch (`:306-312`), and `Err(InvalidParameter)` on any negative feature (`:315-326`) — mirroring sklearn `if np.any(X < 0): raise ValueError("Input X must be non-negative.")` (`sklearn/feature_selection/_univariate_selection.py:265-266`). Non-test consumer: boundary re-export (`lib.rs:173-175`). Verification: `cargo test -p ferrolearn-preprocess --lib` → `test_f_classif_empty_input`, `test_f_classif_shape_mismatch`, `test_f_classif_single_class_error`, `test_f_regression_too_few_samples`, `test_f_regression_shape_mismatch`, `test_chi2_negative_value_error`, `test_chi2_empty_input`, `test_chi2_shape_mismatch` green. |
| REQ-6 (`f_regression` `center=False` + `force_finite`) | NOT-STARTED | open prereq blocker #1418. `f_regression in feature_scoring.rs` (`:181`) has no `center`/`force_finite` parameters: it always centers (`x_mean`/`y_mean` subtracted `:205-231`), returns `r=0`→`F=0` on a constant feature (`:234-237`) and `F=inf` on `r²>=1` (`:241-243`) — no `finfo.max` clamp / `p=0.0`. sklearn's `f_regression(..., center=True, force_finite=True)` (`:405`) maps the nan case to `F=0.0`/`p=1.0` and the perfect-correlation case to `F=finfo.max`/`p=0.0` (`:447-461`) and supports `center=False` (`:369-381`). |
| REQ-7 (`r_regression` free fn + correlation surface) | SHIPPED | `r_regression` and `r_regression_with_options` in `feature_scoring.rs` mirror sklearn's signed Pearson correlation helper (`sklearn/feature_selection/_univariate_selection.py:301-393`). Defaults are `center=true` and `force_finite=true`; the options helper exposes both controls. Non-test consumer: crate-root re-export in `lib.rs` plus API proof. Verification: `tests/divergence_r_regression.rs` pins centered, uncentered, constant-feature/target force-finite, and validation behavior against live sklearn 1.5.2. |
| REQ-8 (sparse `chi2` CSR) | NOT-STARTED | open prereq blocker #1420. sklearn's `chi2` accepts `accept_sparse="csr"` (`:264`) and forms `observed = safe_sparse_dot(Y.T, X)` over sparse `X` (`:275`). ferrolearn's `chi2` (`:294`) takes only dense `Array2<F>`; there is no `sprs::CsMat` overload. |
| REQ-9 (`mutual_info_classif` / `mutual_info_regression`) | NOT-STARTED | open prereq blocker #1421 (out of this file's scope). The `mutual_info_*` score functions live in the SEPARATE `sklearn/feature_selection/_mutual_info.py` (k-NN entropy estimators with RNG), not in `_univariate_selection.py`. They are absent from `feature_scoring.rs` — not-translated. |
| REQ-10 (PyO3 binding) | NOT-STARTED | open prereq blocker #1422. No CPython binding for `f_classif`/`f_regression`/`chi2` exists in `ferrolearn-python/src` (`grep -rn "f_classif\|f_regression\|chi2" ferrolearn-python/src` finds none), so the scoring functions are unreachable from Python. |
| REQ-11 (ferray substrate) | NOT-STARTED | open prereq blocker #1423. The scoring fns + SF helpers compute over `ndarray::Array2`/`Array1` (`x.column(j)`, `Array1::zeros`) and `num_traits::Float`, not `ferray-core` arrays (R-SUBSTRATE-1/2). |

## Architecture

**ferrolearn (existing, compiled at baseline `3439ff7f`).** `feature_scoring.rs`
exposes three standalone generic scoring functions, each
`<F: Float + Send + Sync + 'static>(x: &Array2<F>, y: &...) -> Result<(Array1<F>,
Array1<F>), FerroError>` returning `(statistics, p_values)`:

- `f_classif(x, y: &Array1<usize>)` (`:62`) — one-way ANOVA. Per feature: grand mean
  over the column, per-class mean over the class row-indices (collected into a
  `HashMap<usize, Vec<usize>>` `:83-87`), `ss_between = Σ n_k·(class_mean -
  grand_mean)²`, `ss_within = Σ (col[i] - class_mean)²`, then
  `f = (ss_between/df_b)/(ss_within/df_w)` with `df_b = n_classes-1`, `df_w =
  n_samples-n_classes` (`:99-139`). `ms_within == 0` → `F = inf` (`:134-135`).
- `f_regression(x, y: &Array1<F>)` (`:181`) — Pearson-F. Per feature: `x_mean`,
  `x_var`, `cov = Σ (xi-x_mean)(yi-y_mean)`, `r = cov/sqrt(x_var·y_var)` (0 if denom
  0), `f = r²·(n-2)/(1-r²)` (inf if `r²>=1`) (`:217-245`).
- `chi2(x, y: &Array1<usize>)` (`:294`) — chi-squared. Per feature: `total_sum`,
  per-class `observed` sum and `expected = total_sum·n_k/n`,
  `chi2 = Σ (obs-exp)²/exp`, `df = n_classes-1` (`:342-372`). An all-zero feature
  short-circuits to `stat=0`/`p=1` (`:346-351`).

Both `compute_scores_classif(x, y, func)` (`:656`, dispatching `"f_classif"` /
`"chi2"`, score-only) and `compute_scores_regression(x, y)` (`:678`) are thin
re-export-grade adapters returning only the statistic vector (the p-values are
dropped). The module is **compiled** (`pub mod feature_scoring;`, `lib.rs:96`) and
**re-exported** (`pub use feature_scoring::{chi2, compute_scores_classif,
compute_scores_regression, f_classif, f_regression};`, `lib.rs:173-175`) — that
boundary re-export is the grandfathered consumer pinning the SHIPPED rows.

NOTE: the `SelectPercentile` / `SelectKBest` paths carry their OWN `anova_f_scores`
implementation; `feature_scoring.rs`'s functions are NOT consumed internally by the
selectors — the consumer is the boundary re-export (S5 / R-DEFER-1).

**Distribution helpers (the p-value machinery, REQ-4).** ferrolearn reimplements
the survival functions scipy gets from `special.fdtrc` / `special.chdtrc`:

- `f_distribution_sf(x, df1, df2)` (`:414`): `P(F > x) = I_{d2/(d2+d1·x)}(d2/2, d1/2)`
  via `regularized_incomplete_beta` (`:425-430`). The mapping is correct (Probe D:
  `special.betainc(d2/2, d1/2, z) == fdtrc`).
- `regularized_incomplete_beta(x, a, b)` (`:549`): the Numerical Recipes §6.4 `front`
  prefactor `x^a (1-x)^b / (a·B(a,b))` (computed in log space, `:561`) times the
  `betacf` (`:574`) Lentz continued fraction, with a symmetry flip at
  `x > (a+1)/(a+b+2)` (`:564-568`) so the CF runs only in its fast-converging regime.
  **This is the rewritten helper** (blocker #1417); it now matches `special.betainc`
  to ~13-15 significant figures.
- `betacf(a, b, x)` (`:574`): the Numerical Recipes §6.4 Lentz-method continued
  fraction (even/odd steps, `tiny`/`eps` guards, `MAXIT=200`), the core of the F-SF
  p-value path.
- `chi2_distribution_sf(x, df)` (`:441`): `Q(df/2, x/2)` via `upper_regularized_gamma`
  (`:460`), which uses a series for `x < a+1` (`lower_regularized_gamma_series:513`)
  and a Lentz CF otherwise (`:474-495`). **This path is accurate** (Probe C, ~1e-9).
- `ln_gamma` (`:638`) / `ln_beta` (`:633`): Lanczos (g=7, n=9) — `ln_gamma` pinned by
  `test_ln_gamma_known_values` to ~1e-8.

**sklearn (target contract).** `f_classif` (`:127`) → `f_oneway(*args)` (`:43-117`):
`msb = ssbn/dfbn`, `msw = sswn/dfwn`, `f = msb/msw`, `prob = special.fdtrc(dfbn,
dfwn, f)` (`:108-116`). `f_regression` (`:405`) → `corr = r_regression(...)` then
`F = corr²/(1-corr²)·deg_of_freedom`, `p = stats.f.sf(F, 1, deg)`. `chi2` (`:202`) →
`observed = Y.T@X`, `expected = class_prob.T@feature_count`, then `_chisquare`
(`:176`): `chisq = ((f_obs-f_exp)²/f_exp).sum(axis=0)`, `p = special.chdtrc(k-1,
chisq)` (`:192`). All p-values are scipy-exact.

**The gap.** ferrolearn matches sklearn on the three STATISTICS to ~1e-15 (REQ-1,
REQ-2, REQ-3), the error/parameter contracts (REQ-5), and — after the #1417 rewrite
of `regularized_incomplete_beta` + `betacf` — the p-values for all three functions
(REQ-4): the chi-squared p-values match `chdtrc` to ~1e-9 and the F-distribution
p-values match `fdtrc` / `stats.f.sf` to ~13-15 significant figures. The remaining
gaps are parameterization/surface: no `center`/`force_finite` (REQ-6), no
`r_regression` (REQ-7), no sparse `chi2` (REQ-8), no `mutual_info_*` (REQ-9,
separate unit), no PyO3 binding (REQ-10), and the non-ferray substrate (REQ-11).
This is a **shipped-partial** unit (5 SHIPPED / 6 NOT-STARTED).

## Verification

Commands establishing the SHIPPED claims (REQ-1/2/3 statistic parity, REQ-4 p-value
parity, REQ-5 error contracts):

```bash
# Module is compiled + re-exported (the boundary consumer):
grep -n "mod feature_scoring" ferrolearn-preprocess/src/lib.rs           # lib.rs:96
grep -n "f_classif\|f_regression\|chi2" ferrolearn-preprocess/src/lib.rs # lib.rs:173-175

# REQ-1/2/3 statistic value parity + REQ-5 error contracts (in-module tests):
cargo test -p ferrolearn-preprocess --lib
#   REQ-1: test_f_classif_basic, test_f_classif_perfect_separation, test_f_classif_p_values_bounded
#   REQ-2: test_f_regression_perfect_correlation, test_f_regression_no_correlation,
#          test_f_regression_constant_feature, test_f_regression_p_values_bounded
#   REQ-3: test_chi2_basic, test_chi2_all_zeros
#   REQ-5: test_f_classif_empty_input, test_f_classif_shape_mismatch,
#          test_f_classif_single_class_error, test_f_regression_too_few_samples,
#          test_f_regression_shape_mismatch, test_chi2_negative_value_error,
#          test_chi2_empty_input, test_chi2_shape_mismatch

# REQ-4 p-value parity — the divergence/re-audit oracle suite (29 tests, all GREEN
# post-#1417): F-SF small-p (tail to ~1e-23)/large-p/moderate/p≈0.5/varied-df +
# chi2 chdtrc parity:
cargo test -p ferrolearn-preprocess --test divergence_feature_scoring
#   REQ-4 F-SF: divergence_f_classif_pvalues, divergence_f_regression_pvalues,
#               divergence_f_classif_pvalue_significant, divergence_f_regression_pvalue_nonsignificant,
#               reaudit_fclassif_smallp_df1_1_df2_4, reaudit_fclassif_smallp_df1_2_df2_3,
#               reaudit_fclassif_smallp_df1_4_df2_3, reaudit_fclassif_tinytail_df1_2_df2_48 (~1e-23),
#               reaudit_fclassif_smallp_df1_4_df2_20, reaudit_fclassif_midp_df1_2_df2_3,
#               reaudit_fclassif_phalf_df1_1_df2_4, reaudit_fclassif_largep_df1_2_df2_3(,_b),
#               reaudit_fclassif_df1_1_df2_18, reaudit_fregression_n5_df2_3,
#               reaudit_fregression_n20_df2_18, reaudit_fregression_n50_df2_48_phalf
#   REQ-4 chi2: greenguard_chi2_pvalues, reaudit_chi2_3class_df2_pvalues
cargo clippy -p ferrolearn-preprocess --all-targets -- -D warnings
cargo fmt --all --check

# REQ-1 oracle gate — f_classif F-statistic (df=2,3); p-value DIVERGES (REQ-4):
python3 -c "
import numpy as np; from sklearn.feature_selection import f_classif
X=np.array([[1,2,3],[4,5,6],[7,8,1],[2,1,9],[5,3,2],[8,9,4]],float)
y=np.array([0,0,1,1,2,2]); f,p=f_classif(X,y); print('F',f,'p',p)"
#   -> F [1.11627907 0.20212766 0.16883117]   (ferrolearn matches to ~1e-15)
#   -> p [0.43412099 0.82727270 0.85215439]   (ferrolearn matches to ~13-15 sig figs — REQ-4 SHIPPED)

# REQ-2 oracle gate — f_regression F-statistic (df=1,4):
python3 -c "
import numpy as np; from sklearn.feature_selection import f_regression
X=np.array([[1,2,3],[4,5,6],[7,8,1],[2,1,9],[5,3,2],[8,9,4]],float)
y=np.array([1.5,2.0,3.5,4.0,5.5,6.0]); f,p=f_regression(X,y); print('F',f,'p',p)"
#   -> F [3.01785714 0.57791853 0.04359837]   (ferrolearn matches to ~1e-15)
#   -> p [0.15735256 0.48947183 0.84480465]   (ferrolearn matches to ~13-15 sig figs — REQ-4 SHIPPED)

# REQ-3 oracle gate — chi2 statistic (df=1); p-value MATCHES to ~1e-9:
python3 -c "
import numpy as np; from sklearn.feature_selection import chi2
X=np.array([[1,0,3],[0,2,1],[2,1,0],[1,3,2]],float)
y=np.array([0,1,0,1]); s,p=chi2(X,y); print('stat',s,'p',p)"
#   -> stat [1. 2.66666667 0.]       (ferrolearn matches to ~1e-15)
#   -> p    [0.31731051 0.10247043 1.]   (ferrolearn matches to ~1e-9)

# REQ-4 verification gate — the F<->beta mapping AND the rewritten CF both agree:
python3 -c "
from scipy import special
F=1.1162790697674418; d1=2; d2=3; z=d2/(d2+d1*F)
print('betainc', special.betainc(d2/2,d1/2,z), 'fdtrc', special.fdtrc(d1,d2,F))"
#   -> betainc 0.4341209935... fdtrc 0.4341209935...  (ferrolearn's rewritten beta CF now matches)
```

The in-module `#[test]`s pin REQ-1/2/3 (the statistics) and REQ-5 (every error
path); the in-module `test_*_p_values_bounded` only assert `p ∈ [0,1]`, but REQ-4 is
established by the `tests/divergence_feature_scoring.rs` oracle suite above, which
pins the F-SF and chi2 p-values against scipy (29 tests GREEN post-#1417). No green
command establishes REQ-6 and REQ-8..REQ-11 (`f_regression center`/`force_finite`, sparse
`chi2`, `mutual_info_*`, PyO3, ferray).

## Blockers

REQ-1 (`f_classif` F-statistic), REQ-2 (`f_regression` F-statistic), REQ-3 (`chi2`
statistic), REQ-4 (p-value parity), REQ-5 (error / parameter contracts), and
REQ-7 (`r_regression`) are
SHIPPED — the module is compiled (`lib.rs:96`) and re-exported (`lib.rs:173-175`, the
grandfathered boundary consumer), the statistics match the live oracle to ~1e-15,
the p-values match scipy `fdtrc`/`chdtrc`, and the tests are green.

RESOLVED:

- #1417 — REQ-4 (was DIV-1): the F-distribution p-value path was wrong (the
  hand-rolled `regularized_incomplete_beta` continued fraction diverged from scipy
  `fdtrc` by up to ~0.29 absolute). A fixer rewrote `regularized_incomplete_beta`
  (`feature_scoring.rs:549`) and added the `betacf` Lentz continued fraction
  (`:574`, Numerical Recipes §6.4); the F-SF p-values now match `fdtrc` / `stats.f.sf`
  to ~13-15 significant figures across small-p (deep tail to ~1e-23), large-p,
  moderate, p≈0.5 and varied `df1∈{1,2,4}`/`df2∈{3..48}` regimes (29 oracle tests in
  `tests/divergence_feature_scoring.rs`, all GREEN). The chi-squared path was already
  correct (~1e-9). RESOLVED.

The remaining REQs are NOT-STARTED, each filed as a `-l blocker` issue against
tracking issue #1416:

- #1418 — REQ-6: no `center` / `force_finite` parameters on `f_regression`
  (`sklearn/feature_selection/_univariate_selection.py:405`, `:369-381`, `:447-461`).
- #1420 — REQ-8: no sparse (CSR) `chi2` path (`:264`, `:275`).
- #1421 — REQ-9: no `mutual_info_classif` / `mutual_info_regression`
  (separate `_mutual_info.py` unit — out of this file's scope, not-translated).
- #1422 — REQ-10: no PyO3 `f_classif` / `f_regression` / `chi2` binding in
  `ferrolearn-python`.
- #1423 — REQ-11: scoring fns + SF helpers compute on `ndarray` / `num_traits`,
  not ferray (R-SUBSTRATE-1/2).
