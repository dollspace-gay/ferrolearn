"""Divergence guards: ferrolearn.Ridge multi-output path drops the `solver=`.

Targets the uncommitted REQ-RIDGE-PARAMS surface (#2132): the single-output
path threads `solver` into `_RsRidge` (which maps auto/cholesky/svd ->
RidgeSolver and raises NotImplementedError for everything else, #2133), but the
multi-output (2-D `y`, non-positive) path constructs
`_RsRidgeMultiOutput(alpha=..., fit_intercept=...)` only
(`ferrolearn-python/python/ferrolearn/_regressors.py:299-301`), and the binding
`RsRidgeMultiOutput::new(alpha, fit_intercept)`
(`ferrolearn-python/src/regressors.rs:407-414`) cannot receive `solver` at all.

Consequence (vs sklearn 1.5.2 `sklearn/linear_model/_ridge.py`):
  * The SAME `solver` string is rejected on a 1-D `y` but SILENTLY ACCEPTED on
    a 2-D `y` -> a single-vs-multi internal contract inconsistency.
  * A solver string sklearn REJECTS (`InvalidParameterError`, validated against
    the `_parameter_constraints` `StrOptions` set at `_ridge.py:885`) is
    silently swallowed on the multi-output path.

Verification model B (goal.md): every expected behavior is established by the
LIVE sklearn 1.5.2 oracle in-test, then compared against `import ferrolearn`.
No expected value is literal-copied from the ferrolearn side (R-CHAR-3).

Run with SYSTEM python3 (sklearn 1.5.2):
    PYTHONPATH=python python3 -m pytest tests/divergence_ridge_multioutput_solver.py -q

Tracking: crosslink #2134.
"""

import numpy as np
import pytest

import ferrolearn as fl
from sklearn.linear_model import Ridge as SkRidge

# A small well-conditioned multi-output problem. sklearn handles 'lsqr' on BOTH
# the 1-D and the 2-D target and (being a strictly-convex problem) converges to
# the SAME optimum as the closed-form 'cholesky' solver, so the divergence is
# purely about the CONTRACT (accept/reject), not about the fitted value.
_X = np.array(
    [
        [1.0, 2.0],
        [3.0, 1.0],
        [2.0, 5.0],
        [4.0, 2.0],
        [1.0, 1.0],
    ]
)
_Y = np.array(
    [
        [1.0, 2.0],
        [2.0, 1.0],
        [3.0, 0.0],
        [1.0, 2.0],
        [0.0, 1.0],
    ]
)


def test_ridge_solver_single_vs_multi_consistency_lsqr():
    """`solver='lsqr'` must be handled CONSISTENTLY across 1-D and 2-D `y`.

    sklearn accepts 'lsqr' in both cases (`_ridge.py:885` lists it in the valid
    StrOptions set; it routes to `_solve_lsqr`), so the oracle SUCCEEDS on both.

    ferrolearn's documented #2133 stance is to RAISE NotImplementedError for the
    iterative solvers. Whatever the chosen behavior, it must be the SAME for 1-D
    and 2-D `y` (single-output and multi-output cannot disagree on whether a
    given `solver` string is supported). Today the single path raises and the
    multi path silently succeeds -> inconsistent.
    """
    y1 = _Y[:, 0]

    # Oracle: sklearn accepts lsqr for both shapes.
    SkRidge(alpha=1.0, solver="lsqr").fit(_X, y1)
    SkRidge(alpha=1.0, solver="lsqr").fit(_X, _Y)

    # Probe ferrolearn's single-output decision.
    single_raised = False
    try:
        fl.Ridge(alpha=1.0, solver="lsqr").fit(_X, y1)
    except NotImplementedError:
        single_raised = True

    # Probe ferrolearn's multi-output decision.
    multi_raised = False
    try:
        fl.Ridge(alpha=1.0, solver="lsqr").fit(_X, _Y)
    except NotImplementedError:
        multi_raised = True

    # The two paths must agree on whether 'lsqr' is supported. They do not:
    # single raises, multi silently succeeds.
    assert single_raised == multi_raised, (
        f"single-output raised={single_raised} but multi-output raised="
        f"{multi_raised} for the SAME solver='lsqr' (sklearn accepts both); "
        "the multi-output path drops `solver` (_regressors.py:299 / "
        "regressors.rs:419)"
    )


def test_ridge_invalid_solver_multioutput_rejected_like_sklearn():
    """An invalid `solver` string must be REJECTED on the multi-output path.

    sklearn validates `solver` against a fixed StrOptions set (`_ridge.py:885`)
    and raises `InvalidParameterError` (a `ValueError` subclass) for an unknown
    value, regardless of `y` dimensionality. ferrolearn's single-output path
    likewise rejects it (NotImplementedError). The multi-output path must reject
    it too; today it silently accepts a bogus solver because
    `_RsRidgeMultiOutput` never receives `solver`.
    """
    bogus = "definitely-not-a-real-solver"

    # Oracle: sklearn raises on the bogus solver for the 2-D target.
    with pytest.raises(ValueError):
        SkRidge(alpha=1.0, solver=bogus).fit(_X, _Y)

    # ferrolearn must NOT silently accept a solver string that neither sklearn
    # nor its own single-output path tolerates.
    with pytest.raises((ValueError, NotImplementedError)):
        fl.Ridge(alpha=1.0, solver=bogus).fit(_X, _Y)
