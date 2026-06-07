"""Divergence guards: ferrolearn.RANSACRegressor min_samples resolution/validation
diverges from sklearn 1.5.2 at the float/int/bool boundaries.

Targets:
  - ferrolearn-python/python/ferrolearn/_extras.py  (RANSACRegressor._validate,
    the Python-side min_samples resolution before the Rust ABI)
  - mirrored against sklearn/linear_model/_ransac.py:382-397 (fit-time min_samples
    resolution) and _ransac.py:260-286 (_parameter_constraints).

Verification model B (goal.md): every expected value is computed by the LIVE
sklearn 1.5.2 oracle IN THIS test and compared against `import ferrolearn`. No
expected value is literal-copied from the ferrolearn side (R-CHAR-3).

The wrapper resolves a float min_samples as ``ceil(min_samples * n_samples)``
UNCONDITIONALLY for any float in [0, 1] (_extras.py:843-849). sklearn's fit-time
branch (_ransac.py:389-392) uses STRICT inequalities:

    elif 0 < self.min_samples < 1:        # ceil branch
        min_samples = ceil(min_samples * n_samples)
    elif self.min_samples >= 1:           # absolute branch
        min_samples = self.min_samples

so the float endpoints 0.0 and 1.0 do NOT take the ceil branch, and the
``RealNotInt`` constraint accepts True (==1) but rejects 0.0-by-error and False.
"""

import numpy as np
import pytest

import ferrolearn as fl
from sklearn.linear_model import RANSACRegressor as SkRANSACRegressor


def _data(seed=0, n=20, p=3):
    rng = np.random.RandomState(seed)
    X = rng.rand(n, p)
    y = X @ np.arange(1.0, p + 1.0) + 0.5
    return X, y


def test_ransac_min_samples_float_one_resolves_like_sklearn():
    """min_samples=1.0: sklearn takes the ``>= 1`` branch (_ransac.py:391-392)
    -> subset size 1 (NOT ceil(1.0*n_samples)). ferrolearn does
    ceil(1.0 * n_samples) = n_samples (_extras.py:849), a different subset size
    and thus a different consensus fit. Expected drawn from the LIVE oracle."""
    X, y = _data()
    sk = SkRANSACRegressor(min_samples=1.0, random_state=0, max_trials=100).fit(X, y)
    fr = fl.RANSACRegressor(min_samples=1.0, random_state=0, max_trials=100).fit(X, y)
    # sklearn resolves to a size-1 subset (n_inliers reflects that); ferrolearn
    # resolves to the full set (n_inliers == n_samples). The inlier masks differ.
    np.testing.assert_array_equal(
        np.asarray(fr.inlier_mask_), sk.inlier_mask_,
        err_msg=(f"min_samples=1.0: sk n_inliers={sk.inlier_mask_.sum()}, "
                 f"fl n_inliers={int(np.asarray(fr.inlier_mask_).sum())}"),
    )


def test_ransac_min_samples_float_zero_matches_sklearn_rejection():
    """min_samples=0.0: sklearn's fit-time resolution leaves ``min_samples``
    unbound (neither ``0 < 0.0 < 1`` nor ``0.0 >= 1``; _ransac.py:389-392) and
    raises. ferrolearn takes the ``0 <= f <= 1`` ceil branch (_extras.py:844),
    resolves to ceil(0.0*n)=0, and FITS. sklearn rejects, ferrolearn accepts."""
    X, y = _data()
    # Establish the oracle behaviour: sklearn raises for min_samples=0.0.
    with pytest.raises(Exception):
        SkRANSACRegressor(min_samples=0.0, random_state=0, max_trials=100).fit(X, y)
    # ferrolearn must mirror sklearn: it should also reject 0.0.
    with pytest.raises(Exception):
        fl.RANSACRegressor(min_samples=0.0, random_state=0, max_trials=100).fit(X, y)


def test_ransac_min_samples_bool_true_matches_sklearn_acceptance():
    """min_samples=True: sklearn's ``RealNotInt``/``Integral`` constraint accepts
    True (== 1) and the ``>= 1`` branch resolves to a size-1 subset
    (_ransac.py:391) -> sklearn FITS. ferrolearn rejects True as a ValueError
    (_extras.py:828-835). sklearn accepts, ferrolearn rejects."""
    X, y = _data()
    # Oracle: sklearn accepts True and returns a fitted model.
    sk = SkRANSACRegressor(min_samples=True, random_state=0, max_trials=100).fit(X, y)
    assert hasattr(sk, "inlier_mask_")
    # ferrolearn must mirror sklearn: accept True and fit.
    fr = fl.RANSACRegressor(min_samples=True, random_state=0, max_trials=100).fit(X, y)
    assert hasattr(fr, "inlier_mask_")
