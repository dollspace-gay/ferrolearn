"""Divergence pin: ferrolearn.Normalizer computes the row norm in float64 and
only casts the RESULT back to the input dtype, whereas sklearn computes the norm
AND the division IN the reduced input precision (float16/float32).

Targets:
  - ``ferrolearn-python/python/ferrolearn/_extras.py`` (``class Normalizer.transform``):
    the f64 Rust ABI normalizes in float64, then the wrapper does
    ``out.astype(in_dtype)`` (``_extras.py:2305-2314``). The norm is therefore a
    float64 norm, rounded once at the end.
  - sklearn ``normalize`` (``sklearn/preprocessing/_data.py:1933-1969``):
    ``X = check_array(..., dtype=supported_float_dtypes(xp))`` casts X to float16
    FIRST (``_data.py:1933-1938``), then ``norms = row_norms(X)`` (``:1965``) and
    ``X /= norms[:, None]`` (``:1969``) are computed ENTIRELY in float16. The norm
    is a float16 norm; every intermediate is rounded.

For float16 l2 the two rounding paths disagree by ~4.9e-4 (float16 ULPs), well
beyond float16 round-trip noise on the input — an observable value divergence.
(float32 l2 diverges by the same mechanism, ~6e-8 / float32 ULPs.)

Verification model B (goal.md R-CHAR-3): the expected value is the LIVE sklearn
1.5.2 oracle computed in this test; nothing is literal-copied from ferrolearn.

Tracking: #2215.
"""

import numpy as np
import pytest
from sklearn.preprocessing import Normalizer as SkNormalizer

import ferrolearn as fl

# Mixed-sign, multi-row fixture whose exact-in-float16 entries make the only
# source of disagreement the NORM computation precision (the entries themselves
# round-trip through float16 losslessly).
_X16 = np.array(
    [
        [1.0, 2.0, 3.0, 7.0],
        [4.0, 5.0, 6.0, 11.0],
        [-3.0, 2.0, -9.0, 4.0],
    ],
    dtype=np.float16,
)


@pytest.mark.skip(
    reason="divergence: ferrolearn.Normalizer normalizes in float64 then casts to "
    "float16, but sklearn computes row_norms + the division in float16 "
    "(_data.py:1933-1969); float16 l2 differs by ~4.9e-4; tracking #2215"
)
def test_float16_l2_normalized_in_reduced_precision_like_sklearn():
    """float16 l2 output must equal sklearn BIT-FOR-BIT (norm computed in float16).

    Oracle: sklearn ``normalize`` casts X to float16 (``_data.py:1933-1938``),
    then ``norms = row_norms(X)`` (``:1965``) and ``X /= norms`` (``:1969``) are
    float16 ops. ferrolearn does the whole normalization in float64 and casts the
    result back, so its float16 output rounds differently.
    """
    sk = SkNormalizer(norm="l2").fit_transform(_X16)
    fr = fl.Normalizer(norm="l2").fit_transform(_X16)

    assert fr.dtype == sk.dtype == np.float16
    # sklearn's float16-native normalization differs from ferrolearn's
    # float64-then-cast by more than half a float16 ULP on this fixture.
    np.testing.assert_array_equal(fr, sk)
