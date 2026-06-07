"""Additional divergence guards for ferrolearn.Binarizer (#1131) not covered by
``divergence_binarizer.py``.

These pin two observable divergences from ``sklearn.preprocessing.Binarizer``
(``sklearn/preprocessing/_data.py:2177-2306``) that the original 17-test suite
misses. Verification model B (goal.md): every expected value is computed by the
LIVE sklearn 1.5.2 oracle in the same test (R-CHAR-3); none is literal-copied
from the ferrolearn side.

Run (system python3 has sklearn 1.5.2 — the oracle):
    PYTHONPATH=python python3 -m pytest tests/divergence_binarizer_extra.py -q
"""

import numpy as np
import pytest
from sklearn.preprocessing import Binarizer as SkBinarizer

import ferrolearn as fl

_X = np.array(
    [
        [1.0, -1.0, 2.0],
        [2.0, 0.0, 0.0],
        [0.0, 1.0, -1.0],
    ]
)


# ---------------------------------------------------------------------------
# Divergence #2213 (R-DEV-2): Binarizer is STATELESS — transform must work
# WITHOUT a prior fit. sklearn's Binarizer carries the ``stateless`` tag
# (``_data.py:2308`` ``_more_tags`` -> ``{"stateless": True}``) and its class
# docstring (``_data.py:2306``) states: "This estimator is stateless and does
# not need to be fitted. However, we recommend to call fit_transform ...". The
# rendered doctest (``_data.py``) calls ``Binarizer().fit(X)`` then
# ``transform(X)`` but transform on a never-fitted instance ALSO works because
# ``transform`` only reads ``self.threshold``/``self.copy`` (no fitted state).
#
# ferrolearn's ``_TransformerWrapper.transform`` (``_extras.py:2010``) reads
# ``self._rs``, which is created only in ``fit`` (``_extras.py:2005``), so a
# never-fitted ``transform`` raises ``AttributeError`` instead of binarizing.
# ---------------------------------------------------------------------------


def test_transform_without_fit_is_stateless_like_sklearn():
    """fl.Binarizer().transform(X) (NO fit) must binarize, like sklearn.

    Oracle: sklearn's Binarizer is stateless (``_data.py:2308``
    ``{"stateless": True}``); ``Binarizer().transform(X)`` binarizes with the
    constructor threshold and never requires a prior fit. The expected output is
    computed by the live oracle on the SAME, never-fitted instance.
    """
    sk = SkBinarizer(threshold=0.0).transform(_X)  # no fit on sk either
    fr = fl.Binarizer(threshold=0.0).transform(_X)  # ferrolearn: no fit
    np.testing.assert_array_equal(np.asarray(fr), sk)


# ---------------------------------------------------------------------------
# Divergence #2214 (R-DEV-3): output dtype for an INTEGER-dtype input. sklearn's
# ``transform`` -> ``binarize`` -> ``check_array`` (``_data.py:2295``,
# ``_data.py:2160``) does NOT force float: ``check_array`` keeps an integer X as
# integer (dtype="numeric" preserves int dtypes), so ``fit_transform`` on an
# int64 X returns an int64 array. ferrolearn's wrapper coerces every input
# through ``_f64`` (``_extras.py:90`` ``np.ascontiguousarray(a, dtype=float64)``)
# and the Rust core returns ``PyArray2<f64>``, so the output is always float64.
# ---------------------------------------------------------------------------


def test_int_input_output_dtype_matches_sklearn():
    """fl.Binarizer().fit_transform(int64 X) dtype == sklearn (preserves int64).

    Oracle: sklearn ``check_array`` preserves integer dtypes, so the live oracle
    returns ``np.int64`` for an ``int64`` input (computed in-test, not copied).
    """
    Xi = np.array([[1, -1, 2], [0, 5, 0]], dtype=np.int64)
    sk = SkBinarizer(threshold=0).fit_transform(Xi)
    fr = fl.Binarizer(threshold=0).fit_transform(Xi)
    # Value parity holds; the DTYPE is the divergence.
    np.testing.assert_array_equal(np.asarray(fr), sk.astype(np.asarray(fr).dtype))
    assert np.asarray(fr).dtype == sk.dtype, (
        f"ferrolearn output dtype {np.asarray(fr).dtype} != sklearn {sk.dtype}"
    )
