"""Divergence guards for the ferrolearn-python ``Binarizer`` binding (#1131).

Targets: ``ferrolearn-python/src/extras.rs`` (``_RsBinarizer``) + the Python
wrapper ``ferrolearn-python/python/ferrolearn/_extras.py`` (``class Binarizer``),
surfacing the Fit-complete ``ferrolearn_preprocess::{Binarizer, FittedBinarizer}``
(crate commit 832401d58) to Python as ``ferrolearn.Binarizer``, mirroring
``sklearn.preprocessing.Binarizer`` (``sklearn/preprocessing/_data.py:2177-2306``).

Verification model B (goal.md): every expected value is computed by the LIVE
sklearn 1.5.2 oracle in the same test and compared against ``import ferrolearn``.
No expected value is literal-copied from the ferrolearn side (R-CHAR-3).

sklearn oracle anchors:
  - ``Binarizer.__init__(self, *, threshold=0.0, copy=True)`` (``_data.py:2253``).
  - dense binarize is strict ``X > threshold`` (``_data.py:2170-2173``): a value
    EQUAL to the threshold maps to 0.
  - ``fit`` "Only validates estimator's parameters" (``_data.py:2257-2278``); a
    non-finite threshold is ACCEPTED at fit (``_parameter_constraints
    {threshold: [Real]}`` bare type check, ``_data.py:2249``) and REJECTED by the
    free ``binarize``'s open ``Interval(Real, None, None, closed="neither")``
    (``_data.py:2114``) that ``transform`` calls.
  - ``check_array(force_all_finite=True)`` rejects NaN/+-inf INPUT.
"""

import inspect

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.preprocessing import Binarizer as SkBinarizer

import ferrolearn as fl

# Mixed-sign fixture (values straddle every tested threshold, incl. exact
# boundary points so the strict-greater rule is observable).
_X = np.array(
    [
        [1.0, -1.0, 2.0],
        [2.0, 0.0, 0.0],
        [0.0, 1.0, -1.0],
        [-0.5, 0.5, -2.0],
    ]
)


# ---------------------------------------------------------------------------
# Value parity vs the live sklearn oracle across thresholds + boundary.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("threshold", [0.0, 0.5, -1.0])
def test_fit_transform_matches_sklearn(threshold):
    """fl.Binarizer(threshold=t).fit_transform(X) == sklearn, element-wise.

    Oracle: live sklearn 1.5.2 Binarizer on the same (X, threshold). The dense
    rule is strict ``X > threshold`` (``_data.py:2170-2173``); covers thresholds
    0.0, 0.5, -1.0 on a mixed-sign fixture.
    """
    fr = fl.Binarizer(threshold=threshold).fit_transform(_X)
    sk = SkBinarizer(threshold=threshold).fit_transform(_X)
    assert fr.shape == sk.shape == _X.shape
    np.testing.assert_array_equal(fr, sk)
    # Output is exactly {0.0, 1.0}.
    assert set(np.unique(fr).tolist()) <= {0.0, 1.0}


def test_strict_greater_boundary_value_equal_threshold_maps_to_zero():
    """A value EQUAL to the threshold maps to 0 (strict greater-than).

    Oracle: sklearn's dense binarize is ``cond = X > threshold`` then ``X[~cond]
    = 0`` (``_data.py:2170-2173``), so ``threshold`` itself -> 0.
    """
    x = np.array([[0.4999, 0.5, 0.5001]])
    fr = fl.Binarizer(threshold=0.5).fit_transform(x)
    sk = SkBinarizer(threshold=0.5).fit_transform(x)
    np.testing.assert_array_equal(fr, sk)
    # The exact-boundary middle column is 0 in both (strict >).
    assert sk[0, 1] == 0.0
    assert fr[0, 1] == 0.0


def test_fit_then_transform_matches_sklearn():
    """fl.Binarizer().fit(X).transform(X2) matches sklearn (fit-then-transform).

    sklearn ``fit`` only validates (``_data.py:2257-2278``); ``transform`` on a
    DIFFERENT array of the same n_features binarizes by the stored threshold.
    """
    x2 = np.array([[3.0, -3.0, 0.25], [0.0, 0.1, 5.0]])
    fr = fl.Binarizer(threshold=0.2).fit(_X).transform(x2)
    sk = SkBinarizer(threshold=0.2).fit(_X).transform(x2)
    np.testing.assert_array_equal(fr, sk)


def test_default_threshold_is_zero():
    """Default threshold is 0.0 in both signatures and produces the same output.

    Oracle: sklearn ``__init__`` default ``threshold=0.0`` (``_data.py:2253``).
    """
    sk_default = inspect.signature(SkBinarizer.__init__).parameters["threshold"].default
    fl_default = inspect.signature(fl.Binarizer.__init__).parameters["threshold"].default
    assert sk_default == 0.0
    assert fl_default == sk_default

    fr = fl.Binarizer().fit_transform(_X)
    sk = SkBinarizer().fit_transform(_X)
    np.testing.assert_array_equal(fr, sk)


# ---------------------------------------------------------------------------
# Error parity: NaN/+-inf INPUT and non-finite THRESHOLD both raise ValueError.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_input_raises_valueerror_like_sklearn(bad):
    """NaN/+-inf INPUT -> ValueError in BOTH sklearn and ferrolearn.

    Oracle: sklearn ``check_array(force_all_finite=True)`` raises ValueError on a
    non-finite X (``_data.py:2301`` -> ``utils/validation.py:1063``).
    """
    x_bad = np.array([[0.5, bad], [1.0, 2.0]])
    with pytest.raises(ValueError):
        SkBinarizer(threshold=0.0).fit_transform(x_bad)
    with pytest.raises(ValueError):
        fl.Binarizer(threshold=0.0).fit_transform(x_bad)


@pytest.mark.parametrize("bad_threshold", [np.nan, np.inf, -np.inf])
def test_nonfinite_threshold_rejected_at_transform_like_sklearn(bad_threshold):
    """A non-finite THRESHOLD raises ValueError at transform in BOTH (#2208).

    Oracle: ``transform`` calls the free ``binarize`` whose ``@validate_params``
    open ``Interval(Real, None, None, closed="neither")`` (``_data.py:2114``)
    EXCLUDES NaN/+-inf -> ``InvalidParameterError`` (a ValueError subclass).
    """
    with pytest.raises(ValueError):
        SkBinarizer(threshold=bad_threshold).fit_transform(_X)
    with pytest.raises(ValueError):
        fl.Binarizer(threshold=bad_threshold).fit_transform(_X)


def test_nonfinite_threshold_accepted_at_fit_like_sklearn():
    """``fit`` ALONE accepts a non-finite threshold in BOTH (#2209).

    Oracle: sklearn ``Binarizer.fit`` only runs ``_validate_data`` on the DATA;
    ``_parameter_constraints {threshold: [Real]}`` (``_data.py:2249``) is a bare
    type check that ACCEPTS NaN/+-inf. Only ``transform`` rejects it. So
    ``Binarizer(threshold=nan).fit(X)`` must NOT raise in either library.
    """
    # sklearn accepts at fit (no exception).
    SkBinarizer(threshold=np.nan).fit(_X)
    SkBinarizer(threshold=np.inf).fit(_X)
    # ferrolearn must match: fit accepts the non-finite threshold.
    fl.Binarizer(threshold=np.nan).fit(_X)
    fl.Binarizer(threshold=np.inf).fit(_X)


# ---------------------------------------------------------------------------
# Param plumbing: get_params / set_params / clone round-trip threshold + copy.
# ---------------------------------------------------------------------------


def test_get_params_matches_sklearn_keys_and_defaults():
    """get_params() exposes threshold + copy with sklearn's defaults.

    Oracle: sklearn ``Binarizer().get_params()`` is
    ``{'copy': True, 'threshold': 0.0}`` (from ``__init__``, ``_data.py:2253``).
    """
    sk_params = SkBinarizer().get_params()
    fl_params = fl.Binarizer().get_params()
    assert set(fl_params.keys()) == set(sk_params.keys()) == {"threshold", "copy"}
    assert fl_params["threshold"] == sk_params["threshold"] == 0.0
    assert fl_params["copy"] == sk_params["copy"] is True


def test_set_params_round_trips():
    """set_params(threshold=, copy=) updates the params (sklearn BaseEstimator)."""
    est = fl.Binarizer()
    est.set_params(threshold=0.75, copy=False)
    assert est.threshold == 0.75
    assert est.copy is False
    # And the new threshold drives transform.
    fr = est.fit_transform(_X)
    sk = SkBinarizer(threshold=0.75, copy=False).fit_transform(_X)
    np.testing.assert_array_equal(fr, sk)


def test_clone_round_trips_params():
    """sklearn.base.clone(fl.Binarizer(threshold=0.5)) preserves the params.

    clone() uses get_params/__init__, exercising the keyword-only ABI parity.
    """
    cloned = clone(fl.Binarizer(threshold=0.5, copy=False))
    assert cloned.threshold == 0.5
    assert cloned.copy is False
    fr = cloned.fit_transform(_X)
    sk = clone(SkBinarizer(threshold=0.5, copy=False)).fit_transform(_X)
    np.testing.assert_array_equal(fr, sk)


def test_copy_false_true_identical_output_matches_sklearn():
    """copy=True/copy=False produce identical output, matching sklearn.

    sklearn ``copy=False`` does in-place binarization (an optimization); the
    OUTPUT VALUES are identical to ``copy=True``. ferrolearn's Transform always
    allocates, so copy is an accept-and-document no-op (binarizer.rs REQ-2).
    """
    fr_t = fl.Binarizer(threshold=0.3, copy=True).fit_transform(_X)
    fr_f = fl.Binarizer(threshold=0.3, copy=False).fit_transform(_X)
    sk = SkBinarizer(threshold=0.3).fit_transform(_X)
    np.testing.assert_array_equal(fr_t, fr_f)
    np.testing.assert_array_equal(fr_t, sk)
