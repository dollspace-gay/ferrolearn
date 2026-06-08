"""Divergence guards for the ferrolearn-python ``Normalizer`` binding (#1146).

Targets: ``ferrolearn-python/src/extras.rs`` (``_RsNormalizer``) + the Python
wrapper ``ferrolearn-python/python/ferrolearn/_extras.py`` (``class Normalizer``),
surfacing ``ferrolearn_preprocess::{Normalizer, FittedNormalizer, NormType}`` to
Python as ``ferrolearn.Normalizer``, mirroring ``sklearn.preprocessing.Normalizer``
(``sklearn/preprocessing/_data.py:1980-2110``).

Verification model B (goal.md): every expected value is computed by the LIVE
sklearn 1.5.2 oracle in the same test and compared against ``import ferrolearn``.
No expected value is literal-copied from the ferrolearn side (R-CHAR-3).

sklearn oracle anchors:
  - ``Normalizer.__init__(self, norm="l2", *, copy=True)`` (``_data.py:2058``):
    ``norm`` is POSITIONAL-OR-KEYWORD, ``copy`` is keyword-only.
  - ``_parameter_constraints {norm: StrOptions({"l1","l2","max"}), copy: ["boolean"]}``
    (``_data.py:2053-2056``): a bad ``norm`` string -> ``InvalidParameterError``
    (a ValueError subclass), raised at ``fit``.
  - STATELESS: ``_more_tags() -> {"stateless": True}`` (``_data.py:2110``);
    ``transform`` works WITHOUT a prior ``fit``.
  - ``transform`` -> ``normalize(X, norm, axis=1)`` -> ``check_array(dtype=FLOAT_DTYPES)``
    (``_data.py:2104-2108``): floating dtype PRESERVED, integer UPCAST to float64.
  - ``check_array(force_all_finite=True)`` rejects NaN/+-inf INPUT.
  - zero-norm row left unchanged (``_handle_zeros_in_scale``, ``_data.py:1968``).
"""

import inspect

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer as SkNormalizer

import ferrolearn as fl

# Multi-row fixture: mixed signs, a zero-norm row, and distinct per-row norms so
# l1/l2/max all produce observably different outputs.
_X = np.array(
    [
        [1.0, 2.0, 2.0],
        [0.0, 3.0, 4.0],
        [-4.0, 0.0, 3.0],
        [0.0, 0.0, 0.0],
    ]
)


# ---------------------------------------------------------------------------
# Value parity vs the live sklearn oracle across the three norms.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("norm", ["l1", "l2", "max"])
def test_fit_transform_matches_sklearn(norm):
    """fl.Normalizer(norm=n).fit_transform(X) == sklearn, element-wise (tight).

    Oracle: live sklearn 1.5.2 Normalizer on the same (X, norm). l2 is the
    default; covers l1/l2/max on a mixed-sign, multi-row fixture.
    """
    fr = fl.Normalizer(norm=norm).fit_transform(_X)
    sk = SkNormalizer(norm=norm).fit_transform(_X)
    assert fr.shape == sk.shape == _X.shape
    np.testing.assert_allclose(fr, sk, rtol=0, atol=1e-12)


def test_default_norm_is_l2_and_matches_sklearn():
    """Default norm is 'l2' in both signatures and produces the same output.

    Oracle: sklearn ``__init__`` default ``norm='l2'`` (``_data.py:2058``).
    """
    sk_default = inspect.signature(SkNormalizer.__init__).parameters["norm"].default
    fl_default = inspect.signature(fl.Normalizer.__init__).parameters["norm"].default
    assert sk_default == "l2"
    assert fl_default == sk_default

    fr = fl.Normalizer().fit_transform(_X)
    sk = SkNormalizer().fit_transform(_X)
    np.testing.assert_allclose(fr, sk, rtol=0, atol=1e-12)


def test_norm_is_positional_or_keyword_like_sklearn():
    """norm is positional-or-keyword in BOTH; copy is keyword-only in BOTH.

    Oracle: ``def __init__(self, norm="l2", *, copy=True)`` (``_data.py:2058``).
    """
    sk_params = inspect.signature(SkNormalizer.__init__).parameters
    fl_params = inspect.signature(fl.Normalizer.__init__).parameters
    POK = inspect.Parameter.POSITIONAL_OR_KEYWORD
    KO = inspect.Parameter.KEYWORD_ONLY
    assert sk_params["norm"].kind == POK
    assert fl_params["norm"].kind == POK
    assert sk_params["copy"].kind == KO
    assert fl_params["copy"].kind == KO
    # Positional norm works in both.
    np.testing.assert_allclose(
        fl.Normalizer("l1").fit_transform(_X),
        SkNormalizer("l1").fit_transform(_X),
        rtol=0,
        atol=1e-12,
    )


# ---------------------------------------------------------------------------
# Stateless: transform WITHOUT a prior fit works and matches sklearn (#2213).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("norm", ["l1", "l2", "max"])
def test_transform_without_fit_is_stateless_like_sklearn(norm):
    """fl.Normalizer().transform(X) (no fit) == sklearn (stateless, #2213).

    Oracle: sklearn ``_more_tags() -> {"stateless": True}`` (``_data.py:2110``);
    ``Normalizer().transform(X)`` works without ever calling ``fit``.
    """
    fr = fl.Normalizer(norm=norm).transform(_X)
    sk = SkNormalizer(norm=norm).transform(_X)
    np.testing.assert_allclose(fr, sk, rtol=0, atol=1e-12)


# ---------------------------------------------------------------------------
# Dtype parity: float32 preserved, float64 preserved, int64 -> float64 (#2214-analog).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("in_dtype", [np.float64, np.float32, np.int64])
def test_output_dtype_matches_sklearn(in_dtype):
    """Output dtype matches sklearn exactly per FLOAT_DTYPES (#2214-analog).

    Oracle: sklearn ``transform`` -> ``normalize`` -> ``check_array(dtype=
    FLOAT_DTYPES)`` (``_data.py:2104-2108``): float32->float32, float64->float64,
    int64->float64 (integer UPCAST, NOT preserved — unlike Binarizer).
    """
    x = np.array([[3, 4, 0], [1, 0, 0]], dtype=in_dtype)
    fr = fl.Normalizer().fit_transform(x)
    sk = SkNormalizer().fit_transform(x)
    assert fr.dtype == sk.dtype, f"{fr.dtype} != sklearn {sk.dtype}"
    np.testing.assert_allclose(fr, sk, rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------------
# Error parity: NaN/+-inf INPUT -> ValueError (Normalizer does NOT allow-nan).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_input_raises_valueerror_like_sklearn(bad):
    """NaN/+-inf INPUT -> ValueError in BOTH sklearn and ferrolearn.

    Oracle: sklearn ``check_array(force_all_finite=True)`` raises ValueError on a
    non-finite X (``_data.py:2104`` -> ``utils/validation.py:1063``).
    """
    x_bad = np.array([[0.5, bad], [1.0, 2.0]])
    with pytest.raises(ValueError):
        SkNormalizer().fit_transform(x_bad)
    with pytest.raises(ValueError):
        fl.Normalizer().fit_transform(x_bad)


# ---------------------------------------------------------------------------
# Zero-norm row left as zeros in BOTH (_handle_zeros_in_scale).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("norm", ["l1", "l2", "max"])
def test_zero_norm_row_left_as_zeros_like_sklearn(norm):
    """A zero-norm row stays all-zeros in BOTH.

    Oracle: sklearn ``_handle_zeros_in_scale`` maps a zero norm to 1 so the row is
    divided by 1, i.e. left unchanged (``_data.py:1968``).
    """
    x = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 2.0]])
    fr = fl.Normalizer(norm=norm).fit_transform(x)
    sk = SkNormalizer(norm=norm).fit_transform(x)
    np.testing.assert_allclose(fr, sk, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(fr[0], np.zeros(3))


# ---------------------------------------------------------------------------
# Bad norm string -> ValueError in BOTH (StrOptions).
# ---------------------------------------------------------------------------


def test_bad_norm_string_raises_valueerror_like_sklearn():
    """fl.Normalizer(norm='l3') raises ValueError, matching sklearn StrOptions.

    Oracle: sklearn ``_parameter_constraints {norm: StrOptions({"l1","l2","max"})}``
    (``_data.py:2054``) raises ``InvalidParameterError`` (⊂ ValueError) at fit.
    sklearn raises at fit (not construct); verify SOME call raises ValueError in
    ferrolearn too (construction must not crash before that).
    """
    # sklearn: construction is fine, fit raises.
    sk = SkNormalizer(norm="l3")
    with pytest.raises(ValueError):
        sk.fit(_X)
    with pytest.raises(ValueError):
        SkNormalizer(norm="l3").fit_transform(_X)

    # ferrolearn: construction must not raise; some call (fit/transform) raises
    # ValueError like sklearn.
    fr = fl.Normalizer(norm="l3")  # construct OK
    with pytest.raises(ValueError):
        fr.fit_transform(_X)
    with pytest.raises(ValueError):
        fl.Normalizer(norm="l3").transform(_X)


# ---------------------------------------------------------------------------
# Param plumbing: get_params / set_params / clone round-trip norm + copy.
# ---------------------------------------------------------------------------


def test_get_params_matches_sklearn_keys_and_defaults():
    """get_params() exposes norm + copy with sklearn's defaults.

    Oracle: sklearn ``Normalizer().get_params()`` is ``{'copy': True,
    'norm': 'l2'}`` (from ``__init__``, ``_data.py:2058``).
    """
    sk_params = SkNormalizer().get_params()
    fl_params = fl.Normalizer().get_params()
    assert set(fl_params.keys()) == set(sk_params.keys()) == {"norm", "copy"}
    assert fl_params["norm"] == sk_params["norm"] == "l2"
    assert fl_params["copy"] == sk_params["copy"] is True


def test_set_params_round_trips():
    """set_params(norm=, copy=) updates the params (sklearn BaseEstimator)."""
    est = fl.Normalizer()
    est.set_params(norm="l1", copy=False)
    assert est.norm == "l1"
    assert est.copy is False
    fr = est.fit_transform(_X)
    sk = SkNormalizer(norm="l1", copy=False).fit_transform(_X)
    np.testing.assert_allclose(fr, sk, rtol=0, atol=1e-12)


def test_clone_round_trips_params():
    """sklearn.base.clone(fl.Normalizer(norm='l1', copy=False)) preserves params.

    clone() uses get_params/__init__, exercising the keyword-only ABI parity.
    """
    cloned = clone(fl.Normalizer(norm="l1", copy=False))
    assert cloned.norm == "l1"
    assert cloned.copy is False
    fr = cloned.fit_transform(_X)
    sk = clone(SkNormalizer(norm="l1", copy=False)).fit_transform(_X)
    np.testing.assert_allclose(fr, sk, rtol=0, atol=1e-12)


def test_copy_false_true_identical_output_matches_sklearn():
    """copy=True/copy=False produce identical output, matching sklearn.

    sklearn ``copy=False`` does in-place row normalization (an optimization); the
    OUTPUT VALUES are identical to ``copy=True``. ferrolearn's Transform always
    allocates, so copy is an accept-and-document no-op (normalizer.rs REQ-5).
    """
    fr_t = fl.Normalizer(norm="l2", copy=True).fit_transform(_X)
    fr_f = fl.Normalizer(norm="l2", copy=False).fit_transform(_X)
    sk = SkNormalizer(norm="l2").fit_transform(_X)
    np.testing.assert_allclose(fr_t, fr_f, rtol=0, atol=0)
    np.testing.assert_allclose(fr_t, sk, rtol=0, atol=1e-12)


# ---------------------------------------------------------------------------
# Pipeline integration parity.
# ---------------------------------------------------------------------------


def test_pipeline_fit_transform_matches_sklearn():
    """make_pipeline(fl.Normalizer()).fit_transform(X) == sklearn pipeline."""
    fr = make_pipeline(fl.Normalizer()).fit_transform(_X)
    sk = make_pipeline(SkNormalizer()).fit_transform(_X)
    np.testing.assert_allclose(fr, sk, rtol=0, atol=1e-12)
