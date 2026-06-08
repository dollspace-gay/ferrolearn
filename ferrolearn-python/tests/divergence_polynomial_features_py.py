"""Divergence guards for the ferrolearn-python ``PolynomialFeatures`` binding
(#1188, REQ-10).

Targets: ``ferrolearn-python/src/extras.rs`` (``_RsPolynomialFeatures``) + the
Python wrapper ``ferrolearn-python/python/ferrolearn/_extras.py``
(``class PolynomialFeatures``), surfacing
``ferrolearn_preprocess::{PolynomialFeatures, FittedPolynomialFeatures}`` to
Python as ``ferrolearn.PolynomialFeatures``, mirroring
``sklearn.preprocessing.PolynomialFeatures``
(``sklearn/preprocessing/_polynomial.py:99-564``).

Verification model B (goal.md): every expected value is computed by the LIVE
sklearn 1.5.2 oracle in the same test and compared against ``import ferrolearn``.
No expected value is literal-copied from the ferrolearn side (R-CHAR-3).

sklearn oracle anchors:
  - ``PolynomialFeatures.__init__(self, degree=2, *, interaction_only=False,
    include_bias=True, order="C")`` (``_polynomial.py:201``): ``degree`` is
    POSITIONAL-OR-KEYWORD, the rest keyword-only.
  - ``_parameter_constraints`` keys = {degree, interaction_only, include_bias,
    order} (``_polynomial.py:194-199``).
  - STATEFUL: ``transform`` calls ``check_is_fitted`` (``_polynomial.py:430``);
    a pre-fit ``transform`` raises ``NotFittedError``.
  - fitted attrs ``n_features_in_`` (``:323``), ``n_output_features_`` (``:362``),
    ``powers_`` shape ``(n_output_features_, n_features_in_)`` (``:250-264``).
  - ``transform`` -> ``check_array(dtype=FLOAT_DTYPES)`` (``:432``): floating dtype
    PRESERVED, integer UPCAST to float64.
  - wrong feature count at transform -> ``ValueError`` (``X has N features, but
    PolynomialFeatures is expecting M``).

#2215-class f64-ABI caveat: for FLOAT32 input the products are computed in float64
(the f64 ABI) then cast back to float32; sklearn computes them in float32, so the
cast-back float32 VALUES can differ from sklearn's by ~1e-6 on high-degree terms.
float64 input is bit-exact. The float32 value test uses a TOLERANT comparison.
"""

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures as SkPoly

import ferrolearn as fl

# A multi-row, multi-feature fixture with distinct magnitudes so degree-2/3,
# interaction, and bias variants all produce observably different columns.
_X = np.array(
    [
        [1.0, 2.0, 3.0],
        [2.0, 0.5, -1.0],
        [-1.5, 4.0, 0.25],
        [3.0, -2.0, 1.0],
    ],
    dtype=np.float64,
)


# ---------------------------------------------------------------------------
# Value parity vs the live sklearn oracle
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("degree", [2, 3])
@pytest.mark.parametrize("interaction_only", [False, True])
@pytest.mark.parametrize("include_bias", [True, False])
def test_fit_transform_values_match_sklearn(degree, interaction_only,
                                            include_bias):
    sk = SkPoly(degree=degree, interaction_only=interaction_only,
                include_bias=include_bias)
    fr = fl.PolynomialFeatures(degree=degree, interaction_only=interaction_only,
                               include_bias=include_bias)
    sk_out = sk.fit_transform(_X)
    fr_out = fr.fit_transform(_X)
    assert fr_out.shape == sk_out.shape
    # float64 path is bit-exact column-for-column (same incremental build).
    np.testing.assert_allclose(fr_out, sk_out, rtol=0, atol=0)


def test_default_ctor_matches_sklearn():
    # default degree=2, interaction_only=False, include_bias=True
    sk_out = SkPoly().fit_transform(_X)
    fr_out = fl.PolynomialFeatures().fit_transform(_X)
    np.testing.assert_allclose(fr_out, sk_out, rtol=0, atol=0)


def test_degree_positional_or_keyword():
    # `degree` is positional-or-keyword in sklearn; positional must work.
    sk_out = SkPoly(3).fit_transform(_X)
    fr_out = fl.PolynomialFeatures(3).fit_transform(_X)
    np.testing.assert_allclose(fr_out, sk_out, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# Fitted attributes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("degree", [2, 3])
@pytest.mark.parametrize("interaction_only", [False, True])
@pytest.mark.parametrize("include_bias", [True, False])
def test_powers_matches_sklearn(degree, interaction_only, include_bias):
    sk = SkPoly(degree=degree, interaction_only=interaction_only,
                include_bias=include_bias).fit(_X)
    fr = fl.PolynomialFeatures(degree=degree, interaction_only=interaction_only,
                               include_bias=include_bias).fit(_X)
    assert fr.powers_.shape == sk.powers_.shape
    # exact integer match, entry-for-entry
    np.testing.assert_array_equal(fr.powers_, sk.powers_)


def test_n_output_features_matches_sklearn():
    for degree in (2, 3):
        sk = SkPoly(degree=degree).fit(_X)
        fr = fl.PolynomialFeatures(degree=degree).fit(_X)
        assert fr.n_output_features_ == sk.n_output_features_


def test_n_features_in_matches_ncols():
    sk = SkPoly().fit(_X)
    fr = fl.PolynomialFeatures().fit(_X)
    assert fr.n_features_in_ == sk.n_features_in_ == _X.shape[1]


# ---------------------------------------------------------------------------
# Stateful contract: pre-fit transform -> NotFittedError (both)
# ---------------------------------------------------------------------------

def test_prefit_transform_raises_not_fitted_both():
    with pytest.raises(NotFittedError):
        SkPoly().transform(_X)
    with pytest.raises(NotFittedError):
        fl.PolynomialFeatures().transform(_X)


def test_prefit_attribute_raises_not_fitted():
    fr = fl.PolynomialFeatures()
    with pytest.raises(NotFittedError):
        _ = fr.n_features_in_
    with pytest.raises(NotFittedError):
        _ = fr.powers_


# ---------------------------------------------------------------------------
# Wrong feature count at transform -> ValueError (both)
# ---------------------------------------------------------------------------

def test_transform_wrong_n_features_raises_both():
    sk = SkPoly().fit(_X)
    fr = fl.PolynomialFeatures().fit(_X)
    bad = _X[:, :2]  # 2 cols vs 3 fitted
    with pytest.raises(ValueError):
        sk.transform(bad)
    with pytest.raises(ValueError):
        fr.transform(bad)


# ---------------------------------------------------------------------------
# DTYPE parity (sklearn check_array(dtype=FLOAT_DTYPES))
# ---------------------------------------------------------------------------

def test_dtype_int_upcast_to_float64():
    xi = _X.astype(np.int64)
    sk_out = SkPoly().fit_transform(xi)
    fr_out = fl.PolynomialFeatures().fit_transform(xi)
    assert sk_out.dtype == np.float64
    assert fr_out.dtype == sk_out.dtype
    np.testing.assert_allclose(fr_out, sk_out, rtol=0, atol=0)


def test_dtype_float64_preserved_bitexact():
    sk = SkPoly().fit(_X)
    fr = fl.PolynomialFeatures().fit(_X)
    sk_out = sk.transform(_X)
    fr_out = fr.transform(_X)
    assert fr_out.dtype == sk_out.dtype == np.float64
    np.testing.assert_allclose(fr_out, sk_out, rtol=0, atol=0)


def test_dtype_float32_preserved_tolerant():
    # #2215-class caveat: float32 products computed in float64 then cast back, so
    # VALUES match sklearn's float32 output only to ~1e-5 (NOT bit-exact). The
    # DTYPE, however, is preserved exactly (float32->float32).
    xf = _X.astype(np.float32)
    sk_out = SkPoly(degree=3).fit_transform(xf)
    fr_out = fl.PolynomialFeatures(degree=3).fit_transform(xf)
    assert sk_out.dtype == np.float32
    assert fr_out.dtype == sk_out.dtype  # float32 preserved
    np.testing.assert_allclose(fr_out, sk_out, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# clone / get_params / set_params parity
# ---------------------------------------------------------------------------

def test_get_params_keys_match_sklearn():
    sk_keys = set(SkPoly().get_params().keys())
    fr_keys = set(fl.PolynomialFeatures().get_params().keys())
    assert fr_keys == sk_keys == {
        "degree", "interaction_only", "include_bias", "order"}


def test_get_params_defaults_match_sklearn():
    sk = SkPoly().get_params()
    fr = fl.PolynomialFeatures().get_params()
    assert fr == sk


def test_clone_roundtrips_and_refits():
    fr = fl.PolynomialFeatures(degree=3, interaction_only=True,
                               include_bias=False)
    fr2 = clone(fr)
    assert fr2.get_params() == fr.get_params()
    sk_out = SkPoly(degree=3, interaction_only=True,
                    include_bias=False).fit_transform(_X)
    np.testing.assert_allclose(fr2.fit_transform(_X), sk_out, rtol=0, atol=0)


def test_set_params():
    fr = fl.PolynomialFeatures()
    fr.set_params(degree=3, include_bias=False)
    assert fr.degree == 3
    assert fr.include_bias is False
    sk_out = SkPoly(degree=3, include_bias=False).fit_transform(_X)
    np.testing.assert_allclose(fr.fit_transform(_X), sk_out, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# Pipeline integration
# ---------------------------------------------------------------------------

def test_pipeline_fit_transform():
    sk_out = make_pipeline(SkPoly(2)).fit_transform(_X)
    fr_out = make_pipeline(fl.PolynomialFeatures(2)).fit_transform(_X)
    np.testing.assert_allclose(fr_out, sk_out, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# order='F' honest contract: values match sklearn (memory-layout-only knob),
# bad order string -> ValueError
# ---------------------------------------------------------------------------

def test_order_f_values_match_sklearn():
    # order is a memory-layout knob; VALUES + column order are identical to 'C'.
    # ferrolearn accepts 'F' (documented: returned array is C-contiguous; only
    # contiguity differs from sklearn's F-ordered output).
    sk_out = SkPoly(degree=2, order="F").fit_transform(_X)
    fr_out = fl.PolynomialFeatures(degree=2, order="F").fit_transform(_X)
    np.testing.assert_allclose(fr_out, sk_out, rtol=0, atol=0)


def test_bad_order_string_raises_value_error_both():
    with pytest.raises(ValueError):
        SkPoly(order="Z").fit_transform(_X)
    with pytest.raises(ValueError):
        fl.PolynomialFeatures(order="Z").fit_transform(_X)


# ---------------------------------------------------------------------------
# degree=0 edge: sklearn allows degree=0 with include_bias=True (bias-only
# output column); ferrolearn's core requires degree>=1 and RAISES.
# Divergence: ferrolearn.PolynomialFeatures(degree=0, include_bias=True) raises
# ValueError ("degree must be at least 1") whereas sklearn
# (sklearn/preprocessing/_polynomial.py:325-330 only rejects degree==0 when
# include_bias is False) succeeds and emits a single all-ones bias column.
# Tracking: #2216
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="divergence: PolynomialFeatures(degree=0, include_bias=True) "
                         "raises ValueError vs sklearn bias-only output; tracking #2216")
def test_degree0_include_bias_true_matches_sklearn():
    # sklearn ORACLE: degree=0 with include_bias=True is VALID and produces a
    # single all-ones bias column (_polynomial.py:325-330 only rejects the
    # include_bias=False case). powers_ is a single all-zero row.
    sk = SkPoly(degree=0, include_bias=True)
    sk_out = sk.fit_transform(_X)
    sk_powers = sk.powers_
    sk_n_out = sk.n_output_features_

    fr = fl.PolynomialFeatures(degree=0, include_bias=True)
    fr_out = fr.fit_transform(_X)  # ferrolearn RAISES here (degree>=1 required)

    np.testing.assert_allclose(fr_out, sk_out, rtol=0, atol=0)
    np.testing.assert_array_equal(np.asarray(fr.powers_), sk_powers)
    assert int(fr.n_output_features_) == int(sk_n_out)
