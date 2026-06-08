"""Live-sklearn parity for the QuantileTransformer PyO3 binding (#1329, REQ-11).

Targets: `ferrolearn-python/src/extras.rs` (`_RsQuantileTransformer`) + the Python
wrapper `ferrolearn-python/python/ferrolearn/_extras.py` (class
QuantileTransformer), mirroring `sklearn.preprocessing.QuantileTransformer`
(`sklearn/preprocessing/_data.py:2540-2975`).

Verification model B (goal.md): every expected value is computed by the LIVE
sklearn 1.5.2 oracle in the same test and compared against `import ferrolearn`.
No expected value is literal-copied from the ferrolearn side (R-CHAR-3).

SCOPE (matches the binding's honest scope):
  - The deterministic path uses ALL samples (REQ-6 actual subsampling NOT-STARTED
    #1324, RNG-gated #2118), so every fixture here has n_samples <= subsample
    (the default 10000). Fixtures use n_samples >= n_quantiles so there is no
    clamp (except the explicit clamp test), and the sklearn oracle is constructed
    with subsample=None so sklearn ALSO uses all samples — making the comparison
    deterministic.
  - sklearn emits a `n_quantiles (k) is greater than the total number of samples`
    UserWarning when n_quantiles > n_samples (`_data.py:2784-2789`); it is
    suppressed where the clamp is exercised.

Tolerances (binding caveats):
  - uniform float64: ~1e-9
  - normal output: ~1e-7 (Acklam ppf / ndtr, quantile_transformer.rs REQ-3)
  - float32 input: ~1e-5 (#2215-class f64-ABI cast-back)
"""

import warnings

import numpy as np
import pytest

import ferrolearn as fl
from sklearn.preprocessing import QuantileTransformer as SkQT


# Deterministic fixtures. n_samples large enough that n_samples >= n_quantiles
# (no clamp) AND n_samples <= subsample (no subsample divergence).
_RNG = np.random.RandomState(0)
# Distinct-value single feature.
_X_DISTINCT = np.linspace(-3.0, 7.0, 50).reshape(-1, 1)
# Tied-value / plateau fixture: many repeats so the forward+reversed averaging
# (`_data.py:2843-2846`) is exercised.
_X_TIED = np.array(
    [[1.0]] * 10 + [[2.0]] * 10 + [[3.0]] * 10 + [[4.0]] * 10 + [[5.0]] * 10
)
# Multi-feature fixture (two columns, different scales).
_X_MULTI = np.column_stack(
    [np.linspace(0.0, 1.0, 40), np.linspace(-10.0, 10.0, 40) ** 2]
)


def _sk(n_quantiles=10, output_distribution="uniform"):
    # subsample=None so sklearn ALSO uses ALL samples (matches the binding's
    # deterministic no-subsample path).
    return SkQT(
        n_quantiles=n_quantiles,
        output_distribution=output_distribution,
        subsample=None,
    )


# ---------------------------------------------------------------------------
# Forward transform — uniform
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("X", [_X_DISTINCT, _X_TIED, _X_MULTI])
def test_uniform_fit_transform_matches_sklearn(X):
    n_quantiles = 10
    got = fl.QuantileTransformer(
        n_quantiles=n_quantiles, output_distribution="uniform"
    ).fit_transform(X)
    want = _sk(n_quantiles, "uniform").fit_transform(X)
    np.testing.assert_allclose(got, want, atol=1e-9)


def test_uniform_n_quantiles_equals_n_samples_no_clamp():
    # n_quantiles == n_samples (50): no clamp, exact landmark grid.
    X = _X_DISTINCT
    got = fl.QuantileTransformer(n_quantiles=50).fit_transform(X)
    want = _sk(50, "uniform").fit_transform(X)
    np.testing.assert_allclose(got, want, atol=1e-9)


# ---------------------------------------------------------------------------
# Forward transform — normal
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("X", [_X_DISTINCT, _X_MULTI])
def test_normal_fit_transform_matches_sklearn(X):
    n_quantiles = 10
    got = fl.QuantileTransformer(
        n_quantiles=n_quantiles, output_distribution="normal"
    ).fit_transform(X)
    want = _sk(n_quantiles, "normal").fit_transform(X)
    np.testing.assert_allclose(got, want, atol=1e-7)


# ---------------------------------------------------------------------------
# inverse_transform
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dist,atol", [("uniform", 1e-9), ("normal", 1e-7)])
def test_inverse_transform_round_trip(dist, atol):
    # inverse_transform(transform(X)) ~= X (where X is within the landmark range).
    X = _X_DISTINCT
    qt = fl.QuantileTransformer(n_quantiles=50, output_distribution=dist).fit(X)
    Y = qt.transform(X)
    back = qt.inverse_transform(Y)
    np.testing.assert_allclose(back, X, atol=atol)


@pytest.mark.parametrize("dist,atol", [("uniform", 1e-9), ("normal", 1e-7)])
def test_inverse_transform_matches_sklearn(dist, atol):
    # inverse_transform of an arbitrary in-range Y matches sklearn's.
    X = _X_DISTINCT
    fqt = fl.QuantileTransformer(n_quantiles=10, output_distribution=dist).fit(X)
    sqt = _sk(10, dist).fit(X)
    if dist == "uniform":
        Y = np.linspace(0.0, 1.0, 13).reshape(-1, 1)
    else:
        Y = np.linspace(-2.0, 2.0, 13).reshape(-1, 1)
    got = fqt.inverse_transform(Y)
    want = sqt.inverse_transform(Y)
    np.testing.assert_allclose(got, want, atol=atol)


# ---------------------------------------------------------------------------
# Fitted attributes
# ---------------------------------------------------------------------------


def test_quantiles_attribute_matches_sklearn():
    X = _X_MULTI
    fqt = fl.QuantileTransformer(n_quantiles=10).fit(X)
    sqt = _sk(10, "uniform").fit(X)
    assert fqt.quantiles_.shape == sqt.quantiles_.shape == (10, 2)
    np.testing.assert_allclose(fqt.quantiles_, sqt.quantiles_, atol=1e-9)


def test_references_attribute_matches_sklearn():
    X = _X_DISTINCT
    fqt = fl.QuantileTransformer(n_quantiles=10).fit(X)
    sqt = _sk(10, "uniform").fit(X)
    # references_ is exactly np.linspace(0, 1, n_quantiles_) (`_data.py:2795`).
    np.testing.assert_allclose(fqt.references_, sqt.references_, atol=1e-12)
    np.testing.assert_allclose(
        fqt.references_, np.linspace(0.0, 1.0, 10), atol=1e-12
    )


def test_n_quantiles_attribute_no_clamp():
    X = _X_DISTINCT  # 50 samples
    fqt = fl.QuantileTransformer(n_quantiles=10).fit(X)
    sqt = _sk(10, "uniform").fit(X)
    assert fqt.n_quantiles_ == sqt.n_quantiles_ == 10


def test_n_features_in_attribute_matches_sklearn():
    X = _X_MULTI
    fqt = fl.QuantileTransformer(n_quantiles=10).fit(X)
    sqt = _sk(10, "uniform").fit(X)
    assert fqt.n_features_in_ == sqt.n_features_in_ == 2


# ---------------------------------------------------------------------------
# n_quantiles > n_samples CLAMP (UserWarning + n_quantiles_ == n_samples)
# ---------------------------------------------------------------------------


def test_n_quantiles_greater_than_n_samples_clamps():
    # 8 samples, n_quantiles=20 -> sklearn warns and clamps n_quantiles_ to 8.
    X = np.linspace(0.0, 1.0, 8).reshape(-1, 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # sklearn `n_quantiles > n_samples` warn
        fqt = fl.QuantileTransformer(n_quantiles=20).fit(X)
        sqt = SkQT(n_quantiles=20, subsample=None).fit(X)
    assert fqt.n_quantiles_ == sqt.n_quantiles_ == 8
    assert fqt.references_.shape == sqt.references_.shape == (8,)
    np.testing.assert_allclose(fqt.references_, sqt.references_, atol=1e-12)
    # transform under the clamp still matches.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        got = fl.QuantileTransformer(n_quantiles=20).fit_transform(X)
        want = SkQT(n_quantiles=20, subsample=None).fit_transform(X)
    np.testing.assert_allclose(got, want, atol=1e-9)


# ---------------------------------------------------------------------------
# Error / NotFittedError / parameter contracts
# ---------------------------------------------------------------------------


def test_pre_fit_transform_raises_not_fitted():
    from sklearn.exceptions import NotFittedError

    qt = fl.QuantileTransformer(n_quantiles=10)
    with pytest.raises(NotFittedError):
        qt.transform(_X_DISTINCT)
    sqt = SkQT(n_quantiles=10, subsample=None)
    with pytest.raises(NotFittedError):
        sqt.transform(_X_DISTINCT)


def test_pre_fit_attribute_raises_not_fitted():
    from sklearn.exceptions import NotFittedError

    qt = fl.QuantileTransformer(n_quantiles=10)
    for attr in ("quantiles_", "references_", "n_quantiles_", "n_features_in_"):
        with pytest.raises(NotFittedError):
            getattr(qt, attr)


def test_bad_output_distribution_raises_value_error():
    # sklearn StrOptions({"uniform","normal"}) -> InvalidParameterError ⊂ ValueError.
    with pytest.raises(ValueError):
        fl.QuantileTransformer(
            n_quantiles=10, output_distribution="foo"
        ).fit(_X_DISTINCT)
    with pytest.raises(ValueError):
        SkQT(
            n_quantiles=10, output_distribution="foo", subsample=None
        ).fit(_X_DISTINCT)


def test_n_quantiles_greater_than_subsample_raises_value_error():
    # sklearn raises ValueError when n_quantiles > subsample (`_data.py:2774`).
    with pytest.raises(ValueError):
        fl.QuantileTransformer(n_quantiles=100, subsample=10).fit(_X_DISTINCT)
    with pytest.raises(ValueError):
        SkQT(n_quantiles=100, subsample=10).fit(_X_DISTINCT)


# ---------------------------------------------------------------------------
# dtype contract
# ---------------------------------------------------------------------------


def test_dtype_int_to_float64():
    X = np.arange(20).reshape(-1, 1).astype(np.int64)
    got = fl.QuantileTransformer(n_quantiles=10).fit_transform(X)
    want = _sk(10, "uniform").fit_transform(X)
    assert got.dtype == np.float64
    np.testing.assert_allclose(got, want, atol=1e-9)


def test_dtype_float64_preserved():
    X = _X_DISTINCT.astype(np.float64)
    got = fl.QuantileTransformer(n_quantiles=10).fit_transform(X)
    assert got.dtype == np.float64


def test_dtype_float32_tolerant():
    X = _X_DISTINCT.astype(np.float32)
    got = fl.QuantileTransformer(n_quantiles=10).fit_transform(X)
    want = _sk(10, "uniform").fit_transform(X.astype(np.float64))
    assert got.dtype == np.float32
    # #2215-class f64-ABI cast-back: tolerant comparison.
    np.testing.assert_allclose(got, want, atol=1e-5)


# ---------------------------------------------------------------------------
# clone / get_params / set_params / pipeline
# ---------------------------------------------------------------------------


def test_get_params_keys_match_sklearn():
    fp = set(fl.QuantileTransformer().get_params().keys())
    sp = set(SkQT().get_params().keys())
    assert fp == sp == {
        "n_quantiles",
        "output_distribution",
        "ignore_implicit_zeros",
        "subsample",
        "random_state",
        "copy",
    }


def test_clone_round_trips():
    from sklearn.base import clone

    qt = fl.QuantileTransformer(n_quantiles=7, output_distribution="normal")
    c = clone(qt)
    assert c.n_quantiles == 7
    assert c.output_distribution == "normal"
    # fitted output of the clone matches.
    got = c.fit_transform(_X_DISTINCT)
    want = _sk(7, "normal").fit_transform(_X_DISTINCT)
    np.testing.assert_allclose(got, want, atol=1e-7)


def test_set_params():
    qt = fl.QuantileTransformer(n_quantiles=10)
    qt.set_params(n_quantiles=15, output_distribution="normal")
    assert qt.n_quantiles == 15
    assert qt.output_distribution == "normal"


def test_pipeline():
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler as SkStandardScaler

    X = _X_MULTI
    pipe = Pipeline([
        ("qt", fl.QuantileTransformer(n_quantiles=10)),
        ("sc", SkStandardScaler()),
    ])
    out = pipe.fit_transform(X)
    assert out.shape == X.shape
    # equivalent sklearn-only pipeline (subsample=None matches the binding).
    ref = Pipeline([
        ("qt", _sk(10, "uniform")),
        ("sc", SkStandardScaler()),
    ]).fit_transform(X)
    np.testing.assert_allclose(out, ref, atol=1e-9)
