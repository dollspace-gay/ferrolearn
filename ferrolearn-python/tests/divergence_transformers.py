"""Divergence guards for ferrolearn-python transformer bindings (unit #2034).

Targets: `ferrolearn-python/src/transformers.rs` (RsStandardScaler / RsPCA) + the
Python wrappers `ferrolearn-python/python/ferrolearn/_transformers.py`
(class StandardScaler / class PCA), mirroring
`sklearn.preprocessing.StandardScaler` (`sklearn/preprocessing/_data.py:696`) and
`sklearn.decomposition.PCA` (`sklearn/decomposition/_pca.py:121`).

Verification model B (goal.md): every expected value is computed by the LIVE
sklearn 1.5.2 oracle in the same test and compared against `import ferrolearn`.
No expected value is literal-copied from the ferrolearn side (R-CHAR-3).

Contents:
  - TWO FAILING PCA constructor-ABI pins (single-wrapper-fixable, R-DEV-2):
      * n_components positional   -> sklearn PCA(2) works; ferrolearn raises TypeError
      * n_components default None  -> sklearn default None keeps min(n,p) components;
                                      ferrolearn defaults to 2 (drops components)
  - PASSING API-conformance green guards for BOTH estimators (method/attribute
    surface + deterministic element-wise value parity against the live oracle).

The structural NOT-STARTED gaps (StandardScaler with_mean/with_std/copy ignored,
PCA whiten/svd_solver/... params, PCA n_components_/noise_variance_ attrs) are
filed as -l blocker crosslink issues, NOT pinned here as failing tests, because
they cannot be closed by a single-wrapper change THIS iteration (R-DEFER-3).
"""

import inspect

import numpy as np
import pytest

import ferrolearn as fl
from sklearn.decomposition import PCA as SkPCA
from sklearn.preprocessing import StandardScaler as SkStandardScaler


# Deterministic fixtures.
_X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 4.0], [3.0, 9.0]])
# n_features=3 so the default-None component count is OBSERVABLE: sklearn keeps
# min(n_samples, n_features) = 3 components; ferrolearn (default 2) keeps only 2.
_X3 = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 2.0, 3.0],
        [2.0, 4.0, 1.0],
        [3.0, 1.0, 2.0],
        [4.0, 5.0, 6.0],
    ]
)


# ---------------------------------------------------------------------------
# RED pins: PCA constructor-ABI divergences (single-wrapper-fixable, R-DEV-2).
# Expected values come from the live sklearn oracle, never from ferrolearn.
# ---------------------------------------------------------------------------


def test_red_pca_n_components_positional():
    """Divergence: ferrolearn.PCA makes n_components keyword-only.

    sklearn `PCA.__init__(self, n_components=None, *, ...)`
    (`sklearn/decomposition/_pca.py:407-409`) places n_components BEFORE the `*`,
    so `PCA(2)` is valid positionally and yields `n_components == 2`.
    ferrolearn `_transformers.py::PCA.__init__(self, *, n_components=2)` makes it
    keyword-only, so `ferrolearn.PCA(2)` raises TypeError.

    Fix: move `n_components` before the `*` in the wrapper. Tracking: see report.
    """
    # Oracle: sklearn accepts n_components positionally.
    sk_value = SkPCA(2).n_components
    assert sk_value == 2

    # ferrolearn must accept it positionally and store the same value.
    fl_value = fl.PCA(2).n_components
    assert fl_value == sk_value


def test_red_pca_n_components_default_none():
    """Divergence: ferrolearn.PCA defaults n_components to 2, not None.

    sklearn `PCA.__init__` defaults `n_components=None`
    (`sklearn/decomposition/_pca.py:409`), keeping all
    min(n_samples, n_features) components. ferrolearn defaults to 2, which DROPS
    components when n_features > 2.

    On _X3 (n_samples=5, n_features=3): sklearn `PCA().fit(X3).components_.shape`
    is (3, 3); ferrolearn forces 2 -> (2, 3). Fix: default None, resolve to
    min(n, p) at fit. Tracking: see report.
    """
    # Oracle: default is None and keeps all components.
    sk_default = inspect.signature(SkPCA.__init__).parameters["n_components"].default
    assert sk_default is None
    sk_shape = SkPCA().fit(_X3).components_.shape
    assert sk_shape == (3, 3)

    # ferrolearn default must match None and keep min(n, p) components.
    fl_default = inspect.signature(fl.PCA.__init__).parameters["n_components"].default
    assert fl_default == sk_default

    fl_shape = fl.PCA().fit(_X3).components_.shape
    assert fl_shape == sk_shape


# ---------------------------------------------------------------------------
# GREEN guards: StandardScaler API conformance (deterministic, must PASS).
# ---------------------------------------------------------------------------


def test_green_standardscaler_attributes_match_oracle():
    """StandardScaler exposes mean_/scale_/var_/n_samples_seen_ matching sklearn.

    Default path (with_mean=True/with_std=True) is deterministic
    (`sklearn/preprocessing/_data.py:756-790`).
    """
    fl_m = fl.StandardScaler().fit(_X)
    sk_m = SkStandardScaler().fit(_X)

    assert np.allclose(fl_m.mean_, sk_m.mean_, atol=1e-8)
    assert np.allclose(fl_m.scale_, sk_m.scale_, atol=1e-8)
    assert np.allclose(fl_m.var_, sk_m.var_, atol=1e-8)
    assert int(fl_m.n_samples_seen_) == int(sk_m.n_samples_seen_)


def test_green_standardscaler_transform_roundtrip():
    """transform / inverse_transform round-trip to X, shapes match sklearn."""
    fl_m = fl.StandardScaler().fit(_X)
    sk_m = SkStandardScaler().fit(_X)

    fl_t = fl_m.transform(_X)
    sk_t = sk_m.transform(_X)
    assert fl_t.shape == sk_t.shape == _X.shape
    assert np.allclose(fl_t, sk_t, atol=1e-8)
    assert np.allclose(fl_m.inverse_transform(fl_t), _X, atol=1e-8)


def test_green_standardscaler_fit_transform_mixin():
    """fit_transform (TransformerMixin) works and matches sklearn element-wise."""
    fl_ft = fl.StandardScaler().fit_transform(_X)
    sk_ft = SkStandardScaler().fit_transform(_X)
    assert fl_ft.shape == sk_ft.shape == _X.shape
    assert np.allclose(fl_ft, sk_ft, atol=1e-8)


# ---------------------------------------------------------------------------
# GREEN guards: PCA API conformance on the default deterministic-eigh path.
# ---------------------------------------------------------------------------


def test_green_pca_attribute_surface_and_shapes():
    """PCA exposes components_/explained_variance_/ratio/singular_values_/mean_
    with the right shapes for n_components=2."""
    fl_m = fl.PCA(n_components=2).fit(_X)
    n_features = _X.shape[1]

    assert fl_m.components_.shape == (2, n_features)
    assert fl_m.explained_variance_.shape == (2,)
    assert fl_m.explained_variance_ratio_.shape == (2,)
    assert fl_m.singular_values_.shape == (2,)
    assert fl_m.mean_.shape == (n_features,)


def test_green_pca_explained_variance_and_singular_values_match_oracle():
    """explained_variance_ratio_ and singular_values_ match sklearn element-wise.

    Default solver is the deterministic covariance-eigh path
    (`sklearn/decomposition/_pca.py:489`).
    """
    fl_m = fl.PCA(n_components=2).fit(_X)
    sk_m = SkPCA(n_components=2).fit(_X)

    assert np.allclose(
        fl_m.explained_variance_ratio_, sk_m.explained_variance_ratio_, atol=1e-6
    )
    assert np.allclose(fl_m.singular_values_, sk_m.singular_values_, atol=1e-6)
    assert np.allclose(fl_m.explained_variance_, sk_m.explained_variance_, atol=1e-6)
    assert np.allclose(fl_m.mean_, sk_m.mean_, atol=1e-8)


def test_green_pca_components_match_oracle_up_to_sign():
    """components_ match sklearn up to per-component sign.

    sklearn fixes signs deterministically via `svd_flip(U, Vt,
    u_based_decision=False)` (`sklearn/decomposition/_pca.py:647`); we compare
    absolute values to stay robust to any residual sign convention.
    """
    fl_m = fl.PCA(n_components=2).fit(_X)
    sk_m = SkPCA(n_components=2).fit(_X)
    assert np.allclose(np.abs(fl_m.components_), np.abs(sk_m.components_), atol=1e-6)


def test_green_pca_transform_shape_and_reconstruction():
    """transform shape is (n_samples, 2); inverse_transform reconstructs X.

    With n_components == n_features (2) there is no truncation, so the
    reconstruction is exact up to numerical error.
    """
    fl_m = fl.PCA(n_components=2).fit(_X)
    sk_m = SkPCA(n_components=2).fit(_X)

    fl_t = fl_m.transform(_X)
    assert fl_t.shape == (_X.shape[0], 2)
    # transform columns match sklearn up to per-component sign.
    assert np.allclose(np.abs(fl_t), np.abs(sk_m.transform(_X)), atol=1e-6)
    assert np.allclose(fl_m.inverse_transform(fl_t), _X, atol=1e-6)


def test_green_pca_fit_transform_mixin():
    """fit_transform (TransformerMixin) works; shape matches sklearn."""
    fl_ft = fl.PCA(n_components=2).fit_transform(_X)
    sk_ft = SkPCA(n_components=2).fit_transform(_X)
    assert fl_ft.shape == sk_ft.shape == (_X.shape[0], 2)
    assert np.allclose(np.abs(fl_ft), np.abs(sk_ft), atol=1e-6)
