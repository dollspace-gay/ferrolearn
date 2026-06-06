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


# ---------------------------------------------------------------------------
# StandardScaler with_mean/with_std/copy parity (unit #2037, R-DEV-1/-2).
#
# The wrapper previously ignored with_mean/with_std/copy: `fit` always built a
# no-arg `_RsStandardScaler()` (always centers+scales) and always set
# `mean_`/`scale_`/`var_`. These compare `ferrolearn.StandardScaler` to the live
# sklearn 1.5.2 oracle across all 4 (with_mean, with_std) configs (R-CHAR-3:
# expected values come from sklearn in the same test, never literal-copied).
#
# Oracle (sklearn 1.5.2), X=[[1,10],[2,20],[3,30]]:
#   (T,T): mean_=[2,20], scale_=[0.8165..,8.165..], var_=[0.6667,66.667],
#          transform[0]=[-1.224745,-1.224745]
#   (T,F): mean_=[2,20], scale_=None, var_=None, transform[0]=[-1,-10]
#   (F,T): mean_=[2,20], scale_=[0.8165..,8.165..], var_=[0.6667,66.667],
#          transform[0]=[1.224745,1.224745]
#   (F,F): mean_=None,  scale_=None, var_=None, transform[0]=[1,10]
# ---------------------------------------------------------------------------

_XSS = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])


def _assert_attr_parity(fl_attr, sk_attr, name):
    """Assert a fitted attribute matches sklearn, including the None case."""
    if sk_attr is None:
        assert fl_attr is None, f"{name}: sklearn is None, ferrolearn is {fl_attr!r}"
    else:
        assert fl_attr is not None, f"{name}: sklearn is {sk_attr!r}, ferrolearn None"
        np.testing.assert_allclose(
            fl_attr, sk_attr, rtol=1e-9, atol=1e-12, err_msg=f"{name} mismatch"
        )


@pytest.mark.parametrize("with_mean", [True, False])
@pytest.mark.parametrize("with_std", [True, False])
def test_red_standardscaler_with_mean_std_transform_matches_oracle(with_mean, with_std):
    """transform(X) matches sklearn element-wise for every (with_mean, with_std).

    sklearn skips centering when `with_mean=False` and skips scaling when
    `with_std=False` (`sklearn/preprocessing/_data.py:1064-1067`), so
    (False, False) is the identity. The previous ferrolearn wrapper ignored both
    flags and always centered+scaled.
    """
    fl_m = fl.StandardScaler(with_mean=with_mean, with_std=with_std).fit(_XSS)
    sk_m = SkStandardScaler(with_mean=with_mean, with_std=with_std).fit(_XSS)

    np.testing.assert_allclose(
        fl_m.transform(_XSS), sk_m.transform(_XSS), rtol=1e-9, atol=1e-12
    )


@pytest.mark.parametrize("with_mean", [True, False])
@pytest.mark.parametrize("with_std", [True, False])
def test_red_standardscaler_with_mean_std_attrs_match_oracle(with_mean, with_std):
    """mean_/scale_/var_ match sklearn including the None nulling rule.

    sklearn nulls `mean_`/`var_` when `with_mean=False` AND `with_std=False`
    (`_data.py:993-995`) and nulls `scale_`/`var_` when `with_std=False`
    (`_data.py:1022-1023`). Expected None-ness comes from the live oracle.
    """
    fl_m = fl.StandardScaler(with_mean=with_mean, with_std=with_std).fit(_XSS)
    sk_m = SkStandardScaler(with_mean=with_mean, with_std=with_std).fit(_XSS)

    _assert_attr_parity(fl_m.mean_, sk_m.mean_, "mean_")
    _assert_attr_parity(fl_m.scale_, sk_m.scale_, "scale_")
    _assert_attr_parity(fl_m.var_, sk_m.var_, "var_")


def test_red_standardscaler_copy_param_in_signature():
    """ferrolearn.StandardScaler exposes the `copy` ctor param like sklearn.

    sklearn `StandardScaler.__init__(self, *, copy=True, with_mean=True,
    with_std=True)` (`_data.py:835`). The oracle confirms `copy` is present with
    default True.
    """
    import inspect

    sk_params = inspect.signature(SkStandardScaler.__init__).parameters
    assert "copy" in sk_params
    assert sk_params["copy"].default is True

    fl_params = inspect.signature(fl.StandardScaler.__init__).parameters
    assert "copy" in fl_params
    assert fl_params["copy"].default is True


def test_red_standardscaler_copy_roundtrip():
    """StandardScaler(copy=True) round-trips inverse_transform(transform(X)) ~ X."""
    fl_m = fl.StandardScaler(copy=True).fit(_XSS)
    recovered = fl_m.inverse_transform(fl_m.transform(_XSS))
    np.testing.assert_allclose(recovered, _XSS, rtol=1e-9, atol=1e-12)


# ---------------------------------------------------------------------------
# StandardScaler.var_ on a CONSTANT (zero-variance) column (unit #2086).
#
# The wrapper previously derived `var_ = scale_**2`, which is WRONG on a
# constant column: sklearn's `_handle_zeros_in_scale` clamps `scale_=1.0` there
# (`sklearn/preprocessing/_data.py:1019-1020`), so `scale_**2=1.0`, while the
# TRUE population variance is 0.0. sklearn's `var_` is the raw variance computed
# BEFORE that clamp (`_data.py:1013-1023`). The fix reads the binding's true
# variance (`FittedStandardScaler::var()`, downstream REQ-5 #1192).
#
# Oracle (sklearn 1.5.2), X=[[1,5],[2,5],[3,5]] (column 1 constant):
#   var_   == [0.6666666666666666, 0.0]
#   scale_ == [0.816496580927726, 1.0]
#   buggy scale_**2 == [0.6666666666666666, 1.0]  (var_[1] wrong: 1.0 != 0.0)
# ---------------------------------------------------------------------------

# Column 1 is constant (all 5.0) -> a genuine zero-variance feature.
_XCONST = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])


def test_red_standardscaler_var_constant_column_matches_oracle():
    """var_ matches sklearn on a constant column (0.0), not the buggy scale_**2.

    R-CHAR-3: every expected value comes from the live sklearn 1.5.2 oracle in
    this test, never literal-copied from ferrolearn. The previous wrapper set
    `var_ = scale_**2`; on the constant column `scale_=1.0` so `scale_**2=1.0`,
    diverging from sklearn's true `var_=0.0`.
    """
    fl_m = fl.StandardScaler().fit(_XCONST)
    sk_m = SkStandardScaler().fit(_XCONST)

    # var_ matches the live oracle element-wise (tight: it is a deterministic
    # two-pass population variance).
    np.testing.assert_allclose(fl_m.var_, sk_m.var_, rtol=1e-12)

    # The constant column's variance is exactly 0.0 — NOT the buggy scale_**2.
    # The oracle confirms 0.0; ferrolearn now matches.
    assert sk_m.var_[1] == 0.0
    assert fl_m.var_[1] == 0.0

    # Prove this is the TRUE-variance path, not the old `scale_**2` path: with
    # sklearn's `_handle_zeros_in_scale` the constant-column scale is 1.0, so the
    # OLD buggy ferrolearn `var_` was `sk.scale_**2 == 1.0` there — wrong. The
    # oracle's own scale_ confirms the trap the fix avoids.
    assert sk_m.scale_[1] == 1.0
    assert sk_m.scale_[1] ** 2 == 1.0  # the value the old buggy var_[1] held
    assert fl_m.var_[1] != sk_m.scale_[1] ** 2  # fixed: 0.0 != 1.0


# ---------------------------------------------------------------------------
# StandardScaler.scale_ on a CONSTANT (zero-variance) column (unit #2087).
#
# The `scale_` getter previously marshalled `FittedStandardScaler::std()` (the
# RAW std), which is 0.0 on a constant column, instead of `scale()` (the
# `_handle_zeros_in_scale`-clamped value, 1.0 there — what sklearn exposes as
# `scale_`). sklearn computes `scale_ = _handle_zeros_in_scale(sqrt(var_))`
# (`sklearn/preprocessing/_data.py:1019-1020`, helper `:88`), clamping the
# constant column to 1.0 to avoid division by zero. The fix reads the binding's
# `scale()` getter (`FittedStandardScaler::scale()`, downstream REQ-5 #1192).
#
# Oracle (sklearn 1.5.2), X=[[1,5],[2,5],[3,5]] (column 1 constant):
#   scale_     == [0.816496580927726, 1.0]
#   raw std    == [0.816496580927726, 0.0]  (the old buggy scale_[1])
#   transform(X) column 1 all 0.0
# ---------------------------------------------------------------------------


def test_red_standardscaler_scale_constant_column_matches_oracle():
    """scale_ matches sklearn on a constant column (1.0), not the raw std (0.0).

    R-CHAR-3: every expected value comes from the live sklearn 1.5.2 oracle in
    this test, never literal-copied from ferrolearn. The previous wrapper read
    `FittedStandardScaler::std()`; on the constant column the raw std is 0.0,
    diverging from sklearn's `_handle_zeros_in_scale`-clamped `scale_=1.0`.
    """
    fl_m = fl.StandardScaler().fit(_XCONST)
    sk_m = SkStandardScaler().fit(_XCONST)

    # scale_ matches the live oracle element-wise (tight: it is a deterministic
    # _handle_zeros_in_scale(sqrt(population variance)) value).
    np.testing.assert_allclose(fl_m.scale_, sk_m.scale_, rtol=1e-12)

    # The constant column's scale is clamped to exactly 1.0 — NOT the old buggy
    # raw std of 0.0. The oracle confirms 1.0; ferrolearn now matches.
    assert sk_m.scale_[1] == 1.0
    assert fl_m.scale_[1] == 1.0

    # Prove this is the clamped-scale path, not the old raw-std path: sklearn's
    # raw std on the constant column is 0.0 (the value the OLD buggy getter
    # returned), so the fixed scale_ must differ from it there.
    assert np.sqrt(sk_m.var_[1]) == 0.0  # the value the old buggy scale_[1] held
    assert fl_m.scale_[1] != np.sqrt(sk_m.var_[1])  # fixed: 1.0 != 0.0

    # The change does not perturb transform: the constant column still maps to
    # 0.0 in both ferrolearn and sklearn (centered then divided by 1.0).
    np.testing.assert_allclose(
        fl_m.transform(_XCONST), sk_m.transform(_XCONST), rtol=1e-12, atol=1e-12
    )
    assert np.all(sk_m.transform(_XCONST)[:, 1] == 0.0)
    assert np.all(fl_m.transform(_XCONST)[:, 1] == 0.0)


def test_standardscaler_pca_get_feature_names_out():
    """REQ #2095: StandardScaler/PCA expose get_feature_names_out matching sklearn.

    StandardScaler is 1:1 (OneToOneFeatureMixin) → input names (default x0..xn);
    PCA emits pca0..pca{n_components-1} (ClassNamePrefixFeaturesOutMixin).
    """
    from sklearn.decomposition import PCA as SkPCA
    from sklearn.preprocessing import StandardScaler as SkSS

    X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [2.0, 1.0, 0.0]])
    ss = fl.StandardScaler().fit(X)
    assert ss.get_feature_names_out().tolist() == SkSS().fit(X).get_feature_names_out().tolist()
    assert ss.get_feature_names_out(["a", "b", "c"]).tolist() == ["a", "b", "c"]

    pca = fl.PCA(n_components=2).fit(X)
    assert (
        pca.get_feature_names_out().tolist()
        == SkPCA(n_components=2).fit(X).get_feature_names_out().tolist()
    )
