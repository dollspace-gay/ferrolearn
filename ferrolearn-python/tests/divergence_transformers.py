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


# ---------------------------------------------------------------------------
# PCA n_components_ + noise_variance_ fitted attrs (unit #2097, R-DEV-3).
#
# The wrapper previously exposed only components_/explained_variance_/
# explained_variance_ratio_/mean_/singular_values_. The Rust
# FittedPCA::n_components_()/noise_variance() already match sklearn (downstream
# ferrolearn-decomp REQ-16 #1505 / REQ-15 #1507); this surfaces them on the
# binding. noise_variance_ is the mean of the discarded tail eigenvalues over
# the FULL spectrum (`sklearn/decomposition/_pca.py:686-688`), so with
# n_components=2 on a rank>2 fixture it is NON-ZERO (a meaningful regression
# guard, not the trivial 0.0 all-components-kept case).
# ---------------------------------------------------------------------------


def test_pca_n_components_and_noise_variance_match_sklearn():
    """PCA exposes n_components_ (int) and noise_variance_ (float) matching sklearn.

    R-CHAR-3: every expected value comes from the live sklearn 1.5.2 oracle in
    this test, never literal-copied from ferrolearn. The fixture has 6 samples ×
    3 features so with n_components=2 one eigenvalue is discarded and
    noise_variance_ is non-zero — exercising the tail-mean path, not just 0.0.
    """
    X = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 7.0],
            [2.0, 0.0, 1.0],
            [8.0, 6.0, 5.0],
            [3.0, 3.0, 2.0],
            [0.0, 1.0, 4.0],
        ]
    )

    fr = fl.PCA(n_components=2).fit(X)
    sk = SkPCA(n_components=2).fit(X)

    # The attributes are present on the ferrolearn wrapper.
    assert hasattr(fr, "n_components_")
    assert hasattr(fr, "noise_variance_")

    # n_components_ matches the live oracle exactly.
    assert fr.n_components_ == sk.n_components_

    # noise_variance_ matches the live oracle. It is meaningfully non-zero here
    # (one discarded eigenvalue), so this is not the trivial 0.0 path.
    assert sk.noise_variance_ > 0.0
    assert abs(float(fr.noise_variance_) - float(sk.noise_variance_)) < 1e-9


# ---------------------------------------------------------------------------
# PCA whiten constructor param (unit #2098, R-DEV-2).
#
# The wrapper previously exposed only n_components and built the Rust PCA with
# whiten=false. The downstream `PCA::with_whiten` builder
# (`ferrolearn-decomp/src/pca.rs:180`, REQ-11 #1502) already implements sklearn's
# whitening: `Transform::transform` divides each projected column j by
# `sqrt(explained_variance_[j])` (`sklearn/decomposition/_base.py:157-165`),
# `inverse_transform` re-multiplies (`:192-196`). This threads the whiten
# constructor param (`sklearn/decomposition/_pca.py:412`) through the binding.
#
# R-CHAR-3: every expected value is computed by the LIVE sklearn 1.5.2 oracle in
# the same test, never literal-copied from ferrolearn.
# ---------------------------------------------------------------------------


def test_pca_whiten_transform_matches_sklearn():
    """PCA(whiten=True).fit_transform(X) matches the live sklearn oracle.

    Also guards that whiten defaults to False and that threading the param did
    not disturb the whiten=False (default) path: fl.PCA(n_components=2) must still
    equal sklearn's non-whitened transform element-wise.
    """
    X = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 7.0],
            [2.0, 0.0, 1.0],
            [8.0, 6.0, 5.0],
            [3.0, 3.0, 2.0],
            [0.0, 1.0, 4.0],
        ]
    )

    # whiten=True: ferrolearn vs the live sklearn oracle, element-wise.
    fr_t = fl.PCA(n_components=2, whiten=True).fit_transform(X)
    sk_t = SkPCA(n_components=2, whiten=True).fit_transform(X)
    assert fr_t.shape == sk_t.shape == (X.shape[0], 2)
    np.testing.assert_allclose(fr_t, sk_t, atol=1e-6)

    # Default: whiten is False (sklearn stores the ctor param verbatim,
    # `sklearn/decomposition/_pca.py:412`/`:422`).
    assert fl.PCA(n_components=2).whiten is False
    sk_default = inspect.signature(SkPCA.__init__).parameters["whiten"].default
    assert sk_default is False

    # Regression guard: the whiten=False (default) path still equals sklearn's
    # non-whitened transform — threading the param did not perturb the default.
    fr_nw = fl.PCA(n_components=2).fit_transform(X)
    sk_nw = SkPCA(n_components=2).fit_transform(X)
    np.testing.assert_allclose(fr_nw, sk_nw, atol=1e-6)


# ---------------------------------------------------------------------------
# PCA score + score_samples Gaussian log-likelihood (unit #2099, R-DEV-3).
#
# The wrapper previously exposed no score/score_samples. The Rust
# FittedPCA::score_samples (ferrolearn-decomp/src/pca.rs:484) /
# FittedPCA::score (pca.rs:533) already match sklearn (downstream
# ferrolearn-decomp REQ-15 #1507); this surfaces them on the binding. They
# compute the Gaussian log-likelihood under the probabilistic-PCA model
# (`sklearn/decomposition/_pca.py:805-853`): score_samples is the per-sample
# log-likelihood, score is its mean.
#
# R-CHAR-3: every expected value is computed by the LIVE sklearn 1.5.2 oracle in
# the same test, never literal-copied from ferrolearn.
# ---------------------------------------------------------------------------


def test_pca_score_and_score_samples_match_sklearn():
    """PCA.score / PCA.score_samples match the live sklearn oracle element-wise.

    R-CHAR-3: every expected value comes from the live sklearn 1.5.2 oracle in
    this test, never literal-copied from ferrolearn. The fixture has 6 samples ×
    3 features with n_components=2 so the discarded tail eigenvalue makes
    noise_variance_ (and hence the get_precision-based log-likelihood) non-trivial.
    """
    X = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 7.0],
            [2.0, 0.0, 1.0],
            [8.0, 6.0, 5.0],
            [3.0, 3.0, 2.0],
            [0.0, 1.0, 4.0],
        ]
    )

    fr = fl.PCA(n_components=2).fit(X)
    sk = SkPCA(n_components=2).fit(X)

    # The methods are present on the ferrolearn wrapper.
    assert hasattr(fr, "score")
    assert hasattr(fr, "score_samples")

    # score(X) is the mean log-likelihood — match the live oracle to 1e-9.
    assert abs(fr.score(X) - sk.score(X)) < 1e-9

    # score_samples(X) is the per-sample log-likelihood — match element-wise.
    np.testing.assert_allclose(fr.score_samples(X), sk.score_samples(X), atol=1e-9)


# ---------------------------------------------------------------------------
# PCA get_covariance + get_precision (unit #2100, R-DEV-3).
#
# The wrapper previously exposed no get_covariance/get_precision — the last two
# unsurfaced FittedPCA methods. The Rust FittedPCA::get_covariance
# (ferrolearn-decomp/src/pca.rs:328, INFALLIBLE) / FittedPCA::get_precision
# (pca.rs:390, returns Result) already match sklearn (downstream
# ferrolearn-decomp REQ-14 #1505); this surfaces them on the binding. They
# compute the data covariance of the generative probabilistic-PCA model and its
# inverse (`sklearn/decomposition/_base.py:30-101`): get_covariance is
# `components_.T * exp_var_diff * components_ + noise_variance_ * I`,
# get_precision is its inverse. Neither takes an X argument.
#
# R-CHAR-3: every expected value is computed by the LIVE sklearn 1.5.2 oracle in
# the same test, never literal-copied from ferrolearn.
# ---------------------------------------------------------------------------


def test_pca_get_covariance_and_precision_match_sklearn():
    """PCA.get_covariance / PCA.get_precision match the live sklearn oracle.

    R-CHAR-3: every expected value comes from the live sklearn 1.5.2 oracle in
    this test, never literal-copied from ferrolearn. The fixture has 6 samples ×
    3 features with n_components=2 so the discarded tail eigenvalue makes
    noise_variance_ (and hence the generative-model covariance/precision)
    non-trivial.
    """
    X = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 7.0],
            [2.0, 0.0, 1.0],
            [8.0, 6.0, 5.0],
            [3.0, 3.0, 2.0],
            [0.0, 1.0, 4.0],
        ]
    )

    fr = fl.PCA(n_components=2).fit(X)
    sk = SkPCA(n_components=2).fit(X)

    # The methods are present on the ferrolearn wrapper.
    assert hasattr(fr, "get_covariance")
    assert hasattr(fr, "get_precision")

    # get_covariance() — (n_features, n_features), match the live oracle.
    fr_cov = fr.get_covariance()
    sk_cov = sk.get_covariance()
    assert fr_cov.shape == sk_cov.shape == (X.shape[1], X.shape[1])
    np.testing.assert_allclose(fr_cov, sk_cov, atol=1e-9)

    # get_precision() — the inverse covariance, match the live oracle.
    fr_prec = fr.get_precision()
    sk_prec = sk.get_precision()
    assert fr_prec.shape == sk_prec.shape == (X.shape[1], X.shape[1])
    np.testing.assert_allclose(fr_prec, sk_prec, atol=1e-9)


# ---------------------------------------------------------------------------
# RED pins: PCA whiten x {get_covariance, score, score_samples, get_precision}
# cross-param divergences (critic re-audit of #2098/#2099/#2100).
#
# sklearn's get_covariance/get_precision/score/score_samples are computed from
# the fitted model and, when whiten=True, rescale components_ by
# sqrt(explained_variance_) (`sklearn/decomposition/_base.py:44-45` and
# `:84-86`). They are therefore well-defined for whiten=True and DIFFER from the
# whiten=False values (because sklearn stores `components_` identically for both
# whiten settings, so the in-method rescale changes the result).
#
# ferrolearn's FittedPCA::get_covariance (`ferrolearn-decomp/src/pca.rs:328`)
# ignores the whiten flag, and FittedPCA::precision_and_logdet
# (`pca.rs:407-411`) short-circuits with an error whenever whiten==true, so the
# whitened score/score_samples/get_precision raise ValueError.
#
# R-CHAR-3: every expected value is computed by the LIVE sklearn 1.5.2 oracle in
# the same test, never literal-copied from ferrolearn.
# ---------------------------------------------------------------------------

# 6x3 fixture (rank 3, n_components=2 discards one eigenvalue -> noise_variance_>0).
_XW = np.array(
    [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 7.0],
        [2.0, 0.0, 1.0],
        [8.0, 6.0, 5.0],
        [3.0, 3.0, 2.0],
        [0.0, 1.0, 4.0],
    ]
)


def test_red_pca_whiten_get_covariance_matches_sklearn():
    """Divergence: PCA(whiten=True).get_covariance() ignores whiten in ferrolearn.

    sklearn `get_covariance` rescales `components_` by `sqrt(explained_variance_)`
    when `whiten=True` (`sklearn/decomposition/_base.py:44-45`), so the whitened
    covariance DIFFERS from the whiten=False covariance. ferrolearn
    `FittedPCA::get_covariance` (`ferrolearn-decomp/src/pca.rs:328`) ignores the
    whiten flag and returns the whiten=False covariance.

    Tracking: #2107.
    """
    sk_w = SkPCA(n_components=2, whiten=True).fit(_XW)
    sk_nw = SkPCA(n_components=2, whiten=False).fit(_XW)

    # Oracle invariant: the whitened covariance is genuinely different from the
    # un-whitened one (otherwise ferrolearn returning the un-whitened value would
    # not be a divergence). This pins the test to sklearn's actual behavior.
    assert not np.allclose(sk_w.get_covariance(), sk_nw.get_covariance())

    fr_w = fl.PCA(n_components=2, whiten=True).fit(_XW)
    # ferrolearn's whitened covariance must equal sklearn's whitened covariance.
    np.testing.assert_allclose(fr_w.get_covariance(), sk_w.get_covariance(), atol=1e-6)


def test_red_pca_whiten_score_and_precision_match_sklearn():
    """Divergence: PCA(whiten=True) score/score_samples/get_precision raise in ferrolearn.

    sklearn computes these from the model regardless of `whiten`
    (`sklearn/decomposition/_base.py:84-86`, `_pca.py:805-853`), returning finite
    values for whiten=True. ferrolearn `FittedPCA::precision_and_logdet`
    (`ferrolearn-decomp/src/pca.rs:407-411`) errors whenever `whiten==true`, so
    the PyO3 binding raises ValueError for all three.

    Tracking: #2108.
    """
    sk_w = SkPCA(n_components=2, whiten=True).fit(_XW)

    fr_w = fl.PCA(n_components=2, whiten=True).fit(_XW)

    # score(X): match the live oracle (whiten-specific value).
    assert abs(fr_w.score(_XW) - sk_w.score(_XW)) < 1e-6

    # score_samples(X): match element-wise.
    np.testing.assert_allclose(
        fr_w.score_samples(_XW), sk_w.score_samples(_XW), atol=1e-6
    )

    # get_precision(): match the live oracle.
    np.testing.assert_allclose(
        fr_w.get_precision(), sk_w.get_precision(), atol=1e-6
    )


# ---------------------------------------------------------------------------
# RED pin: PCA(n_components=<float ratio>) (critic re-audit of #2097).
#
# sklearn PCA accepts a float n_components in (0, 1] and selects the smallest
# number of components whose CUMULATIVE explained_variance_ratio_ is >= the
# value (`sklearn/decomposition/_pca.py:409`, float branch `:659-681`). The
# ferrolearn wrapper `_transformers.py::PCA.fit` passes n_components straight
# into `_RsPCA(n_components=...)` whose PyO3 signature is `usize`, so a float
# raises TypeError. The downstream Rust DOES support NComponents::Ratio
# (REQ-13a) but the binding never threads the float.
#
# R-CHAR-3: the expected n_components_ comes from the live sklearn oracle.
# ---------------------------------------------------------------------------


def test_red_pca_n_components_float_ratio_matches_sklearn():
    """Divergence: ferrolearn.PCA(n_components=0.95) raises; sklearn selects by ratio.

    sklearn `PCA(n_components=0.95)` selects the smallest component count whose
    cumulative `explained_variance_ratio_` is >= 0.95
    (`sklearn/decomposition/_pca.py:659-681`); on the 6x3 fixture this is
    `n_components_ == 2`. ferrolearn's wrapper passes the float into a usize-typed
    binding and raises TypeError.

    Tracking: #2109.
    """
    sk = SkPCA(n_components=0.95).fit(_XW)
    # Oracle: a float ratio is accepted and resolves to a concrete count.
    assert sk.n_components_ == 2

    fr = fl.PCA(n_components=0.95).fit(_XW)
    assert fr.n_components_ == sk.n_components_
    assert fr.components_.shape == sk.components_.shape


def test_pca_n_components_float_ratio_variants_match_sklearn():
    """PCA(n_components=<float ratio>) selects by cumulative variance like sklearn.

    Covers the float branch across several ratios, the components_ value parity on
    the selected subset, and the out-of-range float rejection. The wrapper threads
    the float into the Rust `NComponents::Ratio` path
    (`sklearn/decomposition/_pca.py:659-681` searchsorted-after-SVD); a float
    `>= 1` is rejected at fit by sklearn's `_parameter_constraints`
    (`_pca.py:386-391`).

    R-CHAR-3: every expected value comes from the live sklearn 1.5.2 oracle in
    this test, never literal-copied from ferrolearn.
    """
    # ratio -> count: smallest count whose cumulative explained_variance_ratio_
    # is >= ratio. Oracle on _XW: 0.5 -> 1, 0.95 -> 2, 0.999 -> 3.
    for ratio in (0.5, 0.95, 0.999):
        sk = SkPCA(n_components=ratio).fit(_XW)
        fr = fl.PCA(n_components=ratio).fit(_XW)

        # n_components_ resolved by the ratio matches the live oracle.
        assert fr.n_components_ == sk.n_components_
        assert fr.components_.shape == sk.components_.shape

        # The retained components match sklearn up to per-component sign
        # (deterministic svd_flip, `_pca.py:647`).
        np.testing.assert_allclose(
            np.abs(fr.components_), np.abs(sk.components_), atol=1e-6
        )
        # explained_variance_ratio_ on the selected subset matches element-wise.
        np.testing.assert_allclose(
            fr.explained_variance_ratio_, sk.explained_variance_ratio_, atol=1e-6
        )

    # Cross-check the headline ratios resolve to the oracle's concrete counts.
    assert SkPCA(n_components=0.5).fit(_XW).n_components_ == 1
    assert SkPCA(n_components=0.999).fit(_XW).n_components_ == 3

    # A float >= 1 is out of range: sklearn raises a ValueError at fit
    # (`_parameter_constraints`, `_pca.py:386-391`). ferrolearn must too — and
    # construction itself must NOT raise (validation is at fit, like sklearn's
    # `_validate_params`-at-fit).
    with pytest.raises(ValueError):
        SkPCA(n_components=2.0).fit(_XW)

    fr_bad = fl.PCA(n_components=2.0)  # constructs fine (validation deferred to fit)
    with pytest.raises(ValueError):
        fr_bad.fit(_XW)


# ---------------------------------------------------------------------------
# RED pin: PCA(whiten=True) get_precision/score/score_samples on a
# rank-deficient (n_samples < n_features) fit (critic re-audit of #2107/#2108).
#
# The #2107/#2108 fix removed the whiten bail-out in
# FittedPCA::precision_and_logdet (ferrolearn-decomp/src/pca.rs) and folds the
# `components_ * sqrt(exp_var)` whiten rescale (sklearn _base.py:46-47) into
# get_covariance. That makes whiten=True work for the 6x3 fixture. But the
# precision path still inverts get_covariance by eigendecomposing it and taking
# 1/lambda, returning NumericalInstability whenever any eigenvalue is <= 0
# (pca.rs precision_and_logdet, the `lambda <= 0` guard).
#
# When n_samples < n_features the data covariance is rank-deficient (it has
# n_samples-1 nonzero eigenvalues at most), so get_covariance has ZERO
# eigenvalues. sklearn computes get_precision via the matrix-inversion lemma
# (sklearn/decomposition/_base.py:85-101, dividing by the tiny-but-nonzero
# noise_variance_) and returns FINITE precision/score/score_samples; ferrolearn
# raises ValueError instead. The whiten=True get_covariance itself DOES match
# sklearn here (verified to ~1e-13), so this is purely the inverse/score path.
#
# R-CHAR-3: every expected value is computed by the LIVE sklearn 1.5.2 oracle in
# the test, never literal-copied from ferrolearn. Magnitudes are ~1e15 (driven
# by noise_variance_ ~= 1e-31), so the value comparison uses RELATIVE tolerance.
# ---------------------------------------------------------------------------

# 4x5 fixture: n_samples (4) < n_features (5) -> rank-deficient covariance with
# zero eigenvalues. n_components=3 keeps a nonzero (dust) noise_variance_, so
# sklearn takes the matrix-inversion-lemma branch and returns finite values.
_XW_WIDE = np.array(
    [
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [2.0, 1.0, 0.0, 3.0, 2.0],
        [5.0, 4.0, 2.0, 1.0, 0.0],
        [0.0, 3.0, 1.0, 2.0, 4.0],
    ]
)


def test_red_pca_whiten_precision_rank_deficient_matches_sklearn():
    """PCA(whiten=True) on a rank-deficient (n_samples < n_features) fit:
    get_covariance / get_precision match sklearn, and score is FINITE.

    sklearn `get_precision` uses the matrix-inversion lemma
    (`sklearn/decomposition/_base.py:85-101`) and returns finite values for this
    singular-covariance fit. ferrolearn now mirrors that: `FittedPCA::fit` routes
    the SVD through ferray's LAPACK `gesdd` (#2116) so `noise_variance_` is
    bit-identical to scipy, `precision_via_lemma` routes the inner inverse through
    ferray `inv_lapack` (LAPACK getri) + the products through ferray `gemm`
    (openblas) left-to-right (#2117), and `fast_logdet` uses LAPACK `getrf` — so
    `get_precision` matches sklearn (rtol 1e-6) and `score_samples`/`score` are
    finite (the prior code raised / returned -inf on the zero eigenvalues).

    DELIBERATELY NOT asserted (documented R-DEV-1 / S8 judgment, issue #2110,
    user-approved): the exact `score_samples`/`score` VALUE on this fit. The
    score is `Xr·precision·Xrᵀ` on a precision with condition number ~6.7e15
    (the lemma's inner-P off-diagonal is cancelling rounding noise
    ~2e-16 / noise_variance_ ~1e-31). A 1-ULP precision perturbation swings the
    score 25-83%, and the host openblas is a different build than scipy's bundled
    one (a 3-ULP `inv` delta that this conditioning amplifies). Both sklearn's and
    ferrolearn's values are numerically meaningless noise of opposite sign
    (sklearn ~-3.5e15, ferrolearn ~+3.3e15) in this degenerate n<p regime, so the
    VALUE is implementation-defined and not a meaningful divergence. Matching it
    would require linking scipy's exact openblas build. We assert the meaningful
    contract: the model (precision, covariance, noise_variance_) matches and the
    score is computable (finite), not the rounding of singular-matrix noise.

    Tracking: #2110 (meaningful parity SHIPPED; exact rank-deficient score value
    is NOT-STARTED / won't-fix per the above).
    """
    sk = SkPCA(n_components=3, whiten=True).fit(_XW_WIDE)
    fr = fl.PCA(n_components=3, whiten=True).fit(_XW_WIDE)

    # Oracle invariant: sklearn returns finite values for this rank-deficient
    # whiten=True fit (pins the test to sklearn's actual behavior).
    sk_prec = sk.get_precision()
    assert np.all(np.isfinite(sk_prec))
    assert np.isfinite(sk.score(_XW_WIDE))

    # Meaningful parity: get_covariance + get_precision match the live oracle.
    np.testing.assert_allclose(fr.get_covariance(), sk.get_covariance(), atol=1e-6)
    np.testing.assert_allclose(fr.get_precision(), sk_prec, rtol=1e-6)

    # ferrolearn must NOT raise and must produce FINITE score_samples/score
    # (the prior bug: ValueError / -inf). The exact value is intentionally not
    # asserted (see docstring).
    fr_ss = fr.score_samples(_XW_WIDE)
    assert fr_ss.shape == sk.score_samples(_XW_WIDE).shape
    assert np.all(np.isfinite(fr_ss))
    assert np.isfinite(fr.score(_XW_WIDE))


# ---------------------------------------------------------------------------
# RED pin #2111: PCA(n_components=<float ratio>) off-by-one at the cumulative
# explained_variance_ratio_ boundary (critic re-audit of #2109).
#
# sklearn resolves a float ratio AFTER the SVD with
#   n_components = searchsorted(cumsum(explained_variance_ratio_), ratio,
#                               side="right") + 1
# (sklearn/decomposition/_pca.py:680-681). The `side="right"` tie-break means a
# threshold that lands EXACTLY ON (or fractionally below, within ULPs of)
# ferrolearn's reconstructed cumulative-sum boundary must keep the SAME count
# sklearn keeps. ferrolearn's `PCA::with_variance_ratio`
# (ferrolearn-decomp/src/pca.rs) computes its own cumulative sum and selects a
# count that disagrees with sklearn by ONE for thresholds just below a boundary:
# for the same X and the same nc, sklearn keeps N components but ferrolearn keeps
# N+1.
#
# R-CHAR-3: the expected count is taken from the LIVE sklearn 1.5.2 oracle on the
# SAME (X, nc) pair in this test, never literal-copied from ferrolearn.
# ---------------------------------------------------------------------------

# 8x5 fixture (np.random.RandomState(0) draw). Its explained_variance_ratio_
# cumulative sums are ~[0.5246, 0.7668, 0.9047, 0.9798, 1.0]; the threshold
# below sits one ULP-cluster under the 0.7668 boundary.
_XRATIO = np.array(
    [
        [0.74159174, 1.55291372, -2.2683282, 1.33354538, -0.84272405],
        [1.96992445, 1.26611853, -0.50587654, 2.54520078, 1.08081191],
        [0.48431215, 0.57914048, -0.18158257, 1.41020463, -0.37447169],
        [0.27519832, -0.96075461, 0.37692697, 0.03343893, 0.68056724],
        [-1.56349669, -0.56669762, -0.24214951, 1.51439128, -0.3330574],
        [0.04736482, 1.46274045, 1.53502913, 0.56644004, 0.14926509],
        [-1.078278, 1.39547227, 1.78748405, -0.56951726, 0.17538653],
        [-0.46250554, -1.0858006, 0.63973599, -0.38586334, -0.77576235],
    ]
)


def test_red_pca_ncomp_float_ratio_cumsum_boundary_off_by_one():
    """Divergence: float n_components off-by-one at the cumsum tie-break vs sklearn.

    sklearn `_fit_full` selects the float-ratio count with
    `searchsorted(cumsum(explained_variance_ratio_), ratio, side="right") + 1`
    (`sklearn/decomposition/_pca.py:680-681`). For `nc` fractionally below the
    second cumulative-variance boundary of `_XRATIO`, sklearn keeps 2 components;
    ferrolearn's `PCA::with_variance_ratio`
    (`ferrolearn-decomp/src/pca.rs`) keeps 3 — an off-by-one for the SAME (X, nc).

    R-CHAR-3: the expected count is the live sklearn 1.5.2 oracle on this exact
    (X, nc); nothing is copied from ferrolearn.

    Tracking: #2111.
    """
    nc = 0.7667943106601404  # one ULP-cluster below the 0.7668 cumsum boundary

    sk = SkPCA(n_components=nc).fit(_XRATIO)
    fr = fl.PCA(n_components=nc).fit(_XRATIO)

    # Live oracle: sklearn keeps exactly this many components for (X, nc).
    assert sk.n_components_ == 2  # documents the oracle outcome
    # ferrolearn must agree with the live oracle for the SAME (X, nc).
    assert fr.n_components_ == sk.n_components_
    # The retained component count drives components_ row count, so a mismatch
    # propagates to the whole fitted model.
    assert fr.components_.shape[0] == sk.components_.shape[0]


def test_red_pca_ncomp_bool_accepted_by_sklearn():
    """Divergence: ferrolearn rejects bool n_components; sklearn accepts True as 1.

    sklearn's `n_components` constraint is `Interval(Integral, 0, None, ...)`
    (`sklearn/decomposition/_pca.py:386-391`); Python `bool` is an `Integral`
    subclass, so `PCA(n_components=True)` is ACCEPTED and fits with one component
    (`n_components_` is the truthy int 1). ferrolearn's `_resolve_n_components`
    (`ferrolearn-python/python/ferrolearn/_transformers.py`) explicitly guards
    `not isinstance(nc, bool)` and raises ValueError, over-rejecting an input
    sklearn accepts.

    R-CHAR-3: the expected behavior (no raise, 1 component) is read from the live
    sklearn 1.5.2 oracle in this test.

    Tracking: #2112.
    """
    X = _XRATIO

    # Live oracle: sklearn accepts bool True and keeps 1 component.
    sk = SkPCA(n_components=True).fit(X)
    assert int(sk.n_components_) == 1

    # ferrolearn must match: accept True and keep 1 component (it currently
    # raises ValueError in _resolve_n_components).
    fr = fl.PCA(n_components=True).fit(X)
    assert int(fr.n_components_) == int(sk.n_components_)


# ---------------------------------------------------------------------------
# RED pin #2115: PCA 'auto' solver-selection diverges from sklearn on the
# medium-large, not-tall-skinny shape where sklearn's documented 'auto' policy
# selects the `randomized` truncated solver.
#
# sklearn's 'auto' policy (sklearn/decomposition/_pca.py:531-543):
#   - covariance_eigh  if n_features <= 1000 and n_samples >= 10*n_features
#   - full             elif max(X.shape) <= 500 or n_components == 'mle'
#   - randomized       elif 1 <= n_components < 0.8 * min(X.shape)
#   - full             else
# ferrolearn's migrated `fit` (ferrolearn-decomp/src/pca.rs:1297) only implements
#   use_covariance_eigh = n_features <= 1000 and n_samples >= 10*n_features
# and routes EVERYTHING ELSE to the `full` SVD path -- the `randomized` branch is
# folded into `full` (REQ-12 NOT-STARTED). So for a shape that lands in sklearn's
# `randomized` branch, ferrolearn returns the EXACT full-SVD spectrum while
# sklearn returns the (different) randomized-SVD approximation.
#
# Fixture 600x100, n_components=10: max(shape)=600>500 (not full-by-size),
# 600 < 10*100=1000 (not covariance_eigh), and 1 <= 10 < 0.8*100=80, so sklearn
# selects `randomized`. The randomized solver with random_state=None is
# stochastic, but the gap vs ferrolearn's full SVD (~0.02-0.06 on
# explained_variance_[0]) is two orders of magnitude beyond rtol=1e-6 and stable
# across seeds; we pin the oracle with a fixed seed for reproducibility.
#
# R-CHAR-3: the expected spectrum is computed by the LIVE sklearn 1.5.2 'auto'
# oracle in-test (seeded for determinism); nothing is copied from ferrolearn.
# ---------------------------------------------------------------------------


def test_red_pca_auto_solver_randomized_branch_matches_sklearn():
    """Divergence: PCA 'auto' picks full where sklearn 'auto' picks randomized.

    For a 600x100 fixture with n_components=10, sklearn's 'auto' policy
    (`sklearn/decomposition/_pca.py:539-540`) selects the `randomized` truncated
    SVD solver (max(shape)=600>500, 600<10*100, and 1<=10<0.8*100). ferrolearn's
    migrated `fit` (`ferrolearn-decomp/src/pca.rs:1297`) has no `randomized`
    solver and falls back to the exact `full` SVD, so for the SAME (X, n_components)
    its explained_variance_/singular_values_/noise_variance_ differ from sklearn's
    'auto' output by ~1e-2 -- far beyond rtol=1e-6.

    R-CHAR-3: the oracle is the live sklearn 1.5.2 'auto' fit in this test
    (random_state fixed only to make the stochastic randomized result
    reproducible); ferrolearn values are never copied into the expectation.

    Tracking: #2115.
    """
    rng = np.random.RandomState(0)
    X = rng.randn(600, 100)

    # Live oracle: sklearn 'auto' selects `randomized` here.
    sk = SkPCA(n_components=10, svd_solver="auto", random_state=42).fit(X)
    assert sk._fit_svd_solver == "randomized"  # documents the policy outcome

    # Pass the SAME seed as the oracle: the randomized solver's range-finder
    # draw is stochastic, so reproducing sklearn's spectrum requires the matching
    # random_state. ferrolearn's RNG (ferray::random::RandomState) is bit-identical
    # to numpy's RandomState, so seed 42 reproduces sklearn's seed-42 draw.
    fr = fl.PCA(n_components=10, random_state=42).fit(X)

    # ferrolearn must reproduce sklearn's 'auto' (randomized) spectrum for the
    # SAME (X, n_components, random_state).
    np.testing.assert_allclose(
        np.asarray(fr.explained_variance_),
        sk.explained_variance_,
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(fr.singular_values_),
        sk.singular_values_,
        rtol=1e-6,
        atol=1e-6,
    )
    assert abs(float(fr.noise_variance_) - float(sk.noise_variance_)) <= 1e-6 * abs(
        float(sk.noise_variance_)
    )
