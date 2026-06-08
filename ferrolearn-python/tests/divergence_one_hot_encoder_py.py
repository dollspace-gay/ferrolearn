"""Live-sklearn parity for the OneHotEncoder PyO3 binding (#1155, REQ-8).

Targets: `ferrolearn-python/src/extras.rs` (`_RsOneHotEncoder`) + the Python
wrapper `ferrolearn-python/python/ferrolearn/_extras.py` (class OneHotEncoder),
mirroring `sklearn.preprocessing.OneHotEncoder` for the DENSE numeric path
(`sparse_output=False`, `sklearn/preprocessing/_encoders.py:458-1230`).

Verification model B (goal.md R-CHAR-3): every expected value is computed by the
LIVE sklearn 1.5.2 oracle (`SkOHE(sparse_output=False, ...)`) in the same test
and compared against `import ferrolearn`. No expected value is literal-copied
from the ferrolearn side.

SCOPE (matches the binding's honest scope, R-HONEST-3):
  - DENSE-ONLY: sklearn defaults `sparse_output=True` (scipy CSR); ferrolearn's
    core is dense-only (one_hot_encoder.rs REQ-2 NOT-STARTED #1149). The binding
    accepts `sparse_output` for the ctor/get_params ABI but REQUIRES `False` at
    `fit` (a `True` value raises a clear dense-only error). The oracle is always
    `SkOHE(sparse_output=False)`.
  - `handle_unknown`: 'error'/'ignore' supported; 'infrequent_if_exist'
    (REQ-5 #1152) -> NotImplementedError; bad string -> ValueError.
  - `drop`/`min_frequency`/`max_categories`/`feature_name_combiner`/`categories`
    (non-default) -> NotImplementedError (REQ-5/REQ-7 #1152/#1154).
  - handle_unknown='ignore' inverse: an all-zero block inverts to `None` in
    sklearn, to `NaN` in ferrolearn (Array2<f64>, #2227); the KNOWN feature
    recovers and is asserted equal.
"""

import numpy as np
import pytest

import ferrolearn as fl
from sklearn.base import clone
from sklearn.preprocessing import OneHotEncoder as SkOHE


def test_fit_transform_multifeature_noncontiguous_matches_sklearn():
    # Non-contiguous categories in col0 ({2,5,9}) + col1 ({0,1}). Oracle:
    # sklearn OneHotEncoder(sparse_output=False).fit_transform.
    X = np.array([[2.0, 0.0], [5.0, 1.0], [9.0, 0.0], [5.0, 1.0]])
    sk = SkOHE(sparse_output=False).fit(X)
    sk_out = sk.transform(X)

    fl_enc = fl.OneHotEncoder(sparse_output=False).fit(X)
    fl_out = fl_enc.transform(X)

    np.testing.assert_array_equal(fl_out, sk_out)
    # categories_ is a list of arrays, per feature.
    assert len(fl_enc.categories_) == len(sk.categories_)
    for fl_c, sk_c in zip(fl_enc.categories_, sk.categories_):
        np.testing.assert_array_equal(fl_c, sk_c)


def test_fit_transform_method_separate_matches_sklearn():
    X = np.array([[10.0, -3.0], [10.0, 4.0], [20.0, -3.0]])
    sk_out = SkOHE(sparse_output=False).fit_transform(X)
    fl_out = fl.OneHotEncoder(sparse_output=False).fit_transform(X)
    np.testing.assert_array_equal(fl_out, sk_out)


def test_handle_unknown_ignore_all_zero_block_matches_sklearn():
    # Train on a small known set; transform with an UNKNOWN value in each column.
    X_train = np.array([[0.0, 5.0], [1.0, 99.0], [1.0, 5.0]])
    X_test = np.array([[100.0, 5.0], [1.0, 7.0]])  # 100, 7 are unknown
    sk = SkOHE(sparse_output=False, handle_unknown="ignore").fit(X_train)
    sk_out = sk.transform(X_test)

    fl_enc = fl.OneHotEncoder(sparse_output=False, handle_unknown="ignore").fit(X_train)
    fl_out = fl_enc.transform(X_test)

    np.testing.assert_array_equal(fl_out, sk_out)


def test_inverse_transform_roundtrip_matches_sklearn():
    X = np.array([[2.0, 0.0], [5.0, 1.0], [9.0, 0.0], [5.0, 1.0]])
    sk = SkOHE(sparse_output=False).fit(X)
    fl_enc = fl.OneHotEncoder(sparse_output=False).fit(X)

    enc = fl_enc.transform(X)
    sk_inv = sk.inverse_transform(sk.transform(X))
    fl_inv = fl_enc.inverse_transform(enc)
    np.testing.assert_array_equal(fl_inv, X)
    np.testing.assert_array_equal(fl_inv, sk_inv)


def test_inverse_transform_ignore_all_zero_known_feature_recovers():
    # handle_unknown='ignore': an all-zero block is the unknown sentinel. sklearn
    # inverts it to None (object dtype); ferrolearn to NaN (f64 ABI, #2227). The
    # KNOWN feature must still recover to the sklearn value.
    X_train = np.array([[0.0, 5.0], [1.0, 9.0], [1.0, 5.0]])
    sk = SkOHE(sparse_output=False, handle_unknown="ignore").fit(X_train)
    fl_enc = fl.OneHotEncoder(sparse_output=False, handle_unknown="ignore").fit(X_train)

    # A held-out matrix: col0 unknown (all-zero block), col1 known = 9.0.
    # categories_: col0 -> [0,1], col1 -> [5,9]. Block layout [c0_0,c0_1,c1_5,c1_9].
    held = np.array([[0.0, 0.0, 0.0, 1.0]])  # col0 all-zero, col1 -> 9.0
    sk_inv = sk.inverse_transform(held)
    fl_inv = fl_enc.inverse_transform(held)

    # sklearn col0 == None; ferrolearn col0 == NaN. col1 recovers to 9.0 in both.
    assert sk_inv[0, 0] is None
    assert np.isnan(fl_inv[0, 0])
    assert float(fl_inv[0, 1]) == float(sk_inv[0, 1]) == 9.0


def test_get_feature_names_out_matches_sklearn():
    X = np.array([[2.0, 0.0], [5.0, 1.0], [9.0, 0.0], [5.0, 1.0]])
    sk = SkOHE(sparse_output=False).fit(X)
    fl_enc = fl.OneHotEncoder(sparse_output=False).fit(X)
    np.testing.assert_array_equal(
        fl_enc.get_feature_names_out(), sk.get_feature_names_out()
    )
    # Spot the headline values explicitly (x0_2.0 etc.).
    assert list(sk.get_feature_names_out()) == [
        "x0_2.0", "x0_5.0", "x0_9.0", "x1_0.0", "x1_1.0",
    ]


def test_prefit_transform_raises_not_fitted():
    from sklearn.exceptions import NotFittedError

    enc = fl.OneHotEncoder(sparse_output=False)
    X = np.array([[0.0], [1.0]])
    with pytest.raises(NotFittedError):
        enc.transform(X)
    with pytest.raises(NotFittedError):
        _ = enc.categories_
    with pytest.raises(NotFittedError):
        _ = enc.n_features_in_
    with pytest.raises(NotFittedError):
        enc.get_feature_names_out()


def test_bad_handle_unknown_raises_value_error():
    # sklearn: InvalidParameterError (a ValueError subclass).
    X = np.array([[0.0], [1.0]])
    with pytest.raises(ValueError):
        SkOHE(sparse_output=False, handle_unknown="foo").fit(X)
    with pytest.raises(ValueError):
        fl.OneHotEncoder(sparse_output=False, handle_unknown="foo").fit(X)


def test_infrequent_if_exist_raises_not_implemented():
    # 'infrequent_if_exist' is a VALID sklearn option but ferrolearn does not
    # implement infrequent grouping (REQ-5 NOT-STARTED #1152).
    X = np.array([[0.0], [1.0]])
    with pytest.raises(NotImplementedError):
        fl.OneHotEncoder(
            sparse_output=False, handle_unknown="infrequent_if_exist"
        ).fit(X)


def test_sparse_output_true_default_raises_dense_only():
    # sklearn DEFAULTS sparse_output=True (returns CSR). ferrolearn is dense-only
    # and raises a clear error rather than silently returning dense (R-HONEST-3).
    X = np.array([[0.0], [1.0]])
    with pytest.raises(NotImplementedError):
        fl.OneHotEncoder().fit(X)  # default sparse_output=True
    with pytest.raises(NotImplementedError):
        fl.OneHotEncoder(sparse_output=True).fit(X)
    # sparse_output=False works.
    fl.OneHotEncoder(sparse_output=False).fit(X)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"drop": "first"},
        {"min_frequency": 2},
        {"max_categories": 3},
        {"categories": [[0.0, 1.0]]},
        {"feature_name_combiner": lambda f, c: f"{f}_{c}"},
    ],
)
def test_unsupported_params_raise_not_implemented(kwargs):
    X = np.array([[0.0], [1.0]])
    with pytest.raises(NotImplementedError):
        fl.OneHotEncoder(sparse_output=False, **kwargs).fit(X)


def test_get_params_keys_match_sklearn():
    # All 8 sklearn ctor params must be present for clone/get_params parity.
    sk_keys = set(SkOHE().get_params().keys())
    fl_keys = set(fl.OneHotEncoder().get_params().keys())
    assert fl_keys == sk_keys, (fl_keys, sk_keys)


def test_clone_preserves_params_and_fits():
    base = fl.OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    cloned = clone(base)
    assert cloned.get_params() == base.get_params()
    X_train = np.array([[0.0, 5.0], [1.0, 9.0]])
    X_test = np.array([[100.0, 5.0]])  # unknown col0 -> all-zero in ignore mode
    cloned.fit(X_train)
    sk = SkOHE(sparse_output=False, handle_unknown="ignore").fit(X_train)
    np.testing.assert_array_equal(cloned.transform(X_test), sk.transform(X_test))
