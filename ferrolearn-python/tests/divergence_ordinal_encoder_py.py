"""Live-sklearn parity for the OrdinalEncoder PyO3 binding (#1167, REQ-12).

Targets: `ferrolearn-python/src/extras.rs` (`_RsOrdinalEncoder`, the FIRST
STRING-INPUT binding) + the Python wrapper
`ferrolearn-python/python/ferrolearn/_extras.py` (class OrdinalEncoder), mirroring
`sklearn.preprocessing.OrdinalEncoder` for the STRING input path
(`sklearn/preprocessing/_encoders.py:1235-1679`).

Verification model B (goal.md R-CHAR-3): every expected value is computed by the
LIVE sklearn 1.5.2 oracle (`SkOrd(...)`) in the same test and compared against
`import ferrolearn`. No expected value is literal-copied from the ferrolearn side.

SCOPE (matches the binding's honest scope, R-HONEST-3):
  - STRING-only input (Array2<String>, REQ-4 #1159). Numeric/object input is
    coerced to str by `_to_rows` (matching sklearn's object/str acceptance).
  - `handle_unknown`: 'error'/'use_encoded_value' supported; bad string ->
    ValueError (sklearn InvalidParameterError, `_encoders.py:1425`).
  - `encoded_missing_value` (non-NaN, REQ-6 #1161), `min_frequency`/
    `max_categories` (REQ-8 #1163), non-float64 `dtype` (REQ-3 #1158) ->
    NotImplementedError.
  - `use_encoded_value` inverse: a cell == unknown_value inverts to None in
    sklearn, to a ValueError in ferrolearn (Array2<String> can't hold None,
    REQ-9 #1167) — NOT exercised in the roundtrip tests below.
"""

import numpy as np
import pytest

import ferrolearn as fl
from sklearn.base import clone
from sklearn.preprocessing import OrdinalEncoder as SkOrd


def test_fit_transform_single_feature_matches_sklearn():
    # Oracle: sklearn OrdinalEncoder().fit_transform -> [[0.],[1.],[0.]] for
    # cat/dog/cat (sorted-unique: cat=0, dog=1).
    X = [["cat"], ["dog"], ["cat"]]
    sk = SkOrd().fit(X)
    sk_out = sk.transform(X)

    fe = fl.OrdinalEncoder().fit(X)
    fe_out = fe.transform(X)

    np.testing.assert_array_equal(fe_out, sk_out)
    # categories_ == sklearn's (sorted-unique).
    assert len(fe.categories_) == len(sk.categories_)
    for fc, sc in zip(fe.categories_, sk.categories_):
        np.testing.assert_array_equal(fc.astype(str), sc.astype(str))


def test_fit_transform_value_is_zero_one_zero():
    # Pin the exact sklearn-documented output (`_encoders.py:1374`).
    X = [["cat"], ["dog"], ["cat"]]
    sk_out = SkOrd().fit_transform(X)
    fe_out = fl.OrdinalEncoder().fit_transform(X)
    np.testing.assert_array_equal(fe_out, sk_out)
    np.testing.assert_array_equal(fe_out, np.array([[0.0], [1.0], [0.0]]))


def test_multifeature_matches_sklearn():
    X = [["cat", "small"], ["dog", "large"], ["cat", "medium"], ["bird", "small"]]
    sk = SkOrd().fit(X)
    fe = fl.OrdinalEncoder().fit(X)
    np.testing.assert_array_equal(fe.transform(X), sk.transform(X))
    for fc, sc in zip(fe.categories_, sk.categories_):
        np.testing.assert_array_equal(fc.astype(str), sc.astype(str))


def test_inverse_transform_roundtrip_matches_original():
    X = [["cat", "small"], ["dog", "large"], ["cat", "medium"], ["bird", "small"]]
    sk = SkOrd().fit(X)
    fe = fl.OrdinalEncoder().fit(X)

    enc = fe.transform(X)
    sk_inv = sk.inverse_transform(sk.transform(X))
    fe_inv = fe.inverse_transform(enc)

    np.testing.assert_array_equal(fe_inv.astype(str), sk_inv.astype(str))
    # Roundtrip recovers the original strings exactly.
    np.testing.assert_array_equal(fe_inv.astype(str), np.asarray(X, dtype=object).astype(str))


def test_handle_unknown_use_encoded_value_matches_sklearn():
    # Oracle: unknown category -> -1.0 (the unknown_value sentinel).
    X_train = [["cat"], ["dog"]]
    X_test = [["cat"], ["fish"], ["dog"]]
    sk = SkOrd(handle_unknown="use_encoded_value", unknown_value=-1).fit(X_train)
    fe = fl.OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1).fit(X_train)
    sk_out = sk.transform(X_test)
    fe_out = fe.transform(X_test)
    np.testing.assert_array_equal(fe_out, sk_out)
    # The unknown 'fish' row encodes to -1.0.
    assert fe_out[1, 0] == -1.0


def test_explicit_categories_given_order_matches_sklearn():
    # Oracle: explicit categories used in the GIVEN order (NOT re-sorted), so
    # dog=0, cat=1, bird=2 (not the sorted cat<bird<dog).
    X = [["dog"], ["cat"], ["bird"], ["dog"]]
    cats = [["dog", "cat", "bird"]]
    sk = SkOrd(categories=cats).fit(X)
    fe = fl.OrdinalEncoder(categories=cats).fit(X)
    np.testing.assert_array_equal(fe.transform(X), sk.transform(X))
    # dog -> 0, cat -> 1, bird -> 2 (given order).
    np.testing.assert_array_equal(
        fe.transform([["dog"], ["cat"], ["bird"]]),
        np.array([[0.0], [1.0], [2.0]]),
    )


def test_n_features_in_matches_sklearn():
    X = [["cat", "small"], ["dog", "large"]]
    sk = SkOrd().fit(X)
    fe = fl.OrdinalEncoder().fit(X)
    assert fe.n_features_in_ == sk.n_features_in_ == 2


def test_get_feature_names_out_matches_sklearn():
    X = [["cat", "small"], ["dog", "large"]]
    sk = SkOrd().fit(X)
    fe = fl.OrdinalEncoder().fit(X)
    np.testing.assert_array_equal(
        fe.get_feature_names_out().astype(str),
        sk.get_feature_names_out().astype(str),
    )
    np.testing.assert_array_equal(fe.get_feature_names_out().astype(str),
                                  np.array(["x0", "x1"]))


def test_get_feature_names_out_with_input_features_matches_sklearn():
    X = [["cat", "small"], ["dog", "large"]]
    sk = SkOrd().fit(X)
    fe = fl.OrdinalEncoder().fit(X)
    names = ["a", "b"]
    np.testing.assert_array_equal(
        fe.get_feature_names_out(names).astype(str),
        sk.get_feature_names_out(names).astype(str),
    )


def test_pre_fit_transform_raises_not_fitted():
    from sklearn.exceptions import NotFittedError
    with pytest.raises(NotFittedError):
        fl.OrdinalEncoder().transform([["cat"]])


def test_pre_fit_attribute_raises_not_fitted():
    from sklearn.exceptions import NotFittedError
    enc = fl.OrdinalEncoder()
    with pytest.raises(NotFittedError):
        _ = enc.categories_
    with pytest.raises(NotFittedError):
        _ = enc.n_features_in_


def test_bad_handle_unknown_raises_value_error():
    # sklearn raises InvalidParameterError (a ValueError subclass) for a bad
    # handle_unknown; ferrolearn raises ValueError.
    with pytest.raises(ValueError):
        fl.OrdinalEncoder(handle_unknown="foo").fit([["cat"], ["dog"]])
    # Confirm sklearn also rejects it (oracle).
    with pytest.raises(ValueError):
        SkOrd(handle_unknown="foo").fit([["cat"], ["dog"]])


def test_unsupported_encoded_missing_value_raises_not_implemented():
    with pytest.raises(NotImplementedError):
        fl.OrdinalEncoder(encoded_missing_value=0).fit([["cat"], ["dog"]])


def test_unsupported_min_frequency_raises_not_implemented():
    with pytest.raises(NotImplementedError):
        fl.OrdinalEncoder(min_frequency=2).fit([["cat"], ["dog"], ["cat"]])


def test_unsupported_max_categories_raises_not_implemented():
    with pytest.raises(NotImplementedError):
        fl.OrdinalEncoder(max_categories=2).fit([["cat"], ["dog"], ["cat"]])


def test_unsupported_dtype_raises_not_implemented():
    with pytest.raises(NotImplementedError):
        fl.OrdinalEncoder(dtype=np.int64).fit([["cat"], ["dog"]])


def test_get_params_keys_match_sklearn():
    # The 7-key constructor surface (`_encoders.py:1435-1452`).
    fe_keys = set(fl.OrdinalEncoder().get_params().keys())
    sk_keys = set(SkOrd().get_params().keys())
    assert fe_keys == sk_keys, (fe_keys, sk_keys)
    assert sk_keys == {
        "categories", "dtype", "handle_unknown", "unknown_value",
        "encoded_missing_value", "min_frequency", "max_categories",
    }


def test_clone_round_trips_params():
    enc = fl.OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    cloned = clone(enc)
    assert cloned.handle_unknown == "use_encoded_value"
    assert cloned.unknown_value == -1
    # The clone fits + transforms identically.
    X_train = [["cat"], ["dog"]]
    X_test = [["cat"], ["fish"]]
    cloned.fit(X_train)
    sk = SkOrd(handle_unknown="use_encoded_value", unknown_value=-1).fit(X_train)
    np.testing.assert_array_equal(cloned.transform(X_test), sk.transform(X_test))


def test_numpy_object_array_input_works():
    # Input as a numpy object/str array (not just list-of-lists) — sklearn accepts
    # object/str arrays; ferrolearn's `_to_rows` coerces to str.
    X = np.array([["cat", "small"], ["dog", "large"], ["cat", "medium"]], dtype=object)
    sk = SkOrd().fit(X)
    fe = fl.OrdinalEncoder().fit(X)
    np.testing.assert_array_equal(fe.transform(X), sk.transform(X))

    X_str = np.array([["cat"], ["dog"], ["cat"]])  # numpy str (U) dtype
    sk2 = SkOrd().fit(X_str)
    fe2 = fl.OrdinalEncoder().fit(X_str)
    np.testing.assert_array_equal(fe2.transform(X_str), sk2.transform(X_str))
