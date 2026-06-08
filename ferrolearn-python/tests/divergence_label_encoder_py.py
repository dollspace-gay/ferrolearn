"""Live-sklearn parity for the LabelEncoder PyO3 binding (#1137, REQ-7).

Targets: `ferrolearn-python/src/extras.rs` (`_RsLabelEncoder`, the 1-D STRING-INPUT
analog of `_RsOrdinalEncoder`) + the Python wrapper
`ferrolearn-python/python/ferrolearn/_extras.py` (class LabelEncoder), mirroring
`sklearn.preprocessing.LabelEncoder` for the STRING input path
(`sklearn/preprocessing/_label.py:34-165`).

Verification model B (goal.md R-CHAR-3): every expected value is computed by the
LIVE sklearn 1.5.2 oracle (`SkLE(...)`) in the same test and compared against
`import ferrolearn`. No expected value is literal-copied from the ferrolearn side.

SCOPE (matches the binding's honest scope, R-HONEST-3):
  - STRING-only input (Array1<String>, label_encoder.rs REQ-4 #1135). A numeric
    label that looks like a number is string-sorted, matching sklearn for STRING
    input; numeric-DTYPE input is REJECTED (NotImplementedError, #2230 lesson:
    numeric labels would be string-sorted = wrong codes).
  - LabelEncoder has NO constructor params (`get_params() == {}`, `_label.py:34`).
  - unseen label at transform / out-of-range code at inverse_transform ->
    ValueError (`_label.py:137`, `:158-160`).
  - pre-fit transform/inverse/classes_ -> NotFittedError.
"""

import numpy as np
import pytest

import ferrolearn as fl
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import LabelEncoder as SkLE


def test_fit_transform_matches_sklearn():
    # Oracle: sklearn LabelEncoder().fit_transform(['b','a','c','a']) -> [1,0,2,0]
    # (classes_ = ['a','b','c'] sorted; b->1, a->0, c->2). `_label.py:101/98`.
    labels = ["b", "a", "c", "a"]
    sk = SkLE()
    sk_out = sk.fit_transform(labels)

    fe = fl.LabelEncoder()
    fe_out = fe.fit_transform(labels)

    np.testing.assert_array_equal(fe_out, sk_out)
    np.testing.assert_array_equal(fe_out, np.array([1, 0, 2, 0]))


def test_classes_matches_sklearn():
    # Oracle: classes_ == sorted-unique (`_label.py:98` `_unique(y)`).
    labels = ["b", "a", "c", "a"]
    sk = SkLE().fit(labels)
    fe = fl.LabelEncoder().fit(labels)
    np.testing.assert_array_equal(
        fe.classes_.astype(str), sk.classes_.astype(str)
    )
    np.testing.assert_array_equal(fe.classes_.astype(str), np.array(["a", "b", "c"]))


def test_transform_after_fit_matches_sklearn():
    # Oracle: transform of a fresh label sequence (`_label.py:118`).
    train = ["b", "a", "c", "a"]
    query = ["c", "a", "b", "c"]
    sk = SkLE().fit(train)
    fe = fl.LabelEncoder().fit(train)
    np.testing.assert_array_equal(fe.transform(query), sk.transform(query))


def test_inverse_transform_roundtrip_matches_sklearn():
    # Oracle: inverse_transform([0,2]) -> ['a','c'] (`_label.py:139`).
    labels = ["b", "a", "c", "a"]
    sk = SkLE().fit(labels)
    fe = fl.LabelEncoder().fit(labels)
    sk_inv = sk.inverse_transform([0, 2])
    fe_inv = fe.inverse_transform([0, 2])
    np.testing.assert_array_equal(fe_inv.astype(str), sk_inv.astype(str))
    np.testing.assert_array_equal(fe_inv.astype(str), np.array(["a", "c"]))


def test_inverse_transform_full_roundtrip_recovers_labels():
    # fit_transform then inverse_transform recovers the original labels (both libs).
    labels = ["dog", "cat", "bird", "cat", "dog"]
    sk = SkLE()
    sk_codes = sk.fit_transform(labels)
    sk_recovered = sk.inverse_transform(sk_codes)

    fe = fl.LabelEncoder()
    fe_codes = fe.fit_transform(labels)
    fe_recovered = fe.inverse_transform(fe_codes)

    np.testing.assert_array_equal(fe_recovered.astype(str), sk_recovered.astype(str))
    np.testing.assert_array_equal(fe_recovered.astype(str), np.array(labels))


def test_unknown_label_at_transform_raises_valueerror():
    # Oracle: sklearn transform of an unseen label -> ValueError
    # ("y contains previously unseen labels", `_label.py:137`).
    fe = fl.LabelEncoder().fit(["a", "b", "c"])
    sk = SkLE().fit(["a", "b", "c"])
    with pytest.raises(ValueError):
        sk.transform(["z"])
    with pytest.raises(ValueError):
        fe.transform(["z"])


def test_inverse_out_of_range_code_raises_valueerror():
    # Oracle: sklearn inverse_transform of an out-of-range code -> ValueError
    # (`_label.py:158-160` setdiff1d guard).
    fe = fl.LabelEncoder().fit(["a", "b", "c"])
    sk = SkLE().fit(["a", "b", "c"])
    with pytest.raises(ValueError):
        sk.inverse_transform([5])
    with pytest.raises(ValueError):
        fe.inverse_transform([5])


def test_inverse_negative_code_raises_valueerror():
    # A negative code has no usize image; sklearn's setdiff1d guard also rejects it
    # (it is not in arange(n_classes)). Both raise ValueError.
    fe = fl.LabelEncoder().fit(["a", "b", "c"])
    sk = SkLE().fit(["a", "b", "c"])
    with pytest.raises(ValueError):
        sk.inverse_transform([-1])
    with pytest.raises(ValueError):
        fe.inverse_transform([-1])


def test_numeric_input_rejected_with_notimplementederror():
    # #2230 lesson: numeric-dtype labels would be string-sorted (1,10,2) =/=
    # sklearn's numeric sort (1,2,10), so ferrolearn REJECTS numeric input rather
    # than mis-encode it. sklearn ACCEPTS it (numeric sort); the honest divergence
    # is a NotImplementedError on the ferrolearn side.
    numeric = [1, 2, 10]
    # sklearn's numeric sort (the behavior ferrolearn cannot reproduce string-only).
    sk = SkLE().fit(numeric)
    np.testing.assert_array_equal(sk.classes_, np.array([1, 2, 10]))
    with pytest.raises(NotImplementedError):
        fl.LabelEncoder().fit(numeric)
    with pytest.raises(NotImplementedError):
        fl.LabelEncoder().fit_transform(numeric)


def test_numeric_looking_string_input_string_sorted_matches_sklearn():
    # STRING input that looks numeric ('1','2','10') is string-sorted by BOTH libs
    # (sklearn's _unique on a str array sorts lexicographically too): classes_ ==
    # ['1','10','2']. This is the supported string path.
    labels = ["1", "2", "10"]
    sk = SkLE()
    sk_codes = sk.fit_transform(labels)
    fe = fl.LabelEncoder()
    fe_codes = fe.fit_transform(labels)
    np.testing.assert_array_equal(fe_codes, sk_codes)
    np.testing.assert_array_equal(
        fe.classes_.astype(str), sk.classes_.astype(str)
    )
    np.testing.assert_array_equal(fe.classes_.astype(str), np.array(["1", "10", "2"]))


def test_prefit_transform_raises_notfitted():
    # sklearn `transform` calls check_is_fitted -> NotFittedError (`_label.py:131`).
    with pytest.raises(NotFittedError):
        SkLE().transform(["a"])
    with pytest.raises(NotFittedError):
        fl.LabelEncoder().transform(["a"])


def test_prefit_inverse_raises_notfitted():
    # sklearn `inverse_transform` calls check_is_fitted (`_label.py:152`).
    with pytest.raises(NotFittedError):
        SkLE().inverse_transform([0])
    with pytest.raises(NotFittedError):
        fl.LabelEncoder().inverse_transform([0])


def test_prefit_classes_raises_notfitted():
    # `classes_` is a fitted attribute; pre-fit access raises (AttributeError in
    # sklearn, NotFittedError in ferrolearn via check_is_fitted). Both raise.
    with pytest.raises((NotFittedError, AttributeError)):
        _ = SkLE().classes_
    with pytest.raises(NotFittedError):
        _ = fl.LabelEncoder().classes_


def test_two_d_input_raises_valueerror():
    # LabelEncoder is 1-D (`column_or_1d`, `_label.py:97`); a 2-D input -> ValueError.
    X2d = np.array([["a", "b"], ["c", "d"]])
    with pytest.raises(ValueError):
        SkLE().fit(X2d)
    with pytest.raises(ValueError):
        fl.LabelEncoder().fit(X2d)


def test_get_params_empty():
    # sklearn LabelEncoder has NO params: get_params() == {} (`_label.py:34`).
    assert SkLE().get_params() == {}
    assert fl.LabelEncoder().get_params() == {}


def test_clone():
    # clone() round-trips via get_params/set_params (empty params).
    fe = fl.LabelEncoder()
    cloned = clone(fe)
    assert isinstance(cloned, fl.LabelEncoder)
    # The clone fits identically to a fresh instance.
    labels = ["b", "a", "c", "a"]
    np.testing.assert_array_equal(
        cloned.fit_transform(labels), fl.LabelEncoder().fit_transform(labels)
    )


def test_numpy_object_array_input():
    # Input as a numpy str/object 1-D array works (matches sklearn).
    labels = np.array(["x", "y", "x", "z"], dtype=object)
    sk = SkLE()
    sk_codes = sk.fit_transform(labels)
    fe = fl.LabelEncoder()
    fe_codes = fe.fit_transform(labels)
    np.testing.assert_array_equal(fe_codes, sk_codes)
    np.testing.assert_array_equal(
        fe.classes_.astype(str), sk.classes_.astype(str)
    )


def test_numpy_str_array_input():
    # Input as a numpy '<U' str 1-D array works.
    labels = np.array(["paris", "tokyo", "paris", "amsterdam"])
    sk = SkLE()
    sk_codes = sk.fit_transform(labels)
    fe = fl.LabelEncoder()
    fe_codes = fe.fit_transform(labels)
    np.testing.assert_array_equal(fe_codes, sk_codes)


def test_single_class():
    # A single repeated label -> all codes 0, classes_ == [label] (both libs).
    labels = ["only", "only", "only"]
    sk = SkLE()
    sk_codes = sk.fit_transform(labels)
    fe = fl.LabelEncoder()
    fe_codes = fe.fit_transform(labels)
    np.testing.assert_array_equal(fe_codes, sk_codes)
    np.testing.assert_array_equal(fe_codes, np.zeros(3, dtype=fe_codes.dtype))


def test_transform_output_dtype_int64():
    # sklearn transform returns an integer array; ferrolearn returns int64.
    fe = fl.LabelEncoder().fit(["a", "b"])
    out = fe.transform(["a", "b"])
    assert out.dtype == np.int64
