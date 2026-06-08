"""Divergence pin: LabelEncoder.transform([]) on a STRING-fitted encoder.

Targets `ferrolearn-python/python/ferrolearn/_extras.py` (class LabelEncoder,
`_to_labels`) vs `sklearn.preprocessing.LabelEncoder.transform`
(`sklearn/preprocessing/_label.py:118-137`).

DIVERGENCE
----------
After fitting on STRING labels (the binding's supported path), calling
`transform([])` with an empty Python list is a legal, in-scope call: the encoder
holds a valid string `classes_`, and sklearn returns an empty array with no
error:

    sklearn/preprocessing/_label.py:128-137
        if _num_samples(y) == 0:
            return np.array([])
    -> SkLE().fit(['a','b','c']).transform([]) == array([], dtype=float64)

ferrolearn's `LabelEncoder.transform` funnels the input through
`_to_labels`, which does `np.asarray([])` -> dtype float64 -> the numeric-dtype
guard (`_extras.py:3048`) raises NotImplementedError. So a *fitted-on-strings*
encoder rejects the empty-transform call that sklearn answers with an empty
array. This is NOT the documented numeric-input scope limitation (the labels at
fit were strings); it is an inference accident of `np.asarray([])` defaulting to
float64, surfaced as a behavioral divergence (RAISE vs return-empty).

Verification model B (goal.md R-CHAR-3): the expected value is the LIVE sklearn
1.5.2 oracle in this same test; nothing is copied from the ferrolearn side.

The 20 existing tests in `divergence_label_encoder_py.py` cover non-empty
transform/inverse but never the empty `transform([])` path.
"""

import numpy as np
import pytest

import ferrolearn as fl
from sklearn.preprocessing import LabelEncoder as SkLE


@pytest.mark.skip(
    reason="divergence: LabelEncoder.transform([]) on a string-fitted encoder "
    "raises NotImplementedError (np.asarray([]) -> float64 -> numeric guard) "
    "vs sklearn returning an empty array (_label.py:128-137); tracking #2231"
)
def test_empty_transform_after_string_fit_matches_sklearn():
    # Fit on STRING labels (the binding's supported, in-scope path).
    train = ["a", "b", "c"]

    # Oracle: sklearn returns an empty array for transform([]) (no error).
    sk = SkLE().fit(train)
    sk_out = sk.transform([])
    assert sk_out.shape == (0,)

    # ferrolearn: must mirror sklearn — an empty array, not an exception.
    fe = fl.LabelEncoder().fit(train)
    fe_out = fe.transform([])  # <-- ferrolearn raises NotImplementedError here

    np.testing.assert_array_equal(fe_out, sk_out)
