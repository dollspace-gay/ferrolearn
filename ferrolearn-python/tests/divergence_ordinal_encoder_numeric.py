"""ferrolearn.OrdinalEncoder rejects NUMERIC input rather than mis-encoding it
(#2230, REQ-12 / REQ-4 scope).

Verification model B (goal.md R-CHAR-3): the sklearn behavior is computed by the
LIVE sklearn 1.5.2 oracle (`SkOrd`) and contrasted with `import ferrolearn`.

Background: ferrolearn's OrdinalEncoder core is STRING-only and sorts categories
LEXICOGRAPHICALLY; sklearn sorts NUMERIC categories numerically (1, 2, 10 not
'1', '10', '2', `sklearn/preprocessing/_encoders.py:99`). The binding originally
coerced numeric input via `np.asarray(X).astype(str)`, which SILENTLY produced
wrong ordinals (string-sort order) for numeric categories.

Fix (#2230): `OrdinalEncoder._to_rows` now REJECTS numeric-dtype input with
`NotImplementedError` (numeric/mixed-dtype input is REQ-4 NOT-STARTED) instead of
mis-encoding it. The honest scope: ferrolearn.OrdinalEncoder supports STRING
categories; for numeric categories sklearn is required. This avoids the
silent-wrong-ordinal release blocker.

Tracking: #2230
"""

import numpy as np
import pytest

import ferrolearn as fl
from sklearn.preprocessing import OrdinalEncoder as SkOrd


def test_numeric_input_rejected_not_silently_miscoded():
    # Integers whose string order (1, 10, 2) differs from their numeric order
    # (1, 2, 10): the input that exposed the silent .astype(str) divergence.
    X = [[1], [2], [10]]

    # sklearn handles numeric categories (numeric sort): categories_ == [1,2,10].
    sk_out = SkOrd().fit_transform(X)
    assert sk_out.tolist() == [[0.0], [1.0], [2.0]]

    # ferrolearn is string-only: it REJECTS numeric input (NotImplementedError),
    # rather than string-sorting it into wrong ordinals ([[0],[2],[1]]).
    with pytest.raises(NotImplementedError):
        fl.OrdinalEncoder().fit(X)
    with pytest.raises(NotImplementedError):
        fl.OrdinalEncoder().fit_transform(np.array([[1.0], [2.0], [10.0]]))


def test_string_input_still_works():
    # String categories that LOOK numeric are still string-sorted, matching
    # sklearn for STRING input ('1' < '10' < '2').
    X = [["1"], ["2"], ["10"]]
    sk = SkOrd().fit(X)
    fe = fl.OrdinalEncoder().fit(X)
    np.testing.assert_array_equal(
        np.asarray(fe.transform(X)), sk.transform(X)
    )
