"""Divergence pin: ferrolearn.OrdinalEncoder SILENTLY produces wrong ordinals for
NUMERIC input (#2230, REQ-12 follow-on).

Verification model B (goal.md R-CHAR-3): the expected values are computed by the
LIVE sklearn 1.5.2 oracle (`SkOrd`) in the test itself and compared against
`import ferrolearn`. No expected value is literal-copied from the ferrolearn side.

Root cause: `OrdinalEncoder._to_rows` (ferrolearn-python/python/ferrolearn/_extras.py:2854)
does `np.asarray(X).astype(str).tolist()`, coercing numeric cells to strings BEFORE
the Rust core sorts them. sklearn instead determines each feature's category set with
`_unique(Xi)` (`sklearn/preprocessing/_encoders.py:99`, appended at `:164`), which
sorts NUMERICALLY for int/float input.

For X = [[1], [2], [10]]:
  sklearn:    categories_ = [array([ 1,  2, 10])] (int64)  -> transform [[0],[1],[2]]
  ferrolearn: categories_ = [array(['1','10','2'])] (str)  -> transform [[0],[2],[1]]
because string-sort orders '1' < '10' < '2'. This is a SILENT wrong-ordinal
divergence: the binding ACCEPTS numeric input and returns a *different* (incorrect)
encoding rather than erroring, so the documented "string-only scope, numeric coerced
to str" claim hides an observable mismatch that breaks every numeric downstream model.
This is NOT documented scope (no NotImplementedError is raised); it is a release
blocker.

Tracking: #2230
"""

import numpy as np
import pytest

import ferrolearn as fl
from sklearn.preprocessing import OrdinalEncoder as SkOrd


@pytest.mark.skip(
    reason="divergence: OrdinalEncoder._to_rows .astype(str) string-sorts numeric "
    "input -> wrong ordinals vs sklearn numeric sort; tracking #2230"
)
def test_numeric_int_input_ordinal_matches_sklearn():
    # Integers whose string order (1, 10, 2) differs from their numeric order
    # (1, 2, 10) -- the minimal input that exposes the .astype(str) bug.
    X = [[1], [2], [10]]

    sk = SkOrd().fit(X)
    sk_out = sk.transform(X)

    fe = fl.OrdinalEncoder().fit(X)
    fe_out = np.asarray(fe.transform(X))

    # sklearn sorts numerically: categories_ == [1, 2, 10] -> transform [[0],[1],[2]].
    # ferrolearn string-sorts: categories_ == ['1','10','2'] -> transform [[0],[2],[1]].
    np.testing.assert_array_equal(fe_out, sk_out)
