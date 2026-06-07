"""Divergence pin: ferrolearn.OPTICS does not enforce sklearn's
``min_cluster_size >= 2`` integer lower bound (#2199).

Verification model B (goal.md): compare ``import ferrolearn`` against
``import sklearn`` 1.5.2 (the installed package IS the oracle). The expected
behavior is produced by a LIVE ``sklearn.cluster.OPTICS`` call here (R-CHAR-3 —
never copied from the ferrolearn side).

Divergence
----------
sklearn ``OPTICS._parameter_constraints["min_cluster_size"]``
(``sklearn/cluster/_optics.py:255-259``) is::

    "min_cluster_size": [
        Interval(Integral, 2, None, closed="left"),   # int >= 2
        Interval(RealNotInt, 0, 1, closed="right"),    # float in (0, 1]
        None,
    ],

so an INTEGER ``min_cluster_size < 2`` (e.g. 0 or 1) is an
``InvalidParameterError`` (a ``ValueError`` subclass) raised at fit:

    >>> SkOPTICS(min_samples=2, min_cluster_size=1).fit(X)
    InvalidParameterError: The 'min_cluster_size' parameter of OPTICS must be
    an int in the range [2, inf), ...

ferrolearn's ``_extras.py::OPTICS._validate_unsupported`` only rejects a FLOAT
``min_cluster_size`` (the documented (0, 1] fraction, REQ-10 NOT-STARTED #1088);
``_extras.py:1684-1692``). There is NO lower-bound check for an INTEGER
``min_cluster_size``: an int < 2 is threaded straight to ``_RsOPTICS``
(``min_cluster_size: Option<usize>``, ``extras.rs:2798/2863-2864``) and the fit
SUCCEEDS. ferrolearn therefore ACCEPTS where sklearn REJECTS — an undocumented
divergence (the design-doc REQ row claims the binding "rejects the
core-unsupported surface BEFORE the ABI", but the int lower bound is not
covered).

Tracking: #2199
"""

import numpy as np
import pytest
from sklearn.cluster import OPTICS as SkOPTICS

import ferrolearn as fl

# The `_optics.py:891-896` doctest fixture.
DOCTEST_X = np.array(
    [[1, 2], [2, 5], [3, 6], [8, 7], [8, 8], [7, 3]], dtype=np.float64
)


@pytest.mark.parametrize("mcs", [0, 1])
def test_int_min_cluster_size_below_two_raises_like_sklearn(mcs):
    """Fixed (#2199): `_validate_unsupported` now rejects an integer
    min_cluster_size < 2 with ValueError, matching sklearn's
    `Interval(Integral, 2, None)` constraint."""
    # LIVE ORACLE (R-CHAR-3): sklearn rejects an integer min_cluster_size < 2.
    with pytest.raises(ValueError):
        SkOPTICS(min_samples=2, min_cluster_size=mcs).fit(DOCTEST_X)

    # ferrolearn mirrors that rejection.
    with pytest.raises(ValueError):
        fl.OPTICS(min_samples=2, min_cluster_size=mcs).fit(DOCTEST_X)
