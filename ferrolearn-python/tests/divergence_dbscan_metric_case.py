"""Divergence pin (#2194): `ferrolearn.DBSCAN` metric string is CASE-INSENSITIVE
(`resolve_dbscan_metric` lowercases via `metric.to_lowercase()`,
`ferrolearn-python/src/extras.rs:2500`) whereas `sklearn.cluster.DBSCAN`'s
`_parameter_constraints["metric"]` is
`StrOptions(set(_VALID_METRICS) | {"precomputed"})`
(`sklearn/cluster/_dbscan.py:334-337`). `_VALID_METRICS` holds only LOWERCASE
metric names, and `StrOptions` membership is EXACT, so a capitalized string like
`'Euclidean'` is NOT in the option set and sklearn raises
`InvalidParameterError` (a `ValueError` subclass) at fit.

Verification model B (goal.md R-CHAR-3): the LIVE installed sklearn 1.5.2 oracle
is the contract. We assert the sklearn behavior (REJECT 'Euclidean' with a
ValueError); ferrolearn currently ACCEPTS it and fits, so this is a divergence.

This is NOT a documented NOT-STARTED item: the design doc / REQ table describe
the lowercased-alias mapping as the intended behavior but never state that
ferrolearn intentionally widens sklearn's accepted set to case-insensitive — so
ferrolearn silently accepts input sklearn rejects.
"""

import numpy as np
import pytest
from sklearn.cluster import DBSCAN as SkDBSCAN

import ferrolearn as fl

_X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])


@pytest.mark.parametrize("metric", ["Euclidean", "EUCLIDEAN", "Manhattan", "L2"])
def test_capitalized_metric_rejected_like_sklearn(metric):
    """Fixed (#2194): `resolve_dbscan_metric` matches the raw string (no
    `to_lowercase`), so a capitalized metric falls through to the ValueError
    arm, matching sklearn's exact (case-sensitive) `StrOptions` membership."""
    # Live oracle (the contract): sklearn 1.5.2 rejects a capitalized metric
    # string as InvalidParameterError (a ValueError subclass), because its
    # StrOptions set only contains the lowercase _VALID_METRICS names.
    with pytest.raises(ValueError):
        SkDBSCAN(eps=1.5, min_samples=2, metric=metric).fit(_X)

    # ferrolearn MUST mirror that rejection. It currently does NOT: the Rust
    # `resolve_dbscan_metric` lowercases the string and fits, so this raise does
    # not happen -> the test body completes without raising -> xfail captures
    # the divergence (and `strict=True` flips to a hard failure once fixed).
    with pytest.raises(ValueError):
        fl.DBSCAN(1.5, min_samples=2, metric=metric).fit(_X)
