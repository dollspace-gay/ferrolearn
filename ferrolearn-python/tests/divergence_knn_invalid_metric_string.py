"""Divergence: KNeighbors `metric` invalid-string exception type.

sklearn validates the `metric` parameter of KNeighborsClassifier /
KNeighborsRegressor against a FIXED set of metric names (a `StrOptions`
constraint over `VALID_METRICS`, `sklearn/neighbors/_base.py:402`). A string
that is NOT one of those names raises `InvalidParameterError`, which IS a
subclass of `ValueError`.

ferrolearn's shim (`_classifiers.py:434` / `_extras.py:367`) forwards the raw
metric string to the Rust binding, whose catch-all branch
(`classifiers.rs:487-492`, `extras.rs:634-639`) raises `NotImplementedError`
(the #876 "valid-but-unsupported metric" path) for EVERY string that is not
`euclidean` / `l2` / `minkowski`. `NotImplementedError` is NOT a subclass of
`ValueError`.

The divergence is user-observable: an INVALID metric string (one sklearn
itself rejects, e.g. ``''`` or ``'nonsense_metric'``) must raise a
`ValueError` (the standard sklearn idiom `except ValueError:` relies on this).
ferrolearn raises `NotImplementedError` instead, which `except ValueError:`
does NOT catch.

This is distinct from the honest #876 rejection of a VALID-but-unsupported
metric such as ``'cityblock'`` (which IS in sklearn's set): for those,
`NotImplementedError` is the documented contract and NOT a divergence. The
tests below use ONLY strings that are NOT in sklearn's valid set, confirmed
live against the oracle, so the expected behavior is unambiguously ValueError.

Tracking: #3
"""

import numpy as np
import pytest

import ferrolearn as fl

sklearn_neighbors = pytest.importorskip("sklearn.neighbors")

# Strings that are NOT in sklearn's KNeighbors metric StrOptions set, so sklearn
# rejects them as an invalid parameter (InvalidParameterError ⊂ ValueError),
# rather than treating them as a recognized-but-unsupported metric.
INVALID_METRIC_STRINGS = ["", "nonsense_metric", "not_a_real_metric", "L2 "]

X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
Y_CLF = np.array([0, 1, 1, 0, 1])
Y_REG = np.array([10.0, 20.0, 30.0, 40.0, 50.0])


def _sklearn_rejects_as_valueerror(estimator_cls, y, metric):
    """Oracle: confirm live sklearn rejects this metric with a ValueError."""
    with pytest.raises(ValueError):
        estimator_cls(n_neighbors=3, metric=metric).fit(X, y)


@pytest.mark.parametrize("metric", INVALID_METRIC_STRINGS)
def test_knn_clf_invalid_metric_string_is_valueerror(metric):
    """An invalid metric string is a ValueError in sklearn; ferrolearn raises
    NotImplementedError (not a ValueError subclass) — divergence #3."""
    # Oracle: sklearn rejects with a ValueError subclass.
    _sklearn_rejects_as_valueerror(sklearn_neighbors.KNeighborsClassifier, Y_CLF, metric)
    # ferrolearn must mirror sklearn: an INVALID metric string is a ValueError.
    with pytest.raises(ValueError):
        fl.KNeighborsClassifier(n_neighbors=3, metric=metric).fit(X, Y_CLF)


@pytest.mark.parametrize("metric", INVALID_METRIC_STRINGS)
def test_knn_reg_invalid_metric_string_is_valueerror(metric):
    """An invalid metric string is a ValueError in sklearn; ferrolearn raises
    NotImplementedError (not a ValueError subclass) — divergence #3."""
    _sklearn_rejects_as_valueerror(sklearn_neighbors.KNeighborsRegressor, Y_REG, metric)
    with pytest.raises(ValueError):
        fl.KNeighborsRegressor(n_neighbors=3, metric=metric).fit(X, Y_REG)
