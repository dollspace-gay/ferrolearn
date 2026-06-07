"""Divergence pin (#2190 follow-up): `ferrolearn.DBSCAN.components_` does NOT
preserve a float32 input dtype, whereas `sklearn.cluster.DBSCAN` does.

sklearn `sklearn/cluster/_dbscan.py:442-446`:
    if len(self.core_sample_indices_):
        self.components_ = X[self.core_sample_indices_].copy()
    else:
        self.components_ = np.empty((0, X.shape[1]))

DBSCAN does NOT upcast X in `_validate_data` (no `dtype=np.float64` cast), so a
float32 `X` flows through to `components_ = X[core].copy()` AS float32. The
fitted `components_` therefore has dtype float32 for a float32 input.

ferrolearn's PyO3 binding (`ferrolearn-python/src/extras.rs:2507-2524`) accepts
`x: PyReadonlyArray2<'_, f64>` and the `_extras.py::DBSCAN.fit` wrapper coerces
via `_f64(X)`, so the Rust core only ever sees float64. The `components_`
getter (`extras.rs:2554-2561`) returns `PyArray2<f64>`. Result: for a float32
`X`, ferrolearn's `components_.dtype` is float64 (sklearn: float32) — a silent
dtype divergence that breaks any downstream consumer keying on the dtype.

Verification model B (goal.md R-CHAR-3): the expected dtype comes from the LIVE
installed sklearn 1.5.2 oracle fit in the SAME process, never copied from
ferrolearn.

Tracking: #2191
"""

import numpy as np
import pytest
from sklearn.cluster import DBSCAN as SkDBSCAN

import ferrolearn as fl


def test_components_dtype_preserves_float32_like_sklearn():
    """Fixed (#2191): the wrapper now takes ``components_`` from the validated
    input array (``X[core].copy()``) preserving its float32 dtype, not the f64
    Rust core copy."""
    # Two dense clusters + one noise outlier, supplied as float32.
    X = np.array(
        [
            [0.0, 0.0], [0.3, 0.0], [0.0, 0.3],
            [5.0, 5.0], [5.3, 5.0],
            [50.0, 50.0],
        ],
        dtype=np.float32,
    )

    s = SkDBSCAN(eps=0.5, min_samples=2).fit(X)
    f = fl.DBSCAN(0.5, min_samples=2).fit(X)

    # Oracle anchor: sklearn keeps the float32 dtype through components_.
    assert s.components_.dtype == np.float32

    # ferrolearn mirrors sklearn's components_ dtype AND values.
    assert f.components_.dtype == s.components_.dtype
    np.testing.assert_array_equal(f.components_, s.components_)


def test_one_dim_X_raises_valueerror_like_sklearn():
    """Fixed (#2191): a 1-D X raises ValueError (sklearn check_array), not the
    TypeError pyo3 would raise rejecting a 1-D array against PyReadonlyArray2."""
    x1d = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        SkDBSCAN(eps=0.5).fit(x1d)
    with pytest.raises(ValueError):
        fl.DBSCAN(0.5).fit(x1d)


def test_two_dim_sample_weight_raises_valueerror_like_sklearn():
    """Fixed (#2191): a 2-D column sample_weight raises ValueError
    (_check_sample_weight: "must be 1D array or scalar"), not TypeError."""
    X = np.array([[0.0, 0.0], [0.3, 0.0], [5.0, 5.0]])
    sw2d = np.array([[5.0], [1.0], [1.0]])
    with pytest.raises(ValueError):
        SkDBSCAN(eps=0.5, min_samples=2).fit(X, sample_weight=sw2d)
    with pytest.raises(ValueError):
        fl.DBSCAN(0.5, min_samples=2).fit(X, sample_weight=sw2d)


def test_scalar_sample_weight_broadcasts_like_sklearn():
    """sklearn _check_sample_weight broadcasts a scalar sample_weight to all n
    samples; the wrapper must mirror that (not error on a length-1 array)."""
    X = np.array([[0.0, 0.0], [0.3, 0.0], [0.0, 0.3], [5.0, 5.0], [5.3, 5.0]])
    s = SkDBSCAN(eps=0.5, min_samples=2).fit(X, sample_weight=2.0)
    f = fl.DBSCAN(0.5, min_samples=2).fit(X, sample_weight=2.0)
    np.testing.assert_array_equal(np.asarray(f.labels_), s.labels_)
    np.testing.assert_array_equal(
        np.asarray(f.core_sample_indices_), s.core_sample_indices_
    )
