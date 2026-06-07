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


@pytest.mark.xfail(
    reason="ferrolearn DBSCAN.components_ is float64 for a float32 X; sklearn "
    "preserves float32 (X[core].copy(), _dbscan.py:442); tracking #2191",
    strict=True,
)
def test_components_dtype_preserves_float32_like_sklearn():
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

    # The divergence: ferrolearn must mirror sklearn's components_ dtype.
    assert f.components_.dtype == s.components_.dtype
