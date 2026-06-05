"""Divergence guards for ferrolearn-python/src/conversions.rs (unit #2024).

Verification model B (goal.md): every expected value is computed by the LIVE
sklearn 1.5.2 oracle in the same test and compared against `import ferrolearn`.
No expected value is literal-copied from the ferrolearn side (R-CHAR-3).

These pin the numpy <-> ndarray PyO3 marshalling contract of conversions.rs:
  - f64 2D/1D round-trip (dtype + shape + value preservation through an estimator)
  - i64 <-> usize label marshalling guarded by the wrapper's _encode_labels:
    negative / string / non-contiguous labels survive (the `v as usize` cast
    in conversions.rs never sees them un-encoded).

The structural NOT-STARTED gaps (float32 dtype narrowing, sparse input, ferray
substrate) are filed as -l blocker crosslink issues, NOT pinned here, because no
f32/sparse marshalling path exists in conversions.rs to fix locally.
"""

import numpy as np
import pytest

import ferrolearn as fl
from sklearn.linear_model import LinearRegression as SkLinearRegression
from sklearn.linear_model import LogisticRegression as SkLogisticRegression
from sklearn.preprocessing import StandardScaler as SkStandardScaler


# --- f64 round-trip: numpy(f64) -> Array{1,2}<f64> -> numpy(f64) ---


def test_f64_2d_roundtrip_scaler_matches_sklearn():
    """StandardScaler.transform round-trips a float64 2D array through
    numpy2_to_ndarray -> ndarray2_to_numpy. Oracle = sklearn StandardScaler.
    """
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])

    expected = SkStandardScaler().fit_transform(X)
    actual = fl.StandardScaler().fit_transform(X)

    assert actual.dtype == np.float64  # dtype preserved across the boundary
    assert actual.shape == expected.shape == X.shape  # shape preserved
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-8)


def test_f64_1d_roundtrip_predict_matches_sklearn():
    """LinearRegression.predict round-trips a float64 array out through
    ndarray1_to_numpy. Oracle = sklearn LinearRegression.
    """
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([2.0, 4.0, 6.0, 8.0])
    X_test = np.array([[5.0], [6.0], [10.0]])

    expected = SkLinearRegression().fit(X, y).predict(X_test)
    actual = fl.LinearRegression().fit(X, y).predict(X_test)

    assert actual.dtype == np.float64
    assert actual.shape == expected.shape == (3,)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-8)


# --- i64 <-> usize label marshalling (wrapper-guarded `v as usize`) ---


def test_negative_integer_labels_match_sklearn():
    """Negative labels [-1, 1]: sklearn keeps them in classes_ and predict.
    ferrolearn must match because _encode_labels maps [-1,1] -> [0,1] BEFORE
    the `v as usize` cast in conversions.rs (R-CODE-5: no silent wrap to u64::MAX).
    """
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = [-1, 1, -1, 1]

    sk = SkLogisticRegression().fit(X, y)
    fr = fl.LogisticRegression().fit(X, y)

    assert fr.classes_.tolist() == sk.classes_.tolist() == [-1, 1]
    # the cast hazard would surface here as 18446744073709551615
    assert 18446744073709551615 not in fr.classes_.tolist()
    np.testing.assert_array_equal(fr.predict(X), sk.predict(X))


def test_string_labels_match_sklearn():
    """String labels: an i64 marshaller cannot carry them, so the wrapper's
    _encode_labels/_decode_labels must handle them above the Rust boundary.
    Oracle = sklearn classes_ + predict.
    """
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = ["cat", "dog", "cat", "dog"]

    sk = SkLogisticRegression().fit(X, y)
    fr = fl.LogisticRegression().fit(X, y)

    assert fr.classes_.tolist() == sk.classes_.tolist() == ["cat", "dog"]
    np.testing.assert_array_equal(fr.predict(X), sk.predict(X))
    np.testing.assert_array_equal(
        fr.predict(np.array([[0.5]])), sk.predict(np.array([[0.5]]))
    )


def test_noncontiguous_integer_labels_match_sklearn():
    """Non-contiguous labels [0, 2] (gap at 1): classes_ stays [0, 2] and the
    decode maps the internal 0..n-1 encoding back. Oracle = sklearn.
    """
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = [0, 2, 0, 2]

    sk = SkLogisticRegression().fit(X, y)
    fr = fl.LogisticRegression().fit(X, y)

    assert fr.classes_.tolist() == sk.classes_.tolist() == [0, 2]
    np.testing.assert_array_equal(fr.predict(X), sk.predict(X))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
