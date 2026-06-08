"""Divergence pin: ferrolearn `QuantileTransformer.transform` accepts +-inf and
silently maps it (+inf -> 1.0, -inf -> 0.0), whereas sklearn's `transform`
raises a ValueError on a non-finite (inf) input.

sklearn cite: `sklearn/preprocessing/_data.py:2926` `transform` ->
`_data.py:2943` `X = self._check_inputs(X, in_fit=False, copy=self.copy)` ->
`_data.py:2870-2879` `_validate_data(..., force_all_finite="allow-nan")`.
`force_all_finite="allow-nan"` permits NaN but REJECTS +-inf, so sklearn's
forward `transform` raises:
    ValueError: Input X contains infinity or a value too large for dtype('...').

ferrolearn cite: the Rust core's forward path
`ferrolearn-preprocess/src/quantile_transformer.rs:560`
`FittedQuantileTransformer::transform` has NO inf guard (only the *inverse*
path `:225` guards `x.iter().any(|v| v.is_infinite())`), and the binding
`ferrolearn-python/src/extras.rs:3579` `RsQuantileTransformer::transform`
forwards directly with no finite check. The Python wrapper
`_extras.py:2538 transform` likewise does not check. Result: a +-inf input is
silently transformed (interpolate_cdf clamps +inf -> references[-1] == 1.0,
-inf -> references[0] == 0.0) instead of raising.

Input: a single feature with one +inf row.
Expected (live sklearn 1.5.2 oracle): `transform` raises ValueError.
Actual (ferrolearn): no raise; returns [..., 1.0, ...] for +inf and 0.0 for -inf.

Note: the *inverse* path already matches sklearn (raises on inf, #2212) — only
the FORWARD transform path diverges.

Tracking: #2217
"""

import warnings

import numpy as np
import pytest

import ferrolearn as fl
from sklearn.preprocessing import QuantileTransformer as SkQT


@pytest.mark.skip(reason="divergence: transform accepts +-inf instead of raising "
                         "ValueError (sklearn _data.py:2879 force_all_finite="
                         "'allow-nan'); tracking #2217")
def test_transform_inf_raises_value_error_like_sklearn():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X = np.linspace(0.0, 1.0, 20).reshape(-1, 1)
        fqt = fl.QuantileTransformer(n_quantiles=10).fit(X)
        sqt = SkQT(n_quantiles=10, subsample=None).fit(X)

    X_inf = np.array([[0.5], [np.inf], [0.3]])

    # Oracle: sklearn's forward transform rejects the inf row.
    with pytest.raises(ValueError):
        sqt.transform(X_inf)

    # ferrolearn must raise the SAME ValueError. Today it silently maps
    # +inf -> 1.0 (interpolate_cdf clamp), so this assertion fails.
    with pytest.raises(ValueError):
        fqt.transform(X_inf)
