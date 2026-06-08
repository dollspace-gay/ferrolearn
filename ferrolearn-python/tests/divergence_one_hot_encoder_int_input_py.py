"""Live-sklearn divergence pin for `ferrolearn.OneHotEncoder` on INTEGER input
(#1155, REQ-8). Tracking: #2228.

Divergence: ferrolearn's `OneHotEncoder` (ferrolearn-python/src/extras.rs
`_RsOneHotEncoder` + ferrolearn-python/python/ferrolearn/_extras.py class
`OneHotEncoder`) silently coerces integer input X to float64 at the `_f64(X)`
ABI boundary (R-DEV-3, f64-only core). Two SILENT observable divergences from
`sklearn.preprocessing.OneHotEncoder(sparse_output=False)` 1.5.2 result:

  1. `get_feature_names_out()` (no-arg, the path the binding CLAIMS to match
     sklearn): sklearn emits the INT category label `'x0_2'`
     (sklearn/preprocessing/_encoders.py:1217
     `name_combiner(input_features[i], t)` with `t` drawn from the int64
     `categories_`); ferrolearn emits the FLOAT label `'x0_2.0'`.
  2. `categories_` dtype: sklearn preserves int64 for int input
     (`_encoders.py` `categories_`); ferrolearn returns float64.

Neither is surfaced as an error/warning — unlike the documented-scope gaps
(`sparse_output`/`input_features`/unsupported-params), which raise. The existing
`test_get_feature_names_out_matches_sklearn` covers only FLOAT input, so it
misses this.

Verification model B (R-CHAR-3): the expected values come from the live sklearn
1.5.2 oracle in-test; nothing is literal-copied from the ferrolearn side.
"""

import numpy as np
import pytest

import ferrolearn as fl
from sklearn.preprocessing import OneHotEncoder as SkOHE


def test_int_input_feature_names_and_categories_dtype_match_sklearn():
    # Integer input. sklearn keeps int64 categories_ and int feature-name labels.
    X = np.array([[2, 0], [5, 1], [9, 0]], dtype=np.int64)

    sk = SkOHE(sparse_output=False).fit(X)
    fl_enc = fl.OneHotEncoder(sparse_output=False).fit(X)

    # (1) No-arg get_feature_names_out: the path the binding claims matches
    #     sklearn. sklearn -> ['x0_2', ...] (int label); ferrolearn -> ['x0_2.0', ...].
    np.testing.assert_array_equal(
        np.asarray(fl_enc.get_feature_names_out()),
        np.asarray(sk.get_feature_names_out()),
    )

    # (2) categories_ dtype: sklearn preserves int64; ferrolearn returns float64.
    for fl_c, sk_c in zip(fl_enc.categories_, sk.categories_):
        assert fl_c.dtype == sk_c.dtype, (fl_c.dtype, sk_c.dtype)
