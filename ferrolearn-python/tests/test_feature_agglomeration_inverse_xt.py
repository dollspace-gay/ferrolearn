"""Divergence pin for ferrolearn.FeatureAgglomeration.inverse_transform (#943).

Verification model B (goal.md): compare `import ferrolearn` against the installed
`import sklearn` 1.5.2 oracle. The expected behaviour comes from the live
`sklearn.cluster.FeatureAgglomeration` (R-CHAR-3) — NEVER copied from ferrolearn.

Divergence
----------
sklearn 1.5.2 `FeatureAgglomeration.inverse_transform` has the signature
``inverse_transform(self, X=None, *, Xt=None)``
(``sklearn/cluster/_feature_agglomeration.py:66``) and routes the deprecated
``Xt=`` keyword through
``_deprecate_Xt_in_inverse_transform(X, Xt)``
(``sklearn/cluster/_feature_agglomeration.py:87``, import at line 15) so that
``inverse_transform(Xt=Xred)`` returns the inverse and emits a ``FutureWarning``
("Xt was deprecated in 1.5 and will be removed in 1.7. Use X instead.").

ferrolearn's wrapper signature is ``inverse_transform(self, Xred)``
(``ferrolearn-python/python/ferrolearn/_extras.py:1955``): it neither accepts the
deprecated ``Xt=`` keyword NOR the canonical ``X=`` keyword, so
``inverse_transform(Xt=...)`` raises ``TypeError`` instead of returning the inverse.
This breaks sklearn-ecosystem callers that pass ``Xt=`` (the still-supported 1.5
deprecation path) through the binding.

Tracking: #2188
"""

import warnings

import numpy as np

import ferrolearn as fl
from sklearn.cluster import FeatureAgglomeration as Sk


def _fixture():
    rng = np.random.RandomState(0)
    base = rng.randn(8, 3) * 10.0
    X = np.empty((8, 6))
    X[:, 0] = base[:, 0]
    X[:, 1] = base[:, 0] + rng.randn(8) * 0.01
    X[:, 2] = base[:, 1]
    X[:, 3] = base[:, 1] + rng.randn(8) * 0.01
    X[:, 4] = base[:, 2]
    X[:, 5] = base[:, 2] + rng.randn(8) * 0.01
    return X


def test_inverse_transform_accepts_deprecated_Xt_kwarg():
    """Fixed (#2188): the wrapper now mirrors sklearn's
    ``inverse_transform(self, X=None, *, Xt=None)`` — it accepts the deprecated
    ``Xt=`` keyword (routed via ``_deprecate_Xt_in_inverse_transform``, emitting a
    FutureWarning) and the canonical ``X=`` keyword."""
    X = _fixture()

    # Live sklearn oracle: inverse_transform(Xt=...) returns the inverse and
    # warns FutureWarning (R-CHAR-3 — expected comes from sklearn, not ferrolearn).
    sk = Sk(n_clusters=3).fit(X)
    Xt_sk = sk.transform(X)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        sk_inv = sk.inverse_transform(Xt=Xt_sk)
    assert any(issubclass(rec.category, FutureWarning) for rec in w)
    assert sk_inv.shape == (8, 6)

    # ferrolearn mirrors: accept Xt= (with FutureWarning) and return the same inverse.
    fa = fl.FeatureAgglomeration(3).fit(X)
    Xt_fa = fa.transform(X)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        fa_inv = fa.inverse_transform(Xt=Xt_fa)
    assert any(issubclass(rec.category, FutureWarning) for rec in w)
    np.testing.assert_allclose(fa_inv, sk_inv, rtol=0, atol=1e-9)

    # The canonical X= keyword and positional both work and agree.
    np.testing.assert_allclose(fa.inverse_transform(X=Xt_fa), sk_inv, rtol=0, atol=1e-9)
    np.testing.assert_allclose(fa.inverse_transform(Xt_fa), sk_inv, rtol=0, atol=1e-9)
