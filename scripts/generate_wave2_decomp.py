#!/usr/bin/env python3
"""Wave 2 — sklearn reference fixtures for previously-untested decomposition models.

Adds: TruncatedSVD, FastICA, KernelPCA, FactorAnalysis, IncrementalPCA,
SparsePCA, DictionaryLearning, MiniBatchNMF, LatentDirichletAllocation,
CCA, PLSRegression, PLSCanonical, Isomap, MDS, LocallyLinearEmbedding,
SpectralEmbedding, TSNE.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import sklearn
from sklearn.cross_decomposition import CCA, PLSCanonical, PLSRegression
from sklearn.datasets import make_blobs, make_classification, make_regression
from sklearn.decomposition import (
    DictionaryLearning,
    FactorAnalysis,
    FastICA,
    IncrementalPCA,
    KernelPCA,
    LatentDirichletAllocation,
    MiniBatchNMF,
    SparsePCA,
    TruncatedSVD,
)
from sklearn.manifold import (
    MDS,
    TSNE,
    Isomap,
    LocallyLinearEmbedding,
    SpectralEmbedding,
)

SKLEARN_VERSION = sklearn.__version__
SKLEARN_PIN = f"scikit-learn=={SKLEARN_VERSION}"

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "fixtures"


def _to_list(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if hasattr(x, "tolist"):
        return x.tolist()
    return x


def write_fixture(name: str, payload: dict) -> None:
    payload.setdefault("sklearn_version", SKLEARN_VERSION)
    payload.setdefault("sklearn_pin", SKLEARN_PIN)
    out = FIXTURES / f"{name}.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"  wrote {out.relative_to(ROOT)}")


def gen_truncated_svd():
    rng = np.random.RandomState(42)
    X = rng.randn(80, 10)
    model = TruncatedSVD(n_components=3, random_state=42)
    Xt = model.fit_transform(X)
    write_fixture(
        "truncated_svd",
        {
            "description": "TruncatedSVD with n_components=3",
            "random_state": 42,
            "input": {"X": _to_list(X)},
            "params": {"n_components": 3, "random_state": 42},
            "expected": {
                "components": _to_list(model.components_),
                "singular_values": _to_list(model.singular_values_),
                "explained_variance": _to_list(model.explained_variance_),
                "explained_variance_ratio": _to_list(model.explained_variance_ratio_),
                "transformed": _to_list(Xt),
            },
        },
    )


def gen_fast_ica():
    rng = np.random.RandomState(42)
    S = np.column_stack([np.sin(np.linspace(0, 8, 200)), np.sign(np.sin(np.linspace(0, 5, 200))), rng.randn(200)])
    A = np.array([[1.0, 1.0, 1.0], [0.5, 2.0, 1.0], [1.5, 1.0, 2.0]])
    X = S @ A.T
    model = FastICA(n_components=3, random_state=42, max_iter=500, tol=1e-5)
    S_est = model.fit_transform(X)
    write_fixture(
        "fast_ica",
        {
            "description": "FastICA on mixed sine+square+noise sources",
            "random_state": 42,
            "input": {"X": _to_list(X)},
            "params": {"n_components": 3, "random_state": 42, "max_iter": 500, "tol": 1e-5},
            "expected": {
                "components": _to_list(model.components_),
                "mixing": _to_list(model.mixing_),
                "mean": _to_list(model.mean_),
                "transformed": _to_list(S_est),
            },
        },
    )


def gen_kernel_pca():
    rng = np.random.RandomState(42)
    X = rng.randn(80, 5)
    model = KernelPCA(n_components=3, kernel="rbf", gamma=0.1, random_state=42)
    Xt = model.fit_transform(X)
    write_fixture(
        "kernel_pca",
        {
            "description": "KernelPCA with RBF kernel",
            "random_state": 42,
            "input": {"X": _to_list(X)},
            "params": {"n_components": 3, "kernel": "rbf", "gamma": 0.1, "random_state": 42},
            "expected": {
                "eigenvalues": _to_list(model.eigenvalues_),
                "transformed": _to_list(Xt),
            },
        },
    )


def gen_factor_analysis():
    rng = np.random.RandomState(42)
    X = rng.randn(100, 6)
    model = FactorAnalysis(n_components=3, random_state=42, max_iter=1000, tol=1e-3)
    Xt = model.fit_transform(X)
    write_fixture(
        "factor_analysis",
        {
            "description": "FactorAnalysis with EM",
            "random_state": 42,
            "input": {"X": _to_list(X)},
            "params": {"n_components": 3, "random_state": 42, "max_iter": 1000, "tol": 1e-3},
            "expected": {
                "components": _to_list(model.components_),
                "mean": _to_list(model.mean_),
                "noise_variance": _to_list(model.noise_variance_),
                "transformed": _to_list(Xt),
            },
        },
    )


def gen_incremental_pca():
    rng = np.random.RandomState(42)
    X = rng.randn(80, 5)
    model = IncrementalPCA(n_components=3, batch_size=20)
    Xt = model.fit_transform(X)
    write_fixture(
        "incremental_pca",
        {
            "description": "IncrementalPCA with batch_size=20",
            "random_state": 42,
            "input": {"X": _to_list(X)},
            "params": {"n_components": 3, "batch_size": 20},
            "expected": {
                "components": _to_list(model.components_),
                "explained_variance": _to_list(model.explained_variance_),
                "explained_variance_ratio": _to_list(model.explained_variance_ratio_),
                "mean": _to_list(model.mean_),
                "transformed": _to_list(Xt),
            },
        },
    )


def gen_sparse_pca():
    rng = np.random.RandomState(42)
    X = rng.randn(60, 8)
    model = SparsePCA(n_components=3, alpha=1.0, random_state=42, max_iter=500)
    Xt = model.fit_transform(X)
    write_fixture(
        "sparse_pca",
        {
            "description": "SparsePCA with alpha=1.0",
            "random_state": 42,
            "input": {"X": _to_list(X)},
            "params": {"n_components": 3, "alpha": 1.0, "random_state": 42, "max_iter": 500},
            "expected": {
                "components": _to_list(model.components_),
                "transformed": _to_list(Xt),
            },
        },
    )


def gen_dictionary_learning():
    rng = np.random.RandomState(42)
    X = rng.randn(60, 8)
    model = DictionaryLearning(n_components=4, alpha=1.0, random_state=42, max_iter=200)
    Xt = model.fit_transform(X)
    write_fixture(
        "dictionary_learning",
        {
            "description": "DictionaryLearning with alpha=1.0",
            "random_state": 42,
            "input": {"X": _to_list(X)},
            "params": {"n_components": 4, "alpha": 1.0, "random_state": 42, "max_iter": 200},
            "expected": {
                "components": _to_list(model.components_),
                "transformed": _to_list(Xt),
            },
        },
    )


def gen_mini_batch_nmf():
    rng = np.random.RandomState(42)
    X = np.abs(rng.randn(60, 8))
    model = MiniBatchNMF(n_components=3, init="nndsvd", random_state=42, max_iter=300, batch_size=16)
    W = model.fit_transform(X)
    write_fixture(
        "mini_batch_nmf",
        {
            "description": "MiniBatchNMF with nndsvd init",
            "random_state": 42,
            "input": {"X": _to_list(X)},
            "params": {"n_components": 3, "init": "nndsvd", "random_state": 42, "max_iter": 300, "batch_size": 16},
            "expected": {
                "components": _to_list(model.components_),
                "W": _to_list(W),
                "reconstruction_error": float(model.reconstruction_err_),
            },
        },
    )


def gen_lda_topic_model():
    rng = np.random.RandomState(42)
    n_topics = 3
    n_docs = 60
    n_terms = 12
    # Synthetic topic-mixed counts.
    X = (rng.rand(n_docs, n_terms) * 10).astype(int) + 1
    model = LatentDirichletAllocation(
        n_components=n_topics, random_state=42, max_iter=20, learning_method="batch"
    )
    Xt = model.fit_transform(X)
    write_fixture(
        "latent_dirichlet_allocation",
        {
            "description": "LDA topic model on synthetic word counts",
            "random_state": 42,
            "input": {"X": _to_list(X)},
            "params": {"n_components": 3, "random_state": 42, "max_iter": 20, "learning_method": "batch"},
            "expected": {
                "components": _to_list(model.components_),
                "transformed": _to_list(Xt),
                "perplexity": float(model.perplexity(X)),
            },
        },
    )


def gen_cca():
    rng = np.random.RandomState(42)
    n = 60
    X = rng.randn(n, 5)
    # Y correlated to X via a linear map + noise.
    Y = X @ rng.randn(5, 3) + 0.1 * rng.randn(n, 3)
    model = CCA(n_components=2)
    model.fit(X, Y)
    Xc, Yc = model.transform(X, Y)
    write_fixture(
        "cca",
        {
            "description": "CCA with 2 components",
            "random_state": 42,
            "input": {"X": _to_list(X), "Y": _to_list(Y)},
            "params": {"n_components": 2},
            "expected": {
                "x_weights": _to_list(model.x_weights_),
                "y_weights": _to_list(model.y_weights_),
                "x_transformed": _to_list(Xc),
                "y_transformed": _to_list(Yc),
            },
        },
    )


def gen_pls_regression():
    rng = np.random.RandomState(42)
    X = rng.randn(60, 5)
    Y = X @ rng.randn(5, 2) + 0.1 * rng.randn(60, 2)
    model = PLSRegression(n_components=2)
    model.fit(X, Y)
    Y_pred = model.predict(X)
    write_fixture(
        "pls_regression",
        {
            "description": "PLSRegression with 2 components",
            "random_state": 42,
            "input": {"X": _to_list(X), "Y": _to_list(Y)},
            "params": {"n_components": 2},
            "expected": {
                "x_weights": _to_list(model.x_weights_),
                "y_weights": _to_list(model.y_weights_),
                "predictions": _to_list(Y_pred),
            },
        },
    )


def gen_pls_canonical():
    rng = np.random.RandomState(42)
    X = rng.randn(60, 5)
    Y = X @ rng.randn(5, 3) + 0.1 * rng.randn(60, 3)
    model = PLSCanonical(n_components=2)
    model.fit(X, Y)
    Xc, Yc = model.transform(X, Y)
    write_fixture(
        "pls_canonical",
        {
            "description": "PLSCanonical with 2 components",
            "random_state": 42,
            "input": {"X": _to_list(X), "Y": _to_list(Y)},
            "params": {"n_components": 2},
            "expected": {
                "x_weights": _to_list(model.x_weights_),
                "y_weights": _to_list(model.y_weights_),
                "x_transformed": _to_list(Xc),
                "y_transformed": _to_list(Yc),
            },
        },
    )


# ---------------------------------------------------------------------------
# Manifold learning
# ---------------------------------------------------------------------------

def gen_isomap():
    X, _ = make_blobs(n_samples=80, n_features=5, centers=3, random_state=42, cluster_std=1.5)
    model = Isomap(n_components=2, n_neighbors=5)
    Xt = model.fit_transform(X)
    write_fixture(
        "isomap",
        {
            "description": "Isomap on 3-blob dataset",
            "random_state": 42,
            "input": {"X": _to_list(X)},
            "params": {"n_components": 2, "n_neighbors": 5},
            "expected": {"transformed": _to_list(Xt)},
        },
    )


def gen_mds():
    X, _ = make_blobs(n_samples=60, n_features=5, centers=3, random_state=42)
    model = MDS(n_components=2, random_state=42, n_init=4, max_iter=300, normalized_stress="auto")
    Xt = model.fit_transform(X)
    write_fixture(
        "mds",
        {
            "description": "MDS (multi-dimensional scaling) on 3-blob dataset",
            "random_state": 42,
            "input": {"X": _to_list(X)},
            "params": {"n_components": 2, "random_state": 42, "n_init": 4, "max_iter": 300},
            "expected": {
                "stress": float(model.stress_),
                "transformed": _to_list(Xt),
            },
        },
    )


def gen_lle():
    X, _ = make_blobs(n_samples=80, n_features=5, centers=3, random_state=42)
    model = LocallyLinearEmbedding(n_components=2, n_neighbors=10, random_state=42)
    Xt = model.fit_transform(X)
    write_fixture(
        "lle",
        {
            "description": "LocallyLinearEmbedding on 3-blob dataset",
            "random_state": 42,
            "input": {"X": _to_list(X)},
            "params": {"n_components": 2, "n_neighbors": 10, "random_state": 42},
            "expected": {
                "reconstruction_error": float(model.reconstruction_error_),
                "transformed": _to_list(Xt),
            },
        },
    )


def gen_spectral_embedding():
    X, _ = make_blobs(n_samples=60, n_features=5, centers=3, random_state=42)
    model = SpectralEmbedding(n_components=2, random_state=42, n_neighbors=5)
    Xt = model.fit_transform(X)
    write_fixture(
        "spectral_embedding",
        {
            "description": "SpectralEmbedding (Laplacian Eigenmaps)",
            "random_state": 42,
            "input": {"X": _to_list(X)},
            "params": {"n_components": 2, "random_state": 42, "n_neighbors": 5},
            "expected": {"transformed": _to_list(Xt)},
        },
    )


def gen_tsne():
    X, _ = make_blobs(n_samples=60, n_features=5, centers=3, random_state=42)
    model = TSNE(n_components=2, random_state=42, perplexity=10, max_iter=300, init="pca", method="barnes_hut")
    Xt = model.fit_transform(X)
    write_fixture(
        "tsne",
        {
            "description": "t-SNE on 3-blob dataset with perplexity=10",
            "random_state": 42,
            "input": {"X": _to_list(X)},
            "params": {"n_components": 2, "random_state": 42, "perplexity": 10, "max_iter": 300, "init": "pca"},
            "expected": {
                "kl_divergence": float(model.kl_divergence_),
                "transformed": _to_list(Xt),
            },
        },
    )


def main():
    print(f"Wave 2 — decomp gap fixtures with sklearn {SKLEARN_VERSION}")
    for fn in [
        gen_truncated_svd,
        gen_fast_ica,
        gen_kernel_pca,
        gen_factor_analysis,
        gen_incremental_pca,
        gen_sparse_pca,
        gen_dictionary_learning,
        gen_mini_batch_nmf,
        gen_lda_topic_model,
        gen_cca,
        gen_pls_regression,
        gen_pls_canonical,
        gen_isomap,
        gen_mds,
        gen_lle,
        gen_spectral_embedding,
        gen_tsne,
    ]:
        try:
            fn()
        except Exception as e:
            print(f"  ERROR in {fn.__name__}: {e}")
    print("Done.")


if __name__ == "__main__":
    main()
