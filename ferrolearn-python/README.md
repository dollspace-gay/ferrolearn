# ferrolearn-python

Python bindings for the [ferrolearn](https://crates.io/crates/ferrolearn) machine learning framework, built with [PyO3](https://pyo3.rs). Provides a scikit-learn compatible API backed by Rust for performance.

## Available models

### Regressors

- `LinearRegression` — Ordinary Least Squares
- `Ridge` — L2-regularized regression
- `Lasso` — L1-regularized regression
- `ElasticNet` — Combined L1/L2 regularization

### Classifiers

- `LogisticRegression` — Binary and multiclass classification
- `DecisionTreeClassifier` — CART decision tree
- `RandomForestClassifier` — Ensemble of decision trees
- `KNeighborsClassifier` — k-nearest neighbors
- `GaussianNB` — Gaussian Naive Bayes

### Transformers

- `StandardScaler` — Zero-mean, unit-variance normalization
- `PCA` — Principal Component Analysis

### Clusterers

- `KMeans` — k-Means clustering

## Installation

```bash
pip install ferrolearn
```

## Example

```python
from ferrolearn import Ridge
import numpy as np

X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
y = np.array([1.0, 2.0, 3.0])

model = Ridge(alpha=1.0)
model.fit(X, y)
predictions = model.predict(X)
```

All models follow the familiar scikit-learn `fit`/`predict`/`transform` interface.

## License

Licensed under either of [Apache License, Version 2.0](../LICENSE-APACHE) or [MIT License](../LICENSE-MIT) at your option.
