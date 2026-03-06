# ferrolearn-bench

Criterion benchmarks for the [ferrolearn](https://crates.io/crates/ferrolearn) machine learning framework. Not published to crates.io.

## Benchmark suites

| Suite | What it benchmarks |
|-------|-------------------|
| `regressors` | Linear regression, Ridge, Lasso, ElasticNet, etc. |
| `classifiers` | Logistic regression, decision trees, random forest, k-NN, Naive Bayes |
| `transformers` | StandardScaler, PCA |
| `clusterers` | KMeans, DBSCAN |
| `metrics` | Scoring functions (accuracy, MSE, silhouette, etc.) |

## Running benchmarks

```bash
# Run all benchmarks
cargo bench -p ferrolearn-bench

# Run a single suite
cargo bench -p ferrolearn-bench --bench regressors
```

Results are written to `target/criterion/` with HTML reports.

## License

Licensed under either of [Apache License, Version 2.0](../LICENSE-APACHE) or [MIT License](../LICENSE-MIT) at your option.
