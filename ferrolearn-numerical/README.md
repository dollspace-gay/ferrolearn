# ferrolearn-numerical

Numerical foundations for the [ferrolearn](https://crates.io/crates/ferrolearn) machine learning framework. Provides scipy-equivalent primitives that other ferrolearn crates depend on.

## Modules

| Module | Replaces | What it does |
|---|---|---|
| `sparse_eig` | `scipy.sparse.linalg.eigsh` | Lanczos eigensolver with implicit restart for large symmetric matrices |
| `sparse_graph` | `scipy.sparse.csgraph` | Dijkstra shortest paths, connected components, minimum spanning tree on CSR matrices |
| `distributions` | `scipy.stats` | Unified PDF/CDF/PPF for Normal, Chi-squared, F, t, Beta, Gamma, Dirichlet |
| `optimize` | `scipy.optimize` | Newton-CG and Trust-Region NCG (Steihaug-Toint) optimizers |
| `interpolate` | `scipy.interpolate` | Cubic spline interpolation with natural and not-a-knot boundary conditions |
| `integrate` | `scipy.integrate` | Adaptive Simpson quadrature and Gauss-Legendre (1-20 points) |

## Usage

```rust
use ferrolearn_numerical::sparse_eig::{LanczosSolver, WhichEigenvalues};
use ferrolearn_numerical::distributions::{Normal, ContinuousDistribution};
use ferrolearn_numerical::interpolate::{CubicSpline, BoundaryCondition};
use ferrolearn_numerical::integrate::quad;

// Sparse eigenvalues
let solver = LanczosSolver::new(5).with_which(WhichEigenvalues::SmallestAlgebraic);
let result = solver.solve_sparse(&sparse_matrix).unwrap();

// Statistical distributions
let n = Normal::new(0.0, 1.0).unwrap();
let p = n.cdf(1.96); // 0.975

// Cubic spline interpolation
let spline = CubicSpline::new(&x, &y, BoundaryCondition::NotAKnot).unwrap();
let y_interp = spline.eval(0.5);

// Numerical integration
let result = quad(|x| x.sin(), 0.0, std::f64::consts::PI, 1e-12);
assert!((result.value - 2.0).abs() < 1e-10);
```

## Performance vs scipy

| Module | Speedup | Accuracy |
|---|---|---|
| Sparse eigensolver | 1.6x -- 23.6x | Matches analytical eigenvalues to 1e-8 |
| Sparse graph | 1.7x -- 4.8x | Identical (deterministic) |
| Distributions | 1.4x -- 2.7x | Machine precision |
| Optimization | 36x -- 143x | Both converge correctly |
| Interpolation | 2x -- 6.2x (build) | Matches scipy CubicSpline |
| Quadrature | 2.6x -- 196x | 1e-8 to 1e-12 |

See [BENCHMARKS.md](BENCHMARKS.md) for full details.

## License

Licensed under either of [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0) or [MIT license](http://opensource.org/licenses/MIT) at your option.
