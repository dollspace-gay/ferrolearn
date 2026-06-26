//! Image feature-extraction helpers.
//!
//! Scoped dense Rust analogues of scikit-learn's
//! `sklearn.feature_extraction.image.grid_to_graph` and `img_to_graph`.
//! The implementations preserve sklearn's row-major voxel numbering, diagonal
//! entries, undirected neighbor edges, and mask renumbering for dense arrays.
//!
//! ## REQ status
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 public `grid_to_graph` surface | SHIPPED scoped | crate-root re-export plus `tests/divergence_image_graph.rs` |
//! | REQ-2 public `img_to_graph` surface | SHIPPED scoped | crate-root re-export plus `tests/divergence_image_graph.rs` |
//! | REQ-3 dense graph values and mask renumbering | SHIPPED scoped | tests pin sklearn 1.5.2 dense oracles for unmasked and masked fixtures |
//! | REQ-4 sparse return classes, 3D `img_to_graph`, dtype/Python ABI | NOT-STARTED | only dense `Array2<F>` output; `img_to_graph` accepts 2D images; no scipy sparse/PyO3 protocol |

use ferrolearn_core::FerroError;
use ndarray::Array2;
use num_traits::Float;

fn invalid_param(name: &str, reason: impl Into<String>) -> FerroError {
    FerroError::InvalidParameter {
        name: name.into(),
        reason: reason.into(),
    }
}

fn validate_grid(n_x: usize, n_y: usize, n_z: usize) -> Result<usize, FerroError> {
    if n_x == 0 {
        return Err(invalid_param("n_x", "must be at least 1"));
    }
    if n_y == 0 {
        return Err(invalid_param("n_y", "must be at least 1"));
    }
    if n_z == 0 {
        return Err(invalid_param("n_z", "must be at least 1"));
    }
    n_x.checked_mul(n_y)
        .and_then(|xy| xy.checked_mul(n_z))
        .ok_or_else(|| invalid_param("shape", "n_x * n_y * n_z overflowed usize"))
}

fn active_map(
    n_voxels: usize,
    mask: Option<&[bool]>,
) -> Result<(Vec<Option<usize>>, usize), FerroError> {
    match mask {
        Some(mask) if mask.len() != n_voxels => {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_voxels],
                actual: vec![mask.len()],
                context: "grid mask".into(),
            });
        }
        _ => {}
    }

    let mut map = vec![None; n_voxels];
    let mut next = 0;
    for old in 0..n_voxels {
        if mask.is_none_or(|m| m[old]) {
            map[old] = Some(next);
            next += 1;
        }
    }
    Ok((map, next))
}

fn flat_index(x: usize, y: usize, z: usize, n_y: usize, n_z: usize) -> usize {
    (x * n_y + y) * n_z + z
}

fn add_undirected<F: Float>(graph: &mut Array2<F>, a: usize, b: usize, weight: F) {
    graph[[a, b]] = weight;
    graph[[b, a]] = weight;
}

/// Return a dense graph of pixel/voxel connectivity for a regular grid.
///
/// The output is the dense equivalent of sklearn's `grid_to_graph(...,
/// return_as=np.ndarray)`: diagonal entries are one, and undirected edges connect
/// immediate neighbors along the x/y/z axes. When `mask` is provided, it must be
/// flattened in sklearn row-major order and selected voxels are renumbered in
/// that order.
///
/// # Errors
///
/// Returns [`FerroError::InvalidParameter`] for zero dimensions or overflow, and
/// [`FerroError::ShapeMismatch`] when `mask.len() != n_x * n_y * n_z`.
pub fn grid_to_graph<F>(
    n_x: usize,
    n_y: usize,
    n_z: usize,
    mask: Option<&[bool]>,
) -> Result<Array2<F>, FerroError>
where
    F: Float + Send + Sync + 'static,
{
    let n_voxels = validate_grid(n_x, n_y, n_z)?;
    let (map, n_active) = active_map(n_voxels, mask)?;
    let mut graph = Array2::<F>::zeros((n_active, n_active));

    for &new_idx in map.iter().flatten() {
        graph[[new_idx, new_idx]] = F::one();
    }

    for x in 0..n_x {
        for y in 0..n_y {
            for z in 0..n_z {
                let old = flat_index(x, y, z, n_y, n_z);
                let Some(a) = map[old] else {
                    continue;
                };
                if z + 1 < n_z {
                    let nbr = flat_index(x, y, z + 1, n_y, n_z);
                    if let Some(b) = map[nbr] {
                        add_undirected(&mut graph, a, b, F::one());
                    }
                }
                if y + 1 < n_y {
                    let nbr = flat_index(x, y + 1, z, n_y, n_z);
                    if let Some(b) = map[nbr] {
                        add_undirected(&mut graph, a, b, F::one());
                    }
                }
                if x + 1 < n_x {
                    let nbr = flat_index(x + 1, y, z, n_y, n_z);
                    if let Some(b) = map[nbr] {
                        add_undirected(&mut graph, a, b, F::one());
                    }
                }
            }
        }
    }

    Ok(graph)
}

/// Return a dense graph of pixel-to-pixel gradient connections for a 2D image.
///
/// The output is the dense equivalent of sklearn's `img_to_graph(img,
/// return_as=np.ndarray)` for 2D images. Diagonal entries are the image values,
/// and neighbor edges are weighted by absolute pixel differences. When `mask` is
/// provided, selected pixels are renumbered in row-major order.
///
/// # Errors
///
/// Returns [`FerroError::ShapeMismatch`] when `mask.dim() != img.dim()`.
pub fn img_to_graph<F>(
    img: &Array2<F>,
    mask: Option<&Array2<bool>>,
) -> Result<Array2<F>, FerroError>
where
    F: Float + Send + Sync + 'static,
{
    match mask {
        Some(mask) if mask.dim() != img.dim() => {
            return Err(FerroError::ShapeMismatch {
                expected: vec![img.nrows(), img.ncols()],
                actual: vec![mask.nrows(), mask.ncols()],
                context: "image mask".into(),
            });
        }
        _ => {}
    }

    let n_x = img.nrows();
    let n_y = img.ncols();
    let n_voxels = validate_grid(n_x, n_y, 1)?;
    let flat_mask: Option<Vec<bool>> = mask.map(|m| m.iter().copied().collect());
    let (map, n_active) = active_map(n_voxels, flat_mask.as_deref())?;
    let mut graph = Array2::<F>::zeros((n_active, n_active));

    for x in 0..n_x {
        for y in 0..n_y {
            let old = flat_index(x, y, 0, n_y, 1);
            if let Some(new_idx) = map[old] {
                graph[[new_idx, new_idx]] = img[[x, y]];
            }
        }
    }

    for x in 0..n_x {
        for y in 0..n_y {
            let old = flat_index(x, y, 0, n_y, 1);
            let Some(a) = map[old] else {
                continue;
            };
            if y + 1 < n_y {
                let nbr = flat_index(x, y + 1, 0, n_y, 1);
                if let Some(b) = map[nbr] {
                    add_undirected(&mut graph, a, b, (img[[x, y]] - img[[x, y + 1]]).abs());
                }
            }
            if x + 1 < n_x {
                let nbr = flat_index(x + 1, y, 0, n_y, 1);
                if let Some(b) = map[nbr] {
                    add_undirected(&mut graph, a, b, (img[[x, y]] - img[[x + 1, y]]).abs());
                }
            }
        }
    }

    Ok(graph)
}
