//! Image feature-extraction helpers.
//!
//! Scoped dense Rust analogues of scikit-learn's
//! `sklearn.feature_extraction.image` helpers. The graph helpers preserve
//! sklearn's row-major voxel numbering, diagonal entries, undirected neighbor
//! edges, and mask renumbering for dense arrays. The patch helpers cover dense
//! grayscale images and batches.
//!
//! ## REQ status
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 public `grid_to_graph` surface | SHIPPED scoped | crate-root re-export plus `tests/divergence_image_graph.rs` |
//! | REQ-2 public `img_to_graph` surface | SHIPPED scoped | crate-root re-export plus `tests/divergence_image_graph.rs` |
//! | REQ-3 dense graph values and mask renumbering | SHIPPED scoped | tests pin sklearn 1.5.2 dense oracles for unmasked and masked fixtures |
//! | REQ-4 public patch surface | SHIPPED scoped | [`extract_patches_2d`], [`reconstruct_from_patches_2d`], and [`PatchExtractor`] re-exported from crate root; grayscale dense oracles in `tests/divergence_image_patches.rs` |
//! | REQ-5 sparse graph return classes, 3D/color image support, dtype/Python ABI | NOT-STARTED | graph helpers return dense `Array2<F>`; `img_to_graph` accepts 2D images; patch helpers are grayscale-only; no scipy sparse/PyO3 protocol |

use ferrolearn_core::FerroError;
use ferrolearn_core::traits::{Fit, FitTransform, Transform};
use ndarray::{Array2, Array3, s};
use num_traits::Float;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

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

/// Maximum number of patches to extract.
///
/// Mirrors sklearn's `max_patches` parameter for the scoped Rust API:
/// `Count(k)` corresponds to an integer and is capped at the total number of
/// available patches, while `Fraction(f)` corresponds to a float in `(0, 1)` and
/// extracts `floor(f * all_patches)` patches.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MaxPatches {
    /// Extract at most this many sampled patches.
    Count(usize),
    /// Extract this fraction of all possible patches.
    Fraction(f64),
}

fn validate_patch_size(
    image_height: usize,
    image_width: usize,
    patch_size: (usize, usize),
) -> Result<(usize, usize, usize, usize), FerroError> {
    let (patch_height, patch_width) = patch_size;
    if patch_height == 0 {
        return Err(invalid_param(
            "patch_size",
            "patch height must be at least 1",
        ));
    }
    if patch_width == 0 {
        return Err(invalid_param(
            "patch_size",
            "patch width must be at least 1",
        ));
    }
    if patch_height > image_height {
        return Err(invalid_param(
            "patch_size",
            "patch height must not exceed image height",
        ));
    }
    if patch_width > image_width {
        return Err(invalid_param(
            "patch_size",
            "patch width must not exceed image width",
        ));
    }
    let n_h = image_height - patch_height + 1;
    let n_w = image_width - patch_width + 1;
    Ok((patch_height, patch_width, n_h, n_w))
}

fn compute_n_patches(
    n_h: usize,
    n_w: usize,
    max_patches: Option<MaxPatches>,
) -> Result<usize, FerroError> {
    let all_patches = n_h
        .checked_mul(n_w)
        .ok_or_else(|| invalid_param("patch_size", "number of patches overflowed usize"))?;
    match max_patches {
        None => Ok(all_patches),
        Some(MaxPatches::Count(count)) => {
            if count == 0 {
                return Err(invalid_param("max_patches", "count must be at least 1"));
            }
            Ok(count.min(all_patches))
        }
        Some(MaxPatches::Fraction(frac)) => {
            if !(frac > 0.0 && frac < 1.0 && frac.is_finite()) {
                return Err(invalid_param(
                    "max_patches",
                    "fraction must be finite and in the open interval (0, 1)",
                ));
            }
            Ok((frac * all_patches as f64) as usize)
        }
    }
}

fn make_rng(random_state: Option<u64>) -> SmallRng {
    match random_state {
        Some(seed) => SmallRng::seed_from_u64(seed),
        None => SmallRng::from_os_rng(),
    }
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

/// Reshape a dense grayscale 2D image into a collection of patches.
///
/// This is a scoped Rust analogue of sklearn's `extract_patches_2d` for
/// grayscale images. With `max_patches == None`, all patches are emitted in
/// sklearn row-major order. With `max_patches`, positions are sampled with
/// replacement from the valid row and column ranges, matching sklearn's sampling
/// structure but not its exact NumPy RNG stream.
///
/// # Errors
///
/// Returns [`FerroError::InvalidParameter`] for zero or oversized patch
/// dimensions, invalid `max_patches`, or patch-count overflow.
pub fn extract_patches_2d<F>(
    image: &Array2<F>,
    patch_size: (usize, usize),
    max_patches: Option<MaxPatches>,
    random_state: Option<u64>,
) -> Result<Array3<F>, FerroError>
where
    F: Float + Send + Sync + 'static,
{
    let (patch_height, patch_width, n_h, n_w) =
        validate_patch_size(image.nrows(), image.ncols(), patch_size)?;
    let n_patches = compute_n_patches(n_h, n_w, max_patches)?;
    let mut patches = Array3::<F>::zeros((n_patches, patch_height, patch_width));

    match max_patches {
        Some(_) => {
            let mut rng = make_rng(random_state);
            for patch_idx in 0..n_patches {
                let i = rng.random_range(0..n_h);
                let j = rng.random_range(0..n_w);
                for pi in 0..patch_height {
                    for pj in 0..patch_width {
                        patches[[patch_idx, pi, pj]] = image[[i + pi, j + pj]];
                    }
                }
            }
        }
        None => {
            let mut patch_idx = 0;
            for i in 0..n_h {
                for j in 0..n_w {
                    for pi in 0..patch_height {
                        for pj in 0..patch_width {
                            patches[[patch_idx, pi, pj]] = image[[i + pi, j + pj]];
                        }
                    }
                    patch_idx += 1;
                }
            }
        }
    }

    Ok(patches)
}

/// Reconstruct a grayscale image from its dense 2D patches.
///
/// Mirrors sklearn's `reconstruct_from_patches_2d`: patches are applied
/// left-to-right, top-to-bottom and overlapping pixels are averaged by the full
/// overlap count implied by `image_size`. If more patches than positions are
/// supplied, extras are ignored; if fewer are supplied, the untouched
/// contributions remain zero before the same overlap normalization.
///
/// # Errors
///
/// Returns [`FerroError::InvalidParameter`] when the image or patch dimensions
/// are zero or when a patch is larger than the target image.
pub fn reconstruct_from_patches_2d<F>(
    patches: &Array3<F>,
    image_size: (usize, usize),
) -> Result<Array2<F>, FerroError>
where
    F: Float + Send + Sync + 'static,
{
    let (image_height, image_width) = image_size;
    if image_height == 0 {
        return Err(invalid_param(
            "image_size",
            "image height must be at least 1",
        ));
    }
    if image_width == 0 {
        return Err(invalid_param(
            "image_size",
            "image width must be at least 1",
        ));
    }
    let (_, patch_height, patch_width) = patches.dim();
    let (_, _, n_h, n_w) =
        validate_patch_size(image_height, image_width, (patch_height, patch_width))?;
    let mut image = Array2::<F>::zeros((image_height, image_width));

    let expected_positions = n_h
        .checked_mul(n_w)
        .ok_or_else(|| invalid_param("image_size", "number of patch positions overflowed usize"))?;
    let limit = patches.len_of(ndarray::Axis(0)).min(expected_positions);
    for patch_idx in 0..limit {
        let i = patch_idx / n_w;
        let j = patch_idx % n_w;
        let patch = patches.slice(s![patch_idx, .., ..]);
        for pi in 0..patch_height {
            for pj in 0..patch_width {
                image[[i + pi, j + pj]] = image[[i + pi, j + pj]] + patch[[pi, pj]];
            }
        }
    }

    for i in 0..image_height {
        for j in 0..image_width {
            let row_overlap = (i + 1).min(patch_height).min(n_h).min(image_height - i);
            let col_overlap = (j + 1).min(patch_width).min(n_w).min(image_width - j);
            let denom = F::from(row_overlap * col_overlap).unwrap_or_else(F::one);
            image[[i, j]] = image[[i, j]] / denom;
        }
    }

    Ok(image)
}

/// Stateless grayscale image patch extractor.
///
/// This is a scoped analogue of sklearn's `PatchExtractor` for batches shaped
/// `(n_images, image_height, image_width)`. The fitted state is the extractor
/// itself because sklearn's estimator is stateless and `fit` only validates
/// parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct PatchExtractor<F> {
    patch_size: Option<(usize, usize)>,
    max_patches: Option<MaxPatches>,
    random_state: Option<u64>,
    _marker: std::marker::PhantomData<F>,
}

impl<F> Default for PatchExtractor<F> {
    fn default() -> Self {
        Self {
            patch_size: None,
            max_patches: None,
            random_state: None,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<F: Float + Send + Sync + 'static> PatchExtractor<F> {
    /// Create a new stateless patch extractor.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set an explicit `(patch_height, patch_width)`.
    #[must_use]
    pub fn patch_size(mut self, patch_size: (usize, usize)) -> Self {
        self.patch_size = Some(patch_size);
        self
    }

    /// Set a maximum patch policy.
    #[must_use]
    pub fn max_patches(mut self, max_patches: MaxPatches) -> Self {
        self.max_patches = Some(max_patches);
        self
    }

    /// Set a deterministic Rust RNG seed for sampled patch positions.
    #[must_use]
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }
}

impl<F: Float + Send + Sync + 'static> Transform<Array3<F>> for PatchExtractor<F> {
    type Output = Array3<F>;
    type Error = FerroError;

    fn transform(&self, x: &Array3<F>) -> Result<Self::Output, Self::Error> {
        let (n_images, image_height, image_width) = x.dim();
        if n_images == 0 {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "PatchExtractor requires at least one image".into(),
            });
        }
        if image_height == 0 {
            return Err(invalid_param("X", "image height must be at least 1"));
        }
        if image_width == 0 {
            return Err(invalid_param("X", "image width must be at least 1"));
        }

        let patch_size = match self.patch_size {
            Some(patch_size) => patch_size,
            None => (image_height / 10, image_width / 10),
        };
        let (patch_height, patch_width, n_h, n_w) =
            validate_patch_size(image_height, image_width, patch_size)?;
        let n_patches_per_image = compute_n_patches(n_h, n_w, self.max_patches)?;
        let total_patches = n_images.checked_mul(n_patches_per_image).ok_or_else(|| {
            invalid_param("X", "total number of extracted patches overflowed usize")
        })?;
        let mut out = Array3::<F>::zeros((total_patches, patch_height, patch_width));

        let mut rng = self.max_patches.map(|_| make_rng(self.random_state));
        let mut out_idx = 0;
        for image_idx in 0..n_images {
            let image = x.slice(s![image_idx, .., ..]);
            match rng.as_mut() {
                Some(rng) => {
                    for _ in 0..n_patches_per_image {
                        let i = rng.random_range(0..n_h);
                        let j = rng.random_range(0..n_w);
                        for pi in 0..patch_height {
                            for pj in 0..patch_width {
                                out[[out_idx, pi, pj]] = image[[i + pi, j + pj]];
                            }
                        }
                        out_idx += 1;
                    }
                }
                None => {
                    for i in 0..n_h {
                        for j in 0..n_w {
                            for pi in 0..patch_height {
                                for pj in 0..patch_width {
                                    out[[out_idx, pi, pj]] = image[[i + pi, j + pj]];
                                }
                            }
                            out_idx += 1;
                        }
                    }
                }
            }
        }

        Ok(out)
    }
}

impl<F: Float + Send + Sync + 'static> Fit<Array3<F>, ()> for PatchExtractor<F> {
    type Fitted = Self;
    type Error = FerroError;

    fn fit(&self, x: &Array3<F>, _y: &()) -> Result<Self::Fitted, Self::Error> {
        let (n_images, image_height, image_width) = x.dim();
        if n_images == 0 {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "PatchExtractor requires at least one image".into(),
            });
        }
        let patch_size = self
            .patch_size
            .unwrap_or((image_height / 10, image_width / 10));
        let (_, _, n_h, n_w) = validate_patch_size(image_height, image_width, patch_size)?;
        compute_n_patches(n_h, n_w, self.max_patches)?;
        Ok(self.clone())
    }
}

impl<F: Float + Send + Sync + 'static> FitTransform<Array3<F>> for PatchExtractor<F> {
    type FitError = FerroError;

    fn fit_transform(&self, x: &Array3<F>) -> Result<Self::Output, Self::FitError> {
        self.fit(x, &())?.transform(x)
    }
}
