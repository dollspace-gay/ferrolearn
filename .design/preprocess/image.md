# Image Feature Extraction Helpers

<!--
tier: 3-component
status: shipped-partial
baseline-commit: fb3447eb
upstream: scikit-learn 1.5.2
upstream-paths:
  - sklearn/feature_extraction/image.py  # __all__ (:23-26); _to_graph (:99-150); img_to_graph (:152-204); grid_to_graph (:206-258); extract_patches_2d (:352-468); reconstruct_from_patches_2d (:471-535); PatchExtractor (:538-693)
ferrolearn-module: ferrolearn-preprocess/src/image.rs
parity-ops: grid_to_graph, img_to_graph, extract_patches_2d, reconstruct_from_patches_2d, PatchExtractor
-->

## Summary

`ferrolearn_preprocess::grid_to_graph`, `img_to_graph`, `extract_patches_2d`,
`reconstruct_from_patches_2d`, and `PatchExtractor` are scoped dense Rust
analogues of sklearn's image feature extraction helpers.

The graph helpers preserve row-major voxel numbering, diagonal entries,
undirected neighbor edges, edge weights, and mask renumbering for dense arrays.
The patch helpers cover dense grayscale images/batches: all-patch extraction in
row-major order, reconstruction by overlap averaging, max-patch count/fraction
shape semantics, and stateless batch extraction.

This closes the exact public-name gap for all five sklearn
`feature_extraction.image.__all__` names. Residual parity gaps remain for scipy
sparse return classes, `return_as`, dtype-selection details, 3D/color image
helpers, exact NumPy RNG stream parity for sampled patches, PyO3/Python
validation ABI, and channel-preserving patch tensors.

## REQ Status

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 `grid_to_graph` public surface | SHIPPED scoped | `pub fn grid_to_graph` re-exported from crate root; `tests/divergence_image_graph.rs` pins dense unmasked and masked oracles. |
| REQ-2 `img_to_graph` public surface | SHIPPED scoped | `pub fn img_to_graph` re-exported from crate root; test pins sklearn's documented dense example and masked oracle. |
| REQ-3 mask renumbering | SHIPPED scoped | `mask` selected pixels/voxels are renumbered in row-major order, matching sklearn `_mask_edges_weights`; covered by `graph_mask_renumbering_matches_sklearn_oracles`. |
| REQ-4 `extract_patches_2d` public surface | SHIPPED scoped | `pub fn extract_patches_2d` re-exported from crate root; `tests/divergence_image_patches.rs` pins dense grayscale all-patch oracle and `max_patches` shape behavior. |
| REQ-5 `reconstruct_from_patches_2d` public surface | SHIPPED scoped | `pub fn reconstruct_from_patches_2d` re-exported from crate root; test pins sklearn complete-patch reconstruction. |
| REQ-6 `PatchExtractor` public surface | SHIPPED scoped | `pub struct PatchExtractor` re-exported from crate root; test pins grayscale batch patch ordering against sklearn. |
| REQ-7 full sklearn image feature extraction contracts | NOT-STARTED | scipy sparse outputs, `return_as`, full dtype handling, 3D/color image helpers, exact NumPy RNG sampled-patch values, channel-preserving patch tensors, and Python error ABI remain open. |

## Verification

```bash
cargo test -p ferrolearn-preprocess --test divergence_image_graph
cargo test -p ferrolearn-preprocess --test divergence_image_patches
cargo test -p ferrolearn-preprocess --test divergence_lib
cargo test -p ferrolearn-preprocess --test api_proof api_proof_feature_engineering
```
