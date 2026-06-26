# Image Graph Helpers

<!--
tier: 3-component
status: shipped-partial
baseline-commit: fb3447eb
upstream: scikit-learn 1.5.2
upstream-paths:
  - sklearn/feature_extraction/image.py  # __all__ (:23-26); _to_graph (:99-150); img_to_graph (:152-204); grid_to_graph (:206-258)
ferrolearn-module: ferrolearn-preprocess/src/image.rs
parity-ops: grid_to_graph, img_to_graph
-->

## Summary

`ferrolearn_preprocess::grid_to_graph` and `img_to_graph` are scoped dense Rust
analogues of sklearn's image graph helpers. They preserve row-major voxel
numbering, diagonal entries, undirected neighbor edges, edge weights, and mask
renumbering for dense arrays.

This closes the exact public-name gap for `grid_to_graph` and `img_to_graph`.
Residual parity gaps remain for scipy sparse return classes, `return_as`,
dtype-selection details, 3D `img_to_graph`, PyO3/Python validation ABI, and the
remaining image patch APIs.

## REQ Status

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 `grid_to_graph` public surface | SHIPPED scoped | `pub fn grid_to_graph` re-exported from crate root; `tests/divergence_image_graph.rs` pins dense unmasked and masked oracles. |
| REQ-2 `img_to_graph` public surface | SHIPPED scoped | `pub fn img_to_graph` re-exported from crate root; test pins sklearn's documented dense example and masked oracle. |
| REQ-3 mask renumbering | SHIPPED scoped | `mask` selected pixels/voxels are renumbered in row-major order, matching sklearn `_mask_edges_weights`; covered by `graph_mask_renumbering_matches_sklearn_oracles`. |
| REQ-4 full sklearn image feature extraction | NOT-STARTED | `PatchExtractor`, `extract_patches_2d`, `reconstruct_from_patches_2d`, scipy sparse outputs, `return_as`, full dtype handling, 3D `img_to_graph`, and Python error ABI remain open. |

## Verification

```bash
cargo test -p ferrolearn-preprocess --test divergence_image_graph
cargo test -p ferrolearn-preprocess --test divergence_lib
cargo test -p ferrolearn-preprocess --test api_proof api_proof_feature_engineering
```
