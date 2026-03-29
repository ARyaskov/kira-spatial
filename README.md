# kira-spatial

Deterministic orchestrator crate for sequential pipeline:

1. `kira-spatial-io` (read H5)
2. `kira-spatial-field` (dense gene field)
3. `kira-spatial-core` (derived scalar: raw / `|grad|` / laplacian)
4. `kira-spatial-3d` (mesh, contours, stitching, metrics, exports)

## Run

```bash
cargo run --release -- run \
  --h5 ../Visium_HD_FF_Human_Breast_Cancer_feature_slice.h5 \
  --gene EPCAM \
  --signal grad-mag \
  --out-dir ./out \
  --contour-levels 0.2,0.4,0.6,0.8
```

Outputs are deterministic for identical inputs and options.

By default the command generates only per-gene images.

Add `--extended` to also generate mesh, contour, OBJ/PLY, and JSON exports:

```bash
cargo run --release -- run \
  --h5 ../Visium_HD_FF_Human_Breast_Cancer_feature_slice.h5 \
  --gene EPCAM \
  --signal grad-mag \
  --out-dir ./out \
  --extended
```
