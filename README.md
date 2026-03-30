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
  --genes EPCAM \
  --signal grad-mag \
  --out-dir ./out \
  --contour-levels 0.2,0.4,0.6,0.8
```

Outputs are deterministic for identical inputs and options.

By default the command generates only per-gene images.

On Windows, install the HDF5 command-line tools before running the pipeline. Download the CLI installer from the official HDF Group releases page: [https://github.com/HDFGroup/hdf5/releases](https://github.com/HDFGroup/hdf5/releases). For example, `hdf5-2.1.1-win-vs2022_cl.msi` provides `h5dump.exe` and the required runtime files.

Use `--genes` with a comma-separated list to aggregate multiple genes, for example `--genes EPCAM,KRT19,MSLN`.

Add `--extended` to also generate mesh, contour, OBJ/PLY, and JSON exports:

```bash
cargo run --release -- run \
  --h5 ../Visium_HD_FF_Human_Breast_Cancer_feature_slice.h5 \
  --genes EPCAM \
  --signal grad-mag \
  --out-dir ./out \
  --extended
```

## Initial Tumor Hypothesis Screening

For an initial hypothesis about tumor biology, start with the following aggregate gene set:

```text
EPCAM,KRT8,KRT19,CDH1,ESR1,PGR,GATA3,ERBB2,KRT5,KRT14,TP63,EGFR,VIM,FN1,COL1A1,ACTA2,PECAM1,CD3D,LYZ,SPP1
```

The commands below run the same gene set through the currently available signal operators and write each output into a separate directory.

### Raw

```powershell
kira-spatial.exe run --h5 .\feature_slice.h5 --genes EPCAM,KRT8,KRT19,CDH1,ESR1,PGR,GATA3,ERBB2,KRT5,KRT14,TP63,EGFR,VIM,FN1,COL1A1,ACTA2,PECAM1,CD3D,LYZ,SPP1 --signal raw --out-dir .\out\raw
```

### Gradient

```powershell
kira-spatial.exe run --h5 .\feature_slice.h5 --genes EPCAM,KRT8,KRT19,CDH1,ESR1,PGR,GATA3,ERBB2,KRT5,KRT14,TP63,EGFR,VIM,FN1,COL1A1,ACTA2,PECAM1,CD3D,LYZ,SPP1 --signal gradient --out-dir .\out\gradient
```

### GradMag

```powershell
kira-spatial.exe run --h5 .\feature_slice.h5 --genes EPCAM,KRT8,KRT19,CDH1,ESR1,PGR,GATA3,ERBB2,KRT5,KRT14,TP63,EGFR,VIM,FN1,COL1A1,ACTA2,PECAM1,CD3D,LYZ,SPP1 --signal grad-mag --out-dir .\out\gradmag
```

### Laplacian

```powershell
kira-spatial.exe run --h5 .\feature_slice.h5 --genes EPCAM,KRT8,KRT19,CDH1,ESR1,PGR,GATA3,ERBB2,KRT5,KRT14,TP63,EGFR,VIM,FN1,COL1A1,ACTA2,PECAM1,CD3D,LYZ,SPP1 --signal laplacian --out-dir .\out\laplacian
```

### HessianRidge

```powershell
kira-spatial.exe run --h5 .\feature_slice.h5 --genes EPCAM,KRT8,KRT19,CDH1,ESR1,PGR,GATA3,ERBB2,KRT5,KRT14,TP63,EGFR,VIM,FN1,COL1A1,ACTA2,PECAM1,CD3D,LYZ,SPP1 --signal hessian-ridge --out-dir .\out\hessian_ridge
```

### HessianValley

```powershell
kira-spatial.exe run --h5 .\feature_slice.h5 --genes EPCAM,KRT8,KRT19,CDH1,ESR1,PGR,GATA3,ERBB2,KRT5,KRT14,TP63,EGFR,VIM,FN1,COL1A1,ACTA2,PECAM1,CD3D,LYZ,SPP1 --signal hessian-valley --out-dir .\out\hessian_valley
```

### StructureTensor

```powershell
kira-spatial.exe run --h5 .\feature_slice.h5 --genes EPCAM,KRT8,KRT19,CDH1,ESR1,PGR,GATA3,ERBB2,KRT5,KRT14,TP63,EGFR,VIM,FN1,COL1A1,ACTA2,PECAM1,CD3D,LYZ,SPP1 --signal structure-tensor --out-dir .\out\structure_tensor
```

### Divergence

```powershell
kira-spatial.exe run --h5 .\feature_slice.h5 --genes EPCAM,KRT8,KRT19,CDH1,ESR1,PGR,GATA3,ERBB2,KRT5,KRT14,TP63,EGFR,VIM,FN1,COL1A1,ACTA2,PECAM1,CD3D,LYZ,SPP1 --signal divergence --out-dir .\out\divergence
```

### Curl

```powershell
kira-spatial.exe run --h5 .\feature_slice.h5 --genes EPCAM,KRT8,KRT19,CDH1,ESR1,PGR,GATA3,ERBB2,KRT5,KRT14,TP63,EGFR,VIM,FN1,COL1A1,ACTA2,PECAM1,CD3D,LYZ,SPP1 --signal curl --out-dir .\out\curl
```

### DistanceTransform

```powershell
kira-spatial.exe run --h5 .\feature_slice.h5 --genes EPCAM,KRT8,KRT19,CDH1,ESR1,PGR,GATA3,ERBB2,KRT5,KRT14,TP63,EGFR,VIM,FN1,COL1A1,ACTA2,PECAM1,CD3D,LYZ,SPP1 --signal distance-transform --out-dir .\out\distance_transform
```

### Curvature

```powershell
kira-spatial.exe run --h5 .\feature_slice.h5 --genes EPCAM,KRT8,KRT19,CDH1,ESR1,PGR,GATA3,ERBB2,KRT5,KRT14,TP63,EGFR,VIM,FN1,COL1A1,ACTA2,PECAM1,CD3D,LYZ,SPP1 --signal curvature --out-dir .\out\curvature
```

### FractalDimension

```powershell
kira-spatial.exe run --h5 .\feature_slice.h5 --genes EPCAM,KRT8,KRT19,CDH1,ESR1,PGR,GATA3,ERBB2,KRT5,KRT14,TP63,EGFR,VIM,FN1,COL1A1,ACTA2,PECAM1,CD3D,LYZ,SPP1 --signal fractal-dimension --out-dir .\out\fractal_dimension
```

### Skeletonization

```powershell
kira-spatial.exe run --h5 .\feature_slice.h5 --genes EPCAM,KRT8,KRT19,CDH1,ESR1,PGR,GATA3,ERBB2,KRT5,KRT14,TP63,EGFR,VIM,FN1,COL1A1,ACTA2,PECAM1,CD3D,LYZ,SPP1 --signal skeletonization --out-dir .\out\skeletonization
```

### Diffusion

```powershell
kira-spatial.exe run --h5 .\feature_slice.h5 --genes EPCAM,KRT8,KRT19,CDH1,ESR1,PGR,GATA3,ERBB2,KRT5,KRT14,TP63,EGFR,VIM,FN1,COL1A1,ACTA2,PECAM1,CD3D,LYZ,SPP1 --signal diffusion --out-dir .\out\diffusion
```

### MultiscaleLoG

```powershell
kira-spatial.exe run --h5 .\feature_slice.h5 --genes EPCAM,KRT8,KRT19,CDH1,ESR1,PGR,GATA3,ERBB2,KRT5,KRT14,TP63,EGFR,VIM,FN1,COL1A1,ACTA2,PECAM1,CD3D,LYZ,SPP1 --signal multiscale-log --out-dir .\out\multiscale_log
```

### CrossGradient

`cross-gradient` is a pairwise operator. In the current CLI implementation it uses the first two genes from `--genes`, so run it as a focused comparison between two specific markers rather than the full aggregate panel.
