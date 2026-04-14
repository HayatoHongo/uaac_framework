# Docker environment (uaac_framework)

This repository contains research scripts that rely on geospatial Python packages (e.g., rasterio/fiona/geopandas). To make the runtime reproducible across machines, this project provides a Docker-based environment.

## What this builds

- Python: 3.13 (slim)
- Python packages: pinned via `requirements.txt`
- System libraries: common geospatial deps (GDAL/PROJ/GEOS)

## Build

From the repository root:

```bash
docker build -t uaac_framework:py313 .
```

## Run (examples)

### 1) Run kmeans clustering on a mounted TIFF

Assuming your input TIFF is on the host in the repo root:

```bash
docker run --rm \
  -v "$PWD":/work \
  -w /work \
  uaac_framework:py313 \
  python clustering/kmeans.py 3 --input DJI_20260409131514_0001_D.tif
```

This will create `segmented_output_kmeans_k_3.tif` in the same directory.

### 2) If you run on a headless environment

`matplotlib` uses a non-interactive backend by default in the image (`MPLBACKEND=Agg`). The script may still call `plt.show()`; in that case, it will emit a warning and continue.

## Troubleshooting

- If `docker build` fails while installing Python packages, it is usually due to wheel availability for your CPU architecture.
  - On Apple Silicon, consider:

```bash
docker buildx build --platform linux/amd64 -t uaac_framework:py313 .
```

- If your environment does not allow Docker (some cloud/HPC), consider using micromamba/conda as an alternative distribution mechanism.
