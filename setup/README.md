# Setup (Cloud Linux)

This document explains how to set up a reproducible environment for this repository on a cloud Linux server.

Recommended approach:
- If Docker is available: use Docker (most reproducible).
- If Docker is not available: use Python `venv` + `pip` with pinned dependencies.

## 1. Get the code

```bash
git clone <YOUR_REPO_URL>
cd uaac_framework
```

## 2. Option A (recommended): Docker

### Build the image

```bash
docker build -t uaac_framework:py313 .
```

### Run a script (example: KMeans clustering)

Mount the repository directory into the container so outputs are written back to the host.

```bash
docker run --rm \
  -v "$PWD":/work \
  -w /work \
  uaac_framework:py313 \
  python clustering/kmeans.py 3 --input DJI_20260409131514_0001_D.tif
```

Expected output:
- `segmented_output_kmeans_k_3.tif`

Notes:
- The container sets `MPLBACKEND=Agg` so headless servers can run matplotlib without a GUI.

## 3. Option B: No Docker (venv + pip)

This repo uses geospatial Python packages (e.g., rasterio/fiona/geopandas) that may require OS-level libraries.

### 3.1 Install OS packages (Debian/Ubuntu)

```bash
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  build-essential pkg-config \
  gdal-bin libgdal-dev \
  proj-bin libproj-dev \
  libgeos-dev \
  ca-certificates
```

### 3.2 Create a virtual environment and install Python deps

Python 3.13 is recommended (the dependencies were pinned from a Python 3.13 environment).

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3.3 Run scripts on a headless server

Some scripts call `plt.show()`.
On servers without a display, set a non-interactive backend:

```bash
export MPLBACKEND=Agg
python clustering/kmeans.py 3 --input DJI_20260409131514_0001_D.tif
```

Expected output:
- `segmented_output_kmeans_k_3.tif`

## 4. Quick checks

### 4.1 Import check

```bash
python -c "import rasterio,fiona,geopandas,sklearn,skimage; print('imports: OK')"
```

### 4.2 KMeans example

```bash
export MPLBACKEND=Agg
python clustering/kmeans.py 6 --input DJI_20260409131514_0001_D.tif
```

Expected output:
- `segmented_output_kmeans_k_6.tif`

## Troubleshooting

- If `pip install -r requirements.txt` fails while building wheels (often `fiona`), ensure the OS packages above are installed.
- If your cloud environment is restricted (no Docker, no sudo), consider using micromamba/conda as an alternative distribution mechanism.
