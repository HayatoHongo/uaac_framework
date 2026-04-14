# Reproducible runtime for uaac_framework (no changes to research code)
# - Uses pip + requirements.txt (generated from the author's .venv)
# - Installs common geospatial system libs for rasterio/fiona/pyproj/shapely

FROM python:3.13.1-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg

# System deps:
# - build-essential/pkg-config: safety net if a wheel is unavailable
# - gdal/proj/geos: commonly required by rasterio/fiona/pyproj/shapely stack
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        pkg-config \
        gdal-bin \
        libgdal-dev \
        proj-bin \
        libproj-dev \
        libgeos-dev \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first for better layer caching
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install -r /app/requirements.txt

# Copy the repository code/data
COPY . /app

# Default command just prints help; run scripts explicitly via `docker run ... python ...`
CMD ["python", "-c", "print('Container ready. Example: python clustering/kmeans.py 3 --input DJI_*.tif')"]
