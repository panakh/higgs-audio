# syntax=docker/dockerfile:1.7

FROM nvidia/cuda:12.4.1-cudnn9-runtime-ubuntu22.04

# Remove stale third-party APT sources that may cause build failures
RUN rm -f /etc/apt/sources.list.d/*.list

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1

WORKDIR /workspace

# Install system dependencies and Python
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        ffmpeg \
        git \
        libsndfile1 \
        python3 \
        python3-dev \
        python3-pip \
        python3-venv \
        python-is-python3 \
        wget \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip tooling
RUN python -m pip install --upgrade pip setuptools wheel

# Install Python dependencies
COPY requirements-runpod.txt ./requirements-runpod.txt
RUN python -m pip install --no-cache-dir -r requirements-runpod.txt

# Copy source and install the package
COPY . .
RUN python -m pip install --no-cache-dir -e .

CMD ["python3", "-u", "-m", "serverless.runpod_worker"]
