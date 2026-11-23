ARG CUDA_VERSION="12.4.1"
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu22.04

LABEL org.opencontainers.image.description="Runpod Serverless worker for face swapping using FaceFusion swapper models and insightface detection"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=on \
    SHELL=/bin/bash

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Upgrade apt packages, add the deadsnakes PPA, and install required dependencies.
# The Launchpad PPA intermittently returns 503s, so retry downloads aggressively
# and retry the install itself before giving up.
RUN apt-get update -o Acquire::Retries=10 && \
    apt-get upgrade -y -o Acquire::Retries=10 && \
    apt-get install -y -o Acquire::Retries=10 software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update -o Acquire::Retries=10 && \
    for attempt in 1 2 3 4 5; do \
      apt-get install -y -o Acquire::Retries=10 \
        python3.12 \
        python3.12-dev \
        python3.12-venv \
        fonts-dejavu-core \
        rsync \
        git \
        git-lfs \
        jq \
        moreutils \
        aria2 \
        wget \
        curl \
        libglib2.0-0 \
        libsm6 \
        libgl1 \
        libxrender1 \
        libxext6 \
        ffmpeg \
        unzip \
        libgoogle-perftools-dev \
        procps && break; \
      echo "apt-get install failed (attempt ${attempt}), retrying in 15s..." >&2; \
      sleep 15; \
    done && \
    python3.12 --version && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean -y

# Make Python 3.12 the default python3 and bootstrap pip for it
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 100 && \
    update-alternatives --set python3 /usr/bin/python3.12 && \
    rm -f /usr/lib/python3.12/EXTERNALLY-MANAGED && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3 -

# Set working directory
WORKDIR /workspace

# Install Torch
ARG INDEX_URL="https://download.pytorch.org/whl/cu124"
ARG TORCH_VERSION="2.6.0+cu124"
RUN pip3 install --no-cache-dir torch==${TORCH_VERSION} torchvision torchaudio --index-url ${INDEX_URL}

# Download models first (stable 5.4 GB layer, cached across code changes)
COPY scripts/download_models.py /tmp/download_models.py
RUN pip3 install --no-cache-dir tqdm requests && \
    python3 /tmp/download_models.py /workspace/models_cache && \
    rm /tmp/download_models.py

# Clone repo and install Python dependencies
RUN git clone https://github.com/ashleykleynhans/runpod-worker-inswapper.git && \
    cd /workspace/runpod-worker-inswapper && \
    pip3 install -r requirements.txt && \
    pip3 uninstall -y onnxruntime && \
    pip3 install onnxruntime-gpu==1.26.0 && \
    mv /workspace/models_cache/checkpoints /workspace/runpod-worker-inswapper/checkpoints

# Install CodeFormer
RUN cd /workspace/runpod-worker-inswapper && \
    git lfs install && \
    git clone https://huggingface.co/spaces/sczhou/CodeFormer && \
    rsync -a /workspace/models_cache/CodeFormer/ /workspace/runpod-worker-inswapper/CodeFormer/

# Clean up model cache
RUN rm -rf /workspace/models_cache

# Copy handler and new modules to ensure latest
COPY --chmod=755 handler.py /workspace/runpod-worker-inswapper/handler.py
COPY --chmod=755 face_swapper.py /workspace/runpod-worker-inswapper/face_swapper.py
COPY --chmod=755 face_swapper_models.py /workspace/runpod-worker-inswapper/face_swapper_models.py
COPY --chmod=755 restoration.py /workspace/runpod-worker-inswapper/restoration.py

# Docker container start script
COPY --chmod=755 start.sh /start.sh

# Start the container
ENTRYPOINT /start.sh
