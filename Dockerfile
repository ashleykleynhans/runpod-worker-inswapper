ARG CUDA_VERSION="12.4.1"
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu22.04

LABEL org.opencontainers.image.description="Runpod Serverless worker for face swapping using FaceFusion swapper models and insightface detection"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=on \
    SHELL=/bin/bash

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Upgrade apt packages and install required dependencies
RUN apt update && \
    apt upgrade -y && \
    apt install -y \
      python3-dev \
      python3-pip \
      python3.10-venv \
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
      procps && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean -y

# Set working directory
WORKDIR /workspace

# Install Torch
ARG INDEX_URL="https://download.pytorch.org/whl/cu124"
ARG TORCH_VERSION="2.6.0+cu124"
RUN pip3 install --no-cache-dir torch==${TORCH_VERSION} torchvision torchaudio --index-url ${INDEX_URL}

# Download models first (stable 5.4 GB layer, cached across code changes)
COPY scripts/download_models.py /tmp/download_models.py
RUN pip3 install --no-cache-dir tqdm requests && \
    python3 /tmp/download_models.py /workspace/runpod-worker-inswapper && \
    rm /tmp/download_models.py

# Clone repo and install Python dependencies
RUN git clone https://github.com/ashleykleynhans/runpod-worker-inswapper.git && \
    cd /workspace/runpod-worker-inswapper && \
    pip3 install -r requirements.txt && \
    pip3 uninstall -y onnxruntime && \
    pip3 install onnxruntime-gpu

# Install CodeFormer
RUN cd /workspace/runpod-worker-inswapper && \
    git lfs install && \
    git clone https://huggingface.co/spaces/sczhou/CodeFormer

# Download CodeFormer weights
RUN cd /workspace/runpod-worker-inswapper && \
    mkdir -p CodeFormer/CodeFormer/weights/CodeFormer && \
    wget -O CodeFormer/CodeFormer/weights/CodeFormer/codeformer.pth "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth" && \
    mkdir -p CodeFormer/CodeFormer/weights/facelib && \
    wget -O CodeFormer/CodeFormer/weights/facelib/detection_Resnet50_Final.pth "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/detection_Resnet50_Final.pth" && \
    wget -O CodeFormer/CodeFormer/weights/facelib/parsing_parsenet.pth "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/parsing_parsenet.pth" && \
    mkdir -p CodeFormer/CodeFormer/weights/realesrgan && \
    wget -O CodeFormer/CodeFormer/weights/realesrgan/RealESRGAN_x2plus.pth "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/RealESRGAN_x2plus.pth"

# Copy handler and new modules to ensure latest
COPY --chmod=755 handler.py /workspace/runpod-worker-inswapper/handler.py
COPY --chmod=755 face_swapper.py /workspace/runpod-worker-inswapper/face_swapper.py
COPY --chmod=755 face_swapper_models.py /workspace/runpod-worker-inswapper/face_swapper_models.py
COPY --chmod=755 restoration.py /workspace/runpod-worker-inswapper/restoration.py

# Docker container start script
COPY --chmod=755 start.sh /start.sh

# Start the container
ENTRYPOINT /start.sh
