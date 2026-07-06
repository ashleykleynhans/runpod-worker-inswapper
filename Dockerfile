ARG CUDA_VERSION="12.4.1"
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu22.04

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

# Install Inswapper Serverless Worker
RUN git clone https://github.com/ashleykleynhans/runpod-worker-inswapper.git && \
    cd /workspace/runpod-worker-inswapper && \
    pip3 install -r requirements.txt && \
    pip3 uninstall -y onnxruntime && \
    pip3 install onnxruntime-gpu

# Download insightface checkpoints and face swapper models
RUN cd /workspace/runpod-worker-inswapper && \
    mkdir -p checkpoints/face_swapper && \
    mkdir -p checkpoints/models && \
    cd checkpoints && \
    # Existing inswapper_128 (keep for backward compat)
    wget -O inswapper_128.onnx "https://huggingface.co/ashleykleynhans/inswapper/resolve/main/inswapper_128.onnx?download=true" && \
    # NEW: Download all face swapper models
    cd face_swapper && \
    wget -O blendswap_256.onnx "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/blendswap_256.onnx" && \
    wget -O ghost_1_256.onnx "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/ghost_unet_1_256.onnx" && \
    wget -O ghost_2_256.onnx "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/ghost_unet_2_256.onnx" && \
    wget -O ghost_3_256.onnx "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/ghost_unet_3_256.onnx" && \
    wget -O hififace_unofficial_256.onnx "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/hififace_unofficial_256.onnx" && \
    wget -O hyperswap_1a_256.onnx "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/hyperswap_unet_1a_256.onnx" && \
    wget -O hyperswap_1b_256.onnx "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/hyperswap_unet_1b_256.onnx" && \
    wget -O hyperswap_1c_256.onnx "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/hyperswap_unet_1c_256.onnx" && \
    wget -O inswapper_128_fp16.onnx "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/inswapper_128_fp16.onnx" && \
    wget -O simswap_256.onnx "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/simswap_256.onnx" && \
    wget -O simswap_unofficial_512.onnx "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/simswap_unofficial_512.onnx" && \
    wget -O uniface_256.onnx "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/uniface_256.onnx" && \
    cd ../models && \
    # Existing buffalo_l download (unchanged)
    wget https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip && \
    mkdir buffalo_l && \
    cd buffalo_l && \
    unzip ../buffalo_l.zip

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
