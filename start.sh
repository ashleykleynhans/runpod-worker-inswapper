#!/usr/bin/env bash

echo "Worker Initiated"

if [ -d "/runpod-volume" ]; then
    echo "Symlinking files from Network Volume"
    ln -s /runpod-volume /workspace
    rm -rf /root/.cache
    rm -rf /root/.ifnude
    rm -rf /root/.insightface
    ln -s /runpod-volume/.cache /root/.cache
    ln -s /runpod-volume/.ifnude /root/.ifnude
    ln -s /runpod-volume/.insightface /root/.insightface
else
    echo "No Network Volume found, skipping symlinks"
fi

# Fallback: ensure insightface models are available locally to prevent
# auto-download hanging in environments without network volume cache.
if [ ! -e "/root/.insightface/models/buffalo_l" ]; then
    echo "Caching insightface models from local checkpoints"
    mkdir -p /root/.insightface/models
    ln -s /workspace/runpod-worker-inswapper/checkpoints/models/buffalo_l \
        /root/.insightface/models/buffalo_l
fi

echo "Starting RunPod Handler"
export PYTHONUNBUFFERED=1
cd /workspace/runpod-worker-inswapper
python3 -u handler.py
