#!/usr/bin/env python3
"""Download all face swapper models and face detection models locally.

Mirrors the model download steps from the Dockerfile so models can be
tested without a full Docker build. Uses tqdm for progress bars (same
library FaceFusion uses).
"""

import os
import zipfile
from pathlib import Path
from urllib.parse import urlparse

import requests
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINTS = ROOT / "checkpoints"
FACE_SWAPPER = CHECKPOINTS / "face_swapper"
MODELS = CHECKPOINTS / "models"

MODELS_3_0_0 = (
    "https://github.com/facefusion/facefusion-assets"
    "/releases/download/models-3.0.0"
)

DOWNLOADS = [
    # (subdirectory, filename, url)
    (FACE_SWAPPER, "inswapper_128.onnx",
     "https://huggingface.co/ashleykleynhans/inswapper/resolve/main/inswapper_128.onnx?download=true"),
    (FACE_SWAPPER, "blendswap_256.onnx", f"{MODELS_3_0_0}/blendswap_256.onnx"),
    (FACE_SWAPPER, "ghost_1_256.onnx", f"{MODELS_3_0_0}/ghost_1_256.onnx"),
    (FACE_SWAPPER, "ghost_2_256.onnx", f"{MODELS_3_0_0}/ghost_2_256.onnx"),
    (FACE_SWAPPER, "ghost_3_256.onnx", f"{MODELS_3_0_0}/ghost_3_256.onnx"),
    (FACE_SWAPPER, "inswapper_128_fp16.onnx", f"{MODELS_3_0_0}/inswapper_128_fp16.onnx"),
    (FACE_SWAPPER, "simswap_256.onnx", f"{MODELS_3_0_0}/simswap_256.onnx"),
    (FACE_SWAPPER, "simswap_unofficial_512.onnx", f"{MODELS_3_0_0}/simswap_unofficial_512.onnx"),
    (FACE_SWAPPER, "uniface_256.onnx", f"{MODELS_3_0_0}/uniface_256.onnx"),
    (
        MODELS,
        "buffalo_l.zip",
        "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
    ),
]


def _download(dest_dir: Path, filename: str, url: str) -> None:
    """Download a single file with tqdm progress bar."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / filename

    with requests.head(url, allow_redirects=True, timeout=30) as head:
        head.raise_for_status()
        expected_size = int(head.headers.get("content-length", 0))

    if dest_path.exists() and expected_size and dest_path.stat().st_size == expected_size:
        tqdm.write(f"✓ {filename}")
        return

    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    with (
        open(dest_path, "wb") as f,
        tqdm(
            total=expected_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=filename,
            ascii=" =",
        ) as pbar,
    ):
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))


def _extract_zip(zip_path: Path) -> None:
    """Extract a zip file into a sibling directory of the same name."""
    extract_dir = zip_path.with_suffix("")
    if extract_dir.exists() and any(extract_dir.iterdir()):
        tqdm.write("✓ buffalo_l already extracted")
        return

    extract_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {zip_path.name} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    print("done")


def main() -> None:
    print(f"Downloading models to {CHECKPOINTS}\n")

    for dest_dir, filename, url in DOWNLOADS:
        try:
            _download(dest_dir, filename, url)
        except Exception as e:
            tqdm.write(f"✗ {filename}: {e}")

    zip_path = MODELS / "buffalo_l.zip"
    if zip_path.exists():
        _extract_zip(zip_path)

    for dest_dir, filename, _ in DOWNLOADS:
        dest = dest_dir / filename
        if dest.exists():
            total += dest.stat().st_size
    print(f"\nDone. Total: {total / (1024**3):.1f} GB")


if __name__ == "__main__":
    main()
