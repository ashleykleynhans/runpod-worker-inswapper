#!/usr/bin/env python3
"""Download all face swapper models and face detection models.

Mirrors the model download steps from the Dockerfile so models can be
tested locally without a full Docker build.
"""

import os
import sys
import zipfile
from pathlib import Path
from urllib.parse import urlparse

import requests
from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

console = Console()

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINTS = ROOT / "checkpoints"
FACE_SWAPPER = CHECKPOINTS / "face_swapper"
MODELS = CHECKPOINTS / "models"

MODELS_3_0_0 = "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0"

DOWNLOADS = [
    # (subdirectory, filename, url)
    (CHECKPOINTS, "inswapper_128.onnx",
     "https://huggingface.co/ashleykleynhans/inswapper/resolve/main/inswapper_128.onnx?download=true"),
    (FACE_SWAPPER, "blendswap_256.onnx",
     f"{MODELS_3_0_0}/blendswap_256.onnx"),
    (FACE_SWAPPER, "ghost_1_256.onnx",
     f"{MODELS_3_0_0}/ghost_1_256.onnx"),
    (FACE_SWAPPER, "ghost_2_256.onnx",
     f"{MODELS_3_0_0}/ghost_2_256.onnx"),
    (FACE_SWAPPER, "ghost_3_256.onnx",
     f"{MODELS_3_0_0}/ghost_3_256.onnx"),
    (FACE_SWAPPER, "inswapper_128_fp16.onnx",
     f"{MODELS_3_0_0}/inswapper_128_fp16.onnx"),
    (FACE_SWAPPER, "simswap_256.onnx",
     f"{MODELS_3_0_0}/simswap_256.onnx"),
    (FACE_SWAPPER, "simswap_unofficial_512.onnx",
     f"{MODELS_3_0_0}/simswap_unofficial_512.onnx"),
    (FACE_SWAPPER, "uniface_256.onnx",
     f"{MODELS_3_0_0}/uniface_256.onnx"),
    (MODELS, "buffalo_l.zip",
     "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"),
]


def human_size(size_bytes: int) -> str:
    """Format byte count to human-readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def download_file(dest_dir: Path, filename: str, url: str,
                  progress: Progress, task_id: int) -> None:
    """Download a single file with progress tracking."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / filename

    # Skip if already fully downloaded (if we can determine size)
    with requests.head(url, allow_redirects=True, timeout=30) as head:
        expected_size = int(head.headers.get("content-length", 0))

    if dest_path.exists() and expected_size > 0 and dest_path.stat().st_size == expected_size:
        progress.update(task_id, completed=expected_size, total=expected_size,
                        description=f"[green]✓ {filename}")
        return

    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    progress.update(task_id, total=expected_size if expected_size else None)

    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            progress.update(task_id, advance=len(chunk))

    progress.update(task_id, description=f"[green]✓ {filename}")


def extract_buffalo_l(zip_path: Path) -> None:
    """Extract buffalo_l.zip into the buffalo_l subdirectory."""
    extract_dir = MODELS / "buffalo_l"
    if extract_dir.exists() and any(extract_dir.iterdir()):
        console.print("[green]✓ buffalo_l already extracted")
        return

    extract_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"Extracting buffalo_l.zip ...", end=" ")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    console.print("done")


def main() -> None:
    console.print("[bold]Downloading face swapper models[/bold]")
    console.print(f"Destination: {CHECKPOINTS}\n")

    # Count total size
    total_size = 0
    for _, filename, _ in DOWNLOADS:
        dest = (
            CHECKPOINTS / filename if filename == "inswapper_128.onnx"
            else MODELS / filename if filename == "buffalo_l.zip"
            else FACE_SWAPPER / filename
        )
        if dest.exists():
            total_size += dest.stat().st_size

    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        for dest_dir, filename, url in DOWNLOADS:
            task_id = progress.add_task(f"↓ {filename}", start=False)
            try:
                download_file(dest_dir, filename, url, progress, task_id)
            except Exception as e:
                progress.update(task_id, description=f"[red]✗ {filename}: {e}")

    # Extract buffalo_l
    zip_path = MODELS / "buffalo_l.zip"
    if zip_path.exists():
        extract_buffalo_l(zip_path)

    console.print()
    total_downloaded = sum(
        (d / f).stat().st_size
        for d, f, _ in DOWNLOADS
        if (d / f).exists()
    )
    console.print(f"[bold green]Done. Total: {human_size(total_downloaded)}[/bold green]")


if __name__ == "__main__":
    main()
