# face_swapper.py
"""Enhanced face swapping with multi-model support and blending.

Forks insightface's INSwapper.get() preprocessing pipeline to add:
- Configurable face alignment resolution (via face_swapper_resolution param)
- Model-specific input normalization (mean/std per model)
- Tanh-output model post-processing
- Weight-based blending between original and swapped face
- Support for 3 model families: emap-projected, raw-embedding, source-face
"""

import os

import cv2
import numpy as np
import onnxruntime
from insightface.utils import face_align
from onnx import numpy_helper
from runpod.serverless.modules.rp_logger import RunPodLogger
from typing import Dict, Tuple

try:
    import onnx
    _ONNX_AVAILABLE = True
except ImportError:
    _ONNX_AVAILABLE = False

from face_swapper_models import get_model_metadata

logger = RunPodLogger()

# Global model cache for lazy loading
FACE_SWAPPER_MODELS: Dict[str, object] = {}


class _SwapperModel:
    """Wraps an ONNX face-swapper session with its embedding projection.

    Bypasses insightface's ModelRouter (which only routes 128x128 models
    to INSwapper).  Opens the session directly and extracts the emap if
    the model is an embedding_projected type.
    """

    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        try:
            self.session = onnxruntime.InferenceSession(model_path, None)
        except Exception as e:
            raise ValueError(
                f"Failed to create inference session for "
                f"'{model_path}': {e}"
            ) from e

        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()

        if len(inputs) < 2:
            raise ValueError(
                f"Model '{model_path}' has {len(inputs)} input(s), "
                f"expected at least 2 for a face swapper model"
            )

        self.input_names = [inp.name for inp in inputs]
        self.output_names = [out.name for out in outputs]

        if len(self.output_names) < 1:
            raise ValueError(f"Model '{model_path}' has no outputs")

        input_shape = inputs[0].shape
        self.input_size = tuple(input_shape[2:4][::-1])  # (w, h)

        # Extract emap only for embedding_projected models
        if _ONNX_AVAILABLE:
            try:
                onnx_model = onnx.load(model_path)
                graph = onnx_model.graph
                if graph.initializer:
                    raw = numpy_helper.to_array(graph.initializer[-1])
                    self.emap = raw if len(raw.shape) == 2 else np.eye(1)
                else:
                    self.emap = np.eye(1)
            except Exception:
                self.emap = np.eye(1)
        else:
            self.emap = np.eye(1)

        logger.info(
            f"Swapper loaded: {os.path.basename(model_path)} "
            f"size={self.input_size}"
        )


def get_face_swapper_model(model_name: str) -> _SwapperModel:
    """Load face swapper model on first use, cache for subsequent calls."""
    if model_name not in FACE_SWAPPER_MODELS:
        model_path = f"checkpoints/face_swapper/{model_name}.onnx"
        logger.info(f"Loading face swapper model: {model_name}")
        FACE_SWAPPER_MODELS[model_name] = _SwapperModel(model_path)

    return FACE_SWAPPER_MODELS[model_name]


# ---------------------------------------------------------------------------
# Source preparation helpers
# ---------------------------------------------------------------------------


def _prepare_source_projected(
    source_face, model: _SwapperModel
) -> np.ndarray:
    """Project embedding through emap, L2-normalize (inswapper family)."""
    latent = source_face.normed_embedding.reshape((1, -1))
    latent = np.dot(latent, model.emap)
    latent /= np.linalg.norm(latent)
    return latent


def _prepare_source_raw(source_face) -> np.ndarray:
    """Raw insightface embedding, L2-normalize (simswap, ghost family)."""
    latent = source_face.normed_embedding.reshape((1, -1))
    latent /= np.linalg.norm(latent)
    return latent


def _prepare_source_face(source_face, temp_frame, source_size: int):
    """Warp the source face to a template for image-input models.

    Used by blendswap_256 (112x112) and uniface_256 (256x256).
    """
    source_img, _ = face_align.norm_crop2(
        temp_frame, source_face.kps, source_size
    )
    return source_img


# ---------------------------------------------------------------------------
# Target image preprocessing
# ---------------------------------------------------------------------------


def _prepare_crop_frame(
    crop_frame: np.ndarray,
    mean: list,
    std: list,
) -> np.ndarray:
    """Normalize a cropped face frame for ONNX inference.

    BGR -> RGB, scale to [0,1], (x - mean) / std, HWC -> CHW, batch.
    """
    mean_np = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
    std_np = np.array(std, dtype=np.float32).reshape(1, 1, 3)

    crop_frame = crop_frame[:, :, ::-1].astype(np.float32) / 255.0
    crop_frame = (crop_frame - mean_np) / std_np
    crop_frame = crop_frame.transpose(2, 0, 1)
    return np.expand_dims(crop_frame, axis=0).astype(np.float32)


def _normalize_crop_frame(
    crop_frame: np.ndarray,
    mean: list,
    std: list,
    tanh_out: bool,
) -> np.ndarray:
    """Reverse _prepare_crop_frame: CHW->HWC, *std+mean if tanh, clip."""
    crop_frame = crop_frame[0].transpose(1, 2, 0)

    if tanh_out:
        mean_np = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        std_np = np.array(std, dtype=np.float32).reshape(1, 1, 3)
        crop_frame = crop_frame * std_np + mean_np

    crop_frame = np.clip(crop_frame, 0.0, 1.0)
    crop_frame = crop_frame[:, :, ::-1] * 255.0
    return crop_frame.astype(np.uint8)


# ---------------------------------------------------------------------------
# Paste-back (forked from insightface INSwapper.get)
# ---------------------------------------------------------------------------


def _paste_back(
    swapped_face: np.ndarray,
    aimg: np.ndarray,
    temp_frame: np.ndarray,
    affine_matrix: np.ndarray,
) -> np.ndarray:
    """Paste swapped face into the original frame with feathering."""
    fake_diff = np.abs(
        swapped_face.astype(np.float32) - aimg.astype(np.float32)
    ).mean(axis=2)
    fake_diff[:2, :] = 0
    fake_diff[-2:, :] = 0
    fake_diff[:, :2] = 0
    fake_diff[:, -2:] = 0

    IM = cv2.invertAffineTransform(affine_matrix)
    img_white = np.full(
        (aimg.shape[0], aimg.shape[1]), 255, dtype=np.float32
    )

    frame_h, frame_w = temp_frame.shape[1], temp_frame.shape[0]
    bgr_fake = cv2.warpAffine(
        swapped_face, IM, (frame_h, frame_w), borderValue=0.0
    )
    img_white = cv2.warpAffine(
        img_white, IM, (frame_h, frame_w), borderValue=0.0
    )
    fake_diff = cv2.warpAffine(
        fake_diff, IM, (frame_h, frame_w), borderValue=0.0
    )
    img_white[img_white > 20] = 255

    fthresh = 10
    fake_diff[fake_diff < fthresh] = 0
    fake_diff[fake_diff >= fthresh] = 255

    img_mask = img_white
    mask_h_inds, mask_w_inds = np.where(img_mask == 255)

    if len(mask_h_inds) == 0 or len(mask_w_inds) == 0:
        return temp_frame

    mask_h = np.max(mask_h_inds) - np.min(mask_h_inds)
    mask_w = np.max(mask_w_inds) - np.min(mask_w_inds)
    mask_size = int(np.sqrt(mask_h * mask_w))
    k = max(mask_size // 10, 10)
    kernel = np.ones((k, k), np.uint8)
    img_mask = cv2.erode(img_mask, kernel, iterations=1)
    kernel = np.ones((2, 2), np.uint8)
    fake_diff = cv2.dilate(fake_diff, kernel, iterations=1)
    k = max(mask_size // 20, 5)
    kernel_size = (k, k)
    blur_size = tuple(2 * i + 1 for i in kernel_size)
    img_mask = cv2.GaussianBlur(img_mask, blur_size, 0)
    k = 5
    kernel_size = (k, k)
    blur_size = tuple(2 * i + 1 for i in kernel_size)
    fake_diff = cv2.GaussianBlur(fake_diff, blur_size, 0)
    img_mask /= 255
    fake_diff /= 255
    img_mask = np.reshape(
        img_mask, [img_mask.shape[0], img_mask.shape[1], 1]
    )

    return (
        img_mask * bgr_fake.astype(np.float32)
        + (1 - img_mask) * temp_frame.astype(np.float32)
    ).astype(np.uint8)


# ---------------------------------------------------------------------------
# Main swap entry point
# ---------------------------------------------------------------------------


def swap_face_enhanced(
    source_face,
    target_face,
    temp_frame: np.ndarray,
    model: _SwapperModel,
    model_name: str,
    resolution: Tuple[int, int],
    weight: float = 1.0,
) -> np.ndarray:
    """Enhanced face swapping with configurable resolution and blending.

    Handles 3 model families:
    - embedding_projected  (inswapper) — emap dot product
    - embedding            (simswap, ghost) — raw insightface embedding
    - source_face          (blendswap, uniface) — warped source image
    """
    if not hasattr(model, "session"):
        raise TypeError(
            f"Expected face swapper model with .session, "
            f"got {type(model).__name__}."
        )

    try:
        metadata = get_model_metadata(model_name)
    except KeyError as e:
        raise ValueError(
            f"Unknown face swapper model: '{model_name}'"
        ) from e

    native_size = metadata["native_size"]
    mean = metadata["mean"]
    std = metadata["std"]
    tanh_out = metadata["tanh_out"]
    source_type = metadata["source_type"]

    target_resolution = resolution[0]

    # Step 1: Warp target face to user-requested resolution
    aimg, M = face_align.norm_crop2(
        temp_frame, target_face.kps, target_resolution
    )

    # Step 2: Resize to model native size & normalize
    aimg_resized = cv2.resize(aimg, native_size)
    blob = _prepare_crop_frame(aimg_resized, mean, std)

    # Step 3: Prepare source input based on model family
    if source_type == "embedding_projected":
        source_input = _prepare_source_projected(source_face, model)
    elif source_type == "embedding":
        source_input = _prepare_source_raw(source_face)
    elif source_type == "source_face":
        # source_face models: second input is a warped source image
        source_size = metadata["source_size"]
        source_img = _prepare_source_face(source_face, temp_frame, source_size)
        source_input = _prepare_crop_frame(source_img, mean, std)
    else:
        raise ValueError(f"Unknown source_type: '{source_type}'")

    # Step 4: ONNX inference
    try:
        pred = model.session.run(
            model.output_names,
            {model.input_names[0]: blob, model.input_names[1]: source_input},
        )[0]
    except Exception as e:
        raise RuntimeError(
            f"ONNX inference failed for model '{model_name}': {e}"
        ) from e

    # Step 5: Post-process
    bgr_fake_native = _normalize_crop_frame(pred, mean, std, tanh_out)

    # Step 6: Resize output back to user resolution
    bgr_fake = cv2.resize(
        bgr_fake_native, (target_resolution, target_resolution)
    )

    # Step 7: Paste back
    result = _paste_back(bgr_fake, aimg, temp_frame, M)

    # Step 8: Weight blending
    if weight < 1.0:
        result = cv2.addWeighted(
            temp_frame, 1.0 - weight, result, weight, 0
        )

    return result
