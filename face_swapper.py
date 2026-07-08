# face_swapper.py
"""Enhanced face swapping with multi-model support and blending.

Forks FaceFusion's pipeline (not insightface's) for preprocessing:
- Configurable face alignment resolution (via face_swapper_resolution)
- Model-specific input normalization (mean/std)
- Tanh-output model post-processing
- Weight-based blending (balance_source_embedding)
- Support for embedding_projected, embedding, and source_face model families
"""

import os

import cv2
import numpy as np
import onnx
import onnxruntime
from insightface.utils import face_align
from onnx import numpy_helper
from runpod.serverless.modules.rp_logger import RunPodLogger
from typing import Dict, Tuple

from face_swapper_models import get_model_metadata

logger = RunPodLogger()

# Global caches
FACE_SWAPPER_MODELS: Dict[str, object] = {}
EMBEDDING_CONVERTERS: Dict[str, object] = {}


class _SwapperModel:
    """Wraps an ONNX face-swapper session with its embedding projection."""

    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.session = onnxruntime.InferenceSession(model_path, None)
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()

        if len(inputs) < 2:
            raise ValueError(
                f"Model '{model_path}' has {len(inputs)} input(s), "
                f"expected at least 2"
            )
        self.input_names = [inp.name for inp in inputs]
        self.output_names = [out.name for out in outputs]
        if not self.output_names:
            raise ValueError(f"Model '{model_path}' has no outputs")

        input_shape = inputs[0].shape
        self.input_size = tuple(input_shape[2:4][::-1])

        # Extract emap for embedding_projected models (inswapper family)
        try:
            onnx_model = onnx.load(model_path)
            graph = onnx_model.graph
            self.emap = (
                numpy_helper.to_array(graph.initializer[-1])
                if graph.initializer else np.eye(1)
            )
        except Exception:
            self.emap = np.eye(1)

        logger.info(
            f"Swapper loaded: {os.path.basename(model_path)} "
            f"size={self.input_size}"
        )


def _load_embedding_converter(model_name: str) -> onnxruntime.InferenceSession:
    """Load crossface ONNX converter from disk."""
    if model_name not in EMBEDDING_CONVERTERS:
        path = f"checkpoints/face_swapper/{model_name}"
        if not os.path.exists(path):
            raise FileNotFoundError(f"Converter not found: {path}")
        EMBEDDING_CONVERTERS[model_name] = onnxruntime.InferenceSession(path, None)
    return EMBEDDING_CONVERTERS[model_name]


def get_face_swapper_model(model_name: str) -> _SwapperModel:
    """Load face swapper model on first use, cache for subsequent calls."""
    if model_name not in FACE_SWAPPER_MODELS:
        model_path = f"checkpoints/face_swapper/{model_name}.onnx"
        logger.info(f"Loading face swapper model: {model_name}")
        FACE_SWAPPER_MODELS[model_name] = _SwapperModel(model_path)
    return FACE_SWAPPER_MODELS[model_name]


# ---------------------------------------------------------------------------
# Source preparation (forks FaceFusion's prepare_source_embedding / frame)
# ---------------------------------------------------------------------------


def _prepare_embedding_projected(source_face, model) -> np.ndarray:
    """inswapper: project embedding through emap.

    FaceFusion normalizes the *source* before the dot, not after.
    """
    latent = source_face.normed_embedding.reshape((1, -1))
    return np.dot(latent, model.emap) / np.linalg.norm(latent)


def _prepare_embedding_raw(source_face, converter_name: str) -> np.ndarray:
    """simswap/ghost: reshape, run through crossface ONNX, L2-norm."""
    embedding = source_face.normed_embedding.reshape((-1, 512))
    converter = _load_embedding_converter(converter_name)
    converted = converter.run(None, {"input": embedding})[0].ravel()
    norm = converted / np.linalg.norm(converted)
    return norm.reshape(1, -1)


def _prepare_source_face(source_face, temp_frame, source_size: int) -> np.ndarray:
    """blendswap/uniface: warp source face to template.

    FaceFusion does NOT apply mean/std here — just BGR→RGB, /255, CHW, batch.
    """
    source_img, _ = face_align.norm_crop2(
        temp_frame, source_face.kps, source_size
    )
    blob = source_img[:, :, ::-1].astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)
    return np.expand_dims(blob, axis=0).astype(np.float32)


def _balance_embedding(
    source_embedding: np.ndarray,
    target_embedding: np.ndarray,
    weight: float,
) -> np.ndarray:
    """FaceFusion's balance_source_embedding.

    Interpolates between source and target: weight=1.0 slightly strengthens
    the swap by anti-mixing the target identity.

    Interpolation: weight [0, 1] → w [0.35, -0.35]
    result = source * (1 - w) + target * w
    """
    w = np.interp(weight, [0, 1], [0.35, -0.35]).astype(np.float32)
    tgt = target_embedding.reshape((1, -1))
    tgt_norm = np.linalg.norm(tgt)
    if tgt_norm > 0:
        tgt = tgt / tgt_norm
    src = source_embedding.reshape((1, -1))
    return src * (1 - w) + tgt * w


# ---------------------------------------------------------------------------
# Target image preprocessing
# ---------------------------------------------------------------------------


def _prepare_crop_frame(
    crop_frame: np.ndarray, mean: list, std: list
) -> np.ndarray:
    """Normalize cropped face for ONNX: BGR→RGB, /255, (x - μ) / σ, CHW, batch."""
    mean_np = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
    std_np = np.array(std, dtype=np.float32).reshape(1, 1, 3)
    frame = crop_frame[:, :, ::-1].astype(np.float32) / 255.0
    frame = (frame - mean_np) / std_np
    frame = frame.transpose(2, 0, 1)
    return np.expand_dims(frame, axis=0).astype(np.float32)


def _normalize_crop_frame(
    crop_frame: np.ndarray, mean: list, std: list, tanh_out: bool
) -> np.ndarray:
    """Reverse normalize: CHW→HWC, tanh remap, clip, RGB→BGR, *255."""
    frame = crop_frame[0].transpose(1, 2, 0)
    if tanh_out:
        mean_np = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        std_np = np.array(std, dtype=np.float32).reshape(1, 1, 3)
        frame = frame * std_np + mean_np
    frame = np.clip(frame, 0.0, 1.0)
    return (frame[:, :, ::-1] * 255.0).astype(np.uint8)


# ---------------------------------------------------------------------------
# Paste-back (forked from insightface INSwapper.get)
# ---------------------------------------------------------------------------


def _paste_back(
    swapped_face: np.ndarray,
    aimg: np.ndarray,
    temp_frame: np.ndarray,
    affine_matrix: np.ndarray,
) -> np.ndarray:
    """Paste swapped face into original frame with diff-based feathering."""
    fake_diff = np.abs(
        swapped_face.astype(np.float32) - aimg.astype(np.float32)
    ).mean(axis=2)
    fake_diff[:2, :] = 0
    fake_diff[-2:, :] = 0
    fake_diff[:, :2] = 0
    fake_diff[:, -2:] = 0

    IM = cv2.invertAffineTransform(affine_matrix)
    img_white = np.full((aimg.shape[0], aimg.shape[1]), 255, dtype=np.float32)

    frame_h, frame_w = temp_frame.shape[1], temp_frame.shape[0]
    bgr_fake = cv2.warpAffine(swapped_face, IM, (frame_h, frame_w), borderValue=0.0)
    img_white = cv2.warpAffine(img_white, IM, (frame_h, frame_w), borderValue=0.0)
    fake_diff = cv2.warpAffine(fake_diff, IM, (frame_h, frame_w), borderValue=0.0)
    img_white[img_white > 20] = 255

    fake_diff[fake_diff < 10] = 0
    fake_diff[fake_diff >= 10] = 255

    img_mask = img_white
    mask_h_inds, mask_w_inds = np.where(img_mask == 255)
    if len(mask_h_inds) == 0 or len(mask_w_inds) == 0:
        return temp_frame

    mask_h = np.max(mask_h_inds) - np.min(mask_h_inds)
    mask_w = np.max(mask_w_inds) - np.min(mask_w_inds)
    mask_size = int(np.sqrt(mask_h * mask_w))

    img_mask = cv2.erode(img_mask, np.ones((max(mask_size // 10, 10), max(mask_size // 10, 10)), np.uint8), iterations=1)
    fake_diff = cv2.dilate(fake_diff, np.ones((2, 2), np.uint8), iterations=1)

    k = max(mask_size // 20, 5)
    blur = tuple(2 * i + 1 for i in (k, k))
    img_mask = cv2.GaussianBlur(img_mask, blur, 0)
    fake_diff = cv2.GaussianBlur(fake_diff, tuple(2 * i + 1 for i in (5, 5)), 0)
    img_mask = (img_mask / 255).reshape(img_mask.shape[0], img_mask.shape[1], 1)
    fake_diff /= 255

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
    """Resolution-aware face swap forking FaceFusion's full pipeline."""
    if not hasattr(model, "session"):
        raise TypeError(f"Expected swapper model with .session, got {type(model).__name__}")

    try:
        meta = get_model_metadata(model_name)
    except KeyError as e:
        raise ValueError(f"Unknown model: '{model_name}'") from e

    native_size = meta["native_size"]
    mean, std = meta["mean"], meta["std"]
    tanh_out = meta["tanh_out"]
    source_type = meta["source_type"]
    target_resolution = resolution[0]

    # --- Target face: warp, resize, normalize ---
    aimg, M = face_align.norm_crop2(temp_frame, target_face.kps, target_resolution)
    aimg_resized = cv2.resize(aimg, native_size)
    target_blob = _prepare_crop_frame(aimg_resized, mean, std)

    # --- Source input: depends on model family ---
    if source_type == "embedding_projected":
        source_input = _prepare_embedding_projected(source_face, model)
        source_input = _balance_embedding(source_input, target_face.normed_embedding, weight)

    elif source_type == "embedding":
        converter = meta.get("converter")
        source_input = _prepare_embedding_raw(source_face, converter)
        source_input = _balance_embedding(source_input, target_face.normed_embedding, weight)

    elif source_type == "source_face":
        source_size = meta["source_size"]
        source_input = _prepare_source_face(source_face, temp_frame, source_size)
        # source_face models don't use balance_embedding or weight beyond the
        # final cv2.addWeighted blending step below
    else:
        raise ValueError(f"Unknown source_type: '{source_type}'")

    # --- ONNX inference ---
    try:
        pred = model.session.run(
            model.output_names,
            {model.input_names[0]: target_blob, model.input_names[1]: source_input},
        )[0]
    except Exception as e:
        raise RuntimeError(f"ONNX inference failed for '{model_name}': {e}") from e

    # --- Post-process ---
    bgr_fake_native = _normalize_crop_frame(pred, mean, std, tanh_out)
    bgr_fake = cv2.resize(bgr_fake_native, (target_resolution, target_resolution))
    result = _paste_back(bgr_fake, aimg, temp_frame, M)

    # --- Final weight blend for source_face models ---
    if source_type == "source_face" and weight < 1.0:
        result = cv2.addWeighted(temp_frame, 1.0 - weight, result, weight, 0)

    return result
