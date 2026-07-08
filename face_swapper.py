# face_swapper.py
"""Enhanced face swapping forking FaceFusion's full preprocessing pipeline.

Key FaceFusion features (not available in insightface):
- cv2.estimateAffinePartial2D + RANSAC for face alignment
- Per-model warp templates (arcface_128, arcface_112_v1, ffhq_512)
- BORDER_REPLICATE (no black borders)
- Box mask + paste_back (smoother blending than insightface's diff-mask)
- Per-model source preparation (emap-projected, crossface converter, source-face warp)
- Embedding balancing for weight control
"""

import os
import cv2
import numpy as np
import onnx
import onnxruntime
from onnx import numpy_helper
from insightface.utils import face_align
from runpod.serverless.modules.rp_logger import RunPodLogger
from typing import Dict, Tuple

from face_swapper_models import get_model_metadata

logger = RunPodLogger()

# Global caches
FACE_SWAPPER_MODELS: Dict[str, object] = {}
EMBEDDING_CONVERTERS: Dict[str, object] = {}

# FaceFusion warp templates (from face_helper.py WARP_TEMPLATE_SET)
WARP_TEMPLATES = {
    "arcface_112_v1": np.array(
        [[0.35473214, 0.45658929],
         [0.64526786, 0.45658929],
         [0.50000000, 0.61154464],
         [0.37913393, 0.77687500],
         [0.62086607, 0.77687500]]
    ),
    "arcface_128": np.array(
        [[0.36167656, 0.40387734],
         [0.63696719, 0.40235469],
         [0.50019687, 0.56044219],
         [0.38710391, 0.72160547],
         [0.61507734, 0.72034453]]
    ),
    "ffhq_512": np.array(
        [[0.37691676, 0.46864664],
         [0.62285697, 0.46912813],
         [0.50123859, 0.61331904],
         [0.39308822, 0.72541100],
         [0.61150205, 0.72490465]]
    ),
}


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
# FaceFusion-forked warp (replaces insightface's norm_crop2)
# ---------------------------------------------------------------------------


def _warp_face_by_landmark_5(
    temp_frame: np.ndarray,
    face_landmark_5: np.ndarray,
    warp_template: str,
    crop_size: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Forks FaceFusion's warp_face_by_face_landmark_5.

    Uses cv2.estimateAffinePartial2D + RANSAC (not skimage.SimilarityTransform)
    and BORDER_REPLICATE (not black borders).
    """
    template = WARP_TEMPLATES[warp_template] * np.array(crop_size)
    affine_matrix = cv2.estimateAffinePartial2D(
        face_landmark_5.astype(np.float32),
        template.astype(np.float32),
        method=cv2.RANSAC,
        ransacReprojThreshold=100,
    )[0]
    if affine_matrix is None:
        # RANSAC failed; fall back to insightface
        return face_align.norm_crop2(
            temp_frame, face_landmark_5, crop_size[0]
        )
    cropped = cv2.warpAffine(
        temp_frame, affine_matrix, crop_size,
        borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_AREA,
    )
    return cropped, affine_matrix


# ---------------------------------------------------------------------------
# FaceFusion-forked paste-back (replaces insightface's diff-mask)
# ---------------------------------------------------------------------------


def _create_box_mask(
    crop_size: Tuple[int, int],
    blur: float = 0.3,
    padding: Tuple[int, int, int, int] = (0, 0, 0, 0),
) -> np.ndarray:
    """FaceFusion's create_box_mask: soft border mask with optional padding."""
    w, h = crop_size
    blur_amount = int(w * 0.5 * blur)
    blur_area = max(blur_amount // 2, 1)
    mask = np.ones((h, w), dtype=np.float32)
    mask[:max(blur_area, int(h * padding[0] / 100)), :] = 0
    mask[-max(blur_area, int(h * padding[2] / 100)):, :] = 0
    mask[:, :max(blur_area, int(w * padding[3] / 100))] = 0
    mask[:, -max(blur_area, int(w * padding[1] / 100)):] = 0
    if blur_amount > 0:
        mask = cv2.GaussianBlur(mask, (0, 0), blur_amount * 0.25)
    return mask


def _transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """FaceFusion's transform_points helper."""
    pts = points.reshape(-1, 1, 2)
    pts = cv2.transform(pts, matrix)
    return pts.reshape(-1, 2)


def _calculate_paste_area(
    temp_frame: np.ndarray,
    crop_frame: np.ndarray,
    affine_matrix: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """FaceFusion's calculate_paste_area."""
    th, tw = temp_frame.shape[:2]
    ch, cw = crop_frame.shape[:2]
    inv = cv2.invertAffineTransform(affine_matrix)
    crop_pts = np.array([[0, 0], [cw, 0], [cw, ch], [0, ch]])
    paste_pts = _transform_points(crop_pts, inv)
    pmin = np.floor(paste_pts.min(axis=0)).astype(int)
    pmax = np.ceil(paste_pts.max(axis=0)).astype(int)
    x1, y1 = np.clip(pmin, 0, [tw, th])
    x2, y2 = np.clip(pmax, 0, [tw, th])
    bbox = np.array([x1, y1, x2, y2])
    paste_m = inv.copy()
    paste_m[0, 2] -= x1
    paste_m[1, 2] -= y1
    return bbox, paste_m


def _paste_back(
    temp_frame: np.ndarray,
    crop_frame: np.ndarray,
    crop_mask: np.ndarray,
    affine_matrix: np.ndarray,
) -> np.ndarray:
    """FaceFusion's paste_back: alpha-blend with box mask."""
    bbox, paste_m = _calculate_paste_area(temp_frame, crop_frame, affine_matrix)
    x1, y1, x2, y2 = bbox
    pw, ph = x2 - x1, y2 - y1
    if pw <= 0 or ph <= 0:
        return temp_frame
    inv_mask = cv2.warpAffine(crop_mask, paste_m, (pw, ph)).clip(0, 1)
    inv_mask = np.expand_dims(inv_mask, axis=-1)
    inv_frame = cv2.warpAffine(
        crop_frame, paste_m, (pw, ph), borderMode=cv2.BORDER_REPLICATE
    )
    out = temp_frame.copy()
    paste_region = out[y1:y2, x1:x2]
    paste_region = paste_region * (1 - inv_mask) + inv_frame * inv_mask
    out[y1:y2, x1:x2] = paste_region.astype(out.dtype)
    return out


# ---------------------------------------------------------------------------
# Source preparation (forks FaceFusion's prepare_source_embedding / frame)
# ---------------------------------------------------------------------------


def _prepare_embedding_projected(source_face, model) -> np.ndarray:
    """inswapper: normalize source BEFORE dot(emap)."""
    latent = source_face.normed_embedding.reshape((1, -1))
    return np.dot(latent, model.emap) / np.linalg.norm(latent)


def _prepare_embedding_raw(source_face, converter_name: str) -> np.ndarray:
    """simswap/ghost: reshape, crossface ONNX, L2-norm."""
    embedding = source_face.normed_embedding.reshape((-1, 512))
    converter = _load_embedding_converter(converter_name)
    converted = converter.run(None, {"input": embedding})[0].ravel()
    return (converted / np.linalg.norm(converted)).reshape(1, -1)


def _prepare_source_face(source_face, temp_frame, source_size: int) -> np.ndarray:
    """blendswap/uniface: warp source face to template, no mean/std."""
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
    """FaceFusion's balance_source_embedding."""
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
    """FaceFusion's prepare_crop_frame: BGR→RGB, /255, (x-μ)/σ, CHW, batch."""
    mean_np = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
    std_np = np.array(std, dtype=np.float32).reshape(1, 1, 3)
    frame = crop_frame[:, :, ::-1].astype(np.float32) / 255.0
    frame = (frame - mean_np) / std_np
    frame = frame.transpose(2, 0, 1)
    return np.expand_dims(frame, axis=0).astype(np.float32)


def _normalize_crop_frame(
    crop_frame: np.ndarray, mean: list, std: list, tanh_out: bool
) -> np.ndarray:
    """FaceFusion's normalize_crop_frame: CHW→HWC, tanh remap, clip."""
    frame = crop_frame[0].transpose(1, 2, 0)
    if tanh_out:
        mean_np = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        std_np = np.array(std, dtype=np.float32).reshape(1, 1, 3)
        frame = frame * std_np + mean_np
    frame = np.clip(frame, 0.0, 1.0)
    return (frame[:, :, ::-1] * 255.0).astype(np.uint8)


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
    warp_template = meta.get("warp_template", "arcface_128")
    target_resolution = resolution[0]

    # --- Target face: FaceFusion warp (cv2.estimateAffinePartial2D + RANSAC) ---
    crop_size = (target_resolution, target_resolution)
    aimg, M = _warp_face_by_landmark_5(
        temp_frame, target_face.kps, warp_template, crop_size
    )

    # Create box mask (FaceFusion's default before pixel_boost)
    crop_mask = _create_box_mask(crop_size)

    # Resize to model native size & normalize
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

    # --- FaceFusion paste_back with box mask ---
    result = _paste_back(temp_frame, bgr_fake, crop_mask, M)

    # --- Final weight blend for source_face models ---
    if source_type == "source_face" and weight < 1.0:
        result = cv2.addWeighted(temp_frame, 1.0 - weight, result, weight, 0)

    return result
