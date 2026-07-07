# face_swapper.py
"""Enhanced face swapping with multi-model support and blending.

Forks insightface's INSwapper.get() preprocessing pipeline to add:
- Configurable face alignment resolution (via face_swapper_resolution param)
- Model-specific input normalization (mean/std per model)
- Tanh-output model post-processing (ghost, hififace, hyperswap, uniface)
- Weight-based blending between original and swapped face
"""

import os
import cv2
import numpy as np
import insightface
from typing import Dict, Tuple
from insightface.utils import face_align
from runpod.serverless.modules.rp_logger import RunPodLogger

from face_swapper_models import get_model_metadata

logger = RunPodLogger()

# Global model cache for lazy loading
FACE_SWAPPER_MODELS: Dict[str, any] = {}


def get_face_swapper_model(model_name: str):
    """
    Load face swapper model on first use, cache for subsequent calls.

    The model is loaded via insightface.model_zoo.get_model() which routes
    to INSwapper (for models with 2 inputs and 128x128 native shape) or
    ArcFaceONNX (for other ONNX models). We access the underlying
    onnxruntime session for direct inference.

    Args:
        model_name: Model name (e.g., 'simswap_256')

    Returns:
        insightface model object with .session, .emap, .input_names,
        .output_names, and .input_size attributes

    Raises:
        FileNotFoundError: If model file doesn't exist
    """
    if model_name not in FACE_SWAPPER_MODELS:
        if model_name == 'inswapper_128':
            model_path = 'checkpoints/inswapper_128.onnx'
        else:
            model_path = f'checkpoints/face_swapper/{model_name}.onnx'

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        logger.info(f"Loading face swapper model: {model_name}")
        FACE_SWAPPER_MODELS[model_name] = insightface.model_zoo.get_model(
            model_path
        )
        logger.info(f"Loaded face swapper model: {model_name}")

    return FACE_SWAPPER_MODELS[model_name]


def _prepare_source_embedding(source_face, model) -> np.ndarray:
    """
    Prepare source face embedding for ONNX inference.

    Projects the source face's normed embedding through the model's
    embedding matrix and L2-normalizes.

    Args:
        source_face: InsightFace face object with .normed_embedding
        model: insightface INSwapper instance with .emap attribute

    Returns:
        L2-normalized latent vector shaped for model input
    """
    latent = source_face.normed_embedding.reshape((1, -1))
    latent = np.dot(latent, model.emap)
    latent /= np.linalg.norm(latent)
    return latent


def _prepare_crop_frame(
    crop_frame: np.ndarray,
    mean: list,
    std: list,
) -> np.ndarray:
    """
    Normalize a cropped face frame for ONNX inference.

    BGR -> RGB, scale to [0,1], normalize with model-specific
    mean/std, HWC -> CHW, add batch dimension.

    Args:
        crop_frame: BGR image in [0, 255] uint8
        mean: Per-channel mean for normalization (BGR order)
        std: Per-channel standard deviation for normalization (BGR order)

    Returns:
        Float32 blob of shape (1, 3, H, W)
    """
    mean_np = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
    std_np = np.array(std, dtype=np.float32).reshape(1, 1, 3)

    crop_frame = crop_frame[:, :, ::-1].astype(np.float32) / 255.0
    crop_frame = (crop_frame - mean_np) / std_np
    crop_frame = crop_frame.transpose(2, 0, 1)
    crop_frame = np.expand_dims(crop_frame, axis=0).astype(np.float32)
    return crop_frame


def _normalize_crop_frame(
    crop_frame: np.ndarray,
    mean: list,
    std: list,
    tanh_out: bool,
) -> np.ndarray:
    """
    Reverse the normalization applied in _prepare_crop_frame.

    CHW -> HWC, RGB -> BGR, scale to [0, 255] uint8.
    For tanh-output models (ghost, hyperswap, uniface), the output
    is in [-1, 1] and needs *std + mean mapping back to [0, 1].

    Args:
        crop_frame: ONNX output of shape (1, 3, H, W) in model range
        mean: Per-channel mean used in normalization (BGR order)
        std: Per-channel std used in normalization (BGR order)
        tanh_out: If True, output is in [-1, 1] range (tanh activation)

    Returns:
        BGR uint8 image in [0, 255]
    """
    crop_frame = crop_frame[0].transpose(1, 2, 0)

    if tanh_out:
        mean_np = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        std_np = np.array(std, dtype=np.float32).reshape(1, 1, 3)
        crop_frame = crop_frame * std_np + mean_np

    crop_frame = np.clip(crop_frame, 0.0, 1.0)
    crop_frame = crop_frame[:, :, ::-1] * 255.0
    return crop_frame.astype(np.uint8)


def _paste_back(
    swapped_face: np.ndarray,
    aimg: np.ndarray,
    temp_frame: np.ndarray,
    affine_matrix: np.ndarray,
) -> np.ndarray:
    """
    Paste the swapped face back into the original frame with feathering.

    Forks insightface's INSwapper.get() paste-back logic:
    - Compute difference mask between swapped and original aligned face
    - Erode/dilate + Gaussian blur for smooth feathering
    - Warp mask and swapped face back via inverse affine transform
    - Alpha blend with original frame

    Args:
        swapped_face: BGR uint8 swapped face at aligned resolution
        aimg: BGR uint8 aligned original face at same resolution
        temp_frame: BGR uint8 full-size original frame
        affine_matrix: 2x3 affine matrix from norm_crop2

    Returns:
        BGR uint8 frame with swapped face pasted back
    """
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

    fake_merged = img_mask * bgr_fake.astype(np.float32) + (
        1 - img_mask
    ) * temp_frame.astype(np.float32)
    return fake_merged.astype(np.uint8)


def swap_face_enhanced(
    source_face,
    target_face,
    temp_frame: np.ndarray,
    model,
    model_name: str,
    resolution: Tuple[int, int],
    weight: float = 1.0,
) -> np.ndarray:
    """
    Enhanced face swapping with configurable resolution and blending.

    Resolution-aware preprocessing pipeline forked from FaceFusion:
    1. Warp target face to user-requested resolution via affine alignment
    2. Resize to model native size for ONNX inference
    3. Apply model-specific normalization (mean/std)
    4. Run ONNX inference with source face embedding
    5. Post-process output (handles tanh models)
    6. Resize back to user resolution
    7. Paste back into original frame with feathering
    8. Apply weight-based blending

    Args:
        source_face: InsightFace face object with .normed_embedding
        target_face: InsightFace face object with .kps
        temp_frame: Full-size BGR frame to swap face in
        model: insightface model object with .session, .emap, .input_names,
               .output_names
        model_name: Model name for metadata lookup
        resolution: (width, height) tuple for face alignment resolution
        weight: Blend weight (0.0-1.0), 1.0 = full swap

    Returns:
        BGR frame with swapped and pasted face
    """
    metadata = get_model_metadata(model_name)
    native_size = metadata['native_size']
    mean = metadata['mean']
    std = metadata['std']
    tanh_out = metadata['tanh_out']

    target_resolution = resolution[0]

    # Step 1: Warp target face to user-requested resolution
    aimg, M = face_align.norm_crop2(
        temp_frame, target_face.kps, target_resolution
    )

    # Step 2: Resize to model native size for ONNX inference
    aimg_resized = cv2.resize(aimg, native_size)

    # Step 3: Apply model-specific normalization
    blob = _prepare_crop_frame(aimg_resized, mean, std)

    # Step 4: Prepare source embedding and run ONNX inference
    latent = _prepare_source_embedding(source_face, model)
    pred = model.session.run(
        model.output_names,
        {model.input_names[0]: blob, model.input_names[1]: latent},
    )[0]

    # Step 5: Post-process (handles tanh models)
    bgr_fake_native = _normalize_crop_frame(pred, mean, std, tanh_out)

    # Step 6: Resize output back to user resolution for paste-back
    bgr_fake = cv2.resize(
        bgr_fake_native, (target_resolution, target_resolution)
    )

    # Step 7: Paste back into original frame with feathering
    result = _paste_back(bgr_fake, aimg, temp_frame, M)

    # Step 8: Apply weight-based blending
    if weight < 1.0:
        result = cv2.addWeighted(
            temp_frame, 1.0 - weight, result, weight, 0
        )

    return result
