# face_swapper.py
"""Enhanced face swapping with multi-model support and blending."""

import os
import cv2
import numpy as np
import insightface
from typing import Dict
from runpod.serverless.modules.rp_logger import RunPodLogger

logger = RunPodLogger()

# Global model cache for lazy loading
FACE_SWAPPER_MODELS: Dict[str, any] = {}


def get_face_swapper_model(model_name: str):
    """
    Load face swapper model on first use, cache for subsequent calls.

    Args:
        model_name: Model name (e.g., 'simswap_256')

    Returns:
        Loaded ONNX model

    Raises:
        FileNotFoundError: If model file doesn't exist
    """
    if model_name not in FACE_SWAPPER_MODELS:
        # Try new location first, fallback to old location for inswapper_128
        if model_name == 'inswapper_128':
            model_path = 'checkpoints/inswapper_128.onnx'
        else:
            model_path = f'checkpoints/face_swapper/{model_name}.onnx'

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        logger.info(f"Loading face swapper model: {model_name}")
        FACE_SWAPPER_MODELS[model_name] = insightface.model_zoo.get_model(model_path)
        logger.info(f"Loaded face swapper model: {model_name}")

    return FACE_SWAPPER_MODELS[model_name]


def swap_face_enhanced(
    source_face,
    target_face,
    temp_frame: np.ndarray,
    model,
    weight: float = 1.0
) -> np.ndarray:
    """
    Enhanced face swapping with blending support.

    Args:
        source_face: Source face object from InsightFace
        target_face: Target face object from InsightFace
        temp_frame: Frame to swap face in
        model: Face swapper model
        weight: Blend weight (0.0-1.0), 1.0 = full swap

    Returns:
        Frame with swapped face
    """
    # Perform the swap using the model
    swapped_frame = model.get(temp_frame, target_face, source_face, paste_back=True)

    # Apply blending if weight < 1.0
    if weight < 1.0:
        swapped_frame = cv2.addWeighted(
            temp_frame, 1.0 - weight,
            swapped_frame, weight,
            0
        )

    return swapped_frame
