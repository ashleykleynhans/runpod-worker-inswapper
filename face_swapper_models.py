# face_swapper_models.py
"""Face swapper model definitions and validation logic."""

from typing import Dict, List

# Model compatibility matrix: model name -> supported resolutions
FACE_SWAPPER_MODEL_SET: Dict[str, List[str]] = {
    'blendswap_256': ['256x256', '384x384', '512x512', '768x768', '1024x1024'],
    'ghost_1_256': ['256x256', '512x512', '768x768', '1024x1024'],
    'ghost_2_256': ['256x256', '512x512', '768x768', '1024x1024'],
    'ghost_3_256': ['256x256', '512x512', '768x768', '1024x1024'],
    'inswapper_128': ['128x128', '256x256', '384x384', '512x512', '768x768', '1024x1024'],
    'inswapper_128_fp16': ['128x128', '256x256', '384x384', '512x512', '768x768', '1024x1024'],
    'simswap_256': ['256x256', '512x512', '768x768', '1024x1024'],
    'simswap_unofficial_512': ['512x512', '768x768', '1024x1024'],
    'uniface_256': ['256x256', '512x512', '768x768', '1024x1024']
}

# Default resolutions when not specified by user
DEFAULT_RESOLUTIONS: Dict[str, str] = {
    'inswapper_128': '512x512',
    'inswapper_128_fp16': '512x512',
    'simswap_unofficial_512': '512x512',
    'default': '1024x1024'
}


def validate_face_swapper_params(model_name: str, resolution: str) -> None:
    """
    Validate model and resolution compatibility.

    Args:
        model_name: Face swapper model name
        resolution: Resolution string (e.g., '512x512')

    Raises:
        ValueError: If model is invalid or resolution not supported by model
    """
    if model_name not in FACE_SWAPPER_MODEL_SET:
        valid_models = ', '.join(sorted(FACE_SWAPPER_MODEL_SET.keys()))
        raise ValueError(
            f"Invalid face_swapper_model: '{model_name}'. "
            f"Valid options: {valid_models}"
        )

    if resolution not in FACE_SWAPPER_MODEL_SET[model_name]:
        valid_resolutions = ', '.join(FACE_SWAPPER_MODEL_SET[model_name])
        raise ValueError(
            f"Model '{model_name}' does not support resolution '{resolution}'. "
            f"Valid resolutions for this model: {valid_resolutions}"
        )


def get_default_resolution(model_name: str) -> str:
    """
    Get default resolution for a model.

    Args:
        model_name: Face swapper model name

    Returns:
        Default resolution string for the model
    """
    return DEFAULT_RESOLUTIONS.get(model_name, DEFAULT_RESOLUTIONS['default'])
