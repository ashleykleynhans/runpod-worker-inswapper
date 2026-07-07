# face_swapper_models.py
"""Face swapper model definitions and validation logic."""

from typing import Dict, List, Tuple

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

# Per-model metadata for FaceFusion-forked preprocessing pipeline
# mean/std arrays are BGR order (matching cv2 image channels)
MODEL_METADATA: Dict[str, dict] = {
    'blendswap_256': {
        'native_size': (256, 256),
        'mean': [0.0, 0.0, 0.0],
        'std': [1.0, 1.0, 1.0],
        'tanh_out': False,
    },
    'ghost_1_256': {
        'native_size': (256, 256),
        'mean': [0.5, 0.5, 0.5],
        'std': [0.5, 0.5, 0.5],
        'tanh_out': True,
    },
    'ghost_2_256': {
        'native_size': (256, 256),
        'mean': [0.5, 0.5, 0.5],
        'std': [0.5, 0.5, 0.5],
        'tanh_out': True,
    },
    'ghost_3_256': {
        'native_size': (256, 256),
        'mean': [0.5, 0.5, 0.5],
        'std': [0.5, 0.5, 0.5],
        'tanh_out': True,
    },
    'inswapper_128': {
        'native_size': (128, 128),
        'mean': [0.0, 0.0, 0.0],
        'std': [1.0, 1.0, 1.0],
        'tanh_out': False,
    },
    'inswapper_128_fp16': {
        'native_size': (128, 128),
        'mean': [0.0, 0.0, 0.0],
        'std': [1.0, 1.0, 1.0],
        'tanh_out': False,
    },
    'simswap_256': {
        'native_size': (256, 256),
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225],
        'tanh_out': False,
    },
    'simswap_unofficial_512': {
        'native_size': (512, 512),
        'mean': [0.0, 0.0, 0.0],
        'std': [1.0, 1.0, 1.0],
        'tanh_out': False,
    },
    'uniface_256': {
        'native_size': (256, 256),
        'mean': [0.5, 0.5, 0.5],
        'std': [0.5, 0.5, 0.5],
        'tanh_out': True,
    },
}


def validate_face_swapper_params(model_name: str, resolution: str) -> None:
    """Validate model and resolution compatibility."""
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
    """Get default resolution for a model."""
    return DEFAULT_RESOLUTIONS.get(model_name, DEFAULT_RESOLUTIONS['default'])


def parse_resolution(resolution: str) -> Tuple[int, int]:
    """Parse a resolution string like '512x512' into an (int, int) tuple."""
    try:
        parts = resolution.lower().split('x')
        if len(parts) != 2:
            raise ValueError(
                f"Invalid resolution format: '{resolution}'. "
                f"Expected format: 'WIDTHxHEIGHT' (e.g., '512x512')"
            )
        width, height = int(parts[0]), int(parts[1])
        return (width, height)
    except (ValueError, AttributeError) as e:
        raise ValueError(
            f"Invalid resolution format: '{resolution}'. "
            f"Expected format: 'WIDTHxHEIGHT' (e.g., '512x512')"
        ) from e


def get_model_metadata(model_name: str) -> dict:
    """Get preprocessing metadata for a face swapper model."""
    if model_name not in MODEL_METADATA:
        raise KeyError(f"Model metadata not found for: '{model_name}'")
    return MODEL_METADATA[model_name]
