# tests/test_face_swapper_models.py
import pytest
from face_swapper_models import (
    FACE_SWAPPER_MODEL_SET,
    DEFAULT_RESOLUTIONS,
    validate_face_swapper_params,
    get_default_resolution
)


def test_face_swapper_model_set_exists():
    """Verify all 13 models are defined"""
    assert len(FACE_SWAPPER_MODEL_SET) == 13
    assert 'inswapper_128' in FACE_SWAPPER_MODEL_SET
    assert 'simswap_256' in FACE_SWAPPER_MODEL_SET


def test_validate_model_resolution_valid():
    """Valid model/resolution combinations should pass"""
    validate_face_swapper_params('simswap_256', '512x512')
    validate_face_swapper_params('inswapper_128', '128x128')
    # Should not raise


def test_validate_model_resolution_invalid_model():
    """Invalid model should raise ValueError"""
    with pytest.raises(ValueError, match="Invalid face_swapper_model"):
        validate_face_swapper_params('invalid_model', '512x512')


def test_validate_model_resolution_invalid_resolution():
    """Invalid resolution for model should raise ValueError"""
    with pytest.raises(ValueError, match="does not support resolution"):
        validate_face_swapper_params('simswap_unofficial_512', '256x256')


def test_get_default_resolution_inswapper():
    """inswapper_128 should default to 512x512"""
    assert get_default_resolution('inswapper_128') == '512x512'


def test_get_default_resolution_simswap():
    """simswap_256 should default to 1024x1024"""
    assert get_default_resolution('simswap_256') == '1024x1024'


def test_get_default_resolution_unknown():
    """Unknown model should use default"""
    assert get_default_resolution('unknown_model') == '1024x1024'
