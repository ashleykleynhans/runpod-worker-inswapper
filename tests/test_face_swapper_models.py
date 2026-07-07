# tests/test_face_swapper_models.py
import pytest
from face_swapper_models import (
    FACE_SWAPPER_MODEL_SET,
    DEFAULT_RESOLUTIONS,
    MODEL_METADATA,
    validate_face_swapper_params,
    get_default_resolution,
    parse_resolution,
    get_model_metadata,
)


def test_face_swapper_model_set_exists():
    """Verify all 9 models are defined"""
    assert len(FACE_SWAPPER_MODEL_SET) == 9
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


def test_model_metadata_has_all_models():
    """MODEL_METADATA should cover all models in FACE_SWAPPER_MODEL_SET"""
    assert len(MODEL_METADATA) == len(FACE_SWAPPER_MODEL_SET)
    for model_name in FACE_SWAPPER_MODEL_SET:
        assert model_name in MODEL_METADATA


def test_get_model_metadata_inswapper():
    """inswapper_128 should have identity normalization"""
    meta = get_model_metadata('inswapper_128')
    assert meta['native_size'] == (128, 128)
    assert meta['mean'] == [0.0, 0.0, 0.0]
    assert meta['std'] == [1.0, 1.0, 1.0]
    assert meta['tanh_out'] is False


def test_get_model_metadata_simswap():
    """simswap_256 should have ImageNet normalization"""
    meta = get_model_metadata('simswap_256')
    assert meta['native_size'] == (256, 256)
    assert meta['tanh_out'] is False


def test_get_model_metadata_ghost():
    """Ghost models should have tanh_out=True"""
    meta = get_model_metadata('ghost_1_256')
    assert meta['tanh_out'] is True
    assert meta['mean'] == [0.5, 0.5, 0.5]


def test_get_model_metadata_missing():
    """Missing model should raise KeyError"""
    with pytest.raises(KeyError, match="Model metadata not found"):
        get_model_metadata('nonexistent_model')


def test_parse_resolution_valid():
    """Valid resolution strings should parse correctly"""
    assert parse_resolution('128x128') == (128, 128)
    assert parse_resolution('256x256') == (256, 256)
    assert parse_resolution('512x512') == (512, 512)


def test_parse_resolution_case_insensitive():
    """Resolution parsing should be case-insensitive"""
    assert parse_resolution('512X512') == (512, 512)


def test_parse_resolution_invalid():
    """Invalid resolution strings should raise ValueError"""
    with pytest.raises(ValueError, match="Invalid resolution format"):
        parse_resolution('not_a_resolution')
