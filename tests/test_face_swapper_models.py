# tests/test_face_swapper_models.py
import pytest
from face_swapper_models import (
    FACE_SWAPPER_MODEL_SET, DEFAULT_RESOLUTIONS, MODEL_METADATA,
    validate_face_swapper_params, get_default_resolution,
    parse_resolution, get_model_metadata,
)


def test_model_set_exists():
    assert len(FACE_SWAPPER_MODEL_SET) == 13
    assert 'inswapper_128' in FACE_SWAPPER_MODEL_SET
    assert 'hyperswap_1a_256' in FACE_SWAPPER_MODEL_SET
    assert 'hififace_unofficial_256' in FACE_SWAPPER_MODEL_SET


def test_validate_ok():
    validate_face_swapper_params('simswap_256', '512x512')
    validate_face_swapper_params('inswapper_128', '128x128')


def test_validate_bad_model():
    with pytest.raises(ValueError, match="Invalid face_swapper_model"):
        validate_face_swapper_params('bad', '512x512')


def test_validate_bad_resolution():
    with pytest.raises(ValueError, match="does not support resolution"):
        validate_face_swapper_params('simswap_unofficial_512', '256x256')


def test_default_res_inswapper():
    assert get_default_resolution('inswapper_128') == '512x512'


def test_default_res_simswap():
    assert get_default_resolution('simswap_256') == '1024x1024'


def test_default_res_unknown():
    assert get_default_resolution('unknown') == '1024x1024'


def test_metadata_covers_all():
    assert len(MODEL_METADATA) == len(FACE_SWAPPER_MODEL_SET)
    for m in FACE_SWAPPER_MODEL_SET:
        assert m in MODEL_METADATA


def test_metadata_inswapper():
    mt = get_model_metadata('inswapper_128')
    assert mt['native_size'] == (128, 128) and mt['source_type'] == 'embedding_projected'
    assert mt['warp_template'] == 'arcface_128'


def test_metadata_simswap():
    mt = get_model_metadata('simswap_256')
    assert mt['source_type'] == 'embedding' and mt['converter'] == 'crossface_simswap.onnx'


def test_metadata_ghost():
    mt = get_model_metadata('ghost_1_256')
    assert mt['tanh_out'] and mt['source_type'] == 'embedding'
    assert mt['warp_template'] == 'arcface_112_v1'


def test_metadata_hififace():
    mt = get_model_metadata('hififace_unofficial_256')
    assert mt['tanh_out'] and mt['source_type'] == 'embedding'
    assert mt['warp_template'] == 'mtcnn_512' and mt['converter'] == 'crossface_hififace.onnx'


def test_metadata_hyperswap():
    mt = get_model_metadata('hyperswap_1a_256')
    assert mt['tanh_out'] and mt['source_type'] == 'embedding_norm'
    assert mt['warp_template'] == 'arcface_128'


def test_metadata_blendswap():
    mt = get_model_metadata('blendswap_256')
    assert mt['source_type'] == 'source_face' and mt['source_size'] == 112


def test_metadata_uniface():
    mt = get_model_metadata('uniface_256')
    assert mt['source_type'] == 'source_face' and mt['source_size'] == 256 and mt['tanh_out']


def test_metadata_missing():
    with pytest.raises(KeyError, match="not found"):
        get_model_metadata('nope')


def test_parse_resolution():
    assert parse_resolution('512x512') == (512, 512)
    assert parse_resolution('128x128') == (128, 128)


def test_parse_resolution_case():
    assert parse_resolution('256X256') == (256, 256)


def test_parse_resolution_invalid():
    with pytest.raises(ValueError, match="Invalid resolution format"):
        parse_resolution('garbage')
