# tests/test_face_swapper.py
import pytest
import numpy as np
from unittest.mock import patch, Mock
from face_swapper import (
    FACE_SWAPPER_MODELS,
    get_face_swapper_model,
    _prepare_crop_frame,
    _normalize_crop_frame,
    _prepare_source_embedding,
    _paste_back,
)


def test_get_face_swapper_model_loads():
    """Model should load via insightface and be cached."""
    FACE_SWAPPER_MODELS.clear()
    mock_instance = Mock()

    with (
        patch('os.path.exists', return_value=True),
        patch('face_swapper.insightface.model_zoo.get_model',
              return_value=mock_instance) as mock_get,
    ):
        model = get_face_swapper_model('inswapper_128')
        assert model is not None
        mock_get.assert_called_once()


def test_get_face_swapper_model_caches():
    """Model should be cached after first load."""
    FACE_SWAPPER_MODELS.clear()
    mock_instance = Mock()

    with (
        patch('os.path.exists', return_value=True),
        patch('face_swapper.insightface.model_zoo.get_model',
              return_value=mock_instance),
    ):
        model1 = get_face_swapper_model('simswap_256')
        model2 = get_face_swapper_model('simswap_256')
        assert model1 is model2


def test_get_face_swapper_model_missing_file():
    """Should raise FileNotFoundError for missing model."""
    with patch('os.path.exists', return_value=False):
        with pytest.raises(FileNotFoundError, match="Model file not found"):
            get_face_swapper_model('nonexistent_model')


# --- Tests for preprocessing pipeline ---


def test_prepare_crop_frame_inswapper():
    """inswapper_128 normalization: BGR->RGB, /255, identity norm."""
    dummy = np.full((128, 128, 3), 128, dtype=np.uint8)
    blob = _prepare_crop_frame(dummy, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    assert blob.shape == (1, 3, 128, 128)
    assert blob.dtype == np.float32
    assert np.allclose(blob, 128.0 / 255.0, atol=0.01)


def test_prepare_crop_frame_simswap():
    """simswap_256 normalization: ImageNet mean/std."""
    dummy = np.zeros((256, 256, 3), dtype=np.uint8)
    blob = _prepare_crop_frame(
        dummy, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    )
    assert blob.shape == (1, 3, 256, 256)
    assert np.all(blob < 0)


def test_prepare_crop_shape():
    """Output should be (1, 3, H, W) for any input shape."""
    dummy = np.zeros((256, 256, 3), dtype=np.uint8)
    blob = _prepare_crop_frame(dummy, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    assert blob.shape == (1, 3, 256, 256)
    assert blob.dtype == np.float32


def test_normalize_crop_frame_nontanh():
    """Non-tanh models: clip output to [0, 255]."""
    pred = np.random.randn(1, 3, 128, 128).astype(np.float32) * 0.3 + 0.5
    out = _normalize_crop_frame(pred, [0.0, 0.0, 0.0],
                                 [1.0, 1.0, 1.0], tanh_out=False)
    assert out.shape == (128, 128, 3)
    assert out.dtype == np.uint8
    assert out.min() >= 0
    assert out.max() <= 255


def test_normalize_crop_frame_tanh():
    """Tanh models: output remapped from [-1,1] to [0,255]."""
    pred = np.zeros((1, 3, 256, 256), dtype=np.float32)
    out = _normalize_crop_frame(pred, [0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5], tanh_out=True)
    assert out.shape == (256, 256, 3)
    assert out.dtype == np.uint8
    assert np.allclose(out.astype(float), 127.5, atol=1.0)


def test_prepare_source_embedding():
    """Embedding should be L2-normalized after projection."""
    mock_model = Mock()
    mock_model.emap = np.eye(512, dtype=np.float32)
    mock_source = Mock()
    mock_source.normed_embedding = np.ones((1, 512), dtype=np.float32)

    latent = _prepare_source_embedding(mock_source, mock_model)
    assert latent.shape == (1, 512)
    assert abs(np.linalg.norm(latent) - 1.0) < 0.001


def test_paste_back_empty_mask_early_return():
    """When all pixels warp off-screen, return original frame (line 204)."""
    swapped = np.ones((128, 128, 3), dtype=np.uint8) * 200
    aimg = np.ones((128, 128, 3), dtype=np.uint8) * 100
    temp_frame = np.ones((256, 256, 3), dtype=np.uint8) * 50
    # Translation off-screen: inverse warp maps everything negative
    M = np.array([[1, 0, 500], [0, 1, 500]], dtype=np.float32)
    result = _paste_back(swapped, aimg, temp_frame, M)
    assert result.shape == temp_frame.shape
    assert np.array_equal(result, temp_frame)


def test_paste_back_full_feathering():
    """Full paste-back with in-frame mask covers lines 206-231."""
    swapped = np.ones((128, 128, 3), dtype=np.uint8) * 200
    aimg = np.ones((128, 128, 3), dtype=np.uint8) * 100
    temp_frame = np.ones((256, 256, 3), dtype=np.uint8) * 50
    M = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)

    result = _paste_back(swapped, aimg, temp_frame, M)
    assert result.shape == (256, 256, 3)
    assert result.dtype == np.uint8


def test_swap_face_enhanced_full_pipeline():
    """Full swap_face_enhanced pipeline with mocked dependencies."""
    from face_swapper import swap_face_enhanced

    mock_source = Mock()
    mock_source.normed_embedding = np.ones((1, 512), dtype=np.float32)

    mock_target = Mock()
    mock_target.kps = np.random.randn(5, 2).astype(np.float32)

    temp_frame = np.ones((512, 512, 3), dtype=np.uint8) * 100

    mock_model = Mock()
    mock_model.emap = np.eye(512, dtype=np.float32)
    mock_model.input_names = ["input", "latent"]
    mock_model.output_names = ["output"]
    mock_model.session.run = Mock(return_value=[
        np.random.randn(1, 3, 128, 128).astype(np.float32) * 0.2 + 0.5
    ])

    resolution = (512, 512)

    with (
        patch('face_swapper.face_align.norm_crop2') as mock_norm,
        patch('face_swapper.cv2.resize') as mock_resize,
        patch('face_swapper._prepare_crop_frame') as mock_prep,
        patch('face_swapper._normalize_crop_frame') as mock_normcrop,
        patch('face_swapper._paste_back') as mock_paste,
        patch('face_swapper.cv2.addWeighted') as mock_blend,
    ):
        mock_norm.return_value = (
            np.ones((512, 512, 3), dtype=np.uint8),
            np.eye(2, 3, dtype=np.float32),
        )
        mock_resize.return_value = np.ones((128, 128, 3), dtype=np.uint8)
        mock_prep.return_value = np.ones((1, 3, 128, 128), dtype=np.float32)
        mock_normcrop.return_value = np.ones((128, 128, 3), dtype=np.uint8)
        mock_paste.return_value = temp_frame.copy()

        result = swap_face_enhanced(
            mock_source, mock_target, temp_frame, mock_model,
            'inswapper_128', resolution, weight=1.0,
        )

        mock_norm.assert_called_once_with(temp_frame, mock_target.kps, 512)
        mock_resize.assert_called()
        mock_prep.assert_called_once()
        mock_model.session.run.assert_called_once()
        mock_normcrop.assert_called_once()
        mock_paste.assert_called_once()
        mock_blend.assert_not_called()
        assert result is not None


def test_swap_face_enhanced_with_weight_blending():
    """swap_face_enhanced with weight < 1.0 triggers blending."""
    from face_swapper import swap_face_enhanced

    mock_source = Mock()
    mock_source.normed_embedding = np.ones((1, 512), dtype=np.float32)
    mock_target = Mock()
    mock_target.kps = np.random.randn(5, 2).astype(np.float32)

    temp_frame = np.ones((256, 256, 3), dtype=np.uint8) * 100

    mock_model = Mock()
    mock_model.emap = np.eye(512, dtype=np.float32)
    mock_model.input_names = ["input", "latent"]
    mock_model.output_names = ["output"]
    mock_model.session.run = Mock(return_value=[
        np.zeros((1, 3, 256, 256), dtype=np.float32)
    ])

    resolution = (256, 256)

    with (
        patch('face_swapper.face_align.norm_crop2') as mock_norm,
        patch('face_swapper.cv2.resize'),
        patch('face_swapper._prepare_crop_frame'),
        patch('face_swapper._normalize_crop_frame'),
        patch('face_swapper._paste_back') as mock_paste,
        patch('face_swapper.cv2.addWeighted') as mock_blend,
    ):
        mock_norm.return_value = (
            np.ones((256, 256, 3), dtype=np.uint8),
            np.eye(2, 3, dtype=np.float32),
        )
        mock_paste.return_value = temp_frame.copy()
        mock_blend.return_value = temp_frame.copy()

        swap_face_enhanced(
            mock_source, mock_target, temp_frame, mock_model,
            'uniface_256', resolution, weight=0.7,
        )

        mock_blend.assert_called_once()
