# tests/test_face_swapper.py
import pytest
import numpy as np
from unittest.mock import patch, Mock, MagicMock
from face_swapper import (
    FACE_SWAPPER_MODELS,
    _SwapperModel,
    get_face_swapper_model,
    _prepare_crop_frame,
    _normalize_crop_frame,
    _prepare_source_embedding,
    _paste_back,
)


# --- Helpers ---

def _make_mock_onnx_model():
    """Build a minimal ONNX graph that _SwapperModel can parse."""
    import onnx
    from onnx import helper, TensorProto

    # Create a graph with 2 inputs and an initializer for emap
    initializer = helper.make_tensor(
        name="emap", data_type=TensorProto.FLOAT,
        dims=[512, 512],
        vals=np.eye(512, dtype=np.float32).flatten().tolist(),
    )
    node = helper.make_node("Conv", ["input", "emap"], ["output"], name="node")
    graph = helper.make_graph(
        [node], "test",
        [
            helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 128, 128]),
            helper.make_tensor_value_info("latent", TensorProto.FLOAT, [1, 512]),
        ],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 128, 128])],
        [initializer],
    )
    model = helper.make_model(graph)
    return model


# --- Model loading tests ---


def test_get_face_swapper_model_loads():
    """_SwapperModel is created with valid ONNX file."""
    FACE_SWAPPER_MODELS.clear()

    onnx_model = _make_mock_onnx_model()

    with (
        patch('os.path.exists', return_value=True),
        patch('face_swapper.onnx.load', return_value=onnx_model),
        patch('face_swapper.onnxruntime.InferenceSession') as mock_session_cls,
    ):
        inp1, inp2 = Mock(), Mock()
        inp1.name, inp1.shape = "input", [1, 3, 128, 128]
        inp2.name, inp2.shape = "latent", [1, 512]
        mock_session_cls.return_value.get_inputs.return_value = [inp1, inp2]

        out = Mock()
        out.name = "output"
        mock_session_cls.return_value.get_outputs.return_value = [out]

        model = get_face_swapper_model('inswapper_128')
        assert isinstance(model, _SwapperModel)
        assert model.emap.shape == (512, 512)
        assert model.input_names == ["input", "latent"]
        assert model.output_names == ["output"]


def test_get_face_swapper_model_caches():
    """Model should be cached after first load."""
    FACE_SWAPPER_MODELS.clear()

    onnx_model = _make_mock_onnx_model()

    with (
        patch('os.path.exists', return_value=True),
        patch('face_swapper.onnx.load', return_value=onnx_model),
        patch('face_swapper.onnxruntime.InferenceSession') as mock_session_cls,
    ):
        mock_session_cls.return_value.get_inputs.return_value = [
            Mock(shape=[1, 3, 256, 256], name="input"),
            Mock(shape=[1, 512], name="latent"),
        ]
        mock_session_cls.return_value.get_outputs.return_value = [
            Mock(name="output"),
        ]

        model1 = get_face_swapper_model('simswap_256')
        model2 = get_face_swapper_model('simswap_256')
        assert model1 is model2


def test_get_face_swapper_model_missing_file():
    """Should raise FileNotFoundError for missing model."""
    with pytest.raises(FileNotFoundError, match="Model file not found"):
        _SwapperModel('/nonexistent/path.onnx')


def test_get_face_swapper_model_corrupt_onnx():
    """Should raise ValueError for corrupt ONNX file."""
    with (
        patch('os.path.exists', return_value=True),
        patch('face_swapper.onnx.load', side_effect=ValueError("bad file")),
    ):
        with pytest.raises(ValueError, match="Failed to load ONNX model"):
            _SwapperModel('/fake/path.onnx')


def test_get_face_swapper_model_single_input():
    """Should raise ValueError for model with only 1 input."""
    import onnx
    from onnx import helper, TensorProto

    graph = helper.make_graph(
        [], "test",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 128, 128])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 128, 128])],
    )
    model = helper.make_model(graph)

    with (
        patch('os.path.exists', return_value=True),
        patch('face_swapper.onnx.load', return_value=model),
    ):
        with pytest.raises(ValueError, match="expected at least 2"):
            _SwapperModel('/fake/path.onnx')


def test_get_face_swapper_model_corrupt_initializer():
    """Should raise ValueError when initializer can't be parsed."""
    onnx_model = _make_mock_onnx_model()

    with (
        patch('os.path.exists', return_value=True),
        patch('face_swapper.onnx.load', return_value=onnx_model),
        patch('face_swapper.numpy_helper.to_array',
              side_effect=ValueError("corrupt data")),
    ):
        with pytest.raises(ValueError, match="Failed to extract embedding"):
            _SwapperModel('/fake/path.onnx')


def test_get_face_swapper_model_no_initializers():
    """Should raise ValueError for model with no initializers."""
    import onnx
    from onnx import helper, TensorProto

    graph = helper.make_graph(
        [], "test",
        [
            helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 128, 128]),
            helper.make_tensor_value_info("latent", TensorProto.FLOAT, [1, 512]),
        ],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 128, 128])],
    )
    model = helper.make_model(graph)

    with (
        patch('os.path.exists', return_value=True),
        patch('face_swapper.onnx.load', return_value=model),
    ):
        with pytest.raises(ValueError, match="has no initializers"):
            _SwapperModel('/fake/path.onnx')


def test_get_face_swapper_model_no_outputs():
    """Should raise ValueError for model with no outputs."""
    import onnx
    from onnx import helper, TensorProto

    initializer = helper.make_tensor(
        name="emap", data_type=TensorProto.FLOAT,
        dims=[512, 512], vals=np.eye(512, dtype=np.float32).flatten().tolist(),
    )
    graph = helper.make_graph(
        [], "test",
        [
            helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 128, 128]),
            helper.make_tensor_value_info("latent", TensorProto.FLOAT, [1, 512]),
        ],
        [],
        [initializer],
    )
    model = helper.make_model(graph)

    with (
        patch('os.path.exists', return_value=True),
        patch('face_swapper.onnx.load', return_value=model),
        patch('face_swapper.onnxruntime.InferenceSession') as mock_session_cls,
    ):
        inp1, inp2 = Mock(), Mock()
        inp1.name, inp1.shape = "input", [1, 3, 128, 128]
        inp2.name, inp2.shape = "latent", [1, 512]
        mock_session_cls.return_value.get_inputs.return_value = [inp1, inp2]
        mock_session_cls.return_value.get_outputs.return_value = []

        with pytest.raises(ValueError, match="has no outputs"):
            _SwapperModel('/fake/path.onnx')


def test_swapper_model_inference_session_failure():
    """Should raise ValueError when InferenceSession fails."""
    onnx_model = _make_mock_onnx_model()

    with (
        patch('os.path.exists', return_value=True),
        patch('face_swapper.onnx.load', return_value=onnx_model),
        patch('face_swapper.onnxruntime.InferenceSession',
              side_effect=RuntimeError("GPU not available")),
    ):
        with pytest.raises(ValueError, match="Failed to create inference session"):
            _SwapperModel('/fake/path.onnx')


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


# --- swap_face_enhanced tests ---


def test_swap_face_enhanced_rejects_non_swapper_model():
    """swap_face_enhanced should raise TypeError for model without session."""
    from face_swapper import swap_face_enhanced

    # Use a plain object (not Mock) so hasattr(model, 'session') is False
    class BadModel:
        pass

    bad_model = BadModel()
    temp_frame = np.zeros((100, 100, 3), dtype=np.uint8)

    with pytest.raises(TypeError, match="Expected face swapper model"):
        swap_face_enhanced(
            Mock(), Mock(), temp_frame, bad_model,
            'inswapper_128', (128, 128), weight=1.0,
        )


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
    mock_model.session = Mock()
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


def test_swap_face_enhanced_inference_failure():
    """swap_face_enhanced should raise RuntimeError on ONNX failure."""
    from face_swapper import swap_face_enhanced

    mock_model = Mock()
    mock_model.emap = np.eye(512, dtype=np.float32)
    mock_model.input_names = ["input", "latent"]
    mock_model.output_names = ["output"]
    mock_model.session = Mock()
    mock_model.session.run = Mock(side_effect=RuntimeError("CUDA error"))

    temp_frame = np.ones((128, 128, 3), dtype=np.uint8)

    with (
        patch('face_swapper.face_align.norm_crop2') as mock_norm,
        patch('face_swapper.cv2.resize'),
        patch('face_swapper._prepare_crop_frame'),
    ):
        mock_norm.return_value = (
            np.ones((128, 128, 3), dtype=np.uint8),
            np.eye(2, 3, dtype=np.float32),
        )

        mock_source = Mock()
        mock_source.normed_embedding = np.ones((1, 512), dtype=np.float32)
        mock_target = Mock()
        mock_target.kps = np.random.randn(5, 2).astype(np.float32)

        with pytest.raises(RuntimeError, match="ONNX inference failed"):
            swap_face_enhanced(
                mock_source, mock_target, temp_frame, mock_model,
                'simswap_256', (256, 256), weight=1.0,
            )


def test_swap_face_enhanced_unknown_model():
    """swap_face_enhanced should raise ValueError for unknown model."""
    from face_swapper import swap_face_enhanced

    mock_model = Mock()
    mock_model.emap = np.eye(512, dtype=np.float32)
    mock_model.session = Mock()

    temp_frame = np.ones((128, 128, 3), dtype=np.uint8)

    mock_source = Mock()
    mock_source.normed_embedding = np.ones((1, 512), dtype=np.float32)
    mock_target = Mock()
    mock_target.kps = np.random.randn(5, 2).astype(np.float32)

    with pytest.raises(ValueError, match="Unknown face swapper model"):
        swap_face_enhanced(
            mock_source, mock_target, temp_frame, mock_model,
            'unknown_model_xyz', (128, 128), weight=1.0,
        )


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
    mock_model.session = Mock()
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
