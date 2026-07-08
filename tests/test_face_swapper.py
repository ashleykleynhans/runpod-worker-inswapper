# tests/test_face_swapper.py
import pytest
import numpy as np
from unittest.mock import patch, Mock, MagicMock
from face_swapper import (
    EMBEDDING_CONVERTERS,
    FACE_SWAPPER_MODELS,
    _SwapperModel,
    get_face_swapper_model,
    swap_face_enhanced,
    _prepare_crop_frame,
    _normalize_crop_frame,
    _prepare_embedding_projected,
    _prepare_embedding_raw,
    _prepare_source_face,
    _balance_embedding,
    _load_embedding_converter,
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
    """Corrupt ONNX gracefully falls back to identity emap, doesn't crash."""
    with (
        patch('os.path.exists', return_value=True),
        patch('face_swapper.onnxruntime.InferenceSession') as mock_cls,
    ):
        inp1, inp2 = Mock(), Mock()
        inp1.name, inp1.shape = "target", [1, 3, 128, 128]
        inp2.name, inp2.shape = "source", [1, 512]
        mock_cls.return_value.get_inputs.return_value = [inp1, inp2]
        out = Mock()
        out.name = "output"
        mock_cls.return_value.get_outputs.return_value = [out]

        with patch('face_swapper.onnx.load', side_effect=ValueError("bad file")):
            # Falls back to identity emap — should not raise
            model = _SwapperModel('/fake/path.onnx')
            assert model.emap.shape == (1, 1)


def test_get_face_swapper_model_single_input():
    """Should raise ValueError for model with only 1 input."""
    with (
        patch('os.path.exists', return_value=True),
        patch('face_swapper.onnxruntime.InferenceSession') as mock_cls,
    ):
        inp1 = Mock()
        inp1.name, inp1.shape = "input", [1, 3, 128, 128]
        mock_cls.return_value.get_inputs.return_value = [inp1]
        out = Mock()
        out.name = "output"
        mock_cls.return_value.get_outputs.return_value = [out]

        with pytest.raises(ValueError, match="expected at least 2"):
            _SwapperModel('/fake/path.onnx')


def test_get_face_swapper_model_no_initializers_ok():
    """Model with no initializers falls back to identity emap."""
    with (
        patch('os.path.exists', return_value=True),
        patch('face_swapper.onnxruntime.InferenceSession') as mock_cls,
    ):
        inp1, inp2 = Mock(), Mock()
        inp1.name, inp1.shape = "target", [1, 3, 128, 128]
        inp2.name, inp2.shape = "source", [1, 512]
        mock_cls.return_value.get_inputs.return_value = [inp1, inp2]
        out = Mock()
        out.name = "output"
        mock_cls.return_value.get_outputs.return_value = [out]

        with patch('face_swapper.onnx.load', return_value=_make_mock_onnx_model()):
            model = _SwapperModel('/fake/path.onnx')
            assert model.emap.shape == (512, 512)


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
    with (
        patch('os.path.exists', return_value=True),
        patch('face_swapper.onnxruntime.InferenceSession') as mock_cls,
    ):
        mock_cls.side_effect = RuntimeError("GPU not available")
        with pytest.raises(RuntimeError, match="GPU not available"):
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


def test_prepare_source_projected():
    """Embedding should be projected through emap (norm BEFORE dot, FF style)."""
    mock_model = Mock()
    mock_model.emap = np.eye(512, dtype=np.float32)
    mock_source = Mock()
    mock_source.normed_embedding = np.ones((1, 512), dtype=np.float32)

    latent = _prepare_embedding_projected(mock_source, mock_model)
    assert latent.shape == (1, 512)
    # normed_embedding is L2-normed (= 1/sqrt(512) per dim), dot with eye = same,
    # then divided by norm(source) = 1. So √512 * 1/√512 = 1 per dim.
    assert abs(np.linalg.norm(latent) - 1.0) < 0.001


def test_balance_embedding_full_swap():
    """At weight=1.0, source is slightly amplified by anti-mixing target."""
    src = np.ones((1, 512), dtype=np.float32) * 2.0
    tgt = -np.ones((1, 512), dtype=np.float32)
    balanced = _balance_embedding(src, tgt, 1.0)
    # w = -0.35, tgt_l2 = sqrt(512) ≈ 22.6
    # balanced = src*(1 - (-0.35)) + (tgt/||tgt||)*(-0.35)
    #          = 2.0*1.35 + (negated_ones_norm)*(-0.35) = 2.70 + 0.35/sqrt(512)
    expected = 2.0 * 1.35 + 0.35 / np.sqrt(512)
    assert abs(balanced[0, 0] - expected) < 0.01


def test_balance_embedding_neutral():
    """At weight=0.5, w=0, result = source only."""
    src = np.ones((1, 512), dtype=np.float32)
    tgt = np.ones((1, 512), dtype=np.float32)
    balanced = _balance_embedding(src, tgt, 0.5)
    np.testing.assert_array_almost_equal(balanced, src)


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


    # Use a plain object (not Mock) so hasattr(model, 'session') is False
    class BadModel:
        pass

    bad_model = BadModel()
    temp_frame = np.zeros((100, 100, 3), dtype=np.uint8)

    with pytest.raises(TypeError, match="Expected swapper model"):
        swap_face_enhanced(
            Mock(), Mock(), temp_frame, bad_model,
            'inswapper_128', (128, 128), weight=1.0,
        )


def test_swap_face_enhanced_full_pipeline():
    """Full swap_face_enhanced pipeline (inswapper) with mocked deps."""


    mock_source = Mock()
    mock_source.normed_embedding = np.ones((1, 512), dtype=np.float32)

    mock_target = Mock()
    mock_target.kps = np.random.randn(5, 2).astype(np.float32)
    mock_target.normed_embedding = np.ones((1, 512), dtype=np.float32)

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
        patch('face_swapper._prepare_embedding_projected') as mock_emb,
        patch('face_swapper._balance_embedding') as mock_bal,
    ):
        mock_norm.return_value = (
            np.ones((512, 512, 3), dtype=np.uint8),
            np.eye(2, 3, dtype=np.float32),
        )
        mock_resize.return_value = np.ones((128, 128, 3), dtype=np.uint8)
        mock_prep.return_value = np.ones((1, 3, 128, 128), dtype=np.float32)
        mock_emb.return_value = np.ones((1, 512), dtype=np.float32)
        mock_bal.return_value = np.ones((1, 512), dtype=np.float32)
        mock_normcrop.return_value = np.ones((128, 128, 3), dtype=np.uint8)
        mock_paste.return_value = temp_frame.copy()

        result = swap_face_enhanced(
            mock_source, mock_target, temp_frame, mock_model,
            'inswapper_128', resolution, weight=1.0,
        )

        mock_norm.assert_called_once()
        mock_resize.assert_called()
        mock_prep.assert_called_once()
        mock_model.session.run.assert_called_once()
        mock_normcrop.assert_called_once()
        mock_paste.assert_called_once()
        assert result is not None


def test_swap_face_enhanced_inference_failure():
    """swap_face_enhanced should raise RuntimeError on ONNX failure."""


    mock_model = Mock()
    mock_model.input_names = ["input", "latent"]
    mock_model.output_names = ["output"]
    mock_model.session = Mock()
    mock_model.session.run = Mock(side_effect=RuntimeError("CUDA error"))

    temp_frame = np.ones((128, 128, 3), dtype=np.uint8)

    mock_source = Mock()
    mock_source.normed_embedding = np.ones((1, 512), dtype=np.float32)
    mock_target = Mock()
    mock_target.kps = np.random.randn(5, 2).astype(np.float32)
    mock_target.normed_embedding = np.ones((1, 512), dtype=np.float32)

    with (
        patch('face_swapper.face_align.norm_crop2') as mock_norm,
        patch('face_swapper.cv2.resize'),
        patch('face_swapper._prepare_crop_frame'),
        patch('face_swapper._prepare_embedding_projected') as mock_emb,
        patch('face_swapper._balance_embedding') as mock_bal,
    ):
        mock_norm.return_value = (
            np.ones((128, 128, 3), dtype=np.uint8),
            np.eye(2, 3, dtype=np.float32),
        )
        mock_emb.return_value = np.ones((1, 512), dtype=np.float32)
        mock_bal.return_value = np.ones((1, 512), dtype=np.float32)

        with pytest.raises(RuntimeError, match="ONNX inference failed"):
            swap_face_enhanced(
                mock_source, mock_target, temp_frame, mock_model,
                'inswapper_128', (128, 128), weight=1.0,
            )


def test_swap_face_enhanced_unknown_model():
    """swap_face_enhanced should raise ValueError for unknown model."""

    mock_model = Mock()
    mock_model.emap = np.eye(512, dtype=np.float32)
    mock_model.session = Mock()
    temp_frame = np.ones((128, 128, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="Unknown model"):
        swap_face_enhanced(
            Mock(), Mock(), temp_frame, mock_model,
            'unknown_model_xyz', (128, 128), weight=1.0,
        )


def test_swap_face_enhanced_with_weight_blending_source_face():
    """Source_face models (blendswap/uniface) blend via cv2.addWeighted."""


    mock_source = Mock()
    mock_source.kps = np.random.randn(5, 2).astype(np.float32)
    mock_target = Mock()
    mock_target.kps = np.random.randn(5, 2).astype(np.float32)

    temp_frame = np.ones((256, 256, 3), dtype=np.uint8) * 100

    mock_model = Mock()
    mock_model.input_names = ["target", "source"]
    mock_model.output_names = ["output"]
    mock_model.session = Mock()
    mock_model.session.run = Mock(return_value=[
        np.zeros((1, 3, 256, 256), dtype=np.float32)
    ])

    with (
        patch('face_swapper.face_align.norm_crop2') as mock_norm,
        patch('face_swapper.cv2.resize'),
        patch('face_swapper._prepare_crop_frame'),
        patch('face_swapper._normalize_crop_frame'),
        patch('face_swapper._prepare_source_face') as mock_src,
        patch('face_swapper._paste_back') as mock_paste,
        patch('face_swapper.cv2.addWeighted') as mock_blend,
    ):
        mock_norm.return_value = (
            np.ones((256, 256, 3), dtype=np.uint8),
            np.eye(2, 3, dtype=np.float32),
        )
        mock_src.return_value = np.ones((1, 3, 112, 112), dtype=np.float32)
        mock_paste.return_value = temp_frame.copy()
        mock_blend.return_value = temp_frame.copy()

        swap_face_enhanced(
            mock_source, mock_target, temp_frame, mock_model,
            'blendswap_256', (256, 256), weight=0.7,
        )
        mock_blend.assert_called_once()


# --- Embedding converter and raw source tests ---


def test_load_embedding_converter_file_not_found():
    """Loading a missing converter should raise FileNotFoundError."""
    EMBEDDING_CONVERTERS.clear()
    with patch('os.path.exists', return_value=False):
        with pytest.raises(FileNotFoundError, match="Converter not found"):
            _load_embedding_converter("nonexistent.onnx")


def test_load_embedding_converter_loads():
    """Loading a converter should cache and return InferenceSession."""
    EMBEDDING_CONVERTERS.clear()
    with (
        patch('os.path.exists', return_value=True),
        patch('face_swapper.onnxruntime.InferenceSession') as mock_sess,
    ):
        mock_sess.return_value = "fake_session"
        s1 = _load_embedding_converter("crossface_ghost.onnx")
        s2 = _load_embedding_converter("crossface_ghost.onnx")
        assert s1 == s2 == "fake_session"
        assert mock_sess.call_count == 1  # cached


def test_prepare_embedding_raw():
    """Raw embedding goes through crossface ONNX converter and L2-norm."""
    EMBEDDING_CONVERTERS.clear()
    mock_source = Mock()
    mock_source.normed_embedding = np.ones((1, 512), dtype=np.float32)

    with (
        patch('os.path.exists', return_value=True),
        patch('face_swapper.onnxruntime.InferenceSession') as mock_sess,
    ):
        mock_conv = Mock()
        mock_conv.run.return_value = [np.ones((1, 512), dtype=np.float32) * 2.0]
        mock_sess.return_value = mock_conv

        result = _prepare_embedding_raw(mock_source, "crossface_ghost.onnx")
        assert result.shape == (1, 512)
        assert abs(np.linalg.norm(result) - 1.0) < 0.001


def test_prepare_source_face_direct():
    """Source_face prep: warp face to template, BGR->RGB, /255, CHW + batch."""
    mock_source = Mock()
    mock_source.kps = np.random.randn(5, 2).astype(np.float32)
    temp_frame = np.ones((256, 256, 3), dtype=np.uint8) * 128

    with patch('face_swapper.face_align.norm_crop2') as mock_norm:
        mock_norm.return_value = (
            np.ones((112, 112, 3), dtype=np.uint8) * 128,
            np.eye(2, 3, dtype=np.float32),
        )
        result = _prepare_source_face(mock_source, temp_frame, 112)
        assert result.shape == (1, 3, 112, 112)
        assert result.dtype == np.float32
        assert abs(result.mean() - 128.0 / 255.0) < 0.05


def test_swap_face_enhanced_embedding_model():
    """Embedding-type model (simswap) goes through embedding_raw + balance."""
    mock_source = Mock()
    mock_source.normed_embedding = np.ones((1, 512), dtype=np.float32)
    mock_target = Mock()
    mock_target.kps = np.random.randn(5, 2).astype(np.float32)
    mock_target.normed_embedding = np.ones((1, 512), dtype=np.float32)
    temp_frame = np.ones((256, 256, 3), dtype=np.uint8) * 100

    mock_model = Mock()
    mock_model.input_names = ["target", "source"]
    mock_model.output_names = ["output"]
    mock_model.session = Mock()
    mock_model.session.run = Mock(return_value=[
        np.zeros((1, 3, 256, 256), dtype=np.float32)
    ])

    with (
        patch('face_swapper.face_align.norm_crop2') as mock_norm,
        patch('face_swapper.cv2.resize'),
        patch('face_swapper._prepare_crop_frame'),
        patch('face_swapper._prepare_embedding_raw') as mock_raw,
        patch('face_swapper._balance_embedding') as mock_bal,
        patch('face_swapper._normalize_crop_frame'),
        patch('face_swapper._paste_back') as mock_paste,
    ):
        mock_norm.return_value = (
            np.ones((256, 256, 3), dtype=np.uint8),
            np.eye(2, 3, dtype=np.float32),
        )
        mock_raw.return_value = np.ones((1, 512), dtype=np.float32)
        mock_bal.return_value = np.ones((1, 512), dtype=np.float32)
        mock_paste.return_value = temp_frame.copy()

        result = swap_face_enhanced(
            mock_source, mock_target, temp_frame, mock_model,
            'simswap_256', (256, 256), weight=1.0,
        )
        mock_raw.assert_called_once()
        mock_bal.assert_called_once()
        assert result is not None
