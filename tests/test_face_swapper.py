# tests/test_face_swapper.py
import pytest
import numpy as np
from unittest.mock import patch, Mock

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
    _create_box_mask,
    _calculate_paste_area,
    _transform_points,
    _warp_face_by_landmark_5,
    WARP_TEMPLATES,
)


def _make_mock_onnx_model():
    import onnx
    from onnx import helper, TensorProto

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
    return helper.make_model(graph)


# --- Model loading tests ---


def test_get_face_swapper_model_loads():
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


def test_get_face_swapper_model_caches():
    FACE_SWAPPER_MODELS.clear()
    onnx_model = _make_mock_onnx_model()

    with (
        patch('os.path.exists', return_value=True),
        patch('face_swapper.onnx.load', return_value=onnx_model),
        patch('face_swapper.onnxruntime.InferenceSession') as mock_cls,
    ):
        mock_cls.return_value.get_inputs.return_value = [
            Mock(name="input", shape=[1, 3, 256, 256]),
            Mock(name="latent", shape=[1, 512]),
        ]
        mock_cls.return_value.get_outputs.return_value = [Mock(name="output")]

        model1 = get_face_swapper_model('simswap_256')
        model2 = get_face_swapper_model('simswap_256')
        assert model1 is model2


def test_get_face_swapper_model_missing_file():
    with pytest.raises(FileNotFoundError, match="Model file not found"):
        _SwapperModel('/nonexistent/path.onnx')


def test_get_face_swapper_model_corrupt_onnx():
    with (
        patch('os.path.exists', return_value=True),
        patch('face_swapper.onnxruntime.InferenceSession') as mock_cls,
    ):
        inp1 = Mock(name="input", shape=[1, 3, 128, 128])
        inp2 = Mock(name="latent", shape=[1, 512])
        mock_cls.return_value.get_inputs.return_value = [inp1, inp2]
        mock_cls.return_value.get_outputs.return_value = [Mock(name="output")]

        with patch('face_swapper.onnx.load', side_effect=ValueError("bad file")):
            model = _SwapperModel('/fake/path.onnx')
            assert model.emap.shape == (1, 1)


def test_get_face_swapper_model_single_input():
    with (
        patch('os.path.exists', return_value=True),
        patch('face_swapper.onnxruntime.InferenceSession') as mock_cls,
    ):
        mock_cls.return_value.get_inputs.return_value = [
            Mock(name="input", shape=[1, 3, 128, 128])
        ]
        mock_cls.return_value.get_outputs.return_value = [Mock(name="output")]
        with pytest.raises(ValueError, match="expected at least 2"):
            _SwapperModel('/fake/path.onnx')


def test_get_face_swapper_model_no_initializers_ok():
    with (
        patch('os.path.exists', return_value=True),
        patch('face_swapper.onnxruntime.InferenceSession') as mock_cls,
    ):
        inp1 = Mock(name="input", shape=[1, 3, 128, 128])
        inp2 = Mock(name="latent", shape=[1, 512])
        mock_cls.return_value.get_inputs.return_value = [inp1, inp2]
        mock_cls.return_value.get_outputs.return_value = [Mock(name="output")]
        with patch('face_swapper.onnx.load', return_value=_make_mock_onnx_model()):
            model = _SwapperModel('/fake/path.onnx')
            assert model.emap.shape == (512, 512)


def test_get_face_swapper_model_no_outputs():
    with (
        patch('os.path.exists', return_value=True),
        patch('face_swapper.onnxruntime.InferenceSession') as mock_cls,
    ):
        inp1 = Mock(name="input", shape=[1, 3, 128, 128])
        inp2 = Mock(name="latent", shape=[1, 512])
        mock_cls.return_value.get_inputs.return_value = [inp1, inp2]
        mock_cls.return_value.get_outputs.return_value = []
        with pytest.raises(ValueError, match="has no outputs"):
            _SwapperModel('/fake/path.onnx')


def test_swapper_model_inference_session_failure():
    with (
        patch('os.path.exists', return_value=True),
        patch('face_swapper.onnxruntime.InferenceSession') as mock_cls,
    ):
        mock_cls.side_effect = RuntimeError("GPU not available")
        with pytest.raises(RuntimeError, match="GPU not available"):
            _SwapperModel('/fake/path.onnx')


# --- Warp tests (FaceFusion-forked) ---


def test_warp_face_by_landmark_5():
    """Warp uses cv2.estimateAffinePartial2D + BORDER_REPLICATE."""
    frame = np.ones((256, 256, 3), dtype=np.uint8) * 128
    lm_5 = np.random.randn(5, 2).astype(np.float32)
    cropped, M = _warp_face_by_landmark_5(frame, lm_5, "arcface_128", (128, 128))
    assert cropped.shape == (128, 128, 3)
    assert M.shape == (2, 3)


def test_warp_face_ransac_fallback():
    """When RANSAC returns None, fall back to insightface norm_crop2."""
    frame = np.ones((256, 256, 3), dtype=np.uint8)
    lm_5 = np.zeros((5, 2), dtype=np.float32)  # degenerate points
    # All-zero landmark points cause RANSAC to fail
    # but the function should not crash
    result = _warp_face_by_landmark_5(frame, lm_5, "arcface_128", (128, 128))
    assert result[0].shape == (128, 128, 3)


# --- Preprocessing tests ---


def test_prepare_crop_frame_inswapper():
    dummy = np.full((128, 128, 3), 128, dtype=np.uint8)
    blob = _prepare_crop_frame(dummy, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    assert blob.shape == (1, 3, 128, 128)
    assert blob.dtype == np.float32
    assert np.allclose(blob, 128.0 / 255.0, atol=0.01)


def test_prepare_crop_frame_simswap():
    dummy = np.zeros((256, 256, 3), dtype=np.uint8)
    blob = _prepare_crop_frame(dummy, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    assert blob.shape == (1, 3, 256, 256)
    assert np.all(blob < 0)


def test_normalize_crop_frame_nontanh():
    pred = np.random.randn(1, 3, 128, 128).astype(np.float32) * 0.3 + 0.5
    out = _normalize_crop_frame(pred, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], tanh_out=False)
    assert out.shape == (128, 128, 3)
    assert out.dtype == np.uint8
    assert out.min() >= 0 and out.max() <= 255


def test_normalize_crop_frame_tanh():
    pred = np.zeros((1, 3, 256, 256), dtype=np.float32)
    out = _normalize_crop_frame(pred, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], tanh_out=True)
    assert out.shape == (256, 256, 3)
    assert np.allclose(out.astype(float), 127.5, atol=1.0)


def test_prepare_source_projected():
    mock_model = Mock()
    mock_model.emap = np.eye(512, dtype=np.float32)
    mock_source = Mock()
    mock_source.normed_embedding = np.ones((1, 512), dtype=np.float32)
    latent = _prepare_embedding_projected(mock_source, mock_model)
    assert latent.shape == (1, 512)
    assert abs(np.linalg.norm(latent) - 1.0) < 0.001


def test_balance_embedding_full_swap():
    src = np.ones((1, 512), dtype=np.float32) * 2.0
    tgt = -np.ones((1, 512), dtype=np.float32)
    balanced = _balance_embedding(src, tgt, 1.0)
    expected = 2.0 * 1.35 + 0.35 / np.sqrt(512)
    assert abs(balanced[0, 0] - expected) < 0.01


def test_balance_embedding_neutral():
    src = np.ones((1, 512), dtype=np.float32)
    tgt = np.ones((1, 512), dtype=np.float32)
    balanced = _balance_embedding(src, tgt, 0.5)
    np.testing.assert_array_almost_equal(balanced, src)


def test_load_embedding_converter_file_not_found():
    EMBEDDING_CONVERTERS.clear()
    with patch('os.path.exists', return_value=False):
        with pytest.raises(FileNotFoundError, match="Converter not found"):
            _load_embedding_converter("nonexistent.onnx")


def test_load_embedding_converter_caches():
    EMBEDDING_CONVERTERS.clear()
    with (
        patch('os.path.exists', return_value=True),
        patch('face_swapper.onnxruntime.InferenceSession') as mock_sess,
    ):
        mock_sess.return_value = "fake"
        s1 = _load_embedding_converter("crossface_ghost.onnx")
        s2 = _load_embedding_converter("crossface_ghost.onnx")
        assert s1 == s2 == "fake"
        assert mock_sess.call_count == 1


def test_prepare_embedding_raw():
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
    mock_source = Mock()
    mock_source.kps = np.random.randn(5, 2).astype(np.float32)
    temp_frame = np.ones((256, 256, 3), dtype=np.uint8) * 128
    with patch('face_swapper.face_align.norm_crop2') as mock_norm:
        mock_norm.return_value = (np.ones((112, 112, 3), dtype=np.uint8) * 128, np.eye(2, 3))
        result = _prepare_source_face(mock_source, temp_frame, 112)
        assert result.shape == (1, 3, 112, 112)
        assert result.dtype == np.float32


# --- Paste-back tests (FaceFusion-forked) ---


def test_create_box_mask_default():
    mask = _create_box_mask((256, 256))
    assert mask.shape == (256, 256)
    assert mask.dtype == np.float32
    assert mask.max() <= 1.0
    assert mask.min() >= 0.0


def test_transform_points():
    pts = np.array([[0, 0], [10, 10], [5, 0]], dtype=np.float32)
    M = np.array([[1, 0, 5], [0, 1, 5]], dtype=np.float32)
    result = _transform_points(pts, M)
    np.testing.assert_array_almost_equal(result, pts + 5)


def test_calculate_paste_area():
    frame = np.zeros((256, 256, 3), dtype=np.uint8)
    crop = np.zeros((128, 128, 3), dtype=np.uint8)
    M = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    bbox, paste_m = _calculate_paste_area(frame, crop, M)
    assert bbox.shape == (4,)
    assert bbox[0] >= 0 and bbox[2] <= 256


def test_paste_back():
    frame = np.ones((256, 256, 3), dtype=np.uint8) * 100
    crop = np.ones((128, 128, 3), dtype=np.uint8) * 200
    mask = np.ones((128, 128), dtype=np.float32)  # full mask = full blend
    M = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    result = _paste_back(frame, crop, mask, M)
    assert result.shape == (256, 256, 3)
    assert result.dtype == np.uint8
    # Top-left should be blended (not original 100)
    assert not np.allclose(result[:50, :50], 100)


def test_paste_back_empty_paste_area():
    """Off-screen crop produces zero-area paste region, returns original."""
    frame = np.ones((256, 256, 3), dtype=np.uint8) * 100
    crop = np.ones((128, 128, 3), dtype=np.uint8) * 200
    mask = np.ones((128, 128), dtype=np.float32)
    M = np.array([[1, 0, 500], [0, 1, 500]], dtype=np.float32)  # off-screen
    result = _paste_back(frame, crop, mask, M)
    assert np.array_equal(result, frame)


# --- swap_face_enhanced tests ---


def test_swap_face_enhanced_rejects_bad_model():
    class BadModel:
        pass

    bad_model = BadModel()
    temp_frame = np.zeros((100, 100, 3), dtype=np.uint8)

    with pytest.raises(TypeError, match="Expected swapper model"):
        swap_face_enhanced(Mock(), Mock(), temp_frame, bad_model, 'inswapper_128', (128, 128))


def test_swap_face_enhanced_full_pipeline():
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

    with (
        patch('face_swapper._warp_face_by_landmark_5') as mock_warp,
        patch('face_swapper._create_box_mask'),
        patch('face_swapper.cv2.resize'),
        patch('face_swapper._prepare_crop_frame'),
        patch('face_swapper._prepare_embedding_projected') as mock_emb,
        patch('face_swapper._balance_embedding') as mock_bal,
        patch('face_swapper._normalize_crop_frame'),
        patch('face_swapper._paste_back') as mock_paste,
    ):
        mock_warp.return_value = (
            np.ones((512, 512, 3), dtype=np.uint8),
            np.eye(2, 3, dtype=np.float32),
        )
        mock_emb.return_value = np.ones((1, 512), dtype=np.float32)
        mock_bal.return_value = np.ones((1, 512), dtype=np.float32)
        mock_paste.return_value = temp_frame.copy()

        result = swap_face_enhanced(
            mock_source, mock_target, temp_frame, mock_model,
            'inswapper_128', (512, 512), weight=1.0,
        )
        mock_warp.assert_called_once()
        mock_emb.assert_called_once()
        mock_bal.assert_called_once()
        mock_model.session.run.assert_called_once()
        mock_paste.assert_called_once()
        assert result is not None


def test_swap_face_enhanced_inference_failure():
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
        patch('face_swapper._warp_face_by_landmark_5') as mock_warp,
        patch('face_swapper._create_box_mask'),
        patch('face_swapper.cv2.resize'),
        patch('face_swapper._prepare_crop_frame'),
        patch('face_swapper._prepare_embedding_projected') as mock_emb,
        patch('face_swapper._balance_embedding') as mock_bal,
    ):
        mock_warp.return_value = (
            np.ones((128, 128, 3), dtype=np.uint8),
            np.eye(2, 3, dtype=np.float32),
        )
        mock_emb.return_value = np.ones((1, 512), dtype=np.float32)
        mock_bal.return_value = np.ones((1, 512), dtype=np.float32)

        with pytest.raises(RuntimeError, match="ONNX inference failed"):
            swap_face_enhanced(mock_source, mock_target, temp_frame, mock_model,
                               'inswapper_128', (128, 128))


def test_swap_face_enhanced_unknown_model():
    mock_model = Mock()
    mock_model.emap = np.eye(512, dtype=np.float32)
    mock_model.session = Mock()
    temp_frame = np.ones((128, 128, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="Unknown model"):
        swap_face_enhanced(Mock(), Mock(), temp_frame, mock_model, 'unknown', (128, 128))


def test_swap_face_enhanced_source_face_blend():
    mock_source = Mock()
    mock_source.kps = np.random.randn(5, 2).astype(np.float32)
    mock_target = Mock()
    mock_target.kps = np.random.randn(5, 2).astype(np.float32)

    temp_frame = np.ones((256, 256, 3), dtype=np.uint8) * 100
    mock_model = Mock()
    mock_model.input_names = ["target", "source"]
    mock_model.output_names = ["output"]
    mock_model.session = Mock()
    mock_model.session.run = Mock(return_value=[np.zeros((1, 3, 256, 256), dtype=np.float32)])

    with (
        patch('face_swapper._warp_face_by_landmark_5') as mock_warp,
        patch('face_swapper._create_box_mask'),
        patch('face_swapper.cv2.resize'),
        patch('face_swapper._prepare_crop_frame'),
        patch('face_swapper._prepare_source_face') as mock_src,
        patch('face_swapper._normalize_crop_frame'),
        patch('face_swapper._paste_back') as mock_paste,
        patch('face_swapper.cv2.addWeighted') as mock_blend,
    ):
        mock_warp.return_value = (
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


def test_swap_face_enhanced_embedding_model():
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
    mock_model.session.run = Mock(return_value=[np.zeros((1, 3, 256, 256), dtype=np.float32)])

    with (
        patch('face_swapper._warp_face_by_landmark_5') as mock_warp,
        patch('face_swapper._create_box_mask'),
        patch('face_swapper.cv2.resize'),
        patch('face_swapper._prepare_crop_frame'),
        patch('face_swapper._prepare_embedding_raw') as mock_raw,
        patch('face_swapper._balance_embedding') as mock_bal,
        patch('face_swapper._normalize_crop_frame'),
        patch('face_swapper._paste_back') as mock_paste,
    ):
        mock_warp.return_value = (
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
