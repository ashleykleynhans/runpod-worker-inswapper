# tests/test_face_swapper.py
import pytest
import numpy as np
from unittest.mock import Mock, patch

from face_swapper import (
    EMBEDDING_CONVERTERS, FACE_SWAPPER_MODELS, WARP_TEMPLATES,
    _SwapperModel, get_face_swapper_model, swap_face_enhanced,
    _prepare_crop_frame, _normalize_crop_frame,
    _prepare_embedding_projected, _prepare_embedding_raw,
    _prepare_embedding_norm, _prepare_source_face, _balance_embedding,
    _load_embedding_converter, _paste_back, _create_box_mask,
    _calculate_paste_area, _transform_points, _warp_face_by_landmark_5,
)


def _make_onnx(shape0=(1, 3, 128, 128), shape1=(1, 512)):
    from onnx import helper, TensorProto
    n = helper.make_node("Conv", ["in0", "in1"], ["out"], name="n")
    in0 = helper.make_tensor_value_info("in0", TensorProto.FLOAT, shape0)
    in1 = helper.make_tensor_value_info("in1", TensorProto.FLOAT, shape1)
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 3, 128, 128])
    init = helper.make_tensor("emap", TensorProto.FLOAT, [512, 512],
                              np.eye(512, dtype=np.float32).flatten().tolist())
    return helper.make_model(helper.make_graph([n], "t", [in0, in1], [out], [init]))


def _mock_session(inputs=None, outputs=None):
    s = Mock()
    s.get_inputs.return_value = inputs or [
        Mock(name="in0", shape=[1, 3, 128, 128]),
        Mock(name="in1", shape=[1, 512]),
    ]
    s.get_outputs.return_value = outputs or [Mock(name="out")]
    return s


# --- Model loading ---


def test_get_face_swapper_model_loads():
    FACE_SWAPPER_MODELS.clear()
    with (
        patch('os.path.exists', return_value=True),
        patch('face_swapper.onnx.load', return_value=_make_onnx()),
        patch('face_swapper.onnxruntime.InferenceSession') as m,
    ):
        m.return_value = _mock_session()
        mdl = get_face_swapper_model('inswapper_128')
        assert isinstance(mdl, _SwapperModel)
        assert mdl.emap.shape == (512, 512)


def test_get_face_swapper_model_caches():
    FACE_SWAPPER_MODELS.clear()
    with (
        patch('os.path.exists', return_value=True),
        patch('face_swapper.onnx.load', return_value=_make_onnx((1, 3, 256, 256))),
        patch('face_swapper.onnxruntime.InferenceSession') as m,
    ):
        m.return_value = _mock_session()
        a = get_face_swapper_model('simswap_256')
        b = get_face_swapper_model('simswap_256')
        assert a is b


def test_missing_file():
    with pytest.raises(FileNotFoundError, match="not found"):
        _SwapperModel('/nonexistent.onnx')


def test_corrupt_onnx():
    with (
        patch('os.path.exists', return_value=True),
        patch('face_swapper.onnxruntime.InferenceSession') as m,
    ):
        m.return_value = _mock_session()
        with patch('face_swapper.onnx.load', side_effect=ValueError("bad")):
            m2 = _SwapperModel('/f.onnx')
            assert m2.emap.shape == (1, 1)


def test_single_input():
    with (
        patch('os.path.exists', return_value=True),
        patch('face_swapper.onnxruntime.InferenceSession') as m,
    ):
        m.return_value.get_inputs.return_value = [Mock(name="x", shape=[1, 3, 128, 128])]
        m.return_value.get_outputs.return_value = [Mock(name="y")]
        with pytest.raises(ValueError, match="expected at least 2"):
            _SwapperModel('/f.onnx')


def test_no_outputs():
    with (
        patch('os.path.exists', return_value=True),
        patch('face_swapper.onnxruntime.InferenceSession') as m,
    ):
        m.return_value = _mock_session()
        m.return_value.get_outputs.return_value = []
        with pytest.raises(ValueError, match="has no outputs"):
            _SwapperModel('/f.onnx')


def test_inference_session_failure():
    with (
        patch('os.path.exists', return_value=True),
        patch('face_swapper.onnxruntime.InferenceSession') as m,
    ):
        m.side_effect = RuntimeError("GPU gone")
        with pytest.raises(RuntimeError, match="GPU gone"):
            _SwapperModel('/f.onnx')


def test_hyperswap_detected_as_swapped():
    """Hyperswap reversed: [1,512] embedding then [1,3,256,256] image."""
    FACE_SWAPPER_MODELS.clear()
    with (
        patch('os.path.exists', return_value=True),
        patch('face_swapper.onnx.load', return_value=_make_onnx((1, 512), (1, 3, 256, 256))),
        patch('face_swapper.onnxruntime.InferenceSession') as m,
    ):
        s0, s1 = Mock(), Mock()
        s0.shape, s0.name = [1, 512], "source"
        s1.shape, s1.name = [1, 3, 256, 256], "target"
        m.return_value.get_inputs.return_value = [s0, s1]
        m.return_value.get_outputs.return_value = [Mock(name="out")]
        mdl = _SwapperModel('/hyperswap_1a_256.onnx')
        assert mdl.input_swapped is True


# --- Warp ---


def test_warp_face():
    frame = np.ones((256, 256, 3), dtype=np.uint8) * 128
    lm = np.random.randn(5, 2).astype(np.float32) * 10 + 128
    crop, M = _warp_face_by_landmark_5(frame, lm, "arcface_128", (128, 128))
    assert crop.shape == (128, 128, 3)
    assert M.shape == (2, 3)


def test_warp_ransac_fallback():
    lm = np.zeros((5, 2), dtype=np.float32)
    frame = np.ones((256, 256, 3), dtype=np.uint8)
    crop, M = _warp_face_by_landmark_5(frame, lm, "arcface_128", (128, 128))
    assert crop.shape == (128, 128, 3)


def test_warp_templates_exist():
    assert set(WARP_TEMPLATES) >= {"arcface_128", "arcface_112_v1", "ffhq_512", "mtcnn_512"}


# --- Preprocessing ---


def test_prepare_crop_inswapper():
    b = _prepare_crop_frame(np.full((128, 128, 3), 128, dtype=np.uint8), [0.]*3, [1.]*3)
    assert b.shape == (1, 3, 128, 128)
    assert np.allclose(b, 128/255, atol=0.01)


def test_prepare_crop_simswap():
    b = _prepare_crop_frame(np.zeros((256, 256, 3), dtype=np.uint8), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    assert np.all(b < 0)


def test_normalize_nontanh():
    p = np.random.randn(1, 3, 128, 128).astype(np.float32) * 0.3 + 0.5
    o = _normalize_crop_frame(p, [0.]*3, [1.]*3, False)
    assert o.dtype == np.uint8 and o.min() >= 0 and o.max() <= 255


def test_normalize_tanh():
    o = _normalize_crop_frame(np.zeros((1, 3, 256, 256), dtype=np.float32), [.5]*3, [.5]*3, True)
    assert np.allclose(o.astype(float), 127.5, atol=1.0)


# --- Source preparation ---


def test_embedding_projected():
    m = Mock(); m.emap = np.eye(512, dtype=np.float32)
    sf = Mock(); sf.embedding = np.ones(512, dtype=np.float32)
    o = _prepare_embedding_projected(sf, m)
    assert o.shape == (1, 512) and abs(np.linalg.norm(o) - 1) < 0.001


def test_embedding_raw():
    EMBEDDING_CONVERTERS.clear()
    sf = Mock(); sf.embedding = np.ones(512, dtype=np.float32)
    with (
        patch('os.path.exists', return_value=True),
        patch('face_swapper.onnxruntime.InferenceSession') as ms,
    ):
        mc = Mock()
        mc.run.return_value = [np.ones((1, 512), dtype=np.float32) * 2.0]
        ms.return_value = mc
        o = _prepare_embedding_raw(sf, "crossface_ghost.onnx")
        assert o.shape == (1, 512) and abs(np.linalg.norm(o) - 1) < 0.001


def test_embedding_norm():
    sf = Mock()
    sf.embedding_norm = np.ones(512, dtype=np.float32) / np.sqrt(512)
    o = _prepare_embedding_norm(sf)
    assert o.shape == (1, 512)
    # embedding_norm is already L2-normalized
    np.testing.assert_array_almost_equal(o, sf.embedding_norm.reshape(1, -1))


def test_source_face():
    sf = Mock(); sf.kps = np.random.randn(5, 2).astype(np.float32)
    with patch('face_swapper.face_align.norm_crop2') as mn:
        mn.return_value = (np.ones((112, 112, 3), dtype=np.uint8) * 128, np.eye(2, 3))
        o = _prepare_source_face(sf, np.ones((256, 256, 3), dtype=np.uint8), 112)
        assert o.shape == (1, 3, 112, 112)


def test_balance_full_swap():
    s = np.ones((1, 512), dtype=np.float32) * 2.0
    t = -np.ones(512, dtype=np.float32)
    b = _balance_embedding(s, t, 1.0)
    exp = 2.0 * 1.35 + 0.35 / np.sqrt(512)
    assert abs(b[0, 0] - exp) < 0.01


def test_balance_neutral():
    s = np.ones((1, 512), dtype=np.float32)
    t = np.ones(512, dtype=np.float32)
    np.testing.assert_array_almost_equal(_balance_embedding(s, t, 0.5), s)


# --- Paste-back ---


def test_box_mask():
    m = _create_box_mask((256, 256))
    assert m.shape == (256, 256) and 0 <= m.min() <= m.max() <= 1


def test_transform_points():
    np.testing.assert_array_almost_equal(
        _transform_points(np.array([[0, 0], [10, 10]], dtype=np.float32),
                          np.array([[1, 0, 5], [0, 1, 5]], dtype=np.float32)),
        np.array([[5, 5], [15, 15]]))


def test_calculate_paste_area():
    f = np.zeros((256, 256, 3), dtype=np.uint8)
    c = np.zeros((128, 128, 3), dtype=np.uint8)
    bbox, _ = _calculate_paste_area(f, c, np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32))
    assert bbox[0] >= 0 and bbox[2] <= 256


def test_paste_back():
    f = np.ones((256, 256, 3), dtype=np.uint8) * 100
    c = np.ones((128, 128, 3), dtype=np.uint8) * 200
    m = np.ones((128, 128), dtype=np.float32)
    r = _paste_back(f, c, m, np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32))
    assert r.shape == (256, 256, 3) and not np.allclose(r[:50, :50], 100)


def test_paste_back_empty():
    f = np.ones((256, 256, 3), dtype=np.uint8) * 100
    c = np.ones((128, 128, 3), dtype=np.uint8) * 200
    r = _paste_back(f, c, np.ones((128, 128)), np.array([[1, 0, 500], [0, 1, 500]], dtype=np.float32))
    np.testing.assert_array_equal(r, f)


# --- Converter ---


def test_converter_missing():
    EMBEDDING_CONVERTERS.clear()
    with patch('os.path.exists', return_value=False):
        with pytest.raises(FileNotFoundError, match="not found"):
            _load_embedding_converter("nope.onnx")


def test_converter_cached():
    EMBEDDING_CONVERTERS.clear()
    with (
        patch('os.path.exists', return_value=True),
        patch('face_swapper.onnxruntime.InferenceSession') as ms,
    ):
        ms.return_value = "ok"
        a = _load_embedding_converter("crossface_ghost.onnx")
        b = _load_embedding_converter("crossface_ghost.onnx")
        assert a == b and ms.call_count == 1


# --- swap_face_enhanced ---


def test_rejects_bad_model():
    class B:
        pass
    with pytest.raises(TypeError, match="Expected swapper"):
        swap_face_enhanced(Mock(), Mock(), np.zeros((100, 100, 3), dtype=np.uint8), B(), 'a', (128, 128))


def test_inswapper_pipeline():
    sf = Mock(); sf.embedding = np.ones(512, dtype=np.float32)
    tf = Mock(); tf.kps = np.random.randn(5, 2).astype(np.float32); tf.embedding = np.ones(512, dtype=np.float32)
    frame = np.ones((512, 512, 3), dtype=np.uint8) * 100
    mdl = Mock(); mdl.emap = np.eye(512); mdl.input_names = ["t", "s"]
    mdl.output_names = ["o"]; mdl.input_swapped = False
    mdl.session = Mock(); mdl.session.run = Mock(return_value=[np.random.randn(1, 3, 128, 128).astype(np.float32) * 0.2 + 0.5])
    with (
        patch('face_swapper._warp_face_by_landmark_5') as mw,
        patch('face_swapper._create_box_mask'), patch('face_swapper.cv2.resize'),
        patch('face_swapper._prepare_crop_frame'),
        patch('face_swapper._prepare_embedding_projected') as me,
        patch('face_swapper._balance_embedding') as mb,
        patch('face_swapper._normalize_crop_frame'), patch('face_swapper._paste_back') as mp,
    ):
        mw.return_value = (np.ones((512, 512, 3), dtype=np.uint8), np.eye(2, 3))
        me.return_value = np.ones((1, 512)); mb.return_value = np.ones((1, 512))
        mp.return_value = frame.copy()
        r = swap_face_enhanced(sf, tf, frame, mdl, 'inswapper_128', (512, 512))
        mw.assert_called_once(); me.assert_called_once(); mb.assert_called_once()
        mdl.session.run.assert_called_once(); mp.assert_called_once(); assert r is not None


def test_inference_failure():
    mdl = Mock(); mdl.session = Mock(); mdl.session.run = Mock(side_effect=RuntimeError("CUDA err"))
    mdl.input_names = ["t", "s"]; mdl.output_names = ["o"]; mdl.input_swapped = False
    sf = Mock(); sf.embedding = np.ones(512)
    tf = Mock(); tf.kps = np.random.randn(5, 2).astype(np.float32); tf.embedding = np.ones(512)
    with (
        patch('face_swapper._warp_face_by_landmark_5') as mw,
        patch('face_swapper._create_box_mask'), patch('face_swapper.cv2.resize'),
        patch('face_swapper._prepare_crop_frame'),
        patch('face_swapper._prepare_embedding_projected') as me,
        patch('face_swapper._balance_embedding') as mb,
    ):
        mw.return_value = (np.ones((128, 128, 3), dtype=np.uint8), np.eye(2, 3))
        me.return_value = np.ones((1, 512)); mb.return_value = np.ones((1, 512))
        with pytest.raises(RuntimeError, match="ONNX inference failed"):
            swap_face_enhanced(sf, tf, np.ones((128, 128, 3), dtype=np.uint8), mdl, 'inswapper_128', (128, 128))


def test_unknown_model():
    mdl = Mock(); mdl.session = Mock(); mdl.emap = np.eye(512)
    with pytest.raises(KeyError, match="not found"):
        swap_face_enhanced(Mock(), Mock(), np.ones((128, 128, 3), dtype=np.uint8), mdl, 'xyz', (128, 128))


def test_source_face_blend_weight():
    sf = Mock(); sf.kps = np.random.randn(5, 2).astype(np.float32)
    tf = Mock(); tf.kps = np.random.randn(5, 2).astype(np.float32)
    frame = np.ones((256, 256, 3), dtype=np.uint8) * 100
    mdl = Mock(); mdl.input_swapped = False
    mdl.input_names = ["t", "s"]; mdl.output_names = ["o"]
    mdl.session = Mock(); mdl.session.run = Mock(return_value=[np.zeros((1, 3, 256, 256), dtype=np.float32)])
    with (
        patch('face_swapper._warp_face_by_landmark_5') as mw,
        patch('face_swapper._create_box_mask'), patch('face_swapper.cv2.resize'),
        patch('face_swapper._prepare_crop_frame'),
        patch('face_swapper._prepare_source_face') as ms,
        patch('face_swapper._normalize_crop_frame'), patch('face_swapper._paste_back') as mp,
        patch('face_swapper.cv2.addWeighted') as mb,
    ):
        mw.return_value = (np.ones((256, 256, 3), dtype=np.uint8), np.eye(2, 3))
        ms.return_value = np.ones((1, 3, 112, 112), dtype=np.float32)
        mp.return_value = frame.copy(); mb.return_value = frame.copy()
        swap_face_enhanced(sf, tf, frame, mdl, 'blendswap_256', (256, 256), weight=0.7)
        mb.assert_called_once()


def test_embedding_model():
    sf = Mock(); sf.embedding = np.ones(512, dtype=np.float32)
    tf = Mock(); tf.kps = np.random.randn(5, 2).astype(np.float32); tf.embedding = np.ones(512, dtype=np.float32)
    frame = np.ones((256, 256, 3), dtype=np.uint8) * 100
    mdl = Mock(); mdl.input_swapped = False
    mdl.input_names = ["t", "s"]; mdl.output_names = ["o"]
    mdl.session = Mock(); mdl.session.run = Mock(return_value=[np.zeros((1, 3, 256, 256), dtype=np.float32)])
    with (
        patch('face_swapper._warp_face_by_landmark_5') as mw,
        patch('face_swapper._create_box_mask'), patch('face_swapper.cv2.resize'),
        patch('face_swapper._prepare_crop_frame'),
        patch('face_swapper._prepare_embedding_raw') as mr,
        patch('face_swapper._balance_embedding') as mb,
        patch('face_swapper._normalize_crop_frame'), patch('face_swapper._paste_back') as mp,
    ):
        mw.return_value = (np.ones((256, 256, 3), dtype=np.uint8), np.eye(2, 3))
        mr.return_value = np.ones((1, 512)); mb.return_value = np.ones((1, 512))
        mp.return_value = frame.copy()
        r = swap_face_enhanced(sf, tf, frame, mdl, 'simswap_256', (256, 256))
        mr.assert_called_once(); mb.assert_called_once(); assert r is not None


def test_hyperswap_swapped_inputs():
    """Hyperswap sends source to input[0] and target to input[1]."""
    sf = Mock(); sf.embedding_norm = np.ones(512, dtype=np.float32)
    tf = Mock(); tf.kps = np.random.randn(5, 2).astype(np.float32)
    tf.embedding = np.ones(512, dtype=np.float32)
    frame = np.ones((256, 256, 3), dtype=np.uint8) * 100
    mdl = Mock(); mdl.input_swapped = True
    mdl.input_names = ["source", "target"]; mdl.output_names = ["out"]
    mdl.session = Mock(); mdl.session.run = Mock(return_value=[np.zeros((1, 3, 256, 256), dtype=np.float32)])
    with (
        patch('face_swapper._warp_face_by_landmark_5') as mw,
        patch('face_swapper._create_box_mask'), patch('face_swapper.cv2.resize'),
        patch('face_swapper._prepare_crop_frame'),
        patch('face_swapper._prepare_embedding_norm') as mn,
        patch('face_swapper._balance_embedding') as mb,
        patch('face_swapper._normalize_crop_frame'), patch('face_swapper._paste_back') as mp,
    ):
        mw.return_value = (np.ones((256, 256, 3), dtype=np.uint8), np.eye(2, 3))
        mn.return_value = np.ones((1, 512)); mb.return_value = np.ones((1, 512))
        mp.return_value = frame.copy()
        r = swap_face_enhanced(sf, tf, frame, mdl, 'hyperswap_1a_256', (256, 256))
        call = mdl.session.run.call_args
        fd = call[1]['fd'] if 'fd' in call[1] else call[0][1]
        assert fd["source"] is not None and fd["target"] is not None
        assert r is not None
