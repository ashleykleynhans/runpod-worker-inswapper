import pytest
import base64
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from PIL import Image
import io
import cv2

# Import functions to test
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from handler import (
    get_face_swap_model,
    get_face_analyser,
    get_one_face,
    get_many_faces,
    swap_face,
    determine_file_extension,
    clean_up_temporary_files,
    handler
)


class TestGetFaceSwapModel:
    """Tests for get_face_swap_model function"""

    @patch('handler.insightface.model_zoo.get_model')
    def test_get_face_swap_model(self, mock_get_model):
        """Test that get_face_swap_model calls insightface correctly"""
        mock_model = Mock()
        mock_get_model.return_value = mock_model

        model_path = 'test_model.onnx'
        result = get_face_swap_model(model_path)

        mock_get_model.assert_called_once_with(model_path)
        assert result == mock_model


class TestGetFaceAnalyser:
    """Tests for get_face_analyser function"""

    @patch('handler.insightface.app.FaceAnalysis')
    def test_get_face_analyser_cuda(self, mock_face_analysis):
        """Test face analyser with CUDA device"""
        mock_analyser = Mock()
        mock_face_analysis.return_value = mock_analyser

        result = get_face_analyser('model_path', 'cuda')

        mock_face_analysis.assert_called_once_with(
            name="buffalo_l",
            root="./checkpoints",
            providers=['CUDAExecutionProvider']
        )
        mock_analyser.prepare.assert_called_once_with(ctx_id=0, det_size=(320, 320))
        assert result == mock_analyser

    @patch('handler.insightface.app.FaceAnalysis')
    def test_get_face_analyser_cpu(self, mock_face_analysis):
        """Test face analyser with CPU device"""
        mock_analyser = Mock()
        mock_face_analysis.return_value = mock_analyser

        result = get_face_analyser('model_path', 'cpu')

        mock_face_analysis.assert_called_once_with(
            name="buffalo_l",
            root="./checkpoints",
            providers=['CPUExecutionProvider']
        )
        mock_analyser.prepare.assert_called_once_with(ctx_id=0, det_size=(320, 320))
        assert result == mock_analyser


class TestGetOneFace:
    """Tests for get_one_face function"""

    def test_get_one_face_single(self):
        """Test getting one face from a frame"""
        mock_analyser = Mock()
        mock_face = Mock()
        mock_face.bbox = [100, 200, 300, 400]
        mock_analyser.get.return_value = [mock_face]

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = get_one_face(mock_analyser, frame)

        assert result == mock_face

    def test_get_one_face_multiple(self):
        """Test getting leftmost face from multiple faces"""
        mock_analyser = Mock()
        mock_face1 = Mock()
        mock_face1.bbox = [200, 200, 300, 400]
        mock_face2 = Mock()
        mock_face2.bbox = [100, 200, 300, 400]
        mock_analyser.get.return_value = [mock_face1, mock_face2]

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = get_one_face(mock_analyser, frame)

        assert result == mock_face2

    def test_get_one_face_no_faces(self):
        """Test when no faces are found"""
        mock_analyser = Mock()
        mock_analyser.get.return_value = []

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = get_one_face(mock_analyser, frame)

        assert result is None


class TestGetManyFaces:
    """Tests for get_many_faces function"""

    def test_get_many_faces_sorted(self):
        """Test getting multiple faces sorted left to right"""
        mock_analyser = Mock()
        mock_face1 = Mock()
        mock_face1.bbox = [300, 200, 400, 400]
        mock_face2 = Mock()
        mock_face2.bbox = [100, 200, 200, 400]
        mock_face3 = Mock()
        mock_face3.bbox = [200, 200, 300, 400]
        mock_analyser.get.return_value = [mock_face1, mock_face2, mock_face3]

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = get_many_faces(mock_analyser, frame)

        assert result == [mock_face2, mock_face3, mock_face1]

    def test_get_many_faces_error(self):
        """Test when face analyser raises an error"""
        mock_analyser = Mock()
        mock_analyser.get.side_effect = IndexError()

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = get_many_faces(mock_analyser, frame)

        assert result is None

    def test_get_many_faces_with_min_size_filter(self):
        """Test filtering faces by minimum size percentage"""
        mock_analyser = Mock()

        mock_face1 = Mock()
        mock_face1.bbox = [100, 100, 300, 300]

        mock_face2 = Mock()
        mock_face2.bbox = [400, 100, 450, 150]

        mock_face3 = Mock()
        mock_face3.bbox = [500, 100, 700, 300]

        mock_analyser.get.return_value = [mock_face1, mock_face2, mock_face3]

        frame = np.zeros((1000, 1000, 3), dtype=np.uint8)
        result = get_many_faces(mock_analyser, frame, min_face_size=20.0)

        assert len(result) == 2
        assert result[0] == mock_face1
        assert result[1] == mock_face3


class TestSwapFace:
    """Tests for swap_face function"""

    def test_swap_face(self):
        """Test face swapping functionality"""
        # Import handler and mock the global FACE_SWAPPER
        import handler

        mock_swapper = Mock()
        handler.FACE_SWAPPER = mock_swapper

        mock_source_face1 = Mock()
        mock_source_face2 = Mock()
        mock_target_face1 = Mock()
        mock_target_face2 = Mock()

        source_faces = [mock_source_face1, mock_source_face2]
        target_faces = [mock_target_face1, mock_target_face2]

        temp_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_result = np.ones((480, 640, 3), dtype=np.uint8)
        mock_swapper.get.return_value = mock_result

        result = swap_face(source_faces, target_faces, 0, 1, temp_frame)

        mock_swapper.get.assert_called_once_with(
            temp_frame, mock_target_face2, mock_source_face1, paste_back=True
        )
        assert np.array_equal(result, mock_result)


class TestDetermineFileExtension:
    """Tests for determine_file_extension function"""

    def test_jpeg_extension(self):
        """Test JPEG file extension detection"""
        jpeg_data = '/9j/test_data'
        result = determine_file_extension(jpeg_data)
        assert result == '.jpg'

    def test_png_extension(self):
        """Test PNG file extension detection"""
        png_data = 'iVBORw0Kgtest_data'
        result = determine_file_extension(png_data)
        assert result == '.png'

    def test_unknown_extension(self):
        """Test unknown file extension defaults to PNG"""
        unknown_data = 'unknown_format_data'
        result = determine_file_extension(unknown_data)
        assert result == '.png'

    def test_exception_handling(self):
        """Test exception handling defaults to PNG"""
        result = determine_file_extension(None)
        assert result == '.png'


class TestCleanUpTemporaryFiles:
    """Tests for clean_up_temporary_files function"""

    @patch('handler.os.remove')
    def test_clean_up_files(self, mock_remove):
        """Test that temporary files are removed"""
        source_path = '/tmp/source.jpg'
        target_path = '/tmp/target.jpg'

        clean_up_temporary_files(source_path, target_path)

        assert mock_remove.call_count == 2
        mock_remove.assert_any_call(source_path)
        mock_remove.assert_any_call(target_path)


class TestHandler:
    """Tests for main handler function"""

    @patch('handler.validate')
    def test_handler_with_validation_errors(self, mock_validate):
        """Test handler returns errors when validation fails"""
        mock_validate.return_value = {'errors': 'Invalid input'}

        event = {
            'id': 'test_job_123',
            'input': {'invalid': 'data'}
        }

        result = handler(event)

        assert 'error' in result
        assert result['error'] == 'Invalid input'

    @patch('handler.face_swap_api')
    @patch('handler.validate')
    def test_handler_success(self, mock_validate, mock_face_swap_api):
        """Test handler with valid input"""
        mock_validate.return_value = {
            'validated_input': {
                'source_image': 'base64_source',
                'target_image': 'base64_target'
            }
        }
        mock_face_swap_api.return_value = {'image': 'result_base64'}

        event = {
            'id': 'test_job_123',
            'input': {
                'source_image': 'base64_source',
                'target_image': 'base64_target'
            }
        }

        result = handler(event)

        mock_face_swap_api.assert_called_once_with(
            'test_job_123',
            {'source_image': 'base64_source', 'target_image': 'base64_target'}
        )
        assert result == {'image': 'result_base64'}
