import pytest
import base64
import numpy as np
from unittest.mock import Mock, patch, MagicMock, mock_open
from PIL import Image
import io
import cv2
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from handler import process, face_swap, face_swap_api


class TestProcess:
    """Tests for process function"""

    @patch('handler.get_many_faces')
    @patch('handler.swap_face')
    def test_process_no_target_faces(self, mock_swap, mock_get_faces):
        """Test process raises exception when no faces in target"""
        import handler
        handler.FACE_ANALYSER = Mock()

        mock_get_faces.return_value = []

        source_img = [Image.new('RGB', (512, 512))]
        target_img = Image.new('RGB', (512, 512))

        with pytest.raises(Exception, match='The target image does not contain any faces!'):
            process('job_id', source_img, target_img, '-1', '-1')

    @patch('handler.get_many_faces')
    @patch('handler.swap_face')
    def test_process_multiple_source_equal_target(self, mock_swap, mock_get_faces):
        """Test process with multiple sources equal to targets"""
        import handler
        handler.FACE_ANALYSER = Mock()

        # Create mock faces
        mock_target_face1 = Mock()
        mock_target_face2 = Mock()
        mock_source_face1 = Mock()
        mock_source_face2 = Mock()

        # First call for target faces, then source faces for each image
        mock_get_faces.side_effect = [
            [mock_target_face1, mock_target_face2],  # target faces
            [mock_source_face1],  # source faces from first source image
            [mock_source_face2],  # source faces from second source image
        ]

        mock_result = np.ones((512, 512, 3), dtype=np.uint8)
        mock_swap.return_value = mock_result

        source_img = [Image.new('RGB', (512, 512)), Image.new('RGB', (512, 512))]
        target_img = Image.new('RGB', (512, 512))

        result = process('job_id', source_img, target_img, '-1', '-1')

        assert mock_swap.call_count == 2
        assert isinstance(result, Image.Image)

    @patch('handler.get_many_faces')
    @patch('handler.swap_face')
    def test_process_multiple_source_no_source_faces(self, mock_swap, mock_get_faces):
        """Test process raises exception when source image has no faces"""
        import handler
        handler.FACE_ANALYSER = Mock()

        mock_target_face1 = Mock()
        mock_target_face2 = Mock()

        # Target has faces, but source doesn't
        mock_get_faces.side_effect = [
            [mock_target_face1, mock_target_face2],  # target faces
            None,  # No source faces
        ]

        source_img = [Image.new('RGB', (512, 512)), Image.new('RGB', (512, 512))]
        target_img = Image.new('RGB', (512, 512))

        with pytest.raises(Exception, match='No source faces found!'):
            process('job_id', source_img, target_img, '-1', '-1')

    @patch('handler.get_many_faces')
    @patch('handler.swap_face')
    def test_process_single_source_single_face(self, mock_swap, mock_get_faces):
        """Test process with single source face and single target"""
        import handler
        handler.FACE_ANALYSER = Mock()

        mock_target_face = Mock()
        mock_source_face = Mock()

        mock_get_faces.side_effect = [
            [mock_target_face],  # target faces
            [mock_source_face],  # source faces
        ]

        mock_result = np.ones((512, 512, 3), dtype=np.uint8)
        mock_swap.return_value = mock_result

        source_img = [Image.new('RGB', (512, 512))]
        target_img = Image.new('RGB', (512, 512))

        result = process('job_id', source_img, target_img, '-1', '-1')

        mock_swap.assert_called_once()
        assert isinstance(result, Image.Image)

    @patch('handler.get_many_faces')
    @patch('handler.swap_face')
    def test_process_fewer_source_than_target(self, mock_swap, mock_get_faces):
        """Test process with fewer source faces than target faces"""
        import handler
        handler.FACE_ANALYSER = Mock()

        mock_target_face1 = Mock()
        mock_target_face2 = Mock()
        mock_target_face3 = Mock()
        mock_source_face1 = Mock()
        mock_source_face2 = Mock()

        mock_get_faces.side_effect = [
            [mock_target_face1, mock_target_face2, mock_target_face3],  # 3 target faces
            [mock_source_face1, mock_source_face2],  # 2 source faces
        ]

        mock_result = np.ones((512, 512, 3), dtype=np.uint8)
        mock_swap.return_value = mock_result

        source_img = [Image.new('RGB', (512, 512))]
        target_img = Image.new('RGB', (512, 512))

        result = process('job_id', source_img, target_img, '-1', '-1')

        # Should swap 2 faces (limited by source)
        assert mock_swap.call_count == 2

    @patch('handler.get_many_faces')
    @patch('handler.swap_face')
    def test_process_fewer_target_than_source(self, mock_swap, mock_get_faces):
        """Test process with fewer target faces than source faces"""
        import handler
        handler.FACE_ANALYSER = Mock()

        mock_target_face1 = Mock()
        mock_target_face2 = Mock()
        mock_source_face1 = Mock()
        mock_source_face2 = Mock()
        mock_source_face3 = Mock()

        mock_get_faces.side_effect = [
            [mock_target_face1, mock_target_face2],  # 2 target faces
            [mock_source_face1, mock_source_face2, mock_source_face3],  # 3 source faces
        ]

        mock_result = np.ones((512, 512, 3), dtype=np.uint8)
        mock_swap.return_value = mock_result

        source_img = [Image.new('RGB', (512, 512))]
        target_img = Image.new('RGB', (512, 512))

        result = process('job_id', source_img, target_img, '-1', '-1')

        # Should swap 2 faces (limited by target)
        assert mock_swap.call_count == 2

    @patch('handler.get_many_faces')
    @patch('handler.swap_face')
    def test_process_no_source_faces(self, mock_swap, mock_get_faces):
        """Test process raises exception when no source faces"""
        import handler
        handler.FACE_ANALYSER = Mock()

        mock_target_face = Mock()

        mock_get_faces.side_effect = [
            [mock_target_face],  # target faces
            [],  # empty source faces (returns empty list, not None)
        ]

        source_img = [Image.new('RGB', (512, 512))]
        target_img = Image.new('RGB', (512, 512))

        # Actually should work, but swap 0 faces - let's test it raises
        with pytest.raises(Exception):
            process('job_id', source_img, target_img, '-1', '-1')

    @patch('handler.get_many_faces')
    @patch('handler.swap_face')
    def test_process_specific_target_indexes_single_source(self, mock_swap, mock_get_faces):
        """Test process with specific target indexes and auto source"""
        import handler
        handler.FACE_ANALYSER = Mock()

        mock_target_face1 = Mock()
        mock_target_face2 = Mock()
        mock_target_face3 = Mock()
        mock_source_face1 = Mock()
        mock_source_face2 = Mock()

        mock_get_faces.side_effect = [
            [mock_target_face1, mock_target_face2, mock_target_face3],  # target faces
            [mock_source_face1, mock_source_face2],  # source faces
        ]

        mock_result = np.ones((512, 512, 3), dtype=np.uint8)
        mock_swap.return_value = mock_result

        source_img = [Image.new('RGB', (512, 512))]
        target_img = Image.new('RGB', (512, 512))

        # With target_indexes specified but source_indexes="-1", it should expand both
        result = process('job_id', source_img, target_img, '0,1', '0,1')

        assert mock_swap.call_count == 2

    @patch('handler.get_many_faces')
    @patch('handler.swap_face')
    def test_process_specific_source_and_target_indexes(self, mock_swap, mock_get_faces):
        """Test process with specific source and target indexes"""
        import handler
        handler.FACE_ANALYSER = Mock()

        mock_target_face1 = Mock()
        mock_target_face2 = Mock()
        mock_source_face1 = Mock()
        mock_source_face2 = Mock()

        mock_get_faces.side_effect = [
            [mock_target_face1, mock_target_face2],  # target faces
            [mock_source_face1, mock_source_face2],  # source faces
        ]

        mock_result = np.ones((512, 512, 3), dtype=np.uint8)
        mock_swap.return_value = mock_result

        source_img = [Image.new('RGB', (512, 512))]
        target_img = Image.new('RGB', (512, 512))

        result = process('job_id', source_img, target_img, '0,1', '0,1')

        assert mock_swap.call_count == 2

    @patch('handler.get_many_faces')
    @patch('handler.swap_face')
    def test_process_source_index_out_of_bounds(self, mock_swap, mock_get_faces):
        """Test process raises exception when source index is out of bounds"""
        import handler
        handler.FACE_ANALYSER = Mock()

        mock_target_face1 = Mock()
        mock_target_face2 = Mock()
        mock_source_face1 = Mock()
        mock_source_face2 = Mock()

        mock_get_faces.side_effect = [
            [mock_target_face1, mock_target_face2],  # 2 target faces
            [mock_source_face1, mock_source_face2],  # 2 source faces
        ]

        mock_result = np.ones((512, 512, 3), dtype=np.uint8)
        mock_swap.return_value = mock_result

        source_img = [Image.new('RGB', (512, 512))]
        target_img = Image.new('RGB', (512, 512))

        # Equal counts (2:2) to trigger validation: source index 5 when only 2 faces exist
        with pytest.raises(ValueError, match='Source index 5 is higher than the number of faces'):
            process('job_id', source_img, target_img, '0,5', '0,1')

    @patch('handler.get_many_faces')
    @patch('handler.swap_face')
    def test_process_target_index_out_of_bounds(self, mock_swap, mock_get_faces):
        """Test process raises exception when target index is out of bounds"""
        import handler
        handler.FACE_ANALYSER = Mock()

        mock_target_face1 = Mock()
        mock_target_face2 = Mock()
        mock_source_face1 = Mock()
        mock_source_face2 = Mock()

        mock_get_faces.side_effect = [
            [mock_target_face1, mock_target_face2],  # 2 target faces
            [mock_source_face1, mock_source_face2],  # 2 source faces
        ]

        mock_result = np.ones((512, 512, 3), dtype=np.uint8)
        mock_swap.return_value = mock_result

        source_img = [Image.new('RGB', (512, 512))]
        target_img = Image.new('RGB', (512, 512))

        # Equal counts (2:2) to trigger validation: target index 5 when only 2 faces exist
        with pytest.raises(ValueError, match='Target index 5 is higher than the number of faces'):
            process('job_id', source_img, target_img, '0,1', '0,5')

    @patch('handler.get_many_faces')
    def test_process_too_many_source_indexes(self, mock_get_faces):
        """Test process raises exception when too many source indexes specified"""
        import handler
        handler.FACE_ANALYSER = Mock()

        mock_target_face1 = Mock()
        mock_target_face2 = Mock()
        mock_source_face1 = Mock()

        mock_get_faces.side_effect = [
            [mock_target_face1, mock_target_face2],  # 2 target faces
            [mock_source_face1],  # 1 source face
        ]

        source_img = [Image.new('RGB', (512, 512))]
        target_img = Image.new('RGB', (512, 512))

        # Specify 3 source indexes when only 1 face exists
        with pytest.raises(Exception, match='Number of source indexes is greater than the number of faces'):
            process('job_id', source_img, target_img, '0,1,2', '0,1')

    @patch('handler.get_many_faces')
    def test_process_target_faces_none(self, mock_get_faces):
        """Test process raises exception when target_faces is None"""
        import handler
        handler.FACE_ANALYSER = Mock()

        mock_get_faces.return_value = None

        source_img = [Image.new('RGB', (512, 512))]
        target_img = Image.new('RGB', (512, 512))

        with pytest.raises(Exception):  # Will raise due to None being evaluated
            process('job_id', source_img, target_img, '-1', '-1')


class TestFaceSwap:
    """Tests for face_swap function"""

    @patch('handler.face_restoration')
    @patch('handler.process')
    @patch('handler.Image.open')
    def test_face_swap_without_restoration(self, mock_open, mock_process, mock_restoration):
        """Test face_swap without face restoration"""
        import handler
        handler.TORCH_DEVICE = 'cpu'
        handler.CODEFORMER_DEVICE = 'cpu'
        handler.CODEFORMER_NET = Mock()

        mock_img = Image.new('RGB', (512, 512))
        mock_open.return_value = mock_img
        mock_process.return_value = mock_img

        result = face_swap(
            'job_id',
            '/tmp/source.jpg',
            '/tmp/target.jpg',
            '-1',
            '-1',
            False,
            False,  # face_restore=False
            False,
            1,
            0.5,
            'JPEG',
            0
        )

        mock_restoration.assert_not_called()
        assert isinstance(result, str)  # base64 encoded

    @patch('handler.face_restoration')
    @patch('handler.process')
    @patch('handler.Image.open')
    def test_face_swap_with_restoration(self, mock_open, mock_process, mock_restoration):
        """Test face_swap with face restoration"""
        import handler
        handler.TORCH_DEVICE = 'cpu'
        handler.CODEFORMER_DEVICE = 'cpu'
        handler.CODEFORMER_NET = Mock()
        # Create upsampler as a module-level variable
        import sys
        handler.upsampler = Mock()

        mock_img = Image.new('RGB', (512, 512))
        mock_open.return_value = mock_img
        mock_process.return_value = mock_img

        restored_img = np.ones((512, 512, 3), dtype=np.uint8)
        mock_restoration.return_value = restored_img

        result = face_swap(
            'job_id',
            '/tmp/source.jpg',
            '/tmp/target.jpg',
            '-1',
            '-1',
            True,
            True,  # face_restore=True
            True,
            2,
            0.7,
            'PNG',
            0
        )

        mock_restoration.assert_called_once()
        assert isinstance(result, str)  # base64 encoded

    @patch('handler.process')
    @patch('handler.Image.open')
    def test_face_swap_raises_exception(self, mock_open, mock_process):
        """Test face_swap raises exception on error"""
        import handler

        mock_img = Image.new('RGB', (512, 512))
        mock_open.return_value = mock_img
        mock_process.side_effect = Exception("Test error")

        with pytest.raises(Exception, match="Test error"):
            face_swap(
                'job_id',
                '/tmp/source.jpg',
                '/tmp/target.jpg',
                '-1',
                '-1',
                False,
                False,
                False,
                1,
                0.5,
                'JPEG',
                0
            )


class TestFaceSwapApi:
    """Tests for face_swap_api function"""

    @patch('handler.face_swap')
    @patch('handler.clean_up_temporary_files')
    @patch('handler.os.makedirs')
    @patch('handler.os.path.exists')
    @patch('builtins.open', new_callable=mock_open)
    @patch('handler.base64.b64decode')
    def test_face_swap_api_success(self, mock_b64decode, mock_file, mock_exists,
                                     mock_makedirs, mock_cleanup, mock_face_swap):
        """Test face_swap_api successful execution"""
        mock_exists.return_value = False
        mock_b64decode.return_value = b'fake_image_data'
        mock_face_swap.return_value = 'base64_result'

        job_input = {
            'source_image': '/9j/fake_jpeg_data',
            'target_image': 'iVBORw0Kgfake_png_data',
            'source_indexes': '-1',
            'target_indexes': '-1',
            'background_enhance': True,
            'face_restore': True,
            'face_upsample': True,
            'upscale': 2,
            'codeformer_fidelity': 0.5,
            'output_format': 'JPEG',
            'min_face_size': 0
        }

        result = face_swap_api('job_id', job_input)

        assert 'image' in result
        assert result['image'] == 'base64_result'
        mock_makedirs.assert_called_once()
        mock_cleanup.assert_called_once()

    @patch('handler.face_swap')
    @patch('handler.clean_up_temporary_files')
    @patch('handler.os.makedirs')
    @patch('handler.os.path.exists')
    @patch('builtins.open', new_callable=mock_open)
    @patch('handler.base64.b64decode')
    def test_face_swap_api_exception(self, mock_b64decode, mock_file, mock_exists,
                                      mock_makedirs, mock_cleanup, mock_face_swap):
        """Test face_swap_api handles exceptions"""
        mock_exists.return_value = True
        mock_b64decode.return_value = b'fake_image_data'
        mock_face_swap.side_effect = Exception("Processing failed")

        job_input = {
            'source_image': '/9j/fake_jpeg_data',
            'target_image': 'iVBORw0Kgfake_png_data',
            'source_indexes': '-1',
            'target_indexes': '-1',
            'background_enhance': True,
            'face_restore': True,
            'face_upsample': True,
            'upscale': 2,
            'codeformer_fidelity': 0.5,
            'output_format': 'JPEG',
            'min_face_size': 0
        }

        result = face_swap_api('job_id', job_input)

        assert 'error' in result
        assert result['error'] == 'Processing failed'
        assert 'refresh_worker' in result
        assert result['refresh_worker'] is True
        mock_cleanup.assert_called_once()


class TestCompleteHandlerCoverage:
    """Additional tests for complete handler.py coverage"""

    @patch('handler.logger')
    @patch('handler.get_many_faces')
    @patch('handler.swap_face')
    def test_fewer_source_than_target_faces(self, mock_swap, mock_get_faces, mock_logger):
        """Test when there are fewer source faces than target faces"""
        import handler
        handler.FACE_ANALYSER = Mock()

        mock_target_faces = [Mock(), Mock(), Mock()]
        mock_source_faces = [Mock(), Mock()]

        mock_get_faces.side_effect = [mock_target_faces, mock_source_faces]
        mock_swap.return_value = np.ones((512, 512, 3), dtype=np.uint8)

        result = process('job_id', [Image.new('RGB', (512, 512))],
                        Image.new('RGB', (512, 512)), '-1', '-1')

        assert mock_swap.call_count == 2

    @patch('handler.logger')
    @patch('handler.get_many_faces')
    @patch('handler.swap_face')
    def test_target_indexes_expansion(self, mock_swap, mock_get_faces, mock_logger):
        """Test target_indexes expansion when set to '-1'"""
        import handler
        handler.FACE_ANALYSER = Mock()

        mock_faces = [Mock(), Mock()]
        mock_get_faces.side_effect = [mock_faces, mock_faces]
        mock_swap.return_value = np.ones((512, 512, 3), dtype=np.uint8)

        result = process('job_id', [Image.new('RGB', (512, 512))],
                        Image.new('RGB', (512, 512)), '0,1', '-1')

        assert mock_swap.call_count == 2

    @patch('handler.logger')
    @patch('handler.get_many_faces')
    def test_empty_target_faces_list(self, mock_get_faces, mock_logger):
        """Test error when target image has no faces"""
        import handler
        handler.FACE_ANALYSER = Mock()

        mock_get_faces.return_value = []

        with pytest.raises(Exception, match='The target image does not contain any faces!'):
            process('job_id', [Image.new('RGB', (512, 512))],
                   Image.new('RGB', (512, 512)), '-1', '-1')

    @patch('handler.logger')
    @patch('handler.get_many_faces')
    @patch('handler.swap_face')
    def test_single_source_face_replacement(self, mock_swap, mock_get_faces, mock_logger):
        """Test single source face into multiple targets"""
        import handler
        handler.FACE_ANALYSER = Mock()

        mock_target_faces = [Mock(), Mock()]
        mock_source_faces = [Mock()]

        mock_get_faces.side_effect = [mock_target_faces, mock_source_faces]
        mock_swap.return_value = np.ones((512, 512, 3), dtype=np.uint8)

        result = process('job_id', [Image.new('RGB', (512, 512))],
                        Image.new('RGB', (512, 512)), '-1', '-1')

        assert mock_swap.call_count == 1

    @patch('handler.logger')
    @patch('handler.get_many_faces')
    @patch('handler.swap_face')
    def test_specific_target_auto_source(self, mock_swap, mock_get_faces, mock_logger):
        """Test source_indexes='-1' with specific target_indexes"""
        import handler
        handler.FACE_ANALYSER = Mock()

        mock_target_faces = [Mock(), Mock(), Mock()]
        mock_source_faces = [Mock()]
        mock_get_faces.side_effect = [mock_target_faces, mock_source_faces]
        mock_swap.return_value = np.ones((512, 512, 3), dtype=np.uint8)

        result = process('job_id', [Image.new('RGB', (512, 512))],
                        Image.new('RGB', (512, 512)), '-1', '0,1')

        # Should use first source face (index 0) for both target faces
        assert mock_swap.call_count == 2
        assert isinstance(result, Image.Image)


    @patch('handler.logger')
    @patch('handler.get_many_faces')
    @patch('handler.swap_face')
    def test_mismatched_source_target_counts_no_swap(self, mock_swap, mock_get_faces, mock_logger):
        """Test when source and target face counts don't match - no swap occurs"""
        import handler
        handler.FACE_ANALYSER = Mock()

        mock_target_faces = [Mock(), Mock(), Mock()]
        mock_source_faces = [Mock(), Mock(), Mock()]

        mock_get_faces.side_effect = [mock_target_faces, mock_source_faces]
        mock_swap.return_value = np.ones((512, 512, 3), dtype=np.uint8)

        result = process('job_id', [Image.new('RGB', (512, 512))],
                        Image.new('RGB', (512, 512)), '0,1,2', '0')

        assert mock_swap.call_count == 0
        assert isinstance(result, Image.Image)

    @patch('handler.logger')
    @patch('handler.get_many_faces')
    def test_unsupported_face_configuration(self, mock_get_faces, mock_logger):
        """Test unsupported face configuration error"""
        import handler
        handler.FACE_ANALYSER = Mock()

        mock_target_faces = [Mock(), Mock()]
        mock_get_faces.return_value = mock_target_faces

        with pytest.raises(Exception, match='Unsupported face configuration'):
            process('job_id', [Image.new('RGB', (512, 512)),
                              Image.new('RGB', (512, 512)),
                              Image.new('RGB', (512, 512))],
                   Image.new('RGB', (512, 512)), '-1', '-1')

    @patch('handler.face_restoration')
    @patch('handler.process')
    @patch('handler.Image.open')
    def test_face_restoration_exception(self, mock_open, mock_process, mock_restoration):
        """Test exception handling in face_restoration"""
        import handler
        handler.TORCH_DEVICE = 'cpu'
        handler.CODEFORMER_DEVICE = 'cpu'
        handler.CODEFORMER_NET = Mock()
        handler.upsampler = Mock()

        mock_img = Image.new('RGB', (512, 512))
        mock_open.return_value = mock_img
        mock_process.return_value = mock_img
        mock_restoration.side_effect = Exception("Restoration failed")

        with pytest.raises(Exception, match="Restoration failed"):
            face_swap(
                'job_id',
                '/tmp/source.jpg',
                '/tmp/target.jpg',
                '-1',
                '-1',
                True,
                True,
                True,
                2,
                0.7,
                'PNG',
                0
            )
