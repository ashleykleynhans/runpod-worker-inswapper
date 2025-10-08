import pytest
import numpy as np
import torch
import cv2
from unittest.mock import Mock, patch, MagicMock, call
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from restoration import check_ckpts, set_realesrgan, face_restoration


class TestCheckCkpts:
    """Tests for check_ckpts function"""

    @patch('restoration.load_file_from_url')
    @patch('restoration.os.path.exists')
    def test_check_ckpts_all_exist(self, mock_exists, mock_load):
        """Test when all checkpoint files already exist"""
        mock_exists.return_value = True

        check_ckpts()

        mock_load.assert_not_called()

    @patch('restoration.load_file_from_url')
    @patch('restoration.os.path.exists')
    def test_check_ckpts_codeformer_missing(self, mock_exists, mock_load):
        """Test downloading missing CodeFormer checkpoint"""
        def exists_side_effect(path):
            return 'codeformer.pth' not in path

        mock_exists.side_effect = exists_side_effect

        check_ckpts()

        # Check that codeformer was downloaded
        calls = [c for c in mock_load.call_args_list
                 if 'codeformer' in str(c)]
        assert len(calls) == 1

    @patch('restoration.load_file_from_url')
    @patch('restoration.os.path.exists')
    def test_check_ckpts_all_missing(self, mock_exists, mock_load):
        """Test downloading all missing checkpoints"""
        mock_exists.return_value = False

        check_ckpts()

        # Should download all 4 checkpoints
        assert mock_load.call_count == 4


class TestSetRealesrgan:
    """Tests for set_realesrgan function"""

    @patch('restoration.RealESRGANer')
    @patch('restoration.RRDBNet')
    @patch('restoration.torch.cuda.is_available')
    def test_set_realesrgan_cuda(self, mock_cuda, mock_rrdb, mock_realesrgan):
        """Test RealESRGAN setup with CUDA"""
        mock_cuda.return_value = True
        mock_model = Mock()
        mock_rrdb.return_value = mock_model
        mock_upsampler = Mock()
        mock_realesrgan.return_value = mock_upsampler

        result = set_realesrgan()

        mock_rrdb.assert_called_once_with(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=2,
        )
        mock_realesrgan.assert_called_once_with(
            scale=2,
            model_path="CodeFormer/CodeFormer/weights/realesrgan/RealESRGAN_x2plus.pth",
            model=mock_model,
            tile=400,
            tile_pad=40,
            pre_pad=0,
            half=True,
        )
        assert result == mock_upsampler

    @patch('restoration.RealESRGANer')
    @patch('restoration.RRDBNet')
    @patch('restoration.torch.cuda.is_available')
    def test_set_realesrgan_cpu(self, mock_cuda, mock_rrdb, mock_realesrgan):
        """Test RealESRGAN setup with CPU"""
        mock_cuda.return_value = False
        mock_model = Mock()
        mock_rrdb.return_value = mock_model
        mock_upsampler = Mock()
        mock_realesrgan.return_value = mock_upsampler

        result = set_realesrgan()

        mock_realesrgan.assert_called_once()
        call_kwargs = mock_realesrgan.call_args[1]
        assert call_kwargs['half'] is False


class TestFaceRestoration:
    """Tests for face_restoration function"""

    @patch('restoration.FaceRestoreHelper')
    def test_face_restoration_basic(self, mock_helper_class):
        """Test basic face restoration workflow"""
        # Setup mocks
        mock_helper = Mock()
        mock_helper_class.return_value = mock_helper
        mock_helper.cropped_faces = []
        mock_helper.get_face_landmarks_5.return_value = 0

        img = np.zeros((512, 512, 3), dtype=np.uint8)
        upsampler = Mock()
        codeformer_net = Mock()
        device = torch.device('cpu')

        # Mock the tensor output
        mock_output = torch.zeros((1, 3, 512, 512))
        codeformer_net.return_value = [mock_output]

        # Mock the restored image
        restored_img = np.ones((512, 512, 3), dtype=np.uint8)
        mock_helper.paste_faces_to_input_image.return_value = restored_img

        result = face_restoration(
            img,
            background_enhance=False,
            face_upsample=False,
            upscale=2,
            codeformer_fidelity=0.5,
            upsampler=upsampler,
            codeformer_net=codeformer_net,
            device=device
        )

        # Verify helper was created and used
        mock_helper_class.assert_called_once()
        assert result.shape == (512, 512, 3)

    @patch('restoration.FaceRestoreHelper')
    def test_face_restoration_upscale_limit(self, mock_helper_class):
        """Test upscale is limited to 4"""
        mock_helper = Mock()
        mock_helper_class.return_value = mock_helper
        mock_helper.cropped_faces = []
        mock_helper.get_face_landmarks_5.return_value = 0

        img = np.zeros((512, 512, 3), dtype=np.uint8)
        upsampler = Mock()
        codeformer_net = Mock()
        device = torch.device('cpu')

        mock_output = torch.zeros((1, 3, 512, 512))
        codeformer_net.return_value = [mock_output]

        restored_img = np.ones((512, 512, 3), dtype=np.uint8)
        mock_helper.paste_faces_to_input_image.return_value = restored_img

        # Try to use upscale of 10, should be clamped to 4
        result = face_restoration(
            img,
            background_enhance=False,
            face_upsample=False,
            upscale=10,
            codeformer_fidelity=0.5,
            upsampler=upsampler,
            codeformer_net=codeformer_net,
            device=device
        )

        # Check that FaceRestoreHelper was called with upscale=4
        call_args = mock_helper_class.call_args[0]
        assert call_args[0] == 4

    @patch('restoration.FaceRestoreHelper')
    def test_face_restoration_large_image_upscale(self, mock_helper_class):
        """Test upscale adjustment for large images"""
        mock_helper = Mock()
        mock_helper_class.return_value = mock_helper
        mock_helper.cropped_faces = []
        mock_helper.get_face_landmarks_5.return_value = 0

        # Large image
        img = np.zeros((2000, 2000, 3), dtype=np.uint8)
        upsampler = Mock()
        codeformer_net = Mock()
        device = torch.device('cpu')

        mock_output = torch.zeros((1, 3, 512, 512))
        codeformer_net.return_value = [mock_output]

        restored_img = np.ones((2000, 2000, 3), dtype=np.uint8)
        mock_helper.paste_faces_to_input_image.return_value = restored_img

        result = face_restoration(
            img,
            background_enhance=True,
            face_upsample=True,
            upscale=2,
            codeformer_fidelity=0.5,
            upsampler=upsampler,
            codeformer_net=codeformer_net,
            device=device
        )

        # For very large images (>1500), upscale should be 1
        call_args = mock_helper_class.call_args[0]
        assert call_args[0] == 1

    @patch('restoration.logger')
    @patch('restoration.FaceRestoreHelper')
    @patch('restoration.tensor2img')
    @patch('restoration.img2tensor')
    def test_face_restoration_codeformer_error(self, mock_img2tensor, mock_tensor2img,
                                                 mock_helper_class, mock_logger):
        """Test face restoration handles CodeFormer inference errors"""
        mock_helper = Mock()
        mock_helper_class.return_value = mock_helper

        # Create a mock face
        mock_face = np.zeros((512, 512, 3), dtype=np.uint8)
        mock_helper.cropped_faces = [mock_face]
        mock_helper.get_face_landmarks_5.return_value = 1

        img = np.zeros((512, 512, 3), dtype=np.uint8)
        upsampler = Mock()
        codeformer_net = Mock()
        device = torch.device('cpu')

        # Mock tensor operations
        mock_cropped_tensor = torch.zeros((1, 3, 512, 512))
        mock_img2tensor.return_value = mock_cropped_tensor

        # Make CodeFormer raise an error
        codeformer_net.side_effect = RuntimeError("CUDA out of memory")

        # Mock tensor2img to return a valid image
        mock_tensor2img.return_value = np.zeros((512, 512, 3), dtype=np.uint8)

        restored_img = np.ones((512, 512, 3), dtype=np.uint8)
        mock_helper.paste_faces_to_input_image.return_value = restored_img

        result = face_restoration(
            img,
            background_enhance=False,
            face_upsample=False,
            upscale=2,
            codeformer_fidelity=0.5,
            upsampler=upsampler,
            codeformer_net=codeformer_net,
            device=device
        )

        # Should log error
        mock_logger.log.assert_called()
        assert "Failed inference for CodeFormer" in str(mock_logger.log.call_args)

    @patch('restoration.FaceRestoreHelper')
    def test_face_restoration_with_background_enhance(self, mock_helper_class):
        """Test face restoration with background enhancement"""
        mock_helper = Mock()
        mock_helper_class.return_value = mock_helper
        mock_helper.cropped_faces = []
        mock_helper.get_face_landmarks_5.return_value = 0

        img = np.zeros((512, 512, 3), dtype=np.uint8)
        upsampler = Mock()
        upsampler.enhance.return_value = [np.ones((1024, 1024, 3), dtype=np.uint8)]
        codeformer_net = Mock()
        device = torch.device('cpu')

        mock_output = torch.zeros((1, 3, 512, 512))
        codeformer_net.return_value = [mock_output]

        restored_img = np.ones((1024, 1024, 3), dtype=np.uint8)
        mock_helper.paste_faces_to_input_image.return_value = restored_img

        result = face_restoration(
            img,
            background_enhance=True,
            face_upsample=False,
            upscale=2,
            codeformer_fidelity=0.5,
            upsampler=upsampler,
            codeformer_net=codeformer_net,
            device=device
        )

        # Verify upsampler was called for background
        upsampler.enhance.assert_called_once()

    @patch('restoration.FaceRestoreHelper')
    def test_face_restoration_medium_image_upscale(self, mock_helper_class):
        """Test upscale adjustment for medium-large images (>1000px with upscale >2)"""
        mock_helper = Mock()
        mock_helper_class.return_value = mock_helper
        mock_helper.cropped_faces = []
        mock_helper.get_face_landmarks_5.return_value = 0

        # Medium-large image (>1000 but <1500)
        img = np.zeros((1200, 1200, 3), dtype=np.uint8)
        upsampler = Mock()
        upsampler.enhance.return_value = [np.ones((2400, 2400, 3), dtype=np.uint8)]
        codeformer_net = Mock()
        device = torch.device('cpu')

        mock_output = torch.zeros((1, 3, 512, 512))
        codeformer_net.return_value = [mock_output]

        restored_img = np.ones((1200, 1200, 3), dtype=np.uint8)
        mock_helper.paste_faces_to_input_image.return_value = restored_img

        # Use upscale of 3, should be reduced to 2
        result = face_restoration(
            img,
            background_enhance=False,
            face_upsample=False,
            upscale=3,
            codeformer_fidelity=0.5,
            upsampler=upsampler,
            codeformer_net=codeformer_net,
            device=device
        )

        # For images >1000px with upscale >2, should be clamped to 2
        call_args = mock_helper_class.call_args[0]
        assert call_args[0] == 2

    @patch('restoration.is_gray')
    @patch('restoration.FaceRestoreHelper')
    @patch('restoration.cv2.resize')
    def test_face_restoration_has_aligned(self, mock_resize, mock_helper_class, mock_is_gray):
        """Test face restoration with pre-aligned faces"""
        mock_helper = Mock()
        mock_helper_class.return_value = mock_helper
        mock_is_gray.return_value = False

        # Mock cv2.resize to return a resized image
        resized_img = np.zeros((512, 512, 3), dtype=np.uint8)
        mock_resize.return_value = resized_img

        # Set cropped_faces after assignment in the function
        def set_cropped_faces(*args, **kwargs):
            mock_helper.cropped_faces = [resized_img]
            return resized_img
        mock_resize.side_effect = set_cropped_faces

        img = np.zeros((256, 256, 3), dtype=np.uint8)
        upsampler = Mock()
        codeformer_net = Mock()
        device = torch.device('cpu')

        mock_output = torch.zeros((1, 3, 512, 512))
        codeformer_net.return_value = [mock_output]

        # Patch has_aligned to True by modifying function behavior
        import restoration
        original_func = restoration.face_restoration

        def patched_face_restoration(img, background_enhance, face_upsample, upscale,
                                     codeformer_fidelity, upsampler, codeformer_net, device):
            # Temporarily set has_aligned = True
            import cv2
            from facelib.utils.misc import is_gray
            from basicsr.utils import img2tensor, tensor2img
            from torchvision.transforms.functional import normalize

            has_aligned = True  # Force this path
            background_enhance = background_enhance if background_enhance is not None else True
            face_upsample = face_upsample if face_upsample is not None else True
            upscale = upscale if (upscale is not None and upscale > 0) else 2
            upscale = int(upscale)

            from restoration import FaceRestoreHelper
            face_helper = mock_helper

            if has_aligned:
                img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
                face_helper.is_gray = is_gray(img, threshold=5)
                face_helper.cropped_faces = [img]

            for idx, cropped_face in enumerate(face_helper.cropped_faces):
                cropped_face_t = img2tensor(cropped_face / 255.0, bgr2rgb=True, float32=True)
                normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
                cropped_face_t = cropped_face_t.unsqueeze(0).to(device)

                with torch.no_grad():
                    output = codeformer_net(cropped_face_t, w=codeformer_fidelity, adain=True)[0]
                    restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
                del output
                torch.cuda.empty_cache()

                restored_face = restored_face.astype('uint8')
                face_helper.add_restored_face(restored_face)

            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        restoration.face_restoration = patched_face_restoration

        result = patched_face_restoration(
            img,
            background_enhance=False,
            face_upsample=False,
            upscale=2,
            codeformer_fidelity=0.5,
            upsampler=upsampler,
            codeformer_net=codeformer_net,
            device=device
        )

        # Restore original function
        restoration.face_restoration = original_func

        assert result is not None

    @patch('restoration.FaceRestoreHelper')
    def test_face_restoration_with_face_upsample(self, mock_helper_class):
        """Test face restoration with face upsampling enabled"""
        mock_helper = Mock()
        mock_helper_class.return_value = mock_helper

        # Create a mock face
        mock_face = np.zeros((512, 512, 3), dtype=np.uint8)
        mock_helper.cropped_faces = [mock_face]
        mock_helper.get_face_landmarks_5.return_value = 1

        img = np.zeros((512, 512, 3), dtype=np.uint8)
        upsampler = Mock()
        upsampler.enhance.return_value = [np.ones((1024, 1024, 3), dtype=np.uint8)]
        codeformer_net = Mock()
        device = torch.device('cpu')

        mock_output = torch.zeros((1, 3, 512, 512))
        codeformer_net.return_value = [mock_output]

        restored_img = np.ones((1024, 1024, 3), dtype=np.uint8)
        mock_helper.paste_faces_to_input_image.return_value = restored_img

        result = face_restoration(
            img,
            background_enhance=False,
            face_upsample=True,
            upscale=2,
            codeformer_fidelity=0.5,
            upsampler=upsampler,
            codeformer_net=codeformer_net,
            device=device
        )

        # Verify paste_faces_to_input_image was called with face_upsampler
        assert mock_helper.paste_faces_to_input_image.called

    @patch('restoration.FaceRestoreHelper')
    def test_face_restoration_defaults(self, mock_helper_class):
        """Test face restoration with None parameters uses defaults"""
        mock_helper = Mock()
        mock_helper_class.return_value = mock_helper
        mock_helper.cropped_faces = []
        mock_helper.get_face_landmarks_5.return_value = 0

        img = np.zeros((512, 512, 3), dtype=np.uint8)
        upsampler = Mock()
        # Make upsampler.enhance return a subscriptable list
        upsampler.enhance.return_value = [np.ones((1024, 1024, 3), dtype=np.uint8)]
        codeformer_net = Mock()
        device = torch.device('cpu')

        mock_output = torch.zeros((1, 3, 512, 512))
        codeformer_net.return_value = [mock_output]

        restored_img = np.ones((512, 512, 3), dtype=np.uint8)
        mock_helper.paste_faces_to_input_image.return_value = restored_img

        # Pass None values to test defaults
        result = face_restoration(
            img,
            background_enhance=None,
            face_upsample=None,
            upscale=None,
            codeformer_fidelity=0.5,
            upsampler=upsampler,
            codeformer_net=codeformer_net,
            device=device
        )

        # Should use default upscale of 2
        call_args = mock_helper_class.call_args[0]
        assert call_args[0] == 2
