import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from schemas.input import INPUT_SCHEMA


class TestInputSchema:
    """Tests for INPUT_SCHEMA"""

    def test_schema_has_required_fields(self):
        """Test that schema contains all required fields"""
        assert 'source_image' in INPUT_SCHEMA
        assert 'target_image' in INPUT_SCHEMA
        assert 'source_indexes' in INPUT_SCHEMA
        assert 'target_indexes' in INPUT_SCHEMA
        assert 'background_enhance' in INPUT_SCHEMA
        assert 'face_restore' in INPUT_SCHEMA
        assert 'face_upsample' in INPUT_SCHEMA
        assert 'upscale' in INPUT_SCHEMA
        assert 'codeformer_fidelity' in INPUT_SCHEMA
        assert 'output_format' in INPUT_SCHEMA

    def test_source_image_required(self):
        """Test source_image is required"""
        assert INPUT_SCHEMA['source_image']['required'] is True
        assert INPUT_SCHEMA['source_image']['type'] == str

    def test_target_image_required(self):
        """Test target_image is required"""
        assert INPUT_SCHEMA['target_image']['required'] is True
        assert INPUT_SCHEMA['target_image']['type'] == str

    def test_source_indexes_optional(self):
        """Test source_indexes is optional with correct default"""
        assert INPUT_SCHEMA['source_indexes']['required'] is False
        assert INPUT_SCHEMA['source_indexes']['default'] == "-1"
        assert INPUT_SCHEMA['source_indexes']['type'] == str

    def test_target_indexes_optional(self):
        """Test target_indexes is optional with correct default"""
        assert INPUT_SCHEMA['target_indexes']['required'] is False
        assert INPUT_SCHEMA['target_indexes']['default'] == "-1"
        assert INPUT_SCHEMA['target_indexes']['type'] == str

    def test_background_enhance_default(self):
        """Test background_enhance has correct type and default"""
        assert INPUT_SCHEMA['background_enhance']['required'] is False
        assert INPUT_SCHEMA['background_enhance']['default'] is True
        assert INPUT_SCHEMA['background_enhance']['type'] == bool

    def test_face_restore_default(self):
        """Test face_restore has correct type and default"""
        assert INPUT_SCHEMA['face_restore']['required'] is False
        assert INPUT_SCHEMA['face_restore']['default'] is True
        assert INPUT_SCHEMA['face_restore']['type'] == bool

    def test_face_upsample_default(self):
        """Test face_upsample has correct type and default"""
        assert INPUT_SCHEMA['face_upsample']['required'] is False
        assert INPUT_SCHEMA['face_upsample']['default'] is True
        assert INPUT_SCHEMA['face_upsample']['type'] == bool

    def test_upscale_default(self):
        """Test upscale has correct type and default"""
        assert INPUT_SCHEMA['upscale']['required'] is False
        assert INPUT_SCHEMA['upscale']['default'] == 1
        assert INPUT_SCHEMA['upscale']['type'] == int

    def test_codeformer_fidelity_default(self):
        """Test codeformer_fidelity has correct type and default"""
        assert INPUT_SCHEMA['codeformer_fidelity']['required'] is False
        assert INPUT_SCHEMA['codeformer_fidelity']['default'] == 0.5
        assert INPUT_SCHEMA['codeformer_fidelity']['type'] == float

    def test_output_format_default(self):
        """Test output_format has correct type and default"""
        assert INPUT_SCHEMA['output_format']['required'] is False
        assert INPUT_SCHEMA['output_format']['default'] == 'JPEG'
        assert INPUT_SCHEMA['output_format']['type'] == str

    def test_output_format_constraints(self):
        """Test output_format constraints"""
        constraint = INPUT_SCHEMA['output_format']['constraints']

        # Valid formats
        assert constraint('JPEG') is True
        assert constraint('PNG') is True

        # Invalid formats
        assert constraint('GIF') is False
        assert constraint('BMP') is False
        assert constraint('jpeg') is False  # Case sensitive
        assert constraint('png') is False   # Case sensitive
        assert constraint('') is False

    def test_all_optional_fields_have_defaults(self):
        """Test that all optional fields have default values"""
        for field_name, field_config in INPUT_SCHEMA.items():
            if not field_config['required']:
                assert 'default' in field_config, f"{field_name} is optional but has no default"

    def test_schema_structure_consistency(self):
        """Test that all schema entries have consistent structure"""
        required_keys = ['type', 'required']

        for field_name, field_config in INPUT_SCHEMA.items():
            for key in required_keys:
                assert key in field_config, f"{field_name} missing required key: {key}"

            # If not required, must have default
            if not field_config['required']:
                assert 'default' in field_config
