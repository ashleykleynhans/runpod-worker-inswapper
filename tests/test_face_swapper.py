# tests/test_face_swapper.py
import pytest
import numpy as np
from face_swapper import get_face_swapper_model


def test_get_face_swapper_model_loads():
    """Model should load successfully"""
    # Note: This test requires model files to exist
    # Skip if not in Docker environment
    try:
        model = get_face_swapper_model('inswapper_128')
        assert model is not None
    except FileNotFoundError:
        pytest.skip("Model files not available")


def test_get_face_swapper_model_caches():
    """Model should be cached after first load"""
    try:
        model1 = get_face_swapper_model('inswapper_128')
        model2 = get_face_swapper_model('inswapper_128')
        assert model1 is model2  # Same instance
    except FileNotFoundError:
        pytest.skip("Model files not available")


def test_get_face_swapper_model_missing_file():
    """Should raise FileNotFoundError for missing model"""
    with pytest.raises(FileNotFoundError, match="Model file not found"):
        get_face_swapper_model('nonexistent_model')
