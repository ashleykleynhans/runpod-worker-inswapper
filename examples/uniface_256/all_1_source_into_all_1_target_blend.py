#!/usr/bin/env python3
# Subtle face blend using uniface_256 with 50% weight
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from util import post_request, encode_image_to_base64

SOURCE_IMAGE = '../../data/src.jpg'
TARGET_IMAGE = '../../data/target.jpg'
SOURCE_INDEXES = '-1'
TARGET_INDEXES = '-1'
BACKGROUND_ENHANCE = True
FACE_RESTORE = True
FACE_UPSAMPLE = True
UPSCALE = 1
CODEFORMER_FIDELITY = 0.5
OUTPUT_FORMAT = 'JPEG'
FACE_SWAPPER_MODEL = 'uniface_256'
FACE_SWAPPER_RESOLUTION = '1024x1024'
FACE_SWAPPER_WEIGHT = 0.5


if __name__ == '__main__':
    payload = {
        "input": {
            "source_image": encode_image_to_base64(SOURCE_IMAGE),
            "target_image": encode_image_to_base64(TARGET_IMAGE),
            "source_indexes": SOURCE_INDEXES,
            "target_indexes": TARGET_INDEXES,
            "background_enhance": BACKGROUND_ENHANCE,
            "face_restore": FACE_RESTORE,
            "face_upsample": FACE_UPSAMPLE,
            "upscale": UPSCALE,
            "codeformer_fidelity": CODEFORMER_FIDELITY,
            "output_format": OUTPUT_FORMAT,
            "face_swapper_model": FACE_SWAPPER_MODEL,
            "face_swapper_resolution": FACE_SWAPPER_RESOLUTION,
            "face_swapper_weight": FACE_SWAPPER_WEIGHT,
        }
    }

    post_request(payload)
