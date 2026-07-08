#!/usr/bin/env python3
# Face swap using hyperswap_1b_256
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from util import post_request, encode_image_to_base64

SOURCE_IMAGE = os.path.join(os.path.dirname(__file__), "../../data/src.jpg")
TARGET_IMAGE = os.path.join(os.path.dirname(__file__), "../../data/target.jpg")
SOURCE_INDEXES = '-1'
TARGET_INDEXES = '-1'
BACKGROUND_ENHANCE = True
FACE_RESTORE = True
FACE_UPSAMPLE = True
UPSCALE = 1
CODEFORMER_FIDELITY = 0.5
OUTPUT_FORMAT = 'JPEG'
FACE_SWAPPER_MODEL = 'hyperswap_1b_256'
FACE_SWAPPER_RESOLUTION = '1024x1024'
FACE_SWAPPER_WEIGHT = 1.0
FACE_MASK_BLUR = 0.3
FACE_MASK_PADDING = "0,0,0,0"
FACE_SELECTOR_MODE = "many"
FACE_SELECTOR_ORDER = "left-right"


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
            "face_mask_blur": FACE_MASK_BLUR,
            "face_mask_padding": FACE_MASK_PADDING,
            "face_selector_mode": FACE_SELECTOR_MODE,
            "face_selector_order": FACE_SELECTOR_ORDER,
        }
    }

    post_request(payload)
