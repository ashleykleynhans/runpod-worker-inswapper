#!/usr/bin/env python3
# Replace both the faces from a source image containing 2 faces
# into all the first 2 faces in a target image containing 3 faces
from util import post_request, encode_image_to_base64

SOURCE_IMAGE = '../data/swap_src.jpeg'
TARGET_IMAGE = '../data/hp-3faces.jpeg'
SOURCE_INDEXES = '-1'
TARGET_INDEXES = '-1'
BACKGROUND_ENHANCE = True
FACE_RESTORE = True
FACE_UPSAMPLE = True
UPSCALE = 1
CODEFORMER_FIDELITY = 0.5
OUTPUT_FORMAT = 'JPEG'


if __name__ == '__main__':
    # Create the payload dictionary
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
            "output_format": OUTPUT_FORMAT
        }
    }

    post_request(payload)
