#!/usr/bin/env python3
# Replace the single source face into the first face in the target image
from util import post_request, encode_image_to_base64

SOURCE_IMAGE = '../data/src.jpg'
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
