INPUT_SCHEMA = {
    'source_image': {
        'type': str,
        'required': True
    },
    'target_image': {
        'type': str,
        'required': True
    },
    'source_indexes': {
        'type': str,
        'required': False,
        'default': "-1"  # Default to using all faces in the source image
    },
    'target_indexes': {
        'type': str,
        'required': False,
        'default': "-1"  # Default to swapping as many faces in the target image as there are in the source image
    },
    'background_enhance': {
        'type': bool,
        'required': False,
        'default': True
    },
    'face_restore': {
        'type': bool,
        'required': False,
        'default': True
    },
    'face_upsample': {
        'type': bool,
        'required': False,
        'default': True
    },
    'upscale': {
        'type': int,
        'required': False,
        'default': 1
    },
    'codeformer_fidelity': {
        'type': float,
        'required': False,
        'default': 0.5
    },
    'output_format': {
        'type': str,
        'required': False,
        'default': 'JPEG',
        'constraints': lambda output_format: output_format in [
            'JPEG',
            'PNG'
        ]
    },
    'min_face_size': {
        'type': float,
        'required': False,
        'default': 0.0
    }
}
