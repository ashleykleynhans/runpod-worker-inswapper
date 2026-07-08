INPUT_SCHEMA = {
    "source_image": {"type": str, "required": True},
    "target_image": {"type": str, "required": True},
    "source_indexes": {
        "type": str, "required": False, "default": "-1",
    },
    "target_indexes": {
        "type": str, "required": False, "default": "-1",
    },
    "background_enhance": {"type": bool, "required": False, "default": True},
    "face_restore": {"type": bool, "required": False, "default": True},
    "face_upsample": {"type": bool, "required": False, "default": True},
    "upscale": {"type": int, "required": False, "default": 1},
    "codeformer_fidelity": {"type": float, "required": False, "default": 0.5},
    "output_format": {
        "type": str, "required": False, "default": "JPEG",
        "constraints": lambda v: v in ["JPEG", "PNG"],
    },
    "min_face_size": {"type": float, "required": False, "default": 0.0},
    # Face swapper model selection
    "face_swapper_model": {"type": str, "required": False, "default": "inswapper_128"},
    "face_swapper_resolution": {"type": str, "required": False, "default": None},
    "face_swapper_weight": {"type": float, "required": False, "default": 1.0},
    # Face mask controls
    "face_mask_blur": {
        "type": float, "required": False, "default": 0.3,
    },
    "face_mask_padding": {
        "type": str, "required": False, "default": "0,0,0,0",
    },
    # Face selector controls
    "face_selector_mode": {
        "type": str, "required": False, "default": "many",
    },
    "face_selector_order": {
        "type": str, "required": False, "default": "left-right",
    },
    "face_selector_gender": {
        "type": str, "required": False, "default": None,
    },
    "face_selector_age_start": {
        "type": int, "required": False, "default": None,
    },
    "face_selector_age_end": {
        "type": int, "required": False, "default": None,
    },
}
