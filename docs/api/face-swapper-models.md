# Face Swapper Models API Documentation

## Overview

The Runpod Inswapper worker supports 13 face swap models with different quality/speed characteristics, configurable resolution, face mask controls, and face selector filtering.

## Parameters

### face_swapper_model

**Type:** string
**Required:** No
**Default:** `inswapper_128`

Selects which face swap model to use for inference.

**Valid values:**
- `blendswap_256`
- `ghost_1_256`, `ghost_2_256`, `ghost_3_256`
- `hififace_unofficial_256`
- `hyperswap_1a_256`, `hyperswap_1b_256`, `hyperswap_1c_256`
- `inswapper_128` (default)
- `inswapper_128_fp16`
- `simswap_256`
- `simswap_unofficial_512`
- `uniface_256`

### face_swapper_resolution

**Type:** string
**Required:** No
**Default:** Auto (model-dependent)

Resolution for face swapping inference.

**Valid values:** `128x128`, `256x256`, `384x384`, `512x512`, `768x768`, `1024x1024`

Not all models support all resolutions. See compatibility matrix below. If not specified, automatically selects the model's default resolution:

- `512x512` for `inswapper_128`, `inswapper_128_fp16`, `simswap_unofficial_512`
- `1024x1024` for all other models

### face_swapper_weight

**Type:** number
**Required:** No
**Default:** 1.0
**Range:** 0.0 - 1.0

Controls blending between original and swapped face identity. Models that accept embedding inputs (inswapper, simswap, ghost, hififace, hyperswap) interpolate source and target embeddings. Source-face models (blendswap, uniface) use pixel-level blending.

- `1.0` - Full face swap (100% swapped identity)
- `0.7` - Subtle blend (70% swapped, 30% original)
- `0.5` - Half blend
- `0.0` - No swap (original identity)

### face_mask_blur

**Type:** number
**Required:** No
**Default:** 0.3
**Range:** 0.0 - 1.0

Controls the softness of the face mask edges when blending the swapped face back into the original image. Higher values produce softer, more blended edges. Lower values produce sharper edges.

### face_mask_padding

**Type:** string
**Required:** No
**Default:** `"0,0,0,0"`

Padding to inset the face mask from the bounding box edges, in CSS order: `"top,right,bottom,left"`. Values are percentages of the face crop size. For example, `"10,10,10,10"` trims 10% from each edge of the mask.

### face_selector_mode

**Type:** string
**Required:** No
**Default:** `many`

Controls how many target faces to swap:
- `many` — Swap all detected/matching target faces
- `one` — Swap only the first matching face (after sorting)

### face_selector_order

**Type:** string
**Required:** No
**Default:** `left-right`

Sort order for detected target faces:
- `left-right` — Sort by x-coordinate (leftmost first)
- `right-left` — Sort by x-coordinate (rightmost first)
- `top-bottom` — Sort by y-coordinate (topmost first)
- `small-large` — Sort by face area (smallest first)
- `large-small` — Sort by face area (largest first)
- `best-worst` — Sort by detection confidence (highest first)
- `worst-best` — Sort by detection confidence (lowest first)

### face_selector_gender

**Type:** string
**Required:** No
**Default:** `null` (no filter)

Filters target faces by perceived gender:
- `male` — Only swap male-presenting faces
- `female` — Only swap female-presenting faces

### face_selector_age_start

**Type:** integer
**Required:** No
**Default:** `null` (no minimum)

Minimum age (inclusive) of target faces to swap. Faces with estimated age below this value are skipped.

### face_selector_age_end

**Type:** integer
**Required:** No
**Default:** `null` (no maximum)

Maximum age (inclusive) of target faces to swap. Faces with estimated age above this value are skipped.

## Model Compatibility Matrix

| Model | Supported Resolutions | Warp Template | Source Type |
|---|---|---|---|
| blendswap_256 | 256x256, 384x384, 512x512, 768x768, 1024x1024 | ffhq_512 | source_face |
| ghost_1_256 | 256x256, 512x512, 768x768, 1024x1024 | arcface_112_v1 | embedding |
| ghost_2_256 | 256x256, 512x512, 768x768, 1024x1024 | arcface_112_v1 | embedding |
| ghost_3_256 | 256x256, 512x512, 768x768, 1024x1024 | arcface_112_v1 | embedding |
| hififace_unofficial_256 | 256x256, 512x512, 768x768, 1024x1024 | mtcnn_512 | embedding |
| hyperswap_1a_256 | 256x256, 512x512, 768x768, 1024x1024 | arcface_128 | embedding_norm |
| hyperswap_1b_256 | 256x256, 512x512, 768x768, 1024x1024 | arcface_128 | embedding_norm |
| hyperswap_1c_256 | 256x256, 512x512, 768x768, 1024x1024 | arcface_128 | embedding_norm |
| inswapper_128 | 128x128, 256x256, 384x384, 512x512, 768x768, 1024x1024 | arcface_128 | embedding_projected |
| inswapper_128_fp16 | 128x128, 256x256, 384x384, 512x512, 768x768, 1024x1024 | arcface_128 | embedding_projected |
| simswap_256 | 256x256, 512x512, 768x768, 1024x1024 | arcface_128 | embedding |
| simswap_unofficial_512 | 512x512, 768x768, 1024x1024 | arcface_128 | embedding |
| uniface_256 | 256x256, 512x512, 768x768, 1024x1024 | ffhq_512 | source_face |

## Error Responses

### Invalid Model

```json
{
  "status": "FAILED",
  "error": "Invalid face_swapper_model: 'invalid_model'. Valid: blendswap_256, ghost_1_256, ..."
}
```

### Invalid Resolution

```json
{
  "status": "FAILED",
  "error": "Model 'simswap_unofficial_512' does not support resolution '256x256'. Valid: 512x512, 768x768, 1024x1024"
}
```

### Invalid Weight

```json
{
  "status": "FAILED",
  "error": "face_swapper_weight must be between 0.0 and 1.0, got 1.5"
}
```

### Invalid Selector Mode

```json
{
  "status": "FAILED",
  "error": "face_selector_mode must be 'many' or 'one', got 'invalid'"
}
```

## Examples

### High quality single-face swap on a female face

```json
{
  "source_image": "base64...",
  "target_image": "base64...",
  "face_swapper_model": "simswap_unofficial_512",
  "face_swapper_resolution": "1024x1024",
  "face_selector_gender": "female",
  "face_selector_mode": "one",
  "face_selector_order": "best-worst"
}
```

### Fast inference with FP16

```json
{
  "source_image": "base64...",
  "target_image": "base64...",
  "face_swapper_model": "inswapper_128_fp16",
  "face_swapper_resolution": "256x256"
}
```

### Subtle blend with uniface

```json
{
  "source_image": "base64...",
  "target_image": "base64...",
  "face_swapper_model": "uniface_256",
  "face_swapper_weight": 0.7
}
```

### Swap only the largest face with soft mask edges

```json
{
  "source_image": "base64...",
  "target_image": "base64...",
  "face_swapper_model": "hyperswap_1a_256",
  "face_selector_mode": "one",
  "face_selector_order": "large-small",
  "face_mask_blur": 0.5
}
```

### Swap faces of people aged 25-45, with padded mask

```json
{
  "source_image": "base64...",
  "target_image": "base64...",
  "face_swapper_model": "simswap_256",
  "face_selector_age_start": 25,
  "face_selector_age_end": 45,
  "face_mask_padding": "10,10,15,10"
}
```

### Backward compatible (no new parameters)

```json
{
  "source_image": "base64...",
  "target_image": "base64..."
}
```

## Performance Considerations

- **First use delay:** First request with each model adds 3-5s one-time loading delay
- **Subsequent requests:** Models remain cached in VRAM for instant inference
- **Memory usage:** Each model uses ~300-500MB VRAM; embedding converters add ~21MB each
- **Resolution impact:** Higher resolutions provide better quality but slower inference
- **Face selector:** Filtering by gender/age adds negligible overhead (attributes already computed by face detection)

## Backward Compatibility

All existing API requests work without changes. If new parameters are omitted:

| Parameter | Default |
|---|---|
| face_swapper_model | inswapper_128 |
| face_swapper_resolution | 512x512 |
| face_swapper_weight | 1.0 |
| face_mask_blur | 0.3 |
| face_mask_padding | "0,0,0,0" |
| face_selector_mode | many |
| face_selector_order | left-right |
| face_selector_gender | null (no filter) |
| face_selector_age_start | null (no minimum) |
| face_selector_age_end | null (no maximum) |

## Acknowledgements

Face swap models and preprocessing pipeline adapted from [FaceFusion](https://github.com/facefusion/facefusion).
