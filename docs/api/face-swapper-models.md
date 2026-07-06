# Face Swapper Models API Documentation

## Overview

The Runpod Inswapper worker supports 13 face swap models with different quality/speed characteristics.

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

**Compatibility:** Not all models support all resolutions. See compatibility matrix below.

If not specified, automatically selects:
- `512x512` for `inswapper_128`, `inswapper_128_fp16`, `simswap_unofficial_512`
- `1024x1024` for all other models

### face_swapper_weight

**Type:** number  
**Required:** No  
**Default:** 1.0  
**Range:** 0.0 - 1.0

Controls blending between original and swapped face.

- `1.0` - Full face swap (100% swapped face)
- `0.7` - Subtle blend (70% swapped, 30% original)
- `0.5` - Half blend
- `0.0` - No swap (original face)

## Model Compatibility Matrix

| Model | Supported Resolutions |
|-------|----------------------|
| blendswap_256 | 256x256, 384x384, 512x512, 768x768, 1024x1024 |
| ghost_1_256 | 256x256, 512x512, 768x768, 1024x1024 |
| ghost_2_256 | 256x256, 512x512, 768x768, 1024x1024 |
| ghost_3_256 | 256x256, 512x512, 768x768, 1024x1024 |
| hififace_unofficial_256 | 256x256, 512x512, 768x768, 1024x1024 |
| hyperswap_1a_256 | 256x256, 512x512, 768x768, 1024x1024 |
| hyperswap_1b_256 | 256x256, 512x512, 768x768, 1024x1024 |
| hyperswap_1c_256 | 256x256, 512x512, 768x768, 1024x1024 |
| inswapper_128 | 128x128, 256x256, 384x384, 512x512, 768x768, 1024x1024 |
| inswapper_128_fp16 | 128x128, 256x256, 384x384, 512x512, 768x768, 1024x1024 |
| simswap_256 | 256x256, 512x512, 768x768, 1024x1024 |
| simswap_unofficial_512 | 512x512, 768x768, 1024x1024 |
| uniface_256 | 256x256, 512x512, 768x768, 1024x1024 |

## Error Responses

### Invalid Model

```json
{
  "status": "FAILED",
  "error": "Invalid face_swapper_model: 'invalid_model'. Valid options: blendswap_256, ghost_1_256, ..."
}
```

### Invalid Resolution

```json
{
  "status": "FAILED",
  "error": "Model 'simswap_unofficial_512' does not support resolution '256x256'. Valid resolutions: 512x512, 768x768, 1024x1024"
}
```

### Invalid Weight

```json
{
  "status": "FAILED",
  "error": "face_swapper_weight must be between 0.0 and 1.0, got 1.5"
}
```

## Performance Considerations

- **First use delay:** First request with each model adds 3-5s one-time loading delay
- **Subsequent requests:** Models remain cached in VRAM for instant inference
- **Memory usage:** Each model uses ~300-500MB VRAM
- **Resolution impact:** Higher resolutions provide better quality but slower inference

## Backward Compatibility

All existing API requests work without changes. If new parameters are omitted:
- Uses `inswapper_128` model
- Uses `512x512` resolution
- Uses `1.0` weight (full swap)

## Acknowledgements

Face swap models provided by [FaceFusion](https://github.com/facefusion/facefusion).
