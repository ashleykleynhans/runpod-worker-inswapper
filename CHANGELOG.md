# Changelog

All notable changes to this project will be documented in this file.

## [6.0.4] - 2026-07-08

### Fixed
- Rename error 'output' field to 'traceback' to avoid Runpod wrapper collision
- Restore runpod SDK pin to >=1.10.0

## [6.0.3] - 2026-07-08

### Fixed
- Remove refresh_worker from error responses to prevent silent failures
- Add handler catch-all try/except for unhandled exceptions

## [6.0.2] - 2026-07-07

### Fixed
- Bypass insightface ModelRouter for non-128x128 swapper models with custom _SwapperModel
- Add proper error handling for invalid ONNX, corrupt initializers, and inference failures

## [6.0.1] - 2026-07-07

### Fixed
- Pin runpod to 1.7.10 to fix refresh_worker swallowing error output
- Fix example script data paths for __file__-relative resolution
- Fix util.py COMPLETED status handling for missing output

## [6.0.0] - 2026-07-06

### Added
- Support for 8 additional face swap models from FaceFusion
- `face_swapper_model` parameter for model selection (9 models available: blendswap_256, ghost_1/2/3_256, inswapper_128/128_fp16, simswap_256/unofficial_512, uniface_256)
- `face_swapper_resolution` parameter for quality control (128x128 to 1024x1024)
- `face_swapper_weight` parameter for blend control (0.0-1.0)
- FP16 model support for faster inference (inswapper_128_fp16)
- Lazy model loading for efficient memory usage

### Changed
- Docker image size increased to ~10GB (includes all 9 models)

### Backward Compatibility
- All existing API requests work unchanged (default: inswapper_128 at 512x512)
- No breaking changes to API schema
