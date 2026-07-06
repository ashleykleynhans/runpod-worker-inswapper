# Changelog

All notable changes to this project will be documented in this file.

## [6.0.0] - 2026-07-06

### Added
- Support for 12 additional face swap models from FaceFusion
- `face_swapper_model` parameter for model selection (13 models available)
- `face_swapper_resolution` parameter for quality control (128x128 to 1024x1024)
- `face_swapper_weight` parameter for blend control (0.0-1.0)
- FP16 model support for faster inference (inswapper_128_fp16)
- Lazy model loading for efficient memory usage

### Changed
- Docker image size increased to ~15GB (includes all 13 models)

### Backward Compatibility
- All existing API requests work unchanged (default: inswapper_128 at 512x512)
- No breaking changes to API schema
