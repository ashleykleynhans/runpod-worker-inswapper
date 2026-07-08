# Changelog

All notable changes to this project will be documented in this file.

## [6.0.12] - 2026-07-08

### Added
- Face mask controls: face_mask_blur and face_mask_padding API params
- Face selector: mode, order, gender, age range API params
- select_faces() utility for filtering/sorting detected faces
- Gender and age filtering via insightface buffalo_l attributes

## [6.0.11] - 2026-07-08

### Added
- hyperswap_1a/1b/1c_256 and hififace_unofficial_256 models
- mtcnn_512 warp template for hififace
- crossface_hififace embedding converter

### Fixed
- Use raw .embedding instead of .normed_embedding matching FaceFusion
- Handle hyperswap reversed ONNX inputs (embedding first)

## [6.0.10] - 2026-07-08

### Changed
- Fork FaceFusion's full pipeline: cv2.estimateAffinePartial2D + RANSAC warp
- Add per-model warp templates (arcface_128, arcface_112_v1, ffhq_512)
- Replace insightface diff-mask paste-back with FaceFusion box-mask + alpha blend
- Fix ghost models to use arcface_112_v1 template (not arcface_128)
- 100% test coverage (135 passed, 4 skipped)

## [6.0.9] - 2026-07-08

### Changed
- Move CodeFormer weight downloads into scripts/download_models.py
- check_ckpts() now validates instead of downloading on the fly
- Remove basicsr scipy dependency from requirements

## [6.0.7] - 2026-07-08

### Fixed
- Pin runpod SDK to version 1.7.10, because 1.10.1 is not picking upjobs from the queue and they just remain on IN_QUEUE status.

## [6.0.6] - 2026-07-08

### Fixed
- Fix UnboundLocalError in download_models.py

### Added
- OCI image labels and annotations in Dockerfile and docker-bake.hcl

## [6.0.5] - 2026-07-08

### Changed
- Consolidate all face swapper models into checkpoints/face_swapper/
- Replace Dockerfile wget downloads with scripts/download_models.py (tqdm)
- Cache model downloads layer before repo clone for faster rebuilds

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
