# Phase 1: Alternative Face Swap Models - Design Specification

**Date:** 2026-07-06  
**Status:** Approved  
**Phase:** 1 of 5

## Overview

Enhance the runpod-worker-inswapper serverless worker by adding support for 12 additional face swap models from FaceFusion, enabling users to choose models based on quality/speed trade-offs. This is Phase 1 of a multi-phase enhancement plan to incorporate FaceFusion capabilities.

## Goals

- Add 12 new face swap models from FaceFusion (13 total including existing inswapper_128)
- Support user-selectable model and resolution via API parameters
- Add face blending capability (weight parameter 0.0-1.0)
- Include FP16 model variants for faster inference
- Maintain full backward compatibility with existing API
- Keep models bundled in Docker image for predictable cold starts

## Non-Goals (Future Phases)

- Phase 2: Enhanced face detection (multiple detector models)
- Phase 3: Advanced face selection (gender, race, quality-based)
- Phase 4: Face masking capabilities
- Phase 5: Multiple enhancement models (GFPGAN, GPEN, RestoreFormer)

## API Changes

### New Optional Parameters

Three new optional parameters added to INPUT_SCHEMA:

**face_swapper_model** (string, optional, default: "inswapper_128")
- Valid values: blendswap_256, ghost_1_256, ghost_2_256, ghost_3_256, hififace_unofficial_256, hyperswap_1a_256, hyperswap_1b_256, hyperswap_1c_256, inswapper_128, inswapper_128_fp16, simswap_256, simswap_unofficial_512, uniface_256
- Selects which face swap model to use

**face_swapper_resolution** (string, optional, default: auto)
- Valid values: 128x128, 256x256, 384x384, 512x512, 768x768, 1024x1024
- Must be compatible with selected model
- Auto-selects model's maximum resolution if not specified

**face_swapper_weight** (number, optional, default: 1.0)
- Range: 0.0 to 1.0
- Controls blending between original and swapped face
- 1.0 = 100% swapped face (current behavior)
- 0.5 = 50/50 blend
- 0.0 = original face (no swap)

### Example API Requests

**Using new model with explicit resolution:**
```json
{
  "source_image": "base64...",
  "target_image": "base64...",
  "face_swapper_model": "simswap_256",
  "face_swapper_resolution": "512x512",
  "face_swapper_weight": 1.0
}
```

**Using FP16 for faster inference:**
```json
{
  "source_image": "base64...",
  "target_image": "base64...",
  "face_swapper_model": "inswapper_128_fp16",
  "face_swapper_resolution": "256x256"
}
```

**Subtle blending:**
```json
{
  "source_image": "base64...",
  "target_image": "base64...",
  "face_swapper_model": "uniface_256",
  "face_swapper_weight": 0.7
}
```

**Backward compatible (no changes):**
```json
{
  "source_image": "base64...",
  "target_image": "base64..."
}
```

### Validation Rules

- If model is invalid: return error with list of valid models
- If resolution is invalid for model: return error with valid resolutions for that model
- If weight is out of range: return error
- If resolution not specified: use default resolution for model

### Backward Compatibility

- All existing API requests work unchanged
- Default model: inswapper_128
- Default resolution: 512x512 (for inswapper_128)
- Default weight: 1.0 (full swap)
- No breaking changes to existing integrations


## Architecture

### Approach: Incremental Enhancement

Extend existing handler architecture minimally while adding new capabilities.

**Key principles:**
- Keep existing swap_face() logic untouched for backward compatibility
- Add new swap_face_enhanced() for new models and features
- Lazy load models on first use (fast cold starts, efficient memory)
- Fork FaceFusion's face_swapper preprocessing logic
- Validate parameters early with clear error messages

### File Structure

**New files:**
- `face_swapper.py` - Enhanced face swapping logic (forked from FaceFusion)
- `face_swapper_models.py` - Model definitions and validation logic

**Modified files:**
- `handler.py` - Integrate new model selection and routing logic
- `schemas/input.py` - Add new optional parameters
- `Dockerfile` - Download and bundle all 13 models
- `README.md` - Document new features
- `requirements.txt` - Add any new dependencies

**Unchanged files:**
- `restoration.py` - Face restoration logic unchanged
- Tests for existing functionality - Should all pass

### Model Configuration

Model compatibility matrix defined in `face_swapper_models.py`:

```python
FACE_SWAPPER_MODEL_SET = {
    'blendswap_256': ['256x256', '384x384', '512x512', '768x768', '1024x1024'],
    'ghost_1_256': ['256x256', '512x512', '768x768', '1024x1024'],
    'ghost_2_256': ['256x256', '512x512', '768x768', '1024x1024'],
    'ghost_3_256': ['256x256', '512x512', '768x768', '1024x1024'],
    'hififace_unofficial_256': ['256x256', '512x512', '768x768', '1024x1024'],
    'hyperswap_1a_256': ['256x256', '512x512', '768x768', '1024x1024'],
    'hyperswap_1b_256': ['256x256', '512x512', '768x768', '1024x1024'],
    'hyperswap_1c_256': ['256x256', '512x512', '768x768', '1024x1024'],
    'inswapper_128': ['128x128', '256x256', '384x384', '512x512', '768x768', '1024x1024'],
    'inswapper_128_fp16': ['128x128', '256x256', '384x384', '512x512', '768x768', '1024x1024'],
    'simswap_256': ['256x256', '512x512', '768x768', '1024x1024'],
    'simswap_unofficial_512': ['512x512', '768x768', '1024x1024'],
    'uniface_256': ['256x256', '512x512', '768x768', '1024x1024']
}

DEFAULT_RESOLUTIONS = {
    'inswapper_128': '512x512',
    'inswapper_128_fp16': '512x512',
    'simswap_unofficial_512': '512x512',
    'default': '1024x1024'  # For all other models
}
```

### Model Loading Strategy

**Lazy loading pattern:**

```python
# Global model cache
FACE_SWAPPER_MODELS = {}

def get_face_swapper_model(model_name: str):
    """Load model on first use, cache for subsequent calls"""
    if model_name not in FACE_SWAPPER_MODELS:
        model_path = f'checkpoints/face_swapper/{model_name}.onnx'
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        FACE_SWAPPER_MODELS[model_name] = insightface.model_zoo.get_model(model_path)
        logger.info(f"Loaded face swapper model: {model_name}")
    return FACE_SWAPPER_MODELS[model_name]
```

**Benefits:**
- Fast cold starts (<10s worker ready)
- Only loads models actually requested
- Each model ~300-500MB VRAM
- Loaded models persist for subsequent requests

### Handler Integration

Modified `handler.py` process() function flow:

```python
def process(job_id, source_img, target_img, source_indexes, target_indexes, min_face_size):
    # 1. Extract parameters (NEW)
    model_name = job.get('face_swapper_model', 'inswapper_128')
    resolution = job.get('face_swapper_resolution') or get_default_resolution(model_name)
    weight = job.get('face_swapper_weight', 1.0)
    
    # 2. Validate parameters (NEW)
    validate_face_swapper_params(model_name, resolution)
    
    # 3. Existing face detection (UNCHANGED)
    source_faces = get_many_faces(face_analyser, source_img, min_face_size)
    target_faces = get_many_faces(face_analyser, target_img, min_face_size)
    
    # 4. Load model (NEW - lazy)
    model = get_face_swapper_model(model_name)
    
    # 5. Face swapping - route to appropriate function
    if model_name == 'inswapper_128' and weight == 1.0:
        # Use existing swap_face() for exact backward compat
        result = swap_face(source_faces, target_faces, source_index, target_index, temp_frame)
    else:
        # Use new enhanced swapper
        result = swap_face_enhanced(
            source_faces[source_index],
            target_faces[target_index],
            temp_frame,
            model,
            resolution,
            weight
        )
    
    # 6. Existing restoration logic (UNCHANGED)
    if do_restoration:
        result = restoration.enhance_face(result, ...)
    
    return result
```

### Face Swapping Logic

New `face_swapper.py` module (forked from FaceFusion):

**Core function:**
```python
def swap_face_enhanced(
    source_face,
    target_face,
    temp_frame: np.ndarray,
    model,
    resolution: str,
    weight: float = 1.0
) -> np.ndarray:
    """
    Enhanced face swapping with model selection and blending.
    
    Steps:
    1. Extract target face region
    2. Warp source face to target alignment
    3. Run model inference at specified resolution
    4. Blend result with original using weight
    5. Paste back into frame with smooth boundaries
    """
    # Face preprocessing (from FaceFusion)
    # - Affine transformation to align faces
    # - Normalize to model's expected input size
    # - Handle different resolutions per model
    
    # Model inference
    # - Run ONNX model
    # - Post-process output
    
    # Blending (NEW feature)
    if weight < 1.0:
        swapped = cv2.addWeighted(original_face, 1.0 - weight, swapped_face, weight, 0)
    
    # Paste back with feathering
    # - Smooth boundaries
    # - Color correction
    
    return result_frame
```

**Key implementation details:**
- Copy FaceFusion's face warping/alignment logic
- Support different input resolutions per model
- Implement weight blending via cv2.addWeighted
- Handle edge cases (face too small, alignment failures)


## Docker Build

### Model Acquisition

Download all 13 face swap models during Docker build using wget from FaceFusion's GitHub releases.

**Modified Dockerfile section:**

```dockerfile
ARG CUDA_VERSION="12.4.1"
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu22.04

# ... existing setup unchanged ...

# Download insightface checkpoints and NEW face swapper models
RUN cd /workspace/runpod-worker-inswapper && \
    mkdir -p checkpoints/face_swapper && \
    mkdir -p checkpoints/models && \
    cd checkpoints && \
    # Existing inswapper_128 (keep for backward compat)
    wget -O inswapper_128.onnx "https://huggingface.co/ashleykleynhans/inswapper/resolve/main/inswapper_128.onnx?download=true" && \
    # NEW: Download all face swapper models
    cd face_swapper && \
    wget -O blendswap_256.onnx "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/blendswap_256.onnx" && \
    wget -O ghost_1_256.onnx "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/ghost_unet_1_256.onnx" && \
    wget -O ghost_2_256.onnx "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/ghost_unet_2_256.onnx" && \
    wget -O ghost_3_256.onnx "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/ghost_unet_3_256.onnx" && \
    wget -O hififace_unofficial_256.onnx "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/hififace_unofficial_256.onnx" && \
    wget -O hyperswap_1a_256.onnx "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/hyperswap_unet_1a_256.onnx" && \
    wget -O hyperswap_1b_256.onnx "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/hyperswap_unet_1b_256.onnx" && \
    wget -O hyperswap_1c_256.onnx "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/hyperswap_unet_1c_256.onnx" && \
    wget -O inswapper_128_fp16.onnx "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/inswapper_128_fp16.onnx" && \
    wget -O simswap_256.onnx "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/simswap_256.onnx" && \
    wget -O simswap_unofficial_512.onnx "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/simswap_unofficial_512.onnx" && \
    wget -O uniface_256.onnx "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/uniface_256.onnx" && \
    cd ../models && \
    # Existing buffalo_l download unchanged
    wget https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip && \
    mkdir buffalo_l && \
    cd buffalo_l && \
    unzip ../buffalo_l.zip

# ... existing CodeFormer installation unchanged ...

# Copy handler files (MODIFIED)
COPY --chmod=755 handler.py /workspace/runpod-worker-inswapper/handler.py
COPY --chmod=755 face_swapper.py /workspace/runpod-worker-inswapper/face_swapper.py
COPY --chmod=755 face_swapper_models.py /workspace/runpod-worker-inswapper/face_swapper_models.py

# ... existing start.sh and ENTRYPOINT unchanged ...
```

### Model Storage Layout

```
/workspace/runpod-worker-inswapper/checkpoints/
├── inswapper_128.onnx                    # Existing, for backward compat
├── face_swapper/                         # NEW directory
│   ├── blendswap_256.onnx
│   ├── ghost_1_256.onnx
│   ├── ghost_2_256.onnx
│   ├── ghost_3_256.onnx
│   ├── hififace_unofficial_256.onnx
│   ├── hyperswap_1a_256.onnx
│   ├── hyperswap_1b_256.onnx
│   ├── hyperswap_1c_256.onnx
│   ├── inswapper_128_fp16.onnx
│   ├── simswap_256.onnx
│   ├── simswap_unofficial_512.onnx
│   └── uniface_256.onnx
└── models/
    └── buffalo_l/                        # Existing face detection
```

### Image Size Impact

- Current image: ~4GB
- With all 13 models: ~14-16GB
- Each model: ~800MB-1.2GB
- Still reasonable for Runpod deployment
- One-time pull cost per worker node

## Error Handling

### Validation Functions

```python
def validate_face_swapper_params(model_name: str, resolution: str) -> None:
    """Validate model and resolution compatibility"""
    if model_name not in FACE_SWAPPER_MODEL_SET:
        valid_models = ', '.join(FACE_SWAPPER_MODEL_SET.keys())
        raise ValueError(
            f"Invalid face_swapper_model: '{model_name}'. "
            f"Valid options: {valid_models}"
        )
    
    if resolution not in FACE_SWAPPER_MODEL_SET[model_name]:
        valid_resolutions = ', '.join(FACE_SWAPPER_MODEL_SET[model_name])
        raise ValueError(
            f"Model '{model_name}' does not support resolution '{resolution}'. "
            f"Valid resolutions for this model: {valid_resolutions}"
        )

def get_default_resolution(model_name: str) -> str:
    """Get default resolution for a model"""
    return DEFAULT_RESOLUTIONS.get(model_name, DEFAULT_RESOLUTIONS['default'])
```

### Error Responses

**Invalid model:**
```json
{
  "status": "FAILED",
  "error": "Invalid face_swapper_model: 'invalid_model'. Valid options: blendswap_256, ghost_1_256, ..."
}
```

**Invalid resolution for model:**
```json
{
  "status": "FAILED",
  "error": "Model 'simswap_unofficial_512' does not support resolution '256x256'. Valid resolutions: 512x512, 768x768, 1024x1024"
}
```

**Model file missing:**
```json
{
  "status": "FAILED",
  "error": "Model file not found: checkpoints/face_swapper/simswap_256.onnx"
}
```

### Graceful Degradation

- Validate parameters before model loading
- Clear error messages with actionable information
- Log all model loading events for debugging
- If VRAM exhausted: clear least-recently-used model and retry (future optimization)


## Testing Strategy

### Unit Tests

New test file: `tests/test_face_swapper_models.py`

```python
def test_validate_model_resolution_valid():
    """Valid model/resolution combinations should pass"""
    validate_face_swapper_params('simswap_256', '512x512')
    validate_face_swapper_params('inswapper_128', '128x128')
    # Should not raise

def test_validate_model_resolution_invalid_model():
    """Invalid model should raise with clear error"""
    with pytest.raises(ValueError, match="Invalid face_swapper_model"):
        validate_face_swapper_params('invalid_model', '512x512')

def test_validate_model_resolution_invalid_resolution():
    """Invalid resolution for model should raise"""
    with pytest.raises(ValueError, match="does not support resolution"):
        validate_face_swapper_params('simswap_unofficial_512', '256x256')

def test_get_default_resolution():
    """Default resolutions should be correct"""
    assert get_default_resolution('inswapper_128') == '512x512'
    assert get_default_resolution('simswap_256') == '1024x1024'
    assert get_default_resolution('simswap_unofficial_512') == '512x512'

def test_lazy_model_loading():
    """Models should load once and be cached"""
    model1 = get_face_swapper_model('simswap_256')
    model2 = get_face_swapper_model('simswap_256')
    assert model1 is model2  # Same instance from cache
```

### Integration Tests

New test file: `tests/test_handler_face_swapper.py`

```python
def test_face_swap_with_simswap():
    """Test API with simswap_256 model"""
    job = {
        "source_image": base64_source,
        "target_image": base64_target,
        "face_swapper_model": "simswap_256",
        "face_swapper_resolution": "512x512"
    }
    result = handler(job)
    assert result['status'] == 'success'
    assert 'output' in result

def test_face_swap_with_fp16():
    """Test FP16 model"""
    job = {
        "source_image": base64_source,
        "target_image": base64_target,
        "face_swapper_model": "inswapper_128_fp16",
        "face_swapper_resolution": "256x256"
    }
    result = handler(job)
    assert result['status'] == 'success'

def test_backward_compatibility():
    """Existing API format should work unchanged"""
    job = {
        "source_image": base64_source,
        "target_image": base64_target
    }
    result = handler(job)
    assert result['status'] == 'success'
    # Should use inswapper_128 at 512x512 by default

def test_weight_blending_full():
    """Test weight=1.0 (full swap)"""
    job = {
        "source_image": base64_source,
        "target_image": base64_target,
        "face_swapper_weight": 1.0
    }
    result = handler(job)
    assert result['status'] == 'success'

def test_weight_blending_half():
    """Test weight=0.5 (half blend)"""
    job = {
        "source_image": base64_source,
        "target_image": base64_target,
        "face_swapper_model": "uniface_256",
        "face_swapper_weight": 0.5
    }
    result = handler(job)
    assert result['status'] == 'success'

def test_invalid_model_error():
    """Invalid model should return error"""
    job = {
        "source_image": base64_source,
        "target_image": base64_target,
        "face_swapper_model": "invalid_model"
    }
    result = handler(job)
    assert result['status'] == 'error'
    assert 'Invalid face_swapper_model' in result['error']

def test_invalid_resolution_error():
    """Invalid resolution should return error"""
    job = {
        "source_image": base64_source,
        "target_image": base64_target,
        "face_swapper_model": "simswap_unofficial_512",
        "face_swapper_resolution": "128x128"
    }
    result = handler(job)
    assert result['status'] == 'error'
    assert 'does not support resolution' in result['error']
```

### Manual Testing Checklist

- [ ] Test each of 13 models with sample images
- [ ] Verify resolution scaling (128x128 vs 1024x1024 visual quality)
- [ ] Test weight blending (0.0, 0.3, 0.5, 0.7, 1.0 visual comparison)
- [ ] Verify FP16 models work on H100/A100 GPUs
- [ ] Test invalid model/resolution combinations return proper errors
- [ ] Benchmark inference times per model/resolution
- [ ] Memory usage monitoring (lazy loading effectiveness)
- [ ] Cold start time measurement
- [ ] Stress test: multiple concurrent jobs with different models

### Regression Testing

All existing test suites must pass:
- [ ] tests/test_handler.py - unchanged
- [ ] tests/test_restoration.py - unchanged
- [ ] tests/test_schemas.py - schema validation still works
- [ ] tests/test_handler_process.py - existing process() tests pass

## Documentation

### README.md Updates

Add new section after "Model" section:

```markdown
## Face Swapper Models

The worker supports 13 different face swap models with varying quality/speed trade-offs.

### Available Models

- `inswapper_128` (default) - Original balanced quality/speed model
- `inswapper_128_fp16` - Faster FP16 version
- `simswap_256` - High quality
- `simswap_unofficial_512` - Highest quality (slower)
- `ghost_1_256`, `ghost_2_256`, `ghost_3_256` - Ghost model variants
- `hyperswap_1a_256`, `hyperswap_1b_256`, `hyperswap_1c_256` - Hyperswap variants
- `blendswap_256` - Blend-focused model
- `uniface_256` - Universal face model
- `hififace_unofficial_256` - High fidelity variant

### Model Selection Examples

See [Face Swapper Models API Documentation](docs/api/face-swapper-models.md) for complete reference.

**High quality swap:**
```json
{
  "face_swapper_model": "simswap_unofficial_512",
  "face_swapper_resolution": "1024x1024"
}
```

**Fast inference:**
```json
{
  "face_swapper_model": "inswapper_128_fp16",
  "face_swapper_resolution": "256x256"
}
```

**Subtle blend:**
```json
{
  "face_swapper_model": "uniface_256",
  "face_swapper_weight": 0.7
}
```
```

### New API Documentation

Create `docs/api/face-swapper-models.md`:

- Complete parameter reference
- Model comparison table (quality/speed/memory)
- Resolution recommendations per model
- Weight blending examples with visual comparisons
- Performance benchmarks
- Troubleshooting common issues

### CHANGELOG.md Entry

```markdown
## [2.0.0] - 2026-07-XX

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
```

### Migration Guide

Create `docs/MIGRATION.md`:

```markdown
## Migration to v2.0.0

### No Action Required
All existing API integrations continue to work without changes.

### Optional: Upgrade to Better Models

To use higher quality models, add these parameters to your requests:

1. **Best quality:** 
   - Model: `simswap_unofficial_512`
   - Resolution: `1024x1024`
   - Note: ~2x slower than inswapper_128

2. **Best speed:**
   - Model: `inswapper_128_fp16`
   - Resolution: `256x256`
   - Note: ~1.5x faster than inswapper_128

3. **Balanced:**
   - Model: `simswap_256`
   - Resolution: `512x512`
   - Note: Better quality, similar speed

### First Use Delay

First request with each new model adds 3-5s one-time loading delay. Subsequent requests with same model are instant.
```

## Implementation Plan Handoff

After this spec is approved, the next step is to invoke the `writing-plans` skill to create a detailed implementation plan with:

- File-by-file implementation steps
- Code snippets for each module
- Testing checkpoints
- Deployment steps
- Rollback plan

## Open Questions

None - all design decisions have been made.

## Risks and Mitigations

**Risk:** Docker image size increase (4GB → 15GB)
**Mitigation:** Acceptable for Runpod, one-time pull cost per node

**Risk:** Model quality differences may surprise users
**Mitigation:** Clear documentation with visual examples and benchmarks

**Risk:** First use of new model adds 3-5s delay
**Mitigation:** Document this clearly, optimize model loading in future

**Risk:** FaceFusion model URLs may change
**Mitigation:** Pin to specific release version (models-3.0.0), can update as needed

## Success Criteria

- [ ] All 13 models successfully load and perform face swapping
- [ ] Backward compatibility maintained (all existing tests pass)
- [ ] API validation provides clear error messages
- [ ] Documentation complete with examples
- [ ] Docker image builds successfully
- [ ] Performance acceptable (lazy loading works as expected)
- [ ] Ready for Phase 2 (enhanced face detection)

## Acknowledgements

This design incorporates face swap models and preprocessing logic from FaceFusion (https://github.com/facefusion/facefusion), an industry-leading face manipulation platform. Appropriate attribution will be added to source code and documentation.

