# Filter small faces using min_face_size parameter

## Overview

The `min_face_size` parameter allows you to filter out faces smaller than a specified
percentage of the source image's minimum dimension (width or height). This is useful when
you want to use only prominent faces in the source image and ignore smaller background faces.

## How it works

- `min_face_size` is specified as a percentage (0-100)
- The filter compares each detected face's bounding box dimensions in the source image against the threshold
- Faces in the source image with width OR height smaller than the threshold are excluded
- The threshold is calculated as: `min_dimension * (min_face_size / 100)`
- Where `min_dimension` is the smaller of the image's width or height

## Example Use Case

In an image with a large face in the foreground and smaller
faces in the background, setting `min_face_size: 30` will filter
out the smaller background faces, allowing you to target
the more predominant face for swapping.

## Request

In this example, setting `min_face_size` to `30` means faces must be at least
30% of the image's minimum dimension. This filters out small background faces
and only processes the main face in the image.

```json
{
  "input": {
    "source_image": "base64 encoded source image content",
    "target_image": "base64 encoded target image content",
    "source_indexes": "-1",
    "target_indexes": "-1",
    "background_enhance": true,
    "face_restore": true,
    "face_upsample": true,
    "upscale": 1,
    "codeformer_fidelity": 0.5,
    "output_format": "JPEG",
    "min_face_size": 30
  }
}
```

## Common Values

- `0` (default): No filtering, all detected faces are processed
- `10-20`: Filter very small faces (distant background faces)
- `20-30`: Filter small to medium faces (background characters)
- `30-50`: Only process large, prominent faces (main subjects)
- `50+`: Only process very large faces (close-up portraits)

## Response

### RUN

```json
{
  "id": "83bbc301-5dcd-4236-9293-a65cdd681858",
  "status": "IN_QUEUE"
}
```

### RUNSYNC

```json
{
  "delayTime": 20275,
  "executionTime": 43997,
  "id": "sync-a3b54383-e671-4e24-a7bd-c5fec16fda3b",
  "output": {
    "status": "ok",
    "image": "base64 encoded output image"
  },
  "status": "COMPLETED"
}
```
