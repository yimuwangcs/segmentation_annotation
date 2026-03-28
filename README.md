# Interactive Segmentation Annotator

This repository provides a lightweight, browser-based image segmentation annotator with optional SAM-assisted mask generation.

## Features
- Brush/eraser painting with undo/redo
- Existing mask loading and mask export
- COCO-style RLE export (uncompressed counts array)
- SAM prompt mode (positive/negative point clicks)
- Per-image QA fields and summary JSON export

## Quick Start
Open the tool:
- Option A (recommended): run a local static server
  ```bash
  cd /Users/yimuwang/Downloads/Our_Benchmark
  python3 -m http.server 8000
  ```
  Then open `http://localhost:8000/`.

- Option B: open `index.html` directly (limited by browser file access rules).

## SAM Server
SAM runs locally via a FastAPI server.

Install dependencies:
```bash
pip install -r requirements-sam.txt
```

Download a SAM checkpoint and place it in `models/`:
```
models/sam_vit_b_01ec64.pth
```

Start the server:
```bash
uvicorn sam_server:app --host 127.0.0.1 --port 8001
```

You can also run both the server and static site with:
```bash
bash run.bash
```

## Output Files
- Mask PNG: `<image>_mask.png`
- RLE JSON (uncompressed COCO format): `<image>_mask_rle.json`
- Metadata JSON: `<image>_meta.json`
- Summary JSON: `annotations_summary.json`

## Summary JSON Schema
The summary file contains an array of entries:
```json
[
  {
    "image": {
      "image_id": 1,
      "width": 1600,
      "height": 900,
      "file_name": "example.jpg"
    },
    "question": "...",
    "answer": "...",
    "task_type": "...",
    "task_domain": "...",
    "num_instances": 1,
    "hallucination": 0,
    "annotations": [
      {
        "id": 1001,
        "segmentation": { "size": [900, 1600], "counts": [0, 42, 7, ...] },
        "bbox": [x, y, width, height],
        "area": 1234
      }
    ]
  }
]
```

Note: `segmentation.counts` uses uncompressed COCO RLE (counts array), not the compressed string format.
