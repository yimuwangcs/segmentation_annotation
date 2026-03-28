import base64
import hashlib
import io
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image

try:
    import torch
    from segment_anything import SamPredictor, sam_model_registry
except Exception as exc:  # pragma: no cover
    torch = None
    SamPredictor = None
    sam_model_registry = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


MODEL_TYPE = os.getenv("SAM_MODEL_TYPE", "vit_b")
CHECKPOINT_PATH = os.getenv("SAM_CHECKPOINT", "./models/sam_vit_b_01ec64.pth")
DEVICE = os.getenv("SAM_DEVICE", "cuda" if torch and torch.cuda.is_available() else "cpu")


class PredictRequest(BaseModel):
    image: str = Field(..., description="data URL encoded image")
    points: list[list[float]]
    labels: list[int]


class PredictResponse(BaseModel):
    mask: str
    score: float


@dataclass
class PredictorCache:
    image_hash: Optional[str] = None
    predictor: Optional["SamPredictor"] = None


cache = PredictorCache()
app = FastAPI(title="Local SAM Service")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def ensure_predictor() -> "SamPredictor":
    if IMPORT_ERROR:
      raise RuntimeError(
          "Missing dependencies. Install fastapi, uvicorn, pillow, numpy, torch, and segment-anything."
      ) from IMPORT_ERROR
    if not os.path.exists(CHECKPOINT_PATH):
        raise RuntimeError(f"SAM checkpoint not found: {CHECKPOINT_PATH}")
    if cache.predictor is None:
        sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
        sam.to(device=DEVICE)
        cache.predictor = SamPredictor(sam)
    return cache.predictor


def decode_image(data_url: str) -> tuple[np.ndarray, str]:
    if "," not in data_url:
        raise ValueError("Expected a base64 data URL.")
    _, encoded = data_url.split(",", 1)
    raw = base64.b64decode(encoded)
    image_hash = hashlib.sha256(raw).hexdigest()
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    return np.array(image), image_hash


def mask_to_data_url(mask: np.ndarray) -> str:
    binary = (mask > 0).astype(np.uint8) * 255
    rgba = np.zeros((binary.shape[0], binary.shape[1], 4), dtype=np.uint8)
    rgba[..., 0] = 255
    rgba[..., 1] = 255
    rgba[..., 2] = 255
    rgba[..., 3] = binary
    image = Image.fromarray(rgba, mode="RGBA")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


@app.get("/health")
def health() -> dict:
    ready = IMPORT_ERROR is None and os.path.exists(CHECKPOINT_PATH)
    return {
        "ready": ready,
        "model_type": MODEL_TYPE,
        "checkpoint": CHECKPOINT_PATH,
        "device": DEVICE,
        "error": None if ready else str(IMPORT_ERROR or f"Checkpoint missing: {CHECKPOINT_PATH}"),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    if len(payload.points) != len(payload.labels):
        raise HTTPException(status_code=400, detail="points and labels length mismatch")
    if not payload.points:
        raise HTTPException(status_code=400, detail="at least one prompt point is required")

    try:
        predictor = ensure_predictor()
        image, image_hash = decode_image(payload.image)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    try:
        if cache.image_hash != image_hash:
            predictor.set_image(image)
            cache.image_hash = image_hash

        point_coords = np.array(payload.points, dtype=np.float32)
        point_labels = np.array(payload.labels, dtype=np.int32)
        masks, scores, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
        )
        best_index = int(np.argmax(scores))
        return PredictResponse(mask=mask_to_data_url(masks[best_index]), score=float(scores[best_index]))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

