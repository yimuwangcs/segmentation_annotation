import base64
import hashlib
import io
import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
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
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROGRESS_DIR = os.path.join(BASE_DIR, "progress")
PROGRESS_FILE = os.path.join(PROGRESS_DIR, "progress.json")
PROGRESS_IMAGES_DIR = os.path.join(PROGRESS_DIR, "images")
PROGRESS_MASKS_DIR = os.path.join(PROGRESS_DIR, "masks")
PROGRESS_SUMMARY_FILE = os.path.join(PROGRESS_DIR, "summary.json")
PROGRESS_RLE_DIR = os.path.join(PROGRESS_DIR, "rle")
PROGRESS_JSON_DIR = os.path.join(PROGRESS_DIR, "json")


class PredictRequest(BaseModel):
    image: str = Field(..., description="data URL encoded image")
    points: list[list[float]]
    labels: list[int]


class PredictResponse(BaseModel):
    mask: str
    score: float


class NuImagesListRequest(BaseModel):
    root_path: str
    channels: list[str]


class ProgressUpdateRequest(BaseModel):
    file_name: str
    status: str


class ProgressCommitRequest(BaseModel):
    file_name: str
    status: str
    masks: list[dict] | None = None
    width: int | None = None
    height: int | None = None
    summary_entry: dict | None = None


class SummaryAddRequest(BaseModel):
    file_name: str
    summary_entry: dict


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


def ensure_progress_store() -> None:
    os.makedirs(PROGRESS_DIR, exist_ok=True)
    os.makedirs(PROGRESS_IMAGES_DIR, exist_ok=True)
    os.makedirs(PROGRESS_MASKS_DIR, exist_ok=True)
    os.makedirs(PROGRESS_RLE_DIR, exist_ok=True)
    os.makedirs(PROGRESS_JSON_DIR, exist_ok=True)
    if not os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "w") as f:
            json.dump({"entries": {}}, f)
    if not os.path.exists(PROGRESS_SUMMARY_FILE):
        with open(PROGRESS_SUMMARY_FILE, "w") as f:
            json.dump({"entries": {}}, f)


def load_progress() -> dict:
    ensure_progress_store()
    with open(PROGRESS_FILE, "r") as f:
        return json.load(f)


def save_progress(data: dict) -> None:
    ensure_progress_store()
    with open(PROGRESS_FILE, "w") as f:
        json.dump(data, f, indent=2)


def load_summary() -> dict:
    ensure_progress_store()
    with open(PROGRESS_SUMMARY_FILE, "r") as f:
        return json.load(f)


def save_summary(data: dict) -> None:
    ensure_progress_store()
    with open(PROGRESS_SUMMARY_FILE, "w") as f:
        json.dump(data, f, indent=2)


def list_nuimages(root_path: str, channels: list[str]) -> list[dict]:
    results: list[dict] = []
    samples_dir = os.path.join(root_path, "samples")
    image_exts = {".jpg", ".jpeg", ".png"}

    if os.path.isdir(samples_dir):
        for channel in channels:
            channel_dir = os.path.join(samples_dir, channel)
            if not os.path.isdir(channel_dir):
                continue
            for name in sorted(os.listdir(channel_dir)):
                ext = os.path.splitext(name)[1].lower()
                if ext not in image_exts:
                    continue
                file_name = os.path.join("samples", channel, name)
                abs_path = os.path.join(root_path, file_name)
                results.append(
                    {
                        "file_name": file_name,
                        "abs_path": abs_path,
                        "channel": channel,
                        "split": None,
                        "sample_token": None,
                        "sample_data_token": None,
                    }
                )
        if results:
            return results

    splits = ["v1.0-train", "v1.0-val", "v1.0-test"]
    for split in splits:
        split_dir = os.path.join(root_path, split)
        sensor_path = os.path.join(split_dir, "sensor.json")
        sample_data_path = os.path.join(split_dir, "sample_data.json")
        if not os.path.exists(sensor_path) or not os.path.exists(sample_data_path):
            continue

        with open(sensor_path, "r") as f:
            sensors = json.load(f)
        sensor_map = {row["token"]: row["channel"] for row in sensors}

        with open(sample_data_path, "r") as f:
            sample_data = json.load(f)

        for row in sample_data:
            channel = sensor_map.get(row["sensor_token"])
            if channel not in channels:
                continue
            file_name = row["file_name"]
            abs_path = os.path.join(root_path, file_name)
            results.append(
                {
                    "file_name": file_name,
                    "abs_path": abs_path,
                    "channel": channel,
                    "split": split,
                    "sample_token": row.get("sample_token"),
                    "sample_data_token": row.get("token"),
                }
            )

    return results


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


@app.post("/nuimages/list")
def nuimages_list(payload: NuImagesListRequest) -> dict:
    if not payload.channels:
        raise HTTPException(status_code=400, detail="channels is required")
    if not os.path.exists(payload.root_path):
        raise HTTPException(status_code=400, detail=f"root_path not found: {payload.root_path}")
    try:
        data = list_nuimages(payload.root_path, payload.channels)
        return {"count": len(data), "items": data}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/nuimages/image")
def nuimages_image(path: str) -> FileResponse:
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="image not found")
    return FileResponse(path)


@app.get("/progress/status")
def progress_status(file_name: str) -> dict:
    data = load_progress()
    entry = data.get("entries", {}).get(file_name)
    return {"entry": entry}


@app.get("/progress/list")
def progress_list() -> dict:
    data = load_progress()
    return {"entries": data.get("entries", {})}


@app.get("/progress/summary")
def progress_summary() -> dict:
    summary = load_summary()
    entries = summary.get("entries", {})
    total = 0
    type_counts: dict[str, dict[str, int]] = {}
    for items in entries.values():
        if not isinstance(items, list):
            continue
        for item in items:
            total += 1
            task_type = item.get("task_type") or "Unknown"
            counts = type_counts.setdefault(task_type, {"total": 0, "hallucination": 0})
            counts["total"] += 1
            if int(item.get("hallucination", 0)) == 1:
                counts["hallucination"] += 1
    labeled_images = len([v for v in load_progress().get("entries", {}).values() if v.get("status") == "labeled"])
    return {"qa_total": total, "type_counts": type_counts, "labeled_images": labeled_images}


@app.post("/progress/update")
def progress_update(payload: ProgressUpdateRequest) -> dict:
    if payload.status not in {"labeled", "skipped"}:
        raise HTTPException(status_code=400, detail="status must be labeled or skipped")
    data = load_progress()
    data.setdefault("entries", {})[payload.file_name] = {
        "status": payload.status,
        "updated_at": datetime.utcnow().isoformat() + "Z",
    }
    save_progress(data)
    return {"ok": True}


@app.post("/progress/commit")
def progress_commit(payload: ProgressCommitRequest) -> dict:
    if payload.status not in {"labeled", "skipped"}:
        raise HTTPException(status_code=400, detail="status must be labeled or skipped")
    if not os.path.exists(payload.file_name):
        raise HTTPException(status_code=404, detail="image not found")

    ensure_progress_store()
    summary = load_summary()
    rel_key = os.path.basename(payload.file_name)
    if os.path.sep in payload.file_name:
        parts = payload.file_name.split(os.path.sep)
        if len(parts) >= 2:
            rel_key = os.path.join(parts[-2], parts[-1])
    summary_entries = summary.setdefault("entries", {}).setdefault(rel_key, [])
    qa_index = (
        sum(len(items) for items in summary.get("entries", {}).values() if isinstance(items, list)) + 1
    )
    is_hallucination = False
    if payload.summary_entry is not None:
        try:
            is_hallucination = int(payload.summary_entry.get("hallucination", 0)) == 1
        except Exception:
            is_hallucination = False

    image_basename = os.path.basename(payload.file_name)
    image_stem = os.path.splitext(image_basename)[0]
    image_out = os.path.join(PROGRESS_IMAGES_DIR, image_basename)

    try:
        if not os.path.exists(image_out):
            shutil.copy2(payload.file_name, image_out)
    except PermissionError:
        if not os.path.exists(image_out):
            raise

    mask_paths = []
    rle_paths = []
    masks = payload.masks or []
    if masks and not is_hallucination:
        for idx, item in enumerate(masks, start=1):
            data_url = item.get("data")
            label = item.get("label") or f"{idx}"
            safe_label = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(label))
            mask_out = os.path.join(PROGRESS_MASKS_DIR, f"{image_stem}_{qa_index}_mask_{safe_label}_{idx}.png")
            rle_out = os.path.join(PROGRESS_RLE_DIR, f"{image_stem}_{qa_index}_mask_{safe_label}_{idx}.json")
            if not data_url or "," not in data_url:
                raise HTTPException(status_code=400, detail="mask data must be a data URL")
            _, encoded = data_url.split(",", 1)
            raw = base64.b64decode(encoded)
            with open(mask_out, "wb") as f:
                f.write(raw)
            mask_paths.append(mask_out)
            if "rle" in item and item["rle"] is not None:
                with open(rle_out, "w") as f:
                    json.dump(item["rle"], f)
                rle_paths.append(rle_out)
    elif not is_hallucination:
        if payload.width is None or payload.height is None:
            raise HTTPException(status_code=400, detail="width/height required for empty mask")
        mask_out = os.path.join(PROGRESS_MASKS_DIR, f"{image_stem}_{qa_index}_mask.png")
        blank = Image.new("L", (payload.width, payload.height), 0)
        blank.save(mask_out, format="PNG")
        mask_paths.append(mask_out)

    progress = load_progress()
    progress.setdefault("entries", {})[payload.file_name] = {
        "status": payload.status,
        "updated_at": datetime.utcnow().isoformat() + "Z",
    }
    save_progress(progress)

    if payload.summary_entry is not None:
        rel_mask_paths = [os.path.relpath(p, PROGRESS_DIR) for p in mask_paths]
        rel_rle_paths = [os.path.relpath(p, PROGRESS_DIR) for p in rle_paths]
        summary_entry = dict(payload.summary_entry)
        summary_entry["qa_index"] = qa_index
        summary_entry["mask_paths"] = rel_mask_paths
        summary_entry["rle_paths"] = rel_rle_paths

        image_path_slug = rel_key.replace(os.sep, "__")
        question_type = (summary_entry.get("task_type") or "Unknown").replace(" ", "_").replace("/", "_")
        hallucination_flag = summary_entry.get("hallucination", 0)
        json_name = f"{image_path_slug}_{question_type}_{qa_index}_{hallucination_flag}.json"
        json_path = os.path.join(PROGRESS_JSON_DIR, json_name)
        rel_json_path = os.path.relpath(json_path, PROGRESS_DIR)
        summary_entry["json_path"] = rel_json_path
        summary_entries.append(summary_entry)
        save_summary(summary)

        with open(json_path, "w") as f:
            json.dump(summary_entry, f, indent=2)

    return {"ok": True, "image_path": image_out}


@app.get("/progress/summary_data")
def progress_summary_data() -> dict:
    summary = load_summary()
    return summary


@app.post("/progress/summary_add")
def progress_summary_add(payload: SummaryAddRequest) -> dict:
    summary = load_summary()
    summary.setdefault("entries", {}).setdefault(payload.file_name, []).append(payload.summary_entry)
    save_summary(summary)
    return {"ok": True}


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
