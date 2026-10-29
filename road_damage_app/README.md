# Road Damage Detection — Deployment App

```
Road image / video / webcam
          |
          v
   Web Dashboard  (app/static: HTML + JS, draws boxes on a canvas)
          |   POST /api/detect  (multipart image + conf + iou)
          v
   FastAPI Backend (app/main.py)
          |
          v
   YOLO model     (app/detector.py, loads models/best.pt once)
          |
          v
   JSON results: boxes, class, confidence, counts per class, timing
```

## Setup

1. Download the best model from the Kaggle notebook **Output**
   (`road_damage/runs_v3/<model>/weights/best.pt`) and put it in `models/best.pt`.
2. Install. With an NVIDIA GPU (e.g. RTX 3060) use the CUDA build of torch; the app then runs on the GPU in FP16 automatically:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
```

No GPU: use `--index-url https://download.pytorch.org/whl/cpu` instead.

## Run

```bash
python run.py
```

(or `uvicorn app.main:app --reload`). If `models/best.pt` is missing, `run.py` falls back to the
pretrained `yolo11n.pt` (COCO classes) so the dashboard can still be tried. The header badge shows
the model and device, e.g. `best.pt · NVIDIA GeForce RTX 3060 Laptop GPU (FP16)`.

- Dashboard: http://localhost:8000
- API docs (Swagger): http://localhost:8000/docs

Use another model file with `MODEL_PATH`, e.g. the ONNX export:

```bash
set MODEL_PATH=models/best.onnx
uvicorn app.main:app
```

## API

| Method | Path | Description |
|---|---|---|
| GET | `/api/models` | All loaded models (id, label, device, classes) and the default one |
| GET | `/api/health` | Default model info and the list of model ids |
| POST | `/api/detect` | Form fields: `file` (image), `model` (id from `/api/models`, e.g. `yolov8s`, `yolo11s`, `yolo26s`; default model if empty), `conf` (0.01–1, default 0.25), `iou` (default 0.45), `imgsz` (320–1280, default 640) |

Example response:

```json
{
  "detections": [
    {"class_id": 2, "class_name": "Pothole", "confidence": 0.87, "box": [120.5, 340.0, 260.1, 420.7]}
  ],
  "counts": {"Crack": 0, "Manhole": 0, "Pothole": 1},
  "image": {"width": 640, "height": 640},
  "timing_ms": {"preprocess": 2.1, "inference": 45.3, "postprocess": 0.8},
  "model": "best.pt"
}
```

## Dashboard modes

- **Image**: upload or drag-and-drop a road image; changing the sliders re-runs detection.
- **Video**: pick a dashcam video; frames are sent to the API while it plays (real-time).
- **Webcam**: live camera frames, same loop.
- Live stats: detections, inference time, FPS, counts per class, detection table, download annotated result.

## Tests

```bash
pip install pytest httpx
pytest -q
```

Uses `models/best.pt` if present, otherwise the pretrained `yolo11n.pt` (downloaded automatically).

## Docker

```bash
docker build -t road-damage-app .
docker run -p 8000:8000 road-damage-app
```
