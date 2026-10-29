"""FastAPI backend for the road damage detection dashboard.

Run:  python run.py        (from the road_damage_app folder)
Docs: http://localhost:8000/docs

Every *.pt / *.onnx file in models/ is loaded at startup; the dashboard picks one per request.
"""
import io
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError

from .detector import Detector

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = Path(os.getenv("MODELS_DIR", BASE_DIR.parent / "models"))
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
MODEL_ORDER = ["yolov8s", "yolo11s", "yolo26s"]

detectors: dict[str, Detector] = {}
default_model: str | None = None


def find_model_files() -> list[str]:
    # MODEL_PATH overrides the folder scan (e.g. MODEL_PATH=yolo11n.pt for a quick demo)
    if os.getenv("MODEL_PATH"):
        return [os.environ["MODEL_PATH"]]
    files = [p for p in MODELS_DIR.iterdir() if p.suffix in (".pt", ".onnx")] if MODELS_DIR.exists() else []
    files.sort(key=lambda p: (MODEL_ORDER.index(p.stem) if p.stem in MODEL_ORDER else len(MODEL_ORDER), p.name))
    if not files:
        raise RuntimeError(f"No .pt/.onnx model found in {MODELS_DIR}")
    return [str(p) for p in files]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global default_model
    for path in find_model_files():
        d = Detector(path)
        detectors[d.id] = d
    wanted = os.getenv("DEFAULT_MODEL", "yolo26s")  # best overall in evaluation/comparison
    default_model = wanted if wanted in detectors else next(iter(detectors))
    yield
    detectors.clear()


app = FastAPI(
    title="Road Damage Detection API",
    description="Detects Crack, Manhole and Pothole in road images with trained YOLO models (YOLOv8s, YOLO11s, YOLO26s).",
    version="1.1.0",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


@app.middleware("http")
async def no_cache_dashboard(request, call_next):
    # Always revalidate the dashboard files so browser caches never serve an old app.js
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static"):
        response.headers["Cache-Control"] = "no-cache"
    return response


@app.get("/", include_in_schema=False)
def dashboard():
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.get("/api/models")
def list_models():
    return {"default": default_model, "models": [d.info() for d in detectors.values()]}


@app.get("/api/health")
def health():
    return {"status": "ok", **detectors[default_model].info(), "default": default_model,
            "models": list(detectors)}


@app.post("/api/detect")
def detect(
    file: UploadFile = File(..., description="Road image (JPG/PNG)"),
    model: str | None = Form(None, description="Model id from /api/models; default model if empty"),
    conf: float = Form(0.25, ge=0.01, le=1.0),
    iou: float = Form(0.45, ge=0.01, le=1.0),
    imgsz: int = Form(640, ge=320, le=1280),
):
    key = model or default_model
    if key not in detectors:
        raise HTTPException(404, f"Unknown model '{key}'. Available: {', '.join(detectors)}")
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Image is larger than 15 MB")
    try:
        image = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "File is not a valid image")
    return detectors[key].detect(image, conf, iou, imgsz)
