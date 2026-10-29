"""Start the app:  python run.py   (then open http://localhost:8000)

Loads every .pt/.onnx in models/ (YOLOv8s, YOLO11s, YOLO26s). If the folder has none, falls back
to the pretrained yolo11n.pt (COCO classes) so the dashboard can still be tried out.
"""
import os
from pathlib import Path

import uvicorn

HERE = Path(__file__).resolve().parent
os.chdir(HERE)

if "MODEL_PATH" not in os.environ and not any(p.suffix in (".pt", ".onnx") for p in (HERE / "models").iterdir()):
    print("no model in models/ -> using pretrained yolo11n.pt for a demo")
    os.environ["MODEL_PATH"] = "yolo11n.pt"

if __name__ == "__main__":
    uvicorn.run("app.main:app", host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "8000")))
