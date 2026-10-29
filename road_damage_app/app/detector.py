"""YOLO wrapper used by the API: loads the model once and turns predictions into JSON."""
import threading
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.cfg import DEFAULT_CFG_DICT

# Newer Ultralytics replaced half=True with quantize=16; support both
FP16_ARGS = {"quantize": 16} if "quantize" in DEFAULT_CFG_DICT else {"half": True}

PRETTY_NAMES = {"yolov8s": "YOLOv8s", "yolo11s": "YOLO11s", "yolo26s": "YOLO26s"}


class Detector:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.id = Path(model_path).stem
        self.label = PRETTY_NAMES.get(self.id, self.id)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.half = self.device != "cpu"  # FP16 on the GPU: faster, same accuracy for inference
        self._precision = FP16_ARGS if self.half else {}
        self.device_name = torch.cuda.get_device_name(0) if self.half else "CPU"
        try:
            self.model = YOLO(model_path, task="detect")
        except FileNotFoundError as e:
            raise RuntimeError(
                f"Model not found: {model_path}. Put your trained best.pt in road_damage_app/models/ "
                "or set the MODEL_PATH environment variable."
            ) from e
        self.names = {int(k): v for k, v in self.model.names.items()}
        # Ultralytics predictors are not thread-safe; FastAPI runs sync endpoints in a thread pool
        self._lock = threading.Lock()
        # Warm-up so the first real request is not slow
        self.model.predict(np.zeros((640, 640, 3), dtype=np.uint8), device=self.device,
                           verbose=False, **self._precision)

    def info(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "model": Path(self.model_path).name,
            "device": self.device,
            "device_name": self.device_name,
            "half": self.half,
            "classes": [self.names[i] for i in sorted(self.names)],
        }

    def detect(self, image, conf: float, iou: float, imgsz: int) -> dict:
        with self._lock:
            result = self.model.predict(image, conf=conf, iou=iou, imgsz=imgsz,
                                        device=self.device, verbose=False, **self._precision)[0]

        detections = []
        for box, cls, score in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist(),
                                   result.boxes.conf.tolist()):
            detections.append({
                "class_id": int(cls),
                "class_name": self.names[int(cls)],
                "confidence": round(float(score), 4),
                "box": [round(v, 1) for v in box],  # x1, y1, x2, y2 in image pixels
            })
        detections.sort(key=lambda d: d["confidence"], reverse=True)

        counts = {self.names[i]: 0 for i in sorted(self.names)}
        for d in detections:
            counts[d["class_name"]] += 1

        h, w = result.orig_shape
        return {
            "detections": detections,
            "counts": counts,
            "image": {"width": w, "height": h},
            "timing_ms": {k: round(v, 1) for k, v in result.speed.items()},
            "model": self.id,
            "model_label": self.label,
        }
