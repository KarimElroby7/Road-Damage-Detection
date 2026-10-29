"""API smoke tests. Uses the models in models/, or the small pretrained yolo11n.pt if there are none."""
import io
import os
from pathlib import Path

import pytest
from PIL import Image

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
if not any(p.suffix in (".pt", ".onnx") for p in MODELS_DIR.iterdir()):
    os.environ.setdefault("MODEL_PATH", "yolo11n.pt")

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _image_bytes(fmt="JPEG"):
    from ultralytics.utils import ASSETS
    buf = io.BytesIO()
    Image.open(ASSETS / "bus.jpg").save(buf, format=fmt)
    return buf.getvalue()


def test_dashboard_served(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "Road Damage Detection" in res.text


def test_health(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["classes"]


def test_detect_returns_boxes(client):
    res = client.post("/api/detect", files={"file": ("bus.jpg", _image_bytes(), "image/jpeg")},
                      data={"conf": "0.25", "iou": "0.45"})
    assert res.status_code == 200
    body = res.json()
    assert body["image"] == {"width": 810, "height": 1080}
    assert sum(body["counts"].values()) == len(body["detections"])
    for d in body["detections"]:
        x1, y1, x2, y2 = d["box"]
        assert 0 <= x1 < x2 <= 810 and 0 <= y1 < y2 <= 1080
        assert 0.25 <= d["confidence"] <= 1


def test_models_listed_and_selectable(client):
    body = client.get("/api/models").json()
    ids = [m["id"] for m in body["models"]]
    assert body["default"] in ids
    for model_id in ids:
        res = client.post("/api/detect", files={"file": ("bus.jpg", _image_bytes(), "image/jpeg")},
                          data={"model": model_id})
        assert res.status_code == 200
        assert res.json()["model"] == model_id


def test_unknown_model_rejected(client):
    res = client.post("/api/detect", files={"file": ("bus.jpg", _image_bytes(), "image/jpeg")},
                      data={"model": "does-not-exist"})
    assert res.status_code == 404


def test_detect_png(client):
    res = client.post("/api/detect", files={"file": ("bus.png", _image_bytes("PNG"), "image/png")})
    assert res.status_code == 200


def test_rejects_non_image(client):
    res = client.post("/api/detect", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert res.status_code == 400


def test_rejects_bad_conf(client):
    res = client.post("/api/detect", files={"file": ("bus.jpg", _image_bytes(), "image/jpeg")},
                      data={"conf": "5"})
    assert res.status_code == 422
