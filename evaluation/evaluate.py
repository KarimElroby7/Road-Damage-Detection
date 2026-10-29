"""Evaluate the trained YOLO models on the same test set and hardware.

Usage (from the evaluation folder, with the app venv):
    ..\\road_damage_app\\.venv\\Scripts\\python evaluate.py

For each road_damage_app/models/<model>.pt it measures, on data/test (Roboflow v3 test split, 429 images):
mAP50, mAP50-95, precision, recall (overall + per class), mean IoU of true positives,
GPU and CPU speed (batch 1), size / params / GFLOPs, and ONNX export (size, CPU speed, mAP).
Writes results/<model>_results.json (same schema as the Kaggle notebooks).
"""
import json
import os
import platform
import shutil
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops

import yaml

HERE = Path(__file__).resolve().parent
TEST_DIR = HERE / "data" / "test" / "images"
WEIGHTS_DIR = HERE.parent / "road_damage_app" / "models"  # same files the app serves
EXPORT_DIR = HERE / "exports"  # ONNX copies live here so the app folder only holds .pt files
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)

# Ultralytics resolves a relative dataset 'path' against the working directory, so pin it to data/
_cfg = yaml.safe_load((HERE / "data" / "data.yaml").read_text())
_cfg["path"] = str(HERE / "data")
DATA = str(OUT / "data_abs.yaml")
Path(DATA).write_text(yaml.safe_dump(_cfg, sort_keys=False))
MODELS = ["yolov8s", "yolo11s", "yolo26s"]
IMGSZ = 640
GPU = 0 if torch.cuda.is_available() else None


def load_gt(img_path, w, h):
    label_path = Path(str(img_path).replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep)).with_suffix(".txt")
    cls, boxes = [], []
    if label_path.exists():
        for line in label_path.read_text().strip().splitlines():
            v = list(map(float, line.split()))
            if len(v) == 5:
                _, xc, yc, bw, bh = v
                x1, y1, x2, y2 = xc - bw / 2, yc - bh / 2, xc + bw / 2, yc + bh / 2
            else:  # polygon label -> bounding box
                xs, ys = v[1::2], v[2::2]
                x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
            cls.append(int(v[0]))
            boxes.append([x1 * w, y1 * h, x2 * w, y2 * h])
    return np.array(cls, dtype=int), np.array(boxes, dtype=float).reshape(-1, 4)


def iou_matrix(a, b):
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def mean_iou(model, names, device):
    """Greedy same-class matching (IoU >= 0.5) of predictions (conf >= 0.25) to ground truth."""
    tp = {n: [] for n in names.values()}
    for r in model.predict(source=str(TEST_DIR), batch=1, imgsz=IMGSZ, conf=0.25, device=device,
                           stream=True, verbose=False):
        h, w = r.orig_shape
        gt_cls, gt_box = load_gt(r.path, w, h)
        if len(r.boxes) == 0 or len(gt_cls) == 0:
            continue
        order = r.boxes.conf.cpu().numpy().argsort()[::-1]
        pr_box = r.boxes.xyxy.cpu().numpy()[order]
        pr_cls = r.boxes.cls.cpu().numpy().astype(int)[order]
        ious = iou_matrix(pr_box, gt_box)
        matched = np.zeros(len(gt_cls), dtype=bool)
        for i in range(len(pr_box)):
            cand = np.where((gt_cls == pr_cls[i]) & ~matched)[0]
            if len(cand) == 0:
                continue
            j = cand[ious[i, cand].argmax()]
            if ious[i, j] >= 0.5:
                matched[j] = True
                tp[names[pr_cls[i]]].append(ious[i, j])
    all_ious = [v for vals in tp.values() for v in vals]
    return (float(np.mean(all_ious)) if all_ious else 0.0,
            {k: float(np.mean(v)) if v else 0.0 for k, v in tp.items()})


def measure_speed(weights, device, n, half=False, warmup=10):
    m = YOLO(weights, task="detect")
    images = sorted(TEST_DIR.iterdir())[:n]
    for img in images[:warmup]:
        m.predict(str(img), imgsz=IMGSZ, device=device, half=half, verbose=False)
    pre = inf = post = 0.0
    for img in images:
        s = m.predict(str(img), imgsz=IMGSZ, device=device, half=half, verbose=False)[0].speed
        pre, inf, post = pre + s["preprocess"], inf + s["inference"], post + s["postprocess"]
    k = len(images)
    total = (pre + inf + post) / k
    return dict(preprocess_ms=pre / k, inference_ms=inf / k, postprocess_ms=post / k, total_ms=total, fps=1000 / total)


def evaluate(name):
    weights = str(WEIGHTS_DIR / f"{name}.pt")
    device = GPU if GPU is not None else "cpu"
    model = YOLO(weights)
    names = model.names

    m = model.val(data=DATA, split="test", imgsz=IMGSZ, batch=16, device=device, plots=True,
                  project=str(OUT), name=f"{name}_test", exist_ok=True, verbose=False)
    per_class = {}
    for i, c in enumerate(m.box.ap_class_index):
        p, r, ap50, ap = m.box.class_result(i)
        per_class[names[c]] = dict(precision=float(p), recall=float(r), ap50=float(ap50), ap50_95=float(ap))

    miou, miou_cls = mean_iou(model, names, device)
    n_test = len(list(TEST_DIR.iterdir()))
    gpu = measure_speed(weights, GPU, n_test, half=True) if GPU is not None else None
    cpu = measure_speed(weights, "cpu", 50)

    EXPORT_DIR.mkdir(exist_ok=True)
    shutil.copy(weights, EXPORT_DIR / f"{name}.pt")
    onnx_path = YOLO(str(EXPORT_DIR / f"{name}.pt")).export(format="onnx", imgsz=IMGSZ, simplify=True)
    onnx_cpu = measure_speed(onnx_path, "cpu", 50)
    onnx_m = YOLO(onnx_path, task="detect").val(data=DATA, split="test", imgsz=IMGSZ, batch=1, device="cpu",
                                                plots=False, verbose=False, project=str(OUT),
                                                name=f"{name}_onnx_test", exist_ok=True)

    summary = dict(
        model=name,
        map50=float(m.box.map50), map50_95=float(m.box.map),
        precision=float(m.box.mp), recall=float(m.box.mr),
        mean_iou=miou,
        gpu_ms=gpu["total_ms"] if gpu else None, gpu_fps=gpu["fps"] if gpu else None,
        cpu_ms=cpu["total_ms"], cpu_fps=cpu["fps"],
        size_mb=os.path.getsize(weights) / 1e6,
        params_m=sum(p.numel() for p in model.model.parameters()) / 1e6,
        gflops=float(get_flops(model.model, IMGSZ)),
        onnx_size_mb=os.path.getsize(onnx_path) / 1e6,
        onnx_cpu_fps=onnx_cpu["fps"],
        onnx_map50_95=float(onnx_m.box.map),
        per_class=per_class,
        mean_iou_per_class=miou_cls,
        speed_detail=dict(gpu_fp16=gpu, cpu=cpu, onnx_cpu=onnx_cpu),
        hardware=dict(gpu=torch.cuda.get_device_name(0) if GPU is not None else None,
                      cpu=platform.processor(), torch=torch.__version__),
        test_images=n_test,
    )
    OUT.mkdir(exist_ok=True)
    (OUT / f"{name}_results.json").write_text(json.dumps(summary, indent=2))
    print(f"[{name}] mAP50={summary['map50']:.3f} mAP50-95={summary['map50_95']:.3f} "
          f"P={summary['precision']:.3f} R={summary['recall']:.3f} IoU={miou:.3f} "
          f"GPU={summary['gpu_fps'] or 0:.0f}fps CPU={summary['cpu_fps']:.1f}fps ONNX-CPU={summary['onnx_cpu_fps']:.1f}fps")


if __name__ == "__main__":
    os.chdir(HERE)
    for model_name in MODELS:
        evaluate(model_name)
