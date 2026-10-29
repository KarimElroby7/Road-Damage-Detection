"""Qualitative figure: ground truth vs. predictions of the three models on the same test images.

Usage:  ..\\road_damage_app\\.venv\\Scripts\\python qualitative.py
"""
import random
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from ultralytics import YOLO  # noqa: E402

HERE = Path(__file__).resolve().parent
TEST = HERE / "data" / "test"
OUT = HERE / "comparison" / "qualitative.png"
MODELS = {"yolov8s": "YOLOv8s", "yolo11s": "YOLO11s", "yolo26s": "YOLO26s"}
NAMES = ["Crack", "Manhole", "Pothole"]
BGR = {0: (214, 120, 42), 1: (52, 104, 235), 2: (122, 175, 27)}  # Crack blue, Manhole orange, Pothole green


def labels(img_path):
    lbl = TEST / "labels" / (img_path.stem + ".txt")
    rows = [list(map(float, l.split())) for l in lbl.read_text().splitlines() if l.strip()]
    return [r for r in rows if len(r) == 5]


def pick_images():
    rng = random.Random(3)
    by_class = {c: [] for c in range(3)}
    mixed = []
    for img in sorted((TEST / "images").iterdir()):
        if any(s in img.name.lower() for s in ("getty", "shutterstock", "istock", "stock", "clipart", "vector")):
            continue  # stock/clip-art images exist in the merged manhole data; show real road photos here
        cls = {int(r[0]) for r in labels(img)}
        n = len(labels(img))
        if len(cls) == 1 and 1 <= n <= 3:
            by_class[cls.pop()].append(img)
        elif len(cls) >= 2 and n <= 4:
            mixed.append(img)
    return [rng.choice(by_class[0]), rng.choice(by_class[2]), rng.choice(by_class[1]), rng.choice(mixed)]


def draw(img, boxes, with_conf):
    out = img.copy()
    t = max(2, img.shape[1] // 250)
    for cls, (x1, y1, x2, y2), conf in boxes:
        color = BGR[cls]
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), color, t)
        text = f"{NAMES[cls]} {conf:.2f}" if with_conf else NAMES[cls]
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        y = max(int(y1), th + 6)
        cv2.rectangle(out, (int(x1), y - th - 6), (int(x1) + tw + 6, y), color, -1)
        cv2.putText(out, text, (int(x1) + 3, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def main():
    images = pick_images()
    models = {k: YOLO(HERE.parent / "road_damage_app" / "models" / f"{k}.pt") for k in MODELS}
    fig, axes = plt.subplots(len(images), 4, figsize=(16, 4.1 * len(images)))
    for r, path in enumerate(images):
        img = cv2.imread(str(path))
        h, w = img.shape[:2]
        gt = [(int(c), ((x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h), 1.0)
              for c, x, y, bw, bh in labels(path)]
        panels = [("Ground truth", draw(img, gt, False))]
        for key, label in MODELS.items():
            res = models[key].predict(str(path), imgsz=640, conf=0.25, verbose=False)[0]
            preds = [(int(c), b, float(s)) for c, b, s in zip(res.boxes.cls.tolist(), res.boxes.xyxy.tolist(), res.boxes.conf.tolist())]
            panels.append((label, draw(img, preds, True)))
        for c, (title, im) in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(im)
            ax.axis("off")
            if r == 0:
                ax.set_title(title, fontsize=15, fontweight="bold", pad=10)
    plt.tight_layout()
    fig.savefig(OUT, dpi=110, bbox_inches="tight", facecolor="white")
    print("saved", OUT, [p.name[:40] for p in images])


if __name__ == "__main__":
    main()
