"""Compare the evaluated models: table (CSV + Markdown), charts (PNG) and a weighted ranking.

Usage (after evaluate.py):
    ..\\road_damage_app\\.venv\\Scripts\\python compare.py
"""
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
OUT = HERE / "comparison"
MODELS = ["yolov8s", "yolo11s", "yolo26s"]
LABELS = {"yolov8s": "YOLOv8s", "yolo11s": "YOLO11s", "yolo26s": "YOLO26s"}
COLORS = {"yolov8s": "#2a78d6", "yolo11s": "#eb6834", "yolo26s": "#1baf7a"}
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#1f1f1e", "#5f5e5a", "#e4e3df"

COLUMNS = [
    ("map50", "mAP50", True, "{:.3f}"), ("map50_95", "mAP50-95", True, "{:.3f}"),
    ("precision", "Precision", True, "{:.3f}"), ("recall", "Recall", True, "{:.3f}"),
    ("mean_iou", "Mean IoU", True, "{:.3f}"),
    ("gpu_fps", "GPU FPS (FP16)", True, "{:.0f}"), ("gpu_ms", "GPU ms/img", False, "{:.1f}"),
    ("cpu_fps", "CPU FPS", True, "{:.1f}"), ("onnx_cpu_fps", "ONNX CPU FPS", True, "{:.1f}"),
    ("size_mb", "best.pt (MB)", False, "{:.1f}"), ("params_m", "Params (M)", False, "{:.2f}"),
    ("gflops", "GFLOPs", False, "{:.1f}"), ("onnx_size_mb", "ONNX (MB)", False, "{:.1f}"),
    ("onnx_map50_95", "ONNX mAP50-95", True, "{:.3f}"),
]
# Weighted ranking for a deployment app: accuracy first, then CPU speed, recall and size
WEIGHTS = {"map50_95": 0.40, "recall": 0.15, "mean_iou": 0.10, "onnx_cpu_fps": 0.20, "gpu_fps": 0.05, "size_mb": 0.10}
LOWER_IS_BETTER = {"size_mb"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "xtick.color": INK_2, "ytick.color": INK_2,
    "xtick.major.size": 0, "ytick.major.size": 0, "font.size": 11, "legend.frameon": False,
})


def annotate(ax, bars, vals, fmt):
    for b, v in zip(bars, vals):
        ax.annotate(fmt.format(v), (b.get_x() + b.get_width() / 2, b.get_height()), xytext=(0, 3),
                    textcoords="offset points", ha="center", fontsize=9, color=INK_2)


def grouped_bars(ax, categories, values_by_model, fmt="{:.3f}"):
    width = 0.8 / len(values_by_model)
    x = np.arange(len(categories))
    for i, (m, vals) in enumerate(values_by_model.items()):
        bars = ax.bar(x - 0.4 + width * (i + 0.5), vals, width, color=COLORS[m], label=LABELS[m],
                      edgecolor=SURFACE, linewidth=2)
        annotate(ax, bars, vals, fmt)
    ax.set_xticks(x, categories)
    ax.legend(loc="upper left", ncol=len(values_by_model))


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    r = {m: json.loads((RES / f"{m}_results.json").read_text()) for m in MODELS}

    # ---- table ----
    with open(OUT / "comparison_table.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Model"] + [c[1] for c in COLUMNS])
        for m in MODELS:
            w.writerow([LABELS[m]] + [round(r[m][k], 4) for k, *_ in COLUMNS])

    best = {k: (max if hib else min)(r[m][k] for m in MODELS) for k, _, hib, _ in COLUMNS}
    md = ["| Model | " + " | ".join(c[1] for c in COLUMNS) + " |", "|---" * (len(COLUMNS) + 1) + "|"]
    for m in MODELS:
        cells = [(f"**{fmt.format(r[m][k])}**" if r[m][k] == best[k] else fmt.format(r[m][k])) for k, _, _, fmt in COLUMNS]
        md.append(f"| {LABELS[m]} | " + " | ".join(cells) + " |")

    classes = list(r[MODELS[0]]["per_class"])
    md += ["", "| Model | " + " | ".join(f"{c} AP50 | {c} AP50-95 | {c} Recall" for c in classes) + " |",
           "|---" * (3 * len(classes) + 1) + "|"]
    for m in MODELS:
        pc = r[m]["per_class"]
        md.append(f"| {LABELS[m]} | " + " | ".join(
            f"{pc[c]['ap50']:.3f} | {pc[c]['ap50_95']:.3f} | {pc[c]['recall']:.3f}" for c in classes) + " |")

    # ---- ranking ----
    raw = {k: np.array([r[m][k] for m in MODELS], dtype=float) for k in WEIGHTS}
    score = np.zeros(len(MODELS))
    for k, wgt in WEIGHTS.items():
        v = raw[k]
        norm = (v - v.min()) / (v.max() - v.min()) if v.max() > v.min() else np.ones_like(v)
        score += wgt * (1 - norm if k in LOWER_IS_BETTER else norm)
    order = np.argsort(-score)
    md += ["", "Weighted score (0-1, min-max normalised): " + ", ".join(
        f"{LABELS[MODELS[i]]} {score[i]:.3f}" for i in order), f"Best overall: **{LABELS[MODELS[order[0]]]}**"]
    (OUT / "comparison.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))

    # ---- charts ----
    keys = ["map50", "map50_95", "precision", "recall", "mean_iou"]
    fig, ax = plt.subplots(figsize=(11, 5))
    grouped_bars(ax, [dict((c[0], c[1]) for c in COLUMNS)[k] for k in keys], {m: [r[m][k] for k in keys] for m in MODELS})
    ax.set_ylim(0, 1.05)
    ax.set_title("Accuracy on the test set (429 images)")
    save(fig, "accuracy")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for ax, (key, title) in zip(axes, [("ap50", "AP50 per class"), ("ap50_95", "AP50-95 per class"), ("recall", "Recall per class")]):
        grouped_bars(ax, classes, {m: [r[m]["per_class"][c][key] for c in classes] for m in MODELS})
        ax.set_ylim(0, 1.1)
        ax.set_title(title)
    plt.tight_layout()
    save(fig, "per_class")

    for group, name in [([("gpu_fps", "GPU FPS (RTX 3060, FP16)", "{:.0f}"), ("cpu_fps", "CPU FPS (PyTorch)", "{:.1f}"),
                          ("onnx_cpu_fps", "CPU FPS (ONNX)", "{:.1f}")], "speed"),
                        ([("size_mb", "best.pt size (MB)", "{:.1f}"), ("params_m", "Parameters (M)", "{:.2f}"),
                          ("gflops", "GFLOPs", "{:.1f}")], "size")]:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
        for ax, (k, title, fmt) in zip(axes, group):
            vals = [r[m][k] for m in MODELS]
            bars = ax.bar([LABELS[m] for m in MODELS], vals, color=[COLORS[m] for m in MODELS],
                          edgecolor=SURFACE, linewidth=2, width=0.6)
            annotate(ax, bars, vals, fmt)
            ax.set_title(title)
            ax.set_ylim(0, max(vals) * 1.18)
        plt.tight_layout()
        save(fig, name)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for m in MODELS:
        x, y = r[m]["gpu_fps"], r[m]["map50_95"]
        ax.scatter(x, y, s=60 + r[m]["size_mb"] * 8, color=COLORS[m], edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.annotate(f"{LABELS[m]} ({r[m]['size_mb']:.0f} MB)", (x, y), xytext=(10, 6), textcoords="offset points", color=INK)
    ax.grid(True, axis="both")
    ax.set_xlabel("GPU FPS on RTX 3060 (higher = faster)")
    ax.set_ylabel("mAP50-95 (test)")
    ax.set_title("Accuracy vs. speed (marker size = model size)")
    xs, ys = [r[m]["gpu_fps"] for m in MODELS], [r[m]["map50_95"] for m in MODELS]
    ax.set_xlim(min(xs) * 0.85, max(xs) * 1.2)
    ax.set_ylim(min(ys) - 0.03, max(ys) + 0.03)
    save(fig, "tradeoff")
    print("\nSaved to", OUT)


if __name__ == "__main__":
    main()
