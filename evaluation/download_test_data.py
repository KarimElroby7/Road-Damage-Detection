"""Download the Roboflow v3 test split into evaluation/data/test (needed by evaluate.py and qualitative.py).

Usage (Windows):
    set ROBOFLOW_API_KEY=<your key>
    ..\\road_damage_app\\.venv\\Scripts\\python -m pip install roboflow
    ..\\road_damage_app\\.venv\\Scripts\\python download_test_data.py
"""
import os
import shutil
import tempfile
from pathlib import Path

from roboflow import Roboflow

HERE = Path(__file__).resolve().parent
DEST = HERE / "data" / "test"

key = os.environ.get("ROBOFLOW_API_KEY")
if not key:
    raise SystemExit("Set the ROBOFLOW_API_KEY environment variable first.")

with tempfile.TemporaryDirectory() as tmp:
    ds = Roboflow(api_key=key).workspace("project-dvd12").project("road-damage-eul7g") \
        .version(3).download("yolov11", location=str(Path(tmp) / "v3"))
    shutil.rmtree(DEST, ignore_errors=True)
    shutil.copytree(Path(ds.location) / "test", DEST)

print("Test split:", len(list((DEST / "images").iterdir())), "images ->", DEST)
