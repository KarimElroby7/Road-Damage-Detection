"""Start the app:  python run.py   (the dashboard opens in your default browser)

Loads every .pt/.onnx in models/ (YOLOv8s, YOLO11s, YOLO26s). If the folder has none, falls back
to the pretrained yolo11n.pt (COCO classes) so the dashboard can still be tried out.
Set OPEN_BROWSER=0 to not open the browser automatically.
"""
import os
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

import uvicorn

HERE = Path(__file__).resolve().parent
os.chdir(HERE)

if "MODEL_PATH" not in os.environ and not any(p.suffix in (".pt", ".onnx") for p in (HERE / "models").iterdir()):
    print("no model in models/ -> using pretrained yolo11n.pt for a demo")
    os.environ["MODEL_PATH"] = "yolo11n.pt"


def open_when_ready(url: str, timeout: float = 120):
    """Open the dashboard in the system's default browser once the API answers."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url + "api/health", timeout=2)
            if os.name == "nt":
                os.startfile(url)  # system default browser; ignores VS Code's BROWSER override
            else:
                webbrowser.open(url)
            return
        except OSError:
            time.sleep(0.5)


if __name__ == "__main__":
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    if os.getenv("OPEN_BROWSER", "1") != "0":
        url = f"http://{'localhost' if host in ('127.0.0.1', '0.0.0.0') else host}:{port}/"
        threading.Thread(target=open_when_ready, args=(url,), daemon=True).start()
    uvicorn.run("app.main:app", host=host, port=port)
