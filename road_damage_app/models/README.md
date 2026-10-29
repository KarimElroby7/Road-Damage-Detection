Every `.pt` / `.onnx` file in this folder is loaded at startup and can be picked in the dashboard
"Model" dropdown (or with the `model` field of `POST /api/detect`).

Current models (trained on Roboflow road-damage v3, same settings):
- `yolov8s.pt`
- `yolo11s.pt`
- `yolo26s.pt`  (default, best overall in evaluation/comparison; change with `DEFAULT_MODEL`)

The model id is the file name without extension.
