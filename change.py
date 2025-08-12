from ultralytics import YOLO

# Load your custom model
model = YOLO("best.pt")  # Ensure this path is correct

# Export to ONNX with optimizations
model.export(
    format="onnx",
    dynamic=False,  # Set True for dynamic input shapes (e.g., variable batch size)
    simplify=True,  # Simplify ONNX model (recommended)
    opset=12,       # ONNX opset version (12-18 are stable)
    imgsz=(640, 640)  # Fixed input size (match your training config)
)