import cv2
import torch

import pathlib
from pathlib import Path
pathlib.PosixPath = pathlib.WindowsPath

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Load custom YOLOv5 model
#model = torch.hub.load('./yolov5', 'custom',
#                       path='best.pt',
#                       source='local')

model = torch.hub.load('yolov5/', 'custom', 
                      path='best.pt', 
                      force_reload=True,
                      source='local').to(device).eval()
model.conf = 0.4 # Confidence threshold (adjust as needed)

# Load image
image_path = 'WhatsApp Image 2025-07-15 at 04.26.15.jpeg'
image = cv2.imread(image_path)  # BGR format

# Check if image loaded correctly
if image is None:
    raise FileNotFoundError(f"Image not found at {image_path}")

# Perform inference (convert BGR to RGB)
results = model(image[..., ::-1])  

# 4. Check if any detections exist
if len(results.xyxy[0]) == 0:
    print("No weapon detected!")
else:
    print(f"Detected {len(results.xyxy[0])} objects")


all_detections = results.xyxy[0][results.xyxy[0][:, 4] > model.conf]

# Process each detection
for *box, conf, cls in all_detections:
    class_id = int(cls)
    class_name = model.names[class_id]  # Get class name from model
    
    # Draw bounding box and label
    x1, y1, x2, y2 = map(int, box)
    color = (0, 0, 255)  # Green for all classes (customize per class if needed)
    
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    # Draw label (with background for better visibility)
    (w, h), _ = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    cv2.putText(image, f"{class_name} {conf:.2f}", (x1, y1 - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
# 6. Save and show output
output_path = 'test_results/gun1.jpeg'
cv2.imwrite(output_path, image)
print(f"Saved result to {output_path}, confidence -")
