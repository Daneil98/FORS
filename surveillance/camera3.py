import cv2, pathlib, torch, os
from .models import Target, Logs 
import numpy as np
from numpy.linalg import norm
pathlib.PosixPath = pathlib.WindowsPath
import torch.quantization
from insightface.app import FaceAnalysis

rcond = None

URL = os.environ.get("URL")

device = torch.device('cpu')

# Load the model in half-precision
weapon_model = torch.hub.load('yolov5/', 'custom',
                               path='best1.pt',
                               force_reload=True,
                               source='local').to('cpu').eval()  # quantization must be done on CPU

print(next(weapon_model.parameters()).device)

# Initialize InsightFace model (do this once globally)

app = FaceAnalysis(name='buffalo_s')
app.prepare(ctx_id=0)
# 🔧 Apply dynamic quantization (only for Linear layers)

quantized_weapon_model = torch.quantization.quantize_dynamic(
    weapon_model,
    {torch.nn.Linear},
    dtype=torch.qint8
)

# Move to device if GPU is available
quantized_weapon_model = quantized_weapon_model.to(device)

# Pre-load known faces
def load_known_faces():
    known_face_encodings = []
    known_face_names = []

    for person in Target.objects.all().only('name', 'photo1'):
        try:
            img = cv2.imread(person.photo1.path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if img is None:
                print(f"Error loading image for {person.name}")
                continue
            
            faces = app.get(img)
            if faces:
                embedding = faces[0].embedding  # 512-d vector
                embedding = embedding / norm(embedding)  # Normalize
                known_face_encodings.append(embedding)
                known_face_names.append(person.name)
            else:
                print(f"No face detected in image for {person.name}")

        except Exception as e:
            print(f"Error processing {person.name}: {str(e)}")
          
    print(f"[INFO] Found {len(faces)} face(s) in saved image,")
    print(f"Loaded {len(known_face_encodings)} known faces: {known_face_names}")
    return known_face_encodings, known_face_names


#def recognize_face(input_encoding, known_face_encodings, known_names):
def recognize_face_insightface(input_embedding, known_embeddings, known_names):
    input_embedding = input_embedding / norm(input_embedding)
    threshold = 0.5  # Try reducing threshold slightly from 0.6 if needed

    similarities = []
    for i, emb in enumerate(known_embeddings):
        emb_norm = emb / norm(emb)
        sim = np.dot(input_embedding, emb_norm)
        similarities.append(sim)

    best_match_idx = np.argmax(similarities)
    best_score = similarities[best_match_idx]

    if best_score > threshold:
        return known_names[best_match_idx]
    else:
        return "Unknown"

def draw_detection(frame, box, class_name, conf):
    # Draw bounding box and label
    x1, y1, x2, y2 = map(int, box)
    color = (0, 0, 255)  # Red for all classes (customize per class if needed)
    
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    # Draw label (with background for better visibility)
    (w, h), _ = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    cv2.putText(frame, f"{class_name} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    

#LOAD KNOWN FACES
known_face_encodings, known_face_names = load_known_faces()


def gen_frames3():
    cap = cv2.VideoCapture(URL)
    frame_count = 0
    
    if not cap.isOpened():
        print("Error: Camera not accessible")
        return 
    
    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to read frame")
            break
        frame = cv2.resize(frame, (320, 320))
        rgb_frame = frame[..., ::-1]
        
        # Inference (convert BGR to RGB)
        if frame_count % 2 == 0: 
            results = quantized_weapon_model(rgb_frame)  # BGR to RGB conversion
            
            #All weapons detections                
            all_detections = results.xyxy[0][
                (results.xyxy[0][:, 4] > 0.7) &
                (results.xyxy[0][:, 5] < 13)
            ] # Only classes 0-12

            front_faces = results.xyxy[0][
                (results.xyxy[0][:, 4] > 0.6) &  # Higher confidence for faces
                (results.xyxy[0][:, 5] == 13)    # Only class 13 (front faces)
            ]
            
            # Process WEAPONS detection
            for *box, conf, cls in all_detections:
                class_name = weapon_model.names[int(cls)]  # Get class name from model
                draw_detection(frame, box, class_name, conf)    #Draw bounding box                    
                #Logs.create_if_not_recent(person='', weapon=class_name, camera=2, frame=frame)    # Log detection (prevents duplicate logging)    
            
            # Separate front faces (class 13) for recognition
            # Run face detection
            
            if frame_count % 20 == 0:    
                for *box, conf, _ in front_faces:
                    x1, y1, x2, y2 = map(int, box)
                    margin = 60
                    h, w, _ = frame.shape
                    x1m = max(0, x1 - margin)
                    y1m = max(0, y1 - margin)
                    x2m = min(w, x2 + margin)
                    y2m = min(h, y2 + margin)

                    face_crop = frame[y1m:y2m, x1m:x2m]
                    face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                    face_resized = cv2.resize(face_rgb, (160, 160))  # 160x160 helps!

                    faces = app.get(face_resized)
                    print(f"[DEBUG] InsightFace found {len(faces)} face(s)")
                    
                    # Skip too-small faces
                    if face_resized.shape[0] < 50 or face_resized.shape[1] < 50:
                        continue
                    
                    if len(faces) == 0:
                        name = "Unknown"
                    else:
                        embedding = faces[0].embedding
                        name = recognize_face_insightface(embedding, known_face_encodings, known_face_names)

                    # Draw result on frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, name, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2) 


        frame_count+=1
        
        # Encode frame for streaming
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        

    cap.release()

    cv2.destroyAllWindows()
