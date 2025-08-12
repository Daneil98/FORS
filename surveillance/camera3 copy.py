import cv2, pathlib, face_recognition, time, torch
from .models import Target, Logs 
from pathlib import Path
import numpy as np
pathlib.PosixPath = pathlib.WindowsPath



URL = 'http://192.168.100.103:8080/video'

"""
#Initialize models on GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#face_model = torch.hub.load('yolov5/', 'yolov5n', force_reload=True, source='local', pretrained = True).to(device).eval()
weapon_model = torch.hub.load('yolov5/', 'custom', 
                      path='best.pt', 
                      force_reload=True,
                      source='local').to(device).eval()

# Half-precision for 2GB GPUs
if device == 'cuda':
    #face_model.half()
    weapon_model.half()
"""
import torch.quantization

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the model as usual
weapon_model = torch.hub.load('yolov5/', 'custom',
                               path='best.pt',
                               force_reload=True,
                               source='local').to('cpu').eval()  # quantization must be done on CPU

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
            image = face_recognition.load_image_file(person.photo1)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_face_encodings.append(encodings[0])
                known_face_names.append(person.name)
        except Exception as e:
            print(f"Error loading {person.name}: {str(e)}")
    
    return known_face_encodings, known_face_names


def recognize_face(encoding):
    matches = face_recognition.compare_faces(known_face_encodings, encoding, tolerance=0.7)
    face_distances = face_recognition.face_distance(known_face_encodings, encoding)
    best_match = np.argmin(face_distances)
    return known_face_names[best_match] if matches[best_match] else "Unknown"


def draw_detection(frame, box, class_name, conf):
    # Draw bounding box and label
    x1, y1, x2, y2 = map(int, box)
    color = (0, 0, 255)  # Red for all classes (customize per class if needed)
    
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    # Draw label (with background for better visibility)
    (w, h), _ = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    cv2.putText(frame, f"{class_name} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    

def draw_face(img, box, name):
    left, top, right, bottom = box
    cv2.rectangle(img, (left, top), (right, bottom), (255, 0, 0), 2)
    cv2.putText(img, name, (left, top-10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)


#LOAD KNOWN FACES
known_face_encodings, known_face_names = load_known_faces()


def gen_frames3():
    cap = cv2.VideoCapture(URL)
    
    if not cap.isOpened():
        print("Error: Camera not accessible")
        return 
    
    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to read frame")
            break
        
        rgb_frame = frame[..., ::-1]
        
        # Inference (convert BGR to RGB)
        results = quantized_weapon_model(rgb_frame)  # BGR to RGB conversion
        #face_results = face_model(rgb_frame)
        min_conf = 0.6      # Only show detections with >60% confidence
        face_conf = 0.7     # Only show detections with >70% confidence
                        
        all_detections = results.xyxy[0][results.xyxy[0][:, 4] > min_conf]  #All weapons detections
        #face_detection = face_results.xyxy[0][    (face_results.xyxy[0][:, 4] > face_conf) & 
        #    (face_results.xyxy[0][:, 5] == 0)]  # Class 0 is 'person' in COCO#All face detections

        # Process WEAPONS
        for *box, conf, cls in all_detections:
            class_name = weapon_model.names[int(cls)]  # Get class name from model
            draw_detection(frame, box, class_name, conf)    #Draw bounding box                    
            Logs.create_if_not_recent(person='', weapon=class_name, camera=2, frame=frame)    # Log detection (prevents duplicate logging)
        
        
#        if len(face_detection) > 0: 
        #Process each detection for face_Recognition  
        # Efficient face processing
        face_locations = face_recognition.face_locations(frame, number_of_times_to_upsample=0, model="hog")
        face_encodings = face_recognition.face_encodings(frame, face_locations)
            
#            for *box, _, _ in face_detection:
#                x1, y1, x2, y2 = map(int, box)
#                face_locations.append((y1, x2, y2, x1))  # Convert to (top, right, bottom, left)
            
            # Only run recognition on high-confidence faces
        #Loop through each face found in the frame
        for face_encoding in face_encodings:
        
            #Check if there are any known faces
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
            name = "Unknown"
            
            """
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]            
            """
                
            face_distances= face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin (face_distances)
            
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                
            
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):   
            #Draw a box around the face and label with the name
            cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            Logs.create_if_not_recent(person=name, weapon='', camera=1, frame=frame)
            if name != "Unknown":
                Logs.create_if_not_recent(person=name, weapon='', camera=2, frame=frame)    
#        else:
#            pass        
        
        # Encode frame for streaming
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        

    cap.release()
    cv2.destroyAllWindows()
    """
    
     # Inside your video loop:
        for *box, conf, _ in front_faces:
            x1, y1, x2, y2 = map(int, box)
            
            # Crop the face from the frame
            face_img = frame[y1:y2, x1:x2]
            
            # Skip too-small faces
            if face_img.shape[0] < 50 or face_img.shape[1] < 50:
                continue

            # Resize to InsightFace-friendly shape if needed (not required usually)
            face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            
            # Get face embedding
            faces = app.get(face_img_rgb)
            if len(faces) == 0:
                name = "Unknown"
            else:
                embedding = faces[0].embedding  # (512,)

                # Compare to known embeddings using cosine similarity
                similarities = np.dot(known_embeddings, embedding) / (
                    np.linalg.norm(known_embeddings, axis=1) * np.linalg.norm(embedding) + 1e-5
                )
                best_match = np.argmax(similarities)
                name = known_face_names[best_match] if similarities[best_match] > 0.5 else "Unknown"

            # Draw result on frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, name, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        #Process each face detection for face_Recognition  
        for *box, conf, _ in front_faces:
            x1, y1, x2, y2 = map(int, box)
            # Safely extract face ROI
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 > x1 and y2 > y1:  # Valid region check
                face_roi = frame[y1:y2, x1:x2]
                
                # Convert to RGB for face_recognition
                face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                
                # Get face locations and encodings
                face_locations = [(0, face_roi.shape[1], face_roi.shape[0], 0)]
                encodings = face_recognition.face_encodings(face_rgb, known_face_locations=face_locations)
                
                if encodings:
                    name = recognize_face(encodings[0], known_face_encodings, known_face_names)
                    cv2.putText(frame, name, (x1, y1-30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    Logs.create_if_not_recent(person=name, weapon='', camera=2, frame=frame) 
                """