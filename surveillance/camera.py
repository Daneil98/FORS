import cv2
import numpy as np
import imutils
from imutils import paths
import face_recognition
import os, time
from django.http import JsonResponse


from .models import *


active_captures = {}


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

known_face_encodings, known_face_names = load_known_faces()

    
gun_cascade = cv2.CascadeClassifier('gun_cascade.xml')
firstFrame = None
gun_exist = False


def gen_frames(request):
    stream_id = request.GET.get('stream_id', str())
    cap = cv2.VideoCapture(0)    
    active_captures[stream_id] = cap 
    
    if not cap.isOpened():
        print("Error: Camera not accessible")
        return

    while True:
        success, frame = cap.read()
        
        if not success:
            print("Failed to read frame")
            break
        
        
        # Resize frame for consistent processing
        frame = imutils.resize(frame, width=500)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        gun = gun_cascade.detectMultiScale(gray, 1.3, 20, minSize=(100, 100))
        
        if len(gun) > 0:
            gun_exist = True
        
        face_locations = face_recognition.face_locations(frame, number_of_times_to_upsample=0, model="hog")
        face_encodings = face_recognition.face_encodings(frame, face_locations)
        
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


        for (x, y, w, h) in gun:
            #Draw a box around the gun and label with a statement
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame,  "Gun detected",  (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2) 
            Logs.create_if_not_recent(person='', weapon='Gun', camera=1, frame=frame)
        
            
        # Encode frame for streaming
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()





        


"""
# Sequential pipeline (optimized)
if YOLOv5.detect(frame) == "person":  # First detect humans
    face_results = face_recognition.process(frame)  # Then analyze faces
"""