import cv2
import numpy as np
import imutils
from imutils import paths
from .tasks import *



known_face_encodings = []
known_face_names = []

known_persons = Target.objects.all()


for known_person in known_persons:
    known_person_image = face_recognition.load_image_file(known_person.photo1)
    known_person_encoding = face_recognition.face_encodings(known_person_image)[0]
    known_face_encodings.append(known_person_encoding)
    known_face_names.append(known_person.name)
    
    



def gen_frames():
    cap = cv2.VideoCapture(0)
    
    #face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
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


        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)
        
        #Loop through each face found in the frame
        for face_encoding in face_encodings:
        
            #Check if there are any known faces
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
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


        
        


        #    face_rec(frame)
        #gun_detection.delay(frame, gray)

            
        """
                # Face Detection
        faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(40, 40))
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 4)
        """
        
            
        # Encode frame for streaming
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
