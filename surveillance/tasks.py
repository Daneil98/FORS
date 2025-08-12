from celery import shared_task
from .models import *
import face_recognition
from .models import Target
import cv2
import numpy as np
import datetime
from imutils import paths



def face_rec_setup():
    #Initialize face encodings and names list
    known_face_encodings = []
    known_face_names = []

    known_persons = Target.objects.all()
    
    
    for known_person in known_persons:
        known_person_image = face_recognition.load_image_file(known_person.photo1)
        known_person_encoding = face_recognition.face_encodings(known_person_image)[0]
        known_face_encodings.append(known_person_encoding)
        known_face_names.append(known_person.name)

    yield known_face_encodings
    yield known_face_names


@shared_task
def face_rec(frame):
    known_face_encodings = []
    known_face_names = []

    known_persons = Target.objects.all()
    
    
    for known_person in known_persons:
        known_person_image = face_recognition.load_image_file(known_person.photo1)
        known_person_encoding = face_recognition.face_encodings(known_person_image)[0]
        known_face_encodings.append(known_person_encoding)
        known_face_names.append(known_person.name)
    
    
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
    
    """
    #Load the known faces and names here
    known_person1_image = face_recognition.load_image_file('surveillance\messi3.jpg')
    known_person2_image = face_recognition.load_image_file('surveillance\Passport.jpg')
    
    known_person1_encoding = face_recognition.face_encodings(known_person1_image)[0]
    known_person2_encoding = face_recognition.face_encodings(known_person2_image)[0]

    #Add the persons face encoding to the list
    known_face_encodings.append(known_person1_encoding)
    known_face_encodings.append(known_person2_encoding)
    
    known_face_names.append("Messi")
    known_face_names.append("Daniel")
    """
    

@shared_task   
def gun_detection(frame, gray):
    gun_cascade = cv2.CascadeClassifier('gun_cascade.xml')
    
    # GUN DETECTION
    gun = gun_cascade.detectMultiScale(gray, 1.3, 20, minSize=(100, 100))
    gun_exist = len(gun) > 0

    #create a box and add a gun detected text on the gun
    for (x, y, w, h) in gun:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, "Gun Detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        

    # Overlay Date and Time
    cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S %p"),
        (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1,)
    
    """
    if gun_exist:
        print("Guns detected")
        # Optional: Add additional handling logic here
        pass """

