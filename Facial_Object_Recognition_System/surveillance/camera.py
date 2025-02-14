import cv2
import numpy as np
import imutils
import datetime

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')





def gen_frames():
    cap = cv2.VideoCapture(0)
    gun_cascade = cv2.CascadeClassifier('cascade.xml')
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    


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

        # Face Detection
        faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(40, 40))
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 4)

        # Gun Detection
        gun = gun_cascade.detectMultiScale(gray, 1.3, 20, minSize=(100, 100))
        gun_exist = len(gun) > 0

        for (x, y, w, h) in gun:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, "Gun Detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            

        # Overlay Date and Time
        cv2.putText(
            frame,
            datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S %p"),
            (10, frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 255, 0),
            1,
        )
        """
        if gun_exist:
            print("Guns detected")
            #Add additional logic here
            pass """
            
        # Encode frame for streaming
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
