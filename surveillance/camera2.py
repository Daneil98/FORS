import cv2
import numpy as np
import face_recognition
import time
from concurrent.futures import ThreadPoolExecutor
from queue import Queue, Empty
from .models import Target, Logs

# Configuration
URL = 'http://192.168.0.179:8080/video'
PROCESSING_WIDTH = 500  # Lower for faster processing
OUTPUT_WIDTH = 640
JPEG_QUALITY = 70
TARGET_FPS = 30
MAX_QUEUE_SIZE = 1  # Minimal queue to prevent lag
PROCESS_TIMEOUT = 1.0  # Seconds before giving up on processing
RECONNECT_DELAY = 2  # Seconds before reconnecting


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


def gen_frames2():
    while True:  # Outer loop for reconnection
        # Initialize fresh resources each reconnect
        cap = None
        executor = ThreadPoolExecutor(max_workers=1)
        frame_queue = Queue(maxsize=MAX_QUEUE_SIZE)
        last_frame_time = time.time()
        
        try:
            # Initialize camera
            cap = cv2.VideoCapture(URL)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
            cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
            
            if not cap.isOpened():
                raise RuntimeError("Camera initialization failed")

            def process_task(frame):
                try:
                    # Fast resize and color conversion
                    small_frame = cv2.resize(frame, (PROCESSING_WIDTH, int(frame.shape[0] * PROCESSING_WIDTH / frame.shape[1])))
                    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                    
                    # Efficient face processing
                    face_locations = face_recognition.face_locations(rgb_frame, number_of_times_to_upsample=0, model="hog")
                    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                    
                    names = []
                    for encoding in face_encodings:
                        matches = face_recognition.compare_faces(known_face_encodings, encoding, tolerance=0.6)
                        name = "Unknown"
                        if True in matches:
                            distances = face_recognition.face_distance(known_face_encodings, encoding)
                            best_match = np.argmin(distances)
                            name = known_face_names[best_match]
                        names.append(name)
                    
                    return (face_locations, names)
                except Exception as e:
                    print(f"Processing failed: {str(e)}")
                    return ([], [])

            while True:  # Main processing loop
                # Read frame with timeout
                ret, frame = cap.read()
                if not ret:
                    print("Frame read failed - reconnecting")
                    time.sleep(0.1)
                    break

                # Process only if queue isn't backed up
                if frame_queue.qsize() < MAX_QUEUE_SIZE:
                    future = executor.submit(process_task, frame.copy())
                    frame_queue.put((frame, future))

                # Get processed results with timeout
                try:
                    frame, future = frame_queue.get(timeout=0.1)
                    locations, names = future.result(timeout=PROCESS_TIMEOUT)
                    
                    # Draw results
                    scale = frame.shape[1] / PROCESSING_WIDTH
                    for (top, right, bottom, left), name in zip(locations, names):
                        top, right, bottom, left = [int(coord * scale) for coord in [top, right, bottom, left]]
                        cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
                        cv2.putText(frame, name, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                        if name != "Unknown":
                            Logs.create_if_not_recent(person=name, weapon='', camera=2, frame=frame)

                    # Encode frame
                    output_frame = cv2.resize(frame, (OUTPUT_WIDTH, int(OUTPUT_WIDTH * frame.shape[0]/frame.shape[1])))
                    _, buffer = cv2.imencode('.jpg', output_frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

                except Empty:
                    # No frames ready - yield last good frame
                    continue
                except Exception as e:
                    print(f"Result handling failed: {str(e)}")
                    continue

                # Maintain FPS
                elapsed = time.time() - last_frame_time
                if elapsed < 1/TARGET_FPS:
                    time.sleep((1/TARGET_FPS) - elapsed)
                last_frame_time = time.time()

        except Exception as e:
            print(f"Fatal error: {str(e)}")
        finally:
            # Cleanup
            if cap and cap.isOpened():
                cap.release()
            executor.shutdown(wait=False)
            time.sleep(RECONNECT_DELAY)