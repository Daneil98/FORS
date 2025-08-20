
  ORION'S SHIELD
- A Facial Recognition and weapon detection web-app

This web-app allows authenticated and authorized users to access a system that recognizes faces from uploaded images and detect weapons (knives, guns, drones) in real time via connected camera feeds. It consists of the integration of a facial recognition model (InsightFace) and quantized weapon detection model (Custom trained YOLOV5n model) into a webapp. This webapp also allows for real time SMS notifications and automatic logging for recognized faces and detected weapons with screenshots for future review.

The facial recognition & objection detection pipeline flowchart is at https://github.com/Daneil98/FORS/blob/master/Facial%20recognition%20and%20weapon%20detection%20pipeline.png

The Django System aAchitecture diagram is at https://github.com/Daneil98/FORS/blob/master/Django%20System%20Architecture.png


The weapon detection model was trained to detect the following classes for more granular data, effective response and threat contextual analysis:

names:
  0: knife
  1: knife_in_hand
  2: pistol
  3: pistol_in_hand
  4: rifle
  5: rifle_in_hand
  6: sniper
  7: shotgun
  8: shotgun_in_hand
  9: RPG
  10: MG
  11: Mounted_MG
  12: Drone
  13: Front_Face
  14: Side_Face


- NOTE: 
1. Please ensure CMake and C++ SDK from Visual Studio is installed on yor local machine.
2. The InsightFace model runs on GPU, so please make sure you have a CUDA enabled GPU on your local machine.
3. Make sure you clone the yolov5 repository and place it in the project directory since you will be running a custom model on your local machine.


## Run Locally

- Clone the project

```bash
  git clone https://github.com/Daneil98/FORS
```

- Go to the project directory

```bash
  cd FORS
```

- Install dependencies

```bash
  pip install -r requirements.txt

```

- Prepare Migrations

```bash
  python manage.py makemigrations

```

- Enact Migrations

```bash
  python manage.py migrate

```

- Start the server

```bash
  python manage.py runserver
```


## Features

- User Authentication: Sign-up, login, and logout functionality with secure password storage.
- Face Recognition: Automatic recognition of faces uploaded into the system that are present/seen in the connected camera feed.
- Weapon Detection: Automatic detection of weapons in connected camera feed.
- Notification Logs: Automatic logging of recognized people and weapons into the database with timestamps and the camera feed with a 30 min recency time for faces and 2 mins for weapons.
- SMS Notifications: Automatic Generation and sending of SMS texts to preconfigured party about the most recent Notification Log entry.

## Environment Variables

To run this project, you will need to add the following environment variables to your .env file

- Camera3.py
  
  URL = '' (For streaming over the wifi)

- Models.py___Twilio Config #ADD THEM TO YOUR .env file

  account_sid = ''
  auth_token = ''
  sending_number = ''
  receiving_number = ''





opencv_python==4.7.0.72
face_recognition==1.3.0
imutils==0.5.4
Django==4.2
python==3.9.13
twilio==9.6.5
torch==2.3.0
scipy==1.13.1
torchvision==0.18.0
insightface==0.7.3 --no-deps
onnxruntime==1.19.2 --no-deps
protobuf==6.31.1
typing_extensions==4.14.1
flatbuffers==25.2.10
coloredlogs==15.0.1
onnx==1.18.0
sckiit-image==0.24.0
albumentations==1.3.1 --no-deps
cython==3.1.2
easydict==1.13
prettytable==3.16.0 
scikit-learn==1.6.1
qudida==0.0.4 --no-deps
numpy==1.26.4
