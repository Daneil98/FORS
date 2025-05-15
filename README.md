
  ORION'S SHIELD
- A Facial Recognition and weapon detection web-app

This webapp that takes live video feed from a connected camera and runs facial recognition with the poplar python 'face_recongition' model on the images of people uploaded into the system and weapon detection with a gun_cascade.xml file gotten from GeeksforGeeks.com pending the completion of a custom-trained YOLOv5 model.

API Endpoints will also created to allow for CRUD operations.

- NOTE: Please ensure CMake and C++ SDK from Visual Studio is installed on yor local machine



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
- Face Recognition: Automatic recognition of faces uploaded into the system that present connected camera feed.
- Gun Detection: Automatic detection of guns in connected camera feed.
- Notification Log: Automatic logging of recognized people and weapons into the database with timestamps and the camera feed with a 30 min recency time.
- SMS Notifications: Automatic Generation and sending of SMS texts to preconfigured party about the most recent Notification Log entry.

## Environment Variables

To run this project, you will need to add the following environment variables to your .env file

- Camera1.py
  
  URL = '' (For streaming over the wifi)

- Models.py___Twilio Config
  to = '' (phone number to receive SMS)
  
  from_= '' (Twilio Phone number)
  
  account_sid = ''
  
  auth_token = ''

- CELERY_BROKER_URL = ''

- CELERY_ACCEPT_CONTENT = ['json']

- CELERY_TASK_SERIALIZER = 'json'
