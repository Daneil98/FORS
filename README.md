
FACIAL AND OBJECT DETECTION WEB-APP

A webapp that takes camera feed and runs facial detection and weapon detection. API Endpoints will also created to allow for CRUD operations.

## Run Locally

Clone the project

```bash
  git clone https://github.com/Daneil98/FORS
```

Go to the project directory

```bash
  cd FACIAL_OBJECT_RECOGNITION_SYSTEM
```

Install dependencies

```bash
  pip install -r requirements.txt

```

Start the server

```bash
  python manage.py runserver
```


## Features

- User Authentication: Sign-up, login, and logout functionality with secure password storage.
- Face Detection: Automatic detection of faces in connected camera feed.
- Gun Detection: Automatic detection of guns in connected camera feed.

## Environment Variables(NOT NECESSARY FOR NOW)

To run this project, you will need to add the following environment variables to your .env file

CELERY_BROKER_URL = ''
CELERY_ACCEPT_CONTENT = ['json']
CELERY_TASK_SERIALIZER = 'json'
