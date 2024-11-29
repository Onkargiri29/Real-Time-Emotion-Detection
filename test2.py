from flask import Flask, render_template, Response
import cv2
from keras.models import model_from_json
import numpy as np
import threading
import time
import os

app = Flask(__name__)

# Load model
with open("emotiondetector.json", "r") as json_file:
    model_json = json_file.read()
model = model_from_json(model_json)
model.load_weights("emotiondetector.h5")

# Initialize face detector
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Define labels for emotions
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

webcam = None

def extract_features(image):
    """Preprocesses the face image for emotion detection."""
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

def generate_frames():
    """Generates video frames with emotion detection."""
    global webcam
    webcam = cv2.VideoCapture(0)

    last_frame_time = time.time()

    while True:
        success, frame = webcam.read()
        if not success:
            break

        # Limit frame rate to ~30 FPS
        current_time = time.time()
        if current_time - last_frame_time < 0.03:  # 30 FPS limit
            continue
        last_frame_time = current_time

        # Convert to grayscale and detect faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (p, q, r, s) in faces:
            face_image = gray[q:q+s, p:p+r]
            face_image = cv2.resize(face_image, (48, 48))
            img = extract_features(face_image)
            pred = model.predict(img)
            prediction_label = labels[pred.argmax()]

            # Draw a rectangle around the face
            cv2.rectangle(frame, (p, q), (p + r, q + s), (255, 0, 0), 2)
            # Put text for the emotion
            cv2.putText(frame, prediction_label, (p, q - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Resize frame for smoother streaming
        frame = cv2.resize(frame, (640, 480))  # Resize to reduce resolution and improve FPS

        # Encode the frame to JPEG and convert to bytes
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    webcam.release()  # Release webcam when done

@app.route('/')
def index():
    """Renders the main webpage."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Feeds the video stream to the frontend."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if _name_ == '_main_':
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
