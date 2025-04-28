from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io
import cv2
import os

app = Flask(__name__)
model = load_model('model.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload-image', methods=['POST'])
def upload_image():
    try:
        file = request.files['file']
        if not file:
            return jsonify({'error': 'No file uploaded.'})

        # Load face detector
        face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

        # Read uploaded image
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        if len(faces) == 0:
            return jsonify({'error': 'No face detected in image.'})

        emotions = []

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            prediction = model.predict(roi)[0]
            label = emotion_labels[prediction.argmax()]
            emotions.append(label)

        return jsonify({'detected_emotions': emotions})

    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/upload-video', methods=['POST'])
def upload_video():
    try:
        file = request.files['file']
        if not file:
            return jsonify({'error': 'No video uploaded.'})

        temp_video_path = 'temp_video.mp4'
        file.save(temp_video_path)

        face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        cap = cv2.VideoCapture(temp_video_path)

        emotions = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                prediction = model.predict(roi)[0]
                label = emotion_labels[prediction.argmax()]
                emotions.append(label)

        cap.release()
        os.remove(temp_video_path)

        if not emotions:
            return jsonify({'error': 'No faces detected in video.'})

        # Most frequent emotion in video
        most_common_emotion = max(set(emotions), key=emotions.count)

        return jsonify({'most_common_emotion': most_common_emotion})

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
