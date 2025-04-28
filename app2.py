from flask import Flask, request, jsonify, render_template, Response
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import cv2
import os

# Initialize Flask app
app = Flask(__name__)

# Load Model
model = load_model('model.h5')

# Load Haarcascade
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Emotion Labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# ========= Home Route =========
@app.route('/')
def home():
    return render_template('index.html')
    
def real():
    return render_template('realtime.html')
# ========= Image Upload Route =========
@app.route('/upload-image', methods=['POST'])
def upload_image():
    try:
        file = request.files['file']
        if not file:
            return jsonify({'error': 'No file uploaded.'})

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

# ========= Video Upload Route =========
@app.route('/upload-video', methods=['POST'])
def upload_video():
    try:
        file = request.files['file']
        if not file:
            return jsonify({'error': 'No video uploaded.'})

        temp_video_path = 'temp_video.mp4'
        file.save(temp_video_path)

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

        most_common_emotion = max(set(emotions), key=emotions.count)

        return jsonify({'most_common_emotion': most_common_emotion})

    except Exception as e:
        return jsonify({'error': str(e)})

# ========= Real-Time Webcam Streaming Route =========
# Global camera object
camera = cv2.VideoCapture(0)

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(gray, 1.1, 5)

            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                prediction = model.predict(roi)[0]
                label = emotion_labels[prediction.argmax()]
                label_position = (x, y-10)

                cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,255), 2)
                cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/realtime')
def realtime():
    return render_template('realtime.html')

# ========= Run Flask App =========
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
