from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import cv2
import numpy as np

face_classifier = cv2.CascadeClassifier(r'C:\\Users\\Pankaj\\Downloads\\Compressed\\Project\\Human_Emotion_Detection-master\\haarcascade_frontalface_default.xml')
classifier = load_model(r'C:\\Users\\Pankaj\\Downloads\\Compressed\\Project\\Human_Emotion_Detection-master\\model.h5')


emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# image_path = 'rifat1.jpg'
image_path = 'rifat2.jpg'

frame = cv2.imread(image_path)

if frame is None:
    print("Could not read the image.")
    exit()

desired_width = 640
desired_height = 480

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
faces = face_classifier.detectMultiScale(gray)

for (x, y, w, h) in faces:
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
    roi_gray = gray[y:y + h, x:x + w]
    roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

    if np.sum([roi_gray]) != 0:
        roi = roi_gray.astype('float') / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

 
        prediction = classifier.predict(roi)[0]
        label = emotion_labels[prediction.argmax()]
        label_position = (x, y - 10)
        
      
        font_scale = 2
        font_thickness = 3

        cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), font_thickness)
    else:
        cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)


frame = cv2.resize(frame, (desired_width, desired_height))


cv2.imshow('Emotion Detector', frame)
cv2.waitKey(0)  
cv2.destroyAllWindows()
