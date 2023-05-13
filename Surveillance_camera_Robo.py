import datetime
import time

import cv2
import pickle

cap = cv2.VideoCapture(0)
labels = {"person_name": 1}
with open('labels.pkl', 'rb') as f:
    og_label = pickle.load(f)
    labels = {v: k for k, v in og_label.items()}

img = cv2.imread('assets/tushar.jpg', -1)
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml')
width = int(cap.get(3))
height = int(cap.get(4))
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
body_cascades = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')
detection = False
detection_stopped_time = None
timer_Satrted = False
seconds_to_detect = 5
frame_size = (int(cap.get(3)), int(cap.get(4)))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, width, height) in faces:
        roi_gray = gray[y:y + height, x:x + width]
        roi_color = frame[y:y + height, x:x + width]
        id_, conf = recognizer.predict(roi_gray)
        if conf >= 65:
            print(id_)
            print(labels[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            img = cv2.putText(frame, labels[id_], (10, height - 10), font, 4, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x + height, y + height), (0, 255, 0), 2)
            if detection:
                timer_Satrted = False
            else:
                detection = True
                current_time = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
                out = cv2.VideoWriter(f"{current_time}video.mp4", fourcc, 20, frame_size)

    if detection:
        if timer_Satrted:
            if time.time() - detection_stopped_time >= seconds_to_detect:
                detection = False
                timer_Satrted = False
                out.release()
        else:
            timer_Satrted = True
            detection_stopped_time = time.time()
    if detection:
        out.write(frame)

    cv2.imshow('mask', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()
