import cv2
import numpy as np

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('model/trained_model2.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
font = cv2.FONT_HERSHEY_SIMPLEX

cam = cv2.VideoCapture(0)
while True:
    ret, im = cam.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.2, 4)
    
    for(x, y, w, h) in faces:
        Id, conf = recognizer.predict(gray[y:y+h, x:x+w])
        
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 260, 0), 5)
        cv2.putText(im, "Leaser Disabled!", (x-30, y+280), font, 1, (255, 255, 0), 3)
        cv2.putText(im, "Door Is Unlocked For ID :" + str(Id), (x - 100, y + 310), font, 1, (255, 255, 0), 3)

    cv2.imshow('Recognition Of Face', im)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
