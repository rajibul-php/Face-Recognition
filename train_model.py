import tkinter as tk                        # for buttons and labels
from tkinter import *
import cv2                                  # opencv python
import csv                                  # comma separated value
import os                                   # operating system
import numpy as np                          # for array and mathematical functions
from PIL import Image, ImageTk              # python image library
import pandas as pd                         # for high performance  manipulation and analyzing tools
import datetime                             # for date
import time                                 # for time

# Make a Window

window = tk.Tk()                                        # from tkinter library we make an object
window.title("Facial Image Capture And Process")        # give window name
window.geometry('400x300')                              # give window size
window.configure()                                      # apply above to the configuration of window

# Capturing image for dataset

def take_img():                                                                     # declare function
    cam = cv2.VideoCapture(0)                                                       # open webcam to capture image
    detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')         # apply default classifier
    ID = txt.get()                                                                  # get it from text box
    Name = txt2.get()                                                               # get name from text box
    sampleNum = 0                                                                   # declare variable
    while (True):
        ret, img = cam.read()                                                       # read the camera input
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                                # convert BGR to GRAY
        faces = detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # incrementing sample number
            sampleNum = sampleNum + 1
            print (sampleNum)
            # saving the captured face in the dataset folder
            cv2.imwrite("dataset/ " + Name + "." + ID + '.' + str(sampleNum) + ".jpg",
                        gray[y:y + h, x:x + w])
            cv2.imshow('Capturing Image', img)
            # wait for 100 miliseconds
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            # break if the sample number is morethan 20
        elif sampleNum > 20:
            break
    cam.release()
    cv2.destroyAllWindows()

    res = "Images Saved!  ID: " + ID + " Name : " + Name
    Notification.configure(text=res, bg="SpringGreen3", width=40, font=('times', 12, 'bold'))
    Notification.place(x=15, y=80)


###For train the model
def trainimg():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    global detector
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    try:
        global faces, Id
        faces, Id = getImagesAndLabels("dataset")
    except Exception as e:
        l = 'please make "dataset" folder & put Images'
        Notification.configure(text=l, bg="SpringGreen3", width=40, font=('times', 12, 'bold'))
        Notification.place(x=15, y=80)

    recognizer.train(faces, np.array(Id))
    try:
        recognizer.save("model/trained_model2.yml")
    except Exception as e:
        q = 'Please make "model" folder'
        Notification.configure(text=q, bg="SpringGreen3", width=40, font=('times', 12, 'bold'))
        Notification.place(x=15, y=40)

    res = "Image Processed Successfully! "  # +",".join(str(f) for f in Id)
    Notification.configure(text=res, bg="SpringGreen3", width=40, font=('times', 12, 'bold'))
    Notification.place(x=15, y=80)


def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    # create  face list
    faceSamples = []
    # create empty ID list
    Ids = []
    # now looping through all the image paths and loading the Ids and the images
    for imagePath in imagePaths:
        # loading the image and converting it to gray scale
        pilImage = Image.open(imagePath).convert('L')
        # Now we are converting the PIL image into numpy array
        imageNp = np.array(pilImage, 'uint8')
        # getting the Id from the image

        Id = int(os.path.split(imagePath)[-1].split(".")[1])
        # extract the face from the training image sample
        faces = detector.detectMultiScale(imageNp)
        # If a face is there then append that in the list as well as Id of it
        for (x, y, w, h) in faces:
            faceSamples.append(imageNp[y:y + h, x:x + w])
            Ids.append(Id)
    return faceSamples, Ids


window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)

Notification = tk.Label(window, text="All things good", bg="Green", fg="white", width=15, height=3)

lbl = tk.Label(window, text="ID", width=20, height=2, fg="black", font=('times', 12))
lbl.place(x=-70, y=0)


def testVal(inStr, acttyp):
    if acttyp == '1':  # insert
        if not inStr.isdigit():
            return False
    return True


txt = tk.Entry(window, validate="key", width=10, fg="black")
txt['validatecommand'] = (txt.register(testVal), '%P', '%d')
txt.place(x=70, y=10)

lbl2 = tk.Label(window, text="Name", width=10, fg="black", height=2, font=('times', 12))
lbl2.place(x=100, y=0)

txt2 = tk.Entry(window, width=20, fg="black")
txt2.place(x=180, y=10)

takeImg = tk.Button(window, text="Capture", command=take_img, fg="black", bg="Seagreen", width=7, height=2,
                    activebackground="Green", font=('times', 12,))
takeImg.place(x=50, y=200)

trainImg = tk.Button(window, text="Process", fg="black", command=trainimg, bg="Seagreen", width=7, height=2,
                     activebackground="Green", font=('times', 12,))

trainImg.place(x=190, y=200)

window.mainloop()
