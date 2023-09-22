import cv2
from tkinter import *
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

offset = 20
imgSize = 300
    
labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "iloveyou", "thankyou", "eat"]

# List to store previous inputs
previous_inputs = []

# Function to update the label text in the GUI
def update_label_text(predictions):
    index = max(range(len(predictions)), key=lambda i: predictions[i])
    letter = labels[index]
    label_text.set(letter)
    # Append the current letter to the list of previous inputs
    previous_inputs.append(letter)
    # Update the Listbox widget to show the history of inputs
    listbox_inputs.delete(0, END)
    for input_letter in previous_inputs:
        listbox_inputs.insert(END, input_letter)

# Create a simple Tkinter GUI window
root = Tk()
root.title("Hand Gesture Recognition")
label_text = StringVar()
label_text.set("")

# Create a label to display the current hand gesture letter
label_output = Label(root, textvariable=label_text, font=("Helvetica", 100))
label_output.pack(pady=20)

# Create a Listbox to display the history of inputs
listbox_inputs = Listbox(root, width=20, height=10, font=("Helvetica", 20))
listbox_inputs.pack(pady=20)

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        imgCropShape = imgCrop.shape
        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            update_label_text(prediction)

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            update_label_text(prediction)

        cv2.rectangle(imgOutput, (x - offset, y - offset - 50), (x - offset + 90, y - offset - 50 + 50),
                      (255, 0, 255), cv2.FILLED)
        cv2.putText(imgOutput, labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
        cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (255, 0, 255), 4)

    cv2.imshow("Image", imgOutput)
    cv2.waitKey(1)

    # Update the GUI to display the current hand gesture letter and the history of inputs
    root.update_idletasks()
    root.update()
