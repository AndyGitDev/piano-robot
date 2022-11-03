#!/usr/bin/env python3
from time import sleep
import rospy
import numpy as np
import sys
import tkinter as tk
import cv2 as cv

from PIL import Image, ImageTk

from util.utils import showImage

from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats
from std_msgs.msg import Float32, UInt16, String, Empty

NOTE= [
"C1",
"C#1",
"D1",
"D#1",
"E1",
"F1",
"F#1",
"G1",
"G#1",
"A1",
"A#1",
"B1",
"C2",
"C#2",
"D2",
"D#2",
"E2",
"F2",
"F#2",
"G2",
"G#2",
"A2",
"A#2",
"B2",
"C3",
"C#3",
"D3",
"D#3",
"E3",
"F3",
"F#3",
"G3",
"G#3",
"A3",
"A#3",
"B3",
"C4",
"C#4",
"D4",
"D#4",
"E4",
"F4",
"F#4",
"G4",
"G#4",
"A4",
"A#4",
"B4",
"C5",
"C#5",
"D5",
"D#5",
"E5",
"F5",
"F#5",
"G5",
"G#5",
"A5",
"A#5",
"B5",
"C6",
"C#6",
"D6",
"D#6",
"E6",
"F6",
"F#6",
"G6",
"G#6",
"A6",
"A#6",
"B6",
"C7",
"C#7",
"D7",
"D#7",
"E7",
"F7",
"F#7",
"G7",
"G#7",
"A7",
"A#7",
"B7",
"C8",
]

currPosition = 0.0
keyDistances = [0]
whiteKeyDistances = np.array([0])
blackKeyDistances = np.array([0])
annotatedImage = np.array([[0]])
detectedNotes = []

rospy.init_node("controller_node")
ctrlCmdPublisher = rospy.Publisher('controller_cmds', UInt16, queue_size=10)
stepperPublisher = rospy.Publisher('stepper_ctrl', Float32, queue_size=10)
imgPublisher = rospy.Publisher('process_image', UInt16, queue_size=1)
micPublisher = rospy.Publisher('record_signal', Empty, queue_size=1)

def currPosCallback(pos):
  global currPosition
  currPosition = pos.data

rospy.Subscriber("current_pos", Float32, currPosCallback)

def setWhiteKeyDistances(dist):
  global whiteKeyDistances
  whiteKeyDistances = np.array(dist.data).copy()

def setBlackKeyDistances(dist):
  global blackKeyDistances
  blackKeyDistances = np.array(dist.data).copy()

rospy.Subscriber('white_key_distances', numpy_msg(Floats), setWhiteKeyDistances)
rospy.Subscriber('black_key_distances', numpy_msg(Floats), setBlackKeyDistances)

def setNote(note):
  global detectedNotes, notesLabelVar

  detectedNotes.append(note.data)
  notesLabelVar.set(f"Detected notes: {detectedNotes}")
  root.update_idletasks()

rospy.Subscriber('detected_note', String, setNote)

def clickAnnotate():
  annotatedImage = cv.imread("/home/odroid/catkin_ws/src/static/keyboard_images/annotated.bmp")
  showImage(annotatedImage, "Annotated Image")

def calibrateSystem():
  global keyDistances, labelVar, notesLabelVar, root, whiteKeyDistances, blackKeyDistances, detectedNotes
  detectedNotes = []

  whiteKeyDistances = np.array([0])
  blackKeyDistances = np.array([0])

  labelVar.set("Begin calibration of motors")
  root.update_idletasks()

  ctrlCmdPublisher.publish(0)
  rospy.sleep(3)
  while (currPosition != 999.0):
    continue
  
  labelVar.set("Begin image processing")
  root.update_idletasks()

  imgPublisher.publish(0)

  while (len(whiteKeyDistances) == 1) or (len(blackKeyDistances) == 1):
    continue

  keyDistances = np.sort(np.concatenate((whiteKeyDistances, blackKeyDistances)))

  labelVar.set("Begin calibration of distances")
  root.update_idletasks()

  for i in range(4, len(keyDistances)//2):
    if (keyDistances[i] in whiteKeyDistances):
      idx = np.where(whiteKeyDistances == keyDistances[i])
      whiteKeyDistances[idx] -= 8
    elif (keyDistances[i] in blackKeyDistances):
      idx = np.where(blackKeyDistances == keyDistances[i])
      blackKeyDistances[idx] -= 8
    keyDistances[i] -= 8
  
  for i in range(len(keyDistances)//2, round(len(keyDistances)*(0.75))):
    if (keyDistances[i] in whiteKeyDistances):
      idx = np.where(whiteKeyDistances == keyDistances[i])
      whiteKeyDistances[idx] -= 9
    elif (keyDistances[i] in blackKeyDistances):
      idx = np.where(blackKeyDistances == keyDistances[i])
      blackKeyDistances[idx] -= 9
    keyDistances[i] -= 9
  
  for i in range(round(len(keyDistances)*(0.75)), len(keyDistances)):
    if (keyDistances[i] in whiteKeyDistances):
      idx = np.where(whiteKeyDistances == keyDistances[i])
      whiteKeyDistances[idx] -= 0
    elif (keyDistances[i] in blackKeyDistances):
      idx = np.where(blackKeyDistances == keyDistances[i])
      blackKeyDistances[idx] -= 0
    keyDistances[i] -= 0

  for i in range(len(keyDistances)):

    dist = keyDistances[i]

    labelVar.set(f"Calibrating key {i+1}")
    stepperPublisher.publish(dist)
    rospy.sleep(1)
    while (round(currPosition) != round(dist)):
      continue

    if (keyDistances[i] in whiteKeyDistances):
      idx = np.where(whiteKeyDistances == keyDistances[i])
      ctrlCmdPublisher.publish(1)
    elif (keyDistances[i] in blackKeyDistances):
      idx = np.where(blackKeyDistances == keyDistances[i])
      ctrlCmdPublisher.publish(2)


    root.update_idletasks()

    rospy.sleep(2)
  
  labelVar.set("Calibration complete")
  root.update_idletasks()

if __name__ == "__main__":
  root = tk.Tk()
  root.title('Piano bot')
  root.geometry('800x600')

  labelVar = tk.StringVar()
  labelVar.set("System start")
  label = tk.Label(root, textvariable=labelVar)

  notesLabelVar = tk.StringVar()
  notesLabelVar.set(f"Detected notes: {detectedNotes}")
  notesLabel = tk.Label(root, textvariable=notesLabelVar, wraplength=100)

  label.grid(row=0, column=0)
  notesLabel.grid(row=1, column=0)

  showImageButton = tk.Button(root, text="Show annotated image", command=clickAnnotate)
  showImageButton.grid(row=0, column=4)
  calButton = tk.Button(root, text='Calibrate System', command=calibrateSystem)
  calButton.grid(row=1, column=4)

  root.mainloop()