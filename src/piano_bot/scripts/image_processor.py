#!/usr/bin/env python3
import rospy

import cv2 as cv
import numpy as np
import os

from util.utils import *
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats
from std_msgs.msg import UInt16

def getEdges(grayImage):
    edges = cannyEdgeDetector(grayImage)
    edges = addBorder(edges, 5)
    closedImage = closeImage(edges, 3)

    binEdges = 1 - (closedImage > 0).astype(np.uint8)
    binEdges = addBorder(binEdges, 1, 0)

    return binEdges

def getWhiteComponents(grayImage):
    # Edge detection and connected components
    edges = getEdges(grayImage)

    numComponents, compMatrix = connectedComponents(edges)

    # Filter out non white elements from image
    whiteComponents = []
    patchSizes = []
    for index in range(1, numComponents):
        grayPatch = grayImage[np.where(compMatrix == index)]
        patchSize = np.prod(grayPatch.shape)

        if (np.sum(grayPatch) / patchSize > 100) and (patchSize > 150):
            whiteComponents.append((index, patchSize))
            patchSizes.append(patchSize)

    avgSize = np.average(patchSizes)

    temp = []
    for val in whiteComponents:
        if ((0.5 * avgSize) < val[1] < (2 * avgSize)):
            temp.append(val[0])

    whiteComponents = temp

    yCenMass, xCenMass = [], []
    for component in whiteComponents:
        zeroes = np.zeros_like(grayImage)
        zeroes[np.where(compMatrix == component)] = 1
        nonZeroIndices = np.nonzero(zeroes)
        yTemp = int(np.mean(nonZeroIndices[0]))
        xTemp = int(np.mean(nonZeroIndices[1]))
        yCenMass.append(yTemp)
        xCenMass.append(xTemp)

    return whiteComponents, np.array(yCenMass), np.array(xCenMass), compMatrix


def classifyWhiteKeys(whiteComponents, yCenMass, xCenMass, compMatrix, grayImage):
    xCenMass, yCenMass, whiteComponents = zip(*sorted(zip(xCenMass, yCenMass, whiteComponents), key=lambda zipped: zipped[0]))

    blackKey = []

    blackComponents = []
    xCenBlack, yCenBlack = [], []

    for index in range(len(whiteComponents) - 1):
        whiteKey1 = np.where(compMatrix == whiteComponents[index])
        whiteKey2 = np.where(compMatrix == whiteComponents[index + 1])
        borderPoints = []

        maxVal = max(whiteKey1[0][0], whiteKey2[0][0])
        minVal = min(whiteKey1[0][-1], whiteKey2[0][-1])

        for y in range(maxVal, minVal):
            maxXKey1 = max(whiteKey1[1][np.where(whiteKey1[0] == y)])
            minXKey2 = min(whiteKey2[1][np.where(whiteKey2[0] == y)])

            borderPoints.append((int((minXKey2 + maxXKey1)/2), y))

        inspectRad = 5
        inspectMidPts = sorted(borderPoints, key=lambda pts: pts[1])
        inspectMidPts = inspectMidPts[int(0.1 * len(inspectMidPts)): int(0.3 * len(inspectMidPts))]
        inspectX = int(np.average([pts[0] for pts in inspectMidPts]))
        inspectY = int(np.average([pts[1] for pts in inspectMidPts]))

        patch = grayImage[(inspectY - inspectRad): (inspectY + inspectRad), (inspectX - inspectRad):(inspectX + inspectRad)]
        patchVal = np.sum(patch)

        blackThreshhold = inspectRad*inspectRad*4*255*0.2
        if patchVal < blackThreshhold and patchVal > 0:
            blackKey.append(1)
            blackComponents.append(compMatrix[yCenBlack, xCenBlack])
            yCenBlack.append(inspectY)
            xCenBlack.append(inspectX)
        else:
            blackKey.append(0)

    convList = np.convolve(np.array(blackKey), [1, 1, 1], mode='same')
    possibleFKeys = list(np.where(convList == 3)[0])
    fKeys = [key-1 for key in possibleFKeys]

    notes = []
    blackNotes = []
    keysOrder = ['F', 'G', 'A', 'B', 'C', 'D', 'E']
    blackKeysOrder = ['F#', 'G#', 'A#', 'C#', 'D#']

    for index in range(len(whiteComponents)):
        offset = (index - fKeys[0]) % 7
        notes.append(keysOrder[offset])
    
    for index in range(len(xCenBlack)):
        offset = (index - (fKeys[0] - 1)) % 5
        blackNotes.append(blackKeysOrder[offset])

    return whiteComponents, yCenMass, xCenMass, notes, blackComponents, yCenBlack, xCenBlack, blackNotes


def captureImage(camPort = 0, fileName = "keyboard.jpeg"):
    cam = cv.VideoCapture(camPort)

    initFrames = 60
    for _ in range(initFrames):
        cam.read()

    result, image = cam.read()

    # im = np.around(255.0 * (np.array(image[0:240, :, :])[:, :, ::-1]/ 255.0)**(2.2))
    # showImage(im.astype(np.uint16))

    return image[0:240, :, :]

def annotateKeyboard():

    im = captureImage()

    grayIm = grayScaleImage(im)

    croppedGray, croppedColour = cropImage(grayIm, im)

    whiteComponents, yCenMass, xCenMass, compMatrix = getWhiteComponents(croppedGray)

    whiteComponents, yCenMass, xCenMass, notes, blackComponents, yCenBlack, xCenBlack, blackNotes = classifyWhiteKeys(whiteComponents, yCenMass, xCenMass, compMatrix, croppedGray)

    notedImage = fullAnnotation(croppedColour.copy(), yCenMass, xCenMass, notes, whiteComponents)
    notedImage = fullAnnotation(notedImage.copy(), yCenBlack, xCenBlack, blackNotes, blackComponents)
    
    whiteKeyDistances, blackKeyDistances = determineKeyDistances(xCenMass, xCenBlack)

    saveImage(notedImage.astype(np.uint16), "annotated.bmp")

    return np.array(notedImage, dtype=np.float32), np.array(whiteKeyDistances, dtype=np.float32), np.array(blackKeyDistances, dtype=np.float32)


def processImage(data):
    rospy.loginfo("Processing image")

    wDistPublisher = rospy.Publisher('white_key_distances', numpy_msg(Floats), queue_size=10)
    bDistPublisher = rospy.Publisher('black_key_distances', numpy_msg(Floats), queue_size=10)

    annotatedImage, whiteKeyDistances, blackKeyDistances = annotateKeyboard()

    wDistPublisher.publish(np.array(whiteKeyDistances, dtype=np.float32))
    bDistPublisher.publish(np.array(blackKeyDistances, dtype=np.float32))

    # print(annotatedImage)
    # showImage(annotatedImage.astype(np.uint16))
    rospy.loginfo("Generated annotated image")

def main():
    rospy.init_node("im_proc_node")
    rospy.loginfo("Initialise image processing node")
    rospy.Subscriber('process_image', UInt16, processImage)

    rospy.spin()

if __name__ == "__main__":
    main()
