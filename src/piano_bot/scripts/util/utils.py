import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

def showImage(image, title=None):
	if title != None:
		plt.title(title)
	plt.imshow(image)
	plt.show()

def saveImage(image, fileName="image.jpg"):
	cv.imwrite(filename=f"/home/odroid/catkin_ws/src/static/keyboard_images/{fileName}", img=image)

def expand2rgb(image):
    return np.transpose(np.stack([image]*3), axes=(1, 2, 0))

def writeLabel(baseImage, y, x, text):
    cv.putText(baseImage, text, (x - 10, y + 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color=255, thickness=1)

def fullAnnotation(baseImage, yCenMass, xCenMass, whiteNotes, whiteComponents):
	for index in range(len(whiteComponents)):
		writeLabel(baseImage, yCenMass[index], xCenMass[index], whiteNotes[index])

	return baseImage

def grayScaleImage(srcImg):
    """
    Simple function used to gray scale an image.

    @param srcImg: Base RGB colour image to grayscale in np.array form
    @return grayImg: The gray scaled image in np.array form.
    """
    redLayer = (srcImg[:, :, 0]*0.299).astype(np.uint8)
    greenLayer = (srcImg[:, :, 1]*0.587).astype(np.uint8)
    blueLayer = (srcImg[:, :, 2]*0.114).astype(np.uint8)

    grayImg = np.add(np.add(redLayer, greenLayer), blueLayer)

    return grayImg

def generateGaussianKernel(size, sigma):
    size = size//2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    n = 1/ (2 * np.pi * sigma**2)
    kernel = n * np.exp(-1*((x**2 + y**2)/(2*sigma**2)))

    return kernel

def imageConvolution(kernel, grayImage):
    lenKernel = len(kernel)
    imHeight, imWidth = grayImage.shape
    paddedIm = np.pad(grayImage, (lenKernel-1, lenKernel-1))

    blurredIm = []
    for i in range(imHeight):
        for j in range(imWidth):
            blurredIm.append(np.sum(paddedIm[i:i+lenKernel, j:j+lenKernel]*kernel))
        
    ret = np.array(blurredIm).reshape((imHeight, imWidth))
    return ret

def sobelFilter(grayImage):
    xKernel = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])
    yKernel = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
    ])

    imgX = imageConvolution(xKernel, grayImage)
    imgY = imageConvolution(yKernel, grayImage)

    gradients = np.hypot(imgX, imgY)
    gradients = (gradients/np.max(gradients)) * 255
    theta = np.arctan2(imgY, imgX)

    return gradients, theta

def nonMaxSuppression(image, theta):
    imHeight, imWidth = image.shape
    tempMatrix = np.zeros((imHeight, imWidth))
    angles = theta * (180/np.pi)
    angles[angles < 0] += 180

    for i in range(1, imHeight-1):
        for j in range(1, imWidth-1):
            x = 255
            y = 255

            if (0 <= angles[i, j] < 22.5) or (157.5 <= angles[i, j] < 180):
                x = image[i, j+1]
                y = image[i, j-1]
            elif (22.5 <= angles[i, j] < 67.5):
                x = image[i+1, j-1]
                y = image[i-1, j+1]
            elif (67.5 <= angles[i, j] < 112.5):
                x = image[i+1, j]
                y = image[i-1, j]
            elif (112.5 <= angles[i, j] < 157.5):
                x = image[i-1, j-1]
                y = image[i+1, j+1]
            
            if (image[i, j] >= x) and (image[i, j] >= y):
                tempMatrix[i, j] = image[i, j]
            else:
                tempMatrix[i, j] = 0

    return tempMatrix

def thresholdImage(image, lowerTreshold, upperThreshold):
    ret = np.zeros(image.shape)

    highThres = np.max(image) * upperThreshold
    lowThres = highThres * lowerTreshold

    weakEdgeVal = 1
    strongEdgeVal = 255

    strongIndices = np.where(image >= highThres)
    weakIndices = np.where((image >= lowThres) & (image <= highThres))

    ret[strongIndices] = strongEdgeVal
    ret[weakIndices] = weakEdgeVal
 
    return ret, weakEdgeVal, strongEdgeVal

def hysteresis(image, weakEdgeVal, strongEdgeVal):
    imHeight, imWidth = image.shape

    for i in range(1, imHeight-1):
        for j in range(1, imWidth-1):
            if image[i, j] == weakEdgeVal:
                neighbours = np.array([image[i+1, j-1], image[i+1, j], image[i+1, j+1], image[i, j-1], image[i, j+1], image[i-1, j-1], image[i-1, j], image[i-1, j+1]])
                if strongEdgeVal in neighbours:
                    image[i, j] = 1
                else:
                    image[i, j] = 0
    
    return image

def imageMorphology(image, kernelSize=3, operation = 0):
    kernel = np.full((kernelSize, kernelSize), fill_value=255)
    imHeight, imWidth = image.shape

    paddedImage = padImage(image, (kernelSize-2))

    padHeight, padWidth = paddedImage.shape
    heightDiff, widthDiff = (padHeight - imHeight), (padWidth - imWidth)

    subMatrices = np.array([
        paddedImage[i:(i + kernelSize), j:(j + kernelSize)] for i in range(padHeight - heightDiff) for j in range(padWidth - widthDiff)
    ])
    
    if operation == 0:
        morphedImage = np.array([
            255 if (i == kernel).any() else 0 for i in subMatrices
        ])
    else:
        morphedImage = np.array([
            255 if (i == kernel).all() else 0 for i in subMatrices
        ])

    morphedImage = morphedImage.reshape((imHeight, imWidth))

    return morphedImage

def addBorder(image, borderSize=5, borderVal = 255):
    imHeight, imWidth = image.shape

    image[0:borderSize, :] = borderVal
    image[imHeight - borderSize:, :] = borderVal
    image[:, 0:borderSize] = borderVal
    image[:, imWidth - borderSize:] = borderVal

    return image

def cannyEdgeDetector(image):
    gKernel = generateGaussianKernel(3, 1.4)
    blurred = imageConvolution(gKernel, image)
    grad, theta = sobelFilter(blurred)
    suppressed = nonMaxSuppression(grad, theta)
    thresh, weak, strong = thresholdImage(suppressed, 0.015, 0.1)
    edges = hysteresis(thresh, weak, strong)


    # fig, axs = plt.subplots(2, 3)
    # axs[0, 0].imshow(image)
    # axs[0, 0].set_title("Gray image")
    # axs[0, 1].imshow(blurred)
    # axs[0, 1].set_title("Blurred image")
    # axs[0, 2].imshow(grad)
    # axs[0, 2].set_title("Gradients")
    # axs[1, 0].imshow(suppressed)
    # axs[1, 0].set_title("Non max suppression")
    # axs[1, 1].imshow(thresh)
    # axs[1, 1].set_title("Double threshold")
    # axs[1, 2].imshow(edges)
    # axs[1, 2].set_title("Edges by hysteresis")
    # fig.tight_layout()
    # plt.show()

    return np.array(edges, dtype=np.uint8)

def nnz(neighbours):
    count = 0
    for neighbour in neighbours:
        if neighbour != 0:
            count += 1

    return count


def padImage(grayImg, padValue = 1):
    paddedImg = np.pad(grayImg, ((padValue, padValue),), mode="constant")

    return paddedImg


def removePadding(paddedImg):
    baseImg = np.delete(paddedImg, [0, -1], axis=1)
    baseImg = np.delete(baseImg, [0, -1], axis=0)

    return baseImg


def connectedComponents(srcImg, connectivity=4):
    paddedImg = padImage(srcImg)

    newLabel = 1
    equivalencyList = {}

    for i in range(1, paddedImg.shape[0] - 1):
        for j in range(1, paddedImg.shape[1] - 1):
            if paddedImg[i][j] == 1:
                if connectivity == 4:
                    neighbours = np.array([paddedImg[i-1][j], paddedImg[i][j-1]])
                elif connectivity == 8:
                    neighbours = np.array([paddedImg[i-1][j], paddedImg[i][j-1], paddedImg[i-1][j-1], paddedImg[i-1][j+1]])
                else:
                    print("Unexpected value for connectivity")
                    break

                if nnz(neighbours) == 0:
                    paddedImg[i][j] = newLabel
                    newLabel += 1
                elif nnz(neighbours) == 1:
                    paddedImg[i][j] = np.max(neighbours)
                else:
                    label = np.min(neighbours[np.where(neighbours != 0)])
                    paddedImg[i][j] = label

                    for index in np.where(neighbours != 0)[0]:
                        tempLabel = neighbours[index]

                        if tempLabel != label:
                            if label not in equivalencyList.keys():
                                equivalencyList[label] = [tempLabel]
                            else:
                                if tempLabel not in equivalencyList[label]:
                                    equivalencyList[label].append(tempLabel)

    compMatrix = removePadding(paddedImg)

    ignoreKey = []

    for key in equivalencyList.keys():
        if key not in ignoreKey:
            lenVals = len(equivalencyList[key])
            i = 0

            while i < lenVals:
                val = equivalencyList[key][i]
                if val in equivalencyList.keys() and val not in ignoreKey:
                    # equivalencyList[key].pop(i)
                    equivalencyList[key] = equivalencyList[key] + equivalencyList[val]
                    equivalencyList[key] = list(np.unique(equivalencyList[key]))

                    i = 0
                    lenVals = len(equivalencyList[key])
                    if val not in ignoreKey:
                        ignoreKey.append(val)
                else:
                    i += 1
        else:
            continue

    for key in ignoreKey:
        equivalencyList.pop(key)

    newEquivalentList = []
    for key in equivalencyList.keys():
        for val in equivalencyList[key]:
            newEquivalentList.append((val, key))

    newEquivalentList = np.array(newEquivalentList).T

    if len(newEquivalentList) > 0:
        for i in range(len(newEquivalentList[0])):
            compMatrix[np.where(compMatrix == newEquivalentList[0][i])] = newEquivalentList[1][i]

    for i in range(1, len(np.unique(compMatrix))):
        compMatrix[np.where(compMatrix == np.unique(compMatrix)[i])] = i

    count = len(np.unique(compMatrix)) - 1

    return count, compMatrix

def cropImage(grayImage, colourImage):
    threshHold = round(0.6 * np.max(grayImage))

    top = np.where(grayImage[:, 2] > threshHold)[0][0]
    left = np.where(grayImage[top, :] > threshHold)[0][0]

    right = np.max(np.where(grayImage[top] > threshHold)[0])
    bottom = np.max(np.where(grayImage[:, left] > threshHold)[0])

    if top > 5:
        top += 5
    if bottom < len(grayImage)-5:
        bottom -= 5

    croppedGray = grayImage[top:bottom, left:right]
    croppedColour = colourImage[top:bottom, left:right]

    return croppedGray, croppedColour

def binariseImage(image, threshValue = 127):
    initConversion = np.where((image <= threshValue), image, 255)
    finalConversion = np.where((initConversion > threshValue), initConversion, 0)

    return finalConversion

def closeImage(image, morphLevel = 3):
    dilated = imageMorphology(image, morphLevel, 0)
    closedImage = imageMorphology(dilated, morphLevel, 1)
    
    return closedImage

def openImage(image, morphLevel = 3):
    eroded = imageMorphology(image, morphLevel, 1)
    openImage = imageMorphology(eroded, morphLevel, 9)
    
    return openImage

if __name__ == "__main__":
	pass
