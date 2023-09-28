import cv2
import numpy as np 
from tensorflow.keras.models import load_model


def InitializePredictionModel() :
    model = load_model("model-OCR.h5")

    return model


# ImagePreProcessor Function
def ImagePreProcessor(img) :
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # image to Gray Scale
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)   # added Gaussian Blur
    imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, 1, 1, 11, 2) # applied adaptive threshold

    return imgThreshold


# BiggestContour Function
def BiggestContour(contours) :
    biggest = np.array([])
    maxArea = 0

    for i in contours :
        area = cv2.contourArea(i)

        if area > 50 :
            perimeter = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02*perimeter, True)

            if area > maxArea and len(approx) == 4 :
                biggest = approx
                maxArea = area

    return biggest, maxArea


# Reorder Function
def Reorder(points) :
    points = points.reshape((4, 2))
    pointsNew = np.zeros((4, 1, 2), dtype = np.int32)
    add = points.sum(1)
    pointsNew[0] = points[np.argmin(add)]
    pointsNew[3] = points[np.argmax(add)]
    diff = np.diff(points, axis = 1)
    pointsNew[1] = points[np.argmin(diff)]
    pointsNew[2] = points[np.argmax(diff)]

    return pointsNew


# SplitBoxes Function
def SplitBoxes(img) :
    rows = np.vsplit(img, 9)
    boxes = []

    for i in rows :
        cols = np.hsplit(i, 9)

        for box in cols :
            boxes.append(box)
    
    return boxes


# GetPrediction Function
def GetPrediction(boxes, model) :
    result = []

    for image in boxes :
        # Prepare image
        img = np.asarray(image)
        img = img[4:img.shape[0]-4, 4:img.shape[1]-4]
        img = cv2.resize(img, (48, 48))
        img = img / 255
        img = img.reshape(1, 48, 48, 1)

        # Get prediction
        prediction = model.predict(img)
        classIndex = np.argmax(prediction, axis = -1)
        probabilityValue = np.amax(prediction)
        #print(classIndex, probabilityValue)

        # Save the result
        if probabilityValue > 0.8 :
            result.append(classIndex[0])
        else : 
            result.append(0)
    
    return result


# DisplayNumbers Function 
def DisplayNumbers(img, numbers, color = (0, 255, 0)) :
    secW = int(img.shape[1]/9)
    secH = int(img.shape[0]/9)

    for x in range (0, 9) : 
        for y in range (0, 9) : 
            if numbers[(y*9)+x] != 0 :
                cv2.putText(
                    img,                     # Image on which to draw the text
                    str(numbers[(y*9)+x]),   # Text to display (convert to string)
                    (x*secW+int(secW/2)-10, int((y+0.8)*secH)),  # Position of the text
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, # Font type
                    2,                       # Font scale
                    color,                   # Text color (B, G, R)
                    2,                       # Thickness of the text
                    cv2.LINE_AA              # Anti-aliasing
                )
 
    return img



# Draw 
def DrawGrid(img):
    secW = int(img.shape[1]/9)
    secH = int(img.shape[0]/9)
    for i in range (0,9):
        pt1 = (0,secH*i)
        pt2 = (img.shape[1],secH*i)
        pt3 = (secW * i, 0)
        pt4 = (secW*i,img.shape[0])
        cv2.line(img, pt1, pt2, (255, 255, 0),2)
        cv2.line(img, pt3, pt4, (255, 255, 0),2)
    return img



