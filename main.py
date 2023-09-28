from utility import *
import solver

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

model = InitializePredictionModel()

##### Dimension initialization
image = "SudokuPuzzle/1.png"
height =  450
width = 450


##### Image preparation
img = cv2.imread(image) 
img = cv2.resize(img, (width, height))  # image resizing
cv2.imshow("Input", img)
imgBlank = np.zeros((height, width, 3), np.uint8)  # blank img for testing and debugging
imgThreshold = ImagePreProcessor(img) # ImagePreProcessor defined in utility

##### Contour Determination
imgContours = img.copy() 
imgBigContour = img.copy()
contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 3)


##### Biggest Contour Determination; To be used in Sudoku
biggest, maxArea = BiggestContour(contours) # BiggestContour defined in utility  
if biggest.size != 0 :
    biggest = Reorder(biggest) # Reorder defined in utility
    cv2.drawContours(imgBigContour, biggest, -1, (0, 255, 0), 10) # to draw biggest contour 
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgWarpColored = cv2.warpPerspective(img, matrix, (width, height))
    imgDetectedDigits = imgBlank.copy()
    imgWarpColored = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)


    ##### Digit Recognition 
    imgSolvedDigits = imgBlank.copy()

    boxes = SplitBoxes(imgWarpColored) 
   
    numbers = GetPrediction(boxes, model)
    
    imgDetectedDigits = DisplayNumbers(imgDetectedDigits, numbers, color = (255, 0, 255))

    numbers = np.asarray(numbers)
    
    posArray = np.where(numbers > 0, 0, 1)


    ##### Finding Solution of the board
    board = np.array_split(numbers, 9)

    try:
        solver.solve(board)
    except:
        pass

    flatList = []
    for sublist in board:
        for item in sublist:
            flatList.append(item)
    solvedNumbers = flatList*posArray
    imgSolvedDigits = DisplayNumbers(imgSolvedDigits,solvedNumbers)



    ##### 6. OVERLAY SOLUTION
    pts2 = np.float32(biggest) # PREPARE POINTS FOR WARP
    pts1 =  np.float32([[0, 0],[width, 0], [0, height],[width, height]]) # PREPARE POINTS FOR WARP
    matrix = cv2.getPerspectiveTransform(pts1, pts2)  # GER
    imgInvWarpColored = img.copy()
    imgInvWarpColored = cv2.warpPerspective(imgSolvedDigits, matrix, (width, height))
    invPerspective = cv2.addWeighted(imgInvWarpColored, 1, img, 0.5, 1)
    imgDetectedDigits = DrawGrid(imgDetectedDigits)
    imgSolvedDigits = DrawGrid(imgSolvedDigits)

    ##### Output
    cv2.imshow("Output", invPerspective)

else:
    print("No Sudoku Found")


cv2.waitKey(0)
cv2.destroyAllWindows()