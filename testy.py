import cv2
import numpy as np
import imutils
import time
import traceback
from matplotlib import pyplot as plt

try:
    cap = cv2.VideoCapture(2)
    _, frame = cap.read()
    cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
except:
    cap = cv2.VideoCapture(0)


while(True):
    start_time = time.time()

    _, frame = cap.read()

    output = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)



    red_lower = np.array([0, 182, 163])
    red_upper = np.array([23, 255, 255])
    white_lower = np.array([0, 0, 229])
    white_upper = np.array([180, 118, 255])
    white_lower2 = np.array([0, 0, 200])
    white_upper2 = np.array([80, 74, 255])
    brown_lower = np.array([9, 44, 119])
    brown_upper = np.array([53, 239, 255])

    mask_white = cv2.inRange(frame2, white_lower, white_upper)
    mask_red = cv2.inRange(frame2, red_lower, red_upper)
    mask_brown = cv2.inRange(frame2, brown_lower, brown_upper)
    mask_comp = mask_white | mask_red
    res = cv2.bitwise_and(frame, frame, mask=mask_white)

    mask_white = cv2.GaussianBlur(mask_white, (11,11), 0)


    cnts = cv2.findContours(mask_white.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)



    # detect circles in the image
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 100)
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
            # draw the circle in the output image, then draw a rectangle
            # corresponding to the center of the circle
            cv2.circle(output, (x, y), r, (0, 255, 0), 4)
            cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

            if cv2.waitKey(1) == ord('q'):
                break


    edges = cv2.Canny(res, 100, 200)

    #indices = np.where(edges != [0])
    #coordinates = zip(indices[0], indices[1])
    #print(indices)
    #print(set(coordinates))


    #cv2.imshow('Edges', np.hstack([mask_white, output]))
    cv2.imshow('Edges2', gray)
    cv2.imshow('Edges', output)
    cv2.imshow('Edges3', edges)

    if cv2.waitKey(1) == ord('q'):
        break

    print("FPS: ", int(1.0 / (time.time() - start_time)))

cap.release()
cv2.destroyAllWindows()