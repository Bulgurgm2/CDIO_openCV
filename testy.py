import cv2
import numpy as np
from matplotlib import pyplot as plt

try:
    cap = cv2.VideoCapture(2)
    _, frame = cap.read()
    cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
except:
    cap = cv2.VideoCapture(0)


def kk(x):
    pass


cv2.namedWindow("hej")
cv2.createTrackbar("a1", 'hej', 0, 255, kk)
cv2.createTrackbar("b1", 'hej', 0, 255, kk)
cv2.createTrackbar("c1", 'hej', 0, 255, kk)
cv2.createTrackbar("a2", 'hej', 255, 255, kk)
cv2.createTrackbar("b2", 'hej', 255, 255, kk)
cv2.createTrackbar("c2", 'hej', 255, 255, kk)

while(True):

    _, frame = cap.read()

    frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    a1 = cv2.getTrackbarPos("a1", 'hej')
    b1 = cv2.getTrackbarPos("b1", 'hej')
    c1 = cv2.getTrackbarPos("c1", 'hej')

    a2 = cv2.getTrackbarPos("a2", 'hej')
    b2 = cv2.getTrackbarPos("b2", 'hej')
    c2 = cv2.getTrackbarPos("c2", 'hej')



    red_lower = np.array([0, 182, 163])
    red_upper = np.array([23, 255, 255])
    white_lower = np.array([0, 0, 200])
    white_upper = np.array([50, 50, 255])
    brown_lower = np.array([9, 44, 119])
    brown_upper = np.array([53, 239, 255])



    lower = np.array([a1, b1, c1])
    upper = np.array([a2, b2, c2])
    mask = cv2.inRange(frame2, lower, upper)
    mask_white = cv2.inRange(frame2, white_lower, white_upper)
    mask_red = cv2.inRange(frame2, red_lower, red_upper)
    mask_brown = cv2.inRange(frame2, brown_lower, brown_upper)
    mask_comp = mask_white | mask_red
    res = cv2.bitwise_and(frame, frame, mask=mask)


    edges = cv2.Canny(res, 100, 200)

    cv2.imshow('Masked', res)
    cv2.imshow('Direct from camera', frame)
    cv2.imshow('Edges', edges)


    if cv2.waitKey(1) == ord('q'):
        print("------------- ")
        print(lower)
        print(upper)
        print("------------- ")

        break
    elif cv2.waitKey(1) == ord('s'):
        print("------------- ")
        print(lower)
        print(upper)
        print("------------- ")


cap.release()
cv2.destroyAllWindows()