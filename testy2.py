
import cv2
import numpy as np
import time

try:
    cap = cv2.VideoCapture(2)
    _, frame = cap.read()
    cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
except:
    cap = cv2.VideoCapture(0)


while(True):
    start_time = time.time()
    # Read image.
    _, img = cap.read()


    white_lower = np.array([0, 0, 229])
    white_upper = np.array([180, 118, 255])
    frame2 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask_white = cv2.inRange(frame2, white_lower, white_upper)


    # Convert to grayscale.

    res = cv2.bitwise_and(img, img, mask=mask_white)


    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

    # Blur using 3 * 3 kernel.
    gray_blurred = cv2.blur(gray, (3, 3))




    # Apply Hough transform on the blurred image.
    detected_circles = cv2.HoughCircles(gray_blurred,
                                        cv2.HOUGH_GRADIENT, 1.8, 20, param1 = 50,
                                        param2 = 30, maxRadius = 10)

    # Draw circles that are detected.
    if detected_circles is not None:

        # Convert the circle parameters a, b and r to integers.
        detected_circles = np.uint16(np.around(detected_circles))

        for pt in detected_circles[0, :]:
            a, b, r = pt[0], pt[1], pt[2]

            # Draw the circumference of the circle.
            cv2.circle(img, (a, b), r, (0, 255, 0), 2)

            # Draw a small circle (of radius 1) to show the center.
            cv2.circle(img, (a, b), 1, (0, 0, 255), 3)


    cv2.imshow("Detected Circle", img)
    cv2.imshow("mask", gray_blurred)
    cv2.imshow("res", res)
    cv2.imshow("white mask", mask_white)

    if cv2.waitKey(1) == ord('q'):
        break

    print("FPS: ", int(1.0 / (time.time() - start_time)))



cap.release()
cv2.destroyAllWindows()