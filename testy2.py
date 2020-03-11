
import cv2
import numpy as np
import time

fps = 14
time_delta = 1./fps



try:
    cap = cv2.VideoCapture(2)
    _, frame = cap.read()
    cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
except:
    cap = cv2.VideoCapture(0)

x = 0
y = 0
speed = 2

# Radius of circle
radius = 20
radius2 = 30

# Blue color in BGR
color = (255, 0, 0)
color2 = (0, 0, 255)

# Line thickness of 2 px
thickness = 2

while(True):
    start_time = time.time()
    # Read image.
    _, img = cap.read()


    red_lower = np.array([0, 182, 163])
    red_upper = np.array([23, 255, 255])

    white_lower = np.array([0, 0, 210])
    white_upper = np.array([190, 128, 255])

    frame2 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask_white = cv2.inRange(frame2, white_lower, white_upper)
    mask_red = cv2.inRange(frame2, red_lower, red_upper)
    red_res = cv2.bitwise_and(img, img, mask=mask_red)
    red_edges = cv2.Canny(red_res, 100, 200)

    # Convert to grayscale.

    res = cv2.bitwise_and(img, img, mask=mask_white)


    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

    # Blur using 3 * 3 kernel.
    gray_blurred = cv2.blur(gray, (9, 9))

    # Apply Hough transform on the blurred image.
    detected_circles = cv2.HoughCircles(gray_blurred,
                                        cv2.HOUGH_GRADIENT, 1.8, 10, param1 = 300,
                                        param2 = 22, maxRadius = 11, minRadius=4)

    # Draw circles that are detected.
    if detected_circles is not None:

        # Convert the circle parameters a, b and r to integers.
        detected_circles = np.uint16(np.around(detected_circles))

        for pt in detected_circles[0, :]:
            a, b, r = pt[0], pt[1], pt[2]

            print(r)

            # Draw the circumference of the circle.
            cv2.circle(img, (a, b), r, (0, 255, 0), 2)

            # Draw a small circle (of radius 1) to show the center.
            cv2.circle(img, (a, b), 1, (0, 0, 255), 3)

        indices = np.where(red_edges != [0])
        coordinates = zip(indices[0], indices[1])

        if a > x:
            x_move = speed
        if b > y:
            y_move = speed

        if a < x:
            x_move = -speed
        if b < y:
            y_move = -speed



        danger_box = []

        for index_x in range(x-radius2, x+radius2):
            danger_box.append((index_x, y-radius2))
            danger_box.append((index_x, y+radius2))

        for index_y in range(y-radius2, y+radius2):
            danger_box.append((x-radius2, index_y))
            danger_box.append((x+radius2, index_y))

        if not set(coordinates).isdisjoint(danger_box):
            x += x_move * -2
            y += y_move * -4
        else:
            x += x_move
            y += y_move







    # Center coordinates
    center_coordinates = (x, y)

    # Using cv2.circle() method
    # Draw a circle with blue line borders of thickness of 2 px
    img = cv2.circle(img, center_coordinates, radius, color, thickness)
    img = cv2.rectangle(img, (x-radius2,y-radius2), (x+radius2,y+radius2), color2, thickness)

    cv2.imshow("Detected Circle", img)
    cv2.imshow("mask", mask_red)


    if cv2.waitKey(1) == ord('q'):
        break


    time.sleep(max(1. / 14 - (time.time() - start_time), 0))
    #print("FPS: ", int(1.0 / (time.time() - start_time)))





cap.release()
cv2.destroyAllWindows()