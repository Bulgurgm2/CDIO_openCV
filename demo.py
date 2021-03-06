import cv2
import numpy as np
import time

try:
    cap = cv2.VideoCapture(2)
    _, frame = cap.read()
    cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
except:
    cap = cv2.VideoCapture(0)

#  setting for demo

x_coordinate = 500
y_coordinate = 500
speed = 1

# Radius of circle
radius = 20
radius2 = 30

# Blue color in BGR
color = (255, 0, 0)
color2 = (0, 0, 255)
color3 = (0, 255, 0)

# Line thickness of 2 px
thickness = 2

window_name = 'Image'

tresholde = 7

#red_lower = np.array([0, 182, 163])
#red_upper = np.array([23, 255, 255])

white_lower = np.array([0, 0, 190])
white_upper = np.array([190, 128, 255])

red_lower = np.array([75, 23, 139])
red_upper = np.array([107, 90, 229])

# font
font = cv2.FONT_HERSHEY_SIMPLEX

# org
org = (50, 50)

# fontScale
fontScale = 1



class Bane:

    def __init__(self):
        self.list_of_balls = []
        self.border = None

    def add_cordinats(self, xy):
        for ball in self.list_of_balls:
            if (xy[0] - tresholde) < ball.x < (xy[0] + tresholde) and (xy[1] - tresholde) < ball.y < (xy[1] + tresholde):
                ball.number += 1
                ball.x = int((ball.x + xy[0])/2)
                ball.y = int((ball.y + xy[1])/2)
                return
        self.list_of_balls.append(Bold(xy))

    def remove_false_balls(self):
        new_list = []
        for ball in self.list_of_balls:
            if ball.number > 10:
                new_list.append(ball)
        self.list_of_balls = new_list


class Bold:

    def __init__(self, xy):
        self.x = xy[0]
        self.y = xy[1]
        self.number = 1


def main():

    global x_coordinate, y_coordinate
    bane, red_edge = analyse()

    while 1:
        start_time = time.time()

        #  read image.
        _, img = cap.read()

        for xy in bane.list_of_balls:
            img = cv2.circle(img, (xy.x,xy.y), 8, (0, 255, 0), 2)
            img = cv2.circle(img, (xy.x,xy.y), 1, (0, 0, 255), 3)

        img = cv2.putText(img, f"balls: {len(bane.list_of_balls)}", (500, 50), font, fontScale, color, thickness,
                          cv2.LINE_AA)

        target = bane.list_of_balls[0]

        if target.x > x_coordinate:
            x_move = speed
        if target.y > y_coordinate:
            y_move = speed

        if target.x < x_coordinate:
            x_move = -speed
        if target.y < y_coordinate:
            y_move = -speed



        danger_box = []

        for index_x in range(x_coordinate-radius2, x_coordinate+radius2):
            danger_box.append((index_x, y_coordinate-radius2))
            danger_box.append((index_x, y_coordinate+radius2))

        for index_y in range(y_coordinate-radius2, y_coordinate+radius2):
            danger_box.append((x_coordinate-radius2, index_y))
            danger_box.append((x_coordinate+radius2, index_y))

        if not set(bane.borders).isdisjoint(danger_box):
            x_coordinate += x_move * -2
            y_coordinate += y_move * -4
        else:
            x_coordinate += x_move
            y_coordinate += y_move

        if (target.x, target.y) in danger_box:
            bane.list_of_balls.remove(target)

        # Center coordinates
        center_coordinates = (x_coordinate, y_coordinate)

        # Using cv2.circle() method
        # Draw a circle with blue line borders of thickness of 2 px
        img = cv2.circle(img, center_coordinates, radius, color, thickness)
        img = cv2.rectangle(img, (x_coordinate - radius2, y_coordinate - radius2), (x_coordinate + radius2, y_coordinate + radius2), color2, thickness)



        #  draw image
        if cv2.waitKey(1) == ord('q'):
            break
        img = fps(start_time, img)

        cv2.imshow("Detected Circle", img)
        cv2.imshow("red", red_edge)







def analyse():

    bane = Bane()
    list_of_ball_coordinate = []

    for frame_nr in range(70):
        start_time = time.time()

        #  read image.
        _, img = cap.read()

        #  give time to focus before analysing frames
        if frame_nr > 19:

            #  convert img to HSV format
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            #  make color masks and apply them on img
            mask_white = cv2.inRange(img_hsv, white_lower, white_upper)
            white_res = cv2.bitwise_and(img, img, mask=mask_white)

            mask_red = cv2.inRange(img_hsv, red_lower, red_upper)
            red_res = cv2.bitwise_and(img, img, mask=mask_red)
            red_edges = cv2.Canny(red_res, 100, 200)


            #  convert white_res to grayscale.
            gray = cv2.cvtColor(white_res, cv2.COLOR_BGR2GRAY)
            
            #  apply blur effect on gray
            gray_blurred = cv2.blur(gray, (2, 2))
            #gray_blurred = gray

            cv2.imshow("reed", gray_blurred)


            #  apply Hough transform on the blurred image.
            detected_circles = cv2.HoughCircles(gray_blurred,
                                                cv2.HOUGH_GRADIENT, 1.5, 5, param1=250,
                                                param2=19, maxRadius=9, minRadius=5)
            #  draw circles that are detected.
            if detected_circles is not None:

                #  convert the circle parameters a, b and r to integers.
                detected_circles = np.uint16(np.around(detected_circles))


                for pt in detected_circles[0, :]:
                    a, b, r = pt[0], pt[1], pt[2]

                    #  draw the circumference of the circle.
                    img = cv2.circle(img, (a, b), r, (0, 255, 0), 2)

                    #  draw a small circle (of radius 1) to show the center.
                    img = cv2.circle(img, (a, b), 1, (0, 0, 255), 3)

                    #  append xy coordinates to list
                    list_of_ball_coordinate.append((a,b))


        #  draw image
        if cv2.waitKey(1) == ord('q'):
            break
        img = fps(start_time, img)

        cv2.imshow("Detected Circle", img)

    #  edges detection on red_res to find borders

    indices = np.where(red_edges != [0])
    bane.borders = zip(indices[0], indices[1])
    print(set(bane.borders))
    for ball in list_of_ball_coordinate:
        bane.add_cordinats(ball)
    bane.remove_false_balls()

    return bane, red_edges



def fps(start_time, img):
    time.sleep(max(1. / 15 - (time.time() - start_time), 0))

    # Using cv2.putText() method
    img = cv2.putText(img, f"FPS: {int(1.0 / (time.time() - start_time))}", org, font,fontScale, color, thickness, cv2.LINE_AA)
    return img






if __name__ == '__main__':
    main()