import cv2
import numpy as np
import math

class HandDetect:
    def __init__(self):
        pass

    def handGesture(self, image):
        try:  # an error comes if it does not find anything in window as it cannot find contour of max area
            # therefore this try error statement

            #image = cv2.flip(image, 1)
            kernel = np.ones((3, 3), np.uint8)

            # define region of interest
            roi = image[240:480, 0:320]

            cv2.rectangle(image, (0, 240), (320, 480), (0, 255, 0), 0)
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            # define range of skin color in HSV
            lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin = np.array([20, 255, 255], dtype=np.uint8)

            # extract skin colur imagw
            mask = cv2.inRange(hsv, lower_skin, upper_skin)

            # extrapolate the hand to fill dark spots within
            mask = cv2.dilate(mask, kernel, iterations=4)

            # blur the image
            mask = cv2.GaussianBlur(mask, (5, 5), 100)

            # find contours
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # find contour of max area(hand)
            cnt = max(contours, key=lambda x: cv2.contourArea(x))

            # approx the contour a little 손을  오리발처럼 외곽선을 땀
            epsilon = 0.0005 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            # make convex hull around hand 외곽선찾기 알고리즘
            hull = cv2.convexHull(cnt)

            # define area of hull and area of hand 보통 areahull > areacnt 다.
            areahull = cv2.contourArea(hull) #최외곽선 면적값구하는 할고리즘
            areacnt = cv2.contourArea(cnt) #윤곽선의 면적

            # find the percentage of area not covered by hand in convex hull
            arearatio = ((areahull - areacnt) / areacnt) * 100

            # find the defects in convex hull with respect to hand
            hull = cv2.convexHull(approx, returnPoints=False)
            defects = cv2.convexityDefects(approx, hull)

            # l = no. of defects
            l = 0

            # code for finding no. of defects due to fingers
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(approx[s][0])
                end = tuple(approx[e][0])
                far = tuple(approx[f][0])
                pt = (100, 180)

                # find length of all sides of triangle
                a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                s = (a + b + c) / 2
                ar = math.sqrt(s * (s - a) * (s - b) * (s - c))

                # distance between point and convex hull
                d = (2 * ar) / a

                # apply cosine rule here
                angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 57

                # ignore angles > 90 and ignore points very close to convex hull(they generally come due to noise)
                if angle <= 90 and d > 30:
                    l += 1
                    cv2.circle(roi, far, 3, [255, 0, 0], -1)

                # draw lines around hand
                cv2.line(roi, start, end, [0, 255, 0], 2)

            l += 1

            # print corresponding gestures which are in their ranges
            font = cv2.FONT_HERSHEY_SIMPLEX

            check_point = -1

            if l == 1:
                if areacnt < 2000:
                    #cv2.putText(image, 'Put hand in the box', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
                    check_point = -1
                else:
                    if arearatio < 12:
                        check_point = 0

                    elif arearatio < 17.5:
                        check_point = 1

                    else:
                        check_point = 1

            elif l == 2:
                check_point = 2

            elif l == 3:

                if arearatio < 27:
                    check_point = 3

                else:
                    check_point = 3

            elif l == 4:
                check_point = 4

            elif l == 5:
                check_point = 5

            elif l == 6:
                #cv2.putText(image, 'reposition', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
                check_point = -1

            else:
                #cv2.putText(image, 'reposition', (10, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
                check_point = -1

            self.return_point = check_point
            return self.return_point
        except:
            pass
