import cv2
import numpy as np
from math import sqrt


def pipline(frame):
    original = frame
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    frame = cv2.inRange(frame, np.array([36, 0, 234]), np.array([89, 90, 255]))
    frame = cv2.erode(frame, np.ones((3, 3), np.uint8), iterations=5)
    frame = cv2.dilate(frame, np.ones((3, 3), np.uint8), iterations=6)
    cv2.imshow("binary", frame)
    contours, hierarchy = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []

    if len(contours) > 0:
        # contours = sorted(contours, key=lambda x: cv2.contourArea(x))
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            ratio = w / h

            if 0.77 > ratio > 0.38:
                rect = cv2.minAreaRect(contour)
                print(rect)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                print("box", box)
                boxes.append(box)
                original = cv2.drawContours(original, [box], 0, (0, 0, 255), 5)
                original = cv2.circle(original, tuple(map(int, box[0])), 5, (0, 0, 0), 5)
                original = cv2.circle(original, tuple(map(int, box[1])), 5, (0, 255, 0), 5)
                original = cv2.circle(original, tuple(map(int, box[2])), 5, (255, 0, 0), 5)
                original = cv2.circle(original, tuple(map(int, box[3])), 5, (255, 255, 0), 5)

        # dists = { distance(points[0], points[1]) for points in itertools.combinations(
        # [box[0] for box in boxes], 2) : (points[0], points[1])}

        dists = {}
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                dists[distance(boxes[i][0], boxes[j][0])] = get_average_point(boxes[i][0],
                                                                              boxes[j][0],
                                                                              boxes[i][1],
                                                                              boxes[j][1])
        if len(dists) > 0:
            print(dists)
            print(dists[min(dists.keys())])
            original = cv2.circle(original, dists[min(dists.keys())], 5,
                                  (255, 0, 0), thickness=5)
            cv2.imshow("damaged", original)


def distance(point1: tuple, point2: tuple):
    return sqrt(pow(point1[0] - point2[0], 2) + pow(point1[1] - point2[0], 2))


def get_average_point(point1, point2, shape1, shape2):
    return ((point1[0] + point2[0] + shape1[0] + shape2[0]) // 2,
            (point1[1] + point2[1] + shape1[1] + shape2[1]) // 2)


if __name__ == '__main__':
    cam = cv2.VideoCapture(1)
    cv2.namedWindow("test")
    while True:
        ret, frame = cam.read()
        cv2.imshow("original", frame)
        pipline(frame)
        value = cv2.waitKey(1)
        if value == 'q':
            break

    cam.release()
    cv2.destroyAllWindows()
