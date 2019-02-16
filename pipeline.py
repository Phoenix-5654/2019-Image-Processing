import cv2
import numpy as np
from math import sqrt


def pipline(frame):
    original = frame
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    frame = cv2.inRange(frame, np.array([59, 188, 21]), np.array([93, 255, 255]))
    frame = cv2.erode(frame, np.ones((5, 5), np.uint8), iterations=1)
    frame = cv2.dilate(frame, np.ones((3, 3), np.uint8), iterations=1)
    cv2.imshow("binary", frame)
    contours, hierarchy = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    shapes = []

    if len(contours) > 0:
        # contours = sorted(contours, key=lambda x: cv2.contourArea(x))
        for contour in contours:
            rect = cv2.minAreaRect(contour)
            # print(rect)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            print("box", box)
            first = [1000000, 1000000]
            second = [1000000, 1000000]
            for point in list(box):
                if point[1] < first[1]:
                    second = first
                    first = point
                elif point[1] < second[1]:
                    second = point
            print("first", first)
            print("second", second)
            shapes.append((second, rect[0]))
            original = cv2.drawContours(original, [box], 0, (0, 0, 255), 5)
            original = cv2.circle(original, tuple(second), 5, (100, 10, 100), 5)
        dists = {}
        for i in range(len(shapes)):
            for j in range(i + 1, len(shapes)):
                dists[distance(shapes[i][0], shapes[j][0])] = get_average_point(shapes[i][1],
                                                                                shapes[j][1])
        if len(dists) > 0:
            print(dists)
            print(dists[min(dists.keys())])
            original = cv2.circle(original, dists[min(dists.keys())], 5,
                                  (255, 0, 0), thickness=5)
            cv2.imshow("damaged", original)


def distance(point1: tuple, point2: tuple):
    return sqrt(pow(point1[0] - point2[0], 2) + pow(point1[1] - point2[0], 2))


def get_average_point(point1, point2):
    return (int(point1[0] + point2[0]) // 2,
            int(point1[1] + point2[1]) // 2)


def main():
    cam = cv2.VideoCapture(1)
    if not cam.isOpened():
        print("camera is not found")
        return
    cv2.namedWindow("test")
    while True:
        ret, frame = cam.read()
        cv2.imshow("original", frame)
        pipline(frame)
        value = cv2.waitKey(1)
        if value == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
