#!/usr/bin/env python3
from heapq import nsmallest
from math import sqrt, tan, radians
import itertools
import json
from enum import Enum

import cv2
import numpy as np
from networktables import NetworkTables

from cscore import CameraServer, UsbCamera, VideoSource

HORIZONTAL_RES = 240
VERTICAL_RES = 320
FPS = 90
ERROR_VALUE = 0

SET_POINT = (VERTICAL_RES // 2 + 28, HORIZONTAL_RES // 2 + 43)
picamera_config_json = ""


class GripPipeline:
    """
    An OpenCV pipeline generated by GRIP.
    """

    def __init__(self):
        """initializes all values to presets or None if need to be set
        """

        self.__blur_type = BlurType.Box_Blur
        self.__blur_radius = 4.504504504504505

        self.blur_output = None

        self.__hsv_threshold_input = self.blur_output
        self.__hsv_threshold_hue = [20.51079136690646, 100.13651877133107]
        self.__hsv_threshold_saturation = [0.0, 255.0]
        self.__hsv_threshold_value = [96.31294964028777, 255.0]

        self.hsv_threshold_output = None

        self.__cv_dilate_src = self.hsv_threshold_output
        self.__cv_dilate_kernel = None
        self.__cv_dilate_anchor = (-1, -1)
        self.__cv_dilate_iterations = 0.0
        self.__cv_dilate_bordertype = cv2.BORDER_CONSTANT
        self.__cv_dilate_bordervalue = (-1)

        self.cv_dilate_output = None

        self.__find_contours_input = self.cv_dilate_output
        self.__find_contours_external_only = False

        self.find_contours_output = None

        self.__filter_contours_contours = self.find_contours_output
        self.__filter_contours_min_area = 250.0
        self.__filter_contours_min_perimeter = 50.0
        self.__filter_contours_min_width = 0
        self.__filter_contours_max_width = 1000
        self.__filter_contours_min_height = 0.0
        self.__filter_contours_max_height = 1000
        self.__filter_contours_solidity = [0.0, 100.0]
        self.__filter_contours_max_vertices = 1000000
        self.__filter_contours_min_vertices = 0
        self.__filter_contours_min_ratio = 0.4
        self.__filter_contours_max_ratio = 5.0

        self.filter_contours_output = None

        self.__convex_hulls_contours = self.filter_contours_output

        self.convex_hulls_output = None

    def process(self, source0):
        """
        Runs the pipeline and sets all outputs to new values.
        """
        # Step Blur0:
        self.__blur_input = source0
        (self.blur_output) = self.__blur(self.__blur_input, self.__blur_type, self.__blur_radius)

        # Step HSV_Threshold0:
        self.__hsv_threshold_input = self.blur_output
        (self.hsv_threshold_output) = self.__hsv_threshold(self.__hsv_threshold_input, self.__hsv_threshold_hue,
                                                           self.__hsv_threshold_saturation, self.__hsv_threshold_value)

        # Step CV_dilate0:
        self.__cv_dilate_src = self.hsv_threshold_output
        (self.cv_dilate_output) = self.__cv_dilate(self.__cv_dilate_src, self.__cv_dilate_kernel,
                                                   self.__cv_dilate_anchor, self.__cv_dilate_iterations,
                                                   self.__cv_dilate_bordertype, self.__cv_dilate_bordervalue)

        # Step Find_Contours0:
        self.__find_contours_input = self.cv_dilate_output
        (self.find_contours_output) = self.__find_contours(self.__find_contours_input,
                                                           self.__find_contours_external_only)

        # Step Filter_Contours0:
        self.__filter_contours_contours = self.find_contours_output
        (self.filter_contours_output) = self.__filter_contours(self.__filter_contours_contours,
                                                               self.__filter_contours_min_area,
                                                               self.__filter_contours_min_perimeter,
                                                               self.__filter_contours_min_width,
                                                               self.__filter_contours_max_width,
                                                               self.__filter_contours_min_height,
                                                               self.__filter_contours_max_height,
                                                               self.__filter_contours_solidity,
                                                               self.__filter_contours_max_vertices,
                                                               self.__filter_contours_min_vertices,
                                                               self.__filter_contours_min_ratio,
                                                               self.__filter_contours_max_ratio)

        # Step Convex_Hulls0:
        self.__convex_hulls_contours = self.filter_contours_output
        (self.convex_hulls_output) = self.__convex_hulls(self.__convex_hulls_contours)

    @staticmethod
    def __blur(src, type, radius):
        """Softens an image using one of several filters.
        Args:
            src: The source mat (numpy.ndarray).
            type: The blurType to perform represented as an int.
            radius: The radius for the blur as a float.
        Returns:
            A numpy.ndarray that has been blurred.
        """
        if (type is BlurType.Box_Blur):
            ksize = int(2 * round(radius) + 1)
            return cv2.blur(src, (ksize, ksize))
        elif (type is BlurType.Gaussian_Blur):
            ksize = int(6 * round(radius) + 1)
            return cv2.GaussianBlur(src, (ksize, ksize), round(radius))
        elif (type is BlurType.Median_Filter):
            ksize = int(2 * round(radius) + 1)
            return cv2.medianBlur(src, ksize)
        else:
            return cv2.bilateralFilter(src, -1, round(radius), round(radius))

    @staticmethod
    def __hsv_threshold(input, hue, sat, val):
        """Segment an image based on hue, saturation, and value ranges.
        Args:
            input: A BGR numpy.ndarray.
            hue: A list of two numbers the are the min and max hue.
            sat: A list of two numbers the are the min and max saturation.
            lum: A list of two numbers the are the min and max value.
        Returns:
            A black and white numpy.ndarray.
        """
        out = cv2.cvtColor(input, cv2.COLOR_BGR2HSV)
        return cv2.inRange(out, (hue[0], sat[0], val[0]), (hue[1], sat[1], val[1]))

    @staticmethod
    def __cv_dilate(src, kernel, anchor, iterations, border_type, border_value):
        """Expands area of higher value in an image.
        Args:
           src: A numpy.ndarray.
           kernel: The kernel for dilation. A numpy.ndarray.
           iterations: the number of times to dilate.
           border_type: Opencv enum that represents a border type.
           border_value: value to be used for a constant border.
        Returns:
            A numpy.ndarray after dilation.
        """
        return cv2.dilate(src, kernel, anchor, iterations=(int)(iterations + 0.5),
                          borderType=border_type, borderValue=border_value)

    @staticmethod
    def __find_contours(input, external_only):
        """Sets the values of pixels in a binary image to their distance to the nearest black pixel.
        Args:
            input: A numpy.ndarray.
            external_only: A boolean. If true only external contours are found.
        Return:
            A list of numpy.ndarray where each one represents a contour.
        """
        if (external_only):
            mode = cv2.RETR_EXTERNAL
        else:
            mode = cv2.RETR_LIST
        method = cv2.CHAIN_APPROX_SIMPLE
        im2, contours, hierarchy = cv2.findContours(input, mode=mode, method=method)
        return contours

    @staticmethod
    def __filter_contours(input_contours, min_area, min_perimeter, min_width, max_width,
                          min_height, max_height, solidity, max_vertex_count, min_vertex_count,
                          min_ratio, max_ratio):
        """Filters out contours that do not meet certain criteria.
        Args:
            input_contours: Contours as a list of numpy.ndarray.
            min_area: The minimum area of a contour that will be kept.
            min_perimeter: The minimum perimeter of a contour that will be kept.
            min_width: Minimum width of a contour.
            max_width: MaxWidth maximum width.
            min_height: Minimum height.
            max_height: Maximimum height.
            solidity: The minimum and maximum solidity of a contour.
            min_vertex_count: Minimum vertex Count of the contours.
            max_vertex_count: Maximum vertex Count.
            min_ratio: Minimum ratio of width to height.
            max_ratio: Maximum ratio of width to height.
        Returns:
            Contours as a list of numpy.ndarray.
        """
        output = []
        for contour in input_contours:
            x, y, w, h = cv2.boundingRect(contour)
            if (w < min_width or w > max_width):
                continue
            if (h < min_height or h > max_height):
                continue
            area = cv2.contourArea(contour)
            if (area < min_area):
                continue
            if (cv2.arcLength(contour, True) < min_perimeter):
                continue
            hull = cv2.convexHull(contour)
            solid = 100 * area / cv2.contourArea(hull)
            if (solid < solidity[0] or solid > solidity[1]):
                continue
            if (len(contour) < min_vertex_count or len(contour) > max_vertex_count):
                continue
            ratio = (float)(w) / h
            if (ratio < min_ratio or ratio > max_ratio):
                continue
            output.append(contour)
        return output

    @staticmethod
    def __convex_hulls(input_contours):
        """Computes the convex hulls of contours.
        Args:
            input_contours: A list of numpy.ndarray that each represent a contour.
        Returns:
            A list of numpy.ndarray that each represent a contour.
        """
        output = []
        for contour in input_contours:
            output.append(cv2.convexHull(contour))
        return output


BlurType = Enum('BlurType', 'Box_Blur Gaussian_Blur Median_Filter Bilateral_Filter')


def start_camera():
    inst = CameraServer.getInstance()
    camera = UsbCamera('Hatch Panels', '/dev/video0')
    camera.setConfigJson(picamera_config_json)
    # with open("cam.json", encoding='utf-8') as cam_config:
    #     camera.setConfigJson(json.dumps(cam_config.read()))
#     camera.setResolution(VERTICAL_RES, HORIZONTAL_RES)
#     camera.setFPS(FPS)
#     camera.setBrightness(10)
#     camera.setConfigJson("""
# {
#     "fps": """ + str(FPS) + """,
#     "height": """ + str(HORIZONTAL_RES) + """,
#     "pixel format": "mjpeg",
#     "properties": [
#             {
#                 "name": "brightness",
#                 "value": 0
#             },
#             {
#                 "name": "contrast",
#                 "value": 100
#             },
#             {
#                 "name": "saturation",
#                 "value": 100
#             },
#             {
#                 "name": "color_effects",
#                 "value": 3
#             }
#     ],
#     "width": """ + str(VERTICAL_RES) + """
# }
#     """)
#

    # inst.startAutomaticCapture(camera=camera)

    camera.setConnectionStrategy(VideoSource.ConnectionStrategy.kKeepOpen)

    return inst, camera

class Shape:
    def __init__(self, points, center, angle, width, height):
        self.points = points
        self.center = center
        self.angle = abs(angle)
        self.width = width
        self.height = height

    @property
    def lowest_point(self):
        return self.points[0]

    @property
    def second_highest_point(self):
        return nsmallest(2, self.points, key=lambda x: x[1])[-1]

    @property
    def highest_point(self):
        return min(self.points, key=lambda x: x[1])

    @property
    def approx_area(self):
        return self.width * self.height

    def get_middle_point(self, shape):
        return get_average_point(self.center, shape.center)


def find_alignment_center(shapes, k):
    min_val = 100000000
    point = None
    targets = None
    combinations = itertools.combinations(shapes, 2)
    for combination in combinations:
        first = combination[0]
        second = combination[1]
        if (distance(first.lowest_point, second.lowest_point)
            > distance(first.second_highest_point, second.second_highest_point)):
            val = ((distance(first.lowest_point, second.lowest_point) +
                   distance(first.second_highest_point, second.second_highest_point))
                   / (k * (first.approx_area + second.approx_area)))

            if (val < min_val):
                min_val = val
                point = first.get_middle_point(second)
                targets = (first, second)
    return point, targets


def distance(point1: tuple, point2: tuple):
    return sqrt(pow(point1[0] - point2[0], 2) + pow(point1[1] - point2[1], 2))


def get_average_point(point1, point2):
    return (((point1[0] + point2[0]) / 2,
            (point1[1] + point2[1]) / 2))


def main():
    NetworkTables.initialize(server='10.56.54.2')
    NetworkTables.setUpdateRate(0.015)

    sd = NetworkTables.getTable('Vision')
    sd.putNumber('k', 1e-09)

    inst, camera = start_camera()


    pipeline = GripPipeline()

    cvSink = inst.getVideo(camera=camera)

    outputStream = inst.putVideo(
        'Hatch Panels Convex Hulls', HORIZONTAL_RES, VERTICAL_RES)

    img = np.zeros(shape=(VERTICAL_RES, HORIZONTAL_RES, 3), dtype=np.uint8)

    while True:
        k = sd.getNumber("k", 10)

        shapes = []

        targets = None

        time, img = cvSink.grabFrame(img)

        if time == 0:

            outputStream.notifyError(cvSink.getError())

            continue

        pipeline.process(img)

        drawing = np.zeros((HORIZONTAL_RES, VERTICAL_RES, 3), np.uint8)

        if len(pipeline.convex_hulls_output) != 0:

            for hull in pipeline.convex_hulls_output:
                rect = cv2.minAreaRect(hull)
                box = cv2.boxPoints(rect)
                box = np.int0(box)

                x, y, w, h = cv2.boundingRect(hull)

                shapes.append(
                    (Shape(box, rect[0], rect[2], w, h)))

                cv2.drawContours(drawing, [box], 0, (0, 0, 255), 4)
                cv2.circle(drawing, tuple(map(int, shapes[-1].second_highest_point)), 5, (100, 60, 200), 5)
                cv2.circle(drawing, tuple(map(int, shapes[-1].lowest_point)), 5, (200, 30, 100), 5)

            for i in range(len(pipeline.convex_hulls_output)):
                cv2.drawContours(
                    drawing, pipeline.convex_hulls_output, i, (255, 0, 0), 3)

            if len(shapes) == 1:
                # alignment_center = None
                if shapes[0].angle > 45:
                    alignment_center = (VERTICAL_RES, shapes[0].center[1])
                else:
                    alignment_center = (0, shapes[0].center[1])
            else:
                alignment_center, targets = find_alignment_center(shapes, k)

            if alignment_center is not None:

                cv2.circle(drawing, tuple(
                    map(int, alignment_center)), 5, (0, 255, 0), 5)

                sd.putNumber('X Error', SET_POINT[0] - alignment_center[0])
                sd.putNumber('Y Error', SET_POINT[1] - alignment_center[1])
                target_y_angle = abs((160 - alignment_center[1]) * 24.4) / 120

                sd.putNumber(
                    'X Angle', (abs(160 - alignment_center[0]) * 31.1) / 160)
                sd.putNumber('Y Angle', target_y_angle)

                sd.putNumber(
                    'Distance', 11.5 / tan(radians(target_y_angle + 15)))
            else:
                    sd.putNumber("X Error", ERROR_VALUE)

            if targets is not None:
                pass
                # if targets[0].lowest_point[0] < targets[1].lowest_point[0]:
                #     sd.putNumber(
                #         'Target Difference', targets[0].lowest_point[1] - targets[1].lowest_point[1])
                # else:
                #     sd.putNumber(
                #         'Target Difference', targets[1].lowest_point[1] - targets[0].lowest_point[1])
                # sd.putNumber("Dist Up",
                #              abs(targets[0].highest_point[0]
                #                             - targets[1].highest_point[0]))
                # sd.putNumber("Dist Second highest",
                #              abs(targets[0].second_highest_point[0]
                #                  - targets[1].second_highest_point[0]))
                # sd.putNumber("Dist Down",
                #              abs(targets[0].lowest_point[0]
                #                               - targets[1].lowest_point[0]))
                # sum1 = (abs(targets[0].second_highest_point[0]
                #                  - targets[1].second_highest_point[0])
                #              + abs(targets[0].lowest_point[0]
                #                    - targets[1].lowest_point[0]))
                # sd.putNumber("SUM 1", sum1)
                # sum2 = (abs(targets[0].highest_point[1] - targets[0].lowest_point[1])
                #              + abs(targets[1].highest_point[1] - targets[1].highest_point[1]))
                # sd.putNumber("SUM 2", sum2)
                # sd.putNumber("FRACTION 1", sum1 / sum2)
                # sd.putNumber("FRACTION 2", sum2 / sum1)
                # sd.putNumber("FRACTION 3", abs(targets[0].second_highest_point[0]
                #                  - targets[1].second_highest_point[0]) / sum2)
                # sd.putNumber("Multiply 1", sum1 * sum2)
                # sd.putNumber("Multiply 2",  abs(targets[0].second_highest_point[0]
                #                  - targets[1].second_highest_point[0]) * sum2)
                # sd.putNumber("approx yaw", ((sum2 / sum1) - 0.3587) / 0.0053)


        else:
            sd.putNumber("X Error", ERROR_VALUE)
        outputStream.putFrame(drawing)


if __name__ == "__main__":
    picamera_config_json = """
    {
    "fps": 90,
    "height": 240,
    "pixel format": "mjpeg",
    "properties": [
        {
            "name": "connect_verbose",
            "value": 1
        },
        {
            "name": "raw_brightness",
            "value": 0
        },
        {
            "name": "brightness",
            "value": 0
        },
        {
            "name": "raw_contrast",
            "value": 100
        },
        {
            "name": "contrast",
            "value": 100
        },
        {
            "name": "raw_saturation",
            "value": 100
        },
        {
            "name": "saturation",
            "value": 100
        },
        {
            "name": "red_balance",
            "value": 1000
        },
        {
            "name": "blue_balance",
            "value": 1000
        },
        {
            "name": "horizontal_flip",
            "value": false
        },
        {
            "name": "vertical_flip",
            "value": false
        },
        {
            "name": "power_line_frequency",
            "value": 1
        },
        {
            "name": "raw_sharpness",
            "value": 0
        },
        {
            "name": "sharpness",
            "value": 50
        },
        {
            "name": "color_effects",
            "value": 0
        },
        {
            "name": "rotate",
            "value": 0
        },
        {
            "name": "color_effects_cbcr",
            "value": 32896
        },
        {
            "name": "video_bitrate_mode",
            "value": 0
        },
        {
            "name": "video_bitrate",
            "value": 10000000
        },
        {
            "name": "repeat_sequence_header",
            "value": false
        },
        {
            "name": "h264_i_frame_period",
            "value": 60
        },
        {
            "name": "h264_level",
            "value": 11
        },
        {
            "name": "h264_profile",
            "value": 4
        },
        {
            "name": "auto_exposure",
            "value": 0
        },
        {
            "name": "exposure_time_absolute",
            "value": 1000
        },
        {
            "name": "exposure_dynamic_framerate",
            "value": false
        },
        {
            "name": "auto_exposure_bias",
            "value": 12
        },
        {
            "name": "white_balance_auto_preset",
            "value": 1
        },
        {
            "name": "image_stabilization",
            "value": false
        },
        {
            "name": "iso_sensitivity",
            "value": 0
        },
        {
            "name": "iso_sensitivity_auto",
            "value": 1
        },
        {
            "name": "exposure_metering_mode",
            "value": 0
        },
        {
            "name": "scene_mode",
            "value": 0
        },
        {
            "name": "compression_quality",
            "value": 30
        }
    ],
    "width": 320
}
    """
    main()
