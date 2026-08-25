"""Small adapter for the OpenCV 4.7+ ArUco API used by this project."""

import cv2


def dictionary():
    return cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)


def detector_parameters():
    return cv2.aruco.DetectorParameters()


def detect_markers(frame, marker_dictionary=None, parameters=None):
    marker_dictionary = marker_dictionary or dictionary()
    parameters = parameters or detector_parameters()
    detector = cv2.aruco.ArucoDetector(marker_dictionary, parameters)
    return detector.detectMarkers(frame)


def generate_marker(marker_id, side_pixels=140):
    image = cv2.aruco.generateImageMarker(dictionary(), marker_id, side_pixels)
    return image
