import cv2
import aruco_compat


marker_dictionary = aruco_compat.dictionary()
parameters = aruco_compat.detector_parameters()

cap = cv2.VideoCapture(1)


while True:
    success, img = cap.read()
    markerCorners, markerIds, rejectedCandidates = aruco_compat.detect_markers(
        img, marker_dictionary, parameters
    )
    img = cv2.aruco.drawDetectedMarkers(img, markerCorners, markerIds)
    print(markerIds)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
