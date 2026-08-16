import cv2
import aruco_compat

for id in range(4):
    img = aruco_compat.generate_marker(id, side_pixels=140)
    cv2.imshow("Aruco", img)
    cv2.imwrite(f"ArucoID{id}.png", img)
    cv2.waitKey(100)
