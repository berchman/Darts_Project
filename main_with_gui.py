import pickle
import os
import sys
from statistics import mode
from time import sleep

import cv2
import numpy as np
from PySide6 import QtCore
from PySide6.QtCore import QThreadPool, QRunnable
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication, QMainWindow, QMessageBox

import CalibrationWithUncertainty
import ContourUtils
from Dart_Scoring import dart_scorer_util, DartScore
import opencv_gui_sliders
import utils
from FPS import FPS
from QT_GUI_Elements.qt_ui_classes import DartPositionLabel
from QT_GUI_Elements.ui_dart_main_gui import Ui_DartScorer

# #############  Config  ####################
# The bundled matrices belong to the original author's webcam, so do not use
# them with a different camera. Enable this only after adding your own files.
USE_CAMERA_CALIBRATION_TO_UNDISTORT = False
loadSavedParameters = False
CAMERA_NUMBER = int(os.environ.get("DARTS_CAMERA_INDEX", "1"))
TRIANGLE_DETECT_THRESH = 11
minArea = 800
maxArea = 4000
score1 = DartScore.Score(501, True)
score2 = DartScore.Score(501, True)
scored_values = []
scored_mults = []

# Globals
points = []
dart_tip = None
ACTIVE_PLAYER = 1
UNDO_LAST_FLAG = False
DARTBOARD_AREA = 0
center_ellipse = (0, 0)
values_of_round = []
mults_of_round = []
current_settings = None
OPENCV_GUI_CREATED = False

new_dart_tip = None
update_dart_point = False

ellipse = None
x_offset_current, y_offset_current = 0, 0

#############################################
STOP_DETECTION = False
dart_id = 0


def open_camera(camera_number):
    camera = cv2.VideoCapture(camera_number)
    camera.set(cv2.CAP_PROP_FPS, 30)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    return camera


cap = None

if USE_CAMERA_CALIBRATION_TO_UNDISTORT:
    if loadSavedParameters:
        pickle_in_MTX = open("PickleFiles/mtx_cheap_webcam_good_target.pickle", "rb")
        # pickle_in_MTX = open("PickleFiles/mtx_surface_back.pickle", "rb")
        meanMTX = pickle.load(pickle_in_MTX)
        print(meanMTX)
        pickle_in_DIST = open("PickleFiles/dist_cheap_webcam_good_target.pickle", "rb")
        # pickle_in_DIST = open("PickleFiles/dist_surface_back.pickle", "rb")
        meanDIST = pickle.load(pickle_in_DIST)
        print(meanDIST)
        print("Parameters Loaded")
    else:
        meanMTX, meanDIST, uncertaintyMTX, uncertaintyDIST = CalibrationWithUncertainty.calibrateCamera(cap=cap, rows=6,
                                                                                                        columns=9,
                                                                                                        squareSize=30,
                                                                                                        runs=1,
                                                                                                        saveImages=False,
                                                                                                        webcam=True)
target_ROI_size = (600, 600)
resize_for_squish = (600, 600)  # Squish the image if the circle doesnt quite fit
dart_board_in_gui_dimensions = (501, 501)
Scaling_factor_for_x_placing_in_gui = (dart_board_in_gui_dimensions[0] / resize_for_squish[0], dart_board_in_gui_dimensions[1] / resize_for_squish[1])

previous_img = np.zeros((target_ROI_size[0], target_ROI_size[1], 3)).astype(np.uint8)
difference = np.zeros(target_ROI_size).astype(np.uint8)
img_undist = np.zeros(target_ROI_size).astype(np.uint8)

default_img = None


class WorkerSignals(QtCore.QObject):
    finished = QtCore.Signal()


def detect_dart_circle_and_set_limits(img_roi):
    # cannyLow, cannyHigh, noGauss, minArea, erosions, dilations, epsilon, showFilters, automaticMode, threshold_new = opencv_gui_sliders.updateTrackBar()
    cannyLow = 80
    cannyHigh = 160
    noGauss = 2
    minArea = 800
    erosions = 1
    dilations = 1
    epsilon = 5 / 1000
    showFilters = 0
    global contours, radius_1, radius_2, radius_3, radius_4, radius_5, radius_6, cnt, ellipse, x, y, a, b, angle, center_ellipse, x_offset_current, \
        y_offset_current, TRIANGLE_DETECT_THRESH, DARTBOARD_AREA, current_settings
    imgContours, contours, imgCanny = ContourUtils.get_contours(img=img_roi, cThr=(cannyLow, cannyHigh),
                                                                gaussFilters=noGauss, minArea=minArea,
                                                                epsilon=epsilon, draw=False,
                                                                erosions=erosions, dilations=dilations,
                                                                showFilters=showFilters)
    radius_1, radius_2, radius_3, radius_4, radius_5, radius_6, x_offset, y_offset = opencv_gui_sliders.update_dart_trackbars()
    new_settings = [radius_1, radius_2, radius_3, radius_4, radius_5, radius_6, x_offset, y_offset]
    image_area = img_roi.shape[0] * img_roi.shape[1]
    contours = [cnt for cnt in contours if image_area * 0.5 < cnt[1] < image_area * 0.9]  # Filter out contours that are too small or too big
    # get biggest contour
    cnt = get_biggest_contour(contours)
    if cnt is None:
        return
    # Create the outermost Circle
    # if a radius changed
    if ellipse is None or new_settings != current_settings:  # Save the outermost ellipse for later to avoid useless re calculation !
        print("Recalculating the outermost ellipse")
        radius_1, radius_2, radius_3, radius_4, radius_5, radius_6, x_offset, y_offset = new_settings
        ellipse = cv2.fitEllipse(cnt[4])  # Also a benefit for stability of the outer ellipse --> not jumping from frame to frame
        # get area of ellipse
        DARTBOARD_AREA = cv2.contourArea(cnt[4])
        x, y = ellipse[0]
        a, b = ellipse[1]
        angle = ellipse[2]
        center_ellipse = (int(x + x_offset / 10), int(y + y_offset / 10))
        a = a / 2
        b = b / 2
        # set the limits also only once
        dart_scorer_util.bullsLimit = a * (radius_1 / 100)
        dart_scorer_util.singleBullsLimit = a * (radius_2 / 100)
        dart_scorer_util.innerTripleLimit = a * (radius_3 / 100)
        dart_scorer_util.outerTripleLimit = a * (radius_4 / 100)
        dart_scorer_util.innerDoubleLimit = a * (radius_5 / 100)
        dart_scorer_util.outerBoardLimit = a * (radius_6 / 100)
        current_settings = [radius_1, radius_2, radius_3, radius_4, radius_5, radius_6, x_offset, y_offset]
        print("Limits set")


class MainWindow(QMainWindow):

    def __init__(self):
        QMainWindow.__init__(self)
        self.ui = Ui_DartScorer()
        self.ui.setupUi(self)  # Set up the external generated ui
        self.setWindowIcon(QIcon('icons/dart_icon.ico'))
        self.setWindowTitle("Dart Master")

        # Buttons
        self.ui.set_default_img_button.clicked.connect(lambda: UIFunctions.set_default_image(self))
        self.ui.start_measuring_button.clicked.connect(lambda: UIFunctions.start_detection_and_scoring(self))
        self.ui.stop_measuring_button.clicked.connect(lambda: UIFunctions.stop_detection_and_scoring(self))
        self.ui.undo_last_throw_button.clicked.connect(lambda: UIFunctions.undo_last_throw(self))
        self.ui.initial_score_comboBox.currentIndexChanged.connect(lambda: UIFunctions.update_game_settings(self))
        self.ui.detection_sensitivity_slider.valueChanged.connect(lambda: UIFunctions.update_detection_sensitivity(self))
        self.ui.continue_button.clicked.connect(lambda: UIFunctions.start_detection_and_scoring(self))
        self.DartPositions = {}
        self.detection_worker = None
        self.default_image_worker = None
        self.configure_01_game_options()

        self.show()

    def warning(self, message="Default"):
        QMessageBox.about(self, "Congratulations !", message)

    def configure_01_game_options(self):
        """Offer the common 01 games while allowing any positive score ending in 01."""
        self.ui.initial_score_comboBox.setEditable(True)
        for score in (701, 901, 1001):
            if self.ui.initial_score_comboBox.findText(str(score)) == -1:
                self.ui.initial_score_comboBox.addItem(str(score))
        self.ui.initial_score_comboBox.lineEdit().editingFinished.connect(
            lambda: UIFunctions.update_game_settings(self)
        )

    def closeEvent(self, event):
        global STOP_DETECTION
        STOP_DETECTION = True
        if cap is not None and cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()
        event.accept()


class DefaultImageSetter(QRunnable):
    def __init__(self):
        super().__init__()
        self.signals = WorkerSignals()

    def run(self):
        global default_img, markerCorners, markerIds
        found_markers = False
        try:
            while True:
                success, img = cap.read()
                if success:
                    if USE_CAMERA_CALIBRATION_TO_UNDISTORT:
                        img_undist = utils.undistortFunction(img, meanMTX, meanDIST)
                    else:
                        img_undist = img
                    img_roi = ContourUtils.extract_roi_from_4_aruco_markers(img_undist, target_ROI_size, use_outer_corners=False, draw=True)
                    if img_roi is not None and img_roi.shape[1] > 0 and img_roi.shape[0] > 0:
                        img_roi = cv2.resize(img_roi, resize_for_squish)
                        default_img = img_roi
                        print("Set default image")
                        cv2.imshow("Default", default_img)
                        cv2.waitKey(1)
                        found_markers = True
                    if found_markers:
                        cv2.putText(img_undist, "Found markers press/hold x to save", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        if cv2.waitKey(1) & 0xff == ord('x'):
                            cv2.destroyWindow("Preview")
                            cv2.destroyWindow("Default")
                            break
                    else:
                        cv2.putText(img_undist, "No markers found", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.imshow("Preview", img_undist)
        finally:
            self.signals.finished.emit()


def get_biggest_contour(contours):
    """
    Get the biggest contour from a list of contours
    :param contours:
    :return:
    """
    contour = None
    for contour in contours:
        # if there are still multiple contours, take the one with the biggest area
        if len(contours) > 1:
            contour = max(contours, key=cv2.contourArea)
        # get the points of the contour
    return contour


class DetectionAndScoring(QRunnable):
    def __init__(self):
        global points, dart_tip, TRIANGLE_DETECT_THRESH, \
            score1, score2, scored_values, scored_mults, mults_of_round, values_of_round, img_undist, default_img, OPENCV_GUI_CREATED
        super().__init__()
        if not OPENCV_GUI_CREATED:  # create the gui only once
            opencv_gui_sliders.create_gui()
            OPENCV_GUI_CREATED = True
        default_img = utils.reset_default_image(img_undist, target_ROI_size, resize_for_squish)
        cv2.destroyWindow("Object measurement")
        self.signals = WorkerSignals()

    def run(self):
        global previous_img, difference, default_img, ACTIVE_PLAYER, UNDO_LAST_FLAG
        global points, dart_tip, TRIANGLE_DETECT_THRESH, score1, score2, scored_values, scored_mults, mults_of_round, values_of_round
        global new_dart_tip, update_dart_point, minArea, DARTBOARD_AREA
        try:
            while True:
                if STOP_DETECTION:
                    break
                fpsReader = FPS()
                success, img = cap.read()
                if success:
                    if USE_CAMERA_CALIBRATION_TO_UNDISTORT:
                        img_undist = utils.undistortFunction(img, meanMTX, meanDIST)
                    else:
                        img_undist = img
                    img_roi = ContourUtils.extract_roi_from_4_aruco_markers(img_undist, target_ROI_size, use_outer_corners=False, hold_position=True)
                    if img_roi is not None and img_roi.shape[1] > 0 and img_roi.shape[0] > 0:
                        img_roi = cv2.resize(img_roi, resize_for_squish)
                        img_show = cv2.resize(img_roi, dsize=(400, 400))
                        cv2.imshow("Live", img_show)

                        detect_dart_circle_and_set_limits(img_roi=img_roi)
                        if center_ellipse == (0, 0):
                            print("No dartboard detected!")
                            continue

                        if default_img is None or np.all(default_img == 0):
                            default_img = img_roi.copy()

                        difference = cv2.absdiff(img_roi, default_img)
                        gray, thresh = self.prepare_differnce_image(TRIANGLE_DETECT_THRESH, difference)

                        minimal_darts_area = 0.005 * DARTBOARD_AREA
                        maximal_darts_area = 0.1 * DARTBOARD_AREA
                        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                        noise_contours = [i for i in contours if cv2.contourArea(i) < minimal_darts_area]
                        darts_contours = [i for i in contours if minimal_darts_area < cv2.contourArea(i) < maximal_darts_area]
                        if len(noise_contours) > 10 and len(darts_contours) == 0:
                            print("Too much noise")
                            default_img = utils.reset_default_image(img_undist, target_ROI_size, resize_for_squish)
                            continue
                        for contour in darts_contours:
                            points_list = contour.reshape(contour.shape[0], contour.shape[2])
                            triangle = cv2.minEnclosingTriangle(cv2.UMat(points_list.astype(np.float32)))
                            triangle_np_array = cv2.UMat.get(triangle[1])
                            if triangle_np_array is not None:
                                pt1, pt2, pt3 = triangle_np_array.astype(np.int32)
                            else:
                                pt1, pt2, pt3 = np.array([-1, -1]), np.array([-1, -1]), np.array([-1, -1])

                            dart_tip, rest_pts = dart_scorer_util.find_tip_of_dart(pt1, pt2, pt3)
                            cv2.circle(img_roi, dart_tip, 4, (0, 0, 255), -1)
                            self.draw_detected_darts(dart_tip, pt1, pt2, pt3, thresh)

                            bottom_point = dart_scorer_util.get_bottom_point(rest_pts[0], rest_pts[1])
                            cv2.line(img_roi, dart_tip, bottom_point, (0, 0, 255), 2)
                            cv2.line(thresh, dart_tip, bottom_point, (255, 0, 255), 2)

                            k = -0.215
                            vect = dart_tip - bottom_point
                            new_dart_tip = dart_tip + k * vect

                            cv2.circle(img_roi, new_dart_tip.astype(np.int32), 4, (0, 255, 0), -1)
                            new_radius, new_angle = dart_scorer_util.get_radius_and_angle(center_ellipse[0], center_ellipse[1], new_dart_tip[0], new_dart_tip[1])
                            new_val, new_mult = dart_scorer_util.evaluate_throw(new_radius, new_angle)

                            if len(scored_values) <= 20:
                                scored_values.append(new_val)
                                scored_mults.append(new_mult)
                            else:
                                update_dart_point = True
                                final_val = mode(scored_values)
                                final_mult = mode(scored_mults)
                                values_of_round.append(final_val)
                                mults_of_round.append(final_mult)
                                default_img = utils.reset_default_image(img_undist, target_ROI_size, resize_for_squish)
                                if len(values_of_round) == 3:
                                    self.reset_default_image_after_player()
                                    self.enter_score_of_one_player(score1, score2)
                                scored_values = []
                                scored_mults = []

                        cv2.imshow("Threshold", thresh)
                        previous_img = img_roi
                        cv2.circle(img_roi, center_ellipse, int(a * (radius_1 / 100)), (255, 0, 255), 1)
                        cv2.circle(img_roi, center_ellipse, int(a * (radius_2 / 100)), (255, 0, 255), 1)
                        cv2.circle(img_roi, center_ellipse, int(a * (radius_3 / 100)), (255, 0, 255), 1)
                        cv2.circle(img_roi, center_ellipse, int(a * (radius_4 / 100)), (255, 0, 255), 1)
                        cv2.circle(img_roi, center_ellipse, int(a * (radius_5 / 100)), (255, 0, 255), 1)
                        cv2.circle(img_roi, center_ellipse, int(a * (radius_6 / 100)), (255, 0, 255), 1)
                        cv2.ellipse(img_roi, ellipse, (0, 255, 0), 2)

                        fps, img_roi = fpsReader.update(img_roi)
                        cv2.imshow("Dart Settings", utils.rez(img_roi, 1.5))
                    else:
                        print("NO MARKERS FOUND!")
                        cv2.putText(img_undist, "NO MARKERS FOUND", (300, 300), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 0, 255), 3)
                        cv2.imshow("Dart Settings", img_undist)
                        sleep(0.1)

                    cv2.waitKey(1)
                    if cv2.waitKey(1) & 0xff == ord('q'):
                        break
                # UIFunctions.update_labels(window)
        finally:
            self.signals.finished.emit()

    def reset_default_image_after_player(self):
        """
        Resets the default image after a player has thrown 3 darts and triggers the corresponding gui functions
        :return:
        """
        global default_img, STOP_DETECTION
        STOP_DETECTION = True
        success, img = cap.read()  # Reset the default image after every dart
        if success:
            if USE_CAMERA_CALIBRATION_TO_UNDISTORT:
                img = utils.undistortFunction(img, meanMTX, meanDIST)
            else:
                img = img
        default_img = utils.reset_default_image(img, target_ROI_size, resize_for_squish)

    def enter_score_of_one_player(self, score1, score2):
        """
        Enters the score of one player into the dart scorer util
        :param score1:
        :param score2:
        :return:
        """
        global default_img, ACTIVE_PLAYER, values_of_round, mults_of_round
        if ACTIVE_PLAYER == 1:
            dart_scorer_util.update_score(score1, values_of_round=values_of_round, mults_of_round=mults_of_round)
            ACTIVE_PLAYER = 2
        elif ACTIVE_PLAYER == 2:
            dart_scorer_util.update_score(score2, values_of_round=values_of_round, mults_of_round=mults_of_round)
            ACTIVE_PLAYER = 1
        values_of_round = []
        mults_of_round = []

    def draw_detected_darts(self, dart_point, pt1, pt2, pt3, thresh):
        """
        Draw the detected darts on the threshold image as triangle and a dot indicating the dart tip
        :param dart_point: The point of the dart tip
        :param pt1: Triangle point 1
        :param pt2: Triangle point 2
        :param pt3: Triangle point 3
        :param thresh: the threshold image
        :return:
        """
        # Display the Dart point
        cv2.circle(thresh, dart_point, 4, (0, 0, 255), -1)
        # Display the triangles
        cv2.line(thresh, pt1.ravel(), pt2.ravel(), (255, 0, 255), 2)
        cv2.line(thresh, pt2.ravel(), pt3.ravel(), (255, 0, 255), 2)
        cv2.line(thresh, pt3.ravel(), pt1.ravel(), (255, 0, 255), 2)

    def prepare_differnce_image(self, TRIANGLE_DETECT_THRESH, difference):
        """
        Prepare the difference image for triangle detection, by applying a bilateral filter, gaussian blur and thresholding
        :param TRIANGLE_DETECT_THRESH:
        :param difference:
        :return:
        """
        blur = cv2.GaussianBlur(difference, (5, 5), 0)
        for i in range(10):
            blur = cv2.GaussianBlur(blur, (9, 9), 1)
        blur = cv2.bilateralFilter(blur, 9, 75, 75)
        ret, thresh = cv2.threshold(blur, TRIANGLE_DETECT_THRESH, 255, 0)
        gray = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
        return gray, thresh


class UIFunctions(QMainWindow):

    def update_detection_sensitivity(self):
        global TRIANGLE_DETECT_THRESH
        TRIANGLE_DETECT_THRESH = self.ui.detection_sensitivity_slider.value()
        self.ui.current_detection_sensitivity_lable.setText(f"{TRIANGLE_DETECT_THRESH}")

    def undo_last_throw(self):
        global values_of_round, mults_of_round
        self.ui.press_enter_label.setText("    Please throw again")
        removed_throw = DartScore.undo_last_throw(values_of_round, mults_of_round)
        if removed_throw is not None:
            val, mult = removed_throw
            if ACTIVE_PLAYER == 1:
                self.ui.player1_sum_round.setText(str(int(self.ui.player1_sum_round.text()) - val * mult))
                player_labels = (self.ui.player1_1, self.ui.player1_2, self.ui.player1_3)
            elif ACTIVE_PLAYER == 2:
                self.ui.player2_sum_round.setText(str(int(self.ui.player2_sum_round.text()) - val * mult))
                player_labels = (self.ui.player2_1, self.ui.player2_2, self.ui.player2_3)
            player_labels[len(values_of_round)].setText("")
            if self.DartPositions:
                _, dart_position = self.DartPositions.popitem()
                dart_position.deleteLater()

    def delete_all_x_on_board(self):
        print("LEN:", len(self.DartPositions.values()))
        for lable in self.DartPositions.values():
            lable.setText("")
        self.DartPositions = {}

    def place_x_on_board(self, pos_x, pos_y):
        global dart_id
        DartPositionId = dart_id
        dart_id = dart_id + 1
        self.DartPositions[DartPositionId] = DartPositionLabel(self.ui.dart_board_image)
        print(f"Placing: {str(int(pos_x*Scaling_factor_for_x_placing_in_gui[0])), str(int(pos_y*Scaling_factor_for_x_placing_in_gui[1]))}")
        self.DartPositions[DartPositionId].addDartPosition(int(pos_x * Scaling_factor_for_x_placing_in_gui[0]),
                                                           int(pos_y * Scaling_factor_for_x_placing_in_gui[1]))

    def set_default_image(self):
        if self.default_image_worker is not None or self.detection_worker is not None:
            return
        pool = QThreadPool.globalInstance()
        default_img_setter = DefaultImageSetter()
        default_img_setter.signals.finished.connect(
            lambda: UIFunctions.default_image_capture_finished(self)
        )
        self.default_image_worker = default_img_setter
        self.ui.set_default_img_button.setEnabled(False)
        self.ui.start_measuring_button.setEnabled(False)
        pool.start(default_img_setter)

    def start_detection_and_scoring(self):
        global STOP_DETECTION, default_img, img_undist
        if self.detection_worker is not None or self.default_image_worker is not None:
            return
        if cap is None or not cap.isOpened():
            QMessageBox.warning(self, "Camera unavailable", "The selected camera could not be opened.")
            return
        STOP_DETECTION = False
        # change color of stop_measuring_button to transparent
        self.ui.stop_measuring_button.setStyleSheet("background-color: transparent")
        # change color of start_measuring_button to green
        self.ui.start_measuring_button.setStyleSheet("background-color: green")
        window.ui.press_enter_label.setText("")
        pool = QThreadPool.globalInstance()
        default_img = utils.reset_default_image(img_undist, target_ROI_size, resize_for_squish)
        detection_and_scoring = DetectionAndScoring()
        detection_and_scoring.signals.finished.connect(
            lambda: UIFunctions.detection_finished(self)
        )
        self.detection_worker = detection_and_scoring
        UIFunctions.delete_all_x_on_board(window)  ################################
        self.ui.start_measuring_button.setEnabled(False)
        self.ui.set_default_img_button.setEnabled(False)
        pool.start(detection_and_scoring)

    def stop_detection_and_scoring(self):
        global STOP_DETECTION
        # change color of stop_measuring_button to red
        self.ui.stop_measuring_button.setStyleSheet("background-color: red")
        self.ui.start_measuring_button.setStyleSheet("background-color: transparent")
        window.ui.press_enter_label.setText("    1. Remove all Darts\n    2. Press Continue to start next round")
        STOP_DETECTION = True

    def default_image_capture_finished(self):
        self.default_image_worker = None
        self.ui.set_default_img_button.setEnabled(True)
        self.ui.start_measuring_button.setEnabled(True)

    def detection_finished(self):
        self.detection_worker = None
        self.ui.stop_measuring_button.setStyleSheet("background-color: red")
        self.ui.start_measuring_button.setStyleSheet("background-color: transparent")
        self.ui.press_enter_label.setText("    1. Remove all Darts\n    2. Press Continue to start next round")
        self.ui.start_measuring_button.setEnabled(True)
        self.ui.set_default_img_button.setEnabled(True)

    def update_game_settings(self):
        global ACTIVE_PLAYER, values_of_round, mults_of_round
        try:
            score = int(self.ui.initial_score_comboBox.currentText())
        except ValueError:
            return
        if score < 101 or score % 100 != 1:
            QMessageBox.warning(self, "Invalid 01 score", "Enter a positive score ending in 01, such as 501 or 1001.")
            return
        score1.setNominalScore(score)
        score2.setNominalScore(score)
        ACTIVE_PLAYER = 1
        values_of_round = []
        mults_of_round = []
        UIFunctions.delete_all_x_on_board(self)

    def update_labels(self):
        global values_of_round, mults_of_round, ACTIVE_PLAYER, new_dart_tip, update_dart_point
        if update_dart_point and new_dart_tip is not None:
            print(f"Updating dart point in image {new_dart_tip[0], new_dart_tip[1]}")
            X_OFFSET = 8
            Y_OFFSET = 17
            UIFunctions.place_x_on_board(self, new_dart_tip[0] -X_OFFSET, new_dart_tip[1]- Y_OFFSET)

            update_dart_point = False
        if ACTIVE_PLAYER == 1:
            self.ui.player_frame.setStyleSheet("background-color: #3a3a3a;")
            self.ui.player_frame_2.setStyleSheet("background-color: rgb(35, 35, 35);")
            # self.ui.player1_sum_round.setText(str(sum(values_of_round)))
            if len(values_of_round) == 1:
                self.ui.player1_1.setText(f"{values_of_round[0] * mults_of_round[0]}")
                self.ui.player1_sum_round.setText(str(values_of_round[0] * mults_of_round[0]))
            elif len(values_of_round) == 2:
                self.ui.player1_1.setText(f"{values_of_round[0] * mults_of_round[0]}")
                self.ui.player1_2.setText(f"{values_of_round[1] * mults_of_round[1]}")
                self.ui.player1_sum_round.setText(str(values_of_round[0] * mults_of_round[0] + values_of_round[1] * mults_of_round[1]))
            elif len(values_of_round) == 3:
                self.ui.player1_1.setText(f"{values_of_round[0] * mults_of_round[0]}")
                self.ui.player1_2.setText(f"{values_of_round[1] * mults_of_round[1]}")
                self.ui.player1_3.setText(f"{values_of_round[2] * mults_of_round[2]}")
                self.ui.player1_sum_round.setText(str(values_of_round[0] * mults_of_round[0] + values_of_round[1] * mults_of_round[1] + values_of_round[2] * mults_of_round[2]))
            else:
                self.ui.player1_1.setText("-")
                self.ui.player1_2.setText("-")
                self.ui.player1_3.setText("-")
                self.ui.player1_sum_round.setText("")
        elif ACTIVE_PLAYER == 2:
            self.ui.player_frame_2.setStyleSheet("background-color: #3a3a3a;")
            self.ui.player_frame.setStyleSheet("background-color: rgb(35, 35, 35);")
            self.ui.player2_sum_round.setText(str(sum(values_of_round)))
            if len(values_of_round) == 1:
                self.ui.player2_1.setText(f"{values_of_round[0] * mults_of_round[0]}")
                self.ui.player2_sum_round.setText(str(values_of_round[0] * mults_of_round[0]))
            elif len(values_of_round) == 2:
                self.ui.player2_1.setText(f"{values_of_round[0] * mults_of_round[0]}")
                self.ui.player2_2.setText(f"{values_of_round[1] * mults_of_round[1]}")
                self.ui.player2_sum_round.setText(str(values_of_round[0] * mults_of_round[0] + values_of_round[1] * mults_of_round[1]))
            elif len(values_of_round) == 3:
                self.ui.player2_1.setText(f"{values_of_round[0] * mults_of_round[0]}")
                self.ui.player2_2.setText(f"{values_of_round[1] * mults_of_round[1]}")
                self.ui.player2_3.setText(f"{values_of_round[2] * mults_of_round[2]}")
                self.ui.player2_sum_round.setText(str(values_of_round[0] * mults_of_round[0] + values_of_round[1] * mults_of_round[1] + values_of_round[2] * mults_of_round[2]))
            else:
                self.ui.player2_1.setText("-")
                self.ui.player2_2.setText("-")
                self.ui.player2_3.setText("-")
                self.ui.player2_sum_round.setText("")

        # if one of the players has won the game, show the winner
        if score1.currentScore == 0:
            window.warning("Player 1 has won the game!")
        elif score2.currentScore == 0:
            window.warning("Player 2 has won the game!")

        self.ui.player1_overall.setText(str(score1.currentScore))
        self.ui.player2_overall.setText(str(score2.currentScore))


if __name__ == "__main__":
    cap = open_camera(CAMERA_NUMBER)
    app = QApplication(sys.argv)
    window = MainWindow()
    label_update_timer = QtCore.QTimer()
    label_update_timer.timeout.connect(lambda: UIFunctions.update_labels(window))
    label_update_timer.start(10)  # every 10 milliseconds

    sys.exit(app.exec())
