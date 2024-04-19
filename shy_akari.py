#!/usr/bin/env python3

import threading
import time
from queue import Queue
from time import sleep
from typing import Any

import cv2
import depthai as dai
import numpy as np
from akari_client import AkariClient

import blobconverter

pan_target_angle = 0.0
tilt_target_angle = 0.0
eyes_meet_check = False

# resize input to smaller size for faster inference
NN_WIDTH, NN_HEIGHT = 160, 120
VIDEO_WIDTH, VIDEO_HEIGHT = 300, 300

debug = True    # とりあえず
camera = True   # とりあえず

openvino_version = '2021.4'
detections = None

def frame_norm(frame, bbox):
    norm_vals = np.full(len(bbox), frame.shape[0])
    norm_vals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * norm_vals).astype(int)


def to_planar(arr: np.ndarray, shape: tuple) -> list:
    return [val for channel in cv2.resize(arr, shape).transpose(2, 0, 1) for y_col in channel for val in y_col]


def to_tensor_result(packet):
    return {
        tensor.name: np.array(packet.getLayerFp16(tensor.name)).reshape(tensor.dims)
        for tensor in packet.getRaw().tensors
    }


def padded_point(point, padding, frame_shape=None):
    if frame_shape is None:
        return [
            point[0] - padding,
            point[1] - padding,
            point[0] + padding,
            point[1] + padding
        ]
    else:
        def norm(val, dim):
            return max(0, min(val, dim))
        if np.any(point - padding > frame_shape[:2]) or np.any(point + padding < 0):
            print(f"Unable to create padded box for point {point} with padding {padding} and frame shape {frame_shape[:2]}")
            return None

        return [
            norm(point[0] - padding, frame_shape[0]),
            norm(point[1] - padding, frame_shape[1]),
            norm(point[0] + padding, frame_shape[0]),
            norm(point[1] + padding, frame_shape[1])
        ]


def create_pipeline():
    print("Creating pipeline...")
    pipeline = dai.Pipeline()
    pipeline.setOpenVINOVersion(version=dai.OpenVINO.Version.VERSION_2021_4)

    if camera:
        print("Creating Color Camera...")
        cam = pipeline.create(dai.node.ColorCamera)
        cam.setPreviewSize(300, 300)
        cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam.setInterleaved(False)
        cam.setBoardSocket(dai.CameraBoardSocket.RGB)

        cam_xout = pipeline.create(dai.node.XLinkOut)
        cam_xout.setStreamName("cam_out")
        cam.preview.link(cam_xout.input)

    # NeuralNetwork
    print("Creating Face Detection Neural Network...")
    face_nn = pipeline.create(dai.node.NeuralNetwork)
    face_nn.setBlobPath(blobconverter.from_zoo(name="face-detection-retail-0004", version=openvino_version, shaves=4))

    if camera:
        cam.preview.link(face_nn.input)
    else:
        face_in = pipeline.create(dai.node.XLinkIn)
        face_in.setStreamName("face_in")
        face_in.out.link(face_nn.input)

    face_nn_xout = pipeline.create(dai.node.XLinkOut)
    face_nn_xout.setStreamName("face_nn")
    face_nn.out.link(face_nn_xout.input)

    # NeuralNetwork
    print("Creating Landmarks Detection Neural Network...")
    land_nn = pipeline.create(dai.node.NeuralNetwork)
    land_nn.setBlobPath(blobconverter.from_zoo(name="landmarks-regression-retail-0009", version=openvino_version, shaves=4))
    land_nn_xin = pipeline.create(dai.node.XLinkIn)
    land_nn_xin.setStreamName("landmark_in")
    land_nn_xin.out.link(land_nn.input)
    land_nn_xout = pipeline.create(dai.node.XLinkOut)
    land_nn_xout.setStreamName("landmark_nn")
    land_nn.out.link(land_nn_xout.input)

    # NeuralNetwork
    print("Creating Head Pose Neural Network...")
    pose_nn = pipeline.create(dai.node.NeuralNetwork)
    pose_nn.setBlobPath(blobconverter.from_zoo(name="head-pose-estimation-adas-0001", version=openvino_version, shaves=4))
    pose_nn_xin = pipeline.create(dai.node.XLinkIn)
    pose_nn_xin.setStreamName("pose_in")
    pose_nn_xin.out.link(pose_nn.input)
    pose_nn_xout = pipeline.create(dai.node.XLinkOut)
    pose_nn_xout.setStreamName("pose_nn")
    pose_nn.out.link(pose_nn_xout.input)

    # NeuralNetwork
    print("Creating Gaze Estimation Neural Network...")
    gaze_nn = pipeline.create(dai.node.NeuralNetwork)
    path = blobconverter.from_zoo(
        "gaze-estimation-adas-0002",
        shaves=4,
        version=openvino_version,
        compile_params=['-iop head_pose_angles:FP16,right_eye_image:U8,left_eye_image:U8'],
    )
    gaze_nn.setBlobPath(path)
    gaze_nn_xin = pipeline.create(dai.node.XLinkIn)
    gaze_nn_xin.setStreamName("gaze_in")
    gaze_nn_xin.out.link(gaze_nn.input)
    gaze_nn_xout = pipeline.create(dai.node.XLinkOut)
    gaze_nn_xout.setStreamName("gaze_nn")
    gaze_nn.out.link(gaze_nn_xout.input)

    return pipeline


# 顔追従するクラス
class FaceTracker:
    """face tracking class"""

    def __init__(self) -> None:
        global pan_target_angle
        global tilt_target_angle

        # AkariClientのインスタンスを作成する。
        self.akari = AkariClient()
        # 関節制御用のインスタンスを取得する。
        self.joints = self.akari.joints

        self._default_x = 0
        self._default_y = 0
        # サーボトルクON
        self.joints.enable_all_servo()
        # モータ速度設定
        self.joints.set_joint_velocities(pan=10, tilt=10)
        # モータ加速度設定
        self.joints.set_joint_accelerations(pan=30, tilt=30)

        # Initialize motor position
        self.joints.move_joint_positions(sync=True, pan=0, tilt=0.26)
        self.currentMotorAngle = self.joints.get_joint_positions()

        # Dynamixel Input Value
        pan_target_angle = self.currentMotorAngle["pan"]
        tilt_target_angle = self.currentMotorAngle["tilt"]

    def _tracker(self) -> None:
        global pan_target_angle
        global tilt_target_angle
        global eyes_meet_check
        while True:
            self.joints.set_joint_velocities(pan=10, tilt=10)

            # 目があっていたときの処理
            if eyes_meet_check:
                self.joints.set_joint_velocities(pan=15, tilt=15)
                self.currentMotorAngle = self.joints.get_joint_positions()
                self.joints.move_joint_positions(
                    sync=True,
                    pan=0.6 * -(np.sign(pan_target_angle)),
                    tilt=-tilt_target_angle
                )
                current_time = time.time()
                while True:
                    if (time.time() - current_time) > 2:
                        break

                # ゆっくりもとの座標に戻る
                self.joints.set_joint_velocities(pan=2, tilt=1)
                self.joints.move_joint_positions(
                    sync=True,
                    pan=self.currentMotorAngle["pan"],
                    tilt=self.currentMotorAngle["tilt"]
                )

                # 復帰中に顔を検出した場合に備えて、直前で変数上書き
                pan_target_angle = self.currentMotorAngle['pan']
                tilt_target_angle = self.currentMotorAngle['tilt']
                eyes_meet_check = False

            self.joints.move_joint_positions(
                pan=pan_target_angle, tilt=tilt_target_angle
            )
            sleep(0.01)


class DirectionUpdater:
    """Update direction from face info"""

    _H_PIX_WIDTH = VIDEO_WIDTH
    _H_PIX_HEIGHT = VIDEO_HEIGHT
    _PAN_THRESHOLD = 0.1
    _TILT_THRESHOLD = 0.1
    _pan_dev = 0
    _tilt_dev = 0
    # モータゲインの最大幅。追従性の最大はここで変更
    _MAX_PAN_GAIN = 0.09
    _MAX_TILT_GAIN = 0.09
    # モータゲインの最小幅。追従性の最小はここで変更
    _MIN_PAN_GAIN = 0.07
    _MIN_TILT_GAIN = 0.07
    # 顔の距離によってモータゲインを変化させる係数。上げると早い動きについていきやすいが、オーバーシュートしやすくなる。
    _GAIN_COEF_PAN = 0.0001
    _GAIN_COEF_TILT = 0.0001

    _pan_p_gain = _MIN_PAN_GAIN
    _tilt_p_gain = _MIN_TILT_GAIN

    _PAN_POS_MAX = 1.047
    _PAN_POS_MIN = -1.047
    _TILT_POS_MAX = 0.523
    _TILT_POS_MIN = -0.523

    def __init__(self) -> None:
        global prev_time
        global cur_time
        self._old_pan: float = 0
        self._old_tilt: float = 0
        self._old_face_x: float = 0
        self._old_face_y: float = 0
        self.meet_log = False

    def _calc_p_gain(self, face_width: int) -> None:
        self._pan_p_gain = self._GAIN_COEF_PAN * face_width
        if self._pan_p_gain > self._MAX_PAN_GAIN:
            self._pan_p_gain = self._MAX_PAN_GAIN
        elif self._pan_p_gain < self._MIN_PAN_GAIN:
            self._pan_p_gain = self._MIN_PAN_GAIN
        self._tilt_p_gain = self._GAIN_COEF_TILT * face_width
        if self._tilt_p_gain > self._MAX_TILT_GAIN:
            self._tilt_p_gain = self._MAX_TILT_GAIN
        elif self._tilt_p_gain < self._MIN_TILT_GAIN:
            self._tilt_p_gain = self._MIN_TILT_GAIN

    def _face_info_cb(self, q_detection: Any) -> None:
        global detections
        while True:
            try:
                detections, eyes_meet = q_detection.get(timeout=0.1)
                if detections is not None:
                    face_x, face_y, face_width, face_height = detections
                    self._set_goal_pos(
                        face_x + face_width / 2,
                        face_y + face_height / 2,
                        eyes_meet,
                    )
                    self._calc_p_gain(face_width)
                print("detection:{} meet:{}".format(detections, eyes_meet))
            except:
                detections = None
                pass

    def _set_goal_pos(self, face_x: float, face_y: float, eyes_meet: bool) -> None:
        global pan_target_angle
        global tilt_target_angle
        global eyes_meet_check

        if face_x >= 1000:
            face_x = 0
        if face_y >= 1000:
            face_y = 0

        pan_error = -(face_x + self._pan_dev - self._H_PIX_WIDTH / 2.0) / (
            self._H_PIX_WIDTH / 2.0
        )  # -1 ~ 1
        tilt_error = -(face_y + self._tilt_dev - self._H_PIX_HEIGHT / 2.0) / (
            self._H_PIX_HEIGHT / 2.0
        )  # -1 ~ 1

        if abs(pan_error) > self._PAN_THRESHOLD:
            pan_target_angle += self._pan_p_gain * pan_error
        if pan_target_angle < self._PAN_POS_MIN:
            pan_target_angle = self._PAN_POS_MIN
        elif pan_target_angle > self._PAN_POS_MAX:
            pan_target_angle = self._PAN_POS_MAX

        if abs(tilt_error) > self._TILT_THRESHOLD:
            tilt_target_angle += self._tilt_p_gain * tilt_error
        if tilt_target_angle < self._TILT_POS_MIN:
            tilt_target_angle = self._TILT_POS_MIN
        elif tilt_target_angle > self._TILT_POS_MAX:
            tilt_target_angle = self._TILT_POS_MAX

        eyes_meet_check = eyes_meet


class GazeRecognition:
    def __init__(self, device):
        self.device = device
        print("Starting pipeline...")
        self.device.startPipeline()
        if camera:
            self.cam_out = self.device.getOutputQueue("cam_out")
        else:
            self.face_in = self.device.getInputQueue("face_in")

        self.frame = None
        self.face_box_q = Queue()
        self.bboxes = []
        self.left_bbox = None
        self.right_bbox = None
        self.nose = None
        self.pose = None
        self.gaze = None

        self.running = True

    def face_thread(self):
        face_nn = self.device.getOutputQueue("face_nn")
        landmark_in = self.device.getInputQueue("landmark_in")
        pose_in = self.device.getInputQueue("pose_in")

        while self.running:
            if self.frame is None:
                continue
            try:
                bboxes = np.array(face_nn.get().getFirstLayerFp16())
            except RuntimeError:
                continue
            bboxes = bboxes.reshape((bboxes.size // 7, 7))
            self.bboxes = bboxes[bboxes[:, 2] > 0.7][:, 3:7]

            for raw_bbox in self.bboxes:
                bbox = frame_norm(self.frame, raw_bbox)
                det_frame = self.frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]

                land_data = dai.NNData()
                land_data.setLayer("0", to_planar(det_frame, (48, 48)))
                landmark_in.send(land_data)

                pose_data = dai.NNData()
                pose_data.setLayer("data", to_planar(det_frame, (60, 60)))
                pose_in.send(pose_data)

                self.face_box_q.put(bbox)

    def land_pose_thread(self):
        landmark_nn = self.device.getOutputQueue(name="landmark_nn", maxSize=1, blocking=False)
        pose_nn = self.device.getOutputQueue(name="pose_nn", maxSize=1, blocking=False)
        gaze_in = self.device.getInputQueue("gaze_in")

        while self.running:
            try:
                land_in = landmark_nn.get().getFirstLayerFp16()
            except RuntimeError:
                continue

            try:
                face_bbox = self.face_box_q.get(block=True, timeout=100)
            except Queue.Empty:
                continue

            self.face_box_q.task_done()
            left = face_bbox[0]
            top = face_bbox[1]
            face_frame = self.frame[face_bbox[1]:face_bbox[3], face_bbox[0]:face_bbox[2]]
            land_data = frame_norm(face_frame, land_in)
            land_data[::2] += left
            land_data[1::2] += top
            left_bbox = padded_point(land_data[:2], padding=30, frame_shape=self.frame.shape)
            if left_bbox is None:
                print("Point for left eye is corrupted, skipping nn result...")
                continue
            self.left_bbox = left_bbox
            right_bbox = padded_point(land_data[2:4], padding=30, frame_shape=self.frame.shape)
            if right_bbox is None:
                print("Point for right eye is corrupted, skipping nn result...")
                continue
            self.right_bbox = right_bbox
            self.nose = land_data[4:6]
            left_img = self.frame[self.left_bbox[1]:self.left_bbox[3], self.left_bbox[0]:self.left_bbox[2]]
            right_img = self.frame[self.right_bbox[1]:self.right_bbox[3], self.right_bbox[0]:self.right_bbox[2]]

            try:
                # The output of  pose_nn is in YPR  format, which is the required sequence input for pose in  gaze
                # https://docs.openvinotoolkit.org/2020.1/_models_intel_head_pose_estimation_adas_0001_description_head_pose_estimation_adas_0001.html
                # https://docs.openvinotoolkit.org/latest/omz_models_model_gaze_estimation_adas_0002.html
                # ... three head pose angles – (yaw, pitch, and roll) ...
                values = to_tensor_result(pose_nn.get())
                self.pose = [
                        values['angle_y_fc'][0][0],
                        values['angle_p_fc'][0][0],
                        values['angle_r_fc'][0][0]
                        ]
            except RuntimeError:
                continue

            gaze_data = dai.NNData()
            gaze_data.setLayer("left_eye_image", to_planar(left_img, (60, 60)))
            gaze_data.setLayer("right_eye_image", to_planar(right_img, (60, 60)))
            gaze_data.setLayer("head_pose_angles", self.pose)
            gaze_in.send(gaze_data)

    def gaze_thread(self):
        gaze_nn = self.device.getOutputQueue("gaze_nn")
        while self.running:
            try:
                self.gaze = np.array(gaze_nn.get().getFirstLayerFp16())
            except RuntimeError:
                continue

    def should_run(self):
        if self.running:
            return True if camera else self.cap.isOpened()
        else:
            return False

    def get_frame(self, retries=0):
        if camera:
            return True, np.array(self.cam_out.get().getData()).reshape((3, 300, 300)).transpose(1, 2, 0).astype(np.uint8)
        else:
            read_correctly, new_frame = self.cap.read()
            if not read_correctly or new_frame is None:
                if retries < 5:
                    return self.get_frame(retries+1)
                else:
                    print("Source closed, terminating...")
                    return False, None
            else:
                return read_correctly, new_frame

    def run(self, q_detection):
        meet_flag = False
        meet_count = 0
        face_reco = False
        self.threads = [
            threading.Thread(target=self.face_thread),
            threading.Thread(target=self.land_pose_thread),
            threading.Thread(target=self.gaze_thread)
        ]
        for thread in self.threads:
            thread.start()

        while self.should_run():
            try:
                read_correctly, new_frame = self.get_frame()
            except RuntimeError:
                continue

            if not read_correctly:
                break

            self.frame = new_frame
            self.debug_frame = self.frame.copy()

            # if debug:  # face
            if self.gaze is not None and self.left_bbox is not None and self.right_bbox is not None:
                re_x = (self.right_bbox[0] + self.right_bbox[2]) // 2
                re_y = (self.right_bbox[1] + self.right_bbox[3]) // 2
                le_x = (self.left_bbox[0] + self.left_bbox[2]) // 2
                le_y = (self.left_bbox[1] + self.left_bbox[3]) // 2

                x, y = (self.gaze * 100).astype(int)[:2]

                if self.bboxes is not None and detections is not None:
                    face_reco = True
                    cv2.arrowedLine(self.debug_frame, (le_x, le_y), (le_x + x, le_y - y), (255, 0, 255), 3)
                    cv2.arrowedLine(self.debug_frame, (re_x, re_y), (re_x + x, re_y - y), (255, 0, 255), 3)
                else:
                    face_reco = False

                # 目線の矢印の始点と終点の座標値の差がしきい値以下だった場合は目があっているとする
                if x < 2.0 and y < 2.0 and face_reco:
                    # print("Eyes meet!!")
                    meet_count = meet_count + 1
                    if meet_count > 3:
                        meet_flag = True
                        meet_count = 0

                else:
                    meet_flag = False
                    meet_count = 0

            for raw_bbox in self.bboxes:
                bbox = frame_norm(self.frame, raw_bbox)
                cv2.rectangle(self.debug_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (10, 245, 10), 2)
                q_detection.put((bbox, meet_flag))
            # if self.nose is not None:
            #     cv2.circle(self.debug_frame, (self.nose[0], self.nose[1]), 2, (0, 255, 0), thickness=5, lineType=8, shift=0)
            # if self.left_bbox is not None:
            #     cv2.rectangle(self.debug_frame, (self.left_bbox[0], self.left_bbox[1]), (self.left_bbox[2], self.left_bbox[3]), (245, 10, 10), 2)
            # if self.right_bbox is not None:
            #     cv2.rectangle(self.debug_frame, (self.right_bbox[0], self.right_bbox[1]), (self.right_bbox[2], self.right_bbox[3]), (245, 10, 10), 2)
            # if self.pose is not None and self.nose is not None:
            #     draw_3d_axis(self.debug_frame, self.pose, self.nose)

            resize_frame = cv2.resize(self.debug_frame, (720, 720))
            if camera:
                cv2.imshow("Camera view", resize_frame)
            # else:
            #     aspect_ratio = self.frame.shape[1] / self.frame.shape[0]
            #     cv2.imshow("Video view", cv2.resize(self.debug_frame, (int(900),  int(900 / aspect_ratio))))

            meet_flag = False

            if cv2.waitKey(1) == ord('q'):
                cv2.destroyAllWindows()
                break

        if not camera:
            self.cap.release()
        cv2.destroyAllWindows()
        for i in range(1, 5):  # https://stackoverflow.com/a/25794701/5494277
            cv2.waitKey(1)
        self.running = False


def Recognition_run(q_detection):
    with dai.Device(create_pipeline()) as device:
        app = GazeRecognition(device)
        app.run(q_detection)


def main() -> None:
    q_detection: Any = Queue()

    face_tracker = FaceTracker()
    direction_updater = DirectionUpdater()

    t1 = threading.Thread(target=Recognition_run, args=(q_detection,), daemon=True)
    t2 = threading.Thread(target=direction_updater._face_info_cb, args=(q_detection,), daemon=True)
    t3 = threading.Thread(target=face_tracker._tracker, daemon=True)
    t1.start()
    t2.start()
    t3.start()
    t1.join()
    t2.join()
    t3.join()


if __name__ == "__main__":
    main()
