import json
import os

import cv2
import numpy as np

from pose_estimation.modules.draw import Plotter3d, draw_poses
from pose_estimation.modules.parse_poses import parse_poses

class PoseEstimation:
    def __init__(self, use_openvino, device, extrinsics_path, images, video, fx, height_size, mean_time=None):
        if use_openvino:
            from pose_estimation.modules.inference_engine_openvino import InferenceEngineOpenVINO
            self.net = InferenceEngineOpenVINO('trained_models/human-pose-estimation-3d.pth', device)
        else:
            from pose_estimation.modules.inference_engine_pytorch import InferenceEnginePyTorch
            self.net = InferenceEnginePyTorch('trained_models/human-pose-estimation-3d.pth', device)

        self.canvas_3d = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.plotter = Plotter3d(self.canvas_3d.shape[:2])
        self.canvas_3d_window_name = 'Canvas 3D'
        # cv2.namedWindow(self.canvas_3d_window_name)
        # cv2.setMouseCallback(self.canvas_3d_window_name, Plotter3d.mouse_callback)

        self.file_path = extrinsics_path
        if self.file_path is None:
            self.file_path = os.path.join('pose_estimation', 'data', 'extrinsics.json')
        with open(self.file_path, 'r') as f:
            self.extrinsics = json.load(f)

        self.R = np.array(self.extrinsics['R'], dtype=np.float32)
        self.t = np.array(self.extrinsics['t'], dtype=np.float32)

        self.is_video = False
        if video != '':
            self.is_video = True
        self.base_height = height_size
        self.fx = fx
        self.stride = 8
        if mean_time is None:
            self.mean_time = 0
        else:
            self.mean_time = mean_time

    def generatePoses(self, frame):
        input_scale = self.base_height / frame.shape[0]
        scaled_img = cv2.resize(frame, dsize=None, fx=input_scale, fy=input_scale)
        if self.fx < 0:  # Focal length is unknown
            self.fx = np.float32(0.8 * frame.shape[1])

        inference_result = self.net.infer(scaled_img)
        poses_3d, poses_2d = parse_poses(inference_result, input_scale, self.stride, self.fx, self.is_video)
        edges = []
        if len(poses_3d):
            poses_3d = PoseEstimation.rotate_poses(poses_3d, self.R, self.t)
            poses_3d_copy = poses_3d.copy()
            x = poses_3d_copy[:, 0::4]
            y = poses_3d_copy[:, 1::4]
            z = poses_3d_copy[:, 2::4]
            
            poses_3d[:, 0::4], poses_3d[:, 1::4], poses_3d[:, 2::4] = -z, x, -y

            poses_3d = poses_3d.reshape(poses_3d.shape[0], 19, -1)[:, :, 0:3]
            edges = (Plotter3d.SKELETON_EDGES + 19 * np.arange(poses_3d.shape[0]).reshape((-1, 1, 1))).reshape((-1, 2))
        # self.plotPoses3D(poses_3d, edges)

        return poses_3d, edges, poses_2d

    def plotPoses3D(self, poses_3d, edges):
        self.plotter.plot(self.canvas_3d, poses_3d, edges)
        cv2.imshow(self.canvas_3d_window_name, self.canvas_3d)

    def plotPoses2D(self, frame, poses_2d):
        draw_poses(frame, poses_2d)
        current_time = (cv2.getTickCount() - current_time) / cv2.getTickFrequency()
        if self.mean_time == 0:
            self.mean_time = current_time
        else:
            self.mean_time = self.mean_time * 0.95 + current_time * 0.05
        cv2.putText(frame, 'FPS: {}'.format(int(1 / self.mean_time * 10) / 10),
                    (40, 80), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))
        cv2.imshow('ICV 3D Human Pose Estimation', frame)

    @staticmethod
    def rotate_poses(poses_3d, R, t):
        R_inv = np.linalg.inv(R)
        for pose_id in range(len(poses_3d)):
            pose_3d = poses_3d[pose_id].reshape((-1, 4)).transpose()
            pose_3d[0:3, :] = np.dot(R_inv, pose_3d[0:3, :] - t)
            poses_3d[pose_id] = pose_3d.transpose().reshape(-1)

        return poses_3d