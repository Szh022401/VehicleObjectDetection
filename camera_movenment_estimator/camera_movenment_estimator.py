import pickle

import cv2
import numpy as np
import sys
import os

sys.path.append("../")
from utils import *


class CameraMovementEstimator():
    def __init__(self, frame, minimum_distance=1, max_corners=300, quality_level=0.5, min_distance=3, block_size=7):
        self.minimum_distance = minimum_distance
        self.lk_params = dict(winSize=(21, 21),
                              maxLevel=5,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        first_frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask_features = np.zeros_like(first_frame_grayscale)
        mask_features[:, 0:20] = 1
        mask_features[:, 500:1400] = 1
        self.features = dict(
            maxCorners=max_corners,
            qualityLevel=quality_level,
            minDistance=min_distance,
            blockSize=block_size,
            mask=mask_features
        )

    def add_adjust_position_to_tracks(self, tracks, camera_movement_per_frame):
        for objects, object_tracks in tracks.items():
            for frame_num, track_infos in enumerate(object_tracks):
                print(f"Frame {frame_num}, Object: {objects}, Track Info: {track_infos}")
                for track_id, track_info in track_infos.items():
                    if 'position' in track_info:
                        position = track_info['position']
                        camera_movement = camera_movement_per_frame[frame_num]
                        position_adjusted = (position[0] - camera_movement[0], position[1] - camera_movement[1])
                        tracks[objects][frame_num][track_id]['position_adjusted'] = position_adjusted
                        print(tracks[objects][frame_num][track_id]['position_adjusted'])
                    else:
                        print(f"Position key not found in track_info: {track_info}")
                        continue

    def get_camera_movement(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)
        camera_movement = [[0, 0]] * len(frames)

        old_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        old_features = cv2.goodFeaturesToTrack(old_gray, **self.features)

        for frame_num in range(1, len(frames), 3):
            frame_gray = cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2GRAY)
            new_features, _, _ = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, old_features, None, **self.lk_params)

            max_distance = 0
            camera_movement_x, camera_movement_y = 0, 0

            for i, (new, old) in enumerate(zip(new_features, old_features)):
                new_features_point = new.ravel()
                old_features_point = old.ravel()

                distance = measure_distance(new_features_point, old_features_point)

                if distance > max_distance:
                    max_distance = distance
                    camera_movement_x, camera_movement_y = measure_xy_distance(old_features_point, new_features_point)

            if max_distance > self.minimum_distance:
                camera_movement[frame_num] = [camera_movement_x, camera_movement_y]
            if len(new_features) < self.features['maxCorners'] * 0.5:
                old_features = cv2.goodFeaturesToTrack(frame_gray, **self.features)
            else:
                old_features = new_features

            old_gray = frame_gray.copy()

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(camera_movement, f)

        return camera_movement

    def draw_camera_movement(self, frames, camera_movement_per_frame):
        output_frame = []

        for frame_num, frame in enumerate(frames):
            frame = frame.copy()

            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (500, 100), (255, 255, 255), -1)
            alpha = 0.6
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            x_movement, y_movement = camera_movement_per_frame[frame_num]
            frame = cv2.putText(frame, f"Camera Movement X: {x_movement:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 0, 0), 3)
            frame = cv2.putText(frame, f"Camera Movement Y: {y_movement:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 0, 0), 3)
            output_frame.append(frame)

        return output_frame
