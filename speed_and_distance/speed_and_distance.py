import cv2
import sys
import numpy as np
from collections import defaultdict
from scipy.spatial import distance
from filterpy.kalman import KalmanFilter
from utils import measure_distance, get_car_position


class SpeedAndDistance:
    def __init__(self):
        self.frame_window = 5
        self.frame_rate = 60

    def add_speed_and_distance(self, tracks):
        total_distance = {}
        for objects, object_tracks in tracks.items():
            if objects == "people" or objects == "licence_plate":
                continue
            number_of_frames = len(object_tracks)
            for frame_num in range(0, number_of_frames, self.frame_window):
                last_frame = min(frame_num + self.frame_window, number_of_frames - 1)

                for track_id, _ in object_tracks[frame_num].items():
                    if track_id not in object_tracks[last_frame]:
                        continue

                    start_position = object_tracks[frame_num][track_id]['position_transformed']
                    end_position = object_tracks[last_frame][track_id]['position_transformed']

                    if start_position is None or end_position is None:
                        continue

                    distance_covered = measure_distance(start_position, end_position)
                    time_elapsed = (last_frame - frame_num) / self.frame_rate
                    speed_meters_per_second = distance_covered / time_elapsed
                    speed_km_per_hour = speed_meters_per_second * 3.6
                    if objects not in total_distance:
                        total_distance[objects] = {}
                    if track_id not in total_distance[objects]:
                        total_distance[objects][track_id] = 0
                    total_distance[objects][track_id] += distance_covered

                    for frame_num_batch in range(frame_num, last_frame):
                        if track_id not in tracks[objects][frame_num_batch]:
                            continue
                        tracks[objects][frame_num_batch][track_id]['speed'] = speed_km_per_hour
                        tracks[objects][frame_num_batch][track_id]['distance'] = total_distance[objects][track_id]

    # def initialize_kalman_filter(self, track_id, initial_position):
    #     kf = KalmanFilter(dim_x=4, dim_z=2)
    #     kf.x = np.array([initial_position[0], initial_position[1], 0, 0])
    #     kf.F = np.array([[1, 0, 1, 0],
    #                      [0, 1, 0, 1],
    #                      [0, 0, 1, 0],
    #                      [0, 0, 0, 1]])
    #     kf.H = np.array([[1, 0, 0, 0],
    #                      [0, 1, 0, 0]])
    #     kf.P *= 1000
    #     kf.R = np.array([[5, 0],
    #                      [0, 5]])
    #     kf.Q = np.eye(4)
    #     self.kalman_filters[track_id] = kf

    # def smooth_tracks(self):
    #     for track_id, track_data in self.trackers.items():
    #         if len(track_data) < 2:
    #             for j in range(len(track_data)):
    #                 data = list(track_data[j])
    #                 if len(data) < 4:
    #                     data.append(0.0)
    #                 track_data[j] = tuple(data)
    #             continue
    #
    #         smoothed_speeds = []
    #         for i in range(1, len(track_data)):
    #             prev_data = track_data[i - 1]
    #             curr_data = track_data[i]
    #             distance_covered = distance.euclidean(prev_data[2], curr_data[1])
    #             time_elapsed = (curr_data[0] - prev_data[0]) / 60.0
    #             speed_meters_per_second = distance_covered / time_elapsed
    #             speed_km_per_hour = speed_meters_per_second * 3.6
    #             smoothed_speeds.append(speed_km_per_hour)
    #
    #         avg_speed = sum(smoothed_speeds) / len(smoothed_speeds)
    #         for j in range(len(track_data)):
    #             data = list(track_data[j])
    #             data[3] = avg_speed
    #             track_data[j] = tuple(data)
    #
    # def apply_kalman_filter(self, track_id, position):
    #     if track_id not in self.kalman_filters:
    #         self.initialize_kalman_filter(track_id, position)
    #     kf = self.kalman_filters[track_id]
    #     kf.predict()
    #     kf.update(np.array(position).reshape(2, 1))
    #     return kf.x[:2]

    def draw_speed_and_distance(self, frames, tracks):
        # self.smooth_tracks()
        output_frame = []
        for frame_num, frame in enumerate(frames):
            for objects, object_tracks in tracks.items():

                if objects == "people" or objects == "licence_plate":
                    continue
                for _, track_info in object_tracks[frame_num].items():
                    # print(track_info)
                    if "speed" in track_info:
                        speed = track_info.get('speed', None)
                        distance = track_info.get('distance', None)

                        if speed is None or distance is None:
                            continue

                        box = track_info['box']
                        position = get_car_position(box)
                        position = list(position)
                        position[1] += 40

                        position = tuple(map(int, position))

                        cv2.putText(
                            frame, f"{speed:.2f} km/h", position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                        cv2.putText(frame, f"{distance:.2f} m", (
                            position[0], position[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

                    output_frame.append(frame)

        return output_frame
