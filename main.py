import cv2
from utils.video_utils import read_video, save_video, get_video_frames
from trackers.tracker import Tracker
import os
import pickle
import sys
from speed_and_distance import SpeedAndDistance
from camera_movenment_estimator import CameraMovementEstimator
from transformer import Transformer

sys.path.append('yolov7')


def main():
    video_frames = read_video('input/test1.mp4')
    tracker = Tracker('models/')
    frames_rate = get_video_frames('input/test1.mp4')
    tracks = tracker.get_objects(video_frames, read_from_stub=False, stub_path='stubs/track_stubs.pkl')

    for frame_num, frame_tracks in enumerate(tracks["car"]):
        print(f'Frame {frame_num}: {len(frame_tracks)} cars detected')
    for frame_num, frame_tracks in enumerate(tracks["people"]):
        print(f'Frame {frame_num}: {len(frame_tracks)} people detected')
    for frame_num, frame_tracks in enumerate(tracks["licence_plate"]):
        print(f'Frame {frame_num}: {len(frame_tracks)} licences detected')

    tracker.add_position_to_tracker(tracks)
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,read_from_stub=True,stub_path='stubs/camera_movement_stub.pkl')
    camera_movement_estimator.add_adjust_position_to_tracks(tracks, camera_movement_per_frame)
    transformer = Transformer()
    transformer.add_transformed_position_to_tracks(tracks)
    speed_and_distance = SpeedAndDistance()
    speed_and_distance.add_speed_and_distance(tracks)

    output_video_frames = tracker.draw_annotations(video_frames, tracks)
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)

    speed_and_distance.draw_speed_and_distance(output_video_frames, tracks)
    save_video(output_video_frames, 'output/output_video.avi', frames_rate)


if __name__ == '__main__':
    main()
