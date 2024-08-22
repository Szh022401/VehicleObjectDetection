import cv2
from utils.video_utils import read_video, save_video
from trackers.tracker import Tracker
import os
import pickle
import sys
sys.path.append('yolov7')

def main():
    video_frames = read_video('input/test.mp4')
    tracker = Tracker('models/')


    tracks = tracker.get_objects(video_frames, read_from_stub=False, stub_path='stubs/track_stubs.pkl')

    # 检查和打印检测结果
    for frame_num, frame_tracks in enumerate(tracks["car"]):
        print(f'Frame {frame_num}: {len(frame_tracks)} cars detected')
    for frame_num, frame_tracks in enumerate(tracks["people"]):
        print(f'Frame {frame_num}: {len(frame_tracks)} people detected')
    for frame_num, frame_tracks in enumerate(tracks["licence_plate"]):
        print(f'Frame {frame_num}: {len(frame_tracks)} licences detected')

    tracker.add_position_to_tracker(tracks)
    output_video_frames = tracker.draw_annotations(video_frames, tracks)
    save_video(output_video_frames, 'output/output_video.avi')

if __name__ == '__main__':
    main()