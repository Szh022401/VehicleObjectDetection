
import cv2
import numpy as np
import os


def read_video(video_path):
    if not os.path.exists(video_path):
        print(f"Video file {video_path} does not exist.")
        return []

    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def get_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return frame_rate


def save_video(output_video_frames, output_video_path,frames_rate):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    height, width, _ = output_video_frames[0].shape
    out = cv2.VideoWriter(output_video_path, fourcc, frames_rate, (width, height))
    for frame in output_video_frames:
        out.write(frame)
    out.release()
