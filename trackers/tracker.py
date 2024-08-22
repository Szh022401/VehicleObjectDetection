import os
import numpy as np
import torch
from yolov7.models.experimental import attempt_load
from yolov7.utils.general import non_max_suppression, scale_coords
from yolov7.utils.datasets import letterbox
from yolov7.utils.torch_utils import select_device
from utils import get_center_of_box, get_box_width, get_car_position
import supervision as sv
import pickle
import cv2


class Tracker:
    def __init__(self, model_dir):
        device = select_device('')
        self.models = {
            'car': attempt_load(os.path.join(model_dir, 'car.pt'), map_location=device),
            'licence_plate': attempt_load(os.path.join(model_dir, 'LicencePlate.pt'), map_location=device),
            'people': attempt_load(os.path.join(model_dir, 'people.pt'), map_location=device)
        }
        self.device = device
        self.tracker = sv.ByteTrack()

    def draw_ellipses(self, frames, box, color, tracks_id):
        y2 = int(box[3])
        x_center, _ = get_center_of_box(box)
        width = get_box_width(box)
        cv2.ellipse(
            frames,
            center=(x_center, y2),
            axes=(int(width), int(0.35 * width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )
        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width // 2
        x2_rect = x_center + rectangle_width // 2
        y1_rect = (y2 - rectangle_height // 2) + 15
        y2_rect = (y2 + rectangle_height // 2) + 15
        if tracks_id is not None:
            cv2.rectangle(
                frames,
                (int(x1_rect), int(y1_rect)),
                (int(x2_rect), int(y2_rect)),
                color,
                cv2.FILLED
            )
            x1_text = x1_rect + 12
            if tracks_id > 99:
                x1_text -= 10

            cv2.putText(frames,
                        f"{tracks_id}",
                        (int(x1_text), int(y1_rect + 15)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 0),
                        2
                        )

        return frames

    def add_position_to_tracker(self, tracks):
        for objects, object_track in tracks.items():
            for frame_num, track in enumerate(object_track):
                for track_id, track_info in track.items():
                    box = track_info['box']
                    if objects == 'licence_plate':
                        position = get_center_of_box(box)
                    else:
                        position = get_car_position(box)
                    tracks[objects][frame_num][track_id]['position'] = position
                    print(tracks[objects][frame_num][track_id]['position'])

    def detect(self, frames):
        batch_size = 20
        img_size = 640
        stride = 32
        conf_thres_dict = {
            'car': 0.8,
            'licence_plate': 0.8,
            'people': 0.3
        }
        iou_thres = 0.5
        detections = {'car': [], 'licence_plate': [], 'people': []}

        for model_name, model in self.models.items():
            for i in range(0, len(frames), batch_size):
                batch_frames = frames[i:i + batch_size]
                imgs = [letterbox(frame, img_size, stride=stride)[0] for frame in batch_frames]
                imgs = [img[:, :, ::-1].transpose(2, 0, 1) for img in imgs]
                imgs = np.ascontiguousarray(imgs)
                imgs = torch.from_numpy(imgs).to(self.device)
                imgs = imgs.float()
                imgs /= 255.0

                if imgs.ndimension() == 3:
                    imgs = imgs.unsqueeze(0)

                with torch.no_grad():
                    pred = model(imgs, augment=False)[0]

                pred = non_max_suppression(pred, conf_thres=min(conf_thres_dict.values()), iou_thres=iou_thres,
                                           classes=None, agnostic=False)

                for j, det in enumerate(pred):
                    frame_detections = []
                    frame = batch_frames[j]
                    gn = torch.tensor(frame.shape)[[1, 0, 1, 0]]
                    if len(det):
                        det[:, :4] = scale_coords(imgs.shape[2:], det[:, :4], frame.shape).round()
                        for *xyxy, conf, cls in reversed(det):
                            cls_name = 'car' if cls == 0 else 'licence_plate' if cls == 1 else 'people'
                            conf_thres = conf_thres_dict[cls_name]
                            if conf >= conf_thres:
                                xyxy = [int(x.item()) for x in xyxy]
                                frame_detections.append((xyxy, conf.item(), cls.item()))
                                print(f"Model: {model_name}, Box: {xyxy}, Conf: {conf}, Class: {cls}")
                    detections[model_name].append(frame_detections)
                    print(f"Frame {i + j}: {model_name} detections: {frame_detections}")
        print(f"Final detections: {detections}")
        return detections

    def get_objects(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracker = pickle.load(f)
            return tracker

        detections = self.detect(frames)
        tracks = {
            "car": [],
            "licence_plate": [],
            "people": []
        }

        for frame_num, frame_detections in enumerate(detections['car']):
            tracks["car"].append({})
            for track_id, det in enumerate(frame_detections):
                box, conf, cls = det
                if isinstance(box, (np.ndarray, torch.Tensor)):
                    box = box.tolist()
                tracks["car"][frame_num][track_id] = {"box": box, "conf": conf, "class": cls}

        for frame_num, frame_detections in enumerate(detections['licence_plate']):
            tracks["licence_plate"].append({})
            for track_id, det in enumerate(frame_detections):
                box, conf, cls = det
                if isinstance(box, (np.ndarray, torch.Tensor)):
                    box = box.tolist()
                tracks["licence_plate"][frame_num][track_id] = {"box": box, "conf": conf, "class": cls}

        for frame_num, frame_detections in enumerate(detections['people']):
            tracks["people"].append({})
            for track_id, det in enumerate(frame_detections):
                box, conf, cls = det
                if isinstance(box, (np.ndarray, torch.Tensor)):
                    box = box.tolist()
                tracks["people"][frame_num][track_id] = {"box": box, "conf": conf, "class": cls}

        if stub_path is not None:
            with open(stub_path, "wb") as f:
                pickle.dump(tracks, f)
        return tracks

    def draw_annotations(self, video_frames, tracks):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            car_dict = tracks["car"][frame_num]
            people_dict = tracks["people"][frame_num]
            licence_dict = tracks["licence_plate"][frame_num]

            for track_id, car_info in car_dict.items():
                box = car_info['box']
                color = car_info.get('color', (0, 0, 255))
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)

            for track_id, people_info in people_dict.items():
                box = people_info['box']
                color = people_info.get('color', (0, 255, 255))
                frame = self.draw_ellipses(frame, box, color, track_id)

            for track_id, licence_info in licence_dict.items():
                box = licence_info['box']
                color = licence_info.get('color', (0, 0, 0))
                cv2.rectangle(frame, box, color, track_id)

            output_video_frames.append(frame)
            print(
                f'Frame {frame_num} processed with {len(car_dict)} cars, {len(people_dict)} people, and {len(licence_dict)} licences.')
        return output_video_frames


def xyxy2xywh(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2
    y[:, 2] = x[:, 2] - x[:, 0]
    y[:, 3] = x[:, 3] - x[:, 1]
    return y
