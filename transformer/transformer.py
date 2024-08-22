
import numpy as np
import cv2


class Transformer(object):
    def __init__(self):
        street_width  = 50
        street_length  = 100

        self.pixel_vertices = np.array([
            [200, 1080],
            [200, 0],
            [1720, 0],
            [1720, 1080]
        ])

        self.target_vertices = np.array([
            [0, street_width],
            [0, 0],
            [street_length, 0],
            [street_length, street_width]
        ])

        self.pixel_vertices = self.pixel_vertices.astype(np.float32)
        self.target_vertices = self.target_vertices.astype(np.float32)
        self.perspective_transform = cv2.getPerspectiveTransform(self.pixel_vertices,self.target_vertices)

    def transform_points(self,points):
        p = (int(points[0]), int(points[1]))

        is_inside = cv2.pointPolygonTest(self.pixel_vertices, p, False) >= 0

        if not is_inside:
            return None

        reshape_points = points.reshape(-1,1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshape_points,self.perspective_transform)
        return transformed_points.reshape(-1,2)

    def add_transformed_position_to_tracks(self, tracks):
        for obj, object_track in tracks.items():
            for frame_num, track in enumerate(object_track):
                for track_id, track_info in track.items():
                    #print(track_id, track_info)
                    if 'position_adjusted' in track_info:
                        position = track_info['position_adjusted']
                        position = np.array(position)

                        position_transformed = self.transform_points(position)

                        if position_transformed is not None:
                            position_transformed = position_transformed.squeeze().tolist()
                        tracks[obj][frame_num][track_id]['position_transformed'] = position_transformed
                        print(f"Transformed Position: {position_transformed}")
                    else:
                        print(f"Key 'position_adjusted' not found in track_info: {track_info}")