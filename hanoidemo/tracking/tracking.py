"""
tracking.py
-----------
Cài đặt SORT tối thiểu + Kalman cho bbox:
- iou_xyxy
- KalmanBoxTracker
- Sort
"""

import math
from collections import deque

import numpy as np
from filterpy.kalman import KalmanFilter

from hanoidemo.config import SORT_MAX_AGE, SORT_MIN_HITS, SORT_IOU_THRESH


def iou_xyxy(bb_test, bb_gt):
    """
    Tính IoU giữa hai bbox dạng [x1,y1,x2,y2].
    """
    xx1 = max(bb_test[0], bb_gt[0])
    yy1 = max(bb_test[1], bb_gt[1])
    xx2 = min(bb_test[2], bb_gt[2])
    yy2 = min(bb_test[3], bb_gt[3])

    w = max(0.0, xx2 - xx1)
    h = max(0.0, yy2 - yy1)
    inter = w * h

    area1 = (bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
    area2 = (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1])

    o = inter / (area1 + area2 - inter + 1e-6)
    return o


class KalmanBoxTracker:
    """
    Một đối tượng (track) dùng Kalman Filter cho bbox.
    State vector: [cx, cy, s, r, vx, vy, vs]
    """

    count = 0

    def __init__(self, bbox, fps=30.0, frame_height=1080):
        # bbox: [x1,y1,x2,y2]
        self.kf = KalmanFilter(dim_x=7, dim_z=4)

        dt = 1.0 / max(1.0, fps)

        self.kf.F = np.array(
            [
                [1, 0, 0, 0, dt, 0, 0],
                [0, 1, 0, 0, 0, dt, 0],
                [0, 0, 1, 0, 0, 0, dt],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1],
            ],
            dtype=float,
        )

        self.kf.H = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
            ],
            dtype=float,
        )

        self.kf.R *= 0.01
        self.kf.P *= 10.0

        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        s = (x2 - x1) * (y2 - y1)
        r = (x2 - x1) / max(1.0, (y2 - y1))

        self.kf.x[:4, 0] = np.array([cx, cy, s, r], dtype=float)

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1

        self.history = deque(maxlen=64)
        self.hits = 1
        self.age = 0

        self.conf = 0.0
        self.cls = -1

        self.frame_height = frame_height

        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

        # theo code gốc: cờ hỗ trợ tracking line crossing
        self.crossed_line = False
        self.line_cross_y_prev = None

    def predict(self):
        """
        Bước predict của Kalman, trả về bbox [x1,y1,x2,y2] dạng float.
        """
        self.kf.predict()

        self.age += 1
        if self.time_since_update > 0:
            self.hits = 0
        self.time_since_update += 1

        cx = self.kf.x[0, 0]
        cy = self.kf.x[1, 0]
        s = self.kf.x[2, 0]
        r = self.kf.x[3, 0]

        w = math.sqrt(max(1e-6, s * r))
        h = max(1.0, s / (w + 1e-6))

        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0

        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2
        self.history.append((int((x1 + x2) // 2), int((y1 + y2) // 2)))

        return [x1, y1, x2, y2]

    def update(self, bbox, conf=0.0, cls=-1):
        """
        Cập nhật Kalman với bbox đo mới + gán lại conf, cls.
        """
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        s = (x2 - x1) * (y2 - y1)
        r = (x2 - x1) / max(1.0, (y2 - y1))

        z = np.array([cx, cy, s, r], dtype=float)
        self.kf.update(z)

        self.time_since_update = 0
        self.hits += 1

        self.conf = conf
        self.cls = cls

        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2

    def get_state(self):
        """
        Lấy bbox hiện tại từ state Kalman, dạng [x1,y1,x2,y2].
        """
        cx = self.kf.x[0, 0]
        cy = self.kf.x[1, 0]
        s = self.kf.x[2, 0]
        r = self.kf.x[3, 0]

        w = math.sqrt(max(1e-6, s * r))
        h = max(1.0, s / (w + 1e-6))

        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0

        return [x1, y1, x2, y2]

    def calculate_speed(self):
        """
        Hàm gốc trong code: trả về khoảng cách pixel giữa 2 điểm lịch sử gần nhất.
        (TrafficProcessor sẽ xử lý thêm để suy ra tốc độ km/h).
        """
        if len(self.history) < 2:
            return 0.0

        (x1, y1) = self.history[-2]
        (x2, y2) = self.history[-1]
        dist = math.hypot(x2 - x1, y2 - y1)
        return dist


class Sort:
    """
    SORT tối giản, dùng KalmanBoxTracker cho từng track.
    """

    def __init__(
        self,
        max_age: int = SORT_MAX_AGE,
        min_hits: int = SORT_MIN_HITS,
        iou_threshold: float = SORT_IOU_THRESH,
        fps: float = 30.0,
        frame_height: int = 1080,
    ):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold

        self.trackers = []
        self.fps = fps
        self.frame_height = frame_height

    def update(self, dets):
        """
        Cập nhật tracker với detections:
        dets: Nx6 [x1,y1,x2,y2,cls,conf] (float).
        Trả về Nx7 [x1,y1,x2,y2,cls,conf,tid].
        """
        if dets is None:
            dets = np.zeros((0, 6), dtype=np.float32)

        ret = []

        # predict cho tất cả trackers hiện tại
        for t in self.trackers:
            t.predict()

        used_d = set()

        # gán từng detection cho tracker tốt nhất (nếu IoU đủ lớn)
        for i, d in enumerate(dets):
            best_iou = 0.0
            best_t = None

            for t in self.trackers:
                pred = t.get_state()
                score = iou_xyxy(d[:4], pred)
                if score > best_iou:
                    best_iou = score
                    best_t = t

            if best_iou >= self.iou_threshold and best_t is not None:
                best_t.update(d[:4], conf=float(d[5]), cls=int(d[4]))
                used_d.add(i)
            else:
                # tạo track mới
                newt = KalmanBoxTracker(
                    d[:4], fps=self.fps, frame_height=self.frame_height
                )
                newt.update(d[:4], conf=float(d[5]), cls=int(d[4]))
                self.trackers.append(newt)
                used_d.add(i)

        # xoá tracker “già” (không được update quá max_age)
        for t in self.trackers[:]:
            if t.time_since_update > self.max_age:
                self.trackers.remove(t)
                continue

            bbox = t.get_state()
            cls = int(t.cls)
            conf = float(t.conf)
            tid = t.id + 1  # ID 1-based cho dễ quan sát

            ret.append(
                [
                    int(bbox[0]),
                    int(bbox[1]),
                    int(bbox[2]),
                    int(bbox[3]),
                    int(cls),
                    float(conf),
                    int(tid),
                ]
            )

        if len(ret) == 0:
            return np.zeros((0, 7), dtype=np.float32)

        return np.array(ret, dtype=np.float32)

    def getTrackers(self):
        """
        Trả về danh sách đối tượng KalmanBoxTracker (dùng cho vẽ trajectory, speed...).
        """
        return self.trackers
