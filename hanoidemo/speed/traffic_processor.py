"""
traffic_processor.py
--------------------
Ghép:
- YOLO (hanoidemo.detection.yolo_utils)
- SORT (hanoidemo.tracking.tracking)
- SpeedEstimator + ViewTransformer (hanoidemo.speed.raafat)
- Heatmap + Lanes

Tính tốc độ:
- Mỗi frame, lấy điểm bottom-center (cx, y2) của bbox
- Dùng homography từ GITHUB_SOURCE_POINTS -> GITHUB_TARGET_POINTS để đổi sang mét
- Dựa vào history theo tracker_id để suy ra km/h
"""

import time
import json
import logging
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime

import cv2
import numpy as np

from hanoidemo.config import *  # dùng các hằng đã định nghĩa trong config
from hanoidemo.detection.yolo_utils import (
    load_yolo,
    detect_yolo,
    manual_nms,
    post_filter_dets,
)
from hanoidemo.tracking.tracking import Sort
from hanoidemo.heatmap.heatmap import HeatmapAccumulator, build_lane_roi_mask
from hanoidemo.speed.raafat.view_transformer import ViewTransformer
from hanoidemo.speed.raafat.speed_estimator import SpeedEstimator


# =========================================================
# 1. Hàm tiện ích lanes / bbox / JSON
# =========================================================


def compute_display_rect(label_size, frame_size):
    Wl, Hl = label_size
    Wf, Hf = frame_size
    if not KEEP_ASPECT:
        return (0, 0, Wl, Hl)
    scale = min(Wl / max(1, Wf), Hl / max(1, Hf))
    new_w, new_h = int(Wf * scale), int(Hf * scale)
    off_x, off_y = (Wl - new_w) // 2, (Hl - new_h) // 2
    return (off_x, off_y, new_w, new_h)


def map_lanes_label_to_frame(lanes_label, label_size, frame_size, display_rect=None):
    if not lanes_label:
        return []
    Wl, Hl = label_size
    Wf, Hf = frame_size
    if display_rect is None:
        display_rect = compute_display_rect(label_size, frame_size)
    off_x, off_y, new_w, new_h = display_rect
    out = []
    for poly in lanes_label:
        pts = []
        for (x, y) in poly:
            fx = int((x - off_x) * (Wf / max(1, new_w)))
            fy = int((y - off_y) * (Hf / max(1, new_h)))
            pts.append([fx, fy])
        out.append(np.array(pts, dtype=np.int32) if len(pts) >= 3 else None)
    return out


def draw_lanes(img, lanes_frame):
    for i, poly in enumerate(lanes_frame):
        if poly is None or len(poly) < 3:
            continue
        color = LANE_COLORS[i % len(LANE_COLORS)]
        cv2.polylines(img, [poly], isClosed=True, color=color, thickness=2)
        overlay = img.copy()
        cv2.fillPoly(
            overlay,
            [poly],
            color=(color[0] // 3, color[1] // 3, color[2] // 3),
        )
        img[:] = cv2.addWeighted(overlay, 0.12, img, 0.88, 0)


def _class_color(cls_id: int):
    return CLASS_COLORS.get(int(cls_id), DEFAULT_BOX_COLOR)


def draw_boxes_with_trajectory_and_speed(
    img,
    tracks,
    yolo_names=None,
):
    """
    Vẽ bbox + nhãn speed từ list `tracks`.
    tracks: list dict {"id","bbox","cls","conf","speed"}.
    """
    for tr in tracks:
        x1, y1, x2, y2 = tr["bbox"]
        cls_id = int(tr["cls"])
        conf = float(tr["conf"])
        spd = float(tr["speed"])

        color = _class_color(cls_id) if USE_CLASS_COLORS else DEFAULT_BOX_COLOR

        cv2.rectangle(img, (x1, y1), (x2, y2), color, BOX_THICKNESS)

        if DRAW_CENTER_DOT:
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            cv2.circle(img, (cx, cy), CENTER_DOT_RADIUS, CENTER_DOT_COLOR, -1)

        if DRAW_LABEL_FULL:
            name = (
                yolo_names.get(cls_id, str(cls_id))
                if isinstance(yolo_names, dict)
                else str(cls_id)
            )
            txt = f"ID {tr['id']} | {name} {conf:.2f} | {spd:.1f} km/h"
            (tw, th), base = cv2.getTextSize(
                txt,
                cv2.FONT_HERSHEY_SIMPLEX,
                LABEL_FONT_SCALE,
                LABEL_THICKNESS,
            )
            cv2.rectangle(
                img,
                (x1, max(0, y1 - th - 6)),
                (x1 + tw + 6, y1),
                LABEL_BG_COLOR,
                -1,
            )
            cv2.putText(
                img,
                txt,
                (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                LABEL_FONT_SCALE,
                LABEL_TEXT_COLOR,
                LABEL_THICKNESS,
                cv2.LINE_AA,
            )

    return img


def assign_lane_id(bbox_xyxy, lanes_frame):
    if not lanes_frame:
        return -1
    x1, y1, x2, y2 = bbox_xyxy
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    for i, poly in enumerate(lanes_frame):
        if poly is None or len(poly) < 3:
            continue
        if cv2.pointPolygonTest(poly, (cx, cy), False) >= 0:
            return i
    return -1


def save_lanes_json(lanes_label, out_path: Path):
    try:
        data = [[list(map(int, pt)) for pt in poly] for poly in lanes_label]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps({"lanes": data}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return True
    except Exception as e:
        logging.exception(e)
        return False


def load_lanes_json(in_path: Path):
    try:
        obj = json.loads(in_path.read_text(encoding="utf-8"))
        lanes = obj.get("lanes", [])
        return [[(int(x), int(y)) for (x, y) in poly] for poly in lanes]
    except Exception as e:
        logging.exception(e)
        return []


# ===== Gom class YOLO thành 4 nhóm Bike/Car/Truck/Bus =====


def map_cls_to_group(cls_id: int) -> str:
    """
    Gom class YOLO (COCO) thành 4 nhóm chính để đếm xe từng loại.
    Có thể chỉnh lại mapping này tuỳ mô hình.
    """
    cid = int(cls_id)
    if cid in (1, 3):      # bicycle, motorcycle
        return "Bike"
    if cid in (2,):        # car
        return "Car"
    if cid in (7,):        # truck
        return "Truck"
    if cid in (5,):        # bus
        return "Bus"
    return "Car"


# =========================================================
# 2. TrafficProcessor – YOLO + SORT + SpeedEstimator
# =========================================================


class TrafficProcessor:
    def __init__(self, cam_id="cam0", model_key=YOLO_DEFAULT, max_history_seconds=1.0):
        self.cam_id = cam_id

        self.model = load_yolo(model_key)
        self.yolo_names = self.model.names if hasattr(self.model, "names") else {}

        self.tracker = Sort(
            max_age=SORT_MAX_AGE,
            min_hits=SORT_MIN_HITS,
            iou_threshold=SORT_IOU_THRESH,
            fps=30.0,
            frame_height=1080,
        )

        self.frame_size = (0, 0)
        self.video_fps = 30.0

        self.lanes_label = []
        self.lanes_frame = []

        self.heatmap = None
        self.hm_roi_mask = None
        self.hm_enabled = USE_HEATMAP

        self.writer = None
        self.cap = None

        # ViewTransformer + SpeedEstimator
        self.view_transformer = ViewTransformer(
            np.array(GITHUB_SOURCE_POINTS, dtype=np.float32),
            np.array(GITHUB_TARGET_POINTS, dtype=np.float32),
        )
        self.speed_estimator = SpeedEstimator(
            fps=self.video_fps,
            view_transformer=self.view_transformer,
            max_history_seconds=max_history_seconds,
        )

        # Thống kê IN/OUT tích lũy
        self.lane_class_counts = []   # list[Counter] cho từng lane
        self.lane_vehicle_total = []  # tổng số xe đã qua line
        self.lane_speed_sum = []      # tổng speed_kmh đã qua line

        self.global_vehicle_total = 0
        self.global_speed_sum = 0.0

        # Thống kê theo từng ID để tính “Vận tốc TB/xe”
        self.vehicle_speed_sum = defaultdict(float)
        self.vehicle_speed_count = defaultdict(int)

        # Đảm bảo mỗi xe chỉ đếm 1 lần IN/OUT
        self.logged_events = set()

    # ---------- Video I/O ----------

    def open_video(self, path: str):
        p = Path(path)
        self.cap = cv2.VideoCapture(str(p))
        if not self.cap.isOpened():
            raise RuntimeError(f"Không mở được video: {p}")

        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0

        self.video_fps = float(fps)
        self.frame_size = (w, h)

        if self.video_fps < 5 or self.video_fps > 120:
            self.video_fps = 30.0

        self.tracker = Sort(
            max_age=SORT_MAX_AGE,
            min_hits=SORT_MIN_HITS,
            iou_threshold=SORT_IOU_THRESH,
            fps=self.video_fps,
            frame_height=h,
        )

        self.speed_estimator.reset_for_new_video(self.video_fps)

        if self.hm_enabled:
            self.heatmap = HeatmapAccumulator(
                (h, w),
                decay=HEATMAP_DECAY,
                blur_kernel=HEATMAP_BLUR_KERNEL,
                colormap=HEATMAP_COLORMAP,
            )
            self.hm_roi_mask = None

        if WRITE_OVERLAY:
            fourcc = cv2.VideoWriter_fourcc(*FOURCC)
            out_path = BASE_OUT / f"overlay_{self.cam_id}_{p.stem}.mp4"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            self.writer = cv2.VideoWriter(
                str(out_path),
                fourcc,
                self.video_fps,
                (w, h),
            )

        self._reset_lane_stats()

    def close(self):
        try:
            if self.cap:
                self.cap.release()
        except Exception:
            pass
        try:
            if self.writer:
                self.writer.release()
        except Exception:
            pass
        self.cap = None
        self.writer = None

    # ---------- Quản lý lanes & thống kê ----------

    def _reset_lane_stats(self):
        L = len(self.lanes_label)
        self.lane_class_counts = [Counter() for _ in range(L)]
        self.lane_vehicle_total = [0 for _ in range(L)]
        self.lane_speed_sum = [0.0 for _ in range(L)]
        self.global_vehicle_total = 0
        self.global_speed_sum = 0.0
        self.vehicle_speed_sum = defaultdict(float)
        self.vehicle_speed_count = defaultdict(int)
        self.logged_events.clear()

    def set_lanes_label(self, lanes_label):
        valid_lanes = []
        for lane in lanes_label:
            if lane is not None and len(lane) >= 3:
                valid_lanes.append(lane)
        self.lanes_label = valid_lanes
        self.hm_roi_mask = None
        self._reset_lane_stats()

    # ---------- xử lý 1 frame ----------

    def process_one(self, frame_bgr: np.ndarray, label_size: tuple, display_rect=None):
        # Map lanes từ toạ độ label (GUI) sang toạ độ frame
        self.lanes_frame = map_lanes_label_to_frame(
            self.lanes_label,
            label_size,
            self.frame_size,
            display_rect=display_rect,
        )

        # YOLO detections
        det_raw = detect_yolo(
            self.model,
            frame_bgr,
            imgsz=IMGSZ,
            conf=CONF,
            iou=IOU,
            classes=None,
        )
        det_nms = manual_nms(det_raw, iou_thr=NMS_IOU_THRESH) if USE_MANUAL_NMS else det_raw
        det = post_filter_dets(det_nms, frame_bgr.shape)

        # SORT tracking
        trackers_output = self.tracker.update(det)

        # ===== LỌC THEO SPEED_ROI: chỉ giữ track trong vùng đa giác xanh lá =====
        roi_rows = []
        for row in trackers_output:
            x1, y1, x2, y2, cls, conf, tid = (
                int(row[0]),
                int(row[1]),
                int(row[2]),
                int(row[3]),
                int(row[4]),
                float(row[5]),
                int(row[6]),
            )
            cx = int((x1 + x2) / 2)
            cy = int(y2)  # tâm đáy bbox
            if cv2.pointPolygonTest(SPEED_ROI_POLYGON, (cx, cy), False) >= 0:
                roi_rows.append(row)

        if roi_rows:
            trackers_output = np.vstack(roi_rows).reshape(-1, 7)
        else:
            trackers_output = np.empty((0, 7), dtype=float)

        # Heatmap
        if self.hm_enabled and HEATMAP_USE_LANE_ROI and self.hm_roi_mask is None:
            self.hm_roi_mask = build_lane_roi_mask(frame_bgr.shape, self.lanes_frame)

        if self.hm_enabled and self.heatmap is not None:
            self.heatmap.apply_decay()
            allowed = set(HEATMAP_CLASSES) if HEATMAP_CLASSES is not None else None
            for row in trackers_output:
                x1, y1, x2, y2, cls, conf, tid = (
                    int(row[0]),
                    int(row[1]),
                    int(row[2]),
                    int(row[3]),
                    int(row[4]),
                    float(row[5]),
                    int(row[6]),
                )
                if allowed is not None and int(cls) not in allowed:
                    continue
                cx = int((x1 + x2) / 2)
                cy = int(y2)
                if self.hm_roi_mask is not None:
                    if (
                        cy < 0
                        or cx < 0
                        or cy >= self.hm_roi_mask.shape[0]
                        or cx >= self.hm_roi_mask.shape[1]
                        or self.hm_roi_mask[cy, cx] == 0
                    ):
                        continue
                self.heatmap.add_point(cx, cy)

        # ===== SpeedEstimator + thống kê IN/OUT + events =====
        tracks = []
        events = []

        for row in trackers_output:
            x1, y1, x2, y2, cls, conf, tid = (
                int(row[0]),
                int(row[1]),
                int(row[2]),
                int(row[3]),
                int(row[4]),
                float(row[5]),
                int(row[6]),
            )
            cx = int((x1 + x2) / 2)
            cy = int(y2)  # bottom-center

            # Tâm đáy đã được lọc nằm trong SPEED_ROI_POLYGON
            self.speed_estimator.update(
                tracker_id=tid,
                bottom_center_px=(cx, cy),
            )
            speed_kmh = self.speed_estimator.get_speed_kmh(tracker_id=tid)

            # Vận tốc TB/xe theo ID
            if speed_kmh > 0:
                self.vehicle_speed_sum[tid] += speed_kmh
                self.vehicle_speed_count[tid] += 1
                avg_vehicle = (
                    self.vehicle_speed_sum[tid] / self.vehicle_speed_count[tid]
                )
            else:
                avg_vehicle = 0.0

            lane_id = assign_lane_id([x1, y1, x2, y2], self.lanes_frame)
            group = map_cls_to_group(cls)

            tracks.append(
                {
                    "id": tid,
                    "bbox": [x1, y1, x2, y2],
                    "conf": conf,
                    "cls": cls,
                    "speed": speed_kmh,
                    "lane": lane_id,
                    "group": group,
                }
            )

            # Thống kê IN/OUT & toàn mạng (mỗi xe 1 lần)
            if speed_kmh > 0 and 0 <= lane_id < len(self.lane_vehicle_total):
                if tid not in self.logged_events:
                    self.logged_events.add(tid)

                    self.lane_vehicle_total[lane_id] += 1
                    self.lane_speed_sum[lane_id] += speed_kmh
                    if group:
                        self.lane_class_counts[lane_id][group] += 1

                    self.global_vehicle_total += 1
                    self.global_speed_sum += speed_kmh

            # Sự kiện gửi GUI (mỗi frame)
            if speed_kmh > 0:
                events.append(
                    {
                        "lane": lane_id,
                        "id": tid,
                        "type": group,
                        "speed": float(speed_kmh),
                        "avg_speed": float(avg_vehicle),
                        # thời gian thực có ngày-tháng-năm
                        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    }
                )

        # ===== Lane metrics tức thời (density/speed/occ hiện tại) =====
        L = len(self.lanes_frame)
        dens = [0] * L
        speeds = [[] for _ in range(L)]
        for tr in tracks:
            lid = tr.get("lane", -1)
            if lid < 0 or lid >= L:
                continue
            dens[lid] += 1
            speeds[lid].append(tr["speed"])
        avg = [(sum(s) / len(s) if s else 0.0) for s in speeds]
        occ = [min(100.0, d * 8.0) for d in dens]
        flow = [0] * L

        vis = frame_bgr.copy()

        # Không còn vẽ LINE_CROSSING màu xanh dương

        if DRAW_SPEED_ROI:
            cv2.polylines(
                vis,
                [SPEED_ROI_POLYGON],
                isClosed=True,
                color=SPEED_ROI_COLOR,
                thickness=SPEED_ROI_THICKNESS,
            )

        draw_lanes(vis, self.lanes_frame)
        vis = draw_boxes_with_trajectory_and_speed(
            vis,
            tracks,
            yolo_names=self.yolo_names,
        )

        if self.hm_enabled and self.heatmap is not None:
            vis = self.heatmap.render(
                vis,
                alpha=HEATMAP_ALPHA,
                roi_mask=self.hm_roi_mask,
            )

        cv2.putText(
            vis,
            f"YOLO+SORT | dets:{int(det.shape[0])} | Speed(PT) | HM:{'on' if self.hm_enabled else 'off'}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

        if self.writer:
            self.writer.write(vis)

        # lane_metrics: density/speed hiện tại + thống kê tích lũy
        lane_metrics = []
        for i in range(L):
            if i < len(self.lane_vehicle_total) and self.lane_vehicle_total[i] > 0:
                lane_avg_total = self.lane_speed_sum[i] / self.lane_vehicle_total[i]
                veh_total = int(self.lane_vehicle_total[i])
            else:
                lane_avg_total = 0.0
                veh_total = 0

            cc = self.lane_class_counts[i] if i < len(self.lane_class_counts) else Counter()

            lane_metrics.append(
                {
                    "lane": i,
                    "density": dens[i],
                    "speed": avg[i],
                    "flow": flow[i],
                    "occ": occ[i],
                    "lane_avg_total": float(lane_avg_total),
                    "vehicle_total": veh_total,
                    "class_counts": dict(cc),
                }
            )

        return vis, lane_metrics, events
