"""
yolo_utils.py
-------------
Các hàm liên quan đến YOLO:
- load_yolo: nạp model theo YOLO_WEIGHTS
- manual_nms_xyxy / manual_nms: NMS thủ công trên bbox xyxy
- post_filter_dets: hậu xử lý & lọc bbox
- detect_yolo: gọi model.predict và trả về [x1,y1,x2,y2,cls,conf]
"""

from hanoidemo.config import (
    YOLO_WEIGHTS,
    YOLO_DEFAULT,
    DEVICE,
    USE_MARGIN_FILTER,
    MARGIN_PCT,
    MIN_BOX_AREA_REL,
    MAX_BOX_AREA_REL,
    ASPECT_MIN,
    ASPECT_MAX,
    USE_CLASS_FILTER,
    VEHICLE_CLASSES,
    CONF_CLASS_MIN,
    CONF,
)
from ultralytics import YOLO
import numpy as np
import cv2


def load_yolo(model_key: str):
    """
    Nạp model YOLO theo key trong YOLO_WEIGHTS.
    Ví dụ: model_key = "yolo11n".
    """
    w = YOLO_WEIGHTS.get(model_key, YOLO_WEIGHTS[YOLO_DEFAULT])
    model = YOLO(w)

    try:
        # Một số phiên bản ultralytics cho phép fuse() để tối ưu inference
        model.fuse()
    except Exception:
        # Nếu không hỗ trợ thì bỏ qua, không ảnh hưởng chạy chính
        pass

    return model


def _box_area_xyxy(x1, y1, x2, y2):
    """Diện tích bbox (xyxy)."""
    return max(0, x2 - x1) * max(0, y2 - y1)


def _aspect_ratio_xyxy(x1, y1, x2, y2):
    """Tỉ lệ w/h của bbox (tránh box quá dẹt hoặc quá cao)."""
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    return w / h


def manual_nms_xyxy(dets, iou_thr=0.65):
    """
    NMS thủ công.
    dets: Nx6 [x1,y1,x2,y2,cls,conf] (float32).
    Trả về mảng cùng format sau khi loại bớt bbox trùng.
    """
    if dets is None or len(dets) == 0:
        return dets

    # sort theo confidence giảm dần
    dets = dets[dets[:, 5].argsort()[::-1]]

    keep = []
    while len(dets) > 0:
        cur = dets[0]
        keep.append(cur)

        if len(dets) == 1:
            break

        rest = dets[1:]

        x1 = np.maximum(cur[0], rest[:, 0])
        y1 = np.maximum(cur[1], rest[:, 1])
        x2 = np.minimum(cur[2], rest[:, 2])
        y2 = np.minimum(cur[3], rest[:, 3])

        inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

        area_cur = _box_area_xyxy(cur[0], cur[1], cur[2], cur[3])
        area = (rest[:, 2] - rest[:, 0]) * (rest[:, 3] - rest[:, 1])

        iou = inter / (area_cur + area - inter + 1e-6)

        dets = rest[iou < iou_thr]

    return np.array(keep, dtype=np.float32)


# Tên cũ dùng trong pipeline — giữ alias cho thuận tiện
manual_nms = manual_nms_xyxy


def post_filter_dets(dets_xyxy, frame_shape):
    """
    Lọc bbox sau YOLO + NMS:
    - ép vào trong frame
    - bỏ box quá nhỏ/quá to
    - bỏ box tỉ lệ w/h bất thường
    - lọc theo class & conf từng lớp
    """
    H, W = frame_shape[:2]
    out = []

    for x1, y1, x2, y2, cls, conf in dets_xyxy:
        # ép toạ độ vào trong frame
        x1 = max(0, min(W - 1, x1))
        y1 = max(0, min(H - 1, y1))
        x2 = max(0, min(W - 1, x2))
        y2 = max(0, min(H - 1, y2))

        if x2 <= x1 or y2 <= y1:
            continue

        # margin filter (bỏ box sát biên)
        if USE_MARGIN_FILTER:
            m = int(MARGIN_PCT * min(W, H))
            if x1 <= m or y1 <= m or x2 >= (W - 1 - m) or y2 >= (H - 1 - m):
                continue

        # kiểm tra diện tích theo tỉ lệ frame
        area = _box_area_xyxy(x1, y1, x2, y2) / (W * H)
        if area < MIN_BOX_AREA_REL or area > MAX_BOX_AREA_REL:
            continue

        # kiểm tra tỉ lệ w/h
        ar = _aspect_ratio_xyxy(x1, y1, x2, y2)
        if ar < ASPECT_MIN or ar > ASPECT_MAX:
            continue

        # lọc class
        if USE_CLASS_FILTER and int(cls) not in VEHICLE_CLASSES:
            continue

        # ngưỡng conf riêng cho từng class
        cmin = CONF_CLASS_MIN.get(int(cls), 0.0)
        if conf < max(CONF, cmin):
            continue

        out.append([x1, y1, x2, y2, int(cls), float(conf)])

    if len(out) == 0:
        return np.zeros((0, 6), dtype=np.float32)

    return np.array(out, dtype=np.float32)


def detect_yolo(model, frame_bgr, imgsz=1280, conf=0.25, iou=0.45, classes=None):
    """
    Chạy YOLO trên 1 frame BGR:
    - model: đối tượng YOLO đã load
    - frame_bgr: ảnh BGR (numpy)
    - imgsz, conf, iou, classes: tham số infer
    Trả về: Nx6 [x1,y1,x2,y2,cls,conf] (float32).
    """
    res = model.predict(
        source=frame_bgr,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        classes=classes,
        device=DEVICE,
        verbose=False,
    )

    r = res[0]
    if getattr(r, "boxes", None) is None or len(r.boxes) == 0:
        return np.zeros((0, 6), dtype=np.float32)

    b = r.boxes
    xyxy = b.xyxy.cpu().numpy()
    confs = b.conf.cpu().numpy()
    clss = b.cls.cpu().numpy()

    out = np.concatenate(
        [xyxy[:, :4], clss.reshape(-1, 1), confs.reshape(-1, 1)],
        axis=1,
    )

    return out.astype(np.float32)
