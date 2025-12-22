"""
detection package
-----------------
Chứa các hàm tiện ích liên quan đến YOLO:
- load_yolo: nạp model theo key
- manual_nms_xyxy / manual_nms: NMS thủ công trên bbox dạng xyxy
- post_filter_dets: lọc bbox theo kích thước, tỉ lệ, class, conf
- detect_yolo: chạy YOLO và trả về mảng [x1,y1,x2,y2,cls,conf]
"""

from .yolo_utils import (
    load_yolo,
    manual_nms_xyxy,
    manual_nms,
    post_filter_dets,
    detect_yolo,
)

__all__ = [
    "load_yolo",
    "manual_nms_xyxy",
    "manual_nms",
    "post_filter_dets",
    "detect_yolo",
]
