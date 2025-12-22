"""
tracking package
----------------
Chứa các thành phần:
- iou_xyxy: hàm tính IoU giữa hai bbox xyxy
- KalmanBoxTracker: 1 track (đối tượng) dùng Kalman Filter
- Sort: tracker đa đối tượng (dùng KalmanBoxTracker)
"""

from .tracking import iou_xyxy, KalmanBoxTracker, Sort

__all__ = ["iou_xyxy", "KalmanBoxTracker", "Sort"]
