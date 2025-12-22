"""
speed package
-------------
Expose các thành phần để GUI sử dụng:
- TrafficProcessor: lõi YOLO + SORT + Speed + heatmap + lanes
- Các hàm tiện ích: viewport, vẽ lanes, vẽ bbox + speed, lưu/đọc lanes JSON
"""

from .traffic_processor import (
    TrafficProcessor,
    compute_display_rect,
    map_lanes_label_to_frame,
    draw_lanes,
    draw_boxes_with_trajectory_and_speed,
    assign_lane_id,
    save_lanes_json,
    load_lanes_json,
)

__all__ = [
    "TrafficProcessor",
    "compute_display_rect",
    "map_lanes_label_to_frame",
    "draw_lanes",
    "draw_boxes_with_trajectory_and_speed",
    "assign_lane_id",
    "save_lanes_json",
    "load_lanes_json",
]
