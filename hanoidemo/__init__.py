"""
Package chính cho dự án HanoiFlow.

Các module con:
- config: cấu hình & hằng số dùng chung
- detection: YOLO & xử lý bounding box
- tracking: SORT / Kalman / trajectory
- speed: logic tính tốc độ, line crossing, homography
- heatmap: tích luỹ & vẽ heatmap giao thông
- gui: giao diện PyQt5 (HanoiFlowApp, VideoLabel, Worker, ...)

Bạn sẽ import từ đây theo dạng:
    from hanoidemo.config import *
    from hanoidemo.gui.gui import HanoiFlowApp
"""
