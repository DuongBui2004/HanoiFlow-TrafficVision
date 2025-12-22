"""
gui package
-----------
Chứa:
- VideoLabel: widget vẽ làn đường trên khung hình
- Worker: QThread xử lý video + gọi TrafficProcessor
- HanoiFlowApp: QMainWindow chính của ứng dụng (layout mới)
"""

from .gui import VideoLabel, Worker, HanoiFlowApp

__all__ = ["VideoLabel", "Worker", "HanoiFlowApp"]
