"""
speed_estimator.py
------------------
SpeedEstimator:
- Lưu history toạ độ thế giới (m) cho từng tracker_id
- Khi đủ lịch sử, dùng điểm đầu & cuối để tính tốc độ trung bình.

Thiết kế:
- Dùng ViewTransformer để đổi pixel (bottom-center bbox) sang mét.
- Số frame tối thiểu để tính tốc độ được cấu hình qua SPEED_MIN_HISTORY_FRAMES
  trong config.py để tốc độ hiển thị nhanh hơn (không phải đợi ~0.5s).
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, Deque, Tuple, Optional

import numpy as np

from hanoidemo.config import SPEED_MIN_HISTORY_FRAMES  # lấy giá trị từ config
from .view_transformer import ViewTransformer


class SpeedEstimator:
    """
    SpeedEstimator(fps, view_transformer, max_history_seconds)

    - fps: số frame/giây của VIDEO (không phải tốc độ chạy model)
    - view_transformer: đối tượng ViewTransformer dùng để đổi pixel -> mét
    - max_history_seconds: chiều dài tối đa cửa sổ thời gian (s)
    """

    def __init__(
        self,
        fps: float,
        view_transformer: ViewTransformer,
        max_history_seconds: float = 1.0,
        min_history_frames: Optional[int] = None,
    ) -> None:
        self.fps = float(max(1.0, fps))
        self.view_transformer = view_transformer
        self.max_history_seconds = float(max_history_seconds)

        # số frame tối thiểu để tính tốc độ (có thể nhỏ hơn fps/2)
        if min_history_frames is None:
            self.min_history_frames = int(max(1, SPEED_MIN_HISTORY_FRAMES))
        else:
            self.min_history_frames = int(max(1, min_history_frames))

        hist_len = max(
            self.min_history_frames, int(self.fps * self.max_history_seconds)
        )
        self._coords: Dict[int, Deque[np.ndarray]] = defaultdict(
            lambda: deque(maxlen=hist_len)
        )

    # ------------- cấu hình lại khi đổi video -------------

    def reset_for_new_video(
        self,
        fps: Optional[float] = None,
        min_history_frames: Optional[int] = None,
    ) -> None:
        """
        Gọi mỗi khi mở video mới hoặc fps thay đổi.
        Có thể đồng thời cập nhật lại min_history_frames nếu muốn.
        """
        if fps is not None:
            self.fps = float(max(1.0, fps))

        if min_history_frames is not None:
            self.min_history_frames = int(max(1, min_history_frames))
        else:
            # nếu không truyền vào thì vẫn dùng cấu hình đang có
            self.min_history_frames = int(max(1, SPEED_MIN_HISTORY_FRAMES))

        hist_len = max(
            self.min_history_frames, int(self.fps * self.max_history_seconds)
        )
        self._coords = defaultdict(lambda: deque(maxlen=hist_len))

    # ------------- cập nhật & tính tốc độ -------------

    def update(self, tracker_id: int, bottom_center_px: Tuple[float, float]) -> None:
        """
        Thêm một điểm mới cho tracker_id.

        bottom_center_px: (cx, cy) pixel TẠI ĐÁY bbox.
        """
        px_arr = np.asarray(bottom_center_px, dtype=np.float32).reshape(1, 2)
        world_pt = self.view_transformer.transform_points(px_arr)[0]  # (x_m, y_m)
        self._coords[tracker_id].append(world_pt)

    def get_speed_kmh(self, tracker_id: int) -> float:
        """
        Trả về tốc độ km/h nếu lịch sử đủ dài, ngược lại 0.

        Điều kiện: số mẫu >= self.min_history_frames.
        Với fps ~25 và min_history_frames=5, thời gian quan sát ~0.2s.
        """
        coords = self._coords.get(tracker_id)
        if not coords:
            return 0.0

        n = len(coords)
        if n < self.min_history_frames:
            return 0.0

        start = np.asarray(coords[0], dtype=np.float32)
        end = np.asarray(coords[-1], dtype=np.float32)

        dist_m = float(np.linalg.norm(end - start))
        time_s = n / self.fps
        if time_s <= 1e-3:
            return 0.0

        return float((dist_m / time_s) * 3.6)
