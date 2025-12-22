"""
view_transformer.py
-------------------
Lớp ViewTransformer:
- Nhận vào 4 điểm nguồn (pixel) và 4 điểm đích (mét)
- Tính ma trận homography H
- Biến đổi một tập điểm 2D từ ảnh sang mặt phẳng thế giới

Ý tưởng giống repo gốc nhưng cách cài đặt/đặt tên khác để phù hợp dự án HanoiFlow.
"""

from __future__ import annotations

import cv2
import numpy as np


class ViewTransformer:
    def __init__(
        self,
        source_points: np.ndarray,
        target_points: np.ndarray,
    ) -> None:
        """
        Parameters
        ----------
        source_points : np.ndarray
            Mảng (4,2) chứa 4 điểm góc trong ảnh gốc (pixel).
        target_points : np.ndarray
            Mảng (4,2) chứa 4 điểm tương ứng trong hệ toạ độ thực (thường là mét).
        """
        src = np.asarray(source_points, dtype=np.float32).reshape(4, 2)
        dst = np.asarray(target_points, dtype=np.float32).reshape(4, 2)

        self.source = src
        self.target = dst

        # Tính homography: ảnh -> thế giới
        self._H = cv2.getPerspectiveTransform(self.source, self.target)

    @property
    def homography(self) -> np.ndarray:
        """Trả về ma trận homography 3x3."""
        return self._H.copy()

    def transform_points(self, pts_xy: np.ndarray) -> np.ndarray:
        """
        Biến đổi tập điểm từ ảnh sang toạ độ thế giới.

        Parameters
        ----------
        pts_xy : np.ndarray
            Mảng (N,2) hoặc (N,1,2) các điểm (x,y) pixel trong ảnh.

        Returns
        -------
        np.ndarray
            Mảng (N,2) điểm trong hệ toạ độ mục tiêu (đơn vị mét).
        """
        pts = np.asarray(pts_xy, dtype=np.float32)
        if pts.ndim == 2:
            pts = pts.reshape(-1, 1, 2)
        if pts.size == 0:
            return np.zeros((0, 2), dtype=np.float32)

        warped = cv2.perspectiveTransform(pts, self._H)
        return warped.reshape(-1, 2)
