"""
heatmap.py
----------
Cài đặt:
- HeatmapAccumulator: tích luỹ heatmap từ các điểm (cx, cy) hoặc từ mask
  foreground theo thời gian, có decay, blur, colormap, snapshot PNG.
- build_lane_roi_mask: tạo mask nhị phân ROI từ danh sách polygon lanes.
"""

import cv2
import numpy as np

from hanoidemo.config import (
    HEATMAP_DECAY,
    HEATMAP_BLUR_KERNEL,
    HEATMAP_POINT_WEIGHT,
    HEATMAP_PX_RADIUS,
    HEATMAP_POINT_RADIUS_REL,
    HEATMAP_POINT_SHAPE,
    HEATMAP_GAIN,
)


class HeatmapAccumulator:
    """
    Bộ tích luỹ heatmap 2D theo thời gian.

    - mat: ma trận float32 (H x W) lưu cường độ.
    - apply_decay(): giảm dần cường độ (phai màu theo thời gian).
    - add_point(): thêm điểm/patch vào heatmap (từ cx, cy, radius).
    - add_mask(): cộng nguyên một mask foreground (0/255).
    - render(): phủ heatmap lên frame BGR.
    - snapshot(): lưu heatmap hiện tại ra file PNG.
    """

    def __init__(self, shape_hw, decay=None, blur_kernel=None, colormap=cv2.COLORMAP_TURBO):
        h, w = shape_hw[:2]
        self.h, self.w = h, w

        # hệ số phai màu
        self.decay = float(decay if decay is not None else HEATMAP_DECAY)

        # kernel blur (bắt buộc lẻ)
        bk = int(blur_kernel if blur_kernel is not None else HEATMAP_BLUR_KERNEL)
        self.blur_kernel = bk if (bk % 2 == 1) else (bk + 1)

        self.colormap = colormap
        self.mat = np.zeros((h, w), dtype=np.float32)

    # ================= reset / decay =================

    def reset(self):
        """Đặt toàn bộ heatmap về 0."""
        self.mat[:] = 0.0

    def apply_decay(self):
        """Nhân toàn bộ ma trận với hệ số decay (<1)."""
        if self.decay < 1.0:
            self.mat *= self.decay

    # ================= add từ điểm (cx, cy) =================

    def add_point(self, cx, cy, radius=None, weight=None):
        """
        Thêm một điểm (cx, cy) lên heatmap.

        - radius:
            + Nếu truyền vào (ví dụ: max(w, h) / 2 của bbox) thì patch heatmap
              sẽ phủ gần trọn diện tích xe.
            + Nếu None: dùng HEATMAP_PX_RADIUS (cố định) hoặc HEATMAP_POINT_RADIUS_REL.
        - weight: trọng số cộng thêm. Nếu None dùng HEATMAP_POINT_WEIGHT.
        """
        if cx < 0 or cy < 0 or cx >= self.w or cy >= self.h:
            return

        point_weight = float(weight if weight is not None else HEATMAP_POINT_WEIGHT)

        # bán kính theo bbox (nếu có) hoặc theo config
        try:
            if radius is not None:
                # radius nên là nửa cạnh dài bbox => patch ≈ box
                px_radius = int(max(3, radius))
            elif HEATMAP_PX_RADIUS is not None:
                px_radius = int(max(3, HEATMAP_PX_RADIUS))
            else:
                px_radius = max(3, int(HEATMAP_POINT_RADIUS_REL * max(self.h, self.w)))
        except Exception:
            px_radius = 10

        shape = HEATMAP_POINT_SHAPE

        y1 = max(0, cy - px_radius)
        y2 = min(self.h, cy + px_radius + 1)
        x1 = max(0, cx - px_radius)
        x2 = min(self.w, cx + px_radius + 1)
        if x2 <= x1 or y2 <= y1:
            return

        if shape == "square":
            # patch vuông: kích thước ~ 2*px_radius, che gần hết bbox
            self.mat[y1:y2, x1:x2] += point_weight
        else:
            # Gaussian patch mịn, vẫn mở rộng bằng px_radius (cỡ bbox)
            yy, xx = np.ogrid[y1:y2, x1:x2]
            sigma = max(1.0, px_radius * 0.6)
            g = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * sigma ** 2))
            self.mat[y1:y2, x1:x2] += (point_weight * g).astype(np.float32)

    # ================= add từ mask foreground =================

    def add_mask(self, fg_mask: np.ndarray, weight: float = 1.0):
        """
        Tích luỹ heatmap từ mask foreground (0/255) giống code mẫu dùng
        background_subtractor.

        - fg_mask: ảnh đơn kênh uint8 cùng kích thước frame (0: nền, 255: foreground).
        - weight: hệ số nhân thêm để điều chỉnh mức đóng góp.
        """
        if fg_mask is None:
            return
        if fg_mask.shape[:2] != (self.h, self.w):
            fg_mask = cv2.resize(fg_mask, (self.w, self.h), interpolation=cv2.INTER_NEAREST)

        fg = cv2.GaussianBlur(fg_mask, (3, 3), 0)
        fg = fg.astype(np.float32) / 255.0
        self.mat += fg * float(weight)

    # ================= render / snapshot =================

    def _build_color_heatmap(self):
        """Từ self.mat -> blur -> gain -> normalize -> colormap (không overlay)."""
        k = int(self.blur_kernel)
        k = k if (k % 2 == 1) else (k + 1)

        hm = cv2.GaussianBlur(self.mat, (k, k), 0)

        gain = float(HEATMAP_GAIN)
        hm = hm * gain

        hm_norm = cv2.normalize(hm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cmap = cv2.applyColorMap(hm_norm, self.colormap)
        return cmap

    def render(self, frame_bgr, alpha=0.35, roi_mask=None):
        """
        Phủ heatmap (sau blur + colormap) lên frame BGR.
        - alpha: độ trong suốt của heatmap (0–1).
        - roi_mask: mask (H x W) =255 trong vùng muốn vẽ, =0 ở ngoài.
        """
        cmap = self._build_color_heatmap()

        if roi_mask is not None:
            if roi_mask.shape[:2] != (self.h, self.w):
                roi_mask = cv2.resize(roi_mask, (self.w, self.h), interpolation=cv2.INTER_NEAREST)
            inv = (roi_mask == 0)
            cmap[inv] = 0

        out = cv2.addWeighted(cmap, alpha, frame_bgr, 1.0 - alpha, 0.0)
        return out

    def snapshot(self, out_path_png):
        """
        Lưu heatmap hiện tại ra file PNG (không overlay lên frame).
        """
        cmap = self._build_color_heatmap()
        cv2.imwrite(str(out_path_png), cmap)


def build_lane_roi_mask(shape_hw, lanes_frame):
    """
    Tạo mask ROI (uint8 HxW) =255 trong vùng union các lane polygon, =0 ngoài.
    - shape_hw: (H, W) của frame.
    - lanes_frame: list các polygon (np.ndarray Nx2) trong toạ độ frame.
    """
    h, w = shape_hw[:2]
    if not lanes_frame:
        return None

    polys = [p for p in lanes_frame if p is not None and len(p) >= 3]
    if not polys:
        return None

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, polys, 255)
    return mask
