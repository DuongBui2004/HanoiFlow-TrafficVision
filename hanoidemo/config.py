"""
config.py
---------
Toàn bộ import chung và cấu hình global cho HanoiFlow:
- Thư viện dùng cho SORT, YOLO, GUI
- Đường dẫn output, logging
- Tham số YOLO, lọc hình học, SORT tracker
- Cấu hình heatmap, trajectory overlay
- Homography & SPEED ROI (hình thang) + line crossing
"""

# =========================================
# Import Libraries — Đầy đủ cho SORT Kalman Tracker
# =========================================

import sys, os, time, json, logging, math, shutil, csv
from pathlib import Path
from datetime import datetime
from collections import deque, Counter, defaultdict
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO
from random import randint

# Import SORT tracker dependencies
try:
    from filterpy.kalman import KalmanFilter
    FILTERPY_AVAILABLE = True
except ImportError:
    print("Warning: filterpy not installed. Install with: pip install filterpy")
    FILTERPY_AVAILABLE = False

try:
    import lap
    LAP_AVAILABLE = True
except ImportError:
    from scipy.optimize import linear_sum_assignment
    LAP_AVAILABLE = False

print("=" * 80)
print("Imports loaded successfully!")
try:
    print(f"- PyTorch: {torch.__version__}")
    print(f"- CUDA available: {torch.cuda.is_available()}")
except Exception:
    print("- PyTorch info unavailable")
print(f"- filterpy: {FILTERPY_AVAILABLE}")
print(f"- lap: {LAP_AVAILABLE}")
print("=" * 80)

# =========================================
# GLOBAL CONFIGURATION
# =========================================

BASE_OUT = Path(r"D:\Kỳ 1 25-26\Kết quả\Đồ án thị giác máy tính")
BASE_OUT.mkdir(parents=True, exist_ok=True)

LANES_DIR = BASE_OUT / "lanes"
LANES_DIR.mkdir(exist_ok=True)

LOG_DIR = BASE_OUT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "runtime.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)

# YOLO models
YOLO_WEIGHTS = {
    "yolov8n": "yolov8n.pt",
    "yolov8s": "yolov8s.pt",
    "yolov8m": "yolov8m.pt",
    "yolov8l": "yolov8l.pt",
    "yolov8x": "yolov8x.pt",
    "yolo11n": "yolo11n.pt",
    "yolo11s": "yolo11s.pt",
    "yolo11m": "yolo11m.pt",
    "yolo11l": "yolo11l.pt",
    "yolo11x": "yolo11x.pt",
    "yolo12n": "yolo12n.pt",
    "yolo12s": "yolo12s.pt",
    "yolo12x": "yolo12x.pt",
}

YOLO_DEFAULT = "yolo11n"
DEVICE = 0 if torch.cuda.is_available() else "cpu"

CONF = 0.23
IOU = 0.45
IMGSZ = 1280

# Class filtering
USE_CLASS_FILTER = True
VEHICLE_CLASSES = [0, 1, 2, 3, 5, 7]

# Geometric filters
MIN_BOX_AREA_REL = 0.0012
MAX_BOX_AREA_REL = 0.15
ASPECT_MIN, ASPECT_MAX = 0.25, 5.5

# Per-class conf thresholds
CONF_CLASS_MIN = {0: 0.28, 1: 0.25, 2: 0.26, 3: 0.26, 5: 0.48, 7: 0.48}

# NMS
USE_MANUAL_NMS = True
NMS_IOU_THRESH = 0.65

# Class smoothing
USE_CLASS_SMOOTHING = True
CLASS_SMOOTH_WINDOW = 7
USE_CAR_BIAS = True

# SORT Tracker
SORT_MAX_AGE = 5
SORT_MIN_HITS = 2
SORT_IOU_THRESH = 0.2

# Frame safety filters
USE_MARGIN_FILTER = True
MARGIN_PCT = 0.05
USE_OUT_OF_FRAME_CHECK = True

# Visualization (GitHub-style)
DRAW_CENTER_DOT = True
CENTER_DOT_COLOR = (255, 255, 255)
CENTER_DOT_RADIUS = 4

DRAW_LABEL_FULL = True
LABEL_BG_COLOR = (0, 255, 255)
LABEL_TEXT_COLOR = (0, 0, 0)
LABEL_FONT_SCALE = 0.6
LABEL_THICKNESS = 1

USE_CLASS_COLORS = True
BOX_THICKNESS = 2

CLASS_COLORS = {
    0: (255, 0, 255),
    1: (0, 255, 0),
    2: (0, 0, 255),
    3: (255, 255, 0),
    5: (255, 165, 0),
    7: (128, 0, 128),
}

DEFAULT_BOX_COLOR = (0, 255, 255)

# Heatmap
USE_HEATMAP = True
HEATMAP_COLORMAP = cv2.COLORMAP_TURBO
HEATMAP_ALPHA = 0.60

HEATMAP_POINT_SHAPE = "square"
HEATMAP_POINT_RADIUS_REL = 0.020
HEATMAP_PX_RADIUS = None
HEATMAP_BLUR_KERNEL = 41
HEATMAP_DECAY = 0.990
HEATMAP_POINT_WEIGHT = 40.0
HEATMAP_GAIN = 3.0
HEATMAP_HISTORY_STEPS = 8
HEATMAP_PERCENTILE = 97
HEATMAP_BASELINE = 0.06
HEATMAP_MIN_ROI_RATIO = 0.02
HEATMAP_CLASSES = VEHICLE_CLASSES
HEATMAP_USE_LANE_ROI = True

HEATMAP_SNAPSHOT_DIR = BASE_OUT / "heatmap_snapshots"
HEATMAP_SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

# Trajectory overlay
DRAW_TRAJECTORY = True
TRAJECTORY_THICKNESS = 2

# Misc
USE_DRIVABLE_MASK = False
WRITE_OVERLAY = True
FOURCC = "mp4v"
AGG_WINDOW_SEC = 10
KEEP_ASPECT = True
SYNC_TO_INPUT_FPS = True

# Homography placeholder
HOMO_FILE = BASE_OUT / "calibration.json"
if not HOMO_FILE.exists():
    HOMO_FILE.write_text('{"H": null, "meters_per_px": 1.0}', encoding="utf-8")

# UI/metrics
LANE_NAMES = ["A", "B", "C", "D", "E", "F"]
LANE_COLORS = [
    (102, 255, 255),
    (255, 204, 153),
    (204, 255, 204),
    (204, 204, 255),
    (255, 204, 229),
    (173, 216, 230),
]

RIGHT_PANEL_COLUMNS = [
    "Lane",
    "Density",
    "AvgSpeed(km/h)",
    "Flow(q veh/h)",
    "Occ(%)",
    "TravelTime(s)",
]

TEST_IMGSZ = 1280
TEST_CONF = 0.20

# CSV outputs
CSV_EVENTS = BASE_OUT / "cv_events.csv"
CSV_FLOWS = BASE_OUT / "cv_flows_agg.csv"
CSV_TT = BASE_OUT / "cv_traveltime.csv"

# =========================================
# PERSPECTIVE TRANSFORMATION (theo repo)
# =========================================

GITHUB_SOURCE_POINTS = np.array(
    [
        [450.0, 300.0],   # Top-left
        [860.0, 300.0],   # Top-right
        [1900.0, 720.0],  # Bottom-right (ngoài frame 1280x720)
        [-660.0, 720.0],  # Bottom-left  (ngoài frame 1280x720)
    ],
    dtype=np.float32,
)

WIDTH_METERS = 25.0
HEIGHT_METERS = 100.0

GITHUB_TARGET_POINTS = np.array(
    [
        [0.0, 0.0],
        [WIDTH_METERS, 0.0],
        [WIDTH_METERS, HEIGHT_METERS],
        [0.0, HEIGHT_METERS],
    ],
    dtype=np.float32,
)

GITHUB_HOMOGRAPHY_MATRIX = cv2.getPerspectiveTransform(
    GITHUB_SOURCE_POINTS, GITHUB_TARGET_POINTS
)

print("\n" + "=" * 80)
print("✓ GITHUB PERSPECTIVE TRANSFORMATION LOADED")
print("=" * 80)
print("SOURCE_POINTS (Image pixels):")
for i, pt in enumerate(GITHUB_SOURCE_POINTS):
    print(f"  [{i}] ({pt[0]:.0f}, {pt[1]:.0f})")
print("TARGET_POINTS (Real-world meters):")
for i, pt in enumerate(GITHUB_TARGET_POINTS):
    print(f"  [{i}] ({pt[0]:.1f}, {pt[1]:.1f})")
print("=" * 80 + "\n")

# =========================================
# LINE CROSSING
# =========================================

LINE_CROSSING_Y = 480
LINE_CROSSING_BUFFER = 20

DRAW_LINE_CROSSING = True
LINE_CROSSING_COLOR = (255, 255, 0)
LINE_CROSSING_THICKNESS = 3

# =========================================
# SPEED ROI TRAPEZOID — DÙNG NGUYÊN SOURCE_POINTS (GIỐNG REPO)
# =========================================
# Repo dùng SOURCE_POINTS làm polygon_zone, nên ở đây ta cũng dùng chính
# GITHUB_SOURCE_POINTS (kể cả 2 điểm ngoài khung). OpenCV sẽ tự cắt polygon
# theo biên ảnh khi vẽ, nên ROI trong frame giống hệt repo. [attached_file:72][attached_file:73]

SPEED_ROI_POLYGON = GITHUB_SOURCE_POINTS.astype(np.int32)

DRAW_SPEED_ROI = True
SPEED_ROI_COLOR = (0, 255, 0)
SPEED_ROI_THICKNESS = 2

# Tham số cho SpeedEstimator (số frame tối thiểu để tính tốc độ)
SPEED_MIN_HISTORY_FRAMES = 5
SPEED_FILTER_KMH = 0  # 0 = không lọc theo ngưỡng

print(f"✓ SPEED ROI: TRAPEZOID via GITHUB_SOURCE_POINTS ✓")
print(f"✓ LINE CROSSING at Y={LINE_CROSSING_Y} pixels")
print("✓ SPEED ROI & homography now match original GitHub repo geometry")
print("=" * 80 + "\n")
