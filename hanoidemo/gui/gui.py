"""
gui.py
------
Giao diện PyQt5 cho ứng dụng HanoiFlow:

- VideoLabel: vẽ lanes trực tiếp trên QLabel
- Worker: QThread đọc video, gọi TrafficProcessor.process_one()
- HanoiFlowApp: cửa sổ chính, quản lý nút bấm, lanes, heatmap, metrics
"""


import sys
import time
import logging
from pathlib import Path


import cv2
from PyQt5 import QtWidgets, QtCore, QtGui


from hanoidemo.config import (
    YOLO_WEIGHTS,
    YOLO_DEFAULT,
    USE_HEATMAP,
    RIGHT_PANEL_COLUMNS,
    LANE_NAMES,
    LANES_DIR,
    TEST_IMGSZ,
    TEST_CONF,
    IOU,
    DEVICE,
    HEATMAP_DECAY,
    HEATMAP_BLUR_KERNEL,
    HEATMAP_COLORMAP,
)
from hanoidemo.speed import (
    TrafficProcessor,
    compute_display_rect,
    map_lanes_label_to_frame,
    draw_lanes,
    save_lanes_json,
    load_lanes_json,
)
from hanoidemo.heatmap import HeatmapAccumulator


# ===== THÊM DÒNG NÀY: thư mục mặc định =====
DEFAULT_DATA_DIR = r"D:\Kỳ 1 25-26\Data\Đồ án thị giác máy tính"


# =========================================================
# VideoLabel: widget vẽ lane
# =========================================================


class VideoLabel(QtWidgets.QLabel):
    pointAdded = QtCore.pyqtSignal(tuple)
    polygonCommitted = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setStyleSheet("background-color:black;")
        self.setMinimumSize(960, 540)
        self.setAlignment(QtCore.Qt.AlignCenter)

        self.drawing = False
        self.curr = []       # lane đang vẽ
        self.lanes = []      # danh sách lanes đã commit
        self.redo_stack = [] # undo điểm

        self.setContextMenuPolicy(QtCore.Qt.NoContextMenu)

    # ---------- API ----------
    def start_draw(self):
        self.drawing = True
        self.curr = []
        self.redo_stack.clear()
        self.update()

    def undo_point(self):
        if self.curr:
            self.redo_stack.append(self.curr.pop())
            self.update()

    def clear_draw(self):
        self.drawing = False
        self.curr = []
        self.redo_stack.clear()
        self.update()

    def finish_current_lane(self):
        if self.drawing and len(self.curr) >= 3:
            self.lanes.append(self.curr.copy())
            self.polygonCommitted.emit()
        self.clear_draw()

    # ---------- Events ----------
    def mousePressEvent(self, ev: QtGui.QMouseEvent):
        if self.drawing:
            if ev.button() == QtCore.Qt.LeftButton:
                p = ev.pos()
                self.curr.append((p.x(), p.y()))
                self.pointAdded.emit((p.x(), p.y()))
                self.update()
                ev.accept()
                return
            elif ev.button() == QtCore.Qt.RightButton:
                self.finish_current_lane()
                ev.accept()
                return
        super().mousePressEvent(ev)

    def keyPressEvent(self, ev: QtGui.QKeyEvent):
        if not self.drawing:
            super().keyPressEvent(ev)
            return

        if ev.key() == QtCore.Qt.Key_Backspace:
            self.undo_point()
            ev.accept()
            return
        if ev.key() in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter):
            self.finish_current_lane()
            ev.accept()
            return
        if ev.key() == QtCore.Qt.Key_Escape:
            self.clear_draw()
            ev.accept()
            return

        super().keyPressEvent(ev)

    def paintEvent(self, ev):
        super().paintEvent(ev)
        if not self.drawing or not self.curr:
            return

        qp = QtGui.QPainter(self)
        qp.setRenderHint(QtGui.QPainter.Antialiasing, True)

        qp.setPen(QtCore.Qt.NoPen)
        qp.setBrush(QtGui.QBrush(QtGui.QColor(0, 255, 0)))
        for pt in self.curr:
            qp.drawEllipse(QtCore.QPoint(*pt), 5, 5)

        if len(self.curr) >= 2:
            pen2 = QtGui.QPen(QtGui.QColor(0, 200, 100))
            pen2.setWidth(2)
            pen2.setStyle(QtCore.Qt.DashLine)
            qp.setPen(pen2)
            qp.setBrush(QtCore.Qt.NoBrush)
            for j in range(1, len(self.curr)):
                qp.drawLine(
                    QtCore.QPoint(*self.curr[j - 1]),
                    QtCore.QPoint(*self.curr[j]),
                )


# =========================================================
# Worker: QThread đọc video + xử lý
# =========================================================


class Worker(QtCore.QThread):
    # vis, lane_metrics, events
    frameReady = QtCore.pyqtSignal(object, list, list)

    def __init__(self, proc: "TrafficProcessor", cap, label: VideoLabel):
        super().__init__()
        self.proc = proc
        self.cap = cap
        self.label = label
        self.running = True

        in_fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 30.0)
        self.frame_interval_ms = int(1000 / max(1e-3, in_fps))
        self.sync_to_input = bool(globals().get("SYNC_TO_INPUT_FPS", True))

    def run(self):
        timer = QtCore.QElapsedTimer()
        timer.start()
        last = timer.elapsed()

        while self.running:
            now = timer.elapsed()
            if self.sync_to_input and now - last < self.frame_interval_ms:
                QtCore.QThread.msleep(1)
                continue

            ok, frm = self.cap.read()
            if not ok:
                break

            try:
                label_w, label_h = self.label.width(), self.label.height()
                frame_w, frame_h = frm.shape[1], frm.shape[0]
                display_rect = compute_display_rect(
                    (label_w, label_h),
                    (frame_w, frame_h),
                )

                vis, metrics, events = self.proc.process_one(
                    frm,
                    (label_w, label_h),
                    display_rect=display_rect,
                )
            except Exception as e:
                logging.exception(e)
                vis, metrics, events = frm, [], []

            self.frameReady.emit(vis, metrics, events)
            last = timer.elapsed()

        try:
            if self.cap:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        except Exception:
            pass

    def stop(self):
        self.running = False


# =========================================================
# HanoiFlowApp
# =========================================================


class HanoiFlowApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("HanoiFlow — Quan trắc đa làn thời gian thực")
        self.resize(1400, 850)

        self.proc = None
        self.cap = None
        self.worker = None
        self.frame_idx = 0

        # Lưu vận tốc TB/làn (IN, OUT) lần gần nhất để giữ khi frame không có xe
        self.last_lane_speed = [0.0, 0.0]

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main = QtWidgets.QHBoxLayout(central)
        main.setContentsMargins(4, 4, 4, 4)
        main.setSpacing(8)

        # ============= LEFT: controls + video ============
        left = QtWidgets.QVBoxLayout()
        left.setSpacing(4)
        main.addLayout(left, 3)

        # --------- Thanh điều khiển 1 hàng ngang ----------
        controls = QtWidgets.QHBoxLayout()
        controls.setSpacing(8)
        left.addLayout(controls)

        # Nhóm 1: Video & model & Start/Stop/Lưu kết quả
        grp1 = QtWidgets.QFrame()
        g1 = QtWidgets.QHBoxLayout(grp1)
        g1.setContentsMargins(0, 0, 0, 0)
        g1.setSpacing(4)

        self.btn_open = QtWidgets.QPushButton("Chọn video")
        self.cmb = QtWidgets.QComboBox()
        self.cmb.addItems(list(YOLO_WEIGHTS.keys()))
        self.cmb.setCurrentText(YOLO_DEFAULT)
        self.cmb.setMaximumWidth(120)
        self.btn_start = QtWidgets.QPushButton("Start")
        self.btn_stop = QtWidgets.QPushButton("Stop")
        self.btn_save = QtWidgets.QPushButton("Lưu kết quả")

        for w in [self.btn_open, self.cmb, self.btn_start, self.btn_stop, self.btn_save]:
            g1.addWidget(w)

        # Nhóm 2: Vẽ làn & quản lý lanes & Test
        grp2 = QtWidgets.QFrame()
        g2 = QtWidgets.QHBoxLayout(grp2)
        g2.setContentsMargins(0, 0, 0, 0)
        g2.setSpacing(4)

        self.btn_draw = QtWidgets.QPushButton("Vẽ làn")
        self.btn_undo = QtWidgets.QPushButton("Hoàn tác điểm")
        self.btn_del = QtWidgets.QPushButton("Xóa lane cuối")
        self.btn_save_lanes = QtWidgets.QPushButton("Lưu lanes")
        self.btn_load_lanes = QtWidgets.QPushButton("Mở lanes")
        self.btn_test = QtWidgets.QPushButton("Test detect")

        for w in [
            self.btn_draw,
            self.btn_undo,
            self.btn_del,
            self.btn_save_lanes,
            self.btn_load_lanes,
            self.btn_test,
        ]:
            g2.addWidget(w)

        # Nhóm 3: Heatmap
        grp3 = QtWidgets.QFrame()
        g3 = QtWidgets.QHBoxLayout(grp3)
        g3.setContentsMargins(0, 0, 0, 0)
        g3.setSpacing(4)

        self.chk_hm = QtWidgets.QCheckBox("Heatmap")
        self.chk_hm.setChecked(USE_HEATMAP)
        self.btn_hm_reset = QtWidgets.QPushButton("Reset HM")
        self.btn_hm_snap = QtWidgets.QPushButton("Lưu ảnh HM")

        for w in [self.chk_hm, self.btn_hm_reset, self.btn_hm_snap]:
            g3.addWidget(w)

        # Thêm 3 nhóm vào cùng 1 hàng
        controls.addWidget(grp1)
        controls.addSpacing(10)
        controls.addWidget(grp2)
        controls.addSpacing(10)
        controls.addWidget(grp3)
        controls.addStretch(1)

        # --------- Vùng video ----------
        self.lbl = VideoLabel()
        left.addWidget(self.lbl, stretch=1)

        # ============= RIGHT: metrics ============
        right = QtWidgets.QVBoxLayout()
        right.setSpacing(6)
        main.addLayout(right, 2)

        grp_metrics = QtWidgets.QGroupBox("Lane Metrics")
        lay_metrics = QtWidgets.QVBoxLayout(grp_metrics)
        lay_metrics.setContentsMargins(6, 6, 6, 6)
        right.addWidget(grp_metrics, stretch=1)

        # --- Bảng sự kiện theo xe ---
        lbl_events = QtWidgets.QLabel("Bảng sự kiện theo xe")
        f_bold = lbl_events.font()
        f_bold.setBold(True)
        lbl_events.setFont(f_bold)
        lay_metrics.addWidget(lbl_events)

        # Vận tốc TB/xe toàn mạng
        self.lbl_avg_speed = QtWidgets.QLabel("Vận tốc TB/xe (toàn mạng): 0.0 km/h")
        lay_metrics.addWidget(self.lbl_avg_speed)

        self.tbl_events = QtWidgets.QTableWidget(0, 6)
        self.tbl_events.setHorizontalHeaderLabels(
            [
                "Làn",
                "ID",
                "Loại xe",
                "Tốc độ hiện tại (km/h)",
                "Tốc độ TB/xe (km/h)",
                "Thời gian",
            ]
        )
        self.tbl_events.verticalHeader().setVisible(False)
        self.tbl_events.setAlternatingRowColors(True)
        self.tbl_events.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)

        h_evt = self.tbl_events.horizontalHeader()
        h_evt.setStretchLastSection(False)
        h_evt.setSectionResizeMode(QtWidgets.QHeaderView.Fixed)
        self.tbl_events.setColumnWidth(0, 40)
        self.tbl_events.setColumnWidth(1, 50)
        self.tbl_events.setColumnWidth(2, 70)
        h_evt.setStyleSheet(
            "QHeaderView::section {"
            " border-bottom: 1px solid gray;"
            " border-top: 0px;"
            " border-left: 0px;"
            " border-right: 0px;"
            " padding: 2px; }"
        )

        lay_metrics.addWidget(self.tbl_events, stretch=2)

        # --- Bảng tổng hợp IN / OUT ---
        lbl_sum = QtWidgets.QLabel("Tổng hợp IN / OUT")
        lbl_sum.setFont(f_bold)
        lay_metrics.addWidget(lbl_sum)

        self.tbl_summary = QtWidgets.QTableWidget(2, 3)
        self.tbl_summary.setHorizontalHeaderLabels(
            [
                "Tổng số xe từng loại",
                "Vận tốc TB/làn (km/h)",
                "Tổng số xe/làn",
            ]
        )
        self.tbl_summary.setVerticalHeaderLabels(["IN", "OUT"])
        self.tbl_summary.setAlternatingRowColors(True)
        for r in range(2):
            self.tbl_summary.setRowHeight(r, 70)

        h_sum = self.tbl_summary.horizontalHeader()
        h_sum.setStretchLastSection(False)
        h_sum.setSectionResizeMode(QtWidgets.QHeaderView.Fixed)
        self.tbl_summary.setColumnWidth(0, 170)
        self.tbl_summary.setColumnWidth(1, 140)
        self.tbl_summary.setColumnWidth(2, 80)
        h_sum.setStyleSheet(
            "QHeaderView::section {"
            " border-bottom: 1px solid gray;"
            " border-top: 0px;"
            " border-left: 0px;"
            " border-right: 0px;"
            " padding: 2px; }"
        )

        self.tbl_summary.setMaximumHeight(180)
        lay_metrics.addWidget(self.tbl_summary, stretch=0)

        self.statusBar().showMessage("Sẵn sàng")

        # ---------- Signals ----------
        self.btn_open.clicked.connect(self.on_open)
        self.cmb.currentTextChanged.connect(self.on_change_model)
        self.btn_start.clicked.connect(self.on_start)
        self.btn_stop.clicked.connect(self.on_stop)

        self.btn_draw.clicked.connect(self.on_draw)
        self.btn_undo.clicked.connect(self.lbl.undo_point)
        self.btn_del.clicked.connect(self.on_del_lane)
        self.btn_save_lanes.clicked.connect(self.on_save_lanes)
        self.btn_load_lanes.clicked.connect(self.on_load_lanes)
        self.btn_save.clicked.connect(self.on_save_flush)
        self.btn_test.clicked.connect(self.on_test_detect)

        self.lbl.polygonCommitted.connect(self.on_lanes_changed)

        self.chk_hm.toggled.connect(self.on_toggle_heatmap)
        self.btn_hm_reset.clicked.connect(self.on_reset_heatmap)
        self.btn_hm_snap.clicked.connect(self.on_snapshot_heatmap)

    # =================================================
    # Helpers
    # =================================================
    def ensure_proc(self, model_key=None):
        if self.proc is not None:
            return True
        try:
            mk = model_key if model_key is not None else (
                self.cmb.currentText() if self.cmb else YOLO_DEFAULT
            )
            self.proc = TrafficProcessor(cam_id="cam0", model_key=mk)
            return True
        except Exception as e:
            import traceback
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(
                self,
                "HanoiFlow",
                f"Failed to create TrafficProcessor:\n{e}",
            )
            return False

    # =================================================
    # Video & model
    # =================================================
    def on_open(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Chọn video",
            DEFAULT_DATA_DIR,   # ===== SỬA THÀNH DEFAULT_DATA_DIR =====
            "Video (*.mp4 *.avi *.mkv)",
        )
        if not path:
            return
        if not self.ensure_proc(model_key=self.cmb.currentText()):
            return

        if self.worker:
            self.worker.stop()
            self.worker.wait(2000)
            self.worker = None

        try:
            if self.proc:
                self.proc.close()
        except Exception:
            pass

        try:
            self.proc.open_video(path)
            self.cap = self.proc.cap
            QtWidgets.QMessageBox.information(
                self,
                "HanoiFlow",
                f"Đã mở video: {Path(path).name}",
            )
            self.statusBar().showMessage(f"Video: {Path(path).name}")
        except RuntimeError as e:
            QtWidgets.QMessageBox.critical(
                self,
                "HanoiFlow",
                f"Không thể mở video:\n{e}",
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self,
                "HanoiFlow",
                f"Lỗi không xác định khi mở video:\n{e}",
            )

    def on_change_model(self, key: str):
        try:
            if not self.ensure_proc():
                return
            from hanoidemo.detection.yolo_utils import load_yolo

            new_model = load_yolo(key)
            if new_model is not None:
                self.proc.model = new_model
                self.proc.yolo_names = getattr(new_model, "names", {})
                logging.info(f"Model switched to: {key}")
                self.statusBar().showMessage(f"Đã chuyển model: {key}")
        except Exception as e:
            QtWidgets.QMessageBox.warning(
                self,
                "HanoiFlow",
                f"Không load model: {e}",
            )

    def on_start(self):
        if not self.cap:
            QtWidgets.QMessageBox.warning(
                self,
                "HanoiFlow",
                "Chưa mở video.",
            )
            return

        if self.worker:
            self.worker.stop()
            self.worker.wait(1500)
            self.worker = None

        try:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        except Exception:
            pass

        self.worker = Worker(self.proc, self.cap, self.lbl)
        self.worker.frameReady.connect(self.on_frame_ready)
        self.worker.start()
        self.statusBar().showMessage("Đang chạy...")

    def on_stop(self):
        if self.worker:
            self.worker.stop()
            self.worker.wait(2000)
            self.worker = None
        self.statusBar().showMessage("Đã dừng")

    # =================================================
    # Lanes
    # =================================================
    def on_draw(self):
        self.lbl.setFocus()
        self.lbl.start_draw()
        self.statusBar().showMessage(
            "Chế độ vẽ lane: click trái thêm điểm, phải / Enter kết thúc lane."
        )

    def on_del_lane(self):
        if self.lbl.lanes:
            self.lbl.lanes.pop()
            self.lbl.update()
            self.on_lanes_changed()

    def on_lanes_changed(self):
        if self.proc is None:
            return
        try:
            self.proc.set_lanes_label(self.lbl.lanes)
        except Exception as e:
            logging.exception(e)
            QtWidgets.QMessageBox.warning(
                self,
                "HanoiFlow",
                f"Lỗi khi cập nhật lanes:\n{e}",
            )

    def on_save_lanes(self):
        if not self.lbl.lanes:
            QtWidgets.QMessageBox.warning(
                self,
                "HanoiFlow",
                "Chưa có lanes để lưu!",
            )
            return

        name = f"lanes_{int(time.time())}.json"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Lưu lanes",
            str(LANES_DIR / name),
            "JSON (*.json)",
        )
        if not path:
            return

        success = save_lanes_json(self.lbl.lanes, Path(path))
        if success:
            QtWidgets.QMessageBox.information(
                self,
                "HanoiFlow",
                f"Đã lưu {len(self.lbl.lanes)} lanes vào:\n{Path(path).name}",
            )
        else:
            QtWidgets.QMessageBox.critical(
                self,
                "HanoiFlow",
                "Lỗi khi lưu lanes!",
            )

    def on_load_lanes(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Mở lanes",
            DEFAULT_DATA_DIR,   # ===== SỬA THÀNH DEFAULT_DATA_DIR =====
            "JSON (*.json)",
        )
        if not path:
            return

        lanes = load_lanes_json(Path(path))
        if lanes:
            self.lbl.lanes = lanes
            self.lbl.update()
            self.on_lanes_changed()
            QtWidgets.QMessageBox.information(
                self,
                "HanoiFlow",
                f"Đã tải {len(lanes)} lanes từ:\n{Path(path).name}",
            )
        else:
            QtWidgets.QMessageBox.warning(
                self,
                "HanoiFlow",
                "File lanes không hợp lệ hoặc rỗng.",
            )

    # =================================================
    # Heatmap & lưu kết quả
    # =================================================
    def on_save_flush(self):
        QtWidgets.QMessageBox.information(
            self,
            "HanoiFlow",
            "Đã flush CSV/video (placeholder).",
        )

    def on_toggle_heatmap(self, checked: bool):
        try:
            if not self.ensure_proc():
                return
            self.proc.hm_enabled = bool(checked)
            if self.proc.hm_enabled:
                setattr(self.proc, "hm_roi_mask", None)
                if self.proc.heatmap is None and self.proc.frame_size != (0, 0):
                    h, w = self.proc.frame_size[1], self.proc.frame_size[0]
                    self.proc.heatmap = HeatmapAccumulator(
                        (h, w),
                        decay=HEATMAP_DECAY,
                        blur_kernel=HEATMAP_BLUR_KERNEL,
                        colormap=HEATMAP_COLORMAP,
                    )
        except Exception as e:
            logging.error(f"Error toggling heatmap: {e}")

    def on_reset_heatmap(self):
        try:
            if getattr(self.proc, "heatmap", None) is not None:
                self.proc.heatmap.reset()
        except Exception as e:
            logging.error(f"Error resetting heatmap: {e}")

    def on_snapshot_heatmap(self):
        try:
            p = self.proc.save_heatmap_png("hm")
            if p:
                QtWidgets.QMessageBox.information(
                    self,
                    "HanoiFlow",
                    f"Đã lưu ảnh heatmap:\n{p}",
                )
        except Exception as e:
            logging.error(f"Error saving heatmap snapshot: {e}")

    # =================================================
    # Nhận frame + cập nhật bảng
    # =================================================
    def on_frame_ready(self, vis, lane_metrics, events):
        rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QtGui.QImage(
            rgb.data,
            w,
            h,
            bytes_per_line,
            QtGui.QImage.Format.Format_RGB888,
        )
        pix = QtGui.QPixmap.fromImage(qimg).scaled(
            self.lbl.size(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation,
        )
        self.lbl.setPixmap(pix)

        self.append_events(events)
        self.update_summary_table(lane_metrics)

    def append_events(self, events):
        """
        Bảng trên: mỗi ID xe 1 dòng.
        - Cột 3: tốc độ hiện tại (km/h).
        - Cột 4: tốc độ TB/xe (km/h) từ đầu tới giờ.
        Khi (Làn, ID) đã tồn tại: chỉ cập nhật lại dòng.
        """
        if not events:
            return

        for ev in events:
            lane_idx = ev.get("lane", -1)
            if lane_idx == 0:
                lane_name = "IN"
            elif lane_idx == 1:
                lane_name = "OUT"
            else:
                lane_name = str(lane_idx)

            id_str = str(ev.get("id", ""))
            type_str = ev.get("type", "")
            speed_cur = f"{float(ev.get('speed', 0.0)):.1f}"
            speed_avg = f"{float(ev.get('avg_speed', 0.0)):.1f}"
            time_str = ev.get("time", "")

            found_row = -1
            for row in range(self.tbl_events.rowCount() - 1, -1, -1):
                it_lane = self.tbl_events.item(row, 0)
                it_id = self.tbl_events.item(row, 1)
                if it_lane and it_id:
                    if it_lane.text() == lane_name and it_id.text() == id_str:
                        found_row = row
                        break

            if found_row == -1:
                row = self.tbl_events.rowCount()
                self.tbl_events.insertRow(row)
            else:
                row = found_row

            self.tbl_events.setItem(row, 0, QtWidgets.QTableWidgetItem(lane_name))
            self.tbl_events.setItem(row, 1, QtWidgets.QTableWidgetItem(id_str))
            self.tbl_events.setItem(row, 2, QtWidgets.QTableWidgetItem(type_str))
            self.tbl_events.setItem(row, 3, QtWidgets.QTableWidgetItem(speed_cur))
            self.tbl_events.setItem(row, 4, QtWidgets.QTableWidgetItem(speed_avg))
            self.tbl_events.setItem(row, 5, QtWidgets.QTableWidgetItem(time_str))

        self.tbl_events.scrollToBottom()

    def update_summary_table(self, lane_metrics):
        """
        Bảng IN/OUT:
        - Hàng 0: lane 0 -> IN
        - Hàng 1: lane 1 -> OUT
        - Vận tốc TB/xe toàn mạng hiện ở self.lbl_avg_speed.
        - Cột Vận tốc TB/làn: dùng speed tức thời nhưng giữ
          giá trị cuối cùng khi lane trống.
        """
        global_avg = 0.0
        if self.proc is not None:
            try:
                if self.proc.global_vehicle_total > 0:
                    global_avg = (
                        self.proc.global_speed_sum / self.proc.global_vehicle_total
                    )
            except Exception:
                global_avg = 0.0
        self.lbl_avg_speed.setText(
            f"Vận tốc TB/xe (toàn mạng): {global_avg:.1f} km/h"
        )

        if not lane_metrics:
            return

        lane0 = lane_metrics[0] if len(lane_metrics) >= 1 else None
        lane1 = lane_metrics[1] if len(lane_metrics) >= 2 else None

        def set_row(row_idx: int, lane_dict: dict | None):
            if not lane_dict:
                cc = {}
                bike = car = truck = bus = 0
            else:
                cc = lane_dict.get("class_counts", {}) or {}
                bike = int(cc.get("Bike", 0))
                car = int(cc.get("Car", 0))
                truck = int(cc.get("Truck", 0))
                bus = int(cc.get("Bus", 0))

                dens = int(lane_dict.get("density", 0))
                v_inst = float(lane_dict.get("speed", 0.0))
                if dens > 0 and v_inst > 0.0 and 0 <= row_idx < len(self.last_lane_speed):
                    self.last_lane_speed[row_idx] = v_inst

            text_classes = f"Bike: {bike}\nCar: {car}\nTruck: {truck}\nBus: {bus}"
            self.tbl_summary.setItem(
                row_idx, 0, QtWidgets.QTableWidgetItem(text_classes)
            )

            v_display = (
                self.last_lane_speed[row_idx]
                if 0 <= row_idx < len(self.last_lane_speed)
                else 0.0
            )
            self.tbl_summary.setItem(
                row_idx, 1, QtWidgets.QTableWidgetItem(f"{v_display:.1f}")
            )

            veh_total = int(lane_dict.get("vehicle_total", 0)) if lane_dict else 0
            self.tbl_summary.setItem(
                row_idx, 2, QtWidgets.QTableWidgetItem(str(veh_total))
            )

        set_row(0, lane0)
        set_row(1, lane1)

    # =================================================
    # Tự động căn lại độ rộng cột khi resize
    # =================================================
    def resizeEvent(self, ev: QtGui.QResizeEvent):
        super().resizeEvent(ev)
        try:
            if self.tbl_events.columnCount() == 6:
                total = self.tbl_events.viewport().width()
                fixed = 40 + 50 + 70
                remain = max(0, total - fixed)
                w3 = int(remain * 0.33)
                w4 = int(remain * 0.33)
                w5 = remain - w3 - w4
                self.tbl_events.setColumnWidth(0, 40)
                self.tbl_events.setColumnWidth(1, 50)
                self.tbl_events.setColumnWidth(2, 70)
                self.tbl_events.setColumnWidth(3, w3)
                self.tbl_events.setColumnWidth(4, w4)
                self.tbl_events.setColumnWidth(5, w5)

            if self.tbl_summary.columnCount() == 3:
                total2 = self.tbl_summary.viewport().width()
                col0 = int(total2 * 0.4)
                col1 = int(total2 * 0.4)
                col2 = total2 - col0 - col1
                self.tbl_summary.setColumnWidth(0, col0)
                self.tbl_summary.setColumnWidth(1, col1)
                self.tbl_summary.setColumnWidth(2, col2)
        except Exception:
            pass

    # =================================================
    # Test detect
    # =================================================
    def on_test_detect(self):
        if not self.cap:
            QtWidgets.QMessageBox.warning(
                self,
                "HanoiFlow",
                "Chưa mở video.",
            )
            return

        pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        ok, frm = self.cap.read()
        if not ok:
            QtWidgets.QMessageBox.warning(
                self,
                "HanoiFlow",
                "Không đọc được frame.",
            )
            return

        try:
            res = self.proc.model.predict(
                source=frm,
                imgsz=TEST_IMGSZ,
                conf=TEST_CONF,
                iou=IOU,
                classes=None,
                device=DEVICE,
                verbose=False,
            )
            r = res[0]
            vis = r.plot()
        except Exception as e:
            QtWidgets.QMessageBox.warning(
                self,
                "HanoiFlow",
                f"Test detect fail: {e}",
            )
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            return

        label_w, label_h = self.lbl.width(), self.lbl.height()
        frame_w, frame_h = frm.shape[1], frm.shape[0]
        display_rect = compute_display_rect(
            (label_w, label_h),
            (frame_w, frame_h),
        )
        mapped = map_lanes_label_to_frame(
            self.lbl.lanes,
            (label_w, label_h),
            (frame_w, frame_h),
            display_rect=display_rect,
        )
        draw_lanes(vis, mapped)

        rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QtGui.QImage(
            rgb.data,
            w,
            h,
            bytes_per_line,
            QtGui.QImage.Format.Format_RGB888,
        )
        pix = QtGui.QPixmap.fromImage(qimg).scaled(
            self.lbl.size(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation,
        )
        self.lbl.setPixmap(pix)

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, pos)

    # =================================================
    # Đóng app
    # =================================================
    def closeEvent(self, ev: QtGui.QCloseEvent):
        try:
            if self.worker:
                self.worker.stop()
                self.worker.wait(2000)
        except Exception:
            pass
        try:
            if self.proc:
                self.proc.close()
        except Exception:
            pass
        ev.accept()
