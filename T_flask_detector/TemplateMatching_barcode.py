import sys
import cv2
import numpy as np
from datetime import datetime
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QSlider,
    QProgressBar,
    QGroupBox,
    QFrame,
    QScrollArea,
    QTextEdit,
)
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal

# --- [바코드 라이브러리 추가] ---
try:
    from pyzbar import pyzbar
except ImportError:
    print("pyzbar가 설치되지 않았습니다. 'pip install pyzbar'를 실행해주세요.")
    sys.exit()

# --- [스타일 시트] ---
STYLESHEET = """
    QMainWindow { background-color: #1e1e1e; }
    QLabel { color: #ffffff; font-family: 'Segoe UI', sans-serif; }
    QGroupBox { 
        color: #aaaaaa; font-weight: bold; border: 1px solid #444; 
        border-radius: 6px; margin-top: 10px; padding-top: 15px; 
    }
    QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
    QPushButton {
        background-color: #3a3a3a; color: white; border: 1px solid #555;
        border-radius: 5px; padding: 10px; font-size: 14px;
    }
    QPushButton:hover { background-color: #505050; border: 1px solid #777; }
    QPushButton:pressed { background-color: #2a2a2a; }
    QProgressBar {
        border: 2px solid #555; border-radius: 5px; text-align: center; color: white; background-color: #222;
    }
    QProgressBar::chunk { background-color: #4CAF50; width: 10px; margin: 0.5px; }
    QScrollArea { border: none; background-color: transparent; }
    QWidget#ResultContainer { background-color: transparent; }
    QTextEdit {
        background-color: #222; color: #00ff00; border: 1px solid #444; font-family: Consolas; font-size: 12px;
    }
"""


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    update_multi_score_signal = pyqtSignal(list)
    update_barcode_signal = pyqtSignal(str)
    status_signal = pyqtSignal(str, bool)

    def __init__(self):
        super().__init__()
        self.running = True
        
        # [라즈베리파이 수정] V4L2 백엔드를 명시적으로 사용하여 카메라 호환성 확보
        # 일반적으로 라즈베리파이 카메라는 인덱스 0번에 매핑됩니다.
        # 만약 화면이 안 나온다면 0을 -1 또는 1, 2로 변경해보세요.
        self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        
        # 해상도 설정 (라즈베리파이 성능 고려하여 적절히 유지)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # FPS 설정 (선택 사항: 필요 시 주석 해제하여 30fps로 제한)
        # self.cap.set(cv2.CAP_PROP_FPS, 30)

        self.templates = []
        self.search_roi = None
        self.threshold = 0.7
        self.mode = "IDLE"

        self.req_register_template = False
        self.req_set_search_roi = False
        self.last_barcode_data = ""

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("카메라 프레임을 읽을 수 없습니다.")
                # 카메라 연결이 끊어지면 잠시 대기 후 재시도
                self.msleep(100)
                continue

            # [라즈베리파이 수정] 카메라가 거꾸로 설치된 경우 주석을 해제하세요.
            # 0: 상하 반전, 1: 좌우 반전, -1: 상하좌우 반전(180도 회전)
            # frame = cv2.flip(frame, -1)

            display_frame = frame.copy()
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 이벤트 처리 (등록, ROI 설정 등)
            if self.req_register_template:
                # GUI 쓰레드 충돌 방지를 위해 메인 루프 밖 처리가 이상적이나 간단한 구현을 위해 유지
                # 라즈베리파이에서는 selectROI 창이 전체화면 뒤로 갈 수 있으므로 주의
                self.register_template_func(frame)
                self.req_register_template = False

            if self.req_set_search_roi:
                self.set_search_roi_func(frame)
                self.req_set_search_roi = False

            # ROI 설정
            sx, sy = 0, 0
            if self.search_roi is not None:
                sx, sy, sw, sh = self.search_roi
                cv2.rectangle(
                    display_frame, (sx, sy), (sx + sw, sy + sh), (255, 165, 0), 2
                )
                cv2.putText(
                    display_frame, "Search Area", (sx, sy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2,
                )
                if sw > 0 and sh > 0:
                    search_img = gray_frame[sy : sy + sh, sx : sx + sw]
                else:
                    search_img = gray_frame
                    sx, sy = 0, 0
            else:
                search_img = gray_frame

            # --- 바코드 인식 파이프라인 ---
            kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
            search_img_sharp = cv2.filter2D(search_img, -1, kernel)
            _, search_img_binary = cv2.threshold(search_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            current_scale = 1.0
            
            # 시도 1: 원본
            barcodes = pyzbar.decode(search_img)
            # 시도 2: 샤프닝
            if not barcodes:
                barcodes = pyzbar.decode(search_img_sharp)
            # 시도 3: 이진화
            if not barcodes:
                barcodes = pyzbar.decode(search_img_binary)
            # 시도 4: 확대 (CM5 성능이 좋으므로 2배 확대도 무리 없음)
            if not barcodes:
                current_scale = 2.0
                scaled_img = cv2.resize(search_img, None, fx=current_scale, fy=current_scale, interpolation=cv2.INTER_LINEAR)
                barcodes = pyzbar.decode(scaled_img)

            for barcode in barcodes:
                (bx, by, bw, bh) = barcode.rect
                if current_scale > 1.0:
                    bx = int(bx / current_scale)
                    by = int(by / current_scale)
                    bw = int(bw / current_scale)
                    bh = int(bh / current_scale)

                final_bx = bx + sx
                final_by = by + sy

                barcode_data = barcode.data.decode("utf-8")
                barcode_type = barcode.type

                cv2.rectangle(display_frame, (final_bx, final_by), (final_bx + bw, final_by + bh), (255, 0, 0), 2)
                text = f"{barcode_data} ({barcode_type})"
                cv2.putText(display_frame, text, (final_bx, final_by - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                if barcode_data != self.last_barcode_data:
                    self.update_barcode_signal.emit(f"[{barcode_type}] {barcode_data}")
                    self.last_barcode_data = barcode_data

            # --- 객체 매칭 로직 ---
            results = []
            if self.mode == "MATCHING" and len(self.templates) > 0:
                for idx, tmpl in enumerate(self.templates):
                    best_match = self.detect_multiscale(search_img, tmpl["img"])
                    if best_match:
                        max_val, max_loc, t_w, t_h = best_match
                        if max_val >= self.threshold:
                            final_x = max_loc[0] + sx
                            final_y = max_loc[1] + sy
                            cv2.rectangle(display_frame, (final_x, final_y), (final_x + t_w, final_y + t_h), (0, 255, 0), 2)
                            cv2.putText(display_frame, f"ID:{idx+1}", (final_x, final_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            results.append({"id": idx + 1, "score": max_val, "found": True})
                        else:
                            results.append({"id": idx + 1, "score": max_val, "found": False})
                    else:
                        results.append({"id": idx + 1, "score": 0.0, "found": False})

            self.update_multi_score_signal.emit(results)
            self.change_pixmap_signal.emit(display_frame)

    def detect_multiscale(self, image, template):
        best_score = -1
        best_loc = None
        best_size = (0, 0)
        t_h, t_w = template.shape[:2]

        # 라즈베리파이에서는 연산량을 줄이기 위해 스케일 단계를 줄이거나 범위를 좁히는 것도 고려 가능
        for scale in np.linspace(0.8, 1.2, 5):
            resized_w = int(t_w * scale)
            resized_h = int(t_h * scale)

            if resized_h > image.shape[0] or resized_w > image.shape[1]:
                continue

            resized_tmpl = cv2.resize(template, (resized_w, resized_h))

            try:
                res = cv2.matchTemplate(image, resized_tmpl, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                if max_val > best_score:
                    best_score = max_val
                    best_loc = max_loc
                    best_size = (resized_w, resized_h)
            except Exception:
                continue

        if best_score != -1:
            return best_score, best_loc, best_size[0], best_size[1]
        return None

    def register_template_func(self, frame):
        # 라즈베리파이에서 전체화면 이슈 방지를 위해 윈도우 속성 일부 제거
        win_name = "Add Object"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL) 
        # cv2.setWindowProperty(win_name, cv2.WND_PROP_TOPMOST, 1) # 라즈베리파이에서는 이 설정이 가끔 충돌날 수 있음
        
        roi = cv2.selectROI(win_name, frame, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow(win_name)

        if roi != (0, 0, 0, 0):
            x, y, w, h = roi
            roi_img = frame[y : y + h, x : x + w]
            gray_tmpl = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
            self.templates.append({"img": gray_tmpl, "w": w, "h": h})
            self.mode = "MATCHING"
            self.status_signal.emit(f"객체 #{len(self.templates)} 등록 완료", True)
        else:
            self.status_signal.emit("등록 취소됨", False)

    def set_search_roi_func(self, frame):
        win_name = "Set Search Area"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        # cv2.setWindowProperty(win_name, cv2.WND_PROP_TOPMOST, 1) 
        
        roi = cv2.selectROI(win_name, frame, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow(win_name)

        if roi != (0, 0, 0, 0):
            self.search_roi = roi
            self.status_signal.emit("검색 영역 설정됨", True)
        else:
            self.status_signal.emit("영역 설정 취소됨", False)

    def reset_all(self):
        self.templates = []
        self.search_roi = None
        self.mode = "IDLE"
        self.last_barcode_data = ""
        self.update_multi_score_signal.emit([])
        self.status_signal.emit("전체 초기화 완료", False)

    def stop(self):
        self.running = False
        self.wait()
        self.cap.release()


class App(QMainWindow):
    def __init__(self):
        super().__init__()
        # 라즈베리파이 화면 해상도에 맞춰 윈도우 크기 조정
        self.setWindowTitle("RPi Multi-Object & Barcode Dashboard")
        self.setGeometry(0, 0, 1024, 600)  # 터치스크린 등 작은 화면 대응
        self.setStyleSheet(STYLESHEET)

        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.update_multi_score_signal.connect(self.update_result_ui)
        self.thread.status_signal.connect(self.update_status_led)
        self.thread.update_barcode_signal.connect(self.append_barcode_log)
        self.thread.start()

        self.result_widgets = {}
        self.initUI()

    def initUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)

        # 1. 왼쪽: 카메라 화면
        video_container = QWidget()
        video_layout = QVBoxLayout()
        video_container.setLayout(video_layout)

        self.image_label = QLabel("Camera Feed Loading...")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet(
            "background-color: #000; border: 2px solid #333; border-radius: 10px;"
        )
        self.image_label.setMinimumSize(640, 480)
        video_layout.addWidget(self.image_label)
        main_layout.addWidget(video_container, stretch=3)

        # 2. 오른쪽: 제어 패널
        control_panel = QFrame()
        control_panel.setStyleSheet("background-color: #2b2b2b; border-radius: 10px;")
        control_layout = QVBoxLayout()
        control_panel.setLayout(control_layout)

        title = QLabel("🤖 RPi Detector")
        title.setFont(QFont("Arial", 16, QFont.Bold)) # 폰트 사이즈 살짝 조정
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #4CAF50; margin-bottom: 5px;")
        control_layout.addWidget(title)

        # --- [A] 객체 모니터링 영역 ---
        monitor_group = QGroupBox("📊 STATUS")
        monitor_layout = QVBoxLayout()
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.result_container = QWidget()
        self.result_container.setObjectName("ResultContainer")
        self.result_layout = QVBoxLayout()
        self.result_layout.setAlignment(Qt.AlignTop)
        self.result_container.setLayout(self.result_layout)
        self.scroll_area.setWidget(self.result_container)
        monitor_layout.addWidget(self.scroll_area)

        self.empty_label = QLabel("No Objects")
        self.empty_label.setAlignment(Qt.AlignCenter)
        self.empty_label.setStyleSheet("color: #777; font-style: italic;")
        self.result_layout.addWidget(self.empty_label)

        monitor_group.setLayout(monitor_layout)
        control_layout.addWidget(monitor_group, stretch=2)

        # --- [NEW] 바코드 로그 영역 ---
        barcode_group = QGroupBox("📅 LOG")
        barcode_layout = QVBoxLayout()
        self.barcode_log = QTextEdit()
        self.barcode_log.setReadOnly(True)
        self.barcode_log.setPlaceholderText("Scanning...")
        barcode_layout.addWidget(self.barcode_log)
        barcode_group.setLayout(barcode_layout)
        control_layout.addWidget(barcode_group, stretch=1)

        # --- [B] 설정 그룹 ---
        setting_group = QGroupBox("⚙️ SENSITIVITY")
        setting_layout = QVBoxLayout()
        self.thresh_label = QLabel("Threshold: 70%")
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(1, 99)
        self.slider.setValue(70)
        self.slider.valueChanged.connect(self.change_threshold)
        setting_layout.addWidget(self.thresh_label)
        setting_layout.addWidget(self.slider)
        setting_group.setLayout(setting_layout)
        control_layout.addWidget(setting_group)

        # --- [C] 작업 버튼 ---
        action_group = QGroupBox("🛠️ ACTION")
        action_layout = QVBoxLayout()

        step1_layout = QHBoxLayout()
        self.led_template = QLabel("0")
        self.led_template.setStyleSheet(
            "background-color: #444; color: white; border-radius: 10px; padding: 5px 10px;"
        )
        self.btn_template = QPushButton("1. Add Object")
        self.btn_template.setStyleSheet("background-color: #2196F3; font-weight: bold;")
        self.btn_template.clicked.connect(
            lambda: setattr(self.thread, "req_register_template", True)
        )
        step1_layout.addWidget(self.led_template)
        step1_layout.addWidget(self.btn_template)

        step2_layout = QHBoxLayout()
        self.led_roi = QLabel("❌")
        self.btn_roi = QPushButton("2. Set Area")
        self.btn_roi.setStyleSheet("background-color: #FF9800; font-weight: bold;")
        self.btn_roi.clicked.connect(
            lambda: setattr(self.thread, "req_set_search_roi", True)
        )
        step2_layout.addWidget(self.led_roi)
        step2_layout.addWidget(self.btn_roi)

        action_layout.addLayout(step1_layout)
        action_layout.addLayout(step2_layout)
        action_group.setLayout(action_layout)
        control_layout.addWidget(action_group)

        # --- [D] 시스템 버튼 ---
        system_layout = QHBoxLayout()
        self.btn_reset = QPushButton("RESET")
        self.btn_reset.setStyleSheet("background-color: #607D8B;")
        self.btn_reset.clicked.connect(self.reset_app)
        self.btn_quit = QPushButton("EXIT")
        self.btn_quit.setStyleSheet("background-color: #f44336;")
        self.btn_quit.clicked.connect(self.close)
        system_layout.addWidget(self.btn_reset)
        system_layout.addWidget(self.btn_quit)
        control_layout.addLayout(system_layout)

        main_layout.addWidget(control_panel, stretch=1)

    def update_result_ui(self, results):
        if not results:
            for i in reversed(range(self.result_layout.count())):
                widget = self.result_layout.itemAt(i).widget()
                if widget is not None and widget != self.empty_label:
                    widget.setParent(None)
            self.empty_label.show()
            self.result_widgets = {}
            return

        self.empty_label.hide()

        for res in results:
            obj_id = res["id"]
            score = int(res["score"] * 100)
            is_found = res["found"]

            if obj_id not in self.result_widgets:
                self.create_result_widget(obj_id)

            label, pbar = self.result_widgets[obj_id]
            status_text = "DETECTED" if is_found else "MISSING"
            color_text = "#4CAF50" if is_found else "#F44336"
            label.setText(
                f"Obj #{obj_id}: <span style='color:{color_text}'>{status_text}</span>"
            )
            pbar.setValue(score)
            if is_found:
                pbar.setStyleSheet("QProgressBar::chunk { background-color: #4CAF50; }")
            else:
                pbar.setStyleSheet("QProgressBar::chunk { background-color: #555; }")

    def create_result_widget(self, obj_id):
        container = QWidget()
        container.setStyleSheet(
            "background-color: #333; border-radius: 5px; margin-bottom: 5px;"
        )
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 5, 10, 5)
        lbl = QLabel(f"Obj #{obj_id}")
        lbl.setFont(QFont("Arial", 11, QFont.Bold))
        pbar = QProgressBar()
        pbar.setRange(0, 100)
        pbar.setTextVisible(True)
        pbar.setFixedHeight(15)
        layout.addWidget(lbl)
        layout.addWidget(pbar)
        container.setLayout(layout)
        self.result_layout.addWidget(container)
        self.result_widgets[obj_id] = (lbl, pbar)

    def append_barcode_log(self, text):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.barcode_log.append(f"[{timestamp}] {text}")
        self.barcode_log.moveCursor(self.barcode_log.textCursor().End)

    def update_image(self, cv_img):
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_img.shape
        qt_img = QImage(rgb_img.data, w, h, ch * w, QImage.Format_RGB888)
        self.image_label.setPixmap(
            QPixmap.fromImage(qt_img).scaled(
                self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio
            )
        )

    def change_threshold(self):
        val = self.slider.value()
        self.thread.threshold = val / 100.0
        self.thresh_label.setText(f"Threshold: {val}%")

    def update_status_led(self, msg, is_success):
        if "객체" in msg:
            count = len(self.thread.templates)
            self.led_template.setText(str(count))
            self.led_template.setStyleSheet(
                "background-color: #4CAF50; color: white; border-radius: 10px; padding: 5px 10px;"
            )
        elif "검색 영역" in msg and is_success:
            self.led_roi.setText("✅")
        elif "초기화" in msg:
            self.led_template.setText("0")
            self.led_template.setStyleSheet(
                "background-color: #444; color: white; border-radius: 10px; padding: 5px 10px;"
            )
            self.led_roi.setText("❌")
            self.barcode_log.clear()

        print(f"Log: {msg}")

    def reset_app(self):
        self.thread.reset_all()

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = App()
    window.show()
    sys.exit(app.exec_())