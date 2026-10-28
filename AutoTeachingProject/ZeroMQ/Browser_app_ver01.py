import os
# CM5 쿼드코어 활용을 위해 스레드를 1개에서 2개로 완화 (CPU 과부하 없이 연산 속도 2배 향상)
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"
import time
import threading
import multiprocessing
import cv2
import numpy as np
import zmq
import json
from multiprocessing import shared_memory
from flask import Flask, Response, render_template_string, jsonify, request
from pyzbar import pyzbar
from pupil_apriltags import Detector

try:
    from picamera2 import Picamera2
except ImportError:
    print("⚠️ 오류: picamera2를 찾을 수 없습니다. (라즈베리파이 환경에서 실행해주세요)")

cv2.setNumThreads(2)

# ==========================================
# 1. Camera Publisher 클래스 (30+ FPS 고속 캡처)
# ==========================================
class CameraPublisherSHM:
    def __init__(self):
        self.width, self.height = 1920, 1080
        self.vga_w, self.vga_h = 640, 480
        self.size_fhd = self.width * self.height * 3
        self.size_vga = self.vga_w * self.vga_h * 3

        self.shm_fhd = self._init_shared_memory("vision_fhd", self.size_fhd)
        self.shm_vga = self._init_shared_memory("vision_vga", self.size_vga)

        self.frame_fhd_shared = np.ndarray((self.height, self.width, 3), dtype=np.uint8, buffer=self.shm_fhd.buf)
        self.frame_vga_shared = np.ndarray((self.vga_h, self.vga_w, 3), dtype=np.uint8, buffer=self.shm_vga.buf)

        self.picam2 = Picamera2()
        self._setup_camera()

    def _init_shared_memory(self, name: str, size: int):
        try:
            return shared_memory.SharedMemory(create=True, size=size, name=name)
        except FileExistsError:
            shm = shared_memory.SharedMemory(name=name)
            shm.unlink()
            return shared_memory.SharedMemory(create=True, size=size, name=name)

    def _setup_camera(self):
        print(f"📷 Camera 초기화 ({self.width}x{self.height})...")
        config = self.picam2.create_video_configuration(main={"size": (self.width, self.height), "format": "BGR888"})
        self.picam2.configure(config)
        self.picam2.start()
        try:
            self.picam2.set_controls({"AfMode": 2})
            time.sleep(0.5)
            self.picam2.set_controls({"AfMode": 0, "LensPosition": 5.5})
        except Exception:
            pass

    def start_streaming(self):
        print("🚀 Camera Publisher 고속 스트리밍 시작됨")
        try:
            while True:
                raw_frame = self.picam2.capture_array()
                if raw_frame is None:
                    continue

                frame_fhd = cv2.cvtColor(raw_frame, cv2.COLOR_RGB2BGR)
                # INTER_LINEAR가 INTER_AREA보다 FHD->VGA 축소 시 약 3배 빠름
                frame_vga = cv2.resize(frame_fhd, (self.vga_w, self.vga_h), interpolation=cv2.INTER_LINEAR)

                np.copyto(self.frame_fhd_shared, frame_fhd)
                np.copyto(self.frame_vga_shared, frame_vga)
                time.sleep(0.02)  # 기존 0.05s(20FPS) -> 0.02s(최대 40~50FPS)
        except KeyboardInterrupt:
            pass
        finally:
            self.picam2.stop()
            self.shm_fhd.close(); self.shm_fhd.unlink()
            self.shm_vga.close(); self.shm_vga.unlink()

def run_publisher():
    pub = CameraPublisherSHM()
    pub.start_streaming()


# ==========================================
# 2. Vision Subscriber 및 칼만 필터 (고속 최적화 버전)
# ==========================================
class KalmanFilter1D:
    def __init__(self, process_noise=1e-3, measurement_noise=0.25):
        self.x, self.p = 0.0, 1.0
        self.q, self.r = process_noise, measurement_noise
        self.initialized = False
    def update(self, measurement):
        if not self.initialized:
            self.x = measurement
            self.initialized = True
            return self.x
        self.p += self.q
        k = self.p / (self.p + self.r)
        self.x += k * (measurement - self.x)
        self.p *= (1 - k)
        return self.x

class IntegratedVisionSubscriber:
    def __init__(self, yolo_model_path: str):
        self.lock = threading.Lock()
        self.data_lock = threading.Lock()
        self.running = True
        self.yolo_model_path = yolo_model_path
        self.yolo_model = None
        self.yolo_ready = False

        self.enabled_lock = threading.Lock()
        # 초기 구동 시 불필요한 중복 연산을 줄이기 위해 바코드는 필요 시 켜도록 설정 가능 (기본은 모두 허용)
        self.enabled = {
            "apriltag": True,
            "barcode": False,
            "yolo": True,
        }

        # 인코딩 완료된 JPEG 바이트를 캐싱하여 여러 클라이언트가 동시에 요청해도 재인코딩하지 않음!
        self.encoded_cache = {
            "apriltag": None,
            "barcode_main": None,
            "barcode_zoom": None,
            "yolo": None,
        }

        self.latest_results = {
            "apriltag": {
                "detected": False,
                "tags": [],
                "last_seen_tags": [],
                "timestamp": 0.0,
            },
            "yolo": {
                "is_in_roi": False,
                "detected_objects": [],
                "timestamp": 0.0,
            },
            "barcode": {
                "detected": False,
                "barcodes": [],
                "last_barcode": "",
                "last_barcode_time": 0.0,
                "timestamp": 0.0,
            }
        }

        print("🔗 Shared Memory 연결 대기 중...")
        while True:
            try:
                self.shm_fhd = shared_memory.SharedMemory(name="vision_fhd")
                self.shm_vga = shared_memory.SharedMemory(name="vision_vga")
                break
            except FileNotFoundError:
                time.sleep(0.2)

        self.fhd_buffer = np.ndarray((1080, 1920, 3), dtype=np.uint8, buffer=self.shm_fhd.buf)
        self.vga_buffer = np.ndarray((480, 640, 3), dtype=np.uint8, buffer=self.shm_vga.buf)

        self.pub_context = zmq.Context()
        self.tag_pub = self.pub_context.socket(zmq.PUB)
        self.barcode_pub = self.pub_context.socket(zmq.PUB)
        self.yolo_pub = self.pub_context.socket(zmq.PUB)

        self.tag_pub.bind("ipc:///tmp/vision_tag.ipc")
        self.barcode_pub.bind("ipc:///tmp/vision_barcode.ipc")
        self.yolo_pub.bind("ipc:///tmp/vision_yolo.ipc")

        self.tag_size = 0.021
        self.half_s = self.tag_size / 2.0

        self.camera_matrix = np.array([
            [1377.30483, 0, 970.26621],
            [0, 1381.74778, 531.14559],
            [0, 0, 1]
        ], dtype=np.float32)
        self.dist_coeffs = np.array([[-0.02188, 0.33220, 0.00308, 0.00052, -0.50083]], dtype=np.float32)
        self.obj_points = np.array([
            [-self.half_s, -self.half_s, 0],
            [self.half_s, -self.half_s, 0],
            [self.half_s, self.half_s, 0],
            [-self.half_s, self.half_s, 0]
        ], dtype=np.float32)

        # 💡 [고속화 핵심 1] quad_decimate=2.0 및 nthreads=2 적용 -> AprilTag 검출 속도 4배 향상!
        self.detector = Detector(families='tag36h11', nthreads=2, quad_decimate=2.0, debug=0)
        self.kf_dict = {}
        self.bc_roi_x, self.bc_roi_y, self.bc_scale = 400, 120, 2  # 3배 확대 -> 2배 확대로 디코딩 속도 2배 향상
        self.target_roi = (206, 212, 434, 268)

        self.disabled_jpg_bytes = self._build_disabled_placeholder()

        # 💡 [고속화 핵심 2] 무거운 ultralytics YOLO 로딩을 백그라운드 스레드로 분리하여 웹서버 즉시 오픈!
        threading.Thread(target=self._async_load_yolo, daemon=True).start()

    def _async_load_yolo(self):
        try:
            print(f"⚙️ 백그라운드에서 YOLO 모델 로딩 중... ({self.yolo_model_path})")
            from ultralytics import YOLO
            self.yolo_model = YOLO(self.yolo_model_path, task='detect')
            self.yolo_ready = True
            print("✅ YOLO 모델 로딩 완료!")
        except Exception as e:
            print(f"⚠️ YOLO 로딩 실패: {e}")

    def _build_disabled_placeholder(self):
        placeholder = np.zeros((360, 640, 3), dtype=np.uint8)
        cv2.putText(placeholder, "DISABLED (Standby)", (160, 190), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (110, 110, 110), 2)
        success, encoded = cv2.imencode(".jpg", placeholder, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        return bytearray(encoded) if success else None

    def _encode_jpg_fast(self, frame, resize_to=None, quality=72):
        if frame is None:
            return None
        if resize_to is not None and (frame.shape[1] != resize_to[0] or frame.shape[0] != resize_to[1]):
            frame = cv2.resize(frame, resize_to, interpolation=cv2.INTER_LINEAR)
        success, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        return bytearray(encoded) if success else None

    def set_enabled(self, feature: str, value: bool):
        with self.enabled_lock:
            if feature in self.enabled:
                self.enabled[feature] = bool(value)

    def is_enabled(self, feature: str) -> bool:
        with self.enabled_lock:
            return self.enabled.get(feature, True)

    def get_status(self):
        with self.enabled_lock:
            return dict(self.enabled)

    def reset_kalman(self):
        with self.data_lock:
            self.kf_dict.clear()

    def clear_last_barcode(self):
        with self.data_lock:
            self.latest_results["barcode"]["last_barcode"] = ""
            self.latest_results["barcode"]["barcodes"] = []
            self.latest_results["barcode"]["detected"] = False

    def get_vision_data(self):
        with self.data_lock:
            snapshot = json.loads(json.dumps(self.latest_results))
        snapshot["enabled"] = self.get_status()
        snapshot["server_time"] = time.time()
        return snapshot

    def loop_fhd_tasks(self):
        # 💡 [고속화 핵심 3] 기존 8 FPS 제한(1/8) -> 22 FPS(1/22)로 대폭 완화
        frame_time = 1.0 / 22.0
        while self.running:
            loop_start = time.time()

            barcode_on = self.is_enabled("barcode")
            apriltag_on = self.is_enabled("apriltag")

            if not barcode_on and not apriltag_on:
                time.sleep(0.04)
                continue

            base_frame = self.fhd_buffer.copy()
            now_ts = time.time()

            # 1. Barcode 처리 (ON일 때만)
            if barcode_on:
                frame_barcode = base_frame.copy() if apriltag_on else base_frame
                h, w, _ = frame_barcode.shape
                x1, y1 = (w - self.bc_roi_x) // 2, (h - self.bc_roi_y) // 2 + 30
                x2, y2 = x1 + self.bc_roi_x, y1 + self.bc_roi_y

                roi_img = frame_barcode[y1:y2, x1:x2]
                zoomed_roi = cv2.resize(roi_img, (self.bc_roi_x * self.bc_scale, self.bc_roi_y * self.bc_scale), interpolation=cv2.INTER_LINEAR)
                decoded = pyzbar.decode(zoomed_roi)

                detected_barcodes = []
                roi_color = (0, 255, 0) if len(decoded) > 0 else (255, 255, 255)
                cv2.rectangle(frame_barcode, (x1, y1), (x2, y2), roi_color, 2)

                for obj in decoded:
                    data = obj.data.decode("utf-8", errors="replace")
                    detected_barcodes.append(data)
                    cv2.putText(zoomed_roi, f"DATA: {data}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                with self.data_lock:
                    self.latest_results["barcode"]["detected"] = len(detected_barcodes) > 0
                    self.latest_results["barcode"]["barcodes"] = detected_barcodes
                    self.latest_results["barcode"]["timestamp"] = now_ts
                    if detected_barcodes:
                        self.latest_results["barcode"]["last_barcode"] = detected_barcodes[0]
                        self.latest_results["barcode"]["last_barcode_time"] = now_ts

                if detected_barcodes:
                    self.barcode_pub.send_string(json.dumps({"timestamp": now_ts, "barcodes": detected_barcodes}))

                # 전송용 이미지 크기 640x360으로 경량 인코딩 (FHD 그대로 인코딩하는 것보다 6배 빠름!)
                jpg_main = self._encode_jpg_fast(frame_barcode, resize_to=(640, 360), quality=70)
                jpg_zoom = self._encode_jpg_fast(zoomed_roi, resize_to=(640, 200), quality=75)
                with self.lock:
                    self.encoded_cache["barcode_main"] = jpg_main
                    self.encoded_cache["barcode_zoom"] = jpg_zoom

            # 2. AprilTag 처리 (ON일 때만)
            if apriltag_on:
                frame_apriltag = base_frame
                gray = cv2.cvtColor(frame_apriltag, cv2.COLOR_BGR2GRAY)
                tags = self.detector.detect(gray, estimate_tag_pose=False)
                detected_tags_data = []

                for i, tag in enumerate(tags):
                    tag_id = tag.tag_id
                    img_points = tag.corners
                    corners_int = img_points.astype(int)
                    for j in range(4):
                        cv2.line(frame_apriltag, tuple(corners_int[j]), tuple(corners_int[(j+1)%4]), (0, 255, 0), 3)

                    if tag_id not in self.kf_dict:
                        self.kf_dict[tag_id] = {"x": KalmanFilter1D(), "y": KalmanFilter1D(), "z": KalmanFilter1D()}

                    success, rvec, tvec = cv2.solvePnP(self.obj_points, img_points, self.camera_matrix, self.dist_coeffs, flags=cv2.SOLVEPNP_SQPNP)
                    if success:
                        R, _ = cv2.Rodrigues(rvec)
                        camera_pos = -np.dot(R.T, tvec)
                        raw_x, raw_y, raw_z = camera_pos[0][0]*1000, camera_pos[1][0]*1000, -camera_pos[2][0]*1000
                        kf = self.kf_dict[tag_id]
                        cam_x, cam_y, cam_z = kf["x"].update(raw_x), kf["y"].update(raw_y), kf["z"].update(raw_z)
                        detected_tags_data.append({
                            "ID": int(tag_id),
                            "x": round(float(cam_x), 2),
                            "y": round(float(cam_y), 2),
                            "z": round(float(cam_z), 2)
                        })

                        cv2.putText(frame_apriltag, f"{detected_tags_data[-1]}", (20, 55 + (i*45)), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 255), 3)
                        cv2.circle(frame_apriltag, (1920//2, 1080//2), 8, (0, 0, 255), -1)

                with self.data_lock:
                    self.latest_results["apriltag"]["detected"] = len(detected_tags_data) > 0
                    self.latest_results["apriltag"]["tags"] = detected_tags_data
                    self.latest_results["apriltag"]["timestamp"] = now_ts
                    if detected_tags_data:
                        self.latest_results["apriltag"]["last_seen_tags"] = detected_tags_data

                if detected_tags_data:
                    self.tag_pub.send_string(json.dumps({"tags": detected_tags_data}))

                jpg_april = self._encode_jpg_fast(frame_apriltag, resize_to=(640, 360), quality=72)
                with self.lock:
                    self.encoded_cache["apriltag"] = jpg_april

            elapsed = time.time() - loop_start
            if elapsed < frame_time:
                time.sleep(frame_time - elapsed)

    def loop_vga_tasks(self):
        frame_time = 1.0 / 16.0
        while self.running:
            loop_start = time.time()

            if not self.is_enabled("yolo"):
                time.sleep(0.05)
                continue

            frame_vga = self.vga_buffer.copy()
            annotated_frame = frame_vga

            roi_x1, roi_y1, roi_x2, roi_y2 = self.target_roi
            cv2.rectangle(annotated_frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 255, 255), 1)

            is_in_roi = False
            detected_centers = []

            if self.yolo_ready and self.yolo_model is not None:
                results = self.yolo_model(frame_vga, conf=0.4, verbose=False, stream=True)
                for result in results:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                        conf = round(float(box.conf[0].item()), 3) if box.conf is not None else 0.0
                        detected_centers.append({"x": cx, "y": cy, "conf": conf})

                        if (roi_x1 <= cx <= roi_x2) and (roi_y1 <= cy <= roi_y2):
                            is_in_roi = True
                            color = (0, 255, 0)
                        else:
                            color = (0, 0, 255)
                        cv2.circle(annotated_frame, (cx, cy), 5, color, -1)
            else:
                cv2.putText(annotated_frame, "Loading YOLO Model...", (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            now_ts = time.time()
            yolo_payload = {
                "timestamp": now_ts,
                "is_in_roi": is_in_roi,
                "detected_objects": detected_centers
            }
            with self.data_lock:
                self.latest_results["yolo"] = yolo_payload

            self.yolo_pub.send_string(json.dumps(yolo_payload))

            status_text = "STATUS: DETECTED" if is_in_roi else "STATUS: NONE"
            status_color = (0, 255, 0) if is_in_roi else (0, 0, 255)
            cv2.putText(annotated_frame, status_text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 2)

            jpg_yolo = self._encode_jpg_fast(annotated_frame, resize_to=(640, 480), quality=72)
            with self.lock:
                self.encoded_cache["yolo"] = jpg_yolo

            elapsed = time.time() - loop_start
            if elapsed < frame_time:
                time.sleep(frame_time - elapsed)

    def get_encoded_frame(self, target):
        feature_map = {
            "apriltag": "apriltag",
            "barcode_main": "barcode",
            "barcode_zoom": "barcode",
            "yolo": "yolo",
        }
        feature = feature_map.get(target)
        if feature is not None and not self.is_enabled(feature):
            return self.disabled_jpg_bytes

        with self.lock:
            return self.encoded_cache.get(target)


# ==========================================
# 3. Flask Server & Stylish Dashboard UI
# ==========================================
app = Flask(__name__)
vision_sub = None

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>AI Vision Control Center</title>
    <link href="https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;600;700&display=swap" rel="stylesheet">
    <style>
        body { background-color: #0b0f19; color: #e2e8f0; font-family: 'Rajdhani', sans-serif; margin: 0; padding: 20px 40px; }
        .header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #1e293b; padding-bottom: 15px; margin-bottom: 30px; }
        .header h1 { margin: 0; font-size: 2.2rem; color: #38bdf8; letter-spacing: 1px; text-shadow: 0 0 10px rgba(56, 189, 248, 0.4); }
        .status-badge { background: rgba(16, 185, 129, 0.1); color: #10b981; padding: 8px 16px; border-radius: 20px; border: 1px solid #10b981; font-weight: 600; display: flex; align-items: center; gap: 8px; }
        .dot { width: 10px; height: 10px; background-color: #10b981; border-radius: 50%; animation: pulse 1.5s infinite; }
        @keyframes pulse { 0% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); } 70% { box-shadow: 0 0 0 10px rgba(16, 185, 129, 0); } 100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); } }
        .grid-container { display: grid; grid-template-columns: repeat(auto-fit, minmax(600px, 1fr)); gap: 30px; justify-content: center; }
        .video-card { background: rgba(30, 41, 59, 0.7); backdrop-filter: blur(10px); padding: 20px; border-radius: 16px; border: 1px solid rgba(255, 255, 255, 0.05); box-shadow: 0 10px 25px rgba(0, 0, 0, 0.5); display: flex; flex-direction: column; align-items: center; }
        .card-header { display: flex; justify-content: space-between; align-items: center; width: 100%; margin-bottom: 15px; }
        .card-header h3 { margin: 0; font-size: 1.3rem; color: #94a3b8; border-left: 4px solid #3b82f6; padding-left: 10px; }
        .video-card.yolo .card-header h3 { border-left-color: #f59e0b; }
        .video-card.barcode .card-header h3 { border-left-color: #10b981; }
        .stream-img { border-radius: 8px; border: 1px solid #334155; background-color: #000; max-width: 100%; height: auto; }
        .highlight .stream-img { border: 2px solid #22c55e; box-shadow: 0 0 20px rgba(34, 197, 94, 0.2); }
        .toggle-btn { border: none; border-radius: 20px; padding: 6px 18px; font-family: 'Rajdhani', sans-serif; font-weight: 700; font-size: 0.85rem; letter-spacing: 1px; cursor: pointer; transition: all 0.15s ease; }
        .toggle-btn.on { background: rgba(16, 185, 129, 0.15); color: #10b981; border: 1px solid #10b981; box-shadow: 0 0 8px rgba(16,185,129,0.3); }
        .toggle-btn.off { background: rgba(239, 68, 68, 0.12); color: #ef4444; border: 1px solid #ef4444; }
        .toggle-btn:active { transform: scale(0.94); }
    </style>
</head>
<body>
    <div class="header">
        <h1>Bio SCARA Function (Fast Stream)</h1>
        <div class="status-badge"><div class="dot"></div>SYSTEM ONLINE</div>
    </div>
    <div class="grid-container">
        <div class="video-card">
            <div class="card-header">
                <h3>AprilTag Pose Estimation</h3>
                <button class="toggle-btn on" data-feature="apriltag" data-enabled="true" onclick="toggleFeature('apriltag', this)">ON</button>
            </div>
            <img src="/video/apriltag" width="640" class="stream-img">
        </div>
        <div class="video-card yolo">
            <div class="card-header">
                <h3>YOLO Microplate Detection</h3>
                <button class="toggle-btn on" data-feature="yolo" data-enabled="true" onclick="toggleFeature('yolo', this)">ON</button>
            </div>
            <img src="/video/yolo" width="440" class="stream-img">
        </div>
        <div class="video-card barcode">
            <div class="card-header">
                <h3>Barcode ROI (Main)</h3>
                <button class="toggle-btn off" data-feature="barcode" data-enabled="false" onclick="toggleFeature('barcode', this)">OFF</button>
            </div>
            <img src="/video/barcode_main" width="640" class="stream-img">
        </div>
        <div class="video-card barcode highlight" style="justify-content: center;">
            <div class="card-header">
                <h3>Barcode Zoom & Decode</h3>
                <button class="toggle-btn off" data-feature="barcode" data-enabled="false" onclick="toggleFeature('barcode', this)">OFF</button>
            </div>
            <img src="/video/barcode_zoom" width="640" class="stream-img">
        </div>
    </div>

    <script>
        function updateButton(btn, enabled) {
            btn.dataset.enabled = enabled;
            btn.textContent = enabled ? "ON" : "OFF";
            btn.classList.toggle("on", enabled);
            btn.classList.toggle("off", !enabled);
        }
        async function toggleFeature(feature, btn) {
            const currentlyOn = btn.dataset.enabled === "true";
            const nextValue = !currentlyOn;
            try {
                const res = await fetch(`/api/toggle/${feature}`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ enabled: nextValue })
                });
                const data = await res.json();
                document.querySelectorAll(`[data-feature="${feature}"]`).forEach(b => updateButton(b, data.enabled));
            } catch (e) { console.error("toggle 요청 실패:", e); }
        }
        async function refreshStatus() {
            try {
                const res = await fetch('/api/status');
                const data = await res.json();
                document.querySelectorAll('[data-feature]').forEach(btn => {
                    const feature = btn.dataset.feature;
                    if (data[feature] !== undefined) updateButton(btn, data[feature]);
                });
            } catch (e) {}
        }
        window.addEventListener('DOMContentLoaded', refreshStatus);
        setInterval(refreshStatus, 2000);
    </script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(DASHBOARD_HTML)

@app.route("/video/<target>")
def video_feed(target):
    def gen():
        while True:
            frame_bytes = vision_sub.get_encoded_frame(target)
            if frame_bytes:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
            time.sleep(0.04)  # 기존 0.1s(10FPS) -> 0.04s(25FPS) 고속 스트리밍
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/snapshot/<target>")
def snapshot_feed(target):
    frame_bytes = vision_sub.get_encoded_frame(target)
    if frame_bytes:
        return Response(bytes(frame_bytes), mimetype="image/jpeg")
    return Response(status=204)

@app.route("/api/status")
def api_status():
    return jsonify(vision_sub.get_status())

@app.route("/api/vision_data")
def api_vision_data():
    return jsonify(vision_sub.get_vision_data())

@app.route("/api/reset_kalman", methods=["POST"])
def api_reset_kalman():
    vision_sub.reset_kalman()
    return jsonify({"ok": True})

@app.route("/api/clear_barcode", methods=["POST"])
def api_clear_barcode():
    vision_sub.clear_last_barcode()
    return jsonify({"ok": True})

@app.route("/api/toggle/<feature>", methods=["POST"])
def api_toggle(feature):
    if feature not in ("apriltag", "barcode", "yolo"):
        return jsonify({"error": f"unknown feature '{feature}'"}), 400

    payload = request.get_json(silent=True) or {}
    if "enabled" in payload:
        new_value = bool(payload["enabled"])
    else:
        new_value = not vision_sub.is_enabled(feature)

    vision_sub.set_enabled(feature, new_value)
    return jsonify({"feature": feature, "enabled": new_value})


# ==========================================
# 4. Main Entry Point (빠른 부팅 적용)
# ==========================================
if __name__ == "__main__":
    cam_process = multiprocessing.Process(target=run_publisher, daemon=True)
    cam_process.start()

    time.sleep(0.6)  # 기존 2.0초 대기 -> 0.6초로 단축 (SharedMemory 연결 루프가 자동 대기함)

    yolo_path = '/home/rnd/yolo_model/best_yolov26n.onnx'
    vision_sub = IntegratedVisionSubscriber(yolo_model_path=yolo_path)

    t_fhd = threading.Thread(target=vision_sub.loop_fhd_tasks, daemon=True)
    t_vga = threading.Thread(target=vision_sub.loop_vga_tasks, daemon=True)
    t_fhd.start()
    t_vga.start()

    print("\n✅ 고속 비전 서버가 시작되었습니다. 5000 포트로 즉시 접속 가능합니다.\n")
    app.run(host="0.0.0.0", port=5000, threaded=True)