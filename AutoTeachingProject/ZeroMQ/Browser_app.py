import time
import threading
import multiprocessing
import cv2
import os
import numpy as np
import zmq
import json
from multiprocessing import shared_memory
from flask import Flask, Response, render_template_string, jsonify, request
from pyzbar import pyzbar
from pupil_apriltags import Detector
from ultralytics import YOLO

try:
    from picamera2 import Picamera2
except ImportError:
    print("⚠️ 오류: picamera2를 찾을 수 없습니다. (라즈베리파이 환경에서 실행해주세요)")

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
cv2.setNumThreads(1)  # OpenCV 스레드 수 제한 (YOLO 모델 로딩 시 CPU 과부하 방지)

# ==========================================
# 1. Camera Publisher 클래스
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
            time.sleep(1.0)
            self.picam2.set_controls({"AfMode": 0, "LensPosition": 5.5})
        except Exception as e:
            pass

    def start_streaming(self):
        print(f"🚀 Camera Publisher 프로세스 시작됨")
        try:
            while True:
                raw_frame = self.picam2.capture_array()
                if raw_frame is None: continue

                frame_fhd = cv2.cvtColor(raw_frame, cv2.COLOR_RGB2BGR)
                frame_vga = cv2.resize(frame_fhd, (self.vga_w, self.vga_h), interpolation=cv2.INTER_AREA)

                np.copyto(self.frame_fhd_shared, frame_fhd)
                np.copyto(self.frame_vga_shared, frame_vga)
                time.sleep(0.05)
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
# 2. Vision Subscriber 및 칼만 필터
# ==========================================
class KalmanFilter1D:
    def __init__(self, process_noise=1e-3, measurement_noise=0.3):
        self.x, self.p = 0.0, 1.0
        self.q, self.r = process_noise, measurement_noise
        self.initialized = False
    def update(self, measurement):
        if not self.initialized:
            self.x = measurement; self.initialized = True
            return self.x
        self.p += self.q
        k = self.p / (self.p + self.r)
        self.x += k * (measurement - self.x)
        self.p *= (1 - k)
        return self.x

class IntegratedVisionSubscriber:
    def __init__(self, yolo_model_path: str):
        self.lock = threading.Lock()
        self.running = True

        self.enabled_lock = threading.Lock()
        self.enabled = {
            "apriltag": True,
            "barcode": True,
            "yolo": True,
        }

        self.apriltag_frame = None
        self.barcode_main_frame = None
        self.barcode_zoom_frame = None
        self.yolo_frame = None

        print("🔗 Shared Memory 연결 대기 중...")
        while True:
            try:
                self.shm_fhd = shared_memory.SharedMemory(name="vision_fhd")
                self.shm_vga = shared_memory.SharedMemory(name="vision_vga")
                break
            except FileNotFoundError:
                time.sleep(0.5)

        self.fhd_buffer = np.ndarray((1080, 1920, 3), dtype=np.uint8, buffer=self.shm_fhd.buf)
        self.vga_buffer = np.ndarray((480, 640, 3), dtype=np.uint8, buffer=self.shm_vga.buf)

        self.pub_context = zmq.Context()
        self.tag_pub = self.pub_context.socket(zmq.PUB)
        self.barcode_pub = self.pub_context.socket(zmq.PUB)
        self.yolo_pub = self.pub_context.socket(zmq.PUB)
        
        # 통신 방식을 TCP에서 IPC로 변경
        self.tag_pub.bind("ipc:///tmp/vision_tag.ipc")
        self.barcode_pub.bind("ipc:///tmp/vision_barcode.ipc")
        self.yolo_pub.bind("ipc:///tmp/vision_yolo.ipc")

        self.tag_size = 0.021
        self.half_s = self.tag_size / 2.0
        
        # 1920x1080 비율에 맞게 스케일링된 카메라 매트릭스
        # self.camera_matrix = np.array([
        #     [1377.30483, 0, 968.26621],
        #     [0, 1381.74778, 531.14559],
        #     [0, 0, 1]
        # ], dtype=np.float32)
        # self.dist_coeffs = np.array([[-0.02188, 0.33220, 0.00308, 0.00052, -0.50083]], dtype=np.float32)
        self.camera_matrix = np.array([
                    [1377.30483, 0, 970.26621],
                    [0, 1381.74778, 531.14559],
                    [0, 0, 1]
                ], dtype=np.float32)
        self.dist_coeffs = np.array([[-0.02188, 0.33220, 0.00308, 0.00052, -0.50083]], dtype=np.float32)
        self.obj_points = np.array([[-self.half_s, -self.half_s, 0], [self.half_s, -self.half_s, 0],
                                    [self.half_s, self.half_s, 0], [-self.half_s, self.half_s, 0]], dtype=np.float32)
        self.detector = Detector(families='tag36h11', nthreads=1, quad_decimate=2.0, debug=0)
        self.kf_dict = {}
        self.bc_roi_x, self.bc_roi_y, self.bc_scale = 400, 120, 3

        print(f"⚙️ YOLO 모델 로딩 중... ({yolo_model_path})")
        self.yolo_model = YOLO(yolo_model_path, task='detect')
        self.target_roi = (206, 212, 434, 268)

        self.disabled_jpg_bytes = self._build_disabled_placeholder()

    def _build_disabled_placeholder(self):
        placeholder = np.zeros((360, 640, 3), dtype=np.uint8)
        cv2.putText(placeholder, "DISABLED", (220, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (110, 110, 110), 3)
        # cv2.putText(placeholder, "function is turned off", (140, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (70, 70, 70), 1)
        success, encoded = cv2.imencode(".jpg", placeholder, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
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

    def loop_fhd_tasks(self):
        frame_time = 1.0 / 8.0
        while self.running:
            loop_start = time.time()

            barcode_on = self.is_enabled("barcode")
            apriltag_on = self.is_enabled("apriltag")

            if not barcode_on and not apriltag_on:
                time.sleep(max(0.05, frame_time - (time.time() - loop_start)))
                continue

            # 💡 [핵심 수정] 원본 프레임을 가져온 후, 도화지를 분리합니다.
            base_frame = self.fhd_buffer.copy()
            frame_barcode = None
            frame_apriltag = None

            # 둘 다 켜져있다면 서로 겹치지 않게 프레임을 독립적으로 복사합니다.
            if barcode_on and apriltag_on:
                frame_barcode = base_frame.copy()
                frame_apriltag = base_frame  # 원본은 그대로 AprilTag에 할당
            elif barcode_on:
                frame_barcode = base_frame
            elif apriltag_on:
                frame_apriltag = base_frame

            zoomed_roi = None

            # 1. Barcode 처리 (ON일 때만)
            if barcode_on:
                h, w, _ = frame_barcode.shape
                x1, y1 = (w - self.bc_roi_x) // 2, (h - self.bc_roi_y) // 2 + 30
                x2, y2 = x1 + self.bc_roi_x, y1 + self.bc_roi_y
                
                roi_img = frame_barcode[y1:y2, x1:x2]
                zoomed_roi = cv2.resize(roi_img, (self.bc_roi_x * self.bc_scale, self.bc_roi_y * self.bc_scale), interpolation=cv2.INTER_NEAREST)
                decoded = pyzbar.decode(zoomed_roi)

                detected_barcodes = []
                roi_color = (0, 255, 0) if len(decoded) > 0 else (255, 255, 255)

                # 바코드 전용 프레임에만 윤곽선을 그립니다.
                cv2.rectangle(frame_barcode, (x1, y1), (x2, y2), roi_color, 2)

                for obj in decoded:
                    data = obj.data.decode("utf-8")
                    detected_barcodes.append(data)
                    cv2.putText(zoomed_roi, f"DATA: {data}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                if detected_barcodes:
                    self.barcode_pub.send_string(json.dumps({"timestamp": time.time(), "barcodes": detected_barcodes}))

            # 2. AprilTag 처리 (ON일 때만)
            if apriltag_on:
                gray = cv2.cvtColor(frame_apriltag, cv2.COLOR_BGR2GRAY)
                tags = self.detector.detect(gray, estimate_tag_pose=False)
                detected_tags_data = []

                for i, tag in enumerate(tags):
                    tag_id = tag.tag_id
                    img_points = tag.corners
                    corners_int = img_points.astype(int)
                    for j in range(4):
                        # AprilTag 전용 프레임에만 윤곽선을 그립니다.
                        cv2.line(frame_apriltag, tuple(corners_int[j]), tuple(corners_int[(j+1)%4]), (0, 255, 0), 2)

                    if tag_id not in self.kf_dict:
                        self.kf_dict[tag_id] = {"x": KalmanFilter1D(), "y": KalmanFilter1D(), "z": KalmanFilter1D()}

                    success, rvec, tvec = cv2.solvePnP(self.obj_points, img_points, self.camera_matrix, self.dist_coeffs, flags=cv2.SOLVEPNP_SQPNP)
                    if success:
                        R, _ = cv2.Rodrigues(rvec)
                        camera_pos = -np.dot(R.T, tvec)
                        raw_x, raw_y, raw_z = camera_pos[0][0]*1000, camera_pos[1][0]*1000, -camera_pos[2][0]*1000
                        kf = self.kf_dict[tag_id]
                        cam_x, cam_y, cam_z = kf["x"].update(raw_x), kf["y"].update(raw_y), kf["z"].update(raw_z)
                        detected_tags_data.append({"ID": int(tag_id), "x": round(float(cam_x), 1), "y": round(float(cam_y), 1), "z": round(float(cam_z), 1)})
                        
                        # AprilTag 전용 프레임에만 텍스트를 작성합니다.
                        cv2.putText(frame_apriltag, f"{detected_tags_data[-1]}", (20, 50 + (i*40)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
                        cv2.circle(frame_apriltag, (1920//2, 1080//2), 8, (0, 0, 255), -1)

                if detected_tags_data:
                    self.tag_pub.send_string(json.dumps({"tags": detected_tags_data}))

            # 최종적으로 완성된 각 프레임을 전역 변수에 저장합니다.
            with self.lock:
                if barcode_on:
                    self.barcode_main_frame = frame_barcode
                    self.barcode_zoom_frame = zoomed_roi
                if apriltag_on:
                    self.apriltag_frame = frame_apriltag

            time.sleep(max(0.01, frame_time - (time.time() - loop_start)))

    def loop_vga_tasks(self):
        frame_time = 1.0 / 10.0
        while self.running:
            loop_start = time.time()

            if not self.is_enabled("yolo"):
                time.sleep(max(0.05, frame_time - (time.time() - loop_start)))
                continue

            frame_vga = self.vga_buffer.copy()
            results = self.yolo_model(frame_vga, conf=0.4, verbose=False, stream=True)
            annotated_frame = frame_vga.copy()

            roi_x1, roi_y1, roi_x2, roi_y2 = self.target_roi
            cv2.rectangle(annotated_frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 255, 255), 1)

            is_in_roi = False
            detected_centers = []

            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    detected_centers.append({"x": cx, "y": cy})

                    if (roi_x1 <= cx <= roi_x2) and (roi_y1 <= cy <= roi_y2):
                        is_in_roi = True
                        color = (0, 255, 0)
                    else:
                        color = (0, 0, 255)
                    cv2.circle(annotated_frame, (cx, cy), 5, color, -1)

            yolo_payload = {
                "timestamp": time.time(),
                "is_in_roi": is_in_roi,
                "detected_objects": detected_centers
            }
            self.yolo_pub.send_string(json.dumps(yolo_payload))

            status_text = "STATUS: DETECTED" if is_in_roi else "STATUS: NONE"
            status_color = (0, 255, 0) if is_in_roi else (0, 0, 255)
            cv2.putText(annotated_frame, status_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)

            with self.lock:
                self.yolo_frame = annotated_frame

            time.sleep(max(0.01, frame_time - (time.time() - loop_start)))

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
            if target == "apriltag":
                frame = self.apriltag_frame
            elif target == "barcode_main":
                frame = self.barcode_main_frame
            elif target == "barcode_zoom":
                frame = self.barcode_zoom_frame
            elif target == "yolo":
                frame = self.yolo_frame
            else:
                frame = None

        if frame is None:
            return None

        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
        success, encoded_jpg = cv2.imencode(".jpg", frame, encode_params)

        if success:
            return bytearray(encoded_jpg)
        return None


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
        <h1>Bio SCARA Function</h1>
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
            <img src="/video/yolo" width="480" class="stream-img">
        </div>
        <div class="video-card barcode">
            <div class="card-header">
                <h3>Barcode ROI (Main)</h3>
                <button class="toggle-btn on" data-feature="barcode" data-enabled="true" onclick="toggleFeature('barcode', this)">ON</button>
            </div>
            <img src="/video/barcode_main" width="640" class="stream-img">
        </div>
        <div class="video-card barcode highlight" style="justify-content: center;">
            <div class="card-header">
                <h3>Barcode Zoom & Decode</h3>
                <button class="toggle-btn on" data-feature="barcode" data-enabled="true" onclick="toggleFeature('barcode', this)">ON</button>
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
            } catch (e) {
                console.error("toggle 요청 실패:", e);
            }
        }

        async function refreshStatus() {
            try {
                const res = await fetch('/api/status');
                const data = await res.json();
                document.querySelectorAll('[data-feature]').forEach(btn => {
                    const feature = btn.dataset.feature;
                    if (data[feature] !== undefined) updateButton(btn, data[feature]);
                });
            } catch (e) {
                console.error("status 요청 실패:", e);
            }
        }

        window.addEventListener('DOMContentLoaded', refreshStatus);
        setInterval(refreshStatus, 3000);
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
            time.sleep(0.1)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/status")
def api_status():
    return jsonify(vision_sub.get_status())

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
# 4. Main Entry Point
# ==========================================
if __name__ == "__main__":
    cam_process = multiprocessing.Process(target=run_publisher, daemon=True)
    cam_process.start()

    time.sleep(2.0)

    yolo_path = '/home/rnd/yolo_model/best_yolov26n.onnx'
    vision_sub = IntegratedVisionSubscriber(yolo_model_path=yolo_path)

    t_fhd = threading.Thread(target=vision_sub.loop_fhd_tasks, daemon=True)
    t_vga = threading.Thread(target=vision_sub.loop_vga_tasks, daemon=True)
    t_fhd.start()
    t_vga.start()

    print("\n✅ 모든 시스템이 시작되었습니다. 웹 브라우저에서 서버 IP의 5000 포트로 접속하세요.\n")
    app.run(host="0.0.0.0", port=5000)