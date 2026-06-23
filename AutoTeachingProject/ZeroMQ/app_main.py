import time
import threading
import multiprocessing
import cv2
import numpy as np
import zmq
import json
from multiprocessing import shared_memory
from flask import Flask, Response, render_template_string
from pyzbar import pyzbar
from pupil_apriltags import Detector
from ultralytics import YOLO

try:
    from picamera2 import Picamera2
except ImportError:
    print("⚠️ 오류: picamera2를 찾을 수 없습니다. (라즈베리파이 환경에서 실행해주세요)")

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

# Publisher를 실행할 래퍼 함수 (멀티프로세싱용)
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
        self.apriltag_jpg = self.barcode_main_jpg = self.barcode_zoom_jpg = self.yolo_jpg = None

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
        self.tag_pub.bind("tcp://*:5557")
        self.barcode_pub = self.pub_context.socket(zmq.PUB)
        self.barcode_pub.bind("tcp://*:5558")
        self.yolo_pub = self.pub_context.socket(zmq.PUB)
        self.yolo_pub.bind("tcp://*:5559")

        self.tag_w, self.tag_h = 1280, 720
        self.tag_size = 0.03; self.half_s = self.tag_size / 2.0
        scale_x, scale_y = self.tag_w / 1920.0, self.tag_h / 1080.0
        self.camera_matrix = np.array([[1522.36*(1920/1280)*scale_x, 0, 612.83*(1920/1280)*scale_x], 
                                       [0, 1520.01*(1080/720)*scale_y, 370.20*(1080/720)*scale_y], [0, 0, 1]], dtype=np.float32)
        self.dist_coeffs = np.array([[-0.040964, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        self.obj_points = np.array([[-self.half_s, -self.half_s, 0], [self.half_s, -self.half_s, 0],
                                    [self.half_s, self.half_s, 0], [-self.half_s, self.half_s, 0]], dtype=np.float32)
        self.detector = Detector(families='tag36h11', nthreads=1, quad_decimate=2.0, debug=0)
        self.kf_dict = {}
        self.bc_roi_x, self.bc_roi_y, self.bc_scale = 400, 120, 3

        print(f"⚙️ YOLO 모델 로딩 중... ({yolo_model_path})")
        self.yolo_model = YOLO(yolo_model_path, task='detect')
        self.target_roi = (206, 212, 434, 268)

    def loop_fhd_tasks(self):
        frame_time = 1.0 / 8.0
        while self.running:
            loop_start = time.time()
            frame_fhd = self.fhd_buffer.copy()

            # Barcode
            h, w, _ = frame_fhd.shape
            x1, y1 = (w - self.bc_roi_x) // 2, (h - self.bc_roi_y) // 2 + 30
            x2, y2 = x1 + self.bc_roi_x, y1 + self.bc_roi_y
            roi_img = frame_fhd[y1:y2, x1:x2]
            zoomed_roi = cv2.resize(roi_img, (self.bc_roi_x * self.bc_scale, self.bc_roi_y * self.bc_scale))
            decoded = pyzbar.decode(zoomed_roi)
            
            detected_barcodes = []
            roi_color = (0, 255, 0) if len(decoded) > 0 else (255, 255, 255)
            display_bc_main = frame_fhd.copy()
            cv2.rectangle(display_bc_main, (x1, y1), (x2, y2), roi_color, 2)

            for obj in decoded:
                data = obj.data.decode("utf-8")
                detected_barcodes.append(data)
                cv2.putText(zoomed_roi, f"DATA: {data}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            if detected_barcodes:
                self.barcode_pub.send_string(json.dumps({"timestamp": time.time(), "barcodes": detected_barcodes}))

            # AprilTag
            frame_tag = cv2.resize(frame_fhd, (self.tag_w, self.tag_h))
            gray = cv2.cvtColor(frame_tag, cv2.COLOR_BGR2GRAY)
            tags = self.detector.detect(gray, estimate_tag_pose=False)
            detected_tags_data = []

            for i, tag in enumerate(tags):
                tag_id = tag.tag_id
                img_points = tag.corners
                corners_int = img_points.astype(int)
                for j in range(4):
                    cv2.line(frame_tag, tuple(corners_int[j]), tuple(corners_int[(j+1)%4]), (0, 255, 0), 2)

                if tag_id not in self.kf_dict:
                    self.kf_dict[tag_id] = {"x": KalmanFilter1D(), "y": KalmanFilter1D(), "z": KalmanFilter1D()}

                success, rvec, tvec = cv2.solvePnP(self.obj_points, img_points, self.camera_matrix, self.dist_coeffs, flags=cv2.SOLVEPNP_SQPNP)
                if success:
                    R, _ = cv2.Rodrigues(rvec)
                    camera_pos = -np.dot(R.T, tvec)
                    raw_x, raw_y, raw_z = camera_pos[0][0]*1000, camera_pos[1][0]*1000, -camera_pos[2][0]*1000
                    kf = self.kf_dict[tag_id]
                    cam_x, cam_y, cam_z = kf["x"].update(raw_x), kf["y"].update(raw_y), kf["z"].update(raw_z)
                    detected_tags_data.append({"id": int(tag_id), "x": round(float(cam_x), 2), "y": round(float(cam_y), 2), "z": round(float(cam_z), 2)})
                    cv2.putText(frame_tag, f"ID:{tag_id} | Z: {cam_z:.0f}", (20, 50 + (i*40)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            if detected_tags_data:
                self.tag_pub.send_string(json.dumps({"tags": detected_tags_data}))

            _, bc_main_jpg = cv2.imencode(".jpg", display_bc_main)
            _, bc_zoom_jpg = cv2.imencode(".jpg", zoomed_roi)
            _, tag_jpg = cv2.imencode(".jpg", frame_tag)

            with self.lock:
                self.barcode_main_jpg = bytearray(bc_main_jpg)
                self.barcode_zoom_jpg = bytearray(bc_zoom_jpg)
                self.apriltag_jpg = bytearray(tag_jpg)

            time.sleep(max(0.01, frame_time - (time.time() - loop_start)))

    def loop_vga_tasks(self):
        """스레드 2: YOLO 객체 인식 처리 (목표 15 FPS)"""
        frame_time = 1.0 / 15.0
        
        while self.running:
            loop_start = time.time()
            
            # 1. 공유 메모리에서 VGA 프레임 복사
            frame_vga = self.vga_buffer.copy()

            # 2. YOLO 모델 추론 (제너레이터 스트림 방식)
            results = self.yolo_model(frame_vga, conf=0.4, verbose=False, stream=True)
            annotated_frame = frame_vga.copy()
            
            # 3. 관심 영역(ROI) 렌더링
            roi_x1, roi_y1, roi_x2, roi_y2 = self.target_roi
            cv2.rectangle(annotated_frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 255, 255), 1)

            is_in_roi = False
            detected_centers = [] # 추적된 객체들의 중심 좌표 저장 리스트

            # 4. 바운딩 박스 및 중심점 추출
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    
                    detected_centers.append({"x": cx, "y": cy})
                    
                    # 중심점이 ROI 안에 들어왔는지 판별
                    if (roi_x1 <= cx <= roi_x2) and (roi_y1 <= cy <= roi_y2):
                        is_in_roi = True
                        color = (0, 255, 0)
                    else:
                        color = (0, 0, 255)
                        
                    cv2.circle(annotated_frame, (cx, cy), 5, color, -1)

            # 5. 💡 [ZMQ Publish] YOLO 인식 상태 및 좌표 JSON 송신 (5559 포트)
            yolo_payload = {
                "timestamp": time.time(),
                "is_in_roi": is_in_roi,
                "detected_objects": detected_centers
            }
            # 외부 로봇 제어기가 구독할 수 있도록 데이터 발행
            self.yolo_pub.send_string(json.dumps(yolo_payload))

            # 6. 화면 송출용 상태 텍스트 오버레이
            status_text = "STATUS: DETECTED" if is_in_roi else "STATUS: NONE"
            status_color = (0, 255, 0) if is_in_roi else (0, 0, 255)
            cv2.putText(annotated_frame, status_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)

            # 7. 💡 [GUI 렌더링용] NumPy Array 원본을 안전하게 변수에 저장
            _, yolo_encoded = cv2.imencode(".jpg", annotated_frame)
            with self.lock:
                self.yolo_jpg = bytearray(yolo_encoded)

            # 8. 목표 프레임 레이트(15FPS) 유지를 위한 스마트 슬립
            elapsed_time = time.time() - loop_start
            time.sleep(max(0.01, frame_time - elapsed_time))

    def get_frame(self, target):
        with self.lock:
            if target == "apriltag": return self.apriltag_jpg
            elif target == "barcode_main": return self.barcode_main_jpg
            elif target == "barcode_zoom": return self.barcode_zoom_jpg
            elif target == "yolo": return self.yolo_jpg
            return None


# ==========================================
# 3. Flask Server & Stylish Dashboard UI
# ==========================================
app = Flask(__name__)
vision_sub = None 

# 모던하고 사이버네틱한 UI가 적용된 HTML 템플릿
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>AI Vision Control Center</title>
    <link href="https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;600;700&display=swap" rel="stylesheet">
    <style>
        body {
            background-color: #0b0f19;
            color: #e2e8f0;
            font-family: 'Rajdhani', sans-serif;
            margin: 0;
            padding: 20px 40px;
        }
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid #1e293b;
            padding-bottom: 15px;
            margin-bottom: 30px;
        }
        .header h1 {
            margin: 0;
            font-size: 2.2rem;
            color: #38bdf8;
            letter-spacing: 1px;
            text-shadow: 0 0 10px rgba(56, 189, 248, 0.4);
        }
        .status-badge {
            background: rgba(16, 185, 129, 0.1);
            color: #10b981;
            padding: 8px 16px;
            border-radius: 20px;
            border: 1px solid #10b981;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .dot {
            width: 10px; height: 10px;
            background-color: #10b981;
            border-radius: 50%;
            animation: pulse 1.5s infinite;
        }
        @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
            70% { box-shadow: 0 0 0 10px rgba(16, 185, 129, 0); }
            100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
        }
        .grid-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(600px, 1fr));
            gap: 30px;
            justify-content: center;
        }
        .video-card {
            background: rgba(30, 41, 59, 0.7);
            backdrop-filter: blur(10px);
            padding: 20px;
            border-radius: 16px;
            border: 1px solid rgba(255, 255, 255, 0.05);
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.5);
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        .video-card h3 {
            margin: 0 0 15px 0;
            font-size: 1.3rem;
            color: #94a3b8;
            width: 100%;
            text-align: left;
            border-left: 4px solid #3b82f6;
            padding-left: 10px;
        }
        .video-card.yolo h3 { border-left-color: #f59e0b; }
        .video-card.barcode h3 { border-left-color: #10b981; }
        .stream-img {
            border-radius: 8px;
            border: 1px solid #334155;
            background-color: #000;
            max-width: 100%;
            height: auto;
        }
        .highlight .stream-img {
            border: 2px solid #22c55e;
            box-shadow: 0 0 20px rgba(34, 197, 94, 0.2);
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🤖 AI Vision Control Center</h1>
        <div class="status-badge"><div class="dot"></div>SYSTEM ONLINE</div>
    </div>
    
    <div class="grid-container">
        <div class="video-card">
            <h3>AprilTag Pose Estimation</h3>
            <img src="/video/apriltag" width="640" class="stream-img">
        </div>
        
        <div class="video-card yolo">
            <h3>YOLO Microplate Detection</h3>
            <img src="/video/yolo" width="480" class="stream-img">
        </div>
        
        <div class="video-card barcode">
            <h3>Barcode ROI (Main)</h3>
            <img src="/video/barcode_main" width="640" class="stream-img">
        </div>
        
        <div class="video-card barcode highlight" style="justify-content: center;">
            <h3>Barcode Zoom & Decode</h3>
            <img src="/video/barcode_zoom" width="480" class="stream-img">
        </div>
    </div>
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
            frame = vision_sub.get_frame(target)
            if frame: 
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            time.sleep(0.1)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


# ==========================================
# 4. Main Entry Point (Process 분기)
# ==========================================
if __name__ == "__main__":
    # 1. 카메라 연산을 백그라운드 프로세스로 분리 실행
    # (Picamera2는 메인 프로세스와 분리되어야 리소스 충돌이 방지됩니다)
    cam_process = multiprocessing.Process(target=run_publisher, daemon=True)
    cam_process.start()

    # Shared Memory가 생성될 시간을 약간 확보
    time.sleep(2.0)

    # 2. 비전 분석 Subscriber 초기화 및 스레드 실행
    yolo_path = '/home/rnd/yolo_model/best_yolov26n.onnx'
    vision_sub = IntegratedVisionSubscriber(yolo_model_path=yolo_path)
    
    t_fhd = threading.Thread(target=vision_sub.loop_fhd_tasks, daemon=True)
    t_vga = threading.Thread(target=vision_sub.loop_vga_tasks, daemon=True)
    t_fhd.start()
    t_vga.start()
    
    # 3. Flask 웹 서버 실행 (블로킹 함수이므로 맨 마지막에 호출)
    print("\n✅ 모든 시스템이 시작되었습니다. 웹 브라우저에서 서버 IP의 5000 포트로 접속하세요.\n")
    app.run(host="0.0.0.0", port=5000)