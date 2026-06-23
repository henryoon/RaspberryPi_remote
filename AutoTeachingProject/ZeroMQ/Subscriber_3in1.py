import time
import threading
import cv2
import numpy as np
import zmq
import json
from multiprocessing import shared_memory
from flask import Flask, Response, render_template_string
from pyzbar import pyzbar
from pupil_apriltags import Detector
from ultralytics import YOLO


class KalmanFilter1D:
    def __init__(self, process_noise=1e-3, measurement_noise=0.3):
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
    """공유 메모리를 읽어 AprilTag, Barcode(FHD), YOLO(VGA)를 동시 수행하는 통합 클래스"""
    
    def __init__(self, yolo_model_path: str):
        self.lock = threading.Lock()
        self.running = True

        # 웹 스트리밍 캐싱용 (가장 최근 압축된 프레임만 유지)
        self.apriltag_jpg = None
        self.barcode_main_jpg = None
        self.barcode_zoom_jpg = None
        self.yolo_jpg = None

        # --- 공유 메모리 연결 (Publisher와 동일한 이름 사용) ---
        print("🔗 Shared Memory 연결 대기 중...")
        while True:
            try:
                self.shm_fhd = shared_memory.SharedMemory(name="vision_fhd")
                self.shm_vga = shared_memory.SharedMemory(name="vision_vga")
                break
            except FileNotFoundError:
                time.sleep(1) # Publisher가 켜질 때까지 대기

        self.fhd_buffer = np.ndarray((1080, 1920, 3), dtype=np.uint8, buffer=self.shm_fhd.buf)
        self.vga_buffer = np.ndarray((480, 640, 3), dtype=np.uint8, buffer=self.shm_vga.buf)

        # --- ZMQ 퍼블리셔 설정 ---
        self.pub_context = zmq.Context()
        self.tag_pub = self.pub_context.socket(zmq.PUB)
        self.tag_pub.bind("tcp://*:5557")
        self.barcode_pub = self.pub_context.socket(zmq.PUB)
        self.barcode_pub.bind("tcp://*:5558")

        # --- AprilTag 설정 ---
        self.tag_w, self.tag_h = 1280, 720
        self.tag_size = 0.03
        self.half_s = self.tag_size / 2.0
        scale_x = self.tag_w / 1920.0
        scale_y = self.tag_h / 1080.0
        self.camera_matrix = np.array(
            [[1522.36 * (1920/1280) * scale_x, 0, 612.83 * (1920/1280) * scale_x], 
             [0, 1520.01 * (1080/720) * scale_y, 370.20 * (1080/720) * scale_y], 
             [0, 0, 1]], dtype=np.float32
        )
        self.dist_coeffs = np.array([[-0.040964, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        self.obj_points = np.array([[-self.half_s, -self.half_s, 0], [self.half_s, -self.half_s, 0],
                                    [self.half_s, self.half_s, 0], [-self.half_s, self.half_s, 0]], dtype=np.float32)
        self.detector = Detector(families='tag36h11', nthreads=1, quad_decimate=2.0, debug=0)
        self.kf_dict = {}

        # --- Barcode 설정 ---
        self.bc_roi_x, self.bc_roi_y = 400, 120
        self.bc_scale = 3

        # --- YOLO 설정 ---
        print(f"⚙️ YOLO 모델 로딩 중... ({yolo_model_path})")
        self.yolo_model = YOLO(yolo_model_path, task='detect')
        self.target_roi = (206, 212, 434, 268)

        print("🚀 통합 비전 시스템 (SHM) 초기화 완료")

    def loop_fhd_tasks(self):
        """스레드 1: Barcode 및 AprilTag 처리 (목표 8 FPS)"""
        frame_time = 1.0 / 8.0
        while self.running:
            loop_start = time.time()
            
            # 💡 공유 메모리에서 원본 프레임 복사 (안전한 처리를 위해 copy 필수)
            frame_fhd = self.fhd_buffer.copy()

            # 1. Barcode 처리
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

            # 2. AprilTag 처리
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

            # 웹 송출용 인코딩 및 캐싱
            _, bc_main_jpg = cv2.imencode(".jpg", display_bc_main)
            _, bc_zoom_jpg = cv2.imencode(".jpg", zoomed_roi)
            _, tag_jpg = cv2.imencode(".jpg", frame_tag)

            with self.lock:
                self.barcode_main_jpg = bytearray(bc_main_jpg)
                self.barcode_zoom_jpg = bytearray(bc_zoom_jpg)
                self.apriltag_jpg = bytearray(tag_jpg)

            elapsed = time.time() - loop_start
            time.sleep(max(0.01, frame_time - elapsed))

    def loop_vga_tasks(self):
        """스레드 2: YOLO 객체 인식 처리 (목표 10 FPS)"""
        frame_time = 1.0 / 10.0
        while self.running:
            loop_start = time.time()
            
            # 공유 메모리에서 VGA 프레임 복사
            frame_vga = self.vga_buffer.copy()

            results = self.yolo_model(frame_vga, conf=0.4, verbose=False, stream=True)
            annotated_frame = frame_vga.copy()
            roi_x1, roi_y1, roi_x2, roi_y2 = self.target_roi
            cv2.rectangle(annotated_frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 255, 255), 1)

            is_in_roi = False
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    
                    if (roi_x1 <= cx <= roi_x2) and (roi_y1 <= cy <= roi_y2):
                        is_in_roi = True
                        color = (0, 255, 0)
                    else:
                        color = (0, 0, 255)
                    cv2.circle(annotated_frame, (cx, cy), 5, color, -1)

            status_text = "STATUS: DETECTED" if is_in_roi else "STATUS: NONE"
            status_color = (0, 255, 0) if is_in_roi else (0, 0, 255)
            cv2.putText(annotated_frame, status_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)

            _, yolo_jpg = cv2.imencode(".jpg", annotated_frame)
            with self.lock:
                self.yolo_jpg = bytearray(yolo_jpg)

            elapsed = time.time() - loop_start
            time.sleep(max(0.01, frame_time - elapsed))

    def get_frame(self, target):
        with self.lock:
            if target == "apriltag": return self.apriltag_jpg
            elif target == "barcode_main": return self.barcode_main_jpg
            elif target == "barcode_zoom": return self.barcode_zoom_jpg
            elif target == "yolo": return self.yolo_jpg
            return None

    def release(self):
        self.running = False
        self.shm_fhd.close()
        self.shm_vga.close()
        self.tag_pub.close()
        self.barcode_pub.close()
        self.pub_context.term()


# --- 단일 Flask Web Server (통합 대시보드) ---
app = Flask(__name__)
# 실제 환경의 모델 경로로 변경해주세요
vision_sub = IntegratedVisionSubscriber(yolo_model_path='/home/rnd/yolo_model/best_yolov26n.onnx')

@app.route("/")
def index():
    return render_template_string("""
    <html>
      <head>
        <style>
          body {
            background-color: #111;
            color: #eee;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            text-align: center;
            margin: 0;
            padding: 20px;
          }
          h2 {
            margin-bottom: 30px;
            color: #fff;
            letter-spacing: 1px;
          }
          .grid-container {
            display: grid;
            grid-template-columns: repeat(2, auto);
            justify-content: center;
            gap: 40px; /* 각 화면 사이의 간격 */
          }
          .video-card {
            display: flex;
            flex-direction: column;
            align-items: center;
            background-color: #222;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.5);
          }
          .video-card h3 {
            margin-top: 0;
            margin-bottom: 15px;
            font-size: 16px;
            color: #aaa;
          }
          .stream-img {
            border: 2px solid #444;
            border-radius: 6px;
            background-color: #000; /* 로딩 전 빈 공간 처리 */
          }
          .stream-img.highlight {
            border-color: #0f0;
            box-shadow: 0 0 10px rgba(0, 255, 0, 0.2);
          }
        </style>
      </head>
      <body>
        <!-- <h2>Integrated Robot Vision Dashboard</h2> -->
        <div class="grid-container">
          
          <div class="video-card">
            <h3>AprilTag Pose (1280x720)</h3>
            <img src="/video/apriltag" width="640" class="stream-img">
          </div>
          
          <div class="video-card">
            <h3>YOLO Microplate (640x480)</h3>
            <img src="/video/yolo" width="480" class="stream-img">
          </div>
          
          <div class="video-card">
            <h3>Barcode Main (1920x1080)</h3>
            <img src="/video/barcode_main" width="640" class="stream-img">
          </div>
          
          <div class="video-card" style="justify-content: center;">
            <h3>Barcode Zoomed ROI</h3>
            <img src="/video/barcode_zoom" width="480" class="stream-img highlight">
          </div>
          
        </div>
      </body>
    </html>
    """)

@app.route("/video/<target>")
def video_feed(target):
    def gen():
        while True:
            frame = vision_sub.get_frame(target)
            if frame: 
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            time.sleep(0.1) # 10Hz 송출 제한
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    t_fhd = threading.Thread(target=vision_sub.loop_fhd_tasks)
    t_vga = threading.Thread(target=vision_sub.loop_vga_tasks)
    t_fhd.daemon = True
    t_vga.daemon = True
    t_fhd.start()
    t_vga.start()
    
    app.run(host="0.0.0.0", port=5000)