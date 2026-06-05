import time
import threading
import cv2
import numpy as np
import zmq
import json
from flask import Flask, Response, render_template_string
from pyzbar import pyzbar
from pupil_apriltags import Detector


class ZMQReceiverThread_LazyFHD:
    """백그라운드에서 IPC 통신으로 수신 후 1번만 디코딩하여 메모리에 유지하는 클래스"""
    def __init__(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.set_hwm(2)
        # 💡 네트워크 오버헤드가 없는 IPC 통신 사용 (Publisher 코드와 동일한 경로)
        self.socket.connect("ipc:///tmp/vision_fhd")
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")

        self.latest_packet = None
        self.running = True
        self.lock = threading.Lock()

        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            try:
                packet = self.socket.recv()
                with self.lock:
                    self.latest_packet = packet
            except Exception as e:
                print(f"⚠️ ZMQ 수신 에러: {e}")
                break

    def read_and_decode(self):
        with self.lock:
            packet = self.latest_packet
            self.latest_packet = None 

        if packet is not None:
            np_arr = np.frombuffer(packet, dtype=np.uint8)
            return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return None

    def stop(self):
        self.running = False
        self.thread.join()
        self.socket.close()
        self.context.term()


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


class UnifiedVisionSubscriber:
    """AprilTag와 Barcode 인식을 단일 프레임 공유 및 캐싱 웹 스트리밍으로 통합한 클래스"""
    def __init__(self):
        self.lock = threading.Lock()
        self.running = True

        # 💡 웹 스트리밍 캐싱용 (압축된 Byte 데이터 보관)
        self.apriltag_frame_bytes = None
        self.barcode_main_frame_bytes = None
        self.barcode_zoom_frame_bytes = None

        # --- 통합 속도 설정 ---
        self.target_fps = 8.0  # CPU 부하를 고려한 최적의 FPS (8Hz)
        self.frame_time = 1.0 / self.target_fps
        self.receiver = ZMQReceiverThread_LazyFHD()

        # --- ZMQ 퍼블리셔 설정 (로봇 제어기 송신용 TCP) ---
        self.pub_context = zmq.Context()
        
        # AprilTag 퍼블리셔 (5557 포트)
        self.tag_pub = self.pub_context.socket(zmq.PUB)
        self.tag_pub.set_hwm(10)
        self.tag_pub.bind("tcp://*:5557")
        
        # Barcode 퍼블리셔 (5558 포트)
        self.barcode_pub = self.pub_context.socket(zmq.PUB)
        self.barcode_pub.set_hwm(10)
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
        self.obj_points = np.array(
            [[-self.half_s, -self.half_s, 0], [self.half_s, -self.half_s, 0],
             [self.half_s, self.half_s, 0], [-self.half_s, self.half_s, 0]], dtype=np.float32
        )
        
        # quad_decimate=2.0으로 연산량 대폭 감소
        self.detector = Detector(families='tag36h11', nthreads=1, quad_decimate=2.0, debug=0)
        self.kf_dict = {}

        # --- Barcode 설정 ---
        self.bc_roi_x, self.bc_roi_y = 400, 120
        self.bc_scale = 3
        self.is_focus_locked = False
        self.last_bc_time = 0
        self.lock_duration = 30

        print("🚀 통합 비전 시스템(AprilTag + Barcode) 초기화 완료")

    def process_loop(self):
        print(f"📥 통합 추론 루프 시작 (목표 FPS: {self.target_fps})")
        while self.running:
            loop_start = time.time()

            # 💡 Lazy Decoding: 한 번만 원본(1920x1080) 복원
            frame_fhd = self.receiver.read_and_decode()
            if frame_fhd is None:
                time.sleep(0.01)
                continue

            # ==========================================
            # 1. Barcode 처리 (원본 FHD 기반)
            # ==========================================
            h, w, _ = frame_fhd.shape
            x1, y1 = (w - self.bc_roi_x) // 2, (h - self.bc_roi_y) // 2 + 30
            x2, y2 = x1 + self.bc_roi_x, y1 + self.bc_roi_y

            roi_img = frame_fhd[y1:y2, x1:x2]
            zoomed_roi = cv2.resize(roi_img, (self.bc_roi_x * self.bc_scale, self.bc_roi_y * self.bc_scale), interpolation=cv2.INTER_CUBIC)

            decoded = pyzbar.decode(zoomed_roi)
            current_time = time.time()
            detected_barcodes = []

            # 바코드 감지 여부에 따른 ROI 색상 동적 변경
            if len(decoded) > 0:
                self.last_bc_time = current_time
                self.is_focus_locked = True
                roi_color = (0, 255, 0)
            else:
                if self.is_focus_locked and (current_time - self.last_bc_time >= self.lock_duration):
                    self.is_focus_locked = False
                roi_color = (255, 255, 255)

            display_bc_main = frame_fhd.copy()
            cv2.rectangle(display_bc_main, (x1, y1), (x2, y2), roi_color, 2)

            for obj in decoded:
                data = obj.data.decode("utf-8")
                detected_barcodes.append(data)
                cv2.putText(zoomed_roi, f"DATA: {data}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Barcode 데이터 ZMQ Publish
            if detected_barcodes:
                payload = json.dumps({"timestamp": current_time, "barcodes": detected_barcodes})
                self.barcode_pub.send_string(payload)

            # ==========================================
            # 2. AprilTag 처리 (1280x720 리사이징 기반)
            # ==========================================
            frame_tag = cv2.resize(frame_fhd, (self.tag_w, self.tag_h), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(frame_tag, cv2.COLOR_BGR2GRAY)
            tags = self.detector.detect(gray, estimate_tag_pose=False)
            detected_tags_data = []

            for i, tag in enumerate(tags):
                tag_id = tag.tag_id
                img_points = tag.corners

                corners_int = img_points.astype(int)
                for j in range(4):
                    cv2.line(frame_tag, tuple(corners_int[j]), tuple(corners_int[(j+1)%4]), (0, 255, 0), 2)
                
                cx, cy = int(tag.center[0]), int(tag.center[1])
                cv2.circle(frame_tag, (cx, cy), 6, (255, 255, 255), -1)

                if tag_id not in self.kf_dict:
                    self.kf_dict[tag_id] = {
                        "x": KalmanFilter1D(), "y": KalmanFilter1D(), "z": KalmanFilter1D(),
                        "r": KalmanFilter1D(1e-3, 0.5), "p": KalmanFilter1D(1e-3, 0.5), "yw": KalmanFilter1D(1e-3, 0.5),
                    }

                success, rvec, tvec = cv2.solvePnP(self.obj_points, img_points, self.camera_matrix, self.dist_coeffs, flags=cv2.SOLVEPNP_SQPNP)
                
                if success:
                    R, _ = cv2.Rodrigues(rvec)
                    camera_pos = -np.dot(R.T, tvec)
                    raw_cam_x, raw_cam_y, raw_cam_z = camera_pos[0][0]*1000, camera_pos[1][0]*1000, -camera_pos[2][0]*1000
                    
                    proj_matrix = np.hstack((R, tvec))
                    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)
                    
                    kf = self.kf_dict[tag_id]
                    cam_x, cam_y, cam_z = kf["x"].update(raw_cam_x), kf["y"].update(raw_cam_y), kf["z"].update(raw_cam_z)
                    
                    detected_tags_data.append({
                        "id": int(tag_id), "x": round(float(cam_x), 2), "y": round(float(cam_y), 2), "z": round(float(cam_z), 2)
                    })

                    coord_text = f"ID:{tag_id} | X: {cam_x:.0f}, Y: {cam_y:.0f}, Z: {cam_z:.0f}"
                    text_y = self.tag_h - 50 - (i * 60)
                    cv2.putText(frame_tag, coord_text, (20, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # AprilTag 좌표 데이터 ZMQ Publish
            if detected_tags_data:
                payload = json.dumps({"tags": detected_tags_data})
                self.tag_pub.send_string(payload)

            # ==========================================
            # 3. 💡 웹 스트리밍용 JPEG 캐싱 (CPU 부하 방지)
            # ==========================================
            _, bc_main_jpg = cv2.imencode(".jpg", display_bc_main)
            _, bc_zoom_jpg = cv2.imencode(".jpg", zoomed_roi)
            _, tag_jpg = cv2.imencode(".jpg", frame_tag)

            with self.lock:
                # 완성된 JPEG 바이트만 덮어씀
                self.barcode_main_frame_bytes = bytearray(bc_main_jpg)
                self.barcode_zoom_frame_bytes = bytearray(bc_zoom_jpg)
                self.apriltag_frame_bytes = bytearray(tag_jpg)

            # 지능형 Sleep (목표 FPS 유지)
            elapsed_time = time.time() - loop_start
            sleep_duration = max(0.01, self.frame_time - elapsed_time)
            time.sleep(sleep_duration)

    def get_frame(self, target):
        """웹 브라우저가 프레임을 요청하면, 이미 압축된 바이트 데이터만 즉시 반환"""
        with self.lock:
            if target == "apriltag":
                return self.apriltag_frame_bytes
            elif target == "barcode_main":
                return self.barcode_main_frame_bytes
            elif target == "barcode_zoom":
                return self.barcode_zoom_frame_bytes
            return None

    def release(self):
        self.running = False
        self.receiver.stop()
        self.tag_pub.close()
        self.barcode_pub.close()
        self.pub_context.term()


# --- Flask Web Server ---
app = Flask(__name__)
vision_sub = UnifiedVisionSubscriber()

@app.route("/")
def index():
    return render_template_string("""
    <html>
      <body style="background-color:#111; color:white; text-align:center; font-family:sans-serif;">
        <h2>Unified Vision System (AprilTag & Barcode)</h2>
        <div style="display:flex; justify-content:center; gap:20px; margin-bottom:20px;">
          <div>
            <h3>AprilTag Pose (1280x720)</h3>
            <img src="/video/apriltag" width="640" style="border: 1px solid #444;">
          </div>
          <div>
            <h3>Barcode Main (1920x1080)</h3>
            <img src="/video/barcode_main" width="640" style="border: 1px solid #444;">
          </div>
        </div>
        <div>
          <h3>Barcode Zoomed ROI</h3>
          <img src="/video/barcode_zoom" width="480" style="border: 1px solid #0f0;">
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
            
            # 💡 웹 송출 속도를 추론 속도와 맞춰 0.1초(10Hz)로 제한하여 네트워크 및 CPU 과부하 방지
            time.sleep(0.1)
            
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    t = threading.Thread(target=vision_sub.process_loop)
    t.daemon = True
    t.start()
    
    # 통합된 단일 웹 서버 포트 (5000)
    app.run(host="0.0.0.0", port=5000)