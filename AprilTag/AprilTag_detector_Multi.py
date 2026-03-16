import time
import threading
import cv2
import numpy as np
from flask import Flask, Response, render_template_string

try:
    from picamera2 import Picamera2
except ImportError:
    print("오류: picamera2를 찾을 수 없습니다.")
    exit()

class SingleAprilTagVision:
    def __init__(self):
        self.lock = threading.Lock()
        self.output_frame = None 
        self.running = True
        
        # 1. 카메라 해상도 및 태그 설정
        self.width, self.height = 1280, 720
        self.tag_size = 0.03 # 3cm = 0.03m
        self.half_s = self.tag_size / 2.0
        
        # 2. 임시 카메라 내부 파라미터
        focal_length_x = 1530.0
        focal_length_y = 1530.0
        center_x = self.width / 2.0
        center_y = self.height / 2.0
        self.camera_matrix = np.array([
            [focal_length_x, 0, center_x],
            [0, focal_length_y, center_y],
            [0, 0, 1]
        ], dtype=np.float32)
        self.dist_coeffs = np.zeros((4,1))

        # 3. GitHub 공식 문서 기준 태그 좌표계
        self.obj_points = np.array([
            [-self.half_s, -self.half_s, 0], # Top-Left
            [ self.half_s, -self.half_s, 0], # Top-Right
            [ self.half_s,  self.half_s, 0], # Bottom-Right
            [-self.half_s,  self.half_s, 0]  # Bottom-Left
        ], dtype=np.float32)

        # AprilTag 36h11 디텍터 설정
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36H11)
        parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

        self.picam2 = Picamera2()
        self._setup_camera()

    def _setup_camera(self):
        print("📷 Camera Module 3 초기화 (1280x720)...")
        config = self.picam2.create_video_configuration(
            main={"size": (self.width, self.height), "format": "BGR888"}
        )
        self.picam2.configure(config)
        self.picam2.start()
        
        # 💡 수정된 부분: 초기화 직후 초점을 잡고 렌즈를 물리적으로 고정시킵니다.
        self._set_af_mode(continuous=True)

    def _set_af_mode(self, continuous=True):
        """초점 모드 제어: 렌즈 진동으로 인한 좌표 Jitter 방지"""
        try:
            if continuous:
                # 1. 먼저 연속 AF로 초점을 잡도록 명령
                self.picam2.set_controls({"AfMode": 2})
                # 2. 잠시 대기 (렌즈가 목표 위치로 이동할 시간 확보)
                time.sleep(1.0) 
                # 3. 특정 거리(17~18cm 대역)로 렌즈 모터 완벽 고정
                self.picam2.set_controls({"AfMode": 0, "LensPosition": 5.5})
                status = "초점 보정 후 고정 완료 (LensPosition: 5.5)"
            else:
                self.picam2.set_controls({"AfMode": 0, "LensPosition": 5.5})
                status = "고정 모드 유지"
            
            print(f"🔄 {status}")
        except Exception as e:
            print(f"⚠️ AF 오류: {e}")

    def draw_coordinate_legend(self, frame):
        """좌측 상단 고정 X/Y 축 범례"""
        origin = (40, 40)
        length = 60
        
        # X축 (빨간색)
        x_end = (origin[0] + length, origin[1])
        cv2.arrowedLine(frame, origin, x_end, (0, 0, 255), 3, tipLength=0.2)
        cv2.putText(frame, "X", (x_end[0] + 10, x_end[1] + 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Y축 (초록색)
        y_end = (origin[0], origin[1] + length)
        cv2.arrowedLine(frame, origin, y_end, (0, 255, 0), 3, tipLength=0.2)
        cv2.putText(frame, "Y", (y_end[0] - 10, y_end[1] + 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    def process_loop(self):
        while self.running:
            raw_frame = self.picam2.capture_array()
            if raw_frame is None: continue
            
            frame = cv2.cvtColor(raw_frame, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            self.draw_coordinate_legend(frame)

            corners, ids, rejected = self.detector.detectMarkers(gray)

            if ids is not None and len(ids) > 0:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)

                for i in range(len(ids)):
                    img_points = corners[i][0]

                    cx = int(np.mean(img_points[:, 0]))
                    cy = int(np.mean(img_points[:, 1]))
                    cv2.circle(frame, (cx, cy), 6, (255, 255, 255), -1)
                    cv2.circle(frame, (self.width//2, self.height//2), 6, (0, 0, 0), -1)

                    success, rvec, tvec = cv2.solvePnP(
                        self.obj_points, img_points, 
                        self.camera_matrix, self.dist_coeffs, 
                        flags=cv2.SOLVEPNP_SQPNP
                    )

                    if success:
                        R, _ = cv2.Rodrigues(rvec)
                        camera_pos = -np.dot(R.T, tvec)

                        cam_x = camera_pos[0][0] * 1000
                        cam_y = -camera_pos[1][0] * 1000
                        cam_z = camera_pos[2][0] * 1000

                        coord_text = f"Cam Pos -> X: {cam_x:.0f}, Y: {cam_y:.0f}, Z: {cam_z:.0f} mm"
                        
                        text_y = self.height - 30 - (i * 40)
                        cv2.putText(frame, coord_text, (20, text_y), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                        
                        print(f"[Tag {ids[i][0]}] {coord_text}")

            with self.lock:
                self.output_frame = frame

            time.sleep(0.01)

    def get_frame(self):
        with self.lock:
            if self.output_frame is None: return None
            _, buffer = cv2.imencode(".jpg", self.output_frame)
            return bytearray(buffer)

# --- Flask Server ---
app = Flask(__name__)
vision = SingleAprilTagVision()

@app.route("/")
def index():
    return render_template_string("""
    <html>
      <body style="background-color:#111; color:white; text-align:center; font-family: sans-serif;">
        <h2>🎯 AprilTag 30mm - Official Coordinate System</h2>
        <img src="/video_feed" width="800" style="border: 2px solid #00ffff;">
        <p>태그 중심을 원점으로 한 카메라의 실제 위치(mm)</p>
        <p style="color: #aaaaaa; font-size: 0.9em;">(상태: 카메라 렌즈 물리적 초점 고정 완료)</p>
      </body>
    </html>
    """)

@app.route("/video_feed")
def video_feed():
    def gen():
        while True:
            frame = vision.get_frame()
            if frame:
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.04)
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    t = threading.Thread(target=vision.process_loop)
    t.daemon = True
    t.start()
    app.run(host="0.0.0.0", port=5000)