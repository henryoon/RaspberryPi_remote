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

# 💡 1차원 칼만 필터 클래스 추가
class KalmanFilter1D:
    def __init__(self, process_noise=1e-3, measurement_noise=0.3):
        """
        process_noise (Q): 시스템 자체의 변화율. 작을수록 이전 값을 신뢰 (반응 느림, 부드러움)
        measurement_noise (R): 측정값의 노이즈. 클수록 현재 측정값을 불신 (스무딩 강함)
        """
        self.x = 0.0  # 상태 추정값
        self.p = 1.0  # 추정 오차 공분산
        self.q = process_noise
        self.r = measurement_noise
        self.initialized = False

    def update(self, measurement):
        if not self.initialized:
            self.x = measurement
            self.initialized = True
            return self.x

        # 1. 예측 (Prediction)
        self.p = self.p + self.q

        # 2. 업데이트 (Update)
        k = self.p / (self.p + self.r) # 칼만 이득 (Kalman Gain)
        self.x = self.x + k * (measurement - self.x)
        self.p = (1 - k) * self.p

        return self.x

class SingleAprilTagVision:
    def __init__(self):
        self.lock = threading.Lock()
        self.output_frame = None 
        self.running = True
        
        # 카메라 해상도 및 태그 설정
        self.width, self.height = 1280, 720
        self.tag_size = 0.03 # 30mm
        self.half_s = self.tag_size / 2.0
        
        # 카메라 내부 파라미터
        focal_length_x = 1522.36
        focal_length_y = 1520.01
        center_x = 612.83
        center_y = 370.20
        self.camera_matrix = np.array([
            [focal_length_x, 0, center_x],
            [0, focal_length_y, center_y],
            [0, 0, 1]
        ], dtype=np.float32)
        self.dist_coeffs = np.array([[-0.040964, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        
        # 태그 좌표계 정의
        self.obj_points = np.array([
            [-self.half_s, -self.half_s, 0],
            [ self.half_s, -self.half_s, 0],
            [ self.half_s,  self.half_s, 0],
            [-self.half_s,  self.half_s, 0] 
        ], dtype=np.float32)

        # AprilTag 디텍터 설정
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36H11)
        parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

        # 💡 태그 ID별로 칼만 필터를 관리하기 위한 딕셔너리
        self.kf_dict = {}

        self.picam2 = Picamera2()
        self._setup_camera()

    def _setup_camera(self):
        print("📷 Camera Module 3 초기화 (1280x720)...")
        config = self.picam2.create_video_configuration(
            main={"size": (self.width, self.height), "format": "BGR888"}
        )
        self.picam2.configure(config)
        self.picam2.start()
        self._set_af_mode(continuous=True)

    def _set_af_mode(self, continuous=True):
        try:
            if continuous:
                self.picam2.set_controls({"AfMode": 2})
                time.sleep(1.0) 
                self.picam2.set_controls({"AfMode": 0, "LensPosition": 5.5})
                status = "초점 보정 후 고정 완료 (LensPosition: 5.5)"
            else:
                self.picam2.set_controls({"AfMode": 0, "LensPosition": 5.5})
                status = "고정 모드 유지"
            print(f"🔄 {status}")
        except Exception as e:
            print(f"⚠️ AF 오류: {e}")

    def draw_coordinate_legend(self, frame):
        origin = (40, 40)
        length = 60
        
        x_end = (origin[0] + length, origin[1])
        cv2.arrowedLine(frame, origin, x_end, (0, 0, 255), 3, tipLength=0.2)
        cv2.putText(frame, "X", (x_end[0] + 10, x_end[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        y_end = (origin[0], origin[1] + length)
        cv2.arrowedLine(frame, origin, y_end, (0, 255, 0), 3, tipLength=0.2)
        cv2.putText(frame, "Y", (y_end[0] - 10, y_end[1] + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

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
                    tag_id = ids[i][0]
                    img_points = corners[i][0]

                    # 💡 새로운 태그가 발견되면 필터 세트 생성
                    if tag_id not in self.kf_dict:
                        self.kf_dict[tag_id] = {
                            'x': KalmanFilter1D(1e-3, 0.1),
                            'y': KalmanFilter1D(1e-3, 0.1),
                            'z': KalmanFilter1D(1e-3, 0.1),
                            'r': KalmanFilter1D(1e-3, 0.5), # 각도는 더 민감하게 튈 수 있으므로 R값을 크게 줌
                            'p': KalmanFilter1D(1e-3, 0.5),
                            'yw': KalmanFilter1D(1e-3, 0.5)
                        }

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

                        # Raw Data (노이즈가 있는 원본)
                        raw_cam_x = camera_pos[0][0] * 1000
                        raw_cam_y = camera_pos[1][0] * 1000
                        raw_cam_z = -camera_pos[2][0] * 1000

                        proj_matrix = np.hstack((R, tvec))
                        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)
                        
                        raw_pitch = euler_angles[0][0]
                        raw_yaw = euler_angles[1][0]
                        raw_roll = euler_angles[2][0]

                        # 💡 칼만 필터 업데이트 (스무딩 처리)
                        kf = self.kf_dict[tag_id]
                        cam_x = kf['x'].update(raw_cam_x)
                        cam_y = kf['y'].update(raw_cam_y)
                        cam_z = kf['z'].update(raw_cam_z)
                        
                        roll = kf['r'].update(raw_roll)
                        pitch = kf['p'].update(raw_pitch)
                        yaw = kf['yw'].update(raw_yaw)

                        # 화면에 필터링된 좌표 출력
                        coord_text = f"Cam Pos -> X: {cam_x:.0f}, Y: {cam_y:.0f}, Z: {cam_z:.0f} mm"
                        angle_text = f"Angle -> R: {roll:.0f}, P: {pitch:.0f}, Y: {yaw:.0f} deg"
                        
                        text_y = self.height - 50 - (i * 60)
                        cv2.putText(frame, coord_text, (20, text_y), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        cv2.putText(frame, angle_text, (20, text_y + 25), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

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
        <h2>🎯 AprilTag 30mm - Kalman Filtered</h2>
        <img src="/video_feed" width="800" style="border: 2px solid #00ffff;">
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