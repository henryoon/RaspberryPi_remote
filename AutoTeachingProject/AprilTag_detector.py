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

        # 3. GitHub 공식 문서 기준의 태그 좌표계
        # Origin(0,0,0)은 태그 중앙. 
        # X축: 오른쪽(+), Y축: 아래쪽(+), Z축: 태그 안쪽으로 들어가는 방향(+)
        self.obj_points = np.array([
            [-self.half_s, -self.half_s, 0], # Top-Left (X:음수, Y:음수)
            [ self.half_s, -self.half_s, 0], # Top-Right (X:양수, Y:음수)
            [ self.half_s,  self.half_s, 0], # Bottom-Right (X:양수, Y:양수)
            [-self.half_s,  self.half_s, 0]  # Bottom-Left (X:음수, Y:양수)
        ], dtype=np.float32)

        # AprilTag 36h11 디텍터 설정
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36H11)
        parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

        self.picam2 = Picamera2()
        self._setup_camera()

    def _setup_camera(self):
        print("📷 Camera Module 3 초기화 (1280x720, 연속 AF)...")
        config = self.picam2.create_video_configuration(
            main={"size": (self.width, self.height), "format": "BGR888"}
        )
        self.picam2.configure(config)
        self.picam2.start()
        self.picam2.set_controls({"AfMode": 2})

    def draw_coordinate_legend(self, frame):
        """좌측 상단에 고정된 X/Y 좌표계 화살표를 그립니다."""
        origin = (40, 40) # 화살표 시작점
        length = 60       # 화살표 길이
        
        # X축 화살표 (오른쪽 방향, 빨간색)
        x_end = (origin[0] + length, origin[1])
        cv2.arrowedLine(frame, origin, x_end, (0, 0, 255), 3, tipLength=0.2)
        cv2.putText(frame, "X", (x_end[0] + 10, x_end[1] + 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Y축 화살표 (아래 방향, 초록색)
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

            # 💡 오버레이: 프레임이 캡처될 때마다 좌측 상단에 축 범례 그리기
            self.draw_coordinate_legend(frame)

            # 태그 감지
            corners, ids, rejected = self.detector.detectMarkers(gray)

            if ids is not None and len(ids) > 0:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)

                for i in range(len(ids)):
                    img_points = corners[i][0]

                    # 태그 중앙점에 하얀 점, 카메라 중앙에 검은 점
                    cx = int(np.mean(img_points[:, 0]))
                    cy = int(np.mean(img_points[:, 1]))
                    cv2.circle(frame, (cx, cy), 6, (255, 255, 255), -1)
                    cv2.circle(frame, (self.width//2, self.height//2), 6, (0, 0, 0), -1)

                    # 카메라 자세 추정 (SQPNP 사용)
                    success, rvec, tvec = cv2.solvePnP(
                        self.obj_points, img_points, 
                        self.camera_matrix, self.dist_coeffs, 
                        flags=cv2.SOLVEPNP_SQPNP
                    )

                    if success:
                        # 태그 기준 카메라 좌표로 역변환
                        R, _ = cv2.Rodrigues(rvec)
                        camera_pos = -np.dot(R.T, tvec)

                        cam_x = - camera_pos[0][0] * 1000
                        cam_y = - camera_pos[1][0] * 1000 # 사용자 수정 반영
                        cam_z = camera_pos[2][0] * 1000

                        # 💡 텍스트 위치 변경: 좌측 하단 (Y좌표를 화면 높이 근처로 설정)
                        coord_text = f"Cam Pos -> X: {cam_x:.0f}, Y: {cam_y:.0f}, Z: {cam_z:.0f} mm"
                        
                        # 다수의 태그가 있을 경우를 대비하여 아래에서 위로 쌓이도록 오프셋 적용
                        text_y = self.height - 30 - (i * 40)
                        cv2.putText(frame, coord_text, (20, text_y), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                        
                        # 터미널 출력
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
        <p style="color: #aaaaaa; font-size: 0.9em;">(X: 오른쪽, Y: 아래쪽, Z: 태그 안쪽)</p>
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