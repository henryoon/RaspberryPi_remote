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

class DualAprilTagVision:
    def __init__(self):
        self.lock = threading.Lock()
        self.output_frame = None 
        self.running = True
        
        # 1. 태그 규격 및 배치 설정 (Meter 단위)
        self.width, self.height = 1920, 1080
        self.tag_size = 0.030         # 30mm AprilTag
        self.tag_spacing = 0.115      # 태그 간격 115mm (중심 대 중심)
        half_s = self.tag_size / 2.0
        
        # 2. FHD 카메라 내부 파라미터 (제공된 값 유지)
        focal_length_x = 1421.35
        focal_length_y = 1420.57
        center_x = 951.42
        center_y = 551.88

        self.camera_matrix = np.array([
            [focal_length_x, 0, center_x],
            [0, focal_length_y, center_y],
            [0, 0, 1]
        ], dtype=np.float32)
        self.dist_coeffs = np.array([[-0.040964, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        
        # 3. 3D 공간 상의 복합 태그 좌표 정의 (원점: Tag ID 0의 중심)
        # Tag 0 (원점 중심)
        obj_p0 = np.array([
            [-half_s,  half_s, 0],  # 좌상
            [ half_s,  half_s, 0],  # 우상
            [ half_s, -half_s, 0],  # 우하
            [-half_s, -half_s, 0]   # 좌하
        ], dtype=np.float32)
        
        # Tag 1 (X축으로 +115mm 이동한 중심)
        obj_p1 = np.array([
            [self.tag_spacing - half_s,  half_s, 0],
            [self.tag_spacing + half_s,  half_s, 0],
            [self.tag_spacing + half_s, -half_s, 0],
            [self.tag_spacing - half_s, -half_s, 0]
        ], dtype=np.float32)
        
        # 딕셔너리로 관리하여 검출된 ID에 맞게 매핑
        self.obj_points_dict = {0: obj_p0, 1: obj_p1}

        # ArUco / AprilTag 디텍터 설정
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36H11)
        parameters = cv2.aruco.DetectorParameters()
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        parameters.cornerRefinementWinSize = 6
        self.detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

        self.picam2 = Picamera2()
        self._setup_camera()

    def _setup_camera(self):
        print("📷 Dual Tag 고정밀 FHD 모드 초기화...")
        config = self.picam2.create_video_configuration(
            main={"size": (self.width, self.height), "format": "BGR888"}
        )
        self.picam2.configure(config)
        self.picam2.start()
        self.picam2.set_controls({"AfMode": 0, "LensPosition": 5.5})

    def process_loop(self):
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 150, 0.001)

        while self.running:
            raw_frame = self.picam2.capture_array()
            if raw_frame is None: continue
            
            frame = cv2.cvtColor(raw_frame, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            corners, ids, rejected = self.detector.detectMarkers(gray)

            # 점들을 담을 리스트 초기화
            valid_obj_pts = []
            valid_img_pts = []

            if ids is not None:
                for i in range(len(ids)):
                    tag_id = ids[i][0]
                    if tag_id in self.obj_points_dict:
                        # corners[i]의 shape는 (1, 4, 2)이므로 [0]을 취해 (4, 2)로 만듭니다.
                        img_pts_2d = corners[i][0]
                        
                        # 서브픽셀 정밀화 (입력은 반드시 (4, 2) 형태여야 합니다)
                        refined_corners = cv2.cornerSubPix(gray, img_pts_2d, (6, 6), (-1, -1), criteria)
                        
                        # 각 태그의 3D, 2D 좌표(4개씩)를 리스트에 추가
                        valid_obj_pts.append(self.obj_points_dict[tag_id])
                        valid_img_pts.append(refined_corners)
                        
                        # 시각화
                        cv2.aruco.drawDetectedMarkers(frame, [corners[i]], np.array([[tag_id]]))

                # 최소 1개 이상의 태그(점 4개 이상)가 검출되었을 때만 실행
                if len(valid_obj_pts) > 0:
                    # 리스트에 담긴 배열들을 수직으로 쌓아 (N, 3) 및 (N, 2) 구조로 변환
                    combined_obj_pts = np.vstack(valid_obj_pts).astype(np.float32)
                    combined_img_pts = np.vstack(valid_img_pts).astype(np.float32)

                    # 🔥 OpenCV solvePnP가 가장 좋아하는 (N, 1, 3), (N, 1, 2) 구조로 명시적 변형
                    combined_obj_pts = combined_obj_pts.reshape(-1, 1, 3)
                    combined_img_pts = combined_img_pts.reshape(-1, 1, 2)

                    # 태그 개수에 따른 알고리즘 플래그 선택
                    pnp_flag = cv2.SOLVEPNP_SQPNP if len(valid_obj_pts) > 1 else cv2.SOLVEPNP_IPPE_SQUARE
                    
                    # 1차 Pose Estimation
                    success, rvec, tvec = cv2.solvePnP(
                        combined_obj_pts, combined_img_pts, 
                        self.camera_matrix, self.dist_coeffs, 
                        flags=pnp_flag
                    )

                    if success:
                        # 2차 Levenberg-Marquardt 고정밀 최적화
                        rvec, tvec = cv2.solvePnPRefineLM(
                            combined_obj_pts, combined_img_pts,
                            self.camera_matrix, self.dist_coeffs,
                            rvec, tvec,
                            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-6)
                        )

                        # mm 단위 변환
                        x_mm = tvec[0][0] * 1000
                        y_mm = tvec[1][0] * 1000
                        z_mm = tvec[2][0] * 1000

                        # Tag 0 위치에 기준 좌표축 그리기 (축 길이 2cm)
                        cv2.drawFrameAxes(frame, self.camera_matrix, self.dist_coeffs, rvec, tvec, 0.02)
                        
                        # 화면 출력
                        info_text = f"Tag0 Pos -> X:{x_mm:.2f} Y:{y_mm:.2f} Z:{z_mm:.2f} mm"
                        detected_count_text = f"Tracked Tags: {len(valid_obj_pts)}/2"
                        
                        cv2.putText(frame, info_text, (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3)
                        cv2.putText(frame, detected_count_text, (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                        
                        if len(valid_obj_pts) == 2:
                            print(f"[Dual Mode] Tag0 X: {x_mm:6.2f}, Y: {y_mm:6.2f}, Z: {z_mm:6.2f}")

            center = (self.width // 2, self.height // 2)
            cv2.circle(frame, center, 5, (0, 255, 255), -1)

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
vision = DualAprilTagVision()

@app.route("/")
def index():
    return render_template_string("""
    <html>
      <body style="background-color:#111; color:white; text-align:center; font-family: sans-serif;">
        <h2>🎯 Dual AprilTag Precision System (Tag0 Center Origin)</h2>
        <img src="/video_feed" width="960" style="border: 2px solid #00ff00;">
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