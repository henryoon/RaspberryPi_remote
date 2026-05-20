import time
import threading
import cv2
import numpy as np
from flask import Flask, Response, render_template_string, jsonify

try:
    from picamera2 import Picamera2
except ImportError:
    print("오류: picamera2를 찾을 수 없습니다.")
    exit()

class WebCameraCalibrator:
    def __init__(self):
        self.lock = threading.Lock()
        self.output_frame = None 
        self.current_gray = None
        self.current_corners = None
        self.is_board_detected = False
        self.running = True
        
        # 1. 체커보드 환경 설정 (가로 8, 세로 6, 칸 크기 0.008m)
        self.CHESSBOARD_SIZE = (8, 6) 
        self.SQUARE_SIZE = 0.008
        
        self.objp = np.zeros((self.CHESSBOARD_SIZE[0] * self.CHESSBOARD_SIZE[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:self.CHESSBOARD_SIZE[0], 0:self.CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
        self.objp *= self.SQUARE_SIZE

        self.objpoints = [] 
        self.imgpoints = [] 
        self.captured_count = 0
        
        # 카메라 해상도
        # self.width, self.height = 1280, 720
        self.width, self.height = 1920, 1080
        self.picam2 = Picamera2()
        self._setup_camera()

    def _setup_camera(self):
        print("📷 Camera Module 3 초기화 중...")
        config = self.picam2.create_video_configuration(
            main={"size": (self.width, self.height), "format": "BGR888"}
        )
        self.picam2.configure(config)
        self.picam2.start()
        
        # 💡 핵심 수정: 실행 즉시 자동 초점을 끄고 LensPosition을 5.5로 강제 고정합니다.
        # 이렇게 하면 캘리브레이션 내내 렌즈가 절대 움직이지 않아 정확한 fx, fy를 얻을 수 있습니다.
        self.picam2.set_controls({"AfMode": 0, "LensPosition": 5.5})
        print("🔒 초점 고정 완료 (LensPosition: 5.5). 렌즈 구동 모터가 정지되었습니다.")

    def process_loop(self):
        while self.running:
            raw_frame = self.picam2.capture_array()
            if raw_frame is None: continue
            
            frame = cv2.cvtColor(raw_frame, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            display_frame = frame.copy()

            # 체커보드 찾기
            ret, corners = cv2.findChessboardCorners(
                gray, self.CHESSBOARD_SIZE, 
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
            )

            # UI 오버레이
            if ret:
                cv2.drawChessboardCorners(display_frame, self.CHESSBOARD_SIZE, corners, ret)
                cv2.putText(display_frame, "Ready to Capture!", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(display_frame, "Searching...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
            cv2.putText(display_frame, f"Captured: {self.captured_count} / 15", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            with self.lock:
                self.output_frame = display_frame
                self.current_gray = gray
                self.current_corners = corners
                self.is_board_detected = ret

            time.sleep(0.01)

    def capture_frame(self):
        with self.lock:
            if self.is_board_detected and self.current_gray is not None:
                # 💡 서브픽셀 정밀화: 코너 좌표의 소수점 단위 오차까지 보정합니다.
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners2 = cv2.cornerSubPix(self.current_gray, self.current_corners, (11, 11), (-1, -1), criteria)
                
                self.objpoints.append(self.objp)
                self.imgpoints.append(corners2)
                self.captured_count += 1
                return True, self.captured_count
            return False, self.captured_count

    def calculate_calibration(self):
        if self.captured_count < 10:
            return False, f"데이터가 부족합니다. 최소 10장 (현재: {self.captured_count}장) 필요합니다."

        print("🔄 캘리브레이션 연산 중...")
        image_size = (self.width, self.height)
        ret, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(
            self.objpoints, self.imgpoints, image_size, None, None
        )

        if ret:
            mean_error = 0
            for i in range(len(self.objpoints)):
                imgpoints2, _ = cv2.projectPoints(self.objpoints[i], rvecs[i], tvecs[i], cameraMatrix, distCoeffs)
                error = cv2.norm(self.imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
                mean_error += error
            mean_error = mean_error / len(self.objpoints)

            fx, fy = cameraMatrix[0, 0], cameraMatrix[1, 1]
            cx, cy = cameraMatrix[0, 2], cameraMatrix[1, 2]
            dist_str = ", ".join([f"{x[0]:.6f}" for x in distCoeffs])
            
            result_text = f"""# --- 재투영 오차(Error): {mean_error:.4f} px ---
# 이 코드를 복사하여 AprilTag 클래스의 __init__ 에 붙여넣으세요.

focal_length_x = {fx:.2f}
focal_length_y = {fy:.2f}
center_x = {cx:.2f}
center_y = {cy:.2f}

self.camera_matrix = np.array([
    [focal_length_x, 0, center_x],
    [0, focal_length_y, center_y],
    [0, 0, 1]
], dtype=np.float32)

self.dist_coeffs = np.array([[{dist_str}]], dtype=np.float32)
"""
            return True, result_text
        else:
            return False, "연산에 실패했습니다. 이미지를 다시 캡처해 주세요."

    def get_frame(self):
        with self.lock:
            if self.output_frame is None: return None
            _, buffer = cv2.imencode(".jpg", self.output_frame)
            return bytearray(buffer)

# --- Flask Web Server ---
app = Flask(__name__)
calibrator = WebCameraCalibrator()

html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Web Camera Calibrator (Fixed Focus)</title>
    <style>
        body { background-color: #111; color: white; font-family: sans-serif; text-align: center; }
        .container { display: flex; flex-direction: column; align-items: center; gap: 15px; margin-top: 20px; }
        img { border: 2px solid #00ffff; border-radius: 8px; }
        
        .control-panel { background-color: #222; padding: 15px; border-radius: 8px; width: 800px; display: flex; justify-content: center; align-items: center; }
        
        button { padding: 15px 30px; font-size: 16px; font-weight: bold; cursor: pointer; border-radius: 5px; border: none; margin: 10px; }
        .btn-capture { background-color: #28a745; color: white; }
        .btn-calibrate { background-color: #007bff; color: white; }
        
        #result-box { width: 800px; background-color: #222; padding: 20px; border-radius: 8px; text-align: left; display: none; }
        pre { color: #00ff00; font-size: 15px; white-space: pre-wrap; word-wrap: break-word; }
        .log { color: #ffcc00; margin-top: 10px; font-weight: bold; }
    </style>
</head>
<body>
    <h2>📐 Web-based Camera Calibrator</h2>
    <p style="color: #ff4444; font-weight: bold;">(현재 렌즈 초점이 LensPosition: 5.5로 완벽히 고정되어 있습니다)</p>
    
    <div class="container">
        <img src="/video_feed" width="800" id="video-stream">
        
        <div class="control-panel">
            <button class="btn-capture" onclick="captureFrame()">📸 캡처 (Capture)</button>
            <button class="btn-calibrate" onclick="runCalibration()">⚙️ 연산 수행 (Calibrate)</button>
        </div>
        
        <div class="log" id="log-message">체커보드를 비추고 다양한 각도에서 15장 이상 캡처해주세요.</div>

        <div id="result-box">
            <h3>✅ 캘리브레이션 결과 (코드 복사)</h3>
            <pre id="result-code"></pre>
        </div>
    </div>

    <script>
        function captureFrame() {
            fetch('/capture')
                .then(response => response.json())
                .then(data => {
                    const log = document.getElementById('log-message');
                    if(data.success) {
                        log.innerHTML = "✅ 캡처 성공! 현재 저장된 이미지: " + data.count + "장";
                        log.style.color = "#00ff00";
                    } else {
                        log.innerHTML = "⚠️ 체커보드를 인식할 수 없습니다. 화면을 확인하세요.";
                        log.style.color = "#ff4444";
                    }
                });
        }

        function runCalibration() {
            const log = document.getElementById('log-message');
            log.innerHTML = "⏳ 연산 중입니다... (데이터가 많을수록 수 초 이상 소요될 수 있습니다)";
            log.style.color = "#ffcc00";

            fetch('/calibrate')
                .then(response => response.json())
                .then(data => {
                    if(data.success) {
                        log.innerHTML = "🎉 연산 완료!";
                        log.style.color = "#00ff00";
                        document.getElementById('result-box').style.display = "block";
                        document.getElementById('result-code').innerText = data.result;
                    } else {
                        log.innerHTML = "❌ " + data.result;
                        log.style.color = "#ff4444";
                    }
                });
        }
    </script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(html_template)

@app.route("/video_feed")
def video_feed():
    def gen():
        while True:
            frame = calibrator.get_frame()
            if frame:
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.04)
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/capture")
def capture():
    success, count = calibrator.capture_frame()
    return jsonify({"success": success, "count": count})

@app.route("/calibrate")
def calibrate():
    success, result_text = calibrator.calculate_calibration()
    return jsonify({"success": success, "result": result_text})

if __name__ == "__main__":
    t = threading.Thread(target=calibrator.process_loop)
    t.daemon = True
    t.start()
    app.run(host="0.0.0.0", port=5000)