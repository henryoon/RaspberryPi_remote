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

class CameraCalibrator:
    def __init__(self):
        self.lock = threading.Lock()
        self.output_frame = None 
        self.running = True
        
        # 캘리브레이션 설정 (환경에 맞게 수정하세요)
        self.checkerboard = (8, 6)
        self.square_size = 0.008
        self.target_images = 20
        
        # 상태 관리 변수
        self.is_capturing = False
        self.captured_count = 0
        self.last_capture_time = 0
        self.calibration_done = False
        
        # 3D 공간상의 체커보드 포인트 생성
        self.objp = np.zeros((self.checkerboard[0] * self.checkerboard[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:self.checkerboard[0], 0:self.checkerboard[1]].T.reshape(-1, 2)
        self.objp *= self.square_size
        
        self.objpoints = []
        self.imgpoints = []

        # 카메라 설정
        self.width, self.height = 640, 480
        self.picam2 = Picamera2()
        self._setup_camera()

    def _setup_camera(self):
        print(f"📷 카메라 초기화 중... ({self.width}x{self.height})")
        config = self.picam2.create_video_configuration(
            main={"size": (self.width, self.height), "format": "BGR888"}
        )
        self.picam2.configure(config)
        self.picam2.start()
        try:
            # AprilTag 구동 환경과 동일하게 초점 고정
            self.picam2.set_controls({"AfMode": 0, "LensPosition": 5.5})
            print("🔄 초점 고정 완료 (LensPosition: 5.5)")
        except Exception as e:
            print(f"⚠️ AF 설정 오류: {e}")

    def trigger_capture(self):
        """웹에서 호출하여 자동 캡처 프로세스를 시작하는 트리거"""
        with self.lock:
            if not self.is_capturing:
                self.is_capturing = True
                self.captured_count = 0
                self.objpoints = []
                self.imgpoints = []
                self.calibration_done = False
                print("▶️ 자동 캡처 트리거 작동! 카메라에 체커보드를 비춰주세요.")

    def run_calibration_task(self, image_size):
        """웹 스트리밍이 멈추지 않도록 백그라운드에서 연산"""
        print("\n🧮 20장 수집 완료! 캘리브레이션 연산을 시작합니다 (수 초 소요)...")
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            self.objpoints, self.imgpoints, image_size, None, None
        )

        print("\n" + "="*50)
        print("🚀 [결과] 아래 코드를 복사해서 AprilTag 코드에 붙여넣으세요!")
        print("="*50)
        print(f"self.camera_matrix = np.array([")
        print(f"    [{camera_matrix[0][0]:.5f}, 0, {camera_matrix[0][2]:.5f}],")
        print(f"    [0, {camera_matrix[1][1]:.5f}, {camera_matrix[1][2]:.5f}],")
        print(f"    [0, 0, 1]")
        print(f"], dtype=np.float32)")

        dist_list = ", ".join([f"{x:.5f}" for x in dist_coeffs[0]])
        print(f"\nself.dist_coeffs = np.array([[{dist_list}]], dtype=np.float32)")
        print("="*50 + "\n")

        with self.lock:
            self.calibration_done = True

    def process_loop(self):
        while self.running:
            raw_frame = self.picam2.capture_array()
            if raw_frame is None: continue
            
            frame = cv2.cvtColor(raw_frame, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            display_frame = frame.copy()

            # 상태 읽기
            with self.lock:
                is_cap = self.is_capturing
                count = self.captured_count
                target = self.target_images
                done = self.calibration_done

            if is_cap and count < target:
                # 체커보드 찾기
                ret, corners = cv2.findChessboardCorners(gray, self.checkerboard, None)
                
                if ret:
                    # 화면에 인식된 코너 그리기
                    cv2.drawChessboardCorners(display_frame, self.checkerboard, corners, ret)
                    
                    # 마지막 캡처 후 1.5초가 지났다면 자동으로 사진 수집
                    if (time.time() - self.last_capture_time) > 1.5:
                        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

                        with self.lock:
                            self.objpoints.append(self.objp)
                            self.imgpoints.append(corners2)
                            self.captured_count += 1
                            self.last_capture_time = time.time()
                            print(f"📸 캡처: {self.captured_count}/{self.target_images}")

                            # 목표 장수에 도달하면 연산 쓰레드 시작
                            if self.captured_count >= self.target_images:
                                self.is_capturing = False
                                threading.Thread(target=self.run_calibration_task, args=(gray.shape[::-1],)).start()

            # 웹 화면에 현재 상태 안내 텍스트 띄우기
            if done:
                cv2.putText(display_frame, "Done! Check Terminal for Matrix.", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif is_cap:
                cv2.putText(display_frame, f"Capturing: {count}/{target} (Move Board)", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            else:
                cv2.putText(display_frame, "Ready. Press 'Start' on Web.", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            with self.lock:
                self.output_frame = display_frame

            time.sleep(0.01)

    def get_frame(self):
        with self.lock:
            if self.output_frame is None: return None
            _, buffer = cv2.imencode(".jpg", self.output_frame)
            return bytearray(buffer)

# --- Flask Server ---
app = Flask(__name__)
calibrator = CameraCalibrator()

@app.route("/")
def index():
    # '시작' 버튼이 추가된 웹 페이지
    return render_template_string("""
    <html>
      <body style="background-color:#111; color:white; text-align:center; font-family: sans-serif;">
        <h2>📷 카메라 파라미터 웹 캘리브레이션</h2>
        <p>1. 체커보드를 카메라 앞에 준비하세요.<br>
           2. '캡처 시작' 버튼을 누르고 체커보드를 이리저리 움직이세요.<br>
           3. 20장이 모두 찍히면 터미널 창에 결과가 출력됩니다.</p>
           
        <button onclick="startTrigger()" style="padding: 15px 30px; font-size: 18px; font-weight: bold; background-color: #00cc66; color: white; border: none; border-radius: 5px; cursor: pointer; margin-bottom: 20px;">
          ▶️ 캡처 시작
        </button>
        <br>
        <img src="/video_feed" width="800" style="border: 2px solid #00cc66;">
        
        <script>
          function startTrigger() {
            fetch('/trigger')
              .then(response => response.text())
              .then(data => console.log(data));
          }
        </script>
      </body>
    </html>
    """)

@app.route("/video_feed")
def video_feed():
    def gen():
        while True:
            frame = calibrator.get_frame()
            if frame:
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.04)
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/trigger")
def trigger():
    calibrator.trigger_capture()
    return "Triggered!"

if __name__ == "__main__":
    t = threading.Thread(target=calibrator.process_loop)
    t.daemon = True
    t.start()
    # 외부 기기에서 접속할 수 있도록 host="0.0.0.0" 설정
    app.run(host="0.0.0.0", port=5000)