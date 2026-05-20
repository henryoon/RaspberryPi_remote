import time
import threading
import cv2
import numpy as np
from flask import Flask, Response, render_template_string
from pyzbar import pyzbar

try:
    from picamera2 import Picamera2
except ImportError:
    print("오류: picamera2를 찾을 수 없습니다.")
    exit()

class BarcodeDualVision:
    def __init__(self):
        self.lock = threading.Lock()
        self.main_frame = None   
        self.zoomed_frame = None 
        self.running = True
        
        # 초점 제어 관련 변수
        self.is_focus_locked = False
        self.last_detection_time = 0
        self.lock_duration = 30  # 초점 유지 시간 (초)
        
        # 카메라 설정 (1920x1080)
        self.width, self.height = 1920, 1080
        self.roi_x, self.roi_y = 400, 120
        # self.width, self.height = 3840, 2160
        # self.roi_x, self.roi_y = 800, 240
        self.scale = 3
        
        self.picam2 = Picamera2()
        self._setup_camera()

    def _setup_camera(self):
        print("📷 Camera Module 3 초기화...")
        config = self.picam2.create_video_configuration(
            main={"size": (self.width, self.height), "format": "BGR888"}
        )
        self.picam2.configure(config)
        self.picam2.start()
        # 초기 실행 시 1회 초점 보정 후 5.5로 고정
        self._set_af_mode(continuous=True)

    def _set_af_mode(self, continuous=True):
        """AF 모드 전환: 실험적으로 찾은 LensPosition 5.5 적용"""
        try:
            if continuous:
                self.picam2.set_controls({"AfMode": 2}) # Continuous AF
                time.sleep(1.0) 
                self.picam2.set_controls({"AfMode": 0, "LensPosition": 5.5})
                status = "초점 보정 후 5.5 고정 완료"
            else:
                self.picam2.set_controls({"AfMode": 0, "LensPosition": 5.5})
                status = "고정 모드 유지(5.5)"
            
            print(f"🔄 {status}")
        except Exception as e:
            print(f"⚠️ AF 오류: {e}")

    def process_loop(self):
        while self.running:
            raw_frame = self.picam2.capture_array()
            if raw_frame is None: continue
            
            # RGB -> BGR 변환 (색상 반전 해결)
            frame = cv2.cvtColor(raw_frame, cv2.COLOR_RGB2BGR)
            h, w, _ = frame.shape
            
            # ROI 좌표 계산 (중앙에서 약간 아래로 +30 오프셋 적용)
            x1, y1 = (w - self.roi_x) // 2, (h - self.roi_y) // 2 + 30
            x2, y2 = x1 + self.roi_x, y1 + self.roi_y

            display_main = frame.copy()
            # 메인 뷰에 ROI 영역 표시 (파란색)
            cv2.rectangle(display_main, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # ROI 추출 및 확대
            roi_img = frame[y1:y2, x1:x2]
            zoomed_roi = cv2.resize(roi_img, (self.roi_x * self.scale, self.roi_y * self.scale), 
                                    interpolation=cv2.INTER_CUBIC)

            # 1. 바코드 인식 시도
            decoded = pyzbar.decode(zoomed_roi)
            current_time = time.time()

            if len(decoded) > 0:
                self.last_detection_time = current_time
                if not self.is_focus_locked:
                    self._set_af_mode(continuous=False)
                    self.is_focus_locked = True
            else:
                if self.is_focus_locked:
                    elapsed = current_time - self.last_detection_time
                    if elapsed >= self.lock_duration:
                        print(f"⏱️ 30초 경과: 초점 다시 잡기")
                        self._set_af_mode(continuous=True)
                        self.is_focus_locked = False

            # 2. 바코드 테두리 시각화 로직 추가
            for obj in decoded:
                # 바코드 영역의 폴리곤(다각형) 좌표 가져오기
                points = obj.polygon
                if len(points) > 0:
                    # 넘파이 배열로 변환하여 다각형 그리기
                    pts = np.array(points, np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    # 확대된 ROI 화면에 녹색 테두리 그리기
                    cv2.polylines(zoomed_roi, [pts], True, (0, 255, 0), 3)
                
                # 바코드 데이터 텍스트 표시
                data = obj.data.decode('utf-8')
                # 텍스트가 바코드 바로 위에 표시되도록 좌표 설정
                (tx, ty) = points[0].x, points[0].y
                cv2.putText(zoomed_roi, f"DATA: {data}", (tx, ty - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # 상태 표시 (메인 뷰)
            status_text = "FOCUS LOCKED (5.5)" if self.is_focus_locked else "SCANNING (AF)"
            cv2.putText(display_main, status_text, (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if self.is_focus_locked else (255, 255, 0), 2)

            with self.lock:
                self.main_frame = display_main
                self.zoomed_frame = zoomed_roi

            time.sleep(0.01)

    def get_frame(self, target='main'):
        with self.lock:
            frame = self.main_frame if target == 'main' else self.zoomed_frame
            if frame is None: return None
            _, buffer = cv2.imencode(".jpg", frame)
            return bytearray(buffer)

# --- Flask Server ---
app = Flask(__name__)
vision = BarcodeDualVision()

@app.route("/")
def index():
    return render_template_string("""
    <html>
      <body style="background-color:#111; color:white; text-align:center; font-family: sans-serif;">
        <h2>📸 Digital Zoom Barcode Scanner</h2>
        <div style="display: flex; justify-content: center; gap: 20px;">
          <div><h3>Main View (Original)</h3><img src="/video_main" width="640"></div>
          <div><h3>Zoomed ROI (Barcode View)</h3><img src="/video_zoom" width="640"></div>
        </div>
        <div style="margin-top: 20px; font-size: 1.2em;">
            상태: <span style="color: #00ff00;">바코드 인식 시 초점 고정(30s) 및 실시간 테두리 트래킹 활성화</span>
        </div>
      </body>
    </html>
    """)

@app.route("/video_main")
def video_main():
    return Response(gen_stream('main'), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/video_zoom")
def video_zoom():
    return Response(gen_stream('zoom'), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_stream(target):
    while True:
        frame = vision.get_frame(target)
        if frame:
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.04)

if __name__ == "__main__":
    t = threading.Thread(target=vision.process_loop)
    t.daemon = True
    t.start()
    app.run(host="0.0.0.0", port=5000)