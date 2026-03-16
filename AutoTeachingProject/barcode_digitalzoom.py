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
        
        # 카메라 설정
        self.width, self.height = 1920, 1080
        # self.width, self.height = 3840, 2160
        self.roi_x, self.roi_y = 400, 120
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
        self._set_af_mode(continuous=True)

    # 화면 중앙에 초점 맞추기
    # def _set_af_mode(self, continuous=True):
    #     """AF 모드 전환: continuous=True(연속 AF), False(수동/고정)"""
    #     try:
    #         af_x = (self.width - self.roi_x) // 2
    #         af_y = (self.height - self.roi_y) // 2
    #         mode = 2 if continuous else 0
    #         self.picam2.set_controls({
    #             "AfMode": mode,
    #             "AfRange": 0,
    #             "AfWindows": [[af_x, af_y, self.roi_x, self.roi_y]]
    #         })
    #         self.is_focus_locked = not continuous
    #         status = "연속 AF 활성화" if continuous else "초점 고정(Lock) 활성화"
    #         print(f"🔄 {status}")
    #     except Exception as e:
    #         print(f"⚠️ AF 설정 오류: {e}")
    
    # 근접 거리 고정 초점
    def _set_af_mode(self, continuous=True):
        try:
            if continuous:
                # 1. 먼저 연속 AF로 초점을 잡도록 명령
                self.picam2.set_controls({"AfMode": 2})
                # 2. 잠시 대기 (렌즈 이동 시간 확보)
                time.sleep(1.0) 
                # 3. 그 후 특정 거리(실험으로 찾은 값)로 고정
                self.picam2.set_controls({"AfMode": 0, "LensPosition": 5.5})
                status = "초점 보정 후 고정 완료"
            else:
                self.picam2.set_controls({"AfMode": 0, "LensPosition": 5.5})
                status = "고정 모드 유지"
            
            print(f"🔄 {status}")
        except Exception as e:
            print(f"⚠️ AF 오류: {e}")

    def process_loop(self):
        while self.running:
            raw_frame = self.picam2.capture_array()
            if raw_frame is None: continue
            
            frame = cv2.cvtColor(raw_frame, cv2.COLOR_RGB2BGR)
            h, w, _ = raw_frame.shape
            x1, y1 = (w - self.roi_x) // 2, (h - self.roi_y) // 2 + 30
            x2, y2 = x1 + self.roi_x, y1 + self.roi_y

            display_main = frame.copy()
            cv2.rectangle(display_main, (x1, y1), (x2, y2), (255, 0, 0), 2)

            roi_img = frame[y1:y2, x1:x2]
            zoomed_roi = cv2.resize(roi_img, (self.roi_x * self.scale, self.roi_y * self.scale), 
                                    interpolation=cv2.INTER_CUBIC)

            # 1. 바코드 인식 시도
            decoded = pyzbar.decode(zoomed_roi)
            current_time = time.time()

            if len(decoded) > 0:
                # 바코드 감지됨: 마지막 감지 시간 업데이트 및 초점 고정
                self.last_detection_time = current_time
                if not self.is_focus_locked:
                    self._set_af_mode(continuous=False)
                    self.is_focus_locked = True
            else:
                # 바코드 감지 안 됨: 고정 상태인데 30초가 지났다면 고정 해제
                if self.is_focus_locked:
                    elapsed = current_time - self.last_detection_time
                    if elapsed >= self.lock_duration:
                        print(f"⏱️ 30초 경과: 초점 고정 해제")
                        self._set_af_mode(continuous=True)
                        

            # 2. 화면 표시 로직
            for obj in decoded:
                data = obj.data.decode('utf-8')
                cv2.putText(zoomed_roi, f"DATA: {data}", (20, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # 상태 표시
            status_text = "FOCUS LOCKED" if self.is_focus_locked else "SCANNING (AF)"
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
          <div><h3>Main View</h3><img src="/video_main" width="640"></div>
          <div><h3>Zoomed ROI</h3><img src="/video_zoom" width="480"></div>
        </div>
        <p>상태: 바코드 인식 시 30초간 초점 고정 모드 작동 중</p>
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