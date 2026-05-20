import cv2
import numpy as np
from flask import Flask, render_template_string, Response
import threading
import time

# Picamera2 임포트 시도
try:
    from picamera2 import Picamera2
except ImportError:
    print("오류: Picamera2 라이브러리를 찾을 수 없습니다.")
    exit()

class GlobalShutterStreamer:
    def __init__(self):
        self.picam2 = Picamera2()
        self.output_frame = None
        self.lock = threading.Lock()
        self.running = True
        
        # 1. 카메라 설정 (Global Shutter IMX296 최적화)
        # Global Shutter는 해상도가 1456x1088이지만, 스트리밍을 위해 640x480으로 설정 가능
        config = self.picam2.create_video_configuration(
            main={"size": (640, 480), "format": "BGR888"}
        )
        self.picam2.configure(config)
        self.picam2.start()
        
        # 2. 백그라운드 캡처 스레드 시작
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def _capture_loop(self):
        """카메라로부터 프레임을 지속적으로 읽어 저장하는 루프"""
        while self.running:
            # capture_array()는 최신 프레임을 numpy 배열로 가져옴
            frame = self.picam2.capture_array()
            
            # Picamera2는 기본적으로 RGB888이므로 OpenCV용 BGR로 변환
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            with self.lock:
                self.output_frame = frame_bgr
            
            time.sleep(0.01) # CPU 점유율 조절

    def generate_frames(self):
        """Flask 전송을 위한 제너레이터"""
        while True:
            with self.lock:
                if self.output_frame is None:
                    continue
                # JPEG 인코딩
                ret, buffer = cv2.imencode('.jpg', self.output_frame)
                frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# Flask 서버 부분은 동일하게 유지
app = Flask(__name__)
streamer = GlobalShutterStreamer()

@app.route('/')
def index():
    return "<h1>Global Shutter Live Stream</h1><img src='/video_feed' width='640'>"

@app.route('/video_feed')
def video_feed():
    return Response(streamer.generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)