import cv2
import numpy as np
import time
import subprocess
import threading
import os
from datetime import datetime
from flask import Flask, Response

app = Flask(__name__)

# === [설정 영역] ===
# 이미지를 저장할 폴더 경로 (없으면 자동 생성됨)
SAVE_DIR = "/home/rnd/HJ/AI_camera/captures"
CAPTURE_INTERVAL = 0.5  # 저장 간격 (초 단위)
START_DELAY = 15.0

# === [전역 변수] ===
output_frame = None
lock = threading.Lock()

# 폴더가 없으면 생성
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
    print(f"📁 저장 폴더 생성됨: {SAVE_DIR}")

# === [카메라 명령어] ===
# 고화질 캡처를 원하면 width, height를 늘리세요 (예: 1920, 1080)
cmd = [
    "rpicam-vid",
    "-t", "0",
    "--width", "640",
    "--height", "480",
    "--codec", "mjpeg",
    "--framerate", "30",
    "-o", "-"
]

def camera_processing_thread():
    global output_frame
    
    print("📷 카메라 스레드 시작 & 자동 캡처 대기 중...")
    print(f"⏳ {START_DELAY}초 후에 캡처가 시작됩니다.")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
    buffer = b""
    
    program_start_time = time.time()
    last_capture_time = time.time()
    
    # 저장 표시(빨간점)를 위한 타이머
    show_indicator_until = 0

    while True:
        data = process.stdout.read(4096)
        if not data:
            break
        buffer += data
        
        a = buffer.find(b'\xff\xd8')
        b = buffer.find(b'\xff\xd9')
        
        if a != -1 and b != -1:
            jpg_data = buffer[a:b+2]
            buffer = buffer[b+2:]
            
            # 1. 바이트 -> 이미지 변환
            frame = cv2.imdecode(np.frombuffer(jpg_data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None: continue

            current_time = time.time()
            elapsed_time = current_time - program_start_time
            
            if elapsed_time < START_DELAY:
                remain_time = int(START_DELAY - elapsed_time) + 1
                text = f"Starting in {remain_time}s"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                text_x = (frame.shape[1] - text_size[0]) // 2
                text_y = (frame.shape[0] - text_size[1]) // 2
                cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                
            else:
                # === [자동 저장 로직] ===
                if current_time - last_capture_time >= CAPTURE_INTERVAL:
                    # 파일명 생성 (년월일_시분초_마이크로초)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = os.path.join(SAVE_DIR, f"img_{timestamp}.jpg")
                    
                    # 이미지 저장 (비동기적으로 처리하지 않으면 순간 렉이 걸릴 수 있음, 
                    cv2.imwrite(filename, frame)
                    print(f"💾 저장됨: {filename}")
                    
                    last_capture_time = current_time
                    show_indicator_until = current_time + 0.2 # 0.2초 동안 빨간 점 표시

                # === [화면 표시 로직] ===
                # 저장이 일어났을 때 화면 오른쪽 상단에 빨간 원 그리기
                if current_time < show_indicator_until:
                    cv2.circle(frame, (620, 20), 10, (0, 0, 255), -1) # 빨간 원
                    cv2.putText(frame, "REC", (550, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # 현재 시간 표시
            time_str = datetime.now().strftime("%H:%M:%S")
            cv2.putText(frame, time_str, (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # === [웹 전송용 인코딩] ===
            ret, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ret:
                with lock:
                    output_frame = jpeg.tobytes()

def generate_frames():
    global output_frame
    while True:
        with lock:
            if output_frame is None:
                continue
            encoded_bytes = output_frame
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + encoded_bytes + b'\r\n')
        time.sleep(0.05)

@app.route('/')
def index():
    return """
    <html>
        <head><title>Auto Capture Cam</title></head>
        <body style="background:black; color:white; text-align:center;">
            <h1>📸 Auto Capture (img/0.5s)</h1>
            <p>Images are saved to: /home/rnd/HJ/AI_camera/captures</p>
            <img src="/video_feed" style="border:2px solid green; width:640px;">
        </body>
    </html>
    """

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # 메인 앱 시작 전 카메라 스레드 실행
    t = threading.Thread(target=camera_processing_thread)
    t.daemon = True
    t.start()
    
    print(f"🚀 웹 서버 시작: http://0.0.0.0:5000")
    print(f"📂 이미지가 {SAVE_DIR} 폴더에 저장됩니다.")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)