import cv2
import numpy as np
import time
import subprocess
import threading
from flask import Flask, Response
from pyzbar.pyzbar import decode  # 바코드 인식 라이브러리
from collections import deque

app = Flask(__name__)

# === [전역 변수 설정] ===
output_frame = None
lock = threading.Lock()

# === [1] 카메라 설정 ===
# 해상도를 너무 높이면 바코드 인식 연산이 느려질 수 있습니다 (640x480 권장)
FRAME_WIDTH = 640   
FRAME_HEIGHT = 480

# === [2] 카메라 명령어 (rpicam-vid) ===
cmd = [
    "rpicam-vid",
    "-t", "0",
    "--width", str(FRAME_WIDTH),
    "--height", str(FRAME_HEIGHT),
    "--codec", "mjpeg",
    "--framerate", "30",
    "--contrast", "1.1",  # 바코드 인식률을 높이기 위해 대비를 약간 높임
    "-o", "-"
]

def draw_barcode(frame, decoded_objects):
    """
    인식된 바코드 정보를 이미지에 그리는 함수
    """
    for obj in decoded_objects:
        # 1. 바코드 영역 다각형 그리기
        # obj.polygon은 바코드의 4개 꼭짓점 좌표를 담고 있습니다.
        points = obj.polygon
        
        # 점이 4개 이상일 때만 그리기 (가끔 찌그러진 경우 대비)
        if len(points) >= 4:
            pts = np.array(points, dtype=np.int32)
            pts = pts.reshape((-1, 1, 2))
            
            # 녹색 테두리 그리기
            cv2.polylines(frame, [pts], True, (0, 255, 0), 3)
        else:
            # 다각형 감지가 잘 안되면 rect 정보로 사각형 그리기
            (x, y, w, h) = obj.rect
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

        # 2. 바코드 데이터 및 타입 텍스트 표시
        barcode_data = obj.data.decode("utf-8") # 바이트 데이터를 문자열로 변환
        barcode_type = obj.type
        
        text = f"[{barcode_type}] {barcode_data}"
        
        # 텍스트 위치 계산 (바코드 바로 위)
        (x, y, w, h) = obj.rect
        cv2.putText(frame, text, (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # 콘솔에도 출력 (디버깅용)
        # print(f"Found {barcode_type}: {barcode_data}")

def camera_processing_thread():
    """
    카메라 영상을 획득하고 바코드를 처리하는 백그라운드 스레드
    """
    global output_frame

    print("📷 카메라 스레드 시작 (rpicam-vid 실행)")
    print("ℹ️  바코드/QR코드를 카메라에 비춰주세요.")

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
    buffer = b""
    
    frame_counter = 0
    # 바코드 인식은 연산량이 꽤 있으므로, 모든 프레임보다는 
    # 2~3프레임마다 한 번씩 수행하는 것이 웹 스트리밍 부드러움에 유리합니다.
    SKIP_FRAMES = 2 
    last_decoded_objects = []

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
            
            # 디코딩
            frame = cv2.imdecode(np.frombuffer(jpg_data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None: continue

            frame_counter += 1
            
            # === 바코드 인식 로직 ===
            # 매 프레임마다 인식하면 CPU 부하가 클 수 있으므로 SKIP_FRAMES 활용
            if frame_counter % (SKIP_FRAMES + 1) == 0:
                # 흑백 변환 (바코드 인식률 향상 및 속도 개선)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # pyzbar를 이용한 디코딩
                last_decoded_objects = decode(gray)
            
            # === 시각화 (Drawing) ===
            # 인식된 정보가 있으면 화면에 표시 (이전 프레임 정보라도 그려줌으로써 끊김 방지)
            if last_decoded_objects:
                draw_barcode(frame, last_decoded_objects)
            else:
                # 인식된 게 없을 때 화면 중앙 조준점 표시 (선택사항)
                h, w = frame.shape[:2]
                cv2.line(frame, (w//2 - 20, h//2), (w//2 + 20, h//2), (0, 255, 255), 2)
                cv2.line(frame, (w//2, h//2 - 20), (w//2, h//2 + 20), (0, 255, 255), 2)

            # === JPEG 인코딩 및 전역 변수 업데이트 ===
            ret, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ret:
                with lock:
                    output_frame = jpeg.tobytes()

def generate_frames():
    """ 웹 클라이언트로 프레임 전송 """
    global output_frame
    while True:
        with lock:
            if output_frame is None:
                continue
            encoded_bytes = output_frame
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + encoded_bytes + b'\r\n')
        
        time.sleep(0.05) # 전송 부하 조절

@app.route('/')
def index():
    return "<html><body><h1>Barcode Scanner</h1><img src='/video_feed' style='width:100%; max-width:640px;'></body></html>"

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # 데몬 스레드로 카메라 프로세스 시작
    t = threading.Thread(target=camera_processing_thread)
    t.daemon = True
    t.start()
    
    print("🚀 웹 서버 시작: http://0.0.0.0:5000")
    # debug=False로 설정해야 스레드가 두 번 실행되는 것을 방지함
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)