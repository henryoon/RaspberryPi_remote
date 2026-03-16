import cv2
import numpy as np
import time
import subprocess
import threading
from flask import Flask, Response
from pyzbar.pyzbar import decode

app = Flask(__name__)

# === [전역 변수] ===
output_frame = None
lock = threading.Lock()

# === [1] 카메라 해상도 설정 (고해상도로 변경) ===
# 640x480 -> 1280x720 (HD)
# 픽셀 밀도를 높여 바코드의 얇은 선을 구분할 수 있게 합니다.
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# === [2] ROI (관심 영역) 설정 ===
# 전체 화면을 다 분석하면 느리므로, 중앙의 이 크기만큼만 잘라내서 분석합니다.
# 사실상 "디지털 줌" 역할을 하여 작은 바코드 인식률을 비약적으로 높입니다.
ROI_WIDTH = 640
ROI_HEIGHT = 360

# === [3] 카메라 명령어 ===
cmd = [
    "rpicam-vid",
    "-t", "0",
    "--width", str(FRAME_WIDTH),
    "--height", str(FRAME_HEIGHT),
    "--codec", "mjpeg",
    "--framerate", "30",
    "--contrast", "1.1",     # 하드웨어 대비 약간 증가
    "--sharpness", "1.5",    # 하드웨어 샤프닝 켜기
    "-o", "-"
]

# === [4] 이미지 전처리 도구 준비 ===
# (1) 샤프닝 커널: 이미지를 더 날카롭게 만듦
sharpening_kernel = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
])

# (2) CLAHE 객체: 빛 반사가 심한 플라스틱 표면에 효과적임
# clipLimit가 높을수록 대비가 강해짐 (2.0 ~ 4.0 추천)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

def preprocess_image(img_roi):
    """ 바코드 인식을 돕기 위한 이미지 전처리 함수 """
    # 1. 흑백 변환
    gray = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)
    
    # 2. CLAHE 적용 (조명 불균형 해결 & 대비 극대화)
    # 일반 histogram equalization보다 반사광 억제에 유리함
    gray_clahe = clahe.apply(gray)
    
    # 3. 샤프닝 필터 적용 (경계선 강화)
    gray_sharp = cv2.filter2D(gray_clahe, -1, sharpening_kernel)
    
    return gray_sharp

def camera_processing_thread():
    global output_frame

    print("📷 고성능 바코드 인식 스레드 시작 (HD + ROI + CLAHE)")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
    buffer = b""
    
    frame_counter = 0
    SKIP_FRAMES = 2 
    last_decoded_objects = []

    # ROI 시작 좌표 계산 (화면 정중앙)
    roi_x = (FRAME_WIDTH - ROI_WIDTH) // 2
    roi_y = (FRAME_HEIGHT - ROI_HEIGHT) // 2

    while True:
        # 해상도가 높아졌으므로 버퍼 읽는 크기를 늘림
        data = process.stdout.read(8192)
        if not data: break
        buffer += data
        
        a = buffer.find(b'\xff\xd8')
        b = buffer.find(b'\xff\xd9')
        
        if a != -1 and b != -1:
            jpg_data = buffer[a:b+2]
            buffer = buffer[b+2:]
            
            frame = cv2.imdecode(np.frombuffer(jpg_data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None: continue

            frame_counter += 1
            
            # === [핵심 프로세스] ===
            if frame_counter % (SKIP_FRAMES + 1) == 0:
                # 1. ROI 잘라내기 (디지털 줌)
                # 전체 1280x720 이미지가 아니라 중앙 640x360만 떼어냅니다.
                roi_frame = frame[roi_y:roi_y+ROI_HEIGHT, roi_x:roi_x+ROI_WIDTH]
                
                # 2. 강력한 전처리 (CLAHE + Sharpen)
                processed_roi = preprocess_image(roi_frame)
                
                # 3. 인식 수행
                last_decoded_objects = decode(processed_roi)

            # === [시각화] ===
            # 1. 사용자가 바코드를 맞춰야 할 가이드 박스 표시 (파란색)
            cv2.rectangle(frame, (roi_x, roi_y), (roi_x+ROI_WIDTH, roi_y+ROI_HEIGHT), (255, 200, 0), 2)
            cv2.putText(frame, "Scan Here", (roi_x, roi_y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)

            # 2. 인식된 바코드 그리기
            if last_decoded_objects:
                for obj in last_decoded_objects:
                    # ROI 내부 좌표이므로, 전체 화면 좌표로 변환해야 함
                    (x, y, w, h) = obj.rect
                    real_x = x + roi_x
                    real_y = y + roi_y
                    
                    # 바코드 박스 (녹색)
                    cv2.rectangle(frame, (real_x, real_y), (real_x + w, real_y + h), (0, 255, 0), 3)
                    
                    # 텍스트
                    barcode_data = obj.data.decode("utf-8")
                    barcode_type = obj.type
                    text = f"[{barcode_type}] {barcode_data}"
                    
                    cv2.putText(frame, text, (real_x, real_y - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # 전송 (웹 전송용 품질 설정)
            ret, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            if ret:
                with lock:
                    output_frame = jpeg.tobytes()

def generate_frames():
    global output_frame
    while True:
        with lock:
            if output_frame is None: continue
            encoded_bytes = output_frame
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + encoded_bytes + b'\r\n')
        time.sleep(0.05)

@app.route('/')
def index():
    return "<html><body><h1>Enhanced Barcode Scanner</h1><img src='/video_feed' style='width:100%; max-width:800px;'></body></html>"

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    t = threading.Thread(target=camera_processing_thread)
    t.daemon = True
    t.start()
    print("🚀 웹 서버 시작: http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)