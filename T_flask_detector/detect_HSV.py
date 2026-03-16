import cv2
import numpy as np
import time
import subprocess
import threading
from flask import Flask, Response

app = Flask(__name__)

# === [전역 변수 설정] ===
output_frame = None
lock = threading.Lock()

# === [1] 사용자 설정 (ROI 및 색상) ===
# 카메라 해상도 (640x480 권장)
FRAME_WIDTH = 640   
FRAME_HEIGHT = 480

# ROI (관심 영역) 설정 [x, y, width, height]
ROI_RECT = [120, 170, 360, 70] 

# 감지할 색상 범위 (HSV 기준) - 예: 초록색
# (H, S, V) - 색상, 채도, 명도
LOWER_COLOR = np.array([0, 0, 200])
UPPER_COLOR = np.array([180, 50, 255])   # 180, 255, 255

# 판단 기준 (면적의 x% 이상)
THRESHOLD_RATIO = 0.35

# === [2] 카메라 명령어 (rpicam-vid) ===
# MJPEG 스트림을 stdout으로 쏘도록 설정
cmd = [
    "rpicam-vid",
    "-t", "0",
    "--width", str(FRAME_WIDTH),
    "--height", str(FRAME_HEIGHT),
    "--codec", "mjpeg",
    "--framerate", "30",
    "--contrast", "1.0", # 일반적인 색상 검출을 위해 기본값 사용
    "-o", "-"
]

def process_color_detection(frame):
    """
    이미지 프레임을 받아 ROI 내 색상 비율을 계산하고 그리기
    """
    global ROI_RECT
    
    # 1. ROI 좌표 추출
    x, y, w, h = ROI_RECT
    
    # 2. ROI 영역 잘라내기
    # (주의: ROI가 프레임 밖으로 나가지 않도록 예외처리)
    img_h, img_w = frame.shape[:2]
    if x + w > img_w or y + h > img_h:
        cv2.putText(frame, "ROI Error", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        return frame

    roi = frame[y:y+h, x:x+w]
    
    # 3. BGR -> HSV 변환
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # 4. 색상 마스크 생성
    mask = cv2.inRange(hsv_roi, LOWER_COLOR, UPPER_COLOR)
    
    # 5. 비율 계산
    total_pixels = w * h
    detected_pixels = cv2.countNonZero(mask)
    ratio = detected_pixels / total_pixels if total_pixels > 0 else 0
    
    # 6. 판단 및 시각화
    status_text = f"Ratio: {ratio*100:.1f}%"
    color = (0, 0, 255) # 빨강 (탐지 안됨)
    
    if ratio >= THRESHOLD_RATIO:
        status_text += " - DETECTED!"
        color = (0, 255, 0) # 초록 (탐지됨)
        # 필요한 경우 여기서 GPIO 제어 등을 수행
    
    # 결과 그리기
    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
    cv2.putText(frame, status_text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    return frame

def camera_processing_thread():
    """
    rpicam-vid 프로세스에서 MJPEG 스트림을 읽어와 OpenCV로 변환 및 처리하는 스레드
    """
    global output_frame

    print("📷 카메라 스레드 시작 (rpicam-vid 실행)")
    
    # subprocess 실행
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
    buffer = b""
    
    while True:
        # 스트림 데이터 읽기 (청크 단위)
        data = process.stdout.read(4096)
        if not data:
            break
        buffer += data
        
        # JPEG 시작(0xffd8)과 끝(0xffd9) 찾기
        a = buffer.find(b'\xff\xd8')
        b = buffer.find(b'\xff\xd9')
        
        if a != -1 and b != -1:
            # 온전한 JPEG 데이터 추출
            jpg_data = buffer[a:b+2]
            buffer = buffer[b+2:]
            
            # 디코딩 (JPEG -> OpenCV 이미지)
            frame = cv2.imdecode(np.frombuffer(jpg_data, dtype=np.uint8), cv2.IMREAD_COLOR)
            
            if frame is None:
                continue

            # === [핵심] 색상 검출 로직 적용 ===
            processed_frame = process_color_detection(frame)
            
            # === 웹 전송을 위한 인코딩 ===
            # 다시 JPEG로 인코딩하여 전역 변수에 저장
            ret, jpeg = cv2.imencode('.jpg', processed_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            
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
        
        # MJPEG 포맷 전송
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + encoded_bytes + b'\r\n')
        
        time.sleep(0.03) # 전송 부하 조절 (약 30fps)

@app.route('/')
def index():
    return "<html><body><h1>Flask Color Detector</h1><img src='/video_feed' style='width:100%; max-width:640px;'></body></html>"

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # 데몬 스레드로 카메라 프로세스 시작
    t = threading.Thread(target=camera_processing_thread)
    t.daemon = True
    t.start()
    
    print("🚀 웹 서버 시작: http://0.0.0.0:5000")
    # debug=False 필수 (스레드 중복 실행 방지)
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)