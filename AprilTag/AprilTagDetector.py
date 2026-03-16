import cv2
import numpy as np
import time
import subprocess
import threading
from flask import Flask, Response
from pupil_apriltags import Detector
from collections import deque, defaultdict

app = Flask(__name__)

# === [전역 변수 설정] ===
# 여러 접속자가 공유할 프레임 저장소
output_frame = None
lock = threading.Lock()

# === [1] 하드웨어 파라미터 ===
Tag_size = 0.02
# fx, fy = 523.86, 461.65
# cx, cy = 315.58, 245.79
fx, fy = 460.85, 523.14
cx, cy = 298.03, 238.96
camera_params = (fx, fy, cx, cy)
camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

# === [2] Detector 설정 (발열 제어 유지) ===
detector = Detector(
    families="tag36h11",
    quad_decimate=1.0, 
    quad_sigma=0.0, 
    refine_edges=0, 
    decode_sharpening=0.25,
    nthreads=1 
)

pose_history = defaultdict(lambda: deque(maxlen=5))

# === [3] 카메라 명령어 ===
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
    """
    백그라운드에서 단 1번만 실행되어 계속 영상을 처리하고
    전역 변수(output_frame)를 업데이트하는 함수
    """
    global output_frame, pose_history

    print("📷 카메라 스레드 시작 (rpicam-vid 실행)")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
    buffer = b""
    
    frame_counter = 0
    SKIP_FRAMES = 2  # 발열 제어용 스킵 설정 유지
    last_detections = []

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

            # === 이미지 처리 로직 (기존과 동일) ===
            # 왜곡 보정 (필요 시 주석 처리)
            # frame = cv2.undistort(frame, camera_matrix, dist_coeffs)

            frame_counter += 1
            
            # Detection (Skip logic)
            if frame_counter % (SKIP_FRAMES + 1) == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                last_detections = detector.detect(gray, estimate_tag_pose=True, camera_params=camera_params, tag_size=Tag_size)
            
            # Drawing
            for detection in last_detections:
                tag_id = detection.tag_id
                
                raw_x = detection.pose_t[0][0] * 1000
                raw_y = detection.pose_t[1][0] * 1000
                raw_z = detection.pose_t[2][0] * 1000
                
                pose_history[tag_id].append([raw_x, raw_y, raw_z])
                avg_x = np.mean([p[0] for p in pose_history[tag_id]])
                avg_y = np.mean([p[1] for p in pose_history[tag_id]])
                avg_z = np.mean([p[2] for p in pose_history[tag_id]])

                corners = detection.corners.astype(int)
                for i in range(4):
                    cv2.line(frame, tuple(corners[i]), tuple(corners[(i + 1) % 4]), (0, 255, 0), 2)
                
                # draw_axes 함수는 외부에 정의되어 있다고 가정하거나, 필요한 경우 이 함수 내부에 포함
                # (이전 코드에 있던 draw_axes 로직을 간단히 구현)
                center = tuple(detection.center.astype(int))
                cv2.putText(frame, f"<ID:{tag_id}> {avg_x:.0f}, {avg_y:.0f}, {avg_z:.0f} (mm)", (center[0]-110, center[1]-40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                # cv2.putText(frame, f"<ID:{tag_id}>", (center[0], center[1]-40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.circle(frame, center, 3, (0, 0, 255), -1)
                # print(f"Tag ID: {tag_id}, Position (mm): X={avg_x:.1f}, Y={avg_y:.1f}, Z={avg_z:.1f}")

            # === [중요] 처리된 이미지를 전역 변수에 저장 ===
            ret, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ret:
                with lock:
                    output_frame = jpeg.tobytes()

def generate_frames():
    """
    웹 클라이언트에게 이미 처리된 최신 프레임만 전달하는 함수
    """
    global output_frame
    while True:
        with lock:
            if output_frame is None:
                continue
            # 현재 저장된 최신 프레임 복사
            encoded_bytes = output_frame
        
        # 전송
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + encoded_bytes + b'\r\n')
        
        # 클라이언트 전송 속도 조절 (너무 빠르면 브라우저 렉 유발)
        time.sleep(0.05) 

@app.route('/')
def index():
    return "<html><body><h1>AprilTag Detector</h1><img src='/video_feed'></body></html>"

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # 서버 시작 전에 카메라 스레드를 먼저 실행 (데몬 스레드로 실행하여 메인 앱 종료 시 같이 종료됨)
    t = threading.Thread(target=camera_processing_thread)
    t.daemon = True
    t.start()
    
    print("🚀 웹 서버 시작: http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    