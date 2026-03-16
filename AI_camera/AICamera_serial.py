import cv2
import numpy as np
import time
import subprocess
import threading
import serial  # [추가] 시리얼 통신 라이브러리
from flask import Flask, Response
from pupil_apriltags import Detector
from collections import deque, defaultdict

app = Flask(__name__)

# === [전역 변수 설정] ===
output_frame = None
lock = threading.Lock()

# === [시리얼 통신 설정] ===
# 로봇과 연결된 포트 이름 확인 필요 (터미널에서 'ls /dev/tty*' 로 확인)
# 보통 아두이노는 '/dev/ttyACM0' 또는 '/dev/ttyUSB0' 입니다.
SERIAL_PORT = '/dev/ttyACM0'  
BAUD_RATE = 115200
robot_serial = None

try:
    robot_serial = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    print(f"✅ 로봇 시리얼 연결 성공: {SERIAL_PORT}")
except Exception as e:
    print(f"⚠️ 로봇 시리얼 연결 실패 (통신 없이 진행): {e}")

# === [1] 하드웨어 파라미터 ===
Tag_size = 0.02
fx, fy = 460.85, 523.14
cx, cy = 298.03, 238.96
camera_params = (fx, fy, cx, cy)
camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

# === [2] Detector 설정 ===
detector = Detector(
    families="tag36h11",
    quad_decimate=1.0, 
    quad_sigma=0.0, 
    refine_edges=1, 
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

def send_data_to_robot(tag_id, x, y, z):
    """
    로봇에게 좌표 데이터를 전송하는 함수
    포맷 예시: <1, 100.5, -50.2, 300.0>\n
    """
    if robot_serial and robot_serial.is_open:
        try:
            # 데이터 포맷팅 (시작문자 '<', 끝문자 '>', 구분자 ',')
            data_str = f"<{tag_id},{x:.1f},{y:.1f},{z:.1f}>\n"
            robot_serial.write(data_str.encode('utf-8'))
            # print(f"Sent: {data_str.strip()}") # 디버깅용 출력
        except Exception as e:
            print(f"전송 에러: {e}")

def camera_processing_thread():
    global output_frame, pose_history

    print("📷 카메라 스레드 시작 (rpicam-vid 실행)")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
    buffer = b""
    
    frame_counter = 0
    SKIP_FRAMES = 2
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
            
            frame = cv2.imdecode(np.frombuffer(jpg_data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None: continue

            frame_counter += 1
            
            # Detection
            if frame_counter % (SKIP_FRAMES + 1) == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                last_detections = detector.detect(gray, estimate_tag_pose=True, camera_params=camera_params, tag_size=Tag_size)
            
            # Drawing & Sending Data
            for detection in last_detections:
                tag_id = detection.tag_id
                
                raw_x = detection.pose_t[0][0] * 1000
                raw_y = detection.pose_t[1][0] * 1000
                raw_z = detection.pose_t[2][0] * 1000
                
                pose_history[tag_id].append([raw_x, raw_y, raw_z])
                avg_x = np.mean([p[0] for p in pose_history[tag_id]])
                avg_y = np.mean([p[1] for p in pose_history[tag_id]])
                avg_z = np.mean([p[2] for p in pose_history[tag_id]])

                # [추가됨] 로봇에게 데이터 전송 (검출 주기와 동일하게 전송됨)
                # 너무 자주 보낸다면 여기에 카운터를 추가하여 조절 가능
                if frame_counter % (SKIP_FRAMES + 1) == 0:
                    send_data_to_robot(tag_id, avg_x, avg_y, avg_z)

                corners = detection.corners.astype(int)
                for i in range(4):
                    cv2.line(frame, tuple(corners[i]), tuple(corners[(i + 1) % 4]), (0, 255, 0), 2)
                
                center = tuple(detection.center.astype(int))
                cv2.putText(frame, f"<ID:{tag_id}> {avg_x:.0f}, {avg_y:.0f}, {avg_z:.0f}", (center[0]-110, center[1]-40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.circle(frame, center, 3, (0, 0, 255), -1)

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
    return "<html><body><h1>AprilTag Detector</h1><img src='/video_feed'></body></html>"

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    t = threading.Thread(target=camera_processing_thread)
    t.daemon = True
    t.start()
    
    print("🚀 웹 서버 시작: http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)