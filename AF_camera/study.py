import cv2
import numpy as np
import subprocess
import threading
import time
from flask import Flask, Response
from pupil_apriltags import Detector

app = Flask(__name__)

# === [1. 제공해주신 캘리브레이션 데이터 적용] ===
RES_WIDTH, RES_HEIGHT = 640, 480

# Camera Matrix (K)
K = np.array([
    [920.73777437, 0.0, 311.42603026],
    [0.0, 920.76420988, 239.08127547],
    [0.0, 0.0, 1.0]
], dtype=np.float32)

# Distortion Coefficients (dist)
DIST = np.array([-1.99067671e-02, 9.62222700e-01, 1.38609840e-03, 3.54970950e-04, -7.18444763e+00], dtype=np.float32)

# AprilTag Detector용 파라미터 [fx, fy, cx, cy]
CAMERA_PARAMS = [K[0,0], K[1,1], K[0,2], K[1,2]]
TAG_SIZE = 0.03  # 30mm

TAG_LAYOUT = {
    0: np.array([0.0, 0.0, 0.0]),
    1: np.array([0.115, 0.0, 0.0])
}

# 캘리브레이션 최적화를 위한 맵 생성 (연산 속도 향상)
new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(K, DIST, (RES_WIDTH, RES_HEIGHT), 0)
map1, map2 = cv2.initUndistortRectifyMap(K, DIST, None, new_camera_matrix, (RES_WIDTH, RES_HEIGHT), cv2.CV_32FC1)

output_frame = None
lock = threading.Lock()
at_detector = Detector(families='tag36h11', nthreads=1)

cmd = [
    "rpicam-vid", "-t", "0", "--width", str(RES_WIDTH), "--height", str(RES_HEIGHT),
    "--codec", "mjpeg", "--framerate", "30", "--inline", "--flush",
    "--autofocus-mode", "continuous", "-o", "-"
]

# === [2. 좌표 보정 로직] ===
def get_refined_pose(results):
    origin_candidates = []
    for r in results:
        if r.tag_id in TAG_LAYOUT:
            R_tag = r.pose_R
            t_tag = r.pose_t.flatten()
            tag_offset = TAG_LAYOUT[r.tag_id]
            ref_origin = t_tag - np.dot(R_tag, tag_offset)
            origin_candidates.append(ref_origin)
    return np.mean(origin_candidates, axis=0) if origin_candidates else None

# === [3. 카메라 스레드] ===
def camera_thread_func():
    global output_frame
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)
    buffer = b""

    while True:
        data = process.stdout.read(8192)
        if not data: break
        buffer += data
        a, b = buffer.find(b'\xff\xd8'), buffer.find(b'\xff\xd9')
        
        if a != -1 and b != -1:
            jpg_data = buffer[a:b+2]
            buffer = buffer[b+2:]
            raw_frame = cv2.imdecode(np.frombuffer(jpg_data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if raw_frame is None: continue

            # [핵심] 렌즈 왜곡 보정 실행
            frame = cv2.remap(raw_frame, map1, map2, cv2.INTER_LINEAR)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # 보정된 파라미터(new_camera_matrix)를 사용하여 인식
            results = at_detector.detect(gray, estimate_tag_pose=True, 
                                        camera_params=[new_camera_matrix[0,0], new_camera_matrix[1,1], new_camera_matrix[0,2], new_camera_matrix[1,2]], 
                                        tag_size=TAG_SIZE)

            refined_t = get_refined_pose(results)

            # UI 그리기
            for r in results:
                pts = np.array(r.corners, dtype=np.int32)
                cv2.polylines(frame, [pts], True, (0, 255, 0), 1)
                center = tuple(r.center.astype(int))
                cv2.circle(frame, center, 3, (255, 0, 0), -1)
                cv2.circle(frame, (RES_WIDTH//2, RES_HEIGHT//2), 3, (0, 0, 255), -1)
            
            if refined_t is not None:
                text = f"REF POS: X:{refined_t[0]*1000:.1f} Y:{refined_t[1]*1000:.1f} Z:{refined_t[2]*1000:.1f} (mm)"
                cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

            ret_enc, jpeg = cv2.imencode('.jpg', frame)
            if ret_enc:
                with lock:
                    output_frame = jpeg.tobytes()

def generate_frames():
    while True:
        with lock:
            if output_frame is None: continue
            data = output_frame
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + data + b'\r\n')
        time.sleep(0.03)

@app.route('/')
def index():
    return """
    <html>
      <head>
        <title>AprilTag Responsive Stream</title>
        <style>
          body {
            margin: 0;
            padding: 0;
            background-color: #000;
            display: flex;
            align-items: center;
            justify-content: center;
            height: 100vh;
            overflow: hidden; /* 스크롤바 방지 */
          }
          img {
            /* 창 크기에 맞춰 가로/세로 중 꽉 차는 쪽에 맞춤 */
            max-width: 100%;
            max-height: 100%;
            object-fit: contain; /* 비율 유지하며 창 안에 꽉 채움 */
          }
        </style>
      </head>
      <body>
        <img src="/video_feed">
      </body>
    </html>
    """

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    threading.Thread(target=camera_thread_func, daemon=True).start()
    app.run(host='0.0.0.0', port=5000)