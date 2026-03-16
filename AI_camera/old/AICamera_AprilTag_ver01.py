import cv2
import numpy as np
import time
import subprocess
import sys
from flask import Flask, Response
from pupil_apriltags import Detector
from collections import deque, defaultdict

app = Flask(__name__)

# === [1] AprilTag 및 카메라 설정 ===
# AI 카메라의 하드웨어 스펙에 맞춘 파라미터가 필요합니다. (보정값이 없다면 추정치 사용)
Tag_size = 0.02
fx, fy = 523.86, 523.14
cx, cy = 315.58, 245.79
camera_params = (fx, fy, cx, cy)

camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

detector = Detector(families="tag36h11")
pose_history = defaultdict(lambda: deque(maxlen=5))

# === [2] rpicam-vid 실행 명령어 ===
# OpenCV가 아닌 시스템 명령어로 카메라를 직접 제어합니다.
cmd = [
    "rpicam-vid",
    "-t", "0",              # 무제한 실행
    "--width", "640",       # 해상도 (낮출수록 FPS 상승)
    "--height", "480",
    "--codec", "mjpeg",     # MJPEG로 받아서 파이썬에서 디코딩
    "-o", "-"               # 표준 출력(stdout)으로 데이터 전송
]

def draw_axes(frame, camera_params, tag_size, pose_R, pose_t):
    fx, fy, cx, cy = camera_params
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.zeros(5)
    axis_length = tag_size / 4

    object_points = np.array(
        [[0, 0, 0], [axis_length, 0, 0], [0, axis_length, 0], [0, 0, -axis_length]],
        dtype=np.float32,
    )
    rvec, _ = cv2.Rodrigues(pose_R)
    tvec = pose_t
    image_points, _ = cv2.projectPoints(object_points, rvec, tvec, K, dist_coeffs)
    p = image_points.astype(int)
    origin = tuple(p[0].ravel())

    cv2.line(frame, origin, tuple(p[1].ravel()), (0, 0, 255), 2)
    cv2.line(frame, origin, tuple(p[2].ravel()), (0, 255, 0), 2)
    cv2.line(frame, origin, tuple(p[3].ravel()), (255, 0, 0), 2)

def generate_frames():
    # 서브프로세스로 rpicam-vid 실행
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
    buffer = b""
    prev_time = 0

    print("📷 AI Camera 연결 성공. 스트리밍 시작...")

    while True:
        # 프로세스에서 데이터 읽기
        data = process.stdout.read(4096)
        if not data:
            break
        
        buffer += data
        
        # MJPEG 프레임 추출 (시작: 0xff 0xd8, 끝: 0xff 0xd9)
        a = buffer.find(b'\xff\xd8')
        b = buffer.find(b'\xff\xd9')
        
        if a != -1 and b != -1:
            jpg_data = buffer[a:b+2]
            buffer = buffer[b+2:]
            
            # [핵심] 바이트 데이터를 OpenCV 이미지로 디코딩
            frame = cv2.imdecode(np.frombuffer(jpg_data, dtype=np.uint8), cv2.IMREAD_COLOR)
            
            if frame is None:
                continue

            # === 여기서부터 AprilTag 인식 로직 시작 ===
            
            # 1. 렌즈 왜곡 보정
            frame = cv2.undistort(frame, camera_matrix, dist_coeffs)

            # FPS 계산
            current_time = time.time()
            fps = 0
            if current_time - prev_time > 0:
                fps = 1 / (current_time - prev_time)
            prev_time = current_time

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detections = detector.detect(gray, estimate_tag_pose=True, camera_params=camera_params, tag_size=Tag_size)

            for detection in detections:
                tag_id = detection.tag_id
                
                # 좌표 스무딩
                raw_x = detection.pose_t[0][0] * 1000
                raw_y = detection.pose_t[1][0] * 1000
                raw_z = detection.pose_t[2][0] * 1000
                pose_history[tag_id].append([raw_x, raw_y, raw_z])

                avg_x = np.mean([p[0] for p in pose_history[tag_id]])
                avg_y = np.mean([p[1] for p in pose_history[tag_id]])
                avg_z = np.mean([p[2] for p in pose_history[tag_id]])

                # 시각화
                corners = detection.corners.astype(int)
                for i in range(4):
                    cv2.line(frame, tuple(corners[i]), tuple(corners[(i + 1) % 4]), (0, 255, 0), 2)

                draw_axes(frame, camera_params, Tag_size, detection.pose_R, detection.pose_t)
                
                center = tuple(detection.center.astype(int))
                text_avg = f"XYZ: {avg_x:.0f} {avg_y:.0f} {avg_z:.0f} mm"
                cv2.putText(frame, text_avg, (center[0]-50, center[1]+40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(frame, f"ID:{tag_id}", (center[0]-20, center[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # === 결과 이미지를 다시 JPEG로 인코딩하여 웹으로 전송 ===
            ret, jpeg = cv2.imencode('.jpg', frame)
            frame_bytes = jpeg.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return """
    <html>
        <head><title>AI Camera AprilTag</title></head>
        <body style="background:black; color:white; text-align:center;">
            <h1>AprilTag Detection</h1>
            <img src="/video_feed" style="border:2px solid green; width:640px;">
        </body>
    </html>
    """

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("🚀 웹 서버 시작: http://0.0.0.0:5000")
    try:
        app.run(host='0.0.0.0', port=5000, debug=False)
    except KeyboardInterrupt:
        print("종료 중...")