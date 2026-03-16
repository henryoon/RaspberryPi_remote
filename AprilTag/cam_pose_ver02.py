import cv2
import numpy as np
import subprocess
import threading
import time
from dt_apriltags import Detector
from flask import Flask, Response, render_template_string
from scipy.spatial.transform import Rotation as R

app = Flask(__name__)

# === [1. 설정 데이터] ===
RES_WIDTH, RES_HEIGHT = 640, 480
# 실제 카메라 캘리브레이션 값 (fx, fy, cx, cy)
CAMERA_PARAMS = [485.43, 484.96, 314.78, 237.50]
TAG_SIZE = 0.03  # 30mm

# 월드 좌표계 내 태그 위치 (ID: [X, Y, Z] mm)
WORLD_TAG_POS = {
    0: np.array([0, 0, 0]),
    1: np.array([115, 0, 0])
}

# === [2. 전역 변수 및 초기화] ===
output_frame = None
lock = threading.Lock()
# nthreads를 CPU 코어 수에 맞춰 조정하여 성능 향상
at_detector = Detector(families='tag36h11', nthreads=4)

def get_camera_pose_advanced(tag_R_mat, tag_t_vec, world_pos):
    """쿼터니언을 활용한 정밀 역변환 및 월드 좌표 계산"""
    # 회전 행렬로부터 Rotation 객체 생성 및 역행렬 계산
    R_tag_to_cam = R.from_matrix(tag_R_mat)
    R_cam_to_tag = R_tag_to_cam.inv()
    
    # 태그 기준 카메라의 상대 위치 계산 (m -> mm)
    t_mm = tag_t_vec.flatten() * 1000
    cam_pos_rel = -R_cam_to_tag.as_matrix() @ t_mm
    
    # 월드 좌표 = 태그의 월드 좌표 + 카메라의 상대 좌표
    world_camera_pos = world_pos + cam_pos_rel
    # 회전 정보 추출 (Euler Angles: Yaw, Pitch, Roll)
    world_camera_quat = R_cam_to_tag.as_quat()
    
    return world_camera_pos, world_camera_quat

def draw_enhanced_dashboard(frame, pos, quat):
    """회전 정보(Yaw)가 포함된 개선된 GUI 오버레이"""
    if pos is None: return

    # Yaw(Z축 회전) 계산
    euler = R.from_quat(quat).as_euler('zyx', degrees=True)
    yaw = euler[0]

    # 디자인: 반투명 배경 상자
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (260, 140), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # 텍스트 정보 출력
    color = (0, 255, 255) # Cyan
    cv2.putText(frame, "SYSTEM: POSE ESTIMATION", (20, 30), 1, 0.8, color, 1, cv2.LINE_AA)
    
    data_text = [
        f"X: {pos[0]:8.1f} mm",
        f"Y: {pos[1]:8.1f} mm",
        f"Z: {pos[2]:8.1f} mm",
        f"Yaw: {yaw:7.1f} deg"
    ]
    
    for i, text in enumerate(data_text):
        cv2.putText(frame, text, (25, 55 + (i * 20)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

def camera_thread_func():
    global output_frame
    # rpicam-vid: 라즈베리 파이 카메라 하드웨어 가속 호출
    cmd = ["rpicam-vid", "-t", "0", "--width", str(RES_WIDTH), "--height", str(RES_HEIGHT), 
           "--codec", "mjpeg", "--framerate", "30", "-o", "-", "--nopreview"]
    
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)
    buffer = b""

    while True:
        data = process.stdout.read(8192)
        if not data: break
        buffer += data
        
        # JPEG 이미지의 시작(\xff\xd8)과 끝(\xff\xd9) 탐색
        a, b = buffer.find(b'\xff\xd8'), buffer.find(b'\xff\xd9')
        if a != -1 and b != -1:
            jpg_data = buffer[a:b+2]
            buffer = buffer[b+2:]
            
            frame = cv2.imdecode(np.frombuffer(jpg_data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None: continue

            # 연산 성능 향상을 위해 그레이스케일 사용
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            tags = at_detector.detect(gray, estimate_tag_pose=True, 
                                    camera_params=CAMERA_PARAMS, tag_size=TAG_SIZE)

            pos_list, quat_list = [], []
            for tag in tags:
                if tag.tag_id in WORLD_TAG_POS:
                    # 태그 시각화 (선택 사항)
                    cv2.polylines(frame, [tag.corners.astype(int)], True, (0, 255, 0), 2)
                    
                    p, q = get_camera_pose_advanced(tag.pose_R, tag.pose_t, WORLD_TAG_POS[tag.tag_id])
                    pos_list.append(p)
                    quat_list.append(q)

            # 데이터 평균화 (여러 태그 인식 시 안정성 확보)
            if pos_list:
                avg_pos = np.mean(pos_list, axis=0)
                avg_quat = np.mean(quat_list, axis=0)
                avg_quat /= np.linalg.norm(avg_quat) # 쿼터니언 정규화
                draw_enhanced_dashboard(frame, avg_pos, avg_quat)

            # 웹 전송용 인코딩 (품질 80으로 성능 타협)
            _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            with lock:
                output_frame = jpeg.tobytes()

def generate_frames():
    """웹 서버에 프레임을 공급하는 제너레이터"""
    while True:
        with lock:
            if output_frame is None:
                continue
            frame_data = output_frame
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
        time.sleep(0.03) # 약 30fps 제안

# === [3. 웹 서버 라우팅] ===
@app.route('/')
def index():
    """메인 모니터링 페이지 HTML"""
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI Robot Pose Monitor</title>
        <style>
            body { background: #0f1012; color: #00d4ff; font-family: 'Segoe UI', sans-serif; 
                   margin: 0; display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100vh; }
            .container { position: relative; border: 2px solid #333; border-radius: 12px; overflow: hidden; 
                         box-shadow: 0 0 50px rgba(0, 212, 255, 0.2); }
            h1 { font-weight: 200; letter-spacing: 5px; margin-bottom: 20px; text-transform: uppercase; }
            img { display: block; width: 100%; max-width: 800px; height: auto; }
            .status { margin-top: 15px; font-size: 0.9rem; color: #666; }
        </style>
    </head>
    <body>
        <h1>Robot Vision System</h1>
        <div class="container">
            <img src="/video_feed">
        </div>
        <p class="status">● LIVE STREAMING | SENSOR FUSION ACTIVE</p>
    </body>
    </html>
    """)

@app.route('/video_feed')
def video_feed():
    """영상 스트리밍 엔드포인트"""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # 카메라 스레드를 데몬으로 시작
    threading.Thread(target=camera_thread_func, daemon=True).start()
    # Flask 서버 실행
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=False)