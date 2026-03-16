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
output_frame = None
lock = threading.Lock()

# === [1] 하드웨어 파라미터 ===
Tag_size = 0.02
fx, fy = 295.40, 294.00
cx, cy = 303.70, 248.74
camera_params = (fx, fy, cx, cy)
camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

# === [지터링 방지 설정] ===
# 0.0 ~ 1.0 사이 값. 
# 작을수록(0.1) 떨림은 줄어들지만 반응이 느려짐(Lag). 
# 클수록(0.9) 빠릿하지만 떨림이 남음.
SMOOTHING_FACTOR = 0.2 
smoothed_poses = {} # { tag_id: {'t': vec, 'R': mat} }

# === [각 태그별 고정 오프셋 포인트 정의] ===
# TAG_OFFSET_POINTS = {
#     0: np.array([[50, 60, 0], [50, -15, 0], [50, -85, 0], [50, -165, 0]], dtype=np.float32),
#     1: np.array([[50, 60, 0], [50, -15, 0], [50, -85, 0], [50, -165, 0]], dtype=np.float32),
#     2: np.array([[50, 60, 0], [50, -15, 0], [50, -85, 0], [50, -165, 0]], dtype=np.float32),
#     3: np.array([[50, 60, 0], [50, -15, 0], [50, -85, 0], [50, -165, 0]], dtype=np.float32),
#     4: np.array([[50, 60, 0], [50, -15, 0], [50, -85, 0], [50, -165, 0]], dtype=np.float32),
#     15: np.array([[50, 60, 0], [50, -15, 0], [50, -85, 0], [50, -165, 0]], dtype=np.float32),
# }

# 추가 position이 필요 없으면 이 줄로 대체
TAG_OFFSET_POINTS = {0: np.array([[0, 0, 0]], dtype=np.float32)} 

# === [2] Detector 설정 ===
detector = Detector(
    families="tag36h11",
    quad_decimate=1.0, 
    quad_sigma=0.0, 
    refine_edges=0, 
    decode_sharpening=0.25,
    nthreads=1 
)

# === [3] 카메라 명령어 ===
cmd = [
    "rpicam-vid", "-t", "0", "--width", "640", "--height", "480",
    "--codec", "mjpeg", "--framerate", "30", "-o", "-"
]

def project_point(pose_R, pose_t, point_mm):
    """ 3D 좌표를 2D 픽셀로 변환 """
    point_m = point_mm / 1000.0
    point_cam = pose_R @ point_m + pose_t.reshape(3)
    
    if point_cam[2] <= 0: return None

    u = fx * (point_cam[0] / point_cam[2]) + cx
    v = fy * (point_cam[1] / point_cam[2]) + cy
    return (int(u), int(v))

def camera_processing_thread():
    global output_frame, smoothed_poses

    print("📷 카메라 스레드 시작 (rpicam-vid 실행)")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
    buffer = b""
    
    frame_counter = 0
    SKIP_FRAMES = 2 
    last_detections = []

    while True:
        data = process.stdout.read(4096)
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
            
            # === Detection 수행 ===
            if frame_counter % (SKIP_FRAMES + 1) == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                last_detections = detector.detect(gray, estimate_tag_pose=True, camera_params=camera_params, tag_size=Tag_size)
            
            # 현재 프레임에서 감지된 태그 ID 목록
            current_tag_ids = set()

            # === Drawing & Smoothing Loop ===
            for detection in last_detections:
                tag_id = detection.tag_id
                current_tag_ids.add(tag_id)

                # 1. 지수 이동 평균(EMA) 필터 적용 (핵심)
                current_t = detection.pose_t
                current_R = detection.pose_R

                if tag_id not in smoothed_poses:
                    # 처음 발견된 태그면 현재 값 그대로 저장
                    smoothed_poses[tag_id] = {'t': current_t, 'R': current_R}
                else:
                    # 이전에 발견된 태그면 가중치 적용하여 업데이트
                    # New = Alpha * Current + (1 - Alpha) * Previous
                    prev_t = smoothed_poses[tag_id]['t']
                    prev_R = smoothed_poses[tag_id]['R']
                    
                    new_t = SMOOTHING_FACTOR * current_t + (1 - SMOOTHING_FACTOR) * prev_t
                    new_R = SMOOTHING_FACTOR * current_R + (1 - SMOOTHING_FACTOR) * prev_R
                    
                    smoothed_poses[tag_id] = {'t': new_t, 'R': new_R}

                # 이제부터 모든 그림은 'smoothed' 값을 사용
                use_t = smoothed_poses[tag_id]['t']
                use_R = smoothed_poses[tag_id]['R']

                # 2. 태그 기본 그리기 (Box)
                # Box 코너는 detection.corners에 있는데 이것도 튈 수 있음.
                # 정확히 하려면 3D Box를 use_R, use_t로 투영해야 하지만, 
                # 간단히 시각적 안정감을 위해 원본 코너를 그대로 쓰거나 중심점만 찍음.
                corners = detection.corners.astype(int)
                for i in range(4):
                    cv2.line(frame, tuple(corners[i]), tuple(corners[(i + 1) % 4]), (0, 255, 0), 2)
                
                center = tuple(detection.center.astype(int))
                cv2.circle(frame, center, 5, (0, 0, 255), -1)

                # 3. 태그 중심 좌표 텍스트 (안정화된 값 사용)
                t_vec = use_t.flatten() * 1000 
                cv2.putText(frame, f"ID:{tag_id} ({t_vec[0]:.0f}, {t_vec[1]:.0f}, {t_vec[2]:.0f})", 
                            (center[0]-70, center[1]+30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

                # 4. 오프셋 포인트 시각화 (안정화된 값 사용)
                if tag_id in TAG_OFFSET_POINTS:
                    points_3d = TAG_OFFSET_POINTS[tag_id]
                    prev_point = None 
                    
                    for i, p3d_local in enumerate(points_3d):
                        # 부드러운 use_R, use_t를 사용하여 투영
                        p2d = project_point(use_R, use_t, p3d_local)
                        
                        # 실제 좌표 계산도 부드러운 값 사용
                        point_local_m = p3d_local / 1000.0 
                        point_cam_m = use_R @ point_local_m + use_t.reshape(3)
                        point_cam_mm = point_cam_m * 1000.0 
                        
                        if p2d is not None:
                            cv2.circle(frame, p2d, 5, (0, 255, 255), -1)
                            
                            real_coord_text = f"({point_cam_mm[0]:.0f}, {point_cam_mm[1]:.0f}, {point_cam_mm[2]:.0f})"
                            
                            cv2.putText(frame, str(i+1), (p2d[0]+8, p2d[1]-5), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                            
                            cv2.putText(frame, real_coord_text, (p2d[0]+25, p2d[1]-5), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
                            
                            if prev_point is not None:
                                cv2.line(frame, prev_point, p2d, (0, 255, 255), 1)
                            prev_point = p2d
            
            # 화면에서 사라진 태그는 딕셔너리에서 제거 (메모리 관리 및 재진입 시 초기화)
            # (선택 사항: 잠깐 사라져도 기억하고 싶으면 이 부분 주석 처리)
            smoothed_poses = {k: v for k, v in smoothed_poses.items() if k in current_tag_ids}

            ret, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
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