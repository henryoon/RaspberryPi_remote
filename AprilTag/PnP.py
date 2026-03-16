import cv2
import numpy as np
import time
import subprocess
import threading
from flask import Flask, Response
from pupil_apriltags import Detector
from collections import deque

app = Flask(__name__)

# === [전역 변수] ===
output_frame = None
lock = threading.Lock()

# === [1] 설정: 태그 중심점 배치도 (World 좌표계, mm 단위) ===
# ID 0번을 원점(0,0,0)으로 둠
TAG_LAYOUT = {
    0: np.array([0, 0, 0], dtype=np.float32),      
    1: np.array([100, 0, 0], dtype=np.float32),    
}

# === [2] 파라미터 ===
TAG_SIZE_MM = 30.0  # 태그 크기 (mm)
TAG_SIZE_M = TAG_SIZE_MM / 1000.0

# 카메라 매트릭스
fx, fy = 608.87, 607.39
cx, cy = 299.96, 236.05
camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros(5)

# 스무딩 및 깜빡임 방지 설정
SMOOTHING_FACTOR = 0.1
KEEP_ALIVE_TIME = 0.5  # 0.5초간 잔상 유지

# 변수 초기화
last_smoothed_pos = None
tag_memory = {}  # 태그 기억 저장소

# Detector 설정
detector = Detector(families="tag36h11", nthreads=1)

# 카메라 명령어
cmd = [
    "rpicam-vid", "-t", "0", "--width", "640", "--height", "480",
    "--codec", "mjpeg", "--framerate", "30", "-o", "-"
]

# === [헬퍼 함수] ===
def get_tag_corners_in_world(center_pos, size_mm):
    """ 태그 중심점과 크기로 3D 코너 좌표 4개 생성 """
    half_s = size_mm / 2.0
    x, y, z = center_pos
    corners_3d = np.array([
        [x - half_s, y - half_s, z], # Left-Top
        [x + half_s, y - half_s, z], # Right-Top
        [x + half_s, y + half_s, z], # Right-Bottom
        [x - half_s, y + half_s, z]  # Left-Bottom
    ], dtype=np.float32)
    return corners_3d

def draw_dashboard(frame, tvec, tag_count):
    if tvec is None: return
    
    # mm 단위 변환
    x, y, z = tvec[0][0], tvec[1][0], tvec[2][0]

    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (280, 130), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "PNP Camera Pose", (25, 35), font, 0.5, (0, 255, 127), 2)
    
    # 상태 표시 (메모리에 있는 태그 개수가 아니라, 실제 계산에 쓰인 태그 개수 기준)
    status_text = "Stable" if tag_count >= 2 else ("Single Tag" if tag_count == 1 else "Holding...")
    color = (0, 255, 0) if tag_count >= 2 else (0, 255, 255)
    
    cv2.circle(frame, (25, 55), 6, color, -1)
    cv2.putText(frame, f"{status_text} (Active: {tag_count})", (40, 60), font, 0.5, (200, 200, 200), 1)

    cv2.putText(frame, f"X: {x:6.1f} mm", (25, 85), font, 0.5, (0, 255, 255), 2)
    cv2.putText(frame, f"Y: {y:6.1f} mm", (25, 105), font, 0.5, (0, 255, 255), 2)
    cv2.putText(frame, f"Z: {z:6.1f} mm", (25, 125), font, 0.5, (0, 255, 255), 2)

def camera_processing_thread():
    global output_frame, last_smoothed_pos, tag_memory
    
    print("📷 시스템 시작 (PnP + Anti-Flicker)")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
    buffer = b""
    frame_counter = 0

    # PnP 스무딩용 변수
    smooth_rvec = None
    smooth_tvec = None

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
            current_time = time.time()
            
            # === Detection ===
            if frame_counter % 2 == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                detections = detector.detect(gray, estimate_tag_pose=False)
                
                # PnP 계산을 위한 점 리스트
                all_3d_points = []
                all_2d_points = []
                
                # 1. 감지된 태그 처리
                for detection in detections:
                    tag_id = detection.tag_id
                    
                    # (A) 메모리에 저장 (시각화용)
                    tag_memory[tag_id] = {
                        'corners': detection.corners.astype(int),
                        'center': detection.center.astype(int),
                        'last_seen': current_time
                    }
                    
                    # (B) PnP 데이터 수집 (배치도에 있는 태그만)
                    if tag_id in TAG_LAYOUT:
                        corners_2d = detection.corners.astype(np.float32)
                        center_3d = TAG_LAYOUT[tag_id]
                        corners_3d = get_tag_corners_in_world(center_3d, TAG_SIZE_MM)
                        
                        for i in range(4):
                            all_2d_points.append(corners_2d[i])
                            all_3d_points.append(corners_3d[i])

                # 2. PnP 계산 (현재 보이는 태그만 사용)
                active_tag_count = len(all_3d_points) // 4
                
                if active_tag_count >= 1:
                    object_points = np.array(all_3d_points, dtype=np.float32)
                    image_points = np.array(all_2d_points, dtype=np.float32)
                    
                    success, rvec, tvec = cv2.solvePnP(
                        object_points, image_points, 
                        camera_matrix, dist_coeffs, 
                        # flags=cv2.SOLVEPNP_IPPE
                        flags=cv2.SOLVEPNP_SQPNP
                    )
                    
                    if success:
                        # EMA 스무딩 필터
                        if smooth_tvec is None:
                            smooth_tvec = tvec
                            smooth_rvec = rvec
                        else:
                            smooth_tvec = SMOOTHING_FACTOR * tvec + (1-SMOOTHING_FACTOR) * smooth_tvec
                            smooth_rvec = SMOOTHING_FACTOR * rvec + (1-SMOOTHING_FACTOR) * smooth_rvec
                        
                        # 좌표 변환 (Camera -> World)
                        R, _ = cv2.Rodrigues(smooth_rvec)
                        cam_pos = -np.matrix(R).T @ np.matrix(smooth_tvec)
                        
                        # 대시보드 표시용 데이터 업데이트
                        cam_pos_flat = np.array(cam_pos).flatten()
                        last_smoothed_pos = cam_pos_flat.reshape(3, 1)

            # === Drawing (메모리 기반 - 깜빡임 방지) ===
            # PnP 계산은 '현재 데이터'로 하지만, 그리기는 '메모리'를 참조
            for tag_id in list(tag_memory.keys()):
                tag_info = tag_memory[tag_id]
                time_diff = current_time - tag_info['last_seen']
                
                if time_diff < KEEP_ALIVE_TIME:
                    corners = tag_info['corners']
                    center = tag_info['center']
                    
                    # 방금 인식됨(0.1초 이내): 초록색 / 놓쳤지만 기억함: 회색
                    if time_diff < 0.1:
                        color = (0, 255, 0)
                        text_color = (0, 255, 0)
                    else:
                        color = (150, 150, 150)
                        text_color = (200, 200, 200)
                    
                    # 박스 및 ID 그리기
                    cv2.polylines(frame, [corners], True, color, 2)
                    cv2.putText(frame, f"ID:{tag_id}", (center[0]-25, center[1]-15), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
                    cv2.circle(frame, tuple(center), 3, (0, 0, 255), -1)
                    cv2.circle(frame, (320, 240), 3, (0, 0, 0), -1)
                else:
                    # 시간 초과 시 메모리 삭제
                    del tag_memory[tag_id]

            # 대시보드 그리기 (마지막 유효 위치 사용)
            if last_smoothed_pos is not None:
                # 현재 PnP에 쓰인 태그 개수는 0일 수도 있음 (이 경우 Holding 상태 표시)
                # Drawing 루프에서 active 태그 개수를 다시 세거나, 위 변수를 가져옴
                # 간단히 active_tag_count가 없으면 0 처리
                try:
                    display_count = active_tag_count
                except NameError:
                    display_count = 0
                    
                draw_dashboard(frame, last_smoothed_pos, display_count)

            ret, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
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
    return "<html><body style='margin:0; background:black;'><img src='/video_feed' style='width:100%; height:auto;'></body></html>"

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    t = threading.Thread(target=camera_processing_thread)
    t.daemon = True
    t.start()
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)