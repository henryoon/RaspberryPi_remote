import cv2
import numpy as np
import time
import subprocess
import threading
from flask import Flask, Response
from pupil_apriltags import Detector
from collections import deque

app = Flask(__name__)

# === [전역 변수 및 설정] ===
output_frame = None
lock = threading.Lock()

# [설정 1] 태그 배치도 (월드 좌표계 기준, 단위: mm)
# 실제 환경에 맞게 수정 필수!
TAG_LAYOUT = {
    0: np.array([0, 0, 0], dtype=np.float32),      # 기준점
    1: np.array([100, 0, 0], dtype=np.float32),    # 기준점 오른쪽 100mm
    # 필요한 만큼 추가...
}

# [설정 2] 카메라 및 태그 파라미터
Tag_size = 0.03  # 30mm
fx, fy = 510.75, 508.83
cx, cy = 319.08, 239.76
camera_params = (fx, fy, cx, cy)

# [설정 3] 깜빡임 방지 유지 시간 (초)
KEEP_ALIVE_TIME = 0.5

# [설정 4] 위치 스무딩을 위한 버퍼 (이동 평균)
pos_buffer = deque(maxlen=10)

# 태그 메모리 (깜빡임 방지용)
tag_memory = {}

# === [Detector] ===
detector = Detector(
    families="tag36h11", quad_decimate=1.0, quad_sigma=0.0,
    refine_edges=1, decode_sharpening=0.25, nthreads=1
)

cmd = [
    "rpicam-vid", "-t", "0", "--width", "640", "--height", "480",
    "--codec", "mjpeg", "--framerate", "30", "-o", "-"
]

# === [헬퍼 함수] ===
def get_camera_position_from_tag(pose_R, pose_t, tag_world_pos):
    """ 개별 태그를 기준으로 카메라의 월드 좌표 계산 """
    cam_rel_pos = -np.matrix(pose_R).T @ np.matrix(pose_t)
    cam_rel_pos_mm = np.array(cam_rel_pos).flatten() * 1000.0
    camera_global_pos = tag_world_pos + cam_rel_pos_mm
    return camera_global_pos

def draw_dashboard(frame, position, tag_count):
    """ 세련된 UI 대시보드 그리기 """
    if position is None: return

    # 1. 반투명 배경 만들기
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (280, 130), (0, 0, 0), -1) # 검은 박스
    alpha = 0.3  # 투명도 설정
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # 2. 텍스트 스타일 설정
    font = cv2.FONT_HERSHEY_SIMPLEX
    label_color = (255, 255, 255) # 연회색
    value_color = (0, 255, 255)   # 형광 노랑 (강조)
    title_color = (0, 255, 127)   # 밝은 초록

    # 3. 정보 표시
    # 타이틀
    cv2.putText(frame, "CAMERA POSE", (25, 35), font, 0.5, title_color, 2)

    # 상태 표시기
    status_color = (0, 255, 0) if tag_count >= 2 else (0, 165, 255) # 2개 이상이면 초록, 아니면 주황
    status_text = "Stable" if tag_count >= 2 else "Acquiring..."
    cv2.circle(frame, (25, 55), 6, status_color, -1)
    cv2.putText(frame, f"Status: {status_text} (Tags: {tag_count})", (40, 60), font, 0.5, label_color, 1)

    # 좌표 값 표시
    cv2.putText(frame, f"X: {position[0]:6.1f} mm", (25, 85), font, 0.5, value_color, 2)
    cv2.putText(frame, f"Y: {position[1]:6.1f} mm", (25, 105), font, 0.5, value_color, 2)
    cv2.putText(frame, f"Z: {position[2]:6.1f} mm", (25, 125), font, 0.5, value_color, 2)

    cv2.circle(frame, (320, 240), 3, (0, 0, 0), -1)

def camera_processing_thread():
    global output_frame, tag_memory
    
    print("📷 시스템 시작 (Anti-Flicker + Sensor Fusion + UI)")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
    buffer = b""
    frame_counter = 0

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
                detections = detector.detect(gray, estimate_tag_pose=True, camera_params=camera_params, tag_size=Tag_size)
                
                # 메모리 업데이트 및 포즈 계산용 리스트 준비
                current_frame_positions = []
                
                for detection in detections:
                    tag_id = detection.tag_id
                    # 메모리에 저장 (시각화용)
                    tag_memory[tag_id] = {
                        'corners': detection.corners.astype(int),
                        'center': detection.center.astype(int),
                        'last_seen': current_time
                    }

                    # 배치도에 있는 태그라면 카메라 위치 계산 (포즈용)
                    if tag_id in TAG_LAYOUT:
                        world_pos = TAG_LAYOUT[tag_id]
                        cam_pos = get_camera_position_from_tag(detection.pose_R, detection.pose_t, world_pos)
                        current_frame_positions.append(cam_pos)

                # 다중 태그 위치 평균 계산 및 버퍼 추가
                if current_frame_positions:
                    avg_pos_frame = np.mean(current_frame_positions, axis=0)
                    pos_buffer.append(avg_pos_frame)

            # === Drawing & UI ===
            # 1. 태그 박스 그리기 (깜빡임 방지 적용)
            active_tags_count = 0
            for tag_id in list(tag_memory.keys()):
                tag_info = tag_memory[tag_id]
                time_diff = current_time - tag_info['last_seen']
                
                if time_diff < KEEP_ALIVE_TIME:
                    corners = tag_info['corners']
                    # 방금 인식됨: 초록색 / 잔상: 회색
                    color = (0, 255, 0) if time_diff < 0.1 else (150, 150, 150)
                    cv2.circle(frame, tuple(tag_info['center']), 5, (0, 0, 255), -1)
                    cv2.polylines(frame, [corners], True, color, 2)
                    if time_diff < 0.1: active_tags_count += 1 # 현재 활성 태그 카운트
                else:
                    del tag_memory[tag_id]

            # 2. 세련된 대시보드 그리기
            if pos_buffer:
                # 버퍼에 있는 값들의 평균을 최종 위치로 사용 (스무딩)
                smoothed_cam_pos = np.mean(pos_buffer, axis=0)
                draw_dashboard(frame, smoothed_cam_pos, active_tags_count)

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
    print("🚀 서버 시작: http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)