import time
import threading
import cv2
import numpy as np
# import paho.mqtt.client as mqtt
from flask import Flask, Response
from ultralytics import YOLO

try:
    from picamera2 import Picamera2
except ImportError:
    print("오류: 라즈베리 파이 환경에서 'picamera2' 모듈을 찾을 수 없습니다.")
    exit()

class RobotVisionSystem:
    def __init__(self, model_path):
        # 1. 상태 및 동기화 초기화
        self.lock = threading.Lock()
        self.output_frame = None
        self.running = True
        
        # 2. 객체 정보 저장용 변수
        self.detected_objects = []  # [{label, center_x, center_y, conf}, ...]
        
        # 3. 모델 로드
        print("⚙️ YOLO 모델 로딩 중...")
        self.model = YOLO(model_path)

        # 4. 카메라 설정
        self.width, self.height = 640, 480
        self.picam2 = Picamera2()
        self._setup_camera()

    def _setup_camera(self):
        """Camera Module 3 및 중앙 집중 AF 설정"""
        print("📷 Camera Module 3 초기화...")
        config = self.picam2.create_video_configuration(
            main={"size": (self.width, self.height), "format": "BGR888"}
        )
        self.picam2.configure(config)
        self.picam2.start()

        try:
            # 중앙 영역에 초점을 맞추기 위한 AfWindows 설정 (0~1000 상대 좌표)
            # 사각형 하나를 포함하는 리스트의 리스트 [[x, y, w, h]] 형태
            self.picam2.set_controls({
                "AfMode": 2,                
                "AfRange": 0,               
                "AfWindows": [[350, 350, 300, 300]] 
            })
            print("✅ 중앙 집중 Auto Focus 활성화")
        except Exception as e:
            print(f"⚠️ AF 설정 오류: {e}")

    def process_loop(self):
        """메인 추론 및 중심점 계산 루프"""
        frame_count = 0
        SKIP_FRAMES = 3
        last_display_frame = None

        while self.running:
            raw_frame = self.picam2.capture_array()
            frame = cv2.cvtColor(raw_frame, cv2.COLOR_RGB2BGR)
            if frame is None: continue

            if frame_count % SKIP_FRAMES == 0:
                # 1. YOLO 추론
                results = self.model(frame, conf=0.5, imgsz=320, verbose=False)
                
                current_detections = []
                display_frame = frame.copy()

                # 2. 바운딩 박스 정보 파싱 및 중심점 계산
                for box in results[0].boxes:
                    # 좌표 추출 (x1, y1, x2, y2)
                    coords = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = map(int, coords)
                    
                    # 중심점(Centroid) 계산
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    
                    conf = box.conf[0].item()
                    label = self.model.names[int(box.cls[0])]

                    current_detections.append({
                        "label": label,
                        "center": (center_x, center_y),
                        "conf": conf
                    })

                    # 3. 시각화 피드백 추가 (중심점 표시)
                    cv2.circle(display_frame, (center_x, center_y), 5, (0, 0, 255), -1)
                    cv2.putText(display_frame, f"{label} ({center_x},{center_y})", 
                                (x1 + 200, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # 4. 정보 공유 및 출력 (이 부분에서 Publish 수행 가능)
                with self.lock:
                    self.detected_objects = current_detections
                
                if current_detections:
                    # 터미널에 중심점 좌표 출력 (Robot Task Planning에 활용 가능)
                    # mqtt_client.publish("robot/vision/detections", str(current_detections))
                    for obj in current_detections:
                        print(f"🎯 [PUB] {obj['label']} detected at: {obj['center']}")

                # 5. 결과 프레임 저장
                annotated_frame = results[0].plot() # 기본 바운딩 박스
                # 직접 그린 중심점 정보를 합침
                combined_frame = cv2.addWeighted(annotated_frame, 0.8, display_frame, 0.2, 0)
                last_display_frame = combined_frame
            else:
                combined_frame = last_display_frame if last_display_frame is not None else frame

            with self.lock:
                self.output_frame = combined_frame.copy()

            frame_count += 1
            time.sleep(0.01)

    def get_stream_frame(self):
        with self.lock:
            if self.output_frame is None: return None
            success, buffer = cv2.imencode(".jpg", self.output_frame)
            return bytearray(buffer) if success else None

# --- Flask Server ---
app = Flask(__name__)
# 실제 모델 경로로 수정하세요
vision = RobotVisionSystem(model_path='/home/rnd/yolo_model/best_yolov26n.pt')

@app.route("/")
def index():
    return "<h1>Robot Centroid Vision System</h1><img src='/video_feed' width='640'>"

@app.route("/video_feed")
def video_feed():
    def gen():
        while True:
            frame = vision.get_stream_frame()
            if frame:
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.03)
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    t = threading.Thread(target=vision.process_loop)
    t.daemon = True
    t.start()
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)