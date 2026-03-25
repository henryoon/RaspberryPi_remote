import time
import threading
import cv2
import numpy as np
from flask import Flask, Response
from ultralytics import YOLO

try:
    from picamera2 import Picamera2
except ImportError:
    print("오류: 라즈베리 파이 환경에서 'picamera2' 모듈을 찾을 수 없습니다.")
    exit()

class RobotVisionSystem:
    def __init__(self, model_path):
        self.lock = threading.Lock()
        self.output_frame = None
        self.running = True
        
        self.detected_objects = []  
        
        # 목표 ROI 설정 (x_min, y_min, x_max, y_max)
        self.target_roi = (206, 212, 434, 268)
        
        print("⚙️ YOLO 모델 로딩 중...")
        self.model = YOLO(model_path)

        self.width, self.height = 640, 480
        self.picam2 = Picamera2()
        self._setup_camera()

    def _setup_camera(self):
        print("📷 Camera Module 3 초기화...")
        config = self.picam2.create_video_configuration(
            main={"size": (self.width, self.height), "format": "BGR888"}
        )
        self.picam2.configure(config)
        self.picam2.start()

        try:
            self.picam2.set_controls({
                "AfMode": 0, 
                "LensPosition": 5.5 
            })
            print("✅ 렌즈 초점 물리적 고정 완료 (LensPosition: 5.5)")
        except Exception as e:
            print(f"⚠️ AF 설정 오류: {e}")

    def process_loop(self):
        frame_count = 0
        SKIP_FRAMES = 3
        last_display_frame = None

        while self.running:
            raw_frame = self.picam2.capture_array()
            frame = cv2.cvtColor(raw_frame, cv2.COLOR_RGB2BGR)
            if frame is None: continue

            if frame_count % SKIP_FRAMES == 0:
                results = self.model(frame, conf=0.5, imgsz=320, verbose=False)
                
                current_detections = []
                annotated_frame = results[0].plot() 

                roi_x1, roi_y1, roi_x2, roi_y2 = self.target_roi
                cv2.rectangle(annotated_frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 255, 255), 1)
                cv2.putText(annotated_frame, "Target ROI", (roi_x1, roi_y1 - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                is_in_roi = False

                for box in results[0].boxes:
                    coords = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = map(int, coords)
                    
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    
                    conf = box.conf[0].item()
                    label = self.model.names[int(box.cls[0])]

                    current_detections.append({
                        "label": label,
                        "center": (center_x, center_y),
                        "conf": conf
                    })

                    if (roi_x1 <= center_x <= roi_x2) and (roi_y1 <= center_y <= roi_y2):
                        is_in_roi = True
                        status_color = (0, 255, 0) 
                    else:
                        status_color = (0, 0, 255) 

                    cv2.circle(annotated_frame, (center_x, center_y), 5, status_color, -1)
                    cv2.putText(annotated_frame, f"({center_x},{center_y})", 
                                (center_x, center_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, status_color, 1)

                with self.lock:
                    self.detected_objects = current_detections
                
                # 💡 수정된 부분: 터미널 출력뿐만 아니라 화면에도 텍스트 오버레이 적용
                if is_in_roi:
                    print("detected")
                    # ROI 내부에 있을 때 초록색으로 'DETECTED' 출력
                    cv2.putText(annotated_frame, "STATUS: DETECTED", (10, 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                else:
                    print("none")
                    # ROI 외부에 있거나 객체가 없을 때 빨간색으로 'NONE' 출력
                    cv2.putText(annotated_frame, "STATUS: NONE", (10, 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                last_display_frame = annotated_frame
            else:
                annotated_frame = last_display_frame if last_display_frame is not None else frame

            with self.lock:
                self.output_frame = annotated_frame.copy()

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
    return "<h1>Microplate detection</h1><img src='/video_feed' width='640'>"

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