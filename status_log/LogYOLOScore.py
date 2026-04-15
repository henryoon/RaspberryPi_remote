import time
import threading
import cv2
import numpy as np
import zmq
import csv
import os
import psutil  # 시스템 자원 모니터링용 라이브러리
from flask import Flask, Response
from ultralytics import YOLO

try:
    from picamera2 import Picamera2
except ImportError:
    print("오류: 라즈베리 파이 환경에서 'picamera2' 모듈을 찾을 수 없습니다.")
    exit()

class RobotVisionSystem:
    def __init__(self, model_path, dataset_path=None):
        self.lock = threading.Lock()
        self.output_frame = None
        self.running = True
        self.detected_objects = []
        self.model_path = model_path
        self.dataset_path = dataset_path  # mAP 계산을 위한 yaml 경로
        
        # --- Performance Metrics Logging Setup ---
        # 저장 경로 확인 및 파일 초기화
        self.csv_file = "/home/rnd/HJ/status_log/LogData/logyoloscore260415.csv"
        self._init_csv()
        
        # --- ZeroMQ Setup ---
        self.zmq_context = zmq.Context()
        self.zmq_socket = self.zmq_context.socket(zmq.PUB)
        self.zmq_socket.bind("tcp://*:5555")

        # ROI 설정 (x1, y1, x2, y2)
        self.target_roi = (206, 212, 434, 268)
        
        print(f"⚙️ YOLO 모델 로딩 중: {model_path}")
        self.model = YOLO(model_path)

        self.width, self.height = 640, 480
        self.picam2 = Picamera2()
        self._setup_camera()

    def _init_csv(self):
        """CSV 로그 파일 초기화 및 헤더 작성"""
        directory = os.path.dirname(self.csv_file)
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        with open(self.csv_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Timestamp', 'Avg_Conf(%)', 'InferenceTime(ms)', 'TotalLatencyTime(ms)', 
                'FPS', 'CPU_Temp(℃)', 'CPU_Usage(%)', 'Memory_Usage(%)'
            ])

    def _get_cpu_temp(self):
        """라즈베리 파이 시스템 온도를 읽어옴"""
        try:
            with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
                return float(f.read()) / 1000.0
        except FileNotFoundError:
            return 0.0

    def _setup_camera(self):
        print("📷 Camera Module 3 초기화...")
        config = self.picam2.create_video_configuration(
            main={"size": (self.width, self.height), "format": "BGR888"}
        )
        self.picam2.configure(config)
        self.picam2.start()

        try:
            # 렌즈 초점 수동 고정 (Auto-Focus 비활성화)
            self.picam2.set_controls({"AfMode": 0, "LensPosition": 5.5})
            print("✅ 렌즈 초점 물리적 고정 완료 (LensPosition: 5.5)")
        except Exception as e:
            print(f"⚠️ AF 설정 오류: {e}")

    def process_loop(self):
        frame_count = 0
        SKIP_FRAMES = 3  # 연산 부하를 줄이기 위해 3프레임마다 추론
        last_display_frame = None
        prev_time = time.time()

        while self.running:
            loop_start_time = time.time() # Total Latency 측정을 위한 시작점
            
            # 카메라 프레임 획득
            raw_frame = self.picam2.capture_array()
            frame = cv2.cvtColor(raw_frame, cv2.COLOR_RGB2BGR)
            if frame is None: continue

            if frame_count % SKIP_FRAMES == 0:
                # --- 객체 인식 추론 (Inference) ---
                start_inference = time.time()
                results = self.model(frame, conf=0.5, imgsz=320, verbose=False)
                end_inference = time.time()

                # --- 지표 계산 ---
                inf_time_ms = (end_inference - start_inference) * 1000
                total_latency_ms = (time.time() - loop_start_time) * 1000
                
                curr_time = time.time()
                fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
                prev_time = curr_time

                # Confidence 추출 및 평균 계산
                confidences = results[0].boxes.conf.tolist()
                avg_conf = np.mean(confidences) if confidences else 0.0
                
                # 시스템 자원 정보
                cpu_temp = self._get_cpu_temp()
                cpu_usage = psutil.cpu_percent()
                mem_usage = psutil.virtual_memory().percent

                # 실시간 CSV 로그 기록
                self._log_performance(avg_conf, inf_time_ms, total_latency_ms, fps, cpu_temp, cpu_usage, mem_usage)
                
                # 결과 시각화 및 ROI 검사
                current_detections = []
                annotated_frame = results[0].plot() 
                roi_x1, roi_y1, roi_x2, roi_y2 = self.target_roi
                cv2.rectangle(annotated_frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 255, 255), 1)

                is_in_roi = False
                for box in results[0].boxes:
                    coords = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = map(int, coords)
                    center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
                    
                    conf = box.conf[0].item()
                    label = self.model.names[int(box.cls[0])]
                    current_detections.append({"label": label, "center": (center_x, center_y), "conf": conf})

                    if (roi_x1 <= center_x <= roi_x2) and (roi_y1 <= center_y <= roi_y2):
                        is_in_roi = True
                        status_color = (0, 255, 0) # Green
                    else:
                        status_color = (0, 0, 255) # Red
                    cv2.circle(annotated_frame, (center_x, center_y), 5, status_color, -1)

                # ZeroMQ 상태 전송
                status_msg = "detected" if is_in_roi else "none"
                self.zmq_socket.send_multipart([b"status", status_msg.encode('utf-8')])

                # 화면 정보 오버레이
                self._draw_status(annotated_frame, is_in_roi, fps, inf_time_ms, cpu_temp)

                with self.lock:
                    self.detected_objects = current_detections
                last_display_frame = annotated_frame
            else:
                annotated_frame = last_display_frame if last_display_frame is not None else frame

            with self.lock:
                self.output_frame = annotated_frame.copy()

            frame_count += 1
            time.sleep(0.01)

    def _log_performance(self, avg_conf, inf_time, total_lat, fps, temp, cpu, mem):
        """7가지 지표를 CSV에 추가 기록"""
        try:
            with open(self.csv_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                    round(avg_conf*100, 2),
                    round(inf_time, 1),
                    round(total_lat, 1),
                    round(fps, 2),
                    round(temp, 1),
                    cpu,
                    mem
                ])
        except Exception as e:
            print(f"CSV 쓰기 오류: {e}")

    def _draw_status(self, frame, is_in_roi, fps, inf_time, temp):
        """프레임 위에 실시간 정보 표시"""
        status_color = (0, 255, 0) if is_in_roi else (0, 0, 255)
        status_text = f"STATUS: {'DETECTED' if is_in_roi else 'NONE'}"
        perf_text = f"FPS: {fps:.1f} | Inf: {inf_time:.1f}ms | Temp: {temp:.1f}C"
        
        cv2.putText(frame, status_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        cv2.putText(frame, perf_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    def calculate_map(self):
        """시스템 종료 전 검증 데이터셋을 통한 mAP 평가"""
        if self.dataset_path and os.path.exists(self.dataset_path):
            print("\n📊 Validation mAP 측정 시작...")
            val_results = self.model.val(data=self.dataset_path, verbose=False)
            map50 = val_results.results_dict['metrics/mAP50(B)']
            map50_95 = val_results.results_dict['metrics/mAP50-95(B)']
            
            print(f"✅ mAP50: {map50:.4f} | mAP50-95: {map50_95:.4f}")
            
            with open("final_report.txt", "w") as f:
                f.write(f"Model: {self.model_path}\n")
                f.write(f"mAP50: {map50:.4f}\n mAP50-95: {map50_95:.4f}\n")
        else:
            print("\n⚠️ mAP 계산을 위한 dataset_path가 유효하지 않습니다.")

    def stop(self):
        self.running = False
        self.calculate_map()
        self.picam2.stop()
        print("👋 시스템을 성공적으로 종료합니다.")

    def get_stream_frame(self):
        with self.lock:
            if self.output_frame is None: return None
            success, buffer = cv2.imencode(".jpg", self.output_frame)
            return bytearray(buffer) if success else None

# --- Flask Server ---
app = Flask(__name__)
vision = RobotVisionSystem(
    model_path='/home/rnd/yolo_model/best_yolov26n.pt',
    dataset_path='/home/rnd/yolo_model/data.yaml' 
)

@app.route("/")
def index():
    return "<h1>Robot Vision Monitoring</h1><img src='/video_feed' width='640'>"

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
    
    try:
        app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
    except KeyboardInterrupt:
        vision.stop()