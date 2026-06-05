import time
import threading
import cv2
import numpy as np
import zmq
from flask import Flask, Response
from ultralytics import YOLO


class ZMQReceiverThread:
    """ZeroMQ 데이터를 바이트 형태로만 유지하여 디코딩 오버헤드를 없앤 클래스"""
    def __init__(self, ip="localhost", port=5556):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.set_hwm(2)
        # self.socket.connect(f"tcp://{ip}:{port}")
        self.socket.connect("ipc:///tmp/vision_vga")
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")

        self.latest_packet = None
        self.running = True
        self.lock = threading.Lock()

        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            try:
                # 💡 데이터를 받기만 하고, 여기서 무거운 imdecode를 하지 않습니다.
                packet = self.socket.recv()
                with self.lock:
                    self.latest_packet = packet
            except Exception as e:
                print(f"⚠️ ZMQ 수신 에러: {e}")
                break

    def read_and_decode(self):
        """필요할 때만 최신 패킷을 꺼내어 디코딩합니다."""
        with self.lock:
            packet = self.latest_packet
            # 중복 디코딩 방지를 위해 사용한 패킷은 초기화
            self.latest_packet = None 

        if packet is not None:
            np_arr = np.frombuffer(packet, dtype=np.uint8)
            return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return None

    def stop(self):
        self.running = False
        self.thread.join()
        self.socket.close()
        self.context.term()


class OptimizedRobotVision:
    def __init__(self, model_path: str, ip: str = "localhost"):
        self.lock = threading.Lock()
        self.output_frame = None
        self.running = True
        self.target_roi = (206, 212, 434, 268)
        
        # 💡 목표 추론 속도 설정 (예: 10 FPS = 0.1초 대기)
        self.target_fps = 10.0
        self.frame_time = 1.0 / self.target_fps

        print(f"⚙️ YOLO 모델 로딩 중... ({model_path})")
        self.model = YOLO(model_path, task='detect')
        self.receiver = ZMQReceiverThread(ip=ip, port=5556)

    def process_loop(self):
        print("📥 YOLO 추론 루프 시작 (Lazy Decoding & Target FPS 적용)...")
        prev_time = time.time()

        while self.running:
            loop_start = time.time()

            # 💡 이 시점에만 디코딩을 수행 (CPU 절약)
            frame = self.receiver.read_and_decode()
            if frame is None:
                time.sleep(0.01)
                continue

            results_generator = self.model(frame, conf=0.4, verbose=False, stream=True)
            
            for result in results_generator:
                # 💡 무거운 result.plot() 제거, 원본 프레임 복사본 사용
                annotated_frame = frame.copy()

                roi_x1, roi_y1, roi_x2, roi_y2 = self.target_roi
                cv2.rectangle(annotated_frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 255, 255), 1)

                is_in_roi = False
                
                for box in result.boxes:
                    coords = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = map(int, coords)
                    center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)

                    if (roi_x1 <= center_x <= roi_x2) and (roi_y1 <= center_y <= roi_y2):
                        is_in_roi = True
                        color = (0, 255, 0)
                    else:
                        color = (0, 0, 255)
                    cv2.circle(annotated_frame, (center_x, center_y), 5, color, -1)

                curr_time = time.time()
                fps = 1.0 / (curr_time - prev_time)
                prev_time = curr_time

                if is_in_roi:
                    cv2.putText(annotated_frame, "STATUS: DETECTED", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                else:
                    cv2.putText(annotated_frame, "STATUS: NONE", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                # 💡 수정: 루프 내에서 딱 1번만 JPEG로 압축하여 저장 (Caching)
                success, buffer = cv2.imencode(".jpg", annotated_frame)
                if success:
                    encoded_jpg = bytearray(buffer)
                    with self.lock:
                        self.output_frame = encoded_jpg # 배열이 아닌 압축된 바이트 자체를 저장

            elapsed_time = time.time() - loop_start
            sleep_duration = max(0.01, self.frame_time - elapsed_time)
            time.sleep(sleep_duration)

    def get_stream_frame(self):
        with self.lock:
            # 💡 수정: 연산 없이 이미 압축된 바이트만 즉시 반환
            return self.output_frame

    def release(self):
        self.running = False
        self.receiver.stop()


# --- Flask Web Server ---
app = Flask(__name__)
vision_sub = OptimizedRobotVision(model_path='/home/rnd/yolo_model/best_yolov26n.onnx', ip="localhost")

@app.route("/")
def index():
    return "<h1>Microplate detection</h1><img src='/video_feed' width='640'>"

@app.route("/video_feed")
def video_feed():
    def gen():
        while True:
            frame = vision_sub.get_stream_frame()
            if frame:
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            # 💡 수정: YOLO 추론 속도(약 10FPS)에 맞춰 웹 송출 속도도 낮춤 (0.04 -> 0.1)
            time.sleep(0.1) 
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    t = threading.Thread(target=vision_sub.process_loop)
    t.daemon = True
    t.start()
    app.run(host="0.0.0.0", port=5002, debug=False, use_reloader=False)