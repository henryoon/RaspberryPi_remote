import time
import threading
import cv2
import numpy as np
import zmq
from flask import Flask, Response
from ultralytics import YOLO


class ZMQReceiverThread:
    """ZeroMQ 데이터를 백그라운드에서 수신하고 가장 최신 프레임만 유지하는 클래스"""
    # 💡 YOLO는 VGA 전용 스트림(5556)에 연결합니다.
    def __init__(self, ip="localhost", port=5556):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.set_hwm(2)
        self.socket.connect(f"tcp://{ip}:{port}")
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")

        self.latest_frame = None
        self.running = True
        self.lock = threading.Lock()

        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            try:
                packet = self.socket.recv()
                np_arr = np.frombuffer(packet, dtype=np.uint8)
                frame_vga = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                
                if frame_vga is not None:
                    # 💡 Publisher가 640x480으로 보내주므로 리사이징 생략
                    with self.lock:
                        self.latest_frame = frame_vga
            except Exception as e:
                print(f"⚠️ ZMQ 수신 에러: {e}")
                break

    def read(self):
        with self.lock:
            return self.latest_frame

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

        print(f"⚙️ YOLO 모델 로딩 중... ({model_path})")
        self.model = YOLO(model_path, task='detect')
        self.receiver = ZMQReceiverThread(ip=ip, port=5556)

    def process_loop(self):
        print("📥 YOLO 추론 루프 시작 (ONNX 모드)...")
        prev_time = time.time()

        while self.running:
            frame = self.receiver.read()
            if frame is None:
                time.sleep(0.01)
                continue

            # 💡 stream=True를 유지하여 메모리 효율성 극대화
            results_generator = self.model(frame, conf=0.5, verbose=False, stream=True)
            
            # 제너레이터에서 첫 번째(이자 유일한) 프레임의 결과 추출
            for result in results_generator:
                annotated_frame = result.plot()

                roi_x1, roi_y1, roi_x2, roi_y2 = self.target_roi
                cv2.rectangle(annotated_frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 255, 255), 1)

                is_in_roi = False
                
                # 💡 result 객체의 boxes에 직접 접근
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

                # cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 50),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                if is_in_roi:
                    cv2.putText(annotated_frame, "STATUS: DETECTED", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                else:
                    cv2.putText(annotated_frame, "STATUS: NONE", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                with self.lock:
                    self.output_frame = annotated_frame.copy()

            time.sleep(0.01)

    def get_stream_frame(self):
        with self.lock:
            if self.output_frame is None: return None
            success, buffer = cv2.imencode(".jpg", self.output_frame)
            return bytearray(buffer) if success else None

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
            time.sleep(0.03)
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    t = threading.Thread(target=vision_sub.process_loop)
    t.daemon = True
    t.start()
    app.run(host="0.0.0.0", port=5002, debug=False, use_reloader=False)