import time
import threading
import cv2
import numpy as np
import zmq
import json
from flask import Flask, Response, render_template_string
from pyzbar import pyzbar

class ZMQReceiverThread_LazyFHD:
    def __init__(self, ip="localhost", port=5555):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.set_hwm(2)
        self.socket.connect(f"tcp://{ip}:{port}")
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")

        self.latest_packet = None
        self.running = True
        self.lock = threading.Lock()

        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            try:
                packet = self.socket.recv()
                with self.lock:
                    self.latest_packet = packet
            except Exception:
                break

    def read_and_decode(self):
        with self.lock:
            packet = self.latest_packet
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


class BarcodeDualVisionSubscriber:
    def __init__(self, ip: str = "localhost"):
        self.lock = threading.Lock()
        self.main_frame = None
        self.zoomed_frame = None
        self.running = True

        self.is_focus_locked = False
        self.last_detection_time = 0
        self.lock_duration = 30

        self.width, self.height = 1920, 1080
        self.roi_x, self.roi_y = 400, 120
        self.scale = 3
        
        self.target_fps = 5.0
        self.frame_time = 1.0 / self.target_fps

        self.receiver = ZMQReceiverThread_LazyFHD(ip=ip, port=5555)

        self.pub_context = zmq.Context()
        self.barcode_pub_socket = self.pub_context.socket(zmq.PUB)
        self.barcode_pub_socket.set_hwm(10)
        self.pub_port = 5558
        self.barcode_pub_socket.bind(f"tcp://*:{self.pub_port}")
        print(f"📡 바코드 전송 퍼블리셔 열림 (포트 {self.pub_port})")

    def process_loop(self):
        print(f"📥 바코드 수신 시작 (Lazy Decoding & Target FPS 5)")
        while self.running:
            loop_start = time.time()

            frame = self.receiver.read_and_decode()
            if frame is None:
                time.sleep(0.01)
                continue

            h, w, _ = frame.shape
            x1, y1 = (w - self.roi_x) // 2, (h - self.roi_y) // 2 + 30
            x2, y2 = x1 + self.roi_x, y1 + self.roi_y

            roi_img = frame[y1:y2, x1:x2]
            zoomed_roi = cv2.resize(roi_img, (self.roi_x * self.scale, self.roi_y * self.scale), interpolation=cv2.INTER_CUBIC)

            decoded = pyzbar.decode(zoomed_roi)
            current_time = time.time()

            detected_barcodes = []

            # 💡 바코드 감지 여부에 따라 ROI 색상 결정
            if len(decoded) > 0:
                self.last_detection_time = current_time
                self.is_focus_locked = True
                roi_color = (0, 255, 0)  # 초록색 (인식됨)
            else:
                if self.is_focus_locked and (current_time - self.last_detection_time >= self.lock_duration):
                    self.is_focus_locked = False
                roi_color = (255, 255, 255)  # 하얀색 (미인식)

            # 💡 결정된 색상으로 메인 프레임에 ROI 박스 그리기
            display_main = frame.copy()
            cv2.rectangle(display_main, (x1, y1), (x2, y2), roi_color, 2)

            for obj in decoded:
                data = obj.data.decode("utf-8")
                detected_barcodes.append(data)
                cv2.putText(zoomed_roi, f"DATA: {data}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            if detected_barcodes:
                payload = json.dumps({
                    "timestamp": current_time,
                    "barcodes": detected_barcodes
                })
                self.barcode_pub_socket.send_string(payload)

            with self.lock:
                self.main_frame = display_main
                self.zoomed_frame = zoomed_roi

            elapsed_time = time.time() - loop_start
            sleep_duration = max(0.01, self.frame_time - elapsed_time)
            time.sleep(sleep_duration)

    def get_frame(self, target="main"):
        with self.lock:
            frame = self.main_frame if target == "main" else self.zoomed_frame
            if frame is None: return None
            _, buffer = cv2.imencode(".jpg", frame)
            return bytearray(buffer)

    def release(self):
        self.running = False
        self.receiver.stop()
        self.barcode_pub_socket.close()
        self.pub_context.term()


# --- Flask Web Server ---
app = Flask(__name__)
vision_sub = BarcodeDualVisionSubscriber(ip="localhost")

@app.route("/")
def index():
    return render_template_string("""
    <html><body style="background-color:#111; color:white; text-align:center;">
    <h2>📸 Barcode</h2>
    <div style="display:flex; justify-content:center; gap:20px;">
      <div><img src="/video_main" width="640"></div>
      <div><img src="/video_zoom" width="480"></div>
    </div></body></html>
    """)

@app.route("/video_main")
def video_main():
    def gen():
        while True:
            frame = vision_sub.get_frame("main")
            if frame: yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            time.sleep(0.04)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/video_zoom")
def video_zoom():
    def gen():
        while True:
            frame = vision_sub.get_frame("zoom")
            if frame: yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            time.sleep(0.04)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    t = threading.Thread(target=vision_sub.process_loop)
    t.daemon = True
    t.start()
    app.run(host="0.0.0.0", port=5001)