import time
import threading
import cv2
import numpy as np
import zmq
from flask import Flask, Response, render_template_string
from pyzbar import pyzbar

class ZMQReceiverThread_FHD:
    """백그라운드에서 FHD(1920x1080) 해상도를 원본 그대로 수신하는 클래스"""
    def __init__(self, ip="localhost", port=5555):
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
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                
                if frame is not None:
                    with self.lock:
                        self.latest_frame = frame
            except Exception as e:
                break

    def read(self):
        with self.lock:
            return self.latest_frame

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

        # 비동기 수신기 초기화 (포트 5555)
        self.receiver = ZMQReceiverThread_FHD(ip=ip, port=5555)

    def process_loop(self):
        print(f"📥 바코드 수신 시작 (비동기 처리)")
        while self.running:
            frame = self.receiver.read()
            if frame is None:
                time.sleep(0.01)
                continue

            h, w, _ = frame.shape
            x1, y1 = (w - self.roi_x) // 2, (h - self.roi_y) // 2 + 30
            x2, y2 = x1 + self.roi_x, y1 + self.roi_y

            display_main = frame.copy()
            cv2.rectangle(display_main, (x1, y1), (x2, y2), (255, 0, 0), 2)

            roi_img = frame[y1:y2, x1:x2]
            zoomed_roi = cv2.resize(roi_img, (self.roi_x * self.scale, self.roi_y * self.scale), interpolation=cv2.INTER_CUBIC)

            decoded = pyzbar.decode(zoomed_roi)
            current_time = time.time()

            if len(decoded) > 0:
                self.last_detection_time = current_time
                self.is_focus_locked = True
            else:
                if self.is_focus_locked and (current_time - self.last_detection_time >= self.lock_duration):
                    self.is_focus_locked = False

            for obj in decoded:
                data = obj.data.decode("utf-8")
                cv2.putText(zoomed_roi, f"DATA: {data}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            with self.lock:
                self.main_frame = display_main
                self.zoomed_frame = zoomed_roi

            time.sleep(0.01)

    def get_frame(self, target="main"):
        with self.lock:
            frame = self.main_frame if target == "main" else self.zoomed_frame
            if frame is None: return None
            _, buffer = cv2.imencode(".jpg", frame)
            return bytearray(buffer)

    def release(self):
        self.running = False
        self.receiver.stop()


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