import threading
import time
import cv2
import numpy as np
from flask import Flask, Response, render_template_string
import zmq
from pupil_apriltags import Detector  # 💡 원상 복구된 pupil_apriltags


class ZMQReceiverThread_Resize:
    """백그라운드에서 수신 후 1080x720으로 축소하여 저장하는 비동기 수신 클래스"""
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
                frame_fhd = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                
                if frame_fhd is not None:
                    frame = cv2.resize(frame_fhd, (1080, 720), interpolation=cv2.INTER_AREA)
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


class KalmanFilter1D:
    def __init__(self, process_noise=1e-3, measurement_noise=0.3):
        self.x, self.p = 0.0, 1.0
        self.q, self.r = process_noise, measurement_noise
        self.initialized = False

    def update(self, measurement):
        if not self.initialized:
            self.x = measurement
            self.initialized = True
            return self.x
        self.p += self.q
        k = self.p / (self.p + self.r)
        self.x += k * (measurement - self.x)
        self.p *= (1 - k)
        return self.x


class AprilTagPoseSubscriber:
    def __init__(self, ip: str = "localhost"):
        self.lock = threading.Lock()
        self.output_frame = None
        self.running = True

        self.target_w, self.target_h = 1080, 720
        self.tag_size = 0.03
        self.half_s = self.tag_size / 2.0

        scale_x = self.target_w / 1920.0
        scale_y = self.target_h / 1080.0
        focal_length_x = 1522.36 * (1920 / 1280) * scale_x
        focal_length_y = 1520.01 * (1080 / 720) * scale_y
        center_x = 612.83 * (1920 / 1280) * scale_x
        center_y = 370.20 * (1080 / 720) * scale_y

        self.camera_matrix = np.array(
            [[focal_length_x, 0, center_x], [0, focal_length_y, center_y], [0, 0, 1]], dtype=np.float32
        )
        self.dist_coeffs = np.array([[-0.040964, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        
        self.obj_points = np.array(
            [[-self.half_s, -self.half_s, 0], [self.half_s, -self.half_s, 0],
             [self.half_s, self.half_s, 0], [-self.half_s, self.half_s, 0]], dtype=np.float32
        )

        # 💡 pupil_apriltags 디텍터 초기화
        self.detector = Detector(
            families='tag36h11',
            nthreads=1,
            quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0
        )
        
        self.kf_dict = {}

        # 비동기 수신기 초기화 (포트 5555)
        self.receiver = ZMQReceiverThread_Resize(ip=ip, port=5555)

    def process_loop(self):
        print(f"📥 AprilTag 추론 시작 (비동기 수신, pupil_apriltags 사용)")
        while self.running:
            frame = self.receiver.read()
            if frame is None:
                time.sleep(0.01)
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 💡 pupil_apriltags 알고리즘으로 태그 검출
            tags = self.detector.detect(gray, estimate_tag_pose=False)

            for i, tag in enumerate(tags):
                tag_id = tag.tag_id
                # corners는 (4, 2) 형태의 numpy array
                img_points = tag.corners

                # 태그 외곽선 및 중심점 그리기
                corners_int = img_points.astype(int)
                for j in range(4):
                    cv2.line(frame, tuple(corners_int[j]), tuple(corners_int[(j+1)%4]), (0, 255, 0), 2)
                
                cx, cy = int(tag.center[0]), int(tag.center[1])
                cv2.circle(frame, (cx, cy), 6, (255, 255, 255), -1)

                if tag_id not in self.kf_dict:
                    self.kf_dict[tag_id] = {
                        "x": KalmanFilter1D(), "y": KalmanFilter1D(), "z": KalmanFilter1D(),
                        "r": KalmanFilter1D(1e-3, 0.5), "p": KalmanFilter1D(1e-3, 0.5), "yw": KalmanFilter1D(1e-3, 0.5),
                    }

                # solvePnP를 이용한 자세 추정
                success, rvec, tvec = cv2.solvePnP(
                    self.obj_points, img_points, self.camera_matrix, self.dist_coeffs, flags=cv2.SOLVEPNP_SQPNP
                )
                
                if success:
                    R, _ = cv2.Rodrigues(rvec)
                    camera_pos = -np.dot(R.T, tvec)
                    raw_cam_x, raw_cam_y, raw_cam_z = camera_pos[0][0]*1000, camera_pos[1][0]*1000, -camera_pos[2][0]*1000
                    
                    proj_matrix = np.hstack((R, tvec))
                    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)
                    
                    kf = self.kf_dict[tag_id]
                    cam_x, cam_y, cam_z = kf["x"].update(raw_cam_x), kf["y"].update(raw_cam_y), kf["z"].update(raw_cam_z)
                    roll, pitch, yaw = kf["r"].update(euler_angles[2][0]), kf["p"].update(euler_angles[0][0]), kf["yw"].update(euler_angles[1][0])

                    coord_text = f"ID:{tag_id} | X: {cam_x:.0f}, Y: {cam_y:.0f}, Z: {cam_z:.0f}"
                    text_y = self.target_h - 50 - (i * 60)
                    cv2.putText(frame, coord_text, (20, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            with self.lock:
                self.output_frame = frame.copy()
            
            time.sleep(0.01)

    def get_frame(self):
        with self.lock:
            if self.output_frame is None: return None
            _, buffer = cv2.imencode(".jpg", self.output_frame)
            return bytearray(buffer)

    def release(self):
        self.running = False
        self.receiver.stop()


# --- Flask Web Server ---
app = Flask(__name__)
vision_sub = AprilTagPoseSubscriber(ip="localhost")

@app.route("/")
def index():
    return render_template_string("""
    <html><body style="background-color:#111; color:white; text-align:center;">
    <h2>🎯 AprilTag</h2><img src="/video_feed" width="800"></body></html>
    """)

@app.route("/video_feed")
def video_feed():
    def gen():
        while True:
            frame = vision_sub.get_frame()
            if frame: yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            time.sleep(0.04)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    t = threading.Thread(target=vision_sub.process_loop)
    t.daemon = True
    t.start()
    app.run(host="0.0.0.0", port=5000)