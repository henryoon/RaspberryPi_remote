import time
import cv2
import zmq

try:
    from picamera2 import Picamera2
except ImportError:
    print("오류: picamera2를 찾을 수 없습니다.")
    exit()


class CameraPublisher:
    """Picamera2 영상을 FHD(5555)와 VGA(5556) 두 개의 스트림으로 브로드캐스팅하는 클래스"""

    def __init__(self, host: str = "*"):
        self.host = host
        self.width, self.height = 1920, 1080

        self.context = zmq.Context()
        
        # 1. 고해상도(FHD) 퍼블리셔 (Barcode, AprilTag 용) - 포트 5555
        self.socket_fhd = self.context.socket(zmq.PUB)
        self.socket_fhd.set_hwm(2)
        # self.socket_fhd.bind(f"tcp://{self.host}:5555")
        self.socket_fhd.bind("ipc:///tmp/vision_fhd")

        # 2. 저해상도(VGA) 퍼블리셔 (YOLO 용) - 포트 5556
        self.socket_vga = self.context.socket(zmq.PUB)
        self.socket_vga.set_hwm(2)
        # self.socket_vga.bind(f"tcp://{self.host}:5556")
        self.socket_vga.bind("ipc:///tmp/vision_vga")

        self.picam2 = Picamera2()
        self._setup_camera()

    def _setup_camera(self):
        print(f"📷 Camera Module 3 초기화 ({self.width}x{self.height})...")
        config = self.picam2.create_video_configuration(
            main={"size": (self.width, self.height), "format": "BGR888"}
        )
        self.picam2.configure(config)
        self.picam2.start()
        self._set_af_mode(continuous=True)

    def _set_af_mode(self, continuous: bool = True):
        try:
            if continuous:
                self.picam2.set_controls({"AfMode": 2})
                time.sleep(1.0)
                self.picam2.set_controls({"AfMode": 0, "LensPosition": 5.5})
                status = "초점 보정 후 고정 완료 (LensPosition: 5.5)"
            else:
                self.picam2.set_controls({"AfMode": 0, "LensPosition": 5.5})
                status = "고정 모드 유지"
            print(f"🔄 {status}")
        except Exception as e:
            print(f"⚠️ AF 오류: {e}")

    def start_streaming(self, jpeg_quality: int = 80):
        print(f"🚀 Dual Streaming started -> FHD:5555 | VGA:5556")
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
        
        try:
            while True:
                raw_frame = self.picam2.capture_array()
                if raw_frame is None:
                    continue

                frame_fhd = cv2.cvtColor(raw_frame, cv2.COLOR_RGB2BGR)

                # 1. FHD 데이터 송신
                _, enc_fhd = cv2.imencode(".jpg", frame_fhd, encode_param)
                self.socket_fhd.send(enc_fhd.tobytes())

                # 2. VGA 리사이징 및 데이터 송신
                frame_vga = cv2.resize(frame_fhd, (640, 480), interpolation=cv2.INTER_AREA)
                _, enc_vga = cv2.imencode(".jpg", frame_vga, encode_param)
                self.socket_vga.send(enc_vga.tobytes())

                # 송신 주기를 0.05초(최대 20FPS)로 제한하여 CPU 및 네트워크 과부하 방지
                time.sleep(0.05)
        except KeyboardInterrupt:
            print("\n스트리밍을 중단합니다.")
        finally:
            self.release()

    def release(self):
        self.picam2.stop()
        self.socket_fhd.close()
        self.socket_vga.close()
        self.context.term()
        print("Publisher 자원이 해제되었습니다.")


if __name__ == "__main__":
    pub = CameraPublisher(host="*")
    pub.start_streaming(jpeg_quality=80)