import time
import cv2
import numpy as np
from multiprocessing import shared_memory

try:
    from picamera2 import Picamera2
except ImportError:
    print("오류: picamera2를 찾을 수 없습니다.")
    exit()


class CameraPublisherSHM:
    """Picamera2 영상을 FHD와 VGA 두 개의 Shared Memory 블록으로 매핑하는 클래스"""

    def __init__(self):
        self.width, self.height = 1920, 1080
        self.vga_w, self.vga_h = 640, 480

        # RGB 8비트 채널(3) 크기 계산
        self.size_fhd = self.width * self.height * 3
        self.size_vga = self.vga_w * self.vga_h * 3

        # 공유 메모리 블록 생성 (기존에 비정상 종료된 메모리가 있다면 정리 후 생성)
        self.shm_fhd = self._init_shared_memory("vision_fhd", self.size_fhd)
        self.shm_vga = self._init_shared_memory("vision_vga", self.size_vga)

        # 공유 메모리 버퍼를 NumPy 배열로 매핑
        self.frame_fhd_shared = np.ndarray((self.height, self.width, 3), dtype=np.uint8, buffer=self.shm_fhd.buf)
        self.frame_vga_shared = np.ndarray((self.vga_h, self.vga_w, 3), dtype=np.uint8, buffer=self.shm_vga.buf)

        self.picam2 = Picamera2()
        self._setup_camera()

    def _init_shared_memory(self, name: str, size: int):
        """기존 메모리 블록 충돌 방지를 포함한 Shared Memory 초기화 로직"""
        try:
            return shared_memory.SharedMemory(create=True, size=size, name=name)
        except FileExistsError:
            print(f"⚠️ 기존 공유 메모리 '{name}' 정리 중...")
            shm = shared_memory.SharedMemory(name=name)
            shm.unlink()
            return shared_memory.SharedMemory(create=True, size=size, name=name)

    def _setup_camera(self):
        print(f"📷 Camera Module 3 초기화 ({self.width}x{self.height})...")
        config = self.picam2.create_video_configuration(
            main={"size": (self.width, self.height), "format": "BGR888"}
        )
        self.picam2.configure(config)
        self.picam2.start()
        
        # AF 모드 고정
        try:
            self.picam2.set_controls({"AfMode": 2})
            time.sleep(1.0)
            self.picam2.set_controls({"AfMode": 0, "LensPosition": 5.5})
            print("🔄 초점 보정 후 고정 완료 (LensPosition: 5.5)")
        except Exception as e:
            print(f"⚠️ AF 오류: {e}")

    def start_streaming(self):
        print(f"🚀 Shared Memory Streaming started -> [vision_fhd], [vision_vga]")
        
        try:
            while True:
                raw_frame = self.picam2.capture_array()
                if raw_frame is None:
                    continue

                frame_fhd = cv2.cvtColor(raw_frame, cv2.COLOR_RGB2BGR)
                frame_vga = cv2.resize(frame_fhd, (self.vga_w, self.vga_h), interpolation=cv2.INTER_AREA)

                # 💡 Shared Memory 버퍼에 직접 복사 (인코딩 오버헤드 0)
                np.copyto(self.frame_fhd_shared, frame_fhd)
                np.copyto(self.frame_vga_shared, frame_vga)

                # 송신 주기를 0.05초(최대 20FPS)로 제한
                time.sleep(0.05)
        except KeyboardInterrupt:
            print("\n스트리밍을 중단합니다.")
        finally:
            self.release()

    def release(self):
        self.picam2.stop()
        self.shm_fhd.close()
        self.shm_fhd.unlink()
        self.shm_vga.close()
        self.shm_vga.unlink()
        print("Publisher 공유 메모리 자원이 안전하게 해제되었습니다.")


if __name__ == "__main__":
    pub = CameraPublisherSHM()
    pub.start_streaming()