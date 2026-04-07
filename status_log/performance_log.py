import time
import csv
import os

class InferencePerformanceLogger:
    """
    라즈베리 파이 추론 성능 및 시스템 상태를 CSV로 기록하는 클래스
    """
    def __init__(self, file_name="inference_log.csv"):
        self.file_name = file_name
        self.headers = [
            "Timestamp", 
            "Frame_ID", 
            "Inference_Latency_ms", 
            "FPS", 
            "Throughput_Total_Frames", 
            "CPU_Temperature_C"
        ]
        self.start_experiment_time = time.time()
        self.frame_count = 0
        
        # CSV 파일 초기화 및 헤더 작성
        self._initialize_csv()

    def _initialize_csv(self):
        # 파일이 이미 존재하면 덮어쓰지 않고 이어서 쓰거나 새로 생성
        file_exists = os.path.isfile(self.file_name)
        with open(self.file_name, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.headers)

    def _get_cpu_temp(self):
        """라즈베리 파이 시스템 온도를 읽어옴"""
        try:
            with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
                temp = int(f.read()) / 1000.0
            return temp
        except FileNotFoundError:
            return 0.0

    def log_inference(self, latency_ms):
        """
        추론 한 주기 결과를 기록
        :param latency_ms: 추론에 소요된 시간 (miliseconds)
        """
        self.frame_count += 1
        current_time = time.time()
        timestamp = current_time - self.start_experiment_time
        
        # FPS 계산 (순간 FPS)
        fps = 1000.0 / latency_ms if latency_ms > 0 else 0
        
        # 시스템 온도 확인
        cpu_temp = self._get_cpu_temp()

        # 데이터 행 구성
        row = [
            round(timestamp, 3),        # 실험 시작 후 경과 시간
            self.frame_count,            # 현재 프레임 번호
            round(latency_ms, 2),        # 추론 지연 시간
            round(fps, 2),               # 프레임 속도
            self.frame_count,            # 누적 처리량 (Throughput)
            round(cpu_temp, 2)           # CPU 온도
        ]

        with open(self.file_name, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

# --- 사용 예시 (Inference Loop) ---
if __name__ == "__main__":
    # 로거 인스턴스 생성 (쿨링 유무에 따라 파일명을 다르게 설정)
    logger = InferencePerformanceLogger("/home/rnd/HJ/status_log/yolov26_no_cooling_test.csv")

    try:
        print("Starting Inference Logging... (Press Ctrl+C to stop)")
        while True:
            # 추론 시작 시간 측정
            start_time = time.perf_counter()
            
            # ----------------------------------------------
            # 여기에 YOLOv26 추론 코드 실행 (더미 코드로 대체)
            # results = model.predict(frame) 
            time.sleep(0.05) # 예: 20 FPS 상황 가정 (50ms 소요)
            # ----------------------------------------------
            
            # 추론 종료 시간 측정 및 지연 시간 계산
            end_time = time.perf_counter()
            latency = (end_time - start_time) * 1000  # ms 단위 변환
            
            # 로그 기록
            logger.log_inference(latency)
            
    except KeyboardInterrupt:
        print("\nLogging finished.")