import psutil
import time
import csv
import os
from datetime import datetime

class YoloHybridMonitor:
    """
    시스템 상태와 추론 지표는 실시간으로 기록하고,
    mAP는 프로세스 종료 시 최종적으로 기록하는 하이브리드 모니터입니다.
    """
    def __init__(self, file_path, model_name="YOLOv11"):
        self.file_path = file_path
        self.directory = os.path.dirname(self.file_path)
        self.model_name = model_name
        
        # 실시간 기록용 헤더
        self.header = [
            'Timestamp', 'Model', 'CPU Usage (%)', 'Memory Usage (%)', 
            'Temperature (°C)', 'Inference Time (ms)', 'FPS'
        ]
        
        # 평균 계산을 위한 데이터 보관
        self.inference_buffer = []
        
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

    def get_cpu_temp(self):
        try:
            with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
                temp = int(f.read()) / 1000.0
            return round(temp, 1)
        except:
            return 0.0

    def log_realtime_status(self, inf_time, fps):
        """시스템 상태와 실시간 추론 성능을 CSV에 즉시 저장합니다."""
        file_exists = os.path.isfile(self.file_path)
        self.inference_buffer.append(inf_time) # 평균 산출용

        try:
            with open(self.file_path, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(self.header)

                now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                row = [
                    now, self.model_name, 
                    psutil.cpu_percent(), 
                    psutil.virtual_memory().percent, 
                    self.get_cpu_temp(), 
                    round(inf_time, 2), 
                    round(fps, 2)
                ]
                writer.writerow(row)
                f.flush() # 실시간 데이터 보존
        except Exception as e:
            print(f"Logging Error: {e}")

    def finalize_with_map(self, map_score):
        """종료 시 호출되어 mAP와 전체 평균 지표를 하단에 기록합니다."""
        if not self.inference_buffer:
            return

        avg_inf = sum(self.inference_buffer) / len(self.inference_buffer)
        
        try:
            with open(self.file_path, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([]) # 가독성을 위한 빈 줄
                writer.writerow(['--- Final Summary Report ---'])
                writer.writerow(['Total Avg Inference (ms)', 'Final mAP Score'])
                writer.writerow([round(avg_inf, 2), round(map_score, 4)])
            
            print(f"\n[Final Report] mAP: {map_score} 및 평균 지표가 {self.file_path}에 추가되었습니다.")
        except Exception as e:
            print(f"Finalizing Error: {e}")

# --- 실행 환경 예시 ---
if __name__ == "__main__":
    SAVE_PATH = '/home/rnd/HJ/status_log/LogData/YOLO_Raspi_log.csv'
    monitor = YoloHybridMonitor(SAVE_PATH)

    print("Monitoring and Inference Started. Press Ctrl+C to stop.")
    
    try:
        while True:
            # 1. 추론 시간 측정 (예시 데이터)
            start_t = time.time()
            time.sleep(0.1) # 실제 model.predict() 가동 가정
            end_t = time.time()
            
            inf_time = (end_t - start_t) * 1000
            fps = 1.0 / (end_t - start_t)

            # 2. 실시간 기록 (mAP 제외 항목들)
            monitor.log_realtime_status(inf_time, fps)
            
    except KeyboardInterrupt:
        # 3. 종료 시 mAP 산출 및 마지막 기록
        print("\nTerminating... Calculating mAP.")
        
        # 실제 환경: val_results = model.val(); target_map = val_results.results_dict['metrics/mAP50-95(B)']
        target_map = 0.925  # 임의의 최종 mAP 결과값
        
        monitor.finalize_with_map(target_map)