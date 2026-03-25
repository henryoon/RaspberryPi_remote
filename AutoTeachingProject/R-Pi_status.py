import psutil
import time
import csv
import os
from datetime import datetime

class RaspberryPiMonitor:
    """
    라즈베리 파이의 시스템 상태(CPU, 메모리, 온도)를 모니터링하고 
    CSV 파일로 저장하는 클래스입니다.
    """
    def __init__(self, file_path):
        self.file_path = file_path
        self.directory = os.path.dirname(self.file_path)
        self.header = ['Timestamp', 'CPU Usage (%)', 'Memory Usage (%)', 'Temperature (°C)']
        
        # 저장 디렉토리가 없으면 생성
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
            print(f"Directory created: {self.directory}")

    def get_cpu_temp(self):
        """CPU 온도를 읽어옵니다. (Linux 시스템 전용)"""
        try:
            # vcgencmd 대신 시스템 파일에서 직접 읽기 (라즈베리 파이 표준 방식)
            with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
                temp = int(f.read()) / 1000.0
            return round(temp, 1)
        except Exception as e:
            print(f"Error reading temperature: {e}")
            return 0.0

    def run(self, interval=1):
        """모니터링을 시작하고 CSV 파일에 기록합니다."""
        print(f"Monitoring started. Saving to: {self.file_path}")
        print("Press Ctrl+C to stop.")

        try:
            # 파일이 없으면 헤더를 작성, 있으면 이어쓰기(append)
            file_exists = os.path.isfile(self.file_path)
            
            with open(self.file_path, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(self.header)

                while True:
                    # 데이터 수집
                    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    cpu_usage = psutil.cpu_percent(interval=None)
                    memory_info = psutil.virtual_memory()
                    temp = self.get_cpu_temp()

                    # 리스트로 구성하여 쓰기
                    row = [now, cpu_usage, memory_info.percent, temp]
                    writer.writerow(row)
                    f.flush()  # 실시간 저장을 위해 버퍼 비우기

                    time.sleep(interval)

        except KeyboardInterrupt:
            print("\nMonitoring stopped by user.")
        except Exception as e:
            print(f"\nAn error occurred: {e}")
        finally:
            print("Data has been safely saved.")

if __name__ == "__main__":
    # 요청하신 경로 설정
    SAVE_PATH = '/home/rnd/HJ/AutoTeachingProject/R-Pi_status.csv'
    
    # 모니터 객체 생성 및 실행 (1초 간격)
    monitor = RaspberryPiMonitor(SAVE_PATH)
    monitor.run(interval=1)