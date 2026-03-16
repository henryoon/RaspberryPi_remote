import time
import psutil
from gpiozero import CPUTemperature
import pandas as pd
from datetime import datetime
import os

# --- 설정 ---
# 저장할 경로 (사용자 환경에 맞춤)
SAVE_DIR = '/home/rnd/HJ/status_log/'
# 파일명에 날짜와 시간을 붙여서 겹치지 않게 함
CSV_FILENAME = f"system_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
FULL_PATH = os.path.join(SAVE_DIR, CSV_FILENAME)

def main():
    # 1. 저장 디렉토리가 없으면 생성
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        print(f"디렉토리 생성됨: {SAVE_DIR}")

    # 2. 센서 준비
    cpu_temp = CPUTemperature()
    data = []

    print("-" * 40)
    print(f" 시스템 모니터링 시작")
    print(f" 저장 경로: {FULL_PATH}")
    print(" [Ctrl + C]를 누르면 종료하고 저장합니다.")
    print("-" * 40)
    print("Time\t\tCPU(%)\tMem(MB)\tTemp(C)")

    try:
        while True:
            # 현재 시간
            now = datetime.now()
            time_str = now.strftime('%H:%M:%S')

            # --- 데이터 측정 ---
            # interval=1로 설정하면 1초 동안 측정하여 평균을 냅니다.
            # 따라서 별도의 time.sleep(1)이 필요 없습니다.
            cpu_usage = psutil.cpu_percent(interval=1)
            
            # 메모리
            memory = psutil.virtual_memory()
            mem_used_mb = memory.used / (1024 * 1024) # MB 단위 변환
            
            # 온도
            temp = cpu_temp.temperature

            # 데이터 저장용 리스트에 추가
            data.append([time_str, cpu_usage, mem_used_mb, temp])

            # 터미널에 실시간 출력 (작동 확인용)
            print(f"{time_str}\t{cpu_usage:.1f}\t{mem_used_mb:.1f}\t{temp:.1f}")

    except KeyboardInterrupt:
        # Ctrl + C가 눌렸을 때 실행
        print("\n\n>>> 측정 종료 요청을 받았습니다.")
        
    finally:
        # CSV 파일 저장
        if data:
            print(">>> 데이터 저장 중...")
            df = pd.DataFrame(data, columns=['Time', 'CPU', 'Memory', 'Temp'])
            df.to_csv(FULL_PATH, index=False)
            print(f"완료! 파일이 생성되었습니다: {FULL_PATH}")
        else:
            print(">>> 저장할 데이터가 없습니다.")

if __name__ == "__main__":
    main()