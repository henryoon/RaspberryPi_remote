import time
import os
import cv2
import numpy as np
from ultralytics import YOLO

# 라즈베리파이 공식 카메라 라이브러리 import
try:
    from picamera2 import Picamera2
except ImportError:
    print("❌ 오류: 'picamera2' 모듈을 찾을 수 없습니다.")
    print("   해결법: 가상환경을 만들 때 '--system-site-packages' 옵션을 사용했는지 확인하세요.")
    exit()

def run_yolo_picamera2(model_path, confidence_threshold=0.5):
    # 1. 모델 경로 확인
    if not os.path.exists(model_path):
        print(f"오류: 파일 경로를 찾을 수 없습니다 -> {model_path}")
        return

    # 2. 모델 로드
    try:
        model = YOLO(model_path)
        print(f"✅ 모델 로드 성공: {model_path}")
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        return

    # 3. Picamera2 초기화
    print("📷 라즈베리파이 AI 카메라(Picamera2) 초기화 중...")
    try:
        picam2 = Picamera2()
        
        # 카메라 설정: 해상도 640x480, 포맷 BGR888
        config = picam2.create_video_configuration(
            main={"size": (640, 480), "format": "BGR888"}
        )
        picam2.configure(config)
        
        # 카메라 시작
        picam2.start()
        print("✅ 카메라 시작 완료!")
        
    except Exception as e:
        print(f"❌ 카메라 초기화 실패: {e}")
        print("   팁: 'rpicam-hello'가 작동하는지 먼저 확인하세요.")
        return

    print("🚀 실시간 탐지를 시작합니다.")
    print("🛑 종료하려면 'Ctrl + C'를 누르세요.\n")

    # FPS 계산을 위한 변수 초기화
    prev_time = 0
    
    try:
        while True:
            # 4. 프레임 캡처
            frame = picam2.capture_array()
            
            if frame is None:
                continue

            # FPS 계산 로직
            curr_time = time.time()
            fps = 0
            if prev_time != 0:
                fps = 1 / (curr_time - prev_time)
            prev_time = curr_time

            # 5. YOLO 추론
            results = model.predict(source=frame, conf=confidence_threshold, save=False, verbose=False)
            detections = results[0].boxes
            
            # 6. 결과 출력 (수정됨)
            if len(detections) > 0:
                # 감지된 객체가 있을 때만 상세 정보 출력
                detected_info = []
                for box in detections:
                    cls_id = int(box.cls[0])
                    class_name = model.names[cls_id]
                    conf = float(box.conf[0])
                    detected_info.append(f"{class_name}({conf:.2f})")
                
                # 감지되었을 때는 줄바꿈(\n)을 하여 로그를 남김
                # f-string 안의 :.1f는 소수점 첫째 자리까지 표시하라는 의미
                print(f"\n🟢 [FPS: {fps:.1f}] 감지됨: {', '.join(detected_info)}")
            
            else:
                # 감지되지 않았을 때는 같은 줄에 덮어쓰기 (end='\r')
                # 터미널이 지저분해지는 것을 방지함
                print(f"👀 모니터링 중... (FPS: {fps:.1f})   ", end='\r')

    except KeyboardInterrupt:
        print("\n🛑 사용자에 의해 종료되었습니다.")

    except Exception as e:
        print(f"\n❌ 실행 중 오류 발생: {e}")

    finally:
        # 카메라 리소스 해제
        if 'picam2' in locals():
            picam2.stop()
            picam2.close()
        print("리소스 해제 완료.")

if __name__ == "__main__":
    user_model = r'/home/rnd/HJ/yolo_model/flask.pt'
    run_yolo_picamera2(model_path=user_model, confidence_threshold=0.5)