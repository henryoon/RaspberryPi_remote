import zmq
import json
from datetime import datetime

def main():
    context = zmq.Context()
    poller = zmq.Poller()

    # 1. 수신 소켓 설정 및 구독 (AprilTag: 5557)
    tag_sub = context.socket(zmq.SUB)
    tag_sub.connect("tcp://localhost:5557")
    tag_sub.setsockopt_string(zmq.SUBSCRIBE, "")
    poller.register(tag_sub, zmq.POLLIN)

    # 2. 수신 소켓 설정 및 구독 (Barcode: 5558)
    bc_sub = context.socket(zmq.SUB)
    bc_sub.connect("tcp://localhost:5558")
    bc_sub.setsockopt_string(zmq.SUBSCRIBE, "")
    poller.register(bc_sub, zmq.POLLIN)

    # 3. 수신 소켓 설정 및 구독 (YOLO: 5559)
    yolo_sub = context.socket(zmq.SUB)
    yolo_sub.connect("tcp://localhost:5559")
    yolo_sub.setsockopt_string(zmq.SUBSCRIBE, "")
    poller.register(yolo_sub, zmq.POLLIN)

    print("📡 ZMQ 실시간 다중 포트 모니터링 시작 (Ports: 5557, 5558, 5559)")
    print("종료하려면 Ctrl+C를 누르세요.\n" + "="*60)

    try:
        while True:
            # 10ms 대기하며 데이터가 들어온 소켓 확인
            socks = dict(poller.poll(10))
            now = datetime.now().strftime('%H:%M:%S.%f')[:-3]

            # AprilTag 데이터 출력
            if tag_sub in socks:
                msg = tag_sub.recv_string()
                data = json.loads(msg)
                print(f"[{now}] ⚫ [AprilTag 5557] {data}")

            # Barcode 데이터 출력
            if bc_sub in socks:
                msg = bc_sub.recv_string()
                data = json.loads(msg)
                print(f"[{now}] ⚪ [Barcode  5558] {data}")

            # YOLO 데이터 출력
            if yolo_sub in socks:
                msg = yolo_sub.recv_string()
                data = json.loads(msg)
                print(f"[{now}] 🟢 [YOLO     5559] {data}")

    except KeyboardInterrupt:
        print("\n모니터링을 종료합니다.")
    finally:
        tag_sub.close()
        bc_sub.close()
        yolo_sub.close()
        context.term()

if __name__ == "__main__":
    main()