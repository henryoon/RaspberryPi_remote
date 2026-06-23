import time
import json
import threading
import socket
import zmq

class RobotControllerBridge:
    """외부 로봇 제어기(Master)와 연결하여 데이터를 전송하는 TCP 클라이언트 클래스"""
    def __init__(self, robot_ip: str, robot_port: int):
        self.robot_ip = robot_ip
        self.robot_port = robot_port
        self.sock = None
        self.connected = False
        self.lock = threading.Lock()
        self.connect_to_robot()

    def connect_to_robot(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # Nagle 알고리즘 비활성화: 실시간 제어 데이터 지연 최소화
            self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1) 
            self.sock.settimeout(2.0)
            self.sock.connect((self.robot_ip, self.robot_port))
            self.connected = True
            print(f"✅ 외부 로봇 제어기 연결 성공 ({self.robot_ip}:{self.robot_port})")
        except Exception as e:
            self.connected = False
            print(f"⚠️ 외부 로봇 제어기 연결 대기 중... ({e})")

    def send_data(self, topic: str, data_dict: dict):
        """데이터를 JSON 포맷으로 패키징하여 로봇 제어기로 전송"""
        if not self.connected:
            # 연결이 끊어진 경우 재연결 시도 (1초 간격)
            time.sleep(1.0)
            self.connect_to_robot()
            return

        # 통신 규약: {"topic": "apriltag", "data": {...}} 형태로 래핑
        payload = json.dumps({"topic": topic, "data": data_dict}) + "\n"
        
        with self.lock:
            try:
                self.sock.sendall(payload.encode('utf-8'))
            except Exception as e:
                print(f"❌ 데이터 전송 실패: {e}")
                self.connected = False
                self.sock.close()

    def close(self):
        if self.connected and self.sock:
            self.sock.close()
            self.connected = False


class VisionDataReceiver:
    """비전 노드에서 5557, 5558, 5559 포트 데이터를 수신하여 로봇으로 릴레이하는 클래스"""
    def __init__(self, vision_ip="localhost", robot_ip="192.168.1.100", robot_port=9000):
        self.running = True
        self.context = zmq.Context()
        
        # 외부 로봇 통신 브리지 초기화
        self.robot_bridge = RobotControllerBridge(robot_ip, robot_port)

        # --- ZMQ 수신 소켓 설정 ---
        self.sub_tag = self._create_subscriber(vision_ip, 5557)
        self.sub_barcode = self._create_subscriber(vision_ip, 5558)
        self.sub_yolo = self._create_subscriber(vision_ip, 5559)

        # 다중 소켓 비동기 수신을 위한 Poller 등록
        self.poller = zmq.Poller()
        self.poller.register(self.sub_tag, zmq.POLLIN)
        self.poller.register(self.sub_barcode, zmq.POLLIN)
        self.poller.register(self.sub_yolo, zmq.POLLIN)

        print("📡 비전 데이터 수신 및 브리지 노드 초기화 완료")

    def _create_subscriber(self, ip: str, port: int):
        sock = self.context.socket(zmq.SUB)
        sock.set_hwm(10)
        sock.connect(f"tcp://{ip}:{port}")
        sock.setsockopt_string(zmq.SUBSCRIBE, "")
        return sock

    def process_loop(self):
        print("📥 다중 포트(5557, 5558, 5559) 데이터 수신 및 릴레이 시작...")
        
        while self.running:
            try:
                # 이벤트가 발생한 소켓들을 가져옴 (Timeout 100ms)
                socks = dict(self.poller.poll(100))

                # 1. AprilTag 데이터 수신 처리
                if self.sub_tag in socks and socks[self.sub_tag] == zmq.POLLIN:
                    msg = self.sub_tag.recv_string()
                    tag_data = json.loads(msg)
                    self._handle_tag_data(tag_data)

                # 2. Barcode 데이터 수신 처리
                if self.sub_barcode in socks and socks[self.sub_barcode] == zmq.POLLIN:
                    msg = self.sub_barcode.recv_string()
                    barcode_data = json.loads(msg)
                    self._handle_barcode_data(barcode_data)

                # 3. YOLO 데이터 수신 처리
                if self.sub_yolo in socks and socks[self.sub_yolo] == zmq.POLLIN:
                    msg = self.sub_yolo.recv_string()
                    yolo_data = json.loads(msg)
                    self._handle_yolo_data(yolo_data)

            except Exception as e:
                print(f"⚠️ 수신 루프 에러: {e}")

    # ==========================================
    # 데이터 파싱 및 로봇 제어기 송신 로직
    # ==========================================
    def _handle_tag_data(self, data: dict):
        """AprilTag 자세 정보 처리 (로봇 엔드이펙터 타겟 위치 생성용)"""
        # print(f"[Tag] {data}")
        self.robot_bridge.send_data("apriltag", data)

    def _handle_barcode_data(self, data: dict):
        """바코드 정보 처리 (물류 정보 매칭 및 검수용)"""
        # print(f"[Barcode] {data}")
        self.robot_bridge.send_data("barcode", data)

    def _handle_yolo_data(self, data: dict):
        """YOLO ROI 상태 처리 (파지 가능 영역 진입 판단용)"""
        # print(f"[YOLO] {data}")
        self.robot_bridge.send_data("yolo", data)

    def release(self):
        self.running = False
        self.robot_bridge.close()
        self.sub_tag.close()
        self.sub_barcode.close()
        self.sub_yolo.close()
        self.context.term()


if __name__ == "__main__":
    # 비전 시스템이 돌아가는 PC의 IP (동일 PC면 localhost)
    VISION_IP = "localhost" 
    # 실제 데이터를 수신할 외부 로봇 제어기(마스터)의 IP와 포트
    ROBOT_IP = "192.168.1.100" 
    ROBOT_PORT = 9000

    receiver_node = VisionDataReceiver(vision_ip=VISION_IP, robot_ip=ROBOT_IP, robot_port=ROBOT_PORT)
    
    try:
        receiver_node.process_loop()
    except KeyboardInterrupt:
        print("\n브리지 노드를 종료합니다.")
    finally:
        receiver_node.release()