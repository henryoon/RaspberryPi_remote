import time
import json
import threading
import socket
import zmq

class BrooksControllerBridge:
    """Brooks Automation 로봇 제어기(Master)와 연결하여 ASCII 명령어를 전송하는 클라이언트 클래스"""
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
            # 실시간 로봇 모션 제어를 위한 통신 지연(Nagle 알고리즘) 비활성화
            self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1) 
            self.sock.settimeout(2.0)
            self.sock.connect((self.robot_ip, self.robot_port))
            self.connected = True
            print(f"✅ Brooks 로봇 제어기 연결 성공 ({self.robot_ip}:{self.robot_port})")
        except Exception as e:
            self.connected = False
            print(f"⚠️ Brooks 로봇 제어기 연결 대기 중... ({e})")

    def send_brooks_command(self, command_string: str):
        """Brooks 제어기가 해석할 수 있는 ASCII 명령어 전송 (\r\n 포함)"""
        if not self.connected:
            time.sleep(1.0)
            self.connect_to_robot()
            return
            
        # 산업용 제어기는 통상적으로 캐리지 리턴(\r)과 라인 피드(\n)로 명령의 종료를 인식합니다.
        payload = f"{command_string}\r\n"
        
        with self.lock:
            try:
                self.sock.sendall(payload.encode('ascii'))
                # 디버깅용 출력 (실제 운용 시 주석 처리 권장)
                # print(f"[TCP 송신] {command_string}")
            except Exception as e:
                print(f"❌ 데이터 전송 실패: {e}")
                self.connected = False
                self.sock.close()

    def close(self):
        if self.connected and self.sock:
            self.sock.close()
            self.connected = False


class VisionDataReceiver:
    """비전 노드의 ZMQ 포트 데이터를 수신하여 Brooks 제어용 명령어로 번역/릴레이하는 클래스"""
    def __init__(self, vision_ip="localhost", robot_ip="192.168.1.100", robot_port=10100):
        self.running = True
        self.context = zmq.Context()
        
        # Brooks 제어기 통신 브리지 초기화 (포트는 매뉴얼에 명시된 포트로 설정 필요)
        self.brooks_bridge = BrooksControllerBridge(robot_ip, robot_port)

        # --- ZMQ 수신 소켓 설정 ---
        self.sub_tag = self._create_subscriber(vision_ip, 5557)
        self.sub_barcode = self._create_subscriber(vision_ip, 5558)
        self.sub_yolo = self._create_subscriber(vision_ip, 5559)

        self.poller = zmq.Poller()
        self.poller.register(self.sub_tag, zmq.POLLIN)
        self.poller.register(self.sub_barcode, zmq.POLLIN)
        self.poller.register(self.sub_yolo, zmq.POLLIN)

        print("📡 비전 데이터 수신 및 Brooks 변환 브리지 노드 초기화 완료")

    def _create_subscriber(self, ip: str, port: int):
        sock = self.context.socket(zmq.SUB)
        sock.set_hwm(10)
        sock.connect(f"tcp://{ip}:{port}")
        sock.setsockopt_string(zmq.SUBSCRIBE, "")
        return sock

    def process_loop(self):
        print("📥 데이터 수신 및 Brooks 명령어 릴레이 시작...")
        
        while self.running:
            try:
                socks = dict(self.poller.poll(100))

                # 1. AprilTag
                if self.sub_tag in socks and socks[self.sub_tag] == zmq.POLLIN:
                    msg = self.sub_tag.recv_string()
                    tag_data = json.loads(msg)
                    self._handle_tag_data(tag_data)

                # 2. Barcode
                if self.sub_barcode in socks and socks[self.sub_barcode] == zmq.POLLIN:
                    msg = self.sub_barcode.recv_string()
                    barcode_data = json.loads(msg)
                    self._handle_barcode_data(barcode_data)

                # 3. YOLO
                if self.sub_yolo in socks and socks[self.sub_yolo] == zmq.POLLIN:
                    msg = self.sub_yolo.recv_string()
                    yolo_data = json.loads(msg)
                    self._handle_yolo_data(yolo_data)

            except Exception as e:
                print(f"⚠️ 수신 루프 에러: {e}")

    # ==========================================
    # 💡 데이터 파싱 및 Brooks 제어기 명령어 변환 로직
    # ==========================================
    def _handle_tag_data(self, data: dict):
        """AprilTag 좌표를 로봇의 Vision Offset 변수에 쓰는 명령어로 변환"""
        for tag in data.get("tags", []):
            x, y, z = tag["x"], tag["y"], tag["z"]
            # 예: 로봇 내부의 'VisionX', 'VisionY' 전역 변수를 갱신하거나 오프셋 명령어 전송
            # 실제 제어기 매뉴얼에 명시된 문법으로 변경해야 합니다.
            brooks_cmd = f"SetVisionOffset {x} {y} {z}" 
            self.brooks_bridge.send_brooks_command(brooks_cmd)

    def _handle_barcode_data(self, data: dict):
        """바코드 정보를 로봇 시스템의 문자열 변수로 전달"""
        barcodes = data.get("barcodes", [])
        if barcodes:
            # 첫 번째 바코드 데이터를 로봇의 'CurrentBarcode' 변수에 저장하는 명령어 예시
            current_code = barcodes[0]
            brooks_cmd = f'SetString CurrentBarcode "{current_code}"'
            self.brooks_bridge.send_brooks_command(brooks_cmd)

    def _handle_yolo_data(self, data: dict):
        """YOLO ROI 진입 여부에 따라 파지 시퀀스 실행 판단"""
        is_in_roi = data.get("is_in_roi", False)
        
        if is_in_roi:
            # 물체가 ROI에 진입하여 안착되었음을 알림
            self.brooks_bridge.send_brooks_command("SetVar IsTargetReady 1")
            # 또는 바로 동작 매크로 호출: self.brooks_bridge.send_brooks_command("Execute PickMicroplate")
        else:
            self.brooks_bridge.send_brooks_command("SetVar IsTargetReady 0")

    def release(self):
        self.running = False
        self.brooks_bridge.close()
        self.sub_tag.close()
        self.sub_barcode.close()
        self.sub_yolo.close()
        self.context.term()


if __name__ == "__main__":
    # 비전 시스템 PC의 IP (동일 PC면 localhost)
    VISION_IP = "localhost" 
    
    # 💡 Brooks 로봇 제어기의 실제 IP와 허용된 포트로 반드시 수정하십시오.
    # (일반적으로 10100, 10200 등의 특정 포트가 ASCII 명령어 수신용으로 예약되어 있습니다.)
    BROOKS_ROBOT_IP = "192.168.0.1" 
    BROOKS_ROBOT_PORT = 10100 

    bridge_node = VisionDataReceiver(vision_ip=VISION_IP, robot_ip=BROOKS_ROBOT_IP, robot_port=BROOKS_ROBOT_PORT)
    
    try:
        bridge_node.process_loop()
    except KeyboardInterrupt:
        print("\n브리지 노드를 종료합니다.")
    finally:
        bridge_node.release()