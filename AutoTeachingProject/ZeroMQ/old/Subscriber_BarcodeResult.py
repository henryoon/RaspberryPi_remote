import zmq
import json

context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.connect("tcp://localhost:5558") # 5558 포트 접속
socket.setsockopt_string(zmq.SUBSCRIBE, "")

print("Waiting for Barcode data...")

while True:
    json_data = socket.recv_string()
    data = json.loads(json_data)
    
    for code in data["barcodes"]:
        print(f"[{data['timestamp']}] Scanned Barcode: {code}")