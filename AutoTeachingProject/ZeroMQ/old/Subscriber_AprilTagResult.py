import zmq
import json

context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.connect("tcp://localhost:5557") # 5557 포트 접속
socket.setsockopt_string(zmq.SUBSCRIBE, "")

print("Waiting for AprilTag coordinates...")

while True:
    json_data = socket.recv_string()
    data = json.loads(json_data)
    
    for tag in data["tags"]:
        print(f"Target Acquired -> ID: {tag['id']}, X: {tag['x']}mm, Y: {tag['y']}mm, Z: {tag['z']}mm")