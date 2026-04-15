import zmq
import time

context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind("tcp://*:5555")

while True:
    message = "object_detected"
    socket.send_string(message)
    print(f"Sent: {message}")
    time.sleep(1)
    # message = True
    # socket.send_string(str(message))
    # print(f"Sent: {message}")
    # time.sleep(1)