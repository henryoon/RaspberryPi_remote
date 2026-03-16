import time
import subprocess
import threading
from flask import Flask, Response

app = Flask(__name__)

# 전역 변수로 최신 프레임 저장
output_frame = None
lock = threading.Lock()

# 라즈베리파이 AI 카메라 실행 명령어
cmd = [
    "rpicam-vid",
    "-t", "0",
    "--width", "640",
    "--height", "480",
    "--codec", "mjpeg",
    "--post-process-file", "/home/rnd/HJ/AI_camera/flask_config.json",
    # "--post-process-file", "/usr/share/rpi-camera-assets/imx500_mobilenet_ssd.json",
    "-o", "-"
]

def camera_stream():
    """
    백그라운드에서 계속 실행되면서 최신 프레임을 output_frame에 업데이트하는 함수
    """
    global output_frame
    
    # 프로세스 시작 (앱 켜질 때 1회만 실행됨)
    print("📷 카메라 프로세스 시작 및 AI 모델 로딩 중... (잠시만 기다려주세요)")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
    
    buffer = b""
    while True:
        data = process.stdout.read(4096)
        if not data:
            break
        
        buffer += data
        
        # MJPEG 프레임 추출 로직
        a = buffer.find(b'\xff\xd8')
        b = buffer.find(b'\xff\xd9')
        
        if a != -1 and b != -1:
            jpg = buffer[a:b+2]
            buffer = buffer[b+2:]
            
            # 스레드 안전하게 전역 변수 업데이트
            with lock:
                output_frame = jpg

def generate_frames():
    """
    클라이언트에게 저장된 최신 프레임을 전달하는 제너레이터
    """
    global output_frame
    
    while True:
        # 최신 프레임이 준비될 때까지 잠시 대기
        with lock:
            if output_frame is None:
                continue
            
            # 현재 저장된 최신 프레임 복사
            encodedImage = output_frame

        # 클라이언트로 전송
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + encodedImage + b'\r\n')
        
        # 너무 빠른 루프 방지 (약 30fps 수준 조절)
        time.sleep(0.03)

@app.route('/')
def index():
    return """
    <html>
        <head>
            <title>Raspberry Pi AI Camera Stream</title>
            <style>
                body { background-color: #111; color: white; text-align: center; font-family: sans-serif; }
                h1 { margin-top: 20px; }
                img { border: 2px solid #00ff00; border-radius: 10px; box-shadow: 0 0 20px #00ff00; }
                p { color: #aaa; }
            </style>
        </head>
        <body>
            <h1>🚀 AI Camera Live Stream</h1>
            <p>Background Thread Mode - Instant Loading</p>
            <img src="/video_feed" width="640" height="480">
        </body>
    </html>
    """

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # 웹 서버 시작 전에 카메라 스레드를 먼저 실행
    t = threading.Thread(target=camera_stream)
    t.daemon = True # 메인 프로그램 종료 시 스레드도 같이 종료되도록 설정
    t.start()
    
    print(f"🚀 웹 서버 시작: http://0.0.0.0:5000")
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\n🛑 종료합니다.")