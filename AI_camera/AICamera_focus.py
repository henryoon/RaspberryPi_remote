import time
import subprocess
import cv2
import numpy as np
from flask import Flask, Response

app = Flask(__name__)

# --post-process-file 옵션을 뺍니다. (순수 영상의 선명도만 보기 위해)
cmd = [
    "rpicam-vid",
    "-t", "0",
    "--width", "640",
    "--height", "480",
    "--codec", "mjpeg",
    "-o", "-"
]


def calculate_focus_score(image):
    """
    이미지의 라플라시안 변동성(Variance of Laplacian)을 계산합니다.
    이 값이 클수록 이미지가 선명(Edge가 뚜렷)하다는 뜻입니다.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return score

def generate_frames():
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
    buffer = b""

    while True:
        data = process.stdout.read(4096)
        if not data:
            break
        
        buffer += data
        a = buffer.find(b'\xff\xd8')
        b = buffer.find(b'\xff\xd9')
        
        if a != -1 and b != -1:
            jpg_data = buffer[a:b+2]
            buffer = buffer[b+2:]
            
            # 1. 바이트 데이터를 OpenCV 이미지로 디코딩
            frame = cv2.imdecode(np.frombuffer(jpg_data, dtype=np.uint8), cv2.IMREAD_COLOR)
            
            # 2. 초점 점수 계산
            score = calculate_focus_score(frame)
            
            # 3. 화면에 점수 표시 (점수가 높을수록 좋음)
            text = f"Focus Score: {int(score)}"
            
            # 시각적 효과: 점수가 100 미만이면 빨간색(흐림), 높으면 초록색
            color = (0, 0, 255) if score < 100 else (0, 255, 0)
            
            cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                        1.2, color, 3)
            
            # 중앙 십자선 그리기 (이 부분을 보며 초점을 맞추세요)
            h, w = frame.shape[:2]
            cv2.line(frame, (w//2 - 20, h//2), (w//2 + 20, h//2), (0, 255, 255), 2)
            cv2.line(frame, (w//2, h//2 - 20), (w//2, h//2 + 20), (0, 255, 255), 2)

            # 4. 다시 JPEG로 인코딩하여 웹으로 전송
            ret, jpeg = cv2.imencode('.jpg', frame)
            frame_bytes = jpeg.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return """
    <html>
    <body style="background:black; color:white; text-align:center;">
        <h1>📷 Focus Assistant</h1>
        <p>Rotate the lens until the 'Focus Score' is maximized.</p>
        <img src="/video_feed" style="border:2px solid yellow;">
    </body>
    </html>
    """

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("🚀 초점 도우미 실행 중... http://localhost:5000")
    print("렌즈를 돌려 점수를 가장 높게 만드세요.")
    app.run(host='0.0.0.0', port=5000, debug=False)