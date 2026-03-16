import cv2
import numpy as np
import subprocess
import time
from flask import Flask, Response, render_template_string, request, jsonify

app = Flask(__name__)

# === [사용자 설정 영역] ===
# 1. 체커보드의 가로, 세로 "내부 코너"의 개수 (타일 개수 아님!)
CHECKERBOARD = (9, 6)

# 2. 프린트된 체커보드 사각형 한 변의 실제 길이 (단위: mm)
# 자로 정확하게 재서 입력하세요.
SQUARE_SIZE_MM = 24 

# === 전역 변수 ===
objpoints = [] # 3D points in real world space
imgpoints = [] # 2D points in image plane
capture_flag = False
captured_count = 0
last_frame_corners = False # 현재 프레임에 코너가 잡혔는지 여부
latest_gray = None # 캘리브레이션 계산용 흑백 이미지 저장

# 3D 기준점 좌표 생성 (0,0,0), (1,0,0), (2,0,0) ...
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0 : CHECKERBOARD[0], 0 : CHECKERBOARD[1]].T.reshape(-1, 2)
objp = objp * SQUARE_SIZE_MM

# 카메라 실행 명령어 (640x480 권장)
cmd = [
    "rpicam-vid",
    "-t", "0",
    "--width", "640",
    "--height", "480",
    "--codec", "mjpeg",
    "-o", "-"
]

def generate_frames():
    global capture_flag, captured_count, last_frame_corners, latest_gray, objpoints, imgpoints
    
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
    buffer = b""

    print("📷 캘리브레이션 서버 시작...")

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
            
            # 이미지 디코딩
            frame = cv2.imdecode(np.frombuffer(jpg_data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None: continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            latest_gray = gray # 나중에 해상도 참조용

            # 체커보드 찾기
            ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
            last_frame_corners = ret

            # 화면에 그리기 (시각적 확인용)
            if ret:
                cv2.drawChessboardCorners(frame, CHECKERBOARD, corners, ret)
                cv2.putText(frame, "CORNERS FOUND!", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Searching...", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # [캡처 명령이 들어왔고] + [코너가 보일 때] 데이터 저장
            if capture_flag and ret:
                objpoints.append(objp)
                # 정밀도 향상
                corners2 = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1),
                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                )
                imgpoints.append(corners2)
                captured_count += 1
                print(f"✅ 캡처 완료! (현재 {captured_count}장)")
                capture_flag = False # 플래그 초기화

            # 정보 표시
            cv2.putText(frame, f"Captured: {captured_count}", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # 전송
            ret, jpeg = cv2.imencode('.jpg', frame)
            frame_bytes = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template_string("""
    <html>
        <head>
            <title>RPi AI Camera Calibration</title>
            <style>
                body { background-color: #222; color: white; text-align: center; font-family: sans-serif; }
                img { border: 2px solid #555; margin-bottom: 10px; }
                button { padding: 15px 30px; font-size: 18px; margin: 10px; cursor: pointer; border-radius: 5px; border: none; }
                .btn-capture { background-color: #28a745; color: white; }
                .btn-calc { background-color: #007bff; color: white; }
                #result { margin-top: 20px; white-space: pre-wrap; text-align: left; display: inline-block; background: #333; padding: 20px; }
            </style>
            <script>
                function capture() {
                    fetch('/capture_trigger').then(response => response.json()).then(data => {
                        if(data.success) alert("캡처되었습니다!");
                        else alert("체커보드가 인식되지 않았습니다. 화면을 확인하세요.");
                    });
                }
                function calibrate() {
                    document.getElementById('result').innerText = "계산 중입니다... 잠시만 기다려주세요.";
                    fetch('/calculate').then(response => response.json()).then(data => {
                        document.getElementById('result').innerText = data.result;
                    });
                }
            </script>
        </head>
        <body>
            <h1>📷 Camera Calibration Tool</h1>
            <img src="/video_feed" width="640" height="480"><br>
            <p>다양한 각도와 거리에서 체커보드를 비추고 캡처하세요 (최소 15장 권장)</p>
            <button class="btn-capture" onclick="capture()">📸 찰칵 (Capture)</button>
            <button class="btn-calc" onclick="calibrate()">🧮 계산하기 (Calibrate)</button>
            <br>
            <div id="result">결과가 여기에 표시됩니다.</div>
        </body>
    </html>
    """)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture_trigger')
def capture_trigger():
    global capture_flag, last_frame_corners
    if last_frame_corners:
        capture_flag = True
        return jsonify({"success": True})
    else:
        return jsonify({"success": False})

@app.route('/calculate')
def calculate():
    global objpoints, imgpoints, latest_gray
    if len(objpoints) < 5:
        return jsonify({"result": "❌ 데이터가 너무 적습니다. 최소 5장 이상 찍어주세요."})
    
    try:
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, latest_gray.shape[::-1], None, None
        )
        
        if ret:
            result_text = f"""
✅ Calibration 성공!

# Camera Parameters (복사해서 사용하세요)
fx, fy = {mtx[0,0]:.2f}, {mtx[1,1]:.2f}
cx, cy = {mtx[0,2]:.2f}, {mtx[1,2]:.2f}

# Distortion Coefficients
dist_coeffs = np.array([{dist[0][0]:.5f}, {dist[0][1]:.5f}, {dist[0][2]:.5f}, {dist[0][3]:.5f}, {dist[0][4]:.5f}])
"""
            return jsonify({"result": result_text})
        else:
            return jsonify({"result": "❌ 계산 실패 (ret=False)"})
    except Exception as e:
        return jsonify({"result": f"❌ 오류 발생: {str(e)}"})

if __name__ == '__main__':
    print("🚀 웹 서버 시작: http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)