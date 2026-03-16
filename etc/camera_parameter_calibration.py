import cv2
import numpy as np
import subprocess
import threading
import time
from flask import Flask, Response, render_template_string, request, jsonify

app = Flask(__name__)

# === [사용자 설정 영역: 실제 사용할 해상도로 변경 필수!] ===
RES_WIDTH = 640  # 실제 사용할 해상도 (가로)
RES_HEIGHT = 480  # 실제 사용할 해상도 (세로)
CHECKERBOARD = (9, 6) # 내부 코너 개수 (가로, 세로)
SQUARE_SIZE_MM = 24   # 사각형 한 변의 길이 (mm)

# === 전역 변수 ===
output_frame = None
lock = threading.Lock()

objpoints = [] 
imgpoints = [] 
capture_cmd = False
captured_count = 0
last_frame_corners = False
latest_gray_shape = None # 캘리브레이션 계산용 해상도 저장

# 3D 기준점 좌표 생성
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0 : CHECKERBOARD[0], 0 : CHECKERBOARD[1]].T.reshape(-1, 2)
objp = objp * SQUARE_SIZE_MM

# 카메라 실행 명령어
cmd = [
    "rpicam-vid",
    "-t", "0",
    "--width", str(RES_WIDTH),
    "--height", str(RES_HEIGHT),
    "--codec", "mjpeg",
    "--framerate", "30",
    "-o", "-"
]

def camera_thread_func():
    """ 카메라 영상을 읽고 처리하는 백그라운드 스레드 """
    global output_frame, capture_cmd, captured_count, last_frame_corners, latest_gray_shape, objpoints, imgpoints

    print(f"📷 카메라 스레드 시작 (해상도: {RES_WIDTH}x{RES_HEIGHT})")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
    buffer = b""

    while True:
        data = process.stdout.read(8192) # 버퍼 크기 약간 증가
        if not data: break
        buffer += data
        
        a = buffer.find(b'\xff\xd8')
        b = buffer.find(b'\xff\xd9')
        
        if a != -1 and b != -1:
            jpg_data = buffer[a:b+2]
            buffer = buffer[b+2:]
            
            frame = cv2.imdecode(np.frombuffer(jpg_data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None: continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            latest_gray_shape = gray.shape[::-1] 

            # 체커보드 찾기
            ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
            last_frame_corners = ret

            if ret:
                cv2.drawChessboardCorners(frame, CHECKERBOARD, corners, ret)
                
                # 캡처 로직 (스레드 내부에서 처리하여 안전함)
                if capture_cmd:
                    objpoints.append(objp)
                    corners2 = cv2.cornerSubPix(
                        gray, corners, (11, 11), (-1, -1),
                        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                    )
                    imgpoints.append(corners2)
                    captured_count += 1
                    print(f"✅ 캡처 완료! (Total: {captured_count})")
                    capture_cmd = False # 플래그 즉시 초기화

            # 상태 표시
            status_color = (0, 255, 0) if ret else (0, 0, 255)
            status_text = "Ready to Capture" if ret else "Searching..."
            cv2.putText(frame, status_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
            cv2.putText(frame, f"Count: {captured_count}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            # 웹 전송용 프레임 업데이트
            ret_enc, jpeg = cv2.imencode('.jpg', frame)
            if ret_enc:
                with lock:
                    output_frame = jpeg.tobytes()

def generate_frames():
    """ 웹 클라이언트에 최신 프레임만 전달 """
    global output_frame
    while True:
        with lock:
            if output_frame is None:
                continue
            data = output_frame
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + data + b'\r\n')
        time.sleep(0.05) # 전송 부하 조절

@app.route('/')
def index():
    return render_template_string("""
    <html>
        <head>
            <title>RPi Camera Calibration</title>
            <style>
                body { background-color: #222; color: white; text-align: center; font-family: sans-serif; }
                img { border: 2px solid #555; max-width: 100%; }
                button { padding: 15px 30px; font-size: 18px; margin: 10px; cursor: pointer; border-radius: 5px; border: none; }
                .btn-capture { background-color: #28a745; color: white; }
                .btn-calc { background-color: #007bff; color: white; }
                #result { margin-top: 20px; white-space: pre-wrap; text-align: left; display: inline-block; background: #333; padding: 20px; }
            </style>
            <script>
                function capture() {
                    fetch('/capture_trigger').then(res => res.json()).then(data => {
                        if(data.success) console.log("Capture triggered");
                        else alert("체커보드가 인식되지 않았습니다!");
                    });
                }
                function calibrate() {
                    document.getElementById('result').innerText = "계산 중입니다... (약 5-10초 소요)";
                    fetch('/calculate').then(res => res.json()).then(data => {
                        document.getElementById('result').innerText = data.result;
                    });
                }
            </script>
        </head>
        <body>
            <h1>📷 Calibration Tool</h1>
            <p>설정된 해상도: <b>{{ w }} x {{ h }}</b></p>
            <img src="/video_feed"><br>
            <button class="btn-capture" onclick="capture()">📸 캡처 (Capture)</button>
            <button class="btn-calc" onclick="calibrate()">🧮 계산 (Calibrate)</button>
            <br>
            <div id="result">결과 대기 중...</div>
        </body>
    </html>
    """, w=RES_WIDTH, h=RES_HEIGHT)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture_trigger')
def capture_trigger():
    global capture_cmd
    if last_frame_corners:
        capture_cmd = True
        return jsonify({"success": True})
    return jsonify({"success": False})

@app.route('/calculate')
def calculate():
    global objpoints, imgpoints, latest_gray_shape
    if len(objpoints) < 10: # 최소 장수 증가 권장
        return jsonify({"result": "❌ 데이터 부족: 최소 10장 이상 캡처해주세요."})
    
    try:
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, latest_gray_shape, None, None
        )
        
        if ret:
            result_text = f"""
✅ 캘리브레이션 완료! (RMS Error: {ret:.4f})

# Camera Matrix (Intrinsic)
fx, fy = {mtx[0,0]:.2f}, {mtx[1,1]:.2f}
cx, cy = {mtx[0,2]:.2f}, {mtx[1,2]:.2f}

# Distortion Coefficients (k1, k2, p1, p2, k3)
dist_coeffs = np.array({dist.tolist()})

# 사용 예시 코드:
# camera_matrix = np.array([[{mtx[0,0]:.2f}, 0, {mtx[0,2]:.2f}], [0, {mtx[1,1]:.2f}, {mtx[1,2]:.2f}], [0, 0, 1]])
# dist_coeffs = np.array({dist.tolist()})
"""
            return jsonify({"result": result_text})
        else:
            return jsonify({"result": "❌ 계산 실패"})
    except Exception as e:
        return jsonify({"result": f"❌ 오류: {str(e)}"})

if __name__ == '__main__':
    # 카메라 스레드 시작
    t = threading.Thread(target=camera_thread_func)
    t.daemon = True
    t.start()
    
    print("🚀 웹 서버 시작: http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)