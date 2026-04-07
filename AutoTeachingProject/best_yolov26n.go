package main

import (
	"fmt"
	"image"
	"image/color"
	"log"
	"net/http"
	"sync"
	"time"

	"gocv.io/x/gocv"
)

type SimpleVisionSystem struct {
	mu          sync.RWMutex
	outputFrame gocv.Mat
	model       gocv.Net
}

func NewSimpleSystem(modelPath string) *SimpleVisionSystem {
	// 1. ONNX 모델 로드 확인 (추론은 하지 않고 로드 여부만 체크)
	net := gocv.ReadNetFromONNX(modelPath)
	if net.Empty() {
		log.Fatalf("❌ 모델 파일을 찾을 수 없거나 로드에 실패했습니다: %s", modelPath)
	}
	fmt.Println("✅ 모델 로드 성공:", modelPath)

	return &SimpleVisionSystem{
		model:       net,
		outputFrame: gocv.NewMat(),
	}
}

func (s *SimpleVisionSystem) RunCamera() {
	// Raspberry Pi 5 libcamera 파이프라인
	pipeline := "libcamerasrc ! videoconvert ! videoscale ! video/x-raw, width=640, height=480, format=BGR ! appsink drop=true"
	
	cam, err := gocv.OpenVideoCaptureWithAPI(pipeline, gocv.VideoCaptureGstreamer)
	if err != nil {
		log.Fatalf("❌ 카메라 연결 실패: %v", err)
	}
	defer cam.Close()

	img := gocv.NewMat()
	defer img.Close()

	for {
		if ok := cam.Read(&img); !ok || img.Empty() {
			continue
		}

		// 화면에 가이드 텍스트 표시
		gocv.PutText(&img, "Camera Streaming...", image.Pt(20, 40), 
			gocv.FontHersheySimplex, 0.8, color.RGBA{0, 255, 0, 0}, 2)

		// 웹 스트리밍용 프레임 업데이트
		s.mu.Lock()
		img.CopyTo(&s.outputFrame)
		s.mu.Unlock()

		time.Sleep(10 * time.Millisecond)
	}
}

func (s *SimpleVisionSystem) StreamHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "multipart/x-mixed-replace; boundary=frame")
	for {
		s.mu.RLock()
		if s.outputFrame.Empty() {
			s.mu.RUnlock()
			time.Sleep(10 * time.Millisecond)
			continue
		}
		// JPEG 인코딩
		buf, _ := gocv.IMEncode(".jpg", s.outputFrame)
		s.mu.RUnlock()

		w.Write([]byte("--frame\r\n"))
		w.Write([]byte("Content-Type: image/jpeg\r\n\r\n"))
		w.Write(buf.GetBytes())
		w.Write([]byte("\r\n"))
		buf.Close()

		time.Sleep(40 * time.Millisecond) // 약 25 FPS
	}
}

func main() {
	modelPath := "/home/rnd/HJ/AutoTeachingProject/best_yolov26n.onnx"
	system := NewSimpleSystem(modelPath)

	// 카메라 프로세스 시작
	go system.RunCamera()

	// 웹 서버 경로 설정
	http.HandleFunc("/video_feed", system.StreamHandler)
	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
    w.Header().Set("Content-Type", "text/html")
    fmt.Fprintf(w, `
    <html>
    <head>
        <title>Robot Vision Monitor</title>
        <style>
            body {
                margin: 0;
                padding: 0;
                background-color: #1a1a1a;
                color: white;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                min-height: 100vh;
                font-family: sans-serif;
            }
            .container {
                width: 90%%;
                max-width: 1280px; /* 최대 폭 제한 */
                text-align: center;
            }
            img {
                width: 100%%;      /* 컨테이너 폭에 맞춤 */
                height: auto;     /* 비율 유지 */
                border: 4px solid #333;
                border-radius: 8px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.5);
            }
            h1 {
                margin-bottom: 20px;
                font-weight: 300;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Raspberry Pi 5 Vision Monitor</h1>
            <img src="/video_feed" alt="Camera Stream">
        </div>
    </body>
    </html>`)
})

	fmt.Println("🚀 서버 실행 중: http://192.168.137.2:5000")
	if err := http.ListenAndServe(":5000", nil); err != nil {
		log.Fatal(err)
	}
}