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

type RobotVisionSystem struct {
	mu          sync.Mutex
	outputFrame gocv.Mat
	model       gocv.Net
	targetROI   image.Rectangle
	running     bool
}

func NewRobotVisionSystem(modelPath string) *RobotVisionSystem {
	// YOLO (ONNX) 모델 로드
	net := gocv.ReadNetFromONNX(modelPath)
	if net.Empty() {
		log.Fatalf("모델 로딩 실패: %s", modelPath)
	}
	net.SetPreferableBackend(gocv.NetBackendDefault)
	net.SetPreferableTarget(gocv.NetTargetCPU)

	return &RobotVisionSystem{
		model:       net,
		targetROI:   image.Rect(206, 212, 434, 268),
		running:     true,
		outputFrame: gocv.NewMat(),
	}
}

func (rv *RobotVisionSystem) ProcessLoop() {
	pipeline := "libcamerasrc ! videoconvert ! videoscale ! video/x-raw, width=640, height=480, format=BGR ! appsink drop=true"

	cam, err := gocv.OpenVideoCaptureWithAPI(pipeline, gocv.VideoCaptureGstreamer)
	if err != nil {
		log.Fatalf("카메라 열기 실패 (파이프라인 확인 필요): %v", err)
	}
	defer cam.Close()

	img := gocv.NewMat()
	defer img.Close()

	frameCount := 0
	skipFrames := 3
	isInROI := false // 추론 결과 상태 유지

	fmt.Println("📷 Vision System 가동 중...")

	for rv.running {
		if ok := cam.Read(&img); !ok || img.Empty() {
			continue
		}

		if frameCount%skipFrames == 0 {
			blob := gocv.BlobFromImage(img, 1.0/255.0, image.Pt(320, 320), gocv.NewScalar(0, 0, 0, 0), true, false)
			rv.model.SetInput(blob, "")
			detections := rv.model.Forward("")

			isInROI = rv.processDetections(&img, detections)

			blob.Close()
			detections.Close()
		}

		gocv.Rectangle(&img, rv.targetROI, color.RGBA{255, 255, 255, 0}, 1)
		gocv.PutText(&img, "Target ROI", image.Pt(rv.targetROI.Min.X, rv.targetROI.Min.Y-5), gocv.FontHersheySimplex, 0.5, color.RGBA{255, 255, 255, 0}, 1)

		statusText := "STATUS: NONE"
		statusColor := color.RGBA{255, 0, 0, 0}
		if isInROI {
			statusText = "STATUS: DETECTED"
			statusColor = color.RGBA{0, 255, 0, 0}
			fmt.Println("LOG: [Object Detected in ROI]")
		}

		gocv.PutText(&img, statusText, image.Pt(10, 25), gocv.FontHersheySimplex, 0.6, statusColor, 2)

		rv.mu.Lock()
		img.CopyTo(&rv.outputFrame)
		rv.mu.Unlock()

		frameCount++
		time.Sleep(10 * time.Millisecond)
	}
}

func (rv *RobotVisionSystem) processDetections(frame *gocv.Mat, detections gocv.Mat) bool {
	res := detections.Reshape(1, detections.Size()[1])
	defer res.Close()

	gocv.Transpose(res, &res)

	detectedInROI := false

	for i := 0; i < res.Rows(); i++ {
		confidence := res.GetFloatAt(i, 4)

		if confidence > 0.5 {
			centerX := float64(res.GetFloatAt(i, 0)) * (640.0 / 320.0)
			centerY := float64(res.GetFloatAt(i, 1)) * (480.0 / 320.0)

			currPoint := image.Pt(int(centerX), int(centerY))

			if currPoint.In(rv.targetROI) {
				detectedInROI = true
				gocv.Circle(frame, currPoint, 4, color.RGBA{0, 255, 0, 0}, -1)
				break
			}
		}
	}
	return detectedInROI
}

func (rv *RobotVisionSystem) StreamHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "multipart/x-mixed-replace; boundary=frame")

	for rv.running {
		rv.mu.Lock()
		if rv.outputFrame.Empty() {
			rv.mu.Unlock()
			continue
		}
		buf, err := gocv.IMEncode(".jpg", rv.outputFrame)
		rv.mu.Unlock()

		if err != nil {
			continue
		}

		w.Write([]byte("--frame\r\n"))
		w.Write([]byte("Content-Type: image/jpeg\r\n\r\n"))
		w.Write(buf.GetBytes())
		w.Write([]byte("\r\n"))
		buf.Close()

		time.Sleep(40 * time.Millisecond)
	}
}

func main() {
	modelPath := "/home/rnd/HJ/AutoTeachingProject/best_yolov26n.onnx"
	vision := NewRobotVisionSystem(modelPath)

	go vision.ProcessLoop()

	http.HandleFunc("/video_feed", vision.StreamHandler)
	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/html")
		fmt.Fprintf(w, "<html><head><title>Microplate Detection</title></head>"+
			"<body><h1>Microplate Detection (Go + ONNX)</h1>"+
			"<img src='/video_feed' width='640'></body></html>")
	})

	fmt.Println("🚀 Go Vision Server starting on http://0.0.0.0:5000")
	if err := http.ListenAndServe(":5000", nil); err != nil {
		log.Fatalf("서버 실행 에러: %v", err)
	}
}

// go 언어로 작성된 위 스크립트에 새로운 기능을 추가하려고 해. 먼저 raspberrypi camera module 3의 초점을 고정시켜 일정한 거리에 있는 객체를 항상 뚜렷하게 볼 수 있는 기능을 추가해줘.