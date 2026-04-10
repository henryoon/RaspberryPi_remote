package main

import (
	"bufio"
	"bytes"
	"fmt"
	"image"
	"image/color"
	"log"
	"net/http"
	"os"
	"os/exec"
	"sync"
	"time"

	"gocv.io/x/gocv"
)

// ==========================================
// 1. PiCamera Struct
// ==========================================
type PiCamera struct {
	cmd     *exec.Cmd
	scanner *bufio.Scanner
}

func NewPiCamera() (*PiCamera, error) {
	log.Println("📷 Camera Module 3 네이티브 초기화...")

	cmd := exec.Command("rpicam-vid",
		"-t", "0",
		"-n",
		"--codec", "mjpeg",
		"--width", "640",
		"--height", "480",
		"--framerate", "30",
		"--autofocus-mode", "manual",
		"--lens-position", "5.5",
		"-o", "-",
	)
	cmd.Stderr = os.Stderr

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, err
	}

	if err := cmd.Start(); err != nil {
		return nil, err
	}

	scanner := bufio.NewScanner(stdout)
	buf := make([]byte, 2*1024*1024)
	scanner.Buffer(buf, len(buf))

	splitJPEG := func(data []byte, atEOF bool) (advance int, token []byte, err error) {
		start := bytes.Index(data, []byte{0xff, 0xd8})
		if start == -1 {
			return 0, nil, nil
		}
		end := bytes.Index(data[start:], []byte{0xff, 0xd9})
		if end == -1 {
			return 0, nil, nil
		}
		return start + end + 2, data[start : start+end+2], nil
	}
	scanner.Split(splitJPEG)
	return &PiCamera{cmd: cmd, scanner: scanner}, nil
}

func (pc *PiCamera) Read(img *gocv.Mat) bool {
	if pc.scanner.Scan() {
		frameBytes := pc.scanner.Bytes()
		parsedImg, err := gocv.IMDecode(frameBytes, gocv.IMReadColor)
		if err == nil && !parsedImg.Empty() {
			parsedImg.CopyTo(img)
			parsedImg.Close()
			return true
		}
	}
	return false
}

func (pc *PiCamera) Close() {
	pc.cmd.Process.Kill()
}

// ==========================================
// 2. WebStreamer Struct
// ==========================================
type WebStreamer struct {
	frame []byte
	mutex sync.RWMutex
}

func (ws *WebStreamer) UpdateFrame(newFrame []byte) {
	ws.mutex.Lock()
	defer ws.mutex.Unlock()
	ws.frame = newFrame
}

func (ws *WebStreamer) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "multipart/x-mixed-replace; boundary=frame")
	ctx := r.Context()

	for {
		select {
		case <-ctx.Done():
			log.Println("⚠️ 브라우저 연결 종료 감지 (자원 회수 완료)")
			return
		default:
			ws.mutex.RLock()
			frameData := ws.frame
			ws.mutex.RUnlock()

			if len(frameData) > 0 {
				if _, err := w.Write([]byte("--frame\r\n")); err != nil { return }
				if _, err := w.Write([]byte("Content-Type: image/jpeg\r\n\r\n")); err != nil { return }
				if _, err := w.Write(frameData); err != nil { return }
				if _, err := w.Write([]byte("\r\n")); err != nil { return }
			}
			time.Sleep(30 * time.Millisecond)
		}
	}
}

// ==========================================
// 3. YOLO Tensor Parser (클래스 ID 추출 추가)
// ==========================================
func parseYOLOOutput(prob *gocv.Mat, confThreshold float32, imgW, imgH int) ([]image.Rectangle, []float32, []int) {
	var boxes []image.Rectangle
	var confidences []float32
	var classIDs []int // 💡 [추가] 각 박스의 클래스 ID를 저장할 배열

	dims := prob.Size()
	if len(dims) < 3 {
		return boxes, confidences, classIDs
	}
	rows, cols := dims[1], dims[2]

	xFactor, yFactor := float32(imgW)/640.0, float32(imgH)/640.0
	data, err := prob.DataPtrFloat32()
	if err != nil || len(data) < rows*cols {
		return boxes, confidences, classIDs
	}

	for c := 0; c < cols; c++ {
		var maxClassConf float32 = 0
		var classID int = -1 // 💡 [추가] 가장 높은 확률을 가진 클래스의 인덱스
		
		// YOLO 출력에서 인덱스 4번부터가 클래스 확률값입니다.
		for r := 4; r < rows; r++ {
			conf := data[r*cols+c]
			if conf > maxClassConf {
				maxClassConf = conf
				classID = r - 4 // 실제 클래스 번호는 0부터 시작하도록 4를 빼줍니다.
			}
		}

		if maxClassConf > confThreshold {
			cx, cy := data[0*cols+c], data[1*cols+c]
			w, h := data[2*cols+c], data[3*cols+c]

			left := int((cx - w/2) * xFactor)
			top := int((cy - h/2) * yFactor)
			width, height := int(w*xFactor), int(h*yFactor)

			boxes = append(boxes, image.Rect(left, top, left+width, top+height))
			confidences = append(confidences, maxClassConf)
			classIDs = append(classIDs, classID) // 💡 [추가]
		}
	}
	return boxes, confidences, classIDs
}

// ==========================================
// 4. Main 로직
// ==========================================
func main() {
	// 💡 [핵심 추가]: 직접 학습시킨 커스텀 모델의 클래스 이름들
	classNames := []string{"flask01", "flask02"}

	streamer := &WebStreamer{}

	go func() {
		http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "text/html; charset=utf-8")
			w.Write([]byte("<h1>Flask Detection System</h1><img src='/video_feed' width='640'>"))
		})
		http.Handle("/video_feed", streamer)
		log.Fatal(http.ListenAndServe("0.0.0.0:5000", nil))
	}()

	camera, err := NewPiCamera()
	if err != nil {
		log.Fatalf("카메라 에러: %v", err)
	}
	defer camera.Close()

	log.Println("⚙️ 커스텀 YOLO 모델 로딩 중...")
	// 💡 [핵심 수정]: 절대 경로를 사용하여 실행 위치와 무관하게 동작하도록 고정
	net := gocv.ReadNet("/home/rnd/HJ/YOLOGo/best_yolov26s.onnx", "")
	if net.Empty() {
		log.Fatal("모델 로드 실패! 경로를 다시 확인해주세요.")
	}
	defer net.Close()
	net.SetPreferableBackend(gocv.NetBackendDefault)
	net.SetPreferableTarget(gocv.NetTargetCPU)

	img := gocv.NewMat()
	defer img.Close()
	displayImg := gocv.NewMat()
	defer displayImg.Close()

	green := color.RGBA{0, 255, 0, 0}
	red := color.RGBA{255, 0, 0, 0}
	white := color.RGBA{255, 255, 255, 0}
	roiRect := image.Rect(206, 212, 434, 268)

	var finalBoxes []image.Rectangle
	var finalConfs []float32
	var finalClassIDs []int // 💡 [추가] 공유 변수에 클래스 ID 추가
	var yoloMutex sync.RWMutex
	var isYoloRunning bool

	log.Println("✅ 비전 시스템 가동 시작 (다중 클래스 매핑 적용)")

	frameCount := 0
	skipFrames := 3

	for {
		if ok := camera.Read(&img); !ok || img.Empty() {
			time.Sleep(5 * time.Millisecond)
			continue
		}

		img.CopyTo(&displayImg)

		if frameCount%skipFrames == 0 && !isYoloRunning {
			isYoloRunning = true
			yoloImg := img.Clone()

			go func() {
				defer yoloImg.Close()

				blob := gocv.BlobFromImage(yoloImg, 1.0/255.0, image.Pt(640, 640), gocv.NewScalar(0, 0, 0, 0), true, false)
				net.SetInput(blob, "")
				prob := net.Forward("")

				// 💡 클래스 ID 반환값 수신
				boxes, confidences, classIDs := parseYOLOOutput(&prob, 0.5, 640, 480)
				var tempBoxes []image.Rectangle
				var tempConfs []float32
				var tempClassIDs []int

				if len(boxes) > 0 {
					indices := gocv.NMSBoxes(boxes, confidences, 0.5, 0.4)
					for _, idx := range indices {
						tempBoxes = append(tempBoxes, boxes[idx])
						tempConfs = append(tempConfs, confidences[idx])
						tempClassIDs = append(tempClassIDs, classIDs[idx]) // 💡 추출된 ID 저장
					}
				}

				prob.Close()
				blob.Close()

				yoloMutex.Lock()
				finalBoxes = tempBoxes
				finalConfs = tempConfs
				finalClassIDs = tempClassIDs // 💡 공유 변수 업데이트
				yoloMutex.Unlock()

				isYoloRunning = false
			}()
		}

		yoloMutex.RLock()
		boxesToDraw := finalBoxes
		confsToDraw := finalConfs
		classIDsToDraw := finalClassIDs // 💡 렌더링을 위해 읽어오기
		yoloMutex.RUnlock()

		gocv.Rectangle(&displayImg, roiRect, white, 1)
		gocv.PutText(&displayImg, "Target ROI", image.Pt(roiRect.Min.X, roiRect.Min.Y-5), gocv.FontHersheySimplex, 0.5, white, 1)

		isInROI := false
		for i, box := range boxesToDraw {
			cx := (box.Min.X + box.Max.X) / 2
			cy := (box.Min.Y + box.Max.Y) / 2

			statusColor := red
			if cx >= roiRect.Min.X && cx <= roiRect.Max.X && cy >= roiRect.Min.Y && cy <= roiRect.Max.Y {
				isInROI = true
				statusColor = green
			}

			gocv.Rectangle(&displayImg, box, statusColor, 2)

			// 💡 [핵심 수정]: 인덱스 번호를 기반으로 매핑된 클래스 이름을 가져옵니다.
			className := "Unknown"
			if classIDsToDraw[i] >= 0 && classIDsToDraw[i] < len(classNames) {
				className = classNames[classIDsToDraw[i]]
			}

			// 가져온 이름과 신뢰도 출력 (예: "flask01 (95.4%)")
			confText := fmt.Sprintf("%s (%.1f%%)", className, confsToDraw[i]*100)
			gocv.PutText(&displayImg, confText, image.Pt(box.Min.X, box.Min.Y-10), gocv.FontHersheySimplex, 0.5, statusColor, 2)
		}

		if isInROI {
			gocv.PutText(&displayImg, "STATUS: DETECTED", image.Pt(10, 20), gocv.FontHersheySimplex, 0.5, green, 2)
		} else {
			gocv.PutText(&displayImg, "STATUS: NONE", image.Pt(10, 20), gocv.FontHersheySimplex, 0.5, red, 2)
		}

		buf, _ := gocv.IMEncode(".jpg", displayImg)
		rawBytes := buf.GetBytes()
		frameCopy := make([]byte, len(rawBytes))
		copy(frameCopy, rawBytes)
		streamer.UpdateFrame(frameCopy)
		buf.Close()

		frameCount++
	}
}