## RaspberryPi 원격 사용방법

### PuTTY 접속
- IP: ```192.168.137.2``` (SubnetSyncer에서 확인)
- user name: ```rnd```
- password: ```801```

### VScode 설정
- VScode 실행 후 F1 → Remote-SSH: Add New SSH Host.... 선택 → ```rnd@192.168.137.2``` 입력 → 'C:\User\robot\.ssh\config' 선택 → 오른쪽 하단 'connect' 클릭 → password 입력

### 인터넷 연결
- [window + r] → ```ncpa.cpl``` 입력 → 이더넷 3 (raspberry pi 네트워크) 주소 확인: 인터넷 프로토콜 버전 4(TCP/IPv4)에서 수동으로 IP ```192.168.137.1```, 서브넷 ```255.255.255.0``` 설정
- [window + r] → ```ncpa.cpl``` 입력 → 이더넷 (유선 LAN: 인터넷) → 속성 → 공유 → (N) 체크 → 이더넷 3를 홈 네트워킹 연결로 선택
- 인터넷 연결 확인: ```ping -c 3 8.8.8.8```

## Github update

### Push code to github in terminal
```
git init
```
### See connected repository & push
```
git remote -v
```
```
git add .
```
```
git commit -m "20251202"
```
```
git push
```

### Pull code from github in terminal
```
git pull
```
```
git pull origin main
```

### Virtual environment
- Activate
```
source /home/rnd/myenv/bin/activate
```
- Deactivate
```
deactivate
```

### 새로운 카메라 추가
```
sudo nano /boot/firmware/config.txt
```
- dtoverlay=imx~~~을 추가하여 드라이버 활성화

### Docker 실행
- Docker 데몬의 현재 상태 확인
```
sudo systemctl status docker
```

- Docker 데몬의 실행
```
sudo systemctl start docker
```

- Docker 데몬의 종료
```
sudo systemctl stop docker
```

### Taskset을 이용한 코어 분산 후 코드 실행
```
taskset -c 0 python Publisher_cameradata.py
```
```
taskset -c 1,2 python Subscriber_AprilTagBarcodeScanner.py
```
```
taskset -c 0,3 python Subscriber_yolo26n.py
```

### 라이선스 검증 도구 Trivy 사용법
- 파이썬 의존성 파일(requirements.txt) 수동 생성
```
pip freeze > ./~~~/~~~/requirements.txt
```

- 생성된 .txt 파일을 Trivy로 스캔
```
trivy fs --scanners license ./~~~/~~~/requirements.txt
```