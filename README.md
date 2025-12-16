## RaspberryPi 원격 사용방법

### PuTTY 접속
- IP: ```192.168.137.2``` (SubnetSyncer에서 확인)
- user name: ```rnd```
- password: ```12341234```

### VScode 설정
- VScode 실행 후 F1 → Remote-SSH: Add New SSH Host.... 선택 → ```rnd@192.168.137.2``` 입력 → 'C:\User\robot\.ssh\config' 선택 → 오른쪽 하단 'connect' 클릭 → password 입력

### 인터넷 연결
- [window + r] → ```ncpa.cpl``` 입력 → 이더넷 3 (raspberry pi 네트워크) 주소 확인: 인터넷 프로토콜 버전 4(TCP/IPv4)에서 수동으로 IP ```192.168.137.1```, 서브넷 ```255.255.255.0``` 설정
- [window + r] → ```ncpa.cpl``` 입력 → 이더넷 (유선 LAN: 인터넷) → 속성 → 공유 → (N) 체크 → 이더넷 3를 홈 넽,워킹 연결로 선택

<img width="1220" height="542" alt="image" src="https://github.com/user-attachments/assets/784b6ae7-3f39-4da5-935b-754bdac79bd1" />
