# RobotLink Pi

Raspberry Pi CM5-NANO-B에서 사용할 로봇 제어기 자동 탐색 및 WTR 수동 제어 웹 앱입니다. 첨부해주신 Flask 기반 비전 프로그램의 실행 환경을 고려해 다음 두 방식으로 제공합니다.

1. **기존 Flask 프로그램에 통합**: 현재 비전 앱과 같은 `0.0.0.0:5000` 서버에서 `/robotlink/`로 실행
2. **별도 실행**: Flask 앱을 독립 실행하며 기본 포트 `5001` 사용

Python 프로세스가 시작될 때 OpenMP/BLAS 계열 스레드 제한 환경 변수도 1로 지정합니다. RobotLink Pi 단독 기능은 표준 Python 네트워크 라이브러리를 사용하고, 웹 화면을 제공할 때 Flask를 사용합니다.

## 기존 비전 Flask 프로그램에 통합

`app.py`, `flask_app.py` 두 파일을 비전 프로그램의 Python 파일과 같은 폴더에 복사합니다. 첨부 프로그램에서 `app = Flask(__name__)`를 만든 뒤, `app.run(...)` 전에 다음을 추가하세요.

```python
from flask_app import create_blueprint
app.register_blueprint(create_blueprint(), url_prefix="/robotlink")
```

기존 비전 프로그램의 실행 설정은 그대로 둡니다. 예를 들어 기존 코드가 다음처럼 실행되면 비전 화면과 RobotLink Pi가 같은 포트에서 제공됩니다.

```python
app.run(host="0.0.0.0", port=5000, threaded=True)
```

브라우저에서 `http://<라즈베리파이-IP>:5000/robotlink/`로 엽니다. Flask가 첨부 프로그램 환경에 설치되어 있어야 합니다. 다른 PC에서도 접근 가능한 앱이 되므로, 신뢰할 수 있는 실험실 네트워크에서만 사용하세요.

## 별도 Flask 앱으로 실행

```bash
sudo apt update
sudo apt install -y python3 python3-flask
cd RobotLinkPi
python3 flask_app.py
```

`http://<라즈베리파이-IP>:5001/robotlink/`로 접속합니다. 주소/포트는 `ROBOTLINK_HOST`, `ROBOTLINK_PORT` 환경 변수로 바꿀 수 있습니다.

기존 Flask 환경을 사용하지 않고 표준 라이브러리 기반 단독 서버를 실행하는 방법도 있습니다.

```bash
python3 app.py
```

이 경우 `http://127.0.0.1:8765`에서 열립니다. 원격 접속이 필요하면 기본 서버보다 기존 Flask 서버 또는 SSH 터널을 사용하세요.

USB 시리얼을 쓸 때는 다음 패키지를 설치합니다.

```bash
sudo apt install -y python3-serial
```

## 제어기 검색 및 연결

- 활성 IPv4 인터페이스의 최대 `/24` 대역을 검색하고 mDNS 응답도 확인합니다.
- 기본 TCP 검색 포트는 21, 23, 80, 443, 4000입니다. 열린 포트는 장비 모델이나 WTR 프로토콜을 증명하지 않습니다.
- 설정된 TCP 포트의 후보가 한 대일 때만 TCP 연결을 시도합니다. 로봇 명령은 자동 송신하지 않습니다.
- 검색 결과가 없어도 연결 화면에 제어기 IP와 TCP 포트를 직접 입력해 연결할 수 있습니다. 입력 중인 값은 화면 자동 갱신으로 덮어쓰지 않습니다.
- IP 주소/라우팅 설정을 변경하지 않습니다. 제어기가 다른 서브넷에 있으면 라즈베리파이와 제어기의 네트워크 설정을 별도로 맞춰야 합니다.

## WTR 통신 값 확인 필요

기존 EFEM 프로그램에서 추정 가능한 명령 이름은 `GET`, `PUT`, `PRDY`, `EXTA`, `RETA`, `GRDY`, `P2P`, `CRO`, `ORG`, `HOME`이며, `GET`/`PUT`에 `1,1,1` 형태의 예시 인자가 있었습니다. TCP 4000도 기존 예제의 값으로만 제공됩니다. Robot & Design에서 개조한 Brooks 제어기의 공통 사양으로 확인된 값은 아닙니다.

명령 구분자, 종단 문자, 응답 형식, 실제 포트와 인터록은 컨트롤러 매뉴얼 또는 정상 동작 중인 EFEM 통신 로그로 확인해야 합니다. 화면의 기본 형식(`명령 + 공백 + 인자 + CRLF`)은 검증되지 않았습니다. 연결만으로 로봇을 움직이지 않지만, 송신을 활성화하고 명령을 보내면 실제 로봇이 움직일 수 있습니다. 매 송신 확인과 활성화 버튼이 있으며 자동 반복/자동 시퀀스는 없습니다.

## 설정 및 로그

설정 파일은 `~/.config/robotlinkpi/config.json`에 저장합니다. 다른 위치를 사용하려면 앱 실행 전에 `ROBOTLINK_CONFIG`를 지정하세요. `wtr_armed` 상태는 재시작 시 항상 해제됩니다.
