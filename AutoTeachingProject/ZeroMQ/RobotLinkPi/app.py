#!/usr/bin/env python3
"""RobotLink Pi: local, operator-confirmed WTR TCP/serial control prototype."""
from __future__ import annotations

import os
for _thread_env in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_env] = "1"

import concurrent.futures
import glob
import ipaddress
import json
import os
import socket
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

APP_DIR = Path(__file__).resolve().parent
CONFIG_PATH = Path(os.environ.get("ROBOTLINK_CONFIG", Path.home() / ".config/robotlinkpi/config.json"))
DEFAULTS = {
    "interface": "",
    "controller_ip": "",
    "controller_port": 4000,
    "scan_ports": [21, 23, 80, 443, 4000],
    "transport": "tcp",
    "serial_port": "/dev/ttyUSB0",
    "baud_rate": 57600,
    "command_separator": " ",
    "command_terminator": "\\r\\n",
    "timeout_seconds": 1.5,
    "wtr_armed": False,
}
COMMANDS = ["GET", "PUT", "PRDY", "EXTA", "RETA", "GRDY", "P2P", "CRO", "ORG", "HOME"]
PORT_NAMES = {21: "FTP", 23: "Telnet", 80: "HTTP", 443: "HTTPS", 4000: "WTR (예시 포트)"}
LOCK = threading.RLock()
STATE = {"connected": False, "transport": "", "peer": "", "last_rx": "", "last_tx": "", "scan": [], "log": []}
LINK = {"socket": None, "serial": None}


def load_config():
    cfg = dict(DEFAULTS)
    try:
        loaded = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            cfg.update(loaded)
    except (OSError, ValueError):
        pass
    return cfg


CONFIG = load_config()
# A previous session's enable flag must never re-arm command transmission after restart.
CONFIG["wtr_armed"] = False


def save_config():
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(CONFIG, ensure_ascii=False, indent=2), encoding="utf-8")


def log(message):
    stamp = time.strftime("%H:%M:%S")
    with LOCK:
        STATE["log"].append(f"{stamp}  {message}")
        STATE["log"] = STATE["log"][-160:]


def interfaces():
    """Return active IPv4 interfaces using the Raspberry Pi OS `ip` utility."""
    try:
        raw = subprocess.run(["ip", "-j", "-4", "addr", "show"], check=True,
                             capture_output=True, text=True, timeout=3)
        data = json.loads(raw.stdout)
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        return [], f"네트워크 인터페이스를 읽지 못했습니다: {exc}"
    found = []
    for item in data:
        if item.get("ifname") == "lo" or "UP" not in item.get("flags", []):
            continue
        for addr in item.get("addr_info", []):
            if addr.get("family") == "inet" and addr.get("scope") != "host":
                found.append({"name": item.get("ifname", ""), "address": addr.get("local", ""),
                              "prefix": addr.get("prefixlen", 24), "cidr": f"{addr.get('local')}/{addr.get('prefixlen')}"})
    return found, ""


def serial_ports():
    paths = set(glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*") + glob.glob("/dev/serial/by-id/*"))
    return sorted(paths)


def mdns_query(service="_services._dns-sd._udp.local"):
    """Passive-impact DNS-SD browse query; returns A records included by responders."""
    labels = service.rstrip(".").split(".")
    qname = b"".join(bytes([len(label)]) + label.encode("ascii") for label in labels) + b"\0"
    packet = b"\0\0\0\0\0\1\0\0\0\0\0\0" + qname + b"\0\x0c\0\x01"
    answers = set()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    try:
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 255)
        sock.settimeout(0.45)
        sock.sendto(packet, ("224.0.0.251", 5353))
        deadline = time.monotonic() + 0.9
        while time.monotonic() < deadline:
            try:
                data, _ = sock.recvfrom(9000)
            except socket.timeout:
                break
            if len(data) < 12:
                continue
            qd, an, ns, ar = [int.from_bytes(data[i:i + 2], "big") for i in (4, 6, 8, 10)]
            offset = 12
            try:
                for _ in range(qd):
                    offset = _skip_dns_name(data, offset) + 4
                for _ in range(an + ns + ar):
                    offset = _skip_dns_name(data, offset)
                    rtype = int.from_bytes(data[offset:offset + 2], "big")
                    rdlen = int.from_bytes(data[offset + 8:offset + 10], "big")
                    offset += 10
                    rdata = data[offset:offset + rdlen]
                    if rtype == 1 and len(rdata) == 4:
                        answers.add(socket.inet_ntoa(rdata))
                    offset += rdlen
                    if offset > len(data):
                        break
            except (IndexError, ValueError):
                continue
    except OSError:
        pass
    finally:
        sock.close()
    return sorted(answers)


def _skip_dns_name(data, offset):
    """Advance over a possibly compressed DNS name."""
    while offset < len(data):
        length = data[offset]
        if length == 0:
            return offset + 1
        if length & 0xC0 == 0xC0:
            return offset + 2
        offset += 1 + length
    raise ValueError("invalid DNS name")


def probe_host(ip, ports):
    open_ports = []
    for port in ports:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(0.22)
        try:
            if sock.connect_ex((ip, port)) == 0:
                open_ports.append({"port": port, "service": PORT_NAMES.get(port, "TCP")})
        except OSError:
            pass
        finally:
            sock.close()
    if not open_ports:
        return None
    return {"ip": ip, "ports": open_ports, "source": "TCP 응답"}


def scan_network(interface_name):
    rows, err = interfaces()
    if err:
        raise RuntimeError(err)
    row = next((x for x in rows if x["name"] == interface_name), None)
    if not row:
        raise RuntimeError("선택한 인터페이스에서 IPv4 주소를 찾지 못했습니다.")
    network = ipaddress.ip_interface(row["cidr"]).network
    if network.num_addresses > 256:
        raise RuntimeError(f"{network}는 검색 범위가 큽니다. 제어기 전용 인터페이스 또는 더 작은 대역을 사용하세요.")
    hosts = {str(ip) for ip in network.hosts() if str(ip) != row["address"]}
    # mDNS adds candidates on the local link even where a host is not in the current IPv4 subnet.
    hosts.update(mdns_query())
    ports = sorted({int(p) for p in CONFIG.get("scan_ports", DEFAULTS["scan_ports"]) if 1 <= int(p) <= 65535})
    with concurrent.futures.ThreadPoolExecutor(max_workers=48) as pool:
        results = [r for r in pool.map(lambda ip: probe_host(ip, ports), sorted(hosts, key=lambda x: tuple(int(q) for q in x.split(".")))) if r]
    results.sort(key=lambda x: tuple(int(q) for q in x["ip"].split(".")))
    with LOCK:
        STATE["scan"] = results
    log(f"검색 완료: {network}, 후보 {len(results)}대. 열린 포트만으로 제어기를 확정하지 않습니다.")
    return results


def disconnect():
    with LOCK:
        sock, ser = LINK["socket"], LINK["serial"]
        LINK["socket"] = LINK["serial"] = None
        STATE.update(connected=False, transport="", peer="", last_rx="")
    for item in (sock, ser):
        if item:
            try:
                item.close()
            except Exception:
                pass
    log("연결을 닫았습니다.")


def connect_tcp(host, port):
    disconnect()
    sock = socket.create_connection((host, int(port)), timeout=float(CONFIG["timeout_seconds"]))
    sock.settimeout(float(CONFIG["timeout_seconds"]))
    with LOCK:
        LINK["socket"] = sock
        STATE.update(connected=True, transport="TCP", peer=f"{host}:{port}")
    CONFIG.update(transport="tcp", controller_ip=host, controller_port=int(port))
    save_config()
    log(f"TCP 연결됨: {host}:{port}")


def connect_serial(port, baud):
    try:
        import serial
    except ImportError as exc:
        raise RuntimeError("시리얼 제어에는 pyserial이 필요합니다. README 설치 절차를 확인하세요.") from exc
    disconnect()
    ser = serial.Serial(port=port, baudrate=int(baud), timeout=float(CONFIG["timeout_seconds"]),
                        write_timeout=float(CONFIG["timeout_seconds"]))
    with LOCK:
        LINK["serial"] = ser
        STATE.update(connected=True, transport="Serial", peer=f"{port} @ {baud}")
    CONFIG.update(transport="serial", serial_port=port, baud_rate=int(baud))
    save_config()
    log(f"시리얼 연결됨: {port} @ {baud}")


def decode_escapes(value):
    return value.replace("\\r", "\r").replace("\\n", "\n").replace("\\t", "\t").replace("\\0", "\0")


def send_wtr(command, params, expected):
    command = str(command).strip().upper()
    params = str(params).strip()
    expected = str(expected).strip()
    if command not in COMMANDS:
        raise RuntimeError("허용된 명령 목록에 없는 명령입니다.")
    if any(ch in command + params for ch in "\r\n\0"):
        raise RuntimeError("명령과 인자에 줄바꿈/NUL 문자를 넣을 수 없습니다.")
    if len(params) > 96:
        raise RuntimeError("인자가 너무 깁니다.")
    with LOCK:
        if not CONFIG.get("wtr_armed"):
            raise RuntimeError("먼저 WTR 송신을 활성화해야 합니다.")
        sock, ser = LINK["socket"], LINK["serial"]
    if not (sock or ser):
        raise RuntimeError("먼저 TCP 또는 시리얼로 연결하세요.")
    separator = decode_escapes(str(CONFIG.get("command_separator", " ")))
    terminator = decode_escapes(str(CONFIG.get("command_terminator", "\\r\\n")))
    body = command + (separator + params if params else "")
    wire = (body + terminator).encode("ascii", errors="strict")
    with LOCK:
        if sock:
            sock.sendall(wire)
        else:
            ser.write(wire)
            ser.flush()
        STATE["last_tx"] = wire.hex(" ").upper()
    log(f"TX {body!r} + terminator ({wire.hex(' ').upper()})")
    chunks = []
    deadline = time.monotonic() + float(CONFIG["timeout_seconds"])
    while time.monotonic() < deadline:
        try:
            if sock:
                data = sock.recv(4096)
            else:
                data = ser.read(ser.in_waiting or 1)
        except (socket.timeout, TimeoutError):
            break
        except OSError as exc:
            raise RuntimeError(f"응답 읽기 실패: {exc}") from exc
        if not data:
            break
        chunks.append(data)
        if len(b"".join(chunks)) >= 8192:
            break
        if sock:
            sock.settimeout(0.18)
    response = b"".join(chunks).decode("ascii", errors="replace").strip()
    with LOCK:
        STATE["last_rx"] = response
    log(f"RX {response!r}" if response else "RX 응답 없음 (시간 초과 또는 명령 무응답)")
    matched = not expected or expected.lower() in response.lower()
    if expected and response and not matched:
        log(f"기대 응답 키워드 {expected!r}와 일치하지 않습니다.")
    return {"sent_hex": wire.hex(" ").upper(), "response": response, "matched": matched}


PAGE = r'''<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RobotLink Pi</title><style>
:root{color-scheme:dark;--bg:#111827;--panel:#1f2937;--line:#374151;--muted:#9ca3af;--accent:#38bdf8;--warn:#fbbf24}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:#f3f4f6;font:15px system-ui,sans-serif}main{max-width:1100px;margin:auto;padding:24px}h1{margin:0 0 6px;font-size:26px}p{color:var(--muted);margin:4px 0 16px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}.panel{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:18px}.wide{grid-column:1/-1}h2{font-size:18px;margin:0 0 14px}.row{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin:10px 0}label{display:block;color:#d1d5db;font-size:13px;margin:8px 0 4px}input,select,button{font:inherit;border-radius:7px;border:1px solid #4b5563;padding:9px;background:#111827;color:#fff}input,select{min-width:0}input[type=text]{flex:1}button{background:#075985;border-color:#0284c7;cursor:pointer}button.secondary{background:#374151;border-color:#4b5563}button.danger{background:#7f1d1d;border-color:#b91c1c}button:disabled{opacity:.5;cursor:not-allowed}.status{padding:10px;border-radius:8px;background:#111827;margin:10px 0}.warn{color:var(--warn)}.candidate{display:flex;justify-content:space-between;gap:10px;border-top:1px solid var(--line);padding:10px 0}.candidate small{color:var(--muted)}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#0b1220;padding:12px;border-radius:8px;min-height:110px;max-height:260px;overflow:auto}.commands{display:grid;grid-template-columns:repeat(5,1fr);gap:8px}#result{color:#bae6fd;min-height:24px}.hint{font-size:12px;color:var(--muted)}@media(max-width:760px){.grid{grid-template-columns:1fr}.wide{grid-column:auto}.commands{grid-template-columns:repeat(2,1fr)}}
</style></head><body><main>
<h1>RobotLink Pi</h1><p>CM5용 로봇 제어기 검색 및 WTR 수동 제어</p>
<div class="grid">
<section class="panel"><h2>제어기 검색</h2><label>검색할 유선 인터페이스</label><div class="row"><select id="iface"></select><button onclick="scan()">대역 검색</button></div><div class="hint">현재 인터페이스의 IPv4 대역과 mDNS 응답을 검색합니다. TCP 4000 후보가 하나면 연결을 시도하고, 그 외에는 목록에서 선택합니다.</div><div id="candidates"></div></section>
<section class="panel"><h2>연결</h2><div class="row"><select id="transport" onchange="toggleTransport()"><option value="tcp">TCP</option><option value="serial">시리얼</option></select></div>
<div id="tcpFields"><label>제어기 IP (검색 없이 직접 입력 가능)</label><input id="host" type="text" placeholder="예: 192.168.0.10" autocomplete="off"><label>TCP 포트</label><input id="port" type="number" min="1" max="65535" value="4000"><div class="hint">검색되지 않아도 제어기의 IP와 포트를 입력한 뒤 아래 연결 버튼을 누르면 됩니다.</div></div>
<div id="serialFields" hidden><label>장치 경로</label><select id="serialPort"></select><label>Baud rate</label><input id="baud" type="number" value="57600"></div>
<div class="row"><button onclick="connect()">연결</button><button class="secondary" onclick="disconnect()">연결 해제</button></div><div class="status" id="status">상태 확인 중…</div></section>
<section class="panel wide"><h2>WTR 명령</h2><div class="row"><label style="margin:0">명령</label><select id="cmd"></select><label style="margin:0">인자</label><input id="params" type="text" placeholder="예: 1,1,1"><label style="margin:0">기대 응답</label><input id="expected" type="text" placeholder="예: GET" style="max-width:120px"></div>
<details><summary>통신 형식 설정 (장비 매뉴얼 값으로 확인)</summary><div class="row"><label>명령과 인자 사이 문자</label><input id="separator" value=" "><label>종단 문자</label><input id="terminator" value="\\r\\n"><label>응답 시간(초)</label><input id="timeout" type="number" min="0.1" max="10" step="0.1" value="1.5"></div><div class="hint">예제 기본값은 검증되지 않았습니다. 컨트롤러 매뉴얼에 맞춰 확인한 뒤 저장하세요. \r, \n 표기는 이스케이프 입력입니다.</div><button class="secondary" onclick="saveFormat()">형식 저장</button></details>
<div class="row"><button class="danger" id="arm" onclick="toggleArm()">송신 비활성</button><label><input type="checkbox" id="confirm"> 이동 가능성이 있는 로봇 명령임을 확인했습니다</label></div><div class="row"><button id="send" onclick="send()" disabled>선택한 명령 1회 송신</button><span class="hint">자동 반복/시퀀스 실행은 없습니다.</span></div><div id="result"></div></section>
<section class="panel wide"><h2>통신 기록</h2><pre id="log">아직 기록이 없습니다.</pre></section>
</div></main><script>
let cfg={},state={};
const API_ROOT=location.pathname.startsWith('/robotlink')?'/robotlink/api':'/api';
async function api(path,data){let r=await fetch(API_ROOT+'/'+path,{method:data?'POST':'GET',headers:data?{'Content-Type':'application/json'}:{},body:data?JSON.stringify(data):undefined});let j=await r.json();if(!r.ok)throw Error(j.error||r.statusText);return j}
function esc(s){return String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
async function refresh(){try{let x=await api('status');cfg=x.config;state=x.state;document.querySelector('#status').textContent=state.connected?`연결됨: ${state.transport} ${state.peer}`:'연결되지 않음';document.querySelector('#transport').value=cfg.transport;let hostInput=document.querySelector('#host'),portInput=document.querySelector('#port');if(document.activeElement!==hostInput)hostInput.value=cfg.controller_ip;if(document.activeElement!==portInput)portInput.value=cfg.controller_port;document.querySelector('#baud').value=cfg.baud_rate;document.querySelector('#separator').value=cfg.command_separator;document.querySelector('#terminator').value=cfg.command_terminator;document.querySelector('#timeout').value=cfg.timeout_seconds;document.querySelector('#arm').textContent=cfg.wtr_armed?'송신 활성':'송신 비활성';document.querySelector('#send').disabled=!state.connected||!cfg.wtr_armed;document.querySelector('#log').textContent=state.log.join('\n')||'아직 기록이 없습니다.';drawCandidates(state.scan);toggleTransport()}catch(e){msg(e.message)}}
function drawCandidates(items){document.querySelector('#candidates').innerHTML=items.map(x=>`<div class="candidate"><div><b>${esc(x.ip)}</b><br><small>${x.ports.map(p=>`${esc(p.service)}:${p.port}`).join(' · ')}</small></div><button class="secondary" onclick="choose('${esc(x.ip)}',${x.ports[0].port})">선택</button></div>`).join('')||'<div class="hint">검색된 후보가 없습니다.</div>'}
function choose(ip,p){document.querySelector('#host').value=ip;document.querySelector('#port').value=p===21||p===23||p===80||p===443?4000:p}
function msg(s){document.querySelector('#result').textContent=s}
function toggleTransport(){let s=document.querySelector('#transport').value;document.querySelector('#tcpFields').hidden=s!=='tcp';document.querySelector('#serialFields').hidden=s!=='serial'}
async function init(){let x=await api('status');cfg=x.config;state=x.state;document.querySelector('#cmd').innerHTML=['GET','PUT','PRDY','EXTA','RETA','GRDY','P2P','CRO','ORG','HOME'].map(s=>`<option>${s}</option>`).join('');let a=await api('interfaces');document.querySelector('#iface').innerHTML=a.interfaces.map(i=>`<option value="${esc(i.name)}">${esc(i.name)} — ${esc(i.cidr)}</option>`).join('')||'<option value="">IPv4 인터페이스 없음</option>';if(cfg.interface)document.querySelector('#iface').value=cfg.interface;document.querySelector('#serialPort').innerHTML=a.serial_ports.map(p=>`<option>${esc(p)}</option>`).join('')||'<option>/dev/ttyUSB0</option>';document.querySelector('#serialPort').value=cfg.serial_port;await refresh()}
async function scan(){try{msg('검색 중…');let x=await api('scan',{interface:document.querySelector('#iface').value});drawCandidates(x.results);await refresh();msg(x.auto_connected?`유일한 TCP 4000 후보 ${x.auto_connected}에 연결을 시도했습니다. 장비를 확인하세요.`:`${x.results.length}개 후보를 찾았습니다. 대상 장비를 확인한 뒤 선택하세요.`)}catch(e){msg(e.message)}}
async function connect(){try{let body={transport:document.querySelector('#transport').value,host:document.querySelector('#host').value.trim(),port:+document.querySelector('#port').value,serial_port:document.querySelector('#serialPort').value,baud:+document.querySelector('#baud').value};if(body.transport==='tcp'&&!body.host)throw Error('제어기 검색 결과를 선택하거나 IP 주소를 직접 입력하세요.');if(body.transport==='tcp'&&(!Number.isInteger(body.port)||body.port<1||body.port>65535))throw Error('TCP 포트는 1~65535 범위로 입력하세요.');await api('connect',body);await refresh();msg('연결되었습니다. WTR 형식과 대상 장비를 확인하세요.')}catch(e){msg(e.message)}}
async function disconnect(){try{await api('disconnect',{});await refresh();msg('연결을 닫았습니다.')}catch(e){msg(e.message)}}
async function toggleArm(){try{let enable=!cfg.wtr_armed;if(enable&&!confirm('송신을 활성화하면 실제 로봇이 움직일 수 있습니다. 대상·통신형식·인터록을 확인했습니까?'))return;await api('arm',{enable});await refresh()}catch(e){msg(e.message)}}
async function saveFormat(){try{await api('format',{separator:document.querySelector('#separator').value,terminator:document.querySelector('#terminator').value,timeout:+document.querySelector('#timeout').value});await refresh();msg('통신 형식을 저장했습니다.')}catch(e){msg(e.message)}}
async function send(){if(!document.querySelector('#confirm').checked){msg('명령 송신 확인 상자를 체크하세요.');return}let body={command:document.querySelector('#cmd').value,params:document.querySelector('#params').value,expected:document.querySelector('#expected').value,confirm:true};try{let x=await api('send',body);msg(`송신 완료 · RX: ${x.response||'(응답 없음)'} · 기대 응답 ${x.matched?'일치':'미확인'}`);await refresh()}catch(e){msg(e.message)}}
init().catch(e=>msg(e.message));setInterval(refresh,2500);
</script></body></html>'''


class Handler(BaseHTTPRequestHandler):
    server_version = "RobotLinkPi/0.1"

    def log_message(self, *_args):
        pass

    def send_json(self, code, obj):
        raw = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def read_json(self):
        size = int(self.headers.get("Content-Length", "0"))
        if size > 8192:
            raise ValueError("요청이 너무 큽니다.")
        return json.loads(self.rfile.read(size) or b"{}")

    def do_GET(self):
        route = urlparse(self.path).path
        if route == "/":
            body = PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)
        elif route == "/api/status":
            with LOCK:
                self.send_json(200, {"config": CONFIG, "state": STATE})
        elif route == "/api/interfaces":
            rows, err = interfaces()
            self.send_json(200, {"interfaces": rows, "error": err, "serial_ports": serial_ports()})
        else:
            self.send_json(404, {"error": "경로를 찾지 못했습니다."})

    def do_POST(self):
        route = urlparse(self.path).path
        try:
            body = self.read_json()
            if route == "/api/scan":
                iface = str(body.get("interface", ""))
                CONFIG["interface"] = iface
                save_config()
                results = scan_network(iface)
                port = int(CONFIG.get("controller_port", 4000))
                matches = [item for item in results if any(p["port"] == port for p in item["ports"])]
                auto_connected = ""
                if len(matches) == 1:
                    try:
                        connect_tcp(matches[0]["ip"], port)
                        auto_connected = f"{matches[0]['ip']}:{port}"
                    except OSError as exc:
                        log(f"후보 자동 연결 실패: {exc}")
                elif len(matches) > 1:
                    log(f"TCP {port} 후보가 {len(matches)}대여서 자동 연결하지 않았습니다.")
                self.send_json(200, {"results": results, "auto_connected": auto_connected})
            elif route == "/api/connect":
                if body.get("transport") == "serial":
                    connect_serial(str(body.get("serial_port", "")), int(body.get("baud", 57600)))
                else:
                    host = str(body.get("host", ""))
                    ipaddress.ip_address(host)
                    port = int(body.get("port", 4000))
                    if not 1 <= port <= 65535:
                        raise ValueError("포트는 1~65535 범위여야 합니다.")
                    connect_tcp(host, port)
                self.send_json(200, {"ok": True})
            elif route == "/api/disconnect":
                disconnect()
                self.send_json(200, {"ok": True})
            elif route == "/api/arm":
                enable = bool(body.get("enable"))
                if enable and not STATE["connected"]:
                    raise RuntimeError("연결된 상태에서만 송신을 활성화할 수 있습니다.")
                CONFIG["wtr_armed"] = enable
                save_config()
                log("WTR 송신 활성화" if enable else "WTR 송신 비활성화")
                self.send_json(200, {"ok": True})
            elif route == "/api/format":
                sep, term = str(body.get("separator", " ")), str(body.get("terminator", ""))
                timeout = float(body.get("timeout", 1.5))
                if len(sep) > 8 or len(term) > 8 or not 0.1 <= timeout <= 10:
                    raise ValueError("형식 값 또는 응답 시간 범위를 확인하세요.")
                if any(c not in "\\rn t0" for c in term):
                    raise ValueError("종단 문자는 \\r, \\n, \\t, \\0 표기만 허용합니다.")
                CONFIG.update(command_separator=sep, command_terminator=term, timeout_seconds=timeout)
                save_config()
                log(f"통신 형식 변경: separator={sep!r}, terminator={term!r}, timeout={timeout}s")
                self.send_json(200, {"ok": True})
            elif route == "/api/send":
                if body.get("confirm") is not True:
                    raise RuntimeError("명령 송신 확인이 필요합니다.")
                self.send_json(200, send_wtr(body.get("command", ""), body.get("params", ""), body.get("expected", "")))
            else:
                self.send_json(404, {"error": "경로를 찾지 못했습니다."})
        except Exception as exc:
            log(f"오류: {exc}")
            self.send_json(400, {"error": str(exc)})


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Local operator console for WTR discovery and control")
    parser.add_argument("--host", default="127.0.0.1", help="기본은 로컬 전용; 원격 접근은 SSH 터널 권장")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        print("주의: 원격 인터페이스에 바인딩합니다. 방화벽과 접근 통제를 확인하세요.")
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"RobotLink Pi: http://{args.host}:{args.port} (종료: Ctrl+C)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        disconnect()
        server.server_close()


if __name__ == "__main__":
    main()
