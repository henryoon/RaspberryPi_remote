#!/usr/bin/env python3
"""Flask adapter for mounting RobotLink Pi in an existing dashboard."""
import os

# Apply the same conservative numerical-library thread limits before importing
# the RobotLink core (or Flask and any application modules that may load BLAS).
for _thread_env in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_env] = "1"

from flask import Blueprint, Flask, jsonify, redirect, request

import app as core


def create_blueprint():
    """Create a blueprint to mount into the existing Raspberry Pi Flask app.

    Example: app.register_blueprint(create_blueprint(), url_prefix="/robotlink")
    """
    bp = Blueprint("robotlink", __name__)

    @bp.get("/")
    def home():
        return core.PAGE

    @bp.get("/api/status")
    def status():
        with core.LOCK:
            return jsonify(config=core.CONFIG, state=core.STATE)

    @bp.get("/api/interfaces")
    def get_interfaces():
        rows, err = core.interfaces()
        return jsonify(interfaces=rows, error=err, serial_ports=core.serial_ports())

    @bp.post("/api/scan")
    def scan():
        body = request.get_json(silent=True) or {}
        interface = str(body.get("interface", ""))
        core.CONFIG["interface"] = interface
        core.save_config()
        results = core.scan_network(interface)
        port = int(core.CONFIG.get("controller_port", 4000))
        matches = [item for item in results if any(p["port"] == port for p in item["ports"])]
        auto_connected = ""
        if len(matches) == 1:
            try:
                core.connect_tcp(matches[0]["ip"], port)
                auto_connected = f"{matches[0]['ip']}:{port}"
            except OSError as exc:
                core.log(f"후보 자동 연결 실패: {exc}")
        elif len(matches) > 1:
            core.log(f"TCP {port} 후보가 {len(matches)}대여서 자동 연결하지 않았습니다.")
        return jsonify(results=results, auto_connected=auto_connected)

    @bp.post("/api/connect")
    def connect():
        body = request.get_json(silent=True) or {}
        try:
            if body.get("transport") == "serial":
                core.connect_serial(str(body.get("serial_port", "")), int(body.get("baud", 57600)))
            else:
                host = str(body.get("host", ""))
                import ipaddress
                ipaddress.ip_address(host)
                port = int(body.get("port", 4000))
                if not 1 <= port <= 65535:
                    raise ValueError("포트는 1~65535 범위여야 합니다.")
                core.connect_tcp(host, port)
            return jsonify(ok=True)
        except Exception as exc:
            core.log(f"오류: {exc}")
            return jsonify(error=str(exc)), 400

    @bp.post("/api/disconnect")
    def disconnect():
        core.disconnect()
        return jsonify(ok=True)

    @bp.post("/api/arm")
    def arm():
        body = request.get_json(silent=True) or {}
        try:
            enable = bool(body.get("enable"))
            if enable and not core.STATE["connected"]:
                raise RuntimeError("연결된 상태에서만 송신을 활성화할 수 있습니다.")
            core.CONFIG["wtr_armed"] = enable
            core.save_config()
            core.log("WTR 송신 활성화" if enable else "WTR 송신 비활성화")
            return jsonify(ok=True)
        except Exception as exc:
            core.log(f"오류: {exc}")
            return jsonify(error=str(exc)), 400

    @bp.post("/api/format")
    def format_config():
        body = request.get_json(silent=True) or {}
        try:
            sep, term = str(body.get("separator", " ")), str(body.get("terminator", ""))
            timeout = float(body.get("timeout", 1.5))
            if len(sep) > 8 or len(term) > 8 or not 0.1 <= timeout <= 10:
                raise ValueError("형식 값 또는 응답 시간 범위를 확인하세요.")
            if any(c not in "\\rn t0" for c in term):
                raise ValueError("종단 문자는 \\r, \\n, \\t, \\0 표기만 허용합니다.")
            core.CONFIG.update(command_separator=sep, command_terminator=term,
                               timeout_seconds=timeout)
            core.save_config()
            core.log(f"통신 형식 변경: separator={sep!r}, terminator={term!r}, timeout={timeout}s")
            return jsonify(ok=True)
        except Exception as exc:
            core.log(f"오류: {exc}")
            return jsonify(error=str(exc)), 400

    @bp.post("/api/send")
    def send():
        body = request.get_json(silent=True) or {}
        try:
            if body.get("confirm") is not True:
                raise RuntimeError("명령 송신 확인이 필요합니다.")
            return jsonify(core.send_wtr(body.get("command", ""), body.get("params", ""),
                                         body.get("expected", "")))
        except Exception as exc:
            core.log(f"오류: {exc}")
            return jsonify(error=str(exc)), 400

    return bp


def create_app():
    flask_app = Flask(__name__)
    flask_app.register_blueprint(create_blueprint(), url_prefix="/robotlink")

    @flask_app.get("/")
    def index():
        return redirect("/robotlink/")

    return flask_app


if __name__ == "__main__":
    flask_app = create_app()
    flask_app.run(host=os.environ.get("ROBOTLINK_HOST", "0.0.0.0"),
                  port=int(os.environ.get("ROBOTLINK_PORT", "5001")), threaded=True)
