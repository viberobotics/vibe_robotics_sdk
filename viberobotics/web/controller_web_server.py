from viberobotics.constants import ControlMode

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse
from pathlib import Path
import numpy as np
import webbrowser

def round1(value: float) -> float:
    return round(value, 1)

class ControllerWebServer:
    class Handler(BaseHTTPRequestHandler):
        def __init__(self, request, client_address, server):
            # Store shared state references provided by ControllerWebServer
            self.lock = server.lock
            self.state = server.state
            with open(Path(__file__).parent / 'controller_client.html', 'r', encoding='utf-8') as f:
                self.page = f.read().encode('utf-8')
            # Base class __init__ will start handling the request immediately.
            super().__init__(request, client_address, server)
            
        def _send(self, code, body=b"ok", ctype="text/plain; charset=utf-8"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _status_json(self):
            mode = self.state['mode']
            return json.dumps(
                {
                    "vector": self.state['vector'],
                    "mode": mode.name if isinstance(mode, ControlMode) else str(mode),
                }
            ).encode("utf-8")

        def do_GET(self):
            if urlparse(self.path).path in ("/", "/index.html"):
                return self._send(200, self.page, "text/html; charset=utf-8")
            if urlparse(self.path).path == "/status":
                with self.lock:
                    body = self._status_json()
                return self._send(200, body=body, ctype="application/json; charset=utf-8")
            return self._send(404, b"not found")

        def do_POST(self):
            path = urlparse(self.path).path

            if path == "/state":
                n = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(n)
                try:
                    msg = json.loads(raw.decode("utf-8"))
                    vector = msg["vector"]
                    with self.lock:
                        self.state['vector']["x"] = round1(vector["x"])
                        self.state['vector']["y"] = round1(vector["y"])
                        self.state['vector']["yaw"] = round1(vector["yaw"])
                        body = self._status_json()
                    return self._send(200, body=body, ctype="application/json; charset=utf-8")
                except Exception:
                    return self._send(400, b"bad request")

            if path == "/mode":
                n = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(n)
                try:
                    msg = json.loads(raw.decode("utf-8"))
                    new_mode = msg.get("mode")
                    if isinstance(new_mode, str):
                        try:
                            new_mode_parsed = ControlMode[new_mode.upper()]
                        except KeyError:
                            return self._send(400, b"bad mode")
                    elif isinstance(new_mode, (int, float)):
                        try:
                            new_mode_parsed = ControlMode(int(new_mode))
                        except Exception:
                            return self._send(400, b"bad mode")
                    else:
                        return self._send(400, b"bad mode")

                    with self.lock:
                        self.state['mode'] = new_mode_parsed
                        print(self.state['mode'])
                        body = self._status_json()
                    return self._send(200, body=body, ctype="application/json; charset=utf-8")
                except Exception:
                    return self._send(400, b"bad request")

            if path == "/release_all":
                with self.lock:
                    self.state['vector']["x"] = 0.0
                    self.state['vector']["y"] = 0.0
                    self.state['vector']["yaw"] = 0.0
                return self._send(200)

            return self._send(404, b"not found")
        
        def log_message(self, format, *args):
            return

    @staticmethod
    def _normalize_mode(mode_value):
        if isinstance(mode_value, ControlMode):
            return mode_value
        if isinstance(mode_value, str):
            try:
                return ControlMode[mode_value.upper()]
            except KeyError as exc:
                raise ValueError(f"Unknown control mode '{mode_value}'") from exc
        try:
            return ControlMode(int(mode_value))
        except Exception as exc:
            raise ValueError(f"Unknown control mode '{mode_value}'") from exc

    def __init__(self, initial_mode=ControlMode.NONE):
        self.lock = threading.Lock()
        self.state = {
            'vector': {"x": 0.0, "y": 0.0, "yaw": 0.0},
            'mode': self._normalize_mode(initial_mode)
        }
    
    def start_server(self, host="127.0.0.1", port=3000):
        httpd = ThreadingHTTPServer((host, port), ControllerWebServer.Handler)
        # Expose shared state to the handler via the server instance
        httpd.lock = self.lock
        httpd.state = self.state
        t = threading.Thread(target=httpd.serve_forever, daemon=True)
        t.start()
        print(f"Controller web server listening on http://{host}:{port}")
        webbrowser.open(f"http://{host}:{port}")
    
    def get_key_state_snapshot(self):
        with self.lock:
            return {"vector": dict(self.state['vector']), "mode": self.state['mode'].name}
    
    def get_control_input(self):
        with self.lock:
            return np.array([self.state['vector']["x"], self.state['vector']["y"], self.state['vector']["yaw"]], dtype=np.float32)
    
    def get_control_mode(self):
        with self.lock:
            return self.state['mode']
    
if __name__ == "__main__":
    server = ControllerWebServer()
    server.start_server()
    while True:
        time.sleep(5)
        print("Current key state:", server.get_key_state_snapshot())
