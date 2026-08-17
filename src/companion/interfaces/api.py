"""Minimal stdlib HTTP API.

Endpoints:
  GET  /health     status, hardware, loaded models, memory guard
  GET  /metrics    Metrics snapshot
  GET  /runtime    full runtime report
  POST /chat       {"text": "...", "speak": false} -> {"text": ..., "plan": ...}
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class _Handler(BaseHTTPRequestHandler):
    app = None

    def log_message(self, fmt, *args):  # quieter
        return

    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload, indent=2, default=str).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        app = type(self).app
        path = self.path.split("?", 1)[0]
        if path == "/health":
            self._send(200, app.health())
        elif path == "/metrics":
            self._send(200, app.components.metrics.snapshot())
        elif path == "/runtime":
            self._send(200, app.runtime_report())
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):  # noqa: N802
        app = type(self).app
        path = self.path.split("?", 1)[0]
        if path != "/chat":
            self._send(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length) or b"{}")
            text = str(payload.get("text", ""))
            if not text.strip():
                self._send(400, {"error": "'text' required"})
                return
            result = app.submit_from_thread(
                app.respond(text, source="api", speak=bool(payload.get("speak", False)))
            ).result()
            self._send(200, result)
        except Exception as exc:
            self._send(500, {"error": str(exc)})


async def serve(app, host: str = "127.0.0.1", port: int = 8611) -> int:
    await app.start()
    handler = type("AppHandler", (_Handler,), {"app": app})
    httpd = ThreadingHTTPServer((host, port), handler)
    print(f"companion API listening on http://{host}:{port}")
    try:
        # The stdlib server is blocking, so run it in an adapter thread while
        # CompanionApp's lifecycle remains alive on the CLI-owned event loop.
        import asyncio

        await asyncio.to_thread(httpd.serve_forever)
    finally:
        httpd.shutdown()
        httpd.server_close()
        await app.aclose()
    return 0
