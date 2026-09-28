"""A tiny support-ticket API for the M02 lab, with a switch that makes it fail.

    python m02/ticket_api.py                 # healthy, on http://localhost:8765
    python m02/ticket_api.py --fail-first 4  # the first 4 requests get HTTP 503, then it recovers

    POST /tickets        {"order_id": "A1002", "issue": "..."}  -> {"ticket_id": "T-1001", ...}
    GET  /tickets/T-1001                                        -> the ticket
    GET  /health                                                -> {"ok": true}

Tickets live in memory only; restart the server to start again.
"""
import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

TICKETS: dict[str, dict] = {}
STATE = {"requests": 0, "fail_first": 0, "next_id": 1001}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: dict):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _flaky(self) -> bool:
        """True (and a 503 already sent) while the server is pretending to be down."""
        STATE["requests"] += 1
        if STATE["requests"] <= STATE["fail_first"]:
            print(f"request {STATE['requests']}: 503 (simulated outage)", flush=True)
            self._send(503, {"error": "ticket service unavailable"})
            return True
        return False

    def do_GET(self):
        if self.path == "/health":
            return self._send(200, {"ok": True})
        if self._flaky():
            return
        if self.path.startswith("/tickets/"):
            t = TICKETS.get(self.path.rsplit("/", 1)[-1].upper())
            return self._send(200, t) if t else self._send(404, {"error": "no such ticket"})
        self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/tickets":
            return self._send(404, {"error": "not found"})
        if self._flaky():
            return
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        tid = f"T-{STATE['next_id']}"
        STATE["next_id"] += 1
        TICKETS[tid] = {"ticket_id": tid, "order_id": str(body.get("order_id", "")).upper(),
                        "issue": body.get("issue", ""), "status": "open"}
        print(f"request {STATE['requests']}: created {tid}", flush=True)
        self._send(201, TICKETS[tid])

    def log_message(self, *args):  # keep the terminal readable
        pass


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--fail-first", type=int, default=0, help="answer the first N requests with HTTP 503")
    a = ap.parse_args()
    STATE["fail_first"] = a.fail_first
    print(f"ticket API on http://localhost:{a.port} (failing the first {a.fail_first} requests)", flush=True)
    ThreadingHTTPServer(("localhost", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
