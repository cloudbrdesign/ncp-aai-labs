"""Step 1: run several replicas of the M6 support desk, each one an HTTP service (`nat serve`).

    python m07/serve_fleet.py --replicas 3          # r1..r3 on ports 8101..8103; Ctrl-C stops them
    curl -s localhost:8101/health
    curl -s -X POST localhost:8101/v1/chat -H 'content-type: application/json' \
         -d '{"messages":[{"role":"user","content":"Where is order A1003?"}]}'

Each replica is one `nat serve` process (NeMo Agent Toolkit 1.9.0's FastAPI front end) running
configs/desk_serve.yml: the M6 desk (m06/desk_app.py:agent) as an API with /health, /v1/chat,
/generate and the rest of NAT's endpoints.

What the replicas share and what they don't:
  shared      the model tier: one Ollama on this laptop (llama3.2:3b and embeddinggemma). Every
              replica sends its LLM calls there, as agent pods share a NIM in a cluster.
  per replica its own state folder, m07/state/replicas/rN (M04_STATE_DIR): a copy of the M4
              index (Milvus Lite locks its file to one process) and the desk's own memory.db and
              threads.db. So whatever one replica remembers, the others don't know: state kept
              inside a replica is lost when the replica dies and is invisible to its neighbours
              (lesson 7.1: keep conversation state in a shared store, or pin a user to a replica).

Other scripts (balancer.py's tests, load_test.py, failover_drill.py, check.py) use the Fleet class.
Offline self-test only: M07_FAKE_LLM=1 starts M6's scripted stand-in for Ollama (m06/tests/fake_oai.py)
with 1 slot and M07_FAKE_DELAY seconds (default 0.3) per call, so the model tier is a bottleneck as
on a laptop. Learners never need it.
"""
import argparse
import json
import os
import pathlib
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
LABS = HERE.parent
STATE = HERE / "state"
CONFIG = HERE / "configs" / "desk_serve.yml"
M04_STATE = LABS / "m04" / "state"
BASE_PORT = 8101
_fake = {"proc": None, "url": None}


def fake_mode() -> bool:
    return os.environ.get("M07_FAKE_LLM") == "1"


def nat_cmd() -> list[str]:
    """The `nat` command of this venv (next to this Python), else the one on PATH."""
    local = pathlib.Path(sys.executable).parent / "nat"
    if local.exists():
        return [str(local)]
    found = shutil.which("nat")
    if not found:
        raise SystemExit("[ERROR] `nat` not found: activate the course venv (Module 0) first")
    return [found]


def port_free(port: int) -> bool:
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def get_json(url: str, timeout: float = 2.0):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.status, json.loads(r.read() or b"null")


def healthy(url: str, timeout: float = 1.0) -> bool:
    try:
        status, body = get_json(url + "/health", timeout)
        return status == 200 and (body or {}).get("status") == "healthy"
    except Exception:
        return False


def start_fake_model() -> str:
    """Offline self-test: one scripted model server shared by every replica (like one Ollama)."""
    if _fake["url"]:
        return _fake["url"]
    sys.path.insert(0, str(LABS / "m06" / "tests"))
    import fake_oai
    port = fake_oai.free_port()
    delay = os.environ.get("M07_FAKE_DELAY", "0.3")
    slots = os.environ.get("M07_FAKE_SLOTS", "1")
    proc = subprocess.Popen([sys.executable, str(LABS / "m06" / "tests" / "fake_oai.py"), "--port", str(port),
                             "--delay", delay, "--slots", slots],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"http://127.0.0.1:{port}"
    for _ in range(100):
        if not port_free(port):
            break
        time.sleep(0.1)
    _fake.update(proc=proc, url=url)
    return url


def stop_fake_model() -> None:
    if _fake["proc"]:
        _fake["proc"].kill()
        _fake.update(proc=None, url=None)


def child_env(state_dir: pathlib.Path) -> dict:
    env = dict(os.environ)
    env["M04_STATE_DIR"] = str(state_dir)
    env.setdefault("PYTHONUNBUFFERED", "1")
    if fake_mode():
        env["M06_FAKE_LLM"] = "1"
        env["M06_FAKE_URL"] = start_fake_model()
    return env


def ensure_index(say=print) -> None:
    """Build the M4 index (m04/state/manuals.db) and orders.db once, as M5 and M6 do."""
    if (M04_STATE / "manuals.db").exists() and (LABS / "m04" / "data" / "orders.db").exists():
        return
    say("[index] building the M4 index and orders.db (first run only)")
    code = "import sys; sys.path.insert(0, 'm06'); import testset; testset.ensure_index(); import desk_app; desk_app.close()"
    subprocess.run([sys.executable, "-c", code], cwd=LABS, env=child_env(M04_STATE), check=True)


class Fleet:
    """N desk replicas: r1 on BASE_PORT, r2 on BASE_PORT+1, ..."""

    def __init__(self, n: int, base_port: int = BASE_PORT, say=print):
        self.say = say
        self.replicas = [{"name": f"r{i + 1}", "port": base_port + i, "url": f"http://127.0.0.1:{base_port + i}",
                          "state_dir": STATE / "replicas" / f"r{i + 1}", "proc": None} for i in range(n)]

    def urls(self, n: int | None = None) -> list[str]:
        return [r["url"] for r in self.replicas[:n]]

    def get(self, name: str) -> dict:
        return next(r for r in self.replicas if r["name"] == name)

    def _launch(self, r: dict, fresh_state: bool) -> None:
        if not port_free(r["port"]):
            raise SystemExit(f"[ERROR] port {r['port']} is in use (an old replica? stop it, or use --base-port)")
        if fresh_state:
            shutil.rmtree(r["state_dir"], ignore_errors=True)
            r["state_dir"].mkdir(parents=True)
            shutil.copytree(M04_STATE / "manuals.db", r["state_dir"] / "manuals.db")
        log = (STATE / "logs" / f"{r['name']}.log").open("a")
        cmd = nat_cmd() + ["serve", "--config_file", str(CONFIG), "--host", "127.0.0.1", "--port", str(r["port"])]
        # start_new_session: the replica and its children (Milvus Lite's server) form one process group,
        # so kill() stops all of them, like a pod's containers going away together
        r["proc"] = subprocess.Popen(cmd, cwd=LABS, env=child_env(r["state_dir"]), stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        r["started"] = time.time()

    def start(self, timeout: float = 240) -> "Fleet":
        STATE.joinpath("logs").mkdir(parents=True, exist_ok=True)
        ensure_index(self.say)
        for r in self.replicas:
            self._launch(r, fresh_state=True)
        self.say(f"[fleet] starting {len(self.replicas)} replica(s): "
                 + ", ".join(f"{r['name']} :{r['port']}" for r in self.replicas))
        self.wait_healthy([r["name"] for r in self.replicas], timeout)
        return self

    def wait_healthy(self, names: list[str], timeout: float = 240) -> None:
        deadline = time.time() + timeout
        waiting = set(names)
        while waiting and time.time() < deadline:
            for name in sorted(waiting):
                r = self.get(name)
                if r["proc"].poll() is not None:
                    raise SystemExit(f"[ERROR] replica {name} exited (code {r['proc'].returncode}); "
                                     f"see m07/state/logs/{name}.log")
                if healthy(r["url"]):
                    waiting.discard(name)
                    self.say(f"[fleet] {name} healthy after {time.time() - r['started']:.1f} s")
            time.sleep(0.5)
        if waiting:
            raise SystemExit(f"[ERROR] not healthy after {timeout:.0f} s: {sorted(waiting)} (see m07/state/logs/)")

    def kill(self, name: str) -> float:
        """Kill a replica hard (SIGKILL to its process group), as a crashed pod or a lost node. Returns the time."""
        r = self.get(name)
        os.killpg(r["proc"].pid, signal.SIGKILL)
        t = time.time()
        r["proc"].wait()
        return t

    def restart(self, name: str) -> float:
        """Start a killed replica again with the state it had (as a pod restarting on the same volume)."""
        r = self.get(name)
        self._launch(r, fresh_state=False)
        return r["started"]

    def stop(self) -> None:
        for r in self.replicas:
            p = r["proc"]
            if p and p.poll() is None:
                try:
                    os.killpg(p.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        for r in self.replicas:
            p = r["proc"]
            if p:
                try:
                    p.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL)
        stop_fake_model()

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replicas", type=int, default=3)
    ap.add_argument("--base-port", type=int, default=BASE_PORT)
    a = ap.parse_args()
    fleet = Fleet(a.replicas, a.base_port)
    try:
        fleet.start()
        (STATE / "fleet.json").write_text(json.dumps([{k: r[k] for k in ("name", "port", "url")}
                                                      for r in fleet.replicas], indent=1))
        print("[fleet] up: " + " ".join(fleet.urls()) + "   (Ctrl-C stops them)")
        print("[fleet] next, in another terminal: python m07/balancer.py --replicas " + ",".join(fleet.urls()))
        while all(r["proc"].poll() is None for r in fleet.replicas):
            time.sleep(1)
        print("[fleet] a replica exited; stopping the rest")
    except KeyboardInterrupt:
        pass
    finally:
        fleet.stop()
        print("[fleet] stopped")


if __name__ == "__main__":
    main()
