"""Step 1: the M7 desk fleet with telemetry: three replicas with metrics, traces and a request log.

    python m08/fleet_obs.py                          # r1..r3 (v7) on 8101..8103, metrics on 9101..9103
    python m08/fleet_obs.py --canary r3=v8           # r3 runs v8 (DESK_VERSION=v8, OLLAMA_MODEL=llama3.2:1b)
    python m08/fleet_obs.py --no-phoenix             # traces to files only (no Phoenix running)

It is M7's Fleet (m07/serve_fleet.py) with m08/configs/desk_obs.yml instead of the plain desk, and
per replica:
  M08_METRICS_PORT   9101, 9102, ...: the replica's /metrics for Prometheus (m08/observability/)
  M08_TRACE_FILE     m08/state/traces/rN.jsonl: the file exporter's copy of every trace
  PHOENIX_ENDPOINT   http://localhost:6006/v1/traces (--phoenix URL to change it)
  DESK_VERSION       v7 unless --canary names the replica; the versions are in VERSIONS below

The balancer (balancer_obs.py) reads m08/state/fleet.json to know which version each replica runs.
A replica that exits (or is killed, as probe.py --kill does) is started again after 5 seconds with
the state it had, as Kubernetes restarts a crashed pod; m08/state/fleet.json gets its new PID.
Offline self-test only: M08_FAKE_LLM=1 uses M7's fake mode (one scripted model server).
"""
import argparse
import json
import os
import pathlib
import shutil
import signal
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
LABS = HERE.parent
sys.path.insert(0, str(LABS / "m07"))
if os.environ.get("M08_FAKE_LLM") == "1":
    os.environ["M07_FAKE_LLM"] = "1"
import serve_fleet  # noqa: E402  (m07)

STATE = HERE / "state"
CONFIGS = HERE / "configs"
METRICS_BASE = 9101
PHOENIX = "http://localhost:6006/v1/traces"
# What a version is: the settings a replica of that version runs with. v8 is the release candidate:
# the smaller model, which would cut cost and latency if its answers hold up (lessons 8.3 and 8.4).
VERSIONS = {"v7": {"OLLAMA_MODEL": "llama3.2:3b"}, "v8": {"OLLAMA_MODEL": "llama3.2:1b"}}
say = serve_fleet.say


class ObsFleet(serve_fleet.Fleet):
    def __init__(self, n: int, versions: dict | None = None, phoenix: str | None = PHOENIX,
                 base_port: int = serve_fleet.BASE_PORT, say=say):
        super().__init__(n, base_port, say)
        self.phoenix = phoenix
        for i, r in enumerate(self.replicas):
            r["version"] = (versions or {}).get(r["name"], "v7")
            r["metrics_port"] = METRICS_BASE + i
            r["metrics"] = f"http://127.0.0.1:{r['metrics_port']}/metrics"
            r["trace_file"] = STATE / "traces" / f"{r['name']}.jsonl"
            r["state_dir"] = STATE / "replicas" / r["name"]

    def _launch(self, r: dict, fresh_state: bool) -> None:
        for port in (r["port"], r["metrics_port"]):
            if not serve_fleet.port_free(port):
                raise SystemExit(f"[ERROR] port {port} is in use (an old replica? stop it first)")
        if fresh_state:
            shutil.rmtree(r["state_dir"], ignore_errors=True)
            r["state_dir"].mkdir(parents=True)
            shutil.copytree(serve_fleet.M04_STATE / "manuals.db", r["state_dir"] / "manuals.db")
        (STATE / "logs").mkdir(parents=True, exist_ok=True)
        r["trace_file"].parent.mkdir(parents=True, exist_ok=True)
        env = serve_fleet.child_env(r["state_dir"])
        env.update(VERSIONS[r["version"]], DESK_VERSION=r["version"], M08_METRICS_PORT=str(r["metrics_port"]),
                   M08_TRACE_FILE=str(r["trace_file"]), PHOENIX_ENDPOINT=self.phoenix or "")
        config = CONFIGS / ("desk_obs.yml" if self.phoenix else "desk_obs_file.yml")
        cmd = serve_fleet.nat_cmd() + ["serve", "--config_file", str(config), "--host", "127.0.0.1",
                                       "--port", str(r["port"])]
        log = (STATE / "logs" / f"{r['name']}.log").open("a")
        r["proc"] = subprocess.Popen(cmd, cwd=LABS, env=env, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
        r["started"] = time.time()

    def start(self, timeout: float = 240) -> "ObsFleet":
        serve_fleet.ensure_index(self.say)
        for r in self.replicas:
            self._launch(r, fresh_state=True)
        self.say(f"[fleet] starting {len(self.replicas)} replica(s): "
                 + ", ".join(f"{r['name']} :{r['port']} {r['version']} (metrics :{r['metrics_port']})"
                             for r in self.replicas))
        self.wait_healthy([r["name"] for r in self.replicas], timeout)
        self.write_json()
        return self

    def write_json(self) -> None:
        STATE.mkdir(parents=True, exist_ok=True)
        (STATE / "fleet.json").write_text(json.dumps(
            [{"name": r["name"], "url": r["url"], "version": r["version"], "metrics": r["metrics"],
              "model": VERSIONS[r["version"]]["OLLAMA_MODEL"], "pid": r["proc"].pid if r["proc"] else None}
             for r in self.replicas], indent=1))

    def supervise(self, delay: float = 5.0) -> None:
        """Restart a replica that exits, after `delay` seconds, as a Deployment restarts a crashed pod."""
        while True:
            for r in self.replicas:
                code = r["proc"].poll()
                if code is not None:
                    self.say(f"[fleet] {r['name']} exited (code {code}); restarting it in {delay:g} s")
                    time.sleep(delay)
                    self.restart(r["name"])
                    self.wait_healthy([r["name"]])
                    self.write_json()
            time.sleep(1)


def parse_canary(text: str) -> dict:
    out = {}
    for part in filter(None, (text or "").split(",")):
        name, _, version = part.partition("=")
        if version not in VERSIONS:
            raise SystemExit(f"[ERROR] unknown version {version!r} (known: {', '.join(VERSIONS)})")
        out[name.strip()] = version
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replicas", type=int, default=3)
    ap.add_argument("--canary", default="", help="replica=version pairs, e.g. r3=v8")
    ap.add_argument("--phoenix", default=PHOENIX, help="Phoenix OTLP/HTTP traces endpoint")
    ap.add_argument("--no-phoenix", action="store_true", help="traces to files only")
    a = ap.parse_args()
    fleet = ObsFleet(a.replicas, parse_canary(a.canary), None if a.no_phoenix else a.phoenix)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt))
    try:
        fleet.start()
        say("[fleet] up: " + "  ".join(f"{r['name']} {r['url']} ({r['version']})" for r in fleet.replicas)
            + "   (Ctrl-C stops them)")
        say("[fleet] next, in another terminal: python m08/balancer_obs.py")
        fleet.supervise()
    except KeyboardInterrupt:
        pass
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        fleet.stop()
        say("[fleet] stopped")


if __name__ == "__main__":
    main()
