"""Live training viewer: watch a run's newest checkpoint play, beside its live training numbers.

    .venv\\Scripts\\python.exe scripts\\run_low.py --cpus 8,9 viewer\\live_viewer.py --run runs/nj3-b
    then open http://127.0.0.1:8765

What it shows
  - a live top-down match: the run's NEWEST saved checkpoint against an opponent (default: itself),
    simulated headless in RocketSim at real-time pace. It reloads the policy when training saves a
    newer checkpoint (every --reload-min minutes at most).
  - training numbers, refreshed every few seconds: steps and steps/s, reward terms, trend rows
    (BallChaser / v3 / Necto / Nexto), the latest jump audit, and the run's safety markers.

Project rules it follows (CLAUDE.md)
  - read-only on every run file; it never signals, pauses or configures a trainer
  - headless and CPU-only: torch is pinned to 1 thread and the launch line pins it to CPUs 8-9
  - preflight.ensure() before the first simulated game, like every eval
  - bound to 127.0.0.1 only
  - a no-jump policy choosing a jump row stops the simulation and says so on screen
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import queue
import random
import re
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
HERE = Path(__file__).resolve().parent
V3 = REPO / "runs" / "v3" / "archive" / "v3_final_2534033664"
TPS = 120
TICK_SKIP = 8      # decisions every 8 ticks, as in training
FRAME_EVERY = 2    # physics every tick; a frame to the page every 2 ticks (60 Hz)


# --------------------------------------------------------------------------------------------- files
def tail_bytes(path: Path, n: int = 400_000) -> str:
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - n))
            return f.read().decode("utf-8", "replace")
    except OSError:
        return ""


def head_text(path: Path, n: int = 20_000) -> str:
    try:
        with open(path, "rb") as f:
            return f.read(n).decode("utf-8", "replace")
    except OSError:
        return ""


def downsample(rows: list, n: int = 400) -> list:
    if len(rows) <= n:
        return rows
    step = len(rows) / n
    return [rows[int(i * step)] for i in range(n)] + [rows[-1]]


def trainer_log_for(run: Path) -> Path:
    name = run.name
    cand = REPO / ("runs_" + name.replace("-", "") + "_train.log")
    if cand.is_file():
        return cand
    return REPO / f"runs_{name}_train.log"


def complete_checkpoints(run: Path) -> list[Path]:
    out = []
    for d in (run / "checkpoints").glob("*"):
        if d.is_dir() and d.name.isdigit() and (d / "POLICY.lt").is_file() and (d / "SHARED_HEAD.lt").is_file() \
                and (d / "RUNNING_STATS.json").is_file():
            out.append(d)
    return sorted(out, key=lambda p: int(p.name))


# --------------------------------------------------------------------------------------------- arena
def arena_mesh() -> dict:
    """RocketSim's soccar collision meshes (the files rlgym-rocket-league ships), in uu.

    These are the arena geometry the simulator actually collides with: corners, goal ends, the
    floor-to-wall ramps and the ceiling ramps. RocketSim adds the flat floor, flat side walls and ceiling
    as planes; the viewer draws those itself. .cmf: int32 tris, int32 verts, tris*3 int32, verts*3 float32,
    in Bullet units (1 = 50 uu)."""
    import rlgym.rocket_league.sim as simmod
    folder = Path(simmod.__file__).parent / "collision_meshes" / "soccar"
    verts, tris, base = [], [], 0
    for f in sorted(folder.glob("*.cmf"), key=lambda p: int(p.stem.split("_")[1])):
        b = f.read_bytes()
        nt, nv = np.frombuffer(b[:8], dtype=np.int32)
        t = np.frombuffer(b[8:8 + 12 * nt], dtype=np.int32).reshape(-1, 3)
        v = np.frombuffer(b[8 + 12 * nt:8 + 12 * nt + 12 * nv], dtype=np.float32).reshape(-1, 3) * 50.0
        verts.append(np.round(v, 1))
        tris.append(t + base)
        base += nv
    v = np.concatenate(verts)
    t = np.concatenate(tris)
    return {"verts": v.ravel().tolist(), "tris": t.ravel().tolist(), "files": len(verts)}


# --------------------------------------------------------------------------------------------- stats
class Stats:
    def __init__(self, run: Path):
        self.run = run
        self.log = trainer_log_for(run)
        self.cache = {}
        self.lock = threading.Lock()

    def status(self) -> dict:
        txt = tail_bytes(self.log, 250_000)
        steps = re.findall(r"Total Timesteps: ([\d,]+)", txt)
        sps = re.findall(r"Overall Steps/Second: ([\d,\.]+)", txt)
        audits = re.findall(r"> jump audit: (.*)", txt)
        stage = re.search(r"STAGE (\d) \(([^)]*)\): gaeGamma ([\d.]+), LR ([\de.\-]+)", head_text(self.log, 20_000))
        nonzero = len(re.findall(r"learner acts=[1-9]", txt))
        opp = {"external": txt.count("opponent[external]"), "none": txt.count("opponent[none]")}
        cfg = {}
        try:
            cfg = json.loads((self.run / "effective_config.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
        try:
            control = json.loads((self.run / "opponents.json").read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            control = {}
        archives = sorted((int(d.name) for d in (self.run / "archive").glob("*") if d.name.isdigit()), reverse=True)
        mtime = self.log.stat().st_mtime if self.log.is_file() else 0
        markers = {m: (self.run / m).exists() for m in ("WATCHDOG_HOLD", "HARD_STOP_JUMP.txt", "CONFIG_CHECK_FAILED.txt")}
        beat = ""
        try:
            beat = (self.run / "watchdog_heartbeat.txt").read_text(encoding="utf-8-sig").strip()
        except OSError:
            pass
        stop_at = tail_bytes(self.run / "stop_at.log", 4000).strip().splitlines()[-1:] or [""]
        return {
            "run": self.run.name,
            "steps": int(steps[-1].replace(",", "")) if steps else None,
            "steps_per_sec_reported": float(sps[-1].replace(",", "")) if sps else None,
            "log_age_s": round(time.time() - mtime, 1) if mtime else None,
            "stage": {"stage": int(stage.group(1)), "source": stage.group(2), "gamma": stage.group(3), "lr": stage.group(4)} if stage else None,
            "config": {k: cfg.get(k) for k in ("stage", "stage_source", "gae_gamma", "policy_lr", "varied_starts",
                                               "external_opponent", "external_opponent_chance", "resume_steps")},
            "opponent_control": control,
            "last_jump_audit": audits[-1] if audits else None,
            "nonzero_jump_audits_in_tail": nonzero,
            "opponent_lines_in_tail": opp,
            "archives": archives[:12],
            "markers": markers,
            "watchdog_heartbeat": beat,
            "stop_at": stop_at[0],
        }

    def series(self) -> dict:
        out = {}
        # steps over time and realised steps/s, from the GPU logger
        g = []
        try:
            with open(self.run / "gpu_log.csv", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    if r.get("total_timesteps"):
                        g.append((r["utc"], float(r["total_timesteps"]), float(r.get("steps_per_sec") or 0),
                                  r.get("gaming"), r.get("utilization.gpu")))
        except OSError:
            pass
        realised = []
        for a, b in zip(g, g[1:]):
            try:
                dt = (np.datetime64(b[0][:26]) - np.datetime64(a[0][:26])) / np.timedelta64(1, "s")
            except Exception:  # noqa: BLE001
                continue
            if 0 < dt <= 120 and b[1] >= a[1]:
                realised.append((b[0], (b[1] - a[1]) / dt))
        out["steps"] = downsample([(r[0], r[1]) for r in g])
        out["realised_sps"] = downsample(realised)
        out["gaming"] = g[-1][3] if g else None
        out["gpu_util"] = g[-1][4] if g else None
        # reward terms per iteration (unweighted), from the newest metrics CSV
        rewards = {}
        files = sorted(glob.glob(str(self.run / "metrics" / "*.csv")), key=os.path.getmtime)
        if files:
            try:
                with open(files[-1], encoding="utf-8") as f:
                    rd = csv.reader(f)
                    head = next(rd)
                    rows = list(rd)
                ts = head.index("Total Timesteps") if "Total Timesteps" in head else None
                start = out["steps"][0][1] if out["steps"] else 0
                rows = [r for r in rows if ts is not None and len(r) > ts and r[ts] and float(r[ts]) >= start]
                rows = downsample(rows, 300)
                for i, c in enumerate(head):
                    if c.startswith("Rewards/") or c in ("Player/Boost", "Player/Ball Touch Ratio", "Average Step Reward"):
                        pts = [(float(r[ts]), float(r[i])) for r in rows if len(r) > i and r[i] not in ("", "nan")]
                        if pts:
                            rewards[c.split("/")[-1]] = pts
            except (OSError, ValueError, StopIteration):
                pass
        out["metrics"] = rewards
        # trend rows
        trend = []
        try:
            with open(self.run / "trend.csv", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    if not r.get("timesteps"):
                        continue
                    row = {"steps": float(r["timesteps"])}
                    for k in ("bc_win_rate", "v1f_win_rate", "v3_win_rate", "nc_win_rate", "nx_win_rate",
                              "nx_goal_diff_per_game", "m1_score_rate", "open_touch_rate"):
                        try:
                            row[k] = float(r[k]) if r.get(k) not in (None, "") else None
                        except ValueError:
                            row[k] = None
                    trend.append(row)
        except OSError:
            pass
        out["trend"] = trend
        return out


# --------------------------------------------------------------------------------------------- sim
class LiveMatch(threading.Thread):
    def __init__(self, run: Path, opponent: str, reload_min: float, speed: float, match_seconds: float):
        super().__init__(daemon=True)
        self.run_dir, self.opponent, self.reload_s, self.speed = run, opponent, reload_min * 60, speed
        self.match_seconds = match_seconds
        self.subs: list[queue.Queue] = []
        self.lock = threading.Lock()
        self.info = {"state": "starting", "error": None, "ckpt": None, "opponent": opponent}
        self.last_match = None  # replayed to a page that connects mid-match (it carries the pad layout)

    def subscribe(self) -> queue.Queue:
        q = queue.Queue(maxsize=240)
        with self.lock:
            if self.last_match is not None:
                q.put_nowait(self.last_match)
            self.subs.append(q)
        return q

    def unsubscribe(self, q):
        with self.lock:
            if q in self.subs:
                self.subs.remove(q)

    def publish(self, msg: dict):
        data = json.dumps(msg, separators=(",", ":"))
        with self.lock:
            if msg.get("type") == "match":
                self.last_match = data
            for q in list(self.subs):
                try:
                    q.put_nowait(data)
                except queue.Full:
                    try:
                        q.get_nowait()
                        q.put_nowait(data)
                    except (queue.Empty, queue.Full):
                        pass

    def load(self):
        import policies as P
        cks = complete_checkpoints(self.run_dir)
        if not cks:
            raise RuntimeError(f"no complete checkpoint in {self.run_dir / 'checkpoints'}")
        ck = cks[-1]
        # a checkpoint still being written: take the one before it
        if time.time() - max(p.stat().st_mtime for p in ck.iterdir()) < 5 and len(cks) > 1:
            ck = cks[-2]
        learner = P.GigaLearnPolicy(ck, name=f"{self.run_dir.name}@{ck.name}")
        if not learner.forbid_jump:
            print(f"note: {ck} is not marked forbid_jump", flush=True)
        if self.opponent == "self":
            opp = P.GigaLearnPolicy(ck, name=f"{self.run_dir.name}@{ck.name} (mirror)")
        elif self.opponent == "v3":
            opp = P.GigaLearnPolicy(V3, name="v3-final")
        elif self.opponent == "ballchaser":
            opp = P.BallChaserPolicy()
        elif self.opponent == "nexto":
            opp = P.NextoPolicy(name="Nexto")
        else:
            raise SystemExit(f"unknown opponent {self.opponent}")
        self.info.update(ckpt=int(ck.name), opponent_name=opp.name, learner_name=learner.name,
                         forbid_jump=bool(learner.forbid_jump))
        return learner, opp, time.time()

    def run_forever(self):
        """Physics one tick at a time (120 Hz), decisions every 8 ticks as in training, frames at 60 Hz.

        The policies act exactly as they do in training and evaluation: one decision per 8 ticks, held
        in between. Stepping the simulator per tick only gives the viewer the positions between
        decisions, so motion is smooth rather than 15 Hz. Pacing is against an absolute clock, so a slow
        step does not push every later frame back."""
        import torch
        torch.set_num_threads(1)
        import eval as E
        from rlgym.rocket_league.common_values import BOOST_LOCATIONS
        pads = [[float(v) for v in p] for p in BOOST_LOCATIONS]
        env = E.build_env(1, 45.0, "kickoff")
        learner, opp, loaded_at = self.load()
        tick_dt = 1.0 / TPS
        match = 0
        while True:
            match += 1
            random.seed(time.time_ns() % (2 ** 31))
            np.random.seed(time.time_ns() % (2 ** 31))
            learner_blue = match % 2 == 1
            env.reset()
            agents = list(env.agents)
            learner.reset(agents)
            opp.reset(agents)
            score = [0, 0]
            ticks = 0
            ep_tick = 0
            held = {}
            reset_next = True
            self.info["state"] = "playing"
            self.publish({"type": "match", "match": match, "learner_team": 0 if learner_blue else 1, "pads": pads,
                          "speed": self.speed, "frame_hz": TPS // FRAME_EVERY,
                          **{k: self.info.get(k) for k in ("ckpt", "learner_name", "opponent_name", "forbid_jump")}})
            deadline = time.perf_counter()
            while ticks < self.match_seconds * TPS:
                s = env.state
                if ep_tick % TICK_SKIP == 0:
                    for a in agents:
                        mine = (s.cars[a].team_num == 0) == learner_blue
                        held[a] = (learner if mine else opp).act(s, [a])[a]
                acts = {}
                for a in agents:
                    act = held[a]
                    if np.ndim(act) == 0:
                        acts[a] = np.array([act], dtype=np.int64)          # a row of the 90-action table
                    else:
                        arr = np.asarray(act, dtype=np.float32)
                        acts[a] = arr[ep_tick % TICK_SKIP % len(arr)] if arr.ndim == 2 else arr   # per-tick rows (Nexto's kickoff)
                _, _, term, trunc = env.step(acts)
                ticks += 1
                ep_tick += 1
                s = env.state
                done = any(term.values()) or any(trunc.values())
                goal = None
                if done and s.goal_scored:
                    score[s.scoring_team] += 1
                    goal = int(s.scoring_team)
                if ticks % FRAME_EVERY == 0 or done:
                    cars = []
                    for a in agents:
                        c = s.cars[a]
                        f = c.physics.forward
                        cars.append({"team": int(c.team_num), "learner": (c.team_num == 0) == learner_blue,
                                     "x": round(float(c.physics.position[0]), 1), "y": round(float(c.physics.position[1]), 1),
                                     "z": round(float(c.physics.position[2]), 1), "yaw": round(math.atan2(float(f[1]), float(f[0])), 3),
                                     "boost": round(float(c.boost_amount), 1), "demoed": bool(c.is_demoed),
                                     "ground": bool(c.on_ground), "jumped": bool(c.has_jumped or c.has_flipped),
                                     "speed": round(float(np.linalg.norm(c.physics.linear_velocity))),
                                     "f": [round(float(v), 3) for v in c.physics.forward],
                                     "u": [round(float(v), 3) for v in c.physics.up],
                                     "boosting": bool(c.boost_active_time > 0)})
                    b = s.ball
                    self.publish({"type": "frame", "t": ticks / TPS, "left": round(self.match_seconds - ticks / TPS, 1),
                                  "ball": {"x": round(float(b.position[0]), 1), "y": round(float(b.position[1]), 1),
                                           "z": round(float(b.position[2]), 1),
                                           "w": [round(float(v), 3) for v in b.angular_velocity]},
                                  "cars": cars, "pads": [bool(v == 0) for v in np.asarray(s.boost_pad_timers)],
                                  "score": score, "goal": goal, "reset": reset_next})
                    reset_next = False
                if done:
                    env.reset()
                    agents = list(env.agents)
                    learner.reset(agents)
                    opp.reset(agents)
                    held = {}
                    ep_tick = 0
                    reset_next = True
                    if goal is not None:
                        time.sleep(1.2 / self.speed)
                        deadline = time.perf_counter()
                if self.speed > 0:
                    deadline += tick_dt / self.speed
                    wait = deadline - time.perf_counter()
                    if wait > 0.002:
                        time.sleep(wait)
                    elif wait < -0.5:
                        deadline = time.perf_counter()   # far behind: resynchronise rather than race to catch up
            self.publish({"type": "final", "match": match, "score": score})
            if time.time() - loaded_at > self.reload_s and complete_checkpoints(self.run_dir) \
                    and int(complete_checkpoints(self.run_dir)[-1].name) != self.info.get("ckpt"):
                self.info["state"] = "reloading"
                learner, opp, loaded_at = self.load()

    def run(self):
        try:
            from scripts.preflight import ensure
            self.info["state"] = "preflight"
            ensure()
            self.run_forever()
        except BaseException as e:  # noqa: BLE001 - show it on the page, never die silently
            msg = f"{type(e).__name__}: {e}"
            if "jump row" in msg:
                msg = "STOPPED: a no-jump policy chose a jump row - that is a hard-stop condition. " + msg
            self.info.update(state="stopped", error=msg, trace=traceback.format_exc()[-2000:])
            self.publish({"type": "error", "error": msg})
            print(traceback.format_exc(), flush=True)


# --------------------------------------------------------------------------------------------- http
def make_handler(stats: Stats, sim: LiveMatch):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def send_json(self, obj):
            body = json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                body = (HERE / "index.html").read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/api/status":
                self.send_json({**stats.status(), "sim": {k: v for k, v in sim.info.items() if k != "trace"}})
            elif self.path == "/api/arena":
                if "arena" not in stats.cache:
                    stats.cache["arena"] = arena_mesh()
                self.send_json(stats.cache["arena"])
            elif self.path == "/api/assets":
                d = HERE / "assets"
                self.send_json({n: (d / n).is_file() for n in ("ball.png", "ball.jpg", "octane.glb")})
            elif self.path.startswith("/assets/"):
                name = self.path[len("/assets/"):]
                f = (HERE / "assets" / name).resolve()
                ctype = {".png": "image/png", ".jpg": "image/jpeg", ".glb": "model/gltf-binary"}.get(f.suffix.lower())
                if f.parent != (HERE / "assets").resolve() or not f.is_file() or not ctype:
                    self.send_response(404); self.end_headers(); return
                body = f.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/api/series":
                self.send_json(stats.series())
            elif self.path == "/api/stream":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                q = sim.subscribe()
                try:
                    while True:
                        try:
                            data = q.get(timeout=15)
                            self.wfile.write(f"data: {data}\n\n".encode())
                        except queue.Empty:
                            self.wfile.write(b": keepalive\n\n")
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass
                finally:
                    sim.unsubscribe(q)
            else:
                self.send_response(404)
                self.end_headers()
    return H


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/nj3-b", help="run directory to watch")
    ap.add_argument("--opponent", default="self", choices=("self", "v3", "ballchaser", "nexto"))
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--reload-min", type=float, default=5.0, help="reload a newer checkpoint at most this often")
    ap.add_argument("--speed", type=float, default=1.0, help="1.0 = real time")
    ap.add_argument("--match-seconds", type=float, default=300.0)
    args = ap.parse_args()
    run = (REPO / args.run).resolve()
    if not run.is_dir():
        raise SystemExit(f"no run directory {run}")
    stats = Stats(run)
    sim = LiveMatch(run, args.opponent, args.reload_min, args.speed, args.match_seconds)
    sim.start()
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(stats, sim))
    print(f"live viewer for {run.name}: http://127.0.0.1:{args.port}  (opponent: {args.opponent}; Ctrl+C to stop)", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
