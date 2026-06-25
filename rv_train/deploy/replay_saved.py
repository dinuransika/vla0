"""
Replay SAVED observations against the VLA-0 inference server.

Loads the .npz observations captured by `python so101_deploy.py capture` (front
frame + wrist frame + joint state) and POSTs them to the running server in a loop.
NO ROBOT, NO CAMERAS needed — use this to debug / iterate on the server with a
fixed, repeatable input.

Encoding is reused from so101_deploy (frame_to_b64 / build_payload / parse_actions),
so the bytes the server sees here are identical to a live deploy.

Usage:
    # 1. once, with robot + cameras plugged in:
    python so101_deploy.py capture 3
    # 2. then, repeatedly, with no hardware (server can be remote via ssh -L):
    python replay_saved.py
    python replay_saved.py --loops 50 --obs-dir captured_obs --delay 0.5
"""

import argparse
import glob
import os
import time

import numpy as np
import requests

from so101_deploy import frame_to_b64, build_payload, parse_actions, SERVER_URL, TASK, CAPTURE_DIR


def load_obs(obs_dir):
    """Load every obs_*.npz from obs_dir as (path, front, wrist, state, task) tuples."""
    paths = sorted(glob.glob(os.path.join(obs_dir, "obs_*.npz")))
    if not paths:
        raise FileNotFoundError(
            f"No obs_*.npz in {obs_dir!r}. Capture some first:\n"
            f"    python so101_deploy.py capture 3"
        )
    obs = []
    for p in paths:
        d = np.load(p, allow_pickle=True)
        task = str(d["task"]) if "task" in d else TASK
        obs.append((p, d["front"], d["wrist"], d["state"], task))
    return obs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs-dir", default=CAPTURE_DIR)
    ap.add_argument("--server", default=SERVER_URL)
    ap.add_argument("--loops", type=int, default=0,
                    help="passes over the saved obs; 0 = forever (Ctrl-C to stop)")
    ap.add_argument("--delay", type=float, default=0.0, help="seconds to sleep between queries")
    args = ap.parse_args()

    # health check first
    try:
        requests.get(f"{args.server}/health", timeout=5).raise_for_status()
        print(f"Server healthy at {args.server}")
    except requests.exceptions.RequestException as e:
        print(f"Cannot reach server at {args.server}: {e}")
        return

    obs = load_obs(args.obs_dir)
    print(f"Loaded {len(obs)} saved observation(s) from {args.obs_dir!r}\n")

    loop = 0
    try:
        while args.loops == 0 or loop < args.loops:
            for path, front, wrist, state, task in obs:
                payload = build_payload(frame_to_b64(front), frame_to_b64(wrist), state, task)
                t0 = time.time()
                r = requests.post(f"{args.server}/predict_base64", json=payload, timeout=6000)
                r.raise_for_status()
                chunk = parse_actions(r.json())
                dt = time.time() - t0
                print(f"[loop {loop} | {os.path.basename(path)}] {dt:6.2f}s  "
                      f"chunk{chunk.shape}  action[0]={np.round(chunk[0], 2)}")
                if args.delay:
                    time.sleep(args.delay)
            loop += 1
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
