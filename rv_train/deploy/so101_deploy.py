"""
SO-101 control loop driven by the VLA-0 inference server.

Runs on the Mac (robot attached). Each cycle: grab front+wrist frames + current joint
state -> POST to the VLA-0 service -> receive a 6-dim action chunk -> execute it on the
follower. No leader arm.

  Cluster:  ROBOVERSE_DEPLOY_CHECKPOINT=.../checkpoint-XXXX/model_last.pth python rv_train/deploy/service.py
  If remote: ssh -L 10000:localhost:10000 dinura@<cluster>   (then SERVER_URL stays localhost)
  Mac:      python so101_deploy.py

============================  SAFETY — READ FIRST  ============================
- This commands a real arm from a model. A bad prediction WILL move the arm hard.
- DRY_RUN=True by default: it prints actions and sends NOTHING. Verify the printed
  values look sane (6 numbers, in the same range as your recorded actions) before
  setting DRY_RUN=False.
- BETTER YET: run replay_eval.py first to confirm predictions track recorded actions
  offline, with no robot involved. Only deploy if those numbers look good.
- Keep a hand on the power switch for the first real rollouts.
- MAX_RELATIVE_TARGET caps how far any joint can move in one step (the built-in jerk
  limiter). Start small and raise it only once behavior looks safe.
- The model outputs actions in the SAME space it was trained in. The deploy robot MUST
  use the same calibration and use_degrees as during recording, or the action space
  won't match and motion will be wrong.
==============================================================================
"""

import base64
import io
import os
import time

import numpy as np
import requests
from rollout_logger import RolloutLogger

# ----------------------------- config -----------------------------
DRY_RUN          = False                      # <-- prints actions, sends nothing. Flip to False to move the arm.
SERVER_URL       = "http://localhost:10000"
TASK             = "Pick up the Lego block and place it in the container"

FOLLOWER_PORT    = "/dev/tty.usbmodem5A680130541"
FOLLOWER_ID      = "my_awesome_follower_arm"  # calibration id used during recording
FRONT_INDEX      = 0
WRIST_INDEX      = 1

CONTROL_HZ       = 30                          # execution rate of actions within a chunk (match dataset fps)
                  # safety: max per-step joint move (deg). None = unlimited (risky).
MAX_QUERIES      = 200                         # stop after this many server calls (safety cap)
EXEC_PER_CHUNK   = None                        # actions of each chunk to execute before re-querying.
                                               # None = whole chunk. Lower = more reactive, more calls.

# Action vector order — MUST match your dataset's `action` feature order (meta/info.json "names").
MOTOR_ORDER = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
MAX_RELATIVE_TARGET = {m: 10.0 for m in MOTOR_ORDER}

# Camera observation keys returned by get_observation().
FRONT_CAM = "front"
WRIST_CAM = "wrist"

CAPTURE_DIR = "captured_obs"                   # where `capture` mode saves obs_*.npz (replayed by replay_saved.py)
# -------------------------------------------------------------------


# ===== Server protocol — matches rv_train/deploy/data_models.py + service.py =====
# Server expects: {"base64_rgb": [npy_b64, npy_b64], "state": [6 floats], "instr": str}
# Server returns: a bare JSON list of shape (horizon, action_dim).

def frame_to_b64(arr):
    """Camera frame (HWC uint8 RGB) -> base64 .npy buffer.
    MUST match the server's rgb_from_base64, which uses np.load (NOT JPEG)."""
    arr = np.asarray(arr)
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
    buf = io.BytesIO()
    np.save(buf, arr)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def build_payload(front_b64, wrist_b64, state, task):
    return {
        "base64_rgb": [front_b64, wrist_b64],     # front first, then wrist — match training (3p1, 3p2)
        "state": [float(x) for x in state],        # 6 current joint positions, MOTOR_ORDER order
        "instr": task,
    }



def parse_actions(resp_json):
    return np.asarray(resp_json, dtype=float)      # bare list -> (horizon, action_dim)
# =================================================================================


def query_server(front_arr, wrist_arr, state):
    payload = build_payload(frame_to_b64(front_arr), frame_to_b64(wrist_arr), state, TASK)
    r = requests.post(f"{SERVER_URL}/predict_base64", json=payload, timeout=60)
    r.raise_for_status()
    return parse_actions(r.json())


def connect_robot():
    """Instantiate + connect the follower (lerobot 0.5.2: classes live in so_follower)."""
    from lerobot.robots.so_follower import SO101Follower as Follower, SO101FollowerConfig as Config
    try:
        from lerobot.cameras.opencv import OpenCVCameraConfig
    except ImportError:
        from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

    # Both cameras at the resolution you RECORDED with (1280x720). Server resizes to 224
    # internally; matching the recording aspect ratio keeps the crop consistent.
    cameras = {
        FRONT_CAM: OpenCVCameraConfig(index_or_path=FRONT_INDEX, width=1280, height=720, fps=30),
        WRIST_CAM: OpenCVCameraConfig(index_or_path=WRIST_INDEX, width=1280, height=720, fps=30),
    }
    # NOTE: SOFollowerConfig in 0.5.2 has no `id` field. If SO101FollowerConfig adds one,
    # add `id=FOLLOWER_ID` below. Otherwise calibration is resolved from the on-disk
    # calibration file by robot type/port. Verify with:
    #   python -c "from lerobot.robots.so_follower import SO101FollowerConfig; import inspect; print(inspect.signature(SO101FollowerConfig.__init__))"
    cfg = Config(
        port=FOLLOWER_PORT,
        cameras=cameras,
        use_degrees=True,
        max_relative_target=MAX_RELATIVE_TARGET,
        id=FOLLOWER_ID,  # <-- uncomment if SO101FollowerConfig has an `id` field
    )
    robot = Follower(cfg)
    robot.connect()
    return robot


def action_to_dict(vec):
    """6-dim model output -> {motor.pos: value} dict expected by send_action."""
    assert len(vec) == len(MOTOR_ORDER), f"Got {len(vec)}-dim action, expected {len(MOTOR_ORDER)}"
    return {f"{m}.pos": float(v) for m, v in zip(MOTOR_ORDER, vec)}


def capture_observations(n=1, out_dir=CAPTURE_DIR, interval=1.0, warmup=30):
    os.makedirs(out_dir, exist_ok=True)
    robot = connect_robot()
    try:
        # macOS AVFoundation + LeRobot's threaded reader hand back black frames for the
        # first several reads while the capture session spins up. main() does this; capture didn't.
        print(f"warming up cameras ({warmup} frames)...")
        for _ in range(warmup):
            robot.get_observation()
            time.sleep(1 / 30)

        for i in range(n):
            obs = robot.get_observation()
            missing = [k for k in (FRONT_CAM, WRIST_CAM) if k not in obs]
            if missing:
                raise KeyError(f"Camera keys {missing} not in observation. Available: {list(obs.keys())}")
            front = np.asarray(obs[FRONT_CAM])
            wrist = np.asarray(obs[WRIST_CAM])
            state = np.array([obs[f"{m}.pos"] for m in MOTOR_ORDER], dtype=float)

            # guard: don't silently save dead frames
            for name, fr in (("front", front), ("wrist", wrist)):
                if fr.size == 0 or int(fr.max()) == 0:
                    raise RuntimeError(
                        f"{name} frame is all black after warmup (max={fr.max()}). "
                        f"Increase warmup, or check the camera index/resolution."
                    )

            path = os.path.join(out_dir, f"obs_{i:03d}.npz")
            np.savez(path, front=front, wrist=wrist, state=state, task=TASK)
            print(f"saved {path}  front={front.shape} wrist={wrist.shape} "
                  f"front.mean={front.mean():.1f} state={np.round(state, 2)}")

            try:  # optional human-viewable previews; safe to skip if Pillow is missing
                from PIL import Image
                Image.fromarray(front).save(os.path.join(out_dir, f"obs_{i:03d}_front.png"))
                Image.fromarray(wrist).save(os.path.join(out_dir, f"obs_{i:03d}_wrist.png"))
            except Exception as e:
                print(f"  (skipped PNG preview: {e})")

            if i < n - 1:
                time.sleep(interval)
    finally:
        robot.disconnect()   # disable_torque_on_disconnect=True releases the arm
        print("Disconnected (torque released).")


def main():
    robot = connect_robot()
    obs = robot.get_observation()
    

    period = 1.0 / CONTROL_HZ
    print(f"Connected. DRY_RUN={DRY_RUN}. Task: {TASK!r}")

    print("warming up cameras...")
    for _ in range(30):
        robot.get_observation()
        time.sleep(1/30)
    logger = RolloutLogger()

    try:
        # --- one query up front, as a safety preview ---
        obs = robot.get_observation()
        # sanity-check the observation keys before using them
        missing = [k for k in (FRONT_CAM, WRIST_CAM) if k not in obs]
        if missing:
            raise KeyError(f"Camera keys {missing} not in observation. Available: {list(obs.keys())}")
        cur = np.array([obs[f"{m}.pos"] for m in MOTOR_ORDER], dtype=float)

        chunk = query_server(obs[FRONT_CAM], obs[WRIST_CAM], cur)

        print("Current state:", np.round(cur, 2))
        print(f"Predicted action {len(chunk)}-step chunk:\n", np.round(chunk, 2))
        assert chunk.shape[1] == len(MOTOR_ORDER), \
            f"Action dim {chunk.shape[1]} != {len(MOTOR_ORDER)} — check original_action_dim and MOTOR_ORDER."

        if not DRY_RUN:
            input("\n>>> Review the numbers above. Hand on the power switch. "
                  "Press ENTER to ALLOW MOTION (Ctrl-C to abort)...")

        # --- control loop ---
        for q in range(MAX_QUERIES):
            obs = robot.get_observation()
            cur = np.array([obs[f"{m}.pos"] for m in MOTOR_ORDER], dtype=float)
            chunk = query_server(obs[FRONT_CAM], obs[WRIST_CAM], cur)
            logger.log_query(q, obs[FRONT_CAM], obs[WRIST_CAM], cur, chunk)
            n_exec = len(chunk) if EXEC_PER_CHUNK is None else min(EXEC_PER_CHUNK, len(chunk))

            for a in chunk[:n_exec]:
                j = np.round(a, 2)
                if DRY_RUN:
                    print("would send:", np.round(a, 2))
                else:
                    robot.send_action(action_to_dict(a))
                    logger.log_exec(q, j, a, cur)     # j = enumerate index of the inner loop

                time.sleep(period)

            print(f"[query {q+1}/{MAX_QUERIES}] executed {n_exec} actions")

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        robot.disconnect()   # disable_torque_on_disconnect=True releases the arm
        print("Disconnected (torque released).")
        logger.close()



if __name__ == "__main__":
    import sys
    # `python so101_deploy.py capture [N]` -> save N obs (default 1) for offline replay.
    # `python so101_deploy.py`             -> normal control loop.
    if len(sys.argv) > 1 and sys.argv[1] == "capture":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 1
        capture_observations(n)
    else:
        main()
