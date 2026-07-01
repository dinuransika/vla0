"""
Offline replay evaluation for the VLA-0 inference server.

Streams recorded frames from a LeRobot dataset to the running VLA-0 service and
compares the model's predicted actions against the ground-truth actions you
recorded. NO ROBOT INVOLVED — this is the sanity check to run BEFORE so101_deploy.py.

What "good" looks like:
  - predicted action[0] tracks the recorded action at each frame (small L1/L2 error)
  - error stays bounded across the episode (doesn't blow up after a few frames)
  - gripper dimension flips at roughly the right moments

If predictions are way off, the bug is in the pipeline (encoding, camera order,
state convention, action dim) — fix it here, not on the arm.

Usage:
    # 1. start the server on the cluster (or ssh -L 10000:localhost:10000 ...)
    # 2. run this against your local dataset copy:
    python replay_eval.py --repo-id dinura/so101_pickup_lego --episode 0
"""

import argparse
import base64
import io

import numpy as np
import requests

# ----------------------------- config (match training) -----------------------------
SERVER_URL  = "http://localhost:10000"
TASK        = "Pick up the square Lego block and place it in the container"

# Camera observation keys, in the ORDER the server expects (base64_rgb[0], base64_rgb[1]).
# Must match training: front -> 3p1, wrist -> 3p2. Check meta/info.json if unsure.
FRONT_CAM   = "front"
WRIST_CAM   = "wrist"

# Action / state vector order — MUST match the dataset 'action' feature order.
MOTOR_ORDER = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
# ------------------------------------------------------------------------------------



def frame_to_b64(arr):
    """Camera frame (HWC uint8 RGB) -> base64 .npy buffer.
    Matches the server's rgb_from_base64 (which uses np.load), NOT JPEG."""
    arr = np.asarray(arr)
    if arr.dtype != np.uint8:
        # LeRobot frames may come back as float [0,1] (CHW) or uint8 (HWC). Normalize to HWC uint8.
        if arr.max() <= 1.0:
            arr = (arr * 255).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
    buf = io.BytesIO()
    np.save(buf, arr)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def to_hwc_uint8(img):
    """LeRobot image tensors are often CHW float. Convert to HWC uint8 RGB."""
    img = np.asarray(img)
    if img.ndim == 3 and img.shape[0] in (1, 3):   # CHW -> HWC
        img = np.transpose(img, (1, 2, 0))
    if img.dtype != np.uint8:
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
    return img


def query_server(front_arr, wrist_arr, state):
    payload = {
        "base64_rgb": [frame_to_b64(front_arr), frame_to_b64(wrist_arr)],  # front first, then wrist
        "state": [float(x) for x in state],
        "instr": TASK,
    }
    r = requests.post(f"{SERVER_URL}/predict_base64", json=payload, timeout=6000)
    r.raise_for_status()
    return np.asarray(r.json(), dtype=float)   # bare list -> (horizon, action_dim)


def load_episode(repo_id, episode, root=None):
    """Load a single episode from a LeRobot dataset. Returns a list of frame dicts."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset(repo_id, root=root)

    # find the frame index range for this episode.
    # LeRobot >=0.4 dropped ds.episode_data_index; boundaries now live in meta.episodes.
    ep_meta = ds.meta.episodes[episode]
    from_idx = int(ep_meta["dataset_from_index"])
    to_idx   = int(ep_meta["dataset_to_index"])
    print(f"Episode {episode}: frames [{from_idx}, {to_idx}) = {to_idx - from_idx} steps")
    return ds, range(from_idx, to_idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", default="dinura/so101_pickup_lego_square")
    ap.add_argument("--episode", type=int, default=2)
    ap.add_argument("--root", default="/Users/dinura.dissanayake/lerobot_data/so101_pickup_lego_square", help="local dataset root if not in HF cache")
    ap.add_argument("--max-frames", type=int, default=None, help="cap frames for a quick test")
    ap.add_argument("--stride", type=int, default=1, help="evaluate every Nth frame")
    args = ap.parse_args()

    # health check first
    try:
        requests.get(f"{SERVER_URL}/health", timeout=5).raise_for_status()
        print(f"Server healthy at {SERVER_URL}")
    except requests.exceptions.RequestException as e:
        print(f"Cannot reach server at {SERVER_URL}: {e}")
        return

    ds, frame_range = load_episode(args.repo_id, args.episode, args.root)

    # figure out the image / state / action keys actually present in the dataset
    sample = ds[frame_range[0]]
    print("Available keys:", list(sample.keys()))

    # common LeRobot key patterns; adjust if your dataset names differ
    front_key = next((k for k in sample if FRONT_CAM in k and "image" in k.lower() or k.endswith(FRONT_CAM)), None)
    wrist_key = next((k for k in sample if WRIST_CAM in k and "image" in k.lower() or k.endswith(WRIST_CAM)), None)
    if front_key is None or wrist_key is None:
        # fall back to observation.images.* convention
        img_keys = [k for k in sample if "image" in k.lower()]
        print("Image-like keys found:", img_keys)
        front_key = front_key or next((k for k in img_keys if FRONT_CAM in k), img_keys[0] if img_keys else None)
        wrist_key = wrist_key or next((k for k in img_keys if WRIST_CAM in k), img_keys[1] if len(img_keys) > 1 else None)
    print(f"Using front_key={front_key!r}, wrist_key={wrist_key!r}")

    state_key  = next((k for k in sample if "state" in k.lower()), "observation.state")
    action_key = next((k for k in sample if k == "action" or k.endswith(".action") or "action" in k.lower()), "action")
    print(f"Using state_key={state_key!r}, action_key={action_key!r}\n")

    l1_errors, l2_errors = [], []
    n = 0
    frames = list(frame_range)[::args.stride]
    if args.max_frames:
        frames = frames[:args.max_frames]

    for i in frames:
        frame = ds[i]
        front = to_hwc_uint8(frame[front_key])
        wrist = to_hwc_uint8(frame[wrist_key])
        state = np.asarray(frame[state_key], dtype=float)
        gt_action = np.asarray(frame[action_key], dtype=float)

        chunk = query_server(front, wrist, state)
        pred = chunk[0]   # first action of the predicted chunk = what you'd execute now

        l1 = np.abs(pred - gt_action)
        l2 = np.linalg.norm(pred - gt_action)
        l1_errors.append(l1)
        l2_errors.append(l2)
        n += 1

        if n <= 10 or n % 20 == 0:
            print(f"frame {i:4d} | L2={l2:6.2f} | pred={np.round(pred,1)} | gt={np.round(gt_action,1)}")

    l1_errors = np.array(l1_errors)   # (n, action_dim)
    l2_errors = np.array(l2_errors)   # (n,)

    print("\n" + "=" * 60)
    print(f"Evaluated {n} frames from episode {args.episode}")
    print(f"Mean L2 error : {l2_errors.mean():.3f}")
    print(f"Median L2     : {np.median(l2_errors):.3f}")
    print(f"Max L2        : {l2_errors.max():.3f}")
    print(f"Per-joint mean L1 ({MOTOR_ORDER}):")
    print(f"  {np.round(l1_errors.mean(axis=0), 3)}")
    print("=" * 60)
    print("\nInterpretation:")
    print("  - Per-joint L1 small relative to the joint's range (stats min/max) = good.")
    print("  - One joint with huge error = likely wrong order in MOTOR_ORDER or a sign flip.")
    print("  - ALL joints huge = encoding / camera-order / state-convention mismatch.")
    print("  - Error growing across frames = fine here (open-loop replay); real control is closed-loop.")


if __name__ == "__main__":
    main()
