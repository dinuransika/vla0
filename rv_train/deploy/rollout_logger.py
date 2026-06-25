"""
rollout_logger.py — record what the model sees and predicts at every control step.

Drop-in for so101_deploy.py. Logs, for each step of the live loop:
  - front + wrist frames (exactly as sent to the server)
  - the joint state sent
  - the FULL predicted chunk returned by the server
  - the action actually executed
into a timestamped run folder, so you can replay "why did it get stuck" offline.

USAGE in so101_deploy.py
------------------------
from rollout_logger import RolloutLogger

def main():
    robot = connect_robot()
    logger = RolloutLogger()                       # <-- creates rollouts/run_<timestamp>/
    ...
    for q in range(MAX_QUERIES):
        obs = robot.get_observation()
        cur = np.array([obs[f"{m}.pos"] for m in MOTOR_ORDER], dtype=float)
        front, wrist = obs[FRONT_CAM], obs[WRIST_CAM]
        chunk = query_server(front, wrist, cur)
        logger.log_query(q, front, wrist, cur, chunk)   # <-- inputs + full chunk

        n_exec = len(chunk) if EXEC_PER_CHUNK is None else min(EXEC_PER_CHUNK, len(chunk))
        for j, a in enumerate(chunk[:n_exec]):
            if not DRY_RUN:
                robot.send_action(action_to_dict(a))
            logger.log_exec(q, j, a, cur)               # <-- what was executed
            time.sleep(period)
    ...
    finally:
        logger.close()                                  # writes summary.json + CSV

Then inspect with:  python rollout_logger.py rollouts/run_<timestamp>
"""

import csv
import json
import os
import time

import numpy as np

MOTOR_ORDER = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


class RolloutLogger:
    def __init__(self, root="rollouts", save_images=True):
        self.run_dir = os.path.join(root, f"run_{time.strftime('%Y%m%d_%H%M%S')}")
        self.img_dir = os.path.join(self.run_dir, "frames")
        os.makedirs(self.img_dir, exist_ok=True)
        self.save_images = save_images
        self.records = []          # one row per query (state + chunk)
        self.exec_rows = []        # one row per executed action
        self._t0 = time.time()
        print(f"[logger] writing to {self.run_dir}")

    def log_query(self, q, front, wrist, state, chunk):
        front = np.asarray(front)
        wrist = np.asarray(wrist)
        state = np.asarray(state, dtype=float)
        chunk = np.asarray(chunk, dtype=float)

        # raw arrays for exact offline replay (npz keeps dtype/shape)
        np.savez(
            os.path.join(self.img_dir, f"q{q:04d}.npz"),
            front=front, wrist=wrist, state=state, chunk=chunk,
        )

        if self.save_images:
            try:
                from PIL import Image
                Image.fromarray(self._to_uint8(front)).save(
                    os.path.join(self.img_dir, f"q{q:04d}_front.png"))
                Image.fromarray(self._to_uint8(wrist)).save(
                    os.path.join(self.img_dir, f"q{q:04d}_wrist.png"))
            except Exception as e:
                if q == 0:
                    print(f"[logger] PNG previews skipped: {e}")

        self.records.append({
            "query": q,
            "t": round(time.time() - self._t0, 3),
            "state": state.tolist(),
            "pred_step0": chunk[0].tolist(),
            "chunk": chunk.tolist(),
            "front_shape": list(front.shape),
            "wrist_shape": list(wrist.shape),
            # how far the first predicted action asks each joint to move from current
            "delta_step0": (chunk[0] - state).tolist(),
        })

    def log_exec(self, q, j, action, state_before):
        action = np.asarray(action, dtype=float)
        self.exec_rows.append({
            "query": q, "substep": j,
            **{f"act_{m}": float(action[k]) for k, m in enumerate(MOTOR_ORDER)},
            **{f"cur_{m}": float(state_before[k]) for k, m in enumerate(MOTOR_ORDER)},
        })

    def close(self):
        with open(os.path.join(self.run_dir, "summary.json"), "w") as f:
            json.dump(self.records, f, indent=2)

        if self.exec_rows:
            keys = list(self.exec_rows[0].keys())
            with open(os.path.join(self.run_dir, "executed.csv"), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                w.writerows(self.exec_rows)

        # quick "is it stuck?" diagnostic: how much does the commanded state actually change?
        if len(self.records) >= 2:
            states = np.array([r["state"] for r in self.records])
            step_motion = np.abs(np.diff(states, axis=0)).sum(axis=1)  # total joint motion between queries
            print(f"[logger] {len(self.records)} queries. "
                  f"Per-query total joint motion: mean={step_motion.mean():.2f} "
                  f"min={step_motion.min():.2f} max={step_motion.max():.2f}")
            stuck = np.where(step_motion < 0.5)[0]
            if len(stuck):
                print(f"[logger] {len(stuck)} queries with <0.5deg total motion "
                      f"(possibly stuck) at indices: {stuck.tolist()[:20]}")
        print(f"[logger] saved summary.json + executed.csv in {self.run_dir}")

    @staticmethod
    def _to_uint8(img):
        img = np.asarray(img)
        if img.ndim == 3 and img.shape[0] in (1, 3):       # CHW -> HWC
            img = np.transpose(img, (1, 2, 0))
        if img.dtype != np.uint8:
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
        return img


def inspect(run_dir):
    """Print a per-query table: current wrist_roll/gripper, predicted step-0, and the
    delta the model is asking for. Makes 'stuck in the middle' visible at a glance."""
    summ = json.load(open(os.path.join(run_dir, "summary.json")))
    print(f"{'q':>4} {'t':>7} | "
          + " ".join(f"{m[:6]:>7}" for m in MOTOR_ORDER)
          + "  ||  pred_step0 (same order)")
    for r in summ:
        cur = r["state"]
        pred = r["pred_step0"]
        delta = r["delta_step0"]
        cur_s = " ".join(f"{v:7.1f}" for v in cur)
        pred_s = " ".join(f"{v:7.1f}" for v in pred)
        max_d = max(abs(d) for d in delta)
        flag = "  <-- tiny move" if max_d < 0.5 else ""
        print(f"{r['query']:>4} {r['t']:>7.2f} | {cur_s}  ||  {pred_s}{flag}")

    # show where motion stalls
    states = np.array([r["state"] for r in summ])
    if len(states) >= 2:
        motion = np.abs(np.diff(states, axis=0)).sum(axis=1)
        print("\nPer-query total joint motion (deg):")
        print(" ", np.round(motion, 2).tolist())
        print("\nIf motion drops to ~0 and stays there, the model is predicting "
              "actions ~equal to current state (it 'thinks' it's done or is stuck "
              "in a state it can't escape open-loop). Check the frames at that query "
              "— is the lego still visible / in the expected place?")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("usage: python rollout_logger.py rollouts/run_<timestamp>")
        sys.exit(1)
    inspect(sys.argv[1])