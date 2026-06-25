"""
Standalone camera probe — bypasses LeRobot entirely.

Opens each camera index, asks for 1280x720, reports what the camera ACTUALLY
gives back, warms up, then saves a frame + prints content stats so we can tell
a real frame from the dark gradient.

    python cam_probe.py            # probe indices 0 and 1
    python cam_probe.py 0 1 2 3    # probe specific indices
"""
import sys
import time
import numpy as np
import cv2


def stats(fr):
    fr = fr.astype(float)
    rows = fr.mean(axis=(1, 2))
    within_row = fr.reshape(fr.shape[0], -1).std(axis=1).mean()
    patch = fr[fr.shape[0]//2-10:fr.shape[0]//2+10,
               fr.shape[1]//2-10:fr.shape[1]//2+10]
    return (f"mean={fr.mean():.1f} min={fr.min():.0f} max={fr.max():.0f} "
            f"row[top/mid/bot]={rows[:10].mean():.1f}/{rows[len(rows)//2-5:len(rows)//2+5].mean():.1f}/{rows[-10:].mean():.1f} "
            f"within_row_std={within_row:.2f} patch_std={patch.std():.2f}")


def probe(index, want_w=1280, want_h=720, warmup=60):
    print(f"\n=== index {index} ===")
    # On macOS, cv2.CAP_AVFOUNDATION is the backend LeRobot uses too.
    cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        print("  could NOT open")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  want_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, want_h)
    cap.set(cv2.CAP_PROP_FPS,          30)

    got_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    got_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  requested {want_w}x{want_h}, camera reports {got_w}x{got_h}"
          + ("  <-- MISMATCH" if (got_w, got_h) != (want_w, want_h) else "  (ok)"))

    # warm up so auto-exposure converges and the pipeline flushes
    last = None
    means = []
    for i in range(warmup):
        ok, fr = cap.read()
        if ok and fr is not None:
            last = fr
            if i % 10 == 0:
                means.append(round(float(fr.mean()), 1))
        time.sleep(1/30)
    print(f"  mean every 10th warmup frame: {means}  (should drift/vary, not be constant)")

    if last is None:
        print("  no frame read")
        cap.release()
        return

    # BGR from OpenCV; convert to match what you'd save
    rgb = cv2.cvtColor(last, cv2.COLOR_BGR2RGB)
    print("  final frame:", stats(rgb))
    out = f"probe_index{index}.png"
    cv2.imwrite(out, last)
    print(f"  saved {out}")
    cap.release()


if __name__ == "__main__":
    idxs = [int(a) for a in sys.argv[1:]] or [0, 1]
    for ix in idxs:
        probe(ix)
    print("\nNow open the probe_index*.png files and compare to Photo Booth.")