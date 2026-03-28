import argparse
from pathlib import Path

import numpy as np


def moving_average(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return x.copy()
    pad = w // 2
    x_pad = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(w, dtype=np.float64) / float(w)
    return np.convolve(x_pad, kernel, mode="valid")


def segments_from_binary(binary: np.ndarray, min_len: int):
    segs = []
    n = len(binary)
    i = 0
    while i < n:
        if binary[i] == 1:
            j = i
            while j < n and binary[j] == 1:
                j += 1
            if (j - i) >= min_len:
                segs.append((i, j - 1, j - i))
            i = j
        else:
            i += 1
    return segs


def parse_args():
    p = argparse.ArgumentParser(description="Temporal smoothing + segment detection")
    p.add_argument("--score", required=True, help="Input score .npy")
    p.add_argument("--out-score", required=True, help="Smoothed score .npy")
    p.add_argument("--out-segments", required=True, help="Segment csv output")
    p.add_argument("--window", type=int, default=9, help="Moving average window")
    p.add_argument("--quantile", type=float, default=0.95, help="Threshold quantile")
    p.add_argument("--min-len", type=int, default=5, help="Min consecutive frames")
    p.add_argument("--fps", type=float, default=2.0, help="Frame rate for time conversion")
    return p.parse_args()


def main():
    args = parse_args()
    score = np.load(args.score)
    if score.ndim != 1:
        raise ValueError("Score must be a 1D array")

    sm = moving_average(score, args.window)
    th = float(np.quantile(sm, args.quantile))
    binary = (sm >= th).astype(np.int32)
    segs = segments_from_binary(binary, args.min_len)

    out_score = Path(args.out_score)
    out_seg = Path(args.out_segments)
    out_score.parent.mkdir(parents=True, exist_ok=True)
    out_seg.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_score, sm)

    lines = ["start_idx,end_idx,length,start_sec,end_sec"]
    for s, e, l in segs:
        lines.append(f"{s},{e},{l},{s/args.fps:.3f},{e/args.fps:.3f}")
    out_seg.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("Input score:", args.score)
    print("Samples:", len(score))
    print("Threshold:", th)
    print("Anomaly ratio:", float(binary.mean()))
    print("Segment count:", len(segs))
    print("Smoothed saved:", out_score)
    print("Segments saved:", out_seg)


if __name__ == "__main__":
    main()
