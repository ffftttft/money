import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate score/prediction with optional GT.")
    p.add_argument("--score", required=True, help="Input score .npy")
    p.add_argument("--segments", required=True, help="Input segment csv")
    p.add_argument("--out", required=True, help="Output txt")
    p.add_argument("--gt", default="", help="Optional GT binary .npy")
    p.add_argument("--threshold-quantile", type=float, default=0.95)
    return p.parse_args()


def parse_segments(path: Path):
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    if len(lines) <= 1:
        return []
    segs = []
    for ln in lines[1:]:
        if not ln.strip():
            continue
        s, e, l, *_ = ln.split(",")
        segs.append((int(s), int(e), int(l)))
    return segs


def main():
    args = parse_args()
    score = np.load(args.score)
    th = float(np.quantile(score, args.threshold_quantile))
    pred = (score >= th).astype(np.int32)
    segs = parse_segments(Path(args.segments))

    lines = []
    lines.append(f"score_file={args.score}")
    lines.append(f"samples={len(score)}")
    lines.append(f"mean={float(score.mean()):.6f}")
    lines.append(f"std={float(score.std()):.6f}")
    lines.append(f"p95={float(np.percentile(score,95)):.6f}")
    lines.append(f"p99={float(np.percentile(score,99)):.6f}")
    lines.append(f"threshold_q={args.threshold_quantile}")
    lines.append(f"threshold={th:.6f}")
    lines.append(f"anomaly_ratio={float(pred.mean()):.6f}")
    lines.append(f"segment_count={len(segs)}")
    lines.append(f"max_segment_len={max([s[2] for s in segs], default=0)}")

    if args.gt:
        gt = np.load(args.gt).astype(np.int32)
        if len(gt) != len(score):
            raise ValueError("GT length mismatch with score length")
        auc = roc_auc_score(gt, score)
        p, r, f1, _ = precision_recall_fscore_support(gt, pred, average="binary", zero_division=0)
        lines.append(f"auc={auc:.6f}")
        lines.append(f"precision={p:.6f}")
        lines.append(f"recall={r:.6f}")
        lines.append(f"f1={f1:.6f}")
    else:
        lines.append("gt=none")
        lines.append("auc=NA")
        lines.append("precision=NA")
        lines.append("recall=NA")
        lines.append("f1=NA")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Saved:", out)


if __name__ == "__main__":
    main()
