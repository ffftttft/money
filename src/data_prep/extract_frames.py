import argparse
from pathlib import Path

import cv2


def parse_args():
    p = argparse.ArgumentParser(description="Extract frames from a video.")
    p.add_argument("--video", required=True, help="Path to input video file")
    p.add_argument("--out-train", required=True, help="Output dir for train frames")
    p.add_argument("--out-test", required=True, help="Output dir for test frames")
    p.add_argument("--fps", type=float, default=2.0, help="Target FPS to extract")
    p.add_argument(
        "--train-minutes",
        type=float,
        default=12.0,
        help="Minutes from start used for training split",
    )
    p.add_argument(
        "--size",
        default="224,224",
        help="Resize frames to W,H (e.g. 224,224). Use 0,0 to keep size.",
    )
    p.add_argument("--prefix", default="frame", help="Filename prefix")
    return p.parse_args()


def main():
    args = parse_args()
    video_path = Path(args.video)
    out_train = Path(args.out_train)
    out_test = Path(args.out_test)
    out_train.mkdir(parents=True, exist_ok=True)
    out_test.mkdir(parents=True, exist_ok=True)

    size_w, size_h = [int(x) for x in args.size.split(",")]
    resize = size_w > 0 and size_h > 0

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS)
    if native_fps <= 0:
        native_fps = 25.0

    train_seconds = args.train_minutes * 60.0
    target_interval = 1.0 / max(args.fps, 0.001)
    next_capture_t = 0.0

    train_count = 0
    test_count = 0
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        t = frame_idx / native_fps
        if t + 1e-6 >= next_capture_t:
            if resize:
                frame = cv2.resize(frame, (size_w, size_h), interpolation=cv2.INTER_AREA)

            if t < train_seconds:
                out_dir = out_train
                idx = train_count
                train_count += 1
            else:
                out_dir = out_test
                idx = test_count
                test_count += 1

            filename = f"{args.prefix}_{idx:06d}.jpg"
            cv2.imwrite(str(out_dir / filename), frame)
            next_capture_t += target_interval

        frame_idx += 1

    cap.release()

    print("Video:", video_path)
    print("Native FPS:", native_fps)
    print("Target FPS:", args.fps)
    print("Train minutes:", args.train_minutes)
    print("Train frames:", train_count)
    print("Test frames:", test_count)
    print("Train dir:", out_train)
    print("Test dir:", out_test)


if __name__ == "__main__":
    main()
