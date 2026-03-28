import cv2
from PIL import Image
from pathlib import Path

video_path = "data/videos/2.mp4"
output_path = "outputs/deliverables/PPT_Assets/input_video_demo.gif"

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frames = []

# Read first 5 seconds
max_frames = int(fps * 5)
frame_count = 0

# Sample every nth frame to reduce gif size (e.g. 10 fps)
sample_rate = max(1, int(fps / 10))

while cap.isOpened() and frame_count < max_frames:
    ret, frame = cap.read()
    if not ret:
        break
        
    if frame_count % sample_rate == 0:
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize to make GIF manageable (width 480)
        h, w = frame_rgb.shape[:2]
        new_w = 480
        new_h = int(h * (new_w / w))
        frame_resized = cv2.resize(frame_rgb, (new_w, new_h))
        
        frames.append(Image.fromarray(frame_resized))
        
    frame_count += 1

cap.release()

if frames:
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        optimize=False,
        duration=100, # 100ms per frame = 10fps
        loop=0
    )
    print(f"Successfully created GIF: {output_path}")
else:
    print("Failed to read frames from video.")
