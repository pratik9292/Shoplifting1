import cv2
import os
import numpy as np

# --- CONFIG ---
VIDEO_PATH = "dataset"  # Folder containing videos
OUTPUT_RGB = "data/frames/rgb"
OUTPUT_FLOW = "data/frames/flow"
IMG_SIZE = (128, 128)
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")  # Add more if needed

# --- CREATE OUTPUT FOLDERS ---
os.makedirs(OUTPUT_RGB, exist_ok=True)
os.makedirs(OUTPUT_FLOW, exist_ok=True)


# --- FUNCTION TO PROCESS A SINGLE VIDEO ---
def process_video(video_path, video_name):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"❌ Error: Unable to open {video_path}")
        return

    ret, prev_frame = cap.read()
    if not ret:
        print(f"❌ Error: Failed to read first frame of {video_path}")
        cap.release()
        return

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize RGB frame
        frame_resized = cv2.resize(frame, IMG_SIZE)
        rgb_filename = os.path.join(OUTPUT_RGB, f"{video_name}_frame_{frame_idx:05d}.png")
        cv2.imwrite(rgb_filename, frame_resized)  # Save frame as image

        # Calculate Optical Flow
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # Normalize flow for visualization
        flow_magnitude, flow_angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv = np.zeros_like(frame)
        hsv[..., 1] = 255  # Full saturation
        hsv[..., 0] = flow_angle * 90 / np.pi  # Hue based on direction
        hsv[..., 2] = cv2.normalize(flow_magnitude, None, 0, 255, cv2.NORM_MINMAX)  # Brightness based on magnitude
        flow_visual = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Resize and save Optical Flow frame
        flow_resized = cv2.resize(flow_visual, IMG_SIZE)
        flow_filename = os.path.join(OUTPUT_FLOW, f"{video_name}_flow_{frame_idx:05d}.png")
        cv2.imwrite(flow_filename, flow_resized)  # Save flow image

        prev_gray = gray.copy()
        frame_idx += 1

    cap.release()
    print(f"✅ Processed {video_path}: {frame_idx} frames saved")


# --- ITERATE THROUGH SUBFOLDERS & FIND VIDEOS ---
for root, _, files in os.walk(VIDEO_PATH):
    for file in files:
        if file.lower().endswith(VIDEO_EXTENSIONS):
            video_path = os.path.join(root, file)
            video_name = os.path.splitext(file)[0]  # Remove file extension

            # Process the video
            process_video(video_path, video_name)

print("\n🎉 All videos processed successfully!")
