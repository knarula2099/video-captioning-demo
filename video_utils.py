import cv2
import numpy as np
import os
from typing import List, Tuple

def extract_frames(video_path: str, output_dir: str = "frames", frame_interval: int = 30) -> List[str]:
    """
    Extract frames from a video file, but only save frames whose
    grayscale‐histogram differs “enough” from the last saved frame.
    This replaces the previous “save every Nth frame” logic with a
    histogram‐difference filter.

    Args:
        video_path (str): Path to the input video file
        output_dir (str): Directory to save extracted frames
        frame_interval (int): Check every Nth frame (e.g. 30 → ~1 second at 30 FPS)

    Returns:
        List[str]: List of paths to the saved frames
    """
    # ----------------------------------------------------------------------------
    # You can tune HIST_THRESHOLD to be more or less sensitive.
    # Lower → more frames get saved. Higher → fewer frames.
    HIST_THRESHOLD = 0.8
    # ----------------------------------------------------------------------------

    # 1) Create (or clear) the output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        for f in os.listdir(output_dir):
            if f.startswith("frame_") and f.endswith(".jpg"):
                os.remove(os.path.join(output_dir, f))

    cap = cv2.VideoCapture(video_path)
    frame_paths: List[str] = []

    last_saved_hist: np.ndarray = None
    frame_count = 0
    saved_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Only process every `frame_interval` frames
        if frame_count % frame_interval == 0:
            # Compute a normalized grayscale histogram (256 bins)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()

            # If this is the first sampled frame, save it unconditionally
            if last_saved_hist is None:
                save_flag = True
            else:
                # Compute χ² distance between last_saved_hist and current hist
                chi_sq = cv2.compareHist(last_saved_hist.astype(np.float32),
                                         hist.astype(np.float32),
                                         method=cv2.HISTCMP_CHISQR)
                save_flag = (chi_sq > HIST_THRESHOLD)

            if save_flag:
                filename = f"frame_{frame_count:06d}.jpg"
                out_path = os.path.join(output_dir, filename)
                cv2.imwrite(out_path, frame)
                frame_paths.append(out_path)

                last_saved_hist = hist.copy()
                saved_count += 1

        frame_count += 1

    cap.release()
    print(f"Scanned {frame_count} frames, saved {saved_count} “interesting” frames.")
    return frame_paths

def get_video_info(video_path: str) -> Tuple[int, int, float]:
    """
    Get basic information about the video.

    Args:
        video_path (str): Path to the video file

    Returns:
        Tuple[int, int, float]: (width, height, fps)
    """
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return width, height, fps
