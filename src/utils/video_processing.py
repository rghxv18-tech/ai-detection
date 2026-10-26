import os, tempfile
from statistics import mean, median
import cv2
from src import config as C

class VideoError(RuntimeError):
    pass

def save_temp(data: bytes, suffix: str) -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    return path

def sample_frames(path: str, interval_s: float = C.DEFAULT_FRAME_INTERVAL_S, max_frames: int = C.MAX_FRAMES):
    """Return list of (timestamp_s, RGB ndarray). Raises VideoError on unreadable/too-long video."""
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            raise VideoError("Video could not be opened (corrupt file or unsupported codec).")
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        total = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        if fps <= 0 or total <= 0:
            raise VideoError("Could not read video duration/frame rate.")
        duration = total / fps
        if duration > C.MAX_VIDEO_SECONDS:
            raise VideoError(f"Video is {duration:.0f}s; the limit is {C.MAX_VIDEO_SECONDS}s.")
        step = max(1, int(round(interval_s * fps)))
        frames = []
        for idx in range(0, int(total), step):
            if len(frames) >= max_frames:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, bgr = cap.read()
            if ok:
                frames.append((idx / fps, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
        if not frames:
            raise VideoError("No frames could be decoded.")
        return frames
    finally:
        cap.release()

_cascade = None
def crop_largest_face(rgb, margin: float = 0.25):
    """Haar-cascade face crop (OpenCV built-in; basic, misses angled/small faces). Returns RGB crop or None."""
    global _cascade
    if _cascade is None:
        _cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    faces = _cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    m = int(margin * max(w, h))
    y0, y1, x0, x1 = max(0, y - m), min(rgb.shape[0], y + h + m), max(0, x - m), min(rgb.shape[1], x + w + m)
    return rgb[y0:y1, x0:x1]

def aggregate(scores: list) -> dict:
    """Aggregate per-frame 'fake' scores. Reports mean, median, max and fraction of frames above 0.5."""
    if not scores:
        raise VideoError("No scores to aggregate.")
    return {"mean": mean(scores), "median": median(scores), "max": max(scores),
            "frac_flagged": sum(s > 0.5 for s in scores) / len(scores), "n": len(scores)}
