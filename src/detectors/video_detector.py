from dataclasses import dataclass, field
from PIL import Image
from src import config as C
from src.validators import validate_video_file
from src.utils.video_processing import save_temp, sample_frames, crop_largest_face, aggregate, VideoError
from src.detectors.image_detector import classify_image
import os

FAKE_HINTS = ("fake", "artificial", "ai", "deepfake", "synthetic")

def fake_score(scores: dict) -> float:
    """Pick the score of the label that looks like 'fake'. Raises if the model's labels are unrecognised."""
    for label, s in scores.items():
        if any(h == label.lower() or h in label.lower().split() for h in FAKE_HINTS):
            return s
    raise VideoError(f"Cannot tell which label means 'fake' from {list(scores)}. Adjust FAKE_HINTS.")

@dataclass
class VideoResult:
    summary: dict
    frames: list = field(default_factory=list)   # (timestamp, thumbnail, score)
    faces_found: int = 0
    method: str = "mean/median/max/fraction-flagged over sampled face crops"

def detect_video(name: str, data: bytes, interval_s=C.DEFAULT_FRAME_INTERVAL_S, pipe=None) -> VideoResult:
    validate_video_file(name, len(data))
    path = save_temp(data, os.path.splitext(name)[1].lower())
    try:
        sampled = sample_frames(path, interval_s)
    finally:
        os.remove(path)
    rows = []
    for ts, rgb in sampled:
        face = crop_largest_face(rgb)
        if face is None:
            continue
        img = Image.fromarray(face)
        rows.append((ts, img, fake_score(classify_image(img, C.DEEPFAKE_MODEL_ID, pipe))))
    if not rows:
        raise VideoError("No faces detected in sampled frames. This detector only analyses faces; "
                         "it cannot assess faceless AI-generated video.")
    return VideoResult(aggregate([r[2] for r in rows]), rows, len(rows))
