import os, cv2, numpy as np
from src.utils.video_processing import *
from src.detectors.video_detector import fake_score

def make_video(path, secs=4, fps=10):
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 64))
    for _ in range(secs * fps): w.write(np.zeros((64, 64, 3), np.uint8))
    w.release()

def test_sampling_interval():
    p = save_temp(b"", ".mp4"); make_video(p)
    try:
        fr = sample_frames(p, interval_s=1.0)
        assert len(fr) == 4 and abs(fr[1][0] - 1.0) < 0.2
        assert len(sample_frames(p, 1.0, max_frames=2)) == 2
    finally: os.remove(p)

def test_corrupt_video():
    p = save_temp(b"garbage", ".mp4")
    try: sample_frames(p)
    except VideoError: return
    finally: os.remove(p)
    assert False

def test_aggregate():
    a = aggregate([0.1, 0.9, 0.6])
    assert a["max"] == 0.9 and abs(a["frac_flagged"] - 2/3) < 1e-9

def test_fake_label_pick():
    assert fake_score({"Real": 0.2, "Fake": 0.8}) == 0.8
