from src.detectors.text_detector import detect_text
from src.validators import ValidationError

class Fake:  # mocked model - NOT a real evaluation
    def __call__(self, t, **kw): return [{"label": "Real", "score": 0.9}]

def test_mocked_prediction():
    r = detect_text("hello world " * 10, pipe=Fake())
    assert r.label == "Real" and r.score == 0.9

def test_empty_rejected():
    try: detect_text("  ", pipe=Fake())
    except ValidationError: return
    assert False
