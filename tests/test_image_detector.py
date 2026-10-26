import io
from PIL import Image
from src.detectors.image_detector import detect_image

class Fake:
    def __call__(self, img): return [{"label": "artificial", "score": 0.7}, {"label": "human", "score": 0.3}]

def test_mocked_image():
    b = io.BytesIO(); Image.new("RGB", (8, 8)).save(b, "PNG")
    assert detect_image("x.png", b.getvalue(), pipe=Fake()).top_label == "artificial"

def test_corrupt_image():
    try: detect_image("x.png", b"junk", pipe=Fake())
    except ValueError: return
    assert False
