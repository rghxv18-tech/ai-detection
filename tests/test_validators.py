import io
from PIL import Image
from src.validators import *
from src import config as C

def raises(fn, *a):
    try: fn(*a)
    except ValidationError: return True
    return False

def png(size=(10, 10)):
    b = io.BytesIO(); Image.new("RGB", size).save(b, "PNG"); return b.getvalue()

def test_text():
    assert raises(validate_text, "")
    assert raises(validate_text, "short")
    assert raises(validate_text, "x" * (C.MAX_TEXT_CHARS + 1))
    assert validate_text("word " * 20)

def test_image():
    assert validate_image_bytes("a.png", png()).size == (10, 10)
    assert raises(validate_image_bytes, "a.gif", png())
    assert raises(validate_image_bytes, "a.png", b"not an image")
    assert raises(validate_image_bytes, "a.png", b"")

def test_video_name():
    assert raises(validate_video_file, "a.avi", 100)
    validate_video_file("a.mp4", 100)
