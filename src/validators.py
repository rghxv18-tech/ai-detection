import io, os
from PIL import Image, UnidentifiedImageError
from src import config as C

class ValidationError(ValueError):
    """Raised with a beginner-friendly message."""

def validate_text(text: str) -> str:
    text = (text or "").strip()
    if not text:
        raise ValidationError("Please enter some text.")
    if len(text) < C.MIN_TEXT_CHARS:
        raise ValidationError(f"Text is too short ({len(text)} chars). Use at least {C.MIN_TEXT_CHARS}; short text is unreliable.")
    if len(text) > C.MAX_TEXT_CHARS:
        raise ValidationError(f"Text is too long ({len(text)} chars). Maximum is {C.MAX_TEXT_CHARS}.")
    return text

def _check_file(name: str, size: int, exts: set, max_mb: int):
    ext = os.path.splitext(name or "")[1].lower()
    if ext not in exts:
        raise ValidationError(f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(exts))}.")
    if size <= 0:
        raise ValidationError("The file is empty.")
    if size > max_mb * 1024 * 1024:
        raise ValidationError(f"File is {size/1e6:.1f} MB; the limit is {max_mb} MB.")

def validate_image_bytes(name: str, data: bytes) -> Image.Image:
    _check_file(name, len(data), C.IMAGE_EXTS, C.MAX_IMAGE_MB)
    try:
        img = Image.open(io.BytesIO(data)); img.verify()
        img = Image.open(io.BytesIO(data)); img.load()
    except (UnidentifiedImageError, OSError, SyntaxError):
        raise ValidationError("This file could not be read as an image (it may be corrupt or mislabeled).")
    if img.width * img.height > C.MAX_IMAGE_PIXELS:
        raise ValidationError(f"Image is too large ({img.width}x{img.height}).")
    return img.convert("RGB")

def validate_video_file(name: str, size: int):
    _check_file(name, size, C.VIDEO_EXTS, C.MAX_VIDEO_MB)
